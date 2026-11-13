"""Focused two-trial EfficientNet-B0 tuning; validation only, seed 42.
Training requires CUDA. No test evaluation or threshold search is implemented.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.sharedctypes import RawArray
import argparse
import sys
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
import time
import uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageOps
import sklearn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path("/content/drive/MyDrive/cocoa-diseases") if Path("/content/drive/MyDrive").is_dir() else Path(r"H:\My Drive\cocoa-diseases")
BASELINE_RUN_NAME = "run_20261101T115128Z_6c0d08"
CLASS_MAP = {"other_plants": 0, "cocoa": 1}
NAMES = ["other_plants", "cocoa"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MODEL_NAME = "efficientnet_b0"
FORMAT_VERSION = "model1-efficientnet-focused-tuning-v1"
TRIALS = {"trial_1_lr_0.0003": 0.0003, "trial_2_lr_0.0001": 0.0001}
IMAGE_SIZE, BATCH_SIZE, MAX_EPOCHS = 224, 64, 1000
EARLY_STOP_PATIENCE, MIN_DELTA, SEED, THRESHOLD = 30, 0.0, 42, 0.5
WEIGHT_DECAY, NUM_WORKERS, CHECKPOINT_EVERY = 0.0001, 2, 5
AUGMENTATION = {"enabled": True, "horizontal_flip": 0.5, "vertical_flip": 0.5,
                "rotation_degrees": 20, "zoom": 0.1, "translate": 0.1, "contrast": 0.2, "fill": 0}
STOPPING_NOTE = ("Baseline: maximum 100 epochs / patience 20. New trials: maximum 1000 / patience 30. "
                 "Differences cannot be attributed to learning rate alone.")
NEAR_TIE = {"cocoa_recall": 1/894, "balanced_accuracy": 0.001, "macro_f1": 0.001}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def save_json(path, data):
    atomic_bytes(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))

def save_csv(path, frame):
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8-sig"))

def save_checkpoint(path, data):
    # Serialize on local disk before replacing the destination checkpoint.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="model1_checkpoint_") as folder:
            local = Path(folder) / "checkpoint.pth"
            torch.save(data, local)
            shutil.copyfile(local, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def check_files(cfg, frame, splits):
    """Check file availability and recorded sizes without decoding images."""
    rows = frame.loc[frame["split"].isin(splits)]
    for split in splits:
        require(set(rows.loc[rows["split"].eq(split), "class_name"]) == set(CLASS_MAP), f"Missing class in {split}")
    root = Path(cfg["dataset_root"])
    errors, checked = [], 0
    for parent, part in rows.groupby(rows["image_path"].map(lambda p: str(PurePosixPath(p).parent)), sort=False):
        directory = root / parent
        try:
            with os.scandir(directory) as entries:
                sizes = {e.name: e.stat(follow_symlinks=False).st_size for e in entries if e.is_file(follow_symlinks=False)}
        except OSError as exc:
            errors.append(f"{directory}: {exc}")
            continue
        for row in part.to_dict("records"):
            filename = PurePosixPath(row["image_path"]).name
            expected = row.get("output_filesize") or row.get("filesize")
            if filename not in sizes:
                errors.append(f"ID {row['image_id']}: missing {root / row['image_path']}")
            elif sizes[filename] == 0 or (expected and sizes[filename] != int(expected)):
                errors.append(f"ID {row['image_id']}: file size differs: {root / row['image_path']}")
            checked += 1
    require(not errors, "Dataset files unavailable/incomplete (check cloud sync):\n" + "\n".join(errors[:12]))

def cache_training_images(cfg, frame, manifest_path, target):
    """Cache training and validation images without changing the manifest."""
    source, target = Path(cfg["dataset_root"]).resolve(), Path(target).resolve()
    require(source != target and source not in target.parents and target not in source.parents,
            "Cache must be separate from the source dataset")
    relative_manifest = Path(cfg["manifest"])
    require(not relative_manifest.is_absolute() and ".." not in relative_manifest.parts,
            "Caching requires a manifest path relative to dataset_root")
    target.mkdir(parents=True, exist_ok=True)
    manifest_bytes = manifest_path.read_bytes()
    local_manifest = target / relative_manifest
    if local_manifest.is_file():
        require(local_manifest.read_bytes() == manifest_bytes, "Existing cache belongs to a different manifest")
    else:
        require(not any(target.iterdir()), "Cache directory is not empty and has no matching manifest")
        atomic_bytes(local_manifest, manifest_bytes)
    rows = frame.loc[frame["split"].isin(["train", "val"])].to_dict("records")
    planned = []
    for row in rows:
        expected = row.get("output_filesize") or row.get("filesize")
        size = int(expected) if expected else (source / row["image_path"]).stat().st_size
        require(size > 0, f"Empty source ID {row['image_id']}")
        destination = target / row["image_path"]
        ready = destination.is_file() and destination.stat().st_size == size
        planned.append((row["image_path"], size, ready))
    needed = sum(size for _, size, ready in planned if not ready)
    require(shutil.disk_usage(target).free > needed + 1024**3,
            f"Runtime disk needs {needed/1024**3:.1f} GiB plus 1 GiB headroom")
    ready_count = sum(ready for _, _, ready in planned)
    print(f"CACHE: {len(planned):,} train/val images; already present {ready_count:,}; "
          f"remaining {needed/1024**3:.2f} GiB. Test images are not copied.", flush=True)
    print("This first copy still reads Drive; later training epochs read runtime disk. No resize, cleaning or split change.", flush=True)
    for parent in {str(PurePosixPath(relative).parent) for relative, _, _ in planned}:
        (target / parent).mkdir(parents=True, exist_ok=True)

    def copy_one(item):
        relative, size, ready = item
        if ready:
            return 0
        destination = target / relative
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            for attempt in range(3):
                try:
                    shutil.copyfile(source / relative, temporary)
                    require(temporary.stat().st_size == size, f"Cache size mismatch: {relative}")
                    os.replace(temporary, destination)
                    return size
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt+1)
        finally:
            temporary.unlink(missing_ok=True)

    start, copied_bytes = time.perf_counter(), 0
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        for count, copied in enumerate(pool.map(copy_one, planned), 1):
            copied_bytes += copied
            if count == 1 or count % 1000 == 0 or count == len(planned):
                elapsed = time.perf_counter()-start
                print(f"CACHE {count:,}/{len(planned):,} | newly copied {copied_bytes/1024**3:.2f} GiB | "
                      f"{copied_bytes/1024**2/max(elapsed,1e-9):.1f} MiB/s", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    save_json(target / "_cache_ready.json", dict(manifest_sha256=sha(manifest_bytes),
              class_mapping=CLASS_MAP, images=len(planned), splits=["train", "val"], verification="file sizes only", created_utc=stamp()))
    print("CACHE READY:", target, "| Use --dataset-root with this path when training. Runtime disk is temporary.", flush=True)

def create_transforms(cfg, training, pre_resized=False):
    operations = [] if pre_resized else [transforms.Resize((cfg["image_size"], cfg["image_size"]), interpolation=InterpolationMode.BILINEAR)]
    a = cfg["augmentation"]
    if training and a["enabled"]:
        operations += [transforms.RandomHorizontalFlip(a["horizontal_flip"]),
                       transforms.RandomVerticalFlip(a["vertical_flip"]),
                       transforms.RandomAffine(a["rotation_degrees"], translate=(a["translate"], a["translate"]),
                                               scale=(1-a["zoom"], 1+a["zoom"]),
                                               interpolation=InterpolationMode.BILINEAR, fill=a["fill"]),
                       transforms.ColorJitter(contrast=a["contrast"])]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class ManifestDataset(Dataset):
    def __init__(self, root, frame, transform, image_cache=None):
        self.image_cache = image_cache
        self.root = Path(root)
        self.rows = frame[["image_id", "image_path", "label"]].to_dict("records")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.image_cache is not None:
            return self.transform(self.image_cache.image(row["image_id"])), row["label"], row["image_id"]
        path = self.root / row["image_path"]
        for attempt in range(3):
            try:
                with Image.open(path) as im:
                    # Avoid copying an already upright RGB image before resizing.
                    orientation = im.getexif().get(274, 1)
                    image = ImageOps.exif_transpose(im) if orientation not in (None, 0, 1) else im
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return self.transform(image), row["label"], row["image_id"]
            except OSError as exc:
                if attempt == 2:
                    raise OSError(f"Unreadable image ID {row['image_id']}: {path}: {exc}") from exc
                time.sleep(0.5 * (attempt+1))

class SharedImageCache:
    """Store pre-resized uint8 pixels in one shared allocation for all workers."""
    def __init__(self, image_ids, image_size):
        self.positions = {image_id: i for i, image_id in enumerate(image_ids)}
        self.shape = (len(image_ids), image_size, image_size, 3)
        self.buffer = RawArray("B", math.prod(self.shape))
        self.statistics = {}

    def pixels(self):
        # Recreate a lightweight view; the pixel buffer is not pickled per worker.
        return np.frombuffer(self.buffer, dtype=np.uint8).reshape(self.shape)

    def image(self, image_id):
        view = self.pixels()[self.positions[image_id]]
        view.setflags(write=False)
        return Image.fromarray(view)

def build_shared_image_cache(cfg, frame):
    """Decode and resize train/val once; leave random augmentation to Dataset."""
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("RAM caching requires psutil. Install requirements.txt before training.") from error
    rows = frame.loc[frame["split"].isin(["train", "val"])].to_dict("records")
    size = cfg["image_size"]
    required = len(rows)*size*size*3
    available = psutil.virtual_memory().available
    reserve = max(2*1024**3, int(required*.25))
    require(available > required+reserve,
            f"RAM cache needs {required/1024**3:.2f} GiB plus {reserve/1024**3:.2f} GiB headroom; "
            f"available {available/1024**3:.2f} GiB. Omit --ram-cache or use a higher-memory runtime.")
    print(f"RAM CACHE: preparing {len(rows):,} train/val images at {size}x{size}; "
          f"shared pixel storage {required/1024**3:.2f} GiB. This initial pass opens images once.", flush=True)
    started = time.perf_counter()
    cache = SharedImageCache([row["image_id"] for row in rows], size)
    pixels = cache.pixels()
    resize = transforms.Resize((size,size), interpolation=InterpolationMode.BILINEAR)
    root = Path(cfg["dataset_root"])

    def prepare(item):
        index, row = item
        path = root / row["image_path"]
        for attempt in range(3):
            try:
                with Image.open(path) as im:
                    orientation = im.getexif().get(274,1)
                    image = ImageOps.exif_transpose(im) if orientation not in (None,0,1) else im
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    pixels[index] = np.asarray(resize(image), dtype=np.uint8)
                return None
            except OSError as error:
                if attempt == 2:
                    raise OSError(f"RAM cache failed for ID {row['image_id']}: {path}: {error}") from error
                time.sleep(attempt+1)

    pool = ThreadPoolExecutor(max_workers=min(4,os.cpu_count() or 1))
    try:
        for count,_ in enumerate(pool.map(prepare, enumerate(rows)),1):
            if count == 1 or count % 1000 == 0 or count == len(rows):
                elapsed = time.perf_counter()-started
                print(f"RAM CACHE {count:,}/{len(rows):,} | {count/max(elapsed,1e-9):.1f} images/s", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    cache.statistics = dict(images=len(rows), bytes=required, seconds=time.perf_counter()-started,
                            image_size=size, dtype="uint8", resize="bilinear stretch", exif="upright",
                            augmentation_cached=False, splits=["train","val"])
    print("RAM CACHE READY: reused by all unfinished models; augmentation remains random per epoch.", flush=True)
    return cache

def seed_worker(_):
    value = torch.initial_seed() % 2**32
    np.random.seed(value)
    random.seed(value)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_dataloaders(cfg, frame, device, image_cache=None):
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = {}
    for split in ("train", "val"):
        data = ManifestDataset(cfg["dataset_root"], frame.loc[frame["split"].eq(split)],
                               create_transforms(cfg, split == "train", pre_resized=image_cache is not None), image_cache)
        loaders[split] = DataLoader(data, batch_size=cfg["batch_size"], shuffle=split == "train",
                                   num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                                   worker_init_fn=seed_worker,
                                   generator=generator if split == "train" else torch.Generator().manual_seed(cfg["seed"]+1))
    return loaders, generator

def create_model(name, cfg, pretrained=True):
    """Initialize the selected backbone and replace its classifier with two outputs."""
    weights = models.get_model_weights(name)[cfg["weights"][name]] if pretrained else None
    model = models.get_model(name, weights=weights)
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def evaluate_model(model, loader, device, amp=False):
    model.eval()
    total_loss, labels_all, probabilities, ids = 0.0, [], [], []
    # Limit progress messages to one per minute during evaluation.
    phase_start = time.perf_counter()
    last_progress = phase_start
    with torch.inference_mode():
        for batch_index, (images, labels, image_ids) in enumerate(loader, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            require(torch.isfinite(loss).item(), "Non-finite evaluation loss")
            total_loss += loss.item() * len(labels)
            labels_all.extend(labels.cpu().tolist())
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            ids.extend(image_ids)
            if time.perf_counter() - last_progress >= 60:
                elapsed = time.perf_counter() - phase_start
                last_progress = time.perf_counter()
                print(f"EVAL batch {batch_index:,}/{len(loader):,} | images {len(ids):,}/{len(loader.dataset):,} | "
                      f"{len(ids)/max(elapsed, 1e-9):.1f} images/s", flush=True)
    return total_loss / len(ids), np.asarray(labels_all), np.concatenate(probabilities), ids

def classification_metrics(labels, probabilities, threshold):
    positive = (labels == 1).astype(int)
    scores = probabilities[:, 1]
    predicted = np.where(scores >= threshold, 1, 0)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    metrics = dict(accuracy=float(accuracy_score(labels, predicted)),
                   balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                   macro_f1=float(f1_score(labels, predicted, average="macro", zero_division=0)),
                   weighted_f1=float(f1_score(labels, predicted, average="weighted", zero_division=0)),
                   macro_precision=float(precision_score(labels, predicted, average="macro", zero_division=0)),
                   macro_recall=float(recall_score(labels, predicted, average="macro", zero_division=0)),
                   cocoa_precision=float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
                   cocoa_recall=float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
                   other_plants_fpr=float(cm[0, 1] / cm[0].sum()),
                   roc_auc_cocoa=float(roc_auc_score(positive, scores)),
                   average_precision_cocoa=float(average_precision_score(positive, scores)),
                   other_plants_precision=float(precision_score(labels, predicted, pos_label=0, zero_division=0)),
                   other_plants_recall=float(recall_score(labels, predicted, pos_label=0, zero_division=0)),
                   TN=int(cm[0,0]), FP=int(cm[0,1]), FN=int(cm[1,0]), TP=int(cm[1,1]),
                   threshold=threshold, positive_class="cocoa", positive_class_id=1)
    report = classification_report(labels, predicted, labels=[0, 1], target_names=NAMES, output_dict=True, zero_division=0)
    return metrics, report, predicted

def prediction_artifacts(folder, prefix, frame, labels, probabilities, ids, threshold, make_plots=True, save_tables=True):
    metrics, report, predicted = classification_metrics(labels, probabilities, threshold)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    if save_tables:
        lookup = frame.set_index("image_id")["image_path"]
        predictions = pd.DataFrame({"image_id": ids, "image_path": [lookup[i] for i in ids],
                                    "true_class_id": labels, "true_class_name": [NAMES[i] for i in labels],
                                    "predicted_class_id": predicted, "predicted_class_name": [NAMES[i] for i in predicted],
                                    "p_cocoa": probabilities[:, 1], "p_other_plants": probabilities[:, 0]})
        predictions["threshold"] = threshold
        predictions["model_name"] = folder.name
        save_csv(folder / f"{prefix}_predictions.csv", predictions)
        save_csv(folder / "confusion_matrix_normalized.csv", pd.DataFrame(cm / cm.sum(1, keepdims=True), columns=NAMES).assign(true_class=NAMES))
        save_csv(folder / "classification_report.csv", pd.DataFrame(report).T.reset_index(names="class_or_average"))
        report_frame = pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T.rename_axis("class_or_average").reset_index()
        save_csv(folder / f"{prefix}_classification_report.csv", report_frame)
        save_csv(folder / f"{prefix}_confusion_matrix.csv", pd.DataFrame(cm, columns=NAMES).assign(true_class=NAMES)[["true_class"]+NAMES])
    if not make_plots:
        return metrics, report
    for normalized in (False, True):
        matrix = cm.astype(float) / cm.sum(1, keepdims=True) if normalized else cm
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(matrix, display_labels=NAMES).plot(ax=ax, cmap="Blues", values_format=".3f" if normalized else "d", colorbar=False)
        ax.set_title(f"EVALUATION SPLIT: {prefix.upper()}: " + ("row-normalized" if normalized else "counts"))
        fig.tight_layout()
        fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
        fig.savefig(folder / f"{prefix}_confusion_matrix{'_normalized' if normalized else ''}.png", dpi=300)
        fig.savefig(folder / f"{prefix}_confusion_matrix{'_normalized' if normalized else ''}.pdf")
        plt.close(fig)
    target, scores = (labels == 1).astype(int), probabilities[:, 1]
    precision, recall, _ = precision_recall_curve(target, scores)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=f"Cocoa AP = {metrics['average_precision_cocoa']:.4f}")
    ax.axhline(target.mean(), color="gray", linestyle="--", label="Class prevalence")
    ax.set(xlabel="Recall (cocoa)", ylabel="Precision (cocoa)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
    fig.savefig(folder / f"{prefix}_precision_recall_curve.png", dpi=300)
    fig.savefig(folder / f"{prefix}_precision_recall_curve.pdf")
    plt.close(fig)
    fpr, tpr, _ = roc_curve(target, scores)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f"Cocoa ROC-AUC = {metrics['roc_auc_cocoa']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate (other plants predicted as cocoa)", ylabel="True positive rate (cocoa recall)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
    fig.savefig(folder / f"{prefix}_roc_curve.png", dpi=300)
    fig.savefig(folder / f"{prefix}_roc_curve.pdf")
    plt.close(fig)
    return metrics, report

def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}

def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])

def train_model(name, cfg, frame, manifest_sha, protocol, run, device, image_cache=None):
    folder = run / name
    folder.mkdir(exist_ok=True)
    metrics_path, last_path, best_path = folder / "metrics.json", folder / "last.pth", folder / "best.pth"
    if metrics_path.is_file():
        done = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(done["protocol"] == protocol and done["status"] == "COMPLETED" and best_path.is_file() and (folder / "validation_predictions.csv").is_file(), f"Completed model metadata mismatch: {folder}")
        print("SKIP COMPLETED:", name, flush=True)
        return done
    seed_all(cfg["seed"])
    started = time.perf_counter()
    loaders, generator = create_dataloaders(cfg, frame, device, image_cache)
    loading = time.perf_counter()
    print(f"MODEL: {name} | creating model/loading weights...", flush=True)
    model = create_model(name, cfg, pretrained=not last_path.is_file()).to(device)
    print(f"MODEL READY: {name}", flush=True)
    synchronize(device)
    model_load_seconds = time.perf_counter() - loading
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    amp = bool(device.type == "cuda" and cfg["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_loss, best_epoch, best_state, wait, first_epoch = math.inf, 0, None, 0, 1
    history, train_seconds, validation_seconds, checkpoint_seconds, previous_seconds = [], 0., 0., 0., 0.
    info = dict(format_version=FORMAT_VERSION, architecture=name, weights=cfg["weights"][name], seed=cfg["seed"],
                manifest_sha256=manifest_sha, protocol=protocol, class_mapping=CLASS_MAP, threshold=cfg["threshold"],
                preprocessing=dict(image_size=cfg["image_size"], resize="stretch", interpolation="bilinear",
                                   mean=MEAN, std=STD, color="RGB", exif="upright"))
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        require(checkpoint["protocol"] == protocol, f"Cannot resume changed experiment: {last_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history, first_epoch = checkpoint["history"], checkpoint["epoch"] + 1
        best_loss, best_epoch, best_state, wait = checkpoint["best_loss"], checkpoint["best_epoch"], load_resume_best(folder, checkpoint), checkpoint["wait"]
        train_seconds, validation_seconds = checkpoint["train_seconds"], checkpoint["validation_seconds"]
        checkpoint_seconds, previous_seconds = checkpoint["checkpoint_seconds"], checkpoint["elapsed_seconds"]
        restore_rng(checkpoint["rng"], generator)
        save_csv(folder / "history.csv", pd.DataFrame(history))
        del checkpoint
        print(f"RESUME {name}: epoch {first_epoch}; best epoch {best_epoch}", flush=True)
    if wait < cfg["early_stop_patience"]:
        for epoch in range(first_epoch, cfg["max_epochs"] + 1):
            model.train()
            synchronize(device)
            epoch_start = time.perf_counter()
            running_loss, correct, seen = 0., 0, 0
            data_wait_seconds = 0.0
            batch_ready = time.perf_counter()
            last_progress = batch_ready
            if epoch == first_epoch:
                print(f"{name}: waiting for first training batch ({len(loaders['train']):,} batches/epoch)...", flush=True)
            for batch_index, (images, labels, _) in enumerate(loaders["train"], 1):
                data_wait_seconds += time.perf_counter() - batch_ready
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                require(torch.isfinite(loss).item(), f"Non-finite loss: {name} epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * len(labels)
                correct += (logits.detach().argmax(1) == labels).sum().item()
                seen += len(labels)
                if (epoch == first_epoch and batch_index == 1) or time.perf_counter() - last_progress >= 60:
                    last_progress = time.perf_counter()
                    elapsed = time.perf_counter() - epoch_start
                    rate = seen / max(elapsed, 1e-9)
                    remaining = (len(loaders["train"].dataset)-seen) / max(rate, 1e-9)
                    print(f"{name} epoch {epoch} batch {batch_index:,}/{len(loaders['train']):,} | "
                          f"{seen:,}/{len(loaders['train'].dataset):,} images | {rate:.1f} images/s | "
                          f"ETA {remaining/60:.1f} min | loader wait {data_wait_seconds:.1f}s", flush=True)
                batch_ready = time.perf_counter()
            synchronize(device)
            train_time = time.perf_counter() - epoch_start
            evaluation_start = time.perf_counter()
            val_loss, labels_all, probs, _ = evaluate_model(model, loaders["val"], device, amp)
            synchronize(device)
            val_time = time.perf_counter() - evaluation_start
            train_seconds += train_time
            validation_seconds += val_time
            val_accuracy = float(accuracy_score(labels_all, np.where(probs[:, 1] >= cfg["threshold"], 1, 0)))
            improved = val_loss < best_loss - cfg["min_delta"]
            save_start = time.perf_counter()
            if improved:
                best_loss, best_epoch, wait = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_checkpoint(best_path, dict(info, epoch=epoch, val_loss=val_loss, model_state_dict=best_state))
            else:
                wait += 1
            history.append(dict(epoch=epoch, train_loss=running_loss/seen, train_accuracy=correct/seen,
                                val_loss=val_loss, val_accuracy=val_accuracy, learning_rate=optimizer.param_groups[0]["lr"],
                                train_seconds=train_time, validation_seconds=val_time, loader_wait_seconds=data_wait_seconds, loader_workers=cfg["num_workers"], ram_cache=image_cache is not None, early_stop_wait=wait))
            save_csv(folder / "history.csv", pd.DataFrame(history))
            checkpoint_seconds += time.perf_counter() - save_start
            finish = wait >= cfg["early_stop_patience"] or epoch == cfg["max_epochs"]
            if epoch == 1 or epoch % cfg["checkpoint_every"] == 0 or finish:
                save_start = time.perf_counter()
                resume_best_ref = store_resume_best(folder, info, best_epoch, best_loss, best_state)
                save_checkpoint(last_path, dict(info, epoch=epoch, model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                                history=history, best_loss=best_loss, best_epoch=best_epoch, resume_best=resume_best_ref, wait=wait,
                                train_seconds=train_seconds, validation_seconds=validation_seconds,
                                checkpoint_seconds=checkpoint_seconds,
                                elapsed_seconds=previous_seconds + time.perf_counter()-started, rng=rng_state(generator)))
                for stale in folder.glob("resume_best_epoch_*.pth"):
                    if stale.name != resume_best_ref["file"]: stale.unlink()
                checkpoint_seconds += time.perf_counter() - save_start
            print(f"{name} epoch {epoch}/{cfg['max_epochs']} | loss {running_loss/seen:.4f}/{val_loss:.4f} | "
                  f"accuracy {correct/seen:.4f}/{val_accuracy:.4f} | train {train_time:.1f}s val {val_time:.1f}s | "
                  f"loader wait {data_wait_seconds:.1f}s | early-stop wait {wait}", flush=True)
            if finish:
                break
    require(best_state is not None, "No completed epoch is available")
    model.load_state_dict(best_state)
    # Restore the best checkpoint associated with the resumed history.
    save_checkpoint(best_path, dict(info, epoch=best_epoch, val_loss=best_loss, model_state_dict=best_state))
    evaluation_start = time.perf_counter()
    val_loss, labels_all, probs, ids = evaluate_model(model, loaders["val"], device, amp)
    synchronize(device)
    validation_seconds += time.perf_counter() - evaluation_start
    metrics, report = prediction_artifacts(folder, "validation", frame, labels_all, probs, ids, cfg["threshold"], make_plots=False)
    # Defer figures until the full model comparison is complete.
    latency = dict(forward_ms_batch1=None)
    result = dict(metrics, **latency, model=name, status="COMPLETED", protocol=protocol, stage="validation",
                  epochs_run=len(history), best_epoch=best_epoch, best_val_loss=best_loss, evaluated_val_loss=val_loss,
                  train_seconds=train_seconds, validation_seconds=validation_seconds, checkpoint_seconds=checkpoint_seconds,
                  model_load_seconds=model_load_seconds, run_seconds=previous_seconds+time.perf_counter()-started,
                  parameters=sum(p.numel() for p in model.parameters()), best_file_bytes=best_path.stat().st_size,
                  mixed_precision=amp, class_mapping=CLASS_MAP, classification_report=report, test_evaluated=False)
    save_json(folder / "validation_metrics.json", result)
    save_csv(folder / "training_history.csv", pd.DataFrame(history))
    save_json(metrics_path, result)
    print(f"SAVED {name}: {folder}", flush=True)
    del model, optimizer, scaler, best_state, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024*1024), b""): h.update(data)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def store_resume_best(folder, info, epoch, loss, state):
    """Keep an immutable best snapshot referenced by the last checkpoint."""
    path = folder / f"resume_best_epoch_{epoch}.pth"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        equal = (saved["protocol"] == info["protocol"] and saved["val_loss"] == loss and
                 all(torch.equal(saved["model_state_dict"][k], v) for k,v in state.items()))
        del saved
        if not equal: path = folder / f"resume_best_epoch_{epoch}_{uuid.uuid4().hex[:8]}.pth"
    if not path.exists():
        save_checkpoint(path, dict(info, epoch=epoch, val_loss=loss, model_state_dict=state))
    return dict(file=path.name, sha256=file_sha(path))


def load_resume_best(folder, checkpoint):
    reference = checkpoint["resume_best"]
    relative = Path(reference["file"])
    require(relative.name == str(relative), "Unsafe resume snapshot reference")
    path = folder / relative
    require(file_sha(path) == reference["sha256"], "Resume best snapshot checksum differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved["protocol"] == checkpoint["protocol"] and saved["epoch"] == checkpoint["best_epoch"],
            "Resume snapshot identity differs")
    return saved["model_state_dict"]


def configuration(root, output, ram_cache=True):
    return dict(dataset_root=str(root), output_root=str(output), manifest="model1_metadata.csv",
        models=[MODEL_NAME], weights={MODEL_NAME:"IMAGENET1K_V1"}, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, learning_rate=.0003, weight_decay=WEIGHT_DECAY,
        early_stop_patience=EARLY_STOP_PATIENCE, min_delta=MIN_DELTA, seed=SEED, threshold=THRESHOLD,
        device="cuda",num_workers=NUM_WORKERS,mixed_precision=True,checkpoint_every=CHECKPOINT_EVERY,
        ram_cache=ram_cache,augmentation=dict(AUGMENTATION))


def read_dataset(cfg, baseline_info):
    path = Path(cfg["dataset_root"]) / cfg["manifest"]
    dataset_sha = file_sha(path)
    require(dataset_sha == baseline_info["manifest_sha256"], "Dataset differs from the initial benchmark")
    frame = pd.read_csv(path,dtype=str,keep_default_na=False,encoding="utf-8-sig")
    require({"image_id","image_path","class_name","class_id","split","status"} <= set(frame), "Missing metadata columns")
    require(frame.image_id.is_unique and frame.image_path.is_unique, "Duplicate IDs or image paths")
    require(frame.status.eq("KEEP").all(), "Metadata contains non-KEEP rows")
    require(set(frame.class_name)==set(CLASS_MAP), "Class names differ")
    frame["label"] = frame.class_name.map(CLASS_MAP)
    require(frame.class_id.astype(int).eq(frame.label).all(), "Class mapping differs")
    for path_value in frame.image_path:
        path_obj=PurePosixPath(path_value.replace("\\","/"))
        require(not path_obj.is_absolute() and ".." not in path_obj.parts and ":" not in path_value, "Unsafe image path")
    expected={("train","other_plants"):10721,("train","cocoa"):7147,("val","other_plants"):1341,("val","cocoa"):894}
    working=frame.loc[frame.split.isin(["train","val"])].copy()
    require(working.groupby(["split","class_name"]).size().to_dict()==expected, "Train/validation counts differ")
    for key in ["split_group_id","PN_observation_id","PN_hash","file_sha256","output_sha256","output_pixel_sha256"]:
        if key in frame:
            used=frame.loc[frame[key].ne("")]
            require(not used.groupby(key).split.nunique().gt(1).any(), f"Known split leakage: {key}")
    marker=Path(cfg["dataset_root"])/"_cache_ready.json"
    if marker.exists():
        identity=load_json(marker)
        require(identity["manifest_sha256"]==dataset_sha and identity.get("class_mapping")==CLASS_MAP,
                "Cache identity differs")
    # The returned dataset contains train/validation rows only. Test pixels are never loaded.
    return working,dataset_sha,path


def baseline_inputs(folder):
    cfg=load_json(folder/"config.json")
    info=load_json(folder/"run_info.json")
    model_dir=folder/MODEL_NAME
    metrics=load_json(model_dir/"metrics.json")
    require(info["class_mapping"]==CLASS_MAP and metrics["class_mapping"]==CLASS_MAP, "Baseline class mapping differs")
    require(metrics["status"]=="COMPLETED" and metrics["stage"]=="validation" and not metrics["test_evaluated"], "Baseline is incomplete")
    require(metrics["protocol"]==info["protocol"], "Baseline protocol differs")
    for key,value in dict(image_size=224,batch_size=64,learning_rate=.001,weight_decay=.0001,
                          max_epochs=100,early_stop_patience=20,min_delta=0.,seed=42,threshold=.5).items():
        require(cfg[key]==value,f"Unexpected baseline setting: {key}")
    require(cfg["augmentation"]==AUGMENTATION and cfg["weights"][MODEL_NAME]=="IMAGENET1K_V1",
            "Baseline augmentation or pretrained weights differ")
    sources={"config.json":folder/"config.json","run_info.json":folder/"run_info.json",
        "metrics.json":model_dir/"metrics.json","history.csv":model_dir/"history.csv",
        "validation_predictions.csv":model_dir/"validation_predictions.csv","best.pth":model_dir/"best.pth"}
    hashes={key:file_sha(path) for key,path in sources.items()}
    return cfg,info,metrics,sources,hashes


def validate_predictions(folder, frame, protocol, metrics):
    require(metrics["status"]=="COMPLETED" and metrics["stage"]=="validation" and
            metrics["protocol"]==protocol and metrics["class_mapping"]==CLASS_MAP and
            metrics["test_evaluated"] is False, f"Incompatible validation result: {folder}")
    p=pd.read_csv(folder/"validation_predictions.csv",dtype={"image_id":str},keep_default_na=False)
    require(p.image_id.is_unique,"Duplicate validation predictions")
    actual=frame.loc[frame.split.eq("val")].set_index("image_id").sort_index()
    p=p.set_index("image_id").sort_index()
    require(p.index.equals(actual.index), "Validation image IDs differ")
    require(p.image_path.eq(actual.image_path).all() and p.true_class_id.eq(actual.label).all(), "Validation paths or labels differ")
    probabilities=p[["p_other_plants","p_cocoa"]].to_numpy(float)
    require(np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all() and
            np.allclose(probabilities.sum(1),1,atol=1e-5), "Invalid saved probabilities")
    require(p.threshold.eq(.5).all(), "Threshold differs")
    recomputed,_,predicted=classification_metrics(p.true_class_id.to_numpy(int),probabilities,.5)
    require(np.array_equal(predicted,p.predicted_class_id.to_numpy()),"Saved predicted labels differ")
    for key in ["accuracy","balanced_accuracy","macro_f1","cocoa_precision","cocoa_recall","other_plants_fpr","roc_auc_cocoa","average_precision_cocoa"]:
        require(abs(recomputed[key]-metrics[key])<1e-7,f"Saved validation metric differs: {key}")
    history=pd.read_csv(folder/"history.csv",encoding="utf-8-sig")
    require(len(history)==metrics["epochs_run"] and metrics["best_epoch"] in history.epoch.values, "Incomplete training history")
    return p.reset_index(),history


def identity(cfg, dataset_sha, hashes):
    portable={k:v for k,v in cfg.items() if k not in {"dataset_root","output_root","ram_cache","device"}}
    return sha(json.dumps(dict(version=FORMAT_VERSION,config=portable,manifest=dataset_sha,baseline=hashes,
        trials=TRIALS,near_tie=NEAR_TIE,script_sha256=file_sha(Path(__file__))),sort_keys=True).encode())


def trial_protocol(job_id, name, lr):
    return sha(json.dumps([job_id,name,lr]).encode())


def resolve_run(output, job_id, cfg, dataset_sha, sources, hashes, new_run=False):
    active=output/"active_run.json"
    if active.exists() and not new_run:
        saved=load_json(active)
        require(saved["identity"]==job_id,"Active run is incompatible. Inspect it or explicitly use --new-run.")
        require(Path(saved["run_name"]).name==saved["run_name"], "Unsafe active run path")
        run=output/saved["run_name"]
        require(load_json(run/"tuning_run.json")["identity"]==job_id,"Active run metadata differs")
        return run
    run=output/("tuning_"+stamp()+"_"+uuid.uuid4().hex[:6])
    run.mkdir(parents=True,exist_ok=False)
    snapshot=run/"baseline_reference";snapshot.mkdir()
    for name,src in sources.items():
        shutil.copyfile(src,snapshot/name)
        require(file_sha(snapshot/name)==hashes[name],f"Baseline snapshot checksum differs: {name}")
    save_json(run/"config.json",dict(cfg,trials=TRIALS,class_mapping=CLASS_MAP,stopping_note=STOPPING_NOTE))
    save_json(run/"dataset_identity.json",dict(manifest_sha256=dataset_sha,class_mapping=CLASS_MAP,train_images=17868,validation_images=2235))
    save_json(run/"tuning_run.json",dict(identity=job_id,created_utc=stamp(),baseline_hashes=hashes,seed=42,
        script_sha256=file_sha(Path(__file__)),test_evaluated=False,stopping_note=STOPPING_NOTE))
    atomic_bytes(run/"code_snapshot.py",Path(__file__).read_bytes())
    save_json(active,dict(identity=job_id,run_name=run.name))
    return run


def completed_trial(run,name,protocol,frame):
    folder=run/name/MODEL_NAME
    if not (folder/"metrics.json").exists(): return False
    metrics=load_json(folder/"metrics.json")
    validate_predictions(folder,frame,protocol,metrics)
    receipt=folder/"completed.json"
    require((folder/"best.pth").is_file() and (folder/"last.pth").is_file(),"Completed trial lacks checkpoints")
    if receipt.exists():
        for filename,expected in load_json(receipt)["files"].items():
            require(file_sha(folder/filename)==expected,f"Completed artifact changed: {name}/{filename}")
    else:
        checkpoint = torch.load(folder/"best.pth", map_location="cpu", weights_only=True)
        require(checkpoint["protocol"]==protocol and checkpoint["architecture"]==MODEL_NAME and
                checkpoint["class_mapping"]==CLASS_MAP and checkpoint["epoch"]==metrics["best_epoch"] and
                checkpoint["threshold"]==.5, "Completed checkpoint metadata differs")
        del checkpoint
        save_json(receipt,dict(status="COMPLETED",protocol=protocol,files={
            filename:file_sha(folder/filename) for filename in ["best.pth","last.pth","history.csv","validation_predictions.csv","metrics.json"]}))
    return True


def save_publication(fig,folder,name):
    fig.text(.5,.005,"EVALUATION SPLIT: VALIDATION | Seed 42 | Threshold 0.5",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.97])
    fig.savefig(folder/(name+".png"),dpi=300)
    fig.savefig(folder/(name+".pdf"))
    plt.close(fig)


def export_application_candidate(run,experiment,folder,selection):
    target=run/"application_candidate"
    target.mkdir(exist_ok=True)
    previous = target/"inference_config.json"
    if previous.exists() and (target/"best.pth").exists():
        saved = load_json(previous)
        if (saved.get("source_checkpoint_sha256")==file_sha(folder/"best.pth") and
            saved.get("selection")==selection and saved.get("weights_sha256")==file_sha(target/"best.pth")):
            print("APPLICATION CANDIDATE WEIGHTS ALREADY READY:", target, flush=True)
            return
    checkpoint=torch.load(folder/"best.pth",map_location="cpu",weights_only=True)
    require(checkpoint["class_mapping"]==CLASS_MAP and checkpoint["architecture"]==MODEL_NAME,"Application checkpoint mapping differs")
    payload=dict(architecture=MODEL_NAME,model_state_dict=checkpoint["model_state_dict"],class_mapping=CLASS_MAP,
        preprocessing=checkpoint["preprocessing"],threshold=.5,epoch=checkpoint["epoch"],
        source_checkpoint_sha256=file_sha(folder/"best.pth"),experiment=experiment,test_evaluated=False,
        status="VALIDATION_SELECTED_CANDIDATE_NOT_FINAL_TESTED")
    save_checkpoint(target/"best.pth",payload)
    del checkpoint,payload
    save_json(target/"class_mapping.json",CLASS_MAP)
    save_json(target/"inference_config.json",dict(architecture=MODEL_NAME,image_size=224,resize="stretch",
        interpolation="bilinear",color="RGB",exif="upright",mean=MEAN,std=STD,
        threshold=.5,positive_class_id=1,class_mapping=CLASS_MAP,
        status="VALIDATION_SELECTED_CANDIDATE_NOT_FINAL_TESTED",selection=selection,
        source_checkpoint_sha256=file_sha(folder/"best.pth"),weights_sha256=file_sha(target/"best.pth")))
    predictor = r'''"""Load the validation-selected EfficientNet-B0 candidate for application inference."""
from pathlib import Path
import json
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps

class Predictor:
    def __init__(self, folder, device=None):
        folder = Path(folder)
        self.config = json.loads((folder / "inference_config.json").read_text())
        assert self.config["class_mapping"] == {"other_plants": 0, "cocoa": 1}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(folder / "best.pth", map_location="cpu", weights_only=True)
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 2)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(self.config["mean"], self.config["std"]),
        ])

    @torch.inference_mode()
    def predict(self, path):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        probs = self.model(tensor).float().softmax(1)[0].cpu().tolist()
        label = int(probs[1] >= self.config["threshold"])
        return {"class_id": label, "class_name": ["other_plants", "cocoa"][label],
                "p_other_plants": probs[0], "p_cocoa": probs[1],
                "threshold": self.config["threshold"]}
'''
    atomic_bytes(target/"inference.py",predictor.encode())
    print("APPLICATION CANDIDATE WEIGHTS:",target,flush=True)


def report(run,frame):
    run=Path(run)
    info=load_json(run/"tuning_run.json")
    for trial_name,lr in TRIALS.items():
        require(completed_trial(run,trial_name,trial_protocol(info["identity"],trial_name,lr),frame),
                f"Report requires a complete trial: {trial_name}")
    baseline=run/"baseline_reference"
    bc=load_json(baseline/"config.json")
    locations=[("baseline",baseline,bc["learning_rate"],bc["max_epochs"],bc["early_stop_patience"])]
    locations += [(name,run/name/MODEL_NAME,lr,1000,30) for name,lr in TRIALS.items()]
    entries={};rows=[]
    for name,folder,lr,epochs,patience in locations:
        metrics=load_json(folder/"metrics.json")
        protocol=metrics["protocol"] if name=="baseline" else trial_protocol(info["identity"],name,lr)
        if name=="baseline":
            for filename,expected in info["baseline_hashes"].items():
                require(file_sha(baseline/filename)==expected,"Baseline snapshot changed")
        p,history=validate_predictions(folder,frame,protocol,metrics)
        entries[name]=(folder,metrics,p,history)
        rows.append(dict(evaluation_split="VALIDATION",experiment=name,learning_rate=lr,weight_decay=.0001,max_epochs=epochs,
            early_stop_patience=patience,seed=42,best_epoch=metrics["best_epoch"],epochs_run=metrics["epochs_run"],
            validation_loss=metrics["evaluated_val_loss"],**{k:metrics[k] for k in [
            "accuracy","balanced_accuracy","cocoa_precision","cocoa_recall","other_plants_fpr","macro_f1",
            "roc_auc_cocoa","average_precision_cocoa","TP","TN","FP","FN","train_seconds","validation_seconds"]}))
    table=pd.DataFrame(rows)
    save_csv(run/"hyperparameter_comparison.csv",table)
    figures=run/"publication_figures";figures.mkdir(exist_ok=True)
    for name,(folder,metrics,p,history) in entries.items():
        target_dir=figures/name;target_dir.mkdir(exist_ok=True)
        labels=p.true_class_id.to_numpy(int);probs=p[["p_other_plants","p_cocoa"]].to_numpy(float)
        prediction_artifacts(target_dir,"validation",p,labels,probs,p.image_id.tolist(),.5,make_plots=True)
        # The copied/report prediction rows retain the actual experiment identifier.
        saved=pd.read_csv(target_dir/"validation_predictions.csv",dtype={"image_id":str})
        saved["experiment"]=name
        save_csv(target_dir/"validation_predictions.csv",saved)
    for kind in ["roc","precision_recall"]:
        fig,ax=plt.subplots(figsize=(7,5))
        for name,(_,metrics,p,_) in entries.items():
            labels=p.true_class_id.to_numpy(int);scores=p.p_cocoa.to_numpy(float)
            if kind=="roc":
                x,y,thresholds=roc_curve(labels,scores)
                label=f"{name} (AUC={metrics['roc_auc_cocoa']:.5f})"
                coordinates=pd.DataFrame(dict(fpr=x,tpr=y,threshold=thresholds))
            else:
                y,x,thresholds=precision_recall_curve(labels,scores)
                label=f"{name} (AP={metrics['average_precision_cocoa']:.5f})"
                coordinates=pd.DataFrame(dict(recall=x,precision=y))
            save_csv(figures/(name+"_"+kind+"_coordinates.csv"),coordinates)
            ax.plot(x,y,label=label)
        if kind=="roc":
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.set(xlabel="Other Plants false positive rate",ylabel="Cocoa recall",title="Validation ROC comparison")
        else:
            ax.set(xlabel="Cocoa recall",ylabel="Cocoa precision",title="Validation precision-recall comparison")
        ax.set(xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8);ax.grid(alpha=.2)
        save_publication(fig,figures,kind+"_comparison")
    for metric in ["loss","accuracy"]:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,(name,(_,metrics,_,history)) in zip(axes,entries.items()):
            ax.plot(history.epoch,history["train_"+metric],label="Train")
            ax.plot(history.epoch,history["val_"+metric],label="Validation")
            ax.axvline(metrics["best_epoch"],color="gray",linestyle=":",label="Best epoch")
            ax.set(title=name,xlabel="Epoch",ylabel=metric.title());ax.legend(fontsize=8);ax.grid(alpha=.2)
        save_publication(fig,figures,"training_validation_"+metric)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,key in zip(axes.flat,["cocoa_recall","other_plants_fpr","balanced_accuracy","validation_loss"]):
        for row in rows:
            ax.scatter(row["learning_rate"],row[key],label=row["experiment"])
        ax.set_xscale("log");ax.set(xlabel="Learning rate",ylabel=key.replace("_"," "),title="Validation results")
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Learning-rate configurations: stopping protocols also differ")
    save_publication(fig,figures,"learning_rate_validation_performance")
    eligible=table.loc[table.other_plants_fpr.le(.01)].sort_values(
        ["cocoa_recall","balanced_accuracy","macro_f1","validation_loss"],ascending=[False,False,False,True])
    candidate=None;ties=[]
    if len(eligible):
        top=eligible.iloc[0]
        mask=pd.Series(True,index=eligible.index)
        for key,tolerance in NEAR_TIE.items(): mask &= (eligible[key]-top[key]).abs().le(tolerance+1e-12)
        ties=eligible.loc[mask,"experiment"].tolist()
        candidate="baseline" if "baseline" in ties else str(top.experiment)
    selection=dict(candidate=candidate,status="NO_ELIGIBLE_CONFIGURATION" if candidate is None else
        ("NEAR_TIE" if len(ties)>1 else "DESCRIPTIVE_CANDIDATE"),
        near_tied_configurations=ties,near_tie_tolerances=NEAR_TIE,threshold=.5,fpr_limit=.01,seed=42,
        criteria=["FPR <= 0.01","maximize Cocoa recall","maximize balanced accuracy","maximize macro F1","minimize validation loss"],
        improvement_supported=candidate is not None and candidate!="baseline" and "baseline" not in ties,
        interpretation="Descriptive validation comparison; no statistical significance claim. "+STOPPING_NOTE,
        validation_only=True,test_evaluated=False)
    save_json(run/"selection.json",selection)
    if candidate:
        export_application_candidate(run,candidate,entries[candidate][0],selection)
    save_json(run/"report_summary.json",dict(status="COMPLETED",evaluation_split="VALIDATION",
        selection=selection,stopping_note=STOPPING_NOTE,baseline_reinferred=False,
        test_evaluated=False,seed=42,configurations=rows,versions=dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__))))
    print(table.to_string(index=False),flush=True)
    print("SELECTION:",json.dumps(selection),flush=True)
    print("VALIDATION REPORT:",figures,flush=True)
    return selection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["check","cache","train","report"],default="check")
    parser.add_argument("--project",type=Path,default=PROJECT_ROOT)
    parser.add_argument("--dataset-root",type=Path)
    parser.add_argument("--baseline-run",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--cache-root",type=Path)
    parser.add_argument("--no-ram-cache",action="store_true")
    parser.add_argument("--new-run",action="store_true")
    args=parser.parse_args(argv)
    project=args.project.resolve()
    baseline=(args.baseline_run or project/"results"/"model1_initial_benchmark"/BASELINE_RUN_NAME).resolve()
    output=(args.output_root or project/"results"/"model1_efficientnet_tuning").resolve()
    root=(args.dataset_root or project/"dataset").resolve()
    require(output!=baseline and baseline not in output.parents and output not in baseline.parents,"Tuning output must be separate from baseline")
    print("STAGE:",args.stage,flush=True)
    print("Reading current metadata and saved baseline; no test images will be opened.",flush=True)
    bc,bi,bm,sources,hashes=baseline_inputs(baseline)
    cfg=configuration(root,output,not args.no_ram_cache)
    frame,dataset_sha,manifest_path=read_dataset(cfg,bi)
    validate_predictions(baseline/MODEL_NAME,frame,bi["protocol"],bm)
    job_id=identity(cfg,dataset_sha,hashes)
    if args.stage=="cache":
        require(args.cache_root is not None,"Specify --cache-root")
        cache_training_images(cfg,frame,manifest_path,args.cache_root)
        return 0
    if args.stage=="check":
        if not (root/"_cache_ready.json").exists():
            print("Checking train/validation file availability once.",flush=True)
            check_files(cfg,frame,["train","val"])
        checkpoint=torch.load(sources["best.pth"],map_location="cpu",weights_only=True)
        require(checkpoint["manifest_sha256"]==dataset_sha and checkpoint["class_mapping"]==CLASS_MAP and
            checkpoint["protocol"]==bi["protocol"] and checkpoint["architecture"]==MODEL_NAME,"Baseline checkpoint identity differs")
        require(checkpoint["preprocessing"]==dict(image_size=224,resize="stretch",interpolation="bilinear",
            mean=MEAN,std=STD,color="RGB",exif="upright"),"Baseline preprocessing differs")
        # A two-image forward pass checks the new ImageNet-initialized head; no optimizer is created.
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_all(SEED)
        model=create_model(MODEL_NAME,cfg,True).to(device).eval()
        samples=pd.concat([frame.loc[frame.split.eq("train") & frame.label.eq(label)].head(1) for label in [0,1]])
        data=ManifestDataset(root,samples,create_transforms(cfg,False))
        images=torch.stack([data[i][0] for i in range(2)]).to(device)
        with torch.inference_mode(): logits=model(images)
        require(tuple(logits.shape)==(2,2) and torch.isfinite(logits).all().item(),"Forward smoke test failed")
        require([data[i][1] for i in range(2)]==[0,1],"Smoke labels differ")
        output.mkdir(parents=True,exist_ok=True)
        save_json(output/("preflight_"+job_id+".json"),dict(status="PASSED",identity=job_id,
            script_sha256=file_sha(Path(__file__)),manifest_sha256=dataset_sha,training_started=False,
            device=str(device),class_mapping=CLASS_MAP,test_evaluated=False))
        print("PREFLIGHT PASSED. No training started.",flush=True)
        return 0
    output.mkdir(parents=True,exist_ok=True)
    if args.stage=="report":
        active=load_json(output/"active_run.json")
        require(active["identity"]==job_id,"Report configuration or dataset differs")
        report(output/active["run_name"],frame)
        return 0
    require(torch.cuda.is_available(),"CUDA GPU is required. Select a Colab GPU runtime; CPU training is disabled.")
    device=torch.device("cuda")
    print("GPU:",torch.cuda.get_device_name(0),"| mixed precision: enabled",flush=True)
    run=resolve_run(output,job_id,cfg,dataset_sha,sources,hashes,args.new_run)
    print("ACTIVE TUNING RUN:",run,flush=True)
    pending=[]
    for name,lr in TRIALS.items():
        if completed_trial(run,name,trial_protocol(job_id,name,lr),frame):
            print("SKIP COMPLETED:",name,flush=True)
        else: pending.append((name,lr))
    if not pending:
        print("Both trials are complete; regenerating validation reports only.",flush=True)
        report(run,frame);return 0
    receipt=run/"data_preflight.json"
    if not receipt.exists():
        if (root/"_cache_ready.json").exists():
            print("Using matching runtime disk cache; no full Drive image scan.",flush=True)
        else:
            print("No runtime cache marker; checking train/validation files once.",flush=True)
            check_files(cfg,frame,["train","val"])
        save_json(receipt,dict(identity=job_id,manifest_sha256=dataset_sha,class_mapping=CLASS_MAP))
    else:
        require(load_json(receipt)["identity"]==job_id,"Stored preflight differs")
    image_cache=None
    if cfg["ram_cache"]:
        import psutil
        needed=len(frame)*224*224*3
        if psutil.virtual_memory().available>needed+max(2*1024**3,int(needed*.25)):
            image_cache=build_shared_image_cache(cfg,frame)
        else: print("RAM cache skipped: insufficient headroom. Training will use disk cache.",flush=True)
    for name,lr in pending:
        trial_root=run/name;trial_root.mkdir(exist_ok=True)
        trial_cfg=dict(cfg,learning_rate=lr)
        protocol=trial_protocol(job_id,name,lr)
        save_json(trial_root/"trial_config.json",dict(trial_cfg,protocol=protocol,initialization="ImageNet IMAGENET1K_V1; fresh per trial except same-trial resume"))
        print(f"TRIAL: {name} | LR={lr} | max epochs=1000 | patience=30 | seed=42",flush=True)
        train_model(MODEL_NAME,trial_cfg,frame,dataset_sha,protocol,trial_root,device,image_cache)
        require(completed_trial(run,name,protocol,frame),"Trial did not finalize completely")
    del image_cache
    gc.collect();torch.cuda.empty_cache()
    report(run,frame)
    save_json(run/"COMPLETED.json",dict(status="COMPLETED",identity=job_id,seed=42,test_evaluated=False))
    return 0


if __name__=="__main__":
    try:
        import contextlib
        @contextlib.contextmanager
        def local_writer_lock():
            args_text = sys.argv[1:]
            if "--stage" not in args_text or args_text[args_text.index("--stage")+1] not in ("train","report"):
                yield
                return
            lock_path = Path(tempfile.gettempdir())/"model1_efficientnet_tuning_writer.lock"
            handle = lock_path.open("a+b")
            handle.seek(0)
            if handle.read(1)==b"": handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                handle.close()
                raise RuntimeError("Another tuning/report writer is active in this runtime.")
            try: yield
            finally: handle.close()
        with local_writer_lock():
            raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED. Re-run the training cell to resume the active tuning run.",flush=True)
        raise SystemExit(130)
