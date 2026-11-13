"""Run an initial four-model benchmark with the current Cocoa/Other Plants manifest.
Stages are check, cache, train, report, and a reserved disabled test stage.
No tuning or threshold optimization is performed."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.sharedctypes import RawArray
import argparse
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

PROJECT_ROOT = Path(r"H:\My Drive\cocoa-diseases")
DATASET_ROOT = PROJECT_ROOT / "dataset"
MANIFEST_NAME = "model1_metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "model1_initial_benchmark"
CLASS_MAP = {"other_plants": 0, "cocoa": 1}
NAMES = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
MODEL_NAMES = ["mobilenet_v2", "resnet50", "convnext_tiny", "efficientnet_b0"]
PRETRAINED_WEIGHTS = {"mobilenet_v2": "IMAGENET1K_V2", "resnet50": "IMAGENET1K_V2", "convnext_tiny": "IMAGENET1K_V1", "efficientnet_b0": "IMAGENET1K_V1"}
EXPECTED_COUNTS = {"train": {"other_plants": 10721, "cocoa": 7147}, "val": {"other_plants": 1341, "cocoa": 894}, "test": {"other_plants": 1341, "cocoa": 894}}
IMAGE_SIZE, BATCH_SIZE, MAX_EPOCHS = 224, 64, 100
LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_PATIENCE, MIN_DELTA = 0.001, 0.0001, 20, 0.0
SEED, THRESHOLD, NUM_WORKERS, MIXED_PRECISION, CHECKPOINT_EVERY = 42, 0.5, 2, True, 5
AUGMENTATION = {"enabled": True, "horizontal_flip": 0.5, "vertical_flip": 0.5, "rotation_degrees": 20, "zoom": 0.1, "translate": 0.1, "contrast": 0.2, "fill": 0}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
FORMAT_VERSION = "model1-initial-benchmark-current-manifest-v1"
SUPPORTED = {"mobilenet_v2", "resnet50", "convnext_tiny", "efficientnet_b0"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
FORMAT_VERSION = "model1-initial-benchmark-current-manifest-v1"


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


def load_config(dataset_root=None, output_root=None):
    root = Path(dataset_root or DATASET_ROOT).expanduser().resolve()
    output = Path(output_root or OUTPUT_ROOT).expanduser().resolve()
    return dict(dataset_root=str(root), manifest=MANIFEST_NAME, output_root=str(output), models=list(MODEL_NAMES),
                weights=dict(PRETRAINED_WEIGHTS), image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, early_stop_patience=EARLY_STOP_PATIENCE,
                min_delta=MIN_DELTA, seed=SEED, threshold=THRESHOLD, device="auto", num_workers=NUM_WORKERS,
                mixed_precision=MIXED_PRECISION, checkpoint_every=CHECKPOINT_EVERY, ram_cache=False,
                augmentation=dict(AUGMENTATION))

def read_manifest(cfg):
    path = Path(cfg["dataset_root"]) / cfg["manifest"]
    content = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    require({"image_id", "image_path", "class_name", "class_id", "split"} <= set(frame), f"Manifest columns missing; found: {list(frame.columns)}")
    require(len(frame) > 0, "Empty manifest")
    require(not frame["image_id"].eq("").any() and frame["image_id"].is_unique, "Blank/duplicate image_id")
    require(set(frame["class_name"]) == set(CLASS_MAP), f"Expected class names exactly: {CLASS_MAP}; found {sorted(set(frame['class_name']))}")
    require(set(frame["split"]) <= {"train", "val", "test"}, "Invalid split; use train/val/test")
    for status in ("status", "final_status"):
        if status in frame:
            require(frame[status].eq("KEEP").all(), f"Manifest contains non-KEEP rows in {status}")
    frame["label"] = frame["class_name"].map(CLASS_MAP).astype(int)
    require(frame["class_id"].eq(frame["label"].astype(str)).all(), "class_id must match other_plants=0, cocoa=1; registry_class_id is provenance only")
    frame["image_path"] = frame["image_path"].str.replace("\\", "/", regex=False)
    for value in frame["image_path"]:
        p = PurePosixPath(value)
        require(value and not p.is_absolute() and ".." not in p.parts and ":" not in value, f"Unsafe relative image_path: {value}")
    require(frame["image_path"].is_unique, "Duplicate image_path in manifest")
    if "split_group_id" in frame:
        groups = frame.loc[frame["split_group_id"].ne("")]
        require(not groups.groupby("split_group_id")["split"].nunique().gt(1).any(), "A split_group_id crosses train/val/test")
    for key in ("PN_observation_id", "PN_hash", "file_sha256", "source_sha256", "output_sha256", "output_pixel_sha256"):
        if key in frame:
            values = frame.loc[frame[key].ne("")].groupby(key)["split"].nunique()
            require(not values.gt(1).any(), f"Known cross-split leakage in {key}")
    actual = frame.groupby(["split", "class_name"]).size().to_dict()
    expected = {(split, name): count for split, classes in EXPECTED_COUNTS.items() for name, count in classes.items()}
    require(actual == expected, f"Dataset split counts differ: actual={actual}; expected={expected}")
    marker = Path(cfg["dataset_root"]) / "_cache_ready.json"
    if marker.exists():
        cache = json.loads(marker.read_text(encoding="utf-8"))
        require(cache["manifest_sha256"] == sha(content), "Cache manifest identity differs")
        require(cache.get("class_mapping") == CLASS_MAP, "Cache class mapping differs")
    return frame, sha(content), path


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



def benchmark_forward(model, cfg, device):
    """Measure float32 batch-one latency after 20 warmups over 100 repeats.

    File loading, preprocessing and host-to-device transfer are excluded.
    """
    model.eval()
    image = torch.zeros(1, 3, cfg["image_size"], cfg["image_size"], device=device)
    timings = []
    with torch.inference_mode():
        for _ in range(20):
            model(image)
        synchronize(device)
        for _ in range(100):
            start = time.perf_counter()
            model(image)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    return dict(forward_ms_batch1=float(np.mean(timings)), forward_ms_batch1_median=float(np.median(timings)),
                forward_ms_batch1_p95=float(np.percentile(timings, 95)), forward_precision="float32",
                forward_repeats=100, forward_warmup=20,
                forward_scope="model forward only; excludes file loading, preprocessing and host/device transfer")


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


def history_plots(folder, history):
    frame = pd.DataFrame(history)
    for suffix, columns, ylabel in (("loss", ["train_loss", "val_loss"], "Cross-entropy loss"),
                                    ("accuracy", ["train_accuracy", "val_accuracy"], "Accuracy")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for column in columns:
            ax.plot(frame.epoch, frame[column], label=column)
        ax.axvline(int(frame.loc[frame.val_loss.idxmin(), "epoch"]), linestyle=":", color="gray", label="Best epoch")
        ax.set(title="EVALUATION SPLIT: VALIDATION", xlabel="Epoch", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
        fig.savefig(folder / f"{suffix}_curve.png", dpi=300)
        fig.savefig(folder / f"{suffix}_curve.pdf")
        plt.close(fig)


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


def smoke_test(cfg, frame, device):
    """Load known-label examples and run inference without updating model weights."""
    rows = []
    for name, label in sorted(CLASS_MAP.items(), key=lambda item: item[1]):
        subset = frame.loc[(frame["split"] == "train") & (frame["class_name"] == name)]
        require(len(subset) > 0 and int(subset.iloc[0]["label"]) == label, f"Invalid smoke label: {name}")
        rows.append(subset.iloc[0].to_dict())
    dataset = ManifestDataset(cfg["dataset_root"], pd.DataFrame(rows), create_transforms(cfg, False))
    batch, labels = [], []
    for i in range(len(dataset)):
        image, label, image_id = dataset[i]
        require(tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE), f"Invalid tensor for ID {image_id}")
        batch.append(image); labels.append(label)
    require(labels == [0, 1], f"Smoke labels do not match class map: {labels}")
    tiny_cache = SharedImageCache([row["image_id"] for row in rows], cfg["image_size"])
    resize = transforms.Resize((cfg["image_size"],cfg["image_size"]), interpolation=InterpolationMode.BILINEAR)
    for index,row in enumerate(rows):
        with Image.open(Path(cfg["dataset_root"]) / row["image_path"]) as im:
            tiny_cache.pixels()[index] = np.asarray(resize(ImageOps.exif_transpose(im).convert("RGB")))
    for training in (False, True):
        direct = ManifestDataset(cfg["dataset_root"], pd.DataFrame(rows), create_transforms(cfg, training))
        cached = ManifestDataset(cfg["dataset_root"], pd.DataFrame(rows), create_transforms(cfg, training, True), tiny_cache)
        for index in range(len(rows)):
            seed_all(cfg["seed"]); a = direct[index][0]
            seed_all(cfg["seed"]); b = cached[index][0]
            require(torch.equal(a,b), "RAM and disk preprocessing differ")
    print("CACHE EQUIVALENCE PASS: identical tensors, including seeded online augmentation.", flush=True)
    inputs = torch.stack(batch).to(device)
    for name in MODEL_NAMES:
        model = create_model(name, cfg, pretrained=True).to(device).eval()
        with torch.inference_mode(): logits = model(inputs)
        require(tuple(logits.shape) == (2, 2), f"Invalid output dimension: {name}")
        require(torch.isfinite(logits).all().item(), f"Invalid logits: {name}")
        print(f"SMOKE PASS: {name}; ImageNet weights loaded; output dimension=2; Cocoa index=1", flush=True)
        del model, logits
        if device.type == "cuda": torch.cuda.empty_cache()
    print("Known-label smoke test passed: other_plants=0, cocoa=1.", flush=True)

def protocol_id(cfg, manifest_sha):
    """Identify training settings independently of machine-specific paths."""
    portable = {k: v for k, v in cfg.items() if k not in {"dataset_root", "output_root", "manifest", "device", "num_workers", "ram_cache"}}
    return sha(json.dumps([FORMAT_VERSION, portable, manifest_sha], sort_keys=True).encode())


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
        best_loss, best_epoch, best_state, wait = checkpoint["best_loss"], checkpoint["best_epoch"], checkpoint["best_state"], checkpoint["wait"]
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
                save_checkpoint(last_path, dict(info, epoch=epoch, model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                                history=history, best_loss=best_loss, best_epoch=best_epoch, best_state=best_state, wait=wait,
                                train_seconds=train_seconds, validation_seconds=validation_seconds,
                                checkpoint_seconds=checkpoint_seconds,
                                elapsed_seconds=previous_seconds + time.perf_counter()-started, rng=rng_state(generator)))
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


def write_comparison(run, results):
    columns = ["model", "epochs_run", "best_epoch", "best_val_loss", "accuracy", "balanced_accuracy", "macro_f1",
               "cocoa_precision", "cocoa_recall", "other_plants_fpr", "roc_auc_cocoa", "average_precision_cocoa", "train_seconds",
               "validation_seconds", "checkpoint_seconds", "run_seconds", "forward_ms_batch1", "parameters", "best_file_bytes"]
    table = pd.DataFrame(results)[columns].sort_values(["balanced_accuracy", "macro_f1", "cocoa_recall", "best_val_loss"], ascending=[False, False, False, True])
    save_csv(run / "comparison.csv", table)
    return table



def create_comparison_report(run):
    """Generate comparison figures from saved validation predictions and histories."""
    run = Path(run).resolve()
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    names = info["selected_models"]
    require(names and len(names) == len(set(names)) and set(names) <= SUPPORTED, "Invalid run model list")
    entries, reference = {}, None
    sources = {}
    for name in names:
        folder = run / name
        require((folder / "metrics.json").is_file(), f"Report waits for every model; unfinished: {name}")
        result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        require(result["status"] == "COMPLETED" and result["protocol"] == info["protocol"], f"Incomplete/mixed run: {name}")
        require(result["stage"] == "validation" and not result["test_evaluated"], f"Expected validation result: {name}")
        path = folder / "validation_predictions.csv"
        content = path.read_bytes()
        predictions = pd.read_csv(io.BytesIO(content), dtype={"image_id": str}, keep_default_na=False, encoding="utf-8-sig")
        require(len(predictions)>0 and predictions["image_id"].is_unique, f"Empty/duplicate predictions: {name}")
        predictions = predictions.sort_values("image_id").reset_index(drop=True)
        require(predictions["true_class_id"].isin([0,1]).all(), f"Invalid true labels: {name}")
        labels = predictions["true_class_id"].to_numpy(dtype=int)
        require(set(labels) == {0,1}, f"Missing validation class: {name}")
        probs = predictions[["p_other_plants", "p_cocoa"]].to_numpy(dtype=float)
        require(np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
                and np.allclose(probs.sum(1), 1, atol=1e-5), f"Invalid probabilities: {name}")
        identity = predictions[["image_id", "image_path", "true_class_id"]]
        if reference is None:
            reference = identity
        else:
            require(identity.equals(reference), f"Models did not evaluate the same validation images/labels: {name}")
        history = pd.read_csv(folder / "history.csv", encoding="utf-8-sig")
        require(len(history) == result["epochs_run"], f"Incomplete history: {name}")
        entries[name] = (result, predictions, labels, probs, history)
        sources[name] = sha(content)
    print(f"ALL {len(names)} MODELS COMPLETE: creating validation comparison figures from saved CSVs. No training or image reads.", flush=True)
    # Check all model outputs before generating figures.
    for name, (result, predictions, labels, probs, history) in entries.items():
        prediction_artifacts(run / name, "validation", predictions, labels, probs,
                             predictions["image_id"].tolist(), result["threshold"], save_tables=False)
        history_plots(run / name, history.to_dict("records"))
    folder = run / "publication_figures"
    folder.mkdir(exist_ok=True)
    ncols, nrows = min(2, len(names)), math.ceil(len(names)/2)
    for normalized in (False, True):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4*ncols, 4.6*nrows), squeeze=False)
        for ax, name in zip(axes.flat, names):
            result, predictions, labels, probs, _ = entries[name]
            predicted = np.where(probs[:,1]>=result["threshold"], 1, 0)
            cm = confusion_matrix(labels, predicted, labels=[0,1])
            matrix = cm.astype(float)/cm.sum(1,keepdims=True) if normalized else cm
            ConfusionMatrixDisplay(matrix, display_labels=NAMES).plot(ax=ax, colorbar=False, cmap="Blues", values_format=".3f" if normalized else "d")
            ax.set_title(name)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle("Validation confusion matrices" + (" (row-normalized)" if normalized else " (counts)"))
        fig.tight_layout(rect=(0,0,1,.96))
        fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
        fig.savefig(folder / ("confusion_matrices_normalized.png" if normalized else "confusion_matrices.png"), dpi=300)
        fig.savefig(folder / ("confusion_matrices_normalized.pdf" if normalized else "confusion_matrices.pdf"))
        plt.close(fig)
    for kind in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(7.3,5.5))
        for name in names:
            result, _, labels, probs, _ = entries[name]
            target = (labels==1).astype(int)
            if kind == "roc":
                x, y, _ = roc_curve(target, probs[:,1])
                label = f"{name}: AUC={roc_auc_score(target,probs[:,1]):.4f}"
            else:
                y, x, _ = precision_recall_curve(target, probs[:,1])
                label = f"{name}: AP={average_precision_score(target,probs[:,1]):.4f}"
            ax.plot(x,y,label=label)
        if kind == "roc":
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Validation ROC: cocoa is positive")
        else:
            ax.axhline(target.mean(),linestyle="--",color="gray",label="Class prevalence")
            ax.set(xlabel="Recall",ylabel="Precision",title="Validation Precision-Recall: cocoa is positive")
        ax.set(xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=9,loc="best")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
        fig.savefig(folder / f"{kind}_comparison.png",dpi=300)
        fig.savefig(folder / f"{kind}_comparison.pdf")
        plt.close(fig)
    for suffix, columns, ylabel in (("loss",["train_loss","val_loss"],"Cross-entropy loss"),
                                    ("accuracy",["train_accuracy","val_accuracy"],"Accuracy")):
        fig, axes = plt.subplots(nrows,ncols,figsize=(6*ncols,4*nrows),squeeze=False)
        for ax,name in zip(axes.flat,names):
            history = entries[name][4]
            for column in columns:
                ax.plot(history.epoch,history[column],label=column)
            ax.axvline(entries[name][0]["best_epoch"],color="gray",linestyle=":",label="Best epoch")
            ax.set(title=name,xlabel="Epoch",ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle(f"Training and validation {suffix}")
        fig.tight_layout(rect=(0,0,1,.96))
        fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
        fig.savefig(folder / f"{suffix}_comparison.png",dpi=300)
        fig.savefig(folder / f"{suffix}_comparison.pdf")
        plt.close(fig)
    table = write_comparison(run,[entries[name][0] for name in names])
    ordered = table.set_index("model").loc[names]
    fig,axes = plt.subplots(2,3,figsize=(16,8))
    for ax,key,title in zip(axes.flat,["accuracy","balanced_accuracy","macro_f1","cocoa_recall","other_plants_fpr","cocoa_precision"],
                           ["Accuracy","Balanced accuracy (higher is better)","Macro F1 (higher is better)",
                            "Cocoa recall (higher is better)","Other Plants FPR (lower is better)", "Cocoa precision"]):
        bars=ax.bar(names,ordered[key].to_numpy())
        ax.set(title=title,ylim=(0,1.12))
        ax.tick_params(axis="x",rotation=20)
        ax.bar_label(bars,fmt="%.4f",fontsize=8,padding=2)
    fig.tight_layout()
    fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
    fig.savefig(folder / "metrics_comparison.png",dpi=300)
    fig.savefig(folder / "metrics_comparison.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    x=np.arange(len(names))
    axes[0].bar(x-.18,ordered.train_seconds.to_numpy()/60,width=.36,label="Train")
    axes[0].bar(x+.18,ordered.validation_seconds.to_numpy()/60,width=.36,label="Validation")
    axes[0].set_xticks(x,names,rotation=20)
    axes[0].set(title="Total measured time (actual epochs run)",ylabel="Minutes")
    axes[0].legend()
    axes[1].bar(names,ordered.accuracy.to_numpy())
    axes[1].tick_params(axis="x",rotation=20)
    axes[1].set(title="Validation accuracy",ylabel="Accuracy")
    fig.tight_layout()
    fig.text(0.5, 0.002, "EVALUATION SPLIT: VALIDATION", ha="center", fontsize=8)
    fig.savefig(folder / "time_comparison.png",dpi=300)
    fig.savefig(folder / "time_comparison.pdf")
    plt.close(fig)
    eligible = table.loc[table.other_plants_fpr.le(0.01)].sort_values(
        ["cocoa_recall", "other_plants_fpr", "balanced_accuracy", "macro_f1", "best_val_loss"],
        ascending=[False, True, False, False, True])
    candidate, selection_status = None, "NO_MODEL_MEETS_FPR_REQUIREMENT"
    if len(eligible):
        top = eligible.iloc[0]
        tied = eligible.loc[(eligible.cocoa_recall-top.cocoa_recall).abs().le(1/894) &
                            (eligible.other_plants_fpr-top.other_plants_fpr).abs().le(1/1341) &
                            (eligible.balanced_accuracy-top.balanced_accuracy).abs().le(0.001) &
                            (eligible.macro_f1-top.macro_f1).abs().le(0.001)]
        selection_status = "NOT_CLEARLY_DISTINGUISHABLE" if len(tied)>1 else "WINNER_CANDIDATE"
        if len(tied)==1: candidate = str(top["model"])
    selection = dict(evaluation_split="VALIDATION", threshold=0.5, fpr_limit=0.01,
                     candidate=candidate, status=selection_status,
                     note="Descriptive selection only; near ties within one class error and 0.001 BA/F1 are unresolved. No significance claim.",
                     test_evaluated=False)
    save_json(run / "winner_candidate.json", selection)
    save_json(run / "report_summary.json", selection)
    print("WINNER SELECTION:", selection_status, candidate, flush=True)
    save_json(folder / "report_summary.json",dict(status="COMPLETED",models=names,stage="validation",
              source_predictions_sha256=sources,protocol=info["protocol"],created_utc=stamp(),
              report_script_sha256=sha(Path(__file__).read_bytes()),test_evaluated=False))
    print("COMPARISON FIGURES:",folder,flush=True)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "cache", "train", "report", "test"], default="check")
    parser.add_argument("--dataset-root")
    parser.add_argument("--ram-cache", action="store_true", help="Share pre-resized train/val images in RAM across models and workers")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader worker count for this execution")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "report":
        require(args.resume is not None, "Report requires --resume <completed run folder>")
        create_comparison_report(args.resume)
        return 0
    cfg = load_config(args.dataset_root, args.output_root)
    if args.num_workers is not None:
        require(args.num_workers >= 0, "--num-workers must be non-negative")
        cfg["num_workers"] = args.num_workers
    if args.ram_cache:
        cfg["ram_cache"] = True
    frame, manifest_sha, manifest_path = read_manifest(cfg)
    counts = frame.groupby(["split", "class_name"]).size().rename("images").reset_index()
    print(counts.to_string(index=False), flush=True)
    print("CLASS MAPPING:", CLASS_MAP, "| MANIFEST:", manifest_path, flush=True)
    selected = args.models or cfg["models"]
    require(len(selected) == len(set(selected)) and set(selected) <= set(cfg["models"]), "--models must be a unique subset of settings.models")
    require(set(selected) == set(MODEL_NAMES), "The initial benchmark requires all four models")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"])
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    if args.stage == "cache":
        require(args.cache_root is not None, "Cache requires --cache-root /content/model1_dataset")
        cache_training_images(cfg, frame, manifest_path, args.cache_root)
        return 0
    if args.stage == "check":
        check_files(cfg, frame, ["train", "val"] if (Path(cfg["dataset_root"]) / "_cache_ready.json").exists() else ["train", "val", "test"])
        smoke_test(cfg, frame, device)
        receipt = dict(status="PASSED", manifest_sha256=manifest_sha, protocol=protocol_id(cfg, manifest_sha),
                       script_sha256=sha(Path(__file__).read_bytes()), class_mapping=CLASS_MAP,
                       counts=counts.to_dict("records"), pretrained_weights=PRETRAINED_WEIGHTS,
                       cache_equivalence="known-label samples, exact tensors with seeded augmentation",
                       duplicate_scope="metadata IDs, paths, supplied hashes and groups only; no exhaustive pixel audit",
                       test_evaluated=False, training_started=False, created_utc=stamp())
        save_json(Path(cfg["output_root"]) / "preflight" / (protocol_id(cfg,manifest_sha)+".json"), receipt)
        print("PREFLIGHT PASSED: no training or test evaluation.", flush=True)
        return 0
    output = Path(cfg["output_root"])
    if args.stage == "test":
        raise RuntimeError("Test stage is reserved and disabled for the initial benchmark")
    require(args.checkpoint is None, "Training uses --resume <run folder>, not --checkpoint")
    receipt_path = Path(cfg["output_root"]) / "preflight" / (protocol_id(cfg,manifest_sha)+".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    preflight_valid = receipt.get("manifest_sha256")==manifest_sha and receipt.get("script_sha256")==sha(Path(__file__).read_bytes())
    if not args.resume and not preflight_valid:
        check_files(cfg, frame, ["train", "val"])
        smoke_test(cfg, frame, device)
    protocol = protocol_id(cfg, manifest_sha)
    if args.resume:
        run = args.resume.resolve()
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        require(info["protocol"] == protocol, "Settings or manifest changed; start a new run instead of resuming")
    else:
        run = output / ("run_" + stamp() + "_" + uuid.uuid4().hex[:6])
        run.mkdir(parents=True, exist_ok=False)
        save_json(run / "run_info.json", dict(format_version=FORMAT_VERSION, protocol=protocol, manifest_sha256=manifest_sha,
                  class_mapping=CLASS_MAP, seed=cfg["seed"], selected_models=selected, created_utc=stamp(),
                  script_sha256=sha(Path(__file__).read_bytes()), loader_workers=cfg["num_workers"], ram_cache=cfg["ram_cache"],
                  device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  versions=dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__), numpy=np.__version__,
                                pandas=pd.__version__, sklearn=sklearn.__version__, pillow=PIL.__version__),
                  selection="minimum validation cross-entropy; test is not evaluated",
                  timing="train/validation include loader and preprocessing; CUDA synchronized; checkpoint I/O recorded separately",
                  reproducibility="One seed; bitwise identity across hardware/library versions is not guaranteed"))
        save_json(run / "config.json", cfg)
        save_json(run / "class_mapping.json", CLASS_MAP)
        save_json(run / "dataset_identity.json", dict(manifest_sha256=manifest_sha, rows=len(frame), class_mapping=CLASS_MAP))
        save_csv(run / "dataset_summary.csv", counts)
        atomic_bytes(run / "code_snapshot.py", Path(__file__).read_bytes())
        save_csv(run / "split_counts.csv", counts)
    print("DEVICE:", device, "| RUN DIRECTORY:", run, flush=True)
    print("RESUME: add --resume", str(run), flush=True)
    print(f"DataLoader workers: {cfg['num_workers']} | batch size: {cfg['batch_size']}", flush=True)
    print("TRAIN/VAL ONLY. AdamW, all layers trainable, no scheduler, no class reweighting. Best epoch uses val_loss.", flush=True)
    results = []
    image_cache = None
    try:
        pending = [name for name in selected if not (run / name / "metrics.json").is_file()]
        if cfg["ram_cache"] and pending:
            image_cache = build_shared_image_cache(cfg, frame)
        preparation = image_cache.statistics if image_cache is not None else dict(images=0, bytes=0, seconds=0.0)
        save_json(run / f"input_preparation_{stamp()}_{uuid.uuid4().hex[:6]}.json",
                  dict(preparation, ram_cache=image_cache is not None, requested_ram_cache=cfg["ram_cache"],
                       loader_workers=cfg["num_workers"], dataset_root=cfg["dataset_root"], manifest_sha256=manifest_sha,
                       script_sha256=sha(Path(__file__).read_bytes()), created_utc=stamp(),
                       timing="Shared preparation time is excluded from per-model train/validation time"))
        for name in selected:
            results.append(train_model(name, cfg, frame, manifest_sha, protocol, run, device, image_cache))
            write_comparison(run, results)
    except KeyboardInterrupt:
        print("INTERRUPTED. Resume the same run directory; up to checkpoint_every-1 completed epochs may repeat.", flush=True)
        return 130
    image_cache = None
    gc.collect()
    table = write_comparison(run, results)
    expected = json.loads((run / "run_info.json").read_text(encoding="utf-8"))["selected_models"]
    missing = [name for name in expected if not (run / name / "metrics.json").is_file()]
    if not missing:
        create_comparison_report(run)
    else:
        print("Comparison figures deferred; unfinished models:", ", ".join(missing), flush=True)
    save_json(run / "experiment_summary.json", dict(status="PARTIAL" if missing else "COMPLETED", models=expected, seed=cfg["seed"],
              manifest_sha256=manifest_sha, test_evaluated=False, class_mapping=CLASS_MAP))
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print("RESULTS:", run, "\nFilesystem completion does not certify Google Drive cloud synchronization.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

