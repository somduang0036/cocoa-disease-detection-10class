"""Generate validation-only Grad-CAM evidence from completed Model 1 checkpoints."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse
import hashlib
import json
import uuid
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

CLASS_MAP = {"other_plants": 0, "cocoa": 1}
NAMES = ["other_plants", "cocoa"]
MODEL_NAMES = ["mobilenet_v2", "resnet50", "convnext_tiny", "efficientnet_b0"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def child(root, relative):
    value = str(relative).replace("\\", "/")
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts and ":" not in value,
            f"Unsafe relative image path: {value}")
    resolved = (root / value).resolve()
    require(root.resolve() in resolved.parents, f"Image path escapes dataset: {value}")
    return resolved


def load_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image.resize((224, 224), resample=Image.Resampling.BILINEAR)


def build_model(name):
    model = models.get_model(name, weights=None)
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, 2)
        layer, layer_name = model.layer4[-1], "layer4[-1]"
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        layer, layer_name = model.features[-1], "features[-1]"
    return model, layer, layer_name


def gradcam(model, layer, image, target, device):
    """Compute ReLU Grad-CAM for a selected pre-softmax class logit."""
    activation = []
    hook = layer.register_forward_hook(lambda module, inputs, output: activation.append(output))
    try:
        tensor = transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image))
        tensor = tensor.unsqueeze(0).to(device).requires_grad_(True)
        # Evaluation mode disables dropout and prevents BatchNorm updates.
        with torch.enable_grad():
            logits = model(tensor)
            require(tuple(logits.shape) == (1, 2), "Expected two model outputs")
            features = activation[0]
            gradient, = torch.autograd.grad(logits[0, target], features)
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * features).sum(dim=1, keepdim=True))
            heatmap = F.interpolate(heatmap, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            peak = heatmap.max()
            valid = bool(torch.isfinite(heatmap).all() and peak > 0)
            heatmap = heatmap / peak if valid else torch.zeros_like(heatmap)
        return heatmap.detach().cpu().numpy(), logits.detach().softmax(1)[0].cpu().numpy(), valid
    finally:
        hook.remove()


def overlay(image, heatmap):
    rgb = np.asarray(image, dtype=np.float32) / 255
    color = plt.get_cmap("jet")(heatmap)[..., :3]
    # Zero positive attribution leaves the original image visible.
    alpha = 0.45 * (heatmap > 0)[..., None]
    return np.clip((1-alpha)*rgb + alpha*color, 0, 1)


def save_figure(fig, destination):
    fig.suptitle("Grad-CAM | EVALUATION SPLIT: VALIDATION", fontsize=13)
    fig.text(.5, .008, "Maps are normalized individually; color is relative attribution, not calibrated confidence.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .035, 1, .95])
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    require(args.samples_per_class > 0, "samples-per-class must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run, root = args.run.resolve(), args.dataset_root.resolve()
    config, info = read_json(run/"config.json"), read_json(run/"run_info.json")
    require(info["class_mapping"] == CLASS_MAP, "Run class mapping differs")
    require(config["image_size"] == 224 and config["threshold"] == .5, "Expected the fixed 224-pixel benchmark")
    require(set(info["selected_models"]) == set(MODEL_NAMES), "Expected four completed models")
    manifest_path = root / config["manifest"]
    require(digest(manifest_path) == info["manifest_sha256"], "Dataset manifest identity differs")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    require(manifest.image_id.is_unique, "Duplicate manifest image IDs")
    val = manifest.loc[manifest.split.eq("val")].copy().set_index("image_id")
    require(set(val.class_name) == set(CLASS_MAP), "Validation classes differ")
    require(val.class_id.astype(int).eq(val.class_name.map(CLASS_MAP)).all(), "Validation labels differ")
    predictions, artifacts = {}, {}
    for name in MODEL_NAMES:
        folder = run/name
        metrics = read_json(folder/"metrics.json")
        require(metrics["status"] == "COMPLETED" and metrics["stage"] == "validation" and
                metrics["protocol"] == info["protocol"] and metrics["class_mapping"] == CLASS_MAP,
                f"Incompatible model result: {name}")
        path = folder/"validation_predictions.csv"
        frame = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False).set_index("image_id")
        require(frame.index.is_unique and set(frame.index) == set(val.index), f"Validation IDs differ: {name}")
        frame = frame.loc[val.index]
        require(frame.true_class_id.astype(int).eq(val.class_id.astype(int)).all(), f"Validation labels differ: {name}")
        require(frame.image_path.eq(val.image_path).all(), f"Validation paths differ: {name}")
        require(frame.threshold.eq(.5).all(), "Saved threshold differs")
        require(np.array_equal(frame.predicted_class_id, (frame.p_cocoa >= .5).astype(int)),
                f"Prediction mapping differs: {name}")
        predictions[name] = frame
        artifacts[name] = dict(predictions_sha256=digest(path))
    # Use the same seeded examples for every model; record this selection rule.
    correct = pd.Series(True, index=val.index)
    for frame in predictions.values():
        correct &= frame.predicted_class_id.eq(frame.true_class_id)
    selection = []
    for class_name, class_id in CLASS_MAP.items():
        pool = val.loc[correct & val.class_id.astype(int).eq(class_id)].sort_index()
        require(len(pool) >= args.samples_per_class, f"Too few jointly correct examples: {class_name}")
        for image_id in pool.sample(args.samples_per_class, random_state=args.seed).index:
            selection.append(dict(image_id=image_id, selection="jointly_correct", true_class_id=class_id))
    winner_file = run/"winner_candidate.json"
    winner = read_json(winner_file).get("candidate") if winner_file.exists() else None
    if winner is None:
        print("No distinct winner candidate; error analysis will be skipped.", flush=True)
    require(winner is None or winner in MODEL_NAMES, "Invalid winner architecture")
    errors = []
    if winner:
        f = predictions[winner]
        errors = [dict(image_id=i, selection="winner_error", true_class_id=int(f.loc[i,"true_class_id"]))
                  for i in sorted(f.index[f.predicted_class_id.ne(f.true_class_id)])]
    selected = pd.DataFrame(selection + errors)
    for image_id in selected.image_id.unique():
        require(child(root, val.loc[image_id,"image_path"]).is_file(), f"Image unavailable: {image_id}")
    output = run / ("xai_gradcam_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    output.mkdir(exist_ok=False)
    selected.to_csv(output/"selected_images.csv", index=False)
    records, maps, images = [], {}, {}
    for name in MODEL_NAMES:
        checkpoint_path = run/name/"best.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(checkpoint["architecture"] == name and checkpoint["protocol"] == info["protocol"] and
                checkpoint["manifest_sha256"] == info["manifest_sha256"] and
                checkpoint["class_mapping"] == CLASS_MAP and checkpoint["threshold"] == .5,
                f"Checkpoint identity differs: {name}")
        require(checkpoint["preprocessing"] == dict(image_size=224, resize="stretch", interpolation="bilinear",
                mean=MEAN, std=STD, color="RGB", exif="upright"), "Checkpoint preprocessing differs")
        model, layer, layer_name = build_model(name)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.to(device).eval().requires_grad_(False)
        artifacts[name].update(checkpoint_sha256=digest(checkpoint_path), target_layer=layer_name)
        rows = selection + (errors if name == winner else [])
        for item in rows:
            image_id, true_id = item["image_id"], item["true_class_id"]
            saved = predictions[name].loc[image_id]
            predicted_id = int(saved.predicted_class_id)
            targets = sorted({predicted_id, true_id})
            image = load_image(child(root, val.loc[image_id,"image_path"]))
            images[image_id] = image
            for target in targets:
                heatmap, probs, valid = gradcam(model, layer, image, target, device)
                maps[(name,image_id,target)] = heatmap
                stem = f"{name}_{image_id}_target{target}"
                np.save(output/(stem+".npy"), heatmap)
                records.append(dict(model=name,image_id=image_id,image_path=val.loc[image_id,"image_path"],
                    selection=item["selection"],true_class_id=true_id,predicted_class_id=predicted_id,
                    target_class_id=target,target_class_name=NAMES[target],p_cocoa_saved=float(saved.p_cocoa),
                    p_cocoa_gradcam_float32=float(probs[1]),prediction_matches_saved=int(probs[1]>=.5)==predicted_id,
                    nonzero_positive_attribution=valid,target_layer=layer_name,heatmap_file=stem+".npy"))
            print(f"GRAD-CAM: {name} | ID {image_id} | true={NAMES[true_id]} | predicted={NAMES[predicted_id]}",flush=True)
        del model
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    for sample_index in range(args.samples_per_class):
        pair = [next(x for j,x in enumerate([s for s in selection if s["true_class_id"]==c])
                     if j==sample_index) for c in (0,1)]
        fig, axes = plt.subplots(4,4,figsize=(12,12),squeeze=False)
        for r,name in enumerate(MODEL_NAMES):
            for c,item in enumerate(pair):
                image_id, target = item["image_id"], item["true_class_id"]
                saved = predictions[name].loc[image_id]
                axes[r,2*c].imshow(images[image_id])
                axes[r,2*c].set_title(f"{name}\nTrue: {NAMES[target]} | ID {image_id}",fontsize=9)
                axes[r,2*c+1].imshow(overlay(images[image_id],maps[(name,image_id,target)]))
                axes[r,2*c+1].set_title(f"Target: {NAMES[target]}\nP(Cocoa)={saved.p_cocoa:.6f}",fontsize=9)
        for ax in axes.flat: ax.axis("off")
        save_figure(fig,output/f"four_model_comparison_{sample_index+1}")
    for start in range(0,len(errors),4):
        page = errors[start:start+4]
        fig, axes = plt.subplots(len(page),3,figsize=(10,3.4*len(page)),squeeze=False)
        for r,item in enumerate(page):
            image_id,true_id=item["image_id"],item["true_class_id"]
            predicted_id=int(predictions[winner].loc[image_id,"predicted_class_id"])
            axes[r,0].imshow(images[image_id])
            axes[r,0].set_title(f"ID {image_id} | True: {NAMES[true_id]}",fontsize=9)
            for c,target in enumerate((predicted_id,true_id),1):
                axes[r,c].imshow(overlay(images[image_id], maps[(winner,image_id,target)]))
                axes[r,c].set_title(("Predicted" if c==1 else "True")+" class attribution: "+NAMES[target],fontsize=9)
        for ax in axes.flat: ax.axis("off")
        save_figure(fig,output/f"winner_errors_page_{start//4+1}")
    pd.DataFrame(records).to_csv(output/"gradcam_records.csv",index=False)
    summary=dict(status="COMPLETED",evaluation_split="VALIDATION",class_mapping=CLASS_MAP,seed=args.seed,
        manifest_sha256=info["manifest_sha256"],artifacts=artifacts,winner=winner,winner_errors=len(errors),
        selection_rule="Seeded random selection from samples correctly classified by all four models; all winner errors included.",
        method="Grad-CAM: spatial mean gradient weights, channel sum, ReLU, bilinear upsampling, per-map max normalization.",
        target="Pre-softmax class logit; predicted and true classes are both explained for winner errors.",
        preprocessing="EXIF upright, RGB, bilinear stretch 224x224, ImageNet normalization; no random augmentation.",
        limitations=["Selected correct examples are illustrative and not a representative accuracy estimate.",
        "Maps describe local relative attribution; no causal or localization accuracy claim.",
        "Float32 recomputation can differ slightly from saved mixed-precision predictions; discrepancies are recorded.",
        "A zero positive map is recorded explicitly and is not replaced with fabricated attribution."],
        training_performed=False,test_evaluated=False,device=str(device),
        versions=dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__)),
        script_sha256=digest(Path(__file__)))
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("GRAD-CAM COMPLETED:",output,flush=True)


if __name__ == "__main__":
    main()
