"""Generate detection-specific Grad-CAM for the locked YOLO26n checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--benchmark-run", required=True, type=Path)
    parser.add_argument("--image-id", nargs="+", help="Validation image IDs. Defaults to the first three distinct CSV IDs.")
    parser.add_argument("--errors", type=Path, help="CSV of matched wrong-class boxes from this run.")
    args = parser.parse_args()
    run = args.benchmark_run.resolve()
    review = run / "yolo26n/validation_error_review"
    errors = args.errors or review / "wrong_class_boxes.csv"
    cases = {}
    with errors.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            cases.setdefault(row["image_id"], row)
    image_ids = list(dict.fromkeys(args.image_id)) if args.image_id else list(cases)[:3]
    missing = set(image_ids) - cases.keys()
    if missing:
        raise ValueError(f"No audited wrong-class box for: {sorted(missing)}")
    if not image_ids:
        print("No matched wrong-class detections were recorded for this run.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from matplotlib.patches import Rectangle
    from ultralytics import YOLO, __version__

    project = args.project
    training = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model_dir = run / "yolo26n"
    complete = json.loads((model_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    if complete["identity"] != training["identity"]:
        raise RuntimeError("Checkpoint completion record does not belong to this run.")
    mapping = training["class_mapping"]
    protocol = {
        "class_names": [mapping[str(i)] for i in range(len(mapping))],
        "checkpoint_sha256": complete["best_sha256"],
        "ultralytics": training["ultralytics"],
    }
    checkpoint = model_dir / "weights/best.pt"
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != protocol["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint hash mismatch.")
    if __version__ != protocol["ultralytics"]:
        raise RuntimeError(f"Install Ultralytics {protocol['ultralytics']}.")
    dataset = project / "dataset"
    with (dataset / "model2_metadata_detection.csv").open(encoding="utf-8-sig") as stream:
        rows = {r["image_id"]: r for r in csv.DictReader(stream) if r["split"] == "val"}
    missing = set(image_ids) - rows.keys()
    if missing:
        raise ValueError(f"Not in validation metadata: {sorted(missing)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(checkpoint)).model.to(device).float().eval()
    model.requires_grad_(False)
    head = model.model[-1]
    end_to_end = bool(head.end2end)
    print("Detection head:", "one-to-one" if end_to_end else "one-to-many", flush=True)
    features = []

    def capture_head(module, inputs):
        features[:] = list(inputs[0])

    def capture_classification_branch(module, inputs):
        # The one-to-one head detaches these tensors. Preserve their values and
        # enable differentiation of the selected score with respect to them.
        feature = inputs[0].detach().requires_grad_(True)
        features.append(feature)
        return (feature, *inputs[1:])

    if end_to_end:
        towers = getattr(head, "one2one_cv3", None)
        if towers is None or len(towers) == 0:
            raise RuntimeError("Cannot locate the one-to-one classification branches.")
        handles = [tower.register_forward_pre_hook(capture_classification_branch) for tower in towers]
    else:
        handles = [head.register_forward_pre_hook(capture_head)]
    out = review / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    generated = 0

    try:
        for image_id in image_ids:
            source = dataset / rows[image_id]["image_path"]
            with Image.open(source) as image:
                original = ImageOps.exif_transpose(image).convert("RGB")
            width, height = original.size
            ratio = min(640 / width, 640 / height)
            w, h = round(width * ratio), round(height * ratio)
            left, top = (640-w)//2, (640-h)//2
            canvas = Image.new("RGB", (640, 640), (114,)*3)
            canvas.paste(original.resize((w, h), Image.Resampling.BILINEAR), (left, top))
            with torch.enable_grad():
                tensor = torch.from_numpy(np.array(canvas)).to(device).float()
                tensor = (tensor.permute(2, 0, 1)[None] / 255).requires_grad_(True)
                features.clear()
                prediction = model(tensor)
                detections = prediction[0] if isinstance(prediction, tuple) else prediction
                if not end_to_end:
                    nc = len(protocol["class_names"])
                    if detections.ndim != 3 or detections.shape[1] != 4 + nc:
                        raise RuntimeError(f"Unexpected one-to-many output: {detections.shape}")
                    raw = detections.transpose(1, 2)
                    coords = raw[..., :4]
                    if not getattr(head, "xyxy", False):
                        center, half = coords[..., :2], coords[..., 2:4] / 2
                        coords = torch.cat((center - half, center + half), dim=-1)
                    scores = raw[..., 4:]
                    confidence, classes = scores.max(dim=-1)
                    detections = torch.cat(
                        (coords, confidence.unsqueeze(-1), classes.unsqueeze(-1)), dim=-1
                    )
                if detections.ndim != 3 or detections.shape[-1] != 6:
                    raise RuntimeError(f"Unexpected detection shape: {detections.shape}")
                case = cases[image_id]
                x, y, bw, bh = json.loads(case["pred_bbox"])
                target = detections.new_tensor([
                    left+x*w/width, top+y*h/height,
                    left+(x+bw)*w/width, top+(y+bh)*h/height,
                ])
                candidates = detections[0]
                area = (candidates[:, 2:4]-candidates[:, :2]).clamp(min=0).prod(1)
                overlap = (torch.minimum(candidates[:, 2:4], target[2:])
                           - torch.maximum(candidates[:, :2], target[:2])).clamp(min=0).prod(1)
                iou = overlap / (area + (target[2:]-target[:2]).prod() - overlap).clamp(min=1e-8)
                cls_id = protocol["class_names"].index(case["predicted_class"])
                iou = iou.masked_fill((candidates[:, 5] != cls_id) | (candidates[:, 4] < .25), -1)
                index = int(iou.argmax())
                if float(iou[index].detach()) < .5:
                    print(f"Skipping {image_id}: audited prediction was not reproduced.")
                    continue
                box = detections[0, index]
                gradients = torch.autograd.grad(box[4], features, allow_unused=True)
                maps = []
                for feature, gradient in zip(features, gradients):
                    if gradient is not None:
                        cam = (feature * gradient.mean(dim=(2, 3), keepdim=True)).sum(1, keepdim=True).relu()
                        maps.append(F.interpolate(cam, (640, 640), mode="bilinear", align_corners=False))
                if not maps:
                    raise RuntimeError("No gradients reached the detection feature maps.")
                cam = torch.stack(maps).sum(0)[0, 0, top:top+h, left:left+w].detach().cpu().numpy()
            peak = float(cam.max())
            if peak > 0:
                cam /= peak
            x1, y1, x2, y2, confidence, cls = box.detach().cpu().tolist()
            x1, x2 = [min(width, max(0, (x-left)*width/w)) for x in (x1,x2)]
            y1, y2 = [min(height, max(0, (y-top)*height/h)) for y in (y1,y2)]
            name = protocol["class_names"][int(cls)]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            for ax in axes:
                ax.imshow(original, extent=(0,width,height,0))
                ax.axis("off")
            axes[0].set_title("Validation image")
            if case:
                x, y, bw, bh = json.loads(case["gt_bbox"])
                axes[0].add_patch(Rectangle((x,y), bw, bh, fill=False, edgecolor="lime", linewidth=2))
                axes[0].set_title(f"Ground truth | {case['true_class']}")
            axes[1].imshow(cam, cmap="jet", alpha=0.45*(cam > 0), extent=(0,width,height,0))
            axes[1].add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor="white", linewidth=2))
            axes[1].set_title(f"Grad-CAM | {name} | {confidence:.3f}")
            fig.suptitle(f"YOLO26n | Validation image {image_id}")
            for extension in ("png", "pdf"):
                fig.savefig(out / f"gradcam_{image_id}.{extension}", dpi=300)
            plt.close(fig)
            (out / f"gradcam_{image_id}.json").write_text(json.dumps(dict(
                image_id=image_id, split="validation", predicted_class=name,
                confidence=confidence, box_xyxy=[x1,y1,x2,y2],
                target=("matched wrong-class detection score" if case else "highest-confidence detection score"),
                true_class=case["true_class"] if case else None,
                audited_prediction_iou=float(iou[index].detach()) if case else None,
                feature_maps=("one-to-one classification branch inputs" if end_to_end else "detection head inputs"),
                benchmark_run=run.name,
                detection_head="one-to-one" if end_to_end else "one-to-many", positive_cam_peak=peak,
                checkpoint_sha256=protocol["checkpoint_sha256"],
            ), indent=2), encoding="utf-8")
            generated += 1
            print(f"{image_id}: {name} ({confidence:.3f})", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    print(f"Grad-CAM figures generated: {generated}/{len(image_ids)}")
    print("Results:", out)

if __name__ == "__main__":
    main()
