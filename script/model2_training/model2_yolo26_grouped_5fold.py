"""Run grouped five-fold cross-validation of the YOLO26n baseline."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)

def run(args):
    import numpy as np
    import pandas as pd
    import torch
    import ultralytics
    import yaml
    from sklearn.model_selection import StratifiedGroupKFold
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths

    project = args.project.resolve()
    baseline = args.baseline_run.resolve()
    ref = json.loads((baseline / "protocol.json").read_text())
    manifest = project / "dataset/model2_metadata_detection.csv"
    if sha(manifest) != ref["manifest_sha256"] or sha(args.data) != ref["dataset_yaml_sha256"]:
        raise RuntimeError("Use the original full baseline cache and manifest.")
    if not torch.cuda.is_available() or ultralytics.__version__ != ref["ultralytics"]:
        raise RuntimeError(f"Use CUDA and Ultralytics {ref['ultralytics']}.")
    names = ref["class_names"]
    config = {**ref["config"], "cls_pw": 0.0}
    base = yaml.safe_load(args.data.read_text())
    root = args.data.parent / Path(base.get("path") or args.data.parent)
    aliases = defaultdict(list)
    for split in ("train", "val"):
        entries = base[split]
        for entry in entries if isinstance(entries, list) else [entries]:
            path = root / entry
            paths = (sorted(p for p in path.rglob("*") if p.is_file() and
                            p.suffix.lower().lstrip(".") in IMG_FORMATS)
                     if path.is_dir() else
                     [path.parent / line.strip() for line in path.read_text().splitlines() if line.strip()])
            for image in paths:
                original = image.resolve(strict=True)
                aliases[original.stem].append((original, image))

    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pool = sorted((r for r in rows if r["split"] in ("train", "val")), key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in pool]
    groups = [r["split_group_id"].strip() for r in pool]
    test_groups = {r["split_group_id"].strip() for r in rows if r["split"] == "test"}
    if len(ids) != 8041 or len(set(ids)) != len(ids) or set(ids) != set(aliases):
        raise RuntimeError("Expected the 8,041 baseline train/validation images.")
    if not all(groups) or set(groups) & test_groups:
        raise RuntimeError("Missing group IDs or groups overlapping the held-out test split.")

    classes = [json.loads(r["yolo_class_ids"]) for r in pool]
    counts = np.bincount([c for cs in classes for c in cs], minlength=len(names))
    strata = [min(cs, key=lambda c: (counts[c], c)) for cs in classes]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(ids, strata, groups))
    output = project / "results/model2_yolo26_grouped_5fold"
    cache = Path("/content/model2_yolo26_grouped_5fold")
    assignments = []
    prepared = {}
    for number, (train, val) in enumerate(folds, 1):
        if {groups[i] for i in train} & {groups[i] for i in val}:
            raise RuntimeError("Cross-fold group leakage.")
        for indices in (train, val):
            if {c for i in indices for c in classes[i]} != set(range(len(names))):
                raise RuntimeError(f"Fold {number} lacks a class in train or validation.")
        assignments.extend(dict(image_id=ids[i], group=groups[i], validation_fold=number) for i in val)
        prepared[number] = (train, val)

    output.mkdir(parents=True, exist_ok=True)
    identity = dict(manifest_sha256=sha(manifest), dataset_yaml_sha256=sha(args.data),
                    ultralytics=ultralytics.__version__, torch=str(torch.__version__),
                    config=config, assignments=assignments,
                    sampling="Three entries per pod-borer training image")
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != identity:
        raise RuntimeError("Saved cross-validation protocol differs.")
    save(protocol, identity)
    pd.DataFrame(assignments).to_csv(output / "fold_assignments.csv", index=False)
    if args.plan_only:
        for number, (train, val) in prepared.items():
            print(f"Fold {number}: train={len(train)}, val={len(val)}")
        return

    for number, (train, val) in prepared.items():
        if args.fold and args.fold != number:
            continue
        folder = output / f"fold_{number}"
        done = folder / "metrics.json"
        best = folder / "yolo26n/weights/best.pt"
        last = folder / "yolo26n/weights/last.pt"
        finished = folder / "training_finished.json"
        if done.exists():
            if json.loads(done.read_text())["best_sha256"] != sha(best):
                raise RuntimeError(f"Checkpoint hash mismatch in fold {number}.")
            continue

        view = cache / f"fold_{number}"
        for split, indices in (("train", train), ("val", val)):
            image_dir, label_dir = view / f"images/{split}", view / f"labels/{split}"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i in indices:
                original, alias = next(
                    ((p, a) for p, a in aliases[ids[i]] if a == p),
                    aliases[ids[i]][0],
                )
                label = Path(img2label_paths([str(alias)])[0]).resolve(strict=True)
                repeats = 3 if split == "train" and names.index("pod-borer") in classes[i] else 1
                for repeat in range(repeats):
                    stem = f"{ids[i]}_r{repeat}"
                    image_link = image_dir / (stem + original.suffix)
                    label_link = label_dir / (stem + ".txt")
                    for link, target in ((image_link, original), (label_link, label)):
                        if link.is_symlink():
                            if link.resolve(strict=True) != target:
                                raise RuntimeError(f"Unexpected existing alias: {link}")
                        else:
                            link.symlink_to(target)
                    paths.append(str(image_link))
            (view / f"{split}.txt").write_text("\n".join(paths) + "\n")
        data = view / "data.yaml"
        data.write_text(yaml.safe_dump(dict(path=str(view), train="train.txt",
                              val="val.txt", names=dict(enumerate(names)))))
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Fold {number}: {len(train)} unique train, {len(val)} validation", flush=True)
        if not finished.exists():
            model = YOLO(str(last) if last.exists() else "yolo26n.pt")
            def finish(trainer):
                save(finished, {"epoch": int(trainer.epoch + 1), "best_sha256": sha(best)})
            model.add_callback("on_train_end", finish)
            if last.exists():
                model.train(resume=True, device=0)
            else:
                model.train(data=str(data), project=str(folder), name="yolo26n",
                            exist_ok=True, device=0, **config)
            del model
            torch.cuda.empty_cache()
        if json.loads(finished.read_text())["best_sha256"] != sha(best):
            raise RuntimeError("Best checkpoint changed after training.")
        m = YOLO(str(best)).val(data=str(data), split="val", imgsz=640, batch=32,
                               workers=2, conf=.001, iou=.7, device=0,
                               project=str(folder), name="validation", exist_ok=True,
                               plots=True, save_json=True)
        class_rows = []
        for index, cls in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = map(float, m.box.class_result(index))
            cls = int(cls)
            class_rows.append(dict(class_id=cls, class_name=names[cls], precision=p,
                                   recall=r, ap50=ap50, ap50_95=ap,
                                   validation_images=sum(cls in classes[i] for i in val)))
        pd.DataFrame(class_rows).to_csv(folder / "per_class.csv", index=False)
        save(done, dict(fold=number, split="CV_VALIDATION", map50_95=float(m.box.map),
                        map50=float(m.box.map50), precision=float(m.box.mp),
                        recall=float(m.box.mr), best_sha256=sha(best)))

    reports = sorted(output.glob("fold_*/metrics.json"))
    summary = pd.DataFrame([json.loads(p.read_text()) for p in reports])
    summary.to_csv(output / "fold_metrics.csv", index=False)
    print(summary.drop(columns="best_sha256").to_string(index=False))
    if len(reports) == 5:
        columns = ["map50_95", "map50", "precision", "recall"]
        summary[columns].agg(["mean", "std"]).to_csv(output / "summary_mean_sd.csv")
        classes_all = pd.concat([pd.read_csv(p.parent / "per_class.csv") for p in reports])
        classes_all.groupby("class_name")[["ap50", "ap50_95", "precision", "recall"]].agg(
            ["count", "mean", "std"]
        ).to_csv(output / "per_class_mean_sd.csv")
        print(summary[columns].agg(["mean", "std"]).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--baseline-run", required=True, type=Path,
                        help="Original YOLO26n baseline run containing protocol.json.")
    args = parser.parse_args()
    import fcntl  # Linux/Colab runtime lock.
    with Path("/tmp/cocoa_yolo26_cv.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
