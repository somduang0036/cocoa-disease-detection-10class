"""Evaluate the validation-selected YOLO26n checkpoint on the held-out cocoa test split."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

from model2_training_script import FORMAT, MODELS, NAMES, ULTRALYTICS_VERSION

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--benchmark-run", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    dataset = project / "dataset"
    benchmark = args.benchmark_run.resolve()
    model_dir = benchmark / "yolo26n"
    best = model_dir / "weights/best.pt"
    manifest = dataset / "model2_metadata_detection.csv"
    annotations = dataset / "matadata_model 2/annotations_test.coco.json"
    output = project / "results/model2_yolo26_final_test" / benchmark.name
    config = dict(split="test", imgsz=640, batch=32, workers=2,
                  conf=0.001, iou=0.7, device=0)

    training = read(benchmark / "config.json")
    winner = read(benchmark / "winner_candidate.json")
    if winner.get("candidate") != "yolo26n":
        raise RuntimeError("This test script requires YOLO26n to be selected on validation.")
    names = [training["class_mapping"][str(index)] for index in range(len(NAMES))]
    if (names != NAMES or training["models"] != MODELS
            or training["ultralytics"] != ULTRALYTICS_VERSION):
        raise RuntimeError("Unexpected benchmark models, class mapping, or Ultralytics version.")

    hashes = dict(checkpoint_sha256=digest(best), manifest_sha256=digest(manifest),
                  annotations_sha256=digest(annotations))
    identity_inputs = dict(
        format=FORMAT, script_sha256=digest(benchmark / "code_snapshot.py"),
        manifest=hashes["manifest_sha256"],
        mapping=digest(dataset / "matadata_model 2/class_mapping.json"),
        config=training["protocol"], models=training["models"],
        pod_borer_exposures=training["pod_borer_exposures"],
    )
    identity = hashlib.sha256(json.dumps(identity_inputs, sort_keys=True).encode()).hexdigest()
    if identity != training["identity"]:
        raise RuntimeError("Dataset or benchmark identity does not match the training run.")

    scores = {}
    for name in MODELS:
        folder = benchmark / name
        metrics_path = folder / "validation_metrics.json"
        completion = read(folder / "COMPLETED.json")
        metrics = read(metrics_path)
        if (completion["identity"] != identity
                or completion["metrics_sha256"] != digest(metrics_path)
                or metrics["model"] != name
                or metrics["evaluation_split"] != "VALIDATION"):
            raise RuntimeError(f"Invalid validation record for {name}.")
        scores[name] = metrics["map50_95"]
    if scores["yolo26n"] < max(scores.values()):
        raise RuntimeError("YOLO26n does not have the highest validation mAP50-95.")
    if hashes["checkpoint_sha256"] != read(model_dir / "COMPLETED.json")["best_sha256"]:
        raise RuntimeError("YOLO26n checkpoint differs from the one evaluated on validation.")

    protocol = dict(**hashes, model="yolo26n", selected_run=benchmark.name,
                    benchmark_identity=identity, selection_split="VALIDATION",
                    selection_metric="mAP50-95", class_names=names,
                    ultralytics=training["ultralytics"], evaluation=config)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "locked_test_protocol.json"
    done = output / "COMPLETED.json"
    if lock.exists() and read(lock) != protocol:
        raise RuntimeError("Existing test protocol differs from the requested protocol.")
    if done.exists():
        if not lock.exists() or read(done) != hashes:
            raise RuntimeError("Completed test identity mismatch.")
        print((output / "test_metrics.json").read_text())
        print((output / "per_class_test.csv").read_text())
        return

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    if not torch.cuda.is_available() or ultralytics.__version__ != training["ultralytics"]:
        raise RuntimeError(f"Use CUDA and Ultralytics {training['ultralytics']}.")
    save(lock, protocol)
    print("Locked checkpoint:", best, flush=True)
    print("Checkpoint SHA-256:", hashes["checkpoint_sha256"], flush=True)

    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    coco = read(annotations)
    images = {str(image["id"]): image for image in coco["images"]}
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    if (len(rows) != 894 or len(images) != 894
            or {row["image_id"] for row in rows} != images.keys()
            or len(coco["annotations"]) != 1270
            or len(names) != 10 or set(categories.values()) != set(names)):
        raise RuntimeError("Unexpected test split or class mapping.")
    labels = defaultdict(list)
    boxes = Counter()
    support = defaultdict(set)
    for annotation in coco["annotations"]:
        image_id = str(annotation["image_id"])
        width, height = images[image_id]["width"], images[image_id]["height"]
        x, y, w, h = annotation["bbox"]
        if not (w > 0 and h > 0 and x >= 0 and y >= 0
                and x + w <= width + 1e-5 and y + h <= height + 1e-5):
            raise RuntimeError(f"Invalid annotation for image {image_id}")
        cls = names.index(categories[annotation["category_id"]])
        coords = ((x+w/2)/width, (y+h/2)/height, w/width, h/height)
        labels[image_id].append(f"{cls} " + " ".join(f"{v:.10f}" for v in coords))
        boxes[cls] += 1
        support[cls].add(image_id)

    cache = Path("/content/model2_yolo26_locked_test") / hashes["annotations_sha256"][:16]
    image_dir, label_dir = cache / "images/test", cache / "labels/test"
    for folder in (image_dir, label_dir):
        folder.mkdir(parents=True, exist_ok=True)

    def prepare(row):
        image_id = row["image_id"]
        source = dataset / row["image_path"]
        destination = image_dir / (image_id + source.suffix.lower())
        image = images[image_id]
        if ((int(row["width_px"]), int(row["height_px"])) != (image["width"], image["height"])
                or len(labels[image_id]) != int(row["bbox_count"])):
            raise RuntimeError(f"Metadata and annotations disagree for {image_id}")
        expected = row["output_sha256"]
        if not destination.exists() or digest(destination) != expected:
            temporary = destination.with_suffix(".part")
            shutil.copyfile(source, temporary)
            if digest(temporary) != expected:
                raise RuntimeError(f"Image hash mismatch: {source}")
            temporary.replace(destination)
        (label_dir / f"{image_id}.txt").write_text("\n".join(labels[image_id]) + "\n")
        return str(destination)

    paths = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path in pool.map(prepare, rows):
            paths.append(path)
            if len(paths) % 200 == 0 or len(paths) == 894:
                print(f"Prepared {len(paths)}/894 test images", flush=True)
    (cache / "test.txt").write_text("\n".join(paths) + "\n")
    data = cache / "data.yaml"
    data.write_text(yaml.safe_dump(dict(path=str(cache), train="test.txt",
                    val="test.txt", test="test.txt", names=dict(enumerate(names)))))

    model = YOLO(str(best))
    if model.names != dict(enumerate(names)):
        raise RuntimeError("Checkpoint class mapping differs from the benchmark.")
    metrics = model.val(data=str(data), project=str(output),
                                name="test_evidence", exist_ok=True,
                                plots=True, save_json=True, **config)
    result = dict(model="yolo26n", evaluation_split="TEST", images=len(paths),
                  ground_truth_boxes=sum(boxes.values()), mAP50_95=float(metrics.box.map),
                  mAP50=float(metrics.box.map50), precision=float(metrics.box.mp),
                  recall=float(metrics.box.mr))
    positions = {int(cls): i for i, cls in enumerate(metrics.box.ap_class_index)}
    with (output / "per_class_test.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_name", "test_images", "ground_truth_boxes",
                         "precision", "recall", "AP50", "AP50_95"])
        for cls, name in enumerate(names):
            values = metrics.box.class_result(positions[cls]) if cls in positions else [""] * 4
            writer.writerow([cls, name, len(support[cls]), boxes[cls], *values])
    save(output / "test_metrics.json", result)
    save(done, hashes)
    print(json.dumps(result, indent=2))
    print((output / "per_class_test.csv").read_text())
    print("Results directory:", output)

if __name__ == "__main__":
    main()
