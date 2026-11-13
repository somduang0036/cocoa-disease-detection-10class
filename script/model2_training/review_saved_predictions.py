"""Review saved YOLO26n validation predictions and export matched class errors."""
import argparse
import collections
import concurrent.futures
import csv
import json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--benchmark-run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    ROOT = args.project.resolve()
    RUN = args.benchmark_run.resolve()
    OUT = (args.output or RUN / "yolo26n/validation_error_review").resolve()

    predictions = RUN / "yolo26n/validation/predictions.json"
    annotations = ROOT / "dataset/matadata_model 2/annotations_val.coco.json"
    manifest = ROOT / "dataset/model2_metadata_detection.csv"
    for path in (predictions, annotations, manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    a = json.loads(annotations.read_text(encoding="utf-8-sig"))
    p = json.loads(predictions.read_text(encoding="utf-8-sig"))
    names = {category["id"]: category["name"] for category in a["categories"]}
    dims_by_id = {str(image["id"]): image for image in a["images"]}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        meta = {row["image_id"]: row for row in csv.DictReader(stream) if row["split"] == "val"}
    if dims_by_id.keys() != meta.keys():
        raise ValueError("COCO annotations and manifest contain different validation image IDs.")

    gts = collections.defaultdict(list)
    preds = collections.defaultdict(list)
    for box in a["annotations"]:
        gts[str(box["image_id"])].append(box)
    for box in p:
        image_id = str(box["image_id"])
        if box["category_id"] not in names or image_id not in meta:
            raise ValueError(f"Prediction has an unknown validation image or category: {box}")
        if box["score"] >= 0.25:
            preds[image_id].append(box)

    def iou(a,b):
     x,y,w,h=a;u,v,s,t=b
     inter=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
     return inter/(w*h+s*t-inter) if w*h+s*t-inter>0 else 0
    errors=[];unmatched=[];correct=0
    for iid,gs in gts.items():
     ps=preds[iid];ug=set(range(len(gs)));up=set(range(len(ps)))
     # Match correct-class boxes first, then inspect remaining cross-class matches.
     for same in (True,False):
      pairs=sorted([(iou(gs[g]['bbox'],ps[q]['bbox']),g,q) for g in ug for q in up if (gs[g]['category_id']==ps[q]['category_id'])==same],reverse=True)
      for ov,g,q in pairs:
       if ov<.5 or g not in ug or q not in up:continue
       ug.remove(g);up.remove(q)
       if same:correct+=1
       else:
        gt,pr=gs[g],ps[q]
        errors.append(dict(image_id=iid,image_path=meta[iid]['image_path'],true_class=names[gt['category_id']],predicted_class=names[pr['category_id']],confidence=pr['score'],iou=ov,gt_annotation_id=gt['id'],gt_bbox=gt['bbox'],pred_bbox=pr['bbox']))
     for g in ug: unmatched.append(dict(image_id=iid,type='unmatched_ground_truth',class_name=names[gs[g]['category_id']],bbox=gs[g]['bbox']))
     for q in up: unmatched.append(dict(image_id=iid,type='unmatched_prediction',class_name=names[ps[q]['category_id']],confidence=ps[q]['score'],bbox=ps[q]['bbox']))
    errors.sort(key=lambda r:-r['confidence'])
    OUT.mkdir(parents=True, exist_ok=True)
    error_fields = ("image_id", "image_path", "true_class", "predicted_class",
                    "confidence", "iou", "gt_annotation_id", "gt_bbox", "pred_bbox")
    unmatched_fields = ("image_id", "type", "class_name", "bbox", "confidence")

    def writecsv(name, rows, fields):
        with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    writecsv("wrong_class_boxes.csv", errors, error_fields)
    writecsv("unmatched_boxes.csv", unmatched, unmatched_fields)
    pairs = collections.Counter((row["true_class"], row["predicted_class"]) for row in errors)
    summary = dict(
        benchmark_run=RUN.name, predictions_path=str(predictions), evaluation_split="VALIDATION",
        confidence_min=0.25, matching_iou_min=0.50,
        matching_method="Greedy IoU with same-class matches first, followed by cross-class matches.",
        correct_matches=correct, wrong_class_boxes=len(errors),
        wrong_class_images=len({row["image_id"] for row in errors}),
        unmatched_counts=dict(collections.Counter(row["type"] for row in unmatched)),
        pairs=[dict(true_class=pair[0], predicted_class=pair[1], boxes=count)
               for pair, count in pairs.most_common()],
    )
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Validation error review:", OUT)
    if args.no_previews:
        return

    # Show one error per image, prioritizing rare-class confusions for review.
    priority={'pod-swollen-shoot-virus':0,'pod-anthracnose':1,'pod-borer':2}
    selected=[];seen=set()
    for r in sorted(errors,key=lambda r:(min(priority.get(r['true_class'],3),priority.get(r['predicted_class'],3)),-r['confidence'])):
     if r['image_id'] in seen:continue
     seen.add(r['image_id']);selected.append(r)
     if len(selected)>=12:break
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    def render(r):
     with Image.open(ROOT/'dataset'/r['image_path']) as im: src=ImageOps.exif_transpose(im).convert('RGB')
     # Check coordinate space against the annotation's declared image dimensions.
     dims = dims_by_id[r['image_id']]
     assert src.size==(dims['width'],dims['height']),(r['image_id'],src.size)
     tile=Image.new('RGB',(1300,500),'white');d=ImageDraw.Draw(tile)
     d.text((12,6),f"ID {r['image_id']} | GT: {r['true_class']} -> Pred: {r['predicted_class']}",font=font,fill='black')
     d.text((12,34),f"Confidence {r['confidence']:.3f} | IoU {r['iou']:.3f} | matched box pair only",font=small,fill='black')
     for side,(key,color,label) in enumerate([('gt_bbox','#008000','GROUND TRUTH (annotation)'),('pred_bbox','#c03800','MODEL PREDICTION')]):
      view=ImageOps.contain(src,(630,400));left=side*650+(650-view.width)//2;top=95+(400-view.height)//2
      tile.paste(view,(left,top));d.text((side*650+15,68),label,font=font,fill=color)
      x,y,w,h=r[key];sx=view.width/src.width;sy=view.height/src.height
      d.rectangle((left+x*sx,top+y*sy,left+(x+w)*sx,top+(y+h)*sy),outline=color,width=4)
     return tile
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: tiles=list(pool.map(render,selected))
    for start in range(0,len(tiles),4):
     sheet=Image.new('RGB',(1300,500*len(tiles[start:start+4])),'white')
     for n,t in enumerate(tiles[start:start+4]):sheet.paste(t,(0,n*500))
     sheet.save(OUT/f'wrong_class_review_{start//4+1:02}.jpg',quality=92)
    writecsv("preview_cases.csv", selected, error_fields)
    print("Preview cases:", len(tiles))


if __name__ == "__main__":
    main()
