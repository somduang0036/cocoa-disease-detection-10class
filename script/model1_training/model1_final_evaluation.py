"""Evaluate the frozen EfficientNet-B0 candidate on the held-out test split.
Stages: check (metadata only), test (resumable inference), report (saved predictions).
No training, threshold search, or model selection is implemented.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from contextlib import contextmanager
import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
import PIL
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

PROJECT = Path("/content/drive/MyDrive/cocoa-diseases") if Path("/content/drive/MyDrive").is_dir() else Path(r"H:\My Drive\cocoa-diseases")
CLASS_MAP = {"other_plants": 0, "cocoa": 1}
CLASS_NAMES = ["other_plants", "cocoa"]
ARCHITECTURE = "efficientnet_b0"
EXPECTED_TEST_COUNTS = {"other_plants": 1341, "cocoa": 894}
BATCH_SIZE, NUM_WORKERS, SEED, THRESHOLD = 64, 2, 42, 0.5
PREPROCESSING = dict(image_size=224,resize="stretch",interpolation="bilinear",color="RGB",exif="upright",
                     mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
VERSION = "model1-frozen-final-test-v1"


def require(condition, message):
    if not condition: raise ValueError(message)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def save_json(path, value):
    atomic_bytes(path,json.dumps(value,indent=2,allow_nan=False).encode())


def save_csv(path, frame):
    atomic_bytes(path,frame.to_csv(index=False).encode("utf-8-sig"))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def writer_lock():
    """Use an OS-released local lock; a crashed process cannot leave a stale lock."""
    path=Path(tempfile.gettempdir())/"model1_final_evaluation.lock"
    handle=path.open("a+b")
    handle.seek(0)
    if not handle.read(1): handle.write(b"0");handle.flush()
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
        raise RuntimeError("Another final-evaluation writer is active in this runtime.")
    try: yield
    finally: handle.close()


def resolve_tuning_run(project, explicit):
    if explicit is not None: return explicit.resolve()
    output=project/"results/model1_efficientnet_tuning"
    active=read_json(output/"active_run.json")
    name=Path(active["run_name"])
    require(name.name==str(name),"Unsafe active run path")
    run=output/name
    require(read_json(run/"tuning_run.json")["identity"]==active["identity"],"Active tuning identity differs")
    return run


def load_inputs(run,root):
    """Read metadata and frozen weights only; do not open test images."""
    selection=read_json(run/"selection.json")
    require(selection["candidate"] is not None,"No candidate has been selected")
    require(selection["threshold"]==THRESHOLD and selection["test_evaluated"] is False,
            "Expected a validation-selected candidate at threshold 0.5")
    candidate=run/"application_candidate"
    config=read_json(candidate/"inference_config.json")
    require(config["selection"]==selection,"Application candidate does not match saved selection")
    require(config["architecture"]==ARCHITECTURE and config["class_mapping"]==CLASS_MAP and
            config["positive_class_id"]==1 and config["threshold"]==THRESHOLD,"Inference mapping differs")
    for key,value in PREPROCESSING.items():
        require(config[key]==value,f"Inference preprocessing differs: {key}")
    weights=candidate/"best.pth"
    weights_sha=sha_file(weights)
    require(weights_sha==config["weights_sha256"],"Application weights checksum differs")
    identity=read_json(run/"dataset_identity.json")
    manifest=root/"model1_metadata.csv"
    require(sha_file(manifest)==identity["manifest_sha256"],"Dataset differs from tuning")
    require(identity["class_mapping"]==CLASS_MAP,"Dataset class mapping differs")
    frame=pd.read_csv(manifest,dtype=str,keep_default_na=False,encoding="utf-8-sig")
    require({"image_id","class_name","class_id","split","image_path","status"}<=set(frame),"Required metadata columns missing")
    require(frame.image_id.is_unique and frame.image_path.is_unique,"Duplicate image IDs or paths")
    require(frame.status.eq("KEEP").all(),"Metadata contains non-KEEP rows")
    require(set(frame.class_name)==set(CLASS_MAP),"Class names differ")
    require(frame.class_id.astype(int).eq(frame.class_name.map(CLASS_MAP)).all(),"Class indices differ")
    for key in ["split_group_id","PN_observation_id","PN_hash","file_sha256","output_sha256","output_pixel_sha256"]:
        if key in frame:
            known=frame.loc[frame[key].ne("")]
            require(not known.groupby(key).split.nunique().gt(1).any(),f"Known split leakage: {key}")
    test=frame.loc[frame.split.eq("test")].copy().sort_values("image_id").reset_index(drop=True)
    require(test.groupby("class_name").size().to_dict()==EXPECTED_TEST_COUNTS,"Test counts differ")
    for value in test.image_path:
        p=PurePosixPath(value.replace("\\","/"))
        require(value and not p.is_absolute() and ".." not in p.parts and ":" not in value,"Unsafe test image path")
    selected=selection["candidate"]
    if selected=="baseline":
        source=run/"baseline_reference/best.pth"
    else:
        allowed=read_json(run/"config.json")["trials"]
        require(selected in allowed,"Unrecognized selected trial")
        source=run/selected/ARCHITECTURE/"best.pth"
    require(sha_file(source)==config["source_checkpoint_sha256"],"Selected source checkpoint differs")
    checkpoint=torch.load(weights,map_location="cpu",weights_only=True)
    require(checkpoint["architecture"]==ARCHITECTURE and checkpoint["class_mapping"]==CLASS_MAP and
            checkpoint["threshold"]==THRESHOLD and checkpoint["preprocessing"]==PREPROCESSING,
            "Candidate checkpoint metadata differs")
    require(checkpoint["experiment"]==selected and checkpoint["source_checkpoint_sha256"]==config["source_checkpoint_sha256"],
            "Candidate checkpoint source differs")
    original=torch.load(source,map_location="cpu",weights_only=True)
    require(original["manifest_sha256"]==identity["manifest_sha256"] and original["class_mapping"]==CLASS_MAP,
            "Source checkpoint dataset identity differs")
    require(original["epoch"]==checkpoint["epoch"],"Selected epoch differs")
    require(set(original["model_state_dict"])==set(checkpoint["model_state_dict"]) and
            all(torch.equal(original["model_state_dict"][k],v) for k,v in checkpoint["model_state_dict"].items()),
            "Application parameters differ from selected best checkpoint")
    del original
    protocol=dict(version=VERSION,architecture=ARCHITECTURE,selected_configuration=selected,best_epoch=checkpoint["epoch"],
        manifest_sha256=identity["manifest_sha256"],weights_sha256=weights_sha,
        source_checkpoint_sha256=config["source_checkpoint_sha256"],selection_sha256=sha_file(run/"selection.json"),
        inference_config_sha256=sha_file(candidate/"inference_config.json"),class_mapping=CLASS_MAP,
        preprocessing=PREPROCESSING,threshold=THRESHOLD,test_counts=EXPECTED_TEST_COUNTS,
        inference_precision="float32; matches the application predictor; no mixed precision",
        batch_size=BATCH_SIZE,seed=SEED,
        software_versions=dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),
            numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
            pillow=PIL.__version__,cuda=torch.version.cuda),
        script_sha256=sha_file(Path(__file__)))
    protocol_id=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return test,checkpoint,protocol,protocol_id


def make_model(checkpoint,device):
    model=models.efficientnet_b0(weights=None)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,2)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    return model.to(device).eval().requires_grad_(False)


class TestDataset(Dataset):
    def __init__(self,root,rows):
        self.root=root;self.rows=rows.to_dict("records")
        self.transform=transforms.Compose([
            transforms.Resize((224,224),interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),transforms.Normalize(PREPROCESSING["mean"],PREPROCESSING["std"])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];path=self.root/row["image_path"]
        for attempt in range(3):
            try:
                with Image.open(path) as source:
                    image=ImageOps.exif_transpose(source).convert("RGB")
                    return self.transform(image),int(row["class_id"]),row["image_id"]
            except OSError as error:
                if attempt==2:raise OSError(f"Unreadable test image ID {row['image_id']}: {path}") from error
                time.sleep(attempt+1)


def validate_rows(predictions,test):
    expected=test.set_index("image_id")
    require(predictions.image_id.is_unique,"Duplicate evaluated image IDs")
    require(set(predictions.image_id)<=set(expected.index),"Unknown test image IDs")
    rows=predictions.set_index("image_id")
    reference=expected.loc[rows.index]
    require(rows.image_path.eq(reference.image_path).all() and rows.true_class_id.eq(reference.class_id.astype(int)).all(),
            "Prediction metadata differs")
    require(rows.true_class_name.eq(reference.class_name).all(),"Prediction true class names differ")
    probs=rows[["p_other_plants","p_cocoa"]].to_numpy(float)
    require(np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all() and np.allclose(probs.sum(1),1,atol=1e-5),
            "Invalid saved test probabilities")
    predicted=(rows.p_cocoa>=THRESHOLD).astype(int)
    require(np.array_equal(predicted,rows.predicted_class_id) and rows.threshold.eq(THRESHOLD).all(),
            "Saved decisions differ from the frozen threshold")
    require(rows.predicted_class_name.eq(predicted.map({0:"other_plants",1:"cocoa"})).all(),"Predicted class names differ")
    require(np.isfinite(rows.cross_entropy_loss).all() and rows.cross_entropy_loss.ge(0).all(),"Invalid saved losses")


def load_batches(folder,test,job_id):
    records=[];seen=set()
    for path in sorted(folder.glob("batch_*.json")):
        batch=read_json(path)
        require(batch["protocol_id"]==job_id,"A saved batch belongs to a different frozen protocol")
        index=batch["batch_index"]
        require(isinstance(index,int) and 0<=index<math.ceil(len(test)/BATCH_SIZE),"Invalid batch index")
        require(path.name==f"batch_{index:05d}.json" and index not in seen,"Duplicate/misnamed batch")
        rows=pd.DataFrame(batch["rows"])
        expected=test.iloc[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        require(rows.image_id.tolist()==expected.image_id.tolist(),"Saved batch image order differs")
        validate_rows(rows,test)
        seen.add(index);records.extend(batch["rows"])
    return records,seen


def save_fig(fig,folder,name):
    fig.text(.5,.005,"EVALUATION SPLIT: TEST | Frozen threshold = 0.5",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.97])
    fig.savefig(folder/(name+".png"),dpi=300)
    fig.savefig(folder/(name+".pdf"))
    plt.close(fig)


def report(folder,test,job_id):
    marker=read_json(folder/"COMPLETED.json") if (folder/"COMPLETED.json").exists() else None
    path=folder/"test_predictions.csv"
    if marker:
        require(marker["protocol_id"]==job_id and sha_file(path)==marker["predictions_sha256"],"Completed predictions changed")
    p=pd.read_csv(path,dtype={"image_id":str},keep_default_na=False,encoding="utf-8-sig")
    validate_rows(p,test)
    require(len(p)==len(test) and set(p.image_id)==set(test.image_id),"Incomplete final test predictions")
    labels=p.true_class_id.to_numpy(int);predicted=p.predicted_class_id.to_numpy(int);scores=p.p_cocoa.to_numpy(float)
    cm=confusion_matrix(labels,predicted,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    classification=classification_report(labels,predicted,labels=[0,1],target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    metrics=dict(evaluation_split="TEST",images=len(p),threshold=THRESHOLD,class_mapping=CLASS_MAP,positive_class_id=1,
        test_loss=float(p.cross_entropy_loss.mean()),accuracy=float(accuracy_score(labels,predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels,predicted)),
        cocoa_precision=classification["cocoa"]["precision"],cocoa_recall=tp/(tp+fn),
        other_plants_precision=classification["other_plants"]["precision"],other_plants_recall=tn/(tn+fp),
        other_plants_fpr=fp/(fp+tn),macro_precision=classification["macro avg"]["precision"],
        macro_recall=classification["macro avg"]["recall"],macro_f1=classification["macro avg"]["f1-score"],
        weighted_f1=classification["weighted avg"]["f1-score"],roc_auc_cocoa=float(roc_auc_score(labels,scores)),
        average_precision_cocoa=float(average_precision_score(labels,scores)),TP=tp,TN=tn,FP=fp,FN=fn,
        protocol_id=job_id,threshold_optimized=False,model_selection_performed=False,
        note="Held-out split from the existing dataset; not an external field validation.")
    if marker:
        require(sha_file(folder/"test_metrics.json")==marker["metrics_sha256"],"Completed test metrics changed")
        saved=read_json(folder/"test_metrics.json")
        require(saved==metrics,"Regenerated test metrics differ")
    else:save_json(folder/"test_metrics.json",metrics)
    rows=[]
    for name,values in classification.items():
        if isinstance(values,dict):rows.append(dict(evaluation_split="TEST",class_or_average=name,**values))
    save_csv(folder/"classification_report.csv",pd.DataFrame(rows))
    save_csv(folder/"test_metrics.csv",pd.DataFrame([metrics]).drop(columns=["class_mapping"]))
    for normalized in [False,True]:
        values=cm/cm.sum(1,keepdims=True) if normalized else cm
        name="confusion_matrix_normalized" if normalized else "confusion_matrix"
        save_csv(folder/(name+".csv"),pd.DataFrame(values,columns=CLASS_NAMES).assign(true_class=CLASS_NAMES,evaluation_split="TEST"))
        fig,ax=plt.subplots(figsize=(6,5))
        ConfusionMatrixDisplay(values,display_labels=CLASS_NAMES).plot(ax=ax,cmap="Blues",colorbar=False,
            values_format=".3f" if normalized else "d")
        ax.set_title("Test confusion matrix"+(" (row-normalized)" if normalized else " (counts)"))
        save_fig(fig,folder,name)
    fpr,tpr,thresholds=roc_curve(labels,scores)
    save_csv(folder/"roc_coordinates.csv",pd.DataFrame(dict(fpr=fpr,tpr=tpr,threshold=thresholds,evaluation_split="TEST")))
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(fpr,tpr,label=f"Cocoa AUC = {metrics['roc_auc_cocoa']:.6f}")
    ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(xlabel="Other Plants false positive rate",ylabel="Cocoa recall",title="Test ROC curve",xlim=(0,1),ylim=(0,1.02))
    ax.legend();ax.grid(alpha=.2);save_fig(fig,folder,"roc_curve")
    precision,recall,thresholds=precision_recall_curve(labels,scores)
    save_csv(folder/"pr_coordinates.csv",pd.DataFrame(dict(precision=precision,recall=recall,evaluation_split="TEST")))
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(recall,precision,label=f"Cocoa AP = {metrics['average_precision_cocoa']:.6f}")
    ax.axhline(labels.mean(),color="gray",linestyle="--",label="Cocoa prevalence")
    ax.set(xlabel="Cocoa recall",ylabel="Cocoa precision",title="Test precision-recall curve",xlim=(0,1),ylim=(0,1.02))
    ax.legend();ax.grid(alpha=.2);save_fig(fig,folder,"precision_recall_curve")
    frozen=read_json(folder/"frozen_protocol.json")
    save_json(folder/"deployment_metadata.json",dict(status="FROZEN_CANDIDATE_TEST_EVALUATED",
        architecture=ARCHITECTURE,class_mapping=CLASS_MAP,threshold=THRESHOLD,preprocessing=PREPROCESSING,
        selected_configuration=frozen["protocol"]["selected_configuration"],
        weights_sha256=frozen["protocol"]["weights_sha256"],best_epoch=frozen["protocol"]["best_epoch"],
        test_metrics_file="test_metrics.json",test_predictions_sha256=sha_file(path),
        external_field_validation_performed=False,
        note="Test evaluation records performance; it does not certify reliability for arbitrary non-plant inputs."))
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics


def check_test_files(root,test):
    """Check test file availability once without hashing every image."""
    for parent,group in test.groupby(test.image_path.map(lambda p:str(PurePosixPath(p).parent))):
        with os.scandir(root/parent) as entries:
            sizes={entry.name:entry.stat().st_size for entry in entries if entry.is_file()}
        for row in group.to_dict("records"):
            name=PurePosixPath(row["image_path"]).name
            expected=row.get("output_filesize") or row.get("filesize")
            require(name in sizes and sizes[name]>0,f"Missing/empty test image: {row['image_id']}")
            require(not expected or sizes[name]==int(expected),f"Test image size differs: {row['image_id']}")


def evaluate(root,run,test,checkpoint,protocol,job_id):
    folder=run/"final_evaluation"
    frozen_path=folder/"frozen_protocol.json"
    if frozen_path.exists():
        frozen=read_json(frozen_path)
        require(frozen["protocol_id"]==job_id and frozen["protocol"]==protocol,
            "Final evaluation is already frozen to a different candidate or protocol. Do not overwrite it.")
    else:
        require(not folder.exists() or not any(folder.iterdir()),"Nonempty final-evaluation folder has no frozen identity")
        folder.mkdir(exist_ok=True)
        save_json(frozen_path,dict(protocol_id=job_id,protocol=protocol,frozen_at_utc=timestamp()))
        save_csv(folder/"test_manifest_snapshot.csv",test.drop(columns=["label"],errors="ignore"))
        atomic_bytes(folder/"code_snapshot.py",Path(__file__).read_bytes())
    completed=folder/"COMPLETED.json"
    if completed.exists():
        saved=read_json(completed)
        require(saved["protocol_id"]==job_id and saved["status"]=="COMPLETED","Completion identity differs")
        require(sha_file(folder/"test_predictions.csv")==saved["predictions_sha256"] and
                sha_file(folder/"test_metrics.json")==saved["metrics_sha256"],"Completed evaluation artifacts changed")
        print("TEST ALREADY COMPLETE. No model inference will be repeated.",flush=True)
        print(json.dumps(read_json(folder/"test_metrics.json"),indent=2),flush=True)
        return folder
    require(torch.cuda.is_available(),"CUDA GPU is required for test inference. Select a Colab GPU runtime.")
    batches=folder/"prediction_batches";batches.mkdir(exist_ok=True)
    existing,seen=load_batches(batches,test,job_id)
    total_batches=math.ceil(len(test)/BATCH_SIZE)
    missing=[index for index in range(total_batches) if index not in seen]
    print(f"TEST RESUME: saved {len(existing):,}/{len(test):,}; pending {len(test)-len(existing):,}",flush=True)
    if missing:
        if not (folder/"file_preflight.json").exists():
            print("Checking test file availability and recorded sizes once.",flush=True)
            check_test_files(root,test)
            save_json(folder/"file_preflight.json",dict(protocol_id=job_id,status="PASSED"))
        else:require(read_json(folder/"file_preflight.json")["protocol_id"]==job_id,"File preflight identity differs")
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        device=torch.device("cuda")
        print("GPU:",torch.cuda.get_device_name(0),"| float32 inference | no augmentation",flush=True)
        model=make_model(checkpoint,device)
        del checkpoint
        indices=[j for i in missing for j in range(i*BATCH_SIZE,min((i+1)*BATCH_SIZE,len(test)))]
        pending=test.iloc[indices].copy()
        loader=DataLoader(TestDataset(root,pending),batch_size=BATCH_SIZE,shuffle=False,
            num_workers=NUM_WORKERS,pin_memory=True)
        lookup=test.set_index("image_id")
        completed_images=len(existing);torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            for index,(images,labels,ids) in zip(missing,loader):
                images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
                logits=model(images).float()
                require(torch.isfinite(logits).all().item(),"Non-finite test logits")
                losses=nn.functional.cross_entropy(logits,labels,reduction="none").cpu().numpy()
                probs=logits.softmax(1).cpu().numpy();true=labels.cpu().numpy()
                require(tuple(probs.shape)==(len(ids),2),"Unexpected output shape")
                records=[]
                for j,image_id in enumerate(ids):
                    predicted=int(probs[j,1]>=THRESHOLD)
                    records.append(dict(image_id=image_id,image_path=lookup.loc[image_id,"image_path"],
                        true_class_id=int(true[j]),true_class_name=CLASS_NAMES[int(true[j])],
                        predicted_class_id=predicted,predicted_class_name=CLASS_NAMES[predicted],
                        p_other_plants=float(probs[j,0]),p_cocoa=float(probs[j,1]),
                        cross_entropy_loss=float(losses[j]),threshold=THRESHOLD,
                        architecture=ARCHITECTURE,selected_configuration=protocol["selected_configuration"],evaluation_split="TEST"))
                validate_rows(pd.DataFrame(records),test)
                path=batches/f"batch_{index:05d}.json"
                require(not path.exists(),"Another process already committed this batch")
                save_json(path,dict(protocol_id=job_id,batch_index=index,rows=records))
                completed_images+=len(records)
                torch.cuda.synchronize()
                print(f"TEST {completed_images:,}/{len(test):,} ({100*completed_images/len(test):.1f}%) | saved batch {index+1}/{total_batches}",flush=True)
        save_json(folder/("inference_session_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]+".json"),
            dict(evaluated_images=len(pending),seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(0),
                note="End-to-end test session including loading and batch commits; not a CPU latency benchmark."))
        del model
        torch.cuda.empty_cache()
    records,seen=load_batches(batches,test,job_id)
    require(len(seen)==total_batches and len(records)==len(test),"Not all test batches are committed")
    predictions=pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    validate_rows(predictions,test)
    save_csv(folder/"test_predictions.csv",predictions)
    report(folder,test,job_id)
    save_json(completed,dict(status="COMPLETED",protocol_id=job_id,images=len(test),
        predictions_sha256=sha_file(folder/"test_predictions.csv"),metrics_sha256=sha_file(folder/"test_metrics.json"),
        completed_at_utc=timestamp(),training_performed=False,threshold_optimized=False,
        note="Completed committed batches are reused. An interrupted, uncommitted batch can be evaluated again."))
    print("TEST COMPLETED:",folder,flush=True)
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["check","test","report"],default="check")
    parser.add_argument("--project",type=Path,default=PROJECT)
    parser.add_argument("--tuning-run",type=Path)
    parser.add_argument("--dataset-root",type=Path)
    args=parser.parse_args(argv)
    project=args.project.resolve();root=(args.dataset_root or project/"dataset").resolve()
    run=resolve_tuning_run(project,args.tuning_run)
    print("STAGE:",args.stage,"| TUNING RUN:",run,flush=True)
    test,checkpoint,protocol,job_id=load_inputs(run,root)
    print("FROZEN CANDIDATE:",protocol["selected_configuration"],"| best epoch:",protocol["best_epoch"],
          "| threshold:",THRESHOLD,"| Test images:",len(test),flush=True)
    if args.stage=="check":
        model=make_model(checkpoint,torch.device("cpu"))
        with torch.inference_mode(): logits=model(torch.zeros(1,3,224,224))
        require(tuple(logits.shape)==(1,2),"Output dimension differs")
        print("CHECK PASSED: metadata and model structure verified; no test images opened or evaluated.",flush=True)
        return 0
    with writer_lock():
        if args.stage=="report":
            folder=run/"final_evaluation"
            require(read_json(folder/"frozen_protocol.json")["protocol_id"]==job_id,"Frozen protocol differs")
            require((folder/"COMPLETED.json").exists(),"Report requires completed final evaluation")
            report(folder,test,job_id)
            return 0
        evaluate(root,run,test,checkpoint,protocol,job_id)
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED. Re-run the same test cell to reuse committed prediction batches.",flush=True)
        raise SystemExit(130)
