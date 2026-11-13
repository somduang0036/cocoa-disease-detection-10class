"""Train and evaluate four YOLO detectors using shared Cocoa images and validation-only selection."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import zipfile
import io

MODELS = ['yolov8m', 'yolo11n', 'yolo12n', 'yolo26n']
NAMES = ['leaf-anthracnose', 'leaf-healthy', 'leaf-swollen-shoot-virus',
         'pod-anthracnose', 'pod-blackpod', 'pod-borer', 'pod-frostyrot',
         'pod-healthy', 'pod-mirid', 'pod-swollen-shoot-virus']
CONFIG = dict(imgsz=640, epochs=100, patience=10, batch=32, workers=2,
              seed=42, optimizer='AdamW', lr0=0.001, weight_decay=0.0005,
              amp=True, cache=False, cos_lr=False, close_mosaic=10,
              degrees=10.0, translate=0.1, scale=0.5, shear=0.0,
              perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0,
              mixup=0.0, copy_paste=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
              deterministic=True, rect=False, save=True, save_period=5,
              plots=True, val=True)
# Repeat each pod-borer training image three times in the training view.
# Links reuse original pixels and labels; validation remains unchanged.
POD_BORER_EXPOSURES = 3
FORMAT = 'model2-four-yolo-benchmark-v1'
ULTRALYTICS_VERSION = '8.4.198'



def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        require(destination.resolve() == source.resolve(), f'Link identity differs: {destination}')
    elif destination.exists():
        raise RuntimeError(f'Unexpected existing file: {destination}')
    else:
        destination.symlink_to(source.resolve())



def repair_image_cache(args, records, cached):
    """Repair only missing or size-mismatched runtime images from verified sources."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pending=[]
    for r in records:
        old=cached.get(r['image_id'])
        require(old and old['split']==r['split'] and old['output_sha256']==r['output_sha256'],
                f"Cache manifest mismatch: {r['image_id']}")
        relative=Path(old['image_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe cache image path.')
        destination=args.image_cache/relative
        require(destination.resolve().is_relative_to(args.image_cache.resolve()), 'Cache image escapes cache root.')
        if not destination.is_file() or destination.stat().st_size != int(r['output_filesize']):
            pending.append((r,destination))
    if not pending:
        print('IMAGE CACHE: all required Cocoa image sizes match; no images copied.',flush=True)
        return
    print(f'IMAGE CACHE REPAIR: {len(pending)} missing/incomplete images; existing complete images reused.',flush=True)
    def restore(item):
        row,destination=item
        relative=Path(row['image_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source image path.')
        source=args.project/'dataset'/relative
        temporary=destination.with_name(destination.name+'.repair.tmp')
        destination.parent.mkdir(parents=True,exist_ok=True)
        try:
            require(source.is_file(),f'Missing source on Drive: {source}')
            require(source.stat().st_size==int(row['output_filesize']),f'Incomplete source on Drive: {source}')
            shutil.copyfile(source,temporary)
            require(temporary.stat().st_size==int(row['output_filesize']),f'Incomplete repaired image: {source}')
            require(sha(temporary)==row['output_sha256'],f'Source checksum mismatch: {source}')
            os.replace(temporary,destination)
        finally:
            temporary.unlink(missing_ok=True)
        return row['image_id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(restore,item) for item in pending]
        for n,future in enumerate(as_completed(futures),1):
            image_id=future.result()
            if n==1 or n%25==0 or n==len(pending):
                print(f'IMAGE CACHE REPAIRED {n}/{len(pending)} | ID {image_id}',flush=True)


def prepare(args):
    require(platform.system() == 'Linux', 'Run this cache-sharing pipeline in Colab/Linux.')
    root = args.project / 'dataset'
    manifest = root / 'model2_metadata_detection.csv'
    rows = csv_rows(manifest)
    require(len(rows) == 8935, 'Expected 8,935 detection images.')
    require(len({r['image_id'] for r in rows}) == len(rows), 'Duplicate image IDs.')
    require(Counter(r['split'] for r in rows) == {'train':7147,'val':894,'test':894}, 'Split counts differ.')
    for column in ['split_group_id', 'output_sha256']:
        groups = {}
        for r in rows:
            if r.get(column):
                groups.setdefault(r[column], set()).add(r['split'])
        require(all(len(v) == 1 for v in groups.values()), f'Known split leakage: {column}')
    mapping_path = root / 'matadata_model 2/class_mapping.json'
    classes = read(mapping_path)['classes']
    require({int(c['yolo_class_id']):c['class_name'] for c in classes} == dict(enumerate(NAMES)), 'Detection mapping differs.')
    marker = read(args.image_cache / '_cache_ready.json')
    cached_manifest = args.image_cache / 'model1_metadata.csv'
    require(marker['manifest_sha256'] == sha(cached_manifest), 'Model 1 cache manifest identity differs.')
    cached = {r['image_id']:r for r in csv_rows(cached_manifest) if r['class_name'] == 'cocoa'}
    identity = hashlib.sha256(json.dumps(dict(format=FORMAT, script_sha256=sha(__file__), manifest=sha(manifest), mapping=sha(mapping_path),
                  config=CONFIG, models=MODELS, pod_borer_exposures=POD_BORER_EXPOSURES), sort_keys=True).encode()).hexdigest()
    view = args.work / identity[:16]
    ready = view / 'ready.json'
    repair_image_cache(args, [r for r in rows if r['split'] in ('train','val')], cached)
    if ready.exists():
        previous = read(ready)
        require(previous['identity'] == identity and previous['image_cache'] == str(args.image_cache.resolve()), 'Runtime view identity differs.')
        require(all(Path(p).is_file() for p in previous['required_paths']), 'Runtime view is incomplete; remove ready.json and rerun prepare.')
        print('DATA READY: verified runtime links and labels reused.', flush=True)
        return identity, view
    archive_path = root / 'model2_train_val_labels.zip'
    print('LABELS: loading one packed archive from Drive...', flush=True)
    require(archive_path.is_file(), f'Wait for Drive sync: {archive_path}')
    archive_bytes = archive_path.read_bytes()
    label_pack = zipfile.ZipFile(io.BytesIO(archive_bytes))
    packed_identity = json.loads(label_pack.read('identity.json'))
    require(packed_identity['manifest_sha256'] == sha(manifest), 'Packed labels belong to a different manifest.')
    require(packed_identity['labels'] == 8041, 'Packed label count differs.')
    records = [r for r in rows if r['split'] in ('train','val')]
    started_prepare = last_progress = time.monotonic()
    lists = {'train':[], 'val':[]}
    required = []
    label_digest = hashlib.sha256()
    for index, r in enumerate(records, 1):
        cid = r['image_id']; old = cached.get(cid)
        require(old and old['split'] == r['split'] and old['output_sha256'] == r['output_sha256'], f'Cache record mismatch: {cid}')
        image = args.image_cache / old['image_path']
        require(image.is_file() and image.stat().st_size == int(r['output_filesize']), f'Missing/incomplete original image: {image}')
        source_label = root / r['label_path']
        payload = label_pack.read(r['label_path'])
        label_digest.update(cid.encode() + b'\0' + payload)
        labels = [line.split() for line in payload.decode('utf-8-sig').splitlines() if line.strip()]
        require(len(labels) == int(r['bbox_count']), f'Box count differs: {cid}')
        present = set()
        for box in labels:
            require(len(box) == 5, f'Invalid YOLO row: {cid}')
            cls = int(box[0]); coords = list(map(float,box[1:]))
            require(0 <= cls < 10 and all(0 <= x <= 1 for x in coords) and coords[2] > 0 and coords[3] > 0, f'Invalid box: {cid}')
            present.add(cls)
        require(present == set(json.loads(r['yolo_class_ids'])), f'Class list differs: {cid}')
        copies = POD_BORER_EXPOSURES if r['split']=='train' and 5 in present else 1
        for copy in range(copies):
            filename = cid + (f'_repeat{copy}' if copy else '')
            image_link = view / 'images' / r['split'] / (filename + image.suffix)
            label = view / 'labels' / r['split'] / (filename + '.txt')
            link(image, image_link)
            label.parent.mkdir(parents=True, exist_ok=True)
            if not label.is_file() or label.read_bytes() != payload:
                temporary_label = label.with_name(label.name + '.tmp')
                temporary_label.write_bytes(payload)
                os.replace(temporary_label, label)
            lists[r['split']].append(str(image_link.absolute()))
            required += [str(image_link),str(label)]
        now = time.monotonic()
        if index == 1 or index % 500 == 0 or now-last_progress >= 10 or index == len(records):
            elapsed = now-started_prepare
            rate = index/max(elapsed,0.001)
            print(f'PREPARE {index:,}/{len(records):,} ({100*index/len(records):.1f}%) | '
                  f'{rate:.1f} images/s | ETA {(len(records)-index)/max(rate,.001):.0f}s | local labels', flush=True)
            last_progress = now
    label_pack.close()
    view.mkdir(parents=True, exist_ok=True)
    for split, paths in lists.items():
        (view / f'{split}.txt').write_text('\n'.join(paths)+'\n', encoding='utf-8')
    import yaml
    (view/'data.yaml').write_text(yaml.safe_dump(dict(path=str(view),train='train.txt',val='val.txt',names=dict(enumerate(NAMES))),sort_keys=False),encoding='utf-8')
    save(ready, dict(identity=identity, image_cache=str(args.image_cache.resolve()), required_paths=required,
                    labels_sha256=label_digest.hexdigest(), unique_train=7147, unique_val=894,
                    training_exposures=len(lists['train']), test_images_loaded=0))
    print(f'DATA READY: {len(lists["train"])} training exposures; 894 validation images.', flush=True)
    return identity, view


def report(run):
    import pandas as pd
    records=[]
    for name in MODELS:
        path=run/name/'validation_metrics.json'
        if path.exists(): records.append(read(path))
    require(len(records) == len(MODELS), 'Complete validation for all four models before comparing.')
    frame=pd.DataFrame(records).sort_values('map50_95',ascending=False)
    frame.to_csv(run/'comparison.csv',index=False)
    print('\nEVALUATION SPLIT: VALIDATION\n'+frame.to_string(index=False),flush=True)
    save(run/'winner_candidate.json',dict(candidate=frame.iloc[0]['model'],criterion='Highest validation mAP50-95',
         descriptive_only=True, statistical_significance_claimed=False, test_evaluated=False))
    class_frames=[]
    for model_name in MODELS:
        part=pd.read_csv(run/model_name/'per_class_validation.csv')
        part.insert(0,'model',model_name)
        class_frames.append(part)
    class_results=pd.concat(class_frames,ignore_index=True)
    class_results.to_csv(run/'per_class_comparison.csv',index=False)
    print('\nVALIDATION RESULTS BY CLASS\n'+class_results.to_string(index=False),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for measure in ['ap50_95','ap50','precision','recall','f1']:
        table=class_results.pivot(index='class_name',columns='model',values=measure).reindex(NAMES)[MODELS]
        table.to_csv(run/f'class_{measure}_comparison.csv')
        fig,ax=plt.subplots(figsize=(9,8))
        values=table.to_numpy()
        im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu');fig.colorbar(im,ax=ax,label=measure)
        ax.set_xticks(range(len(MODELS)),MODELS,rotation=20,ha='right');ax.set_yticks(range(10),NAMES)
        for i in range(10):
            for j in range(len(MODELS)):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>.65 else 'black')
        ax.set_title(f'Validation  -  Per-class {measure}')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(run/f'class_{measure}_comparison.{ext}',dpi=300)
        plt.close(fig)
    save(run/'per_class_interpretation.json',dict(evaluation_split='VALIDATION',
        metric_for_class_comparison='AP50-95',
        per_class_leaders_are_descriptive=True,
        statistical_significance_claimed=False,
        note='Use support counts when discussing rare classes. Per-class Precision/Recall/F1 operating points are framework-selected, not a shared deployment threshold.'))
    fig,ax=plt.subplots(figsize=(9,5))
    frame.set_index('model')[['map50_95','map50','precision','recall']].plot.bar(ax=ax)
    ax.set_ylim(0,1);ax.set_title('Model 2  -  Validation Results');ax.set_ylabel('Score');ax.set_xlabel('Model')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(run/f'validation_comparison.{ext}',dpi=300)
    plt.close(fig)
    labels = NAMES + ['background']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout='constrained')
    for axis, name in zip(axes.flat, MODELS):
        matrix = pd.read_csv(run / name / 'confusion_matrix_normalized.csv', index_col=0)
        require(list(matrix.index) == labels and list(matrix.columns) == labels,
                f'Confusion matrix class order differs: {name}')
        values = matrix.to_numpy()
        heatmap = axis.imshow(values, cmap='Blues', vmin=0, vmax=1)
        axis.set_title(name)
        axis.set_xticks(range(len(labels)), labels, rotation=65, ha='right', fontsize=8)
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        axis.set_xlabel('Ground-truth class')
        axis.set_ylabel('Predicted class')
        for row, col in zip(*values.nonzero()):
            value = values[row, col]
            axis.text(col, row, f'{value:.2f}', ha='center', va='center',
                      fontsize=7, color='white' if value > 0.5 else 'black')
    fig.colorbar(heatmap, ax=axes.ravel().tolist(), label='Proportion within ground-truth column')
    fig.suptitle('Validation confusion matrices')
    for ext in ('png', 'pdf'):
        fig.savefig(run / f'confusion_matrix_normalized_2x2.{ext}', dpi=300)
    plt.close(fig)
    print('REPORT READY:',run,flush=True)



def save_evidence(confusion_matrix, metrics, folder, history_path):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = folder / 'publication_figures'
    figures.mkdir(exist_ok=True)
    def finish(fig, name):
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(figures / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    matrix = np.asarray(confusion_matrix)
    require(matrix.shape == (len(NAMES)+1,len(NAMES)+1), 'Invalid validation confusion-matrix shape.')
    labels = NAMES + ['background']
    pd.DataFrame(matrix,index=labels,columns=labels).to_csv(folder/'confusion_matrix.csv')
    # Ultralytics uses predicted rows and true columns.
    normalized = matrix / np.maximum(matrix.sum(axis=0,keepdims=True),1)
    for name, data in [('confusion_matrix',matrix),('confusion_matrix_normalized',normalized)]:
        pd.DataFrame(data,index=labels,columns=labels).to_csv(folder/f'{name}.csv')
        fig,ax=plt.subplots(figsize=(12,11))
        im=ax.imshow(data,cmap='Blues');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(11),labels,rotation=65,ha='right');ax.set_yticks(range(11),labels)
        ax.set_xlabel('True class');ax.set_ylabel('Predicted class');ax.set_title('Validation  -  '+name.replace('_',' '))
        for i in range(11):
            for j in range(11):ax.text(j,i, f'{data[i,j]:.2f}' if 'normalized' in name else str(int(data[i,j])),ha='center',va='center',fontsize=7)
        finish(fig,name)
    for index, curve in enumerate(metrics.curves_results):
        x,y,xlabel,ylabel=curve
        y=np.asarray(y);x=np.asarray(x)
        table=pd.DataFrame(y.T,index=x);table.index.name=xlabel
        table.to_csv(folder/f'curve_{index}.csv')
        fig,ax=plt.subplots(figsize=(8,6))
        for row in y:ax.plot(x,row,alpha=.35)
        ax.plot(x,y.mean(axis=0),color='black',label='Mean over evaluated classes')
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title('Validation  -  '+str(metrics.curves[index]));ax.legend()
        finish(fig,f'curve_{index}')
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history.columns = history.columns.str.strip()
        for component in ('box_loss', 'cls_loss', 'dfl_loss', 'l1_loss'):
            columns = [f'{split}/{component}' for split in ('train', 'val')
                       if f'{split}/{component}' in history]
            if not columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for column in columns:
                ax.plot(history['epoch'], history[column], label=column.split('/')[0])
            ax.set(xlabel='Epoch', ylabel='Loss', title='Training / Validation  -  ' + component)
            ax.legend()
            finish(fig, component)
    save(folder/'evaluation_spec.json',dict(split='VALIDATION',nms_iou=0.7,prediction_confidence_floor=0.001,
        ap_iou_range='0.50:0.05:0.95',per_class_precision_recall_f1='Framework maximum mean F1 operating point',
        confusion_matrix='Framework confidence and IoU defaults; see pinned validator implementation',
        curves='Native metric curve arrays retained; no smoothing added by report code'))


def evaluate_checkpoint(args, view, identity, name, best, folder, started=None):
    from ultralytics import YOLO
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter() if started is None else started
    trained = YOLO(str(best))
    require(list(trained.names.values()) == NAMES, f'Checkpoint class mapping differs: {best}')
    validation_state = {}
    def capture_validation_state(validator):
        matrix = validator.confusion_matrix.matrix.copy()
        require(matrix.shape == (len(NAMES)+1,len(NAMES)+1), 'Unexpected confusion matrix dimensions.')
        validation_state['matrix'] = matrix
        save(folder/'validation_confusion_snapshot.json',dict(
            model=name, checkpoint_sha256=sha(best), evaluation_split='VALIDATION',
            class_names=NAMES, orientation='predicted_rows_true_columns',
            matrix=matrix.tolist(),
            confidence=getattr(validator.confusion_matrix,'conf',None),
            matching_iou=getattr(validator.confusion_matrix,'iou_thres',None)))
    trained.add_callback('on_val_end',capture_validation_state)
    print(f'VALIDATION EVIDENCE: {name}; finished training will not be repeated.',flush=True)
    metrics=trained.val(data=str(view/'data.yaml'),split='val',imgsz=640,batch=CONFIG['batch'],workers=2,
          device=0,conf=0.001,iou=0.7,plots=True,save_json=True,project=str(folder),name='validation',exist_ok=True)
    import pandas as pd
    perclass=[]
    validation_rows=[r for r in csv_rows(args.project/'dataset/model2_metadata_detection.csv') if r['split']=='val']
    coco = read(args.project / 'dataset/matadata_model 2/annotations_val.coco.json')
    class_by_category = {category['id']: category['name'] for category in coco['categories']}
    box_counts = Counter(class_by_category[item['category_id']] for item in coco['annotations'])
    for i,cls in enumerate(metrics.box.ap_class_index):
        support = [r for r in validation_rows if int(cls) in json.loads(r['yolo_class_ids'])]
        box_support=box_counts[NAMES[int(cls)]]
        perclass.append(dict(class_id=int(cls),class_name=NAMES[int(cls)],validation_images=len(support),ground_truth_boxes=box_support,precision=float(metrics.box.p[i]),
              recall=float(metrics.box.r[i]),f1=float(metrics.box.f1[i]),ap50=float(metrics.box.ap50[i]),ap50_95=float(metrics.box.ap[i])))
    pd.DataFrame(perclass).to_csv(folder/'per_class_validation.csv',index=False)
    require('matrix' in validation_state, 'Validation callback did not supply a confusion matrix.')
    save_evidence(validation_state['matrix'], metrics, folder, best.parent.parent / 'results.csv')
    result=dict(model=name,evaluation_split='VALIDATION',map50_95=float(metrics.box.map),map50=float(metrics.box.map50),
         precision=float(metrics.box.mp),recall=float(metrics.box.mr),
         f1=float(metrics.box.f1.mean()),stage_seconds=time.perf_counter()-started)
    save(folder/'validation_metrics.json',result)
    save(folder / 'COMPLETED.json',dict(identity=identity,best_sha256=sha(best),metrics_sha256=sha(folder/'validation_metrics.json')))


def evaluate_existing(args, output, view, identity):
    three_run = args.three_model_run or (
        args.project / 'results/model2_initial_benchmark/run_20261101T175940Z_3ff02efb'
    )
    yolo26_run = args.yolo26_run or (
        args.project / 'results/model2_yolo26_benchmark/run_20261102T111137Z_c24e0f'
    )
    checkpoints = {
        name: ((yolo26_run if name == 'yolo26n' else three_run) / name / 'weights/best.pt').resolve()
        for name in MODELS
    }
    for checkpoint in checkpoints.values():
        require(checkpoint.is_file(), f'Missing checkpoint: {checkpoint}')
    provenance = {
        'mode': 'validation_of_existing_checkpoints',
        'data_identity': identity,
        'ultralytics': ULTRALYTICS_VERSION,
        'checkpoints': {name: {'path': str(path), 'sha256': sha(path)}
                        for name, path in checkpoints.items()},
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = output / f'evaluation_{stamp}'
    run.mkdir(exist_ok=False)
    save(run / 'evaluation_inputs.json', provenance)
    for name, checkpoint in checkpoints.items():
        evaluate_checkpoint(args, view, identity, name, checkpoint, run / name)
    report(run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--image-cache',type=Path,default=Path('/content/model1_current_dataset'))
    parser.add_argument('--work',type=Path,default=Path('/content/model2_runtime'))
    parser.add_argument('--stage',choices=['check','prepare','train','evaluate','report'],required=True)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--run', type=Path, help='Report an explicit completed run.')
    parser.add_argument('--three-model-run', type=Path)
    parser.add_argument('--yolo26-run', type=Path)
    args=parser.parse_args()
    args.project = args.project.resolve()
    args.image_cache = args.image_cache.resolve()
    args.work = args.work.resolve()
    output = (args.output_root or args.project / 'results/model2_four_model_benchmark').resolve()
    if args.stage == 'report':
        run = args.run or Path(read(output / 'active_run.json')['run'])
        report(run)
        return

    os.environ['YOLO_VERBOSE'] = 'False'
    identity,view=prepare(args)
    print('PROTOCOL: 640 px | 100 epochs | patience 10 | batch 32 | seed 42 | AdamW',flush=True)
    print('POD-BORER: 3 training exposures per image; no test evaluation.',flush=True)
    if args.stage in ('check','prepare'):return
    import torch
    import ultralytics
    from ultralytics import YOLO
    require(ultralytics.__version__ == ULTRALYTICS_VERSION,
            f'Install ultralytics=={ULTRALYTICS_VERSION} before running this protocol.')
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'evaluate':
        require(torch.cuda.is_available(), 'Select a GPU runtime for validation.')
        evaluate_existing(args, output, view, identity)
        return
    require(torch.cuda.is_available(),'CUDA is required. Select a GPU runtime in Colab.')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    from ultralytics import YOLO
    active_path=output/'active_run.json'
    if active_path.exists() and read(active_path)['identity']==identity:
        run=Path(read(active_path)['run'])
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run=output/f'run_{stamp}_{identity[:8]}'
        run.mkdir(exist_ok=False)
        save(run/'config.json',dict(protocol=CONFIG,models=MODELS,identity=identity,class_mapping=dict(enumerate(NAMES)),
             ultralytics=ultralytics.__version__,torch=torch.__version__,pod_borer_exposures=POD_BORER_EXPOSURES, runtime_data=str(view / 'data.yaml')))
        shutil.copyfile(__file__,run/'code_snapshot.py')
        save(active_path,dict(identity=identity,run=str(run)))
    cfg=read(run/'config.json')
    require(cfg['ultralytics']==ultralytics.__version__ and cfg['torch']==torch.__version__,'Resume software versions differ.')
    require(cfg['protocol'] == CONFIG and cfg['models'] == MODELS,
            'Resume configuration differs from this benchmark.')
    require(cfg['runtime_data'] == str(view / 'data.yaml'),
            'Resume requires the same runtime data path. Use the original --work path.')

    for name in MODELS:
        folder=run/name;done=folder/'COMPLETED.json';best=folder/'weights/best.pt';last=folder/'weights/last.pt'
        if done.exists():
            saved=read(done)
            require(saved['best_sha256']==sha(best) and saved['metrics_sha256']==sha(folder/'validation_metrics.json'),'Completed artifacts differ.')
            print('SKIP COMPLETED:',name,flush=True);continue
        finished=folder/'TRAINING_FINISHED.json'
        model=YOLO(str(best) if finished.exists() else (str(last) if last.exists() else f'{name}.pt'))
        def mark_finished(trainer):
            save(folder/'TRAINING_FINISHED.json',dict(identity=identity,epoch=int(trainer.epoch)+1))
        def log_epoch_summary(trainer):
            epoch = int(trainer.epoch) + 1
            if getattr(trainer, '_last_logged_epoch', None) == epoch:
                return
            trainer._last_logged_epoch = epoch
            losses = trainer.label_loss_items(trainer.tloss, prefix='train')
            values = {**losses, **trainer.metrics}
            keys = ['train/box_loss','train/cls_loss','train/dfl_loss','train/l1_loss',
                    'val/box_loss','val/cls_loss','val/dfl_loss','val/l1_loss',
                    'metrics/precision(B)','metrics/recall(B)',
                    'metrics/mAP50(B)','metrics/mAP50-95(B)']
            details = ' | '.join(f'{key}={float(values[key]):.4f}' for key in keys if key in values)
            print(f'{name} epoch {epoch}/{trainer.epochs} | {details}', flush=True)
        model.add_callback('on_fit_epoch_end',log_epoch_summary)
        model.add_callback('on_train_end',mark_finished)
        started=time.perf_counter()
        if not finished.exists():
            require(last.exists() or not folder.exists(),
                    f'Run directory has no resumable checkpoint: {folder}. Choose a new output root.')
            print('TRAIN/RESUME:',name,flush=True)
            if last.exists():model.train(resume=True,device=0,batch=CONFIG['batch'])
            else:model.train(data=str(view/'data.yaml'),project=str(run),name=name,exist_ok=True,device=0,**CONFIG)
        require(best.is_file(),'Missing best checkpoint: '+str(best))
        evaluate_checkpoint(args, view, identity, name, best, folder, started)
        print('COMPLETED:',name,flush=True)
        del model
        torch.cuda.empty_cache()
    report(run)

if __name__=='__main__':
    # Plotting and CLI help also work on Windows. Dataset preparation uses Linux links.
    if '--help' in sys.argv or '-h' in sys.argv or platform.system() != 'Linux':
        main()
        sys.exit()
    import fcntl
    import tempfile
    lock_path = Path(tempfile.gettempdir()) / 'cocoa_model2_benchmark.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Model 2 process is running in this runtime.')
        main()
