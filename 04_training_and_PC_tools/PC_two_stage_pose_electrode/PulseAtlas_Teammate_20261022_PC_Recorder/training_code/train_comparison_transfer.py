"""Ten-class replay plus manually corrected paired USB examples. Select only on old/new validation."""
import os,json,shutil,csv,gc,time
from pathlib import Path
RUN=Path('artifacts/pulseatlas/comparison_transfer_20261022').resolve()
DATA=Path('datasets/pulseatlas_comparison_transfer_20261022').resolve()
OLD=Path('datasets/pulseatlas_v4_focus_20261021').resolve()
PAIRS=Path('datasets/pulseatlas_v4_rl_comparison_20261021').resolve()
BASE=Path('artifacts/pulseatlas/fusion_transfer_20261021/final/electrode_fusion_transfer.pt').resolve()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def identity(cases):
    from evaluate_pulseatlas import iou
    counts={str(c):dict(truth=0,top1=0,unambiguous=0,missed=0,conflicting=0) for c in (2,7)}
    for row in cases:
        for gt in row['truth']:
            if int(gt[0]) not in (2,7):continue
            r=counts[str(int(gt[0]))];r['truth']+=1
            options=sorted([p for p in row['prediction'] if iou(p[:4],gt[1:])>=.5],key=lambda p:-p[4])
            if not options:r['missed']+=1;continue
            right=int(options[0][5])==int(gt[0]);r['top1']+=right
            conflict=any(int(p[5])!=int(gt[0]) for p in options);r['conflicting']+=conflict;r['unambiguous']+=right and not conflict
    return counts
def build():
    import cv2,numpy as np,yaml,hashlib
    rng=np.random.default_rng(20261022);manifest=[]
    # Original general ten-class views; use the clean Val29 and untouched historical test sets.
    original=Path('datasets/pulseatlas_v3_fullclasses_20261021').resolve()
    for split in ('train','val','legacy_val','test','legacy_test'):
        source=original if split=='train' else OLD
        for kind in ('images','labels'):shutil.copytree(source/kind/split,DATA/kind/split,dirs_exist_ok=True)
    for split in ('val','test'):
        for kind in ('images','labels'):shutil.copytree(PAIRS/kind/split,DATA/kind/('comparison_'+split),dirs_exist_ok=True)
    audit=json.loads((PAIRS/'audit.json').read_text());assert audit['passed'] and not audit['automatic_proposals_used']
    rows=json.loads((PAIRS/'manifest.json').read_text(encoding='utf-8'));held={r['sha256'] for r in rows if r['split']!='train'}
    for row in rows:
        if row['split']!='train':continue
        if row['sha256'] in held:raise RuntimeError('Video-parent split leakage')
        im=cv2.imread(row['image']);h,w=im.shape[:2];truth=row['truth']
        for k in range(6):
            # Keep both heads in every view; random context helps avoid fixed screen-position shortcuts.
            x1,y1,x2,y2=(0,0,w,h) if k==0 else (int(rng.integers(100,450)),0,int(rng.integers(1550,w+1)),h)
            patch=im[y1:y2,x1:x2].copy();ph,pw=patch.shape[:2]
            if k%3==1:patch=cv2.GaussianBlur(patch,(3,3),.5)
            if k>0:patch=np.clip(patch.astype(np.float32)*rng.uniform(.9,1.1),0,255).astype(np.uint8)
            stem=Path(row['image']).stem+f'_view{k}';cv2.imwrite(str(DATA/'images/train'/(stem+'.jpg')),patch)
            lines=[]
            for cls,a,b,c,d in truth:
                a-=x1;c-=x1;b-=y1;d-=y1
                assert 0<=a<c<=pw and 0<=b<d<=ph
                lines.append(f'{cls} {(a+c)/2/pw:.8f} {(b+d)/2/ph:.8f} {(c-a)/pw:.8f} {(d-b)/ph:.8f}')
            (DATA/'labels/train'/(stem+'.txt')).write_text('\n'.join(lines)+'\n')
            manifest.append(dict(image=stem,parent=row['sha256'],seconds=row['seconds'],kind='paired_RL_V4',split='train'))
    # Previously inspected hand-only negatives; do not crop around unreviewed parts of the new clip.
    runtime=Path('artifacts/pulseatlas/roi_usb_v3_20261021/source_runtime/snapshots')
    for name,hands in [('case_20261021_220642_585719',[[950,680,1200,875],[960,740,1150,890]]),('case_20261021_220648_209192',[[880,685,1150,890],[900,750,1100,900]])]:
        im=cv2.imread(str(runtime/name/'raw.jpg'))
        for j,(a,b,c,d) in enumerate(hands):
            for k in range(3):
                stem=f'hand_{name}_{j}_{k}';patch=im[b:d,a:c];cv2.imwrite(str(DATA/'images/train'/(stem+'.jpg')),cv2.resize(patch,None,fx=k+1,fy=k+1));(DATA/'labels/train'/(stem+'.txt')).write_text('')
                manifest.append(dict(image=stem,parent=name,kind='reviewed_hand_background',split='train'))
    cfg=yaml.safe_load((OLD/'data.yaml').read_text());cfg['path']=str(DATA);(DATA/'data.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True))
    # Verify every exported YOLO record and the explicitly excluded original parent.
    classes={str(i):0 for i in range(10)};images=0;negative=0
    for image in (DATA/'images/train').glob('*.jpg'):
        assert 'clip_144523__frame_00062_t000061856ms' not in image.stem
        label=DATA/'labels/train'/(image.stem+'.txt');lines=label.read_text().splitlines();images+=1;negative+=not bool(lines)
        for line in lines:
            cls,x,y,w,h=map(float,line.split());assert int(cls)==cls and 0<=cls<10 and 0<w<=1 and 0<h<=1 and 0<=x-w/2<x+w/2<=1.000001 and 0<=y-h/2<y+h/2<=1.000001;classes[str(int(cls))]+=1
    result=dict(passed=True,train_images=images,train_background_images=negative,train_class_boxes=classes,added_views=len(manifest),paired_unique_parents=13,new_val_images=3,new_test_images=4,old_val_images=29,old_legacy_val_images=19,old_tests_unchanged=True,source_automatic_proposals_used=False,same_video_splits_correlated=True,manifest=manifest)
    dump(DATA/'dataset_report.json',result);dump(RUN/'dataset_report.json',result)
def main():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'training_run.json').exists():raise RuntimeError('Run exists; do not retrain silently')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(RUN/'ultralytics'));os.environ.setdefault('MPLCONFIGDIR',str(RUN/'matplotlib'))
    import torch,cv2
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import strip_optimizer
    import fusion_transfer_common as common
    torch.set_num_threads(4);cv2.setNumThreads(4);build();common.DATA=DATA
    baseline=None;best=None;history=[]
    def review(path,epoch):
        nonlocal baseline,best
        model=YOLO(str(path));cases={s:common.collect(model,s,[736,1280]) for s in ('val','legacy_val','comparison_val')}
        metrics={s:common.summarize(v,s=='val')['strict'] for s,v in cases.items()};ids=identity(cases['comparison_val'])
        if baseline is None:baseline=metrics
        eligible=metrics['val']['f1']>=.96 and metrics['val']['macro_f1']>=.95 and metrics['legacy_val']['f1']>=.98
        for s in ('val','legacy_val'):
            for name,old in baseline[s]['per_class'].items():
                now=metrics[s]['per_class'][name]
                # Preserve every old non-V4 true-positive. Modest extra FP budget is separately penalized/reported.
                if name!='V4' and now['tp']<old['tp']:eligible=False
                if now['fp']>old['fp']+1:eligible=False
        score=(min(r['unambiguous'] for r in ids.values()),sum(r['unambiguous'] for r in ids.values()),sum(r['top1'] for r in ids.values()),-metrics['comparison_val']['fp'],metrics['val']['macro_f1'],metrics['legacy_val']['f1'],-epoch)
        row=dict(epoch=epoch,eligible=eligible,identity=ids,metrics=metrics);history.append(row)
        if eligible and (best is None or score>best[0]):best=(score,row);shutil.copy2(path,RUN/'selected_full.pt')
        dump(RUN/'selection.json',dict(best=None if best is None else best[1],baseline=baseline,history=history,test_used_for_selection=False,selection_rule='paired Val both identity top1 + ambiguity, old ten-class TP protection, F1/macro floors; no test tuning'))
        for s,c in cases.items():dump(RUN/f'latest_{s}_cases.json',c)
        print('REVIEW '+json.dumps(dict(epoch=epoch,eligible=eligible,paired=ids,old_f1=metrics['val']['f1'],best_epoch=None if best is None else best[1]['epoch'])),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    review(BASE,0)
    cfg=dict(imgsz=1280,epochs=32,patience=32,batch=4,nbs=16,device=0,workers=0,amp=True,optimizer='AdamW',lr0=.00004,lrf=.1,cos_lr=True,warmup_epochs=1,weight_decay=.0005,cls=1.,seed=20261022,deterministic=True,fliplr=0.,flipud=0.,degrees=2.,translate=.10,scale=.15,hsv_h=0.,hsv_s=.025,hsv_v=.08,mosaic=0.,mixup=0.,freeze=5,project='runs/pulseatlas_comparison_transfer_20261022',name='paired_finetune',save_period=4,plots=False)
    dump(RUN/'training_run.json',dict(status='running',pid=os.getpid(),base=str(BASE),config=cfg,started=time.time(),test_used=False))
    model=YOLO(str(BASE))
    def callback(trainer):
        epoch=trainer.epoch+1
        dump(RUN/'progress.json',dict(epoch=epoch,epochs=trainer.epochs,updated=time.time()))
        if epoch==1 or epoch%2==0:review(trainer.last,epoch)
    model.add_callback('on_fit_epoch_end',callback);model.train(data=str(DATA/'data.yaml'),**cfg)
    if best is None:raise RuntimeError('No safe checkpoint, preserve original')
    strip_optimizer(str(RUN/'selected_full.pt'),s=str(RUN/'electrode_comparison.pt'))
    dump(RUN/'training_run.json',dict(status='finished',base=str(BASE),config=cfg,best=best[1],finished=time.time(),test_used=False))
if __name__=='__main__':main()
