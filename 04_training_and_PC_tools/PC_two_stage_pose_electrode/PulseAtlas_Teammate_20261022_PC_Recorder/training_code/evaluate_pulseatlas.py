"""Select operational confidence only on val; evaluate the locked test split once."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import os
import numpy as np
import yaml
from pulseatlas_audit import NAMES,sha256


def targets(path,shape):
    h,w=shape[:2]
    result=[]
    for line in Path(path).read_text().splitlines():
        cls,x,y,bw,bh=map(float,line.split())
        result.append([int(cls),(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return result


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def operating_metrics(cases,threshold):
    tp,fp,fn=0,0,0
    per=Counter()
    for boxes,truth in cases:
        matched=set()
        for box in sorted((b for b in boxes if b[4]>=threshold),key=lambda b:-b[4]):
            candidates=[(iou(box[:4],g[1:]),j) for j,g in enumerate(truth) if j not in matched and int(box[5])==g[0]]
            overlap,j=max(candidates,default=(0,None))
            if overlap>=.5:
                tp+=1
                matched.add(j)
                per[(int(box[5]),'tp')]+=1
            else:
                fp+=1
                per[(int(box[5]),'fp')]+=1
        for j,g in enumerate(truth):
            if j not in matched:
                fn+=1
                per[(g[0],'fn')]+=1
    precision=tp/max(1,tp+fp)
    recall=tp/max(1,tp+fn)
    return dict(conf=threshold,tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
        f1=2*precision*recall/max(1e-9,precision+recall),
        per_class={name:{key:per[(c,key)] for key in ('tp','fp','fn')} for c,name in enumerate(NAMES)})


def collect(model,root,split,imgsz,iou_threshold):
    import cv2
    cases=[]
    for image in sorted((root/'images'/split).glob('*.jpg')):
        frame=cv2.imread(str(image))
        result=model.predict(frame,imgsz=imgsz,device=0,conf=.001,iou=iou_threshold,verbose=False)[0]
        truth=targets(root/'labels'/split/(image.stem+'.txt'),frame.shape)
        cases.append((result.boxes.data.cpu().tolist(),truth))
    return cases


def ap_summary(result,ground_truth_counts):
    observed={int(c):i for i,c in enumerate(result.box.ap_class_index)}
    return dict(aggregate={k:float(v) for k,v in result.results_dict.items()},
        per_class={name:dict(ground_truth_boxes=ground_truth_counts.get(name,0),
            ap50=float(result.box.ap50[observed[c]]) if c in observed else None,
            ap50_95=float(result.box.ap[observed[c]]) if c in observed else None)
            for c,name in enumerate(NAMES)},speed_ms=result.speed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--audit',default='artifacts/pulseatlas/audit.json')
    p.add_argument('--output',default='artifacts/pulseatlas/evaluation')
    p.add_argument('--imgsz',type=int,default=960)
    a=p.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path('artifacts/ultralytics').resolve()))
    os.environ.setdefault('MPLCONFIGDIR',str(Path('artifacts/matplotlib').resolve()))
    from ultralytics import YOLO
    audit=json.loads(Path(a.audit).read_text())
    root=Path(audit['dataset_root'])
    output=Path(a.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'test_metrics.json').exists():
        raise RuntimeError('Test already evaluated in this output; do not retune on it')
    model=YOLO(a.weights)
    if dict(model.names)!=dict(enumerate(NAMES)):
        raise RuntimeError('Incorrect detector class mapping')
    data=Path(a.audit).resolve().parent/'data.resolved.yaml'
    val=model.val(data=str(data),split='val',imgsz=a.imgsz,batch=4,workers=0,device=0,
        project=str(output),name='val',plots=True,iou=.6)
    (output/'val_metrics.json').write_text(json.dumps(ap_summary(val,audit['splits']['val']['boxes']),indent=2))
    val_cases=collect(model,root,'val',a.imgsz,.6)
    grid=[operating_metrics(val_cases,float(round(t,2))) for t in np.arange(.05,.951,.05)]
    # Prefer the conventional .25 within an equally good plateau; 19 RL/LL
    # images do not justify pushing a ten-class threshold to an extreme.
    chosen=max(grid,key=lambda row:(row['f1'],row['recall'],-abs(row['conf']-.25)))
    config=dict(conf=chosen['conf'],iou=.6,imgsz=a.imgsz,weights_sha256=sha256(a.weights),
        selected_using='val only: RL/LL, same subject; remaining eight classes not calibrated',
        matching_iou=.5,val_operating_metrics=chosen,frozen_unix=time.time())
    (output/'operating_point.json').write_text(json.dumps(config,indent=2))
    (output/'val_threshold_grid.json').write_text(json.dumps(grid,indent=2))
    # No further model/threshold changes are made after this point.
    test=model.val(data=str(data),split='test',imgsz=a.imgsz,batch=4,workers=0,device=0,
        project=str(output),name='test',plots=True,iou=.6)
    test_summary=ap_summary(test,audit['splits']['test']['boxes'])
    test_summary['operating_metrics']=operating_metrics(collect(model,root,'test',a.imgsz,.6),chosen['conf'])
    test_summary['limitation']='Only V3/V5 on one same-subject held-out video; NOT ten-class or subject-independent validation'
    (output/'test_metrics.json').write_text(json.dumps(test_summary,indent=2))
    print(json.dumps(dict(operating_point=config,test=test_summary),indent=2))


if __name__=='__main__':
    main()
