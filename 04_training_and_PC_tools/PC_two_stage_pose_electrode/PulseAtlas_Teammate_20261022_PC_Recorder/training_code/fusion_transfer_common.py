"""Fixed operating contract for transfer from teammate connector checkpoint."""
from pathlib import Path
from evaluate_pulseatlas import targets,operating_metrics,iou
from train_rebuilt_v2 import macro_f1

NAMES=['RA','LA','RL','LL','V1','V2','V3','V4','V5','V6']
ROOT=Path('artifacts/pulseatlas/fusion_transfer_20261021').resolve()
DATA=Path('datasets/pulseatlas_v3_fullclasses_20261021').resolve()
SOURCE=Path('PulseAtlas_PC_Fusion_20261013_v1').resolve()


def options(shape):return dict(imgsz=shape,rect=False,device=0,half=False,conf=.15,iou=.5,max_det=50,verbose=False)


def collect(model,split,shape):
    import cv2
    result=[]
    for image in sorted((DATA/'images'/split).glob('*.jpg')):
        frame=cv2.imread(str(image));pred=model.predict(frame,**options(shape))[0]
        result.append(dict(image=str(image),prediction=pred.boxes.data.cpu().tolist(),
            truth=targets(DATA/'labels'/split/(image.stem+'.txt'),frame.shape)))
    return result


def summarize(cases,all_classes=False):
    strict=operating_metrics([(row['prediction'],row['truth']) for row in cases],.15)
    if all_classes:strict['macro_f1']=macro_f1(strict)
    matrix=[[0]*11 for _ in range(11)];matched_count=correct=truth_count=0
    for row in cases:
        used=set();truth=row['truth'];truth_count+=len(truth)
        for box in sorted(row['prediction'],key=lambda b:-b[4]):
            # Loose localization diagnostic only: ignores class, IoU .10; NOT the primary metric.
            overlap,index=max([(iou(box[:4],gt[1:]),i) for i,gt in enumerate(truth) if i not in used],default=(0,None))
            predicted=int(box[5])
            if overlap>=.1:
                used.add(index);actual=truth[index][0];matched_count+=1;correct+=actual==predicted
                matrix[actual][predicted]+=1
            else:matrix[10][predicted]+=1
        for index,gt in enumerate(truth):
            if index not in used:matrix[gt[0]][10]+=1
    return dict(strict=strict,identity_diagnostic=dict(agnostic_match_iou=.1,matched=matched_count,
        correct_identity=correct,identity_accuracy_given_loose_match=correct/max(1,matched_count),
        truth_boxes=truth_count,loose_geometry_recall=matched_count/max(1,truth_count),
        confusion_rows_truth_columns_prediction=matrix,names=NAMES+['background'],
        note='IoU .10 is a loose diagnostic for connector-versus-pad box definitions; never replace strict IoU .50 P/R/F1 with it'))


def dual(model):
    internal=summarize(collect(model,'val',[736,1280]),True)['strict']
    legacy=summarize(collect(model,'legacy_val',[736,1280]))['strict']
    a,b=internal['macro_f1'],legacy['f1']
    return dict(internal=internal,legacy_val=legacy,worst=min(a,b),harmonic=2*a*b/max(1e-9,a+b))


def rank(score):return score['worst'],score['harmonic'],score['internal']['macro_f1']
