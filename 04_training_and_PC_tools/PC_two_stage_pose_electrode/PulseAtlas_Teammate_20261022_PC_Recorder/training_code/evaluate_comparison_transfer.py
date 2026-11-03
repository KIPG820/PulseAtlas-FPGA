"""Frozen old/new regression and temporal USB replay. No test-driven checkpoint selection."""
import os,json,time,hashlib
from pathlib import Path
RUN=Path('artifacts/pulseatlas/comparison_transfer_20261022').resolve()
DATA=Path('datasets/pulseatlas_comparison_transfer_20261022').resolve()
BASE=Path('artifacts/pulseatlas/fusion_transfer_20261021/final/electrode_fusion_transfer.pt').resolve()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    if (RUN/'evaluation/results.json').exists():raise RuntimeError('Frozen evaluation already exists')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(RUN/'ultralytics'))
    import torch,cv2,numpy as np
    from ultralytics import YOLO
    import fusion_transfer_common as common
    from train_comparison_transfer import identity
    torch.set_num_threads(4);cv2.setNumThreads(4);common.DATA=DATA
    state=json.loads((RUN/'training_run.json').read_text());assert state['status']=='finished'
    selection=json.loads((RUN/'selection.json').read_text());weight=RUN/'electrode_comparison.pt'
    (RUN/'evaluation').mkdir(exist_ok=True)
    dump(RUN/'evaluation/selection_frozen.json',dict(selected_epoch=selection['best']['epoch'],sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),test_used_for_selection=False))
    report={};all_cases={}
    for tag,path in [('baseline',BASE),('selected',weight)]:
        model=YOLO(str(path));splits={};all_cases[tag]={}
        for split in ('val','legacy_val','comparison_val','test','legacy_test','comparison_test'):
            cases=common.collect(model,split,[736,1280]);all_cases[tag][split]=cases;splits[split]=common.summarize(cases,split in ('val','test'))
            splits[split]['identity']=identity(cases);dump(RUN/'evaluation'/f'{tag}_{split}_cases.json',cases)
        report[tag]=dict(id=tag,splits=splits)
        del model;torch.cuda.empty_cache()
    result=dict(training=state,baseline=report['baseline'],selected=report['selected'],test_used_for_selection=False,new_video_test_independent_scene=False,scope='old retrospective regression + time-separated portions of one comparison video')
    dump(RUN/'evaluation/results.json',result)
    # Both models on exactly the same freshly decoded held-out frames, including motion between labelled samples.
    models={tag:YOLO(str(path)) for tag,path in [('baseline',BASE),('selected',weight)]}
    source=Path('artifacts/pulseatlas/v4_rl_comparison_20261021/source.mp4')
    cap=cv2.VideoCapture(str(source));writer=cv2.VideoWriter(str(RUN/'evaluation/heldout_replay_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(1920,1080));rows=[];names=common.NAMES
    (RUN/'evaluation/replay_frames').mkdir(exist_ok=True)
    for n,t in enumerate(np.arange(52.,62.,.2)):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,im=cap.read()
        if not ok:break
        halves=[];row=dict(seconds=float(t),predictions={})
        for tag,model in models.items():
            tick=time.perf_counter();boxes=model.predict(im,**common.options([736,1280]))[0].boxes.data.cpu().tolist();ms=(time.perf_counter()-tick)*1000
            row['predictions'][tag]=dict(boxes=boxes,detector_ms=ms);drawn=im.copy()
            for p in boxes:
                b=[round(x) for x in p[:4]];color=(0,255,0) if int(p[5])==7 else (0,220,255) if int(p[5])==2 else (255,180,0)
                cv2.rectangle(drawn,tuple(b[:2]),tuple(b[2:]),color,3);cv2.putText(drawn,f'{names[int(p[5])]} {p[4]:.2f}',(b[0],max(30,b[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            half=cv2.resize(drawn,(960,540));cv2.putText(half,f'{tag} {t:.1f}s',(20,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2);halves.append(half)
        frame=np.zeros((1080,1920,3),np.uint8);frame[270:810,:960]=halves[0];frame[270:810,960:]=halves[1];writer.write(frame);rows.append(row)
        if n%5==0:cv2.imwrite(str(RUN/'evaluation/replay_frames'/f't{t:.1f}.jpg'),frame)
    cap.release();writer.release();dump(RUN/'evaluation/heldout_replay.json',dict(frames=rows,note='50 time-separated decoded frames from 52..62s, not 50 independent scenes; only four individually labelled test frames have scored ground truth. Video encoded at 5FPS for inspection, not speed benchmark.'))
    # Prior actual USB failures are retrospective development regression, never new blind acceptance.
    prior=Path('artifacts/pulseatlas/roi_usb_v3_20261021/source_runtime');snapshots=[]
    from evaluate_pulseatlas import iou
    for folder in sorted((prior/'snapshots').iterdir()):
        if not any(t in folder.name for t in ('220642','220644','220648','220649')):continue
        record=json.loads((folder/'prediction.json').read_text(encoding='utf-8'));reference=record['roi_guard']['raw_predictions'][0]['bbox_xyxy'];im=cv2.imread(str(folder/'raw.jpg'));row=dict(image=folder.name,expected_user_identity='V4',reference_box_from_original_wrong_identity_prediction=reference)
        for tag,model in models.items():
            boxes=model.predict(im,**common.options([736,1280]))[0].boxes.data.cpu().tolist();overlap=sorted([p for p in boxes if iou(p[:4],reference)>=.5],key=lambda p:-p[4]);row[tag]=dict(boxes=boxes,top1_v4=bool(overlap) and int(overlap[0][5])==7,unambiguous_v4=bool(overlap) and all(int(p[5])==7 for p in overlap))
        snapshots.append(row)
    cap=cv2.VideoCapture(str(prior/'sessions/session_20261021_220652_b17d5c/raw.mp4'));previous_video=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        row=dict(index=len(previous_video))
        for tag,model in models.items():row[tag]=model.predict(im,**common.options([736,1280]))[0].boxes.data.cpu().tolist()
        previous_video.append(row)
    cap.release();dump(RUN/'evaluation/prior_usb_development_regression.json',dict(snapshots=snapshots,video=previous_video,scope='retrospective earlier failures; ROI-head experiments already used them. Some hand-only crops from two snapshots are in current negative training. Not independent blind evidence; video class presence alone is not fully annotated accuracy.'))
    print(json.dumps({tag:{s:dict(f1=v['strict']['f1'],tp=v['strict']['tp'],fp=v['strict']['fp'],fn=v['strict']['fn'],identity=v['identity']) for s,v in r['splits'].items()} for tag,r in report.items()},indent=2),flush=True)
if __name__=='__main__':main()
