import os,json,time
from pathlib import Path
ROOT=Path('artifacts/pulseatlas/comparison_transfer_20261022').resolve();OUT=ROOT/'temporal'
def main():
    global OUT
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));args=parser.parse_args();OUT=Path(args.output).resolve()
    OUT.mkdir(exist_ok=True)
    if (OUT/'replay.json').exists():raise RuntimeError('Replay already exists')
    os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'ultralytics'))
    import cv2,torch,numpy as np
    from ultralytics import YOLO
    from temporal_identity import TemporalIdentity
    from fusion_transfer_common import options,NAMES,summarize
    from train_comparison_transfer import identity
    from evaluate_pulseatlas import iou
    torch.set_num_threads(4);cv2.setNumThreads(4);model=YOLO(str(ROOT/'electrode_comparison.pt'));tracker=TemporalIdentity()
    cap=cv2.VideoCapture('artifacts/pulseatlas/v4_rl_comparison_20261021/source.mp4');fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_FRAMES,round(45*fps));index=round(45*fps);rows=[];matched=[]
    truth=json.loads((ROOT/'evaluation/selected_comparison_test_cases.json').read_text());truth={int(Path(r['image']).stem.split('t')[-1]):r for r in truth}
    sink=cv2.VideoWriter(str(OUT/'stream_replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/2,(1920,1080));(OUT/'frames').mkdir(exist_ok=True)
    next_truth={t for t in truth};recorded=set()
    while index/fps<62:
        ok,im=cap.read()
        if not ok:break
        if index%2:
            index+=1;continue
        seconds=index/fps;boxes=model.predict(im,**options([736,1280]))[0].boxes.data.cpu().tolist();out,report=tracker.update(boxes,seconds);rows.append(dict(seconds=seconds,raw=boxes,prediction=out,report=report))
        # Detection on decoded frame nearest labelled second, compared with manually reviewed reference geometry.
        for t in next_truth:
            if t not in recorded and abs(seconds-t)<.045:
                matched.append(dict(image=truth[t]['image'],truth=truth[t]['truth'],prediction=out,raw=boxes,seconds=seconds));recorded.add(t)
        for p in out:
            b=[round(v) for v in p[:4]];color=(0,255,0) if int(p[5])==7 else (0,220,255);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,2);cv2.putText(im,NAMES[int(p[5])],(b[0],max(25,b[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        for p in report['pending']:
            b=[round(v) for v in p[:4]];cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(160,160,160),2);cv2.putText(im,'Pending',(b[0],max(25,b[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.8,(160,160,160),2)
        cv2.putText(im,f'{seconds:.2f}s',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2);sink.write(im)
        for t in truth:
            if abs(seconds-t)<.035:cv2.imwrite(str(OUT/'frames'/f't{t}.jpg'),im)
        index+=1
    cap.release();sink.release()
    report=dict(frames=rows,matched_scored_frames=matched,metrics=summarize(matched),identity=identity(matched),scope='continuous replay with previous decoded frames at ~15fps; current test targets scored at nearest decoded frame, differs from isolated JPG detector test. Test scene already observed, not blind acceptance. Initial warmup pending and gaps reported, no invented positions.')
    (OUT/'replay.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('frames','matched_scored_frames')},indent=2))
if __name__=='__main__':main()
