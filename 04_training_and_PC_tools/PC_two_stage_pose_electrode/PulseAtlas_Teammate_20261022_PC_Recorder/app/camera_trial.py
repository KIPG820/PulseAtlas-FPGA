"""Same-frame unchanged R12 pose and V4/RL paired contrast fine-tuned detector."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
import os, sys, json, hashlib, time, threading, queue, uuid, argparse, traceback
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'runtime'; OUT.mkdir(parents=True,exist_ok=True)
(BASE/'runtime/yolo_config').mkdir(parents=True,exist_ok=True)
(BASE/'runtime/mpl_config').mkdir(parents=True,exist_ok=True)
if os.environ.get('PULSEATLAS_DEPENDENCIES'): sys.path.insert(0,os.environ['PULSEATLAS_DEPENDENCIES'])
os.environ.update({'YOLO_CONFIG_DIR':str(BASE/'runtime/yolo_config'),'MPLCONFIGDIR':str(BASE/'runtime/mpl_config'),
                  'YOLO_AUTOINSTALL':'false','YOLO_OFFLINE':'true','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','WANDB_MODE':'disabled'})
import cv2, torch, ultralytics
import numpy as np
from ultralytics import YOLO, settings
# Resolve standard Python Tcl/Tk resources for a moved/venv executable.
import _tkinter
for env_name,folder_name,marker in [('TCL_LIBRARY','tcl'+_tkinter.TCL_VERSION,'init.tcl'),('TK_LIBRARY','tk'+_tkinter.TK_VERSION,'tk.tcl')]:
    existing=os.environ.get(env_name)
    if not existing or not (Path(existing)/marker).is_file():
        for candidate in [Path(sys.base_prefix)/'tcl'/folder_name,Path(sys.base_prefix)/'lib'/folder_name]:
            if (candidate/marker).is_file():os.environ[env_name]=str(candidate);break
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from placement_matching import StableOverlapMatcher, CONTRACT as MATCH_CONTRACT, MAX_RESULT_AGE_SECONDS, draw_matching
CLASSES=['R','L','N','F','C1','C2','C3','C4','C5','C6']
TZ=timezone(timedelta(hours=8)); NOMINAL_FPS=30
RUNTIME_READY=False
RUNTIME_LOCK=threading.Lock()

def now(): return datetime.now(TZ).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def write_jpeg(path,im):
    ok,data=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok: raise RuntimeError('JPEG编码失败')
    data.tofile(str(path))

def configure():
    global RUNTIME_READY
    with RUNTIME_LOCK:
        if not RUNTIME_READY:
            torch.set_num_threads(8); torch.set_num_interop_threads(2); RUNTIME_READY=True
    settings.update({k:False for k in ['sync','hub','wandb','mlflow','comet','clearml','neptune','dvc','raytune','tensorboard'] if k in settings})
    package=read(BASE/'package_config.json')
    selection=read(BASE/'evaluation/operating_point_selection.json')
    baseline_mode=os.environ.get('PULSEATLAS_PREVIOUS_BASELINE')=='1'
    if baseline_mode:
        package['connector_model']['path']='models/connector_previous.pt'
        package['connector_model']['sha256']=sha(BASE/'models/connector_previous.pt')
        selection['checkpoint_sha256']=package['connector_model']['sha256']
        package['identity_guard']['enabled']=False
        package['training_run']='previous_fusion_baseline'
    weight=BASE/package['connector_model']['path']
    assert sha(weight)==selection['checkpoint_sha256']==package['connector_model']['sha256']
    assert package['connector_model']['confidence']==selection['confidence']==.15 and not selection['test_used_for_selection']
    model=YOLO(str(weight)); assert [model.names[i] for i in range(10)]==CLASSES
    contract={'checkpoint':str(weight),'checkpoint_sha256':sha(weight),'model_version':package['training_run'],'confidence':.15,
              'nms_iou':.5,'imgsz':package['connector_model']['imgsz'],'rect':False,'max_det':50,'device':(package['connector_model'].get('device','0') if torch.cuda.is_available() else 'cpu'),'threads':8,
              'raw_frame_mirrored':False,'annotation_type':'model_prediction_not_ground_truth',
              'torch':torch.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__}
    guard_config=package.get('identity_guard',{})
    guard_enabled=guard_config.get('enabled',False) and os.environ.get('PULSEATLAS_DISABLE_IDENTITY_GUARD')!='1'
    contract['identity_guard']={'enabled':guard_enabled,'independent_usb_acceptance':False}
    if guard_enabled:
        from roi_identity_guard import ROIIdentityGuard
        guard_path=BASE/guard_config['path']
        assert sha(guard_path)==guard_config['sha256'], 'ROI guard checksum mismatch'
        model._pulseatlas_roi_guard=ROIIdentityGuard(model,guard_path,contract['device'])
        contract['identity_guard'].update(checkpoint_sha256=guard_config['sha256'],policy=model._pulseatlas_roi_guard.policy)
    temporal=package.get('temporal_identity',{})
    enabled=temporal.get('enabled',False) and not baseline_mode and os.environ.get('PULSEATLAS_DISABLE_TEMPORAL')!='1'
    contract['temporal_identity']=dict(temporal,enabled=enabled,scope='live worker; current proposal coordinates only; detector predict() itself remains stateless')
    if enabled:
        from temporal_identity import TemporalIdentity
        model._pulseatlas_temporal=TemporalIdentity(temporal['confirm_frames'],temporal['min_confidence'],temporal['max_gap_seconds'])
    return model,contract

def predict(model,raw,contract):
    started=time.perf_counter()
    result=model.predict(raw,imgsz=contract['imgsz'],rect=False,conf=contract['confidence'],iou=contract['nms_iou'],
                         max_det=50,device=contract['device'],verbose=False,save=False,augment=False)[0]
    predictions=[{'class_id':int(b.cls.item()),'lead_id':CLASSES[int(b.cls.item())],
                  'confidence':float(b.conf.item()),'bbox_xyxy':b.xyxy[0].cpu().tolist()} for b in result.boxes]
    guard=getattr(model,'_pulseatlas_roi_guard',None)
    if guard is not None:
        raw_predictions=predictions
        rows=[p['bbox_xyxy']+[p['confidence'],p['class_id']] for p in predictions]
        refined,decisions=guard.refine(raw,rows)
        predictions=[{'class_id':int(p[5]),'lead_id':CLASSES[int(p[5])],'confidence':float(p[4]),'bbox_xyxy':p[:4]} for p in refined]
        model._pulseatlas_guard_report={'raw_predictions':raw_predictions,'decisions':decisions}
    return predictions,(time.perf_counter()-started)*1000


def stabilize_predictions(model,predictions,timestamp):
    tracker=getattr(model,'_pulseatlas_temporal',None)
    if tracker is None:return predictions,{'enabled':False,'pending':[]}
    boxes=[p['bbox_xyxy']+[p['confidence'],p['class_id']] for p in predictions]
    rows,report=tracker.update(boxes,timestamp)
    current=[{'class_id':int(p[5]),'lead_id':CLASSES[int(p[5])],'confidence':float(p[4]),'bbox_xyxy':p[:4]} for p in rows]
    return current,dict(report,enabled=True,raw_predictions=predictions)

def draw_pending(frame,report):
    if not report.get('pending'):return frame
    frame=frame.copy()
    for p in report['pending']:
        x1,y1,x2,y2=[round(v) for v in p[:4]]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(160,160,160),2)
        cv2.putText(frame,'Pending ID',(x1,max(25,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.7,(160,160,160),2)
    return frame

def overlay(raw,predictions,neutral=False):
    im=raw.copy(); h,w=im.shape[:2]
    for p in predictions:
        x1,y1,x2,y2=p['bbox_xyxy']; color=(60,210,60)
        if p['lead_id'] in ['C1','C2','C3','C5']: color=(255,210,60)
        if neutral:color=(150,150,150)
        a=(max(0,min(w-1,int(x1))),max(0,min(h-1,int(y1))))
        b=(max(0,min(w-1,int(x2))),max(0,min(h-1,int(y2))))
        cv2.rectangle(im,a,b,color,2)
        cv2.putText(im,f"{p['lead_id']} {p['confidence']:.2f}",(a[0],max(18,a[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
    return im

class Recorder:
    def __init__(self,contract,camera_index,size,folder=None):
        self.folder=folder or OUT/'sessions'/('session_'+datetime.now(TZ).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.folder.mkdir(parents=True); self.contract=dict(contract)
        self.contract.update(started_at=now(),camera_index=camera_index,raw_size=list(size),nominal_video_fps=NOMINAL_FPS,
                             note='Video uses nominal 30 FPS; exact capture timestamps are in predictions.jsonl. This is a trial, not scored blind-test ground truth.')
        self.writer=cv2.VideoWriter(str(self.folder/'raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),NOMINAL_FPS,size)
        if not self.writer.isOpened():
            self.writer.release()
            self.writer=cv2.VideoWriter(str(self.folder/'raw.avi'),cv2.VideoWriter_fourcc(*'MJPG'),NOMINAL_FPS,size)
        if not self.writer.isOpened(): raise RuntimeError('无法创建本地视频文件，请用保存误例按钮记录单帧')
        self.log=(self.folder/'predictions.jsonl').open('w',encoding='utf-8')
        self.index=0; self.start=time.monotonic(); dump(self.folder/'session_metadata.json',self.contract)
    def write(self,raw,record):
        assert [raw.shape[1],raw.shape[0]]==self.contract['raw_size']
        self.writer.write(raw)
        logged=dict(record,recorded_video_frame_index=self.index,recorded_video_pts_seconds=self.index/NOMINAL_FPS)
        self.log.write(json.dumps(logged,ensure_ascii=False)+'\n'); self.log.flush(); self.index+=1
    def close(self):
        self.writer.release(); self.log.close()
        self.contract.update(ended_at=now(),frames_recorded=self.index,wall_duration_seconds=round(time.monotonic()-self.start,3))
        dump(self.folder/'session_metadata.json',self.contract)


class DemoRenderer:
    """Render the annotated inference result as a stable 720p presentation frame."""
    SIZE=(1280,720)
    LEAD_NAMES={'R':'RA','L':'LA','N':'RL','F':'LL',**{f'C{i}':f'V{i}' for i in range(1,7)}}

    def __init__(self):
        self.fonts={}
        candidates=[Path(r'C:\Windows\Fonts\msyh.ttc'),Path(r'C:\Windows\Fonts\simhei.ttf')]
        self.font_path=next((p for p in candidates if p.is_file()),None)
        self.background=np.full((self.SIZE[1],self.SIZE[0],3),(252,248,245),dtype=np.uint8)
        cv2.rectangle(self.background,(0,0),(1279,78),(91,55,28),-1)
        cv2.rectangle(self.background,(18,92),(892,652),(255,255,255),-1)
        cv2.rectangle(self.background,(910,92),(1262,652),(255,255,255),-1)
        cv2.rectangle(self.background,(18,92),(892,652),(238,226,217),2)
        cv2.rectangle(self.background,(910,92),(1262,652),(238,226,217),2)

    def font(self,size,bold=False):
        key=(size,bold)
        if key not in self.fonts:
            path=Path(r'C:\Windows\Fonts\msyhbd.ttc') if bold else self.font_path
            if not path or not path.is_file(): path=self.font_path
            self.fonts[key]=ImageFont.truetype(str(path),size) if path else ImageFont.load_default()
        return self.fonts[key]

    @staticmethod
    def fit(frame,width,height):
        h,w=frame.shape[:2]; scale=min(width/w,height/h)
        resized=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        canvas=np.full((height,width,3),(38,27,18),dtype=np.uint8)
        y=(height-resized.shape[0])//2; x=(width-resized.shape[1])//2
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        return canvas

    def render(self,drawn,record,recording=False):
        out=self.background.copy()
        out[104:638,30:880]=self.fit(drawn,850,534)
        pil=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(pil)
        d.text((30,17),'PulseAtlas  电极贴片视觉检测',font=self.font(32,True),fill=(255,255,255))
        badge='● 正在录制' if recording else 'PC 推理预览'
        badge_color=(238,84,84) if recording else (79,174,126)
        d.rounded_rectangle((1035,18,1248,61),radius=12,fill=badge_color)
        d.text((1060,25),badge,font=self.font(20,True),fill=(255,255,255))
        d.text((930,111),'检测状态',font=self.font(27,True),fill=(28,55,91))
        fps=record.get('processing_fps')
        fps_text='预热' if fps is None else f'{fps:.1f} FPS'
        d.text((930,153),f'联合处理  {fps_text}',font=self.font(18),fill=(79,98,122))
        d.text((930,181),f"电极模型  {record.get('inference_ms',0):.0f} ms",font=self.font(18),fill=(79,98,122))
        d.text((930,209),f"姿态模型  {record.get('pose_inference_ms',0):.0f} ms",font=self.font(18),fill=(79,98,122))
        predictions=record.get('predictions',[])
        best={}
        for p in predictions:
            lead=self.LEAD_NAMES.get(p.get('lead_id'),p.get('lead_id','?'))
            if lead not in best or p.get('confidence',0)>best[lead]: best[lead]=p.get('confidence',0)
        leads=['RA','LA','RL','LL','V1','V2','V3','V4','V5','V6']
        for i,lead in enumerate(leads):
            col=i%2; row=i//2; x=930+col*155; y=258+row*58
            found=lead in best; color=(31,154,92) if found else (157,169,184)
            d.ellipse((x,y+5,x+15,y+20),fill=color)
            d.text((x+24,y),lead,font=self.font(20,True),fill=(35,52,74))
            d.text((x+67,y+2),f'{best[lead]:.2f}' if found else '--',font=self.font(17),fill=color)
        placement=record.get('placement_result') or {}
        success=placement.get('success_count',0); unknown=placement.get('unknown_count',10)
        d.line((930,558,1240,558),fill=(220,227,237),width=2)
        d.text((930,574),f'位置重合：{success}/10',font=self.font(20,True),fill=(31,154,92) if success else (79,98,122))
        d.text((930,608),f'待确认：{unknown}',font=self.font(17),fill=(112,125,144))
        d.text((24,674),'PC 端推理录制画面 · 用于 PYNQ HDMI 离线回放',font=self.font(18),fill=(82,96,116))
        d.text((1110,675),f"#{record.get('frame_id',0):06d}",font=self.font(16),fill=(126,139,156))
        return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


class DemoRecorder:
    """Write a constant-frame-rate presentation video using capture timestamps."""
    FPS=30

    def __init__(self,contract,camera_index,size=DemoRenderer.SIZE):
        self.folder=OUT/'demo_exports'/('demo_'+datetime.now(TZ).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.folder.mkdir(parents=True)
        self.path=self.folder/'pulseatlas_pc_offline_demo.mp4'
        self.writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'mp4v'),self.FPS,size)
        if not self.writer.isOpened(): raise RuntimeError('无法创建演示视频文件')
        self.log=(self.folder/'source_results.jsonl').open('w',encoding='utf-8')
        self.contract=dict(contract)
        self.contract.update(mode='pc_offline_inference_recording',intended_playback='pynq_hdmi_offline_replay',
                             disclosure_embedded=True,pynq_performs_inference=False,camera_index=camera_index,
                             video_size=list(size),video_fps=self.FPS,started_at=now())
        self.start_timestamp=None; self.frames=0; self.source_frames=0; self.started=time.monotonic()
        dump(self.folder/'demo_metadata.json',self.contract)

    def write(self,frame,record,timestamp):
        if self.start_timestamp is None:self.start_timestamp=timestamp
        target=max(1,int((timestamp-self.start_timestamp)*self.FPS)+1)
        while self.frames<target:
            self.writer.write(frame); self.frames+=1
        self.log.write(json.dumps(dict(record,demo_video_frame_count=self.frames),ensure_ascii=False)+'\n')
        self.log.flush(); self.source_frames+=1

    def close(self):
        self.writer.release(); self.log.close()
        self.contract.update(ended_at=now(),encoded_frames=self.frames,source_inference_frames=self.source_frames,
                             video_duration_seconds=round(self.frames/self.FPS,3),wall_duration_seconds=round(time.monotonic()-self.started,3))
        dump(self.folder/'demo_metadata.json',self.contract)

class App:
    def __init__(self,root,use_pose=True,demo_ui=False):
        self.use_pose=use_pose; self.show_reference_points=True
        self.demo_ui=demo_ui; self.demo_renderer=DemoRenderer() if demo_ui else None
        self.root=root; root.title('PulseAtlas PC 离线推理录制台' if demo_ui else ('PulseAtlas 人体姿态＋扣头联合试运行' if use_pose else 'PulseAtlas 扣头单模型试运行')); root.geometry('1400x900' if demo_ui else '1180x840')
        self.stop_event=threading.Event(); self.events=queue.Queue(); self.commands=queue.Queue(); self.thread=None
        self.current=None; self.latest_packet=None; self.packet_lock=threading.Lock(); self.closed=False; self.snapshot_index=0
        self.camera=tk.IntVar(value=0); self.resolution=tk.StringVar(value='1920×1080'); self.notes=tk.StringVar()
        self.status=tk.StringVar(value='点击“开始摄像头”加载模型并打开设备。'); self.metrics=tk.StringVar(value='等待画面')
        self.detected=tk.StringVar(value='类别：R / L / N / F / C1–C6')
        controls=ttk.Frame(root,padding=10); controls.pack(fill='x')
        ttk.Label(controls,text='摄像头编号').pack(side='left'); self.camera_box=ttk.Spinbox(controls,from_=0,to=10,textvariable=self.camera,width=4); self.camera_box.pack(side='left',padx=6)
        self.size_box=ttk.Combobox(controls,textvariable=self.resolution,values=['1280×720','1920×1080','640×480'],state='readonly',width=12); self.size_box.pack(side='left',padx=6)
        self.start_button=ttk.Button(controls,text='开始摄像头',command=self.start); self.start_button.pack(side='left',padx=6)
        ttk.Button(controls,text='停止摄像头',command=self.stop).pack(side='left',padx=6)
        self.record_button=ttk.Button(controls,text='记录原始数据',command=lambda:self.command('record'),state='disabled'); self.record_button.pack(side='left',padx=6)
        self.demo_button=None
        if demo_ui:
            self.demo_button=ttk.Button(controls,text='录制演示视频',command=lambda:self.command('demo_record'),state='disabled')
            self.demo_button.pack(side='left',padx=6)
        self.snapshot_button=ttk.Button(controls,text='保存当前误例',command=self.snapshot,state='disabled'); self.snapshot_button.pack(side='left',padx=6)
        ttk.Button(controls,text='打开结果目录',command=lambda:os.startfile(str(OUT))).pack(side='right')
        ttk.Label(root,text='对应检测框覆盖预设点：连续3帧匹配成功变绿；按当前预设点判断画面重合，不验证皮肤接触或导线连接。',padding=6).pack(fill='x')
        self.pose_status=tk.StringVar(value='人体姿态：等待画面' if use_pose else '人体姿态：未启用')
        if use_pose:
            pose_controls=ttk.Frame(root,padding=6); pose_controls.pack(fill='x')
            self.reference_checkbox=tk.BooleanVar(value=True)
            ttk.Checkbutton(pose_controls,text='显示旧标定参考点（当前机位需重新核对）',variable=self.reference_checkbox,
                            command=lambda:setattr(self,'show_reference_points',bool(self.reference_checkbox.get()))).pack(side='left')
            ttk.Label(pose_controls,textvariable=self.pose_status).pack(side='left',padx=12)
        self.canvas=tk.Canvas(root,bg='#182029',highlightthickness=0); self.canvas.pack(fill='both',expand=True,padx=10)
        ttk.Label(root,textvariable=self.metrics,padding=6).pack(fill='x')
        ttk.Label(root,textvariable=self.detected,padding=6).pack(fill='x')
        self.placement_status=tk.StringVar(value='匹配：等待画面')
        self.placement_details=tk.StringVar(value='')
        self.placement_label=ttk.Label(root,textvariable=self.placement_status,padding=6);self.placement_label.pack(fill='x')
        ttk.Label(root,textvariable=self.placement_details,padding=6,wraplength=1120).pack(fill='x')
        notes=ttk.Frame(root,padding=6); notes.pack(fill='x'); ttk.Label(notes,text='误例备注（可写实际身份，例如：右侧L被报成C2）').pack(side='left')
        ttk.Entry(notes,textvariable=self.notes).pack(side='left',fill='x',expand=True,padx=6)
        ttk.Label(root,textvariable=self.status,padding=8,wraplength=1120).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close); root.after(40,self.tick)
    def start(self):
        if self.thread and self.thread.is_alive(): return
        try:
            index=int(self.camera.get()); width,height=map(int,self.resolution.get().split('×'))
        except Exception:
            messagebox.showerror('设置错误','摄像头编号需要是整数。'); return
        self.stop_event.clear(); self.commands=queue.Queue(); self.current=None; self.latest_packet=None
        self.start_button.config(state='disabled'); self.camera_box.config(state='disabled'); self.size_box.config(state='disabled')
        self.snapshot_button.config(state='disabled'); self.status.set('正在加载模型和打开摄像头，请稍候…')
        self.thread=threading.Thread(target=self.worker,args=(index,width,height),daemon=True); self.thread.start()
    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set(); self.status.set('正在停止并保存记录…');self.invalidate_display('已请求停止，无法判断')
    def command(self,name):
        if self.thread and self.thread.is_alive(): self.commands.put(name)
    def worker(self,index,width,height):
        cap=None; recorder=None; demo_recorder=None
        try:
            model,contract=configure()
            pose_pipeline=None
            if self.use_pose:
                from pose_pipeline import PosePipeline
                pose_pipeline=PosePipeline(); contract.update(pose_pipeline=pose_pipeline.contract,mode='same_frame_pose_and_connector')
            else: contract['mode']='connector_only'
            contract['placement_matching']=dict(MATCH_CONTRACT,enabled=self.use_pose)
            matcher=StableOverlapMatcher()
            if self.stop_event.is_set(): return
            for backend in [cv2.CAP_DSHOW,cv2.CAP_MSMF]:
                cap=cv2.VideoCapture(index,backend)
                if cap.isOpened(): break
                cap.release()
            if cap is None or not cap.isOpened(): raise RuntimeError(f'无法打开摄像头{index}。关闭Windows相机/会议软件，确认权限，再尝试编号1或2。')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); cap.set(cv2.CAP_PROP_FPS,30)
            self.events.put(('ready','摄像头已开启。先观察类别和框；出现错误时保存当前误例。'))
            count=0; started=time.monotonic(); last_started=started; fps=None
            while not self.stop_event.is_set():
                iteration=time.monotonic(); ok,raw=cap.read(); captured_at=now(); captured_monotonic=time.monotonic()
                if not ok: raise RuntimeError('摄像头读取失败，可能已断开或被其他程序占用')
                count+=1
                while not self.commands.empty():
                    command=self.commands.get_nowait()
                    if command=='record':
                        if recorder:
                            folder=recorder.folder; recorder.close(); recorder=None; self.events.put(('status',f'记录已保存：{folder}'))
                        else:
                            recorder=Recorder(contract,index,(raw.shape[1],raw.shape[0])); self.events.put(('status',f'正在记录原始画面和预测：{recorder.folder}'))
                    elif command=='demo_record':
                        if demo_recorder:
                            folder=demo_recorder.folder; demo_recorder.close(); demo_recorder=None
                            self.events.put(('demo_state',(False,folder))); self.events.put(('status',f'演示视频已保存：{folder}'))
                        else:
                            demo_recorder=DemoRecorder(contract,index); self.events.put(('demo_state',(True,demo_recorder.folder)))
                            self.events.put(('status',f'正在录制带标注的演示视频：{demo_recorder.folder}'))
                predictions,inference_ms=predict(model,raw,contract)
                raw_predictions=predictions
                predictions,temporal_report=stabilize_predictions(model,predictions,captured_monotonic)
                pose_info=None; pose_ms=0.0; pose_drawn=raw
                if pose_pipeline is not None:
                    pose_drawn,pose_info,pose_ms=pose_pipeline.process(raw,self.show_reference_points)
                    pose_info.update(frame_id=count,capture_timestamp=captured_at)
                result_age=time.monotonic()-captured_monotonic
                placement=matcher.update(predictions,pose_info,count,raw.shape[1],raw.shape[0],result_age,captured_monotonic)
                if pose_info is not None:
                    pose_info['placement_verdict']='all_overlap_success' if placement['all_success'] else 'partial_or_unknown_overlap'
                interval=iteration-last_started; last_started=iteration
                if interval>0 and count>1: fps=1/interval if fps is None else .9*fps+.1/interval
                record={'frame_id':count,'capture_timestamp':captured_at,'capture_monotonic_seconds':captured_monotonic,
                        'source_elapsed_seconds':captured_monotonic-started,'width':raw.shape[1],'height':raw.shape[0],
                        'inference_ms':round(inference_ms,2),'pose_inference_ms':round(pose_ms,2),'pose_result':pose_info,
                        'inference_models_same_frame':True,'processing_fps':round(fps,2) if fps else None,
                        'predictions':predictions,'raw_predictions':raw_predictions,'temporal_identity':temporal_report,'roi_guard':getattr(model,'_pulseatlas_guard_report',None),'coordinate_space':'original_unmirrored_camera_pixels',
                        'model_version':contract['model_version'],'checkpoint_sha256':contract['checkpoint_sha256'],
                        'confidence':contract['confidence'],'nms_iou':contract['nms_iou'],'imgsz':contract['imgsz'],
                        'result_age_seconds':round(result_age,3),'placement_result':placement,
                        'recording':bool(recorder),'demo_recording':bool(demo_recorder)}
                drawn=draw_pending(overlay(pose_drawn,predictions,neutral=self.use_pose),temporal_report)
                if self.use_pose: drawn=draw_matching(drawn,placement,self.show_reference_points)
                if recorder: recorder.write(raw,record)
                display=self.demo_renderer.render(drawn,record,bool(demo_recorder)) if self.demo_renderer else drawn
                if demo_recorder: demo_recorder.write(display,record,captured_monotonic)
                with self.packet_lock: self.latest_packet=(raw,display,record,contract)
                self.stop_event.wait(max(0,1/NOMINAL_FPS-(time.monotonic()-iteration)))
        except Exception as exc:
            with (OUT/'app_errors.log').open('a',encoding='utf-8') as h: h.write(now()+'\n'+traceback.format_exc()+'\n')
            self.events.put(('error',str(exc)))
        finally:
            try:
                if recorder: recorder.close(); self.events.put(('status',f'记录已保存：{recorder.folder}'))
                if demo_recorder: demo_recorder.close(); self.events.put(('status',f'演示视频已保存：{demo_recorder.folder}'))
            finally:
                if cap: cap.release()
                self.events.put(('stopped',None))
    def tick(self):
        if self.closed: return
        while not self.events.empty():
            event,value=self.events.get_nowait()
            if event=='ready':
                self.record_button.config(state='normal')
                if self.demo_button:self.demo_button.config(state='normal')
                self.status.set(value)
            elif event=='error': self.status.set(value); messagebox.showerror('摄像头试运行',value)
            elif event=='status': self.status.set(value)
            elif event=='demo_state':
                active,folder=value
                if self.demo_button:self.demo_button.config(text='停止演示录制' if active else '录制演示视频')
            elif event=='stopped':
                self.start_button.config(state='normal'); self.camera_box.config(state='normal'); self.size_box.config(state='readonly'); self.record_button.config(state='disabled')
                if self.demo_button:self.demo_button.config(state='disabled',text='录制演示视频')
                if self.status.get().startswith('正在停止'): self.status.set('摄像头已停止。')
                with self.packet_lock:self.latest_packet=None
                self.invalidate_display('摄像头已停止，无法判断')
        with self.packet_lock: packet=self.latest_packet
        if self.stop_event.is_set():packet=None
        if packet is not None and time.monotonic()-packet[2]['capture_monotonic_seconds']>MAX_RESULT_AGE_SECONDS:
            self.invalidate_display('当前结果已过期，无法判断');packet=None
        if packet is not None and (self.current is None or packet[2]['frame_id']!=self.current[2]['frame_id']):
            self.current=packet; raw,drawn,record,contract=packet
            im=Image.fromarray(cv2.cvtColor(drawn,cv2.COLOR_BGR2RGB)); im.thumbnail((max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())))
            self.photo=ImageTk.PhotoImage(im); self.canvas.delete('all'); self.canvas.create_image(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,image=self.photo)
            fps=record['processing_fps']; fps_text=f'{fps:.1f}' if fps is not None else '预热中'
            mode='演示录制' if record.get('demo_recording') else ('原始数据记录' if record['recording'] else '预览')
            self.metrics.set(f"原图 {record['width']}×{record['height']} | 帧 {record['frame_id']} | 联合处理FPS {fps_text} | 扣头 {record['inference_ms']:.1f} ms | 姿态 {record['pose_inference_ms']:.1f} ms | {mode}")
            if self.use_pose:
                info=record['pose_result']
                if info['status']=='valid_torso':
                    targets=len(info['torso_reference_points'])+len(info['chest_reference_points'])
                    self.pose_status.set(f'人体姿态有效；预设参考点{targets}个')
                else:self.pose_status.set('肩/髋关键点不足；请让肩到髋进入画面；本帧无参考点')
            counts=Counter(p['lead_id'] for p in record['predictions']); duplicate=[name for name,n in counts.items() if n>1]
            listing='  '.join(f'{name}:{counts[name]}' for name in CLASSES if counts[name]) or '未检测到扣头'
            self.detected.set(f"检测框数 {len(record['predictions'])} | {listing}"+(' | 同类重复：'+','.join(duplicate) if duplicate else ''))
            if self.use_pose:
                placement=record['placement_result']
                self.placement_status.set(f"相对当前预设点：匹配成功 {placement['success_count']}/10 ｜需调整 {placement['adjust_count']} ｜无法判断 {placement['unknown_count']}"+(' ｜全部匹配成功' if placement['all_success'] else ''))
                self.placement_label.configure(foreground='#13873b' if placement['success_count'] else '#9b5e14')
                labels={'correct':'成功','adjust':'需调整','unknown':'未知'}
                self.placement_details.set('  '.join(f"{lead}→{item['target_label']}:{labels[item['status']]}" for lead,item in placement['per_lead'].items()))
            else:self.placement_status.set('扣头单模型模式：不判定与预设点的匹配')
            self.snapshot_button.config(state='normal')
        self.root.after(40,self.tick)
    def invalidate_display(self,reason):
        self.current=None;self.snapshot_button.config(state='disabled')
        self.placement_status.set('匹配成功 0/10 ｜'+reason);self.placement_label.configure(foreground='#9b5e14')
        self.placement_details.set('');self.pose_status.set(reason);self.detected.set('当前没有有效的实时判定')
        self.metrics.set('无有效实时输出')
        self.canvas.delete('all');self.canvas.create_text(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,text=reason,fill='white',font=('Microsoft YaHei',18))
    def snapshot(self):
        if self.current is None: return
        raw,drawn,record,contract=self.current; self.snapshot_index+=1
        folder=OUT/'snapshots'/('case_'+datetime.now(TZ).strftime('%Y%m%d_%H%M%S_%f')); folder.mkdir(parents=True)
        write_jpeg(folder/'raw.jpg',raw); write_jpeg(folder/'prediction.jpg',drawn)
        dump(folder/'prediction.json',dict(record,user_note=self.notes.get(),saved_at=now(),contract=contract,annotation_type='model_prediction_not_ground_truth'))
        self.status.set(f'当前显示帧已保存：{folder}；备注与预测一并保存。')
    def close(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.status.set('正在关闭摄像头并保存记录…'); self.root.after(150,self.close); return
        self.closed=True; self.root.destroy()

def check(use_pose=True):
    model,contract=configure()
    import numpy as np
    raw=np.zeros((480,640,3),dtype=np.uint8)
    predictions,ms=predict(model,raw,contract); pose_info=None; pose_ms=0.0
    if use_pose:
        from pose_pipeline import PosePipeline
        pose_pipeline=PosePipeline(); contract.update(pose_pipeline=pose_pipeline.contract,mode='same_frame_pose_and_connector')
        original=raw.copy(); pose_drawn,pose_info,pose_ms=pose_pipeline.process(raw)
        import numpy as np
        assert np.array_equal(original,raw), 'Pose stage modified connector input frame'
        drawn=overlay(pose_drawn,predictions,neutral=True)
        blank=np.zeros((480,640,3),dtype=np.uint8); _,blank_info,_=pose_pipeline.process(blank)
        assert blank_info['pose'] is None and not blank_info['chest_reference_points'] and not blank_info['torso_reference_points']
        from electrode_placement import PersonPose
        keypoints=np.zeros((17,2),dtype=np.float32); keypoints[6]=[220,160]; keypoints[5]=[380,160]; keypoints[11]=[335,360]; keypoints[12]=[265,360]
        synthetic=PersonPose((180,140,420,390),.9,keypoints,np.ones(17,dtype=np.float32))
        reference_image,reference_info,_=pose_pipeline.render(blank,synthetic)
        assert len(reference_info['torso_reference_points'])==4 and len(reference_info['chest_reference_points'])==6
        v2=reference_info['chest_reference_points']['V2']; v4=reference_info['chest_reference_points']['V4']; v3=reference_info['chest_reference_points']['V3']
        assert v3['x_px']==round((v2['x_px']+v4['x_px'])/2) and v3['y_px']==round((v2['y_px']+v4['y_px'])/2)
        from placement_matching import LEAD_TARGET
        reference_points={**reference_info['torso_reference_points'],**reference_info['chest_reference_points']}
        demo_predictions=[]
        for lead,label in LEAD_TARGET.items():
            point=reference_points[label]; x,y=point['x_px'],point['y_px']
            demo_predictions.append({'lead_id':lead,'confidence':.95,'bbox_xyxy':[x-3,y-3,x+3,y+3]})
        demo_matcher=StableOverlapMatcher()
        for frame in [1,2,3]:demo_result=demo_matcher.update(demo_predictions,dict(reference_info,frame_id=frame),frame,640,480)
        assert demo_result['all_success'] and demo_result['success_count']==10
        demo_image=draw_matching(overlay(reference_image,demo_predictions,neutral=True),demo_result)
    else: drawn=overlay(raw,predictions)
    folder=OUT/'offline_check'; folder.mkdir(exist_ok=True); write_jpeg(folder/'prediction.jpg',drawn)
    if use_pose: write_jpeg(folder/'synthetic_reference_overlay.jpg',reference_image)
    if use_pose: write_jpeg(folder/'synthetic_matching_success.jpg',demo_image);dump(folder/'synthetic_matching_success.json',dict(demo_result,synthetic_test_fixture=True))
    contract['placement_matching']=dict(MATCH_CONTRACT,enabled=use_pose)
    matcher=StableOverlapMatcher()
    if pose_info is not None:pose_info.update(frame_id=1,capture_timestamp=now())
    placement=matcher.update(predictions,pose_info,1,raw.shape[1],raw.shape[0])
    recorder=Recorder(dict(contract,offline_check=True),-1,(raw.shape[1],raw.shape[0]),folder/('recording_check_'+uuid.uuid4().hex[:6]))
    record={'frame_id':1,'capture_timestamp':now(),'width':raw.shape[1],'height':raw.shape[0],'predictions':predictions,'roi_guard':getattr(model,'_pulseatlas_guard_report',None),
            'inference_ms':ms,'pose_inference_ms':pose_ms,'pose_result':pose_info,'placement_result':placement,'inference_models_same_frame':True,'offline_check':True}
    for index in range(3):
        current_pose=None if pose_info is None else dict(pose_info,frame_id=index+1)
        match_result=match_frame_for_check(predictions,current_pose,index+1,raw)
        recorder.write(raw,dict(record,frame_id=index+1,pose_result=current_pose,placement_result=match_result))
    recorder.close()
    records=[json.loads(line) for line in (recorder.folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==3
    for item in records:
        assert item['placement_result']['frame_id']==item['frame_id']
        if item['pose_result'] is not None:assert item['pose_result']['frame_id']==item['frame_id']
    path=recorder.folder/'raw.mp4'
    if not path.exists(): path=recorder.folder/'raw.avi'
    cap=cv2.VideoCapture(str(path)); decoded=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        assert frame.shape==raw.shape; decoded+=1
    cap.release(); assert decoded==3
    Tcl=tk.Tcl(); assert Tcl.eval('info patchlevel')
    report={'passed':True,'checked_at':now(),'camera_opened':False,'source_split':'synthetic_blank_runtime_smoke_test','model_sha256':contract['checkpoint_sha256'],
            'conf':contract['confidence'],'original_size':[raw.shape[1],raw.shape[0]],'detections':len(predictions),
            'video_roundtrip_frames':decoded,'snapshot_jpeg_verified':cv2.imdecode(np.fromfile(folder/'prediction.jpg',dtype=np.uint8),cv2.IMREAD_COLOR) is not None,
            'recording_check_folder':str(recorder.folder),'tcl_version':Tcl.eval('info patchlevel'),'combined_models':use_pose,
            'real_frame_pose_status':pose_info['status'] if pose_info else 'not_enabled','pose_inference_ms':pose_ms,
            'blank_has_no_stale_reference_points':True if use_pose else None,
            'synthetic_reference_projection_verified':True if use_pose else None}
    if use_pose: report['pose_contract']=pose_pipeline.contract
    report['placement_matching_contract']=contract['placement_matching']
    report['matching_json_frame_consistency_verified']=True
    report['synthetic_all10_overlap_success_verified']=True if use_pose else None
    dump(folder/'check_report.json',report); print(json.dumps(report,ensure_ascii=True),flush=True)

def match_frame_for_check(predictions,pose_info,frame_id,raw):
    from placement_matching import match_frame
    return match_frame(predictions,pose_info,frame_id,raw.shape[1],raw.shape[0])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); parser.add_argument('--connector-only',action='store_true'); parser.add_argument('--demo-ui',action='store_true'); args=parser.parse_args()
    if args.check: check(use_pose=not args.connector_only)
    else: root=tk.Tk(); App(root,use_pose=not args.connector_only,demo_ui=args.demo_ui); root.mainloop()

if __name__=='__main__': main()
