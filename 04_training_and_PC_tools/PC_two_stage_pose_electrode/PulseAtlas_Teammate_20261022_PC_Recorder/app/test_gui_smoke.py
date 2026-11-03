"""Hidden UI test: render success and revoke it; never open a camera."""
import time, tkinter as tk
import numpy as np
import camera_trial as app
from placement_matching import LEAD_TARGET, StableOverlapMatcher, draw_matching

def main():
    root=tk.Tk();root.withdraw();window=app.App(root)
    raw=np.zeros((480,640,3),dtype=np.uint8);targets={};pred=[]
    for i,(lead,label) in enumerate(LEAD_TARGET.items()):
        x,y=40+(i%5)*90,80+(i//5)*100
        targets[label]={'x_px':x,'y_px':y};pred.append({'lead_id':lead,'confidence':.95,'bbox_xyxy':[x-8,y-8,x+8,y+8]})
    info={'status':'valid_torso','torso_reference_points':{k:v for k,v in targets.items() if k in ['RA','LA','RL','LL']},
          'chest_reference_points':{k:v for k,v in targets.items() if k.startswith('V')}}
    matcher=StableOverlapMatcher()
    for frame in [1,2,3]:result=matcher.update(pred,dict(info,frame_id=frame),frame,640,480)
    record={'frame_id':3,'capture_monotonic_seconds':time.monotonic(),'processing_fps':15.0,'width':640,'height':480,
            'inference_ms':30.0,'pose_inference_ms':20.0,'recording':False,'predictions':pred,'pose_result':info,'placement_result':result}
    window.latest_packet=(raw,draw_matching(app.overlay(raw,pred,neutral=True),result),record,{})
    window.tick();root.update_idletasks()
    assert '10/10' in window.placement_status.get() and '全部匹配成功' in window.placement_status.get()
    window.invalidate_display('测试：结果过期')
    assert '0/10' in window.placement_status.get() and window.current is None
    assert str(window.snapshot_button.cget('state'))=='disabled'
    root.destroy();print('Hidden GUI success/revocation check passed; camera was never opened.')

if __name__=='__main__':main()
