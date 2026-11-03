"""Render representative board pages for desktop visual QA only."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review"
sys.path.insert(0, str(ROOT))

from pulseatlas_board_render import render_page


def fake_leads():
    t = np.arange(10000, dtype=np.float32) / 1000.0
    base = 150*np.sin(2*np.pi*0.25*t)
    phase = np.mod(t, 0.8)
    qrs = 650*np.exp(-0.5*((phase-0.18)/0.012)**2)
    q = -110*np.exp(-0.5*((phase-0.155)/0.009)**2)
    s = -170*np.exp(-0.5*((phase-0.205)/0.014)**2)
    p = 85*np.exp(-0.5*((phase-0.08)/0.025)**2)
    tw = 210*np.exp(-0.5*((phase-0.38)/0.07)**2)
    core = base+p+q+qrs+s+tw+np.random.default_rng(7).normal(0,8,t.size)
    factors=(.75,1.0,.68,-.5,.32,.8,.72,.9,1.12,1.05,.85,.64)
    return np.stack([core*factors[i] + 14*np.sin(2*np.pi*(i+1)*t/13)
                     for i in range(12)],axis=1).astype(np.int32)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    leads=fake_leads()
    state={
        "status":{"id_ok":True,"running":True,"device_id":146,"drop_count":0,
                  "frame_errors":0,"config_errors":0},
        "pipeline_status":{"packet_raw_backpressure_events":0,
            "packet_lead_backpressure_events":0,"packet_qrs_backpressure_events":0},
        "leads":leads,"heart_rate":75.0,
        "heart_rate_history":np.linspace(70,78,30),
        "monitor_running":True,"camera_connected":False,"wifi_connected":True,
        "golden_passed":True,"golden_status":"通过","ai_error":None,
        "ai_result":{"probabilities":{"NORM":.80,"MI":.1,"STTC":.04,"CD":.04,"HYP":.02},
                     "same_codes":True,"ps_seconds":1.32,"pl_seconds":.33,"speedup":4.0},
    }
    for page in ("home","monitor","leads","lead_focus","spectrum","trend","ai","vision","history","settings"):
        render_page(page,state).save(OUT/(page+".png"))
    print("Rendered review pages to {}".format(OUT))


if __name__ == "__main__":
    main()
