"""Offline smoke tests for screen hit areas; no PYNQ hardware is required."""

import sys
import threading
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub only the PYNQ import surface so the screen routes can be tested here.
pynq = types.ModuleType("pynq")
pynq.allocate = lambda *args, **kwargs: None
lib = types.ModuleType("pynq.lib")
video = types.ModuleType("pynq.lib.video")
video.VideoMode = object
dma = types.ModuleType("pynq.lib.video.dma")
dma.AxiVDMA = type("AxiVDMA", (), {"MM2SChannel": type("MM2SChannel", (), {})})
sys.modules.update({"pynq":pynq,"pynq.lib":lib,"pynq.lib.video":video,
                    "pynq.lib.video.dma":dma})
monitor = types.ModuleType("ecg_live_monitor")
monitor.ECGLiveLeadMonitor = type("ECGLiveLeadMonitor", (), {})
pipeline = types.ModuleType("ecg_full_pipeline")
pipeline.TYPE_RAW = 1
pipeline.TYPE_LEAD12 = 2
pipeline.TYPE_QRS = 3
sys.modules.update({"ecg_live_monitor":monitor,"ecg_full_pipeline":pipeline})

from pulseatlas_r9_app import PulseAtlasR9App


def fake_app(page):
    app=object.__new__(PulseAtlasR9App)
    app.width=1280
    app.height=720
    app.page=page
    app._data_lock=threading.RLock()
    app._ai_lead_history=[]
    app._ai_frame_history=[]
    app._ai_quality_history=[]
    app._ai_cached_samples=None
    app._ai_backend_results={}
    app.ai_ps_seconds=None
    app.ai_pl_seconds=None
    app.ai_result=None
    app.ai_error=None
    app._normal_frames_since_start=0
    app.monitor_running=False
    app.ai_busy=False
    app.ai_backend_mode="ps"
    app._ai_cached_codes=None
    app.gain=1.0
    app.filter_enabled=True
    app._settings_return_page="home"
    return app


def main():
    app=fake_app("home")
    app._on_click(1040,250)
    assert app.page=="ai", app.page

    app=fake_app("leads")
    app._on_click(640,210)
    assert app.page=="lead_focus" and app.focused_lead_index==1

    app=fake_app("lead_focus")
    app._on_click(80,100)
    assert app.page=="leads"

    app=fake_app("monitor")
    app.start_monitor=lambda: setattr(app,"monitor_running",True)
    app.stop_monitor=lambda: setattr(app,"monitor_running",False)
    app._on_click(1120,610)
    assert app.monitor_running
    app._on_click(1120,610)
    assert not app.monitor_running

    app=fake_app("ai")
    app._on_click(690,120)
    assert app.ai_backend_mode=="ps"
    app._on_click(870,120)
    assert app.ai_backend_mode=="pl"
    called=[]
    app._run_ai_analysis=lambda: called.append(True)
    app._on_click(1120,100)
    assert called==[True]

    vision_actions=[]
    app=fake_app("home")
    app._vision_runtime=types.SimpleNamespace(
        start=lambda: vision_actions.append("start"),
        toggle=lambda: vision_actions.append("toggle"),
    )
    app._on_tft_click(280,220)
    assert app.page=="vision" and vision_actions==["start"]
    app._on_tft_click(260,90)
    assert vision_actions==["start","toggle"]

    app=fake_app("monitor")
    app._on_tft_click(30,190)
    assert not app.filter_enabled
    app._on_tft_click(285,190)
    assert app.gain==1.25
    app._on_tft_click(50,55)
    assert app.page=="home"
    app._on_tft_click(220,184)
    assert app.page=="settings"
    app._on_tft_click(190,143)
    assert app.gain==1.0
    app._on_tft_click(285,143)
    assert app.gain==1.25
    app._on_tft_click(50,55)
    assert app.page=="home"
    app._on_tft_click(50,136)
    assert app.page=="monitor"

    app=fake_app("settings")
    app._on_tft_click(220,112)
    assert not app.filter_enabled
    app._on_tft_click(190,143)
    assert app.gain==0.75
    app._on_tft_click(285,143)
    assert app.gain==1.0

    app=fake_app("monitor")
    app._on_click(60,610)
    assert not app.filter_enabled
    app._on_click(280,610)
    assert app.gain==0.75
    app._on_click(80,125)
    assert app.page=="home"
    app._on_click(900,450)
    assert app.page=="settings"
    app._on_click(955,465)
    assert app.gain==1.0
    app._on_click(80,125)
    assert app.page=="home"
    app._on_click(80,250)
    assert app.page=="monitor"

    print("TFT and HDMI touch-route smoke tests passed")


if __name__=="__main__":
    main()
