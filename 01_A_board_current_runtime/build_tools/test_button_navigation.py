"""Offline tests for the four-key PYNQ-Z2 page navigation."""

import sys
import types
import io
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pynq = types.ModuleType("pynq")
pynq.allocate = lambda *args, **kwargs: None
lib = types.ModuleType("pynq.lib")
video = types.ModuleType("pynq.lib.video")
video.VideoMode = object
dma = types.ModuleType("pynq.lib.video.dma")
dma.AxiVDMA = type("AxiVDMA", (), {"MM2SChannel": type("MM2SChannel", (), {})})
sys.modules.update({"pynq": pynq, "pynq.lib": lib, "pynq.lib.video": video,
                    "pynq.lib.video.dma": dma})
monitor = types.ModuleType("ecg_live_monitor")
monitor.ECGLiveLeadMonitor = type("ECGLiveLeadMonitor", (), {})
pipeline = types.ModuleType("ecg_full_pipeline")
pipeline.TYPE_RAW = 1
pipeline.TYPE_LEAD12 = 2
pipeline.TYPE_QRS = 3
sys.modules.update({"ecg_live_monitor": monitor, "ecg_full_pipeline": pipeline})

from pulseatlas_r9_app import (
    PulseAtlasR9App, _home_render_signature, _vision_render_signature,
)


class FakeVisionRuntime:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.toggles = 0

    def start(self):
        self.starts += 1

    def stop(self, **_kwargs):
        self.stops += 1

    def toggle(self):
        self.toggles += 1


def fake_app(page):
    app = object.__new__(PulseAtlasR9App)
    app.page = page
    app.home_focus = 0
    app.focused_lead_index = 1
    app.focused_lead_name = "II 导联"
    app.monitor_running = False
    app.ai_busy = False
    app.ai_backend_mode = "ps"
    app._ai_cached_codes = None
    app.ai_status = "waiting"
    app.ui_error = None
    app._vision_runtime = FakeVisionRuntime()
    app.start_monitor = lambda: setattr(app, "monitor_running", True)
    app.stop_monitor = lambda: setattr(app, "monitor_running", False)
    app._run_ai_analysis = lambda: setattr(app, "analysis_ran", True)
    return app


def main():
    def network_file(path, mode="r", encoding=None):
        if path.replace("\\", "/").endswith("/carrier"):
            return io.StringIO("1")
        raise OSError("not present")

    with mock.patch("pulseatlas_r9_app.os.listdir", return_value=["lo", "eth0"]), \
            mock.patch("builtins.open", side_effect=network_file):
        assert PulseAtlasR9App._has_network_link()  # wired link is a live network
    with mock.patch("pulseatlas_r9_app.glob.glob", return_value=["/dev/video1"]), \
            mock.patch("pulseatlas_r9_app.os.path.exists", return_value=True):
        assert PulseAtlasR9App._has_local_camera()

    vision_state = {
        "vision_frames": 4,
        "vision_status": "人体定位 · ECG markers 10",
        "vision_active": True,
        "vision_camera_connected": True,
        "vision_camera_checked": True,
        "vision_fps": 1.0,
        "vision_inference_ms": 240.0,
        "vision_detected": True,
        "vision_stale": False,
        "vision_marker_count": 10,
        "vision_pose_valid": True,
        "camera_connected": True,
        "wifi_connected": True,
        "buttons_available": True,
    }
    assert _vision_render_signature(vision_state) == _vision_render_signature(dict(vision_state))
    updated_vision_state = dict(vision_state, vision_frames=5)
    assert _vision_render_signature(vision_state) != _vision_render_signature(updated_vision_state)

    home_state = {
        "status": {"id_ok": False, "enabled": False},
        "camera_connected": False,
        "wifi_connected": True,
        "home_focus": 0,
        "buttons_available": True,
    }
    assert _home_render_signature(home_state) == _home_render_signature(dict(home_state))
    assert _home_render_signature(home_state) != _home_render_signature(
        dict(home_state, home_focus=1))
    assert _home_render_signature(home_state) != _home_render_signature(
        dict(home_state, camera_connected=True))

    app = fake_app("home")
    app._handle_button(1)
    assert app.home_focus == 1
    app._handle_button(2)
    assert app.page == "leads"

    app._handle_button(1)
    assert app.focused_lead_index == 2
    app._handle_button(2)
    assert app.page == "lead_focus" and app.focused_lead_name == "III 导联"
    app._handle_button(3)
    assert app.page == "leads"

    app = fake_app("monitor")
    app._handle_button(2)
    assert app.monitor_running
    app._handle_button(2)
    assert not app.monitor_running
    app._handle_button(3)
    assert app.page == "home"

    app = fake_app("ai")
    app._handle_button(1)
    assert app.ai_backend_mode == "pl"
    app._handle_button(0)
    assert app.ai_backend_mode == "ps"
    app._handle_button(2)
    assert app.analysis_ran

    app = fake_app("spectrum")
    app._handle_button(3)
    assert app.page == "monitor"

    app = fake_app("home")
    app._set_page("vision")
    assert app.page == "vision" and app._vision_runtime.starts == 1
    app._handle_button(2)
    assert app._vision_runtime.toggles == 1
    app._handle_button(3)
    assert app.page == "home" and app._vision_runtime.stops == 0
    app._set_page("ai")
    assert app.page == "ai" and app._vision_runtime.stops == 0

    # Landscape TFT taps dispatch to the same page and action methods as HDMI
    # navigation. Coordinate regions match the 320x240 compact layout.
    app = fake_app("home")
    app._on_tft_click(180, 142)
    assert app.page == "leads"
    app._on_tft_click(150, 110)
    assert app.page == "lead_focus" and app.focused_lead_index == 4
    app._on_tft_click(15, 60)
    assert app.page == "leads"
    app._on_tft_click(150, 110)
    assert app.page == "lead_focus"
    app._on_tft_click(250, 85)
    assert app.focused_lead_index == 3
    app._on_tft_click(290, 85)
    assert app.focused_lead_index == 4
    app._on_tft_click(232, 220)
    assert app.page == "ai"
    app._on_tft_click(45, 88)
    assert app.ai_backend_mode == "ps"
    app._on_tft_click(250, 88)
    assert app.analysis_ran

    app = fake_app("monitor")
    app._on_tft_click(190, 90)
    assert app.page == "trend"
    app = fake_app("monitor")
    app._on_tft_click(260, 90)
    assert app.monitor_running
    app = fake_app("vision")
    app._on_tft_click(260, 90)
    assert app._vision_runtime.toggles == 1
    print("four-button navigation tests passed")


if __name__ == "__main__":
    main()
