"""Regression: the DMA drain must be running before ADS1298 starts streaming."""

from collections import deque
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
import types


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
pipeline_module = types.ModuleType("ecg_full_pipeline")
pipeline_module.TYPE_RAW = 1
pipeline_module.TYPE_LEAD12 = 2
pipeline_module.TYPE_QRS = 3
sys.modules.update({"ecg_live_monitor": monitor, "ecg_full_pipeline": pipeline_module})

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pulseatlas_r9_app as module


class FakeChannel:
    def transfer(self, _buffer):
        EVENTS.append("dma_armed")

    def stop(self):
        EVENTS.append("dma_stopped")


class FakePipeline:
    def __init__(self, start_ok=True):
        self.dma = types.SimpleNamespace(recvchannel=FakeChannel())
        self.start_ok = start_ok

    def configure_processing(self):
        EVENTS.append("configured")

    def start_ads1298(self, **_kwargs):
        EVENTS.append("adc_started")
        return self.start_ok

    @staticmethod
    def adc_status():
        return {"running": False}

    @staticmethod
    def stop():
        EVENTS.append("pipeline_stopped")


class FakeThread:
    def __init__(self, target, **_kwargs):
        self.target = target
        self.started = False

    def start(self):
        self.started = True
        EVENTS.append("worker_started")

    def is_alive(self):
        return self.started

    def join(self, timeout=None):
        del timeout
        EVENTS.append("worker_joined")
        self.started = False


EVENTS = []


def make_app(start_ok=True):
    app = object.__new__(module.PulseAtlasR9App)
    app._command_lock = nullcontext()
    app._data_lock = nullcontext()
    app._closed = False
    app.monitor_running = False
    app._worker = None
    app.ai_busy = False
    app.ai_backend_mode = "ps"
    app.lead_history = deque()
    app.raw_lead_history = deque()
    app.frame_history = deque()
    app.quality_history = deque()
    app.heart_rate_history = deque()
    app.history = deque()
    app.pipeline = FakePipeline(start_ok=start_ok)
    app.buffer = object()
    app._prepare_for_run = lambda: EVENTS.append("prepared")
    app._acquire_loop = lambda: None
    app._clear_ai_comparison = lambda: None
    return app


def main():
    original_thread = module.threading.Thread
    module.threading.Thread = FakeThread
    try:
        app = make_app()
        app.start_monitor()
        assert EVENTS.index("dma_armed") < EVENTS.index("worker_started")
        assert EVENTS.index("worker_started") < EVENTS.index("adc_started")
        assert app.monitor_running

        EVENTS.clear()
        failed = make_app(start_ok=False)
        try:
            failed.start_monitor()
        except RuntimeError:
            pass
        else:
            raise AssertionError("ADS1298 startup failure must propagate")
        assert "dma_stopped" in EVENTS
        assert "pipeline_stopped" in EVENTS
        assert "worker_joined" in EVENTS
        assert failed._worker is None
    finally:
        module.threading.Thread = original_thread

    print("monitor DMA startup ordering checks passed")


if __name__ == "__main__":
    main()
