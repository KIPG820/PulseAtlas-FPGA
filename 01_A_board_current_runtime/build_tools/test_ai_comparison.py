"""Offline tests for isolated 3-run PS/FPGA inference on identical input."""

from collections import deque
import sys
import threading
import time
import types

import numpy as np


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
monitor.ECGLiveLeadMonitor = type("ECGLiveLeadMonitor", (),
                                  {"_consume": lambda self, records: None})
pipeline_module = types.ModuleType("ecg_full_pipeline")
pipeline_module.TYPE_RAW = 1
pipeline_module.TYPE_LEAD12 = 2
pipeline_module.TYPE_QRS = 3
sys.modules.update({"ecg_live_monitor": monitor, "ecg_full_pipeline": pipeline_module})

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulseatlas_r9_app import (AI_FILTER_WARMUP_FRAMES, AI_HISTORY_FRAMES,
                               AI_LOOKBACK_FRAMES, AI_WINDOW_FRAMES,
                               PulseAtlasR9App)
from pulseatlas_lvgl_backend import PulseAtlasLVGLBackend


class FakePipeline:
    lead_off_enabled = True

    def lead_off_ready(self):
        return self.lead_off_enabled

    @staticmethod
    def adc_status():
        # These counters are historical; the latest 10-second window is clean.
        return {"id_ok": True, "running": True, "drop_count": 7,
                "frame_errors": 1, "config_errors": 0}

    @staticmethod
    def pipeline_status():
        # FIFO stalls are not equivalent to lost lead frames.
        return {"packet_lead_backpressure_events": 3}


class FakeNN:
    def __init__(self):
        self.ps_calls = 0
        self.pl_calls = 0
        self.encode_calls = 0
        self.events = []
        self.inputs = []

    def encode_physical_leads(self, samples):
        self.encode_calls += 1
        assert samples.shape == (10000, 12)
        return np.zeros((5000, 12), dtype=np.uint8)

    def infer_ps(self, _codes):
        self.ps_calls += 1
        self.events.append("ps")
        self.inputs.append(np.asarray(_codes).copy())
        return np.asarray([1, 2, 3, 4, 5], dtype=np.uint8), 1.0 + self.ps_calls / 10

    def infer_pl(self, _codes):
        self.pl_calls += 1
        self.events.append("pl")
        self.inputs.append(np.asarray(_codes).copy())
        return np.asarray([1, 2, 3, 4, 5], dtype=np.uint8), 0.2 + self.pl_calls * 0.02

    @staticmethod
    def logits_and_probabilities(_codes):
        return {"NORM": 0.8, "MI": 0.1, "STTC": 0.04, "CD": 0.04, "HYP": 0.02}


def main():
    # AI preview now uses the same 6 s / 600-point samples and display scaling
    # as real-time monitoring; model inference still uses 10,000 full frames.
    preview_state = {
        "leads": np.arange(10000 * 12, dtype=np.int32).reshape(10000, 12),
        "display_half_range": 100000,
    }
    preview = PulseAtlasLVGLBackend._ai_input(preview_state)
    assert preview.shape == (600, 12)
    assert np.array_equal(preview.T, PulseAtlasLVGLBackend._waveforms(preview_state))

    app = object.__new__(PulseAtlasR9App)
    app.ai_busy = False
    app.ai_result = None
    app.ai_error = None
    app.ai_progress = 0
    app.ai_status = "等待开始"
    app.ai_ps_seconds = None
    app.ai_pl_seconds = None
    app.ai_backend_mode = "ps"
    app._ai_active_backend = "ps"
    app._ai_cached_codes = None
    app._ai_cached_samples = None
    app._ai_backend_results = {}
    app._ai_lock = threading.Lock()
    app._ai_cancel = threading.Event()
    app._data_lock = threading.RLock()
    app.lead_history = deque(np.zeros((10000, 12), dtype=np.int32), maxlen=10000)
    app.frame_history = deque(np.arange(10000, dtype=np.uint32), maxlen=10000)
    app.quality_history = deque(np.zeros((10000, 5), dtype=np.uint32), maxlen=10000)
    app._normal_frames_since_start = AI_HISTORY_FRAMES
    app.golden_passed = True
    app.monitor_running = True
    app.pipeline = FakePipeline()
    app.nn = FakeNN()
    app.heart_rate_history = deque([74.0], maxlen=120)
    app.heart_rate = 74.0
    app.camera_connected = True
    app.wifi_connected = True
    app.focused_lead_index = 1
    app.focused_lead_name = "II 导联"
    app.home_focus = 0
    app.buttons_available = True
    app.gain = 1.0
    app.display_half_range = 100000
    app.golden_status = "通过"
    event_order = []
    app._probe_electrodes_for_ai = lambda **_kwargs: event_order.append("contact")

    ai_state = app._snapshot(page="ai")
    assert ai_state["leads"].shape == (6000, 12)
    assert ai_state["frames"].size == 0
    assert ai_state["quality"].size == 0
    assert ai_state["pipeline_status"] == {}
    assert app._snapshot(page="monitor")["leads"].shape == (6000, 12)
    assert app._snapshot(page="home")["leads"].shape == (0, 12)
    full_state = app._snapshot()
    assert full_state["leads"].shape == (10000, 12)
    assert full_state["frames"].shape == (10000,)
    assert full_state["quality"].shape == (10000, 5)

    probe_frames = np.arange(100, dtype=np.uint32)
    probe_quality = np.zeros((100, 5), dtype=np.uint32)
    probe_quality[:, 2] = 1  # DC probe can trip flatline; contact still valid.
    probe_quality[:, 3] = 1  # Other waveform diagnostics are irrelevant here.
    probe_quality[:, 4] = 0xFFF
    assert PulseAtlasR9App._assess_probe_window(
        probe_frames, probe_quality) == "ready"
    probe_quality[50, 0] = 0x01
    assert PulseAtlasR9App._assess_probe_window(
        probe_frames, probe_quality) == "lead_off"
    probe_quality[50, 0] = 0
    probe_frames[50:] += 1
    assert PulseAtlasR9App._assess_probe_window(
        probe_frames, probe_quality) == "gap"
    assert PulseAtlasR9App._assess_probe_window(
        probe_frames[:99], probe_quality[:99]) == "waiting"

    probe_app = object.__new__(PulseAtlasR9App)
    probe_app.monitor_running = True
    probe_app._closed = False
    probe_app._lead_off_probe_active = False
    probe_app._data_lock = threading.RLock()
    probe_app._probe_frame_history = deque(maxlen=1000)
    probe_app._probe_quality_history = deque(maxlen=1000)
    probe_app._ai_cancel = threading.Event()
    probe_app._ai_cancel_reason = None
    probe_app.ai_progress = 0
    probe_app.ai_status = "等待开始"
    probe_app.pipeline = FakePipeline()
    probe_app.rx_blocks = 0
    probe_app._worker_error = None
    probe_calls = []

    def fake_stop_monitor():
        probe_app.monitor_running = False

    def fake_start_monitor(internal_test=False, lead_off_test=False, update_page=True):
        probe_calls.append(lead_off_test)
        probe_app.monitor_running = True
        if lead_off_test:
            test_quality = np.zeros((100, 5), dtype=np.uint32)
            test_quality[:, 2:] = 1  # DC-mode quality flags must not block LOFF.
            probe_app._probe_frame_history.extend(range(100))
            probe_app._probe_quality_history.extend(test_quality.tolist())

    probe_app.stop_monitor = fake_stop_monitor
    probe_app.start_monitor = fake_start_monitor
    probe_app._probe_electrodes_for_ai(timeout_s=0.2)
    assert probe_calls == [True, False]
    assert probe_app.monitor_running and not probe_app._lead_off_probe_active

    wrapped_frames = ((np.arange(10000, dtype=np.uint64) +
                       (0xFFFFFFFF - 5000)) & 0xFFFFFFFF).astype(np.uint32)
    clean_quality = np.zeros((10000, 5), dtype=np.uint32)
    issue, detail = app._assess_ai_window(wrapped_frames, clean_quality)
    assert issue is None and detail["gap_locations"] == 0
    flagged_quality = clean_quality.copy()
    flagged_quality[777, 2] = 1
    issue, detail = app._assess_ai_window(wrapped_frames, flagged_quality)
    assert issue is None and detail["quality_columns"] == [0, 0, 1, 0, 0]
    assert detail["quality_warning_frames"] == 1
    broken_frames = wrapped_frames.copy()
    broken_frames[5000:] += 4
    issue, detail = app._assess_ai_window(broken_frames, clean_quality)
    assert issue == "帧缺口1处" and detail["gap_steps"] == [5]

    # Search the recent history for the newest fully clean window. A transient
    # flag in the newest 10 s must not hide a clean segment immediately before it.
    lookback_frames = np.arange(AI_HISTORY_FRAMES, dtype=np.uint32)
    lookback_quality = np.zeros((AI_HISTORY_FRAMES, 5), dtype=np.uint32)
    lookback_quality[AI_FILTER_WARMUP_FRAMES + AI_WINDOW_FRAMES, 2] = 1
    candidate, issue, detail = app._find_clean_ai_window(
        lookback_frames, lookback_quality, AI_HISTORY_FRAMES)
    assert candidate == AI_HISTORY_FRAMES - AI_WINDOW_FRAMES and issue is None
    assert detail["candidate_age_frames"] == 0

    # Every candidate window crosses one of these two genuine frame gaps.
    no_clean_frames = lookback_frames.copy()
    no_clean_frames[AI_FILTER_WARMUP_FRAMES + AI_WINDOW_FRAMES - 1:] += 1
    no_clean_frames[AI_HISTORY_FRAMES - 2:] += 1
    candidate, issue, _ = app._find_clean_ai_window(
        no_clean_frames, np.zeros((AI_HISTORY_FRAMES, 5), dtype=np.uint32),
        AI_HISTORY_FRAMES)
    assert candidate is None and issue.startswith("帧缺口")

    original_wait_for_clean = app._wait_for_clean_ai_window
    def record_capture_order(**kwargs):
        event_order.append("capture")
        return original_wait_for_clean(**kwargs)
    app._wait_for_clean_ai_window = record_capture_order
    app._run_ai_analysis()
    app._ai_thread.join(timeout=3.0)
    assert not app._ai_thread.is_alive(), "comparison worker did not finish"
    assert event_order == ["capture", "contact"], event_order
    assert app.nn.ps_calls == 3 and app.nn.pl_calls == 0
    assert app.nn.events == ["ps", "ps", "ps"]
    assert app.nn.encode_calls == 1
    assert app.ai_result["ps_seconds"] == 1.2
    assert app.ai_result["pl_seconds"] is None
    assert app.ai_result["speedup"] is None
    assert app.ai_result["same_codes"] is None
    assert app.ai_progress == 100 and not app.ai_busy
    with app._data_lock:
        app.lead_history = deque(np.full((10000, 12), 123, dtype=np.int32),
                                 maxlen=AI_HISTORY_FRAMES)
    assert np.all(app._snapshot(page="ai")["leads"] == 0)

    # Selecting the other engine must reuse the exact encoded 10-second
    # window, without invoking the first engine or starting a second capture.
    app._set_ai_backend_mode("pl")
    app._run_ai_analysis()
    app._ai_thread.join(timeout=3.0)
    assert not app._ai_thread.is_alive()
    assert app.nn.ps_calls == 3 and app.nn.pl_calls == 3
    assert app.nn.encode_calls == 1
    assert app.nn.events == ["ps", "ps", "ps", "pl", "pl", "pl"]
    assert all(np.array_equal(app.nn.inputs[0], data) for data in app.nn.inputs)
    assert event_order == ["capture", "contact", "contact"], event_order
    assert app.ai_result["ps_seconds"] == 1.2
    assert abs(app.ai_result["pl_seconds"] - 0.24) < 1e-12
    assert abs(app.ai_result["speedup"] - 5.0) < 1e-12
    assert app.ai_result["same_codes"] is True
    assert app._snapshot(page="ai")["leads"].shape == (6000, 12)

    app._consume([{"type": 2, "samples": np.zeros(12, dtype=np.int32),
                   "frame": 10000, "lead_off": 1, "saturation": 0,
                   "flatline": 0, "jump": 0, "lead_invalid": 1}])
    assert app.ai_result is None and app._ai_cancel.is_set()
    assert "结果已清除" in app.ai_status
    app._ai_cancel.clear()

    # Startup frames are never used for inference. Require 3,000 frames for
    # filter settling and 10,000 more for the model's exact input window.
    app.pipeline.lead_off_enabled = False
    prewarm_count = AI_FILTER_WARMUP_FRAMES + AI_WINDOW_FRAMES - 1
    with app._data_lock:
        warm_leads = np.full((prewarm_count, 12), 700, dtype=np.int32)
        warm_leads[:AI_FILTER_WARMUP_FRAMES] = -700
        app.lead_history = deque(warm_leads, maxlen=AI_HISTORY_FRAMES)
        app.frame_history = deque(np.arange(prewarm_count, dtype=np.uint32),
                                  maxlen=AI_HISTORY_FRAMES)
        app.quality_history = deque(np.zeros((prewarm_count, 5), dtype=np.uint32),
                                    maxlen=AI_HISTORY_FRAMES)
        app._normal_frames_since_start = prewarm_count
    try:
        app._wait_for_clean_ai_window(timeout_s=0.02)
    except RuntimeError as exc:
        assert "稳定心电 9999/10000 帧" in str(exc)
    else:
        raise AssertionError("startup samples bypassed the settling gate")
    with app._data_lock:
        app.lead_history.append(np.full(12, 700, dtype=np.int32))
        app.frame_history.append(np.uint32(prewarm_count))
        app.quality_history.append(np.zeros(5, dtype=np.uint32))
        app._normal_frames_since_start = prewarm_count + 1
    selected = app._wait_for_clean_ai_window(timeout_s=0.01)
    assert selected.shape == (AI_WINDOW_FRAMES, 12)
    assert np.all(selected == 700), "filter warm-up frames reached inference"
    app.pipeline.lead_off_enabled = True

    detached_quality = np.zeros((10000, 5), dtype=np.uint32)
    detached_quality[333, 0] = 0x01  # CH1P/LL disconnected for one frame
    issue, detail = app._assess_ai_window(wrapped_frames, detached_quality)
    assert issue == "电极接触异常1帧" and detail["quality_columns"][0] == 1

    # Actual frame gaps are never passed into the model, even if a previous
    # clean segment exists elsewhere in the recent-history buffer.
    broken = np.arange(AI_HISTORY_FRAMES, dtype=np.uint32)
    broken[AI_FILTER_WARMUP_FRAMES + AI_WINDOW_FRAMES - 1:] += 1
    broken[AI_HISTORY_FRAMES - 2:] += 1
    with app._data_lock:
        app.lead_history = deque(
            np.full((AI_HISTORY_FRAMES, 12), 123, dtype=np.int32),
            maxlen=AI_HISTORY_FRAMES)
        app.frame_history = deque(broken, maxlen=AI_HISTORY_FRAMES)
        app.quality_history = deque(
            np.zeros((AI_HISTORY_FRAMES, 5), dtype=np.uint32),
            maxlen=AI_HISTORY_FRAMES)
        app._normal_frames_since_start = AI_HISTORY_FRAMES
    try:
        app._wait_for_clean_ai_window(timeout_s=0.02)
    except RuntimeError as exc:
        assert "帧缺口" in str(exc)
    else:
        raise AssertionError("a real frame gap was accepted")

    def new_clean_window():
        time.sleep(0.05)
        with app._data_lock:
            app.frame_history = deque(np.arange(10000, 20000, dtype=np.uint32),
                                      maxlen=10000)
            app.quality_history = deque(np.zeros((10000, 5), dtype=np.uint32),
                                        maxlen=10000)
            app.lead_history = deque(
                np.full((10000, 12), 700, dtype=np.int32), maxlen=AI_HISTORY_FRAMES)
            app._normal_frames_since_start = AI_HISTORY_FRAMES + 10000

    producer = threading.Thread(target=new_clean_window)
    producer.start()
    assert app._wait_for_clean_ai_window(timeout_s=0.6).shape == (10000, 12)
    producer.join()

    def finish_golden_check():
        time.sleep(0.05)
        app.golden_passed = True
        app.golden_status = "通过"

    app.golden_passed = False
    app.golden_status = "自检进行中"
    app._golden_thread = threading.Thread(target=finish_golden_check)
    app._golden_thread.start()
    app._run_ai_analysis()
    app._ai_thread.join(timeout=3.0)
    assert not app._ai_thread.is_alive() and app.ai_error is None
    assert app.nn.ps_calls == 3 and app.nn.pl_calls == 6
    app._golden_thread.join()
    # Electrode checks use separate, discarded acquisition bursts and always
    # restore normal acquisition, whether contact passes or fails.
    app._closed = False
    app.history = deque(maxlen=10000)
    app._lead_off_probe_active = False
    app._probe_frame_history = deque(maxlen=1000)
    app._probe_quality_history = deque(maxlen=1000)
    app._ai_cancel_reason = None
    del app._probe_electrodes_for_ai
    probe_modes = []
    app.stop_monitor = lambda: setattr(app, "monitor_running", False)

    def fake_start_monitor(lead_off_test=False, update_page=True):
        probe_modes.append(bool(lead_off_test))
        app.monitor_running = True
        with app._data_lock:
            app.lead_history.clear()
            app.history.clear()
            app.frame_history.clear()
            app.quality_history.clear()
            if getattr(app, "simulate_no_probe_data", False):
                app._probe_frame_history.clear()
                app._probe_quality_history.clear()
                return
            app._probe_frame_history = deque(np.arange(250, dtype=np.uint32), maxlen=1000)
            q = np.zeros((250, 5), dtype=np.uint32)
            if lead_off_test and getattr(app, "simulate_detached_probe", False):
                q[10, 0] = 1
            app._probe_quality_history = deque(q, maxlen=1000)

    app.start_monitor = fake_start_monitor
    app.simulate_detached_probe = False
    app._probe_electrodes_for_ai(timeout_s=0.01)
    assert probe_modes == [True, False] and app.monitor_running
    assert not app.frame_history and not app.quality_history
    app._lead_off_probe_active = True
    app._consume([{"type": 2, "samples": np.ones(12, dtype=np.int32),
                   "frame": 250, "lead_off": 0, "saturation": 0,
                   "flatline": 0, "jump": 0, "lead_invalid": 0}])
    assert not app.lead_history and not app.frame_history
    assert app._probe_frame_history[-1] == 250
    app._lead_off_probe_active = False
    probe_modes.clear()
    app.simulate_detached_probe = True
    try:
        app._probe_electrodes_for_ai(timeout_s=0.01)
    except RuntimeError as exc:
        assert "接触不良" in str(exc)
    else:
        raise AssertionError("detached electrode probe was accepted")
    assert probe_modes == [True, False] and app.monitor_running
    probe_modes.clear()
    app.simulate_detached_probe = False
    app.simulate_no_probe_data = True
    try:
        app._probe_electrodes_for_ai(timeout_s=0.01)
    except RuntimeError as exc:
        assert "0 / 100 帧" in str(exc)
    else:
        raise AssertionError("empty probe stream did not time out diagnostically")
    assert probe_modes == [True, False] and app.monitor_running
    print("isolated 3-run PS/FPGA same-input tests passed")


if __name__ == "__main__":
    main()
