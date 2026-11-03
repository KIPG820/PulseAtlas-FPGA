"""Standalone, screen-operated PulseAtlas UI for the R9 PYNQ-Z2 package."""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # Offline desktop interaction smoke tests run on Windows.
    fcntl = None
import glob
import os
import select
import struct
import sys
import threading
import time
from collections import deque
from itertools import islice

import numpy as np
from PIL import ImageDraw

from ecg_live_monitor import ECGLiveLeadMonitor
from ecg_full_pipeline import TYPE_LEAD12, TYPE_QRS
from pulseatlas_board_render import FONTS, render_page
from pulseatlas_history import PulseAtlasHistoryStore
from pulseatlas_lvgl_backend import PulseAtlasLVGLBackend
try:
    from pulseatlas_tft_backend import PulseAtlasTFTBackend
    from pulseatlas_tft_render import HOME_ACTIONS, MAIN_PAGES, render_tft_page
    from pulseatlas_tft_render import rgb565_bytes as tft_rgb565_bytes
    _TFT_IMPORT_ERROR = None
except ImportError as _tft_import_error:  # Older R9 archives predate the TFT mirror.
    PulseAtlasTFTBackend = None
    HOME_ACTIONS = (("monitor", "实时监测"), ("leads", "12 导联"),
                    ("ai", "AI 分析"), ("vision", "视觉分析"),
                    ("history", "历史记录"), ("settings", "设置"))
    MAIN_PAGES = ("home", "monitor", "leads", "spectrum", "ai", "vision")
    render_tft_page = None
    tft_rgb565_bytes = None
    _TFT_IMPORT_ERROR = _tft_import_error
from vision_service_client import VisionServiceClient


# The 0.5 Hz DC blocker has pole 1 - 206/65536 (about 318 samples per
# time constant). Nine time constants are under 2,900 samples; 3,000 frames
# also comfortably exceeds the wavelet's 105-sample startup history.
AI_WINDOW_FRAMES = 10000
AI_FILTER_WARMUP_FRAMES = 3000
AI_LOOKBACK_FRAMES = 10000
AI_HISTORY_FRAMES = (AI_FILTER_WARMUP_FRAMES + AI_WINDOW_FRAMES +
                     AI_LOOKBACK_FRAMES)
AI_BACKEND_LABELS = {"ps": "PS NumPy", "pl": "FPGA + DMA"}


def _vision_render_signature(state):
    """Return the state that can change pixels on the PS-rendered Vision page."""
    keys = (
        "vision_frames", "vision_camera_frames", "vision_preview_frames",
        "vision_status", "vision_result_status", "vision_error", "vision_active",
        "vision_camera_connected", "vision_camera_checked", "vision_fps",
        "vision_camera_fps", "vision_preview_fps", "vision_inference_ms",
        "vision_inference_running", "vision_inference_count", "vision_result_age_ms",
        "vision_confidence", "vision_stale",
        "vision_detected", "vision_marker_count", "vision_pose_valid",
        "camera_connected", "wifi_connected", "buttons_available",
    )
    return tuple(state.get(key) for key in keys)


def _home_render_signature(state):
    """Return the small set of home-page values that can change its pixels."""
    status = state.get("status") or {}
    return (
        bool(status.get("id_ok", False)),
        bool(state.get("camera_connected", False)),
        bool(state.get("wifi_connected", False)),
        int(state.get("home_focus", 0)),
        bool(state.get("buttons_available", False)),
    )


class PulseAtlasR9App(ECGLiveLeadMonitor):
    """HDMI UI with PYNQ-Z2 pushbutton input and R9 capture/inference.

    The monitor base class owns the known-good ECG RX DMA worker and PYNQ
    HDMI frame lifecycle. This class keeps acquisition start/stop separate
    from display lifetime so the same on-board UI survives repeated runs.
    """

    def __init__(self, pipeline, nn, width=1280, height=720,
                 refresh_seconds=0.15):
        super().__init__(pipeline, seconds=6.0, refresh_seconds=refresh_seconds,
                         width=width, height=height)
        self.nn = nn
        self.page = "home"
        self.lead_history = deque(maxlen=AI_HISTORY_FRAMES)
        self.frame_history = deque(maxlen=AI_HISTORY_FRAMES)
        self.quality_history = deque(maxlen=AI_HISTORY_FRAMES)
        self._ai_lead_history = deque(maxlen=AI_HISTORY_FRAMES)
        self._ai_frame_history = deque(maxlen=AI_HISTORY_FRAMES)
        self._ai_quality_history = deque(maxlen=AI_HISTORY_FRAMES)
        self._normal_frames_since_start = 0
        self._lead_off_probe_active = False
        self._probe_frame_history = deque(maxlen=1000)
        self._probe_quality_history = deque(maxlen=1000)
        self.heart_rate_history = deque(maxlen=120)
        # (sample-frame, bpm) pairs preserve the real beat timing.  The old
        # value-only deque remains for compatibility with the report page.
        self.heart_rate_events = deque(maxlen=512)
        self.monitor_running = False
        self.golden_passed = False
        self.golden_status = "自检进行中"
        self.ai_result = None
        self.ai_error = None
        self.ai_progress = 0
        self.ai_backend_mode = "ps"
        self._ai_active_backend = "ps"
        self._ai_cached_codes = None
        self._ai_cached_samples = None
        self._ai_backend_results = {}
        self.ai_status = "模式：PS NumPy · BTN0/1 切换，BTN2 开始"
        self.ai_busy = False
        self.ai_ps_seconds = None
        self.ai_pl_seconds = None
        self.ui_error = None
        self.camera_connected = os.path.exists("/dev/video0")
        self.wifi_connected = self._has_default_route()
        self.focused_lead_index = 1
        self.focused_lead_name = "II 导联"
        self.home_focus = 0
        self.gain = 1.0
        self.filter_enabled = True
        try:
            self._history_store = PulseAtlasHistoryStore()
        except Exception as exc:
            self._history_store = None
            self.ui_error = "历史数据库不可用：{}".format(exc)
        self._history_session_id = None
        self._ai_session_id = None
        self._history_records = []
        self._history_detail = None
        self._history_selected_id = None
        self._history_cache_time = 0.0
        self._ui_stop = threading.Event()
        self._input_stop = threading.Event()
        self._ui_thread = None
        self._input_thread = None
        self._button_thread = None
        self._golden_thread = None
        self._ai_thread = None
        self._ai_lock = threading.Lock()
        self._ai_cancel = threading.Event()
        self._ai_cancel_reason = None
        # TFT full-frame transfers and HDMI rendering can hold the Python GIL
        # long enough for the single RX DMA buffer to miss its re-arm window.
        # AI capture raises this event only while it is locking a strict,
        # contiguous 10-second ECG window.  Both display workers then pause,
        # while the ECG DMA drain thread continues at full priority.
        self._ecg_capture_priority = threading.Event()
        self._button_channel = None
        self.buttons_available = False
        self.button_error = None
        self._hdmi_switch_channel = None
        self._hdmi_switch_candidate = None
        self._hdmi_switch_candidate_since = 0.0
        self._hdmi_switch_warned = False
        self._hdmi_switch_logged = False
        self._hdmi_active = True  # The base monitor opens HDMI during construction.
        self._command_lock = threading.RLock()
        self._pointer = [self.width // 2, self.height // 2]
        self._mouse_enabled = False
        self._input_devices = []
        self._lvgl_backend = None
        self._tft_backend = None
        self._tft_thread = None
        self._tft_stop = threading.Event()
        self._tft_snapshot_lock = threading.Lock()
        self._tft_latest_snapshot = None
        self._tft_snapshot_generation = 0
        self._tft_error = None
        # Vision inference runs on the dedicated B board.  Later A-board UI
        # merges accidentally restored PSVisionRuntime here, which made A look
        # for a local pose ONNX and never connect to B.  Keep the same runtime
        # snapshot API while routing start/stop/status/preview over TCP.
        self._vision_runtime = VisionServiceClient()
        self._vision_ui_fps_started = time.monotonic()
        self._vision_ui_fps_frames = 0
        self._vision_ui_fps = 0.0
        self._vision_ui_last_preview = 0
        self._touch = {"x": 0, "y": 0, "down": False, "was_down": False,
                       "xcode": 0, "ycode": 1, "xmin": 0, "xmax": 65535,
                       "ymin": 0, "ymax": 65535}

    @staticmethod
    def _has_default_route():
        try:
            with open("/proc/net/route", "r", encoding="ascii") as stream:
                return any((lambda parts: len(parts) > 3 and
                            parts[0].startswith(("wlan", "wlp")) and
                            parts[1] == "00000000")(row.split())
                           for row in stream.readlines()[1:])
        except OSError:
            return False

    @staticmethod
    def _load_font(size):
        return FONTS.get(size, True)

    def _consume(self, records):
        leads, frames, qualities, rr_values = [], [], [], []
        probe_active = getattr(self, "_lead_off_probe_active", False)
        for record in records:
            if record["type"] == TYPE_LEAD12:
                frame = int(record["frame"]) & 0xFFFFFFFF
                quality = tuple(int(record[key]) for key in
                                ("lead_off", "saturation", "flatline", "jump", "lead_invalid"))
                if probe_active:
                    frames.append(frame)
                    qualities.append(quality)
                else:
                    leads.append(np.asarray(record["samples"], dtype=np.int32).reshape(12))
                    frames.append(frame)
                    qualities.append(quality)
            elif record["type"] == TYPE_QRS:
                rr_values.append((int(record.get("frame", 0)) & 0xFFFFFFFF,
                                  int(record["rr_samples"])))
        if probe_active:
            if frames:
                with self._data_lock:
                    self._probe_frame_history.extend(frames)
                    self._probe_quality_history.extend(qualities)
            # Probe records are intentionally withheld from the plot, rate
            # history, AI window, and the base monitor's display buffers.
            return super()._consume([])
        if leads:
            with self._data_lock:
                self.lead_history.extend(leads)
                self.frame_history.extend(frames)
                self.quality_history.extend(qualities)
                if getattr(self, "filter_enabled", True):
                    if hasattr(self, "_ai_lead_history"):
                        self._ai_lead_history.extend(leads)
                        self._ai_frame_history.extend(frames)
                        self._ai_quality_history.extend(qualities)
                    self._normal_frames_since_start += len(frames)
            if any(row[0] for row in qualities):
                # Never leave a previous classification visible after a lead
                # disconnects, including during a running comparison.
                self._ai_cancel_reason = "电极接触异常，推理已取消。"
                self._ai_cancel.set()
                self.ai_result = None
                self._clear_ai_comparison()
                if not self.ai_busy:
                    self.ai_status = "电极接触异常，推理结果已清除"
        if (leads and getattr(self, "_history_store", None) is not None and
                getattr(self, "_history_session_id", None) is not None):
            try:
                self._history_store.append_samples(
                    self._history_session_id,
                    np.asarray(leads, dtype=np.int32).reshape(-1, 12),
                    np.asarray(frames, dtype=np.uint32),
                    np.asarray(qualities, dtype=np.uint8).reshape(-1, 5),
                    np.asarray(rr_values, dtype=np.uint32).reshape(-1, 2)
                    if rr_values else (),
                )
            except Exception as exc:
                self.ui_error = "历史记录写入失败：{}".format(exc)
        super()._consume(records)
        if rr_values:
            with self._data_lock:
                for frame, rr in rr_values:
                    if 200 <= rr <= 3000:
                        bpm = 60000.0 / rr
                        self.heart_rate_history.append(bpm)
                        if not hasattr(self, "heart_rate_events"):
                            self.heart_rate_events = deque(maxlen=512)
                        self.heart_rate_events.append((frame, bpm))

    def _snapshot(self, page=None):
        # The HDMI LVGL path needs only data visible on its current page.
        # Keep the full snapshot for the offline PIL renderer.
        with self._data_lock:
            if page == "leads":
                start = max(0, len(self.lead_history) - 10000)
                lead_rows = tuple(islice(self.lead_history, start, None))
                frame_rows = ()
            elif page in ("monitor", "lead_focus", "ai"):
                start = max(0, len(self.lead_history) - 6000)
                lead_rows = tuple(islice(self.lead_history, start, None))
                frame_rows = ()
            elif page == "spectrum":
                start = max(0, len(self.lead_history) - 10000)
                lead_rows = tuple(islice(self.lead_history, start, None))
                frame_rows = tuple(islice(self.frame_history, start, None))
            elif page is None:
                lead_rows = tuple(self.lead_history)
                frame_rows = tuple(self.frame_history)
            else:
                lead_rows = ()
                frame_rows = ()
            quality_rows = tuple(self.quality_history) if page is None else ()
            rate_rows = tuple(self.heart_rate_history) if page in (None, "trend", "history") else ()
            heart_rate_events_source = getattr(self, "heart_rate_events", ())
            rate_event_rows = (tuple(heart_rate_events_source)
                               if page in (None, "trend", "history") else ())
            current_frame = (int(self.frame_history[-1]) if self.frame_history
                             else (int(heart_rate_events_source[-1][0])
                                   if heart_rate_events_source else None))
            heart_rate = self.heart_rate
            if page == "ai" and self._ai_cached_samples is not None:
                lead_rows = tuple(self._ai_cached_samples[-6000:])
        leads = (np.asarray(lead_rows, dtype=np.int32).reshape(-1, 12)
                 if lead_rows else np.empty((0, 12), dtype=np.int32))
        frames = np.asarray(frame_rows, dtype=np.uint32)
        quality = (np.asarray(quality_rows, dtype=np.uint32).reshape(-1, 5)
                   if quality_rows else np.empty((0, 5), dtype=np.uint32))
        heart_rates = np.asarray(rate_rows, dtype=np.float64)
        heart_rate_events = (np.asarray(rate_event_rows, dtype=np.float64).reshape(-1, 2)
                             if rate_event_rows else np.empty((0, 2), dtype=np.float64))
        try:
            status = self.pipeline.adc_status()
            stage = self.pipeline.pipeline_status() if page is None else {}
        except Exception as exc:
            status, stage = {}, {}
            self.ui_error = str(exc)
        vision_runtime = getattr(self, "_vision_runtime", None)
        vision_state = vision_runtime.snapshot() if vision_runtime is not None else {
            "vision_active": False,
            "vision_camera_connected": False,
            "vision_camera_checked": False,
            "vision_frame_bgr": None,
            "vision_camera_frames": 0,
            "vision_camera_fps": 0.0,
            "vision_preview_frames": 0,
            "vision_preview_fps": 0.0,
            "vision_status": "按 BTN2 启动视觉识别",
            "vision_error": None,
            "vision_fps": 0.0,
            "vision_inference_ms": None,
            "vision_inference_running": False,
            "vision_inference_count": 0,
            "vision_result_age_ms": None,
            "vision_result_status": "等待推理",
            "vision_confidence": None,
            "vision_stale": False,
            "vision_detected": False,
            "vision_frames": 0,
            "vision_marker_count": 0,
            "vision_pose_valid": False,
        }
        camera_connected = (
            vision_state["vision_camera_connected"]
            if vision_state.get("vision_camera_checked", False)
            else self.camera_connected
        )
        visible_page = page or getattr(self, "page", "home")
        if visible_page == "history":
            self._refresh_history_cache()
        return {
            "status": status,
            "pipeline_status": stage,
            "leads": leads,
            "frames": frames,
            "quality": quality,
            "heart_rate_history": heart_rates,
            "heart_rate_events": heart_rate_events,
            "current_frame": current_frame,
            "sample_rate": float(getattr(self.pipeline, "sample_rate", 1000.0)),
            "heart_rate": heart_rate if np.isfinite(heart_rate) else None,
            "monitor_running": self.monitor_running,
            "ai_result": self.ai_result,
            "ai_error": self.ai_error,
            "ai_progress": self.ai_progress,
            "ai_status": self.ai_status,
            "ai_busy": self.ai_busy,
            "ai_ps_seconds": self.ai_ps_seconds,
            "ai_pl_seconds": self.ai_pl_seconds,
            "ai_backend_mode": self.ai_backend_mode,
            "golden_passed": self.golden_passed,
            "golden_status": self.golden_status,
            "camera_connected": camera_connected,
            **vision_state,
            "wifi_connected": self.wifi_connected,
            "focused_lead_index": self.focused_lead_index,
            "focused_lead_name": self.focused_lead_name,
            "home_focus": self.home_focus,
            "buttons_available": self.buttons_available,
            "gain": self.gain,
            "display_half_range": self.display_half_range,
            "filter_enabled": bool(getattr(self, "filter_enabled", True)),
            "history_records": list(getattr(self, "_history_records", ())),
            "history_detail": getattr(self, "_history_detail", None),
            "history_selected_id": getattr(self, "_history_selected_id", None),
        }

    def _refresh_history_cache(self, force=False):
        now = time.monotonic()
        if not force and now - getattr(self, "_history_cache_time", 0.0) < 1.0:
            return
        store = getattr(self, "_history_store", None)
        if store is None:
            self._history_records, self._history_detail = [], None
            return
        try:
            rows = store.list_recent(4)
            available = {int(row["session_id"]) for row in rows}
            selected = getattr(self, "_history_selected_id", None)
            if selected not in available:
                selected = int(rows[0]["session_id"]) if rows else None
            detail = store.get_detail(selected, lead_index=1) if selected else None
            self._history_records = rows
            self._history_selected_id = selected
            self._history_detail = detail
            self._history_cache_time = now
        except Exception as exc:
            self.ui_error = "历史数据库读取失败：{}".format(exc)

    def _render(self, source=None):
        state = self._snapshot()
        image = render_page(self.page, state, self.width, self.height)
        if self._mouse_enabled:
            draw = ImageDraw.Draw(image)
            x, y = self._pointer
            draw.ellipse((x-8, y-8, x+8, y+8), fill=(255,255,255), outline=(37,79,130), width=2)
            draw.ellipse((x-2, y-2, x+2, y+2), fill=(37,79,130))
        return image

    def start(self, run_golden_check=True):
        """Start the HDMI UI and input listener; ECG capture remains stopped."""
        if self._closed:
            raise RuntimeError("app is shut down; construct a fresh PulseAtlasR9App")
        self._ui_stop.clear()
        self._input_stop.clear()
        # Subscribe immediately for B-board connectivity/status without
        # starting its camera. Entering the Vision page promotes the same
        # connection to active inference, so USB/NET state is live on Home.
        self._vision_runtime.start(active=False)
        self._start_tft_mirror()
        if self._ui_thread is None or not self._ui_thread.is_alive():
            self._ui_thread = threading.Thread(target=self._display_loop,
                                               name="pulseatlas-hdmi", daemon=True)
            self._ui_thread.start()
        self._input_thread = threading.Thread(target=self._input_loop,
                                              name="pulseatlas-input", daemon=True)
        self._input_thread.start()
        self._button_thread = threading.Thread(target=self._button_loop,
                                                name="pulseatlas-buttons", daemon=True)
        self._button_thread.start()
        if run_golden_check:
            self._golden_thread = threading.Thread(target=self._run_golden_check,
                                                   name="pulseatlas-nn-selftest", daemon=True)
            self._golden_thread.start()
        return self

    def _start_tft_mirror(self):
        """Attach to the existing R9 IOP; never load a different overlay here."""
        if PulseAtlasTFTBackend is None:
            self._tft_error = "TFT 镜像模块未随软件包安装：{}".format(_TFT_IMPORT_ERROR)
            print("PulseAtlas TFT mirror unavailable: {}".format(self._tft_error),
                  file=sys.stderr, flush=True)
            return
        overlay = getattr(self.pipeline, "overlay", None)
        if overlay is None or not hasattr(overlay, "ARDUINO"):
            self._tft_error = "当前 R9 bitstream 尚未集成 Arduino IOP，TFT 镜像未启动"
            print("PulseAtlas TFT mirror unavailable: {}".format(self._tft_error),
                  file=sys.stderr, flush=True)
            return

        # Releases are flattened into one board directory, while the source
        # checkout keeps the IOP binary at the repository root.
        module_root = os.path.dirname(os.path.abspath(__file__))
        repository_root = os.path.abspath(
            os.path.join(module_root, "..", "..", "..")
        )
        binary_candidates = [
            os.path.join(module_root, "tools", "tft_pulseatlas_iop.bin"),
            os.path.join(repository_root, "tools", "tft_pulseatlas_iop.bin"),
        ]
        default_binary = next(
            (path for path in binary_candidates if os.path.isfile(path)),
            binary_candidates[0],
        )
        binary = os.environ.get("PULSEATLAS_TFT_IOP_BIN", default_binary)
        if not os.path.isfile(binary):
            self._tft_error = "TFT IOP 程序不存在：{}".format(binary)
            print("PulseAtlas TFT mirror unavailable: {}".format(self._tft_error),
                  file=sys.stderr, flush=True)
            return
        try:
            self._tft_backend = PulseAtlasTFTBackend(
                overlay, binary,
                rotation=int(os.environ.get("PULSEATLAS_TFT_ROTATION", "270")),
            )
            self._tft_stop.clear()
            self._tft_thread = threading.Thread(
                target=self._tft_mirror_loop, name="pulseatlas-tft", daemon=True
            )
            self._tft_thread.start()
            self._tft_error = None
            print("PulseAtlas TFT mirror active: landscape 320x240, shared HDMI page/state.",
                  flush=True)
        except Exception as exc:
            self._tft_backend = None
            self._tft_error = str(exc)
            print("PulseAtlas TFT mirror initialization failed: {}".format(exc),
                  file=sys.stderr, flush=True)

    def _publish_tft_snapshot(self, page, state):
        if self._tft_backend is None:
            return
        with self._tft_snapshot_lock:
            self._tft_latest_snapshot = (page, state)
            self._tft_snapshot_generation += 1

    def _tft_mirror_loop(self):
        """Render the latest HDMI state, then deliver taps to the same actions."""
        generation = -1
        last_frame = None
        last_render_time = 0.0
        last_page = None
        first_frame_logged = False
        first_vision_frame_logged = False
        measured_since = time.monotonic()
        measured_frames = 0
        measured_transfer_seconds = 0.0
        measured_render_seconds = 0.0
        measured_backend_seconds = 0.0
        measured_pixels = 0
        while not self._tft_stop.is_set():
            capture_priority = getattr(self, "_ecg_capture_priority", None)
            if capture_priority is not None and capture_priority.is_set():
                self._tft_stop.wait(0.02)
                continue
            latest = None
            with self._tft_snapshot_lock:
                if self._tft_snapshot_generation != generation:
                    generation = self._tft_snapshot_generation
                    latest = self._tft_latest_snapshot
            try:
                now = time.monotonic()
                if latest is not None and (
                        latest[0] != last_page or now - last_render_time >=
                        (0.12 if not self._hdmi_active else 0.25)):
                    page, state = latest
                    render_started = time.monotonic()
                    image = render_tft_page(page, state, 320, 240)
                    frame = tft_rgb565_bytes(image)
                    render_seconds = time.monotonic() - render_started
                    if frame != last_frame:
                        present_started = time.monotonic()
                        self._tft_backend.present(image)
                        present_seconds = time.monotonic() - present_started
                        last_frame = frame
                        measured_frames += 1
                        measured_transfer_seconds += self._tft_backend.last_transfer_seconds
                        measured_render_seconds += render_seconds
                        measured_backend_seconds += present_seconds
                        measured_pixels += self._tft_backend.last_transfer_pixels
                        if not first_frame_logged:
                            first_frame_logged = True
                            print(
                                "PulseAtlas TFT first frame transfer OK: page={}, 320x240.".format(page),
                                flush=True,
                            )
                    if page != last_page:
                        print(
                            "PulseAtlas TFT page synchronized: {} -> {}.".format(
                                last_page or "startup", page
                            ),
                            flush=True,
                        )
                    if (page == "vision" and
                            state.get("vision_frame_bgr") is not None and
                            not first_vision_frame_logged):
                        first_vision_frame_logged = True
                        print(
                            "PulseAtlas TFT vision preview transfer OK: camera_frames={}.".format(
                                int(state.get("vision_camera_frames") or 0)
                            ),
                            flush=True,
                        )
                    last_page = page
                    last_render_time = time.monotonic()
                    elapsed_measure = last_render_time - measured_since
                    if elapsed_measure >= 5.0:
                        if measured_frames >= 3:
                            print("PulseAtlas TFT page={} mode={} {:.2f} changed fps, "
                                  "mean render {:.3f}s, backend {:.3f}s "
                                  "(transfer {:.3f}s), "
                                  "mean {} pixels".format(
                                      page, "TFT" if not self._hdmi_active else "dual",
                                      measured_frames / elapsed_measure,
                                      measured_render_seconds / measured_frames,
                                      measured_backend_seconds / measured_frames,
                                      measured_transfer_seconds / measured_frames,
                                      round(measured_pixels / measured_frames)), flush=True)
                        measured_since = last_render_time
                        measured_frames = 0
                        measured_render_seconds = 0.0
                        measured_backend_seconds = 0.0
                        measured_transfer_seconds = 0.0
                        measured_pixels = 0
                touch = self._tft_backend.poll_touch()
                if touch is not None:
                    track, x, y = touch
                    page_before = self.page
                    self._on_tft_click(x, y)
                    print(
                        "PulseAtlas TFT touch: track={} x={} y={} page={} -> {}".format(
                            track, x, y, page_before, self.page
                        ),
                        flush=True,
                    )
            except Exception as exc:
                self._tft_error = str(exc)
                print("PulseAtlas TFT mirror stopped after I/O error: {}".format(exc),
                      file=sys.stderr, flush=True)
                try:
                    self._tft_backend.close()
                except Exception:
                    pass
                self._tft_backend = None
                return
            self._tft_stop.wait(0.03)

    def _set_page(self, page):
        previous = self.page
        self.page = page
        if page == "vision" and previous != "vision":
            self._vision_runtime.start()
        # Keep an explicitly started camera worker alive across page changes.
        # This lets PS vision coexist with ECG capture/FPGA inference. Users
        # can stop it from the Vision page; shutdown always releases it.

    def _toggle_vision(self):
        self._vision_runtime.toggle()

    def _button_input(self):
        """Return the four-bit PYNQ AXI GPIO input channel, if present."""
        if self._button_channel is not None:
            return self._button_channel
        overlay = getattr(self.pipeline, "overlay", None)
        gpio = getattr(overlay, "btns_gpio", None)
        if gpio is None:
            raise RuntimeError("R9 overlay has no btns_gpio; load the button-enabled bitstream")
        channel = gpio.channel1
        channel.setlength(4)
        channel.setdirection("in")
        self._button_channel = channel
        self.buttons_available = True
        self.button_error = None
        return channel

    def _button_loop(self):
        """Poll BTN0..BTN3 with a stable-state debounce and rising-edge events."""
        try:
            channel = self._button_input()
            stable = candidate = int(channel.read()) & 0x0F
        except Exception as exc:
            self.buttons_available = False
            self.button_error = str(exc)
            self.ui_error = "物理按键不可用：{}".format(exc)
            return

        candidate_since = time.monotonic()
        debounce_seconds = 0.045
        while not self._input_stop.is_set():
            try:
                raw = int(channel.read()) & 0x0F
            except Exception as exc:
                self.buttons_available = False
                self.button_error = str(exc)
                self._input_stop.wait(0.25)
                continue
            now = time.monotonic()
            if raw != candidate:
                candidate = raw
                candidate_since = now
            elif candidate != stable and now - candidate_since >= debounce_seconds:
                pressed = candidate & ~stable
                stable = candidate
                for index in range(4):
                    if pressed & (1 << index):
                        self._handle_button(index)
            self._input_stop.wait(0.01)

    def _handle_button(self, index):
        """BTN0 previous, BTN1 next, BTN2 select/action, BTN3 back."""
        if index in (0, 1):
            self._navigate(-1 if index == 0 else 1)
        elif index == 2:
            self._activate_selection()
        elif index == 3:
            self._go_back()

    def _navigate(self, step):
        home_pages = ("monitor", "leads", "ai", "vision", "history", "settings")
        main_pages = ("monitor", "leads", "spectrum", "trend", "ai",
                      "vision", "history", "settings")
        if self.page == "home":
            self.home_focus = (self.home_focus + step) % len(home_pages)
        elif self.page == "leads":
            self.focused_lead_index = (self.focused_lead_index + step) % 12
            self._set_focused_lead_name()
        elif self.page == "lead_focus":
            self.focused_lead_index = (self.focused_lead_index + step) % 12
            self._set_focused_lead_name()
        elif self.page == "ai":
            self._set_ai_backend_mode("ps" if step < 0 else "pl")
        elif self.page in main_pages:
            self._set_page(main_pages[(main_pages.index(self.page) + step) % len(main_pages)])

    def _set_focused_lead_name(self):
        names = ("I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6")
        self.focused_lead_name = names[self.focused_lead_index] + " 导联"

    def _activate_selection(self):
        if self.page == "home":
            pages = ("monitor", "leads", "ai", "vision", "history", "settings")
            self._set_page(pages[self.home_focus])
        elif self.page == "monitor":
            try:
                self.stop_monitor() if self.monitor_running else self.start_monitor()
                self.ui_error = None
            except Exception as exc:
                self.ui_error = "采集控制失败：{}".format(exc)
        elif self.page == "leads":
            self._set_focused_lead_name()
            self.page = "lead_focus"
        elif self.page == "ai":
            self._run_ai_analysis()
        elif self.page == "vision":
            self._toggle_vision()

    def _clear_ai_comparison(self):
        """Discard cached input and per-backend results as one comparison unit."""
        self._ai_cached_codes = None
        self._ai_cached_samples = None
        self._ai_backend_results = {}
        self.ai_ps_seconds = None
        self.ai_pl_seconds = None

    def _set_ai_backend_mode(self, backend):
        if backend not in AI_BACKEND_LABELS or self.ai_busy:
            return
        self.ai_backend_mode = backend
        label = AI_BACKEND_LABELS[backend]
        self.ai_status = "模式：{} · BTN2 开始".format(label)

    def _clear_ai_window_history(self):
        """Reset only classifier input history while keeping ECG capture live."""
        with self._data_lock:
            if hasattr(self, "_ai_lead_history"):
                self._ai_lead_history.clear()
                self._ai_frame_history.clear()
                self._ai_quality_history.clear()
            self._normal_frames_since_start = 0
        self._clear_ai_comparison()
        self.ai_result = None
        self.ai_error = None

    def _set_filter_enabled(self, enabled, from_user=True):
        enabled = bool(enabled)
        if getattr(self, "ai_busy", False) and from_user:
            return False
        if enabled == bool(getattr(self, "filter_enabled", True)):
            return True
        try:
            if getattr(self, "monitor_running", False):
                # CTRL[2] bypasses the filter. Restoring the normal path also
                # clears old IIR/FIR state, without restarting ADC or DMA.
                self.pipeline.filter.write(0x08, 0x03 if enabled else 0x05)
                if enabled:
                    time.sleep(0.002)
                    self.pipeline.filter.write(0x08, 0x01)
        except Exception as exc:
            self.ui_error = "滤波设置失败：{}".format(exc)
            return False
        self.filter_enabled = enabled
        self._clear_ai_window_history()
        if not enabled:
            self.ai_status = "滤波已关闭；AI 分析前将自动恢复标准滤波"
        elif getattr(self, "monitor_running", False):
            self.ai_status = "标准滤波已恢复；正在等待滤波稳定"
        return True

    def _adjust_gain(self, direction):
        levels = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
        current = float(getattr(self, "gain", 1.0))
        index = min(range(len(levels)), key=lambda i: abs(levels[i] - current))
        index = max(0, min(len(levels) - 1,
                           index + (1 if direction > 0 else -1)))
        self.gain = levels[index]
        self.display_half_range = 100000.0 / self.gain

    def _select_history_record(self, index):
        rows = getattr(self, "_history_records", ())
        if 0 <= int(index) < len(rows):
            self._history_selected_id = int(rows[int(index)]["session_id"])
            self._refresh_history_cache(force=True)

    def _go_back(self):
        if self.page == "home":
            return
        if self.page == "lead_focus":
            self._set_page("leads")
        elif self.page in ("spectrum", "trend"):
            self._set_page("monitor")
        else:
            self._set_page("home")

    def _requested_hdmi_mode(self):
        """Read SW0; debounce changes without delaying TFT or ECG workers."""
        override = os.environ.get("PULSEATLAS_HDMI_MODE", "switch").lower()
        if override in ("on", "off"):
            return override == "on"
        if self._hdmi_switch_channel is None:
            gpio = getattr(self.overlay, "sws_gpio", None)
            if gpio is None:
                if not self._hdmi_switch_warned:
                    print("PulseAtlas SW0 unavailable in overlay; HDMI remains on.",
                          file=sys.stderr, flush=True)
                    self._hdmi_switch_warned = True
                return True
            channel = gpio.channel1
            channel.setlength(2)
            channel.setdirection("in")
            self._hdmi_switch_channel = channel
        try:
            requested = bool(int(self._hdmi_switch_channel.read()) & 1)
        except Exception as exc:
            if not self._hdmi_switch_warned:
                print("PulseAtlas SW0 read failed: {}; retaining HDMI mode.".format(exc),
                      file=sys.stderr, flush=True)
                self._hdmi_switch_warned = True
            return self._hdmi_active
        if not self._hdmi_switch_logged:
            print("PulseAtlas SW0 GPIO active: initial level {}.".format(
                int(requested)), flush=True)
            self._hdmi_switch_logged = True
        now = time.monotonic()
        if requested != self._hdmi_switch_candidate:
            self._hdmi_switch_candidate = requested
            self._hdmi_switch_candidate_since = now
        if now - self._hdmi_switch_candidate_since < 0.05:
            return self._hdmi_active
        return requested

    def _set_hdmi_mode(self, enabled):
        if enabled == self._hdmi_active:
            return False
        if not enabled:
            # Keep the HDMI VDMA and pixel clock running across SW0 changes.
            # Closing HDMI here and reconfiguring it later can leave scanout
            # stopped even though HDMIOut.start() returns without an error.
            # A black frame gives TFT-only mode a deterministic blank output
            # while preserving a ready-to-use HDMI path for the next toggle.
            frame = self.hdmi.newframe()
            frame.fill(0)
            self.hdmi.writeframe(frame)
        self._hdmi_active = enabled
        print("PulseAtlas SW0 display mode: {}".format(
            "HDMI + TFT" if enabled else "TFT only"), flush=True)
        return True

    def _display_loop(self):
        backend = None
        interval = max(0.05, min(self.refresh_seconds, 0.10))
        last_render = time.monotonic()
        last_vision_signature = None
        last_home_signature = None
        last_rendered_page = None
        while not self._ui_stop.is_set():
            capture_priority = getattr(self, "_ecg_capture_priority", None)
            if capture_priority is not None and capture_priority.is_set():
                self._ui_stop.wait(0.02)
                continue
            page = self.page
            page_changed = page != last_rendered_page
            if page_changed:
                last_rendered_page = page
                last_vision_signature = None
                last_home_signature = None
            try:
                state = self._snapshot(page=page)
                self._publish_tft_snapshot(page, state)
                switched = self._set_hdmi_mode(self._requested_hdmi_mode())
                if not self._hdmi_active:
                    # TFT has its own renderer and IOP transfer thread. Keep
                    # publishing snapshots, but skip the 1280x720 render/VDMA.
                    last_rendered_page = None
                    last_render = time.monotonic()
                    self._ui_stop.wait(interval)
                    continue
                if backend is None:
                    backend = PulseAtlasLVGLBackend(
                        width=self.width, height=self.height)
                    self._lvgl_backend = backend
                    switched = True
                if switched:
                    last_vision_signature = None
                    last_home_signature = None
                if page == "vision":
                    # Keep the live camera view on the PS UI path. The ECG
                    # overlay renderer and its PL/DMA state remain untouched.
                    signature = _vision_render_signature(state)
                    if signature == last_vision_signature:
                        frame_bgr = None
                    else:
                        rgb = np.asarray(render_page(
                            page, state, self.width, self.height
                        ).convert("RGB"))
                        frame_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                        last_vision_signature = signature
                elif page == "home":
                    signature = _home_render_signature(state)
                    if signature == last_home_signature:
                        frame_bgr = None
                    else:
                        frame_bgr = backend.render(page, state)
                        last_home_signature = signature
                else:
                    last_home_signature = None
                    frame_bgr = backend.render(page, state)
                if page == "ai" and not self.ai_busy:
                    selected_backend = getattr(backend, "selected_ai_backend", None)
                    if selected_backend in AI_BACKEND_LABELS and selected_backend != self.ai_backend_mode:
                        self._set_ai_backend_mode(selected_backend)
                if frame_bgr is not None:
                    frame = self.hdmi.newframe()
                    frame[:] = frame_bgr
                    self.hdmi.writeframe(frame)
                    if page == "vision":
                        preview_sequence = int(state.get("vision_preview_frames") or 0)
                        if preview_sequence != self._vision_ui_last_preview:
                            self._vision_ui_last_preview = preview_sequence
                            self._vision_ui_fps_frames += 1
                            elapsed_ui = time.monotonic() - self._vision_ui_fps_started
                            if elapsed_ui >= 1.0:
                                self._vision_ui_fps = self._vision_ui_fps_frames / elapsed_ui
                                self._vision_ui_fps_frames = 0
                                self._vision_ui_fps_started = time.monotonic()
                                self._vision_runtime._publish(
                                    vision_preview_fps=self._vision_ui_fps
                                )
            except Exception as exc:
                self.ui_error = str(exc)
                print("PulseAtlas LVGL frame error: {}".format(exc), file=sys.stderr, flush=True)
            elapsed = time.monotonic() - last_render
            last_render = time.monotonic()
            page_interval = interval
            if page == "vision":
                # Camera preview and PS inference have separate cadences. Do not
                # let a multi-second model pass throttle the live HDMI preview.
                page_interval = min(interval, 0.05)
            self._ui_stop.wait(max(0.0, page_interval - elapsed))

    def _run_golden_check(self):
        self.golden_status = "自检进行中"
        try:
            rows = self.nn.check_golden_vectors()
            self.golden_passed = bool(rows) and all(row.get("pass", False) for row in rows)
            self.golden_status = "通过" if self.golden_passed else "失败：向量不匹配"
        except Exception as exc:
            self.golden_passed = False
            self.golden_status = "失败：{}".format(str(exc)[:60])

    def start_monitor(self, internal_test=False, lead_off_test=False,
                      update_page=True):
        """Arm fresh RX DMA, configure R9 processing and begin acquisition."""
        with self._command_lock:
            if self._closed:
                raise RuntimeError("app is shut down")
            if self.monitor_running:
                return
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous ECG DMA worker is still stopping")
            self._stop_event = threading.Event()
            self._worker_error = None
            with self._data_lock:
                self.lead_history.clear()
                self.frame_history.clear()
                self.quality_history.clear()
                self._ai_lead_history.clear()
                self._ai_frame_history.clear()
                self._ai_quality_history.clear()
                self._normal_frames_since_start = 0
                self.heart_rate_history.clear()
                if hasattr(self, "heart_rate_events"):
                    self.heart_rate_events.clear()
                self.history.clear()
                self.heart_rate = float("nan")
                self.last_qrs_frame = None
            if not self.ai_busy:
                self.ai_result = None
                self.ai_error = None
                self.ai_progress = 0
                self._clear_ai_comparison()
                self.ai_status = "模式：{} · BTN2 开始".format(
                    AI_BACKEND_LABELS[self.ai_backend_mode])
            try:
                self._prepare_for_run()
                self.pipeline.configure_processing()
                self.pipeline.dma.recvchannel.transfer(self.buffer)
                if not self.pipeline.start_ads1298(
                        internal_test=internal_test, lead_off=lead_off_test):
                    raise RuntimeError("ADS1298 initialization failed: {}".format(
                        self.pipeline.adc_status()))
                if not self.filter_enabled:
                    self.pipeline.filter.write(0x08, 0x05)
                try:
                    if self._history_store is None:
                        raise RuntimeError("历史数据库未初始化")
                    self._history_session_id = self._history_store.start_session(
                        self.sample_rate, lead_count=12)
                    self._refresh_history_cache(force=True)
                except Exception as exc:
                    self._history_session_id = None
                    self.ui_error = "历史记录无法启动：{}".format(exc)
                self._worker = threading.Thread(target=self._acquire_loop,
                                                name="ecg-dma-drain", daemon=True)
                self.monitor_running = True
                self._worker.start()
                if update_page:
                    self.page = "monitor"
            except Exception:
                self._stop_event.set()
                try:
                    self.pipeline.dma.recvchannel.stop()
                except Exception:
                    pass
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                raise

    def stop_monitor(self):
        """Stop only acquisition; keep HDMI output and the UI alive."""
        with self._command_lock:
            if not self.monitor_running and (self._worker is None or not self._worker.is_alive()):
                return
            self._stop_event.set()
            try:
                self.pipeline.dma.recvchannel.stop()
            except Exception:
                pass
            try:
                self.pipeline.stop()
            finally:
                if self._worker is not None:
                    self._worker.join(timeout=2.0)
                    if self._worker.is_alive():
                        raise RuntimeError("ECG DMA worker did not stop")
                self._worker = None
                self.monitor_running = False
                history_id = self._history_session_id
                self._history_session_id = None
                if history_id is not None and self._history_store is not None:
                    try:
                        status = "complete" if getattr(self, "_worker_error", None) is None else "incomplete"
                        self._history_store.finish_session(history_id, status=status)
                        self._refresh_history_cache(force=True)
                    except Exception as exc:
                        self.ui_error = "历史记录未完整保存：{}".format(exc)
                self.ai_result = None
                if not self.ai_busy:
                    self._clear_ai_comparison()

    def _run_ai_analysis(self):
        with self._ai_lock:
            if self.ai_busy:
                return
            # After both engines have completed, the next request starts a
            # fresh comparison window. Before that, either engine reuses the
            # exact encoded input from the first successful run.
            if len(getattr(self, "_ai_backend_results", {})) == 2:
                self._clear_ai_comparison()
            if not getattr(self, "filter_enabled", True):
                if not self._set_filter_enabled(True, from_user=False):
                    self.ai_error = "AI 分析要求开启标准滤波。"
                    self.ai_status = self.ai_error
                    return
            self._ai_session_id = getattr(self, "_history_session_id", None)
            self._ai_cancel.clear()
            self._ai_cancel_reason = None
            self.ai_result = None
            self.ai_error = None
            self.ai_progress = 0
            self._ai_active_backend = self.ai_backend_mode
            label = AI_BACKEND_LABELS[self._ai_active_backend]
            if self._ai_cached_codes is None:
                self.ai_status = "{}：检查稳定心电数据…".format(label)
                self.ai_ps_seconds = None
                self.ai_pl_seconds = None
            else:
                self.ai_status = "{}：准备推理…".format(label)
            self.ai_busy = True
            self._ai_thread = threading.Thread(
                target=self._run_ai_analysis_worker,
                name="pulseatlas-ai-{}".format(self._ai_active_backend), daemon=True,
            )
            self._ai_thread.start()

    @staticmethod
    def _find_clean_ai_window(frame_rows, quality_rows, total_frames):
        """Find the newest valid 10-second window in the recent live history.

        Keep the neural-network's exact 10,000-sample input and reject every
        actual frame gap or quality flag. Searching a short lookback avoids
        rejecting good data just because a newer transient bad frame entered
        the rolling window. Startup settling is based on each candidate's
        absolute position in the current normal-mode acquisition.
        """
        if len(frame_rows) != len(quality_rows):
            return None, "数据标记未同步", {"frames": len(frame_rows),
                                             "quality": len(quality_rows)}
        if len(frame_rows) < AI_WINDOW_FRAMES:
            return None, "有效数据不足", {"frames": len(frame_rows)}
        frames = np.asarray(frame_rows, dtype=np.uint32)
        quality = np.asarray(quality_rows, dtype=np.uint32)
        if quality.shape != (len(frames), 5):
            return None, "数据标记格式异常", {"quality_shape": quality.shape}

        history_start = max(0, int(total_frames) - len(frames))
        first_eligible = max(0, AI_FILTER_WARMUP_FRAMES - history_start)
        last_start = len(frames) - AI_WINDOW_FRAMES
        eligible_frames = max(0, int(total_frames) - AI_FILTER_WARMUP_FRAMES)
        if first_eligible > last_start:
            return None, "稳定心电 {}/{} 帧".format(
                min(eligible_frames, AI_WINDOW_FRAMES), AI_WINDOW_FRAMES), {
                    "eligible_frames": eligible_frames,
                    "warmup_frames": AI_FILTER_WARMUP_FRAMES,
                    "window_frames": AI_WINDOW_FRAMES,
                }

        # Only electrode-off blocks a candidate.  Saturation, flatline, jump
        # and derived-invalid are masks from the unfiltered physical channels;
        # keep them as diagnostics for the filtered 12-lead result instead of
        # making a single asserted bit permanently deadlock inference.
        bad_quality = (quality[:, 0] != 0).astype(np.int64)
        quality_prefix = np.concatenate(([0], np.cumsum(bad_quality)))
        steps = ((frames[1:].astype(np.uint64) -
                  frames[:-1].astype(np.uint64)) & 0xFFFFFFFF)
        bad_edges = (steps != 1).astype(np.int64)
        gap_prefix = np.concatenate(([0], np.cumsum(bad_edges)))
        starts = np.arange(first_eligible, last_start + 1, dtype=np.int64)
        quality_counts = (quality_prefix[starts + AI_WINDOW_FRAMES] -
                          quality_prefix[starts])
        gap_counts = (gap_prefix[starts + AI_WINDOW_FRAMES - 1] -
                      gap_prefix[starts])
        valid = starts[(quality_counts == 0) & (gap_counts == 0)]
        if valid.size:
            candidate_ends = valid + AI_WINDOW_FRAMES - 1
            ages = ((int(frames[-1]) - frames[candidate_ends].astype(np.uint64)) &
                    0xFFFFFFFF)
            fresh = ages <= AI_LOOKBACK_FRAMES
            valid = valid[fresh]
            ages = ages[fresh]
        if valid.size:
            start = int(valid[-1])
            age_frames = int(ages[-1])
            selected_quality = quality[start:start + AI_WINDOW_FRAMES]
            warning_frames = int(np.count_nonzero(
                np.any(selected_quality[:, 1:] != 0, axis=1)))
            return start, None, {"candidate_age_frames": age_frames,
                                 "searched_frames": len(frames) - first_eligible,
                                 "candidate_count": int(valid.size),
                                 "quality_warning_frames": warning_frames}

        issue, detail = PulseAtlasR9App._assess_ai_window(
            frames[-AI_WINDOW_FRAMES:], quality[-AI_WINDOW_FRAMES:])
        detail.update({"candidate_count": 0,
                       "lookback_frames": AI_LOOKBACK_FRAMES,
                       "eligible_frames": eligible_frames})
        return None, issue or "等待连续稳定心电", detail

    def _wait_for_clean_ai_window(self, timeout_s=30.0):
        """Wait for the newest clean, contiguous 10-second input window.

        The normal capture is not restarted when AI is pressed. We retain a
        10-second lookback and choose the newest clean window from it, while
        still rejecting every real frame gap and every hardware quality flag.
        ADC counters are reported as per-attempt deltas, never as a cumulative
        proxy for the selected lead-frame window.
        """
        deadline = time.monotonic() + timeout_s
        initial_status = self.pipeline.adc_status()
        initial_stage = self.pipeline.pipeline_status()
        initial_rx_blocks = getattr(self, "rx_blocks", 0)
        last_problem = "采集不足"
        last_window = {}
        while time.monotonic() < deadline:
            if self._ai_cancel.is_set():
                raise InterruptedError(self._ai_cancel_reason or "推理已取消。")
            if not self.monitor_running:
                raise RuntimeError("请先开始实时监测，再进行 AI 分析。")
            worker_error = getattr(self, "_worker_error", None)
            if worker_error is not None:
                raise RuntimeError("ECG 接收线程已停止：{}".format(worker_error))

            with self._data_lock:
                normal_frames = int(getattr(self, "_normal_frames_since_start", 0))
                frame_source = getattr(self, "_ai_frame_history", self.frame_history)
                quality_source = getattr(self, "_ai_quality_history", self.quality_history)
                frame_rows = tuple(frame_source)
                quality_rows = tuple(quality_source)
            candidate, last_problem, last_window = self._find_clean_ai_window(
                frame_rows, quality_rows, normal_frames)
            if candidate is None:
                settled_frames = max(0, normal_frames - AI_FILTER_WARMUP_FRAMES)
                self.ai_progress = min(
                    5, int(5 * min(settled_frames, AI_WINDOW_FRAMES) /
                           AI_WINDOW_FRAMES))
                if normal_frames < AI_FILTER_WARMUP_FRAMES:
                    self.ai_status = "等待滤波稳定：{}/{} 帧".format(
                        normal_frames, AI_FILTER_WARMUP_FRAMES)
                elif normal_frames < AI_FILTER_WARMUP_FRAMES + AI_WINDOW_FRAMES:
                    self.ai_status = "采集稳定数据：{}/{} 帧".format(
                        min(settled_frames, AI_WINDOW_FRAMES), AI_WINDOW_FRAMES)
                else:
                    self.ai_status = "寻找连续稳定的 10 秒数据：{}".format(last_problem)
                self._ai_cancel.wait(min(0.25, max(0.0, deadline - time.monotonic())))
                continue

            # Recheck one atomic snapshot. The candidate index and all three
            # data streams must still refer to exactly the same frame rows.
            with self._data_lock:
                normal_frames = int(getattr(self, "_normal_frames_since_start", 0))
                frame_source = getattr(self, "_ai_frame_history", self.frame_history)
                quality_source = getattr(self, "_ai_quality_history", self.quality_history)
                lead_source = getattr(self, "_ai_lead_history", self.lead_history)
                frame_rows = tuple(frame_source)
                quality_rows = tuple(quality_source)
                lead_rows = tuple(lead_source)
            if len(lead_rows) != len(frame_rows):
                candidate, last_problem, last_window = (
                    None, "12 导联与帧标记未同步", {
                        "lead_rows": len(lead_rows), "frame_rows": len(frame_rows)})
            else:
                candidate, last_problem, last_window = self._find_clean_ai_window(
                    frame_rows, quality_rows, normal_frames)
            if candidate is not None:
                samples = np.asarray(
                    lead_rows[candidate:candidate + AI_WINDOW_FRAMES], dtype=np.int32)
                if samples.shape == (AI_WINDOW_FRAMES, 12):
                    status = self.pipeline.adc_status()
                    if not status["id_ok"] or not status.get("running", True):
                        raise RuntimeError("ECG 采集设备未正常运行。")
                    last_window["selected_window_age_seconds"] = round(
                        last_window.get("candidate_age_frames", 0) /
                        float(getattr(self.pipeline, "sample_rate", 1000)), 2)
                    final_status = self.pipeline.adc_status()
                    final_stage = self.pipeline.pipeline_status()
                    last_window["adc_drop_delta"] = (
                        (int(final_status.get("drop_count", 0)) -
                         int(initial_status.get("drop_count", 0))) & 0xFFFFFFFF)
                    last_window["raw_backpressure_delta"] = (
                        (int(final_stage.get("packet_raw_backpressure_events", 0)) -
                         int(initial_stage.get("packet_raw_backpressure_events", 0))) &
                        0xFFFFFFFF)
                    last_window["lead_backpressure_delta"] = (
                        (int(final_stage.get("packet_lead_backpressure_events", 0)) -
                         int(initial_stage.get("packet_lead_backpressure_events", 0))) &
                        0xFFFFFFFF)
                    print("PulseAtlas AI selected valid ECG window: {}".format(
                        last_window), file=sys.stderr, flush=True)
                    self._ai_window_quality_warning = int(
                        last_window.get("quality_warning_frames", 0))
                    return samples
                last_problem = "12 导联数据与帧标记未同步"
                last_window = {"lead_rows": len(lead_rows),
                               "frame_rows": len(frame_rows),
                               "candidate_start": candidate}
            self.ai_status = "等待稳定心电：{}".format(last_problem)
            self._ai_cancel.wait(min(0.25, max(0.0, deadline - time.monotonic())))

        status = self.pipeline.adc_status()
        stage = self.pipeline.pipeline_status()
        counter_keys = (
            "quality_frame_count", "filter_frame_count", "wavelet_frame_count",
            "lead12_frame_count", "packet_raw_backpressure_events",
            "packet_lead_backpressure_events", "packet_qrs_backpressure_events",
            "packet_record_count", "packet_block_count")
        counters = {key: stage.get(key) for key in counter_keys}
        deltas = {key: ((int(stage[key]) - int(initial_stage[key])) & 0xFFFFFFFF)
                  for key in counter_keys if key in stage and key in initial_stage}
        adc_deltas = {key: ((int(status[key]) - int(initial_status[key])) & 0xFFFFFFFF)
                      for key in ("frame_seq", "frame_count", "drop_count", "frame_errors")
                      if key in status and key in initial_status}
        print("PulseAtlas AI window timeout: issue={}, window={}, ADC delta={}, ADC status={}, pipeline delta={}, pipeline counters={}, RX block delta={}, RX blocks={}, RX words={}, worker error={}".format(
            last_problem, last_window, adc_deltas, status, deltas, counters,
            getattr(self, "rx_blocks", 0) - initial_rx_blocks,
            getattr(self, "rx_blocks", "unknown"), getattr(self, "rx_words", "unknown"),
            getattr(self, "_worker_error", None)),
            file=sys.stderr, flush=True)
        new_adc_drops = adc_deltas.get("drop_count", "unknown")
        raise RuntimeError(
            "未找到连续、无质量异常的 10 秒心电窗口（回看最多 {} 秒）；"
            "最新窗口：{}；本次新增 ADC 掉帧：{}。".format(
                AI_LOOKBACK_FRAMES // 1000, last_problem, new_adc_drops))

    def _probe_electrodes_for_ai(self, timeout_s=10.0, required_frames=100):
        """Briefly enable ADS1298 lead-off, discard probe samples, then restore ECG."""
        if not self.monitor_running:
            raise RuntimeError("请先开始实时监测，再进行 AI 分析。")
        problem = None
        restore_error = None
        initial_status = self.pipeline.adc_status()
        initial_stage = self.pipeline.pipeline_status()
        initial_rx_blocks = getattr(self, "rx_blocks", 0)
        capture_priority = getattr(self, "_ecg_capture_priority", None)
        if capture_priority is None:
            capture_priority = threading.Event()
            self._ecg_capture_priority = capture_priority
        priority_was_set = capture_priority.is_set()
        capture_priority.set()
        try:
            self.ai_progress = max(self.ai_progress, 1)
            self.ai_status = "正在切换到电极接触检查模式…"
            self.stop_monitor()
            self._lead_off_probe_active = True
            with self._data_lock:
                self._probe_frame_history.clear()
                self._probe_quality_history.clear()
            self.start_monitor(lead_off_test=True, update_page=False)
            if not self.pipeline.lead_off_ready():
                raise RuntimeError("电极接触检测配置未启用，已取消分析。")
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self._ai_cancel.is_set() and self._ai_cancel_reason != "电极接触异常，推理已取消。":
                    raise InterruptedError(self._ai_cancel_reason or "推理已取消。")
                if not self.monitor_running:
                    raise RuntimeError("电极检测期间采集意外停止。")
                with self._data_lock:
                    frames = tuple(self._probe_frame_history)
                    qualities = tuple(self._probe_quality_history)
                got = min(len(frames), len(qualities))
                self.ai_progress = max(1, min(5, 1 + int(4 * got / required_frames)))
                self.ai_status = "电极接触检查 {}/{} 帧".format(got, required_frames)
                probe_state = self._assess_probe_window(
                    frames, qualities, required_frames)
                if probe_state == "ready":
                    break
                if probe_state == "lead_off":
                    problem = "检测到电极脱落或接触不良，已取消 AI 分析。"
                    break
                if probe_state == "gap":
                    problem = "电极检测数据存在帧缺口，已取消 AI 分析。"
                    break
                self._ai_cancel.wait(0.05)
            else:
                status = self.pipeline.adc_status()
                stage = self.pipeline.pipeline_status()
                adc_delta = {key: ((int(status[key]) - int(initial_status[key])) & 0xFFFFFFFF)
                             for key in ("frame_seq", "frame_count", "drop_count")
                             if key in status and key in initial_status}
                stage_delta = {key: ((int(stage[key]) - int(initial_stage[key])) & 0xFFFFFFFF)
                               for key in ("lead12_frame_count", "packet_record_count",
                                           "packet_block_count")
                               if key in stage and key in initial_stage}
                rx_delta = getattr(self, "rx_blocks", 0) - initial_rx_blocks
                quality_counts = (np.count_nonzero(
                    np.asarray(qualities, dtype=np.uint32), axis=0).astype(int).tolist()
                    if qualities else [0, 0, 0, 0, 0])
                probe_diag = {"probe_frames": len(frames), "required_frames": required_frames,
                              "quality_flag_counts": quality_counts,
                              "adc_delta": adc_delta, "pipeline_delta": stage_delta,
                              "rx_blocks_delta": rx_delta,
                              "worker_error": repr(getattr(self, "_worker_error", None))}
                print("PulseAtlas electrode probe timeout: {}".format(probe_diag),
                      file=sys.stderr, flush=True)
                problem = "接触检测收帧超时：收到{} / {} 帧；采集计数 {}，DMA块增量 {}。".format(
                    len(frames), required_frames, adc_delta.get("frame_count", "?"), rx_delta)
        finally:
            try:
                if self.monitor_running:
                    self.stop_monitor()
                self._lead_off_probe_active = False
                if not self._closed:
                    # Probe frames must never enter the display/inference window.
                    self._ai_cancel.clear()
                    self._ai_cancel_reason = None
                    self.start_monitor(lead_off_test=False, update_page=False)
            except Exception as exc:
                restore_error = exc
            finally:
                if not priority_was_set:
                    capture_priority.clear()
        if restore_error is not None:
            raise RuntimeError("恢复正常 ECG 采集失败：{}".format(restore_error))
        if problem:
            raise RuntimeError(problem)

    @staticmethod
    def _assess_probe_window(frame_rows, quality_rows, required_frames=100):
        """Check contact via LOFF bits, not ECG waveform quality flags.

        The short probe uses a DC electrode-off test mode, so flatline and
        other signal-quality bits may be expected. The normal-mode inference
        window separately enforces every quality flag.
        """
        if min(len(frame_rows), len(quality_rows)) < required_frames:
            return "waiting"
        frames = np.asarray(frame_rows[-required_frames:], dtype=np.uint32)
        quality = np.asarray(quality_rows[-required_frames:], dtype=np.uint32)
        if frames.shape != (required_frames,) or quality.shape != (required_frames, 5):
            return "waiting"
        all_quality = np.asarray(quality_rows, dtype=np.uint32)
        if all_quality.ndim != 2 or all_quality.shape[1] != 5:
            return "waiting"
        if np.any(all_quality[:, 0]):
            return "lead_off"
        steps = ((frames[1:].astype(np.uint64) -
                  frames[:-1].astype(np.uint64)) & 0xFFFFFFFF)
        if not np.all(steps == 1):
            return "gap"
        return "ready"

    @staticmethod
    def _assess_ai_window(frame_rows, quality_rows):
        if len(frame_rows) < AI_WINDOW_FRAMES:
            return "仅{}帧".format(len(frame_rows)), {"frames": len(frame_rows)}
        if len(quality_rows) != AI_WINDOW_FRAMES:
            return "标记未同步", {"frames": len(frame_rows), "quality": len(quality_rows)}
        frames = np.asarray(frame_rows, dtype=np.uint32)
        quality = np.asarray(quality_rows, dtype=np.uint32)
        if quality.shape != (AI_WINDOW_FRAMES, 5):
            return "标记未同步", {"quality_shape": quality.shape}
        steps = ((frames[1:].astype(np.uint64) - frames[:-1].astype(np.uint64)) & 0xFFFFFFFF)
        gaps = int(np.count_nonzero(steps != 1))
        bad_by_type = np.count_nonzero(quality, axis=0).astype(int).tolist()
        bad_frames = int(np.count_nonzero(np.any(quality, axis=1)))
        advisory_frames = int(np.count_nonzero(
            np.any(quality[:, 1:] != 0, axis=1)))
        detail = {"gap_locations": gaps, "gap_steps": steps[steps != 1][:20].tolist(),
                  "bad_frames": bad_frames, "quality_columns": bad_by_type,
                  "quality_warning_frames": advisory_frames}
        if gaps:
            return "帧缺口{}处".format(gaps), detail
        if bad_by_type[0]:
            return "电极接触异常{}帧".format(bad_by_type[0]), detail
        return None, detail

    def _run_ai_analysis_worker(self):
        backend = getattr(self, "_ai_active_backend",
                          getattr(self, "ai_backend_mode", "ps"))
        backend_label = AI_BACKEND_LABELS[backend]
        try:
            golden_thread = getattr(self, "_golden_thread", None)
            while (not self.golden_passed and golden_thread is not None and
                   golden_thread.is_alive()):
                if self._ai_cancel.is_set():
                    raise InterruptedError(self._ai_cancel_reason or "推理已取消。")
                self.ai_status = "等待 NN 固定向量自检完成…"
                self._ai_cancel.wait(0.25)
            if not self.golden_passed:
                raise RuntimeError("NN 固定向量自检未通过：{}".format(
                    getattr(self, "golden_status", "未知错误")))
            codes = getattr(self, "_ai_cached_codes", None)
            if codes is None:
                self.ai_status = "锁定采集通道：准备连续 10 秒心电…"
                # Allow one UI frame to expose the state before freezing the
                # two expensive display paths.  Any frame lost during this
                # short handover is outside the subsequently selected window.
                self._ai_cancel.wait(0.20)
                capture_priority = getattr(self, "_ecg_capture_priority", None)
                if capture_priority is None:
                    capture_priority = threading.Event()
                    self._ecg_capture_priority = capture_priority
                capture_priority.set()
                try:
                    samples = self._wait_for_clean_ai_window()
                    self.ai_progress = 5
                    self.ai_status = "正在预处理稳定的 10 秒心电…"
                    codes = self.nn.encode_physical_leads(samples)
                    codes = np.asarray(codes, dtype=np.uint8).copy()
                    self.ai_progress = 8
                finally:
                    capture_priority.clear()
            else:
                codes = np.asarray(codes, dtype=np.uint8).copy()
                samples = getattr(self, "_ai_cached_samples", None)
                self.ai_progress = 8
                self.ai_status = "{}：正在推理…".format(backend_label)

            run_times, run_outputs = [], []
            for run in range(3):
                if self._ai_cancel.is_set():
                    raise InterruptedError(self._ai_cancel_reason or "推理已取消。")
                self.ai_status = "{} 正在推理…".format(backend_label)
                if backend == "ps":
                    output, seconds = self.nn.infer_ps(codes)
                else:
                    output, seconds = self.nn.infer_pl(codes)
                run_outputs.append(np.asarray(output).copy())
                run_times.append(float(seconds))
                self.ai_progress = 8 + int(74 * (run + 1) / 3)

            repeated_output_consistent = all(
                np.array_equal(run_outputs[0], output) for output in run_outputs[1:])
            probabilities = self.nn.logits_and_probabilities(run_outputs[-1])
            if self._ai_cancel.is_set():
                raise InterruptedError(self._ai_cancel_reason or "推理已取消。")
            # The normal ECG acquisition stays live while the stable 10-second
            # window is captured and inferred. Only after the model has used
            # that immutable snapshot do we briefly switch ADS1298 into LOFF
            # mode to validate electrode contact. The probe's restart can
            # therefore never contaminate the model input.
            self.ai_status = "复核电极接触…"
            try:
                self._probe_electrodes_for_ai()
            except Exception:
                self._clear_ai_comparison()
                self.ai_result = None
                raise

            if self._ai_cancel.is_set():
                raise InterruptedError(self._ai_cancel_reason or "推理已取消。")
            backend_results = getattr(self, "_ai_backend_results", {})
            backend_results[backend] = {
                "codes": run_outputs[-1].copy(),
                "runs_codes": [output.copy() for output in run_outputs],
                "seconds": float(np.mean(run_times)),
                "runs_seconds": list(run_times),
                "runs_consistent": bool(repeated_output_consistent),
                "probabilities": probabilities,
            }
            self._ai_backend_results = backend_results
            self._ai_cached_codes = codes.copy()
            if samples is not None:
                self._ai_cached_samples = np.asarray(samples, dtype=np.int32).copy()

            ps_result = backend_results.get("ps")
            pl_result = backend_results.get("pl")
            same_codes = (None if ps_result is None or pl_result is None else
                          bool(np.array_equal(ps_result["codes"], pl_result["codes"])))
            all_runs_consistent = all(
                item["runs_consistent"] for item in backend_results.values())
            all_outputs_consistent = all_runs_consistent and (
                same_codes is not False)
            selected_result = backend_results[backend]
            self.ai_ps_seconds = (ps_result["seconds"] if ps_result else None)
            self.ai_pl_seconds = (pl_result["seconds"] if pl_result else None)
            self.ai_result = {
                "backend": backend,
                "probabilities": selected_result["probabilities"],
                "ps_codes": (ps_result["codes"].astype(int).tolist()
                             if ps_result else None),
                "pl_codes": (pl_result["codes"].astype(int).tolist()
                             if pl_result else None),
                "same_codes": same_codes,
                "same_backend_runs": bool(all_runs_consistent),
                "ps_seconds": self.ai_ps_seconds,
                "pl_seconds": self.ai_pl_seconds,
                "ps_runs_seconds": (ps_result["runs_seconds"] if ps_result else None),
                "pl_runs_seconds": (pl_result["runs_seconds"] if pl_result else None),
                "speedup": (float(ps_result["seconds"] / max(pl_result["seconds"], 1e-9))
                            if ps_result and pl_result else None),
                "quality_warning_frames": int(getattr(
                    self, "_ai_window_quality_warning", 0)),
            }
            session_id = getattr(self, "_ai_session_id", None)
            if session_id is not None and self._history_store is not None:
                try:
                    best_label = max(probabilities, key=probabilities.get)
                    self._history_store.record_inference(
                        session_id, backend, best_label,
                        probabilities[best_label], probabilities,
                        elapsed_seconds=float(np.mean(run_times)))
                    self._refresh_history_cache(force=True)
                except Exception as exc:
                    self.ui_error = "推理结果未写入历史记录：{}".format(exc)
            if not all_outputs_consistent:
                self.ai_error = "重复推理或 PS / FPGA 输出存在不一致，请检查模型结果。"
        except InterruptedError as exc:
            self.ai_error = str(exc)
        except Exception as exc:
            self.ai_error = str(exc)
        finally:
            if self.ai_result is not None:
                self.ai_progress = 100
                if self.ai_error:
                    self.ai_status = self.ai_error
                else:
                    self.ai_status = "{} 推理完成".format(backend_label)
            elif self.ai_error:
                self.ai_status = self.ai_error
            else:
                self.ai_status = "推理未完成"
            self.ai_busy = False

    def _open_input_devices(self):
        event_size = struct.calcsize("@llHHi")
        for path in sorted(glob.glob("/dev/input/event*")):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            axes = {}
            for code in (0, 1, 53, 54):
                request = (2 << 30) | (24 << 16) | (ord("E") << 8) | (0x40 + code)
                try:
                    data = bytearray(24)
                    fcntl.ioctl(fd, request, data, True)
                    _, minimum, maximum, _, _, _ = struct.unpack("=iiiiii", data)
                    if maximum > minimum:
                        axes[code] = (minimum, maximum)
                except (OSError, AttributeError):
                    pass
            if 53 in axes and 54 in axes:
                xcode, ycode = 53, 54
                kind = "touch"
            elif 0 in axes and 1 in axes:
                xcode, ycode = 0, 1
                kind = "touch"
            else:
                xcode = ycode = None
                kind = "other"
            self._input_devices.append({"path": path, "fd": fd, "kind": kind,
                                        "axes": axes, "pending": bytearray(),
                                        "x": None, "y": None, "down": False,
                                        "relx": 0, "rely": 0})
            if kind == "touch" and self._touch["xcode"] == 0 and self._touch["ycode"] == 1:
                # Prefer the first absolute-capable device; raw range maps the
                # digitizer coordinate system to the current HDMI resolution.
                self._touch.update({"xcode": xcode, "ycode": ycode,
                                    "xmin": axes[xcode][0], "xmax": axes[xcode][1],
                                    "ymin": axes[ycode][0], "ymax": axes[ycode][1]})
                self._touch["device"] = path

    def _input_loop(self):
        self._open_input_devices()
        self._mouse_enabled = False
        size = struct.calcsize("@llHHi")
        while not self._input_stop.is_set():
            fds = [d["fd"] for d in self._input_devices]
            if not fds:
                self._keyboard_input()
                self._input_stop.wait(0.1)
                continue
            try:
                ready, _, _ = select.select(fds, [], [], 0.1)
            except (OSError, ValueError):
                self._input_stop.wait(0.1)
                continue
            for device in self._input_devices:
                if device["fd"] not in ready:
                    continue
                try:
                    device["pending"].extend(os.read(device["fd"], size*32))
                except BlockingIOError:
                    continue
                except OSError:
                    continue
                while len(device["pending"]) >= size:
                    chunk = bytes(device["pending"][:size])
                    del device["pending"][:size]
                    _, _, etype, code, value = struct.unpack("@llHHi", chunk)
                    self._handle_input_event(device, etype, code, value)
        for device in self._input_devices:
            try:
                os.close(device["fd"])
            except OSError:
                pass
        self._input_devices = []

    def _handle_input_event(self, device, etype, code, value):
        # Linux input_event constants: EV_SYN=0, EV_KEY=1, EV_REL=2, EV_ABS=3.
        if etype == 3:
            if code in (0, 53):
                device["x"] = value
            elif code in (1, 54):
                device["y"] = value
            elif code == 57:
                was = device["down"]
                device["down"] = value >= 0
                if device["down"] and not was:
                    self._touch_click(device)
        elif etype == 1:
            if code == 330:  # BTN_TOUCH
                was = device["down"]
                device["down"] = bool(value)
                if value and not was:
                    self._touch_click(device)
            elif code == 272 and value == 1:  # BTN_LEFT
                self._on_click(*self._pointer)
            elif value == 1:
                # USB keyboard keys for a board without a touch panel/mouse.
                keys = {35: "h", 2: "1", 3: "2", 4: "3", 5: "4",
                        30: "a", 16: "q", 57: " "}
                key = keys.get(code)
                if key == "h": self._set_page("home")
                elif key == "1": self._set_page("monitor")
                elif key == "2": self._set_page("leads")
                elif key == "3": self._set_page("ai")
                elif key == "4": self._set_page("spectrum")
                elif key == "v": self._set_page("vision")
                elif key == " " and self.page=="monitor":
                    self.stop_monitor() if self.monitor_running else self.start_monitor()
                elif key == "a" and self.page=="ai": self._run_ai_analysis()
                elif key == "a" and self.page=="vision": self._toggle_vision()
                elif key == "q": self.shutdown()
        elif etype == 2:
            if code == 0:
                self._pointer[0] = max(0, min(self.width-1, self._pointer[0]+value))
            elif code == 1:
                self._pointer[1] = max(0, min(self.height-1, self._pointer[1]+value))
            self._mouse_enabled = True

    def _touch_click(self, device):
        if device["x"] is None or device["y"] is None:
            return
        xcode, ycode = ((53,54) if 53 in device["axes"] else (0,1))
        xmin,xmax=device["axes"][xcode]
        ymin,ymax=device["axes"][ycode]
        x=int((device["x"]-xmin)*max(1,self.width-1)/(xmax-xmin))
        y=int((device["y"]-ymin)*max(1,self.height-1)/(ymax-ymin))
        self._on_click(max(0,min(self.width-1,x)),max(0,min(self.height-1,y)))

    def _keyboard_input(self):
        if not sys.stdin or not sys.stdin.isatty():
            return
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                key = sys.stdin.read(1).lower()
                if key == "h": self._set_page("home")
                elif key == "1": self._set_page("monitor")
                elif key == "2": self._set_page("leads")
                elif key == "3": self._set_page("ai")
                elif key == "4": self._set_page("spectrum")
                elif key == "v": self._set_page("vision")
                elif key == " " and self.page=="monitor":
                    self.stop_monitor() if self.monitor_running else self.start_monitor()
                elif key == "a" and self.page=="ai": self._run_ai_analysis()
                elif key == "a" and self.page=="vision": self._toggle_vision()
                elif key == "q": self.shutdown()
        except (OSError, ValueError):
            pass

    def _on_tft_click(self, x, y):
        """Dispatch a landscape TFT tap through this app's existing actions."""
        x, y = int(x), int(y)
        if y >= 212:
            index = min(len(MAIN_PAGES) - 1, max(0, (x - 5) // 51))
            self._set_page(MAIN_PAGES[index])
            return

        if self.page == "home":
            if 136 <= y < 208 and 9 <= x < 311:
                col = 0 if x < 160 else 1
                row = min(2, max(0, (y - 136) // 24))
                self._set_page(HOME_ACTIONS[row * 2 + col][0])
            return

        if y < 70 and x < 70:
            self._go_back()
            return

        if self.page == "lead_focus" and 74 <= y <= 98:
            if 239 <= x <= 266:
                self._navigate(-1)
            elif 276 <= x <= 304:
                self._navigate(1)
            return

        if self.page == "monitor" and 78 <= y <= 104:
            if 162 <= x < 224:
                self._set_page("trend")
            elif x >= 225:
                self._activate_selection()
            return

        if self.page == "monitor" and 176 <= y < 207:
            if 14 <= x < 100:
                self._set_filter_enabled(not self.filter_enabled)
            elif 104 <= x < 157:
                self._adjust_gain(-1)
            elif 214 <= x < 304:
                self._adjust_gain(1)
            return

        if self.page == "leads" and 181 <= y < 208:
            if 106 <= x < 166:
                self._set_filter_enabled(not self.filter_enabled)
            elif 170 <= x < 220:
                self._adjust_gain(-1)
            elif 224 <= x < 274:
                self._adjust_gain(1)
            return

        if self.page == "leads" and 73 <= y < 180 and 8 <= x < 312:
            col = min(2, max(0, (x - 8) // 102))
            row = min(3, max(0, (y - 73) // 27))
            self.focused_lead_index = int(row * 3 + col)
            self._set_focused_lead_name()
            self._set_page("lead_focus")
            return

        if self.page == "lead_focus" and 176 <= y < 207:
            if 14 <= x < 88:
                self._set_filter_enabled(not self.filter_enabled)
            elif 91 <= x < 151:
                self._adjust_gain(-1)
            elif 210 <= x < 304:
                self._adjust_gain(1)
            return

        if self.page == "ai" and 76 <= y <= 101:
            if 12 <= x < 79:
                self._set_ai_backend_mode("ps")
            elif 80 <= x < 158:
                self._set_ai_backend_mode("pl")
            elif 218 <= x <= 307:
                self._run_ai_analysis()
            return

        if self.page == "vision" and 78 <= y <= 106 and x >= 210:
            self._toggle_vision()
            return
        if self.page == "history" and 77 <= y < 194:
            self._select_history_record((y - 78) // 28)
            return

    def _on_click(self,x,y):
        if self.page=="home":
            top,bottom,margin,gap_x,gap_y=160,630,35,20,18
            card_w,card_h=390,226
            if y<top or y>=bottom:
                return
            col=min(2,max(0,(x-margin)//(card_w+gap_x)))
            row=min(1,max(0,(y-top)//(card_h+gap_y)))
            if x<margin or x>=self.width-margin:
                return
            page=("monitor","leads","ai","vision","history","settings")[row*3+col]
            self._set_page(page)
            return
        if 82<=y<=126 and 34<=x<=132:
            if self.page=="lead_focus": self._set_page("leads")
            elif self.page in ("spectrum","trend"): self._set_page("monitor")
            else: self._set_page("home")
            return
        if self.page=="monitor" and 580<=y<=642:
            if 35<=x<185:
                self._set_filter_enabled(not self.filter_enabled)
            elif 228<=x<358:
                if x < 292:
                    self._adjust_gain(-1)
                elif x >= 300:
                    self._adjust_gain(1)
            elif 400<=x<550:
                return
            elif 593<=x<723:
                self.page="leads"
            elif 765<=x<875:
                self.page="spectrum"
            elif 918<=x<1053:
                self.page="trend"
            elif 1095<=x<1245:
                self.stop_monitor() if self.monitor_running else self.start_monitor()
            return
        if self.page == "settings":
            if 470 <= y <= 520 and 190 <= x < 270:
                self._set_filter_enabled(not self.filter_enabled)
                return
            if 440 <= y <= 495 and 850 <= x < 895:
                self._adjust_gain(-1)
                return
            if 440 <= y <= 495 and 910 <= x < 960:
                self._adjust_gain(1)
                return
        if self.page == "history" and 270 <= y < 600 and 35 <= x < 820:
            self._select_history_record((y - 272) // 82)
            return
        if self.page=="leads":
            margin,top,gap_x,gap_y=35,174,20,12
            card_w,card_h=390,116
            if y>=top:
                col=min(2,max(0,(x-margin)//(card_w+gap_x)))
                row=min(3,max(0,(y-top)//(card_h+gap_y)))
                index=int(row*3+col)
                if margin<=x<self.width-margin and index<12:
                    self.focused_lead_index=index
                    self.focused_lead_name=("II" if index==1 else
                                            ("I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6".split(", ")[index]))+" 导联"
                    self._set_page("lead_focus")
            return
        if self.page=="ai":
            if 96<=y<=148 and 595<=x<780:
                self._set_ai_backend_mode("ps")
                return
            if 96<=y<=148 and 780<=x<1000:
                self._set_ai_backend_mode("pl")
                return
            if 96<=y<=148 and 1015<=x<=1255:
                self._run_ai_analysis()
                return
        if self.page=="vision" and 565<=y<=636 and x>=730:
            self._toggle_vision()
            return

    def shutdown(self):
        self._input_stop.set()
        self._ui_stop.set()
        self._tft_stop.set()
        self._ai_cancel.set()
        self._vision_runtime.stop(timeout=3.0)
        self.stop_monitor()
        for thread in (self._input_thread,self._button_thread,
                       self._golden_thread,self._ui_thread,self._tft_thread,
                       self._ai_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=22.0 if thread is self._ai_thread else 2.0)
        if self._tft_backend is not None:
            try:
                self._tft_backend.close()
            except Exception:
                pass
            self._tft_backend = None
        if self._history_store is not None:
            try:
                self._history_store.close()
            except Exception as exc:
                self.ui_error = "历史数据库关闭失败：{}".format(exc)
            self._history_store = None
        super().close()

    def close(self):
        self.shutdown()

