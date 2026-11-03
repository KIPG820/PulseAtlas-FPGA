"""ctypes bridge to the shared PulseAtlas LVGL 9.4 board renderer."""

from __future__ import annotations

import ctypes
from datetime import datetime
from pathlib import Path
import time

import numpy as np

from ecg_display_metrics import (
    SPECTROGRAM_COLUMNS,
    SPECTROGRAM_ROWS,
    TREND_POINTS,
    spectrogram_dbfs,
    trend_points,
)


PAGE_IDS = {
    "home": 0,
    "check": 1,
    "monitor": 2,
    "leads": 3,
    "lead_focus": 4,
    "spectrum": 5,
    "trend": 6,
    "ai": 8,
    "vision": 9,
    "history": 10,
    "settings": 11,
}
LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
I32_PTR = ctypes.POINTER(ctypes.c_int32)
I8_PTR = ctypes.POINTER(ctypes.c_int8)


class BoardHistoryRow(ctypes.Structure):
    _fields_ = [
        ("session_id", ctypes.c_int32),
        ("lead_count", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("avg_hr", ctypes.c_int32),
        ("confidence", ctypes.c_int32),
        ("sample_count", ctypes.c_int32),
        ("stored_sample_count", ctypes.c_int32),
        ("quality_bad_frames", ctypes.c_int32),
        ("dropped_chunks", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
        ("started_at", ctypes.c_char * 32),
        ("result", ctypes.c_char * 24),
        ("status", ctypes.c_char * 16),
    ]


class PulseAtlasLVGLBackend:
    """Run the same LVGL page builders as the Windows desktop preview."""

    def __init__(self, library_path=None, width=1280, height=720):
        self.width = int(width)
        self.height = int(height)
        self.path = Path(library_path or Path(__file__).with_name("libpulseatlas_lvgl.so"))
        if not self.path.is_file():
            raise FileNotFoundError("LVGL board library missing: {}".format(self.path))
        self.lib = ctypes.CDLL(str(self.path))
        self._bind()
        result = self.lib.pulseatlas_board_init()
        if result != 0:
            raise RuntimeError("LVGL board display initialization failed: {}".format(result))
        self._last_step = time.monotonic()
        self._bgr = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._last_waveforms = None
        self._last_trend = None
        self._last_ai_input = None
        self._last_spectrogram = None
        self._last_spectrogram_at = 0.0
        self._last_ai_backend_mode = None
        self.selected_ai_backend = "ps"

    def _bind(self):
        self.lib.pulseatlas_board_init.argtypes = []
        self.lib.pulseatlas_board_init.restype = ctypes.c_int
        self.lib.pulseatlas_board_show_page.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.pulseatlas_board_show_page.restype = ctypes.c_int
        self.lib.pulseatlas_board_set_focus.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.pulseatlas_board_set_focus.restype = None
        self.lib.pulseatlas_board_set_status.argtypes = [ctypes.c_int] * 5
        self.lib.pulseatlas_board_set_status.restype = None
        self.lib.pulseatlas_board_set_display_controls.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.pulseatlas_board_set_display_controls.restype = None
        self.lib.pulseatlas_board_set_history.argtypes = [
            ctypes.POINTER(BoardHistoryRow), ctypes.c_int, ctypes.c_int,
            I32_PTR, ctypes.c_int,
        ]
        self.lib.pulseatlas_board_set_history.restype = None
        self.lib.pulseatlas_board_set_waveforms.argtypes = [I32_PTR, ctypes.c_int, ctypes.c_int]
        self.lib.pulseatlas_board_set_waveforms.restype = None
        self.lib.pulseatlas_board_set_trend.argtypes = [I32_PTR, ctypes.c_int]
        self.lib.pulseatlas_board_set_trend.restype = None
        self.lib.pulseatlas_board_set_spectrogram.argtypes = [I8_PTR]
        self.lib.pulseatlas_board_set_spectrogram.restype = None
        self.lib.pulseatlas_board_set_ai_input.argtypes = [I32_PTR, ctypes.c_int]
        self.lib.pulseatlas_board_set_ai_input.restype = None
        self.lib.pulseatlas_board_set_ai_result.argtypes = [I32_PTR, ctypes.c_char_p, ctypes.c_int]
        self.lib.pulseatlas_board_set_ai_result.restype = None
        self.lib.pulseatlas_board_set_ai_progress.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_double, ctypes.c_double,
        ]
        self.lib.pulseatlas_board_set_ai_progress.restype = None
        self.lib.pulseatlas_board_set_ai_backend_mode.argtypes = [ctypes.c_int]
        self.lib.pulseatlas_board_set_ai_backend_mode.restype = None
        self.lib.pulseatlas_board_get_ai_backend_mode.argtypes = []
        self.lib.pulseatlas_board_get_ai_backend_mode.restype = ctypes.c_int
        self.lib.pulseatlas_board_step.argtypes = [ctypes.c_uint32]
        self.lib.pulseatlas_board_step.restype = None
        self.lib.pulseatlas_board_take_frame_dirty.argtypes = []
        self.lib.pulseatlas_board_take_frame_dirty.restype = ctypes.c_int
        self.lib.pulseatlas_board_framebuffer.argtypes = []
        self.lib.pulseatlas_board_framebuffer.restype = ctypes.POINTER(ctypes.c_uint16)

    @staticmethod
    def _int32_ptr(array):
        return array.ctypes.data_as(I32_PTR)

    @staticmethod
    def _int8_ptr(array):
        return array.ctypes.data_as(I8_PTR)

    @staticmethod
    def _waveforms(state, point_count=600):
        data = np.asarray(state.get("leads", ()), dtype=np.int32)
        point_count = int(point_count)
        output = np.zeros((12, point_count), dtype=np.int32)
        if data.ndim != 2 or data.shape[1] != 12 or not data.shape[0]:
            return output
        source_count = min(data.shape[0], point_count * 10)
        data = data[-source_count:]
        data = data - np.median(data, axis=0, keepdims=True).astype(np.int32)
        n = data.shape[0]
        if n >= point_count:
            usable = (n // point_count) * point_count
            bins = data[-usable:].reshape(point_count, usable // point_count, 12)
            peak_index = np.argmax(np.abs(bins.astype(np.int64)), axis=1)
            lead_index = np.arange(12, dtype=np.int32)[None, :]
            row_index = np.arange(point_count, dtype=np.int32)[:, None]
            sampled = bins[row_index, peak_index, lead_index]
        else:
            sampled = data
        output[:, -sampled.shape[0]:] = sampled.T
        limit = max(1, int(round(float(state.get("display_half_range", 100000)))))
        np.clip(output, -limit, limit, out=output)
        return np.ascontiguousarray(output)

    @staticmethod
    def _trend(state):
        return np.ascontiguousarray(trend_points(state), dtype=np.int32)

    @staticmethod
    def _spectrum(state):
        return np.ascontiguousarray(spectrogram_dbfs(state), dtype=np.int8)

    @staticmethod
    def _ai_input(state):
        # Keep the AI-page preview identical to the monitor's 6 s / 600-point
        # display path. Compressing 10 s to 64 points undersamples QRS peaks.
        return np.ascontiguousarray(
            PulseAtlasLVGLBackend._waveforms(state).T, dtype=np.int32
        )

    @staticmethod
    def _ai_scores(state):
        result = state.get("ai_result")
        if not isinstance(result, dict):
            return np.zeros(5, dtype=np.int32), b"", False
        probabilities = result.get("probabilities") or {}
        values = []
        for label in LABELS:
            value = float(probabilities.get(label, 0.0))
            values.append(int(np.clip(round(value * 100.0), 0, 100)))
        best = int(np.argmax(values))
        return np.asarray(values, dtype=np.int32), LABELS[best].encode("ascii"), True

    def _copy_rgb565_to_bgr(self):
        source = np.ctypeslib.as_array(
            self.lib.pulseatlas_board_framebuffer(),
            shape=(self.width * self.height,),
        ).reshape(self.height, self.width)
        blue5 = source & 0x1F
        green6 = (source >> 5) & 0x3F
        red5 = (source >> 11) & 0x1F
        self._bgr[:, :, 0] = ((blue5 << 3) | (blue5 >> 2)).astype(np.uint8)
        self._bgr[:, :, 1] = ((green6 << 2) | (green6 >> 4)).astype(np.uint8)
        self._bgr[:, :, 2] = ((red5 << 3) | (red5 >> 2)).astype(np.uint8)
        return self._bgr

    def render(self, page, state):
        page_id = PAGE_IDS.get(page, PAGE_IDS["home"])
        status = state.get("status") or {}
        connected = bool(status.get("id_ok", False) and status.get("enabled", False))
        heart_rate = state.get("heart_rate")
        bpm = int(round(float(heart_rate))) if heart_rate is not None and np.isfinite(heart_rate) else -1
        self.lib.pulseatlas_board_set_status(
            connected, bool(state.get("camera_connected")), bool(state.get("wifi_connected")),
            bool(state.get("monitor_running")), bpm,
        )
        self.lib.pulseatlas_board_set_focus(
            int(state.get("home_focus", -1)) if page == "home" else -1,
            int(state.get("focused_lead_index", -1)) if page in ("leads", "lead_focus") else -1,
        )
        gain_percent = int(np.clip(round(float(state.get("gain", 1.0)) * 100), 50, 300))
        self.lib.pulseatlas_board_set_display_controls(
            int(bool(state.get("filter_enabled", True))), gain_percent)
        history_rows = list(state.get("history_records") or ())[:4]
        history_buffer = (BoardHistoryRow * 4)()
        for index, item in enumerate(history_rows):
            row = history_buffer[index]
            row.session_id = int(item.get("session_id", 0))
            row.lead_count = int(item.get("lead_count", 12))
            row.sample_rate = int(round(float(item.get("sample_rate", 1000))))
            avg_hr = item.get("avg_heart_rate")
            row.avg_hr = 0 if avg_hr is None else int(round(float(avg_hr)))
            confidence = item.get("result_confidence")
            row.confidence = 0 if confidence is None else int(round(float(confidence) * 100))
            row.sample_count = int(item.get("sample_count", 0))
            row.stored_sample_count = int(item.get("stored_sample_count", 0))
            row.quality_bad_frames = int(item.get("quality_bad_frames", 0))
            row.dropped_chunks = int(item.get("dropped_chunks", 0))
            status = str(item.get("status") or "")
            row.status_code = 0 if status == "complete" else 1
            started = str(item.get("started_at") or "")
            try:
                started = datetime.fromisoformat(started).strftime("%m-%d %H:%M")
            except ValueError:
                started = started[:20]
            row.started_at = started.encode("utf-8")[:31]
            row.result = str(item.get("result_label") or "").encode("utf-8")[:23]
            status_text = {"recording": "采集中", "complete": "已完成",
                           "incomplete": "不完整"}.get(status, "已停止")
            row.status = status_text.encode("utf-8")[:15]
        detail = state.get("history_detail") or {}
        preview = np.zeros(80, dtype=np.int32)
        preview_values = np.asarray(detail.get("preview", ()), dtype=np.int32).reshape(-1)
        if preview_values.size:
            count = min(80, preview_values.size)
            preview[:count] = preview_values[-count:]
        self.lib.pulseatlas_board_set_history(
            history_buffer, len(history_rows),
            int(state.get("history_selected_id") or -1),
            self._int32_ptr(preview), int(min(80, preview_values.size)))

        if page in ("monitor", "leads", "lead_focus", "spectrum"):
            point_count = 1000 if page == "leads" else 600
            waveforms = self._waveforms(state, point_count=point_count)
            if self._last_waveforms is None or not np.array_equal(waveforms, self._last_waveforms):
                self.lib.pulseatlas_board_set_waveforms(self._int32_ptr(waveforms), point_count,
                                                        int(round(self._display_range(state))))
                self._last_waveforms = waveforms

        if page == "trend":
            trend = self._trend(state)
            if self._last_trend is None or not np.array_equal(trend, self._last_trend):
                self.lib.pulseatlas_board_set_trend(self._int32_ptr(trend), TREND_POINTS)
                self._last_trend = trend

        now = time.monotonic()
        if page == "spectrum" and (self._last_spectrogram is None or now - self._last_spectrogram_at >= 0.8):
            spectrogram = self._spectrum(state)
            if self._last_spectrogram is None or not np.array_equal(spectrogram, self._last_spectrogram):
                self.lib.pulseatlas_board_set_spectrogram(self._int8_ptr(spectrogram))
                self._last_spectrogram = spectrogram
            self._last_spectrogram_at = now

        if page == "ai":
            mode = 1 if state.get("ai_backend_mode") == "pl" else 0
            if mode != self._last_ai_backend_mode:
                self.lib.pulseatlas_board_set_ai_backend_mode(mode)
                self._last_ai_backend_mode = mode
            ai_input = self._ai_input(state)
            if ai_input.size and (self._last_ai_input is None or not np.array_equal(ai_input, self._last_ai_input)):
                self.lib.pulseatlas_board_set_ai_input(self._int32_ptr(ai_input), ai_input.shape[0])
                self._last_ai_input = ai_input
        scores, label, valid = self._ai_scores(state)
        self.lib.pulseatlas_board_set_ai_result(self._int32_ptr(scores), label, int(valid))
        self.lib.pulseatlas_board_set_ai_progress(
            int(state.get("ai_progress", 0)), int(bool(state.get("ai_busy", False))),
            str(state.get("ai_status") or "等待开始").encode("utf-8"),
            float(state.get("ai_ps_seconds") or -1.0),
            float(state.get("ai_pl_seconds") or -1.0),
        )

        self.lib.pulseatlas_board_show_page(page_id, int(state.get("focused_lead_index", 1)))
        # Page construction creates the labels that display these values;
        # update them once more after the page switch to avoid stale widgets.
        self.lib.pulseatlas_board_set_display_controls(
            int(bool(state.get("filter_enabled", True))), gain_percent)
        elapsed_ms = max(0, min(100, int((now - self._last_step) * 1000)))
        self._last_step = now
        self.lib.pulseatlas_board_step(elapsed_ms)
        if page == "ai":
            selected = int(self.lib.pulseatlas_board_get_ai_backend_mode())
            self.selected_ai_backend = "pl" if selected else "ps"
            self._last_ai_backend_mode = selected
        if self.lib.pulseatlas_board_take_frame_dirty():
            return self._copy_rgb565_to_bgr()
        return None

    @staticmethod
    def _display_range(state):
        return state.get("display_half_range", 100000)


__all__ = ["PulseAtlasLVGLBackend"]
