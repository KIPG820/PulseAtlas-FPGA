"""Landscape, compact PulseAtlas renderer for the 320x240 ST7789 panel.

This renderer is a small-screen layout for the same PulseAtlas page/state model
used by the HDMI application. It consumes live snapshot values and never draws
sample ECG traces when the acquisition buffer is empty.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pulseatlas_board_render import (
    BG, BLUE, BLUE_DARK, BLUE_LIGHT, GREEN, LINE, MUTED, NAVY, RED, WHITE,
    FONTS, LEAD_NAMES, _text,
)
from ecg_display_metrics import (
    SPECTROGRAM_DB_MAX, SPECTROGRAM_DB_MIN, SPECTROGRAM_INVALID,
    TREND_INVALID, trend_points, spectrogram_dbfs,
)


WIDTH = 320
HEIGHT = 240
NAV_Y = 212
NAV_H = 23
MAIN_PAGES = ("home", "monitor", "leads", "spectrum", "ai", "vision")
PAGE_TITLES = {
    "home": "心电监测", "check": "设备自检", "monitor": "实时监测",
    "leads": "12 导联", "lead_focus": "导联详情", "spectrum": "时频图",
    "trend": "心率趋势", "ai": "AI 分析", "vision": "视觉分析",
    "history": "历史记录", "settings": "设置",
}
HOME_ACTIONS = (
    ("monitor", "实时监测"), ("leads", "12 导联"),
    ("ai", "AI 分析"), ("vision", "视觉分析"),
    ("history", "历史记录"), ("settings", "设置"),
)


def _signal(state):
    data = np.asarray(state.get("leads", ()), dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 12 or data.shape[0] < 2:
        return None
    return data


def _has_signal(state):
    return _signal(state) is not None


def _card(draw, box, fill=WHITE, outline=LINE, radius=8):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _status(draw, label, x, y, connected):
    color = GREEN if connected else MUTED
    draw.ellipse((x, y + 4, x + 6, y + 10), fill=color)
    _text(draw, (x + 8, y), label, 8, color, True)


def _tft_logo(draw, x=10, y=14, size=20):
    """Scale the original HDMI mark uniformly for the small TFT header."""
    scale = size / 54.0
    sx = lambda value: int(round(x + value * scale))
    sy = lambda value: int(round(y + value * scale))
    cx = sx(27)
    cy = sy(27)
    radius = max(1, int(round(27 * scale)))
    stroke = max(1, int(round(2 * scale)))
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                 outline=BLUE, width=stroke)
    draw.arc((sx(12), sy(2), sx(42), sy(52)), 180, 360,
             fill=(193, 214, 235), width=max(1, int(round(scale))))
    draw.line((sx(4), cy, sx(50), cy), fill=(206, 221, 238),
              width=max(1, int(round(scale))))
    points = [(sx(5), sy(31)), (sx(19), sy(31)), (sx(24), sy(19)),
              (sx(29), sy(40)), (sx(35), sy(11)), (sx(40), sy(31)),
              (sx(47), sy(31)), (sx(49), sy(31))]
    draw.line(points, fill=RED, width=max(1, int(round(3 * scale))),
              joint="curve")
    draw.ellipse((sx(50), sy(2), sx(58), sy(10)), fill=BLUE)


def _header(draw, state, page, title=None):
    draw.rounded_rectangle((6, 5, 314, 43), 8, fill=WHITE, outline=LINE, width=1)
    _tft_logo(draw, 10, 14, 20)
    _text(draw, (36, 16), "PulseAtlas", 14, NAVY, True)
    status = state.get("status") or {}
    _status(draw, "ECG", 163, 13,
            bool(status.get("id_ok", False) and status.get("enabled", False)))
    _status(draw, "USB", 211, 13, bool(state.get("camera_connected", False)))
    _status(draw, "Wi-Fi", 260, 13, bool(state.get("wifi_connected", False)))
    if page != "home":
        _text(draw, (9, 49), "←", 15, BLUE, True)
        _text(draw, (29, 50), title or PAGE_TITLES.get(page, "PulseAtlas"),
              13, NAVY, True)
    else:
        _text(draw, (11, 50), title or PAGE_TITLES[page], 14, NAVY, True)


def _nav(draw, page):
    labels = (("home", "首页"), ("monitor", "实时监测"), ("leads", "12导联"),
              ("spectrum", "时频图"), ("ai", "AI分析"), ("vision", "视觉"))
    x0, gap, width = 5, 2, 49
    for index, (key, label) in enumerate(labels):
        x = x0 + index * (width + gap)
        selected = page == key or (key == "leads" and page == "lead_focus")
        fill, fg = (BLUE, WHITE) if selected else (WHITE, BLUE)
        draw.rounded_rectangle((x, NAV_Y, x + width, NAV_Y + NAV_H), 6,
                               fill=fill, outline=BLUE, width=1)
        _text(draw, (x + width // 2, NAV_Y + NAV_H // 2), label, 8, fg, True,
              anchor="mm")


def _grid(draw, box, minor=8, major=40, minor_x=None, major_x=None):
    x0, y0, x1, y1 = box
    minor_x = max(1, int(round(minor if minor_x is None else minor_x)))
    major_x = max(minor_x, int(round(major if major_x is None else major_x)))
    draw.rectangle(box, fill=WHITE, outline=LINE, width=1)
    for x in range(x0 + minor_x, x1, minor_x):
        color = (239, 180, 192) if (x - x0) % major_x < minor_x else (250, 226, 231)
        draw.line((x, y0 + 1, x, y1 - 1), fill=color, width=1)
    for y in range(y0 + minor, y1, minor):
        color = (239, 180, 192) if (y - y0) % major == 0 else (250, 226, 231)
        draw.line((x0 + 1, y, x1 - 1, y), fill=color, width=1)


def _trace_points(values, box, scale=1.0):
    x0, y0, x1, y1 = box
    values = np.asarray(values, dtype=np.float32)
    if values.size < 2:
        return []
    bins = max(2, x1 - x0 + 1)
    values = values - float(np.median(values))
    if values.size > bins:
        usable = (values.size // bins) * bins
        grouped = values[-usable:].reshape(bins, usable // bins)
        values = grouped[np.arange(bins), np.argmax(np.abs(grouped), axis=1)]
    clip = max(float(np.percentile(np.abs(values), 95)), 1.0)
    clip /= max(float(scale), 0.1)
    half_h = max(1.0, (y1 - y0) * 0.43)
    center = (y0 + y1) / 2.0
    points = []
    for i, value in enumerate(values):
        y = center - float(np.clip(value / clip, -1.0, 1.0)) * half_h
        points.append((x0 + i, int(round(y))))
    return points


def _draw_trace(draw, state, lead_index, box, grid=True, line_width=1, seconds=4.0):
    if grid:
        width = max(1, box[2] - box[0])
        minor_x = width * 0.04 / max(float(seconds), 0.04)
        _grid(draw, box, minor=5 if width < 100 else 8,
              major=25 if width < 100 else 40,
              minor_x=minor_x, major_x=minor_x * 5)
    data = _signal(state)
    if data is None:
        return False
    sample_rate = float(state.get("sample_rate", 1000.0))
    sample_count = max(2, int(round(float(seconds) * sample_rate)))
    points = _trace_points(data[-sample_count:, lead_index], box,
                           state.get("gain", 1.0))
    if len(points) > 1:
        draw.line(points, fill=RED, width=line_width, joint="curve")
    return True


def _button(draw, box, label, primary=False, size=9):
    fill, outline, fg = (BLUE, BLUE, WHITE) if primary else (WHITE, BLUE, BLUE)
    draw.rounded_rectangle(box, 5, fill=fill, outline=outline, width=1)
    _text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), label,
          size, fg, True, anchor="mm")


def _home(draw, state):
    _card(draw, (8, 72, 312, 130), radius=8)
    _text(draw, (17, 79), "实时心率", 9, MUTED, True)
    heart_rate = state.get("heart_rate")
    hr = "--" if heart_rate is None else str(int(round(float(heart_rate))))
    _text(draw, (16, 91), hr, 29, RED, True)
    _text(draw, (70, 108), "bpm", 10, RED, True)
    connected = bool((state.get("status") or {}).get("id_ok", False))
    _text(draw, (167, 86), "采集运行中" if state.get("monitor_running") else
          ("ECG 已连接" if connected else "等待 ECG 连接"),
          9, GREEN if connected else MUTED, True)
    _draw_trace(draw, state, 1, (164, 101, 299, 123), grid=False, line_width=1)

    for index, (page, label) in enumerate(HOME_ACTIONS):
        row, col = divmod(index, 2)
        x, y = 9 + col * 153, 136 + row * 24
        _card(draw, (x, y, x + 148, y + 21), radius=5)
        draw.ellipse((x + 7, y + 7, x + 13, y + 13),
                     fill=RED if page == "leads" else BLUE)
        _text(draw, (x + 19, y + 4), label, 10, NAVY, True)
        _text(draw, (x + 136, y + 3), "›", 13, BLUE, True, anchor="mm")


def _filter_switch(draw, state, box=(14, 177, 100, 201)):
    """Compact, explicit on/off control for classic ECG display filtering."""
    enabled = bool(state.get("filter_enabled", True))
    x0, y0, x1, y1 = box
    cy = (y0 + y1) // 2
    draw.rounded_rectangle(
        box, radius=5, fill=BLUE_LIGHT if enabled else WHITE,
        outline=BLUE if enabled else LINE, width=1)
    _text(draw, (x0 + 7, cy), "滤波", 9, NAVY, True, anchor="lm")
    _text(draw, (x0 + 32, cy), "开" if enabled else "关", 9,
          BLUE if enabled else MUTED, True, anchor="lm")
    track = (x1 - 31, cy - 6, x1 - 5, cy + 6)
    draw.rounded_rectangle(track, radius=6,
                           fill=BLUE if enabled else (177, 187, 199))
    knob_x = track[2] - 6 if enabled else track[0] + 6
    draw.ellipse((knob_x - 4, cy - 4, knob_x + 4, cy + 4), fill=WHITE)


def _monitor(draw, state):
    _card(draw, (8, 72, 312, 204))
    hr = state.get("heart_rate")
    value = "--" if hr is None else str(int(round(float(hr))))
    _text(draw, (17, 80), value, 26, RED, True)
    _text(draw, (65, 96), "bpm", 9, RED, True)
    _button(draw, (162, 80, 223, 101), "心率趋势", size=7)
    _button(draw, (234, 80, 303, 101),
            "停止采集" if state.get("monitor_running") else "开始采集",
            primary=bool(state.get("monitor_running")), size=8)
    _text(draw, (17, 107), "II 导联 · 25 mm/s · ADC 码", 8, MUTED, True)
    _draw_trace(draw, state, 1, (17, 118, 303, 169), grid=True, line_width=2,
                seconds=4.0)
    if not _has_signal(state):
        _text(draw, (160, 145), "等待 ECG 信号", 10, MUTED, True, anchor="mm")
    _filter_switch(draw, state)
    _button(draw, (104, 177, 157, 201), "比例 -", size=8)
    _button(draw, (160, 177, 210, 201), "×{:.2g}".format(
        float(state.get("gain", 1.0))), size=8)
    _button(draw, (214, 177, 304, 201), "比例 ＋", size=8)

def _leads(draw, state):
    data = _signal(state)
    for index, name in enumerate(LEAD_NAMES):
        row, col = divmod(index, 3)
        x, y = 8 + col * 102, 73 + row * 27
        box = (x, y, x + 97, y + 24)
        focused = index == int(state.get("focused_lead_index", -1))
        _card(draw, box, fill=WHITE, outline=BLUE if focused else LINE, radius=5)
        _text(draw, (x + 4, y + 2), name, 8, BLUE, True)
        _grid(draw, (x + 25, y + 4, x + 93, y + 21), minor=6, major=30)
        if data is not None:
            points = _trace_points(data[-2500:, index], (x + 26, y + 5, x + 92, y + 20),
                                   state.get("gain", 1.0))
            if len(points) > 1:
                draw.line(points, fill=RED, width=1)
    if data is None:
        _text(draw, (160, 177), "等待 ECG 信号", 8, MUTED, anchor="mm")
    _text(draw, (11, 181), "25 mm/s · ADC 码", 7, MUTED)
    _button(draw, (106, 181, 166, 205),
            "滤波开" if state.get("filter_enabled", True) else "滤波关",
            primary=bool(state.get("filter_enabled", True)), size=8)
    _button(draw, (170, 181, 220, 205), "−", size=9)
    _button(draw, (224, 181, 274, 205), "＋", size=9)
    _text(draw, (278, 185), "×{:.1f}".format(float(state.get("gain", 1.0))),
          7, BLUE, True)


def _lead_focus(draw, state):
    index = max(0, min(11, int(state.get("focused_lead_index", 1))))
    _card(draw, (8, 72, 312, 204))
    _text(draw, (17, 78), LEAD_NAMES[index] + " 导联", 11, BLUE, True)
    _button(draw, (241, 76, 263, 95), "<", size=11)
    _button(draw, (279, 76, 301, 95), ">", size=11)
    _draw_trace(draw, state, index, (17, 98, 303, 169), grid=True, line_width=2,
                seconds=6.0)
    if not _has_signal(state):
        _text(draw, (160, 138), "等待 ECG 信号", 10, MUTED, True, anchor="mm")
    _button(draw, (14, 177, 88, 201),
            "滤波开" if state.get("filter_enabled", True) else "滤波关",
            primary=bool(state.get("filter_enabled", True)), size=8)
    _button(draw, (92, 177, 150, 201), "比例 −", size=8)
    _button(draw, (154, 177, 210, 201), "×{:.2g}".format(
        float(state.get("gain", 1.0))), size=8)
    _button(draw, (214, 177, 304, 201), "比例 ＋", size=8)


def _ai(draw, state):
    _card(draw, (8, 72, 312, 204))
    mode = state.get("ai_backend_mode") or "ps"
    _button(draw, (14, 77, 77, 98), "PS NumPy", primary=mode == "ps", size=8)
    _button(draw, (82, 77, 155, 98), "FPGA", primary=mode == "pl", size=8)
    _button(draw, (224, 77, 304, 98),
            "推理中" if state.get("ai_busy") else "开始推理",
            primary=True, size=8)
    progress = int(np.clip(state.get("ai_progress", 0), 0, 100))
    draw.rounded_rectangle((14, 103, 304, 111), 4, fill=BLUE_LIGHT)
    if progress:
        draw.rounded_rectangle((14, 103, 14 + int(290 * progress / 100), 111),
                               4, fill=BLUE)
    _text(draw, (15, 114), str(state.get("ai_status") or "等待开始")[:38],
          7, MUTED)

    result = state.get("ai_result")
    scores = result.get("probabilities", {}) if isinstance(result, dict) else {}
    labels = (("NORM", "正常"), ("MI", "心肌梗死"), ("STTC", "ST-T"),
              ("CD", "传导异常"), ("HYP", "心室肥厚"))
    best_label = "待分析"
    best_score = 0.0
    if scores:
        best_label, best_score = max(
            ((key, float(scores.get(key, 0.0))) for key, _ in labels),
            key=lambda item: item[1])
    _text(draw, (15, 126), "模型建议", 9, BLUE, True)
    display_name = dict(labels).get(best_label, best_label)
    _text(draw, (15, 138), display_name, 13, NAVY, True)
    _text(draw, (94, 141), "{:.0f}%".format(best_score * 100), 11, BLUE, True)
    y0 = 155
    for index, (key, label) in enumerate(labels):
        y = y0 + index * 9
        score = float(scores.get(key, 0.0)) if scores else 0.0
        _text(draw, (15, y), label, 7, MUTED)
        draw.rounded_rectangle((76, y + 1, 264, y + 7), 3, fill=BLUE_LIGHT)
        if score > 0:
            draw.rounded_rectangle((76, y + 1, 76 + int(188 * np.clip(score, 0, 1)), y + 7),
                                   3, fill=BLUE if key == best_label else MUTED)
        _text(draw, (270, y - 1), "{:d}%".format(int(round(score * 100))), 7, NAVY)
    ps_time = state.get("ai_ps_seconds")
    pl_time = state.get("ai_pl_seconds")
    if ps_time is not None or pl_time is not None:
        ps_label = "--" if ps_time is None else "{:.2f}s".format(float(ps_time))
        pl_label = "--" if pl_time is None else "{:.2f}s".format(float(pl_time))
        speedup = result.get("speedup") if isinstance(result, dict) else None
        speed_label = "" if speedup is None else "  {:.2f}×".format(float(speedup))
        _text(draw, (15, 199), "PS {}  FPGA {}{}".format(ps_label, pl_label, speed_label),
              7, MUTED)


def _spectrogram(state):
    matrix = spectrogram_dbfs(state)
    return None if np.all(matrix == SPECTROGRAM_INVALID) else matrix


def _heat_color(value):
    if int(value) == SPECTROGRAM_INVALID:
        return (220, 227, 234)
    t = float(np.clip((float(value) - SPECTROGRAM_DB_MIN) /
                      (SPECTROGRAM_DB_MAX - SPECTROGRAM_DB_MIN), 0.0, 1.0))
    if t < 0.5:
        u = t * 2.0
        return (int(26 + 20 * u), int(54 + 135 * u), int(99 + 135 * u))
    u = (t - 0.5) * 2.0
    return (int(46 + 190 * u), int(189 - 135 * u), int(234 - 151 * u))


def _spectrum(draw, image, state):
    box = (40, 105, 285, 190)
    _card(draw, (8, 72, 312, 204))
    _text(draw, (14, 77), "II 导联 · 10 s 时频图", 8, NAVY, True)
    _text(draw, (230, 77), "PSD dBFS/Hz", 7, MUTED, True, anchor="ra")
    _draw_trace(draw, state, 1, (40, 86, 285, 102), grid=False, line_width=1,
                seconds=6.0)
    draw.rectangle(box, fill=WHITE, outline=LINE, width=1)
    _text(draw, (12, 103), "100", 7, MUTED)
    _text(draw, (18, 181), "0 Hz", 7, MUTED)
    _text(draw, (244, 192), "0 s", 7, MUTED)
    _text(draw, (38, 192), "-10", 7, MUTED)
    for i in range(1, 5):
        y = box[1] + i * (box[3] - box[1]) // 5
        draw.line((box[0] + 1, y, box[2] - 1, y), fill=LINE, width=1)
    matrix = _spectrogram(state)
    if matrix is None:
        _text(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
              "等待 ECG 信号", 9, MUTED, True, anchor="mm")
        return
    spec = Image.new("RGB", (matrix.shape[1], matrix.shape[0]))
    spec.putdata([_heat_color(int(v)) for v in matrix.ravel()])
    resampling = getattr(Image, "Resampling", Image)
    spec = spec.resize((box[2] - box[0] - 2, box[3] - box[1] - 2), resampling.BILINEAR)
    image.paste(spec, (box[0] + 1, box[1] + 1))
    draw.rectangle(box, outline=LINE, width=1)
    # Fixed color scale; values remain comparable between refreshes.
    for y in range(box[1], box[3] + 1):
        db = SPECTROGRAM_DB_MAX - (y - box[1]) * (
            SPECTROGRAM_DB_MAX - SPECTROGRAM_DB_MIN) / max(1, box[3] - box[1])
        draw.line((291, y, 298, y), fill=_heat_color(db))
    _text(draw, (300, 104), "-20", 5, MUTED)
    _text(draw, (299, 180), "-120", 5, MUTED)


def _trend(draw, state):
    _card(draw, (8, 72, 312, 204))
    box = (20, 89, 301, 189)
    draw.rectangle(box, fill=WHITE, outline=LINE, width=1)
    for val in (40, 60, 80, 100, 120):
        y = box[3] - int((val - 40) / 80 * (box[3] - box[1]))
        draw.line((box[0], y, box[2], y), fill=LINE, width=1)
        _text(draw, (box[0] + 2, y - 8), str(val), 6, MUTED)
    rates = trend_points(state)
    valid = rates != TREND_INVALID
    if np.count_nonzero(valid) >= 2:
        segment = []
        for i, val in enumerate(rates):
            if val == TREND_INVALID:
                if len(segment) > 1:
                    draw.line(segment, fill=BLUE, width=2)
                segment = []
                continue
            x = box[0] + int(i * (box[2] - box[0]) / (rates.size - 1))
            y = box[3] - int(np.clip((val - 40) / 80, 0, 1) * (box[3] - box[1]))
            segment.append((x, y))
        if len(segment) > 1:
            draw.line(segment, fill=BLUE, width=2)
        _text(draw, (208, 76), "逐搏 · 120 s", 7, MUTED)
    else:
        current = state.get("heart_rate")
        msg = "等待心率数据" if current is None else "当前心率 {} bpm".format(int(round(current)))
        _text(draw, (160, 139), msg, 9, MUTED, True, anchor="mm")


def _vision(draw, image, state):
    _card(draw, (8, 72, 312, 204))
    frame = state.get("vision_frame_bgr")
    if frame is not None:
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[2] >= 3:
            preview = Image.fromarray(array[:, :, :3][:, :, ::-1].astype(np.uint8))
            preview.thumbnail((190, 110))
            image.paste(preview, (14, 82))
    active = bool(state.get("vision_active"))
    _button(draw, (212, 80, 303, 101), "停止视觉" if active else "启动视觉",
            primary=active, size=8)
    status = state.get("vision_status") or "等待启动"
    connected = bool(state.get("vision_camera_connected", state.get("camera_connected")))
    inference_running = bool(state.get("vision_inference_running"))
    replay = state.get("vision_source_mode") == "recorded_demo"
    run_status = ("演示视频播放中" if replay and active else
                  "相机在线 · 推理中" if connected and inference_running else
                  "相机在线 · 等待推理" if connected and active else status)
    _text(draw, (212, 106), str(run_status)[:15], 7,
          GREEN if connected else MUTED, True)
    fps = float(state.get("vision_fps") or 0.0)
    camera_fps = float(state.get("vision_camera_fps") or 0.0)
    inference_ms = state.get("vision_inference_ms")
    _text(draw, (212, 120), "模型 {:.2f} FPS".format(fps), 8, BLUE, True)
    _text(draw, (212, 133), "相机 {:.1f} FPS".format(camera_fps), 7, NAVY)
    latency = "--" if inference_ms is None else "{:.0f} ms".format(float(inference_ms))
    _text(draw, (212, 145), "单帧 " + latency, 7, NAVY)
    marker_count = int(state.get("vision_marker_count") or 0)
    result_status = state.get("vision_result_status") or "等待推理"
    if state.get("vision_pose_valid"):
        result_status = "定位有效 · {}点".format(marker_count)
    _text(draw, (212, 158), str(result_status)[:15], 7, BLUE if state.get("vision_pose_valid") else MUTED)
    confidence = state.get("vision_confidence")
    confidence_text = "--" if confidence is None else "{:.0f}%".format(float(confidence) * 100)
    _text(draw, (212, 171), "置信度 " + confidence_text, 7, NAVY)
    if frame is None:
        _text(draw, (108, 137), "暂无摄像头画面", 9, MUTED, True, anchor="mm")
    if state.get("vision_source_mode") == "recorded_demo":
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        badge = ImageDraw.Draw(layer)
        label = "演示回放 · 非实时"
        font = FONTS.get(6, False)
        box = badge.textbbox((0, 0), label, font=font)
        tw = box[2] - box[0]
        x = WIDTH - tw - 9
        y = 202
        badge.rounded_rectangle((x - 4, y - 2, WIDTH - 5, y + 9), radius=3,
                                fill=(23, 34, 48, 105))
        badge.text((x, y), label, font=font, fill=(255, 255, 255, 185))
        image.paste(layer, (0, 0), layer)


def _simple(draw, state, page):
    _card(draw, (8, 72, 312, 204))
    if page == "check":
        status = state.get("status") or {}
        rows = (
            ("ECG 采集器", bool(status.get("id_ok"))),
            ("采集运行", bool(state.get("monitor_running"))),
            ("USB 摄像头", bool(state.get("camera_connected"))),
            ("Wi-Fi 网络", bool(state.get("wifi_connected"))),
        )
        for i, (label, ok) in enumerate(rows):
            y = 86 + i * 25
            draw.ellipse((20, y + 3, 28, y + 11), fill=GREEN if ok else MUTED)
            _text(draw, (36, y), label, 10, NAVY, True)
            _text(draw, (257, y), "正常" if ok else "待连接", 9,
                  GREEN if ok else MUTED, True)
    elif page == "history":
        rows = list(state.get("history_records") or ())
        selected_id = state.get("history_selected_id")
        _text(draw, (17, 79), "测量记录  {} 条".format(len(rows)), 9, BLUE, True)
        if rows:
            for index, row in enumerate(rows[:4]):
                y = 96 + index * 25
                selected = int(row.get("session_id", -1)) == selected_id
                draw.rounded_rectangle((13, y - 2, 307, y + 20), 4,
                                       fill=BLUE_LIGHT if selected else WHITE,
                                       outline=LINE, width=1)
                session_id = int(row.get("session_id", 0))
                date_text = str(row.get("started_at") or "")[:16].replace("T", " ")
                hr = row.get("avg_heart_rate")
                hr_text = "--" if hr is None else "{:.0f}".format(float(hr))
                result = str(row.get("result_label") or "未分析")[:5]
                _text(draw, (19, y + 2), "PA-{:06d}".format(session_id),
                      7, NAVY, True)
                _text(draw, (84, y + 2), date_text[-11:], 7, MUTED)
                _text(draw, (207, y + 2), hr_text, 7, NAVY, True)
                _text(draw, (238, y + 2), result, 7, BLUE)
        else:
            _text(draw, (160, 137), "暂无测量记录", 10, MUTED, True, anchor="mm")
    else:
        _text(draw, (18, 84), "显示设置", 10, BLUE, True)
        _text(draw, (18, 110), "波形增益", 9, MUTED)
        _text(draw, (260, 108), "×{:.1f}".format(float(state.get("gain", 1.0))),
              11, NAVY, True)
        _text(draw, (18, 137), "垂直范围", 9, MUTED)
        _text(draw, (238, 135), str(int(state.get("display_half_range", 100000))),
              10, NAVY, True)


def render_tft_page(page="home", state=None, width=WIDTH, height=HEIGHT):
    """Render the selected HDMI app page using its current shared snapshot."""
    state = state or {}
    page = page if page in PAGE_TITLES else "home"
    image = Image.new("RGB", (int(width), int(height)), BG)
    draw = ImageDraw.Draw(image)
    _header(draw, state, page)

    if page == "home":
        _home(draw, state)
    elif page == "monitor":
        _monitor(draw, state)
    elif page == "leads":
        _leads(draw, state)
    elif page == "lead_focus":
        _lead_focus(draw, state)
    elif page == "ai":
        _ai(draw, state)
    elif page == "spectrum":
        _spectrum(draw, image, state)
    elif page == "trend":
        _trend(draw, state)
    elif page == "vision":
        _vision(draw, image, state)
    else:
        _simple(draw, state, page)

    _nav(draw, page)
    return image


def rgb565_bytes(image):
    """Return tightly packed RGB565 data in ST7789 wire byte order."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint16)
    words = ((rgb[:, :, 0] >> 3) << 11) | ((rgb[:, :, 1] >> 2) << 5) | (rgb[:, :, 2] >> 3)
    return words.astype("<u2", copy=False).tobytes(order="C")


__all__ = ["HEIGHT", "WIDTH", "MAIN_PAGES", "PAGE_TITLES", "HOME_ACTIONS",
           "render_tft_page", "rgb565_bytes"]
