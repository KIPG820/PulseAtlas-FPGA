"""Board-free contract checks for the landscape TFT mirror."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image


BOARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOARD))

from pulseatlas_tft_backend import touch_to_landscape
from pulseatlas_tft_render import (GREEN, MAIN_PAGES, PAGE_TITLES,
                                   _ai_status_label, render_tft_page, rgb565_bytes)
from pulseatlas_board_render import WHITE, _wave_points, render_twelve
from ecg_display_metrics import waveform_envelope_indices


def main():
    assert "vision" in MAIN_PAGES
    empty_state = {
        "status": {"id_ok": False, "enabled": False},
        "heart_rate": None,
        "leads": np.empty((0, 12), dtype=np.int32),
    }
    for page in PAGE_TITLES:
        image = render_tft_page(page, empty_state)
        assert image.size == (320, 240), (page, image.size)

    # The TFT logo must remain inside its compact mark and leave a clean gap
    # before the PulseAtlas wordmark (the HDMI logo's scaled path used to spill).
    home = render_tft_page("home", empty_state)
    assert all(pixel == WHITE for pixel in home.crop((33, 6, 36, 42)).getdata())
    # A device ID remains a live connection even while acquisition is stopped.
    idle_connected = render_tft_page(
        "home", dict(empty_state, status={"id_ok": True, "enabled": False}))
    assert idle_connected.getpixel((165, 20)) == GREEN

    # No connected electrode/data must not be represented by a fabricated ECG.
    monitor = render_tft_page("monitor", empty_state)
    chart = monitor.crop((17, 136, 304, 196))
    assert (221, 60, 83) not in chart.getdata()
    monitor_filter_on = render_tft_page(
        "monitor", dict(empty_state, filter_enabled=True, gain=1.0))
    monitor_filter_off = render_tft_page(
        "monitor", dict(empty_state, filter_enabled=False, gain=1.0))
    assert monitor_filter_on.tobytes() != monitor_filter_off.tobytes()
    monitor_gain_up = render_tft_page(
        "monitor", dict(empty_state, filter_enabled=True, gain=1.5))
    assert monitor_filter_on.tobytes() != monitor_gain_up.tobytes()
    settings_filter_on = render_tft_page(
        "settings", dict(empty_state, filter_enabled=True, gain=1.0))
    settings_filter_off = render_tft_page(
        "settings", dict(empty_state, filter_enabled=False, gain=1.0))
    assert settings_filter_on.tobytes() != settings_filter_off.tobytes()
    settings_gain_up = render_tft_page(
        "settings", dict(empty_state, filter_enabled=True, gain=1.5))
    assert settings_filter_on.tobytes() != settings_gain_up.tobytes()

    # AI status exposes user progress, never the hidden three-run/equality copy.
    assert _ai_status_label({"ai_status": "三次输出一致"}) == "等待开始"
    assert _ai_status_label({"ai_result": {"probabilities": {"NORM": 1.0}}}) == "分析完成"

    # The display envelope must retain a peak in both the early and late parts
    # of a long window rather than stretching only the last pixel-count samples.
    signal = np.zeros(6000, dtype=np.float32)
    signal[500] = 20.0
    signal[5500] = -18.0
    indices = waveform_envelope_indices(signal, 98)
    points = _wave_points(signal, (0, 0, 100, 60))
    assert indices[0] <= 500 < indices[-1] and 5500 in indices
    assert min(x for x, _ in points) < 15 and max(x for x, _ in points) > 85

    # The HDMI 12-lead grid is labeled as six seconds and must pass that window
    # to every trace; prior code drew only 600 ms.
    import pulseatlas_board_render as hdmi_render
    captured = []
    original_plot = hdmi_render._plot_ecg
    def capture_plot(draw, trace, box, **kwargs):
        captured.append(np.asarray(trace).size)
    hdmi_render._plot_ecg = capture_plot
    try:
        render_twelve(dict(empty_state, leads=np.zeros((6000, 12), dtype=np.int32)), 1280, 720)
    finally:
        hdmi_render._plot_ecg = original_plot
    assert captured == [6000] * 12, captured

    # Vision must show the same live, annotated camera frame on the TFT page;
    # merely exposing a "视觉" navigation button is not enough.
    camera_color_bgr = np.empty((60, 100, 3), dtype=np.uint8)
    camera_color_bgr[:] = (13, 155, 230)
    vision_state = dict(
        empty_state,
        vision_frame_bgr=camera_color_bgr,
        vision_active=True,
        vision_camera_connected=True,
        vision_camera_checked=True,
        vision_camera_fps=12.0,
        vision_fps=2.0,
    )
    vision = render_tft_page("vision", vision_state)
    assert vision.getpixel((100, 130)) == (230, 155, 13)
    assert vision.size == (320, 240)

    # The mailbox is little-endian RGB565; the IOP emits each word high byte first.
    assert rgb565_bytes(Image.new("RGB", (1, 1), (255, 0, 0))) == b"\x00\xf8"
    assert rgb565_bytes(Image.new("RGB", (1, 1), (0, 255, 0))) == b"\xe0\x07"
    assert rgb565_bytes(Image.new("RGB", (1, 1), (0, 0, 255))) == b"\x1f\x00"

    assert touch_to_landscape(0, 0, 90) == (319, 0)
    assert touch_to_landscape(239, 319, 90) == (0, 239)
    assert touch_to_landscape(0, 0, 270) == (0, 239)
    assert touch_to_landscape(239, 319, 270) == (319, 0)
    print("landscape TFT render/touch contract passed")


if __name__ == "__main__":
    main()
