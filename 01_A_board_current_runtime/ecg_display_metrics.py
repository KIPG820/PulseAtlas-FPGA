"""Scientific display preparation shared by HDMI and TFT renderers.

The physical ADC-to-voltage gain has not yet been verified with a calibrated
input, so spectral power is expressed against the fixed ADS1298 full-scale
code (dBFS/Hz).  This keeps colours comparable over time without claiming an
unverified microvolt calibration.
"""

from __future__ import annotations

import numpy as np


SAMPLE_RATE = 1000.0
TREND_SECONDS = 120.0
TREND_POINTS = 240
TREND_INVALID = np.iinfo(np.int32).min
SPECTROGRAM_WINDOW = 512
SPECTROGRAM_HOP = 64
SPECTROGRAM_COLUMNS = 149
SPECTROGRAM_ROWS = 52
SPECTROGRAM_INVALID = -128
SPECTROGRAM_DB_MIN = -120.0
SPECTROGRAM_DB_MAX = -20.0
ADS1298_FULL_SCALE_CODE = float(1 << 23)


def _u32_age(current, earlier):
    """Return unsigned 32-bit sample age, preserving counter wrap."""
    return (np.uint64(int(current) & 0xFFFFFFFF) -
            np.asarray(earlier, dtype=np.uint32).astype(np.uint64)) & np.uint64(0xFFFFFFFF)



def waveform_envelope_indices(values, pixel_width):
    """Return time-ordered min/max sample indices for a pixel-width plot.

    Every horizontal pixel bucket contributes its minimum and maximum sample,
    preserving narrow QRS peaks while spanning the complete requested window.
    """
    samples = np.asarray(values).reshape(-1)
    count = int(samples.size)
    width = max(1, int(pixel_width))
    if count <= 2 * width:
        return np.arange(count, dtype=np.int32)
    edges = np.linspace(0, count, width + 1, dtype=np.int64)
    selected = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        bucket = samples[start:stop]
        low = int(start + np.argmin(bucket))
        high = int(start + np.argmax(bucket))
        if low == high:
            selected.append(low)
        elif low < high:
            selected.extend((low, high))
        else:
            selected.extend((high, low))
    return np.asarray(selected, dtype=np.int32)


def trend_points(state, points=TREND_POINTS, seconds=TREND_SECONDS,
                 sample_rate=SAMPLE_RATE):
    """Place valid beat-to-beat rates on a real, fixed-duration time axis.

    Empty bins remain LVGL's point-none sentinel.  Values are never invented
    and lines therefore break across missing QRS events or acquisition gaps.
    """
    output = np.full(int(points), TREND_INVALID, dtype=np.int32)
    events = np.asarray(state.get("heart_rate_events", ()), dtype=np.float64)
    if events.size == 0:
        return output
    try:
        events = events.reshape(-1, 2)
    except ValueError:
        return output
    current = state.get("current_frame")
    if current is None:
        current = int(events[-1, 0])
    age_samples = _u32_age(current, events[:, 0])
    ages = age_samples.astype(np.float64) / float(sample_rate)
    rates = events[:, 1]
    valid = (ages >= 0.0) & (ages <= float(seconds)) & np.isfinite(rates)
    valid &= (rates >= 30.0) & (rates <= 220.0)
    if not np.any(valid):
        return output
    ages, rates = ages[valid], rates[valid]
    bins = np.floor((float(seconds) - ages) * int(points) / float(seconds)).astype(np.int32)
    bins = np.clip(bins, 0, int(points) - 1)
    for index in np.unique(bins):
        output[index] = int(round(float(np.median(rates[bins == index]))))
    # Connect adjacent measured beats across physiologically ordinary RR
    # intervals.  Longer intervals stay empty, making acquisition/QRS gaps
    # visible instead of silently extending or inventing a value.
    measured = np.flatnonzero(output != TREND_INVALID)
    max_connect_bins = max(1, int(round(3.0 * int(points) / float(seconds))))
    for left, right in zip(measured[:-1], measured[1:]):
        if right - left <= max_connect_bins:
            interpolated = np.linspace(output[left], output[right], right - left + 1)
            output[left:right + 1] = np.rint(interpolated).astype(np.int32)
    return output


def spectrogram_dbfs(state, rows=SPECTROGRAM_ROWS,
                     columns=SPECTROGRAM_COLUMNS,
                     sample_rate=SAMPLE_RATE):
    """Return a fixed-scale, gap-aware Lead-II PSD spectrogram.

    Output is frequency-descending int8 dBFS/Hz.  -128 marks unavailable data.
    The 512-point Hann window and 64-sample hop give 1.953125 Hz bins and a
    64 ms column interval at 1000 SPS.  Windows crossing a frame discontinuity
    are explicitly invalidated.
    """
    result = np.full((int(rows), int(columns)), SPECTROGRAM_INVALID, dtype=np.int8)
    leads = np.asarray(state.get("leads", ()))
    frames = np.asarray(state.get("frames", ()), dtype=np.uint32).reshape(-1)
    if leads.ndim != 2 or leads.shape[1] < 2 or leads.shape[0] < SPECTROGRAM_WINDOW:
        return result
    count = min(10000, leads.shape[0])
    signal = leads[-count:, 1].astype(np.float64, copy=True)
    if frames.size >= count:
        frames = frames[-count:]
    else:
        frames = np.empty(0, dtype=np.uint32)
    starts = np.arange(0, count - SPECTROGRAM_WINDOW + 1,
                       SPECTROGRAM_HOP, dtype=np.int32)
    if starts.size > int(columns):
        starts = starts[-int(columns):]
    windows = np.stack([signal[start:start + SPECTROGRAM_WINDOW]
                        for start in starts])
    windows -= windows.mean(axis=1, keepdims=True)
    hann = np.hanning(SPECTROGRAM_WINDOW)
    transformed = np.fft.rfft(windows * hann[None, :], axis=1)
    psd = np.abs(transformed) ** 2
    psd /= float(sample_rate) * float(np.sum(hann ** 2)) * ADS1298_FULL_SCALE_CODE ** 2
    psd[:, 1:-1] *= 2.0
    source_frequency = np.fft.rfftfreq(SPECTROGRAM_WINDOW, 1.0 / float(sample_rate))
    target_frequency = np.linspace(0.0, 100.0, int(rows))
    sampled = np.stack([np.interp(target_frequency, source_frequency, row)
                        for row in psd])
    db = 10.0 * np.log10(np.maximum(sampled, np.finfo(np.float64).tiny))
    db = np.clip(db, SPECTROGRAM_DB_MIN, SPECTROGRAM_DB_MAX)
    window_valid = np.ones(starts.size, dtype=bool)
    if frames.size == count:
        step_ok = ((frames[1:].astype(np.uint64) - frames[:-1].astype(np.uint64)) &
                   np.uint64(0xFFFFFFFF)) == 1
        bad_prefix = np.concatenate(([0], np.cumsum(~step_ok, dtype=np.int32)))
        for index, start in enumerate(starts):
            window_valid[index] = (bad_prefix[start + SPECTROGRAM_WINDOW - 1] -
                                   bad_prefix[start]) == 0
    encoded = np.rint(db[:, ::-1].T).astype(np.int8)
    encoded[:, ~window_valid] = SPECTROGRAM_INVALID
    result[:, -encoded.shape[1]:] = encoded
    return np.ascontiguousarray(result)


__all__ = [
    "TREND_INVALID", "TREND_POINTS", "SPECTROGRAM_COLUMNS",
    "SPECTROGRAM_ROWS", "SPECTROGRAM_INVALID", "SPECTROGRAM_DB_MIN",
    "SPECTROGRAM_DB_MAX", "trend_points", "spectrogram_dbfs",
    "waveform_envelope_indices",
]
