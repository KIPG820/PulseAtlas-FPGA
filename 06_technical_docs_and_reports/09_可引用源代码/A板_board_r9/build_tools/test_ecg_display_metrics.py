"""Numerical regression checks for the scientific ECG displays."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BOARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOARD))

from ecg_display_metrics import (  # noqa: E402
    SPECTROGRAM_INVALID,
    TREND_INVALID,
    spectrogram_dbfs,
    trend_points,
)


class TrendTests(unittest.TestCase):
    def test_empty_history_stays_empty(self):
        output = trend_points({})
        self.assertTrue(np.all(output == TREND_INVALID))

    def test_normal_beats_connect_but_long_gap_does_not(self):
        # 2 Hz plot bins; one-second RR intervals should form a continuous
        # trend, while a ten-second acquisition gap must remain visible.
        frames = np.asarray([1000, 2000, 3000, 13000, 14000], dtype=np.float64)
        bpm = np.asarray([60, 62, 61, 70, 69], dtype=np.float64)
        output = trend_points(
            {"heart_rate_events": np.column_stack((frames, bpm)),
             "current_frame": 14000},
            points=40, seconds=20, sample_rate=1000,
        )
        self.assertGreater(np.count_nonzero(output == TREND_INVALID), 10)
        self.assertEqual(int(output[-1]), 69)

    def test_frame_counter_wrap_is_handled(self):
        output = trend_points(
            {"heart_rate_events": np.asarray([[0xFFFFFFFE, 80.0]]),
             "current_frame": 2},
            points=10, seconds=1, sample_rate=10,
        )
        self.assertIn(80, output)


class SpectrogramTests(unittest.TestCase):
    @staticmethod
    def _tone(amplitude=0.1, discontinuity=False):
        count = 10000
        time = np.arange(count, dtype=np.float64) / 1000.0
        leads = np.zeros((count, 12), dtype=np.int32)
        leads[:, 1] = np.rint(amplitude * (1 << 23) *
                              np.sin(2.0 * np.pi * 50.0 * time)).astype(np.int32)
        frames = np.arange(count, dtype=np.uint32)
        if discontinuity:
            frames[5000:] += np.uint32(25)
        return {"leads": leads, "frames": frames}

    def test_peak_is_at_fifty_hz(self):
        matrix = spectrogram_dbfs(self._tone())
        means = np.mean(np.where(matrix == SPECTROGRAM_INVALID, -120, matrix), axis=1)
        peak_frequency = (100.0 * (matrix.shape[0] - 1 - int(np.argmax(means))) /
                          (matrix.shape[0] - 1))
        self.assertAlmostEqual(peak_frequency, 50.0, delta=2.1)

    def test_doubling_amplitude_adds_six_db(self):
        quiet = spectrogram_dbfs(self._tone(0.05))
        loud = spectrogram_dbfs(self._tone(0.10))
        delta = float(np.max(loud[loud != SPECTROGRAM_INVALID]) -
                      np.max(quiet[quiet != SPECTROGRAM_INVALID]))
        self.assertAlmostEqual(delta, 6.0, delta=1.0)

    def test_windows_crossing_frame_gap_are_invalid(self):
        matrix = spectrogram_dbfs(self._tone(discontinuity=True))
        invalid_columns = np.all(matrix == SPECTROGRAM_INVALID, axis=0)
        self.assertGreaterEqual(int(np.count_nonzero(invalid_columns)), 7)
        self.assertLessEqual(int(np.count_nonzero(invalid_columns)), 9)


if __name__ == "__main__":
    unittest.main()
