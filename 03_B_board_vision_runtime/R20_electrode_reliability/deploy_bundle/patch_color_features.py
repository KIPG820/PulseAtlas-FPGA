"""Small OpenCV-only feature extractor shared by training and the board."""

import cv2
import numpy as np


FEATURE_REGION = (0.25, 0.75, 0.05, 0.75)  # x0, x1, y0, y1 in the 96x96 ROI


def color_features(bgr):
    """Return normalized BGR means and standard deviations from the pad center."""
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR crop")
    height, width = bgr.shape[:2]
    x0, x1, y0, y1 = FEATURE_REGION
    crop = bgr[int(height * y0):max(int(height * y0) + 1, int(height * y1)),
               int(width * x0):max(int(width * x0) + 1, int(width * x1))]
    mean, std = cv2.meanStdDev(crop)
    return np.concatenate((mean.reshape(-1), std.reshape(-1))).astype(np.float32) / 255.0


def hsv_circular_features(bgr):
    """Return six illumination-tolerant HSV features from the pad center.

    Hue is represented as a saturation-weighted circular mean so red does not
    split at OpenCV hue 0/180. Saturation and value statistics preserve cues
    for low-saturation dark pads where hue itself is undefined.
    """
    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 BGR crop")
    height, width = bgr.shape[:2]
    x0, x1, y0, y1 = FEATURE_REGION
    crop = bgr[int(height * y0):max(int(height * y0) + 1, int(height * y1)),
               int(width * x0):max(int(width * x0) + 1, int(width * x1))]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * (2.0 * np.pi / 180.0)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    weight = float(saturation.sum())
    hue_cos = float((np.cos(hue) * saturation).sum() / weight) if weight > 1e-5 else 0.0
    hue_sin = float((np.sin(hue) * saturation).sum() / weight) if weight > 1e-5 else 0.0
    return np.asarray((hue_cos, hue_sin, saturation.mean(), saturation.std(),
                       value.mean(), value.std()), dtype=np.float32)
