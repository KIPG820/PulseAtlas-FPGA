"""NumPy/OpenCV inference for the calibrated-point electrode color groups."""

from __future__ import print_function

import json
import math
from pathlib import Path

import numpy as np

from patch_color_features import color_features, hsv_circular_features


class PatchColorMatcher(object):
    """Compare the observed color group with the expected calibrated point."""

    def __init__(self, model_path):
        with Path(model_path).open("r", encoding="utf-8") as stream:
            model = json.load(stream)
        if model.get("schema_version") != 1:
            raise ValueError("unsupported color model schema")
        self.groups = tuple(model["groups"])
        self.method = model.get("method", "center_bgr_mean_std_lda")
        if self.method == "center_bgr_mean_std_lda":
            self._feature_fn = color_features
        elif self.method == "center_hsv_circular_lda":
            self._feature_fn = hsv_circular_features
        else:
            raise ValueError("unsupported color feature method: {}".format(self.method))
        self.expected_group_by_point = dict(model["expected_group_by_point"])
        self.ambiguous_pairs = {
            frozenset((str(pair[0]), str(pair[1])))
            for pair in model.get("ambiguous_group_pairs", [])
        }
        self.minimum_confidence = float(
            model.get("minimum_confidence_for_mismatch_candidate", 0.70))
        self.mean = np.asarray(model["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(model["scaler_scale"], dtype=np.float32)
        self.coefficients = np.asarray(model["lda_coefficients"], dtype=np.float32)
        self.intercepts = np.asarray(model["lda_intercepts"], dtype=np.float32)
        if self.coefficients.shape != (len(self.groups), self.mean.size):
            raise ValueError("color model coefficient dimensions are inconsistent")
        if self.intercepts.shape != (len(self.groups),) or self.scale.shape != self.mean.shape:
            raise ValueError("color model scaler/intercept dimensions are inconsistent")
        if np.any(~np.isfinite(self.scale)) or np.any(self.scale == 0):
            raise ValueError("color model contains invalid feature scales")

    def decide(self, observed_group, expected_group, confidence):
        if expected_group is None:
            return "unknown", "position_presence"
        if observed_group == expected_group:
            return "match", "color_match"
        if frozenset((observed_group, expected_group)) in self.ambiguous_pairs:
            return "ambiguous", "position_presence"
        if confidence < self.minimum_confidence:
            return "ambiguous", "position_presence"
        return "mismatch_candidate", "color_mismatch_candidate"

    def classify(self, bgr_crop, expected_point=None):
        features = self._feature_fn(bgr_crop)
        normalized = (features - self.mean) / self.scale
        scores = np.dot(self.coefficients, normalized) + self.intercepts
        scores = scores - float(np.max(scores))
        exp_scores = np.exp(scores)
        denominator = float(np.sum(exp_scores))
        probabilities = exp_scores / denominator if denominator > 0 else np.zeros_like(exp_scores)
        predicted_index = int(np.argmax(probabilities))
        observed = self.groups[predicted_index]
        confidence = float(probabilities[predicted_index])
        expected = self.expected_group_by_point.get(expected_point)
        margin = (float(np.partition(probabilities, -1)[-1] - np.partition(probabilities, -2)[-2])
                  if len(probabilities) > 1 else confidence)
        color_status, decision = self.decide(observed, expected, confidence)
        if not math.isfinite(confidence):
            confidence = 0.0
        return {
            "color_status": color_status,
            "color_decision": decision,
            "expected_color_group": expected,
            "detected_color_group": observed,
            "color_confidence": confidence,
            "color_probability_margin": margin,
        }
