"""Calibrated-point local ROI electrode presence checker for PYNQ-Z2.

The model answers only whether an electrode-like pad is visible around a
trusted calibration point. It does not identify the lead, verify skin contact,
or establish anatomical correctness.
"""

from __future__ import print_function

import math
import time
from collections import deque

import cv2
import numpy as np


LABEL_ORDER = ("RA", "LA", "RL", "LL", "V1", "V2", "V3", "V4", "V5", "V6")
MODEL_INPUT_SIZE = 96
MODEL_BATCH_SIZE = 10
DEFAULT_NEIGHBOR_SCALE = 0.60


def _xy(point):
    """Accept ElectrodePoint-like dicts, tuples, or objects with x/y fields."""
    if isinstance(point, dict):
        x = point.get("x_px", point.get("x"))
        y = point.get("y_px", point.get("y"))
    elif isinstance(point, (tuple, list)) and len(point) >= 2:
        x, y = point[0], point[1]
    else:
        x = getattr(point, "x", None)
        y = getattr(point, "y", None)
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _roi_side(label, point_xy, valid_points, frame_width,
              neighbor_scale=DEFAULT_NEIGHBOR_SCALE, fallback_fraction=0.08):
    """Scale each crop from nearby calibrated sites, with a resolution fallback.

    The scale was calibrated offline against electrode-box crops from the
    supplied dataset. A 0.60 nearest-anchor spacing produced the best balance
    of pad recall and separation in that data.
    """
    width = max(1, int(frame_width))
    min_side = max(24, int(round(width * 0.015)))
    max_side = max(min_side, min(320, int(round(width * 0.25))))
    fallback = int(round(width * float(fallback_fraction)))
    fallback = max(min_side, min(max_side, fallback))
    distances = []
    for other_label, other_xy in valid_points.items():
        if other_label == label:
            continue
        distance = math.hypot(point_xy[0] - other_xy[0], point_xy[1] - other_xy[1])
        if math.isfinite(distance) and distance >= 1.0:
            distances.append(distance)
    reference_spacing = min(distances) if distances else None
    side = fallback if reference_spacing is None else int(round(reference_spacing * neighbor_scale))
    side = max(min_side, min(max_side, side))
    return side, reference_spacing


def crop_square_reflect(frame, center_x, center_y, side):
    """Crop a square using the same reflect-101 edge treatment as training."""
    height, width = frame.shape[:2]
    side = max(16, int(round(side)))
    x1 = int(round(float(center_x) - side / 2.0))
    y1 = int(round(float(center_y) - side / 2.0))
    x2, y2 = x1 + side, y1 + side
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - width), max(0, y2 - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        y1 += pad_top
    crop = frame[y1:y1 + side, x1:x1 + side]
    if crop.shape[:2] != (side, side):
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(crop, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                       interpolation=cv2.INTER_AREA)


class CalibratedPatchPresenceChecker(object):
    """Run one static batch of local crops around supplied calibration points."""

    def __init__(self, model_path, threads=2, neighbor_scale=DEFAULT_NEIGHBOR_SCALE,
                 net=None, color_model_path=None, color_matcher=None,
                 color_neighbor_scale=0.90, color_fallback_fraction=0.10):
        self.model_path = str(model_path)
        self.neighbor_scale = float(neighbor_scale)
        if not math.isfinite(self.neighbor_scale) or self.neighbor_scale <= 0:
            raise ValueError("neighbor_scale must be a positive finite number")
        self.color_neighbor_scale = float(color_neighbor_scale)
        self.color_fallback_fraction = float(color_fallback_fraction)
        if (not math.isfinite(self.color_neighbor_scale) or self.color_neighbor_scale <= 0 or
                not math.isfinite(self.color_fallback_fraction) or
                self.color_fallback_fraction <= 0):
            raise ValueError("color ROI scales must be positive finite numbers")
        self.threads = max(1, int(threads))
        cv2.setNumThreads(self.threads)
        self.net = net if net is not None else cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.color_matcher = color_matcher
        self.temporal_window = 3
        self.temporal_votes_required = 2
        self.temporal_max_gap_s = 2.5
        self._presence_history = {}
        if self.color_matcher is None and color_model_path is not None:
            from patch_color_matcher import PatchColorMatcher
            self.color_matcher = PatchColorMatcher(color_model_path)

    @staticmethod
    def _softmax(logits):
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        values = values - float(np.max(values))
        exp_values = np.exp(values)
        denom = float(np.sum(exp_values))
        if not math.isfinite(denom) or denom <= 0:
            return np.asarray([0.5, 0.5], dtype=np.float32)
        return exp_values / denom

    def _temporal_confirmation(self, label, raw_status, pad_probability,
                               color_result, observed_at):
        """Require a positive in two of the last three fresh frame inferences."""
        state = self._presence_history.get(label)
        if state is None or observed_at - state["updated_at"] > self.temporal_max_gap_s:
            state = {"updated_at": observed_at, "votes": deque(maxlen=self.temporal_window)}
            self._presence_history[label] = state
        positive = (raw_status in ("pad_present", "pad_color_mismatch") and
                    pad_probability >= 0.5)
        vote = {"present": positive, "raw_status": raw_status,
                "color_result": dict(color_result)}
        state["votes"].append(vote)
        state["updated_at"] = observed_at
        votes = list(state["votes"])
        positive_count = sum(item["present"] for item in votes)

        if positive_count >= self.temporal_votes_required:
            mismatch_groups = {}
            for item in votes:
                details = item["color_result"]
                if (item["present"] and item["raw_status"] == "pad_color_mismatch" and
                        details.get("detected_color_group")):
                    group = details["detected_color_group"]
                    mismatch_groups[group] = mismatch_groups.get(group, 0) + 1
            confirmed_group = next((group for group, count in mismatch_groups.items()
                                    if count >= self.temporal_votes_required), None)
            if confirmed_group is not None:
                details = next(item["color_result"] for item in reversed(votes)
                               if item["raw_status"] == "pad_color_mismatch" and
                               item["color_result"].get("detected_color_group") == confirmed_group)
                return "pad_color_mismatch", "temporal_color_confirmation", details, positive_count
            if raw_status == "pad_color_mismatch":
                return "pad_pending", "color_confirmation_pending", color_result, positive_count
            return "pad_present", "temporal_presence_confirmation", color_result, positive_count

        if positive:
            return "pad_pending", "awaiting_temporal_confirmation", color_result, positive_count
        return "pad_not_found", "local_roi_classifier", color_result, positive_count

    def predict(self, frame, calibrated_points):
        """Return per-point visible-pad/absent/unknown results for this frame.

        Coordinates must refer to the same unmirrored camera frame. Expected
        point values are ``{"x_px": ..., "y_px": ...}``, ``(x, y)`` tuples,
        or objects with ``x`` and ``y`` attributes.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        if not isinstance(calibrated_points, dict):
            raise ValueError("calibrated_points must be a label-to-point mapping")

        height, width = frame.shape[:2]
        points = {}
        for label in LABEL_ORDER:
            xy = _xy(calibrated_points.get(label))
            if xy is not None and 0 <= xy[0] < width and 0 <= xy[1] < height:
                points[label] = xy

        results = {}
        jobs = []
        for label in LABEL_ORDER:
            xy = _xy(calibrated_points.get(label))
            if xy is None:
                self._presence_history.pop(label, None)
                results[label] = {"status": "unknown", "reason": "missing_or_invalid_point"}
                continue
            if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                self._presence_history.pop(label, None)
                results[label] = {"status": "unknown", "reason": "point_outside_frame",
                                  "point_xy": [xy[0], xy[1]]}
                continue
            side, spacing = _roi_side(label, xy, points, width, self.neighbor_scale)
            crop = crop_square_reflect(frame, xy[0], xy[1], side)
            if self.color_matcher is not None:
                color_side, _ = _roi_side(
                    label, xy, points, width, self.color_neighbor_scale,
                    self.color_fallback_fraction)
                color_crop = crop_square_reflect(frame, xy[0], xy[1], color_side)
            else:
                color_side, color_crop = side, crop
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            jobs.append((label, xy, side, spacing, tensor, crop, color_side, color_crop))

        if not jobs:
            return {"status": "no_valid_points", "frame_size": [width, height],
                    "coordinate_space": "unmirrored_camera_pixels", "per_point": results}

        batch = np.zeros((MODEL_BATCH_SIZE, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                         dtype=np.float32)
        for index, job in enumerate(jobs):
            if index >= MODEL_BATCH_SIZE:
                break
            batch[index] = job[4]
        # The ONNX export uses a fixed batch of ten; repeat the final valid crop
        # for missing/off-screen targets and ignore those padded outputs.
        for index in range(len(jobs), MODEL_BATCH_SIZE):
            batch[index] = batch[len(jobs) - 1]

        start = time.perf_counter()
        self.net.setInput(batch)
        logits = np.asarray(self.net.forward(), dtype=np.float32).reshape(MODEL_BATCH_SIZE, -1)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        color_started = time.perf_counter()
        observed_at = time.monotonic()
        for index, (label, xy, side, spacing, _, _, color_side, color_crop) in enumerate(
                jobs[:MODEL_BATCH_SIZE]):
            probabilities = self._softmax(logits[index])
            pad_probability = float(probabilities[0])
            no_pad_probability = float(probabilities[1])
            color_result = (self.color_matcher.classify(color_crop, label)
                            if self.color_matcher is not None and
                            pad_probability >= no_pad_probability else {})
            if color_result:
                color_result["color_roi_side_px"] = color_side
            if pad_probability >= no_pad_probability:
                if color_result.get("color_decision") == "color_mismatch_candidate":
                    status = "pad_color_mismatch"
                    reason = "confident_color_group_differs_from_expected"
                else:
                    status = "pad_present"
                    reason = ("color_match" if color_result.get("color_decision") == "color_match"
                              else "position_presence_fallback"
                              if color_result.get("color_decision") == "position_presence"
                              else "local_roi_classifier")
            else:
                status = "pad_not_found"
                reason = "local_roi_classifier"
            raw_status = status
            status, reason, color_result, positive_votes = self._temporal_confirmation(
                label, raw_status, pad_probability, color_result, observed_at)
            result = {
                "status": status,
                "raw_status": raw_status,
                "reason": reason,
                "presence_confirmed": status in ("pad_present", "pad_color_mismatch"),
                "presence_positive_votes": positive_votes,
                "presence_vote_window": self.temporal_window,
                "point_xy": [xy[0], xy[1]],
                "roi_side_px": side,
                "nearest_calibration_spacing_px": None if spacing is None else round(float(spacing), 2),
                "pad_probability": pad_probability,
                "no_pad_probability": no_pad_probability,
            }
            result.update(color_result)
            results[label] = result
        color_elapsed_ms = (time.perf_counter() - color_started) * 1000.0

        return {
            "status": "ok",
            "frame_size": [width, height],
            "coordinate_space": "unmirrored_camera_pixels",
            "scope": "visual pad presence near trusted calibration point only",
            "contact_or_connection_verified": False,
            "lead_identity_verified": False,
            "inference_ms": elapsed_ms,
            "color_match_ms": color_elapsed_ms if self.color_matcher is not None else 0.0,
            "inferred_points": min(len(jobs), MODEL_BATCH_SIZE),
            "per_point": results,
        }
