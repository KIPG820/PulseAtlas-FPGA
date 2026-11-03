"""Body-relative placement helpers for the PC electrode-guidance prototype.

Pose keypoints can only provide rough limb reference points. Chest landmarks
must be palpated and clicked by the operator; V3 is then derived halfway
between V2 and V4 as specified by standard 12-lead placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
from collections import deque


COCO_SHOULDER_KEYPOINTS = {"LA": 5, "RA": 6}

CHEST_CALIBRATION_ORDER = ("V1", "V2", "V4", "V5", "V6")
CHEST_INSTRUCTIONS = {
    "V1": "4th intercostal space, right sternal border",
    "V2": "4th intercostal space, left sternal border",
    "V4": "5th intercostal space, midclavicular line",
    "V5": "Anterior axillary line, same level as V4",
    "V6": "Midaxillary line, same level as V4",
}


@dataclass(frozen=True)
class ElectrodePoint:
    label: str
    x: int
    y: int
    method: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "x_px": int(self.x),
            "y_px": int(self.y),
            "method": self.method,
            "confidence": None if self.confidence is None else round(float(self.confidence), 5),
        }


@dataclass
class PersonPose:
    bbox: tuple[int, int, int, int]
    confidence: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection) / union if union else 0.0


class PersonPoseEstimator:
    """YOLO pose adapter that selects the pose matching the segmented person."""

    def __init__(
        self,
        weights: str | Path = "yolo11n-pose.pt",
        *,
        image_size: int = 384,
        confidence: float = 0.25,
        device: str | int | None = None,
        model: Any | None = None,
    ) -> None:
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.device = device
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(weights))

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def detect(
        self,
        frame: np.ndarray,
        target_bbox: tuple[int, int, int, int],
        *,
        minimum_iou: float = 0.05,
    ) -> PersonPose | None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            max_det=8,
            verbose=False,
        )
        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            return None
        boxes_xyxy = self._numpy(boxes.xyxy)
        box_scores = self._numpy(boxes.conf).reshape(-1)
        points_xy = self._numpy(keypoints.xy)
        points_conf = getattr(keypoints, "conf", None)
        if points_conf is None:
            return None
        points_conf = self._numpy(points_conf)
        if points_xy.ndim != 3 or points_conf.ndim != 2:
            return None

        candidates: list[tuple[float, float, int]] = []
        for index, raw_box in enumerate(boxes_xyxy):
            if index >= len(points_xy) or index >= len(points_conf) or index >= len(box_scores):
                break
            box = tuple(int(round(float(value))) for value in raw_box)
            overlap = _box_iou(target_bbox, box)
            candidates.append((overlap, float(box_scores[index]), index))
        if not candidates:
            return None
        overlap, detection_confidence, index = max(candidates, key=lambda item: (item[0], item[1]))
        if overlap < minimum_iou:
            return None
        x1, y1, x2, y2 = [int(round(float(value))) for value in boxes_xyxy[index]]
        return PersonPose(
            bbox=(x1, y1, x2, y2),
            confidence=detection_confidence,
            keypoints_xy=np.asarray(points_xy[index], dtype=np.float32),
            keypoint_scores=np.asarray(points_conf[index], dtype=np.float32),
        )


def map_torso_electrodes(
    keypoints_xy: np.ndarray | None,
    keypoint_scores: np.ndarray | None,
    upper_body_bbox: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    *,
    min_confidence: float = 0.25,
) -> dict[str, ElectrodePoint]:
    """Estimate torso-style limb electrode sites from a torso box and shoulders.

    RA/LA are placed just medial/inferior to the shoulder keypoints. RL/LL are
    placed in the lower torso band. If the shoulder keypoints are unavailable,
    symmetric positions are estimated from the upper-body box. This is a
    Mason-Likar-style visual guide, not conventional distal limb placement.
    """
    width, height = frame_size
    if width < 2 or height < 2:
        raise ValueError("frame_size must be at least 2x2")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    x1, y1, x2, y2 = [int(value) for value in upper_body_bbox]
    if x2 <= x1 or y2 <= y1:
        return {}
    if x2 - x1 < 30 or y2 - y1 < 80:
        return {}

    left_shoulder = right_shoulder = None
    confidence = None
    if keypoints_xy is not None and keypoint_scores is not None:
        points = np.asarray(keypoints_xy, dtype=np.float32)
        scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
        if points.ndim == 2 and points.shape[1] == 2 and len(scores) >= len(points):
            left_index = COCO_SHOULDER_KEYPOINTS["LA"]
            right_index = COCO_SHOULDER_KEYPOINTS["RA"]
            if max(left_index, right_index) < len(points):
                left_score, right_score = float(scores[left_index]), float(scores[right_index])
                if (
                    np.isfinite(left_score) and np.isfinite(right_score)
                    and left_score >= min_confidence and right_score >= min_confidence
                    and np.isfinite(points[[left_index, right_index]]).all()
                ):
                    left_shoulder = points[left_index]
                    right_shoulder = points[right_index]
                    confidence = min(left_score, right_score)

    box_width, box_height = x2 - x1, y2 - y1
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_vector = left_shoulder - right_shoulder
        shoulder_span = float(np.linalg.norm(shoulder_vector))
        if shoulder_span >= max(18.0, box_width * 0.12):
            center = (left_shoulder + right_shoulder) * 0.5
            drop = max(2.0, shoulder_span * 0.07)
            inward = 0.28
            ra_xy = right_shoulder + inward * (center - right_shoulder) + np.asarray([0.0, drop])
            la_xy = left_shoulder + inward * (center - left_shoulder) + np.asarray([0.0, drop])
            lower_y = y1 + 0.82 * box_height
            rl_x = center[0] + 0.62 * (right_shoulder[0] - center[0])
            ll_x = center[0] + 0.62 * (left_shoulder[0] - center[0])
            candidates = {
                "RA": (ra_xy[0], ra_xy[1]),
                "LA": (la_xy[0], la_xy[1]),
                "RL": (rl_x, lower_y),
                "LL": (ll_x, lower_y),
            }
            method = "pose-mason-likar-estimate"
        else:
            candidates = {}
            method = "silhouette-mason-likar-estimate"
    else:
        candidates = {}
        method = "silhouette-mason-likar-estimate"

    if not candidates:
        candidates = {
            "RA": (x1 + 0.35 * box_width, y1 + 0.30 * box_height),
            "LA": (x1 + 0.65 * box_width, y1 + 0.30 * box_height),
            "RL": (x1 + 0.36 * box_width, y1 + 0.82 * box_height),
            "LL": (x1 + 0.64 * box_width, y1 + 0.82 * box_height),
        }

    placements: dict[str, ElectrodePoint] = {}
    for label, (x, y) in candidates.items():
        x = int(round(float(np.clip(x, x1, min(x2 - 1, width - 1)))))
        y = int(round(float(np.clip(y, y1, min(y2 - 1, height - 1)))))
        placements[label] = ElectrodePoint(
            label=label,
            x=x,
            y=y,
            method=method,
            confidence=confidence if method.startswith("pose-") else None,
        )
    return placements


class TemporalPointStabilizer:
    """Suppress point jitter while still following actual body movement."""

    def __init__(
        self,
        *,
        median_window: int = 3,
        smoothing: float = 0.28,
        fast_move_fraction: float = 0.04,
        reset_move_fraction: float = 0.18,
    ) -> None:
        if median_window < 1:
            raise ValueError("median_window must be positive")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1")
        if not 0.0 < fast_move_fraction < reset_move_fraction <= 1.0:
            raise ValueError("movement thresholds must satisfy 0 < fast < reset <= 1")
        self.median_window = int(median_window)
        self.smoothing = float(smoothing)
        self.fast_move_fraction = float(fast_move_fraction)
        self.reset_move_fraction = float(reset_move_fraction)
        self._history: dict[str, Any] = {}
        self._smoothed: dict[str, np.ndarray] = {}
        self._bbox: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self._smoothed.clear()
        self._bbox = None

    def update(
        self,
        points: dict[str, ElectrodePoint],
        body_bbox: tuple[int, int, int, int] | None,
    ) -> dict[str, ElectrodePoint]:
        """Median-filter short spikes, then adapt smoothing to detected movement."""
        if not points or body_bbox is None:
            self.reset()
            return {}

        current_bbox = np.asarray(body_bbox, dtype=np.float32)
        x1, y1, x2, y2 = current_bbox.tolist()
        if x2 <= x1 or y2 <= y1:
            self.reset()
            return {}

        alpha = self.smoothing
        if self._bbox is None:
            self.reset()
        else:
            old_x1, old_y1, old_x2, old_y2 = self._bbox.tolist()
            old_center = np.asarray([(old_x1 + old_x2) * 0.5, (old_y1 + old_y2) * 0.5])
            new_center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            old_diagonal = max(1.0, float(np.hypot(old_x2 - old_x1, old_y2 - old_y1)))
            movement = float(np.linalg.norm(new_center - old_center)) / old_diagonal
            old_size = np.asarray([max(1.0, old_x2 - old_x1), max(1.0, old_y2 - old_y1)])
            new_size = np.asarray([x2 - x1, y2 - y1])
            scale_change = float(np.max(np.abs(new_size - old_size) / old_size))
            if movement >= self.reset_move_fraction or scale_change >= 0.30:
                self._history.clear()
                self._smoothed.clear()
                alpha = 1.0
            elif movement >= self.fast_move_fraction or scale_change >= 0.08:
                alpha = 0.72

        output: dict[str, ElectrodePoint] = {}
        for label, point in points.items():
            history = self._history.get(label)
            if history is None:
                history = deque(maxlen=self.median_window)
                self._history[label] = history
            raw = np.asarray([point.x, point.y], dtype=np.float32)
            history.append(raw)
            robust = np.median(np.stack(tuple(history)), axis=0)
            previous = self._smoothed.get(label)
            smoothed = robust if previous is None else (1.0 - alpha) * previous + alpha * robust
            self._smoothed[label] = smoothed
            output[label] = ElectrodePoint(
                label=point.label,
                x=int(round(float(smoothed[0]))),
                y=int(round(float(smoothed[1]))),
                method=point.method,
                confidence=point.confidence,
            )

        # A dropped label must not reuse an old track if it reappears later.
        for label in set(self._history) - set(points):
            self._history.pop(label, None)
            self._smoothed.pop(label, None)
        self._bbox = current_bbox
        return output


class ChestLeadCalibration:
    """Store manually selected precordial anchors in the shoulder-to-hip torso frame."""

    VERSION = 2

    def __init__(self) -> None:
        self._anchors: dict[str, tuple[float, float]] = {}

    @property
    def next_label(self) -> str | None:
        for label in CHEST_CALIBRATION_ORDER:
            if label not in self._anchors:
                return label
        return None

    @property
    def is_complete(self) -> bool:
        return self.next_label is None

    @property
    def anchor_labels(self) -> tuple[str, ...]:
        return tuple(label for label in CHEST_CALIBRATION_ORDER if label in self._anchors)

    def clear(self) -> None:
        self._anchors.clear()

    def remove_last(self) -> str | None:
        if not self._anchors:
            return None
        label = self.anchor_labels[-1]
        del self._anchors[label]
        return label

    def add_click(
        self,
        x: int,
        y: int,
        person_bbox: tuple[int, int, int, int],
        person_mask: np.ndarray | None = None,
        pose: PersonPose | None = None,
        *,
        mask_tolerance_px: int = 8,
    ) -> str:
        """Add the next chest anchor; reject points clearly outside the person mask."""
        label = self.next_label
        if label is None:
            raise ValueError("all chest anchors are already calibrated; press C to restart")
        x1, y1, x2, y2 = [int(value) for value in person_bbox]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("person_bbox must have positive width and height")
        if not (x1 <= x < x2 and y1 <= y < y2):
            raise ValueError("click must be inside the detected person bounding box")
        if person_mask is not None:
            mask = np.asarray(person_mask)
            if mask.ndim != 2:
                raise ValueError("person_mask must be a 2D image")
            height, width = mask.shape
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError("click is outside the image")
            radius = max(0, int(mask_tolerance_px))
            if radius:
                kernel_size = radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                allowed = cv2.dilate((mask > 0).astype(np.uint8), kernel)
            else:
                allowed = mask > 0
            if not allowed[y, x]:
                raise ValueError("click is outside the detected body silhouette")

        quad = _torso_pose_quad(pose)
        if quad is None:
            raise ValueError("shoulders and hips are not clear; face the camera and keep them visible")
        unit_quad = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        to_local = cv2.getPerspectiveTransform(quad, unit_quad)
        local = _transform_point((float(x), float(y)), to_local)
        if local is None:
            raise ValueError("could not map click into the torso frame; hold still and try again")
        u, v = local
        if not -0.08 <= u <= 1.08 or not -0.08 <= v <= 1.12:
            raise ValueError("click must be on the torso, between the shoulder and hip landmarks")
        self._anchors[label] = (u, v)
        return label

    def positions(self, pose: PersonPose | None) -> dict[str, ElectrodePoint]:
        """Project saved chest anchors into the current shoulder-to-hip torso frame."""
        quad = _torso_pose_quad(pose)
        if quad is None:
            return {}
        unit_quad = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        to_frame = cv2.getPerspectiveTransform(unit_quad, quad)
        positions: dict[str, ElectrodePoint] = {}
        for label, (u, v) in self._anchors.items():
            mapped = _transform_point((u, v), to_frame)
            if mapped is None:
                continue
            positions[label] = ElectrodePoint(
                label, int(round(mapped[0])), int(round(mapped[1])), "manual-torso-pose-relative"
            )
        if "V2" in positions and "V4" in positions:
            v2, v4 = positions["V2"], positions["V4"]
            positions["V3"] = ElectrodePoint(
                "V3",
                int(round((v2.x + v4.x) / 2.0)),
                int(round((v2.y + v4.y) / 2.0)),
                "midpoint-V2-V4",
            )
        return positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "anchors_body_relative": {
                label: [round(u, 7), round(v, 7)] for label, (u, v) in self._anchors.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChestLeadCalibration":
        if data.get("version") != cls.VERSION:
            raise ValueError("unsupported chest calibration version")
        anchors = data.get("anchors_body_relative")
        if not isinstance(anchors, dict):
            raise ValueError("calibration file is missing anchors_body_relative")
        calibration = cls()
        for label in CHEST_CALIBRATION_ORDER:
            if label not in anchors:
                break
            value = anchors[label]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("invalid normalized anchor for {}".format(label))
            u, v = float(value[0]), float(value[1])
            if (
                not np.isfinite(u) or not np.isfinite(v)
                or not -0.08 <= u <= 1.08 or not -0.08 <= v <= 1.12
            ):
                raise ValueError("anchor coordinates are outside the supported torso margins")
            calibration._anchors[label] = (u, v)
        return calibration

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChestLeadCalibration":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls.from_dict(data)


TORSO_CALIBRATION_ORDER = ("RA", "LA", "RL", "LL")
TORSO_POSE_KEYPOINTS = {"right_shoulder": 6, "left_shoulder": 5, "right_hip": 12, "left_hip": 11}


def _torso_pose_quad(pose: PersonPose | None, min_confidence: float = 0.20) -> np.ndarray | None:
    """Return a perspective-ordered torso quad: right shoulder, left shoulder, left hip, right hip."""
    if pose is None:
        return None
    points = np.asarray(pose.keypoints_xy, dtype=np.float32)
    scores = np.asarray(pose.keypoint_scores, dtype=np.float32).reshape(-1)
    indices = [
        TORSO_POSE_KEYPOINTS["right_shoulder"],
        TORSO_POSE_KEYPOINTS["left_shoulder"],
        TORSO_POSE_KEYPOINTS["left_hip"],
        TORSO_POSE_KEYPOINTS["right_hip"],
    ]
    if points.ndim != 2 or points.shape[1] != 2 or max(indices) >= len(points) or max(indices) >= len(scores):
        return None
    quad = points[indices]
    if not np.isfinite(quad).all() or not np.isfinite(scores[indices]).all():
        return None
    if np.any(scores[indices] < min_confidence):
        return None
    if abs(float(cv2.contourArea(quad.reshape(-1, 1, 2)))) < 250.0:
        return None
    # Reject self-crossing or degenerate quads before asking OpenCV for a homography.
    if not cv2.isContourConvex(np.rint(quad).astype(np.int32).reshape(-1, 1, 2)):
        return None
    return quad.astype(np.float32)


def _transform_point(point: tuple[float, float], matrix: np.ndarray) -> tuple[float, float] | None:
    vector = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if not np.isfinite(vector).all() or abs(float(vector[2])) < 1e-8:
        return None
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


class TorsoLeadCalibration:
    """Store selected marker positions in a shoulder-to-hip torso coordinate frame."""

    VERSION = 2

    def __init__(self) -> None:
        self._anchors: dict[str, tuple[float, float]] = {}

    @property
    def next_label(self) -> str | None:
        for label in TORSO_CALIBRATION_ORDER:
            if label not in self._anchors:
                return label
        return None

    @property
    def is_complete(self) -> bool:
        return self.next_label is None

    @property
    def anchor_labels(self) -> tuple[str, ...]:
        return tuple(label for label in TORSO_CALIBRATION_ORDER if label in self._anchors)

    def clear(self) -> None:
        self._anchors.clear()

    def remove_last(self) -> str | None:
        if not self._anchors:
            return None
        label = self.anchor_labels[-1]
        del self._anchors[label]
        return label

    def add_click(
        self,
        x: int,
        y: int,
        person_bbox: tuple[int, int, int, int],
        person_mask: np.ndarray | None = None,
        pose: PersonPose | None = None,
        *,
        mask_tolerance_px: int = 8,
    ) -> str:
        """Record a click relative to shoulder/hip landmarks, rejecting invalid geometry."""
        label = self.next_label
        if label is None:
            raise ValueError("all four torso markers are already calibrated; press G to restart")
        x1, y1, x2, y2 = [int(value) for value in person_bbox]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("person_bbox must have positive width and height")
        if not (x1 <= x < x2 and y1 <= y < y2):
            raise ValueError("click must be inside the detected person bounding box")
        if person_mask is not None:
            mask = np.asarray(person_mask)
            if mask.ndim != 2:
                raise ValueError("person_mask must be a 2D image")
            height, width = mask.shape
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError("click is outside the image")
            radius = max(0, int(mask_tolerance_px))
            if radius:
                kernel_size = radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                allowed = cv2.dilate((mask > 0).astype(np.uint8), kernel)
            else:
                allowed = mask > 0
            if not allowed[y, x]:
                raise ValueError("click is outside the detected body silhouette")

        quad = _torso_pose_quad(pose)
        if quad is None:
            raise ValueError("shoulders and hips are not clear; face the camera and keep them visible")
        unit_quad = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        to_local = cv2.getPerspectiveTransform(quad, unit_quad)
        local = _transform_point((float(x), float(y)), to_local)
        if local is None:
            raise ValueError("could not map click into the torso frame; hold still and try again")
        u, v = local
        if not -0.08 <= u <= 1.08 or not -0.08 <= v <= 1.12:
            raise ValueError("click must be on the torso, between the shoulder and hip landmarks")
        self._anchors[label] = (u, v)
        return label

    def positions(self, pose: PersonPose | None) -> dict[str, ElectrodePoint]:
        """Project clicked anchors into the current shoulder-to-hip torso frame."""
        quad = _torso_pose_quad(pose)
        if quad is None:
            return {}
        unit_quad = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        to_frame = cv2.getPerspectiveTransform(unit_quad, quad)
        positions: dict[str, ElectrodePoint] = {}
        for label in self.anchor_labels:
            local = _transform_point(self._anchors[label], to_frame)
            if local is None:
                continue
            positions[label] = ElectrodePoint(
                label,
                int(round(local[0])),
                int(round(local[1])),
                "manual-torso-pose-relative",
            )
        return positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "anchors_body_relative": {
                label: [round(u, 7), round(v, 7)] for label, (u, v) in self._anchors.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TorsoLeadCalibration":
        if data.get("version") != cls.VERSION:
            raise ValueError("unsupported torso calibration version")
        anchors = data.get("anchors_body_relative")
        if not isinstance(anchors, dict):
            raise ValueError("calibration file is missing anchors_body_relative")
        calibration = cls()
        for label in TORSO_CALIBRATION_ORDER:
            if label not in anchors:
                break
            value = anchors[label]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("invalid normalized anchor for {}".format(label))
            u, v = float(value[0]), float(value[1])
            if (
                not np.isfinite(u) or not np.isfinite(v)
                or not -0.08 <= u <= 1.08 or not -0.08 <= v <= 1.12
            ):
                raise ValueError("anchor coordinates are outside the supported torso margins")
            calibration._anchors[label] = (u, v)
        return calibration

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TorsoLeadCalibration":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls.from_dict(data)

