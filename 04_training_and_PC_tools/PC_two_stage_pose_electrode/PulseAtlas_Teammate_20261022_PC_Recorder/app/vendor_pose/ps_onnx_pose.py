"""OpenCV-DNN adapter for YOLO11n-pose on the PYNQ-Z2 PS CPU.

The bundled ONNX graph stops at the three raw YOLO heads so OpenCV 4.5.4
does not need to import the model's dynamic shape-based decode tail. Boxes,
keypoints, and confidence values are decoded here in NumPy using the same
fixed 256x192 anchor grid and unchanged FP32 head weights from the source checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from electrode_placement import ElectrodePoint, PersonPose, _torso_pose_quad


class OpenCVDNNPersonPoseEstimator:
    """Run one FP32 pose model and return the most prominent full torso pose."""

    INPUT_WIDTH = 256
    INPUT_HEIGHT = 192
    DFL_BINS = 16
    KEYPOINT_COUNT = 17
    BOX_OUTPUT = "/model.23/Concat_output_0"
    SCORE_OUTPUT = "/model.23/Concat_1_output_0"
    KEYPOINT_OUTPUT = "/model.23/Concat_2_output_0"
    REQUIRED_OUTPUTS = (BOX_OUTPUT, SCORE_OUTPUT, KEYPOINT_OUTPUT)

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = 0.25,
        keypoint_confidence: float = 0.20,
        nms_iou: float = 0.45,
        max_detections: int = 8,
        net: Any | None = None,
        single_cpu_thread: bool = True,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= keypoint_confidence <= 1.0:
            raise ValueError("keypoint_confidence must be between 0 and 1")
        if not 0.0 < nms_iou < 1.0:
            raise ValueError("nms_iou must be between 0 and 1")
        if max_detections < 1:
            raise ValueError("max_detections must be positive")

        self.weights = str(weights)
        self.confidence = float(confidence)
        self.keypoint_confidence = float(keypoint_confidence)
        self.nms_iou = float(nms_iou)
        self.max_detections = int(max_detections)
        if single_cpu_thread:
            cv2.setNumThreads(1)
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
            cv2.ocl.setUseOpenCL(False)
        self.net = net if net is not None else cv2.dnn.readNetFromONNX(self.weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        missing = set(self.REQUIRED_OUTPUTS).difference(self.output_names)
        if missing:
            raise RuntimeError("unexpected YOLO pose raw outputs; missing {}".format(
                ", ".join(sorted(missing))))
        self._anchor_x, self._anchor_y, self._stride = self._make_anchor_grid()

    def detect(self, frame: np.ndarray) -> PersonPose | None:
        """Return the highest-prominence person with visible shoulders and hips."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 BGR image")
        height, width = frame.shape[:2]
        if width < 2 or height < 2:
            raise ValueError("frame must be at least 2x2 pixels")

        image, gain, pad_left, pad_top = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0,
            size=(self.INPUT_WIDTH, self.INPUT_HEIGHT),
            mean=(0.0, 0.0, 0.0), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        if len(outputs) != len(self.output_names):
            raise RuntimeError("unexpected YOLO pose output count: {}".format(len(outputs)))
        by_name = dict(zip(self.output_names, outputs))
        boxes, scores, keypoints, keypoint_scores = self._decode(
            by_name[self.BOX_OUTPUT], by_name[self.SCORE_OUTPUT],
            by_name[self.KEYPOINT_OUTPUT], self._anchor_x, self._anchor_y, self._stride,
        )

        candidates = np.flatnonzero(scores >= self.confidence)
        if candidates.size == 0:
            return None
        candidates = self._nms(boxes, scores, candidates, self.nms_iou)[:self.max_detections]
        frame_area = float(width * height)
        best = None
        best_rank = (-1.0, -1.0)
        torso_indices = (5, 6, 11, 12)
        for index in candidates:
            model_box = boxes[index]
            left = (float(model_box[0]) - pad_left) / gain
            top = (float(model_box[1]) - pad_top) / gain
            right = (float(model_box[2]) - pad_left) / gain
            bottom = (float(model_box[3]) - pad_top) / gain
            x1 = int(np.clip(np.floor(left), 0, width - 1))
            y1 = int(np.clip(np.floor(top), 0, height - 1))
            x2 = int(np.clip(np.ceil(right), x1 + 1, width))
            y2 = int(np.clip(np.ceil(bottom), y1 + 1, height))
            box_width, box_height = x2 - x1, y2 - y1
            if box_width < 30 or box_height < 80:
                continue

            points = keypoints[index].copy()
            points[:, 0] = np.clip((points[:, 0] - pad_left) / gain, 0, width - 1)
            points[:, 1] = np.clip((points[:, 1] - pad_top) / gain, 0, height - 1)
            kp_scores = keypoint_scores[index].copy()
            pose = PersonPose(
                bbox=(x1, y1, x2, y2),
                confidence=float(scores[index]),
                keypoints_xy=points.astype(np.float32, copy=False),
                keypoint_scores=kp_scores.astype(np.float32, copy=False),
            )
            if np.any(kp_scores[list(torso_indices)] < self.keypoint_confidence):
                continue
            if _torso_pose_quad(pose, min_confidence=self.keypoint_confidence) is None:
                continue

            area = float(box_width * box_height)
            prominence = float(scores[index]) * np.sqrt(area / frame_area)
            rank = (prominence, float(scores[index]))
            if rank > best_rank:
                best, best_rank = pose, rank
        return best

    @classmethod
    def _decode(
        cls,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
        raw_keypoints: np.ndarray,
        anchor_x: np.ndarray,
        anchor_y: np.ndarray,
        stride: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode the YOLO11 raw DFL, person-score, and COCO17 pose heads."""
        raw_boxes = np.asarray(raw_boxes, dtype=np.float32)
        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        raw_keypoints = np.asarray(raw_keypoints, dtype=np.float32)
        if raw_boxes.ndim == 3 and raw_boxes.shape[0] == 1:
            raw_boxes = raw_boxes[0]
        if raw_scores.ndim == 3 and raw_scores.shape[0] == 1:
            raw_scores = raw_scores[0]
        if raw_keypoints.ndim == 3 and raw_keypoints.shape[0] == 1:
            raw_keypoints = raw_keypoints[0]
        candidate_count = anchor_x.size
        if raw_boxes.shape != (4 * cls.DFL_BINS, candidate_count):
            raise RuntimeError("unexpected YOLO pose box tensor: {}".format(raw_boxes.shape))
        if raw_scores.shape != (1, candidate_count):
            raise RuntimeError("unexpected YOLO pose score tensor: {}".format(raw_scores.shape))
        if raw_keypoints.shape != (cls.KEYPOINT_COUNT * 3, candidate_count):
            raise RuntimeError("unexpected YOLO pose keypoint tensor: {}".format(raw_keypoints.shape))
        if not (np.isfinite(raw_boxes).all() and np.isfinite(raw_scores).all()
                and np.isfinite(raw_keypoints).all()):
            raise RuntimeError("YOLO pose model returned non-finite outputs")

        logits = raw_boxes.reshape(4, cls.DFL_BINS, candidate_count)
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        bins = np.arange(cls.DFL_BINS, dtype=np.float32)[None, :, None]
        distances = np.sum(probabilities * bins, axis=1)
        boxes = np.empty((candidate_count, 4), dtype=np.float32)
        boxes[:, 0] = (anchor_x - distances[0]) * stride
        boxes[:, 1] = (anchor_y - distances[1]) * stride
        boxes[:, 2] = (anchor_x + distances[2]) * stride
        boxes[:, 3] = (anchor_y + distances[3]) * stride

        scores = cls._sigmoid(raw_scores[0])
        points = raw_keypoints.reshape(cls.KEYPOINT_COUNT, 3, candidate_count)
        keypoints = np.empty((candidate_count, cls.KEYPOINT_COUNT, 2), dtype=np.float32)
        # Ultralytics uses grid-corner coordinates for pose points but
        # half-cell centers for DFL box decoding. Keep the two offsets distinct.
        pose_anchor_x = anchor_x - 0.5
        pose_anchor_y = anchor_y - 0.5
        keypoints[:, :, 0] = (2.0 * points[:, 0, :].T + pose_anchor_x[:, None]) * stride[:, None]
        keypoints[:, :, 1] = (2.0 * points[:, 1, :].T + pose_anchor_y[:, None]) * stride[:, None]
        keypoint_scores = cls._sigmoid(points[:, 2, :].T)
        return boxes, scores, keypoints, keypoint_scores

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, indices: np.ndarray,
             threshold: float) -> list[int]:
        order = indices[np.argsort(scores[indices], kind="stable")[::-1]]
        kept: list[int] = []
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            left = np.maximum(boxes[current, 0], boxes[rest, 0])
            top = np.maximum(boxes[current, 1], boxes[rest, 1])
            right = np.minimum(boxes[current, 2], boxes[rest, 2])
            bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
            area_current = max(0.0, float(boxes[current, 2] - boxes[current, 0])) * max(
                0.0, float(boxes[current, 3] - boxes[current, 1]))
            area_rest = np.maximum(0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0, boxes[rest, 3] - boxes[rest, 1])
            union = area_current + area_rest - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = rest[iou <= threshold]
        return kept

    @classmethod
    def _make_anchor_grid(cls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_x, grid_y, strides = [], [], []
        for step in (8.0, 16.0, 32.0):
            grid_width = cls.INPUT_WIDTH // int(step)
            grid_height = cls.INPUT_HEIGHT // int(step)
            grid_x.append(np.tile(np.arange(grid_width, dtype=np.float32), grid_height) + 0.5)
            grid_y.append(np.repeat(np.arange(grid_height, dtype=np.float32), grid_width) + 0.5)
            strides.append(np.full(grid_width * grid_height, step, dtype=np.float32))
        return np.concatenate(grid_x), np.concatenate(grid_y), np.concatenate(strides)

    @classmethod
    def _letterbox(cls, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        gain = min(cls.INPUT_WIDTH / width, cls.INPUT_HEIGHT / height)
        new_width = min(cls.INPUT_WIDTH, max(1, int(round(width * gain))))
        new_height = min(cls.INPUT_HEIGHT, max(1, int(round(height * gain))))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_width, pad_height = cls.INPUT_WIDTH - new_width, cls.INPUT_HEIGHT - new_height
        left = int(round(pad_width / 2.0 - 0.1))
        right = int(round(pad_width / 2.0 + 0.1))
        top = int(round(pad_height / 2.0 - 0.1))
        bottom = int(round(pad_height / 2.0 + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, gain, left, top


def draw_pose_overlay(
    frame: np.ndarray,
    pose: PersonPose | None,
    torso_points: dict[str, ElectrodePoint],
    chest_points: dict[str, ElectrodePoint],
    *,
    mirror: bool = True,
) -> np.ndarray:
    """Draw pose and calibrated sites in source coordinates, then mirror for selfie view."""
    output = frame.copy()
    height, width = output.shape[:2]
    if pose is not None:
        x1, y1, x2, y2 = pose.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (150, 177, 210), 1, cv2.LINE_AA)
        quad = _torso_pose_quad(pose)
        if quad is not None:
            cv2.polylines(output, [np.rint(quad).astype(np.int32)], True,
                          (0, 218, 255), 2, cv2.LINE_AA)
        keypoints = np.asarray(pose.keypoints_xy)
        scores = np.asarray(pose.keypoint_scores).reshape(-1)
        for left, right in ((0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
                            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)):
            if max(left, right) >= len(scores) or min(scores[left], scores[right]) < 0.20:
                continue
            a = tuple(np.rint(keypoints[left]).astype(int))
            b = tuple(np.rint(keypoints[right]).astype(int))
            cv2.line(output, a, b, (220, 180, 82), 2, cv2.LINE_AA)
        for point, score in zip(keypoints, scores):
            if score < 0.25 or not np.isfinite(point).all():
                continue
            center = tuple(np.rint(point).astype(int))
            cv2.circle(output, center, 3, (246, 250, 253), -1, cv2.LINE_AA)
            cv2.circle(output, center, 2, (66, 112, 171), -1, cv2.LINE_AA)

    if mirror:
        output = cv2.flip(output, 1)
    for point in torso_points.values():
        display_point = ElectrodePoint(
            point.label, width - 1 - point.x if mirror else point.x,
            point.y, point.method, point.confidence,
        )
        _draw_lead_point(output, display_point, (45, 205, 85), width, height)
    for point in chest_points.values():
        display_point = ElectrodePoint(
            point.label, width - 1 - point.x if mirror else point.x,
            point.y, point.method, point.confidence,
        )
        color = (40, 145, 255) if point.label == "V3" else (60, 70, 225)
        _draw_lead_point(output, display_point, color, width, height,
                         hollow=(point.label == "V3"))
    return output


def _draw_lead_point(
    image: np.ndarray,
    point: ElectrodePoint,
    color: tuple[int, int, int],
    width: int,
    height: int,
    *,
    hollow: bool = False,
) -> None:
    x, y = int(point.x), int(point.y)
    if not (0 <= x < width and 0 <= y < height):
        return
    radius = max(4, min(height, width) // 80)
    cv2.circle(image, (x, y), radius + 2, (250, 252, 255), 1, cv2.LINE_AA)
    cv2.circle(image, (x, y), radius, color, 2 if hollow else -1, cv2.LINE_AA)
    cv2.putText(image, point.label, (min(width - 24, x + radius + 3),
                max(14, y - radius - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA)


__all__ = ["OpenCVDNNPersonPoseEstimator", "draw_pose_overlay"]
