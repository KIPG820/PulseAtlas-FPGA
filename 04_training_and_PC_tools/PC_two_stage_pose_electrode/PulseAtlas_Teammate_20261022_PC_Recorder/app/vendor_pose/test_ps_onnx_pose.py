"""Board-safe regression tests for the CV454 pose decoder and saved markers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from electrode_placement import (  # noqa: E402
    ChestLeadCalibration, ElectrodePoint, PersonPose, TorsoLeadCalibration,
)
from ps_onnx_pose import OpenCVDNNPersonPoseEstimator, draw_pose_overlay  # noqa: E402


class PoseDecoderTests(unittest.TestCase):
    def test_zero_logits_decode_to_centered_dfl_box_and_grid_corner_keypoints(self):
        anchor_x, anchor_y, stride = OpenCVDNNPersonPoseEstimator._make_anchor_grid()
        self.assertEqual(anchor_x.size, 1008)
        boxes, scores, points, point_scores = OpenCVDNNPersonPoseEstimator._decode(
            np.zeros((1, 64, 1008), dtype=np.float32),
            np.zeros((1, 1, 1008), dtype=np.float32),
            np.zeros((1, 51, 1008), dtype=np.float32),
            anchor_x, anchor_y, stride,
        )
        np.testing.assert_allclose(boxes[0], [-56, -56, 64, 64], atol=1e-6)
        self.assertAlmostEqual(float(scores[0]), 0.5, places=6)
        np.testing.assert_allclose(points[0, 0], [0, 0], atol=1e-6)
        self.assertAlmostEqual(float(point_scores[0, 0]), 0.5, places=6)

    def test_nms_keeps_best_overlapping_box_and_separate_person(self):
        boxes = np.asarray([[0, 0, 100, 200], [2, 3, 102, 203], [200, 0, 300, 200]],
                           dtype=np.float32)
        scores = np.asarray([0.8, 0.7, 0.6], dtype=np.float32)
        kept = OpenCVDNNPersonPoseEstimator._nms(
            boxes, scores, np.asarray([0, 1, 2]), 0.45)
        self.assertEqual(kept, [0, 2])

    def test_manual_user_calibrations_are_packaged_and_v3_is_midpoint(self):
        torso_path = ROOT / "calibration" / "torso_lead_calibration.json"
        chest_path = ROOT / "calibration" / "chest_lead_calibration.json"
        torso_data = json.loads(torso_path.read_text(encoding="utf-8"))
        chest_data = json.loads(chest_path.read_text(encoding="utf-8"))
        self.assertEqual(torso_data["anchors_body_relative"]["RA"], [0.0851258, 0.0783916])
        self.assertEqual(torso_data["anchors_body_relative"]["LA"], [0.9788105, 0.0799232])
        self.assertEqual(chest_data["anchors_body_relative"]["V1"], [0.2433057, 0.4225943])

        torso = TorsoLeadCalibration.load(torso_path)
        chest = ChestLeadCalibration.load(chest_path)
        points = np.zeros((17, 2), dtype=np.float32)
        points[6] = [220, 160]  # right shoulder
        points[5] = [380, 160]  # left shoulder
        points[11] = [335, 360]  # left hip
        points[12] = [265, 360]  # right hip
        pose = PersonPose((180, 140, 420, 390), 0.9, points,
                          np.ones(17, dtype=np.float32))
        torso_points = torso.positions(pose)
        chest_points = chest.positions(pose)
        self.assertEqual(set(torso_points), {"RA", "LA", "RL", "LL"})
        self.assertEqual(set(chest_points), {"V1", "V2", "V3", "V4", "V5", "V6"})
        self.assertEqual(chest_points["V3"].x, round((chest_points["V2"].x + chest_points["V4"].x) / 2))
        self.assertEqual(chest_points["V3"].y, round((chest_points["V2"].y + chest_points["V4"].y) / 2))

    def test_bundled_graph_loads_and_blank_frame_has_no_pose(self):
        import cv2

        model = ROOT / "models" / "yolo11n-pose-cv454.onnx"
        self.assertTrue(model.is_file())
        estimator = OpenCVDNNPersonPoseEstimator(model)
        result = estimator.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNone(result)

    def test_portrait_letterbox_is_fixed_size_and_nonnegative(self):
        import cv2

        frame = np.zeros((1080, 810, 3), dtype=np.uint8)
        boxed, gain, left, top = OpenCVDNNPersonPoseEstimator._letterbox(frame)
        self.assertEqual(boxed.shape, (192, 256, 3))
        self.assertAlmostEqual(gain, 192.0 / 1080.0)
        self.assertEqual((left, top), (56, 0))

    def test_mirrored_preview_keeps_manual_lead_labels_readable(self):
        import cv2

        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        point = ElectrodePoint("RA", 12, 32, "manual")
        result = draw_pose_overlay(frame, None, {"RA": point}, {}, mirror=True)
        self.assertTrue(np.array_equal(result[32, 96 - 1 - 12], [45, 205, 85]))
        self.assertTrue(np.array_equal(result[32, 12], [0, 0, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
