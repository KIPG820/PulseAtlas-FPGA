import unittest

import numpy as np

from patch_presence_checker_candidate import CalibratedPatchPresenceChecker
from patch_presence_connector_candidate import presence_to_placement_result


class FixedPositiveNet:
    def setPreferableBackend(self, _value):
        pass

    def setPreferableTarget(self, _value):
        pass

    def setInput(self, _value):
        pass

    def forward(self):
        return np.tile(np.asarray([[3.0, 0.0]], dtype=np.float32), (10, 1))


class PresencePipelineTests(unittest.TestCase):
    def test_pending_then_confirmed_state_maps_to_ui_correct(self):
        checker = CalibratedPatchPresenceChecker("unused.onnx", net=FixedPositiveNet())
        points = {label: (64 + index * 50, 200) for index, label in enumerate(
            ("RA", "LA", "RL", "LL", "V1", "V2", "V3", "V4", "V5", "V6"))}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        first = checker.predict(frame, points)
        second = checker.predict(frame, points)
        pose = {"status": "valid_torso", "frame_id": 1,
                "torso_reference_points": {k: {"x_px": x, "y_px": y}
                                            for k, (x, y) in points.items()},
                "chest_reference_points": {}}
        pending = presence_to_placement_result(first, pose, 1, 640, 480)
        pose["frame_id"] = 2
        confirmed = presence_to_placement_result(second, pose, 2, 640, 480)
        self.assertEqual(first["per_point"]["RA"]["status"], "pad_pending")
        self.assertEqual(pending["unknown_count"], 10)
        self.assertEqual(confirmed["success_count"], 10)


if __name__ == "__main__":
    unittest.main()
