"""Offline checks for vision cadence, calibration loading, and camera cleanup."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps_vision_runtime import PSVisionRuntime  # noqa: E402


class _FakeCamera:
    def __init__(self):
        self.released = False

    @staticmethod
    def isOpened():
        return True

    @staticmethod
    def set(_property, _value):
        return True

    @staticmethod
    def read():
        time.sleep(1.0 / 30.0)
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakePointStabilizer:
    def __init__(self, **_kwargs):
        pass

    @staticmethod
    def update(points, _bbox):
        return points


class _FakeCalibration:
    def __init__(self, labels):
        self.anchor_labels = tuple(labels)
        self.labels = tuple(labels)

    @classmethod
    def load(cls, path):
        if "torso" in str(path):
            return cls(("RA", "LA", "RL", "LL"))
        return cls(("V1", "V2", "V3", "V4", "V5", "V6"))

    def positions(self, _pose):
        return {
            label: types.SimpleNamespace(label=label, x=100 + index, y=120 + index,
                                         method="manual", confidence=None)
            for index, label in enumerate(self.labels)
        }


class _FakePose:
    bbox = (80, 40, 320, 440)
    confidence = 0.9


class _FakePoseEstimator:
    init_kwargs = None

    def __init__(self, *_args, **_kwargs):
        type(self).init_kwargs = _kwargs

    @staticmethod
    def detect(_frame):
        return _FakePose()


class VisionRuntimeTests(unittest.TestCase):
    def test_default_inference_cap_does_not_limit_board_model_rate(self):
        runtime = PSVisionRuntime(model_path="unused.onnx", pin_worker_cpu=False)
        self.assertAlmostEqual(runtime.minimum_update_interval, 1.0 / 60.0)
        self.assertAlmostEqual(
            PSVisionRuntime(model_path="unused.onnx", max_inference_fps=4,
                            pin_worker_cpu=False).minimum_update_interval,
            0.25,
        )
        with self.assertRaises(ValueError):
            PSVisionRuntime(model_path="unused.onnx", max_inference_fps=0)

    def test_worker_affinity_uses_secondary_cpu_and_tolerates_os_errors(self):
        runtime = PSVisionRuntime(model_path="unused.onnx", pin_worker_cpu=True)
        original_get = getattr(os, "sched_getaffinity", None)
        original_set = getattr(os, "sched_setaffinity", None)
        calls = []
        try:
            os.sched_getaffinity = lambda _pid: {0, 1}
            os.sched_setaffinity = lambda pid, cpus: calls.append((pid, cpus))
            runtime._pin_worker_to_secondary_cpu()
            self.assertEqual(calls, [(0, {1})])
            self.assertEqual(runtime.snapshot()["vision_cpu_affinity"], 1)

            os.sched_getaffinity = lambda _pid: {0}
            runtime._publish(vision_cpu_affinity=None)
            runtime._pin_worker_to_secondary_cpu()
            self.assertIsNone(runtime.snapshot()["vision_cpu_affinity"])

            os.sched_getaffinity = lambda _pid: {0, 1}

            def denied(_pid, _cpus):
                raise OSError("affinity unavailable")

            os.sched_setaffinity = denied
            runtime._publish(vision_cpu_affinity=None)
            runtime._pin_worker_to_secondary_cpu()
            self.assertIsNone(runtime.snapshot()["vision_cpu_affinity"])
        finally:
            if original_get is None:
                del os.sched_getaffinity
            else:
                os.sched_getaffinity = original_get
            if original_set is None:
                del os.sched_setaffinity
            else:
                os.sched_setaffinity = original_set

    def test_worker_rate_limits_camera_updates_and_releases_camera(self):
        camera = _FakeCamera()
        cv2 = types.ModuleType("cv2")
        cv2.CAP_V4L2 = 1
        cv2.CAP_PROP_FRAME_WIDTH = 3
        cv2.CAP_PROP_FRAME_HEIGHT = 4
        cv2.CAP_PROP_BUFFERSIZE = 5
        thread_counts = []
        cv2.setNumThreads = lambda count: thread_counts.append(count)
        cv2.VideoCapture = lambda *_args: camera

        pose_module = types.ModuleType("ps_onnx_pose")
        pose_module.OpenCVDNNPersonPoseEstimator = _FakePoseEstimator
        pose_module.draw_pose_overlay = lambda frame, *_args, **_kwargs: frame
        electrode_module = types.ModuleType("electrode_placement")
        electrode_module.ChestLeadCalibration = _FakeCalibration
        electrode_module.TorsoLeadCalibration = _FakeCalibration
        electrode_module.TemporalPointStabilizer = _FakePointStabilizer
        electrode_module._torso_pose_quad = lambda _pose: object()

        names = ("cv2", "ps_onnx_pose", "electrode_placement")
        previous_modules = {name: sys.modules.get(name) for name in names}
        sys.modules.update({"cv2": cv2, "ps_onnx_pose": pose_module,
                            "electrode_placement": electrode_module})
        runtime = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                vision = temp_root / "vision_guide"
                model = vision / "models" / "model.onnx"
                model.parent.mkdir(parents=True)
                model.write_bytes(b"test")
                calibration_dir = vision / "calibration"
                calibration_dir.mkdir()
                torso_file = calibration_dir / "torso.json"
                chest_file = calibration_dir / "chest.json"
                torso_file.write_text("{}", encoding="utf-8")
                chest_file.write_text("{}", encoding="utf-8")
                runtime = PSVisionRuntime(
                    model_path=model,
                    torso_calibration_path=torso_file,
                    chest_calibration_path=chest_file,
                    max_inference_fps=2.0,
                    pin_worker_cpu=False,
                )
                started = time.monotonic()
                runtime.start()
                deadline = started + 4.0
                while runtime.snapshot()["vision_frames"] < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertGreaterEqual(runtime.snapshot()["vision_frames"], 3)
                self.assertGreaterEqual(time.monotonic() - started, 0.9)
                self.assertGreaterEqual(runtime.snapshot()["vision_camera_frames"], 5)
                self.assertGreater(runtime.snapshot()["vision_preview_frames"], 0)
                self.assertGreater(runtime.snapshot()["vision_camera_fps"], 0.0)
                self.assertEqual(runtime.snapshot()["vision_inference_count"],
                                 runtime.snapshot()["vision_frames"])
                self.assertEqual(runtime.snapshot()["vision_marker_count"], 10)
                self.assertTrue(runtime.snapshot()["vision_pose_valid"])
                self.assertEqual(
                    runtime.snapshot()["vision_result_status"],
                    "人体定位 · 10 个电极标记",
                )
                self.assertEqual(thread_counts, [2])
                self.assertEqual(runtime.snapshot()["vision_cpu_threads"], 2)
                self.assertFalse(_FakePoseEstimator.init_kwargs["single_cpu_thread"])
                self.assertTrue(runtime.stop(timeout=1.0))
                self.assertTrue(camera.released)
        finally:
            if runtime is not None:
                runtime.stop(timeout=1.0)
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    def test_worker_priority_is_lowered_without_raising_existing_nice_value(self):
        runtime = PSVisionRuntime(model_path="unused.onnx", worker_nice=10)
        names = ("setpriority", "getpriority", "PRIO_PROCESS")
        previous = {name: getattr(os, name, None) for name in names}
        calls = []
        original_native_id = threading.get_native_id
        try:
            os.PRIO_PROCESS = 0
            os.getpriority = lambda _kind, _tid: 0
            os.setpriority = lambda kind, tid, nice: calls.append((kind, tid, nice))
            threading.get_native_id = lambda: 4321
            runtime._lower_worker_priority()
            self.assertEqual(calls, [(0, 4321, 10)])
            self.assertEqual(runtime.snapshot()["vision_nice"], 10)

            calls.clear()
            os.getpriority = lambda _kind, _tid: 12
            runtime._lower_worker_priority()
            self.assertEqual(calls, [])
            self.assertEqual(runtime.snapshot()["vision_nice"], 12)
        finally:
            threading.get_native_id = original_native_id
            for name, value in previous.items():
                if value is None:
                    if hasattr(os, name):
                        delattr(os, name)
                else:
                    setattr(os, name, value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
