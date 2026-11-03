"""On-board smoke test for the calibrated PS pose guide and optional camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision_guide"
if not VISION.is_dir():
    VISION = ROOT.parent / "vision_guide"
sys.path.insert(0, str(VISION))

from ps_onnx_pose import OpenCVDNNPersonPoseEstimator  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true",
                        help="also run inference on live frames from the USB camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=5,
                        help="number of live frames to test (default: 5)")
    parser.add_argument("--require-torso", "--require-person", dest="require_torso",
                        action="store_true",
                        help="fail unless shoulder and hip landmarks are visible")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.require_torso and not args.camera:
        parser.error("--require-torso requires --camera")
    return args


def _open_camera(cv2, index):
    backend = getattr(cv2, "CAP_V4L2", None)
    camera = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "cannot open USB camera index {}; check /dev/video{} and camera permissions".format(
                index, index
            )
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def _run_camera_test(cv2, estimator, args):
    camera = _open_camera(cv2, args.camera_index)
    latencies = []
    valid_poses = 0
    started_all = time.perf_counter()
    torso_indices = (5, 6, 11, 12)
    try:
        for frame_index in range(args.frames):
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(
                    "USB camera returned no frame at sample {}/{}".format(
                        frame_index + 1, args.frames
                    )
                )
            # Keep inference in sensor coordinates; only the on-screen preview is mirrored.
            started = time.perf_counter()
            pose = estimator.detect(frame)
            latencies.append(time.perf_counter() - started)
            if pose is not None:
                valid_poses += 1
                shoulders_hips = pose.keypoints_xy[list(torso_indices)].round(1).tolist()
                print("frame {}: torso pose confidence {:.3f}, bbox {}, shoulder/hip {}".format(
                    frame_index + 1, pose.confidence, pose.bbox, shoulders_hips))
            else:
                print("frame {}: no complete shoulder/hip pose".format(frame_index + 1))
    finally:
        camera.release()

    total = max(time.perf_counter() - started_all, 1e-6)
    ordered = sorted(latencies)
    p95_latency = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    print("USB camera: opened; captured {} frames".format(len(latencies)))
    print("Complete torso poses: {}/{}".format(valid_poses, len(latencies)))
    print("Mean inference: {:.3f} s; p95: {:.3f} s".format(
        sum(latencies) / len(latencies), p95_latency))
    print("Camera-loop throughput: {:.2f} FPS".format(len(latencies) / total))
    if args.require_torso and valid_poses == 0:
        raise RuntimeError("no complete torso pose; show both shoulders and hips to the camera")


def main():
    args = _parse_args()
    model = VISION / "models" / "yolo11n-pose-cv454.onnx"
    estimator = OpenCVDNNPersonPoseEstimator(model)
    import cv2

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    started = time.perf_counter()
    result = estimator.detect(blank)
    elapsed = time.perf_counter() - started
    if result is not None:
        raise RuntimeError("blank-frame smoke test unexpectedly reported a torso")
    print("PS OpenCV DNN pose model load: OK")
    print("OpenCV: {}; camera: 640x480 BGR; model input: {}x{} FP32; backend: CPU".format(
        cv2.__version__, estimator.INPUT_WIDTH, estimator.INPUT_HEIGHT))
    print("Blank-frame inference: {:.3f} s; no torso detection".format(elapsed))

    for calibration in ("torso_lead_calibration.json", "chest_lead_calibration.json"):
        path = VISION / "calibration" / calibration
        if not path.is_file():
            raise FileNotFoundError("manual calibration file missing: {}".format(path))
        print("Manual calibration available: {}".format(path.name))

    if args.camera:
        _run_camera_test(cv2, estimator, args)
    print("ECG overlay and ECG DMA were not accessed by this test.")


if __name__ == "__main__":
    main()
