"""Separate pose preprocessing, graph, decode and display work on the B board."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pynq import Overlay


ROOT = Path("/home/xilinx/PulseAtlas_B_Vision_R7_20261013")
VISION = ROOT / "vision_guide"
sys.path.insert(0, "/home/xilinx/PulseAtlas_B_Vision_R15_ZeroCopy/vision_guide")
sys.path.insert(1, str(VISION))
from ps_onnx_pose import OpenCVDNNPersonPoseEstimator, draw_pose_overlay
from vision_int8_linecache_net import LineCacheInt8PoseNet


def measure(fn, count=5):
    samples = []
    result = None
    for _ in range(count):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return result, samples


def main():
    cv2.setNumThreads(2)
    overlay = Overlay(
        "/home/xilinx/PulseAtlas_B_Vision_R14_90MHz/vision_accel/vision_bonly_writebase_95mhz.bit",
        download=True)
    net = LineCacheInt8PoseNet(overlay, VISION / "int8_graph",
                               VISION / "libvision_int8_pack.so")
    estimator = OpenCVDNNPersonPoseEstimator(
        VISION / "models/yolo11n-pose-cv454.onnx",
        confidence=0.25, keypoint_confidence=0.20,
        net=net, single_cpu_thread=False)
    with np.load(ROOT / "vision_accel/golden_frame8.npz", allow_pickle=False) as golden:
        fallback = golden["frame"].copy()
    net.forward_frame(fallback)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        for _ in range(4):
            ok, frame = camera.read()
        if not ok or frame is None:
            frame = fallback
            source = "golden_fallback"
        else:
            source = "camera"
    finally:
        camera.release()

    qinput, preprocess = measure(lambda: net.graph.make_input(frame))
    outputs, graph = measure(lambda: net.graph.run(qinput))
    by_name = dict(zip(estimator.output_names, outputs))
    _, decode = measure(lambda: estimator._decode(
        by_name[estimator.BOX_OUTPUT], by_name[estimator.SCORE_OUTPUT],
        by_name[estimator.KEYPOINT_OUTPUT], estimator._anchor_x,
        estimator._anchor_y, estimator._stride))
    pose, detect = measure(lambda: estimator.detect(frame))
    overlay_frame, drawing = measure(lambda: draw_pose_overlay(
        frame, pose, {}, {}, mirror=True))
    _, jpeg = measure(lambda: cv2.imencode(".jpg", overlay_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 70]))
    result = {"source": source, "frame_shape": list(frame.shape),
              "detected": pose is not None,
              "median_ms": {name: statistics.median(values) for name, values in
                            (("preprocess", preprocess), ("graph", graph),
                             ("decode", decode), ("detect_total", detect),
                             ("draw", drawing), ("jpeg", jpeg))},
              "samples_ms": {"preprocess": preprocess, "graph": graph,
                             "decode": decode, "detect_total": detect,
                             "draw": drawing, "jpeg": jpeg}}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    net.backend.release()


if __name__ == "__main__":
    main()
