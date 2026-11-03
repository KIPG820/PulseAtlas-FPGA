"""PS-only webcam inference worker; it never touches the ECG overlay or DMA."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path


class PSVisionRuntime:
    """Own the USB camera and apply saved body-relative ECG lead markers."""

    def __init__(self, model_path=None, camera_index=0, max_inference_fps=60.0,
                 pin_worker_cpu=False, worker_nice=10, torso_calibration_path=None,
                 chest_calibration_path=None):
        max_inference_fps = float(max_inference_fps)
        if not math.isfinite(max_inference_fps) or not 0.0 < max_inference_fps <= 60.0:
            raise ValueError("max_inference_fps must be in (0, 60]")
        package_root = Path(__file__).resolve().parent
        bundled_vision = package_root / "vision_guide"
        source_vision = package_root.parent / "vision_guide"
        vision_dir = bundled_vision if bundled_vision.is_dir() else source_vision
        bundled_model = vision_dir / "models" / "yolo11n-pose-cv454.onnx"
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = bundled_model
        self.torso_calibration_path = Path(torso_calibration_path) if torso_calibration_path else (
            vision_dir / "calibration" / "torso_lead_calibration.json")
        self.chest_calibration_path = Path(chest_calibration_path) if chest_calibration_path else (
            vision_dir / "calibration" / "chest_lead_calibration.json")
        self.camera_index = int(camera_index)
        self.max_inference_fps = max_inference_fps
        self._inference_interval = 1.0 / max_inference_fps
        self._pin_worker_cpu = bool(pin_worker_cpu)
        self._worker_nice = max(0, min(19, int(worker_nice)))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._capture_thread = None
        self._latest_camera_frame = None
        self._latest_camera_sequence = 0
        self._latest_pose = None
        self._latest_torso_points = {}
        self._latest_chest_points = {}
        self._latest_camera_captured_at = None
        self._latest_result_frame_at = None
        self._capture_error = None
        self._state = {
            "vision_active": False,
            "vision_camera_connected": False,
            "vision_camera_checked": False,
            "vision_frame_bgr": None,
            "vision_camera_frames": 0,
            "vision_camera_fps": 0.0,
            "vision_preview_frames": 0,
            "vision_preview_fps": 0.0,
            "vision_status": "按 BTN2 启动视觉识别",
            "vision_error": None,
            "vision_fps": 0.0,
            "vision_inference_ms": None,
            "vision_inference_running": False,
            "vision_inference_count": 0,
            "vision_result_age_ms": None,
            "vision_result_status": "等待推理",
            "vision_capture_running": False,
            "vision_confidence": None,
            "vision_stale": False,
            "vision_detected": False,
            "vision_frames": 0,
            "vision_cpu_affinity": None,
            "vision_cpu_threads": 0,
            "vision_nice": None,
            "vision_marker_count": 0,
            "vision_pose_valid": False,
        }

    @property
    def minimum_update_interval(self):
        return self._inference_interval

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop = threading.Event()
            self._state.update(
                vision_active=True,
                vision_camera_connected=False,
                vision_camera_checked=False,
                vision_frame_bgr=None,
                vision_camera_frames=0,
                vision_camera_fps=0.0,
                vision_preview_frames=0,
                vision_preview_fps=0.0,
                vision_status="正在加载 PS CPU 模型…",
                vision_error=None,
                vision_fps=0.0,
                vision_inference_ms=None,
                vision_inference_running=False,
                vision_inference_count=0,
                vision_result_age_ms=None,
                vision_result_status="等待推理",
                vision_capture_running=False,
                vision_confidence=None,
                vision_stale=False,
                vision_detected=False,
                vision_frames=0,
                vision_cpu_affinity=None,
                vision_cpu_threads=0,
                vision_nice=None,
                vision_marker_count=0,
                vision_pose_valid=False,
            )
            self._capture_thread = None
            self._latest_camera_frame = None
            self._latest_camera_sequence = 0
            self._latest_pose = None
            self._latest_torso_points = {}
            self._latest_chest_points = {}
            self._latest_camera_captured_at = None
            self._latest_result_frame_at = None
            self._capture_error = None
            self._thread = threading.Thread(
                target=self._run, name="pulseatlas-ps-vision", daemon=True
            )
            self._thread.start()
            return True

    def stop(self, timeout=2.0):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        alive = thread is not None and thread.is_alive()
        with self._lock:
            self._state["vision_active"] = bool(alive)
            if not alive:
                self._state["vision_camera_connected"] = False
                self._state["vision_camera_checked"] = False
                self._state["vision_status"] = "视觉识别已停止"
        return not alive

    def toggle(self):
        if self.snapshot()["vision_active"]:
            self.stop()
            return False
        self.start()
        return True

    def _publish(self, **values):
        with self._lock:
            self._state.update(values)

    def _pin_worker_to_secondary_cpu(self):
        """Keep expensive vision inference off the app's primary CPU where possible."""
        if not self._pin_worker_cpu:
            return
        get_affinity = getattr(os, "sched_getaffinity", None)
        set_affinity = getattr(os, "sched_setaffinity", None)
        if get_affinity is None or set_affinity is None:
            return
        try:
            available = sorted(get_affinity(0))
            if len(available) < 2:
                return
            secondary_cpu = available[-1]
            set_affinity(0, {secondary_cpu})
        except (OSError, PermissionError):
            return
        self._publish(vision_cpu_affinity=secondary_cpu)
        print("PulseAtlas PS vision pinned to CPU {}".format(secondary_cpu), flush=True)

    def _lower_worker_priority(self):
        """Let ECG/UI work preempt the two-thread vision inference worker."""
        set_priority = getattr(os, "setpriority", None)
        get_priority = getattr(os, "getpriority", None)
        get_native_id = getattr(threading, "get_native_id", None)
        priority_kind = getattr(os, "PRIO_PROCESS", None)
        if set_priority is None or get_priority is None or get_native_id is None or priority_kind is None:
            return
        thread_id = get_native_id()
        try:
            current = int(get_priority(priority_kind, thread_id))
            target = max(current, self._worker_nice)
            if target != current:
                set_priority(priority_kind, thread_id, target)
            self._publish(vision_nice=target)
            print("PulseAtlas PS vision nice={}".format(target), flush=True)
        except (OSError, PermissionError):
            return

    @staticmethod
    def _available_cpu_count():
        get_affinity = getattr(os, "sched_getaffinity", None)
        if get_affinity is not None:
            try:
                return max(1, len(get_affinity(0)))
            except OSError:
                pass
        return max(1, int(os.cpu_count() or 1))

    def _capture_camera(self, camera, draw_pose_overlay):
        """Capture continuously so a slow model cannot freeze the live preview."""
        sequence = 0
        failures = 0
        window_started = time.monotonic()
        window_frames = 0
        camera_fps = 0.0
        last_preview = 0.0
        try:
            while not self._stop.is_set():
                ok, frame = camera.read()
                captured_at = time.monotonic()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 20:
                        self._capture_error = "摄像头连续 20 次未返回画面"
                        self._publish(vision_status="摄像头采集失败",
                                      vision_error=self._capture_error)
                        return
                    self._stop.wait(0.01)
                    continue

                failures = 0
                sequence += 1
                window_frames += 1
                with self._lock:
                    self._latest_camera_frame = frame
                    self._latest_camera_sequence = sequence
                    self._latest_camera_captured_at = captured_at
                    pose = self._latest_pose
                    torso_points = dict(self._latest_torso_points)
                    chest_points = dict(self._latest_chest_points)
                    result_frame_at = self._latest_result_frame_at

                elapsed = captured_at - window_started
                if elapsed >= 0.5:
                    measured = window_frames / max(elapsed, 1e-6)
                    camera_fps = measured if camera_fps == 0.0 else (
                        0.65 * camera_fps + 0.35 * measured
                    )
                    window_started = captured_at
                    window_frames = 0

                # Keep annotation work bounded while supporting a 20 Hz HDMI path;
                # camera capture itself always continues at its native cadence.
                if captured_at - last_preview >= 0.025:
                    annotated = draw_pose_overlay(
                        frame, pose, torso_points, chest_points, mirror=True
                    )
                    age_ms = (None if result_frame_at is None else
                              max(0.0, 1000.0 * (captured_at - result_frame_at)))
                    with self._lock:
                        self._state.update(
                            vision_frame_bgr=annotated,
                            vision_camera_frames=sequence,
                            vision_camera_fps=camera_fps,
                            vision_preview_frames=(
                                int(self._state.get("vision_preview_frames", 0)) + 1
                            ),
                            vision_result_age_ms=age_ms,
                            vision_stale=bool(age_ms is not None and age_ms > 1500.0),
                        )
                    last_preview = captured_at
        except Exception as exc:
            self._capture_error = "摄像头采集失败：{}".format(exc)
            self._publish(vision_status="摄像头采集失败",
                          vision_error=self._capture_error)
        finally:
            self._publish(vision_capture_running=False)

    def _run(self):
        camera = None
        try:
            self._pin_worker_to_secondary_cpu()
            self._lower_worker_priority()
            if not self.model_path.is_file():
                raise FileNotFoundError("视觉 ONNX 模型不存在：{}".format(self.model_path))

            vision_dir = str(self.model_path.resolve().parents[1])
            if vision_dir not in sys.path:
                sys.path.insert(0, vision_dir)
            try:
                import cv2
                from electrode_placement import (
                    ChestLeadCalibration, TemporalPointStabilizer, TorsoLeadCalibration,
                    _torso_pose_quad,
                )
                from ps_onnx_pose import OpenCVDNNPersonPoseEstimator, draw_pose_overlay
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 OpenCV DNN / NumPy 运行依赖：{}".format(exc)
                ) from exc

            vision_threads = min(2, self._available_cpu_count())
            cv2.setNumThreads(vision_threads)
            estimator = OpenCVDNNPersonPoseEstimator(
                self.model_path, confidence=0.25, keypoint_confidence=0.20,
                single_cpu_thread=False,
            )
            self._publish(vision_cpu_threads=vision_threads)
            print("PulseAtlas PS vision OpenCV threads={}".format(vision_threads), flush=True)
            if not self.torso_calibration_path.is_file() or not self.chest_calibration_path.is_file():
                raise FileNotFoundError("缺少手动标定文件：{} / {}".format(
                    self.torso_calibration_path, self.chest_calibration_path))
            torso_calibration = TorsoLeadCalibration.load(self.torso_calibration_path)
            chest_calibration = ChestLeadCalibration.load(self.chest_calibration_path)
            point_stabilizer = TemporalPointStabilizer(median_window=3, smoothing=0.28)
            print("PulseAtlas loaded manual electrode calibration: torso={} chest={}".format(
                len(torso_calibration.anchor_labels), len(chest_calibration.anchor_labels)),
                flush=True)
            if self._stop.is_set():
                return
            self._publish(vision_status="正在连接 USB 摄像头…")
            self._publish(vision_camera_checked=True)

            backend = getattr(cv2, "CAP_V4L2", 0)
            camera = cv2.VideoCapture(self.camera_index, backend)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(self.camera_index)
            if not camera.isOpened():
                raise RuntimeError(
                    "无法打开 /dev/video{}；请检查 USB 摄像头是否被其他程序占用".format(
                        self.camera_index
                    )
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._publish(
                vision_camera_connected=True,
                vision_capture_running=True,
                vision_status="摄像头已连接，准备推理",
            )
            self._capture_thread = threading.Thread(
                target=self._capture_camera,
                args=(camera, draw_pose_overlay),
                name="pulseatlas-camera-capture",
                daemon=True,
            )
            self._capture_thread.start()

            last_inference_started = None
            first_inference_started = None
            inference_count = 0
            last_processed_sequence = 0

            while not self._stop.is_set():
                if self._capture_error:
                    raise RuntimeError(self._capture_error)
                with self._lock:
                    frame = self._latest_camera_frame
                    sequence = self._latest_camera_sequence
                    frame_captured_at = self._latest_camera_captured_at
                if frame is None or sequence == last_processed_sequence:
                    self._stop.wait(0.005)
                    continue

                now = time.monotonic()
                if (last_inference_started is not None and
                        now - last_inference_started < self._inference_interval):
                    self._stop.wait(min(0.01, self._inference_interval -
                                        (now - last_inference_started)))
                    continue

                last_processed_sequence = sequence
                last_inference_started = now
                if first_inference_started is None:
                    first_inference_started = now
                inference_count += 1
                self._publish(
                    vision_inference_running=True,
                    vision_inference_count=inference_count,
                    vision_result_status="模型正在分析第 {} 帧".format(inference_count),
                )
                started = time.perf_counter()
                pose = estimator.detect(frame)
                torso_points = torso_calibration.positions(pose) if pose is not None else {}
                chest_points = chest_calibration.positions(pose) if pose is not None else {}
                guide_points = dict(torso_points)
                guide_points.update(chest_points)
                stable_points = point_stabilizer.update(
                    guide_points, pose.bbox if pose is not None else None
                )
                stable_torso = {label: stable_points[label] for label in torso_points
                                if label in stable_points}
                stable_chest = {label: stable_points[label] for label in chest_points
                                if label in stable_points}
                infer_seconds = max(time.perf_counter() - started, 1e-6)
                completed_at = time.monotonic()
                with self._lock:
                    self._latest_pose = pose
                    self._latest_torso_points = stable_torso
                    self._latest_chest_points = stable_chest
                    # Report the source-frame age, including the time the model
                    # spent processing it, instead of resetting age at completion.
                    self._latest_result_frame_at = frame_captured_at
                marker_count = len(stable_torso) + len(stable_chest)
                if pose is not None and marker_count:
                    status = "人体定位 · {} 个电极标记".format(marker_count)
                elif pose is not None:
                    status = "人体定位 · 尚无有效电极标记"
                else:
                    status = "当前帧未检测到完整躯干"
                self._publish(
                    vision_active=True,
                    vision_camera_connected=True,
                    vision_status="摄像头已连接",
                    vision_result_status=status,
                    vision_error=None,
                    vision_fps=inference_count / max(
                        completed_at - first_inference_started, 1e-6
                    ),
                    vision_inference_ms=1000.0 * infer_seconds,
                    vision_inference_running=False,
                    vision_confidence=(pose.confidence if pose else None),
                    vision_stale=False,
                    vision_detected=bool(pose is not None),
                    vision_frames=inference_count,
                    vision_marker_count=marker_count,
                    vision_pose_valid=bool(pose is not None and _torso_pose_quad(pose) is not None),
                )

        except Exception as exc:
            message = str(exc)
            print("PulseAtlas PS vision stopped: {}".format(message), file=sys.stderr, flush=True)
            self._publish(
                vision_active=False,
                vision_camera_connected=False,
                vision_status="视觉启动失败" if self.snapshot()["vision_frames"] == 0 else "视觉任务已停止",
                vision_error=message,
                vision_inference_running=False,
            )
        finally:
            self._stop.set()
            if camera is not None:
                camera.release()
            capture_thread = self._capture_thread
            if capture_thread is not None and capture_thread is not threading.current_thread():
                capture_thread.join(timeout=1.0)
            with self._lock:
                if self._stop.is_set() and not self._state.get("vision_error"):
                    self._state["vision_active"] = False
                    self._state["vision_camera_connected"] = False
                    self._state["vision_status"] = "视觉识别已停止"


__all__ = ["PSVisionRuntime"]
