"""A-board client for the future B-board camera/inference service.

Wire protocol: each packet is a 4-byte big-endian JSON-header length, the
UTF-8 JSON header, then ``payload_len`` raw bytes. Preview payloads are JPEG.
See VISION_B_SERVICE_PROTOCOL.md for the complete contract.
"""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import socket
import struct
import threading
import time

import numpy as np
from PIL import Image


HEADER_LIMIT = 64 * 1024
PAYLOAD_LIMIT = 2 * 1024 * 1024
PREVIEW_LIMIT = (640, 480)


def _load_peer_config():
    config = {}
    path = Path(__file__).resolve().with_name("vision_peer.json")
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError):
        pass
    host = os.environ.get("PULSEATLAS_B_HOST", config.get("host", "192.168.2.100"))
    try:
        port = int(os.environ.get("PULSEATLAS_B_PORT", config.get("port", 8765)))
    except (TypeError, ValueError):
        port = 8765
    if not host or not 1 <= port <= 65535:
        raise ValueError("invalid B-board vision endpoint")
    return str(host), port


class VisionServiceClient:
    """Maintain a reconnecting, bounded-memory connection to B's vision service."""

    def __init__(self, host=None, port=None, connect_timeout=0.7,
                 reconnect_seconds=1.5):
        default_host, default_port = _load_peer_config()
        self.host = str(host or default_host)
        self.port = int(port or default_port)
        self.connect_timeout = float(connect_timeout)
        self.reconnect_seconds = float(reconnect_seconds)
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._socket = None
        self._desired_active = True
        self._state = self._empty_state()

    def _empty_state(self):
        return {
            "vision_active": True,
            "vision_camera_connected": False,
            "vision_camera_checked": True,
            "vision_frame_bgr": None,
            "vision_camera_frames": 0,
            "vision_camera_fps": 0.0,
            "vision_preview_frames": 0,
            "vision_preview_fps": 0.0,
            "vision_status": "等待 B 板视觉服务",
            "vision_error": None,
            "vision_fps": 0.0,
            "vision_inference_ms": None,
            "vision_inference_running": False,
            "vision_inference_count": 0,
            "vision_result_age_ms": None,
            "vision_result_status": "等待 B 板",
            "vision_confidence": None,
            "vision_stale": False,
            "vision_detected": False,
            "vision_frames": 0,
            "vision_marker_count": 0,
            "vision_pose_valid": False,
            "vision_service_connected": False,
            "vision_connector_status": "等待电极检测",
            "vision_connector_error": None,
            "vision_connector_inference_ms": None,
            "vision_connector_frame_age_ms": None,
            "vision_connector_correct": 0,
            "vision_connector_adjust": 0,
            "vision_connector_unknown": 10,
            "vision_connector_per_lead": {},
        }

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _publish(self, **updates):
        with self._lock:
            self._state.update(updates)

    def start(self, active=True):
        """Connect for status, optionally requesting active capture/inference."""
        with self._lock:
            self._desired_active = bool(active)
            self._state.update(
                vision_active=bool(active),
                vision_error=None,
                vision_status=("正在连接 B 板视觉服务 {}:{}".format(self.host, self.port)
                               if active else "正在监测 B 板摄像头连接 {}:{}".format(
                                   self.host, self.port)),
            )
            if self._thread is not None and self._thread.is_alive():
                self._send_command_locked("start" if active else "stop")
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="pulseatlas-b-vision-client", daemon=True
            )
            self._thread.start()

    def toggle(self):
        with self._lock:
            self._desired_active = not self._desired_active
            active = self._desired_active
            self._state["vision_active"] = active
            self._state["vision_status"] = (
                "正在请求 B 板启动视觉" if active else "已请求 B 板停止视觉"
            )
            self._send_command_locked("start" if active else "stop")

    def stop(self, timeout=3.0):
        self._stop.set()
        with self._send_lock:
            sock = self._socket
            if sock is not None:
                try:
                    self._send_packet(sock, {"type": "command", "action": "stop"})
                except OSError:
                    pass
        with self._lock:
            sock = self._socket
            self._socket = None
            thread = self._thread
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, float(timeout)))
        with self._lock:
            self._state.update(
                vision_active=False,
                vision_camera_connected=False,
                vision_service_connected=False,
                vision_status="A 板视觉客户端已停止",
                vision_inference_running=False,
                vision_frame_bgr=None,
            )
        return thread is None or not thread.is_alive()

    def _send_command_locked(self, action):
        # Caller holds self._lock; the independent send lock prevents packet
        # interleaving with the receive/reconnect worker and shutdown path.
        sock = self._socket
        if sock is None:
            return
        try:
            with self._send_lock:
                self._send_packet(sock, {"type": "command", "action": action})
        except OSError as exc:
            self._publish(vision_error="无法向 B 板发送命令：{}".format(exc))

    @staticmethod
    def _send_packet(sock, header, payload=b""):
        header = dict(header)
        header["payload_len"] = len(payload)
        encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > HEADER_LIMIT or len(payload) > PAYLOAD_LIMIT:
            raise ValueError("vision packet exceeds protocol limits")
        sock.sendall(struct.pack(">I", len(encoded)) + encoded + payload)

    @staticmethod
    def _recv_exact(sock, count, timeout=5.0):
        deadline = time.monotonic() + timeout
        chunks = []
        remaining = int(count)
        while remaining:
            left = deadline - time.monotonic()
            if left <= 0:
                raise socket.timeout("timed out waiting for a complete vision packet")
            sock.settimeout(min(left, 0.5))
            try:
                chunk = sock.recv(remaining)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("B-board vision service closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _receive_packet(self, sock):
        header_size = struct.unpack(">I", self._recv_exact(sock, 4))[0]
        if not 1 <= header_size <= HEADER_LIMIT:
            raise ValueError("invalid vision packet header length")
        header = json.loads(self._recv_exact(sock, header_size).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("vision packet header must be an object")
        payload_size = int(header.get("payload_len", 0))
        if not 0 <= payload_size <= PAYLOAD_LIMIT:
            raise ValueError("invalid vision packet payload length")
        payload = self._recv_exact(sock, payload_size) if payload_size else b""
        return header, payload

    def _run(self):
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout
                )
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                with self._lock:
                    self._socket = sock
                    desired = self._desired_active
                    self._state.update(
                        vision_service_connected=True,
                        vision_camera_checked=True,
                        vision_error=None,
                        vision_status="B 板服务已连接，等待摄像头状态",
                    )
                    with self._send_lock:
                        self._send_packet(sock, {
                            "type": "subscribe",
                            "version": 1,
                            "preview": "jpeg",
                            "active": bool(desired),
                        })
                while not self._stop.is_set():
                    header, payload = self._receive_packet(sock)
                    self._handle_packet(header, payload)
            except (OSError, ValueError, UnicodeError, ConnectionError) as exc:
                if not self._stop.is_set():
                    self._publish(
                        vision_service_connected=False,
                        vision_camera_checked=True,
                        vision_camera_connected=False,
                        vision_inference_running=False,
                        vision_frame_bgr=None,
                        vision_error="B 板视觉服务暂不可达：{}".format(str(exc)[:90]),
                        vision_status="等待 B 板视觉服务 {}:{}".format(self.host, self.port),
                        vision_result_status="等待 B 板连接",
                    )
            finally:
                with self._lock:
                    if self._socket is sock:
                        self._socket = None
                    self._state["vision_service_connected"] = False
                    self._state["vision_camera_connected"] = False
                    self._state["vision_inference_running"] = False
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self._stop.wait(self.reconnect_seconds)

    def _handle_packet(self, header, payload):
        kind = header.get("type")
        if kind == "status":
            updates = {
                "vision_service_connected": True,
                "vision_camera_checked": True,
                "vision_camera_connected": bool(header.get("camera_connected", False)),
                "vision_camera_frames": int(header.get("camera_frames", 0) or 0),
                "vision_camera_fps": float(header.get("camera_fps", 0.0) or 0.0),
                "vision_fps": float(header.get("inference_fps", 0.0) or 0.0),
                "vision_inference_ms": header.get("inference_ms"),
                "vision_inference_running": bool(header.get("inference_running", False)),
                "vision_inference_count": int(header.get("inference_count", 0) or 0),
                "vision_result_age_ms": header.get("result_age_ms"),
                "vision_result_status": str(header.get("result_status", "等待推理")),
                "vision_confidence": header.get("confidence"),
                "vision_stale": bool(header.get("stale", False)),
                "vision_detected": bool(header.get("detected", False)),
                "vision_frames": int(header.get("inference_count", 0) or 0),
                "vision_marker_count": int(header.get("marker_count", 0) or 0),
                "vision_pose_valid": bool(header.get("pose_valid", False)),
                "vision_status": str(header.get("status", "B 板视觉服务在线")),
                "vision_error": header.get("error"),
                "vision_connector_status": str(
                    header.get("connector_status", "等待电极检测")
                ),
                "vision_connector_error": header.get("connector_error"),
                "vision_connector_inference_ms": header.get("connector_inference_ms"),
                "vision_connector_frame_age_ms": header.get("connector_frame_age_ms"),
                "vision_connector_correct": int(header.get("connector_correct", 0) or 0),
                "vision_connector_adjust": int(header.get("connector_adjust", 0) or 0),
                "vision_connector_unknown": int(header.get("connector_unknown", 10) or 0),
                "vision_connector_per_lead": (
                    header.get("connector_per_lead")
                    if isinstance(header.get("connector_per_lead"), dict) else {}
                ),
            }
            self._publish(**updates)
            return
        if kind == "frame":
            if not payload:
                return
            with Image.open(BytesIO(payload)) as source:
                if source.width > 1920 or source.height > 1080:
                    raise ValueError("B-board preview exceeds 1920x1080")
                preview = source.convert("RGB")
                preview.thumbnail(PREVIEW_LIMIT, getattr(Image, "Resampling", Image).BILINEAR)
                rgb = np.asarray(preview, dtype=np.uint8)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            self._publish(
                vision_frame_bgr=bgr,
                vision_preview_frames=self.snapshot()["vision_preview_frames"] + 1,
                vision_service_connected=True,
                vision_camera_checked=True,
                vision_camera_connected=True,
                vision_status="B 板相机画面已接收",
                vision_error=None,
            )
            return
        if kind in ("hello", "ack", "pong"):
            if header.get("error"):
                self._publish(vision_error=str(header["error"]))
            return
        raise ValueError("unsupported B-board message type: {!r}".format(kind))


__all__ = ["VisionServiceClient"]
