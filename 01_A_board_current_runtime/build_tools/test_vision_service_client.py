"""Local fake-server tests for the A-to-B vision transport adapter."""

from __future__ import annotations

from io import BytesIO
import json
import queue
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_service_client import VisionServiceClient  # noqa: E402


def _read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("peer closed")
        data.extend(part)
    return bytes(data)


def _read_packet(sock):
    size = struct.unpack(">I", _read_exact(sock, 4))[0]
    header = json.loads(_read_exact(sock, size).decode("utf-8"))
    payload = _read_exact(sock, int(header.get("payload_len", 0)))
    return header, payload


def _send_packet(sock, header, payload=b""):
    header = dict(header, payload_len=len(payload))
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(">I", len(encoded)) + encoded + payload)


class VisionServiceClientTests(unittest.TestCase):
    def test_passive_start_subscribes_for_status_without_activating_vision(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2.0)
        port = listener.getsockname()[1]
        events = queue.Queue()

        def serve():
            conn = None
            try:
                conn, _ = listener.accept()
                conn.settimeout(2.0)
                subscribe, _ = _read_packet(conn)
                events.put(subscribe)
                _send_packet(conn, {"type": "status", "camera_connected": True})
                command, _ = _read_packet(conn)
                events.put(command)
            finally:
                if conn is not None:
                    conn.close()
                listener.close()

        server = threading.Thread(target=serve, name="fake-passive-vision", daemon=True)
        client = VisionServiceClient("127.0.0.1", port, connect_timeout=0.5,
                                     reconnect_seconds=0.1)
        server.start()
        client.start(active=False)
        try:
            subscribe = events.get(timeout=2.0)
            self.assertFalse(subscribe["active"])
            self.assertFalse(client.snapshot()["vision_active"])
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not client.snapshot()["vision_camera_connected"]:
                time.sleep(0.01)
            self.assertTrue(client.snapshot()["vision_camera_connected"])
        finally:
            client.stop(timeout=1.0)
        self.assertEqual(events.get(timeout=2.0), {"type": "command", "action": "stop",
                                                  "payload_len": 0})
        server.join(timeout=1.0)

    def test_receives_status_and_jpeg_and_sends_stop_command(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3.0)
        port = listener.getsockname()[1]
        server_events = queue.Queue()
        server_error = []

        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[:, :] = (245, 20, 30)
        image_data = BytesIO()
        Image.fromarray(rgb).save(image_data, format="JPEG", quality=90)
        jpeg = image_data.getvalue()

        def serve():
            conn = None
            try:
                conn, _ = listener.accept()
                conn.settimeout(3.0)
                subscribe, _ = _read_packet(conn)
                server_events.put(subscribe)
                _send_packet(conn, {
                    "type": "status", "camera_connected": True,
                    "camera_frames": 12, "camera_fps": 24.0,
                    "inference_fps": 3.5, "inference_ms": 280.0,
                    "inference_count": 4, "result_status": "定位有效",
                    "marker_count": 10, "pose_valid": True,
                    "confidence": 0.9, "status": "视觉服务在线",
                    "connector_status": "标定点局部电极与颜色模型已加载",
                    "connector_inference_ms": 18.5,
                    "connector_frame_age_ms": 42.0,
                    "connector_correct": 8,
                    "connector_adjust": 1,
                    "connector_unknown": 1,
                    "connector_per_lead": {
                        "RA": {"status": "correct", "confidence": 0.94},
                        "LA": {"status": "adjust", "confidence": 0.83},
                    },
                })
                _send_packet(conn, {"type": "frame", "seq": 12}, jpeg)
                command, _ = _read_packet(conn)
                server_events.put(command)
            except Exception as exc:  # surfaced as an assertion in the test
                server_error.append(exc)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass
                listener.close()

        server = threading.Thread(target=serve, name="fake-vision-service", daemon=True)
        client = VisionServiceClient("127.0.0.1", port, connect_timeout=0.5,
                                     reconnect_seconds=0.1)
        server.start()
        client.start()
        try:
            deadline = time.monotonic() + 3.0
            state = client.snapshot()
            while time.monotonic() < deadline:
                state = client.snapshot()
                if state["vision_preview_frames"] and state["vision_pose_valid"]:
                    break
                time.sleep(0.01)
            self.assertEqual(state["vision_result_status"], "定位有效")
            self.assertEqual(state["vision_marker_count"], 10)
            self.assertEqual(state["vision_inference_count"], 4)
            self.assertEqual(state["vision_camera_frames"], 12)
            self.assertEqual(state["vision_connector_status"],
                             "标定点局部电极与颜色模型已加载")
            self.assertEqual(state["vision_connector_inference_ms"], 18.5)
            self.assertEqual(state["vision_connector_correct"], 8)
            self.assertEqual(state["vision_connector_adjust"], 1)
            self.assertEqual(state["vision_connector_unknown"], 1)
            self.assertEqual(state["vision_connector_per_lead"]["LA"]["status"],
                             "adjust")
            self.assertTrue(state["vision_camera_connected"])
            self.assertEqual(state["vision_frame_bgr"].shape, (8, 12, 3))
            self.assertGreater(int(state["vision_frame_bgr"][0, 0, 2]), 200)

            client.toggle()
            subscribe = server_events.get(timeout=2.0)
            command = server_events.get(timeout=2.0)
            self.assertEqual(subscribe["type"], "subscribe")
            self.assertEqual(subscribe["version"], 1)
            self.assertFalse(subscribe["payload_len"])
            self.assertEqual(command, {"type": "command", "action": "stop",
                                       "payload_len": 0})
            self.assertFalse(client.snapshot()["vision_active"])
            self.assertFalse(server_error, server_error)
        finally:
            client.stop(timeout=1.0)
            server.join(timeout=1.0)

    def test_unreachable_peer_is_nonfatal_and_reported(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        client = VisionServiceClient("127.0.0.1", port, connect_timeout=0.1,
                                     reconnect_seconds=0.05)
        client.start()
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if client.snapshot()["vision_error"]:
                    break
                time.sleep(0.01)
            state = client.snapshot()
            self.assertFalse(state["vision_camera_connected"])
            self.assertFalse(state["vision_service_connected"])
            self.assertIn("B 板视觉服务暂不可达", state["vision_error"])
        finally:
            client.stop(timeout=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
