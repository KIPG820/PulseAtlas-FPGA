"""B-board TCP server for PulseAtlas A's remote vision page."""

from __future__ import annotations

import json
import os
import queue
import socket
import struct
import threading
import time

import numpy as np
import cv2


HEADER_LIMIT = 64 * 1024
PAYLOAD_LIMIT = 2 * 1024 * 1024
PREVIEW_SIZE = (640, 480)


def _recv_exact(sock, count):
    chunks = []
    remaining = int(count)
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            # Preserve a partially read packet while polling for shutdown or
            # the peer's next command. An EOF is still detected immediately.
            continue
        if not chunk:
            raise ConnectionError("A-board client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_packet(sock):
    header_size = struct.unpack(">I", _recv_exact(sock, 4))[0]
    if not 1 <= header_size <= HEADER_LIMIT:
        raise ValueError("invalid A-board packet header length")
    header = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("A-board packet header must be an object")
    payload_size = int(header.get("payload_len", 0))
    if not 0 <= payload_size <= PAYLOAD_LIMIT:
        raise ValueError("invalid A-board packet payload length")
    payload = _recv_exact(sock, payload_size) if payload_size else b""
    return header, payload


def _send_packet(sock, send_lock, header, payload=b""):
    header = dict(header)
    header["payload_len"] = len(payload)
    encoded = json.dumps(header, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(encoded) > HEADER_LIMIT or len(payload) > PAYLOAD_LIMIT:
        raise ValueError("B-board vision packet exceeds protocol limits")
    packet = struct.pack(">I", len(encoded)) + encoded + payload
    with send_lock:
        sock.sendall(packet)


def _number(value):
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


class VisionServiceServer:
    """Serve one A-board client with bounded preview rate and latest-frame only."""

    def __init__(self, runtime, host="0.0.0.0", port=8765, preview_fps=5.0,
                 jpeg_quality=72):
        if not 0.2 <= float(preview_fps) <= 15.0:
            raise ValueError("preview_fps must be between 0.2 and 15")
        if not 30 <= int(jpeg_quality) <= 95:
            raise ValueError("jpeg_quality must be between 30 and 95")
        self.runtime = runtime
        self.host = str(host)
        self.port = int(port)
        self.preview_interval = 1.0 / float(preview_fps)
        self.jpeg_quality = int(jpeg_quality)
        self._stop = threading.Event()
        self._listener = None
        self._client = None
        self._client_lock = threading.Lock()

    def serve_forever(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(2)
        listener.settimeout(0.5)
        self._listener = listener
        print("PulseAtlas B vision service listening on {}:{}".format(
            self.host, self.port), flush=True)
        try:
            while not self._stop.is_set():
                try:
                    client, address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                print("PulseAtlas A client connected: {}:{}".format(*address), flush=True)
                client.settimeout(1.0)
                with self._client_lock:
                    self._client = client
                try:
                    self._serve_client(client)
                except (OSError, ValueError, UnicodeError, ConnectionError) as exc:
                    if not self._stop.is_set():
                        print("PulseAtlas A connection ended: {}".format(exc), flush=True)
                finally:
                    self._set_runtime_active(False)
                    with self._client_lock:
                        if self._client is client:
                            self._client = None
                    try:
                        client.close()
                    except OSError:
                        pass
        finally:
            self._set_runtime_active(False)
            try:
                listener.close()
            except OSError:
                pass
            self._listener = None

    def _serve_client(self, client):
        first, payload = _recv_packet(client)
        if payload or first.get("type") != "subscribe" or int(first.get("version", 0)) != 1:
            raise ValueError("expected version-1 subscribe packet")
        if first.get("preview") not in ("jpeg", None):
            raise ValueError("only JPEG previews are supported")

        commands = queue.Queue(maxsize=8)
        send_lock = threading.Lock()
        reader_stop = threading.Event()

        def read_commands():
            try:
                while not reader_stop.is_set() and not self._stop.is_set():
                    header, body = _recv_packet(client)
                    if body:
                        raise ValueError("commands cannot have a payload")
                    kind = header.get("type")
                    if kind == "command" and header.get("action") in ("start", "stop"):
                        try:
                            commands.put_nowait(header["action"])
                        except queue.Full:
                            # Only the newest button state matters.
                            try:
                                commands.get_nowait()
                            except queue.Empty:
                                pass
                            commands.put_nowait(header["action"])
                    elif kind == "ping":
                        try:
                            commands.put_nowait("pong")
                        except queue.Full:
                            pass
                    else:
                        raise ValueError("unsupported A-board command: {!r}".format(kind))
            except (OSError, ValueError, UnicodeError, ConnectionError):
                if not reader_stop.is_set() and not self._stop.is_set():
                    try:
                        commands.put_nowait("disconnect")
                    except queue.Full:
                        pass

        reader = threading.Thread(target=read_commands,
                                  name="pulseatlas-a-command-reader", daemon=True)
        reader.start()
        self._set_runtime_active(bool(first.get("active", True)))
        last_status = 0.0
        last_preview = 0.0
        last_frame_sequence = -1
        try:
            while not self._stop.is_set():
                while True:
                    try:
                        action = commands.get_nowait()
                    except queue.Empty:
                        break
                    if action == "disconnect":
                        return
                    if action == "pong":
                        _send_packet(client, send_lock, {"type": "pong"})
                    else:
                        self._set_runtime_active(action == "start")

                now = time.monotonic()
                state = self.runtime.snapshot()
                if now - last_status >= 1.0:
                    _send_packet(client, send_lock, self._status_message(state))
                    last_status = now

                seq = int(state.get("vision_preview_frames") or 0)
                frame = state.get("vision_frame_bgr")
                if (bool(state.get("vision_active")) and
                        bool(state.get("vision_camera_connected")) and
                        frame is not None and seq != last_frame_sequence and
                        now - last_preview >= self.preview_interval):
                    jpeg = self._encode_jpeg(frame)
                    if jpeg:
                        _send_packet(client, send_lock,
                                     {"type": "frame", "seq": seq}, jpeg)
                        last_frame_sequence = seq
                        last_preview = now
                self._stop.wait(0.02)
        finally:
            reader_stop.set()
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            if reader is not threading.current_thread():
                reader.join(timeout=0.5)

    def _camera_device_present(self):
        """Check Linux enumeration only; this does not open the camera."""
        camera_index = getattr(self.runtime, "camera_index", None)
        if camera_index is None:
            return None
        return os.path.exists("/dev/video{}".format(int(camera_index)))

    def _status_message(self, state):
        camera_device_present = self._camera_device_present()
        camera_connected = (bool(state.get("vision_camera_connected", False))
                            if camera_device_present is None
                            else camera_device_present)
        return {
            "type": "status",
            # Report live Linux device-node presence; this works during idle
            # status monitoring and also catches hot-unplug during capture.
            "camera_connected": camera_connected,
            "camera_frames": int(state.get("vision_camera_frames", 0) or 0),
            "camera_fps": _number(state.get("vision_camera_fps")) or 0.0,
            "inference_fps": _number(state.get("vision_fps")) or 0.0,
            "inference_ms": _number(state.get("vision_inference_ms")),
            "inference_running": bool(state.get("vision_inference_running", False)),
            "inference_count": int(state.get("vision_inference_count", 0) or 0),
            "result_age_ms": _number(state.get("vision_result_age_ms")),
            "result_status": str(state.get("vision_result_status") or "等待推理"),
            "confidence": _number(state.get("vision_confidence")),
            "stale": bool(state.get("vision_stale", False)),
            "detected": bool(state.get("vision_detected", False)),
            "marker_count": int(state.get("vision_marker_count", 0) or 0),
            "pose_valid": bool(state.get("vision_pose_valid", False)),
            "status": str(state.get("vision_status") or "B 板视觉服务在线"),
            "error": state.get("vision_error"),
            "backend": str(state.get("vision_backend") or "PS CPU"),
        }

    def _encode_jpeg(self, frame_bgr):
        array = np.asarray(frame_bgr)
        if array.ndim != 3 or array.shape[2] < 3:
            return b""
        bgr = np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)
        height, width = bgr.shape[:2]
        if width > PREVIEW_SIZE[0] or height > PREVIEW_SIZE[1]:
            scale = min(PREVIEW_SIZE[0] / width, PREVIEW_SIZE[1] / height)
            bgr = cv2.resize(bgr, (max(1, round(width * scale)),
                                   max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return b""
        data = encoded.tobytes()
        return data if len(data) <= PAYLOAD_LIMIT else b""

    def _set_runtime_active(self, active):
        current = bool(self.runtime.snapshot().get("vision_active", False))
        if active and not current:
            self.runtime.start()
        elif not active and current:
            self.runtime.stop(timeout=3.0)

    def close(self):
        self._stop.set()
        with self._client_lock:
            client = self._client
            self._client = None
        for sock in (client, self._listener):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        self._set_runtime_active(False)


__all__ = ["VisionServiceServer"]
