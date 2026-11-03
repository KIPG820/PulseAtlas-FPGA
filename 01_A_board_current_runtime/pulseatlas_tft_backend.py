"""PYNQ Arduino-IOP transport for the shared PulseAtlas TFT UI."""

from __future__ import annotations

import time

import numpy as np

from pulseatlas_tft_render import HEIGHT, WIDTH, rgb565_bytes


MAILBOX_COMMAND = 0
MAILBOX_STATUS = 4
MAILBOX_PAYLOAD = 8
# PYNQ 3.x maps the Arduino MicroBlaze mailbox into the final 4 KiB of its
# 64 KiB BRAM window. Raw MMIO writes must include this base offset too.
MAILBOX_BASE_OFFSET = 0xF000
CMD_FRAME_BEGIN = 0x46524247
CMD_FRAME_CHUNK = 0x43484E4B
CMD_FRAME_RECT = 0x52454354
CMD_FRAME_END = 0x4652454E
STATE_READY = 0x52454144
STATE_ACK_BEGIN = 0x41434B31
STATE_ACK_CHUNK = 0x41434B32
STATE_ACK_RECT = 0x41434B34
STATE_ACK_END = 0x41434B33
STATE_TOUCH_EVENT = 0x544F5543
MAX_CHUNK_WORDS = 1020
DIRTY_TILE = 16


def touch_to_landscape(raw_x, raw_y, rotation=270):
    """Rotate native GT911 coordinates (240x320) into 320x240 display space."""
    if rotation == 90:
        x, y = 319 - int(raw_y), int(raw_x)
    elif rotation == 270:
        x, y = int(raw_y), 239 - int(raw_x)
    else:
        raise ValueError("rotation must be 90 or 270 degrees")
    return max(0, min(319, x)), max(0, min(239, y))


def _changed_rectangles(previous, current):
    """Group modified 16px tiles into rectangles sent within one SPI session."""
    if previous is None:
        return [(0, 0, WIDTH - 1, HEIGHT - 1)]
    old = np.frombuffer(previous, dtype="<u2").reshape(HEIGHT, WIDTH)
    new = np.frombuffer(current, dtype="<u2").reshape(HEIGHT, WIDTH)
    changed = old != new
    if not changed.any():
        return []
    tiles = changed.reshape(HEIGHT // DIRTY_TILE, DIRTY_TILE,
                            WIDTH // DIRTY_TILE, DIRTY_TILE).any(axis=(1, 3))
    finished = []
    active = {}
    for tile_y, row in enumerate(tiles):
        runs = []
        start = None
        for tile_x, is_dirty in enumerate(row):
            if is_dirty and start is None:
                start = tile_x
            elif not is_dirty and start is not None:
                runs.append((start, tile_x - 1))
                start = None
        if start is not None:
            runs.append((start, len(row) - 1))
        following = {}
        for run in runs:
            if run in active:
                rect = active.pop(run)
                rect[3] = (tile_y + 1) * DIRTY_TILE - 1
            else:
                rect = [run[0] * DIRTY_TILE, tile_y * DIRTY_TILE,
                        (run[1] + 1) * DIRTY_TILE - 1,
                        (tile_y + 1) * DIRTY_TILE - 1]
            following[run] = rect
        finished.extend(active.values())
        active = following
    finished.extend(active.values())
    cost = sum((x1 - x0 + 1) * (y1 - y0 + 1)
               for x0, y0, x1, y1 in finished)
    if cost >= WIDTH * HEIGHT * 0.7 or len(finished) > 48:
        return [(0, 0, WIDTH - 1, HEIGHT - 1)]
    return [tuple(rect) for rect in finished]


class PulseAtlasTFTBackend:
    """Drive ST7789 frames and return GT911 taps through a running R9 overlay.

    The bitstream must expose the PYNQ Arduino IOP as ``overlay.ARDUINO`` and
    the MicroBlaze program must implement ``tools/tft_pulseatlas_iop.c``.
    This class never loads PYNQ Base or changes the active overlay.
    """

    def __init__(self, overlay, iop_binary, rotation=270, ready_timeout=45.0):
        if not hasattr(overlay, "ARDUINO"):
            raise RuntimeError(
                "当前 R9 overlay 未暴露 Arduino IOP；需要先把 IOP 纳入 R9 bitstream"
            )
        if rotation not in (90, 270):
            raise ValueError("rotation must be 90 or 270 degrees")
        from pynq.lib.arduino import Arduino

        self.rotation = int(rotation)
        self._pending_touches = []
        self._last_frame = None
        self.last_transfer_pixels = 0
        self.last_transfer_rectangles = 0
        self.last_transfer_seconds = 0.0
        self.mb = Arduino(overlay.ARDUINO, str(iop_binary))
        self.mb.run()
        self._wait_mailbox(MAILBOX_COMMAND, STATE_READY, ready_timeout)

    def _wait_mailbox(self, offset, expected, timeout):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            value = self.mb.read_mailbox(offset)
            if value == expected:
                return
            time.sleep(0.002)
        actual = self.mb.read_mailbox(offset)
        raise TimeoutError(
            "TFT IOP mailbox 0x{:02X}: got 0x{:08X}, wanted 0x{:08X}".format(
                offset, actual, expected)
        )

    def _write_payload(self, data):
        if len(data) & 3:
            data += b"\x00" * (4 - (len(data) & 3))
        words = np.frombuffer(data, dtype="<u4")
        start = (MAILBOX_BASE_OFFSET + MAILBOX_PAYLOAD) // 4
        self.mb.mmio.array[start:start + len(words)] = words

    def present(self, image):
        frame = rgb565_bytes(image)
        expected = WIDTH * HEIGHT * 2
        if len(frame) != expected:
            raise ValueError("expected {} RGB565 bytes, got {}".format(expected, len(frame)))
        # A tap can arrive after the UI loop's poll and just before a frame
        # transfer. Preserve it before reusing the shared status mailbox for
        # the LCD protocol instead of silently discarding the event.
        touch = self._read_touch_event()
        if touch is not None:
            self._pending_touches.append(touch)
        rectangles = _changed_rectangles(getattr(self, "_last_frame", None), frame)
        if not rectangles:
            return
        pixels = np.frombuffer(frame, dtype="<u2").reshape(HEIGHT, WIDTH)
        started = time.monotonic()
        sent_pixels = 0
        for index, rect in enumerate(rectangles):
            x0, y0, x1, y1 = rect
            self.mb.write_mailbox(MAILBOX_PAYLOAD, list(rect))
            self.mb.write_mailbox(
                MAILBOX_COMMAND, CMD_FRAME_BEGIN if index == 0 else CMD_FRAME_RECT)
            self._wait_mailbox(
                MAILBOX_STATUS, STATE_ACK_BEGIN if index == 0 else STATE_ACK_RECT, 5.0)
            self.mb.write_mailbox(MAILBOX_STATUS, 0)
            region = pixels[y0:y1 + 1, x0:x1 + 1].tobytes()
            count_pixels = (x1 - x0 + 1) * (y1 - y0 + 1)
            for first in range(0, count_pixels, MAX_CHUNK_WORDS * 2):
                count = min(MAX_CHUNK_WORDS * 2, count_pixels - first)
                self._write_payload(region[first * 2:(first + count) * 2])
                self.mb.write_mailbox(MAILBOX_STATUS, count)
                self.mb.write_mailbox(MAILBOX_COMMAND, CMD_FRAME_CHUNK)
                self._wait_mailbox(MAILBOX_STATUS, STATE_ACK_CHUNK, 45.0)
                self.mb.write_mailbox(MAILBOX_STATUS, 0)
            sent_pixels += count_pixels

        self.mb.write_mailbox(MAILBOX_COMMAND, CMD_FRAME_END)
        self._wait_mailbox(MAILBOX_STATUS, STATE_ACK_END, 10.0)
        self.mb.write_mailbox(MAILBOX_STATUS, 0)
        self._last_frame = frame
        self.last_transfer_pixels = sent_pixels
        self.last_transfer_rectangles = len(rectangles)
        self.last_transfer_seconds = time.monotonic() - started

    def poll_touch(self):
        if self._pending_touches:
            return self._pending_touches.pop(0)
        return self._read_touch_event()

    def _read_touch_event(self):
        if self.mb.read_mailbox(MAILBOX_STATUS) != STATE_TOUCH_EVENT:
            return None
        track, raw_x, raw_y = self.mb.read_mailbox(MAILBOX_PAYLOAD, 3)
        self.mb.write_mailbox(MAILBOX_STATUS, 0)
        x, y = touch_to_landscape(raw_x, raw_y, self.rotation)
        return int(track), x, y

    def close(self):
        try:
            self.mb.reset()
        except Exception:
            pass


__all__ = ["PulseAtlasTFTBackend", "touch_to_landscape"]
