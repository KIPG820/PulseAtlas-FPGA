"""Verify TFT payload writes stay inside the PYNQ Arduino mailbox BRAM."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

BOARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOARD))

from pulseatlas_tft_backend import (  # noqa: E402
    CMD_FRAME_BEGIN,
    CMD_FRAME_CHUNK,
    CMD_FRAME_END,
    MAILBOX_BASE_OFFSET,
    MAILBOX_PAYLOAD,
    MAILBOX_STATUS,
    PulseAtlasTFTBackend,
    STATE_ACK_BEGIN,
    STATE_ACK_CHUNK,
    STATE_ACK_END,
    CMD_FRAME_RECT,
    STATE_ACK_RECT,
    STATE_TOUCH_EVENT,
)
from pulseatlas_tft_render import HEIGHT, WIDTH


class FakeMailbox:
    def __init__(self):
        self.status = STATE_TOUCH_EVENT
        self.payload = [7, 0, 0]
        self.mmio = SimpleNamespace(array=np.zeros(0x10000 // 4, dtype=np.uint32))

    def read_mailbox(self, offset, count=1):
        if offset == MAILBOX_STATUS:
            return self.status
        if offset == MAILBOX_PAYLOAD and count == 3:
            return list(self.payload)
        raise AssertionError((offset, count))

    def write_mailbox(self, offset, value):
        if offset == MAILBOX_STATUS:
            self.status = value
        elif offset == MAILBOX_PAYLOAD:
            # Production writes rectangle coordinates here before issuing the
            # frame command; raw pixel payload is written directly to BRAM.
            self.payload = value
        elif offset == 0:
            self.status = {
                CMD_FRAME_BEGIN: STATE_ACK_BEGIN,
                CMD_FRAME_CHUNK: STATE_ACK_CHUNK,
                CMD_FRAME_RECT: STATE_ACK_RECT,
                CMD_FRAME_END: STATE_ACK_END,
            }[value]
        else:
            raise AssertionError((offset, value))


def main():
    memory = np.zeros(0x10000 // 4, dtype=np.uint32)
    backend = object.__new__(PulseAtlasTFTBackend)
    backend.mb = SimpleNamespace(mmio=SimpleNamespace(array=memory))
    payload = np.asarray([0x12345678, 0x90ABCDEF], dtype="<u4")

    backend._write_payload(payload.tobytes())

    mailbox_word = (MAILBOX_BASE_OFFSET + MAILBOX_PAYLOAD) // 4
    np.testing.assert_array_equal(memory[mailbox_word:mailbox_word + 2], payload)
    assert not memory[MAILBOX_PAYLOAD // 4:MAILBOX_PAYLOAD // 4 + 2].any(), \
        "payload must not overwrite MicroBlaze program memory at BRAM offset 0"

    # A touch that arrives between the app's poll and a TFT refresh must be
    # queued across frame transfer, not cleared as stale mailbox state.
    backend.rotation = 90
    backend.mb = FakeMailbox()
    backend._pending_touches = []
    from PIL import Image
    backend.present(Image.new("RGB", (WIDTH, HEIGHT), (12, 34, 56)))
    assert backend.mb.status == 0
    assert backend.poll_touch() == (7, 319, 0)
    print("TFT mailbox offset and touch preservation passed")


if __name__ == "__main__":
    main()
