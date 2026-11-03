"""Regression: AI capture must re-arm S2MM before parsing each DMA block."""

import sys
import types
from pathlib import Path

import numpy as np


# The tested capture path only needs PYNQ's allocate symbol; replace it before
# importing the board module so the test also runs on a development PC.
pynq = types.ModuleType("pynq")
pynq.allocate = None
sys.modules["pynq"] = pynq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ecg_full_pipeline as module


events = []


class FakeBuffer(np.ndarray):
    def __new__(cls, shape, dtype):
        return np.zeros(shape, dtype=dtype).view(cls)

    def freebuffer(self):
        events.append("free_buffer")


class FakeRecvChannel:
    def __init__(self):
        self.transferred = 256 * 18 * 4
        self.transfer_count = 0

    def transfer(self, _buffer):
        self.transfer_count += 1
        events.append("transfer_{}".format(self.transfer_count))

    def wait(self):
        events.append("wait")

    def stop(self):
        events.append("dma_stop")


class FakeDMA:
    def __init__(self):
        self.recvchannel = FakeRecvChannel()


def fake_parse(words):
    assert events[-1] == "transfer_2", (
        "the next DMA transfer must be armed before Python parses the block")
    assert words.size == 256 * 18
    events.append("parse")
    return [{"type": module.TYPE_LEAD12, "frame": index,
             "samples": np.zeros(12, dtype=np.int32),
             "lead_off": 0, "saturation": 0, "flatline": 0,
             "jump": 0, "lead_invalid": 0}
            for index in range(256)]


def main():
    original_allocate = module.allocate
    original_parse = module.PacketParser.parse
    module.allocate = lambda shape, dtype: FakeBuffer(shape, dtype)
    module.PacketParser.parse = staticmethod(fake_parse)
    try:
        pipeline = module.ECGFullPipeline.__new__(module.ECGFullPipeline)
        pipeline.sample_rate = 1000
        pipeline.dma = FakeDMA()
        pipeline.configure_processing = lambda: events.append("configure")
        pipeline.start_ads1298 = lambda **_kwargs: True
        pipeline.stop = lambda: events.append("pipeline_stop")
        pipeline.adc_status = lambda: {"drop_count": 0}
        pipeline.pipeline_status = lambda: {}

        result = pipeline.capture(frame_count=256, timeout_s=2.0)
        assert result["leads"].shape == (256, 12)
        assert events.index("transfer_2") < events.index("parse")
        assert events.index("dma_stop") < events.index("free_buffer")
        assert events.count("transfer_1") == 1
        assert events.count("transfer_2") == 1
    finally:
        module.allocate = original_allocate
        module.PacketParser.parse = original_parse

    print("capture DMA re-arm and shutdown ordering checks passed")


if __name__ == "__main__":
    main()
