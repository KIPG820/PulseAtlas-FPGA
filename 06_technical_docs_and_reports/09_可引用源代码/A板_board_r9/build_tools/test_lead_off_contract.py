"""Board-free checks for the ADS1298 electrode-detection contract."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


pynq = types.ModuleType("pynq")
pynq.allocate = lambda *args, **kwargs: None
sys.modules.setdefault("pynq", pynq)


def _pipeline_source():
    package_root = Path(__file__).resolve().parents[1]
    packaged = package_root / "ecg_full_pipeline.py"
    if packaged.is_file():
        return packaged
    relative = Path("PYNQ-3.1.1") / "boards" / "Pynq-Z2" / "ecg_nn_r9" / "board_test" / "ecg_full_pipeline.py"
    for checkout in Path(__file__).resolve().parents:
        candidate = checkout / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "ecg_full_pipeline.py was not found in this package or its source checkout"
    )


source = _pipeline_source()
spec = importlib.util.spec_from_file_location("ecg_pipeline_leadoff_test", source)
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)


class FakeADC:
    def __init__(self, capability=0x4C4F4646, writable=True):
        self.regs = {0x34: capability, 0x04: 0xA1, 0x30: 0x92,
                     0x24: 0x00, 0x28: 0x00, 0x2C: 0x00, 0x00: 0x00}
        self.writable = writable

    def read(self, offset):
        return self.regs.get(offset, 0)

    def write(self, offset, value):
        if not self.writable and offset in (0x24, 0x28, 0x2C):
            return
        if offset == 0x00:
            self.regs[0x04] = 0xA1 if value else 0x80
        self.regs[offset] = value


def main():
    p = object.__new__(pipeline_module.ECGFullPipeline)
    p.adc = FakeADC(capability=0)
    assert not p.lead_off_ready()
    try:
        p.start_ads1298()
    except RuntimeError as exc:
        assert "不支持电极脱落检测" in str(exc)
    else:
        raise AssertionError("old overlay was accepted")

    p.adc = FakeADC(writable=False)
    try:
        p.start_ads1298()
    except RuntimeError as exc:
        assert "寄存器回读失败" in str(exc)
    else:
        raise AssertionError("bad register readback was accepted")

    p.adc = FakeADC()
    assert p.start_ads1298()
    assert not p.lead_off_ready(), "normal monitor must keep lead-off off"
    assert p.adc.read(0x00) & 0x10 == 0
    assert p.adc_status()["lead_off_config"] == (0x13, 0xFF, 0x03)
    assert p.start_ads1298(lead_off=True)
    assert p.lead_off_ready()
    assert p.adc.read(0x00) & 0x10
    p.adc.regs[0x2C] = 0
    assert not p.lead_off_ready()
    p.adc.regs[0x2C] = 0x03
    p.adc.regs[0x00] |= 0x04
    assert not p.lead_off_ready(), "internal ADC test may not become a clinical result"

    # Packet quality word must present the combined channel fault mask.
    packet = np.zeros(18, dtype=np.uint32)
    packet[0] = pipeline_module.PACKET_MAGIC
    packet[1] = (pipeline_module.TYPE_LEAD12 << 24) | 18
    packet[3] = 0xC80040  # CH8P and CH3N set in the ADS status word
    packet[4] = 0x80      # only CH8P is monitored; CH3N is internal WCT
    record = pipeline_module.PacketParser.parse(packet)[0]
    assert record["ads_status"] == 0xC80040
    assert record["lead_off"] == 0x80
    print("lead-off contract checks passed")


if __name__ == "__main__":
    main()
