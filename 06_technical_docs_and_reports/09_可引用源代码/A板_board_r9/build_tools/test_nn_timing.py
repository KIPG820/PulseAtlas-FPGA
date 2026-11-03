"""Verify FPGA timing covers the complete FPGA+DMA inference call."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def _board_test_dir():
    package_root = Path(__file__).resolve().parents[1]
    if (package_root / "nn_accel.py").is_file():
        return package_root
    relative = Path("PYNQ-3.1.1") / "boards" / "Pynq-Z2" / "ecg_nn_r9" / "board_test"
    for checkout in Path(__file__).resolve().parents:
        candidate = checkout / relative
        if (candidate / "nn_accel.py").is_file():
            return candidate
    raise FileNotFoundError("nn_accel.py was not found in this package or source checkout")


BOARD_TEST = _board_test_dir()
sys.modules["pynq"] = types.SimpleNamespace(allocate=lambda **_kwargs: None)
sys.modules["ps_numpy_reference"] = types.SimpleNamespace(NumpyIntegerECG=object)
spec = importlib.util.spec_from_file_location("nn_accel_timing_test",
                                               BOARD_TEST / "nn_accel.py")
nn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nn_module)


def main():
    events = []
    clock = {"value": 0.0}

    def mark(event):
        events.append(event)
        clock["value"] += 1.0

    class FakeBuffer:
        nbytes = 60000

        def __setitem__(self, _key, _value):
            mark("copy")

        @staticmethod
        def flush():
            mark("flush")

        @staticmethod
        def freebuffer():
            mark("free")

    class FakeSendChannel:
        _max_size = 100000

        @staticmethod
        def transfer(_buffer):
            mark("transfer")

        @property
        def idle(self):
            mark("idle")
            return True

        @staticmethod
        def stop():
            mark("stop")

    class FakeIP:
        @staticmethod
        def write(_offset, _value):
            mark("start_ip")

        @staticmethod
        def read(offset):
            mark("read_{}".format(offset))
            if offset == 0x00:
                return 0x02
            return 0

    class FakeDMA:
        sendchannel = FakeSendChannel()

    nn_module.allocate = lambda **_kwargs: (mark("allocate") or FakeBuffer())
    nn_module.perf_counter = lambda: clock["value"]
    nn_module.monotonic = lambda: 0.0
    nn_module.sleep = lambda _seconds: None

    accelerator = object.__new__(nn_module.ECGNNAccelerator)
    accelerator.ip = FakeIP()
    accelerator.dma = FakeDMA()
    output, seconds = accelerator.infer_pl(np.zeros((5000, 12), dtype=np.uint8))

    assert output.shape == (5,)
    assert events.index("allocate") < events.index("copy")
    assert events.index("flush") < events.index("transfer")
    assert events.index("transfer") < events.index("free")
    assert events.index("free") < len(events)
    assert seconds == len(events), (seconds, events)
    print("FPGA+DMA end-to-end timing boundary tests passed")


if __name__ == "__main__":
    main()
