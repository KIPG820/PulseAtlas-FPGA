"""PYNQ-Z2 R8: fixed-vector validation and optional 12-lead NN comparison."""

from pathlib import Path
from time import monotonic, perf_counter, sleep
import json

import numpy as np
from pynq import allocate

from ps_numpy_reference import NumpyIntegerECG

ROOT = Path(__file__).resolve().parent
LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
ADC_LSB_MV = 2.4 * 1000.0 / (6.0 * (1 << 23))


class ECGNNAccelerator:
    """MM2S -> full INT8 ResNet IP, with PS NumPy integer baseline."""

    def __init__(self, overlay):
        self.overlay = overlay
        self.ip = overlay.nn_accel_0
        self.dma = overlay.nn_dma_0
        self.contract_path = ROOT / "model_ptq_int8_hw.json"
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.cpu = NumpyIntegerECG(self.contract_path)
        self.output_scale = float(self.contract["quantized_layers"][-1]["output_scale"])
        self.output_zp = int(self.contract["quantized_layers"][-1]["output_zero_point"])
        self.input_scale = float(self.contract["graph_ops"][0]["input_scale"])
        self.input_zp = int(self.contract["graph_ops"][0]["input_zero_point"])
        self.norm = json.loads((ROOT / "normalization.json").read_text(encoding="utf-8"))

    def infer_pl(self, time_major_codes, timeout_s=20.0):
        codes = np.asarray(time_major_codes, dtype=np.uint8)
        if codes.shape != (5000, 12):
            raise ValueError("expected exactly 5000 samples x 12 leads, uint8")
        # Match the PS timing boundary at the preprocessed [5000, 12] input.
        # FPGA+DMA includes its input packing, PYNQ allocation/copy/flush,
        # transfer, inference, output reads, DMA-idle wait, and buffer release.
        start = perf_counter()
        buffer = None
        output = None
        try:
            payload = np.ascontiguousarray(codes).reshape(-1).view("<u4")
            buffer = allocate(shape=(15000,), dtype=np.uint32)
            buffer[:] = payload
            buffer.flush()
            if buffer.nbytes > self.dma.sendchannel._max_size:
                raise RuntimeError("NN DMA transfer limit is smaller than 60000 bytes; wrong bit/hwh")
            self.ip.write(0x00, 0x01)
            self.dma.sendchannel.transfer(buffer)
            deadline = monotonic() + timeout_s
            while monotonic() < deadline:
                control = self.ip.read(0x00)
                if control & 0x02:
                    break
                sleep(0.001)
            else:
                self.dma.sendchannel.stop()
                raise TimeoutError("NN IP did not assert ap_done")
            status = self.ip.read(0x10)
            if status:
                raise RuntimeError("NN input AXI stream error bits: 0x{:08x}".format(status))
            output = np.array([self.ip.read(offset) & 255
                               for offset in (0x20, 0x30, 0x40, 0x50, 0x60)],
                              dtype=np.uint8)
            # ap_done follows complete input and inference. DMA idle should
            # already be true, but wait briefly before releasing its buffer.
            deadline = monotonic() + 2.0
            while not self.dma.sendchannel.idle and monotonic() < deadline:
                sleep(0.001)
            if not self.dma.sendchannel.idle:
                raise RuntimeError("NN DMA did not return to idle")
        finally:
            if buffer is not None:
                buffer.freebuffer()
        return output, perf_counter() - start

    def infer_ps(self, time_major_codes):
        start = perf_counter()
        codes = self.cpu.run_codes(time_major_codes)
        return codes, perf_counter() - start

    def logits_and_probabilities(self, output_codes):
        logits = (np.asarray(output_codes, dtype=np.float32) - self.output_zp) * self.output_scale
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return dict(zip(LABELS, probabilities.astype(float).tolist()))

    def check_golden_vectors(self, indices=(0, 333, 1024, 2000)):
        rows = []
        vector_dir = ROOT / "golden_vectors"
        for index in indices:
            prefix = "fold9_{:04d}".format(index)
            data = np.fromfile(vector_dir / (prefix + "_input_time_major_u8.bin"),
                               dtype=np.uint8).reshape(5000, 12)
            expected = np.fromfile(vector_dir / (prefix + "_expected_output_u8.bin"),
                                   dtype=np.uint8)
            cpu, cpu_s = self.infer_ps(data)
            pl, pl_s = self.infer_pl(data)
            row = {"record": prefix, "expected": expected.tolist(),
                   "ps_codes": cpu.tolist(), "pl_codes": pl.tolist(),
                   "ps_seconds": cpu_s, "pl_seconds": pl_s,
                   "pass": bool(np.array_equal(cpu, expected) and
                                np.array_equal(pl, expected))}
            rows.append(row)
            print(row)
            if not row["pass"]:
                raise AssertionError("PL or PS differs from golden vector: " + prefix)
        return rows

    def encode_physical_leads(self, samples_1000sps):
        samples = np.asarray(samples_1000sps, dtype=np.int32)
        if samples.shape != (10000, 12):
            raise ValueError("expected 10 seconds of uninterrupted 1000-SPS 12-lead data")
        # R7's classic filter already limits the band to ~150 Hz, below the
        # 500-SPS Nyquist frequency. Select every other sample, then apply the
        # frozen training-fold normalization and exported input quantizer.
        millivolts = samples[::2].astype(np.float32) * ADC_LSB_MV
        mean = np.asarray(self.norm["mean_mV"], dtype=np.float32)
        std = np.asarray(self.norm["std_mV"], dtype=np.float32)
        normalized = np.clip((millivolts - mean) / std, -8.0, 8.0)
        return np.clip(np.rint(normalized / self.input_scale + self.input_zp),
                       0, 255).astype(np.uint8)

    def capture_and_compare(self, pipeline):
        result = pipeline.capture(frame_count=10000, internal_test=False,
                                  timeout_s=45.0)
        verdict = pipeline.verdict(result)
        print("R7 acquisition:", verdict)
        if not verdict["pass"] or result["lead_frame_sequence_gaps"]:
            raise RuntimeError("capture contains errors/gaps; inference comparison aborted")
        codes = self.encode_physical_leads(result["leads"])
        cpu, cpu_s = self.infer_ps(codes)
        pl, pl_s = self.infer_pl(codes)
        answer = {"ps_codes": cpu.tolist(), "pl_codes": pl.tolist(),
                  "same_codes": bool(np.array_equal(cpu, pl)),
                  "ps_seconds": cpu_s, "pl_seconds": pl_s,
                  "pl_probabilities": self.logits_and_probabilities(pl)}
        print("12-lead model comparison:", answer)
        return result, answer
