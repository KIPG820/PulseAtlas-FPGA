"""PYNQ driver for the complete 1000 SPS ECG processing overlay.

Data path in PL:
ADS1298 -> quality -> classic filter -> wavelet -> {12-lead, QRS} ->
fair deep-buffered packetizer -> AXI DMA. Raw quality-tagged frames are packetized alongside
processed 12-lead frames and QRS events for verification.
"""

from pathlib import Path
import time

import numpy as np
from pynq import allocate


# ADS1298 acquisition register map
ADC_CTRL = 0x00
ADC_STATUS = 0x04
ADC_FRAME_SEQ = 0x08
ADC_FRAME_COUNT = 0x0C
ADC_DROP_COUNT = 0x10
ADC_FRAME_ERROR_COUNT = 0x14
ADC_CONFIG_ERROR_COUNT = 0x18
ADC_DEVICE_ID = 0x30
ADC_LEAD_OFF = 0x24
ADC_LEAD_OFF_P = 0x28
ADC_LEAD_OFF_N = 0x2C
ADC_LEAD_OFF_CAPABILITY = 0x34
LEAD_OFF_CAPABILITY_MAGIC = 0x4C4F4646
ADC_CAPTURE_LIMIT = 0x7C

PACKET_MAGIC = 0x45434731
TYPE_RAW = 1
TYPE_LEAD12 = 2
TYPE_QRS = 3


class PacketParser:
    """Decode mixed-length records emitted by the fair PL packetizer."""

    @staticmethod
    def signed_words(words):
        return np.asarray(words, dtype=np.uint32).view(np.int32).copy()

    @classmethod
    def parse(cls, words, include_raw=True):
        """Decode one DMA block.

        Live display/inference consumers do not use the raw eight-channel
        verification records.  Setting ``include_raw=False`` skips their
        dictionary and sample-array allocations while still advancing across
        every record, substantially shortening the interval before Python can
        service the next completed DMA block.
        """
        words = np.asarray(words, dtype=np.uint32).reshape(-1)
        records = []
        index = 0
        while index + 2 <= words.size:
            if int(words[index]) != PACKET_MAGIC:
                index += 1
                continue
            descriptor = int(words[index + 1])
            kind = (descriptor >> 24) & 0xFF
            length = descriptor & 0xFF
            if length < 6 or index + length > words.size:
                break
            record = words[index:index + length]
            if kind in (TYPE_RAW, TYPE_LEAD12):
                expected = 14 if kind == TYPE_RAW else 18
                if length == expected and (kind != TYPE_RAW or include_raw):
                    quality = int(record[4])
                    records.append({
                        "type": kind,
                        "frame": int(record[2]),
                        "ads_status": int(record[3]) & 0x00FF_FFFF,
                        "lead_off": quality & 0xFF,
                        "saturation": (quality >> 8) & 0xFF,
                        "flatline": (quality >> 16) & 0xFF,
                        "jump": (quality >> 24) & 0xFF,
                        "lead_invalid": int(record[5]) & 0xFFF,
                        "samples": cls.signed_words(record[6:]),
                    })
            elif kind == TYPE_QRS and length == 8:
                payload = int(record[6]) | (int(record[7]) << 32)
                records.append({
                    "type": kind,
                    "frame": payload & 0xFFFF_FFFF,
                    "rr_samples": (payload >> 32) & 0xFFFF,
                    "lead_mask": (payload >> 48) & 0xFF,
                    "flags": (payload >> 56) & 0xFF,
                })
            index += length
        return records


class ECGFullPipeline:
    """Configure, acquire, decode and verify all seven PL ECG functions."""

    LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6")

    def __init__(self, overlay):
        self.overlay = overlay
        self.adc = overlay.ecg_adc_0
        self.quality = overlay.ecg_quality_0
        self.filter = overlay.ecg_classic_filter_0
        self.wavelet = overlay.ecg_wavelet_0
        self.lead12 = overlay.ecg_lead12_0
        self.qrs = overlay.ecg_qrs_0
        self.packet = overlay.ecg_packet_0
        self.dma = overlay.ecg_dma_0
        self.sample_rate = 1000
        # D1..D4 soft-shrink thresholds.  Zero is the integration baseline;
        # set_wavelet_thresholds() can select real denoising before run().
        self.wavelet_thresholds = (0, 0, 0, 0)

    def set_wavelet_thresholds(self, d1, d2, d3, d4):
        """Set signed soft-shrink thresholds for wavelet detail bands.

        Values are PL ADC-code units.  The setting is applied by the next
        configure_processing() call, which monitor.run() performs before
        starting acquisition.  Start with small values and compare the same
        electrode placement against the all-zero baseline.
        """
        values = tuple(int(value) for value in (d1, d2, d3, d4))
        if any(value < 0 or value > 0x7FFFFFFF for value in values):
            raise ValueError("wavelet thresholds must be unsigned 31-bit values")
        self.wavelet_thresholds = values
        return values

    @staticmethod
    def _read(ip, offset):
        return int(ip.read(offset))

    def adc_status(self):
        status = self._read(self.adc, ADC_STATUS)
        return {
            "device_id": self._read(self.adc, ADC_DEVICE_ID) & 0xFF,
            "status": status,
            "enabled": bool(status & 0x01),
            "error_state": bool(status & 0x02),
            "config_error": bool(status & 0x04),
            "frame_error": bool(status & 0x08),
            "overflow": bool(status & 0x10),
            "running": bool(status & 0x20),
            "spi_busy": bool(status & 0x40),
            "id_ok": bool(status & 0x80),
            "frame_seq": self._read(self.adc, ADC_FRAME_SEQ),
            "frame_count": self._read(self.adc, ADC_FRAME_COUNT),
            "drop_count": self._read(self.adc, ADC_DROP_COUNT),
            "frame_errors": self._read(self.adc, ADC_FRAME_ERROR_COUNT),
            "config_errors": self._read(self.adc, ADC_CONFIG_ERROR_COUNT),
            "capture_limit": self._read(self.adc, ADC_CAPTURE_LIMIT),
            "lead_off_capability": self._read(self.adc, ADC_LEAD_OFF_CAPABILITY),
            "lead_off_config": tuple(self._read(self.adc, offset) & 0xFF
                                     for offset in (ADC_LEAD_OFF, ADC_LEAD_OFF_P,
                                                    ADC_LEAD_OFF_N)),
        }

    def lead_off_ready(self):
        """Return whether transient electrode detection is actively enabled."""
        return (self._read(self.adc, ADC_LEAD_OFF_CAPABILITY)
                == LEAD_OFF_CAPABILITY_MAGIC and
                tuple(self._read(self.adc, offset) & 0xFF for offset in
                      (ADC_LEAD_OFF, ADC_LEAD_OFF_P, ADC_LEAD_OFF_N))
                == (0x13, 0xFF, 0x03) and
                bool(self._read(self.adc, ADC_CTRL) & 0x10) and
                not bool(self._read(self.adc, ADC_CTRL) & 0x04))

    def pipeline_status(self):
        """Read counters from every integrated IP without changing state."""
        return {
            "quality_frame_count": self._read(self.quality, 0x20),
            "filter_id": hex(self._read(self.filter, 0x00)),
            "filter_version": hex(self._read(self.filter, 0x04)),
            "filter_ctrl": hex(self._read(self.filter, 0x08)),
            "filter_frame_count": self._read(self.filter, 0x18),
            "filter_saturation_count": self._read(self.filter, 0x60),
            "filter_latency_samples": self._read(self.filter, 0x64),
            "wavelet_id": hex(self._read(self.wavelet, 0x00)),
            "wavelet_version": hex(self._read(self.wavelet, 0x04)),
            "wavelet_frame_count": self._read(self.wavelet, 0x18),
            "wavelet_saturation_count": self._read(self.wavelet, 0x60),
            "wavelet_latency_samples": self._read(self.wavelet, 0x64),
            "wavelet_ctrl": hex(self._read(self.wavelet, 0x08)),
            "wavelet_config": hex(self._read(self.wavelet, 0x40)),
            "wavelet_thresholds": tuple(self._read(self.wavelet, offset)
                                         for offset in (0x44, 0x48, 0x4C, 0x50)),
            "lead12_frame_count": self._read(self.lead12, 0x20),
            "qrs_id": hex(self._read(self.qrs, 0x00)),
            "qrs_version": hex(self._read(self.qrs, 0x04)),
            "qrs_count": self._read(self.qrs, 0x18),
            "qrs_searchback_count": self._read(self.qrs, 0x64),
            "packet_block_fill": self._read(self.packet, 0x20) & 0xFF,
            "packet_raw_backpressure_events": self._read(self.packet, 0x24),
            "packet_lead_backpressure_events": self._read(self.packet, 0x28),
            "packet_qrs_backpressure_events": self._read(self.packet, 0x2C),
            "packet_record_count": self._read(self.packet, 0x30),
            "packet_block_count": self._read(self.packet, 0x34),
        }

    def configure_processing(self):
        """Explicitly configure every processing IP for 1000 SPS operation."""
        # IP2: pass records and enable all four quality checks.
        self.quality.write(0x00, 0x0F)

        # IP3: 0.5 Hz high-pass + 50/150 Hz biquad notches + 150 Hz low-pass.
        self.filter.write(0x08, 0x00)
        self.filter.write(0x40, 0x07)
        self.filter.write(0x44, 0x01)
        self.filter.write(0x48, 0x01)
        self.filter.write(0x54, 0x02)
        self.filter.write(0x60, 0x01)
        self.filter.write(0x08, 0x03)
        time.sleep(0.002)
        self.filter.write(0x08, 0x01)

        # IP4: execute the soft-shrinkage wavelet path. Zero thresholds make
        # this an exactly observable transform/reconstruction integration test;
        # set_wavelet_thresholds() selects actual coefficient suppression.
        self.wavelet.write(0x08, 0x00)
        self.wavelet.write(0x40, 0x03)
        for offset, threshold in zip((0x44, 0x48, 0x4C, 0x50), self.wavelet_thresholds):
            self.wavelet.write(offset, threshold & 0x7FFFFFFF)
        self.wavelet.write(0x60, 0x01)
        self.wavelet.write(0x08, 0x03)
        time.sleep(0.002)
        self.wavelet.write(0x08, 0x01)

        # IP5: derived 12-lead calculation.
        self.lead12.write(0x00, 0x01)

        # IP6: two-lead QRS detector, 1000 SPS coefficients and 2 s learning.
        self.qrs.write(0x08, 0x00)
        self.qrs.write(0x40, 0x07)
        self.qrs.write(0x44, 0x00)  # lead II (stream channel 0)
        self.qrs.write(0x48, 0x07)  # V1 (stream channel 7)
        self.qrs.write(0x4C, 0x02)  # OR fusion
        self.qrs.write(0x50, 25)
        self.qrs.write(0x54, 2000)
        self.qrs.write(0x60, 0x01)
        self.qrs.write(0x64, 0x01)
        self.qrs.write(0x08, 0x03)
        time.sleep(0.002)
        self.qrs.write(0x08, 0x01)

        # IP7 packetizer; bit 0 enables ingress. Its round-robin RTL ensures
        # raw, lead12 and QRS records all reach the shared DMA.
        self.packet.write(0x00, 0x01)

    def start_ads1298(self, internal_test=False, timeout=2.5, lead_off=False):
        """Start the ADS1298 at 1000 SPS after a DMA receive is armed."""
        if self._read(self.adc, ADC_LEAD_OFF_CAPABILITY) != LEAD_OFF_CAPABILITY_MAGIC:
            raise RuntimeError("当前 FPGA 固件不支持电极脱落检测，请更新 bit/hwh。")
        ctrl = 0x01 | 0x02 | 0x08 | 0x20  # enable + reinit + 1000 SPS + stream
        if internal_test:
            ctrl |= 0x04
        if lead_off:
            ctrl |= 0x10
        self.adc.write(ADC_CTRL, 0)
        time.sleep(0.01)
        for offset, value in ((ADC_LEAD_OFF, 0x13),
                              (ADC_LEAD_OFF_P, 0xFF),
                              (ADC_LEAD_OFF_N, 0x03)):
            self.adc.write(offset, value)
            if (self._read(self.adc, offset) & 0xFF) != value:
                raise RuntimeError("电极脱落检测寄存器回读失败：0x{:02X}".format(offset))
        self.adc.write(ADC_STATUS, 0x01)
        self.adc.write(ADC_CAPTURE_LIMIT, 0)
        self.adc.write(ADC_CTRL, ctrl)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.adc_status()
            if state["error_state"]:
                return False
            if state["running"]:
                return True
            time.sleep(0.002)
        return False

    def stop(self):
        self.adc.write(ADC_CTRL, 0)
        self.packet.write(0x00, 0)

    def hold_for_status(self, timeout=1.0):
        """Keep the ADC powered for low-rate ID polling, with no data stream."""
        self.packet.write(0x00, 0)
        self.adc.write(ADC_CTRL, 0x01)  # power/status enabled, RDATAC disabled
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            status = self.adc_status()
            if not status["running"]:
                return status
            time.sleep(0.002)
        return self.adc_status()

    @staticmethod
    def _words_received(channel, buffer):
        transferred = int(getattr(channel, "transferred", buffer.nbytes))
        return max(0, min(buffer.size, transferred // 4))

    @staticmethod
    def _frame_gaps(frames):
        frames = np.asarray(frames, dtype=np.uint32)
        if frames.size < 2:
            return 0
        return int(np.count_nonzero(np.diff(frames.astype(np.int64)) != 1))

    def capture(self, frame_count=4096, internal_test=False, timeout_s=None,
                dma_words=8192):
        """Capture all packet types until `frame_count` processed 12-lead frames.

        The raw stream reaches the packetizer earlier than the filtered/wavelet
        path. The stop condition is therefore based on processed lead12 frames,
        which proves that the full chain has produced the requested dataset.
        """
        if frame_count < 256:
            raise ValueError("frame_count must be at least 256")
        # 256 mixed records can require 4,608 words when every record is lead12.
        # 8,192 words leaves margin and cuts Python DMA re-arm frequency to ~8 Hz.
        if dma_words < 8192:
            raise ValueError("dma_words must be at least 8192 for 256-record packet blocks")
        if timeout_s is None:
            timeout_s = max(12.0, frame_count / self.sample_rate * 3.0 + 6.0)

        buffer = allocate(shape=(dma_words,), dtype=np.uint32)
        raw_records, lead_records, qrs_records = [], [], []
        try:
            self.configure_processing()
            self.dma.recvchannel.transfer(buffer)
            if not self.start_ads1298(internal_test=internal_test):
                raise RuntimeError("ADS1298 did not enter continuous conversion: {}".format(
                    self.adc_status()))
            deadline = time.monotonic() + timeout_s
            while len(lead_records) < frame_count:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for {} processed lead12 frames; got {}".format(
                            frame_count, len(lead_records)))
                self.dma.recvchannel.wait()
                count = self._words_received(self.dma.recvchannel, buffer)
                # Copy the completed DMA block, then immediately arm the next
                # simple-mode S2MM transfer before doing Python packet parsing.
                # Parsing creates a record list and per-record NumPy arrays;
                # doing that while DMA is idle lets the finite PL packet FIFOs
                # fill and eventually backpressure the ADC's two-frame FIFO.
                payload = np.asarray(buffer[:count]).copy()
                self.dma.recvchannel.transfer(buffer)
                records = PacketParser.parse(payload)
                for record in records:
                    if record["type"] == TYPE_RAW:
                        raw_records.append(record)
                    elif record["type"] == TYPE_LEAD12:
                        lead_records.append(record)
                    elif record["type"] == TYPE_QRS:
                        qrs_records.append(record)

            before_stop = self.adc_status()
            pipeline = self.pipeline_status()
            return self._result(raw_records, lead_records[:frame_count], qrs_records,
                                before_stop, pipeline)
        finally:
            try:
                self.stop()
            finally:
                # A transfer is now deliberately armed while the last block
                # is parsed. Stop it on every exit path before releasing its
                # buffer; the final partial block is not part of the capture.
                try:
                    self.dma.recvchannel.stop()
                except Exception:
                    pass
                buffer.freebuffer()

    def _result(self, raw_records, lead_records, qrs_records, ads_status, pipeline):
        raw = np.asarray([r["samples"] for r in raw_records], dtype=np.int32)
        leads = np.asarray([r["samples"] for r in lead_records], dtype=np.int32)
        raw_frames = np.asarray([r["frame"] for r in raw_records], dtype=np.uint32)
        lead_frames = np.asarray([r["frame"] for r in lead_records], dtype=np.uint32)
        quality = {
            key: np.asarray([r[key] for r in lead_records], dtype=np.uint32)
            for key in ("lead_off", "saturation", "flatline", "jump", "lead_invalid")
        }
        qrs = {
            "frame": np.asarray([r["frame"] for r in qrs_records], dtype=np.uint32),
            "rr_samples": np.asarray([r["rr_samples"] for r in qrs_records], dtype=np.uint16),
            "lead_mask": np.asarray([r["lead_mask"] for r in qrs_records], dtype=np.uint8),
            "flags": np.asarray([r["flags"] for r in qrs_records], dtype=np.uint8),
        }
        return {
            "raw": raw,
            "raw_frames": raw_frames,
            "leads": leads,
            "lead_frames": lead_frames,
            "quality": quality,
            "qrs": qrs,
            "ads": ads_status,
            "pipeline": pipeline,
            "raw_frame_sequence_gaps": self._frame_gaps(raw_frames),
            "lead_frame_sequence_gaps": self._frame_gaps(lead_frames),
        }

    @staticmethod
    def estimate_hr(qrs_events):
        rr = np.asarray(qrs_events["rr_samples"], dtype=np.float64)
        rr = rr[(rr >= 200) & (rr <= 3000)]
        if rr.size == 0:
            return float("nan")
        return float(60000.0 / np.median(rr))

    @staticmethod
    def verdict(result):
        """Return a concise pass/fail summary for a captured full-pipeline run."""
        ads = result["ads"]
        stage = result["pipeline"]
        errors = []
        if not ads["id_ok"]:
            errors.append("ADS1298 ID readback failed")
        if ads["drop_count"] or ads["frame_errors"] or ads["config_errors"]:
            errors.append("ADS acquisition reports drops or errors")
        if result["lead_frame_sequence_gaps"]:
            errors.append("processed lead12 stream contains sequence gaps")
        for key in ("packet_raw_backpressure_events",
                    "packet_lead_backpressure_events",
                    "packet_qrs_backpressure_events"):
            if stage[key]:
                errors.append("{}={}".format(key, stage[key]))
        if result["leads"].shape[0] == 0:
            errors.append("no processed lead12 records received")
        return {
            "pass": not errors,
            "errors": errors,
            "raw_records": int(result["raw"].shape[0]),
            "lead12_records": int(result["leads"].shape[0]),
            "qrs_events": int(result["qrs"]["frame"].size),
            "estimated_hr_bpm": ECGFullPipeline.estimate_hr(result["qrs"]),
        }
