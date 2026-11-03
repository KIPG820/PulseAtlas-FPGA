"""Concurrent-acquisition, restart-clean HDMI Lead-II monitor.

The DMA worker owns RX completion and immediately re-arms the next transfer.
The Jupyter/UI thread only snapshots the accumulated Lead-II history and draws
HDMI frames.  Slow PIL/HDMI work therefore cannot leave S2MM unarmed.
"""

import threading
import time
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pynq import allocate
from pynq.lib.video import VideoMode

from ecg_full_pipeline import PacketParser, TYPE_LEAD12, TYPE_QRS


class ECGLiveLeadMonitor:
    """Show a red Lead-II trace and QRS-derived heart rate on HDMI.

    Only one background thread touches the RX DMA channel.  For every completed
    buffer it copies the received words, re-arms the DMA immediately, then
    parses packets.  HDMI rendering is intentionally outside that path.
    """

    def __init__(self, pipeline, seconds=6.0, refresh_seconds=5.0,
                 display_half_range=100000, width=1280, height=720,
                 beep_callback=None):
        self.pipeline = pipeline
        self.overlay = pipeline.overlay
        self.sample_rate = pipeline.sample_rate
        self.seconds = float(seconds)
        self.refresh_seconds = float(refresh_seconds)
        self.display_half_range = float(display_half_range)
        self.width, self.height = int(width), int(height)
        self.beep_callback = beep_callback
        self.history = deque(maxlen=max(100, round(self.seconds * self.sample_rate)))
        self.heart_rate = float("nan")
        self.last_qrs_frame = None
        self.rx_blocks = 0
        self.rx_words = 0
        self._data_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None
        self._worker_error = None
        self._closed = False
        # R11 packetizer emits 1024 mixed records per TLAST (at most 18,432
        # words).  Keep ample headroom so simple DMA can receive the entire
        # block before Python performs one re-arm every roughly 0.5 seconds.
        self.buffer = allocate(shape=(32768,), dtype=np.uint32)

        previous = getattr(self.overlay, "_ecg_active_display", None)
        if previous is not None:
            previous.close()
        self.hdmi = self.overlay.video.hdmi_out
        if getattr(self.hdmi, "_vdma", None) is None:
            raise RuntimeError("Load the overlay with ECGFrontendOverlay before creating HDMI monitor")
        # Do not merely reset MM2S here.  A previous monitor left the HDMI
        # frontend running, which allowed old scanout descriptors to survive
        # into the next invocation and tear the image.  Close stops both the
        # VDMA frame cache and the HDMI frontend before a clean reconfigure.
        try:
            self.hdmi.close()
        except Exception:
            # A never-started output can reject close on some PYNQ versions.
            # Resetting its MM2S channel is the safe fallback.
            try:
                self.hdmi._vdma.writechannel.reset()
            except Exception:
                pass
        self.hdmi.configure(VideoMode(self.width, self.height, 24))
        self.hdmi.start()
        self.overlay._ecg_active_display = self
        self._font = ImageFont.load_default()
        self._title_font = self._load_font(22)
        self._hr_font = self._load_font(46)

    @staticmethod
    def _load_font(size):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        return ImageFont.load_default()

    def _consume(self, records):
        lead_values = []
        qrs_events = []
        for record in records:
            if record["type"] == TYPE_LEAD12:
                # PL lead order: I, II, III, aVR, aVL, aVF, V1..V6.
                lead_values.append(int(record["samples"][1]))
            elif record["type"] == TYPE_QRS:
                qrs_events.append((int(record["frame"]), int(record["rr_samples"])))

        beep_count = 0
        with self._data_lock:
            self.history.extend(lead_values)
            for frame, rr in qrs_events:
                if 200 <= rr <= 3000:
                    self.heart_rate = 60000.0 / rr
                    beep_count += 1
                self.last_qrs_frame = frame

        if self.beep_callback is not None:
            for _ in range(beep_count):
                try:
                    self.beep_callback()
                except Exception:
                    break

    def _acquire_loop(self):
        """Drain RX DMA continuously; never draw or access HDMI here."""
        channel = self.pipeline.dma.recvchannel
        try:
            while not self._stop_event.is_set():
                # PYNQ's channel.wait() is an uninterruptible busy loop. Poll
                # until idle so the close path can stop this worker before
                # releasing the DMA buffer.
                if not channel.running:
                    break
                while not self._stop_event.is_set() and not channel.idle:
                    if channel.error:
                        raise RuntimeError("ECG RX DMA error while waiting")
                    time.sleep(0.001)
                if self._stop_event.is_set():
                    break
                # At this point the transfer is idle; wait() returns
                # immediately and performs PYNQ's cache invalidate/cleanup.
                channel.wait()
                words = self.pipeline._words_received(channel, self.buffer)
                payload = np.asarray(self.buffer[:words]).copy()
                # Re-arm before packet parsing or history updates.  The copy is
                # only 32 KiB at most and prevents a buffer reuse race.
                channel.transfer(self.buffer)
                with self._data_lock:
                    self.rx_blocks += 1
                    self.rx_words += words
                # Raw frames are present for engineering verification but the
                # live UI and AI consume only processed 12-lead/QRS records.
                # Avoid allocating and decoding the unused half of the stream
                # in this latency-sensitive DMA worker.
                self._consume(PacketParser.parse(payload, include_raw=False))
        except Exception as exc:
            if not self._stop_event.is_set():
                self._worker_error = exc

    def _render(self, source):
        with self._data_lock:
            signal = np.asarray(self.history, dtype=np.float64)
            heart_rate = self.heart_rate
            rx_blocks = self.rx_blocks

        image = Image.new("RGB", (self.width, self.height), (10, 13, 19))
        draw = ImageDraw.Draw(image)
        status = self.pipeline.adc_status()
        ok = status["id_ok"] and status["running"] and not status["config_error"]
        label = "RUNNING" if ok else "CHECK HARDWARE"
        color = (70, 230, 145) if ok else (255, 95, 95)
        draw.text((32, 24), "LEAD II  |  PL FILTER + WAVELET + QRS", fill=(235, 240, 245), font=self._title_font)
        draw.text((32, 58), "{}   1000 SPS   ID=0x{:02X}   frames={}   drops={}   DMA blocks={}".format(
            label, status["device_id"], status["frame_count"], status["drop_count"], rx_blocks),
            fill=color, font=self._font)
        hr_text = "--" if not np.isfinite(heart_rate) else "{:.0f}".format(heart_rate)
        draw.text((self.width - 290, 20), hr_text, fill=(255, 75, 75), font=self._hr_font)
        draw.text((self.width - 286, 74), "BPM", fill=(255, 165, 165), font=self._title_font)

        left, right, top, bottom = 70, self.width - 35, 122, self.height - 50
        for x in range(left, right + 1, 50):
            draw.line((x, top, x, bottom), fill=(25, 45, 47))
        for y in range(top, bottom + 1, 50):
            draw.line((left, y, right, y), fill=(25, 45, 47))
        mid = (top + bottom) / 2.0
        draw.line((left, int(mid), right, int(mid)), fill=(45, 80, 80), width=1)

        if signal.size >= 2:
            center = np.median(signal)
            scale = max(self.display_half_range, 1.0)
            x = np.linspace(left, right, signal.size)
            y = mid - np.clip((signal - center) / scale, -1.0, 1.0) * (bottom - top) * 0.43
            draw.line([(int(px), int(py)) for px, py in zip(x, y)], fill=(245, 55, 55), width=2)
            draw.text((left, bottom + 12), "window {:.1f}s | display +/-{:.0f} ADC codes".format(
                self.seconds, self.display_half_range), fill=(155, 175, 185), font=self._font)
        else:
            draw.text((left + 20, int(mid)), "Waiting for processed Lead-II packets", fill=(255, 190, 80), font=self._title_font)
        return image

    def _show(self, source):
        image = self._render(source)
        frame = self.hdmi.newframe()
        frame[:] = np.asarray(image)[:, :, ::-1]
        self.hdmi.writeframe(frame)

    def _prepare_for_run(self):
        """Put HDMI, RX DMA and the monitor state into a fresh-run state."""
        if self._closed:
            # The same monitor object may be run again after close().  Recreate
            # the buffer/event state and reopen the video frontend explicitly.
            self.buffer = allocate(shape=(32768,), dtype=np.uint32)
            self.history = deque(maxlen=max(100, round(self.seconds * self.sample_rate)))
            self.heart_rate = float("nan")
            self.last_qrs_frame = None
            self.rx_blocks = 0
            self.rx_words = 0
            self._stop_event = threading.Event()
            self._worker = None
            self._worker_error = None
            self._closed = False
            try:
                self.hdmi.close()
            except Exception:
                try:
                    self.hdmi._vdma.writechannel.reset()
                except Exception:
                    pass
            self.hdmi.configure(VideoMode(self.width, self.height, 24))
            self.hdmi.start()
            self.overlay._ecg_active_display = self

        # close() stops S2MM.  Conversely every run must start it again.  A
        # stop/start pair also clears a stale in-flight transfer left by an
        # interrupted previous run, avoiding "DMA channel not idle".
        channel = self.pipeline.dma.recvchannel
        try:
            channel.stop()
        except Exception:
            pass
        channel.start()

    def run(self, internal_test=False):
        """Run until the Jupyter interrupt button is pressed, then clean up."""
        source = "INTERNAL TEST" if internal_test else "ELECTRODES"
        final_status = None
        try:
            self._prepare_for_run()
            self.pipeline.configure_processing()
            self.pipeline.dma.recvchannel.transfer(self.buffer)
            if not self.pipeline.start_ads1298(internal_test=internal_test):
                raise RuntimeError("ADS1298 initialization failed: {}".format(self.pipeline.adc_status()))
            self._worker = threading.Thread(
                target=self._acquire_loop, name="ecg-dma-drain", daemon=True)
            self._worker.start()
            next_draw = 0.0
            while True:
                if self._worker_error is not None:
                    raise RuntimeError("ECG DMA worker failed: {!r}".format(self._worker_error))
                now = time.monotonic()
                if now >= next_draw:
                    self._show(source)
                    next_draw = now + self.refresh_seconds
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("ECG monitor stopped")
        finally:
            self.close()
            final_status = self.pipeline.adc_status()
        return final_status

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        try:
            # Stop S2MM first. This wakes the worker and guarantees that the
            # buffer is no longer owned by DMA before it is freed.
            channel = self.pipeline.dma.recvchannel
            stop = getattr(channel, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
            self.pipeline.stop()
        finally:
            if self._worker is not None:
                self._worker.join(timeout=2.0)
                if self._worker.is_alive():
                    raise RuntimeError("ECG DMA worker did not stop; restart the Jupyter kernel before reusing the overlay")
            try:
                # VideoOut.close() stops MM2S, frees its frame ring and stops
                # the HDMI frontend.  This makes the next Run start from a
                # clean scanout state instead of reusing an old descriptor.
                self.hdmi.close()
            except Exception:
                try:
                    self.hdmi._vdma.writechannel.reset()
                except Exception:
                    pass
            finally:
                self.buffer.freebuffer()
                if getattr(self.overlay, "_ecg_active_display", None) is self:
                    delattr(self.overlay, "_ecg_active_display")

