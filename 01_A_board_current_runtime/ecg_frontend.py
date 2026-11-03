"""PYNQ-Z2 ECG frontend overlay loader."""

from pathlib import Path

from pynq import DefaultIP, Overlay
import pynq.lib.video  # Register HDMI and VDMA drivers before HWH parsing.
from pynq.lib.video.dma import AxiVDMA


class _RestartSafeMM2SChannel(AxiVDMA.MM2SChannel):
    """Output VDMA channel for an HWH without parser-visible interrupts."""

    def writeframe(self, frame):
        if not self.running:
            raise RuntimeError("HDMI VDMA channel is not running")

        # PYNQ's stock method waits on an Interrupt object. This output-only
        # design has no parser-visible MM2S interrupt, and waiting for the
        # stale completion flag after a restart is what blackened the second
        # frame. At the 2-12 Hz UI rate, the VDMA's four-frame ring is safely
        # ahead of the 60 Hz display scanout.
        status = self._mmio.read(0x04)
        if status & 0x8980:
            self._mmio.write(0x04, status & 0x8980)
        self._writeframe_internal(frame)


class _OutputOnlyVDMA(DefaultIP):
    def __init__(self, description):
        super().__init__(description)
        parameters = description["parameters"]
        self.framecount = int(parameters["C_NUM_FSTORES"])
        self.writechannel = _RestartSafeMM2SChannel(
            self, None, description["device"].default_memory)


class ECGFrontendOverlay(Overlay):
    """Load the integrated ADS1298, ECG processing, DMA and HDMI overlay."""

    def __init__(self, bitfile=None, **kwargs):
        if bitfile is None:
            bitfile = Path(__file__).with_suffix(".bit")
        super().__init__(str(bitfile), **kwargs)

        # ECGFrontendOverlay subclasses the generic Overlay, not PYNQ-Z2's
        # board-specific BaseOverlay. Publish the PYNQ Arduino hierarchy using
        # the same mb_info contract that BaseOverlay.ARDUINO provides.
        if self.is_loaded() and hasattr(self, "iop_arduino"):
            self.iop_arduino.mbtype = "Arduino"
            arduino_info = dict(self.iop_arduino.mb_info)
            # This compact BD uses a named slice on PS GPIO bit 2 rather than
            # the stock BaseOverlay reset name; PYNQ indexes it through
            # PL.gpio_dict when resetting the MicroBlaze.
            arduino_info["rst_name"] = "tft_iop_reset_slice"
            # The TFT firmware polls its mailbox and GT911; it does not use an
            # interrupt event, and this R9 overlay does not export a PYNQ
            # interrupt_pins entry for the IOP hierarchy.
            arduino_info.pop("intr_pin_name", None)
            arduino_info.pop("intr_ack_name", None)
            self.ARDUINO = arduino_info

        # PYNQ's stock HDMIWrapper binds the shared VDMA automatically only
        # when both HDMI input and output pipelines exist.  This reduced ECG
        # overlay intentionally removes HDMI input, so bind the retained MM2S
        # VDMA to the output pipeline explicitly.
        if self.is_loaded() and hasattr(self, "video"):
            video = self.video
            if hasattr(video, "hdmi_out"):
                vdma_description = next(
                    (desc for name, desc in self.ip_dict.items()
                     if name.endswith("axi_vdma")), None)
                if vdma_description is None:
                    raise RuntimeError("axi_vdma is missing from overlay.ip_dict")
                video.hdmi_out._vdma = _OutputOnlyVDMA(vdma_description)
