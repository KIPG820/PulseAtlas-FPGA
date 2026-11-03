"""Board-free check for SW0 display selection and HDMI lifecycle."""

from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pynq = types.ModuleType("pynq")
pynq.allocate = lambda *args, **kwargs: None
lib = types.ModuleType("pynq.lib")
video = types.ModuleType("pynq.lib.video")
video.VideoMode = lambda width, height, depth: (width, height, depth)
dma = types.ModuleType("pynq.lib.video.dma")
dma.AxiVDMA = type("AxiVDMA", (), {"MM2SChannel": type("MM2SChannel", (), {})})
sys.modules.update({"pynq": pynq, "pynq.lib": lib, "pynq.lib.video": video,
                    "pynq.lib.video.dma": dma})
monitor = types.ModuleType("ecg_live_monitor")
monitor.ECGLiveLeadMonitor = type("ECGLiveLeadMonitor", (), {})
pipeline = types.ModuleType("ecg_full_pipeline")
pipeline.TYPE_RAW = 1
pipeline.TYPE_LEAD12 = 2
pipeline.TYPE_QRS = 3
sys.modules.update({"ecg_live_monitor": monitor, "ecg_full_pipeline": pipeline})

from pulseatlas_r9_app import PulseAtlasR9App


class SwitchChannel:
    value = 0

    def setlength(self, length):
        assert length == 2

    def setdirection(self, direction):
        assert direction == "in"

    def read(self):
        return self.value


def main():
    calls = []
    channel = SwitchChannel()

    class Frame:
        def fill(self, value):
            self.value = value

    app = object.__new__(PulseAtlasR9App)
    app.overlay = types.SimpleNamespace(sws_gpio=types.SimpleNamespace(channel1=channel))
    app.hdmi = types.SimpleNamespace(
        newframe=lambda: calls.append("newframe") or Frame(),
        writeframe=lambda frame: calls.append(("writeframe", frame.value)),
        close=lambda: calls.append("close"),
        configure=lambda mode: calls.append(("configure", mode)),
        start=lambda: calls.append("start"),
    )
    app.width, app.height = 1280, 720
    app._hdmi_active = True
    app._hdmi_switch_channel = None
    app._hdmi_switch_candidate = None
    app._hdmi_switch_candidate_since = 0.0
    app._hdmi_switch_warned = False
    app._hdmi_switch_logged = False

    assert app._requested_hdmi_mode() is True  # Debounce the first reading.
    time.sleep(0.06)
    assert app._requested_hdmi_mode() is False
    assert app._set_hdmi_mode(False)
    assert calls == ["newframe", ("writeframe", 0)]
    channel.value = 1
    assert app._requested_hdmi_mode() is False
    time.sleep(0.06)
    assert app._requested_hdmi_mode() is True
    assert app._set_hdmi_mode(True)
    assert calls == ["newframe", ("writeframe", 0)]
    print("SW0 display mode contract passed")


if __name__ == "__main__":
    main()
