#!/usr/bin/env python3
"""Launch PulseAtlas on PYNQ-Z2 HDMI/TFT without Jupyter controls."""

from pathlib import Path
import json
import signal
import sys
import threading
import time

from ecg_frontend import ECGFrontendOverlay
from ecg_full_pipeline import ECGFullPipeline
from nn_accel import ECGNNAccelerator
from pulseatlas_r9_app import PulseAtlasR9App


ROOT = Path(__file__).resolve().parent
shutdown_requested = threading.Event()


def repair_pynq_global_state_cache():
    """Remove only an unreadable PYNQ PL cache left by an interrupted boot."""
    from pynq.pl_server.global_state import GlobalState, _get_state_path

    state_path = Path(_get_state_path())
    try:
        GlobalState.parse_obj(json.loads(state_path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
        state_path.unlink(missing_ok=True)
        print(
            "Removed invalid PYNQ PL state cache before overlay load: {} ({})".format(
                state_path, exc
            ),
            flush=True,
        )


def request_shutdown(_signum, _frame):
    shutdown_requested.set()


def main():
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    repair_pynq_global_state_cache()
    print("PulseAtlas A R14 UI/history/TFT acceptance 2026-10-27 (audio deferred)",
          flush=True)
    print("Loading the shared R9 ECG/NN/Arduino-IOP overlay...")
    overlay = ECGFrontendOverlay(str(ROOT / "ecg_frontend.bit"))
    pipeline = ECGFullPipeline(overlay)
    nn = ECGNNAccelerator(overlay)
    app = PulseAtlasR9App(pipeline, nn)
    try:
        app.start(run_golden_check=True)
        print("PulseAtlas TFT active; SW0=0 TFT only, SW0=1 TFT + HDMI.")
        print("Keyboard fallback: 1 monitor, 2 leads, 3 AI, 4 spectrum, V vision, H home, Space start/stop, A analyze, Q quit.")
        print("ECG capture remains stopped until you select Monitor > Start.")
        while not shutdown_requested.wait(0.5):
            if app._closed:
                break
            if app._worker_error is not None and app.monitor_running:
                print("ECG acquisition stopped after DMA error: {!r}".format(app._worker_error))
                try:
                    app.stop_monitor()
                except Exception as exc:
                    print("Could not stop capture cleanly: {}".format(exc))
    finally:
        try:
            app.shutdown()
        except Exception as exc:
            print("PulseAtlas shutdown reported: {}".format(exc))
    print("PulseAtlas stopped.")


if __name__ == "__main__":
    main()
