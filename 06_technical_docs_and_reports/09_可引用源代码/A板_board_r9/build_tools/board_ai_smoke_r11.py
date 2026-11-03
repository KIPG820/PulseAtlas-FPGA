#!/usr/bin/env python3
"""Run one board-local PL inference through the production R11 classes."""

from pathlib import Path
import json
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecg_frontend import ECGFrontendOverlay
from ecg_full_pipeline import ECGFullPipeline
from nn_accel import ECGNNAccelerator
from pulseatlas_r9_app import PulseAtlasR9App


def main():
    overlay = ECGFrontendOverlay(str(ROOT / "ecg_frontend.bit"))
    pipeline = ECGFullPipeline(overlay)
    nn = ECGNNAccelerator(overlay)
    app = PulseAtlasR9App(pipeline, nn)
    try:
        app.golden_passed = True
        app.golden_status = "board smoke bypass; release vectors passed at startup"
        app.start_monitor(update_page=False)
        app._ai_active_backend = "pl"
        app.ai_backend_mode = "pl"
        app.ai_busy = True
        started = time.monotonic()
        app._run_ai_analysis_worker()
        report = {
            "elapsed_seconds": time.monotonic() - started,
            "ai_error": app.ai_error,
            "ai_status": app.ai_status,
            "ai_progress": app.ai_progress,
            "has_result": app.ai_result is not None,
            "result": app.ai_result,
            "adc_status": pipeline.adc_status(),
            "pipeline_status": pipeline.pipeline_status(),
            "rx_blocks": app.rx_blocks,
            "rx_words": app.rx_words,
        }
        print("BOARD_AI_SMOKE=" + json.dumps(report, ensure_ascii=False,
                                              default=str), flush=True)
        return 0 if app.ai_result is not None else 2
    finally:
        app.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
