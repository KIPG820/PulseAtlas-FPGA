"""Measure the current board UI, camera, inference and TFT paths together.

Run from an isolated copy of an R12 release; PULSEATLAS_HDMI_MODE=off/on
selects the mode without depending on physical SW0. ECG acquisition is left
stopped so the display/vision pipeline can be profiled first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_frontend import ECGFrontendOverlay
from ecg_full_pipeline import ECGFullPipeline
from nn_accel import ECGNNAccelerator
from pulseatlas_r9_app import PulseAtlasR9App


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= args.warmup or args.warmup < 0:
        parser.error("seconds must exceed nonnegative warmup")
    root = Path(__file__).resolve().parents[1]
    overlay = ECGFrontendOverlay(str(root / "ecg_frontend.bit"))
    pipeline = ECGFullPipeline(overlay)
    nn = ECGNNAccelerator(overlay)
    app = PulseAtlasR9App(pipeline, nn)
    samples = []
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        app.start(run_golden_check=False)
        app._set_page("vision")
        deadline = started + args.seconds
        while time.monotonic() < deadline:
            time.sleep(1.0)
            snap = app._vision_runtime.snapshot()
            sample = {
                "t": round(time.monotonic() - started, 3),
                "camera_frames": int(snap.get("vision_camera_frames") or 0),
                "model_frames": int(snap.get("vision_frames") or 0),
                "preview_frames": int(snap.get("vision_preview_frames") or 0),
                "model_ms": snap.get("vision_inference_ms"),
                "camera_fps": snap.get("vision_camera_fps"),
                "model_fps": snap.get("vision_fps"),
                "preview_fps": snap.get("vision_preview_fps"),
                "result_age_ms": snap.get("vision_result_age_ms"),
                "status": snap.get("vision_result_status"),
                "error": snap.get("vision_error"),
            }
            samples.append(sample)
            if len(samples) == 1 or len(samples) % 5 == 0:
                print("PIPELINE_SAMPLE " + json.dumps(sample, ensure_ascii=False), flush=True)
        valid = [s for s in samples if s["t"] >= args.warmup]
        if len(valid) < 2:
            raise RuntimeError("insufficient post-warmup samples")
        first, last = valid[0], valid[-1]
        elapsed = last["t"] - first["t"]
        rates = {}
        for key in ("camera_frames", "model_frames", "preview_frames"):
            rates[key] = round((last[key] - first[key]) / elapsed, 4)
        result = {
            "mode": os.getenv("PULSEATLAS_HDMI_MODE", "switch"),
            "elapsed_post_warmup_s": round(elapsed, 3),
            "rates": rates,
            "process_cpu_seconds": round(time.process_time() - cpu_started, 3),
            "process_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "last": last,
        }
        print("PIPELINE_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if last["error"] is None else 2
    finally:
        app.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())


