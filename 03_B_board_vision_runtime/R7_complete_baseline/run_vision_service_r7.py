#!/usr/bin/env python3
"""Live-camera service for the B-only R7 prepared INT8 IP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=ROOT,
                        help="R7 installation with model, graph and calibration assets")
    parser.add_argument("--bitstream", type=Path,
                        default=ROOT / "vision_accel" / "vision_bonly_fused_silu_v14.bit")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview-fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    args = parser.parse_args(argv)
    if not args.runtime_root.is_dir():
        parser.error("runtime root not found: {}".format(args.runtime_root))
    if not args.bitstream.is_file() or not args.bitstream.with_suffix(".hwh").is_file():
        parser.error("matching R7 .bit/.hwh files are required")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(args.runtime_root))
    from ps_vision_runtime_prepared import PSVisionRuntime
    from vision_service_server import VisionServiceServer
    from pynq import Overlay

    overlay = Overlay(str(args.bitstream), download=False)
    if "vision_conv_0" not in getattr(overlay, "ip_dict", {}):
        raise RuntimeError("R7 handoff has no vision_conv_0; refusing CPU fallback")
    runtime = PSVisionRuntime(
        model_path=None,
        camera_index=args.camera,
        max_inference_fps=60.0,
        torso_calibration_path=args.runtime_root / "vision_guide" / "calibration" / "torso_lead_calibration.json",
        chest_calibration_path=args.runtime_root / "vision_guide" / "calibration" / "chest_lead_calibration.json",
        overlay=overlay,
        linecache_int8_pl=True,
        runtime_root=args.runtime_root,
    )
    service = VisionServiceServer(runtime, args.host, args.port,
                                  args.preview_fps, args.jpeg_quality)
    print("PulseAtlas B Vision R7 prepared INT8 | camera={} | tcp={}:{}".format(
        args.camera, args.host, args.port), flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
