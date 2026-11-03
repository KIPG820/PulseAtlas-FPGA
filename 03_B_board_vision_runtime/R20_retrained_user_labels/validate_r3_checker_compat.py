"""Check the candidate ONNX through B R3's unchanged fixed-ROI checker."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\zsd88\Desktop\1\frames_review")
R3 = Path(r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ColorHybrid_HSV_R3_SourceAgeGateFix_20261014")
MODEL = ROOT / "tiny_patch_presence_batch10.onnx"
LABELS = ("RA", "LA", "RL", "LL", "V1", "V2", "V3", "V4", "V5", "V6")


def main():
    spec = importlib.util.spec_from_file_location("r3_patch_presence_checker", R3 / "patch_presence_checker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (DATA / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if r["source_video"] in (
        "20261018_144022_clothed_no_pads.avi", "20261018_144217_clothed_no_pads.avi"))
    image = cv2.imread(str(DATA / row["image_path"]), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    relative = Path(row["image_path"]).relative_to("images")
    label_path = DATA / "labels" / relative.with_suffix(".txt")
    points = {}
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        xc, yc = map(float, parts[1:3])
        points[LABELS[class_id]] = (xc * width, yc * height)
    for i, label in enumerate(LABELS):
        points.setdefault(label, (width * (0.16 + 0.075 * i), height * (0.22 + 0.12 * (i % 5))))

    checker = module.CalibratedPatchPresenceChecker(str(MODEL), threads=2)
    result = checker.predict(image, points)
    if result.get("status") != "ok" or result.get("inferred_points") != 10:
        raise RuntimeError("R3 checker failed candidate integration: " + repr(result))
    verification = {
        "status": "passed",
        "opencv_version": cv2.__version__,
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "input_frame_shape": list(image.shape),
        "fixed_batch_points": result["inferred_points"],
        "checker_status": result["status"],
        "inference_ms_local_pc_only": result["inference_ms"],
        "method": "unchanged R3 CalibratedPatchPresenceChecker loaded candidate ONNX and ran one full 10-point batch",
        "scope_note": "This validates file/interface compatibility on PC OpenCV 4.12; it is not a B-board deployment or accuracy test.",
    }
    check_path = ROOT / "r3_checker_compatibility.json"
    check_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = ROOT / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["existing_r3_checker_compatibility"] = verification
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
