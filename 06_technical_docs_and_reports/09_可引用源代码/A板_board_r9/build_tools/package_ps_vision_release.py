"""Build a new R9 deployment archive with calibrated PS pose guidance."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261004_psvision_cv454fix_r4.zip"
DESTINATION = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261006_pose_manualcal_r9.zip"
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
VISION = ROOT / "ui" / "pulseatlas" / "vision_guide"


def _sources():
    files = {
        "README.md": BOARD / "README_PulseAtlas_R9.md",
        "pulseatlas_r9_app.py": BOARD / "pulseatlas_r9_app.py",
        "run_pulseatlas_r9.py": BOARD / "run_pulseatlas_r9.py",
        "pulseatlas_board_render.py": BOARD / "pulseatlas_board_render.py",
        "ps_vision_runtime.py": BOARD / "ps_vision_runtime.py",
        "build_tools/test_button_navigation.py": BOARD / "build_tools" / "test_button_navigation.py",
        "build_tools/test_lead_off_contract.py": BOARD / "build_tools" / "test_lead_off_contract.py",
        "build_tools/test_nn_timing.py": BOARD / "build_tools" / "test_nn_timing.py",
        "build_tools/test_ps_vision_runtime.py": BOARD / "build_tools" / "test_ps_vision_runtime.py",
        "build_tools/test_ps_vision_model.py": BOARD / "build_tools" / "test_ps_vision_model.py",
        "build_tools/export_psvision_cv454_pose_model.py": BOARD / "build_tools" / "export_psvision_cv454_pose_model.py",
        "fonts/PulseAtlasCJK.ttf": BOARD / "fonts" / "PulseAtlasCJK.ttf",
        "vision_guide/README.md": VISION / "README.md",
        "vision_guide/electrode_placement.py": VISION / "electrode_placement.py",
        "vision_guide/ps_onnx_pose.py": VISION / "ps_onnx_pose.py",
        "vision_guide/test_ps_onnx_pose.py": VISION / "test_ps_onnx_pose.py",
        "vision_guide/models/model_manifest.json": VISION / "models" / "model_manifest.json",
        "vision_guide/models/yolo11n-pose-cv454.onnx": VISION / "models" / "yolo11n-pose-cv454.onnx",
        "vision_guide/calibration/torso_lead_calibration.json": VISION / "calibration" / "torso_lead_calibration.json",
        "vision_guide/calibration/chest_lead_calibration.json": VISION / "calibration" / "chest_lead_calibration.json",
    }
    return {name: path.read_bytes() for name, path in files.items()}


def main():
    if not BASE.is_file():
        raise FileNotFoundError("base R9 release archive missing: {}".format(BASE))
    if DESTINATION.exists():
        raise FileExistsError("refusing to overwrite existing archive: {}".format(DESTINATION))

    replacements = _sources()
    with zipfile.ZipFile(BASE, "r") as base:
        contents = {
            info.filename: (base.read(info.filename), info.compress_type)
            for info in base.infolist()
            if not info.is_dir() and info.filename != "SHA256SUMS.txt"
        }
    for obsolete in (
        "build_tools/export_psvision_cv454_model.py",
        "vision_guide/upper_body_segmentation.py",
        "vision_guide/ps_onnx_segmenter.py",
        "vision_guide/test_ps_onnx_segmenter.py",
        "vision_guide/models/yolo26n-seg.onnx",
    ):
        contents.pop(obsolete, None)
    for name, data in replacements.items():
        contents[name] = (data, zipfile.ZIP_DEFLATED)

    checksums = []
    for name in sorted(contents):
        checksums.append("{}  {}".format(hashlib.sha256(contents[name][0]).hexdigest(), name))
    contents["SHA256SUMS.txt"] = (
        ("\n".join(checksums) + "\n").encode("utf-8"), zipfile.ZIP_DEFLATED
    )

    temporary = DESTINATION.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name in sorted(contents):
                data, compression = contents[name]
                archive.writestr(name, data, compress_type=compression, compresslevel=6)
        temporary.replace(DESTINATION)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("Created {}".format(DESTINATION))
    print("Files: {}".format(len(contents)))
    print("Size: {:.2f} MiB".format(DESTINATION.stat().st_size / (1024 * 1024)))
    print("Pose model SHA256: {}".format(
        hashlib.sha256(replacements["vision_guide/models/yolo11n-pose-cv454.onnx"]).hexdigest()
    ))


if __name__ == "__main__":
    main()
