"""Package the landscape TFT mirror alongside the verified R10 R9 release."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261006_pose_livepreview_r10.zip"
DESTINATION = ROOT / "artifacts" / "PulseAtlas_R9_HDMI_TFT_Landscape_20261009_r11_fix9.zip"
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
BUILD = ROOT / "bld" / "ecg_nn_r9_tft_lean"
IOP_FIRMWARE = ROOT / "bld" / "ecg_nn_r9_tft" / "tft_pulseatlas_iop.bin"
HWH = BUILD / "ecg_nn_r9.gen" / "sources_1" / "bd" / "base" / "hw_handoff" / "base.hwh"


def _sources():
    files = {
        "README.md": BOARD / "README_PulseAtlas_R9.md",
        "run_pulseatlas_r9.py": BOARD / "run_pulseatlas_r9.py",
        "ecg_frontend.py": BOARD / "ecg_frontend.py",
        "pulseatlas_r9_app.py": BOARD / "pulseatlas_r9_app.py",
        "pulseatlas_tft_render.py": BOARD / "pulseatlas_tft_render.py",
        "pulseatlas_tft_backend.py": BOARD / "pulseatlas_tft_backend.py",
        "build_tools/test_tft_ui.py": BOARD / "build_tools" / "test_tft_ui.py",
        "build_tools/test_button_navigation.py": BOARD / "build_tools" / "test_button_navigation.py",
        "build_tools/test_touch_routes.py": BOARD / "build_tools" / "test_touch_routes.py",
        "build_tools/test_tft_mailbox.py": BOARD / "build_tools" / "test_tft_mailbox.py",
        "tools/tft_pulseatlas_iop.c": ROOT / "tools" / "tft_pulseatlas_iop.c",
        "tools/xio_switch.c": (
            ROOT / "PYNQ-3.1.1" / "boards" / "Pynq-Z2" / "ecg_nn_r9" /
            "ip_repo_video" / "io_switch_1.1" / "drivers" /
            "io_switch_v1_0" / "src" / "xio_switch.c"
        ),
        "tools/tft_pulseatlas_iop.bin": IOP_FIRMWARE,
        "ecg_frontend.bit": BUILD / "ecg_nn_r9_tft_lean.bit",
        "ecg_frontend.hwh": HWH,
    }
    return {name: path.read_bytes() for name, path in files.items()}


def main():
    if not BASE.is_file():
        raise FileNotFoundError("verified R10 base archive missing: {}".format(BASE))
    if DESTINATION.exists():
        raise FileExistsError("refusing to overwrite existing archive: {}".format(DESTINATION))
    additions = _sources()

    with zipfile.ZipFile(BASE, "r") as base:
        contents = {
            info.filename: (base.read(info.filename), info.compress_type)
            for info in base.infolist()
            if not info.is_dir() and info.filename != "SHA256SUMS.txt"
        }
    for name, data in additions.items():
        contents[name] = (data, zipfile.ZIP_DEFLATED)

    contents["SHA256SUMS.txt"] = (
        ("\n".join("{}  {}".format(hashlib.sha256(data).hexdigest(), name)
                    for name, (data, _) in sorted(contents.items())) + "\n").encode("utf-8"),
        zipfile.ZIP_DEFLATED,
    )

    temporary = DESTINATION.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, (data, compression) in sorted(contents.items()):
                archive.writestr(name, data, compress_type=compression, compresslevel=6)
        temporary.replace(DESTINATION)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(DESTINATION, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError("archive CRC check failed for {}".format(bad_file))
    print("Created {}".format(DESTINATION))
    print("Files: {}".format(len(contents)))
    print("Size: {:.2f} MiB".format(DESTINATION.stat().st_size / (1024 * 1024)))
    for name in ("ecg_frontend.bit", "ecg_frontend.hwh", "tools/tft_pulseatlas_iop.bin"):
        print("SHA256 {} {}".format(hashlib.sha256(contents[name][0]).hexdigest(), name))


if __name__ == "__main__":
    main()
