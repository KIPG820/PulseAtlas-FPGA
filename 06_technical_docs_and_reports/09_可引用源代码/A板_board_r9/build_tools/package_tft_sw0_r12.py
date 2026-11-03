"""Package the SW0-selectable, faster landscape TFT build on the R11 release."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[4]
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
BUILD = ROOT / "bld" / "ecg_nn_r9_tft_lean"
BASE = ROOT / "artifacts" / "PulseAtlas_R9_HDMI_TFT_Landscape_20261009_r11_fix9.zip"
DESTINATION = ROOT / "artifacts" / "PulseAtlas_R9_HDMI_TFT_SW0_20261010_r12.zip"


def main():
    if DESTINATION.exists():
        raise FileExistsError("refusing to overwrite {}".format(DESTINATION))
    replacements = {
        "README.md": BOARD / "README_PulseAtlas_R9.md",
        "run_pulseatlas_r9.py": BOARD / "run_pulseatlas_r9.py",
        "pulseatlas_r9_app.py": BOARD / "pulseatlas_r9_app.py",
        "pulseatlas_tft_backend.py": BOARD / "pulseatlas_tft_backend.py",
        "tools/tft_pulseatlas_iop.c": ROOT / "tools" / "tft_pulseatlas_iop.c",
        "tools/tft_pulseatlas_iop.bin": (
            ROOT / "bld" / "ecg_nn_r9_tft" / "tft_pulseatlas_iop_r12.bin"),
        "ecg_frontend.bit": BUILD / "ecg_nn_r9_tft_lean.bit",
        "ecg_frontend.hwh": (
            BUILD / "ecg_nn_r9.gen" / "sources_1" / "bd" / "base" /
            "hw_handoff" / "base.hwh"),
        "build_tools/test_sw0_display_mode.py": (
            BOARD / "build_tools" / "test_sw0_display_mode.py"),
    }
    with zipfile.ZipFile(BASE) as source:
        contents = {
            info.filename: (source.read(info.filename), info.compress_type)
            for info in source.infolist()
            if not info.is_dir() and info.filename != "SHA256SUMS.txt"
        }
    for name, path in replacements.items():
        contents[name] = (path.read_bytes(), zipfile.ZIP_DEFLATED)
    if b'sws_gpio' not in contents["ecg_frontend.hwh"][0]:
        raise RuntimeError("hardware handoff is missing the SW0 GPIO")
    if len(contents["tools/tft_pulseatlas_iop.bin"][0]) < 7000:
        raise RuntimeError("unexpected TFT IOP binary; optimized firmware missing")
    sums = "\n".join("{}  {}".format(hashlib.sha256(data).hexdigest(), name)
                     for name, (data, _) in sorted(contents.items())) + "\n"
    contents["SHA256SUMS.txt"] = (sums.encode(), zipfile.ZIP_DEFLATED)
    temporary = DESTINATION.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, (data, compression) in sorted(contents.items()):
                archive.writestr(name, data, compress_type=compression,
                                 compresslevel=6)
        temporary.replace(DESTINATION)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(DESTINATION) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError("archive CRC failed: {}".format(bad))
    print("Created {} ({} files, {:.2f} MiB)".format(
        DESTINATION, len(contents), DESTINATION.stat().st_size / 1048576))
    for name in ("ecg_frontend.bit", "ecg_frontend.hwh",
                 "tools/tft_pulseatlas_iop.bin"):
        print("{} {}".format(hashlib.sha256(contents[name][0]).hexdigest(), name))


if __name__ == "__main__":
    main()
