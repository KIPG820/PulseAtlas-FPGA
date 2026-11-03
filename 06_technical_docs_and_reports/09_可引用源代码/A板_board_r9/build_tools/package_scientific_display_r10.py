"""Package the reviewed scientific ECG displays on the verified A-board R9."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
BASE = ROOT / "artifacts" / "PulseAtlas_A_Board_R9_DMAStartup_20261025.zip"
OUTPUT = ROOT / "artifacts" / "PulseAtlas_A_Board_R10_ScientificDisplay_20261026.zip"


def main():
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    replacements = {
        "pulseatlas_r9_app.py": (BOARD / "pulseatlas_r9_app.py").read_bytes(),
        "pulseatlas_lvgl_backend.py": (BOARD / "pulseatlas_lvgl_backend.py").read_bytes(),
        "pulseatlas_board_render.py": (BOARD / "pulseatlas_board_render.py").read_bytes(),
        "pulseatlas_tft_render.py": (BOARD / "pulseatlas_tft_render.py").read_bytes(),
        "ecg_display_metrics.py": (BOARD / "ecg_display_metrics.py").read_bytes(),
        "libpulseatlas_lvgl.so": (BOARD / "libpulseatlas_lvgl.so").read_bytes(),
        "build_tools/test_ecg_display_metrics.py": (
            BOARD / "build_tools" / "test_ecg_display_metrics.py").read_bytes(),
        "A_BOARD_R10_SCIENTIFIC_DISPLAY.txt": (
            "PulseAtlas A-board R10 scientific display\n"
            "Base: A-board R9 DMA startup release\n"
            "- ECG paper grid uses the real 1000 SPS time window.\n"
            "- 12-lead view uses 4 x 3 synchronized 2.5 s segments and a 10 s Lead-II rhythm strip.\n"
            "- Beat-to-beat heart-rate trend uses QRS frame timestamps; long gaps remain empty.\n"
            "- Spectrogram uses Hann 512, hop 64, 0-100 Hz, fixed -120..-20 dBFS/Hz.\n"
            "- HDMI and TFT use the same numerical preparation.\n"
            "- Amplitude remains in ADC codes until the analog chain is calibrated.\n"
            "- VCG is intentionally not included in this release.\n"
        ).encode("utf-8"),
    }
    with zipfile.ZipFile(BASE) as source:
        payload = {info.filename: source.read(info.filename)
                   for info in source.infolist()
                   if not info.is_dir() and info.filename != "SHA256SUMS.txt"}
    payload.update(replacements)
    payload["SHA256SUMS.txt"] = "".join(
        "{}  {}\n".format(sha256(data).hexdigest(), name)
        for name, data in sorted(payload.items())
    ).encode("ascii")
    temporary = OUTPUT.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as target:
            for name, data in sorted(payload.items()):
                target.writestr(name, data)
        temporary.replace(OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        for name, data in replacements.items():
            assert check.read(name) == data
    print("{} ({} bytes, {} files)".format(OUTPUT, OUTPUT.stat().st_size,
                                           len(payload)))


if __name__ == "__main__":
    main()
