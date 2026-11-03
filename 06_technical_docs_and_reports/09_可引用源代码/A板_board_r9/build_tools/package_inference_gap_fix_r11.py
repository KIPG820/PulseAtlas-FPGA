"""Package the A-board display release with lossless AI-window capture."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
BASE = ROOT / "artifacts" / "PulseAtlas_A_Board_R10_ScientificDisplay_20261026.zip"
OUTPUT = ROOT / "artifacts" / "PulseAtlas_A_Board_R11_InferenceGapFix_20261026.zip"


def main():
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    replacements = {
        "pulseatlas_r9_app.py": (BOARD / "pulseatlas_r9_app.py").read_bytes(),
        "run_pulseatlas_r9.py": (BOARD / "run_pulseatlas_r9.py").read_bytes(),
        # The R10 base archive predates the standalone PS vision worker even
        # though the current UI imports it unconditionally at startup.
        "ps_vision_runtime.py": (BOARD / "ps_vision_runtime.py").read_bytes(),
        "ecg_full_pipeline.py": (BOARD / "ecg_full_pipeline.py").read_bytes(),
        "ecg_live_monitor.py": (BOARD / "ecg_live_monitor.py").read_bytes(),
        "ecg_frontend.bit": (BOARD / "ecg_frontend.bit").read_bytes(),
        "ecg_frontend.hwh": (BOARD / "ecg_frontend.hwh").read_bytes(),
        "A_BOARD_R11_INFERENCE_GAP_FIX.txt": (
            "PulseAtlas A-board R11 inference gap fix\n"
            "Base: A-board R10 scientific display\n"
            "- Pause HDMI rendering and TFT transfers only while acquiring the strict 10 s AI window.\n"
            "- Keep the ECG DMA drain active and preserve real gap/quality rejection.\n"
            "- Resume both displays before the selected PS or FPGA three-run inference.\n"
            "- Apply the same display exclusion to the short lead-off contact probe.\n"
            "- Log ADC drop and packet backpressure deltas for every accepted window.\n"
            "- Skip unused RAW verification records in the live DMA parser.\n"
            "- Keep non-contact raw quality masks as diagnostics; frame gaps and LOFF remain blocking.\n"
            "- Extend the PL packet block from 256 to 1024 records and use a 32768-word DMA buffer.\n"
            "  This reduces simple-DMA re-arm boundaries by 4x while preserving every ECG frame.\n"
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
