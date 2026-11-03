from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts" / "PulseAtlas_A_Board_R14_UI_TFT_Acceptance_20261027_v4.zip"
STAGE = ROOT / "tmp" / "goal_r14_acceptance"
OUTPUT = ROOT / "artifacts" / "PulseAtlas_A_Board_R15_TFT_Filter_20261027.zip"
NOTE_NAME = "A_BOARD_R15_TFT_FILTER_SWITCH.txt"

NOTE = """PulseAtlas A board R15 TFT filter control

The TFT real-time monitoring page now has a clear, two-state classic-filter switch.
Touching the switch updates the visible state and invokes the existing FPGA filter
control path without restarting ECG acquisition. The adjacent gain controls have
matching touch hitboxes. No FPGA bitstream or HDMI renderer change is included.
""".encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    replacements = {
        "pulseatlas_tft_render.py": (STAGE / "pulseatlas_tft_render.py").read_bytes(),
        "pulseatlas_r9_app.py": (STAGE / "pulseatlas_r9_app.py").read_bytes(),
        NOTE_NAME: NOTE,
    }
    with ZipFile(BASE, "r") as source:
        payload = {
            info.filename: source.read(info.filename)
            for info in source.infolist()
            if not info.is_dir() and info.filename != "SHA256SUMS.txt"
        }
    payload.update(replacements)
    payload["SHA256SUMS.txt"] = ("\n".join(
        "{}  {}".format(sha256(data), name)
        for name, data in sorted(payload.items())
    ) + "\n").encode("utf-8")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for name, data in sorted(payload.items()):
            info = ZipInfo(name)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 << 16)
            archive.writestr(info, data)
    temporary.replace(OUTPUT)

    with ZipFile(OUTPUT, "r") as archive:
        assert archive.testzip() is None
        checks = {}
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            checks[name] = digest
        assert set(checks) == set(payload) - {"SHA256SUMS.txt"}
        for name, digest in checks.items():
            assert sha256(archive.read(name)) == digest, name
    print("Created {}".format(OUTPUT))
    print("Files: {}; SHA256: {}".format(len(payload), sha256(OUTPUT.read_bytes())))


if __name__ == "__main__":
    main()
