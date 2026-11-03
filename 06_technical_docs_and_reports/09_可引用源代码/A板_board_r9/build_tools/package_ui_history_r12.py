"""Package the A-board UI/history update on top of the verified R11 release."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
BASE = ROOT / "artifacts" / "PulseAtlas_A_Board_R11_InferenceGapFix_20261026.zip"
OUTPUT = ROOT / "artifacts" / "PulseAtlas_A_Board_R12_HistoryUI_20261026.zip"
RELEASE_NOTE = BOARD / "A_BOARD_R12_UI_HISTORY.md"


def main():
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    replacements = {
        "pulseatlas_r9_app.py": (BOARD / "pulseatlas_r9_app.py").read_bytes(),
        "pulseatlas_history.py": (BOARD / "pulseatlas_history.py").read_bytes(),
        "pulseatlas_lvgl_backend.py": (BOARD / "pulseatlas_lvgl_backend.py").read_bytes(),
        "pulseatlas_board_render.py": (BOARD / "pulseatlas_board_render.py").read_bytes(),
        "pulseatlas_tft_render.py": (BOARD / "pulseatlas_tft_render.py").read_bytes(),
        "libpulseatlas_lvgl.so": (BOARD / "libpulseatlas_lvgl.so").read_bytes(),
        "A_BOARD_R12_UI_HISTORY.md": RELEASE_NOTE.read_bytes(),
    }
    with zipfile.ZipFile(BASE) as source:
        payload = {item.filename: source.read(item.filename)
                   for item in source.infolist()
                   if not item.is_dir() and item.filename != "SHA256SUMS.txt"}
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
        temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        for name, data in replacements.items():
            assert check.read(name) == data
    print("{} ({} bytes, {} files)".format(
        OUTPUT, OUTPUT.stat().st_size, len(payload)))


if __name__ == "__main__":
    main()
