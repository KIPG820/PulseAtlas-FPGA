"""Package the R10 vision-cadence fix on top of the verified R9 archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261006_pose_manualcal_r9.zip"
DESTINATION = ROOT / "artifacts" / "PulseAtlas_R9_LVGL_HDMI_UI_20261006_pose_livepreview_r10.zip"
BOARD = ROOT / "ui" / "pulseatlas" / "board_r9"
VISION = ROOT / "ui" / "pulseatlas" / "vision_guide"


def _sources():
    files = {
        "README.md": BOARD / "README_PulseAtlas_R9.md",
        "run_pulseatlas_r9.py": BOARD / "run_pulseatlas_r9.py",
        "pulseatlas_r9_app.py": BOARD / "pulseatlas_r9_app.py",
        "pulseatlas_board_render.py": BOARD / "pulseatlas_board_render.py",
        "ps_vision_runtime.py": BOARD / "ps_vision_runtime.py",
        "build_tools/test_ps_vision_runtime.py": BOARD / "build_tools" / "test_ps_vision_runtime.py",
        "vision_guide/README.md": VISION / "README.md",
        "fonts/PulseAtlasCJK.ttf": BOARD / "fonts" / "PulseAtlasCJK.ttf",
    }
    return {name: path.read_bytes() for name, path in files.items()}


def main():
    if not BASE.is_file():
        raise FileNotFoundError("verified R9 base archive missing: {}".format(BASE))
    if DESTINATION.exists():
        raise FileExistsError("refusing to overwrite existing archive: {}".format(DESTINATION))

    with zipfile.ZipFile(BASE, "r") as base:
        contents = {
            info.filename: (base.read(info.filename), info.compress_type)
            for info in base.infolist()
            if not info.is_dir() and info.filename != "SHA256SUMS.txt"
        }
    for name, data in _sources().items():
        contents[name] = (data, zipfile.ZIP_DEFLATED)

    checksums = [
        "{}  {}".format(hashlib.sha256(contents[name][0]).hexdigest(), name)
        for name in sorted(contents)
    ]
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
    for name in ("ps_vision_runtime.py", "fonts/PulseAtlasCJK.ttf"):
        print("SHA256 {} {}".format(hashlib.sha256(contents[name][0]).hexdigest(), name))


if __name__ == "__main__":
    main()
