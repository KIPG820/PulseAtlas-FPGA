"""Build portable complete-project and minimal board-test R9QF archives."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "PYNQ-3.1.1/boards/Pynq-Z2/ecg_nn_r9qf"
OUT = ROOT / "artifacts"
SKIP_PARTS = {
    "__pycache__", ".pytest_cache", "ecg_nn_r9qf.cache", "ecg_nn_r9qf.gen",
    "ecg_nn_r9qf.hw", ".Xil",
}
SKIP_SUFFIXES = {".dcp", ".pyc", ".jou", ".log", ".stdout"}
SKIP_NAMES = {"ecg_nn_r9qf_refined.bit"}


def files(base: Path):
    for path in sorted(base.rglob("*")):
        if path.is_file() and not (set(path.relative_to(base).parts) & SKIP_PARTS):
            if path.suffix.lower() not in SKIP_SUFFIXES and path.name not in SKIP_NAMES:
                yield path


def pack(name: str, base: Path, prefix: str) -> Path:
    OUT.mkdir(exist_ok=True)
    destination = OUT / name
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in files(base):
            archive.write(path, str(Path(prefix) / path.relative_to(base)).replace("\\", "/"))
    with ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"CRC error in {destination}: {bad}")
        entries = set(archive.namelist())
        if prefix == "ecg_nn_r9qf":
            required = {f"{prefix}/ecg_nn_r9qf.bit", f"{prefix}/ecg_nn_r9qf.hwh",
                        f"{prefix}/ecg_nn_r9qf.xpr", f"{prefix}/rebuild_r9qf.tcl",
                        f"{prefix}/board_test/R9QF_NN_TEST.ipynb"}
        else:
            required = {f"{prefix}/ecg_frontend.bit", f"{prefix}/ecg_frontend.hwh",
                        f"{prefix}/R9QF_NN_TEST.ipynb"}
        if not required <= entries:
            raise RuntimeError(f"Missing archive files: {required - entries}")
        print(f"{destination} ({destination.stat().st_size} bytes, {len(entries)} files, CRC OK)")
    return destination


def main() -> None:
    manifest = json.loads((PROJECT / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_bit_and_board_bit_match"]
    assert manifest["source_hwh_and_board_hwh_match"]
    pack("ECG_NN_R9QF_BOARD_TEST_20260930.zip", PROJECT / "board_test", "board_test")
    pack("ECG_NN_R9QF_20260930.zip", PROJECT, "ecg_nn_r9qf")


if __name__ == "__main__":
    main()

