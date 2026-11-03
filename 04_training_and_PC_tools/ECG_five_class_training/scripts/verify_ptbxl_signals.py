"""Verify selected PTB-XL records against the official SHA256SUMS.txt."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "nn" / "data" / "ptb-xl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "nn" / "data" / "cache" / "records_manifest.csv")
    parser.add_argument("--folds", default="1-9",
                        help="fold selection to verify; accepts comma-separated values/ranges")
    args = parser.parse_args()
    checksums_path = args.dataset_root / "SHA256SUMS.txt"
    if not checksums_path.is_file():
        raise FileNotFoundError(f"missing official checksum file: {checksums_path}")
    checksum_map = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            checksum_map[fields[1].lstrip("*").replace("\\", "/")] = fields[0].lower()

    selected_folds = set()
    for item in args.folds.split(","):
        parts = item.strip().split("-")
        selected_folds.update(range(int(parts[0]), int(parts[-1]) + 1))

    failures = []
    checked = 0
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not int(row["label_mask"]) or int(row["strat_fold"]) not in selected_folds:
                continue
            stem = row["filename_hr"].replace("\\", "/")
            for extension in (".hea", ".dat"):
                relative = stem + extension
                path = args.dataset_root / Path(relative)
                expected = checksum_map.get(relative)
                if expected is None:
                    failures.append((relative, "not listed in official SHA256SUMS.txt"))
                elif not path.is_file():
                    failures.append((relative, "missing"))
                else:
                    actual = sha256_file(path)
                    if actual != expected:
                        failures.append((relative, "SHA-256 mismatch"))
                    checked += 1
    print(f"checked={checked}, failures={len(failures)}, folds={sorted(selected_folds)}")
    if failures:
        for relative, reason in failures[:30]:
            print(f"FAIL {relative}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
