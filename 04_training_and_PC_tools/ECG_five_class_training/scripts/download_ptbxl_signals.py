"""Parallel, resumable downloader for the labeled 500 Hz PTB-XL records."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# PhysioNet publishes the same versioned files in its anonymous-read S3 mirror.
BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"


def is_complete(path: Path, relative: str, expected_bytes: int) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if expected_bytes:
        return size == expected_bytes
    if size <= 0 or not relative.endswith(".hea"):
        return size > 0
    try:
        lines = path.read_text(encoding="ascii", errors="replace").splitlines()
        fields = lines[0].split()
        return (len(fields) >= 4 and fields[1:4] == ["12", "500", "5000"]
                and len(lines) >= 13)
    except (OSError, IndexError):
        return False


def download_one(args: tuple[str, Path, int]) -> tuple[str, bool, str]:
    relative, dataset_root, expected_bytes = args
    url = BASE_URL + "/" + relative.replace("\\", "/")
    destination = dataset_root / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(destination, relative, expected_bytes):
        return relative, True, "cached"
    temporary = destination.with_suffix(destination.suffix + ".part")
    command = [
        "curl.exe", "--fail", "--location", "--silent", "--show-error",
        "--retry", "4", "--retry-delay", "1", "--connect-timeout", "20",
        "--max-time", "180", "--output", str(temporary), url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=190)
    except Exception as exc:  # noqa: BLE001
        return relative, False, str(exc)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        return relative, False, result.stderr.strip()[-400:]
    size = temporary.stat().st_size if temporary.exists() else -1
    if size <= 0 or (expected_bytes and size != expected_bytes):
        temporary.unlink(missing_ok=True)
        return relative, False, f"size mismatch: {size}, expected {expected_bytes}"
    if relative.endswith(".hea") and not is_complete(temporary, relative, 0):
        temporary.unlink(missing_ok=True)
        return relative, False, "WFDB header is incomplete or not 12x5000 at 500 Hz"
    temporary.replace(destination)
    return relative, True, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "nn" / "data" / "ptb-xl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "nn" / "data" / "cache" / "records_manifest.csv")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--folds", type=str, default="1-10",
                        help="comma/range selection, e.g. 1-8,9; defaults to all folds")
    parser.add_argument("--max-records-per-fold", type=int, default=0,
                        help="optional deterministic cap; 0 means all labeled records")
    args = parser.parse_args()
    if args.workers < 1 or args.max_records_per_fold < 0:
        parser.error("workers must be positive and max-records-per-fold nonnegative")

    selected_folds: set[int] = set()
    for item in args.folds.split(","):
        parts = item.strip().split("-")
        if len(parts) == 1:
            selected_folds.add(int(parts[0]))
        elif len(parts) == 2:
            selected_folds.update(range(int(parts[0]), int(parts[1]) + 1))
        else:
            parser.error(f"invalid fold selector: {item}")
    if not selected_folds or not selected_folds.issubset(set(range(1, 11))):
        parser.error("folds must be selected from 1..10")

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_fold: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["label_mask"]) and int(row["strat_fold"]) in selected_folds:
            by_fold.setdefault(int(row["strat_fold"]), []).append(row)

    paths: list[tuple[str, Path, int]] = []
    for fold in sorted(selected_folds):
        fold_rows = by_fold.get(fold, [])
        if args.max_records_per_fold:
            fold_rows = fold_rows[:args.max_records_per_fold]
        for row in fold_rows:
            stem = row["filename_hr"]
            paths.append((stem + ".hea", args.dataset_root, 0))
            paths.append((stem + ".dat", args.dataset_root, 120000))
    total_records = len(paths) // 2
    print(f"Downloading {total_records} records ({len(paths)} files), folds={sorted(selected_folds)}, workers={args.workers}", flush=True)

    failures: list[tuple[str, str]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item) for item in paths]
        for future in concurrent.futures.as_completed(futures):
            relative, ok, message = future.result()
            completed += 1
            if not ok:
                failures.append((relative, message))
            if completed % 100 == 0 or completed == len(paths):
                print(f"files {completed}/{len(paths)}; failures={len(failures)}", flush=True)

    if failures:
        report = args.dataset_root / "download_failures.csv"
        with report.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("path", "error"))
            writer.writerows(failures)
        raise SystemExit(f"{len(failures)} downloads failed; see {report}")
    print(f"Downloaded/cached {total_records} labeled PTB-XL records under {args.dataset_root}")


if __name__ == "__main__":
    main()
