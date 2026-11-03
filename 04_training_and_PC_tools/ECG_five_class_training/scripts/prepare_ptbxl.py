"""Download/audit PTB-XL and build filtered, fold-isolated NumPy caches."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nn"))
from preprocess import (LEAD_ORDER, canonicalize_lead_names, filter_record,
                        load_contract)  # noqa: E402

DB = "ptb-xl/1.0.3"
LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv")


def _parse_codes(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must decode to a dictionary")
    return parsed


def map_labels(codes: dict, statements: pd.DataFrame) -> np.ndarray:
    y = np.zeros(len(LABELS), dtype=np.uint8)
    for code in codes:
        if code not in statements.index:
            continue
        row = statements.loc[code]
        if not bool(row.get("diagnostic", 0)):
            continue
        cls = row.get("diagnostic_class")
        if cls in LABELS:
            y[LABELS.index(cls)] = 1
    return y


def fetch_metadata(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in METADATA_FILES
               if not (dataset_root / name).is_file()]
    if missing:
        try:
            import wfdb
        except ImportError as exc:
            raise RuntimeError("Install WFDB in the zsd environment first.") from exc
        wfdb.dl_files(DB, str(dataset_root), missing, keep_subdirs=False)


def audit_metadata(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    db_path = dataset_root / "ptbxl_database.csv"
    stmt_path = dataset_root / "scp_statements.csv"
    if not db_path.is_file() or not stmt_path.is_file():
        raise FileNotFoundError(
            f"expected {db_path.name} and {stmt_path.name} under {dataset_root}"
        )
    records = pd.read_csv(db_path, index_col="ecg_id")
    statements = pd.read_csv(stmt_path, index_col=0)
    required = {"patient_id", "strat_fold", "scp_codes", "filename_hr"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"PTB-XL metadata missing columns: {sorted(missing)}")
    if "diagnostic" not in statements or "diagnostic_class" not in statements:
        raise ValueError("scp_statements.csv lacks diagnostic columns")

    records["labels"] = [
        map_labels(_parse_codes(codes), statements) for codes in records["scp_codes"]
    ]
    records["label_mask"] = [int(y.sum() > 0) for y in records["labels"]]
    records["filename_hr"] = records["filename_hr"].astype(str)
    records["strat_fold"] = records["strat_fold"].astype(int)
    if not records["strat_fold"].between(1, 10).all():
        raise ValueError("strat_fold must be in the range 1..10")
    patient_folds = records.groupby("patient_id")["strat_fold"].nunique()
    if (patient_folds > 1).any():
        examples = patient_folds[patient_folds > 1].index[:10].tolist()
        raise ValueError(f"patient leakage across official folds: {examples}")

    report = {
        "dataset": "PTB-XL",
        "version": "1.0.3",
        "record_count": int(len(records)),
        "unique_patient_count": int(records["patient_id"].nunique()),
        "records_with_supported_labels": int(records["label_mask"].sum()),
        "records_without_supported_labels": int((records["label_mask"] == 0).sum()),
        "folds": {},
        "train_label_positive_counts": {},
        "missing_metadata": {},
    }
    for fold in range(1, 11):
        subset = records[records.strat_fold == fold]
        report["folds"][str(fold)] = {
            "records": int(len(subset)),
            "patients": int(subset["patient_id"].nunique()),
            "labeled_records": int(subset["label_mask"].sum()),
        }
        if fold <= 8 and len(subset):
            counts = np.stack(subset["labels"].to_numpy()).sum(axis=0)
            for i, label in enumerate(LABELS):
                report["train_label_positive_counts"][label] = (
                    report["train_label_positive_counts"].get(label, 0)
                    + int(counts[i])
                )
    for field in ("patient_id", "strat_fold", "scp_codes", "filename_hr"):
        report["missing_metadata"][field] = int(records[field].isna().sum())
    return records, statements, report


def download_records(dataset_root: Path, records: pd.DataFrame,
                     folds: list[int]) -> None:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Install WFDB in the zsd environment first.") from exc
    files = []
    # Do not download signals with no supervised label in this five-label task.
    selected = records.loc[
        records["label_mask"].eq(1) & records["strat_fold"].isin(folds),
        "filename_hr",
    ].drop_duplicates()
    for stem in selected:
        files.extend((f"{stem}.hea", f"{stem}.dat"))
    wfdb.dl_files(DB, str(dataset_root), files, keep_subdirs=True)


def build_cache(dataset_root: Path, cache_dir: Path, records: pd.DataFrame,
                contract: dict, folds: list[int]) -> dict:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("Install WFDB in the zsd environment first.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {}
    for fold in folds:
        subset = records[(records.strat_fold == fold) & (records.label_mask == 1)]
        if subset.empty:
            continue
        x_path = cache_dir / f"fold_{fold}_x.npy"
        y_path = cache_dir / f"fold_{fold}_y.npy"
        ids_path = cache_dir / f"fold_{fold}_ids.npy"
        x_mem = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float32, shape=(len(subset), 12, 5000)
        )
        y_mem = np.lib.format.open_memmap(
            y_path, mode="w+", dtype=np.uint8, shape=(len(subset), len(LABELS))
        )
        ids = np.empty(len(subset), dtype=np.int32)
        for pos, (ecg_id, row) in enumerate(tqdm(
            subset.iterrows(), total=len(subset), desc=f"filter/cache fold {fold}"
        )):
            signal, fields = wfdb.rdsamp(str(dataset_root / row["filename_hr"]))
            if int(round(float(fields["fs"]))) != 500 or signal.shape != (5000, 12):
                raise ValueError(
                    f"{ecg_id}: expected 500 Hz and (5000,12), "
                    f"got fs={fields['fs']}, shape={signal.shape}"
                )
            names = canonicalize_lead_names(fields["sig_name"])
            if names != LEAD_ORDER:
                raise ValueError(f"{ecg_id}: unexpected lead order {names}")
            filtered = filter_record(signal, contract)
            if not np.isfinite(filtered).all():
                raise ValueError(f"{ecg_id}: invalid filtered signal")
            x_mem[pos] = filtered.T
            y_mem[pos] = row["labels"]
            ids[pos] = int(ecg_id)
        x_mem.flush()
        y_mem.flush()
        np.save(ids_path, ids, allow_pickle=False)
        cached[str(fold)] = int(len(subset))

    # Statistics are computed only from folds 1-8 and streamed from disk.
    count = 0
    sums = np.zeros(12, dtype=np.float64)
    sums_sq = np.zeros(12, dtype=np.float64)
    for fold in range(1, 9):
        path = cache_dir / f"fold_{fold}_x.npy"
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        for start in range(0, arr.shape[0], 32):
            chunk = np.asarray(arr[start:start + 32], dtype=np.float64)
            sums += chunk.sum(axis=(0, 2))
            sums_sq += np.square(chunk).sum(axis=(0, 2))
            count += chunk.shape[0] * chunk.shape[2]
    if count == 0:
        raise RuntimeError("no labeled training records were cached")
    mean = sums / count
    std = np.sqrt(np.maximum(sums_sq / count - np.square(mean), 1e-12))
    normalization = {
        "source_folds": list(range(1, 9)),
        "sample_count_per_lead": int(count),
        "lead_order": list(LEAD_ORDER),
        "mean_mV": mean.tolist(),
        "std_mV": std.tolist(),
    }
    (cache_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )
    return {"cached_records_by_fold": cached, "normalization": normalization}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "nn" / "data" / "ptb-xl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "nn" / "data" / "cache")
    parser.add_argument("--metadata-only", action="store_true",
                        help="download metadata and audit, but do not fetch ECG records")
    parser.add_argument("--audit-only", action="store_true",
                        help="audit metadata without reading signal files")
    parser.add_argument("--skip-download", action="store_true",
                        help="use signal files already present under dataset-root")
    parser.add_argument("--include-test", action="store_true",
                        help="also download/cache fold 10; keep disabled until final evaluation")
    parser.add_argument("--test-only-cache", action="store_true",
                        help="cache only fold 10 after the model and thresholds are frozen")
    args = parser.parse_args()
    if args.include_test and args.test_only_cache:
        parser.error("--include-test and --test-only-cache cannot be combined")
    fetch_metadata(args.dataset_root)
    records, _statements, report = audit_metadata(args.dataset_root)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "dataset_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    source_hashes = {
        name: hashlib.sha256((args.dataset_root / name).read_bytes()).hexdigest()
        for name in (*METADATA_FILES, "SHA256SUMS.txt")
        if (args.dataset_root / name).is_file()
    }
    license_path = args.dataset_root / "LICENSE.txt"
    license_hash = (hashlib.sha256(license_path.read_bytes()).hexdigest()
                    if license_path.is_file() else None)
    (args.cache_dir / "source_manifest.json").write_text(
        json.dumps({
            "source": "PhysioNet PTB-XL 1.0.3",
            "source_url": "https://physionet.org/content/ptb-xl/1.0.3/",
            "signal_mirror": "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/",
            "license": "CC BY 4.0",
            "license_sha256": license_hash,
            "metadata_sha256": source_hashes,
            "preprocess_json_sha256": hashlib.sha256(
                (ROOT / "nn" / "preprocess.json").read_bytes()
            ).hexdigest(),
        }, indent=2), encoding="utf-8"
    )
    manifest = records[["patient_id", "strat_fold", "filename_hr", "label_mask"]].copy()
    manifest["labels"] = records["labels"].map(lambda x: "".join(map(str, x.tolist())))
    manifest.to_csv(args.cache_dir / "records_manifest.csv", index_label="ecg_id")
    print(json.dumps(report, indent=2))
    if args.metadata_only or args.audit_only:
        return
    folds = ([10] if args.test_only_cache
             else list(range(1, 11 if args.include_test else 10)))
    if not args.skip_download:
        download_records(args.dataset_root, records, folds)
    result = build_cache(args.dataset_root, args.cache_dir, records,
                         load_contract(), folds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
