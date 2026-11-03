"""Bind the trained checkpoint to its preprocessing and evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "nn" / "artifacts")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "nn" / "data" / "cache")
    args = parser.parse_args()
    artifact_dir, cache_dir = args.artifact_dir.resolve(), args.cache_dir.resolve()
    checkpoint_path = artifact_dir / "model_fp32.pt"
    fold9_path = artifact_dir / "fold9_metrics.json"
    fold10_path = artifact_dir / "fold10_metrics.json"
    preprocess_path = ROOT / "nn" / "preprocess.json"
    config_path = ROOT / "nn" / "configs" / "train_ptbxl.yaml"
    normalization_path = cache_dir / "normalization.json"
    source_path = cache_dir / "source_manifest.json"
    required = (checkpoint_path, fold9_path, preprocess_path,
                config_path, normalization_path, source_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing manifest inputs: " + ", ".join(missing))

    checkpoint_hash = sha256_file(checkpoint_path)
    fold9 = read_json(fold9_path)
    source = read_json(source_path)
    fold10 = read_json(fold10_path) if fold10_path.is_file() else None
    if fold10 and fold10.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("fold-10 report was produced from a different checkpoint")

    manifest = {
        "model_id": "ptbxl-12lead-resnet1d-fp32-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "record-level five-label ECG classification",
        "architecture": "ECGResNet1D",
        "parameter_count": fold9["parameter_count"],
        "precision": "FP32",
        "input": {
            "shape_per_record": [12, 5000],
            "sample_rate_hz": 500,
            "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF",
                           "V1", "V2", "V3", "V4", "V5", "V6"],
            "physical_unit": "mV before fixed global normalization",
        },
        "outputs": ["NORM", "MI", "STTC", "CD", "HYP"],
        "data": {
            "name": source["source"],
            "license": source.get("license"),
            "source_url": source["source_url"],
            "official_splits": {"train": list(range(1, 9)), "validation": 9, "final_test": 10},
            "training_records": fold9["training_records"],
            "validation_records": fold9["validation_records"],
            "final_test_records": fold10["records"] if fold10 else None,
        },
        "selected_epoch": fold9["best_epoch"],
        "fold9_thresholds": {
            label: data["threshold"]
            for label, data in fold9["thresholds"].items()
        },
        "validation_metrics": {
            "macro_auroc": fold9["macro_auroc"],
            "macro_auprc": fold9["macro_auprc"],
            "per_class": fold9["per_class"],
        },
        "final_test_metrics": ({
            "split": fold10["split"],
            "macro_auroc": fold10["macro_auroc"],
            "macro_auprc": fold10["macro_auprc"],
            "per_class": fold10["per_class"],
            "patient_bootstrap_95pct_ci": fold10["patient_bootstrap_95pct_ci"],
            "frozen_threshold_metrics": fold10["frozen_threshold_metrics"],
        } if fold10 else None),
        "runtime": fold9["runtime"],
        "hashes_sha256": {
            "checkpoint": checkpoint_hash,
            "preprocess_json": sha256_file(preprocess_path),
            "train_config": sha256_file(config_path),
            "normalization": sha256_file(normalization_path),
            "source_manifest": sha256_file(source_path),
            "fold9_metrics": sha256_file(fold9_path),
            "fold10_metrics": sha256_file(fold10_path) if fold10 else None,
        },
        "limitations": [
            "Public-dataset research and competition result; not clinical validation.",
            "Labels are recording-level diagnostic superclass labels, not beat-level truth.",
        ],
    }
    output = artifact_dir / "model_manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"saved: {output}")
    print(f"checkpoint_sha256: {checkpoint_hash}")


if __name__ == "__main__":
    main()
