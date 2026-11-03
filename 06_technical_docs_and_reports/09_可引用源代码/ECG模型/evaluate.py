"""Run one frozen final evaluation on PTB-XL official fold 10."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nn"))
sys.path.insert(0, str(ROOT / "nn" / "scripts"))
from models import ECGResNet1D  # noqa: E402
from preprocess import LEAD_ORDER  # noqa: E402
from train import ECGCache, LABELS, load_cache_normalization, predict, score  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_metrics(y: np.ndarray, p: np.ndarray,
                      thresholds: np.ndarray) -> dict:
    report = {}
    for i, label in enumerate(LABELS):
        predicted = p[:, i] >= thresholds[i]
        truth = y[:, i].astype(bool)
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        tn = int(np.sum(~predicted & ~truth))
        report[label] = {
            "threshold_from_fold9": float(thresholds[i]),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    return report


def patient_bootstrap(logits: np.ndarray, y: np.ndarray,
                      patient_ids: np.ndarray, replicates: int,
                      seed: int) -> dict:
    """Percentile 95% CI with patients, not individual records, as clusters."""
    unique_patients = np.unique(patient_ids)
    rows_by_patient = {
        patient: np.flatnonzero(patient_ids == patient)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    macro_auc, macro_ap = [], []
    per_class = {label: {"auroc": [], "auprc": []} for label in LABELS}
    for _ in range(replicates):
        chosen = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([rows_by_patient[patient] for patient in chosen])
        _p, class_scores, auc, ap = score(logits[indices], y[indices])
        macro_auc.append(auc)
        macro_ap.append(ap)
        for label in LABELS:
            per_class[label]["auroc"].append(class_scores[label]["auroc"])
            per_class[label]["auprc"].append(class_scores[label]["auprc"])

    def interval(values):
        values = np.asarray(values, dtype=np.float64)
        return [float(v) for v in np.nanpercentile(values, [2.5, 97.5])]

    return {
        "method": "patient-cluster bootstrap, percentile 95% CI",
        "replicates": int(replicates),
        "seed": int(seed),
        "macro_auroc": interval(macro_auc),
        "macro_auprc": interval(macro_ap),
        "per_class": {
            label: {metric: interval(values) for metric, values in per_class[label].items()}
            for label in LABELS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "nn" / "configs" / "train_ptbxl.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true",
                        help="explicitly repeat a fold-10 report; never changes the frozen model")
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap-replicates must be positive")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache_dir = (ROOT / cfg["cache_dir"]).resolve()
    artifact_dir = (ROOT / cfg["artifact_dir"]).resolve()
    checkpoint_path = artifact_dir / "model_fp32.pt"
    output_path = artifact_dir / "fold10_metrics.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; use --overwrite only to repeat the same frozen evaluation"
        )
    if not (cache_dir / "fold_10_x.npy").is_file():
        raise FileNotFoundError(
            "fold 10 signal cache is missing; download folds=10 and run "
            "prepare_ptbxl.py --test-only-cache --skip-download"
        )

    checkpoint_hash_before = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("checkpoint label order does not match the frozen five-label task")
    if checkpoint.get("lead_order") != list(LEAD_ORDER):
        raise ValueError("checkpoint lead order does not match the preprocessing contract")
    thresholds = np.asarray(checkpoint.get("fold9_thresholds"), dtype=np.float32)
    if thresholds.shape != (len(LABELS),):
        raise ValueError("checkpoint does not contain frozen fold-9 thresholds")

    device_name = cfg.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    model = ECGResNet1D().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    mean, std = load_cache_normalization(cache_dir)
    mean, std = mean.to(device), std.to(device).clamp_min(1e-8)
    test_ds = ECGCache(cache_dir, [10])
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=(device.type == "cuda"),
    )

    manifest = pd.read_csv(cache_dir / "records_manifest.csv", index_col="ecg_id")
    manifest = manifest[(manifest["strat_fold"] == 10) & (manifest["label_mask"] == 1)]
    expected_ids = manifest.index.to_numpy(dtype=np.int32)
    cached_ids = np.load(cache_dir / "fold_10_ids.npy", allow_pickle=False)
    if not np.array_equal(expected_ids, cached_ids):
        raise ValueError("fold-10 cache IDs do not match the official manifest order")
    encoded_labels = np.stack([
        np.fromiter((int(bit) for bit in f"{int(value):0{len(LABELS)}d}"),
                    dtype=np.uint8, count=len(LABELS))
        for value in manifest["labels"]
    ])
    cached_labels = np.load(cache_dir / "fold_10_y.npy", mmap_mode="r")
    if not np.array_equal(encoded_labels, cached_labels):
        raise ValueError("fold-10 cache labels do not match the official manifest")

    started = time.perf_counter()
    logits, y = predict(model, test_loader, device, mean, std,
                        float(cfg["normalization_clip"]))
    probabilities, per_class, macro_auc, macro_ap = score(logits, y)
    patients = manifest["patient_id"].to_numpy()
    bootstrap = patient_bootstrap(
        logits, y, patients,
        replicates=args.bootstrap_replicates,
        seed=int(cfg["seed"]) + 10,
    )
    checkpoint_hash_after = sha256_file(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("frozen checkpoint changed during final evaluation")

    result = {
        "split": "official PTB-XL fold 10; one final evaluation",
        "records": int(len(test_ds)),
        "unique_patients": int(manifest["patient_id"].nunique()),
        "best_epoch_from_fold9": int(checkpoint["epoch"]),
        "checkpoint_sha256": checkpoint_hash_before,
        "thresholds_source": "frozen fold 9 only",
        "macro_auroc": macro_auc,
        "macro_auprc": macro_ap,
        "per_class": per_class,
        "frozen_threshold_metrics": threshold_metrics(y, probabilities, thresholds),
        "patient_bootstrap_95pct_ci": bootstrap,
        "device": str(device),
        "runtime": checkpoint.get("runtime", {}),
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Research/competition evaluation only; not clinical validation.",
    }
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps({
        "saved": str(output_path),
        "macro_auroc": macro_auc,
        "macro_auprc": macro_ap,
        "records": len(test_ds),
        "patients": int(manifest["patient_id"].nunique()),
        "checkpoint_sha256": checkpoint_hash_before,
        "fold10_thresholds_were_not_changed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
