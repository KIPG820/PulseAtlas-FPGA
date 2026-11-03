"""Train the agreed FP32 multi-label ECGResNet1D on PTB-XL folds 1-8."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nn"))
from models import ECGResNet1D  # noqa: E402
from preprocess import LEAD_ORDER  # noqa: E402

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")


class ECGCache(Dataset):
    def __init__(self, cache_dir: Path, folds: list[int]):
        xs, ys = [], []
        for fold in folds:
            xp, yp = cache_dir / f"fold_{fold}_x.npy", cache_dir / f"fold_{fold}_y.npy"
            if not xp.is_file() or not yp.is_file():
                raise FileNotFoundError(f"missing cached fold {fold}; run prepare_ptbxl.py")
            xs.append(np.load(xp, mmap_mode="r"))
            ys.append(np.load(yp, mmap_mode="r"))
        self.x_parts, self.y_parts = xs, ys
        self.ends = np.cumsum([0] + [len(x) for x in xs])
        self.length = int(self.ends[-1])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        part = int(np.searchsorted(self.ends[1:], index, side="right"))
        local = index - int(self.ends[part])
        x = np.array(self.x_parts[part][local], dtype=np.float32, copy=True)
        y = np.array(self.y_parts[part][local], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_batch(x, mean, std, clip):
    return ((x - mean[None, :, None]) / std[None, :, None]).clamp(-clip, clip)


def augment_batch(x, cfg, sample_rate: int):
    """Small training-fold-only perturbations in physical mV units."""
    if not cfg.get("enabled", False):
        return x
    b, _leads, n = x.shape
    device, dtype = x.device, x.dtype
    scale = torch.empty((b, 1, 1), device=device, dtype=dtype).uniform_(
        float(cfg["amplitude_min"]), float(cfg["amplitude_max"])
    )
    x = x * scale
    t = torch.arange(n, device=device, dtype=dtype)[None, None, :] / sample_rate
    baseline_amp = torch.rand((b, 12, 1), device=device, dtype=dtype)
    baseline_phase = torch.rand((b, 12, 1), device=device, dtype=dtype) * (2 * np.pi)
    baseline_hz = torch.empty((b, 1, 1), device=device, dtype=dtype).uniform_(0.15, 0.45)
    x = x + baseline_amp * float(cfg["baseline_wander_mv"]) * torch.sin(
        2 * np.pi * baseline_hz * t + baseline_phase
    )
    mains_phase = torch.rand((b, 1, 1), device=device, dtype=dtype) * (2 * np.pi)
    x = x + float(cfg["mains_50hz_mv"]) * torch.sin(2 * np.pi * 50.0 * t + mains_phase)
    x = x + torch.randn_like(x) * float(cfg["gaussian_noise_mv"])
    max_shift = int(round(float(cfg.get("time_shift_ms", 0)) * sample_rate / 1000))
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (b, 1, 1), device=device)
        indices = (torch.arange(n, device=device)[None, None, :] + shifts).clamp(0, n - 1)
        indices = indices.expand(-1, 12, -1)
        x = x.gather(2, indices)
    return x


@torch.inference_mode()
def predict(model, loader, device, mean, std, clip):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = normalize_batch(x.to(device), mean, std, clip)
        all_logits.append(model(x).float().cpu().numpy())
        all_y.append(y.numpy())
    if not all_y:
        raise RuntimeError("empty validation split")
    return np.concatenate(all_logits), np.concatenate(all_y).astype(np.uint8)


def safe_auc(y, p):
    values = []
    for i in range(y.shape[1]):
        if np.unique(y[:, i]).size == 2:
            values.append(roc_auc_score(y[:, i], p[:, i]))
    return float(np.mean(values)) if values else float("nan")


def score(logits, y):
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    per_class = {}
    aucs, aps = [], []
    for i, label in enumerate(LABELS):
        if np.unique(y[:, i]).size < 2:
            auc, ap = float("nan"), float("nan")
        else:
            auc = float(roc_auc_score(y[:, i], p[:, i]))
            ap = float(average_precision_score(y[:, i], p[:, i]))
            aucs.append(auc)
            aps.append(ap)
        per_class[label] = {"auroc": auc, "auprc": ap, "positive_count": int(y[:, i].sum())}
    return p, per_class, float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(aps)) if aps else float("nan")


def choose_thresholds(y, p):
    thresholds = []
    report = {}
    for i, label in enumerate(LABELS):
        if np.unique(y[:, i]).size < 2:
            threshold = 0.5
        else:
            precision, recall, values = precision_recall_curve(y[:, i], p[:, i])
            # The final PR point has no corresponding threshold.
            f1 = (2 * precision[:-1] * recall[:-1]
                  / np.maximum(precision[:-1] + recall[:-1], 1e-12))
            threshold = float(values[int(np.nanargmax(f1))]) if len(values) else 0.5
        pred = p[:, i] >= threshold
        tp = int(np.sum(pred & (y[:, i] == 1)))
        fp = int(np.sum(pred & (y[:, i] == 0)))
        fn = int(np.sum(~pred & (y[:, i] == 1)))
        tn = int(np.sum(~pred & (y[:, i] == 0)))
        report[label] = {
            "threshold": threshold,
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "f1": float(f1_score(y[:, i], pred, zero_division=0)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        thresholds.append(threshold)
    return np.asarray(thresholds, dtype=np.float32), report


def load_cache_normalization(cache_dir: Path):
    path = cache_dir / "normalization.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run prepare_ptbxl.py first")
    stats = json.loads(path.read_text(encoding="utf-8"))
    if stats["lead_order"] != list(LEAD_ORDER) or stats["source_folds"] != list(range(1, 9)):
        raise ValueError("normalization statistics do not match the fixed training-fold contract")
    return (torch.tensor(stats["mean_mV"], dtype=torch.float32),
            torch.tensor(stats["std_mV"], dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "nn" / "configs" / "train_ptbxl.yaml")
    parser.add_argument("--epochs", type=int, help="optional shorter run for smoke testing")
    parser.add_argument("--smoke-test", action="store_true",
                        help="test model forward/loss on synthetic data; does not train or save a model")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ECGResNet1D().to(smoke_device)
        x = torch.randn(2, 12, 5000, device=smoke_device)
        y = model(x)
        loss = nn.BCEWithLogitsLoss()(y, torch.zeros_like(y))
        assert y.shape == (2, 5) and torch.isfinite(loss)
        print(f"PASS: device={smoke_device}, input={tuple(x.shape)}, "
              f"logits={tuple(y.shape)}, loss={loss.item():.5f}")
        return

    config_path = args.config.resolve()
    cfg_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(cfg_bytes.decode("utf-8"))
    cfg_path_base = ROOT
    cache_dir = (cfg_path_base / cfg["cache_dir"]).resolve()
    artifact_dir = (cfg_path_base / cfg["artifact_dir"]).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["seed"]))
    device_name = cfg.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    mean, std = load_cache_normalization(cache_dir)
    mean, std = mean.to(device), std.to(device).clamp_min(1e-8)
    clip = float(cfg["normalization_clip"])

    train_ds = ECGCache(cache_dir, [int(f) for f in cfg["train_folds"]])
    val_ds = ECGCache(cache_dir, [int(cfg["validation_fold"])])
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=(device.type == "cuda"),
    )
    y_train = np.concatenate([np.load(cache_dir / f"fold_{f}_y.npy")
                              for f in cfg["train_folds"]])
    pos = y_train.sum(axis=0).astype(np.float64)
    neg = y_train.shape[0] - pos
    if np.any(pos == 0):
        raise ValueError("at least one training label has no positive samples")
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)

    model = ECGResNet1D().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config_hash = hashlib.sha256(cfg_bytes).hexdigest()
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    max_epochs = int(args.epochs if args.epochs is not None else cfg["epochs"])
    patience = int(cfg["early_stopping_patience"])
    best_auc, best_loss, stale = -float("inf"), float("inf"), 0
    best_path = artifact_dir / "model_fp32.pt"
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = augment_batch(x, cfg.get("augmentation", {}), int(cfg["sample_rate"]))
            x = normalize_batch(x, mean, std, clip)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            seen += len(y)

        val_logits, val_y = predict(model, val_loader, device, mean, std, clip)
        val_p, per_class, val_auc, val_ap = score(val_logits, val_y)
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(val_logits), torch.from_numpy(val_y.astype(np.float32)),
            pos_weight=pos_weight.cpu(),
        ))
        scheduler.step(val_loss)
        row = {
            "epoch": epoch, "train_loss": train_loss_sum / max(seen, 1),
            "validation_loss": val_loss, "macro_auroc": val_auc,
            "macro_auprc": val_ap, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        improved = np.isfinite(val_auc) and val_auc > best_auc
        if improved:
            best_auc, best_loss, stale = val_auc, val_loss, 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "labels": list(LABELS),
                "lead_order": list(LEAD_ORDER),
                "mean_mV": mean.cpu().tolist(),
                "std_mV": std.cpu().tolist(),
                "normalization_clip": clip,
                "validation_macro_auroc": val_auc,
                "validation_macro_auprc": val_ap,
                "parameter_count": parameter_count,
                "seed": int(cfg["seed"]),
                "config_sha256": config_hash,
                "runtime": runtime,
            }, best_path)
        else:
            stale += 1
        if stale >= patience:
            print(f"early stop: no fold-9 macro-AUROC improvement for {patience} epochs")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    logits, val_y = predict(model, val_loader, device, mean, std, clip)
    p, per_class, macro_auc, macro_ap = score(logits, val_y)
    thresholds, threshold_report = choose_thresholds(val_y, p)
    checkpoint["fold9_thresholds"] = thresholds.tolist()
    checkpoint["fold9_threshold_metrics"] = threshold_report
    torch.save(checkpoint, best_path)
    preprocess_path = ROOT / "nn" / "preprocess.json"
    preprocess_hash = hashlib.sha256(preprocess_path.read_bytes()).hexdigest()
    metrics = {
        "split": "fold 9 validation only",
        "best_epoch": int(checkpoint["epoch"]),
        "macro_auroc": macro_auc,
        "macro_auprc": macro_ap,
        "per_class": per_class,
        "thresholds": threshold_report,
        "preprocess_sha256": preprocess_hash,
        "config_sha256": config_hash,
        "training_records": len(train_ds),
        "validation_records": len(val_ds),
        "training_label_positive_counts": {
            label: int(pos[i]) for i, label in enumerate(LABELS)
        },
        "parameter_count": parameter_count,
        "runtime": runtime,
        "elapsed_seconds": time.perf_counter() - start_time,
        "device": str(device),
        "history": history,
    }
    (artifact_dir / "fold9_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"saved best FP32 model: {best_path}")
    print(f"fold-9 macro AUROC={macro_auc:.4f}, macro AUPRC={macro_ap:.4f}")
    print("Fold 10 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
