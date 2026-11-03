"""Retrain the tiny pad/no-pad model with stronger exposure and white-balance jitter."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SOURCE = Path(r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ConnectorIntegration_20261014")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import train_patch_presence as base


class PhotometricPatchSet(Dataset):
    def __init__(self, rows, train=False):
        self.base = base.PatchSet(rows, augment=train)
        self.train = train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        if self.train:
            # Simulate exposure changes and warm/cool white balance without
            # changing the patch geometry or labels.
            if torch.rand(()) < 0.75:
                gain = 0.78 + torch.rand(3, 1, 1) * 0.44
                exposure = 0.70 + torch.rand(()) * 0.60
                image = image * gain * exposure
            if torch.rand(()) < 0.55:
                gamma = 0.78 + torch.rand(()) * 0.44
                image = image.clamp(0, 1).pow(gamma)
            if torch.rand(()) < 0.30:
                mean = image.mean(dim=(1, 2), keepdim=True)
                contrast = 0.80 + torch.rand(()) * 0.40
                image = (image - mean) * contrast + mean
            if torch.rand(()) < 0.20:
                image = image + torch.randn_like(image) * 0.012
            image = image.clamp(0, 1)
        return image, label


def collect_probabilities(model, loader):
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for image, label in loader:
            scores.extend(torch.softmax(model(image), dim=1)[:, 0].cpu().tolist())
            labels.extend(label.tolist())
    return np.asarray(scores), np.asarray(labels)


def summarize(scores, labels, threshold=0.5):
    pred = scores >= threshold
    pos, neg = labels == 0, labels == 1
    tp, fn = int(np.sum(pred & pos)), int(np.sum(~pred & pos))
    fp, tn = int(np.sum(pred & neg)), int(np.sum(~pred & neg))
    return {
        "threshold": float(threshold), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "pad_recall": tp / max(1, tp + fn),
        "false_electrode_rate_on_none": fp / max(1, fp + tn),
        "none_precision": tn / max(1, tn + fn),
        "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
    }


def main():
    base.seed_everything()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, frame_counts, excluded_counts, bg_side = base.build_patch_sets()
    train_loader = DataLoader(PhotometricPatchSet(samples["train"], True),
                              batch_size=base.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PhotometricPatchSet(samples["val"]),
                            batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(PhotometricPatchSet(samples["test"]),
                             batch_size=128, shuffle=False, num_workers=0)

    model = base.TinyPatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    histogram = np.bincount([label for _, label in samples["train"]], minlength=2).astype(np.float32)
    weights_np = len(samples["train"]) / (2 * np.maximum(histogram, 1.0))
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    best_score, best_state, history = -1.0, None, []
    for epoch in range(60):
        model.train()
        total, seen = 0.0, 0
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(image), label)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(label)
            seen += len(label)
        scheduler.step()
        val_scores, val_labels = collect_probabilities(model, val_loader)
        row = {"epoch": epoch + 1, "train_loss": total / max(1, seen),
               **summarize(val_scores, val_labels)}
        history.append(row)
        # Prefer lower false-trigger rate while retaining high pad recall.
        score = row["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(row), flush=True)

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    val_scores, val_labels = collect_probabilities(model, val_loader)
    test_scores, test_labels = collect_probabilities(model, test_loader)
    thresholds = [round(float(x), 2) for x in np.arange(0.45, 0.851, 0.025)]
    val_curve = [summarize(val_scores, val_labels, threshold) for threshold in thresholds]
    feasible = [r for r in val_curve if r["false_electrode_rate_on_none"] <= 0.05]
    selected = max(feasible, key=lambda r: (r["pad_recall"], r["balanced_accuracy"], r["threshold"]))["threshold"] if feasible else None

    checkpoint = ROOT / "tiny_patch_presence_robust.pt"
    torch.save({"state_dict": best_state, "classes": base.CLASSES,
                "patch_size": base.PATCH_SIZE,
                "architecture": "TinyPatchNetPresence"}, checkpoint)
    model.cpu().eval()
    dummy = torch.zeros(10, 3, base.PATCH_SIZE, base.PATCH_SIZE)
    onnx_path = ROOT / "tiny_patch_presence_robust_batch10.onnx"
    onnx_note = "onnx package unavailable"
    if importlib.util.find_spec("onnx") is not None:
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["patches"],
                          output_names=["logits"], opset_version=11,
                          do_constant_folding=True, dynamo=False)
        onnx_note = "exported with fixed batch size 10"

    report = {
        "experiment": "photometric augmentation for light and white-balance robustness",
        "source_checkpoint": str(SOURCE / "tiny_patch_presence_best.pt"),
        "train_frames": frame_counts,
        "excluded_frames": {f"{k[0]}:{k[1]}": v for k, v in excluded_counts.items()},
        "training_crop_counts": {split: dict(Counter(label for _, label in rows))
                                 for split, rows in samples.items()},
        "augmentations": ["gain/exposure", "gamma", "contrast", "sensor noise",
                          "existing rotation/translation/flip/brightness/saturation"],
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path) if onnx_path.exists() else None,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest() if onnx_path.exists() else None,
        "onnx_note": onnx_note,
        "validation_at_0_5": summarize(val_scores, val_labels, 0.5),
        "frozen_test_at_0_5": summarize(test_scores, test_labels, 0.5),
        "validation_curve": val_curve,
        "threshold_selected_on_validation_only": selected,
        "frozen_test_at_selected_threshold": None if selected is None else summarize(test_scores, test_labels, selected),
        "history": history,
        "limitations": ["All source videos come from one person/capture family.",
                        "The frozen test split is small and contains only the supplied lighting conditions.",
                        "Annotation-centered crops do not fully reproduce pose-point error."],
    }
    report_path = ROOT / "presence_photometric_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("validation_at_0_5", "frozen_test_at_0_5",
                                            "threshold_selected_on_validation_only",
                                            "frozen_test_at_selected_threshold", "onnx", "report")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
