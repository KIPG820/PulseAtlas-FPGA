"""Fine-tune the deployed-style presence model with conservative light jitter."""
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
            # Modest exposure and per-channel gain shifts supplement the
            # existing brightness/saturation augmentation without washing out
            # electrodes into unrealistic near-black/near-white patches.
            if torch.rand(()) < 0.65:
                gain = 0.92 + torch.rand(3, 1, 1) * 0.16
                exposure = 0.86 + torch.rand(()) * 0.28
                image = image * gain * exposure
            if torch.rand(()) < 0.35:
                gamma = 0.90 + torch.rand(()) * 0.20
                image = image.clamp(0, 1).pow(gamma)
            if torch.rand(()) < 0.15:
                image = image + torch.randn_like(image) * 0.008
            image = image.clamp(0, 1)
        return image, label


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


def collect_probabilities(model, loader):
    model.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for image, label in loader:
            scores.extend(torch.softmax(model(image), dim=1)[:, 0].cpu().tolist())
            labels.extend(label.tolist())
    return np.asarray(scores), np.asarray(labels)


def main():
    base.seed_everything()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, frame_counts, excluded_counts, _ = base.build_patch_sets()
    train_loader = DataLoader(PhotometricPatchSet(samples["train"], True),
                              batch_size=base.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PhotometricPatchSet(samples["val"]),
                            batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(PhotometricPatchSet(samples["test"]),
                             batch_size=128, shuffle=False, num_workers=0)

    model = base.TinyPatchNet()
    source_checkpoint = torch.load(SOURCE / "tiny_patch_presence_best.pt",
                                  map_location="cpu", weights_only=False)
    model.load_state_dict(source_checkpoint["state_dict"])
    model.to(device)
    val_scores, val_labels = collect_probabilities(model, val_loader)
    initial = summarize(val_scores, val_labels)
    best_score = initial["balanced_accuracy"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, **initial}]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    histogram = np.bincount([label for _, label in samples["train"]], minlength=2).astype(np.float32)
    weights = torch.tensor(len(samples["train"]) / (2 * np.maximum(histogram, 1.0)),
                           dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.01)
    for epoch in range(25):
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
        if row["balanced_accuracy"] > best_score:
            best_score = row["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps(row), flush=True)

    model.load_state_dict(best_state)
    model.to(device).eval()
    val_scores, val_labels = collect_probabilities(model, val_loader)
    test_scores, test_labels = collect_probabilities(model, test_loader)
    thresholds = [round(float(x), 3) for x in np.arange(0.45, 0.851, 0.025)]
    val_curve = [summarize(val_scores, val_labels, threshold) for threshold in thresholds]
    feasible = [r for r in val_curve if r["false_electrode_rate_on_none"] <= 0.05]
    selected = max(feasible, key=lambda r: (r["pad_recall"], r["balanced_accuracy"], r["threshold"]))["threshold"] if feasible else None

    checkpoint = ROOT / "tiny_patch_presence_finetuned.pt"
    torch.save({"state_dict": best_state, "classes": base.CLASSES,
                "patch_size": base.PATCH_SIZE, "architecture": "TinyPatchNetPresence"}, checkpoint)
    model.cpu().eval()
    dummy = torch.zeros(10, 3, base.PATCH_SIZE, base.PATCH_SIZE)
    onnx_path = ROOT / "tiny_patch_presence_finetuned_batch10.onnx"
    onnx_note = "onnx package unavailable"
    if importlib.util.find_spec("onnx") is not None:
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["patches"],
                          output_names=["logits"], opset_version=11,
                          do_constant_folding=True, dynamo=False)
        onnx_note = "exported with fixed batch size 10"

    report = {
        "experiment": "fine-tune current presence model with conservative illumination jitter",
        "source_checkpoint": str(SOURCE / "tiny_patch_presence_best.pt"),
        "initial_validation_at_0_5": initial,
        "train_frames": frame_counts,
        "excluded_frames": {f"{k[0]}:{k[1]}": v for k, v in excluded_counts.items()},
        "training_crop_counts": {split: dict(Counter(label for _, label in rows))
                                 for split, rows in samples.items()},
        "augmentations": ["existing rotation/translation/flip/brightness/saturation",
                          "modest exposure and channel-gain white-balance jitter",
                          "modest gamma and sensor-noise jitter"],
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
    report_path = ROOT / "presence_finetuned_report.json"
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("initial_validation_at_0_5", "validation_at_0_5",
                                            "frozen_test_at_0_5", "threshold_selected_on_validation_only",
                                            "frozen_test_at_selected_threshold", "onnx", "report")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
