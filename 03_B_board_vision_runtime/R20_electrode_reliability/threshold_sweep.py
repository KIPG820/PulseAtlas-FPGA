"""Select a pad-presence confidence gate on validation; freeze then report test."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SOURCE = Path(r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ConnectorIntegration_20261014")
sys.path.insert(0, str(SOURCE))
import train_patch_presence as train

OUT = Path(__file__).resolve().parent / "presence_threshold_report.json"


def main():
    train.seed_everything()
    torch.set_num_threads(2)
    samples, frame_counts, excluded, _ = train.build_patch_sets()
    checkpoint = torch.load(SOURCE / "tiny_patch_presence_best.pt",
                            map_location="cpu", weights_only=False)
    model = train.TinyPatchNet().eval()
    model.load_state_dict(checkpoint["state_dict"])
    arrays = {}
    with torch.inference_mode():
        for split in ("val", "test"):
            loader = DataLoader(train.PatchSet(samples[split], augment=False),
                                batch_size=64, shuffle=False)
            probs, labels = [], []
            for image, label in loader:
                logits = model(image)
                probs.append(torch.softmax(logits, dim=1)[:, 0].numpy())
                labels.append(label.numpy())
            arrays[split] = (np.concatenate(probs), np.concatenate(labels))

    def metrics(threshold, probs, labels):
        pred = probs >= threshold
        positive, negative = labels == 0, labels == 1
        tp = int(np.sum(pred & positive))
        fn = int(np.sum(~pred & positive))
        fp = int(np.sum(pred & negative))
        tn = int(np.sum(~pred & negative))
        return {
            "threshold": float(threshold), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "pad_recall": tp / max(1, tp + fn),
            "false_electrode_rate_on_none": fp / max(1, fp + tn),
            "none_precision": tn / max(1, tn + fn),
            "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        }

    thresholds = [round(float(t), 2) for t in np.arange(0.45, 0.951, 0.025)]
    val = [metrics(t, *arrays["val"]) for t in thresholds]
    feasible = [row for row in val if row["false_electrode_rate_on_none"] <= 0.05]
    if not feasible:
        raise RuntimeError("validation has no threshold with <=5% false-positive rate")
    # The gate is selected without consulting the frozen test split.
    selected = max(feasible, key=lambda row: (row["pad_recall"],
                                               row["balanced_accuracy"],
                                               row["threshold"]))["threshold"]
    report = {
        "criterion": "choose highest validation pad recall subject to <=5% false-electrode rate",
        "selected_threshold_from_validation_only": selected,
        "split_frame_counts": frame_counts,
        "excluded_frame_counts": {f"{k[0]}:{k[1]}": v for k, v in excluded.items()},
        "validation_curve": val,
        "frozen_test_at_baseline_0_5": metrics(0.5, *arrays["test"]),
        "frozen_test_at_selected_threshold": metrics(selected, *arrays["test"]),
        "limitations": [
            "The source set is a single person's video family.",
            "Validation and test crops are annotation-centered, not live pose-centered ROIs.",
            "This threshold is a first false-trigger gate and does not replace a live replay.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_threshold": selected,
        "val_baseline": metrics(0.5, *arrays["val"]),
        "val_selected": metrics(selected, *arrays["val"]),
        "test_baseline": report["frozen_test_at_baseline_0_5"],
        "test_selected": report["frozen_test_at_selected_threshold"],
        "report": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
