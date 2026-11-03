"""Compare frozen presence checkpoints under controlled photometric shifts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ConnectorIntegration_20261014")
sys.path.insert(0, str(SOURCE))
import train_patch_presence as base


class FixedDataset(Dataset):
    def __init__(self, samples, scenario):
        self.data = base.PatchSet(samples, augment=False)
        self.scenario = scenario

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]
        gains, exposure, gamma = self.scenario
        image = (image * torch.tensor(gains, dtype=image.dtype).view(3, 1, 1) * exposure)
        image = image.clamp(0, 1).pow(gamma).clamp(0, 1)
        return image, label


def metrics(scores, labels, threshold=0.5):
    pred = scores >= threshold
    pos, neg = labels == 0, labels == 1
    tp, fn = int(np.sum(pred & pos)), int(np.sum(~pred & pos))
    fp, tn = int(np.sum(pred & neg)), int(np.sum(~pred & neg))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "pad_recall": tp / max(1, tp + fn),
            "false_electrode_rate_on_none": fp / max(1, fp + tn),
            "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))}


def evaluate(model, loader):
    scores, labels = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            scores.extend(torch.softmax(model(x), dim=1)[:, 0].tolist())
            labels.extend(y.tolist())
    scores, labels = np.asarray(scores), np.asarray(labels)
    return {"threshold_{}".format(threshold): metrics(scores, labels, threshold)
            for threshold in (0.50, 0.55, 0.60)}


def load_model(path):
    model = base.TinyPatchNet()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main():
    base.seed_everything()
    samples, _, _, _ = base.build_patch_sets()
    checkpoints = {
        "baseline": SOURCE / "tiny_patch_presence_best.pt",
        "photometric_finetune": ROOT / "tiny_patch_presence_finetuned.pt",
    }
    scenarios = {
        "nominal": ((1.0, 1.0, 1.0), 1.0, 1.0),
        "dim": ((1.0, 1.0, 1.0), 0.72, 1.0),
        "bright": ((1.0, 1.0, 1.0), 1.20, 1.0),
        "warm_white_balance": ((1.10, 1.0, 0.88), 1.0, 1.0),
        "cool_white_balance": ((0.88, 1.0, 1.10), 1.0, 1.0),
        "dim_warm_gamma": ((1.08, 1.0, 0.90), 0.82, 0.90),
        "bright_cool_gamma": ((0.90, 1.0, 1.08), 1.08, 1.10),
    }
    output = {"experiment": "synthetic held-out test illumination replay",
              "scenarios_are_synthetic": True, "results": {}}
    for scenario_name, scenario in scenarios.items():
        loader = DataLoader(FixedDataset(samples["test"], scenario), batch_size=128,
                            shuffle=False, num_workers=0)
        output["results"][scenario_name] = {}
        for model_name, checkpoint in checkpoints.items():
            output["results"][scenario_name][model_name] = evaluate(load_model(checkpoint), loader)
    path = ROOT / "illumination_replay_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
