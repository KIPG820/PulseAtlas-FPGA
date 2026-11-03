"""Export the GPU-trained checkpoint with the base Python ONNX toolchain."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnx
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train_presence_from_user_annotations as train  # noqa: E402


def main():
    checkpoint_path = ROOT / "tiny_patch_presence_best.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model = train.TinyPatchNet().eval()
    model.load_state_dict(checkpoint["state_dict"])
    onnx_path = ROOT / "tiny_patch_presence_batch10.onnx"
    torch.onnx.export(
        model,
        torch.zeros(10, 3, train.PATCH_SIZE, train.PATCH_SIZE),
        str(onnx_path),
        input_names=["patches"],
        output_names=["logits"],
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    frames, _, _ = train.load_frames_and_boxes()
    samples, split_counts = train.build_samples(frames)
    baseline = Path(
        r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ColorHybrid_HSV_R3_SourceAgeGateFix_20261014"
    ) / "tiny_patch_presence_batch10.onnx"
    baseline_val, _ = train.evaluate_opencv(baseline, samples["val"])
    baseline_test, _ = train.evaluate_opencv(baseline, samples["test"])
    candidate_val, val_probs = train.evaluate_opencv(onnx_path, samples["val"])
    candidate_test, test_probs = train.evaluate_opencv(onnx_path, samples["test"])

    pad_rows = [row for row in samples["test"] if row[1] == 0][:5]
    none_rows = [row for row in samples["test"] if row[1] == 1][:5]
    batch_rows = [row for pair in zip(pad_rows, none_rows) for row in pair]
    batch = np.stack([
        cv2.cvtColor(row[0], cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        for row in batch_rows
    ]).astype(np.float32)
    np.save(ROOT / "heldout_test_batch10.npy", batch)
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(batch)
    cv_logits = np.asarray(net.forward(), dtype=np.float32).reshape(10, 2)
    torch_logits = model(torch.from_numpy(batch)).detach().numpy()
    parity_max_abs = float(np.max(np.abs(cv_logits - torch_logits)))
    np.save(ROOT / "heldout_test_logits_batch10.npy", torch_logits.astype(np.float32))

    run_meta = json.loads((ROOT / "training_run.json").read_text(encoding="utf-8"))
    report = {
        "status": "offline-trained deployment candidate; active B model not replaced",
        "task": "binary pad presence only around fixed calibrated points; original 10 pad labels merged",
        "source_dataset": str(train.DATA_ROOT),
        "source_frames": sum(len(v) for v in frames.values()),
        "source_split_by_video": run_meta["source_split_by_video"],
        "source_box_counts_by_original_class": run_meta["source_box_counts_by_original_class"],
        "source_split_frames_and_crops": split_counts,
        "source_reviewed_scene": "bare_torso_electrodes_visible",
        "roi_crop": {
            "size": train.PATCH_SIZE,
            "training_neighbor_scale": train.NEIGHBOR_SCALE,
            "side_rule": "nearest annotated pad-center spacing * 0.60, clamped as R3 checker",
            "positive_training_jitter": "center +/- 0.12 ROI side; scale 0.88-1.16",
            "negative_method": "unlabeled crops sampled around pad cluster, excluded from pad boxes with a margin, and gated to at least 30% loose YCrCb skin pixels",
        },
        "important_data_limit": "No explicitly annotated pad-absent frames exist. Negative crops are pseudo-negatives from unlabeled torso regions, so a missed/unlabeled pad could contaminate them.",
        "model": "TinyPatchNet four Conv2d/ReLU stride-2 blocks 3-16-24-32-48 + global pool + logits [pad,none]",
        "parameter_count": run_meta["parameter_count"],
        "patch_contract": {
            "input": [10, 3, train.PATCH_SIZE, train.PATCH_SIZE],
            "dtype": "float32 RGB [0,1]",
            "output": [10, 2],
            "class_order": ["pad", "none"],
        },
        "training_device": run_meta["device"],
        "epochs": run_meta["epochs"],
        "best_epoch": run_meta["best_epoch"],
        "class_weights": run_meta["class_weights"],
        "baseline_current_r3": {"val": baseline_val, "test_frozen_once": baseline_test},
        "candidate_opencv_dnn": {
            "val": candidate_val,
            "val_threshold_sweep": train.threshold_sweep(
                [row[1] for row in samples["val"]], val_probs),
            "test_frozen_once": candidate_test,
            "test_threshold_sweep": train.threshold_sweep(
                [row[1] for row in samples["test"]], test_probs),
        },
        "opencv_vs_pytorch_max_abs_logit_error": parity_max_abs,
        "onnx_checker": "passed",
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "onnx_bytes": onnx_path.stat().st_size,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "heldout_test_batch": str(ROOT / "heldout_test_batch10.npy"),
        "heldout_test_logits": str(ROOT / "heldout_test_logits_batch10.npy"),
        "crop_audit_montage": str(ROOT / "crop_audit_montage.jpg"),
        "history": run_meta["history"],
        "limitations": [
            "The train/validation/test split is by source video, but all six videos are from one person and one recording session.",
            "There are no fully empty/no-electrode frames; synthesized negatives are not equivalent to independent negative recordings.",
            "Crop-classifier metrics do not verify pose-point localization or actual electrode placement.",
            "This candidate preserves R3's local ROI search and output contract but has not been deployed to B.",
        ],
    }
    report_path = ROOT / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "onnx": str(onnx_path),
        "report": str(report_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "candidate_val": candidate_val,
        "candidate_test": candidate_test,
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "opencv_vs_pytorch_max_abs_logit_error": parity_max_abs,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
