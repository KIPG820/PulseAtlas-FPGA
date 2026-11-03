"""Retrain B-board's fixed-ROI binary electrode presence model.

Uses the user's 259-frame YOLO annotations, merges all ten pad identity labels
into one ``pad`` class, and creates conservative hard negatives from unlabeled
torso regions. Video groups are kept intact across train/validation/test.
The model architecture and fixed [10, 3, 96, 96] ONNX interface match R3.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import argparse

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = Path(r"C:\Users\zsd88\Desktop\1\frames_review")
OUT = Path(__file__).resolve().parent
PATCH_SIZE = 96
NEIGHBOR_SCALE = 0.60
BATCH_SIZE = 64
EPOCHS = 60
SEED = 20261021
CLASSES = ("pad", "none")

VIDEO_SPLIT = {
    "20261018_144523_clothed_no_pads.avi": "train",
    "20261018_144409_clothed_no_pads.avi": "train",
    "20261018_144312_clothed_no_pads.avi": "val",
    "20261018_144128_clothed_no_pads.avi": "val",
    "20261018_144022_clothed_no_pads.avi": "test",
    "20261018_144217_clothed_no_pads.avi": "test",
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def crop_square_reflect(image: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(16, int(round(side)))
    x1, y1 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    x2, y2 = x1 + side, y1 + side
    pl, pt = max(0, -x1), max(0, -y1)
    pr, pb = max(0, x2 - w), max(0, y2 - h)
    if pl or pt or pr or pb:
        image = cv2.copyMakeBorder(image, pt, pb, pl, pr, cv2.BORDER_REFLECT_101)
        x1 += pl
        y1 += pt
    crop = image[y1:y1 + side, x1:x1 + side]
    if crop.shape[:2] != (side, side):
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)


def roi_side_for_box(index, boxes, frame_width):
    x, y = boxes[index]["cx"], boxes[index]["cy"]
    distances = [math.hypot(x - other["cx"], y - other["cy"])
                 for j, other in enumerate(boxes) if j != index]
    width = max(1, int(frame_width))
    min_side = max(24, int(round(width * 0.015)))
    max_side = max(min_side, min(320, int(round(width * 0.25))))
    side = int(round(min(distances) * NEIGHBOR_SCALE)) if distances else int(round(width * 0.08))
    return max(min_side, min(max_side, side))


def intersects(a, b, margin=0.0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 + margin and ax2 > bx1 - margin and ay1 < by2 + margin and ay2 > by1 - margin


def skin_fraction(bgr_crop):
    """Loose YCrCb skin-color gate used only to avoid background negatives."""
    ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = (y > 45) & (cr >= 128) & (cr <= 185) & (cb >= 65) & (cb <= 150)
    return float(mask.mean())


def load_frames_and_boxes():
    classes = [line.strip() for line in (DATA_ROOT / "labels" / "classes.txt").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    with (DATA_ROOT / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    frames = {"train": [], "val": [], "test": []}
    original_class_counts = Counter()
    for row in manifest:
        split = VIDEO_SPLIT.get(row["source_video"])
        if split is None:
            raise ValueError("No group split configured for " + row["source_video"])
        image_path = DATA_ROOT / Path(row["image_path"])
        relative = Path(row["image_path"]).relative_to("images")
        label_path = DATA_ROOT / "labels" / relative.with_suffix(".txt")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Cannot read image " + str(image_path))
        h, w = image.shape[:2]
        boxes = []
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("Bad YOLO row {}:{}".format(label_path, line_no))
            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            if not (0 <= class_id < len(classes) and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError("Invalid annotation in {}:{}".format(label_path, line_no))
            cx, cy = xc * w, yc * h
            box_w, box_h = bw * w, bh * h
            boxes.append({"class_id": class_id, "cx": cx, "cy": cy,
                          "x1": cx - box_w / 2, "y1": cy - box_h / 2,
                          "x2": cx + box_w / 2, "y2": cy + box_h / 2,
                          "box_w": box_w, "box_h": box_h})
            original_class_counts[classes[class_id]] += 1
        frames[split].append({"image": image, "boxes": boxes, "path": str(image_path),
                              "video": row["source_video"], "frame": row["frame_index"],
                              "scene": row.get("reviewed_scene_tag", "")})
    return frames, classes, dict(original_class_counts)


def sample_empty_centers(frame, rng, wanted):
    """Sample blank crops around the annotated pad cluster, never over a label.

    These are pseudo-negatives because no truly empty scene was annotated.
    The exclusion margin makes accidental partial-pad crops less likely.
    """
    image, boxes = frame["image"], frame["boxes"]
    h, w = image.shape[:2]
    if not boxes or wanted <= 0:
        return []
    sides = [roi_side_for_box(i, boxes, w) for i in range(len(boxes))]
    pad_extent = float(np.median([max(b["box_w"], b["box_h"]) for b in boxes]))
    xs = [b["cx"] for b in boxes]
    ys = [b["cy"] for b in boxes]
    # The torso area around a sparse one- or two-pad frame is much larger than
    # the labeled cluster. Expand by roughly two pad widths so we can find
    # realistic bare-skin negatives without sampling the far image corners.
    pad_x = max(2.0 * pad_extent, 0.50 * (max(xs) - min(xs)))
    pad_y = max(2.0 * pad_extent, 0.50 * (max(ys) - min(ys)))
    xlo, xhi = max(8, min(xs) - pad_x), min(w - 8, max(xs) + pad_x)
    ylo, yhi = max(8, min(ys) - pad_y), min(h - 8, max(ys) + pad_y)
    accepted = []
    attempts = 0
    max_attempts = max(250, wanted * 160)
    while len(accepted) < wanted and attempts < max_attempts:
        attempts += 1
        anchor_index = rng.randrange(len(boxes))
        side = sides[anchor_index]
        cx = rng.uniform(xlo, xhi) if xhi > xlo else (xlo + xhi) / 2
        cy = rng.uniform(ylo, yhi) if yhi > ylo else (ylo + yhi) / 2
        crop_box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        # Keep the crop near the body cluster and away from every annotated pad.
        if any(intersects(crop_box, (b["x1"], b["y1"], b["x2"], b["y2"]), margin=0.22 * side)
               for b in boxes):
            continue
        if any(math.hypot(cx - ox, cy - oy) < 0.6 * max(side, oside)
               for ox, oy, oside in accepted):
            continue
        # Prefer bare-skin crops so negative examples model an absent pad at a
        # candidate electrode site, rather than easy room/background patches.
        if skin_fraction(crop_square_reflect(image, cx, cy, side)) < 0.30:
            continue
        accepted.append((cx, cy, side))
    return accepted


def build_samples(frames):
    samples = {split: [] for split in frames}
    counts = {}
    for split, split_frames in frames.items():
        empty_negative_frames = 0
        for frame_index, frame in enumerate(split_frames):
            image, boxes = frame["image"], frame["boxes"]
            rng = random.Random(SEED + frame_index * 7919 + sum(map(ord, frame["video"])))
            width = image.shape[1]
            for box_index, box in enumerate(boxes):
                side = roi_side_for_box(box_index, boxes, width)
                variants = 2 if split == "train" else 1
                for variant in range(variants):
                    if split == "train":
                        scale = rng.uniform(0.88, 1.16)
                        jitter = 0.12 * side
                        cx = box["cx"] + rng.uniform(-jitter, jitter)
                        cy = box["cy"] + rng.uniform(-jitter, jitter)
                        crop_side = side * scale
                    else:
                        cx, cy, crop_side = box["cx"], box["cy"], side
                    crop = crop_square_reflect(image, cx, cy, crop_side)
                    samples[split].append((crop, 0, frame["path"], "pad"))

            # Make approximately one no-pad crop per annotated pad instance.
            negative_centers = sample_empty_centers(frame, rng, max(2, len(boxes)))
            if not negative_centers:
                empty_negative_frames += 1
            for cx, cy, side in negative_centers:
                crop = crop_square_reflect(image, cx, cy, side)
                samples[split].append((crop, 1, frame["path"], "none"))
        counts[split] = {
            "source_frames": len(split_frames),
            "videos": sorted({f["video"] for f in split_frames}),
            "positive_patches": sum(1 for _, y, _, _ in samples[split] if y == 0),
            "negative_patches": sum(1 for _, y, _, _ in samples[split] if y == 1),
            "frames_with_no_generated_negative": empty_negative_frames,
        }
    return samples, counts


class PatchSet(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, label, _, _ = self.rows[index]
        image = image.copy()
        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.92, 1.08)
            tx, ty = random.uniform(-0.05, 0.05) * PATCH_SIZE, random.uniform(-0.05, 0.05) * PATCH_SIZE
            mat = cv2.getRotationMatrix2D((PATCH_SIZE / 2, PATCH_SIZE / 2), angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty
            image = cv2.warpAffine(image, mat, (PATCH_SIZE, PATCH_SIZE),
                                   borderMode=cv2.BORDER_REFLECT_101)
            if random.random() < 0.85:
                image = cv2.convertScaleAbs(image, alpha=random.uniform(0.72, 1.30),
                                            beta=random.uniform(-24, 24))
            if random.random() < 0.45:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + random.randint(-5, 5)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) *
                                       random.uniform(0.72, 1.28), 0, 255).astype(np.uint8)
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        return tensor, label


class TinyPatchNet(nn.Module):
    """Exactly the compact network used by B R3 (24,842 parameters)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )
        self.head = nn.Linear(48, 2)

    def forward(self, x):
        return self.head(self.features(x))


def metrics_from_probs(truth, probs, threshold=0.5):
    truth = np.asarray(truth, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    # Label/output order is [pad=0, none=1], so a high pad probability maps to 0.
    pred = np.where(probs[:, 0] >= threshold, 0, 1).astype(np.int64)
    # Recode to matrix order [pad, none] and the deployed decision convention.
    cm = np.zeros((2, 2), dtype=np.int64)
    for y, p in zip(truth, pred):
        cm[int(y), int(p)] += 1
    pad_recall = float(cm[0, 0] / max(1, cm[0].sum()))
    none_recall = float(cm[1, 1] / max(1, cm[1].sum()))
    return {
        "count": int(len(truth)), "threshold_pad_probability": float(threshold),
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "balanced_accuracy": (pad_recall + none_recall) / 2,
        "pad_recall": pad_recall, "none_recall": none_recall,
        "false_pad_rate_on_none": float(cm[1, 0] / max(1, cm[1].sum())),
        "confusion_matrix_rows_true_pad_none_cols_pred_pad_none": cm.tolist(),
    }


@torch.no_grad()
def evaluate_torch(model, rows, device):
    loader = DataLoader(PatchSet(rows, augment=False), batch_size=128, shuffle=False, num_workers=0)
    truth, probabilities = [], []
    model.eval()
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        truth.extend(y.numpy().tolist())
        probabilities.extend(p.tolist())
    return metrics_from_probs(truth, probabilities), np.asarray(probabilities, dtype=np.float32)


def evaluate_opencv(model_path, rows):
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    truth, probs = [], []
    # The deployed model requires a fixed batch of ten; repeat last patch as R3 does.
    for start in range(0, len(rows), 10):
        batch_rows = rows[start:start + 10]
        batch = np.zeros((10, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for i, (image, label, _, _) in enumerate(batch_rows):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            batch[i] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            truth.append(label)
        for i in range(len(batch_rows), 10):
            batch[i] = batch[len(batch_rows) - 1]
        net.setInput(batch)
        logits = np.asarray(net.forward(), dtype=np.float32).reshape(10, 2)[:len(batch_rows)]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs.extend((e / e.sum(axis=1, keepdims=True)).tolist())
    return metrics_from_probs(truth, probs), np.asarray(probs, dtype=np.float32)


def write_audit_montage(samples):
    # One example per image and class, then uniformly sample across all videos.
    unique = {"pad": {}, "none": {}}
    for split in ("train", "val", "test"):
        for crop, label, path, class_name in samples[split]:
            key = (path, class_name)
            unique[class_name].setdefault(key, (crop, class_name, Path(path).name, split))
    selected = []
    for class_name in ("pad", "none"):
        rows = list(unique[class_name].values())
        if rows:
            indices = np.linspace(0, len(rows) - 1, min(50, len(rows)), dtype=np.int64)
            selected.extend(rows[int(i)] for i in indices)
    cols, cell, label_h = 10, 160, 26
    tile_h = cell + label_h
    canvas = np.full((math.ceil(len(selected) / cols) * tile_h, cols * cell, 3), 250, np.uint8)
    for i, (crop, label, filename, split) in enumerate(selected):
        x, y = (i % cols) * cell, (i // cols) * tile_h
        thumb = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_NEAREST)
        canvas[y:y + cell, x:x + cell] = thumb
        text = ("PAD" if label == "pad" else "NONE") + " " + split + " " + filename[:12]
        color = (20, 130, 20) if label == "pad" else (20, 20, 200)
        cv2.putText(canvas, text[:24], (x + 3, y + cell + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "crop_audit_montage.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def threshold_sweep(truth, probabilities):
    return [metrics_from_probs(truth, probabilities, t) for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true",
                        help="prepare and inspect crop samples without training")
    args = parser.parse_args()
    seed_everything()
    torch.backends.cudnn.benchmark = True
    frames, source_classes, original_class_counts = load_frames_and_boxes()
    samples, split_counts = build_samples(frames)
    write_audit_montage(samples)
    print(json.dumps({"source_class_order": source_classes,
                      "source_class_box_counts": original_class_counts,
                      "split_summary": split_counts,
                      "crop_audit": str(OUT / "crop_audit_montage.jpg")},
                     ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return

    # The candidate can be independently reverted because R3's active model is untouched.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    train_rows = samples["train"]
    y = np.asarray([row[1] for row in train_rows], dtype=np.int64)
    class_counts = np.bincount(y, minlength=2).astype(np.float32)
    class_weights = len(y) / (2.0 * np.maximum(class_counts, 1.0))
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.015)
    train_loader = DataLoader(PatchSet(train_rows, augment=True), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    best_score, best_loss, best_state, best_epoch = -1.0, float("inf"), None, 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, seen = 0.0, 0
        for x, target in train_loader:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            seen += len(target)
        scheduler.step()
        val_metrics, _ = evaluate_torch(model, samples["val"], device)
        score = val_metrics["balanced_accuracy"]
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(1, seen),
               "val_balanced_accuracy": score, "val_pad_recall": val_metrics["pad_recall"],
               "val_none_recall": val_metrics["none_recall"]}
        history.append(row)
        if score > best_score or (score == best_score and row["train_loss"] < best_loss):
            best_score, best_loss, best_epoch = score, row["train_loss"], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(json.dumps(row), flush=True)

    if best_state is None:
        raise RuntimeError("No trained checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint = OUT / "tiny_patch_presence_best.pt"
    torch.save({"state_dict": best_state, "classes": CLASSES, "patch_size": PATCH_SIZE,
                "architecture": "TinyPatchNetPresence"}, checkpoint)
    run_metadata = {
        "source_split_frames_and_crops": split_counts,
        "source_box_counts_by_original_class": original_class_counts,
        "source_split_by_video": {s: sorted({f["video"] for f in v}) for s, v in frames.items()},
        "device": str(device), "epochs": EPOCHS, "best_epoch": best_epoch,
        "class_weights": {CLASSES[i]: float(class_weights[i]) for i in range(2)},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
    }
    (OUT / "training_run.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if importlib.util.find_spec("onnx") is None:
        print(json.dumps({"checkpoint": str(checkpoint),
                          "finalization": "ONNX package is unavailable in the GPU training environment; run finalize_candidate.py with D:\\Anaconda\\python.exe"},
                         ensure_ascii=False), flush=True)
        return
    onnx_path = OUT / "tiny_patch_presence_batch10.onnx"
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(10, 3, PATCH_SIZE, PATCH_SIZE), str(onnx_path),
                      input_names=["patches"], output_names=["logits"], opset_version=11,
                      do_constant_folding=True, dynamo=False)

    # Validate the fixed-batch ONNX path against PyTorch and the current production model.
    if len(samples["test"]) < 10:
        raise RuntimeError("Held-out test requires at least 10 crops")
    baseline_model = (Path(r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ColorHybrid_HSV_R3_SourceAgeGateFix_20261014") /
                      "tiny_patch_presence_batch10.onnx")
    baseline_test, _ = evaluate_opencv(baseline_model, samples["test"])
    baseline_val, _ = evaluate_opencv(baseline_model, samples["val"])
    candidate_val, val_probs = evaluate_opencv(onnx_path, samples["val"])
    candidate_test, test_probs = evaluate_opencv(onnx_path, samples["test"])
    test_truth = [row[1] for row in samples["test"]]

    test_batch_rows = samples["test"][:10]
    test_batch = np.stack([
        cv2.cvtColor(row[0], cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        for row in test_batch_rows
    ])
    np.save(OUT / "heldout_test_batch10.npy", test_batch.astype(np.float32))
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(test_batch)
    cv_logits = np.asarray(net.forward(), dtype=np.float32).reshape(10, 2)
    torch_logits = model(torch.from_numpy(test_batch)).detach().numpy()
    parity_max_abs = float(np.max(np.abs(cv_logits - torch_logits)))

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    report = {
        "status": "candidate trained offline; active B model not replaced",
        "task": "binary presence only around fixed calibrated points; original 10 pad labels merged",
        "source_dataset": str(DATA_ROOT),
        "source_frames": sum(len(v) for v in frames.values()),
        "source_split_by_video": {s: sorted({f["video"] for f in v}) for s, v in frames.items()},
        "source_box_counts_by_original_class": original_class_counts,
        "source_split_frames_and_crops": split_counts,
        "source_reviewed_scene": "bare_torso_electrodes_visible",
        "roi_crop": {"size": PATCH_SIZE, "training_neighbor_scale": NEIGHBOR_SCALE,
                     "side_rule": "nearest annotated pad-center spacing * 0.60, clamped as R3 checker",
                     "positive_training_jitter": "center +/- 0.12 ROI side; scale 0.88-1.16",
                     "negative_method": "unlabeled crops sampled around pad cluster, excluded from all pad boxes with a margin, and gated to at least 30% loose YCrCb skin pixels"},
        "important_data_limit": "There are no explicitly annotated pad-absent frames. Negative crops are pseudo-negatives from unlabeled torso regions, so a missed/unlabeled pad could contaminate them.",
        "model": "TinyPatchNet four Conv2d/ReLU stride-2 blocks 3-16-24-32-48 + global pool + two logits [pad,none]",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "patch_contract": {"input": [10, 3, PATCH_SIZE, PATCH_SIZE], "dtype": "float32 RGB [0,1]", "output": [10, 2], "class_order": list(CLASSES)},
        "device": str(device), "epochs": EPOCHS, "best_epoch": best_epoch,
        "class_weights": {CLASSES[i]: float(class_weights[i]) for i in range(2)},
        "baseline_current_r3": {"val": baseline_val, "test_frozen_once": baseline_test},
        "candidate_opencv_dnn": {"val": candidate_val, "val_threshold_sweep": threshold_sweep(
            [r[1] for r in samples["val"]], val_probs),
            "test_frozen_once": candidate_test,
            "test_threshold_sweep": threshold_sweep(test_truth, test_probs)},
        "opencv_vs_pytorch_max_abs_logit_error": parity_max_abs,
        "onnx_checker": "passed",
        "onnx_sha256": onnx_sha,
        "onnx_bytes": onnx_path.stat().st_size,
        "checkpoint": str(checkpoint), "onnx": str(onnx_path),
        "frozen_test_batch": str(OUT / "heldout_test_batch10.npy"),
        "crop_audit_montage": str(OUT / "crop_audit_montage.jpg"),
        "history": history,
        "limitations": [
            "The train/validation/test split is by source video, but all six videos are from one person and one recording session.",
            "There are no fully empty/no-electrode frames; synthesized negatives are not equivalent to independent negative recordings.",
            "Crop classifier metrics do not verify pose-point localization or actual electrode placement.",
            "This candidate preserves R3's local ROI search and output contract but has not been deployed to B.",
        ],
    }
    report_path = OUT / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "onnx": str(onnx_path),
                      "candidate_val": candidate_val, "candidate_test": candidate_test,
                      "baseline_val": baseline_val, "baseline_test": baseline_test,
                      "onnx_bytes": onnx_path.stat().st_size,
                      "opencv_vs_pytorch_max_abs_logit_error": parity_max_abs},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
