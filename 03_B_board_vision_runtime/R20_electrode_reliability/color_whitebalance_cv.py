"""Video-held-out CV for HSV color grouping with training-only light augmentation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = Path(r"D:\ECG-fpga\b_board\PulseAtlas_B_Vision_ConnectorIntegration_20261014")
sys.path.insert(0, str(SOURCE))
import evaluate_patch_color_grouped_cv as source
from train_patch_color_hsv import GROUPS as COLOR_GROUPS

ROOT = Path(__file__).resolve().parent
GROUPS = ["red", "yellow", "dark", "green", "purple"]


def transform(crop, gains_bgr, exposure, gamma):
    x = crop.astype(np.float32) / 255.0
    x *= np.asarray(gains_bgr, np.float32).reshape(1, 1, 3)
    x *= float(exposure)
    x = np.clip(x, 0.0, 1.0) ** float(gamma)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def feature(crop):
    return source._feature_variants(crop)["hsv_circular"]


def summarize(y, pred, groups, conf):
    acc = float(np.mean(y == pred))
    folds = []
    for group in sorted(set(groups)):
        mask = groups == group
        folds.append({"video_group": str(group), "samples": int(mask.sum()),
                      "accuracy": float(np.mean(y[mask] == pred[mask]))})
    return {"samples": len(y), "accuracy": acc,
            "worst_video_accuracy": min(row["accuracy"] for row in folds),
            "median_video_accuracy": float(np.median([row["accuracy"] for row in folds])),
            "mean_confidence": float(np.mean(conf)), "folds": folds}


def eval_color_crops(matcher, crops, labels, class_to_group, scenario):
    gains, exposure, gamma = scenario
    pred, truth, confidence = [], [], []
    for crop, label in zip(crops, labels):
        result = matcher.classify(transform(crop, gains, exposure, gamma))
        pred.append(result["detected_color_group"])
        label = str(label)
        truth.append(label if label in COLOR_GROUPS else class_to_group[label])
        confidence.append(float(result["color_confidence"]))
    return {"samples": len(truth), "accuracy": float(np.mean(np.asarray(truth) == np.asarray(pred))),
            "confidence_mean": float(np.mean(confidence))}


def main():
    rows, skipped = source._collect_training_objects()
    x = np.asarray([feature(row["crop"]) for row in rows], dtype=np.float32)
    y = np.asarray([row["group"] for row in rows], dtype=object)
    video_groups = np.asarray([row["video_group"] for row in rows], dtype=object)
    splitter = LeaveOneGroupOut()
    baseline_pred = np.empty(len(y), dtype=object)
    aug_pred = np.empty(len(y), dtype=object)
    baseline_conf = np.zeros(len(y), dtype=np.float32)
    aug_conf = np.zeros(len(y), dtype=np.float32)
    shifted = {name: {"baseline": [], "augmented": []}
               for name in ("dim", "warm", "cool", "dim_warm")}
    heldout_transforms = {
        "dim": ((1, 1, 1), 0.72, 1.0),
        "warm": ((0.88, 1.0, 1.10), 1.0, 1.0),
        "cool": ((1.10, 1.0, 0.88), 1.0, 1.0),
        "dim_warm": ((0.92, 1.0, 1.08), 0.82, 0.90),
    }
    transform_bank = [
        ((1, 1, 1), 0.78, 1.0),
        ((0.88, 1.0, 1.12), 1.0, 1.0),
        ((1.12, 1.0, 0.88), 1.0, 1.0),
        ((0.92, 1.0, 1.08), 0.86, 0.92),
        ((1.08, 1.0, 0.92), 1.12, 1.08),
    ]
    augmented_features = []
    for i, row in enumerate(rows):
        sample_features = [x[i]]
        for gains, exposure, gamma in transform_bank:
            sample_features.append(feature(transform(row["crop"], gains, exposure, gamma)))
        augmented_features.append(np.asarray(sample_features, dtype=np.float32))
    shifted_features = {
        name: np.asarray([feature(transform(row["crop"], gains, exposure, gamma))
                          for row in rows], dtype=np.float32)
        for name, (gains, exposure, gamma) in heldout_transforms.items()
    }
    for train_idx, test_idx in splitter.split(x, y, video_groups):
        baseline = make_pipeline(StandardScaler(),
                                 LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        baseline.fit(x[train_idx], y[train_idx])
        aug_x, aug_y = [], []
        for idx in train_idx:
            aug_x.extend(augmented_features[int(idx)])
            aug_y.extend([y[idx]] * len(augmented_features[int(idx)]))
        augmented = make_pipeline(StandardScaler(),
                                  LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        augmented.fit(np.asarray(aug_x, dtype=np.float32), np.asarray(aug_y, dtype=object))
        for model, pred, conf in ((baseline, baseline_pred, baseline_conf),
                                  (augmented, aug_pred, aug_conf)):
            probs = model.predict_proba(x[test_idx])
            classes = model.classes_
            pred[test_idx] = classes[np.argmax(probs, axis=1)]
            conf[test_idx] = np.max(probs, axis=1)
        for name, (gains, exposure, gamma) in heldout_transforms.items():
            features = shifted_features[name][test_idx]
            for model_name, model in (("baseline", baseline), ("augmented", augmented)):
                probs = model.predict_proba(features)
                shifted[name][model_name].extend(
                    (str(y[i]), str(model.classes_[np.argmax(p)]), float(np.max(p)))
                    for i, p in zip(test_idx, probs))

    output = {
        "experiment": "HSV circular LDA video-group CV with training-only synthetic photometric augmentation",
        "source_samples": len(y), "video_groups": int(len(set(video_groups))),
        "source_skipped": skipped,
        "augmentation_count_per_training_crop": len(transform_bank),
        "baseline_held_out_original": summarize(y, baseline_pred, video_groups, baseline_conf),
        "augmented_held_out_original": summarize(y, aug_pred, video_groups, aug_conf),
        "augmented_held_out_illumination": {},
        "limitations": ["All source images come from one person/capture family.",
                        "Lighting shifts are synthetic and do not replace camera captures.",
                        "Fold metrics evaluate color-group classification on annotated pad crops, not real placement mistakes."],
    }
    for name, records in shifted.items():
        output["augmented_held_out_illumination"][name] = {}
        for model_name, records_for_model in records.items():
            truth = np.asarray([item[0] for item in records_for_model], dtype=object)
            pred = np.asarray([item[1] for item in records_for_model], dtype=object)
            conf = np.asarray([item[2] for item in records_for_model], dtype=np.float32)
            output["augmented_held_out_illumination"][name][model_name] = {
                "samples": len(truth), "accuracy": float(np.mean(truth == pred)),
                "mean_confidence": float(np.mean(conf)),
            }

    # Fit the deployable JSON representation using only training split data.
    group_ids = np.asarray([list(COLOR_GROUPS).index(name) for name in y], dtype=np.int64)
    full_x = np.vstack(augmented_features)
    full_y = np.repeat(group_ids, len(augmented_features[0]))
    deploy_model = make_pipeline(StandardScaler(),
                                 LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    deploy_model.fit(full_x, full_y)
    scaler = deploy_model.named_steps["standardscaler"]
    lda = deploy_model.named_steps["lineardiscriminantanalysis"]
    template_path = SOURCE / "patch_color_groups_hsv_20261014.json"
    candidate = json.loads(template_path.read_text(encoding="utf-8"))
    candidate["scaler_mean"] = scaler.mean_.astype(float).tolist()
    candidate["scaler_scale"] = scaler.scale_.astype(float).tolist()
    candidate["lda_coefficients"] = lda.coef_.astype(float).tolist()
    candidate["lda_intercepts"] = lda.intercept_.astype(float).tolist()
    candidate["training_samples"] = int(len(full_y))
    candidate["photometric_augmentation"] = {
        "source_original_training_crops": int(len(y)),
        "augmented_variants_per_crop": len(transform_bank),
        "fit_split": "reviewed training videos only",
    }
    candidate_path = ROOT / "patch_color_groups_hsv_augmented_candidate.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    output["candidate_model"] = str(candidate_path)
    output["holdout_split_color_metrics"] = {}
    from patch_color_matcher import PatchColorMatcher
    class_to_group = candidate["class_to_group"]
    baseline_matcher = PatchColorMatcher(template_path)
    candidate_matcher = PatchColorMatcher(candidate_path)
    for split in ("val", "test"):
        data_path = SOURCE / ("patch_color_hsv_board_{}_20261014.npz".format(split))
        data = np.load(data_path)
        crops, labels = data["crops"], data["labels"]
        if len(crops) == 0:
            continue
        output["holdout_split_color_metrics"][split] = {}
        for scenario_name, scenario in {
            "nominal": ((1, 1, 1), 1.0, 1.0),
            "dim": ((1, 1, 1), 0.72, 1.0),
            "warm_white_balance": ((0.88, 1.0, 1.10), 1.0, 1.0),
            "cool_white_balance": ((1.10, 1.0, 0.88), 1.0, 1.0),
        }.items():
            output["holdout_split_color_metrics"][split][scenario_name] = {
                "baseline": eval_color_crops(baseline_matcher, crops, labels,
                                             class_to_group, scenario),
                "augmented": eval_color_crops(candidate_matcher, crops, labels,
                                              class_to_group, scenario),
            }
    out_path = ROOT / "color_whitebalance_cv_report.json"
    output["report"] = str(out_path)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
