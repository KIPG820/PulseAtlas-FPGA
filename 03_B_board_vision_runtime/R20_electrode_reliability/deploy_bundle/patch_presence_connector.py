"""Bridge calibrated-point pad presence to the existing B service result API."""

import math


LEAD_TARGET = {
    "R": "RA", "L": "LA", "N": "RL", "F": "LL",
    "C1": "V1", "C2": "V2", "C3": "V3",
    "C4": "V4", "C5": "V5", "C6": "V6",
}


def _point_xy(point):
    if not isinstance(point, dict):
        return None
    try:
        x = float(point.get("x_px"))
        y = float(point.get("y_px"))
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def summarize(result):
    counts = {status: sum(item["status"] == status
                         for item in result["per_lead"].values())
              for status in ("correct", "adjust", "unknown")}
    result.update(
        success_count=counts["correct"],
        adjust_count=counts["adjust"],
        unknown_count=counts["unknown"],
        all_success=counts["correct"] == len(LEAD_TARGET),
    )
    return result


def presence_to_placement_result(presence, pose_info, frame_id, width, height,
                                 age_seconds=0.0, max_age_seconds=2.5):
    """Map local visual presence to UI-compatible correct/adjust/unknown states.

    "correct" means the classifier sees an electrode-like pad near the
    calibration point. It does not prove contact, electrical connection, or
    clinical/anatomical placement.
    """
    # Age includes B's pose-model execution time before the local electrode
    # checker receives this captured frame. A one-second cutoff invalidates
    # every otherwise usable result when pose inference itself takes ~1 s.
    reason = None
    if not math.isfinite(float(age_seconds)) or age_seconds < 0 or age_seconds > max_age_seconds:
        reason = "stale_result"
    elif not pose_info or pose_info.get("status") != "valid_torso":
        reason = "no_pose"
    elif pose_info.get("frame_id") != frame_id:
        reason = "frame_mismatch"

    targets = {} if not pose_info else {
        **pose_info.get("torso_reference_points", {}),
        **pose_info.get("chest_reference_points", {}),
    }
    predictions = {} if not presence else presence.get("per_point", {})
    color_mismatch_count = 0
    result = {
        "frame_id": frame_id,
        "age_seconds": float(age_seconds),
        "per_lead": {},
        "meaning": "visual_pad_presence_and_color_group_check_near_calibration_point",
        "contact_or_connection_verified": False,
        "anatomical_placement_verified": False,
        "inference_ms": None if not presence else presence.get("inference_ms"),
    }

    for lead, label in LEAD_TARGET.items():
        target = targets.get(label)
        xy = _point_xy(target)
        candidate = predictions.get(label, {})
        item_reason = reason
        status = "unknown"
        pad_detected = False
        if item_reason is None:
            if xy is None:
                item_reason = "missing_point"
            elif not (0 <= xy[0] < width and 0 <= xy[1] < height):
                item_reason = "point_outside_frame"
            else:
                raw_status = candidate.get("status")
                if raw_status == "pad_color_mismatch":
                    status = "adjust"
                    item_reason = "pad_color_differs_from_expected_group"
                    color_mismatch_count += 1
                    pad_detected = True
                elif raw_status == "pad_pending":
                    status = "unknown"
                    item_reason = candidate.get("reason", "pad_presence_waiting_for_confirmation")
                elif raw_status == "pad_present":
                    status = "correct"
                    item_reason = candidate.get("reason", "pad_present_near_calibration_point")
                    pad_detected = True
                elif raw_status == "pad_not_found":
                    status = "adjust"
                    item_reason = "pad_not_found_in_local_roi"
                else:
                    item_reason = candidate.get("reason", "no_classifier_result")
        confidence = None
        if status != "unknown":
            if candidate.get("status") == "pad_color_mismatch":
                key = "color_confidence"
            else:
                key = "pad_probability" if status == "correct" else "no_pad_probability"
            try:
                confidence = float(candidate.get(key))
            except (TypeError, ValueError):
                confidence = None
        result["per_lead"][lead] = {
            "lead_id": lead,
            "target_label": label,
            "status": status,
            "pad_detected": pad_detected,
            "reason": item_reason,
            "target_xy": None if xy is None else [xy[0], xy[1]],
            "prediction_index": None,
            "bbox_xyxy": None,
            "confidence": confidence,
            "pad_probability": candidate.get("pad_probability"),
            "no_pad_probability": candidate.get("no_pad_probability"),
            "roi_side_px": candidate.get("roi_side_px"),
            "color_status": candidate.get("color_status"),
            "expected_color_group": candidate.get("expected_color_group"),
            "detected_color_group": candidate.get("detected_color_group"),
            "color_confidence": candidate.get("color_confidence"),
        }
    result["color_mismatch_count"] = color_mismatch_count
    return summarize(result)
