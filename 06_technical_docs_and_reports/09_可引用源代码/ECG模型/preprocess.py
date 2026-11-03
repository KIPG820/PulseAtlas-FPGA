"""Versioned preprocessing shared by PTB-XL preparation and later deployment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILTER_MODEL_DIR = PROJECT_ROOT / "ecg_ip_357" / "ecg_ip_357" / "model"
if str(FILTER_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FILTER_MODEL_DIR))

from filter_ref import ClassicFilterModel, float_cascade  # noqa: E402

LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6")
_LEAD_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def canonicalize_lead_names(names) -> tuple[str, ...]:
    """Normalize WFDB's uppercase augmented-limb names without reordering."""
    canonical = []
    for name in names:
        upper = str(name).strip().upper()
        canonical.append(_LEAD_ALIASES.get(upper, upper))
    return tuple(canonical)


def load_contract(path: str | Path | None = None) -> dict:
    contract_path = Path(path) if path else Path(__file__).with_name("preprocess.json")
    with contract_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_record(signal_mV: np.ndarray, contract: dict | None = None) -> np.ndarray:
    """Filter a [samples, 12] ECG using IP3 quantized coefficients.

    A constant pad approximates the warmed-up streaming filter at the start
    of an independently stored record. Data remain in mV; no per-record
    normalization is performed.
    """
    cfg = contract or load_contract()
    x = np.asarray(signal_mV, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(LEAD_ORDER):
        raise ValueError(f"expected [samples, 12], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("record contains NaN or infinite values")
    filt_cfg = cfg["filter"]
    if not filt_cfg.get("enabled_stages"):
        return x.astype(np.float32)

    pad = int(round(float(filt_cfg.get("settling_pad_seconds", 2.0))
                    * float(cfg["sample_rate_hz"])))
    out = np.empty_like(x, dtype=np.float64)
    # float_cascade reads this object's fixed coefficients only; its streaming
    # state dictionaries are not mutated by that vectorized reference path.
    model = ClassicFilterModel(
        fs=float(cfg["sample_rate_hz"]),
        hp_fc=float(filt_cfg["highpass_hz"]),
        notch_mode=str(filt_cfg["notch_mode"]),
        lp_fc=float(filt_cfg["lowpass_hz"]),
        bypass=(),
        fir_taps=int(filt_cfg["fir_taps"]),
    )
    for lead in range(x.shape[1]):
        column = x[:, lead]
        padded = np.pad(column, (pad, 0), mode="edge") if pad else column
        out[:, lead] = float_cascade(padded, model)[pad:]
    return out.astype(np.float32)


def normalize(signal_mV: np.ndarray, mean: np.ndarray, std: np.ndarray,
              clip_sigma: float = 8.0) -> np.ndarray:
    """Apply fixed training-fold statistics to [12,samples] or [samples,12]."""
    x = np.asarray(signal_mV, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), 1e-8)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D ECG array, got {x.shape}")
    if x.shape[0] == 12:
        m, s = mean[:, None], std[:, None]
    elif x.shape[1] == 12:
        m, s = mean[None, :], std[None, :]
    else:
        raise ValueError(f"expected one axis to contain 12 leads, got {x.shape}")
    return np.clip((x - m) / s, -clip_sigma, clip_sigma).astype(np.float32)
