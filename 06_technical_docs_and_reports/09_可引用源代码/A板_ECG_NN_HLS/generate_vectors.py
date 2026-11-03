"""Export reproducible fold-9 input/output vectors for integer HLS simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from integer_reference import IntegerECGReference, ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 333, 1024, 2000])
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "nn" / "pl" / "golden_vectors")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = ROOT / "nn" / "data" / "cache"
    artifacts = ROOT / "nn" / "artifacts"
    raw = np.load(cache / "fold_9_x.npy", mmap_mode="r")
    record_ids = np.load(cache / "fold_9_ids.npy", allow_pickle=False)
    labels = np.load(cache / "fold_9_y.npy", allow_pickle=False)
    stats = json.loads((cache / "normalization.json").read_text(encoding="utf-8"))
    mean = torch.tensor(stats["mean_mV"], dtype=torch.float32).view(1, 12, 1)
    std = torch.tensor(stats["std_mV"], dtype=torch.float32).view(1, 12, 1)
    reference = IntegerECGReference()
    model = torch.jit.load(str(artifacts / "model_ptq_int8_x86.ts")).eval()
    entries = []

    for index in args.indices:
        if not 0 <= index < len(raw):
            parser.error("index {} is outside fold 9".format(index))
        x = torch.from_numpy(np.array(raw[index:index + 1], copy=True))
        normalized = ((x - mean) / std).clamp(-8.0, 8.0)
        input_codes = reference.quantize_input(normalized)
        output_codes = reference.run_codes(input_codes).flatten().numpy().astype(np.uint8)
        with torch.inference_mode():
            torchscript_logits = model(normalized).flatten().numpy()
        expected_float = reference.dequantize_output(torch.from_numpy(output_codes)).numpy()
        output_scale = float(reference.final_layer["output_scale"])
        output_zero_point = int(reference.final_layer["output_zero_point"])
        torchscript_codes = np.clip(
            np.rint(torchscript_logits / output_scale + output_zero_point), 0, 255
        ).astype(np.uint8)
        prefix = "fold9_{:04d}".format(index)
        input_path = args.output_dir / (prefix + "_input_time_major_u8.bin")
        output_path = args.output_dir / (prefix + "_expected_output_u8.bin")
        input_bytes = input_codes.squeeze(0).T.contiguous().numpy().tobytes()
        input_path.write_bytes(input_bytes)
        output_path.write_bytes(output_codes.tobytes())
        entries.append({
            "fold9_index": int(index),
            "ptbxl_ecg_id": int(record_ids[index]),
            "ground_truth_labels": labels[index].astype(int).tolist(),
            "input_file": input_path.name,
            "input_sha256": sha256_file(input_path),
            "input_bytes": len(input_bytes),
            "output_file": output_path.name,
            "output_sha256": sha256_file(output_path),
            "reference_qcodes": output_codes.tolist(),
            "torchscript_qcodes": torchscript_codes.tolist(),
            "qcode_delta": (output_codes.astype(int) - torchscript_codes.astype(int)).tolist(),
            "reference_logits": expected_float.tolist(),
            "torchscript_logits": torchscript_logits.tolist(),
        })

    manifest = {
        "source": "PTB-XL 1.0.3 official fold 9 validation cache; CC BY 4.0",
        "shape": [5000, 12],
        "input_encoding": "affine uint8, time-major, one byte per lead per sample",
        "output_encoding": "five affine uint8 logit codes in NORM,MI,STTC,CD,HYP order",
        "output_scale": float(reference.final_layer["output_scale"]),
        "output_zero_point": int(reference.final_layer["output_zero_point"]),
        "contract_sha256": sha256_file(artifacts / "model_ptq_int8_hw.json"),
        "weights_sha256": sha256_file(artifacts / "model_ptq_int8_weights.npz"),
        "torchscript_sha256": sha256_file(artifacts / "model_ptq_int8_x86.ts"),
        "vectors": entries,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(path),
        "vectors": len(entries),
        "all_exact_qcode_match": all(
            entry["reference_qcodes"] == entry["torchscript_qcodes"]
            for entry in entries
        ),
        "max_abs_qcode_delta": max(
            max(abs(v) for v in entry["qcode_delta"]) for entry in entries
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
