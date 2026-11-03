"""Calibrate and evaluate an INT8 post-training quantized ECGResNet1D.

The frozen FP32 checkpoint is never modified. Calibration uses a deterministic
sample from training folds 1-8; fold 9 is used only for the before/after report.
This x86 quantized model is a numerical reference, not a bitstream or an FPGA
deployment package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.fx import Node
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nn"))
sys.path.insert(0, str(ROOT / "nn" / "scripts"))

from models import ECGResNet1D  # noqa: E402
from train import ECGCache, LABELS, load_cache_normalization, normalize_batch, score  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_metrics(y: np.ndarray, probabilities: np.ndarray,
                      thresholds: np.ndarray) -> dict:
    output = {}
    for index, label in enumerate(LABELS):
        predicted = probabilities[:, index] >= thresholds[index]
        truth = y[:, index].astype(bool)
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        tn = int(np.sum(~predicted & ~truth))
        output[label] = {
            "threshold_from_frozen_fp32_fold9": float(thresholds[index]),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        }
    return output


@torch.inference_mode()
def infer(model, loader, mean, std, clip: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = normalize_batch(x, mean, std, clip)
        logits.append(model(x).float().cpu().numpy())
        labels.append(y.numpy())
    if not labels:
        raise RuntimeError("evaluation split is empty")
    return np.concatenate(logits), np.concatenate(labels).astype(np.uint8)


@torch.inference_mode()
def benchmark(model, example: torch.Tensor, repeats: int = 80) -> dict:
    """Model-only CPU batch-1 latency; excludes input normalization and I/O."""
    model.eval()
    for _ in range(8):
        model(example)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(example)
        timings.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "batch_size": 1,
        "warmup_runs": 8,
        "measured_runs": repeats,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def quantized_weight_summary(model) -> dict:
    """Report deploy-relevant integer weights and their per-channel scales."""
    layers = []
    total_weight_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.ao.nn.quantized.Conv1d,
                               torch.ao.nn.quantized.Linear)):
            weight = module.weight()
            raw = weight.int_repr()
            total_weight_count += int(raw.numel())
            entry = {
                "name": name,
                "operator": type(module).__name__,
                "weight_shape": list(raw.shape),
                "weight_dtype": str(raw.dtype),
                "weight_zero_points": weight.q_per_channel_zero_points().tolist()
                if weight.qscheme() in (torch.per_channel_affine,
                                        torch.per_channel_symmetric)
                else [int(weight.q_zero_point())],
                "weight_scales": weight.q_per_channel_scales().tolist()
                if weight.qscheme() in (torch.per_channel_affine,
                                        torch.per_channel_symmetric)
                else [float(weight.q_scale())],
                "output_scale": float(module.scale),
                "output_zero_point": int(module.zero_point),
            }
            bias = module.bias()
            if bias is not None:
                entry["bias_shape"] = list(bias.shape)
                entry["bias_dtype"] = str(bias.dtype)
            layers.append(entry)
    return {"int8_weight_count": total_weight_count, "layers": layers}


def is_quantized_weight_layer(module) -> bool:
    return isinstance(module, (torch.ao.nn.quantized.Conv1d,
                               torch.ao.nn.quantized.Linear))


def resolve_node_value(model, value):
    if not isinstance(value, Node):
        return value
    if value.op != "get_attr":
        raise ValueError("expected a graph constant, got {}".format(value.name))
    result = model
    for part in str(value.target).split("."):
        result = getattr(result, part)
    return result.item() if isinstance(result, torch.Tensor) else result


def node_output_qparams(model, node, visited=None):
    """Resolve affine activation scale/zero-point through quantized passthroughs."""
    if not isinstance(node, Node):
        return None
    visited = set() if visited is None else visited
    if node.name in visited:
        return None
    visited.add(node.name)
    if node.op == "call_function" and "quantize_per_tensor" in str(node.target):
        return (float(resolve_node_value(model, node.args[1])),
                int(resolve_node_value(model, node.args[2])))
    if node.op == "call_function" and "quantized.add_relu" in str(node.target):
        return (float(resolve_node_value(model, node.args[2])),
                int(resolve_node_value(model, node.args[3])))
    if node.op == "call_module":
        module = model.get_submodule(str(node.target))
        if is_quantized_weight_layer(module):
            return float(module.scale), int(module.zero_point)
    for arg in node.args:
        if isinstance(arg, Node):
            qparams = node_output_qparams(model, arg, visited)
            if qparams is not None:
                return qparams
    return None


def export_hardware_parameters(model, artifact_dir: Path) -> tuple[Path, Path, dict]:
    """Write an x86-independent INT8 tensor archive and layer/graph contract."""
    tensor_path = artifact_dir / "model_ptq_int8_weights.npz"
    contract_path = artifact_dir / "model_ptq_int8_hw.json"
    arrays = {}
    layers = []
    modules = dict(model.named_modules())
    graph_ops = []

    for node in model.graph.nodes:
        if node.op == "call_module":
            module = modules[str(node.target)]
            if is_quantized_weight_layer(module):
                weight = module.weight()
                if weight.qscheme() in (torch.per_channel_affine,
                                        torch.per_channel_symmetric):
                    weight_scales = weight.q_per_channel_scales().cpu().numpy().astype(np.float32)
                    weight_zero_points = weight.q_per_channel_zero_points().cpu().numpy().astype(np.int32)
                    weight_axis = int(weight.q_per_channel_axis())
                else:
                    weight_scales = np.asarray([weight.q_scale()], dtype=np.float32)
                    weight_zero_points = np.asarray([weight.q_zero_point()], dtype=np.int32)
                    weight_axis = 0
                input_qparams = node_output_qparams(model, node.args[0])
                if input_qparams is None:
                    raise RuntimeError("cannot determine input quantization for {}".format(node.target))
                input_scale, input_zero_point = input_qparams
                bias = module.bias()
                if bias is None:
                    bias_int32 = np.zeros(int(weight.shape[0]), dtype=np.int32)
                else:
                    bias_np = bias.detach().cpu().numpy().astype(np.float64)
                    if len(weight_scales) == 1:
                        effective_scales = np.full_like(bias_np, input_scale * weight_scales[0])
                    else:
                        effective_scales = input_scale * weight_scales
                    bias_int32 = np.rint(bias_np / effective_scales).astype(np.int32)

                key = str(node.target).replace(".", "_")
                arrays[key + "_weight_int8"] = weight.int_repr().cpu().numpy().astype(np.int8)
                arrays[key + "_weight_scales_f32"] = weight_scales
                arrays[key + "_weight_zero_points_i32"] = weight_zero_points
                arrays[key + "_bias_int32"] = bias_int32
                output_scale, output_zero_point = float(module.scale), int(module.zero_point)
                layer = {
                    "node": node.name,
                    "module": str(node.target),
                    "operator": type(module).__name__,
                    "input_node": node.args[0].name if isinstance(node.args[0], Node) else None,
                    "input_scale": float(input_scale),
                    "input_zero_point": int(input_zero_point),
                    "output_scale": output_scale,
                    "output_zero_point": output_zero_point,
                    "weight_shape": list(weight.shape),
                    "weight_axis": weight_axis,
                    "weight_zero_points": weight_zero_points.tolist(),
                    "weight_scales": weight_scales.tolist(),
                    "bias_quantization": "round(bias / (input_scale * per_output_weight_scale))",
                    "bias_int32_array": key + "_bias_int32",
                    "weight_int8_array": key + "_weight_int8",
                    "weight_scale_array": key + "_weight_scales_f32",
                    "weight_zero_point_array": key + "_weight_zero_points_i32",
                    "requantization_multiplier": (
                        (input_scale * weight_scales / output_scale).astype(np.float64).tolist()
                        if len(weight_scales) > 1
                        else [float(input_scale * weight_scales[0] / output_scale)]
                    ),
                    "activation": {
                        "stride": list(module.stride) if hasattr(module, "stride") else None,
                        "padding": list(module.padding) if hasattr(module, "padding") else None,
                        "dilation": list(module.dilation) if hasattr(module, "dilation") else None,
                        "groups": int(module.groups) if hasattr(module, "groups") else None,
                    },
                }
                layers.append(layer)
                graph_ops.append({"op": "quantized_layer", "node": node.name,
                                  "module": str(node.target)})
            elif str(node.target) == "pool":
                graph_ops.append({
                    "op": "adaptive_avg_pool1d",
                    "node": node.name,
                    "input_node": node.args[0].name if isinstance(node.args[0], Node) else None,
                    "output_size": 1,
                })
            elif isinstance(module, torch.nn.Identity):
                graph_ops.append({
                    "op": "identity",
                    "node": node.name,
                    "input_node": node.args[0].name if isinstance(node.args[0], Node) else None,
                })
            elif str(node.target) == "dropout":
                graph_ops.append({
                    "op": "dropout_identity_in_eval",
                    "node": node.name,
                    "input_node": node.args[0].name if isinstance(node.args[0], Node) else None,
                })
        elif node.op == "call_function" and "quantized.add_relu" in str(node.target):
            graph_ops.append({
                "op": "int8_add_then_relu",
                "node": node.name,
                "input_nodes": [arg.name if isinstance(arg, Node) else str(arg)
                                for arg in node.args[:2]],
                "output_scale": float(resolve_node_value(model, node.args[2])),
                "output_zero_point": int(resolve_node_value(model, node.args[3])),
            })
        elif node.op == "call_function" and "quantize_per_tensor" in str(node.target):
            graph_ops.append({
                "op": "input_affine_quantize",
                "node": node.name,
                "input_scale": float(resolve_node_value(model, node.args[1])),
                "input_zero_point": int(resolve_node_value(model, node.args[2])),
                "input_dtype": str(resolve_node_value(model, node.args[3])),
            })
        elif node.op == "call_method" and node.target == "squeeze":
            graph_ops.append({"op": "squeeze", "node": node.name,
                              "input_node": node.args[0].name if isinstance(node.args[0], Node) else None,
                              "dimension": node.args[1] if len(node.args) > 1 else None})
        elif node.op == "call_method" and node.target == "dequantize":
            graph_ops.append({
                "op": "dequantize_output",
                "node": node.name,
                "input_node": node.args[0].name if isinstance(node.args[0], Node) else None,
            })
        elif node.op == "output":
            graph_ops.append({"op": "model_output", "node": node.name,
                              "input_node": node.args[0].name if isinstance(node.args[0], Node) else None})

    if not layers:
        raise RuntimeError("no quantized Conv1d/Linear weights were found to export")
    np.savez_compressed(tensor_path, **arrays)
    contract = {
        "format": "ECG FPGA INT8 parameter/quantization contract v1",
        "status": "reference export; verify arithmetic and rounding against RTL/HLS simulation",
        "integer_convention": {
            "weights": "signed int8, per-output-channel symmetric (zero point expected 0)",
            "activations": "affine uint8 for this PyTorch x86 PTQ calibration",
            "accumulator": "int32 MAC; bias exported as int32 in input_scale * weight_scale units",
            "requantization": "accumulator * input_scale * weight_scale / output_scale, then add output_zero_point and saturate",
            "rounding": "bias uses NumPy round-to-nearest-even; RTL must match or compare and adjust",
        },
        "input_shape": [1, 12, 5000],
        "output_labels": list(LABELS),
        "weights_file": tensor_path.name,
        "weights_file_bytes": int(tensor_path.stat().st_size),
        "quantized_layers": layers,
        "graph_ops": graph_ops,
        "limitations": [
            "Quantized convolution operator ordering/padding must match PyTorch semantics.",
            "Residual add requantization and global average pooling need bit-accurate verification.",
            "This contract is not a bitstream and does not establish FPGA timing or resource use.",
        ],
    }
    temporary = contract_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    temporary.replace(contract_path)
    return tensor_path, contract_path, {"layers": len(layers), "arrays": len(arrays)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "nn" / "configs" / "train_ptbxl.yaml")
    parser.add_argument("--calibration-records", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only the PTQ artifacts; never the frozen FP32 checkpoint")
    args = parser.parse_args()
    if args.calibration_records < 1 or args.batch_size < 1:
        parser.error("calibration-records and batch-size must be positive")

    torch.set_num_threads(1)
    torch.backends.quantized.engine = "x86"
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache_dir = (ROOT / cfg["cache_dir"]).resolve()
    artifact_dir = (ROOT / cfg["artifact_dir"]).resolve()
    checkpoint_path = artifact_dir / "model_fp32.pt"
    quant_model_path = artifact_dir / "model_ptq_int8_x86.ts"
    report_path = artifact_dir / "ptq_int8_fold9_metrics.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (cache_dir / "fold_8_x.npy").is_file() or not (cache_dir / "fold_9_x.npy").is_file():
        raise FileNotFoundError("PTB-XL cache folds 1-9 are required for PTQ")
    if not args.overwrite and (quant_model_path.exists() or report_path.exists()):
        raise FileExistsError("PTQ artifacts already exist; pass --overwrite to replace them")

    checkpoint_hash_before = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("checkpoint label order does not match the fixed five-label task")

    mean, std = load_cache_normalization(cache_dir)
    clip = float(cfg["normalization_clip"])
    training_ds = ECGCache(cache_dir, list(range(1, 9)))
    rng = np.random.default_rng(args.seed)
    calibration_count = min(args.calibration_records, len(training_ds))
    calibration_indices = rng.choice(len(training_ds), size=calibration_count, replace=False)
    calibration_loader = DataLoader(
        Subset(training_ds, calibration_indices.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    float_model = ECGResNet1D().cpu().eval()
    float_model.load_state_dict(checkpoint["state_dict"])
    example = next(iter(calibration_loader))[0][:1]
    example = normalize_batch(example, mean, std, clip).contiguous()

    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig("x86"))
    prepared = prepare_fx(float_model, qconfig_mapping, (example,))
    with torch.inference_mode():
        for x, _ in calibration_loader:
            prepared(normalize_batch(x, mean, std, clip).contiguous())
    int8_model = convert_fx(prepared).eval()

    validation_ds = ECGCache(cache_dir, [9])
    validation_loader = DataLoader(
        validation_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    fp32_logits, y = infer(float_model, validation_loader, mean, std, clip)
    int8_logits, int8_y = infer(int8_model, validation_loader, mean, std, clip)
    if not np.array_equal(y, int8_y):
        raise RuntimeError("FP32 and INT8 evaluation labels differ")
    fp32_prob, fp32_per_class, fp32_auc, fp32_ap = score(fp32_logits, y)
    int8_prob, int8_per_class, int8_auc, int8_ap = score(int8_logits, y)

    thresholds = np.asarray(checkpoint["fold9_thresholds"], dtype=np.float32)
    if thresholds.shape != (len(LABELS),):
        raise ValueError("checkpoint is missing the frozen FP32 fold-9 thresholds")
    weight_summary = quantized_weight_summary(int8_model)
    hardware_weights_path, hardware_contract_path, hardware_export = export_hardware_parameters(
        int8_model, artifact_dir
    )
    script_model = torch.jit.trace(int8_model, example, check_trace=True)
    script_model.save(str(quant_model_path))
    # Re-open the saved artifact and verify it executes and agrees with FX output.
    reloaded = torch.jit.load(str(quant_model_path)).eval()
    with torch.inference_mode():
        export_logits = reloaded(example).float().numpy()
    if not np.allclose(export_logits, int8_model(example).float().numpy(), rtol=0, atol=1e-6):
        raise RuntimeError("saved TorchScript INT8 model failed output-parity verification")

    baseline_example = example.contiguous()
    fp32_latency = benchmark(float_model, baseline_example)
    int8_latency = benchmark(int8_model, baseline_example)
    report = {
        "method": "PyTorch FX post-training static INT8 quantization",
        "backend": "x86 (oneDNN/FBGEMM CPU reference)",
        "not_an_fpga_bitstream": True,
        "task": "PTB-XL five-label record classification",
        "validation_split": "official fold 9; used only for comparison, no calibration or threshold fitting",
        "calibration_split": "official training folds 1-8",
        "calibration_records": int(calibration_count),
        "calibration_seed": int(args.seed),
        "calibration_batch_size": int(args.batch_size),
        "torch_version": torch.__version__,
        "input_shape": [1, 12, 5000],
        "labels": list(LABELS),
        "fp32_checkpoint_sha256": checkpoint_hash_before,
        "fp32_fold9": {
            "macro_auroc": fp32_auc,
            "macro_auprc": fp32_ap,
            "per_class": fp32_per_class,
            "frozen_threshold_metrics": threshold_metrics(y, fp32_prob, thresholds),
            "model_only_cpu_latency": fp32_latency,
        },
        "int8_fold9": {
            "macro_auroc": int8_auc,
            "macro_auprc": int8_ap,
            "per_class": int8_per_class,
            "frozen_threshold_metrics": threshold_metrics(y, int8_prob, thresholds),
            "model_only_cpu_latency": int8_latency,
            "max_abs_logit_delta_vs_fp32": float(np.max(np.abs(int8_logits - fp32_logits))),
            "mean_abs_logit_delta_vs_fp32": float(np.mean(np.abs(int8_logits - fp32_logits))),
        },
        "metric_delta_int8_minus_fp32": {
            "macro_auroc": float(int8_auc - fp32_auc),
            "macro_auprc": float(int8_ap - fp32_ap),
        },
        "int8_weights": weight_summary,
        "quantized_torchscript_file": quant_model_path.name,
        "quantized_torchscript_bytes": int(quant_model_path.stat().st_size),
        "hardware_reference_export": {
            "weights_file": hardware_weights_path.name,
            "weights_file_bytes": int(hardware_weights_path.stat().st_size),
            "contract_file": hardware_contract_path.name,
            "quantized_layer_count": hardware_export["layers"],
            "array_count": hardware_export["arrays"],
        },
        "note": "x86 runtime latency is not a prediction of FPGA latency. Repeat calibration/validation before making a hardware precision choice.",
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    if sha256_file(checkpoint_path) != checkpoint_hash_before:
        raise RuntimeError("frozen FP32 checkpoint changed during PTQ")
    print(json.dumps({
        "saved_model": str(quant_model_path),
        "saved_report": str(report_path),
        "fp32_macro_auroc": fp32_auc,
        "int8_macro_auroc": int8_auc,
        "delta_macro_auroc": int8_auc - fp32_auc,
        "fp32_macro_auprc": fp32_ap,
        "int8_macro_auprc": int8_ap,
        "delta_macro_auprc": int8_ap - fp32_ap,
        "checkpoint_unchanged": True,
        "int8_torchscript_bytes": quant_model_path.stat().st_size,
        "hardware_weights": hardware_weights_path.name,
        "hardware_contract": hardware_contract_path.name,
    }, indent=2))


if __name__ == "__main__":
    main()
