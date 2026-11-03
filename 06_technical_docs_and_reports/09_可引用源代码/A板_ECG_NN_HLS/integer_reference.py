"""Integer inference reference for the exported 12-lead ECG PTQ network.

The input and every activation are uint8 affine codes. Weights are signed
int8, accumulators/biases are int32, and requantization uses a signed Q31
multiplier with round-to-nearest-even. This module also produces fixed vectors
for HLS C simulation. It does not claim to duplicate x86 kernel rounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "nn" / "artifacts" / "model_ptq_int8_hw.json"


def round_shift_nearest_even(value: torch.Tensor, bits: int = 31) -> torch.Tensor:
    """Signed integer divide by 2**bits, rounding ties to an even result."""
    value = value.to(torch.int64)
    magnitude = value.abs()
    whole = magnitude >> bits
    remainder = magnitude & ((1 << bits) - 1)
    half = 1 << (bits - 1)
    increment = (remainder > half) | ((remainder == half) & ((whole & 1) == 1))
    rounded = whole + increment.to(torch.int64)
    return torch.where(value < 0, -rounded, rounded)


def round_divide_nearest_even(value: torch.Tensor, divisor: int) -> torch.Tensor:
    if divisor < 1:
        raise ValueError("divisor must be positive")
    whole = torch.div(value, divisor, rounding_mode="floor")
    remainder = value - whole * divisor
    increment = (2 * remainder > divisor) | (
        (2 * remainder == divisor) & ((whole & 1) == 1)
    )
    return whole + increment.to(torch.int64)


class IntegerECGReference:
    def __init__(self, contract_path: Path = DEFAULT_CONTRACT):
        self.contract_path = Path(contract_path)
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.arrays = np.load(self.contract_path.parent / self.contract["weights_file"])
        self.layers = {layer["node"]: layer for layer in self.contract["quantized_layers"]}
        self.graph = self.contract["graph_ops"]
        if self.graph[0]["op"] != "input_affine_quantize":
            raise ValueError("the graph must start with affine input quantization")
        self.input_scale = float(self.graph[0]["input_scale"])
        self.input_zero_point = int(self.graph[0]["input_zero_point"])
        self.final_layer = self.contract["quantized_layers"][-1]
        torch.set_num_threads(1)

    def quantize_input(self, normalized: torch.Tensor) -> torch.Tensor:
        if tuple(normalized.shape) != (1, 12, 5000):
            raise ValueError("expected one normalized [1,12,5000] ECG window")
        return torch.quantize_per_tensor(
            normalized.cpu().float(), self.input_scale,
            self.input_zero_point, torch.quint8,
        ).int_repr()

    def _layer(self, layer: dict, source: torch.Tensor) -> torch.Tensor:
        key = layer["module"].replace(".", "_")
        weight = torch.from_numpy(self.arrays[key + "_weight_int8"].astype(np.float64))
        bias = torch.from_numpy(self.arrays[key + "_bias_int32"].astype(np.int64))
        centered = source.to(torch.float64) - float(layer["input_zero_point"])
        if len(layer["weight_shape"]) == 3:
            setting = layer["activation"]
            acc = F.conv1d(
                centered, weight,
                stride=int(setting["stride"][0]),
                padding=int(setting["padding"][0]),
                dilation=int(setting["dilation"][0]),
                groups=int(setting["groups"]),
            ).round().to(torch.int64)
            acc += bias.view(1, -1, 1)
        else:
            acc = F.linear(centered, weight).round().to(torch.int64)
            acc += bias.view(1, -1)
        multiplier = np.rint(
            np.asarray(layer["requantization_multiplier"], dtype=np.float64)
            * (1 << 31)
        ).astype(np.int64)
        if len(multiplier) == 1:
            multiplier = np.repeat(multiplier, int(layer["weight_shape"][0]))
        shape = (1, -1, 1) if acc.ndim == 3 else (1, -1)
        product = acc * torch.from_numpy(multiplier).view(*shape)
        quantized = round_shift_nearest_even(product) + int(layer["output_zero_point"])
        if "ConvReLU" in layer["operator"]:
            quantized = quantized.clamp_min(int(layer["output_zero_point"]))
        return quantized.clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _add_relu(a: torch.Tensor, a_q: tuple[float, int],
                  b: torch.Tensor, b_q: tuple[float, int],
                  scale: float, zero_point: int) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError("residual add input shapes differ")
        am = int(np.rint((a_q[0] / scale) * (1 << 31)))
        bm = int(np.rint((b_q[0] / scale) * (1 << 31)))
        numerator = (a.to(torch.int64) - a_q[1]) * am
        numerator += (b.to(torch.int64) - b_q[1]) * bm
        result = round_shift_nearest_even(numerator) + zero_point
        return result.clamp(zero_point, 255).to(torch.uint8)

    def run_codes(self, input_codes: torch.Tensor) -> torch.Tensor:
        if tuple(input_codes.shape) != (1, 12, 5000) or input_codes.dtype != torch.uint8:
            raise ValueError("expected one uint8 [1,12,5000] input window")
        values: dict[str, torch.Tensor] = {}
        qparams: dict[str, tuple[float, int]] = {}
        for op in self.graph:
            kind, name = op["op"], op["node"]
            if kind == "input_affine_quantize":
                values[name] = input_codes
                qparams[name] = (self.input_scale, self.input_zero_point)
            elif kind == "quantized_layer":
                layer = self.layers[name]
                values[name] = self._layer(layer, values[layer["input_node"]])
                qparams[name] = (float(layer["output_scale"]),
                                 int(layer["output_zero_point"]))
            elif kind == "int8_add_then_relu":
                a_name, b_name = op["input_nodes"]
                values[name] = self._add_relu(
                    values[a_name], qparams[a_name],
                    values[b_name], qparams[b_name],
                    float(op["output_scale"]), int(op["output_zero_point"]),
                )
                qparams[name] = (float(op["output_scale"]),
                                 int(op["output_zero_point"]))
            elif kind in {"identity", "dropout_identity_in_eval"}:
                source = op["input_node"]
                values[name] = values[source]
                qparams[name] = qparams[source]
            elif kind == "adaptive_avg_pool1d":
                source = op["input_node"]
                x = values[source].to(torch.int64)
                values[name] = round_divide_nearest_even(
                    x.sum(dim=-1, keepdim=True), x.shape[-1]
                ).clamp(0, 255).to(torch.uint8)
                qparams[name] = qparams[source]
            elif kind == "squeeze":
                source = op["input_node"]
                values[name] = values[source].squeeze(int(op["dimension"]))
                qparams[name] = qparams[source]
            elif kind == "dequantize_output":
                source = op["input_node"]
                values[name] = values[source]
                qparams[name] = qparams[source]
            elif kind == "model_output":
                return values[op["input_node"]]
            else:
                raise ValueError("unsupported graph operation: {}".format(kind))
        raise ValueError("model_output is missing")

    def dequantize_output(self, codes: torch.Tensor) -> torch.Tensor:
        return (codes.float() - int(self.final_layer["output_zero_point"])) * float(
            self.final_layer["output_scale"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-index", type=int, default=0,
                        help="zero-based index into cached PTB-XL validation fold 9")
    args = parser.parse_args()
    cache = ROOT / "nn" / "data" / "cache"
    raw = np.load(cache / "fold_9_x.npy", mmap_mode="r")
    if not 0 <= args.record_index < len(raw):
        parser.error("record-index is outside fold 9")
    stats = json.loads((cache / "normalization.json").read_text(encoding="utf-8"))
    x = torch.from_numpy(np.array(raw[args.record_index:args.record_index + 1], copy=True))
    mean = torch.tensor(stats["mean_mV"], dtype=torch.float32).view(1, 12, 1)
    std = torch.tensor(stats["std_mV"], dtype=torch.float32).view(1, 12, 1)
    normalized = ((x - mean) / std).clamp(-8.0, 8.0)

    reference = IntegerECGReference()
    codes = reference.run_codes(reference.quantize_input(normalized))
    logits = reference.dequantize_output(codes)
    model = torch.jit.load(str(ROOT / "nn" / "artifacts" / "model_ptq_int8_x86.ts")).eval()
    with torch.inference_mode():
        torchscript_logits = model(normalized)
    print(json.dumps({
        "fold9_index": args.record_index,
        "reference_codes": codes.flatten().tolist(),
        "reference_logits": logits.flatten().tolist(),
        "torchscript_logits": torchscript_logits.flatten().tolist(),
        "max_abs_logit_error": float((logits - torchscript_logits).abs().max()),
    }, indent=2))


if __name__ == "__main__":
    main()
