"""Portable NumPy PS baseline for the exact FPGA INT8 inference contract.

Only model inference is timed by the board test; preprocessing and transfers
are reported separately. This is a correctness baseline, not a tuned ARM NN
library, so its timing must be labelled as such in comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np


def _round_q31(values):
    values = np.asarray(values, dtype=np.int64)
    magnitude = np.abs(values)
    whole = magnitude >> 31
    remainder = magnitude & ((1 << 31) - 1)
    increment = (remainder > (1 << 30)) | (
        (remainder == (1 << 30)) & ((whole & 1) != 0)
    )
    rounded = whole + increment.astype(np.int64)
    return np.where(values < 0, -rounded, rounded)


def _round_divide(values, divisor):
    whole, remainder = np.divmod(values.astype(np.int64), int(divisor))
    return whole + ((2 * remainder > divisor) |
                    ((2 * remainder == divisor) & ((whole & 1) != 0)))


class NumpyIntegerECG:
    def __init__(self, contract_path):
        self.contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        self.weights = np.load(Path(contract_path).parent / self.contract["weights_file"])
        self.layers = {layer["node"]: layer for layer in self.contract["quantized_layers"]}
        self.graph = self.contract["graph_ops"]

    def _layer(self, layer, source):
        key = layer["module"].replace(".", "_")
        weight = self.weights[key + "_weight_int8"].astype(np.int32)
        bias = self.weights[key + "_bias_int32"].astype(np.int64)
        input_zp = int(layer["input_zero_point"])
        if weight.ndim == 3:
            setting = layer["activation"]
            pad = int(setting["padding"][0])
            stride = int(setting["stride"][0])
            padded = np.pad(source, ((0, 0), (pad, pad)),
                            constant_values=input_zp).astype(np.int32) - input_zp
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, weight.shape[2], axis=1)[:, ::stride, :]
            windows = windows.transpose(0, 2, 1)
            acc = np.einsum("ock,ckt->ot", weight, windows,
                            optimize=True, dtype=np.int64) + bias[:, None]
        else:
            acc = weight.astype(np.int64) @ (source.astype(np.int64) - input_zp)
            acc += bias
        ratios = np.asarray(layer["requantization_multiplier"], dtype=np.float64)
        q31 = np.rint(ratios * (1 << 31)).astype(np.int64)
        if len(q31) == 1:
            q31 = np.repeat(q31, weight.shape[0])
        product = acc * (q31[:, None] if acc.ndim == 2 else q31)
        output_zp = int(layer["output_zero_point"])
        code = _round_q31(product) + output_zp
        if "ConvReLU" in layer["operator"]:
            code = np.maximum(code, output_zp)
        return np.clip(code, 0, 255).astype(np.uint8)

    def run_codes(self, time_major_codes):
        codes = np.asarray(time_major_codes, dtype=np.uint8)
        if codes.shape != (5000, 12):
            raise ValueError("expected [5000,12] affine uint8 codes")
        values = {}
        qparams = {}
        for op in self.graph:
            kind, name = op["op"], op["node"]
            if kind == "input_affine_quantize":
                values[name] = codes.T.copy()
                qparams[name] = (float(op["input_scale"]), int(op["input_zero_point"]))
            elif kind == "quantized_layer":
                layer = self.layers[name]
                values[name] = self._layer(layer, values[layer["input_node"]])
                qparams[name] = (float(layer["output_scale"]),
                                 int(layer["output_zero_point"]))
            elif kind == "int8_add_then_relu":
                a_name, b_name = op["input_nodes"]
                a, b = values[a_name], values[b_name]
                a_scale, a_zp = qparams[a_name]
                b_scale, b_zp = qparams[b_name]
                out_scale = float(op["output_scale"])
                out_zp = int(op["output_zero_point"])
                am = int(np.rint(a_scale / out_scale * (1 << 31)))
                bm = int(np.rint(b_scale / out_scale * (1 << 31)))
                total = (a.astype(np.int64) - a_zp) * am
                total += (b.astype(np.int64) - b_zp) * bm
                values[name] = np.clip(np.maximum(_round_q31(total) + out_zp,
                                                  out_zp), 0, 255).astype(np.uint8)
                qparams[name] = (out_scale, out_zp)
            elif kind in ("identity", "dropout_identity_in_eval",
                          "dequantize_output"):
                source = op["input_node"]
                values[name], qparams[name] = values[source], qparams[source]
            elif kind == "adaptive_avg_pool1d":
                source = op["input_node"]
                x = values[source]
                values[name] = np.clip(_round_divide(
                    x.astype(np.int64).sum(axis=1, keepdims=True), x.shape[1]),
                    0, 255).astype(np.uint8)
                qparams[name] = qparams[source]
            elif kind == "squeeze":
                source = op["input_node"]
                axis = int(op["dimension"])
                values[name] = np.squeeze(values[source],
                                          axis=axis if axis < 0 else axis - 1)
                qparams[name] = qparams[source]
            elif kind == "model_output":
                return values[op["input_node"]].astype(np.uint8)
            else:
                raise ValueError("unsupported graph operation {}".format(kind))
        raise ValueError("model_output missing")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = here.parents[0]
    contract = here / "model_ptq_int8_hw.json"
    if not contract.exists():
        contract = root / "artifacts" / "model_ptq_int8_hw.json"
    engine = NumpyIntegerECG(contract)
    vectors = here / "golden_vectors"
    for index in (0, 333, 1024, 2000):
        prefix = "fold9_{:04d}".format(index)
        stimulus = np.fromfile(vectors / (prefix + "_input_time_major_u8.bin"),
                               dtype=np.uint8).reshape(5000, 12)
        expected = np.fromfile(vectors / (prefix + "_expected_output_u8.bin"),
                               dtype=np.uint8)
        start = perf_counter()
        got = engine.run_codes(stimulus)
        seconds = perf_counter() - start
        print(prefix, got.tolist(), "seconds", round(seconds, 3))
        if not np.array_equal(got, expected):
            raise SystemExit("integer reference mismatch for " + prefix)
    print("PASS: NumPy PS reference exactly matches four golden vectors")
