"""Integer YOLO pose QOperator graph runner for PYNQ-Z2.

Intermediate image tensors remain uint8. Convolution is delegated to a PL
backend; PS elementwise, pooling, resize and attention use integer arrays.
Only the three public model outputs are dequantized for existing pose decode.
"""

from __future__ import annotations

from collections import Counter
import ctypes
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def _fixed_scale(ratio: float) -> tuple[int, int]:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"invalid quantized scale ratio {ratio}")
    significand, exponent = math.frexp(ratio)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1
    return multiplier, 31 - exponent


def _round_shift_even(value: np.ndarray, shift: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.int64)
    if shift <= 0:
        return value << -shift
    magnitude = np.abs(value)
    whole = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    rounded = whole + ((remainder > halfway) |
                       ((remainder == halfway) & ((whole & 1) != 0)))
    return np.where(value < 0, -rounded, rounded)


def _requant(value: np.ndarray, factor: tuple[int, int], zero: int) -> np.ndarray:
    multiplier, shift = factor
    result = _round_shift_even(np.asarray(value, dtype=np.int64) * multiplier, shift)
    return np.clip(result + zero, 0, 255).astype(np.uint8)


def _max_pool_separable(source, kernel, pads, stride):
    """Exact UINT8 rectangular max pool, using two one-dimensional passes."""
    if source.dtype != np.uint8 or source.ndim != 4:
        raise ValueError("separable pool requires NCHW uint8")
    kh, kw = map(int, kernel)
    sh, sw = map(int, stride)
    padded = np.pad(source, ((0, 0), (0, 0), (pads[0], pads[2]),
                            (pads[1], pads[3])), constant_values=0)
    oh = (padded.shape[-2] - kh) // sh + 1
    ow = (padded.shape[-1] - kw) // sw + 1
    horizontal = padded[..., :ow * sw:sw].copy()
    for x in range(1, kw):
        np.maximum(horizontal, padded[..., x:x + ow * sw:sw], out=horizontal)
    result = horizontal[..., :oh * sh:sh, :].copy()
    for y in range(1, kh):
        np.maximum(result, horizontal[..., y:y + oh * sh:sh, :], out=result)
    return result


class Int8Graph:
    def __init__(self, directory: str | Path, conv_backend, separable_pool=False):
        self.separable_pool = bool(separable_pool)
        directory = Path(directory)
        self.manifest = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        shape = self.manifest.get("input_shape", [1, 3, 192, 256])
        if len(shape) != 4 or shape[:2] != [1, 3] or min(shape[2:]) < 32:
            raise ValueError(f"unsupported fixed pose input shape: {shape}")
        self.input_height, self.input_width = map(int, shape[2:])
        with np.load(directory / "tensors.npz", allow_pickle=False) as archive:
            self.constants = {name: archive[name].copy() for name in archive.files}
        self.nodes = self.manifest["nodes"]
        self.conv_backend = conv_backend
        self.native = getattr(conv_backend, "native", None)
        if self.native is not None:
            u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
            self.native.apply_lut_u8.argtypes = [u8, u8, u8, ctypes.c_size_t]
            self.native.apply_lut_u8.restype = None
            self.native.apply_binary_lut_u8.argtypes = [u8, u8, u8, u8, ctypes.c_size_t]
            self.native.apply_binary_lut_u8.restype = None
        self.factors = {}
        self.sigmoid_luts = {}
        self.fused_silu_luts = {}
        self.fused_sigmoid_outputs = set()
        self.concat_luts = {}
        self.add_luts = {}
        self.softmax_luts = {}
        self.refcounts = Counter(name for node in self.nodes for name in node["inputs"] if name)
        sigmoid_by_output = {node["outputs"][0]: node for node in self.nodes
                             if node["op"] == "QLinearSigmoid"}
        conv_by_output = {node["outputs"][0]: node for node in self.nodes
                          if node["op"] == "QLinearConv"}
        consumers_by_input = {}
        for node in self.nodes:
            for input_name in node["inputs"]:
                if input_name:
                    consumers_by_input.setdefault(input_name, []).append(node)
        for node in self.nodes:
            op, names = node["op"], node["inputs"]
            if op in ("QLinearMul", "QLinearMatMul"):
                scale = self._scalar(names[1]) * self._scalar(names[4]) / self._scalar(names[6])
                self.factors[node["name"]] = _fixed_scale(scale)
            elif op == "QLinearConcat":
                self.factors[node["name"]] = [
                    _fixed_scale(self._scalar(names[index + 1]) / self._scalar(names[0]))
                    for index in range(2, len(names), 3)
                ]
                self.concat_luts[node["name"]] = [
                    None if (self._scalar(names[index + 1]) == self._scalar(names[0]) and
                             self._scalar(names[index + 2]) == self._scalar(names[1])) else
                    _requant(np.arange(256, dtype=np.int32) -
                             int(self._scalar(names[index + 2])),
                             self.factors[node["name"]][part],
                             int(self._scalar(names[1])))
                    for part, index in enumerate(range(2, len(names), 3))
                ]
            elif op == "QLinearSigmoid":
                x = (np.arange(256, dtype=np.float64) - self._scalar(names[2])) * self._scalar(names[1])
                sigmoid = 1.0 / (1.0 + np.exp(-x))
                self.sigmoid_luts[node["name"]] = np.clip(
                    np.rint(sigmoid / self._scalar(names[3])) + self._scalar(names[4]), 0, 255
                ).astype(np.uint8)
            elif op == "QLinearAdd":
                a = np.arange(256, dtype=np.int64)[:, None] - int(self._scalar(names[2]))
                b = np.arange(256, dtype=np.int64)[None, :] - int(self._scalar(names[5]))
                first = _fixed_scale(self._scalar(names[1]) / self._scalar(names[6]))
                second = _fixed_scale(self._scalar(names[4]) / self._scalar(names[6]))
                shift = max(first[1], second[1])
                combined = ((a * first[0]) << (shift - first[1])) + \
                           ((b * second[0]) << (shift - second[1]))
                self.add_luts[node["name"]] = np.clip(
                    _round_shift_even(combined, shift) + int(self._scalar(names[7])),
                    0, 255).astype(np.uint8)
            elif op == "QLinearSoftmax":
                self.softmax_luts[node["name"]] = np.maximum(
                    1, np.rint(np.exp(-np.arange(256) * self._scalar(names[1])) * (1 << 24))
                ).astype(np.int64)
            if op == "QLinearMul" and names[3] in sigmoid_by_output:
                sigmoid = sigmoid_by_output[names[3]]
                if sigmoid["inputs"][0] == names[0]:
                    self.fused_sigmoid_outputs.add(names[3])
                    source = np.arange(256, dtype=np.int32)
                    self.fused_silu_luts[node["name"]] = _requant(
                        (source - int(self._scalar(names[2]))) *
                        (self.sigmoid_luts[sigmoid["name"]].astype(np.int32) -
                         int(self._scalar(names[5]))),
                        self.factors[node["name"]], int(self._scalar(names[7])))

        # Move a complete Conv -> Sigmoid -> Mul SiLU into the PL epilogue.
        # This is safe only when the raw Conv tensor has exactly those two
        # consumers and is not itself a public graph output.
        self.conv_output_luts = {}
        self.pl_fused_silu_muls = set()
        if getattr(self.conv_backend, "supports_output_lut", False):
            graph_outputs = set(self.manifest["outputs"])
            for node in self.nodes:
                if node["op"] != "QLinearMul" or node["name"] not in self.fused_silu_luts:
                    continue
                source = node["inputs"][0]
                sigmoid = sigmoid_by_output.get(node["inputs"][3])
                conv = conv_by_output.get(source)
                consumers = consumers_by_input.get(source, [])
                if (sigmoid is not None and conv is not None and
                        sigmoid["inputs"][0] == source and source not in graph_outputs and
                        self.refcounts[source] == 2 and len(consumers) == 2 and
                        {item["name"] for item in consumers} ==
                        {sigmoid["name"], node["name"]}):
                    self.conv_output_luts[conv["name"]] = self.fused_silu_luts[node["name"]]
                    self.pl_fused_silu_muls.add(node["name"])

        self.prepared_convs = {}
        binder = getattr(self.conv_backend, "bind_conv", None)
        if binder is not None:
            for node in self.nodes:
                if node["op"] != "QLinearConv":
                    continue
                bound_node = dict(node)
                if node["name"] in self.conv_output_luts:
                    bound_node["fused_output_lut"] = self.conv_output_luts[node["name"]]
                names = node["inputs"]
                self.prepared_convs[node["name"]] = binder(
                    bound_node, *(self.constants[name] for name in names[1:9]))

    def _scalar(self, name: str) -> float:
        return float(self.constants[name].item())

    def _unary_lut(self, source: np.ndarray, lut: np.ndarray) -> np.ndarray:
        if self.native is None or not source.flags.c_contiguous or source.size < 256:
            return lut[source]
        result = np.empty(source.shape, dtype=np.uint8)
        self.native.apply_lut_u8(source, result, lut, source.size)
        return result

    def _binary_lut(self, a: np.ndarray, b: np.ndarray, lut: np.ndarray) -> np.ndarray:
        if (self.native is None or a.shape != b.shape or
                not a.flags.c_contiguous or not b.flags.c_contiguous or a.size < 256):
            return lut[a, b]
        result = np.empty(a.shape, dtype=np.uint8)
        self.native.apply_binary_lut_u8(a, b, result, lut, a.size)
        return result

    def make_input(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox + RGB + direct byte lookup; no float image tensor on PS."""
        height, width = frame.shape[:2]
        gain = min(self.input_width / width, self.input_height / height)
        new_width = min(self.input_width, max(1, int(round(width * gain))))
        new_height = min(self.input_height, max(1, int(round(height * gain))))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_width = self.input_width - new_width
        pad_height = self.input_height - new_height
        left, top = pad_width // 2, pad_height // 2
        boxed = cv2.copyMakeBorder(resized, top, pad_height - top,
                                   left, pad_width - left, cv2.BORDER_CONSTANT,
                                   value=(114, 114, 114))
        qnode = next(node for node in self.nodes if node["op"] == "QuantizeLinear" and
                     node["inputs"][0] == self.manifest["input"])
        names = qnode["inputs"]
        lut = np.clip(np.rint((np.arange(256) / 255.0) / self._scalar(names[1])) +
                      self._scalar(names[2]), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(lut[boxed[:, :, ::-1]].transpose(2, 0, 1)[None])

    def run(self, qinput: np.ndarray, trace=None, profile=None) -> list[np.ndarray]:
        first = next(node for node in self.nodes if node["op"] == "QuantizeLinear" and
                     node["inputs"][0] == self.manifest["input"])
        values = {first["outputs"][0]: np.asarray(qinput, dtype=np.uint8)}
        remaining = self.refcounts.copy()
        output_set = set(self.manifest["outputs"])
        for node in self.nodes:
            if node is first:
                continue
            names = node["inputs"]
            get = lambda index: values.get(names[index], self.constants.get(names[index]))
            op = node["op"]
            attrs = node["attrs"]
            if op == "QLinearSigmoid" and node["outputs"][0] in self.fused_sigmoid_outputs:
                for name in names:
                    if name in remaining:
                        remaining[name] -= 1
                        if remaining[name] == 0 and name not in output_set:
                            values.pop(name, None)
                continue
            op_started = time.perf_counter() if profile is not None else None
            trace_node = node
            if op == "QLinearConv":
                if node["name"] in self.conv_output_luts:
                    trace_node = dict(node)
                    trace_node["fused_output_lut"] = self.conv_output_luts[node["name"]]
                bound = self.prepared_convs.get(node["name"])
                if bound is not None:
                    result = bound(get(0))
                else:
                    result = self.conv_backend.run(trace_node, get(0),
                        self.constants[names[1]], self.constants[names[2]],
                        self.constants[names[3]], self.constants[names[4]],
                        self.constants[names[5]], self.constants[names[6]],
                        self.constants[names[7]], self.constants[names[8]])
            elif op == "QLinearMul" and node["name"] in self.fused_silu_luts:
                result = (get(0) if node["name"] in self.pl_fused_silu_muls else
                          self._unary_lut(get(0), self.fused_silu_luts[node["name"]]))
            elif op == "QLinearAdd":
                result = self._binary_lut(get(0), get(3), self.add_luts[node["name"]])
            elif op == "QLinearMul":
                a = get(0).astype(np.int32) - int(self._scalar(names[2]))
                b = get(3).astype(np.int32) - int(self._scalar(names[5]))
                result = _requant(a * b, self.factors[node["name"]],
                                  int(self._scalar(names[7])))
            elif op == "QLinearSigmoid":
                result = self._unary_lut(get(0), self.sigmoid_luts[node["name"]])
            elif op == "QLinearConcat":
                parts = []
                for item, index in enumerate(range(2, len(names), 3)):
                    lut = self.concat_luts[node["name"]][item]
                    parts.append(get(index) if lut is None else self._unary_lut(get(index), lut))
                result = np.concatenate(parts, axis=int(attrs["axis"]))
            elif op == "QLinearMatMul":
                a = get(0).astype(np.int32) - int(self._scalar(names[2]))
                b = get(3).astype(np.int32) - int(self._scalar(names[5]))
                result = _requant(np.matmul(a, b), self.factors[node["name"]],
                                  int(self._scalar(names[7])))
            elif op == "QLinearSoftmax":
                axis = int(attrs.get("axis", -1))
                source = get(0)
                difference = np.max(source, axis=axis, keepdims=True).astype(np.int16) - source.astype(np.int16)
                terms = self.softmax_luts[node["name"]][difference]
                total = terms.sum(axis=axis, keepdims=True)
                numerator = terms * int(round(1.0 / self._scalar(names[3])))
                quotient, remainder = np.divmod(numerator, total)
                quotient += ((2 * remainder > total) |
                             ((2 * remainder == total) & ((quotient & 1) != 0)))
                result = np.clip(quotient + int(self._scalar(names[4])), 0, 255).astype(np.uint8)
            elif op == "Split":
                axis = int(attrs.get("axis", 0))
                sizes = attrs.get("split")
                if sizes is None:
                    result = list(np.array_split(get(0), len(node["outputs"]), axis=axis))
                else:
                    result = list(np.split(get(0), np.cumsum(sizes)[:-1], axis=axis))
            elif op == "Reshape":
                shape = np.asarray(get(1), dtype=np.int64).tolist()
                shape = [get(0).shape[index] if dim == 0 else dim
                         for index, dim in enumerate(shape)]
                result = get(0).reshape(shape)
            elif op == "Transpose":
                result = get(0).transpose(attrs.get("perm"))
            elif op == "MaxPool":
                source = get(0)
                kernel = attrs["kernel_shape"]
                pads = attrs.get("pads", [0, 0, 0, 0])
                stride = attrs.get("strides", [1, 1])
                if self.separable_pool:
                    result = _max_pool_separable(source, kernel, pads, stride)
                else:
                    padded = np.pad(source, ((0, 0), (0, 0), (pads[0], pads[2]),
                                             (pads[1], pads[3])), constant_values=0)
                    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=(-2, -1))
                    result = windows[:, :, ::stride[0], ::stride[1]].max(axis=(-1, -2))
            elif op == "Resize":
                scales = np.asarray(get(2)).tolist()
                if scales[:2] != [1.0, 1.0] or any(int(v) != v for v in scales[2:]):
                    raise NotImplementedError(f"unsupported Resize scales {scales}")
                result = np.repeat(np.repeat(get(0), int(scales[2]), axis=2),
                                   int(scales[3]), axis=3)
            elif op == "DequantizeLinear":
                result = (get(0).astype(np.float32) - self._scalar(names[2])) * self._scalar(names[1])
            elif op == "QuantizeLinear":
                result = np.clip(np.rint(get(0) / self._scalar(names[1])) +
                                 self._scalar(names[2]), 0, 255).astype(np.uint8)
            else:
                raise NotImplementedError(f"{node['name']}: {op}")
            if profile is not None:
                elapsed = time.perf_counter() - op_started
                profile[op] = profile.get(op, 0.0) + elapsed
                by_node = profile.setdefault("by_node", {})
                by_node[node["name"]] = elapsed
            results = result if isinstance(result, list) else [result]
            if len(results) != len(node["outputs"]):
                raise RuntimeError(f"{node['name']}: output count mismatch")
            for name, tensor in zip(node["outputs"], results):
                values[name] = tensor
                if trace is not None:
                    trace(trace_node, name, tensor)
            for name in names:
                if name in remaining:
                    remaining[name] -= 1
                    if remaining[name] == 0 and name not in output_set:
                        values.pop(name, None)
        return [values[name] for name in self.manifest["outputs"]]
