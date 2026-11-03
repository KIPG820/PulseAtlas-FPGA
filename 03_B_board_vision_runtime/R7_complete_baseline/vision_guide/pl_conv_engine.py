"""YOLO11n-pose benchmark with selected Conv nodes dispatched to the PYNQ PL.

Run on the PYNQ-Z2 as root. The test overlay is temporary: the original overlay
is reloaded in a finally block. Results include OpenCV preprocessing/decoding,
FPGA layer calls, and the same camera frames on both PS and FPGA paths.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pynq import Overlay, allocate


PTRS = {
    "patches": 0x18,
    "weights": 0x24,
    "bias": 0x30,
    "multipliers": 0x3C,
    "shifts": 0x48,
    "output": 0x54,
    "descriptor": 0x60,
}
MAX_WORDS = {"patches": 1_048_576, "weights": 1_048_576, "output": 131_072}
LANES = 8
BYTE_SHIFTS = np.arange(LANES, dtype=np.uint64) * np.uint64(8)
NATIVE = None


def load_native(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.make_multipliers.argtypes = [ctypes.c_double, f32, f32, i32, i32, ctypes.c_int]
    lib.make_multipliers.restype = ctypes.c_int
    lib.quantize_u8.argtypes = [f32, u8, ctypes.c_size_t, ctypes.c_float]
    lib.max_abs.argtypes = [f32, ctypes.c_size_t]
    lib.max_abs.restype = ctypes.c_float
    lib.pack_im2col.argtypes = [u8, u8] + [ctypes.c_int] * 14
    lib.dequantize_u8.argtypes = [u8, f32, f32] + [ctypes.c_int] * 3
    for name in ("quantize_u8", "pack_im2col", "dequantize_u8"):
        getattr(lib, name).restype = None
    return lib


def set_ptr(ip, offset: int, address: int) -> None:
    ip.write(offset, int(address) & 0xFFFFFFFF)
    ip.write(offset + 4, (int(address) >> 32) & 0xFFFFFFFF)


def pack_bytes(values: np.ndarray) -> np.ndarray:
    """Pack the last axis of eight bytes into little-endian uint64 words."""
    if values.dtype != np.uint8 or values.shape[-1] != LANES:
        raise ValueError("packing requires uint8 with eight byte lanes")
    return np.ascontiguousarray(values).view("<u8").reshape(values.shape[:-1])


def quantize_multiplier(real_scale: float) -> tuple[int, int]:
    if not np.isfinite(real_scale) or real_scale < 0:
        raise ValueError("invalid requantization scale {}".format(real_scale))
    if real_scale == 0:
        return 0, 0
    significand, shift = math.frexp(real_scale)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1
    if shift < -31 or shift > 30:
        raise ValueError("requantization shift {} is outside HLS range".format(shift))
    return multiplier, shift


def output_shape(input_shape, params, weight_shape):
    batch, _, height, width = input_shape
    kernel_h, kernel_w = params["kernel_size"]
    stride_h, stride_w = params.get("stride", (1, 1))
    dilation_h, dilation_w = params.get("dilation", (1, 1))
    pad = params.get("pad", (0, 0, 0, 0))
    pad_top, pad_left, pad_bottom, pad_right = pad
    out_h = (height + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    return [batch, int(weight_shape[0]), out_h, out_w]


class SharedConvEngine:
    """PYNQ driver for the descriptor-driven shared convolution HLS IP."""

    def __init__(self, overlay: Overlay):
        self.ip = overlay.shared_conv_0
        self.buffers = {}
        self.layers = []
        self.last_stats = {}
        self.frame_stats = {}
        self.direct_patch_buffer = False
        self.bias = allocate(shape=(512,), dtype=np.int32)
        self.multipliers = allocate(shape=(512,), dtype=np.int32)
        self.shifts = allocate(shape=(512,), dtype=np.int32)
        self.descriptor = allocate(shape=(8,), dtype=np.int32)
        for name, buf in (("bias", self.bias), ("multipliers", self.multipliers),
                          ("shifts", self.shifts), ("descriptor", self.descriptor)):
            set_ptr(self.ip, PTRS[name], buf.physical_address)

    def register_layer(self, layer) -> None:
        packed = layer.packed_weights.reshape(-1)
        if packed.size > MAX_WORDS["weights"]:
            raise ValueError("weight buffer exceeds shared convolution IP AXI range")
        layer.weight_buffer = allocate(shape=packed.shape, dtype=np.uint64)
        np.copyto(layer.weight_buffer, packed)
        layer.weight_buffer.flush()
        self.layers.append(layer)

    def _ensure_buffer(self, name: str, words: int):
        if words < 1 or words > MAX_WORDS[name]:
            raise ValueError("{} words {} exceeds HLS interface capacity".format(name, words))
        old = self.buffers.get(name)
        if old is None or old.size < words:
            if old is not None:
                old.freebuffer()
            self.buffers[name] = allocate(shape=(words,), dtype=np.uint64)
        return self.buffers[name]

    def run(self, layer, x: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        batch, channels, height, width = x.shape
        if batch != 1:
            raise ValueError("shared vision layer currently requires batch=1")
        kh, kw = layer.kernel_size
        sh, sw = layer.stride
        dh, dw = layer.dilation
        pt, pl, pb, pr = layer.padding
        groups = layer.groups
        m = layer.output_channels // groups
        out_h = (height + pt + pb - dh * (kh - 1) - 1) // sh + 1
        out_w = (width + pl + pr - dw * (kw - 1) - 1) // sw + 1
        n = out_h * out_w
        k = (channels // groups) * kh * kw
        nblocks = (n + 7) // 8
        mblocks = (m + 7) // 8

        if NATIVE is None:
            input_scale = max(float(np.max(np.abs(x))) / 127.0, 1e-8)
            qx = np.clip(np.rint(x / input_scale) + 128, 0, 255).astype(np.uint8)
        else:
            x = np.ascontiguousarray(x)
            input_scale = max(float(NATIVE.max_abs(x, x.size)) / 127.0, 1e-8)
            qx = np.empty(x.shape, dtype=np.uint8)
            NATIVE.quantize_u8(x, qx, x.size, input_scale)
        quantized_at = time.perf_counter()
        patch_buffer = None
        if NATIVE is None:
            grouped = qx.reshape(1, groups, channels // groups, height, width)
            padded = np.pad(grouped, ((0, 0), (0, 0), (0, 0), (pt, pb), (pl, pr)),
                            constant_values=128)
            eff_h, eff_w = dh * (kh - 1) + 1, dw * (kw - 1) + 1
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, (eff_h, eff_w), axis=(3, 4)
            )
            windows = windows[:, :, :, 0:out_h * sh:sh, 0:out_w * sw:sw, ::dh, ::dw]
            matrix = windows[0].transpose(0, 2, 3, 1, 4, 5).reshape(groups, n, k)
            tile_bytes = np.full((groups, nblocks * 8, k), 128, dtype=np.uint8)
            tile_bytes[:, :n] = matrix
            tile_bytes = tile_bytes.reshape(groups, nblocks, 8, k).transpose(0, 1, 3, 2)
            patches = pack_bytes(tile_bytes).reshape(-1)
        else:
            patch_words = groups * nblocks * k
            if self.direct_patch_buffer:
                patch_buffer = self._ensure_buffer("patches", patch_words)
                patch_bytes = patch_buffer[:patch_words].view(np.uint8)
                if not np.shares_memory(patch_buffer, patch_bytes):
                    raise RuntimeError("DMA patch byte view unexpectedly copied its storage")
                NATIVE.pack_im2col(qx, patch_bytes, channels, height, width,
                                   groups, kh, kw, sh, sw, dh, dw, pt, pl, out_h, out_w)
                patches = None
            else:
                patches = np.empty(patch_words, dtype=np.uint64)
                NATIVE.pack_im2col(qx, patches.view(np.uint8), channels, height, width,
                                   groups, kh, kw, sh, sw, dh, dw, pt, pl, out_h, out_w)
        patches_at = time.perf_counter()

        bias_q = np.rint(layer.bias / (input_scale * layer.weight_scales)).astype(np.int64)
        bias_q = np.clip(bias_q, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        if NATIVE is None:
            real_scales = (input_scale * layer.weight_scales.astype(np.float64)
                           / layer.output_scales.astype(np.float64))
            significands, shifts = np.frexp(real_scales)
            multipliers = np.rint(significands * (1 << 31)).astype(np.int64)
            carry = multipliers == (1 << 31)
            multipliers[carry] //= 2
            shifts[carry] += 1
            if np.any(shifts < -31) or np.any(shifts > 30):
                raise ValueError("requantization shift outside HLS range")
            multipliers = multipliers.astype(np.int32)
            shifts = shifts.astype(np.int32)
        else:
            multipliers = np.empty(layer.output_channels, dtype=np.int32)
            shifts = np.empty(layer.output_channels, dtype=np.int32)
            if NATIVE.make_multipliers(input_scale, layer.weight_scales, layer.output_scales,
                                       multipliers, shifts, layer.output_channels):
                raise ValueError("requantization shift outside HLS range")

        packed_time = time.perf_counter() - started
        if patch_buffer is None:
            patch_buffer = self._ensure_buffer("patches", patches.size)
        output_words = layer.output_channels * nblocks
        output_buffer = self._ensure_buffer("output", output_words)
        if patches is not None:
            np.copyto(patch_buffer[:patches.size], patches)
        np.copyto(self.bias[:layer.output_channels], bias_q)
        np.copyto(self.multipliers[:layer.output_channels], multipliers)
        np.copyto(self.shifts[:layer.output_channels], shifts)
        self.descriptor[:] = [m, n, k, groups, 128, 128, 0, 255]
        patch_buffer.flush()
        self.bias.flush()
        self.multipliers.flush()
        self.shifts.flush()
        self.descriptor.flush()
        set_ptr(self.ip, PTRS["patches"], patch_buffer.physical_address)
        set_ptr(self.ip, PTRS["weights"], layer.weight_buffer.physical_address)
        set_ptr(self.ip, PTRS["output"], output_buffer.physical_address)

        core_start = time.perf_counter()
        self.ip.write(0x00, 1)
        deadline = core_start + 5.0
        while not (self.ip.read(0x00) & 0x2):
            if time.perf_counter() >= deadline:
                raise TimeoutError("shared convolution IP timed out at layer {}".format(layer.index))
        status = self.ip.read(0x10)
        core_time = time.perf_counter() - core_start
        if status:
            raise RuntimeError("shared convolution IP returned {} at layer {}".format(status, layer.index))
        output_buffer.invalidate()
        words = np.array(output_buffer[:output_words], dtype=np.uint64,
                         copy=True).reshape(layer.output_channels, nblocks)
        qout = words.astype("<u8", copy=False).view(np.uint8).reshape(
            layer.output_channels, nblocks * 8)[:, :n]
        clip_fraction = float(np.mean((qout == 0) | (qout == 255)))
        if NATIVE is None:
            result = ((qout.astype(np.float32) - 128.0) * layer.output_scales[:, None])
        else:
            result = np.empty((layer.output_channels, n), dtype=np.float32)
            NATIVE.dequantize_u8(words.view(np.uint8), result, layer.output_scales,
                                 layer.output_channels, n, nblocks * 8)
        result = result.reshape(1, layer.output_channels, out_h, out_w)
        self.last_stats = {"pack_ms": packed_time * 1000.0,
                           "quantize_ms": (quantized_at - started) * 1000.0,
                           "patches_ms": (patches_at - quantized_at) * 1000.0,
                           "metadata_ms": (started + packed_time - patches_at) * 1000.0,
                           "transfer_ms": (core_start - started - packed_time) * 1000.0,
                           "kernel_ms": core_time * 1000.0,
                           "unpack_ms": (time.perf_counter() - core_start - core_time) * 1000.0,
                           "clip_fraction": clip_fraction,
                           "layer": layer.index,
                           "shape": [int(x) for x in result.shape]}
        return result.astype(np.float32, copy=False)

    def clear_frame_stats(self):
        self.frame_stats = {"calls": 0, "pack_ms": 0.0, "kernel_ms": 0.0,
                            "unpack_ms": 0.0, "transfer_ms": 0.0, "clip_fraction_sum": 0.0, "layers": []}

    def add_frame_stats(self, elapsed):
        if "calls" not in self.frame_stats:
            self.clear_frame_stats()
        self.frame_stats["calls"] += 1
        for key in ("pack_ms", "kernel_ms", "unpack_ms", "transfer_ms"):
            self.frame_stats[key] += elapsed.get(key, 0.0)
        self.frame_stats["clip_fraction_sum"] += elapsed.get("clip_fraction", 0.0)
        self.frame_stats["layers"].append(dict(elapsed))

    def release(self):
        for layer in self.layers:
            if layer.weight_buffer is not None:
                layer.weight_buffer.freebuffer()
                layer.weight_buffer = None
        for buf in list(self.buffers.values()) + [self.bias, self.multipliers, self.shifts, self.descriptor]:
            if buf is not None:
                buf.freebuffer()


ENGINE = None
CALIBRATION_SCALES = None
FPGA_LAYERS = []


class FpgaConv:
    """OpenCV Python custom layer; replaces one ONNX Conv node with the PL IP."""

    def __init__(self, params, blobs):
        global ENGINE, CALIBRATION_SCALES
        self.index = int(params.get("pl_index", -1))
        if self.index < 0 or self.index >= len(CALIBRATION_SCALES):
            raise RuntimeError("missing or invalid pl_index on custom Conv node")
        self.params = dict(params)
        self.weights = np.asarray(blobs[0], dtype=np.float32).copy()
        self.bias = (np.asarray(blobs[1], dtype=np.float32).reshape(-1).copy()
                     if len(blobs) > 1 else np.zeros(self.weights.shape[0], np.float32))
        self.output_channels = int(self.weights.shape[0])
        self.groups = int(self.params.get("group", 1))
        self.kernel_size = tuple(int(x) for x in self.params["kernel_size"])
        self.stride = tuple(int(x) for x in self.params.get("stride", (1, 1)))
        self.dilation = tuple(int(x) for x in self.params.get("dilation", (1, 1)))
        self.padding = tuple(int(x) for x in self.params.get("pad", (0, 0, 0, 0)))
        self.output_scales = np.maximum(
            np.asarray(CALIBRATION_SCALES[self.index], dtype=np.float32), 1e-8
        ).reshape(-1)
        if self.output_scales.size != self.output_channels:
            raise RuntimeError("calibration channel count mismatch at Conv {}".format(self.index))
        self.weight_scales = np.maximum(
            np.max(np.abs(self.weights), axis=(1, 2, 3)) / 127.0, 1e-8
        ).astype(np.float32)
        qweights = np.clip(
            np.rint(self.weights / self.weight_scales[:, None, None, None]), -127, 127
        ).astype(np.int8)
        m = self.output_channels // self.groups
        mblocks = (m + 7) // 8
        k = qweights.shape[1] * qweights.shape[2] * qweights.shape[3]
        packed = np.zeros((self.groups, mblocks, k, 8), dtype=np.uint8)
        for group in range(self.groups):
            for channel in range(m):
                block, lane = divmod(channel, 8)
                packed[group, block, :, lane] = qweights[group * m + channel].reshape(-1).view(np.uint8)
        self.packed_weights = pack_bytes(packed)
        self.weight_buffer = None
        ENGINE.register_layer(self)
        FPGA_LAYERS.append(self)

    def getMemoryShapes(self, inputs):
        return [output_shape(inputs[0], self.params, self.weights.shape)]

    def forward(self, inputs):
        result = ENGINE.run(self, np.asarray(inputs[0], dtype=np.float32))
        self.last_output = result
        ENGINE.add_frame_stats(ENGINE.last_stats)
        return [result]


