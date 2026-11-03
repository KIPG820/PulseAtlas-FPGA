"""PYNQ v12 shared_conv backend for persistent UINT8 YOLO feature maps."""

from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from pynq import allocate

from pl_conv_engine import SharedConvEngine, MAX_WORDS, PTRS, pack_bytes, set_ptr, quantize_multiplier


class Int8PlConvBackend(SharedConvEngine):
    def __init__(self, overlay, native_library):
        super().__init__(overlay)
        self.native = ctypes.CDLL(str(Path(native_library).resolve()))
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        self.native.pack_im2col_zp.argtypes = [u8, u8] + [ctypes.c_int] * 15
        self.native.pack_im2col_zp.restype = None
        self.prepared = {}
        self.totals = {"pack_s": 0.0, "metadata_s": 0.0,
                       "cache_clean_s": 0.0, "mmio_s": 0.0,
                       "kernel_s": 0.0, "unpack_s": 0.0,
                       "call_s": 0.0, "calls": 0}

    def _prepare(self, node, weights, scales, zero_points, bias, input_scale, output_scale):
        key = node["name"]
        if key in self.prepared:
            return self.prepared[key]
        if weights.dtype != np.int8 or bias.dtype != np.int32 or np.any(zero_points != 0):
            raise ValueError(f"{key}: expected INT8 symmetric weights and INT32 bias")
        attrs = node["attrs"]
        groups = int(attrs.get("group", 1))
        channels = weights.shape[0]
        m = channels // groups
        k = int(np.prod(weights.shape[1:]))
        packed = np.zeros((groups, (m + 7) // 8, k, 8), dtype=np.uint8)
        for group in range(groups):
            for channel in range(m):
                block, lane = divmod(channel, 8)
                packed[group, block, :, lane] = weights[group * m + channel].reshape(-1).view(np.uint8)
        weight_scales = np.broadcast_to(scales, (channels,))
        multipliers, shifts = zip(*[
            quantize_multiplier(float(input_scale) * float(weight_scales[index]) /
                                float(output_scale))
            for index in range(channels)
        ])
        layer = SimpleNamespace(packed_weights=pack_bytes(packed), weight_buffer=None)
        self.register_layer(layer)
        layer.multipliers = np.asarray(multipliers, dtype=np.int32)
        layer.shifts = np.asarray(shifts, dtype=np.int32)
        layer.bias = bias
        layer.metadata = None
        layer.layout = None
        self.prepared[key] = layer
        return layer

    def run(self, node, qx, input_scale, input_zero, weights, weight_scales,
            weight_zero, output_scale, output_zero, bias):
        started = time.perf_counter()
        if qx.dtype != np.uint8 or qx.ndim != 4 or qx.shape[0] != 1:
            raise ValueError(f"{node['name']}: expected NCHW UINT8 batch 1")
        qx = np.ascontiguousarray(qx)
        input_zero, output_zero = int(input_zero), int(output_zero)
        layer = self._prepare(node, weights, weight_scales, weight_zero, bias,
                              input_scale, output_scale)
        attrs = node["attrs"]
        groups = int(attrs.get("group", 1))
        kh, kw = weights.shape[-2:]
        sh, sw = attrs.get("strides", [1, 1])
        dh, dw = attrs.get("dilations", [1, 1])
        pt, pl, pb, pr = attrs.get("pads", [0, 0, 0, 0])
        _, channels, height, width = qx.shape
        output_channels = weights.shape[0]
        output_height = (height + pt + pb - dh * (kh - 1) - 1) // sh + 1
        output_width = (width + pl + pr - dw * (kw - 1) - 1) // sw + 1
        n = output_height * output_width
        k = (channels // groups) * kh * kw
        nblocks = (n + 7) // 8
        patch_words = groups * nblocks * k
        output_words = output_channels * nblocks
        if patch_words > MAX_WORDS["patches"] or output_words > MAX_WORDS["output"]:
            raise ValueError(f"{node['name']}: exceeds HLS buffer range")
        patches = self._ensure_buffer("patches", patch_words)
        patch_bytes = patches[:patch_words].view(np.uint8)
        if not np.shares_memory(patches, patch_bytes):
            raise RuntimeError("PYNQ patch buffer byte view made a copy")
        self.native.pack_im2col_zp(
            qx, patch_bytes, channels, height, width, groups, kh, kw,
            sh, sw, dh, dw, pt, pl, output_height, output_width, input_zero)
        packed_at = time.perf_counter()
        output = self._ensure_buffer("output", output_words)
        layout = (output_channels // groups, n, k, groups,
                  input_zero, output_zero, 0, 255)
        if layer.metadata is None:
            # Each node has fixed shape and quantization in this exported graph.
            # Keep its PL descriptor and parameters resident instead of copying
            # and cleaning four CPU cache ranges on every inference.
            layer.metadata = allocate(shape=(3 * output_channels + 8,), dtype=np.int32)
            layer.metadata[0:output_channels] = layer.bias
            layer.metadata[output_channels:2 * output_channels] = layer.multipliers
            layer.metadata[2 * output_channels:3 * output_channels] = layer.shifts
            layer.metadata[3 * output_channels:] = layout
            layer.metadata.flush()
            layer.layout = layout
        elif layer.layout != layout:
            raise ValueError(f"{node['name']}: convolution shape or quantization changed")
        metadata_at = time.perf_counter()
        # _ensure_buffer grows to the largest layer and reuses that allocation.
        # PynqBuffer slices retain the DMA allocation and carry their own byte
        # length, so only the bytes consumed by this layer need cache cleaning.
        patches[:patch_words].flush()
        cleaned_at = time.perf_counter()
        set_ptr(self.ip, PTRS["patches"], patches.physical_address)
        set_ptr(self.ip, PTRS["weights"], layer.weight_buffer.physical_address)
        set_ptr(self.ip, PTRS["output"], output.physical_address)
        metadata_address = layer.metadata.physical_address
        set_ptr(self.ip, PTRS["bias"], metadata_address)
        set_ptr(self.ip, PTRS["multipliers"], metadata_address + 4 * output_channels)
        set_ptr(self.ip, PTRS["shifts"], metadata_address + 8 * output_channels)
        set_ptr(self.ip, PTRS["descriptor"], metadata_address + 12 * output_channels)
        kernel_at = time.perf_counter()
        self.ip.write(0x00, 1)
        deadline = kernel_at + 5.0
        while not (self.ip.read(0x00) & 0x2):
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"{node['name']}: shared_conv timed out")
        status = self.ip.read(0x10)
        if status:
            raise RuntimeError(f"{node['name']}: shared_conv status {status}")
        done_at = time.perf_counter()
        output[:output_words].invalidate()
        # frombuffer exposes the DMA bytes without preserving PynqBuffer's
        # ndarray subclass. Always copy the valid bytes: for spatial lengths
        # divisible by eight, ascontiguousarray could return a view into the
        # shared DMA buffer, which the next convolution overwrites.
        qout = np.frombuffer(output[:output_words], dtype=np.uint8).reshape(
            output_channels, nblocks * 8)[:, :n]
        result = np.array(qout.reshape(1, output_channels,
                                       output_height, output_width),
                          dtype=np.uint8, copy=True, order="C")
        finished_at = time.perf_counter()
        self.totals["pack_s"] += packed_at - started
        self.totals["metadata_s"] += metadata_at - packed_at
        self.totals["cache_clean_s"] += cleaned_at - metadata_at
        self.totals["mmio_s"] += kernel_at - cleaned_at
        self.totals["kernel_s"] += done_at - kernel_at
        self.totals["unpack_s"] += finished_at - done_at
        self.totals["call_s"] += finished_at - started
        self.totals["calls"] += 1
        return result

    def release(self):
        for layer in self.prepared.values():
            if layer.metadata is not None:
                layer.metadata.freebuffer()
                layer.metadata = None
        super().release()
