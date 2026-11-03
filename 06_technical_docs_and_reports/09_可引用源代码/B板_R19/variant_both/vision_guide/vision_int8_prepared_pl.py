"""Experimental PYNQ backend for the raw-input, tiled-im2col INT8 HLS core.

This is an isolated B-board candidate. It implements the same conv_backend.run
contract as Int8Graph and does not alter the known-good R3 service.
"""

from __future__ import annotations

import math
import time
import copy

import numpy as np
from pynq import allocate
from pynq.buffer import PynqBuffer

from vision_int8_im2col_layout import unpack_row_tiled_output


PTRS = {
    "input": 0x18,
    "weights": 0x24,
    "bias": 0x30,
    "multipliers": 0x3C,
    "shifts": 0x48,
    "output": 0x54,
    "descriptor": 0x60,
    "output_lut": 0x6C,
}
DESCRIPTOR_WORDS = 23
MAX_INPUT_BYTES = 1_048_576
MAX_WEIGHT_WORDS = 1_048_576
MAX_OUTPUT_WORDS = 131_072


def _set_ptr(ip, offset: int, address: int) -> None:
    ip.write(offset, int(address) & 0xFFFFFFFF)
    ip.write(offset + 4, (int(address) >> 32) & 0xFFFFFFFF)


def _quantize_multiplier(real_scale: float) -> tuple[int, int]:
    if not np.isfinite(real_scale) or real_scale < 0:
        raise ValueError(f"invalid requantization scale {real_scale}")
    if real_scale == 0:
        return 0, 0
    significand, shift = math.frexp(real_scale)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1
    if shift < -31 or shift > 30:
        raise ValueError(f"requantization shift {shift} is outside HLS bounds")
    return multiplier, shift


def _conv_geometry(node, input_shape, weights, input_zero, output_zero,
                   fused_output_lut=None):
    if len(input_shape) != 4 or input_shape[0] != 1 or len(weights.shape) != 4:
        raise ValueError(f"{node['name']}: expected batch-one NCHW and OIHW weights")
    _, channels, height, width = map(int, input_shape)
    out_channels, channels_per_group, kh, kw = map(int, weights.shape)
    attrs = node["attrs"]
    groups = int(attrs.get("group", 1))
    sh, sw = map(int, attrs.get("strides", (1, 1)))
    dh, dw = map(int, attrs.get("dilations", (1, 1)))
    pt, pl, pb, pr = map(int, attrs.get("pads", (0, 0, 0, 0)))
    if groups < 1 or channels != groups * channels_per_group or out_channels % groups:
        raise ValueError(f"{node['name']}: channel/group dimensions disagree")
    if min(sh, sw, dh, dw, kh, kw, height, width) < 1:
        raise ValueError(f"{node['name']}: invalid convolution geometry")
    if sw > 2:
        raise ValueError(f"{node['name']}: line-cache core supports width stride 1 or 2")
    oh = (height + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (width + pl + pr - dw * (kw - 1) - 1) // sw + 1
    k = channels_per_group * kh * kw
    m = out_channels // groups
    if (oh < 1 or ow < 1 or height > 8192 or width > 512 or
            oh > 8192 or ow > 512 or channels_per_group > 512 or
            m > 512 or groups > 512 or kh > 15 or kw > 15 or k > 4096 or
            groups * channels_per_group * height * width > MAX_INPUT_BYTES):
        raise ValueError(f"{node['name']}: geometry exceeds line-cache HLS bounds")
    if not (0 <= int(input_zero) <= 255 and 0 <= int(output_zero) <= 255):
        raise ValueError(f"{node['name']}: zero point is outside UINT8 range")
    descriptor = np.asarray([
        m, oh * ow, k, groups, int(input_zero), int(output_zero),
        3 if fused_output_lut is not None else 0, 255,
        height, width, channels_per_group, oh, ow, kh, kw,
        sh, sw, dh, dw, pt, pl, pb, pr,
    ], dtype=np.int32)
    return descriptor, (1, out_channels, oh, ow), groups, m, k


class LineCacheInt8PlBackend:
    """Drive the v14 HLS core with optional exact UINT8 activation fusion."""

    supports_output_lut = True

    def __init__(self, overlay, batch_mmio=False):
        self.ip = overlay.vision_conv_0
        self.batch_mmio = bool(batch_mmio)
        self.pointer_word_indices = np.asarray([
            word for offset in PTRS.values() for word in (offset // 4, offset // 4 + 1)
        ], dtype=np.intp)
        if self.batch_mmio:
            registers = self.ip.mmio.array
            if not (isinstance(registers, np.ndarray) and registers.dtype == np.uint32
                    and registers.ndim == 1 and registers.size > int(self.pointer_word_indices.max())):
                raise TypeError("batch MMIO requires a local mapped uint32 register array")
        self.input_buffer = None
        self.output_buffer = None
        self.layers = {}
        self._pl_output_ranges = []
        self.closed = False
        self.totals = {
            "input_copy_s": 0.0, "prepare_s": 0.0, "cache_clean_s": 0.0,
            "mmio_s": 0.0, "kernel_s": 0.0, "output_copy_s": 0.0,
            "call_s": 0.0, "calls": 0,
        }

    @staticmethod
    def _ensure_buffer(owner, name, size, dtype, limit):
        if size < 1 or size > limit:
            raise ValueError(f"{name} length {size} is outside 1..{limit}")
        current = getattr(owner, name)
        if current is None or current.size < size:
            if current is not None:
                current.freebuffer()
            current = allocate(shape=(size,), dtype=dtype)
            setattr(owner, name, current)
        return current

    def _prepare_layer(self, node, input_shape, input_scale, input_zero,
                       weights, weight_scales, weight_zero, output_scale,
                       output_zero, bias, fused_output_lut=None):
        name = node["name"]
        weights = np.asarray(weights)
        bias = np.asarray(bias)
        weight_scales = np.asarray(weight_scales, dtype=np.float64).reshape(-1)
        weight_zero = np.asarray(weight_zero)
        if weights.dtype != np.int8 or bias.dtype != np.int32 or np.any(weight_zero != 0):
            raise ValueError(f"{name}: expected symmetric INT8 weights and INT32 bias")
        out_channels = int(weights.shape[0])
        if bias.size != out_channels or weight_scales.size not in (1, out_channels):
            raise ValueError(f"{name}: bias or per-channel weight scale shape mismatch")
        lut = None
        if fused_output_lut is not None:
            lut = np.asarray(fused_output_lut, dtype=np.uint8).reshape(-1)
            if lut.size != 256:
                raise ValueError(f"{name}: fused activation LUT must have 256 bytes")
            lut = np.ascontiguousarray(lut)
        descriptor, output_shape, groups, m, k = _conv_geometry(
            node, input_shape, weights, input_zero, output_zero, lut)
        key = name
        cached = self.layers.get(key)
        if cached is not None:
            if (cached["descriptor"].shape != descriptor.shape or
                    not np.array_equal(cached["descriptor"], descriptor)):
                raise ValueError(f"{name}: descriptor changed between frames")
            cached_lut = cached.get("fused_output_lut")
            if ((cached_lut is None) != (lut is None) or
                    (lut is not None and not np.array_equal(cached_lut, lut))):
                raise ValueError(f"{name}: fused output LUT changed between frames")
            return cached

        mblocks = (m + 7) // 8
        packed = np.zeros((groups, mblocks, k, 8), dtype=np.uint8)
        for group in range(groups):
            for channel in range(m):
                block, lane = divmod(channel, 8)
                packed[group, block, :, lane] = weights[
                    group * m + channel
                ].reshape(-1).view(np.uint8)
        packed_words = np.ascontiguousarray(packed).view("<u8").reshape(-1)
        if packed_words.size > MAX_WEIGHT_WORDS:
            raise ValueError(f"{name}: packed weights exceed HLS AXI limit")

        scales = np.broadcast_to(weight_scales, (out_channels,))
        multipliers, shifts = zip(*[
            _quantize_multiplier(float(input_scale) * float(scales[ch]) /
                                 float(output_scale))
            for ch in range(out_channels)
        ])
        metadata = allocate(shape=(3 * out_channels + DESCRIPTOR_WORDS,),
                            dtype=np.int32)
        metadata[:out_channels] = bias.reshape(-1)
        metadata[out_channels:2 * out_channels] = np.asarray(multipliers, dtype=np.int32)
        metadata[2 * out_channels:3 * out_channels] = np.asarray(shifts, dtype=np.int32)
        metadata[3 * out_channels:] = descriptor
        weight_buffer = allocate(shape=packed_words.shape, dtype=np.uint64)
        np.copyto(weight_buffer, packed_words)
        weight_buffer.flush()
        lut_buffer = None
        if lut is not None:
            lut_buffer = allocate(shape=(32,), dtype=np.uint64)
            np.copyto(lut_buffer, lut.view("<u8"))
            lut_buffer.flush()
        metadata.flush()
        cached = {
            "weights": weight_buffer, "metadata": metadata,
            "output_lut": lut_buffer, "fused_output_lut": lut,
            "out_channels": out_channels, "groups": groups, "m": m,
            "descriptor": descriptor, "output_shape": output_shape,
            "output_buffer": None,
            "host_copy_output": bool(node.get("host_copy_output", True)),
        }
        self.layers[key] = cached
        return cached

    def bind_conv(self, node, input_scale, input_zero, weights, weight_scales,
                  weight_zero, output_scale, output_zero, bias):
        """Bind immutable model parameters once; specialize shape on first call.

        This caches host-side preparation, not hardware register contents.
        The shared IP still receives every argument pointer on every launch.
        """
        frozen_node = copy.deepcopy(node)
        constants = (weights, weight_scales, weight_zero, bias)
        if any(not isinstance(value, np.ndarray) for value in constants):
            raise TypeError("bound convolution parameters must be NumPy arrays")
        for value in constants:
            value.setflags(write=False)
        sx = float(np.asarray(input_scale).item())
        zx = int(np.asarray(input_zero).item())
        sy = float(np.asarray(output_scale).item())
        zy = int(np.asarray(output_zero).item())
        lut = frozen_node.get("fused_output_lut")
        if lut is not None:
            lut.setflags(write=False)
        layer = None
        fixed_shape = None

        def execute(qx):
            nonlocal layer, fixed_shape
            started = time.perf_counter()
            if self.closed:
                raise RuntimeError("prepared convolution backend has been released")
            qx = np.asanyarray(qx)
            if qx.dtype != np.uint8 or qx.ndim != 4 or qx.shape[0] != 1:
                raise ValueError(f"{node['name']}: expected batch-one NCHW UINT8")
            if layer is None:
                layer = self._prepare_layer(
                    frozen_node, qx.shape, sx, zx, weights, weight_scales,
                    weight_zero, sy, zy, bias, lut)
                fixed_shape = qx.shape
            elif qx.shape != fixed_shape:
                raise ValueError(f"{node['name']}: prepared input shape changed")
            if not qx.flags.c_contiguous:
                qx = np.ascontiguousarray(qx)
            prepared_at = time.perf_counter()
            return self._execute_layer(frozen_node, qx, layer, started, prepared_at)

        return execute

    def run(self, node, qx, input_scale, input_zero, weights, weight_scales,
            weight_zero, output_scale, output_zero, bias):
        started = time.perf_counter()
        qx = np.asanyarray(qx)
        if qx.dtype != np.uint8 or qx.ndim != 4 or qx.shape[0] != 1:
            raise ValueError(f"{node['name']}: expected batch-one NCHW UINT8")
        if not qx.flags.c_contiguous:
            qx = np.ascontiguousarray(qx)
        layer = self._prepare_layer(
            node, qx.shape, float(np.asarray(input_scale).item()), int(input_zero),
            weights, weight_scales, weight_zero, float(np.asarray(output_scale).item()),
            int(output_zero), bias, node.get("fused_output_lut"))
        prepared_at = time.perf_counter()
        return self._execute_layer(node, qx, layer, started, prepared_at)

    def _execute_layer(self, node, qx, layer, started, prepared_at):
        input_flat = qx.reshape(-1)
        # A contiguous PynqBuffer view can be passed directly to the IP.
        # Only reuse one of our PL-written outputs: the graph never mutates
        # those tensors, so no CPU cache flush is needed after invalidation.
        address = getattr(input_flat, "physical_address", None)
        from_pl = (isinstance(input_flat, PynqBuffer) and address is not None and
                   input_flat.flags.c_contiguous and
                   0 < input_flat.size <= MAX_INPUT_BYTES and
                   any(start <= address and address + input_flat.nbytes <= end
                       for start, end in self._pl_output_ranges))
        if from_pl:
            input_buffer = input_flat
        else:
            input_buffer = self._ensure_buffer(
                self, "input_buffer", input_flat.size, np.uint8, MAX_INPUT_BYTES)
            np.copyto(input_buffer[:input_flat.size], input_flat)
        copied_at = time.perf_counter()

        _, out_channels, out_h, out_w = layer["output_shape"]
        blocks_per_row = (out_w + 7) // 8
        output_words = out_channels * out_h * blocks_per_row
        # Graph skip connections retain earlier tensors. Give each bound
        # convolution its own small CMA output so a later layer cannot
        # overwrite a still-live tensor.
        if output_words < 1 or output_words > MAX_OUTPUT_WORDS:
            raise ValueError("output size exceeds HLS limit")
        if layer["host_copy_output"]:
            output_buffer = self._ensure_buffer(
                self, "output_buffer", output_words, np.uint64, MAX_OUTPUT_WORDS)
        else:
            output_buffer = layer["output_buffer"]
            if output_buffer is None:
                output_buffer = allocate(shape=(output_words,), dtype=np.uint64)
                layer["output_buffer"] = output_buffer
                address = int(output_buffer.physical_address)
                self._pl_output_ranges.append((address, address + output_buffer.nbytes))
        # Default XRT BOs are mapped noncacheable on this board. No CPU cache
        # lines exist to flush before the PL reads this input buffer.
        cleaned_at = time.perf_counter()

        self._program_pointers(layer, input_buffer.physical_address,
                               output_buffer.physical_address)
        kernel_started = time.perf_counter()
        self.ip.write(0x00, 1)
        deadline = kernel_started + 5.0
        while not (self.ip.read(0x00) & 0x2):
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"{node['name']}: line-cache HLS core timed out")
        status = int(self.ip.read(0x10))
        if status:
            raise RuntimeError(f"{node['name']}: line-cache HLS status {status}")
        kernel_done = time.perf_counter()

        # Likewise the PL output BO has no cache lines to invalidate.
        if out_w % 8 == 0 and not layer["host_copy_output"]:
            # All 96 current pose convolutions have aligned widths. A view
            # preserves the physical address and avoids a host-side copy.
            result = output_buffer.view(np.uint8).reshape(
                1, out_channels, out_h, out_w)
        else:
            result = unpack_row_tiled_output(
                np.asarray(output_buffer[:output_words]), layer["groups"],
                layer["m"], out_h, out_w)
        finished = time.perf_counter()
        self.totals["prepare_s"] += prepared_at - started
        self.totals["input_copy_s"] += copied_at - prepared_at
        self.totals["cache_clean_s"] += cleaned_at - copied_at
        self.totals["mmio_s"] += kernel_started - cleaned_at
        self.totals["kernel_s"] += kernel_done - kernel_started
        self.totals["output_copy_s"] += finished - kernel_done
        self.totals["call_s"] += finished - started
        self.totals["calls"] += 1
        return result

    def _program_pointers(self, layer, input_address, output_address):
        metadata = layer["metadata"]
        ch = layer["out_channels"]
        base = int(metadata.physical_address)
        lut_buffer = layer["output_lut"]
        addresses = (
            input_address, layer["weights"].physical_address, base,
            base + 4 * ch, base + 8 * ch, output_address, base + 12 * ch,
            0 if lut_buffer is None else lut_buffer.physical_address,
        )
        if self.batch_mmio:
            words = layer.get("pointer_words")
            if words is None:
                words = np.empty(16, dtype=np.uint32)
                layer["pointer_words"] = words
            for i, address in enumerate(addresses):
                address = int(address)
                words[2 * i] = address & 0xffffffff
                words[2 * i + 1] = address >> 32
            # Each selected uint32 address is an argument register from the
            # generated v14 register map. No control/status/reserved word is
            # included. Rewrite ALL pointers on EVERY layer launch.
            self.ip.mmio.array[self.pointer_word_indices] = words
        else:
            for offset, address in zip(PTRS.values(), addresses):
                _set_ptr(self.ip, offset, address)

    def release(self):
        self.closed = True
        for layer in self.layers.values():
            layer["weights"].freebuffer()
            layer["metadata"].freebuffer()
            if layer["output_lut"] is not None:
                layer["output_lut"].freebuffer()
            if layer["output_buffer"] is not None:
                layer["output_buffer"].freebuffer()
        self.layers.clear()
        for name in ("input_buffer", "output_buffer"):
            buffer = getattr(self, name)
            if buffer is not None:
                buffer.freebuffer()
                setattr(self, name, None)

