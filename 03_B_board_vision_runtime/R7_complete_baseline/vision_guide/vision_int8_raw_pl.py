"""Raw-NCHW INT8 convolution backend for the R13 single-master HLS core."""

from __future__ import annotations

import time

import numpy as np
from pynq import allocate

from pl_conv_engine import load_native, quantize_multiplier, set_ptr


PTRS = {
    "input": 0x18,
    "weights": 0x24,
    "bias": 0x30,
    "multipliers": 0x3C,
    "shifts": 0x48,
    "output": 0x54,
    "descriptor": 0x60,
}
DESCRIPTOR_WORDS = 19
MAX_BYTES = 1_048_576
MAX_WEIGHT_BYTES = 1_048_576


def convolution_descriptor(node, input_shape, weights, input_zero, output_zero):
    """Return the HLS geometry descriptor and output NCHW shape."""
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(f"{node['name']}: expected NCHW batch size 1")
    if len(weights.shape) != 4:
        raise ValueError(f"{node['name']}: expected OIHW convolution weights")

    _, in_channels, height, width = map(int, input_shape)
    out_channels, channels_per_group, kernel_h, kernel_w = map(int, weights.shape)
    attrs = node["attrs"]
    groups = int(attrs.get("group", 1))
    strides = tuple(map(int, attrs.get("strides", (1, 1))))
    dilations = tuple(map(int, attrs.get("dilations", (1, 1))))
    pads = tuple(map(int, attrs.get("pads", (0, 0, 0, 0))))
    if len(strides) != 2 or len(dilations) != 2 or len(pads) != 4:
        raise ValueError(f"{node['name']}: malformed convolution attributes")
    if dilations != (1, 1):
        raise ValueError(f"{node['name']}: HLS v8 supports dilation 1 only")
    if groups < 1 or in_channels != groups * channels_per_group:
        raise ValueError(f"{node['name']}: input/group dimensions disagree")
    if out_channels % groups:
        raise ValueError(f"{node['name']}: output channels are not divisible by groups")
    if min(strides) < 1 or min(kernel_h, kernel_w, height, width) < 1:
        raise ValueError(f"{node['name']}: invalid convolution dimensions")

    pad_top, pad_left, pad_bottom, pad_right = pads
    out_h = (height + pad_top + pad_bottom - kernel_h) // strides[0] + 1
    out_w = (width + pad_left + pad_right - kernel_w) // strides[1] + 1
    if out_h < 1 or out_w < 1:
        raise ValueError(f"{node['name']}: convolution output is empty")
    if (height > 8192 or width > 512 or in_channels > 512 or
            out_channels > 512 or out_channels // groups > 512):
        raise ValueError(f"{node['name']}: channel count exceeds HLS v8 bounds")
    if (out_h > 8192 or out_w > 512 or max(kernel_h, kernel_w) > 15 or
            max(strides) > 16 or max(pad_top, pad_left) > 15):
        raise ValueError(f"{node['name']}: geometry exceeds HLS v8 bounds")
    if not (0 <= int(input_zero) <= 255 and 0 <= int(output_zero) <= 255):
        raise ValueError(f"{node['name']}: zero point is outside UINT8 range")
    if channels_per_group * kernel_h * kernel_w > 2304:
        raise ValueError(f"{node['name']}: reduction exceeds HLS v8 bound")

    descriptor = np.asarray([
        height, width, in_channels, out_channels, out_h, out_w,
        kernel_h, kernel_w, strides[0], strides[1], pad_top, pad_left,
        groups, channels_per_group, out_channels // groups,
        int(input_zero), int(output_zero), 0, 255,
    ], dtype=np.int32)
    return descriptor, (1, out_channels, out_h, out_w)


class RawNchwInt8PlBackend:
    """Drive a raw feature-map HLS core; no PS im2col or output unpacking."""

    def __init__(self, overlay, native_library):
        self.ip = overlay.vision_conv_0
        self.native = load_native(native_library)
        self.input_buffer = None
        self.output_buffer = None
        self.layers = {}
        self.totals = {
            "input_copy_s": 0.0,
            "prepare_s": 0.0,
            "cache_clean_s": 0.0,
            "mmio_s": 0.0,
            "kernel_s": 0.0,
            "output_copy_s": 0.0,
            "call_s": 0.0,
            "calls": 0,
        }

    def _ensure_buffer(self, attr, size):
        if size < 1 or size > MAX_BYTES:
            raise ValueError(f"HLS v8 buffer size {size} is outside 1..{MAX_BYTES}")
        current = getattr(self, attr)
        if current is None or current.size < size:
            if current is not None:
                current.freebuffer()
            current = allocate(shape=(size,), dtype=np.uint8)
            setattr(self, attr, current)
        return current

    def _prepare_layer(self, node, weights, weight_scales, weight_zero,
                       output_scale, bias, input_scale, input_zero, output_zero,
                       input_shape):
        name = node["name"]
        weights = np.asarray(weights)
        bias = np.asarray(bias)
        weight_scales = np.asarray(weight_scales, dtype=np.float64).reshape(-1)
        weight_zero = np.asarray(weight_zero)
        if weights.dtype != np.int8 or bias.dtype != np.int32:
            raise ValueError(f"{name}: expected INT8 weights and INT32 bias")
        if np.any(weight_zero != 0):
            raise ValueError(f"{name}: HLS v8 requires symmetric INT8 weights")
        if weights.size > MAX_WEIGHT_BYTES:
            raise ValueError(f"{name}: weight tensor exceeds HLS v8 AXI range")
        out_channels = int(weights.shape[0])
        if bias.size != out_channels or weight_scales.size not in (1, out_channels):
            raise ValueError(f"{name}: bias or per-channel scale shape mismatch")

        descriptor, output_shape = convolution_descriptor(
            node, input_shape, weights, input_zero, output_zero)
        if name in self.layers:
            cached = self.layers[name]
            if (cached["descriptor"].shape != descriptor.shape or
                    not np.array_equal(cached["descriptor"], descriptor)):
                raise ValueError(f"{name}: convolution descriptor changed between frames")
            return cached

        scales = np.broadcast_to(weight_scales, (out_channels,))
        multipliers, shifts = zip(*[
            quantize_multiplier(float(input_scale) * float(scales[channel]) /
                                float(output_scale))
            for channel in range(out_channels)
        ])
        metadata = allocate(shape=(3 * out_channels + DESCRIPTOR_WORDS,), dtype=np.int32)
        metadata[:out_channels] = bias.reshape(-1)
        metadata[out_channels:2 * out_channels] = np.asarray(multipliers, dtype=np.int32)
        metadata[2 * out_channels:3 * out_channels] = np.asarray(shifts, dtype=np.int32)
        descriptor_offset = 3 * out_channels
        metadata[descriptor_offset:] = descriptor
        weight_buffer = allocate(shape=(weights.size,), dtype=np.int8)
        np.copyto(weight_buffer, np.ascontiguousarray(weights).reshape(-1))
        weight_buffer.flush()
        metadata.flush()
        prepared = {
            "weights": weight_buffer,
            "metadata": metadata,
            "out_channels": out_channels,
            "descriptor_offset": descriptor_offset,
            "descriptor": descriptor,
            "output_shape": output_shape,
        }
        self.layers[name] = prepared
        return prepared

    def run(self, node, qx, input_scale, input_zero, weights, weight_scales,
            weight_zero, output_scale, output_zero, bias):
        started = time.perf_counter()
        qx = np.asarray(qx)
        if qx.dtype != np.uint8 or qx.ndim != 4 or qx.shape[0] != 1:
            raise ValueError(f"{node['name']}: expected NCHW UINT8 batch 1")
        qx = np.ascontiguousarray(qx)
        layer = self._prepare_layer(
            node, weights, weight_scales, weight_zero, output_scale, bias,
            input_scale, input_zero, output_zero, qx.shape)
        prepared_at = time.perf_counter()

        input_flat = qx.reshape(-1)
        if input_flat.size > MAX_BYTES:
            raise ValueError(f"{node['name']}: input feature map exceeds HLS v8 AXI range")
        input_buffer = self._ensure_buffer("input_buffer", input_flat.size)
        np.copyto(input_buffer[:input_flat.size], input_flat)
        copied_at = time.perf_counter()

        output_size = int(np.prod(layer["output_shape"]))
        output_buffer = self._ensure_buffer("output_buffer", output_size)
        input_buffer[:input_flat.size].flush()
        cleaned_at = time.perf_counter()

        metadata = layer["metadata"]
        descriptor_address = metadata.physical_address + 4 * layer["descriptor_offset"]
        set_ptr(self.ip, PTRS["input"], input_buffer.physical_address)
        set_ptr(self.ip, PTRS["weights"], layer["weights"].physical_address)
        set_ptr(self.ip, PTRS["bias"], metadata.physical_address)
        set_ptr(self.ip, PTRS["multipliers"], metadata.physical_address +
                4 * layer["out_channels"])
        set_ptr(self.ip, PTRS["shifts"], metadata.physical_address +
                8 * layer["out_channels"])
        set_ptr(self.ip, PTRS["output"], output_buffer.physical_address)
        set_ptr(self.ip, PTRS["descriptor"], descriptor_address)
        kernel_started = time.perf_counter()
        self.ip.write(0x00, 1)
        deadline = kernel_started + 5.0
        while not (self.ip.read(0x00) & 0x2):
            if time.perf_counter() >= deadline:
                raise TimeoutError(f"{node['name']}: raw-NCHW vision core timed out")
        status = int(self.ip.read(0x10))
        if status:
            raise RuntimeError(f"{node['name']}: raw-NCHW vision status {status}")
        kernel_done = time.perf_counter()

        output_buffer[:output_size].invalidate()
        result = np.array(output_buffer[:output_size], dtype=np.uint8, copy=True)
        result = result.reshape(layer["output_shape"])
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

    def release(self):
        for layer in self.layers.values():
            layer["weights"].freebuffer()
            layer["metadata"].freebuffer()
        self.layers.clear()
        for attr in ("input_buffer", "output_buffer"):
            buffer = getattr(self, attr)
            if buffer is not None:
                buffer.freebuffer()
                setattr(self, attr, None)
