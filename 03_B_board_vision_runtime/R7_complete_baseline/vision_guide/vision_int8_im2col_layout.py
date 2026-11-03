"""Host layout adapter for the row-tiled v12 raw-activation HLS output."""

from __future__ import annotations

import numpy as np


def unpack_row_tiled_output(words: np.ndarray, groups: int, channels_per_group: int,
                            output_height: int, output_width: int) -> np.ndarray:
    """Remove each row's 8-byte tail padding and return dense NCHW UINT8.

    The HLS kernel stores one 64-bit word per output-channel and per group of
    eight adjacent pixels *within each row*. A flat ``[:, :H*W]`` slice is
    incorrect when ``W % 8 != 0`` because it retains row-tail padding.
    """
    if words.dtype != np.uint64 or words.ndim != 1:
        raise ValueError("packed output must be a flat uint64 array")
    if min(groups, channels_per_group, output_height, output_width) < 1:
        raise ValueError("output dimensions must be positive")
    channels = groups * channels_per_group
    blocks_per_row = (output_width + 7) // 8
    expected_words = channels * output_height * blocks_per_row
    if words.size != expected_words:
        raise ValueError(f"expected {expected_words} words, received {words.size}")

    packed_bytes = words.view(np.uint8).reshape(
        channels, output_height, blocks_per_row, 8
    )
    dense = packed_bytes[:, :, :].reshape(
        channels, output_height, blocks_per_row * 8
    )[:, :, :output_width]
    return np.array(dense.reshape(1, channels, output_height, output_width),
                    dtype=np.uint8, copy=True, order="C")
