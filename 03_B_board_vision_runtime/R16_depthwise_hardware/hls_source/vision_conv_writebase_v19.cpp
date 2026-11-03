#include <ap_int.h>
#include <stdint.h>

// Keep the external 8-channel weight packing and register ABI unchanged.
#ifndef PULSEATLAS_OC_BLOCKS
#define PULSEATLAS_OC_BLOCKS 2
#endif
static const int kOCBlocks = PULSEATLAS_OC_BLOCKS;
static_assert(kOCBlocks == 1 || kOCBlocks == 2, "one or two output-channel blocks");
static const int kLanes = 8;
static const int kOutputs = kOCBlocks * kLanes;
static const int kMaxK = 4096;
static const int kMaxGroupWeightWords = 32768;
// Retain a small spatial tile of im2col words on chip while all output-channel
// blocks consume it. This bounds BRAM use independently of the full feature map.
static const int kPatchTileBlocks = 4;
// A 4-block output tile spans at most 287 input bytes for the supported
// stride/dilation/kernel limits. Include up to seven bytes of word alignment.
static const int kMaxLineWords = 40;

static int32_t round_shift_even(int64_t value, ap_uint<6> shift) {
#pragma HLS INLINE
    // Each active channel is validated to -31 <= sh <= 30 before MAC.
    // Hence 31-sh is in 1..62: there is no left-shift/saturation path.
    // Preserve the original int32 conversion, signed ties-to-even and clipping.
    bool neg = value < 0;
    uint64_t mag = neg ? (uint64_t)(-value) : (uint64_t)value;
    uint64_t whole = mag >> shift;
    uint64_t rem = mag & (((uint64_t)1 << shift) - 1);
    uint64_t half = (uint64_t)1 << (shift - 1);
    int64_t rounded = whole + (rem > half || (rem == half && (whole & 1)));
    return (int32_t)(neg ? -rounded : rounded);
}

static uint8_t requant(int32_t acc, int32_t mult, int32_t shift,
                       int zero, int activation, int act_max) {
#pragma HLS INLINE
    int64_t value = round_shift_even((int64_t)acc * mult, ap_uint<6>(31 - shift));
    value += zero;
    if ((activation == 1 || activation == 2) && value < zero) value = zero;
    if (activation == 2 && value > act_max) value = act_max;
    if (value < 0) value = 0;
    if (value > 255) value = 255;
    return (uint8_t)value;
}

// Depthwise 3x3 uses one output channel per group.  The generic output-channel
// tile spends most of its work on 15 invalid channels in this case.  Keep the
// same NCHW input, packed weight and row-tiled output ABI, but compute only the
// one real channel.  Widths used by the current pose graph are multiples of 8.
static int32_t depthwise3x3_aligned(
    const ap_uint<64>* input, const ap_uint<64>* weights,
    const int32_t* bias, const int32_t* multipliers_q31,
    const int32_t* shifts, ap_uint<64>* output,
    const ap_uint<64>* output_lut, int groups, int input_h, int input_w,
    int iz, int oz, int activation, int act_max) {
    const int row_words = input_w / 8;
    const int plane_words = input_h * row_words;
    uint8_t fused_lut[256];
#pragma HLS ARRAY_PARTITION variable=fused_lut cyclic factor=8 dim=1
    if (activation == 3) {
        for (int word_index = 0; word_index < 32; ++word_index) {
#pragma HLS PIPELINE II=1
            const ap_uint<64> lut_word = output_lut[word_index];
            for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                fused_lut[word_index * 8 + lane] =
                    (uint8_t)lut_word.range(lane * 8 + 7, lane * 8);
            }
        }
    }
    for (int group = 0; group < groups; ++group) {
        const int32_t bs = bias[group];
        const int32_t mq = multipliers_q31[group];
        const int32_t sh = shifts[group];
        if (sh < -31 || sh > 30) return 2;
        ap_int<8> taps[9];
#pragma HLS ARRAY_PARTITION variable=taps complete
        for (int r = 0; r < 9; ++r) {
#pragma HLS PIPELINE II=1
            taps[r] = weights[group * 9 + r].range(7, 0);
        }
        ap_uint<64> row_cache[3][64];
#pragma HLS ARRAY_PARTITION variable=row_cache complete dim=1
#pragma HLS BIND_STORAGE variable=row_cache type=ram_2p impl=bram
        const ap_uint<64> pad_word =
            (ap_uint<64>)(uint8_t)iz * 0x0101010101010101ULL;
        for (int y = 0; y < input_h; ++y) {
            // Read each source row as a sequential burst instead of issuing
            // nine independent DDR reads for every output word.
            for (int ky = 0; ky < 3; ++ky) {
                const int iy = y + ky - 1;
                const int row_base = group * plane_words + iy * row_words;
                for (int xb = 0; xb < row_words; ++xb) {
#pragma HLS PIPELINE II=1
                    row_cache[ky][xb] = iy >= 0 && iy < input_h
                        ? input[row_base + xb] : pad_word;
                }
            }
            for (int xb = 0; xb < row_words; ++xb) {
                int32_t acc8[8];
#pragma HLS ARRAY_PARTITION variable=acc8 complete
                for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                    acc8[p] = bs;
                }
                for (int ky = 0; ky < 3; ++ky) {
                    const ap_uint<64> left = xb > 0
                        ? row_cache[ky][xb - 1] : pad_word;
                    const ap_uint<64> center = row_cache[ky][xb];
                    const ap_uint<64> right = xb + 1 < row_words
                        ? row_cache[ky][xb + 1] : pad_word;
                    for (int kx = 0; kx < 3; ++kx) {
#pragma HLS PIPELINE II=1
                        const ap_uint<64> pv = kx == 0
                            ? (left >> 56) | (center << 8)
                            : (kx == 1 ? center : (center >> 8) | (right << 56));
                        for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                            const ap_int<9> x =
                                (ap_int<9>)pv.range(p * 8 + 7, p * 8) - (ap_int<9>)iz;
                            const ap_int<17> product = x * taps[ky * 3 + kx];
                            acc8[p] += (int32_t)product;
                        }
                    }
                }
                ap_uint<64> word = 0;
                for (int p = 0; p < 8; ++p) {
#pragma HLS PIPELINE II=1
                    const uint8_t raw = requant(acc8[p], mq, sh, oz,
                                                activation, act_max);
                    const uint8_t value = activation == 3
                        ? fused_lut[raw] : raw;
                    word.range(p * 8 + 7, p * 8) = value;
                }
                output[group * plane_words + y * row_words + xb] = word;
            }
        }
    }
    return 0;
}

// Fused raw-activation im2col + packed GEMM + optional UINT8 SiLU LUT. The descriptor is:
// [M,N,K,G,iz,oz,activation,activation_max,IH,IW,C/G,OH,OW,KH,KW,
//  SH,SW,DH,DW,PT,PL,PB,PR]. Input is NCHW UINT8; weights are
// [G,ceil(M/8),K] packed by output channel; output is [G,M,ceil(N/8)].
extern "C" int32_t vision_conv_writebase_v19(
    const ap_uint<64>* input, const ap_uint<64>* weights,
    const int32_t* bias, const int32_t* multipliers_q31, const int32_t* shifts,
    ap_uint<64>* output, const int32_t* descriptor,
    const ap_uint<64>* output_lut) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=ACT depth=131072 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=PARAM depth=1048576 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=PARAM depth=512
#pragma HLS INTERFACE m_axi port=multipliers_q31 offset=slave bundle=PARAM depth=512
#pragma HLS INTERFACE m_axi port=shifts offset=slave bundle=PARAM depth=512
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=OUT depth=131072 max_write_burst_length=64 num_write_outstanding=8
#pragma HLS INTERFACE m_axi port=descriptor offset=slave bundle=PARAM depth=23
#pragma HLS INTERFACE m_axi port=output_lut offset=slave bundle=PARAM depth=32
#pragma HLS INTERFACE s_axilite port=input bundle=CTRL
#pragma HLS INTERFACE s_axilite port=weights bundle=CTRL
#pragma HLS INTERFACE s_axilite port=bias bundle=CTRL
#pragma HLS INTERFACE s_axilite port=multipliers_q31 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=shifts bundle=CTRL
#pragma HLS INTERFACE s_axilite port=output bundle=CTRL
#pragma HLS INTERFACE s_axilite port=descriptor bundle=CTRL
#pragma HLS INTERFACE s_axilite port=output_lut bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
// Share setup/address multipliers, not the separately pipelined MAC datapath.
#pragma HLS ALLOCATION operation instances=mul limit=4
    const int m = descriptor[0], n = descriptor[1], k = descriptor[2];
    const int groups = descriptor[3], iz = descriptor[4], oz = descriptor[5];
    const int activation = descriptor[6], act_max = descriptor[7];
    const int input_h = descriptor[8], input_w = descriptor[9];
    const int input_channels_per_group = descriptor[10];
    const int output_h = descriptor[11], output_w = descriptor[12];
    const int kernel_h = descriptor[13], kernel_w = descriptor[14];
    const int stride_h = descriptor[15], stride_w = descriptor[16];
    const int dilation_h = descriptor[17], dilation_w = descriptor[18];
    const int pad_top = descriptor[19], pad_left = descriptor[20];
    const int pad_bottom = descriptor[21], pad_right = descriptor[22];
    if (m < 1 || m > 512 || n < 1 || n > 65536 || k < 1 || k > kMaxK ||
        groups < 1 || groups > 512 || iz < 0 || iz > 255 || oz < 0 || oz > 255 ||
        activation < 0 || activation > 3 || act_max < 0 || act_max > 255 ||
        input_h < 1 || input_h > 8192 || input_w < 1 || input_w > 512 ||
        input_channels_per_group < 1 || input_channels_per_group > 512 ||
        output_h < 1 || output_h > 8192 || output_w < 1 || output_w > 512 ||
        kernel_h < 1 || kernel_h > 15 || kernel_w < 1 || kernel_w > 15 ||
        stride_h < 1 || stride_h > 16 || stride_w < 1 || stride_w > 16 ||
        dilation_h < 1 || dilation_h > 16 || dilation_w < 1 || dilation_w > 16 ||
        pad_top < 0 || pad_top > 15 || pad_left < 0 || pad_left > 15 ||
        pad_bottom < 0 || pad_bottom > 15 || pad_right < 0 || pad_right > 15 ||
        (uint64_t)groups * input_channels_per_group * input_h * input_w > 1048576ULL ||
        (uint64_t)input_channels_per_group * kernel_h * kernel_w != (uint64_t)k ||
        (uint64_t)output_h * output_w != (uint64_t)n) return 1;
    const int expected_h = (input_h + pad_top + pad_bottom -
                            dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int expected_w = (input_w + pad_left + pad_right -
                            dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (expected_h != output_h || expected_w != output_w) return 1;
    if (stride_w > 2) return 1; // The spatial byte-gather bank covers YOLO stride 1/2.

    if (m == 1 && input_channels_per_group == 1 && kernel_h == 3 &&
        kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1 && pad_top == 1 &&
        pad_left == 1 && pad_bottom == 1 && pad_right == 1 &&
        (input_w & 7) == 0) {
        return depthwise3x3_aligned(input, weights, bias, multipliers_q31,
                                    shifts, output, output_lut, groups, input_h,
                                    input_w, iz, oz, activation, act_max);
    }

    ap_uint<64> wcache[kMaxGroupWeightWords];
#pragma HLS BIND_STORAGE variable=wcache type=ram_2p impl=bram
    ap_uint<64> patch_cache[kPatchTileBlocks * kMaxK];
#pragma HLS BIND_STORAGE variable=patch_cache type=ram_1p impl=bram
    uint8_t fused_lut[256];
#pragma HLS ARRAY_PARTITION variable=fused_lut cyclic factor=8 dim=1
    if (activation == 3) {
        LOAD_FUSED_LUT: for (int word_index = 0; word_index < 32; ++word_index) {
#pragma HLS PIPELINE II=1
            const ap_uint<64> lut_word = output_lut[word_index];
            for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
                fused_lut[word_index * 8 + lane] =
                    (uint8_t)lut_word.range(lane * 8 + 7, lane * 8);
            }
        }
    }
    int32_t bias_cache[8][64], multiplier_cache[8][64], shift_cache[8][64];
#pragma HLS ARRAY_PARTITION variable=bias_cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=multiplier_cache complete dim=1
#pragma HLS ARRAY_PARTITION variable=shift_cache complete dim=1
#pragma HLS BIND_STORAGE variable=bias_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=multiplier_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=shift_cache type=ram_2p impl=bram
    int32_t acc[kOutputs][kLanes];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
    int32_t bs[kOutputs], mq[kOutputs], sh[kOutputs];
#pragma HLS ARRAY_PARTITION variable=bs complete
#pragma HLS ARRAY_PARTITION variable=mq complete
#pragma HLS ARRAY_PARTITION variable=sh complete
    const int mblocks = (m + 7) / 8;
    // Use row-aligned output blocks. This keeps each eight-position activation
    // vector within one image row, including narrow feature maps in the head.
    const int blocks_per_row = (output_w + 7) / 8;
    const int nblocks = output_h * blocks_per_row;
    const int group_weight_words = mblocks * k;
    const uint32_t input_word_count =
        ((uint32_t)groups * input_channels_per_group * input_h * input_w + 7) / 8;
    const uint32_t output_group_stride = (uint32_t)m * (uint32_t)nblocks;
    uint32_t group_output_base = 0;
    GROUPS: for (int group = 0; group < groups; ++group) {
        // Read per-channel quantization metadata once per group. Patch tiling
        // revisits every output-channel block, so keep these small arrays on
        // chip instead of issuing the same three DDR reads for every tile.
        LOAD_METADATA: for (int ch = 0; ch < m; ++ch) {
#pragma HLS PIPELINE II=1
            const int global_ch = group * m + ch;
            const int lane = ch & 7;
            const int block = ch >> 3;
            bias_cache[lane][block] = bias[global_ch];
            multiplier_cache[lane][block] = multipliers_q31[global_ch];
            shift_cache[lane][block] = shifts[global_ch];
        }
        const bool cache_all_weights = group_weight_words <= kMaxGroupWeightWords;
        if (cache_all_weights) {
            CACHE_GROUP_WEIGHTS: for (int index = 0; index < group_weight_words; ++index) {
#pragma HLS PIPELINE II=1
                wcache[index] = weights[group * group_weight_words + index];
            }
        }
        PATCH_ROWS: for (int output_y = 0; output_y < output_h; ++output_y) {
            const int input_y_base = output_y * stride_h - pad_top;
            PATCH_TILES: for (int tile_x_base = 0; tile_x_base < blocks_per_row;
                           tile_x_base += kPatchTileBlocks) {
                int tile_blocks = blocks_per_row - tile_x_base;
                if (tile_blocks > kPatchTileBlocks) tile_blocks = kPatchTileBlocks;
                const int tile_base = output_y * blocks_per_row + tile_x_base;
                const int tile_output_x = tile_x_base * 8;
                const int tile_output_pixels = tile_blocks * 8;
                const int line_x_start = tile_output_x * stride_w - pad_left;
                const int line_x_end =
                    (tile_output_x + tile_output_pixels - 1) * stride_w - pad_left +
                    (kernel_w - 1) * dilation_w;
                ap_uint<64> line_cache[kMaxLineWords];
#pragma HLS BIND_STORAGE variable=line_cache type=ram_1p impl=bram
                // 1x1 / stride-one / no-padding needs no byte gather. For
                // aligned NCHW rows each spatial block is already one AXI word.
                // Keep the exact existing cache layout consumed by STREAM_MAC.
                const bool direct_pointwise = kernel_h == 1 && kernel_w == 1 &&
                    stride_h == 1 && stride_w == 1 &&
                    pad_top == 0 && pad_left == 0 &&
                    pad_bottom == 0 && pad_right == 0 && (input_w & 7) == 0;
                const bool direct_3x3 = kernel_h == 3 && kernel_w == 3 &&
                    stride_h == 1 && stride_w == 1 &&
                    dilation_h == 1 && dilation_w == 1 &&
                    pad_top == 1 && pad_left == 1 &&
                    pad_bottom == 1 && pad_right == 1 &&
                    output_h == input_h && output_w == input_w &&
                    (input_w & 7) == 0;
                if (direct_pointwise) {
                    const int row_words = input_w >> 3;
                    const int plane_words = input_h * row_words;
                    LOAD_POINTWISE_CHANNELS: for (int ic = 0;
                             ic < input_channels_per_group; ++ic) {
                        const int channel = group * input_channels_per_group + ic;
                        const int input_base = channel * plane_words +
                            output_y * row_words + tile_x_base;
                        LOAD_POINTWISE_WORDS: for (int tb = 0; tb < tile_blocks; ++tb) {
#pragma HLS PIPELINE II=1
                            patch_cache[tb * k + ic] = input[input_base + tb];
                        }
                    }
                } else if (direct_3x3) {
                    // One aligned word per eight pixels plus one word of halo on
                    // each side. Shifts construct all three kernel columns; the
                    // replicated input zero point supplies border padding.
                    ap_uint<64> pad_word = 0;
                    for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                        pad_word.range(p * 8 + 7, p * 8) = (uint8_t)iz;
                    }
                    const int row_words = input_w >> 3;
                    const int plane_words = input_h * row_words;
                    FAST3_CHANNELS: for (int ic = 0;
                             ic < input_channels_per_group; ++ic) {
                        const int channel = group * input_channels_per_group + ic;
                        FAST3_ROWS: for (int ky = 0; ky < 3; ++ky) {
                            const int input_y = output_y + ky - 1;
                            ap_uint<64> halo[kPatchTileBlocks + 2];
#pragma HLS ARRAY_PARTITION variable=halo complete
                            const int row_base = channel * plane_words +
                                input_y * row_words;
                            FAST3_LOAD_HALO: for (int j = 0;
                                    j < tile_blocks + 2; ++j) {
#pragma HLS PIPELINE II=1
                                const int word_x = tile_x_base + j - 1;
                                halo[j] = input_y >= 0 && input_y < input_h &&
                                    word_x >= 0 && word_x < row_words
                                    ? input[row_base + word_x] : pad_word;
                            }
                            FAST3_PATCH: for (int tb = 0;
                                    tb < tile_blocks; ++tb) {
                                const ap_uint<64> left = halo[tb];
                                const ap_uint<64> center = halo[tb + 1];
                                const ap_uint<64> right = halo[tb + 2];
                                FAST3_COLUMNS: for (int kx = 0; kx < 3; ++kx) {
#pragma HLS PIPELINE II=1
                                    const ap_uint<64> word = kx == 0
                                        ? (left >> 56) | (center << 8)
                                        : (kx == 1 ? center
                                            : (center >> 8) | (right << 56));
                                    patch_cache[tb * k + ic * 9 + ky * 3 + kx] = word;
                                }
                            }
                        }
                    }
                } else {
                int patch_r = 0;
                LOAD_PATCH_ROWS: for (int ic = 0; ic < input_channels_per_group; ++ic) {
                    const int input_channel = group * input_channels_per_group + ic;
                    LOAD_PATCH_KERNEL_ROWS: for (int kernel_y = 0; kernel_y < kernel_h;
                                                   ++kernel_y) {
                        const int input_y = input_y_base + kernel_y * dilation_h;
                        const uint32_t row_base =
                            ((uint32_t)input_channel * input_h + input_y) * input_w;
                        int line_first_word = 0;
                        int line_words = 0;
                        if (input_y >= 0 && input_y < input_h &&
                            line_x_start < input_w && line_x_end >= 0) {
                            const int valid_x_start = line_x_start < 0 ? 0 : line_x_start;
                            const int valid_x_end = line_x_end >= input_w
                                ? input_w - 1 : line_x_end;
                            if (valid_x_start <= valid_x_end) {
                                line_first_word = (row_base + valid_x_start) >> 3;
                                const int line_last_word = (row_base + valid_x_end) >> 3;
                                line_words = line_last_word - line_first_word + 1;
                                if (line_words > kMaxLineWords) return 3;
                                LOAD_INPUT_ROW: for (int word = 0; word < line_words; ++word) {
#pragma HLS PIPELINE II=1
                                    const uint32_t address = line_first_word + word;
                                    if (address < input_word_count)
                                        line_cache[word] = input[address];
                                    else
                                        line_cache[word] = 0;
                                }
                            }
                        }
                        LOAD_PATCH_KERNEL_COLS: for (int kernel_x = 0; kernel_x < kernel_w;
                                                       ++kernel_x) {
                            LOAD_PATCH_BLOCKS: for (int tb = 0; tb < tile_blocks; ++tb) {
                                const int output_x_base = (tile_x_base + tb) * 8;
                                const int input_x_base = output_x_base * stride_w +
                                                         kernel_x * dilation_w - pad_left;
                                const int first_input_x = input_x_base < 0 ? 0 : input_x_base;
                                const uint32_t first_byte = row_base + first_input_x;
                                const uint32_t first_word = first_byte >> 3;
                                const int first_byte_lane = first_byte & 7;
                                ap_uint<64> input_cache[3];
#pragma HLS ARRAY_PARTITION variable=input_cache complete
                                for (int word = 0; word < 3; ++word) {
#pragma HLS PIPELINE II=1
                                    const int cache_index =
                                        (int)(first_word - line_first_word) + word;
                                    if (input_y >= 0 && input_y < input_h &&
                                        input_x_base <= input_w - 1 &&
                                        input_x_base + 7 * stride_w >= 0 &&
                                        cache_index >= 0 && cache_index < line_words)
                                        input_cache[word] = line_cache[cache_index];
                                    else
                                        input_cache[word] = 0;
                                }
                                ap_uint<192> input_window = 0;
                                input_window.range(63, 0) = input_cache[0];
                                input_window.range(127, 64) = input_cache[1];
                                input_window.range(191, 128) = input_cache[2];
                                const ap_uint<192> aligned_window =
                                    input_window >> (first_byte_lane * 8);
                                ap_uint<64> patch_word = 0;
                                for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                                    const int output_x = output_x_base + p;
                                    const int input_x = input_x_base + p * stride_w;
                                    int value = iz;
                                    if (output_x < output_w && input_y >= 0 &&
                                        input_y < input_h && input_x >= 0 &&
                                        input_x < input_w) {
                                        if (input_x_base >= 0) {
                                            if (stride_w == 1)
                                                value = (int)(uint8_t)aligned_window.range(
                                                    p * 8 + 7, p * 8);
                                            else
                                                value = (int)(uint8_t)aligned_window.range(
                                                    p * 16 + 7, p * 16);
                                        } else {
                                            const int input_byte_offset = first_byte_lane +
                                                                          input_x - first_input_x;
                                            const int local_word = input_byte_offset >> 3;
                                            const int byte_lane = input_byte_offset & 7;
                                            const ap_uint<64> input_word =
                                                input_cache[local_word];
                                            value = (int)(uint8_t)input_word.range(
                                                byte_lane * 8 + 7, byte_lane * 8);
                                        }
                                    }
                                    patch_word.range(p * 8 + 7, p * 8) = (uint8_t)value;
                                }
                                const int r = patch_r + kernel_x;
                                patch_cache[tb * k + r] = patch_word;
                            }
                        }
                        patch_r += kernel_w;
                    }
                }
                } // General im2col fallback.
                uint32_t channel_block_output_base = group_output_base + (uint32_t)tile_base;
                OUTPUT_CHANNEL_BLOCKS: for (int mb = 0; mb < mblocks; mb += kOCBlocks) {
                    METADATA: for (int lane = 0; lane < kOutputs; ++lane) {
#pragma HLS UNROLL
                        const int ch = mb * 8 + lane;
                        bs[lane] = ch < m ? bias_cache[lane % 8][mb + lane / 8] : 0;
                        mq[lane] = ch < m ? multiplier_cache[lane % 8][mb + lane / 8] : 0;
                        sh[lane] = ch < m ? shift_cache[lane % 8][mb + lane / 8] : 0;
                        if (sh[lane] < -31 || sh[lane] > 30) return 2;
                    }
                    if (!cache_all_weights) {
                        CACHE_WEIGHTS: for (int r = 0; r < k * kOCBlocks; ++r) {
#pragma HLS PIPELINE II=1
                            wcache[r] = mb * k + r < group_weight_words
                                ? weights[(uint32_t)(group * group_weight_words + mb * k + r)] : ap_uint<64>(0);
                        }
                    }
                    int cache_base = 0;
                    SPATIAL_BLOCKS_IN_TILE: for (int tb = 0; tb < tile_blocks; ++tb) {
                        const int nb = tile_base + tb;
                        for (int oc = 0; oc < kOutputs; ++oc) {
#pragma HLS UNROLL
                            for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                                acc[oc][p] = bs[oc];
                            }
                        }
                        STREAM_MAC: for (int r = 0; r < k; ++r) {
#pragma HLS PIPELINE II=1
                            ap_uint<64> pv = patch_cache[cache_base + r];
                            ap_uint<64> wv[kOCBlocks];
#pragma HLS ARRAY_PARTITION variable=wv complete
                            for (int bank = 0; bank < kOCBlocks; ++bank) {
#pragma HLS UNROLL
                                wv[bank] = mb + bank < mblocks
                                    ? wcache[(cache_all_weights ? mb * k : 0) + bank * k + r]
                                    : ap_uint<64>(0);
                            }
                            for (int oc = 0; oc < kOutputs; ++oc) {
#pragma HLS UNROLL
                                ap_int<8> w = wv[oc / 8].range((oc % 8) * 8 + 7, (oc % 8) * 8);
                                for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                                    ap_uint<8> byte = pv.range(p * 8 + 7, p * 8);
                                    ap_int<9> x = (ap_int<9>)byte - (ap_int<9>)iz;
                                    ap_int<17> product = x * w;
#pragma HLS BIND_OP variable=product op=mul impl=dsp
                                    acc[oc][p] += (int32_t)product;
                                }
                            }
                        }
                        // Compute the first AXI word address once and increment by
                        // one channel plane after each output, avoiding a wide
                        // per-channel address multiply on the writeback path.
                        uint32_t output_address = channel_block_output_base +
                            (uint32_t)tb;
                        WRITE_CHANNELS: for (int oc = 0; oc < kOutputs; ++oc) {
#pragma HLS PIPELINE II=1
                            ap_uint<64> word = 0;
                            WRITE_POSITIONS: for (int p = 0; p < 8; ++p) {
#pragma HLS UNROLL
                                const uint8_t raw_value = (tile_x_base + tb) * 8 + p < output_w
                                    ? requant(acc[oc][p], mq[oc], sh[oc], oz, activation, act_max)
                                    : (uint8_t)oz;
                                const uint8_t value = activation == 3
                                    ? fused_lut[raw_value] : raw_value;
                                word.range(p * 8 + 7, p * 8) = value;
                            }
                            if (mb * 8 + oc < m)
                                output[output_address] = word;
                            output_address += (uint32_t)nblocks;
                        }
                        cache_base += k;
                    }
                    channel_block_output_base += (uint32_t)kOutputs * (uint32_t)nblocks;
                }
            }
        }

        group_output_base += output_group_stride;
    }
    return 0;
}


