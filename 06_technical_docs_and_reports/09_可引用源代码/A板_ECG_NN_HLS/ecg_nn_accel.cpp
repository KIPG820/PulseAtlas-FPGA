#include "ecg_nn_params.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

typedef ap_axiu<32, 0, 0, 0> AxisWord;

static uint32_t raw[15000];
static uint8_t act0[kActivationCapacity];
static uint8_t act1[kActivationCapacity];
static uint8_t act2[kActivationCapacity];
static uint8_t act3[kActivationCapacity];
static uint8_t pooled[64];

// The same signed Q31 ties-to-even rule is used by integer_reference.py.
static int32_t round_q31(int64_t value) {
#pragma HLS INLINE
    bool negative = value < 0;
    uint64_t magnitude = negative ? (uint64_t)(-value) : (uint64_t)value;
    uint64_t whole = magnitude >> 31;
    uint32_t remainder = (uint32_t)(magnitude & 0x7fffffffULL);
    bool up = remainder > 0x40000000U ||
              (remainder == 0x40000000U && (whole & 1U));
    int64_t rounded = (int64_t)(whole + (up ? 1 : 0));
    return (int32_t)(negative ? -rounded : rounded);
}

static uint8_t clip_code(int32_t code, int zero_point, bool relu) {
#pragma HLS INLINE
    if (relu && code < zero_point) code = zero_point;
    if (code < 0) return 0;
    if (code > 255) return 255;
    return (uint8_t)code;
}

static int8_t weight_at(int index) {
#pragma HLS INLINE
    int word = index >> 2;
    uint32_t packed;
    if (word < 16384) packed = kPackedWeights0[word];
    else if (word < 32768) packed = kPackedWeights1[word - 16384];
    else packed = kPackedWeights2[word - 32768];
    return (int8_t)(packed >> ((index & 3) * 8));
}

static uint8_t raw_code(int channel, int time) {
#pragma HLS INLINE
    uint32_t packed = raw[time * 3 + (channel >> 2)];
    return (uint8_t)(packed >> ((channel & 3) * 8));
}

static void stem_layer(uint8_t *destination) {
    const LayerDesc d = kLayers[0];
    for (int oc = 0; oc < 16; ++oc) {
        for (int t = 0; t < 1250; t += 4) {
            int32_t accumulator[4];
#pragma HLS ARRAY_PARTITION variable=accumulator complete
            for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                accumulator[lane] = kBiases[d.parameter_offset + oc];
            }
            for (int ic = 0; ic < 12; ++ic) {
                for (int tap = 0; tap < 15; ++tap) {
#pragma HLS PIPELINE II=1
                    int x0 = t * 4 + tap - 7;
                    int8_t weight = weight_at(d.weight_offset +
                        (oc * 12 + ic) * 15 + tap);
                    for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                        int x = x0 + lane * 4;
                        if (x >= 0 && x < 5000)
                            accumulator[lane] += ((int16_t)raw_code(ic, x) - 64) * weight;
                    }
                }
            }
            for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                if (t + lane < 1250) {
                    int32_t code = round_q31((int64_t)accumulator[lane] *
                        kMultipliersQ31[d.parameter_offset + oc]);
                    destination[oc * 1250 + t + lane] = clip_code(code, 0, true);
                }
            }
        }
    }
}

static void conv_layer(int layer_index, const uint8_t *source, uint8_t *destination) {
    const LayerDesc d = kLayers[layer_index];
    for (int oc = 0; oc < d.out_channels; ++oc) {
        for (int t = 0; t < d.output_length; t += 4) {
            int32_t accumulator[4];
#pragma HLS ARRAY_PARTITION variable=accumulator complete
            for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                accumulator[lane] = kBiases[d.parameter_offset + oc];
            }
            for (int ic = 0; ic < d.in_channels; ++ic) {
                for (int tap = 0; tap < d.kernel; ++tap) {
#pragma HLS PIPELINE II=1
                    int x0 = t * d.stride + tap - d.padding;
                    int8_t weight = weight_at(d.weight_offset +
                        (oc * d.in_channels + ic) * d.kernel + tap);
                    for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                        int x = x0 + lane * d.stride;
                        if (t + lane < d.output_length && x >= 0 && x < d.input_length)
                            accumulator[lane] +=
                                ((int16_t)source[ic * d.input_length + x] -
                                 d.input_zero_point) * weight;
                    }
                }
            }
            for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                if (t + lane < d.output_length) {
                    int32_t code = round_q31((int64_t)accumulator[lane] *
                        kMultipliersQ31[d.parameter_offset + oc]) + d.output_zero_point;
                    destination[oc * d.output_length + t + lane] =
                        clip_code(code, d.output_zero_point, d.fused_relu != 0);
                }
            }
        }
    }
}

static void add_relu(int add_index, const uint8_t *a, const uint8_t *b,
                     uint8_t *destination) {
    const AddDesc d = kAdds[add_index];
    const int total = d.channels * d.length;
    for (int i = 0; i < total; ++i) {
#pragma HLS PIPELINE II=1
        int64_t numerator = ((int32_t)a[i] - d.a_zero_point) * d.a_multiplier_q31;
        numerator += ((int32_t)b[i] - d.b_zero_point) * d.b_multiplier_q31;
        int32_t code = round_q31(numerator) + d.output_zero_point;
        destination[i] = clip_code(code, d.output_zero_point, true);
    }
}

static uint8_t rounded_average(int32_t sum, int divisor) {
#pragma HLS INLINE
    int32_t whole = sum / divisor;
    int32_t remainder = sum % divisor;
    if (2 * remainder > divisor ||
        (2 * remainder == divisor && (whole & 1))) ++whole;
    return (uint8_t)whole;
}

static void average_pool(const uint8_t *source) {
    for (int c = 0; c < 64; ++c) {
        int32_t sum = 0;
        for (int t = 0; t < 313; ++t) {
#pragma HLS PIPELINE II=1
            sum += source[c * 313 + t];
        }
        pooled[c] = rounded_average(sum, 313);
    }
}

static void classifier(uint8_t result[5]) {
    const LayerDesc d = kLayers[15];
    for (int oc = 0; oc < 5; ++oc) {
        int32_t accumulator = kBiases[d.parameter_offset + oc];
        for (int ic = 0; ic < 64; ++ic) {
#pragma HLS PIPELINE II=1
            int16_t centered = (int16_t)pooled[ic] - d.input_zero_point;
            accumulator += centered * weight_at(d.weight_offset + oc * 64 + ic);
        }
        result[oc] = clip_code(
            round_q31((int64_t)accumulator * kMultipliersQ31[d.parameter_offset + oc]) +
            d.output_zero_point, d.output_zero_point, false);
    }
}

extern "C" void ecg_nn_accel(hls::stream<AxisWord> &input,
                              ap_uint<32> &status,
                              ap_uint<32> &out0, ap_uint<32> &out1,
                              ap_uint<32> &out2, ap_uint<32> &out3,
                              ap_uint<32> &out4) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE s_axilite port=status bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out2 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out3 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out4 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS BIND_STORAGE variable=raw type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=raw cyclic factor=8 dim=1
#pragma HLS BIND_STORAGE variable=act0 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=act0 cyclic factor=8 dim=1
#pragma HLS BIND_STORAGE variable=act1 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=act1 cyclic factor=8 dim=1
#pragma HLS BIND_STORAGE variable=act2 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=act2 cyclic factor=8 dim=1
#pragma HLS BIND_STORAGE variable=act3 type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=act3 cyclic factor=8 dim=1
    ap_uint<32> error = 0;
    for (int t = 0; t < 5000; ++t) {
        for (int block = 0; block < 3; ++block) {
#pragma HLS PIPELINE II=1
            AxisWord word = input.read();
            if (word.keep != 15) error |= 1;
            if ((bool)word.last != (t == 4999 && block == 2)) error |= 2;
            raw[t * 3 + block] = (uint32_t)word.data;
        }
    }
    if (error != 0) {
        status = error;
        out0 = out1 = out2 = out3 = out4 = 0;
        return;
    }

    stem_layer(act0);
    conv_layer(1, act0, act1);
    conv_layer(2, act1, act2);
    add_relu(0, act2, act0, act1);
    conv_layer(3, act1, act0);
    conv_layer(4, act0, act2);
    add_relu(1, act2, act1, act0);

    conv_layer(5, act0, act1);
    conv_layer(6, act0, act2);
    conv_layer(7, act2, act3);
    add_relu(2, act3, act1, act0);
    conv_layer(8, act0, act1);
    conv_layer(9, act1, act2);
    add_relu(3, act2, act0, act1);

    conv_layer(10, act1, act0);
    conv_layer(11, act1, act2);
    conv_layer(12, act2, act3);
    add_relu(4, act3, act0, act1);
    conv_layer(13, act1, act0);
    conv_layer(14, act0, act2);
    add_relu(5, act2, act1, act0);

    average_pool(act0);
    uint8_t result[5];
    classifier(result);
    out0 = result[0]; out1 = result[1]; out2 = result[2];
    out3 = result[3]; out4 = result[4];
    status = 0;
}
