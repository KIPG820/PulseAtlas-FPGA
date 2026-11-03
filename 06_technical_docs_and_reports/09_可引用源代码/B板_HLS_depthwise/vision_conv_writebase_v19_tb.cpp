#include <ap_int.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>

extern "C" int32_t vision_conv_writebase_v19(
    const ap_uint<64>*, const ap_uint<64>*, const int32_t*, const int32_t*,
    const int32_t*, ap_uint<64>*, const int32_t*, const ap_uint<64>*);

static int8_t signed_byte(ap_uint<64> word, int lane) {
    return (int8_t)(uint8_t)word.range(lane * 8 + 7, lane * 8);
}

static uint8_t unsigned_byte(ap_uint<64> word, int lane) {
    return (uint8_t)word.range(lane * 8 + 7, lane * 8);
}

static ap_uint<64> input_words[1 << 17];
static ap_uint<64> weights[1 << 20];
static ap_uint<64> output[1 << 17];
static ap_uint<64> output_lut[32];
static int32_t bias[512], multipliers[512], shifts[512], descriptor[23];

static uint8_t get_input(int channel, int y, int x, int height, int width) {
    const uint32_t byte_index = ((uint32_t)channel * height + y) * width + x;
    return unsigned_byte(input_words[byte_index >> 3], byte_index & 7);
}

static bool run_case(int groups, int m, int cpg, int ih, int iw,
                     int kh, int kw, int sh, int sw,
                     int dh, int dw, int pt, int pl, int pb, int pr,
                     bool fused_silu = false) {
    const int oh = (ih + pt + pb - dh * (kh - 1) - 1) / sh + 1;
    const int ow = (iw + pl + pr - dw * (kw - 1) - 1) / sw + 1;
    const int n = oh * ow, k = cpg * kh * kw;
    const int mb = (m + 7) / 8;
    const int blocks_per_row = (ow + 7) / 8;
    const int nb = oh * blocks_per_row;
    const int total_channels = groups * cpg;
    const int input_size = total_channels * ih * iw;
    if (input_size > (1 << 20) || groups * m > 512 || k > 4096) return false;
    std::fill(input_words, input_words + (input_size + 7) / 8, ap_uint<64>(0));
    std::fill(weights, weights + groups * mb * k, ap_uint<64>(0));
    std::fill(output, output + groups * m * nb, ap_uint<64>(0));
    std::fill(bias, bias + groups * m, 0);
    std::fill(multipliers, multipliers + groups * m, 1 << 30);
    std::fill(shifts, shifts + groups * m, 1);

    for (int c = 0; c < total_channels; ++c) {
        for (int y = 0; y < ih; ++y) {
            for (int x = 0; x < iw; ++x) {
                const uint32_t index = ((uint32_t)c * ih + y) * iw + x;
                const uint8_t value = (uint8_t)(128 + ((c * 11 + y * 5 + x * 3) % 15) - 7);
                const int lane = index & 7;
                input_words[index >> 3].range(lane * 8 + 7, lane * 8) = value;
            }
        }
    }

    descriptor[0] = m; descriptor[1] = n; descriptor[2] = k;
    descriptor[3] = groups; descriptor[4] = 128; descriptor[5] = 23;
    descriptor[6] = fused_silu ? 3 : 0; descriptor[7] = 255;
    descriptor[8] = ih; descriptor[9] = iw; descriptor[10] = cpg;
    descriptor[11] = oh; descriptor[12] = ow; descriptor[13] = kh;
    descriptor[14] = kw; descriptor[15] = sh; descriptor[16] = sw;
    descriptor[17] = dh; descriptor[18] = dw; descriptor[19] = pt;
    descriptor[20] = pl; descriptor[21] = pb; descriptor[22] = pr;

    for (int g = 0; g < groups; ++g) {
        for (int block = 0; block < mb; ++block) {
            for (int r = 0; r < k; ++r) {
                ap_uint<64> word = 0;
                for (int lane = 0; lane < 8; ++lane) {
                    const int ch = block * 8 + lane;
                    const int8_t value = ch < m
                        ? (int8_t)(((g * 3 + ch * 2 + r) % 7) - 3) : 0;
                    word.range(lane * 8 + 7, lane * 8) = (uint8_t)value;
                }
                weights[g * mb * k + block * k + r] = word;
            }
        }
        for (int ch = 0; ch < m; ++ch) bias[g * m + ch] = (g - ch) * 3;
    }

    std::fill(output_lut, output_lut + 32, ap_uint<64>(0));
    for (int value = 0; value < 256; ++value) {
        const uint8_t mapped = (uint8_t)((value * 13 + 17) & 255);
        const int word = value / 8, lane = value % 8;
        output_lut[word].range(lane * 8 + 7, lane * 8) = mapped;
    }

    const int status = vision_conv_writebase_v19(input_words, weights, bias,
        multipliers, shifts, output, descriptor, output_lut);
    if (status) {
        printf("kernel status %d for G=%d M=%d Cpg=%d IH=%d IW=%d KH=%d KW=%d\n",
               status, groups, m, cpg, ih, iw, kh, kw);
        return false;
    }

    const int kernel_area = kh * kw;
    for (int g = 0; g < groups; ++g) {
        for (int ch = 0; ch < m; ++ch) {
            for (int pos = 0; pos < n; ++pos) {
                const int oy = pos / ow, ox = pos % ow;
                int32_t acc = bias[g * m + ch];
                for (int r = 0; r < k; ++r) {
                    const int ic = r / kernel_area;
                    const int kernel_offset = r % kernel_area;
                    const int ky = kernel_offset / kw;
                    const int kx = kernel_offset % kw;
                    const int iy = oy * sh + ky * dh - pt;
                    const int ix = ox * sw + kx * dw - pl;
                    const uint8_t qx = (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                        ? get_input(g * cpg + ic, iy, ix, ih, iw) : 128;
                    const int32_t w = signed_byte(
                        weights[g * mb * k + (ch / 8) * k + r], ch % 8);
                    acc += ((int32_t)qx - 128) * w;
                }
                int32_t expected = acc + 23;
                if (expected < 0) expected = 0;
                if (expected > 255) expected = 255;
                if (fused_silu)
                    expected = (uint8_t)((expected * 13 + 17) & 255);
                const int output_word = (pos / ow) * blocks_per_row +
                                        (pos % ow) / 8;
                const int output_lane = (pos % ow) % 8;
                const int index = (g * m + ch) * nb + output_word;
                const uint8_t actual = unsigned_byte(output[index], output_lane);
                if (actual != expected) {
                    printf("mismatch G=%d M=%d Cpg=%d pos=%d ch=%d got=%u want=%d\n",
                           groups, m, cpg, pos, ch, actual, expected);
                    return false;
                }
            }
        }
    }
    return true;
}


// Independent old-kernel rounding reference. Exercise every valid shift plus
// signed accumulator/multiplier extremes; no assumptions about calibration.
static bool run_rounding_case(int seed) {
    // Co-simulation dumps the declared AXI depths, including unused tails.
    // Reuse the full-depth buffers rather than tiny stack arrays.
    ap_uint<64>* x=input_words; ap_uint<64>* w=weights;
    ap_uint<64>* out=output; ap_uint<64>* lut=output_lut;
    int32_t* b=bias; int32_t* mul=multipliers; int32_t* sh=shifts;
    x[0]=0x8080808080808080ULL;
    std::fill(w,w+8,ap_uint<64>(0));
    std::fill(out,out+64,ap_uint<64>(0));
    std::fill(lut,lut+32,ap_uint<64>(0));
    const int32_t values[] = {INT32_MIN, INT32_MAX, -65537, -3, -1, 0, 1,
                              3, 65537, 1073741824};
    int32_t d[23] = {64,8,1,1,128,23,3,255,1,8,1,1,8,1,1,1,1,1,1,0,0,0,0};
    for (int i = 0; i < 256; ++i)
        lut[i/8].range((i%8)*8+7,(i%8)*8) = (uint8_t)((13*i+17)&255);
    for (int ch=0; ch<64; ++ch) {
        b[ch] = values[(ch+seed)%10];
        mul[ch] = values[(ch*3+seed*7)%10];
        sh[ch] = (ch+seed)%62-31;
    }
    if (vision_conv_writebase_v19(x,w,b,mul,sh,out,d,lut) != 0) return false;
    for (int ch=0; ch<64; ++ch) {
        int64_t v = (int64_t)b[ch]*mul[ch];
        bool neg = v < 0;
        uint64_t mag = neg ? (uint64_t)(-v) : (uint64_t)v;
        int shift = 31-sh[ch];
        uint64_t whole = mag>>shift, rem=mag&((1ULL<<shift)-1);
        uint64_t half=1ULL<<(shift-1);
        int64_t r=whole+(rem>half || (rem==half && (whole&1)));
        int64_t expect=(int32_t)(neg ? -r : r);
        expect+=23;
        if (expect<0) expect=0;
        if (expect>255) expect=255;
        expect=(13*expect+17)&255;
        for (int pos=0; pos<8; ++pos) {
            if (unsigned_byte(out[ch],pos) != expect) {
                printf("ROUND_FAIL seed=%d ch=%d shift=%d\n",seed,ch,sh[ch]);
                return false;
            }
        }
    }
    // Invalid metadata must fail before executing either channel block.
    sh[9]=31;
    if (vision_conv_writebase_v19(x,w,b,mul,sh,out,d,lut) != 2) return false;
    sh[9]=-32;
    return vision_conv_writebase_v19(x,w,b,mul,sh,out,d,lut) == 2;
}

int main() {
    bool ok =
        run_case(1, 3, 3, 5, 7, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, true) &&
        run_case(4, 1, 1, 4, 6, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1) &&
        run_case(1, 17, 5, 5, 9, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0) &&
        run_case(2, 9, 2, 6, 5, 3, 3, 2, 2, 1, 1, 0, 1, 1, 2, true) &&
        run_case(1, 64, 64, 48, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1);
    // Pointwise fast-path and fallback checks, including groups, partial tiles,
    // odd rows, padding and stride. Independent scalar reference checks bytes.
    ok = ok && run_case(1, 64, 48, 48, 64, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, true);
    ok = ok && run_case(2, 9, 5, 5, 40, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, true);
    ok = ok && run_case(1, 17, 5, 3, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    ok = ok && run_case(1, 9, 3, 5, 13, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, true);
    ok = ok && run_case(1, 8, 4, 5, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    ok = ok && run_case(1, 8, 4, 5, 16, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0);
    // New aligned 3x3 same-padding path: left/right/top/bottom halos,
    // multiple spatial tiles, fused LUT, grouped and narrow cases.
    ok = ok && run_case(1, 16, 8, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true);
    ok = ok && run_case(2, 9, 3, 7, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    ok = ok && run_case(1, 8, 4, 5, 15, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    // Exercise the dedicated depthwise branch, including all four image
    // boundaries, several channel planes, and the fused activation LUT.
    ok = ok && run_case(8, 1, 1, 8, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
    ok = ok && run_case(32, 1, 1, 12, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, true);
    for (int seed=0; seed<10 && ok; ++seed) ok = run_rounding_case(seed);
    if (!ok) return 1;
    puts("VISION_CONV_ADDRESSINC_V18_SIM_PASS");
    return 0;
}


