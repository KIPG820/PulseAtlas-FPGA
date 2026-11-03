#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

typedef ap_axiu<32, 0, 0, 0> AxisWord;
extern "C" void ecg_nn_accel(hls::stream<AxisWord> &input,
                              ap_uint<32> &status,
                              ap_uint<32> &out0, ap_uint<32> &out1,
                              ap_uint<32> &out2, ap_uint<32> &out3,
                              ap_uint<32> &out4);

static bool read_file(const std::string &path, std::vector<uint8_t> &bytes,
                      size_t expected) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }
    bytes.resize(expected);
    size_t count = fread(bytes.data(), 1, expected, f);
    int extra = fgetc(f);
    fclose(f);
    if (count != expected || extra != EOF) {
        fprintf(stderr, "Wrong vector size: %s\n", path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: ecg_nn_tb <golden_vectors_directory>\n");
        return 2;
    }
    const int indices[4] = {0, 333, 1024, 2000};
    const char *limit_env = getenv("ECG_NN_VECTOR_LIMIT");
    int vector_count = limit_env ? atoi(limit_env) : 4;
    if (vector_count < 1 || vector_count > 4) return 2;
    for (int v = 0; v < vector_count; ++v) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "fold9_%04d", indices[v]);
        std::string base = std::string(argv[1]) + "/" + prefix;
        std::vector<uint8_t> stimulus, expected;
        if (!read_file(base + "_input_time_major_u8.bin", stimulus, 60000) ||
            !read_file(base + "_expected_output_u8.bin", expected, 5)) return 2;

        hls::stream<AxisWord> input;
        for (int w = 0; w < 15000; ++w) {
            AxisWord word;
            word.data = (uint32_t)stimulus[w * 4] |
                        ((uint32_t)stimulus[w * 4 + 1] << 8) |
                        ((uint32_t)stimulus[w * 4 + 2] << 16) |
                        ((uint32_t)stimulus[w * 4 + 3] << 24);
            word.keep = 15;
            word.strb = 15;
            word.last = w == 14999;
            input.write(word);
        }
        ap_uint<32> status, o0, o1, o2, o3, o4;
        ecg_nn_accel(input, status, o0, o1, o2, o3, o4);
        uint8_t got[5] = {(uint8_t)o0, (uint8_t)o1, (uint8_t)o2,
                          (uint8_t)o3, (uint8_t)o4};
        printf("fold9_%04d status=%u codes=[%u,%u,%u,%u,%u] expected=[%u,%u,%u,%u,%u]\n",
               indices[v], (unsigned)status, got[0], got[1], got[2], got[3], got[4],
               expected[0], expected[1], expected[2], expected[3], expected[4]);
        if (status != 0) return 1;
        for (int i = 0; i < 5; ++i) if (got[i] != expected[i]) return 1;
    }
    printf("PASS: %d complete 12-lead window(s) exactly match the integer reference\n",
           vector_count);
    return 0;
}
