// Standalone CPU/source oracle for the unadvertised Q1_0/NVFP4 AoS MMVQ slice.
// Build: c++ -std=c++17 -O2 test-q1-nvfp4-direct-mmvq.cpp -o /tmp/test-q1-nvfp4-direct-mmvq
// Run from the repository root: /tmp/test-q1-nvfp4-direct-mmvq
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
constexpr int                    QK1  = 128;
constexpr int                    QKNV = 64;
constexpr int                    Q8   = 32;
constexpr std::array<int8_t, 16> fp4  = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

struct Q1Block {
    float                   d;
    std::array<uint8_t, 16> qs;
};

struct NVBlock {
    std::array<uint8_t, 4>  d;
    std::array<uint8_t, 32> qs;
};

struct Q8Block {
    float                  d;
    std::array<int8_t, 32> qs;
};

float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7f || x == 0xff) {
        return 0.0f;
    }
    const int e = (x >> 3) & 15;
    const int m = x & 7;
    return 0.5f * std::ldexp(e == 0 ? float(m) : 1.0f + float(m) / 8.0f, e == 0 ? -9 : e - 7);
}

float q1_oracle(const Q1Block * x, const Q8Block * y, int blocks) {
    float sum = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int k = 0; k < QK1; ++k) {
            const int v = (x[b].qs[k / 8] >> (k % 8)) & 1 ? 1 : -1;
            sum += x[b].d * y[4 * b + k / Q8].d * v * y[4 * b + k / Q8].qs[k % Q8];
        }
    }
    return sum;
}

// Mirrors QI1_0=4,VDR=1: lane classes select one 32-bit bit word and
// therefore one corresponding Q8_1 block. Summing iqs=0..3 proves coverage.
float q1_kernel_indexing(const Q1Block * x, const Q8Block * y, int blocks) {
    float sum = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int iqs = 0; iqs < 4; ++iqs) {
            for (int j = 0; j < 32; ++j) {
                const int k = 32 * iqs + j;
                const int v = (x[b].qs[k / 8] >> (k % 8)) & 1 ? 1 : -1;
                sum += x[b].d * y[4 * b + iqs].d * v * y[4 * b + iqs].qs[j];
            }
        }
    }
    return sum;
}

float nv_oracle(const NVBlock * x, const Q8Block * y, int blocks) {
    float sum = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int sub = 0; sub < 4; ++sub) {
            const Q8Block & q8    = y[2 * b + sub / 2];
            const int       q8off = (sub % 2) * 16;
            for (int j = 0; j < 8; ++j) {
                const uint8_t q = x[b].qs[sub * 8 + j];
                sum +=
                    ue4m3(x[b].d[sub]) * q8.d * (fp4[q & 15] * q8.qs[q8off + j] + fp4[q >> 4] * q8.qs[q8off + j + 8]);
            }
        }
    }
    return sum;
}

// Mirrors VDR=4, QI=8. The two lane classes start at iqs 0 and 4;
// each invocation handles iqs {base,base+1} as sub-blocks {base/2,...}.
float nv_kernel_indexing(const NVBlock * x, const Q8Block * y, int blocks) {
    float sum = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int base : { 0, 4 }) {
            for (int pair = 0; pair < 2; ++pair) {
                const int       iqs0  = base + 2 * pair;
                const int       sub   = iqs0 / 2;
                const Q8Block & q8    = y[2 * b + sub / 2];
                const int       q8off = (sub & 1) * 16;
                for (int word = 0; word < 2; ++word) {
                    const uint32_t packed =
                        uint32_t(x[b].qs[4 * (iqs0 + word) + 0]) | uint32_t(x[b].qs[4 * (iqs0 + word) + 1]) << 8 |
                        uint32_t(x[b].qs[4 * (iqs0 + word) + 2]) << 16 | uint32_t(x[b].qs[4 * (iqs0 + word) + 3]) << 24;
                    for (int byte = 0; byte < 4; ++byte) {
                        const uint8_t q = (packed >> (8 * byte)) & 0xff;
                        const int     j = 4 * word + byte;
                        sum += ue4m3(x[b].d[sub]) * q8.d *
                               (fp4[q & 15] * q8.qs[q8off + j] + fp4[q >> 4] * q8.qs[q8off + j + 8]);
                    }
                }
            }
        }
    }
    return sum;
}

void close(float a, float b) {
    assert(std::isfinite(a) && std::isfinite(b));
    assert(std::fabs(a - b) <= 1e-5f * (1.0f + std::fabs(a)));
}

template <class B> void fill_q8(B & rows) {
    for (size_t r = 0; r < rows.size(); ++r) {
        rows[r].d = 0.125f * float(1 + r % 5);
        for (int k = 0; k < Q8; ++k) {
            rows[r].qs[k] = int8_t(((17 * int(r) + 11 * k) % 31) - 15);
        }
    }
}

void numerical_tests() {
    // Two full blocks exercise the block boundary; selecting row 1 proves the
    // ordinary wrapper's row_low * blocks_per_row pointer adjustment.
    std::vector<Q1Block> q1(3 * 2 * 2);
    for (size_t b = 0; b < q1.size(); ++b) {
        q1[b].d = 0.25f * float(1 + b);
        for (int j = 0; j < 16; ++j) {
            q1[b].qs[j] = uint8_t(0x81u ^ (13u * j + 7u * b));
        }
    }
    std::vector<Q8Block> y1(8);
    fill_q8(y1);
    close(q1_oracle(q1.data() + 2, y1.data(), 2), q1_kernel_indexing(q1.data() + 2, y1.data(), 2));

    std::vector<NVBlock> nv(3 * 2);
    for (size_t b = 0; b < nv.size(); ++b) {
        // Deliberately asymmetric sub-block scales and nibbles.
        nv[b].d = { uint8_t(0x20 + b), uint8_t(0x31 + b), uint8_t(0x42 + b), uint8_t(0x53 + b) };
        for (int sub = 0; sub < 4; ++sub) {
            for (int j = 0; j < 8; ++j) {
                nv[b].qs[sub * 8 + j] = uint8_t(((sub * 5 + j + b) & 15) | (((15 - sub * 3 - j - b) & 15) << 4));
            }
        }
    }
    std::vector<Q8Block> ynv(4);
    fill_q8(ynv);
    close(nv_oracle(nv.data() + 2, ynv.data(), 2), nv_kernel_indexing(nv.data() + 2, ynv.data(), 2));

    // ID order is [id][token]. Repeated expert 2 plus non-monotonic IDs prove
    // pointer-table indexing; distinct Q8 rows prove (id % ne11, token).
    constexpr int                nids = 2, ntokens = 2, rows = 2, blocks = 2;
    const std::array<int32_t, 4> ids = { 2, 0, 2, 1 };
    std::vector<Q8Block>         yid(nids * ntokens * 4);
    fill_q8(yid);
    for (int id = 0; id < nids; ++id) {
        for (int token = 0; token < ntokens; ++token) {
            const int batch  = id * ntokens + token;
            const int expert = ids[batch];
            for (int row = 0; row < rows; ++row) {
                const Q1Block * wx = q1.data() + (expert * rows + row) * blocks;
                const Q8Block * yy = yid.data() + (id * ntokens + token) * 4 * blocks;
                close(q1_oracle(wx, yy, blocks), q1_kernel_indexing(wx, yy, blocks));
            }
        }
    }
}

std::string slurp(const char * path) {
    std::ifstream f(path);
    assert(f && "run test from repository root");
    return { std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>() };
}

void source_contract_tests() {
    const std::string cpp = slurp("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string hpp = slurp("ggml/src/ggml-sycl/mmvq.hpp");
    for (const char * needle :
         { "bool mmvq_submit_q1_nvfp4_aos(", "bool mmvq_submit_q1_nvfp4_aos_id(",
           "case GGML_TYPE_Q1_0:", "case GGML_TYPE_NVFP4:", "vec_dot_q1_0_q8_1>", "vec_dot_nvfp4_q8_1>",
           "weight_layout != GGML_LAYOUT_AOS", "row_low * blocks_per_row", "total_batches != n_ids * n_tokens" }) {
        assert(cpp.find(needle) != std::string::npos);
    }
    assert(hpp.find("bool mmvq_submit_q1_nvfp4_aos(") != std::string::npos);
    assert(hpp.find("bool mmvq_submit_q1_nvfp4_aos_id(") != std::string::npos);
    // This slice must remain local: no route capability or central policy edit.
    const std::string policy = slurp("ggml/src/ggml-sycl/ggml-sycl.cpp");
    assert(policy.find("mmvq_submit_q1_nvfp4_aos") == std::string::npos);
}
}  // namespace

int main() {
    numerical_tests();
    source_contract_tests();
    std::cout << "Q1_0/NVFP4 direct AoS MMVQ CPU/source oracle: PASS\n";
}
