// Standalone geometry benchmark for Q4_0 coalesced MMVQ-style kernels.
// Build manually:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -O3 -std=c++17 -fsycl -fsycl-targets=intel_gpu_bmg_g21,spir64 \
//     -Xs "-cl-intel-greater-than-4GB-buffer-required" \
//     tests/test-mmvq-q4-coalesced-geometry.cpp -o /tmp/test-mmvq-q4-coalesced-geometry

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

constexpr int QK4_0 = 32;
constexpr int QK8_1 = 32;

struct block_q4_0 {
    sycl::half d;
    uint8_t    qs[QK4_0 / 2];
};

struct block_q8_1 {
    sycl::half d;
    sycl::half s;
    int8_t     qs[QK8_1];
};

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static inline int get_i32(const int8_t * x, int i32) {
    int v;
    std::memcpy(&v, x + sizeof(int) * i32, sizeof(v));
    return v;
}

static inline int dp4a_i8(int a, int b) {
    int sum = 0;
    for (int i = 0; i < 4; ++i) {
        const int av = (a >> (8 * i)) & 0xff;
        const int bv = static_cast<int8_t>((b >> (8 * i)) & 0xff);
        sum += av * bv;
    }
    return sum;
}

template <typename T>
static inline sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
    return sycl::vec<T, 1>(val)
        .template as<sycl::vec<std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>, 4>>()
        .template convert<T>();
}

template <typename T1, typename T2, typename T3>
static inline auto dpct_style_dp4a(T1 a, T2 b, T3 c) {
    int res = c;
    auto va = extract_and_sign_or_zero_extend4(a);
    auto vb = extract_and_sign_or_zero_extend4(b);
    res += va[0] * vb[0];
    res += va[1] * vb[1];
    res += va[2] * vb[2];
    res += va[3] * vb[3];
    return res;
}

static std::vector<uint8_t> make_q4_0_coalesced(int ncols, int nrows, int tile_blocks) {
    const int blocks_per_row = ncols / QK4_0;
    const int tiles_per_row  = blocks_per_row / tile_blocks;
    const int x_row_stride   = ncols / 2;

    std::vector<uint8_t> out(nrows * x_row_stride + nrows * blocks_per_row * sizeof(sycl::half));
    uint8_t *            qs = out.data();
    sycl::half *         ds = reinterpret_cast<sycl::half *>(out.data() + nrows * x_row_stride);

    std::mt19937 rng(1);
    std::uniform_int_distribution<int> qdist(0, 255);
    std::uniform_real_distribution<float> ddist(0.001f, 0.05f);

    for (int row = 0; row < nrows; ++row) {
        for (int tile = 0; tile < tiles_per_row; ++tile) {
            const int tile_base   = row * x_row_stride + tile * tile_blocks * 16;
            const int word_stride = tile_blocks * 4;
            for (int block = 0; block < tile_blocks; ++block) {
                for (int word = 0; word < 4; ++word) {
                    const int base = tile_base + word * word_stride + block * 4;
                    qs[base + 0]   = static_cast<uint8_t>(qdist(rng));
                    qs[base + 1]   = static_cast<uint8_t>(qdist(rng));
                    qs[base + 2]   = static_cast<uint8_t>(qdist(rng));
                    qs[base + 3]   = static_cast<uint8_t>(qdist(rng));
                }
            }
        }
        for (int block = 0; block < blocks_per_row; ++block) {
            ds[row * blocks_per_row + block] = sycl::half(ddist(rng));
        }
    }
    return out;
}

static std::vector<uint8_t> make_q8_1_reordered(int ncols) {
    const int blocks = ncols / QK8_1;
    std::vector<uint8_t> out(ncols + blocks * sizeof(sycl::half) * 2);
    int8_t *             qs = reinterpret_cast<int8_t *>(out.data());
    sycl::half *         ds = reinterpret_cast<sycl::half *>(out.data() + ncols);

    std::mt19937 rng(2);
    std::uniform_int_distribution<int> qdist(-64, 63);
    std::uniform_real_distribution<float> ddist(0.001f, 0.05f);
    for (int i = 0; i < ncols; ++i) {
        qs[i] = static_cast<int8_t>(qdist(rng));
    }
    for (int b = 0; b < blocks; ++b) {
        ds[2 * b + 0] = sycl::half(ddist(rng));
        ds[2 * b + 1] = sycl::half(ddist(rng));
    }
    return out;
}

template <int SG, int TILE_BLOCKS, bool USE_DPCT_DP4A, bool USE_HALF2_DS>
static sycl::event submit_kernel(sycl::queue & q, const uint8_t * x, const uint8_t * y, float * dst, int ncols,
                                 int nrows) {
    constexpr int num_subgroups = 16;
    const int     block_num_y   = nrows;
    const auto    global        = sycl::range<3>(1, 1, block_num_y * SG);
    const auto    local         = sycl::range<3>(1, 1, num_subgroups * SG);

    return q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it)
                                                                 [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg           = it.get_sub_group();
            const int  sg_range     = sg.get_group_linear_range();
            const int  workgroup_id = it.get_group_linear_id();
            const int  sg_id        = sg.get_group_linear_id();
            const int  lane_id      = sg.get_local_linear_id();
            const int  row          = workgroup_id * sg_range + sg_id;
            if (row >= nrows) {
                return;
            }

            const int blocks_per_row = ncols / QK4_0;
            const int tiles_per_row  = blocks_per_row / TILE_BLOCKS;
            const int word_stride    = TILE_BLOCKS * 4;
            const int x_row_stride   = ncols / 2;

            const uint8_t * x_qs = x;
            const sycl::half * x_d =
                reinterpret_cast<const sycl::half *>(reinterpret_cast<const char *>(x) + nrows * x_row_stride);
            const int8_t * y_qs = reinterpret_cast<const int8_t *>(y);
            const sycl::half * y_ds = reinterpret_cast<const sycl::half *>(reinterpret_cast<const char *>(y) + ncols);
            const sycl::half2 * y_ds2 =
                reinterpret_cast<const sycl::half2 *>(reinterpret_cast<const char *>(y) + ncols);

            float partial_sum = 0.0f;
            for (int tile = 0; tile < tiles_per_row; ++tile) {
                const int tile_base = row * x_row_stride + tile * TILE_BLOCKS * 16;
                for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += SG) {
                    const int   block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                    const float d         = static_cast<float>(x_d[block_idx]);
                    const int   y_block   = tile * TILE_BLOCKS + block_in_tile;
                    const int   y_base    = y_block * QK8_1;

                    for (int half = 0; half < 2; ++half) {
                        const int word_base    = half * (2 * word_stride);
                        const int word0_offset = word_base + block_in_tile * 4;
                        const int word1_offset = word_base + word_stride + block_in_tile * 4;
                        int       v0;
                        int       v1;
                        std::memcpy(&v0, x_qs + tile_base + word0_offset, sizeof(v0));
                        std::memcpy(&v1, x_qs + tile_base + word1_offset, sizeof(v1));

                        const int y_offset = half * 8;
                        const int u0       = get_i32(y_qs + y_base, y_offset / 4);
                        const int u1       = get_i32(y_qs + y_base, y_offset / 4 + 4);
                        const int u2       = get_i32(y_qs + y_base, y_offset / 4 + 1);
                        const int u3       = get_i32(y_qs + y_base, y_offset / 4 + 5);

                        const int vi0_0 = (v0 >> 0) & 0x0F0F0F0F;
                        const int vi1_0 = (v0 >> 4) & 0x0F0F0F0F;
                        const int vi0_1 = (v1 >> 0) & 0x0F0F0F0F;
                        const int vi1_1 = (v1 >> 4) & 0x0F0F0F0F;

                        int sumi = 0;
                        if constexpr (USE_DPCT_DP4A) {
                            sumi = dpct_style_dp4a(vi0_0, u0, sumi);
                            sumi = dpct_style_dp4a(vi1_0, u1, sumi);
                            sumi = dpct_style_dp4a(vi0_1, u2, sumi);
                            sumi = dpct_style_dp4a(vi1_1, u3, sumi);
                        } else {
                            sumi += dp4a_i8(vi0_0, u0);
                            sumi += dp4a_i8(vi1_0, u1);
                            sumi += dp4a_i8(vi0_1, u2);
                            sumi += dp4a_i8(vi1_1, u3);
                        }

                        float ds0;
                        float ds1;
                        if constexpr (USE_HALF2_DS) {
                            const sycl::float2 ds8f = y_ds2[y_block].convert<float, sycl::rounding_mode::automatic>();
                            ds0                     = ds8f.x();
                            ds1                     = ds8f.y();
                        } else {
                            ds0 = static_cast<float>(y_ds[2 * y_block + 0]);
                            ds1 = static_cast<float>(y_ds[2 * y_block + 1]);
                        }
                        partial_sum += d * (static_cast<float>(sumi) * ds0 - 4.0f * ds1);
                    }
                }
            }

            const float sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());
            if (sg.leader()) {
                dst[row] = sum;
            }
        });
    });
}

template <int SG, int TILE_BLOCKS, bool USE_DPCT_DP4A = false, bool USE_HALF2_DS = false>
static double bench_variant(sycl::queue & q, int ncols, int nrows, int iters) {
    auto hx = make_q4_0_coalesced(ncols, nrows, TILE_BLOCKS);
    auto hy = make_q8_1_reordered(ncols);

    uint8_t * dx = sycl::malloc_device<uint8_t>(hx.size(), q);
    uint8_t * dy = sycl::malloc_device<uint8_t>(hy.size(), q);
    float *   dd = sycl::malloc_device<float>(nrows, q);
    q.memcpy(dx, hx.data(), hx.size()).wait();
    q.memcpy(dy, hy.data(), hy.size()).wait();

    for (int i = 0; i < 10; ++i) {
        submit_kernel<SG, TILE_BLOCKS, USE_DPCT_DP4A, USE_HALF2_DS>(q, dx, dy, dd, ncols, nrows).wait();
    }

    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto ev = submit_kernel<SG, TILE_BLOCKS, USE_DPCT_DP4A, USE_HALF2_DS>(q, dx, dy, dd, ncols, nrows);
        ev.wait();
        const uint64_t start = ev.template get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t end   = ev.template get_profiling_info<sycl::info::event_profiling::command_end>();
        total_ms += static_cast<double>(end - start) * 1e-6;
    }

    sycl::free(dx, q);
    sycl::free(dy, q);
    sycl::free(dd, q);
    return total_ms / iters;
}

template <int SG, int TILE_BLOCKS, bool USE_DPCT_DP4A = false, bool USE_HALF2_DS = false>
static double bench_variant_wall(sycl::queue & q, int ncols, int nrows, int iters) {
    auto hx = make_q4_0_coalesced(ncols, nrows, TILE_BLOCKS);
    auto hy = make_q8_1_reordered(ncols);

    uint8_t * dx = sycl::malloc_device<uint8_t>(hx.size(), q);
    uint8_t * dy = sycl::malloc_device<uint8_t>(hy.size(), q);
    float *   dd = sycl::malloc_device<float>(nrows, q);
    q.memcpy(dx, hx.data(), hx.size()).wait();
    q.memcpy(dy, hy.data(), hy.size()).wait();

    for (int i = 0; i < 10; ++i) {
        submit_kernel<SG, TILE_BLOCKS, USE_DPCT_DP4A, USE_HALF2_DS>(q, dx, dy, dd, ncols, nrows).wait();
    }

    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        submit_kernel<SG, TILE_BLOCKS, USE_DPCT_DP4A, USE_HALF2_DS>(q, dx, dy, dd, ncols, nrows).wait();
        const auto t1 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    sycl::free(dx, q);
    sycl::free(dy, q);
    sycl::free(dd, q);
    return total_ms / iters;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
    auto dev = q.get_device();
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("subgroups:");
    for (auto sg : dev.get_info<sycl::info::device::sub_group_sizes>()) {
        printf(" %zu", sg);
    }
    printf("\n");

    const int ncols = 4096;
    const int iters = 100;
    for (int nrows : { 4096, 11008, 32000 }) {
        printf("shape nrows=%d ncols=%d\n", nrows, ncols);
        printf("  sg16_tile16 %.4f ms\n", bench_variant<16, 16>(q, ncols, nrows, iters));
        printf("  sg16_tile32 %.4f ms\n", bench_variant<16, 32>(q, ncols, nrows, iters));
        printf("  sg32_tile16 %.4f ms\n", bench_variant<32, 16>(q, ncols, nrows, iters));
        printf("  sg32_tile32 %.4f ms\n", bench_variant<32, 32>(q, ncols, nrows, iters));
        printf("  sg32_tile32_dpct %.4f ms\n", bench_variant<32, 32, true, false>(q, ncols, nrows, iters));
        printf("  sg32_tile32_dpct_half2 %.4f ms\n", bench_variant<32, 32, true, true>(q, ncols, nrows, iters));
        printf("  sg32_tile32_dpct_half2_wall %.4f ms\n",
               bench_variant_wall<32, 32, true, true>(q, ncols, nrows, iters));
    }
    return 0;
}
