// Standalone probe for MMVQ-like reads from raw device allocations versus
// suballocations at offsets inside a large sycl::malloc_device allocation.
// Build manually:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -O3 -std=c++17 -fsycl -fsycl-targets=intel_gpu_bmg_g21,spir64 \
//     -Xs "-cl-intel-greater-than-4GB-buffer-required" \
//     tests/test-mmvq-q4-large-offset.cpp -o /tmp/test-mmvq-q4-large-offset

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

constexpr int QK4_0 = 32;
constexpr int QK8_1 = 32;
constexpr int SG    = 32;
constexpr int TILE  = 32;

static size_t env_mb(const char * name, size_t fallback_mb) {
    const char * v = std::getenv(name);
    if (!v || !*v) {
        return fallback_mb;
    }
    return static_cast<size_t>(std::strtoull(v, nullptr, 10));
}

static bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v && v[0] == '1';
}

template <typename T>
static T * alloc_device(size_t count, sycl::queue & q, bool aligned) {
    const size_t bytes = count * sizeof(T);
    if (aligned) {
        return static_cast<T *>(sycl::aligned_alloc_device(2ull * 1024ull * 1024ull, bytes, q));
    }
    return sycl::malloc_device<T>(count, q);
}

static inline int get_i32(const int8_t * x, int i32) {
    int v;
    std::memcpy(&v, x + sizeof(int) * i32, sizeof(v));
    return v;
}

template <typename T>
static inline sycl::vec<T, 4> extract4(T val) {
    return sycl::vec<T, 1>(val)
        .template as<sycl::vec<std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>, 4>>()
        .template convert<T>();
}

template <typename T1, typename T2, typename T3>
static inline int dp4a_style(T1 a, T2 b, T3 c) {
    int res = c;
    auto va = extract4(a);
    auto vb = extract4(b);
    res += va[0] * vb[0];
    res += va[1] * vb[1];
    res += va[2] * vb[2];
    res += va[3] * vb[3];
    return res;
}

static std::vector<uint8_t> make_x(int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;
    const int x_row_stride   = ncols / 2;
    std::vector<uint8_t> out(nrows * x_row_stride + nrows * blocks_per_row * sizeof(sycl::half));
    uint8_t *            qs = out.data();
    sycl::half *         ds = reinterpret_cast<sycl::half *>(out.data() + nrows * x_row_stride);

    std::mt19937 rng(1);
    std::uniform_int_distribution<int> qdist(0, 255);
    std::uniform_real_distribution<float> ddist(0.001f, 0.05f);
    for (int row = 0; row < nrows; ++row) {
        for (int tile = 0; tile < blocks_per_row / TILE; ++tile) {
            const int tile_base   = row * x_row_stride + tile * TILE * 16;
            const int word_stride = TILE * 4;
            for (int block = 0; block < TILE; ++block) {
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

static std::vector<uint8_t> make_y(int ncols) {
    const int blocks = ncols / QK8_1;
    std::vector<uint8_t> out(ncols + blocks * sizeof(sycl::half2));
    int8_t *             qs = reinterpret_cast<int8_t *>(out.data());
    sycl::half2 *        ds = reinterpret_cast<sycl::half2 *>(out.data() + ncols);

    std::mt19937 rng(2);
    std::uniform_int_distribution<int> qdist(-64, 63);
    std::uniform_real_distribution<float> ddist(0.001f, 0.05f);
    for (int i = 0; i < ncols; ++i) {
        qs[i] = static_cast<int8_t>(qdist(rng));
    }
    for (int b = 0; b < blocks; ++b) {
        ds[b] = sycl::half2(sycl::half(ddist(rng)), sycl::half(ddist(rng)));
    }
    return out;
}

class q4_offset_kernel;

static sycl::event submit_kernel(sycl::queue & q, const uint8_t * x, const uint8_t * y, float * dst, int ncols,
                                 int nrows) {
    constexpr int num_subgroups = 16;
    return q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<q4_offset_kernel>(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nrows * SG), sycl::range<3>(1, 1, num_subgroups * SG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg           = it.get_sub_group();
                const int  row          = it.get_group_linear_id() * sg.get_group_linear_range() +
                                 sg.get_group_linear_id();
                const int lane_id = sg.get_local_linear_id();
                if (row >= nrows) {
                    return;
                }

                const int blocks_per_row = ncols / QK4_0;
                const int tiles_per_row  = blocks_per_row / TILE;
                const int word_stride    = TILE * 4;
                const int x_row_stride   = ncols / 2;

                const sycl::half *  x_d  = reinterpret_cast<const sycl::half *>(x + nrows * x_row_stride);
                const int8_t *      y_qs = reinterpret_cast<const int8_t *>(y);
                const sycl::half2 * y_ds = reinterpret_cast<const sycl::half2 *>(y + ncols);

                float partial = 0.0f;
                for (int tile = 0; tile < tiles_per_row; ++tile) {
                    const int tile_base = row * x_row_stride + tile * TILE * 16;
                    for (int block = lane_id; block < TILE; block += SG) {
                        const int   block_idx = row * blocks_per_row + tile * TILE + block;
                        const float d         = static_cast<float>(x_d[block_idx]);
                        const int   y_block   = tile * TILE + block;
                        const int   y_base    = y_block * QK8_1;
                        for (int half = 0; half < 2; ++half) {
                            const int word_base = half * (2 * word_stride);
                            const int v0        = *reinterpret_cast<const int *>(x + tile_base + word_base + block * 4);
                            const int v1 =
                                *reinterpret_cast<const int *>(x + tile_base + word_base + word_stride + block * 4);
                            const int y_offset = half * 8;
                            const int u0       = get_i32(y_qs + y_base, y_offset / 4);
                            const int u1       = get_i32(y_qs + y_base, y_offset / 4 + 4);
                            const int u2       = get_i32(y_qs + y_base, y_offset / 4 + 1);
                            const int u3       = get_i32(y_qs + y_base, y_offset / 4 + 5);

                            int sumi = 0;
                            sumi     = dp4a_style((v0 >> 0) & 0x0F0F0F0F, u0, sumi);
                            sumi     = dp4a_style((v0 >> 4) & 0x0F0F0F0F, u1, sumi);
                            sumi     = dp4a_style((v1 >> 0) & 0x0F0F0F0F, u2, sumi);
                            sumi     = dp4a_style((v1 >> 4) & 0x0F0F0F0F, u3, sumi);

                            const sycl::float2 ds = y_ds[y_block].convert<float, sycl::rounding_mode::automatic>();
                            partial += d * (static_cast<float>(sumi) * ds.x() - 4.0f * ds.y());
                        }
                    }
                }
                const float sum = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
                if (sg.leader()) {
                    dst[row] = sum;
                }
            });
    });
}

static double bench(sycl::queue & q, uint8_t * dx, uint8_t * dy, float * dd, int ncols, int nrows, int iters) {
    for (int i = 0; i < 10; ++i) {
        submit_kernel(q, dx, dy, dd, ncols, nrows).wait();
    }
    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto ev = submit_kernel(q, dx, dy, dd, ncols, nrows);
        ev.wait();
        const uint64_t start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t end   = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        total += static_cast<double>(end - start) * 1e-6;
    }
    return total / iters;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
    const int ncols = 4096;
    const int nrows = 4096;
    const int iters = 100;
    const size_t large_mb = env_mb("MMVQ_OFFSET_LARGE_MB", 4096);
    const bool use_aligned_alloc = env_flag("MMVQ_OFFSET_ALIGNED_ALLOC");
    const std::vector<size_t> offsets_mb = { 0, 16, 256, 1024, 2048, large_mb > 512 ? large_mb - 512 : 0 };

    auto hx = make_x(ncols, nrows);
    auto hy = make_y(ncols);

    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    printf("x_bytes=%zu large_mb=%zu aligned_alloc=%d\n", hx.size(), large_mb, use_aligned_alloc ? 1 : 0);

    uint8_t * raw_x = alloc_device<uint8_t>(hx.size(), q, use_aligned_alloc);
    uint8_t * raw_y = alloc_device<uint8_t>(hy.size(), q, use_aligned_alloc);
    float * raw_d = alloc_device<float>(nrows, q, use_aligned_alloc);
    q.memcpy(raw_x, hx.data(), hx.size()).wait();
    q.memcpy(raw_y, hy.data(), hy.size()).wait();
    printf("raw ptr=%p %.4f ms\n", raw_x, bench(q, raw_x, raw_y, raw_d, ncols, nrows, iters));

    const size_t large_bytes = large_mb * 1024ull * 1024ull;
    uint8_t * arena = alloc_device<uint8_t>(large_bytes, q, use_aligned_alloc);
    printf("large ptr=%p bytes=%zu\n", arena, large_bytes);
    for (size_t off_mb : offsets_mb) {
        const size_t off = off_mb * 1024ull * 1024ull;
        if (!arena || off + hx.size() > large_bytes) {
            continue;
        }
        uint8_t * x = arena + off;
        q.memcpy(x, hx.data(), hx.size()).wait();
        printf("offset_mb=%zu ptr=%p %.4f ms\n", off_mb, x, bench(q, x, raw_y, raw_d, ncols, nrows, iters));
    }
    const size_t packed_off = (large_mb > 512 ? large_mb - 512 : 0) * 1024ull * 1024ull;
    const size_t y_off      = (packed_off + hx.size() + 4095) & ~size_t(4095);
    const size_t d_off      = (y_off + hy.size() + 4095) & ~size_t(4095);
    if (arena && d_off + nrows * sizeof(float) <= large_bytes) {
        uint8_t * x = arena + packed_off;
        uint8_t * y = arena + y_off;
        float *   d = reinterpret_cast<float *>(arena + d_off);
        q.memcpy(x, hx.data(), hx.size()).wait();
        q.memcpy(y, hy.data(), hy.size()).wait();
        printf("packed_tail x=%p y=%p d=%p x_off_mb=%.1f %.4f ms\n", x, y, d,
               packed_off / (1024.0 * 1024.0), bench(q, x, y, d, ncols, nrows, iters));
    }

    sycl::free(raw_x, q);
    sycl::free(raw_y, q);
    sycl::free(raw_d, q);
    if (arena) {
        sycl::free(arena, q);
    }
    return 0;
}
