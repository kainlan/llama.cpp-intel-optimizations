// Canary: joint_matrix_apply / fragment element access for fp16 m1n64k32.
//
// Xe2 reports A[1,32] @ B[32,64] -> C[1,64] for fp16 inputs and fp32
// accumulators. This canary proves the ext_intel_packed B layout and the
// per-lane accumulator mapping before the FA decode path relies on K=32.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

static constexpr int XMX_M              = 1;
static constexpr int XMX_K              = 32;
static constexpr int XMX_N              = 64;
static constexpr int SG_SIZE            = 16;
static constexpr int MAX_ELEMS_PER_LANE = XMX_N / SG_SIZE;
static constexpr int TOTAL_ELEMS        = XMX_M * XMX_N;
static constexpr int ACTIVE_LANES       = 8;

static int packed_b_index(int k, int active_col) {
    return (k / 2) * (XMX_N * 2) + active_col * 2 + (k & 1);
}

static bool has_m1n64k32(const sycl::device & dev) {
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }
    const auto combos = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    for (const auto & c : combos) {
        using sycl_xmx::matrix_type;
        if (c.atype == matrix_type::fp16 && c.btype == matrix_type::fp16 && c.ctype == matrix_type::fp32 &&
            c.dtype == matrix_type::fp32 && (int) c.msize == XMX_M && (int) c.ksize == XMX_K &&
            (int) c.nsize == XMX_N) {
            return true;
        }
    }
    return false;
}

static uint32_t ulp_diff(float a, float b) {
    uint32_t ua;
    uint32_t ub;
    std::memcpy(&ua, &a, sizeof(uint32_t));
    std::memcpy(&ub, &b, sizeof(uint32_t));
    return ua > ub ? ua - ub : ub - ua;
}

static void host_matmul(const sycl::half * A, const sycl::half * B_packed, float * C) {
    for (int j = 0; j < XMX_N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < XMX_K; ++k) {
            acc += static_cast<float>(A[k]) * static_cast<float>(B_packed[packed_b_index(k, j)]);
        }
        C[j] = acc;
    }
}

static bool run_case(sycl::queue &              q,
                     const char *               label,
                     int                        load_stride,
                     sycl::half *               d_A,
                     sycl::half *               d_B,
                     const std::vector<float> & h_ref) {
    float * d_apply  = sycl::malloc_device<float>(SG_SIZE * MAX_ELEMS_PER_LANE, q);
    int *   d_counts = sycl::malloc_device<int>(SG_SIZE, q);
    float * d_store  = sycl::malloc_device<float>(TOTAL_ELEMS, q);
    if (!d_apply || !d_counts || !d_store) {
        throw std::runtime_error("device allocation failed");
    }

    q.memset(d_apply, 0, SG_SIZE * MAX_ELEMS_PER_LANE * sizeof(float)).wait();
    q.memset(d_counts, 0, SG_SIZE * sizeof(int)).wait();
    q.memset(d_store, 0, TOTAL_ELEMS * sizeof(float)).wait();

    sycl::nd_range<1> range{ sycl::range<1>(SG_SIZE), sycl::range<1>(SG_SIZE) };
    q.submit([&](sycl::handler & h) {
        h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg   = item.get_sub_group();
            int  lane = static_cast<int>(sg.get_local_id()[0]);

            sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                mat_A;
            sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::ext_intel_packed>
                                                                                                     mat_B;
            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, XMX_M, XMX_N> mat_C;

            sycl_xmx::joint_matrix_fill(sg, mat_C, 0.0f);
            sycl_xmx::joint_matrix_load(
                sg, mat_A,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_A),
                XMX_K);
            sycl_xmx::joint_matrix_load(
                sg, mat_B,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(d_B),
                load_stride);
            sycl_xmx::joint_matrix_mad(sg, mat_C, mat_A, mat_B, mat_C);

            int slot = 0;
            sycl_xmx::joint_matrix_apply(sg, mat_C, [&](float & elem) {
                if (slot < MAX_ELEMS_PER_LANE) {
                    d_apply[lane * MAX_ELEMS_PER_LANE + slot] = elem;
                }
                ++slot;
            });
            d_counts[lane] = slot;

            sycl_xmx::joint_matrix_store(
                sg, mat_C,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                    d_store),
                XMX_N, sycl_xmx::layout::row_major);
        });
    });
    q.wait();

    std::vector<float> h_apply(SG_SIZE * MAX_ELEMS_PER_LANE);
    std::vector<int>   h_counts(SG_SIZE);
    std::vector<float> h_store(TOTAL_ELEMS);
    q.memcpy(h_apply.data(), d_apply, h_apply.size() * sizeof(float)).wait();
    q.memcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int)).wait();
    q.memcpy(h_store.data(), d_store, h_store.size() * sizeof(float)).wait();

    int total_count      = 0;
    int bad_counts       = 0;
    int mapping_mismatch = 0;
    int store_mismatch   = 0;
    for (int lane = 0; lane < SG_SIZE; ++lane) {
        total_count += h_counts[lane];
        if (h_counts[lane] != MAX_ELEMS_PER_LANE) {
            ++bad_counts;
        }
        for (int slot = 0; slot < MAX_ELEMS_PER_LANE; ++slot) {
            const float v = h_apply[lane * MAX_ELEMS_PER_LANE + slot];
            if (lane < ACTIVE_LANES) {
                const int active_col = slot * SG_SIZE + lane;
                if (ulp_diff(v, h_ref[active_col]) > 256 || ulp_diff(h_store[active_col], h_ref[active_col]) > 256) {
                    ++mapping_mismatch;
                }
            } else if (v != 0.0f) {
                ++mapping_mismatch;
            }
        }
    }
    for (int col = 0; col < XMX_N; ++col) {
        if (ulp_diff(h_store[col], h_ref[col]) > 256) {
            ++store_mismatch;
        }
    }

    std::printf("\n=== m1n64k32 apply canary: %s ===\n", label);
    std::printf("  load_stride      : %d\n", load_stride);
    std::printf("  Coverage         : %s (%d/%d)\n", total_count == TOTAL_ELEMS ? "PASS" : "FAIL", total_count,
                TOTAL_ELEMS);
    std::printf("  Per-lane counts  : %s\n", bad_counts == 0 ? "PASS" : "FAIL");
    std::printf("  Apply mapping    : %s (%d mismatches)\n", mapping_mismatch == 0 ? "PASS" : "FAIL", mapping_mismatch);
    std::printf("  Store vs host ref: %s (%d mismatches)\n", store_mismatch == 0 ? "PASS" : "FAIL", store_mismatch);

    sycl::free(d_apply, q);
    sycl::free(d_counts, q);
    sycl::free(d_store, q);

    return total_count == TOTAL_ELEMS && bad_counts == 0 && mapping_mismatch == 0 && store_mismatch == 0;
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        const auto  dev = q.get_device();
        std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        if (!has_m1n64k32(dev)) {
            std::printf("No fp16/fp32 M=1,K=32,N=64 matrix combination; skipping.\n");
            return 0;
        }

        std::vector<sycl::half> h_A(XMX_K);
        std::vector<sycl::half> h_B(XMX_N * XMX_K);
        std::fill(h_B.begin(), h_B.end(), sycl::half(0.0f));
        for (int k = 0; k < XMX_K; ++k) {
            h_A[k] = sycl::half(0.25f + 0.03125f * (float) k);
        }
        for (int logical_col = 0; logical_col < ACTIVE_LANES * MAX_ELEMS_PER_LANE; ++logical_col) {
            const int lane       = logical_col % ACTIVE_LANES;
            const int slot       = logical_col / ACTIVE_LANES;
            const int active_col = slot * SG_SIZE + lane;
            for (int k = 0; k < XMX_K; ++k) {
                h_B[packed_b_index(k, active_col)] = sycl::half(0.125f + 0.0078125f * (float) (logical_col + k));
            }
        }

        std::vector<float> h_ref(TOTAL_ELEMS);
        host_matmul(h_A.data(), h_B.data(), h_ref.data());

        sycl::half * d_A = sycl::malloc_device<sycl::half>(h_A.size(), q);
        sycl::half * d_B = sycl::malloc_device<sycl::half>(h_B.size(), q);
        if (!d_A || !d_B) {
            throw std::runtime_error("device allocation failed");
        }
        q.memcpy(d_A, h_A.data(), h_A.size() * sizeof(sycl::half)).wait();
        q.memcpy(d_B, h_B.data(), h_B.size() * sizeof(sycl::half)).wait();

        const bool packed_stride_ok = run_case(q, "packed-row-stride", XMX_N * 2, d_A, d_B, h_ref);
        const bool k_stride_ok      = run_case(q, "k-stride", XMX_K, d_A, d_B, h_ref);

        sycl::free(d_A, q);
        sycl::free(d_B, q);

        std::printf("\nVerdict: %s\n", packed_stride_ok ? "PASS" : "FAIL");
        std::printf("Note: k-stride comparison is diagnostic only: %s\n", k_stride_ok ? "PASS" : "FAIL");
        return packed_stride_ok ? 0 : 1;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
