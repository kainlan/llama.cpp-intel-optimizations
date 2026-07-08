// Canary: joint_matrix_apply / fragment element access for fp16 m1n64k16.
//
// This is the decode-shaped matrix combination exposed by Xe2 devices:
// A[1,16] @ B[16,64] -> C[1,64], SG_SIZE=16.  Xe2 exposes four accumulator
// slots per lane; with ext_intel_packed B, lanes 0..7 carry useful columns
// and lanes 8..15 are zero for this M=1 shape.  The FA decode prototype uses
// two packed M=1/N=64 MADs to cover a compact 64-token KV tile.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

static constexpr int XMX_M              = 1;
static constexpr int XMX_K              = 16;
static constexpr int XMX_N              = 64;
static constexpr int SG_SIZE            = 16;
static constexpr int MAX_ELEMS_PER_LANE = XMX_N / SG_SIZE;
static constexpr int TOTAL_ELEMS        = XMX_M * XMX_N;
static constexpr int ACTIVE_LANES       = 8;
static constexpr int ACTIVE_ELEMS       = ACTIVE_LANES * MAX_ELEMS_PER_LANE;

static int packed_b_index(int k, int active_col) {
    return (k / 2) * (XMX_N * 2) + active_col * 2 + (k & 1);
}

static void host_matmul(const sycl::half * A, const sycl::half * B_packed, float * C) {
    for (int j = 0; j < XMX_N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < XMX_K; ++k) {
            const int packed_idx = packed_b_index(k, j);
            acc += static_cast<float>(A[k]) * static_cast<float>(B_packed[packed_idx]);
        }
        C[j] = acc;
    }
}

static uint32_t ulp_diff(float a, float b) {
    uint32_t ua;
    uint32_t ub;
    std::memcpy(&ua, &a, sizeof(uint32_t));
    std::memcpy(&ub, &b, sizeof(uint32_t));
    return ua > ub ? ua - ub : ub - ua;
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        const auto  dev = q.get_device();
        std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        if (!dev.has(sycl::aspect::ext_intel_matrix)) {
            std::printf("No ext_intel_matrix support; skipping.\n");
            return 0;
        }

        std::vector<sycl::half> h_A(XMX_M * XMX_K);
        std::vector<sycl::half> h_B(XMX_N * XMX_K);
        for (int k = 0; k < XMX_K; ++k) {
            h_A[k] = sycl::half(k == 0 ? 1.0f : 0.0f);
        }
        std::fill(h_B.begin(), h_B.end(), sycl::half(0.0f));
        for (int logical_col = 0; logical_col < ACTIVE_ELEMS; ++logical_col) {
            const int lane                     = logical_col % ACTIVE_LANES;
            const int slot                     = logical_col / ACTIVE_LANES;
            const int active_col               = slot * SG_SIZE + lane;
            h_B[packed_b_index(0, active_col)] = sycl::half((float) (logical_col + 1));
        }

        std::vector<float> h_ref(TOTAL_ELEMS);
        host_matmul(h_A.data(), h_B.data(), h_ref.data());

        sycl::half * d_A      = sycl::malloc_device<sycl::half>(h_A.size(), q);
        sycl::half * d_B      = sycl::malloc_device<sycl::half>(h_B.size(), q);
        float *      d_apply  = sycl::malloc_device<float>(SG_SIZE * MAX_ELEMS_PER_LANE, q);
        int *        d_counts = sycl::malloc_device<int>(SG_SIZE, q);
        float *      d_store  = sycl::malloc_device<float>(TOTAL_ELEMS, q);
        if (!d_A || !d_B || !d_apply || !d_counts || !d_store) {
            throw std::runtime_error("device allocation failed");
        }

        q.memcpy(d_A, h_A.data(), h_A.size() * sizeof(sycl::half)).wait();
        q.memcpy(d_B, h_B.data(), h_B.size() * sizeof(sycl::half)).wait();
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
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        d_A),
                    XMX_K);
                sycl_xmx::joint_matrix_load(
                    sg, mat_B,
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        d_B),
                    XMX_K);
                sycl_xmx::joint_matrix_mad(sg, mat_C, mat_A, mat_B, mat_C);

                int slot = 0;
                sycl_xmx::joint_matrix_apply(sg, mat_C, [&](float & elem) {
                    if (slot < MAX_ELEMS_PER_LANE) {
                        d_apply[lane * MAX_ELEMS_PER_LANE + slot] = elem;
                    }
                    ++slot;
                });
                d_counts[lane] = slot;
            });
        });
        q.wait();

        q.submit([&](sycl::handler & h) {
            h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();

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
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        d_A),
                    XMX_K);
                sycl_xmx::joint_matrix_load(
                    sg, mat_B,
                    sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        d_B),
                    XMX_K);
                sycl_xmx::joint_matrix_mad(sg, mat_C, mat_A, mat_B, mat_C);
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

        int total_count = 0;
        int bad_counts  = 0;
        for (int lane = 0; lane < SG_SIZE; ++lane) {
            total_count += h_counts[lane];
            if (h_counts[lane] != MAX_ELEMS_PER_LANE) {
                ++bad_counts;
            }
        }

        std::printf("Per-lane element counts:\n  ");
        for (int lane = 0; lane < SG_SIZE; ++lane) {
            std::printf("lane%d=%d ", lane, h_counts[lane]);
        }
        std::printf("\n");

        std::printf("Store row (active packed columns only):\n  ");
        for (int col = 0; col < XMX_N; ++col) {
            std::printf("%.0f%s", h_store[col], (col + 1) % 16 == 0 ? "\n  " : " ");
        }
        std::printf("\n");

        std::printf("Lane mapping:\n");
        int mapping_mismatch = 0;
        for (int lane = 0; lane < SG_SIZE; ++lane) {
            std::printf("  lane %2d:", lane);
            for (int slot = 0; slot < MAX_ELEMS_PER_LANE; ++slot) {
                const float v = h_apply[lane * MAX_ELEMS_PER_LANE + slot];
                if (lane < ACTIVE_LANES) {
                    const int   expected_logical = slot * ACTIVE_LANES + lane;
                    const int   active_col       = slot * SG_SIZE + lane;
                    const float expected_value   = (float) (expected_logical + 1);
                    if (ulp_diff(v, expected_value) > 1 || ulp_diff(h_store[active_col], expected_value) > 1) {
                        ++mapping_mismatch;
                    }
                    std::printf(" slot%d->compact%d/store_col%d", slot, expected_logical, active_col);
                } else {
                    if (v != 0.0f) {
                        ++mapping_mismatch;
                    }
                    std::printf(" slot%d->zero", slot);
                }
            }
            std::printf("\n");
        }

        int ref_mismatch = 0;
        for (int col = 0; col < XMX_N; ++col) {
            if (ulp_diff(h_store[col], h_ref[col]) > 256) {
                ++ref_mismatch;
            }
        }

        std::vector<float> apply_vals;
        apply_vals.reserve(TOTAL_ELEMS);
        for (int lane = 0; lane < SG_SIZE; ++lane) {
            for (int slot = 0; slot < std::min(h_counts[lane], MAX_ELEMS_PER_LANE); ++slot) {
                apply_vals.push_back(h_apply[lane * MAX_ELEMS_PER_LANE + slot]);
            }
        }
        std::vector<float> store_vals = h_store;
        std::sort(apply_vals.begin(), apply_vals.end());
        std::sort(store_vals.begin(), store_vals.end());

        int value_mismatch = apply_vals.size() == store_vals.size() ? 0 : -1;
        if (value_mismatch == 0) {
            for (size_t i = 0; i < apply_vals.size(); ++i) {
                if (ulp_diff(apply_vals[i], store_vals[i]) > 1) {
                    ++value_mismatch;
                }
            }
        }

        const bool pass = total_count == TOTAL_ELEMS && bad_counts == 0 && mapping_mismatch == 0 && value_mismatch == 0;
        std::printf("\n=== m1n64k16 apply canary ===\n");
        std::printf("  Coverage          : %s (%d/%d)\n", total_count == TOTAL_ELEMS ? "PASS" : "FAIL", total_count,
                    TOTAL_ELEMS);
        std::printf("  Per-lane counts   : %s\n", bad_counts == 0 ? "PASS" : "FAIL");
        std::printf("  Apply mapping     : %s\n", mapping_mismatch == 0 ? "PASS" : "FAIL");
        std::printf("  Apply == Store    : %s\n", value_mismatch == 0 ? "PASS" : "FAIL");
        std::printf("  Store vs host ref : %d/%d fp16/f32 rounding diffs > 256 ULP\n", ref_mismatch, TOTAL_ELEMS);
        std::printf("Verdict: %s\n", pass ? "PASS" : "FAIL");

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_apply, q);
        sycl::free(d_counts, q);
        sycl::free(d_store, q);
        return pass ? 0 : 1;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
