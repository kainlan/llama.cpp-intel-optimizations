// Canary 5: joint_matrix_apply / fragment element access on Xe2 (Arc B580)
//
// Tests whether joint_matrix_apply exposes per-lane register-resident fragment
// elements after joint_matrix_mad on oneAPI 2025.3 + Xe2, without SLM.
//
// Tile: m8n16k16, fp16 A/B, fp32 C/D.  SG_SIZE=16.  Each lane owns 8 elements.
//
// Phase A: host reference  A[8][16] @ B[16][16] -> C_ref[8][16] (float)
// Phase B: joint_matrix_apply — gather per-lane elements, check coverage + values
// Phase C: joint_matrix_store to global — cross-check Phase B values
// Phase D: lane layout report — which (row,col) each lane owns
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   cd tests/sycl-canary && ./build.sh joint-matrix-apply
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./joint-matrix-apply

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

static constexpr int XMX_M  = 8;
static constexpr int XMX_K  = 16;
static constexpr int XMX_N  = 16;
static constexpr int SG_SIZE = 16;

// Each lane in a sub-group of 16 owns (8*16)/16 = 8 elements of the m8n16k16 accumulator.
static constexpr int MAX_ELEMS_PER_LANE = 8;
static constexpr int TOTAL_ELEMS        = XMX_M * XMX_N; // 128

// One sub-group for a clean, isolated experiment.
static constexpr int SG_PER_WG = 1;

// Host reference: float C_ref[8][16] = A[8][16] @ B[16][16]
static void host_matmul(const sycl::half * A, const sycl::half * B, float * C) {
    for (int i = 0; i < XMX_M; i++) {
        for (int j = 0; j < XMX_N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < XMX_K; k++) {
                acc += static_cast<float>(A[i * XMX_K + k]) *
                       static_cast<float>(B[k * XMX_N + j]);
            }
            C[i * XMX_N + j] = acc;
        }
    }
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});

        auto dev = q.get_device();
        std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

        // --- Build inputs (same formula as canary 3) ---
        std::vector<sycl::half> h_A(XMX_M * XMX_K);
        std::vector<sycl::half> h_B(XMX_K * XMX_N);
        for (int i = 0; i < XMX_M; i++) {
            for (int j = 0; j < XMX_K; j++) {
                h_A[i * XMX_K + j] = sycl::half(std::sin(i * 13.0f + j * 7.0f) * 0.5f);
            }
        }
        for (int i = 0; i < XMX_K; i++) {
            for (int j = 0; j < XMX_N; j++) {
                h_B[i * XMX_N + j] = sycl::half(std::cos(i * 11.0f + j * 5.0f) * 0.5f);
            }
        }

        // Phase A: host reference
        std::vector<float> h_C_ref(TOTAL_ELEMS);
        host_matmul(h_A.data(), h_B.data(), h_C_ref.data());

        std::printf("\nPhase A — host reference C_ref row 0:\n  ");
        for (int j = 0; j < XMX_N; j++) {
            std::printf("%.4f ", h_C_ref[j]);
        }
        std::printf("\n");

        // Upload inputs
        sycl::half * d_A = sycl::malloc_device<sycl::half>(XMX_M * XMX_K, q);
        sycl::half * d_B = sycl::malloc_device<sycl::half>(XMX_K * XMX_N, q);
        q.memcpy(d_A, h_A.data(), h_A.size() * sizeof(sycl::half)).wait();
        q.memcpy(d_B, h_B.data(), h_B.size() * sizeof(sycl::half)).wait();

        // Phase B output buffers
        // dst_apply[lane][elem_slot] — per-lane gathered elements
        float * d_apply  = sycl::malloc_device<float>(SG_SIZE * MAX_ELEMS_PER_LANE, q);
        int   * d_counts = sycl::malloc_device<int>(SG_SIZE, q);

        // Phase C output buffer: full 8×16 store from joint_matrix_store
        float * d_store = sycl::malloc_device<float>(TOTAL_ELEMS, q);

        q.memset(d_apply,  0, SG_SIZE * MAX_ELEMS_PER_LANE * sizeof(float)).wait();
        q.memset(d_counts, 0, SG_SIZE * sizeof(int)).wait();
        q.memset(d_store,  0, TOTAL_ELEMS * sizeof(float)).wait();

        // ---------------------------------------------------------------
        // Phase B kernel: joint_matrix_mad + joint_matrix_apply
        // ---------------------------------------------------------------
        sycl::nd_range<1> range(sycl::range<1>(SG_PER_WG * SG_SIZE),
                                sycl::range<1>(SG_PER_WG * SG_SIZE));

        q.submit([&](sycl::handler & h) {
            h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = item.get_sub_group();
                int lane = static_cast<int>(sg.get_local_id()[0]);

                sycl_xmx::joint_matrix<
                    sycl::sub_group, sycl::half,
                    sycl_xmx::use::a, XMX_M, XMX_K,
                    sycl_xmx::layout::row_major> mat_A;

                sycl_xmx::joint_matrix<
                    sycl::sub_group, sycl::half,
                    sycl_xmx::use::b, XMX_K, XMX_N,
                    sycl_xmx::layout::row_major> mat_B;

                sycl_xmx::joint_matrix<
                    sycl::sub_group, float,
                    sycl_xmx::use::accumulator, XMX_M, XMX_N> mat_C;

                sycl_xmx::joint_matrix_fill(sg, mat_C, 0.0f);

                sycl_xmx::joint_matrix_load(
                    sg, mat_A,
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(d_A),
                    XMX_K);

                sycl_xmx::joint_matrix_load(
                    sg, mat_B,
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(d_B),
                    XMX_N);

                sycl_xmx::joint_matrix_mad(sg, mat_C, mat_A, mat_B, mat_C);

                // Per-lane slot counter via atomic_ref — confirms the pattern Phase 4 will use.
                // memory_scope::sub_group: no cross-lane ordering needed, relaxed suffices.
                sycl::atomic_ref<int,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::sub_group,
                    sycl::access::address_space::global_space> cnt_ref(d_counts[lane]);

                sycl_xmx::joint_matrix_apply(sg, mat_C, [&](float & elem) {
                    int slot = cnt_ref.fetch_add(1);
                    d_apply[lane * MAX_ELEMS_PER_LANE + slot] = elem;
                });
            });
        });
        q.wait();

        // ---------------------------------------------------------------
        // Phase C kernel: joint_matrix_store to global
        // ---------------------------------------------------------------
        q.submit([&](sycl::handler & h) {
            h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = item.get_sub_group();

                sycl_xmx::joint_matrix<
                    sycl::sub_group, sycl::half,
                    sycl_xmx::use::a, XMX_M, XMX_K,
                    sycl_xmx::layout::row_major> mat_A;

                sycl_xmx::joint_matrix<
                    sycl::sub_group, sycl::half,
                    sycl_xmx::use::b, XMX_K, XMX_N,
                    sycl_xmx::layout::row_major> mat_B;

                sycl_xmx::joint_matrix<
                    sycl::sub_group, float,
                    sycl_xmx::use::accumulator, XMX_M, XMX_N> mat_C;

                sycl_xmx::joint_matrix_fill(sg, mat_C, 0.0f);

                sycl_xmx::joint_matrix_load(
                    sg, mat_A,
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(d_A),
                    XMX_K);

                sycl_xmx::joint_matrix_load(
                    sg, mat_B,
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(d_B),
                    XMX_N);

                sycl_xmx::joint_matrix_mad(sg, mat_C, mat_A, mat_B, mat_C);

                sycl_xmx::joint_matrix_store(
                    sg, mat_C,
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::no>(d_store),
                    XMX_N, sycl_xmx::layout::row_major);
            });
        });
        q.wait();

        // Copy results to host
        std::vector<float> h_apply(SG_SIZE * MAX_ELEMS_PER_LANE);
        std::vector<int>   h_counts(SG_SIZE);
        std::vector<float> h_store(TOTAL_ELEMS);

        q.memcpy(h_apply.data(),  d_apply,  h_apply.size()  * sizeof(float)).wait();
        q.memcpy(h_counts.data(), d_counts, h_counts.size() * sizeof(int)).wait();
        q.memcpy(h_store.data(),  d_store,  h_store.size()  * sizeof(float)).wait();

        // ---------------------------------------------------------------
        // Validation
        // ---------------------------------------------------------------
        std::printf("\nPhase C — store row 0:\n  ");
        for (int j = 0; j < XMX_N; j++) {
            std::printf("%.4f ", h_store[j]);
        }
        std::printf("\n");

        // Check (b): total element count == 128
        int total_count = 0;
        for (int l = 0; l < SG_SIZE; l++) {
            total_count += h_counts[l];
        }
        std::printf("\nPhase B coverage: total elements = %d (expected %d)\n",
                    total_count, TOTAL_ELEMS);

        // Check (a): no empty lanes
        int empty_lanes = 0;
        for (int l = 0; l < SG_SIZE; l++) {
            if (h_counts[l] == 0) {
                empty_lanes++;
            }
        }
        std::printf("Empty lanes: %d (expected 0)\n", empty_lanes);

        // Per-lane element counts
        std::printf("Per-lane element counts:\n  ");
        for (int l = 0; l < SG_SIZE; l++) {
            std::printf("lane%d=%d ", l, h_counts[l]);
        }
        std::printf("\n");

        // Check (c): Phase B values match Phase C store values (same set)
        // Build sorted list of Phase B values and compare to sorted Phase C values
        std::vector<float> apply_vals;
        apply_vals.reserve(TOTAL_ELEMS);
        for (int l = 0; l < SG_SIZE; l++) {
            for (int s = 0; s < h_counts[l]; s++) {
                apply_vals.push_back(h_apply[l * MAX_ELEMS_PER_LANE + s]);
            }
        }
        std::vector<float> store_vals(h_store.begin(), h_store.end());
        std::sort(apply_vals.begin(), apply_vals.end());
        std::sort(store_vals.begin(), store_vals.end());

        int value_mismatch = 0;
        if (apply_vals.size() == store_vals.size()) {
            for (size_t i = 0; i < apply_vals.size(); i++) {
                // Allow 1 ULP tolerance for fp16 round-trip
                uint32_t a, b;
                std::memcpy(&a, &apply_vals[i], 4);
                std::memcpy(&b, &store_vals[i], 4);
                uint32_t diff = (a > b) ? (a - b) : (b - a);
                if (diff > 1) {
                    value_mismatch++;
                }
            }
        } else {
            value_mismatch = -1;
        }
        std::printf("Phase B vs Phase C value match (sorted): %s\n",
                    value_mismatch == 0 ? "PASS" : "FAIL");

        // Diagnostic only (not in pass criteria): Phase C vs host f32 reference.
        // DPAS fp16 MAD rounds fp16 inputs before accumulation; host f32 doesn't.
        // ULP differences of 4-32 are expected and benign.
        int ref_mismatch = 0;
        for (int i = 0; i < TOTAL_ELEMS; i++) {
            uint32_t a, b;
            std::memcpy(&a, &h_store[i], 4);
            std::memcpy(&b, &h_C_ref[i], 4);
            uint32_t diff = (a > b) ? (a - b) : (b - a);
            if (diff > 2) {
                ref_mismatch++;
            }
        }
        std::printf("Phase C vs host ref (diagnostic): %d/%d fp16/f32 rounding diffs (expected, not a failure)\n",
                    ref_mismatch, TOTAL_ELEMS);

        // ---------------------------------------------------------------
        // Phase D: lane layout report
        // ---------------------------------------------------------------
        // For each lane and each slot, find the (row, col) in C_ref that
        // matches the apply value — this reveals the DPAS register layout.
        std::printf("\nPhase D — Xe2 DPAS m8n16k16 accumulator lane layout:\n");
        std::printf("  Each cell shows (row,col) in the 8x16 output tile owned by that lane.\n");
        std::printf("  Format: lane -> [slot0:(r,c) slot1:(r,c) ...]\n\n");

        for (int l = 0; l < SG_SIZE; l++) {
            std::printf("  lane %2d: ", l);
            for (int s = 0; s < h_counts[l]; s++) {
                float v = h_apply[l * MAX_ELEMS_PER_LANE + s];
                // Find matching (row,col) in h_store (which matches C_ref)
                int best_r = -1, best_c = -1;
                uint32_t best_diff = 0xFFFFFFFF;
                for (int r = 0; r < XMX_M; r++) {
                    for (int c = 0; c < XMX_N; c++) {
                        float sv = h_store[r * XMX_N + c];
                        uint32_t a, b;
                        std::memcpy(&a, &v, 4);
                        std::memcpy(&b, &sv, 4);
                        uint32_t d = (a > b) ? (a - b) : (b - a);
                        if (d < best_diff) {
                            best_diff = d;
                            best_r    = r;
                            best_c    = c;
                        }
                    }
                }
                std::printf("(%d,%d) ", best_r, best_c);
            }
            std::printf("\n");
        }

        // ---------------------------------------------------------------
        // Final verdict (spec criteria only: (a) no empty, (b) coverage, (c) apply==store)
        // ---------------------------------------------------------------
        bool coverage_ok = (total_count == TOTAL_ELEMS);
        bool no_empty    = (empty_lanes == 0);
        bool values_ok   = (value_mismatch == 0);
        bool pass        = coverage_ok && no_empty && values_ok;

        std::printf("\n=== Canary 5 result ===\n");
        std::printf("  Coverage 128 elems : %s\n", coverage_ok ? "PASS" : "FAIL");
        std::printf("  No empty lanes     : %s\n", no_empty    ? "PASS" : "FAIL");
        std::printf("  Apply == Store vals: %s\n", values_ok   ? "PASS" : "FAIL");
        std::printf("Verdict: %s\n", pass ? "PASS" : "FAIL");

        sycl::free(d_A,      q);
        sycl::free(d_B,      q);
        sycl::free(d_apply,  q);
        sycl::free(d_counts, q);
        sycl::free(d_store,  q);

        return pass ? 0 : 1;

    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
