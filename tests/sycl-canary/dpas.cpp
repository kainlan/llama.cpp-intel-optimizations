// Canary 3: DPAS (joint_matrix_mad) determinism on Xe2 (Arc B580)
//
// Loads A[8][16] and B[16][16] from device global memory (NO SLM) and runs
// joint_matrix_mad in isolation across N_SUBGROUPS work-items, each writing to
// its own output tile.  FNV-64 hash of all output tiles must match across 3
// in-run passes.  Exit 0 if all 3 match, 1 otherwise.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -std=c++17 -O2 -Wall -Wextra -o dpas dpas.cpp
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./dpas

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

// Xe2 canonical tile: M=8, K=16, N=16 for fp16->fp32
static constexpr int XMX_M = 8;
static constexpr int XMX_K = 16;
static constexpr int XMX_N = 16;

static constexpr int SG_SIZE     = 16;
static constexpr int SG_PER_WG   = 8;
static constexpr int N_SUBGROUPS = 10000;
static constexpr int N_WGS       = (N_SUBGROUPS + SG_PER_WG - 1) / SG_PER_WG; // 1250

// FNV-64 hash over a float buffer
static uint64_t fnv64(const float * data, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(data);
    size_t n_bytes = count * sizeof(float);
    for (size_t i = 0; i < n_bytes; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t run_pass(
        sycl::queue & q,
        const sycl::half * d_A,    // [XMX_M][XMX_K]
        const sycl::half * d_B,    // [XMX_K][XMX_N]
        float * d_dst,             // [N_SUBGROUPS][XMX_M][XMX_N]
        std::vector<float> & h_dst) {

    constexpr size_t dst_size = (size_t)N_SUBGROUPS * XMX_M * XMX_N;

    // Zero dst before each pass so stale values don't mask non-writes
    q.memset(d_dst, 0, dst_size * sizeof(float)).wait();

    sycl::nd_range<1> range(sycl::range<1>(N_WGS * SG_PER_WG * SG_SIZE),
                            sycl::range<1>(SG_PER_WG * SG_SIZE));

    q.submit([&](sycl::handler & h) {
        h.parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = item.get_sub_group();

            // Global sub-group index
            int wg_id  = static_cast<int>(item.get_group(0));
            int sg_off = static_cast<int>(sg.get_group_id()[0]);
            int sg_gid = wg_id * SG_PER_WG + sg_off;

            if (sg_gid >= N_SUBGROUPS) {
                return;
            }

            // Declare joint_matrix tiles
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

            // Load A and B directly from device global memory — no SLM
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

            // DPAS: the instruction under test
            sycl_xmx::joint_matrix_mad(sg, mat_C, mat_A, mat_B, mat_C);

            // Store to each sub-group's private output slot
            float * out = d_dst + (size_t)sg_gid * XMX_M * XMX_N;
            sycl_xmx::joint_matrix_store(
                sg, mat_C,
                sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(out),
                XMX_N, sycl_xmx::layout::row_major);
        });
    });
    q.wait();

    q.memcpy(h_dst.data(), d_dst, dst_size * sizeof(float)).wait();
    return fnv64(h_dst.data(), dst_size);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});

        auto dev = q.get_device();
        std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

        // Build deterministic input matrices on the host
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

        // Upload inputs to device (fixed for all passes)
        sycl::half * d_A = sycl::malloc_device<sycl::half>(XMX_M * XMX_K, q);
        sycl::half * d_B = sycl::malloc_device<sycl::half>(XMX_K * XMX_N, q);
        q.memcpy(d_A, h_A.data(), h_A.size() * sizeof(sycl::half)).wait();
        q.memcpy(d_B, h_B.data(), h_B.size() * sizeof(sycl::half)).wait();

        constexpr size_t dst_size = (size_t)N_SUBGROUPS * XMX_M * XMX_N;
        float * d_dst = sycl::malloc_device<float>(dst_size, q);
        std::vector<float> h_dst(dst_size);

        uint64_t hashes[3];
        for (int pass = 0; pass < 3; pass++) {
            hashes[pass] = run_pass(q, d_A, d_B, d_dst, h_dst);
            std::printf("Pass %d hash: %016llx\n", pass, (unsigned long long)hashes[pass]);
        }

        // Print first tile row 0 for sanity check (h_dst already populated from last pass)
        std::printf("\nOutput tile [sg_gid=0] row 0:\n  ");
        for (int j = 0; j < XMX_N; j++) {
            std::printf("%.4f ", h_dst[j]);
        }
        std::printf("\n");

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_dst, q);

        bool all_match = (hashes[0] == hashes[1]) && (hashes[1] == hashes[2]);
        std::printf("\nIn-run determinism: %s\n", all_match ? "PASS" : "FAIL");
        return all_match ? 0 : 1;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
