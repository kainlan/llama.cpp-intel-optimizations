// WOQ repack kernel microbench (perf-recovery epic, track D, llama.cpp-0vqt):
// times, via SYCL DEVICE EVENTS (never host chrono -- see repo memory
// host-chrono-cannot-see-past-submission-backpressure), the four forms of
// each of the two repack kernel families -- per-slot / batched, original /
// coalesced (SLM-tiled) -- at GPT-OSS 20B's real expert shape
// (blocks_per_row=90, nrows=2880, confirmed identical for gate/up/down --
// hidden_size==intermediate_size==2880 for this model) with 32 active
// slots (GPT-OSS 20B's real max).
//
// NOT a ctest: this is a benchmark, not a correctness gate (the byte-
// identity oracles live in test-sycl-mxfp4-woq-repack.cpp and
// test-sycl-mxfp4-woq-tiled-repack.cpp). Build explicitly:
//   ./scripts/sycl-build.sh test-sycl-mxfp4-woq-repack-bench
// Run (source oneAPI first -- see CLAUDE.md "A SYCL test binary run WITHOUT
// sourcing setvars.sh prints SKIP and exits 0"):
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-mxfp4-woq-repack-bench   # B50
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-mxfp4-woq-repack-bench   # B70
//
// Timing method: brackets the work under test between two trivial
// single_task marker kernels submitted on the SAME in-order queue (mirrors
// the production PP-profile instrument's begin/end marker convention,
// ggml-sycl.cpp mxfp4_pp_profile_repack_{begin,end}_marker) and reads
// (end.command_end - begin.command_start) from queue-profiling timestamps
// after a q.wait(). This captures true device execution time for
// everything submitted in between, including a per-slot loop's N separate
// kernel launches, without any host-side stopwatch racing driver
// submission backpressure.
//
// GB/s is reported as (bytes read from src + bytes written to dst) across
// all n_slots, divided by the measured device time -- i.e. total HBM
// traffic for the logical repack, matching how the task's static
// attribution (comment c-ajw4) phrases "effective bandwidth".
#include "ggml-common.h"
#include "ggml-sycl/convert.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Forward-declared at namespace scope (SYCL kernel names must be
// forward-declarable at namespace scope -- see CLAUDE.md's
// mxfp4_pp_profile_repack_*_marker precedent, ggml-sycl.cpp:64655-64656).
struct repack_bench_begin_marker;
struct repack_bench_end_marker;

static sycl::event submit_begin_marker(sycl::queue & q) {
    return q.submit([&](sycl::handler & cgh) { cgh.single_task<repack_bench_begin_marker>([] {}); });
}

static sycl::event submit_end_marker(sycl::queue & q) {
    return q.submit([&](sycl::handler & cgh) { cgh.single_task<repack_bench_end_marker>([] {}); });
}

// Runs `work(q)` bracketed by begin/end markers, waits, and returns the
// device time between them in milliseconds.
template <typename F> static double time_device_ms(sycl::queue & q, F && work) {
    sycl::event begin = submit_begin_marker(q);
    work(q);
    sycl::event end = submit_end_marker(q);
    q.wait();
    const uint64_t t0 = begin.get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t t1 = end.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(t1 - t0) / 1.0e6;  // ns -> ms
}

struct bench_result {
    double min_ms  = 0.0;
    double mean_ms = 0.0;
};

template <typename F> static bench_result run_bench(sycl::queue & q, int warmup, int iters, F && work) {
    for (int i = 0; i < warmup; ++i) {
        (void) time_device_ms(q, work);
    }
    double sum    = 0.0;
    double min_ms = -1.0;
    for (int i = 0; i < iters; ++i) {
        const double ms = time_device_ms(q, work);
        sum += ms;
        if (min_ms < 0.0 || ms < min_ms) {
            min_ms = ms;
        }
    }
    return bench_result{ min_ms, sum / iters };
}

static void report(const char * form, int n_slots, size_t total_bytes, const bench_result & r) {
    const double gbps_mean = (total_bytes / 1.0e9) / (r.mean_ms / 1000.0);
    const double gbps_min  = (total_bytes / 1.0e9) / (r.min_ms / 1000.0);
    std::printf(
        "[REPACK-BENCH] form=%-24s slots=%d total_bytes=%zu mean_ms=%.4f min_ms=%.4f mean_GBps=%.2f max_GBps=%.2f\n",
        form, n_slots, total_bytes, r.mean_ms, r.min_ms, gbps_mean, gbps_min);
}

int main() {
    try {
        sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::enable_profiling{} };

        // GPT-OSS 20B expert shape -- gate/up/down all identical (hidden_size
        // == intermediate_size == 2880 for this model; see task llama.cpp-0vqt
        // plan comment for the derivation).
        constexpr int blocks_per_row = 90;
        constexpr int nrows          = 2880;
        constexpr int tile_n_total   = 16;  // caps.N * optimal_tiles_n, current device caps
        constexpr int n_slots        = 32;  // GPT-OSS 20B's real max active experts/dispatch
        constexpr int WARMUP         = 3;
        constexpr int ITERS          = 10;

        const int64_t K = (int64_t) blocks_per_row * QK_MXFP4;
        const int64_t N = nrows;

        const size_t nibble_bytes      = (size_t) (K * N / 2);
        const size_t scale_bytes       = (size_t) blocks_per_row * N;
        const size_t weight_slot_bytes = nibble_bytes + scale_bytes;

        // ---- SOA source buffers ----
        const int64_t nblocks       = (int64_t) nrows * blocks_per_row;
        const size_t  soa_qs_bytes  = (size_t) nblocks * (QK_MXFP4 / 2);
        const size_t  soa_e_bytes   = (size_t) nblocks;
        const size_t  soa_src_bytes = soa_qs_bytes + soa_e_bytes;

        // ---- XMX_TILED source buffers ----
        const int64_t n_tile_groups_n = N / tile_n_total;
        const int64_t group_bytes     = (int64_t) tile_n_total * (1 + QK_MXFP4 / 2);
        const size_t  xmx_src_bytes   = (size_t) (blocks_per_row * n_tile_groups_n * group_bytes);

        std::mt19937                       rng(1729);
        std::uniform_int_distribution<int> byte_dist(0, 255);

        auto fill_random_device = [&](size_t bytes) -> uint8_t * {
            std::vector<uint8_t> host(bytes);
            for (auto & b : host) {
                b = (uint8_t) byte_dist(rng);
            }
            uint8_t * dev = (uint8_t *) sycl::malloc_device(bytes, q);
            q.memcpy(dev, host.data(), bytes).wait();
            return dev;
        };

        std::vector<uint8_t *> soa_src(n_slots);
        std::vector<uint8_t *> xmx_src(n_slots);
        for (int s = 0; s < n_slots; ++s) {
            soa_src[s] = fill_random_device(soa_src_bytes);
            xmx_src[s] = fill_random_device(xmx_src_bytes);
        }
        std::vector<const void *> soa_srcs(soa_src.begin(), soa_src.end());
        std::vector<const void *> xmx_srcs(xmx_src.begin(), xmx_src.end());

        uint8_t * dst = (uint8_t *) sycl::malloc_device(weight_slot_bytes * n_slots, q);

        const size_t soa_total_bytes = (size_t) n_slots * (soa_src_bytes + nibble_bytes + scale_bytes);
        const size_t xmx_total_bytes = (size_t) n_slots * (xmx_src_bytes + nibble_bytes + scale_bytes);

        std::printf("[REPACK-BENCH] shape blocks_per_row=%d nrows=%d tile_n_total=%d n_slots=%d K=%lld N=%lld\n",
                    blocks_per_row, nrows, tile_n_total, n_slots, (long long) K, (long long) N);

        // ---- SOA forms ----
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    uint8_t * nib = dst + (size_t) s * weight_slot_bytes;
                    uint8_t * sca = nib + nibble_bytes;
                    repack_mxfp4_soa_to_woq(soa_src[s], nib, sca, blocks_per_row, nrows, &qq);
                }
            };
            report("soa-per-slot", n_slots, soa_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                repack_mxfp4_soa_to_woq_batched(soa_srcs.data(), n_slots, dst, weight_slot_bytes, nibble_bytes,
                                                blocks_per_row, nrows, &qq);
            };
            report("soa-batched", n_slots, soa_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    uint8_t * nib = dst + (size_t) s * weight_slot_bytes;
                    uint8_t * sca = nib + nibble_bytes;
                    repack_mxfp4_soa_to_woq_coalesced(soa_src[s], nib, sca, blocks_per_row, nrows, &qq);
                }
            };
            report("soa-coalesced-per-slot", n_slots, soa_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                repack_mxfp4_soa_to_woq_coalesced_batched(soa_srcs.data(), n_slots, dst, weight_slot_bytes,
                                                          nibble_bytes, blocks_per_row, nrows, &qq);
            };
            report("soa-coalesced-batched", n_slots, soa_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }

        // ---- XMX_TILED forms ----
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    uint8_t * nib = dst + (size_t) s * weight_slot_bytes;
                    uint8_t * sca = nib + nibble_bytes;
                    repack_mxfp4_xmx_tiled_to_woq(xmx_src[s], nib, sca, blocks_per_row, nrows, tile_n_total, &qq);
                }
            };
            report("xmx-per-slot", n_slots, xmx_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                repack_mxfp4_xmx_tiled_to_woq_batched(xmx_srcs.data(), n_slots, dst, weight_slot_bytes, nibble_bytes,
                                                      blocks_per_row, nrows, tile_n_total, &qq);
            };
            report("xmx-batched", n_slots, xmx_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    uint8_t * nib = dst + (size_t) s * weight_slot_bytes;
                    uint8_t * sca = nib + nibble_bytes;
                    repack_mxfp4_xmx_tiled_to_woq_coalesced(xmx_src[s], nib, sca, blocks_per_row, nrows, tile_n_total,
                                                            &qq);
                }
            };
            report("xmx-coalesced-per-slot", n_slots, xmx_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                repack_mxfp4_xmx_tiled_to_woq_coalesced_batched(xmx_srcs.data(), n_slots, dst, weight_slot_bytes,
                                                                nibble_bytes, blocks_per_row, nrows, tile_n_total, &qq);
            };
            report("xmx-coalesced-batched", n_slots, xmx_total_bytes, run_bench(q, WARMUP, ITERS, work));
        }

        for (int s = 0; s < n_slots; ++s) {
            sycl::free(soa_src[s], q);
            sycl::free(xmx_src[s], q);
        }
        sycl::free(dst, q);
        return 0;
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
}
