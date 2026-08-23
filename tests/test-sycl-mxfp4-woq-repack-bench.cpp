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

// ---------------------------------------------------------------------------
// Diagnostic reference forms (iteration 3, llama.cpp-0vqt): written when the
// committed coalesced SOA nibbles kernel (convert.cpp) still measured only
// ~20-25% of peak bandwidth on both cards after iteration 2's SLM-padding +
// vectorized-read fixes -- these forms isolate WHERE the loss is by
// measuring pieces of that kernel's work in total isolation, at the SAME
// WORKGROUP GRID as the committed kernel (one workgroup per (source block
// b, n-tile of DIAG_NBLK consecutive n's)), with their OWN fixed WG_SIZE=256
// (unchanged by this correction -- see the geometry printf below for the
// committed kernel's own WG_SIZE, which iteration 3's write-vectorization
// dropped to 32). Phase 1 (read) is one lane per n_local doing 4x aligned
// uint32 reads, matching the committed kernel's read phase, which iteration
// 3 did not touch -- scoped to the NIBBLES plane only (94% of the real
// kernel's total volume: 4,147,200 of 4,406,400 bytes/slot for the GPT-OSS
// shape), since the scales plane uses a structurally different (b,n) 2D
// tile.
//
// DIAG_NBLK must track MXFP4_WOQ_REPACK_COALESCED_NBLK in convert.cpp (both
// 16) -- this file is a separate translation unit and convert.cpp's
// constant is TU-local, so it is duplicated here rather than shared.
constexpr int DIAG_NBLK = 16;

// (2) Read-only: identical phase-1 read pattern (one lane per n_local, four
// aligned uint32 loads -- 16 contiguous bytes per lane) to the committed
// kernel, but the loaded words are only XOR-accumulated into a single
// global result via atomic (forces the compiler to keep the loads; the
// write volume is 4 bytes total, not counted in this form's reported
// bytes) -- no SLM, no destination write of real size.
static void diag_read_only_kernel(const uint8_t * __restrict__ qs,
                                  int     blocks_per_row,
                                  int64_t N,
                                  uint32_t * __restrict__ result,
                                  const sycl::nd_item<3> & item) {
    const int     b   = static_cast<int>(item.get_group(1));
    const int64_t n0  = static_cast<int64_t>(item.get_group(2)) * DIAG_NBLK;
    const int     lid = static_cast<int>(item.get_local_id(2));

    if (lid < DIAG_NBLK) {
        const int              n_local = lid;
        const int64_t          block_i = (n0 + n_local) * blocks_per_row + b;
        const uint32_t * const blk32   = reinterpret_cast<const uint32_t *>(qs + block_i * (QK_MXFP4 / 2));
        const uint32_t         acc     = blk32[0] ^ blk32[1] ^ blk32[2] ^ blk32[3];
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            aref(*result);
        aref.fetch_xor(acc);
    }
    (void) N;
}

// (3) Write-only: identical phase-2 coalesced global-write pattern (32
// k_local x DIAG_NBLK/2 n_pair threads per tile, one byte per lane,
// contiguous over n_pair for fixed k_local) to the committed kernel, but
// writes a value synthesized from the thread index -- no source reads at
// all.
static void diag_write_only_kernel(uint8_t * __restrict__ dst_nibbles, int64_t N, const sycl::nd_item<3> & item) {
    const int     b                            = static_cast<int>(item.get_group(1));
    const int64_t n0                           = static_cast<int64_t>(item.get_group(2)) * DIAG_NBLK;
    const int     lid                          = static_cast<int>(item.get_local_id(2));
    const int     k_local                      = lid / (DIAG_NBLK / 2);
    const int     n_pair                       = lid % (DIAG_NBLK / 2);
    const int64_t k                            = static_cast<int64_t>(b) * QK_MXFP4 + k_local;
    dst_nibbles[k * (N / 2) + n0 / 2 + n_pair] = static_cast<uint8_t>(lid);
}

// (5) Write-only, VECTORIZED (confirmation step before touching the real
// kernel -- the lead's diagnostics found the write-only form alone sits at
// ~80% of the full kernel's time on both cards, so the 1-byte-per-lane
// destination store, not the strided read, is the suspected limiter): same
// destination coverage as diag_write_only_kernel above, but each lane
// assembles the WHOLE row of DIAG_NBLK/2 contiguous dst bytes for one
// k_local (8 bytes for DIAG_NBLK=16) in a register and issues a SINGLE
// uint64 store, instead of DIAG_NBLK/2 lanes each doing one scalar byte
// store. Alignment: k*(N/2) is always a multiple of 8 (N/2=1440 for the
// GPT-OSS shape, itself a multiple of 8) and n0/2 = n_tile*(DIAG_NBLK/2) is
// always a multiple of 8 too (DIAG_NBLK/2=8) -- both terms 8-byte aligned
// for any k, n_tile.
static void diag_write_only_vectorized_kernel(uint8_t * __restrict__ dst_nibbles,
                                              int64_t                  N,
                                              const sycl::nd_item<3> & item) {
    constexpr int ROW_BYTES = DIAG_NBLK / 2;  // 8 for DIAG_NBLK=16
    const int     b         = static_cast<int>(item.get_group(1));
    const int64_t n0        = static_cast<int64_t>(item.get_group(2)) * DIAG_NBLK;
    const int     k_local   = static_cast<int>(item.get_local_id(2));  // one lane per k_local, 0..31
    const int64_t k         = static_cast<int64_t>(b) * QK_MXFP4 + k_local;

    uint64_t word = 0;
#pragma unroll
    for (int i = 0; i < ROW_BYTES; ++i) {
        const uint8_t v = static_cast<uint8_t>(k_local + i);  // synthesized, same spirit as diag_write_only_kernel
        word |= static_cast<uint64_t>(v) << (i * 8);
    }
    uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_nibbles + k * (N / 2) + n0 / 2);
    *dst64                 = word;
}

// (4) SLM round-trip, no transpose: reads the source block the SAME way as
// phase 1 into a PADDED SLM buffer (row = DIAG_NBLK+1, matching the
// committed kernel's bank-conflict fix -- isolates "cost of the SLM round
// trip itself" from "cost of the transpose permutation", not from the
// already-fixed bank-conflict pattern), then a straight (non-transposing)
// coalesced readback and global write of the same byte count the real
// kernel's nibble plane produces per tile.
static void diag_slm_roundtrip_kernel(const uint8_t * __restrict__ qs,
                                      uint8_t * __restrict__ dst,
                                      int     blocks_per_row,
                                      int64_t n_tiles,
                                      uint8_t * __restrict__ local,  // [DIAG_NBLK][DIAG_NBLK+1], padded
                                      const sycl::nd_item<3> & item) {
    constexpr int ROW    = DIAG_NBLK + 1;
    const int     b      = static_cast<int>(item.get_group(1));
    const int64_t n_tile = static_cast<int64_t>(item.get_group(2));
    const int64_t n0     = n_tile * DIAG_NBLK;
    const int     lid    = static_cast<int>(item.get_local_id(2));

    if (lid < DIAG_NBLK) {
        const int              n_local = lid;
        const int64_t          block_i = (n0 + n_local) * blocks_per_row + b;
        const uint32_t * const blk32   = reinterpret_cast<const uint32_t *>(qs + block_i * (QK_MXFP4 / 2));
        uint8_t * const        row_ptr = local + n_local * ROW;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t word = blk32[w];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                row_ptr[w * 4 + byte] = static_cast<uint8_t>(word >> (byte * 8));
            }
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Straight pass-through: thread lid = n_local*16+byteIdx reads back the
    // SAME (n_local, byteIdx) slot it (or another lane) wrote -- no
    // permutation, unlike the real kernel's transpose.
    const int     n_local                  = lid / 16;
    const int     byteIdx                  = lid % 16;
    const uint8_t val                      = local[n_local * ROW + byteIdx];
    const int64_t tile_lin                 = static_cast<int64_t>(b) * n_tiles + n_tile;
    dst[tile_lin * (DIAG_NBLK * 16) + lid] = val;
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

        // Committed-kernel tile geometry (quality-review R7, llama.cpp-0vqt:
        // this used to describe iteration 2's structure -- WG_SIZE=256,
        // 256 phase-2 lanes, 1 write byte/lane -- which iteration 3's
        // write-vectorization superseded; corrected to match convert.cpp's
        // committed repack_mxfp4_soa_to_woq_coalesced_nibbles_kernel).
        // read_bytes_per_WI is unchanged from iteration 2 (reads weren't
        // touched by iteration 3); the diag-* forms above are standalone
        // reference measurements at their OWN fixed geometry (DIAG_NBLK=16,
        // WG_SIZE=256) and are not affected by this correction -- see N1
        // (deferred) for reconciling the diag forms' geometry to match.
        {
            const int64_t n_tiles_geom       = N / DIAG_NBLK;
            const int     wg_size_geom       = 32;  // committed kernel's WG_SIZE_NIBBLES (convert.cpp)
            const int     read_bytes_per_wi  = 16;  // 4x uint32, one lane per n_local, 16 active lanes/WG (unchanged)
            const int     write_bytes_per_wi = 8;   // one uint64 row per lane, 32 active lanes/WG (iteration 3)
            std::printf(
                "[REPACK-BENCH] geometry workgroups_per_slot=%lldx%lld(=b x n_tile) WG_SIZE=%d "
                "phase1_active_lanes=%d read_bytes_per_WI=%d(one_burst, %d-byte-separated_across_lanes) "
                "phase2_active_lanes=%d write_bytes_per_WI=%d\n",
                (long long) blocks_per_row, (long long) n_tiles_geom, wg_size_geom, DIAG_NBLK, read_bytes_per_wi,
                (int) (blocks_per_row * (QK_MXFP4 / 2)), wg_size_geom, write_bytes_per_wi);
        }

        const size_t  nibbles_read_bytes  = (size_t) n_slots * soa_qs_bytes;
        const size_t  nibbles_write_bytes = (size_t) n_slots * nibble_bytes;
        const int64_t diag_n_tiles        = N / DIAG_NBLK;

        uint32_t * diag_result = (uint32_t *) sycl::malloc_device(sizeof(uint32_t), q);
        q.memset(diag_result, 0, sizeof(uint32_t)).wait();

        uint8_t * diag_dst =
            (uint8_t *) sycl::malloc_device(nibble_bytes, q);  // reused across diag forms (single slot's worth)

        // D2D memcpy reference: a single bulk copy moving the SAME combined
        // (read+write) byte count as the read-only/write-only/slm-roundtrip
        // forms below, across n_slots -- the practical upper bound for this
        // metric on this card.
        uint8_t * memcpy_src = (uint8_t *) sycl::malloc_device(nibbles_read_bytes, q);
        uint8_t * memcpy_dst = (uint8_t *) sycl::malloc_device(nibbles_read_bytes, q);
        q.memset(memcpy_src, 0xAB, nibbles_read_bytes).wait();  // touch every page; content is irrelevant to a copy

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

        // ---- Diagnostic reference forms (iteration 3) -- see the kernel
        // comments above. Scoped to the nibbles plane only (94% of the real
        // kernel's volume); nibbles_read_bytes == nibbles_write_bytes here.
        {
            auto work = [&](sycl::queue & qq) {
                qq.memcpy(memcpy_dst, memcpy_src, nibbles_read_bytes);
            };
            report("diag-d2d-memcpy", n_slots, nibbles_read_bytes + nibbles_write_bytes,
                   run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    const uint8_t * qs = soa_src[s];
                    qq.submit([&](sycl::handler & cgh) {
                        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, diag_n_tiles * 256),
                                                           sycl::range<3>(1, 1, 256)),
                                         [=](sycl::nd_item<3> item) {
                                             diag_read_only_kernel(qs, blocks_per_row, N, diag_result, item);
                                         });
                    });
                }
            };
            report("diag-read-only", n_slots, nibbles_read_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    (void) s;
                    uint8_t * dd = diag_dst;
                    qq.submit([&](sycl::handler & cgh) {
                        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, diag_n_tiles * 256),
                                                           sycl::range<3>(1, 1, 256)),
                                         [=](sycl::nd_item<3> item) { diag_write_only_kernel(dd, N, item); });
                    });
                }
            };
            report("diag-write-only", n_slots, nibbles_write_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    (void) s;
                    uint8_t * dd = diag_dst;
                    qq.submit([&](sycl::handler & cgh) {
                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, diag_n_tiles * 32),
                                              sycl::range<3>(1, 1, 32)),
                            [=](sycl::nd_item<3> item) { diag_write_only_vectorized_kernel(dd, N, item); });
                    });
                }
            };
            report("diag-write-only-vec", n_slots, nibbles_write_bytes, run_bench(q, WARMUP, ITERS, work));
        }
        {
            auto work = [&](sycl::queue & qq) {
                for (int s = 0; s < n_slots; ++s) {
                    const uint8_t * qs = soa_src[s];
                    uint8_t *       dd = diag_dst;
                    qq.submit([&](sycl::handler & cgh) {
                        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(DIAG_NBLK * (DIAG_NBLK + 1)), cgh);
                        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, diag_n_tiles * 256),
                                                           sycl::range<3>(1, 1, 256)),
                                         [=](sycl::nd_item<3> item) {
                                             diag_slm_roundtrip_kernel(qs, dd, blocks_per_row, diag_n_tiles,
                                                                       get_pointer(local_acc), item);
                                         });
                    });
                }
            };
            report("diag-slm-roundtrip", n_slots, nibbles_read_bytes + nibbles_write_bytes,
                   run_bench(q, WARMUP, ITERS, work));
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
        sycl::free(diag_result, q);
        sycl::free(diag_dst, q);
        sycl::free(memcpy_src, q);
        sycl::free(memcpy_dst, q);
        return 0;
    } catch (const sycl::exception & ex) {
        std::printf("SKIP: no SYCL GPU (%s)\n", ex.what());
        return 77;
    }
}
