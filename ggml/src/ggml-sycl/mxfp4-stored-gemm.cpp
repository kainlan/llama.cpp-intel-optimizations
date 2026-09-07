// See mxfp4-stored-gemm.hpp for the file-level design note (llama.cpp-vtfs,
// option C step 2 -- plan Task G4). No production dispatch is touched here;
// ggml-sycl.cpp does not reference this file.

#include "mxfp4-stored-gemm.hpp"

#include "common.hpp"
#include "ggml-quants.h"
#include "ggml.h"
#include "unified-kernel.hpp"  // GGML_SYCL_ESIMD_AVAILABLE, esimd/xmx aliases, SYCL_ESIMD_KERNEL

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ggml_sycl_mxfp4_stored_gemm {

q8_1_activation_pack quantize_activations_q8_1(const float * x, int64_t M, int64_t K) {
    GGML_ASSERT(x != nullptr && M > 0 && K > 0 && K % QK8_1 == 0);
    const int64_t nb = K / QK8_1;

    q8_1_activation_pack pack;
    pack.qs.resize((size_t) (M * K));
    pack.scales.resize((size_t) (M * nb));

    std::vector<block_q8_1> blocks((size_t) nb);
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_1_ref(x + m * K, blocks.data(), K);
        for (int64_t b = 0; b < nb; ++b) {
            const block_q8_1 & blk = blocks[(size_t) b];
            std::memcpy(pack.qs.data() + (size_t) (m * K + b * QK8_1), blk.qs, (size_t) QK8_1);
            // block_q8_1's `d`/`s` union member names depend on which
            // GGML_COMMON_DECL_* branch won in THIS translation unit (see
            // ggml-common.h's GGML_COMMON_AGGR_S/_U) -- this backend's own
            // headers (pulled in via common.hpp above) can already have
            // resolved ggml-common.h under a different branch than this
            // file's own `#include "ggml-quants.h"` (GGML_COMMON_DECL_C)
            // requests, since ggml-common.h is included-once keyed on
            // GGML_COMMON_DECL. Read the leading 2 bytes by LAYOUT instead
            // of by member name: `d` (ggml_half, fp16) is always the first
            // 2 bytes of block_q8_1 regardless of which branch won -- only
            // the accessor name changes, never the memory layout. Already
            // pinned tree-wide by ggml-common.h:269's own
            // `static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) +
            // QK8_1, "wrong q8_1 block size/padding")` -- a duplicate
            // assert here would check nothing that assert does not already
            // guarantee build-wide (llama.cpp-6f73 c-py5n nit, correcting
            // round 1's redundant local static_assert).
            ggml_fp16_t d_bits;
            std::memcpy(&d_bits, &blk, sizeof(d_bits));
            pack.scales[(size_t) (m * nb + b)] = ggml_fp16_to_fp32(d_bits);
        }
    }
    return pack;
}

}  // namespace ggml_sycl_mxfp4_stored_gemm

#if GGML_SYCL_ESIMD_AVAILABLE

// SYCL kernel-name type: deliberately declared at FILE SCOPE (external
// linkage), NOT inside the anonymous namespace below, matching mmvq.cpp's
// own convention (e.g. `template <int Repeat> struct
// mxfp4_dpas_down_single_col_kernel;`) -- a kernel-name type needs external
// linkage for the SYCL runtime to identify it, and llama.cpp-6cgq is the
// precedent for what goes wrong with kernel-name handling: there, two TUs
// defined the SAME kernel name identically, causing a link-time collision.
// This name is unique to this file (verified against the rest of the tree),
// so external linkage here does not reintroduce that hazard.
template <int M_TILE> struct mxfp4_soa_gemm_int8_dpas_kernel;

namespace {

// Private re-derivations of mmvq.cpp's mxfp4_e8m0_to_fp32_half_esimd /
// mxfp4_code_values_esimd bit-tricks (mmvq.cpp:~7370-7405), renamed and
// confined to this TU's anonymous namespace. The two symbols do NOT share
// one linkage story (llama.cpp-6f73 c-py5n should-fix 1, correcting the
// round-1 comment): mxfp4_e8m0_to_fp32_half_esimd is a plain, non-static,
// non-template file-scope function at mmvq.cpp:7368, outside that file's
// own anonymous namespace at :823-957 -- ordinary external linkage, and this
// file COULD declare an extern prototype and link against the exact symbol
// mmvq.cpp's object file emits. mxfp4_code_values_esimd (mmvq.cpp:7390-7391)
// is instead an `inline` function TEMPLATE: its instantiations are formally
// external linkage too, but the compiler is free to discard (COMDAT-fold or
// simply never emit) any instantiation mmvq.cpp's own TU does not itself
// use with matching template arguments -- there is no single, guaranteed
// object-file symbol here to link against the way there is for the plain
// function, only the template DEFINITION, which is not declared in any
// shared header. Either way, the reason for a private copy is the same:
// neither symbol is declared in any shared header, so reaching them from
// here would mean hand-declaring a prototype (or including this file's
// private definition) for another TU's internal helper -- worse than a
// small, self-contained re-derivation this TU owns outright. Confining the
// copy to this TU's own anonymous namespace rules out any duplicate-symbol
// collision either way, regardless of which linkage story applies to which
// original. The formulas are reproduced verbatim -- E8M0 "halved"
// convention to match GGML_E8M0_TO_FP32_HALF, and the same e2m1
// magnitude/sign decomposition for the 4-bit code table.
//
// The scalar E8M0 decode CANNOT reuse this backend's own
// sycl_e8m0_to_fp32_half (common.hpp) here: that helper's `memcpy` is a
// non-ESIMD SYCL device call, and this kernel runs in an ESIMD context
// (SYCL_ESIMD_KERNEL) -- "SYCL device function cannot be called from an
// ESIMD context" is a hard compile error, not a style choice. ESIMD's own
// `simd<uint32_t,1>::bit_cast_view<float>()` is the ESIMD-legal
// reinterpret-bits path, exactly what mmvq.cpp's own scalar ESIMD decode
// already uses.
SYCL_ESIMD_FUNCTION inline float mxfp4_stored_gemm_e8m0_half_esimd(uint8_t e) {
    using namespace sycl::ext::intel::esimd;
    uint32_t bits;
    if (e < 2) {
        bits = 0x00200000u << e;
    } else {
        bits = static_cast<uint32_t>(e - 1) << 23;
    }
    simd<uint32_t, 1> packed(bits);
    return packed.template bit_cast_view<float>()[0];
}

template <int N>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<int8_t, N> mxfp4_stored_gemm_code_values_esimd(
    sycl::ext::intel::esimd::simd<uint8_t, N> codes) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, N> base_mag = codes & uint8_t{ 7 };
    simd<uint8_t, N> extra    = base_mag - uint8_t{ 4 };
    extra.merge(simd<uint8_t, N>(0), base_mag <= uint8_t{ 4 });
    simd<uint8_t, N> mag = base_mag + extra;
    mag.merge(simd<uint8_t, N>(12), base_mag == uint8_t{ 7 });

    simd<int8_t, N> values = mag;
    simd<int8_t, N> neg    = -values;
    values.merge(neg, (codes & uint8_t{ 8 }) != uint8_t{ 0 });
    return values;
}

// The generalised kernel: one work-GROUP per N-tile (16 output rows,
// GGML_SYCL_MXFP4_MOE_XMX_N -- the DPAS execution-size hardware constant),
// with K_PARTITIONS work-ITEMS per group splitting the K reduction (llama.cpp-6f73
// spec finding, round 2 -- see the K_PARTITIONS comment below for why this
// replaced the original single-item-per-tile launch). Each work-item's own
// per-k-tile body is otherwise unchanged: 32-element blocks (QK_MXFP4)
// reading the raw SOA qs bytes at `row_qs_offset = row * blocks_per_row * 16`
// and the E8M0 scale byte at `total_qs_bytes + row * blocks_per_row +
// k_block` -- the same addressing mxfp4_soa_load_a_vec (mmvq.cpp:7566)
// already uses on the decode path and
// tests/mxfp4-stored-layout-oracle.hpp's decode_soa_mxfp4 mirrors.
//
// DPAS operand assignment is INTENTIONALLY SWAPPED relative to
// mxfp4_pair_glu_soa_dpas_m4_sycl / mxfp4_dpas_down_single_col_sycl: those
// kernels put the WEIGHT in the Repeat-tiled "A" operand and the (single-
// column, M=1) activation in the exec_n=16-tiled "B" operand, because decode
// only ever has one real activation row and Repeat naturally tiles the
// weight/output dimension instead. Here M > 1 is the whole point of the
// task, so Repeat=M_TILE tiles the ACTIVATION rows (a real GEMM "M" axis)
// and the fixed exec_n=16 lanes tile 16 real WEIGHT/output rows instead of
// mostly-wasted padding. The DPAS call itself keeps the exact same shape
// used everywhere else in this backend: `dpas<8, Repeat, int, int, int8_t,
// int8_t>(acc, B, A)` with B VNNI-packed and A plain row-major; only which
// logical quantity (weight vs activation) is placed in which operand
// differs, and the epilogue keeps the same per-K-block
// scale-then-accumulate order mxfp4_dpas_down_single_col_sycl uses.
template <int M_TILE>
static sycl::event mxfp4_soa_gemm_int8_dpas_launch(sycl::queue &                    queue,
                                                   const uint8_t *                  soa_base,
                                                   const int8_t *                   act_qs,
                                                   const float *                    act_scales,
                                                   float *                          dst,
                                                   int                              n_out,
                                                   int                              n_k,
                                                   const std::vector<sycl::event> & deps) {
    constexpr int exec_n       = (int) GGML_SYCL_MXFP4_MOE_XMX_N;  // 16, DPAS execution-size (weight/N tile)
    constexpr int k_per        = (int) GGML_SYCL_MXFP4_MOE_XMX_K;  // 32, QK_MXFP4
    constexpr int packed_bytes = k_per / 2;                        // 16 nibble-packed bytes per block
    constexpr int an           = M_TILE * k_per;                   // activation "A" operand size
    constexpr int bn           = k_per * exec_n;                   // weight "B" operand size (VNNI-packed)
    constexpr int acc_width    = M_TILE * exec_n;                  // per-partition partial-accumulator width

    // Occupancy fix (llama.cpp-6f73, spec finding round 2): the original
    // single-item-per-tile launch created only n_tiles work-groups (180 at
    // N=2880), each serializing all k_tiles (90 at K=2880) in ONE dependent
    // chain (load -> DPAS -> accumulate, repeated). Measured on hardware:
    // time was flat across N (37 vs 2880 rows: 200-260 us) and flat across
    // cards (128 vs 256 CU: within 2%) -- the signature of a fixed
    // per-work-item latency floor, not a bandwidth- or occupancy-bound
    // kernel, since ~180 threads on a 128-CU device (~1.4 threads/CU) leaves
    // no concurrent work per CU to hide global-memory latency via SMT.
    //
    // K_PARTITIONS work-items per N-tile now split the K reduction, each
    // summing a private PARTIAL accumulator over its own slice of k_tiles,
    // then combine via an SLM hierarchical reduction (stride halved each
    // round) -- the exact pattern fattn-esimd-f16.hpp's ESIMD partitioned
    // decode kernel already uses for its KV-length split (slm_init,
    // slm_block_store/slm_block_load, `barrier()`, then
    // `for (stride = N/2; stride > 0; stride /= 2)`). 8 (not 16) is chosen
    // to stay at or under the "up to 8 threads" per compute-unit budget the
    // hardware diagnosis assumed, so one work-group's threads can be
    // co-resident on one CU without oversubscribing it; N-tiling itself is
    // deliberately UNCHANGED (n_tiles work-groups, same as before) -- this is
    // a single, isolated lever, not combined with a second untested change
    // in the same commit. K_PARTITIONS=8 DOES undershoot the >=50% peak-
    // bandwidth target this task set out to reach (measured best case:
    // ~15% of B50 peak at M=8; two further levers -- batched/coalesced
    // weight loads, then software prefetch -- both REGRESSED it instead of
    // closing the gap, see this file's git history for llama.cpp-6f73
    // rounds 4-5). This kernel is checked in at this K_PARTITIONS=8 state as
    // a numerically verified checkpoint; the bandwidth criterion itself is
    // carried forward to the follow-up task llama.cpp-kcya (tiling N for
    // more resident work-groups per CU, or reworking to the single-item-
    // per-tile + deep-prefetch shape mxfp4_pair_glu_xmx_tiled_dpas_m2 uses,
    // are both candidate levers there -- not attempted here).
    constexpr int K_PARTITIONS = 8;  // power of 2, required by the tree reduction below
    static_assert(K_PARTITIONS > 0 && (K_PARTITIONS & (K_PARTITIONS - 1)) == 0,
                  "tree reduction requires a power of two");

    // SLM budget: K_PARTITIONS partial accumulators, acc_width floats each.
    // Worst case (M_TILE=8): 8 * 128 * 4 B = 4 KiB, far under the 128 KiB
    // total SLM this hardware provides per fattn-esimd-f16.hpp:1315's
    // `constexpr size_t slm_budget = 128 * 1024; // Intel Arc has 128 KB
    // SLM`. moe-xmx-fused.hpp:66's `FusedMoEConfig::slm_size` member default
    // (65536, 64 KiB) is a DIFFERENT number for a different reason: it is
    // only the fallback `from_device()` falls back to when the actual
    // per-device query (`xmx.slm_size`, sourced from
    // `sycl::info::device::local_mem_size`, xmx-esimd-common.hpp:98) reports
    // 0 -- on any device where that query succeeds, that struct's slm_size
    // is the REAL queried capacity, not 65536. This kernel does not query
    // the device at all; it asserts directly against Arc's known 128 KiB
    // ceiling, which is the correct fixed bound to check a compile-time SLM
    // size against here. Enforced, not just asserted in prose, below.
    constexpr size_t slm_acc_size = (size_t) K_PARTITIONS * acc_width * sizeof(float);
    static_assert(slm_acc_size <= 128 * 1024,
                  "mxfp4_soa_gemm_int8_dpas_launch: SLM accumulator storage exceeds the 128 KiB/work-group "
                  "budget (fattn-esimd-f16.hpp:1315)");

    const int64_t k_tiles          = n_k / k_per;
    const int64_t n_tiles          = (static_cast<int64_t>(n_out) + exec_n - 1) / exec_n;
    const int64_t row_qs_stride    = k_tiles * packed_bytes;
    const int64_t total_qs_size    = (static_cast<int64_t>(n_k) / 2) * static_cast<int64_t>(n_out);
    const int64_t act_row_stride_q = static_cast<int64_t>(n_k);
    const int64_t act_row_stride_s = k_tiles;
    const int64_t kt_per_partition = (k_tiles + K_PARTITIONS - 1) / K_PARTITIONS;

    // Launch-geometry diagnostic (llama.cpp-6f73, spec finding round 4):
    // prints exactly what nd_range this launch used, for comparing against
    // mxfp4_pair_glu_xmx_tiled_dpas_m2's own geometry at the GPT-OSS shape
    // (see this file's commit body for that comparison) without needing a
    // GPU-side profiler.
    //
    // kt_per_partition (below) is the CEILING, i.e. what every partition
    // except the last one actually does; the last partition's own
    // k-tile count is clamped by the same `kt_end = min(kt_start +
    // kt_per_partition, k_tiles)` the kernel body uses (llama.cpp-6f73
    // spec round 3, c-p2gm nit 6) -- e.g. at K=2880 (k_tiles=90,
    // K_PARTITIONS=8) partitions 0-6 each process 12 k-tiles and
    // partition 7 processes only 6. Print both so the geometry is
    // complete without running the binary.
    if (std::getenv("GGML_SYCL_STORED_GEMM_DEBUG")) {
        const int64_t kt_start_last = (K_PARTITIONS - 1) * kt_per_partition;
        const int64_t kt_end_last   = std::min(kt_start_last + kt_per_partition, k_tiles);
        const int64_t kt_last       = std::max<int64_t>(0, kt_end_last - kt_start_last);
        std::fprintf(stderr,
                     "[mxfp4-stored-gemm] M_TILE=%d n_out=%d n_k=%d global_range=%lld local_range=%d "
                     "work_groups(n_tiles)=%lld rows_per_work_item(exec_n)=%d k_tiles_total=%lld "
                     "k_tiles_per_work_item(max)=%lld k_tiles_per_work_item(last_partition)=%lld\n",
                     M_TILE, n_out, n_k, (long long) (n_tiles * K_PARTITIONS), K_PARTITIONS, (long long) n_tiles,
                     exec_n, (long long) k_tiles, (long long) kt_per_partition, (long long) kt_last);
    }

    return queue.submit([&](sycl::handler & h) {
        if (!deps.empty()) {
            h.depends_on(deps);
        }
        h.parallel_for<mxfp4_soa_gemm_int8_dpas_kernel<M_TILE>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(n_tiles * K_PARTITIONS)),
                              sycl::range<1>(static_cast<size_t>(K_PARTITIONS))),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                slm_init<slm_acc_size>();

                const int64_t n_tile    = static_cast<int64_t>(item.get_group(0));
                const int     partition = static_cast<int>(item.get_local_id(0));
                const int64_t n_base    = n_tile * exec_n;

                const int64_t kt_start = static_cast<int64_t>(partition) * kt_per_partition;
                const int64_t kt_end   = std::min(kt_start + kt_per_partition, k_tiles);

                simd<float, acc_width> acc = 0.0f;

                for (int64_t kt = kt_start; kt < kt_end; ++kt) {
                    // Activation "A" operand: M_TILE rows x 32 int8 q8_1 codes,
                    // plain row-major (no VNNI packing for A).
                    simd<int8_t, an> a_vec;
                    // Plain C array, not a simd<> vector: y_scale is only
                    // ever read back one SCALAR element at a time (never as
                    // a whole-width DPAS operand), and `simd<float,
                    // M_TILE>::operator[]` returns a simd_view rather than a
                    // plain float in this ESIMD version -- multiplying that
                    // simd_view directly against the exec_n-wide `w_scale`
                    // vector below has no matching operator*. A plain float
                    // sidesteps the mismatch entirely, matching how
                    // mxfp4_dpas_down_single_col_sycl extracts its own
                    // per-row scalar scale (`const float w_scale =
                    // w_scale_vec[r];`) before using it in a mixed
                    // vector*scalar multiply.
                    float            y_scale[M_TILE];
#    pragma unroll
                    for (int r = 0; r < M_TILE; ++r) {
                        const int8_t * qs_ptr                      = act_qs + r * act_row_stride_q + kt * k_per;
                        a_vec.template select<k_per, 1>(r * k_per) = block_load<int8_t, k_per>(qs_ptr);
                        y_scale[r]                                 = act_scales[r * act_row_stride_s + kt];
                    }

                    // Weight "B" operand: 16 output rows x 32 MXFP4 elements,
                    // decoded from the raw stored SOA bytes and VNNI-packed
                    // ((kk/4)*exec_n*4 + n*4 + (kk%4)) -- the same int8
                    // B-operand addressing this backend's own activation
                    // packers already use (mmvq.cpp's
                    // mxfp4_dpas_pack_q8_single_col_groups_sycl), applied here
                    // to the weight side instead. Rows past n_out are left as
                    // zero (both codes and scale), matching the boundary
                    // check mxfp4_soa_load_a_vec uses on the decode path.
                    simd<int8_t, bn>    b_vec   = 0;
                    simd<float, exec_n> w_scale = 0.0f;
#    pragma unroll
                    for (int n = 0; n < exec_n; ++n) {
                        const int64_t row = n_base + n;
                        if (row < n_out) {
                            const uint8_t *             packed_ptr = soa_base + row * row_qs_stride + kt * packed_bytes;
                            simd<uint8_t, packed_bytes> packed     = block_load<uint8_t, packed_bytes>(packed_ptr);
                            simd<uint8_t, k_per>        codes;
                            codes.template select<packed_bytes, 1>(0)            = packed & uint8_t{ 0x0f };
                            codes.template select<packed_bytes, 1>(packed_bytes) = packed >> 4;
                            simd<int8_t, k_per> vals = mxfp4_stored_gemm_code_values_esimd<k_per>(codes);
#    pragma unroll
                            for (int kk = 0; kk < k_per; ++kk) {
                                b_vec[(kk / 4) * exec_n * 4 + n * 4 + (kk % 4)] = vals[kk];
                            }
                            // Plain scalar dereference, not block_load: the
                            // production reference (mxfp4_soa_load_a_vec,
                            // mmvq.cpp:7591) uses block_load<uint8_t,1> here,
                            // but that assumes 4-byte alignment of a
                            // 1-byte-granularity pointer per the oneAPI ESIMD
                            // header, which is not actually guaranteed for
                            // this address. Fixing that would mean adding
                            // `overaligned_tag<1>{}` at both this site and
                            // the production one, and touching mmvq.cpp is
                            // outside this task's scope -- so this reverts to
                            // the plain, definitely-correct scalar read
                            // rather than copying the questionable pattern
                            // (llama.cpp-6f73 c-gngd).
                            const uint8_t scale_byte = soa_base[total_qs_size + row * k_tiles + kt];
                            w_scale[n]               = mxfp4_stored_gemm_e8m0_half_esimd(scale_byte);
                        }
                    }

                    // Both weight and activation scales vary per K block
                    // (MXFP4 sub-block scaling and q8_1 block scaling
                    // respectively), so the multiply happens INSIDE this
                    // loop, per k-tile -- not once after the full K
                    // reduction. Mirrors mxfp4_dpas_down_single_col_sycl's
                    // epilogue exactly.
                    simd<int, M_TILE * exec_n> part = 0;
                    part = xmx::dpas<8, M_TILE, int, int, int8_t, int8_t>(part, b_vec, a_vec);

#    pragma unroll
                    for (int r = 0; r < M_TILE; ++r) {
                        simd<int, exec_n>   row_i                  = part.template select<exec_n, 1>(r * exec_n);
                        simd<float, exec_n> row_f                  = convert<float>(row_i) * (w_scale * y_scale[r]);
                        acc.template select<exec_n, 1>(r * exec_n) = acc.template select<exec_n, 1>(r * exec_n) + row_f;
                    }
                }

                // Combine the K_PARTITIONS work-items' partial accumulators
                // via an SLM hierarchical (tree) reduction -- same structure
                // as fattn-esimd-f16.hpp's partitioned decode kernel: each
                // work-item stores its own partial sum, a barrier makes every
                // store visible, then log2(K_PARTITIONS) rounds each halve
                // the active partition count, summing pairs, with a barrier
                // between rounds. Plain addition (not the online-softmax
                // merge fattn needs) since these are independent partial
                // dot-product sums, not incrementally-normalized values.
                slm_block_store(static_cast<size_t>(partition) * acc_width * sizeof(float), acc);
                barrier();

                for (int stride = K_PARTITIONS / 2; stride > 0; stride /= 2) {
                    if (partition < stride) {
                        simd<float, acc_width> my_acc = slm_block_load<float, acc_width>(
                            static_cast<size_t>(partition) * acc_width * sizeof(float));
                        simd<float, acc_width> partner_acc = slm_block_load<float, acc_width>(
                            static_cast<size_t>(partition + stride) * acc_width * sizeof(float));
                        slm_block_store(static_cast<size_t>(partition) * acc_width * sizeof(float),
                                        my_acc + partner_acc);
                    }
                    barrier();
                }

                // Partition 0 now holds the fully-reduced sum; it alone
                // writes the output (matching fattn-esimd-f16.hpp's
                // `if (partition_id == 0)` final-store convention).
                if (partition == 0) {
                    simd<float, acc_width> final_acc = slm_block_load<float, acc_width>(0);
#    pragma unroll
                    for (int r = 0; r < M_TILE; ++r) {
#    pragma unroll
                        for (int n = 0; n < exec_n; ++n) {
                            const int64_t row = n_base + n;
                            if (row < n_out) {
                                simd<float, 1> value = final_acc.template select<1, 1>(r * exec_n + n);
                                block_store<float, 1>(dst + (int64_t) r * n_out + row, value);
                            }
                        }
                    }
                }
            });
    });
}

}  // namespace

#endif  // GGML_SYCL_ESIMD_AVAILABLE

namespace ggml_sycl_mxfp4_stored_gemm {

sycl::event ggml_sycl_mxfp4_soa_gemm_dpas(sycl::queue &                    queue,
                                          const void *                     soa_weight_device,
                                          const int8_t *                   act_qs_device,
                                          const float *                    act_scales_device,
                                          float *                          dst_device,
                                          int                              M,
                                          int                              n_out,
                                          int                              n_k,
                                          const std::vector<sycl::event> & deps) {
    GGML_ASSERT(soa_weight_device != nullptr);
    GGML_ASSERT(act_qs_device != nullptr);
    GGML_ASSERT(act_scales_device != nullptr);
    GGML_ASSERT(dst_device != nullptr);
    GGML_ASSERT(n_out > 0);
    GGML_ASSERT(n_k > 0 && n_k % (int) GGML_SYCL_MXFP4_MOE_XMX_K == 0);
    // 32-byte-aligned base pointers, for the two device pointers the kernel
    // actually vector-loads: `block_load<uint8_t, 16>` off `soa_weight_device`
    // (the weight load) and `block_load<int8_t, 32>` off `act_qs_device`
    // (the activation load) both carry ESIMD vector-alignment requirements,
    // satisfied only because every offset the kernel computes off these two
    // bases is itself a multiple of the vector width (`row_qs_stride` and
    // `act_row_stride_q` are both multiples of 32) -- a caller passing a
    // mid-buffer sub-pointer that is not itself 32-byte aligned would get a
    // silent misaligned load. `act_scales_device` is deliberately NOT
    // asserted here (llama.cpp-6f73 c-py5n should-fix 2, correcting round
    // 1): it is only ever read by scalar subscript (`act_scales[...]`
    // above, in mxfp4_soa_gemm_int8_dpas_launch), never block_loaded, so it
    // carries no such alignment requirement and an assert on it would guard
    // nothing.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(soa_weight_device) % 32 == 0);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(act_qs_device) % 32 == 0);

#if GGML_SYCL_ESIMD_AVAILABLE
    const auto * soa_base = static_cast<const uint8_t *>(soa_weight_device);

    ggml_sycl_profile_label profile_label{};
    profile_label.name       = "mxfp4.stored_gemm.soa";
    profile_label.category   = "mxfp4";
    profile_label.queue_kind = "compute";
    const std::string metadata =
        "M=" + std::to_string(M) + ";n_out=" + std::to_string(n_out) + ";n_k=" + std::to_string(n_k);
    profile_label.metadata    = metadata.c_str();
    profile_label.device      = ggml_sycl_get_device_id_from_queue(queue);
    // Bytes actually touched: the whole expert's SOA weight buffer (qs +
    // E8M0 scales) is read once per launch (no reuse across n-tiles for a
    // different m-tile since M_TILE covers all of M in one launch), plus the
    // M x n_k activation codes/scales and the M x n_out f32 output. The
    // activation is counted ONCE even though every one of the n_tiles
    // work-items re-reads the whole M x n_k activation independently -- this
    // is the correct figure for DRAM traffic (not for total bytes loaded by
    // all lanes) under the reasonable assumption that the activation (at
    // most 8 x 2880 = 23 KB for M=8, K=2880) stays cache-resident across the
    // launch rather than being evicted and re-fetched from DRAM per
    // work-item (llama.cpp-6f73 c-ru7x item 4).
    //
    // Per-launch bandwidth is profile_label.bytes / mean_ns for ONE launch,
    // NOT the kernel-profiler CSV's aggregate `bytes` column divided by
    // mean_ns: sycl-kernel-profiler.cpp accumulates
    // `aggregate.bytes += label.bytes` once per recorded launch, so that
    // column is the SUM over `count` launches, and dividing it by mean_ns
    // overstates bandwidth by a factor of `count` (llama.cpp-6f73 c-irug).
    const size_t weight_bytes = (size_t) n_out * ((size_t) (n_k / 2) + (size_t) (n_k / 32));
    const size_t act_bytes    = (size_t) M * ((size_t) n_k + (size_t) (n_k / 32) * sizeof(float));
    const size_t dst_bytes    = (size_t) M * (size_t) n_out * sizeof(float);
    profile_label.bytes       = weight_bytes + act_bytes + dst_bytes;

    return ggml_sycl_profile_submit(queue, profile_label, [&](sycl::queue & profiled_queue) {
        switch (M) {
            case 1:
                return mxfp4_soa_gemm_int8_dpas_launch<1>(profiled_queue, soa_base, act_qs_device, act_scales_device,
                                                          dst_device, n_out, n_k, deps);
            case 2:
                return mxfp4_soa_gemm_int8_dpas_launch<2>(profiled_queue, soa_base, act_qs_device, act_scales_device,
                                                          dst_device, n_out, n_k, deps);
            case 4:
                return mxfp4_soa_gemm_int8_dpas_launch<4>(profiled_queue, soa_base, act_qs_device, act_scales_device,
                                                          dst_device, n_out, n_k, deps);
            case 8:
                return mxfp4_soa_gemm_int8_dpas_launch<8>(profiled_queue, soa_base, act_qs_device, act_scales_device,
                                                          dst_device, n_out, n_k, deps);
            default:
                GGML_ABORT("ggml_sycl_mxfp4_soa_gemm_dpas: unsupported M=%d (must be one of {1,2,4,8})", M);
        }
    });
#else
    GGML_UNUSED(soa_weight_device);
    GGML_UNUSED(act_qs_device);
    GGML_UNUSED(act_scales_device);
    GGML_UNUSED(dst_device);
    GGML_UNUSED(M);
    GGML_UNUSED(n_out);
    GGML_UNUSED(n_k);
    GGML_UNUSED(deps);
    GGML_ABORT(
        "ggml_sycl_mxfp4_soa_gemm_dpas: ESIMD/DPAS not available in this SYCL toolchain (GGML_SYCL_ESIMD_AVAILABLE=0)");
#endif
}

}  // namespace ggml_sycl_mxfp4_stored_gemm
