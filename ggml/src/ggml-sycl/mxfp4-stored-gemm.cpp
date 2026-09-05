// See mxfp4-stored-gemm.hpp for the file-level design note (llama.cpp-vtfs,
// option C step 2 -- plan Task G4). No production dispatch is touched here;
// ggml-sycl.cpp does not reference this file.

#include "mxfp4-stored-gemm.hpp"

#include "common.hpp"
#include "ggml-quants.h"
#include "ggml.h"
#include "unified-kernel.hpp"  // GGML_SYCL_ESIMD_AVAILABLE, esimd/xmx aliases, SYCL_ESIMD_KERNEL

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
            // the accessor name changes, never the memory layout.
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
// confined to this TU's anonymous namespace: those two symbols are file-local
// to mmvq.cpp (not declared in any shared header), so this file cannot link
// against them and must not risk a duplicate-symbol collision with them
// either. The formulas are reproduced verbatim -- E8M0 "halved" convention to
// match GGML_E8M0_TO_FP32_HALF, and the same e2m1 magnitude/sign
// decomposition for the 4-bit code table.
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
    // in the same commit. If K_PARTITIONS=8 undershoots the >=50% peak-
    // bandwidth target, tiling N for more resident work-groups per CU is the
    // next, separate lever (not applied here).
    constexpr int K_PARTITIONS = 8;  // power of 2, required by the tree reduction below

    // SLM budget: K_PARTITIONS partial accumulators, acc_width floats each.
    // Worst case (M_TILE=8): 8 * 128 * 4 B = 4 KiB, far under the 128 KiB/
    // work-group budget this hardware provides (CLAUDE.md, SLM budget note).
    constexpr size_t slm_acc_size = (size_t) K_PARTITIONS * acc_width * sizeof(float);

    const int64_t k_tiles          = n_k / k_per;
    const int64_t n_tiles          = (static_cast<int64_t>(n_out) + exec_n - 1) / exec_n;
    const int64_t row_qs_stride    = k_tiles * packed_bytes;
    const int64_t total_qs_size    = (static_cast<int64_t>(n_k) / 2) * static_cast<int64_t>(n_out);
    const int64_t act_row_stride_q = static_cast<int64_t>(n_k);
    const int64_t act_row_stride_s = k_tiles;
    const int64_t kt_per_partition = (k_tiles + K_PARTITIONS - 1) / K_PARTITIONS;

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

            // The DPAS math below (per k-tile: build a_vec, dpas,
            // scale-then-accumulate) is factored into a MACRO, not a
            // lambda, so the fast (chunked-load) and remainder loops
            // share IDENTICAL per-k-tile logic -- only how B_VEC/
            // W_SCALE are SOURCED differs between them, never the math
            // itself. A lambda would be the natural C++ tool here, but
            // fattn-esimd-f16.hpp (this same file's own proven SLM/
            // partitioned-reduction reference) documents "lambdas don't
            // work in ESIMD kernels" and uses exactly this macro idiom
            // (its own COMPUTE_KV_PTRS) for the identical reason.
#    define ACCUMULATE_ONE_KT(KT, B_VEC, W_SCALE)                                                                       \
        do {                                                                                                            \
            /* xmx::dpas's B operand must be a concrete simd<int8_t,bn>, */                                             \
            /* not a simd_view -- passing (B_VEC) directly when the caller */                                           \
            /* supplies a select<>() slice (the fast path's chunk-extracted */                                          \
            /* view) fails to resolve against dpas's overload set ("could */                                            \
            /* not match sycl::ext::intel::esimd::simd against simd_view"). */                                          \
            /* A copy-construction here materializes the view into a real */                                            \
            /* simd<> first; the remainder path's plain simd<> arguments */                                             \
            /* copy-construct just as validly. */                                                                       \
            simd<int8_t, bn>    b_vec_local   = (B_VEC);                                                                \
            simd<float, exec_n> w_scale_local = (W_SCALE);                                                              \
            /* Activation "A" operand: M_TILE rows x 32 int8 q8_1 codes, */                                             \
            /* plain row-major (no VNNI packing for A). */                                                              \
            simd<int8_t, an>    a_vec;                                                                                  \
            /* Plain C array, not a simd<> vector: y_scale is only ever */                                              \
            /* read back one SCALAR element at a time (never as a */                                                    \
            /* whole-width DPAS operand), and simd<float,M_TILE>:: */                                                   \
            /* operator[] returns a simd_view rather than a plain float */                                              \
            /* in this ESIMD version -- multiplying that simd_view */                                                   \
            /* directly against the exec_n-wide scale vector below has */                                               \
            /* no matching operator*. A plain float sidesteps the */                                                    \
            /* mismatch, matching mxfp4_dpas_down_single_col_sycl's own */                                              \
            /* per-row scalar scale extraction. */                                                                      \
            float               y_scale[M_TILE];                                                                        \
            _Pragma("unroll") for (int r = 0; r < M_TILE; ++r) {                                                        \
                const int8_t * qs_ptr                      = act_qs + r * act_row_stride_q + (KT) * k_per;              \
                a_vec.template select<k_per, 1>(r * k_per) = block_load<int8_t, k_per>(qs_ptr);                         \
                y_scale[r]                                 = act_scales[r * act_row_stride_s + (KT)];                   \
            }                                                                                                           \
                                                                                                                        \
            simd<int, M_TILE * exec_n> part = 0;                                                                        \
            part                            = xmx::dpas<8, M_TILE, int, int, int8_t, int8_t>(part, b_vec_local, a_vec); \
                                                                                                                        \
            /* Both weight and activation scales vary per K block (MXFP4 */                                             \
            /* sub-block scaling and q8_1 block scaling respectively), */                                               \
            /* so the multiply happens per k-tile -- not once after the */                                              \
            /* full K reduction. Mirrors mxfp4_dpas_down_single_col_sycl's */                                           \
            /* epilogue exactly. */                                                                                     \
            _Pragma("unroll") for (int r = 0; r < M_TILE; ++r) {                                                        \
                simd<int, exec_n>   row_i                  = part.template select<exec_n, 1>(r * exec_n);               \
                simd<float, exec_n> row_f                  = convert<float>(row_i) * (w_scale_local * y_scale[r]);      \
                acc.template select<exec_n, 1>(r * exec_n) = acc.template select<exec_n, 1>(r * exec_n) + row_f;        \
            }                                                                                                           \
        } while (0)

                int64_t kt = kt_start;

                // FAST PATH (llama.cpp-6f73, spec finding round 3 -- measured
                // 34f170d60/2ff74942f's bandwidth at ~45 GB/s on the B50, 3x
                // short of target; mxfp4_pair_glu_xmx_tiled_dpas_m2,
                // mmvq.cpp:10034, reaches ~195 GB/s reading the SAME MXFP4
                // SOA-family bytes on the SAME card): batches CHUNK_KT=2
                // k-tiles' WEIGHT reads (packed nibbles + E8M0 scale) into
                // ONE combined load per output row instead of CHUNK_KT
                // separate small reads. SOA stores one row's k-tiles
                // CONTIGUOUSLY (row_qs_stride = k_tiles*packed_bytes), so
                // CHUNK_KT consecutive k-tiles for a fixed row are one
                // contiguous byte run -- halving the weight-side transaction
                // count (exec_n=16 rows x 2 reads/k-tile before, x 2
                // reads/2-k-tiles now), the dominant contributor since it is
                // 2x the activation side's <= M_TILE=8 reads/k-tile (left
                // unbatched here; the smaller, already-cheaper side). The
                // CROSS-row stride between the 16 output rows
                // (row_qs_stride bytes apart) is unavoidable under SOA
                // regardless of this batching -- m2's XMX_TILED layout
                // groups rows contiguously instead specifically to avoid
                // that gather, which is why it does not need this at all;
                // SOA (this task's own scope, the "down" role) has no such
                // layout freedom, so batching K instead of N is the
                // available lever. CHUNK_KT=2 (not 4) to bound the
                // temporary b_chunk register footprint (bn*CHUNK_KT =
                // 1024 B) against GRF pressure from the DPAS operands
                // already live per iteration.
                constexpr int64_t CHUNK_KT = 2;
                for (; kt + CHUNK_KT <= kt_end; kt += CHUNK_KT) {
                    simd<int8_t, bn * CHUNK_KT>    b_chunk       = 0;
                    simd<float, exec_n * CHUNK_KT> w_scale_chunk = 0.0f;
#    pragma unroll
                    for (int n = 0; n < exec_n; ++n) {
                        const int64_t row = n_base + n;
                        if (row < n_out) {
                            const uint8_t * packed_ptr = soa_base + row * row_qs_stride + kt * packed_bytes;
                            simd<uint8_t, packed_bytes * CHUNK_KT> packed_chunk =
                                block_load<uint8_t, packed_bytes * CHUNK_KT>(packed_ptr);
                            // Plain scalar base, not block_load, matching the
                            // remainder path below (c-gngd: block_load here
                            // assumes 4-byte alignment of a 1-byte-
                            // granularity pointer that is not guaranteed;
                            // "neither site" uses it, see the remainder
                            // path's own comment).
                            const uint8_t * scale_ptr = soa_base + total_qs_size + row * k_tiles + kt;
#    pragma unroll
                            for (int c = 0; c < CHUNK_KT; ++c) {
                                simd<uint8_t, packed_bytes> packed =
                                    packed_chunk.template select<packed_bytes, 1>(c * packed_bytes);
                                simd<uint8_t, k_per> codes;
                                codes.template select<packed_bytes, 1>(0)            = packed & uint8_t{ 0x0f };
                                codes.template select<packed_bytes, 1>(packed_bytes) = packed >> 4;
                                simd<int8_t, k_per> vals = mxfp4_stored_gemm_code_values_esimd<k_per>(codes);
#    pragma unroll
                                for (int kk = 0; kk < k_per; ++kk) {
                                    b_chunk[c * bn + (kk / 4) * exec_n * 4 + n * 4 + (kk % 4)] = vals[kk];
                                }
                                w_scale_chunk[c * exec_n + n] = mxfp4_stored_gemm_e8m0_half_esimd(scale_ptr[c]);
                            }
                        }
                    }

#    pragma unroll
                    for (int c = 0; c < CHUNK_KT; ++c) {
                        // Each B_VEC/W_SCALE argument below is wrapped in its
                        // own parentheses: the preprocessor's macro-argument
                        // scanner tracks nesting via () only, not <>, so the
                        // comma inside `select<bn, 1>` would otherwise be
                        // misread as a 4th macro argument separator.
                        ACCUMULATE_ONE_KT(kt + c, (b_chunk.template select<bn, 1>(c * bn)),
                                          (w_scale_chunk.template select<exec_n, 1>(c * exec_n)));
                    }
                }

                // REMAINDER: fewer than CHUNK_KT k-tiles left in this
                // partition's range (only reachable when k_tiles is not a
                // multiple of K_PARTITIONS*CHUNK_KT -- for this task's
                // actual shapes, K=2880 => k_tiles=90, K_PARTITIONS=8,
                // CHUNK_KT=2, every partition's range is even-length and
                // this loop never executes; kept for correctness at other
                // n_k values the entry point's contract still allows).
                // Weight "B" operand: 16 output rows x 32 MXFP4 elements,
                // decoded from the raw stored SOA bytes and VNNI-packed
                // ((kk/4)*exec_n*4 + n*4 + (kk%4)) -- the same int8
                // B-operand addressing this backend's own activation
                // packers already use (mmvq.cpp's
                // mxfp4_dpas_pack_q8_single_col_groups_sycl), applied here
                // to the weight side instead. Rows past n_out are left as
                // zero (both codes and scale), matching the boundary check
                // mxfp4_soa_load_a_vec uses on the decode path.
                for (; kt < kt_end; ++kt) {
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
                    ACCUMULATE_ONE_KT(kt, b_vec, w_scale);
                }
#    undef ACCUMULATE_ONE_KT

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
