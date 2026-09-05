// GPU numerics guard for the D=512 decode-shaped flash-attention routes --
// both the "fattn.decode.tile_d512" kernel (submit_fattn_tile_d512 /
// launch_fattn_tile_d512, fattn-tile.hpp) and, since llama.cpp-zwsj Phase 2,
// the "fattn.decode.esimd_partitioned" ESIMD tier that now serves ne01==1
// by default -- gemma4's seven D=512 global-attention layers run one or the
// other every decode token (llama.cpp-ebxw part 2, llama.cpp-zwsj / plan
// Task P3). Built as a REGRESSION GUARD, not a discovery test: it must PASS
// on the current kernel(s) first, and covers BOTH routes via the TWO-STATE
// DRIVER below, since GGML_SYCL_FA_D512_DECODE_ESIMD selects between them.
//
// SEVEN CASES (run_state_in_current_process()): ne01=1 mask=nullptr at
// n_kv in {32,512,4096}; ne01=1 WITH a real (windowed) mask at n_kv=512 and
// n_kv=4096 -- the actual production decode shape, llama always passes a
// KQ mask even at single-token decode; ne01=4 WITH a mask, which the ESIMD
// tier REFUSES into the tile route regardless of the toggle (see the
// ROUND 2 note below and llama.cpp-wais); H_q=8/H_kv=2 at ne01=1. Every
// ne01=1 case (masked or not) resolves through the toggle's selected
// route; only the ne01=4 case is pinned to tile unconditionally.
//
// SHAPE: this is the exact shape the D=512 route has no alternative for.
// ggml_sycl_flash_attn_ext_supported() (fattn.cpp) admits D=512 only via
// EITHER the oneDNN SDPA route (llama.cpp-jahv, requires ne01 >=
// GGML_SYCL_FA_ONEDNN_MIN_NCOLS, default 8) or this file's two routes
// (llama.cpp-dtpk's tile, llama.cpp-zwsj's ESIMD tier, no ne01 floor).
// At ne01=1 (decode) the oneDNN route is structurally unreachable
// regardless of scale. This mirrors test-sycl-fattn-onednn-gates.cpp's own
// build_d512_tile_flash_attn_ext_op()/test_supports_op_admits_d512_tile_decode_shape
// admission case (same file, ne01_q=1) -- that test proves the op is
// ADMITTED into these routes; this file proves their OUTPUT is correct.
//
// GQA SHAPE: gemma4's exact H_q/H_kv for its D=512 global-attention layers
// is not recorded on llama.cpp-ebxw (checked; the profiling comments there
// give per-token kernel costs, not per-tensor head counts). Per this task's
// own instruction, falling back to H_q=8, H_kv=1 (gqa_ratio=8) --
// SAYING SO HERE per that instruction. K->ne[2]=1, Q->ne[2]=8 satisfies
// ggml_sycl_fattn_d512_tile_admissible()'s `Q->ne[2] % K->ne[2] == 0` gate.
//
// n_kv in {32, 512, 4096} covers below/at/well-above one FATTN_KQ_STRIDE
// (256) -- relevant to the spike this test unblocks (llama.cpp-zwsj Phase
// 1/2): the lead's launch-bound-vs-compute-bound verdict comes from how
// fattn.decode.tile_d512's mean cost moves across exactly this axis
// (GGML_SYCL_KERNEL_PROFILE captures), and this file's job is only to prove
// each of those n_kv values still produces a numerically correct kernel
// after any tiering change under test.
//
// COVERAGE ADDED FOR THE ESIMD TIER, ROUND 1 (spec review llama.cpp-zwsj/
// c-7iey, findings 2 and 4): `launch_fattn_esimd_f16_optimized`'s per-query
// mask indexing (`stride_mask * query_idx`, fattn-esimd-f16.hpp) was
// entirely untested (every case passed mask=nullptr), and H_kv=1 (this
// file's only GQA shape until then) makes kv_head = head/gqa_ratio
// identically 0 for every one of the 8 Q heads, so a wrong head-mapping
// bug in the ESIMD kernel's own gqa_ratio/kv_head computation would be
// undetectable. Round 1 added an ne01=4 case with a real per-query
// causal-style mask (query qr may see kv positions <= n_kv-ne01+qr,
// matching a genuine speculative-decode causal pattern; masked positions
// originally carried -10000.0f -- switched to -INFINITY in spec review
// llama.cpp-zwsj/c-ej1v nit 5, see build_causal_like_mask()'s comment)
// and an H_q=8/H_kv=2 (gqa_ratio=4) case.
//
// ROUND 2, A REAL DEFECT (lead hardware finding, llama.cpp-zwsj/c-1ha7;
// finding A): running round 1's ne01=4+mask case on hardware found
// launch_fattn_esimd_f16_optimized<512,...> returns GARBAGE for it (94%
// of elements wrong, max_diff ~1.6) while the tile kernel agrees with the
// same CPU reference to ~3.6e-3 (its own known f16 precision floor, not a
// defect) -- proving the reference and mask construction are correct and
// the ESIMD D=512 dispatch is wrong for ne01>1. The plan's ne01<=8
// engagement range for the ESIMD tier could not be verified correct
// without hardware to debug it against, so fattn.cpp's D==512 branch was
// narrowed to engage the ESIMD tier at ne01==1 ONLY -- a dispatch refusal,
// not a kernel fix -- and ne01 in 2..8 now falls through to the tile route
// unconditionally (see that file's finding-A comment for the full
// reasoning, including why this is also gemma4's actual production
// decode shape). Root cause (D=512-specific vs. shared with the D<=256
// ESIMD family's own multi-query path) is not established; tracked as
// follow-up work on llama.cpp-wais, which owns a D=256 ne01=4 mask=1
// control to decide shared-kernel vs D=512-specific. Two consequences for
// THIS file's coverage:
//   - The ne01=4+mask case now tests the REFUSAL BOUNDARY, not the ESIMD
//     tier's multi-query correctness: BOTH toggle states must route it
//     through the (unchanged) tile kernel and each independently agree
//     with the reference within tolerance (the two states are never
//     cross-compared directly in this driver -- see main()'s comment).
//   - The production shape (ne01=1 WITH a mask -- llama always passes a
//     KQ mask to flash attention, even at single-token decode) had NEVER
//     been tested; two new cases add it at n_kv=512 and n_kv=4096. A plain
//     causal mask at ne01=1 sees every kv position (visible_upto=n_kv-1),
//     a no-op, so these cases pass a `window` narrower than n_kv,
//     masking out the leading half of positions -- matching a
//     sliding-window layer and genuinely exercising the mask path (see
//     build_causal_like_mask()'s comment).
//
// REFERENCE: double-precision CPU softmax(scale * Q K^T) V, transcribed
// directly (no shared code with the kernel or with any other reference
// file) -- Q converted to double from its F32 storage, K/V from their F16
// storage via ggml_fp16_to_fp32(), matching what the GPU kernel actually
// consumes (Q is F32 by construction here -- flash_attn_tile<> has no Q_type
// template and llama-graph.cpp's build_attn_mha() skips the blanket F16 Q
// cast specifically for D=512, so a real op reaching this route always has
// F32 Q; K/V are F16, the only type ggml_sycl_fattn_d512_tile_admissible()
// accepts).
//
// METRIC (lead hardware finding, llama.cpp-zwsj/c-1ha7, finding B -- REPLACES
// the round-1 per-element `1e-3 + 1e-3*|ref|` predicate entirely): hardware
// testing at the round-1 amplitude (below) found the tile kernel's own
// correct f16 accumulation error is 2.0e-3 to 5.0e-3 ABSOLUTE on |out| in
// the 0.3-0.9 range -- 2 to 5x the round-1 abs tolerance -- so that
// predicate FAILED THE CORRECT TILE KERNEL on every single case. Per-
// element tolerance and mutant-sensitivity are in direct tension here: the
// tile kernel's real error is concentrated in a SMALL FRACTION of elements
// (4/4096 to 411/16384 observed, i.e. 0.1%-2.5%) with occasional larger
// deviations, so any per-element bound loose enough to admit that outlier
// tail is also loose enough to admit a UNIFORM few-percent multiplicative
// bug in every element, since the two error shapes look identical to a
// max-diff-based check.
//
// The fix is to compare on NMSE (normalized mean squared error,
// `mse(ref,got)/mse(ref,0)`) instead -- the exact metric and comparator
// tests/test-backend-ops.cpp uses for GGML_OP_FLASH_ATTN_EXT
// (`test_flash_attn_ext::max_nmse_err()` = 5e-4) -- because NMSE aggregates
// over the WHOLE output vector: a sparse outlier tail contributes only its
// own small fraction to the sum (numerically small even when individual
// diffs are 5e-3), while a uniform multiplicative bug contributes to EVERY
// term and is scale-invariant by construction (a 1% multiplicative error
// has NMSE = 0.01^2 = 1e-4 exactly, regardless of n_kv or amplitude -- this
// also makes the round-1 "degenerate near-zero reference" concern moot for
// the NMSE term itself, since NMSE is already normalized by the reference's
// own scale). A GROSS per-element check (any single |diff| > 0.05) is kept
// alongside NMSE specifically to catch finding A's failure mode (94% wrong,
// max_diff ~1.6) even if some future shape's NMSE were diluted by a huge N.
//
// Threshold: NMSE <= 5e-5 (lead hardware finding, llama.cpp-zwsj/c-j4k7,
// superseding a round-2 choice of 5e-4 -- upstream's own precedent for
// this exact op family, which turned out too loose here: see NMSE_MAX's
// own comment for why). Verified with the mutant the review specified
// (scratch-only, not committed): a standalone host program computed the
// SAME double-precision reference (same RNG seeds, same amplitude below)
// and simulated an output scaled by 1.01x (a uniform 1% multiplicative
// bug). Its NMSE is EXACTLY 1e-4 at every n_kv (mathematically --
// (scale-1)^2, independent of amplitude, mask, or shape) -- 2x ABOVE the
// 5e-5 threshold, so the mutant is now CAUGHT everywhere. The lead's
// hardware-measured NMSE for the (correct) tile kernel against this same
// reference -- 8.25e-7/4.40e-6/2.49e-5/1.48e-5 across the n_kv=32, 512,
// 4096-unmasked, and 4096-masked cases -- clears 5e-5 with >=2x headroom
// at every point (worst case 2.49e-5, 2.0x below); the ESIMD tier's own
// measured NMSE (~1e-13, both toggle states, every case) clears it by
// nine orders of magnitude. What NMSE (plus the gross check) ALSO
// robustly catches is finding A's class of defect -- for that failure
// NMSE is far above 1.0 (order-of-magnitude wrong on 94% of elements) and
// the gross check trips immediately (max_diff ~1.6 >> 0.05).
//
// AMPLITUDE (spec review llama.cpp-zwsj/c-7iey round 1, finding 3): Q/K
// magnitude is deliberately wide (U(-3,3), not U(-0.1,0.1)) so the QK
// logits actually spread -- at the narrow amplitude the softmax was
// essentially uniform (logit stddev ~3.3e-3) and the output collapsed to
// the plain mean of n_kv V samples, degenerate for a DIFFERENT reason NMSE
// does not fix: it barely exercises the softmax/max-tracking mechanism at
// all (any reasonable "average of V" implementation would pass). Widening
// Q/K raises the logit stddev to ~O(1), making the softmax genuinely
// peaked at every n_kv and the output track a dominant V sample's
// magnitude (V ~ U(-1,1)) -- a real exercise of the attention mechanism,
// independent of the metric-choice fix above.
//
// DISPATCH VISIBILITY: this file does not assert which named kernel ran --
// that is the lead's job, either from a GGML_SYCL_KERNEL_PROFILE capture
// (the fattn.decode.tile_d512 row, fattn-tile.hpp's llama.cpp-86a7 profile
// wrapper) or by re-running with GGML_SYCL_FA_DISPATCH_DEBUG=1, which prints
// a line of the form
//   [SYCL] fattn selected [d512] tile_d512 D=512 ne01=1 ne11=<n_kv> H_q=8 H_kv=1
// per dispatch (fattn.cpp, the branch just above launch_fattn_tile_d512()'s
// call site). Numerics-wrong-on-a-different-kernel and
// numerics-wrong-on-this-kernel would look identical from inside this file
// alone; the profiled/debug capture is what pins which kernel produced the
// output this file scores. With the llama.cpp-zwsj Phase 2 tier landed,
// this now names one of TWO routes ("tile_d512" or "esimd_partitioned"),
// selected by GGML_SYCL_FA_D512_DECODE_ESIMD -- see the TWO-STATE DRIVER
// note below for why this file exercises both rather than trusting one.
//
// TWO-STATE DRIVER (Phase 2): GGML_SYCL_FA_D512_DECODE_ESIMD picks between
// the tile route (the only one Phase 1 exercised) and the new decode-shaped
// ESIMD route this file's dispatch site also latches into a function-local
// static on first use (fattn.cpp, mirroring materialize_enabled()'s pattern
// in test-sycl-fattn-onednn-gates.cpp -- see that file's header for why an
// in-process setenv() after the first D=512 dispatch cannot move a plan a
// previous call already latched). So testing both states needs two SEPARATE
// PROCESSES, not two setenv() calls in one: main() forks and re-execs itself
// once per state (mirroring the ONEAPI_DEVICE_SELECTOR self-reexec already
// below, done AFTER that one so ONEAPI_DEVICE_SELECTOR is already inherited
// and the child does not re-trigger it), each running the full n_kv sweep
// against the double reference before either child has touched the SYCL
// device at all -- so the toggle is set from process start, not raced
// against a cached dispatch decision.
//
// GPU and model-loading binaries in this fork are run only from the lead
// session, one at a time (CLAUDE.md, Hard-Won Rules) -- this binary is no
// exception:
//
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-fattn-tile-d512-decode

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "test-skip.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int g_failures = 0;

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                                 ggml_tensor *              tensor,
                                                 ggml_backend_buffer_usage  usage) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static void fill_f32_random(std::vector<float> & out, int64_t n, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    out.resize((size_t) n);
    for (float & v : out) {
        v = dist(rng);
    }
}

static void fill_f16_random(std::vector<ggml_fp16_t> & out, int64_t n, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    out.resize((size_t) n);
    for (ggml_fp16_t & v : out) {
        v = ggml_fp32_to_fp16(dist(rng));
    }
}

// A genuine speculative-decode-style causal (optionally windowed) mask:
// query row qr (0-indexed, out of ne01) may see kv positions in
// [visible_from, visible_upto] where visible_upto = n_kv - ne01 + qr (the
// ne01 queries occupy the LAST ne01 kv-cache slots in order, each seeing
// everything up to and including its own slot) and visible_from =
// max(0, visible_upto - window + 1). `window` defaults to n_kv (no lower
// bound -- plain causal, masking a different, growing SUFFIX of positions
// for higher qr); a smaller window ALSO masks a PREFIX, matching a
// sliding-window layer -- the only way to exercise the mask path at
// ne01=1 at all, since a plain causal mask at ne01=1 sees every position
// (visible_upto = n_kv-1) and is a no-op. Masked positions carry
// -INFINITY (matching llama's own KQ_mask convention, src/llama-graph.cpp;
// this fork's bench-sycl-fattn-gptoss.cpp uses -10000.0f instead, but that
// bench never runs through the CPU reference below, so there is no reason
// to match it here in preference to the real convention). Verified benign
// for both kernels' arithmetic: `grep -c INFINITY` is 0 in both
// fattn-tile.hpp and fattn-esimd-f16.hpp, so neither branches on infinity
// -- each simply adds the mask value to the score, and compute_reference()
// below does the same, so exp(-inf - finite_max) = 0 exactly as a real
// -10000.0f underflow would, with no risk of a masked position ever being
// the row's own max (every query row has >=1 visible, finite-logit
// position by construction, since window >= 1). visible positions carry
// 0.0f. Layout ne=[n_kv, ne01, 1, 1] contiguous (flat index t + n_kv*qr),
// matching what ggml_flash_attn_ext's own mask-shape asserts require here
// (mask->ne[2]=1 divides Q->ne[2]=H_q trivially; mask->ne[3]=1 divides
// Q->ne[3]=1).
static void build_causal_like_mask(std::vector<ggml_fp16_t> & out, int n_kv, int ne01, int window = -1) {
    if (window <= 0 || window > n_kv) {
        window = n_kv;
    }
    out.assign((size_t) n_kv * ne01, ggml_fp32_to_fp16(0.0f));
    for (int qr = 0; qr < ne01; ++qr) {
        const int visible_upto = n_kv - ne01 + qr;                        // inclusive
        const int visible_from = std::max(0, visible_upto - window + 1);  // inclusive
        for (int t = 0; t < n_kv; ++t) {
            if (t < visible_from || t > visible_upto) {
                out[(size_t) t + (size_t) n_kv * qr] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
    }
}

// Double-precision softmax(scale * Q K^T + mask) V, per (query row, head),
// MQA/GQA-aware via kv_h = h / (H_q/H_kv) (matches the production kv-head
// mapping; degenerate to kv_h=0 for every h at H_kv=1). Q is D*ne01*H_q
// floats (ne=[D,ne01,H_q,1] contiguous, so flat index d + D*(qr + ne01*h));
// K/V are D*n_kv*H_kv half-precision elements (ne=[D,n_kv,H_kv,1]
// contiguous, flat index d + D*(t + n_kv*kv_h)); mask (nullable) is
// n_kv*ne01 half-precision elements (ne=[n_kv,ne01,1,1] contiguous, flat
// index t + n_kv*qr) added to the pre-softmax logit, matching what
// ggml_flash_attn_ext actually does with a mask; output is D*H_q*ne01
// floats in the SAME [d + D*(h + H_q*qr)] layout ggml_flash_attn_ext's own
// dst ne={DV,H_q,ne01,ne03} produces at ne03=1.
static void compute_reference(const std::vector<float> &       Q,
                              const std::vector<ggml_fp16_t> & K,
                              const std::vector<ggml_fp16_t> & V,
                              const std::vector<ggml_fp16_t> * mask,
                              int                              D,
                              int                              H_q,
                              int                              H_kv,
                              int                              n_kv,
                              int                              ne01,
                              float                            scale,
                              std::vector<float> &             out) {
    out.assign((size_t) D * H_q * ne01, 0.0f);
    const int           n_rep = H_q / H_kv;
    std::vector<double> logits((size_t) n_kv);

    for (int qr = 0; qr < ne01; ++qr) {
        const size_t q_row_offset = (size_t) D * qr;

        for (int h = 0; h < H_q; ++h) {
            const int    kv_h       = h / n_rep;
            const size_t q_head_off = q_row_offset + (size_t) D * ne01 * h;

            double max_logit = -std::numeric_limits<double>::infinity();
            for (int t = 0; t < n_kv; ++t) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double q = (double) Q[q_head_off + (size_t) d];
                    const double k = (double) ggml_fp16_to_fp32(K[(size_t) kv_h * n_kv * D + (size_t) t * D + d]);
                    dot += q * k;
                }
                double logit = dot * (double) scale;
                if (mask) {
                    logit += (double) ggml_fp16_to_fp32((*mask)[(size_t) t + (size_t) n_kv * qr]);
                }
                logits[(size_t) t] = logit;
                max_logit          = std::max(max_logit, logit);
            }

            double              denom = 0.0;
            std::vector<double> weight((size_t) n_kv);
            for (int t = 0; t < n_kv; ++t) {
                weight[(size_t) t] = std::exp(logits[(size_t) t] - max_logit);
                denom += weight[(size_t) t];
            }

            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int t = 0; t < n_kv; ++t) {
                    const double v = (double) ggml_fp16_to_fp32(V[(size_t) kv_h * n_kv * D + (size_t) t * D + d]);
                    acc += weight[(size_t) t] * v;
                }
                const size_t out_idx = (size_t) d + (size_t) D * h + (size_t) D * H_q * qr;
                out[out_idx]         = (float) (denom > 0.0 ? acc / denom : 0.0);
            }
        }
    }
}

// NMSE (mse(ref,got)/mse(ref,0)) plus a gross per-element check, replacing
// the round-1 per-element |got-ref| <= abs_tol + rel_tol*|ref| predicate --
// see the file header "METRIC" note for why (round 2, finding B: that
// predicate failed the CORRECT tile kernel on hardware). GROSS_ABS_MAX
// exists only to catch a finding-A-class failure (a wrong dispatch/kernel,
// not a precision difference) even if NMSE were ever diluted by a very
// large output vector.
//
// NMSE_MAX = 5e-5 (lead hardware finding, llama.cpp-zwsj/c-j4k7): the
// round-2 choice, 5e-4 (tests/test-backend-ops.cpp's own
// test_flash_attn_ext::max_nmse_err() for this exact op), let the 1%
// mutant through -- 1e-4 < 5e-4 -- because the round-2 estimate of the
// tile kernel's real NMSE (a pessimistic per-element upper bound, since no
// real per-element data was available) was 1e-4 to 3e-4, an order of
// magnitude too high. The lead's HARDWARE-MEASURED NMSE against the same
// double-precision reference this file uses: 8.25e-7 (n_kv=32), 4.40e-6
// (n_kv=512), 2.49e-5 (n_kv=4096, unmasked), 1.48e-5 (n_kv=4096, masked);
// ESIMD tier ~1e-13 (near machine precision) in every state. 5e-5 sits
// >=2x ABOVE the worst measured correct-kernel value (2.49e-5) and exactly
// 2x BELOW the 1% mutant's 1e-4 -- see the file header "METRIC" note for
// the mutant re-verification at this threshold.
static constexpr double NMSE_MAX      = 5e-5;
static constexpr float  GROSS_ABS_MAX = 0.05f;

static void compare(const char * what, const std::vector<float> & got, const std::vector<float> & ref) {
    if (got.empty() || ref.empty() || got.size() != ref.size()) {
        std::printf("FAIL [%s]: size mismatch or missing output (got=%zu ref=%zu)\n", what, got.size(), ref.size());
        ++g_failures;
        return;
    }
    double mse_diff = 0.0;
    double mse_ref  = 0.0;
    float  max_diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = (double) got[i] - (double) ref[i];
        mse_diff += diff * diff;
        mse_ref += (double) ref[i] * (double) ref[i];
        max_diff = std::max(max_diff, (float) std::fabs(diff));
    }
    if (mse_ref <= 0.0) {
        std::printf("FAIL [%s]: degenerate all-zero reference -- NMSE denominator is zero\n", what);
        ++g_failures;
        return;
    }
    const double nmse     = mse_diff / mse_ref;
    const bool   nmse_ok  = nmse <= NMSE_MAX;
    const bool   gross_ok = max_diff <= GROSS_ABS_MAX;
    const bool   ok       = nmse_ok && gross_ok;
    std::printf("%s [%s]: nmse=%.6e (max %.1e) max_diff=%.6e (gross max %.2f) n=%zu\n", ok ? "OK" : "FAIL", what, nmse,
                NMSE_MAX, max_diff, (double) GROSS_ABS_MAX, got.size());
    if (!ok) {
        ++g_failures;
    }
}

// ne01=1, use_mask=false is the original decode-shaped, mask-free case;
// ne01>1 and/or use_mask=true exercise the coverage the spec review added
// (findings 2 and 4 -- see the file header "COVERAGE ADDED FOR THE ESIMD
// TIER" notes). `window` (default -1 = plain causal, no lower bound) is
// forwarded to build_causal_like_mask() -- pass a value < n_kv to also mask
// a PREFIX, the only way to exercise the mask path at ne01=1 (see that
// function's comment).
static void run_shape(ggml_backend_t backend,
                      int            D,
                      int            H_q,
                      int            H_kv,
                      int            n_kv,
                      int            ne01     = 1,
                      bool           use_mask = false,
                      int            window   = -1) {
    std::printf("== shape D=%d H_q=%d H_kv=%d n_kv=%d ne01=%d mask=%d window=%d ==\n", D, H_q, H_kv, n_kv, ne01,
                (int) use_mask, window);

    std::mt19937 rng(0xd512u ^ ((unsigned) n_kv * 2654435761u) ^ ((unsigned) H_q * 40503u) ^
                     ((unsigned) H_kv * 0x9e3779b9u) ^ ((unsigned) ne01 * 0x85ebca6bu));

    std::vector<float>       q_data;
    std::vector<ggml_fp16_t> k_data;
    std::vector<ggml_fp16_t> v_data;
    // Wide magnitude (spec review llama.cpp-zwsj/c-7iey, finding 3 -- see
    // the file header "AMPLITUDE" note): U(-0.1,0.1) made the softmax
    // near-uniform, barely exercising the softmax/max-tracking mechanism.
    // Q/K at U(-3,3) gives a logit stddev of order 1, a genuinely peaked
    // softmax at every n_kv, and an output magnitude that tracks V's
    // amplitude (U(-1,1)) instead of collapsing to an averaged mean.
    fill_f32_random(q_data, (int64_t) D * ne01 * H_q, rng, -3.0f, 3.0f);
    fill_f16_random(k_data, (int64_t) D * n_kv * H_kv, rng, -3.0f, 3.0f);
    fill_f16_random(v_data, (int64_t) D * n_kv * H_kv, rng, -1.0f, 1.0f);

    std::vector<ggml_fp16_t> mask_data;
    if (use_mask) {
        build_causal_like_mask(mask_data, n_kv, ne01, window);
    }

    const float scale = 1.0f / std::sqrt((float) D);

    std::vector<float> ref;
    compute_reference(q_data, k_data, v_data, use_mask ? &mask_data : nullptr, D, H_q, H_kv, n_kv, ne01, scale, ref);

    // No separate degeneracy-floor check here (round 1 had one, keyed off
    // the old per-element abs_tol): NMSE is already normalized by the
    // reference's own scale, so compare() below detects a genuinely
    // all-zero reference on its own (mse_ref <= 0.0) -- see the file
    // header "METRIC" note.
    char label[112];
    std::snprintf(label, sizeof(label), "D512-decode-vs-cpu-reference n_kv=%d ne01=%d H_kv=%d mask=%d window=%d", n_kv,
                  ne01, H_kv, (int) use_mask, window);

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL: ggml_init failed\n");
        ++g_failures;
        return;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, ne01, H_q, 1);
    ggml_set_name(q, "fattn_d512_decode_q");
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H_kv, 1);
    ggml_set_name(k, "fattn_d512_decode_k");
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H_kv, 1);
    ggml_set_name(v, "fattn_d512_decode_v");

    ggml_tensor * mask = nullptr;
    if (use_mask) {
        // ne=[n_kv, ne01, 1, 1] -- matches build_causal_like_mask()'s layout
        // and ggml_flash_attn_ext's own mask-shape asserts (mask->ne[2]=1
        // divides Q->ne[2]=H_q; mask->ne[3]=1 divides Q->ne[3]=1).
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, ne01, 1, 1);
        ggml_set_name(mask, "fattn_d512_decode_mask");
    }

    // `mask` is nullptr for the mask-free cases, or the windowed/causal
    // tensor built above (use_mask) -- see build_causal_like_mask()'s
    // comment for how each case's mask is shaped.
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_set_name(out, "fattn_d512_decode_out");

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);

    ggml_backend_buffer_t q_buf = alloc_tensor_buffer(dev_buft, q, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t k_buf = alloc_tensor_buffer(dev_buft, k, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t v_buf = alloc_tensor_buffer(dev_buft, v, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t mask_buf =
        mask ? alloc_tensor_buffer(dev_buft, mask, GGML_BACKEND_BUFFER_USAGE_COMPUTE) : nullptr;
    ggml_backend_buffer_t out_buf = alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    auto cleanup = [&]() {
        if (q_buf) {
            ggml_backend_buffer_free(q_buf);
        }
        if (k_buf) {
            ggml_backend_buffer_free(k_buf);
        }
        if (v_buf) {
            ggml_backend_buffer_free(v_buf);
        }
        if (mask_buf) {
            ggml_backend_buffer_free(mask_buf);
        }
        if (out_buf) {
            ggml_backend_buffer_free(out_buf);
        }
        ggml_free(ctx);
    };

    if (!q_buf || !k_buf || !v_buf || !out_buf || (mask && !mask_buf)) {
        std::printf("FAIL: buffer allocation failed (n_kv=%d)\n", n_kv);
        ++g_failures;
        cleanup();
        return;
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(ggml_fp16_t));
    if (mask) {
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: graph compute failed (status=%d, n_kv=%d)\n", (int) status, n_kv);
        ++g_failures;
        cleanup();
        return;
    }

    std::vector<float> gpu_output((size_t) D * H_q * ne01, 0.0f);
    ggml_backend_tensor_get(out, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    cleanup();

    // NMSE + gross check -- see the file header "METRIC" note for why this
    // replaced the round-1 per-element abs/rel tolerance.
    compare(label, gpu_output, ref);
}

// Runs the full n_kv sweep against whichever D=512 decode route the CURRENT
// process's environment selects (GGML_SYCL_FA_D512_DECODE_ESIMD, read once
// on first dispatch and latched -- see the TWO-STATE DRIVER file-header
// note). Returns 0/1/LLAMA_TEST_EXIT_SKIP exactly like the old single-state
// main() did; g_failures is process-global and starts at 0 fresh in every
// state's own process, so no reset is needed between states.
static int run_state_in_current_process(const char * state_label) {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP [%s]: no SYCL GPU device available\n", state_label);
        return LLAMA_TEST_EXIT_SKIP;
    }

    // gemma4 GQA shape fallback -- see the file header "GQA SHAPE" note.
    const int D    = 512;
    const int H_q  = 8;
    const int H_kv = 1;

    std::printf("== state %s: GGML_SYCL_FA_D512_DECODE_ESIMD=%s ==\n", state_label,
                std::getenv("GGML_SYCL_FA_D512_DECODE_ESIMD") ? std::getenv("GGML_SYCL_FA_D512_DECODE_ESIMD") :
                                                                "(unset, default ON)");
    // Original decode-shaped (ne01=1), mask-free sweep across the FATTN_KQ_STRIDE axis.
    run_shape(backend, D, H_q, H_kv, 32);
    run_shape(backend, D, H_q, H_kv, 512);
    run_shape(backend, D, H_q, H_kv, 4096);
    // Lead hardware finding llama.cpp-zwsj/c-1ha7: the PRODUCTION decode
    // shape -- ne01=1 WITH a mask (llama always passes a KQ mask to flash
    // attention, even at single-token decode) -- had never been tested. A
    // plain causal mask at ne01=1 sees every kv position (a no-op), so
    // `window` restricts visibility to the last half of n_kv, matching a
    // sliding-window layer and genuinely exercising the mask path. This is
    // the shape the ESIMD tier now engages on in production (ne01==1
    // only, after the finding-A dispatch refusal below).
    run_shape(backend, D, H_q, H_kv, 512, /*ne01=*/1, /*use_mask=*/true, /*window=*/256);
    run_shape(backend, D, H_q, H_kv, 4096, /*ne01=*/1, /*use_mask=*/true, /*window=*/2048);
    // Spec review llama.cpp-zwsj/c-7iey round 1, finding 2 / lead hardware
    // finding c-1ha7, finding A: ne01=4 (speculative-decode-shaped ubatch)
    // WITH a real per-query mask. Round 1 added this to exercise the ESIMD
    // kernel's per-query mask indexing; running it on hardware found
    // launch_fattn_esimd_f16_optimized<512,...> returns garbage for it
    // (finding A -- see the file header, and llama.cpp-wais for the open
    // root-cause follow-up). fattn.cpp's D==512 branch now engages the
    // ESIMD tier at ne01==1 ONLY, so this shape is refused into the tile
    // route REGARDLESS of the toggle -- this case now tests THAT refusal
    // (each toggle state independently agrees with the reference; the two
    // states are not cross-compared here -- see main()'s comment), not
    // the ESIMD tier's multi-query correctness. `window` narrows visibility
    // so the mask is load-bearing (a plain causal window here would mask
    // only 3 of 512 positions per query, per spec review c-ej1v nit 6).
    run_shape(backend, D, H_q, H_kv, 512, /*ne01=*/4, /*use_mask=*/true, /*window=*/128);
    // Spec review llama.cpp-zwsj/c-7iey round 1, finding 4: H_kv=2
    // (gqa_ratio=4, not the degenerate kv_head==0-for-every-head case
    // H_kv=1 gives), so a mis-mapped GQA head in the ESIMD kernel's own
    // gqa_ratio/kv_head computation would be observable. ne01=1 -- the only
    // shape the ESIMD tier engages on after the finding-A narrowing.
    run_shape(backend, D, H_q, /*H_kv=*/2, 512);

    ggml_backend_free(backend);

    if (g_failures) {
        std::printf("FAILED [%s]: %d check(s)\n", state_label, g_failures);
        return 1;
    }
    std::printf("PASS [%s]: D=512 decode flash-attention numerics\n", state_label);
    return 0;
}

// Re-execs self with GGML_SYCL_FA_D512_DECODE_ESIMD set to `value` (before
// any SYCL/backend call in the child -- see the TWO-STATE DRIVER note),
// waits for it, and returns its exit status (or -1 on a fork/exec failure,
// which the caller treats as a hard failure).
static int run_state_in_child(char ** argv, const char * value) {
    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "FATAL: fork() failed for state=%s (%s)\n", value, std::strerror(errno));
        return -1;
    }
    if (pid == 0) {
        setenv("GGML_SYCL_FA_D512_DECODE_ESIMD", value, 1);
        setenv("GGML_SYCL_FA_D512_DECODE_ESIMD_TEST_STATE", value, 1);
        execv("/proc/self/exe", argv);
        std::fprintf(stderr, "FATAL: re-exec failed for state=%s (%s)\n", value, std::strerror(errno));
        _exit(1);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "FATAL: waitpid() failed for state=%s (%s)\n", value, std::strerror(errno));
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

int main(int, char ** argv) {
    // ctest supplies ONEAPI_DEVICE_SELECTOR via the registration's ENVIRONMENT. Bare
    // invocation falls back to the B50, but setenv() here is too late: libccl's static
    // initializer constructs a sycl::event at load, which makes libsycl memoize the
    // selector before main() runs (llama.cpp-2x3m, gdb-traced 2026-09-04). Re-exec so
    // the child starts with the variable set (llama.cpp-403s: unpinned, the iGPU's
    // 231 GB "VRAM" is claimed); it then takes the getenv branch and cannot loop.
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
        execv("/proc/self/exe", argv);
        std::fprintf(stderr, "warning: re-exec failed (%s); continuing unpinned\n", std::strerror(errno));
    }

    // GGML_SYCL_FA_D512_DECODE_ESIMD_TEST_STATE is this file's own sentinel
    // (never read by production code), set only by run_state_in_child()
    // below. Its presence means this process IS one of the two per-state
    // children; its absence means this is the top-level process, which owns
    // spawning both and reporting the combined result.
    const char * state = std::getenv("GGML_SYCL_FA_D512_DECODE_ESIMD_TEST_STATE");
    if (state) {
        return run_state_in_current_process(state);
    }

    // Top-level driver: state "1" (the new ESIMD tier) first, then "0" (the
    // old tile route, Phase 1's only coverage) -- see the TWO-STATE DRIVER
    // file-header note for why this cannot be done with two setenv() calls
    // in one process. If the first child reports no device (SKIP), the
    // second would report the same thing for the same reason, so skip
    // immediately rather than forking a second child that cannot succeed.
    // Each child is scored ONLY against the CPU reference (run_shape's own
    // compare() call) -- the two states' outputs are never cross-compared
    // against each other here, so "both states pass" means "both
    // independently agree with the reference," not "the two states agree
    // with each other" (spec review llama.cpp-zwsj/c-ej1v nit 7).
    const int rc_esimd = run_state_in_child(argv, "1");
    if (rc_esimd == LLAMA_TEST_EXIT_SKIP) {
        std::printf("SKIP: no SYCL GPU device available (state=1 child)\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    const int rc_tile = run_state_in_child(argv, "0");

    if (rc_esimd != 0 || rc_tile != 0) {
        std::printf("FAILED: state=1 (ESIMD) rc=%d, state=0 (tile) rc=%d\n", rc_esimd, rc_tile);
        return 1;
    }
    std::printf("PASS: D=512 decode flash-attention numerics (both GGML_SYCL_FA_D512_DECODE_ESIMD states)\n");
    return 0;
}
