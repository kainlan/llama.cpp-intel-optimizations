// GPU numerics guard for the D=512 flash-attention TILE decode path
// (submit_fattn_tile_d512 / launch_fattn_tile_d512, fattn-tile.hpp) -- the
// "fattn.decode.tile_d512" kernel gemma4's seven D=512 global-attention
// layers run every decode token (llama.cpp-ebxw part 2, llama.cpp-zwsj / plan
// Task P3). Built as a REGRESSION GUARD, not a discovery test: it must PASS
// on the current (un-tiered) kernel first, and stays green across the
// planned decode-shaped split-KV/ESIMD specialisation the spike may add.
//
// SHAPE: this is the exact shape the D=512 route has no alternative for.
// ggml_sycl_flash_attn_ext_supported() (fattn.cpp) admits D=512 only via
// EITHER the oneDNN SDPA route (llama.cpp-jahv, requires ne01 >=
// GGML_SYCL_FA_ONEDNN_MIN_NCOLS, default 8) or this tile route
// (llama.cpp-dtpk, no ne01 floor). At ne01=1 (decode) the oneDNN route is
// structurally unreachable regardless of scale, so every case below --
// mask=nullptr, standard 1/sqrt(D) scale -- resolves through
// launch_fattn_tile_d512() and its ncols2==1 fallback tier
// (submit_fattn_tile_d512<512,512,2,1,...> at ne01=1, fattn-tile.hpp). This
// mirrors test-sycl-fattn-onednn-gates.cpp's own
// build_d512_tile_flash_attn_ext_op()/test_supports_op_admits_d512_tile_decode_shape
// admission case (same file, ne01_q=1) -- that test proves the op is
// ADMITTED into this route; this file proves the route's OUTPUT is correct.
//
// mask is intentionally omitted (src[3] = nullptr): at decode every prior KV
// position is valid for the new token, so a real KQ mask here would be all
// zeros -- functionally identical to no mask, and omitting it keeps this
// file state-independent of GGML_SYCL_FLASH_ATTN_EXT's use_gqa_opt branch
// (which requires a mask; see launch_fattn_tile_d512's own comment on why
// DV==512 makes ne01 irrelevant to that branch's own gqa_ratio<=4 gate).
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
// TOLERANCE: `1e-3 + 1e-3*|ref|` per element, per this task's own acceptance
// criterion -- generous relative to test-sycl-mmvq-q8-0-soa-numerics.cpp's
// `1e-3/1e-2` precedent for a similarly reduced-precision GPU accumulation,
// because flash_attn_tile<>'s VKQ accumulator is sycl::half2 under
// SYCL_FAST_FP16 (always true in this build; CLAUDE.md's GGML_SYCL_F16=ON
// note), not f32 -- a whole-context weighted sum of V accumulated in f16 at
// n_kv=4096 is expected to carry more rounding error than a single Q8_0 x
// Q8_1 dot product does, and the ticket's own acceptance line spells out
// this tolerance rather than leaving it to be re-derived.
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

// Double-precision softmax(scale * Q K^T) V, per head, MQA/GQA-aware via
// kv_h = h / (H_q/H_kv) (matches the production kv-head mapping; degenerate
// to kv_h=0 for every h at H_kv=1, this file's shape). Q is D*H_q floats
// (ne=[D,1,H_q,1] contiguous, so flat index d + D*h); K/V are D*n_kv*H_kv
// half-precision elements (ne=[D,n_kv,H_kv,1] contiguous, flat index
// d + D*(t + n_kv*kv_h)); output is D*H_q floats in the SAME [d + D*h]
// layout ggml_flash_attn_ext's own dst ne={DV,H_q,ne01,ne03} produces at
// ne01=ne03=1.
static void compute_reference(const std::vector<float> &       Q,
                              const std::vector<ggml_fp16_t> & K,
                              const std::vector<ggml_fp16_t> & V,
                              int                              D,
                              int                              H_q,
                              int                              H_kv,
                              int                              n_kv,
                              float                            scale,
                              std::vector<float> &             out) {
    out.assign((size_t) D * H_q, 0.0f);
    const int           n_rep = H_q / H_kv;
    std::vector<double> logits((size_t) n_kv);

    for (int h = 0; h < H_q; ++h) {
        const int kv_h = h / n_rep;

        double max_logit = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < n_kv; ++t) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) {
                const double q = (double) Q[(size_t) h * D + d];
                const double k = (double) ggml_fp16_to_fp32(K[(size_t) kv_h * n_kv * D + (size_t) t * D + d]);
                dot += q * k;
            }
            const double logit = dot * (double) scale;
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
            out[(size_t) h * D + d] = (float) (denom > 0.0 ? acc / denom : 0.0);
        }
    }
}

// Per-element |got-ref| <= abs_tol + rel_tol*|ref| with a violation counter,
// matching test-sycl-mmvq-q8-0-soa-numerics.cpp's compare() convention.
static void compare(const char *               what,
                    const std::vector<float> & got,
                    const std::vector<float> & ref,
                    float                      rel_tol,
                    float                      abs_tol) {
    if (got.empty() || ref.empty() || got.size() != ref.size()) {
        std::printf("FAIL [%s]: size mismatch or missing output (got=%zu ref=%zu)\n", what, got.size(), ref.size());
        ++g_failures;
        return;
    }
    float  max_diff   = 0.0f;
    float  max_rel    = 0.0f;
    size_t violations = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        max_diff         = std::max(max_diff, diff);
        const float rel  = std::fabs(ref[i]) > 1e-6f ? diff / std::fabs(ref[i]) : diff;
        max_rel          = std::max(max_rel, rel);
        if (diff > abs_tol + rel_tol * std::fabs(ref[i])) {
            ++violations;
        }
    }
    const bool ok = violations == 0;
    std::printf("%s [%s]: max_diff=%.6e max_rel=%.6e violations=%zu/%zu (tol rel=%.1e abs=%.1e)\n", ok ? "OK" : "FAIL",
                what, max_diff, max_rel, violations, got.size(), rel_tol, abs_tol);
    if (!ok) {
        ++g_failures;
    }
}

static void run_shape(ggml_backend_t backend, int D, int H_q, int H_kv, int n_kv) {
    std::printf("== shape D=%d H_q=%d H_kv=%d n_kv=%d (decode, ne01=1) ==\n", D, H_q, H_kv, n_kv);

    std::mt19937 rng(0xd512u ^ ((unsigned) n_kv * 2654435761u) ^ ((unsigned) H_q * 40503u));

    std::vector<float>       q_data;
    std::vector<ggml_fp16_t> k_data;
    std::vector<ggml_fp16_t> v_data;
    // Small magnitude keeps the D=512-wide QK dot products (and therefore
    // the softmax logits) in a numerically comfortable range -- these are
    // synthetic activations, not real model weights, so nothing about their
    // distribution needs to match gemma4's actual statistics for this to be
    // a valid correctness check.
    fill_f32_random(q_data, (int64_t) D * H_q, rng, -0.1f, 0.1f);
    fill_f16_random(k_data, (int64_t) D * n_kv * H_kv, rng, -0.1f, 0.1f);
    fill_f16_random(v_data, (int64_t) D * n_kv * H_kv, rng, -0.1f, 0.1f);

    const float scale = 1.0f / std::sqrt((float) D);

    std::vector<float> ref;
    compute_reference(q_data, k_data, v_data, D, H_q, H_kv, n_kv, scale, ref);

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL: ggml_init failed\n");
        ++g_failures;
        return;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, H_q, 1);
    ggml_set_name(q, "fattn_d512_decode_q");
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H_kv, 1);
    ggml_set_name(k, "fattn_d512_decode_k");
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, n_kv, H_kv, 1);
    ggml_set_name(v, "fattn_d512_decode_v");

    // No mask (src[3] = nullptr): at decode every prior KV position is
    // valid, so a real causal mask here would be all-zero -- see the file
    // header for why this is the representative shape, not a shortcut.
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, scale, /*max_bias=*/0.0f,
                                            /*logit_softcap=*/0.0f);
    ggml_set_name(out, "fattn_d512_decode_out");

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);

    ggml_backend_buffer_t q_buf   = alloc_tensor_buffer(dev_buft, q, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t k_buf   = alloc_tensor_buffer(dev_buft, k, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t v_buf   = alloc_tensor_buffer(dev_buft, v, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
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
        if (out_buf) {
            ggml_backend_buffer_free(out_buf);
        }
        ggml_free(ctx);
    };

    if (!q_buf || !k_buf || !v_buf || !out_buf) {
        std::printf("FAIL: buffer allocation failed (n_kv=%d)\n", n_kv);
        ++g_failures;
        cleanup();
        return;
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(ggml_fp16_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: graph compute failed (status=%d, n_kv=%d)\n", (int) status, n_kv);
        ++g_failures;
        cleanup();
        return;
    }

    std::vector<float> gpu_output((size_t) D * H_q, 0.0f);
    ggml_backend_tensor_get(out, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    cleanup();

    // 1e-3 + 1e-3*|ref| per element -- this task's own acceptance criterion;
    // see the file header for why it is looser than the Q8_0 MMVQ precedent.
    char label[64];
    std::snprintf(label, sizeof(label), "tile_d512-vs-cpu-reference n_kv=%d", n_kv);
    compare(label, gpu_output, ref, /*rel_tol=*/1e-3f, /*abs_tol=*/1e-3f);
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
    run_shape(backend, D, H_q, H_kv, 32);
    run_shape(backend, D, H_q, H_kv, 512);
    run_shape(backend, D, H_q, H_kv, 4096);

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
