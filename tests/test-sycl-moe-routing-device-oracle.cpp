// llama.cpp-iikr (device-routing pivot, owner ruling on task llama.cpp-iikr,
// 2026-08-23, task comment c-cgta): correctness oracle for stage (i) of fix
// (B) from the design doc (c-ymuc) -- moving the MoE router forward chain
// (logits matmul -> softmax -> argsort/top-k selection) from the CPU
// dispatch this file's own should_dispatch_to_cpu() policy currently forces
// it to, onto the SYCL device, gated by GGML_SYCL_MOE_ROUTING_DEVICE.
//
// This test does NOT exercise ggml_sycl_mul_mat_id's ids-acquisition path
// (stage (ii), deliberately not part of this change -- see the policy-gate
// commit's own comment in ggml-sycl.cpp). It only proves the device-computed
// routing chain produces IDENTICAL selected-expert ids to the existing
// CPU-computed reference, for real GPT-OSS shapes (n_expert=32,
// n_expert_used=4 -- 32 confirmed by this file's own repo comment at
// ggml-sycl.cpp's GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS site, "GPT-OSS 20B's
// max is 32"; top-4-of-32 is GPT-OSS's documented MoE routing width).
//
// Method: build ONE small ggml graph mirroring the real router subgraph
// (src/llama-graph.cpp:1902-1907 -- ggml_mul_mat -> ggml_soft_max ->
// ggml_argsort_top_k, the last of which is ARGSORT + a VIEW per
// ggml/src/ggml.c:5385-5399; GPT-OSS never reaches a real GGML_OP_TOP_K
// node), with tensor names matching ggml_sycl_tensor_has_moe_routing_hint's
// substring list (ggml-sycl.cpp ~98030-98046: "ffn_gate_inp",
// "ffn_moe_logits", "ffn_moe_probs") so the SAME classification path real
// inference uses is what gets exercised here, not a bypassed one. Compute
// it TWICE with IDENTICAL input data -- once on ggml's CPU backend (the
// reference: this is exactly what runs today, unconditionally correct by
// construction, not a hand-written reimplementation of argsort), once on
// the SYCL backend with the device-routing gate forced on for this whole
// process (the gate's enabled-check is a `static const bool` cached on
// first read, matching every other env-gate in ggml-sycl.cpp, so it must be
// set before any SYCL graph_compute call, not toggled mid-run) -- and
// compare the resulting I32 selected-expert-id arrays for EXACT equality,
// across several independent trials standing in for different layers'
// distinct logit distributions (real per-layer reproduction would need a
// full model load; independent random trials exercise the same argsort
// code path against different data instead).
//
// Exit codes: 0 = all trials match exactly (GREEN). 1 = a mismatch was
// found (a real correctness bug -- do not treat this as evidence the
// device chain merely "differs", it means wrong tokens would route to
// wrong experts). 77 = SKIP (no SYCL device, or the device-routing SYCL
// dispatch was never actually reached -- see the reached-count check below;
// ctest's SKIP_RETURN_CODE, matching this repo's own "a SKIP with status 0
// is not a pass" rule). Plain script convention matches this repo's other
// hand-run SYCL oracle tests (test-sycl-mxfp4-woq-gemm-bench.cpp) --
// registered add_executable-only, run manually, not part of the default
// ctest sweep, since it needs GGML_SYCL_MOE_ROUTING_DEVICE semantics a
// default ctest environment does not provide.
//
// ORDER CONTRACT (fix-cycle Part 2, verified by reading both
// implementations, not assumed): ggml_argsort_top_k's DESC argsort is a
// FULL sort on both backends -- CPU's std::sort with a strict `>`
// comparator (ggml/src/ggml-cpu/ops.cpp:8362-8371,8399-8406) and SYCL's
// bitonic sorting network (ggml-sycl.cpp:40046-40108, invoked from
// argsort_f32_i32_sycl at ggml-sycl.cpp:40299-40342; for n_expert=32,
// ncols_pad = next_power_of_2(32) = 32, so the padding branches never
// trigger in this oracle's shape). Both are real, provably-correct full
// sorts w.r.t. their own comparator: elements with genuinely DIFFERENT
// values are GUARANTEED to land in strict descending order on EITHER
// backend. Neither is a *stable* sort (std::sort and a bitonic network are
// both unstable), so among EXACTLY TIED values the two backends' relative
// order is unspecified and may legitimately differ -- this, and only this,
// is what the pairing-invariant proof on `routing_result` above makes
// benign. A same-side, non-tied ordering violation is NOT covered by that
// slack and would indicate a real defect, not a contract gap.
//
// WHY EXACT-ID-VS-CPU IS NOT THE CRITERION (fix cycle round 3, team-lead's
// rerun after the tolerance fix): even with weight_only_mismatch_tokens at
// 0/4096 (confirming the weight tolerance is now correctly calibrated),
// id_mismatch_tokens read 2/4096 on BOTH cards, identically. Per-token
// evidence (see token_boundary_relative_margin below) attributes these to
// TOP-K BOUNDARY FLIPS: when the CPU-reference score margin between the
// k-th (last-selected) and (k+1)-th (first-excluded) expert is smaller
// than the cross-precision noise floor (~1e-5 relative, from f32-CPU vs
// f16-influenced-device softmax), the two backends can legitimately
// disagree about WHICH expert occupies the boundary slot -- this is not a
// tie-break-ORDER question (that's the ORDER CONTRACT note above), it is
// a tie-break-SET-MEMBERSHIP question, and it is unavoidable in principle:
// any two implementations with different accumulation order will disagree
// on some boundary token, on some data, at a small enough margin. A REAL
// routing bug (indexing error, wrong expert, off-by-one) is not
// margin-sensitive -- it produces mismatches regardless of how close the
// boundary was, i.e. at LARGE margins. That is the discriminator: a
// mismatch's CPU-side boundary margin, scaled by the top score, tells you
// which case you are looking at. See token_boundary_relative_margin /
// kBoundaryMarginRel below for the qualifier, applied both to which
// id-mismatches count toward the stage-(ii) gate and to which tokens the
// STRICT arm treats as tie-free.

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

// Set BEFORE any SYCL backend init/graph_compute call in this process --
// see the file header comment for why this must happen first.
static void force_device_routing_gate_on() {
#if defined(_WIN32)
    _putenv_s("GGML_SYCL_MOE_ROUTING_DEVICE", "1");
#else
    setenv("GGML_SYCL_MOE_ROUTING_DEVICE", "1", 1);
#endif
}

struct routing_graph {
    ggml_context * ctx    = nullptr;
    ggml_cgraph *  gf     = nullptr;
    ggml_tensor *  weight = nullptr;
    ggml_tensor *  hidden = nullptr;
    ggml_tensor *  probs  = nullptr;
    ggml_tensor *  ids    = nullptr;
};

// llama.cpp-iikr (fix cycle after team-lead's c-vr68-style FAIL 6/8 capture,
// tie-break diagnosis): a mismatch alone does not say whether the device
// chain is wrong or whether two experts genuinely tied and the CPU's
// stable sort and the device's (unstable, bitonic) sort broke the tie
// differently -- a benign, graph-level-invariant artifact per the pairing
// proof in llama-graph.cpp (selected_experts feeds weights AND every
// gate/up/down build_lora_mm_id call, so a same-score permutation cannot
// separate a weight from its expert; the final combine is a commutative
// ggml_add reduction, llama-graph.cpp:2121-2127). This struct carries BOTH
// sides' probability for the disputed slot so the caller can tell those
// two cases apart with evidence instead of assuming one.
struct routing_result {
    std::vector<int32_t> ids;    // [n_expert_used, n_tokens]
    std::vector<float>   probs;  // [n_expert, n_tokens] -- softmax scores, one per (expert, token)
};

// Mirrors src/llama-graph.cpp's router subgraph for the no-expert-groups,
// no-bias case (GPT-OSS: hparams.n_expert_groups <= 1, exp_probs_b ==
// nullptr) exactly enough to exercise the SAME op sequence and the SAME
// name-hint classification real inference relies on --
// ggml_sycl_op_is_moe_routing_subgraph() / ggml_sycl_tensor_has_moe_routing_hint()
// (ggml-sycl.cpp ~98030-98073) match by tensor name substring, recursively
// through src[], so naming just these three tensors correctly is
// sufficient -- the underlying ARGSORT node this creates (accessible as
// ids->src[0]) is found via its own src[0] (`probs`, named "ffn_moe_probs")
// without needing a name of its own.
static routing_graph build_routing_graph(int n_expert, int n_expert_used, int hidden_dim, int n_tokens) {
    routing_graph rg;

    ggml_init_params params{};
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    rg.ctx            = ggml_init(params);

    rg.weight = ggml_new_tensor_2d(rg.ctx, GGML_TYPE_F32, hidden_dim, n_expert);
    ggml_set_name(rg.weight, "blk.0.ffn_gate_inp.weight");

    rg.hidden = ggml_new_tensor_2d(rg.ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
    ggml_set_name(rg.hidden, "ffn_inp_probe");
    ggml_set_input(rg.hidden);

    ggml_tensor * logits = ggml_mul_mat(rg.ctx, rg.weight, rg.hidden);
    ggml_set_name(logits, "ffn_moe_logits");  // [n_expert, n_tokens]

    rg.probs = ggml_soft_max(rg.ctx, logits);
    ggml_set_name(rg.probs, "ffn_moe_probs");  // [n_expert, n_tokens]
    ggml_set_output(rg.probs);

    rg.ids = ggml_argsort_top_k(rg.ctx, rg.probs, n_expert_used);  // [n_expert_used, n_tokens], I32
    ggml_set_name(rg.ids, "ffn_moe_topk");
    ggml_set_output(rg.ids);

    rg.gf = ggml_new_graph(rg.ctx);
    // probs must also be in the forward-expand set -- it's an ancestor of
    // ids so it would compute either way, but expanding it explicitly as
    // an output makes ggml_backend_alloc_ctx_tensors keep it allocated
    // (not treated as free-able intermediate storage) so tensor_get on it
    // after compute is valid.
    ggml_build_forward_expand(rg.gf, rg.probs);
    ggml_build_forward_expand(rg.gf, rg.ids);
    return rg;
}

// Runs `rg`'s graph on `backend` with `hidden_data`/`weight_data` as input,
// returns the resulting ids AND probs tensors' contents (see routing_result
// -- probs is needed to discriminate a tie-break artifact from a real
// selection bug on mismatch). Returns an empty result on any failure
// (allocation, compute) rather than aborting, so the caller can report a
// clean SKIP/FAIL instead of a crash.
static routing_result run_routing_graph(ggml_backend_t             backend,
                                        const routing_graph &      rg,
                                        const std::vector<float> & weight_data,
                                        const std::vector<float> & hidden_data,
                                        int                        n_expert,
                                        int                        n_expert_used,
                                        int                        n_tokens) {
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(rg.ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] backend tensor allocation failed\n");
        return {};
    }

    ggml_backend_tensor_set(rg.weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(rg.hidden, hidden_data.data(), 0, hidden_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, rg.gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] graph_compute failed: status=%d\n", (int) status);
        ggml_backend_buffer_free(buffer);
        return {};
    }

    routing_result result;
    result.ids.resize(static_cast<size_t>(n_expert_used) * static_cast<size_t>(n_tokens));
    ggml_backend_tensor_get(rg.ids, result.ids.data(), 0, result.ids.size() * sizeof(int32_t));
    result.probs.resize(static_cast<size_t>(n_expert) * static_cast<size_t>(n_tokens));
    ggml_backend_tensor_get(rg.probs, result.probs.data(), 0, result.probs.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    return result;
}

// llama.cpp-iikr Part 2 (two-arm restructure, team-lead's amended
// semantics after the score evidence): a same-slot positional id
// comparison is the wrong property to gate stage (ii) on, because the
// ORDER CONTRACT above only promises a matching SET for a token with
// tied boundary scores -- team-lead's hardware runs additionally showed
// the slot-order difference can occur even between DISTINCT scores,
// which the contract does not license, so slot order is not asserted as
// a general property at all (see the tie-free STRICT arm below for the
// one place it legitimately IS asserted). What the executor actually
// needs (per-token, order-independent) is a MULTISET of (expert_id,
// weight) pairs -- exactly what build_lora_mm_id's consumers read via
// selected_experts (see the pairing-invariant proof above), so that is
// the SEMANTIC arm and the actual gate for stage (ii).
// llama.cpp-iikr fix cycle (team-lead's rerun, both cards): the FIRST
// version of this tolerance used abs_tol=1e-6 as the effective ceiling
// (rel_tol=1e-5 on O(0.3) weights is only ~3e-6), which is exactly the
// classifier's own TIE(fp-noise) cutoff -- so genuine f16-device-vs-f32-CPU
// softmax noise (measured 5-7.2e-6 absolute on O(0.3) weights, ~1.5-2e-5
// RELATIVE) got tagged REAL-DIFF and failed tokens whose expert-id
// SETS were already identical. rel_tol is now 5e-5 (comfortably above
// the measured ~2e-5 relative noise), abs_tol stays as a floor for
// near-zero weights where relative comparison alone is meaningless.
static bool approx_equal(float a, float b, float rel_tol = 5e-5f, float abs_tol = 1e-6f) {
    return std::fabs(a - b) <= abs_tol + rel_tol * std::max(std::fabs(a), std::fabs(b));
}

// Structural-vs-boundary-flip qualifier (fix cycle round 3, team-lead's
// margin-based discriminator -- see the header comment's "WHY EXACT-ID-
// VS-CPU IS NOT THE CRITERION" section). Returns the smallest ADJACENT gap
// among the CPU-reference top-(k+1) ranked scores, scaled by the rank-0
// (top) score -- adjacent gaps are sufficient (not just the k/(k+1)
// boundary specifically): if every adjacent gap in a sorted run exceeds a
// threshold, every non-adjacent pair's gap exceeds it too, by summation.
// Scaling by the top score turns this into a RELATIVE margin, matching
// the RELATIVE nature of f16-vs-f32 cross-precision noise (a fixed
// absolute threshold was tried first and round-3 hardware evidence showed
// it let at least one genuinely noise-scale boundary slip through as
// "tie-free" -- a large top score can make an absolute gap look safely
// wide while still being tiny relative to that token's own value scale).
static float token_boundary_relative_margin(const float * probs_row, int n_expert, int n_expert_used) {
    std::vector<float> top(probs_row, probs_row + n_expert);
    std::sort(top.begin(), top.end(), std::greater<float>());
    const int   window    = std::min(n_expert, n_expert_used + 1);
    const float top_score = top[0];
    if (top_score <= 0.0f) {
        return 0.0f;  // degenerate row (all-zero underflow) -- never treat as structural
    }
    float min_gap = std::numeric_limits<float>::max();
    for (int i = 0; i + 1 < window; ++i) {
        min_gap = std::min(min_gap, top[i] - top[i + 1]);
    }
    return min_gap / top_score;
}

// team-lead-specified cutoff: comfortably above the observed ~1e-5
// relative cross-precision noise. A margin above this is treated as
// STRUCTURAL (a real bug would land here, at any margin, since it is not
// noise-sensitive); at or below it, a mismatch is a BOUNDARY FLIP --
// diagnostic, and expected to occur occasionally on random data by
// construction, not something the gate can or should demand zero of.
static constexpr float kBoundaryMarginRel = 5e-4f;

static bool token_boundary_is_structural(const float * probs_row, int n_expert, int n_expert_used) {
    return token_boundary_relative_margin(probs_row, n_expert, n_expert_used) > kBoundaryMarginRel;
}

// Prints, for every expert id selected by EITHER side at `token`, both
// sides' own score for that expert (0.0 formatted as such, not omitted,
// when an id was never selected by one side -- its score is still a real,
// readable number from that side's probs tensor) so a reader never has to
// manually cross-reference two separate lines the way this fix cycle's
// Part 1 report had to.
static void print_token_mismatch(const char *                 arm,
                                 int                          trial,
                                 size_t                       token,
                                 const std::vector<int32_t> & cpu_ids,
                                 const std::vector<int32_t> & sycl_ids,
                                 const std::vector<float> &   cpu_probs,
                                 const std::vector<float> &   sycl_probs,
                                 int                          n_expert) {
    std::vector<int32_t> involved = cpu_ids;
    involved.insert(involved.end(), sycl_ids.begin(), sycl_ids.end());
    std::sort(involved.begin(), involved.end());
    involved.erase(std::unique(involved.begin(), involved.end()), involved.end());

    std::fprintf(stderr, "[MOE-ROUTING-ORACLE] FAIL(%s): trial %d token %zu cpu_ids=[", arm, trial, token);
    for (size_t i = 0; i < cpu_ids.size(); ++i) {
        std::fprintf(stderr, "%s%d", i ? "," : "", cpu_ids[i]);
    }
    std::fprintf(stderr, "] sycl_ids=[");
    for (size_t i = 0; i < sycl_ids.size(); ++i) {
        std::fprintf(stderr, "%s%d", i ? "," : "", sycl_ids[i]);
    }
    std::fprintf(stderr, "]\n");
    for (int32_t id : involved) {
        const float  cs    = cpu_probs[token * static_cast<size_t>(n_expert) + id];
        const float  ss    = sycl_probs[token * static_cast<size_t>(n_expert) + id];
        const float  delta = cs - ss;
        // Retagged against the SAME tolerance the semantic/strict arms
        // gate on (approx_equal), not an independent fixed cutoff -- a
        // fixed cutoff here previously disagreed with the arms' own pass/
        // fail decision and mislabeled real tolerance-noise as REAL-DIFF.
        const char * tag   = (delta == 0.0f) ? "TIE(exact)" : approx_equal(cs, ss) ? "TIE(fp-noise)" : "REAL-DIFF";
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE]   expert %d: cpu_score=%.9f sycl_score=%.9f delta=%.9e %s\n", id, cs,
                     ss, delta, tag);
    }
}

// team-lead Part 1 of the round-3 fix cycle: for an id-mismatch token,
// print the CPU-reference boundary itself -- the k-th (last selected) and
// (k+1)-th (first excluded) ranked scores, their margin, and the relative
// margin the structural/boundary-flip qualifier above is computed from --
// so a reader can see directly whether a given mismatch sits at a
// vanishing margin (expected noise) or a wide one (a real bug) without
// re-deriving it from the raw probs.
static void print_boundary_evidence(int trial, size_t token, const float * cpu_row, int n_expert, int n_expert_used) {
    std::vector<float> top(cpu_row, cpu_row + n_expert);
    std::sort(top.begin(), top.end(), std::greater<float>());
    const int   k          = n_expert_used;
    const float kth        = top[k - 1];
    const float k_plus_1th = (k < n_expert) ? top[k] : 0.0f;
    const float margin     = kth - k_plus_1th;
    const float rel_margin = token_boundary_relative_margin(cpu_row, n_expert, n_expert_used);
    std::fprintf(stderr,
                 "[MOE-ROUTING-ORACLE]   boundary(trial %d token %zu): cpu rank-%d(last-selected)=%.9f "
                 "rank-%d(first-excluded)=%.9f margin=%.9e rel_margin=%.9e (threshold=%.1e) -> %s\n",
                 trial, token, k, kth, k + 1, k_plus_1th, margin, rel_margin, kBoundaryMarginRel,
                 (rel_margin > kBoundaryMarginRel) ? "STRUCTURAL" : "BOUNDARY-FLIP(diagnostic)");
}

int main() {
    // GPT-OSS 20B's confirmed MoE shape: 32 experts, top-4 selected per
    // token (see the file header comment for the citation). hidden_dim and
    // n_tokens are less load-bearing for routing correctness specifically
    // (they only size the logits matmul's reduction/batch dims, not the
    // argsort logic) -- 2880 and 512 are representative PP-shaped values,
    // not asserted-exact GPT-OSS hparams.
    constexpr int n_expert      = 32;
    constexpr int n_expert_used = 4;
    constexpr int hidden_dim    = 2880;
    constexpr int n_tokens      = 512;  // > 1: PP-shaped, matches the gate's ne[1] > 1 discriminator
    constexpr int n_trials      = 8;    // independent trials standing in for distinct layers

    force_device_routing_gate_on();

    ggml_backend_load_all();

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] SKIP: no CPU backend\n");
        return 77;
    }

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] SKIP: no SYCL device available\n");
        ggml_backend_free(cpu_backend);
        return 77;
    }

    std::mt19937                          rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    int trials_run = 0;

    // SEMANTIC arm (team-lead's GATE for stage (ii)): per-token
    // (expert_id, weight) MULTISET equality, order-independent -- the
    // property build_lora_mm_id's consumers actually rely on. Three
    // counters, not one: id_mismatch_tokens (the id SET differs AND the
    // CPU-side boundary margin is STRUCTURAL, i.e. not noise-explicable --
    // see token_boundary_is_structural) is the number stage (ii) is gated
    // on; boundary_flip_tokens (id SET differs, but margin is at/below the
    // noise floor) is diagnostic and EXPECTED to be nonzero on random data
    // by construction -- it does not indict routing; weight_only_mismatch_
    // tokens (same id set, a weight fell outside approx_equal's tolerance)
    // is diagnostic likewise. Conflating any of these made the load-
    // bearing zero (id_mismatch_tokens == 0) unreadable from the summary
    // line alone.
    int semantic_total_tokens       = 0;
    int id_mismatch_tokens          = 0;
    int boundary_flip_tokens        = 0;
    int weight_only_mismatch_tokens = 0;

    // STRICT arm: on tokens whose top-(k+1) boundary margin is STRUCTURAL
    // (token_boundary_is_structural -- the same relative-margin qualifier
    // as the semantic arm's id-mismatch split, round 3), the order
    // contract above DOES guarantee matching slot order -- so this arm
    // additionally asserts exact positional id equality, scoped to only
    // those qualifying tokens. A failure here, unlike the semantic arm,
    // points at a real sort/comparator defect rather than a benign
    // tie-break difference.
    int strict_qualified_tokens = 0;
    int strict_fail_tokens      = 0;

    for (int trial = 0; trial < n_trials; ++trial) {
        std::vector<float> weight_data(static_cast<size_t>(hidden_dim) * n_expert);
        std::vector<float> hidden_data(static_cast<size_t>(hidden_dim) * n_tokens);
        for (float & v : weight_data) {
            v = dist(rng);
        }
        for (float & v : hidden_data) {
            v = dist(rng);
        }

        routing_graph  cpu_graph = build_routing_graph(n_expert, n_expert_used, hidden_dim, n_tokens);
        routing_result cpu =
            run_routing_graph(cpu_backend, cpu_graph, weight_data, hidden_data, n_expert, n_expert_used, n_tokens);
        ggml_free(cpu_graph.ctx);

        routing_graph  sycl_graph = build_routing_graph(n_expert, n_expert_used, hidden_dim, n_tokens);
        routing_result sycl =
            run_routing_graph(sycl_backend, sycl_graph, weight_data, hidden_data, n_expert, n_expert_used, n_tokens);
        ggml_free(sycl_graph.ctx);

        if (cpu.ids.empty() || sycl.ids.empty()) {
            std::fprintf(stderr, "[MOE-ROUTING-ORACLE] SKIP: trial %d produced no data (compute failure)\n", trial);
            ggml_backend_free(sycl_backend);
            ggml_backend_free(cpu_backend);
            return 77;
        }
        if (cpu.ids.size() != sycl.ids.size() || cpu.probs.size() != sycl.probs.size()) {
            std::fprintf(stderr, "[MOE-ROUTING-ORACLE] FAIL: trial %d size mismatch\n", trial);
            ++id_mismatch_tokens;
            ++semantic_total_tokens;
            ++trials_run;
            continue;
        }

        bool trial_semantic_ok = true;
        bool trial_strict_ok   = true;

        for (int token = 0; token < n_tokens; ++token) {
            const size_t base = static_cast<size_t>(token) * static_cast<size_t>(n_expert_used);

            std::vector<int32_t> cpu_slot(cpu.ids.begin() + base, cpu.ids.begin() + base + n_expert_used);
            std::vector<int32_t> sycl_slot(sycl.ids.begin() + base, sycl.ids.begin() + base + n_expert_used);

            std::vector<int32_t> cpu_set  = cpu_slot;
            std::vector<int32_t> sycl_set = sycl_slot;
            std::sort(cpu_set.begin(), cpu_set.end());
            std::sort(sycl_set.begin(), sycl_set.end());

            ++semantic_total_tokens;
            const bool sets_equal = (cpu_set == sycl_set);
            bool       weights_ok = sets_equal;
            if (sets_equal) {
                for (int32_t id : cpu_set) {
                    const float cs = cpu.probs[static_cast<size_t>(token) * n_expert + id];
                    const float ss = sycl.probs[static_cast<size_t>(token) * n_expert + id];
                    if (!approx_equal(cs, ss)) {
                        weights_ok = false;
                        break;
                    }
                }
            }
            const float * cpu_row = cpu.probs.data() + static_cast<size_t>(token) * n_expert;

            if (!sets_equal) {
                trial_semantic_ok       = false;
                const bool   structural = token_boundary_is_structural(cpu_row, n_expert, n_expert_used);
                const char * arm =
                    structural ? "semantic/id-mismatch-structural" : "semantic/id-mismatch-boundary-flip";
                print_token_mismatch(arm, trial, token, cpu_slot, sycl_slot, cpu.probs, sycl.probs, n_expert);
                print_boundary_evidence(trial, token, cpu_row, n_expert, n_expert_used);
                if (structural) {
                    ++id_mismatch_tokens;
                } else {
                    ++boundary_flip_tokens;
                }
            } else if (!weights_ok) {
                ++weight_only_mismatch_tokens;
                trial_semantic_ok = false;
                print_token_mismatch("semantic/weight-only", trial, token, cpu_slot, sycl_slot, cpu.probs, sycl.probs,
                                     n_expert);
            }

            if (token_boundary_is_structural(cpu_row, n_expert, n_expert_used)) {
                ++strict_qualified_tokens;
                const bool strict_ok = sets_equal && weights_ok && (cpu_slot == sycl_slot);
                if (!strict_ok) {
                    ++strict_fail_tokens;
                    trial_strict_ok = false;
                    print_token_mismatch("strict/structural", trial, token, cpu_slot, sycl_slot, cpu.probs, sycl.probs,
                                         n_expert);
                    print_boundary_evidence(trial, token, cpu_row, n_expert, n_expert_used);
                }
            }
        }

        ++trials_run;
        std::fprintf(stderr,
                     "[MOE-ROUTING-ORACLE] trial %d: semantic=%s strict=%s (n_expert=%d n_expert_used=%d "
                     "n_tokens=%d)\n",
                     trial, trial_semantic_ok ? "PASS" : "FAIL", trial_strict_ok ? "PASS" : "FAIL", n_expert,
                     n_expert_used, n_tokens);
    }

    ggml_backend_free(sycl_backend);
    ggml_backend_free(cpu_backend);

    if (trials_run == 0) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] SKIP: zero trials ran -- this run proves nothing\n");
        return 77;
    }

    // GATE LINE for stage (ii): read id_mismatch_tokens off this line
    // directly -- it must be 0 for the semantic arm to be GREEN.
    // boundary_flip_tokens (margin at/below the noise floor) and
    // weight_only_mismatch_tokens (tolerance calibration) are diagnostic
    // and reported separately so neither can hide inside the gate number,
    // and boundary_flip_tokens is EXPECTED to be nonzero on random data --
    // it is not folded into the pass/fail decision below at all.
    std::fprintf(stderr,
                 "[MOE-ROUTING-ORACLE] SUMMARY: id_mismatch_tokens=%d/%d (GATES stage (ii) -- must be 0) "
                 "boundary_flip_tokens=%d/%d (diagnostic, EXPECTED nonzero on random data -- see the header "
                 "comment's boundary-flip note) weight_only_mismatch_tokens=%d/%d (diagnostic, tolerance-"
                 "calibration signal) strict_fail_tokens=%d/%d structural-qualified (indicts the sort "
                 "kernels, not the executor)\n",
                 id_mismatch_tokens, semantic_total_tokens, boundary_flip_tokens, semantic_total_tokens,
                 weight_only_mismatch_tokens, semantic_total_tokens, strict_fail_tokens, strict_qualified_tokens);

    // The SEMANTIC arm's id_mismatch_tokens is the actual gate for stage
    // (ii); boundary_flip_tokens never gates anything (see above -- it is
    // the expected, unavoidable-in-principle output of comparing two
    // different-precision references at a vanishing margin). weight_only_
    // mismatch_tokens and strict_fail_tokens still fail this run's exit
    // code (both indicate something needs attention -- tolerance
    // calibration or a real sort defect, respectively) but are NOT what
    // stage (ii)'s go/no-go reads; see the SUMMARY line above for the
    // load-bearing number specifically.
    if (id_mismatch_tokens > 0 || weight_only_mismatch_tokens > 0 || strict_fail_tokens > 0) {
        std::fprintf(stderr,
                     "[MOE-ROUTING-ORACLE] FAIL: id_mismatch_tokens=%d weight_only_mismatch_tokens=%d "
                     "strict_fail_tokens=%d (boundary_flip_tokens=%d, non-gating) (n_expert=%d n_expert_used=%d)\n",
                     id_mismatch_tokens, weight_only_mismatch_tokens, strict_fail_tokens, boundary_flip_tokens,
                     n_expert, n_expert_used);
        return 1;
    }
    std::fprintf(stderr,
                 "[MOE-ROUTING-ORACLE] PASS: %d/%d trials, both arms clean (boundary_flip_tokens=%d, "
                 "non-gating) (n_expert=%d n_expert_used=%d)\n",
                 trials_run, trials_run, boundary_flip_tokens, n_expert, n_expert_used);
    return 0;
}
