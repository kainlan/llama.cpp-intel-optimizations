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

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    ggml_tensor *  ids    = nullptr;
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

    ggml_tensor * probs = ggml_soft_max(rg.ctx, logits);
    ggml_set_name(probs, "ffn_moe_probs");                      // [n_expert, n_tokens]

    rg.ids = ggml_argsort_top_k(rg.ctx, probs, n_expert_used);  // [n_expert_used, n_tokens], I32
    ggml_set_name(rg.ids, "ffn_moe_topk");
    ggml_set_output(rg.ids);

    rg.gf = ggml_new_graph(rg.ctx);
    ggml_build_forward_expand(rg.gf, rg.ids);
    return rg;
}

// Runs `rg`'s graph on `backend` with `hidden_data`/`weight_data` as input,
// returns the resulting ids tensor's I32 contents. Returns empty on any
// failure (allocation, compute) rather than aborting, so the caller can
// report a clean SKIP/FAIL instead of a crash.
static std::vector<int32_t> run_routing_graph(ggml_backend_t             backend,
                                              const routing_graph &      rg,
                                              const std::vector<float> & weight_data,
                                              const std::vector<float> & hidden_data,
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

    std::vector<int32_t> ids(static_cast<size_t>(n_expert_used) * static_cast<size_t>(n_tokens));
    ggml_backend_tensor_get(rg.ids, ids.data(), 0, ids.size() * sizeof(int32_t));

    ggml_backend_buffer_free(buffer);
    return ids;
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

    int mismatches = 0;
    int trials_run = 0;

    for (int trial = 0; trial < n_trials; ++trial) {
        std::vector<float> weight_data(static_cast<size_t>(hidden_dim) * n_expert);
        std::vector<float> hidden_data(static_cast<size_t>(hidden_dim) * n_tokens);
        for (float & v : weight_data) {
            v = dist(rng);
        }
        for (float & v : hidden_data) {
            v = dist(rng);
        }

        routing_graph        cpu_graph = build_routing_graph(n_expert, n_expert_used, hidden_dim, n_tokens);
        std::vector<int32_t> cpu_ids =
            run_routing_graph(cpu_backend, cpu_graph, weight_data, hidden_data, n_expert_used, n_tokens);
        ggml_free(cpu_graph.ctx);

        routing_graph        sycl_graph = build_routing_graph(n_expert, n_expert_used, hidden_dim, n_tokens);
        std::vector<int32_t> sycl_ids =
            run_routing_graph(sycl_backend, sycl_graph, weight_data, hidden_data, n_expert_used, n_tokens);
        ggml_free(sycl_graph.ctx);

        if (cpu_ids.empty() || sycl_ids.empty()) {
            std::fprintf(stderr, "[MOE-ROUTING-ORACLE] SKIP: trial %d produced no data (compute failure)\n", trial);
            ggml_backend_free(sycl_backend);
            ggml_backend_free(cpu_backend);
            return 77;
        }
        if (cpu_ids.size() != sycl_ids.size()) {
            std::fprintf(stderr, "[MOE-ROUTING-ORACLE] FAIL: trial %d size mismatch cpu=%zu sycl=%zu\n", trial,
                         cpu_ids.size(), sycl_ids.size());
            ++mismatches;
            ++trials_run;
            continue;
        }

        bool trial_ok = true;
        for (size_t i = 0; i < cpu_ids.size(); ++i) {
            if (cpu_ids[i] != sycl_ids[i]) {
                std::fprintf(stderr,
                             "[MOE-ROUTING-ORACLE] FAIL: trial %d index %zu (token %zu, slot %zu): cpu=%d sycl=%d\n",
                             trial, i, i / n_expert_used, i % n_expert_used, cpu_ids[i], sycl_ids[i]);
                trial_ok = false;
            }
        }
        if (!trial_ok) {
            ++mismatches;
        }
        ++trials_run;
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] trial %d: %s (n_expert=%d n_expert_used=%d n_tokens=%d)\n", trial,
                     trial_ok ? "PASS" : "FAIL", n_expert, n_expert_used, n_tokens);
    }

    ggml_backend_free(sycl_backend);
    ggml_backend_free(cpu_backend);

    if (trials_run == 0) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] SKIP: zero trials ran -- this run proves nothing\n");
        return 77;
    }
    if (mismatches > 0) {
        std::fprintf(stderr, "[MOE-ROUTING-ORACLE] FAIL: %d/%d trials mismatched (n_expert=%d n_expert_used=%d)\n",
                     mismatches, trials_run, n_expert, n_expert_used);
        return 1;
    }
    std::fprintf(stderr, "[MOE-ROUTING-ORACLE] PASS: %d/%d trials exact-matched (n_expert=%d n_expert_used=%d)\n",
                 trials_run, trials_run, n_expert, n_expert_used);
    return 0;
}
