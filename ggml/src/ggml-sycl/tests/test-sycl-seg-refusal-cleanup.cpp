//
// A refusal inside the segmented MoE record fails the graph, and graph_compute's fallback catch leaves nothing behind
// (docs/design/sycl-mmap-host-tier.md §4.4 "A refusal fails the graph on every path", "The catch's cleanup step",
// §7 C4 test-sycl-seg-refusal-cleanup).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The fixture is a hand-built MoE model, no libllama: two Q8_0 expert tensors in a WEIGHTS buffer of the card's device
// buffer type, registered and filled inside a load transaction that stages no placement plan, and a bound execution
// context. The model is unplanned because the non-keyed record function is reached only by a prompt graph, and
// graph_compute runs a prompt graph holding a MUL_MAT_ID directly whenever a placement plan is active. Decode never
// reaches it: a decode split records through its keyed slot. The process holds the AoS layout override, so the prompt
// admission publishes each operand's experts as AoS views of the device buffer (the canonical publication), the record
// its resolver needs when no preload published one. The graph,
//
//   x -> a1 a2 a3 -> reshape -> m1 = MUL_MAT_ID -> b1 b2 -> m2 = MUL_MAT_ID -> c1 c2 c3 [c4]
//
// runs at four tokens, so every call is a prompt phase and no decode admission runs. The record function finds two
// dispatch boundaries and three segments: [a1..reshape] recorded with moe_after m1, [b1 b2] direct (two nodes, below
// MIN_SEGMENT_NODES) with moe_after m2, and [c1..] recorded. Every case first runs the graph twice, the prompt warmup
// and a second call, which must leave moe_graph_rerecord set: the second call either records the whole graph, which
// pauses at m1, or already records segments, when the warmup's MoE dispatch asked for selective re-record. The
// context's recorded state is then invalidated, so the next call records segments through the non-keyed record
// function. That call is the one each case perturbs.
//
// The refusal is a seam in ggml-sycl.cpp: at the armed node and helper role it snapshots the context, then refuses
// through ggml_sycl::host_dispatch_fail_closed, the one function a real refusal throws from. Five cases:
//   direct          the refusal at b1, a direct segment's node;
//   recording       at c2, a recording node inside begin_recording;
//   active-valid    at c2, with active_exec_graph.valid set alone before the call (exec_graph null);
//   moe-after       at m1, the moe_after of the recorded first segment, whose executable was just submitted;
//   stale-vectors   no refusal: a first record, then a prompt graph with one more node, recorded while every
//                   retire fails, so the record function meets a context that still holds segments.
// The decode keyed recorder is not covered: its refusal reaches graph_compute through its own slot catch and the
// keyed record's drain-and-rethrow catch, not through this cleanup.
//
// A case's preconditions decide whether it reached its mechanism; a failed one is VOID (77), never a pass. Its
// assertions run in the design's order and the case stops at the first failure, printing its message.

#include "../ggml-sycl-test.hpp"
#include "file-mapped-test-common.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ggml_sycl;
using host_ledger_test_harness::log_lines_containing;
using scratch_return_test::checks;
using scratch_return_test::lines;

namespace {

constexpr int64_t k_embd     = 256;  // rows and columns of every expert: the chain feeds m1's output to m2
constexpr int64_t k_experts  = 4;
constexpr int64_t k_used     = 2;
constexpr int     k_prompt_n = 4;  // inside the fused MoE kernel's recordable batch (2..32 tokens)

constexpr const char * k_recoverable = "[SYCL] recoverable runtime fallback failed";

std::string status_name(ggml_status s) {
    switch (s) {
        case GGML_STATUS_SUCCESS:
            return "GGML_STATUS_SUCCESS";
        case GGML_STATUS_FAILED:
            return "GGML_STATUS_FAILED";
        case GGML_STATUS_ALLOC_FAILED:
            return "GGML_STATUS_ALLOC_FAILED";
        case GGML_STATUS_ABORTED:
            return "GGML_STATUS_ABORTED";
    }
    return "status " + std::to_string(static_cast<int>(s));
}

// Real expert names, so every name-classified path (the MoE registry, the expert tables) treats them as experts.
const char * const k_names[2] = { "blk.0.ffn_gate_exps.weight", "blk.1.ffn_gate_exps.weight" };

// The model: a backend on the pinned card and the two expert tensors, registered inside a load transaction so the cache
// knows them as weights. The process ends with the case, so nothing is torn down.
struct model_fixture {
    ggml_backend_t            backend = nullptr;
    ggml_sycl_load_txn        load{};
    ggml_sycl_model_token     model{};
    ggml_sycl_exec_context_id context{};
    ggml_context *            weight_ctx    = nullptr;
    ggml_backend_buffer_t     weight_buffer = nullptr;
    ggml_tensor *             w[2]          = {};
};

bool model_open(checks & c, model_fixture & f) {
    // Held for the process: the case ends with it.
    test_set_layout_override(GGML_LAYOUT_AOS);
    if (ggml_backend_sycl_model_load_begin(&f.load) != GGML_SYCL_LIFECYCLE_OK) {
        c.void_case("ggml_backend_sycl_model_load_begin failed");
        return false;
    }
    f.backend = ggml_backend_sycl_init(0);
    if (f.backend == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return false;
    }
    const ggml_init_params params = { 4 * ggml_tensor_overhead(), nullptr, true };
    f.weight_ctx                  = ggml_init(params);
    if (f.weight_ctx == nullptr) {
        c.void_case("ggml_init for the weights failed");
        return false;
    }
    const char * split_path = "test-sycl-seg-refusal-cleanup.gguf";
    for (int i = 0; i < 2; ++i) {
        f.w[i] = ggml_new_tensor_3d(f.weight_ctx, GGML_TYPE_Q8_0, k_embd, k_embd, k_experts);
        ggml_set_name(f.w[i], k_names[i]);
        const size_t nbytes = ggml_nbytes(f.w[i]);
        ggml_backend_sycl_register_gguf_file_identity(0, split_path, uint64_t(2 * nbytes));
        ggml_backend_sycl_register_weight_identity(f.w[i], 0, size_t(i) * nbytes, nbytes, 0);
        if (!ggml_backend_sycl_try_register_weight_usage(k_names[i], GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT)) {
            c.void_case(std::string("usage registration refused for ") + k_names[i]);
            return false;
        }
    }
    f.weight_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(f.weight_ctx, ggml_backend_get_default_buffer_type(f.backend));
    if (f.weight_buffer == nullptr) {
        c.void_case("the weight buffer allocation failed");
        return false;
    }
    ggml_backend_buffer_set_usage(f.weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> rows(size_t(k_embd * k_embd * k_experts));
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = 0.01f * static_cast<float>(static_cast<int>(i % 23) - 11);
    }
    std::vector<uint8_t> q(ggml_nbytes(f.w[0]));
    ggml_quantize_chunk(GGML_TYPE_Q8_0, rows.data(), q.data(), 0, k_embd * k_experts, k_embd, nullptr);
    for (ggml_tensor * t : f.w) {
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
    if (ggml_backend_sycl_model_load_end(f.load, true, &f.model) != GGML_SYCL_LIFECYCLE_OK) {
        c.void_case("ggml_backend_sycl_model_load_end did not commit the load");
        return false;
    }
    if (ggml_backend_sycl_execution_context_create(&f.context) != GGML_SYCL_EXECUTION_OK ||
        ggml_backend_sycl_execution_context_bind_backend(f.backend, f.context) != GGML_SYCL_EXECUTION_OK) {
        c.void_case("the execution context could not be created and bound");
        return false;
    }
    return true;
}

// One graph of the fixture's shape at n_tokens, its tensors in one compute buffer on the card.
struct moe_graph {
    ggml_context *        ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph *         graph  = nullptr;
    ggml_tensor *         m1     = nullptr;
};

ggml_tensor * scaled(ggml_context * ctx, ggml_tensor * src, const char * name) {
    ggml_tensor * t = ggml_scale(ctx, src, 1.0f);
    ggml_set_name(t, name);
    return t;
}

bool graph_build(checks & c, moe_graph & g, const model_fixture & f, int n_tokens, bool extra_tail) {
    const ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx                         = ggml_init(params);
    if (g.ctx == nullptr) {
        c.void_case("ggml_init for a graph failed");
        return false;
    }
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k_embd, n_tokens);
    ggml_set_name(x, "seg.x");
    ggml_set_input(x);
    ggml_tensor * ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, k_used, n_tokens);
    ggml_set_name(ids, "seg.ids");
    ggml_set_input(ids);

    ggml_tensor * a = scaled(g.ctx, scaled(g.ctx, scaled(g.ctx, x, "seg.a1"), "seg.a2"), "seg.a3");
    ggml_tensor * r = ggml_reshape_3d(g.ctx, a, k_embd, 1, n_tokens);
    ggml_set_name(r, "seg.r");
    g.m1 = ggml_mul_mat_id(g.ctx, f.w[0], r, ids);
    ggml_set_name(g.m1, "seg.m1");
    ggml_tensor * b  = scaled(g.ctx, scaled(g.ctx, g.m1, "seg.b1"), "seg.b2");
    ggml_tensor * m2 = ggml_mul_mat_id(g.ctx, f.w[1], b, ids);
    ggml_set_name(m2, "seg.m2");
    ggml_tensor * out = scaled(g.ctx, scaled(g.ctx, scaled(g.ctx, m2, "seg.c1"), "seg.c2"), "seg.c3");
    if (extra_tail) {
        out = scaled(g.ctx, out, "seg.c4");
    }
    ggml_set_output(out);
    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, out);

    g.buffer = ggml_backend_alloc_ctx_tensors(g.ctx, f.backend);
    if (g.buffer == nullptr) {
        c.void_case("a graph's compute buffer allocation failed");
        return false;
    }
    ggml_backend_buffer_set_usage(g.buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    std::vector<float> xs(size_t(k_embd * n_tokens));
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = 0.02f * static_cast<float>(static_cast<int>(i % 17) - 8);
    }
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
    std::vector<int32_t> picks(size_t(k_used * n_tokens));
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < k_used; ++u) {
            picks[size_t(t * k_used + u)] = (t + u) % k_experts;
        }
    }
    ggml_backend_tensor_set(ids, picks.data(), 0, picks.size() * sizeof(int32_t));
    return true;
}

int node_index(const moe_graph & g, const char * name) {
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
        if (std::strcmp(ggml_graph_node(g.graph, i)->name, name) == 0) {
            return i;
        }
    }
    return -1;
}

ggml_sycl_test_seg_state state_of(const model_fixture & f) {
    ggml_sycl_test_seg_state s{};
    (void) ggml_backend_sycl_test_seg_state_read(f.backend, &s);
    return s;
}

ggml_sycl_test_seg_seam seam_report() {
    ggml_sycl_test_seg_seam r{};
    ggml_backend_sycl_test_seg_seam_read(&r);
    return r;
}

std::vector<const void *> ids_of(const void * const * ids, size_t n) {
    return std::vector<const void *>(ids, ids + n);
}

// The common prologue: the prompt graph's warmup and a second call, which must leave moe_graph_rerecord set, then the
// invalidation of whatever the second call recorded. After it, the next call of `prompt` records segments.
bool prologue(checks & c, model_fixture & f, moe_graph & prompt) {
    if (!file_mapped_test::entry_ok(c) || !model_open(c, f) || !graph_build(c, prompt, f, k_prompt_n, false)) {
        return false;
    }
    const char * pass_names[2] = { "the prompt warmup", "the second prompt call" };
    for (const char * pass : pass_names) {
        const ggml_status s = ggml_backend_graph_compute(f.backend, prompt.graph);
        if (s != GGML_STATUS_SUCCESS) {
            c.void_case(std::string(pass) + " returned " + status_name(s));
            return false;
        }
    }
    if (!state_of(f).rerecord) {
        c.void_case(
            "the prompt warmup and second call did not set moe_graph_rerecord: the next call would not "
            "reach the segmented record");
        return false;
    }
    // A no-op when nothing was recorded. It keeps moe_graph_rerecord and the prompt warmup.
    ggml_backend_sycl_graph_invalidate(f.backend, "test-sycl-seg-refusal-cleanup prologue");
    const ggml_sycl_test_seg_state s = state_of(f);
    if (s.n_segments != 0 || s.exec_graph || !s.rerecord) {
        c.void_case("the prologue's invalidation left " + std::to_string(s.n_segments) + " segment(s)" +
                    (s.exec_graph ? " and an exec_graph" : "") + (s.rerecord ? "" : " and cleared moe_graph_rerecord") +
                    ": the next call would not record");
        return false;
    }
    return true;
}

// The refusal's own precondition: the seam fired, and the refused call had pinned at least one lease, without which
// assertion 4 passes vacuously.
bool refusal_reached(checks & c, const ggml_sycl_test_seg_seam & r, const char * node, const char * role) {
    if (!r.fired) {
        c.void_case(std::string("the seam did not fire at ") + node + " (" + role + ")");
        return false;
    }
    if (r.weight_leases + r.moe_leases == 0) {
        c.void_case("the refused call pinned no lease in either vector: assertion 4 would pass vacuously");
        return false;
    }
    return true;
}

// Assertions 2 to 5, in the design's order; false at the first failure, with its message printed.
bool assertions_2_to_5(checks &                        c,
                       model_fixture &                 f,
                       moe_graph &                     prompt,
                       ggml_status                     status,
                       const ggml_sycl_test_seg_seam & r) {
    if (status != GGML_STATUS_FAILED) {
        c.fail("status " + status_name(status) + ", expected GGML_STATUS_FAILED");
        return false;
    }
    const ggml_sycl_test_seg_state s = state_of(f);
    if (s.recording) {
        c.fail("recording flag still set after a FAILED return");
        return false;
    }
    if (s.graph_ptr || s.queue_ptr) {
        c.fail(std::string(s.graph_ptr ? "g_recording_graph_ptr" : "g_recording_queue_ptr") +
               " non-null after a FAILED return");
        return false;
    }
    if (s.sink) {
        c.fail("the retained-handle sink still attached after a FAILED return");
        return false;
    }
    if (s.queue_recording) {
        c.fail("the queue still recording after a FAILED return");
        return false;
    }
    if (s.weight_leases + s.moe_leases != 0) {
        c.fail("lease vectors " + std::to_string(r.weight_leases) + "+" + std::to_string(r.moe_leases) +
               " non-empty after a FAILED return" + (s.active_valid ? ", active_exec_graph live" : "") + " (now " +
               std::to_string(s.weight_leases) + "+" + std::to_string(s.moe_leases) + ")");
        return false;
    }
    if (s.exec_graph || s.active_valid) {
        c.fail(std::string(s.exec_graph ? "exec_graph" : "active_exec_graph") + " live after a FAILED return");
        return false;
    }
    ggml_backend_sycl_test_seg_seam_arm(nullptr, nullptr, false);
    const ggml_status again = ggml_backend_graph_compute(f.backend, prompt.graph);
    if (again != GGML_STATUS_SUCCESS) {
        c.fail("a second graph_compute on the context, with the seam off, returned " + status_name(again));
        return false;
    }
    return true;
}

bool assertion_1(checks & c) {
    if (lines(k_recoverable) == 0) {
        c.fail("no \"recoverable runtime fallback failed\" line after a FAILED return");
        return false;
    }
    return true;
}

// direct, recording and active-valid: one refusal, its preconditions, then assertions 1 to 5.
int run_refusal(checks & c, const char * node, const char * role, bool active_valid_alone) {
    model_fixture f;
    moe_graph     prompt;
    if (!prologue(c, f, prompt)) {
        return c.status();
    }
    const int node_idx = node_index(prompt, node);
    ggml_backend_sycl_test_seg_seam_arm(node, role, false);
    // The whole-graph record left an executable: drop it through the context's own invalidation, outside
    // graph_compute, so the flag can be set with exec_graph null.
    if (active_valid_alone) {
        ggml_backend_sycl_graph_invalidate(f.backend, "test-sycl-seg-refusal-cleanup active-valid");
    }
    if (active_valid_alone && !ggml_backend_sycl_test_seg_set_active_valid_alone(f.backend)) {
        c.void_case("exec_graph was set before the call: active_exec_graph.valid cannot be set alone");
        return c.status();
    }
    const ggml_status             status = ggml_backend_graph_compute(f.backend, prompt.graph);
    const ggml_sycl_test_seg_seam r      = seam_report();
    if (!refusal_reached(c, r, node, role)) {
        return c.status();
    }
    if (std::strcmp(role, "recording") == 0) {
        // The reach witness: without it a segment that direct-dispatches would make this a second direct case.
        const std::string at = "recording failed at node " + std::to_string(node_idx) + " op=SCALE name=" + node;
        if (lines("[SYCL-SEG] Segment ", at) == 0) {
            c.void_case("no \"[SYCL-SEG] Segment ... " + at + "\" WARN: the refusal did not happen inside a recording");
            return c.status();
        }
    }
    if (active_valid_alone && !r.active_valid) {
        c.void_case("active_exec_graph.valid was cleared before the refusal: the case cannot reach its cleanup");
        return c.status();
    }
    fprintf(stderr, "[SEG-REFUSAL] %s refused at %s (node %d): leases %zu+%zu exec_graph=%d active_valid=%d\n", c.name,
            node, node_idx, r.weight_leases, r.moe_leases, r.exec_graph ? 1 : 0, r.active_valid ? 1 : 0);
    if (!assertion_1(c)) {
        return c.status();
    }
    (void) assertions_2_to_5(c, f, prompt, status, r);
    return c.status();
}

// moe-after: the refusal at the first segment's moe_after, just after that segment's executable was submitted.
int run_moe_after(checks & c) {
    model_fixture f;
    moe_graph     prompt;
    if (!prologue(c, f, prompt)) {
        return c.status();
    }
    const int moe_idx = node_index(prompt, "seg.m1");
    ggml_backend_sycl_test_seg_seam_arm("seg.m1", "moe_after of a recorded segment", true);
    const ggml_status             status = ggml_backend_graph_compute(f.backend, prompt.graph);
    const ggml_sycl_test_seg_seam r      = seam_report();
    if (!refusal_reached(c, r, "seg.m1", "moe_after of a recorded segment")) {
        return c.status();
    }
    // (a) exactly this segment's executable is the one newly submitted.
    if (!r.noted || r.moe_after != moe_idx || r.seg_exec == nullptr || !r.submitted || r.n_recorded_ids == 0 ||
        r.recorded_ids[r.n_recorded_ids - 1] != r.seg_exec) {
        c.void_case("the refusal was not at the moe_after of a segment whose executable was just submitted: noted=" +
                    std::to_string(r.noted) + " moe_after=" + std::to_string(r.moe_after) + " (m1 is node " +
                    std::to_string(moe_idx) + ") exec=" + std::to_string(r.seg_exec != nullptr) +
                    " submitted=" + std::to_string(r.submitted));
        return c.status();
    }
    // (b) the inner catch's WARN for that segment names its last recorded node, not the moe node.
    const std::string seg_at = "[SYCL-SEG] Segment " + std::to_string(r.seg_idx) + " recording failed at node " +
                               std::to_string(r.last_recorded_node) + " ";
    if (r.last_recorded_node == moe_idx || lines(seg_at.c_str()) == 0) {
        c.void_case("no \"" + seg_at + "\" WARN naming the segment's last recorded node");
        return c.status();
    }
    fprintf(stderr, "[SEG-REFUSAL] moe-after refused at node %d after segment %zu: %zu recorded, leases %zu+%zu\n",
            moe_idx, r.seg_idx, r.n_recorded, r.weight_leases, r.moe_leases);
    if (!assertion_1(c)) {
        return c.status();
    }
    // E, before 2-5: a destroyed executable could crash the second compute before any message.
    if (r.s1 != static_cast<long>(r.n_recorded) ||
        ids_of(r.s1_ids, r.n_s1_ids) != ids_of(r.recorded_ids, r.n_recorded_ids)) {
        c.fail("executables destroyed in the unwind: moe_segments " + std::to_string(r.s1) + " at catch entry, " +
               std::to_string(r.n_recorded) + " recorded at the throw");
        return c.status();
    }
    if (r.s2 != r.s1) {
        c.fail("executables destroyed before the waits: moe_segments " + std::to_string(r.s1) + " at catch entry, " +
               std::to_string(r.s2) + " at the cleanup step");
        return c.status();
    }
    if (r.s3 != 0) {
        c.fail("the cleanup step left moe_segments " + std::to_string(r.s3) + " at its exit, expected 0");
        return c.status();
    }
    (void) assertions_2_to_5(c, f, prompt, status, r);
    return c.status();
}

// stale-vectors: the record function's entry guard, met with every retire failing.
int run_stale_vectors(checks & c) {
    model_fixture f;
    moe_graph     prompt;
    moe_graph     longer;
    if (!prologue(c, f, prompt) || !graph_build(c, longer, f, k_prompt_n, true)) {
        return c.status();
    }
    ggml_backend_sycl_test_seg_seam_arm(nullptr, nullptr, false);
    const ggml_status              first    = ggml_backend_graph_compute(f.backend, prompt.graph);
    const ggml_sycl_test_seg_state recorded = state_of(f);
    if (first != GGML_STATUS_SUCCESS || recorded.n_ids == 0) {
        c.void_case("the first record returned " + status_name(first) + " with " + std::to_string(recorded.n_ids) +
                    " executable(s)");
        return c.status();
    }
    // The longer graph's warmup, so its next call misses segments_match on n_nodes and reaches the record path.
    const ggml_status warm = ggml_backend_graph_compute(f.backend, longer.graph);
    if (warm != GGML_STATUS_SUCCESS) {
        c.void_case("the longer graph's warmup returned " + status_name(warm));
        return c.status();
    }
    const ggml_sycl_test_seg_state before = state_of(f);
    if (before.optional_layout_epoch != before.global_optional_layout_epoch) {
        c.void_case("the optional-layout epochs differ (" + std::to_string(before.optional_layout_epoch) + " vs " +
                    std::to_string(before.global_optional_layout_epoch) +
                    "): the call would clear the segments before the record");
        return c.status();
    }
    const size_t                    lines_before = lines("[SYCL-SEG] record refused");
    const size_t                    k            = before.n_segments;
    const std::vector<const void *> kept         = ids_of(before.ids, before.n_ids);
    ggml_backend_sycl_test_seg_seam_arm(nullptr, nullptr, false);
    ggml_backend_sycl_test_seg_retire_fail(true);
    const ggml_status              status = ggml_backend_graph_compute(f.backend, longer.graph);
    const ggml_sycl_test_seg_seam  r      = seam_report();
    const ggml_sycl_test_seg_state after  = state_of(f);
    ggml_backend_sycl_test_seg_retire_fail(false);  // before the context's destruction

    if (r.record_entries == 0) {
        c.void_case("the call did not reach the record function (segments_match hit, or the record path declined)");
        return c.status();
    }
    if (r.retire_hits < 2 || r.entry_segments == 0 || !r.entry_disabled || ids_of(r.entry_ids, r.n_entry_ids) != kept) {
        c.void_case("the record function's entry did not meet the kept segments: retires failed " +
                    std::to_string(r.retire_hits) + ", segments " + std::to_string(r.entry_segments) +
                    ", disabled=" + std::to_string(r.entry_disabled));
        return c.status();
    }
    fprintf(stderr, "[SEG-REFUSAL] stale-vectors: %zu segment(s), %zu executable(s) kept, %d failed retire(s)\n", k,
            kept.size(), r.retire_hits);
    const std::vector<std::string> refused = log_lines_containing("[SYCL-SEG] record refused");
    if (refused.size() == lines_before) {
        c.fail("no \"record refused\" line while the context holds " + std::to_string(k) + " segment(s)");
        return c.status();
    }
    const std::string quoted = "still holds " + std::to_string(k) + " segment(s)";
    if (refused.size() != lines_before + 1 || refused.back().find(quoted) == std::string::npos) {
        c.fail("the \"record refused\" line printed " + std::to_string(refused.size() - lines_before) +
               " time(s) or does not quote " + std::to_string(k) + " segment(s): " + refused.back());
        return c.status();
    }
    if (status != GGML_STATUS_SUCCESS || lines(k_recoverable) != 0) {
        c.fail("the refused record did not return through the caller's fallback: " + status_name(status));
        return c.status();
    }
    if (ids_of(after.ids, after.n_ids) != kept) {
        c.fail("the kept executables changed while the retirement was refused: " + std::to_string(after.n_ids) +
               " held, " + std::to_string(kept.size()) + " kept");
        return c.status();
    }
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "direct") {
        return run_refusal(c, "seg.b1", "direct segment", false);
    }
    if (a.case_name == "recording") {
        return run_refusal(c, "seg.c2", "recording", false);
    }
    if (a.case_name == "active-valid") {
        return run_refusal(c, "seg.c2", "recording", true);
    }
    if (a.case_name == "moe-after") {
        return run_moe_after(c);
    }
    if (a.case_name == "stale-vectors") {
        return run_stale_vectors(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "direct", "recording", "active-valid", "moe-after",
                                                   "stale-vectors" };
    return file_mapped_test::gated_main(argc, argv, all, run_case, std::chrono::minutes(4));
}
