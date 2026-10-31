"""Source contract for the batched prompt executor's ring windows (llama.cpp-8epx, Task 3 of
docs/plans/2026-10-24-prefill-window-ring.md, W1 and W4).

Host-only: reads sources, builds nothing, loads no model and touches no device.

On a layer the executor claims (pp_onednn_batched_claims_staged_layer, from moe_stream_phase::executor_claims), the
batched executor is the ring's consumer one window at a time, for each op that reads windows. What this gate pins,
each claim with mutants that must turn it red (its positive controls):
  - the stream call lives at ggml_sycl_mul_mat_id's function scope, ahead of the prompt branch, so the op window_begin
    hands the ring (whose host_lease reads the call through op.graph_ctx) outlives the claim block; the claim is
    executor_claims(stream_call.layer);
  - whether an op reads windows is decided per op before the expert loop (pp_stream_windowed): its layer is claimed,
    its role is GATE, UP or DOWN, and it runs the f16 arm. Every window block keys on it, never on the layer's claim,
    and nothing asserts the WOQ arm away: a windowless op of a claimed layer runs as on an unclaimed one;
  - the expert loop defers a windowed op's HOST experts (expert.deferred, AOS, the ring's only layout) and asks the
    stage table nothing: their slots are read in their phase, after window_begin;
  - for a windowed op a phase is one window: phase_experts is capped at phase_experts(), G_plan, the phase the f16
    slot holds (moe_stream_phase_experts), still bounded by the f16 group size (moe_pp_batched_phase_experts), never
    at the ring region's slots: a phase spans device and host experts while its window stages only its host matrices
    of one role, at most min(G_plan, role_items_max), a region. The phases and the f16 halves are built from the
    capped count, and the phase no longer has an accessor that answers the region's slots (window_experts());
  - each phase of a windowed op asks window_begin exactly once with the phase's experts and the window after it (the
    next phase of the role, or phase 0 of the next role in the order the graph dispatches them, GATE, UP, DOWN; none
    after DOWN), sums its return into pp_staged_experts and its misses, counted per window, into pp_window_misses; then
    reads each deferred expert's slot (resolve_deferred_expert: the entry's consumer, AOS, view and tag checks, the
    entry's ready event onto a per-window barrier, the slot's lease held), dequantizes, submits the readers_done barrier
    and hands it to window_done, all before the phase's GEMMs;
  - window_begin itself (moe-stream-plan.cpp, read here, not edited) feeds both counts from the same two facts: its
    *misses is the window's missing matrices unless the window is exposed, and note_window is fed (exposed, missing,
    the window's staged bytes) after both are known (c-qz9m item 2: `*misses = missing` survived every gate);
  - a window is exposed (its copies counted apart from misses=) only when it is its layer's first in the micro-batch:
    every later window, UP phase 0 included, has the one before it to hide under, so moe_stream_window_exposed takes
    no role and no phase, only the layer and the previous window's layer;
  - a deferred expert the window did not stage (the ring could not take the matrix) is declined, never aborted
    (llama.cpp-9433), and every decline in the phase loop first drains the phases already submitted;
  - the [PP-MOE] first-run line counts the windows the dispatch began, and the executor trace line its misses
    (Task 6's gate sums misses=);
  - the prose that named the retired staging (moe_stream_phase::stage, reuse_after_readers, stage()) is gone.
"""
import re

import pytest
from sycl_gate import body, first_missing, first_present, mutate, norm, ordered, read


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
TEST_HPP = read("ggml/src/ggml-sycl/ggml-sycl-test.hpp")
PLAN_CPP = read("ggml/src/ggml-sycl/moe-stream-plan.cpp")
PLAN_HPP = read("ggml/src/ggml-sycl/moe-stream-plan.hpp")


def depth_at(text: str, pos: int) -> int:
    """The brace depth at `pos` of `text` (string literals skipped): 1 directly inside the outermost braces."""
    depth = 0
    i = 0
    while i < pos:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < pos and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        depth += 1 if ch == "{" else -1 if ch == "}" else 0
        i += 1
    return depth


MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try"
EXEC_SIG = "auto try_pp_onednn_f16_batched = [&]() -> bool"
EXPERT_STRUCT = "struct pp_onednn_batched_expert {"
EARLY_LOOP = "for (int64_t i02 = 0; i02 < n_as; ++i02) {"
PHASE_LOOP = "for (size_t p = 0; p < phases.size() && !gemm_declined; ++p) {"

# ---- the stream call at function scope, and the claim ----------------------------------------------------------------

STREAM_CALL_DECL = "ggml_sycl_moe_stream_call stream_call;"
PROMPT_BRANCH = "if (ne12 != 1) {"
CLAIM_BEGIN = "if (ids->ne[1] > 1 && ggml_sycl_moe_stream_prepare(ctx, src0, ids, stream_call)) {"
CLAIM = "pp_onednn_batched_claims_staged_layer = ctx.moe_stream->executor_claims(stream_call.layer);"


def claim_stream_call_at_function_scope(sycl: str) -> bool:
    """One stream call per op, declared directly in ggml_sycl_mul_mat_id's body ahead of the prompt branch, so the op
    the executor lambda hands window_begin points (op.graph_ctx) at a live call; the claim block fills it and asks
    executor_claims for its layer."""
    m = body(norm(sycl), MMID_SIG)
    at = m.find(STREAM_CALL_DECL)
    return (m != "" and m.count(STREAM_CALL_DECL) == 1 and depth_at(m, at) == 1
            and ordered(m, STREAM_CALL_DECL, PROMPT_BRANCH, CLAIM_BEGIN, CLAIM, EXEC_SIG)
            and norm(sycl).count(CLAIM) == 1)


def test_stream_call_at_function_scope():
    assert claim_stream_call_at_function_scope(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of ggml_sycl_mul_mat_id", body(norm(SYCL), MMID_SIG),
        STREAM_CALL_DECL, PROMPT_BRANCH, CLAIM_BEGIN, CLAIM, EXEC_SIG,
        signature=MMID_SIG)


def test_stream_call_at_function_scope_mutants():
    c = claim_stream_call_at_function_scope
    # The call scoped to the claim block again: the executor lambda would read a dead call through op.graph_ctx.
    m = mutate(SYCL, STREAM_CALL_DECL, "")
    m = m.replace(CLAIM_BEGIN, "{ " + STREAM_CALL_DECL + " " + CLAIM_BEGIN, 1)
    assert not c(m)
    # Declared inside the prompt branch, a nested scope.
    m = mutate(SYCL, STREAM_CALL_DECL, "")
    assert not c(m.replace(PROMPT_BRANCH, PROMPT_BRANCH + " " + STREAM_CALL_DECL, 1))
    # A second call shadowing the first.
    assert not c(mutate(SYCL, CLAIM_BEGIN, STREAM_CALL_DECL + " " + CLAIM_BEGIN))
    # The claim keyed by another layer.
    assert not c(mutate(SYCL, CLAIM, CLAIM.replace("stream_call.layer", "moe_cache_layer_id(src0->name)")))


# ---- the window path is decided per op -------------------------------------------------------------------------------

ROLE_DECL = ("const ggml_sycl::expert_tensor_role pp_stream_role = ctx.moe_stream ? "
             "ggml_sycl::expert_tensor_role_from_tensor_name(src0->name) : ggml_sycl::expert_tensor_role::UNKNOWN;")
WOQ_ARM = "const bool pp_woq_arm = pp_woq_enabled && moe_pp_batched_woq_supports_layout(src0->type, route_layout);"
WINDOWED = ("const bool pp_stream_windowed = pp_onednn_batched_claims_staged_layer && "
            "ggml_sycl::detail::moe_expert_role_is_triplet(pp_stream_role) && !pp_woq_arm;")
LAYER_FLAG = "pp_onednn_batched_claims_staged_layer"
WOQ_ASSERT = "GGML_ASSERT(!pp_woq_arm || !pp_onednn_batched_claims_staged_layer);"
SORT = "std::sort(active_experts.begin(), active_experts.end(),"
# Each block that reads windows, as it opens: the deferral, the phase cap, the phase ids, the window request and the
# readers_done barrier.
WINDOWED_USES = (
    "} else if (pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) {",
    "if (pp_stream_windowed) { phase_experts = moe_pp_batched_phase_experts(",
    "if (pp_stream_windowed) { phase_ids.reserve(active_count);",
    "if (pp_stream_windowed) { ggml_sycl::moe_stream_window_request window_req;",
    "if (pp_stream_windowed) { const sycl::event readers_done =",
)


def claim_window_path_is_per_op(sycl: str) -> bool:
    """Whether an op reads ring windows is decided once per op, before the expert loop: its layer is claimed, its role
    is one a window holds (GATE, UP or DOWN, the roles moe_stream_window_id takes, by the predicate the ring's index
    uses), and it runs the f16 arm (the WOQ arm reads no window). A claimed layer can carry an op that is none of these,
    a fused GATE_UP or CHUNK_* weight or a WOQ-arm MXFP4 role, because the claim counts only host matrices and such an
    op's are all resident. Every block after the decision keys on pp_stream_windowed, never on the layer's claim, and
    nothing asserts the WOQ arm away: the cache can evict a planned-device expert at runtime, so a windowless op's HOST
    route is reachable, and it declines EXPERT_NOT_LOCAL as on an unclaimed layer (the deferral's else branch)."""
    lam = body(norm(sycl), EXEC_SIG)
    at = lam.find(WINDOWED)
    return (at >= 0 and lam.count(WINDOWED) == 1 and lam.count(WOQ_ARM) == 1
            and ordered(lam, ROLE_DECL, WOQ_ARM, WINDOWED, EARLY_LOOP)
            and all(lam.count(use) == 1 for use in WINDOWED_USES)
            and LAYER_FLAG not in lam[at + len(WINDOWED):] and "GGML_ASSERT(!pp_woq_arm" not in lam)


def test_window_path_is_per_op():
    assert claim_window_path_is_per_op(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of the try_pp_onednn_f16_batched lambda", body(norm(SYCL), EXEC_SIG),
        ROLE_DECL, WOQ_ARM, WINDOWED, EARLY_LOOP, *WINDOWED_USES,
        signature=EXEC_SIG)


def test_window_path_is_per_op_mutants():
    c = claim_window_path_is_per_op
    # The role test dropped: a claimed layer's fused GATE_UP or CHUNK_* op would ask window_begin, whose window id
    # aborts on any role but GATE, UP or DOWN.
    assert not c(mutate(SYCL, WINDOWED,
                        WINDOWED.replace("ggml_sycl::detail::moe_expert_role_is_triplet(pp_stream_role) && ", "")))
    # The WOQ term dropped: a claimed layer's resident MXFP4 op would stage windows its arm never reads.
    assert not c(mutate(SYCL, WINDOWED, WINDOWED.replace(" && !pp_woq_arm", "")))
    # The WOQ arm asserted away again instead of reading no window.
    assert not c(mutate(SYCL, WOQ_ARM, WOQ_ARM + " " + WOQ_ASSERT))
    # Decided after the expert loop, which defers on it.
    m = mutate(SYCL, WINDOWED, "")
    assert not c(m.replace(SORT, WINDOWED + " " + SORT, 1))


@pytest.mark.parametrize("use", WINDOWED_USES)
def test_window_path_is_per_op_layer_flag_mutants(use):
    # The block keyed on the layer's claim again: the op's role and arm no longer decide it.
    assert not claim_window_path_is_per_op(mutate(SYCL, use, use.replace("pp_stream_windowed", LAYER_FLAG)))


# ---- the expert loop defers a windowed op's host experts -------------------------------------------------------------

DEFERRED_FIELD = "bool deferred = false;"
DEFERRED = ("} else if (pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) { "
            "expert.deferred = true; expert.layout = ggml_sycl::MOE_STREAM_RING_SLOT_LAYOUT; } else { "
            "return reject_batched(moe_pp_batched_refusal::EXPERT_NOT_LOCAL, "
            "ggml_sycl_expert_resolve_reason_name(route.reason), i02); }")


def claim_early_loop_defers(sycl: str) -> bool:
    """The expert loop marks a windowed op's HOST expert deferred, with the AOS reader the ring's slots take (the ring
    stages AOS only), and asks the stage table nothing: the slot is read in its phase, after window_begin. Any other
    non-device expert, a windowless op's included, is declined EXPERT_NOT_LOCAL as before."""
    lam = body(norm(sycl), EXEC_SIG)
    early = body(lam, EARLY_LOOP)
    return (early != "" and lam.count(EARLY_LOOP) == 1 and early.count(DEFERRED) == 1
            and "->find(" not in early and "window_begin(" not in early
            and body(lam, EXPERT_STRUCT).count(DEFERRED_FIELD) == 1)


def test_early_loop_defers():
    assert claim_early_loop_defers(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of the expert loop in the try_pp_onednn_f16_batched lambda "
        "(the deferred field is searched in pp_onednn_batched_expert)",
        body(body(norm(SYCL), EXEC_SIG), EARLY_LOOP), DEFERRED,
        signature=EARLY_LOOP)


def test_early_loop_defers_mutants():
    c = claim_early_loop_defers
    # find restored in the expert loop: it reads a window not staged yet.
    assert not c(mutate(SYCL, "expert.deferred = true;",
                        "(void) ctx.moe_stream->find(pp_stream_layer, static_cast<int>(i02), pp_stream_role); "
                        "expert.deferred = true;"))
    # A host expert of a windowed op declined instead of deferred (the interim inert ring).
    assert not c(mutate(SYCL, "expert.deferred = true; ", ""))
    # Any route deferred, not only HOST.
    assert not c(mutate(SYCL, "pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) {",
                        "pp_stream_windowed) {"))
    # A windowless op's host expert deferred: no window would ever stage it.
    assert not c(mutate(SYCL, "} else if (pp_stream_windowed && route.kind", "} else if (route.kind"))
    # The struct without the flag.
    assert not c(mutate(SYCL, DEFERRED_FIELD, ""))


# ---- a phase is one window -------------------------------------------------------------------------------------------

PHASE_CAP = ("size_t phase_experts = pp_woq_arm ? active_count : f16_groups.experts_per_group; "
             "if (pp_stream_windowed) { "
             "phase_experts = moe_pp_batched_phase_experts(f16_groups.experts_per_group, "
             "ctx.moe_stream->phase_experts()); "
             "GGML_ASSERT(phase_experts > 0); }")
PHASE_CAP_CALL = "moe_pp_batched_phase_experts(f16_groups.experts_per_group, ctx.moe_stream->phase_experts())"
HALF_ELEMS = "const size_t half_elems = phase_experts * weight_elems;"
PHASES_BUILD = "for (size_t begin = 0; begin < active_count; begin += phase_experts) {"
PHASE_IDS = ("std::vector<int32_t> phase_ids; if (pp_stream_windowed) { "
             "phase_ids.reserve(active_count); for (const pp_onednn_batched_expert & expert : active_experts) { "
             "phase_ids.push_back(expert.expert_id); } }")


def claim_phase_is_a_window(sycl: str) -> bool:
    """For a windowed op the phase is capped at phase_experts(), G_plan, bounded by the f16 group size. A window then
    holds at most min(G_plan, role_items_max) host matrices of one role, a region, so it never exceeds its region
    (window_begin aborts on one that does), and a layer with few host matrices per role is not split into that many
    experts per phase. The f16 halves and the phases follow the capped count. The phase ids are read after the experts
    are sorted, since a phase indexes the sorted list."""
    lam = body(norm(sycl), EXEC_SIG)
    return (lam.count(PHASE_CAP) == 1 and "const size_t phase_experts" not in lam
            and ordered(lam, WINDOWED, PHASE_CAP, HALF_ELEMS, PHASES_BUILD)
            and lam.count(PHASE_IDS) == 1 and ordered(lam, SORT, PHASE_IDS, PHASE_LOOP))


def test_phase_is_a_window():
    assert claim_phase_is_a_window(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of the try_pp_onednn_f16_batched lambda", body(norm(SYCL), EXEC_SIG),
        WINDOWED, PHASE_CAP, HALF_ELEMS, PHASES_BUILD, SORT, PHASE_IDS, PHASE_LOOP,
        signature=EXEC_SIG)


def test_phase_is_a_window_mutants():
    c = claim_phase_is_a_window
    # The phase cap removed: a group the slot takes whole would exceed the ring's region.
    assert not c(mutate(SYCL, PHASE_CAP, "size_t phase_experts = pp_woq_arm ? active_count : "
                                         "f16_groups.experts_per_group;"))
    # The cap restored to the region's slots: a layer with few host matrices per role fragments into that many experts
    # per phase.
    assert not c(mutate(SYCL, PHASE_CAP_CALL, "std::min(f16_groups.experts_per_group, "
                                              "static_cast<size_t>(ctx.moe_stream->window_experts()))"))
    # The cap without the f16 group bound: a G_plan above the group size would overrun the phase's f16 buffer.
    assert not c(mutate(SYCL, PHASE_CAP_CALL, "ctx.moe_stream->phase_experts()"))
    # The halves sized before the cap.
    m = mutate(SYCL, HALF_ELEMS, "")
    assert not c(m.replace(PHASE_CAP, HALF_ELEMS + " " + PHASE_CAP, 1))
    # The phase ids read before the sort: a phase's window names the wrong experts.
    m = mutate(SYCL, PHASE_IDS, "")
    assert not c(m.replace(SORT, PHASE_IDS + " " + SORT, 1))


# ---- the phase's cap is G_plan, never the region's slots -------------------------------------------------------------

PHASE_EXPERTS_FN = ("static inline uint32_t moe_stream_phase_experts(const moe_stream_plan_index & idx, "
                    "const moe_stream_ring_geometry & geo)")
PHASE_EXPERTS_RULE = "return geo.n_slots != 0 ? idx.window_experts : 0;"
PHASE_EXPERTS_FWD = "uint32_t phase_experts() const { return moe_stream_phase_experts(index_, geo_); }"
REGION_ACCESSOR = "window_experts() const"


def claim_phase_experts_is_g_plan(hpp: str) -> bool:
    """moe_stream_phase::phase_experts() forwards to the one rule, moe_stream_phase_experts, which answers the index's
    G_plan (window_experts, the experts the f16 slot's phase holds) when a ring geometry exists, never the region's
    slots (geo.region_slots, index window_slots). No accessor answering the region's slots remains on the phase."""
    h = norm(hpp)
    return (body(h, PHASE_EXPERTS_FN).strip() == "{ " + PHASE_EXPERTS_RULE + " }" and h.count(PHASE_EXPERTS_FWD) == 1
            and REGION_ACCESSOR not in h)


def test_phase_experts_is_g_plan():
    h = norm(PLAN_HPP)
    assert claim_phase_experts_is_g_plan(PLAN_HPP), (
        first_present("ggml/src/ggml-sycl/moe-stream-plan.hpp: whole file", h, REGION_ACCESSOR)
        if REGION_ACCESSOR in h else
        first_missing("ggml/src/ggml-sycl/moe-stream-plan.hpp: body of moe_stream_phase_experts and the class",
                      body(h, PHASE_EXPERTS_FN) + " " + h, PHASE_EXPERTS_RULE, PHASE_EXPERTS_FWD,
                      signature=PHASE_EXPERTS_FN))


def test_phase_experts_is_g_plan_mutants():
    c = claim_phase_experts_is_g_plan
    # The rule answers the region's slots: the defect this cap fixed, through either field that holds them.
    assert not c(mutate(PLAN_HPP, PHASE_EXPERTS_RULE, "return geo.region_slots;"))
    assert not c(mutate(PLAN_HPP, PHASE_EXPERTS_RULE, "return geo.n_slots != 0 ? idx.window_slots : 0;"))
    # The forwarder bypasses the rule and reads the region.
    assert not c(mutate(PLAN_HPP, PHASE_EXPERTS_FWD, "uint32_t phase_experts() const { return geo_.region_slots; }"))
    # An accessor answering the region's slots comes back beside it.
    assert not c(mutate(PLAN_HPP, PHASE_EXPERTS_FWD,
                        PHASE_EXPERTS_FWD + " uint32_t window_experts() const { return geo_.region_slots; }"))


# ---- each phase of a claimed layer is bracketed by window_begin and window_done --------------------------------------

WINDOW_MISSES_DECL = "uint32_t window_misses = 0;"
WINDOW_REQ = (
    "if (pp_stream_windowed) { ggml_sycl::moe_stream_window_request window_req; "
    "window_req.layer = pp_stream_layer; window_req.role = pp_stream_role; window_req.phase = p; "
    "window_req.experts = phase_ids.data() + phase.begin; window_req.n = phase.end - phase.begin;"
)
NEXT_FILL = (
    "if (p + 1 < phases.size()) { window_req.next = phase_ids.data() + phases[p + 1].begin; "
    "window_req.n_next = phases[p + 1].end - phases[p + 1].begin; } "
    "else if (pp_stream_role != ggml_sycl::expert_tensor_role::DOWN) { "
    "window_req.next_role = pp_stream_role == ggml_sycl::expert_tensor_role::GATE ? "
    "ggml_sycl::expert_tensor_role::UP : ggml_sycl::expert_tensor_role::DOWN; "
    "window_req.next = phase_ids.data() + phases[0].begin; window_req.n_next = phases[0].end - phases[0].begin; }"
)
WINDOW_BEGIN = ("pp_staged_experts += ctx.moe_stream->window_begin(window_req, *stream_call.cache, stream_call.op, "
                "&window_misses);")
MISSES_SUM = "pp_window_misses += window_misses;"
WINDOWS_COUNT = "++pp_windows;"
WINDOW_READY = "std::vector<sycl::event> window_ready;"
DEFERRED_LOOP = ("for (size_t i = phase.begin; i < phase.end; ++i) { if (active_experts[i].deferred && "
                 "!resolve_deferred_expert(active_experts[i], window_ready)) { return false; } }")
WINDOW_BARRIER = "if (!window_ready.empty()) { stream->ext_oneapi_submit_barrier(window_ready); } }"
DEQUANT = "dequant_expert_f16(active_experts[i], phase_weights + (i - phase.begin) * weight_elems);"
READERS_DONE = ("if (pp_stream_windowed) { "
                "const sycl::event readers_done = stream->ext_oneapi_submit_barrier(); "
                "ctx.moe_stream->window_done(pp_stream_layer, pp_stream_role, p, readers_done); }")
GEMM_TRY = "try { const sycl::event deps_in = stream->ext_oneapi_submit_barrier();"
GEMM = "DnnlGemmWrapper::gemm_batch_strided("
# The misses are declared per window, directly ahead of the call that writes them.
PHASE_STEPS = (WINDOW_REQ, NEXT_FILL, WINDOW_MISSES_DECL + " " + WINDOW_BEGIN, MISSES_SUM, WINDOWS_COUNT, WINDOW_READY,
               DEFERRED_LOOP, WINDOW_BARRIER, DEQUANT, READERS_DONE, GEMM_TRY, GEMM)


def claim_phase_brackets_its_window(sycl: str) -> bool:
    """Each phase of a windowed op asks window_begin exactly once, with the phase's experts and the window after it
    (W1: the next phase of the role, else phase 0 of the next role in the order the graph dispatches them, GATE, UP,
    DOWN, and nothing after DOWN, since no window prefetches across layers). The order is the graph's, not the FFN
    builder's: the builder creates the up matmul first, but the gated activation takes the gate output as src[0]
    and ggml's graph walk visits src[0] first. window_begin's not-prefetched branch stages only what is
    missing, so a second call would decline experts its own prefetch evicted (c-qz9m). Its return is the window's
    staged host experts (pp_staged_experts) and its *misses the window's unhidden copies (pp_window_misses, the trace's
    misses=). The deferred experts are read after it, their copies ordered by a per-window barrier, then the dequants
    run, then readers_done follows them and becomes the window's read gate (window_done), all before the GEMMs."""
    n = norm(sycl)
    lam = body(n, EXEC_SIG)
    loop = body(lam, PHASE_LOOP)
    return (loop != "" and n.count("->window_begin(") == 1 and n.count("->window_done(") == 1
            and loop.count(WINDOW_BEGIN) == 1 and loop.count(MISSES_SUM) == 1 and loop.count(WINDOWS_COUNT) == 1
            and lam.count(WINDOW_MISSES_DECL) == 1 and ordered(loop, *PHASE_STEPS)
            and lam.count("pp_staged_experts +=") == 1 and "++pp_staged_experts" not in lam)


def test_phase_brackets_its_window():
    assert claim_phase_brackets_its_window(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of the phase loop in the try_pp_onednn_f16_batched lambda",
        body(body(norm(SYCL), EXEC_SIG), PHASE_LOOP), *PHASE_STEPS,
        signature=PHASE_LOOP)


def test_phase_brackets_its_window_mutants():
    c = claim_phase_brackets_its_window
    # The window_begin call removed: the deferred experts would find nothing and decline.
    assert not c(mutate(SYCL, WINDOW_BEGIN, ""))
    # Asked for the whole layer before the loop instead of per phase.
    m = mutate(SYCL, WINDOW_BEGIN, "")
    assert not c(m.replace(PHASE_LOOP, WINDOW_BEGIN + " " + PHASE_LOOP, 1))
    # Asked twice per window.
    assert not c(mutate(SYCL, WINDOW_BEGIN, WINDOW_BEGIN + " " + WINDOW_BEGIN))
    # The staged count counted per expert again instead of summed from window_begin.
    assert not c(mutate(SYCL, WINDOW_BEGIN, WINDOW_BEGIN.replace("pp_staged_experts += ", "(void) ")))
    # window_done removed: the region holding the window has no read gate, and its refill waits for a later barrier.
    assert not c(mutate(SYCL, "ctx.moe_stream->window_done(pp_stream_layer, pp_stream_role, p, readers_done);", ""))
    # The readers_done barrier removed: window_done would be handed an event that precedes the dequants.
    assert not c(mutate(SYCL, "const sycl::event readers_done = stream->ext_oneapi_submit_barrier();",
                        "const sycl::event readers_done = sycl::event();"))
    # window_done before the dequants.
    m = mutate(SYCL, READERS_DONE, "")
    assert not c(m.replace(WINDOW_READY, READERS_DONE + " " + WINDOW_READY, 1))
    # The deferred experts read before window_begin staged their window.
    m = mutate(SYCL, DEFERRED_LOOP, "")
    assert not c(m.replace(WINDOW_REQ, DEFERRED_LOOP + " " + WINDOW_REQ, 1))
    # The per-window barrier on the deferred experts' copies dropped.
    assert not c(mutate(SYCL, WINDOW_BARRIER, "}"))
    # The misses accumulation removed: the trace's misses= would read 0 whatever window_begin counted.
    assert not c(mutate(SYCL, MISSES_SUM, ""))
    # The window count removed.
    assert not c(mutate(SYCL, WINDOWS_COUNT, ""))
    # The next/next_role fill removed: no window would be prefetched, every copy exposed.
    assert not c(mutate(SYCL, NEXT_FILL, ""))
    # The next role's fill removed: the last phase of UP and GATE prefetch nothing.
    assert not c(mutate(SYCL, NEXT_FILL, NEXT_FILL[:NEXT_FILL.index(" else if")]))
    # The FFN builder's source order (UP, GATE, DOWN) instead of the dispatched one: GATE's last phase would prefetch
    # DOWN, which UP's staging then evicts, UP phase 0 would copy on demand, and UP's last phase would prefetch GATE.
    assert not c(mutate(SYCL, "expert_tensor_role::GATE ? ggml_sycl::expert_tensor_role::UP : "
                              "ggml_sycl::expert_tensor_role::DOWN;",
                        "expert_tensor_role::UP ? ggml_sycl::expert_tensor_role::GATE : "
                        "ggml_sycl::expert_tensor_role::DOWN;"))
    # UP's last phase prefetching GATE, which already ran.
    assert not c(mutate(SYCL, "expert_tensor_role::GATE ? ggml_sycl::expert_tensor_role::UP : "
                              "ggml_sycl::expert_tensor_role::DOWN;",
                        "expert_tensor_role::GATE ? ggml_sycl::expert_tensor_role::UP : "
                        "ggml_sycl::expert_tensor_role::GATE;"))
    # DOWN prefetching across the layer boundary.
    assert not c(mutate(SYCL, "else if (pp_stream_role != ggml_sycl::expert_tensor_role::DOWN) {", "else {"))
    # The next phase's experts taken from this phase.
    assert not c(mutate(SYCL, "window_req.next = phase_ids.data() + phases[p + 1].begin;",
                        "window_req.next = phase_ids.data() + phase.begin;"))
    # The misses not asked for.
    assert not c(mutate(SYCL, "stream_call.op, &window_misses);", "stream_call.op, nullptr);"))
    # The misses declared once for the whole layer instead of per window.
    m = mutate(SYCL, WINDOW_MISSES_DECL + " ", "")
    assert not c(m.replace(PHASE_LOOP, WINDOW_MISSES_DECL + " " + PHASE_LOOP, 1))


# ---- window_begin feeds *misses and note_window from the same facts --------------------------------------------------

WB_SIG = "uint32_t moe_stream_phase::window_begin(const moe_stream_window_request & req,"
WB_MISSING = "missing += find(item.layer, item.expert_id, item.role) == nullptr ? 1u : 0u;"
WB_EXPOSED = "const bool exposed = moe_stream_window_exposed(req.layer, prev_window_layer_);"
WB_MISSES = "if (misses != nullptr) { *misses = exposed ? 0 : missing; }"
# (exposed, missing, the window's staged bytes): the three-argument form Task 2c (llama.cpp-4huq) landed.
WB_NOTE_RE = re.compile(r"if \(log_ != nullptr\) \{ log_->note_window\(exposed, missing, [A-Za-z_]\w*\); \}")


def claim_window_begin_feeds_misses_and_note(plan_cpp: str) -> bool:
    """window_begin answers the executor's *misses and feeds the decision log's note_window from the same two facts:
    the window's matrices no entry holds after any staging (missing), and whether the window's copies were exposed.
    *misses is the missing count unless the window was exposed (an exposed window's copies are the trace's other
    count, so the executor's misses= reads only hidden-copy misses), and note_window gets (exposed, missing) and the
    window's staged bytes. Both follow the exposed decision, and each is fed once."""
    wb = body(norm(plan_cpp), WB_SIG)
    notes = WB_NOTE_RE.findall(wb)
    if wb == "" or len(notes) != 1 or wb.count("note_window(") != 1 or wb.count("*misses") != 1:
        return False
    return (wb.count(WB_MISSING) == 1 and wb.count(WB_EXPOSED) == 1 and wb.count(WB_MISSES) == 1
            and ordered(wb, WB_MISSING, WB_EXPOSED, WB_MISSES) and ordered(wb, WB_EXPOSED, notes[0]))


def test_window_begin_feeds_misses_and_note():
    assert claim_window_begin_feeds_misses_and_note(PLAN_CPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::window_begin "
        "(the note_window feed is a regex there)", body(norm(PLAN_CPP), WB_SIG),
        WB_MISSING, WB_EXPOSED, WB_MISSES, "note_window(",
        signature=WB_SIG)


def test_window_begin_feeds_misses_and_note_mutants():
    c = claim_window_begin_feeds_misses_and_note
    note = WB_NOTE_RE.search(norm(PLAN_CPP)).group(0)
    # The exposed window's hide dropped: misses= would count copies the trace already reports as exposed.
    assert not c(mutate(PLAN_CPP, "*misses = exposed ? 0 : missing;", "*misses = missing;"))
    # The misses never reported.
    assert not c(mutate(PLAN_CPP, "*misses = exposed ? 0 : missing;", "*misses = 0;"))
    assert not c(mutate(PLAN_CPP, WB_MISSES, ""))
    # The misses counted over the entries present instead of the ones missing.
    assert not c(mutate(PLAN_CPP, WB_MISSING, WB_MISSING.replace("== nullptr", "!= nullptr")))
    # The note not fed.
    assert not c(mutate(PLAN_CPP, note, ""))
    # The note fed the opposite exposure, or no missing count.
    assert not c(mutate(PLAN_CPP, "note_window(exposed, missing", "note_window(!exposed, missing"))
    assert not c(mutate(PLAN_CPP, "note_window(exposed, missing", "note_window(exposed, 0u"))
    # The note fed without the window's staged bytes, the two-argument form before Task 2c.
    assert not c(mutate(PLAN_CPP, note, re.sub(r", [A-Za-z_]\w*\); \}$", "); }", note)))
    # The note fed twice.
    assert not c(mutate(PLAN_CPP, note, note + " " + note))
    # The note fed before the window's exposure is known.
    m = mutate(PLAN_CPP, note, "")
    assert not c(m.replace(WB_EXPOSED, note + " " + WB_EXPOSED, 1))


# ---- a window is exposed only as its layer's first in the micro-batch ------------------------------------------------

EXPOSED_SIG = "static inline bool moe_stream_window_exposed(int layer, int prev_layer) {"
EXPOSED_RETURN = "return layer != prev_layer; }"
BEGIN_UBATCH_SIG = "void moe_stream_phase::begin_ubatch(uint32_t n_tokens) {"
UBATCH_RESET = "prev_window_layer_ = -1;"


def claim_exposed_is_a_layers_first_window(plan_hpp: str, plan_cpp: str) -> bool:
    """A window is exposed, its copies counted apart from the trace's misses=, only when no earlier window of its layer
    could hide it: its layer's first window in the micro-batch. Every later window has the one before it to hide under,
    which prefetches the next window (the next phase of its role, or phase 0 of the next role), UP phase 0 included, so
    the predicate names no role and no phase. begin_ubatch resets the last window's layer to -1, which is what makes a
    micro-batch's first window exposed even when it is the layer the last micro-batch ended at. This is an
    exact-expression pin: it requires the signature to be exactly `(int layer, int prev_layer)`, so a restored parameter
    fails it, the body to return `layer != prev_layer` and name no role or phase, and begin_ubatch to keep its reset; it
    never evaluates the predicate. The exposure block of test-moe-stream-phase.cpp (the sycl-moe-stream-phase ctest)
    pins the behaviour, both polarities."""
    b = body(norm(plan_hpp), EXPOSED_SIG)
    ub = body(norm(plan_cpp), BEGIN_UBATCH_SIG)
    return (b != "" and b.endswith(EXPOSED_RETURN) and b.count("return ") == 1 and "expert_tensor_role::" not in b
            and "role ==" not in b and "phase_index ==" not in b and ub.count(UBATCH_RESET) == 1)


def test_exposed_is_a_layers_first_window():
    assert claim_exposed_is_a_layers_first_window(PLAN_HPP, PLAN_CPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.hpp: body of moe_stream_window_exposed "
        "(begin_ubatch's reset is searched in moe-stream-plan.cpp)", body(norm(PLAN_HPP), EXPOSED_SIG),
        EXPOSED_RETURN,
        signature=EXPOSED_SIG)


def test_exposed_is_a_layers_first_window_mutants():
    c = claim_exposed_is_a_layers_first_window
    # The UP-phase-0 clause back: the predicate names a role again.
    assert not c(mutate(PLAN_HPP, EXPOSED_RETURN,
                        "return layer != prev_layer || (role == expert_tensor_role::UP && phase_index == 0); }"),
                 PLAN_CPP)
    # A GATE-phase-0 clause: the predicate names a role, a second source for the dispatch order.
    assert not c(mutate(PLAN_HPP, EXPOSED_RETURN,
                        "return layer != prev_layer || (role == expert_tensor_role::GATE && phase_index == 0); }"),
                 PLAN_CPP)
    # Any other return expression. The exact-text check does not tell these two apart; the phase test does.
    assert not c(mutate(PLAN_HPP, EXPOSED_RETURN, "return true; }"), PLAN_CPP)
    assert not c(mutate(PLAN_HPP, EXPOSED_RETURN, "return false; }"), PLAN_CPP)
    # A restored parameter: the signature takes the layer and the previous layer and nothing else.
    assert not c(mutate(PLAN_HPP, EXPOSED_SIG,
                        "static inline bool moe_stream_window_exposed(int layer, expert_tensor_role role, "
                        "int prev_layer) {"), PLAN_CPP)
    assert not c(mutate(PLAN_HPP, EXPOSED_SIG,
                        "static inline bool moe_stream_window_exposed(int layer, size_t phase_index, "
                        "int prev_layer) {"), PLAN_CPP)
    # begin_ubatch's reset of the last window's layer dropped.
    assert not c(PLAN_HPP, mutate(PLAN_CPP, UBATCH_RESET, ""))


# ---- a deferred expert's slot, and the declines that drain -----------------------------------------------------------

DRAIN = ("auto drain_submitted_phases = [&]() { const sycl::event drained = gemm_events.empty() ? "
         "stream->ext_oneapi_submit_barrier() : stream->ext_oneapi_submit_barrier(gemm_events); "
         "retain_expert_leases_until(drained); batched_scratch_claim.finish_or_release(); };")
RESOLVE_SIG = ("auto resolve_deferred_expert = [&](pp_onednn_batched_expert & expert, "
               "std::vector<sycl::event> & window_ready) -> bool {")
DRAINED_DECLINE = "drain_submitted_phases(); return reject_batched("
RESOLVE_STEPS = (
    "const ggml_sycl::moe_stream_staged * staged = "
    "ctx.moe_stream->find(pp_stream_layer, expert.expert_id, pp_stream_role);",
    "const bool executor_entry = staged != nullptr && staged->consumer == ggml_sycl::moe_stream_consumer::EXECUTOR;",
    "if (!executor_entry) { " + DRAINED_DECLINE + "moe_pp_batched_refusal::EXPERT_NOT_LOCAL, "
    "\"window-not-staged\", expert.expert_id); }",
    "GGML_ASSERT(staged->layout == ggml_sycl::MOE_STREAM_RING_SLOT_LAYOUT);",
    "const ggml_sycl::resolved_ptr view = staged->aos.resolve(ctx.device);",
    "if (view.ptr == nullptr || !view.on_device) { " + DRAINED_DECLINE + "moe_pp_batched_refusal::EXPERT_NOT_LOCAL, "
    "\"staged-slot-view\", expert.expert_id); }",
    "if (view.layout != staged->layout) { " + DRAINED_DECLINE + "moe_pp_batched_refusal::EXPERT_LAYOUT, "
    "ggml_sycl_layout_mode_name(view.layout), expert.expert_id); }",
    "window_ready.push_back(staged->ready);",
    "expert.ptr = view.ptr;",
    "expert.layout = staged->layout;",
    "expert.lease = staged->aos;",
    "return true; }",
)
GEMM_FAIL_SIG = "auto gemm_stage_failed = [&](const char * what) -> bool {"


def claim_deferred_slot_and_draining_declines(sycl: str) -> bool:
    """resolve_deferred_expert reads a deferred expert's slot once its window is staged: the stage table once, keyed
    as the staged route keys it; the entry only when its consumer is the executor; AOS asserted; the view on this
    device and tagged with the entry's layout; the entry's ready event onto the window's barrier; the slot, its layout
    and its lease from the entry (held until the done marker like every expert's). A missing entry is a matrix the ring
    could not take, which window_begin skipped by contract (llama.cpp-9433): it is declined, never aborted. Every
    decline there runs after earlier phases' dequants and GEMMs were submitted, so it drains them first (the barrier on
    the GEMMs, every lease held to it, the scratch claim finished), the one drain the GEMM stage's failure takes too."""
    lam = body(norm(sycl), EXEC_SIG)
    res = body(lam, RESOLVE_SIG)
    gemm_fail = body(lam, GEMM_FAIL_SIG)
    return (res != "" and lam.count(RESOLVE_SIG) == 1 and ordered(res, *RESOLVE_STEPS) and "GGML_ABORT(" not in res
            and res.count("reject_batched(") == res.count(DRAINED_DECLINE) == 3
            and res.count("->find(") == 1 and lam.count("->find(") == 1
            and lam.count(DRAIN) == 1 and gemm_fail.count("drain_submitted_phases();") == 1
            and lam.count("const sycl::event drained =") == 1
            and ordered(lam, DRAIN, GEMM_FAIL_SIG, RESOLVE_SIG, PHASE_LOOP))


def test_deferred_slot_and_draining_declines():
    assert claim_deferred_slot_and_draining_declines(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of resolve_deferred_expert "
        "(the drain and the GEMM failure path are searched in the enclosing lambda)",
        body(body(norm(SYCL), EXEC_SIG), RESOLVE_SIG), *RESOLVE_STEPS,
        signature=RESOLVE_SIG)


def test_deferred_slot_and_draining_declines_mutants():
    c = claim_deferred_slot_and_draining_declines
    # A matrix the ring could not take aborts the prompt instead of declining (the plan's first wording).
    assert not c(mutate(SYCL, "if (!executor_entry) { drain_submitted_phases();",
                        "if (!executor_entry) { GGML_ABORT(\"not staged\"); drain_submitted_phases();"))
    # A decline that drops the in-flight phases' leases: no drain before the refusal.
    assert not c(mutate(SYCL, "if (!executor_entry) { drain_submitted_phases(); return",
                        "if (!executor_entry) { return"))
    assert not c(mutate(SYCL, "if (view.layout != staged->layout) { drain_submitted_phases(); return",
                        "if (view.layout != staged->layout) { return"))
    # The copy's ready event dropped, so the dequant may read the slot before the copy lands.
    assert not c(mutate(SYCL, "window_ready.push_back(staged->ready);", ""))
    # The slot's lease not held.
    assert not c(mutate(SYCL, "expert.lease = staged->aos;", ""))
    # The consumer the ring wrote not asked.
    assert not c(mutate(SYCL, "staged != nullptr && staged->consumer == ggml_sycl::moe_stream_consumer::EXECUTOR;",
                        "staged != nullptr;"))
    # The tag check dropped.
    assert not c(mutate(SYCL, RESOLVE_STEPS[6], ""))
    # The drain without its barrier, or releasing the claim before the leases are held.
    assert not c(mutate(SYCL, "const sycl::event drained = gemm_events.empty() ? stream->ext_oneapi_submit_barrier() "
                              ": stream->ext_oneapi_submit_barrier(gemm_events);",
                        "const sycl::event drained = sycl::event();"))
    assert not c(mutate(SYCL, "retain_expert_leases_until(drained); batched_scratch_claim.finish_or_release(); };",
                        "batched_scratch_claim.finish_or_release(); retain_expert_leases_until(drained); };"))
    # The GEMM stage's failure keeping its own copy of the drain instead of the one helper.
    assert not c(mutate(SYCL, GEMM_FAIL_SIG + " ", GEMM_FAIL_SIG + " const sycl::event drained = sycl::event(); "))
    # A matrix the ring could not take asserted instead of declined.
    assert not c(mutate(SYCL, RESOLVE_STEPS[2], "GGML_ASSERT(executor_entry);"))


# The three declines in resolve_deferred_expert. Each drains the phases already submitted before it refuses (lead
# decision on llama.cpp-8epx): a refusal that returns first, or never drains, leaves earlier phases' GEMMs reading
# expert slots whose leases and scratch claim the op then gives back.
DECLINE_HEADS = ("if (!executor_entry) { ", "if (view.ptr == nullptr || !view.on_device) { ",
                 "if (view.layout != staged->layout) { ")


def decline_text(head: str) -> str:
    """The whole decline that opens with `head`, from the normalized source."""
    res = body(body(norm(SYCL), EXEC_SIG), RESOLVE_SIG)
    at = res.find(head + DRAINED_DECLINE)
    assert at >= 0, f"no drained decline opens with {head!r}"
    return res[at:res.find("); }", at) + len("); }")]


@pytest.mark.parametrize("head", DECLINE_HEADS)
def test_each_decline_drains_before_it_refuses(head):
    c = claim_deferred_slot_and_draining_declines
    whole = decline_text(head)
    refusal = whole[len(head + "drain_submitted_phases(); "):-len(" }")]
    assert refusal.startswith("return reject_batched(") and refusal.endswith(");")
    # The refusal returned before the drain, which then never runs.
    assert not c(mutate(SYCL, whole, head + refusal + " drain_submitted_phases(); }"))
    # The refusal with no drain at all.
    assert not c(mutate(SYCL, whole, head + refusal + " }"))


# ---- the trace lines -------------------------------------------------------------------------------------------------

COUNTERS = "size_t pp_windows = 0; uint32_t pp_window_misses = 0;"
PP_MOE_FMT = ("\"[PP-MOE] device=%d type=%s layout=%s arm=%s experts=%zu groups=%zu planned_weight_slot=%zu "
              "windows=%zu\\n\"")
PP_MOE_ARGS = "phases.size(), planned_weight, pp_windows);"
EXEC_TRACE_FMT = "\"[MOE-EXEC-TRACE] executor device %d tensor=%s experts=%zu staged=%zu arm=%s misses=%u\""
EXEC_TRACE_ARGS = "pp_staged_experts, pp_woq_arm ? \"woq\" : \"f16\", pp_window_misses);"


def claim_trace_lines(sycl: str) -> bool:
    """The [PP-MOE] first-run line names the windows the dispatch began (0 on a layer the executor does not claim), and
    the executor trace line the misses of its windows, which Task 6's gate sums (RE_TRACE_MISSES). Both counters start
    at zero once per dispatch."""
    n = norm(sycl)
    lam = body(n, EXEC_SIG)
    return (n.count(PP_MOE_FMT) == 1 and n.count(EXEC_TRACE_FMT) == 1 and lam.count(COUNTERS) == 1
            and ordered(lam, COUNTERS, PHASE_LOOP, PP_MOE_FMT, PP_MOE_ARGS, EXEC_TRACE_FMT, EXEC_TRACE_ARGS))


def test_trace_lines():
    assert claim_trace_lines(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of the try_pp_onednn_f16_batched lambda", body(norm(SYCL), EXEC_SIG),
        COUNTERS, PHASE_LOOP, PP_MOE_FMT, PP_MOE_ARGS, EXEC_TRACE_FMT, EXEC_TRACE_ARGS,
        signature=EXEC_SIG)


def test_trace_lines_mutants():
    c = claim_trace_lines
    assert not c(mutate(SYCL, " misses=%u\"", "\""))
    assert not c(mutate(SYCL, EXEC_TRACE_ARGS, EXEC_TRACE_ARGS.replace("pp_window_misses", "0u")))
    assert not c(mutate(SYCL, " windows=%zu\\n\"", "\\n\""))
    assert not c(mutate(SYCL, PP_MOE_ARGS, PP_MOE_ARGS.replace("pp_windows", "phases.size()")))


# ---- the retired staging's prose -------------------------------------------------------------------------------------

STALE_SYCL = ("moe_stream_phase::stage,", "reuse_after_readers", "which the ring's staging asks too")
STALE_TEST_HPP = ("as stage() does", "its reorders on the backend's queue")


def claim_no_stale_prose(sycl: str, test_hpp: str) -> bool:
    """The comments no longer name the staging Task 2b retired: stage() and its reuse_after_readers gate, a read gate
    behind the down op, the ring's reorders, or a staging that asks the prompt-kernel predicate."""
    return (sycl != "" and test_hpp != "" and not any(s in sycl for s in STALE_SYCL)
            and not any(s in test_hpp for s in STALE_TEST_HPP))


def test_no_stale_prose():
    assert claim_no_stale_prose(SYCL, TEST_HPP), (
        first_present("ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the claim also fails on an empty file)", SYCL,
                      *STALE_SYCL)
        if any(s in SYCL for s in STALE_SYCL) or not SYCL else
        first_present("ggml/src/ggml-sycl/ggml-sycl-test.hpp: whole file (the claim also fails on an empty file)",
                      TEST_HPP, *STALE_TEST_HPP))


@pytest.mark.parametrize("stale", STALE_SYCL)
def test_no_stale_prose_sycl_mutants(stale):
    assert not claim_no_stale_prose(SYCL + "\n// " + stale + "\n", TEST_HPP)


@pytest.mark.parametrize("stale", STALE_TEST_HPP)
def test_no_stale_prose_test_hpp_mutants(stale):
    assert not claim_no_stale_prose(SYCL, TEST_HPP + "\n// " + stale + "\n")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
