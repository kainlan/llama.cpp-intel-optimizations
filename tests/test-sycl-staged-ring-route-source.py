"""Source contract for llama.cpp-o9uw (plan Task 8 of llama.cpp-36j0): STAGED_RING is the one route by which a host
expert the prompt's stream phase staged into a VRAM ring slot answers PRIMARY_DEVICE.

Host-only: reads sources, builds nothing, loads no model and touches no device.

What this gate pins:
  - one writer: `recipe.transfer = moe_recipe_transfer::STAGED_RING` is assigned at exactly one site in the backend,
    inside ggml_sycl_build_moe_resolved_batch's resolver lambda, after the phase's `moe_stream->find(` lookup, and
    before the route core; moe_batch_route carries no `staged_ring` flag (brief amendment: the recipe's transfer is
    the fact);
  - a decode build costs nothing: the phase is taken only for a prompt build (`rows > 1` is the first term of the
    phase pointer's condition), so a decode build computes no layer and no role. The refusal lambda runs only at a
    stage-table hit, and it refuses an entry the batched executor consumes by the entry's consumer, so the build asks
    no claim at all (llama.cpp-3460);
  - the phase is consulted only from the context that owns it, on that context's own queue, outside a graph recording,
    and keyed by the layer and role the phase itself uses (ggml_sycl_moe_stream_layer_of, the plan's role classifier);
  - ruling C (c-3bde): the slot and its layout come from the stage-table entry (`s->layout`), never from the slice's
    tag; the lease is the slot and the ready event is the entry's;
  - lead ruling c-o9uw-1: the slice's tag agrees with the entry by construction. mem_handle::slice(offset, size,
    layout) names the layout only on an owned allocation (it refuses before anything else otherwise), the view applies
    it on every resolve and all four special members carry it; the ring slices each slot with the layout it is
    acquired for, the phase acquires its one slot per matrix as AOS and records AOS (the SOA staging is retired,
    llama.cpp-9433), and the resolver still refuses a slot whose resolved layout is not the entry's;
  - c-las5 (1): a staged expert is served only as a plain MMVQ_COMPAT kernel on the submit queue, for a type that
    does not need its direct kernel. A weight the oneDNN batched prompt executor claims, any other kernel, a Q1_0 or
    NVFP4 matrix, a slice whose resolved layout is not the entry's, or a route either validator refuses falls through
    to the route core. Every lookup is counted served or refused, and every refusal with its cause, in the order
    asked, and with the capability's reason or the validator's reject reason (plan Task 8: the trace says why); the
    capability part is one helper, the one the prompt-kernel predicate answers through;
  - the batched executor consumes a staged slot whose entry names it as the consumer, for an expert it would otherwise
    decline (llama.cpp-3460): its expert loop defers a claimed layer's HOST expert to its window (llama.cpp-8epx),
    and one stage-table lookup per deferred expert after that window's window_begin, keyed as the staged route keys
    it, takes the slot and layout from the entry, the entry's ready event onto the window's barrier, and the slot's
    lease until the done marker;
  - a prompt layer the executor claims leaves the hybrid world (plan Task 10, llama.cpp-3460; W2 of llama.cpp-a2xb):
    ggml_sycl_mul_mat_id asks moe_stream_phase::executor_claims once, after begin_layer decided the layer, for the
    layer begin_layer was given, and the one bool clears plan_has_cpu_experts and joins the resident claim at the
    MMVQ-probe skip and the prompt route; decode keeps `ne12 == 1 ||` and the retained prompt batch keeps
    allow_materialize=false. The resolver asks the same question, once per build, for the executor's handoff;
  - the device recipe is one function shared by a resident expert and a staged slot (one fact, one source);
  - the staged branch keeps no persistent handle: no moe_expert_handles[] write, no remember_moe_storage_handle, no
    handle retention; the transient pointer-table upload clears a staged slot's handle entry and keeps it out of the
    table's active leases, and both upload callers (the batch view and the hybrid route slots) name the staged experts;
  - the staged slots' copies are ordered ahead of every consumer by one device barrier on the context's queue after
    the build (not every consumer chains a ready event); a throw aborts with "[MOE-EXEC] staged ring barrier failed";
  - ruling A (c-3bde): the resolver build, the shared device recipe and the admission predicate wait on nothing
    and query no event status;
  - the prompt admission cache serves a bundle naming a staged slot only within the graph that built it, through one
    predicate used at both lookups, and a role bundle naming one is refused by the fused-path validator;
  - the triad builds and the retained prompt build pass the invocation's context and queue, so the route can answer;
  - the validator requires a STAGED_RING recipe to be PRIMARY_DEVICE on the submit device and queue, with a ready event;
  - the phase reports its served/refused counts once per prompt through the decision log's trace sink, from
    end_prompt, before end_prompt's early return, and under the same guard a second line naming each refusal's cause
    and detail when it refused any;
  - the two executors that cannot read a staged slot refuse a staged batch by name: the non-hybrid retained prompt
    per-expert path, which multiplies each expert at the tensor's route_layout from route.ptr, throws a fallback error
    directly after the batched executor's try, before its scratch setup claims, allocates or can abort, and before its
    first route; try_xmx_sorted_moe rejects it at its head, before it uploads anything;
  - a staged slot's raw pointer does not outlive the ring that holds it, since a raw pointer must not outlive its
    owning handle (llama.cpp-o9uw, c-8jtz): the upload records the payload slot it gives a staged pointer, keyed by
    the weight extra's cache identity and holding no device pointer, the phase counts every ring release in teardown,
    and once the count moves (or the phase is gone) the backend nulls each recorded payload slot that still holds a
    pointer and no live handle, marks the device table invalid and resets the full-local probe, so moe_ptrs_ptr
    answers null until the next upload rewrites the table. It runs after end_prompt at the first decode graph and in
    the test hook, after begin_layer (a new geometry re-reserves the ring), and once the phase is gone in the context
    destructor. Each record is counted on its extra, the context's member documents that a record's extra outlives
    it, and release_extra_gpu asserts no record is left.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, and has a
mutant that must make it fail.

Pytest-style (module-level test_* functions): registered with llama_test_pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, so neither a clang-format
    re-wrap nor a re-split format string can turn a claim red. Idempotent, so a mutant built on normalized text is
    read the same way as the source."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    return t


def read(rel: str) -> str:
    """The file's text, or empty when it does not exist yet, so a missing file fails the claims instead of erroring."""
    p = ROOT / rel
    return p.read_text(errors="replace") if p.exists() else ""


SYCL_REL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
BATCH_HPP_REL = "ggml/src/ggml-sycl/moe-resolved-batch.hpp"
PHASE_REL = "ggml/src/ggml-sycl/moe-stream-plan.cpp"
MEM_HPP_REL = "ggml/src/ggml-sycl/mem-handle.hpp"
MEM_CPP_REL = "ggml/src/ggml-sycl/mem-handle.cpp"
RING_HPP_REL = "ggml/src/ggml-sycl/moe-stream-ring.hpp"
RING_CPP_REL = "ggml/src/ggml-sycl/moe-stream-ring.cpp"
PLAN_HPP_REL = "ggml/src/ggml-sycl/moe-stream-plan.hpp"
COMMON_REL = "ggml/src/ggml-sycl/common.hpp"
COMMON_CPP_REL = "ggml/src/ggml-sycl/common.cpp"

SYCL = read(SYCL_REL)
BATCH_HPP = read(BATCH_HPP_REL)
PHASE = read(PHASE_REL)
MEM_HPP = read(MEM_HPP_REL)
MEM_CPP = read(MEM_CPP_REL)
RING_HPP = read(RING_HPP_REL)
RING_CPP = read(RING_CPP_REL)
PLAN_HPP = read(PLAN_HPP_REL)
COMMON = read(COMMON_REL)
COMMON_CPP = read(COMMON_CPP_REL)
# Every C/C++ source of the SYCL backend outside its tests/, by path from the repository root.
SYCL_TREE = {
    p.relative_to(ROOT).as_posix(): p.read_text(errors="replace")
    for p in sorted((ROOT / "ggml/src/ggml-sycl").rglob("*"))
    if p.is_file() and p.suffix in (".c", ".cpp", ".h", ".hpp") and "tests" not in p.relative_to(ROOT).parts[3:]
}


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). A signature ending in "{" opens the body itself. Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = at + len(signature) - 1 if signature.endswith("{") else text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


def call_args(text: str, name: str) -> list:
    """The argument text of every call `name(...)` (name ends with "("), parentheses balanced, string literals
    skipped. A definition or declaration matches too; callers pick the shape they expect."""
    out = []
    at = text.find(name)
    while at >= 0:
        i = at + len(name)
        depth = 1
        n = len(text)
        start = i
        while i < n and depth:
            ch = text[i]
            if ch in "\"'":
                j = i + 1
                while j < n and text[j] != ch:
                    j += 2 if text[j] == "\\" else 1
                i = j + 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        out.append(text[start:i - 1])
        at = text.find(name, at + len(name))
    return out


def mutate(text: str, old: str, new: str) -> str:
    """`text` (normalized) with `old` replaced once; the mutant must exist, or it would prove nothing."""
    n = norm(text)
    assert old in n, f"mutant precondition: {old!r} is not in the text"
    return n.replace(old, new, 1)


BUILD_SIG = "moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch(const ggml_tensor * src0,"
STAGED_CAP_SIG = ("static ggml_sycl::moe_staged_route_refusal ggml_sycl_moe_staged_capability_refusal(const "
                  "moe_route_capability & cap, ggml_type type)")
XMX_SIG = "static bool try_xmx_sorted_moe(ggml_backend_sycl_context & ctx,"
RECIPE_SIG = "static moe_route_capability ggml_sycl_moe_build_device_recipe("
UPLOAD_SIG = "static const void * const * ggml_sycl_upload_moe_transient_ptr_table("
FROM_BATCH_SIG = "static const void * const * ggml_sycl_upload_moe_ptr_table_from_batch("
MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try"
PREDICATE_SIG = "static bool ggml_sycl_moe_admission_entry_reusable("
ROLE_VALIDATE_SIG = "static bool ggml_sycl_moe_validate_retained_role_bundle("
NAMES_STAGED_SIG = "static bool ggml_sycl_moe_bundle_names_staged_ring("
PREPARE_SIG = "static bool ggml_sycl_moe_stream_prepare("
END_PROMPT_SIG = "void moe_stream_phase::end_prompt()"
REPORT_SIG = "void moe_stream_phase::report_staged_routes()"
VALIDATOR_SIG = "inline bool validate_moe_execution_recipe("
ROUTE_STRUCT_SIG = "struct moe_batch_route {"

ASSIGN = "staged.recipe.transfer = moe_recipe_transfer::STAGED_RING;"
ASSIGN_RE = r"(?<![=!<>])=\s*(?:ggml_sycl::)?moe_recipe_transfer::STAGED_RING\b"
GUARD = "if (moe_stream != nullptr && stream_layer >= 0) {"
FIND = "if (const moe_stream_staged * s = moe_stream->find(stream_layer, expert_id, stream_role)) {"
# An executor entry on a weight the executor takes whole is its handoff, counted apart from the refusals; CLAIMED is
# a refusal only on a weight the ring staged partly (llama.cpp-3460).
HANDOFF = ("if (refusal == moe_staged_route_refusal::CLAIMED && executor_takes_weight()) { "
           "moe_stream->count_executor_handoff(); } else { moe_stream->count_staged_route(refusal, refusal_detail); } ")
SERVE_RETURN = HANDOFF + "if (refusal == moe_staged_route_refusal::NONE) { return staged; }"
# Asked once per build, at the first executor entry, for the build's layer: the answer the prompt path takes after
# begin_layer (STAGED_LAYER_SET).
EXECUTOR_TAKES = ("bool executor_asked = false; bool executor_takes = false; "
                  "const auto executor_takes_weight = [&]() { if (!executor_asked) { executor_asked = true; "
                  "executor_takes = moe_stream->executor_claims(stream_layer); } return executor_takes; };")
ROUTE_CORE = "::ggml_sycl_resolve_moe_expert_route_for_dispatch("
LAMBDA_OPEN = ("auto result = build_moe_resolved_batch(ids, count, slots_per_token, submit_device, "
               "[&](int32_t expert_id) {")

STREAM_DECL = ("moe_stream_phase * moe_stream = rows > 1 && sctx != nullptr && sctx->device == submit_device && "
               "sctx->moe_stream && invocation_queue == sctx->stream() && !g_ggml_sycl_graph_recording ? "
               "sctx->moe_stream.get() : nullptr;")
LAYER_DECL = "const int stream_layer = moe_stream ? ggml_sycl_moe_stream_layer_of(src0) : -1;"
ROLE_DECL = ("const expert_tensor_role stream_role = moe_stream && src0 ? "
             "expert_tensor_role_from_tensor_name(src0->name) : expert_tensor_role::UNKNOWN;")
PREPARE_LAYER = "call.layer = ggml_sycl_moe_stream_layer_of(src0);"
# The batched executor's claims: the route asks neither; the ring's index build asked the second (llama.cpp-3460).
CLAIM_CALLS = ("ggml_sycl_moe_pp_onednn_batched_claims_tensor(",
               "ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor(")
EXECUTOR_REFUSAL = "if (s->consumer == moe_stream_consumer::EXECUTOR) { return moe_staged_route_refusal::CLAIMED; }"

SLOT = "const mem_handle & slot = s->layout == GGML_LAYOUT_SOA ? s->soa : s->aos;"
VIEW = "const resolved_ptr view = slot.resolve(submit_device);"
ENTRY_FIELDS = (
    "staged.residency = moe_batch_residency::PRIMARY_DEVICE;",
    "staged.transient_ptr = view.ptr;",
    "staged.owning_device = submit_device;",
    "staged.planned_device = mem_handle::HOST_DEVICE;",
    "staged.plan_found = true;",
    "staged.planned_on_device = false;",
    "staged.actual_layout = s->layout;",
    "staged.has_ready_event = true;",
    "staged.ready_event = s->ready;",
    "staged.lease = slot;",
    "staged.recipe.layout = s->layout;",
    "staged.recipe.owner_device = submit_device;",
)
REFUSAL_OPEN = ("const char * refusal_detail = nullptr; const moe_staged_route_refusal refusal = [&]() {")
# The resolver's refusals, in the order it asks them; each names its cause and, where there is one, its detail.
REFUSAL_STEPS = (
    EXECUTOR_REFUSAL,
    "const moe_staged_route_refusal by_cap = ggml_sycl_moe_staged_capability_refusal(cap, src0->type);",
    "if (by_cap != moe_staged_route_refusal::NONE) { refusal_detail = cap.reason; return by_cap; }",
    "if (view.ptr == nullptr || !view.on_device) { return moe_staged_route_refusal::VIEW; }",
    "if (view.layout != s->layout) { return moe_staged_route_refusal::LAYOUT; }",
    "if (!staged.recipe.valid) { "
    "refusal_detail = moe_batch_reject_reason_name(moe_batch_reject_reason::RECIPE_MISSING); "
    "return moe_staged_route_refusal::RECIPE; }",
    "const moe_batch_reject_reason route_why = detail::validate_moe_batch_route(staged, submit_device);",
    "if (route_why != moe_batch_reject_reason::NONE) { refusal_detail = moe_batch_reject_reason_name(route_why); "
    "return moe_staged_route_refusal::ROUTE; }",
    "if (!validate_moe_execution_recipe(staged.recipe, moe_admitted_recipe_signature(staged.recipe, staged.lease), "
    "staged.lease, staged.residency, staged.has_ready_event, submit_device, false, 0, &why)) { "
    "refusal_detail = moe_batch_reject_reason_name(why); return moe_staged_route_refusal::RECIPE; }",
    "return moe_staged_route_refusal::NONE; }();",
)
# What a staged slot is served with: the plain MMVQ_COMPAT kernel, for a type with no direct-kernel requirement.
STAGED_CAP_STEPS = (
    "if (!cap.supported) { return ggml_sycl::moe_staged_route_refusal::UNSUPPORTED; }",
    "if (cap.kernel != moe_route_kernel::MMVQ_COMPAT) { return ggml_sycl::moe_staged_route_refusal::KERNEL; }",
    "if (ggml_sycl::moe_type_needs_direct_kernel(type)) { return ggml_sycl::moe_staged_route_refusal::DIRECT_KERNEL; }",
    "return ggml_sycl::moe_staged_route_refusal::NONE;",
)
STAGED_RECIPE_CALL = ("const moe_route_capability cap = ggml_sycl_moe_build_device_recipe(src0, submit_device, rows, "
                      "invocation_backend, invocation_queue, queue_capability, staged);")
# The batched prompt executor (llama.cpp-348c), and every place an active expert's weight pointer goes in it: the
# MXFP4 repacks into the WOQ slot, the f16 dequants into the f16 slot, and the route log, which records the address.
# The pointer's writers are counted apart. oneDNN takes only the executor's own slots.
EXEC_SIG = "auto try_pp_onednn_f16_batched = [&]() -> bool"
EXEC_PTR_WRITE_RE = r"\bexpert\.ptr\s*=(?!=)"
EXEC_PTR_USES = (
    "repack_srcs.push_back(expert.ptr);",
    "repack_mxfp4_xmx_tiled_to_woq_coalesced(expert.ptr,",
    "repack_mxfp4_soa_to_woq_coalesced(expert.ptr,",
    "dequantize_row_mxfp4_soa_to_fp16_rowmajor(expert.ptr,",
    "const void * expert_src = expert.ptr;",
    "append_pp_log_entries(pp_local_entries_for_log, expert.row_begin, expert.row_end, expert.expert_id, expert.ptr,",
)
EXEC_SRC_USES = (
    "dequantize_row_q8_0_soa_to_fp16_rowmajor(expert_src,",
    "expert.to_fp16(expert_src,",
)
EXEC_SRCS_USES = (
    "std::vector<const void *> repack_srcs;",
    "repack_srcs.reserve(active_experts.size());",
    "repack_srcs.push_back(expert.ptr);",
    "repack_mxfp4_xmx_tiled_to_woq_coalesced_batched(repack_srcs.data(),",
    "repack_mxfp4_soa_to_woq_coalesced_batched(repack_srcs.data(),",
)
ONEDNN_CALLS = ("DnnlGemmWrapper::gemm_batch_strided(", "DnnlGemmWrapper::woq_gemm_batch_mxfp4(")
EXPERT_NAMES = ("expert.ptr", "expert_src", "repack_srcs", "active_experts", "route.ptr", "moe_stream")
# The executor's read of a staged slot (llama.cpp-3460): the stage-table key, built as the staged route builds it.
# Since the window ring (llama.cpp-8epx) the expert loop only defers the HOST expert of an op that reads windows
# (pp_stream_windowed: a claimed layer's GATE, UP or DOWN op on the f16 arm), and the phase loop reads its slot after
# that phase's window_begin staged it, through resolve_deferred_expert, onto the window's barrier.
EXEC_STREAM_KEY = (
    "const int pp_stream_layer = ctx.moe_stream ? ggml_sycl_moe_stream_layer_of(src0) : -1;",
    "const ggml_sycl::expert_tensor_role pp_stream_role = ctx.moe_stream ? "
    "ggml_sycl::expert_tensor_role_from_tensor_name(src0->name) : ggml_sycl::expert_tensor_role::UNKNOWN;",
)
EXEC_FIND = "ctx.moe_stream->find(pp_stream_layer, expert.expert_id, pp_stream_role)"
# The deferred branch is a HOST route's only, and only for an op that reads windows: a device route takes the device
# branch, and any other route the core can answer (UNAVAILABLE, REFUSED) is recoverable and is declined like an expert
# the executor cannot read (lead ruling, quality r2). An executor entry is AOS-only, and its view is checked as the
# staged route checks a slot's: on this device, and tagged with the entry's layout (lead ruling c-o9uw-1). A deferred
# expert the window did not stage is declined, never aborted (the 9433 contract), and every decline in the phase loop
# first drains the phases it already submitted.
EXEC_DEFER = ("} else if (pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) { "
              "expert.deferred = true; expert.layout = GGML_LAYOUT_AOS; } else { ")
EXEC_HOST_ONLY = ("return reject_batched(moe_pp_batched_refusal::EXPERT_NOT_LOCAL, "
                  "ggml_sycl_expert_resolve_reason_name(route.reason), i02); }")
EXEC_RESOLVE_SIG = ("auto resolve_deferred_expert = [&](pp_onednn_batched_expert & expert, "
                    "std::vector<sycl::event> & window_ready) -> bool {")
EXEC_ENTRY = ("const bool executor_entry = staged != nullptr && "
              "staged->consumer == ggml_sycl::moe_stream_consumer::EXECUTOR;")
EXEC_NOT_STAGED = ("if (!executor_entry) { drain_submitted_phases(); "
                   "return reject_batched(moe_pp_batched_refusal::EXPERT_NOT_LOCAL, \"window-not-staged\", "
                   "expert.expert_id); }")
EXEC_AOS_ONLY = "GGML_ASSERT(staged->layout == GGML_LAYOUT_AOS);"
EXEC_VIEW_CHECK = ("if (view.ptr == nullptr || !view.on_device) { drain_submitted_phases(); "
                   "return reject_batched(moe_pp_batched_refusal::EXPERT_NOT_LOCAL, \"staged-slot-view\", "
                   "expert.expert_id); }")
EXEC_TAG_CHECK = ("if (view.layout != staged->layout) { drain_submitted_phases(); "
                  "return reject_batched(moe_pp_batched_refusal::EXPERT_LAYOUT, "
                  "ggml_sycl_layout_mode_name(view.layout), expert.expert_id); }")
EXEC_RESOLVE_CALL = ("if (active_experts[i].deferred && !resolve_deferred_expert(active_experts[i], window_ready)) { "
                     "return false; }")
EXEC_STAGED_COUNT = "pp_staged_experts += ctx.moe_stream->window_begin("
EXEC_STAGED_STEPS = (
    "moe_expert_route route = retained_prompt_group_route_for_expert(static_cast<int>(i02));",
    "if (route.kind == moe_expert_route_kind::LOCAL_DEVICE && route.ptr && route.lease.valid()) {",
    "expert.lease = std::move(route.lease);",
    EXEC_DEFER,
    EXEC_HOST_ONLY,
    "active_experts.push_back(std::move(expert));",
    "stream->ext_oneapi_submit_barrier(ready_events);",
    EXEC_RESOLVE_SIG,
    "const ggml_sycl::moe_stream_staged * staged = " + EXEC_FIND + ";",
    EXEC_ENTRY,
    EXEC_NOT_STAGED,
    EXEC_AOS_ONLY,
    "const ggml_sycl::resolved_ptr view = staged->aos.resolve(ctx.device);",
    EXEC_VIEW_CHECK,
    EXEC_TAG_CHECK,
    "window_ready.push_back(staged->ready);",
    "expert.ptr = view.ptr;",
    "expert.layout = staged->layout;",
    "expert.lease = staged->aos;",
    "return true; };",
    EXEC_STAGED_COUNT,
    "std::vector<sycl::event> window_ready;",
    EXEC_RESOLVE_CALL,
    "stream->ext_oneapi_submit_barrier(window_ready);",
)
EXEC_LEASE_HOLD = "leases.push_back(std::move(expert.lease));"
CORE_RECIPE_CALL = ("(void) ggml_sycl_moe_build_device_recipe(src0, submit_device, rows, invocation_backend, "
                    "invocation_queue, queue_capability, normalized);")
CAP_QUERY = "ggml_sycl_moe_query_route_capability("

PERSISTENT = ("moe_expert_handles[", "remember_moe_storage_handle(", "retain_handles_until_event(",
              "ggml_sycl_set_moe_ptr_table_leases(", "tag_persistent_lease_site(")

UPLOAD_STAGED = ("const bool staged = staged_experts != nullptr && std::find(staged_experts->begin(), "
                 "staged_experts->end(), eid) != staged_experts->end();")
UPLOAD_RECORD = "ggml_sycl_moe_stream_record_payload(ctx, extra, slot, static_cast<size_t>(n_experts));"
UPLOAD_CLEAR = ("if (staged) { expert_handles[slot] = ggml_sycl::mem_handle{}; ptr_payload[slot] = resolved.ptr; "
                "any_updated = true; " + UPLOAD_RECORD + " } else if (!same_entry) {")
UPLOAD_LEASE = "if (!staged) { leases.push_back(std::move(handle)); }"
FROM_BATCH_STAGED = "const std::vector<int32_t> staged_experts = ggml_sycl::moe_batch_staged_ring_experts(batch);"
FROM_BATCH_PASS = "table_event_out, table_event_set, staged_experts.empty() ? nullptr : &staged_experts);"
HYBRID_COLLECT = ("if (entry->admitted_recipe_ticket.valid() && entry->admitted_recipe_ticket.recipe().transfer == "
                  "ggml_sycl::moe_recipe_transfer::STAGED_RING &&")
HYBRID_PASS = ("ggml_sycl_upload_moe_transient_ptr_table(ctx, src0, route_slots, layer_id, dispatch_layout, nullptr, "
               "&route_ptrs_event, &route_ptrs_event_set, route_staged_experts.empty() ? nullptr : "
               "&route_staged_experts);")

BARRIER = (
    "if (moe_stream != nullptr && result) {",
    "(void) moe_batch_staged_ring_experts(result.batch, &staged_ready);",
    "sctx->stream()->ext_oneapi_submit_barrier(staged_ready);",
    "catch (const ggml_sycl_fallback_error &) { throw; }",
    'GGML_ABORT("[MOE-EXEC] staged ring barrier failed: %s", e.what());',
    "return result;",
)

PREDICATE_BODY = "{ return entry.valid && (!entry.names_staged_ring || entry.graph_seq == graph_seq); }"
EARLY_LOOKUP = ("early_it != g_moe_prompt_admission_cache.end() && "
                "ggml_sycl_moe_admission_entry_reusable(early_it->second, ctx.graph_compute_seq)")
MAIN_LOOKUP = ("admit_it != g_moe_prompt_admission_cache.end() && "
               "ggml_sycl_moe_admission_entry_reusable(admit_it->second, ctx.graph_compute_seq)")
POPULATE = ("new_entry.names_staged_ring = ggml_sycl_moe_bundle_names_staged_ring("
            "new_entry.retained_prompt_roles_result.bundle); new_entry.graph_seq = ctx.graph_compute_seq;")
ROLE_REFUSE = "if (ggml_sycl_moe_bundle_names_staged_ring(roles)) { return false; }"
NAMES_STAGED_BODY = ("{ return ggml_sycl::moe_batch_names_staged_ring(roles.gate.batch) || "
                     "ggml_sycl::moe_batch_names_staged_ring(roles.up.batch) || "
                     "ggml_sycl::moe_batch_names_staged_ring(roles.down.batch); }")

TRIAD_ROLES = ("gate_layout", "up_layout", "down_layout")
RETAINED_PROMPT = ("retained_prompt_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(src0, ctx.device, "
                   "prompt_ids_snapshot.data(), prompt_ids_snapshot.size(), static_cast<size_t>(ids->ne[0]), "
                   "retained_prompt_layout, false, &ctx, ctx.stream());")

VALIDATOR_CLAUSE = ("if (recipe.transfer == moe_recipe_transfer::STAGED_RING && (recipe.kind != "
                    "moe_batch_executor::PRIMARY_DEVICE || residency != moe_batch_residency::PRIMARY_DEVICE || "
                    "!has_ready_event || !resolved.on_device || recipe.owner_device != submit_device || "
                    "recipe.queue != moe_recipe_queue::SUBMIT)) { "
                    "return refuse(moe_batch_reject_reason::RECIPE_MISMATCH); }")
VALIDATOR_PARAM = "moe_batch_residency residency, bool has_ready_event, int submit_device,"
WORKSPACE_REFUSE = "return refuse(moe_batch_reject_reason::WORKSPACE_UNDERSIZED);"

TRACE_FMT = "[MOE-EXEC-TRACE] staged route device %d: served=%zu handed_off=%zu refused=%zu"
TALLY_ARGS = "device_, staged_routes_.served, staged_routes_.handed_off, staged_routes_.refused);"
TRACE_EMIT = "log_->trace_sink(log_->sink_ctx, line);"
TRACE_REFUSALS_FMT = "[MOE-EXEC-TRACE] staged route refusals device %d: "
TRACE_REFUSALS = (
    "if (staged_routes_.refused != 0) {",
    TRACE_REFUSALS_FMT,
    "const std::string causes = head + staged_routes_.refusals_text();",
    "log_->trace_sink(log_->sink_ctx, causes.c_str()); } }",
    "staged_routes_ = moe_staged_route_tally(); }",
)
COUNT_SIG = "void moe_stream_phase::count_staged_route(moe_staged_route_refusal refusal, const char * detail)"
COUNT_BODY = "{ staged_routes_.count(refusal, detail); }"
HANDOFF_SIG = "void moe_stream_phase::count_executor_handoff()"
HANDOFF_BODY = "{ staged_routes_.count_handoff(); }"
TALLY_HANDED_OFF = "size_t handed_off = 0;"
TALLY_HANDOFF_SIG = "void moe_staged_route_tally::count_handoff()"
TALLY_HANDOFF_BODY = "{ ++handed_off; }"
END_PROMPT_HEAD = "{ report_staged_routes(); if (!ubatch_live_ && !ring_ && !pending_ring_ && table_.empty()) {"
DTOR_SIG = "moe_stream_phase::~moe_stream_phase()"
DTOR_HEAD = "{ report_staged_routes(); teardown("
REPORT_HEAD = ("{ if ((staged_routes_.served != 0 || staged_routes_.handed_off != 0 || "
               "staged_routes_.refused != 0) && log_ != nullptr && "
               "log_->trace >= 1 && log_->trace_sink != nullptr) {")
REPORT_RESET = "staged_routes_ = moe_staged_route_tally();"
REFUSALS_WARN_FMT = "[MOE-EXEC] staged route refusals device %d: served=%zu handed_off=%zu refused=%zu "
REFUSALS_WARN = (
    ("if (staged_routes_.refused != 0 && policy_ == moe_exec_policy::RING && log_ != nullptr && "
     "log_->sink != nullptr) {"),
    REFUSALS_WARN_FMT + '", ' + TALLY_ARGS,
    "const std::string warn_line = warn_head + staged_routes_.refusals_text();",
    "log_->sink(log_->sink_ctx, warn_line.c_str()); }",
    REPORT_RESET + " }",
)
BEGIN_LAYER_SIG = "moe_exec_decision moe_stream_phase::begin_layer("
POLICY_MEMBER = "moe_exec_policy policy_ = moe_exec_policy::AUTO;"
POLICY_SET = "policy_ = op.policy;"

WAIT_RE = re.compile(r"\bwait\s*\(|\bwait_and_throw\s*\(|\bget_info\s*<")


def build_body(sycl: str) -> str:
    return body(norm(sycl), BUILD_SIG)


def staged_branch(sycl: str) -> str:
    """The text of the staged branch: the block of GUARD (a phase pointer and a layer) inside the build function. A
    decode build never enters it: `rows > 1` is the first term of the phase pointer's own condition (STREAM_DECL)."""
    b = build_body(sycl)
    at = b.find(GUARD)
    return body(b[at:], GUARD) if at >= 0 else ""


# ---- one writer, one place -------------------------------------------------------------------------------------------


def claim_single_writer(sycl: str, tree: dict, batch_hpp: str) -> bool:
    """STAGED_RING is assigned once in the whole backend, in the build function's lambda: after the prompt guard
    and the stage-table lookup, before the route core. moe_batch_route has no staged_ring flag."""
    writers = sum(len(re.findall(ASSIGN_RE, norm(text))) for text in tree.values())
    b = build_body(sycl)
    lam = b[b.find(LAMBDA_OPEN):] if LAMBDA_OPEN in b else ""
    route = body(norm(batch_hpp), ROUTE_STRUCT_SIG)
    return (writers == 1 and len(re.findall(ASSIGN_RE, b)) == 1
            and ordered(lam, LAMBDA_OPEN, GUARD, FIND, ASSIGN, SERVE_RETURN, ROUTE_CORE)
            and route != "" and "staged_ring" not in route)


def test_single_writer():
    assert claim_single_writer(SYCL, SYCL_TREE, BATCH_HPP)


def test_single_writer_mutants():
    # A second writer elsewhere in the backend.
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = "void f(moe_execution_recipe & r) { r.transfer = moe_recipe_transfer::STAGED_RING; }"
    assert not claim_single_writer(SYCL, tree, BATCH_HPP)
    # The staged branch outside the lambda's phase guard.
    m = mutate(SYCL, GUARD, "if (stream_layer >= 0) {")
    assert not claim_single_writer(m, {SYCL_REL: m}, BATCH_HPP)
    # The route core runs first and the staged branch only after it.
    m = mutate(SYCL, LAMBDA_OPEN, LAMBDA_OPEN + " (void) ::ggml_sycl_resolve_moe_expert_route_for_dispatch(src0);")
    m = m.replace(ROUTE_CORE + "src0, submit_device", "other_core(src0, submit_device", 1)
    assert not claim_single_writer(m, {SYCL_REL: m}, BATCH_HPP)
    # A flag on the route instead of the recipe's transfer.
    hpp = mutate(BATCH_HPP, ROUTE_STRUCT_SIG, ROUTE_STRUCT_SIG + " bool staged_ring = false;")
    assert not claim_single_writer(SYCL, SYCL_TREE, hpp)


# ---- the phase is consulted from its owner, keyed as the phase keys -------------------------------------------------


def claim_owner_and_keys(sycl: str) -> bool:
    """The phase is the invocation context's own, on its own queue, outside a recording; the layer comes from the same
    helper the phase's begin_layer gets it from, and the role from the plan's classifier."""
    n = norm(sycl)
    b = build_body(sycl)
    prep = body(n, PREPARE_SIG)
    return (ordered(b, STREAM_DECL, LAYER_DECL, ROLE_DECL, LAMBDA_OPEN) and PREPARE_LAYER in prep
            and "parse_layer_id_from_name(" not in prep)


def test_owner_and_keys():
    assert claim_owner_and_keys(SYCL)


def test_owner_and_keys_mutants():
    assert not claim_owner_and_keys(mutate(SYCL, " && !g_ggml_sycl_graph_recording ?", " ?"))
    assert not claim_owner_and_keys(mutate(SYCL, "invocation_queue == sctx->stream() && ", ""))
    assert not claim_owner_and_keys(mutate(SYCL, "sctx->device == submit_device && ", ""))
    assert not claim_owner_and_keys(
        mutate(SYCL, LAYER_DECL, "const int stream_layer = moe_stream ? parse_layer_id_from_name(src0->name) : -1;"))
    assert not claim_owner_and_keys(mutate(SYCL, PREPARE_LAYER, "call.layer = parse_layer_id_from_name(src0->name);"))


# ---- ruling C and c-las5 ---------------------------------------------------------------------------------------------


def claim_entry_is_the_fact(sycl: str) -> bool:
    """Ruling C (c-3bde): the slot, its layout, the lease and the ready event are the stage-table entry's."""
    br = staged_branch(sycl)
    return ordered(br, FIND, SLOT, VIEW) and all(f in br for f in ENTRY_FIELDS)


def test_entry_is_the_fact():
    assert claim_entry_is_the_fact(SYCL)


def test_entry_is_the_fact_mutants():
    assert not claim_entry_is_the_fact(
        mutate(SYCL, "staged.actual_layout = s->layout;", "staged.actual_layout = view.layout;"))
    assert not claim_entry_is_the_fact(
        mutate(SYCL, "staged.recipe.layout = s->layout;", "staged.recipe.layout = view.layout;"))
    assert not claim_entry_is_the_fact(mutate(SYCL, SLOT, "const mem_handle & slot = s->aos;"))
    assert not claim_entry_is_the_fact(
        mutate(SYCL, "staged.ready_event = s->ready;", "staged.ready_event = view.ready_event;"))


def executor_reads_through_plain_kernels(sycl: str) -> bool:
    """The batched executor's side of c-las5 (llama.cpp-3460): an active expert's weight pointer, a staged ring slot's
    among them, goes only to the named repack and dequant kernels (and the route log), each place once; oneDNN is
    handed no expert pointer, only the executor's f16 and WOQ slots."""
    lam = body(norm(sycl), EXEC_SIG)
    if lam == "":
        return False
    writes = len(re.findall(EXEC_PTR_WRITE_RE, lam))
    uses_ok = (all(lam.count(u) == 1 for u in EXEC_PTR_USES + EXEC_SRC_USES)
               and all(lam.count(u) == 1 for u in EXEC_SRCS_USES)
               and lam.count("expert.ptr") == writes + len(EXEC_PTR_USES)
               and lam.count("expert_src") == 1 + len(EXEC_SRC_USES)
               and lam.count("repack_srcs") == len(EXEC_SRCS_USES))
    onednn = [a for name in ONEDNN_CALLS for a in call_args(lam, name)]
    return (uses_ok and writes >= 1 and len(onednn) == len(ONEDNN_CALLS)
            and not any(e in a for a in onednn for e in EXPERT_NAMES))


def claim_served_only_as_plain_kernel(sycl: str) -> bool:
    """c-las5 (1): a ring slot is read only by plain kernels on the submit queue, never by a library. It has two
    readers, pinned by name (llama.cpp-3460). The route serves a slot only as MMVQ_COMPAT, for a type with no
    direct-kernel requirement, never for an entry the batched executor consumes, never for a slice whose layout is not
    the entry's, and only when both validators pass. Every lookup is counted, a refusal with its cause and detail, in
    the order asked (plan Task 8: the trace says why), and the slot is served only when nothing refused it. The
    capability part is one helper, which names MMVQ_COMPAT and no other kernel. An entry the executor consumes is read
    by the executor's dequant kernels (executor_reads_through_plain_kernels), never handed to oneDNN."""
    br = staged_branch(sycl)
    at = br.find(REFUSAL_OPEN)
    lam = body(br[at:], "const moe_staged_route_refusal refusal = [&]()") if at >= 0 else ""
    cap = body(norm(sycl), STAGED_CAP_SIG)
    return (ordered(br, ASSIGN, REFUSAL_OPEN, *REFUSAL_STEPS, SERVE_RETURN)
            and lam.count("return moe_staged_route_refusal::NONE;") == 1 and br.count("return staged;") == 1
            and ordered(cap, *STAGED_CAP_STEPS) and re.findall(r"moe_route_kernel::(\w+)", cap) == ["MMVQ_COMPAT"]
            and norm(sycl).count(STAGED_CAP_SIG + " {") == 1 and executor_reads_through_plain_kernels(sycl))


def test_served_only_as_plain_kernel():
    assert claim_served_only_as_plain_kernel(SYCL)


def test_served_only_as_plain_kernel_mutants():
    c = claim_served_only_as_plain_kernel
    # An entry the batched executor consumes served to the route.
    assert not c(mutate(SYCL, REFUSAL_STEPS[0], ""))
    # Another kernel served.
    assert not c(mutate(SYCL, "if (cap.kernel != moe_route_kernel::MMVQ_COMPAT) {",
                        "if (cap.kernel != moe_route_kernel::MMVQ_COMPAT && "
                        "cap.kernel != moe_route_kernel::XMX_TILED) {"))
    # A Q1_0/NVFP4 slot staged and served past the capability helper.
    assert not c(mutate(SYCL, STAGED_CAP_STEPS[2], ""))
    # A slot whose resolved layout is not the entry's.
    assert not c(mutate(SYCL, REFUSAL_STEPS[4], ""))
    # The route validator skipped.
    assert not c(mutate(SYCL, REFUSAL_STEPS[7], ""))
    # A refusal counted without its detail, so the trace cannot say why.
    assert not c(mutate(SYCL, "refusal_detail = cap.reason; return by_cap;", "return by_cap;"))
    assert not c(mutate(SYCL, "moe_stream->count_staged_route(refusal, refusal_detail);",
                        "moe_stream->count_staged_route(refusal, nullptr);"))
    # Served whatever the cause.
    assert not c(mutate(SYCL, "if (refusal == moe_staged_route_refusal::NONE) { return staged; }",
                        "return staged;"))
    # The executor hands oneDNN an expert's pointer, a staged slot's among them, instead of its f16 slot.
    assert not c(mutate(SYCL, "1.0f, phase_weights + (lo - phase.begin) * weight_elems,",
                        "1.0f, static_cast<const sycl::half *>(active_experts[lo].ptr),"))
    # The WOQ GEMM handed the repack sources instead of the WOQ slot.
    assert not c(mutate(SYCL, "static_cast<int64_t>(act_slot_elems), batched_weight_bytes + lo * weight_slot_bytes,",
                        "static_cast<int64_t>(act_slot_elems), (const uint8_t *) repack_srcs[lo],"))
    # A new reader of the expert's pointer beside the pinned kernels.
    assert not c(mutate(SYCL, "const void * expert_src = expert.ptr;",
                        "const void * expert_src = expert.ptr; ggml_sycl_onednn_read(expert.ptr);"))
    # The dequant reads its source through a name the pin does not follow.
    assert not c(mutate(SYCL, "expert.to_fp16(expert_src,", "expert.to_fp16(expert.lease.resolve(ctx.device).ptr,"))


def claim_executor_entry_refused_by_its_consumer(sycl: str) -> bool:
    """The route refuses an entry the batched executor consumes by the entry's own consumer, the fact the ring wrote
    when it staged the slot (one fact, one source, c-gi44), as the first step of its refusal lambda. It asks no claim
    of the executor's: neither claim predicate is called in the build, so a build pays no plan read for it, and the
    staged claim stays the index build's alone (llama.cpp-3460)."""
    b = build_body(sycl)
    lam_at = staged_branch(sycl).find(REFUSAL_OPEN)
    lam = staged_branch(sycl)[lam_at:] if lam_at >= 0 else ""
    return (lam.find(EXECUTOR_REFUSAL) >= 0 and lam.find(EXECUTOR_REFUSAL) == lam.find("if (", len(REFUSAL_OPEN))
            and not any(c in b for c in CLAIM_CALLS) and "stream_library_claimed" not in b
            and "stream_claim_asked" not in b)


def test_executor_entry_refused_by_its_consumer():
    assert claim_executor_entry_refused_by_its_consumer(SYCL)


def test_executor_entry_refused_by_its_consumer_mutants():
    c = claim_executor_entry_refused_by_its_consumer
    # The route asks the plan claim again (the shape before llama.cpp-3460).
    assert not c(mutate(SYCL, EXECUTOR_REFUSAL,
                        "if (ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, submit_device)) { "
                        "return moe_staged_route_refusal::CLAIMED; }"))
    # The route re-derives the consumer from the slot's layout instead of reading the entry's.
    assert not c(mutate(SYCL, EXECUTOR_REFUSAL,
                        "if (s->layout == GGML_LAYOUT_AOS) { return moe_staged_route_refusal::CLAIMED; }"))
    # The staged claim asked in the build beside the entry's consumer.
    assert not c(mutate(SYCL, EXECUTOR_REFUSAL,
                        "if (ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor(*sctx, src0, submit_device)) { "
                        "return moe_staged_route_refusal::CLAIMED; } " + EXECUTOR_REFUSAL))
    # The refusal asked after the capability, so a cause other than the consumer is counted for the executor's entry.
    m = mutate(SYCL, EXECUTOR_REFUSAL + " ", "")
    m = m.replace("if (view.ptr == nullptr", EXECUTOR_REFUSAL + " if (view.ptr == nullptr", 1)
    assert not c(m)


def claim_executor_handoff_counted_apart(sycl: str, phase: str, hpp: str) -> bool:
    """An executor entry is a refusal only when the executor will not read it (llama.cpp-3460). On a weight the
    executor takes (moe_stream_phase::executor_claims: the layer's RING decision stands, the executor claims every host
    matrix of it and the ring is reserved, the answer the prompt path takes after begin_layer), the entry is the
    executor's handoff: counted apart from the refusals, in the tally's own field, so the report's WARN stays for real
    refusals. The build asks executor_claims once, lazily, at its first executor entry, before the lambda, for its own
    layer. Either way the expert falls through to the route core, whose HOST answer the executor's expert loop takes
    to the stage table."""
    b = build_body(sycl)
    br = staged_branch(sycl)
    n = norm(phase)
    return (b.count("->executor_claims(") == 1 and "->executor_staged(" not in b and b.count(EXECUTOR_TAKES) == 1
            and ordered(b, EXECUTOR_TAKES, LAMBDA_OPEN) and br.count("executor_takes_weight()") == 1
            and ordered(br, REFUSAL_OPEN, EXECUTOR_REFUSAL, SERVE_RETURN)
            and body(n, HANDOFF_SIG) == HANDOFF_BODY and body(n, TALLY_HANDOFF_SIG) == TALLY_HANDOFF_BODY
            and norm(hpp).count(TALLY_HANDED_OFF) == 1)


def test_executor_handoff_counted_apart():
    assert claim_executor_handoff_counted_apart(SYCL, PHASE, PLAN_HPP)


def test_executor_handoff_counted_apart_mutants():
    def c(sycl=SYCL, phase=PHASE, hpp=PLAN_HPP):
        return claim_executor_handoff_counted_apart(sycl, phase, hpp)
    # Every executor entry counted as a handoff, a partly staged weight's too, so its wasted copies never warn.
    assert not c(mutate(SYCL, "refusal == moe_staged_route_refusal::CLAIMED && executor_takes_weight()",
                        "refusal == moe_staged_route_refusal::CLAIMED"))
    # Every executor entry counted as a refusal again (the shape before the handoff field).
    assert not c(mutate(SYCL, HANDOFF, "moe_stream->count_staged_route(refusal, refusal_detail); "))
    # The phase asked per expert instead of once per build.
    assert not c(mutate(SYCL, "&& executor_takes_weight()", "&& moe_stream->executor_claims(stream_layer)"))
    # The phase asked about another layer than the build's.
    assert not c(mutate(SYCL, "executor_claims(stream_layer);", "executor_claims(stream_layer + 1);"))
    # The retired per-ids stage-table question back in its place.
    assert not c(mutate(SYCL, "executor_takes = moe_stream->executor_claims(stream_layer);",
                        "executor_takes = moe_stream->executor_staged(stream_layer, stream_role, ids, count, "
                        "static_cast<int>(src0->ne[2]));"))
    # A handoff counted as a refusal by the phase.
    assert not c(phase=mutate(PHASE, HANDOFF_BODY,
                              "{ staged_routes_.count(moe_staged_route_refusal::CLAIMED, nullptr); }"))
    # A handoff counted in the tally's refused field.
    assert not c(phase=mutate(PHASE, TALLY_HANDOFF_SIG + " " + TALLY_HANDOFF_BODY,
                              TALLY_HANDOFF_SIG + " { ++refused; }"))
    # The tally without its own field.
    assert not c(hpp=mutate(PLAN_HPP, TALLY_HANDED_OFF, ""))


def claim_executor_reads_staged_slots(sycl: str) -> bool:
    """The batched executor consumes a staged slot for an expert it would otherwise decline (llama.cpp-3460), from the
    ring window its phase stages (llama.cpp-8epx). Its expert loop asks no stage table: on a layer the executor claims
    it defers a HOST route's expert (marked, AOS), and declines any other non-device expert as EXPERT_NOT_LOCAL. The
    phase loop asks the stage table once per deferred expert, after that phase's window_begin, keyed by the layer and
    role the staged route keys it with. It takes an entry only when the entry's consumer is the executor (the fact the
    ring wrote), and the slot and its layout from the entry (ruling C). It pushes the entry's ready event onto the
    window's barrier, holds the slot's lease like any expert's until the done marker, and counts window_begin's staged
    answer for the trace and the refusal WARN. A deferred expert the window did not stage is declined, never aborted.
    The entry is AOS-only (asserted; no SOA arm), and its view is checked as the staged route checks a slot's: on this
    device (EXPERT_NOT_LOCAL, staged-slot-view) and tagged with the entry's layout (EXPERT_LAYOUT). Each decline drains
    the phases already submitted first."""
    lam = body(norm(sycl), EXEC_SIG)
    return (lam != "" and lam.count("->find(") == 1 and lam.count(EXEC_FIND) == 1 and "staged->soa" not in lam
            and "route.kind == moe_expert_route_kind::HOST);" not in lam and "GGML_ASSERT(executor_entry" not in lam
            and lam.count("expert.deferred = true;") == 1 and lam.count("resolve_deferred_expert") == 2
            and lam.count(EXEC_STAGED_COUNT) == 1 and "++pp_staged_experts" not in lam
            and all(lam.count(k) == 1 for k in EXEC_STREAM_KEY) and ordered(lam, *EXEC_STREAM_KEY, *EXEC_STAGED_STEPS)
            and lam.count("window_ready.push_back(staged->ready);") == 1 and EXEC_LEASE_HOLD in lam)


def test_executor_reads_staged_slots():
    assert claim_executor_reads_staged_slots(SYCL)


def test_executor_reads_staged_slots_mutants():
    c = claim_executor_reads_staged_slots
    # The executor declines a staged expert without asking the table (the shape before llama.cpp-3460).
    assert not c(mutate(SYCL, EXEC_FIND, "nullptr"))
    # The slot taken without its copy's ready event, so the dequant may read it before the copy lands.
    assert not c(mutate(SYCL, "window_ready.push_back(staged->ready);", ""))
    # The window's ready events gathered but no barrier submitted on them.
    assert not c(mutate(SYCL, "if (!window_ready.empty()) { stream->ext_oneapi_submit_barrier(window_ready); }", ""))
    # An entry the route serves taken too: the consumer the ring wrote is not asked.
    assert not c(mutate(SYCL, "staged != nullptr && staged->consumer == ggml_sycl::moe_stream_consumer::EXECUTOR;",
                        "staged != nullptr;"))
    # The slot's lease not held, so the ring's release could free it under the dequant.
    assert not c(mutate(SYCL, "expert.lease = staged->aos;", ""))
    # An expert deferred under a route other than HOST (UNAVAILABLE, REFUSED), so the executor reads a slot whose
    # expert the route core could not place.
    assert not c(mutate(SYCL, "pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) {",
                        "pp_stream_windowed) {"))
    # A host expert deferred for an op that reads no window, where no window is staged for it.
    assert not c(mutate(SYCL, "pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) {",
                        "route.kind == moe_expert_route_kind::HOST) {"))
    # Deferred on the layer's claim instead: a claimed layer's fused or WOQ-arm op would defer a host expert no window
    # stages.
    assert not c(mutate(SYCL, "pp_stream_windowed && route.kind == moe_expert_route_kind::HOST) {",
                        "pp_onednn_batched_claims_staged_layer && route.kind == moe_expert_route_kind::HOST) {"))
    # A device expert deferred as well.
    assert not c(mutate(SYCL, "expert.lease = std::move(route.lease); } else if",
                        "expert.lease = std::move(route.lease); expert.deferred = true; } else if"))
    # The recoverable route aborts instead of declining (the r1 shape).
    assert not c(mutate(SYCL, "} else { " + EXEC_HOST_ONLY,
                        "} else { GGML_ASSERT(route.kind == moe_expert_route_kind::HOST); " + EXEC_HOST_ONLY))
    # A deferred expert the window did not stage aborts instead of declining (the 9433 contract).
    assert not c(mutate(SYCL, EXEC_NOT_STAGED, "GGML_ASSERT(executor_entry);"))
    # A decline in the phase loop that leaves the phases it already submitted running.
    assert not c(mutate(SYCL, "if (!executor_entry) { drain_submitted_phases(); return",
                        "if (!executor_entry) { return"))
    assert not c(mutate(SYCL, "if (view.ptr == nullptr || !view.on_device) { drain_submitted_phases(); return",
                        "if (view.ptr == nullptr || !view.on_device) { return"))
    # The dead SOA arm back: an executor entry is AOS-only.
    assert not c(mutate(SYCL, "staged->aos.resolve(ctx.device);",
                        "(staged->layout == GGML_LAYOUT_SOA ? staged->soa : staged->aos).resolve(ctx.device);"))
    assert not c(mutate(SYCL, EXEC_AOS_ONLY, ""))
    # A slot that does not resolve on this device taken.
    assert not c(mutate(SYCL, EXEC_VIEW_CHECK, EXEC_VIEW_CHECK.replace(" || !view.on_device", "")))
    # A slot whose tag is not the entry's layout taken.
    assert not c(mutate(SYCL, EXEC_TAG_CHECK, ""))
    # The layout re-derived from the route instead of read from the entry.
    assert not c(mutate(SYCL, "expert.layout = staged->layout;", "expert.layout = route_layout;"))
    # The table asked for every expert of the phase, a device one's too.
    assert not c(mutate(SYCL, "if (active_experts[i].deferred && !resolve_deferred_expert(",
                        "if (!resolve_deferred_expert("))
    # The table asked in the expert loop again, ahead of the window that stages the slot (the shape before 8epx).
    m = mutate(SYCL, "moe_expert_route route = retained_prompt_group_route_for_expert(static_cast<int>(i02));",
               "(void) ctx.moe_stream->find(pp_stream_layer, static_cast<int>(i02), pp_stream_role); "
               "moe_expert_route route = retained_prompt_group_route_for_expert(static_cast<int>(i02));")
    assert not c(m)
    # Keyed by the phase-local index instead of the expert's id.
    assert not c(mutate(SYCL, EXEC_FIND, "ctx.moe_stream->find(pp_stream_layer, static_cast<int>(i), pp_stream_role)"))
    # Keyed by another layer than the staged route's.
    assert not c(mutate(SYCL, "ctx.moe_stream ? ggml_sycl_moe_stream_layer_of(src0) : -1;",
                        "ctx.moe_stream ? moe_cache_layer_id(src0->name) : -1;"))
    # The staged slots not counted, so a decline after them stays quiet.
    assert not c(mutate(SYCL, EXEC_STAGED_COUNT, "(void) ctx.moe_stream->window_begin("))
    # Counted per resolved expert instead of from window_begin's answer.
    assert not c(mutate(SYCL, "expert.lease = staged->aos; return true; };",
                        "expert.lease = staged->aos; ++pp_staged_experts; return true; };"))


# ---- a fully staged prompt layer leaves the hybrid world -------------------------------------------------------------

# plan Task 10 (llama.cpp-3460), W2 of llama.cpp-a2xb: one bool, set once after begin_layer decided the layer, from the
# phase's executor_claims: the consumer the index build recorded from the staged claim, the layer's decision and the
# ring. The claim is not asked again on that line (one fact, one source). The bool clears plan_has_cpu_experts and joins
# the resident claim at the MMVQ-probe skip and at the prompt route, so the op reaches the batched executor in the
# non-hybrid world.
STAGED_LAYER_DECL = "bool pp_onednn_batched_claims_staged_layer = false;"
STAGED_LAYER_BEGIN = ("if (ids->ne[1] > 1 && ggml_sycl_moe_stream_prepare(ctx, src0, ids, stream_call)) { "
                      "ctx.moe_stream->begin_layer(")
STAGED_LAYER_SET = "pp_onednn_batched_claims_staged_layer = ctx.moe_stream->executor_claims(stream_call.layer);"
STAGED_LAYER_ASSIGN = "pp_onednn_batched_claims_staged_layer = ctx.moe_stream->"
# The release site stays directly after begin_layer (claim_staged_pointer_released_with_ring); the bool follows it,
# inside the same block, the claim's own (stream_call lives at function scope since llama.cpp-8epx).
STAGED_LAYER_SITE = ("prompt_ids_snapshot.size(), stream_call.op); ggml_sycl_moe_stream_release_payload(ctx); "
                     + STAGED_LAYER_SET + " }")
RETAINED_BUILD = ("retained_prompt_batch_result = ggml_sycl::ggml_sycl_build_moe_resolved_batch(src0, ctx.device, "
                  "prompt_ids_snapshot.data(), prompt_ids_snapshot.size(), static_cast<size_t>(ids->ne[0]), "
                  "retained_prompt_layout, false, &ctx, ctx.stream());")
PROBE_CLAIM = ("const bool pp_onednn_batched_claims_op = ne12 > 1 && (pp_onednn_batched_claims_staged_layer || "
               "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device));")
PLAN_HAS_CPU = ("plan_has_cpu_experts = plan_owner->has_host_experts(src0->name, n_exp, ctx.device) && "
                "!pp_onednn_batched_claims_staged_layer;")
ROUTE_CLAIM = ("const bool pp_onednn_batched_route_claims_prompt = prompt_batch && "
               "(pp_onednn_batched_claims_staged_layer || "
               "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device));")
HYBRID_WITH_PLAN = "const bool moe_hybrid_with_plan = ne12 == 1 || selected_hybrid_route;"


def claim_fully_staged_layer_leaves_hybrid(sycl: str) -> bool:
    """A prompt op whose layer the batched executor claims, its host experts to be read from ring windows it stages,
    leaves the hybrid world (plan Task 10, llama.cpp-3460; W2 of llama.cpp-a2xb). ggml_sycl_mul_mat_id asks the phase
    exactly once (`->executor_claims(`), after begin_layer decided the layer, keyed by the layer the phase decides
    under. The phase answers from the consumer its index build recorded from the staged claim, the layer's decision and
    the ring, so the bool's line asks no claim: a layer with a matrix the executor does not claim stays hybrid because
    that matrix stages NONE. The one bool clears plan_has_cpu_experts and joins the resident claim at the MMVQ-probe
    skip and at the prompt route; decode keeps its unconditional `ne12 == 1 ||` term, and the retained prompt batch
    still builds with allow_materialize=false."""
    m = body(norm(sycl), MMID_SIG)
    at = m.find(STAGED_LAYER_ASSIGN)
    assign = m[at:m.find(";", at) + 1] if at >= 0 else ""
    return (m != "" and m.count("->executor_claims(") == 1 and "->executor_staged(" not in m
            and "->fully_staged(" not in m
            and "claims_staged_tensor(" not in assign and "claims_tensor(" not in assign
            and m.count(STAGED_LAYER_DECL) == 1
            and m.count(STAGED_LAYER_SET) == 1 and m.count(STAGED_LAYER_SITE) == 1
            and m.count("pp_onednn_batched_claims_staged_layer =") == 2
            and ordered(m, STAGED_LAYER_DECL, STAGED_LAYER_BEGIN, STAGED_LAYER_SITE, RETAINED_BUILD, PROBE_CLAIM,
                        PLAN_HAS_CPU, ROUTE_CLAIM, HYBRID_WITH_PLAN))


def test_fully_staged_layer_leaves_hybrid():
    assert claim_fully_staged_layer_leaves_hybrid(SYCL)


def test_fully_staged_layer_leaves_hybrid_mutants():
    c = claim_fully_staged_layer_leaves_hybrid
    # The claim asked again on the bool's line: a second source for the consumer the index build recorded.
    assert not c(mutate(SYCL, STAGED_LAYER_ASSIGN + "executor_claims(",
                        "pp_onednn_batched_claims_staged_layer = ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor("
                        "*stream_call.plan, src0, ctx.device) && ctx.moe_stream->executor_claims("))
    # The layer-wide stage-table question back in its place: it never reads the recorded consumer.
    assert not c(mutate(SYCL, "ctx.moe_stream->executor_claims(stream_call.layer)",
                        "ctx.moe_stream->fully_staged(stream_call.layer)"))
    # The retired per-ids question back: it asks whether begin_layer staged the ids, which it no longer does.
    assert not c(mutate(SYCL, "ctx.moe_stream->executor_claims(stream_call.layer)",
                        "ctx.moe_stream->executor_staged(stream_call.layer, "
                        "ggml_sycl::expert_tensor_role_from_tensor_name(src0->name), prompt_ids_snapshot.data(), "
                        "prompt_ids_snapshot.size(), stream_call.op.n_expert)"))
    # The stage table asked a second time, on the plan_has_cpu_experts line, as the brief's one-liner wrote it.
    assert not c(mutate(SYCL, PLAN_HAS_CPU,
                        "plan_has_cpu_experts = plan_owner->has_host_experts(src0->name, n_exp, ctx.device) && "
                        "!pp_onednn_batched_claims_staged_layer && !(ctx.moe_stream && "
                        "ctx.moe_stream->executor_claims(stream_call_layer));"))
    # Keyed by another layer than the one the phase stages under.
    assert not c(mutate(SYCL, "executor_claims(stream_call.layer)", "executor_claims(moe_cache_layer_id(src0->name))"))
    # The phase asked before begin_layer decided the layer, so the first op of a layer always stays hybrid.
    m = mutate(SYCL, " " + STAGED_LAYER_SET + " }", " }")
    early = STAGED_LAYER_BEGIN.replace("ctx.moe_stream->begin_layer(",
                                       STAGED_LAYER_SET + " ctx.moe_stream->begin_layer(")
    m = m.replace(STAGED_LAYER_BEGIN, early, 1)
    assert not c(m)
    # The plan's host experts keep the op hybrid: plan_has_cpu_experts is not cleared.
    assert not c(mutate(SYCL, " && !pp_onednn_batched_claims_staged_layer;", ";"))
    # The MMVQ probes may consume a staged op before the executor sees it.
    assert not c(mutate(SYCL, "ne12 > 1 && (pp_onednn_batched_claims_staged_layer || "
                              "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device));",
                        "ne12 > 1 && ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device);"))
    # The planner's prompt route keeps a staged op hybrid.
    assert not c(mutate(SYCL, "prompt_batch && (pp_onednn_batched_claims_staged_layer || "
                              "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device));",
                        "prompt_batch && ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, ctx.device);"))
    # Decode steered by the staged bool: the unconditional decode term dropped.
    assert not c(mutate(SYCL, HYBRID_WITH_PLAN, "const bool moe_hybrid_with_plan = selected_hybrid_route;"))
    # The retained prompt batch materializes, so a staged layer's host experts are copied twice.
    assert not c(mutate(SYCL, "retained_prompt_layout, false, &ctx, ctx.stream());",
                        "retained_prompt_layout, true, &ctx, ctx.stream());"))
    # A second writer of the bool.
    assert not c(mutate(SYCL, PLAN_HAS_CPU, "pp_onednn_batched_claims_staged_layer = false; " + PLAN_HAS_CPU))


def claim_decode_builds_skip_the_phase(sycl: str) -> bool:
    """A decode build never takes the phase: `rows > 1` is the first term of the phase pointer's condition, ahead of
    the layer, the role and the stage-table lookup, which all read it. The refusal lambda runs only at a stage-table
    hit, so a build that finds no entry (every prompt build while the ring stages nothing) asks nothing of it."""
    b = build_body(sycl)
    return (b.count(STREAM_DECL) == 1
            and ordered(b, STREAM_DECL, LAYER_DECL, ROLE_DECL, LAMBDA_OPEN, GUARD, FIND, REFUSAL_OPEN,
                        REFUSAL_STEPS[0]))


def test_decode_builds_skip_the_phase():
    assert claim_decode_builds_skip_the_phase(SYCL)


def test_decode_builds_skip_the_phase_mutants():
    c = claim_decode_builds_skip_the_phase
    # The rows test left to the lambda, so a decode build takes the phase, its layer and its role.
    late_rows = mutate(SYCL, "= rows > 1 && sctx != nullptr", "= sctx != nullptr")
    late_rows = late_rows.replace(GUARD, "if (rows > 1 && moe_stream != nullptr && stream_layer >= 0) {", 1)
    assert not c(late_rows)
    # The refusal lambda run ahead of the lookup.
    m = mutate(SYCL, REFUSAL_OPEN, "const char * refusal_detail0 = nullptr; " + REFUSAL_OPEN)
    m = m.replace(FIND, "", 1)
    assert not c(m)


# ---- one device recipe -----------------------------------------------------------------------------------------------


def claim_one_device_recipe(sycl: str) -> bool:
    """A resident expert and a staged slot take their device recipe from the same function; the capability query sits
    in that function, not in the build."""
    b = build_body(sycl)
    r = body(norm(sycl), RECIPE_SIG)
    return (STAGED_RECIPE_CALL in b and CORE_RECIPE_CALL in b and CAP_QUERY in r and CAP_QUERY not in b
            and ordered(staged_branch(sycl), STAGED_RECIPE_CALL, ASSIGN))


def test_one_device_recipe():
    assert claim_one_device_recipe(SYCL)


def test_one_device_recipe_mutants():
    assert not claim_one_device_recipe(mutate(SYCL, STAGED_RECIPE_CALL,
                                              "const moe_route_capability cap = "
                                              "ggml_sycl_moe_query_route_capability(src0->type);"))
    # The transfer written before the recipe, which then overwrites it.
    m = mutate(SYCL, ASSIGN + " ", "")
    m = m.replace(STAGED_RECIPE_CALL, ASSIGN + " " + STAGED_RECIPE_CALL, 1)
    assert not claim_one_device_recipe(m)


# ---- no persistent handle --------------------------------------------------------------------------------------------


def claim_no_persistent_handle(sycl: str) -> bool:
    """The staged branch holds its slot only in the route; the transient upload clears a staged slot's handle entry
    and keeps it out of the active leases; both upload callers name the staged experts."""
    br = staged_branch(sycl)
    if br == "" or any(p in br for p in PERSISTENT):
        return False
    n = norm(sycl)
    up = body(n, UPLOAD_SIG)
    fb = body(n, FROM_BATCH_SIG)
    mmid = body(n, MMID_SIG)
    return (ordered(up, UPLOAD_STAGED, UPLOAD_CLEAR, UPLOAD_LEASE) and up.count("leases.push_back(") == 1
            and ordered(fb, FROM_BATCH_STAGED, FROM_BATCH_PASS) and ordered(mmid, HYBRID_COLLECT, HYBRID_PASS))


def test_no_persistent_handle():
    assert claim_no_persistent_handle(SYCL)


def test_no_persistent_handle_mutants():
    assert not claim_no_persistent_handle(mutate(SYCL, SERVE_RETURN,
                                                 "remember_moe_storage_handle(src0, slot); " + SERVE_RETURN))
    assert not claim_no_persistent_handle(mutate(SYCL, UPLOAD_LEASE, "leases.push_back(std::move(handle));"))
    assert not claim_no_persistent_handle(
        mutate(SYCL, "if (staged) { expert_handles[slot] = ggml_sycl::mem_handle{};",
               "if (staged) { expert_handles[slot] = handle;"))
    assert not claim_no_persistent_handle(mutate(SYCL, FROM_BATCH_PASS, "table_event_out, table_event_set);"))
    assert not claim_no_persistent_handle(
        mutate(SYCL, "&route_ptrs_event_set, route_staged_experts.empty() ? nullptr : &route_staged_experts);",
               "&route_ptrs_event_set);"))


# ---- a staged slot's pointer goes with its ring ----------------------------------------------------------------------

RELEASE_SIG = "static void ggml_sycl_moe_stream_release_payload(ggml_backend_sycl_context & ctx)"
RELEASE_SKIP = ("if (ctx.moe_stream && ctx.moe_stream->ring_releases() == ctx.moe_stream_payload_releases) { return; } "
                "ctx.moe_stream_payload_releases = ctx.moe_stream ? ctx.moe_stream->ring_releases() : 0;")
RELEASE_LOOP = "for (auto & kv : ctx.moe_stream_payload) {"
RELEASE_IDENTITY = ("ggml_tensor_extra_gpu * extra = kv.second.extra; "
                    "GGML_ASSERT(extra != nullptr && extra->cache_uuid == kv.first);")
RELEASE_PAYLOAD = "std::vector<void *> & payload = extra->weight().moe_expert_ptr_payload[device];"
RELEASE_HANDLES = "const std::vector<ggml_sycl::mem_handle> & handles = extra->weight().moe_expert_handles[device];"
RELEASE_NULL = ("if (staged[slot] != 0 && payload[slot] != nullptr && (slot >= handles.size() || "
                "!handles[slot].valid())) { payload[slot] = nullptr; nulled = true; }")
# The one member that marks the table stale and forgets the probe, with the device-index guard (llama.cpp-8xdz M3);
# the release path used to repeat its body inline.
RELEASE_INVALIDATE = "if (nulled) { extra->weight().mark_moe_device_table_stale(device); }"
RELEASE_INVALIDATE_INLINE = ("if (nulled) { extra->weight().moe_device_table_valid[device] = false; "
                             "ggml_sycl_invalidate_moe_full_local_probe(extra, device); }")
RELEASE_UNCOUNT = "extra->weight().moe_stream_payload_records.fetch_sub(1, std::memory_order_release);"
RELEASE_CLEAR = "ctx.moe_stream_payload.clear();"
RECORD_SIG = ("static void ggml_sycl_moe_stream_record_payload(ggml_backend_sycl_context & ctx, "
              "ggml_tensor_extra_gpu * extra, size_t slot, size_t n_slots)")
RECORD_STEPS = (
    "ctx.moe_stream_payload[ggml_sycl_assign_cache_uuid(extra)];",
    "if (rec.extra == nullptr) { rec.extra = extra; "
    "extra->weight().moe_stream_payload_records.fetch_add(1, std::memory_order_relaxed); }",
    "GGML_ASSERT(rec.extra == extra);",
    "rec.staged[slot] = 1;",
)
RECORD_STRUCT = ("struct moe_stream_payload_record { ggml_tensor_extra_gpu * extra = nullptr; "
                 "std::vector<uint8_t> staged; };")
RECORD_COUNTER = "std::atomic<uint32_t> moe_stream_payload_records{ 0 };"
LIFETIME_DOC = "Lifetime precondition: a record's extra outlives the record."
EXTRA_RELEASE_SIG = "void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams)"
EXTRA_RELEASE_ASSERT = ("GGML_ASSERT(extra->weight_ext == nullptr || "
                        "extra->weight_ext->moe_stream_payload_records.load(std::memory_order_acquire) == 0);")
RELEASE_SITES = (
    "sycl_ctx->moe_stream->end_prompt(); ggml_sycl_moe_stream_release_payload(*sycl_ctx);",
    "prompt_ids_snapshot.size(), stream_call.op); ggml_sycl_moe_stream_release_payload(ctx);",
    "moe_stream.reset(); ggml_sycl_moe_stream_release_payload(*this);",
    "ctx->moe_stream->end_prompt(); ggml_sycl_moe_stream_release_payload(*ctx);",
)
PAYLOAD_MEMBERS = ("std::unordered_map<uint64_t, moe_stream_payload_record> moe_stream_payload; "
                   "uint64_t moe_stream_payload_releases = 0;")
RELEASE_COUNT = "ring_->release(last_use); ring_.reset(); ++ring_releases_;"
RELEASES_ACCESSOR = "uint64_t ring_releases() const { return ring_releases_; }"
TEARDOWN_SIG = "void moe_stream_phase::teardown(sycl::event last_use)"


def claim_staged_pointer_released_with_ring(sycl: str, common: str, common_cpp: str, phase: str,
                                            plan_hpp: str) -> bool:
    """A raw pointer must not outlive its owning handle (llama.cpp-o9uw, c-8jtz). The upload records the payload slot it
    gives a staged slot's pointer, keyed by the weight extra's cache identity, never its address, and the record holds
    no device pointer; teardown counts the ring's release; when the count moves, or the phase is gone, the backend
    nulls only the recorded payload slots that still hold a pointer and no live handle (a later upload may have given
    the slot a resident expert), invalidates the device table and the full-local probe when it nulled any, and forgets
    the records. It runs at every point the ring can go, and nowhere submits anything. The record's extra is held
    under a documented lifetime precondition, counted on the extra, and release_extra_gpu asserts the count is zero
    before the extra goes."""
    n = norm(sycl)
    at = n.find(RELEASE_SIG + " {")
    rel = body(n[at:], RELEASE_SIG) if at >= 0 else ""
    rec = body(n, RECORD_SIG)
    td = body(norm(phase), TEARDOWN_SIG)
    cm = norm(common)
    struct_at = cm.find("struct moe_stream_payload_record {")
    record_struct = body(cm[struct_at:], "struct moe_stream_payload_record") if struct_at >= 0 else ""
    extra_release = body(norm(common_cpp), EXTRA_RELEASE_SIG)
    return (n.count(RELEASE_SIG + " {") == 1
            and ordered(rel, RELEASE_SKIP, RELEASE_LOOP, RELEASE_IDENTITY, RELEASE_PAYLOAD, RELEASE_HANDLES,
                        RELEASE_NULL, RELEASE_INVALIDATE, RELEASE_UNCOUNT, RELEASE_CLEAR)
            and "submit" not in rel and "wait" not in rel
            and n.count(RECORD_SIG + " {") == 1 and ordered(rec, *RECORD_STEPS)
            and UPLOAD_RECORD in body(n, UPLOAD_SIG) and n.count("moe_stream_payload[") == 1
            and all(n.count(site) == 1 for site in RELEASE_SITES)
            and n.count("ggml_sycl_moe_stream_release_payload(") == 1 + len(RELEASE_SITES)
            and PAYLOAD_MEMBERS in cm and RECORD_STRUCT in cm and "void *" not in record_struct
            and LIFETIME_DOC in common and RECORD_COUNTER in cm
            and ordered(extra_release, "if (prev > 1) { return; }", EXTRA_RELEASE_ASSERT, "delete extra;")
            and RELEASE_COUNT in td and norm(phase).count("++ring_releases_;") == 1
            and RELEASES_ACCESSOR in norm(plan_hpp))


def test_staged_pointer_released_with_ring():
    assert claim_staged_pointer_released_with_ring(SYCL, COMMON, COMMON_CPP, PHASE, PLAN_HPP)


def test_staged_pointer_release_mutants():
    def c(sycl=SYCL, common=COMMON, common_cpp=COMMON_CPP, phase=PHASE, plan_hpp=PLAN_HPP):
        return claim_staged_pointer_released_with_ring(sycl, common, common_cpp, phase, plan_hpp)

    # The upload writes the pointer and records nothing, so nothing ever nulls it.
    assert not c(sycl=mutate(SYCL, UPLOAD_RECORD, ""))
    # The payload slot left holding the pointer: only the guard would stand between it and a kernel.
    assert not c(sycl=mutate(SYCL, "{ payload[slot] = nullptr; nulled = true; }", "{ nulled = true; }"))
    # A slot a later upload gave a resident expert is nulled under it.
    assert not c(sycl=mutate(SYCL, " && (slot >= handles.size() || !handles[slot].valid())", ""))
    # The device table kept valid and the full-local probe kept, so moe_ptrs_ptr hands a kernel the stale device copy.
    assert not c(sycl=mutate(SYCL, RELEASE_INVALIDATE, "if (nulled) { }"))
    # The body repeated inline instead of calling the one member (the pre-8xdz form): the table is marked, but through
    # a second copy of the member's body that can drift from it.
    assert not c(sycl=mutate(SYCL, RELEASE_INVALIDATE, RELEASE_INVALIDATE_INLINE))
    # Nulled while the ring that holds the pointers is still held.
    assert not c(sycl=mutate(SYCL, "if (ctx.moe_stream && ctx.moe_stream->ring_releases() == "
                                   "ctx.moe_stream_payload_releases) { return; }", ""))
    # The phase gone and the records kept, as when an earlier release ran with no phase.
    assert not c(sycl=mutate(SYCL, RELEASE_SKIP,
                             "const uint64_t releases = ctx.moe_stream ? ctx.moe_stream->ring_releases() : UINT64_MAX; "
                             "if (releases == ctx.moe_stream_payload_releases) { return; } "
                             "ctx.moe_stream_payload_releases = releases;"))
    # The records kept, so a later release nulls a slot that now holds another ring's pointer by coincidence.
    assert not c(sycl=mutate(SYCL, RELEASE_CLEAR, ""))
    # Keyed by the extra's address again.
    assert not c(sycl=mutate(SYCL, "ctx.moe_stream_payload[ggml_sycl_assign_cache_uuid(extra)];",
                             "ctx.moe_stream_payload[reinterpret_cast<uintptr_t>(extra)];"))
    # The record's extra trusted without its identity.
    assert not c(sycl=mutate(SYCL, "GGML_ASSERT(extra != nullptr && extra->cache_uuid == kv.first);", ""))
    # A record counted on its extra and never uncounted, or never counted.
    assert not c(sycl=mutate(SYCL, RELEASE_UNCOUNT, ""))
    assert not c(sycl=mutate(SYCL, "extra->weight().moe_stream_payload_records.fetch_add(1, "
                                   "std::memory_order_relaxed);", ""))
    # The record holds the device pointer again.
    assert not c(common=mutate(COMMON, "std::vector<uint8_t> staged; };", "std::vector<void *> staged; };"))
    # The lifetime precondition undocumented.
    assert not c(common=COMMON.replace(LIFETIME_DOC, ""))
    # The extra freed under a live record.
    assert not c(common_cpp=mutate(COMMON_CPP, EXTRA_RELEASE_ASSERT, ""))
    # Any one of the release points missing.
    for site in RELEASE_SITES:
        head = site[:site.index(" ggml_sycl_moe_stream_release_payload(")]
        assert not c(sycl=mutate(SYCL, site, head))
    # Teardown releases the ring without counting it.
    assert not c(phase=mutate(PHASE, RELEASE_COUNT, "ring_->release(last_use); ring_.reset();"))
    # One release counted twice.
    assert not c(phase=mutate(PHASE, RELEASE_COUNT, RELEASE_COUNT + " ++ring_releases_;"))
    # The context forgets the records.
    assert not c(common=mutate(COMMON, "uint64_t moe_stream_payload_releases = 0;", ""))


# ---- the executors that cannot read a staged slot refuse a staged batch by name --------------------------------------

PER_EXPERT_REFUSE = ("if (ggml_sycl::moe_batch_names_staged_ring(retained_prompt_batch_result.batch)) { "
                     "throw ggml_sycl_fallback_error(\"MUL_MAT_ID retained prompt per-expert path refused a "
                     "staged ring slot\"); }")
PER_EXPERT_ANCHOR = "if (try_pp_onednn_f16_batched()) { return; }"
PER_EXPERT_STEPS = (
    "pp_moe_onednn_scratch_claim pp_moe_onednn_scratch_claim_state;",
    'GGML_ABORT("[MoE] Failed to allocate PP F32 activation staging for',
    "std::vector<ggml_sycl::mem_handle> pp_gpu_leases;",
    "moe_expert_route route = retained_prompt_group_route_for_expert(static_cast<int>(i02));",
    "expert_ptr = route.ptr;",
    "ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row, &route_layout);",
)
XMX_REJECT = "auto reject_xmx = [&](const char * reason) -> bool {"
XMX_REFUSE = "if (ggml_sycl::moe_batch_names_staged_ring(batch)) { return reject_xmx(\"staged-ring\"); }"
XMX_STEPS = (
    "const ggml_sycl_xmx_moe_policy xmx_policy = ggml_sycl_get_xmx_moe_policy();",
    "ggml_sycl_upload_moe_ptr_table_from_batch(",
    "expert_weights = expert_ptrs_host[e];",
)


def claim_staged_batch_refused_by_name(sycl: str) -> bool:
    """The non-hybrid retained prompt per-expert path multiplies each expert at the tensor's route_layout from
    route.ptr, which a staged slot (holding its entry's layout) does not match: a batch naming a staged slot throws a
    fallback error directly after the batched executor's try, before the path's scratch setup claims the oneDNN
    scratch slot, allocates its staging or aborts on a failed allocation, and before its first route.
    try_xmx_sorted_moe reads each expert through a payload with no staged slot: it rejects such a batch at its head,
    before the policy check and before any upload, instead of through the payload's missing entry."""
    n = norm(sycl)
    mmid = body(n, MMID_SIG)
    xmx = body(n, XMX_SIG)
    return (n.count(PER_EXPERT_REFUSE) == 1 and mmid.count(PER_EXPERT_ANCHOR + " " + PER_EXPERT_REFUSE) == 1
            and ordered(mmid, PER_EXPERT_ANCHOR, PER_EXPERT_REFUSE, *PER_EXPERT_STEPS)
            and n.count(XMX_REFUSE) == 1 and ordered(xmx, XMX_REJECT, XMX_REFUSE, *XMX_STEPS))


def test_staged_batch_refused_by_name():
    assert claim_staged_batch_refused_by_name(SYCL)


def test_staged_batch_refused_by_name_mutants():
    c = claim_staged_batch_refused_by_name
    # The per-expert path reads a staged slot at the tensor's layout.
    assert not c(mutate(SYCL, PER_EXPERT_REFUSE, ""))
    # Refused only after the routes were taken.
    m = mutate(SYCL, PER_EXPERT_REFUSE + " ", "")
    m = m.replace(PER_EXPERT_STEPS[-1], PER_EXPERT_STEPS[-1] + " " + PER_EXPERT_REFUSE, 1)
    assert not c(m)
    # Refused after the scratch setup, where a staged batch can hit the staging allocation's abort first.
    m = mutate(SYCL, PER_EXPERT_REFUSE + " ", "")
    m = m.replace(PER_EXPERT_STEPS[2], PER_EXPERT_REFUSE + " " + PER_EXPERT_STEPS[2], 1)
    assert not c(m)
    # The XMX path refuses only through the payload's missing entry.
    assert not c(mutate(SYCL, XMX_REFUSE, ""))
    # Refused only after the upload wrote the staged slot into the table (moved inside try_xmx_sorted_moe's body: the
    # upload helper's name also appears at six other sites).
    n = norm(SYCL)
    x = body(n, XMX_SIG)
    after_upload = "has_table_event = table_event_set; }"
    assert x.count(XMX_REFUSE + " ") == 1 and x.count(after_upload) == 1
    moved = x.replace(XMX_REFUSE + " ", "", 1).replace(after_upload, after_upload + " " + XMX_REFUSE, 1)
    assert not c(n.replace(x, moved, 1))


# ---- ordering: one barrier, no host waits ----------------------------------------------------------------------------


def claim_barrier_after_build(sycl: str) -> bool:
    """After the build, the staged slots' ready events go into one device barrier on the context's queue; a throw
    aborts. The barrier is the build function's only submission."""
    b = build_body(sycl)
    at = b.find(LAMBDA_OPEN)
    return at >= 0 and ordered(b[at:], *BARRIER) and b.count("ext_oneapi_submit_barrier(") == 1


def test_barrier_after_build():
    assert claim_barrier_after_build(SYCL)


def test_barrier_after_build_mutants():
    assert not claim_barrier_after_build(
        mutate(SYCL, "sctx->stream()->ext_oneapi_submit_barrier(staged_ready);", "(void) staged_ready;"))
    assert not claim_barrier_after_build(
        mutate(SYCL, 'GGML_ABORT("[MOE-EXEC] staged ring barrier failed: %s", e.what());', "throw;"))


def claim_no_host_waits(sycl: str) -> bool:
    """Ruling A (c-3bde): the build, the shared device recipe and the admission predicate wait on nothing and query
    no event status."""
    n = norm(sycl)
    parts = (build_body(sycl), body(n, RECIPE_SIG), body(n, PREDICATE_SIG))
    return all(p != "" and not WAIT_RE.search(p) for p in parts)


def test_no_host_waits():
    assert claim_no_host_waits(SYCL)


def test_no_host_waits_mutants():
    assert not claim_no_host_waits(mutate(SYCL, "sctx->stream()->ext_oneapi_submit_barrier(staged_ready);",
                                          "sctx->stream()->ext_oneapi_submit_barrier(staged_ready).wait();"))
    assert not claim_no_host_waits(
        mutate(SYCL, SERVE_RETURN,
               "s->ready.get_info<sycl::info::event::command_execution_status>(); " + SERVE_RETURN))


# ---- the admission cache and the fused path --------------------------------------------------------------------------


def claim_admission_cache_graph_keyed(sycl: str) -> bool:
    """A cached bundle naming a staged slot answers only in the graph that built it, through one predicate used at both
    lookups; the populate records both facts; the fused-path validator refuses such a bundle."""
    n = norm(sycl)
    mmid = body(n, MMID_SIG)
    pred_at = n.find(PREDICATE_SIG)
    pred = n[pred_at:n.find("}", pred_at) + 1] if pred_at >= 0 else ""
    role = body(n, ROLE_VALIDATE_SIG)
    return (pred.endswith(PREDICATE_BODY) and mmid.count("ggml_sycl_moe_admission_entry_reusable(") == 2
            and EARLY_LOOKUP in mmid and MAIN_LOOKUP in mmid and POPULATE in mmid and ROLE_REFUSE in role
            and body(n, NAMES_STAGED_SIG) == NAMES_STAGED_BODY)


def test_admission_cache_graph_keyed():
    assert claim_admission_cache_graph_keyed(SYCL)


def test_admission_cache_graph_keyed_mutants():
    assert not claim_admission_cache_graph_keyed(
        mutate(SYCL, PREDICATE_BODY, "{ return entry.valid; }"))
    assert not claim_admission_cache_graph_keyed(
        mutate(SYCL, "ggml_sycl_moe_admission_entry_reusable(admit_it->second, ctx.graph_compute_seq)",
               "admit_it->second.valid"))
    assert not claim_admission_cache_graph_keyed(mutate(SYCL, " new_entry.graph_seq = ctx.graph_compute_seq;", ""))
    assert not claim_admission_cache_graph_keyed(mutate(SYCL, ROLE_REFUSE, ""))
    assert not claim_admission_cache_graph_keyed(
        mutate(SYCL, " || ggml_sycl::moe_batch_names_staged_ring(roles.down.batch)", ""))


# ---- the builds that can receive a staged operand --------------------------------------------------------------------


def claim_prompt_builds_pass_invocation(sycl: str) -> bool:
    """The triad and the retained prompt build pass the invocation's context and queue."""
    mmid = body(norm(sycl), MMID_SIG)
    triad = all(f"{role}, false, &ctx, ctx.stream());" in mmid for role in TRIAD_ROLES)
    return triad and RETAINED_PROMPT in mmid


def test_prompt_builds_pass_invocation():
    assert claim_prompt_builds_pass_invocation(SYCL)


def test_prompt_builds_pass_invocation_mutants():
    assert not claim_prompt_builds_pass_invocation(
        mutate(SYCL, "up_layout, false, &ctx, ctx.stream());", "up_layout, false);"))
    assert not claim_prompt_builds_pass_invocation(
        mutate(SYCL, "retained_prompt_layout, false, &ctx, ctx.stream());", "retained_prompt_layout, false);"))


# ---- the validator ---------------------------------------------------------------------------------------------------


def claim_validator(batch_hpp: str) -> bool:
    """A STAGED_RING recipe is PRIMARY_DEVICE on the submit device and queue, with a ready event, after the workspace
    check; the validator takes the operand's has_ready_event."""
    v = body(norm(batch_hpp), VALIDATOR_SIG)
    sig_at = norm(batch_hpp).find(VALIDATOR_SIG)
    sig = norm(batch_hpp)[sig_at:sig_at + 400] if sig_at >= 0 else ""
    return VALIDATOR_PARAM in sig and ordered(v, WORKSPACE_REFUSE, VALIDATOR_CLAUSE)


def test_validator():
    assert claim_validator(BATCH_HPP)


def test_validator_mutants():
    assert not claim_validator(mutate(BATCH_HPP, "!has_ready_event || ", ""))
    assert not claim_validator(mutate(BATCH_HPP, " || recipe.queue != moe_recipe_queue::SUBMIT", ""))


# ---- the trace -------------------------------------------------------------------------------------------------------


def claim_trace(phase: str, tree: dict) -> bool:
    """The served/refused line exists once in the backend, goes through the decision log's trace sink, and end_prompt
    reports it before its early return, so a prompt's counts are never carried into the next. Under the same guard, a
    prompt that refused any also gets one line naming each refusal's cause and detail (plan Task 8), and the phase
    counts each lookup with the cause the resolver gave."""
    count = sum(norm(text).count(TRACE_FMT) for text in tree.values())
    refusals = sum(norm(text).count(TRACE_REFUSALS_FMT) for text in tree.values())
    n = norm(phase)
    return (count == 1 and refusals == 1
            and ordered(body(n, REPORT_SIG), TRACE_FMT + '", ' + TALLY_ARGS, TRACE_EMIT, *TRACE_REFUSALS)
            and body(n, END_PROMPT_SIG).startswith(END_PROMPT_HEAD) and body(n, COUNT_SIG) == COUNT_BODY)


def test_trace():
    assert claim_trace(PHASE, SYCL_TREE)


def test_trace_mutants():
    m = mutate(PHASE, END_PROMPT_HEAD,
               "{ if (!ubatch_live_ && !ring_ && !pending_ring_ && table_.empty()) { report_staged_routes();")
    assert not claim_trace(m, {PHASE_REL: m})
    # The served/refused line formatted but never emitted. The window trace emits through the same sink call, so the
    # mutant is made inside the report.
    n = norm(PHASE)
    rep = body(n, REPORT_SIG)
    assert TRACE_EMIT in rep, "mutant precondition: the report emits its line"
    m = n.replace(rep, rep.replace(TRACE_EMIT, "(void) line;", 1), 1)
    assert not claim_trace(m, {PHASE_REL: m})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + TRACE_FMT + '"'
    assert not claim_trace(PHASE, tree)
    # No refusals line: the trace counts refusals but cannot say why.
    m = mutate(PHASE, "log_->trace_sink(log_->sink_ctx, causes.c_str());", "(void) causes;")
    assert not claim_trace(m, {PHASE_REL: m})
    # The refusals line outside the guard, printed with tracing off.
    m = mutate(PHASE, "log_->trace_sink(log_->sink_ctx, causes.c_str()); } }", "} } (void) 0;")
    assert not claim_trace(m, {PHASE_REL: m})
    # A lookup counted without its cause.
    m = mutate(PHASE, COUNT_BODY, "{ staged_routes_.count(moe_staged_route_refusal::RECIPE, nullptr); }")
    assert not claim_trace(m, {PHASE_REL: m})


def claim_reported_once_on_every_exit(phase: str) -> bool:
    """A prompt's lookups are reported when it ends and, for a prompt the context outlives no end_prompt of, when the
    phase is destroyed. report_staged_routes prints only a non-empty tally and clears it as its last statement, so the
    destructor after an end_prompt prints nothing again; nothing else reports, so a prompt is never split."""
    n = norm(phase)
    r = body(n, REPORT_SIG)
    return (body(n, DTOR_SIG).startswith(DTOR_HEAD) and body(n, END_PROMPT_SIG).startswith(END_PROMPT_HEAD)
            and n.count("report_staged_routes();") == 2 and r.startswith(REPORT_HEAD)
            and r.endswith(REPORT_RESET + " }") and r.count(REPORT_RESET) == 1)


def test_reported_once_on_every_exit():
    assert claim_reported_once_on_every_exit(PHASE)


def test_reported_once_on_every_exit_mutants():
    # The destructor drops a prompt no end_prompt reported.
    assert not claim_reported_once_on_every_exit(mutate(PHASE, DTOR_HEAD, "{ teardown("))
    # The tally survives a report, so the destructor prints the prompt end_prompt already printed.
    assert not claim_reported_once_on_every_exit(mutate(PHASE, REPORT_RESET + " }", "}"))
    # Cleared before anything reads it: nothing is ever printed.
    assert not claim_reported_once_on_every_exit(
        mutate(PHASE, REPORT_HEAD, "{ " + REPORT_RESET + " " + REPORT_HEAD[2:]))
    # An empty tally printed: the destructor after an end_prompt prints served=0 refused=0.
    assert not claim_reported_once_on_every_exit(
        mutate(PHASE, "(staged_routes_.served != 0 || staged_routes_.handed_off != 0 || "
               "staged_routes_.refused != 0) && ", ""))
    # A report per micro-batch splits a prompt's counts.
    assert not claim_reported_once_on_every_exit(
        mutate(PHASE, "reserve_failed_ = false; table_.clear();", "report_staged_routes(); reserve_failed_ = false; "
               "table_.clear();"))


def claim_refusals_warn_under_ring(phase: str, hpp: str, tree: dict) -> bool:
    """Under an explicit ring a refused lookup leaves its experts to another route, so a prompt that refused any says
    so at WARN, once: the line is part of the report, after the trace's own lines and before the tally is cleared. The
    policy is the last op's, set where the decision log is. The format exists once in the backend."""
    n = norm(phase)
    r = body(n, REPORT_SIG)
    bl = body(n, BEGIN_LAYER_SIG)
    count = sum(norm(text).count(REFUSALS_WARN_FMT) for text in tree.values())
    return (count == 1 and ordered(r, "log_->trace_sink(log_->sink_ctx, causes.c_str()); } }", *REFUSALS_WARN)
            and r.count(REPORT_RESET) == 1
            and norm(hpp).count(POLICY_MEMBER) == 1 and n.count(POLICY_SET) == 1
            and ordered(bl, "log_ = op.log;", POLICY_SET))


def test_refusals_warn_under_ring():
    assert claim_refusals_warn_under_ring(PHASE, PLAN_HPP, SYCL_TREE)


def test_refusals_warn_under_ring_mutants():
    def c(phase=PHASE, hpp=PLAN_HPP, tree=None):
        return claim_refusals_warn_under_ring(phase, hpp, tree if tree is not None else {PHASE_REL: phase})
    # Under AUTO or CPU as well.
    assert not c(mutate(PHASE, " && policy_ == moe_exec_policy::RING", ""))
    # A prompt that refused nothing.
    assert not c(mutate(PHASE, "if (staged_routes_.refused != 0 && policy_", "if (policy_"))
    # At INFO, through the trace sink, so a run without -v never sees it.
    assert not c(mutate(PHASE, "log_->sink(log_->sink_ctx, warn_line.c_str());",
                        "log_->trace_sink(log_->sink_ctx, warn_line.c_str());"))
    # The policy never recorded, so it stays AUTO and the line never prints.
    assert not c(mutate(PHASE, POLICY_SET, ""))
    assert not c(PHASE, mutate(PLAN_HPP, POLICY_MEMBER, "moe_exec_policy policy_ = moe_exec_policy::RING;"))
    # The tally cleared before the WARN reads it.
    m = mutate(PHASE, "if (staged_routes_.refused != 0 && policy_",
               REPORT_RESET + " if (staged_routes_.refused != 0 && policy_")
    assert not c(m)
    # A second copy of the format elsewhere in the backend.
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + REFUSALS_WARN_FMT + '"'
    assert not c(tree=tree)


# ---- lead ruling c-o9uw-1: the slice's tag is the layout -------------------------------------------------------------

TAGGED_SLICE_DECL = "mem_handle slice(size_t byte_offset, size_t byte_size, ggml_layout_mode layout) const;"
TAGGED_SLICE_SIG = "mem_handle mem_handle::slice(size_t byte_offset, size_t byte_size, ggml_layout_mode layout) const"
OWNER_FIRST = "{ if (!owned_alloc_) { return {}; }"
TAG_SET = "view_layout_ = layout;"
VIEW_SIG = "resolved_ptr mem_handle::resolved_view_locked() const"
VIEW_APPLY = "if (view_layout_) { view.layout = *view_layout_; }"
SPECIAL_MEMBERS = (
    ("mem_handle::mem_handle(const mem_handle & other)", ("view_layout_ = other.view_layout_;",)),
    ("mem_handle::mem_handle(mem_handle && other) noexcept", ("view_layout_ = other.view_layout_;",)),
    ("mem_handle & mem_handle::operator=(const mem_handle & other)",
     ("new_view_layout = other.view_layout_;", "view_layout_ = new_view_layout;")),
    ("mem_handle & mem_handle::operator=(mem_handle && other) noexcept",
     ("new_view_layout = other.view_layout_;", "view_layout_ = new_view_layout;")),
)
RING_ACQUIRE_DECL = "mem_handle acquire_slot(ggml_layout_mode layout);"
RING_ACQUIRE_SIG = "mem_handle moe_stream_ring::acquire_slot(ggml_layout_mode layout)"
# The slot's own allocation, re-tagged with the caller's layout (llama.cpp-am87: one allocation per slot).
RING_SLICE = "return slots_[index].slice(0, geo_.slot_bytes, layout);"
STAGE_SIG = "moe_stream_phase::stage_counts moe_stream_phase::stage_window("
PHASE_AOS_ACQUIRE = "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_AOS);"
PHASE_AOS_ENTRY = "table_[key] = { mem_handle(), aos, copied, GGML_LAYOUT_AOS, item.consumer };"
TAG_AGREES = "if (view.layout != s->layout) { return moe_staged_route_refusal::LAYOUT; }"


def claim_tag_is_the_layout(mem_hpp: str, mem_cpp: str, ring_hpp: str, ring_cpp: str, phase: str, sycl: str) -> bool:
    """A ring slot's resolved layout is the layout the phase acquired it for, which is the layout its stage-table entry
    records: an owned allocation's slice may name its layout (any other handle refuses before deriving anything), the
    view applies the name on every resolve, every copy and move carries it, the ring slices each slot's own allocation
    with it, the phase asks AOS for its one slot per matrix and records AOS, and asks no slot for SOA (the SOA staging
    is retired), and the resolver still refuses, as a layout refusal, a slot whose resolved layout is not its
    entry's."""
    mh, mc, rh, rc, ph = norm(mem_hpp), norm(mem_cpp), norm(ring_hpp), norm(ring_cpp), norm(phase)
    stage = body(ph, STAGE_SIG)
    tagged = body(mc, TAGGED_SLICE_SIG)
    view = body(mc, VIEW_SIG)
    members_carry = all(all(line in body(mc, sig) for line in lines) for sig, lines in SPECIAL_MEMBERS)
    return (TAGGED_SLICE_DECL in mh and tagged.startswith(OWNER_FIRST) and TAG_SET in tagged
            and ordered(view, "if (!view.ptr) { return view; }", VIEW_APPLY, "if (!is_slice_) {")
            and members_carry
            and RING_ACQUIRE_DECL in rh and RING_SLICE in body(rc, RING_ACQUIRE_SIG)
            and ordered(stage, PHASE_AOS_ACQUIRE, PHASE_AOS_ENTRY) and stage.count("acquire_slot(") == 1
            and "acquire_slot(GGML_LAYOUT_SOA)" not in ph
            and "GGML_LAYOUT_SOA" not in ph
            and TAG_AGREES in build_body(sycl))


def test_tag_is_the_layout():
    assert claim_tag_is_the_layout(MEM_HPP, MEM_CPP, RING_HPP, RING_CPP, PHASE, SYCL)


def test_tag_is_the_layout_mutants():
    args = [MEM_HPP, MEM_CPP, RING_HPP, RING_CPP, PHASE, SYCL]

    def with_(i, old, new):
        a = list(args)
        a[i] = mutate(a[i], old, new)
        return a

    # Any handle may name a layout, a weight lease included.
    assert not claim_tag_is_the_layout(
        *with_(1, "if (!owned_alloc_) { return {}; } mem_handle h = slice(", "mem_handle h = slice("))
    # The name kept but never applied, so a slice resolves with its root's layout.
    assert not claim_tag_is_the_layout(*with_(1, VIEW_APPLY, ""))
    # A copy, or a copy assignment, that drops the name.
    assert not claim_tag_is_the_layout(*with_(1, "view_layout_ = other.view_layout_;", ""))
    assert not claim_tag_is_the_layout(*with_(1, "view_layout_ = new_view_layout;", ""))
    # The ring slicing every slot with its root's AOS tag again.
    assert not claim_tag_is_the_layout(*with_(3, RING_SLICE, "return slots_[index].slice(0, geo_.slot_bytes);"))
    # The single-allocation ring's slice of one base, even with the caller's tag.
    assert not claim_tag_is_the_layout(
        *with_(3, RING_SLICE, "return base_.slice(index * geo_.slot_bytes, geo_.slot_bytes, layout);"))
    # The AOS slot acquired as SOA: its tag would disagree with the entry's layout.
    assert not claim_tag_is_the_layout(
        *with_(4, PHASE_AOS_ACQUIRE, "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_SOA);"))
    # The entry recorded as SOA for an AOS slot.
    assert not claim_tag_is_the_layout(
        *with_(4, PHASE_AOS_ENTRY, PHASE_AOS_ENTRY.replace("GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA")))
    # A SOA slot acquired beside it again.
    assert not claim_tag_is_the_layout(
        *with_(4, PHASE_AOS_ACQUIRE, PHASE_AOS_ACQUIRE + " mem_handle soa = ring_->acquire_slot(GGML_LAYOUT_SOA);"))
    # A second AOS slot per matrix.
    assert not claim_tag_is_the_layout(
        *with_(4, PHASE_AOS_ACQUIRE, PHASE_AOS_ACQUIRE + " mem_handle aos2 = ring_->acquire_slot(GGML_LAYOUT_AOS);"))
    # The resolver trusting the entry without checking the slot agrees.
    assert not claim_tag_is_the_layout(*with_(5, TAG_AGREES, ""))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
