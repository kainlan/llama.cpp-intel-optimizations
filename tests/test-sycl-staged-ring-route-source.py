"""Source contract for llama.cpp-o9uw (plan Task 8 of llama.cpp-36j0): STAGED_RING is the one route by which a host
expert the prompt's stream phase staged into a VRAM ring slot answers PRIMARY_DEVICE.

Host-only: reads sources, builds nothing, loads no model and touches no device.

What this gate pins:
  - one writer: `recipe.transfer = moe_recipe_transfer::STAGED_RING` is assigned at exactly one site in the backend,
    inside ggml_sycl_build_moe_resolved_batch's resolver lambda, after the `rows > 1` prompt guard and the phase's
    `moe_stream->find(` lookup, and before the route core; moe_batch_route carries no `staged_ring` flag (brief
    amendment: the recipe's transfer is the fact);
  - the phase is consulted only from the context that owns it, on that context's own queue, outside a graph recording,
    and keyed by the layer and role the phase itself uses (ggml_sycl_moe_stream_layer_of, the plan's role classifier);
  - ruling C (c-3bde): the slot and its layout come from the stage-table entry (`s->layout`), never from the slice's
    tag; the lease is the slot and the ready event is the entry's;
  - lead ruling c-o9uw-1: the slice's tag agrees with the entry by construction. mem_handle::slice(offset, size,
    layout) names the layout only on an owned allocation (it refuses before anything else otherwise), the view applies
    it on every resolve and all four special members carry it; the ring slices each slot with the layout it is
    acquired for, the phase acquires the AOS slot as AOS and the SOA slot as SOA, and the resolver still refuses a slot
    whose resolved layout is not the entry's;
  - c-las5 (1): a staged expert is served only as a plain MMVQ_COMPAT kernel on the submit queue. A weight the oneDNN
    batched prompt executor claims, any other kernel, a slice whose resolved layout is not the entry's, or a route
    either validator refuses falls through to the route core; every lookup is counted served or refused;
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
    end_prompt, before end_prompt's early return;
  - a staged slot's raw pointer does not outlive the ring that holds it (lead item 6, c-ned2): the upload records the
    pointer it writes into a weight's persistent payload, the phase counts every ring release in teardown, and once
    the count moves the backend nulls each recorded payload slot that still holds that pointer and no live handle,
    marks the device table invalid and resets the full-local probe, so moe_ptrs_ptr answers null until the next upload
    rewrites the table. It runs after end_prompt at the first decode graph and in the test hook, after begin_layer
    (a new geometry re-reserves the ring), and once the phase is gone in the context destructor.

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

SYCL = read(SYCL_REL)
BATCH_HPP = read(BATCH_HPP_REL)
PHASE = read(PHASE_REL)
MEM_HPP = read(MEM_HPP_REL)
MEM_CPP = read(MEM_CPP_REL)
RING_HPP = read(RING_HPP_REL)
RING_CPP = read(RING_CPP_REL)
PLAN_HPP = read(PLAN_HPP_REL)
COMMON = read(COMMON_REL)
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
GUARD = "if (rows > 1 && moe_stream != nullptr && stream_layer >= 0) {"
FIND = "if (const moe_stream_staged * s = moe_stream->find(stream_layer, expert_id, stream_role)) {"
SERVE_RETURN = "moe_stream->count_staged_route(served); if (served) { return staged; }"
ROUTE_CORE = "::ggml_sycl_resolve_moe_expert_route_for_dispatch("
LAMBDA_OPEN = "auto result = build_moe_resolved_batch(ids, count, slots_per_token, submit_device, [&](int32_t expert_id) {"

STREAM_DECL = ("moe_stream_phase * moe_stream = sctx != nullptr && sctx->device == submit_device && sctx->moe_stream && "
               "invocation_queue == sctx->stream() && !g_ggml_sycl_graph_recording ? sctx->moe_stream.get() : nullptr;")
LAYER_DECL = "const int stream_layer = moe_stream ? ggml_sycl_moe_stream_layer_of(src0) : -1;"
ROLE_DECL = ("const expert_tensor_role stream_role = moe_stream && src0 ? expert_tensor_role_from_tensor_name(src0->name) : "
             "expert_tensor_role::UNKNOWN;")
PREPARE_LAYER = "call.layer = ggml_sycl_moe_stream_layer_of(src0);"
CLAIMED_DECL = ("const bool stream_library_claimed = moe_stream != nullptr && "
                "ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, submit_device);")

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
SERVED_TERMS = (
    "!stream_library_claimed",
    "cap.supported",
    "cap.kernel == moe_route_kernel::MMVQ_COMPAT",
    "view.ptr != nullptr",
    "view.on_device",
    "view.layout == s->layout",
    "staged.recipe.valid",
    "detail::validate_moe_batch_route(staged, submit_device) == moe_batch_reject_reason::NONE",
    "validate_moe_execution_recipe(staged.recipe, moe_admitted_recipe_signature(staged.recipe, staged.lease), "
    "staged.lease, staged.residency, staged.has_ready_event, submit_device, false, 0, &why)",
)
STAGED_RECIPE_CALL = ("const moe_route_capability cap = ggml_sycl_moe_build_device_recipe(src0, submit_device, rows, "
                      "invocation_backend, invocation_queue, queue_capability, staged);")
CORE_RECIPE_CALL = ("(void) ggml_sycl_moe_build_device_recipe(src0, submit_device, rows, invocation_backend, "
                    "invocation_queue, queue_capability, normalized);")
CAP_QUERY = "ggml_sycl_moe_query_route_capability("

PERSISTENT = ("moe_expert_handles[", "remember_moe_storage_handle(", "retain_handles_until_event(",
              "ggml_sycl_set_moe_ptr_table_leases(", "tag_persistent_lease_site(")

UPLOAD_STAGED = ("const bool staged = staged_experts != nullptr && std::find(staged_experts->begin(), "
                 "staged_experts->end(), eid) != staged_experts->end();")
UPLOAD_RECORD = ("std::vector<void *> & ring_ptrs = ctx.moe_stream_payload[extra]; if (ring_ptrs.size() != "
                 "static_cast<size_t>(n_experts)) { ring_ptrs.assign(static_cast<size_t>(n_experts), nullptr); } "
                 "ring_ptrs[slot] = resolved.ptr;")
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

TRACE_FMT = "[MOE-EXEC-TRACE] staged route device %d: served=%zu refused=%zu"
TRACE_EMIT = "log_->trace_sink(log_->sink_ctx, line);"
END_PROMPT_HEAD = "{ report_staged_routes(); if (!ubatch_live_ && !ring_ && table_.empty()) {"

WAIT_RE = re.compile(r"\bwait\s*\(|\bwait_and_throw\s*\(|\bget_info\s*<")


def build_body(sycl: str) -> str:
    return body(norm(sycl), BUILD_SIG)


def staged_branch(sycl: str) -> str:
    """The text of the staged branch: the `rows > 1` guard's block inside the build function."""
    b = build_body(sycl)
    at = b.find(GUARD)
    return body(b[at:], GUARD) if at >= 0 else ""


# ---- one writer, one place --------------------------------------------------------------------------------------------


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
    # Decode consults it: the rows guard is gone.
    m = mutate(SYCL, GUARD, "if (moe_stream != nullptr && stream_layer >= 0) {")
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
    return (ordered(b, STREAM_DECL, LAYER_DECL, ROLE_DECL, CLAIMED_DECL, LAMBDA_OPEN) and PREPARE_LAYER in prep
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


def claim_served_only_as_plain_kernel(sycl: str) -> bool:
    """c-las5 (1): served only as MMVQ_COMPAT on the submit queue, never for a library-claimed weight, never for a slice
    whose layout is not the entry's, and only when both validators pass; every lookup is counted."""
    br = staged_branch(sycl)
    at = br.find("const bool served =")
    if at < 0:
        return False
    expr = br[at:br.find(";", at)]
    for term in SERVED_TERMS:
        if term not in expr:
            return False
    # MMVQ_COMPAT is the only kernel named in the served test; no other kernel can be served.
    kernels = re.findall(r"moe_route_kernel::(\w+)", expr)
    return kernels == ["MMVQ_COMPAT"] and ordered(br, ASSIGN, "const bool served =", SERVE_RETURN)


def test_served_only_as_plain_kernel():
    assert claim_served_only_as_plain_kernel(SYCL)


def test_served_only_as_plain_kernel_mutants():
    assert not claim_served_only_as_plain_kernel(mutate(SYCL, "!stream_library_claimed && cap.supported", "cap.supported"))
    assert not claim_served_only_as_plain_kernel(
        mutate(SYCL, "cap.kernel == moe_route_kernel::MMVQ_COMPAT",
               "(cap.kernel == moe_route_kernel::MMVQ_COMPAT || cap.kernel == moe_route_kernel::XMX_TILED)"))
    assert not claim_served_only_as_plain_kernel(mutate(SYCL, " && view.layout == s->layout", ""))
    assert not claim_served_only_as_plain_kernel(
        mutate(SYCL, " && detail::validate_moe_batch_route(staged, submit_device) == moe_batch_reject_reason::NONE", ""))
    assert not claim_served_only_as_plain_kernel(mutate(SYCL, SERVE_RETURN, "if (served) { return staged; }"))


def claim_library_claim_predicate(sycl: str) -> bool:
    """The library refusal asks the oneDNN batched executor's own claim predicate (forward-declared before the
    resolver, defined with the executors), not a restatement of it."""
    n = norm(sycl)
    fwd = "static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_tensor * weight, int device);"
    return fwd in n and n.find(fwd) < n.find(BUILD_SIG) and CLAIMED_DECL in build_body(sycl)


def test_library_claim_predicate():
    assert claim_library_claim_predicate(SYCL)


def test_library_claim_predicate_mutants():
    assert not claim_library_claim_predicate(
        mutate(SYCL, CLAIMED_DECL, "const bool stream_library_claimed = src0->type == GGML_TYPE_MXFP4;"))


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
                                              "const moe_route_capability cap = ggml_sycl_moe_query_route_capability(src0->type);"))
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
        mutate(SYCL, "if (staged) { expert_handles[slot] = ggml_sycl::mem_handle{};", "if (staged) { expert_handles[slot] = handle;"))
    assert not claim_no_persistent_handle(mutate(SYCL, FROM_BATCH_PASS, "table_event_out, table_event_set);"))
    assert not claim_no_persistent_handle(
        mutate(SYCL, "&route_ptrs_event_set, route_staged_experts.empty() ? nullptr : &route_staged_experts);",
               "&route_ptrs_event_set);"))


# ---- a staged slot's pointer goes with its ring -----------------------------------------------------------------------

RELEASE_SIG = "static void ggml_sycl_moe_stream_release_payload(ggml_backend_sycl_context & ctx)"
RELEASE_SKIP = ("const uint64_t releases = ctx.moe_stream ? ctx.moe_stream->ring_releases() : UINT64_MAX; "
                "if (releases == ctx.moe_stream_payload_releases) { return; } "
                "ctx.moe_stream_payload_releases = releases;")
RELEASE_LOOP = "for (auto & kv : ctx.moe_stream_payload) {"
RELEASE_PAYLOAD = "std::vector<void *> & payload = extra->weight().moe_expert_ptr_payload[device];"
RELEASE_HANDLES = "const std::vector<ggml_sycl::mem_handle> & handles = extra->weight().moe_expert_handles[device];"
RELEASE_NULL = ("if (ring_ptrs[slot] != nullptr && payload[slot] == ring_ptrs[slot] && (slot >= handles.size() || "
                "!handles[slot].valid())) { payload[slot] = nullptr; nulled = true; }")
RELEASE_INVALIDATE = ("if (nulled) { extra->weight().moe_device_table_valid[device] = false; "
                      "ggml_sycl_invalidate_moe_full_local_probe(extra, device); }")
RELEASE_CLEAR = "ctx.moe_stream_payload.clear();"
RELEASE_SITES = (
    "sycl_ctx->moe_stream->end_prompt(); ggml_sycl_moe_stream_release_payload(*sycl_ctx);",
    "prompt_ids_snapshot.size(), stream_call.op); ggml_sycl_moe_stream_release_payload(ctx);",
    "moe_stream.reset(); ggml_sycl_moe_stream_release_payload(*this);",
    "ctx->moe_stream->end_prompt(); ggml_sycl_moe_stream_release_payload(*ctx);",
)
PAYLOAD_MEMBERS = ("std::unordered_map<ggml_tensor_extra_gpu *, std::vector<void *>> moe_stream_payload; "
                   "uint64_t moe_stream_payload_releases = 0;")
RELEASE_COUNT = "ring_->release(); ring_.reset(); ++ring_releases_;"
RELEASES_ACCESSOR = "uint64_t ring_releases() const { return ring_releases_; }"
TEARDOWN_SIG = "void moe_stream_phase::teardown(sycl::event last_use)"


def claim_staged_pointer_released_with_ring(sycl: str, common: str, phase: str, plan_hpp: str) -> bool:
    """A raw pointer must not outlive its owning handle (lead item 6, c-ned2). The upload records the staged slot's
    pointer it writes into the persistent payload; teardown counts the ring's release; when the count moves, the
    backend nulls only the payload slots that still hold a recorded pointer and no live handle (a later upload may
    have given the slot a resident expert), invalidates the device table and the full-local probe when it nulled any,
    and forgets the records. It runs at every point the ring can go, and nowhere submits anything."""
    n = norm(sycl)
    at = n.find(RELEASE_SIG + " {")
    rel = body(n[at:], RELEASE_SIG) if at >= 0 else ""
    td = body(norm(phase), TEARDOWN_SIG)
    return (n.count(RELEASE_SIG + " {") == 1
            and ordered(rel, RELEASE_SKIP, RELEASE_LOOP, RELEASE_PAYLOAD, RELEASE_HANDLES, RELEASE_NULL,
                        RELEASE_INVALIDATE, RELEASE_CLEAR)
            and "submit" not in rel and "wait" not in rel
            and UPLOAD_RECORD in body(n, UPLOAD_SIG)
            and all(n.count(site) == 1 for site in RELEASE_SITES)
            and n.count("ggml_sycl_moe_stream_release_payload(") == 1 + len(RELEASE_SITES)
            and PAYLOAD_MEMBERS in norm(common)
            and RELEASE_COUNT in td and norm(phase).count("++ring_releases_;") == 1
            and RELEASES_ACCESSOR in norm(plan_hpp))


def test_staged_pointer_released_with_ring():
    assert claim_staged_pointer_released_with_ring(SYCL, COMMON, PHASE, PLAN_HPP)


def test_staged_pointer_release_mutants():
    def c(sycl=SYCL, common=COMMON, phase=PHASE, plan_hpp=PLAN_HPP):
        return claim_staged_pointer_released_with_ring(sycl, common, phase, plan_hpp)

    # The upload writes the pointer and records nothing, so nothing ever nulls it.
    assert not c(sycl=mutate(SYCL, UPLOAD_RECORD, ""))
    # The payload slot left holding the pointer: only the guard would stand between it and a kernel.
    assert not c(sycl=mutate(SYCL, "{ payload[slot] = nullptr; nulled = true; }", "{ nulled = true; }"))
    # A slot a later upload gave a resident expert is nulled under it.
    assert not c(sycl=mutate(SYCL, " && (slot >= handles.size() || !handles[slot].valid())", ""))
    # The device table kept valid, so moe_ptrs_ptr hands a kernel the stale device copy.
    assert not c(sycl=mutate(SYCL, "extra->weight().moe_device_table_valid[device] = false; ", ""))
    # The full-local probe kept.
    assert not c(sycl=mutate(SYCL, " ggml_sycl_invalidate_moe_full_local_probe(extra, device); }", " }"))
    # Nulled while the ring that holds the pointers is still held.
    assert not c(sycl=mutate(SYCL, "if (releases == ctx.moe_stream_payload_releases) { return; }", ""))
    # The records kept, so a later release nulls a slot that now holds another ring's pointer by coincidence.
    assert not c(sycl=mutate(SYCL, RELEASE_CLEAR, ""))
    # Any one of the release points missing.
    for site in RELEASE_SITES:
        head = site[:site.index(" ggml_sycl_moe_stream_release_payload(")]
        assert not c(sycl=mutate(SYCL, site, head))
    # Teardown releases the ring without counting it.
    assert not c(phase=mutate(PHASE, RELEASE_COUNT, "ring_->release(); ring_.reset();"))
    # One release counted twice.
    assert not c(phase=mutate(PHASE, "ring_->release(); ring_.reset(); ++ring_releases_;",
                              "ring_->release(); ring_.reset(); ++ring_releases_; ++ring_releases_;"))
    # The context forgets the records.
    assert not c(common=mutate(COMMON, "uint64_t moe_stream_payload_releases = 0;", ""))


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
    assert not claim_barrier_after_build(mutate(SYCL, "sctx->stream()->ext_oneapi_submit_barrier(staged_ready);", "(void) staged_ready;"))
    assert not claim_barrier_after_build(mutate(SYCL, 'GGML_ABORT("[MOE-EXEC] staged ring barrier failed: %s", e.what());', "throw;"))


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
    assert not claim_no_host_waits(mutate(SYCL, SERVE_RETURN, "s->ready.get_info<sycl::info::event::command_execution_status>(); " + SERVE_RETURN))


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
        mutate(SYCL, "ggml_sycl_moe_admission_entry_reusable(admit_it->second, ctx.graph_compute_seq)", "admit_it->second.valid"))
    assert not claim_admission_cache_graph_keyed(mutate(SYCL, " new_entry.graph_seq = ctx.graph_compute_seq;", ""))
    assert not claim_admission_cache_graph_keyed(mutate(SYCL, ROLE_REFUSE, ""))
    assert not claim_admission_cache_graph_keyed(mutate(SYCL, " || ggml_sycl::moe_batch_names_staged_ring(roles.down.batch)", ""))


# ---- the builds that can receive a staged operand --------------------------------------------------------------------


def claim_prompt_builds_pass_invocation(sycl: str) -> bool:
    """The triad and the retained prompt build pass the invocation's context and queue."""
    mmid = body(norm(sycl), MMID_SIG)
    triad = all(f"{role}, false, &ctx, ctx.stream());" in mmid for role in TRIAD_ROLES)
    return triad and RETAINED_PROMPT in mmid


def test_prompt_builds_pass_invocation():
    assert claim_prompt_builds_pass_invocation(SYCL)


def test_prompt_builds_pass_invocation_mutants():
    assert not claim_prompt_builds_pass_invocation(mutate(SYCL, "up_layout, false, &ctx, ctx.stream());", "up_layout, false);"))
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
    reports it before its early return, so a prompt's counts are never carried into the next."""
    count = sum(norm(text).count(TRACE_FMT) for text in tree.values())
    n = norm(phase)
    return count == 1 and ordered(body(n, REPORT_SIG), TRACE_FMT, TRACE_EMIT) and body(n, END_PROMPT_SIG).startswith(END_PROMPT_HEAD)


def test_trace():
    assert claim_trace(PHASE, SYCL_TREE)


def test_trace_mutants():
    m = mutate(PHASE, END_PROMPT_HEAD, "{ if (!ubatch_live_ && !ring_ && table_.empty()) { report_staged_routes();")
    assert not claim_trace(m, {PHASE_REL: m})
    m = mutate(PHASE, TRACE_EMIT, "(void) line;")
    assert not claim_trace(m, {PHASE_REL: m})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + TRACE_FMT + '"'
    assert not claim_trace(PHASE, tree)


# ---- lead ruling c-o9uw-1: the slice's tag is the layout ---------------------------------------------------------------

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
RING_SLICE = "return base_.slice(index * geo_.slot_bytes, geo_.slot_bytes, layout);"
PHASE_AOS_ACQUIRE = "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_AOS);"
PHASE_SOA_ACQUIRE = "mem_handle soa = aos_only ? mem_handle() : ring_->acquire_slot(GGML_LAYOUT_SOA);"
PHASE_AOS_ENTRY = "table_[key] = { mem_handle(), aos, copied, GGML_LAYOUT_AOS };"
PHASE_SOA_ENTRY = "table_[key] = { soa, aos, reorder, GGML_LAYOUT_SOA };"
TAG_AGREES = "view.layout == s->layout"


def claim_tag_is_the_layout(mem_hpp: str, mem_cpp: str, ring_hpp: str, ring_cpp: str, phase: str, sycl: str) -> bool:
    """A ring slot's resolved layout is the layout the phase acquired it for, which is the layout its stage-table entry
    records: an owned allocation's slice may name its layout (any other handle refuses before deriving anything), the
    view applies the name on every resolve, every copy and move carries it, the ring slices with it, the phase asks
    AOS for the AOS slot and SOA for the SOA slot and records those layouts, and the resolver still checks that the
    two agree before it serves."""
    mh, mc, rh, rc, ph = norm(mem_hpp), norm(mem_cpp), norm(ring_hpp), norm(ring_cpp), norm(phase)
    tagged = body(mc, TAGGED_SLICE_SIG)
    view = body(mc, VIEW_SIG)
    members_carry = all(all(line in body(mc, sig) for line in lines) for sig, lines in SPECIAL_MEMBERS)
    return (TAGGED_SLICE_DECL in mh and tagged.startswith(OWNER_FIRST) and TAG_SET in tagged
            and ordered(view, "if (!view.ptr) { return view; }", VIEW_APPLY, "if (!is_slice_) {")
            and members_carry
            and RING_ACQUIRE_DECL in rh and RING_SLICE in body(rc, RING_ACQUIRE_SIG)
            and ordered(ph, PHASE_AOS_ACQUIRE, PHASE_SOA_ACQUIRE) and PHASE_AOS_ENTRY in ph and PHASE_SOA_ENTRY in ph
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
    assert not claim_tag_is_the_layout(*with_(1, "if (!owned_alloc_) { return {}; } mem_handle h = slice(", "mem_handle h = slice("))
    # The name kept but never applied, so a slice resolves with its root's layout.
    assert not claim_tag_is_the_layout(*with_(1, VIEW_APPLY, ""))
    # A copy, or a copy assignment, that drops the name.
    assert not claim_tag_is_the_layout(*with_(1, "view_layout_ = other.view_layout_;", ""))
    assert not claim_tag_is_the_layout(*with_(1, "view_layout_ = new_view_layout;", ""))
    # The ring slicing every slot with its root's AOS tag again.
    assert not claim_tag_is_the_layout(
        *with_(3, RING_SLICE, "return base_.slice(index * geo_.slot_bytes, geo_.slot_bytes);"))
    # The SOA slot acquired as AOS.
    assert not claim_tag_is_the_layout(
        *with_(4, PHASE_SOA_ACQUIRE, "mem_handle soa = aos_only ? mem_handle() : ring_->acquire_slot(GGML_LAYOUT_AOS);"))
    # The resolver trusting the entry without checking the slot agrees.
    assert not claim_tag_is_the_layout(*with_(5, TAG_AGREES + " && ", ""))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
