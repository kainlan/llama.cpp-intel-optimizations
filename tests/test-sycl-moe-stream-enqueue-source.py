"""Source contract for llama.cpp-xod7 (plan Task 7b of llama.cpp-36j0): a prompt layer's host experts are weighed at its
first MUL_MAT_ID, and the next layer's are copied on the BCS queue and reordered on the compute queue a layer ahead.

Host-only: reads sources, builds nothing, loads no model and touches no device.

What this gate pins:
  - ggml_sycl_mul_mat_id consults the prompt phase at exactly one site: inside the prompt ids-admission branch
    (`ne12 != 1`, guarded by the same `ids->ne[1] > 1` that defines a prompt batch), after the graph-sink refusal (so a
    ring copy is never recorded into a command graph) and the ids snapshot the layer routes on, and before
    `moe_hybrid_with_plan` is computed. The site is the admission branch rather than the later `prompt_batch` local
    because the executor fast paths return before that local is computed, which would skip the layer-ahead staging after
    every all-device layer;
  - only an explicit GGML_SYCL_MOE_EXEC=ring stages: the phase's one stage() call sits under a guard that needs both
    the RING choice and the RING policy, the ring's reserve is reachable only through it, and the op's policy is the
    parsed config's. Nothing consumes a staged expert until Task 8 (llama.cpp-o9uw), so under AUTO the decision log
    records what the cost model would choose and no bytes move (lead ruling c-3skf);
  - a refused ring reserve is not retried within a graph and is tried once more in the next: the latch is cleared by
    begin_ubatch, which every new micro-batch runs, and by end_prompt. An expert loan runs at the graph boundary
    (ruling c-tdra, llama.cpp-mwmr), so a per-prompt latch would waste the loan its own prompt asked for (lead ruling
    c-i8rz);
  - a layer answering RING_UNAVAILABLE under an explicit ring or a trace prints one WARN per prompt that names the rule
    and the ring's inputs, in a fixed format, through the decision log's WARN sink; the flag is cleared by end_prompt
    and never per graph (lead ruling c-kcyf);
  - resolve() reports every layer it resolves, AUTO included, so the decision log keeps the cost model's choice
    (the second half of c-3skf), and the next layer is weighed only below the decision log's MAX_LAYERS;
  - the copy is mem_copy_async on the unified cache's BCS queue, chained behind the ring region's gate (ruling A,
    c-3bde: slot reuse is ordered on the device, never by a host status query), and the reorder is
    ggml_sycl_reorder_weight_gpu on the compute queue with the copy as its one dependency;
  - the AOS slot and the host lease are retained until the reorder's event, and the stage-table entry holds the SOA slot
    with the layout it really carries, SOA (ruling C: the slice's AOS tag is not the answer);
  - a throw from either submission aborts with "[MOE-EXEC] stream enqueue failed: %s" (design 9.3);
  - moe-stream-plan.cpp waits on nothing (.wait(), wait_and_throw()), queries no event status (get_info<), has no
    const_cast (ruling E) and calls no allocator: its slots come from the ring's one reservation;
  - the number of waits in ggml_sycl_mul_mat_id is the one master had at 84fb449da;
  - only a host lease whose retained owner is a host-pinned WEIGHT allocation of this device's host zones is copied
    (9y0p M4: any other lease would take the staged path), and only an item whose type has an AOS->SOA reorder in the
    one table (soa-reorder-types.hpp) is staged;
  - teardown gates the open region and holds one slot handle until the last use before the ring is released
    (ruling D, c-fjx1), and the ring is offered only on an in-order compute queue (the SOA overwrite of a reused region
    is ordered after its readers by that queue);
  - the backend context owns the phase, resets it after draining its queues and before the retained handles drain,
    ends the prompt at the first decode graph, and closes a layer's region after its down MUL_MAT_ID;
  - the trace line carries the plan's fixed format.

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
PHASE_REL = "ggml/src/ggml-sycl/moe-stream-plan.cpp"
PLAN_HPP_REL = "ggml/src/ggml-sycl/moe-stream-plan.hpp"
COMMON_REL = "ggml/src/ggml-sycl/common.hpp"

SYCL = read(SYCL_REL)
PHASE = read(PHASE_REL)
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
    skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
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


MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try"
ADMIT_BRANCH = "if (ne12 != 1)"
BEGIN_CALL = "moe_stream->begin_layer("
BEGIN_GUARD = "if (ids->ne[1] > 1 && ggml_sycl_moe_stream_prepare(ctx, src0, ids, stream_call)) {"
GRAPH_SINK = 'throw ggml_sycl_fallback_error("MUL_MAT_ID retained prompt batch cannot enter graph sink");'
SNAPSHOT = "prompt_ids_snapshot = prompt_ids_entry.host_ids;"
HYBRID_WITH_PLAN = "const bool moe_hybrid_with_plan ="

COPY_DECL = "const sycl::event copy = mem_copy_async("
REORDER_CALL = "ggml_sycl_reorder_weight_gpu("
RETAIN = "retain_handles_until_event({ aos, host }, reorder);"
TABLE_WRITE = "table_[key] = { soa, aos, reorder, GGML_LAYOUT_SOA };"
ABORT = 'GGML_ABORT("[MOE-EXEC] stream enqueue failed: %s", e.what());'
RETHROW = "catch (const ggml_sycl_fallback_error &) { throw; }"
TRACE_FMT = ("[MOE-EXEC] trace layer %d ubatch %u: %s regime=%s p=%.2f t_cpu=%.0f t_ring=%.0f window=%.0f "
             "staged=%d/%d bytes=%zu")
TRACE_EMIT = "op.log->trace_sink(op.log->sink_ctx, line);"

# Waits in ggml_sycl_mul_mat_id at master 84fb449da, counted on the normalized body: 5 `.wait()`, 3 `->wait()`,
# 8 `wait_and_throw()`. The prompt phase adds none.
MASTER_MMID_WAITS = 16
MMID_WAIT_RE = re.compile(r"\.wait\(\)|->wait\(\)|\bwait_and_throw\(\)")
ANY_WAIT_RE = re.compile(r"\bwait\s*\(|\bwait_and_throw\s*\(")


# ---- the consultation site --------------------------------------------------------------------------------------------


def claim_begin_layer_once_in_prompt_branch(sycl: str) -> bool:
    """One consultation site, in ggml_sycl_mul_mat_id's prompt ids-admission branch: after the graph-sink refusal and
    the ids snapshot, guarded by the prompt-batch predicate, routed on the snapshot, and before moe_hybrid_with_plan."""
    n = norm(sycl)
    if n.count(BEGIN_CALL) != 1:
        return False
    b = body(n, MMID_SIG)
    if b.count(BEGIN_CALL) != 1:
        return False
    at = b.find(ADMIT_BRANCH + " {")
    branch = body(b[at:], ADMIT_BRANCH) if at >= 0 else ""
    call_at = branch.find(BEGIN_CALL)
    if call_at < 0:
        return False
    args = call_args(branch[call_at:], BEGIN_CALL)
    return (ordered(branch[:call_at], "ctx.moe_graphs_disabled_once = true;", GRAPH_SINK, SNAPSHOT, BEGIN_GUARD)
            and bool(args) and "prompt_ids_snapshot.data(), prompt_ids_snapshot.size()" in args[0]
            and "*ctx.stream()" in args[0] and ordered(b, BEGIN_CALL, HYBRID_WITH_PLAN))


def test_begin_layer_once_in_prompt_branch():
    assert claim_begin_layer_once_in_prompt_branch(SYCL)


def test_begin_layer_mutants():
    n = norm(SYCL)
    start = n.find("ctx." + BEGIN_CALL)
    assert start >= 0, "mutant precondition: the consultation is a ctx.moe_stream->begin_layer( statement"
    stmt = n[start:n.find(";", start) + 1]
    # a second consultation
    assert not claim_begin_layer_once_in_prompt_branch(mutate(SYCL, stmt, stmt + " " + stmt))
    # the guard dropped
    assert not claim_begin_layer_once_in_prompt_branch(mutate(SYCL, BEGIN_GUARD, "{"))
    # consulted before the graph-sink refusal: a recorded graph could capture the ring copies
    moved = mutate(SYCL, GRAPH_SINK, stmt + " " + GRAPH_SINK).replace(BEGIN_GUARD + " " + stmt, BEGIN_GUARD, 1)
    assert not claim_begin_layer_once_in_prompt_branch(moved)
    # consulted after moe_hybrid_with_plan is computed
    late = mutate(SYCL, BEGIN_GUARD + " " + stmt, BEGIN_GUARD).replace(HYBRID_WITH_PLAN, stmt + " " + HYBRID_WITH_PLAN, 1)
    assert not claim_begin_layer_once_in_prompt_branch(late)
    # routed on something other than the snapshot
    assert not claim_begin_layer_once_in_prompt_branch(
        mutate(SYCL, "prompt_ids_snapshot.data(), prompt_ids_snapshot.size()", "nullptr, 0"))


# ---- the enqueue ------------------------------------------------------------------------------------------------------


def claim_copy_on_bcs_behind_region_gate(phase: str) -> bool:
    """One copy: the host lease's AOS bytes into the AOS slot, on the BCS queue, behind the region's gate."""
    n = norm(phase)
    args = call_args(n, "mem_copy_async(")
    return (len(args) == 1 and COPY_DECL in n and args[0].startswith("aos, host, item.src_bytes, ")
            and "cache.get_bcs_queue()" in args[0] and args[0].endswith("{ ring_->current_region_gate() }"))


def test_copy_on_bcs_behind_region_gate():
    assert claim_copy_on_bcs_behind_region_gate(PHASE)


def test_copy_mutants():
    assert not claim_copy_on_bcs_behind_region_gate(mutate(PHASE, "cache.get_bcs_queue()", "*ccs_"))
    assert not claim_copy_on_bcs_behind_region_gate(mutate(PHASE, "{ ring_->current_region_gate() }", "{}"))
    assert not claim_copy_on_bcs_behind_region_gate(
        mutate(PHASE, "aos, host, item.src_bytes, ", "aos, host, item.dst_bytes, "))


def claim_reorder_on_ccs_after_copy(phase: str) -> bool:
    """One reorder: the AOS slot to the SOA slot on the compute queue, its one dependency the copy."""
    n = norm(phase)
    args = call_args(n, REORDER_CALL)
    return (len(args) == 1 and args[0].startswith("*ccs_, ") and args[0].endswith(", { copy }")
            and ordered(n, COPY_DECL, REORDER_CALL))


def test_reorder_on_ccs_after_copy():
    assert claim_reorder_on_ccs_after_copy(PHASE)


def test_reorder_mutants():
    assert not claim_reorder_on_ccs_after_copy(mutate(PHASE, ", { copy })", ", {})"))
    assert not claim_reorder_on_ccs_after_copy(mutate(PHASE, ", { copy })", ", { ring_->current_region_gate() })"))
    assert not claim_reorder_on_ccs_after_copy(mutate(PHASE, REORDER_CALL + "*ccs_, ", REORDER_CALL + "cache.get_bcs_queue(), "))


def claim_handles_retained_and_entry_is_soa(phase: str, hpp: str) -> bool:
    """The AOS slot and the host lease live until the reorder's event; the table holds the SOA slot as the entry's
    handle with layout SOA, the field the stage table carries for Task 8's resolver (ruling C)."""
    n = norm(phase)
    staged = body(norm(hpp), "struct moe_stream_staged")
    return (n.count(RETAIN) == 1 and ordered(n, REORDER_CALL, RETAIN, TABLE_WRITE)
            and "ggml_layout_mode layout = GGML_LAYOUT_SOA;" in staged)


def test_handles_retained_and_entry_is_soa():
    assert claim_handles_retained_and_entry_is_soa(PHASE, PLAN_HPP)


def test_retain_mutants():
    assert not claim_handles_retained_and_entry_is_soa(mutate(PHASE, RETAIN, "retain_handles_until_event({ aos }, reorder);"), PLAN_HPP)
    assert not claim_handles_retained_and_entry_is_soa(mutate(PHASE, RETAIN, "retain_handles_until_event({ aos, host }, copy);"), PLAN_HPP)
    assert not claim_handles_retained_and_entry_is_soa(mutate(PHASE, TABLE_WRITE, "table_[key] = { soa, aos, reorder, GGML_LAYOUT_AOS };"), PLAN_HPP)
    assert not claim_handles_retained_and_entry_is_soa(
        PHASE, mutate(PLAN_HPP, "ggml_layout_mode layout = GGML_LAYOUT_SOA;", "ggml_layout_mode layout = GGML_LAYOUT_AOS;"))


def claim_submission_failure_aborts(phase: str) -> bool:
    """The copy, the reorder and the retention sit in one try block: a planned refusal is rethrown first (the alloc-zone
    contract's G-CATCH rule), and every other exception aborts with the design's message."""
    n = norm(phase)
    at = n.find("try {")
    while at >= 0:
        blk = body(n[at:], "try")
        end = n.find("{", at) + len(blk)
        if all(s in blk for s in (COPY_DECL, REORDER_CALL, RETAIN)):
            rest = n[end:].lstrip()
            if not rest.startswith(RETHROW):
                return False
            rest = rest[len(RETHROW):].lstrip()
            handler = body(rest, "catch (const std::exception & e)")
            return rest.startswith("catch (const std::exception & e) {") and ABORT in handler
        at = n.find("try {", at + 1)
    return False


def test_submission_failure_aborts():
    assert claim_submission_failure_aborts(PHASE)


def test_abort_mutants():
    assert not claim_submission_failure_aborts(mutate(PHASE, ABORT, 'GGML_ABORT("stream failed: %s", e.what());'))
    # the enqueue's handler narrowed (the phase has other std::exception handlers, so the mutant names this one)
    assert not claim_submission_failure_aborts(
        mutate(PHASE, "catch (const std::exception & e) { " + ABORT, "catch (const sycl::exception & e) { " + ABORT))
    # the planned refusal swallowed by the abort
    assert not claim_submission_failure_aborts(mutate(PHASE, RETHROW + " ", ""))
    # the copy moved out of the guarded block
    n = norm(PHASE)
    copy_stmt = n[n.find(COPY_DECL):n.find(";", n.find(COPY_DECL)) + 1]
    assert not claim_submission_failure_aborts(n.replace(copy_stmt, "", 1).replace("try {", copy_stmt + " try {", 1))


# ---- what the phase never does ----------------------------------------------------------------------------------------


def claim_phase_never_waits(phase: str) -> bool:
    n = norm(phase)
    return bool(n) and not ANY_WAIT_RE.search(n)


def test_phase_never_waits():
    assert claim_phase_never_waits(PHASE)


def test_phase_wait_mutants():
    assert not claim_phase_never_waits(mutate(PHASE, RETAIN, RETAIN + " reorder.wait();"))
    assert not claim_phase_never_waits(mutate(PHASE, RETAIN, RETAIN + " ccs_->wait_and_throw();"))


def claim_mul_mat_id_waits_unchanged(sycl: str) -> bool:
    b = body(norm(sycl), MMID_SIG)
    return bool(b) and len(MMID_WAIT_RE.findall(b)) == MASTER_MMID_WAITS


def test_mul_mat_id_waits_unchanged():
    assert claim_mul_mat_id_waits_unchanged(SYCL)


def test_mul_mat_id_wait_mutants():
    assert not claim_mul_mat_id_waits_unchanged(mutate(SYCL, SNAPSHOT, SNAPSHOT + " ctx.stream()->wait();"))
    assert not claim_mul_mat_id_waits_unchanged(mutate(SYCL, SNAPSHOT, SNAPSHOT + " ctx.stream()->wait_and_throw();"))


def claim_phase_queries_no_status_and_casts_no_const(phase: str) -> bool:
    n = norm(phase)
    return bool(n) and "get_info<" not in n and "const_cast" not in n


def test_phase_queries_no_status_and_casts_no_const():
    assert claim_phase_queries_no_status_and_casts_no_const(PHASE)


def test_status_and_const_cast_mutants():
    assert not claim_phase_queries_no_status_and_casts_no_const(
        mutate(PHASE, RETAIN, RETAIN + " (void) reorder.get_info<sycl::info::event::command_execution_status>();"))
    assert not claim_phase_queries_no_status_and_casts_no_const(
        mutate(PHASE, RETAIN, RETAIN + " (void) const_cast<moe_stream_ring *>(ring_.get());"))


ALLOCATOR_NAMES = ("malloc", "unified_alloc", "unified_allocate", "zone_allocate", "from_owned_alloc", "pinned_pool")


def claim_phase_allocates_only_ring_slots(phase: str) -> bool:
    """No allocator: every slot is ring_->acquire_slot() from the ring's one reservation."""
    n = norm(phase)
    return bool(n) and not any(a in n for a in ALLOCATOR_NAMES) and n.count("ring_->acquire_slot()") == 2


def test_phase_allocates_only_ring_slots():
    assert claim_phase_allocates_only_ring_slots(PHASE)


def test_allocator_mutants():
    assert not claim_phase_allocates_only_ring_slots(mutate(PHASE, RETAIN, RETAIN + " (void) sycl::malloc_device(16, *ccs_);"))
    assert not claim_phase_allocates_only_ring_slots(mutate(PHASE, "ring_->acquire_slot()", "mem_handle()"))


# ---- what is streamed -------------------------------------------------------------------------------------------------

OWNER_SIG = "static inline bool moe_stream_owner_streamable(const alloc_metadata & owner, int device)"
REORDERABLE_SIG = "static inline bool moe_stream_item_reorderable(const moe_stream_item & item, size_t slot_bytes)"
OWNER_CALL = "host.owner_metadata(&owner) && moe_stream_owner_streamable(owner, device)"
REORDERABLE_CALL = "moe_stream_item_reorderable(item, geo_.slot_bytes)"


def claim_only_direct_owned_typed_items_stream(phase: str, hpp: str) -> bool:
    """9y0p M4: the lease's retained owner is a host-pinned WEIGHT allocation of this device's host zones (the shape
    mem-ops.cpp copies directly); the type has an AOS->SOA reorder in the one table. Both before the copy."""
    h = norm(hpp)
    owner = body(h, OWNER_SIG)
    reorderable = body(h, REORDERABLE_SIG)
    n = norm(phase)
    return ("owner.tier == alloc_tier::HOST_PINNED" in owner and "owner.role == alloc_role::WEIGHT" in owner
            and "owner.host_zone != host_zone_id::COUNT" in owner and "owner.device == device" in owner
            and "ggml_sycl_soa_reorder_supported_type(item.type)" in reorderable
            and ordered(n, REORDERABLE_CALL, OWNER_CALL, "ring_->acquire_slot()", COPY_DECL))


def test_only_direct_owned_typed_items_stream():
    assert claim_only_direct_owned_typed_items_stream(PHASE, PLAN_HPP)


def test_streamed_item_mutants():
    assert not claim_only_direct_owned_typed_items_stream(mutate(PHASE, OWNER_CALL, "host.valid()"), PLAN_HPP)
    assert not claim_only_direct_owned_typed_items_stream(mutate(PHASE, REORDERABLE_CALL, "true"), PLAN_HPP)
    assert not claim_only_direct_owned_typed_items_stream(
        PHASE, mutate(PLAN_HPP, "owner.role == alloc_role::WEIGHT", "owner.role != alloc_role::OTHER"))
    assert not claim_only_direct_owned_typed_items_stream(
        PHASE, mutate(PLAN_HPP, "ggml_sycl_soa_reorder_supported_type(item.type)", "item.type != GGML_TYPE_COUNT"))


# ---- the policy gate --------------------------------------------------------------------------------------------------

RESOLVE_SIG = "void moe_stream_phase::resolve("
STAGE_SIG = "moe_stream_phase::stage_counts moe_stream_phase::stage("
RING_READY_SIG = "bool moe_stream_phase::ring_ready()"
POLICY_GUARD = "if (st.decision.choice == moe_exec_choice::RING && op.policy == moe_exec_policy::RING) {"
STAGE_CALL = "n = stage(li, st.decision.regime, cache, op);"
POLICY_FROM_CONFIG = "call.op.policy = ggml_sycl_moe_exec_config().policy;"
STAGE_CALL_RE = re.compile(r"(?<![:\w])stage\(")
RING_READY_CALL_RE = re.compile(r"(?<![:\w])ring_ready\(\)")


def claim_staging_only_under_explicit_ring(phase: str, sycl: str) -> bool:
    """Lead ruling c-3skf: until Task 8 (llama.cpp-o9uw) consumes a staged expert, only an explicit
    GGML_SYCL_MOE_EXEC=ring copies, reorders or reserves the ring, and AUTO only records the cost model's choice. The
    phase's one stage() call is the first statement under a guard that needs both the RING choice and the RING policy,
    the reserve is reachable only through stage() and ring_ready(), and the op's policy is the parsed config's."""
    n = norm(phase)
    return (POLICY_GUARD + " " + STAGE_CALL in body(n, RESOLVE_SIG)
            and len(STAGE_CALL_RE.findall(n)) == 1
            and len(RING_READY_CALL_RE.findall(n)) == 1
            and len(RING_READY_CALL_RE.findall(body(n, STAGE_SIG))) == 1
            and n.count("->reserve(") == 1 and "->reserve(" in body(n, RING_READY_SIG)
            and norm(sycl).count(POLICY_FROM_CONFIG) == 1)


def test_staging_only_under_explicit_ring():
    assert claim_staging_only_under_explicit_ring(PHASE, SYCL)


def test_policy_gate_mutants():
    # AUTO reaching the stage call: the guard tests the choice alone.
    assert not claim_staging_only_under_explicit_ring(mutate(PHASE, " && op.policy == moe_exec_policy::RING", ""), SYCL)
    # A guard that still admits AUTO.
    assert not claim_staging_only_under_explicit_ring(
        mutate(PHASE, "op.policy == moe_exec_policy::RING", "op.policy != moe_exec_policy::CPU"), SYCL)
    # A second stage call outside the guard.
    assert not claim_staging_only_under_explicit_ring(
        mutate(PHASE, "st.resolved = true;", STAGE_CALL + " st.resolved = true;"), SYCL)
    # The ring reserved outside stage().
    assert not claim_staging_only_under_explicit_ring(
        mutate(PHASE, "in_order_ = ccs.is_in_order();", "in_order_ = ccs.is_in_order(); ring_ready();"), SYCL)
    # The op's policy pinned to RING instead of read from GGML_SYCL_MOE_EXEC.
    assert not claim_staging_only_under_explicit_ring(
        PHASE, mutate(SYCL, POLICY_FROM_CONFIG, "call.op.policy = moe_exec_policy::RING;"))


BEGIN_UBATCH_SIG = "void moe_stream_phase::begin_ubatch(uint32_t n_tokens)"
BEGIN_LAYER_SIG = "moe_exec_decision moe_stream_phase::begin_layer("
END_PROMPT_SIG = "void moe_stream_phase::end_prompt()"
LATCH_CLEAR = "reserve_failed_ = false;"
LATCH_SET = "reserve_failed_ = true;"
LATCH_GUARD = "if (reserve_failed_ || geo_.n_slots == 0 || !in_order_) { return false; }"
NEW_UBATCH = "if (new_ubatch) { begin_ubatch(op.n_tokens);"


def claim_reserve_failure_lasts_one_graph(phase: str) -> bool:
    """Lead ruling c-i8rz: a refused reserve is not retried within a graph and is tried once more in the next graph of
    the same prompt. The latch is set only where the reserve fails, stops ring_ready() before any reserve, and is
    cleared by begin_ubatch, which every new micro-batch runs, and by end_prompt."""
    n = norm(phase)
    ready = body(n, RING_READY_SIG)
    return (n.count(LATCH_SET) == 1 and ordered(ready, LATCH_GUARD, "->reserve(", LATCH_SET)
            and LATCH_CLEAR in body(n, BEGIN_UBATCH_SIG)
            and LATCH_CLEAR in body(n, END_PROMPT_SIG)
            and NEW_UBATCH in body(n, BEGIN_LAYER_SIG))


def drop_in(text: str, signature: str, old: str) -> str:
    """`text` (normalized) with `old` removed once from the body of `signature`; the mutant must exist."""
    n = norm(text)
    b = body(n, signature)
    assert old in b, f"mutant precondition: {old!r} is not in {signature!r}"
    return n.replace(b, b.replace(old, "", 1), 1)


def test_reserve_failure_lasts_one_graph():
    assert claim_reserve_failure_lasts_one_graph(PHASE)


def test_reserve_latch_mutants():
    # The per-graph clear removed: a refused reserve would last the whole prompt.
    assert not claim_reserve_failure_lasts_one_graph(drop_in(PHASE, BEGIN_UBATCH_SIG, LATCH_CLEAR))
    # The prompt-end clear removed.
    assert not claim_reserve_failure_lasts_one_graph(drop_in(PHASE, END_PROMPT_SIG, LATCH_CLEAR))
    # No latch check: every staging layer would retry the reserve.
    assert not claim_reserve_failure_lasts_one_graph(mutate(PHASE, "if (reserve_failed_ || geo_.n_slots", "if (geo_.n_slots"))
    # The latch never set.
    assert not claim_reserve_failure_lasts_one_graph(mutate(PHASE, LATCH_SET, ""))
    # A new micro-batch that skips begin_ubatch.
    assert not claim_reserve_failure_lasts_one_graph(mutate(PHASE, NEW_UBATCH, "if (new_ubatch) {"))


DIAG_FMT = ("[MOE-EXEC] device %d: stream ring unavailable: %s (host_matrices_per_layer_max=%u matrix_bytes_max=%zu "
            "resident_expert_bytes=%zu n_slots=%u bytes=%zu)")
DIAG_EMIT = "op.log->sink(op.log->sink_ctx, line);"
DIAG_RULE_CALL = "moe_stream_ring_unavailable_rule(index_, geo_, in_order_, reserve_failed_)"
WARN_SIG = "void moe_stream_phase::warn_ring_unavailable(const moe_stream_op & op)"
RULE_SIG = "static inline const char * moe_stream_ring_unavailable_rule("
# Each rule's condition, then its name, in the order the helper tests them.
RULE_ORDER = (
    "idx.host_matrices_per_layer_max == 0", '"no host matrices of a streamable type"',
    "idx.matrix_bytes_max == 0", '"no matrix bytes"',
    "geo.n_slots == 0", '"ring exceeds 10% of resident expert bytes"',
    "!in_order", '"queue not in order"',
    "reserve_failed", '"reserve refused"',
)
WARN_GUARD = ("if (warned_unavailable_ || op.log == nullptr || op.log->sink == nullptr || "
              "(op.policy != moe_exec_policy::RING && op.log->trace < 1)) { return; }")
WARN_SET = "warned_unavailable_ = true;"
WARN_CLEAR = "warned_unavailable_ = false;"
WARN_CALL = "if (st.decision.reason == moe_exec_reason::RING_UNAVAILABLE) { warn_ring_unavailable(op); }"


def claim_ring_unavailable_diagnostic(phase: str, hpp: str, tree: dict) -> bool:
    """Lead ruling c-kcyf: a ring-unavailable answer is explained in the log. The fixed format exists once in the
    backend, carries the rule the helper names and the ring's inputs, and goes out through the decision log's WARN sink;
    the helper names each rule right after testing its condition, in a fixed order."""
    w = body(norm(phase), WARN_SIG)
    count = sum(norm(text).count(DIAG_FMT) for text in tree.values())
    return (count == 1 and ordered(w, DIAG_FMT, DIAG_RULE_CALL, DIAG_EMIT)
            and ordered(body(norm(hpp), RULE_SIG), *RULE_ORDER))


def test_ring_unavailable_diagnostic():
    assert claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, SYCL_TREE)


def test_ring_unavailable_diagnostic_mutants():
    renamed = mutate(PHASE, "n_slots=%u ", "slots=%u ")
    assert not claim_ring_unavailable_diagnostic(renamed, PLAN_HPP, {PHASE_REL: renamed})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + DIAG_FMT + '"'
    assert not claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, tree)
    assert not claim_ring_unavailable_diagnostic(mutate(PHASE, DIAG_EMIT, "(void) line;"), PLAN_HPP, SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(PHASE, mutate(PLAN_HPP, '"reserve refused"', '"reserve failed"'), SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(PHASE, mutate(PLAN_HPP, "geo.n_slots == 0", "geo.bytes == 0"), SYCL_TREE)


def claim_ring_unavailable_once_per_prompt(phase: str) -> bool:
    """Lead ruling c-kcyf: at most one diagnostic per prompt, and only under an explicit ring or a trace. The guard comes
    first, the flag is set before the line goes out, end_prompt clears it and begin_ubatch does not, and resolve calls
    it for the decision the layer ends with (after a refused reserve rewrote it), before the layer's report."""
    n = norm(phase)
    w = body(n, WARN_SIG)
    return (w.startswith("{ " + WARN_GUARD)
            and n.count(WARN_SET) == 1 and ordered(w, WARN_GUARD, WARN_SET, DIAG_EMIT)
            and n.count(WARN_CLEAR) == 1 and WARN_CLEAR in body(n, END_PROMPT_SIG)
            and WARN_CLEAR not in body(n, BEGIN_UBATCH_SIG)
            and ordered(body(n, RESOLVE_SIG), STAGE_CALL, WARN_CALL, "report(layer, st, n, op);"))


def test_ring_unavailable_once_per_prompt():
    assert claim_ring_unavailable_once_per_prompt(PHASE)


def test_ring_unavailable_once_mutants():
    # The flag never set: every unavailable layer would warn.
    assert not claim_ring_unavailable_once_per_prompt(mutate(PHASE, WARN_SET, ""))
    # The flag not consulted.
    assert not claim_ring_unavailable_once_per_prompt(mutate(PHASE, "if (warned_unavailable_ || ", "if ("))
    # Cleared per graph instead of per prompt.
    assert not claim_ring_unavailable_once_per_prompt(
        mutate(PHASE, "reserve_failed_ = false; table_.clear();", "reserve_failed_ = false; " + WARN_CLEAR + " table_.clear();"))
    # Never cleared: only the first prompt of the context would warn.
    assert not claim_ring_unavailable_once_per_prompt(drop_in(PHASE, END_PROMPT_SIG, WARN_CLEAR))
    # AUTO without a trace would warn too.
    assert not claim_ring_unavailable_once_per_prompt(
        mutate(PHASE, " || (op.policy != moe_exec_policy::RING && op.log->trace < 1)", ""))
    # Not called for the layer's decision.
    assert not claim_ring_unavailable_once_per_prompt(mutate(PHASE, WARN_CALL, ""))


REPORT_CALL = "report(layer, st, n, op);"
NEXT_LOOKUP = "const auto nxt = next < moe_exec_decision_log::MAX_LAYERS ? index_.layers.find(next) : index_.layers.end();"


def brace_depth(text: str, at: int) -> int:
    """Braces open at `at` in a body that holds no string literal (resolve's does not)."""
    return text.count("{", 0, at) - text.count("}", 0, at)


def claim_auto_still_reports(phase: str) -> bool:
    """The second half of c-3skf: under AUTO the decision log still records what the cost model would choose. resolve()
    calls report() once, at the top level of its body after the policy guard has closed, with no return before it."""
    r = body(norm(phase), RESOLVE_SIG)
    guard = r.find(POLICY_GUARD)
    at = r.find(REPORT_CALL)
    return (r.count(REPORT_CALL) == 1 and 0 <= guard < at and brace_depth(r, at) == 1 and "return" not in r)


def test_auto_still_reports():
    assert claim_auto_still_reports(PHASE)


def test_auto_report_mutants():
    # The spec reviewer's mutant: only an explicit ring reports.
    assert not claim_auto_still_reports(
        mutate(PHASE, REPORT_CALL, "if (op.policy == moe_exec_policy::RING) { " + REPORT_CALL + " }"))
    # AUTO returns before the report.
    assert not claim_auto_still_reports(
        mutate(PHASE, "stage_counts n; if (st.decision.choice",
               "stage_counts n; if (op.policy != moe_exec_policy::RING) { return; } if (st.decision.choice"))
    # No report at all.
    assert not claim_auto_still_reports(mutate(PHASE, REPORT_CALL, ""))


def claim_next_layer_bounded(phase: str) -> bool:
    """The next layer is weighed, and so noted in the decision log (which asserts layer < MAX_LAYERS), only below
    MAX_LAYERS: ggml_sycl_moe_stream_prepare bounds only the current layer."""
    b = body(norm(phase), BEGIN_LAYER_SIG)
    return NEXT_LOOKUP in b and b.count("index_.layers.find(next)") == 1


def test_next_layer_bounded():
    assert claim_next_layer_bounded(PHASE)


def test_next_layer_bound_mutants():
    assert not claim_next_layer_bounded(mutate(PHASE, NEXT_LOOKUP, "const auto nxt = index_.layers.find(next);"))
    assert not claim_next_layer_bounded(
        mutate(PHASE, "next < moe_exec_decision_log::MAX_LAYERS", "next <= moe_exec_decision_log::MAX_LAYERS"))


# ---- the ring's lifetime ----------------------------------------------------------------------------------------------

TEARDOWN_SIG = "void moe_stream_phase::teardown(sycl::event last_use)"


def claim_teardown_gates_before_release(phase: str) -> bool:
    """Ruling D: the open region is gated on the last use, one slot handle is held until it, then the ring releases;
    the destructor and end_prompt both tear down through it, and the ring is offered only on an in-order queue."""
    n = norm(phase)
    td = body(n, TEARDOWN_SIG)
    return (ordered(td, "close_region(last_use);", "retain_handles_until_event({ anchor_ }, last_use);",
                    "ring_->release();")
            and "teardown(" in body(n, "moe_stream_phase::~moe_stream_phase()")
            and "teardown(" in body(n, "void moe_stream_phase::end_prompt()")
            and "in_order_ = ccs.is_in_order();" in n)


def test_teardown_gates_before_release():
    assert claim_teardown_gates_before_release(PHASE)


def test_teardown_mutants():
    assert not claim_teardown_gates_before_release(mutate(PHASE, "close_region(last_use);", ""))
    assert not claim_teardown_gates_before_release(mutate(PHASE, "retain_handles_until_event({ anchor_ }, last_use);", ""))
    assert not claim_teardown_gates_before_release(mutate(PHASE, "in_order_ = ccs.is_in_order();", "in_order_ = true;"))


# ---- the context's wiring ---------------------------------------------------------------------------------------------

MEMBER = "std::unique_ptr<ggml_sycl::moe_stream_phase, ggml_sycl::moe_stream_phase_deleter> moe_stream;"
DTOR_SIG = "ggml_backend_sycl_context::~ggml_backend_sycl_context()"
END_PROMPT = ("sycl_ctx->graph_phase_is_decode.store(cached_is_decode, std::memory_order_relaxed); "
              "if (cached_is_decode && sycl_ctx->moe_stream) { sycl_ctx->moe_stream->end_prompt(); }")
DISPATCH = ("ggml_sycl_mul_mat_id(ctx, dst); ggml_sycl_moe_down_shadow_compare(ctx, dst); "
            "ggml_sycl_moe_stream_layer_done(ctx, dst); break;")
LAYER_DONE_SIG = "static void ggml_sycl_moe_stream_layer_done(ggml_backend_sycl_context & ctx, const ggml_tensor * dst)"


def claim_context_owns_and_ends_the_phase(common: str, sycl: str) -> bool:
    n = norm(sycl)
    dtor = body(n, DTOR_SIG)
    done = body(n, LAYER_DONE_SIG)
    return (MEMBER in norm(common)
            and ordered(dtor, "qptrs[dev][s]->wait();", "moe_stream.reset();", "ggml_sycl::drain_retained_handles(true);")
            and END_PROMPT in n and DISPATCH in n and n.count("moe_stream->layer_done(") == 1
            and ordered(done, "!ctx.moe_stream->region_open() || g_ggml_sycl_graph_recording",
                        "it->second.down_weight != dst->src[0]", "moe_stream->layer_done("))


def test_context_owns_and_ends_the_phase():
    assert claim_context_owns_and_ends_the_phase(COMMON, SYCL)


def test_context_mutants():
    assert not claim_context_owns_and_ends_the_phase(mutate(COMMON, MEMBER, ""), SYCL)
    assert not claim_context_owns_and_ends_the_phase(COMMON, mutate(SYCL, "moe_stream.reset();", ""))
    assert not claim_context_owns_and_ends_the_phase(
        COMMON, mutate(SYCL, "if (cached_is_decode && sycl_ctx->moe_stream) {", "if (sycl_ctx->moe_stream) {"))
    assert not claim_context_owns_and_ends_the_phase(COMMON, mutate(SYCL, " ggml_sycl_moe_stream_layer_done(ctx, dst); break;", " break;"))
    assert not claim_context_owns_and_ends_the_phase(
        COMMON, mutate(SYCL, "it->second.down_weight != dst->src[0]", "false"))


REORDER_DEF = "sycl::event ggml_sycl_reorder_weight_gpu(sycl::queue & queue,"


def claim_reorder_is_shared_not_copied(sycl: str, hpp: str) -> bool:
    """The ring submits the load path's own reorder dispatch: defined once in ggml-sycl.cpp without `static`, declared
    in moe-stream-plan.hpp, and the load-time fill still calls it."""
    n = norm(sycl)
    return (n.count(REORDER_DEF) == 1 and "static " + REORDER_DEF not in n
            and "sycl::event ggml_sycl_reorder_weight_gpu(sycl::queue & queue," in norm(hpp)
            and "ggml_sycl_reorder_weight_gpu(queue, static_cast<const uint8_t *>(temp_vram)," in n)


def test_reorder_is_shared_not_copied():
    assert claim_reorder_is_shared_not_copied(SYCL, PLAN_HPP)


def test_reorder_shared_mutants():
    assert not claim_reorder_is_shared_not_copied(mutate(SYCL, REORDER_DEF, "static " + REORDER_DEF), PLAN_HPP)
    assert not claim_reorder_is_shared_not_copied(SYCL, mutate(PLAN_HPP, REORDER_DEF, "sycl::event other_reorder(sycl::queue & queue,"))


# ---- the trace --------------------------------------------------------------------------------------------------------


def claim_trace_line(phase: str, tree: dict) -> bool:
    """The plan's fixed format, once in the backend, emitted through the decision log's trace sink."""
    count = sum(norm(text).count(TRACE_FMT) for text in tree.values())
    return count == 1 and ordered(norm(phase), TRACE_FMT, TRACE_EMIT)


def test_trace_line():
    assert claim_trace_line(PHASE, SYCL_TREE)


def test_trace_mutants():
    assert not claim_trace_line(mutate(PHASE, "window=%.0f", "win=%.0f"), {PHASE_REL: mutate(PHASE, "window=%.0f", "win=%.0f")})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + TRACE_FMT + '"'
    assert not claim_trace_line(PHASE, tree)
    assert not claim_trace_line(mutate(PHASE, TRACE_EMIT, "(void) line;"), {PHASE_REL: PHASE})


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
