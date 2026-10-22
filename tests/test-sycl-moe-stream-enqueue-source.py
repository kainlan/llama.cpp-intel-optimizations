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
    parsed config's. What an explicit ring stages, the resolver's STAGED_RING route consumes (llama.cpp-o9uw); under
    AUTO the decision log records what the cost model would choose and no bytes move (lead ruling c-3skf);
  - a refused ring reserve is not retried within a micro-batch and is tried once more in the next: the latch is cleared
    by begin_ubatch, which every new micro-batch runs, and by end_prompt. An expert loan (ruling c-tdra, llama.cpp-mwmr)
    can make room before the next micro-batch, so a per-prompt latch would waste the loan its own prompt asked for (lead
    ruling c-i8rz);
  - a layer answering RING_UNAVAILABLE under an explicit ring or a trace prints one WARN per prompt that names the rule
    and the ring's inputs, in a fixed format, through the decision log's WARN sink; the flag is cleared by end_prompt
    and never per micro-batch (lead ruling c-kcyf), and the phase logs nothing else of its own;
  - resolve() reports every layer it resolves, AUTO included, so the decision log keeps the cost model's choice
    (the second half of c-3skf), and the next layer is weighed only below the decision log's MAX_LAYERS;
  - the copy is mem_copy_async on the unified cache's BCS queue, chained behind the ring region's gate (ruling A,
    c-3bde: slot reuse is ordered on the device, never by a host status query) and behind the host lease's ready event,
    since the lease admits an entry still being filled; the reorder is ggml_sycl_reorder_weight_gpu on the compute queue
    with the copy as its one dependency;
  - the AOS slot, the host lease and the SOA slot are retained until the reorder's event, so each submission owns what
    it touches, and the stage-table entry holds the SOA slot with the layout it really carries, SOA (ruling C: the
    entry is the answer);
  - each slot is acquired with the layout it will hold, the AOS slot as AOS and the SOA slot as SOA (lead ruling
    c-o9uw-1), so the slice's tag agrees with the entry's layout by construction;
  - AOS-only staging (c-hmey): a matrix the device reads only as AOS takes ONE slot, its copy is its last submission
    (no reorder), the AOS slot and the host lease are retained until the copy, and its stage-table entry holds the AOS
    slot with layout AOS; how a matrix stages comes from the device's prompt kernels, which the backend answers from
    ggml_sycl_moe_query_route_capability (one fact, one source), asked once per micro-batch in the index build;
  - staging mirrors serving (llama.cpp-o9uw): the prompt-kernel predicate answers through the one helper the staged
    route refuses on, ggml_sycl_moe_staged_capability_refusal, which admits only a supported MMVQ_COMPAT kernel for a
    type with no direct-kernel requirement. Q1_0 and NVFP4 answer MMVQ_COMPAT in a prompt but run only through their
    direct kernel, which the recipe validator and the executor choice ask through moe_type_needs_direct_kernel, so
    the ring stages none of them and sizes no slot for them;
  - a matrix whose weight the oneDNN batched prompt executor claims stages NONE: the backend answers the index build's
    library-claims predicate with ggml_sycl_moe_pp_onednn_batched_claims_tensor, the check the staged route refuses a
    weight on (c-las5), so the ring copies nothing the route would refuse and sizes no slot for it (one fact, one
    source; llama.cpp-o9uw). The predicate asks that check's plan-taking form with call.plan, the plan the index
    build holds, so it never reads the device's plan a second time (lead ruling c-gi44); only the check's
    two-argument entry point, the batched executor's own route, reads the device's plan, by passing no plan;
  - a region closes on an event after its last copy and its last reorder: AOS-only items go first, so the region's
    last submission is a reorder when it holds a SOA item and the last copy otherwise;
  - when a region's last use or its new one holds an AOS-only item, the new copies also wait for the region's read
    gate: the gate layer_done sets belongs to the region holding the next layer, so it precedes the readers of an
    AOS-only slot (and of a SOA slot that AOS-only items moved off a fixed odd index). The read gate is the barrier
    layer_done submitted behind the down op of the layer the region last held (lead ruling c-o9uw-1), and a
    compute-queue barrier submitted at the first copy stands in when no layer_done recorded one. Each region's last
    use (its layer, AOS-only flag and read gate) is kept in one place, moe_stream_region_reads, which stage() refills,
    layer_done() gates and reset() forgets (test-moe-stream-phase drives it). One helper, reset_regions, forgets every
    region's keys and resets it, and both a new ring and teardown run it, so neither a new ring nor a reused phase
    inherits a gate or a layer;
  - a throw from either submission aborts with "[MOE-EXEC] stream enqueue failed: %s" (design 9.3);
  - moe-stream-plan.cpp waits on nothing (.wait(), wait_and_throw()), queries no event status (get_info<), has no
    const_cast (ruling E) and calls no allocator: its slots come from the ring's one reservation;
  - the number of waits in ggml_sycl_mul_mat_id is the one master had at 84fb449da;
  - only a host lease whose retained owner is a host-pinned WEIGHT allocation of this device's host zones is copied
    (llama.cpp-9y0p: any other lease would take the staged path), and only an item staged SOA (its type has an AOS->SOA
    reorder in the one table, soa-reorder-types.hpp) or AOS-only that fits the slot is staged;
  - teardown gates the open region and holds one slot handle until the last use before the ring is released
    (ruling D, c-fjx1), and the ring is offered only on an in-order compute queue (the SOA overwrite of a reused region
    is ordered after its readers by that queue);
  - the backend context owns the phase, resets it after draining its queues and before the retained handles drain,
    ends the prompt at the first decode graph, and closes a layer's region after its down MUL_MAT_ID;
  - begin_layer and resolve decide through the header's pure predicates (a micro-batch's start, resolving the next
    layer ahead, staging now), which test-moe-stream-phase pins, and never restate them;
  - the stage trace line carries the plan's fixed fields under its own "[MOE-EXEC-TRACE] stage layer" prefix (plan
    amendment for Task 15, lead ruling c-xod7-q1), byte-identical to master's; the staged matrices' split by layout,
    soa= and aos_only=, is a separate "[MOE-EXEC-TRACE] stage layouts layer" line under the same trace guard (lead
    ruling c-o9uw-1, c-9wgo).

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


_PRIVATE_IF_RE = re.compile(
    r"\s*#\s*if(?:\s+defined\s*\(\s*GGML_SYCL_PRIVATE_TESTING\s*\)|def\s+GGML_SYCL_PRIVATE_TESTING)\s*$")
_PP_OPEN_RE = re.compile(r"\s*#\s*if")
_PP_ARM_RE = re.compile(r"\s*#\s*(?:else|elif)\b")
_PP_END_RE = re.compile(r"\s*#\s*endif\b")


def production(text: str) -> str:
    """`text` without its GGML_SYCL_PRIVATE_TESTING arms. The test seams there are compiled into the private-testing
    fixtures only, so they neither add to nor stand in for the production sites the claims count. An #else or #elif arm
    is production and stays, and a seam moved outside its block is counted again. Line count is kept."""
    out, depth, dropping = [], 0, False
    for line in text.split("\n"):
        if depth == 0:
            if _PRIVATE_IF_RE.match(line):
                depth, dropping = 1, True
                out.append("")
            else:
                out.append(line)
            continue
        if _PP_OPEN_RE.match(line):
            depth += 1
        elif _PP_END_RE.match(line):
            depth -= 1
            if depth == 0:
                out.append("")
                continue
        elif depth == 1 and _PP_ARM_RE.match(line):
            dropping = False
            out.append("")
            continue
        out.append("" if dropping else line)
    return "\n".join(out)


SYCL_REL = "ggml/src/ggml-sycl/ggml-sycl.cpp"
PHASE_REL = "ggml/src/ggml-sycl/moe-stream-plan.cpp"
PLAN_HPP_REL = "ggml/src/ggml-sycl/moe-stream-plan.hpp"
COMMON_REL = "ggml/src/ggml-sycl/common.hpp"

SYCL = read(SYCL_REL)
# The production phase: its PRIVATE_TESTING seams reserve a ring and acquire a slot for a device case, and are not
# production sites (test_claims_count_production_sites_only).
PHASE = production(read(PHASE_REL))
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
RETAIN = "retain_handles_until_event({ aos, host, soa }, reorder);"
TABLE_WRITE = "table_[key] = { soa, aos, reorder, GGML_LAYOUT_SOA };"
ABORT = 'GGML_ABORT("[MOE-EXEC] stream enqueue failed: %s", e.what());'
RETHROW = "catch (const ggml_sycl_fallback_error &) { throw; }"
TRACE_FMT = ("[MOE-EXEC-TRACE] stage layer %d ubatch %u: %s regime=%s p=%.2f t_cpu=%.0f t_ring=%.0f window=%.0f "
             "staged=%d/%d bytes=%zu")
TRACE_EMIT = "op.log->trace_sink(op.log->sink_ctx, line);"

# Waits in ggml_sycl_mul_mat_id at master 84fb449da, counted on the normalized body: 5 `.wait()`, 3 `->wait()`,
# 8 `wait_and_throw()`. The prompt phase adds none.
MASTER_MMID_WAITS = 16
MMID_WAIT_RE = re.compile(r"\.wait\(\)|->wait\(\)|\bwait_and_throw\(\)")
ANY_WAIT_RE = re.compile(r"\bwait\s*\(|\bwait_and_throw\s*\(")


# ---- the consultation site -------------------------------------------------------------------------------------------


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
    late = mutate(SYCL, BEGIN_GUARD + " " + stmt, BEGIN_GUARD)
    late = late.replace(HYBRID_WITH_PLAN, stmt + " " + HYBRID_WITH_PLAN, 1)
    assert not claim_begin_layer_once_in_prompt_branch(late)
    # routed on something other than the snapshot
    assert not claim_begin_layer_once_in_prompt_branch(
        mutate(SYCL, "prompt_ids_snapshot.data(), prompt_ids_snapshot.size()", "nullptr, 0"))


# ---- the enqueue -----------------------------------------------------------------------------------------------------


HOST_RESOLVE = "const resolved_ptr hr = host.resolve(device);"
COPY_DEPS = "std::vector<sycl::event> deps = { ring_->current_region_gate() };"
READY_DEP = "if (hr.has_ready_event) { deps.push_back(hr.ready_event); }"


def claim_copy_on_bcs_behind_region_gate(phase: str) -> bool:
    """One copy: the host lease's AOS bytes into the AOS slot, on the BCS queue, behind the region's gate and the
    lease's ready event. The lease resolves with require_ready=false, so an entry still being filled is admitted, and
    its ready event orders the copy after that fill on the device. The lease is resolved once, for that event."""
    n = norm(phase)
    args = call_args(n, "mem_copy_async(")
    return (len(args) == 1 and COPY_DECL in n and args[0].startswith("aos, host, item.src_bytes, ")
            and "cache.get_bcs_queue()" in args[0] and args[0].endswith(", deps")
            and n.count("host.resolve(") == 1 and ordered(n, HOST_RESOLVE, COPY_DEPS, READY_DEP, COPY_DECL))


def test_copy_on_bcs_behind_region_gate():
    assert claim_copy_on_bcs_behind_region_gate(PHASE)


def test_copy_mutants():
    assert not claim_copy_on_bcs_behind_region_gate(mutate(PHASE, "cache.get_bcs_queue()", "*ccs_"))
    assert not claim_copy_on_bcs_behind_region_gate(mutate(PHASE, COPY_DEPS, "std::vector<sycl::event> deps;"))
    # The host lease's ready event dropped: a copy could read an entry before its fill lands.
    assert not claim_copy_on_bcs_behind_region_gate(mutate(PHASE, READY_DEP, ""))
    assert not claim_copy_on_bcs_behind_region_gate(
        mutate(PHASE, "if (hr.has_ready_event)", "if (!hr.has_ready_event)"))
    # The copy given the gate alone, as before the ready event was honoured.
    assert not claim_copy_on_bcs_behind_region_gate(
        mutate(PHASE, "cache.get_bcs_queue(), deps)", "cache.get_bcs_queue(), { ring_->current_region_gate() })"))
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
    assert not claim_reorder_on_ccs_after_copy(
        mutate(PHASE, REORDER_CALL + "*ccs_, ", REORDER_CALL + "cache.get_bcs_queue(), "))


def claim_handles_retained_and_entry_is_soa(phase: str, hpp: str) -> bool:
    """The AOS slot, the host lease and the SOA slot live until the reorder's event, so the submission's ownership does
    not lean on the table (CLAUDE.md: retain the handles behind a kernel's raw pointers for the queued work); the table
    holds the SOA slot as the entry's handle with layout SOA, the field the resolver's STAGED_RING route reads
    (ruling C)."""
    n = norm(phase)
    staged = body(norm(hpp), "struct moe_stream_staged")
    return (n.count(RETAIN) == 1 and ordered(n, REORDER_CALL, RETAIN, TABLE_WRITE)
            and "ggml_layout_mode layout = GGML_LAYOUT_SOA;" in staged)


def test_handles_retained_and_entry_is_soa():
    assert claim_handles_retained_and_entry_is_soa(PHASE, PLAN_HPP)


def test_retain_mutants():
    assert not claim_handles_retained_and_entry_is_soa(
        mutate(PHASE, RETAIN, "retain_handles_until_event({ aos }, reorder);"), PLAN_HPP)
    # The SOA slot left to the table's hold.
    assert not claim_handles_retained_and_entry_is_soa(
        mutate(PHASE, RETAIN, "retain_handles_until_event({ aos, host }, reorder);"), PLAN_HPP)
    assert not claim_handles_retained_and_entry_is_soa(
        mutate(PHASE, RETAIN, "retain_handles_until_event({ aos, host, soa }, copy);"), PLAN_HPP)
    assert not claim_handles_retained_and_entry_is_soa(
        mutate(PHASE, TABLE_WRITE, "table_[key] = { soa, aos, reorder, GGML_LAYOUT_AOS };"), PLAN_HPP)
    assert not claim_handles_retained_and_entry_is_soa(
        PHASE,
        mutate(PLAN_HPP, "ggml_layout_mode layout = GGML_LAYOUT_SOA;", "ggml_layout_mode layout = GGML_LAYOUT_AOS;"))


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


# ---- what the phase never does ---------------------------------------------------------------------------------------


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
    """No allocator: every slot is ring_->acquire_slot(layout) from the ring's one reservation."""
    n = norm(phase)
    return bool(n) and not any(a in n for a in ALLOCATOR_NAMES) and n.count("ring_->acquire_slot(") == 2


def test_phase_allocates_only_ring_slots():
    assert claim_phase_allocates_only_ring_slots(PHASE)


def test_allocator_mutants():
    assert not claim_phase_allocates_only_ring_slots(
        mutate(PHASE, RETAIN, RETAIN + " (void) sycl::malloc_device(16, *ccs_);"))
    assert not claim_phase_allocates_only_ring_slots(
        mutate(PHASE, "ring_->acquire_slot(GGML_LAYOUT_SOA)", "mem_handle()"))


# ---- what is streamed ------------------------------------------------------------------------------------------------

OWNER_SIG = "static inline bool moe_stream_owner_streamable(const alloc_metadata & owner, int device)"
REORDERABLE_SIG = "static inline bool moe_stream_item_reorderable(const moe_stream_item & item, size_t slot_bytes)"
AOS_FITS_SIG = "static inline bool moe_stream_item_aos_fits(const moe_stream_item & item, size_t slot_bytes)"
OWNER_CALL = "host.owner_metadata(&owner) && moe_stream_owner_streamable(owner, device)"
REORDERABLE_CALL = "moe_stream_item_reorderable(item, geo_.slot_bytes)"
TYPED_GUARD = ("if (item.staging == moe_stream_staging::NONE || "
               "!(aos_only ? moe_stream_item_aos_fits(item, geo_.slot_bytes) "
               ": " + REORDERABLE_CALL + ")) { ++n.untyped; continue; }")


def claim_only_direct_owned_typed_items_stream(phase: str, hpp: str) -> bool:
    """llama.cpp-9y0p: the lease's retained owner is a host-pinned WEIGHT allocation of this device's host zones (the
    shape mem-ops.cpp copies directly). The item is staged SOA, whose type has an AOS->SOA reorder in the one table, or
    AOS-only (c-hmey), and fits the slot as it stages; anything staged NONE is counted untyped. Both before the copy."""
    h = norm(hpp)
    owner = body(h, OWNER_SIG)
    reorderable = body(h, REORDERABLE_SIG)
    aos_fits = body(h, AOS_FITS_SIG)
    n = norm(phase)
    return ("owner.tier == alloc_tier::HOST_PINNED" in owner and "owner.role == alloc_role::WEIGHT" in owner
            and "owner.host_zone != host_zone_id::COUNT" in owner and "owner.device == device" in owner
            and "ggml_sycl_soa_reorder_supported_type(item.type)" in reorderable
            and "detail::moe_stream_item_rows_bytes(item) != 0" in aos_fits
            and "item.src_bytes <= slot_bytes" in aos_fits
            and ordered(n, TYPED_GUARD, OWNER_CALL, "ring_->acquire_slot(", COPY_DECL))


def test_only_direct_owned_typed_items_stream():
    assert claim_only_direct_owned_typed_items_stream(PHASE, PLAN_HPP)


def test_streamed_item_mutants():
    assert not claim_only_direct_owned_typed_items_stream(mutate(PHASE, OWNER_CALL, "host.valid()"), PLAN_HPP)
    assert not claim_only_direct_owned_typed_items_stream(mutate(PHASE, REORDERABLE_CALL, "true"), PLAN_HPP)
    assert not claim_only_direct_owned_typed_items_stream(
        PHASE, mutate(PLAN_HPP, "owner.role == alloc_role::WEIGHT", "owner.role != alloc_role::OTHER"))
    assert not claim_only_direct_owned_typed_items_stream(
        PHASE, mutate(PLAN_HPP, "ggml_sycl_soa_reorder_supported_type(item.type)", "item.type != GGML_TYPE_COUNT"))
    # An AOS-only item copied without checking that it fits the slot.
    assert not claim_only_direct_owned_typed_items_stream(
        mutate(PHASE, "moe_stream_item_aos_fits(item, geo_.slot_bytes)", "true"), PLAN_HPP)
    assert not claim_only_direct_owned_typed_items_stream(
        PHASE, mutate(PLAN_HPP, "item.src_bytes <= slot_bytes", "true"))
    # An item staged NONE copied anyway.
    assert not claim_only_direct_owned_typed_items_stream(
        mutate(PHASE, "item.staging == moe_stream_staging::NONE || ", ""), PLAN_HPP)


# ---- AOS-only staging (c-hmey) ---------------------------------------------------------------------------------------

AOS_ONLY_FLAG = "const bool aos_only = item.staging == moe_stream_staging::AOS_ONLY;"
SLOTS_CHECK = "if (ring_->slots_free() < moe_stream_staging_slots(item.staging)) { ++n.exhausted; continue; }"
AOS_ACQUIRE = "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_AOS);"
SOA_ACQUIRE = "mem_handle soa = aos_only ? mem_handle() : ring_->acquire_slot(GGML_LAYOUT_SOA);"
STAGING_SLOTS_SIG = "static inline uint32_t moe_stream_staging_slots(moe_stream_staging staging)"
STAGING_SLOTS_BODY = ("{ return staging == moe_stream_staging::SOA ? 2u : "
                      "staging == moe_stream_staging::AOS_ONLY ? 1u : 0u; }")
REGION_SIZE = "li.region_slots += static_cast<int>(moe_stream_staging_slots(item.staging));"
INDEX_BUILD_SIG = "static inline moe_stream_plan_index moe_stream_plan_index_build("
RING_GEOMETRY_SIG = "static inline moe_stream_ring_geometry moe_stream_plan_ring_geometry("
# The one place the ring is sized (llama.cpp-kjef): the phase's prepare() and the auto-ubatch fit both call the helper.
RING_SIZE_CALL = "moe_stream_ring_size(index.region_slots_max, index.matrix_bytes_max, index.resident_expert_bytes)"


def claim_aos_only_item_takes_one_slot(phase: str, hpp: str) -> bool:
    """An AOS-only matrix takes ONE slot and a SOA matrix two, both when the ring is sized (each layer's slots summed
    from the one slot-count helper, the ring sized from the heaviest layer's) and when a stage acquires them: the region
    is checked for that many free slots, then the AOS slot is acquired, and the SOA slot only for a SOA item. Each is
    acquired with the layout it will hold (lead ruling c-o9uw-1)."""
    h = norm(hpp)
    n = norm(phase)
    s = body(n, STAGE_SIG)
    return (body(h, STAGING_SLOTS_SIG) == STAGING_SLOTS_BODY and REGION_SIZE in body(h, INDEX_BUILD_SIG)
            and RING_SIZE_CALL in body(h, RING_GEOMETRY_SIG)
            and "moe_stream_plan_ring_geometry(" in body(n, PREPARE_SIG)
            and n.count("ring_->acquire_slot(") == 2
            and ordered(s, AOS_ONLY_FLAG, SLOTS_CHECK, AOS_ACQUIRE, SOA_ACQUIRE, COPY_DECL))


def test_aos_only_item_takes_one_slot():
    assert claim_aos_only_item_takes_one_slot(PHASE, PLAN_HPP)


def test_aos_only_slot_mutants():
    # An AOS-only item takes a SOA slot too.
    assert not claim_aos_only_item_takes_one_slot(
        mutate(PHASE, SOA_ACQUIRE, "mem_handle soa = ring_->acquire_slot(GGML_LAYOUT_SOA);"), PLAN_HPP)
    assert not claim_aos_only_item_takes_one_slot(
        mutate(PHASE, SOA_ACQUIRE,
               "mem_handle soa = aos_only ? ring_->acquire_slot(GGML_LAYOUT_AOS) : "
               "ring_->acquire_slot(GGML_LAYOUT_SOA);"),
        PLAN_HPP)
    # A slot tagged with the layout it will not hold: the slice's tag would disagree with the entry's layout.
    assert not claim_aos_only_item_takes_one_slot(
        mutate(PHASE, AOS_ACQUIRE, "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_SOA);"), PLAN_HPP)
    assert not claim_aos_only_item_takes_one_slot(
        mutate(PHASE, SOA_ACQUIRE, "mem_handle soa = aos_only ? mem_handle() : ring_->acquire_slot(GGML_LAYOUT_AOS);"),
        PLAN_HPP)
    # The ring sized, or the region checked, as if every item took two.
    assert not claim_aos_only_item_takes_one_slot(
        PHASE, mutate(PLAN_HPP, "moe_stream_staging::AOS_ONLY ? 1u", "moe_stream_staging::AOS_ONLY ? 2u"))
    assert not claim_aos_only_item_takes_one_slot(
        mutate(PHASE, "ring_->slots_free() < moe_stream_staging_slots(item.staging)", "ring_->slots_free() < 2u"),
        PLAN_HPP)
    # The ring sized from the matrix count, as before c-hmey.
    assert not claim_aos_only_item_takes_one_slot(
        PHASE, mutate(PLAN_HPP, RING_SIZE_CALL, "moe_stream_ring_size(2 * index.host_matrices_per_layer_max, "
                                                "index.matrix_bytes_max, index.resident_expert_bytes)"))
    # The branch keyed on the SOA staging.
    assert not claim_aos_only_item_takes_one_slot(
        mutate(PHASE, AOS_ONLY_FLAG, "const bool aos_only = item.staging == moe_stream_staging::SOA;"), PLAN_HPP)


AOS_ONLY_SUBMIT = "if (aos_only) { retain_handles_until_event({ aos, host }, copy); copied = copy; } else {"
AOS_ONLY_TABLE = "table_[key] = { mem_handle(), aos, copied, GGML_LAYOUT_AOS };"
AOS_ONLY_RECORD = ("if (aos_only) { region_last_ = copied; aos_only_staged = true; " + AOS_ONLY_TABLE
                   + " ++n.aos_only; } else { region_last_ = reorder; " + TABLE_WRITE + " ++n.soa; }")


def claim_aos_only_entry_reads_in_place(phase: str) -> bool:
    """An AOS-only matrix is read where its copy lands: the copy is its only submission, the AOS slot and the host
    lease are retained until the copy's event (the submission owns what it touches), and the stage-table entry holds
    no SOA slot, the AOS slot, the copy's event and layout AOS (ruling C: the layout comes from the entry, which for an
    AOS-only matrix is AOS). The reorder runs only for a SOA item."""
    n = norm(phase)
    s = body(n, STAGE_SIG)
    return (n.count(AOS_ONLY_SUBMIT) == 1 and n.count(AOS_ONLY_TABLE) == 1 and AOS_ONLY_RECORD in s
            and ordered(s, COPY_DECL, AOS_ONLY_SUBMIT, REORDER_CALL, RETAIN, AOS_ONLY_RECORD))


def test_aos_only_entry_reads_in_place():
    assert claim_aos_only_entry_reads_in_place(PHASE)


def test_aos_only_entry_mutants():
    # Tagged SOA: the resolver would read the slot as a layout it does not hold.
    assert not claim_aos_only_entry_reads_in_place(
        mutate(PHASE, AOS_ONLY_TABLE, "table_[key] = { mem_handle(), aos, copied, GGML_LAYOUT_SOA };"))
    # A SOA slot named that no reorder wrote.
    assert not claim_aos_only_entry_reads_in_place(
        mutate(PHASE, AOS_ONLY_TABLE, "table_[key] = { soa, aos, copied, GGML_LAYOUT_AOS };"))
    # The slot left to the table's hold.
    assert not claim_aos_only_entry_reads_in_place(
        mutate(PHASE, "retain_handles_until_event({ aos, host }, copy);",
               "retain_handles_until_event({ host }, copy);"))
    # A reorder submitted for an AOS-only item as well.
    assert not claim_aos_only_entry_reads_in_place(mutate(PHASE, "copied = copy; } else {", "copied = copy; } {"))
    # The entry's event a stale reorder instead of the copy.
    assert not claim_aos_only_entry_reads_in_place(
        mutate(PHASE, AOS_ONLY_TABLE, "table_[key] = { mem_handle(), aos, reorder, GGML_LAYOUT_AOS };"))


AOS_ONLY_LAMBDA = ("const auto aos_only_item = [](const moe_stream_item & it) { return it.staging == "
                   "moe_stream_staging::AOS_ONLY; };")
PARTITION = "std::stable_partition(items.begin(), items.end(), aos_only_item);"
STAGE_CLOSE = "close_region(region_last_);"
ITEM_LOOP = "for (const moe_stream_item & item : items) {"


def claim_region_closes_after_last_copy(phase: str) -> bool:
    """c-hmey: the event a region closes on follows its last copy as well as its last reorder. The stage puts its
    AOS-only items first, so its last submission is a reorder when it holds a SOA item (and that reorder follows its own
    copy, which the in-order copy queue runs after every earlier copy) and the last copy otherwise; each item records
    its own last event as the region's, and the next stage closes the region on it."""
    s = body(norm(phase), STAGE_SIG)
    return (s.count(PARTITION) == 1 and s.count(STAGE_CLOSE) == 1 and s.count("region_last_ = copied;") == 1
            and s.count("region_last_ = reorder;") == 1
            and ordered(s, STAGE_CLOSE, AOS_ONLY_LAMBDA, PARTITION, ITEM_LOOP, AOS_ONLY_RECORD))


def test_region_closes_after_last_copy():
    assert claim_region_closes_after_last_copy(PHASE)


def test_region_close_mutants():
    # No partition: a SOA item's reorder could close a region before a later AOS-only copy into it.
    assert not claim_region_closes_after_last_copy(mutate(PHASE, PARTITION, ""))
    # SOA items first.
    assert not claim_region_closes_after_last_copy(
        mutate(PHASE, AOS_ONLY_LAMBDA,
               AOS_ONLY_LAMBDA.replace("moe_stream_staging::AOS_ONLY", "moe_stream_staging::SOA")))
    # An AOS-only region closed on a reorder of another region.
    assert not claim_region_closes_after_last_copy(
        mutate(PHASE, "region_last_ = copied;", "region_last_ = reorder;"))
    # The region closed on the last reorder only, as before c-hmey.
    assert not claim_region_closes_after_last_copy(mutate(PHASE, "region_last_ = copied; ", ""))


REUSE_COND = ("const bool reuse_after_readers = region_reads_.aos_only[cur_region_] || "
              "std::any_of(items.begin(), items.end(), aos_only_item);")
REUSE_TAKE = ("if (reuse_after_readers && !reuse_gate) { reuse_gate = region_reads_.read_gate[cur_region_] ? "
              "*region_reads_.read_gate[cur_region_] : ccs_->ext_oneapi_submit_barrier(); }")
REUSE_DEP = "if (reuse_gate) { deps.push_back(*reuse_gate); }"
COPIED_IN = "copied_in = true;"
REUSE_RECORD = "if (copied_in) { region_reads_.refill(cur_region_, items.front().layer, aos_only_staged); }"
REGION_FLAGS_RESET = "region_reads_.reset();"
# The regions' last uses, kept in one place (moe-stream-plan.hpp) and held by the phase over sycl::event.
READS_SIG = "template <typename event_t> struct moe_stream_region_reads"
READS_FIELDS = "int layer[2] = { -1, -1 }; bool aos_only[2] = { false, false }; std::optional<event_t> read_gate[2];"
READS_MEMBER = "moe_stream_region_reads<sycl::event> region_reads_;"
REFILL_SIG = "void refill(int region, int l, bool left_aos_only)"
REFILL_BODY = "{ layer[region] = l; aos_only[region] = left_aos_only; read_gate[region].reset(); }"
READS_DONE_SIG = "void layer_done(int l, const event_t & down_done)"
READS_DONE_BODY = "{ for (int r = 0; r < 2; ++r) { if (layer[r] == l) { read_gate[r] = down_done; } } }"
READS_RESET_SIG = "void reset()"
READS_RESET_BODY = "{ for (int r = 0; r < 2; ++r) { layer[r] = -1; aos_only[r] = false; read_gate[r].reset(); } }"
RESET_REGIONS_SIG = "void moe_stream_phase::reset_regions()"
REGION_STATE_RESET = ("region_keys_[0].clear(); region_keys_[1].clear(); cur_region_ = 0; region_submitted_ = false; "
                      "region_last_ = sycl::event();")
RING_READY_RESET = "ring_ = std::move(ring); reset_regions(); return true;"
TEARDOWN_RESET = "table_.clear(); reset_regions();"
PHASE_LAYER_DONE_SIG = "void moe_stream_phase::layer_done(int layer, sycl::event down_done)"
READ_GATE_RECORD = "region_reads_.layer_done(layer, down_done);"


def claim_aos_only_reuse_waits_for_readers(phase: str, hpp: str = PLAN_HPP) -> bool:
    """When a region's last use or the stage reusing it holds an AOS-only item, a copy may rewrite a slot a kernel
    read in that last use, and the region's gate does not follow those readers (layer_done gates the region holding the
    next layer). So every copy of the stage also waits for the region's read gate (lead ruling c-o9uw-1): the barrier
    layer_done submitted on the readers' queue (c-las5) behind the down op of the layer the region last held, recorded
    for the region holding that layer. Without one, the copies wait for a compute-queue barrier submitted at the first
    copy. A copy going in makes the stage the region's last use: its layer, whether it left an AOS-only slot, and no
    read gate yet. A new ring starts with no layer, no AOS-only slot and no read gate, and teardown leaves none behind:
    both run the one helper that forgets them, so the two cannot drift apart. The three facts per region live in one
    struct (moe_stream_region_reads), whose refill, layer_done and reset are the only writers, and the phase holds no
    copy of its own."""
    n = norm(phase)
    s = body(n, STAGE_SIG)
    done = body(n, PHASE_LAYER_DONE_SIG)
    reset = body(n, RESET_REGIONS_SIG)
    h = norm(hpp)
    at = h.find(READS_SIG + " {")
    reads = body(h[at:], READS_SIG) if at >= 0 else ""
    return (s.count(REUSE_TAKE) == 1 and s.count(REUSE_DEP) == 1 and s.count(REUSE_RECORD) == 1
            and s.count(COPIED_IN) == 1 and "bool copied_in = false;" in s
            and ordered(s, PARTITION, REUSE_COND, ITEM_LOOP, COPY_DEPS, READY_DEP, REUSE_TAKE, REUSE_DEP, COPY_DECL,
                        COPIED_IN, AOS_ONLY_RECORD, REUSE_RECORD)
            and n.count(READ_GATE_RECORD) == 1 and done.count(READ_GATE_RECORD) == 1
            and ordered(done, "if (layer != cur_layer_) { return; }", READ_GATE_RECORD)
            and h.count(READS_SIG + " {") == 1 and READS_FIELDS in reads
            and body(reads, REFILL_SIG) == REFILL_BODY and body(reads, READS_DONE_SIG) == READS_DONE_BODY
            and body(reads, READS_RESET_SIG) == READS_RESET_BODY
            and reads.count("read_gate[") == 4 and h.count(READS_MEMBER) == 1
            and n.count("region_reads_.refill(") == 1 and n.count("region_reads_.reset();") == 1
            and n.count("region_reads_.read_gate[") == 2 and n.count("region_reads_.aos_only[") == 1
            and n.count(RESET_REGIONS_SIG + " {") == 1 and ordered(reset, REGION_STATE_RESET, REGION_FLAGS_RESET)
            and RING_READY_RESET in body(n, RING_READY_SIG) and TEARDOWN_RESET in body(n, TEARDOWN_SIG)
            and n.count("reset_regions();") == 2)


def test_aos_only_reuse_waits_for_readers():
    assert claim_aos_only_reuse_waits_for_readers(PHASE)


def test_aos_only_reuse_mutants():
    # The gate taken but not waited on.
    assert not claim_aos_only_reuse_waits_for_readers(mutate(PHASE, REUSE_DEP, ""))
    # The fallback barrier on the copy queue, which the readers do not run on.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, ": ccs_->ext_oneapi_submit_barrier();", ": cache.get_bcs_queue().ext_oneapi_submit_barrier();"))
    # The read gate never used: every reuse waits for a stage-time barrier, which also waits for the work queued after
    # the last use's down op.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, "region_reads_.read_gate[cur_region_] ? *region_reads_.read_gate[cur_region_] : ccs_", "ccs_"))
    # The other region's read gate.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, "region_reads_.read_gate[cur_region_] ? *region_reads_.read_gate[cur_region_] :",
               "region_reads_.read_gate[1 - cur_region_] ? *region_reads_.read_gate[1 - cur_region_] :"))
    # A SOA-only stage after an AOS-only use, unordered after that use's readers.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, "region_reads_.aos_only[cur_region_] || std::any_of(", "std::any_of("))
    # An AOS-only stage after a SOA-only use: its single slots land on the use's SOA slots.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, " || std::any_of(items.begin(), items.end(), aos_only_item);", ";"))
    # The last use recorded when nothing was copied in.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, REUSE_RECORD, REUSE_RECORD.replace("if (copied_in) { ", "{ ")))
    # Never recorded.
    assert not claim_aos_only_reuse_waits_for_readers(mutate(PHASE, REUSE_RECORD, ""))
    # A refilled region keeps the read gate of its previous layer, which precedes the new layer's readers.
    assert not claim_aos_only_reuse_waits_for_readers(PHASE, mutate(PLAN_HPP, " read_gate[region].reset(); }", " }"))
    # The region's layer never recorded, so layer_done finds no region to gate.
    assert not claim_aos_only_reuse_waits_for_readers(PHASE, mutate(PLAN_HPP, "{ layer[region] = l; ", "{ "))
    # The refill drops the AOS-only flag, so a SOA-only stage after an AOS-only use skips the read gate.
    assert not claim_aos_only_reuse_waits_for_readers(
        PHASE, mutate(PLAN_HPP, " aos_only[region] = left_aos_only;", ""))
    # The stage refills with the wrong layer.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, "region_reads_.refill(cur_region_, items.front().layer,",
               "region_reads_.refill(cur_region_, items.front().layer + 1,"))
    # layer_done gates the region holding the next layer, the gate's old defect.
    assert not claim_aos_only_reuse_waits_for_readers(
        PHASE, mutate(PLAN_HPP, "if (layer[r] == l) {", "if (layer[r] == l + 1) {"))
    # A second copy of the read gates in the phase beside the struct's.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, READ_GATE_RECORD, READ_GATE_RECORD + " region_reads_.read_gate[0] = down_done;"))
    # layer_done records no read gate.
    assert not claim_aos_only_reuse_waits_for_readers(mutate(PHASE, READ_GATE_RECORD, ""))
    # A down op of a layer this micro-batch has not begun records a gate.
    assert not claim_aos_only_reuse_waits_for_readers(
        mutate(PHASE, "if (layer != cur_layer_) { return; } " + READ_GATE_RECORD,
               READ_GATE_RECORD + " if (layer != cur_layer_) { return; }"))
    # A new ring inherits the old ring's read gates.
    assert not claim_aos_only_reuse_waits_for_readers(PHASE, mutate(PLAN_HPP, " read_gate[r].reset(); } }", " } }"))
    # A new ring, or a torn-down phase, keeps the layers its regions last held, so layer_done gates a region by a
    # layer it no longer holds.
    assert not claim_aos_only_reuse_waits_for_readers(PHASE, mutate(PLAN_HPP, "{ layer[r] = -1; ", "{ "))
    # ... or the AOS-only flags of the old ring's last uses.
    assert not claim_aos_only_reuse_waits_for_readers(PHASE, mutate(PLAN_HPP, " aos_only[r] = false;", ""))
    # reset_regions no longer resets the regions' last uses.
    assert not claim_aos_only_reuse_waits_for_readers(mutate(PHASE, REGION_FLAGS_RESET, ""))
    # A new ring never forgets the old ring's regions.
    assert not claim_aos_only_reuse_waits_for_readers(in_body(PHASE, RING_READY_SIG, "reset_regions(); ", ""))
    # Teardown leaves the regions' layers and read gates behind for the next ring of a reused phase.
    assert not claim_aos_only_reuse_waits_for_readers(in_body(PHASE, TEARDOWN_SIG, " reset_regions();", ""))


PROMPT_KERNEL_SIG = ("static bool ggml_sycl_moe_stream_prompt_kernel(void * graph_ctx, ggml_type type, "
                     "ggml_layout_mode layout, int64_t ne0, int64_t ne1)")
PROMPT_KERNEL_QUERY = ("ggml_sycl_moe_query_route_capability(type, layout, moe_route_phase::PROMPT, ne0, ne1, "
                       "static_cast<size_t>(call->op.n_tokens), call->device, moe_layer_route_residency::DEVICE, "
                       "call->device);")
PROMPT_KERNEL_FILL = "call.op.prompt_kernel = ggml_sycl_moe_stream_prompt_kernel;"
PROMPT_KERNEL_RETURN = ("return ggml_sycl_moe_staged_capability_refusal(cap, type) == "
                        "ggml_sycl::moe_staged_route_refusal::NONE;")
STAGING_SIG = "static inline moe_stream_staging moe_stream_item_staging("
INDEX_CALL = ("moe_stream_plan_ring_geometry(plan, device, op.prompt_kernel, op.graph_ctx, op.library_claims, "
              "&index_);")
INDEX_BUILD_CALL = "moe_stream_plan_index_build(plan, device, prompt_kernel, graph_ctx, library_claims)"
PREPARE_SIG = "void moe_stream_phase::prepare("


def claim_staging_from_the_capability_query(sycl: str, hpp: str, phase: str) -> bool:
    """c-hmey item 1, one fact one source: whether the device reads a (type, layout) in a prompt comes from
    ggml_sycl_moe_query_route_capability, the query the staged route's recipe is built from, asked for the call's own
    device as both the expert's and the submitting device; the backend's predicate consults no table of its own. The
    staging asks it for SOA only after the reorder test and for AOS only after the AOS fit, and the phase asks it only
    through the index build, once per micro-batch, never per stage."""
    n = norm(sycl)
    # The forward declaration ends in ";" and the definition in " {": the body is the definition's.
    at = n.find(PROMPT_KERNEL_SIG + " {")
    pk = body(n[at:], PROMPT_KERNEL_SIG) if at >= 0 else ""
    st = body(norm(hpp), STAGING_SIG)
    p = norm(phase)
    return (n.count(PROMPT_KERNEL_SIG + " {") == 1 and PROMPT_KERNEL_QUERY in pk and PROMPT_KERNEL_RETURN in pk
            and "moe_mmvq_" not in pk and "soa_reorder" not in pk
            and n.count(PROMPT_KERNEL_FILL) == 1
            and ordered(st, "if (prompt_kernel == nullptr) { return moe_stream_staging::NONE; }",
                        "moe_stream_item_reorderable(item, slot_bytes) && prompt_kernel(graph_ctx, item.type, "
                        "GGML_LAYOUT_SOA,", "return moe_stream_staging::SOA;",
                        "moe_stream_item_aos_fits(item, slot_bytes) && prompt_kernel(graph_ctx, item.type, "
                        "GGML_LAYOUT_AOS,", "return moe_stream_staging::AOS_ONLY;")
            and INDEX_CALL in body(p, PREPARE_SIG) and p.count("prompt_kernel(") == 0
            and p.count("moe_stream_plan_ring_geometry(") == 1 and p.count("moe_stream_plan_index_build(") == 0
            and INDEX_BUILD_CALL in body(norm(hpp), RING_GEOMETRY_SIG))


def test_staging_from_the_capability_query():
    assert claim_staging_from_the_capability_query(SYCL, PLAN_HPP, PHASE)


def test_staging_source_mutants():
    # A second table answering what the device reads.
    assert not claim_staging_from_the_capability_query(
        mutate(SYCL, PROMPT_KERNEL_QUERY,
               PROMPT_KERNEL_QUERY + " (void) moe_mmvq_capability_supports_layout(type, layout);"),
        PLAN_HPP, PHASE)
    # Asked for the decode phase.
    assert not claim_staging_from_the_capability_query(
        mutate(SYCL, "type, layout, moe_route_phase::PROMPT, ne0, ne1,",
               "type, layout, moe_route_phase::DECODE, ne0, ne1,"),
        PLAN_HPP, PHASE)
    # The backend leaves the predicate unset: nothing would ever stage.
    assert not claim_staging_from_the_capability_query(mutate(SYCL, PROMPT_KERNEL_FILL, ""), PLAN_HPP, PHASE)
    # The SOA question asked for AOS.
    assert not claim_staging_from_the_capability_query(
        SYCL, mutate(PLAN_HPP, "prompt_kernel(graph_ctx, item.type, GGML_LAYOUT_SOA,",
                     "prompt_kernel(graph_ctx, item.type, GGML_LAYOUT_AOS,"), PHASE)
    # The index built blind.
    assert not claim_staging_from_the_capability_query(
        SYCL, PLAN_HPP,
        mutate(PHASE, INDEX_CALL,
               "moe_stream_plan_ring_geometry(plan, device, nullptr, nullptr, op.library_claims, &index_);"))
    # The shared geometry helper builds the index blind or without the claims hook, whatever prepare() passes.
    assert not claim_staging_from_the_capability_query(
        SYCL, mutate(PLAN_HPP, INDEX_BUILD_CALL, "moe_stream_plan_index_build(plan, device, nullptr, nullptr, "
                                                 "library_claims)"), PHASE)
    assert not claim_staging_from_the_capability_query(
        SYCL, mutate(PLAN_HPP, INDEX_BUILD_CALL, "moe_stream_plan_index_build(plan, device, prompt_kernel, "
                                                 "graph_ctx, nullptr)"), PHASE)
    # The predicate asked again at every stage.
    asked_again = AOS_ONLY_FLAG + " (void) op.prompt_kernel(op.graph_ctx, item.type, GGML_LAYOUT_AOS, item.ncols, " \
                                  "item.nrows);"
    assert not claim_staging_from_the_capability_query(SYCL, PLAN_HPP, mutate(PHASE, AOS_ONLY_FLAG, asked_again))


BATCH_HPP_REL = "ggml/src/ggml-sycl/moe-resolved-batch.hpp"
BATCH_HPP = read(BATCH_HPP_REL)
BUILD_SIG = "moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch(const ggml_tensor * src0,"
STAGED_CAP_SIG = ("static ggml_sycl::moe_staged_route_refusal ggml_sycl_moe_staged_capability_refusal(const "
                  "moe_route_capability & cap, ggml_type type)")
STAGED_CAP_STEPS = (
    "if (!cap.supported) { return ggml_sycl::moe_staged_route_refusal::UNSUPPORTED; }",
    "if (cap.kernel != moe_route_kernel::MMVQ_COMPAT) { return ggml_sycl::moe_staged_route_refusal::KERNEL; }",
    "if (ggml_sycl::moe_type_needs_direct_kernel(type)) { return ggml_sycl::moe_staged_route_refusal::DIRECT_KERNEL; }",
    "return ggml_sycl::moe_staged_route_refusal::NONE;",
)
ROUTE_CAP_ASK = "ggml_sycl_moe_staged_capability_refusal(cap, src0->type);"
DIRECT_SIG = "inline bool moe_type_needs_direct_kernel(ggml_type type)"
DIRECT_BODY = "{ return type == GGML_TYPE_Q1_0 || type == GGML_TYPE_NVFP4; }"
VALIDATE_SIG = "inline bool validate_moe_execution_recipe("
CHOOSE_SIG = "inline moe_batch_executor_choice choose_moe_batch_executor("
DIRECT_USES = (
    "if (recipe.kind == moe_batch_executor::HOST_CPU && moe_type_needs_direct_kernel(recipe.request.type)) {",
    "const bool direct_q1_nvfp4 = moe_type_needs_direct_kernel(recipe.request.type);",
    "operand.recipe().kind != moe_batch_executor::HOST_CPU && "
    "moe_type_needs_direct_kernel(operand.recipe().request.type);",
)


def claim_staging_mirrors_serving(sycl: str, batch_hpp: str) -> bool:
    """llama.cpp-o9uw: the ring stages only what the staged route can serve. The prompt-kernel predicate and the
    resolver ask one helper, which admits a supported MMVQ_COMPAT kernel and nothing else, and refuses a type that
    runs only through its direct kernel. Which types those are is one predicate, the one the recipe validator and the
    executor choice ask, and neither of those spells the types out, so the staging predicate cannot drift from the
    served test: Q1_0 and NVFP4, which the capability query answers MMVQ_COMPAT for in a prompt, stage NONE. The
    predicate's only answer other than the helper's is false for a missing call."""
    n = norm(sycl)
    b = norm(batch_hpp)
    at = n.find(PROMPT_KERNEL_SIG + " {")
    pk = body(n[at:], PROMPT_KERNEL_SIG) if at >= 0 else ""
    cap = body(n, STAGED_CAP_SIG)
    return (n.count(STAGED_CAP_SIG + " {") == 1 and ordered(cap, *STAGED_CAP_STEPS)
            and re.findall(r"moe_route_kernel::(\w+)", cap) == ["MMVQ_COMPAT"]
            and pk.endswith(PROMPT_KERNEL_RETURN + " }") and "return true" not in pk
            and ROUTE_CAP_ASK in body(n, BUILD_SIG)
            and n.count("ggml_sycl_moe_staged_capability_refusal(") == 3
            and b.count(DIRECT_SIG + " " + DIRECT_BODY) == 1 and all(b.count(use) == 1 for use in DIRECT_USES)
            and not any(t in body(b, sig) for t in ("GGML_TYPE_Q1_0", "GGML_TYPE_NVFP4")
                        for sig in (VALIDATE_SIG, CHOOSE_SIG)))


def test_staging_mirrors_serving():
    assert claim_staging_mirrors_serving(SYCL, BATCH_HPP)


def test_staging_mirrors_serving_mutants():
    c = claim_staging_mirrors_serving
    # The predicate answering from the capability alone again, so a Q1_0/NVFP4 matrix stages and is never served.
    assert not c(mutate(SYCL, PROMPT_KERNEL_RETURN, "return cap.supported;"), BATCH_HPP)
    # The helper admitting a direct-kernel type.
    assert not c(mutate(SYCL, STAGED_CAP_STEPS[2], ""), BATCH_HPP)
    # The helper admitting another kernel.
    assert not c(mutate(SYCL, "if (cap.kernel != moe_route_kernel::MMVQ_COMPAT) {",
                        "if (cap.kernel != moe_route_kernel::MMVQ_COMPAT && "
                        "cap.kernel != moe_route_kernel::XMX_TILED) {"), BATCH_HPP)
    # The resolver asking a test of its own.
    assert not c(mutate(SYCL, ROUTE_CAP_ASK, "moe_staged_route_refusal::NONE;"), BATCH_HPP)
    # The validator spelling the types out again, so the two answers can drift.
    assert not c(SYCL, mutate(BATCH_HPP, DIRECT_USES[1],
                              "const bool direct_q1_nvfp4 = recipe.request.type == GGML_TYPE_Q1_0 || "
                              "recipe.request.type == GGML_TYPE_NVFP4;"))
    # The executor choice spelling the types out again.
    assert not c(SYCL, mutate(BATCH_HPP, DIRECT_USES[2],
                              "operand.recipe().kind != moe_batch_executor::HOST_CPU && "
                              "operand.recipe().request.type == GGML_TYPE_NVFP4;"))
    # The predicate answering yes before it asks the helper.
    head = PROMPT_KERNEL_SIG + " { const auto * call = static_cast<const ggml_sycl_moe_stream_call *>(graph_ctx); "
    assert not c(mutate(SYCL, head + "if (call == nullptr) { return false; }",
                        head + "if (call != nullptr) { return true; }"), BATCH_HPP)
    # The predicate losing one of its types.
    assert not c(SYCL, mutate(BATCH_HPP, DIRECT_BODY, "{ return type == GGML_TYPE_Q1_0; }"))


LIBRARY_CLAIMS_SIG = ("static bool ggml_sycl_moe_stream_library_claims(void * graph_ctx, int layer, "
                      "ggml_sycl::expert_tensor_role role)")
LIBRARY_CLAIMS_WEIGHT = "const ggml_tensor * w = ggml_sycl_moe_stream_layer_weight(layer, role);"
LIBRARY_CLAIMS_QUERY = "ggml_sycl_moe_pp_onednn_batched_claims_tensor(call->plan.get(), w, call->device);"
LIBRARY_CLAIMS_GUARD = "return call != nullptr && call->plan && w != nullptr &&"
CLAIMS_PLAN_SIG = ("static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_sycl::placement_plan * plan, "
                   "const ggml_tensor * weight, int device)")
CLAIMS_PLAN_LAYOUT = ("const bool planned = plan != nullptr ? "
                      "ggml_sycl_moe_tensor_plan_primary_layout(*plan, weight, device, &planned_layout) : "
                      "ggml_sycl_moe_tensor_plan_primary_layout(weight, device, &planned_layout);")
CLAIMS_ENTRY = ("static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_tensor * weight, int device) { "
                "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight, device); }")
PRIMARY_PLAN_SIG = ("static bool ggml_sycl_moe_tensor_plan_primary_layout(const ggml_sycl::placement_plan & plan, "
                    "const ggml_tensor * src0, int device, layout_mode * out_layout)")
PRIMARY_ENTRY_TAIL = "return ggml_sycl_moe_tensor_plan_primary_layout(*plan_owner, src0, device, out_layout); }"
PLAN_READERS = ("ggml_sycl_cache_plan_owner(", "coherent_placement_plan_owner(", "coherent_cache_placement_plan_owner(",
                "global_placement_plan_owner(", "ggml_sycl_global_plan_owner(", "get_unified_cache_for_device(")
LIBRARY_CLAIMS_FILL = "call.op.library_claims = ggml_sycl_moe_stream_library_claims;"
ROUTE_LIBRARY_CLAIM = "stream_library_claimed = ggml_sycl_moe_pp_onednn_batched_claims_tensor(src0, submit_device);"
LAYER_WEIGHT_SIG = ("static const ggml_tensor * ggml_sycl_moe_stream_layer_weight(int layer, "
                    "ggml_sycl::expert_tensor_role role)")
INDEX_STAGING_ASK = "item.staging = moe_stream_item_staging(item, SIZE_MAX, prompt_kernel, graph_ctx);"
INDEX_CLAIM_GUARD = "if (item.staging != moe_stream_staging::NONE && library_claims != nullptr) {"
INDEX_CLAIM_ASK = "c = claims.emplace(ck, library_claims(graph_ctx, item.layer, item.role)).first;"
INDEX_CLAIM_APPLY = "if (c->second) { item.staging = moe_stream_staging::NONE; }"
INDEX_SLOT_SKIP = "if (item.staging == moe_stream_staging::NONE) { continue; }"


def claim_staging_declines_library_claimed(sycl: str, hpp: str, phase: str) -> bool:
    """One fact, one source (llama.cpp-o9uw): the staged route refuses a weight the oneDNN batched prompt executor
    claims (c-las5), so the index build stages none of it. The backend answers the build's library-claims predicate
    with that same claim, for the call's device and the graph's own weight tensor of (layer, role); the build asks it
    after the kernels' answer and before the slot accounting, so a claimed matrix stays listed (a stage counts it
    untyped) and takes no slot; and the phase asks it only through the index build."""
    n = norm(sycl)
    at = n.find(LIBRARY_CLAIMS_SIG + " {")
    lc = body(n[at:], LIBRARY_CLAIMS_SIG) if at >= 0 else ""
    build = body(norm(hpp), INDEX_BUILD_SIG)
    p = norm(phase)
    return (n.count(LIBRARY_CLAIMS_SIG + " {") == 1 and LIBRARY_CLAIMS_WEIGHT in lc and LIBRARY_CLAIMS_QUERY in lc
            and "MXFP4" not in lc and "moe_classify_tensor" not in lc
            and "g_moe_gate_up_pairs.find(layer)" in body(n, LAYER_WEIGHT_SIG)
            and n.count(LIBRARY_CLAIMS_FILL) == 1 and n.count(ROUTE_LIBRARY_CLAIM) == 1
            and "moe_stream_library_claims_fn library_claims)" in norm(hpp)
            and ordered(build, INDEX_STAGING_ASK, INDEX_CLAIM_GUARD, INDEX_CLAIM_ASK, INDEX_CLAIM_APPLY,
                        "li.items.push_back(item);", INDEX_SLOT_SKIP, "++li.n_streamable;", REGION_SIZE)
            and norm(hpp).count("library_claims(graph_ctx") == 1
            and INDEX_CALL in body(p, PREPARE_SIG) and p.count("library_claims(") == 0)


def test_staging_declines_library_claimed():
    assert claim_staging_declines_library_claimed(SYCL, PLAN_HPP, PHASE)


def test_library_claim_mutants():
    # The backend answers from something other than the claim the route refuses on.
    assert not claim_staging_declines_library_claimed(
        mutate(SYCL, LIBRARY_CLAIMS_QUERY, "ggml_sycl_moe_pp_woq_enabled();"), PLAN_HPP, PHASE)
    # Asked for another device than the call's.
    assert not claim_staging_declines_library_claimed(
        mutate(SYCL, LIBRARY_CLAIMS_QUERY, "ggml_sycl_moe_pp_onednn_batched_claims_tensor(call->plan.get(), w, 0);"),
        PLAN_HPP, PHASE)
    # A second table of its own beside the claim. The host lease takes its weight with the same line, so the mutant is
    # made inside the definition.
    n = norm(SYCL)
    at = n.find(LIBRARY_CLAIMS_SIG + " {")
    assert at >= 0 and LIBRARY_CLAIMS_WEIGHT in n[at:], "mutant precondition: the claims definition takes its weight"
    second = n[:at] + n[at:].replace(
        LIBRARY_CLAIMS_WEIGHT, LIBRARY_CLAIMS_WEIGHT + " if (w && w->type != GGML_TYPE_MXFP4) { return false; }", 1)
    assert not claim_staging_declines_library_claimed(second, PLAN_HPP, PHASE)
    # The backend leaves the predicate unset: a claimed weight would be staged and then always refused.
    assert not claim_staging_declines_library_claimed(mutate(SYCL, LIBRARY_CLAIMS_FILL, ""), PLAN_HPP, PHASE)
    # The route refuses on something else, so the two would answer differently.
    assert not claim_staging_declines_library_claimed(
        mutate(SYCL, ROUTE_LIBRARY_CLAIM, "stream_library_claimed = false;"), PLAN_HPP, PHASE)
    # The build asks and ignores the answer.
    assert not claim_staging_declines_library_claimed(
        SYCL, mutate(PLAN_HPP, INDEX_CLAIM_APPLY, "if (c->second) { }"), PHASE)
    # A claimed matrix still takes a slot: the claim applied after the slot accounting.
    late = mutate(PLAN_HPP, INDEX_CLAIM_APPLY, "")
    late = mutate(late, "++li.n_streamable;", "++li.n_streamable; " + INDEX_CLAIM_APPLY)
    assert not claim_staging_declines_library_claimed(SYCL, late, PHASE)
    # The phase builds its index without the claims.
    assert not claim_staging_declines_library_claimed(
        SYCL, PLAN_HPP, mutate(PHASE, INDEX_CALL, INDEX_CALL.replace("op.library_claims", "nullptr")))
    # The phase asks again at every stage.
    asked_again = AOS_ONLY_FLAG + " (void) op.library_claims(op.graph_ctx, item.layer, item.role);"
    assert not claim_staging_declines_library_claimed(SYCL, PLAN_HPP, mutate(PHASE, AOS_ONLY_FLAG, asked_again))


def claim_library_claims_read_the_index_plan(sycl: str) -> bool:
    """Lead ruling c-gi44, one fact one source: the library-claim predicate answers from call.plan, the plan the index
    build that asks it holds, through the plan-taking form of the batched executor's claim, and asks only when the
    call holds a plan. That form takes the planned layout from the plan it is given, and the plan-taking walk behind
    it reads no plan of its own. Only the claim's two-argument entry point, the batched executor's own route, reads
    the device's plan: it passes no plan, and the walk's two-argument form reads the cache plan and delegates."""
    n = norm(sycl)
    at = n.find(LIBRARY_CLAIMS_SIG + " {")
    lc = body(n[at:], LIBRARY_CLAIMS_SIG) if at >= 0 else ""
    at = n.find(CLAIMS_PLAN_SIG + " {")
    cl = body(n[at:], CLAIMS_PLAN_SIG) if at >= 0 else ""
    at = n.find(PRIMARY_PLAN_SIG + " {")
    pl = body(n[at:], PRIMARY_PLAN_SIG) if at >= 0 else ""
    return (ordered(lc, LIBRARY_CLAIMS_GUARD, LIBRARY_CLAIMS_QUERY)
            and lc.count("ggml_sycl_moe_pp_onednn_batched_claims_tensor(") == 1
            and n.count(CLAIMS_PLAN_SIG + " {") == 1 and cl.count(CLAIMS_PLAN_LAYOUT) == 1
            and cl.count("ggml_sycl_moe_tensor_plan_primary_layout(") == 2
            and n.count(CLAIMS_ENTRY) == 1
            and n.count(PRIMARY_PLAN_SIG + " {") == 1 and "plan.all_experts_on_device(" in pl
            and not any(r in pl for r in PLAN_READERS) and n.count(PRIMARY_ENTRY_TAIL) == 1)


def test_library_claims_read_the_index_plan():
    assert claim_library_claims_read_the_index_plan(SYCL)


def test_library_claims_index_plan_mutants():
    c = claim_library_claims_read_the_index_plan
    # The predicate reads the device's plan again through the two-argument entry point.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_QUERY, "ggml_sycl_moe_pp_onednn_batched_claims_tensor(w, call->device);"))
    # The predicate passes no plan, so the claim reads the device's plan instead of the index build's.
    assert not c(mutate(SYCL, "ggml_sycl_moe_pp_onednn_batched_claims_tensor(call->plan.get(), w,",
                        "ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, w,"))
    # Asked without a plan in hand.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_GUARD, "return call != nullptr && w != nullptr &&"))
    # The claim ignores the plan it is given.
    assert not c(mutate(SYCL, CLAIMS_PLAN_LAYOUT,
                        "const bool planned = "
                        "ggml_sycl_moe_tensor_plan_primary_layout(weight, device, &planned_layout);"))
    # The plan-taking walk reads the device's plan behind its caller's back.
    assert not c(mutate(SYCL, "if (!plan.all_experts_on_device(src0->name, n_experts, device)) {",
                        "(void) ggml_sycl_cache_plan_owner(ggml_sycl::get_unified_cache_for_device(device)); "
                        "if (!plan.all_experts_on_device(src0->name, n_experts, device)) {"))
    # The executor's own route handed a plan read ahead of its type and role checks.
    assert not c(mutate(SYCL, "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight, device); }",
                        "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(ggml_sycl_cache_plan_owner("
                        "ggml_sycl::get_unified_cache_for_device(device)).get(), weight, device); }"))


# ---- the policy gate -------------------------------------------------------------------------------------------------

RESOLVE_SIG = "void moe_stream_phase::resolve("
STAGE_SIG = "moe_stream_phase::stage_counts moe_stream_phase::stage("
RING_READY_SIG = "bool moe_stream_phase::ring_ready()"
POLICY_GUARD = "if (moe_stream_stage_now(st.decision, op.policy)) {"
STAGE_PRED_SIG = "static inline bool moe_stream_stage_now(const moe_exec_decision & d, moe_exec_policy policy)"
STAGE_PRED_BODY = "{ return d.choice == moe_exec_choice::RING && policy == moe_exec_policy::RING; }"
STAGE_CALL = "n = stage(li, st.decision.regime, cache, op);"
POLICY_FROM_CONFIG = "call.op.policy = ggml_sycl_moe_exec_config().policy;"
STAGE_CALL_RE = re.compile(r"(?<![:\w])stage\(")
RING_READY_CALL_RE = re.compile(r"(?<![:\w])ring_ready\(\)")


def claim_staging_only_under_explicit_ring(phase: str, sycl: str, hpp: str) -> bool:
    """Lead ruling c-3skf: only an explicit GGML_SYCL_MOE_EXEC=ring copies, reorders or reserves the ring, and its
    staged experts are what the resolver's STAGED_RING route consumes (llama.cpp-o9uw); AUTO only records the cost
    model's choice. The phase's one stage() call is the first statement under a guard, moe_stream_stage_now, that needs
    both the RING choice and the RING policy, the reserve is reachable only through stage() and ring_ready(), and the
    op's policy is the parsed config's."""
    n = norm(phase)
    return (POLICY_GUARD + " " + STAGE_CALL in body(n, RESOLVE_SIG)
            and body(norm(hpp), STAGE_PRED_SIG) == STAGE_PRED_BODY
            and len(STAGE_CALL_RE.findall(n)) == 1
            and len(RING_READY_CALL_RE.findall(n)) == 1
            and len(RING_READY_CALL_RE.findall(body(n, STAGE_SIG))) == 1
            and n.count("->reserve(") == 1 and "->reserve(" in body(n, RING_READY_SIG)
            and norm(sycl).count(POLICY_FROM_CONFIG) == 1)


def test_staging_only_under_explicit_ring():
    assert claim_staging_only_under_explicit_ring(PHASE, SYCL, PLAN_HPP)


def test_policy_gate_mutants():
    # AUTO reaching the stage call: the predicate tests the choice alone.
    assert not claim_staging_only_under_explicit_ring(
        PHASE, SYCL, mutate(PLAN_HPP, STAGE_PRED_BODY, "{ return d.choice == moe_exec_choice::RING; }"))
    # A predicate that still admits AUTO.
    assert not claim_staging_only_under_explicit_ring(
        PHASE, SYCL, mutate(PLAN_HPP, STAGE_PRED_BODY,
                            "{ return d.choice == moe_exec_choice::RING && policy != moe_exec_policy::CPU; }"))
    # The guard testing the choice alone, past the predicate.
    assert not claim_staging_only_under_explicit_ring(
        mutate(PHASE, POLICY_GUARD, "if (st.decision.choice == moe_exec_choice::RING) {"), SYCL, PLAN_HPP)
    # A second stage call outside the guard.
    assert not claim_staging_only_under_explicit_ring(
        mutate(PHASE, "st.resolved = true;", STAGE_CALL + " st.resolved = true;"), SYCL, PLAN_HPP)
    # The ring reserved outside stage().
    assert not claim_staging_only_under_explicit_ring(
        mutate(PHASE, "in_order_ = ccs.is_in_order();", "in_order_ = ccs.is_in_order(); ring_ready();"), SYCL, PLAN_HPP)
    # The op's policy pinned to RING instead of read from GGML_SYCL_MOE_EXEC.
    assert not claim_staging_only_under_explicit_ring(
        PHASE, mutate(SYCL, POLICY_FROM_CONFIG, "call.op.policy = moe_exec_policy::RING;"), PLAN_HPP)


UBATCH_PRED_SIG = "static inline bool moe_stream_ubatch_starts("
AHEAD_PRED_SIG = "static inline bool moe_stream_resolve_ahead(const moe_exec_decision & d)"
UBATCH_PRED_CALL = ("const bool new_ubatch = moe_stream_ubatch_starts(ubatch_live_, layer, cur_layer_, op.graph_seq, "
                    "cur_graph_seq_);")
AHEAD_PRED_CALL = "if (moe_stream_resolve_ahead(ns.decision)) { resolve(next, nxt->second, ns, cache, op); }"


def claim_phase_decides_through_tested_predicates(phase: str, hpp: str) -> bool:
    """begin_layer and resolve decide through the header's pure predicates, which test-moe-stream-phase pins: each is
    defined once in moe-stream-plan.hpp and called, never restated, in the phase. A micro-batch starts per
    moe_stream_ubatch_starts, the next layer resolves ahead per moe_stream_resolve_ahead, and bytes move per
    moe_stream_stage_now."""
    h = norm(hpp)
    n = norm(phase)
    b = body(n, BEGIN_LAYER_SIG)
    return (all(h.count(sig) == 1 for sig in (UBATCH_PRED_SIG, STAGE_PRED_SIG, AHEAD_PRED_SIG))
            and UBATCH_PRED_CALL in b and AHEAD_PRED_CALL in b and POLICY_GUARD in body(n, RESOLVE_SIG)
            and "!= cur_graph_seq_" not in n and "moe_exec_regime::ROUTED_ONLY" not in b)


def test_phase_decides_through_tested_predicates():
    assert claim_phase_decides_through_tested_predicates(PHASE, PLAN_HPP)


def test_predicate_mutants():
    # The micro-batch start restated inline.
    assert not claim_phase_decides_through_tested_predicates(
        mutate(PHASE, UBATCH_PRED_CALL, "const bool new_ubatch = !ubatch_live_ || layer < cur_layer_ || "
                                        "(layer == cur_layer_ && op.graph_seq != cur_graph_seq_);"), PLAN_HPP)
    # The routed-ring test restated inline.
    assert not claim_phase_decides_through_tested_predicates(
        mutate(PHASE, "if (moe_stream_resolve_ahead(ns.decision)) {",
               "if (!(ns.decision.choice == moe_exec_choice::RING && "
               "ns.decision.regime == moe_exec_regime::ROUTED_ONLY)) {"), PLAN_HPP)
    # The stage-now test restated inline.
    assert not claim_phase_decides_through_tested_predicates(
        mutate(PHASE, POLICY_GUARD,
               "if (st.decision.choice == moe_exec_choice::RING && op.policy == moe_exec_policy::RING) {"), PLAN_HPP)
    # A predicate the test pins is gone from the header.
    assert not claim_phase_decides_through_tested_predicates(
        PHASE, mutate(PLAN_HPP, AHEAD_PRED_SIG,
                      "static inline bool moe_stream_resolve_later(const moe_exec_decision & d)"))


BEGIN_UBATCH_SIG = "void moe_stream_phase::begin_ubatch(uint32_t n_tokens)"
BEGIN_LAYER_SIG = "moe_exec_decision moe_stream_phase::begin_layer("
END_PROMPT_SIG = "void moe_stream_phase::end_prompt()"
LATCH_CLEAR = "reserve_failed_ = false;"
LATCH_SET = "reserve_failed_ = true;"
LATCH_GUARD = "if (reserve_failed_ || geo_.n_slots == 0 || !in_order_) { return false; }"
NEW_UBATCH = "if (new_ubatch) { begin_ubatch(op.n_tokens);"


def claim_reserve_failure_lasts_one_ubatch(phase: str) -> bool:
    """Lead ruling c-i8rz: a refused reserve is not retried within a micro-batch and is tried once more in the next
    micro-batch of the same prompt (a split micro-batch spans several graphs, so the micro-batch is the unit). The latch
    is set only where the reserve fails, stops ring_ready() before any reserve, and is cleared by begin_ubatch, which
    every new micro-batch runs, and by end_prompt."""
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


def test_claims_count_production_sites_only():
    """The phase's PRIVATE_TESTING seams reserve a ring and acquire a slot so a device case can drive the staged-ring
    route (llama.cpp-o9uw). The claims count the production sites, so production() leaves the seams out; were a seam
    compiled into the shipped library, the claims would count it and fail."""
    raw = read(PHASE_REL)
    guard = "#if defined(GGML_SYCL_PRIVATE_TESTING)"
    assert raw.count(guard) == 1
    assert "moe_stream_phase::test_reserve_ring(" in raw and "moe_stream_phase::test_reserve_ring(" not in PHASE
    assert "moe_stream_phase::test_acquire_slot(" in raw and "moe_stream_phase::test_acquire_slot(" not in PHASE
    assert claim_phase_allocates_only_ring_slots(PHASE)
    assert claim_staging_only_under_explicit_ring(PHASE, SYCL, PLAN_HPP)
    # The seams outside their block: a second slot acquisition and a second ring_ready() call, both counted.
    unguarded = production(raw.replace(guard, "#if 1"))
    assert not claim_phase_allocates_only_ring_slots(unguarded)
    assert not claim_staging_only_under_explicit_ring(unguarded, SYCL, PLAN_HPP)
    # An #else arm is production and stays; a nested conditional inside the block goes with it.
    assert production("a\n#if defined(GGML_SYCL_PRIVATE_TESTING)\nb\n#else\nc\n#endif\nd") == "a\n\n\n\nc\n\nd"
    assert production("#ifdef GGML_SYCL_PRIVATE_TESTING\n#ifdef X\nb\n#else\nc\n#endif\n#endif\nd") == "\n" * 7 + "d"


def test_reserve_failure_lasts_one_ubatch():
    assert claim_reserve_failure_lasts_one_ubatch(PHASE)


def test_reserve_latch_mutants():
    # The per-micro-batch clear removed: a refused reserve would last the whole prompt.
    assert not claim_reserve_failure_lasts_one_ubatch(drop_in(PHASE, BEGIN_UBATCH_SIG, LATCH_CLEAR))
    # The prompt-end clear removed.
    assert not claim_reserve_failure_lasts_one_ubatch(drop_in(PHASE, END_PROMPT_SIG, LATCH_CLEAR))
    # No latch check: every staging layer would retry the reserve.
    assert not claim_reserve_failure_lasts_one_ubatch(
        mutate(PHASE, "if (reserve_failed_ || geo_.n_slots", "if (geo_.n_slots"))
    # The latch never set.
    assert not claim_reserve_failure_lasts_one_ubatch(mutate(PHASE, LATCH_SET, ""))
    # A new micro-batch that skips begin_ubatch.
    assert not claim_reserve_failure_lasts_one_ubatch(mutate(PHASE, NEW_UBATCH, "if (new_ubatch) {"))


DIAG_FMT = ("[MOE-EXEC] device %d: stream ring unavailable: %s (host_matrices_per_layer_max=%u region_slots_max=%u "
            "matrix_bytes_max=%zu resident_expert_bytes=%zu n_slots=%u bytes=%zu)")
# One argument per field, in the format's order: the region_slots_max a region would need is the ring's other input.
DIAG_ARGS = ("index_.host_matrices_per_layer_max, index_.region_slots_max, index_.matrix_bytes_max, "
             "index_.resident_expert_bytes, asked.n_slots, asked.bytes);")
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
    return (count == 1 and ordered(w, DIAG_FMT, DIAG_RULE_CALL, DIAG_ARGS, DIAG_EMIT)
            and ordered(body(norm(hpp), RULE_SIG), *RULE_ORDER))


def test_ring_unavailable_diagnostic():
    assert claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, SYCL_TREE)


def test_ring_unavailable_diagnostic_mutants():
    renamed = mutate(PHASE, "n_slots=%u ", "slots=%u ")
    assert not claim_ring_unavailable_diagnostic(renamed, PLAN_HPP, {PHASE_REL: renamed})
    # The region's slot need left out, or printed under another field's name.
    dropped = mutate(PHASE, " region_slots_max=%u", "")
    assert not claim_ring_unavailable_diagnostic(dropped, PLAN_HPP, {PHASE_REL: dropped})
    # (anchored on the whole argument list: the ring sizing passes the same two fields to moe_stream_ring_size)
    swapped = mutate(PHASE, "index_.host_matrices_per_layer_max, index_.region_slots_max, index_.matrix_bytes_max,",
                     "index_.host_matrices_per_layer_max, index_.matrix_bytes_max, index_.region_slots_max,")
    assert not claim_ring_unavailable_diagnostic(swapped, PLAN_HPP, {PHASE_REL: swapped})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + DIAG_FMT + '"'
    assert not claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, tree)
    assert not claim_ring_unavailable_diagnostic(mutate(PHASE, DIAG_EMIT, "(void) line;"), PLAN_HPP, SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, '"reserve refused"', '"reserve failed"'), SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, "geo.n_slots == 0", "geo.bytes == 0"), SYCL_TREE)


def claim_ring_unavailable_once_per_prompt(phase: str) -> bool:
    """Lead ruling c-kcyf: at most one diagnostic per prompt, and only under an explicit ring or a trace. The guard
    comes first, the flag is set before the line goes out, end_prompt clears it and begin_ubatch does not, and resolve
    calls it for the decision the layer ends with (after a refused reserve rewrote it), before the layer's report. The
    phase has no log call of its own: a queue that is not in order is one of the diagnostic's rules, under the same
    gating (lead ruling c-xod7-q1)."""
    n = norm(phase)
    w = body(n, WARN_SIG)
    return (w.startswith("{ " + WARN_GUARD) and "GGML_LOG_" not in n
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
    # Cleared per micro-batch instead of per prompt.
    assert not claim_ring_unavailable_once_per_prompt(
        mutate(PHASE, "reserve_failed_ = false; table_.clear();",
               "reserve_failed_ = false; " + WARN_CLEAR + " table_.clear();"))
    # Never cleared: only the first prompt of the context would warn.
    assert not claim_ring_unavailable_once_per_prompt(drop_in(PHASE, END_PROMPT_SIG, WARN_CLEAR))
    # AUTO without a trace would warn too.
    assert not claim_ring_unavailable_once_per_prompt(
        mutate(PHASE, " || (op.policy != moe_exec_policy::RING && op.log->trace < 1)", ""))
    # Not called for the layer's decision.
    assert not claim_ring_unavailable_once_per_prompt(mutate(PHASE, WARN_CALL, ""))
    # A WARN of the phase's own, at default verbosity under AUTO.
    assert not claim_ring_unavailable_once_per_prompt(
        mutate(PHASE, "in_order_ = ccs.is_in_order();",
               'in_order_ = ccs.is_in_order(); GGML_LOG_WARN("queue not in order\\n");'))


REPORT_CALL = "report(layer, st, n, op);"
NEXT_LOOKUP = ("const auto nxt = next < moe_exec_decision_log::MAX_LAYERS ? index_.layers.find(next) : "
               "index_.layers.end();")


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
    # Only an explicit ring reports: AUTO's choice would never reach the decision log.
    assert not claim_auto_still_reports(
        mutate(PHASE, REPORT_CALL, "if (op.policy == moe_exec_policy::RING) { " + REPORT_CALL + " }"))
    # AUTO returns before the report.
    assert not claim_auto_still_reports(
        mutate(PHASE, "stage_counts n; if (moe_stream_stage_now(",
               "stage_counts n; if (op.policy != moe_exec_policy::RING) { return; } if (moe_stream_stage_now("))
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


# ---- the ring's lifetime ---------------------------------------------------------------------------------------------

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


def in_body(text: str, signature: str, old: str, new: str) -> str:
    """`text` (normalized) with `old` replaced once inside the body of `signature`; the mutant must exist."""
    n = norm(text)
    b = body(n, signature)
    assert old in b, f"mutant precondition: {old!r} is not in {signature!r}"
    return n.replace(b, b.replace(old, new, 1), 1)


TEARDOWN_RETAIN = "if (anchor_.valid()) { retain_handles_until_event({ anchor_ }, last_use); }"


def test_teardown_mutants():
    assert not claim_teardown_gates_before_release(mutate(PHASE, "close_region(last_use);", ""))
    assert not claim_teardown_gates_before_release(
        mutate(PHASE, "retain_handles_until_event({ anchor_ }, last_use);", ""))
    assert not claim_teardown_gates_before_release(mutate(PHASE, "in_order_ = ccs.is_in_order();", "in_order_ = true;"))
    # The ring released before the slot handle is retained on the last use.
    assert not claim_teardown_gates_before_release(
        in_body(PHASE, TEARDOWN_SIG, TEARDOWN_RETAIN + " ring_->release();", "ring_->release(); " + TEARDOWN_RETAIN))
    # The destructor gives the ring back without teardown().
    assert not claim_teardown_gates_before_release(
        in_body(PHASE, "moe_stream_phase::~moe_stream_phase()", "teardown(", "(void) ("))
    # end_prompt keeps the ring.
    assert not claim_teardown_gates_before_release(
        in_body(PHASE, END_PROMPT_SIG, "teardown(ring_ ? compute_barrier() : sycl::event());", ""))


# ---- the context's wiring --------------------------------------------------------------------------------------------

MEMBER = "std::unique_ptr<ggml_sycl::moe_stream_phase, ggml_sycl::moe_stream_phase_deleter> moe_stream;"
DTOR_SIG = "ggml_backend_sycl_context::~ggml_backend_sycl_context()"
END_PROMPT = ("sycl_ctx->graph_phase_is_decode.store(cached_is_decode, std::memory_order_relaxed); "
              "if (cached_is_decode && sycl_ctx->moe_stream) { sycl_ctx->moe_stream->end_prompt(); "
              "ggml_sycl_moe_stream_release_payload(*sycl_ctx); }")
DISPATCH = ("ggml_sycl_mul_mat_id(ctx, dst); ggml_sycl_moe_down_shadow_compare(ctx, dst); "
            "ggml_sycl_moe_stream_layer_done(ctx, dst); break;")
LAYER_DONE_SIG = "static void ggml_sycl_moe_stream_layer_done(ggml_backend_sycl_context & ctx, const ggml_tensor * dst)"


def claim_context_owns_and_ends_the_phase(common: str, sycl: str) -> bool:
    n = norm(sycl)
    dtor = body(n, DTOR_SIG)
    done = body(n, LAYER_DONE_SIG)
    return (MEMBER in norm(common)
            and ordered(dtor, "qptrs[dev][s]->wait();", "moe_stream.reset();",
                        "ggml_sycl::drain_retained_handles(true);")
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
    assert not claim_context_owns_and_ends_the_phase(
        COMMON, mutate(SYCL, " ggml_sycl_moe_stream_layer_done(ctx, dst); break;", " break;"))
    assert not claim_context_owns_and_ends_the_phase(
        COMMON, mutate(SYCL, "it->second.down_weight != dst->src[0]", "false"))
    # The phase reset before the queues drain: its teardown barrier would not follow the last ring use.
    reset = "moe_stream.reset();"
    drain_q = "qptrs[dev][s]->wait();"
    drain_h = "ggml_sycl::drain_retained_handles(true);"
    early = in_body(SYCL, DTOR_SIG, reset, "")
    early = in_body(early, DTOR_SIG, drain_q, reset + " " + drain_q)
    assert not claim_context_owns_and_ends_the_phase(COMMON, early)
    # The phase reset after the retained handles drain: the slot it retains on teardown would outlive that drain.
    late = in_body(SYCL, DTOR_SIG, reset, "")
    late = in_body(late, DTOR_SIG, drain_h, drain_h + " " + reset)
    assert not claim_context_owns_and_ends_the_phase(COMMON, late)


REORDER_DEF = "sycl::event ggml_sycl_reorder_weight_gpu(sycl::queue & queue,"
LOAD_FILL = "ggml_sycl_reorder_weight_gpu(queue, static_cast<const uint8_t *>(temp_vram),"


def claim_reorder_is_shared_not_copied(sycl: str, hpp: str) -> bool:
    """The ring submits the load path's own reorder dispatch: defined once in ggml-sycl.cpp without `static`, declared
    in moe-stream-plan.hpp, and the load-time fill still calls it."""
    n = norm(sycl)
    return (n.count(REORDER_DEF) == 1 and "static " + REORDER_DEF not in n
            and "sycl::event ggml_sycl_reorder_weight_gpu(sycl::queue & queue," in norm(hpp)
            and n.count(LOAD_FILL) == 1)


def test_reorder_is_shared_not_copied():
    assert claim_reorder_is_shared_not_copied(SYCL, PLAN_HPP)


def test_reorder_shared_mutants():
    assert not claim_reorder_is_shared_not_copied(mutate(SYCL, REORDER_DEF, "static " + REORDER_DEF), PLAN_HPP)
    assert not claim_reorder_is_shared_not_copied(
        SYCL, mutate(PLAN_HPP, REORDER_DEF, "sycl::event other_reorder(sycl::queue & queue,"))
    # The load-time fill moved off the shared dispatch.
    assert not claim_reorder_is_shared_not_copied(
        mutate(SYCL, LOAD_FILL, "other_reorder(queue, static_cast<const uint8_t *>(temp_vram),"), PLAN_HPP)


# ---- the trace -------------------------------------------------------------------------------------------------------


def claim_trace_line(phase: str, tree: dict) -> bool:
    """The plan's fixed fields, once in the backend, emitted through the decision log's trace sink under the
    "[MOE-EXEC-TRACE] stage layer" prefix. The decision log's own note() line is "[MOE-EXEC-TRACE] dev %d layer %d"
    (test-moe-exec-policy pins it), so the two trace lines share the trace prefix and differ in the word after it (plan
    amendment for Task 15, lead ruling c-xod7-q1)."""
    count = sum(norm(text).count(TRACE_FMT) for text in tree.values())
    return count == 1 and ordered(norm(phase), TRACE_FMT, TRACE_EMIT)


def test_trace_line():
    assert claim_trace_line(PHASE, SYCL_TREE)


def test_trace_mutants():
    m = mutate(PHASE, "window=%.0f", "win=%.0f")
    assert not claim_trace_line(m, {PHASE_REL: m})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + TRACE_FMT + '"'
    assert not claim_trace_line(PHASE, tree)
    assert not claim_trace_line(mutate(PHASE, TRACE_EMIT, "(void) line;"), {PHASE_REL: PHASE})
    # The old prefix, which read as the decision log's WARN lines.
    old = mutate(PHASE, "[MOE-EXEC-TRACE] stage layer", "[MOE-EXEC] trace layer")
    assert not claim_trace_line(old, {PHASE_REL: old})


STAGE_LINE_FMT = '"' + TRACE_FMT + ' unowned=%d untyped=%d exhausted=%d"'
STAGE_LINE_ARGS = ("st.decision.t_ring_us, st.window_us, n.staged, n.wanted, n.bytes, n.unowned, n.untyped, "
                   "n.exhausted);")
TRACE_GUARD = "if (op.log->trace >= 1 && op.log->trace_sink != nullptr)"
TRACE_SPLIT_FMT = '"[MOE-EXEC-TRACE] stage layouts layer %d ubatch %u: soa=%d aos_only=%d"'
TRACE_SPLIT_ARGS = "layer, ubatch_seq_, n.soa, n.aos_only);"
TRACE_SPLIT_EMIT = "op.log->trace_sink(op.log->sink_ctx, split);"
STAGE_COUNTS_SPLIT = "int soa = 0; int aos_only = 0;"


def claim_stage_trace_splits_layouts(phase: str, hpp: str) -> bool:
    """The staged matrices split by the layout their slot holds, soa= and aos_only=, so a census can tell which path a
    layer took (the lead's addition to ruling c-o9uw-1). The split is a line of its own, after the stage line and
    under the same trace guard, so the stage line stays byte-identical to master's: its literal ends at exhausted=%d
    (lead ruling c-9wgo). Each staged matrix is counted once, in the branch that writes its stage-table entry with that
    layout."""
    n = norm(phase)
    guarded = body(n, TRACE_GUARD)
    return (n.count(STAGE_LINE_FMT) == 1 and n.count(TRACE_SPLIT_FMT) == 1 and n.count("soa=%d") == 1
            and ordered(guarded, STAGE_LINE_FMT, STAGE_LINE_ARGS, TRACE_EMIT, TRACE_SPLIT_FMT, TRACE_SPLIT_ARGS,
                        TRACE_SPLIT_EMIT)
            and AOS_ONLY_RECORD in body(n, STAGE_SIG) and n.count("++n.soa;") == 1 and n.count("++n.aos_only;") == 1
            and STAGE_COUNTS_SPLIT in norm(hpp))


def test_stage_trace_splits_layouts():
    assert claim_stage_trace_splits_layouts(PHASE, PLAN_HPP)


def test_stage_trace_split_mutants():
    # The split appended to the stage line, which changes its bytes.
    assert not claim_stage_trace_splits_layouts(
        mutate(PHASE, STAGE_LINE_FMT, STAGE_LINE_FMT[:-1] + ' soa=%d aos_only=%d"'), PLAN_HPP)
    # The two counts printed under each other's names.
    assert not claim_stage_trace_splits_layouts(
        mutate(PHASE, TRACE_SPLIT_ARGS, "layer, ubatch_seq_, n.aos_only, n.soa);"), PLAN_HPP)
    # The split line formatted but never emitted.
    assert not claim_stage_trace_splits_layouts(mutate(PHASE, TRACE_SPLIT_EMIT, "(void) split;"), PLAN_HPP)
    # The split line under another prefix.
    assert not claim_stage_trace_splits_layouts(
        mutate(PHASE, "[MOE-EXEC-TRACE] stage layouts layer", "[MOE-EXEC] stage layouts layer"), PLAN_HPP)
    # The split emitted outside the trace guard.
    outside = mutate(PHASE, TRACE_SPLIT_EMIT + " }", "} " + TRACE_SPLIT_EMIT)
    assert not claim_stage_trace_splits_layouts(outside, PLAN_HPP)
    # A staged matrix left uncounted, or counted under the other layout.
    assert not claim_stage_trace_splits_layouts(mutate(PHASE, "++n.soa;", ""), PLAN_HPP)
    assert not claim_stage_trace_splits_layouts(mutate(PHASE, "++n.aos_only;", "++n.soa;"), PLAN_HPP)
    # The counters missing from stage_counts.
    assert not claim_stage_trace_splits_layouts(PHASE, mutate(PLAN_HPP, "int aos_only = 0;", "int aos = 0;"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
