"""Source contract for the MoE stream phase (llama.cpp-xod7, plan Task 7b of llama.cpp-36j0, reshaped by
llama.cpp-9433, Task 2b of llama.cpp-a2xb): a prompt layer's host experts are weighed at its first MUL_MAT_ID, and the
executor's windows of a layer it claims are copied into ring slots on the BCS queue, one executor phase at a time.

Host-only: reads sources, builds nothing, loads no model and touches no device.

What this gate pins:
  - ggml_sycl_mul_mat_id consults the prompt phase at exactly one site: inside the prompt ids-admission branch
    (`ne12 != 1`, guarded by the same `ids->ne[1] > 1` that defines a prompt batch), after the graph-sink refusal (so a
    ring copy is never recorded into a command graph) and the ids snapshot the layer routes on, and before
    `moe_hybrid_with_plan` is computed. The site is the admission branch rather than the later `prompt_batch` local
    because the executor fast paths return before that local is computed;
  - staging follows the layer's decision and the executor's claim (llama.cpp-3460; W2 of
    docs/plans/2026-10-24-prefill-window-ring.md): the phase's one stage_window() call is in window_begin, which aborts
    unless executor_claims answers true for the layer, and executor_claims needs the RING choice, a layer the executor
    claims whole (moe_stream_executor_claims) and a reserved ring. begin_layer keeps a RING decision, under any policy
    but GGML_SYCL_MOE_EXEC=cpu (moe_stream_stage_now), only where that claim holds and the ring reserves. The reserve
    is reachable only through ring_ready(), which those two ask, and the op's policy is the parsed config's;
  - a refused ring reserve is not retried within a micro-batch and is tried once more in the next: the latch is cleared
    by begin_ubatch, which every new micro-batch runs, and by end_prompt. An expert loan (ruling c-tdra, llama.cpp-mwmr)
    can make room before the next micro-batch, so a per-prompt latch would waste the loan its own prompt asked for (lead
    ruling c-i8rz);
  - begin_ubatch clears the stage table and then forgets the last window's layer, the prefetch id and its whole flag
    (llama.cpp-9433): a micro-batch can start at the layer the last one ended on, so a kept id would take a cleared
    window as prefetched and skip staging it, and a kept flag would abort on the entries the clear dropped;
  - a refused reserve asks for that loan (llama.cpp-3460, closing c-0tmn): the request is made once, in ring_ready()'s
    reserve-failed branch after the latch, with the shortfall the reserve reports (the bytes of the slots it still has
    to allocate, llama.cpp-am87), through op hooks the backend wires to ggml_sycl_request_expert_loan and to the
    boundary runner's REFUSED count. A loan this prompt asked for that comes back REFUSED (never BUSY, which the runner
    re-posts) latches until end_prompt, with one WARN under an explicit ring or a trace;
  - a layer answering RING_UNAVAILABLE under an explicit ring or a trace prints one WARN per prompt that names the rule
    (the first is the executor's claim, "executor does not claim") and the ring's inputs, in a fixed format, through the
    decision log's WARN sink; the flag is cleared by end_prompt and never per micro-batch (lead ruling c-kcyf), and the
    phase logs nothing else of its own but the refused-loan line, under the same gating;
  - begin_layer reports every layer it decides, under every policy, so the decision log keeps the cost model's choice;
    it weighs only its own layer, with no copy window (there is no layer-ahead copy), and stages nothing;
  - the copy is mem_copy_async on the unified cache's BCS queue, chained behind the ring region's gate (ruling A,
    c-3bde: slot reuse is ordered on the device, never by a host status query) and behind the host lease's ready event,
    since the lease admits an entry still being filled. It is the window matrix's one submission: stage_window submits
    exactly one copy and no reorder (the SOA staging is retired), and the phase submits no reorder anywhere;
  - the AOS slot and the host lease are retained until the copy's event, so each submission owns what it touches, and
    the stage-table entry holds the AOS slot, the copy's event, layout AOS and the consumer the index build named
    (ruling C: the entry is the answer);
  - every window matrix takes one slot, acquired with the layout it holds, AOS (lead ruling c-o9uw-1), and the ring is
    sized so: one slot per staged matrix (moe_stream_staging_slots), one executor window per region;
  - what stages comes from the library executor's claim alone (W2): the index build marks a matrix whose weight the
    oneDNN batched prompt executor claims AOS_ONLY with consumer EXECUTOR and leaves every other one NONE, its
    default; it takes the prompt-kernel predicate but never asks it. The backend answers the claims predicate with
    ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor, the check the staged route refuses a weight on (c-las5), from
    call.plan, the plan the index build holds (lead ruling c-gi44); only the check's two-argument entry point, the
    batched executor's own route, reads the device's plan, by passing no plan;
  - the prompt-kernel predicate answers through the one helper the staged route refuses on (llama.cpp-o9uw),
    ggml_sycl_moe_staged_capability_refusal, which admits only a supported MMVQ_COMPAT kernel for a type with no
    direct-kernel requirement (Q1_0 and NVFP4 run only through their direct kernel, which the recipe validator and the
    executor choice ask through moe_type_needs_direct_kernel). Since W2 the ring stages from the library claim alone
    and never asks the predicate; the pin keeps the still-wired predicate on the resolver's helper;
  - a region closes on the event of its last copy;
  - a refilled region's copies wait for its read gate: the barrier the executor submits behind the dequants of the
    window the region last held and hands to window_done (lead ruling c-o9uw-1), recorded for the region holding that
    window; a compute-queue barrier submitted at the first copy stands in when none was recorded, and a region no copy
    has gone into has no reader to wait for. Each region's last window and read gate live in one place,
    moe_stream_region_reads, which stage_window refills, window_done gates and reset() forgets (test-moe-stream-phase
    drives it); layer_done records no read gate. One helper, reset_regions, forgets every region's keys and resets it,
    and both a new ring and teardown run it;
  - a throw from the copy aborts with "[MOE-EXEC] stream enqueue failed: %s" (design 9.3);
  - moe-stream-plan.cpp waits on nothing (.wait(), wait_and_throw()), queries no event status (get_info<), has no
    const_cast (ruling E) and calls no allocator: its slots come from the ring's reservation;
  - the ring reserves slot by slot (llama.cpp-am87): each slot is its own allocation, the WEIGHT zone first and then
    the general entry point with the ring's intent, so room scattered in holes a slot wide gives the ring. A refused
    slot stops the loop and the slots already held stay held, reported short by the slots still to allocate; the next
    reserve() of the same ring continues from them, and only another geometry or device gives them back first. A
    partial reservation hands out no slot, and the ring no longer reads the zone's largest free block;
  - the number of waits in ggml_sycl_mul_mat_id is the one master had at 84fb449da;
  - only a host lease whose retained owner is a host-pinned WEIGHT allocation of this device's host zones is copied
    (llama.cpp-9y0p: any other lease would take the staged path), only a matrix one AOS slot holds is staged, and a
    window lists only matrices not staged NONE (moe_stream_window_items);
  - a matrix the ring cannot take (no lease, a short lease, or an owner it cannot copy from) reaches the executor as a
    decline, not an abort (llama.cpp-9433): stage_window skips it before the free-slot check, and window_begin re-stages
    only a window no prefetch took, aborts on a prefetched one only after a whole prefetch, has three aborts (the claim
    guard, a window larger than its region, entries lost after a whole prefetch) and returns what the table holds. No
    host test reaches window_begin, so this pin is the interim decline path's only witness;
  - teardown gates the open region and hands every slot of the ring to the drain until the last use as it releases the
    ring (ruling D, c-fjx1; every slot since llama.cpp-am87, because a region's gate can be its last copy, which
    precedes the region's readers), and the ring is offered only on an in-order compute queue;
  - the backend context owns the phase, resets it after draining its queues and before the retained handles drain,
    ends the prompt at the first decode graph, and closes a layer's region after its down MUL_MAT_ID;
  - the phase says whether the prompt it ended kept a whole ring (llama.cpp-i4ny): the flag is set only where
    ring_ready() keeps a whole reservation, never for a refused one kept pending; end_prompt copies it and clears it
    only past its early return, so the decode graphs after the first leave the answer standing, and marks the prompt
    ended for take_prompt_ended(), which answers true once per ended prompt;
  - begin_layer and window_begin decide through the header's pure predicates (a micro-batch's start, staging now, the
    executor's claim, a window's id, items and exposure), which test-moe-stream-phase pins, and never restate them;
  - the window trace line "[MOE-EXEC-TRACE] stage window layer %d role %s phase %zu: %d items %zu bytes" (plan Task 2b)
    exists once in the backend, under the trace guard, after every window stage; the per-layer stage line and its
    per-layout split are gone with the layer stage.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text with adjacent string literals joined, and has a
mutant that must make it fail.

Pytest-style (module-level test_* functions): registered with llama_test_pytest.
"""

import re
from pathlib import Path
from sycl_gate import body, first_missing, first_present, mutate, norm, ordered, read

ROOT = Path(__file__).resolve().parents[1]


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


MMID_SIG = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try"
ADMIT_BRANCH = "if (ne12 != 1)"
BEGIN_CALL = "moe_stream->begin_layer("
BEGIN_GUARD = "if (ids->ne[1] > 1 && ggml_sycl_moe_stream_prepare(ctx, src0, ids, stream_call)) {"
GRAPH_SINK = 'throw ggml_sycl_fallback_error("MUL_MAT_ID retained prompt batch cannot enter graph sink");'
SNAPSHOT = "prompt_ids_snapshot = prompt_ids_entry.host_ids;"
HYBRID_WITH_PLAN = "const bool moe_hybrid_with_plan ="

COPY_DECL = "copied = mem_copy_async("
REORDER_CALL = "ggml_sycl_reorder_weight_gpu("
RETAIN = "retain_handles_until_event({ aos, host }, copied);"
TABLE_WRITE = "table_[key] = { mem_handle(), aos, copied, MOE_STREAM_RING_SLOT_LAYOUT, item.consumer };"
ABORT = 'GGML_ABORT("[MOE-EXEC] stream enqueue failed: %s", e.what());'
RETHROW = "catch (const ggml_sycl_fallback_error &) { throw; }"
TRACE_FMT = "[MOE-EXEC-TRACE] stage window layer %d role %s phase %zu: %d items %zu bytes"
TRACE_EMIT = "log_->trace_sink(log_->sink_ctx, line);"

# Waits in ggml_sycl_mul_mat_id at master 84fb449da, counted on the normalized body: 5 `.wait()`, 3 `->wait()`,
# 8 `wait_and_throw()`. The prompt phase adds none.
MASTER_MMID_WAITS = 16
MMID_WAIT_RE = re.compile(r"\.wait\(\)|->wait\(\)|\bwait_and_throw\(\)")
ANY_WAIT_RE = re.compile(r"\bwait\s*\(|\bwait_and_throw\s*\(")


# ---- the consultation site -------------------------------------------------------------------------------------------


def claim_begin_layer_once_in_prompt_branch(sycl: str) -> bool:
    """One consultation site, in ggml_sycl_mul_mat_id's prompt ids-admission branch: after the graph-sink refusal and
    the ids snapshot, guarded by the prompt-batch predicate, passing the snapshot, which begin_layer no longer reads
    (a signature pin), and before moe_hybrid_with_plan."""
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
    assert claim_begin_layer_once_in_prompt_branch(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of ggml_sycl_mul_mat_id (the admission branch holds the "
        "begin_layer call)",
        body(norm(SYCL), MMID_SIG),
        BEGIN_CALL, ADMIT_BRANCH + " {", "ctx.moe_graphs_disabled_once = true;", GRAPH_SINK, SNAPSHOT, BEGIN_GUARD,
        HYBRID_WITH_PLAN,
        signature=MMID_SIG)


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
    # the snapshot not passed: begin_layer no longer reads it, so this pins the call's signature
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
    assert claim_copy_on_bcs_behind_region_gate(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: whole file",
        norm(PHASE),
        HOST_RESOLVE, COPY_DEPS, READY_DEP, COPY_DECL)


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
        mutate(PHASE, "aos, host, item.src_bytes, ", "aos, host, host_bytes, "))


def claim_window_copy_is_the_only_submission(phase: str) -> bool:
    """W2 (Task 2b of llama.cpp-a2xb): the executor's dequant kernel reads every window matrix in place as AOS, so
    stage_window submits exactly one thing per matrix, its copy: no reorder and no kernel of its own. The phase submits
    no reorder anywhere: the SOA staging that ggml_sycl_reorder_weight_gpu served is retired."""
    n = norm(phase)
    s = body(n, STAGE_SIG)
    return (bool(s) and s.count("mem_copy_async(") == 1 and REORDER_CALL not in n
            and ".submit(" not in s and "->submit(" not in s)


def test_window_copy_is_the_only_submission():
    assert claim_window_copy_is_the_only_submission(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::stage_window (the claim also fails "
        "on a second submission or a reorder call)",
        body(norm(PHASE), STAGE_SIG),
        "mem_copy_async(",
        signature=STAGE_SIG)


def test_window_copy_only_mutants():
    c = claim_window_copy_is_the_only_submission
    # A second copy of the matrix.
    second_copy = " (void) mem_copy_async(aos, host, item.src_bytes, cache.get_bcs_queue(), { copied });"
    assert not c(mutate(PHASE, RETAIN, RETAIN + second_copy))
    # The SOA reorder back, on the compute queue behind the copy.
    reorder = " (void) ggml_sycl_reorder_weight_gpu(*ccs_, aos, item.type, { copied });"
    assert not c(mutate(PHASE, RETAIN, RETAIN + reorder))
    # A kernel of the stage's own.
    assert not c(mutate(PHASE, RETAIN, RETAIN + " ccs_->submit([&](sycl::handler & h) { (void) h; });"))


def claim_window_entry_reads_in_place(phase: str) -> bool:
    """Each window matrix is read where its copy lands. The AOS slot and the host lease are retained until the copy's
    event, so the submission's ownership does not lean on the table (CLAUDE.md: retain the handles behind a kernel's raw
    pointers for the queued work). The stage-table entry, written once, holds no SOA slot, the AOS slot, the copy's
    event, layout AOS (ruling C: the layout comes from the entry) and the consumer the index build named for the item,
    the executor (llama.cpp-3460)."""
    n = norm(phase)
    s = body(n, STAGE_SIG)
    return (n.count(RETAIN) == 1 and n.count(TABLE_WRITE) == 1 and n.count("table_[key] =") == 1
            and ordered(s, COPY_DECL, RETAIN, TABLE_WRITE))


def test_window_entry_reads_in_place():
    assert claim_window_entry_reads_in_place(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::stage_window",
        body(norm(PHASE), STAGE_SIG),
        COPY_DECL, RETAIN, TABLE_WRITE,
        signature=STAGE_SIG)


def test_window_entry_mutants():
    c = claim_window_entry_reads_in_place
    # The host lease left to the table's hold.
    assert not c(mutate(PHASE, RETAIN, "retain_handles_until_event({ aos }, copied);"))
    # The slot left to the table's hold.
    assert not c(mutate(PHASE, RETAIN, "retain_handles_until_event({ host }, copied);"))
    # Retained until the region's gate, which precedes the copy.
    assert not c(mutate(PHASE, RETAIN, "retain_handles_until_event({ aos, host }, ring_->current_region_gate());"))
    # Tagged SOA: a reader would read the slot as a layout it does not hold.
    assert not c(mutate(PHASE, TABLE_WRITE, TABLE_WRITE.replace("MOE_STREAM_RING_SLOT_LAYOUT", "GGML_LAYOUT_SOA")))
    # A SOA slot named that no reorder wrote.
    assert not c(mutate(PHASE, TABLE_WRITE,
                        "table_[key] = { aos, aos, copied, MOE_STREAM_RING_SLOT_LAYOUT, item.consumer };"))
    # The entry's event the region's gate instead of the copy.
    assert not c(mutate(PHASE, TABLE_WRITE,
                        "table_[key] = { mem_handle(), aos, ring_->current_region_gate(), MOE_STREAM_RING_SLOT_LAYOUT, "
                        "item.consumer };"))
    # The consumer dropped: the entry would read as the staged route's.
    assert not c(mutate(PHASE, TABLE_WRITE,
                        "table_[key] = { mem_handle(), aos, copied, MOE_STREAM_RING_SLOT_LAYOUT };"))


def claim_submission_failure_aborts(phase: str) -> bool:
    """The copy and the retention sit in one try block: a planned refusal is rethrown first (the alloc-zone contract's
    G-CATCH rule), and every other exception aborts with the design's message."""
    n = norm(phase)
    at = n.find("try {")
    while at >= 0:
        blk = body(n[at:], "try")
        end = n.find("{", at) + len(blk)
        if all(s in blk for s in (COPY_DECL, RETAIN)):
            rest = n[end:].lstrip()
            if not rest.startswith(RETHROW):
                return False
            rest = rest[len(RETHROW):].lstrip()
            handler = body(rest, "catch (const std::exception & e)")
            return rest.startswith("catch (const std::exception & e) {") and ABORT in handler
        at = n.find("try {", at + 1)
    return False


def test_submission_failure_aborts():
    assert claim_submission_failure_aborts(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: whole file (the try block holding the copy, and the handler "
        "after it)",
        norm(PHASE),
        COPY_DECL, RETAIN, RETHROW, "catch (const std::exception & e) {", ABORT)


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
    assert claim_phase_never_waits(PHASE), first_present(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: whole file (literal forms of ANY_WAIT_RE; the claim also fails on "
        "an empty file)",
        norm(PHASE),
        "wait(", "wait_and_throw(", "wait (", "wait_and_throw (")


def test_phase_wait_mutants():
    assert not claim_phase_never_waits(mutate(PHASE, RETAIN, RETAIN + " copied.wait();"))
    assert not claim_phase_never_waits(mutate(PHASE, RETAIN, RETAIN + " ccs_->wait_and_throw();"))


def claim_mul_mat_id_waits_unchanged(sycl: str) -> bool:
    b = body(norm(sycl), MMID_SIG)
    return bool(b) and len(MMID_WAIT_RE.findall(b)) == MASTER_MMID_WAITS


def test_mul_mat_id_waits_unchanged():
    assert claim_mul_mat_id_waits_unchanged(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the claim counts the waits in the body of "
        "ggml_sycl_mul_mat_id against the count on master)",
        norm(SYCL),
        MMID_SIG)


def test_mul_mat_id_wait_mutants():
    assert not claim_mul_mat_id_waits_unchanged(mutate(SYCL, SNAPSHOT, SNAPSHOT + " ctx.stream()->wait();"))
    assert not claim_mul_mat_id_waits_unchanged(mutate(SYCL, SNAPSHOT, SNAPSHOT + " ctx.stream()->wait_and_throw();"))


def claim_phase_queries_no_status_and_casts_no_const(phase: str) -> bool:
    n = norm(phase)
    return bool(n) and "get_info<" not in n and "const_cast" not in n


def test_phase_queries_no_status_and_casts_no_const():
    assert claim_phase_queries_no_status_and_casts_no_const(PHASE), first_present(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: whole file (the claim also fails on an empty file)",
        norm(PHASE),
        "get_info<", "const_cast")


def test_status_and_const_cast_mutants():
    assert not claim_phase_queries_no_status_and_casts_no_const(
        mutate(PHASE, RETAIN, RETAIN + " (void) copied.get_info<sycl::info::event::command_execution_status>();"))
    assert not claim_phase_queries_no_status_and_casts_no_const(
        mutate(PHASE, RETAIN, RETAIN + " (void) const_cast<moe_stream_ring *>(ring_.get());"))


ALLOCATOR_NAMES = ("malloc", "unified_alloc", "unified_allocate", "zone_allocate", "from_owned_alloc", "pinned_pool")


def claim_phase_allocates_only_ring_slots(phase: str) -> bool:
    """No allocator: the one slot each window matrix takes is ring_->acquire_slot(MOE_STREAM_RING_SLOT_LAYOUT), from
    the ring's reservation."""
    n = norm(phase)
    return bool(n) and not any(a in n for a in ALLOCATOR_NAMES) and n.count("ring_->acquire_slot(") == 1


def allocates_only_ring_slots_message(phase: str) -> str:
    return first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: whole file (the claim also fails on an allocator name)",
        norm(phase),
        "ring_->acquire_slot(")


def test_phase_allocates_only_ring_slots():
    assert claim_phase_allocates_only_ring_slots(PHASE), allocates_only_ring_slots_message(PHASE)


def test_allocator_mutants():
    assert not claim_phase_allocates_only_ring_slots(
        mutate(PHASE, RETAIN, RETAIN + " (void) sycl::malloc_device(16, *ccs_);"))
    assert not claim_phase_allocates_only_ring_slots(
        mutate(PHASE, "ring_->acquire_slot(MOE_STREAM_RING_SLOT_LAYOUT)", "mem_handle()"))


# ---- what is streamed ------------------------------------------------------------------------------------------------

OWNER_SIG = "static inline bool moe_stream_owner_streamable(const alloc_metadata & owner, int device)"
AOS_FITS_SIG = "static inline bool moe_stream_item_aos_fits(const moe_stream_item & item, size_t slot_bytes)"
OWNER_CALL = "host.owner_metadata(&owner) && moe_stream_owner_streamable(owner, device)"
TYPED_GUARD = "if (!moe_stream_item_aos_fits(item, geo_.slot_bytes)) { continue; }"
WINDOW_ITEMS_SIG = "static inline std::vector<moe_stream_item> moe_stream_window_items("
WINDOW_ITEMS_SKIP = "item.staging != moe_stream_staging::NONE"
WINDOW_BEGIN_SIG = "uint32_t moe_stream_phase::window_begin("


def claim_only_direct_owned_typed_items_stream(phase: str, hpp: str) -> bool:
    """llama.cpp-9y0p: the lease's retained owner is a host-pinned WEIGHT allocation of this device's host zones (the
    shape mem-ops.cpp copies directly), and the matrix fits one AOS slot; both are checked before the slot is taken and
    the copy is made. A window lists only matrices the index build did not stage NONE (W2: the ones the executor
    claims), and window_begin builds both of its windows, this one and the prefetched one, from that helper alone."""
    h = norm(hpp)
    owner = body(h, OWNER_SIG)
    aos_fits = body(h, AOS_FITS_SIG)
    n = norm(phase)
    return ("owner.tier == alloc_tier::HOST_PINNED" in owner and "owner.role == alloc_role::WEIGHT" in owner
            and "owner.host_zone != host_zone_id::COUNT" in owner and "owner.device == device" in owner
            and "detail::moe_stream_item_rows_bytes(item) != 0" in aos_fits
            and "item.src_bytes <= slot_bytes" in aos_fits
            and WINDOW_ITEMS_SKIP in body(h, WINDOW_ITEMS_SIG)
            and n.count("moe_stream_window_items(") == 2
            and body(n, WINDOW_BEGIN_SIG).count("moe_stream_window_items(li->second, ") == 2
            and ordered(body(n, STAGE_SIG), TYPED_GUARD, OWNER_CALL, "ring_->acquire_slot(", COPY_DECL))


def test_only_direct_owned_typed_items_stream():
    assert claim_only_direct_owned_typed_items_stream(PHASE, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::stage_window (the owner, fit and "
        "item predicates are searched in moe-stream-plan.hpp)",
        body(norm(PHASE), STAGE_SIG),
        TYPED_GUARD, OWNER_CALL, "ring_->acquire_slot(", COPY_DECL,
        signature=STAGE_SIG)


def test_streamed_item_mutants():
    c = claim_only_direct_owned_typed_items_stream
    assert not c(mutate(PHASE, OWNER_CALL, "host.valid()"), PLAN_HPP)
    assert not c(PHASE, mutate(PLAN_HPP, "owner.role == alloc_role::WEIGHT", "owner.role != alloc_role::OTHER"))
    # A matrix copied without checking that it fits the slot.
    assert not c(mutate(PHASE, "moe_stream_item_aos_fits(item, geo_.slot_bytes)", "true"), PLAN_HPP)
    assert not c(PHASE, mutate(PLAN_HPP, "item.src_bytes <= slot_bytes", "true"))
    # A window that lists a matrix staged NONE, one the executor does not claim.
    assert not c(PHASE, mutate(PLAN_HPP, " && " + WINDOW_ITEMS_SKIP, ""))
    # The prefetched window built from the layer's every item instead of the helper.
    assert not c(mutate(PHASE, "moe_stream_window_items(li->second, nrole, req.next, req.n_next)", "li->second.items"),
                 PLAN_HPP)


# ---- one AOS slot per window matrix (W2) -----------------------------------------------------------------------------

SLOTS_CHECK = "if (ring_->slots_free() == 0) { ++n.exhausted; continue; }"
AOS_ACQUIRE = "mem_handle aos = ring_->acquire_slot(MOE_STREAM_RING_SLOT_LAYOUT);"
STAGING_ENUM = "enum class moe_stream_staging : uint8_t { NONE, AOS_ONLY, };"
STAGING_SLOTS_SIG = "static inline uint32_t moe_stream_staging_slots(moe_stream_staging staging)"
STAGING_SLOTS_BODY = "{ return staging == moe_stream_staging::AOS_ONLY ? 1u : 0u; }"
REGION_SIZE = "li.region_slots += static_cast<int>(moe_stream_staging_slots(item.staging));"
INDEX_BUILD_SIG = "static inline moe_stream_plan_index moe_stream_plan_index_build("
RING_GEOMETRY_SIG = "static inline moe_stream_ring_geometry moe_stream_plan_ring_geometry("
# The one place the ring is sized (llama.cpp-kjef): the phase's prepare() and the auto-ubatch fit both call the helper.
# A region holds one executor window, not a whole layer (docs/plans/2026-10-24-prefill-window-ring.md, W3).
RING_SIZE_CALL = "moe_stream_ring_size(index.window_slots, index.matrix_bytes_max, resident_with_lent)"


def claim_window_matrix_takes_one_aos_slot(phase: str, hpp: str) -> bool:
    """W2: every slot holds AOS, so a staged matrix takes ONE slot, both when the ring is sized (one slot per staged
    matrix from the one slot-count helper, the ring sized from the index's window) and when a window stages it: the
    region is checked for a free slot, then the one slot is acquired with the layout it will hold, AOS (lead ruling
    c-o9uw-1), then the copy goes in. Staging has no SOA value to size a second slot for."""
    h = norm(hpp)
    n = norm(phase)
    s = body(n, STAGE_SIG)
    return (h.count(STAGING_ENUM) == 1 and body(h, STAGING_SLOTS_SIG) == STAGING_SLOTS_BODY
            and REGION_SIZE in body(h, INDEX_BUILD_SIG) and RING_SIZE_CALL in body(h, RING_GEOMETRY_SIG)
            and "moe_stream_plan_ring_geometry(" in body(n, PREPARE_SIG)
            and n.count("ring_->acquire_slot(") == 1 and ordered(s, SLOTS_CHECK, AOS_ACQUIRE, COPY_DECL))


def test_window_matrix_takes_one_aos_slot():
    assert claim_window_matrix_takes_one_aos_slot(PHASE, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::stage_window (the staging enum and "
        "the geometry are searched in moe-stream-plan.hpp)",
        body(norm(PHASE), STAGE_SIG),
        SLOTS_CHECK, AOS_ACQUIRE, COPY_DECL,
        signature=STAGE_SIG)


def test_window_slot_mutants():
    c = claim_window_matrix_takes_one_aos_slot
    # The slot tagged with a layout it will not hold: the slice's tag would disagree with the entry's layout.
    assert not c(mutate(PHASE, AOS_ACQUIRE, "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_SOA);"), PLAN_HPP)
    # A second slot per matrix.
    assert not c(mutate(PHASE, AOS_ACQUIRE, AOS_ACQUIRE + " mem_handle soa = ring_->acquire_slot(GGML_LAYOUT_SOA);"),
                 PLAN_HPP)
    # The SOA staging back.
    assert not c(PHASE, mutate(PLAN_HPP, STAGING_ENUM,
                               "enum class moe_stream_staging : uint8_t { NONE, SOA, AOS_ONLY, };"))
    # The ring sized as if every matrix took two.
    assert not c(PHASE, mutate(PLAN_HPP, "moe_stream_staging::AOS_ONLY ? 1u", "moe_stream_staging::AOS_ONLY ? 2u"))
    # No free-slot check, or a check for two.
    assert not c(mutate(PHASE, SLOTS_CHECK, ""), PLAN_HPP)
    assert not c(mutate(PHASE, "ring_->slots_free() == 0", "ring_->slots_free() < 2"), PLAN_HPP)
    # The ring sized from the matrix count, as before c-hmey.
    assert not c(PHASE, mutate(PLAN_HPP, RING_SIZE_CALL, "moe_stream_ring_size(2 * index.host_matrices_per_layer_max, "
                                                         "index.matrix_bytes_max, resident_with_lent)"))


STAGE_CLOSE = "close_region(region_last_);"
ITEM_LOOP = "for (const moe_stream_item & item : items) {"
REGION_KEYS_CLEAR = "region_keys_[cur_region_].clear();"
REGION_LAST = "region_last_ = copied;"


def claim_region_closes_after_last_copy(phase: str) -> bool:
    """A region closes on the event of its last copy: each copy is its matrix's only submission, and the in-order copy
    queue runs it after every earlier copy, so every item records its copy as the region's last event and the next
    stage closes the region on it. The stage closes the region in use first, which turns the ring to the other region,
    and only then drops the entries of the window that region last held."""
    s = body(norm(phase), STAGE_SIG)
    return (s.count(STAGE_CLOSE) == 1 and s.count(REGION_LAST) == 1 and s.count("region_last_ =") == 1
            and ordered(s, STAGE_CLOSE, REGION_KEYS_CLEAR, ITEM_LOOP, COPY_DECL, REGION_LAST))


def test_region_closes_after_last_copy():
    assert claim_region_closes_after_last_copy(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::stage_window",
        body(norm(PHASE), STAGE_SIG),
        STAGE_CLOSE, REGION_KEYS_CLEAR, ITEM_LOOP, COPY_DECL, REGION_LAST,
        signature=STAGE_SIG)


def test_region_close_mutants():
    c = claim_region_closes_after_last_copy
    # The copy never recorded: the region would close on an older event.
    assert not c(mutate(PHASE, REGION_LAST, ""))
    # The region closed on its own gate, which precedes the copies.
    assert not c(mutate(PHASE, REGION_LAST, "region_last_ = ring_->current_region_gate();"))
    # No close: the stage would refill the region in use.
    assert not c(in_body(PHASE, STAGE_SIG, STAGE_CLOSE, ""))
    # The entries dropped before the close: they would be the region in use's, the window the executor still reads.
    early = in_body(PHASE, STAGE_SIG, STAGE_CLOSE + " ", "")
    early = in_body(early, STAGE_SIG, REGION_KEYS_CLEAR, REGION_KEYS_CLEAR + " " + STAGE_CLOSE)
    assert not c(early)


REUSE_COND = "const bool refill_after_readers = region_reads_.window[cur_region_] != UINT32_MAX;"
REUSE_TAKE = ("if (refill_after_readers && !reuse_gate) { reuse_gate = region_reads_.read_gate[cur_region_] ? "
              "*region_reads_.read_gate[cur_region_] : ccs_->ext_oneapi_submit_barrier(); }")
REUSE_DEP = "if (reuse_gate) { deps.push_back(*reuse_gate); }"
REUSE_RECORD = "if (n.staged != 0) { region_reads_.refill(cur_region_, window_id); }"
REGION_FLAGS_RESET = "region_reads_.reset();"
# The regions' last uses, kept in one place (moe-stream-plan.hpp) and held by the phase over sycl::event.
READS_SIG = "template <typename event_t> struct moe_stream_region_reads"
READS_FIELDS = "uint32_t window[2] = { UINT32_MAX, UINT32_MAX }; std::optional<event_t> read_gate[2];"
READS_MEMBER = "moe_stream_region_reads<sycl::event> region_reads_;"
REFILL_SIG = "void refill(int region, uint32_t w)"
REFILL_BODY = "{ window[region] = w; read_gate[region].reset(); }"
READS_DONE_SIG = "void window_done(uint32_t w, const event_t & readers_done)"
READS_DONE_BODY = ("{ for (int r = 0; r < 2 && w != UINT32_MAX; ++r) { if (window[r] == w) { "
                   "read_gate[r] = readers_done; } } }")
READS_RESET_SIG = "void reset()"
READS_RESET_BODY = "{ for (int r = 0; r < 2; ++r) { window[r] = UINT32_MAX; read_gate[r].reset(); } }"
RESET_REGIONS_SIG = "void moe_stream_phase::reset_regions()"
REGION_STATE_RESET = ("region_keys_[0].clear(); region_keys_[1].clear(); cur_region_ = 0; region_submitted_ = false; "
                      "region_last_ = sycl::event();")
RING_READY_RESET = "ring_ = std::move(ring); reset_regions(); return true;"
TEARDOWN_RESET = "table_.clear(); reset_regions();"
PHASE_WINDOW_DONE_SIG = ("void moe_stream_phase::window_done(int layer, expert_tensor_role role, size_t phase, "
                         "sycl::event readers_done)")
PHASE_LAYER_DONE_SIG = "void moe_stream_phase::layer_done(int layer, sycl::event down_done)"
READ_GATE_RECORD = "region_reads_.window_done(moe_stream_window_id(layer, role, phase), readers_done);"


def claim_reuse_waits_for_window_readers(phase: str, hpp: str = PLAN_HPP) -> bool:
    """Every slot is read in place, so a copy that refills a region may rewrite a slot a dequant read in the region's
    last fill, and the region's gate does not follow those readers, which ran on the compute queue (c-las5). So every
    copy of a refill also waits for the region's read gate (lead ruling c-o9uw-1): the barrier the executor submitted
    behind the dequants of the window the region last held and handed to window_done, recorded for the region holding
    that window. A region turns over several times per layer, so the gate is the window's, never the layer's: layer_done
    records none. Without one, the copies wait for a compute-queue barrier submitted at the first copy; a region no copy
    has gone into (window UINT32_MAX) has no reader to wait for, and the region's last window is read after the close
    turned the ring to it. A copy going in makes the window the region's last fill, with no read gate yet. A new ring
    starts with no window and no read gate per region, and teardown leaves none behind: both run the one helper that
    forgets them. The two facts per region live in one struct (moe_stream_region_reads), whose refill, window_done and
    reset are the only writers, and the phase holds no copy of its own."""
    n = norm(phase)
    s = body(n, STAGE_SIG)
    done = body(n, PHASE_WINDOW_DONE_SIG)
    reset = body(n, RESET_REGIONS_SIG)
    h = norm(hpp)
    at = h.find(READS_SIG + " {")
    reads = body(h[at:], READS_SIG) if at >= 0 else ""
    return (s.count(REUSE_TAKE) == 1 and s.count(REUSE_DEP) == 1 and s.count(REUSE_RECORD) == 1
            and ordered(s, STAGE_CLOSE, REUSE_COND, ITEM_LOOP, COPY_DEPS, READY_DEP, REUSE_TAKE, REUSE_DEP, COPY_DECL,
                        REUSE_RECORD)
            and n.count(READ_GATE_RECORD) == 1
            and ordered(done, "if (layer != cur_layer_) { return; }", READ_GATE_RECORD)
            and "region_reads_" not in body(n, PHASE_LAYER_DONE_SIG)
            and h.count(READS_SIG + " {") == 1 and READS_FIELDS in reads and "aos_only" not in reads
            and body(reads, REFILL_SIG) == REFILL_BODY and body(reads, READS_DONE_SIG) == READS_DONE_BODY
            and body(reads, READS_RESET_SIG) == READS_RESET_BODY
            and reads.count("read_gate[") == 4 and h.count(READS_MEMBER) == 1
            and n.count("region_reads_.refill(") == 1 and n.count("region_reads_.reset();") == 1
            and n.count("region_reads_.read_gate[") == 2 and n.count("region_reads_.window[") == 1
            and n.count("region_reads_.window_done(") == 1
            and n.count(RESET_REGIONS_SIG + " {") == 1 and ordered(reset, REGION_STATE_RESET, REGION_FLAGS_RESET)
            and RING_READY_RESET in body(n, RING_READY_SIG) and TEARDOWN_RESET in body(n, TEARDOWN_SIG)
            and n.count("reset_regions();") == 2)


def test_reuse_waits_for_window_readers():
    assert claim_reuse_waits_for_window_readers(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::stage_window (the read gate is "
        "searched in window_done, reset_regions and the region reads struct of moe-stream-plan.hpp)",
        body(norm(PHASE), STAGE_SIG),
        STAGE_CLOSE, REUSE_COND, ITEM_LOOP, COPY_DEPS, READY_DEP, REUSE_TAKE, REUSE_DEP, COPY_DECL, REUSE_RECORD,
        signature=STAGE_SIG)


def test_reuse_mutants():
    c = claim_reuse_waits_for_window_readers
    # The gate taken but not waited on.
    assert not c(mutate(PHASE, REUSE_DEP, ""))
    # The fallback barrier on the copy queue, which the readers do not run on.
    assert not c(mutate(PHASE, ": ccs_->ext_oneapi_submit_barrier();",
                        ": cache.get_bcs_queue().ext_oneapi_submit_barrier();"))
    # The read gate never used: every refill waits for a stage-time barrier, which also waits for the work queued after
    # the last fill's dequants.
    assert not c(mutate(PHASE, "region_reads_.read_gate[cur_region_] ? *region_reads_.read_gate[cur_region_] : ccs_",
                        "ccs_"))
    # The other region's read gate.
    assert not c(mutate(PHASE, "region_reads_.read_gate[cur_region_] ? *region_reads_.read_gate[cur_region_] :",
                        "region_reads_.read_gate[1 - cur_region_] ? *region_reads_.read_gate[1 - cur_region_] :"))
    # A refill that never waits for readers.
    assert not c(mutate(PHASE, REUSE_COND, "const bool refill_after_readers = false;"))
    # The region's last window read before the close turns the ring to it: the window in use's region would decide.
    early = in_body(PHASE, STAGE_SIG, REUSE_COND + " ", "")
    early = in_body(early, STAGE_SIG, STAGE_CLOSE, REUSE_COND + " " + STAGE_CLOSE)
    assert not c(early)
    # The last fill recorded when nothing was copied in.
    assert not c(mutate(PHASE, REUSE_RECORD, REUSE_RECORD.replace("if (n.staged != 0) { ", "{ ")))
    # Never recorded.
    assert not c(mutate(PHASE, REUSE_RECORD, ""))
    # Recorded under another window's id.
    assert not c(mutate(PHASE, "region_reads_.refill(cur_region_, window_id);",
                        "region_reads_.refill(cur_region_, window_id + 1);"))
    # A refilled region keeps the read gate of its previous window, which precedes the new window's readers.
    assert not c(PHASE, mutate(PLAN_HPP, " read_gate[region].reset(); }", " }"))
    # The region's window never recorded, so window_done finds no region to gate.
    assert not c(PHASE, mutate(PLAN_HPP, "{ window[region] = w; ", "{ "))
    # window_done gates the region holding the next window.
    assert not c(PHASE, mutate(PLAN_HPP, "if (window[r] == w) {", "if (window[r] == w + 1) {"))
    # A window_done with no window gates every region no copy has gone into.
    assert not c(PHASE, mutate(PLAN_HPP, "r < 2 && w != UINT32_MAX;", "r < 2;"))
    # The phase's window_done keyed by another id than the one the stage refilled with.
    assert not c(mutate(PHASE, READ_GATE_RECORD,
                        "region_reads_.window_done(moe_stream_window_id(layer, role, 0), readers_done);"))
    # window_done records no read gate.
    assert not c(mutate(PHASE, READ_GATE_RECORD, ""))
    # A window of a layer this micro-batch has not begun records a gate.
    assert not c(mutate(PHASE, "if (layer != cur_layer_) { return; } " + READ_GATE_RECORD,
                        READ_GATE_RECORD + " if (layer != cur_layer_) { return; }"))
    # layer_done records the down op as a read gate, the layer-keyed shape: a region turns over several times a layer.
    assert not c(in_body(PHASE, PHASE_LAYER_DONE_SIG, "close_region(down_done);",
                         "close_region(down_done); region_reads_.read_gate[cur_region_] = down_done;"))
    # A second copy of the read gates in the phase beside the struct's.
    assert not c(mutate(PHASE, READ_GATE_RECORD, READ_GATE_RECORD + " region_reads_.read_gate[0] = readers_done;"))
    # The AOS-only flag back beside the window.
    assert not c(PHASE, mutate(PLAN_HPP, READS_FIELDS, READS_FIELDS + " bool aos_only[2] = { false, false };"))
    # A new ring inherits the old ring's read gates, or the windows its regions last held.
    assert not c(PHASE, mutate(PLAN_HPP, " read_gate[r].reset(); } }", " } }"))
    assert not c(PHASE, mutate(PLAN_HPP, "{ window[r] = UINT32_MAX; ", "{ "))
    # reset_regions no longer resets the regions' last fills.
    assert not c(mutate(PHASE, REGION_FLAGS_RESET, ""))
    # A new ring never forgets the old ring's regions.
    assert not c(in_body(PHASE, RING_READY_SIG, "reset_regions(); ", ""))
    # Teardown leaves the regions' windows and read gates behind for the next ring of a reused phase.
    assert not c(in_body(PHASE, TEARDOWN_SIG, " reset_regions();", ""))


PROMPT_KERNEL_SIG = ("static bool ggml_sycl_moe_stream_prompt_kernel(void * graph_ctx, ggml_type type, "
                     "ggml_layout_mode layout, int64_t ne0, int64_t ne1)")
PROMPT_KERNEL_QUERY = ("ggml_sycl_moe_query_route_capability(type, layout, moe_route_phase::PROMPT, ne0, ne1, "
                       "static_cast<size_t>(call->op.n_tokens), call->device, moe_layer_route_residency::DEVICE, "
                       "call->device);")
PROMPT_KERNEL_FILL = "call.op.prompt_kernel = ggml_sycl_moe_stream_prompt_kernel;"
PROMPT_KERNEL_RETURN = ("return ggml_sycl_moe_staged_capability_refusal(cap, type) == "
                        "ggml_sycl::moe_staged_route_refusal::NONE;")
INDEX_CALL = ("moe_stream_plan_ring_geometry(plan, device, op.prompt_kernel, op.graph_ctx, op.library_claims, "
              "op.window, &index_);")
INDEX_BUILD_CALL = "moe_stream_plan_index_build(plan, device, prompt_kernel, graph_ctx, library_claims, hints)"
PREPARE_SIG = "void moe_stream_phase::prepare("
INDEX_PROMPT_UNASKED = "GGML_UNUSED(prompt_kernel);"
# A call of the predicate itself, by the hook's name or through op.: the backend's definition carries a prefix.
PROMPT_KERNEL_CALL_RE = re.compile(r"(?<!\w)prompt_kernel\(")


def claim_staging_from_the_claim_alone(sycl: str, hpp: str, phase: str) -> bool:
    """W2 (Task 2b of llama.cpp-a2xb): what a matrix stages comes from the library executor's claim alone, and the SOA
    and route staging that the MMVQ kernels' answer used to pick is retired. The index build still takes the
    prompt-kernel predicate, whose signature the geometry helper shares with the auto-ubatch fit, but never asks it,
    and the phase asks it nowhere: the phase builds the index once per micro-batch, in prepare(), through the one
    geometry helper, with the op's hooks and window. The backend still fills the predicate from
    ggml_sycl_moe_query_route_capability, asked for the call's own device, with no table of its own."""
    n = norm(sycl)
    # The forward declaration ends in ";" and the definition in " {": the body is the definition's.
    at = n.find(PROMPT_KERNEL_SIG + " {")
    pk = body(n[at:], PROMPT_KERNEL_SIG) if at >= 0 else ""
    h = norm(hpp)
    p = norm(phase)
    return (n.count(PROMPT_KERNEL_SIG + " {") == 1 and PROMPT_KERNEL_QUERY in pk and PROMPT_KERNEL_RETURN in pk
            and "moe_mmvq_" not in pk and "soa_reorder" not in pk and n.count(PROMPT_KERNEL_FILL) == 1
            and body(h, INDEX_BUILD_SIG).startswith("{ " + INDEX_PROMPT_UNASKED)
            and not PROMPT_KERNEL_CALL_RE.search(h) and not PROMPT_KERNEL_CALL_RE.search(p)
            and INDEX_CALL in body(p, PREPARE_SIG)
            and p.count("moe_stream_plan_ring_geometry(") == 1 and p.count("moe_stream_plan_index_build(") == 0
            and INDEX_BUILD_CALL in body(h, RING_GEOMETRY_SIG))


def test_staging_from_the_claim_alone():
    assert claim_staging_from_the_claim_alone(SYCL, PLAN_HPP, PHASE), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the prompt kernel body is searched in it; the index "
        "build is searched in moe-stream-plan.hpp and the prepare call in moe-stream-plan.cpp)",
        norm(SYCL),
        PROMPT_KERNEL_SIG + " {", PROMPT_KERNEL_QUERY, PROMPT_KERNEL_RETURN, PROMPT_KERNEL_FILL)


def test_staging_source_mutants():
    c = claim_staging_from_the_claim_alone
    # The index build asks the kernels again.
    assert not c(SYCL, mutate(PLAN_HPP, INDEX_PROMPT_UNASKED,
                              INDEX_PROMPT_UNASKED + " (void) prompt_kernel(graph_ctx, GGML_TYPE_Q4_0, "
                                                     "GGML_LAYOUT_SOA, 0, 0);"), PHASE)
    # The phase asks them at every window matrix.
    asked = TYPED_GUARD + " (void) op.prompt_kernel(op.graph_ctx, item.type, GGML_LAYOUT_AOS, item.ncols, item.nrows);"
    assert not c(SYCL, PLAN_HPP, mutate(PHASE, TYPED_GUARD, asked))
    # A second table answering what the device reads.
    assert not c(mutate(SYCL, PROMPT_KERNEL_QUERY,
                        PROMPT_KERNEL_QUERY + " (void) moe_mmvq_capability_supports_layout(type, layout);"),
                 PLAN_HPP, PHASE)
    # Asked for the decode phase.
    assert not c(mutate(SYCL, "type, layout, moe_route_phase::PROMPT, ne0, ne1,",
                        "type, layout, moe_route_phase::DECODE, ne0, ne1,"), PLAN_HPP, PHASE)
    # The backend leaves the predicate unset.
    assert not c(mutate(SYCL, PROMPT_KERNEL_FILL, ""), PLAN_HPP, PHASE)
    # The index built without the op's hooks.
    assert not c(SYCL, PLAN_HPP,
                 mutate(PHASE, INDEX_CALL,
                        "moe_stream_plan_ring_geometry(plan, device, nullptr, nullptr, op.library_claims, op.window, "
                        "&index_);"))
    # prepare() stops passing the op's window: the geometry would size no ring, whatever the op carries.
    assert not c(SYCL, PLAN_HPP, mutate(PHASE, "op.library_claims, op.window, &index_);",
                                        "op.library_claims, moe_stream_window_hints(), &index_);"))
    # The shared geometry helper builds the index blind or without the claims hook, whatever prepare() passes.
    assert not c(SYCL, mutate(PLAN_HPP, INDEX_BUILD_CALL, "moe_stream_plan_index_build(plan, device, nullptr, nullptr, "
                                                          "library_claims, hints)"), PHASE)
    assert not c(SYCL, mutate(PLAN_HPP, INDEX_BUILD_CALL, "moe_stream_plan_index_build(plan, device, prompt_kernel, "
                                                          "graph_ctx, nullptr, hints)"), PHASE)


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
    """llama.cpp-o9uw: the prompt-kernel predicate and the staged-route resolver ask one helper, which admits a
    supported MMVQ_COMPAT kernel and nothing else, and refuses a type that runs only through its direct kernel. Which
    types those are is one predicate, the one the recipe validator and the executor choice ask, and neither of those
    spells the types out. Since W2 the ring stages from the library claim alone and never asks the prompt-kernel
    predicate; this pin keeps the still-wired predicate on the resolver's helper, so the two cannot drift apart. The
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
    assert claim_staging_mirrors_serving(SYCL, BATCH_HPP), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the capability refusal and prompt kernel bodies are "
        "searched in it; the direct-kernel helpers are searched in moe-resolved-batch.hpp)",
        norm(SYCL),
        STAGED_CAP_SIG + " {", *STAGED_CAP_STEPS, PROMPT_KERNEL_RETURN, ROUTE_CAP_ASK)


def test_staging_mirrors_serving_mutants():
    c = claim_staging_mirrors_serving
    # The predicate answering from the capability alone again, so it drifts from the resolver's helper on Q1_0/NVFP4.
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
LIBRARY_CLAIMS_QUERY = "ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor(*call->plan, w, call->device);"
LIBRARY_CLAIMS_GUARD = "return call != nullptr && call->plan && w != nullptr &&"
CLAIMS_PLAN_SIG = ("static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_sycl::placement_plan * plan, "
                   "const ggml_tensor * weight, int device, bool host_staged)")
CLAIMS_PLAN_LAYOUT = ("planned = plan != nullptr ? "
                      "ggml_sycl_moe_tensor_plan_primary_layout(*plan, weight, device, &planned_layout) : "
                      "ggml_sycl_moe_tensor_plan_primary_layout(weight, device, &planned_layout);")
CLAIMS_STAGED_LAYOUT = "ggml_sycl_moe_tensor_plan_staged_layout(*plan, weight, device, &planned_layout);"
CLAIMS_ENTRY = ("static bool ggml_sycl_moe_pp_onednn_batched_claims_tensor(const ggml_tensor * weight, int device) { "
                "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight, device, false); }")
STAGED_CLAIM_ENTRY = ("static bool ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor("
                      "const ggml_sycl::placement_plan & plan, const ggml_tensor * weight, int device) { "
                      "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(&plan, weight, device, true); }")
STAGED_PLAN_SIG = ("static bool ggml_sycl_moe_tensor_plan_staged_layout(const ggml_sycl::placement_plan & plan, "
                   "const ggml_tensor * src0, int device, layout_mode * out_layout)")
PRIMARY_PLAN_SIG = ("static bool ggml_sycl_moe_tensor_plan_primary_layout(const ggml_sycl::placement_plan & plan, "
                    "const ggml_tensor * src0, int device, layout_mode * out_layout)")
PRIMARY_ENTRY_TAIL = "return ggml_sycl_moe_tensor_plan_primary_layout(*plan_owner, src0, device, out_layout); }"
PLAN_READERS = ("ggml_sycl_cache_plan_owner(", "coherent_placement_plan_owner(", "coherent_cache_placement_plan_owner(",
                "global_placement_plan_owner(", "ggml_sycl_global_plan_owner(", "get_unified_cache_for_device(")
LIBRARY_CLAIMS_FILL = "call.op.library_claims = ggml_sycl_moe_stream_library_claims;"
ROUTE_CONSUMER_REFUSAL = ("if (s->consumer == moe_stream_consumer::EXECUTOR) { "
                          "return moe_staged_route_refusal::CLAIMED; }")
LAYER_WEIGHT_SIG = ("static const ggml_tensor * ggml_sycl_moe_stream_layer_weight(int layer, "
                    "ggml_sycl::expert_tensor_role role)")
ITEM_DEFAULT_NONE = "moe_stream_staging staging = moe_stream_staging::NONE;"
INDEX_CLAIM_GUARD = "if (library_claims != nullptr && moe_stream_item_aos_fits(item, SIZE_MAX)) {"
INDEX_CLAIM_ASK = "c = claims.emplace(ck, library_claims(graph_ctx, item.layer, item.role)).first;"
INDEX_CLAIM_APPLY = ("if (c->second) { item.staging = moe_stream_staging::AOS_ONLY; "
                     "item.consumer = moe_stream_consumer::EXECUTOR; }")
INDEX_SLOT_SKIP = "if (item.staging == moe_stream_staging::NONE) { continue; }"


def claim_claimed_weight_stages_for_the_executor(sycl: str, hpp: str, phase: str) -> bool:
    """One fact, one source (llama.cpp-3460, resolving llama.cpp-v7np; W2 of llama.cpp-a2xb): a weight the batched
    prompt executor claims once its host experts are staged stages each host matrix into one AOS slot, read by the
    executor's dequant kernel, and the item names the executor as its consumer, which the staged route refuses such an
    entry by. Every other matrix stays NONE, the item's default: the claim is the only assignment of an item's staging,
    so the ring sizes and stages nothing else. The backend answers the build's library-claims predicate with the staged
    claim, for the call's device and the graph's own weight tensor of (layer, role); the build asks it for every matrix
    one AOS slot holds, before the slot accounting, so a claimed matrix takes one slot; and the phase asks it only
    through the index build."""
    n = norm(sycl)
    at = n.find(LIBRARY_CLAIMS_SIG + " {")
    lc = body(n[at:], LIBRARY_CLAIMS_SIG) if at >= 0 else ""
    h = norm(hpp)
    build = body(h, INDEX_BUILD_SIG)
    p = norm(phase)
    return (n.count(LIBRARY_CLAIMS_SIG + " {") == 1 and LIBRARY_CLAIMS_WEIGHT in lc and LIBRARY_CLAIMS_QUERY in lc
            and "MXFP4" not in lc and "moe_classify_tensor" not in lc
            and "g_moe_gate_up_pairs.find(layer)" in body(n, LAYER_WEIGHT_SIG)
            and n.count(LIBRARY_CLAIMS_FILL) == 1 and n.count(ROUTE_CONSUMER_REFUSAL) == 1
            and "moe_stream_library_claims_fn library_claims, const moe_stream_window_hints & hints)" in h
            and ITEM_DEFAULT_NONE in body(h, "struct moe_stream_item")
            and ordered(build, INDEX_CLAIM_GUARD, INDEX_CLAIM_ASK, INDEX_CLAIM_APPLY, "li.items.push_back(item);",
                        INDEX_SLOT_SKIP, "++li.n_streamable;", REGION_SIZE)
            and build.count("item.staging = ") == 1
            and h.count("library_claims(graph_ctx") == 1
            and INDEX_CALL in body(p, PREPARE_SIG) and p.count("library_claims(") == 0)


def test_claimed_weight_stages_for_the_executor():
    assert claim_claimed_weight_stages_for_the_executor(SYCL, PLAN_HPP, PHASE), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the library claims body is searched in it; the index "
        "build is searched in moe-stream-plan.hpp)",
        norm(SYCL),
        LIBRARY_CLAIMS_SIG + " {", LIBRARY_CLAIMS_WEIGHT, LIBRARY_CLAIMS_QUERY, LIBRARY_CLAIMS_FILL,
        ROUTE_CONSUMER_REFUSAL)


def test_library_claim_mutants():
    c = claim_claimed_weight_stages_for_the_executor
    # The backend answers from something other than the staged claim.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_QUERY, "ggml_sycl_moe_pp_woq_enabled();"), PLAN_HPP, PHASE)
    # Asked for another device than the call's.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_QUERY,
                        "ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor(*call->plan, w, 0);"), PLAN_HPP, PHASE)
    # A second table of its own beside the claim. The host lease takes its weight with the same line, so the mutant is
    # made inside the definition.
    n = norm(SYCL)
    at = n.find(LIBRARY_CLAIMS_SIG + " {")
    assert at >= 0 and LIBRARY_CLAIMS_WEIGHT in n[at:], "mutant precondition: the claims definition takes its weight"
    second = n[:at] + n[at:].replace(
        LIBRARY_CLAIMS_WEIGHT, LIBRARY_CLAIMS_WEIGHT + " if (w && w->type != GGML_TYPE_MXFP4) { return false; }", 1)
    assert not c(second, PLAN_HPP, PHASE)
    # The backend leaves the predicate unset: no weight would ever stage for the executor.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_FILL, ""), PLAN_HPP, PHASE)
    # The route serves the executor's entries, so a library-free reader is no longer the only one it hands them to.
    assert not c(mutate(SYCL, ROUTE_CONSUMER_REFUSAL, ""), PLAN_HPP, PHASE)
    # The build asks and ignores the answer.
    assert not c(SYCL, mutate(PLAN_HPP, INDEX_CLAIM_APPLY, "if (c->second) { }"), PHASE)
    # A claimed matrix whose consumer stays the route's: the route would refuse nothing and serve it.
    assert not c(SYCL, mutate(PLAN_HPP, INDEX_CLAIM_APPLY,
                              "if (c->second) { item.staging = moe_stream_staging::AOS_ONLY; }"), PHASE)
    # A matrix the executor does not claim staged anyway, by default or by a second answer.
    assert not c(SYCL, mutate(PLAN_HPP, ITEM_DEFAULT_NONE,
                              "moe_stream_staging staging = moe_stream_staging::AOS_ONLY;"), PHASE)
    assert not c(SYCL, mutate(PLAN_HPP, "li.items.push_back(item);",
                              "if (item.staging == moe_stream_staging::NONE) { item.staging = "
                              "moe_stream_staging::AOS_ONLY; } li.items.push_back(item);"), PHASE)
    # Asked only where something else staged the matrix first (the llama.cpp-v7np shape): the executor's own table is
    # not the authority.
    assert not c(SYCL, mutate(PLAN_HPP, INDEX_CLAIM_GUARD,
                              "if (item.staging != moe_stream_staging::NONE && library_claims != nullptr) {"), PHASE)
    # The claim applied after the slot accounting, so a claimed matrix would take no slot.
    late = mutate(PLAN_HPP, INDEX_CLAIM_APPLY, "")
    late = mutate(late, "++li.n_streamable;", "++li.n_streamable; " + INDEX_CLAIM_APPLY)
    assert not c(SYCL, late, PHASE)
    # The phase builds its index without the claims.
    assert not c(SYCL, PLAN_HPP, mutate(PHASE, INDEX_CALL, INDEX_CALL.replace("op.library_claims", "nullptr")))
    # The phase asks again at every window matrix.
    asked_again = TYPED_GUARD + " (void) op.library_claims(op.graph_ctx, item.layer, item.role);"
    assert not c(SYCL, PLAN_HPP, mutate(PHASE, TYPED_GUARD, asked_again))


def claim_library_claims_read_the_index_plan(sycl: str) -> bool:
    """Lead ruling c-gi44, one fact one source: the library-claim predicate answers from call.plan, the plan the index
    build that asks it holds, through the staged claim (llama.cpp-3460), and asks only when the call holds a plan. The
    staged claim is the batched executor's own claim with its host_staged mode, which takes the planned layout from
    the plan it is given through the staged walk; neither plan-taking walk reads a plan of its own. Only the claim's
    two-argument entry point, the batched executor's own route, reads the device's plan: it passes no plan, and the
    resident walk's two-argument form reads the cache plan and delegates."""
    n = norm(sycl)
    at = n.find(LIBRARY_CLAIMS_SIG + " {")
    lc = body(n[at:], LIBRARY_CLAIMS_SIG) if at >= 0 else ""
    at = n.find(CLAIMS_PLAN_SIG + " {")
    cl = body(n[at:], CLAIMS_PLAN_SIG) if at >= 0 else ""
    at = n.find(PRIMARY_PLAN_SIG + " {")
    pl = body(n[at:], PRIMARY_PLAN_SIG) if at >= 0 else ""
    at = n.find(STAGED_PLAN_SIG + " {")
    sl = body(n[at:], STAGED_PLAN_SIG) if at >= 0 else ""
    return (ordered(lc, LIBRARY_CLAIMS_GUARD, LIBRARY_CLAIMS_QUERY)
            and lc.count("ggml_sycl_moe_pp_onednn_batched_claims") == 1
            and n.count(CLAIMS_PLAN_SIG + " {") == 1 and cl.count(CLAIMS_PLAN_LAYOUT) == 1
            and cl.count(CLAIMS_STAGED_LAYOUT) == 1 and cl.count("ggml_sycl_moe_tensor_plan_primary_layout(") == 2
            and n.count(CLAIMS_ENTRY) == 1 and n.count(STAGED_CLAIM_ENTRY) == 1
            and n.count(PRIMARY_PLAN_SIG + " {") == 1 and "plan.all_experts_on_device(" in pl
            and not any(r in pl for r in PLAN_READERS) and n.count(PRIMARY_ENTRY_TAIL) == 1
            and n.count(STAGED_PLAN_SIG + " {") == 1 and sl != "" and not any(r in sl for r in PLAN_READERS))


def test_library_claims_read_the_index_plan():
    assert claim_library_claims_read_the_index_plan(SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the library claims, claims-tensor, primary-layout and "
        "staged-layout bodies are searched in it)",
        norm(SYCL),
        LIBRARY_CLAIMS_SIG + " {", LIBRARY_CLAIMS_GUARD, LIBRARY_CLAIMS_QUERY, CLAIMS_PLAN_SIG + " {",
        CLAIMS_PLAN_LAYOUT, CLAIMS_STAGED_LAYOUT, CLAIMS_ENTRY, STAGED_CLAIM_ENTRY, PRIMARY_PLAN_SIG + " {",
        PRIMARY_ENTRY_TAIL, STAGED_PLAN_SIG + " {")


def test_library_claims_index_plan_mutants():
    c = claim_library_claims_read_the_index_plan
    # The predicate reads the device's plan again through the two-argument entry point.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_QUERY, "ggml_sycl_moe_pp_onednn_batched_claims_tensor(w, call->device);"))
    # The predicate passes no plan, so the claim reads the device's plan instead of the index build's.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_QUERY,
                        "ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, w, call->device, true);"))
    # Asked without a plan in hand.
    assert not c(mutate(SYCL, LIBRARY_CLAIMS_GUARD, "return call != nullptr && w != nullptr &&"))
    # The resident claim ignores the plan it is given.
    assert not c(mutate(SYCL, CLAIMS_PLAN_LAYOUT,
                        "planned = ggml_sycl_moe_tensor_plan_primary_layout(weight, device, &planned_layout);"))
    # The staged claim walks the resident plan instead (the shape that never claims a weight with a host expert).
    assert not c(mutate(SYCL, CLAIMS_STAGED_LAYOUT,
                        "ggml_sycl_moe_tensor_plan_primary_layout(*plan, weight, device, &planned_layout);"))
    # The plan-taking walk reads the device's plan behind its caller's back.
    assert not c(mutate(SYCL, "if (!plan.all_experts_on_device(src0->name, n_experts, device)) {",
                        "(void) ggml_sycl_cache_plan_owner(ggml_sycl::get_unified_cache_for_device(device)); "
                        "if (!plan.all_experts_on_device(src0->name, n_experts, device)) {"))
    # The staged walk reads the device's plan behind its caller's back.
    n = norm(SYCL)
    at = n.find(STAGED_PLAN_SIG + " {")
    assert at >= 0, "mutant precondition: the staged walk is defined"
    head = STAGED_PLAN_SIG + " {"
    assert not c(n[:at] + head + " (void) ggml_sycl_cache_plan_owner(ggml_sycl::get_unified_cache_for_device(device));"
                 + n[at + len(head):])
    # The executor's own route handed a plan read ahead of its type and role checks.
    assert not c(mutate(SYCL, "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(nullptr, weight, device, false); }",
                        "return ggml_sycl_moe_pp_onednn_batched_claims_tensor(ggml_sycl_cache_plan_owner("
                        "ggml_sycl::get_unified_cache_for_device(device)).get(), weight, device, false); }"))


# ---- the policy gate -------------------------------------------------------------------------------------------------

STAGE_SIG = "moe_stream_phase::stage_counts moe_stream_phase::stage_window("
RING_READY_SIG = "bool moe_stream_phase::ring_ready()"
EXECUTOR_CLAIMS_SIG = "bool moe_stream_phase::executor_claims(int layer)"
EXECUTOR_CLAIMS_BODY = ("{ const auto it = index_.layers.find(layer); return it != index_.layers.end() && "
                        "decision_of(layer).choice == moe_exec_choice::RING && "
                        "moe_stream_executor_claims(it->second, geo_) && ring_ready(); }")
WINDOW_GUARD = "if (req.layer != cur_layer_ || !executor_claims(req.layer)) { GGML_ABORT("
# begin_layer keeps a RING decision only where the executor claims the layer and the ring reserves.
RING_STANDS = ("if (moe_stream_stage_now(st.decision, op.policy) && !(moe_stream_executor_claims(cur->second, geo_) && "
               "ring_ready())) { st.decision.choice = moe_exec_choice::CPU; "
               "st.decision.reason = moe_exec_reason::RING_UNAVAILABLE; }")
STAGE_PRED_SIG = "static inline bool moe_stream_stage_now(const moe_exec_decision & d, moe_exec_policy policy)"
STAGE_PRED_BODY = "{ return d.choice == moe_exec_choice::RING && policy != moe_exec_policy::CPU; }"
CLAIMS_PRED_SIG = ("static inline bool moe_stream_executor_claims(const moe_stream_layer_info & li, "
                   "const moe_stream_ring_geometry & geo)")
CLAIMS_PRED_BODY = "{ return geo.n_slots != 0 && moe_stream_layer_claimed(li); }"
LAYER_CLAIMED_SIG = "static inline bool moe_stream_layer_claimed(const moe_stream_layer_info & li)"
LAYER_CLAIMED_STEPS = (
    "if (li.n_untriplet != 0 || li.items.empty()) { return false; }",
    "if (item.staging == moe_stream_staging::NONE || item.consumer != moe_stream_consumer::EXECUTOR) { return false; }",
    "return true;",
)
POLICY_FROM_CONFIG = "call.op.policy = ggml_sycl_moe_exec_config().policy;"
STAGE_WINDOW_CALL_RE = re.compile(r"(?<![:\w])stage_window\(")
RING_READY_CALL_RE = re.compile(r"(?<![:\w])ring_ready\(\)")


def claim_staging_follows_the_decision(phase: str, sycl: str, hpp: str) -> bool:
    """The ring stages for the executor only, on a layer whose RING decision stands (llama.cpp-3460; W2 of
    llama.cpp-a2xb). The phase's one stage_window() call is in window_begin, whose first statement aborts unless
    executor_claims answers true for the layer; executor_claims needs the RING choice, the layer's claim (every host
    matrix staged for the executor, no untriplet item, a ring geometry) and a reserved ring. begin_layer rewrites a RING
    decision that moe_stream_stage_now would act on (the RING choice under any policy but CPU: AUTO when the cost model
    chooses the ring, ring forcing it in moe_exec_weigh) to CPU, RING_UNAVAILABLE, unless the same claim holds and the
    ring reserves. The reserve is reachable only through ring_ready(), which only those two ask, and the op's policy is
    the parsed config's."""
    n = norm(phase)
    h = norm(hpp)
    wb = body(n, WINDOW_BEGIN_SIG)
    return (wb.startswith("{ " + WINDOW_GUARD) and len(STAGE_WINDOW_CALL_RE.findall(n)) == 1
            and len(STAGE_WINDOW_CALL_RE.findall(wb)) == 1
            and body(n, EXECUTOR_CLAIMS_SIG) == EXECUTOR_CLAIMS_BODY and RING_STANDS in body(n, BEGIN_LAYER_SIG)
            and body(h, STAGE_PRED_SIG) == STAGE_PRED_BODY and body(h, CLAIMS_PRED_SIG) == CLAIMS_PRED_BODY
            and ordered(body(h, LAYER_CLAIMED_SIG), *LAYER_CLAIMED_STEPS)
            and len(RING_READY_CALL_RE.findall(n)) == 2
            and n.count("->reserve(") == 1 and "->reserve(" in body(n, RING_READY_SIG)
            and norm(sycl).count(POLICY_FROM_CONFIG) == 1)


def staging_follows_the_decision_message(phase: str) -> str:
    return first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::window_begin (the executor claim, "
        "the predicates in moe-stream-plan.hpp and the policy in ggml-sycl.cpp are searched too)",
        body(norm(phase), WINDOW_BEGIN_SIG),
        "{ " + WINDOW_GUARD,
        signature=WINDOW_BEGIN_SIG)


def test_staging_follows_the_decision():
    assert claim_staging_follows_the_decision(PHASE, SYCL, PLAN_HPP), staging_follows_the_decision_message(PHASE)


def test_policy_gate_mutants():
    c = claim_staging_follows_the_decision
    # window_begin stages for a layer nobody asked executor_claims about.
    assert not c(mutate(PHASE, " || !executor_claims(req.layer)", ""), SYCL, PLAN_HPP)
    # executor_claims without the RING choice: a CPU layer would stage.
    assert not c(mutate(PHASE, "decision_of(layer).choice == moe_exec_choice::RING && ", ""), SYCL, PLAN_HPP)
    # executor_claims without the layer's claim.
    assert not c(mutate(PHASE, "moe_stream_executor_claims(it->second, geo_) && ring_ready(); }", "ring_ready(); }"),
                 SYCL, PLAN_HPP)
    # begin_layer keeps RING on a layer the executor does not claim.
    assert not c(mutate(PHASE, "!(moe_stream_executor_claims(cur->second, geo_) && ring_ready())", "!ring_ready()"),
                 SYCL, PLAN_HPP)
    # AUTO never staging: the record-only predicate of lead ruling c-3skf.
    assert not c(PHASE, SYCL, mutate(PLAN_HPP, STAGE_PRED_BODY,
                                     "{ return d.choice == moe_exec_choice::RING && "
                                     "policy == moe_exec_policy::RING; }"))
    # CPU staging: the predicate tests the choice alone.
    assert not c(PHASE, SYCL, mutate(PLAN_HPP, STAGE_PRED_BODY, "{ return d.choice == moe_exec_choice::RING; }"))
    # The claim without a ring geometry, or with an unclaimed or untriplet matrix in the layer.
    assert not c(PHASE, SYCL, mutate(PLAN_HPP, CLAIMS_PRED_BODY, "{ return moe_stream_layer_claimed(li); }"))
    assert not c(PHASE, SYCL, mutate(PLAN_HPP, "item.staging == moe_stream_staging::NONE || item.consumer",
                                     "item.consumer"))
    assert not c(PHASE, SYCL, mutate(PLAN_HPP, "li.n_untriplet != 0 || li.items.empty()", "li.items.empty()"))
    # A second stage_window call outside window_begin.
    assert not c(mutate(PHASE, "report(layer, st);", "(void) stage_window({}, 0, cache, op); report(layer, st);"),
                 SYCL, PLAN_HPP)
    # The ring reserved outside the two askers.
    assert not c(mutate(PHASE, "in_order_ = ccs.is_in_order();", "in_order_ = ccs.is_in_order(); ring_ready();"),
                 SYCL, PLAN_HPP)
    # The op's policy pinned to RING instead of read from GGML_SYCL_MOE_EXEC.
    assert not c(PHASE, mutate(SYCL, POLICY_FROM_CONFIG, "call.op.policy = moe_exec_policy::RING;"), PLAN_HPP)


# ---- a matrix the ring cannot take reaches the executor as a decline, not an abort (llama.cpp-9433) -----------------

NO_LEASE_SKIP = ("if (!host.valid() || host_bytes < item.src_bytes || "
                 "!(host.owner_metadata(&owner) && moe_stream_owner_streamable(owner, device))) { continue; }")
PREFETCHED_BRANCH = "if (wid == next_window_id_) { if (next_window_whole_ && missing != 0) { GGML_ABORT("
WHOLE_SET = "next_window_whole_ = n.staged == static_cast<int>(nitems.size());"
RESTAGE_UNPREFETCHED = "} else if (missing != 0) { stage_or_abort(items, req.role, req.phase); }"
STAGED_COUNT = "staged += find(item.layer, item.expert_id, item.role) != nullptr ? 1u : 0u;"


def claim_window_decline_reaches_the_executor(phase: str) -> bool:
    """A matrix the ring cannot take directly (no lease, a short lease, or an owner it cannot copy from) is skipped by
    stage_window before the free-slot check, so it is neither an abort nor an exhausted slot, and is never copied
    another way. window_begin re-stages only a window no prefetch took, aborts on a prefetched one only when that
    prefetch took the whole window, and returns the count the table holds, so the executor finds no entry for the
    skipped expert and declines it. window_begin's three aborts are the claim guard, a window larger than its region
    and entries lost after a whole prefetch. No host test reaches window_begin (it needs a device cache), so this pin
    is the decline path's witness."""
    n = norm(phase)
    wb = body(n, WINDOW_BEGIN_SIG)
    sw = body(n, STAGE_SIG)
    return (ordered(sw, NO_LEASE_SKIP, SLOTS_CHECK, AOS_ACQUIRE) and wb.count("GGML_ABORT(") == 3
            and ordered(wb, PREFETCHED_BRANCH, RESTAGE_UNPREFETCHED, WHOLE_SET, STAGED_COUNT)
            and wb.endswith("return staged; }"))


def test_window_decline_reaches_the_executor():
    assert claim_window_decline_reaches_the_executor(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::window_begin (the no-lease skip is "
        "searched in stage_window)",
        body(norm(PHASE), WINDOW_BEGIN_SIG),
        PREFETCHED_BRANCH, RESTAGE_UNPREFETCHED, WHOLE_SET, STAGED_COUNT,
        signature=WINDOW_BEGIN_SIG)


def test_window_decline_mutants():
    c = claim_window_decline_reaches_the_executor
    # A lease the ring cannot copy aborts the prompt, or counts as an exhausted slot (which window_begin aborts on).
    assert not c(mutate(PHASE, NO_LEASE_SKIP, NO_LEASE_SKIP.replace("{ continue; }", '{ GGML_ABORT("no lease"); }')))
    assert not c(mutate(PHASE, NO_LEASE_SKIP, NO_LEASE_SKIP.replace("{ continue; }", "{ ++n.exhausted; continue; }")))
    # A prefetched window aborts on any missing entry, so a skipped matrix kills the prompt.
    assert not c(mutate(PHASE, "if (next_window_whole_ && missing != 0) {", "if (missing != 0) {"))
    # The prefetch reported whole whatever it took.
    assert not c(mutate(PHASE, WHOLE_SET, "next_window_whole_ = true;"))
    # A prefetched window re-staged, asking the same lease again.
    assert not c(mutate(PHASE, "} else if (missing != 0) {", "} if (missing != 0) {"))
    # A fourth abort: a window short of its items.
    short = 'if (missing != 0) { GGML_ABORT("short"); } const bool exposed ='
    assert not c(mutate(PHASE, "const bool exposed =", short))
    # The window's size returned instead of what the table holds.
    assert not c(mutate(PHASE, "return staged; }", "return static_cast<uint32_t>(items.size()); }"))


# ---- a micro-batch starts with no window (llama.cpp-9433) ------------------------------------------------------------

UBATCH_RESETS = ("table_.clear();", "prev_window_layer_ = -1;", "next_window_id_ = UINT32_MAX;",
                 "next_window_whole_ = false;")


def claim_ubatch_resets_the_window_state(phase: str) -> bool:
    """begin_ubatch clears the table, then forgets the last window's layer and the prefetch. A micro-batch can start at
    the layer the last one ended on (moe_stream_ubatch_starts), so a window prefetched before the boundary has the same
    id as one asked after it. Kept, the id would take the cleared window as prefetched and skip staging it (the executor
    then declines it), and the whole flag would abort on the entries the clear dropped."""
    return ordered(body(norm(phase), BEGIN_UBATCH_SIG), *UBATCH_RESETS)


def test_ubatch_resets_the_window_state():
    assert claim_ubatch_resets_the_window_state(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::begin_ubatch",
        body(norm(PHASE), BEGIN_UBATCH_SIG),
        *UBATCH_RESETS,
        signature=BEGIN_UBATCH_SIG)


def test_ubatch_reset_mutants():
    c = claim_ubatch_resets_the_window_state
    # The prefetch id kept across the boundary.
    assert not c(mutate(PHASE, "next_window_id_ = UINT32_MAX;", ""))
    # The prefetch id and its whole flag kept.
    assert not c(mutate(PHASE, "next_window_id_ = UINT32_MAX; next_window_whole_ = false;", ""))
    # The last window's layer kept, so a micro-batch that starts at the layer the last one ended at reads its first
    # window as prefetched and counts its copies as misses (moe_stream_window_exposed names no role).
    assert not c(mutate(PHASE, "prev_window_layer_ = -1;", ""))


UBATCH_PRED_SIG = "static inline bool moe_stream_ubatch_starts("
EXPOSED_PRED_SIG = "static inline bool moe_stream_window_exposed("
WINDOW_ID_SIG = "static inline uint32_t moe_stream_window_id("
UBATCH_PRED_CALL = ("const bool new_ubatch = moe_stream_ubatch_starts(ubatch_live_, layer, cur_layer_, op.graph_seq, "
                    "cur_graph_seq_);")
EXPOSED_CALL = "const bool exposed = moe_stream_window_exposed(req.layer, prev_window_layer_);"
WINDOW_ID_CALL = "const uint32_t wid = moe_stream_window_id(req.layer, req.role, req.phase);"


def claim_phase_decides_through_tested_predicates(phase: str, hpp: str) -> bool:
    """begin_layer and window_begin decide through the header's pure predicates, which test-moe-stream-phase pins: each
    is defined once in moe-stream-plan.hpp and called, never restated, in the phase. A micro-batch starts per
    moe_stream_ubatch_starts, bytes may move per moe_stream_stage_now and moe_stream_executor_claims, a window is named
    by moe_stream_window_id, listed by moe_stream_window_items and counted exposed per moe_stream_window_exposed. No
    layer is resolved ahead: moe_stream_resolve_ahead is gone, and so is every regime test."""
    h = norm(hpp)
    n = norm(phase)
    b = body(n, BEGIN_LAYER_SIG)
    wb = body(n, WINDOW_BEGIN_SIG)
    sigs = (UBATCH_PRED_SIG, STAGE_PRED_SIG, CLAIMS_PRED_SIG, EXPOSED_PRED_SIG, WINDOW_ID_SIG, WINDOW_ITEMS_SIG)
    return (all(h.count(sig) == 1 for sig in sigs)
            and UBATCH_PRED_CALL in b and RING_STANDS in b and EXPOSED_CALL in wb and WINDOW_ID_CALL in wb
            and "!= cur_graph_seq_" not in n and "moe_exec_regime::" not in n and "4096u" not in n
            and "role == expert_tensor_role::UP" not in n and "resolve_ahead" not in h + n)


def test_phase_decides_through_tested_predicates():
    assert claim_phase_decides_through_tested_predicates(PHASE, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::begin_layer (the window id and "
        "exposure calls are searched in window_begin, the predicate signatures in moe-stream-plan.hpp)",
        body(norm(PHASE), BEGIN_LAYER_SIG),
        UBATCH_PRED_CALL, RING_STANDS,
        signature=BEGIN_LAYER_SIG)


def test_predicate_mutants():
    c = claim_phase_decides_through_tested_predicates
    # The micro-batch start restated inline.
    assert not c(mutate(PHASE, UBATCH_PRED_CALL, "const bool new_ubatch = !ubatch_live_ || layer < cur_layer_ || "
                                                 "(layer == cur_layer_ && op.graph_seq != cur_graph_seq_);"), PLAN_HPP)
    # The stage-now test restated inline.
    assert not c(mutate(PHASE, "moe_stream_stage_now(st.decision, op.policy) &&",
                        "st.decision.choice == moe_exec_choice::RING && op.policy == moe_exec_policy::RING &&"),
                 PLAN_HPP)
    # The exposure restated inline.
    assert not c(mutate(PHASE, EXPOSED_CALL, "const bool exposed = req.layer != prev_window_layer_ || "
                                             "(req.role == expert_tensor_role::UP && req.phase == 0);"), PLAN_HPP)
    # The window id restated inline.
    assert not c(mutate(PHASE, WINDOW_ID_CALL, "const uint32_t wid = static_cast<uint32_t>(req.layer) * 3u * 4096u + "
                                               "static_cast<uint32_t>(req.phase);"), PLAN_HPP)
    # A layer resolved ahead again.
    assert not c(PHASE, mutate(PLAN_HPP, UBATCH_PRED_SIG,
                               "static inline bool moe_stream_resolve_ahead(const moe_exec_decision & d) { return "
                               "d.choice == moe_exec_choice::RING; } " + UBATCH_PRED_SIG))
    # A predicate the test pins is gone from the header.
    assert not c(PHASE, mutate(PLAN_HPP, EXPOSED_PRED_SIG, "static inline bool moe_stream_window_hidden("))


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
    assert claim_phase_allocates_only_ring_slots(PHASE), allocates_only_ring_slots_message(PHASE)
    assert claim_staging_follows_the_decision(PHASE, SYCL, PLAN_HPP), staging_follows_the_decision_message(PHASE)
    # The seams outside their block: a second slot acquisition and a second ring_ready() call, both counted.
    unguarded = production(raw.replace(guard, "#if 1"))
    assert not claim_phase_allocates_only_ring_slots(unguarded)
    assert not claim_staging_follows_the_decision(unguarded, SYCL, PLAN_HPP)
    # An #else arm is production and stays; a nested conditional inside the block goes with it.
    assert production("a\n#if defined(GGML_SYCL_PRIVATE_TESTING)\nb\n#else\nc\n#endif\nd") == "a\n\n\n\nc\n\nd"
    assert production("#ifdef GGML_SYCL_PRIVATE_TESTING\n#ifdef X\nb\n#else\nc\n#endif\n#endif\nd") == "\n" * 7 + "d"


def test_reserve_failure_lasts_one_ubatch():
    assert claim_reserve_failure_lasts_one_ubatch(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::ring_ready (the latch clears are "
        "searched in begin_ubatch, end_prompt and begin_layer)",
        body(norm(PHASE), RING_READY_SIG),
        LATCH_GUARD, "->reserve(", LATCH_SET,
        signature=RING_READY_SIG)


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


# ---- a short reserve asks for a loan (llama.cpp-3460, c-0tmn) --------------------------------------------------------

RING_CPP_REL = "ggml/src/ggml-sycl/moe-stream-ring.cpp"
RING_HPP_REL = "ggml/src/ggml-sycl/moe-stream-ring.hpp"
RING_CPP = read(RING_CPP_REL)
RING_HPP = read(RING_HPP_REL)

# The reserve-failed branch's condition, without its brace: body() opens at the first brace after it.
RESERVE_SHORT_IF = "if (!ring->reserve(device_, geo_, &bytes_short))"
LOAN_CALL = ("moe_stream_loan_on_short_reserve(loan_latch_, device_, bytes_short, geo_.slot_bytes, request_loan_, "
             "loan_refusals_)")
LOAN_REFUSED_WARN = "== moe_stream_loan_step::REFUSED) { warn_loan_refused(bytes_short); }"
LOAN_LATCH_CLEAR = "loan_latch_ = moe_stream_loan_latch();"
LOAN_HOOKS_STORED = ("request_loan_ = op.request_loan;", "loan_refusals_ = op.loan_refusals;")
LOAN_HELPER_SIG = "static inline moe_stream_loan_step moe_stream_loan_on_short_reserve("
# The helper's steps, in order: nothing to ask, a refusal since the last ask latches, else ask with the shortfall.
LOAN_HELPER_ORDER = (
    "if (request_loan == nullptr || latch.refused || bytes_short == 0) { return moe_stream_loan_step::NONE; }",
    "const uint64_t refusals_now = loan_refusals != nullptr ? loan_refusals(device) : 0;",
    "if (latch.asked && refusals_now != latch.refusals_seen) { latch.refused = true; "
    "return moe_stream_loan_step::REFUSED; }",
    "latch.asked = true; latch.refusals_seen = refusals_now; request_loan(device, bytes_short, slot_bytes); "
    "return moe_stream_loan_step::REQUEST;",
)
# The hook's type carries the ring's slot size beside the bytes, so the loan is counted in slots (llama.cpp-9x8t).
LOAN_FN_TYPE = "using moe_stream_request_loan_fn = void (*)(int device, size_t bytes, size_t slot_bytes);"
RESERVE_DECL = "bool reserve(int device, const moe_stream_ring_geometry & geo, size_t * bytes_short = nullptr);"
RESERVE_DEF_SIG = ("bool moe_stream_ring::reserve(int device, const moe_stream_ring_geometry & geo, "
                   "size_t * bytes_short)")
RESERVE_SHORT_SET = ("if (bytes_short != nullptr) { "
                     "*bytes_short = static_cast<size_t>(geo.n_slots - held) * geo.slot_bytes; }")
WEIGHT_REFUSAL_WARN = "[MOE-STREAM-RING] reserve: the WEIGHT zone cannot give"
REFUSALS_DECL = "static std::atomic<uint64_t> g_sycl_expert_loan_refusals[GGML_SYCL_MAX_DEVICES];"
REFUSALS_BUMP = ("if (code == ggml_sycl_txn_result::REFUSED) { "
                 "g_sycl_expert_loan_refusals[device].fetch_add(1, std::memory_order_release); }")
RUN_PENDING_SIG = "static bool ggml_sycl_run_pending_expert_loan_on_device("
RUN_PENDING_TXN = ("ggml_sycl_expert_loan_transaction_run(device, bytes_wanted, slot_bytes, false, resolver, result, "
                   "false);")
RUN_PENDING_BUSY = "if (code != ggml_sycl_txn_result::BUSY) {"
REFUSALS_ACCESSOR_SIG = "static uint64_t ggml_sycl_expert_loan_refusals(int device)"
REFUSALS_LOAD = "return g_sycl_expert_loan_refusals[device].load(std::memory_order_acquire);"
STREAM_PREPARE_SIG = "static bool ggml_sycl_moe_stream_prepare("
WIRE_REQUEST = "call.op.request_loan = ggml_sycl_request_expert_loan;"
WIRE_REFUSALS = "call.op.loan_refusals = ggml_sycl_expert_loan_refusals;"


def claim_short_reserve_requests_a_loan(phase: str, hpp: str, sycl: str, ring_cpp: str, ring_hpp: str) -> bool:
    """A short ring reserve asks the backend for an expert loan (llama.cpp-3460, closing c-0tmn for production). The
    request appears exactly once in the phase, inside ring_ready()'s reserve-failed branch after the reserve latch (so
    once per micro-batch, c-i8rz), with the shortfall the reserve itself reported, never the whole ring's bytes. The
    hooks it calls are the op's, stored at begin_layer before the layer's reserve, and the backend wires them to
    ggml_sycl_request_expert_loan and to the boundary runner's REFUSED count. The pure helper asks only while no loan
    this prompt came back REFUSED: the runner counts a REFUSED outcome (never BUSY, which it re-posts itself), a count
    that moved since the last ask latches, and only end_prompt clears the latch. The reserve's shortfall is the bytes
    of the slots it still has to allocate (llama.cpp-am87), set after the slot allocation the zone refused."""
    n = norm(phase)
    ready = body(n, RING_READY_SIG)
    at = ready.find(RESERVE_SHORT_IF + " {")
    branch = body(ready[at:], RESERVE_SHORT_IF) if at >= 0 else ""
    h = norm(hpp)
    s = norm(sycl)
    rc = norm(ring_cpp)
    return (n.count("moe_stream_loan_on_short_reserve(") == 1
            and ordered(branch, LATCH_SET, LOAN_CALL, LOAN_REFUSED_WARN)
            and ordered(body(n, BEGIN_LAYER_SIG), *LOAN_HOOKS_STORED, "ring_ready()")
            and n.count(LOAN_LATCH_CLEAR) == 1 and LOAN_LATCH_CLEAR in body(n, END_PROMPT_SIG)
            and h.count("request_loan(device, ") == 1 and h.count(LOAN_FN_TYPE) == 1
            and ordered(body(h, LOAN_HELPER_SIG), *LOAN_HELPER_ORDER)
            and RESERVE_DECL in norm(ring_hpp)
            and ordered(body(rc, RESERVE_DEF_SIG), "allocation = unified_allocate_owner(req);", RESERVE_SHORT_SET,
                        WEIGHT_REFUSAL_WARN)
            and s.count(REFUSALS_DECL) == 1
            and s.count("g_sycl_expert_loan_refusals[device].fetch_add(") == 1
            and ordered(body(s, RUN_PENDING_SIG), RUN_PENDING_TXN, REFUSALS_BUMP, RUN_PENDING_BUSY)
            and REFUSALS_LOAD in body(s, REFUSALS_ACCESSOR_SIG)
            and s.count(WIRE_REQUEST) == 1 and WIRE_REQUEST in body(s, STREAM_PREPARE_SIG)
            and s.count(WIRE_REFUSALS) == 1 and WIRE_REFUSALS in body(s, STREAM_PREPARE_SIG))


def test_short_reserve_requests_a_loan():
    assert claim_short_reserve_requests_a_loan(PHASE, PLAN_HPP, SYCL, RING_CPP, RING_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::ring_ready (the loan hooks are "
        "searched in begin_layer, the helper in moe-stream-plan.hpp, the reserve in moe-stream-ring.cpp and "
        "the refusal counters in ggml-sycl.cpp)",
        body(norm(PHASE), RING_READY_SIG),
        RESERVE_SHORT_IF + " {", LATCH_SET, LOAN_CALL, LOAN_REFUSED_WARN,
        signature=RING_READY_SIG)


def test_short_reserve_loan_mutants():
    def c(phase=PHASE, hpp=PLAN_HPP, sycl=SYCL, ring_cpp=RING_CPP, ring_hpp=RING_HPP):
        return claim_short_reserve_requests_a_loan(phase, hpp, sycl, ring_cpp, ring_hpp)

    stmt = "if (" + LOAN_CALL + " " + LOAN_REFUSED_WARN
    # No request: the short reserve asks for nothing.
    assert not c(phase=mutate(PHASE, stmt, ""))
    # The request outside the reserve-failed branch: every ring_ready() call, or a reserve that succeeded, would ask.
    moved = mutate(PHASE, stmt, "").replace(RING_READY_RESET, stmt + " " + RING_READY_RESET, 1)
    assert not c(phase=moved)
    # The request with the whole ring's bytes instead of the shortfall.
    assert not c(phase=mutate(PHASE, "device_, bytes_short, geo_.slot_bytes, request_loan_",
                              "device_, geo_.bytes, geo_.slot_bytes, request_loan_"))
    # The request without the ring's slot: the loan would be counted in bytes, and a lent triplet whose matrices are
    # each below a slot would fund none (llama.cpp-9x8t).
    assert not c(phase=mutate(PHASE, "bytes_short, geo_.slot_bytes, request_loan_", "bytes_short, 0, request_loan_"))
    assert not c(hpp=mutate(PLAN_HPP, "request_loan(device, bytes_short, slot_bytes);",
                            "request_loan(device, bytes_short, 0);"))
    assert not c(hpp=mutate(PLAN_HPP, LOAN_FN_TYPE,
                            "using moe_stream_request_loan_fn = void (*)(int device, size_t bytes);"))
    # The boundary runner asks the transaction in bytes.
    assert not c(sycl=mutate(SYCL, RUN_PENDING_TXN,
                             RUN_PENDING_TXN.replace("bytes_wanted, slot_bytes,", "bytes_wanted, 0,")))
    # The reserve's shortfall not taken: the phase would have to recompute it.
    assert not c(phase=mutate(PHASE, RESERVE_SHORT_IF, "if (!ring->reserve(device_, geo_))"))
    # The hooks never stored: the request would go nowhere.
    assert not c(phase=mutate(PHASE, LOAN_HOOKS_STORED[0], ""))
    # The latch cleared per micro-batch: a REFUSED loan would be asked again in every micro-batch.
    assert not c(phase=mutate(PHASE, "reserve_failed_ = false; table_.clear();",
                              "reserve_failed_ = false; " + LOAN_LATCH_CLEAR + " table_.clear();"))
    # The latch never cleared: one refusal would silence every later prompt of the context.
    assert not c(phase=drop_in(PHASE, END_PROMPT_SIG, LOAN_LATCH_CLEAR))
    # The helper re-asks after a REFUSED outcome, or never latches one.
    assert not c(hpp=mutate(PLAN_HPP, "latch.refused || bytes_short == 0", "bytes_short == 0"))
    assert not c(hpp=mutate(PLAN_HPP, "latch.refused = true; return moe_stream_loan_step::REFUSED;",
                            "return moe_stream_loan_step::REFUSED;"))
    # The helper asks for the whole ring, or asks a second time.
    assert not c(hpp=mutate(PLAN_HPP, "request_loan(device, bytes_short, slot_bytes);",
                            "request_loan(device, SIZE_MAX, slot_bytes);"))
    assert not c(hpp=mutate(PLAN_HPP, "request_loan(device, bytes_short, slot_bytes);",
                            "request_loan(device, bytes_short, slot_bytes); "
                            "request_loan(device, bytes_short, slot_bytes);"))
    # The reserve exposes no shortfall, or exposes the whole ring.
    assert not c(ring_hpp=mutate(RING_HPP, RESERVE_DECL,
                                 "bool reserve(int device, const moe_stream_ring_geometry & geo);"))
    assert not c(ring_cpp=mutate(RING_CPP, RESERVE_SHORT_SET, ""))
    assert not c(ring_cpp=mutate(RING_CPP, "*bytes_short = static_cast<size_t>(geo.n_slots - held) * geo.slot_bytes;",
                                 "*bytes_short = geo.bytes;"))
    # The refusal counted on BUSY too (the runner re-posts BUSY itself), or never counted.
    assert not c(sycl=mutate(SYCL, "if (code == ggml_sycl_txn_result::REFUSED) {",
                             "if (code != ggml_sycl_txn_result::ACCEPTED) {"))
    assert not c(sycl=mutate(SYCL, REFUSALS_BUMP, ""))
    # The hooks not wired by the backend.
    assert not c(sycl=mutate(SYCL, WIRE_REQUEST, ""))
    assert not c(sycl=mutate(SYCL, WIRE_REFUSALS, ""))


# ---- the ring reserves slot by slot (llama.cpp-am87) -----------------------------------------------------------------

SLOT_LOOP = "while (slots_.size() < geo.n_slots)"
FALLBACK_CALL = "allocation = unified_allocate_owner(req);"
FALLBACK_IF = "if (!allocation)"
# The loop body's steps, in order: the WEIGHT zone, then the general entry point (inside the zone refusal's branch),
# a refusal stopping the loop, the slot wrapped, and a slot accepted only once it resolves on the device.
SLOT_LOOP_ORDER = (
    "allocation_result allocation = unified_cache_zone_allocate_owner(device, vram_zone_id::WEIGHT, geo.slot_bytes, "
    "MOE_STREAM_RING_SLOT_ALIGN);",
    FALLBACK_IF + " {",
    FALLBACK_CALL,
    "if (!allocation) { break; }",
    "mem_handle slot = mem_handle::from_owned_alloc(std::move(allocation.owner), MOE_STREAM_RING_SLOT_LAYOUT);",
    "const auto resolved = slot.resolve(device);",
    "if (!resolved.ptr || !resolved.on_device) {",
    "slots_.push_back(std::move(slot));",
)
# The fallback request's fields, each set once before the fallback call, in any order and wherever reserve() builds the
# request: the ring's intent is the rule, not the order the fields are written in.
FALLBACK_REQ = (
    "req.device = device;",
    "req.size = geo.slot_bytes;",
    "req.alignment = MOE_STREAM_RING_SLOT_ALIGN;",
    "req.intent.role = alloc_role::EXPERT_STAGING;",
    "req.intent.category = runtime_category::EXPERT_CACHE;",
    'req.intent.cohort_id = "moe_stream_ring";',
    "req.intent.constraints.must_device = true;",
    "req.intent.constraints.prefer_vram_zone = vram_zone_id::WEIGHT;",
    "req.intent.constraints.forbid_vram_zone_spill = true;",
)
SAME_RING = ("const bool same_ring = device == device_ && geo.slot_bytes == geo_.slot_bytes && "
             "geo.n_slots == geo_.n_slots && geo.bytes == geo_.bytes && geo.region_slots == geo_.region_slots;")
GIVE_BACK_IF = "if (!slots_.empty() && !same_ring)"
HELD = "const uint32_t held = static_cast<uint32_t>(slots_.size());"
REFUSED_IF = "if (held < geo.n_slots)"
HOLDING_SUFFIX = "(holding %u of %u slots)"
# Every use of slots_ in reserve(), each exactly once: the give-back test and its WARN, the capacity, the loop test,
# the unresolved-slot WARN, the append and the held count. A census rather than a list of drops: a drop of a held slot
# spelled any way (clear, assign, swap(slots_), std::exchange, an assignment) adds a use or replaces one of these.
SLOT_USES = (
    GIVE_BACK_IF,
    "device_, slots_.size(), geo_.n_slots,",
    "slots_.reserve(geo.n_slots);",
    SLOT_LOOP,
    "slots_.size(), geo.slot_bytes",
    "slots_.push_back(std::move(slot));",
    HELD,
)
VALID_DECL = "bool valid() const { return geo_.n_slots != 0 && slots_.size() == geo_.n_slots; }"
ACQUIRE_SIG = "mem_handle moe_stream_ring::acquire_slot(ggml_layout_mode layout)"
# What the single-allocation ring read to size its loan: the zone's largest free block.
SHORTFALL_NAMES = ("moe_stream_ring_weight_shortfall", "zone_free_figures", "largest_free")


def claim_ring_reserves_slot_by_slot(ring_cpp: str, ring_hpp: str) -> bool:
    """llama.cpp-am87: reserve() allocates the ring one slot at a time, each slot through
    unified_cache_zone_allocate_owner and then unified_allocate_owner with the ring's intent, so an expert loan's
    scattered room gives it. The two allocation calls exist once each, inside the loop. A refused slot breaks the loop,
    and the refused branch reports the slots still to allocate and WARNs how many it holds. reserve() names slots_ only
    in the uses SLOT_USES lists, so neither the loop nor the refused branch drops a held slot, and the refused branch
    names none; the one release() in reserve() gives back a partial reservation of another geometry or device, so the
    next reserve() of the same ring continues from the slots held. A partial reservation is not valid, and
    acquire_slot hands out nothing until it is. The ring reads no zone figure: the largest free block is gone."""
    rc = norm(ring_cpp)
    rsv = body(rc, RESERVE_DEF_SIG)
    loop = body(rsv, SLOT_LOOP)
    give_back = body(rsv, GIVE_BACK_IF)
    refused = body(rsv, REFUSED_IF)
    call_at = rsv.find(FALLBACK_CALL)
    return (bool(loop) and ordered(loop, *SLOT_LOOP_ORDER) and FALLBACK_CALL in body(loop, FALLBACK_IF)
            and all(rsv.count(f) == 1 and -1 < rsv.find(f) < call_at for f in FALLBACK_REQ)
            and rc.count("unified_cache_zone_allocate_owner(") == 1 and rc.count("unified_allocate_owner(") == 1
            and all(rsv.count(use) == 1 for use in SLOT_USES) and rsv.count("slots_") == len(SLOT_USES)
            and ordered(rsv, SAME_RING, GIVE_BACK_IF, "geo_ = geo;", "device_ = device;", SLOT_LOOP, HELD, REFUSED_IF)
            and rsv.count("release(") == 1 and "release();" in give_back
            and ordered(refused, RESERVE_SHORT_SET, WEIGHT_REFUSAL_WARN, HOLDING_SUFFIX, "return false;")
            and "slots_" not in refused
            and VALID_DECL in norm(ring_hpp)
            and body(rc, ACQUIRE_SIG).startswith("{ if (!valid()) { return {}; }")
            and not any(name in rc for name in SHORTFALL_NAMES))


def test_ring_reserves_slot_by_slot():
    assert claim_ring_reserves_slot_by_slot(RING_CPP, RING_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-ring.cpp: body of moe_stream_ring::reserve (the valid declaration is "
        "searched in moe-stream-ring.hpp)",
        body(norm(RING_CPP), RESERVE_DEF_SIG),
        SAME_RING, GIVE_BACK_IF, "geo_ = geo;", "device_ = device;", SLOT_LOOP, HELD, REFUSED_IF, *FALLBACK_REQ,
        FALLBACK_CALL, *SLOT_USES,
        signature=RESERVE_DEF_SIG)


def test_ring_slot_by_slot_mutants():
    def c(ring_cpp=RING_CPP, ring_hpp=RING_HPP):
        return claim_ring_reserves_slot_by_slot(ring_cpp, ring_hpp)

    # The single allocation restored: the whole ring asked of the zone at once, or the loop run once.
    assert not c(ring_cpp=mutate(RING_CPP, "vram_zone_id::WEIGHT, geo.slot_bytes, MOE_STREAM_RING_SLOT_ALIGN);",
                                 "vram_zone_id::WEIGHT, geo.bytes, MOE_STREAM_RING_SLOT_ALIGN);"))
    assert not c(ring_cpp=mutate(RING_CPP, "req.size = geo.slot_bytes;", "req.size = geo.bytes;"))
    assert not c(ring_cpp=mutate(RING_CPP, SLOT_LOOP, "if (slots_.empty())"))
    # The fallback without the ring's intent: it could spill outside the arena.
    assert not c(ring_cpp=mutate(RING_CPP, "req.intent.constraints.forbid_vram_zone_spill = true;", ""))
    # A field set after the call, so the request goes out without it.
    assert not c(ring_cpp=mutate(RING_CPP, "req.intent.constraints.must_device = true; ", "").replace(
        FALLBACK_CALL, FALLBACK_CALL + " req.intent.constraints.must_device = true;", 1))
    # The fallback called whether or not the zone refused: the zone's slot would be overwritten.
    unguarded = mutate(RING_CPP, FALLBACK_IF + " { alloc_request req{};", "{ alloc_request req{};")
    assert not c(ring_cpp=unguarded)
    # A second allocation site beside the loop.
    assert not c(ring_cpp=mutate(RING_CPP, HELD, "(void) unified_allocate_owner(alloc_request{}); " + HELD))
    # The partial slots released on refusal: in the loop's break, in the refused branch, or before its return.
    assert not c(ring_cpp=mutate(RING_CPP, "if (!allocation) { break; }",
                                 "if (!allocation) { slots_.clear(); break; }"))
    assert not c(ring_cpp=mutate(RING_CPP, REFUSED_IF + " {", REFUSED_IF + " { release();"))
    assert not c(ring_cpp=mutate(RING_CPP, REFUSED_IF + " {", REFUSED_IF + " { slots_.clear();"))
    # The same drops spelled so that no list of drop names would match them, before the return of the refused branch
    # and of the loop's unresolved-slot branch: neither return's text is pinned on its own.
    rsv = body(norm(RING_CPP), RESERVE_DEF_SIG)
    refused = body(rsv[rsv.find(REFUSED_IF + " {"):], REFUSED_IF)
    unresolved = body(body(rsv, SLOT_LOOP), "if (!resolved.ptr || !resolved.on_device)")
    for drop in ("std::vector<mem_handle>().swap(slots_);", "slots_.assign(0, mem_handle());",
                 "(void) std::exchange(slots_, {});"):
        for branch in (refused, unresolved):
            assert not c(ring_cpp=mutate(RING_CPP, branch, branch.replace("return false;", drop + " return false;")))
    # A listed use traded for a drop keeps the count: the census names each use, not only how many there are.
    traded = mutate(RING_CPP, "slots_.reserve(geo.n_slots);", "")
    swapped = unresolved.replace("return false;", "std::vector<mem_handle>().swap(slots_); return false;")
    assert not c(ring_cpp=mutate(traded, unresolved, swapped))
    # Every partial reservation given back, whatever ring the next call asks for: nothing would continue.
    assert not c(ring_cpp=mutate(RING_CPP, GIVE_BACK_IF, "if (!slots_.empty())"))
    assert not c(ring_cpp=mutate(RING_CPP, " && geo.region_slots == geo_.region_slots;", ";"))
    # bytes_short computed from the largest free block again, or as the whole ring.
    assert not c(ring_cpp=mutate(RING_CPP, "*bytes_short = static_cast<size_t>(geo.n_slots - held) * geo.slot_bytes;",
                                 "*bytes_short = moe_stream_ring_weight_shortfall(device, geo.bytes);"))
    assert not c(ring_cpp=mutate(RING_CPP, "static_cast<size_t>(geo.n_slots - held)",
                                 "static_cast<size_t>(geo.n_slots)"))
    assert not c(ring_cpp=mutate(RING_CPP, "namespace ggml_sycl {",
                                 "namespace ggml_sycl { size_t largest_free_of(unified_cache & c) { size_t a, b, d, e; "
                                 "c.zone_free_figures(vram_zone_id::WEIGHT, a, b, d, e); return e; }"))
    # The refusal WARN without how many slots it holds.
    assert not c(ring_cpp=mutate(RING_CPP, " " + HOLDING_SUFFIX, ""))
    # A partial reservation counted valid, or acquire_slot handing out a partial reservation's slots.
    assert not c(ring_hpp=mutate(RING_HPP, VALID_DECL, "bool valid() const { return !slots_.empty(); }"))
    assert not c(ring_cpp=mutate(RING_CPP, "{ if (!valid()) { return {}; }", "{ if (slots_.empty()) { return {}; }"))


def test_ring_slot_by_slot_allows_any_request_field_order():
    """Positive controls: the fallback request's fields in another order, or the request built once before the loop,
    break no rule, so the claim must still hold."""
    req_block = "alloc_request req{}; " + " ".join(FALLBACK_REQ) + " "
    shuffled = mutate(RING_CPP, req_block, "alloc_request req{}; " + " ".join(reversed(FALLBACK_REQ)) + " ")
    assert claim_ring_reserves_slot_by_slot(shuffled, RING_HPP)
    hoisted = mutate(RING_CPP, req_block, "").replace(SLOT_LOOP, req_block + SLOT_LOOP, 1)
    assert claim_ring_reserves_slot_by_slot(hoisted, RING_HPP)


def test_ring_slot_by_slot_fails_on_the_single_allocation_ring():
    """The single-allocation ring (master before llama.cpp-am87), rebuilt from its parts on a copy: one allocation of
    geo.bytes and a shortfall read off the largest free block. The claim must refuse it."""
    single = mutate(RING_CPP, "vram_zone_id::WEIGHT, geo.slot_bytes, MOE_STREAM_RING_SLOT_ALIGN);",
                    "vram_zone_id::WEIGHT, geo.bytes, MOE_STREAM_RING_SLOT_ALIGN);")
    single = mutate(single, SLOT_LOOP, "if (slots_.empty())")
    single = mutate(single, "*bytes_short = static_cast<size_t>(geo.n_slots - held) * geo.slot_bytes;",
                    "*bytes_short = moe_stream_ring_weight_shortfall(device, geo.bytes);")
    assert not claim_ring_reserves_slot_by_slot(single, RING_HPP)


LOAN_WARN_SIG = "void moe_stream_phase::warn_loan_refused(size_t bytes_short) const"
LOAN_WARN_GUARD = ("if (log_ == nullptr || log_->sink == nullptr || "
                   "(policy_ != moe_exec_policy::RING && log_->trace < 1)) { return; }")
LOAN_WARN_FMT = ("[MOE-EXEC] device %d: stream ring %.1f MiB short and its expert loan refused: no further loan this "
                 "prompt")
LOAN_WARN_EMIT = "log_->sink(log_->sink_ctx, line);"


def claim_refused_loan_warns_once(phase: str, tree: dict) -> bool:
    """The brief's one WARN for a REFUSED loan (llama.cpp-3460): the helper answers REFUSED once per prompt, the branch
    calls warn_loan_refused for it, and that WARN goes out through the decision log's sink under the gate of lead
    ruling c-kcyf (an explicit ring or a trace), so the phase still prints nothing of its own at default verbosity
    under AUTO (c-xod7-q1). There, the loan transaction's own "[MOE-EXEC] loan device %d refused" line, which the
    latch now makes print once per prompt, is the one WARN. The format exists once in the backend."""
    w = body(norm(phase), LOAN_WARN_SIG)
    count = sum(norm(text).count(LOAN_WARN_FMT) for text in tree.values())
    return (count == 1 and w.startswith("{ " + LOAN_WARN_GUARD)
            and ordered(w, LOAN_WARN_GUARD, LOAN_WARN_FMT, LOAN_WARN_EMIT))


def test_refused_loan_warns_once():
    assert claim_refused_loan_warns_once(PHASE, SYCL_TREE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::warn_loan_refused (the warn format "
        "is counted across the sycl tree)",
        body(norm(PHASE), LOAN_WARN_SIG),
        LOAN_WARN_GUARD, LOAN_WARN_FMT, LOAN_WARN_EMIT,
        signature=LOAN_WARN_SIG)


def test_refused_loan_warn_mutants():
    # Ungated: AUTO would print the phase's own line at default verbosity.
    ungated = mutate(PHASE, " || (policy_ != moe_exec_policy::RING && log_->trace < 1)", "")
    assert not claim_refused_loan_warns_once(ungated, {PHASE_REL: ungated})
    # Never emitted.
    silent = mutate(PHASE, LOAN_WARN_EMIT, "(void) line;")
    assert not claim_refused_loan_warns_once(silent, {PHASE_REL: silent})
    # The format restated elsewhere.
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + LOAN_WARN_FMT + '"'
    assert not claim_refused_loan_warns_once(PHASE, tree)


DIAG_FMT = ("[MOE-EXEC] device %d: stream ring unavailable: %s (host_matrices_per_layer_max=%u region_slots_max=%u "
            "window_slots=%u matrix_bytes_max=%zu resident_expert_bytes=%zu n_slots=%u bytes=%zu)")
# One argument per field, in the format's order. window_slots and matrix_bytes_max are the ring's sizing inputs (a
# region holds one executor window, so n_slots is 2 x window_slots); region_slots_max is what a whole layer would take.
DIAG_ARGS = ("index_.host_matrices_per_layer_max, index_.region_slots_max, index_.window_slots, "
             "index_.matrix_bytes_max, index_.resident_expert_bytes, asked.n_slots, asked.bytes);")
DIAG_EMIT = "op.log->sink(op.log->sink_ctx, line);"
DIAG_RULE_CALL = "moe_stream_ring_unavailable_rule(index_, li, geo_, in_order_, reserve_failed_)"
WARN_SIG = ("void moe_stream_phase::warn_ring_unavailable(const moe_stream_op & op, "
            "const moe_stream_layer_info & li)")
RULE_SIG = "static inline const char * moe_stream_ring_unavailable_rule("
# Each rule's condition, then its name, in the order the helper tests them.
RULE_ORDER = (
    "!moe_stream_layer_claimed(li)", '"executor does not claim"',
    "idx.host_matrices_per_layer_max == 0", '"no host matrices of a streamable type"',
    "idx.matrix_bytes_max == 0", '"no matrix bytes"',
    "idx.window_slots == 0", '"no executor window"',
    "geo.n_slots == 0", '"ring exceeds an eleventh of resident plus lent expert bytes"',
    "!in_order", '"queue not in order"',
    "reserve_failed", '"reserve refused"',
)
WARN_GUARD = ("if (warned_unavailable_ || op.log == nullptr || op.log->sink == nullptr || "
              "(op.policy != moe_exec_policy::RING && op.log->trace < 1)) { return; }")
WARN_SET = "warned_unavailable_ = true;"
WARN_CLEAR = "warned_unavailable_ = false;"
WARN_CALL = ("if (st.decision.reason == moe_exec_reason::RING_UNAVAILABLE) { "
             "warn_ring_unavailable(op, cur->second); }")


def claim_ring_unavailable_diagnostic(phase: str, hpp: str, tree: dict) -> bool:
    """Lead ruling c-kcyf: a ring-unavailable answer is explained in the log. The fixed format exists once in the
    backend, carries the rule the helper names and the ring's inputs, and goes out through the decision log's WARN sink;
    the helper names each rule right after testing its condition, in a fixed order, the executor's claim first (W2: a
    layer the executor does not claim has no ring to explain)."""
    w = body(norm(phase), WARN_SIG)
    count = sum(norm(text).count(DIAG_FMT) for text in tree.values())
    return (count == 1 and ordered(w, DIAG_FMT, DIAG_RULE_CALL, DIAG_ARGS, DIAG_EMIT)
            and ordered(body(norm(hpp), RULE_SIG), *RULE_ORDER))


def test_ring_unavailable_diagnostic():
    assert claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, SYCL_TREE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::warn_ring_unavailable (the rule is "
        "searched in moe-stream-plan.hpp, the format counted across the sycl tree)",
        body(norm(PHASE), WARN_SIG),
        DIAG_FMT, DIAG_RULE_CALL, DIAG_ARGS, DIAG_EMIT,
        signature=WARN_SIG)


def test_ring_unavailable_diagnostic_mutants():
    renamed = mutate(PHASE, "n_slots=%u ", "slots=%u ")
    assert not claim_ring_unavailable_diagnostic(renamed, PLAN_HPP, {PHASE_REL: renamed})
    # A layer's slot need or the window left out, or printed under another field's name.
    dropped = mutate(PHASE, " region_slots_max=%u", "")
    assert not claim_ring_unavailable_diagnostic(dropped, PLAN_HPP, {PHASE_REL: dropped})
    no_window = mutate(PHASE, " window_slots=%u", "")
    assert not claim_ring_unavailable_diagnostic(no_window, PLAN_HPP, {PHASE_REL: no_window})
    # (anchored on the whole argument list: the ring sizing passes window_slots and matrix_bytes_max to
    # moe_stream_ring_size too)
    swapped = mutate(PHASE, "index_.region_slots_max, index_.window_slots, index_.matrix_bytes_max,",
                     "index_.window_slots, index_.region_slots_max, index_.matrix_bytes_max,")
    assert not claim_ring_unavailable_diagnostic(swapped, PLAN_HPP, {PHASE_REL: swapped})
    # The window rule dropped from the helper, or asked after the cap it would otherwise be reported as.
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, "if (idx.window_slots == 0) { return \"no executor window\"; }", ""), SYCL_TREE)
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + DIAG_FMT + '"'
    assert not claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, tree)
    assert not claim_ring_unavailable_diagnostic(mutate(PHASE, DIAG_EMIT, "(void) line;"), PLAN_HPP, SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, '"reserve refused"', '"reserve failed"'), SYCL_TREE)
    # The cap named as the 10 % of resident bytes it was before the ring's lent bytes were counted back.
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, '"ring exceeds an eleventh of resident plus lent expert bytes"',
                      '"ring exceeds 10% of resident expert bytes"'), SYCL_TREE)
    # The claim rule dropped, or asked after the ring's rules that it would otherwise be reported as.
    claim_rule = 'if (!moe_stream_layer_claimed(li)) { return "executor does not claim"; }'
    assert not claim_ring_unavailable_diagnostic(PHASE, mutate(PLAN_HPP, claim_rule, ""), SYCL_TREE)
    late = mutate(PLAN_HPP, claim_rule, "")
    late = mutate(late, 'return "region smaller than the largest streamable matrix";',
                  claim_rule + ' return "region smaller than the largest streamable matrix";')
    assert not claim_ring_unavailable_diagnostic(PHASE, late, SYCL_TREE)
    # The diagnostic asked about another layer's claim than the one the decision is for.
    assert not claim_ring_unavailable_diagnostic(
        mutate(PHASE, DIAG_RULE_CALL, "moe_stream_ring_unavailable_rule(index_, index_.layers.begin()->second, geo_, "
                                      "in_order_, reserve_failed_)"), PLAN_HPP, SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, "geo.n_slots == 0", "geo.bytes == 0"), SYCL_TREE)


def claim_ring_unavailable_once_per_prompt(phase: str) -> bool:
    """Lead ruling c-kcyf: at most one diagnostic per prompt, and only under an explicit ring or a trace. The guard
    comes first, the flag is set before the line goes out, end_prompt clears it and begin_ubatch does not, and
    begin_layer calls it for the decision the layer ends with (after a missing claim or a refused reserve rewrote it),
    before the layer's report. The
    phase has no log call of its own: a queue that is not in order is one of the diagnostic's rules, under the same
    gating (lead ruling c-xod7-q1)."""
    n = norm(phase)
    w = body(n, WARN_SIG)
    return (w.startswith("{ " + WARN_GUARD) and "GGML_LOG_" not in n
            and n.count(WARN_SET) == 1 and ordered(w, WARN_GUARD, WARN_SET, DIAG_EMIT)
            and n.count(WARN_CLEAR) == 1 and WARN_CLEAR in body(n, END_PROMPT_SIG)
            and WARN_CLEAR not in body(n, BEGIN_UBATCH_SIG)
            and ordered(body(n, BEGIN_LAYER_SIG), RING_STANDS, WARN_CALL, REPORT_CALL))


def test_ring_unavailable_once_per_prompt():
    assert claim_ring_unavailable_once_per_prompt(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::warn_ring_unavailable (the clear "
        "is searched in end_prompt and begin_ubatch, the call in begin_layer)",
        body(norm(PHASE), WARN_SIG),
        WARN_GUARD, WARN_SET, DIAG_EMIT,
        signature=WARN_SIG)


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


REPORT_CALL = "report(layer, st);"
DECIDE_IF = "if (!st.decided)"
WEIGH_CALL = "st.decision = weigh(cur->second, op, 0.0);"


def brace_depth(text: str, at: int) -> int:
    """Braces open at `at` in a body that holds no string literal (the decide block's does not)."""
    return text.count("{", 0, at) - text.count("}", 0, at)


def claim_auto_still_reports(phase: str) -> bool:
    """The decision log records every layer's decision under every policy, AUTO included, whether or not the ring
    stages for it. begin_layer calls report() once, at the top level of the block that decides the layer at its first
    op, after the RING guard has closed, with no return in that block."""
    n = norm(phase)
    b = body(n, BEGIN_LAYER_SIG)
    at = b.find(DECIDE_IF + " {")
    d = body(b[at:], DECIDE_IF) if at >= 0 else ""
    guard = d.find(RING_STANDS)
    rat = d.find(REPORT_CALL)
    return (n.count(REPORT_CALL) == 1 and d.count(REPORT_CALL) == 1 and 0 <= guard < rat and brace_depth(d, rat) == 1
            and "return" not in d)


def test_auto_still_reports():
    assert claim_auto_still_reports(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::begin_layer (the decide branch)",
        body(norm(PHASE), BEGIN_LAYER_SIG),
        DECIDE_IF + " {", RING_STANDS, REPORT_CALL,
        signature=BEGIN_LAYER_SIG)


def test_auto_report_mutants():
    # Only an explicit ring reports: AUTO's choice would never reach the decision log.
    assert not claim_auto_still_reports(
        mutate(PHASE, REPORT_CALL, "if (op.policy == moe_exec_policy::RING) { " + REPORT_CALL + " }"))
    # AUTO returns before the report.
    assert not claim_auto_still_reports(
        mutate(PHASE, WARN_CALL, WARN_CALL + " if (op.policy != moe_exec_policy::RING) { return decision_of(layer); }"))
    # No report at all.
    assert not claim_auto_still_reports(mutate(PHASE, REPORT_CALL, ""))


def claim_begin_layer_weighs_only_its_layer(phase: str, hpp: str) -> bool:
    """Task 2b of llama.cpp-a2xb: begin_layer weighs and decides its own layer at its first op with no copy window,
    since no copy of a layer hides under the previous layer's (there is no cross-layer prefetch), so weigh's window_us
    is 0.0 and the layer's state keeps no window. It looks up no other layer and stages nothing: window_begin stages."""
    b = body(norm(phase), BEGIN_LAYER_SIG)
    return (b.count("index_.layers.find(") == 1 and "index_.layers.find(layer)" in b and b.count("weigh(") == 1
            and WEIGH_CALL in b and "stage_window(" not in b and "window_begin(" not in b
            and "window_us" not in body(norm(hpp), "struct layer_state"))


def test_begin_layer_weighs_only_its_layer():
    assert claim_begin_layer_weighs_only_its_layer(PHASE, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::begin_layer",
        body(norm(PHASE), BEGIN_LAYER_SIG),
        "index_.layers.find(layer)", WEIGH_CALL,
        signature=BEGIN_LAYER_SIG)


def test_begin_layer_weigh_mutants():
    c = claim_begin_layer_weighs_only_its_layer
    # The cost model discounted by a copy window again.
    assert not c(mutate(PHASE, WEIGH_CALL, "st.decision = weigh(cur->second, op, 1000.0);"), PLAN_HPP)
    # The next layer looked up and weighed ahead.
    assert not c(mutate(PHASE, WEIGH_CALL, WEIGH_CALL + " const auto nxt = index_.layers.find(layer + 1); "
                                                        "(void) weigh(nxt->second, op, 0.0);"), PLAN_HPP)
    # The layer staged at its first op.
    assert not c(mutate(PHASE, WEIGH_CALL, WEIGH_CALL + " (void) stage_window({}, 0, cache, op);"), PLAN_HPP)
    # The layer's state keeps a window.
    assert not c(PHASE, mutate(PLAN_HPP, "bool decided = false; };",
                               "bool decided = false; double window_us = 0.0; };"))


# ---- the ring's lifetime ---------------------------------------------------------------------------------------------

TEARDOWN_SIG = "void moe_stream_phase::teardown(sycl::event last_use)"


TEARDOWN_RELEASE = "ring_->release(last_use);"
RING_RELEASE_AFTER_DECL = "void release(sycl::event last_use);"
RING_RELEASE_AFTER_SIG = "void moe_stream_ring::release(sycl::event last_use)"
RING_RETAIN_EVERY_SLOT = "retain_handles_until_event(slots_, std::move(last_use));"
RING_RELEASE_AFTER_BODY = "{ if (!slots_.empty()) { " + RING_RETAIN_EVERY_SLOT + " } release(); }"


def claim_teardown_gates_before_release(phase: str, hpp: str = PLAN_HPP, ring_cpp: str = RING_CPP,
                                        ring_hpp: str = RING_HPP) -> bool:
    """Ruling D: the open region is gated on the last use and the ring is released after it; the destructor and
    end_prompt both tear down through it, and the ring is offered only on an in-order queue. Every slot waits for the
    last use, not only the slots of the regions a gate covers (llama.cpp-am87): a region's gate can be its last copy,
    which precedes the region's readers, the executor's dequants, and the slots are separate allocations, so no one
    slot keeps the others. The ring's release(last_use) hands every slot to the drain on the last use before release()
    applies the region gates; the phase's one release is teardown's, through it, and no anchor slot remains."""
    n = norm(phase)
    td = body(n, TEARDOWN_SIG)
    return (ordered(td, "close_region(last_use);", TEARDOWN_RELEASE) and n.count("->release(") == 1
            and RING_RELEASE_AFTER_DECL in norm(ring_hpp)
            and body(norm(ring_cpp), RING_RELEASE_AFTER_SIG) == RING_RELEASE_AFTER_BODY
            and "anchor_" not in n and "anchor_" not in norm(hpp)
            and "teardown(" in body(n, "moe_stream_phase::~moe_stream_phase()")
            and "teardown(" in body(n, "void moe_stream_phase::end_prompt()")
            and "in_order_ = ccs.is_in_order();" in n)


def test_teardown_gates_before_release():
    assert claim_teardown_gates_before_release(PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::teardown (the ring release is "
        "searched in moe-stream-ring.cpp and moe-stream-ring.hpp)",
        body(norm(PHASE), TEARDOWN_SIG),
        "close_region(last_use);", TEARDOWN_RELEASE,
        signature=TEARDOWN_SIG)


def in_body(text: str, signature: str, old: str, new: str) -> str:
    """`text` (normalized) with `old` replaced once inside the body of `signature`; the mutant must exist."""
    n = norm(text)
    b = body(n, signature)
    assert old in b, f"mutant precondition: {old!r} is not in {signature!r}"
    return n.replace(b, b.replace(old, new, 1), 1)


def test_teardown_mutants():
    def c(phase=PHASE, hpp=PLAN_HPP, ring_cpp=RING_CPP, ring_hpp=RING_HPP):
        return claim_teardown_gates_before_release(phase, hpp, ring_cpp, ring_hpp)

    assert not c(phase=mutate(PHASE, "close_region(last_use);", ""))
    assert not c(phase=mutate(PHASE, "in_order_ = ccs.is_in_order();", "in_order_ = true;"))
    # The ring released on its region gates alone: the current region's slots would wait for nothing past last_use,
    # and an earlier region's for a copy its readers follow.
    assert not c(phase=mutate(PHASE, TEARDOWN_RELEASE, "ring_->release();"))
    # Only the gated regions' slots held until the last use.
    assert not c(ring_cpp=mutate(RING_CPP, RING_RETAIN_EVERY_SLOT,
                                 "for (const region & r : regions_) { if (r.has_gate) { "
                                 "const auto first = slots_.begin() + r.first; "
                                 "retain_handles_until_event(std::vector<mem_handle>(first, first + r.count), "
                                 "last_use); } }"))
    # One slot held until the last use, the single allocation's anchor: the other slots are their own allocations.
    assert not c(ring_cpp=mutate(RING_CPP, RING_RETAIN_EVERY_SLOT,
                                 "retain_handles_until_event({ slots_.front() }, std::move(last_use));"))
    # The slots dropped before they are handed to the drain: release() clears them, so nothing would be retained.
    assert not c(ring_cpp=mutate(RING_CPP, RING_RELEASE_AFTER_BODY,
                                 "{ release(); if (!slots_.empty()) { " + RING_RETAIN_EVERY_SLOT + " } }"))
    # A second release site in the phase, which would skip the last use.
    assert not c(phase=in_body(PHASE, END_PROMPT_SIG, "teardown(", "if (ring_) { ring_->release(); } teardown("))
    # The anchor slot back in the phase.
    assert not c(hpp=mutate(PLAN_HPP, PENDING_DECL, PENDING_DECL + " mem_handle anchor_;"))
    # The destructor gives the ring back without teardown().
    assert not c(phase=in_body(PHASE, "moe_stream_phase::~moe_stream_phase()", "teardown(", "(void) ("))
    # end_prompt keeps the ring.
    assert not c(phase=in_body(PHASE, END_PROMPT_SIG, "teardown(ring_ ? compute_barrier() : sycl::event());", ""))


PENDING_DECL = "std::unique_ptr<moe_stream_ring> pending_ring_;"
PENDING_TAKE = ("std::unique_ptr<moe_stream_ring> ring = pending_ring_ ? std::move(pending_ring_) : "
                "std::make_unique<moe_stream_ring>();")
PENDING_KEEP = "pending_ring_ = std::move(ring);"
PENDING_DROP = "pending_ring_.reset();"
END_PROMPT_EARLY = "if (!ubatch_live_ && !ring_ && !pending_ring_ && table_.empty()) { return; }"
TEARDOWN_RING_IF = "if (ring_)"
# The refused branch keeps the ring only while a loan can still fund it, decided after the loan step (which may latch
# a REFUSED loan in this very call); otherwise the ring gives its slots back as it leaves scope (lead ruling, c-zgrd).
LOAN_CAN_FOLLOW = "const bool loan_can_follow = !loan_latch_.refused && request_loan_ != nullptr;"
KEEP_WHILE_LOAN = "if (loan_can_follow) { " + PENDING_KEEP + " } else { warn_partial_released(ring->slots_held()); }"
RELEASED_WARN_SIG = "void moe_stream_phase::warn_partial_released(uint32_t held) const"
RELEASED_WARN_GUARD = ("if (held == 0 || log_ == nullptr || log_->sink == nullptr || "
                       "(policy_ != moe_exec_policy::RING && log_->trace < 1)) { return; }")
RELEASED_WARN_FMT = ("[MOE-STREAM-RING] partial reservation released on device %d: no loan can follow "
                     "(held %u of %u slots)")
SLOTS_HELD_DECL = "uint32_t slots_held() const { return static_cast<uint32_t>(slots_.size()); }"


def claim_partial_reservation_survives_the_ubatch(phase: str, hpp: str, ring_hpp: str = RING_HPP) -> bool:
    """A refused reserve keeps the slots it got (llama.cpp-am87), and the phase keeps the refused ring for the next
    micro-batch's attempt, which continues from those slots instead of asking the zone for the whole ring again. The
    hold is for the loan path only (lead ruling, c-zgrd): after the loan step, the refused ring is kept only while a
    loan can still fund it, and otherwise gives its slots back to the WEIGHT zone, which mid-prompt runtime
    allocations prefer, with one WARN under the ring-unavailable line's gate naming how many it held (none for an
    empty ring). ring_ready() takes the pending ring when there is one and moves a reserved ring into ring_. The
    pending ring is nobody's ring: only ring_ready() takes or stores it, so neither the executor's claim nor the
    release count can take it for a reserved one. teardown() drops it outside its ring_ branch (a pending ring exists
    only while ring_ does not, and it handed nothing out, so its slots go back at once), and end_prompt() does not
    skip the teardown while one is held. A new geometry or device is the ring's own business: reserve() gives the
    partial slots back and starts over."""
    n = norm(phase)
    ready = body(n, RING_READY_SIG)
    at = ready.find(RESERVE_SHORT_IF + " {")
    branch = body(ready[at:], RESERVE_SHORT_IF) if at >= 0 else ""
    td = body(n, TEARDOWN_SIG)
    warn = body(n, RELEASED_WARN_SIG)
    return (norm(hpp).count(PENDING_DECL) == 1 and n.count("pending_ring_") == 5
            and ordered(ready, LATCH_GUARD, PENDING_TAKE, RESERVE_SHORT_IF, RING_READY_RESET)
            and ordered(branch, LATCH_SET, LOAN_CALL, LOAN_REFUSED_WARN, LOAN_CAN_FOLLOW, KEEP_WHILE_LOAN,
                        "return false;")
            and n.count(PENDING_KEEP) == 1 and n.count("warn_partial_released(") == 2
            and warn.startswith("{ " + RELEASED_WARN_GUARD)
            and ordered(warn, RELEASED_WARN_FMT, LOAN_WARN_EMIT) and SLOTS_HELD_DECL in norm(ring_hpp)
            and td.count(PENDING_DROP) == 1 and PENDING_DROP not in body(td, TEARDOWN_RING_IF)
            and END_PROMPT_EARLY in body(n, END_PROMPT_SIG))


def test_partial_reservation_survives_the_ubatch():
    assert claim_partial_reservation_survives_the_ubatch(PHASE, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::ring_ready (the warn, teardown and "
        "end_prompt bodies are searched too)",
        body(norm(PHASE), RING_READY_SIG),
        LATCH_GUARD, PENDING_TAKE, RESERVE_SHORT_IF, RING_READY_RESET, LATCH_SET, LOAN_CALL, LOAN_REFUSED_WARN,
        LOAN_CAN_FOLLOW, KEEP_WHILE_LOAN,
        signature=RING_READY_SIG)


def test_partial_reservation_mutants():
    def c(phase=PHASE, hpp=PLAN_HPP, ring_hpp=RING_HPP):
        return claim_partial_reservation_survives_the_ubatch(phase, hpp, ring_hpp)

    # The refused ring dropped while a loan can still fund it: the next attempt asks the zone for the whole ring again.
    assert not c(phase=mutate(PHASE, KEEP_WHILE_LOAN, "warn_partial_released(ring->slots_held());"))
    # Kept after a REFUSED loan, or with no loan hook: the hold would outlive any loan, holding WEIGHT room the
    # runtime allocations prefer until end_prompt.
    assert not c(phase=mutate(PHASE, LOAN_CAN_FOLLOW, "const bool loan_can_follow = request_loan_ != nullptr;"))
    assert not c(phase=mutate(PHASE, LOAN_CAN_FOLLOW, "const bool loan_can_follow = !loan_latch_.refused;"))
    assert not c(phase=mutate(PHASE, LOAN_CAN_FOLLOW, "const bool loan_can_follow = true;"))
    # Decided before the loan step: a loan this call latched REFUSED would leave the ring held.
    early = in_body(PHASE, RING_READY_SIG, LOAN_CAN_FOLLOW + " ", "")
    assert not c(phase=in_body(early, RING_READY_SIG, LATCH_SET, LOAN_CAN_FOLLOW + " " + LATCH_SET))
    # Released silently: the ring's "(holding K of N slots)" line would go unexplained.
    assert not c(phase=mutate(PHASE, KEEP_WHILE_LOAN, "if (loan_can_follow) { " + PENDING_KEEP + " }"))
    # The release WARN ungated (AUTO would print it at default verbosity), printed for an empty ring, or never emitted.
    assert not c(phase=in_body(PHASE, RELEASED_WARN_SIG, " || (policy_ != moe_exec_policy::RING && log_->trace < 1)",
                               ""))
    assert not c(phase=in_body(PHASE, RELEASED_WARN_SIG, "held == 0 || ", ""))
    assert not c(phase=in_body(PHASE, RELEASED_WARN_SIG, LOAN_WARN_EMIT, "(void) line;"))
    # The held count is not the slots the ring holds.
    assert not c(ring_hpp=mutate(RING_HPP, SLOTS_HELD_DECL, "uint32_t slots_held() const { return geo_.n_slots; }"))
    # Every attempt starts from an empty ring.
    assert not c(phase=mutate(PHASE, PENDING_TAKE,
                              "std::unique_ptr<moe_stream_ring> ring = std::make_unique<moe_stream_ring>();"))
    # A reserved ring kept pending: nobody would stage into it.
    assert not c(phase=mutate(PHASE, RING_READY_RESET, PENDING_KEEP + " reset_regions(); return true;"))
    # The pending ring taken for a reserved one.
    assert not c(phase=in_body(PHASE, RING_READY_SIG, "if (ring_) { return true; }",
                               "if (ring_ || pending_ring_) { return true; }"))
    # teardown keeps it, or drops it only where a reserved ring is, which a pending ring never is.
    assert not c(phase=in_body(PHASE, TEARDOWN_SIG, " " + PENDING_DROP, ""))
    inside = in_body(PHASE, TEARDOWN_SIG, " " + PENDING_DROP, "")
    assert not c(phase=in_body(inside, TEARDOWN_SIG, "++ring_releases_;", "++ring_releases_; " + PENDING_DROP))
    # end_prompt skips the teardown while only a pending ring is held: its slots would stay out through decode.
    assert not c(phase=mutate(PHASE, "!ring_ && !pending_ring_ &&", "!ring_ &&"))
    # The member gone from the phase's declaration.
    assert not c(hpp=mutate(PLAN_HPP, PENDING_DECL, ""))


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
    assert claim_context_owns_and_ends_the_phase(COMMON, SYCL), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: body of ggml_backend_sycl_context::~ggml_backend_sycl_context (the "
        "member is searched in common.hpp, the layer-done body in ggml-sycl.cpp)",
        body(norm(SYCL), DTOR_SIG),
        "qptrs[dev][s]->wait();", "moe_stream.reset();", "ggml_sycl::drain_retained_handles(true);",
        signature=DTOR_SIG)


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


REORDER_DECL = ("sycl::event ggml_sycl_reorder_weight_gpu(sycl::queue & queue, const uint8_t * aos_device, "
                "uint8_t * soa_device, ggml_type type, int64_t ncols, int64_t nrows, size_t src_bytes, "
                "const std::vector<sycl::event> & deps);")


def claim_reorder_is_the_load_fills_own(sycl: str, hpp: str) -> bool:
    """The load path's reorder dispatch is a static function of ggml-sycl.cpp, defined once and called once, by the
    load-time fill, and moe-stream-plan.hpp names it nowhere outside comments: the SOA staging that shared it is
    retired, and the ring copies AOS and reorders nothing (test_window_copy_is_the_only_submission)."""
    n = norm(sycl)
    return (n.count("static " + REORDER_DEF) == 1 and n.count(REORDER_DEF) == 1
            and n.count(REORDER_CALL) == 2 and n.count(LOAD_FILL) == 1 and REORDER_CALL not in norm(hpp))


def test_reorder_is_the_load_fills_own():
    assert claim_reorder_is_the_load_fills_own(SYCL, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/ggml-sycl.cpp: whole file (the reorder is also searched for in "
        "moe-stream-plan.hpp, where it must be absent)",
        norm(SYCL),
        "static " + REORDER_DEF, REORDER_CALL, LOAD_FILL)


def test_reorder_load_fill_mutants():
    c = claim_reorder_is_the_load_fills_own
    # The stream header declares it again.
    assert not c(SYCL, mutate(PLAN_HPP, "namespace ggml_sycl {", REORDER_DECL + " namespace ggml_sycl {"))
    # The definition is external again.
    assert not c(mutate(SYCL, "static " + REORDER_DEF, REORDER_DEF), PLAN_HPP)
    # A second caller.
    assert not c(mutate(SYCL, LOAD_FILL, REORDER_CALL + "q2, a, s, t, c, r, b, d); " + LOAD_FILL), PLAN_HPP)
    # The load-time fill moved off the dispatch.
    assert not c(mutate(SYCL, LOAD_FILL, "other_reorder(queue, static_cast<const uint8_t *>(temp_vram),"), PLAN_HPP)


# ---- the trace -------------------------------------------------------------------------------------------------------


REPORT_WINDOW_SIG = ("void moe_stream_phase::report_window(int layer, expert_tensor_role role, size_t phase, "
                     "const stage_counts & n) const")
TRACE_GUARD = "if (log_ == nullptr || log_->trace < 1 || log_->trace_sink == nullptr) { return; }"
TRACE_ARGS = "layer, expert_tensor_role_name(role), phase, n.staged, n.bytes);"
STAGE_REPORT = ("const stage_counts n = stage_window(window, moe_stream_window_id(req.layer, role, p), cache, op); "
                "report_window(req.layer, role, p, n);")
GONE_TRACES = ("[MOE-EXEC-TRACE] stage layer", "[MOE-EXEC-TRACE] stage layouts")


def claim_trace_line(phase: str, tree: dict) -> bool:
    """The plan's window trace line (Task 2b of llama.cpp-a2xb) exists once in the backend and goes out through the
    decision log's trace sink, under the trace guard, right after every window stage, with its fields in the format's
    order. It shares the "[MOE-EXEC-TRACE]" prefix with the decision log's note() line ("[MOE-EXEC-TRACE] dev %d layer
    %d", test-moe-exec-policy pins it) and differs in the words after it. The per-layer stage line and its per-layout
    split are gone with the layer stage: every slot holds AOS."""
    n = norm(phase)
    rw = body(n, REPORT_WINDOW_SIG)
    count = sum(norm(text).count(TRACE_FMT) for text in tree.values())
    return (count == 1 and rw.startswith("{ " + TRACE_GUARD) and ordered(rw, TRACE_FMT, TRACE_ARGS, TRACE_EMIT)
            and n.count(STAGE_REPORT) == 1 and len(STAGE_WINDOW_CALL_RE.findall(n)) == 1
            and not any(gone in text for gone in GONE_TRACES for text in tree.values()))


def test_trace_line():
    assert claim_trace_line(PHASE, SYCL_TREE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::report_window (the trace format is "
        "counted across the sycl tree)",
        body(norm(PHASE), REPORT_WINDOW_SIG),
        TRACE_GUARD, TRACE_FMT, TRACE_ARGS, TRACE_EMIT,
        signature=REPORT_WINDOW_SIG)


def test_trace_mutants():
    m = mutate(PHASE, "%d items %zu bytes", "%d staged %zu bytes")
    assert not claim_trace_line(m, {PHASE_REL: m})
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + TRACE_FMT + '"'
    assert not claim_trace_line(PHASE, tree)
    assert not claim_trace_line(mutate(PHASE, TRACE_EMIT, "(void) line;"), {PHASE_REL: PHASE})
    # Under the decision log's WARN prefix.
    old = mutate(PHASE, "[MOE-EXEC-TRACE] stage window layer", "[MOE-EXEC] stage window layer")
    assert not claim_trace_line(old, {PHASE_REL: old})
    # Printed at every trace level, the default included.
    m = mutate(PHASE, TRACE_GUARD, "")
    assert not claim_trace_line(m, {PHASE_REL: m})
    # Two fields printed under each other's names.
    m = mutate(PHASE, TRACE_ARGS, "layer, expert_tensor_role_name(role), phase, n.bytes, n.staged);")
    assert not claim_trace_line(m, {PHASE_REL: m})
    # A window staged without its line.
    m = mutate(PHASE, " report_window(req.layer, role, p, n);", "")
    assert not claim_trace_line(m, {PHASE_REL: m})
    # The per-layout split line back.
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"[MOE-EXEC-TRACE] stage layouts layer %d ubatch %u: soa=%d aos_only=%d"'
    assert not claim_trace_line(PHASE, tree)


# ---- whether the prompt's ring engaged (llama.cpp-i4ny) --------------------------------------------------------------

ENGAGED_DECL = "bool last_prompt_ring_engaged() const;"
ENGAGED_SIG = "bool moe_stream_phase::last_prompt_ring_engaged() const"
ENGAGED_BODY = "{ return last_prompt_ring_engaged_; }"
TAKE_ENDED_DECL = "bool take_prompt_ended();"
TAKE_ENDED_SIG = "bool moe_stream_phase::take_prompt_ended()"
TAKE_ENDED_BODY = "{ const bool ended = prompt_ended_; prompt_ended_ = false; return ended; }"
ENGAGED_DECLS = ("bool ring_engaged_ = false;", "bool last_prompt_ring_engaged_ = false;",
                 "bool prompt_ended_ = false;")
ENGAGED_SET = "ring_engaged_ = true; ring_ = std::move(ring);"
ENGAGED_COPY = "last_prompt_ring_engaged_ = ring_engaged_; ring_engaged_ = false; prompt_ended_ = true;"


def claim_end_prompt_reports_ring_engagement(phase: str, hpp: str) -> bool:
    """A loan exists to fund the prompt's ring, so whether that ring engaged decides how the backend returns the loan:
    at once when it never engaged, a triplet per decode graph when it did (llama.cpp-i4ny). The flag is set at the one
    place ring_ready() keeps a whole reservation, after the refused branch (a refused ring kept pending funds nothing),
    and nowhere else; a ring a new geometry tore down mid-prompt still counts. end_prompt copies the flag and clears it
    for the next prompt only past its early return, so every decode graph after the first leaves the answer standing,
    and it marks the prompt ended; take_prompt_ended() answers true once per ended prompt. The accessor reads the copy,
    not the live flag."""
    n = norm(phase)
    h = norm(hpp)
    ready = body(n, RING_READY_SIG)
    end = body(n, END_PROMPT_SIG)
    return (h.count(ENGAGED_DECL) == 1 and h.count(TAKE_ENDED_DECL) == 1 and body(n, ENGAGED_SIG) == ENGAGED_BODY
            and all(h.count(d) == 1 for d in ENGAGED_DECLS)
            and n.count("ring_engaged_ = true;") == 1 and ready.count(ENGAGED_SET) == 1
            and ordered(ready, LATCH_GUARD, RESERVE_SHORT_IF, "return false;",
                        ENGAGED_SET + " reset_regions(); return true;")
            and n.count("prompt_ended_ = true;") == 1 and n.count("last_prompt_ring_engaged_ =") == 1
            and end.count(ENGAGED_COPY) == 1 and ordered(end, END_PROMPT_EARLY, ENGAGED_COPY, "teardown(")
            and body(n, TAKE_ENDED_SIG) == TAKE_ENDED_BODY)


def test_end_prompt_reports_ring_engagement():
    assert claim_end_prompt_reports_ring_engagement(PHASE, PLAN_HPP), first_missing(
        "ggml/src/ggml-sycl/moe-stream-plan.cpp: body of moe_stream_phase::end_prompt (the ring-ready and "
        "accessor bodies are searched too, the declarations in moe-stream-plan.hpp)",
        body(norm(PHASE), END_PROMPT_SIG),
        END_PROMPT_EARLY, ENGAGED_COPY, "teardown(",
        signature=END_PROMPT_SIG)


def test_end_prompt_reports_ring_engagement_mutants():
    def c(phase=PHASE, hpp=PLAN_HPP):
        return claim_end_prompt_reports_ring_engagement(phase, hpp)

    # Never set: every prompt would read as unengaged and lose a loan its ring used.
    assert not c(phase=mutate(PHASE, "ring_engaged_ = true; ", ""))
    # Set where a refused ring is kept pending instead: a ring that never reserved would read as engaged.
    m = mutate(PHASE, "ring_engaged_ = true; ", "")
    assert not c(phase=mutate(m, PENDING_KEEP, PENDING_KEEP + " ring_engaged_ = true;"))
    # Copied before the early return: the second decode graph would overwrite the answer with an empty prompt's.
    m = in_body(PHASE, END_PROMPT_SIG, " " + ENGAGED_COPY, "")
    assert not c(phase=in_body(m, END_PROMPT_SIG, END_PROMPT_EARLY, ENGAGED_COPY + " " + END_PROMPT_EARLY))
    # Read from ring_ alone: a ring a new geometry tore down mid-prompt would read as unengaged.
    assert not c(phase=mutate(PHASE, "last_prompt_ring_engaged_ = ring_engaged_;",
                              "last_prompt_ring_engaged_ = ring_ != nullptr;"))
    # Not cleared per prompt: an unengaged prompt after an engaged one would read as engaged.
    assert not c(phase=mutate(PHASE, ENGAGED_COPY, "last_prompt_ring_engaged_ = ring_engaged_; prompt_ended_ = true;"))
    # Not marked ended, or marked ended for good: the backend would read the flag never, or at every decode graph.
    assert not c(phase=mutate(PHASE, " prompt_ended_ = true;", ""))
    assert not c(phase=mutate(PHASE, TAKE_ENDED_BODY, "{ return prompt_ended_; }"))
    # The accessor reads the live flag, which end_prompt has just cleared.
    assert not c(phase=mutate(PHASE, ENGAGED_BODY, "{ return ring_engaged_; }"))


# ---- the ring slot's layout is one named constant --------------------------------------------------------------------

SLOT_LAYOUT_NAME = "MOE_STREAM_RING_SLOT_LAYOUT"
SLOT_LAYOUT_DEF = "static constexpr ggml_layout_mode MOE_STREAM_RING_SLOT_LAYOUT = GGML_LAYOUT_AOS;"
SLOT_LAYOUT_DEFINITION_RE = re.compile(
    r"\b(?:ggml_layout_mode|auto)\s+MOE_STREAM_RING_SLOT_LAYOUT\b|#\s*define\s+MOE_STREAM_RING_SLOT_LAYOUT\b")
RING_RESERVE_SIG = "bool moe_stream_ring::reserve("
SLOT_WRAP = ("mem_handle slot = mem_handle::from_owned_alloc(std::move(allocation.owner), "
             "MOE_STREAM_RING_SLOT_LAYOUT);")


def claim_ring_slot_layout_is_one_constant(ring_hpp: str, ring_cpp: str, phase: str) -> bool:
    """The layout every ring slot holds is read from one constant, MOE_STREAM_RING_SLOT_LAYOUT, defined once in
    moe-stream-ring.hpp as GGML_LAYOUT_AOS (W2 of docs/plans/2026-10-24-prefill-window-ring.md: the ring stages AOS
    only). The three sites of this pin's files name it and spell no GGML_LAYOUT_AOS of their own: the ring's slot wrap
    in reserve(), and stage_window's slot acquire and stage-table entry."""
    h, c, p = norm(ring_hpp), norm(ring_cpp), norm(phase)
    own = (RING_HPP_REL, RING_CPP_REL, PHASE_REL)
    elsewhere = (sum(len(SLOT_LAYOUT_DEFINITION_RE.findall(norm(text))) for rel, text in SYCL_TREE.items()
                     if rel not in own)
                 + len(SLOT_LAYOUT_DEFINITION_RE.findall(c)) + len(SLOT_LAYOUT_DEFINITION_RE.findall(p)))
    reserve = body(c, RING_RESERVE_SIG)
    stage = body(p, STAGE_SIG)
    return (h.count(SLOT_LAYOUT_DEF) == 1 and len(SLOT_LAYOUT_DEFINITION_RE.findall(h)) + elsewhere == 1
            and SLOT_WRAP in reserve and "GGML_LAYOUT_AOS" not in reserve
            and AOS_ACQUIRE in stage and TABLE_WRITE in stage and "GGML_LAYOUT_AOS" not in stage)


def test_ring_slot_layout_is_one_constant():
    assert claim_ring_slot_layout_is_one_constant(RING_HPP, RING_CPP, PHASE), first_missing(
        "ggml/src/ggml-sycl/moe-stream-ring.hpp: whole file (the definition is also counted across the sycl "
        "tree with a regex, and the reserve and stage_window bodies are searched for the constant)",
        norm(RING_HPP),
        SLOT_LAYOUT_DEF)


def test_ring_slot_layout_constant_mutants():
    def c(ring_hpp=RING_HPP, ring_cpp=RING_CPP, phase=PHASE):
        return claim_ring_slot_layout_is_one_constant(ring_hpp, ring_cpp, phase)

    # Each site restored to the literal.
    assert not c(ring_cpp=mutate(RING_CPP, SLOT_WRAP, SLOT_WRAP.replace(SLOT_LAYOUT_NAME, "GGML_LAYOUT_AOS")))
    assert not c(phase=mutate(PHASE, AOS_ACQUIRE, AOS_ACQUIRE.replace(SLOT_LAYOUT_NAME, "GGML_LAYOUT_AOS")))
    assert not c(phase=mutate(PHASE, TABLE_WRITE, TABLE_WRITE.replace(SLOT_LAYOUT_NAME, "GGML_LAYOUT_AOS")))
    # A literal beside the constant, in either body.
    assert not c(ring_cpp=mutate(RING_CPP, SLOT_WRAP, SLOT_WRAP + " (void) GGML_LAYOUT_AOS;"))
    assert not c(phase=mutate(PHASE, RETAIN, RETAIN + " (void) GGML_LAYOUT_AOS;"))
    # The constant naming another layout: every site would stage a layout the ring does not hold.
    soa_def = SLOT_LAYOUT_DEF.replace("GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA")
    assert not c(ring_hpp=mutate(RING_HPP, SLOT_LAYOUT_DEF, soa_def))
    # A second definition, in the header or in a source that includes it.
    assert not c(ring_hpp=norm(RING_HPP) + " " + SLOT_LAYOUT_DEF)
    assert not c(phase=norm(PHASE) + " " + SLOT_LAYOUT_DEF)
    # Other spellings of a second definition: brace-initialized, or a macro.
    brace_def = " static constexpr ggml_layout_mode MOE_STREAM_RING_SLOT_LAYOUT{GGML_LAYOUT_AOS};"
    assert not c(ring_hpp=norm(RING_HPP) + brace_def)
    assert not c(phase=norm(PHASE) + " #define MOE_STREAM_RING_SLOT_LAYOUT GGML_LAYOUT_AOS")
    # A shadow in another scope, spelled constexpr auto, before the acquire.
    assert not c(phase=mutate(PHASE, AOS_ACQUIRE, "constexpr auto MOE_STREAM_RING_SLOT_LAYOUT = GGML_LAYOUT_SOA; "
                              + AOS_ACQUIRE))
    # Control: a comparison is a use, not a definition.
    assert c(phase=mutate(PHASE, RETAIN, RETAIN + " (void) (MOE_STREAM_RING_SLOT_LAYOUT == GGML_LAYOUT_SOA);"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
