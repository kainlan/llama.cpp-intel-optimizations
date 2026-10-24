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
  - a refused reserve asks for that loan (llama.cpp-3460, closing c-0tmn): the request is made once, in ring_ready()'s
    reserve-failed branch after the latch, with the shortfall the reserve reports (the ring's bytes past the WEIGHT
    zone's largest free block), through op hooks the backend wires to ggml_sycl_request_expert_loan and to the boundary
    runner's REFUSED count. A loan this prompt asked for that comes back REFUSED (never BUSY, which the runner re-posts)
    latches until end_prompt, with one WARN under an explicit ring or a trace;
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
    const_cast (ruling E) and calls no allocator: its slots come from the ring's one reservation;
  - the number of waits in ggml_sycl_mul_mat_id is the one master had at 84fb449da;
  - only a host lease whose retained owner is a host-pinned WEIGHT allocation of this device's host zones is copied
    (llama.cpp-9y0p: any other lease would take the staged path), only a matrix one AOS slot holds is staged, and a
    window lists only matrices not staged NONE (moe_stream_window_items);
  - teardown gates the open region and holds one slot handle until the last use before the ring is released
    (ruling D, c-fjx1), and the ring is offered only on an in-order compute queue;
  - the backend context owns the phase, resets it after draining its queues and before the retained handles drain,
    ends the prompt at the first decode graph, and closes a layer's region after its down MUL_MAT_ID;
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

COPY_DECL = "copied = mem_copy_async("
REORDER_CALL = "ggml_sycl_reorder_weight_gpu("
RETAIN = "retain_handles_until_event({ aos, host }, copied);"
TABLE_WRITE = "table_[key] = { mem_handle(), aos, copied, GGML_LAYOUT_AOS, item.consumer };"
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


def claim_window_copy_is_the_only_submission(phase: str) -> bool:
    """W2 (Task 2b of llama.cpp-a2xb): the executor's dequant kernel reads every window matrix in place as AOS, so
    stage_window submits exactly one thing per matrix, its copy: no reorder and no kernel of its own. The phase submits
    no reorder anywhere: the SOA staging that ggml_sycl_reorder_weight_gpu served is retired."""
    n = norm(phase)
    s = body(n, STAGE_SIG)
    return (bool(s) and s.count("mem_copy_async(") == 1 and REORDER_CALL not in n
            and ".submit(" not in s and "->submit(" not in s)


def test_window_copy_is_the_only_submission():
    assert claim_window_copy_is_the_only_submission(PHASE)


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
    assert claim_window_entry_reads_in_place(PHASE)


def test_window_entry_mutants():
    c = claim_window_entry_reads_in_place
    # The host lease left to the table's hold.
    assert not c(mutate(PHASE, RETAIN, "retain_handles_until_event({ aos }, copied);"))
    # The slot left to the table's hold.
    assert not c(mutate(PHASE, RETAIN, "retain_handles_until_event({ host }, copied);"))
    # Retained until the region's gate, which precedes the copy.
    assert not c(mutate(PHASE, RETAIN, "retain_handles_until_event({ aos, host }, ring_->current_region_gate());"))
    # Tagged SOA: a reader would read the slot as a layout it does not hold.
    assert not c(mutate(PHASE, TABLE_WRITE, TABLE_WRITE.replace("GGML_LAYOUT_AOS", "GGML_LAYOUT_SOA")))
    # A SOA slot named that no reorder wrote.
    assert not c(mutate(PHASE, TABLE_WRITE, "table_[key] = { aos, aos, copied, GGML_LAYOUT_AOS, item.consumer };"))
    # The entry's event the region's gate instead of the copy.
    assert not c(mutate(PHASE, TABLE_WRITE,
                        "table_[key] = { mem_handle(), aos, ring_->current_region_gate(), GGML_LAYOUT_AOS, "
                        "item.consumer };"))
    # The consumer dropped: the entry would read as the staged route's.
    assert not c(mutate(PHASE, TABLE_WRITE, "table_[key] = { mem_handle(), aos, copied, GGML_LAYOUT_AOS };"))


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
    assert not claim_phase_never_waits(mutate(PHASE, RETAIN, RETAIN + " copied.wait();"))
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
        mutate(PHASE, RETAIN, RETAIN + " (void) copied.get_info<sycl::info::event::command_execution_status>();"))
    assert not claim_phase_queries_no_status_and_casts_no_const(
        mutate(PHASE, RETAIN, RETAIN + " (void) const_cast<moe_stream_ring *>(ring_.get());"))


ALLOCATOR_NAMES = ("malloc", "unified_alloc", "unified_allocate", "zone_allocate", "from_owned_alloc", "pinned_pool")


def claim_phase_allocates_only_ring_slots(phase: str) -> bool:
    """No allocator: the one slot each window matrix takes is ring_->acquire_slot(GGML_LAYOUT_AOS), from the ring's one
    reservation."""
    n = norm(phase)
    return bool(n) and not any(a in n for a in ALLOCATOR_NAMES) and n.count("ring_->acquire_slot(") == 1


def test_phase_allocates_only_ring_slots():
    assert claim_phase_allocates_only_ring_slots(PHASE)


def test_allocator_mutants():
    assert not claim_phase_allocates_only_ring_slots(
        mutate(PHASE, RETAIN, RETAIN + " (void) sycl::malloc_device(16, *ccs_);"))
    assert not claim_phase_allocates_only_ring_slots(
        mutate(PHASE, "ring_->acquire_slot(GGML_LAYOUT_AOS)", "mem_handle()"))


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
    assert claim_only_direct_owned_typed_items_stream(PHASE, PLAN_HPP)


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
AOS_ACQUIRE = "mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_AOS);"
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
    assert claim_window_matrix_takes_one_aos_slot(PHASE, PLAN_HPP)


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
    assert claim_region_closes_after_last_copy(PHASE)


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
    assert claim_reuse_waits_for_window_readers(PHASE)


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
    assert claim_staging_from_the_claim_alone(SYCL, PLAN_HPP, PHASE)


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
    assert claim_staging_mirrors_serving(SYCL, BATCH_HPP)


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
    assert claim_claimed_weight_stages_for_the_executor(SYCL, PLAN_HPP, PHASE)


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
    assert claim_library_claims_read_the_index_plan(SYCL)


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


def test_staging_follows_the_decision():
    assert claim_staging_follows_the_decision(PHASE, SYCL, PLAN_HPP)


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
    assert claim_window_decline_reaches_the_executor(PHASE)


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
    assert claim_ubatch_resets_the_window_state(PHASE)


def test_ubatch_reset_mutants():
    c = claim_ubatch_resets_the_window_state
    # The prefetch id kept across the boundary.
    assert not c(mutate(PHASE, "next_window_id_ = UINT32_MAX;", ""))
    # The prefetch id and its whole flag kept.
    assert not c(mutate(PHASE, "next_window_id_ = UINT32_MAX; next_window_whole_ = false;", ""))
    # The last window's layer kept, so a micro-batch whose first window is not UP phase 0 (a layer whose UP role has
    # no host matrix) reads it as prefetched and counts its copies as misses.
    assert not c(mutate(PHASE, "prev_window_layer_ = -1;", ""))


UBATCH_PRED_SIG = "static inline bool moe_stream_ubatch_starts("
EXPOSED_PRED_SIG = "static inline bool moe_stream_window_exposed("
WINDOW_ID_SIG = "static inline uint32_t moe_stream_window_id("
UBATCH_PRED_CALL = ("const bool new_ubatch = moe_stream_ubatch_starts(ubatch_live_, layer, cur_layer_, op.graph_seq, "
                    "cur_graph_seq_);")
EXPOSED_CALL = ("const bool exposed = moe_stream_window_exposed(req.layer, req.role, req.phase, prev_window_layer_);")
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
    assert claim_phase_decides_through_tested_predicates(PHASE, PLAN_HPP)


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
    assert claim_phase_allocates_only_ring_slots(PHASE)
    assert claim_staging_follows_the_decision(PHASE, SYCL, PLAN_HPP)
    # The seams outside their block: a second slot acquisition and a second ring_ready() call, both counted.
    unguarded = production(raw.replace(guard, "#if 1"))
    assert not claim_phase_allocates_only_ring_slots(unguarded)
    assert not claim_staging_follows_the_decision(unguarded, SYCL, PLAN_HPP)
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


# ---- a short reserve asks for a loan (llama.cpp-3460, c-0tmn) --------------------------------------------------------

RING_CPP_REL = "ggml/src/ggml-sycl/moe-stream-ring.cpp"
RING_HPP_REL = "ggml/src/ggml-sycl/moe-stream-ring.hpp"
RING_CPP = read(RING_CPP_REL)
RING_HPP = read(RING_HPP_REL)

# The reserve-failed branch's condition, without its brace: body() opens at the first brace after it.
RESERVE_SHORT_IF = "if (!ring->reserve(device_, geo_, &bytes_short))"
LOAN_CALL = "moe_stream_loan_on_short_reserve(loan_latch_, device_, bytes_short, request_loan_, loan_refusals_)"
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
    "latch.asked = true; latch.refusals_seen = refusals_now; request_loan(device, bytes_short); "
    "return moe_stream_loan_step::REQUEST;",
)
RESERVE_DECL = "bool reserve(int device, const moe_stream_ring_geometry & geo, size_t * bytes_short = nullptr);"
RESERVE_DEF_SIG = ("bool moe_stream_ring::reserve(int device, const moe_stream_ring_geometry & geo, "
                   "size_t * bytes_short)")
RESERVE_SHORT_SET = ("if (bytes_short != nullptr) { "
                     "*bytes_short = moe_stream_ring_weight_shortfall(device, geo.bytes); }")
WEIGHT_REFUSAL_WARN = "[MOE-STREAM-RING] reserve: the WEIGHT zone cannot give"
SHORTFALL_SIG = "size_t moe_stream_ring_weight_shortfall(int device, size_t bytes)"
SHORTFALL_PARTS = (
    "cache->zone_free_figures(vram_zone_id::WEIGHT, capacity, used, available, largest_free)",
    "return bytes > largest_free ? bytes - largest_free : 0;",
)
REFUSALS_DECL = "static std::atomic<uint64_t> g_sycl_expert_loan_refusals[GGML_SYCL_MAX_DEVICES];"
REFUSALS_BUMP = ("if (code == ggml_sycl_txn_result::REFUSED) { "
                 "g_sycl_expert_loan_refusals[device].fetch_add(1, std::memory_order_release); }")
RUN_PENDING_SIG = "static bool ggml_sycl_run_pending_expert_loan_on_device("
RUN_PENDING_TXN = "ggml_sycl_expert_loan_transaction_run(device, bytes_wanted, false, resolver, result, false);"
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
    that moved since the last ask latches, and only end_prompt clears the latch. The reserve's shortfall is the ring's
    bytes past the WEIGHT zone's largest free block, set where the zone refuses the ring."""
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
            and h.count("request_loan(device, ") == 1
            and ordered(body(h, LOAN_HELPER_SIG), *LOAN_HELPER_ORDER)
            and RESERVE_DECL in norm(ring_hpp)
            and ordered(body(rc, RESERVE_DEF_SIG), "allocation = unified_allocate_owner(req);", RESERVE_SHORT_SET,
                        WEIGHT_REFUSAL_WARN)
            and all(part in body(rc, SHORTFALL_SIG) for part in SHORTFALL_PARTS)
            and s.count(REFUSALS_DECL) == 1
            and s.count("g_sycl_expert_loan_refusals[device].fetch_add(") == 1
            and ordered(body(s, RUN_PENDING_SIG), RUN_PENDING_TXN, REFUSALS_BUMP, RUN_PENDING_BUSY)
            and REFUSALS_LOAD in body(s, REFUSALS_ACCESSOR_SIG)
            and s.count(WIRE_REQUEST) == 1 and WIRE_REQUEST in body(s, STREAM_PREPARE_SIG)
            and s.count(WIRE_REFUSALS) == 1 and WIRE_REFUSALS in body(s, STREAM_PREPARE_SIG))


def test_short_reserve_requests_a_loan():
    assert claim_short_reserve_requests_a_loan(PHASE, PLAN_HPP, SYCL, RING_CPP, RING_HPP)


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
    assert not c(phase=mutate(PHASE, "device_, bytes_short, request_loan_", "device_, geo_.bytes, request_loan_"))
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
    assert not c(hpp=mutate(PLAN_HPP, "request_loan(device, bytes_short);", "request_loan(device, SIZE_MAX);"))
    assert not c(hpp=mutate(PLAN_HPP, "request_loan(device, bytes_short);",
                            "request_loan(device, bytes_short); request_loan(device, bytes_short);"))
    # The reserve exposes no shortfall, or exposes the whole ring.
    assert not c(ring_hpp=mutate(RING_HPP, RESERVE_DECL,
                                 "bool reserve(int device, const moe_stream_ring_geometry & geo);"))
    assert not c(ring_cpp=mutate(RING_CPP, RESERVE_SHORT_SET, ""))
    assert not c(ring_cpp=mutate(RING_CPP, "return bytes > largest_free ? bytes - largest_free : 0;",
                                 "return bytes;"))
    # The refusal counted on BUSY too (the runner re-posts BUSY itself), or never counted.
    assert not c(sycl=mutate(SYCL, "if (code == ggml_sycl_txn_result::REFUSED) {",
                             "if (code != ggml_sycl_txn_result::ACCEPTED) {"))
    assert not c(sycl=mutate(SYCL, REFUSALS_BUMP, ""))
    # The hooks not wired by the backend.
    assert not c(sycl=mutate(SYCL, WIRE_REQUEST, ""))
    assert not c(sycl=mutate(SYCL, WIRE_REFUSALS, ""))


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
    assert claim_refused_loan_warns_once(PHASE, SYCL_TREE)


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
    "geo.n_slots == 0", '"ring exceeds 10% of resident expert bytes"',
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
    assert claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, SYCL_TREE)


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
    # The window rule dropped from the helper, or asked after the 10% bound it would otherwise be reported as.
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, "if (idx.window_slots == 0) { return \"no executor window\"; }", ""), SYCL_TREE)
    tree = dict(SYCL_TREE)
    tree["extra.cpp"] = '"' + DIAG_FMT + '"'
    assert not claim_ring_unavailable_diagnostic(PHASE, PLAN_HPP, tree)
    assert not claim_ring_unavailable_diagnostic(mutate(PHASE, DIAG_EMIT, "(void) line;"), PLAN_HPP, SYCL_TREE)
    assert not claim_ring_unavailable_diagnostic(
        PHASE, mutate(PLAN_HPP, '"reserve refused"', '"reserve failed"'), SYCL_TREE)
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
    assert claim_auto_still_reports(PHASE)


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
    assert claim_begin_layer_weighs_only_its_layer(PHASE, PLAN_HPP)


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
    """The load path's reorder dispatch is defined once in ggml-sycl.cpp without `static` and declared in
    moe-stream-plan.hpp, which ggml-sycl.cpp includes before the definition (-Wmissing-declarations), and the load-time
    fill still calls it. The ring no longer does (test_window_copy_is_the_only_submission)."""
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
    assert claim_trace_line(PHASE, SYCL_TREE)


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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
