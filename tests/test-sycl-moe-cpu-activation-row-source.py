#!/usr/bin/env python3
"""Host MoE experts must read the activation row of their own slot (llama.cpp-4hg7).

THE DEFECT.  GPT-OSS 20B on the B50 with GGML_SYCL_VRAM_BUDGET_PCT=60 places the
experts of layers 12-22 on the host, and decode then emits garbage that is
byte-identical from run to run.  The same command at full budget is coherent.

The CPU-TG route in ggml_sycl_mul_mat_id (ggml-sycl.cpp, the dispatch_cpu_compute
lambda) had a "every expert shares one activation" shortcut keyed on
cpu_expert_tg_active alone:

  * a single D2H copied K floats from offset 0 of src1 into the staging buffer, and
  * every task's act_host pointed at that one copy (and the CPU batch then
    deduplicated the Q8 quantization on act_host pointer equality).

That is true of GATE and UP, whose src1 is [n_embd, 1, n_tokens] (ne11 == 1).  It is
false of DOWN, whose src1 is [n_ff, n_used, n_tokens]: ne11 == n_used (4 on
GPT-OSS), one activation row per expert slot.  So every host expert in the down
projection multiplied row 0 instead of the row of its own slot.  It only shows
when some experts are on the host, which is why full budget never saw it.  The
per-expert branch (row i11 = entry.id % ne11, destination ci * K) was already right;
it simply was not reached.

THE CONTRACT.  There is ONE fact -- the dispatch has a single activation row iff
ne11 == 1 -- and ONE bool carrying it, cpu_shared_act = cpu_expert_tg_active &&
ne11 == 1.  Inside dispatch_cpu_compute (comments blanked):

  1. cpu_shared_act is defined exactly once, as cpu_expert_tg_active && ne11 == 1;
  2. cpu_expert_tg_active is not read there at all -- every "shared" decision goes
     through cpu_shared_act, so there is no second source for the fact;
  3. the single-D2H branch is conditioned on cpu_shared_act (and n_cpu > 1);
  4. the shared pointer choice is conditioned on cpu_shared_act at BOTH task
     builders (task.activations for the Q1_0/NVFP4 recipe path, t.act_host);
  5. the per-expert fall-through survives: it copies row (entry.id % ne11) of token
     entry.iid1 to ci * K, and both builders still fall back to act_pinned + ci * K.  Since
     llama.cpp-yx28 the copy is gather_activation_rows, one copy per contiguous source run;
     moe_gather_runs_build (moe-decode-hostpath.hpp) puts row i at dst_base + i * row_bytes.

SECOND CONTRACT (the hot/cold hazard, same ticket).  The synchronous CPU dispatch
(do_cpu_dispatch) splits the host experts into hot and cold groups.  The hot group's
scatter is published by flush_pending_cpu_scatter(), which submits the H2D copies from
the PinnedBufferPool out buffer ASYNCHRONOUSLY.  acquire() returns one fixed pair, so
the cold dispatch that follows took the SAME out region, zeroed it on the host and had
its CPU kernels write it -- before the hot H2D had executed.  The fix is not a host
wait (the owner rule is no host waits; an earlier commit, 4d70bdea4, waited on the
scatter events and was rejected for that reason).  It is DISJOINT REGIONS:

  6. PinnedBufferPool::reserve(n) is a ring over max_experts_ entries (wraps to 0 when
     the span would run past the end); dispatch_cpu_compute takes a first entry and
     addresses act/out at acquire().{act,out} + first * {K,N}, and every offset that
     names the same slot -- the single-D2H destination, the per-expert D2H destination
     and the scatter entry's src_offset -- carries the same first entry;
  7. do_cpu_dispatch reserves ONE span for hot+cold, hands the hot group the start and
     the cold group start + hot count, and only splits when can_serve(total) -- else
     everything is one dispatch.  When the rows are per-slot (not cpu_shared_act) it gathers
     hot then cold rows once, at the start of that span, and both dispatches skip their own
     gather (llama.cpp-yx28);
  8. the out-region memset is not before the activation D2H wait (that wait, on the
     in-order queue, is what proves an earlier scatter's H2D from this region has run).
     Since llama.cpp-z4kd both run on the pool worker, in the job's prologue, not on the
     submitting thread;
  9. no host wait is (re)introduced in do_cpu_dispatch;
 10. CROSS-OP safety is by ORDERING, not by the ring (entry offsets scale with each op's own K/N
     and a top-K pool restarts at entry 0, so different ops DO alias).  The premises pinned since
     llama.cpp-z4kd moved the waits off the submitting thread: the job's prologue
     (moe_cpu_job_wait_inputs) waits every input event and only then zeroes the out region, and
     dispatch_cpu_compute zeroes nothing itself (pin W1: with an extra conjunct on the wait, or the
     memset first, the zeroing and the CPU read run before the D2H lands); every job form -- the
     CpuExpertPool worker and both std::async forms -- runs that prologue before its kernel, and the
     worker runs a job's resolve step, which builds the tasks of a job submitted before its expert ids
     reached the host, after the prologue and before the kernel (pin W2, llama.cpp-7pvo);
     every activation-copy branch hands its copy events, and every lease producer its ready event,
     to the prologue, unconditionally (pin W4); a scatter flushed AFTER the copy is ordered by
     moe_cpu_job_order_decide, whose WAIT_LAST_FLUSH / WAIT_QUEUE answers add the last flush's events
     or a queue barrier to the job's inputs (pin O1); and the op-entry flushes --
     flush_pending_cpu_scatter_if_consumed, then flush_pending_cpu_scatter -- precede the shared
     activation D2H (pins F1/F3: the earlier scatter's H2D must be enqueued before this op's D2H;
     F2 pinned the opt-in pipeline slot's flush between them, and that slot was removed by
     llama.cpp-z4kd).  Three more pins
     close the ways a pinned statement can be present and still not run: W3 (the memset's guard is
     defined exactly once as `from_pool || !out_owner.valid()`, so `= false` cannot drop the zeroing,
     and the zeroing request it guards covers all n_cpu * N floats),
     and, for W2, W3, O1, F1 and F3, a structural test of each pinned statement -- it follows a statement or
     block boundary (`;`, `{`, `}`), so it is not the body of an `if (...)`/`else`/loop header, and
     the innermost block that holds it is the expected one (the lambda body for the prologue, the
     zeroing request and the order decision, `if (moe_hybrid_with_plan)` for the three flushes), so
     it is not wrapped in a block of its own.
     These are TEXTUAL and STRUCTURAL pins (brace and statement adjacency), not control-flow analysis:
     a conditional hidden behind a macro or a helper function, or an early `return` above the
     statement, is outside what they can see.

11. A PENDING SCATTER IS NEVER OVERWRITTEN (llama.cpp-3bww).  g_pending_scatter holds ONE deferred
     CPU result.  Bias-less MoE (qwen2moe/qwen3moe/qwen3vlmoe/qwen3next) runs the up MUL_MAT_ID and
     then the gate MUL_MAT_ID with no ADD_ID between them, so gate does not consume up's output.
     The op-entry flush used to be a non-blocking try (flush only if the CPU future was already
     ready); with up's CPU compute still running, gate's apply_cpu_result_to_scatter overwrote the
     pending state and up's host-expert rows were never H2D-scattered.  Pinned: the third op-entry
     flush (F3) is the UNCONDITIONAL `flush_pending_cpu_scatter();`, and
     apply_cpu_result_to_scatter opens, right after its validity check, with
     `GGML_ASSERT(!g_pending_scatter.active ...)` (pin A1), so a path that reaches it with an
     unflushed scatter aborts instead of dropping rows.  The flush has to be at op ENTRY, not in
     apply: by apply time this op's CPU kernels have already written the pool bytes that the
     pending H2D has not read yet (the ring is disjoint only within one op), and the pending CPU
     workers read the shared activation staging buffer that this op's D2H rewrites.  Pin A2: the
     only other writer of `g_pending_scatter.active = true` (the direct CPU-TG path) flushes first.
     Pin A3 is retired: it held the opt-in pipeline slot (GGML_SYCL_PIPELINE_CPU=1), which had the
     same overwrite, to the same assertion until llama.cpp-ytc9; llama.cpp-z4kd removed that slot.

     THE yx28 JOIN (llama.cpp-yx28, 7ad9eb21a; lead ruling on llama.cpp-p6zh).  Decode up after gate
     reuses gate's activation copy and makes no D2H, so the third op-entry flush is skipped exactly
     when `defer_entry_flush` holds (pin F3: the flush is the body of `if (!defer_entry_flush)`, and
     that guard is itself unconditional in `if (moe_hybrid_with_plan)`).  The decision moves to the
     join, ahead of dispatch_cpu_compute's pool writes: J1, the deferred path asks
     moe_sibling_pending_keep, which refuses unless the staging serial is unchanged and, when both
     sides use the pool, their spans are disjoint; on keep, gate's job moves to the sibling slot.
     J2, otherwise gate is flushed before the dispatch_cpu_compute call that writes up's pool
     entries.  Until llama.cpp-z4kd the join then waited both slots' scatter_events on the host;
     now the flush only enqueues, and up's order decision (O1) sees the scatter serial move past its
     copy's and makes up's job wait for gate's scatter on the pool worker when the spans overlap.  J3, every flush drains the older
     (sibling) slot first, and a consumed slot of either kind flushes both.  A1 is unchanged and
     still makes an unflushed primary slot at apply time an abort.

WHAT THIS DOES NOT PROVE.  It reads source text.  It shows the shortcut can no longer
be taken when ne11 > 1 and that the per-expert path it falls to is still the
row-selecting one.  It does not show that the numbers are right, that the CPU kernel
consumes the staged rows correctly, or that decode is coherent -- that needs the GPU
repro (GGML_SYCL_VRAM_BUDGET_PCT=60 on the B50), which is run by the lead.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real source and
requires each to FAIL, then requires the unmodified tree to pass.  Run it that way
(it is how ctest registers it): a text check whose anchors stop matching passes
vacuously, and the mutants prove each anchor is still load-bearing.

THE RED FOR THE SECOND CONTRACT (against 4d70bdea4, which waited instead):
  FAIL: do_cpu_dispatch ... (llama.cpp-4hg7)

THE ORIGINAL RED, recorded before the fix (against master d8a67422d):
  FAIL: cpu_shared_act is not defined exactly once, ahead of dispatch_cpu_compute, as
  `cpu_expert_tg_active && ne11 == 1` in ggml_sycl_mul_mat_id (llama.cpp-4hg7)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

LAMBDA_MARKER = "auto dispatch_cpu_compute = [&]("
GATHER_MARKER = "auto gather_activation_rows = [&]("
HOSTPATH_HPP = ROOT / "ggml/src/ggml-sycl/moe-decode-hostpath.hpp"
SHARED = "cpu_shared_act"
TG_ACTIVE = "cpu_expert_tg_active"

DEFINITION = re.compile(r"const\s+bool\s+cpu_shared_act\s*=\s*cpu_expert_tg_active\s*&&\s*ne11\s*==\s*1\s*;")


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and newlines."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def brace_block_from(src: str, start: int) -> str:
    """The `{ ... }` block whose opening brace is the first `{` at or after `start`."""
    open_at = src.find("{", start)
    if open_at < 0:
        raise ContractError("FAIL: expected a brace block and found none")
    depth = 0
    for k in range(open_at, len(src)):
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[open_at : k + 1]
    raise ContractError("FAIL: brace block is unbalanced")


def squash(text: str) -> str:
    """Collapse whitespace, and drop it after `(` so a wrapped call reads like an unwrapped one."""
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", text))


def statement_context(text: str, pos: int) -> tuple[str, int, str]:
    """Where the statement starting at `pos` sits: (previous token char, enclosing `{` index, its header).

    Structural, not control-flow analysis.  The previous non-space character is `;`, `{` or `}` for a
    statement that is not the body of an `if (...)`/`else`/loop header (those leave `)` or a keyword's
    last letter there).  The enclosing block is the innermost unclosed `{`; its header is the text since
    the previous `;`, `{` or `}`, so a statement wrapped in `if (false) { ... }` reports that header.
    """
    prev = text[:pos].rstrip()[-1:]
    depth = 0
    k = pos - 1
    while k >= 0:
        c = text[k]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                break
            depth -= 1
        k -= 1
    j = k - 1
    while j >= 0 and text[j] not in ";{}":
        j -= 1
    return prev, k, " ".join(text[j + 1 : k].split()) if k >= 0 else ""


def require_unconditional(text: str, pos: int, *, header: str | None, pin: str, what: str) -> None:
    """The statement at `pos` must follow a statement/block boundary and sit in the expected block.

    header=None means the enclosing block must be the text's own outermost block (index 0).
    """
    prev, block_at, block_header = statement_context(text, pos)
    where_ok = (block_at == 0) if header is None else (block_header == header)
    if prev not in (";", "{", "}") or not where_ok:
        raise ContractError(
            f"FAIL [pin {pin}]: {what} is not an unconditional statement of its block (preceded by {prev!r}, "
            f"enclosing block `{block_header}`): it is the body of an if/else/loop header, or wrapped in a block "
            "of its own, so the ordering argument does not hold on every path (llama.cpp-4hg7)"
        )


def check(backend_src: str, pool_src: str | None = None, hostpath_src: str | None = None,
          expert_pool_src: str | None = None) -> None:
    code = blank_comments(backend_src)
    hostpath_src = hostpath_src if hostpath_src is not None else HOSTPATH_HPP.read_text()

    # 1. one definition of the fact.  It must sit before the lambda (the lambda captures it).
    lam_at = code.find(LAMBDA_MARKER)
    if lam_at < 0:
        raise ContractError("FAIL: dispatch_cpu_compute lambda not found in ggml-sycl.cpp (llama.cpp-4hg7)")
    defs = list(DEFINITION.finditer(code))
    if len(defs) != 1 or defs[0].start() > lam_at:
        raise ContractError(
            "FAIL: cpu_shared_act is not defined exactly once, ahead of dispatch_cpu_compute, as "
            "`cpu_expert_tg_active && ne11 == 1` in ggml_sycl_mul_mat_id (llama.cpp-4hg7)"
        )

    body = squash(brace_block_from(code, lam_at))

    # 2. the lambda never consults cpu_expert_tg_active directly.
    if re.search(r"\b" + TG_ACTIVE + r"\b", body):
        raise ContractError(
            "FAIL: dispatch_cpu_compute reads cpu_expert_tg_active directly, so the 'all experts share one "
            "activation' decision has a second source that ignores ne11 (llama.cpp-4hg7)"
        )

    # 3. the single-D2H branch.
    single = re.search(r"else if \(\s*cpu_shared_act\s*&&\s*n_cpu\s*>\s*1\s*\)", body)
    if not single:
        raise ContractError(
            "FAIL: the single-D2H branch (one activation copied for all host experts) is not conditioned on "
            "`cpu_shared_act && n_cpu > 1`, so with ne11 > 1 every host expert reads row 0 (llama.cpp-4hg7)"
        )

    # 4. both task builders choose the shared pointer on cpu_shared_act only.
    for what, lhs in (("task.activations", r"task\.activations"), ("t.act_host", r"t\.act_host")):
        pat = re.compile(
            lhs + r"\s*=\s*cpu_shared_act\s*\?\s*\(\s*act_on_host\s*\?\s*shared_act_host\s*:\s*act_pinned\s*\)\s*:\s*"
            r"act_pinned\s*\+\s*ci\s*\*\s*static_cast<size_t>\(K\)\s*;"
        )
        if not pat.search(body):
            raise ContractError(
                f"FAIL: {what} does not pick the shared activation on `cpu_shared_act` with an "
                "`act_pinned + ci * K` per-expert fallback (llama.cpp-4hg7)"
            )

    # 5. the per-expert copy selects the slot's row, of the entry's token, into slot ci.  Since
    #    llama.cpp-yx28 dispatch_cpu_compute hands its entries, in order, to gather_activation_rows at
    #    slot pool_base, unless the caller already gathered them (act_pregathered; pinned in S3).
    if not re.search(
        r"\} else if \(!act_pregathered\) \{ std::vector<const expert_dispatch_entry \*> rows; rows\.reserve\(n_cpu\); "
        r"for \(const expert_dispatch_entry & entry : entries\) \{ rows\.push_back\(&entry\); \} "
        r"job_inputs\.events = gather_activation_rows\(rows, act_handle, pool_base \* static_cast<size_t>\(K\) \* sizeof\(float\)\); \}",
        body,
    ):
        raise ContractError(
            "FAIL: dispatch_cpu_compute's per-expert branch no longer gathers its own entries, in order, into slots "
            "pool_base + ci (llama.cpp-4hg7)"
        )
    check_gather(code, lam_at, hostpath_src)

    check_hot_cold(code, hostpath_src, expert_pool_src if expert_pool_src is not None else EXPERT_POOL_CPP.read_text())
    check_pending_scatter(code)
    check_pool_ring(pool_src if pool_src is not None else POOL_CPP.read_text(), hostpath_src)


def check_gather(code: str, lam_at: int, hostpath_src: str) -> None:
    """Pin 5's row selection, where llama.cpp-yx28 moved it: the gather helper and its run builder."""
    g_at = code.find(GATHER_MARKER)
    if g_at < 0 or code.count(GATHER_MARKER) != 1 or g_at > lam_at:
        raise ContractError(
            "FAIL: gather_activation_rows is not defined exactly once ahead of dispatch_cpu_compute -- renamed or "
            "moved? (llama.cpp-4hg7)"
        )
    g = squash(brace_block_from(code, g_at))
    row = (r"for \(const expert_dispatch_entry \* entry : rows\) \{ src_offsets\.push_back\(src1_storage\.view_offset "
           r"\+ static_cast<size_t>\(entry->id % ne11\) \* nb11 \+ static_cast<size_t>\(entry->iid1\) \* nb12\); \}")
    runs = r"moe_gather_runs_build\(src_offsets, act_first_byte, static_cast<size_t>\(K\) \* sizeof\(float\), runs\);"
    copy = r"mem_copy_async\(act_handle, run\.dst_offset, src1_storage\.handle, run\.src_offset, run\.bytes, \*stream\)"
    if not (re.search(row, g) and re.search(runs, g) and re.search(copy, g)):
        raise ContractError(
            "FAIL: the per-expert D2H no longer copies row (entry.id % ne11) of token entry.iid1 into slot "
            "ci * K (llama.cpp-4hg7)"
        )
    hp = squash(blank_comments(hostpath_src))
    b_at = hp.find("inline void moe_gather_runs_build(")
    if b_at < 0:
        raise ContractError("FAIL: moe_gather_runs_build is not defined in moe-decode-hostpath.hpp -- renamed or moved? (llama.cpp-4hg7)")
    build = brace_block_from(hp, b_at)
    for needle in (
        "const size_t dst = dst_base + i * row_bytes;",
        "last.src_offset + last.bytes == src_offsets[i] && last.dst_offset + last.bytes == dst",
        "run.src_offset = src_offsets[i]; run.dst_offset = dst; run.bytes = row_bytes;",
    ):
        if needle not in build:
            raise ContractError(
                f"FAIL: moe_gather_runs_build no longer puts gathered row i at dst_base + i * row_bytes (missing "
                f"`{needle}`), so slot ci would not hold its own row (llama.cpp-4hg7)"
            )


DISPATCH_MARKER = "auto do_cpu_dispatch = [&]("
POOL_CPP = ROOT / "ggml/src/ggml-sycl/pinned-buffer-pool.cpp"
EXPERT_POOL_CPP = ROOT / "ggml/src/ggml-sycl/cpu-expert-pool.cpp"
WAIT_INPUTS_DEF = "static void moe_cpu_job_wait_inputs(const moe_cpu_job_inputs & inputs) "
PROLOGUE_DEF = "static std::function<void()> moe_cpu_job_prologue(moe_cpu_job_inputs inputs) "
SLOT_FLUSH_DEF = "static void flush_pending_cpu_scatter_slot(pending_cpu_scatter & slot) "


def check_pool_ring(pool_src: str, hostpath_src: str) -> None:
    code = blank_comments(pool_src)
    at = code.find("PinnedBufferPool::reserve(")
    if at < 0:
        raise ContractError("FAIL: PinnedBufferPool::reserve() is not defined (llama.cpp-4hg7)")
    body = squash(brace_block_from(code, at))
    # The wrap lives in moe_pool_reserve_first since llama.cpp-yx28 (the sibling-slot decision predicts a
    # reservation with the same function); reserve() must take its first entry from it.
    hp = squash(blank_comments(hostpath_src))
    w_at = hp.find("inline size_t moe_pool_reserve_first(size_t cursor, size_t n, size_t capacity)")
    wrap = brace_block_from(hp, w_at) if w_at >= 0 else ""
    if (
        "const size_t first = ggml_sycl::moe_pool_reserve_first(next_entry_, n_experts, max_experts_);" not in body
        or wrap != "{ return cursor + n > capacity ? 0 : cursor; }"
    ):
        raise ContractError("FAIL: PinnedBufferPool::reserve() does not wrap to entry 0 past the end (llama.cpp-4hg7)")
    if not re.search(r"next_entry_\s*=\s*\(\s*first\s*\+\s*n_experts\s*\)\s*%\s*max_experts_", body):
        raise ContractError("FAIL: PinnedBufferPool::reserve() does not advance its cursor past the span (llama.cpp-4hg7)")


APPLY_MARKER = "auto apply_cpu_result_to_scatter = [&]("


def check_pending_scatter(code: str) -> None:
    """Pin A1/A2: g_pending_scatter is never assigned while it still holds an unflushed scatter."""
    ap_at = code.find(APPLY_MARKER)
    if ap_at < 0 or code.count(APPLY_MARKER) != 1:
        raise ContractError("FAIL [pin A1]: apply_cpu_result_to_scatter lambda not found exactly once (llama.cpp-3bww)")
    apply_body = squash(brace_block_from(code, ap_at))
    guard = "if (!r.valid) { return; } GGML_ASSERT(!g_pending_scatter.active &&"
    guard_at = apply_body.find(guard)
    first_write = apply_body.find("g_pending_scatter.future =")
    if guard_at < 0 or first_write < 0 or guard_at > first_write:
        raise ContractError(
            "FAIL [pin A1]: apply_cpu_result_to_scatter does not open with `GGML_ASSERT(!g_pending_scatter.active "
            "&& ...)` straight after its validity check and before its first write to g_pending_scatter; a "
            "still-pending scatter would be overwritten and its host-expert rows never reach the device "
            "(llama.cpp-3bww)"
        )
    require_unconditional(apply_body, apply_body.find("GGML_ASSERT(!g_pending_scatter.active"),
                          header=None, pin="A1", what="the pending-scatter assertion in apply_cpu_result_to_scatter")

    # A2: every other writer of `active = true` flushes first.  A new writer must be added here on purpose.
    writers = [m.start() for m in re.finditer(r"g_pending_scatter\.active\s*=\s*true\s*;", code)]
    if len(writers) != 2:
        raise ContractError(
            f"FAIL [pin A2]: g_pending_scatter.active is set true at {len(writers)} sites, expected 2 (the hybrid "
            "apply lambda and the direct CPU-TG path); a new writer needs its own flush-before-write proof "
            "(llama.cpp-3bww)"
        )
    direct_at = code.find("auto dispatch_cpu_entries_now = [&](")
    if direct_at < 0:
        raise ContractError("FAIL [pin A2]: the direct CPU-TG dispatch_cpu_entries_now lambda was not found (llama.cpp-3bww)")
    direct = squash(brace_block_from(code, direct_at))
    # llama.cpp-yx28 added a sibling pending slot; the direct path flushes when either is active.
    if not re.search(
        r"if \(entries\.empty\(\)\) \{ return; \} "
        r"if \(g_pending_scatter\.active \|\| g_pending_scatter_sibling\.active\) \{ flush_pending_cpu_scatter\(\); \}",
        direct,
    ):
        raise ContractError(
            "FAIL [pin A2]: dispatch_cpu_entries_now no longer flushes an active pending scatter before it writes "
            "its own (llama.cpp-3bww)"
        )


JOIN_ANCHOR = "size_t cpu_pool_first = pool_entry_npos;"


def check_sibling_join(code: str, hostpath_src: str | None = None) -> None:
    """J1-J3: what replaced the unconditional op-entry flush when it is deferred (llama.cpp-yx28)."""
    if code.count(JOIN_ANCHOR) != 1:
        raise ContractError(f"FAIL [pin J1]: the CPU join anchor `{JOIN_ANCHOR}` is not unique -- renamed or moved? (llama.cpp-yx28)")
    at = code.find(JOIN_ANCHOR)
    guard_at = code.find("if (defer_entry_flush) {", at)
    if guard_at < 0 or squash(code[at:guard_at]).strip() != JOIN_ANCHOR:
        raise ContractError("FAIL [pin J1]: the join does not open with `if (defer_entry_flush) {` (llama.cpp-yx28)")
    block_text = brace_block_from(code, guard_at)
    block = squash(block_text)
    tail = squash(code[guard_at + len("if (defer_entry_flush) ") + len(block_text):]).lstrip()
    for needle in (
        "const pending_cpu_scatter & gate = g_pending_scatter;",
        "keep.pending_active = gate.active;",
        "keep.sibling_slot_free = !g_pending_scatter_sibling.active;",
        "keep.reuses_activation = reuse_shared_act && gate.shares_activation;",
        "keep.pending_act_serial = gate.act_serial;",
        "keep.current_act_serial = g_moe_shared_act.record.serial;",
        "keep.pending_from_pool = gate.from_pool;",
        "keep.op_from_pool = op_pool;",
        "keep.pending_first = gate.pool_first;",
        "keep.pending_count = gate.pool_count;",
        "keep.op_first = op_pool ? sib_pool.reserve_peek(n_cpu_rows) : 0;",
        "keep.op_count = n_cpu_rows;",
        "keep.same_row_geometry = gate.row_k == K && gate.row_n == N;",
    ):
        if needle not in block:
            raise ContractError(f"FAIL [pin J1]: the deferred join no longer builds its keep request with `{needle}` (llama.cpp-yx28)")
    keep = (r"if \(ggml_sycl::moe_sibling_pending_keep\(keep\)\) \{ move_pending_cpu_scatter_to_sibling\(\); "
            r"if \(op_pool\) \{ cpu_pool_first = sib_pool\.reserve\(n_cpu_rows\); "
            r"GGML_ASSERT\(cpu_pool_first == keep\.op_first\); \} \} else \{")
    if not re.search(keep, block):
        raise ContractError(
            "FAIL [pin J1]: the deferred join does not keep gate's job beside this op only when "
            "moe_sibling_pending_keep(keep) accepts, moving it to the sibling slot and reserving the predicted span "
            "(llama.cpp-yx28)"
        )
    hp = squash(blank_comments(hostpath_src if hostpath_src is not None else HOSTPATH_HPP.read_text()))
    k_at = hp.find("inline bool moe_sibling_pending_keep(const moe_sibling_pending_request & r)")
    d_at = hp.find("inline bool moe_pool_spans_disjoint(size_t a_first, size_t a_count, size_t b_first, size_t b_count)")
    keep_body = brace_block_from(hp, k_at) if k_at >= 0 else ""
    disjoint_body = brace_block_from(hp, d_at) if d_at >= 0 else ""
    # The whole predicate, exactly: a needle check would not see a statement added ahead of the needles
    # (an early `return true;`), and the pool guard -- the only accept without the span check, taken only
    # when one side owns separate buffers -- is part of the contract too.
    expected_keep = (
        "{ if (!r.pending_active || !r.sibling_slot_free || !r.reuses_activation || r.op_count == 0 || "
        "r.pending_act_serial != r.current_act_serial) { return false; } "
        "if (!r.pending_from_pool || !r.op_from_pool) { return true; } "
        "return r.same_row_geometry && moe_pool_spans_disjoint(r.pending_first, r.pending_count, r.op_first, "
        "r.op_count); }"
    )
    if keep_body != expected_keep or disjoint_body != "{ return a_first + a_count <= b_first || b_first + b_count <= a_first; }":
        raise ContractError(
            "FAIL [pin J1]: moe_sibling_pending_keep is not exactly: refuse unless pending, sibling free, reusing an "
            "unrewritten staging (serial match) and non-empty; accept when one side owns separate buffers; else "
            "require equal geometry and disjoint pool spans. A changed predicate could keep a job that reads "
            "rewritten staging or shares pool bytes with this op (llama.cpp-yx28)"
        )
    if not re.search(r"\} else \{ flush_pending_cpu_scatter\(\); \} \}$", block) \
            or not tail.startswith("auto cpu_result = dispatch_cpu_compute(cpu_entries, cpu_pool_first);"):
        raise ContractError(
            "FAIL [pin J2]: when gate is not kept, the join does not flush it (enqueue its join and scatter, no "
            "host wait) before the dispatch_cpu_compute call that writes this op's pool entries; that call's order "
            "decision (pin O1) is what makes the job wait for the flushed scatter when the entries overlap "
            "(llama.cpp-yx28, llama.cpp-z4kd)"
        )
    drain = squash(brace_block_from(code, code.find("static void flush_pending_cpu_scatter() ")))
    consumed_at = code.find("static bool flush_pending_cpu_scatter_if_consumed(")
    consumed = squash(brace_block_from(code, consumed_at)) if consumed_at >= 0 else ""
    if drain != "{ flush_pending_cpu_scatter_slot(g_pending_scatter_sibling); flush_pending_cpu_scatter_slot(g_pending_scatter); }" \
            or "const pending_cpu_scatter * slots[2] = { &g_pending_scatter_sibling, &g_pending_scatter };" not in consumed \
            or not re.search(r"ggml_sycl_op_consumes_tensor\(consuming_dst, slot->dst_tensor\)\) \{ flush_pending_cpu_scatter\(\); return true; \}", consumed):
        raise ContractError(
            "FAIL [pin J3]: a CPU scatter flush no longer drains the older (sibling) slot first, or a consumed slot "
            "no longer flushes both (llama.cpp-yx28)"
        )


def check_hot_cold(code: str, hostpath_src: str | None = None, expert_pool_src: str = "") -> None:
    lam_at = code.find(LAMBDA_MARKER)
    lam = squash(brace_block_from(code, lam_at))
    head = squash(code[lam_at : code.find("{", lam_at)])

    # 6. the dispatch addresses its own slice of the ring.
    if "size_t pool_first_entry" not in head:
        raise ContractError("FAIL [pin R1]: dispatch_cpu_compute takes no pool_first_entry, so a second dispatch cannot get a disjoint region (llama.cpp-4hg7)")
    needles = {
        "if (pool.can_serve(n_cpu) && !immutable_host_recipe) {": "the pool branch is not conditioned on can_serve and a non-recipe type",
        "pool_base = pool_first_entry != pool_entry_npos ? pool_first_entry : pool.reserve(n_cpu)": "the pool entry is not the caller's slice or a fresh reservation",
        "act_pinned = bp.act + pool_base * static_cast<size_t>(K)": "act_pinned does not start at the dispatch's own pool entry",
        "out_pinned = bp.out + pool_base * static_cast<size_t>(N)": "out_pinned does not start at the dispatch's own pool entry",
        "mem_copy_async(act_handle, pool_base * static_cast<size_t>(K) * sizeof(float),": "the single-D2H destination ignores the pool entry",
        "gather_activation_rows(rows, act_handle, pool_base * static_cast<size_t>(K) * sizeof(float));": "the per-expert D2H destination ignores the pool entry",
        "(pool_base + ci) * static_cast<size_t>(N) * sizeof(float) });": "the scatter entry's src_offset ignores the pool entry",
    }
    for needle, why in needles.items():
        if needle not in lam:
            raise ContractError(f"FAIL [pin R2]: dispatch_cpu_compute: {why} (llama.cpp-4hg7)")

    # 8. (I2, llama.cpp-z4kd) the job's input waits and the out-region memset that depends on them run
    #    on the pool worker, in the job's prologue, never on the submitting thread.  W1: the prologue
    #    waits every input event and only then zeroes, and dispatch_cpu_compute never zeroes the region
    #    itself.  W2: every way the job can run (pool worker, recipe std::async, inactive-pool std::async)
    #    runs the prologue before the kernel.  W3: the zeroing is requested on every path that needs it.
    #    W4: every activation-copy branch hands its copy events to the prologue, unconditionally.
    def one_body(sig: str, pin: str) -> str:
        if code.count(sig) != 1:
            raise ContractError(f"FAIL [pin {pin}]: `{sig.strip()}` is not defined exactly once -- renamed or moved? (llama.cpp-z4kd)")
        return squash(brace_block_from(code, code.find(sig)))

    wait_body = one_body(WAIT_INPUTS_DEF, "W1")
    if wait_body != ("{ if (!inputs.events.empty()) { sycl::event::wait(inputs.events); } "
                     "if (inputs.zero_out && inputs.zero_bytes > 0) { std::memset(inputs.zero_out, 0, inputs.zero_bytes); } }"):
        raise ContractError(
            "FAIL [pin W1]: moe_cpu_job_wait_inputs is not exactly `wait every input event, then zero the output "
            "span`; zeroing before the activation copy lands can clear a region an earlier scatter's H2D has not "
            "read yet (llama.cpp-4hg7, llama.cpp-z4kd)"
        )
    if one_body(PROLOGUE_DEF, "W1") != "{ return [inputs = std::move(inputs)]() { moe_cpu_job_wait_inputs(inputs); }; }":
        raise ContractError(
            "FAIL [pin W1]: moe_cpu_job_prologue is not exactly a closure that runs moe_cpu_job_wait_inputs; the "
            "job's input waits would not run, or would run on the submitting thread (llama.cpp-z4kd)"
        )
    if re.search(r"std::memset\(", lam):
        raise ContractError(
            "FAIL [pin W1]: dispatch_cpu_compute zeroes memory itself; the out region is zeroed only by the job's "
            "prologue, after the job's input events (llama.cpp-4hg7, llama.cpp-z4kd)"
        )

    prologue_def = "std::function<void()> job_prologue = moe_cpu_job_prologue(std::move(job_inputs));"
    submits = (
        "prologue = std::move(job_prologue)]() mutable { GGML_UNUSED(workspace_owner); prologue(); "
        "for (const cpu_moe_host_aos_task & task : tasks) {",
        "result.future = cpu_pool.submit_batch(std::move(batch_tasks), std::move(job_prologue));",
        "prologue = std::move(job_prologue)]() mutable { prologue(); "
        "ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size())); }",
    )
    if lam.count(prologue_def) != 1 or lam.count("result.future = ") != 3 or any(lam.count(n) != 1 for n in submits):
        raise ContractError(
            "FAIL [pin W2]: dispatch_cpu_compute does not build one job prologue and hand it to each of its three "
            "job forms (recipe std::async, CpuExpertPool::submit_batch, inactive-pool std::async), each running it "
            "before the kernel (llama.cpp-z4kd)"
        )
    require_unconditional(lam, lam.find(prologue_def), header=None, pin="W2", what="the job prologue")
    ep = squash(blank_comments(expert_pool_src))
    push = ("work_queue_.push([this, tasks = std::move(tasks), prologue = std::move(prologue), "
            "resolve = std::move(resolve), promise, trace, t_submit]() mutable {")
    push_at = ep.find(push)
    worker = ep[push_at : ep.find("promise->set_value(); });", push_at)] if push_at >= 0 else ""
    run_prologue = "if (prologue) { prologue(); }"
    run_resolve = "if (resolve) { resolve(tasks); }"
    # The pool's kernel call runs the row loop on the pool's own row workers (llama.cpp-w4e9).
    kernel = "ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size()), row_workers_);"
    if ep.count(push) != 1:
        raise ContractError(
            "FAIL [pin W2]: CpuExpertPool::submit_batch does not queue exactly one worker lambda with the capture "
            "list `[this, tasks, prologue, resolve, promise, trace, t_submit]`; the worker needs `this` to run the "
            "row loop on the pool's row workers (llama.cpp-z4kd, llama.cpp-w4e9)"
        )
    if worker.count(kernel) != 1:
        raise ContractError(
            "FAIL [pin W2]: the CpuExpertPool worker does not make exactly one kernel call "
            f"`{kernel}`; the row loop of a pool job runs on the pool's row workers (llama.cpp-w4e9)"
        )
    if worker.count(run_prologue) != 1 or worker.find(run_prologue) > worker.find(kernel):
        raise ContractError(
            "FAIL [pin W2]: the CpuExpertPool worker does not run the batch's prologue before the kernel "
            "(llama.cpp-z4kd)"
        )
    # The resolve step reads what the prologue waited for and builds the tasks the kernel computes, so it sits
    # between the two (llama.cpp-7pvo).
    if worker.count(run_resolve) != 1 or not (worker.find(run_prologue) < worker.find(run_resolve)
                                               < worker.find(kernel)):
        raise ContractError(
            "FAIL [pin W2]: the CpuExpertPool worker does not run a job's resolve step after its prologue and "
            "before the kernel; it would read expert ids that have not landed, or build tasks the kernel never "
            "computes (llama.cpp-7pvo)"
        )
    worker_body = worker[worker.find("{"):] + "}"
    require_unconditional(worker_body, worker_body.find(run_prologue), header=None, pin="W2",
                          what="the pool worker's prologue call")
    require_unconditional(worker_body, worker_body.find(run_resolve), header=None, pin="W2",
                          what="the pool worker's resolve call")

    # The guard of the zeroing is a one-line definition; `= false` would silently drop it.
    if lam.count("const bool zero_out_after_act_wait = from_pool || !out_owner.valid();") != 1:
        raise ContractError(
            "FAIL [pin W3]: zero_out_after_act_wait is not defined exactly once as `from_pool || "
            "!out_owner.valid()` in dispatch_cpu_compute; the memset it guards is the zeroing of a region the "
            "CPU kernels accumulate into (llama.cpp-4hg7)"
        )
    zero_req = ("if (zero_out_after_act_wait) { job_inputs.zero_out = out_pinned; "
                "job_inputs.zero_bytes = n_cpu * static_cast<size_t>(N) * sizeof(float); }")
    if lam.count(zero_req) != 1 or lam.find(zero_req) > lam.find(prologue_def):
        raise ContractError(
            "FAIL [pin W3]: the job is not asked to zero all n_cpu * N floats of its out region exactly when "
            "zero_out_after_act_wait holds, before its prologue is built (llama.cpp-4hg7, llama.cpp-z4kd)"
        )
    require_unconditional(lam, lam.find(zero_req), header=None, pin="W3", what="the out-region zeroing request")

    shared = ("if (act_on_host && cpu_shared_act) { job_inputs.events.push_back(act_d2h_event); "
              "copy_serial = g_moe_shared_act.record.scatter_serial; result.act_keepalive = shared_act_handle; }")
    single = ("} else if (cpu_shared_act && n_cpu > 1) { job_inputs.events.push_back(ggml_sycl::mem_copy_async("
              "act_handle, pool_base * static_cast<size_t>(K) * sizeof(float), src1_storage.handle, "
              "src1_storage.view_offset, static_cast<size_t>(K) * sizeof(float), *stream)); }")
    pregathered = ("} else { job_inputs.events = pregather_events; copy_serial = pregather_serial; "
                   "has_activation_copy = !pregather_events.empty(); }")
    lease = "if (entry.has_ready_event) { job_inputs.events.push_back(entry.ready_event); }"
    if (lam.count("moe_cpu_job_inputs job_inputs;") != 1 or any(lam.count(n) != 1 for n in (shared, single, pregathered, lease))
            or lam.count("job_inputs.events = ") != 2):
        raise ContractError(
            "FAIL [pin W4]: not every activation-copy branch hands its copy events to the job's prologue (shared "
            "staging, single D2H, per-row gather, pregathered rows), with the copy's scatter serial, plus every "
            "lease producer event; or a later assignment overwrites them.  A job that does not wait its copy reads "
            "and zeroes before the D2H lands (llama.cpp-4hg7, llama.cpp-z4kd)"
        )

    # O1 (llama.cpp-z4kd): a scatter flushed AFTER this job's activation copy is not ordered by that copy.
    # The job asks moe_cpu_job_order_decide (moe-decode-hostpath.hpp, behaviour pinned by
    # test-sycl-moe-decode-hostpath) and adds the last flush's events, or a queue barrier, to its inputs.
    order_fields = (
        "uint64_t copy_serial = g_cpu_scatter_serial;",
        "bool has_activation_copy = true;",
        "order_req.op_from_pool = from_pool;",
        "order_req.op_first = pool_base;",
        "order_req.op_count = n_cpu;",
        "order_req.row_k = K;",
        "order_req.row_n = N;",
        "order_req.has_activation_copy = has_activation_copy;",
        "order_req.copy_scatter_serial = copy_serial;",
        "order_req.current_scatter_serial = g_cpu_scatter_serial;",
        "order_req.flush_from_pool = g_cpu_scatter_last_flush.from_pool;",
        "order_req.flush_first = g_cpu_scatter_last_flush.first;",
        "order_req.flush_count = g_cpu_scatter_last_flush.count;",
        "order_req.flush_row_k = g_cpu_scatter_last_flush.row_k;",
        "order_req.flush_row_n = g_cpu_scatter_last_flush.row_n;",
    )
    for needle in order_fields:
        if lam.count(needle) != 1:
            raise ContractError(f"FAIL [pin O1]: dispatch_cpu_compute no longer builds its order request with `{needle}` (llama.cpp-z4kd)")
    switch = ("switch (ggml_sycl::moe_cpu_job_order_decide(order_req)) { "
              "case ggml_sycl::MOE_CPU_JOB_ORDER_WAIT_LAST_FLUSH: job_inputs.events.insert(job_inputs.events.end(), "
              "g_cpu_scatter_last_flush.events.begin(), g_cpu_scatter_last_flush.events.end()); break; "
              "case ggml_sycl::MOE_CPU_JOB_ORDER_WAIT_QUEUE: job_inputs.events.push_back(stream->ext_oneapi_submit_barrier()); "
              "break; default: break; }")
    sw_at = lam.find(switch)
    if lam.count(switch) != 1 or sw_at > lam.find(prologue_def) or sw_at < lam.find(lease):
        raise ContractError(
            "FAIL [pin O1]: the job does not add the last flush's events (WAIT_LAST_FLUSH) or a queue barrier "
            "(WAIT_QUEUE) to its inputs, after its copy and lease events and before its prologue is built; its pool "
            "writes could overtake an earlier scatter H2D that reads the same entries (llama.cpp-z4kd)"
        )
    require_unconditional(lam, sw_at, header=None, pin="O1", what="the job order decision")
    sq_code = squash(code)
    flush = one_body(SLOT_FLUSH_DEF, "O1")
    recorded = (
        "g_cpu_scatter_last_flush.events = ws.events;",
        "g_cpu_scatter_last_flush.from_pool = slot.from_pool;",
        "g_cpu_scatter_last_flush.first = slot.pool_first;",
        "g_cpu_scatter_last_flush.count = slot.pool_count;",
        "g_cpu_scatter_last_flush.row_k = slot.row_k;",
        "g_cpu_scatter_last_flush.row_n = slot.row_n;",
    )
    # A flush that takes its partner along (llama.cpp-cre6) counts as one flush per slot, so the serial
    # advances once per group member, unconditionally, ahead of everything that can throw.
    serial_advance = "for (int k = 0; k < n_group; ++k) { ++g_cpu_scatter_serial; }"
    advance_at = flush.find(serial_advance)
    if (not flush.startswith("{ if (!slot.active) { return; }")
            or flush.count(serial_advance) != 1 or advance_at > flush.find("try {")
            or flush.count("const int n_group = partner ? 2 : 1;") != 1
            or any(flush.count(n) != 1 for n in recorded)):
        raise ContractError(
            "FAIL [pin O1]: a scatter flush no longer advances g_cpu_scatter_serial and records its events, span "
            "and geometry as the last flush; the order decision would read stale facts (llama.cpp-z4kd)"
        )
    # Every shared activation copy records the scatter serial it was issued at: the hybrid block's and the
    # event-chained hybrid decode route's (llama.cpp-7pvo), which share the staging. Each record is matched by its
    # own name, so one copy's serial cannot vouch for the other's.
    act_records = re.findall(r"\bg_moe_shared_act\.record = (\w+);", sq_code)
    if len(act_records) != 2 or any(
            len(re.findall(rf"(?<![\w.]){name}\.scatter_serial = g_cpu_scatter_serial;", sq_code))
            != act_records.count(name) for name in set(act_records)):
        raise ContractError(
            "FAIL [pin O1]: the hybrid block and the event-chained hybrid decode route must each publish one shared "
            f"activation record carrying the scatter serial its copy was issued at (found records {act_records}); "
            "the order decision would read a stale serial (llama.cpp-z4kd, llama.cpp-7pvo)"
        )
    require_unconditional(flush, advance_at, header=None, pin="O1", what="the flush's serial advance")

    # 9. (I2) the flushes that make the ordering argument true come before the shared D2H.
    entry_marker = '"[MoE-HYBRID] ne12=%ld hybrid_active=%d plan_hybrid=%d cpu_tg=%d expert_cache=%d\\n"'
    anchor = code.find(entry_marker)
    if code.count(entry_marker) != 1 or anchor < 0:
        raise ContractError("FAIL [pin F0]: the hybrid-op entry anchor is not unique (llama.cpp-4hg7)")
    after = squash(code[anchor:])
    d2h = after.find("act_d2h_event = ggml_sycl::mem_copy_async(shared_act_handle")
    if d2h < 0:
        raise ContractError("FAIL [pin F0]: the shared activation D2H was not found after the hybrid-op entry (llama.cpp-4hg7)")
    entry = after[:d2h]
    order = [
        ("flush_pending_cpu_scatter_if_consumed(dst, ctx.device);", "F1"),
        ("flush_pending_cpu_scatter();", "F3"),
    ]
    at = -1
    for needle, pin in order:
        # A bare call: match on a word boundary so a longer name ending in this call does not count.
        bare = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(needle))
        if len(bare.findall(entry)) != 1:
            raise ContractError(
                f"FAIL [pin {pin}]: `{needle}` must appear exactly once between the hybrid-op entry and the shared "
                "activation D2H; the earlier op's scatter H2D has to be enqueued BEFORE this op's D2H (llama.cpp-4hg7)"
            )
        nxt = bare.search(entry).start()
        if nxt < at:
            raise ContractError(f"FAIL [pin {pin}]: `{needle}` is out of order; the op-entry flushes run consumed, then the scatter flush (llama.cpp-4hg7)")
        at = nxt
        if pin == "F3":
            # yx28: the third flush is the body of the defer guard, and the guard is a statement of the
            # hybrid block.  Directional: an unconditional flush here is a FAIL too, because the join
            # below is then dead and the pin would no longer describe the code.
            guard = "if (!defer_entry_flush) { flush_pending_cpu_scatter(); reuse_shared_act = shared_act_reusable(); }"
            if entry.count(guard) != 1:
                raise ContractError(
                    f"FAIL [pin F3]: the op-entry `{needle}` is not exactly `{guard}`; the yx28 join defers it only "
                    "for decode up after gate, and flushes it on every other path (llama.cpp-yx28)"
                )
            require_unconditional(entry, nxt, header="if (!defer_entry_flush)", pin=pin, what=f"`{needle}`")
            require_unconditional(entry, entry.find(guard), header="if (moe_hybrid_with_plan)", pin=pin,
                                  what="the `if (!defer_entry_flush)` op-entry flush guard")
        else:
            require_unconditional(entry, nxt, header="if (moe_hybrid_with_plan)", pin=pin, what=f"`{needle}`")
    defer_def = ("const bool defer_entry_flush = reuse_shared_act && g_pending_scatter.active && "
                 "!g_pending_scatter_sibling.active && g_pending_scatter.shares_activation && "
                 "g_pending_scatter.act_serial == g_moe_shared_act.record.serial;")
    if squash(code).count(defer_def) != 1:
        raise ContractError(
            f"FAIL [pin F3]: defer_entry_flush is not defined exactly once as `{defer_def}` (llama.cpp-yx28)"
        )
    check_sibling_join(code, hostpath_src)

    # 7. the split.
    d_at = code.find(DISPATCH_MARKER)
    if d_at < 0:
        raise ContractError("FAIL: do_cpu_dispatch lambda not found (llama.cpp-4hg7)")
    body = squash(brace_block_from(code, d_at))
    if "const bool split_needs_pool = !immutable_host_recipe;" not in body:
        raise ContractError("FAIL [pin S2]: do_cpu_dispatch does not derive split_needs_pool from immutable_host_recipe, so recipe types reserve a span they never use (llama.cpp-4hg7)")
    if not re.search(r"\|\| \(split_needs_pool && !hc_pool\.can_serve\(n_cpu_entries\)\)\) \{ dispatch_cpu_and_scatter\(cpu_entries\)", body):
        raise ContractError(
            "FAIL [pin S1]: do_cpu_dispatch splits hot/cold without checking the pool can hold both groups (for "
            "types that use the pool); an over-capacity split would wrap and overlap the hot region (llama.cpp-4hg7)"
        )
    order = [
        "const size_t hot_first = split_needs_pool ? hc_pool.reserve(n_cpu_entries) : 0;",
        "const size_t cold_first = hot_first + hot_entries.size();",
        # llama.cpp-yx28: per-slot rows of both groups are gathered once, hot then cold, at the span's
        # start, so cold's rows land at cold_first; both dispatches are told they are pregathered.
        "const bool pregather = split_needs_pool && !cpu_shared_act;",
        # llama.cpp-z4kd: the gather's events and the scatter serial it was issued at go to both jobs'
        # prologues instead of a host wait.
        "std::vector<sycl::event> pregather_events; const uint64_t pregather_serial = g_cpu_scatter_serial;",
        "if (pregather) { std::vector<const expert_dispatch_entry *> rows; rows.reserve(n_cpu_entries); "
        "for (const expert_dispatch_entry & e : hot_entries) { rows.push_back(&e); } "
        "for (const expert_dispatch_entry & e : cold_entries) { rows.push_back(&e); } "
        "pregather_events = gather_activation_rows(rows, hc_pool.act_handle(), hot_first * static_cast<size_t>(K) * sizeof(float)); }",
        "dispatch_cpu_and_scatter(hot_entries, hot_first, pregather, pregather_events, pregather_serial);",
        "flush_pending_cpu_scatter()",
        "dispatch_cpu_and_scatter(cold_entries, cold_first, pregather, pregather_events, pregather_serial);",
    ]
    at = -1
    for needle in order:
        nxt = body.find(needle, at + 1)
        if nxt < 0:
            raise ContractError(
                f"FAIL [pin S3]: do_cpu_dispatch has no `{needle}` after the previous step; the split must reserve one "
                "span, gather hot then cold rows at its start, give hot its start and cold start + hot count "
                "(llama.cpp-4hg7)"
            )
        at = nxt

    # 10. no host wait of any kind in the split.
    if re.search(r"event::wait|\.wait\(\)|pending_cpu_scatter_wait_prev_events|stream->wait|wait_and_throw", body):
        raise ContractError("FAIL [pin S4]: do_cpu_dispatch waits on the host; the hot/cold hazard is closed by disjoint regions, not a wait (llama.cpp-4hg7)")


def self_test(backend_src: str) -> int:
    failures: list[str] = []
    pool_src = POOL_CPP.read_text()
    hostpath_src = HOSTPATH_HPP.read_text()
    expert_pool_src = EXPERT_POOL_CPP.read_text()
    n_mutants = 0

    def expect_fail(name: str, mutated: str, mutated_pool: str | None = None, pin: str | None = None,
                    mutated_hostpath: str | None = None, mutated_expert_pool: str | None = None) -> None:
        nonlocal n_mutants
        n_mutants += 1
        if mutated == backend_src and (mutated_pool is None or mutated_pool == pool_src) and (
                mutated_hostpath is None or mutated_hostpath == hostpath_src) and (
                mutated_expert_pool is None or mutated_expert_pool == expert_pool_src):
            failures.append(f"{name}: mutation did not change the source (anchor stale)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        try:
            check(mutated, mutated_pool, mutated_hostpath, mutated_expert_pool)
        except ContractError as e:
            if pin is not None and f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: failed, but not on pin {pin}: {str(e)[:120]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:70]}...)")
                return
            print(f"  mutant {name}: caught" + (f" (pin {pin})" if pin else ""))
            return
        failures.append(f"{name}: mutant survived")
        print(f"  mutant {name}: SURVIVED")

    def sub(old: str, new: str, in_lambda: bool = True, after: str | None = None) -> str:
        """Replace the first `old` at or after the dispatch_cpu_compute lambda.

        The secondary-GPU dispatch earlier in the file carries look-alike lines
        (i11/i12/dst_off); a mutant must land in the lambda under test.
        """
        start = backend_src.find(after if after else (LAMBDA_MARKER if in_lambda else ""))
        at = backend_src.find(old, max(start, 0))
        if at < 0:
            return backend_src
        return backend_src[:at] + new + backend_src[at + len(old) :]

    # m1: the fact drops ne11 (the original defect).
    expect_fail("fact-drops-ne11", sub("const bool cpu_shared_act = cpu_expert_tg_active && ne11 == 1;",
                                       "const bool cpu_shared_act = cpu_expert_tg_active;",
                                       in_lambda=False))
    # m2: the single-D2H branch reverts to the TG flag alone.
    expect_fail("single-d2h-on-tg-flag", sub("else if (cpu_shared_act && n_cpu > 1)",
                                             "else if (cpu_expert_tg_active && n_cpu > 1)"))
    # m3: the CPU task pointer reverts to the TG flag alone.
    expect_fail("task-act-host-on-tg-flag", sub("t.act_host     = cpu_shared_act ?",
                                                "t.act_host     = cpu_expert_tg_active ?"))
    # m4: the recipe task pointer reverts to the TG flag alone.
    expect_fail("recipe-activations-on-tg-flag", sub("task.activations               = cpu_shared_act ?",
                                                     "task.activations               = cpu_expert_tg_active ?"))
    # m5: the per-expert fallback loses its slot offset (every task would read slot 0's copy).
    expect_fail("task-act-host-no-slot-offset", sub(
        "act_pinned + ci * static_cast<size_t>(K);\n                    t.output_host",
        "act_pinned;\n                    t.output_host"))
    # m6: the per-expert D2H stops selecting the slot's row.
    expect_fail("per-expert-row-fixed", sub("static_cast<size_t>(entry->id % ne11) * nb11",
                                            "static_cast<size_t>(0) * nb11", in_lambda=False, after=GATHER_MARKER))
    # m7: the per-expert D2H stops selecting the token.
    expect_fail("per-expert-token-fixed", sub("static_cast<size_t>(entry->iid1) * nb12",
                                              "static_cast<size_t>(0) * nb12", in_lambda=False, after=GATHER_MARKER))
    # m8: the per-expert D2H writes every slot to the same staging offset.
    expect_fail("per-expert-dst-collapsed", backend_src, None, None,
                hostpath_src.replace("const size_t dst = dst_base + i * row_bytes;", "const size_t dst = dst_base;", 1))
    # yx28: the per-row branch is skipped although nothing gathered the rows.
    expect_fail("per-row-branch-disabled", sub("} else if (!act_pregathered) {", "} else if (false) {"))
    # yx28: the gather helper is renamed; its pins must report it, not pass on another lambda.
    expect_fail("gather-helper-renamed", sub(GATHER_MARKER, "auto gather_activation_rows_v2 = [&](", in_lambda=False))
    # m10: the cold group shares the hot group's entries (the original hot/cold overwrite).
    expect_fail("cold-shares-hot-region", sub("const size_t cold_first = hot_first + hot_entries.size();",
                                              "const size_t cold_first = hot_first;",
                                              in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m11: the hot group is not handed its slice (it would reserve its own and the cold one overlaps).
    expect_fail("hot-not-given-slice", sub("dispatch_cpu_and_scatter(hot_entries, hot_first, pregather, pregather_events, pregather_serial);",
                                           "dispatch_cpu_and_scatter(hot_entries);",
                                           in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m12: no single reservation for the split.
    expect_fail("split-without-reservation", sub("const size_t hot_first  = split_needs_pool ? hc_pool.reserve(n_cpu_entries) : 0;",
                                                 "const size_t hot_first  = 0;",
                                                 in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m13: the split no longer checks the pool can hold both groups.
    expect_fail("split-unguarded-by-capacity", sub("(split_needs_pool && !hc_pool.can_serve(n_cpu_entries))) {",
                                                   "false) {", in_lambda=False, after=DISPATCH_MARKER), None, "S1")
    # yx28: the one-pass gather of both groups must put hot rows first, at hot_first, and only for per-slot rows.
    expect_fail("pregather-rows-out-of-order", sub(
        "e : hot_entries) {\n                            rows.push_back(&e);\n                        }\n"
        "                        for (const expert_dispatch_entry & e : cold_entries) {",
        "e : cold_entries) {\n                            rows.push_back(&e);\n                        }\n"
        "                        for (const expert_dispatch_entry & e : hot_entries) {",
        in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    expect_fail("pregather-base-not-hot-first", sub("hot_first * static_cast<size_t>(K) * sizeof(float));",
                                                    "cold_first * static_cast<size_t>(K) * sizeof(float));",
                                                    in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    expect_fail("pregather-on-shared-act", sub("const bool   pregather  = split_needs_pool && !cpu_shared_act;",
                                               "const bool   pregather  = split_needs_pool;",
                                               in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # M2: recipe types (Q1_0/NVFP4) reserve a span they never use.
    expect_fail("recipe-types-reserve-pool", sub("const bool   split_needs_pool = !immutable_host_recipe;",
                                                 "const bool   split_needs_pool = true;",
                                                 in_lambda=False, after=DISPATCH_MARKER), None, "S2")
    # m14 (W1): the memset comes back into dispatch_cpu_compute, on the submitting thread.
    expect_fail("memset-back-in-dispatch", sub(
        "const bool zero_out_after_act_wait = from_pool || !out_owner.valid();",
        "const bool zero_out_after_act_wait = from_pool || !out_owner.valid();\n"
        "                if (zero_out_after_act_wait) { std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float)); }"),
        None, "W1")
    wait_then_zero = ("    if (!inputs.events.empty()) {\n        sycl::event::wait(inputs.events);\n    }\n"
                      "    if (inputs.zero_out && inputs.zero_bytes > 0) {\n        std::memset(inputs.zero_out, 0, inputs.zero_bytes);\n    }\n")
    zero_then_wait = ("    if (inputs.zero_out && inputs.zero_bytes > 0) {\n        std::memset(inputs.zero_out, 0, inputs.zero_bytes);\n    }\n"
                      "    if (!inputs.events.empty()) {\n        sycl::event::wait(inputs.events);\n    }\n")
    # W1: the prologue zeroes before it waits the activation copy.
    expect_fail("prologue-zeroes-before-wait", sub(wait_then_zero, zero_then_wait, in_lambda=False, after=WAIT_INPUTS_DEF),
                None, "W1")
    # W1 (I2(c) analogue): the prologue's wait gains a conjunct, so a one-event job skips it.
    expect_fail("prologue-wait-needs-two-events", sub("if (!inputs.events.empty()) {", "if (inputs.events.size() > 1) {",
                                                      in_lambda=False, after=WAIT_INPUTS_DEF), None, "W1")
    # W1: the closure no longer runs the waits.
    expect_fail("prologue-closure-empty", sub("() {\n        moe_cpu_job_wait_inputs(inputs);\n    };", "() {};",
                                              in_lambda=False, after=PROLOGUE_DEF), None, "W1")
    # W2: one job form stops running the prologue, or the pool submission drops it.
    expect_fail("fallback-skips-prologue", sub(
        "prologue();\n                        ggml_sycl_cpu_expert_mul_mat_batched", "ggml_sycl_cpu_expert_mul_mat_batched"),
        None, "W2")
    expect_fail("recipe-skips-prologue", sub("GGML_UNUSED(workspace_owner);\n                        prologue();\n",
                                             "GGML_UNUSED(workspace_owner);\n"), None, "W2")
    expect_fail("pool-submit-drops-prologue", sub("cpu_pool.submit_batch(std::move(batch_tasks), std::move(job_prologue))",
                                                  "cpu_pool.submit_batch(std::move(batch_tasks))"), None, "W2")
    worker_prologue = "            if (prologue) {\n                prologue();\n            }\n"
    worker_resolve = "            if (resolve) {\n                resolve(tasks);\n            }\n"
    worker_kernel = ("            ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size()), "
                     "row_workers_);\n")
    expect_fail("pool-worker-skips-prologue", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_prologue, "", 1))
    expect_fail("pool-worker-kernel-on-the-arena", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_kernel, "            ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), "
                                        "static_cast<int>(tasks.size()));\n", 1))
    expect_fail("pool-worker-prologue-after-kernel", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_prologue, "", 1).replace(worker_kernel, worker_kernel + worker_prologue, 1))
    expect_fail("pool-worker-prologue-conditioned-away", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_prologue,
                                        "            if (prologue && !trace) {\n"
                                        "                prologue();\n            }\n", 1))
    expect_fail("pool-worker-skips-resolve", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_resolve, "", 1))
    expect_fail("pool-worker-resolve-before-prologue", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_resolve, "", 1)
                .replace(worker_prologue, worker_resolve + worker_prologue, 1))
    expect_fail("pool-worker-resolve-after-kernel", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_resolve, "", 1)
                .replace(worker_kernel, worker_kernel + worker_resolve, 1))
    expect_fail("pool-worker-resolve-conditioned-away", backend_src, None, "W2", None,
                expert_pool_src.replace(worker_resolve,
                                        "            if (resolve && trace) {\n"
                                        "                resolve(tasks);\n            }\n", 1))
    # W3: the zeroing request is skipped on some paths, defined away, or wrapped in a block of its own.
    expect_fail("zero-request-conditioned-away", sub(
        "if (zero_out_after_act_wait) {\n                    job_inputs.zero_out",
        "if (zero_out_after_act_wait && n_cpu > 1) {\n                    job_inputs.zero_out"), None, "W3")
    expect_fail("memset-guard-defined-false", sub(
        "const bool zero_out_after_act_wait = from_pool || !out_owner.valid();",
        "const bool zero_out_after_act_wait = false;"), None, "W3")
    zr_start = backend_src.find("                if (zero_out_after_act_wait) {\n", backend_src.find(LAMBDA_MARKER))
    zero_req = backend_src[zr_start : backend_src.find("                }\n", zr_start) + len("                }\n")] \
        if zr_start >= 0 else "<zeroing request missing>"
    expect_fail("zero-request-wrapped-in-if-false", sub(zero_req, "                if (false) {\n" + zero_req + "                }\n"),
                None, "W3")
    # W4 (I2(c) analogue): the shared staging's copy event reaches the job only when n_cpu > 1.
    expect_fail("shared-act-event-needs-n-cpu-gt-1", sub(
        "job_inputs.events.push_back(act_d2h_event);", "if (n_cpu > 1) { job_inputs.events.push_back(act_d2h_event); }"),
        None, "W4")
    expect_fail("pregather-events-dropped", sub("job_inputs.events   = pregather_events;", "job_inputs.events.clear();"),
                None, "W4")
    expect_fail("lease-events-dropped", sub("job_inputs.events.push_back(entry.ready_event);", "(void) entry;"), None, "W4")
    expect_fail("copy-events-overwritten", sub("                // Order this job's pool writes",
                                               "                job_inputs.events = {};\n                // Order this job's pool writes"),
                None, "W4")
    # O1: the order decision loses a fact or an action.
    expect_fail("order-ignores-last-flush", sub(
        "job_inputs.events.insert(job_inputs.events.end(), g_cpu_scatter_last_flush.events.begin(),\n"
        "                                                 g_cpu_scatter_last_flush.events.end());\n", ""), None, "O1")
    expect_fail("order-queue-without-barrier", sub("job_inputs.events.push_back(stream->ext_oneapi_submit_barrier());", ""),
                None, "O1")
    expect_fail("order-current-serial-is-copy", sub("order_req.current_scatter_serial = g_cpu_scatter_serial;",
                                                    "order_req.current_scatter_serial = copy_serial;"), None, "O1")
    expect_fail("shared-copy-serial-fresh", sub("copy_serial          = g_moe_shared_act.record.scatter_serial;", ""), None, "W4")
    expect_fail("flush-skips-serial", sub("        ++g_cpu_scatter_serial;\n", "", in_lambda=False, after=SLOT_FLUSH_DEF),
                None, "O1")
    expect_fail("flush-advances-serial-once-per-group", sub("for (int k = 0; k < n_group; ++k) {\n        ++g_cpu_scatter_serial;",
                                                            "for (int k = 0; k < 1; ++k) {\n        ++g_cpu_scatter_serial;",
                                                            in_lambda=False, after=SLOT_FLUSH_DEF), None, "O1")
    expect_fail("flush-serial-conditioned-away", sub("    for (int k = 0; k < n_group; ++k) {\n        ++g_cpu_scatter_serial;\n    }\n",
                                                     "    if (partner) {\n        for (int k = 0; k < n_group; ++k) {\n"
                                                     "            ++g_cpu_scatter_serial;\n        }\n    }\n",
                                                     in_lambda=False, after=SLOT_FLUSH_DEF), None, "O1")
    expect_fail("flush-not-recorded", sub("    g_cpu_scatter_last_flush.events    = ws.events;\n", "",
                                          in_lambda=False, after=SLOT_FLUSH_DEF), None, "O1")
    expect_fail("shared-copy-serial-not-recorded", sub("record.scatter_serial   = g_cpu_scatter_serial;",
                                                       "record.scatter_serial   = 0;", in_lambda=False), None, "O1")
    expect_fail("route-copy-serial-not-recorded", sub("act_record.scatter_serial = g_cpu_scatter_serial;",
                                                      "act_record.scatter_serial = 0;", in_lambda=False), None, "O1")
    expect_fail("route-copy-not-published", sub("g_moe_shared_act.record   = act_record;", "(void) act_record;",
                                                in_lambda=False), None, "O1")
    consumed = "            flush_pending_cpu_scatter_if_consumed(dst, ctx.device);\n"
    # F3/F1 (structure): present, in order, exactly once -- but dead (the body of `if (false)`).
    expect_fail("entry-flush-under-if-false", sub("            flush_pending_cpu_scatter();\n",
                                                "            if (false) flush_pending_cpu_scatter();\n",
                                                in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F3")
    expect_fail("consumed-flush-under-if-false", sub(consumed, "            if (false) flush_pending_cpu_scatter_if_consumed(dst, ctx.device);\n",
                                                     in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F1")
    # I2(a)/(b): the op-entry flushes that put the earlier scatter's H2D ahead of this op's D2H.
    expect_fail("entry-flush-dropped", sub("            flush_pending_cpu_scatter();\n", "",
                                         in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F3")
    expect_fail("consumed-flush-dropped", sub("            flush_pending_cpu_scatter_if_consumed(dst, ctx.device);\n", "",
                                              in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F1")
    # llama.cpp-3bww: the third entry flush degrades to the non-blocking "only if ready" form, which
    # lets gate's dispatch overwrite up's still-running pending scatter.
    expect_fail("entry-flush-only-when-ready", sub(
        "            flush_pending_cpu_scatter();\n",
        "            if (g_pending_scatter.future.valid() && g_pending_scatter.future.wait_for(std::chrono::seconds(0)) "
        "== std::future_status::ready) { flush_pending_cpu_scatter(); }\n",
        in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F3")
    # yx28 join (lead ruling on llama.cpp-p6zh): one mutant per property.
    guarded = "            if (!defer_entry_flush) {\n                flush_pending_cpu_scatter();\n"
    expect_fail("entry-flush-unconditional-again", sub(
        guarded, "            flush_pending_cpu_scatter();\n            if (!defer_entry_flush) {\n",
        in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F3")
    expect_fail("join-skips-keep-decision", sub("if (ggml_sycl::moe_sibling_pending_keep(keep)) {", "if (true) {",
                                                in_lambda=False, after=JOIN_ANCHOR), None, "J1")
    expect_fail("keep-drops-serial-check", backend_src, None, "J1", hostpath_src.replace(
        " ||\n        r.pending_act_serial != r.current_act_serial", "", 1))
    expect_fail("keep-drops-disjoint-check", backend_src, None, "J1", hostpath_src.replace(
        "return r.same_row_geometry && moe_pool_spans_disjoint(r.pending_first, r.pending_count, r.op_first, r.op_count);",
        "return r.same_row_geometry;", 1))
    expect_fail("keep-pool-guard-weakened", backend_src, None, "J1", hostpath_src.replace(
        "if (!r.pending_from_pool || !r.op_from_pool) {", "if (!r.pending_from_pool || r.op_from_pool) {", 1))
    expect_fail("keep-early-accept-inserted", backend_src, None, "J1", hostpath_src.replace(
        "inline bool moe_sibling_pending_keep(const moe_sibling_pending_request & r) {\n",
        "inline bool moe_sibling_pending_keep(const moe_sibling_pending_request & r) {\n"
        "    if (r.op_count != 0) {\n        return true;\n    }\n", 1))
    expect_fail("keep-request-drops-staging-serial", sub(
        "keep.current_act_serial = g_moe_shared_act.record.serial;", "keep.current_act_serial = gate.act_serial;",
        in_lambda=False, after=JOIN_ANCHOR), None, "J1")
    join_else = ("                        flush_pending_cpu_scatter();\n                    }\n                }\n"
                 "                auto cpu_result = dispatch_cpu_compute(cpu_entries, cpu_pool_first);\n")
    expect_fail("join-else-drops-flush", sub(
        join_else, "                    }\n                }\n                auto cpu_result = dispatch_cpu_compute(cpu_entries, cpu_pool_first);\n",
        in_lambda=False, after=JOIN_ANCHOR), None, "J2")
    expect_fail("join-flush-after-pool-writes", sub(
        join_else, "                    }\n                }\n                auto cpu_result = dispatch_cpu_compute(cpu_entries, cpu_pool_first);\n"
                   "                flush_pending_cpu_scatter();\n", in_lambda=False, after=JOIN_ANCHOR), None, "J2")
    expect_fail("join-else-waits-again", sub(
        join_else, "                        flush_pending_cpu_scatter();\n"
                   "                        sycl::event::wait(g_pending_scatter.prev_scatter.scatter_events);\n"
                   "                    }\n                }\n"
                   "                auto cpu_result = dispatch_cpu_compute(cpu_entries, cpu_pool_first);\n",
        in_lambda=False, after=JOIN_ANCHOR), None, "J2")
    expect_fail("drain-drops-older-slot", sub("    flush_pending_cpu_scatter_slot(g_pending_scatter_sibling);\n", "",
                                              in_lambda=False, after="static void flush_pending_cpu_scatter() {"), None, "J3")
    expect_fail("consumed-check-ignores-sibling", sub(
        "{ &g_pending_scatter_sibling, &g_pending_scatter }", "{ &g_pending_scatter, &g_pending_scatter }",
        in_lambda=False, after="static bool flush_pending_cpu_scatter_if_consumed("), None, "J3")
    # A1: the assertion that a pending scatter is never overwritten.
    # The assertion statement, lifted from the source so the mutants cannot drift from its layout.
    a_start = backend_src.find("                GGML_ASSERT(!g_pending_scatter.active &&", backend_src.find(APPLY_MARKER))
    assert_line = backend_src[a_start : backend_src.find(");\n", a_start) + 3] if a_start >= 0 else "<assertion missing>"
    expect_fail("apply-assert-dropped", sub(assert_line, "", in_lambda=False, after=APPLY_MARKER), None, "A1")
    expect_fail("apply-assert-vacuous", sub("GGML_ASSERT(!g_pending_scatter.active &&", "GGML_ASSERT(true &&",
                                            in_lambda=False, after=APPLY_MARKER), None, "A1")
    expect_fail("apply-assert-after-first-write", sub(
        assert_line + "                g_pending_scatter.future        = std::move(r.future);\n",
        "                g_pending_scatter.future        = std::move(r.future);\n" + assert_line,
        in_lambda=False, after=APPLY_MARKER), None, "A1")
    expect_fail("apply-assert-under-if-false", sub(assert_line, "                if (false)\n" + assert_line,
                                                   in_lambda=False, after=APPLY_MARKER), None, "A1")
    # A2: the direct CPU-TG writer stops flushing, or a third writer appears.
    expect_fail("direct-path-flush-dropped", sub(
        "if (g_pending_scatter.active || g_pending_scatter_sibling.active) {\n                    flush_pending_cpu_scatter();\n                }\n\n                const int64_t         K ",
        "const int64_t         K ", in_lambda=False, after="auto dispatch_cpu_entries_now = [&]("), None, "A2")
    expect_fail("third-writer-appears", sub("g_pending_scatter.active        = true;\n                g_pending_scatter.dst_tensor    = dst;\n                g_pending_scatter.entries",
                                            "g_pending_scatter.active        = true;\n                g_pending_scatter.active = true;\n                g_pending_scatter.dst_tensor    = dst;\n                g_pending_scatter.entries",
                                            in_lambda=False, after=APPLY_MARKER), None, "A2")
    # m15..m19: each offset that names the slot must carry the pool entry.
    expect_fail("out-ptr-ignores-pool-entry", sub("out_pinned = bp.out + pool_base * static_cast<size_t>(N);",
                                                  "out_pinned = bp.out;"), None, "R2")
    expect_fail("act-ptr-ignores-pool-entry", sub("act_pinned = bp.act + pool_base * static_cast<size_t>(K);",
                                                  "act_pinned = bp.act;"), None, "R2")
    expect_fail("scatter-src-offset-ignores-pool-entry", sub(
        "(pool_base + ci) * static_cast<size_t>(N) * sizeof(float) });",
        "ci * static_cast<size_t>(N) * sizeof(float) });"), None, "R2")
    expect_fail("single-d2h-ignores-pool-entry", sub(
        "act_handle, pool_base * static_cast<size_t>(K) * sizeof(float), src1_storage.handle,",
        "act_handle, 0, src1_storage.handle,"), None, "R2")
    expect_fail("per-expert-d2h-ignores-pool-entry", sub(
        "rows, act_handle, pool_base * static_cast<size_t>(K) * sizeof(float));",
        "rows, act_handle, 0);"))
    # m20: a host wait comes back into the split.
    cold_call = "dispatch_cpu_and_scatter(cold_entries, cold_first, pregather, pregather_events, pregather_serial);"
    expect_fail("host-wait-in-split", sub("flush_pending_cpu_scatter();\n                    }\n                    " + cold_call,
                                          "flush_pending_cpu_scatter();\n                        sycl::event::wait(pregather_events);\n"
                                          "                    }\n                    " + cold_call,
                                          in_lambda=False, after=DISPATCH_MARKER), None, "S4")
    # z4kd: cold is not handed the pregather's events, so its prologue does not wait the gather.
    expect_fail("cold-not-given-pregather-events", sub(cold_call, "dispatch_cpu_and_scatter(cold_entries, cold_first, pregather);",
                                                       in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    expect_fail("pregather-events-not-kept", sub("pregather_events = gather_activation_rows(rows, hc_pool.act_handle(),",
                                                 "gather_activation_rows(rows, hc_pool.act_handle(),",
                                                 in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m21/m22: the ring itself.
    expect_fail("ring-never-wraps", backend_src, None, None,
                hostpath_src.replace("return cursor + n > capacity ? 0 : cursor;", "return cursor;", 1))
    expect_fail("reserve-bypasses-ring", backend_src, pool_src.replace(
        "const size_t first = ggml_sycl::moe_pool_reserve_first(next_entry_, n_experts, max_experts_);",
        "const size_t first = next_entry_;", 1))
    expect_fail("ring-cursor-not-advanced", backend_src, pool_src.replace("next_entry_        = (first + n_experts) % max_experts_;", ""))
    # m9: a second, unconditioned shared decision appears in the lambda.
    expect_fail("second-source-in-lambda", sub(
        "if (act_on_host && cpu_shared_act) {",
        "if (act_on_host && cpu_expert_tg_active) {"))

    try:
        check(backend_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {n_mutants} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    backend_src = BACKEND.read_text()
    if "--self-test" in argv:
        return self_test(backend_src)
    try:
        check(backend_src)
    except ContractError as e:
        print(e)
        return 1
    print(
        "PASS: the host-expert CPU dispatch shares one activation only when ne11 == 1, and otherwise copies "
        "each slot's own row, the hot and cold dispatches use disjoint pool regions with no host wait (llama.cpp-4hg7), "
        "and a pending CPU scatter is flushed before another is recorded (llama.cpp-3bww)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
