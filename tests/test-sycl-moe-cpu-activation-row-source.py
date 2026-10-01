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
     entry.iid1 to ci * K, and both builders still fall back to act_pinned + ci * K.

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
     everything is one dispatch;
  8. the out-region memset is not before the activation D2H wait (that wait, on the
     in-order queue, is what proves an earlier scatter's H2D from this region has run);
  9. no host wait is (re)introduced in do_cpu_dispatch;
 10. CROSS-OP safety is by ORDERING, not by the ring (entry offsets scale with each op's own K/N
     and a top-K pool restarts at entry 0, so different ops DO alias).  Three premises are pinned:
     the deferred activation wait is exactly `if (act_deferred_pending) { act_deferred_evt.wait(); }`
     (pin W1: with an extra conjunct such as n_cpu > 1 the memset and the CPU read run before the
     D2H lands); the memset is the very next statement after it at the same nesting level (pin W2:
     it follows the wait in control flow, not only in the text); and the op-entry flushes --
     flush_pending_cpu_scatter_if_consumed, flush_pending_cpu_pipeline_if_consumed,
     try_flush_pending_cpu_scatter, in that order -- precede the shared activation D2H (pins
     F1/F2/F3: the earlier scatter's H2D must be enqueued before this op's D2H).  Three more pins
     close the ways a pinned statement can be present and still not run: W3 (the memset's guard is
     defined exactly once as `from_pool || !out_owner.valid()`, so `= false` cannot drop the zeroing),
     and, for W2 and F1-F3, a structural test of each pinned statement -- it follows a statement or
     block boundary (`;`, `{`, `}`), so it is not the body of an `if (...)`/`else`/loop header, and
     the innermost block that holds it is the expected one (the lambda body for the wait+memset pair,
     `if (moe_hybrid_with_plan)` for the three flushes), so it is not wrapped in a block of its own.
     These are TEXTUAL and STRUCTURAL pins (brace and statement adjacency), not control-flow analysis:
     a conditional hidden behind a macro or a helper function, or an early `return` above the
     statement, is outside what they can see.

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


def check(backend_src: str, pool_src: str | None = None) -> None:
    code = blank_comments(backend_src)

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

    # 5. the per-expert copy selects the slot's row, of the entry's token, into slot ci.
    per_expert = re.search(
        r"const int64_t i11\s*=\s*entry\.id\s*%\s*ne11\s*;\s*const int64_t i12\s*=\s*entry\.iid1\s*;"
        r".*?const size_t dst_off\s*=\s*\(\s*pool_base\s*\+\s*ci\s*\)\s*\*\s*static_cast<size_t>\(K\)\s*\*\s*sizeof\(float\)\s*;",
        body,
    )
    if not per_expert:
        raise ContractError(
            "FAIL: the per-expert D2H no longer copies row (entry.id % ne11) of token entry.iid1 into slot "
            "ci * K (llama.cpp-4hg7)"
        )
    if "src_off" not in body or not re.search(r"static_cast<size_t>\(i11\)\s*\*\s*nb11", body):
        raise ContractError("FAIL: the per-expert D2H source offset no longer uses i11 * nb11 (llama.cpp-4hg7)")

    check_hot_cold(code)
    check_pool_ring(pool_src if pool_src is not None else POOL_CPP.read_text())


DISPATCH_MARKER = "auto do_cpu_dispatch = [&]("
POOL_CPP = ROOT / "ggml/src/ggml-sycl/pinned-buffer-pool.cpp"


def check_pool_ring(pool_src: str) -> None:
    code = blank_comments(pool_src)
    at = code.find("PinnedBufferPool::reserve(")
    if at < 0:
        raise ContractError("FAIL: PinnedBufferPool::reserve() is not defined (llama.cpp-4hg7)")
    body = squash(brace_block_from(code, at))
    if not re.search(r"if \(\s*next_entry_\s*\+\s*n_experts\s*>\s*max_experts_\s*\)\s*\{\s*next_entry_\s*=\s*0\s*;", body):
        raise ContractError("FAIL: PinnedBufferPool::reserve() does not wrap to entry 0 past the end (llama.cpp-4hg7)")
    if not re.search(r"next_entry_\s*=\s*\(\s*first\s*\+\s*n_experts\s*\)\s*%\s*max_experts_", body):
        raise ContractError("FAIL: PinnedBufferPool::reserve() does not advance its cursor past the span (llama.cpp-4hg7)")


def check_hot_cold(code: str) -> None:
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
        "const size_t dst_off = (pool_base + ci) * static_cast<size_t>(K) * sizeof(float);": "the per-expert D2H destination ignores the pool entry",
        "(pool_base + ci) * static_cast<size_t>(N) * sizeof(float) });": "the scatter entry's src_offset ignores the pool entry",
    }
    for needle, why in needles.items():
        if needle not in lam:
            raise ContractError(f"FAIL [pin R2]: dispatch_cpu_compute: {why} (llama.cpp-4hg7)")

    # 8. (I2) the deferred activation wait and the memset that depends on it.
    #    The wait runs whenever the D2H is pending -- no extra conjunct: with n_cpu == 1 the memset
    #    and the CPU read would otherwise race the D2H.  The memset is the very next statement at
    #    the same nesting level, so it follows the wait in control flow, not only in the text.
    memsets = [m.start() for m in re.finditer(r"std::memset\(\s*out_pinned\b", lam)]
    if len(memsets) != 1:
        raise ContractError("FAIL [pin W1]: the out-region memset must occur exactly once in dispatch_cpu_compute (llama.cpp-4hg7)")
    if "if (act_deferred_pending) { act_deferred_evt.wait(); }" not in lam:
        raise ContractError(
            "FAIL [pin W1]: the deferred activation wait is not exactly `if (act_deferred_pending) { "
            "act_deferred_evt.wait(); }` -- an extra conjunct (e.g. n_cpu > 1) lets the memset and the CPU "
            "read run before the D2H lands (llama.cpp-4hg7)"
        )
    if not re.search(
        r"if \(act_deferred_pending\) \{ act_deferred_evt\.wait\(\); \} "
        r"if \(zero_out_after_act_wait\) \{ std::memset\(out_pinned, 0, n_cpu \* static_cast<size_t>\(N\) \* sizeof\(float\)\); \}",
        lam,
    ):
        raise ContractError(
            "FAIL [pin W2]: the out-region memset is not the statement that directly follows the activation "
            "wait at the same nesting level; before it, or under a condition that skips it, it can zero a "
            "region an earlier scatter's H2D has not read yet (llama.cpp-4hg7)"
        )

    # The guard of that memset is a one-line definition; `= false` would silently drop the zeroing
    # while every textual neighbour (W1/W2) still matches.
    if lam.count("const bool zero_out_after_act_wait = from_pool || !out_owner.valid();") != 1:
        raise ContractError(
            "FAIL [pin W3]: zero_out_after_act_wait is not defined exactly once as `from_pool || "
            "!out_owner.valid()` in dispatch_cpu_compute; the memset it guards is the zeroing of a region the "
            "CPU kernels accumulate into (llama.cpp-4hg7)"
        )
    # W2, structural half: the wait+memset pair is a direct statement of the lambda body -- not under an
    # `if (...)`, an `else`, or a block of its own.
    require_unconditional(lam, lam.find("if (act_deferred_pending) { act_deferred_evt.wait(); }"),
                          header=None, pin="W2", what="the activation wait + out-region memset pair")

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
        ("flush_pending_cpu_pipeline_if_consumed(dst, ctx.device);", "F2"),
        ("try_flush_pending_cpu_scatter();", "F3"),
    ]
    at = -1
    for needle, pin in order:
        if entry.count(needle) != 1:
            raise ContractError(
                f"FAIL [pin {pin}]: `{needle}` must appear exactly once between the hybrid-op entry and the shared "
                "activation D2H; the earlier op's scatter H2D has to be enqueued BEFORE this op's D2H (llama.cpp-4hg7)"
            )
        nxt = entry.find(needle)
        if nxt < at:
            raise ContractError(f"FAIL [pin {pin}]: `{needle}` is out of order; the op-entry flushes run consumed, pipeline, try (llama.cpp-4hg7)")
        at = nxt
        require_unconditional(entry, nxt, header="if (moe_hybrid_with_plan)", pin=pin, what=f"`{needle}`")

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
        "dispatch_cpu_and_scatter(hot_entries, hot_first)",
        "flush_pending_cpu_scatter()",
        "dispatch_cpu_and_scatter(cold_entries, cold_first)",
    ]
    at = -1
    for needle in order:
        nxt = body.find(needle, at + 1)
        if nxt < 0:
            raise ContractError(
                f"FAIL [pin S3]: do_cpu_dispatch has no `{needle}` after the previous step; the split must reserve one "
                "span, give hot its start and cold start + hot count (llama.cpp-4hg7)"
            )
        at = nxt

    # 10. no host wait of any kind in the split.
    if re.search(r"event::wait|\.wait\(\)|wait_prev_scatter_events|stream->wait|wait_and_throw", body):
        raise ContractError("FAIL [pin S4]: do_cpu_dispatch waits on the host; the hot/cold hazard is closed by disjoint regions, not a wait (llama.cpp-4hg7)")


def self_test(backend_src: str) -> int:
    failures: list[str] = []
    pool_src = POOL_CPP.read_text()

    def expect_fail(name: str, mutated: str, mutated_pool: str | None = None, pin: str | None = None) -> None:
        if mutated == backend_src and (mutated_pool is None or mutated_pool == pool_src):
            failures.append(f"{name}: mutation did not change the source (anchor stale)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        try:
            check(mutated, mutated_pool)
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
    expect_fail("per-expert-row-fixed", sub("const int64_t i11     = entry.id % ne11;",
                                            "const int64_t i11     = 0;"))
    # m7: the per-expert D2H stops selecting the token.
    expect_fail("per-expert-token-fixed", sub("const int64_t i12     = entry.iid1;",
                                              "const int64_t i12     = 0;"))
    # m8: the per-expert D2H writes every slot to the same staging offset.
    expect_fail("per-expert-dst-collapsed", sub(
        "const size_t dst_off = (pool_base + ci) * static_cast<size_t>(K) * sizeof(float);",
        "const size_t dst_off = 0;"))
    # m10: the cold group shares the hot group's entries (the original hot/cold overwrite).
    expect_fail("cold-shares-hot-region", sub("const size_t cold_first = hot_first + hot_entries.size();",
                                              "const size_t cold_first = hot_first;",
                                              in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m11: the hot group is not handed its slice (it would reserve its own and the cold one overlaps).
    expect_fail("hot-not-given-slice", sub("dispatch_cpu_and_scatter(hot_entries, hot_first);",
                                           "dispatch_cpu_and_scatter(hot_entries);",
                                           in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m12: no single reservation for the split.
    expect_fail("split-without-reservation", sub("const size_t hot_first  = split_needs_pool ? hc_pool.reserve(n_cpu_entries) : 0;",
                                                 "const size_t hot_first  = 0;",
                                                 in_lambda=False, after=DISPATCH_MARKER), None, "S3")
    # m13: the split no longer checks the pool can hold both groups.
    expect_fail("split-unguarded-by-capacity", sub("(split_needs_pool && !hc_pool.can_serve(n_cpu_entries))) {",
                                                   "false) {", in_lambda=False, after=DISPATCH_MARKER), None, "S1")
    # M2: recipe types (Q1_0/NVFP4) reserve a span they never use.
    expect_fail("recipe-types-reserve-pool", sub("const bool   split_needs_pool = !immutable_host_recipe;",
                                                 "const bool   split_needs_pool = true;",
                                                 in_lambda=False, after=DISPATCH_MARKER), None, "S2")
    # m14: the memset returns to before the activation wait.
    expect_fail("memset-before-act-wait", sub(
        "const bool zero_out_after_act_wait = from_pool || !out_owner.valid();",
        "const bool zero_out_after_act_wait = from_pool || !out_owner.valid();\n"
        "                if (zero_out_after_act_wait) { std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float)); }"),
        None, "W1")
    # I2(c): the deferred wait gains a conjunct -- with n_cpu == 1 the memset and the CPU read run before the D2H lands.
    expect_fail("act-wait-needs-n-cpu-gt-1", sub("if (act_deferred_pending) {\n                    act_deferred_evt.wait();",
                                                 "if (act_deferred_pending && n_cpu > 1) {\n                    act_deferred_evt.wait();"),
                None, "W1")
    # I2: the memset moves under the wait's own block -- the guard is no longer exactly the wait.
    expect_fail("memset-inside-wait-block", sub(
        "if (act_deferred_pending) {\n                    act_deferred_evt.wait();\n                }\n                if (zero_out_after_act_wait) {\n                    std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float));\n                }",
        "if (act_deferred_pending) {\n                    act_deferred_evt.wait();\n                    if (zero_out_after_act_wait) {\n                        std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float));\n                    }\n                }"),
        None, "W1")
    # I2: the wait is exact, but the memset that must follow it in control flow can be skipped.
    expect_fail("memset-conditioned-away", sub(
        "if (zero_out_after_act_wait) {\n                    std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float));",
        "if (zero_out_after_act_wait && n_cpu > 1) {\n                    std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float));"),
        None, "W2")
    # W3: the memset's guard is defined away, so the zeroing is silently dropped (W1/W2 text still matches).
    expect_fail("memset-guard-defined-false", sub(
        "const bool zero_out_after_act_wait = from_pool || !out_owner.valid();",
        "const bool zero_out_after_act_wait = false;"), None, "W3")
    # W2 (structure): the wait+memset pair, textually adjacent and exact, wrapped in a block of its own.
    pair = ("if (act_deferred_pending) {\n                    act_deferred_evt.wait();\n                }\n"
            "                if (zero_out_after_act_wait) {\n                    std::memset(out_pinned, 0, n_cpu * static_cast<size_t>(N) * sizeof(float));\n                }")
    expect_fail("wait-memset-pair-wrapped-in-if-false", sub(pair, "if (false) {\n                " + pair + "\n                }"), None, "W2")
    # F2 (order): the consumed and pipeline flushes exchanged.
    consumed = "            flush_pending_cpu_scatter_if_consumed(dst, ctx.device);\n"
    pipeline = "            flush_pending_cpu_pipeline_if_consumed(dst, ctx.device);\n"
    expect_fail("consumed-and-pipeline-flushes-swapped", sub(consumed + pipeline, pipeline + consumed,
                                                             in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F2")
    # F3/F1 (structure): present, in order, exactly once -- but dead (the body of `if (false)`).
    expect_fail("try-flush-under-if-false", sub("            try_flush_pending_cpu_scatter();\n",
                                                "            if (false) try_flush_pending_cpu_scatter();\n",
                                                in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F3")
    expect_fail("consumed-flush-under-if-false", sub(consumed, "            if (false) flush_pending_cpu_scatter_if_consumed(dst, ctx.device);\n",
                                                     in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F1")
    # I2(a)/(b): the op-entry flushes that put the earlier scatter's H2D ahead of this op's D2H.
    expect_fail("try-flush-dropped", sub("            try_flush_pending_cpu_scatter();\n", "",
                                         in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F3")
    expect_fail("consumed-flush-dropped", sub("            flush_pending_cpu_scatter_if_consumed(dst, ctx.device);\n", "",
                                              in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F1")
    expect_fail("pipeline-flush-dropped", sub("            flush_pending_cpu_pipeline_if_consumed(dst, ctx.device);\n", "",
                                              in_lambda=False, after='"[MoE-HYBRID] ne12=%ld hybrid_active'), None, "F2")
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
        "const size_t dst_off = (pool_base + ci) * static_cast<size_t>(K) * sizeof(float);",
        "const size_t dst_off = ci * static_cast<size_t>(K) * sizeof(float);"))
    # m20: a host wait comes back into the split.
    expect_fail("host-wait-in-split", sub("flush_pending_cpu_scatter();\n                    }\n                    dispatch_cpu_and_scatter(cold_entries, cold_first);",
                                          "flush_pending_cpu_scatter();\n                        sycl::event::wait(g_pending_scatter.prev_bufs.scatter_events);\n                    }\n                    dispatch_cpu_and_scatter(cold_entries, cold_first);",
                                          in_lambda=False, after=DISPATCH_MARKER), None, "S4")
    # m21/m22: the ring itself.
    expect_fail("ring-never-wraps", backend_src, pool_src.replace("next_entry_ = 0;\n    }\n    const size_t first", "}\n    const size_t first"))
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
    print("SELF-TEST PASS: 34 mutants caught, unmodified tree passes")
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
        "each slot's own row, and the hot and cold dispatches use disjoint pool regions with no host wait (llama.cpp-4hg7)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
