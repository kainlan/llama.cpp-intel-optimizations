#!/usr/bin/env python3
"""The hybrid MUL_MAT_ID host-expert path makes no host wait after the ids readback (llama.cpp-z4kd).

THE COST.  Qwen3.8-Flash-Next IQ3_XXS on the B70 decodes at ~106-114 ms/token, of which the host MoE
stage is ~76 ms while the CPU expert kernel needs 7-11 ms.  Per MoE layer the submitting thread used to
block 4-6 times after the expert-id readback: the activation-copy and lease waits before submitting a
CPU job, future.get() on the gate/up jobs when GLU consumed them, the down activation-gather wait,
future.get() on the hot down group straight after issuing it, future.get() on the cold group at the
weighted sum, and an event wait on the previous scatter H2D at every flush.  Each one drained the GPU
queue and idled it while the host woke up and submitted the next op.

THE CONTRACT.  The expert-id readback (ggml_sycl_copy_ids_to_host, before the hybrid branch) stays as
the one host block per layer.  Everything after it is event-chained:

  * the waits a CPU job needs before it writes its pool span (its activation copy, the producer events
    of its host weight leases, an earlier scatter H2D it would alias) run on the pool worker, through
    moe_cpu_job_wait_inputs, which only moe_cpu_job_prologue's returned closure calls;
  * a CPU job's completion is joined by a host_task on the op's in-order queue
    (moe_cpu_job_join_on_queue, whose host_task calls moe_cpu_job_join, the one function that takes
    the job's future.get()), and the scatter H2D copies are submitted with that host_task's event as a
    dependency.

Pinned with comments and string literals blanked, over the scanned code: the live hybrid block (the
inner `if (moe_hybrid_with_plan)` of ggml_sycl_mul_mat_id), the lambdas of ggml_sycl_mul_mat_id that the
block calls (defined before it and in scope there), the submitting-thread helpers it calls (the
scatter-slot flushes and the sibling move), and, transitively, every function defined in ggml-sycl.cpp
that any of those calls, except the BOUNDARY entries (each listed with the reason its waits fall outside
this contract; an entry the file no longer defines or the scan no longer reaches fails the gate).
moe_cpu_job_prologue and moe_cpu_job_join_on_queue are scanned like any other callee, apart from the
body of the closure each hands to another thread (OFF_THREAD_SPANS: the lambda the prologue returns runs
on the pool worker, the lambda the join gives host_task runs on the queue's host-task thread).  A
closure's capture list stays in the scan, since it is evaluated where the closure is built.  The gate
fails when the closure can no longer be found, or when it is not the whole expression handed off (a
lambda the submitting thread calls to build the closure runs there, not on the other thread).  One
wait inside the scanned code predates this ticket and is excused by name in KNOWN_WAITS, at exactly one
match, with the ticket that removes it; an entry that matches nothing fails the gate, so that list can
only shrink:

  N1. no `.get()`, `.wait(`, `::wait(`, `wait_and_throw`, `wait_for`, `wait_until`, `*_sync(` call and
      no synchronous `mem_copy(`/`mem_fill(`, except the one allowlisted multi-GPU statement below;
  N2. moe_cpu_job_wait_inputs and the join's future are never touched there directly: the scanned code
      calls only moe_cpu_job_prologue (which builds the worker closure) and moe_cpu_job_join_on_queue;
  N3. in moe_cpu_job_prologue the wait call sits inside the returned lambda, not in the factory body;
  N4. moe_cpu_job_join takes the job's `.get()`, and every call of it, like every `.get()` in
      moe_cpu_job_join_on_queue, sits inside a `host_task(` argument of moe_cpu_job_join_on_queue;
  N5. flush_pending_cpu_scatter_slot submits its H2D copies with the join events among their deps, both
      its own per-run copies and the compact form's (moe_host_scatter_submit_compact, whose every copy
      takes the dependency list it is handed).
      It is the only flush that submits a CPU job's scatter: the opt-in pipeline slot that had a
      flush of its own (GGML_SYCL_PIPELINE_CPU) is gone, subsumed by this one.
  N6. the graph-boundary wait for the last scatters (pending_cpu_scatter_wait_prev_events) is called
      only from ggml_sycl_cpu_tg_flush_pending, which still calls it.

ALLOWLIST (exact statement, exactly once): `wait_pending_secondary_scatter_events(
flush_pending_secondary_scatter());` at op entry, taken only when the secondary-GPU merge ring is full.
That ring is filled only by the opt-in multi-GPU MoE path (GGML_SYCL_MOE_MULTI_GPU), which runs experts
on a second card, not on the CPU, and is outside this ticket.

WHAT THIS DOES NOT PROVE.  It reads source text.  It follows a call only into a function defined at
column 0 of ggml-sycl.cpp, or from the hybrid block into a lambda of ggml_sycl_mul_mat_id.  It does not
follow a call into another file (mem-ops.cpp's synchronous mem_copy/mem_fill are caught at the call by
name), through a function pointer, a std::function value or a lambda another function holds, into a
macro's body (a macro call is read as written, unexpanded), into a definition whose `template <...>`
header shares its line (DEF_SKIP drops such lines; the scanned code calls none today), or into a
BOUNDARY entry.  Of ggml_sycl_mul_mat_id's lambdas it follows only those bound as `auto name = [...]`:
not one bound by reference (`auto && name = [...]`), and not one called through an alias (`auto other =
name;` then `other()`); no code uses either form today.  A helper defined twice is scanned in every
definition.  It does not show the event chain is complete (a missing dependency is a race, not a wait),
and it does not measure anything.  The lead-run GPU arms (Qwen3.8 md5, ABBA llama-bench, chat and
Mistral gates) do.

NOT VACUOUS.  --self-test first requires the unmodified tree to pass (otherwise every mutant would fail
for the tree's own reason), then runs the gate on in-memory mutants of the real source, each re-adding
one of the removed waits or breaking one pin, and requires every one to FAIL.

THE RED, recorded against master 82fa573bb before the fix.  In the hybrid block: the down gather's
wait, the deferred activation-copy and lease-ready waits, the two waits on the previous scatters, and
the census timers of the gather and scatter waits (a `wait_timer(` reads as a wait_* helper).  In the
functions the slot flush calls: pending_cpu_scatter_join's future.get(), its census timer and its
zero-timeout wait_for trace probe (one statement, two matches), and flush_prev_scatter_bufs' wait on
the previous scatter with its timer.  And allocate_managed_host_pinned's zero-fill, reached from the
hybrid block's CPU dispatch, which KNOWN_WAITS excuses (llama.cpp-yzwy):
  FAIL [N1]: 13 host wait(s) on the submitting thread after the ids readback (llama.cpp-z4kd)
A scan of the regions alone, without following their callees, reported the first 7.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
TAG = "llama.cpp-z4kd"

HYBRID_MARKER = "        if (moe_hybrid_with_plan) {"
HELPERS = (
    "static void flush_pending_cpu_scatter_slot(",
    "static void flush_pending_cpu_scatter()",
    "static bool flush_pending_cpu_scatter_if_consumed(",
    "static void move_pending_cpu_scatter_to_sibling(",
)
# The flushes that submit scatter H2D copies: each joins its CPU job on the queue (N2) and makes every
# copy depend on that join (N5).
JOINING_FLUSHES = ("flush_pending_cpu_scatter_slot",)
PROLOGUE_FN = "moe_cpu_job_prologue("
WAIT_FN = "moe_cpu_job_wait_inputs("
JOIN_FN = "moe_cpu_job_join_on_queue("
PROLOGUE_DEF = "static std::function<void()> moe_cpu_job_prologue("
JOIN_DEF = "static sycl::event moe_cpu_job_join_on_queue("
JOINER_FN = "moe_cpu_job_join("
JOINER_DEF = "static void moe_cpu_job_join("
COMPACT_FN = "moe_host_scatter_submit_compact("
COMPACT_DEF = "static moe_host_scatter_form moe_host_scatter_submit_compact("
BOUNDARY_WAIT = "pending_cpu_scatter_wait_prev_events("
BOUNDARY_FN = "void ggml_sycl_cpu_tg_flush_pending() {"

ALLOWED = ("wait_pending_secondary_scatter_events(flush_pending_secondary_scatter());",)

# Functions the transitive scan does not descend into, each with the reason its waits fall outside this
# contract.  The gate fails on an entry the file no longer defines or the scan no longer reaches, so the
# list cannot outlive the code it excuses.
BOUNDARY = {
    "ggml_sycl_mul_mat": (
        "the GPU expert route's per-entry fallback, the generic MUL_MAT dispatch, taken only when no pointer "
        "table is uploaded (a cross-device plan, or several routable cards without a cohesive plan) or "
        "mmvq_moe_batched_dispatch declines.  Every expert type of the Qwen3.8 IQ3_XXS GGUF (IQ2_XXS, IQ2_XS, "
        "IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0) has an AoS batched kernel in moe-mmvq-tables.hpp, so a "
        "single-card Qwen3.8 decode does not take it per token; a decline prints [MOE-ROW-AGG] "
        "stage=dispatch-reject under GGML_SYCL_MOE_ROW_AGG_DEBUG"
    ),
    "dispatch_experts_secondary_gpu_impl": (
        "the multi-GPU MoE path, reached only while g_moe_multi_gpu_active, which needs "
        "GGML_SYCL_MOE_MULTI_GPU with two or more cards or a placement plan that uses a second card; a "
        "single-card decode never runs it"
    ),
    "flush_pending_secondary_scatter": (
        "half of the allowlisted multi-GPU ring statement (ALLOWED); the ring holds only the multi-GPU "
        "path's scatters, so a single-card decode never reaches it"
    ),
    "wait_pending_secondary_scatter_events": "the other half of that allowlisted statement",
    "pipeline_scatter_start_async": (
        "the multi-GPU scatter pipeline, behind ggml_sycl_pipeline_moe_enabled(), which returns false, and "
        "g_moe_multi_gpu_active"
    ),
    "moe_prestage_popular_experts": (
        "the placement prestage: g_prestage_completed latches it to one pass per model load, and it is a "
        "no-op when a placement plan exists unless GGML_SYCL_LEGACY_PRESTAGE=1.  It is also the only "
        "waiting callee of moe_periodic_rerank, which the scan does follow"
    ),
    "ggml_sycl_info": "the device-info singleton: a function-local static, initialised once per process",
    "ggml_sycl_upload_moe_transient_ptr_table": (
        "the GPU expert route's pointer table, outside this ticket's host path.  Its queue waits are the zero "
        "fill when ggml_sycl_ensure_moe_ptr_table creates the table and the queue drain on that function's "
        "allocation-failure path (ggml_sycl_drain_direct_stage_queue).  Its payload staging calls the "
        "synchronous mem_copy host to host, the payload into pinned staging, which mem-ops.cpp serves with a "
        "memcpy and no dependency, so it waits on no device work"
    ),
}

# Functions the submitting thread runs only in part: each is scanned like any other callee, callees
# followed, apart from the body of the closure named here, which another thread runs.  The closure's
# capture list stays in the scan (it is evaluated where the closure is built).  Each entry is the
# pattern that opens the closure, the character that must follow the closure's body (the `;` ending the
# return, the `)` closing host_task), and what the closure is.  The gate fails when the pattern no
# longer finds exactly one closure, or when the closure is followed by anything else, such as the `(`
# of a lambda called on the submitting thread to build the closure; so a rewrite cannot blank the wrong
# code or none.
OFF_THREAD_SPANS = {
    "moe_cpu_job_prologue": (
        r"\breturn\s*\[",
        ";",
        "the lambda it returns, which the pool worker runs before the job",
    ),
    "moe_cpu_job_join_on_queue": (
        r"\bhost_task\s*\(\s*\[",
        ")",
        "the lambda it hands host_task, which the queue's host-task thread runs",
    ),
}

# The function the hybrid block sits in.  The block calls some of its lambdas, which run on the
# submitting thread like the block, so the scan follows a call from the block into one defined before
# the block and in scope there.
ENCLOSING_DEF = "static void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {"
HYBRID_LABEL = "the hybrid MUL_MAT_ID block"
LOCAL_LAMBDA = re.compile(r"\bauto\s+(\w+)\s*=\s*\[")

# Waits the scan finds inside the contract that predate llama.cpp-z4kd and stay for now, each keyed by
# (function, FORBIDDEN label) and excused at exactly one match.  An entry that matches nothing fails the
# gate, and a second match in the same function is reported, so the list can only shrink.
KNOWN_WAITS = {
    ("allocate_managed_host_pinned", "synchronous mem_fill("): (
        "the zero_init fill, taken by the CPU dispatch's over-capacity managed output and by every Q1_0/NVFP4 "
        "recipe workspace; llama.cpp-yzwy moves the zeroing to the job prologue"
    ),
}

DEF_SKIP = re.compile(r"(struct|class|namespace|enum|union|typedef|using|template\b|extern\s+\"|#)")
NOT_FUNCTIONS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else", "case", "alignof", "decltype",
    "static_assert", "defined",
}
CALL = re.compile(r"(?<![.\w])(?<!->)(\w+)\s*\(")

FORBIDDEN = (
    (re.compile(r"(\.|->)\s*get\s*\(\s*\)"), "future-style .get()"),
    (re.compile(r"(\.|->|::)\s*wait\s*\("), "wait()"),
    (re.compile(r"\bwait_and_throw\b"), "wait_and_throw"),
    (re.compile(r"\bwait_for\b|\bwait_until\b"), "wait_for/wait_until"),
    (re.compile(r"\w_sync\s*\("), "a *_sync( call"),
    (re.compile(r"\bmem_copy\s*\("), "synchronous mem_copy("),
    (re.compile(r"\bmem_fill\s*\("), "synchronous mem_fill("),
    (re.compile(r"\bwait_\w+\s*\("), "a wait_* helper"),
)


class ContractError(AssertionError):
    pass


def blank_comments_and_strings(src: str) -> str:
    """Blank // and /* */ comments and the insides of string/char literals, keeping offsets and newlines."""
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
        elif src[i] in "\"'":
            q = src[i]
            j = i + 1
            while j < n and src[j] != q:
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(q + "".join("\n" if c == "\n" else " " for c in src[i + 1 : j]) + q)
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def matching(src: str, open_at: int, open_ch: str = "{", close_ch: str = "}") -> int:
    depth = 0
    for k in range(open_at, len(src)):
        c = src[k]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return k
    raise ContractError(f"FAIL: unbalanced {open_ch}{close_ch} block ({TAG})")


def function_body(code: str, signature: str, required: bool = True) -> tuple[int, str] | None:
    hits = [m.start() for m in re.finditer(re.escape(signature), code)]
    defs = []
    for at in hits:
        brace = code.find("{", at)
        semi = code.find(";", at)
        if brace >= 0 and (semi < 0 or brace < semi):
            defs.append((at, brace))
    if not defs:
        if required:
            raise ContractError(f"FAIL: definition of `{signature}` not found in ggml-sycl.cpp ({TAG})")
        return None
    if len(defs) > 1:
        raise ContractError(f"FAIL: `{signature}` is defined {len(defs)} times; the gate needs one ({TAG})")
    at, brace = defs[0]
    return at, code[brace : matching(code, brace) + 1]


def hybrid_block(code: str) -> tuple[int, str]:
    hits = [m.start() for m in re.finditer(re.escape(HYBRID_MARKER), code)]
    if len(hits) != 1:
        raise ContractError(
            f"FAIL: expected exactly one live hybrid block `{HYBRID_MARKER.strip()}`, found {len(hits)} ({TAG})"
        )
    brace = code.find("{", hits[0])
    block = code[brace : matching(code, brace) + 1]
    for anchor in ("dispatch_cpu_compute", "apply_cpu_result_to_scatter", "flush_pending_cpu_scatter("):
        if anchor not in block:
            raise ContractError(f"FAIL: the hybrid block no longer contains `{anchor}`; the gate is blind ({TAG})")
    return hits[0], block


def line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def lambda_body(text: str, intro: int) -> tuple[int, int] | None:
    """The offsets of the `{` and `}` of the body of the lambda whose capture list opens at `intro`."""
    j = matching(text, intro, "[", "]") + 1
    while j < len(text) and text[j].isspace():
        j += 1
    if j < len(text) and text[j] == "(":
        j = matching(text, j, "(", ")") + 1
    brace = text.find("{", j)
    semi = text.find(";", j)
    if brace < 0 or 0 <= semi < brace:
        return None
    return brace, matching(text, brace)


def off_thread_blanked(name: str, body: str) -> tuple[str, str | None]:
    """`body` with the body of the closure OFF_THREAD_SPANS names blanked, and None; or `body` unchanged
    and why the closure could not be found."""
    pattern, terminator, _ = OFF_THREAD_SPANS[name]
    found = [m.end() - 1 for m in re.finditer(pattern, body)]
    if len(found) != 1:
        return body, f"{name} has {len(found)} closures matching `{pattern}`; the gate needs exactly one"
    span = lambda_body(body, found[0])
    if span is None:
        return body, f"the closure in {name} has no body the gate can find"
    open_at, close_at = span
    after = body[close_at + 1 :].lstrip()[:1]
    if after != terminator:
        return body, (
            f"the closure in {name} is followed by `{after}`, not `{terminator}`, so it is not the whole "
            f"handed-off expression"
        )
    inner = "".join("\n" if c == "\n" else " " for c in body[open_at + 1 : close_at])
    return body[: open_at + 1] + inner + body[close_at:], None


def enclosing_lambdas(code: str, block_at: int) -> dict[str, tuple[int, str]]:
    """The `auto name = [...](...) {...}` lambdas of ggml_sycl_mul_mat_id defined before the hybrid block
    and in scope at it: name -> (offset of the body's `{`, body)."""
    fn_at, fn = function_body(code, ENCLOSING_DEF)
    fn_brace = code.find("{", fn_at)
    rel = block_at - fn_brace
    if not 0 < rel < len(fn):
        raise ContractError(
            f"FAIL: the hybrid block is not inside `{ENCLOSING_DEF}`; the gate cannot find the lambdas it calls ({TAG})"
        )
    head = fn[:rel]
    found = [(m.start(), m.group(1), m.end() - 1) for m in LOCAL_LAMBDA.finditer(head)]
    # The innermost brace open at each definition; the definition is in scope at the block when that
    # brace is still open there.
    scope_of: list[int] = []
    stack: list[int] = []
    k = 0
    for b in re.finditer(r"[{}]", head):
        while k < len(found) and found[k][0] < b.start():
            scope_of.append(stack[-1])
            k += 1
        if b.group() == "{":
            stack.append(b.start())
        else:
            stack.pop()
    while k < len(found):
        scope_of.append(stack[-1])
        k += 1
    open_at_block = set(stack)
    out: dict[str, tuple[int, str]] = {}
    for (_, name, intro), scope in zip(found, scope_of):
        span = lambda_body(head, intro) if scope in open_at_block else None
        if span is not None:
            out[name] = (fn_brace + span[0], head[span[0] : span[1] + 1])
    return out


def file_functions(code: str) -> dict[str, list[tuple[int, str]]]:
    """Every function defined at column 0 of `code`: name -> [(offset of its `{`, body)], one per definition.

    A definition is a column-0 line whose first `name(...)` is followed by `{` (after an optional
    const/noexcept/override) before any `;`, `{` or `=`.  Lambdas and struct members are not column-0
    definitions; calls to members go through `.` or `->`, which CALL skips.
    """
    defs: dict[str, list[tuple[int, str]]] = {}
    for line in re.finditer(r"^[A-Za-z_][^\n]*", code, re.M):
        if DEF_SKIP.match(line.group(0)):
            continue
        ls = line.start()
        for m in re.finditer(r"(?<![.>:\w])(\w+)\s*\(", code[ls : ls + 800]):
            open_at = ls + m.end() - 1
            head = code[ls:open_at]
            if ";" in head or "{" in head or "=" in head.replace("operator=", ""):
                break
            close_at = matching(code, open_at, "(", ")")
            tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?\{", code[close_at + 1 : close_at + 120])
            if tail:
                if m.group(1) not in NOT_FUNCTIONS:
                    brace = close_at + tail.end()
                    defs.setdefault(m.group(1), []).append((brace, code[brace : matching(code, brace) + 1]))
                break
    return defs


def scanned_code(
    regions: list[tuple[str, int, str]],
    defs: dict[str, list[tuple[int, str]]],
    lambdas: dict[str, tuple[int, str]],
) -> tuple[list[tuple[str, int, str]], set[str], set[str]]:
    """The regions plus every in-file function they call, transitively, short of BOUNDARY, and the
    enclosing function's lambdas the hybrid block calls (and those lambdas call).

    Returns the scanned (label, offset, text) list, each callee labelled with the call chain that reached
    it, the BOUNDARY names some scanned code calls, and the names of everything scanned.
    """
    out = list(regions)
    via: dict[str, str | None] = {name: None for name, _, _ in regions}
    reached: set[str] = set()
    queue = [(name, text, name == HYBRID_LABEL) for name, _, text in regions]
    while queue:
        caller, text, sees_lambdas = queue.pop(0)
        for m in CALL.finditer(text):
            callee = m.group(1)
            if sees_lambdas and callee in lambdas:
                key, bodies, local = f"lambda {callee}", [lambdas[callee]], True
            elif callee in defs:
                if callee in BOUNDARY:
                    reached.add(callee)
                    continue
                key, bodies, local = callee, defs[callee], False
            else:
                continue
            if key in via:
                continue
            via[key] = caller
            chain = [key]
            while via.get(chain[-1]) is not None:
                chain.append(via[chain[-1]])
            label = f"{key} (reached via {' <- '.join(chain[1:])})"
            for offset, body in bodies:
                out.append((label, offset, body))
                queue.append((key, body, local))
    return out, reached, set(via)


def inside_call(text: str, pos: int, callee: str) -> bool:
    """True when `pos` lies inside the parentheses of some `callee(` call in `text`."""
    for m in re.finditer(re.escape(callee) + r"\s*\(", text):
        open_at = m.end() - 1
        close_at = matching(text, open_at, "(", ")")
        if open_at < pos < close_at:
            return True
    return False


def inside_lambda_body(text: str, pos: int) -> bool:
    """True when `pos` lies inside the `{...}` body of a lambda (`[...](...) {` or `[...] {`) in `text`."""
    for m in re.finditer(r"\[[^\[\]]*\]\s*(\([^()]*\))?\s*(mutable\s*)?\{", text):
        open_at = m.end() - 1
        close_at = matching(text, open_at)
        if open_at < pos < close_at:
            return True
    return False


def check(backend_src: str) -> None:
    code = blank_comments_and_strings(backend_src)

    regions: list[tuple[str, int, str]] = []
    at, block = hybrid_block(code)
    regions.append((HYBRID_LABEL, at, block))
    for sig in HELPERS:
        found = function_body(code, sig)
        regions.append((sig.replace("static ", "").split("(")[0].split()[-1], found[0], found[1]))
    defs = file_functions(code)
    # A function that is missing, or whose closure cannot be found, is scanned whole (stricter) and
    # reported after N1, so a tree without the event chain still shows its waits.
    off_thread_problems = []
    for name in OFF_THREAD_SPANS:
        bodies = defs.get(name, [])
        if len(bodies) != 1:
            off_thread_problems.append((name, f"{name} is defined {len(bodies)} times in ggml-sycl.cpp"))
            continue
        blanked, problem = off_thread_blanked(name, bodies[0][1])
        if problem:
            off_thread_problems.append((name, problem))
        defs[name] = [(bodies[0][0], blanked)]
    scanned, reached, followed = scanned_code(regions, defs, enclosing_lambdas(code, at))

    # N1: no host wait, apart from the exact allowlisted statement, in the regions or any function they
    # reach short of BOUNDARY.
    hits = []
    allowed_seen = {a: 0 for a in ALLOWED}
    known_seen = {k: 0 for k in KNOWN_WAITS}
    for name, base, text in scanned:
        squashed_lines = text
        masked = squashed_lines
        for a in ALLOWED:
            pat = re.compile(r"\s*".join(re.escape(tok) for tok in re.findall(r"\w+|[^\w\s]", a)))
            for m in pat.finditer(masked):
                allowed_seen[a] += 1
                masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]
        for pat, what in FORBIDDEN:
            for m in pat.finditer(masked):
                key = (name.split(" ")[0], what)
                if key in KNOWN_WAITS and known_seen[key] == 0:
                    known_seen[key] += 1
                    continue
                hits.append(f"{name} line {line_of(code, base) + text.count(chr(10), 0, m.start())}: {what}")
    if hits:
        raise ContractError(
            f"FAIL [N1]: {len(hits)} host wait(s) on the submitting thread after the ids readback "
            f"({TAG}):\n  " + "\n  ".join(hits)
        )
    for a, n in allowed_seen.items():
        if n > 1:
            raise ContractError(f"FAIL [N1]: the allowlisted `{a}` appears {n} times; it is allowed once ({TAG})")
    for (fn, what), n in known_seen.items():
        if n == 0:
            raise ContractError(
                f"FAIL [N1]: KNOWN_WAITS entry `{fn}` / {what} matches nothing in the scanned code; drop it ({TAG})"
            )
    if off_thread_problems:
        name, problem = off_thread_problems[0]
        raise ContractError(
            f"FAIL [N1]: {problem}, so the scan cannot set apart {OFF_THREAD_SPANS[name][2]} ({TAG})"
        )
    for name in OFF_THREAD_SPANS:
        if name not in followed:
            raise ContractError(
                f"FAIL [N1]: OFF_THREAD_SPANS entry `{name}` is no longer called from the scanned code; drop it ({TAG})"
            )
    for name in BOUNDARY:
        if name not in defs:
            raise ContractError(
                f"FAIL [N1]: BOUNDARY entry `{name}` is not defined in ggml-sycl.cpp; drop it or rename it ({TAG})"
            )
        if name not in reached:
            raise ContractError(
                f"FAIL [N1]: BOUNDARY entry `{name}` is no longer called from the scanned code; drop it ({TAG})"
            )

    # N2: the scanned code never calls the worker-side wait, and the flush does join through the queue
    # helper.
    for name, base, text in scanned:
        if WAIT_FN in text:
            raise ContractError(
                f"FAIL [N2]: {name} calls {WAIT_FN.rstrip('(')} on the submitting thread; only the closure "
                f"{PROLOGUE_FN.rstrip('(')} returns may call it, on the pool worker ({TAG})"
            )
    flushes = [(n, t) for n, _, t in regions if n in JOINING_FLUSHES]
    for fname, flush_text in flushes:
        if JOIN_FN not in flush_text:
            raise ContractError(
                f"FAIL [N2]: {fname} does not join the CPU job through {JOIN_FN.rstrip('(')} ({TAG})"
            )
    if PROLOGUE_FN not in block:
        raise ContractError(
            f"FAIL [N2]: the hybrid block does not hand its CPU jobs a worker prologue ({PROLOGUE_FN.rstrip('(')}); "
            f"the activation and lease waits have nowhere to run off the submitting thread ({TAG})"
        )

    # N3: the factory's wait is inside the returned lambda.
    _, prologue = function_body(code, PROLOGUE_DEF)
    waits = [m.start() for m in re.finditer(re.escape(WAIT_FN), prologue)]
    if not waits or any(not inside_lambda_body(prologue, w) for w in waits):
        raise ContractError(
            f"FAIL [N3]: moe_cpu_job_prologue calls {WAIT_FN.rstrip('(')} outside the closure it returns, so the "
            f"submitting thread would wait ({TAG})"
        )
    for pat, what in FORBIDDEN:
        for m in pat.finditer(prologue):
            if not inside_lambda_body(prologue, m.start()):
                raise ContractError(f"FAIL [N3]: moe_cpu_job_prologue makes a {what} outside its closure ({TAG})")

    # N4: the join runs inside its host_task.  The host_task calls moe_cpu_job_join, which takes the job's
    # get(); nothing outside that host_task calls moe_cpu_job_join.
    join_at, join = function_body(code, JOIN_DEF)
    join_body_at = code.find("{", join_at)
    _, joiner = function_body(code, JOINER_DEF)
    if not re.search(r"(\.|->)\s*get\s*\(\s*\)", joiner):
        raise ContractError(f"FAIL [N4]: moe_cpu_job_join no longer takes the CPU job's get() ({TAG})")
    joins_in_task = [m.start() for m in re.finditer(r"(\.|->)\s*get\s*\(\s*\)|\b" + re.escape(JOINER_FN), join)]
    if not joins_in_task or any(not inside_call(join, g, "host_task") for g in joins_in_task):
        raise ContractError(
            f"FAIL [N4]: moe_cpu_job_join_on_queue joins the CPU job outside its host_task, on the "
            f"submitting thread ({TAG})"
        )
    for pat, what in FORBIDDEN[1:]:
        for m in pat.finditer(join):
            if not inside_call(join, m.start(), "host_task"):
                raise ContractError(f"FAIL [N4]: moe_cpu_job_join_on_queue makes a {what} outside its host_task ({TAG})")
    stray = []
    for m in re.finditer(r"\b" + re.escape(JOINER_FN), code):
        if code.startswith(JOINER_DEF, m.start() - len("static void ")):
            continue  # the definition
        rel = m.start() - join_body_at
        if 0 <= rel < len(join) and inside_call(join, rel, "host_task"):
            continue
        stray.append(line_of(code, m.start()))
    if stray:
        raise ContractError(
            f"FAIL [N4]: moe_cpu_job_join is called outside the join host_task (line(s) {stray}); the job would "
            f"be joined on the submitting thread ({TAG})"
        )

    # N5: the scatter H2D depends on the join events, in every joining flush and in the compact form.
    for fname, flush_text in flushes:
        dep_vars = set()
        for m in re.finditer(r"(?:const\s+)?sycl::event\s+(\w+)\s*=\s*moe_cpu_job_join_on_queue\s*\(", flush_text):
            dep_vars.add(m.group(1))
        for m in re.finditer(r"(\w+)\s*\.\s*push_back\s*\(\s*moe_cpu_job_join_on_queue\s*\(", flush_text):
            dep_vars.add(m.group(1))
        if not dep_vars:
            raise ContractError(f"FAIL [N5]: {fname} does not keep the join events ({TAG})")
        for ev in sorted(dep_vars):
            for m in re.finditer(r"std::vector<sycl::event>\s+(\w+)\s*\{\s*" + re.escape(ev) + r"\s*\}", flush_text):
                dep_vars.add(m.group(1))
        calls = [m.start() for m in re.finditer(r"\bmem_copy_async\s*\(|\b" + re.escape(COMPACT_FN), flush_text)]
        if not calls:
            raise ContractError(f"FAIL [N5]: {fname} submits no scatter copy ({TAG})")
        for c in calls:
            close_at = matching(flush_text, flush_text.find("(", c), "(", ")")
            args = flush_text[c:close_at]
            if not any(re.search(r"\b" + re.escape(v) + r"\b", args) for v in dep_vars):
                raise ContractError(
                    f"FAIL [N5]: a scatter H2D in {fname} does not depend on the join events "
                    f"`{'/'.join(sorted(dep_vars))}`; the copy could read the output before the CPU job wrote it ({TAG})"
                )
    compact_at, compact = function_body(code, COMPACT_DEF)
    params = code[compact_at : code.find("{", compact_at)]
    deps = re.search(r"const\s+std::vector<sycl::event>\s*&\s*(\w+)", params)
    if not deps:
        raise ContractError(f"FAIL [N5]: moe_host_scatter_submit_compact takes no dependency list for its copies ({TAG})")
    copies = [m.start() for m in re.finditer(r"\bmem_copy_async\s*\(", compact)]
    if not copies:
        raise ContractError(f"FAIL [N5]: moe_host_scatter_submit_compact submits no copy; the gate is blind ({TAG})")
    for c in copies:
        args = compact[c : matching(compact, compact.find("(", c), "(", ")")]
        if not re.search(r"\b" + re.escape(deps.group(1)) + r"\b", args):
            raise ContractError(
                f"FAIL [N5]: a copy in moe_host_scatter_submit_compact does not depend on `{deps.group(1)}`, the "
                f"joins it is handed; it could read the output before the CPU job wrote it ({TAG})"
            )

    # N6: the graph-boundary wait for the last scatters is called only from the graph-boundary flush.
    calls = [m.start() for m in re.finditer(r"\b" + re.escape(BOUNDARY_WAIT), code)]
    _, boundary = function_body(code, BOUNDARY_FN)
    boundary_at = code.find(BOUNDARY_FN)
    # The drain's own definitions (the per-slot overload and the all-slots drain that calls it).
    drain_bodies = []
    for m in re.finditer(r"^static\s+void\s+pending_cpu_scatter_wait_prev_events\s*\([^)]*\)\s*\{", code, re.M):
        drain_bodies.append((m.start(), matching(code, m.end() - 1)))
    outside = []
    for c in calls:
        line = code[code.rfind("\n", 0, c) + 1 : code.find("\n", c)]
        if any(a <= c <= b for a, b in drain_bodies):
            continue  # a definition, or the drain calling its per-slot overload
        if boundary_at <= c < boundary_at + len(boundary) + len(BOUNDARY_FN):
            continue
        outside.append(line_of(code, c))
    if outside:
        raise ContractError(
            f"FAIL [N6]: pending_cpu_scatter_wait_prev_events is called outside ggml_sycl_cpu_tg_flush_pending (line(s) "
            f"{outside}); it is a host wait for the graph boundary only ({TAG})"
        )
    if BOUNDARY_WAIT + ")" not in boundary:
        raise ContractError(
            f"FAIL [N6]: the graph-boundary flush no longer waits for the last scatters; the host zones could be "
            f"reset under a running CPU job (llama.cpp-skgik, {TAG})"
        )


def mutants(src: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def sub(name: str, old: str, new: str, count: int = 1) -> None:
        if old not in src:
            raise ContractError(f"FAIL: self-test anchor for mutant `{name}` not found: {old!r} ({TAG})")
        out.append((name, src.replace(old, new, count)))

    # The hybrid block's own graphs-off statement: the same line appears at many sites earlier in the
    # file, so the anchor runs from the hybrid marker through it, and a mutant lands inside the block.
    h_at = src.find(HYBRID_MARKER + "\n")
    g_line = "            ctx.moe_graphs_disabled_once = true;\n"
    g_at = src.find(g_line, h_at) if h_at >= 0 else -1
    graphs_off = src[h_at : g_at + len(g_line)] if g_at >= 0 else "<hybrid block graphs-off anchor missing>"
    sub("activation wait on the submitting thread", graphs_off,
        graphs_off + "            act_d2h_event.wait();\n")
    sub("queue drain in the hybrid block", graphs_off, graphs_off + "            stream->wait();\n")
    sub("wait_and_throw in the hybrid block", graphs_off, graphs_off + "            stream->wait_and_throw();\n")
    sub("event-list wait in the hybrid block", graphs_off,
        graphs_off + "            sycl::event::wait(std::vector<sycl::event>{});\n")
    sub("synchronous copy in the hybrid block", graphs_off,
        graphs_off + "            ggml_sycl::mem_copy(shared_act_handle, shared_act_handle, 0, *stream);\n")
    sub("worker wait called directly", graphs_off,
        graphs_off + "            moe_cpu_job_wait_inputs(moe_cpu_job_inputs{});\n")
    sub("second allowlisted secondary flush", graphs_off,
        graphs_off + "            wait_pending_secondary_scatter_events(flush_pending_secondary_scatter());\n")
    slot_open = "static void flush_pending_cpu_scatter_slot(pending_cpu_scatter & slot) {\n"
    sub("future.get in the slot flush", slot_open, slot_open + "    slot.future.get();\n")
    sub("previous-scatter wait in the slot flush", slot_open,
        slot_open + "    sycl::event::wait(slot.prev_scatter.scatter_events);\n")
    sub("slot flush stops joining through the queue", "moe_cpu_job_join_on_queue(*slot.stream",
        "moe_cpu_scatter_join_on_queue_unused(*slot.stream")
    sub("scatter H2D loses the join dependency", "run.src_offset, run.bytes, *slot.stream, joins));",
        "run.src_offset, run.bytes, *slot.stream, {}));")
    sub("compact scatter handed no joins", "moe_host_scatter_submit_compact(slot, ws, joins, &n_copies, &why)",
        "moe_host_scatter_submit_compact(slot, ws, {}, &n_copies, &why)")
    sub("compact copies drop their dependency", "copy.src_offset, copy.bytes, queue, deps));",
        "copy.src_offset, copy.bytes, queue));")
    sub("join get() outside its host_task", "static sycl::event moe_cpu_job_join_on_queue(sycl::queue & queue, std::shared_future<void> job) {\n",
        "static sycl::event moe_cpu_job_join_on_queue(sycl::queue & queue, std::shared_future<void> job) {\n    job.get();\n")
    sub("job join called outside its host_task", "static sycl::event moe_cpu_job_join_on_queue(sycl::queue & queue, std::shared_future<void> job) {\n",
        "static sycl::event moe_cpu_job_join_on_queue(sycl::queue & queue, std::shared_future<void> job) {\n    moe_cpu_job_join(job);\n")
    sub("job join called from the slot flush", slot_open, slot_open + "    moe_cpu_job_join(slot.future.share());\n")
    sub("join helper stops taking get()", "std::future_status::ready);\n    }\n    job.get();\n",
        "std::future_status::ready);\n    }\n")
    sub("prologue waits in the factory body", "static std::function<void()> moe_cpu_job_prologue(moe_cpu_job_inputs inputs) {\n",
        "static std::function<void()> moe_cpu_job_prologue(moe_cpu_job_inputs inputs) {\n    moe_cpu_job_wait_inputs(inputs);\n")
    sub("boundary wait called from a flush", "        ++g_cpu_scatter_serial;\n",
        "        ++g_cpu_scatter_serial;\n        pending_cpu_scatter_wait_prev_events(slot);\n")
    sub("boundary wait dropped from the graph-boundary flush",
        "    pending_cpu_scatter_wait_prev_events();  // The graph-boundary wait", "    // The graph-boundary wait")
    # A wait moved into a helper is still a wait.  Each helper is new, defined at file scope just before
    # the flush, and called from scanned code; none matches the wait_* helper pattern by name.
    sub("future.wait in a helper the slot flush calls", slot_open,
        "static void pending_cpu_scatter_join(pending_cpu_scatter & slot) {\n    slot.future.wait();\n}\n\n"
        + slot_open + "    pending_cpu_scatter_join(slot);\n")
    sub("prev-scatter wait in a helper the slot flush calls", slot_open,
        "static void drain_prev_scatter(pending_cpu_scatter & slot) {\n"
        "    sycl::event::wait(slot.prev_scatter.scatter_events);\n}\n\n"
        + slot_open + "    drain_prev_scatter(slot);\n")
    sub("activation wait in a helper the hybrid block calls", slot_open,
        "static void moe_settle_activation(sycl::event evt) {\n    evt.wait();\n}\n\n" + slot_open)
    # The call for the helper above, placed in the hybrid block; built on the first mutant's text so both
    # edits land together.
    settle = out[-1][1]
    s_h = settle.find(HYBRID_MARKER + "\n")
    s_g = settle.find(g_line, s_h)
    out[-1] = (out[-1][0], settle[: s_g + len(g_line)] + "            moe_settle_activation(sycl::event{});\n"
               + settle[s_g + len(g_line) :])
    # Positive control for the helper mutants: the same join wait written inline in the flush.
    sub("group future wait inline in the slot flush",
        "            joins.push_back(moe_cpu_job_join_on_queue(*slot.stream, group[k]->future.share()));\n",
        "            group[k]->future.wait();\n"
        "            joins.push_back(moe_cpu_job_join_on_queue(*slot.stream, group[k]->future.share()));\n")
    # Two calls deep: flush_pending_cpu_scatter_if_consumed -> ggml_sycl_moe_precomputed_mmid_skip_pending ->
    # ggml_sycl_moe_precomputed_skip_contains.
    deep = ("    moe_precomputed_skip_key key{};\n"
            "    return ggml_sycl_make_moe_precomputed_skip_key(tensor, device, &key)")
    sub("wait two calls below a scanned helper", deep, "    g_pending_scatter.future.wait();\n" + deep)
    sub("BOUNDARY entry no longer reached", "moe_prestage_popular_experts();",
        "moe_prestage_popular_experts_retired();", count=-1)
    sub("BOUNDARY entry no longer defined", "static void moe_prestage_popular_experts() {",
        "static void moe_prestage_popular_experts_v2() {")
    # moe_periodic_rerank left BOUNDARY: the scan follows it, so a wait in its own body is caught.
    sub("wait in the rerank the scan follows", "    int ranks_changed = 0;\n",
        "    g_pending_scatter.future.wait();\n    int ranks_changed = 0;\n")
    fill = "    if (zero_init) {\n        ggml_sycl::mem_fill(handle, 0, bytes, q);\n    }\n"
    sub("KNOWN_WAITS entry matches nothing", fill, "")
    sub("second fill beside the known one", "    *out = std::move(handle);\n    return true;\n}\n",
        "    ggml_sycl::mem_fill(handle, 0, bytes, q);\n    *out = std::move(handle);\n    return true;\n}\n")
    # The prologue and the join run on the submitting thread apart from the closure each hands off; the
    # rest of each body, its closures' capture lists and its callees are scanned.  The unmodified tree
    # passing is the other half: both closures hold a wait (moe_cpu_job_wait_inputs, moe_cpu_job_join's
    # get()) that the scan would report if it read them.
    join_def = (
        "static sycl::event moe_cpu_job_join_on_queue(sycl::queue & queue, std::shared_future<void> job) {\n"
    )
    sub("wait in a helper the join calls before its submit", join_def,
        "static void moe_join_settle(const std::shared_future<void> & j) {\n    j.wait();\n}\n\n"
        + join_def + "    moe_join_settle(job);\n")
    pro_def = "static std::function<void()> moe_cpu_job_prologue(moe_cpu_job_inputs inputs) {\n"
    sub("event wait in a helper the prologue factory calls", pro_def,
        "static void moe_settle_inputs(const moe_cpu_job_inputs & in) {\n    sycl::event::wait(in.events);\n}\n\n"
        + pro_def + "    moe_settle_inputs(inputs);\n")
    sub("worker wait behind a wrapper the prologue factory calls", pro_def,
        "static void moe_inputs_ready(const moe_cpu_job_inputs & in) {\n    moe_cpu_job_wait_inputs(in);\n}\n\n"
        + pro_def + "    moe_inputs_ready(inputs);\n")
    sub("wait in the host_task closure's capture list", "cgh.host_task([job]() { moe_cpu_job_join(job); });",
        "cgh.host_task([job = (job.wait(), job)]() { moe_cpu_job_join(job); });")
    sub("join hands host_task a named closure", "cgh.host_task([job]() { moe_cpu_job_join(job); });",
        "auto task = [job]() { moe_cpu_job_join(job); }; cgh.host_task(task);")
    sub("prologue definition renamed away", "static std::function<void()> moe_cpu_job_prologue(",
        "static std::function<void()> moe_cpu_job_prologue_v2(")
    # A lambda called on the submitting thread to build the handed-off closure runs there: only its result
    # is handed off.
    sub("join builds its host_task closure by calling a lambda", "cgh.host_task([job]() { moe_cpu_job_join(job); });",
        "cgh.host_task([job]() { job.wait(); return [job]() { moe_cpu_job_join(job); }; }());")
    sub("prologue builds its closure by calling a lambda",
        "    return [inputs = std::move(inputs)]() {\n        moe_cpu_job_wait_inputs(inputs);\n    };\n",
        "    return [&]() { sycl::event::wait(inputs.events); return std::function<void()>([inputs]() "
        "{ moe_cpu_job_wait_inputs(inputs); }); }();\n")
    # A lambda of ggml_sycl_mul_mat_id that the hybrid block calls runs on the submitting thread too.
    expert_id_line = "        local_extra.moe_expert_id     = expert_id;\n"
    sub("wait in a lambda of ggml_sycl_mul_mat_id the hybrid block calls", expert_id_line,
        expert_id_line + "        sycl::event{}.wait();\n")
    layout_line = "        return layout_bytes != 0 ? layout_bytes : expert_size;\n"
    sub("wait in a lambda that lambda calls", layout_line, "        sycl::event{}.wait();\n" + layout_line)
    sib = "static void move_pending_cpu_scatter_to_sibling() {\n"
    sub("future.get in the sibling move", sib, sib + "    g_pending_scatter.future.get();\n")
    sub("hybrid block renamed away", HYBRID_MARKER, "        if (moe_hybrid_with_plan && true) {")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true", help="require every in-memory mutant to fail first")
    ap.add_argument("--source", type=Path, default=BACKEND, help="ggml-sycl.cpp to check (default: this tree)")
    args = ap.parse_args()
    src = args.source.read_text()

    if args.self_test:
        # A mutant only proves a pin is load-bearing if the unmutated source passes; otherwise every
        # mutant "fails" for the unmodified source's reason.
        try:
            check(src)
        except ContractError as ex:
            print(ex)
            print(f"FAIL: the unmodified source fails, so the self-test would be vacuous ({TAG})")
            return 1
        try:
            built = mutants(src)
        except ContractError as ex:
            print(ex)
            return 1
        survivors = []
        for name, mutated in built:
            try:
                check(mutated)
            except ContractError:
                continue
            survivors.append(name)
        if survivors:
            print(f"FAIL: {len(survivors)} mutant(s) passed the gate: {', '.join(survivors)} ({TAG})")
            return 1
        print(f"self-test: {len(built)} mutants all fail as required")

    try:
        check(src)
    except ContractError as ex:
        print(ex)
        return 1
    print(f"OK: no host wait on the submitting thread after the ids readback ({TAG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
