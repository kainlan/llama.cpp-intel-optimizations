#!/usr/bin/env python3
"""The hybrid MUL_MAT_ID host-expert path makes no host wait after the ids readback (llama.cpp-z4kd).

THE COST.  Qwen3.8-Flash-Next IQ3_XXS on the B70 decodes at ~106-114 ms/token, of which the host MoE
stage is ~76 ms while the CPU expert kernel needs 7-11 ms.  Per MoE layer the submitting thread used to
block 4-6 times after the expert-id readback: the activation-copy and lease waits before submitting a
CPU job, future.get() on the gate/up jobs when GLU consumed them, the down activation-gather wait,
future.get() on the hot down group straight after issuing it, future.get() on the cold group at the
weighted sum, and an event wait on the previous scatter H2D at every flush.  Each one drained the GPU
queue and idled it while the host woke up and submitted the next op.

THE CONTRACT.  The expert-id readback (ggml_sycl_copy_ids_to_host, before the hybrid branch) is the one
host block per layer left on the retained route, and a decode op the event-chained hybrid route admits
does not reach it (N7).  Everything after it is event-chained:

  * the waits a CPU job needs before it writes its pool span (its activation copy, the producer events
    of its host weight leases, an earlier scatter H2D it would alias) run on the pool worker, through
    moe_cpu_job_wait_inputs, which only moe_cpu_job_prologue's returned closure calls.  Since
    llama.cpp-42zj that wait reads a device-written marker the prologue arms behind those events, and
    when the marker cannot be armed the prologue waits the events itself (KNOWN_WAITS);
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
lambda the submitting thread calls to build the closure runs there, not on the other thread).  Two
waits inside the scanned code are excused by name in KNOWN_WAITS, each at exactly one match.  One
predates this ticket and names the ticket that removes it.  The other is the input marker's refused-arm
fallback, which llama.cpp-42zj puts on the submitting thread on purpose: the pool worker must make no
driver wait, and a job whose marker is not armed has no other thread to wait on.  An entry that matches
nothing fails the gate, so the list cannot outlive the code it excuses:

  N1. no `.get()`, `.wait(`, `::wait(`, `wait_and_throw`, `wait_for`, `wait_until` or `*_sync(` call, no
      call of a `wait_*(` helper (a census timer's `wait_timer(` declaration reads as one), and no
      synchronous `mem_copy(`/`mem_fill(`, and no call of the input marker's spin wait
      (`cpu_job_input_marker_wait(`, a header helper the scan cannot follow into), apart from the
      allowlisted multi-GPU statement below, at most once, and the KNOWN_WAITS entries;
  N2. the exception lists are current: each KNOWN_WAITS entry matches, each OFF_THREAD_SPANS function is
      defined once, hands off one closure that is the whole handed-off expression, and is reached, and
      each BOUNDARY entry is defined and reached;
  N3. the closure that moe_cpu_job_prologue returns calls moe_cpu_job_wait_inputs, so a job's input
      waits still run, on the pool worker;
  N4. moe_cpu_job_join takes the job's `.get()`, the closure that moe_cpu_job_join_on_queue hands
      host_task calls moe_cpu_job_join, and no other code in the file calls it (a call from the scanned
      code is already an N1 wait, since the scan follows it into the get());
  N5. flush_pending_cpu_scatter_slot submits its H2D copies with the join events among their deps, both
      its own per-run copies and the compact form's (moe_host_scatter_submit_compact, whose every copy
      takes the dependency list it is handed), and the flagged form's (moe_host_scatter_submit_flagged,
      likewise) once it exists.
      It is the only flush that submits a CPU job's scatter: the opt-in pipeline slot that had a
      flush of its own (GGML_SYCL_PIPELINE_CPU) is gone, subsumed by this one.
  N6. the graph-boundary wait for the last scatters (pending_cpu_scatter_wait_prev_events) is called
      only from ggml_sycl_cpu_tg_flush_pending, which still calls it (a call from the scanned code is
      already an N1 wait).
  N7. the expert-id readback is not a wait on the submitting thread for a decode op whose experts
      split between the device and the host (llama.cpp-7pvo).  ggml_sycl_mul_mat_id calls
      ggml_sycl_moe_hybrid_decode_event_chained, and returns when it takes the op, before the B1
      readback.  That route is a scanned region, so N1 covers it and everything it calls.  It copies
      the ids into the pinned pool's ids slots with mem_copy_async, and that copy's event is one of the
      CPU job's inputs, so the pool worker waits it, not this thread.  The job, not the host, then
      decides which slots are the host's and flags them, and flush_pending_cpu_scatter_slot places
      the flagged rows through moe_host_scatter_submit_flagged, whose copies N5 checks like the
      compact form's.  ROUTE_BOUNDARY is N7's own exception list: each entry must be defined and
      reached.
  N8. that route takes the op in the default configuration (llama.cpp-7pvo).  None of its refusals
      names a mode that is always on: the MoE dispatch stats are hardwired on, and a refusal on them
      turned the route off for every op while N7 still passed.  Instead the route keeps the retained
      CPU-TG block's per-op observations: its CPU job's resolve step records the dispatch stats'
      residency and the predictor's actual experts on the pool worker, where the ids are, reading the
      ids for the predictor before moe_host_slots_from_ids overwrites them with flags; and the route
      ticks the stats' token on the submitting thread through the one tracker the retained block uses
      too, so a token split between the two routes ticks once.
  N9. that route admits a tensor whose experts all live on the host (llama.cpp-7pvo).  The placement
      fills MoE layers in order, so most host-expert layers hold no device expert at all: on Qwen3.8
      IQ3_XXS two of the 24 do.  A record that required a device expert served those two and left the
      other 22 reading their ids back, while N7 and N8 passed.  So the record carries `all_host`; the
      builder decides it from the plan (`counts.on_device == 0`) and records it; no builder refusal
      tests the device lookup (`materialized`, `substitute`) without `!all_host`, and none refuses on
      the all-host case itself, even as one `||` operand of its condition (`n_as < 0 || all_host`,
      `all_host == true`, `counts.on_device == 0`, either way round), unless it sits in an `if` only
      a device expert enters, where `all_host` means the plan counted no device expert yet one
      resolved there; and the route runs its table copy and device kernel only inside
      `if (!record->all_host) {...}` and never refuses an all-host record.  A refusal returns the
      outcome itself or through the builder's `refuse("reason")` helper, and a retry counts as one,
      since a retry that keeps failing settles as a refusal (moe_decode_direct_stamp_settle).  Every
      one of them must sit in an `if (...) {...}` the scan reads, the helper's own body or the final
      `else` of an if-chain, whose links the rule reads in place of a condition.  A link that tests
      `!all_host`, `all_host == false` or `false == all_host` anywhere in it, or that the plan put an
      expert on the device (`counts.on_device != 0`, `> 0`), makes the else where an all-host tensor
      goes, so it fails even inside `n_as < 0 || !all_host`; and a device-lookup link needs an
      `all_host || ...` link beside it.  A refusal written in another form fails the rule instead of
      escaping it.

ALLOWLIST (exact statement, at most once; both its calls are BOUNDARY entries, so N2 fails once it is
gone): `wait_pending_secondary_scatter_events(flush_pending_secondary_scatter());` at op entry, taken
only when the secondary-GPU merge ring is full.
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
one of the removed waits or breaking one pin, and requires every one to FAIL on the rule it names
(`setup` for one that removes what the gate locates).

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
# The flushes that submit scatter H2D copies: each keeps its CPU jobs' join events and makes every copy
# depend on them (N5).
JOINING_FLUSHES = ("flush_pending_cpu_scatter_slot",)
# The two functions that hand a closure to another thread (OFF_THREAD_SPANS).
PROLOGUE = "moe_cpu_job_prologue"
JOIN = "moe_cpu_job_join_on_queue"
WAIT_FN = "moe_cpu_job_wait_inputs("
JOINER_FN = "moe_cpu_job_join("
JOINER_DEF = "static void moe_cpu_job_join("
COMPACT_FN = "moe_host_scatter_submit_compact("
COMPACT_DEF = "static moe_host_scatter_form moe_host_scatter_submit_compact("
BOUNDARY_WAIT = "pending_cpu_scatter_wait_prev_events("
BOUNDARY_FN = "void ggml_sycl_cpu_tg_flush_pending() {"

# llama.cpp-7pvo (N7): the event-chained hybrid decode route and what it must look like.
TAG_7PVO = "llama.cpp-7pvo"
ROUTE = "ggml_sycl_moe_hybrid_decode_event_chained"
# The call, whitespace squashed: it returns when the route takes the op.
ROUTE_CALL = re.compile(r"if \(" + ROUTE + r"\(ctx, src0, src1, dst\)\) \{ return; \}")
B1_TIMER = re.compile(r"moe_hostpath_wait_timer\s+\w+\(\s*MOE_WAIT_B1\s*\)\s*;")
# The ids copy into the pinned pool's ids slots, bound to the event the job inputs must hold.
IDS_COPY = re.compile(r"\bsycl::event\s+(\w+)\s*=\s*ggml_sycl::mem_copy_async\(\s*pool\.ids_handle\(\)")
FLAGGED_FN = "moe_host_scatter_submit_flagged("
# N8: the route's refusals, the modes that are always on, its CPU job's resolve step, and the token tracker the
# route shares with the retained CPU-TG block.
ROUTE_REFUSAL = re.compile(r"\bif \(([^{}]*?)\) \{ return false; \}")
ALWAYS_ON = ("MoeDispatchStats",)
RESOLVE_INTRO = re.compile(r"\bauto\s+resolve\s*=\s*\[")
TOKEN_TRACKER = "g_moe_cputg_last_seq_layer"
# A user of the tracker both compares against it (the wrap test) and stores its own layer into it.
TRACKER_USES = (re.compile(r"<=\s*" + TOKEN_TRACKER + r"\b"), re.compile(r"\b" + TOKEN_TRACKER + r"\s*=(?!=)"))
RETIRED_TRACKER = re.compile(r"\bstatic\s+thread_local\s+int\s+last_seq_layer_cputg\b")
# N9: the record's all-host mode, decided from the plan in the builder, and the route's device half kept out of it.
RECORD_DEF = "struct moe_hybrid_decode_record {"
RECORD_BUILD = "ggml_sycl_moe_hybrid_decode_record_build"
ALL_HOST_FIELD = re.compile(r"\bbool\s+all_host\s*=\s*false\s*;")
ALL_HOST_DECIDED = re.compile(r"\bconst\s+bool\s+all_host\s*=\s*counts\.on_device\s*==\s*0\s*;")
ALL_HOST_RECORDED = re.compile(r"\brecord->all_host\s*=\s*all_host\s*;")
# A refusal returns the outcome itself or through the builder's `refuse("reason")` helper, which names the reason
# for the once-per-stamp log line. A retry counts as a refusal: one that fails moe_decode_direct_retry_limit times in
# a row for the same inputs settles as one (moe_decode_direct_stamp_settle), so a retry every all-host tensor takes
# refuses them all as surely.
REFUSAL_RETURN = r"return (?:ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_(?:REFUSED|RETRY)|refuse\([^;]*\));"
BUILD_REFUSAL = re.compile(r"\bif \(([^{}]*?)\) \{ " + REFUSAL_RETURN + r" \}")
REFUSAL_RETURNS = re.compile(REFUSAL_RETURN)
# The helper's own body, the one refusal with no condition.
REFUSAL_HELPER = re.compile(r"\{ \*reason = why; " + REFUSAL_RETURN)
# A refusal that is the whole final `else` of an if-chain: it fires when every link of the chain is false.
REFUSAL_ELSE = re.compile(r"\} else \{ " + REFUSAL_RETURN + r" \}")
# A link that tests a tensor is not all-host, anywhere in it: whatever else the link tests, the all-host tensors that
# fail it go on to the final else after it. Searched, not anchored, so `n_as < 0 || !all_host` and `all_host == false`
# count as surely as `!all_host`. Only the bare `counts.on_device` truth test stays anchored, since the name also
# appears in `counts.on_device == 0`, the all-host test itself.
NOT_ALL_HOST = re.compile(
    r"(?<!!)!\s*\(*\s*(?:record->)?all_host\b"
    r"|\b(?:record->)?all_host\s*(?:==\s*false|!=\s*true)\b"
    r"|\b(?:false\s*==|true\s*!=)\s*(?:record->)?all_host\b"
    r"|\bcounts\.on_device\s*(?:!=|>)\s*0\b"
    r"|\b0\s*(?:!=|<)\s*counts\.on_device\b"
    r"|^[(\s]*counts\.on_device[)\s]*$"
)
# A link true for every all-host tensor, so none of them reaches the final else after it.
ALL_HOST_LINK = re.compile(r"(^|\|\|)[(\s]*all_host[)\s]*(\|\||$)")
# What the builder learns only from a device expert: its materialized layout and the table's substitute pointer.
DEVICE_LOOKUP = re.compile(r"\b(materialized|substitute)\b")
# The all-host test itself, matched against one whole `||` operand of a refusal's condition: every all-host tensor
# that reaches the refusal satisfies that operand, whatever the others test.
ALL_HOST_ITSELF = re.compile(
    r"(?:record->)?all_host(?:\s*(?:==\s*true|!=\s*false))?"
    r"|(?:true\s*==|false\s*!=)\s*(?:record->)?all_host"
    r"|counts\.on_device\s*(?:==|<=)\s*0|0\s*(?:==|>=)\s*counts\.on_device|!\s*counts\.on_device"
)
# One `&&` operand of an `if` only a device expert enters. A refusal on `all_host` inside it fires only when the plan
# counted no device expert yet one resolved on the device; an all-host tensor whose plan holds never gets there.
DEVICE_EXPERT_ONLY = re.compile(r"(?:\w+\.)?executor\s*==\s*(?:\w+::)*PRIMARY_DEVICE")
DEVICE_HALF_GUARD = "if (!record->all_host) {"
DEVICE_HALF = ("mmvq_moe_batched_dispatch(", "record->table")
FLAGGED_DEF = "static void moe_host_scatter_submit_flagged("
# N7's own exception list: functions the route calls that the scan does not descend into, each with the reason.
# Kept apart from BOUNDARY so that a tree without the route fails N7, not N2.
ROUTE_BOUNDARY = {
    "ggml_sycl_moe_hybrid_decode_record_build": (
        "the route's per-tensor record: built once per (replan epoch, expert storage generation, selected rows) "
        "stamp, not per token.  It resolves every expert's route through ggml_sycl_build_moe_resolved_batch and "
        "allocates the record's host-pinned pointer table, so it reaches the resolvers' and the allocator's "
        "synchronous paths; a steady decode reuses the record and calls none of it"
    ),
}

ALLOWED = ("wait_pending_secondary_scatter_events(flush_pending_secondary_scatter());",)
# Each ALLOWED statement as a pattern that takes any whitespace between its tokens.
ALLOWED_PATTERNS = tuple(
    (a, re.compile(r"\s*".join(re.escape(tok) for tok in re.findall(r"\w+|[^\w\s]", a)))) for a in ALLOWED
)

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
    PROLOGUE: (
        r"\breturn\s*\[",
        ";",
        "the lambda it returns, which the pool worker runs before the job",
    ),
    JOIN: (
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

# Waits the scan finds inside the contract that stay, each keyed by (function, FORBIDDEN label) and
# excused at exactly one match.  An entry that matches nothing fails the gate, and a second match in the
# same function is reported.
KNOWN_WAITS = {
    ("allocate_managed_host_pinned", "synchronous mem_fill("): (
        "the zero_init fill, taken by the CPU dispatch's over-capacity managed output and by every Q1_0/NVFP4 "
        "recipe workspace; it predates llama.cpp-z4kd, and llama.cpp-yzwy moves the zeroing to the job prologue"
    ),
    ("moe_cpu_job_prologue", "wait()"): (
        "the refused input marker's fallback, `sycl::event::wait(inputs.events)`, taken only when a command graph "
        "records, the stream is not in-order or the marker word cannot be allocated or submitted.  The lead's "
        "decision on llama.cpp-42zj puts it here: a native wait on the pool worker can block on the driver's "
        "device lock behind a wait_and_throw on another thread, which deadlocked the first Qwen3.8 prompt graph"
    ),
}

DEF_SKIP = re.compile(r"(struct|class|namespace|enum|union|typedef|using|template\b|extern\s+\"|#)")
NOT_FUNCTIONS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else", "case", "alignof", "decltype",
    "static_assert", "defined",
}
CALL = re.compile(r"(?<![.\w])(?<!->)(\w+)\s*\(")

# A future-style get(): in moe_cpu_job_join, the call that joins a CPU job (N4); anywhere in the scanned
# code, a wait (N1).
GET_CALL = re.compile(r"(\.|->)\s*get\s*\(\s*\)")

FORBIDDEN = (
    (GET_CALL, "future-style .get()"),
    (re.compile(r"(\.|->|::)\s*wait\s*\("), "wait()"),
    (re.compile(r"\bwait_and_throw\b"), "wait_and_throw"),
    (re.compile(r"\bwait_for\b|\bwait_until\b"), "wait_for/wait_until"),
    (re.compile(r"\w_sync\s*\("), "a *_sync( call"),
    (re.compile(r"\bmem_copy\s*\("), "synchronous mem_copy("),
    (re.compile(r"\bmem_fill\s*\("), "synchronous mem_fill("),
    (re.compile(r"\bwait_\w+\s*\("), "a wait_* helper"),
    # The input marker's spin (cpu-job-signal.hpp, llama.cpp-42zj) is a host wait on device progress.  It is
    # defined in a header, so the scan sees it only at the call.
    (re.compile(r"\bcpu_job_input_marker_wait\s*\("), "the input marker's spin wait"),
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


def function_body(code: str, signature: str) -> tuple[int, str]:
    hits = [m.start() for m in re.finditer(re.escape(signature), code)]
    defs = []
    for at in hits:
        brace = code.find("{", at)
        semi = code.find(";", at)
        if brace >= 0 and (semi < 0 or brace < semi):
            defs.append((at, brace))
    if not defs:
        raise ContractError(f"FAIL: definition of `{signature}` not found in ggml-sycl.cpp ({TAG})")
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


def off_thread_closure(name: str, body: str) -> tuple[tuple[int, int] | None, str]:
    """The offsets in `body` of the `{` and `}` of the closure OFF_THREAD_SPANS names, and ""; or None and
    why the closure could not be found."""
    pattern, terminator, _ = OFF_THREAD_SPANS[name]
    found = [m.end() - 1 for m in re.finditer(pattern, body)]
    if len(found) != 1:
        return None, f"{name} has {len(found)} closures matching `{pattern}`; the gate needs exactly one"
    span = lambda_body(body, found[0])
    if span is None:
        return None, f"the closure in {name} has no body the gate can find"
    after = body[span[1] + 1 :].lstrip()[:1]
    if after != terminator:
        return None, (
            f"the closure in {name} is followed by `{after}`, not `{terminator}`, so it is not the whole "
            f"handed-off expression"
        )
    return span, ""


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
    boundary: set[str] | None = None,
) -> tuple[list[tuple[str, int, str]], set[str], set[str]]:
    """The regions plus every in-file function they call, transitively, short of BOUNDARY, and the
    enclosing function's lambdas the hybrid block calls (and those lambdas call).

    Returns the scanned (label, offset, text) list, each callee labelled with the call chain that reached
    it, the BOUNDARY names some scanned code calls, and the names of everything scanned.
    """
    boundary = set(BOUNDARY) if boundary is None else boundary
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
                if callee in boundary:
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


class Scan:
    """The blanked source, the code the scan reads in it and the waits found there, which the rules check."""

    def __init__(self, backend_src: str):
        code = blank_comments_and_strings(backend_src)
        self.code = code
        self.block_at, self.block = hybrid_block(code)
        self.regions: list[tuple[str, int, str]] = [(HYBRID_LABEL, self.block_at, self.block)]
        for sig in HELPERS:
            at, body = function_body(code, sig)
            self.regions.append((sig.replace("static ", "").split("(")[0].split()[-1], at, body))
        self.defs = file_functions(code)
        # The event-chained hybrid decode route (llama.cpp-7pvo) runs on the submitting thread like the
        # block; when it is defined once it is a region of its own.  N7 fails when it is not.
        route_defs = self.defs.get(ROUTE, [])
        if len(route_defs) == 1:
            self.regions.append((ROUTE, route_defs[0][0], route_defs[0][1]))
        # A function that is missing, or whose closure cannot be found, is scanned whole (stricter) and
        # reported under N2, after N1, so a tree without the event chain still shows its waits.
        self.closures: dict[str, tuple[int, int]] = {}
        self.off_thread_problems: list[tuple[str, str]] = []
        for name in OFF_THREAD_SPANS:
            bodies = self.defs.get(name, [])
            if len(bodies) != 1:
                self.off_thread_problems.append((name, f"{name} is defined {len(bodies)} times in ggml-sycl.cpp"))
                continue
            brace, body = bodies[0]
            span, problem = off_thread_closure(name, body)
            if span is None:
                self.off_thread_problems.append((name, problem))
                continue
            open_at, close_at = span
            self.closures[name] = (brace + open_at, brace + close_at)
            inner = "".join("\n" if c == "\n" else " " for c in body[open_at + 1 : close_at])
            self.defs[name] = [(brace, body[: open_at + 1] + inner + body[close_at:])]
        self.scanned, self.reached, self.followed = scanned_code(
            self.regions, self.defs, enclosing_lambdas(code, self.block_at), set(BOUNDARY) | set(ROUTE_BOUNDARY)
        )
        self.waits: list[str] = []
        self.allowed_seen = {a: 0 for a in ALLOWED}
        self.known_seen = {k: 0 for k in KNOWN_WAITS}
        for name, base, text in self.scanned:
            masked = text
            for a, pat in ALLOWED_PATTERNS:
                for m in pat.finditer(masked):
                    self.allowed_seen[a] += 1
                    masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]
            for pat, what in FORBIDDEN:
                for m in pat.finditer(masked):
                    key = (name.split(" ")[0], what)
                    if key in KNOWN_WAITS and self.known_seen[key] == 0:
                        self.known_seen[key] += 1
                        continue
                    self.waits.append(f"{name} line {line_of(code, base) + text.count(chr(10), 0, m.start())}: {what}")

    def closure_text(self, name: str) -> str:
        """The body of the closure `name` hands off, which N2 has already required the scan to find."""
        open_at, close_at = self.closures[name]
        return self.code[open_at + 1 : close_at]


def check_n1(scan: Scan) -> None:
    """No host wait in the scanned code, apart from the allowlisted statement (at most once) and KNOWN_WAITS."""
    if scan.waits:
        raise ContractError(
            f"FAIL [N1]: {len(scan.waits)} host wait(s) on the submitting thread after the ids readback "
            f"({TAG}):\n  " + "\n  ".join(scan.waits)
        )
    for a, n in scan.allowed_seen.items():
        if n > 1:
            raise ContractError(f"FAIL [N1]: the allowlisted `{a}` appears {n} times; it is allowed once ({TAG})")


def check_n2(scan: Scan) -> None:
    """The exception lists still describe the code: each KNOWN_WAITS entry matches, each OFF_THREAD_SPANS
    function is defined once, hands off one whole closure and is reached, and each BOUNDARY entry is defined
    and reached."""
    for (fn, what), n in scan.known_seen.items():
        if n == 0:
            raise ContractError(
                f"FAIL [N2]: KNOWN_WAITS entry `{fn}` / {what} matches nothing in the scanned code; drop it ({TAG})"
            )
    if scan.off_thread_problems:
        name, problem = scan.off_thread_problems[0]
        raise ContractError(
            f"FAIL [N2]: {problem}, so the scan cannot set apart {OFF_THREAD_SPANS[name][2]} ({TAG})"
        )
    for name in OFF_THREAD_SPANS:
        if name not in scan.followed:
            raise ContractError(
                f"FAIL [N2]: OFF_THREAD_SPANS entry `{name}` is no longer called from the scanned code; drop it ({TAG})"
            )
    for name in BOUNDARY:
        if name not in scan.defs:
            raise ContractError(
                f"FAIL [N2]: BOUNDARY entry `{name}` is not defined in ggml-sycl.cpp; drop it or rename it ({TAG})"
            )
        if name not in scan.reached:
            raise ContractError(
                f"FAIL [N2]: BOUNDARY entry `{name}` is no longer called from the scanned code; drop it ({TAG})"
            )


def check_n3(scan: Scan) -> None:
    """The closure that moe_cpu_job_prologue returns waits for the job's inputs, on the pool worker."""
    if WAIT_FN not in scan.closure_text(PROLOGUE):
        raise ContractError(
            f"FAIL [N3]: the closure that {PROLOGUE} returns no longer calls {WAIT_FN.rstrip('(')}, so a CPU job "
            f"could start before the copies it reads have landed ({TAG})"
        )


def check_n4(scan: Scan) -> None:
    """moe_cpu_job_join takes the job's get(), the closure that moe_cpu_job_join_on_queue hands host_task
    calls it, and no other code in the file does.  A call from the scanned code is already an N1 wait,
    since the scan follows it into the get(); this finds a caller outside the scanned code."""
    code = scan.code
    _, joiner = function_body(code, JOINER_DEF)
    if not GET_CALL.search(joiner):
        raise ContractError(f"FAIL [N4]: moe_cpu_job_join no longer takes the CPU job's get() ({TAG})")
    if not re.search(r"\b" + re.escape(JOINER_FN), scan.closure_text(JOIN)):
        raise ContractError(
            f"FAIL [N4]: the closure that {JOIN} hands host_task no longer calls moe_cpu_job_join, so its event "
            f"completes before the CPU job does and the scatter could read the output first ({TAG})"
        )
    open_at, close_at = scan.closures[JOIN]
    stray = []
    for m in re.finditer(r"\b" + re.escape(JOINER_FN), code):
        if code.startswith(JOINER_DEF, m.start() - len("static void ")):
            continue  # the definition
        if not open_at < m.start() < close_at:
            stray.append(line_of(code, m.start()))
    if stray:
        raise ContractError(
            f"FAIL [N4]: moe_cpu_job_join is called outside the join host_task (line(s) {stray}); the job would "
            f"be joined on the submitting thread ({TAG})"
        )


def check_n5(scan: Scan) -> None:
    """The scatter H2D depends on the join events, in every joining flush and in the compact form."""
    code = scan.code
    flushes = [(n, t) for n, _, t in scan.regions if n in JOINING_FLUSHES]
    for fname, flush_text in flushes:
        dep_vars = set()
        for m in re.finditer(r"(?:const\s+)?sycl::event\s+(\w+)\s*=\s*" + re.escape(JOIN) + r"\s*\(", flush_text):
            dep_vars.add(m.group(1))
        for m in re.finditer(r"(\w+)\s*\.\s*push_back\s*\(\s*" + re.escape(JOIN) + r"\s*\(", flush_text):
            dep_vars.add(m.group(1))
        if not dep_vars:
            raise ContractError(f"FAIL [N5]: {fname} does not keep the join events ({TAG})")
        for ev in sorted(dep_vars):
            for m in re.finditer(r"std::vector<sycl::event>\s+(\w+)\s*\{\s*" + re.escape(ev) + r"\s*\}", flush_text):
                dep_vars.add(m.group(1))
        calls = [
            m.start()
            for m in re.finditer(
                r"\bmem_copy_async\s*\(|\b" + re.escape(COMPACT_FN) + r"|\b" + re.escape(FLAGGED_FN), flush_text
            )
        ]
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
    # The compact form always; the flagged form (llama.cpp-7pvo) once it exists, and N7 requires it to.
    forms = [COMPACT_DEF]
    if FLAGGED_DEF in code:
        forms.append(FLAGGED_DEF)
    for form_def in forms:
        form = form_def.split("(")[0].split()[-1]
        form_at, form_body = function_body(code, form_def)
        params = code[form_at : code.find("{", form_at)]
        deps = re.search(r"const\s+std::vector<sycl::event>\s*&\s*(\w+)", params)
        if not deps:
            raise ContractError(f"FAIL [N5]: {form} takes no dependency list for its copies ({TAG})")
        copies = [m.start() for m in re.finditer(r"\bmem_copy_async\s*\(", form_body)]
        if not copies:
            raise ContractError(f"FAIL [N5]: {form} submits no copy; the gate is blind ({TAG})")
        for c in copies:
            args = form_body[c : matching(form_body, form_body.find("(", c), "(", ")")]
            if not re.search(r"\b" + re.escape(deps.group(1)) + r"\b", args):
                raise ContractError(
                    f"FAIL [N5]: a copy in {form} does not depend on `{deps.group(1)}`, the joins it is handed; it "
                    f"could read the output before the CPU job wrote it ({TAG})"
                )


def check_n6(scan: Scan) -> None:
    """The graph-boundary wait for the last scatters is called only from the graph-boundary flush, which still
    calls it.  A call from the scanned code is already an N1 wait; this finds a caller outside it."""
    code = scan.code
    calls = [m.start() for m in re.finditer(r"\b" + re.escape(BOUNDARY_WAIT), code)]
    _, boundary = function_body(code, BOUNDARY_FN)
    boundary_at = code.find(BOUNDARY_FN)
    # The drain's own definitions (the per-slot overload and the all-slots drain that calls it).
    drain_bodies = []
    drain_def = r"^static\s+void\s+" + re.escape(BOUNDARY_WAIT.rstrip("(")) + r"\s*\([^)]*\)\s*\{"
    for m in re.finditer(drain_def, code, re.M):
        drain_bodies.append((m.start(), matching(code, m.end() - 1)))
    outside = []
    for c in calls:
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


def squash(text: str) -> str:
    return " ".join(text.split())


def check_n7(scan: Scan) -> None:
    """A decode op whose experts split between the device and the host does not read its expert ids on the
    submitting thread (llama.cpp-7pvo): the event-chained route runs ahead of the B1 readback and returns when
    it takes the op, copies the ids asynchronously into the job's inputs, and its scatter goes through the
    flagged form.  N1 already covers the route's body and callees, since it is a scanned region."""
    code = scan.code
    route_defs = scan.defs.get(ROUTE, [])
    if len(route_defs) != 1:
        raise ContractError(
            f"FAIL [N7]: `{ROUTE}` is defined {len(route_defs)} times in ggml-sycl.cpp, so every decode op with "
            f"host experts reads its expert ids back on the submitting thread (the B1 wait, ~0.36 ms per MoE "
            f"layer on Qwen3.8) ({TAG_7PVO})"
        )
    _, fn = function_body(code, ENCLOSING_DEF)
    flat = squash(fn)
    calls = [m.start() for m in ROUTE_CALL.finditer(flat)]
    b1 = [m.start() for m in B1_TIMER.finditer(flat)]
    if len(calls) != 1:
        raise ContractError(
            f"FAIL [N7]: ggml_sycl_mul_mat_id has {len(calls)} `if ({ROUTE}(ctx, src0, src1, dst)) {{ return; }}` "
            f"statements; it needs exactly one, so an op the route takes ends there ({TAG_7PVO})"
        )
    if len(b1) != 1 or calls[0] > b1[0]:
        raise ContractError(
            f"FAIL [N7]: the route call is not ahead of the B1 expert-id readback in ggml_sycl_mul_mat_id "
            f"({len(b1)} B1 timer(s)), so an op it takes would still read its ids back first ({TAG_7PVO})"
        )
    _, body = route_defs[0]
    ids_events = IDS_COPY.findall(body)
    if len(ids_events) != 1:
        raise ContractError(
            f"FAIL [N7]: {ROUTE} has {len(ids_events)} `sycl::event x = ggml_sycl::mem_copy_async(pool.ids_handle(), "
            f"...)` copies; it needs one, the asynchronous ids copy into the pool's ids slots ({TAG_7PVO})"
        )
    if not re.search(r"\bjob_inputs\.events\.push_back\(\s*" + re.escape(ids_events[0]) + r"\s*\)", body):
        raise ContractError(
            f"FAIL [N7]: the ids copy's event `{ids_events[0]}` is not one of the CPU job's inputs, so the job could "
            f"read the ids before they land ({TAG_7PVO})"
        )
    for name in ROUTE_BOUNDARY:
        if name not in scan.defs:
            raise ContractError(
                f"FAIL [N7]: ROUTE_BOUNDARY entry `{name}` is not defined in ggml-sycl.cpp; drop it or rename it "
                f"({TAG_7PVO})"
            )
        if name not in scan.reached:
            raise ContractError(
                f"FAIL [N7]: ROUTE_BOUNDARY entry `{name}` is no longer called from the scanned code; drop it "
                f"({TAG_7PVO})"
            )
    flush_text = next(t for n, _, t in scan.regions if n == "flush_pending_cpu_scatter_slot")
    if FLAGGED_DEF not in code or not re.search(r"\b" + re.escape(FLAGGED_FN), flush_text):
        raise ContractError(
            f"FAIL [N7]: flush_pending_cpu_scatter_slot does not place flagged rows through "
            f"{FLAGGED_FN.rstrip('(')}, so the rows the route's CPU job selects are never scattered ({TAG_7PVO})"
        )


def check_n8(scan: Scan) -> None:
    """The route takes a split-placement decode op in the default configuration (llama.cpp-7pvo): no refusal
    names a mode that is always on, and the route keeps the retained CPU-TG block's per-op observations rather
    than refusing to run without them.  N7 has already required the route to be defined once."""
    _, body = scan.defs[ROUTE][0]
    for m in ROUTE_REFUSAL.finditer(squash(body)):
        for name in ALWAYS_ON:
            if name in m.group(1):
                raise ContractError(
                    f"FAIL [N8]: {ROUTE} refuses an op when `{name}` is on, and it is always on, so the route "
                    f"never takes an op and every host-expert decode op still reads its ids back ({TAG_7PVO})"
                )
    intro = RESOLVE_INTRO.search(body)
    span = lambda_body(body, body.find("[", intro.start())) if intro else None
    if not span:
        raise ContractError(
            f"FAIL [N8]: {ROUTE} has no `auto resolve = [...](...) {{...}}` step for its CPU job, so nothing on "
            f"the pool worker records what the job saw in the ids ({TAG_7PVO})"
        )
    resolve = body[span[0] : span[1]]
    actual_at = resolve.find("record_actual(")
    slots_at = resolve.find("moe_host_slots_from_ids(")
    if "record_route_residency(" not in resolve or actual_at < 0 or slots_at < 0 or actual_at > slots_at:
        raise ContractError(
            f"FAIL [N8]: {ROUTE}'s resolve step does not record the dispatch stats' residency and, before "
            f"moe_host_slots_from_ids turns the ids into flags, the predictor's actual experts; the retained "
            f"CPU-TG block records both for every op it takes ({TAG_7PVO})"
        )
    outside = body[: span[0]] + body[span[1] :]
    _, fn = function_body(scan.code, ENCLOSING_DEF)
    if (not re.search(r"\btick_token\s*\(\s*\)", outside) or RETIRED_TRACKER.search(scan.code)
            or not all(use.search(outside) and use.search(fn) for use in TRACKER_USES)):
        raise ContractError(
            f"FAIL [N8]: {ROUTE} does not tick the dispatch stats' token on the submitting thread through "
            f"`{TOKEN_TRACKER}`, or the retained CPU-TG block keeps a tracker of its own, so a token whose "
            f"host-expert layers split between the two routes ticks twice or not at all ({TAG_7PVO})"
        )


def matching_back(src: str, close_at: int, open_ch: str = "{", close_ch: str = "}") -> int:
    """The opening bracket that the one at `close_at` closes, scanning backwards."""
    depth = 0
    for k in range(close_at, -1, -1):
        c = src[k]
        if c == close_ch:
            depth += 1
        elif c == open_ch:
            depth -= 1
            if depth == 0:
                return k
    raise ContractError(f"FAIL: unbalanced {open_ch}{close_ch} block ({TAG})")


def if_chain_links(text: str, close_at: int) -> list[str] | None:
    """The conditions of the if-chain whose last block closes at `close_at` in squashed code, last link first, or
    None when the code before it is not a plain `if (...) {...} else if (...) {...}` chain."""
    links: list[str] = []
    while True:
        head = text[: matching_back(text, close_at)].rstrip()
        if not head.endswith(")"):
            return None
        cond_open = matching_back(text, len(head) - 1, "(", ")")
        links.append(text[cond_open + 1 : len(head) - 1].strip())
        lead = text[:cond_open].rstrip()
        chained = re.search(r"\}\s*else\s+if$", lead)
        if chained:
            close_at = chained.start()
        elif re.search(r"(^|[^\w])if$", lead):
            return links
        else:
            return None


def split_top_level(cond: str, op: str) -> list[str]:
    """`cond`'s operands of its top-level `op` (`||` or `&&`), each without enclosing parentheses; an operand that is
    itself a parenthesized `op` expression is expanded in place."""
    cond = cond.strip()
    while cond.startswith("(") and matching(cond, 0, "(", ")") == len(cond) - 1:
        cond = cond[1:-1].strip()
    parts: list[str] = []
    depth = start = k = 0
    while k < len(cond):
        c = cond[k]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and cond.startswith(op, k):
            parts.append(cond[start:k])
            k += len(op)
            start = k
            continue
        k += 1
    if not parts:
        return [cond]
    parts.append(cond[start:])
    out: list[str] = []
    for part in parts:
        out.extend(split_top_level(part, op))
    return out


def all_host_operand(cond: str) -> str | None:
    """The `||` operand of `cond` that is the all-host test itself, or None."""
    for operand in split_top_level(cond, "||"):
        if ALL_HOST_ITSELF.fullmatch(operand):
            return operand
    return None


def enclosing_if_conditions(text: str, at: int) -> list[str]:
    """The conditions of the `if (...) {...}` blocks that enclose position `at` in squashed code, innermost first."""
    conds: list[str] = []
    depth = 0
    for k in range(at - 1, -1, -1):
        c = text[k]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth:
                depth -= 1
                continue
            head = text[:k].rstrip()
            if head.endswith(")"):
                cond_open = matching_back(text, len(head) - 1, "(", ")")
                if re.search(r"(^|[^\w])if$", text[:cond_open].rstrip()):
                    conds.append(text[cond_open + 1 : len(head) - 1].strip())
    return conds


def device_expert_only(text: str, at: int) -> bool:
    """Whether position `at` sits in an `if` whose condition requires the expert's executor to be this device."""
    for cond in enclosing_if_conditions(text, at):
        if any(DEVICE_EXPERT_ONLY.fullmatch(part) for part in split_top_level(cond, "&&")):
            return True
    return False


def check_n9(scan: Scan) -> None:
    """The route admits a tensor whose experts all live on the host (llama.cpp-7pvo): the record has an all-host
    mode the builder decides from the plan, no builder refusal needs a device expert outside that mode, and the
    route keeps its table copy and device kernel out of it.  N7 has already required the route to be defined once."""
    code = scan.code
    at = code.find(RECORD_DEF)
    record = code[at : matching(code, at + len(RECORD_DEF) - 1) + 1] if at >= 0 else ""
    builds = scan.defs.get(RECORD_BUILD, [])
    if not ALL_HOST_FIELD.search(record) or len(builds) != 1:
        raise ContractError(
            f"FAIL [N9]: moe_hybrid_decode_record has no `bool all_host = false;` mode, or `{RECORD_BUILD}` is "
            f"not defined once, so a tensor with no device expert gets no record and every all-host layer still "
            f"reads its expert ids back ({TAG_7PVO})"
        )
    _, build = builds[0]
    if not ALL_HOST_DECIDED.search(build) or not ALL_HOST_RECORDED.search(build):
        raise ContractError(
            f"FAIL [N9]: {RECORD_BUILD} does not decide `const bool all_host = counts.on_device == 0;` from the "
            f"plan and store it with `record->all_host = all_host;`, so the route cannot tell an all-host record "
            f"from a split one ({TAG_7PVO})"
        )
    squashed = squash(build)
    elses = list(REFUSAL_ELSE.finditer(squashed))
    read = len(BUILD_REFUSAL.findall(squashed)) + len(REFUSAL_HELPER.findall(squashed)) + len(elses)
    if read != len(REFUSAL_RETURNS.findall(squashed)):
        raise ContractError(
            f"FAIL [N9]: {RECORD_BUILD} has a refusal or retry outside an `if (...) {{...}}` this rule reads, "
            f"the `refuse` helper's body or a final `else`, so its condition goes unchecked and could refuse "
            f"every all-host tensor unseen ({TAG_7PVO})"
        )
    for m in BUILD_REFUSAL.finditer(squashed):
        cond = m.group(1)
        if DEVICE_LOOKUP.search(cond) and "!all_host" not in cond.replace(" ", ""):
            raise ContractError(
                f"FAIL [N9]: {RECORD_BUILD} refuses or retries on `{cond}`, which only a device expert can "
                f"satisfy, without `!all_host`, so every tensor whose experts all live on the host is refused "
                f"({TAG_7PVO})"
            )
        operand = all_host_operand(cond)
        if operand is not None and not device_expert_only(squashed, m.start()):
            raise ContractError(
                f"FAIL [N9]: {RECORD_BUILD} refuses or retries on `{cond}`, whose `||` operand `{operand}` is the "
                f"all-host case itself, outside any `if` only a device expert enters, so the layers the placement "
                f"leaves entirely on the host keep their readback ({TAG_7PVO})"
            )
    for m in elses:
        links = if_chain_links(squashed, m.start())
        if links is None:
            raise ContractError(
                f"FAIL [N9]: {RECORD_BUILD} refuses or retries in a final `else` that closes no plain "
                f"`if (...) {{...}} else if (...) {{...}}` chain, so the rule cannot read when it fires ({TAG_7PVO})"
            )
        for link in links:
            if NOT_ALL_HOST.search(link):
                raise ContractError(
                    f"FAIL [N9]: {RECORD_BUILD} refuses or retries in the final `else` after `if ({link})`, a "
                    f"link that tests a tensor is not all-host, so every all-host tensor that fails it is refused "
                    f"there: the all-host case itself ({TAG_7PVO})"
                )
        lookups = [link for link in links if DEVICE_LOOKUP.search(link)]
        if lookups and not any(ALL_HOST_LINK.search(link) for link in links):
            raise ContractError(
                f"FAIL [N9]: {RECORD_BUILD} refuses or retries in the final `else` of a chain that tests "
                f"`{lookups[0]}`, which only a device expert can satisfy, with no `all_host || ...` link to keep "
                f"an all-host tensor out of it ({TAG_7PVO})"
            )
    _, route = scan.defs[ROUTE][0]
    guard = route.find(DEVICE_HALF_GUARD)
    if guard < 0:
        raise ContractError(
            f"FAIL [N9]: {ROUTE} has no `{DEVICE_HALF_GUARD}...}}` block, so it would copy a table an all-host "
            f"record does not have and run the device kernel for experts that are not on the device ({TAG_7PVO})"
        )
    close = matching(route, guard + len(DEVICE_HALF_GUARD) - 1)
    inside = route[guard:close]
    outside = route[:guard] + route[close:]
    for token in DEVICE_HALF:
        if token not in inside or token in outside:
            raise ContractError(
                f"FAIL [N9]: `{token}` in {ROUTE} is not confined to its `{DEVICE_HALF_GUARD}...}}` block, so an "
                f"all-host record would reach the device half ({TAG_7PVO})"
            )
    for m in ROUTE_REFUSAL.finditer(squash(route)):
        if re.search(r"\ball_host\b", m.group(1)):
            raise ContractError(
                f"FAIL [N9]: {ROUTE} refuses on `{m.group(1)}`, so an all-host record is built and then never "
                f"taken ({TAG_7PVO})"
            )


def check(backend_src: str) -> None:
    scan = Scan(backend_src)
    for rule in (check_n1, check_n2, check_n3, check_n4, check_n5, check_n6, check_n7, check_n8, check_n9):
        rule(scan)


def mutants(src: str) -> list[tuple[str, str, str]]:
    """(name, mutated source, the rule it must fail on): "setup" for one that removes what the gate locates."""
    out: list[tuple[str, str, str]] = []

    def sub(rule: str, name: str, old: str, new: str, count: int = 1) -> None:
        if old not in src:
            raise ContractError(f"FAIL: self-test anchor for mutant `{name}` not found: {old!r} ({TAG})")
        out.append((name, src.replace(old, new, count), rule))

    # The hybrid block's own graphs-off statement: the same line appears at many sites earlier in the
    # file, so the anchor runs from the hybrid marker through it, and a mutant lands inside the block.
    h_at = src.find(HYBRID_MARKER + "\n")
    g_line = "            ctx.moe_graphs_disabled_once = true;\n"
    g_at = src.find(g_line, h_at) if h_at >= 0 else -1
    graphs_off = src[h_at : g_at + len(g_line)] if g_at >= 0 else "<hybrid block graphs-off anchor missing>"
    sub("N1", "activation wait on the submitting thread", graphs_off,
        graphs_off + "            act_d2h_event.wait();\n")
    sub("N1", "queue drain in the hybrid block", graphs_off, graphs_off + "            stream->wait();\n")
    sub("N1", "wait_and_throw in the hybrid block", graphs_off, graphs_off + "            stream->wait_and_throw();\n")
    sub("N1", "event-list wait in the hybrid block", graphs_off,
        graphs_off + "            sycl::event::wait(std::vector<sycl::event>{});\n")
    sub("N1", "synchronous copy in the hybrid block", graphs_off,
        graphs_off + "            ggml_sycl::mem_copy(shared_act_handle, shared_act_handle, 0, *stream);\n")
    sub("N1", "worker wait called directly", graphs_off,
        graphs_off + "            moe_cpu_job_wait_inputs(ggml_sycl::cpu_job_input_marker{}, nullptr, 0);\n")
    sub("N1", "second allowlisted secondary flush", graphs_off,
        graphs_off + "            wait_pending_secondary_scatter_events(flush_pending_secondary_scatter());\n")
    slot_open = "static void flush_pending_cpu_scatter_slot(pending_cpu_scatter & slot) {\n"
    sub("N1", "future.get in the slot flush", slot_open, slot_open + "    slot.future.get();\n")
    sub("N1", "previous-scatter wait in the slot flush", slot_open,
        slot_open + "    sycl::event::wait(slot.prev_scatter.scatter_events);\n")
    sub("N2", "slot flush stops joining through the queue", "moe_cpu_job_join_on_queue(*slot.stream",
        "moe_cpu_job_join_on_queue_unused(*slot.stream")
    sub("N5", "scatter H2D loses the join dependency", "run.src_offset, run.bytes, *slot.stream, joins));",
        "run.src_offset, run.bytes, *slot.stream, {}));")
    sub("N5", "compact scatter handed no joins", "moe_host_scatter_submit_compact(slot, ws, joins, &n_copies, &why)",
        "moe_host_scatter_submit_compact(slot, ws, {}, &n_copies, &why)")
    sub("N5", "compact copies drop their dependency", "copy.src_offset, copy.bytes, queue, deps));",
        "copy.src_offset, copy.bytes, queue));")
    join_def = (
        "static sycl::event moe_cpu_job_join_on_queue(sycl::queue &                                     queue,\n"
        "                                             std::shared_future<void>                          job,\n"
        "                                             std::shared_ptr<ggml_sycl::cpu_expert_job_stamps> stamps) {\n"
    )
    pro_def = ("static std::function<void()> moe_cpu_job_prologue(sycl::queue &       stream,\n"
               "                                                  moe_cpu_job_inputs  inputs,\n"
               "                                                  const ggml_tensor * src0) {\n")
    pro_closure = ("    return [marker = std::move(marker), zero_out = inputs.zero_out, zero_bytes = inputs.zero_bytes]() {\n"
                   "        moe_cpu_job_wait_inputs(marker, zero_out, zero_bytes);\n    };\n")
    pro_fallback = "            sycl::event::wait(inputs.events);\n"
    sub("N1", "join get() outside its host_task", join_def, join_def + "    job.get();\n")
    sub("N1", "job join called outside its host_task", join_def, join_def + "    moe_cpu_job_join(job, nullptr);\n")
    sub("N1", "job join called from the slot flush", slot_open,
        slot_open + "    moe_cpu_job_join(slot.future.share(), nullptr);\n")
    sub("N4", "join helper stops taking get()",
        "ggml_sycl::cpu_expert_pool_trace_note_join(was_ready);\n    }\n    job.get();\n",
        "ggml_sycl::cpu_expert_pool_trace_note_join(was_ready);\n    }\n")
    sub("N1", "prologue waits in the factory body", pro_def,
        pro_def + "    moe_cpu_job_wait_inputs(ggml_sycl::cpu_job_input_marker{}, inputs.zero_out, inputs.zero_bytes);\n")
    # The input marker (llama.cpp-42zj): the submitting thread spinning on it, a second native wait on the
    # armed path, and the excused fallback gone (a refused job would then wait nothing; the input-wait pin
    # W4 catches that too).
    sub("N1", "prologue spins on the input marker itself", pro_closure,
        "    ggml_sycl::cpu_job_input_marker_wait(marker);\n" + pro_closure)
    sub("N1", "prologue waits its inputs on the armed path too", pro_closure,
        "    sycl::event::wait(inputs.events);\n" + pro_closure)
    sub("N2", "refused arm's fallback wait dropped", pro_fallback, "")
    sub("N1", "boundary wait called from a flush", "        ++g_cpu_scatter_serial;\n",
        "        ++g_cpu_scatter_serial;\n        pending_cpu_scatter_wait_prev_events(slot);\n")
    sub("N6", "boundary wait dropped from the graph-boundary flush",
        "    pending_cpu_scatter_wait_prev_events();  // The graph-boundary wait", "    // The graph-boundary wait")
    # A wait moved into a helper is still a wait.  Each helper is new, defined at file scope just before
    # the flush, and called from scanned code; none matches the wait_* helper pattern by name.
    sub("N1", "future.wait in a helper the slot flush calls", slot_open,
        "static void pending_cpu_scatter_join(pending_cpu_scatter & slot) {\n    slot.future.wait();\n}\n\n"
        + slot_open + "    pending_cpu_scatter_join(slot);\n")
    sub("N1", "prev-scatter wait in a helper the slot flush calls", slot_open,
        "static void drain_prev_scatter(pending_cpu_scatter & slot) {\n"
        "    sycl::event::wait(slot.prev_scatter.scatter_events);\n}\n\n"
        + slot_open + "    drain_prev_scatter(slot);\n")
    sub("N1", "activation wait in a helper the hybrid block calls", slot_open,
        "static void moe_settle_activation(sycl::event evt) {\n    evt.wait();\n}\n\n" + slot_open)
    # The call for the helper above, placed in the hybrid block; built on the first mutant's text so both
    # edits land together.
    settle = out[-1][1]
    s_h = settle.find(HYBRID_MARKER + "\n")
    s_g = settle.find(g_line, s_h)
    out[-1] = (out[-1][0], settle[: s_g + len(g_line)] + "            moe_settle_activation(sycl::event{});\n"
               + settle[s_g + len(g_line) :], out[-1][2])
    # The join host_task and the slot flush's join push, as the backend writes them.
    join_task = "cgh.host_task([job, stamps = std::move(stamps)]() { moe_cpu_job_join(job, stamps.get()); });"
    slot_join = ("            joins.push_back(moe_cpu_job_join_on_queue(*slot.stream, group[k]->future.share(), "
                 "group[k]->job_stamps));\n")
    # Positive control for the helper mutants: the same join wait written inline in the flush.
    sub("N1", "group future wait inline in the slot flush", slot_join,
        "            group[k]->future.wait();\n" + slot_join)
    # Two calls deep: flush_pending_cpu_scatter_if_consumed -> ggml_sycl_moe_precomputed_mmid_skip_pending ->
    # ggml_sycl_moe_precomputed_skip_contains.
    deep = ("    moe_precomputed_skip_key key{};\n"
            "    return ggml_sycl_make_moe_precomputed_skip_key(tensor, device, &key)")
    sub("N1", "wait two calls below a scanned helper", deep, "    g_pending_scatter.future.wait();\n" + deep)
    sub("N2", "BOUNDARY entry no longer reached", "moe_prestage_popular_experts();",
        "moe_prestage_popular_experts_retired();", count=-1)
    sub("N2", "BOUNDARY entry no longer defined", "static void moe_prestage_popular_experts() {",
        "static void moe_prestage_popular_experts_v2() {")
    # moe_periodic_rerank left BOUNDARY: the scan follows it, so a wait in its own body is caught.
    sub("N1", "wait in the rerank the scan follows", "    int ranks_changed = 0;\n",
        "    g_pending_scatter.future.wait();\n    int ranks_changed = 0;\n")
    fill = "    if (zero_init) {\n        ggml_sycl::mem_fill(handle, 0, bytes, q);\n    }\n"
    sub("N2", "KNOWN_WAITS entry matches nothing", fill, "")
    sub("N1", "second fill beside the known one", "    *out = std::move(handle);\n    return true;\n}\n",
        "    ggml_sycl::mem_fill(handle, 0, bytes, q);\n    *out = std::move(handle);\n    return true;\n}\n")
    # The prologue and the join run on the submitting thread apart from the closure each hands off; the
    # rest of each body, its closures' capture lists and its callees are scanned.  The unmodified tree
    # passing is the other half: both closures hold a wait (moe_cpu_job_wait_inputs, moe_cpu_job_join's
    # get()) that the scan would report if it read them.
    sub("N1", "wait in a helper the join calls before its submit", join_def,
        "static void moe_join_settle(const std::shared_future<void> & j) {\n    j.wait();\n}\n\n"
        + join_def + "    moe_join_settle(job);\n")
    sub("N1", "event wait in a helper the prologue factory calls", pro_def,
        "static void moe_settle_inputs(const moe_cpu_job_inputs & in) {\n    sycl::event::wait(in.events);\n}\n\n"
        + pro_def + "    moe_settle_inputs(inputs);\n")
    sub("N1", "worker wait behind a wrapper the prologue factory calls", pro_def,
        "static void moe_inputs_ready(const moe_cpu_job_inputs & in) {\n"
        "    moe_cpu_job_wait_inputs(ggml_sycl::cpu_job_input_marker{}, in.zero_out, in.zero_bytes);\n}\n\n"
        + pro_def + "    moe_inputs_ready(inputs);\n")
    sub("N1", "wait in the host_task closure's capture list", join_task,
        "cgh.host_task([job = (job.wait(), job), stamps = std::move(stamps)]() "
        "{ moe_cpu_job_join(job, stamps.get()); });")
    sub("N1", "join hands host_task a named closure", join_task,
        "auto task = [job, stamps = std::move(stamps)]() { moe_cpu_job_join(job, stamps.get()); }; "
        "cgh.host_task(task);")
    sub("N2", "prologue definition renamed away", "static std::function<void()> moe_cpu_job_prologue(",
        "static std::function<void()> moe_cpu_job_prologue_v2(")
    # Every caller: the hybrid block and the event-chained route (llama.cpp-7pvo) each build one.
    sub("N2", "CPU dispatch stops building its jobs' prologue", "moe_cpu_job_prologue(*stream, std::move(job_inputs), src0)",
        "moe_cpu_job_prologue_unused(*stream, std::move(job_inputs), src0)", count=-1)
    # A lambda called on the submitting thread to build the handed-off closure runs there: only its result
    # is handed off.
    sub("N1", "join builds its host_task closure by calling a lambda", join_task,
        "cgh.host_task([job, stamps]() { job.wait(); return [job, stamps]() "
        "{ moe_cpu_job_join(job, stamps.get()); }; }());")
    sub("N1", "prologue builds its closure by calling a lambda", pro_closure,
        "    return [&]() { sycl::event::wait(inputs.events); return std::function<void()>([marker, "
        "zero_out = inputs.zero_out, zero_bytes = inputs.zero_bytes]() "
        "{ moe_cpu_job_wait_inputs(marker, zero_out, zero_bytes); }); }();\n")
    # A lambda of ggml_sycl_mul_mat_id that the hybrid block calls runs on the submitting thread too.
    expert_id_line = "        local_extra.moe_expert_id     = expert_id;\n"
    sub("N1", "wait in a lambda of ggml_sycl_mul_mat_id the hybrid block calls", expert_id_line,
        expert_id_line + "        sycl::event{}.wait();\n")
    layout_line = "        return layout_bytes != 0 ? layout_bytes : expert_size;\n"
    sub("N1", "wait in a lambda that lambda calls", layout_line, "        sycl::event{}.wait();\n" + layout_line)
    sib = "static void move_pending_cpu_scatter_to_sibling() {\n"
    sub("N1", "future.get in the sibling move", sib, sib + "    g_pending_scatter.future.get();\n")
    # One mutant per rule that N1 cannot see: each removes a positive pin or adds a caller outside the
    # scanned code, where the scan never looks for waits.
    sub("N3", "prologue closure stops waiting for its inputs", pro_closure,
        "    return [marker = std::move(marker), zero_out = inputs.zero_out, zero_bytes = inputs.zero_bytes]() {\n"
        "        (void) marker;\n    };\n")
    sub("N4", "join host_task closure stops joining the job", join_task,
        "cgh.host_task([job, stamps = std::move(stamps)]() { (void) job; });")
    boundary_open = (
        "void ggml_sycl_cpu_tg_flush_pending() {\n    moe_hostpath_wait_context boundary_waits(MOE_WAIT_B7);\n"
    )
    sub("N4", "job join called from the graph-boundary flush, outside the scanned code", boundary_open,
        boundary_open + "    moe_cpu_job_join(g_pending_scatter.future.share(), nullptr);\n")
    pending_any_open = "bool ggml_sycl_cpu_tg_pending_any() {\n"
    sub("N6", "boundary wait called from pending_any, outside the scanned code", pending_any_open,
        pending_any_open + "    pending_cpu_scatter_wait_prev_events();\n")
    # llama.cpp-7pvo (N7): the event-chained hybrid decode route.
    route_call = "    if (ggml_sycl_moe_hybrid_decode_event_chained(ctx, src0, src1, dst)) {\n        return;\n    }\n"
    b1_block = (
        "        {\n            moe_hostpath_wait_timer wait_timer(MOE_WAIT_B1);\n"
        "            ids_copied = ggml_sycl_copy_ids_to_host(ctx, ids, ids_host);\n        }\n"
    )
    if route_call in src and b1_block in src:
        moved = src.replace(route_call, "", 1)
        moved = moved.replace(b1_block, b1_block + route_call.replace("    ", "        ", 1), 1)
        out.append(("route call moved after the B1 readback", moved, "N7"))
    else:
        raise ContractError(f"FAIL: self-test anchor for the route-call mutants not found ({TAG_7PVO})")
    sub("N7", "route call dropped", route_call, "")
    sub("N7", "route call no longer returns", route_call,
        "    if (ggml_sycl_moe_hybrid_decode_event_chained(ctx, src0, src1, dst)) {\n    }\n")
    sub("N7", "route renamed away", "static bool ggml_sycl_moe_hybrid_decode_event_chained(",
        "static bool ggml_sycl_moe_hybrid_decode_event_chained_v2(")
    # Up to the call's open paren, so the anchor holds wherever clang-format breaks the arguments.
    ids_copy = "    const sycl::event ids_d2h = ggml_sycl::mem_copy_async("
    sub("N1", "route copies the ids synchronously", ids_copy,
        "    ggml_sycl::mem_copy(pool.ids_handle(), 0, ids_storage.handle, 0, 0, *stream);\n" + ids_copy)
    sub("N7", "ids copy's event left out of the job inputs", "    job_inputs.events.push_back(ids_d2h);\n", "")
    sub("N1", "route waits for the ids copy", "    job_inputs.events.push_back(ids_d2h);\n",
        "    job_inputs.events.push_back(ids_d2h);\n    ids_d2h.wait();\n")
    sub("N1", "route reads the ids back through the retained readback", "    job_inputs.events.push_back(ids_d2h);\n",
        "    job_inputs.events.push_back(ids_d2h);\n"
        "    {\n        std::vector<int32_t> ids_probe;\n"
        "        ggml_sycl_copy_ids_to_host(ctx, ids, ids_probe);\n    }\n")
    sub("N7", "slot flush stops placing flagged rows", "moe_host_scatter_submit_flagged(slot, joins, ws.events);",
        "moe_host_scatter_submit_flagged_unused(slot, joins, ws.events);")
    sub("N5", "flagged scatter handed no joins", "moe_host_scatter_submit_flagged(slot, joins, ws.events);",
        "moe_host_scatter_submit_flagged(slot, {}, ws.events);")
    sub("N5", "flagged scatter copies drop their dependency", "rows_bytes, queue, deps);", "rows_bytes, queue);")
    sub("N7", "record build no longer reached", "ggml_sycl_moe_hybrid_decode_record_build(ctx, src0,",
        "ggml_sycl_moe_hybrid_decode_record_build_retired(ctx, src0,")
    # llama.cpp-7pvo (N8): the route takes the op in the default configuration.
    sub("N8", "route refuses while the always-on dispatch stats are on",
        "g_moe_pp_profile_enabled) {\n        return false;",
        "g_moe_pp_profile_enabled || ggml_sycl::MoeDispatchStats::enabled()) {\n        return false;")
    sub("N8", "CPU job records no dispatch-stats residency", "stats->record_route_residency(",
        "stats->record_route_residency_retired(")
    predictor_block = (
        "        if (predictor) {\n"
        "            predictor->record_actual(seq_layer, std::vector<int>(ids_rows, ids_rows + n_ids));\n        }\n"
    )
    residency_open = "        if (stats) {\n            const int n_host"
    if predictor_block in src and residency_open in src:
        late = src.replace(predictor_block, "", 1).replace(residency_open, predictor_block + residency_open, 1)
        out.append(("predictor reads the flags, not the ids", late, "N8"))
    else:
        raise ContractError(f"FAIL: self-test anchor for the predictor-order mutant not found ({TAG_7PVO})")
    sub("N8", "route ticks no token", "            stats->tick_token();\n", "")
    sub("N8", "route keeps no token tracker", "        g_moe_cputg_last_seq_layer = seq_layer;\n", "")
    sub("N8", "retained block keeps a token tracker of its own",
        "                    // Token boundary: tick when seq_layer wraps (ALWAYS, even without predictor)\n",
        "                    // Token boundary: tick when seq_layer wraps (ALWAYS, even without predictor)\n"
        "                    static thread_local int last_seq_layer_cputg = -1;\n")
    # llama.cpp-7pvo (N9): the route admits a tensor whose experts all live on the host.
    sub("N9", "record has no all-host mode", "    bool                               all_host    = false;",
        "    bool                               all_host_v2 = false;")
    sub("N9", "builder never records the all-host mode", "    record->all_host    = all_host;\n", "")
    sub("N9", "builder decides all-host from nothing", "    const bool all_host = counts.on_device == 0;",
        "    const bool all_host = false;")
    sub("N9", "builder refuses the all-host case", "    const bool all_host = counts.on_device == 0;\n",
        "    const bool all_host = counts.on_device == 0;\n"
        "    if (all_host) {\n        return ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED;\n    }\n")
    sub("N9", "builder refuses a tensor with no device layout", "    if (!all_host && materialized < 0) {",
        "    if (materialized < 0) {")
    sub("N9", "builder refuses a tensor with no substitute", "    if ((!all_host && !substitute) || !any_host) {",
        "    if (!substitute || !any_host) {")
    no_layout = (
        "    if (!all_host && materialized < 0) {\n"
        "        return refuse(\"no device expert is materialized in a layout the batched kernel reads\");\n"
        "    }\n"
    )
    sub("N9", "builder retries a tensor with no device layout", no_layout,
        "    if (materialized < 0) {\n        return ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY;\n    }\n")
    sub("N9", "builder refuses the all-host case in a final else", no_layout,
        "    if (!all_host) {\n        (void) materialized;\n    } else {\n"
        "        return refuse(\"all host\");\n    }\n")
    sub("N9", "builder refuses the all-host case in a final else after a link that ors in !all_host", no_layout,
        "    if (n_as < 0 || !all_host) {\n        (void) materialized;\n    } else {\n"
        "        return refuse(\"all host\");\n    }\n")
    sub("N9", "builder refuses the all-host case in a final else after all_host == false", no_layout,
        "    if (all_host == false) {\n        (void) materialized;\n    } else {\n"
        "        return refuse(\"all host\");\n    }\n")
    sub("N9", "builder refuses a tensor with no device layout in a final else", no_layout,
        "    if (materialized >= 0) {\n        (void) materialized;\n    } else {\n"
        "        return refuse(\"no device layout\");\n    }\n")
    sub("N9", "builder refuses on a condition that ors in all_host", no_layout,
        "    if (n_as < 0 || all_host) {\n        return refuse(\"all host\");\n    }\n")
    sub("N9", "builder refuses an all-host tensor's host expert",
        "            if (!view.ptr || view.on_device || view.retention) {\n"
        "                return refuse(\"a host expert has no plain host view\");",
        "            if (all_host || !view.ptr || view.on_device || view.retention) {\n"
        "                return refuse(\"a host expert has no plain host view\");")
    sub("N9", "builder refusal hides its condition from the scan",
        "    if (!all_host && materialized < 0) {\n"
        "        return refuse(\"no device expert is materialized in a layout the batched kernel reads\");\n"
        "    }\n",
        "    if (materialized < 0)\n"
        "        return refuse(\"no device expert is materialized in a layout the batched kernel reads\");\n")
    sub("N9", "route runs the device half for an all-host record", "    if (!record->all_host) {",
        "    if (true) {")
    sub("N9", "route refuses an all-host record",
        "    if (!ggml_sycl_moe_hybrid_decode_record_current(ctx, src0, n_ids, &record)) {",
        "    if (!ggml_sycl_moe_hybrid_decode_record_current(ctx, src0, n_ids, &record) || record->all_host) {")
    sub("setup", "hybrid block renamed away", HYBRID_MARKER, "        if (moe_hybrid_with_plan && true) {")
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
        misfiled = []
        for name, mutated, rule in built:
            try:
                check(mutated)
            except ContractError as ex:
                tag = re.match(r"FAIL \[(N\d)\]", str(ex))
                got = tag.group(1) if tag else "setup"
                if got != rule:
                    misfiled.append(f"{name} (failed {got}, expected {rule})")
                continue
            survivors.append(name)
        if survivors:
            print(f"FAIL: {len(survivors)} mutant(s) passed the gate: {', '.join(survivors)} ({TAG})")
            return 1
        if misfiled:
            print(f"FAIL: {len(misfiled)} mutant(s) failed on another rule than the one they name: "
                  f"{'; '.join(misfiled)} ({TAG})")
            return 1
        print(f"self-test: {len(built)} mutants all fail as required, each on the rule it names")

    try:
        check(src)
    except ContractError as ex:
        print(ex)
        return 1
    print(f"OK: no host wait on the submitting thread after the ids readback ({TAG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
