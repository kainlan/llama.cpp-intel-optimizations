#!/usr/bin/env python3
"""CPU expert job joins on a join-only in-order queue (llama.cpp-wfsl).

Companion of test-sycl-cpu-expert-pool-stamps-source.py (the join's stamps) and
test-sycl-moe-hostpath-no-host-wait-source.py (no wait on the submitting thread).

THE COST.  The hybrid decode route joins each CPU expert job with a host_task.  A
host_task's command has no queue, so before posting it the SYCL runtime
(Command::flushCrossQueueDeps) queries the status of each device event it depends
on.  On the compute stream that dependency is the device-expert kernel the route
has just submitted, and the Level Zero driver answers a query on an unsignaled
event with a sched_yield, which parks the submitting thread on a busy host.  That
was two yields per layer on Qwen3.8 decode.  On a queue that holds only joins, a
join's in-order predecessor is the previous join, a host-task event with no native
handle, which the runtime never queries.

THE HAZARD.  A join there is posted as soon as its predecessors complete, before
its job's inputs are ready, and then holds the runtime's one host-task thread in
job.get().  Any host task those inputs reach that the runtime posts after it can
never run.  So every other host task the backend submits is recorded with its
cache, each join depends on the latest one on the stream, and anything the record
cannot cover (a host task on another queue or another device) turns the join
queue off for the rest of the process.  The census covers the backend's own sources;
the host tasks third-party libraries post are outside it: the oneDNN and oneMKL ones
are on their CPU-engine paths, the oneMath batch route that could post one is dead
(ggml-sycl.cpp, the batched GEMM route near line 55375), and oneCCL's are only under
tensor parallelism, where the joins are on two devices' streams and the latch refuses.

THE CONTRACT.  With comments blanked:
  K. (ggml-sycl.cpp moe_join_queue_enabled) GGML_SYCL_MOE_JOIN_QUEUE is read once,
     into a function-local static, default on (unset or non-zero); it is the only
     getenv of that name in the backend.
  F. (ggml-sycl.cpp moe_cpu_job_join_may_leave_stream, moe_cpu_job_join_queue) the
     selector returns nullptr, so the join goes on the stream, unless the knob is on
     and neither the backend's recording flag nor the stream's own queue state says a
     command graph records; it asks that one predicate before it asks the cache for
     the join queue.
  Q. (ggml-sycl.cpp moe_cpu_job_join_on_queue) submits the join on the queue the
     selector returns, else on `queue`, with the host_task text unchanged; nothing is
     submitted on `queue` directly.
  G. The guard.  The join depends on the latest recorded host task when there is one;
     a join that falls back to the stream is itself recorded; cpu_submit_async records
     its submissions on a non-CPU queue (all host tasks); and every host_task in the
     backend sources is the join, or goes through cpu_submit_async, or is exempt for a
     stated reason that the gate also pins (host_task_mode is only ever set false; the
     Q1/NVFP4 direct route is not validated; a test hook; dpct's host_buffer, which no
     backend source reaches).  The test programs under ggml-sycl/tests are not backend
     code and are not censused.
  L. (unified-cache.cpp) the latch: a host task or a join on a queue that is not a
     cache's queue_, or on a second cache, refuses the join queue for the process, as
     does a host task submitted while its queue records (a graph node, which no join
     outside the graph may depend on and which runs at each replay); the refusal is
     sticky (exchange(true)) and file-static, with no declaration in unified-cache.hpp.
  U. (unified-cache.cpp shutdown_unified_cache) the owner and the refusal are reset
     when the caches are detached, so a rebuilt backend starts the guard afresh.
  W. (ggml-sycl.cpp) the [MOE-HOSTPATH-WAITS] line prints joinq=, 1 when every join of
     the token went on the join queue: each join is counted by where it went, under the
     trace only, and the counts are reset with the token's.
  O. (unified-cache) the cache owns the join queue: a unique_ptr member made once under
     std::call_once, in_order, without enable_profiling, on queue_'s context and device,
     and drained by drain_all_queues_noexcept.
  D. (ggml-sycl.cpp flush_pending_cpu_scatter_slot) the scatter still depends on the
     join events (compact, per-run copies, flagged), the job buffers are still retained
     until joins[k], and a flush that places no rows puts a barrier on the stream behind
     the joins while they may be off it (the selector's own predicate), so the stream's
     later commands still follow them and GGML_SYCL_MOE_JOIN_QUEUE=0 adds no command.
     This covers the members that are joined.  A member the device waits on its job's
     own event takes no join (llama.cpp-qwn5); test-sycl-moe-cpu-job-device-signal-source.py
     pins that path, its retention and its empty flush.
  C. (docs/backend/sycl-env-vars.md) the GGML_SYCL_MOE_JOIN_QUEUE row states the
     mechanism, the guard and fallbacks, the joinq= field, the atoi parse, and the
     oneAPI 2026.1 / driver 26.35 caveat.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin K]: no definition matching ... moe_join_queue_enabled ... (llama.cpp-wfsl)
"""

from __future__ import annotations

import bisect
import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
BACKEND = SYCL / "ggml-sycl.cpp"
DISPATCH = SYCL / "cpu-dispatch.cpp"
CACHE_CPP = SYCL / "unified-cache.cpp"
CACHE_HPP = SYCL / "unified-cache.hpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-wfsl)"
KNOB = "GGML_SYCL_MOE_JOIN_QUEUE"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n]){1,4}\'', re.S)


@functools.lru_cache(maxsize=512)
def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals and offsets."""
    def rep(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("//"):
            return " " * len(s)
        if s.startswith("/*"):
            return re.sub(r"[^\n]", " ", s)
        return s
    return _TOKEN.sub(rep, src)


@functools.lru_cache(maxsize=512)
def strip_strings(src: str) -> str:
    """Blank the contents of string literals so braces in them do not count."""
    return re.sub(r'"(?:\\.|[^"\\\n])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def body_of(code: str, signature: str, pin: str) -> tuple[str, int]:
    """The body of the first definition matching `signature` (ending at its opening brace), and its offset."""
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    i = code.index("{", m.end() - 1)
    depth = 0
    flat = strip_strings(code)
    for j in range(i, len(flat)):
        if flat[j] == "{":
            depth += 1
        elif flat[j] == "}":
            depth -= 1
            if depth == 0:
                return code[i + 1 : j], i + 1
    fail(pin, f"unbalanced body for {signature!r}")
    return "", 0


@functools.lru_cache(maxsize=64)
def brace_positions(flat: str) -> tuple[list[int], list[str]]:
    pos, kind = [], []
    for m in re.finditer(r"[{}]", flat):
        pos.append(m.start())
        kind.append(m.group(0))
    return pos, kind


def enclosing_headers(code: str, pos: int) -> list[str]:
    """The text before each `{` that encloses `pos`, innermost first (an if's condition, a lambda head, ...)."""
    flat = strip_strings(code)
    positions, kinds = brace_positions(flat)
    heads = []
    depth = 0
    for idx in range(bisect.bisect_left(positions, pos) - 1, -1, -1):
        i = positions[idx]
        if kinds[idx] == "}":
            depth += 1
        elif depth == 0:
            k = max(flat.rfind(";", 0, i), flat.rfind("{", 0, i), flat.rfind("}", 0, i))
            heads.append(code[k + 1 : i].strip())
        else:
            depth -= 1
    return heads


def guarded(code: str, pos: int, guard: str) -> bool:
    return any(re.search(guard + r"\s*$", h) for h in enclosing_headers(code, pos))


def line_of(code: str, pos: int) -> int:
    return code.count("\n", 0, pos) + 1


KNOB_SIG = r"static\s+bool\s+moe_join_queue_enabled\s*\(\s*\)\s*\{"
LEAVE_SIG = r"static\s+bool\s+moe_cpu_job_join_may_leave_stream\s*\(\s*sycl::queue\s*&\s*stream\s*\)\s*\{"
SELECT_SIG = r"static\s+sycl::queue\s*\*\s*moe_cpu_job_join_queue\s*\([^)]*\)\s*\{"
RECORDING = r"ext_oneapi_get_state\s*\(\s*\)\s*{op}\s*sycl::ext::oneapi::experimental::queue_state::recording"
JOIN_ON_QUEUE_SIG = r"static\s+sycl::event\s+moe_cpu_job_join_on_queue\s*\([^)]*\)\s*\{"
FLUSH_SIG = r"static\s+void\s+flush_pending_cpu_scatter_slot\s*\(\s*pending_cpu_scatter\s*&\s*slot\s*\)\s*\{"
SUBMIT_ASYNC_SIG = r"static\s+sycl::event\s+cpu_submit_async\s*\([^)]*\)\s*\{"
NOTE_SIG = r"void\s+unified_cache_note_host_task\s*\([^)]*\)\s*\{"
FOR_SIG = r"sycl::queue\s*\*\s*unified_cache_moe_join_queue_for\s*\([^)]*\)\s*\{"
CACHE_OF_SIG = r"static\s+unified_cache\s*\*\s*moe_join_queue_cache_of\s*\([^)]*\)\s*\{"
REFUSE_SIG = r"void\s+unified_cache_moe_join_queue_refuse\s*\([^)]*\)\s*\{"
GET_QUEUE_SIG = r"sycl::queue\s*\*\s*unified_cache::get_moe_join_queue\s*\(\s*\)\s*\{"
AFTER_SIG = r"sycl::queue\s*\*\s*unified_cache::moe_join_queue_after\s*\([^)]*\)\s*\{"
RECORD_SIG = r"void\s+unified_cache::note_compute_host_task\s*\([^)]*\)\s*\{"
DRAIN_SIG = r"bool\s+unified_cache::drain_all_queues_noexcept\s*\(\s*\)\s*noexcept\s*\{"
SHUTDOWN_SIG = r"\bbool\s+shutdown_unified_cache\s*\(\s*\)\s*\{"
CENSUS_SIG = r"struct\s+moe_hostpath_wait_census\s*\{"
COUNT_JOIN_SIG = r"static\s+void\s+moe_hostpath_waits_join\s*\(\s*bool\s+on_join_queue\s*\)\s*\{"
READBACK_SIG = r"static\s+void\s+moe_hostpath_waits_readback\s*\([^)]*\)\s*\{"
JOIN_TASK = "cgh.host_task([job, stamps = std::move(stamps)]() { moe_cpu_job_join(job, stamps.get()); });"


def check_knob(backend: str, sources: dict) -> None:
    code = blank_comments(backend)
    body, _ = body_of(code, KNOB_SIG, "K")
    if not re.search(r"static\s+const\s+bool\s+enabled\s*=\s*\[\s*\]\s*(\(\s*\))?\s*\{", body):
        fail("K", "moe_join_queue_enabled does not read GGML_SYCL_MOE_JOIN_QUEUE once into a function-local static")
    if not re.search(r'std::getenv\s*\(\s*"' + KNOB + r'"\s*\)', body):
        fail("K", "moe_join_queue_enabled does not read GGML_SYCL_MOE_JOIN_QUEUE")
    if not re.search(r"return\s+env\s*==\s*nullptr\s*\|\|\s*std::atoi\s*\(\s*env\s*\)\s*!=\s*0\s*;", body):
        fail("K", "GGML_SYCL_MOE_JOIN_QUEUE is not on by default (unset or non-zero)")
    if not re.search(r"return\s+enabled\s*;", body):
        fail("K", "moe_join_queue_enabled does not return the value it read once")
    reads = 0
    for name, text in sources.items():
        if KNOB not in text:
            continue
        reads += len(re.findall(r'getenv\s*\(\s*"' + KNOB + r'"', blank_comments(text)))
    if reads != 1:
        fail("K", f"GGML_SYCL_MOE_JOIN_QUEUE is read at {reads} sites; the one read is moe_join_queue_enabled's")


def check_select(backend: str) -> None:
    code = blank_comments(backend)
    leave, _ = body_of(code, LEAVE_SIG, "F")
    pred = re.fullmatch(r"\s*return\s+moe_join_queue_enabled\s*\(\s*\)\s*&&\s*!\s*ggml_sycl_graph_recording_active\s*\(\s*\)"
                        r"\s*&&\s*stream\." + RECORDING.format(op="!=") + r"\s*;\s*", leave)
    if not pred:
        fail("F", "moe_cpu_job_join_may_leave_stream is not: the knob is on, the backend does not record, and the "
                  "stream's own queue state is not recording")
    body, _ = body_of(code, SELECT_SIG, "F")
    off = re.search(r"if\s*\(\s*!moe_cpu_job_join_may_leave_stream\s*\(\s*stream\s*\)\s*\)\s*\{\s*return\s+nullptr\s*;\s*\}",
                    body)
    if not off:
        fail("F", "moe_cpu_job_join_queue does not return nullptr while the joins may not leave the stream")
    ask = re.search(r"return\s+ggml_sycl::unified_cache_moe_join_queue_for\s*\(\s*stream\s*,\s*after\s*,\s*has_after"
                    r"\s*\)\s*;", body)
    if not ask or ask.start() < off.end():
        fail("F", "moe_cpu_job_join_queue does not ask the cache for the join queue after its fallbacks")


def check_join(backend: str) -> None:
    code = blank_comments(backend)
    body, off = body_of(code, JOIN_ON_QUEUE_SIG, "Q")
    if not re.search(r"moe_cpu_job_join_queue\s*\(\s*queue\s*,\s*&after\s*,\s*&has_after\s*\)", body):
        fail("Q", "moe_cpu_job_join_on_queue does not ask moe_cpu_job_join_queue where the join goes")
    if not re.search(r"sycl::queue\s*&\s*target\s*=\s*join_q\s*\?\s*\*join_q\s*:\s*queue\s*;", body):
        fail("Q", "moe_cpu_job_join_on_queue does not submit on the join queue when the selector returns one")
    sub = re.search(r"\btarget\.submit\s*\(", body)
    if not sub:
        fail("Q", "moe_cpu_job_join_on_queue does not submit the join on `target`")
    if re.search(r"\bqueue\s*\.\s*submit\s*\(", body):
        fail("Q", "moe_cpu_job_join_on_queue submits on the stream directly")
    task = body.find(JOIN_TASK)
    if task < 0:
        fail("Q", "the join's host_task is not the one the stamps gate pins")
    dep = re.search(r"cgh\.depends_on\s*\(\s*after\s*\)", body)
    if not dep:
        fail("G", "the join does not depend on the latest recorded host task")
    if dep.start() < sub.end() or dep.start() > task:
        fail("G", "the join's dependency on the latest recorded host task is not in its command group")
    if not guarded(code, off + dep.start(), r"if\s*\(\s*has_after\s*\)"):
        fail("G", "the join depends on `after` outside `if (has_after)`")
    note = re.search(r"ggml_sycl::unified_cache_note_host_task\s*\(\s*queue\s*,\s*join\s*\)", body)
    if not note:
        fail("G", "a join that falls back to the stream is not recorded as one of its host tasks")
    if not guarded(code, off + note.start(), r"if\s*\(\s*!join_q\s*\)"):
        fail("G", "the fallback join is recorded outside `if (!join_q)`")


def check_submit_async(dispatch: str) -> None:
    code = blank_comments(dispatch)
    body, off = body_of(code, SUBMIT_ASYNC_SIG, "G")
    sub = re.search(r"sycl::event\s+evt\s*=\s*cpu_q->submit\s*\(", body)
    note = re.search(r"ggml_sycl::unified_cache_note_host_task\s*\(\s*\*cpu_q\s*,\s*evt\s*\)", body)
    if not sub or not note or note.start() < sub.end():
        fail("G", "cpu_submit_async does not record its submission on a GPU queue as a host task")
    if not guarded(code, off + note.start(), r"if\s*\(\s*!submit_on_cpu_queue\s*\)"):
        fail("G", "cpu_submit_async records a host task outside `if (!submit_on_cpu_queue)`")


def classify_host_task(name: str, heads: list[str]) -> str | None:
    if name == "ggml-sycl.cpp":
        if any(re.search(r"\bmoe_cpu_job_join_on_queue\s*\(", h) for h in heads):
            return "join"
        if any(re.search(r"\btest_moe_ptr_table_dispatch_bundle_retains_table_compact_missing\s*\(", h) for h in heads):
            return "test-hook"
        if any("exact_queue->submit" in h for h in heads):
            return "q1-direct"
    if name == "cpu-dispatch.cpp":
        if any(re.search(r"cpu_submit_async\s*\(\s*gpu_q\b", h) for h in heads):
            return "cpu-submit-async"
        if any(re.search(r"else\s+if\s*\(\s*host_task\s*\)\s*$", h) for h in heads):
            return "host-task-mode"
    if name == "dpct/helper.hpp":
        if any("~host_buffer" in h for h in heads):
            return "dpct-host-buffer"
    return None


def check_census(sources: dict) -> None:
    kinds = set()
    for name, text in sorted(sources.items()):
        if "host_task" not in text:
            continue
        code = blank_comments(text)
        for m in re.finditer(r"\bhost_task\s*\(\s*\[", code):
            kind = classify_host_task(name, enclosing_headers(code, m.start()))
            if kind is None:
                fail("G", f"host_task at {name}:{line_of(code, m.start())} is not recorded for the join queue's guard; "
                     "submit it through cpu_submit_async or record its event with unified_cache_note_host_task")
            kinds.add(kind)
    if "join" not in kinds:
        fail("G", "the census found no join host_task; the census is not reading the backend")
    if "host-task-mode" in kinds:
        backend_calls = []
        for name, text in sources.items():
            if "ggml_sycl_host_task_mode_set" not in text:
                continue
            code = blank_comments(text)
            for m in re.finditer(r"\bggml_sycl_host_task_mode_set\s*\(([^)]*)\)", code):
                prefix = code[code.rfind("\n", 0, m.start()) + 1 : m.start()]
                if re.search(r"\bvoid\b", prefix):
                    continue  # its declaration or definition, not a call
                backend_calls.append((name, m.group(1).strip()))
        if not backend_calls or any(arg != "false" for _, arg in backend_calls):
            fail("G", "ggml_sycl_host_task_mode_set is called with something other than false, so cpu-dispatch's "
                 "host_task_mode branches can run and their host tasks are not recorded")
    if "q1-direct" in kinds:
        backend = blank_comments(sources["ggml-sycl.cpp"])
        if not re.search(r"constexpr\s+bool\s+k_moe_mmid_direct_route_validated\s*=\s*false\s*;", backend):
            fail("G", "the Q1/NVFP4 direct route is validated, so its terminal host task runs and is not recorded")
    if "dpct-host-buffer" in kinds:
        for name, text in sources.items():
            if name.startswith("dpct/") or "dpct_memcpy" not in text:
                continue
            if re.search(r"\b(?:async_)?dpct_memcpy\s*\(", blank_comments(text)):
                fail("G", f"{name} calls dpct_memcpy, whose host_buffer frees through a host task that is not recorded")


def check_latch(cache_cpp: str) -> None:
    code = blank_comments(cache_cpp)
    body, _ = body_of(code, CACHE_OF_SIG, "L")
    if not re.search(r"cache\s*&&\s*cache->get_queue\s*\(\s*\)\s*==\s*q\s*\?\s*cache\s*:\s*nullptr", body):
        fail("L", "moe_join_queue_cache_of accepts a queue that is not the cache's queue_")
    body, _ = body_of(code, REFUSE_SIG, "L")
    if not re.search(r"!\s*g_moe_join_queue_refused\.exchange\s*\(\s*true", body):
        fail("L", "the join queue's refusal is not a sticky exchange(true)")
    if not re.search(r"\bstatic\s+" + REFUSE_SIG, code):
        fail("L", "unified_cache_moe_join_queue_refuse is not file-static")
    for sig, what, last in ((NOTE_SIG, "a host task", r"cache->note_compute_host_task\s*\(\s*e\s*\)"),
                            (FOR_SIG, "a join", r"return\s+cache->moe_join_queue_after\s*\(\s*after\s*,\s*has_after\s*\)")):
        body, _ = body_of(code, sig, "L")
        refused = re.search(r"if\s*\(\s*g_moe_join_queue_refused\.load\s*\([^)]*\)\s*\)\s*\{?\s*return\b", body)
        no_cache = re.search(r"if\s*\(\s*!cache\s*\)\s*\{\s*unified_cache_moe_join_queue_refuse\s*\(", body)
        no_claim = re.search(r"if\s*\(\s*!moe_join_queue_claim\s*\(\s*cache\s*\)\s*\)\s*\{\s*"
                             r"unified_cache_moe_join_queue_refuse\s*\(", body)
        tail = re.search(last, body)
        if not refused:
            fail("L", f"{what} is handled after the join queue was refused")
        if not no_cache:
            fail("L", f"{what} on a queue that is not a cache's queue_ does not refuse the join queue")
        if not no_claim:
            fail("L", f"{what} on a second cache's queue_ does not refuse the join queue")
        if not tail or tail.start() < max(no_cache.end(), no_claim.end()):
            fail("L", f"{what} reaches the cache before the latch's checks")
    body, _ = body_of(code, NOTE_SIG, "L")
    rec = re.search(r"if\s*\(\s*q\." + RECORDING.format(op="==") + r"\s*\)\s*\{\s*"
                    r"unified_cache_moe_join_queue_refuse\s*\([^;]*\)\s*;\s*return\s*;", body)
    look = re.search(r"moe_join_queue_cache_of\s*\(\s*q\s*\)", body)
    if not rec or not look or rec.start() > look.start():
        fail("L", "a host task submitted while its queue records does not refuse the join queue before it is recorded")


def check_owner(cache_hpp: str, cache_cpp: str) -> None:
    hpp = blank_comments(cache_hpp)
    if re.search(r"\bunified_cache_moe_join_queue_refuse\b", hpp):
        fail("L", "unified-cache.hpp declares unified_cache_moe_join_queue_refuse, which only the latch may call")
    if not re.search(r"sycl::queue\s*\*\s*get_moe_join_queue\s*\(\s*\)\s*;", hpp):
        fail("O", "unified_cache declares no get_moe_join_queue()")
    if not re.search(r"std::unique_ptr\s*<\s*sycl::queue\s*>\s+moe_join_queue_\s*;", hpp):
        fail("O", "unified_cache does not own the join queue through a unique_ptr member")
    if not re.search(r"std::once_flag\s+moe_join_queue_once_\s*;", hpp):
        fail("O", "unified_cache has no once_flag for the join queue")
    code = blank_comments(cache_cpp)
    body, _ = body_of(code, GET_QUEUE_SIG, "O")
    once = re.search(r"std::call_once\s*\(\s*moe_join_queue_once_", body)
    make = re.search(r"moe_join_queue_\s*=\s*std::make_unique\s*<\s*sycl::queue\s*>\s*\(\s*queue_\.get_context\s*\(\s*\)"
                     r"\s*,\s*queue_\.get_device\s*\(\s*\)\s*,\s*sycl::property::queue::in_order\s*\{\s*\}\s*\)\s*;", body)
    if not once or not make or make.start() < once.end():
        fail("O", "get_moe_join_queue does not make one in-order queue on queue_'s context and device under call_once")
    if "enable_profiling" in body:
        fail("O", "the join queue is profiling-enabled, which makes any status query on it a busy-wait")
    if not re.search(r"return\s+moe_join_queue_\.get\s*\(\s*\)\s*;", body):
        fail("O", "get_moe_join_queue does not return the owned queue")
    body, _ = body_of(code, DRAIN_SIG, "O")
    if "moe_join_queue_.get()" not in body:
        fail("O", "drain_all_queues_noexcept does not drain the join queue")
    body, _ = body_of(code, AFTER_SIG, "O")
    if not re.search(r"sycl::queue\s*\*\s*q\s*=\s*get_moe_join_queue\s*\(\s*\)\s*;", body):
        fail("O", "moe_join_queue_after does not hand out the cache's own join queue")
    if not re.search(r"\*after\s*=\s*last_host_task_\s*;", body) or not re.search(r"\*has_after\s*=\s*true\s*;", body):
        fail("G", "moe_join_queue_after does not hand the join the latest recorded host task")
    body, _ = body_of(code, RECORD_SIG, "O")
    if not re.search(r"last_host_task_\s*=\s*e\s*;", body):
        fail("G", "note_compute_host_task does not record the host task's event")


def check_scatter(backend: str) -> None:
    code = blank_comments(backend)
    body, off = body_of(code, FLUSH_SIG, "D")
    flat = re.sub(r"\s+", " ", body)
    for needle, what in (
        ("joins.push_back(moe_cpu_job_join_on_queue(*slot.stream, group[k]->future.share(), group[k]->job_stamps));",
         "the joins are not made by moe_cpu_job_join_on_queue for the slot's stream"),
        ("moe_host_scatter_submit_compact(slot, ws, joins, &n_copies, &why)", "the compact scatter does not depend on the joins"),
        ("run.src_offset, run.bytes, *slot.stream, joins));", "the per-run scatter copies do not depend on the joins"),
        ("moe_host_scatter_submit_flagged(slot, joins, ws.events);", "the flagged scatter does not depend on the joins"),
        ("ggml_sycl::retain_handles_until_event(std::move(job_buffers), joins[k]);",
         "the job buffers are not retained until each job's own join"),
    ):
        if needle not in flat:
            fail("D", what)
    bar = re.search(r"slot\.stream->ext_oneapi_submit_barrier\s*\(\s*joins\s*\)", body)
    if not bar or not guarded(code, off + bar.start(), r"if\s*\(\s*ws\.events\.empty\s*\(\s*\)\s*\)"):
        fail("D", "a flush that places no rows does not keep the stream's later commands behind its joins")
    if not guarded(code, off + bar.start(), r"if\s*\(\s*moe_cpu_job_join_may_leave_stream\s*\(\s*\*slot\.stream\s*\)\s*\)"):
        fail("D", "the empty flush's barrier is submitted even when the joins cannot have left the stream")


def check_unload(cache_cpp: str) -> None:
    code = blank_comments(cache_cpp)
    body, _ = body_of(code, SHUTDOWN_SIG, "U")
    clear = re.search(r"g_device_caches\.clear\(\)\s*;", body)
    owner = re.search(r"g_moe_join_queue_owner\.store\s*\(\s*nullptr\b", body)
    refused = re.search(r"g_moe_join_queue_refused\.store\s*\(\s*false\b", body)
    if not clear or not owner or not refused or owner.start() < clear.end() or refused.start() < clear.end():
        fail("U", "shutdown_unified_cache does not reset the join queue's owner and refusal when it detaches the caches")


def check_trace(backend: str) -> None:
    code = blank_comments(backend)
    body, _ = body_of(code, CENSUS_SIG, "W")
    for field in ("joins_join_queue", "joins_stream"):
        if not re.search(r"\buint64_t\s+" + field + r"\s*=\s*0\s*;", body):
            fail("W", f"moe_hostpath_wait_census does not hold {field}")
    body, _ = body_of(code, COUNT_JOIN_SIG, "W")
    flat = re.sub(r"\s+", " ", body)
    if not re.search(r"if \(!moe_hostpath_waits_enabled\(\)\) \{ return; \} if \(on_join_queue\) \{ "
                     r"g_moe_hostpath_waits\.joins_join_queue \+= 1; \} else \{ g_moe_hostpath_waits\.joins_stream \+= 1; \}",
                     flat):
        fail("W", "moe_hostpath_waits_join does not count a join by where it went, under the trace only")
    body, _ = body_of(code, JOIN_ON_QUEUE_SIG, "W")
    if not re.search(r"moe_hostpath_waits_join\s*\(\s*join_q\s*!=\s*nullptr\s*\)\s*;", body):
        fail("W", "moe_cpu_job_join_on_queue does not count its join by where it went")
    body, _ = body_of(code, READBACK_SIG, "W")
    flat = re.sub(r"\s+", " ", body)
    m = re.search(r'GGML_LOG_WARN\(\s*((?:"[^"]*"\s*)+)', flat)
    fmt = "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""
    if not fmt.startswith("[MOE-HOSTPATH-WAITS]") or not fmt.endswith(" joinq=%d\\n"):
        fail("W", "the [MOE-HOSTPATH-WAITS] line does not end with joinq=")
    if "const int joinq = w.joins_join_queue > 0 && w.joins_stream == 0 ? 1 : 0;" not in flat or \
            not re.search(r", joinq\);", flat):
        fail("W", "joinq= is not 1 exactly when every join of the token went on the join queue")
    if not re.search(r"w\.joins_join_queue\s*=\s*0;", flat) or not re.search(r"w\.joins_stream\s*=\s*0;", flat):
        fail("W", "the join counts are not reset with the token's other counts")


def check_doc(doc: str) -> None:
    row = None
    for line in doc.splitlines():
        if line.startswith("| `" + KNOB):
            row = line
            break
    if row is None:
        fail("C", "docs/backend/sycl-env-vars.md has no GGML_SYCL_MOE_JOIN_QUEUE row")
    for needle, what in (("flushCrossQueueDeps", "the runtime's dependency flush"),
                         ("sched_yield", "the driver's yield"),
                         ("never queried", "why a host-task predecessor costs nothing"),
                         ("compute stream", "the fallback"),
                         ("joinq=", "the trace field that shows the join queue in use"),
                         ("atoi", "how the value is parsed"),
                         ("oneAPI 2026.1", "the oneAPI version it was verified on"),
                         ("26.35", "the driver it was verified on"),
                         ("re-measure", "the re-measure caveat")):
        if needle not in row:
            fail("C", f"the GGML_SYCL_MOE_JOIN_QUEUE row does not state {what} ({needle!r})")


def read_sources(root: Path) -> dict:
    out = {}
    for key, path in (("backend", BACKEND), ("dispatch", DISPATCH), ("cache_cpp", CACHE_CPP), ("cache_hpp", CACHE_HPP),
                      ("doc", DOC)):
        p = root / path.relative_to(ROOT)
        out[key] = p.read_text(encoding="utf-8") if p.exists() else None
    sources = {}
    sycl = root / SYCL.relative_to(ROOT)
    for p in sorted(sycl.rglob("*.[ch]pp")):
        name = p.relative_to(sycl).as_posix()
        # ggml-sycl/tests holds test programs, each its own process with its own fixtures, not backend code.
        if name.startswith("tests/"):
            continue
        sources[name] = p.read_text(encoding="utf-8", errors="replace")
    out["sources"] = sources
    return out


def check(s: dict) -> None:
    for key, path in (("backend", BACKEND), ("dispatch", DISPATCH), ("cache_cpp", CACHE_CPP), ("cache_hpp", CACHE_HPP),
                      ("doc", DOC)):
        if s[key] is None:
            fail("0", f"{path.relative_to(ROOT)} is missing")
    sources = dict(s["sources"])
    sources.update({"ggml-sycl.cpp": s["backend"], "cpu-dispatch.cpp": s["dispatch"],
                    "unified-cache.cpp": s["cache_cpp"], "unified-cache.hpp": s["cache_hpp"]})
    check_knob(s["backend"], sources)
    check_select(s["backend"])
    check_join(s["backend"])
    check_submit_async(s["dispatch"])
    check_census(sources)
    check_latch(s["cache_cpp"])
    check_owner(s["cache_hpp"], s["cache_cpp"])
    check_scatter(s["backend"])
    check_unload(s["cache_cpp"])
    check_trace(s["backend"])
    check_doc(s["doc"])


# --self-test ---------------------------------------------------------------------------------------


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    mutants = [
        ("K", "backend", "knob read per call",
         sub(r"(static bool moe_join_queue_enabled\(\) \{\n\s*)static const bool enabled", r"\1const bool enabled")),
        ("K", "backend", "knob off by default",
         sub(r'(GGML_SYCL_MOE_JOIN_QUEUE"\);\n\s*)return env == nullptr \|\| std::atoi\(env\) != 0;',
             r"\1return env != nullptr && std::atoi(env) != 0;")),
        ("K", "dispatch", "a second read of the knob",
         sub(r"(static inline void cpu_wait_chain_event\(\) \{)",
             r'\1\n    (void) std::getenv("GGML_SYCL_MOE_JOIN_QUEUE");')),
        ("F", "backend", "join queue while the backend records",
         sub(r"return moe_join_queue_enabled\(\) && !ggml_sycl_graph_recording_active\(\) &&",
             "return moe_join_queue_enabled() &&")),
        ("F", "backend", "join queue while the stream itself records",
         sub(r"&&\s*stream\.ext_oneapi_get_state\(\) != sycl::ext::oneapi::experimental::queue_state::recording;",
             ";")),
        ("F", "backend", "knob not consulted",
         sub(r"return moe_join_queue_enabled\(\) && !ggml_sycl_graph_recording_active\(\)",
             "return !ggml_sycl_graph_recording_active()")),
        ("F", "backend", "selector skips the predicate",
         sub(r"if \(!moe_cpu_job_join_may_leave_stream\(stream\)\) \{", "if (false) {")),
        ("Q", "backend", "join submitted on the stream",
         sub(r"target\.submit\(\[&\]\(sycl::handler & cgh\) \{", "queue.submit([&](sycl::handler & cgh) {")),
        ("Q", "backend", "join queue ignored",
         sub(r"sycl::queue &(\s+)target(\s+)= join_q \? \*join_q : queue;", r"sycl::queue &\1target\2= queue;")),
        ("G", "backend", "join loses the guard dependency", sub(r"cgh\.depends_on\(after\);", ";")),
        ("G", "backend", "guard dependency unconditional",
         sub(r"if \(has_after\) \{\n(\s*)cgh\.depends_on\(after\);", r"{\n\1cgh.depends_on(after);")),
        ("G", "backend", "fallback join not recorded",
         sub(r"ggml_sycl::unified_cache_note_host_task\(queue, join\);", ";")),
        ("G", "dispatch", "cpu_submit_async stops recording",
         sub(r"ggml_sycl::unified_cache_note_host_task\(\*cpu_q, evt\);", ";")),
        ("G", "dispatch", "an unrecorded host task",
         sub(r"(static inline void cpu_wait_chain_event\(\) \{)",
             r"\1\n    ggml_sycl_get_cpu_queue()->submit([&](sycl::handler & cgh) { cgh.host_task([]() {}); });")),
        ("G", "backend", "host_task_mode switched on", sub(r"ggml_sycl_host_task_mode_set\(false\);",
                                                          "ggml_sycl_host_task_mode_set(true);")),
        ("G", "backend", "Q1/NVFP4 direct route validated",
         sub(r"constexpr bool k_moe_mmid_direct_route_validated = false;",
             "constexpr bool k_moe_mmid_direct_route_validated = true;")),
        ("L", "cache_cpp", "latch accepts any queue",
         sub(r"cache && cache->get_queue\(\) == q \? cache : nullptr", "cache")),
        ("L", "cache_cpp", "refusal not sticky",
         sub(r"!g_moe_join_queue_refused\.exchange\(true", "!g_moe_join_queue_refused.load(")),
        ("L", "cache_cpp", "second cache not refused for a join",
         sub(r"(sycl::queue \* unified_cache_moe_join_queue_for\([^)]*\) \{(?:(?!\n\}).)*?)if \(!moe_join_queue_claim\(cache\)\)",
             r"\1if (false)", re.S)),
        ("L", "cache_cpp", "foreign queue not refused for a host task",
         sub(r"(void unified_cache_note_host_task\([^)]*\) \{(?:(?!\n\}).)*?)if \(!cache\) \{",
             r"\1if (false) {", re.S)),
        ("L", "cache_cpp", "recorded host task kept as the guard",
         sub(r"if \(q\.ext_oneapi_get_state\(\) == sycl::ext::oneapi::experimental::queue_state::recording\) \{",
             "if (false) {")),
        ("O", "cache_cpp", "join queue profiling-enabled",
         sub(r"(moe_join_queue_ = std::make_unique<sycl::queue>\(queue_\.get_context\(\), queue_\.get_device\(\),\s*)"
             r"sycl::property::queue::in_order\{\}\);",
             r"\1sycl::property_list{ sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} });")),
        ("O", "cache_cpp", "join queue out of order",
         sub(r"(moe_join_queue_ = std::make_unique<sycl::queue>\(queue_\.get_context\(\), queue_\.get_device\(\)),\s*"
             r"sycl::property::queue::in_order\{\}\);", r"\1);")),
        ("O", "cache_cpp", "join queue not drained", sub(r",\s*moe_join_queue_\.get\(\)", "")),
        ("O", "cache_hpp", "join queue not owned",
         sub(r"std::unique_ptr<sycl::queue>(\s+)moe_join_queue_;", r"sycl::queue *\1moe_join_queue_;")),
        ("G", "cache_cpp", "guard event not handed to the join", sub(r"\*after(\s*)= last_host_task_;", r"(void) after;")),
        ("D", "backend", "per-run copies lose the joins",
         sub(r"run\.src_offset, run\.bytes, \*slot\.stream, joins\)\);", "run.src_offset, run.bytes, *slot.stream, {}));")),
        ("D", "backend", "empty flush leaves the stream ahead of its joins",
         sub(r"slot\.stream->ext_oneapi_submit_barrier\(joins\);", ";")),
        ("D", "backend", "empty flush barrier with the knob off",
         sub(r"if \(moe_cpu_job_join_may_leave_stream\(\*slot\.stream\)\) \{", "{")),
        ("D", "backend", "buffers retained until the scatter instead of the join",
         sub(r"retain_handles_until_event\(std::move\(job_buffers\), joins\[k\]\);",
             "retain_handles_until_event(std::move(job_buffers), ws.events.front());")),
        ("L", "cache_cpp", "refusal exported",
         sub(r"static void unified_cache_moe_join_queue_refuse\(", "void unified_cache_moe_join_queue_refuse(")),
        ("L", "cache_hpp", "refusal declared for other callers",
         sub(r"(void unified_cache_note_host_task\(sycl::queue & q, const sycl::event & e\);\n)",
             r"\1void unified_cache_moe_join_queue_refuse(const char * why);\n")),
        ("U", "cache_cpp", "owner kept across shutdown", sub(r"g_moe_join_queue_owner\.store\(nullptr,[^;]*;", ";")),
        ("U", "cache_cpp", "refusal kept across shutdown", sub(r"g_moe_join_queue_refused\.store\(false,[^;]*;", ";")),
        ("W", "backend", "line loses joinq", sub(r" joinq=%d", "")),
        ("W", "backend", "join not counted", sub(r"moe_hostpath_waits_join\(join_q != nullptr\);", ";")),
        ("W", "backend", "joins counted with the trace off",
         sub(r"(static void moe_hostpath_waits_join\(bool on_join_queue\) \{\n)\s*if \(!moe_hostpath_waits_enabled\(\)\) \{\n\s*return;\n\s*\}\n",
             r"\1")),
        ("W", "backend", "joinq=1 with some joins on the stream", sub(r" && w\.joins_stream == 0 \? 1 : 0;", " ? 1 : 0;")),
        ("W", "backend", "join counts not reset", sub(r"w\.joins_stream(\s*)= 0;", r"(void) w\1;")),
        ("C", "doc", "doc row loses joinq",
         sub(r"(\| `GGML_SYCL_MOE_JOIN_QUEUE[^\n]*?)joinq=1([^\n]*?)joinq=0", r"\1join_q=1\2join_q=0")),
        ("C", "doc", "doc row loses the parse", sub(r"(\| `GGML_SYCL_MOE_JOIN_QUEUE[^\n]*?)atoi", r"\1a_to_i")),
        ("C", "doc", "doc row missing", sub(r"\| `GGML_SYCL_MOE_JOIN_QUEUE", "| `GGML_SYCL_MOE_JOIN_Q")),
        ("C", "doc", "doc row loses the version caveat",
         sub(r"(\| `GGML_SYCL_MOE_JOIN_QUEUE[^\n]*?)oneAPI 2026\.1", r"\1oneAPI")),
    ]
    failures = []
    caught = 0
    for pin, key, name, mutate in mutants:
        m = dict(s)
        m[key] = mutate(s[key] or "")
        if m[key] == s[key]:
            failures.append(f"mutant '{name}' did not apply")
            print(f"  self-test: mutant '{name}' did not apply")
            continue
        try:
            check(m)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                caught += 1
                print(f"  self-test: mutant '{name}' caught: {e}")
                continue
            failures.append(f"mutant '{name}' failed on the wrong pin: {e}")
            print(f"  self-test: mutant '{name}' failed on the wrong pin: {e}")
            continue
        failures.append(f"mutant '{name}' passed")
        print(f"  self-test: mutant '{name}' PASSED (not caught)")
    try:
        check(s)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        failures.append(f"unmodified tree fails: {e}")
        print(f"  self-test: unmodified tree FAILS: {e}")
    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {caught} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    root = ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    s = read_sources(root)
    if "--self-test" in argv:
        rc = self_test(s)
        if rc != 0:
            return rc
    try:
        check(s)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: CPU expert joins go on the cache's join-only queue behind GGML_SYCL_MOE_JOIN_QUEUE, depend on the "
          f"latest recorded host task, and the scatter still depends on them {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
