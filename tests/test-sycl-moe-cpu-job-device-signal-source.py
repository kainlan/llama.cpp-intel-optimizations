#!/usr/bin/env python3
"""The host-expert scatter waits on the CPU job's own device event (llama.cpp-qwn5).

Companion of test-sycl-cpu-job-signal-book.cpp (the ring's bookkeeping, host-only),
test-sycl-cpu-job-signal-ring.cpp (the device behaviour, lead-run),
test-sycl-moe-join-queue-source.py (the host-task joins, which stay the fallback) and
test-sycl-cpu-expert-pool-stamps-source.py (the trace's signal stamp).

THE COST.  A device command that depends on an unfinished host_task is held by the SYCL
runtime as a blocked user, and so is every command submitted after it on the same
in-order queue.  When the host_task ends, the runtime's one host-task thread enqueues
that backlog serially, about 4 us per command.  The hybrid decode's scatter depended on
a join host_task per CPU expert job, so each layer's GPU work waited for that thread.
Now the job signals a host-signalled Level Zero event after its last write and the
scatter's first commands depend on a fresh wrapper of it, which the device waits itself.

THE HAZARD.  The xe job timeout does not end a device command parked on such an event,
so every way a job ends must signal it; and an event is reset for reuse, so no wrapper
may outlive the flush that submitted its commands, and an entry is reset only after the
host waited a command of its own stream that ran after every command that waited on it.

THE GATES.  With the device waiting a job's event itself, a host wait on the stream
behind the job is a wait inside the driver, which polls for the job's whole length.  At
-ngl 40 on the B70 two threads polled there: the submitting thread, in the graph code's
whole-queue waits (ggml_sycl_trace_queue_wait: the periodic drain and the rest), and the
retained-handle reaper.  Both took CPU from the job's own row threads.  So the flush keeps
the job's future as a gate, and
those waits sleep on it first.  A gate also decides whether such a wait returns: a host
wait in the driver in front of a job that has not started can keep the job from starting
(llama.cpp-42zj; test-sycl-retained-record-gates-source.py pins that every retained
record carries every live ring's gates).  A gate is only a shared_future: wait() on it
returns for a value, an exception or a broken promise, never throws, and runs with no
lock held.  Two paths leave a gate shut.  A job that throws: the pool's handlers rethrow
into a worker with no handler, so the process aborts with the promise unset, and a waiter
sleeps until that abort.  And a worker that CpuExpertPool::stop_workers detaches, as it
leaves the job's event unsignalled.

THE CONTRACT.  With comments blanked:
  A. (ggml-sycl.cpp flush_pending_cpu_scatter_slot) a member whose wrap succeeds pushes
     the wrapper as its dependency, is counted for devsig=, and continues before the
     host-task join; moe_cpu_job_join_on_queue is called only after that branch.
  B. The device waits a job only when moe_cpu_job_device_signal_may() holds: the knob is
     on and neither the backend's recording flag nor the stream's own state records a
     graph (moe_cpu_job_stream_recording, the recording test the CPU job's three decisions
     share).  The flush asks it for the slot's stream; the one acquire helper asks it
     before it asks the cache; the three CpuExpertPool producers acquire through that
     helper and hand the job half to submit_batch; nothing else takes an entry.
  C. (cpu-expert-pool.cpp submit_batch) the worker lambda's body is one try block whose
     first statement declares the job half, signals it after the batched kernel and
     before promise->set_value(), and is followed by exactly the two rethrowing
     handlers; (cpu-job-signal.hpp) the job half's destructor and move assignment signal.
  D. (flush) a natively waited member's buffers are retained until ws.events.back(); an
     empty flush with such a member submits the barrier and keeps the barrier's event,
     and only a flush with none inserts the joins into ws.events; the last flush and the
     retired slots take ws.events and the flush's mark.
  E. cpu_job_signal_recycle is called once in the backend, in
     pending_cpu_scatter_wait_prev_events(slot), after its sycl::event::wait(events);
     the slot-pair wait is called only from ggml_sycl_cpu_tg_flush_pending.
  F. (unified-cache.cpp cpu_job_signal_ring::wrap) each wrap makes a fresh
     sycl::make_event with ownership::keep, from the entry's native event that it copied
     under the lock (it makes the event with the lock released, llama.cpp-42zj); nothing
     the ring class declares outside its function bodies and parameter lists names
     sycl::event, so no member holds a wrapper, bare or in a container; the flush clears
     its dependency list after the retention, and the wrap's out-parameter is a local of
     the member loop.
  G. The abort callback signals every ring before it calls the previous callback, and
     the SYCL watchdog signals every ring before it frees host allocations and exits.
     The pool's stop_workers signals nothing: a worker it detaches may still write the
     job's outputs, which teardown frees once the queues drain, so the job signals itself
     when it ends.
  H. GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL is read once, into a function-local static,
     default on, and is the only getenv of that name; the catalog row documents =0; the
     [MOE-HOSTPATH-WAITS] line prints devsig=, counted under the trace only.
  I. No host waits on the new events.  The flush body and the ring section of
     unified-cache.cpp (from its section comment to the testing hooks' #endif) call no
     wait, wait_and_throw, wait_for or wait_until, no get_info<...> and name neither
     zeEventHostSynchronize nor zeEventQueryStatus; the flush gets no future.  On this
     stack a status query on an in-flight event can itself be a host wait.  The ring
     resolves exactly its six entry points by dlsym and calls no Level Zero function
     directly; every host signal is the one in signal_native_locked, whose first
     statement is the release fence that orders the job's writes before the device can
     see the signal.

  J1. (flush) the natively waited member, after its wrap, pushes its shared future to the
     ring's gates (cpu_job_signal_gate_push, the backend's only call) before its wrapper
     becomes its dependency.
  J2. (ggml_backend_sycl_synchronize) the gate sleep is one unconditional statement after
     the CPU-expert flush and before both stream-wait branches, and no site but the
     synchronize and the graph code's queue wait (B9) calls ggml_sycl_cpu_job_gates_sleep;
     the statement right after the stream wait clears
     what it slept on; the sleep asks the device's list (unified_cache_cpu_job_gates) and
     waits all of it; that lookup reads the device's existing ring
     (existing_cpu_job_signal_ring) and never get_cpu_job_signal_ring, which would make one.
  J3. Clears: the synchronize clears up to the newest gate it slept on before the trace
     check, so an untraced run clears too; the boundary (pending_cpu_scatter_wait_prev_events)
     sleeps on its mark's gates up to the mark's gate serial, inside `if (!events.empty())`,
     before its event wait; cpu_job_signal_recycle clears the ring's gates up to the mark's
     gate serial after the book recycle; the ring's shutdown and abandon clear them.
  J4. No gate is queried, polled or taken: nowhere in the gate list, the synchronize's two
     gate helpers, the boundary wait or the reaper loop is a get(), wait_for() or
     wait_until() call; the list's only wait is the future.wait() of its static sleep_on,
     which wait_all and wait_upto call on a snapshot their lock_guard does not outlive; the
     reaper's gate wait sits directly in its try block, outside every store lock; and no
     lock of any form (a guard of any kind, a raw lock(), lock_shared() or try_lock() call
     through . or ->, or std::lock()) is taken in the synchronize before its sleep, in the
     gate sleep the synchronize and the queue wait share, or anywhere in the graph code's
     queue wait.
  J5. (reaper) the reaper waits each valid gate of a record with wait() inside the
     watchdog guard, right before the record's event wait.  Where a record's gates come
     from (every live ring's, taken when it is published) is the contract of
     test-sycl-retained-record-gates-source.py (llama.cpp-42zj).
  B8. The [MOE-HOSTPATH-WAITS] line names B8 after B7 and prints gate_waits=,
     gate_sleep_us= and driver_us= right after total=, reset with the token's counts; the
     catalog's trace row names B8 and the three fields, and its knob row the gates.  The
     sleep stamps its start before the lookup and its end after the sleep, and the clear
     stamps the stream wait's end as its first statement, before clear_upto, so driver_us
     is the stream wait alone.
  B7D. The graph-boundary drain's split of B7: the drain stamps its start before the gate
     lookup, keeps wait_upto's count, stamps the sleep's end right before its event wait,
     and calls moe_hostpath_waits_drain (its one caller) right after the event wait and
     before the recycle clears the gates; the helper stamps the wait's end first and, under
     the trace only, adds the count, the sleep and the wait to the token's drain fields; the
     waits line prints drain_gate_waits=, drain_gate_sleep_us= and drain_driver_us= right
     after B7=, reset with the token's counts; the catalog's trace row names them.
  B9. The graph code's whole-queue waits (ggml_sycl_trace_queue_wait): each of its two
     q->wait() calls, the untraced one and the traced one, comes right after the gate
     sleep on *q and right before ggml_sycl_cpu_job_gates_clear_queue_wait, whose only
     callers they are.  The traced one sleeps after its begin line and between t0 and t1,
     so GGML_SYCL_WAIT_TRACE prints as before and its wait_ms includes the sleep.  The
     clear stamps the queue wait's end first, clears what the sleep waited before the
     trace check, and under the trace only adds the count, the sleep and the wait to the
     token's queue fields and counts the whole under B9.  The class enum has B9 right
     after B8; the waits line names B9 after B8 and prints queue_gate_waits=,
     queue_gate_sleep_us= and queue_driver_us= right after B9=, reset with the token's
     counts; the catalog's trace row names B1..B9 and the three fields, and its knob row
     the queue waits.  Its J4 half (no lock of any form anywhere in the queue wait's body)
     runs before the B9 shape checks, so every lock there fails as J4, the hazard to
     report (a sleep under a lock its job may need never ends).  A lock around the sleep
     and the wait also breaks a B9 shape, but a lock taken before the traced begin line
     breaks none, so only J4 can catch it.
  R.  The reaper's census, trace only: the trace's enable turns it on and is its one
     caller; the reaper's timer reads no clock while it is off, stamps the gate sleep's end,
     and its waited(), right after the event wait, is the only place the four counters
     grow; the waits line takes the census once (the take zeroes every counter) and prints
     reaper=N/USus, reaper_gate_waits=, reaper_gate_sleep_us= and reaper_driver_us= after
     driver_us=; the catalog's trace row names them.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

GATE RUNTIME.  Comment blanking is one regular-expression pass and, like string
stripping, memoized; pin E finds the recycle's enclosing block within the drain's body
rather than the whole backend; the scans of the backend for a name drop the leading \b,
which keeps the regex engine from skipping ahead to the name, and test the boundary on
each match (word_matches); and the two scans over every backend source skip a file that
does not contain the name the scan looks for (cpu_job_signal_recycle, or the knob):
blanking only removes text, so such a file cannot match either way.

RED on the tree before the change:
  FAIL [pin A]: the flush does not wrap a member's device event under
  `may_signal && group[k]->device_signal` (llama.cpp-qwn5)
RED on the tree before the gates:
  FAIL [pin J1]: the natively waited member does not push its shared future to the ring's
  gates before its wrapper becomes its dependency (llama.cpp-qwn5)
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
BACKEND = SYCL / "ggml-sycl.cpp"
POOL = SYCL / "cpu-expert-pool.cpp"
MEM_CPP = SYCL / "mem-handle.cpp"
SIGNAL_HPP = SYCL / "cpu-job-signal.hpp"
CACHE_CPP = SYCL / "unified-cache.cpp"
COMMON_CPP = SYCL / "common.cpp"
HOSTPATH_HPP = SYCL / "moe-decode-hostpath.hpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-qwn5)"
KNOB = "GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


# One token at a time, leftmost first: a // or /* */ comment (an unterminated one runs to the end), or a string or
# character literal, which is kept so that a comment marker inside it stays. A quote right after a letter or digit is a
# digit separator, not a literal.
_BLANK_TOKEN = re.compile(r'//[^\n]*|/\*.*?(?:\*/|\Z)|"(?:\\.|[^"\\])*(?:"|\Z)|(?<![^\W_])\'(?:\\.|[^\'\\])*(?:\'|\Z)', re.S)
_NOT_NEWLINE = re.compile(r"[^\n]")


def _blank_token(m: re.Match) -> str:
    t = m.group(0)
    if t[0] != "/":
        return t
    return " " * len(t) if t[1] == "/" else _NOT_NEWLINE.sub(" ", t)


# Every check blanks the same few large sources again; one check per mutant reads at most six of them.
@functools.lru_cache(maxsize=8)
def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals, line breaks and offsets."""
    return _BLANK_TOKEN.sub(_blank_token, src)


# block_at strips the whole source it is handed, once per body it finds.
@functools.lru_cache(maxsize=16)
def strip_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


_WORD_CHAR = re.compile(r"\w")


def word_matches(pattern: str, text: str) -> list[re.Match]:
    """re.finditer(r"\\b" + pattern, text) for a pattern that starts with a word.  Without the leading \\b the
    regex engine skips ahead to the pattern's literal start, about ten times faster over the backend's 6 MB; the
    boundary is checked on each match instead."""
    return [m for m in re.finditer(pattern, text) if m.start() == 0 or not _WORD_CHAR.match(text, m.start() - 1)]


def block_at(code: str, brace: int, pin: str) -> tuple[str, int]:
    """The text inside the `{` at `brace` and the offset of its first character."""
    flat = strip_strings(code)
    depth = 0
    for j in range(brace, len(flat)):
        if flat[j] == "{":
            depth += 1
        elif flat[j] == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1 : j], brace + 1
    fail(pin, "unbalanced braces")
    return "", 0


def body_of(code: str, signature: str, pin: str) -> tuple[str, int]:
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    return block_at(code, code.index("{", m.end() - 1), pin)


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def enclosing_headers(code: str, pos: int) -> list[str]:
    flat_code = strip_strings(code)
    heads, depth = [], 0
    for i in range(pos - 1, -1, -1):
        c = flat_code[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                k = i - 1
                while k >= 0 and flat_code[k] not in ";{}":
                    k -= 1
                heads.append(re.sub(r"\s+", " ", code[k + 1 : i]).strip())
            else:
                depth -= 1
    return heads


FLUSH_SIG = r"static\s+void\s+flush_pending_cpu_scatter_slot\s*\(\s*pending_cpu_scatter\s*&\s*slot\s*\)\s*\{"
MAY_SIG = r"static\s+bool\s+moe_cpu_job_device_signal_may\s*\(\s*sycl::queue\s*&\s*stream\s*\)\s*\{"
REC_SIG = r"static\s+bool\s+moe_cpu_job_stream_recording\s*\(\s*sycl::queue\s*&\s*stream\s*\)\s*\{"
ACQ_SIG = r"static\s+void\s+moe_cpu_job_device_signal_acquire\s*\([^)]*\)\s*\{"
KNOB_SIG = r"static\s+bool\s+moe_cpu_job_device_signal_enabled\s*\(\s*\)\s*\{"
WAIT_SLOT_SIG = r"static\s+void\s+pending_cpu_scatter_wait_prev_events\s*\(\s*pending_cpu_scatter\s*&\s*slot\s*\)\s*\{"
WAIT_PAIR_SIG = r"static\s+void\s+pending_cpu_scatter_wait_prev_events\s*\(\s*\)\s*\{"
TG_FLUSH_SIG = r"void\s+ggml_sycl_cpu_tg_flush_pending\s*\(\s*\)\s*\{"
COUNT_SIG = r"static\s+void\s+moe_hostpath_waits_device_signal\s*\(\s*\)\s*\{"
SUBMIT_SIG = r"std::future<void>\s+CpuExpertPool::submit_batch\s*\([^{]*\)\s*\{"
RING_WRAP_SIG = r"\n    bool\s+wrap\s*\(\s*uint32_t\s+id\s*,[^)]*\)\s*\{"
RING_CLASS_SIG = r"class\s+cpu_job_signal_ring\s*\{"
WATCHDOG_EXIT = r"ggml_sycl_cleanup_host_allocations\s*\(\s*\)\s*;"

RING_SECTION_START = "// --- CPU expert job device signals (llama.cpp-qwn5)"
RING_SECTION_LAST = r"void\s+cpu_job_signal_ring_shutdown_for_testing\s*\([^)]*\)\s*\{"
L0_ENTRY_POINTS = ("zeEventPoolCreate", "zeEventPoolDestroy", "zeEventCreate", "zeEventDestroy", "zeEventHostSignal",
                   "zeEventHostReset")
SIGNAL_NATIVE_SIG = r"void\s+signal_native_locked\s*\(\s*uint32_t\s+id\s*\)\s*noexcept\s*\{"
# A host wait, or a status query (on an in-flight event from a profiling queue one busy-waits), by any spelling.
HOST_WAITS = ((r"\bwait(?:_and_throw|_for|_until)?\s*\(", "a host wait"),
              (r"\bget_info\s*<", "a status query"),
              (r"\bzeEventHostSynchronize\b", "a host synchronize"),
              (r"\bzeEventQueryStatus\b", "a status query"))

NATIVE_IF = ("if (may_signal && group[k]->device_signal && "
             "ggml_sycl::cpu_job_signal_wrap(group[k]->device_signal, signal_mark, &device_wait)) {")


def check_flush(backend: str) -> None:
    code = blank_comments(backend)
    body, off = body_of(code, FLUSH_SIG, "A")
    f = flat(body)
    # A: the native branch.
    i = f.find(NATIVE_IF)
    if i < 0:
        fail("A", "the flush does not wrap a member's device event under `may_signal && group[k]->device_signal`")
    native, _ = block_at(f, i + len(NATIVE_IF) - 1, "A")
    for needle, what in (("joins.push_back(std::move(device_wait));", "the wrapper is not the member's dependency"),
                         ("continue;", "the natively waited member falls through to the host-task join"),
                         ("moe_hostpath_waits_device_signal();",
                          "the natively waited member is not counted for devsig=")):
        if needle not in native:
            fail("A", what)
    if "moe_cpu_job_join_on_queue" in native:
        fail("A", "the native branch also joins the job with a host task")
    j = f.find("moe_cpu_job_join_on_queue(")
    if j < 0 or j < i:
        fail("A", "the host-task join is not taken after the native branch")
    # B: the flush asks the predicate for the slot's stream.
    if "const bool may_signal = moe_cpu_job_device_signal_may(*slot.stream);" not in f:
        fail("B", "the flush does not ask moe_cpu_job_device_signal_may for the slot's stream")
    # D: retention and the empty flush.
    if ("if (native[k]) { ggml_sycl::retain_handles_until_event(std::move(job_buffers), ws.events.back()); } "
            "else { ggml_sycl::retain_handles_until_event(std::move(job_buffers), joins[k]); }") not in f:
        fail("D", "a natively waited member's buffers are not retained until the flush's last event")
    empty = ("if (ws.events.empty()) { if (n_native == 0) { ws.events.insert(ws.events.end(), joins.begin(), "
             "joins.end()); if (moe_cpu_job_join_may_leave_stream(*slot.stream)) { "
             "slot.stream->ext_oneapi_submit_barrier(joins); } } else { "
             "ws.events.push_back(slot.stream->ext_oneapi_submit_barrier(joins)); } }")
    if empty not in f:
        fail("D", "an empty flush with a natively waited member does not keep its barrier's event instead of the joins")
    if f.count("ws.events.insert(") != 1:
        fail("D", "the flush inserts its dependencies into ws.events outside the host-task-only empty flush")
    if "g_cpu_scatter_last_flush.events = ws.events;" not in f or \
            "pending_cpu_scatter_retire(*group[k], ws.events, signal_mark);" not in f:
        fail("D", "the last flush or the retired slots do not take the flush's own events and mark")
    # F: no wrapper outlives the flush.
    r = f.find("ggml_sycl::retain_handles_until_event(std::move(job_buffers), joins[k]);")
    c = f.find("joins.clear();", r if r >= 0 else 0)
    if r < 0 or c < 0 or f.find("ws.release_handles();") > c:
        fail("F", "the flush does not clear its dependency list after the retention")
    if not re.search(r"for \(int k = 0; k < n_group; \+\+k\) \{ sycl::event device_wait; if \(may_signal", f):
        fail("F", "the wrap's out-parameter is not a local of the member loop")


def check_predicate(backend: str) -> None:
    code = blank_comments(backend)
    body, _ = body_of(code, MAY_SIG, "B")
    want = "return moe_cpu_job_device_signal_enabled() && !moe_cpu_job_stream_recording(stream);"
    rec, _ = body_of(code, REC_SIG, "B")
    rec_want = ("return ggml_sycl_graph_recording_active() || "
                "stream.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;")
    if flat(body).strip() != want or flat(rec).strip() != rec_want:
        fail("B", "moe_cpu_job_device_signal_may is not the knob and the two recording tests "
                  "(moe_cpu_job_stream_recording: the backend's flag or the stream's state)")
    body, _ = body_of(code, ACQ_SIG, "B")
    if flat(body).strip() != ("if (moe_cpu_job_device_signal_may(stream)) { "
                              "(void) ggml_sycl::unified_cache_cpu_job_signal_acquire(stream, job, ref); }"):
        fail("B", "the acquire helper asks the cache without asking moe_cpu_job_device_signal_may first")
    takes = [m.start() for m in word_matches(r"unified_cache_cpu_job_signal_acquire\s*\(", code)]
    if len(takes) != 1:
        fail("B", f"{len(takes)} sites in ggml-sycl.cpp take a ring entry; want only the acquire helper")
    calls = word_matches(r"moe_cpu_job_device_signal_acquire\s*\(\s*\*stream\s*,\s*&job_signal\s*,", code)
    if len(calls) != 3:
        fail("B", f"{len(calls)} producers acquire a device event through the helper; want the three pool producers")
    for m in calls:
        tail = flat(code[m.end() : m.end() + 900])
        sb = tail.find("submit_batch(")
        if sb < 0 or "std::move(job_signal))" not in tail[sb:]:
            fail("B", "a producer that takes a device event does not hand its job half to submit_batch")
        if "std::async" in tail[:sb]:
            fail("B", "a std::async producer takes a device event")


def check_worker(pool: str, signal_hpp: str) -> None:
    code = blank_comments(pool)
    body, off = body_of(code, SUBMIT_SIG, "C")
    m = re.search(r"work_queue_\.push\(\s*\[[^\]]*\]\s*\(\s*\)\s*mutable\s*\{", body)
    if not m:
        fail("C", "submit_batch pushes no mutable worker lambda")
    lam, _ = block_at(body, m.end() - 1, "C")
    lf = flat(lam).strip()
    if not lf.startswith("try { cpu_job_signal job_signal;"):
        fail("C", "the worker lambda's body is not one try block that first declares the job half")
    t, _ = block_at(lf, lf.index("{"), "C")
    k = t.find("ggml_sycl_cpu_expert_mul_mat_batched(")
    s = t.find("job_signal.signal();")
    p = t.find("promise->set_value();")
    if k < 0 or s < 0 or p < 0 or not (k < s < p):
        fail("C", "the job half is not signalled after the batched kernel and before the promise is set")
    after = lf[lf.index("{") + len(t) + 2 :].strip()
    if after != "catch (const ggml_sycl_fallback_error &) { throw; } catch (...) { throw; }":
        fail("C", "the worker's try is not followed by exactly the two rethrowing handlers")
    h = flat(blank_comments(signal_hpp))
    if "~cpu_job_signal() { signal(); }" not in h:
        fail("C", "the job half's destructor does not signal")
    if not re.search(r"cpu_job_signal & operator=\(cpu_job_signal && o\) noexcept "
                     r"\{ if \(this != &o\) \{ signal\(\);", h):
        fail("C", "the job half's move assignment does not signal the entry it drops")


def check_recycle(backend: str, sources: dict) -> None:
    code = blank_comments(backend)
    sites = [m.start() for m in word_matches(r"cpu_job_signal_recycle\s*\(", code)]
    if len(sites) != 1:
        fail("E", f"{len(sites)} calls of cpu_job_signal_recycle in ggml-sycl.cpp; want one")
    for name, src in sources.items():
        # Blanking only removes text, so a source without the name cannot call it.
        if "cpu_job_signal_recycle" in src and re.search(r"\bcpu_job_signal_recycle\s*\(", blank_comments(src)):
            fail("E", f"{name} recycles ring entries")
    body, off = body_of(code, WAIT_SLOT_SIG, "E")
    if not (off <= sites[0] < off + len(body)):
        fail("E", "cpu_job_signal_recycle is not called in pending_cpu_scatter_wait_prev_events(slot)")
    w = re.search(r"sycl::event::wait\s*\(\s*events\s*\)\s*;", body)
    if not w or sites[0] - off < w.end():
        fail("E", "the slot's events are not waited before its mark is recycled")
    heads = enclosing_headers(body, sites[0] - off)
    if not heads or heads[0] != "if (!events.empty())":
        fail("E", "the mark is recycled outside `if (!events.empty())`, without a wait")
    pair = [m.start() for m in word_matches(r"pending_cpu_scatter_wait_prev_events\s*\(\s*\)\s*;", code)]
    tg, tg_off = body_of(code, TG_FLUSH_SIG, "E")
    if not pair or any(not (tg_off <= s < tg_off + len(tg)) for s in pair):
        fail("E", "the boundary waits of both slots are called outside ggml_sycl_cpu_tg_flush_pending")


def declarations_only(text: str) -> str:
    """What a class declares outside its function bodies and parameter lists."""
    text = strip_strings(text)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{[^{}]*\}", ";", text)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\([^()]*\)", " ", text)
    return text


def check_ring(cache_cpp: str) -> None:
    code = blank_comments(cache_cpp)
    cls, coff = body_of(code, RING_CLASS_SIG, "F")
    if re.search(r"\bsycl::event\b", declarations_only(cls)):
        fail("F", "the ring declares a sycl::event member, bare or in a container: a wrapper that would outlive its "
                  "flush")
    m = re.search(RING_WRAP_SIG, cls)
    if not m:
        fail("F", "cpu_job_signal_ring has no wrap(id, ...) member")
    wrap, _ = block_at(cls, cls.index("{", m.end() - 1), "F")
    wf = flat(wrap)
    # wrap copies the entry's native event under mutex_ and makes the wrapper from that copy with mutex_ released
    # (llama.cpp-42zj); test-sycl-cpu-job-input-wait-source.py W8 pins where the lock is.
    if "native = events_[id];" not in wf or not re.search(
            r"ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>\( \{ native, "
            r"sycl::ext::oneapi::level_zero::ownership::keep \}, \*ctx_\);", wf):
        fail("F", "wrap does not make a fresh wrapper with ownership::keep for each use")
    if "*out = std::move(ev);" not in wf:
        fail("F", "wrap does not hand the fresh wrapper to its caller")


def ring_section(cache_cpp: str) -> str:
    """unified-cache.cpp from the ring's section comment to the testing hooks' #endif, comments blanked."""
    start = cache_cpp.find(RING_SECTION_START)
    m = re.search(RING_SECTION_LAST, cache_cpp[start:]) if start >= 0 else None
    end = cache_cpp.find("\n#endif", start + m.end()) if m else -1
    if start < 0 or end < 0:
        fail("I", "unified-cache.cpp has no ring section ending at the testing hooks' #endif")
    return blank_comments(cache_cpp)[start:end]


def check_no_host_wait(backend: str, cache_cpp: str) -> None:
    code = blank_comments(backend)
    body, _ = body_of(code, FLUSH_SIG, "I")
    body = strip_strings(body)
    ring = ring_section(cache_cpp)
    for where, text in (("the flush", body), ("the ring section", strip_strings(ring))):
        for pattern, what in HOST_WAITS:
            m = re.search(pattern, text)
            if m:
                fail("I", f"{where} makes {what} (`{m.group(0).strip()}`); the device waits a job's event itself")
    m = re.search(r"\.\s*get\s*\(\s*\)", body)
    if m:
        fail("I", "the flush gets a value (`.get()`), which waits for a job on this thread")
    names = sorted(re.findall(r"\bdlsym\s*\(\s*\w+\s*,\s*\"(\w+)\"\s*\)", ring))
    if names != sorted(L0_ENTRY_POINTS):
        fail("I", f"the ring resolves {names}; want exactly its six entry points")
    flat_ring = strip_strings(ring)
    m = re.search(r"\bze[A-Z]\w*\s*\(", flat_ring)
    if m:
        fail("I", f"the ring calls a Level Zero function directly (`{m.group(0)}`), outside its resolved entry points")
    sites = [m.start() for m in re.finditer(r"\bhost_signal\s*\(", flat_ring)]
    sn = re.search(SIGNAL_NATIVE_SIG, flat_ring)
    if not sn:
        fail("I", "the ring has no signal_native_locked(id)")
    sbody, soff = block_at(flat_ring, flat_ring.index("{", sn.end() - 1), "I")
    if len(sites) != 1 or not (soff <= sites[0] < soff + len(sbody)):
        fail("I", f"{len(sites)} host_signal calls in the ring; want only the one in signal_native_locked")
    if not flat(sbody).strip().startswith("std::atomic_thread_fence(std::memory_order_release); "
                                          "const ze_result_t r = l0_->host_signal(events_[id]);"):
        fail("I", "signal_native_locked does not fence the job's writes (release) before the host signal")


STOP_WORKERS_SIG = r"void\s+CpuExpertPool::stop_workers\s*\(\s*\)\s*\{"


def check_signal_all(cache_cpp: str, common_cpp: str, pool: str) -> None:
    code = blank_comments(cache_cpp)
    m = re.search(r"static\s+void\s+cpu_job_signal_abort_callback\s*\([^)]*\)\s*\{", code)
    if not m:
        fail("G", "no abort callback for the rings")
    body, _ = block_at(code, code.index("{", m.end() - 1), "G")
    bf = flat(body).strip()
    if not bf.startswith("cpu_job_signal_signal_all();") or not re.search(r"\bprevious\s*\(\s*message\s*\)", bf):
        fail("G", "the abort callback does not signal every ring first, before it calls the previous callback")
    cc = blank_comments(common_cpp)
    e = re.search(WATCHDOG_EXIT, cc)
    sa = cc.find("ggml_sycl::cpu_job_signal_signal_all();")
    x = cc.find("std::_Exit(1);", e.end()) if e else -1
    if not e or sa < 0 or sa > e.start() or x < 0 or "}" in cc[sa:x]:
        fail("G", "the SYCL watchdog does not signal every ring before it frees host allocations and exits")
    sw, _ = body_of(blank_comments(pool), STOP_WORKERS_SIG, "G")
    if re.search(r"\bcpu_job_signal_signal_all\s*\(|\.signal\s*\(", sw):
        fail("G", "stop_workers signals a job's device event while a detached worker may still write its outputs")


def check_knob(backend: str, sources: dict, doc: str) -> None:
    code = blank_comments(backend)
    body, _ = body_of(code, KNOB_SIG, "H")
    if not re.search(r"static const bool enabled = \[\] \{ const char \* env = std::getenv\(\"" + KNOB +
                     r"\"\); return env == nullptr \|\| std::atoi\(env\) != 0; \}\(\); return enabled;", flat(body)):
        fail("H", f"{KNOB} is not read once into a function-local static, default on")
    n = sum(len(re.findall(r"getenv\s*\(\s*\"" + KNOB + "\"", blank_comments(s)))
            for s in sources.values() if KNOB in s)
    n += len(re.findall(r"getenv\s*\(\s*\"" + KNOB + "\"", code))
    if n != 1:
        fail("H", f"{n} getenv calls of {KNOB}; want one")
    rows = [ln for ln in doc.splitlines() if ln.startswith(f"| `{KNOB}=0`")]
    if len(rows) != 1 or "| ON |" not in rows[0] or "`devsig=" not in rows[0] or "`atoi`" not in rows[0]:
        fail("H", f"the catalog has no single {KNOB}=0 row with its default, devsig= and the atoi parse")
    cbody, _ = body_of(code, COUNT_SIG, "H")
    if flat(cbody).strip() != ("if (!moe_hostpath_waits_enabled()) { return; } "
                               "g_moe_hostpath_waits.joins_device_signal += 1;"):
        fail("H", "a natively waited job is counted with the trace off")
    if not re.search(r'"staged_sites=%s devsig=%llu joinq=%d\\n"', backend) or \
            "(unsigned long long) w.joins_device_signal, joinq);" not in flat(code) or \
            not re.search(r"w\.joins_device_signal\s*=\s*0;", code):
        fail("H", "the [MOE-HOSTPATH-WAITS] line does not print devsig= from the token's count, reset with it")


SYNC_SIG = r"static\s+void\s+ggml_backend_sycl_synchronize\s*\(\s*ggml_backend_t\s+backend\s*\)\s*\{"
GATES_SLEEP_SIG = r"static\s+ggml_sycl_cpu_job_gate_sleep\s+ggml_sycl_cpu_job_gates_sleep\s*\(\s*sycl::queue\s*&\s*stream\s*\)\s*\{"
GATES_CLEAR_SIG = (r"static\s+void\s+ggml_sycl_cpu_job_gates_clear\s*\(\s*const\s+ggml_sycl_cpu_job_gate_sleep\s*&\s*sleep"
                   r"\s*\)\s*\{")
RECYCLE_SIG = r"void\s+cpu_job_signal_recycle\s*\(\s*cpu_job_signal_mark\s*&\s*mark\s*\)\s*noexcept\s*\{"
CACHE_GATES_SIG = r"std::shared_ptr<cpu_job_gate_list>\s+unified_cache_cpu_job_gates\s*\(\s*sycl::queue\s*&\s*stream\s*\)\s*\{"
GATE_LIST_SIG = r"class\s+cpu_job_gate_list\s*\{"
DRAIN_SIG = r"void\s+retained_handle_drain_loop\s*\(\s*\)\s*\{"
GATE_POLLS = ((r"\.\s*get\s*\(\s*\)", "takes a value (`.get()`)"),
              (r"\bwait_for\s*\(", "polls (`wait_for`)"),
              (r"\bwait_until\s*\(", "polls (`wait_until`)"))
LOCKS = r"\b(?:lock_guard|unique_lock|scoped_lock|retained_store_lock)\b"
# Where a gate is waited, or about to be, no lock may be held at all, so any lock form counts: a guard of any kind,
# a raw lock(), lock_shared() or try_lock() call through . or ->, or std::lock().
ANY_LOCKS = (r"\b(?:lock_guard|unique_lock|scoped_lock|shared_lock|retained_store_lock)\b"
             r"|(?:\.|->)\s*(?:lock|lock_shared|try_lock)\s*\("
             r"|\bstd\s*::\s*lock\s*\(")
WAIT_TIMER_SIG = r"struct\s+retained_drain_wait_timer\s*\{"
CENSUS_ENABLE_SIG = r"void\s+retained_handle_drain_census_enable\s*\(\s*\)\s*noexcept\s*\{"
CENSUS_TAKE_SIG = r"retained_handle_drain_census\s+retained_handle_drain_census_take\s*\(\s*\)\s*noexcept\s*\{"
WAITS_ENABLED_SIG = r"static\s+bool\s+moe_hostpath_waits_enabled\s*\(\s*\)\s*\{"
DRAIN_COUNT_SIG = r"static\s+void\s+moe_hostpath_waits_drain\s*\(\s*size_t\s+gate_waits\s*,[^)]*\)\s*\{"
DRAIN_COUNT_BODY = ("const moe_hostpath_clock::time_point t_waited = moe_hostpath_wait_begin(); "
                    "if (!moe_hostpath_waits_enabled()) { return; } moe_hostpath_wait_census & w = g_moe_hostpath_waits; "
                    "w.drain_gate_waits += gate_waits; "
                    "w.drain_gate_sleep_us += std::chrono::duration<double, std::micro>(t_slept - t_begin).count(); "
                    "w.drain_driver_us += std::chrono::duration<double, std::micro>(t_waited - t_slept).count();")
DRAIN_LOOKUP = ("if (const std::shared_ptr<ggml_sycl::cpu_job_gate_list> gates = "
                "ggml_sycl::cpu_job_signal_gates(slot.prev_scatter.signal_mark)) {")
DRAIN_SEQ = ("const moe_hostpath_clock::time_point t_begin = moe_hostpath_wait_begin(); size_t gate_waits = 0; " +
             DRAIN_LOOKUP + " gate_waits = gates->wait_upto(slot.prev_scatter.signal_mark.gate_serial); } "
             "const moe_hostpath_clock::time_point t_slept = moe_hostpath_wait_begin(); sycl::event::wait(events); "
             "moe_hostpath_waits_drain(gate_waits, t_begin, t_slept); events.clear(); "
             "ggml_sycl::cpu_job_signal_recycle(slot.prev_scatter.signal_mark);")
DRAIN_PRINT = ('at += snprintf(line + at, sizeof(line) - at, " %s=%llu/%.0fus", names[c], '
               '(unsigned long long) w.count[c], w.us[c]); '
               'if (c == MOE_WAIT_B7 && at < (int) sizeof(line)) { at += snprintf(line + at, sizeof(line) - at, '
               '" drain_gate_waits=%llu drain_gate_sleep_us=%.0f drain_driver_us=%.0f", (unsigned long long) '
               'w.drain_gate_waits, w.drain_gate_sleep_us, w.drain_driver_us); }')
REAPER_COUNTERS = ("records", "gates", "gate_sleep_ns", "driver_ns")
GATE_SLEEP_STMT = "const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep(*stream);"
READBACK_SIG = r"static\s+void\s+moe_hostpath_waits_readback\s*\(\s*int\s+layer\s*,\s*bool\s+async[^)]*\)\s*\{"
QUEUE_WAIT_SIG = r"static\s+void\s+ggml_sycl_trace_queue_wait\s*\(\s*queue_ptr\s+q\s*,[^)]*\)\s*\{"
QUEUE_CLEAR_SIG = (r"static\s+void\s+ggml_sycl_cpu_job_gates_clear_queue_wait\s*\(\s*const\s+"
                   r"ggml_sycl_cpu_job_gate_sleep\s*&\s*sleep\s*\)\s*\{")
QUEUE_CLEAR_BODY = ("const moe_hostpath_clock::time_point t_waited = moe_hostpath_wait_begin(); "
                    "if (sleep.gates && sleep.newest != 0) { sleep.gates->clear_upto(sleep.newest); } "
                    "if (!moe_hostpath_waits_enabled()) { return; } "
                    "moe_hostpath_wait_census & w = g_moe_hostpath_waits; "
                    "w.queue_gate_waits += sleep.waited; "
                    "w.queue_gate_sleep_us += std::chrono::duration<double, std::micro>(sleep.t_slept - sleep.t_begin)"
                    ".count(); "
                    "w.queue_driver_us += std::chrono::duration<double, std::micro>(t_waited - sleep.t_slept).count(); "
                    "moe_hostpath_wait_end(MOE_WAIT_B9, sleep.t_begin);")
QUEUE_SEQ = ("const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep(*q); q->wait(); "
             "ggml_sycl_cpu_job_gates_clear_queue_wait(gate_sleep);")
QUEUE_PRINT = ('if (c == MOE_WAIT_B9 && at < (int) sizeof(line)) { at += snprintf(line + at, sizeof(line) - at, '
               '" queue_gate_waits=%llu queue_gate_sleep_us=%.0f queue_driver_us=%.0f", (unsigned long long) '
               'w.queue_gate_waits, w.queue_gate_sleep_us, w.queue_driver_us); }')


def no_gate_polls(text: str, where: str) -> None:
    for pattern, what in GATE_POLLS:
        m = re.search(pattern, strip_strings(text))
        if m:
            fail("J4", f"{where} {what} where it may only sleep on a gate")


def check_gates(s: dict) -> None:
    code = blank_comments(s["backend"])
    # J1: the flush pushes the native member's future as its gate.
    body, _ = body_of(code, FLUSH_SIG, "J1")
    f = flat(body)
    i = f.find(NATIVE_IF)
    native = block_at(f, i + len(NATIVE_IF) - 1, "J1")[0] if i >= 0 else ""
    want = ("ggml_sycl::cpu_job_signal_gate_push(signal_mark, group[k]->future.share()); "
            "joins.push_back(std::move(device_wait)); continue;")
    if want not in native:
        fail("J1", "the natively waited member does not push its shared future to the ring's gates before its wrapper "
                   "becomes its dependency")
    if len(word_matches(r"cpu_job_signal_gate_push\s*\(", code)) != 1:
        fail("J1", "cpu_job_signal_gate_push is called outside the flush's native branch")
    # J2: the synchronize.
    sync, _ = body_of(code, SYNC_SIG, "J2")
    sf = flat(sync)
    flush_at = sf.find("ggml_sycl_cpu_tg_flush_pending();")
    sleep_at = sf.find(GATE_SLEEP_STMT)
    waits = [sf.find("sycl_ctx->last_graph_event->wait_and_throw()"), sf.find("stream->wait_and_throw()")]
    if sleep_at < 0 or flush_at < 0 or not (flush_at < sleep_at) or any(w < 0 or w < sleep_at for w in waits):
        fail("J2", "the synchronize does not sleep on the gates after its CPU-expert flush and before both stream "
                   "waits")
    qb, qoff = body_of(code, QUEUE_WAIT_SIG, "J2")
    _, soff = body_of(code, SYNC_SIG, "J2")
    rest = code[:soff] + code[soff + len(sync):]
    rest = rest.replace(qb, "", 1)
    if len(word_matches(r"ggml_sycl_cpu_job_gates_sleep\s*\(", rest)) != 1 or \
            len(word_matches(r"ggml_sycl_cpu_job_gates_sleep\s*\(", sync)) != 1:
        fail("J2", "ggml_sycl_cpu_job_gates_sleep is called outside the synchronize and the graph code's queue wait")
    sraw = sync.find("const ggml_sycl_cpu_job_gate_sleep gate_sleep")
    heads = enclosing_headers(sync, sraw) if sraw >= 0 else []
    if heads[:1] != ["try"]:
        fail("J2", "the synchronize's gate sleep is under a condition")
    if not re.search(r"CHECK_TRY_ERROR\(stream->wait_and_throw\(\)\); ggml_sycl_cpu_job_gates_clear\(gate_sleep\);", sf):
        fail("J2", "the statement after the synchronize's stream wait does not clear the gates it slept on")
    gs, _ = body_of(code, GATES_SLEEP_SIG, "J2")
    if "sleep.gates = ggml_sycl::unified_cache_cpu_job_gates(stream);" not in flat(gs) or \
            "sleep.waited = sleep.gates->wait_all(&sleep.newest);" not in flat(gs):
        fail("J2", "the gate sleep does not wait every gate of its device's list")
    cc = blank_comments(s["cache_cpp"])
    cg, _ = body_of(cc, CACHE_GATES_SIG, "J2")
    if "std::shared_ptr<cpu_job_signal_ring> ring = cache->existing_cpu_job_signal_ring();" not in flat(cg) or \
            re.search(r"\bget_cpu_job_signal_ring\s*\(", cg):
        fail("J2", "the synchronize's gate lookup may make a ring: unified_cache_cpu_job_gates does not read the "
                   "device's existing ring only")
    # J3: the clears.
    gc, _ = body_of(code, GATES_CLEAR_SIG, "J3")
    gcf = flat(gc)
    c = gcf.find("if (sleep.gates && sleep.newest != 0) { sleep.gates->clear_upto(sleep.newest); }")
    t = gcf.find("if (!moe_hostpath_waits_enabled()) { return; }")
    if c < 0 or t < 0 or c > t:
        fail("J3", "the synchronize does not clear what it slept on before the trace check, so an untraced run would "
                   "never clear")
    wb, _ = body_of(code, WAIT_SLOT_SIG, "J3")
    wf = flat(wb).strip()
    look = wf.find(DRAIN_LOOKUP)
    g = wf.find("gates->wait_upto(slot.prev_scatter.signal_mark.gate_serial);")
    e = wf.find("sycl::event::wait(events);")
    lraw = wb.find("if (const std::shared_ptr<ggml_sycl::cpu_job_gate_list> gates")
    lheads = enclosing_headers(wb, lraw) if lraw >= 0 else []
    if look < 0 or g < look or e < 0 or g > e or lheads[:1] != ["if (!events.empty())"] or \
            not wf.startswith("std::vector<sycl::event> & events = slot.prev_scatter.scatter_events; "
                              "if (!events.empty()) { moe_hostpath_wait_timer wait_timer(MOE_WAIT_B7); "):
        fail("J3", "the boundary does not sleep on its mark's gates, inside its B7 timer, before its event wait")
    rb, _ = body_of(cc, RECYCLE_SIG, "J3")
    if flat(rb).strip() != ("if (mark) { (void) mark.ring->recycle(mark.serial); "
                            "mark.ring->gates().clear_upto(mark.gate_serial); } mark = cpu_job_signal_mark{};"):
        fail("J3", "cpu_job_signal_recycle does not clear the ring's gates up to the mark's gate serial")
    cls, _ = body_of(cc, RING_CLASS_SIG, "J3")
    for name in ("shutdown", "abandon"):
        m = re.search(r"\n    void\s+" + name + r"\s*\(\s*\)\s*noexcept\s*\{", cls)
        b = block_at(cls, cls.index("{", m.end() - 1), "J3")[0] if m else ""
        if not flat(b).strip().endswith("live_ = false; gates_.clear();"):
            fail("J3", f"the ring's {name} does not clear its gates")
    # J4: nothing queries, polls or takes a gate, and no lock is held across a gate wait.
    hpp = blank_comments(s["signal_hpp"])
    gl, _ = body_of(hpp, GATE_LIST_SIG, "J4")
    no_gate_polls(gl, "the gate list")
    for where, text in (("the synchronize's gate sleep", gs), ("the synchronize's gate clear", gc),
                        ("the boundary wait", wb)):
        no_gate_polls(text, where)
    if len(re.findall(r"\.\s*wait\s*\(\s*\)", strip_strings(gl))) != 1 or \
            "static size_t sleep_on(const std::vector<std::shared_future<void>> & futures) { for (const " \
            "std::shared_future<void> & future : futures) { future.wait(); } return futures.size(); }" not in flat(gl):
        fail("J4", "the gate list waits elsewhere than the future.wait() of its static sleep_on")
    if "size_t wait_all(uint64_t * newest) const { return sleep_on(snapshot(UINT64_MAX, newest)); }" not in flat(gl) or \
            "size_t wait_upto(uint64_t upto) const { uint64_t newest = 0; return sleep_on(snapshot(upto, &newest)); }" \
            not in flat(gl):
        fail("J4", "a gate list wait is not sleep_on() over a snapshot whose lock is released before the wait")
    pre = sf[:sleep_at] if sleep_at >= 0 else sf
    if re.search(ANY_LOCKS, strip_strings(pre)):
        fail("J4", "the synchronize takes a lock before its gate sleep")
    if re.search(ANY_LOCKS, strip_strings(gs)):
        fail("J4", "the gate sleep takes a lock it would hold across the gates' wait")
    if re.search(LOCKS, cg):
        fail("J4", "unified_cache_cpu_job_gates holds a lock it would return under")
    mem = blank_comments(s["mem_cpp"])
    dl, _ = body_of(mem, DRAIN_SIG, "J4")
    no_gate_polls(dl, "the reaper loop")
    dlf = flat(dl)
    gw = ("retained_drain_wait_timer wait_timer; "
          "for (const std::shared_future<void> & gate : record.gates) { if (gate.valid()) { gate.wait(); } } "
          "wait_timer.slept(record.gates.size()); record.event.wait_and_throw();")
    gpos = dl.find("for (const std::shared_future<void> & gate : record.gates)")
    # The loop body is the only other level: the header scan stops at the `;` inside `for (;;)`.
    if gpos < 0 or not dl.lstrip().startswith("for (;;) {") or enclosing_headers(dl, gpos) != ["try", ")"]:
        fail("J4", "the reaper's gate wait is not directly in its try block, outside every store lock")
    # J5 (reaper half).
    if not re.search(r"\} watchdog_guard; " + re.escape(gw), dlf):
        fail("J5", "the reaper does not sleep on a record's gates inside the watchdog guard, right before the event wait")
    # B8: the census.
    rb_body, _ = body_of(code, r"static\s+void\s+moe_hostpath_waits_readback\s*\(\s*int\s+layer\s*,\s*bool\s+async[^)]*\)\s*\{",
                         "B8")
    rbf = flat(rb_body)
    if '"B7", "B8"' not in rbf:
        fail("B8", "the [MOE-HOSTPATH-WAITS] line does not name B8 after B7")
    m = re.search(r'GGML_LOG_WARN\(\s*((?:"[^"]*"\s*)+)', rbf)
    fmt = "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""
    if "total=%llu/%.0fus gate_waits=%llu gate_sleep_us=%.0f driver_us=%.0f reaper=" not in fmt or \
            "total_us, (unsigned long long) w.gate_waits, w.gate_sleep_us, w.driver_us," not in rbf:
        fail("B8", "the [MOE-HOSTPATH-WAITS] line does not print gate_waits=, gate_sleep_us= and driver_us= after total=")
    for field in ("gate_waits = 0;", "gate_sleep_us = 0.0;", "driver_us = 0.0;"):
        if not re.search(r"\bw\." + re.escape(field).replace("\\ ", r"\s*"), rbf):
            fail("B8", f"w.{field.split()[0]} is not reset with the token's counts")
    trace = [ln for ln in s["doc"].splitlines() if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1")]
    knob = [ln for ln in s["doc"].splitlines() if ln.startswith(f"| `{KNOB}=0`")]
    if len(trace) != 1 or not all(f in trace[0] for f in ("`B8`", "`gate_waits=`", "`gate_sleep_us=`", "`driver_us=`")):
        fail("B8", "the catalog's trace row does not name B8 and its gate_waits=, gate_sleep_us= and driver_us= split")
    if len(knob) != 1 or "gate" not in knob[0]:
        fail("B8", f"the catalog's {KNOB}=0 row does not say that the host waits sleep on the jobs' gates")
    gsf = flat(gs).strip()
    if not gsf.startswith("ggml_sycl_cpu_job_gate_sleep sleep; sleep.t_begin = moe_hostpath_wait_begin(); "
                          "sleep.gates = ") or \
            not gsf.endswith("} sleep.t_slept = moe_hostpath_wait_begin(); return sleep;"):
        fail("B8", "the synchronize's gate sleep is not stamped before its lookup and after its sleep")
    if not gcf.strip().startswith("const moe_hostpath_clock::time_point t_waited = moe_hostpath_wait_begin(); "
                                  "if (sleep.gates && sleep.newest != 0) { sleep.gates->clear_upto(sleep.newest); }") or \
            "w.driver_us += std::chrono::duration<double, std::micro>(t_waited - sleep.t_slept).count();" not in gcf:
        fail("B8", "driver_us is not the stream wait alone: the clear does not stamp its end first, before clear_upto, "
                   "and count it from the sleep's end")
    # B7D: the graph-boundary drain's split of B7.
    dc, _ = body_of(code, DRAIN_COUNT_SIG, "B7D")
    if flat(dc).strip() != DRAIN_COUNT_BODY:
        fail("B7D", "moe_hostpath_waits_drain does not stamp the event wait's end first and add the drain's gate count, "
                    "sleep and event wait, under the trace only")
    if DRAIN_SEQ not in wf or len(word_matches(r"moe_hostpath_waits_drain\s*\(", code)) != 2:
        fail("B7D", "the drain does not stamp its start before the gate lookup and its sleep's end before the event "
                    "wait, and count once, right after the event wait and before the recycle")
    if DRAIN_PRINT not in rbf:
        fail("B7D", "the [MOE-HOSTPATH-WAITS] line does not print drain_gate_waits=, drain_gate_sleep_us= and "
                    "drain_driver_us= right after B7=")
    for field in ("drain_gate_waits = 0;", "drain_gate_sleep_us = 0.0;", "drain_driver_us = 0.0;"):
        if not re.search(r"\bw\." + re.escape(field).replace("\\ ", r"\s*"), rbf):
            fail("B7D", f"w.{field.split()[0]} is not reset with the token's counts")
    if not all(f in trace[0] for f in ("`drain_gate_waits=`", "`drain_gate_sleep_us=`", "`drain_driver_us=`")):
        fail("B7D", "the catalog's trace row does not name the drain's drain_gate_waits=, drain_gate_sleep_us= and "
                    "drain_driver_us= split")
    # R: the reaper's census.
    if "record.event.wait_and_throw(); wait_timer.waited();" not in dlf:
        fail("R", "the reaper does not count a record's wait right after its event wait returns")
    tb, _ = body_of(mem, WAIT_TIMER_SIG, "R")
    tf = flat(tb)
    if not re.search(r"bool on = g_retained_drain_census_on\.load\(std::memory_order_relaxed\); "
                     r"clock::time_point t_begin = on \? clock::now\(\) : clock::time_point\{\};", tf) or \
            "void slept(size_t n) { if (on) { t_slept = clock::now(); gates = n; } }" not in tf or \
            "void waited() const { if (!on) { return; } const clock::time_point t_end = clock::now();" not in tf:
        fail("R", "the reaper's timer reads a clock while the census is off, or does not stamp the sleep's end")
    adds = re.findall(r"\bg_retained_drain_census_(\w+)\.fetch_add\s*\(", mem)
    w_at = tb.find("void waited() const")
    in_waited = re.findall(r"\bg_retained_drain_census_(\w+)\.fetch_add\s*\(", tb[w_at:]) if w_at >= 0 else []
    if sorted(adds) != sorted(REAPER_COUNTERS) or sorted(in_waited) != sorted(REAPER_COUNTERS):
        fail("R", "the reaper's counters are added to elsewhere than once each in the timer's waited()")
    eb, _ = body_of(mem, CENSUS_ENABLE_SIG, "R")
    kb, _ = body_of(mem, CENSUS_TAKE_SIG, "R")
    if flat(eb).strip() != "g_retained_drain_census_on.store(true, std::memory_order_relaxed);" or \
            any(f"g_retained_drain_census_{c}.exchange(0, std::memory_order_relaxed)" not in flat(kb)
                for c in REAPER_COUNTERS):
        fail("R", "the census is not turned on by its enable, or its take does not zero every counter")
    en, _ = body_of(code, WAITS_ENABLED_SIG, "R")
    if "const bool on = env && std::atoi(env) != 0; if (on) { ggml_sycl::retained_handle_drain_census_enable(); } " \
            "return on;" not in flat(en) or len(word_matches(r"retained_handle_drain_census_enable\s*\(", code)) != 1:
        fail("R", "the trace does not turn the reaper's census on exactly when it is on, and only there")
    take = "const ggml_sycl::retained_handle_drain_census rd = ggml_sycl::retained_handle_drain_census_take();"
    if rbf.count(take) != 1 or len(word_matches(r"retained_handle_drain_census_take\s*\(", code)) != 1 or \
            "driver_us=%.0f reaper=%llu/%.0fus reaper_gate_waits=%llu reaper_gate_sleep_us=%.0f reaper_driver_us=%.0f " \
            "ready_queries=" not in fmt or \
            "w.driver_us, (unsigned long long) rd.records, rd.gate_sleep_us + rd.driver_us, " \
            "(unsigned long long) rd.gate_waits, rd.gate_sleep_us, rd.driver_us, (unsigned long long) w.ready_queries," \
            not in rbf:
        fail("R", "the [MOE-HOSTPATH-WAITS] line does not take the reaper's census once and print reaper=, "
                  "reaper_gate_waits=, reaper_gate_sleep_us= and reaper_driver_us= after driver_us=")
    if not all(f in trace[0] for f in ("`reaper=`", "`reaper_gate_waits=`", "`reaper_gate_sleep_us=`",
                                       "`reaper_driver_us=`")):
        fail("R", "the catalog's trace row does not name reaper= and its reaper_gate_waits=, reaper_gate_sleep_us= "
                  "and reaper_driver_us= split")


def check_queue_waits(s: dict) -> None:
    code = blank_comments(s["backend"])
    qb, _ = body_of(code, QUEUE_WAIT_SIG, "B9")
    qf = flat(qb)
    if re.search(ANY_LOCKS, strip_strings(qb)):
        fail("J4", "the graph code's queue wait takes a lock it would sleep under")
    if qf.count(QUEUE_SEQ) != 2 or len(re.findall(r"\bq->wait\(\);", qf)) != 2:
        fail("B9", "a queue wait of the graph code does not sleep on the gates right before it and clear them right "
                   "after it")
    if not re.search(r"if \(!enabled\) \{ " + re.escape(QUEUE_SEQ) + r" return; \}", qf):
        fail("B9", "the untraced queue wait does not sleep on the gates first")
    begin = qf.find('fprintf(stderr, "[SYCL-WAIT] begin reason=')
    traced = ("fflush(stderr); const auto t0 = std::chrono::high_resolution_clock::now(); " + QUEUE_SEQ +
              " const auto t1 = std::chrono::high_resolution_clock::now();")
    if begin < 0 or qf.find(traced) < begin:
        fail("B9", "the traced queue wait does not sleep after its begin line and inside its wait_ms")
    qc, _ = body_of(code, QUEUE_CLEAR_SIG, "B9")
    no_gate_polls(qc, "the queue wait's gate clear")
    if flat(qc).strip() != QUEUE_CLEAR_BODY:
        fail("B9", "the queue wait's clear does not stamp the wait's end first, clear what the sleep waited before "
                   "the trace check, and count the sleep and the queue wait under B9, under the trace only")
    if len(word_matches(r"ggml_sycl_cpu_job_gates_clear_queue_wait\s*\(", code)) != 3:
        fail("B9", "ggml_sycl_cpu_job_gates_clear_queue_wait is called outside the graph code's queue waits")
    hpp = flat(blank_comments(s["hostpath_hpp"]))
    if "MOE_WAIT_B8, MOE_WAIT_B9, MOE_WAIT_COUNT," not in hpp:
        fail("B9", "moe_hostpath_wait_class has no B9 right after B8")
    rb_body, _ = body_of(code, READBACK_SIG, "B9")
    rbf = flat(rb_body)
    if '"B8", "B9" };' not in rbf:
        fail("B9", "the [MOE-HOSTPATH-WAITS] line does not name B9 after B8")
    if DRAIN_PRINT + " " + QUEUE_PRINT not in rbf:
        fail("B9", "the [MOE-HOSTPATH-WAITS] line does not print queue_gate_waits=, queue_gate_sleep_us= and "
                   "queue_driver_us= right after B9=")
    for field in ("queue_gate_waits = 0;", "queue_gate_sleep_us = 0.0;", "queue_driver_us = 0.0;"):
        if not re.search(r"\bw\." + re.escape(field).replace("\\ ", r"\s*"), rbf):
            fail("B9", f"w.{field.split()[0]} is not reset with the token's counts")
    trace = [ln for ln in s["doc"].splitlines() if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1")]
    knob = [ln for ln in s["doc"].splitlines() if ln.startswith(f"| `{KNOB}=0`")]
    if len(trace) != 1 or not all(f in trace[0] for f in ("B1..B9", "`B9`", "`queue_gate_waits=`",
                                                          "`queue_gate_sleep_us=`", "`queue_driver_us=`")):
        fail("B9", "the catalog's trace row does not name B9 and its queue_gate_waits=, queue_gate_sleep_us= and "
                   "queue_driver_us= split")
    if len(knob) != 1 or "`ggml_sycl_trace_queue_wait`" not in knob[0]:
        fail("B9", f"the catalog's {KNOB}=0 row does not say that the graph code's queue waits sleep on the gates")


def read_sources(root: Path) -> dict:
    s = {}
    for key, path in (("backend", BACKEND), ("pool", POOL), ("signal_hpp", SIGNAL_HPP), ("cache_cpp", CACHE_CPP),
                      ("common_cpp", COMMON_CPP), ("doc", DOC), ("mem_cpp", MEM_CPP),
                      ("hostpath_hpp", HOSTPATH_HPP)):
        p = root / path.relative_to(ROOT)
        s[key] = p.read_text(encoding="utf-8") if p.exists() else ""
    others = {}
    for p in sorted((root / SYCL.relative_to(ROOT)).glob("*.[ch]pp")):
        if p.name != "ggml-sycl.cpp":
            others[p.name] = p.read_text(encoding="utf-8", errors="replace")
    s["others"] = others
    return s


def check(s: dict) -> None:
    others = dict(s["others"])
    others.update({"unified-cache.cpp": s["cache_cpp"], "cpu-expert-pool.cpp": s["pool"], "common.cpp": s["common_cpp"],
                   "cpu-job-signal.hpp": s["signal_hpp"]})
    check_flush(s["backend"])
    check_predicate(s["backend"])
    check_worker(s["pool"], s["signal_hpp"])
    recyclers = {k: v for k, v in others.items() if k not in ("unified-cache.cpp", "unified-cache.hpp")}
    check_recycle(s["backend"], recyclers)
    check_ring(s["cache_cpp"])
    check_signal_all(s["cache_cpp"], s["common_cpp"], s["pool"])
    getenvs = {k: v for k, v in others.items()}
    check_knob(s["backend"], getenvs, s["doc"])
    check_no_host_wait(s["backend"], s["cache_cpp"])
    check_gates(s)
    check_queue_waits(s)


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    mutants = [
        ("A", "backend", "the native branch also joins with a host task",
         sub(r"joins\.push_back\(std::move\(device_wait\)\);\n(\s*)continue;",
             r"joins.push_back(std::move(device_wait));\n\1joins.push_back(moe_cpu_job_join_on_queue(*slot.stream, "
             r"group[k]->future.share(), group[k]->job_stamps));\n\1continue;")),
        ("A", "backend", "the native member falls through to the join",
         sub(r"(device_wait\)\);\n)\s*continue;\n", r"\1")),
        ("A", "backend", "the native member is not counted", sub(r"\n\s*moe_hostpath_waits_device_signal\(\);", "")),
        ("A", "backend", "no native branch", sub(r"may_signal && group\[k\]->device_signal &&", "false &&")),
        ("B", "backend", "the flush asks for another stream",
         sub(r"moe_cpu_job_device_signal_may\(\*slot\.stream\);", "moe_cpu_job_device_signal_enabled();")),
        # The recording test is moe_cpu_job_stream_recording, which the join and input-marker predicates share.
        ("B", "backend", "native under the backend's recording",
         sub(r"return ggml_sycl_graph_recording_active\(\) \|\|\s*stream\.", "return stream.")),
        ("B", "backend", "native while the stream records",
         sub(r"return ggml_sycl_graph_recording_active\(\) \|\|\s*"
             r"stream\.ext_oneapi_get_state\(\) == sycl::ext::oneapi::experimental::queue_state::recording;",
             "return ggml_sycl_graph_recording_active();")),
        ("B", "backend", "the device-signal predicate skips the recording test",
         sub(r"return moe_cpu_job_device_signal_enabled\(\) && !moe_cpu_job_stream_recording\(stream\);",
             "return moe_cpu_job_device_signal_enabled();")),
        ("B", "backend", "acquire without the predicate",
         sub(r"if \(moe_cpu_job_device_signal_may\(stream\)\) \{", "if (moe_cpu_job_device_signal_enabled()) {")),
        ("B", "backend", "a producer keeps its job half",
         sub(r"moe_cpu_job_trace_request\(job_trace, src0\), std::move\(job_signal\)\);",
             "moe_cpu_job_trace_request(job_trace, src0));")),
        ("B", "backend", "a second site takes an entry",
         sub(r"(static void moe_cpu_job_device_signal_acquire\()",
             r"static void zz_take(sycl::queue & q, ggml_sycl::cpu_job_signal * j, ggml_sycl::cpu_job_signal_ref * r) "
             r"{ (void) ggml_sycl::unified_cache_cpu_job_signal_acquire(q, j, r); }\n\1")),
        ("C", "pool", "signal after set_value",
         sub(r"(\s*)job_signal\.signal\(\);\n((?:.|\n)*?)promise->set_value\(\);",
             r"\2promise->set_value();\1job_signal.signal();")),
        ("C", "pool", "signal before the kernel",
         sub(r"(\n\s*)(ggml_sycl_cpu_expert_mul_mat_batched\()", r"\1job_signal.signal();\1\2")),
        ("C", "pool", "catch-all dropped", sub(r" catch \(\.\.\.\) \{\n\s*throw;\n\s*\}", "")),
        ("C", "pool", "catch-all swallows",
         sub(r"catch \(\.\.\.\) \{\n(\s*)throw;", r"catch (...) {\n\1promise->set_value();")),
        ("C", "pool", "job half declared outside the try",
         sub(r"try \{\n(\s*)// A local of this try block[^\n]*\n\s*cpu_job_signal job_signal;",
             r"cpu_job_signal job_signal;\n\1try {")),
        ("C", "signal_hpp", "destructor does not signal",
         sub(r"~cpu_job_signal\(\) \{ signal\(\); \}", "~cpu_job_signal() {}")),
        ("D", "backend", "native buffers retained until the wrapper",
         sub(r"retain_handles_until_event\(std::move\(job_buffers\), ws\.events\.back\(\)\);",
             "retain_handles_until_event(std::move(job_buffers), joins[k]);")),
        ("D", "backend", "empty native flush keeps the wrappers",
         sub(r"ws\.events\.push_back\(slot\.stream->ext_oneapi_submit_barrier\(joins\)\);",
             "ws.events.insert(ws.events.end(), joins.begin(), joins.end());")),
        ("D", "backend", "the mark is not retired with the slots",
         sub(r"pending_cpu_scatter_retire\(\*group\[k\], ws\.events, signal_mark\);",
             "pending_cpu_scatter_retire(*group[k], ws.events, {});")),
        ("E", "backend", "recycle before the wait",
         sub(r"(\s*)sycl::event::wait\(events\);\n(\s*)moe_hostpath_waits_drain\(gate_waits, t_begin, t_slept\);"
             r"\n(\s*)events\.clear\(\);\n(\s*)ggml_sycl::cpu_job_signal_recycle\(slot\.prev_scatter\.signal_mark\);",
             r"\1ggml_sycl::cpu_job_signal_recycle(slot.prev_scatter.signal_mark);\1sycl::event::wait(events);"
             r"\n\2moe_hostpath_waits_drain(gate_waits, t_begin, t_slept);\n\3events.clear();")),
        ("E", "backend", "recycle with nothing waited",
         sub(r"\n(\s*)slot\.prev_scatter\.signal_mark = \{\};\n\}",
             r"\n\1ggml_sycl::cpu_job_signal_recycle(slot.prev_scatter.signal_mark);\n}")),
        ("E", "backend", "slot pair waited at a flush",
         sub(r"(static void flush_pending_cpu_scatter\(\) \{\n)", r"\1    pending_cpu_scatter_wait_prev_events();\n")),
        ("F", "cache_cpp", "a cached wrapper",
         sub(r"\*out    = std::move\(ev\);", "*out    = cached_;")),
        ("F", "cache_cpp", "the ring keeps a wrapper",
         sub(r"(class cpu_job_signal_ring \{\n  public:\n)", r"\1    sycl::event cached_;\n")),
        ("F", "cache_cpp", "the ring keeps a vector of wrappers",
         sub(r"(class cpu_job_signal_ring \{\n  public:\n)", r"\1    std::vector<sycl::event> cache_;\n")),
        ("F", "cache_cpp", "the ring keeps an optional wrapper",
         sub(r"(class cpu_job_signal_ring \{\n  public:\n)", r"\1    std::optional<sycl::event> cached_;\n")),
        ("F", "backend", "the dependency list outlives the flush",
         sub(r"\n\s*joins\.clear\(\);\n(\s*auto t2)", r"\n\1")),
        ("G", "cache_cpp", "abort callback without signal-all",
         sub(r"(static void cpu_job_signal_abort_callback\([^)]*\) \{\n)(\s*)cpu_job_signal_signal_all\(\);",
             r"\1\2;")),
        ("G", "cache_cpp", "abort callback signals last",
         sub(r"(static void cpu_job_signal_abort_callback\([^)]*\) \{\n)(\s*)cpu_job_signal_signal_all\(\);\n"
             r"((?:.|\n)*?\n    \}\n)",
             r"\1\3\2cpu_job_signal_signal_all();\n")),
        ("G", "common_cpp", "watchdog exits without signal-all",
         sub(r"ggml_sycl::cpu_job_signal_signal_all\(\);", ";")),
        ("G", "pool", "detach signals every ring",
         sub(r"(\n(\s*))t\.detach\(\);", r"\1ggml_sycl::cpu_job_signal_signal_all();\1t.detach();")),
        ("H", "backend", "knob default off",
         sub(r'return env == nullptr \|\| std::atoi\(env\) != 0;'
             r'(\s*\}\(\);\s*return enabled;\s*\}\s*\n\s*// Whether the scatter)',
             r'return env != nullptr && std::atoi(env) != 0;\1')),
        ("H", "backend", "knob read per call",
         sub(r'static const bool enabled = \[\] \{\n(\s*)const char \* env = std::getenv\("' + KNOB + r'"\);',
             r'const bool enabled = [] {\n\1const char * env = std::getenv("' + KNOB + r'");')),
        ("H", "doc", "catalog row lost", sub(r"\| `" + KNOB + r"=0`", "| `" + KNOB + "_OFF`")),
        ("H", "backend", "devsig counted untraced",
         sub(r"(static void moe_hostpath_waits_device_signal\(\) \{\n)\s*if \(!moe_hostpath_waits_enabled\(\)\) "
             r"\{\n\s*return;\n\s*\}\n", r"\1")),
        ("H", "backend", "line loses devsig", sub(r" devsig=%llu", "")),
        ("I", "backend", "the native branch host-waits its wrapper",
         sub(r"\n(\s*)joins\.push_back\(std::move\(device_wait\)\);",
             r"\n\1device_wait.wait();\n\1joins.push_back(std::move(device_wait));")),
        ("I", "backend", "the flush host-waits its dependencies",
         sub(r"\n(\s*)auto t1 = hrc::now\(\);", r"\n\1sycl::event::wait(joins);\n\1auto t1 = hrc::now();")),
        ("I", "backend", "the native branch queries its wrapper's status",
         sub(r"\n(\s*)joins\.push_back\(std::move\(device_wait\)\);",
             r"\n\1(void) device_wait.get_info<sycl::info::event::command_execution_status>();"
             r"\n\1joins.push_back(std::move(device_wait));")),
        ("I", "backend", "the flush gets a job's future",
         sub(r"\n(\s*)auto t1 = hrc::now\(\);", r"\n\1group[0]->future.share().get();\n\1auto t1 = hrc::now();")),
        ("I", "cache_cpp", "wrap host-synchronizes the event",
         sub(r"\n(\s*)sycl::event ev;\n((?:\s*[^\n]*;\n){0,2}\s*try \{\n\s*ev = sycl::make_event)",
             r"\n\1(void) zeEventHostSynchronize(native, UINT64_MAX);\n\1sycl::event ev;\n\2")),
        ("I", "cache_cpp", "the ring resolves a status query",
         sub(r'(\n(\s*)f\.host_reset    = reinterpret_cast<decltype\(&zeEventHostReset\)>\(dlsym\(lib, '
             r'"zeEventHostReset"\)\);)',
             r'\1\n\2(void) dlsym(lib, "zeEventQueryStatus");')),
        ("I", "cache_cpp", "the signal is not fenced",
         sub(r"\n\s*std::atomic_thread_fence\(std::memory_order_release\);\n(\s*const ze_result_t r = "
             r"l0_->host_signal)", r"\n\1")),
        ("I", "cache_cpp", "a signal bypasses the fence",
         sub(r"(if \(book_\.signal\(id, generation\) && live_\) \{\n\s*)signal_native_locked\(id\);",
             r"\1(void) l0_->host_signal(events_[id]);")),
        # The gates (J1-J5, B8, B7D).
        ("J1", "backend", "the native member pushes no gate",
         sub(r"\n\s*ggml_sycl::cpu_job_signal_gate_push\(signal_mark, group\[k\]->future\.share\(\)\);", "")),
        ("J1", "backend", "the gate push moves after the wrapper is kept",
         sub(r"(\n\s*ggml_sycl::cpu_job_signal_gate_push\(signal_mark, group\[k\]->future\.share\(\)\);)"
             r"(\n\s*joins\.push_back\(std::move\(device_wait\)\);)", r"\2\1")),
        ("J1", "backend", "a host-task-joined member pushes a gate",
         sub(r"(\n(\s*)joins\.push_back\(moe_cpu_job_join_on_queue\()",
             r"\n\2ggml_sycl::cpu_job_signal_gate_push(signal_mark, group[k]->future.share());\1")),
        ("J2", "backend", "the synchronize does not sleep",
         sub(r"ggml_sycl_cpu_job_gates_sleep\(\*stream\);", "ggml_sycl_cpu_job_gate_sleep{};")),
        ("J2", "backend", "the synchronize sleeps after its stream wait",
         lambda t: re.sub(r"(CHECK_TRY_ERROR\(stream->wait_and_throw\(\)\);)",
                          r"\1\n        const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep(*stream);",
                          re.sub(r"\n\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*stream\);",
                                 "", t, count=1), count=1)),
        ("J2", "backend", "the synchronize sleeps before its flush",
         lambda t: re.sub(r"(\n(\s*)if \(!ggml_sycl_cpu_tg_synchronize_flush_skipped_for_test\(\)\) \{)",
                          r"\n\2const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep(*stream);\1",
                          re.sub(r"\n\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*stream\);",
                                 "", t, count=1), count=1)),
        ("J2", "backend", "the synchronize sleeps under a condition",
         sub(r"(\n(\s*))(const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*stream\);)",
             r"\1if (sycl_ctx->last_graph_event_deferred_decode) {\1    \3\1}")),
        ("J2", "backend", "the synchronize keeps its gates", sub(r"\n\s*ggml_sycl_cpu_job_gates_clear\(gate_sleep\);", "")),
        ("J2", "backend", "the sleep waits part of the list",
         sub(r"sleep\.waited = sleep\.gates->wait_all\(&sleep\.newest\);", "sleep.waited = sleep.gates->wait_upto(1);")),
        ("J3", "backend", "an untraced synchronize never clears",
         sub(r"(\n    if \(sleep\.gates && sleep\.newest != 0\) \{\n        sleep\.gates->clear_upto\(sleep\.newest\);\n    \})"
             r"(\n    if \(!moe_hostpath_waits_enabled\(\)\) \{\n        return;\n    \})", r"\2\1")),
        ("J3", "backend", "the boundary sleeps after its event wait",
         sub(r"(\n\s*if \(const std::shared_ptr<ggml_sycl::cpu_job_gate_list> gates =\n\s*"
             r"ggml_sycl::cpu_job_signal_gates\(slot\.prev_scatter\.signal_mark\)\) \{\n\s*"
             r"gate_waits = gates->wait_upto\(slot\.prev_scatter\.signal_mark\.gate_serial\);\n\s*\})"
             r"(\n\s*const moe_hostpath_clock::time_point t_slept = moe_hostpath_wait_begin\(\);)"
             r"(\n\s*sycl::event::wait\(events\);)", r"\2\3\1")),
        ("J3", "backend", "the boundary does not sleep",
         sub(r"gate_waits = gates->wait_upto\(slot\.prev_scatter\.signal_mark\.gate_serial\);", "(void) gates;")),
        ("J3", "backend", "the boundary sleeps outside its non-empty branch",
         lambda t: re.sub(r"(\n(\s*)std::vector<sycl::event> & events = slot\.prev_scatter\.scatter_events;)",
                          r"\1\n\2if (const std::shared_ptr<ggml_sycl::cpu_job_gate_list> gates =\n\2        "
                          r"ggml_sycl::cpu_job_signal_gates(slot.prev_scatter.signal_mark)) {\n\2    "
                          r"(void) gates->wait_upto(slot.prev_scatter.signal_mark.gate_serial);\n\2}",
                          re.sub(r"\n\s*if \(const std::shared_ptr<ggml_sycl::cpu_job_gate_list> gates =\n\s*"
                                 r"ggml_sycl::cpu_job_signal_gates\(slot\.prev_scatter\.signal_mark\)\) \{\n\s*"
                                 r"gate_waits = gates->wait_upto\(slot\.prev_scatter\.signal_mark\.gate_serial\);"
                                 r"\n\s*\}", "", t, count=1), count=1)),
        ("J2", "backend", "a third site sleeps on the gates",
         lambda t: t + ("\nstatic void extra_gate_sleep(sycl::queue & q) {\n"
                        "    (void) ggml_sycl_cpu_job_gates_sleep(q);\n}\n")),
        ("J2", "cache_cpp", "the synchronize's gate lookup makes a ring",
         sub(r"ring = cache->existing_cpu_job_signal_ring\(\);", "ring = cache->get_cpu_job_signal_ring();")),
        ("J3", "cache_cpp", "the recycle keeps the gates",
         sub(r"\n\s*mark\.ring->gates\(\)\.clear_upto\(mark\.gate_serial\);", "")),
        ("J3", "cache_cpp", "the shutdown keeps the gates",
         sub(r"(live_ = false;)\n        gates_\.clear\(\);\n(    \}\n\n    // The SYCL runtime is gone)", r"\1\n\2")),
        ("J4", "signal_hpp", "the gate list waits under its lock",
         sub(r"size_t wait_all\(uint64_t \* newest\) const \{ return sleep_on\(snapshot\(UINT64_MAX, newest\)\); \}",
             "size_t wait_all(uint64_t * newest) const {\n        std::lock_guard<std::mutex> lock(mutex_);\n"
             "        *newest = 0;\n        for (const gate & g : gates_) {\n            g.future.wait();\n"
             "            *newest = g.serial;\n        }\n        return gates_.size();\n    }")),
        ("J4", "signal_hpp", "the gate list polls",
         sub(r"            future\.wait\(\);\n",
             "            while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {\n            }\n")),
        ("J4", "mem_cpp", "the reaper takes a gate's value", sub(r"gate\.wait\(\);", "gate.get();")),
        ("J4", "mem_cpp", "the reaper polls a gate",
         sub(r"gate\.wait\(\);", "(void) gate.wait_for(std::chrono::milliseconds(1));")),
        ("J4", "mem_cpp", "the reaper sleeps under the store lock",
         sub(r"(\n(\s*))(for \(const std::shared_future<void> & gate : record\.gates\) \{\n\s*if \(gate\.valid\(\)\) "
             r"\{\n\s*gate\.wait\(\);\n\s*\}\n\s*\})",
             r"\1{\1    retained_store_lock gate_lock(*g_retained_handles_state);\1    \3\1}")),
        ("J4", "backend", "the synchronize sleeps under a lock",
         sub(r"(\n(\s*))(const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*stream\);)",
             r"\1std::lock_guard<std::mutex> held(g_execution_backend_binding_mutex);\1\3")),
        ("J5", "mem_cpp", "the reaper waits the event before the gates",
         sub(r"(\n\s*for \(const std::shared_future<void> & gate : record\.gates\) \{\n\s*if \(gate\.valid\(\)\) \{\n"
             r"\s*gate\.wait\(\);\n\s*\}\n\s*\})(\n\s*wait_timer\.slept\(record\.gates\.size\(\)\);)"
             r"(\n\s*record\.event\.wait_and_throw\(\);)", r"\3\1\2")),
        ("J5", "mem_cpp", "the reaper sleeps outside the watchdog guard",
         lambda t: re.sub(r"(\n(\s*)struct event_wait_watchdog_guard \{)",
                          r"\n\2for (const std::shared_future<void> & gate : record.gates) {\n\2    if (gate.valid()) {\n"
                          r"\2        gate.wait();\n\2    }\n\2}\1",
                          re.sub(r"\n\s*for \(const std::shared_future<void> & gate : record\.gates\) \{\n\s*"
                                 r"if \(gate\.valid\(\)\) \{\n\s*gate\.wait\(\);\n\s*\}\n\s*\}", "", t, count=1),
                          count=1)),
        ("B8", "backend", "the line loses gate_sleep_us", sub(r'"gate_sleep_us=%\.0f ', '"')),
        ("B8", "backend", "B8 is not named", sub(r'"B7",(\s+)"B8",(\s+)"B9"(\s*)\};', r'"B7",\1"B7",\2"B9"\3};')),
        ("B8", "backend", "driver_us is not reset", sub(r"\n\s*w\.driver_us\s*= 0\.0;", "")),
        ("B8", "backend", "the synchronize stamps its stream wait after the clear",
         sub(r"(\n    const moe_hostpath_clock::time_point t_waited = moe_hostpath_wait_begin\(\);)"
             r"(\n    if \(sleep\.gates && sleep\.newest != 0\) \{\n        sleep\.gates->clear_upto\(sleep\.newest\);"
             r"\n    \})", r"\2\1")),
        ("B8", "backend", "the synchronize stamps its sleep's end before the sleep",
         sub(r"(\n    if \(sleep\.gates\) \{\n        sleep\.waited = sleep\.gates->wait_all\(&sleep\.newest\);\n    \})"
             r"(\n    sleep\.t_slept = moe_hostpath_wait_begin\(\);)", r"\2\1")),
        ("B7D", "backend", "the drain stamps its sleep's end after the event wait",
         sub(r"(\n\s*const moe_hostpath_clock::time_point t_slept = moe_hostpath_wait_begin\(\);)"
             r"(\n\s*sycl::event::wait\(events\);)", r"\2\1")),
        ("B7D", "backend", "the drain counts after the recycle",
         sub(r"(\n\s*moe_hostpath_waits_drain\(gate_waits, t_begin, t_slept\);)(\n\s*events\.clear\(\);"
             r"\n\s*ggml_sycl::cpu_job_signal_recycle\(slot\.prev_scatter\.signal_mark\);)", r"\2\1")),
        ("B7D", "backend", "the drain's event wait is timed from its start",
         sub(r"\(t_waited - t_slept\)", "(t_waited - t_begin)")),
        ("B7D", "backend", "the drain drops its gate count",
         sub(r"gate_waits = gates->wait_upto\(", "(void) gates->wait_upto(")),
        ("B7D", "backend", "the drain counts with the trace off",
         sub(r"(moe_hostpath_waits_drain\([^{]*\{\n\s*const moe_hostpath_clock::time_point t_waited = "
             r"moe_hostpath_wait_begin\(\);)\n\s*if \(!moe_hostpath_waits_enabled\(\)\) \{\n\s*return;\n\s*\}", r"\1")),
        ("B7D", "backend", "the drain's split is printed after B8",
         sub(r"c == MOE_WAIT_B7 && at", "c == MOE_WAIT_B8 && at")),
        ("B7D", "backend", "the line loses drain_driver_us", sub(r' drain_driver_us=%\.0f"', '"')),
        ("B7D", "backend", "drain_gate_sleep_us is not reset", sub(r"\n\s*w\.drain_gate_sleep_us\s*= 0\.0;", "")),
        ("B7D", "doc", "the trace row loses drain_gate_sleep_us",
         lambda t: "\n".join(ln.replace("`drain_gate_sleep_us=`", "`drain_sleep_us=`")
                             if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1") else ln for ln in t.split("\n"))),
        ("B8", "doc", "the trace row loses driver_us",
         lambda t: "\n".join(ln.replace("`driver_us=`", "`driver=`") if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1")
                             else ln for ln in t.split("\n"))),
        # The graph code's queue waits (B9).
        ("B9", "backend", "the untraced queue wait does not sleep",
         sub(r"(if \(!enabled\) \{\n)\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n(\s*q->wait\(\);\n)\s*"
             r"ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\);\n", r"\1\2")),
        ("B9", "backend", "the traced queue wait does not sleep",
         sub(r"(high_resolution_clock::now\(\);\n)\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n(\s*q->wait\(\);\n)\s*"
             r"ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\);\n", r"\1\2")),
        ("B9", "backend", "the untraced queue wait sleeps after it waits",
         sub(r"(if \(!enabled\) \{\n)(\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n)(\s*q->wait\(\);\n)", r"\1\3\2")),
        ("B9", "backend", "the untraced queue wait keeps its gates",
         sub(r"(if \(!enabled\) \{\n\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n\s*q->wait\(\);\n)\s*"
             r"ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\);\n", r"\1        (void) gate_sleep;\n")),
        ("B9", "backend", "the trace's wait_ms leaves out the sleep",
         sub(r"(\n\s*const auto\s+t0\s+= std::chrono::high_resolution_clock::now\(\);)"
             r"(\n\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*q\);)",
             r"\2\1")),
        ("B9", "backend", "the traced queue wait sleeps before its begin line",
         lambda t: re.sub(r"(\n(\s*)fprintf\(stderr, \"\[SYCL-WAIT\] begin reason=)",
                          r"\n\2const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep(*q);\1",
                          re.sub(r"(high_resolution_clock::now\(\);)\n\s*const ggml_sycl_cpu_job_gate_sleep "
                                 r"gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*q\);", r"\1", t, count=1),
                          count=1)),
        ("B9", "backend", "the queue clear stamps its wait after the clear",
         sub(r"(ggml_sycl_cpu_job_gates_clear_queue_wait\(const ggml_sycl_cpu_job_gate_sleep & sleep\) \{)"
             r"(\n    const moe_hostpath_clock::time_point t_waited = moe_hostpath_wait_begin\(\);)"
             r"(\n    if \(sleep\.gates && sleep\.newest != 0\) \{\n        sleep\.gates->clear_upto\(sleep\.newest\);"
             r"\n    \})", r"\1\3\2")),
        ("B9", "backend", "the queue waits count under B8",
         sub(r"moe_hostpath_wait_end\(MOE_WAIT_B9, sleep\.t_begin\);",
             "moe_hostpath_wait_end(MOE_WAIT_B8, sleep.t_begin);")),
        ("B9", "backend", "the queue waits count into the synchronize's split",
         sub(r"w\.queue_driver_us \+=", "w.driver_us +=")),
        ("B9", "backend", "a fourth site clears the queue gates",
         lambda t: t + "\nstatic void extra_gate_clear(const ggml_sycl_cpu_job_gate_sleep & s) {\n"
                       "    ggml_sycl_cpu_job_gates_clear_queue_wait(s);\n}\n"),
        ("J4", "backend", "the queue wait sleeps under a lock",
         sub(r"(if \(!enabled\) \{\n)(\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);)",
             r"\1        std::lock_guard<std::mutex> held(g_execution_backend_binding_mutex);\n\2")),
        ("J4", "backend", "the queue wait takes a raw lock around its traced sleep",
         sub(r"(\n(\s*)const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*q\);\n"
             r"\s*q->wait\(\);\n\s*ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\);\n)"
             r"(\s*const auto\s+t1)",
             r"\n\2g_execution_backend_binding_mutex.lock();\1\2g_execution_backend_binding_mutex.unlock();\n\3")),
        ("J4", "backend", "the queue wait takes an arrow-form lock around its untraced sleep",
         sub(r"(if \(!enabled\) \{\n)(\s*)(const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n)",
             r"\1\2g_execution_backend_binding_mutex_ptr->lock();\n"
             r"\2\3\2g_execution_backend_binding_mutex_ptr->unlock();\n")),
        ("J4", "backend", "the queue wait takes std::lock() around its untraced sleep",
         sub(r"(if \(!enabled\) \{\n)(\s*)(const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n)",
             r"\1\2std::lock(g_a_mutex, g_b_mutex);\n\2\3\2g_a_mutex.unlock();\n\2g_b_mutex.unlock();\n")),
        ("J4", "backend", "the queue wait takes a lock_guard before its begin line",
         sub(r"(\n(\s*)fprintf\(stderr, \"\[SYCL-WAIT\] begin reason=)",
             r"\n\2std::lock_guard<std::mutex> held(g_execution_backend_binding_mutex);\1")),
        ("J4", "backend", "the queue wait takes an arrow-form lock before its begin line",
         lambda t: re.sub(r"(wait_ms=%\.3f\\n\", safe_reason, device, node_idx,\n\s*wait_ms\);\n"
                          r"\s*fflush\(stderr\);\n)",
                          r"\1    g_execution_backend_binding_mutex_ptr->unlock();\n",
                          re.sub(r"(\n(\s*)fprintf\(stderr, \"\[SYCL-WAIT\] begin reason=)",
                                 r"\n\2g_execution_backend_binding_mutex_ptr->lock();\1", t, count=1), count=1)),
        ("J4", "backend", "the gate sleep holds a lock_guard across its wait",
         sub(r"(\n    sleep\.gates   = ggml_sycl::unified_cache_cpu_job_gates\(stream\);\n)",
             r"\1    std::lock_guard<std::mutex> held(g_execution_backend_binding_mutex);\n")),
        ("J4", "backend", "the synchronize takes an arrow-form lock across its sleep",
         sub(r"(\n(\s*))(const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
             r"ggml_sycl_cpu_job_gates_sleep\(\*stream\);\n)",
             r"\1g_execution_backend_binding_mutex_ptr->lock();"
             r"\1\3\2g_execution_backend_binding_mutex_ptr->unlock();\n")),
        ("B9", "backend", "B9 is not named", sub(r'"B8",(\s+)"B9"(\s*)\};', r'"B8",\1"B8"\2};')),
        ("B9", "backend", "the queue split is printed after B8",
         sub(r"c == MOE_WAIT_B9 && at", "c == MOE_WAIT_B8 && at")),
        ("B9", "backend", "the line loses queue_driver_us", sub(r' queue_driver_us=%\.0f"', '"')),
        ("B9", "backend", "queue_gate_sleep_us is not reset", sub(r"\n\s*w\.queue_gate_sleep_us\s*= 0\.0;", "")),
        ("B9", "hostpath_hpp", "no B9 class", sub(r"\n    MOE_WAIT_B9,", "")),
        ("B9", "doc", "the trace row loses queue_gate_sleep_us",
         lambda t: "\n".join(ln.replace("`queue_gate_sleep_us=`", "`queue_sleep_us=`")
                             if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1") else ln for ln in t.split("\n"))),
        ("B9", "doc", "the trace row still counts B1..B8",
         lambda t: "\n".join(ln.replace("B1..B9", "B1..B8") if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1")
                             else ln for ln in t.split("\n"))),
        ("B9", "doc", "the knob row does not name the queue waits",
         lambda t: "\n".join(ln.replace("`ggml_sycl_trace_queue_wait`", "the queue waits")
                             if ln.startswith(f"| `{KNOB}=0`") else ln for ln in t.split("\n"))),
        ("R", "mem_cpp", "the reaper times its waits untraced",
         sub(r"bool(\s+)on(\s+)= g_retained_drain_census_on\.load\(std::memory_order_relaxed\);", "bool on = true;")),
        ("R", "mem_cpp", "the reaper never counts a wait", sub(r"\n\s*wait_timer\.waited\(\);", "")),
        ("R", "mem_cpp", "the reaper does not stamp the sleep's end",
         sub(r"\n\s*t_slept = clock::now\(\);", "")),
        ("R", "mem_cpp", "a publication counts as a reaper wait",
         sub(r"(\n(\s*)std::vector<uint64_t>(\s+)ids = owner_control_ids\(handles\);)",
             r"\1\n\2g_retained_drain_census_records.fetch_add(1, std::memory_order_relaxed);")),
        ("R", "mem_cpp", "the take does not zero the gate count",
         sub(r"g_retained_drain_census_gates\.exchange\(0, std::memory_order_relaxed\)",
             "g_retained_drain_census_gates.load(std::memory_order_relaxed)")),
        ("R", "backend", "the trace never turns the reaper's census on",
         sub(r"\n\s*if \(on\) \{\n(?:\s*//[^\n]*\n)*\s*ggml_sycl::retained_handle_drain_census_enable\(\);\n\s*\}", "")),
        ("R", "backend", "the line does not take the reaper's census",
         sub(r"= ggml_sycl::retained_handle_drain_census_take\(\);", "= ggml_sycl::retained_handle_drain_census{};")),
        ("R", "backend", "the line loses reaper_driver_us", sub(r'"reaper_driver_us=%\.0f ready_queries', '"ready_queries')),
        ("R", "doc", "the trace row loses reaper=",
         lambda t: "\n".join(ln.replace("`reaper=`", "`reaper`") if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE=1")
                             else ln for ln in t.split("\n"))),
        ("B8", "doc", "the knob row does not mention the gates",
         lambda t: "\n".join(re.sub(r"gate", "wait", ln) if ln.startswith(f"| `{KNOB}=0`") else ln
                             for ln in t.split("\n"))),
    ]
    failures, caught = [], 0
    for pin, key, name, mutate in mutants:
        m = dict(s)
        m[key] = mutate(s[key])
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
    print("PASS: the scatter waits a CPU job's own event only where the device may, every way a job ends signals it, "
          "no wrapper outlives its flush or is reset before its stream was waited, and a host wait behind a job sleeps "
          f"on its gate first {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
