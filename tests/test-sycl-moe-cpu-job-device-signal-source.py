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

THE CONTRACT.  With comments blanked:
  A. (ggml-sycl.cpp flush_pending_cpu_scatter_slot) a member whose wrap succeeds pushes
     the wrapper as its dependency, is counted for devsig=, and continues before the
     host-task join; moe_cpu_job_join_on_queue is called only after that branch.
  B. The device waits a job only when moe_cpu_job_device_signal_may() holds: the knob is
     on and neither the backend's recording flag nor the stream's own state records a
     graph.  The flush asks it for the slot's stream; the one acquire helper asks it
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
     sycl::make_event with ownership::keep; nothing the ring class declares outside its
     function bodies and parameter lists names sycl::event, so no member holds a wrapper,
     bare or in a container; the flush clears its dependency list after the retention,
     and the wrap's out-parameter is a local of the member loop.
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

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

GATE RUNTIME.  Comment blanking is memoized, and the two scans over every backend source
skip a file that does not contain the name the scan looks for (cpu_job_signal_recycle, or
the knob): blanking only removes text, so such a file cannot match either way.

RED on the tree before the change:
  FAIL [pin A]: the flush does not wrap a member's device event under
  `may_signal && group[k]->device_signal` (llama.cpp-qwn5)
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
SIGNAL_HPP = SYCL / "cpu-job-signal.hpp"
CACHE_CPP = SYCL / "unified-cache.cpp"
COMMON_CPP = SYCL / "common.cpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-qwn5)"
KNOB = "GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


# Every check blanks the same few large sources again; one check per mutant reads at most six of them.
@functools.lru_cache(maxsize=8)
def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals and offsets."""
    out = []
    i, n = 0, len(src)
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
        elif src[i] == "'" and not (i > 0 and src[i - 1].isalnum()):
            j = i + 1
            while j < n and src[j] != "'":
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def strip_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


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
    want = ("return moe_cpu_job_device_signal_enabled() && !ggml_sycl_graph_recording_active() && "
            "stream.ext_oneapi_get_state() != sycl::ext::oneapi::experimental::queue_state::recording;")
    if flat(body).strip() != want:
        fail("B", "moe_cpu_job_device_signal_may is not the knob and the two recording tests")
    body, _ = body_of(code, ACQ_SIG, "B")
    if flat(body).strip() != ("if (moe_cpu_job_device_signal_may(stream)) { "
                              "(void) ggml_sycl::unified_cache_cpu_job_signal_acquire(stream, job, ref); }"):
        fail("B", "the acquire helper asks the cache without asking moe_cpu_job_device_signal_may first")
    takes = [m.start() for m in re.finditer(r"\bunified_cache_cpu_job_signal_acquire\s*\(", code)]
    if len(takes) != 1:
        fail("B", f"{len(takes)} sites in ggml-sycl.cpp take a ring entry; want only the acquire helper")
    calls = [m for m in re.finditer(r"\bmoe_cpu_job_device_signal_acquire\s*\(\s*\*stream\s*,\s*&job_signal\s*,", code)]
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
    sites = [m.start() for m in re.finditer(r"\bcpu_job_signal_recycle\s*\(", code)]
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
    heads = enclosing_headers(code, sites[0])
    if not heads or heads[0] != "if (!events.empty())":
        fail("E", "the mark is recycled outside `if (!events.empty())`, without a wait")
    pair = [m.start() for m in re.finditer(r"\bpending_cpu_scatter_wait_prev_events\s*\(\s*\)\s*;", code)]
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
    if not re.search(r"ev = sycl::make_event<sycl::backend::ext_oneapi_level_zero>\( \{ events_\[id\], "
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


def read_sources(root: Path) -> dict:
    s = {}
    for key, path in (("backend", BACKEND), ("pool", POOL), ("signal_hpp", SIGNAL_HPP), ("cache_cpp", CACHE_CPP),
                      ("common_cpp", COMMON_CPP), ("doc", DOC)):
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
        ("B", "backend", "native under the backend's recording",
         sub(r"return moe_cpu_job_device_signal_enabled\(\) && !ggml_sycl_graph_recording_active\(\) &&",
             "return moe_cpu_job_device_signal_enabled() &&")),
        ("B", "backend", "native while the stream records",
         sub(r"(return moe_cpu_job_device_signal_enabled\(\) && !ggml_sycl_graph_recording_active\(\)) &&\n\s*"
             r"stream\.ext_oneapi_get_state\(\) != sycl::ext::oneapi::experimental::queue_state::recording;", r"\1;")),
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
         sub(r"(\s*)sycl::event::wait\(events\);\n(\s*)events\.clear\(\);\n(\s*)"
             r"ggml_sycl::cpu_job_signal_recycle\(slot\.prev_scatter\.signal_mark\);",
             r"\1ggml_sycl::cpu_job_signal_recycle(slot.prev_scatter.signal_mark);\1sycl::event::wait(events);"
             r"\n\2events.clear();")),
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
         sub(r"\n(\s*)sycl::event ev;\n(\s*)try \{\n(\s*)ev = sycl::make_event",
             r"\n\1(void) zeEventHostSynchronize(events_[id], UINT64_MAX);\n\1sycl::event ev;\n\2try {\n\3"
             r"ev = sycl::make_event")),
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
          f"and no wrapper outlives its flush or is reset before its stream was waited {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
