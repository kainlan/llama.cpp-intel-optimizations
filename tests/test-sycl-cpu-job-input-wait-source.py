#!/usr/bin/env python3
"""The CPU expert pool worker never enters the driver to wait for a job's inputs (llama.cpp-42zj).

THE DEADLOCK (capture 2 on the fix build 4e39aadb3, B70, Qwen3.8 IQ3_XXS, the first 256-token prompt graph).  The
retained-handle drain worker waited a record's event natively (mem-handle.cpp, record.event.wait_and_throw()).  libze
took a device-level mutex for that wait and held it while polling the device (libze_intel_gpu.so.1.17.39758: G at
0x9b6740 locks this+0x30 at 0x9b6ace and polls in the loop around 0x9b6ca9).  The device work behind that event
stood behind a CPU expert job's host-signalled event.  The job's prologue, on the pool's one worker, waited its input
events natively too (moe_cpu_job_wait_inputs, sycl::event::wait), reached G through the status method's task-count
branch (0x389ba0 > 0x37fd10 > G) and blocked on the same mutex (gdb: its owner was the drain worker's LWP).  The job
could not signal, so the drain worker's wait could not end.  Publish-time gating (G1-G6) cannot close this: whether a
wait takes the lock depends on the wait, and any thread may be the holder (the drain worker, the submitting thread's
synchronous copies, a runtime thread).

THE FIX.  Take the victim out of the driver.  The submitting thread arms a device-written host-USM marker on the op's
in-order stream after the job's input events, and the worker waits for that marker with host loads.  On driver 26.35
zeEventHostSignal (EventImp::hostSignal, 0x37ece0 and 0x36f8d0, slot 3 of the event vtables at 0xcccf80/0xccced0)
reaches neither G nor 0x37fd10 (a direct-call walk that does find G from hostSynchronize and from the status method),
so a job that makes no other driver call cannot join the cycle.

THE CONTRACT.  With comments and string literals blanked:
  W1. The worker's input wait is driver-free.  moe_cpu_job_wait_inputs calls only cpu_job_input_marker_wait,
      ggml_sycl_watchdog_heartbeat, moe_hostpath_waits_enabled, cpu_expert_pool_trace_stamp_inputs_ready and
      memset; the closure moe_cpu_job_prologue returns calls only moe_cpu_job_wait_inputs; neither names anything
      in sycl::.
  W2. The marker is waited once, unconditionally, at the top of moe_cpu_job_wait_inputs, before the watchdog tick,
      the inputs-ready stamp and the zeroing of the output span, which must follow the inputs.  The watchdog is
      ticked exactly once, unconditionally, after the wait (as a CPU expert batch ticks it when it finishes,
      llama.cpp-ff00) and never inside it, so a marker that never arrives fires the watchdog instead of spinning
      unseen.
  W3. The worker cannot be handed an event.  moe_cpu_job_wait_inputs takes no sycl::event and no moe_cpu_job_inputs,
      and the closure captures only marker, zero_out and zero_bytes.
  W4. The marker is armed on the submitting thread, and a refused arm is waited there.  moe_cpu_job_prologue takes
      the stream and, outside its closure, calls unified_cache_cpu_job_input_marker_arm once, as the condition of an
      `if` that first asks `!moe_cpu_job_input_marker_may(stream) ||`; that branch holds
      `sycl::event::wait(inputs.events)`, and every wait outside the closure sits in it.  moe_cpu_job_input_marker_may
      refuses while the backend or the stream records a command graph (a recorded marker would be written only at
      replay).
  W5. Every prologue is built with its stream: each moe_cpu_job_prologue call passes the op's stream first, and the
      event-chained hybrid route builds its prologue before its table copy and its device-expert kernel, so the
      in-order marker does not wait for that kernel and the CPU and GPU halves still overlap.
  W6. The pool's job lambda (cpu-expert-pool.cpp) names nothing in sycl:: and makes no wait, status query, submission
      or ze/ur call of its own: the job's one driver call is its device signal.
  W7. The marker itself (cpu-job-signal.hpp, unified-cache.cpp).  The header's wait helpers call nothing but each
      other, the pause and yield, and the acquire load; none sleeps, locks, ticks the watchdog or names sycl::.  The
      seen test reads `abandoned` before the word, both with acquire loads.  The device task depends on the job's
      inputs, then fences (release, system scope) before it stores the serial.  The arm takes the cache's marker
      mutex for the whole arm, submits before it records the new serial, and takes its word from the unified cache
      as a host-pinned allocation.  shutdown_resources ends the marker word on every branch that ends the signal
      ring, and on the normal branch only after the queues drained.
  W8. The signal ring (cpu_job_signal_ring, unified-cache.cpp) makes no driver call that may block under mutex_, the
      lock the job's own signal takes: a host wait on another thread may hold the driver's device lock behind that
      job.  wrap makes its event and recycle resets its events outside every mutex_ scope, each call counted in
      outside_calls_ (raised before it and lowered after it); the book (consume, begin_flush, recycle_begin,
      recycle_finish) and outside_calls_ change only under mutex_; recycle does not use the book's one-lock recycle;
      and shutdown destroys native objects only inside `if (outside_calls_ == 0)`.  grow_locked, which creates pools
      and events under mutex_, is outside this pin.

WHAT THIS DOES NOT PROVE.  It reads source text.  It does not show that zeEventHostSignal stays clear of the driver's
device lock on another driver (that is the disassembly above, for 26.35 only), that the device writes the marker
after the inputs complete (the in-order stream and the task's dependency do; the lead-run ring case
test-sycl-cpu-job-signal-ring measures it), or that the deadlock is gone (the lead-run Qwen3.8 stall reproducer does).
The worker's other driver contact, destructors of the weight leases it may run, is outside it.

NOT VACUOUS.  --self-test first requires the unmodified tree to pass, then runs the gate on in-memory mutants of the
real sources, each restoring a native wait, moving a call, or dropping a pin, and requires every one to fail on the
pin it names.

RED on 1bc3bbb1e (the base before this fix):
  FAIL [pin W1]: moe_cpu_job_wait_inputs runs on the CPU expert pool worker and calls inputs.events.empty,
  sycl::event::wait; the worker's input wait may call only cpu_expert_pool_trace_stamp_inputs_ready, ...
RED for W8 alone, on the commit that added W1-W7:
  FAIL [pin W8]: cpu_job_signal_ring::wrap makes its event under mutex_ ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = "(llama.cpp-42zj)"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_BLANK_TOKEN = re.compile(r'//[^\n]*|/\*.*?(?:\*/|\Z)|"(?:\\.|[^"\\])*(?:"|\Z)|(?<![^\W_])\'(?:\\.|[^\'\\])*(?:\'|\Z)', re.S)
_NOT_NEWLINE = re.compile(r"[^\n]")


def _blank_token(m: re.Match) -> str:
    t = m.group(0)
    if t[0] != "/":
        return t
    return " " * len(t) if t[1] == "/" else _NOT_NEWLINE.sub(" ", t)


def blank_comments(src: str) -> str:
    return _BLANK_TOKEN.sub(_blank_token, src)


def strip_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def block_at(code: str, brace: int, pin: str) -> tuple[str, int]:
    flat_code = strip_strings(code)
    depth = 0
    for j in range(brace, len(flat_code)):
        if flat_code[j] == "{":
            depth += 1
        elif flat_code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1 : j], brace + 1
    fail(pin, "unbalanced braces")
    return "", 0


def definition(code: str, signature: str, pin: str) -> tuple[re.Match, str]:
    found = list(re.finditer(signature, code))
    if len(found) != 1:
        fail(pin, f"{len(found)} definitions match {signature!r}, not 1")
    m = found[0]
    body, _ = block_at(code, code.index("{", m.end() - 1), pin)
    return m, body


def enclosing_brace(code: str, pos: int) -> int:
    flat_code = strip_strings(code)
    depth = 0
    for i in range(pos - 1, -1, -1):
        c = flat_code[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return i
            depth -= 1
    return -1


def matching_paren(code: str, open_pos: int) -> int:
    flat_code = strip_strings(code)
    depth = 0
    for j in range(open_pos, len(flat_code)):
        if flat_code[j] == "(":
            depth += 1
        elif flat_code[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return len(code)


_NOT_CALLS = {"if", "for", "while", "switch", "return", "sizeof", "alignof", "decltype", "catch", "static_cast",
              "reinterpret_cast", "const_cast", "dynamic_cast", "static_assert", "noexcept", "defined"}
_CALL = re.compile(r"((?:[A-Za-z_]\w*\s*(?:::|\.|->)\s*)*)([A-Za-z_]\w*)\s*\(")


def calls(body: str) -> list[tuple[str, str]]:
    """(qualified text, last name) of every call-like token in `body`, keywords and casts left out."""
    out = []
    for m in _CALL.finditer(strip_strings(body)):
        name = m.group(2)
        if name in _NOT_CALLS:
            continue
        out.append((re.sub(r"\s+", "", m.group(1) + name), name))
    return out


BACKEND_REL = Path("ggml") / "src" / "ggml-sycl" / "ggml-sycl.cpp"
POOL_REL = Path("ggml") / "src" / "ggml-sycl" / "cpu-expert-pool.cpp"
HEADER_REL = Path("ggml") / "src" / "ggml-sycl" / "cpu-job-signal.hpp"
CACHE_REL = Path("ggml") / "src" / "ggml-sycl" / "unified-cache.cpp"

WAIT_SIG = r"static\s+void\s+moe_cpu_job_wait_inputs\s*\(([^)]*)\)\s*\{"
PROLOGUE_SIG = r"static\s+std::function<void\(\)>\s+moe_cpu_job_prologue\s*\(([^)]*)\)\s*\{"
MAY_SIG = r"static\s+bool\s+moe_cpu_job_input_marker_may\s*\(([^)]*)\)\s*\{"
CHAINED_SIG = r"static\s+bool\s+ggml_sycl_moe_hybrid_decode_event_chained\s*\(([^)]*)\)\s*\{"
WAIT_ALLOWED = {"cpu_job_input_marker_wait", "ggml_sycl_watchdog_heartbeat", "moe_hostpath_waits_enabled",
                "cpu_expert_pool_trace_stamp_inputs_ready", "memset"}
CLOSURE_ALLOWED = {"moe_cpu_job_wait_inputs"}
CAPTURES_ALLOWED = {"marker", "zero_out", "zero_bytes"}
DRIVER_CALL = re.compile(r"^(?:wait|wait_and_throw|wait_for|throw_asynchronous|get_info|get_profiling_info|submit|"
                         r"single_task|parallel_for|memcpy|memset_async|fill|copy|depends_on|ext_oneapi_\w+|"
                         r"ze[A-Z]\w*|ur[A-Z]\w*|mem_copy\w*|event_complete\w*)$")
ARM = r"\bunified_cache_cpu_job_input_marker_arm\s*\("

# W7: what the header's marker helpers may call. None may sleep, lock, tick the watchdog or name sycl::.
HELPERS = {
    "cpu_job_input_marker_wait": (r"inline\s+void\s+cpu_job_input_marker_wait\s*\(([^)]*)\)\s*\{",
                                  {"cpu_job_input_marker_seen", "cpu_job_input_marker_pause", "yield"}),
    "cpu_job_input_marker_seen": (r"inline\s+bool\s+cpu_job_input_marker_seen\s*\(([^)]*)\)\s*\{",
                                  {"armed", "load", "__atomic_load_n", "cpu_job_input_marker_reached"}),
    "cpu_job_input_marker_pause": (r"inline\s+void\s+cpu_job_input_marker_pause\s*\(([^)]*)\)\s*\{",
                                   {"__builtin_ia32_pause", "yield"}),
    "cpu_job_input_marker_reached": (r"inline\s+bool\s+cpu_job_input_marker_reached\s*\(([^)]*)\)\s*\{", set()),
}
HELPER_FORBIDDEN = re.compile(r"\b(?:sleep_for|sleep_until|usleep|nanosleep|lock_guard|unique_lock|scoped_lock|"
                              r"mutex|ggml_sycl_watchdog_heartbeat|wait_for|wait_until)\b|\bsycl\s*::")
SUBMIT_SIG = r"static\s+void\s+cpu_job_input_marker_submit\s*\(([^)]*)\)\s*\{"
ARM_METHOD_SIG = r"bool\s+unified_cache::arm_cpu_job_input_marker\s*\(([^)]*)\)\s*\{"
SHUTDOWN_SIG = r"bool\s+unified_cache::shutdown_resources\s*\(([^)]*)\)\s*(?:noexcept\s*)?\{"


def read_sources(root: Path) -> dict[str, str]:
    return {"backend": blank_comments((root / BACKEND_REL).read_text()),
            "pool": blank_comments((root / POOL_REL).read_text()),
            "header": blank_comments((root / HEADER_REL).read_text()),
            "cache": blank_comments((root / CACHE_REL).read_text())}


def check_wait_inputs(s: dict[str, str]) -> None:
    code = s["backend"]
    wm, wbody = definition(code, WAIT_SIG, "W1")
    bad = sorted({q for q, n in calls(wbody) if n not in WAIT_ALLOWED})
    if bad:
        fail("W1", "moe_cpu_job_wait_inputs runs on the CPU expert pool worker and calls " + ", ".join(bad) +
                   "; the worker's input wait may call only " + ", ".join(sorted(WAIT_ALLOWED)) +
                   ", never the driver, which a host wait elsewhere may hold locked for this very job")
    if re.search(r"\bsycl\s*::", wbody):
        fail("W1", "moe_cpu_job_wait_inputs names something in sycl::; the worker's input wait is driver-free")
    pm, pbody = definition(code, PROLOGUE_SIG, "W1")
    lam = re.search(r"return\s*\[([^\]]*)\]\s*\(\s*\)\s*(?:mutable\s*)?\{", pbody)
    if not lam:
        fail("W1", "moe_cpu_job_prologue does not return a closure")
    lbody, _ = block_at(pbody, pbody.index("{", lam.end() - 1), "W1")
    bad = sorted({q for q, n in calls(lbody) if n not in CLOSURE_ALLOWED})
    if bad or re.search(r"\bsycl\s*::", lbody):
        fail("W1", "the closure moe_cpu_job_prologue returns runs on the pool worker and calls " +
                   (", ".join(bad) if bad else "something in sycl::") + "; it may call only moe_cpu_job_wait_inputs")

    # W2: once, at the top level, before the watchdog tick, the stamp and the zeroing; the tick once, right after it.
    hits = [m.start() for m in re.finditer(r"\bcpu_job_input_marker_wait\s*\(", wbody)]
    if len(hits) != 1:
        fail("W2", f"moe_cpu_job_wait_inputs waits the input marker {len(hits)} times, not once")
    if enclosing_brace(wbody, hits[0]) != -1:
        fail("W2", "moe_cpu_job_wait_inputs waits the input marker under a condition; the wait is unconditional "
                   "(an unarmed marker returns at once)")
    for later in ("ggml_sycl_watchdog_heartbeat", "cpu_expert_pool_trace_stamp_inputs_ready", "memset"):
        m = re.search(r"\b" + later + r"\s*\(", wbody)
        if m and m.start() < hits[0]:
            fail("W2", f"moe_cpu_job_wait_inputs calls {later} before the input marker is seen")
    ticks = [m.start() for m in re.finditer(r"\bggml_sycl_watchdog_heartbeat\s*\(", wbody)]
    if len(ticks) != 1 or enclosing_brace(wbody, ticks[0]) != -1:
        fail("W2", f"moe_cpu_job_wait_inputs ticks the SYCL watchdog {len(ticks)} times or under a condition; it ticks "
                   "once, unconditionally, after the marker arrives, as a CPU expert batch does when it finishes")

    # W3: no event can reach the worker.
    params = wm.group(1)
    if re.search(r"\bsycl\s*::\s*event\b|\bmoe_cpu_job_inputs\b", params):
        fail("W3", "moe_cpu_job_wait_inputs takes events (" + re.sub(r"\s+", " ", params).strip() + "); the "
                   "worker gets only the armed marker and the span to zero")
    caps = [c.strip() for c in lam.group(1).split(",") if c.strip()]
    names = {re.match(r"&?\s*(\w+)", c).group(1) if re.match(r"&?\s*\w+", c) else c for c in caps}
    if not names or not names <= CAPTURES_ALLOWED or any(c in ("=", "&") for c in caps):
        fail("W3", "the prologue's closure captures " + ", ".join(caps or ["nothing"]) + "; it may capture only " +
                   ", ".join(sorted(CAPTURES_ALLOWED)))

    # W4: armed on the submitting thread, before the closure exists; a refusal is waited here, never on the worker.
    if not re.search(r"\bsycl\s*::\s*queue\s*&\s*\w+", pm.group(1)):
        fail("W4", "moe_cpu_job_prologue does not take the op's stream, so it cannot arm the input marker on it")
    outside = pbody[: lam.start()]
    arms = [m.start() for m in re.finditer(ARM, outside)]
    if len(arms) != 1 or re.search(ARM, lbody):
        fail("W4", "moe_cpu_job_prologue arms the input marker " + str(len(arms)) +
                   " times before its closure, not once (the submitting thread arms; the worker never does)")
    arm_if = None
    for m in re.finditer(r"\bif\s*\(", outside):
        close = matching_paren(outside, m.end() - 1)
        if m.start() < arms[0] < close:
            arm_if = (m.start(), close)
    if not arm_if:
        fail("W4", "moe_cpu_job_prologue's arm is not the condition of a branch: its refusal must be waited on the "
                   "submitting thread")
    cond = outside[arm_if[0] : arm_if[1]]
    if not re.search(r"!\s*(?:ggml_sycl\s*::\s*)?unified_cache_cpu_job_input_marker_arm\s*\(", cond):
        fail("W4", "moe_cpu_job_prologue's branch is not taken on the arm's refusal")
    may = re.search(r"!\s*moe_cpu_job_input_marker_may\s*\(\s*stream\s*\)\s*\|\|", cond)
    if not may or arm_if[0] + may.start() > arms[0]:
        fail("W4", "moe_cpu_job_prologue arms the input marker without first asking moe_cpu_job_input_marker_may: a "
                   "marker recorded into a command graph is written only at a replay, and the job would wait until then")
    fallback, fstart = block_at(outside, outside.index("{", arm_if[1]), "W4")
    waits = [m.start() for m in re.finditer(r"\bsycl\s*::\s*event\s*::\s*wait\s*\(|\.\s*wait\s*\(", outside)]
    if not re.search(r"\bsycl\s*::\s*event\s*::\s*wait\s*\(\s*inputs\s*\.\s*events\s*\)", fallback):
        fail("W4", "moe_cpu_job_prologue does not wait the inputs when the arm is refused: the worker would get an "
                   "unarmed marker and read inputs that have not landed")
    for w in waits:
        if not (fstart <= w < fstart + len(fallback)):
            fail("W4", "moe_cpu_job_prologue waits on the submitting thread outside the arm's refusal branch; a host "
                       "wait there is only the fallback for a marker that was not armed")
    _, maybody = definition(code, MAY_SIG, "W4")
    if not (re.search(r"!\s*ggml_sycl_graph_recording_active\s*\(\s*\)", maybody) and
            re.search(r"ext_oneapi_get_state\s*\(\s*\)\s*!=\s*sycl::ext::oneapi::experimental::queue_state::recording",
                      maybody)):
        fail("W4", "moe_cpu_job_input_marker_may does not refuse while the backend or the stream records a graph")

    # W5: every prologue is built with its stream, and the event-chained route arms before its device half.
    sites = [m for m in re.finditer(r"\bmoe_cpu_job_prologue\s*\(", code) if not re.search(
        r"std::function<void\(\)>\s+$", code[max(0, m.start() - 40) : m.start()])]
    if len(sites) < 2:
        fail("W5", f"{len(sites)} moe_cpu_job_prologue call sites, fewer than the 2 pool routes")
    for m in sites:
        if not re.match(r"moe_cpu_job_prologue\s*\(\s*\*?\s*\w*stream\w*\s*,", code[m.start() :]):
            line = code.count("\n", 0, m.start()) + 1
            fail("W5", f"ggml-sycl.cpp:{line} builds a prologue without passing the op's stream first")
    _, cbody = definition(code, CHAINED_SIG, "W5")
    pro = re.search(r"\bmoe_cpu_job_prologue\s*\(", cbody)
    dev = re.search(r"\bmmvq_moe_batched_dispatch\s*\(", cbody)
    tab = re.search(r"\btable_h2d\s*=", cbody)
    if not pro or not dev or not tab or not (pro.start() < tab.start() < dev.start()):
        fail("W5", "the event-chained hybrid route does not arm its job's input marker before its device half: on the "
                   "in-order stream the marker would follow the device-expert kernel, and the CPU job would wait for "
                   "it instead of running alongside it")


def check_pool_lambda(s: dict[str, str]) -> None:
    code = s["pool"]
    m = re.search(r"work_queue_\.push\(\s*\[[^\]]*\]\s*\(\s*\)\s*mutable\s*\{", code, re.S)
    if not m:
        fail("W6", "no job lambda pushed onto the pool's work queue")
    body, _ = block_at(code, code.index("{", m.end() - 1), "W6")
    bad = sorted({q for q, n in calls(body) if DRIVER_CALL.match(n)})
    if bad or re.search(r"\bsycl\s*::", body):
        fail("W6", "the pool's job lambda makes driver-facing calls of its own: " +
                   (", ".join(bad) if bad else "something in sycl::") +
                   "; its one driver call is the job's device signal")


def check_helpers(s: dict[str, str]) -> None:
    hdr = s["header"]
    for name, (sig, allowed) in HELPERS.items():
        _, body = definition(hdr, sig, "W7")
        bad = sorted({q for q, n in calls(body) if n not in allowed})
        if bad:
            fail("W7", f"{name} (cpu-job-signal.hpp) runs on the pool worker and calls " + ", ".join(bad) +
                       "; it may call only " + (", ".join(sorted(allowed)) or "nothing"))
        f = HELPER_FORBIDDEN.search(body)
        if f:
            fail("W7", f"{name} (cpu-job-signal.hpp) uses {f.group(0).strip()}: the marker wait never sleeps, locks, "
                       "ticks the watchdog or enters SYCL")
    _, seen = definition(hdr, HELPERS["cpu_job_input_marker_seen"][0], "W7")
    ab = re.search(r"abandoned\s*\.\s*load\s*\(\s*std::memory_order_acquire\s*\)", seen)
    ld = re.search(r"__atomic_load_n\s*\([^;]*__ATOMIC_ACQUIRE\s*\)", seen)
    if not ab or not ld or ab.start() > ld.start():
        fail("W7", "cpu_job_input_marker_seen does not read abandoned, then the word, with acquire loads: a word whose "
                   "allocation the cache let go would be read, or the job's inputs read before the word is")

    cache = s["cache"]
    _, sub = definition(cache, SUBMIT_SIG, "W7")
    dep = re.search(r"\bdepends_on\s*\(\s*after\s*\)", sub)
    fence = re.search(r"\batomic_fence\s*\(\s*sycl::memory_order::release\s*,\s*sycl::memory_scope::system\s*\)", sub)
    store = re.search(r"\*\s*word\s*=\s*serial\s*;", sub)
    task = re.search(r"\bsingle_task\s*\(", sub)
    if not (dep and fence and store and task and task.start() < fence.start() < store.start()):
        fail("W7", "cpu_job_input_marker_submit does not depend on the job's inputs and then, in its device task, "
                   "fence (release, system scope) before it stores the serial: the host could see the word before the "
                   "inputs, or before they completed")
    _, arm = definition(cache, ARM_METHOD_SIG, "W7")
    lock = re.search(r"std::lock_guard<std::mutex>\s+\w+\s*\(\s*cpu_job_input_marker_mutex_\s*\)", arm)
    subm = re.search(r"\bcpu_job_input_marker_submit\s*\(", arm)
    took = re.search(r"\bcpu_job_input_marker_serial_\s*=\s*serial\s*;", arm)
    if not lock or not subm or not took or not (lock.start() < subm.start() < took.start()) or \
       enclosing_brace(arm, lock.start()) != -1:
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not take the next serial and submit its device task "
                   "under cpu_job_input_marker_mutex_, consuming the serial only after the submit: two arms could reach "
                   "the stream out of serial order, and the word would go backwards")
    if not (re.search(r"\bmust_host_pinned\s*=\s*true\s*;", arm) and re.search(r"\bunified_allocate_owner\s*\(", arm)):
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not take its word from the unified cache as a "
                   "host-pinned owner")
    _, shut = definition(cache, SHUTDOWN_SIG, "W7")
    rings = [m.start() for m in re.finditer(r"\bcpu_job_signal_ring_end\s*\(\s*cpu_job_signal_ring_\b", shut)]
    ends = [m.start() for m in re.finditer(r"\bcpu_job_input_marker_end\s*\(", shut)]
    if len(ends) != len(rings) or not rings:
        fail("W7", f"shutdown_resources ends the CPU job signal ring {len(rings)} times and the input-marker word "
                   f"{len(ends)} times; every branch that ends one ends the other")
    drain = re.search(r"\bdrain_all_queues_noexcept\s*\(", shut)
    if not drain or not any(e > drain.start() for e in ends):
        fail("W7", "shutdown_resources lets the input-marker word go before its queues drained: a marker still queued "
                   "would write freed memory")


RING_CLASS = r"class\s+cpu_job_signal_ring\s*\{"
RING_WRAP_SIG = (r"bool\s+wrap\s*\(\s*uint32_t\s+id\s*,\s*uint32_t\s+generation\s*,\s*uint64_t\s*\*\s*serial\s*,"
                 r"\s*sycl::event\s*\*\s*out\s*\)\s*\{")
RING_RECYCLE_SIG = r"size_t\s+recycle\s*\(\s*uint64_t\s+upto\s*\)\s*noexcept\s*\{"
RING_SHUTDOWN_SIG = r"void\s+shutdown\s*\(\s*\)\s*noexcept\s*\{"
_RING_LOCK = re.compile(r"std::(?:lock_guard|unique_lock|scoped_lock)\s*<[^>]*>\s+\w+\s*\(\s*mutex_\b")


def lock_spans(body: str) -> list[tuple[int, int]]:
    """Each mutex_ lock in `body`, from the lock to the end of the innermost block that holds it."""
    spans = []
    for m in _RING_LOCK.finditer(body):
        open_at = enclosing_brace(body, m.start())
        if open_at < 0:
            spans.append((m.start(), len(body)))
        else:
            block, start = block_at(body, open_at, "W8")
            spans.append((m.start(), start + len(block)))
    return spans


def under_lock(spans: list[tuple[int, int]], pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def check_ring_call_outside(name: str, body: str, call: str, what: str) -> None:
    spans = lock_spans(body)
    hits = [m.start() for m in re.finditer(call, body)]
    if len(hits) != 1 or under_lock(spans, hits[0]):
        fail("W8", f"cpu_job_signal_ring::{name} {what} under mutex_ ({len(hits)} call(s)): the job's own signal takes "
                   "mutex_, and a driver call may block behind a host wait that waits for that job")
    up = [m.start() for m in re.finditer(r"\boutside_calls_\s*\+=\s*1\s*;", body)]
    down = [m.start() for m in re.finditer(r"\boutside_calls_\s*-=\s*1\s*;", body)]
    if len(up) != 1 or len(down) != 1 or not (up[0] < hits[0] < down[0]):
        fail("W8", f"cpu_job_signal_ring::{name} does not count its driver call in outside_calls_, raised before it and "
                   "lowered after it: a shutdown meanwhile would destroy the event under it")


def check_ring_book_locked(name: str, body: str, pattern: str) -> None:
    spans = lock_spans(body)
    hits = [m.start() for m in re.finditer(pattern, body)]
    if not hits or any(not under_lock(spans, h) for h in hits):
        fail("W8", f"cpu_job_signal_ring::{name} changes the book or outside_calls_ outside mutex_ (`{pattern}`)")


def check_ring_locks(s: dict[str, str]) -> None:
    cache = s["cache"]
    found = list(re.finditer(RING_CLASS, cache))
    if len(found) != 1:
        fail("W8", f"{len(found)} definitions of cpu_job_signal_ring, not 1")
    ring, _ = block_at(cache, cache.index("{", found[0].end() - 1), "W8")
    _, wrap = definition(ring, RING_WRAP_SIG, "W8")
    check_ring_call_outside("wrap", wrap, r"\bmake_event\s*<", "makes its event")
    for pattern in (r"\bbook_\s*\.\s*consume\s*\(", r"\bbook_\s*\.\s*begin_flush\s*\(",
                    r"\boutside_calls_\s*[-+]="):
        check_ring_book_locked("wrap", wrap, pattern)
    _, rec = definition(ring, RING_RECYCLE_SIG, "W8")
    check_ring_call_outside("recycle", rec, r"\bhost_reset\s*\(", "resets its events")
    for pattern in (r"\bbook_\s*\.\s*recycle_begin\s*\(", r"\bbook_\s*\.\s*recycle_finish\s*\(",
                    r"\boutside_calls_\s*[-+]="):
        check_ring_book_locked("recycle", rec, pattern)
    if re.search(r"\bbook_\s*\.\s*recycle\s*\(", rec):
        fail("W8", "cpu_job_signal_ring::recycle runs the book's one-lock recycle, whose resets run under mutex_")
    _, shut = definition(ring, RING_SHUTDOWN_SIG, "W8")
    for m in re.finditer(r"\b(?:event_destroy|pool_destroy)\s*\(", shut):
        guarded, b = False, enclosing_brace(shut, m.start())
        while b >= 0 and not guarded:
            guarded = re.search(r"\bif\s*\(\s*outside_calls_\s*==\s*0\s*\)\s*$", shut[:b]) is not None
            b = enclosing_brace(shut, b)
        if not guarded:
            fail("W8", "cpu_job_signal_ring::shutdown destroys native objects outside `if (outside_calls_ == 0)`: a wrap "
                       "or a recycle still in a driver call would use a destroyed event")


def check(s: dict[str, str]) -> None:
    check_wait_inputs(s)
    check_pool_lambda(s)
    check_helpers(s)
    check_ring_locks(s)


def sub(pattern: str, repl: str, count: int = 1):
    def apply(text: str) -> str:
        out, n = re.subn(pattern, repl, text, count=count)
        return out if n else text
    return apply


def move(pattern: str, before: str):
    """Deletes the first match of `pattern` and inserts it in front of the first match of `before`."""
    def apply(text: str) -> str:
        m = re.search(pattern, text)
        if not m:
            return text
        rest = text[: m.start()] + text[m.end() :]
        b = re.search(before, rest)
        if not b:
            return text
        return rest[: b.start()] + m.group(0) + rest[b.start() :]
    return apply


def self_test(s: dict[str, str]) -> int:
    marker_wait = r"(\n(\s*)ggml_sycl::cpu_job_input_marker_wait\(marker\);)"
    tick = r"(ggml_sycl::cpu_job_input_marker_wait\(marker\);)\n(\s*)ggml_sycl_watchdog_heartbeat\(\);"
    arm_call = r"!ggml_sycl::unified_cache_cpu_job_input_marker_arm\(stream, inputs\.events, &marker, &why\)"
    fallback = r"\n\s*sycl::event::wait\(inputs\.events\);"
    site1 = r"\n    std::function<void\(\)> job_prologue = moe_cpu_job_prologue\(\*stream, std::move\(job_inputs\), src0\);"
    end_call = (r"\n(\s*)\{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n"
                r"\s*cpu_job_input_marker_end\(cpu_job_input_marker_word_, cpu_job_input_marker_owner_\);\n\s*\}")
    mutants = [
        ("W1", "backend", "the worker waits an event natively again",
         sub(marker_wait, r"\1\n\2sycl::event().wait();")),
        ("W1", "backend", "the worker queries an event's status",
         sub(marker_wait, r"\1\n\2(void) sycl::event().get_info<sycl::info::event::command_execution_status>();")),
        ("W1", "backend", "the worker makes a synchronous copy",
         sub(marker_wait, r"\1\n\2ggml_sycl::mem_copy(ggml_sycl::mem_handle(), 0, ggml_sycl::mem_handle(), 0, 0, q, {});")),
        ("W1", "backend", "the closure waits",
         sub(r"(\n(\s*)moe_cpu_job_wait_inputs\(marker, zero_out, zero_bytes\);)", r"\n\2sycl::event::wait({});\1")),
        ("W2", "backend", "the marker wait is conditional",
         sub(r"\n(\s*)ggml_sycl::cpu_job_input_marker_wait\(marker\);",
             r"\n\1if (zero_out) {\n\1    ggml_sycl::cpu_job_input_marker_wait(marker);\n\1}")),
        ("W2", "backend", "the span is zeroed before the inputs land",
         sub(r"(\n(\s*)ggml_sycl::cpu_job_input_marker_wait\(marker\);)",
             r"\n\2std::memset(zero_out, 0, zero_bytes);\1")),
        ("W2", "backend", "the marker is waited twice", sub(marker_wait, r"\1\1")),
        ("W2", "backend", "the watchdog is ticked before the marker arrives",
         sub(tick, r"ggml_sycl_watchdog_heartbeat();\n\2\1")),
        ("W2", "backend", "the watchdog tick is dropped", sub(tick, r"\1")),
        ("W2", "backend", "the watchdog is ticked only while tracing",
         sub(tick, r"\1\n\2if (moe_hostpath_waits_enabled()) {\n\2    ggml_sycl_watchdog_heartbeat();\n\2}")),
        ("W3", "backend", "the worker is handed the events",
         sub(r"static void moe_cpu_job_wait_inputs\(const ggml_sycl::cpu_job_input_marker & marker,",
             r"static void moe_cpu_job_wait_inputs(const std::vector<sycl::event> & events, "
             r"const ggml_sycl::cpu_job_input_marker & marker,")),
        ("W3", "backend", "the closure captures the inputs",
         sub(r"zero_bytes = inputs\.zero_bytes\]", r"zero_bytes = inputs.zero_bytes, inputs = std::move(inputs)]")),
        ("W3", "backend", "the closure captures by default",
         sub(r"return \[marker = std::move\(marker\),", r"return [=, marker = std::move(marker),")),
        ("W4", "backend", "the prologue never arms the marker", sub(arm_call, r"true")),
        ("W4", "backend", "the prologue does not take the stream",
         sub(r"moe_cpu_job_prologue\(sycl::queue\s*&\s*stream,\s*", r"moe_cpu_job_prologue(")),
        ("W4", "backend", "the prologue skips the recording test",
         sub(r"!moe_cpu_job_input_marker_may\(stream\) \|\|\s*", "")),
        ("W4", "backend", "a refused arm is not waited", sub(fallback, "")),
        ("W4", "backend", "the prologue also waits the inputs when the marker is armed",
         sub(r"(\n(\s*)return \[marker = std::move\(marker\),)", r"\n\2sycl::event::wait(inputs.events);\1")),
        ("W4", "backend", "the recording test misses a recording stream",
         sub(r"return !ggml_sycl_graph_recording_active\(\) &&\s*stream\.ext_oneapi_get_state\(\) != "
             r"sycl::ext::oneapi::experimental::queue_state::recording;", r"return !ggml_sycl_graph_recording_active();")),
        ("W5", "backend", "a site builds its prologue without the stream",
         sub(r"moe_cpu_job_prologue\(\*stream, std::move\(job_inputs\), src0\)",
             r"moe_cpu_job_prologue(std::move(job_inputs), src0)")),
        ("W5", "backend", "the event-chained route arms after its device half",
         move(site1, r"\n\s*moe_hostpath_waits_readback\(record->layer,")),
        ("W6", "pool", "the job lambda waits an event",
         sub(r"(\n(\s*)if \(prologue\) \{)", r"\n\2sycl::event().wait();\1")),
        ("W6", "pool", "the job lambda queries the driver",
         sub(r"(\n(\s*)if \(prologue\) \{)", r"\n\2(void) zeEventQueryStatus(nullptr);\1")),
        ("W7", "header", "the marker wait sleeps",
         sub(r"(\n\s*\} else \{\n\s*)std::this_thread::yield\(\);(\n\s*\}\n\s*\}\n\})",
             r"\1std::this_thread::sleep_for(std::chrono::microseconds(100));\2")),
        ("W7", "header", "the marker wait ticks the watchdog",
         sub(r"(\n(\s*)\+\+spins;)", r"\1\n\2ggml_sycl_watchdog_heartbeat();")),
        ("W7", "header", "the marker wait takes a lock",
         sub(r"(\n(\s*)\+\+spins;)", r"\1\n\2std::lock_guard<std::mutex> hold(g_marker_mutex);")),
        ("W7", "header", "the marker wait enters SYCL",
         sub(r"(\n(\s*)\+\+spins;)", r"\1\n\2(void) sycl::event().get_info<sycl::info::event::command_execution_status>();")),
        ("W7", "header", "seen reads the word before abandoned",
         sub(r"if \(!m\.armed\(\) \|\| m\.word->abandoned\.load\(std::memory_order_acquire\)\) \{",
             r"if (!m.armed() || (__atomic_load_n(m.word->word, __ATOMIC_ACQUIRE) && "
             r"m.word->abandoned.load(std::memory_order_acquire))) {")),
        ("W7", "header", "seen reads the word relaxed",
         sub(r"__atomic_load_n\(m\.word->word, __ATOMIC_ACQUIRE\)", r"__atomic_load_n(m.word->word, __ATOMIC_RELAXED)")),
        ("W7", "cache", "the device task does not depend on the inputs", sub(r"\n\s*cgh\.depends_on\(after\);", "")),
        ("W7", "cache", "the device task stores before its fence",
         move(r"\n\s*\*word = serial;", r"\n\s*sycl::atomic_fence\(sycl::memory_order::release, sycl::memory_scope::system\);")),
        ("W7", "cache", "the fence has device scope only",
         sub(r"sycl::atomic_fence\(sycl::memory_order::release, sycl::memory_scope::system\)",
             r"sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device)")),
        ("W7", "cache", "the serial is submitted outside the lock",
         sub(r"\n(\s*)std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n(\s*)if \(!cpu_job_input_marker_word_\)",
             r"\n\1{\n\1    std::lock_guard<std::mutex> lock(cpu_job_input_marker_mutex_);\n\1}\n\2if (!cpu_job_input_marker_word_)")),
        ("W7", "cache", "the serial is consumed before the submit",
         move(r"\n\s*cpu_job_input_marker_serial_ = serial;", r"\n\s*try \{\n\s*cpu_job_input_marker_submit\(")),
        ("W7", "cache", "the word is not host-pinned",
         sub(r"req\.intent\.constraints\.must_host_pinned = true;", r"req.intent.constraints.must_device = true;")),
        ("W7", "cache", "a teardown branch keeps the word", sub(end_call, "")),
        ("W7", "cache", "the word goes before the drain",
         move(r"\n    \{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n"
              r"\s*cpu_job_input_marker_end\(cpu_job_input_marker_word_, cpu_job_input_marker_owner_\);\n    \}",
              r"\n\s*if \(!drain_all_queues_noexcept\(\)\) return false;")),
        # W8: the signal ring's driver calls and the lock its job's signal takes (llama.cpp-42zj).
        ("W8", "cache", "the ring makes its event under mutex_",
         sub(re.escape("        sycl::event ev;\n        std::string failure;"),
             "        std::lock_guard<std::timed_mutex> held(mutex_);\n        sycl::event ev;\n        std::string failure;")),
        ("W8", "cache", "the ring resets its events under mutex_",
         sub(re.escape("        std::vector<ze_result_t> results("),
             "        std::lock_guard<std::timed_mutex> held(mutex_);\n        std::vector<ze_result_t> results(")),
        ("W8", "cache", "recycle goes back to the book's one-lock recycle",
         sub(re.escape("if (book_.recycle_begin(upto, &ids) == 0) {"),
             "if (book_.recycle(upto, [](uint32_t) { return true; }) == 0) {")),
        ("W8", "cache", "recycle finishes the book outside mutex_",
         sub(re.escape("        std::lock_guard<std::timed_mutex> lock(mutex_);\n        outside_calls_ -= 1;\n        size_t n = 0;"),
             "        outside_calls_ -= 1;\n        size_t n = 0;")),
        ("W8", "cache", "wrap consumes the entry outside mutex_",
         sub(re.escape("        std::lock_guard<std::timed_mutex> lock(mutex_);\n        outside_calls_ -= 1;\n        if (!made) {"),
             "        outside_calls_ -= 1;\n        if (!made) {")),
        ("W8", "cache", "wrap's driver call is not counted",
         sub(re.escape("native = events_[id];\n            outside_calls_ += 1;"), "native = events_[id];")),
        ("W8", "cache", "shutdown destroys under a driver call",
         sub(re.escape("if (outside_calls_ == 0) {"), "if (true) {")),
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
    print("PASS: the CPU expert pool worker waits a job's inputs on a device-written marker with host loads, and makes "
          f"no driver call but the job's device signal {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
