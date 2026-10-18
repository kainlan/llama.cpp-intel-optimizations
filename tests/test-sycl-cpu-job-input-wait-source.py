#!/usr/bin/env python3
"""The CPU expert pool worker never enters the driver to wait for a job's inputs while the input marker is in play,
that is, with GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL on (llama.cpp-42zj, llama.cpp-nqbe).

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

THE MARKER FOLLOWS THE KNOB (llama.cpp-nqbe).  The hang needs device work that waits on a CPU job's own signal, which
only GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL=1 submits (llama.cpp-m5v8); with the knob off the worker's native wait never
hung.  So the marker is in play only with the knob on; with it off the prologue arms, refuses, logs and waits nothing,
and the worker waits a job's inputs natively, as it did before the marker (W10).  The marker once cost Qwen3.8 IQ3_XXS
decode on the B70 without -c 2.8 t/s against 9.7, with the knob off as with it on: its word sat in the host STAGING
zone, which every graph boundary checks for emptiness, so every boundary refused and walked the allocation registry.
The word is now a persistent owner in the WEIGHT host zone, so that cost is removed
(test-sycl-host-zone-boundary-owner-source.py).

THE CONTRACT.  With comments and string literals blanked:
  W1. The worker's input wait is driver-free.  moe_cpu_job_wait_inputs calls only cpu_job_input_marker_wait,
      ggml_sycl_watchdog_heartbeat, moe_hostpath_waits_enabled, cpu_expert_pool_trace_stamp_inputs_ready and
      memset; the closure moe_cpu_job_prologue returns calls only moe_cpu_job_wait_inputs and, while the marker is
      not in play, moe_cpu_job_wait_inputs_native (W10); neither names anything in sycl::.
  W2. The marker is waited once, unconditionally, at the top of moe_cpu_job_wait_inputs, before the watchdog tick,
      the inputs-ready stamp and the zeroing of the output span, which must follow the inputs.  The watchdog is
      ticked exactly once, unconditionally, after the wait (as a CPU expert batch ticks it when it finishes,
      llama.cpp-ff00) and never inside it, so a marker that never arrives fires the watchdog instead of spinning
      unseen.
  W3. While the marker is in play the worker cannot be handed an event.  moe_cpu_job_wait_inputs takes no sycl::event
      and no moe_cpu_job_inputs, and the closure captures only in_play, native, marker, zero_out and zero_bytes;
      native, the one capture that can hold events, is filled only while the marker is not in play (W10).
  W4. The marker is armed on the submitting thread, and a refused arm is waited there.  moe_cpu_job_prologue takes
      the stream and, outside its closure, calls unified_cache_cpu_job_input_marker_arm once, as the condition of an
      `if` that first asks `!moe_cpu_job_input_marker_may(stream) ||`; that branch holds
      `sycl::event::wait(inputs.events)` inside the scope of a `MOE_WAIT_B2` census timer, so the waits line counts
      it, and logs its WARN under one flag per reason (an array of flags indexed by the cpu_job_input_marker_refusal
      the arm names), so a refusal for one reason does not hide the first refusal for another; every wait outside
      the closure sits in that branch.  moe_cpu_job_input_marker_may
      is `!moe_cpu_job_stream_recording(stream)`, the one recording test the CPU job's three decisions share: the
      backend's flag or the stream's own state (a recorded marker would be written only at replay).
  W5. Every prologue is built with its stream: each moe_cpu_job_prologue call passes the op's stream first, and the
      event-chained hybrid route builds its prologue before its table copy and its device-expert kernel, so the
      in-order marker does not wait for that kernel and the CPU and GPU halves still overlap.
  W6. The pool's job lambda (cpu-expert-pool.cpp), and every closure ggml-sycl.cpp hands to submit_batch as a job's
      resolve step, on whatever receiver expression (it runs on the worker after the prologue, before the signal),
      names nothing in sycl:: and makes
      no wait, status query, submission or ze/ur call of its own, template calls such as get_info<...>() included:
      the job's one driver call is its device signal.  At least one resolve closure is read, so the scan is not
      vacuous.
  W7. The marker itself (cpu-job-signal.hpp, unified-cache.cpp).  The header's wait helpers call nothing but each
      other, the pause and yield, the clock, the word's slow report, and the atomic loads and read-modify-writes;
      none sleeps, locks, ticks the watchdog or names sycl::.  The seen test counts itself in the word's readers
      (seq_cst), then reads `abandoned` (seq_cst) and only when it is clear the word (acquire), then uncounts itself;
      the abandon sets `abandoned` (seq_cst) and then yields until readers is 0; cpu_job_input_marker_end abandons
      the word before it releases the owner, so no waiter's load can follow the release.  The compare is the signed
      32-bit distance (the modulo-2^32 invariant is stated beside it).  The wait reports a slow wait at most once,
      through the word's slow, once it has yielded for slow_after; the arm installs the cache's reporter, which only
      logs.  The device task depends on the job's inputs, then fences (release, system scope) before it stores the
      serial.  The arm takes the cache's marker mutex first and releases it once in the submit's catch, which ends in
      `return false;`, and once on the way out, after it records the serial; after either release it touches no marker
      member, and only after one does it log: the word's creation, once after each release, through
      cpu_job_input_marker_report_made (which only logs, at INFO, and is handed the word this arm made), and in the
      catch the throw, once per process, so no arm waits on the log's lock behind another; every refusal of the arm
      and of unified_cache_cpu_job_input_marker_arm names its reason right before it returns, since the caller's log
      flag is chosen by it; the arm refuses a stream that is not in order before it allocates; zeroes the word before
      it publishes it; takes the next serial as the last plus one; submits every marker on queue_, before it records
      the new serial; and takes its word from the unified cache as a host-pinned allocation.
      unified_cache_cpu_job_input_marker_arm reaches the cache through moe_join_queue_cache_of(stream), which returns
      only the cache whose queue_ is that stream.  The owner, the word's allocation, is assigned only by the arm's one
      publish and released only through cpu_job_input_marker_end, in unified-cache.cpp and in unified-cache.hpp's
      inline code alike: a release anywhere else would let the allocation go without the abandon, under a waiter's
      load.  shutdown_resources, in each of its three branches (the SYCL runtime already shutting down, an invalid
      context, and the normal one after the queues drained), ends the signal ring once and then the word once, and
      ends neither anywhere else: the ring is no longer live once ended, so a job the abandon releases signals it
      without a driver call, and on the two SYCL-gone branches no driver call could succeed.
  W8. The signal ring (cpu_job_signal_ring, unified-cache.cpp) makes no driver call that may block under mutex_, the
      lock the job's own signal takes: a host wait on another thread may hold the driver's device lock behind that
      job.  wrap makes its event and recycle resets its events outside every mutex_ scope, each call counted in
      outside_calls_ (raised before it and lowered after it); the book (consume, begin_flush, recycle_begin,
      recycle_finish) and outside_calls_ change only under mutex_; recycle does not use the book's one-lock recycle;
      and shutdown destroys native objects only inside `if (outside_calls_ == 0)`.  grow_locked, which creates pools
      and events under mutex_, is outside this pin (llama.cpp-q0xu).
  W9. The event-chained hybrid route's job waits for its experts' leases: when record->has_ready, the job's inputs
      take record->ready, the barrier over the leases' producer events, before the prologue is built.  Its marker
      would follow that barrier on the in-order stream anyway, but a refused arm waits exactly the inputs on the
      submitting thread, and without it the job would read host weights still being written.
  W10. The marker is in play only with the knob on.  moe_cpu_job_input_marker_in_play is exactly
      `return moe_cpu_job_device_signal_enabled();`.  moe_cpu_job_prologue asks it once, by copy into `in_play`, in
      its first statement and before any call of its own; its first branch, `if (!in_play)`, does nothing but move
      the inputs' events into `native`, its only write, and the arm, its refusal WARN and its B2 wait all sit in that
      branch's `else`, so with the knob off nothing is armed, refused, logged or waited on the submitting thread.  The
      closure is exactly `if (in_play)` the marker wait, else moe_cpu_job_wait_inputs_native(native, zero_out,
      zero_bytes), capturing in_play by copy.  moe_cpu_job_wait_inputs_native is the worker's wait from before the
      marker: it waits the events when there are any, then stamps the inputs-ready trace, then zeroes the span, and
      ticks no watchdog.

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
RED for the readers drain and the slow-wait report, on 2b5fb934a (the commit that added W8):
  FAIL [pin W7]: cpu_job_input_marker_seen does not count itself in readers (seq_cst), then test abandoned ...
The resolve-step scan (W6), the arm's serial, zeroing, in-order refusal, queue and cache (W7) and W9 were gaps: on
2b5fb934a each of their mutants passes the pin as it stood, and fails it here.  So were a resolve step handed to
submit_batch on a call expression (W6), the owner's other writers and the order inside each shutdown branch (W7): on
4b162ae16 each of their mutants passes the pin as it stood.  So were the refused arm's B2 count and its WARN once
per reason (W4), and the arm's throw report outside the mutex and once per process, and its named refusals (W7):
each of their mutants passes the pin as it stood on f60cc2636, with only its lock pattern widened to the
unique_lock the arm now takes.
RED for W10 on eaafb5b83, which arms the marker whatever the knob (Qwen3.8 decode without -c measured 2.79 t/s there
with the knob unset, against 9.69 and 9.65 on master), while the pin as it stood passes that tree:
  FAIL [pin W10]: moe_cpu_job_prologue arms the input marker whatever GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL says: ...
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


def scope_end(code: str, pos: int) -> int:
    """Where the innermost block that holds `pos` closes; the end of `code` when `pos` is at its top level."""
    open_at = enclosing_brace(code, pos)
    if open_at < 0:
        return len(code)
    body, start = block_at(code, open_at, "scope")
    return start + len(body)


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
CACHE_HPP_REL = Path("ggml") / "src" / "ggml-sycl" / "unified-cache.hpp"

WAIT_SIG = r"static\s+void\s+moe_cpu_job_wait_inputs\s*\(([^)]*)\)\s*\{"
PROLOGUE_SIG = r"static\s+std::function<void\(\)>\s+moe_cpu_job_prologue\s*\(([^)]*)\)\s*\{"
MAY_SIG = r"static\s+bool\s+moe_cpu_job_input_marker_may\s*\(([^)]*)\)\s*\{"
STREAM_RECORDING_SIG = r"static\s+bool\s+moe_cpu_job_stream_recording\s*\(\s*sycl::queue\s*&\s*stream\s*\)\s*\{"
CHAINED_SIG = r"static\s+bool\s+ggml_sycl_moe_hybrid_decode_event_chained\s*\(([^)]*)\)\s*\{"
WAIT_ALLOWED = {"cpu_job_input_marker_wait", "ggml_sycl_watchdog_heartbeat", "moe_hostpath_waits_enabled",
                "cpu_expert_pool_trace_stamp_inputs_ready", "memset"}
CLOSURE_ALLOWED = {"moe_cpu_job_wait_inputs", "moe_cpu_job_wait_inputs_native"}
CAPTURES_ALLOWED = {"in_play", "native", "marker", "zero_out", "zero_bytes"}
# W10: the marker follows the knob (llama.cpp-nqbe). Compared with every space removed.
IN_PLAY_SIG = r"static\s+bool\s+moe_cpu_job_input_marker_in_play\s*\(\s*\)\s*\{"
NATIVE_SIG = r"static\s+void\s+moe_cpu_job_wait_inputs_native\s*\(([^)]*)\)\s*\{"
NATIVE_PARAMS = "const std::vector<sycl::event> & events, float * zero_out, size_t zero_bytes"
NATIVE_BODY = ("if (!events.empty()) { sycl::event::wait(events); } "
               "if (moe_hostpath_waits_enabled()) { ggml_sycl::cpu_expert_pool_trace_stamp_inputs_ready(); } "
               "if (zero_out && zero_bytes > 0) { std::memset(zero_out, 0, zero_bytes); }")
IN_PLAY_FIRST = "const bool in_play = moe_cpu_job_input_marker_in_play();"
NOT_IN_PLAY_BODY = "native = std::move(inputs.events);"
CLOSURE_BODY = ("if (in_play) { moe_cpu_job_wait_inputs(marker, zero_out, zero_bytes); } "
                "else { moe_cpu_job_wait_inputs_native(native, zero_out, zero_bytes); }")
DRIVER_CALL = re.compile(r"^(?:wait|wait_and_throw|wait_for|throw_asynchronous|get_info|get_profiling_info|submit|"
                         r"single_task|parallel_for|memcpy|memset_async|fill|copy|depends_on|ext_oneapi_\w+|"
                         r"ze[A-Z]\w*|ur[A-Z]\w*|mem_copy\w*|event_complete\w*)$")
ARM = r"\bunified_cache_cpu_job_input_marker_arm\s*\("
# The prologue's WARN for a refused arm, as its format string reads.
REFUSAL_WARN = "its input marker was not armed (%s)"

# W7: what the header's marker helpers may call. None may sleep, lock, tick the watchdog or name sycl::.
HELPERS = {
    "cpu_job_input_marker_wait": (r"inline\s+void\s+cpu_job_input_marker_wait\s*\(([^)]*)\)\s*\{",
                                  {"cpu_job_input_marker_seen", "cpu_job_input_marker_pause", "yield", "now", "slow",
                                   "count"}),
    "cpu_job_input_marker_seen": (r"inline\s+bool\s+cpu_job_input_marker_seen\s*\(([^)]*)\)\s*\{",
                                  {"armed", "fetch_add", "load", "__atomic_load_n", "cpu_job_input_marker_reached",
                                   "fetch_sub"}),
    "cpu_job_input_marker_abandon": (r"inline\s+void\s+cpu_job_input_marker_abandon\s*\(([^)]*)\)\s*\{",
                                     {"store", "load", "yield"}),
    "cpu_job_input_marker_pause": (r"inline\s+void\s+cpu_job_input_marker_pause\s*\(([^)]*)\)\s*\{",
                                   {"__builtin_ia32_pause", "yield"}),
    "cpu_job_input_marker_reached": (r"inline\s+bool\s+cpu_job_input_marker_reached\s*\(([^)]*)\)\s*\{", set()),
}
HELPER_FORBIDDEN = re.compile(r"\b(?:sleep_for|sleep_until|usleep|nanosleep|lock_guard|unique_lock|scoped_lock|"
                              r"mutex|ggml_sycl_watchdog_heartbeat|wait_for|wait_until)\b|\bsycl\s*::")
SUBMIT_SIG = r"static\s+void\s+cpu_job_input_marker_submit\s*\(([^)]*)\)\s*\{"
END_SIG = r"static\s+void\s+cpu_job_input_marker_end\s*\(([^)]*)\)\s*noexcept\s*\{"
REPORT_SIG = r"static\s+void\s+cpu_job_input_marker_report_slow\s*\(([^)]*)\)\s*\{"
MADE_SIG = r"static\s+void\s+cpu_job_input_marker_report_made\s*\(([^)]*)\)\s*\{"
ARM_FN_SIG = r"bool\s+unified_cache_cpu_job_input_marker_arm\s*\(([^)]*)\)\s*\{"
CACHE_OF_SIG = r"static\s+unified_cache\s*\*\s*moe_join_queue_cache_of\s*\(([^)]*)\)\s*\{"
ARM_METHOD_SIG = r"bool\s+unified_cache::arm_cpu_job_input_marker\s*\(([^)]*)\)\s*\{"
SHUTDOWN_SIG = r"bool\s+unified_cache::shutdown_resources\s*\(([^)]*)\)\s*(?:noexcept\s*)?\{"


def read_sources(root: Path) -> dict[str, str]:
    return {"backend": blank_comments((root / BACKEND_REL).read_text()),
            "pool": blank_comments((root / POOL_REL).read_text()),
            "header": blank_comments((root / HEADER_REL).read_text()),
            "cache": blank_comments((root / CACHE_REL).read_text()),
            "cache_hpp": blank_comments((root / CACHE_HPP_REL).read_text())}


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
                   (", ".join(bad) if bad else "something in sycl::") + "; it may call only " +
                   ", ".join(sorted(CLOSURE_ALLOWED)))

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
    fwait = re.search(r"\bsycl\s*::\s*event\s*::\s*wait\s*\(\s*inputs\s*\.\s*events\s*\)", fallback)
    timer = re.search(r"\bmoe_hostpath_wait_timer\s+\w+\s*\(\s*MOE_WAIT_B2\s*\)\s*;", fallback)
    if not timer or not (timer.start() < fwait.start() < scope_end(fallback, timer.start())):
        fail("W4", "moe_cpu_job_prologue's wait for a refused arm's inputs is not inside the scope of a MOE_WAIT_B2 "
                   "census timer: the [MOE-HOSTPATH-WAITS] line would not show the submitting thread waiting a CPU "
                   "job's inputs")
    latch = None
    for m in re.finditer(r"\bif\s*\(\s*!\s*(\w+)\s*\[\s*why\s*\]\s*\.\s*exchange\s*\(\s*true\b[^)]*\)\s*\)\s*\{",
                         fallback):
        if REFUSAL_WARN in block_at(fallback, m.end() - 1, "W4")[0]:
            latch = m.group(1)
    flags = latch and re.search(r"\bstatic\s+std::atomic<bool>\s+" + latch +
                                r"\s*\[\s*ggml_sycl\s*::\s*CPU_JOB_INPUT_MARKER_REFUSAL_COUNT\s*\]", code)
    named = re.search(r"\bggml_sycl\s*::\s*cpu_job_input_marker_refusal\s+why\s*=", outside)
    if not latch or not flags or not named:
        fail("W4", "moe_cpu_job_prologue does not log a refused arm once per reason, under a flag of an array indexed "
                   "by the cpu_job_input_marker_refusal the arm names: the first refusal for one reason would hide "
                   "every refusal for another")
    _, maybody = definition(code, MAY_SIG, "W4")
    _, recbody = definition(code, STREAM_RECORDING_SIG, "W4")
    if not re.fullmatch(r"\s*return\s*!\s*moe_cpu_job_stream_recording\s*\(\s*stream\s*\)\s*;\s*", maybody) or \
       not re.fullmatch(r"\s*return\s+ggml_sycl_graph_recording_active\s*\(\s*\)\s*\|\|\s*stream\s*\.\s*"
                        r"ext_oneapi_get_state\s*\(\s*\)\s*==\s*sycl::ext::oneapi::experimental::queue_state::recording"
                        r"\s*;\s*", recbody):
        fail("W4", "moe_cpu_job_input_marker_may does not refuse while the backend or the stream records a graph "
                   "(`!moe_cpu_job_stream_recording(stream)`, that helper the backend's flag or the stream's state)")

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


_DRIVER_USE = re.compile(r"\b(wait|wait_and_throw|wait_for|throw_asynchronous|get_info|get_profiling_info|submit|"
                         r"single_task|parallel_for|memcpy|memset_async|fill|copy|depends_on|ext_oneapi_\w+|"
                         r"ze[A-Z]\w*|ur[A-Z]\w*|mem_copy\w*|event_complete\w*)\s*(?:<[^;{}()]*>)?\s*\(")


def driver_uses(body: str) -> list[str]:
    """Driver-facing calls in `body`, template calls (get_info<...>()) included; std:: host helpers left out."""
    flat = strip_strings(body)
    out = set()
    for m in _DRIVER_USE.finditer(flat):
        if re.search(r"\bstd\s*::\s*$", flat[max(0, m.start() - 8) : m.start()]):
            continue
        out.add(m.group(1))
    return sorted(out)


def top_level_args(text: str) -> list[str]:
    args, depth, start = [], 0, 0
    flat = strip_strings(text)
    for i, c in enumerate(flat):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[start:i])
            start = i + 1
    args.append(text[start:])
    return [a.strip() for a in args]


def lambda_body(code: str, open_bracket: int, pin: str) -> str:
    """The body of the lambda whose capture list opens at `open_bracket`."""
    close = code.index("]", open_bracket)
    params = re.match(r"\s*\(", code[close + 1 :])
    if not params:
        fail(pin, "a lambda without a parameter list")
    end = matching_paren(code, close + 1 + params.end() - 1)
    body, _ = block_at(code, code.index("{", end), pin)
    return body


def submit_batch_resolves(code: str) -> list[tuple[int, str]]:
    """(line, body) of every closure ggml-sycl.cpp hands to submit_batch as its resolve argument (the third), whatever
    the receiver expression: `cpu_pool.`, `pool->`, `get_pool().` and the like."""
    out = []
    for m in re.finditer(r"(?:\.|->)\s*submit_batch\s*\(", code):
        line = code.count("\n", 0, m.start()) + 1
        args = top_level_args(code[m.end() : matching_paren(code, m.end() - 1)])
        if len(args) < 3:
            fail("W6", f"ggml-sycl.cpp:{line} calls submit_batch with {len(args)} arguments, so its resolve step "
                       "cannot be read")
        arg = args[2]
        if arg in ("{}", "nullptr"):
            continue
        named = re.fullmatch(r"std\s*::\s*move\s*\(\s*(\w+)\s*\)|(\w+)", arg)
        if named:
            name = named.group(1) or named.group(2)
            defs = list(re.finditer(r"\bauto\s+" + name + r"\s*=\s*\[", code[: m.start()]))
            if not defs:
                fail("W6", f"ggml-sycl.cpp:{line} hands submit_batch the resolve step `{name}`, which is not an "
                           "`auto {name} = [...](...) {{...}}` closure the pin can read")
            out.append((line, lambda_body(code, defs[-1].end() - 1, "W6")))
        elif arg.startswith("["):
            out.append((line, lambda_body(code, m.end() + code[m.end() :].index(arg), "W6")))
        else:
            fail("W6", f"ggml-sycl.cpp:{line} hands submit_batch the resolve step `{arg}`, which the pin cannot read")
    return out


def check_pool_lambda(s: dict[str, str]) -> None:
    code = s["pool"]
    m = re.search(r"work_queue_\.push\(\s*\[[^\]]*\]\s*\(\s*\)\s*mutable\s*\{", code, re.S)
    if not m:
        fail("W6", "no job lambda pushed onto the pool's work queue")
    body, _ = block_at(code, code.index("{", m.end() - 1), "W6")
    bad = driver_uses(body)
    if bad or re.search(r"\bsycl\s*::", body):
        fail("W6", "the pool's job lambda makes driver-facing calls of its own: " +
                   (", ".join(bad) if bad else "something in sycl::") +
                   "; its one driver call is the job's device signal")
    resolves = submit_batch_resolves(s["backend"])
    if not resolves:
        fail("W6", "no closure handed to submit_batch as a resolve step was found, so their scan would pass vacuously")
    for line, body in resolves:
        bad = driver_uses(body)
        if bad or re.search(r"\bsycl\s*::", body):
            fail("W6", f"the resolve step handed to submit_batch at ggml-sycl.cpp:{line} runs on the pool worker before "
                       "the job signals, and makes driver-facing calls: " +
                       (", ".join(bad) if bad else "something in sycl::") +
                       "; a status query or a wait there may take the driver's device lock behind this very job")


def check_helpers(s: dict[str, str]) -> None:
    hdr = s["header"]
    _, seen = definition(hdr, HELPERS["cpu_job_input_marker_seen"][0], "W7")
    inc = re.search(r"\breaders\s*\.\s*fetch_add\s*\(\s*1\s*,\s*std::memory_order_seq_cst\s*\)", seen)
    ab = re.search(r"\babandoned\s*\.\s*load\s*\(\s*std::memory_order_seq_cst\s*\)\s*\|\|\s*"
                   r"cpu_job_input_marker_reached\s*\(\s*__atomic_load_n\s*\([^;]*__ATOMIC_ACQUIRE\s*\)", seen)
    dec = re.search(r"\breaders\s*\.\s*fetch_sub\s*\(\s*1\s*,\s*std::memory_order_(?:release|seq_cst)\s*\)", seen)
    if not inc or not ab or not dec or not (inc.start() < ab.start() < dec.start()) or \
       re.search(r"\breturn\b", seen[inc.start() : dec.start()]):
        fail("W7", "cpu_job_input_marker_seen does not count itself in readers (seq_cst), then test abandoned (seq_cst) "
                   "and load the word (acquire) only when it is clear, and then uncount itself on its one path: the "
                   "cache could release the word's allocation under the load, or the inputs be read before the word")
    _, aban = definition(hdr, HELPERS["cpu_job_input_marker_abandon"][0], "W7")
    st = re.search(r"\babandoned\s*\.\s*store\s*\(\s*true\s*,\s*std::memory_order_seq_cst\s*\)", aban)
    drain = re.search(r"\bwhile\s*\(\s*w\s*\.\s*readers\s*\.\s*load\s*\(\s*std::memory_order_seq_cst\s*\)\s*!=\s*0\s*\)",
                      aban)
    if not st or not drain or st.start() > drain.start():
        fail("W7", "cpu_job_input_marker_abandon does not set abandoned (seq_cst) and then wait until readers is 0: the "
                   "word's allocation could go while a waiter is between its test of the flag and its load")
    _, reached = definition(hdr, HELPERS["cpu_job_input_marker_reached"][0], "W7")
    if not re.fullmatch(r"\s*return\s+static_cast<int32_t>\s*\(\s*seen\s*-\s*serial\s*\)\s*>=\s*0\s*;\s*", reached):
        fail("W7", "cpu_job_input_marker_reached is not the signed 32-bit distance `static_cast<int32_t>(seen - serial) "
                   ">= 0`: the word is compared modulo 2^32, and any other compare releases a job early at the wrap or "
                   "never")
    _, wait = definition(hdr, HELPERS["cpu_job_input_marker_wait"][0], "W7")
    told = re.search(r"\bif\s*\(\s*!\s*told\s*&&\s*m\s*\.\s*word\s*->\s*slow\s*\)", wait)
    late = re.search(r"\bnow\s*-\s*since\s*>=\s*m\s*\.\s*word\s*->\s*slow_after\b", wait)
    once = re.search(r"\btold\s*=\s*true\s*;\s*m\s*\.\s*word\s*->\s*slow\s*\(\s*\*\s*m\s*\.\s*word\s*,\s*m\s*\.\s*serial\s*,",
                     wait)
    if not told or not late or not once or not (told.start() < late.start() < once.start()):
        fail("W7", "cpu_job_input_marker_wait does not report a wait once, through the word's slow and with its "
                   "serial, after it has yielded for slow_after: with the SYCL watchdog off a marker that never "
                   "arrives would be a silent spin")

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
    lock = re.search(r"std::(?:lock_guard|unique_lock)<std::mutex>\s+(\w+)\s*\(\s*cpu_job_input_marker_mutex_\s*\)",
                     arm)
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
    check_arm_unlock(cache, arm, lock)
    if not re.search(r"\bconst\s+uint32_t\s+serial\s*=\s*cpu_job_input_marker_serial_\s*\+\s*1\s*;", arm):
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not take the last serial plus one: a serial that does "
                   "not advance lets a later job pass on an earlier job's marker")
    alloc = re.search(r"\bunified_allocate_owner\s*\(", arm)
    io = re.search(r"\bif\s*\(\s*!\s*queue_\s*\.\s*has_property\s*<\s*sycl::property::queue::in_order\s*>\s*\(\s*\)\s*\)\s*\{", arm)
    if not io or io.start() > alloc.start() or not re.search(r"\breturn\s+false\s*;",
                                                            block_at(arm, io.end() - 1, "W7")[0]):
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not refuse a stream that is not in order before it "
                   "allocates: markers could run out of serial order and the word go backwards")
    zero = re.search(r"\*\s*word\s*->\s*ptr\s*=\s*0\s*;", arm)
    pub = re.search(r"\bcpu_job_input_marker_word_\s*=\s*std::move\s*\(\s*word\s*\)\s*;", arm)
    slow = re.search(r"\bword\s*->\s*slow\s*=\s*cpu_job_input_marker_report_slow\s*;", arm)
    if not zero or not pub or zero.start() > pub.start():
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not zero its word before it publishes it: a reused "
                   "block may hold a value at or past a serial, and that job would read inputs before they land")
    if not slow or slow.start() > pub.start():
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not install the cache's slow-wait reporter before it "
                   "publishes the word: a marker that never arrives would be a silent spin with the watchdog off")
    queues = [q.group(1) for q in re.finditer(r"\bcpu_job_input_marker_submit\s*\(\s*([^,]*?)\s*,", arm)]
    if not queues or any(q != "queue_" for q in queues):
        fail("W7", "unified_cache::arm_cpu_job_input_marker submits a marker on " + ", ".join(queues or ["nothing"]) +
                   ", not on queue_, the in-order stream its serials are ordered on")
    _, armfn = definition(cache, ARM_FN_SIG, "W7")
    if not re.search(r"\bunified_cache\s*\*\s*cache\s*=\s*moe_join_queue_cache_of\s*\(\s*stream\s*\)\s*;", armfn) or \
       not re.search(r"\bcache\s*->\s*arm_cpu_job_input_marker\s*\(", armfn):
        fail("W7", "unified_cache_cpu_job_input_marker_arm does not reach the cache through "
                   "moe_join_queue_cache_of(stream): a marker on another cache's queue would not follow the job's inputs")
    for fn, body in (("unified_cache::arm_cpu_job_input_marker", arm), ("unified_cache_cpu_job_input_marker_arm", armfn)):
        refusals = len(re.findall(r"\breturn\s+false\s*;", body))
        named = len(re.findall(r"\*\s*why\s*=\s*CPU_JOB_INPUT_MARKER_REFUSAL_(?!COUNT\b)\w+\s*;\s*return\s+false\s*;", body))
        if refusals == 0 or named != refusals:
            fail("W7", f"{fn} refuses {refusals - named} time(s) without naming its reason right before it returns: "
                       "the caller would log that refusal under another reason's flag, or not at all")
    _, cof = definition(cache, CACHE_OF_SIG, "W7")
    if not re.search(r"\bcache\s*->\s*get_queue\s*\(\s*\)\s*==\s*q\b", cof):
        fail("W7", "moe_join_queue_cache_of returns a cache whose queue_ is not the stream it was given")
    _, end = definition(cache, END_SIG, "W7")
    ab_at = re.search(r"\bcpu_job_input_marker_abandon\s*\(\s*\*\s*word\s*\)", end)
    rel = re.search(r"\bowner\s*=\s*\{\s*\}", end)
    if not ab_at or not rel or ab_at.start() > rel.start() or re.search(r"\babandoned\s*\.\s*store\b", end):
        fail("W7", "cpu_job_input_marker_end does not abandon the word (cpu_job_input_marker_abandon) before it "
                   "releases the owner: a waiter's load of the word could follow the release of its allocation")
    _, rep_body = definition(cache, REPORT_SIG, "W7")
    bad = sorted({q for q, n in calls(rep_body) if n != "GGML_LOG_WARN"})
    if bad or re.search(r"\bsycl\s*::", rep_body) or not re.search(r"\bGGML_LOG_WARN\s*\(", rep_body):
        fail("W7", "cpu_job_input_marker_report_slow runs on the pool worker and does more than log one WARN: " +
                   (", ".join(bad) if bad else "no WARN, or something in sycl::"))
    _, made_body = definition(cache, MADE_SIG, "W7")
    bad = sorted({q for q, n in calls(made_body) if n != "GGML_LOG_INFO"})
    if bad or re.search(r"\bsycl\s*::", made_body) or not re.search(r"\bGGML_LOG_INFO\s*\(", made_body):
        fail("W7", "cpu_job_input_marker_report_made does more than log the word's creation at INFO: " +
                   (", ".join(bad) if bad else "no INFO, or something in sycl::"))
    check_marker_owner(s)
    check_shutdown_branches(cache)


def check_arm_unlock(cache: str, arm: str, lock: re.Match) -> None:
    """W7: the arm releases its mutex once in the submit's catch, which ends in `return false;`, and once on the way
    out, after it records the serial; after either release it touches no marker member, and it logs only after one:
    the word's creation, once after each release, through cpu_job_input_marker_report_made with the word it made,
    and in the catch the throw, once per process."""
    unlocks = [m.start() for m in re.finditer(r"\b" + lock.group(1) + r"\s*\.\s*unlock\s*\(\s*\)\s*;", arm)]
    catch = list(re.finditer(r"\bcatch\s*\(\s*const\s+sycl::exception\s*&\s*\w+\s*\)\s*\{", arm))
    warns = [m.start() for m in re.finditer(r"\bGGML_LOG_WARN\s*\(", arm)]
    if len(catch) != 1:
        fail("W7", f"unified_cache::arm_cpu_job_input_marker has {len(catch)} catch(const sycl::exception &) blocks; "
                   "it catches its submit's throw once")
    body, start = block_at(arm, catch[0].end() - 1, "W7")
    end = start + len(body)
    if not re.search(r"\breturn\s+false\s*;\Z", body.rstrip()):
        fail("W7", "unified_cache::arm_cpu_job_input_marker's submit catch does not end in `return false;`: a throw "
                   "would fall through to the way out, release the mutex a second time and record the serial and "
                   "*out without it")
    took = re.search(r"\bcpu_job_input_marker_serial_\s*=\s*serial\s*;", arm)
    inside = [u for u in unlocks if start <= u < end]
    outside = [u for u in unlocks if not start <= u < end]
    if len(inside) != 1 or len(outside) != 1 or not took or outside[0] < took.end():
        fail("W7", "unified_cache::arm_cpu_job_input_marker releases cpu_job_input_marker_mutex_ "
                   f"{len(inside)} time(s) in its submit's catch and {len(outside)} time(s) elsewhere; it releases it "
                   "once in the catch and once after it records the serial, so the serial and the submit stay under it")
    released = ((inside[0], end), (outside[0], len(arm)))
    for m in re.finditer(r"\bGGML_LOG_\w+\s*\(|\bcpu_job_input_marker_report_made\s*\(", arm):
        if not any(a <= m.start() < b for a, b in released):
            fail("W7", f"unified_cache::arm_cpu_job_input_marker calls {m.group(0).rstrip('( ')} with "
                       "cpu_job_input_marker_mutex_ held: an arm on another thread would wait on the log's lock "
                       "behind it")
    if not warns or any(not inside[0] <= w < end for w in warns):
        fail("W7", "unified_cache::arm_cpu_job_input_marker logs a WARN outside its submit's catch, or none there")
    reports = [m.start() for m in re.finditer(r"\bcpu_job_input_marker_report_made\s*\(", arm)]
    made = re.search(r"\bmade\s*=\s*word\s*;", arm)
    pub = re.search(r"\bcpu_job_input_marker_word_\s*=\s*std::move\s*\(\s*word\s*\)\s*;", arm)
    if any(sum(a <= r < b for r in reports) != 1 for a, b in released) or not made or not pub or \
       made.start() > pub.start() or \
       any(not re.match(r"cpu_job_input_marker_report_made\s*\(\s*made\s*\.\s*get\s*\(\s*\)\s*\)", arm[r:])
           for r in reports):
        fail("W7", "unified_cache::arm_cpu_job_input_marker does not hand cpu_job_input_marker_report_made the word "
                   "it made once after each release: the word's creation would go unlogged, or be logged twice")
    early = re.search(r"\bcpu_job_input_marker_\w+_\b", arm[: lock.start()])
    if early:
        fail("W7", f"unified_cache::arm_cpu_job_input_marker uses {early.group(0)} before it takes "
                   "cpu_job_input_marker_mutex_")
    for a, b in released:
        late = re.search(r"\bcpu_job_input_marker_\w+_\b", arm[a:b])
        if late:
            fail("W7", f"unified_cache::arm_cpu_job_input_marker uses {late.group(0)} after it released "
                       "cpu_job_input_marker_mutex_")
    latched = False
    for m in re.finditer(r"\bif\s*\(\s*!\s*(\w+)\s*\.\s*exchange\s*\(\s*true\b[^)]*\)\s*\)\s*\{", body):
        b, s0 = block_at(body, m.end() - 1, "W7")
        if all(start + s0 <= w < start + s0 + len(b) for w in warns) and \
           re.search(r"\bstatic\s+std::atomic<bool>\s+" + m.group(1) + r"\s*\{", cache):
            latched = True
    if not latched:
        fail("W7", "unified_cache::arm_cpu_job_input_marker logs every throwing submit: once per process is enough, "
                   "since a stream that throws once throws for every job, and the prologue logs each reason once")


OWNER_PUBLISH = r"\bcpu_job_input_marker_owner_\s*=\s*std::move\s*\(\s*owner\s*\)\s*;"
OWNER_END = r"\bcpu_job_input_marker_end\s*\(\s*cpu_job_input_marker_word_\s*,\s*cpu_job_input_marker_owner_\s*\)"


def check_marker_owner(s: dict[str, str]) -> None:
    """W7: the owner is assigned by the arm's one publish and released only through cpu_job_input_marker_end."""
    cache = s["cache"]
    m, arm = definition(cache, ARM_METHOD_SIG, "W7")
    arm_at = cache.index("{", m.end() - 1) + 1
    allowed = []
    pubs = [arm_at + p.start() for p in re.finditer(OWNER_PUBLISH, arm)]
    if len(pubs) != 1:
        fail("W7", f"unified_cache::arm_cpu_job_input_marker assigns cpu_job_input_marker_owner_ {len(pubs)} times, "
                   "not once, when it publishes the word")
    allowed.append(pubs[0])
    allowed.extend(e.start() + e.group(0).index("cpu_job_input_marker_owner_") for e in re.finditer(OWNER_END, cache))
    for u in re.finditer(r"\bcpu_job_input_marker_owner_\b", cache):
        if u.start() in allowed:
            continue
        line = cache.count("\n", 0, u.start()) + 1
        fail("W7", f"unified-cache.cpp:{line} uses cpu_job_input_marker_owner_ outside the arm's publish and "
                   "cpu_job_input_marker_end: the word's allocation would go without the abandon, under a waiter's "
                   "load of the word")
    hpp = s["cache_hpp"]
    uses = [u.start() for u in re.finditer(r"\bcpu_job_input_marker_owner_\b", hpp)]
    decl = re.search(r"\bmem_handle\s+cpu_job_input_marker_owner_\s*;", hpp)
    if len(uses) != 1 or not decl or decl.start() > uses[0] or uses[0] >= decl.end():
        fail("W7", f"unified-cache.hpp names cpu_job_input_marker_owner_ {len(uses)} times; only its declaration may: "
                   "inline code there could release the word's allocation without the abandon")


SHUTDOWN_BRANCHES = (
    ("the branch for a SYCL runtime already shutting down",
     r"\bif\s*\(\s*g_sycl_shutting_down\s*\.\s*load\s*\(\s*\)\s*\)\s*\{"),
    ("the branch for an invalid context",
     r"\(\s*void\s*\)\s*queue_\s*\.\s*get_context\s*\(\s*\)\s*;\s*\}\s*catch\s*\(\s*\.\.\.\s*\)\s*\{"),
)
RING_END = r"\bcpu_job_signal_ring_end\s*\(\s*cpu_job_signal_ring_\b"
MARKER_END = r"\bcpu_job_input_marker_end\s*\("


def check_shutdown_branches(cache: str) -> None:
    """W7: each branch of shutdown_resources ends the signal ring once and then the word once."""
    _, shut = definition(cache, SHUTDOWN_SIG, "W7")
    regions = []
    for what, opener in SHUTDOWN_BRANCHES:
        found = list(re.finditer(opener, shut))
        if len(found) != 1:
            fail("W7", f"shutdown_resources has {len(found)} of {what}, not 1")
        body, start = block_at(shut, found[0].end() - 1, "W7")
        regions.append((what, start, body, True))
    drain = list(re.finditer(r"\bif\s*\(\s*!\s*drain_all_queues_noexcept\s*\(\s*\)\s*\)\s*return\s+false\s*;", shut))
    if len(drain) != 1:
        fail("W7", "shutdown_resources does not return false on a failed drain_all_queues_noexcept, once")
    regions.append(("the normal branch, after the queues drained", drain[0].end(), shut[drain[0].end() :], False))
    counted = 0
    for what, start, body, returns in regions:
        rings = [r.start() for r in re.finditer(RING_END, body)]
        ends = [e.start() for e in re.finditer(MARKER_END, body)]
        if len(rings) != 1 or len(ends) != 1:
            fail("W7", f"{what} of shutdown_resources ends the CPU job signal ring {len(rings)} times and the "
                       f"input-marker word {len(ends)} times; each branch ends each once")
        if ends[0] < rings[0]:
            fail("W7", f"{what} of shutdown_resources ends the input-marker word before the signal ring: a job the "
                       "abandon releases would signal a ring still live, a driver call (on a branch where SYCL is "
                       "gone, one that cannot succeed)")
        ret = re.search(r"\breturn\s+true\s*;", body)
        if returns and (not ret or ret.start() < ends[0]):
            fail("W7", f"{what} of shutdown_resources returns before it ends the input-marker word")
        counted += 2
    total = len(re.findall(RING_END, shut)) + len(re.findall(MARKER_END, shut))
    if total != counted:
        fail("W7", f"shutdown_resources ends the signal ring or the input-marker word {total - counted} more times "
                   "outside its three branches: before the drain, a marker still queued would write freed memory")


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


def nospace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check_in_play(s: dict[str, str]) -> None:
    code = s["backend"]
    _, pbody = definition(code, PROLOGUE_SIG, "W10")
    lam = re.search(r"return\s*\[([^\]]*)\]\s*\(\s*\)\s*(?:mutable\s*)?\{", pbody)
    if not lam:
        fail("W10", "moe_cpu_job_prologue does not return a closure")
    outside = pbody[: lam.start()]
    branch = re.search(r"\bif\s*\(\s*!\s*in_play\s*\)\s*\{", outside)
    arm = re.search(ARM, outside)
    if not branch or not arm or arm.start() < branch.start():
        fail("W10", "moe_cpu_job_prologue arms the input marker whatever GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL says: "
                    "with the knob off the marker is not in play, since only the knob's device wait has the hang it "
                    "fixes (llama.cpp-m5v8, llama.cpp-nqbe); the knob is asked first, as `if (!in_play)`, and the arm "
                    "sits in that branch's else")
    head = outside[: branch.start()]
    stmts = [nospace(t) for t in head.split(";") if t.strip()]
    if not stmts or stmts[0] != nospace(IN_PLAY_FIRST[:-1]) or \
            sorted({n for _, n in calls(head)}) != ["moe_cpu_job_input_marker_in_play"] or \
            len(re.findall(r"\bmoe_cpu_job_input_marker_in_play\s*\(", pbody)) != 1:
        fail("W10", "moe_cpu_job_prologue does not ask moe_cpu_job_input_marker_in_play once, into `const bool "
                    "in_play`, in its first statement and before any other call of its own")
    nbody, nstart = block_at(outside, branch.end() - 1, "W10")
    # The else: `else if (...) {...}` or `else {...}`, and nothing after it before the closure.
    rest = outside[nstart + len(nbody) + 1 :]
    els = re.match(r"\s*else\b\s*(?:if\s*\()?", rest)
    tail = None
    if els:
        at = els.end()
        if rest[els.end() - 1] == "(":
            at = matching_paren(rest, els.end() - 1) + 1
        brace = re.match(r"\s*\{", rest[at:])
        if brace:
            eb, es = block_at(rest, at + brace.end() - 1, "W10")
            tail = rest[es + len(eb) + 1 :]
    if nospace(nbody) != nospace(NOT_IN_PLAY_BODY) or tail is None or tail.strip():
        fail("W10", "moe_cpu_job_prologue does more than hand the job's input events to the worker while the marker is "
                    "not in play, or does its marker work outside that branch's else: with the knob off nothing is "
                    "armed, refused, logged or waited on the submitting thread, and the worker waits natively")
    writes = [m.start() for m in re.finditer(r"\bnative\s*(?:=(?!=)|\.\s*\w+\s*\(|\[)", outside)]
    if len(writes) != 1 or not (nstart <= writes[0] < nstart + len(nbody)):
        fail("W10", "moe_cpu_job_prologue fills `native` outside its `if (!in_play)` branch: while the marker is in "
                    "play the worker would be handed events")
    caps = [nospace(c) for c in lam.group(1).split(",")]
    lbody, _ = block_at(pbody, pbody.index("{", lam.end() - 1), "W10")
    if "in_play" not in caps or "native=std::move(native)" not in caps or nospace(lbody) != nospace(CLOSURE_BODY):
        fail("W10", "the closure moe_cpu_job_prologue returns is not exactly `if (in_play)` the marker wait, else the "
                    "native wait on `native`, with in_play captured by copy and native moved in")
    nm, native_body = definition(code, NATIVE_SIG, "W10")
    if nospace(nm.group(1)) != nospace(NATIVE_PARAMS) or nospace(native_body) != nospace(NATIVE_BODY):
        fail("W10", "moe_cpu_job_wait_inputs_native is not the worker's input wait from before the marker (wait the "
                    "events when there are any, then the inputs-ready stamp, then the zeroing, and no watchdog tick)")
    _, ibody = definition(code, IN_PLAY_SIG, "W10")
    if nospace(ibody) != nospace("return moe_cpu_job_device_signal_enabled();"):
        fail("W10", "moe_cpu_job_input_marker_in_play is not exactly the GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL knob: the "
                    "marker would be armed with the knob off, where the hang it fixes cannot happen "
                    "(llama.cpp-m5v8, llama.cpp-nqbe)")


def check_chained_inputs(s: dict[str, str]) -> None:
    _, cbody = definition(s["backend"], CHAINED_SIG, "W9")
    ready = re.search(r"\bif\s*\(\s*record\s*->\s*has_ready\s*\)\s*\{\s*job_inputs\s*\.\s*events\s*\.\s*push_back\s*\(\s*"
                      r"record\s*->\s*ready\s*\)\s*;\s*\}", cbody)
    pro = re.search(r"\bmoe_cpu_job_prologue\s*\(", cbody)
    if not ready or not pro or ready.start() > pro.start():
        fail("W9", "the event-chained hybrid route's job inputs do not take record->ready before its prologue is built: "
                   "a refused arm waits exactly those inputs, and the job would read host weights still being written")


def check(s: dict[str, str]) -> None:
    check_wait_inputs(s)
    check_in_play(s)
    check_pool_lambda(s)
    check_helpers(s)
    check_ring_locks(s)
    check_chained_inputs(s)


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


def chain(*steps):
    """Applies each mutation in turn; the text is unchanged unless every step applied."""
    def apply(text: str) -> str:
        out = text
        for step in steps:
            nxt = step(out)
            if nxt == out:
                return text
            out = nxt
        return out
    return apply


def mutants() -> list:
    marker_wait = r"(\n(\s*)ggml_sycl::cpu_job_input_marker_wait\(marker\);)"
    # The arm's two releases, each followed by the word's creation report: in the catch, then the throw's latch; on
    # the way out, then the return.
    catch_tail = (r"\n\s*cpu_job_input_marker_report_made\(made\.get\(\)\);"
                  r"\n\s*if \(!g_cpu_job_input_marker_submit_threw_logged")
    out_tail = r"\n\s*cpu_job_input_marker_report_made\(made\.get\(\)\);\n\s*return true;"
    tick = r"(ggml_sycl::cpu_job_input_marker_wait\(marker\);)\n(\s*)ggml_sycl_watchdog_heartbeat\(\);"
    arm_call = r"!ggml_sycl::unified_cache_cpu_job_input_marker_arm\(stream, inputs\.events, &marker, &why\)"
    fallback = r"\n\s*sycl::event::wait\(inputs\.events\);"
    site1 = r"\n    std::function<void\(\)> job_prologue = moe_cpu_job_prologue\(\*stream, std::move\(job_inputs\), src0\);"
    in_play_ret = (r"(static bool moe_cpu_job_input_marker_in_play\(\) \{\n\s*)"
                   r"return moe_cpu_job_device_signal_enabled\(\);")
    knob_off = r"\n    if (!in_play) {\n        native = std::move(inputs.events);\n    }"
    any_inputs = r"\n    if (!inputs.events.empty()) {"
    knob_branch = (r"\n    if \(!in_play\) \{\n        native = std::move\(inputs\.events\);\n    \} "
                   r"else if \(!inputs\.events\.empty\(\)\) \{")
    native_open = r"(moe_cpu_job_wait_inputs_native\([^)]*\)\s*\{\n    )"
    native_wait = native_open[:-1] + r"if \(!events\.empty\(\)\) \{\n        sycl::event::wait\(events\);\n    \})"
    end_call = (r"\n(\s*)\{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n"
                r"\s*cpu_job_input_marker_end\(cpu_job_input_marker_word_, cpu_job_input_marker_owner_\);\n\s*\}")
    return [
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
        ("W3", "backend", "the closure captures by default", sub(r"return \[in_play,", r"return [=, in_play,")),
        ("W4", "backend", "the prologue never arms the marker", sub(arm_call, r"true")),
        ("W4", "backend", "the prologue does not take the stream",
         sub(r"moe_cpu_job_prologue\(sycl::queue\s*&\s*stream,\s*", r"moe_cpu_job_prologue(")),
        ("W4", "backend", "the prologue skips the recording test",
         sub(r"!moe_cpu_job_input_marker_may\(stream\) \|\|\s*", "")),
        ("W4", "backend", "a refused arm is not waited", sub(fallback, "")),
        # The waits line counts the refused arm's wait as B2, and a refusal is logged once per reason.
        ("W4", "backend", "a refused arm's wait is not counted", sub(r"\n\s*moe_hostpath_wait_timer wait_timer\(MOE_WAIT_B2\);", "")),
        ("W4", "backend", "a refused arm's wait is counted as B1",
         sub(r"moe_hostpath_wait_timer wait_timer\(MOE_WAIT_B2\);", r"moe_hostpath_wait_timer wait_timer(MOE_WAIT_B1);")),
        ("W4", "backend", "a refused arm's timer ends before its wait",
         sub(r"\n(\s*)(moe_hostpath_wait_timer wait_timer\(MOE_WAIT_B2\);)", r"\n\1{\n\1    \2\n\1}")),
        ("W4", "backend", "one latch for every refusal reason",
         chain(sub(r"g_moe_cpu_job_input_marker_refusal_logged\[ggml_sycl::CPU_JOB_INPUT_MARKER_REFUSAL_COUNT\] = \{\};",
                   r"g_moe_cpu_job_input_marker_refusal_logged{ false };"),
               sub(r"g_moe_cpu_job_input_marker_refusal_logged\[why\]\.exchange", r"g_moe_cpu_job_input_marker_refusal_logged.exchange"))),
        ("W4", "backend", "every refused job is logged",
         sub(r"if \(!g_moe_cpu_job_input_marker_refusal_logged\[why\]\.exchange\(true, std::memory_order_acq_rel\)\)",
             r"if (true)")),
        ("W4", "backend", "the prologue also waits the inputs when the marker is armed",
         sub(r"(\n(\s*)return \[in_play,)", r"\n\2sycl::event::wait(inputs.events);\1")),
        # With the knob off the worker waits natively: the submitting thread waiting the inputs there, as the refused
        # arm's B2 fallback does, is not the path from before the marker.
        ("W4", "backend", "the prologue waits the inputs itself while the marker is not in play",
         sub(r"(\n        native = std::move\(inputs\.events\);)", r"\n        sycl::event::wait(inputs.events);\1")),
        ("W4", "backend", "the recording test misses a recording stream",
         sub(r"return ggml_sycl_graph_recording_active\(\) \|\|\s*stream\.ext_oneapi_get_state\(\) == "
             r"sycl::ext::oneapi::experimental::queue_state::recording;", r"return ggml_sycl_graph_recording_active();")),
        ("W4", "backend", "the recording test misses the backend's flag",
         sub(r"return ggml_sycl_graph_recording_active\(\) \|\|\s*stream\.", r"return stream.")),
        ("W4", "backend", "the marker predicate skips the recording test",
         sub(r"(static bool moe_cpu_job_input_marker_may\(sycl::queue & stream\) \{\n\s*)"
             r"return !moe_cpu_job_stream_recording\(stream\);", r"\1return true;")),
        # W10: the marker follows the knob (llama.cpp-nqbe): the predicate, the order of the knob and the arm, the
        # knob-off branch, the closure's two waits and the native wait from before the marker.
        ("W10", "backend", "the marker is in play whatever the knob",
         sub(in_play_ret, r"\1return true;")),
        ("W10", "backend", "the marker is in play only with the knob off",
         sub(in_play_ret, r"\1return !moe_cpu_job_device_signal_enabled();")),
        ("W10", "backend", "the knob is asked after an arm attempt",
         chain(sub(knob_branch, any_inputs), sub(r"(\n    return \[in_play,)", knob_off + r"\1"))),
        ("W10", "backend", "the prologue drops the knob's branch", sub(knob_branch, any_inputs)),
        ("W10", "backend", "the prologue asks the stream's device-signal test instead of the knob",
         sub(r"in_play = moe_cpu_job_input_marker_in_play\(\);", r"in_play = moe_cpu_job_device_signal_may(stream);")),
        ("W10", "backend", "the knob-off branch logs a refusal",
         sub(r"(\n        native = std::move\(inputs\.events\);)",
             r'\1\n        GGML_LOG_WARN("[MOE-INPUT-MARKER] its input marker was not armed (%s)\\n", "off");')),
        ("W10", "backend", "the worker is handed events while the marker is in play",
         sub(r"(\} else if \(!inputs\.events\.empty\(\)\) \{)", r"\1\n        native = inputs.events;")),
        ("W10", "backend", "the closure copies the input events instead of moving native in",
         sub(r"return \[in_play, native = std::move\(native\),", r"return [in_play, native = inputs.events,")),
        ("W10", "backend", "the closure's waits are swapped",
         sub(r"\n        if \(in_play\) \{\n            moe_cpu_job_wait_inputs\(marker",
             r"\n        if (!in_play) {\n            moe_cpu_job_wait_inputs(marker")),
        ("W10", "backend", "the closure waits natively whatever the knob",
         sub(r"\n        if \(in_play\) \{\n            moe_cpu_job_wait_inputs\(marker, zero_out, zero_bytes\);"
             r"\n        \} else \{\n            (moe_cpu_job_wait_inputs_native\(native, zero_out, zero_bytes\);)"
             r"\n        \}",
             r"\n        \1")),
        ("W10", "backend", "the closure captures in_play by reference",
         sub(r"return \[in_play,", r"return [&in_play,")),
        ("W10", "backend", "the native wait ticks the watchdog",
         sub(native_wait, r"\1\n    ggml_sycl_watchdog_heartbeat();")),
        ("W10", "backend", "the native wait skips one-event jobs",
         sub(native_open + r"if \(!events\.empty\(\)\)", r"\1if (events.size() > 1)")),
        ("W10", "backend", "the native wait zeroes before the inputs land",
         sub(native_open, r"\1if (zero_out && zero_bytes > 0) {\n        std::memset(zero_out, 0, zero_bytes);"
                          r"\n    }\n    ")),
        ("W5", "backend", "a site builds its prologue without the stream",
         sub(r"moe_cpu_job_prologue\(\*stream, std::move\(job_inputs\), src0\)",
             r"moe_cpu_job_prologue(std::move(job_inputs), src0)")),
        ("W5", "backend", "the event-chained route arms after its device half",
         move(site1, r"\n\s*moe_hostpath_waits_readback\(record->layer,")),
        ("W6", "pool", "the job lambda waits an event",
         sub(r"(\n(\s*)if \(prologue\) \{)", r"\n\2sycl::event().wait();\1")),
        ("W6", "pool", "the job lambda queries the driver",
         sub(r"(\n(\s*)if \(prologue\) \{)", r"\n\2(void) zeEventQueryStatus(nullptr);\1")),
        # The decode route's resolve step runs on the pool worker after the prologue and before the job signals: a
        # status query or a wait there can take the driver's device lock behind this very job.
        ("W6", "backend", "the decode resolve step queries an event's status",
         sub(r"(\n(\s*)if \(record_warmup\) \{\n\s*g_moe_warmup\.record\()",
             r"\n\2(void) record->ready.get_info<command_execution_status>();\1")),
        ("W6", "backend", "the decode resolve step waits an event",
         sub(r"(\n(\s*)if \(record_warmup\) \{\n\s*g_moe_warmup\.record\()", r"\n\2record->ready.wait();\1")),
        ("W6", "backend", "the decode resolve step calls Level Zero",
         sub(r"(\n(\s*)if \(record_warmup\) \{\n\s*g_moe_warmup\.record\()", r"\n\2(void) zeEventQueryStatus(nullptr);\1")),
        # The decode route's site stays as it is, so the scan is not vacuous; a second site hands over a resolve step
        # through a receiver that is a call.
        ("W6", "backend", "a resolve step handed to submit_batch on a call expression queries the driver",
         sub(r"cpu_pool\.submit_batch\(std::move\(batch_tasks\), \{\}, \{\},",
             r"moe_cpu_pool_of(ctx).submit_batch(std::move(batch_tasks), {}, "
             r"[]() { (void) zeEventQueryStatus(nullptr); },")),
        # A refused arm waits exactly the job's inputs on the submitting thread: without the leases' barrier among
        # them, the job would read host weights still being written.
        ("W9", "backend", "the decode route's job does not wait the leases' barrier",
         sub(r"\n\s*job_inputs\.events\.push_back\(record->ready\);", "")),
        ("W9", "backend", "the decode route takes the leases' barrier after its prologue",
         move(r"\n    if \(record->has_ready\) \{\n\s*job_inputs\.events\.push_back\(record->ready\);\n    \}",
              r"\n    if \(!record->all_host\) \{\n\s*const sycl::event table_h2d")),
        ("W7", "header", "the marker wait sleeps",
         sub(r"(\n        \}\n        )std::this_thread::yield\(\);(\n    \}\n\})",
             r"\1std::this_thread::sleep_for(std::chrono::microseconds(100));\2")),
        ("W7", "header", "the marker wait ticks the watchdog",
         sub(r"(\n(\s*)\+\+spins;)", r"\1\n\2ggml_sycl_watchdog_heartbeat();")),
        ("W7", "header", "the marker wait takes a lock",
         sub(r"(\n(\s*)\+\+spins;)", r"\1\n\2std::lock_guard<std::mutex> hold(g_marker_mutex);")),
        ("W7", "header", "the marker wait enters SYCL",
         sub(r"(\n(\s*)\+\+spins;)", r"\1\n\2(void) sycl::event().get_info<sycl::info::event::command_execution_status>();")),
        ("W7", "header", "seen reads the word before abandoned",
         sub(r"w\.abandoned\.load\(std::memory_order_seq_cst\) \|\|\n(\s*)cpu_job_input_marker_reached\(__atomic_load_n\(w\.ptr, "
             r"__ATOMIC_ACQUIRE\), m\.serial\);",
             r"cpu_job_input_marker_reached(__atomic_load_n(w.ptr, __ATOMIC_ACQUIRE), m.serial) ||\n\1"
             r"w.abandoned.load(std::memory_order_seq_cst);")),
        ("W7", "header", "seen reads the word relaxed",
         sub(r"__atomic_load_n\(w\.ptr, __ATOMIC_ACQUIRE\)", r"__atomic_load_n(w.ptr, __ATOMIC_RELAXED)")),
        # The word's allocation never goes under a load: a waiter counts itself before it tests the flag and loads the
        # word, and the abandon sets the flag and then waits for that count.
        ("W7", "header", "seen does not count itself in readers",
         sub(r"\n\s*w\.readers\.fetch_add\(1, std::memory_order_seq_cst\);", "")),
        ("W7", "header", "seen uncounts itself before its load",
         move(r"\n\s*w\.readers\.fetch_sub\(1, std::memory_order_release\);", r"\n\s*const bool seen = ")),
        ("W7", "header", "seen tests abandoned with acquire only",
         sub(r"w\.abandoned\.load\(std::memory_order_seq_cst\)", r"w.abandoned.load(std::memory_order_acquire)")),
        ("W7", "header", "the abandon does not wait for the readers",
         sub(r"\n\s*while \(w\.readers\.load\(std::memory_order_seq_cst\) != 0\) \{\n\s*std::this_thread::yield\(\);\n\s*\}", "")),
        ("W7", "header", "the abandon waits before it sets the flag",
         sub(r"(\n\s*w\.abandoned\.store\(true, std::memory_order_seq_cst\);)(\n\s*while \(w\.readers\.load\("
             r"std::memory_order_seq_cst\) != 0\) \{\n\s*std::this_thread::yield\(\);\n\s*\})", r"\2\1")),
        ("W7", "header", "the abandon sets the flag with release only",
         sub(r"w\.abandoned\.store\(true, std::memory_order_seq_cst\)", r"w.abandoned.store(true, std::memory_order_release)")),
        # The word is compared modulo 2^32: any other compare releases a job early at the wrap, or never.
        ("W7", "header", "the compare does not wrap",
         sub(r"return static_cast<int32_t>\(seen - serial\) >= 0;", r"return seen >= serial;")),
        ("W7", "header", "the compare is off by one",
         sub(r"return static_cast<int32_t>\(seen - serial\) >= 0;", r"return static_cast<int32_t>(seen - serial) > 0;")),
        # With the SYCL watchdog off, a marker that never arrives is a silent spin unless the wait reports it, once.
        ("W7", "header", "a slow wait is reported on every pass", sub(r"if \(!told && m\.word->slow\) \{", r"if (m.word->slow) {")),
        ("W7", "header", "a slow wait is never reported",
         sub(r"\n\s*m\.word->slow\(\*m\.word, m\.serial, std::chrono::duration<double>\(now - since\)\.count\(\)\);", "")),
        ("W7", "header", "a slow wait is reported at once", sub(r"\} else if \(now - since >= m\.word->slow_after\) \{", r"} else {")),
        ("W7", "cache", "the device task does not depend on the inputs", sub(r"\n\s*cgh\.depends_on\(after\);", "")),
        ("W7", "cache", "the device task stores before its fence",
         move(r"\n\s*\*word = serial;", r"\n\s*sycl::atomic_fence\(sycl::memory_order::release, sycl::memory_scope::system\);")),
        ("W7", "cache", "the fence has device scope only",
         sub(r"sycl::atomic_fence\(sycl::memory_order::release, sycl::memory_scope::system\)",
             r"sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device)")),
        ("W7", "cache", "the serial is submitted outside the lock",
         sub(r"\n(\s*)std::unique_lock<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n",
             r"\n\1{\n\1    std::unique_lock<std::mutex> lock(cpu_job_input_marker_mutex_);\n\1}\n")),
        ("W7", "cache", "the serial is consumed before the submit",
         move(r"\n\s*cpu_job_input_marker_serial_ = serial;", r"\n\s*try \{\n\s*cpu_job_input_marker_submit\(")),
        ("W7", "cache", "the word is not host-pinned",
         sub(r"(cohort_id\s*= \"cpu_job_input_marker\";\n\s*)req\.intent\.constraints\.must_host_pinned = true;",
             r"\1req.intent.constraints.must_device = true;")),
        ("W7", "cache", "a teardown branch keeps the word", sub(end_call, "")),
        # Each of these lets a job pass on a marker that does not follow its inputs: a serial that does not advance, a
        # word not zeroed before a job sees it, an out-of-order stream, or a marker on another queue or cache.
        ("W7", "cache", "every arm takes serial 1",
         sub(r"const uint32_t serial = cpu_job_input_marker_serial_ \+ 1;", r"const uint32_t serial = 1;")),
        ("W7", "cache", "the word is not zeroed", sub(r"\n\s*\*word->ptr\s*= 0;", "")),
        ("W7", "cache", "the word is zeroed after it is published",
         move(r"\n\s*\*word->ptr\s*= 0;", r"\n\s*\}\n\s*const uint32_t serial = cpu_job_input_marker_serial_ \+ 1;")),
        ("W7", "cache", "an out-of-order stream is not refused",
         sub(r"\n\s*if \(!queue_\.has_property<sycl::property::queue::in_order>\(\)\) \{\n\s*"
             r"\*why = CPU_JOB_INPUT_MARKER_REFUSAL_NOT_IN_ORDER;[^}]*\}", "")),
        # A throwing submit is logged with the mutex released, once per process, and every refusal names its reason.
        ("W7", "cache", "the throw is logged with the mutex held",
         sub(r"\n\s*lock\.unlock\(\);(" + catch_tail + r")", r"\1")),
        ("W7", "cache", "the throw is logged before the mutex is released",
         move(r"\n\s*lock\.unlock\(\);(?=" + catch_tail + r")",
              r"\n\s*\*why = CPU_JOB_INPUT_MARKER_REFUSAL_SUBMIT_THREW;")),
        ("W7", "cache", "every throw is logged",
         sub(r"if \(!g_cpu_job_input_marker_submit_threw_logged\.exchange\(true, std::memory_order_acq_rel\)\)", r"if (true)")),
        ("W7", "cache", "the arm releases its mutex before the submit",
         move(r"\n\s*lock\.unlock\(\);(?=" + catch_tail + r")",
              r"\n\s*try \{\n\s*cpu_job_input_marker_submit\(")),
        ("W7", "cache", "the arm reads a member before it locks",
         sub(r"(\n(\s*)std::unique_lock<std::mutex> lock\(cpu_job_input_marker_mutex_\);)",
             r"\n\2(void) cpu_job_input_marker_serial_;\1")),
        ("W7", "cache", "the catch reads a member after the release",
         sub(r"(\n(\s*)lock\.unlock\(\);)(" + catch_tail + r")",
             r"\1\n\2cpu_job_input_marker_tried_ = false;\3")),
        # The word's creation is logged after a release too, through the report, with the word this arm made.
        ("W7", "cache", "the word's creation is logged with the mutex held",
         chain(sub(r"\n\s*made\s*= word;", ""),
               sub(r"(\n(\s*)cpu_job_input_marker_word_  = std::move\(word\);)",
                   r'\1\n\2GGML_LOG_INFO("[MOE-INPUT-MARKER] CPU expert jobs wait their inputs on a '
                   r'device-written word on device %d (llama.cpp-42zj)\\n", device);'))),
        ("W7", "cache", "the way out reports the word before the release",
         move(r"\n\s*lock\.unlock\(\);(?=" + out_tail + r")",
              r"(?<=cpu_job_input_marker_report_made\(made\.get\(\)\);)\n\s*return true;")),
        ("W7", "cache", "the catch reports the word before the release",
         sub(r"(\n\s*lock\.unlock\(\);)(\n\s*cpu_job_input_marker_report_made\(made\.get\(\)\);)"
             r"(?=\n\s*if \(!g_cpu_job_input_marker_submit_threw_logged)", r"\2\1")),
        ("W7", "cache", "the way out reads a member after the release",
         sub(r"(\n(\s*)lock\.unlock\(\);)(" + out_tail + r")",
             r"\1\n\2(void) cpu_job_input_marker_serial_;\3")),
        ("W7", "cache", "the way out does not report the word",
         sub(r"\n\s*cpu_job_input_marker_report_made\(made\.get\(\)\);(?=\n\s*return true;)", "")),
        ("W7", "cache", "the report is never handed the word", sub(r"\n\s*made\s*= word;", "")),
        ("W7", "cache", "the way out reports the word twice",
         sub(r"(\n\s*cpu_job_input_marker_report_made\(made\.get\(\)\);)(?=\n\s*return true;)", r"\1\1")),
        # A catch that falls through reaches the way out: a second release of a released lock, and the serial and
        # *out written without the mutex.
        ("W7", "cache", "the catch falls through to the way out",
         sub(r"(\n\s*\*why = CPU_JOB_INPUT_MARKER_REFUSAL_SUBMIT_THREW;)\n\s*return false;", r"\1")),
        ("W7", "cache", "the report logs more than its INFO",
         sub(r"(\n(\s*)w->device\);)", r"\1\n\2(void) sycl::queue{};")),
        ("W7", "cache", "a failed allocation does not name its reason",
         sub(r"\n\s*\*why = CPU_JOB_INPUT_MARKER_REFUSAL_ALLOC_FAILED;", "")),
        ("W7", "cache", "a stream that is not a cache's does not name its reason",
         sub(r"\n\s*\*why = CPU_JOB_INPUT_MARKER_REFUSAL_NOT_CACHE_STREAM;", "")),
        ("W7", "cache", "the marker goes on the DMA queue",
         sub(r"cpu_job_input_marker_submit\(queue_,", r"cpu_job_input_marker_submit(*dma_queue_,")),
        ("W7", "cache", "the arm reaches device 0's cache",
         sub(r"unified_cache \* cache = moe_join_queue_cache_of\(stream\);\n(\s*)if \(!cache\) \{\n(\s*)\*why",
             r"unified_cache * cache = get_existing_unified_cache_for_device(0);\n\1if (!cache) {\n\2*why")),
        ("W7", "cache", "the stream's cache is any cache of its device",
         sub(r"return cache && cache->get_queue\(\) == q \? cache : nullptr;", r"return cache;")),
        # The cache's half of the drain and of the slow-wait report: the end waits out the readers, and the arm
        # installs a reporter that only logs.
        ("W7", "cache", "the end sets the flag without draining the readers",
         sub(r"cpu_job_input_marker_abandon\(\*word\);", r"word->abandoned.store(true, std::memory_order_release);")),
        ("W7", "cache", "the slow-wait reporter is not installed",
         sub(r"\n\s*word->slow\s*= cpu_job_input_marker_report_slow;", "")),
        ("W7", "cache", "the slow-wait reporter queries the driver",
         sub(r"(\n(\s*)GGML_LOG_WARN\(\n\s*\"\[MOE-INPUT-MARKER\] CPU expert job %u on device %d)",
             r"\n\2(void) zeEventQueryStatus(nullptr);\1")),
        ("W7", "cache", "the word goes before the drain",
         move(r"\n    \{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n"
              r"\s*cpu_job_input_marker_end\(cpu_job_input_marker_word_, cpu_job_input_marker_owner_\);\n    \}",
              r"\n\s*if \(!drain_all_queues_noexcept\(\)\) return false;")),
        # Comments are blanked, so these anchor on code: an 8-space block is in a SYCL-gone branch, the first one in
        # the branch for a runtime already shutting down; the normal branch's blocks are at 4 spaces.
        ("W7", "cache", "the SYCL-gone branch's word end moves to the normal branch",
         move(r"\n        \{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n"
              r"\s*cpu_job_input_marker_end\(cpu_job_input_marker_word_, cpu_job_input_marker_owner_\);\n        \}",
              r"\n    \{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);")),
        ("W7", "cache", "the SYCL-gone branch ends the word before the ring",
         sub(r"(\n        \{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_signal_mutex_\);\n"
             r"\s*cpu_job_signal_ring_end\([^;]*\);\n        \})(\s*)"
             r"(\n        \{\n\s*std::lock_guard<std::mutex> lock\(cpu_job_input_marker_mutex_\);\n"
             r"\s*cpu_job_input_marker_end\([^;]*\);\n        \})", r"\3\2\1")),
        ("W7", "cache", "a shutdown branch releases the owner beside the other owners",
         sub(r"(\n(        )staging_                          = nullptr;\n\s*staging_owner_\s*= \{\};)",
             r"\1\n\2cpu_job_input_marker_owner_ = {};")),
        ("W7", "cache_hpp", "inline code in the header releases the owner",
         sub(r"(\n(\s*)mem_handle(\s+)cpu_job_input_marker_owner_;)",
             r"\1\n\2void drop_cpu_job_input_marker_owner() { cpu_job_input_marker_owner_ = {}; }")),
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


def self_test(s: dict[str, str]) -> int:
    failures, caught = [], 0
    for pin, key, name, mutate in mutants():
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
    print("PASS: with GGML_SYCL_MOE_CPU_JOB_DEVICE_SIGNAL on, the CPU expert pool worker waits a job's inputs on a "
          "device-written marker with host loads and makes no driver call but the job's device signal; with it off no "
          f"marker is armed and the worker waits them natively, as before the marker {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
