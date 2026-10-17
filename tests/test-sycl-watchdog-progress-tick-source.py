#!/usr/bin/env python3
"""The [SYCL-WATCHDOG] ticks where work completes inside a graph (llama.cpp-ff00).

THE DEFECT.  The watchdog (common.cpp) judges one host timestamp, and only while a graph
or a load-time event wait is in flight.  A prompt graph ticked it once, at
record_completion, and its exit is a whole-queue wait.  So a prompt graph whose wall
exceeded GGML_SYCL_OP_TIMEOUT_MS (default 30 s) was stopped with _Exit(1) while its device
kept finishing work: Qwen3.5-9B at -d 32768 on both cards (2048- and 1024-token graphs at
depth), and Qwen3.8 at -ub 256 on the B70 (a 256-token MoE prompt graph with host-tiered
experts).

THE CONTRACT.  With comments blanked:
  W1. (ggml-sycl.cpp ggml_sycl_trace_queue_wait, the funnel of the graph code's whole-queue
      waits) every q->wait() is followed by ggml_sycl_watchdog_heartbeat() before the next
      wait or the end of its branch, and no tick precedes a wait: the calls read exactly
      wait, tick, wait, tick.  The untraced branch ticks after its gate clear and before
      its return.  A wait that never returns therefore never ticks, which is the hang the
      watchdog exists for.
  W2. (cpu-dispatch.cpp cpu_expert_mul_mat_batched_run, which every CPU expert producer
      runs: the pool, std::async, inline) the last statement is the tick, after both row
      executors, and it is the function's only tick.  So it ticks once per batch, never
      inside rows_body, the hot loop.
  W3. (ggml_sycl_cpu_expert_mul_mat_adaptive) the reduced-precision loop, which does not
      go through W2's function, ticks after it when it ran.
  W4. No polling and no knob.  watchdog_thread_fn waits on nothing and queries no event,
      the heartbeat stays one relaxed store, and neither W1's nor W2's function gains a
      getenv.
  W5. The catalog row of GGML_SYCL_OP_TIMEOUT_MS names the tick points and the residual
      limit.  The banner keeps its first line, which scripts count, and no longer claims
      a hung submit is the likely cause.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources; each must
fail on its own pin.  On the unmodified base the gate fails W1.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = "(llama.cpp-ff00)"
BACKEND = "ggml/src/ggml-sycl/ggml-sycl.cpp"
DISPATCH = "ggml/src/ggml-sycl/cpu-dispatch.cpp"
COMMON = "ggml/src/ggml-sycl/common.cpp"
DOC = "docs/backend/sycl-env-vars.md"

TICK = "ggml_sycl_watchdog_heartbeat();"
QUEUE_WAIT_SIG = r"static\s+void\s+ggml_sycl_trace_queue_wait\s*\(\s*queue_ptr\s+q\s*,[^)]*\)\s*\{"
RUN_SIG = r"static\s+void\s+cpu_expert_mul_mat_batched_run\s*\([^)]*\)\s*\{"
ADAPTIVE_SIG = r"void\s+ggml_sycl_cpu_expert_mul_mat_adaptive\s*\([^)]*\)\s*\{"
THREAD_SIG = r"static\s+void\s+watchdog_thread_fn\s*\([^)]*\)\s*\{"
HEARTBEAT_SIG = r"void\s+ggml_sycl_watchdog_heartbeat\s*\(\s*\)\s*\{"
HEARTBEAT_BODY = "g_watchdog_last_tick_ns.store(ggml_sycl_watchdog_now_ns(), std::memory_order_relaxed);"
BANNER_FIRST = "[SYCL-WATCHDOG] No GPU progress for %lld ms (timeout %lld ms, %d devices known).\\n"
BANNER_OLD = "Most likely a SYCL submit is hung"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


_BLANK = re.compile(r'//[^\n]*|/\*.*?(?:\*/|\Z)|"(?:\\.|[^"\\])*(?:"|\Z)|(?<![^\W_])\'(?:\\.|[^\'\\])*(?:\'|\Z)', re.S)


def blank_comments(src: str) -> str:
    """Blank // and /* */ comments, keeping string literals, line breaks and offsets."""
    def one(m: re.Match) -> str:
        t = m.group(0)
        if t[0] != "/":
            return t
        return " " * len(t) if t[1] == "/" else re.sub(r"[^\n]", " ", t)
    return _BLANK.sub(one, src)


def strip_strings(src: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def body_of(code: str, signature: str, pin: str) -> str:
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    brace = code.index("{", m.end() - 1)
    shape = strip_strings(code)
    depth = 0
    for j in range(brace, len(shape)):
        if shape[j] == "{":
            depth += 1
        elif shape[j] == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1 : j]
    fail(pin, "unbalanced braces")
    return ""


def flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_queue_wait(backend: str) -> None:
    body = flat(body_of(blank_comments(backend), QUEUE_WAIT_SIG, "W1"))
    calls = [m.group(0) for m in re.finditer(r"\bq->wait\(\);|\bggml_sycl_watchdog_heartbeat\(\);", body)]
    if calls != ["q->wait();", TICK, "q->wait();", TICK]:
        fail("W1", f"the graph code's queue waits do not each tick the watchdog after they return, and only then "
                   f"(calls in order: {calls})")
    if not re.search(r"if \(!enabled\) \{ [^{}]*q->wait\(\); ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\); "
                     + re.escape(TICK) + r" return; \}", body):
        fail("W1", "the untraced queue wait does not tick after its gate clear and before its return")


def check_batched(dispatch: str) -> None:
    code = blank_comments(dispatch)
    run = flat(body_of(code, RUN_SIG, "W2"))
    if run.count(TICK) != 1 or not run.endswith(TICK):
        fail("W2", "the batched row loop does not end with its one watchdog tick (once per batch, not per row)")
    last_executor = max(run.rfind("row_workers->run("), run.rfind("parallel_for("))
    if last_executor < 0 or run.rfind(TICK) < last_executor:
        fail("W2", "the batched row loop ticks before its row executors ran")
    adaptive = flat(body_of(code, ADAPTIVE_SIG, "W3"))
    want = ("for (int i = 0; i < n_reduced; i++) { cpu_expert_mul_mat_int4(tasks[n_full + i]); } "
            "if (n_reduced > 0) { " + TICK + " }")
    if want not in adaptive or adaptive.count(TICK) != 1:
        fail("W3", "the reduced-precision expert loop does not tick once after it ran")


def check_no_polling(backend: str, dispatch: str, common: str) -> None:
    thread = flat(strip_strings(body_of(blank_comments(common), THREAD_SIG, "W4")))
    if re.search(r"\.wait\s*\(|wait_and_throw|get_info\s*<|ext_oneapi_|get_status|command_execution_status", thread):
        fail("W4", "the watchdog thread waits on or queries the device; it may only read the tick")
    if flat(body_of(blank_comments(common), HEARTBEAT_SIG, "W4")) != HEARTBEAT_BODY:
        fail("W4", "the heartbeat is no longer one relaxed store of the clock")
    qw = body_of(blank_comments(backend), QUEUE_WAIT_SIG, "W4")
    if len(re.findall(r"\bgetenv\s*\(", qw)) != 1:
        fail("W4", "the queue wait reads an environment variable besides GGML_SYCL_WAIT_TRACE")
    if re.search(r"\bgetenv\s*\(", body_of(blank_comments(dispatch), RUN_SIG, "W4")):
        fail("W4", "the batched row loop reads an environment variable")


def check_doc_and_banner(doc: str, common: str) -> None:
    rows = [ln for ln in doc.splitlines() if ln.startswith("| `GGML_SYCL_OP_TIMEOUT_MS=<N>`")]
    need = ("`ggml_sycl_trace_queue_wait`", "CPU expert batch", "llama.cpp-ff00", "command graph", "llama.cpp-4lnb")
    if len(rows) != 1 or not all(n in rows[0] for n in need):
        fail("W5", "the catalog row of GGML_SYCL_OP_TIMEOUT_MS does not name the tick points and the residual limit")
    if BANNER_FIRST not in common:
        fail("W5", "the banner's first line changed; scripts count it")
    if BANNER_OLD in common:
        fail("W5", "the banner still says a hung submit is the likely cause; the watchdog cannot know that")


def read_sources(root: Path) -> dict:
    return {k: (root / p).read_text(encoding="utf-8") for k, p in
            (("backend", BACKEND), ("dispatch", DISPATCH), ("common", COMMON), ("doc", DOC))}


def check(s: dict) -> None:
    check_queue_wait(s["backend"])
    check_batched(s["dispatch"])
    check_no_polling(s["backend"], s["dispatch"], s["common"])
    check_doc_and_banner(s["doc"], s["common"])


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    untraced = r"(ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\);\n)(\s*)ggml_sycl_watchdog_heartbeat\(\);\n(\s*return;)"
    # The untraced branch's head inside ggml_sycl_trace_queue_wait; other functions also have `if (!enabled) {`.
    QW_UNTRACED_HEAD = (r"(if \(!enabled\) \{\n)(\s*const ggml_sycl_cpu_job_gate_sleep gate_sleep = "
                        r"ggml_sycl_cpu_job_gates_sleep\(\*q\);\n)")
    traced_tick = r"(const auto\s+t1\s*=\s*std::chrono::high_resolution_clock::now\(\);\n)\s*ggml_sycl_watchdog_heartbeat\(\);\n"
    mutants = [
        ("untraced wait no tick", "backend", sub(untraced, r"\1\2\3"), "W1"),
        ("traced wait no tick", "backend", sub(traced_tick, r"\1"), "W1"),
        ("traced tick before its wait", "backend",
         lambda t: re.sub(r"(\n(\s*)const ggml_sycl_cpu_job_gate_sleep gate_sleep = ggml_sycl_cpu_job_gates_sleep\(\*q\);\n"
                          r"\s*q->wait\(\);\n\s*ggml_sycl_cpu_job_gates_clear_queue_wait\(gate_sleep\);\n"
                          r"\s*const auto\s+t1)", r"\n\2ggml_sycl_watchdog_heartbeat();\1",
                          re.sub(traced_tick, r"\1", t, count=1), count=1), "W1"),
        ("untraced tick before its wait", "backend",
         lambda t: re.sub(QW_UNTRACED_HEAD, r"\1        ggml_sycl_watchdog_heartbeat();\n\2",
                          re.sub(untraced, r"\1\2\3", t, count=1), count=1), "W1"),
        ("batched loop no tick", "dispatch",
         sub(r"\n\s*ggml_sycl_watchdog_heartbeat\(\);\n\}\n(\s*void ggml_sycl_cpu_expert_mul_mat_batched\()", r"\n}\n\1"),
         "W2"),
        ("batched loop ticks per row", "dispatch",
         lambda t: re.sub(r"(int range_begin, int range_end, int slot\) \{\n)", r"\1        ggml_sycl_watchdog_heartbeat();\n",
                          t, count=1), "W2"),
        ("batched loop ticks before its executors", "dispatch",
         lambda t: re.sub(r"(\n\s*if \(row_workers\) \{)", "\n    ggml_sycl_watchdog_heartbeat();\\1",
                          re.sub(r"\n\s*ggml_sycl_watchdog_heartbeat\(\);\n\}\n(\s*void ggml_sycl_cpu_expert_mul_mat_batched\()",
                                 r"\n}\n\1", t, count=1), count=1), "W2"),
        ("reduced loop no tick", "dispatch",
         sub(r"\n\s*if \(n_reduced > 0\) \{\n\s*ggml_sycl_watchdog_heartbeat\(\);[^\n]*\n\s*\}", ""), "W3"),
        ("watchdog thread polls an event", "common",
         sub(r"(static void watchdog_thread_fn\(int64_t timeout_ms\) \{\n)", r"\1    g_last_event.wait();\n"), "W4"),
        ("heartbeat is not one store", "common",
         sub(r"(void ggml_sycl_watchdog_heartbeat\(\) \{\n)", r"\1    ggml_sycl_watchdog_poll_devices();\n"), "W4"),
        ("queue wait gains a knob", "backend",
         sub(QW_UNTRACED_HEAD, r"\1        static const bool tick = std::getenv(\"GGML_SYCL_X\") != nullptr;\n\2"),
         "W4"),
        ("catalog row not updated", "doc", sub(r" Progress is a host tick, judged only[^|]*\(llama\.cpp-4lnb\)\.", ""), "W5"),
        ("banner first line changed", "common", sub(r"No GPU progress for %lld ms", "No progress for %lld ms"), "W5"),
        ("banner still blames a hung submit", "common",
         sub(r"No queue wait of the graph code returned and no CPU expert batch finished", BANNER_OLD), "W5"),
    ]
    bad = 0
    for name, key, mutate, pin in mutants:
        m = dict(s)
        m[key] = mutate(s[key])
        if m[key] == s[key]:
            print(f"  self-test: mutant '{name}' did not apply")
            bad += 1
            continue
        try:
            check(m)
        except ContractError as e:
            if f"[pin {pin}]" in str(e):
                print(f"  self-test: mutant '{name}' caught: {e}")
                continue
            print(f"  self-test: mutant '{name}' failed on the wrong pin: {e}")
            bad += 1
            continue
        print(f"  self-test: mutant '{name}' NOT caught")
        bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    s = read_sources(args.root)
    try:
        check(s)
    except ContractError as e:
        print(e)
        return 1
    if args.self_test:
        bad = self_test(s)
        if bad:
            print(f"FAIL: {bad} self-test mutant(s) not caught on their own pin {TAG}")
            return 1
    print(f"PASS: the watchdog ticks after each returning queue wait and each CPU expert batch {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
