#!/usr/bin/env python3
"""A CpuExpertPool job's row loop runs on the pool's own row workers (llama.cpp-w4e9).

THE CONTRACT.  Pool jobs ran their row loop (Phase 2 of
ggml_sycl_cpu_expert_mul_mat_batched) on the shared TBB arena.  Between the ~72
jobs of a decode token TBB returns the arena's workers to its sleeping pool, so
a job found 6-9 of 22 threads awake.  Each pool now owns a CpuRowWorkers set
that keeps polling for a bounded spin budget after a job (GGML_SYCL_CPU_ROW_SPIN_US,
default 2000 us, 0 blocks at once), and every poll yields the core so another
pool's runnable thread is not kept waiting.  The unit test test-cpu-row-workers
checks the set itself; this gate checks the wiring that decides whether pool
jobs use it at all, the knob, and the bounds that keep an idle process from
polling.

WHAT IT CHECKS, with comments blanked:
  P1  CpuExpertPool holds a `CpuRowWorkers row_workers_` member.
  P2  init() starts it with ggml_sycl_cpu_row_loop_threads(), the TBB arena's
      size (GGML_SYCL_CPU_THREADS), not a count of its own, and with
      get_cpu_row_spin_us(), not a budget of its own.
  P3  stop_workers() stops it, after its own workers are joined (a running job
      uses the row workers until it returns).
  P4  the pool's one kernel call passes row_workers_.
  P5  get_cpu_row_spin_us() (cpu-expert-pool.cpp) is the one reader of
      GGML_SYCL_CPU_ROW_SPIN_US; it rejects a negative value, falls back to
      CpuRowWorkers::spin_us_default on every early return, and caps the value
      at spin_us_max.
  D1  in cpu-dispatch.cpp the row loop is one body, rows_body, that the row
      workers run when given and the TBB arena runs otherwise; the arena path
      passes its slot index, and the row workers take the body by std::cref
      (no copy, no allocation per job).
  D2  the CpuRowWorkers & overload hands its set to the row loop, and the int
      overload hands none.
  D3  ggml_sycl_cpu_row_loop_threads() is ggml_sycl_cpu_threads_hint().
  R1  the budget's default and maximum are compile-time constants with
      0 < default <= max <= 10 ms; start() takes the default when given no
      budget, clamps its budget to [0, spin_us_max] and gives that to every
      worker; the polling loop compares the clock with it, pauses and then
      yields the core on every poll, and the worker then blocks on the
      condition variable.
  R2  cpu-row-workers.hpp/.cpp read no environment variable, hold no static or
      thread_local variable, catch nothing, and allocate with no new or malloc.
  R3  they include only standard headers (so the host test builds them without
      SYCL) and each other.
  R4  run()'s wait for the chunks other workers claimed also pauses and yields,
      so a worker preempted mid-chunk gets a core back.
  C1  docs/backend/sycl-env-vars.md has one GGML_SYCL_CPU_ROW_SPIN_US row, and
      it states the code's default and maximum and the cores the default spends.
  C2  the GGML_SYCL_CPU_THREADS and GGML_SYCL_CPU_EXPERT_THREADS rows no longer
      say that a pool job's row loop enters the TBB arena.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants, one or more per
pin, and requires each to FAIL on the pin its name starts with; requires a
comment that names getenv in the row-worker source to PASS; and requires the
unmodified tree to pass.  --root runs the gate on another tree (RED: a tree at
e19e726f4 fails P1).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "llama.cpp-w4e9"
SYCL = "ggml/src/ggml-sycl/"
FILES = {
    "pool_hpp": SYCL + "cpu-expert-pool.hpp",
    "pool_cpp": SYCL + "cpu-expert-pool.cpp",
    "disp_cpp": SYCL + "cpu-dispatch.cpp",
    "rw_hpp": SYCL + "cpu-row-workers.hpp",
    "rw_cpp": SYCL + "cpu-row-workers.cpp",
    "doc": "docs/backend/sycl-env-vars.md",
}
BUDGET_MAX_US = 10000


class ContractError(AssertionError):
    pass


def fail(pin: str, why: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {why} ({TAG})")


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, keeping string literals."""
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


def squash(text: str) -> str:
    """Collapse whitespace, and drop it after `(` so a wrapped call reads like an unwrapped one."""
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", text)).strip()


def block_from(src: str, start: int) -> str:
    """From src[start] through the brace that closes the first `{` at or after start."""
    i = src.find("{", start)
    if i < 0:
        return ""
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    return ""


def one_def(code: str, sig: str, pin: str) -> str:
    """The squashed text of the one definition whose head squashes to sig."""
    hits = [m.start() for m in re.finditer(re.escape(sig.split("(")[0]) + r"\s*\(", code)]
    found = [h for h in hits if squash(block_from(code, h)).startswith(sig)]
    if len(found) != 1:
        fail(pin, f"`{sig}` is not defined exactly once (found {len(found)})")
    return squash(block_from(code, found[0]))


def check(src: dict[str, str]) -> None:
    code = {k: blank_comments(v) for k, v in src.items() if k != "doc"}

    # P1-P4: the pool owns the set, starts it at the arena's size, stops it after its workers, uses it.
    m = re.search(r"\bclass\s+CpuExpertPool\s*\{", code["pool_hpp"])
    cls = block_from(code["pool_hpp"], m.start()) if m else ""
    if not re.search(r"\bCpuRowWorkers\s+row_workers_\s*;", cls):
        fail("P1", "CpuExpertPool declares no `CpuRowWorkers row_workers_` member; its jobs run the row loop "
             "on the shared TBB arena, whose workers sleep between a token's jobs")
    init = one_def(code["pool_cpp"], "void CpuExpertPool::init(int n_threads) {", "P2")
    want_start = "row_workers_.start(ggml_sycl_cpu_row_loop_threads(), get_cpu_row_spin_us());"
    if init.count("row_workers_.start(") != 1 or want_start not in init:
        fail("P2", "CpuExpertPool::init does not start its row workers once with ggml_sycl_cpu_row_loop_threads() "
             "and get_cpu_row_spin_us(); the row loop must keep the TBB arena's size, GGML_SYCL_CPU_THREADS, and "
             "the spin budget GGML_SYCL_CPU_ROW_SPIN_US")
    stop = one_def(code["pool_cpp"], "void CpuExpertPool::stop_workers() {", "P3")
    joined, stopped = stop.find("threads_.clear();"), stop.find("row_workers_.stop();")
    if stop.count("row_workers_.stop();") != 1 or not 0 <= joined < stopped:
        fail("P3", "CpuExpertPool::stop_workers does not stop its row workers once, after joining its own workers; "
             "a job still running would lose its row workers mid-loop")
    calls = re.findall(r"ggml_sycl_cpu_expert_mul_mat_batched\s*\([^;]*;", code["pool_cpp"])
    want = "ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size()), row_workers_);"
    if len(calls) != 1 or squash(calls[0]) != want:
        fail("P4", f"the pool's kernel call is not exactly one `{want}`; a pool job would run its row loop "
             "on the TBB arena")
    spin = one_def(code["pool_cpp"], "static int get_cpu_row_spin_us() {", "P5")
    env = 'std::getenv("GGML_SYCL_CPU_ROW_SPIN_US")'
    rets = re.findall(r"\breturn ([^;]*);", spin)
    if (code["pool_cpp"].count(env) != 1 or spin.count(env) != 1 or "us < 0" not in spin or len(rets) < 2
            or any(r != "CpuRowWorkers::spin_us_default" for r in rets[:-1])
            or rets[-1] != "static_cast<int>(std::min<long>(us, CpuRowWorkers::spin_us_max))"):
        fail("P5", "get_cpu_row_spin_us() is not the one reader of GGML_SYCL_CPU_ROW_SPIN_US that rejects a "
             "negative value, falls back to CpuRowWorkers::spin_us_default and caps the value at spin_us_max")

    # D1-D3: one row body, two executors; the overloads pick one; the thread count is the arena's.
    run = one_def(code["disp_cpp"], "static void cpu_expert_mul_mat_batched_run(const cpu_expert_task * tasks, "
                  "int n_tasks, ggml_sycl::CpuRowWorkers * row_workers) {", "D1")
    head = "const auto rows_body = [tasks_ptr, q8_ptrs, meta_ptr, n_tasks, trace, trace_first_p, trace_slots_p]" \
           "(int range_begin, int range_end, int slot) {"
    if run.count("const auto rows_body = [") != 1 or head not in run:
        fail("D1", "the row loop is not one lambda `rows_body(range_begin, range_end, slot)`")
    body = block_from(run, run.find(head))
    if "range.begin()" in body or "range.end()" in body or "current_thread_index" in body:
        fail("D1", "rows_body reads a TBB range or arena slot itself; on a row worker neither exists")
    executors = ("if (row_workers) { row_workers->run(total_rows, std::cref(rows_body)); } else { "
                 "ggml_sycl_cpu_arena().execute([&] { ggml_sycl_tbb::parallel_for(ggml_sycl_tbb::blocked_range<int>"
                 "(0, total_rows, grain), [&rows_body](const ggml_sycl_tbb::blocked_range<int> & range) { "
                 "rows_body(range.begin(), range.end(), ggml_sycl_tbb::this_task_arena::current_thread_index()); "
                 "}); }); }")
    if run.count(executors) != 1 or run.count("rows_body(") != 1 or run.count("std::cref(rows_body)") != 1:
        fail("D1", "rows_body is not run by exactly the two executors, the row workers when given (by "
             "std::cref, so a job allocates no copy) and the TBB arena otherwise (with its slot index)")
    ovl = one_def(code["disp_cpp"], "void ggml_sycl_cpu_expert_mul_mat_batched(const cpu_expert_task * tasks, "
                  "int n_tasks, ggml_sycl::CpuRowWorkers & row_workers) {", "D2")
    if not ovl.endswith("{ cpu_expert_mul_mat_batched_run(tasks, n_tasks, &row_workers); }"):
        fail("D2", "the CpuRowWorkers & overload does not hand its row workers to the row loop")
    tbb = one_def(code["disp_cpp"], "void ggml_sycl_cpu_expert_mul_mat_batched(const cpu_expert_task * tasks, "
                  "int n_tasks, int n_threads) {", "D2")
    if not tbb.endswith("cpu_expert_mul_mat_batched_run(tasks, n_tasks, nullptr); }"):
        fail("D2", "the int overload does not run the row loop on the TBB arena (no row workers)")
    if one_def(code["disp_cpp"], "int ggml_sycl_cpu_row_loop_threads() {", "D3") != \
            "int ggml_sycl_cpu_row_loop_threads() { return ggml_sycl_cpu_threads_hint(); }":
        fail("D3", "ggml_sycl_cpu_row_loop_threads() is not ggml_sycl_cpu_threads_hint(), the TBB arena's size")

    # R1: the polling is bounded and yields the core, then the worker blocks.
    hpp = squash(code["rw_hpp"])
    dflt = re.findall(r"\bstatic constexpr int spin_us_default = (\d+);", hpp)
    smax = re.findall(r"\bstatic constexpr int spin_us_max = (\d+);", hpp)
    if len(dflt) != 1 or len(smax) != 1 or not (0 < int(dflt[0]) <= int(smax[0]) <= BUDGET_MAX_US):
        fail("R1", "CpuRowWorkers::spin_us_default and spin_us_max are not compile-time constants with "
             f"0 < default <= max <= {BUDGET_MAX_US} us; an idle process must not keep the row workers polling")
    if hpp.count("void start(int n_threads, int spin_us = spin_us_default);") != 1:
        fail("R1", "CpuRowWorkers::start does not take spin_us_default when it is given no budget")
    start = one_def(code["rw_cpp"], "void CpuRowWorkers::start(int n_threads, int spin_us) {", "R1")
    clamp = "const int budget_us = std::min(std::max(spin_us, 0), spin_us_max);"
    rest = start.replace("int spin_us) {", "", 1).replace(clamp, "", 1)
    if (start.count(clamp) != 1
            or start.count("threads_.emplace_back(&CpuRowWorkers::worker, this, slot, gen, budget_us);") != 1
            or "spin_us_.store(budget_us, std::memory_order_release);" not in start
            or re.search(r"\bspin_us\b", rest)):
        fail("R1", "CpuRowWorkers::start does not clamp its budget to [0, spin_us_max] and give only the clamped "
             "budget to its workers")
    wait = one_def(code["rw_cpp"], "bool CpuRowWorkers::wait_for_run(uint32_t seen_gen, uint32_t & gen, int spin_us) {",
                   "R1")
    poll = ("if (clock::now() - idle_since < budget) { GGML_SYCL_ROW_WORKERS_PAUSE(); std::this_thread::yield(); "
            "continue; }")
    if ("const auto budget = std::chrono::microseconds(spin_us);" not in wait
            or wait.count("if (clock::now() - idle_since < budget) {") != 1
            or wait.count(poll) != 1
            or wait.count("sleep_cv_.wait(lock, new_run_or_stop);") != 1
            or wait.find(poll) > wait.find("sleep_cv_.wait(")):
        fail("R1", "CpuRowWorkers::wait_for_run does not poll only while the spin budget lasts, pausing and then "
             "yielding the core on every poll, and then block on its condition variable")
    runf = one_def(code["rw_cpp"], "void CpuRowWorkers::run(int n_rows, const row_body & body) {", "R4")
    if runf.count("while (chunks_done_.load(std::memory_order_acquire) < n_chunks) { "
                  "GGML_SYCL_ROW_WORKERS_PAUSE(); std::this_thread::yield(); }") != 1:
        fail("R4", "CpuRowWorkers::run's wait for the chunks other workers claimed does not pause and then yield; "
             "a worker preempted mid-chunk would wait for a core while the caller spins on it")

    # C1-C2: the catalog states the knob as the code has it, and no longer routes pool jobs into the arena.
    rows = [l for l in src["doc"].splitlines() if l.startswith("| `GGML_SYCL_CPU_ROW_SPIN_US")]
    if (len(rows) != 1 or f"Default **{dflt[0]}**" not in rows[0] or f"clamped to [0, {smax[0]}]" not in rows[0]
            or "**The default spends roughly 6-10 cores during decode on this host**" not in rows[0]):
        fail("C1", "docs/backend/sycl-env-vars.md has no one GGML_SYCL_CPU_ROW_SPIN_US row stating the code's "
             "default and maximum and the cores the default spends during decode")
    for l in src["doc"].splitlines():
        if l.startswith(("| `GGML_SYCL_CPU_THREADS", "| `GGML_SYCL_CPU_EXPERT_THREADS")) and (
                "enters the TBB arena above for the row loop" in l
                or "row loop is a `parallel_for` into the ONE arena" in l):
            fail("C2", "a GGML_SYCL_CPU_THREADS / GGML_SYCL_CPU_EXPERT_THREADS catalog row still says a pool job's "
                 "row loop runs in the TBB arena")

    # R2-R3: the row-worker TU's hygiene and host-buildability.
    for key in ("rw_hpp", "rw_cpp"):
        name = Path(FILES[key]).name
        c = code[key]
        for pat, why in ((r"\bgetenv\b", "reads an environment variable"),
                         (r"\bstatic\b(?!\s+constexpr\b)", "declares a static (non-constexpr) entity"),
                         (r"\bthread_local\b", "declares a thread_local"),
                         (r"\bcatch\s*\(", "catches an exception"),
                         (r"\bmalloc\s*\(|\bnew\s+[A-Za-z_]", "allocates with malloc or new")):
            if re.search(pat, c):
                fail("R2", f"{name} {why}")
        for inc in re.findall(r"^\s*#\s*include\s*([<\"][^>\"]+[>\"])", c, re.M):
            if not (inc.startswith("<") or inc == '"cpu-row-workers.hpp"') or "sycl" in inc:
                fail("R3", f"{name} includes {inc}; it must name only standard headers so the host test builds "
                     "it without SYCL")


def read_tree(root: Path) -> dict[str, str]:
    return {k: (root / rel).read_text() if (root / rel).exists() else "" for k, rel in FILES.items()}


def sub(src: dict[str, str], key: str, old: str, new: str, what: str) -> dict[str, str]:
    if src[key].count(old) != 1:
        raise SystemExit(f"SELF-TEST BROKEN: anchor for '{what}' must occur once in {FILES[key]}: {old!r}")
    out = dict(src)
    out[key] = src[key].replace(old, new)
    return out


def mutants(s: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    k = "ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size()), row_workers_);"
    st = "row_workers_.start(ggml_sycl_cpu_row_loop_threads(), get_cpu_row_spin_us());"
    env = 'std::getenv("GGML_SYCL_CPU_ROW_SPIN_US")'
    capped = "return static_cast<int>(std::min<long>(us, CpuRowWorkers::spin_us_max));"
    new_row = ("then runs the row loop on the pool's row workers described under `GGML_SYCL_CPU_ROW_SPIN_US` "
               "below).")
    return [
        ("P1 no member", sub(s, "pool_hpp", "    CpuRowWorkers                     row_workers_;\n", "", "P1")),
        ("P2 never started", sub(s, "pool_cpp", "    " + st + "\n", "", "P2a")),
        ("P2 own count", sub(s, "pool_cpp", st, "row_workers_.start(n_threads, get_cpu_row_spin_us());", "P2b")),
        ("P2 literal budget", sub(s, "pool_cpp", st, "row_workers_.start(ggml_sycl_cpu_row_loop_threads(), 2000);",
                                  "P2c")),
        ("P2 knob ignored", sub(s, "pool_cpp", st, "row_workers_.start(ggml_sycl_cpu_row_loop_threads());", "P2d")),
        ("P3 never stopped", sub(s, "pool_cpp", "    row_workers_.stop();\n", "", "P3")),
        ("P3 stopped before the join", sub(sub(s, "pool_cpp", "    row_workers_.stop();\n", "", "P3"), "pool_cpp",
                                          "    for (auto & t : threads_) {\n",
                                          "    row_workers_.stop();\n    for (auto & t : threads_) {\n", "P3b")),
        ("P4 TBB kernel call", sub(s, "pool_cpp", k,
                                   "ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), "
                                   "static_cast<int>(tasks.size()));", "P4")),
        ("D1 row workers never chosen", sub(s, "disp_cpp", "    if (row_workers) {\n",
                                            "    if (false && row_workers) {\n", "D1a")),
        ("D1 body copied per job", sub(s, "disp_cpp", "row_workers->run(total_rows, std::cref(rows_body));",
                                       "row_workers->run(total_rows, rows_body);", "D1b")),
        ("D1 arena slot dropped", sub(s, "disp_cpp", "ggml_sycl_tbb::this_task_arena::current_thread_index());\n"
                                      "                                        });",
                                      "0);\n                                        });", "D1c")),
        ("D1 body reads the arena slot", sub(s, "disp_cpp", "        if (trace) {\n            const int64_t ns",
                                             "        slot = ggml_sycl_tbb::this_task_arena::current_thread_index();\n"
                                             "        if (trace) {\n            const int64_t ns", "D1d")),
        ("D2 overload drops its set", sub(s, "disp_cpp",
                                          "cpu_expert_mul_mat_batched_run(tasks, n_tasks, &row_workers);",
                                          "cpu_expert_mul_mat_batched_run(tasks, n_tasks, nullptr);", "D2a")),
        ("P5 knob never read", sub(s, "pool_cpp", env, "nullptr", "P5a")),
        ("P5 another variable", sub(s, "pool_cpp", env, 'std::getenv("GGML_SYCL_CPU_SPIN_US")', "P5b")),
        ("P5 negative accepted", sub(s, "pool_cpp", "if (*end != '\\0' || us < 0) {", "if (*end != '\\0') {", "P5c")),
        ("P5 invalid gives the max", sub(s, "pool_cpp", "        return CpuRowWorkers::spin_us_default;\n    }\n"
                                         "    " + capped, "        return CpuRowWorkers::spin_us_max;\n    }\n"
                                         "    " + capped, "P5d")),
        ("P5 uncapped", sub(s, "pool_cpp", capped, "return static_cast<int>(us);", "P5e")),
        ("D3 own thread count", sub(s, "disp_cpp", "int ggml_sycl_cpu_row_loop_threads() {\n"
                                    "    return ggml_sycl_cpu_threads_hint();",
                                    "int ggml_sycl_cpu_row_loop_threads() {\n    return 22;", "D3")),
        ("R1 default of 2 s", sub(s, "rw_hpp", "spin_us_default = 2000;", "spin_us_default = 2000000;", "R1a")),
        ("R1 default of 0", sub(s, "rw_hpp", "spin_us_default = 2000;", "spin_us_default = 0;", "R1b")),
        ("R1 max of 1 s", sub(s, "rw_hpp", "spin_us_max     = 10000;", "spin_us_max     = 1000000;", "R1e")),
        ("R1 no default budget", sub(s, "rw_hpp", "int spin_us = spin_us_default);", "int spin_us);", "R1f")),
        ("R1 no clamp", sub(s, "rw_cpp", "std::min(std::max(spin_us, 0), spin_us_max)", "spin_us", "R1g")),
        ("R1 workers get the raw budget", sub(s, "rw_cpp", "this, slot, gen, budget_us);", "this, slot, gen, spin_us);",
                                              "R1h")),
        ("R1 poll does not yield", sub(s, "rw_cpp", "            std::this_thread::yield();\n            continue;\n",
                                       "            continue;\n", "R1i")),
        ("R1 poll does not pause", sub(s, "rw_cpp", "            GGML_SYCL_ROW_WORKERS_PAUSE();\n"
                                       "            std::this_thread::yield();\n",
                                       "            std::this_thread::yield();\n", "R1j")),
        ("R4 completion wait does not yield", sub(s, "rw_cpp", "        GGML_SYCL_ROW_WORKERS_PAUSE();\n"
                                                  "        std::this_thread::yield();\n    }\n    body_ = nullptr;",
                                                  "        GGML_SYCL_ROW_WORKERS_PAUSE();\n    }\n    body_ = nullptr;",
                                                  "R4")),
        ("C1 doc disagrees with the default", sub(s, "rw_hpp", "spin_us_default = 2000;", "spin_us_default = 1500;",
                                                  "C1a")),
        ("C1 doc disagrees with the max", sub(s, "rw_hpp", "spin_us_max     = 10000;", "spin_us_max     = 5000;",
                                              "C1b")),
        ("C1 cores statement dropped", sub(s, "doc", "**The default spends roughly 6-10 cores during decode on this "
                                           "host**", "The default spins", "C1c")),
        ("C1 row missing", sub(s, "doc", "| `GGML_SYCL_CPU_ROW_SPIN_US=N` |", "| `GGML_SYCL_CPU_ROW_SPINS=N` |",
                               "C1d")),
        ("C2 arena wording restored", sub(s, "doc", new_row, "then enters the TBB arena above for the row loop).",
                                          "C2")),
        ("R1 never blocks", sub(s, "rw_cpp", "        sleep_cv_.wait(lock, new_run_or_stop);\n", "", "R1c")),
        ("R1 polls without the clock", sub(s, "rw_cpp", "if (clock::now() - idle_since < budget) {",
                                           "if (true) {", "R1d")),
        ("R2 getenv", sub(s, "rw_cpp", "namespace ggml_sycl {\n",
                          'namespace ggml_sycl {\nconst char * row_env() { return std::getenv("X"); }\n', "R2a")),
        ("R2 static variable", sub(s, "rw_cpp", "namespace ggml_sycl {\n",
                                   "namespace ggml_sycl {\nstatic int g_runs = 0;\n", "R2b")),
        ("R2 thread_local", sub(s, "rw_cpp", "namespace ggml_sycl {\n",
                                "namespace ggml_sycl {\nthread_local int t_slot;\n", "R2c")),
        ("R2 catch", sub(s, "rw_cpp", "        t.join();\n", "        try { t.join(); } catch (...) {}\n", "R2d")),
        ("R2 new", sub(s, "rw_cpp", "    body_ = nullptr;\n", "    body_ = nullptr;\n    delete new int(1);\n", "R2e")),
        ("R3 SYCL header", sub(s, "rw_hpp", "#include <atomic>\n", '#include "common.hpp"\n#include <atomic>\n', "R3")),
    ]


def harmless(s: dict[str, str]) -> list[tuple[str, dict[str, str]]]:
    return [("a comment naming getenv, static and new in the row-worker source",
             sub(s, "rw_cpp", "namespace ggml_sycl {\n",
                 "namespace ggml_sycl {\n// no getenv here; nothing static, nothing from new\n", "harmless"))]


def self_test(root: Path) -> int:
    src = read_tree(root)
    caught = 0
    for name, m in mutants(src):
        want = f"FAIL [pin {name.split()[0]}]"
        try:
            check(m)
        except ContractError as e:
            if not str(e).startswith(want):
                print(f"SELF-TEST FAIL: mutant '{name}' failed on another pin: {e}")
                return 1
            caught += 1
            continue
        print(f"SELF-TEST FAIL: mutant '{name}' passed the gate")
        return 1
    probes = harmless(src)
    for name, h in probes:
        try:
            check(h)
        except ContractError as e:
            print(f"SELF-TEST FAIL: harmless probe '{name}' failed the gate: {e}")
            return 1
    check(src)
    print(f"SELF-TEST PASS: {caught} mutants caught, {len(probes)} harmless probe(s) passed, unmodified tree passes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    try:
        if args.self_test:
            return self_test(args.root)
        check(read_tree(args.root))
    except ContractError as e:
        print(e)
        return 1
    print(f"PASS: CpuExpertPool jobs run their row loop on the pool's row workers ({TAG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
