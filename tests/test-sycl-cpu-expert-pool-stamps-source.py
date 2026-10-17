#!/usr/bin/env python3
"""Per-job critical-path stamps in the CPU expert pool trace (llama.cpp-j6jf).

Companion of test-sycl-cpu-expert-pool-source.py (the pool owns no memory) and
test-sycl-cpu-expert-pool-stamps.cpp (the per-token sums and their clamp rules).

THE GAP.  Qwen3.8 decode spends ~50 ms per token in neither the CPU expert pool
nor the device.  The pool trace summed each job's phases, and the host timeline
records host spans only, so nothing showed whether that time is the device chain
ahead of a job, the hand-off from a finished job to its join host_task, or the
next job waiting for its inputs.  Each pool job now carries six steady_clock
stamps, and the [CPU-EXPERT-POOL-TRACE] line carries their per-token sums.

THE CONTRACT.  With comments blanked:
  S. (cpu-expert-pool.cpp CpuExpertPool::submit_batch) the job's record is made
     with std::make_shared<cpu_expert_job_stamps> and stamped with t_submit_ns,
     both inside `if (trace)`: no allocation and no clock read while tracing is off.
  R. (ggml-sycl.cpp moe_cpu_job_wait_inputs) the prologue calls
     cpu_expert_pool_trace_stamp_inputs_ready() after cpu_job_input_marker_wait(marker),
     its wait for the job's inputs (llama.cpp-42zj), inside
     `if (moe_hostpath_waits_enabled())`; and (cpu-expert-pool.cpp) that
     function writes t_inputs_ready_ns only through the worker's current-job pointer,
     inside an `if` on that pointer.
  K. (cpu-dispatch.cpp) t_compute_start_ns and t_compute_end_ns are written inside
     `if (trace)`, from the same stamps as compute_us, in the one definition of the
     batched row loop that times Phase 2. Since the row-worker set, the two public
     ggml_sycl_cpu_expert_mul_mat_batched overloads forward to cpu_expert_mul_mat_batched_run,
     so the pin finds the body by what it does, not by the first matching name.
  J. (ggml-sycl.cpp moe_cpu_job_join) the join reads the clock only inside `if (stamps)`,
     once before job.get() and once after, hands both to
     cpu_expert_pool_trace_note_join_stamps, and never reads the shared
     batched-last phase times; moe_cpu_job_join_on_queue's host_task captures the
     job's record.
  L. (ggml-sycl.cpp) the [CPU-EXPERT-POOL-TRACE] token= line carries inputs_wait=,
     chain_gap=, join_lat=, join_waited_us=, join_wake= and pre_compute=, with join_wake=
     right after join_waited_us=, and signalled= and signal_lat=, the jobs whose span
     ended at their native device event's signal (llama.cpp-qwn5).
  W. (cpu-expert-pool-stamps.hpp cpu_expert_job_stamps_sum) join_wake sums t_join_exit -
     t_compute_end over the joins that waited, the else branch of `if (j.join_found_ready)`,
     so the additive terms close on each job's span.
  P. (ggml/src/ggml-sycl) [CPU-EXPERT-POOL-JOB] is printed at exactly one site, inside
     `if (moe_hostpath_trace_level() >= 2)`: GGML_SYCL_MOE_IDS_COPY_TRACE=2 is the only
     per-job printer, and =1 prints no per-job line.
  C. (docs/backend/sycl-env-vars.md) the GGML_SYCL_MOE_IDS_COPY_TRACE row names the =2
     level, the six sums, signalled= and signal_lat=, and says which are additive and
     which are overlapping diagnostics.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and
requires each to FAIL on its own pin, then requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin S]: CpuExpertPool::submit_batch makes no per-job stamp record (llama.cpp-j6jf)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
POOL = SYCL / "cpu-expert-pool.cpp"
DISPATCH = SYCL / "cpu-dispatch.cpp"
STAMPS = SYCL / "cpu-expert-pool-stamps.hpp"
BACKEND = SYCL / "ggml-sycl.cpp"
DOC = ROOT / "docs" / "backend" / "sycl-env-vars.md"
TAG = "(llama.cpp-j6jf)"

TRACE_GUARD = r"if\s*\(\s*trace\s*\)"
WAITS_GUARD = r"if\s*\(\s*moe_hostpath_waits_enabled\s*\(\s*\)\s*\)"
STAMPS_GUARD = r"if\s*\(\s*stamps\s*\)"
LEVEL2_GUARD = r"if\s*\(\s*moe_hostpath_trace_level\s*\(\s*\)\s*>=\s*2\s*\)"
FIELDS = ("inputs_wait=", "chain_gap=", "join_lat=", "join_waited_us=", "join_wake=", "pre_compute=", "signalled=",
          "signal_lat=")


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


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
        elif src[i] == "'":
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
    """Blank the contents of string and char literals so braces in them do not count."""
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group(0)) - 2) + '"', src)


def body_of(code: str, signature: str, pin: str) -> tuple[str, int]:
    """The body of the first definition matching `signature` (which ends at its opening brace), and its offset."""
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    return body_at(code, m, signature, pin)


def bodies_of(code: str, signature: str, pin: str) -> list[tuple[str, int]]:
    """The body and offset of every definition matching `signature`."""
    return [body_at(code, m, signature, pin) for m in re.finditer(signature, code)]


def body_at(code: str, m: re.Match, signature: str, pin: str) -> tuple[str, int]:
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


def enclosing_headers(code: str, pos: int) -> list[str]:
    """The text before each `{` that encloses `pos`, innermost first (an if's condition, a lambda head, ...)."""
    flat = strip_strings(code)
    heads = []
    depth = 0
    for i in range(pos - 1, -1, -1):
        c = flat[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                k = i - 1
                while k >= 0 and flat[k] not in ";{}":
                    k -= 1
                heads.append(code[k + 1 : i].strip())
            else:
                depth -= 1
    return heads


def guarded(code: str, pos: int, guard: str) -> bool:
    return any(re.search(guard + r"\s*$", h) for h in enclosing_headers(code, pos))


SUBMIT_SIG = r"std::future<void>\s+CpuExpertPool::submit_batch\s*\([^{]*\)\s*\{"
STAMP_INPUTS_SIG = r"void\s+cpu_expert_pool_trace_stamp_inputs_ready\s*\(\s*\)\s*\{"
WAIT_INPUTS_SIG = r"static\s+void\s+moe_cpu_job_wait_inputs\s*\([^)]*\)\s*\{"
# Either public overload or the static row loop they forward to.
BATCHED_SIG = r"void\s+(?:ggml_sycl_cpu_expert_mul_mat_batched|cpu_expert_mul_mat_batched_run)\s*\([^)]*\)\s*\{"
JOIN_SIG = r"static\s+void\s+moe_cpu_job_join\s*\([^)]*\)\s*noexcept\s*\{"
JOIN_ON_QUEUE_SIG = r"static\s+sycl::event\s+moe_cpu_job_join_on_queue\s*\([^)]*\)\s*\{"


def check_submit(pool: str) -> None:
    code = blank_comments(pool)
    body, off = body_of(code, SUBMIT_SIG, "S")
    made = list(re.finditer(r"std::make_shared\s*<\s*cpu_expert_job_stamps\s*>", body))
    if not made:
        fail("S", "CpuExpertPool::submit_batch makes no per-job stamp record")
    for m in made:
        if not guarded(code, off + m.start(), TRACE_GUARD):
            fail("S", "submit_batch makes the per-job stamp record outside `if (trace)`")
    stamps = list(re.finditer(r"\bt_submit_ns\s*=", body))
    if not stamps:
        fail("S", "submit_batch does not stamp t_submit_ns")
    for m in stamps:
        if not guarded(code, off + m.start(), TRACE_GUARD):
            fail("S", "submit_batch stamps t_submit_ns outside `if (trace)`")


def check_inputs(pool: str, backend: str) -> None:
    bcode = blank_comments(backend)
    body, off = body_of(bcode, WAIT_INPUTS_SIG, "R")
    w = re.search(r"\bcpu_job_input_marker_wait\s*\(\s*marker\s*\)", body)
    if not w:
        fail("R", "moe_cpu_job_wait_inputs no longer waits for its input marker")
    calls = list(re.finditer(r"\bcpu_expert_pool_trace_stamp_inputs_ready\s*\(\s*\)", body))
    if not calls:
        fail("R", "moe_cpu_job_wait_inputs does not stamp t_inputs_ready after its input wait")
    for m in calls:
        if m.start() < w.end():
            fail("R", "moe_cpu_job_wait_inputs stamps t_inputs_ready before its input wait returns")
        if not guarded(bcode, off + m.start(), WAITS_GUARD):
            fail("R", "moe_cpu_job_wait_inputs stamps t_inputs_ready outside `if (moe_hostpath_waits_enabled())`")
    pcode = blank_comments(pool)
    sbody, soff = body_of(pcode, STAMP_INPUTS_SIG, "R")
    writes = list(re.finditer(r"(\w+)\s*->\s*t_inputs_ready_ns\s*=", sbody))
    if not writes:
        fail("R", "cpu_expert_pool_trace_stamp_inputs_ready does not write t_inputs_ready_ns through the current job")
    for m in writes:
        if not guarded(pcode, soff + m.start(), r"if\s*\(\s*" + re.escape(m.group(1)) + r"\s*\)"):
            fail("R", "cpu_expert_pool_trace_stamp_inputs_ready writes t_inputs_ready_ns without testing the pointer")


def check_compute(dispatch: str) -> None:
    code = blank_comments(dispatch)
    timed = [(b, o) for b, o in bodies_of(code, BATCHED_SIG, "K") if re.search(r"\bcompute_us\s*=", b)]
    if len(timed) != 1:
        fail("K", f"{len(timed)} definitions of the batched row loop time Phase 2 (`compute_us =`); want exactly one")
    body, off = timed[0]
    for name in ("t_compute_start_ns", "t_compute_end_ns"):
        hits = list(re.finditer(r"\b" + name + r"\s*=", body))
        if not hits:
            fail("K", f"the batched row loop does not stamp {name}")
        for m in hits:
            if not guarded(code, off + m.start(), TRACE_GUARD):
                fail("K", f"the batched row loop stamps {name} outside `if (trace)`")


def check_join(backend: str) -> None:
    code = blank_comments(backend)
    body, off = body_of(code, JOIN_SIG, "J")
    get = re.search(r"\bjob\s*\.\s*get\s*\(\s*\)", body)
    if not get:
        fail("J", "moe_cpu_job_join no longer calls job.get()")
    nows = list(re.finditer(r"\bcpu_expert_pool_trace_now_ns\s*\(\s*\)", body))
    if not any(m.start() < get.start() for m in nows):
        fail("J", "moe_cpu_job_join does not stamp t_join_entry before job.get()")
    if not any(m.start() > get.end() for m in nows):
        fail("J", "moe_cpu_job_join does not stamp t_join_exit after job.get()")
    for m in nows:
        if not guarded(code, off + m.start(), STAMPS_GUARD):
            fail("J", "moe_cpu_job_join reads the clock outside `if (stamps)`")
    note = re.search(r"\bcpu_expert_pool_trace_note_join_stamps\s*\(", body)
    if not note or note.start() < get.end():
        fail("J", "moe_cpu_job_join does not record its stamps after job.get()")
    if not guarded(code, off + note.start(), STAMPS_GUARD):
        fail("J", "moe_cpu_job_join records its stamps outside `if (stamps)`")
    if "batched_last" in body:
        fail("J", "moe_cpu_job_join reads the shared batched-last phase times instead of the job's own record")
    qbody, _ = body_of(code, JOIN_ON_QUEUE_SIG, "J")
    lam = re.search(r"host_task\s*\(\s*\[([^\]]*)\]", qbody)
    if not lam or "stamps" not in lam.group(1):
        fail("J", "moe_cpu_job_join_on_queue's host_task does not capture the job's stamp record")


def check_line(backend: str) -> None:
    fmts = re.findall(r'GGML_LOG_WARN\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)', backend)
    line = None
    for f in fmts:
        text = "".join(re.findall(r'"((?:\\.|[^"\\])*)"', f))
        if "[CPU-EXPERT-POOL-TRACE] token=" in text:
            line = text
    if line is None:
        fail("L", "no [CPU-EXPERT-POOL-TRACE] token= line in ggml-sycl.cpp")
    for field in FIELDS:
        if not re.search(r"(?:^|\s)" + re.escape(field), line):
            fail("L", f"the [CPU-EXPERT-POOL-TRACE] line does not carry {field}")
    if line.index("join_wake=") < line.index("join_waited_us="):
        fail("L", "the [CPU-EXPERT-POOL-TRACE] line prints join_wake= before join_waited_us=")


SUM_SIG = r"inline\s+cpu_expert_job_stamp_sums\s+cpu_expert_job_stamps_sum\s*\([^)]*\)\s*\{"


def check_join_wake(stamps: str) -> None:
    code = blank_comments(stamps)
    body, off = body_of(code, SUM_SIG, "W")
    wake = re.search(r"\bjoin_wake_us\s*\+=\s*cpu_expert_job_stamps_span_us\s*\(\s*j\.t_compute_end_ns\s*,"
                     r"\s*j\.t_join_exit_ns\s*,", body)
    if not wake:
        fail("W", "cpu_expert_job_stamps_sum does not sum join_wake as t_join_exit - t_compute_end")
    heads = enclosing_headers(code, off + wake.start())
    if not heads or heads[0] != "else" or not re.search(r"if\s*\(\s*j\.join_found_ready\s*\)", body):
        fail("W", "join_wake is not summed in the else branch of `if (j.join_found_ready)` (the joins that waited)")


def check_per_job(sources: dict[str, str]) -> None:
    sites = []
    for name, src in sources.items():
        code = blank_comments(src)
        for m in re.finditer(r"\[CPU-EXPERT-POOL-JOB\]", code):
            sites.append((name, code, m.start()))
    if len(sites) != 1:
        fail("P", f"expected one [CPU-EXPERT-POOL-JOB] printer, found {len(sites)}")
    name, code, pos = sites[0]
    if name != "ggml-sycl.cpp" or not guarded(code, pos, LEVEL2_GUARD):
        fail("P", "the per-job line is printed outside `if (moe_hostpath_trace_level() >= 2)`")


def check_doc(doc: str) -> None:
    rows = [ln for ln in doc.splitlines() if ln.startswith("| `GGML_SYCL_MOE_IDS_COPY_TRACE")]
    if len(rows) != 1:
        fail("C", f"expected one GGML_SYCL_MOE_IDS_COPY_TRACE catalog row, found {len(rows)}")
    row = rows[0]
    if "`=2`" not in row and "`2`" not in row:
        fail("C", "the GGML_SYCL_MOE_IDS_COPY_TRACE row does not name the =2 per-job level")
    for field in FIELDS:
        if f"`{field}" not in row:
            fail("C", f"the GGML_SYCL_MOE_IDS_COPY_TRACE row does not name `{field}`")
    if "**Additive**" not in row or "**Overlapping diagnostics" not in row:
        fail("C", "the GGML_SYCL_MOE_IDS_COPY_TRACE row does not say which sums are additive and which overlap")


def read_sources(root: Path = ROOT) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, path in (("pool", POOL), ("dispatch", DISPATCH), ("backend", BACKEND), ("doc", DOC), ("stamps", STAMPS)):
        p = root / path.relative_to(ROOT)
        out[key] = p.read_text(encoding="utf-8") if p.exists() else None
    others = {}
    sycl = root / SYCL.relative_to(ROOT)
    for p in sorted(sycl.glob("*.[ch]pp")):
        if p.name not in ("cpu-expert-pool.cpp", "cpu-dispatch.cpp", "ggml-sycl.cpp", "cpu-expert-pool-stamps.hpp"):
            others[p.name] = p.read_text(encoding="utf-8", errors="replace")
    out["others"] = others  # type: ignore[assignment]
    return out


def check(s: dict) -> None:
    for key, path in (("pool", POOL), ("dispatch", DISPATCH), ("backend", BACKEND), ("doc", DOC), ("stamps", STAMPS)):
        if s[key] is None:
            fail("0", f"{path.relative_to(ROOT)} is missing")
    check_submit(s["pool"])
    check_inputs(s["pool"], s["backend"])
    check_compute(s["dispatch"])
    check_join(s["backend"])
    check_line(s["backend"])
    check_join_wake(s["stamps"])
    printers = dict(s["others"])
    printers.update({"cpu-expert-pool.cpp": s["pool"], "cpu-dispatch.cpp": s["dispatch"],
                     "ggml-sycl.cpp": s["backend"], "cpu-expert-pool-stamps.hpp": s["stamps"]})
    check_per_job(printers)
    check_doc(s["doc"])


# --self-test ---------------------------------------------------------------------------------------


def self_test(s: dict) -> int:
    def sub(pattern: str, repl: str, flags: int = 0):
        return lambda text: re.sub(pattern, repl, text, count=1, flags=flags)

    mutants = [
        ("S", "pool", "no stamp record", sub(r"std::make_shared<cpu_expert_job_stamps>\(\)", "nullptr")),
        ("S", "pool", "record made unconditionally",
         sub(r"if \(trace\) \{\n(\s*)stamps(\s*)= std::make_shared",
             r"{\n\1stamps\2= std::make_shared")),
        ("S", "pool", "t_submit not stamped", sub(r"stamps->t_submit_ns(\s*)=", r"stamps->t_queued_ns\1=")),
        ("R", "backend", "inputs-ready stamp dropped",
         sub(r"ggml_sycl::cpu_expert_pool_trace_stamp_inputs_ready\(\);", ";")),
        ("R", "backend", "inputs-ready stamp unguarded",
         sub(r"if \(moe_hostpath_waits_enabled\(\)\) \{((?:\n\s*//[^\n]*)*)\n(\s*)"
             r"ggml_sycl::cpu_expert_pool_trace_stamp_inputs_ready",
             r"{\1\n\2ggml_sycl::cpu_expert_pool_trace_stamp_inputs_ready")),
        ("R", "backend", "inputs-ready stamped before the wait",
         sub(r"(ggml_sycl::cpu_job_input_marker_wait\(marker\);)",
             r"ggml_sycl::cpu_expert_pool_trace_stamp_inputs_ready();\n    \1")),
        ("R", "pool", "current-job pointer not tested",
         sub(r"if \(g_pool_current_job\) \{\n(\s*)g_pool_current_job->t_inputs_ready_ns",
             r"{\n\1g_pool_current_job->t_inputs_ready_ns")),
        ("K", "dispatch", "compute start not stamped",
         sub(r"\.t_compute_start_ns(\s*)=", r".t_compute_first_ns\1=")),
        ("K", "dispatch", "row loop stops timing Phase 2", sub(r"\.compute_us(\s*)=", r".kernel_us\1=")),
        ("K", "dispatch", "compute end stamped untraced",
         sub(r"(\n(\s*)if \(trace\) \{\n)((?:(?!\n\2\}).)*?)"
             r"(\n\s*g_cpu_expert_batched_last\.t_compute_end_ns\s*=[^;]*;)",
             r"\4\1\3", re.S)),
        ("J", "backend", "join entry not stamped",
         sub(r"join_entry_ns(\s*)= ggml_sycl::cpu_expert_pool_trace_now_ns\(\);", r"join_entry_ns\1= 0;")),
        ("J", "backend", "join clock unguarded",
         sub(r"if \(stamps\) \{\n(\s*)join_entry_ns", r"{\n\1join_entry_ns")),
        ("J", "backend", "join reads the shared phase times",
         sub(r"(static void moe_cpu_job_join\([^)]*\) noexcept \{)",
             r"\1\n    (void) ggml_sycl_cpu_expert_batched_last_phase_times();")),
        ("J", "backend", "host_task drops the record",
         sub(r"host_task\(\[job, stamps = std::move\(stamps\)\]\(\) \{ moe_cpu_job_join\(job, stamps\.get\(\)\);",
             "host_task([job]() { moe_cpu_job_join(job, nullptr);")),
        ("L", "backend", "chain_gap missing from the line", sub(r" chain_gap=%", " chain_wait=%")),
        ("L", "backend", "join_waited_us missing from the line", sub(r'"join_waited_us=%', '"join_waits=%')),
        ("L", "backend", "join_wake missing from the line", sub(r" join_wake=%", " join_tail=%")),
        ("L", "backend", "signal_lat missing from the line", sub(r" signal_lat=%", " signal_tail=%")),
        ("L", "backend", "signalled missing from the line", sub(r" signalled=%", " signals=%")),
        ("L", "backend", "join_wake printed before join_waited_us",
         sub(r'"join_waited_us=%\.0fus join_wake=%\.0fus', '"join_wake=%.0fus join_waited_us=%.0fus')),
        ("W", "stamps", "join_wake summed from the join entry",
         sub(r"join_wake_us \+= cpu_expert_job_stamps_span_us\(j\.t_compute_end_ns",
             "join_wake_us += cpu_expert_job_stamps_span_us(j.t_join_entry_ns")),
        ("W", "stamps", "join_wake summed over ready joins",
         sub(r"(s\.join_lat_us \+= [^\n]*\n)((?:.|\n)*?)\n\s*(s\.join_wake_us \+= [^\n]*)",
             r"\1            \3\n\2")),
        ("P", "backend", "per-job line at level 1",
         sub(r"if \(moe_hostpath_trace_level\(\) >= 2\)", "if (moe_hostpath_trace_level() >= 1)")),
        ("P", "backend", "a second per-job printer",
         sub(r"(static void moe_hostpath_waits_readback\()",
             r'static void moe_extra_job_line() { GGML_LOG_WARN("[CPU-EXPERT-POOL-JOB] x\\n"); }\n\1')),
        ("C", "doc", "doc row loses =2",
         sub(r"(\| `GGML_SYCL_MOE_IDS_COPY_TRACE[^\n]*?)`=2`", r"\1`=1`")),
        ("C", "doc", "doc row loses pre_compute",
         sub(r"(\| `GGML_SYCL_MOE_IDS_COPY_TRACE[^\n]*?)`pre_compute=`", r"\1`pre_kernel=`")),
        # The row names signal_lat= twice (the line's fields, then the sum's definition): drop both.
        ("C", "doc", "doc row loses signal_lat", lambda text: text.replace("`signal_lat=`", "`signal_tail=`")),
        ("C", "doc", "doc row stops saying what overlaps",
         sub(r"(\| `GGML_SYCL_MOE_IDS_COPY_TRACE[^\n]*?)\*\*Overlapping diagnostics", r"\1**Also")),
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
    print("PASS: every CPU expert pool job stamp is taken inside a trace guard, the trace line carries the "
          f"per-token sums, and GGML_SYCL_MOE_IDS_COPY_TRACE=2 is the only per-job printer {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
