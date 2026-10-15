#!/usr/bin/env python3
"""The CpuExpertPool runs one worker by default (llama.cpp-w4e9).

Companion of test-sycl-cpu-expert-pool-source.py, which pins that the pool owns no memory.

THE DEFECT.  get_cpu_expert_thread_count() (cpu-expert-pool.cpp) defaulted the pool to
hardware_concurrency - 2 workers (22 on this host).  Each worker runs one host-expert job at a
time, and each job's row loop is a parallel_for into the ONE TBB arena the backend owns
(ggml_sycl_cpu_threads_hint(), cpu-dispatch.cpp, also 22 threads).  Concurrent jobs split that
arena between overlapping parallel_fors, and buy nothing, because the layer chain already
serialises the jobs (gate/up, then down, then the next layer).  Measured 2026-10-15 on Qwen3.8
decode: one worker was never slower than 22, and +6% to +29% faster depending on host load.

THE CONTRACT.
  D. (cpu-expert-pool.cpp get_cpu_expert_thread_count) with no override the function returns 1,
     and it no longer derives the count from hardware_concurrency.
  E. (same function) GGML_SYCL_CPU_EXPERT_THREADS=N still overrides the default for N > 0.
  I. (CpuExpertPool::init) a non-positive request still takes get_cpu_expert_thread_count(), so
     a pool initialised with 0 gets the default above.
  P. (ggml-sycl.cpp moe_hybrid_init_once) the production pool is initialised with init(0), so it
     takes that default; no other count is passed there.
  C. (docs/backend/sycl-env-vars.md) the GGML_SYCL_CPU_EXPERT_THREADS row states default 1 and
     no longer states the hardware_concurrency - 2 default.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each
to FAIL on its own pin, then requires the unmodified tree to pass.

RED on the tree before the change:
  FAIL [pin D]: get_cpu_expert_thread_count() still derives the count from hardware_concurrency (llama.cpp-w4e9)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "ggml" / "src" / "ggml-sycl" / "cpu-expert-pool.cpp"
DOC  = ROOT / "docs" / "backend" / "sycl-env-vars.md"
BACKEND = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"
TAG  = "(llama.cpp-w4e9)"


class ContractError(Exception):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


def blank_comments(src: str) -> str:
    """Drop // and /* */ comments so a pin cannot be satisfied by prose."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def body_of(code: str, signature: str, pin: str) -> str:
    m = re.search(signature, code)
    if not m:
        fail(pin, f"no definition matching {signature!r}")
    i = code.index("{", m.end() - 1 if code[m.end() - 1] == "{" else m.end())
    depth = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return code[i + 1 : j]
    fail(pin, f"unbalanced body for {signature!r}")
    return ""


COUNT_SIG = r"static\s+int\s+get_cpu_expert_thread_count\s*\(\s*\)\s*\{"
INIT_SIG  = r"void\s+CpuExpertPool::init\s*\(\s*int\s+n_threads\s*\)\s*\{"


def check_default(pool: str) -> None:
    body = body_of(blank_comments(pool), COUNT_SIG, "D")
    # The env branch is the `if (env) { ... }` block; what follows it is the default.
    m = re.search(r"if\s*\(\s*env\s*\)\s*\{", body)
    tail = body
    if m:
        depth = 0
        for j in range(m.end() - 1, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    tail = body[j + 1 :]
                    break
    if "hardware_concurrency" in body:
        fail("D", "get_cpu_expert_thread_count() still derives the count from hardware_concurrency")
    returns = re.findall(r"\breturn\s+([^;]+);", tail)
    if returns != ["1"]:
        fail("D", "get_cpu_expert_thread_count() does not default to one worker")


def check_env(pool: str) -> None:
    body = body_of(blank_comments(pool), COUNT_SIG, "E")
    if not re.search(r'std::getenv\s*\(\s*"GGML_SYCL_CPU_EXPERT_THREADS"\s*\)', body):
        fail("E", "get_cpu_expert_thread_count() no longer reads GGML_SYCL_CPU_EXPERT_THREADS")
    if not re.search(r"if\s*\(\s*(\w+)\s*>\s*0\s*\)\s*\{?\s*return\s+\1\s*;", body):
        fail("E", "the GGML_SYCL_CPU_EXPERT_THREADS override no longer returns N for N > 0")


def check_init(pool: str) -> None:
    body = body_of(blank_comments(pool), INIT_SIG, "I")
    fallback = (r"if\s*\(\s*n_threads\s*<=\s*0\s*\)\s*\{?\s*"
                r"n_threads\s*=\s*get_cpu_expert_thread_count\s*\(\s*\)\s*;")
    if not re.search(fallback, body):
        fail("I", "CpuExpertPool::init no longer takes get_cpu_expert_thread_count() for n_threads <= 0")


HYBRID_INIT_SIG = r"static\s+void\s+moe_hybrid_init_once\s*\([^)]*\)\s*\{"


def check_production_init(backend: str) -> None:
    code = blank_comments(backend)
    m = re.search(HYBRID_INIT_SIG, code)
    if not m:
        fail("P", "no definition of moe_hybrid_init_once in ggml-sycl.cpp")
    # The function ends at the first closing brace in column 0 (the file's formatting); this
    # avoids brace-matching across string literals in a 100k-line file.
    end = code.find("\n}\n", m.end())
    body = code[m.end() : end if end >= 0 else len(code)]
    inits = re.findall(r"\bcpu_pool\s*\.\s*init\s*\(([^)]*)\)", body)
    if not inits:
        fail("P", "moe_hybrid_init_once no longer initialises the CPU expert pool with cpu_pool.init(...)")
    if any(arg.strip() != "0" for arg in inits):
        fail("P", "moe_hybrid_init_once initialises the CPU expert pool with a count other than 0 "
                  "(it must take get_cpu_expert_thread_count()'s default)")


def check_doc(doc: str) -> None:
    rows = [ln for ln in doc.splitlines() if ln.startswith("| `GGML_SYCL_CPU_EXPERT_THREADS")]
    if len(rows) != 1:
        fail("C", f"expected one GGML_SYCL_CPU_EXPERT_THREADS catalog row, found {len(rows)}")
    row = rows[0]
    if "hardware_concurrency - 2" in row:
        fail("C", "the GGML_SYCL_CPU_EXPERT_THREADS row still states the hardware_concurrency - 2 default")
    if not re.search(r"Default \*\*1\*\*", row):
        fail("C", "the GGML_SYCL_CPU_EXPERT_THREADS row does not state Default **1**")


def read_sources() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, path in (("pool", POOL), ("doc", DOC), ("backend", BACKEND)):
        out[key] = path.read_text(encoding="utf-8") if path.exists() else None
    return out


def check(s: dict[str, str | None]) -> None:
    if s["pool"] is None:
        fail("D0", f"{POOL.relative_to(ROOT)} is missing")
    if s["doc"] is None:
        fail("C0", f"{DOC.relative_to(ROOT)} is missing")
    if s["backend"] is None:
        fail("P0", f"{BACKEND.relative_to(ROOT)} is missing")
    check_default(s["pool"])
    check_env(s["pool"])
    check_init(s["pool"])
    check_production_init(s["backend"])
    check_doc(s["doc"])


# --self-test ---------------------------------------------------------------------------------------


def self_test(s: dict[str, str | None]) -> int:
    hw_default = ("int hw = static_cast<int>(std::thread::hardware_concurrency());\n"
                  "    return std::max(1, hw - 2);")
    mutants = [
        ("D", "pool", "default back to hardware_concurrency - 2",
         lambda p: re.sub(r"(if\s*\(\s*env\s*\)\s*\{.*?\n    \}\n)(.*?)return\s+1\s*;",
                          lambda mo: mo.group(1) + mo.group(2) + hw_default, p, count=1, flags=re.S)),
        ("D", "pool", "default of two workers",
         lambda p: re.sub(r"(if\s*\(\s*env\s*\)\s*\{.*?\n    \}\n)(.*?)return\s+1\s*;", r"\1\2return 2;", p,
                          count=1, flags=re.S)),
        ("E", "pool", "env override dropped",
         lambda p: p.replace('std::getenv("GGML_SYCL_CPU_EXPERT_THREADS")',
                             'std::getenv("GGML_SYCL_CPU_EXPERT_X")', 1)),
        ("E", "pool", "override ignores N",
         lambda p: re.sub(r"if\s*\(\s*n\s*>\s*0\s*\)\s*\{\s*return\s+n\s*;", "if (n > 0) {\n            return 1;", p,
                          count=1)),
        ("I", "pool", "init no longer falls back",
         lambda p: p.replace("n_threads = get_cpu_expert_thread_count();", "n_threads = 4;", 1)),
        ("P", "backend", "production init passes 22",
         lambda b: re.sub(r"cpu_pool\.init\(0\b", "cpu_pool.init(22", b, count=1)),
        ("P", "backend", "production init passes n_threads",
         lambda b: re.sub(r"cpu_pool\.init\(0 /\* auto thread count \*/\)", "cpu_pool.init(n_threads)", b, count=1)),
        ("C", "doc", "doc row back to hardware_concurrency - 2",
         lambda d: re.sub(r"(\| `GGML_SYCL_CPU_EXPERT_THREADS[^\n]*?)Default \*\*1\*\*",
                          r"\1Default `max(1, hardware_concurrency - 2)`", d, count=1)),
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
    s = read_sources()
    if "--self-test" in argv:
        rc = self_test(s)
        if rc != 0:
            return rc
    try:
        check(s)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: the CpuExpertPool defaults to one worker, GGML_SYCL_CPU_EXPERT_THREADS still overrides it, "
          f"and the catalog row says so {TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
