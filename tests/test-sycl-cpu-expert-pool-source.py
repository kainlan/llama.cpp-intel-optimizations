#!/usr/bin/env python3
"""CpuExpertPool keeps no staging ring and allocates no memory (llama.cpp-b2jc).

THE CONTRACT.  CpuExpertPool is a thread pool: it runs host-expert CPU jobs
and owns threads, a queue and a condition variable, nothing else.  Until
llama.cpp-b2jc it also carried four ring staging slots (acquire_staging /
release_staging), allocated through the unified cache in init() and released
in shutdown().  Nothing ever called them: CPU expert staging comes from the
PinnedBufferPool, with the per-dispatch managed fallback (llama.cpp-sfal).
The ring held host-pinned memory for the life of the process, and its
allocation could fail and leave the pool inactive.  It was removed; this
gate keeps it from coming back as dead weight.

WHAT IT CHECKS.  In cpu-expert-pool.hpp and cpu-expert-pool.cpp, with comments
blanked:
  * none of the ring's names: acquire_staging, release_staging, StagingSlot,
    RingEntry, RING_SLOTS, ring_, ring_handle_, ring_mutex_, or the
    "cpu_expert_ring" allocation tag;
  * no allocation: no unified_allocate, unified_alloc, alloc_request or
    mem_handle member, so the pool owns no memory;
  * CpuExpertPool::init() takes no SYCL queue and no buffer geometry, only
    its thread count.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants (each ring name
declared again, the allocation tag, an allocation call, an alloc_request, a
mem_handle member, a queue parameter on init) and requires each to FAIL; it
then requires a comment that mentions the ring to PASS, which shows the check
reads code rather than prose, and finally requires the unmodified tree to
pass.  --root runs the gate on another tree, which is how the mutants were
also checked against a git-archive copy.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HPP_REL = "ggml/src/ggml-sycl/cpu-expert-pool.hpp"
CPP_REL = "ggml/src/ggml-sycl/cpu-expert-pool.cpp"
TAG = "llama.cpp-b2jc"

RING_NAMES = (
    r"\bacquire_staging\b",
    r"\brelease_staging\b",
    r"\bStagingSlot\b",
    r"\bRingEntry\b",
    r"\bRING_SLOTS\b",
    r"\bring_\b",
    r"\bring_handle_\b",
    r"\bring_mutex_\b",
    r'"cpu_expert_ring"',
)
ALLOC_NAMES = (
    r"\bunified_allocate\w*\s*\(",
    r"\bunified_alloc\s*\(",
    r"\balloc_request\b",
)
MEM_HANDLE_MEMBER = re.compile(r"\bmem_handle\s+\w+_\s*[;={]")


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    """Replace // and /* */ comment bodies with spaces, preserving offsets and newlines."""
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


def check(hpp_raw: str | None, cpp_raw: str | None) -> None:
    if hpp_raw is None or cpp_raw is None:
        raise ContractError(f"FAIL: cpu-expert-pool.hpp/.cpp missing ({TAG})")
    files = (("cpu-expert-pool.hpp", blank_comments(hpp_raw)), ("cpu-expert-pool.cpp", blank_comments(cpp_raw)))

    for name, src in files:
        for pat in RING_NAMES:
            m = re.search(pat, src)
            if m:
                raise ContractError(
                    f"FAIL: {name} still carries the staging ring ({m.group(0)}); its slots had no callers and "
                    f"were removed ({TAG})"
                )
        for pat in ALLOC_NAMES:
            m = re.search(pat, src)
            if m:
                raise ContractError(
                    f"FAIL: {name} allocates memory ({m.group(0).strip()}); the pool owns threads and a queue, "
                    f"and CPU expert staging comes from the PinnedBufferPool ({TAG})"
                )
        m = MEM_HANDLE_MEMBER.search(src)
        if m:
            raise ContractError(f"FAIL: {name} gives the pool a mem_handle member ({m.group(0).strip()}) ({TAG})")

    hpp = files[0][1]
    m = re.search(r"\bvoid\s+init\s*\(([^)]*)\)\s*;", hpp)
    if not m:
        raise ContractError(f"FAIL: cpu-expert-pool.hpp declares no CpuExpertPool::init() ({TAG})")
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    if len(params) != 1 or not re.match(r"int\s+n_threads\b", params[0]):
        raise ContractError(
            f"FAIL: CpuExpertPool::init({m.group(1).strip()}) takes more than its thread count; buffer geometry or "
            f"a SYCL queue means the pool is allocating again ({TAG})"
        )


def run(root: Path) -> None:
    hpp = root / HPP_REL
    cpp = root / CPP_REL
    check(hpp.read_text() if hpp.exists() else None, cpp.read_text() if cpp.exists() else None)


def mutants(hpp: str, cpp: str) -> list[tuple[str, str, str]]:
    def sub(src: str, old: str, new: str, what: str) -> str:
        if old not in src:
            raise SystemExit(f"SELF-TEST BROKEN: mutant anchor for '{what}' not found: {old!r}")
        return src.replace(old, new, 1)

    cls = "class CpuExpertPool {\n  public:\n"
    ns = "namespace ggml_sycl {\n"
    out = []
    for decl in (
        "    int acquire_staging();\n",
        "    void release_staging(int slot_id);\n",
        "    struct StagingSlot { float * act; };\n",
        "    struct RingEntry { float * act; };\n",
        "    float * ring_[4];\n",
        "    std::mutex ring_mutex_;\n",
    ):
        out.append((f"hpp declares {decl.strip()}", sub(hpp, cls, cls + decl, decl), cpp))
    out.append(("cpp ring constant", hpp, sub(cpp, ns, ns + "static constexpr int RING_SLOTS = 4;\n", "ring const")))
    out.append(("cpp ring allocation tag", hpp,
                sub(cpp, ns, ns + 'static const char * k_tag = "cpu_expert_ring";\n', "tag")))
    out.append(("cpp allocates through the unified cache", hpp,
                sub(cpp, ns, ns + "static void grab() { (void) unified_allocate(alloc_request{}); }\n", "alloc")))
    out.append(("cpp builds an alloc_request", hpp, sub(cpp, ns, ns + "static alloc_request g_req;\n", "req")))
    out.append(("hpp mem_handle member", sub(hpp, cls, cls + "    mem_handle ring_handle_;\n", "handle"), cpp))
    out.append(("hpp mem_handle member under another name", sub(hpp, cls, cls + "    mem_handle staging_;\n", "handle2"),
                cpp))
    out.append(("init takes a queue again",
                re.sub(r"\bvoid\s+init\s*\(\s*int\s+n_threads[^)]*\)\s*;",
                       "void init(int n_threads, size_t max_experts, sycl::queue & q);", hpp, count=1), cpp))
    return out


def harmless(hpp: str, cpp: str) -> list[tuple[str, str, str]]:
    ns = "namespace ggml_sycl {\n"
    if ns not in cpp:
        raise SystemExit("SELF-TEST BROKEN: harmless-probe anchor not found")
    return [("a comment naming acquire_staging and unified_allocate()", hpp,
             cpp.replace(ns, ns + "// the old acquire_staging() ring came from unified_allocate()\n", 1))]


def self_test(root: Path) -> int:
    hpp = (root / HPP_REL).read_text()
    cpp = (root / CPP_REL).read_text()
    caught = 0
    for name, h, c in mutants(hpp, cpp):
        try:
            check(h, c)
        except ContractError:
            caught += 1
            continue
        print(f"SELF-TEST FAIL: mutant '{name}' passed the gate")
        return 1
    for name, h, c in harmless(hpp, cpp):
        try:
            check(h, c)
        except ContractError as e:
            print(f"SELF-TEST FAIL: harmless probe '{name}' failed the gate: {e}")
            return 1
    run(root)
    print(f"SELF-TEST PASS: {caught} mutants caught, 1 harmless probe passed, unmodified tree passes")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    try:
        if args.self_test:
            return self_test(args.root)
        run(args.root)
    except ContractError as e:
        print(e)
        return 1
    print(f"PASS: CpuExpertPool keeps no staging ring and allocates no memory ({TAG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
