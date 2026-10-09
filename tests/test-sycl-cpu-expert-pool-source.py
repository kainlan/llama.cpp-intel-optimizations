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
  * no allocation: no unified_allocate, unified_alloc or alloc_request in
    either file, and no mem_handle, alloc_handle or alloc_owner in the class
    body or anywhere in the .cpp, in any form (a member of any name, a
    container of handles, a file-scope handle), so the pool owns no memory;
  * CpuExpertPool::init() takes no SYCL queue and no buffer geometry, only
    its thread count.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants (each ring name
declared again, the allocation tag, an allocation call, an alloc_request, each
handle type in several forms, a queue parameter on init) and requires each to
FAIL; it
then requires a comment that mentions the ring to PASS, which shows the check
reads code rather than prose, and finally requires the unmodified tree to
pass.  The mutants are inserted at whitespace-tolerant regex anchors that
must match exactly once, so a clang-format re-wrap of the pool does not break
the self-test.  --root runs the gate on another tree, which is how the mutants
were also checked against a git-archive copy.
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
OWNERSHIP_TYPES = re.compile(r"\b(?:mem_handle|alloc_handle|alloc_owner)\b")
CLASS_HEAD = re.compile(r"\bclass\s+CpuExpertPool\s*\{")


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


def class_body(src: str) -> str | None:
    """The text between CpuExpertPool's opening brace and its matching closing brace."""
    m = CLASS_HEAD.search(src)
    if not m:
        return None
    depth = 0
    for i in range(m.end() - 1, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[m.end() : i]
    return None


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

    hpp, cpp = files[0][1], files[1][1]
    body = class_body(hpp)
    if body is None:
        raise ContractError(f"FAIL: cpu-expert-pool.hpp declares no class CpuExpertPool body ({TAG})")
    for name, src in (("class CpuExpertPool", body), ("cpu-expert-pool.cpp", cpp)):
        m = OWNERSHIP_TYPES.search(src)
        if m:
            raise ContractError(
                f"FAIL: {name} names {m.group(0)}; the pool owns threads and a queue, not memory, so it holds no "
                f"allocation handle or owner ({TAG})"
            )

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


CLASS_ANCHOR = r"class\s+CpuExpertPool\s*\{\s*public\s*:[ \t]*\n"
NS_ANCHOR = r"namespace\s+ggml_sycl\s*\{[ \t]*\n"
INIT_DECL = r"\bvoid\s+init\s*\(\s*int\s+n_threads\s*\)\s*;"


def insert_after(src: str, anchor: str, text: str, what: str) -> str:
    """Insert `text` after the one match of the whitespace-tolerant regex `anchor`."""
    hits = list(re.finditer(anchor, src))
    if len(hits) != 1:
        raise SystemExit(f"SELF-TEST BROKEN: anchor for '{what}' must match exactly once, matched {len(hits)}: {anchor}")
    end = hits[0].end()
    return src[:end] + text + src[end:]


def replace_once(src: str, anchor: str, text: str, what: str) -> str:
    hits = list(re.finditer(anchor, src))
    if len(hits) != 1:
        raise SystemExit(f"SELF-TEST BROKEN: anchor for '{what}' must match exactly once, matched {len(hits)}: {anchor}")
    return src[: hits[0].start()] + text + src[hits[0].end() :]


def mutants(hpp: str, cpp: str) -> list[tuple[str, str, str]]:
    out = []
    for decl in (
        "    int acquire_staging();\n",
        "    void release_staging(int slot_id);\n",
        "    struct StagingSlot { float * act; };\n",
        "    struct RingEntry { float * act; };\n",
        "    float * ring_[4];\n",
        "    std::mutex ring_mutex_;\n",
        "    mem_handle ring_handle_;\n",
        "    mem_handle staging;\n",
        "    std::vector<mem_handle> slots_;\n",
        "    alloc_handle ring_alloc_;\n",
        "    alloc_owner owner_;\n",
    ):
        out.append((f"class declares {decl.strip()}", insert_after(hpp, CLASS_ANCHOR, decl, decl), cpp))
    for name, line in (
        ("cpp ring constant", "static constexpr int RING_SLOTS = 4;\n"),
        ("cpp ring allocation tag", 'static const char * k_tag = "cpu_expert_ring";\n'),
        ("cpp allocates through the unified cache", "static void grab() { (void) unified_allocate(alloc_request{}); }\n"),
        ("cpp builds an alloc_request", "static alloc_request g_req;\n"),
        ("cpp file-scope mem_handle", "static mem_handle g_ring_handle;\n"),
        ("cpp file-scope alloc_owner", "static alloc_owner g_ring_owner;\n"),
    ):
        out.append((name, hpp, insert_after(cpp, NS_ANCHOR, line, name)))
    out.append(("init takes a queue again", replace_once(
        hpp, INIT_DECL, "void init(int n_threads, size_t max_experts, sycl::queue & q);", "init"), cpp))
    return out


def harmless(hpp: str, cpp: str) -> list[tuple[str, str, str]]:
    note = "// the old acquire_staging() ring came from unified_allocate() and held a mem_handle\n"
    return [("a comment naming acquire_staging, unified_allocate() and mem_handle", hpp,
             insert_after(cpp, NS_ANCHOR, note, "harmless"))]


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
