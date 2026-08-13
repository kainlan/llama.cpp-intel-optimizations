#!/usr/bin/env python3
"""Structural gate: DIRECT handles reaching mem_copy/fill must carry an extent."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
# Deliberately empty. Any exception must name file + handle and be reviewed here.
UNBOUNDED_DIRECT_MEM_OP_ALLOWLIST = frozenset()


def calls(source: str, name: str):
    out = []
    for match in re.finditer(r"\b" + re.escape(name) + r"\s*\(", source):
        opening = source.find("(", match.start())
        depth = 0
        quote = None
        escaped = False
        for pos in range(opening, len(source)):
            char = source[pos]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    out.append((match.start(), opening, pos))
                    break
    return out


def arguments(source: str, opening: int, closing: int):
    result = []
    start = opening + 1
    depth = 0
    quote = None
    escaped = False
    for pos in range(start, closing):
        char = source[pos]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(source[start:pos].strip())
            start = pos + 1
    result.append(source[start:closing].strip())
    return result


def violations(path: str, source: str):
    mem_calls = []
    for name in ("mem_copy", "mem_copy_async", "mem_fill", "mem_fill_async"):
        for start, opening, closing in calls(source, name):
            mem_calls.append((start, closing, source[start:closing + 1]))

    bad = []
    for start, opening, closing in calls(source, "from_direct"):
        if len(arguments(source, opening, closing)) >= 5:
            continue
        line = source.count("\n", 0, start) + 1
        # Catch an inline from_direct operand first.
        if any(mem_start < start < mem_end for mem_start, mem_end, _ in mem_calls):
            bad.append(f"{path}:{line}:inline")
            continue
        statement_start = max(0, source.rfind(";", 0, start) + 1)
        prefix = source[statement_start:start]
        assignment = re.search(r"([A-Za-z_]\w*)\s*=\s*(?:(?:ggml_sycl::)?mem_handle::\s*)?$", prefix)
        if not assignment:
            continue
        handle = assignment.group(1)
        # The 10 KiB cap is intentionally mutation-sensitive while preventing
        # an unrelated same-named local in a later function from matching.
        pattern = re.compile(r"\b" + re.escape(handle) + r"\b")
        if any(abs(mem_start - start) <= 10000 and pattern.search(text)
               for mem_start, _, text in mem_calls):
            bad.append(f"{path}:{line}:{handle}")
    return bad


def test_production_direct_mem_operands_are_bounded():
    found = []
    for path in SYCL.rglob("*"):
        if path.suffix not in {".cpp", ".hpp", ".h"} or "tests" in path.parts:
            continue
        if any(part.startswith(".build") for part in path.parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        found.extend(violations(rel, path.read_text(errors="ignore")))
    unexpected = sorted(set(found) - UNBOUNDED_DIRECT_MEM_OP_ALLOWLIST)
    assert not unexpected, "unbounded DIRECT mem-op authority:\n" + "\n".join(unexpected)


def test_gate_rejects_mutated_unbounded_direct_operand():
    mutated = """
void f(void * p, sycl::queue & q) {
    auto h = ggml_sycl::mem_handle::from_direct(
        p, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE);
    ggml_sycl::mem_fill(h, 0, 64, q);
}
"""
    assert violations("mutation.cpp", mutated) == ["mutation.cpp:3:h"]
