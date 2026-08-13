#!/usr/bin/env python3
"""Deterministic structural gate for DIRECT mem-op byte authority.

This is a balanced-token/data-flow check rather than a proximity grep: it follows
local aliases and ternaries, recognizes helper-returned handles, and checks
constant offset + size against constant extents.
"""
from pathlib import Path
import bisect
import re

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
UNBOUNDED_DIRECT_MEM_OP_ALLOWLIST = frozenset()


def calls(source: str, name: str):
    out = []
    for match in re.finditer(r"\b" + re.escape(name) + r"\s*\(", source):
        opening = source.find("(", match.start())
        depth = 0
        quote = None
        escaped = False
        # A call spanning more than 64 KiB is not reviewable and is not a
        # plausible mem-handle expression. The cap also prevents malformed
        # comments/macros from turning this deterministic CI gate quadratic.
        for pos in range(opening, min(len(source), opening + 65536)):
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


def const_size(expr: str):
    expr = re.sub(r"/\*.*?\*/", "", expr, flags=re.S).strip()
    expr = re.sub(r"\b(?:size_t|uint64_t)\s*\(", "(", expr)
    expr = expr.replace("SIZE_MAX", str((1 << 64) - 1))
    if not re.fullmatch(r"[0-9xXa-fA-FuUlL+*() \t-]+", expr):
        return None
    expr = re.sub(r"(?<=\d)[uUlL]+\b", "", expr)
    try:
        value = eval(expr, {"__builtins__": {}}, {})
        return value if isinstance(value, int) and value >= 0 else None
    except Exception:
        return None


def _assignment_index(source: str):
    result = {}
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=(?!=)", source):
        if match.start() > 0 and source[match.start() - 1] in "=!<>":
            continue
        start = match.end()
        end = source.find(";", start)
        if end >= 0:
            result.setdefault(match.group(1), []).append((match.start(), source[start:end].strip()))
    return result


def _latest_assignment(assignments, ident: str, before: int):
    entries = assignments.get(ident, ())
    index = bisect.bisect_left(entries, (before, "")) - 1
    return entries[index] if index >= 0 else None


def _helper_unbounded(source: str):
    bad = set()
    # Locate unbounded DIRECT returns without a whole-file nested-body regex
    # (the production translation unit is intentionally very large).
    for start, opening, closing in calls(source, "from_direct"):
        if len(arguments(source, opening, closing)) >= 5:
            continue
        brace = source.rfind("{", 0, start)
        if brace < 0 or "return" not in source[max(brace, start - 256):start]:
            continue
        header = source[max(0, brace - 512):brace]
        names = re.findall(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$", header)
        if names:
            bad.add(names[-1])
    return bad


def _authority(source: str, assignments, expr: str, before: int, helper_bad, seen=frozenset()):
    direct = calls(expr, "from_direct")
    if direct:
        _, opening, closing = direct[0]
        args = arguments(expr, opening, closing)
        return (len(args) < 5, const_size(args[4]) if len(args) >= 5 else None)

    for helper in helper_bad:
        if re.search(r"\b" + re.escape(helper) + r"\s*\(", expr):
            return (True, None)

    # Follow every identifier in plain aliases and ternaries. Any unbounded
    # branch taints the resulting handle. Opaque function-call arguments are
    # values, not handle aliases; reviewed bad helpers were handled above.
    if re.search(r"\b[A-Za-z_]\w*\s*\(", expr):
        return (False, None)
    best_extent = None
    for ident in re.findall(r"\b[A-Za-z_]\w*\b", expr):
        if ident in seen:
            continue
        assigned = _latest_assignment(assignments, ident, before)
        if assigned is None:
            continue
        assigned_pos, assigned_expr = assigned
        if before - assigned_pos > 10000:
            continue
        unbounded, extent = _authority(source, assignments, assigned_expr, assigned_pos, helper_bad,
                                       seen | {ident})
        if unbounded:
            return (True, None)
        if extent is not None:
            best_extent = extent if best_extent is None else min(best_extent, extent)
    return (False, best_extent)


def violations(path: str, source: str):
    bad = []
    helper_bad = _helper_unbounded(source)
    assignments = _assignment_index(source)
    for name in ("mem_copy", "mem_copy_async", "mem_fill", "mem_fill_async"):
        for start, opening, closing in calls(source, name):
            args = arguments(source, opening, closing)
            endpoints = []
            if name.startswith("mem_copy") and len(args) in (4, 5):
                endpoints = [(args[0], "0", args[2]), (args[1], "0", args[2])]
            elif name.startswith("mem_copy") and len(args) >= 6:
                endpoints = [(args[0], args[1], args[4]), (args[2], args[3], args[4])]
            elif name.startswith("mem_fill") and len(args) == 4:
                endpoints = [(args[0], "0", args[2])]
            elif name.startswith("mem_fill") and len(args) >= 5:
                endpoints = [(args[0], args[1], args[3])]

            for endpoint, offset_expr, size_expr in endpoints:
                unbounded, extent = _authority(source, assignments, endpoint, start, helper_bad)
                line = source.count("\n", 0, start) + 1
                if unbounded:
                    bad.append(f"{path}:{line}:unbounded")
                    continue
                offset, size = const_size(offset_expr), const_size(size_expr)
                if extent is not None and offset is not None and size is not None and (
                        offset > extent or size > extent - offset):
                    bad.append(f"{path}:{line}:extent<{offset}+{size}")
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
    assert not unexpected, "unbounded/undersized DIRECT mem-op authority:\n" + "\n".join(unexpected)


def test_gate_mutations_cover_alias_ternary_helper_and_offset_extent():
    mutations = {
        "alias": """void f(void*p, queue&q){ auto h=mem_handle::from_direct(p,L,false,-1); auto alias=h; mem_fill(alias,0,64,q); }""",
        "ternary": """void f(void*p, mem_handle ok, bool c, queue&q){ auto h=mem_handle::from_direct(p,L,false,-1); auto selected=c?ok:h; mem_fill(selected,0,64,q); }""",
        "helper": """mem_handle raw(void*p){ return mem_handle::from_direct(p,L,false,-1); } void f(void*p,queue&q){auto h=raw(p);mem_fill(h,0,64,q);}""",
        "offset": """void f(void*p,queue&q){auto h=mem_handle::from_direct(p,L,false,-1,8);mem_fill(h,8,0,1,q);}""",
    }
    for name, source in mutations.items():
        assert violations(f"{name}.cpp", source), f"gate accepted {name} mutation"
