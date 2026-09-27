#!/usr/bin/env python3
"""pool_leg's arena-off free() must route through pool_legacy_release() with
the live graph-recording state, source-asserted so it is checked without a
SYCL device.

The release rule itself is covered by the host-only test
sycl-pool-legacy-release, which drives pool-legacy-release.hpp. That test
cannot see the call site: pool_leg passing `false` for graph_recording, or
writing its free list directly again, would leave it green while a block freed
during graph recording goes back on the free list and a later alloc() reuses
memory the recorded graph still references. This file pins the call site.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

POOL_LEG = "struct ggml_sycl_pool_leg : public ggml_sycl_pool {"
FREE_FN = "void free(void * ptr, size_t size) override {"
DTOR_FN = "~ggml_sycl_pool_leg() {"


def matching_brace(text: str, open_idx: int) -> int:
    """Comment/string/char-literal-aware brace match. A self-contained copy
    of the walker in test-sycl-graph-weight-layout-fallback-source.py -- this
    fork's source-gate convention is one self-contained file per check, not a
    shared import, so each file carries its own copy."""
    assert text[open_idx] == "{"
    depth = 0
    state = "code"
    i = open_idx
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError("unclosed brace")


def matching_paren(text: str, open_idx: int) -> int:
    """Paren match for an argument list; the arguments checked here hold no
    string or character literals."""
    assert text[open_idx] == "("
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unclosed paren")


def block(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    return text[start:matching_brace(text, brace) + 1]


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def pool_leg_violations(source: str) -> list[str]:
    if POOL_LEG not in source:
        return [f"{POOL_LEG} is missing"]
    pool_leg = block(source, POOL_LEG)
    found: list[str] = []

    if FREE_FN not in pool_leg:
        found.append("pool_leg has no free() override")
    else:
        free_code = strip_comments(block(pool_leg, FREE_FN))
        # Qualified or not, with or without explicit template arguments, there
        # is exactly one call, and it passes the live recording state.
        call_count = len(re.findall(r"\bpool_legacy_release\b", free_code))
        calls = re.findall(r"\bpool_legacy_release\s*\(\s*ptr\s*,\s*([^,]+?)\s*,", free_code)
        if call_count != 1:
            found.append(f"free() calls pool_legacy_release {call_count} times, expected once")
        elif len(calls) != 1:
            found.append("free() does not call pool_legacy_release(ptr, ...)")
        elif calls[0] != "ggml_sycl_graph_recording_active()":
            found.append(
                f"free() passes `{calls[0]}` as graph_recording, not ggml_sycl_graph_recording_active() -- "
                "a block freed during graph recording would go back on the free list"
            )
        # The one buffer_pool token allowed is the helper's slots argument;
        # any other (an index, a range-for) touches the free list directly.
        if len(re.findall(r"\bbuffer_pool\b", free_code)) != 1 or not re.search(
            r"\bpool_legacy_release\s*\(\s*ptr\s*,[^,]+,\s*buffer_pool\s*,", free_code
        ):
            found.append("free() uses buffer_pool other than as pool_legacy_release()'s slots argument")

        # The dropped owner is released (a unified-cache free) after the lock
        # scope around the helper call closes, not inside it.
        call = re.search(r"\bpool_legacy_release\s*(\()", free_code)
        call_at = call.start() if call else -1
        lock_block_start = -1
        lock_block_end = -1
        lock_blocks = []
        for m in re.finditer(r"\{\s*std::lock_guard<std::mutex>\s+lock\(arena_handles_mutex\);", free_code):
            end = matching_brace(free_code, m.start())
            lock_blocks.append((m.start(), end))
            if lock_block_end < 0 and m.start() < call_at < end:
                lock_block_start = m.start()
                lock_block_end = end

        # The declaration, the helper argument and the post-lock release are
        # the only legitimate uses; any other (an alias, a second release) can
        # release the owner somewhere this gate does not look.
        dropped_uses = len(re.findall(r"\bdropped\b", free_code))
        if dropped_uses != 3:
            found.append(
                f"free() names dropped {dropped_uses} times, expected 3 (declaration, helper argument, "
                "post-lock release)"
            )
        if lock_block_end < 0:
            found.append("free() does not call pool_legacy_release() inside a lock_guard block")
        else:
            # Inside the lock, dropped may appear only as the helper's
            # argument: any other use (an assignment, a move into a local that
            # dies in scope) can release the owner under the lock.
            args_open = call.start(1)
            args_close = matching_paren(free_code, args_open)
            uses = [
                m.start() for m in re.finditer(r"\bdropped\b", free_code[lock_block_start:lock_block_end + 1])
            ]
            if len(uses) != 1 or not args_open < lock_block_start + uses[0] < args_close:
                found.append(
                    "free() uses dropped inside the lock block other than as pool_legacy_release()'s argument -- "
                    "the unified-cache free could run under arena_handles_mutex"
                )

            releases = [m.start() for m in re.finditer(r"\bdropped\s*=\s*\{\s*\}\s*;", free_code)]
            if len(releases) != 1 or releases[0] < lock_block_end:
                found.append(
                    "free() does not release dropped (`dropped = {};`) exactly once after the lock block closes -- "
                    "the unified-cache free would run under arena_handles_mutex"
                )
            elif any(start < releases[0] < end for start, end in lock_blocks):
                found.append(
                    "free() releases dropped inside an arena_handles_mutex lock_guard block -- "
                    "the unified-cache free would run under the lock"
                )

    if DTOR_FN not in pool_leg:
        found.append("pool_leg has no destructor")
    else:
        dtor_code = strip_comments(block(pool_leg, DTOR_FN))
        release_at = dtor_code.find("release_graph_retained();")
        arena_at = dtor_code.find("if (arena_mode_)")
        if release_at < 0 or arena_at < 0 or release_at > arena_at:
            found.append(
                "the destructor does not call release_graph_retained() before its arena_mode_ branch -- "
                "arena-off graph-retained handles would be released only by member destruction"
            )

    return found


def test_pool_leg_free_routes_through_the_release_rule() -> None:
    assert pool_leg_violations(SOURCE.read_text()) == []


def test_recording_argument_mutation_is_witnessed() -> None:
    """Pass `false` for graph_recording, the edit that would leave the host
    test green, and confirm this checker catches it."""
    source = SOURCE.read_text()
    mutated = re.sub(
        r"(ggml_sycl::pool_legacy_release\(\s*ptr\s*,\s*)ggml_sycl_graph_recording_active\(\)",
        r"\1false",
        source,
        count=1,
    )
    assert mutated != source, "recording-argument mutation did not change the source"
    violations = pool_leg_violations(mutated)
    assert any("as graph_recording" in v for v in violations), f"mutation was not witnessed: {violations}"


def test_destructor_mutation_is_witnessed() -> None:
    """Move release_graph_retained() back inside the arena-only branch."""
    source = SOURCE.read_text()
    dtor = block(block(source, POOL_LEG), DTOR_FN)
    mutated_dtor = dtor.replace("        release_graph_retained();\n", "", 1).replace(
        "if (arena_mode_) {\n", "if (arena_mode_) {\n            release_graph_retained();\n", 1
    )
    assert mutated_dtor != dtor, "destructor mutation did not change the source"
    violations = pool_leg_violations(source.replace(dtor, mutated_dtor, 1))
    assert any("release_graph_retained() before" in v for v in violations), (
        f"mutation was not witnessed: {violations}"
    )


def test_free_list_bypass_mutation_is_witnessed() -> None:
    """Touch the free list with a range-for, which an index-only check missed."""
    source = SOURCE.read_text()
    anchor = "        ggml_sycl::mem_handle                 dropped;\n"
    assert source.count(anchor) == 1, "bypass mutation anchor not found"
    mutated = source.replace(anchor, "        for (auto & b : buffer_pool) {\n            (void) b;\n        }\n" + anchor, 1)
    violations = pool_leg_violations(mutated)
    assert any("slots argument" in v for v in violations), f"mutation was not witnessed: {violations}"


def test_release_under_lock_mutation_is_witnessed() -> None:
    """Move `dropped = {};` inside the lock block."""
    source = SOURCE.read_text()
    call_end = "active_handles, graph_retained_handles, dropped, pool_size);\n"
    assert source.count(call_end) == 1, "lock mutation anchor not found"
    mutated = source.replace(call_end, call_end + "            dropped = {};\n", 1)
    mutated = re.sub(r"(\n        \}\n(?:        //[^\n]*\n)*)        dropped = \{\};\n", r"\1", mutated, count=1)
    assert mutated != source and mutated.count("dropped = {};") == source.count("dropped = {};"), (
        "lock mutation did not move the release"
    )
    violations = pool_leg_violations(mutated)
    assert any("after the lock block closes" in v for v in violations), f"mutation was not witnessed: {violations}"


LOCKED_CALL_END = "active_handles, graph_retained_handles, dropped, pool_size);\n"


def insert_after_locked_call(source: str, statement: str) -> str:
    assert source.count(LOCKED_CALL_END) == 1, "locked-call anchor not found"
    return source.replace(LOCKED_CALL_END, LOCKED_CALL_END + statement, 1)


def test_release_by_constructor_assignment_under_lock_is_witnessed() -> None:
    """A release spelled other than `dropped = {};`, inside the lock."""
    mutated = insert_after_locked_call(SOURCE.read_text(), "            dropped = ggml_sycl::mem_handle();\n")
    violations = pool_leg_violations(mutated)
    assert any("inside the lock block" in v for v in violations), f"mutation was not witnessed: {violations}"


def test_release_by_scoped_move_under_lock_is_witnessed() -> None:
    """Move the owner into a local that dies inside the lock."""
    mutated = insert_after_locked_call(SOURCE.read_text(), "            { auto tmp = std::move(dropped); }\n")
    violations = pool_leg_violations(mutated)
    assert any("inside the lock block" in v for v in violations), f"mutation was not witnessed: {violations}"


def test_second_unqualified_call_is_witnessed() -> None:
    """A second call without the ggml_sycl:: qualifier, passing false. It
    names neither buffer_pool nor dropped, so only the call count can catch
    it."""
    source = SOURCE.read_text()
    anchor = "        dropped = {};\n"
    assert source.count(anchor) == 1, "second-call anchor not found"
    mutated = source.replace(
        anchor,
        anchor + "        pool_legacy_release(ptr, false, (ggml_sycl_buffer *) nullptr, 0, active_handles,\n"
        "                            graph_retained_handles, pool_size);\n",
        1,
    )
    violations = pool_leg_violations(mutated)
    assert violations == ["free() calls pool_legacy_release 2 times, expected once"], (
        f"mutation was not witnessed: {violations}"
    )


def test_reference_alias_release_under_lock_is_witnessed() -> None:
    """Alias dropped by reference, then release through the alias inside the
    lock: no use of the name dropped appears in the lock block."""
    source = SOURCE.read_text()
    decl = "        ggml_sycl::mem_handle                 dropped;\n"
    assert source.count(decl) == 1, "alias mutation anchor not found"
    mutated = source.replace(decl, decl + "        auto & d = dropped;\n", 1)
    mutated = insert_after_locked_call(mutated, "            d = {};\n")
    violations = pool_leg_violations(mutated)
    assert any("names dropped 4 times, expected 3" in v for v in violations), (
        f"mutation was not witnessed: {violations}"
    )


def test_release_in_a_second_lock_block_is_witnessed() -> None:
    """Keep the release after the helper's lock block, but inside another."""
    source = SOURCE.read_text()
    release = "        dropped = {};\n"
    assert source.count(release) == 1, "second-lock mutation anchor not found"
    mutated = source.replace(
        release, "        { std::lock_guard<std::mutex> lock(arena_handles_mutex); dropped = {}; }\n", 1
    )
    violations = pool_leg_violations(mutated)
    assert violations == [
        "free() releases dropped inside an arena_handles_mutex lock_guard block -- "
        "the unified-cache free would run under the lock"
    ], f"mutation was not witnessed: {violations}"


def test_second_call_with_template_arguments_is_witnessed() -> None:
    """A second call spelled with explicit template arguments, which a
    name-then-paren count does not see."""
    source = SOURCE.read_text()
    anchor = "        dropped = {};\n"
    assert source.count(anchor) == 1, "templated-call anchor not found"
    mutated = source.replace(
        anchor,
        anchor + "        ggml_sycl::pool_legacy_release<ggml_sycl_buffer>(ptr, false, nullptr, 0, active_handles,\n"
        "                                                          graph_retained_handles, pool_size);\n",
        1,
    )
    violations = pool_leg_violations(mutated)
    assert violations == ["free() calls pool_legacy_release 2 times, expected once"], (
        f"mutation was not witnessed: {violations}"
    )
