"""Source contract for a refused VRAM arena teardown: what it leaves behind, by whether it had freed a chunk.

unified_cache::arena_destroy() closes every allocator group (CLOSING) and retires every zone's authority before it
frees the arena's chunks. A teardown that refuses before it has freed any chunk puts the arena back as it was before
the attempt: every group gets the state it had, every retired authority is reopened at the generation its handles
carry, so a handle that resolved before the attempt resolves to the same pointer after it, and new allocations are
admitted. A teardown that refuses after it freed a chunk cannot put it back: the arena is partly freed and stays
closed, and that cause outlives any later refusal until a teardown completes.

These checks pin the ordering facts a device test cannot see: a live allocation refuses before anything is closed;
the teardown allocates nothing after it starts closing; every refusal between the close and the first free restores,
under every group lock; no refusal after a chunk was freed restores; the restore reopens what it retired instead of
building a fresh authority; reopening keeps every record and lease; and only FREE_FAILED reads as closed.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UC = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
ZS = (ROOT / "ggml/src/ggml-sycl/zone-sizing.cpp").read_text()
MH = (ROOT / "ggml/src/ggml-sycl/mem-handle.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


def block_at(text: str, start: int) -> str:
    """The brace block that opens at the first '{' at or after `start`, skipping string and char literals."""
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < len(text) and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[o : i + 1]
        i += 1
    return ""


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


DESTROY_SIG = "bool unified_cache::arena_destroy()"
PREDICATE_SIG = "bool zone_arena_teardown_left_closed(zone_arena_teardown_refusal cause)"
REOPEN_SIG = "void arena_authority::reopen(uint64_t restored_generation)"

CLOSING_RE = re.compile(r"\.state = allocator_group_state::CLOSING;")
REFUSAL_RE = re.compile(r"return refuse\(zone_arena_teardown_refusal::(\w+)\);")
FREE_RE = re.compile(r"sycl::free\s*\(")

ALLOC_CHECK = "if (!group.allocations.empty()) {"
RESTORE = "restore();"
RELOCK = ("for (size_t i = 0; i < ARENA_ALLOCATOR_GROUP_COUNT; ++i) { "
          "lifecycle_locks.emplace_back(arena_allocator_groups_[i].mutex);")
UNLOCK = "lifecycle_locks.clear();"
RESTORE_LAMBDA = "const auto restore = [&]() {"
STICKY = "arena_teardown_refusal_.store(zone_arena_teardown_left_closed(earlier) ? earlier : cause,"
STOP_BEFORE_FIRST_FREE = "if (released_chunk_count == 0) break;"
NOTHING_FREED = "if (!all_freed && released_chunk_count == 0) {"


def destroy(uc: str) -> str:
    return fn_body(code(uc), DESTROY_SIG)


def closing_at(d: str) -> int:
    m = CLOSING_RE.search(d)
    return m.start() if m else -1


def refusals(d: str):
    """(cause, position) of every refusal in the teardown, in source order."""
    return [(m.group(1), m.start()) for m in REFUSAL_RE.finditer(d)]


def guard_of(d: str, at: int) -> str:
    """The text from the `if (` that guards the statement at `at` up to it."""
    g = d.rfind("if (", 0, at)
    return d[g:at] if g != -1 else ""


# ---- the claims ----


def claim_nothing_changes_before_a_live_allocation_refuses(uc: str) -> bool:
    """The allocation check and every host-bookkeeping refusal come before the first CLOSING, and after it the
    teardown reserves and builds nothing, so no refusal can be for a host allocation the close made necessary."""
    d = destroy(uc)
    closing = closing_at(d)
    check = d.find(ALLOC_CHECK)
    causes = refusals(d)
    if closing == -1 or check == -1 or d.count(ALLOC_CHECK) != 1:
        return False
    after = d[closing:]
    return (
        check < closing
        and "return refuse(zone_arena_teardown_refusal::LIVE_ALLOCATIONS);" in block_at(d, check)
        and any(c == "HOST_STATE" for c, _ in causes)
        and all(p < closing for c, p in causes if c in ("HOST_STATE", "LIVE_ALLOCATIONS"))
        and ".reserve(" not in after
        and "arena_make_authority(" not in after
        and "make_shared" not in after
    )


def claim_every_refusal_before_a_free_restores(uc: str) -> bool:
    """Every refusal after the close but FREE_FAILED restores in its own guard, and both restorable ones exist."""
    d = destroy(uc)
    closing = closing_at(d)
    late = [(c, p) for c, p in refusals(d) if p > closing]
    restorable = [(c, p) for c, p in late if c != "FREE_FAILED"]
    return (
        closing != -1
        and {c for c, _ in restorable} == {"DRAIN_TIMEOUT", "FIRST_FREE_FAILED"}
        and all(RESTORE in guard_of(d, p) for _, p in restorable)
    )


def claim_restore_holds_every_group_lock(uc: str) -> bool:
    """Each restore runs after a relock of every group with no unlock in between, and the locks drop after it."""
    d = destroy(uc)
    sites = [m.start() for m in re.finditer(re.escape(RESTORE), d)]
    if len(sites) != 2:
        return False
    for r in sites:
        relock = d.rfind(RELOCK, 0, r)
        unlock_before = d.rfind(UNLOCK, 0, r)
        unlock_after = d.find(UNLOCK, r)
        returns = d.find("return refuse(", r)
        if relock == -1 or unlock_before > relock or unlock_after == -1 or returns == -1 or unlock_after > returns:
            return False
    return True


def claim_no_restore_after_a_chunk_is_freed(uc: str) -> bool:
    """The free loop stops at a failed free only while nothing is freed, the only restore after the first free is
    the one for a teardown that freed nothing, and FREE_FAILED, the partly freed arena, restores nothing."""
    d = destroy(uc)
    free = FREE_RE.search(d)
    nothing_freed = d.find(NOTHING_FREED)
    partly = [p for c, p in refusals(d) if c == "FREE_FAILED"]
    if free is None or nothing_freed == -1 or len(partly) != 1:
        return False
    restored = block_at(d, nothing_freed)
    after_free = [m.start() for m in re.finditer(re.escape(RESTORE), d) if m.start() > free.start()]
    return (
        d.count("break;") == 1
        and STOP_BEFORE_FIRST_FREE in d
        and len(after_free) == 1
        and nothing_freed < after_free[0] < nothing_freed + len(restored)
        and "return refuse(zone_arena_teardown_refusal::FIRST_FREE_FAILED);" in restored
        and RESTORE not in guard_of(d, partly[0])
        and partly[0] > nothing_freed + len(restored)
    )


def claim_partly_freed_cause_is_sticky(uc: str) -> bool:
    """A refusal keeps an earlier cause that left the arena closed; only a completed teardown clears it."""
    d = destroy(uc)
    return STICKY in d and "arena_teardown_refusal_.store(zone_arena_teardown_refusal::NONE" in d


def claim_restore_reopens_what_it_retired(uc: str) -> bool:
    """The restore puts back each zone's generation, reopens the authority it retired at that generation and
    publishes that same object, and allocates nothing."""
    d = destroy(uc)
    start = d.find(RESTORE_LAMBDA)
    body = block_at(d, start + len(RESTORE_LAMBDA) - 1) if start != -1 else ""
    closing = closing_at(d)
    return (
        bool(body)
        and start < closing
        and "arena_generations_[i].store(prior_generations[i], std::memory_order_release);" in body
        and "retired[i]->reopen(prior_generations[i]);" in body
        and "arena_publish_prebuilt(zone, std::move(retired[i]));" in body
        and "drains.erase(std::remove(drains.begin(), drains.end(), retired[i]), drains.end());" in body
        and "group.state = prior_states[g];" in body
        and "arena_make_authority(" not in body
        and ".reserve(" not in body
        and "push_back(" not in body
    )


def claim_only_a_partly_freed_arena_stays_closed(zs: str) -> bool:
    return fn_body(code(zs), PREDICATE_SIG).strip() == (
        "{ return cause == zone_arena_teardown_refusal::FREE_FAILED; }"
    )


def claim_reopen_is_the_inverse_of_close(mh: str) -> bool:
    """Reopening sets back what close_and_invalidate() cleared and touches no record, lease or chunk."""
    body = fn_body(code(mh), REOPEN_SIG)
    return (
        "std::lock_guard<std::mutex> guard(mutex);" in body
        and "generation = restored_generation;" in body
        and "alive = true;" in body
        and "admission_open = true;" in body
        and not any(name in body for name in ("allocations", "lease_counts", "chunks", "allowed_ranges"))
    )


# ---- the checks ----


def test_nothing_changes_before_a_live_allocation_refuses():
    assert claim_nothing_changes_before_a_live_allocation_refuses(UC), (
        "arena_destroy() closes a group or retires an authority before it finds a live allocation, or reserves host "
        "state after it starts closing: a refusal for either would leave the arena closed")


def test_every_refusal_before_a_free_restores():
    assert claim_every_refusal_before_a_free_restores(UC), (
        "a refusal of arena_destroy() between the close and the first free returns without restore(): the arena "
        "stays closed although nothing was freed")


def test_restore_holds_every_group_lock():
    assert claim_restore_holds_every_group_lock(UC), (
        "arena_destroy() restores without every allocator group lock held, so a group can be seen between CLOSING and "
        "the state it had")


def test_no_restore_after_a_chunk_is_freed():
    assert claim_no_restore_after_a_chunk_is_freed(UC), (
        "arena_destroy() restores an arena after it freed a chunk, or stops its free loop after one was freed")


def test_partly_freed_cause_is_sticky():
    assert claim_partly_freed_cause_is_sticky(UC), (
        "a later refusal of arena_destroy() overwrites the cause of a partly freed arena")


def test_restore_reopens_what_it_retired():
    assert claim_restore_reopens_what_it_retired(UC), (
        "arena_destroy()'s restore does not reopen and republish the authority it retired at its generation, or "
        "allocates")


def test_only_a_partly_freed_arena_stays_closed():
    assert claim_only_a_partly_freed_arena_stays_closed(ZS), (
        "zone_arena_teardown_left_closed() reads a cause other than FREE_FAILED as closed")


def test_reopen_is_the_inverse_of_close():
    assert claim_reopen_is_the_inverse_of_close(MH), (
        "arena_authority::reopen() does not undo close_and_invalidate(), or changes a record, lease or chunk")


# ---- mutation witnesses ----


def test_witness_allocations_found_after_closing():
    assert not claim_nothing_changes_before_a_live_allocation_refuses(
        mutate_in(UC, DESTROY_SIG, ALLOC_CHECK,
                  "for (auto & g : arena_allocator_groups_) g.state = allocator_group_state::CLOSING; " + ALLOC_CHECK)
    )


def test_witness_teardown_reserves_after_closing():
    assert not claim_nothing_changes_before_a_live_allocation_refuses(
        mutate_in(UC, DESTROY_SIG, "lifecycle_locks.clear(); ",
                  "lifecycle_locks.clear(); drain_snapshot.reserve(drain_snapshot.size() + 1); ")
    )


def test_witness_drain_timeout_without_restore():
    assert not claim_every_refusal_before_a_free_restores(
        mutate_in(UC, DESTROY_SIG, "if (!all_authorities_drained) { restore();", "if (!all_authorities_drained) {")
    )


def test_witness_first_free_failure_without_restore():
    assert not claim_every_refusal_before_a_free_restores(
        mutate_in(UC, DESTROY_SIG, "} restore(); lifecycle_locks.clear();", "} lifecycle_locks.clear();")
    )


def test_witness_new_pre_free_refusal_without_restore():
    assert not claim_every_refusal_before_a_free_restores(
        mutate_in(UC, DESTROY_SIG, "bool all_freed = true;",
                  "if (arena_chunks_.empty()) { return refuse(zone_arena_teardown_refusal::HOST_STATE); } "
                  "bool all_freed = true;")
    )


def test_witness_restore_after_locks_dropped():
    assert not claim_restore_holds_every_group_lock(
        mutate_in(UC, DESTROY_SIG, "if (!all_authorities_drained) { restore(); lifecycle_locks.clear();",
                  "if (!all_authorities_drained) { lifecycle_locks.clear(); restore();")
    )


def test_witness_first_free_restore_without_relock():
    assert not claim_restore_holds_every_group_lock(
        mutate_in(UC, DESTROY_SIG, RELOCK + " } restore();", "restore();")
    )


def test_witness_restore_after_a_chunk_was_freed():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, NOTHING_FREED, "if (!all_freed) {")
    )


def test_witness_partly_freed_arena_restored():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, "return refuse(zone_arena_teardown_refusal::FREE_FAILED);",
                  "restore(); return refuse(zone_arena_teardown_refusal::FREE_FAILED);")
    )


def test_witness_free_loop_stops_after_a_chunk_was_freed():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, STOP_BEFORE_FIRST_FREE, "break;")
    )


def test_witness_cause_not_sticky():
    assert not claim_partly_freed_cause_is_sticky(
        mutate_in(UC, DESTROY_SIG, STICKY, "arena_teardown_refusal_.store(cause,")
    )


def test_witness_restore_builds_a_fresh_authority():
    assert not claim_restore_reopens_what_it_retired(
        mutate_in(UC, DESTROY_SIG,
                  "retired[i]->reopen(prior_generations[i]); arena_publish_prebuilt(zone, std::move(retired[i]));",
                  "arena_publish_prebuilt(zone, arena_make_authority(zone, prior_generations[i]));")
    )


def test_witness_restore_keeps_the_bumped_generation():
    assert not claim_restore_reopens_what_it_retired(
        mutate_in(UC, DESTROY_SIG, "arena_generations_[i].store(prior_generations[i], std::memory_order_release);",
                  "")
    )


def test_witness_predicate_reads_every_late_cause_as_closed():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        mutate_in(ZS, PREDICATE_SIG, "return cause == zone_arena_teardown_refusal::FREE_FAILED;",
                  "return cause != zone_arena_teardown_refusal::NONE && "
                  "cause != zone_arena_teardown_refusal::HOST_STATE;")
    )


def test_witness_reopen_drops_the_leases():
    assert not claim_reopen_is_the_inverse_of_close(
        mutate_in(MH, REOPEN_SIG, "alive = true;", "alive = true; lease_counts.assign(lease_counts.size(), 0);")
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
