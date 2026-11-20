"""Source contract for a refused VRAM arena teardown: what it leaves behind, by whether it had freed a chunk.

unified_cache::arena_destroy() closes every allocator group (CLOSING) and retires every zone's authority before it
frees the arena's chunks. A teardown that refuses before it has freed any chunk puts the arena back as it was before
the attempt: every group gets the state and epoch it had, every retired authority is reopened at the generation its
handles carry, so a handle that resolved before the attempt resolves to the same pointer after it, and new allocations
are admitted. A settle the close interrupted is finished by the restore when its drains are empty. A teardown that
cannot free a chunk after it freed one cannot put it back: the arena is partly freed and stays closed, and that cause
outlives any later refusal until a teardown completes. A chunk a lease still holds when the teardown's wait for it
ends is never freed, and nothing in the free loop aborts: the lease refuses the teardown like a refused free.

These checks pin the ordering facts a device test cannot see: a live allocation refuses before anything is closed;
the teardown allocates nothing after it starts closing; every refusal between the close and the first free restores,
under every group lock; no refusal after a chunk was freed restores; the restore reopens what it retired instead of
building a fresh authority, and finishes an interrupted settle with the settle's own reset; reopening keeps every
record and lease; a leased chunk never reaches the free; no assert or abort is left in the teardown; and only the two
causes found after a chunk was freed read as closed.

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


def block_end(text: str, opener_at: int, opener: str) -> int:
    """The index just past the block of the `opener` (which ends in '{') found at `opener_at`."""
    return opener_at + len(opener) - 1 + len(block_at(text, opener_at))


def fn_body(text: str, sig: str) -> str:
    s = text.find(sig)
    return block_at(text, s + len(sig)) if s != -1 else ""


def ordered(text: str, *needles: str) -> bool:
    at = -1
    for n in needles:
        at = text.find(n, at + 1)
        if at == -1:
            return False
    return True


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


DESTROY_SIG = "bool unified_cache::arena_destroy()"
SETTLE_SIG = "void unified_cache::zone_settle(vram_zone_id zone)"
PREDICATE_SIG = "bool zone_arena_teardown_left_closed(zone_arena_teardown_refusal cause)"
REOPEN_SIG = "void arena_authority::reopen(uint64_t restored_generation)"

CLOSING_RE = re.compile(r"\.state = allocator_group_state::CLOSING;")
REFUSAL_RE = re.compile(r"return refuse\(([\w:]+)\);")
FREE_RE = re.compile(r"sycl::free\s*\(")

HOST_STATE = "zone_arena_teardown_refusal::HOST_STATE"
LIVE_ALLOCATIONS = "zone_arena_teardown_refusal::LIVE_ALLOCATIONS"
DRAIN_TIMEOUT = "zone_arena_teardown_refusal::DRAIN_TIMEOUT"
UNFREED = "unfreed"

ALLOC_CHECK = "if (!group.allocations.empty()) {"
RESTORE = "restore();"
RELOCK = ("for (size_t i = 0; i < ARENA_ALLOCATOR_GROUP_COUNT; ++i) { "
          "lifecycle_locks.emplace_back(arena_allocator_groups_[i].mutex);")
UNLOCK = "lifecycle_locks.clear();"
RESTORE_LAMBDA = "const auto restore = [&]() {"
STICKY = "arena_teardown_refusal_.store(zone_arena_teardown_left_closed(earlier) ? earlier : cause,"
STOP_BEFORE_FIRST_FREE = "if (released_chunk_count == 0) { break; } continue;"
NOTHING_FREED = "if (!all_freed && released_chunk_count == 0) {"
PARTLY_FREED = "if (!all_freed) {"
UNFREED_CAUSE = "unfreed = zone_arena_unfreed_chunk_refusal(leases_outstanding, released_chunk_count > 0);"

FREE_LOOP = "for (size_t ci = 0; ci < arena_chunks_.size(); ++ci) {"
LEASE_WAIT = "while ((outstanding = c.lease_count.load()) > 0 && std::chrono::steady_clock::now() <= deadline) {"
LEASED = "const bool leases_outstanding = outstanding > 0;"
LEASE_GUARD = "if (leases_outstanding) {"
TEST_GUARD = "else if (fail_for_test) {"
FREE_BRANCH = "else {"
ABORTS = ("GGML_ASSERT(", "GGML_ABORT(", "abort(", "std::terminate(")

TAKE_OVER = ("if (group.state != allocator_group_state::RESETTING || !group.draining_authorities.empty()) "
             "{ continue; }")
SETTLE_RESET = "arena_group_reset_allocators_locked(arena_allocator_group_index(zone));"
REPLACEMENT = "settle_replacements[i] = arena_make_authority(zone, arena_generation(zone));"


def destroy(uc: str) -> str:
    return fn_body(code(uc), DESTROY_SIG)


def closing_at(d: str) -> int:
    m = CLOSING_RE.search(d)
    return m.start() if m else -1


def refusals(d: str):
    """(argument, position) of every refusal in the teardown, in source order."""
    return [(m.group(1), m.start()) for m in REFUSAL_RE.finditer(d)]


def guard_of(d: str, at: int) -> str:
    """The text from the `if (` that guards the statement at `at` up to it."""
    g = d.rfind("if (", 0, at)
    return d[g:at] if g != -1 else ""


def nothing_freed_block(d: str):
    """(start, end) of the block for a teardown that freed nothing, or (-1, -1)."""
    s = d.find(NOTHING_FREED)
    return (s, block_end(d, s, NOTHING_FREED)) if s != -1 else (-1, -1)


def partly_freed_at(d: str) -> int:
    """The guard of the partly freed refusal: the first PARTLY_FREED after the nothing-freed block."""
    _, end = nothing_freed_block(d)
    return d.find(PARTLY_FREED, end) if end != -1 else -1


def free_loop(d: str) -> str:
    s = d.find(FREE_LOOP)
    return block_at(d, s) if s != -1 else ""


def restore_body(d: str) -> str:
    start = d.find(RESTORE_LAMBDA)
    return block_at(d, start + len(RESTORE_LAMBDA) - 1) if start != -1 else ""


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
        and f"return refuse({LIVE_ALLOCATIONS});" in block_at(d, check)
        and any(c == HOST_STATE for c, _ in causes)
        and all(p < closing for c, p in causes if c in (HOST_STATE, LIVE_ALLOCATIONS))
        and ".reserve(" not in after
        and "arena_make_authority(" not in after
        and "make_shared" not in after
    )


def claim_every_refusal_before_a_free_restores(uc: str) -> bool:
    """Every refusal after the close but the partly freed one restores in its own guard: the timed-out drain and the
    teardown that freed no chunk, whose cause says whether a lease kept the chunk or the device refused it."""
    d = destroy(uc)
    closing = closing_at(d)
    partly = partly_freed_at(d)
    late = [(c, p) for c, p in refusals(d) if p > closing]
    restorable = [(c, p) for c, p in late if p < partly]
    return (
        closing != -1
        and partly != -1
        and sorted(c for c, _ in restorable) == sorted([DRAIN_TIMEOUT, UNFREED])
        and all(RESTORE in guard_of(d, p) for _, p in restorable)
        and UNFREED_CAUSE in d
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
    """The free loop stops at a chunk it could not free only while nothing is freed and goes on past it after, the
    only restore after the first free is the one for a teardown that freed nothing, and the partly freed refusal
    restores nothing."""
    d = destroy(uc)
    free = FREE_RE.search(d)
    nothing_start, nothing_end = nothing_freed_block(d)
    partly = partly_freed_at(d)
    if free is None or nothing_start == -1 or partly == -1:
        return False
    restored = d[nothing_start:nothing_end]
    partly_block = block_at(d, partly)
    after_free = [m.start() for m in re.finditer(re.escape(RESTORE), d) if m.start() > free.start()]
    return (
        d.count("break;") == 1
        and STOP_BEFORE_FIRST_FREE in free_loop(d)
        and len(after_free) == 1
        and nothing_start < after_free[0] < nothing_end
        and f"return refuse({UNFREED});" in restored
        and f"return refuse({UNFREED});" in partly_block
        and RESTORE not in partly_block
    )


def claim_partly_freed_cause_is_sticky(uc: str) -> bool:
    """A refusal keeps an earlier cause that left the arena closed; only a completed teardown clears it."""
    d = destroy(uc)
    return STICKY in d and "arena_teardown_refusal_.store(zone_arena_teardown_refusal::NONE" in d


def claim_restore_reopens_what_it_retired(uc: str) -> bool:
    """The restore puts back each zone's generation, reopens the authority it retired at that generation and
    publishes that same object, gives each group its state and epoch back, and allocates nothing."""
    d = destroy(uc)
    body = restore_body(d)
    return (
        bool(body)
        and d.find(RESTORE_LAMBDA) < closing_at(d)
        and "arena_generations_[i].store(prior_generations[i], std::memory_order_release);" in body
        and "retired[i]->reopen(prior_generations[i]);" in body
        and "arena_publish_prebuilt(zone, std::move(retired[i]));" in body
        and "drains.erase(std::remove(drains.begin(), drains.end(), retired[i]), drains.end());" in body
        and "group.state = prior_states[g];" in body
        and "group.lifecycle_epoch = prior_epochs[g];" in body
        and "arena_make_authority(" not in body
        and ".reserve(" not in body
        and "push_back(" not in body
    )


def claim_restore_finishes_an_interrupted_settle(uc: str) -> bool:
    """A group the restore gives back RESETTING with empty drains is finished as the settle would have: its
    allocators reset by the settle's own helper, the replacements built before the close published, then OPEN."""
    c = code(uc)
    d = destroy(uc)
    body = restore_body(d)
    take_over = body.find(TAKE_OVER)
    closing = closing_at(d)
    return (
        take_over != -1
        and "group.state = prior_states[g];" in body[:take_over]
        and ordered(body[take_over:], "arena_group_reset_allocators_locked(g);",
                    "arena_publish_prebuilt(zone, std::move(settle_replacements[i]));",
                    "group.state = allocator_group_state::OPEN;")
        and -1 < d.find(REPLACEMENT) < closing
        and SETTLE_RESET in fn_body(c, SETTLE_SIG)
    )


def claim_a_leased_chunk_is_never_freed(uc: str) -> bool:
    """The free loop reads the lease count its wait ended on, and the free sits only in the else branch of the
    lease check: a chunk a lease still holds is reported and kept, and its cause is a lease."""
    loop = free_loop(destroy(uc))
    g = loop.find(LEASE_GUARD)
    if not loop or g == -1 or loop.count(LEASE_GUARD) != 1:
        return False
    lease_block = block_at(loop, g)
    after = loop[block_end(loop, g, LEASE_GUARD):].lstrip()
    if not after.startswith(TEST_GUARD):
        return False
    test_block = block_at(after, 0)
    rest = after[block_end(after, 0, TEST_GUARD):].lstrip()
    if not rest.startswith(FREE_BRANCH):
        return False
    free_block = block_at(rest, 0)
    return (
        ordered(loop, LEASE_WAIT, LEASED, LEASE_GUARD)
        and FREE_RE.search(lease_block) is None
        and FREE_RE.search(test_block) is None
        and FREE_RE.search(free_block) is not None
        and len(FREE_RE.findall(loop)) == 1
        and UNFREED_CAUSE in loop
    )


def claim_the_teardown_never_aborts(uc: str) -> bool:
    """No assert or abort is left in arena_destroy, its free loop included: a lease that outlives the wait refuses."""
    d = destroy(uc)
    return bool(free_loop(d)) and not any(a in d for a in ABORTS)


def claim_only_a_partly_freed_arena_stays_closed(zs: str) -> bool:
    return fn_body(code(zs), PREDICATE_SIG).strip() == (
        "{ return cause == zone_arena_teardown_refusal::FREE_FAILED || "
        "cause == zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE; }"
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
        "arena_destroy()'s restore does not reopen and republish the authority it retired at its generation, does "
        "not give a group its state and epoch back, or allocates")


def test_restore_finishes_an_interrupted_settle():
    assert claim_restore_finishes_an_interrupted_settle(UC), (
        "arena_destroy()'s restore leaves a drained settle it interrupted RESETTING, or finishes it other than the "
        "settle would")


def test_a_leased_chunk_is_never_freed():
    assert claim_a_leased_chunk_is_never_freed(UC), (
        "arena_destroy() can pass a chunk a lease still holds to sycl::free")


def test_the_teardown_never_aborts():
    assert claim_the_teardown_never_aborts(UC), (
        "arena_destroy() keeps an assert or abort, so a lease that outlives the wait kills the process")


def test_only_a_partly_freed_arena_stays_closed():
    assert claim_only_a_partly_freed_arena_stays_closed(ZS), (
        "zone_arena_teardown_left_closed() reads a cause other than the two found after a chunk was freed as closed")


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


def test_witness_nothing_freed_without_restore():
    assert not claim_every_refusal_before_a_free_restores(
        mutate_in(UC, DESTROY_SIG, NOTHING_FREED + " restore();", NOTHING_FREED)
    )


def test_witness_new_pre_free_refusal_without_restore():
    assert not claim_every_refusal_before_a_free_restores(
        mutate_in(UC, DESTROY_SIG, "size_t released_chunk_count = 0;",
                  f"if (arena_chunks_.empty()) {{ return refuse({HOST_STATE}); }} size_t released_chunk_count = 0;")
    )


def test_witness_restore_after_locks_dropped():
    assert not claim_restore_holds_every_group_lock(
        mutate_in(UC, DESTROY_SIG, "if (!all_authorities_drained) { restore(); lifecycle_locks.clear();",
                  "if (!all_authorities_drained) { lifecycle_locks.clear(); restore();")
    )


def test_witness_nothing_freed_restore_without_relock():
    assert not claim_restore_holds_every_group_lock(
        mutate_in(UC, DESTROY_SIG, RELOCK + " } " + NOTHING_FREED, NOTHING_FREED)
    )


def test_witness_restore_after_a_chunk_was_freed():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, NOTHING_FREED, PARTLY_FREED)
    )


def test_witness_partly_freed_arena_restored():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, f"lifecycle_locks.clear(); return refuse({UNFREED});",
                  f"restore(); lifecycle_locks.clear(); return refuse({UNFREED});")
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


def test_witness_restore_keeps_the_bumped_epoch():
    assert not claim_restore_reopens_what_it_retired(
        mutate_in(UC, DESTROY_SIG, "group.lifecycle_epoch = prior_epochs[g];", "")
    )


def test_witness_interrupted_settle_left_resetting():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, DESTROY_SIG, TAKE_OVER, "continue;")
    )


def test_witness_interrupted_settle_opened_without_its_reset():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, DESTROY_SIG, "arena_group_reset_allocators_locked(g);", "")
    )


def test_witness_settle_resets_by_its_own_code():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, SETTLE_SIG, SETTLE_RESET, "z.allocator->reset();")
    )


def test_witness_lease_timeout_asserts_again():
    mutant = mutate_in(UC, DESTROY_SIG, LEASE_GUARD,
                       LEASE_GUARD + ' GGML_ASSERT(false && "VRAM arena chunk freed while leases outstanding");')
    assert not claim_the_teardown_never_aborts(mutant)


def test_witness_lease_timeout_aborts():
    mutant = mutate_in(UC, DESTROY_SIG, LEASE_GUARD,
                       LEASE_GUARD + ' GGML_ABORT("VRAM arena chunk freed while leases outstanding");')
    assert not claim_the_teardown_never_aborts(mutant)


def test_witness_leased_chunk_falls_through_to_the_free():
    assert not claim_a_leased_chunk_is_never_freed(
        mutate_in(UC, DESTROY_SIG, "} " + TEST_GUARD, "} if (fail_for_test) {")
    )


def test_witness_lease_count_ignored():
    assert not claim_a_leased_chunk_is_never_freed(
        mutate_in(UC, DESTROY_SIG, LEASED, "const bool leases_outstanding = false;")
    )


def test_witness_predicate_reads_every_late_cause_as_closed():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        mutate_in(ZS, PREDICATE_SIG,
                  "return cause == zone_arena_teardown_refusal::FREE_FAILED || "
                  "cause == zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE;",
                  "return cause != zone_arena_teardown_refusal::NONE && "
                  "cause != zone_arena_teardown_refusal::HOST_STATE;")
    )


def test_witness_predicate_reads_a_late_lease_as_restorable():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        mutate_in(ZS, PREDICATE_SIG, " || cause == zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE", "")
    )


def test_witness_reopen_drops_the_leases():
    assert not claim_reopen_is_the_inverse_of_close(
        mutate_in(MH, REOPEN_SIG, "alive = true;", "alive = true; lease_counts.assign(lease_counts.size(), 0);")
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
