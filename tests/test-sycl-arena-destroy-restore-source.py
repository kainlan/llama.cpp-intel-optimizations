"""Source contract for a refused VRAM arena teardown: what it leaves behind, by whether it had freed a chunk.

unified_cache::arena_destroy() closes every allocator group (CLOSING) and retires every zone's authority before it
frees the arena's chunks. A teardown that refuses before it has freed any chunk puts the arena back as it was before
the attempt: every group gets the state and epoch it had, every retired authority is reopened at the generation its
handles carry, so a handle that resolved before the attempt resolves to the same pointer after it, and new allocations
are admitted. A settle the close interrupted is finished by the restore when its drains are empty and its group holds
no allocation. A teardown that cannot free a chunk after it freed one cannot put it back: the arena is partly freed and
stays closed, and that cause outlives any later refusal until a teardown completes. A chunk a lease still holds when
the teardown's wait for it ends is never freed, and nothing in the free loop aborts: the lease refuses the teardown
like a refused free.

These checks pin the ordering facts a device test cannot see: the live-allocation check runs under every group lock
and refuses before anything is closed; the teardown allocates nothing after it starts closing; every refusal between
the close and the first free restores, under every group lock; every exit after the close drops the group locks
through the one helper that also wakes the threads parked on a group's state; no refusal after a chunk was freed
restores; the restore reopens what it retired instead of building a fresh authority, finishes an interrupted settle
with the settle's own reset and never over a live allocation, and cannot throw; reopening keeps every record and
lease; a leased chunk never reaches the free; no assert or abort is left in the teardown; the test hooks are read
only in private-testing builds; a refusal that frees no chunk says whether an earlier teardown left the arena partly
freed; and only the two causes found after a chunk was freed read as closed.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UC = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
ZS = (ROOT / "ggml/src/ggml-sycl/zone-sizing.cpp").read_text()
ZSH = (ROOT / "ggml/src/ggml-sycl/zone-sizing.hpp").read_text()
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


def literal_text(text: str) -> str:
    """The characters of every string literal in `text`, adjacent literals joined, escapes left as written."""
    return "".join(re.findall(r'"((?:\\.|[^"\\])*)"', text))


DESTROY_SIG = "bool unified_cache::arena_destroy()"
SETTLE_SIG = "void unified_cache::zone_settle(vram_zone_id zone)"
PREDICATE_SIG = "bool zone_arena_teardown_left_closed(zone_arena_teardown_refusal cause)"
REOPEN_SIG = "void arena_authority::reopen(uint64_t restored_generation)"
CLOSE_SIG = "void arena_authority::close_and_invalidate(uint64_t next)"
GROUP_RESET_SIG = "void unified_cache::arena_group_reset_allocators_locked(size_t group)"
CAUSE_ENUM = "enum class zone_arena_teardown_refusal {"

CLOSING_RE = re.compile(r"\.state = allocator_group_state::CLOSING;")
REFUSAL_RE = re.compile(r"return refuse\(([\w:]+)\);")
FREE_RE = re.compile(r"sycl::free\s*\(")
CASE_RE = re.compile(r"case zone_arena_teardown_refusal::(\w+):")
PUBLISH_RE = re.compile(r"void unified_cache::arena_publish_prebuilt\([^)]*\) noexcept \{")

HOST_STATE = "zone_arena_teardown_refusal::HOST_STATE"
LIVE_ALLOCATIONS = "zone_arena_teardown_refusal::LIVE_ALLOCATIONS"
DRAIN_TIMEOUT = "zone_arena_teardown_refusal::DRAIN_TIMEOUT"
UNFREED = "unfreed"
TERMINAL_CAUSES = {"FREE_FAILED", "LEASES_OUTSTANDING_AFTER_FREE"}

ALLOC_CHECK = "if (!group.allocations.empty()) {"
RESTORE = "restore();"
# The group-lock helpers' calls. The leading space keeps " lock_groups();" from matching inside "unlock_groups();".
RELOCK = " lock_groups();"
UNLOCK = " unlock_groups();"
RELEASE = " release_groups();"
RELEASE_LAMBDA = "const auto release_groups = [&]() {"
RESTORE_LAMBDA = "const auto restore = [&]() noexcept {"
STICKY = "arena_teardown_refusal_.store(zone_arena_teardown_left_closed(earlier) ? earlier : cause,"
COMPLETED = "arena_teardown_refusal_.store(zone_arena_teardown_refusal::NONE"
STOP_BEFORE_FIRST_FREE = "if (released_chunk_count == 0) { break; } continue;"
NOTHING_FREED = "if (unfreed != zone_arena_teardown_refusal::NONE && released_chunk_count == 0) {"
PARTLY_FREED = "if (unfreed != zone_arena_teardown_refusal::NONE) {"
UNFREED_CAUSE = "unfreed = zone_arena_unfreed_chunk_refusal(leases_outstanding, released_chunk_count > 0);"
PARTLY_FREED_EXIT = f"release_groups(); return refuse({UNFREED}); }}"

FREE_LOOP = "for (size_t ci = 0; ci < arena_chunks_.size(); ++ci) {"
LEASE_WAIT = "while ((outstanding = c.lease_count.load()) > 0 && std::chrono::steady_clock::now() <= deadline) {"
LEASED = "const bool leases_outstanding = outstanding > 0;"
LEASE_GUARD = "if (leases_outstanding) {"
TEST_GUARD = "else if (fail_for_test) {"
FREE_BRANCH = "else {"
ABORTS = ("GGML_ASSERT(", "GGML_ABORT(", "abort(", "std::terminate(")

TAKE_OVER = ("if (group.state != allocator_group_state::RESETTING || !group.draining_authorities.empty()) "
             "{ continue; }")
LIVE_IN_GROUP = "const size_t live_in_group = group.allocations.size(); if (live_in_group != 0) {"
GROUP_RESET = "arena_group_reset_allocators_locked(g);"
SETTLE_RESET = "arena_group_reset_allocators_locked(arena_allocator_group_index(zone));"
REPLACEMENT = "settle_replacements[i] = arena_make_authority(zone, arena_generation(zone));"

# Whether an earlier teardown left the arena partly freed, and the two outcomes a refusal that frees no chunk names.
PARTLY_BEFORE = ("const bool partly_freed_before = "
                 "zone_arena_teardown_left_closed(arena_teardown_refusal_.load(std::memory_order_acquire));")
LITERALS = r'((?:"(?:\\.|[^"\\])*" ?)+)'
LEAVES_RE = re.compile(r"const char \* const nothing_freed_leaves = partly_freed_before \? " + LITERALS + r": "
                       + LITERALS + ";")
LEAVES = "nothing_freed_leaves"
LEASE_LINE_FREED = "if (released_chunk_count > 0) {"
LEASE_LINE_PARTLY = "} else if (partly_freed_before) {"

PRIVATE_TESTING = "#if defined(GGML_SYCL_PRIVATE_TESTING)"
HOOK_RE = re.compile(r"\bg_test_\w+")
HOOK_READS = (
    "g_test_arena_free_failure_after.compare_exchange_strong(",
    "g_test_fail_next_arena_drain.exchange(false, std::memory_order_acq_rel)",
)


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


def lambda_body(d: str, opener: str) -> str:
    start = d.find(opener)
    return block_at(d, start + len(opener) - 1) if start != -1 else ""


def restore_body(d: str) -> str:
    return lambda_body(d, RESTORE_LAMBDA)


def enumerators(zsh: str):
    """The enumerators of zone_arena_teardown_refusal, from its declaration in zone-sizing.hpp."""
    c = code(zsh)
    body = block_at(c, c.find(CAUSE_ENUM)).strip("{} ")
    return [re.match(r"\s*(\w+)", part).group(1) for part in body.split(",") if part.strip()]


def move_case(zs: str, cause: str, before: str) -> str:
    """The predicate with `case <cause>:` taken out of its group and put in front of `before`."""
    label = f"case zone_arena_teardown_refusal::{cause}: "
    return mutate_in(mutate_in(zs, PREDICATE_SIG, label, ""), PREDICATE_SIG, before, label + before)


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


def claim_the_allocation_check_holds_every_group_lock(uc: str) -> bool:
    """Every group lock is taken before the live-allocation check and none is dropped between the two, so no
    allocation can appear in a group between the check and the close."""
    d = destroy(uc)
    first_lock = d.find(RELOCK)
    check = d.find(ALLOC_CHECK)
    between = d[first_lock:check] if -1 < first_lock < check else ""
    return bool(between) and check < closing_at(d) and UNLOCK not in between and RELEASE not in between


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
        unlock_before = max(d.rfind(UNLOCK, 0, r), d.rfind(RELEASE, 0, r))
        release_after = d.find(RELEASE, r)
        returns = d.find("return refuse(", r)
        if relock == -1 or unlock_before > relock or release_after == -1 or returns == -1 or release_after > returns:
            return False
    return True


def claim_every_exit_after_the_close_releases_and_wakes(uc: str) -> bool:
    """Every exit after the close, each refusal and the completed teardown, drops the group locks through
    release_groups() on its own path, after its last relock, and release_groups() unlocks every group and then wakes
    every group's waiters: a zone_settle caller parked behind a settle the close interrupted waits on its group's
    state, which the teardown has just decided."""
    d = destroy(uc)
    closing = closing_at(d)
    release = lambda_body(d, RELEASE_LAMBDA)
    exits = [p for _, p in refusals(d) if p > closing]
    done = d.find(COMPLETED, closing)
    if closing == -1 or not exits or done == -1 or not release:
        return False
    previous = closing
    for p in exits + [done]:
        last_release = d.rfind(RELEASE, 0, p)
        if last_release < max(previous, d.rfind(RELOCK, 0, p)):
            return False
        previous = p
    return ordered(release, UNLOCK.strip(), "for (auto & group : arena_allocator_groups_) {",
                   "group.lifecycle_cv.notify_all();")


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
    loop = free_loop(d)
    restored = d[nothing_start:nothing_end]
    partly_block = block_at(d, partly)
    after_free = [m.start() for m in re.finditer(re.escape(RESTORE), d) if m.start() > free.start()]
    return (
        loop.count("break;") == 1
        and STOP_BEFORE_FIRST_FREE in loop
        and len(after_free) == 1
        and nothing_start < after_free[0] < nothing_end
        and f"return refuse({UNFREED});" in restored
        and f"return refuse({UNFREED});" in partly_block
        and RESTORE not in partly_block
    )


def claim_partly_freed_cause_is_sticky(uc: str) -> bool:
    """A refusal keeps an earlier cause that left the arena closed; only a completed teardown clears it."""
    d = destroy(uc)
    return STICKY in d and COMPLETED in d


def claim_nothing_freed_lines_say_what_an_earlier_teardown_left(uc: str) -> bool:
    """A refusal that frees no chunk names one of two outcomes, chosen by whether the recorded cause, read before the
    close, says an earlier teardown left the arena partly freed: still partly freed and closed, or as it was. Both
    restore exits print it, and the lease line makes the same choice."""
    d = destroy(uc)
    m = LEAVES_RE.search(d)
    loop = free_loop(d)
    if m is None or not loop:
        return False
    partly_text, intact_text = literal_text(m.group(1)), literal_text(m.group(2))
    lease_block = block_at(loop, loop.find(LEASE_GUARD))
    sites = [s.start() for s in re.finditer(re.escape(RESTORE), d)]
    return (
        -1 < d.find(PARTLY_BEFORE) < closing_at(d)
        and "partly freed" in partly_text
        and "closed" in partly_text
        and "as it was" not in partly_text
        and intact_text == "the arena is as it was"
        and len(sites) == 2
        and all(LEAVES in d[s : d.find("return refuse(", s)] for s in sites)
        and ordered(lease_block, LEASE_LINE_FREED, LEASE_LINE_PARTLY)
    )


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


def claim_the_restore_cannot_throw(uc: str, mh: str) -> bool:
    """The restore and what it calls to reopen, republish and reset are noexcept: a throw part way through would
    leave the arena half put back."""
    c = code(uc)
    m = code(mh)
    return (
        RESTORE_LAMBDA in destroy(uc)
        and f"{GROUP_RESET_SIG} noexcept {{" in c
        and PUBLISH_RE.search(c) is not None
        and f"{REOPEN_SIG} noexcept {{" in m
        and f"{CLOSE_SIG} noexcept {{" in m
    )


def claim_restore_finishes_an_interrupted_settle(uc: str) -> bool:
    """A group the restore gives back RESETTING with empty drains is finished as the settle would have: its
    allocators reset by the settle's own helper, the replacements built before the close published, then OPEN. A group
    that holds an allocation there is left RESETTING, and its allocators are not reset under it."""
    c = code(uc)
    d = destroy(uc)
    body = restore_body(d)
    take_over = body.find(TAKE_OVER)
    live = body.find(LIVE_IN_GROUP)
    live_block = block_at(body, live) if live != -1 else ""
    return (
        take_over != -1
        and "group.state = prior_states[g];" in body[:take_over]
        and take_over < live < body.find(GROUP_RESET)
        and "continue;" in live_block
        and "arena_group_reset_allocators_locked(" not in live_block
        and "group.state =" not in live_block
        and ordered(body[take_over:], GROUP_RESET, "arena_publish_prebuilt(zone, std::move(settle_replacements[i]));",
                    "group.state = allocator_group_state::OPEN;")
        and -1 < d.find(REPLACEMENT) < closing_at(d)
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
    after = loop[block_end(loop, g, LEASE_GUARD) :].lstrip()
    if not after.startswith(TEST_GUARD):
        return False
    test_block = block_at(after, 0)
    rest = after[block_end(after, 0, TEST_GUARD) :].lstrip()
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


def claim_hooks_are_read_only_in_private_testing_builds(uc: str) -> bool:
    """Every read of a test hook in the teardown sits inside a GGML_SYCL_PRIVATE_TESTING block, so a production
    build takes no failure a test arms, and both the free hook and the drain hook are read."""
    d = destroy(uc)
    reads = [m.start() for m in HOOK_RE.finditer(d)]
    if not reads or not all(r in d for r in HOOK_READS):
        return False
    for p in reads:
        opened = d.rfind(PRIVATE_TESTING, 0, p)
        if opened == -1 or "#endif" in d[opened:p] or "#else" in d[opened:p]:
            return False
    return True


def claim_only_a_partly_freed_arena_stays_closed(zs: str, zsh: str) -> bool:
    """The predicate is a switch with no default that names every cause once, and the causes it reads as closed are
    the two found after a chunk was freed."""
    body = fn_body(code(zs), PREDICATE_SIG)
    switch = body.find("switch (cause) {")
    closed_end = body.find("return true;", switch)
    if switch == -1 or closed_end == -1:
        return False
    closed = set(CASE_RE.findall(body[switch:closed_end]))
    named = CASE_RE.findall(block_at(body, switch))
    return (
        "default:" not in body
        and closed == TERMINAL_CAUSES
        and sorted(named) == sorted(enumerators(zsh))
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


def test_the_allocation_check_holds_every_group_lock():
    assert claim_the_allocation_check_holds_every_group_lock(UC), (
        "arena_destroy() checks for live allocations without every group lock held, so one can appear before the "
        "close")


def test_every_refusal_before_a_free_restores():
    assert claim_every_refusal_before_a_free_restores(UC), (
        "a refusal of arena_destroy() between the close and the first free returns without restore(): the arena "
        "stays closed although nothing was freed")


def test_restore_holds_every_group_lock():
    assert claim_restore_holds_every_group_lock(UC), (
        "arena_destroy() restores without every allocator group lock held, so a group can be seen between CLOSING and "
        "the state it had")


def test_every_exit_after_the_close_releases_and_wakes():
    assert claim_every_exit_after_the_close_releases_and_wakes(UC), (
        "an exit of arena_destroy() after the close drops the group locks without waking the threads parked on a "
        "group's state, so a zone_settle caller waiting behind an interrupted settle hangs")


def test_no_restore_after_a_chunk_is_freed():
    assert claim_no_restore_after_a_chunk_is_freed(UC), (
        "arena_destroy() restores an arena after it freed a chunk, or stops its free loop after one was freed")


def test_partly_freed_cause_is_sticky():
    assert claim_partly_freed_cause_is_sticky(UC), (
        "a later refusal of arena_destroy() overwrites the cause of a partly freed arena")


def test_nothing_freed_lines_say_what_an_earlier_teardown_left():
    assert claim_nothing_freed_lines_say_what_an_earlier_teardown_left(UC), (
        "a refusal of arena_destroy() that frees no chunk says the arena is as it was although an earlier teardown "
        "left it partly freed, or the two outcomes are no longer told apart")


def test_restore_reopens_what_it_retired():
    assert claim_restore_reopens_what_it_retired(UC), (
        "arena_destroy()'s restore does not reopen and republish the authority it retired at its generation, does "
        "not give a group its state and epoch back, or allocates")


def test_the_restore_cannot_throw():
    assert claim_the_restore_cannot_throw(UC, MH), (
        "arena_destroy()'s restore, or what it calls to reopen, republish or reset, is no longer noexcept")


def test_restore_finishes_an_interrupted_settle():
    assert claim_restore_finishes_an_interrupted_settle(UC), (
        "arena_destroy()'s restore leaves a drained settle it interrupted RESETTING, finishes it other than the "
        "settle would, or resets a group that holds an allocation")


def test_a_leased_chunk_is_never_freed():
    assert claim_a_leased_chunk_is_never_freed(UC), (
        "arena_destroy() can pass a chunk a lease still holds to sycl::free")


def test_the_teardown_never_aborts():
    assert claim_the_teardown_never_aborts(UC), (
        "arena_destroy() keeps an assert or abort, so a lease that outlives the wait kills the process")


def test_hooks_are_read_only_in_private_testing_builds():
    assert claim_hooks_are_read_only_in_private_testing_builds(UC), (
        "arena_destroy() reads a test hook outside GGML_SYCL_PRIVATE_TESTING, or no longer reads the free or drain "
        "hook")


def test_only_a_partly_freed_arena_stays_closed():
    assert claim_only_a_partly_freed_arena_stays_closed(ZS, ZSH), (
        "zone_arena_teardown_left_closed() reads a cause other than the two found after a chunk was freed as closed, "
        "has a default, or does not name every cause")


def test_reopen_is_the_inverse_of_close():
    assert claim_reopen_is_the_inverse_of_close(MH), (
        "arena_authority::reopen() does not undo close_and_invalidate(), or changes a record, lease or chunk")


def test_a_break_outside_the_free_loop_is_not_counted():
    # The free loop's own break is the one counted: a break elsewhere in the teardown does not fail the claim.
    assert claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, "drain_snapshot; lock_groups(); try {",
                  "drain_snapshot; lock_groups(); for (;;) { break; } try {")
    )


# ---- mutation witnesses ----


def test_witness_allocations_found_after_closing():
    assert not claim_nothing_changes_before_a_live_allocation_refuses(
        mutate_in(UC, DESTROY_SIG, ALLOC_CHECK,
                  "for (auto & g : arena_allocator_groups_) g.state = allocator_group_state::CLOSING; " + ALLOC_CHECK)
    )


def test_witness_teardown_reserves_after_closing():
    assert not claim_nothing_changes_before_a_live_allocation_refuses(
        mutate_in(UC, DESTROY_SIG, "group.draining_authorities.end()); }" + UNLOCK,
                  "group.draining_authorities.end()); }" + UNLOCK + " drain_snapshot.reserve(drain_snapshot.size() + 1);")
    )


def test_witness_allocations_checked_before_the_locks():
    unlocked = mutate_in(UC, DESTROY_SIG, "drain_snapshot; lock_groups(); try {", "drain_snapshot; try {")
    assert not claim_the_allocation_check_holds_every_group_lock(
        mutate_in(unlocked, DESTROY_SIG, RESTORE_LAMBDA, "lock_groups(); " + RESTORE_LAMBDA)
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
        mutate_in(UC, DESTROY_SIG, "if (!all_authorities_drained) { restore(); release_groups();",
                  "if (!all_authorities_drained) { release_groups(); restore();")
    )


def test_witness_nothing_freed_restore_without_relock():
    assert not claim_restore_holds_every_group_lock(
        mutate_in(UC, DESTROY_SIG, "lock_groups(); " + NOTHING_FREED, NOTHING_FREED)
    )


def test_witness_partly_freed_exit_unlocks_without_waking():
    assert not claim_every_exit_after_the_close_releases_and_wakes(
        mutate_in(UC, DESTROY_SIG, PARTLY_FREED_EXIT, f"unlock_groups(); return refuse({UNFREED}); }}")
    )


def test_witness_completed_teardown_unlocks_without_waking():
    assert not claim_every_exit_after_the_close_releases_and_wakes(
        mutate_in(UC, DESTROY_SIG, "release_groups(); " + COMPLETED, "unlock_groups(); " + COMPLETED)
    )


def test_witness_release_wakes_no_one():
    assert not claim_every_exit_after_the_close_releases_and_wakes(
        mutate_in(UC, DESTROY_SIG, "group.lifecycle_cv.notify_all();", "")
    )


def test_witness_restore_after_a_chunk_was_freed():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, NOTHING_FREED, PARTLY_FREED)
    )


def test_witness_partly_freed_arena_restored():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, PARTLY_FREED_EXIT, "restore(); " + PARTLY_FREED_EXIT)
    )


def test_witness_free_loop_stops_after_a_chunk_was_freed():
    assert not claim_no_restore_after_a_chunk_is_freed(
        mutate_in(UC, DESTROY_SIG, STOP_BEFORE_FIRST_FREE, "break;")
    )


def test_witness_cause_not_sticky():
    assert not claim_partly_freed_cause_is_sticky(
        mutate_in(UC, DESTROY_SIG, STICKY, "arena_teardown_refusal_.store(cause,")
    )


def test_witness_nothing_freed_line_ignores_an_earlier_teardown():
    assert not claim_nothing_freed_lines_say_what_an_earlier_teardown_left(
        mutate_in(UC, DESTROY_SIG, f"unfreed_chunk, {LEAVES});", 'unfreed_chunk, "the arena is as it was");')
    )


def test_witness_drain_timeout_line_ignores_an_earlier_teardown():
    assert not claim_nothing_freed_lines_say_what_an_earlier_teardown_left(
        mutate_in(UC, DESTROY_SIG, f"{DRAIN_TIMEOUT}), {LEAVES});", f'{DRAIN_TIMEOUT}), "the arena is as it was");')
    )


def test_witness_nothing_freed_outcome_ignores_the_recorded_cause():
    assert not claim_nothing_freed_lines_say_what_an_earlier_teardown_left(
        mutate_in(UC, DESTROY_SIG, f"{LEAVES} = partly_freed_before ? ", f"{LEAVES} = false ? ")
    )


def test_witness_nothing_freed_outcomes_not_told_apart():
    assert not claim_nothing_freed_lines_say_what_an_earlier_teardown_left(
        mutate_in(UC, DESTROY_SIG, '? "the arena is still partly freed', '? "the arena is as it was; still partly freed')
    )


def test_witness_lease_line_ignores_an_earlier_teardown():
    assert not claim_nothing_freed_lines_say_what_an_earlier_teardown_left(
        mutate_in(UC, DESTROY_SIG, LEASE_LINE_PARTLY, "} else if (false) {")
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


def test_witness_restore_may_throw():
    assert not claim_the_restore_cannot_throw(
        mutate_in(UC, DESTROY_SIG, RESTORE_LAMBDA, "const auto restore = [&]() {"), MH
    )


def test_witness_reopen_may_throw():
    assert not claim_the_restore_cannot_throw(UC, code(MH).replace(f"{REOPEN_SIG} noexcept", REOPEN_SIG, 1))


def test_witness_close_may_throw():
    assert not claim_the_restore_cannot_throw(UC, code(MH).replace(f"{CLOSE_SIG} noexcept", CLOSE_SIG, 1))


def test_witness_group_reset_may_throw():
    assert not claim_the_restore_cannot_throw(code(UC).replace(f"{GROUP_RESET_SIG} noexcept", GROUP_RESET_SIG, 1), MH)


def test_witness_interrupted_settle_left_resetting():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, DESTROY_SIG, TAKE_OVER, "continue;")
    )


def test_witness_interrupted_settle_opened_without_its_reset():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, DESTROY_SIG, GROUP_RESET, "")
    )


def test_witness_settle_resets_by_its_own_code():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, SETTLE_SIG, SETTLE_RESET, "z.allocator->reset();")
    )


def test_witness_settle_taken_over_with_a_live_allocation():
    assert not claim_restore_finishes_an_interrupted_settle(
        mutate_in(UC, DESTROY_SIG, LIVE_IN_GROUP, "if (false) {")
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


def test_witness_free_hook_read_in_production_builds():
    assert not claim_hooks_are_read_only_in_private_testing_builds(
        mutate_in(UC, DESTROY_SIG, PRIVATE_TESTING + " size_t armed_after", "size_t armed_after")
    )


def test_witness_drain_hook_read_in_production_builds():
    assert not claim_hooks_are_read_only_in_private_testing_builds(
        mutate_in(UC, DESTROY_SIG, "#endif bool all_authorities_drained = true;",
                  "#endif bool all_authorities_drained = !g_test_fail_next_arena_drain.load();")
    )


def test_witness_predicate_reads_a_late_lease_as_restorable():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        move_case(ZS, "LEASES_OUTSTANDING_AFTER_FREE", "case zone_arena_teardown_refusal::NONE:"), ZSH
    )


def test_witness_predicate_reads_a_restorable_cause_as_closed():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        move_case(ZS, "DRAIN_TIMEOUT", "case zone_arena_teardown_refusal::FREE_FAILED:"), ZSH
    )


def test_witness_predicate_has_a_default():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        mutate_in(ZS, PREDICATE_SIG, "return false; }", "return false; default: return false; }"), ZSH
    )


def test_witness_predicate_misses_a_cause():
    assert not claim_only_a_partly_freed_arena_stays_closed(
        mutate_in(ZS, PREDICATE_SIG, "case zone_arena_teardown_refusal::HOST_STATE: ", ""), ZSH
    )


def test_witness_reopen_drops_the_leases():
    assert not claim_reopen_is_the_inverse_of_close(
        mutate_in(MH, REOPEN_SIG, "alive = true;", "alive = true; lease_counts.assign(lease_counts.size(), 0);")
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
