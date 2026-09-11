"""Source contract for llama.cpp-gwno: the oneDNN Graph SYCL allocator that
routes the compiled SDPA partition's per-execute scratch through the unified
cache instead of oneDNN's default SYCL allocator (S4 root cause, jmc5
c-uxch). Host-only, pure text assertions -- no SYCL device required,
matching test-sycl-fattn-packed-k-lifecycle-source.py's pytest-collectible
pattern (llama_test_pytest hands this file to pytest.main(), so checks must
live inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Hardened in the gwno spec-review round 2 (llama.cpp-gwno finding 5): checks
that verify CODE behavior run against comment-stripped text and, where the
call could be re-wrapped by clang-format, whitespace-normalized text --
never a spelling denylist for something that should be a positive structural
check. The one exception is the in-order-queue check, which is deliberately
checking that a PROSE explanation exists in a comment, so it runs against
the full (comment-bearing) text on purpose.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON_HPP = (ROOT / "ggml/src/ggml-sycl/common.hpp").read_text()
CACHE_HPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
CACHE_CPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
# llama.cpp-0oxf: plain file I/O, not the codescout index -- CLAUDE.md
# documents that index as blind inside this specific ~60k-line file, so a
# tool-assisted search here would silently miss real occurrences.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
# llama.cpp-c6ah: source for the doc-presence check below.
MEMORY_DESIGN_MD = (ROOT / "docs/backend/sycl-memory-design.md").read_text()


# One left-to-right pass over string literals, char literals, // comments and
# /* */ comments. The ORDER matters and is why this is a single alternation
# rather than two sequential re.sub() calls: unified-cache.cpp has a // comment
# that mentions a glob like `weight-reclaim/*`, and a block-comment pass run
# FIRST treats that `/*` as an opener and swallows everything up to the next
# `*/` -- measured: 6259 lines (2760-9019), including the high-water log this
# file asserts on. Scanning left-to-right, the `//` wins because it starts
# first, and a `/*` or `//` inside a string literal is consumed as part of the
# literal instead of opening a comment.
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments so a substring check can't be fooled by
    prose that quotes a call the code explicitly does NOT make. String and
    char literals are preserved verbatim (a positive check may look for a log
    format string). Newlines inside a dropped block comment are kept so line
    structure survives for diagnostics."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return " " if tok.startswith("//") else re.sub(r"[^\n]", " ", tok)

    return _LEXEME_RE.sub(repl, src)


def strip_literals(src: str) -> str:
    """Blank the CONTENTS of string/char literals (keeping the quotes) so a
    NEGATIVE check -- 'this body does not call sycl::free()' -- is not tripped
    by a log message that names the call it refuses to make. Apply on top of
    strip_comments()."""
    return re.sub(r'"(?:\\.|[^"\\\n])*"', '""', src)


def extract_function_body(src: str, signature_anchor: str) -> str:
    """Return the brace-balanced body (including the braces) of the function
    whose signature contains signature_anchor, scanning from the first '{'
    at or after the anchor to its matching '}'."""
    start = src.index(signature_anchor)
    brace_start = src.index("{", start)
    depth = 0
    for i in range(brace_start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace_start : i + 1]
    raise AssertionError(f"unbalanced braces extracting function body for {signature_anchor!r}")


def normalize_ws(s: str) -> str:
    """Collapse whitespace runs to a single space, so a match survives
    clang-format re-wrapping a call across lines when an unrelated edit
    changes surrounding line lengths."""
    return re.sub(r"\s+", " ", s).strip()


# llama.cpp-c6ah: CACHE_BACKING is mintable through exactly two
# allowlisted unified_cache_adopt_raw_host_allocation(cache_backing=true)
# call sites -- the cache's own staging buffer, and the oneDNN Graph-scratch
# pool's completion-flag slab this ticket added as the second, reviewed mint
# (allocation-provenance.hpp; docs/design/sycl-canonical-memory-architecture.md
# section 3.1). This is the REGISTERED half of that gate, independently
# re-deriving the same allowlist from unified-cache.cpp's actual call sites;
# tests/test-sycl-owner-allocation-migration.py enforces the identical
# allowlist too (unregistered -- see that file), and the two must never
# disagree.
ADOPT_MINT_HELPER = "unified_cache_adopt_raw_host_allocation"
ADOPT_CACHE_BACKING_ARG = 6  # 0-based: ptr, size, queue, role, category, cohort_id, cache_backing
ADOPT_COHORT_ARG = 5  # 0-based: ptr, size, queue, role, category, cohort_id
ADOPT_CACHE_BACKING_ALLOWLIST = (
    '"unified_cache:staging"',
    '"unified_cache:onednn_graph_scratch_flag_slab"',
)


def _split_top_level_arguments(argument_text: str) -> list:
    """Split a call's argument text on top-level commas only, respecting
    nested (), [], {} and not splitting inside a string literal."""
    arguments, depth, current, in_string = [], 0, "", False
    for char in argument_text:
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                arguments.append(current.strip())
                current = ""
                continue
        current += char
    if current.strip():
        arguments.append(current.strip())
    return arguments


def _adopt_cache_backing_cohorts(code: str) -> list:
    """Cohort tags (with their quotes) of every ADOPT_MINT_HELPER call in
    `code` whose cache_backing argument is the literal `true`. `code` must
    already be comment-stripped (see strip_comments()), so a comment merely
    naming the helper in prose is never mistaken for a call."""
    cohorts = []
    for match in re.finditer(r"\b%s\s*\(" % ADOPT_MINT_HELPER, code):
        if re.search(r"alloc_handle[ \t]+$", code[max(0, match.start() - 40):match.start()]):
            continue  # a forward declaration/signature line, not a call
        depth, index = 1, match.end()
        while index < len(code) and depth:
            if code[index] == "(":
                depth += 1
            elif code[index] == ")":
                depth -= 1
            index += 1
        arguments = _split_top_level_arguments(code[match.end():index - 1])
        if len(arguments) > ADOPT_CACHE_BACKING_ARG and arguments[ADOPT_CACHE_BACKING_ARG] == "true":
            cohorts.append(arguments[ADOPT_COHORT_ARG] if len(arguments) > ADOPT_COHORT_ARG else "<missing>")
    return cohorts


# llama.cpp-c6ah: the withdrawn "the bare query lies once a
# watcher/host_task exists" reading (see docstring's "Hardened" note --
# c6ah's own history, not a hypothetical) must stay HISTORY-framed wherever
# it is still mentioned in prose, never restated as the current explanation.
# Both spellings actually used in this codebase's comments ("LIES" as a
# claim, "lying" in "there is no lying query") must be caught -- a
# single-spelling word boundary would leave the other silently unchecked.
_LIES_WORD_RE = re.compile(r"\bl(?:ies|ying)\b", re.IGNORECASE)
# A boundary a framing check should not cross: the end of a sentence
# ("word. " or "word." at end of text) or a blank "//" comment line (a
# paragraph break). Bounding a match's window at the NEAREST such boundary
# in each direction -- rather than a fixed character count -- means framing
# only counts when it is in the same sentence, or at worst the same
# paragraph if no blank comment line intervenes; a fixed-size window would
# also credit an unframed sentence merely sitting near an unrelated framed
# one.
_SENTENCE_OR_PARAGRAPH_BOUNDARY_RE = re.compile(r"\.(?:\s|$)|\n[ \t]*//[ \t]*\n")


def _sentence_bounded_window(text: str, start: int, end: int) -> str:
    left = 0
    for m in _SENTENCE_OR_PARAGRAPH_BOUNDARY_RE.finditer(text, 0, start):
        left = m.end()
    right_match = _SENTENCE_OR_PARAGRAPH_BOUNDARY_RE.search(text, end)
    right = right_match.end() if right_match else len(text)
    return text[left:right]


def _withdrawn_lies_phrasing_is_history_framed(raw_text: str) -> bool:
    """True iff every occurrence of "lies"/"lying" in raw_text (the RAW,
    comment-bearing text -- this checks comment PROSE, not code, so it must
    not run against a comment-stripped copy) that is actually ABOUT A QUERY
    (its own sentence/paragraph window also contains "quer", matching
    "query"/"queries") sits inside that same window as "earlier" or
    "misdiagnos". A "lies"/"lying" mention with no "quer" nearby is ordinary
    English (e.g. "the slab lies before the queue") -- not the withdrawn
    reading at all, and skipped rather than demanding framing it has no
    reason to carry. Vacuously true if no query-related mention exists --
    this check exists to catch a REINTRODUCTION of the withdrawn reading
    without its framing, not to require the mention to exist."""
    for match in _LIES_WORD_RE.finditer(raw_text):
        window = _sentence_bounded_window(raw_text, match.start(), match.end())
        if not re.search(r"quer", window, re.IGNORECASE):
            continue
        if not re.search(r"earlier|misdiagnos", window, re.IGNORECASE):
            return False
    return True


# llama.cpp-c6ah: the two MEASURED facts that replaced the
# withdrawn reading, as currently worded in docs/backend/sycl-memory-design.md.
# Whitespace-normalized on both sides (normalize_ws) so a markdown reflow
# that does not change the words themselves cannot break this check.
DESIGN_DOC_MEASURED_FACT_QUERY_BLOCKS = (
    "`command_execution_status` query BLOCKS rather than polls on any profiling-enabled queue"
)
DESIGN_DOC_MEASURED_FACT_HOST_TASK_BLOCKS_SUBMITTER = (
    "SUBMITTING a host_task whose `depends_on()` names an event from ANOTHER queue BLOCKS THE SUBMITTING "
    "THREAD until that event completes"
)


def _design_doc_states_measured_facts(doc_text: str = MEMORY_DESIGN_MD) -> bool:
    # llama.cpp-c6ah: `doc_text` defaults to the real doc but accepts a
    # substitute so a mutation witness can call this SAME function against
    # mutated text, rather than re-deriving its own separate check that
    # could silently drift from what the real check actually does.
    normalized = normalize_ws(doc_text)
    return (
        normalize_ws(DESIGN_DOC_MEASURED_FACT_QUERY_BLOCKS) in normalized
        and normalize_ws(DESIGN_DOC_MEASURED_FACT_HOST_TASK_BLOCKS_SUBMITTER) in normalized
    )


BLOCKING_TOKENS = (
    ".wait(",
    "wait_and_throw",
    "event_complete(",
    "get_info<sycl::info::event::command_execution_status>",
)


def _has_no_blocking_token(body: str) -> bool:
    # Negative check: look at code tokens only, never at what a log string
    # says about them.
    code = strip_literals(body)
    return not any(token in code for token in BLOCKING_TOKENS)


RELEASE_COMPLETE_CALL = "onednn_graph_scratch_pool_entry_release_complete("


def _routes_through_release_complete(body: str) -> bool:
    """llama.cpp-c6ah: true iff `body` calls the single choke point every
    pool-entry completion check must use, and calls NO bare event_complete()
    anywhere in its own text -- a positive-presence check alone would still
    pass if a redundant direct query were added alongside it (the same
    reasoning _has_no_blocking_token() above already applies one level up,
    to the wider set of blocking tokens)."""
    code = strip_literals(body)
    return RELEASE_COMPLETE_CALL in code and "event_complete(" not in code


FLAG_SLAB_OWNER_RESET_STMT = "onednn_graph_scratch_flag_slab_owner_ = {};"
DRAIN_CALL = "drain_all_queues_noexcept()"


def _flag_slab_owner_released_after_drain(shutdown_resources_body: str) -> bool:
    """llama.cpp-me60: true iff shutdown_resources()'s own body resets
    onednn_graph_scratch_flag_slab_owner_ to {} strictly AFTER its own
    drain_all_queues_noexcept() call -- the slab must be released on the
    NORMAL teardown path (not left to ~unified_cache() member destruction,
    which runs after shutdown_unified_cache()'s registry sweep and used to
    make that sweep refuse), and only once every queue has been drained, so
    no marker kernel can still be targeting the slab when it is released.
    This body also resets the same field on its two early-return
    (SYCL-already-gone) branches, both of which sit BEFORE the drain call --
    scoping the search to text AFTER the drain call's own (unique) position
    is what keeps those two irrelevant to this specific property."""
    drain_idx = shutdown_resources_body.find(DRAIN_CALL)
    if drain_idx == -1:
        return False
    return FLAG_SLAB_OWNER_RESET_STMT in normalize_ws(shutdown_resources_body[drain_idx:])


PRETEARDOWN_CENSUS_CALL = 'snapshot_allocation_controls("pre-cache-teardown"'
POOL_RECLAIM_CALL = "onednn_graph_scratch_reclaim_pool("


def _pool_reclaimed_before_preteardown_census(shutdown_unified_cache_body: str) -> bool:
    """llama.cpp-me60: true iff shutdown_unified_cache()'s own body calls
    onednn_graph_scratch_reclaim_pool() (on each live cache) BEFORE its
    census call snapshot_allocation_controls(..., preteardown=true) --
    predating this ticket (traced to llama.cpp-0oxf), a DIRECT Graph-scratch
    buffer parked in the reuse pool keeps its own EXTERNAL_EXACT allocation
    control alive, and that census refuses shutdown on ANY live
    non-CACHE_BACKING control still standing when it runs. Uses the FIRST
    occurrence of each anchor (via str.find(), not a mere "one occurs
    somewhere before the other" scan) so a later, correctly-ordered pair
    cannot mask an earlier, wrongly-ordered one -- see this check's own
    mutation witness below, which exercises exactly that."""
    normalized = normalize_ws(shutdown_unified_cache_body)
    reclaim_idx = normalized.find(POOL_RECLAIM_CALL)
    census_idx = normalized.find(PRETEARDOWN_CENSUS_CALL)
    return reclaim_idx != -1 and census_idx != -1 and reclaim_idx < census_idx


PRETEARDOWN_SHUTTING_DOWN_GUARD = "if (!ggml_sycl_is_shutting_down())"
CACHES_LOOP_STMT = "for (auto & item : caches)"


def _preteardown_pool_loop_guard_block(shutdown_unified_cache_body: str) -> str:
    """llama.cpp-3lgu (F7): return the brace-balanced block of the FIRST (and
    only) `if (!ggml_sycl_is_shutting_down())` guard in
    shutdown_unified_cache() -- the pre-census oneDNN Graph-scratch
    drain+reclaim pass -- or "" if that guard is missing entirely. Reused by
    both the F7 check below (does the drain+reclaim loop sit inside it) and
    the F2 check further down (does the loop's own queue-validity probe sit
    inside it too). Normalizes whitespace first, like the sibling
    positional helper _pool_reclaimed_before_preteardown_census() above, so
    a clang-format re-wrap of the guard's own line cannot break the
    match."""
    normalized = normalize_ws(shutdown_unified_cache_body)
    if PRETEARDOWN_SHUTTING_DOWN_GUARD not in normalized:
        return ""
    return extract_function_body(normalized, PRETEARDOWN_SHUTTING_DOWN_GUARD)


def _preteardown_pool_loop_skipped_once_shutting_down(shutdown_unified_cache_body: str) -> bool:
    """llama.cpp-3lgu (F7): true iff the pre-census drain+reclaim loop (the
    FIRST `for (auto & item : caches)` in shutdown_unified_cache() -- the
    second, further down, is the later per-cache shutdown_resources()
    teardown loop, which has its own separate guarding) is nested inside
    `if (!ggml_sycl_is_shutting_down())`, not bare. A true flag on entry
    means shutdown_resources() will abandon cleanup on every live cache's
    own equivalent branch anyway, so this earlier pass has nothing left to
    reclaim, and this whole-pass skip has no validity probe of its own --
    the per-cache probe F2 added is inside the loop to protect a
    drain/reclaim call against an already-torn-down context."""
    guard_block = _preteardown_pool_loop_guard_block(shutdown_unified_cache_body)
    return CACHES_LOOP_STMT in guard_block


QUEUE_CONTEXT_PROBE_CALL = "get_context()"
TRY_STMT = "try {"
CATCH_ALL_STMT = "catch (...)"
SHUTTING_DOWN_STORE_TRUE = "g_sycl_shutting_down.store(true"
CATCH_CONTINUE_STMT = "continue;"


def _preteardown_loop_probes_queue_validity_before_drain(shutdown_unified_cache_body: str) -> bool:
    """llama.cpp-3lgu (F2): true iff, inside the pre-census drain+reclaim
    loop (see _preteardown_pool_loop_guard_block()), each cache's own
    queue-context validity is probed -- a `try { ... get_context(); } catch
    (...) { ... g_sycl_shutting_down.store(true, ...); continue; ... }`
    guard, mirroring shutdown_resources()'s own probe (unified-cache.cpp
    ~4557) -- and that probe's try/get_context()/catch/store/continue
    sequence appears strictly BEFORE the loop's own
    drain_all_queues_noexcept() call. The store and the continue are both
    load-bearing halves of the catch body, not decoration: a catch clause
    that only swallows the exception (no try/catch shape check alone can
    tell the difference) would let this cache's own later
    shutdown_resources() call (in the teardown loop further down) see a
    still-false flag and proceed as if the context were valid, and a catch
    clause that stores the flag but does not then `continue` would still
    fall through to drain_all_queues_noexcept() and the reclaim call for
    THIS cache against its own already-invalid context -- the flag only
    protects every cache probed *after* this one in the loop, not this one,
    unless the continue actually skips its own drain+reclaim. Without the
    whole probe, a context torn down while g_sycl_shutting_down is still
    false reaches drain_all_queues_noexcept() (which swallows the throw)
    and then this cache's pool reclaim, whose mem_handle releases would
    attempt a real free against an already-invalid context. Uses the FIRST
    occurrence of each anchor within the loop body via str.find() so a
    later, correctly-ordered probe cannot mask an earlier, missing one --
    see this check's own mutation witness below. The continue is searched
    for starting from the store, not from the loop head, because the loop
    also has its own earlier, unrelated `if (!item.second) { continue; }`
    that must not be mistaken for this one. Finding *some* continue;
    between the store and the drain call is necessary but not sufficient
    -- an unrelated continue placed just before drain_all_queues_noexcept()
    (in an `if`, say) would satisfy a purely positional check while
    leaving the catch itself unable to skip this cache's own
    drain+reclaim. Both continue checks are kept. Containment alone
    would accept a reordered catch body -- `{ continue;
    g_sycl_shutting_down.store(...); }`, whose store is dead code --
    because the continue is still inside the braces; the positional
    search from the store is what requires the continue to FOLLOW the
    store."""
    guard_block = _preteardown_pool_loop_guard_block(shutdown_unified_cache_body)
    loop_idx = guard_block.find(CACHES_LOOP_STMT)
    if loop_idx == -1:
        return False
    loop_body = normalize_ws(guard_block[loop_idx:])
    try_idx = loop_body.find(TRY_STMT)
    probe_idx = loop_body.find(QUEUE_CONTEXT_PROBE_CALL)
    catch_idx = loop_body.find(CATCH_ALL_STMT)
    drain_idx = loop_body.find(DRAIN_CALL)
    if -1 in (try_idx, probe_idx, catch_idx, drain_idx):
        return False
    if not (try_idx < probe_idx < catch_idx < drain_idx):
        return False
    store_idx = loop_body.find(SHUTTING_DOWN_STORE_TRUE, catch_idx)
    if store_idx == -1 or not store_idx < drain_idx:
        return False
    cont_idx = loop_body.find(CATCH_CONTINUE_STMT, store_idx)
    if cont_idx == -1 or not cont_idx < drain_idx:
        return False
    return CATCH_CONTINUE_STMT in extract_function_body(loop_body, CATCH_ALL_STMT)


COMMON_HPP_CODE = strip_comments(COMMON_HPP)
CACHE_HPP_CODE = strip_comments(CACHE_HPP)
CACHE_CPP_CODE = strip_comments(CACHE_CPP)
GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP)

# Comments are stripped from the WHOLE file FIRST, and every function body is
# then extracted from that already-stripped text -- never the other order
# (llama.cpp-gwno spec-review round 3, finding 4). extract_function_body()'s
# brace count is naive: run on the RAW source, a comment containing a stray
# '{' or '}' (a code snippet in prose, say) can desync it and extract the
# wrong span before strip_comments() ever runs on the result. Stripping first
# turns every such character to whitespace before the scan starts, so the
# scan cannot be confused by comment content at all. (Verified inert against
# today's five function bodies either way -- none currently contains a brace
# inside a comment -- which is exactly why the wrong order was invisible
# until reviewed for it.)
FREE_BODY_CODE = extract_function_body(CACHE_CPP_CODE, "void unified_cache::onednn_graph_scratch_free(")
ALLOC_BODY_CODE = extract_function_body(CACHE_CPP_CODE, "void * unified_cache::onednn_graph_scratch_alloc(")
# The DIRECT (non-arena) allocation tail, factored out of onednn_graph_scratch_alloc()
# into its own function -- this is where the actual unified_alloc() call
# lives now, so the "tries the pool before a fresh allocation" ordering
# check below needs both bodies concatenated in call order.
DIRECT_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "void * unified_cache::onednn_graph_scratch_alloc_direct_locked("
)
# llama.cpp-0oxf round-5 finding G2: the real pop-and-reuse attempt and the
# wait loop's peek must apply the IDENTICAL usability test (both routed
# through onednn_graph_scratch_entry_usable_locked()) or the peek can
# silently diverge again the way it did before this round -- see that
# function's own comment in unified-cache.hpp. Extracted separately from
# POOL_SIZE_READY_BODY_CODE below -- the sibling this file's "both call the
# shared predicate" checks pair it with -- so those checks read against
# exactly the two functions the invariant is actually about. (A third
# extracted body, WAIT_HEADROOM_BODY_CODE, is used separately below only for
# the pqgl eviction-sweep-ordering check, not this pair.)
TRY_REUSE_POOL_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "bool unified_cache::onednn_graph_scratch_try_reuse_pool_locked("
)
POOL_SIZE_READY_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "bool unified_cache::onednn_graph_scratch_pool_size_ready_locked("
)
# Anchor on the open paren only, not the full parameter list: llama.cpp-0oxf
# changed this function's signature twice (first to a single planner_n_ctx
# argument, then to (n_head, n_ubatch, n_ctx) after the lead's measurement
# showed n_ctx alone was the wrong independent variable) -- anchoring past
# the paren would need updating again on the next parameter-list edit, and
# the open paren alone is still a unique match: the anchor's trailing "(" is
# what keeps it from also matching onednn_graph_scratch_zone_floor_bytes_swa(
# below (llama.cpp-o3a0) -- that name is longer, so the literal substring
# "...zone_floor_bytes(" never occurs inside it.
#
# llama.cpp-o3a0: this is now a THIN WRAPPER delegating into the window-aware
# onednn_graph_scratch_zone_floor_bytes_swa() (see FLOOR_SWA_BODY_CODE
# below) -- checks about the FORMULA itself (the 64 MiB constant, the
# min/max window logic) must read FLOOR_SWA_BODY_CODE, not this one; checks
# about the DELEGATION itself belong here.
FLOOR_BODY_CODE = extract_function_body(CACHE_CPP_CODE, "static size_t onednn_graph_scratch_zone_floor_bytes(")
# llama.cpp-o3a0: the window-aware formula's own body -- see the comment
# above FLOOR_BODY_CODE for why the two anchors cannot collide.
FLOOR_SWA_BODY_CODE = extract_function_body(CACHE_CPP_CODE, "static size_t onednn_graph_scratch_zone_floor_bytes_swa(")
# llama.cpp-o3a0: the WITH-FLOOR getter that
# actually calls the floor formula with the shape's SWA fields -- scoping
# the "caller passes the swa fields" check (below) to this one function's
# body, rather than the whole file, so a partial revert of just this
# caller (leaving the unrelated DIRECT-allocation-failure error log's own
# call to the same fields untouched) still fails the check.
WITH_FLOOR_GETTER_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "size_t unified_cache_get_planned_onednn_scratchpad_bytes(int device_id) {"
)
MAKE_ENGINE_BODY_CODE = extract_function_body(COMMON_HPP_CODE, "dnnl::engine make_engine(sycl::queue * q) {")
# llama.cpp-pqgl: the size>cap early-out's ordering relative to the eviction
# sweep, both inside this one function.
WAIT_HEADROOM_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "bool unified_cache::onednn_graph_scratch_wait_for_direct_headroom_locked("
)
# llama.cpp-c6ah: the three functions that actually decide whether a pooled
# entry's release is done. Extracted separately (not reused from
# TRY_REUSE_POOL_BODY_CODE/POOL_SIZE_READY_BODY_CODE above, which call
# onednn_graph_scratch_entry_usable_locked() rather than the completion
# check itself) so the checks below can assert exactly where the query
# lives and where it does not.
ENTRY_USABLE_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "unified_cache::onednn_graph_scratch_entry_fit unified_cache::onednn_graph_scratch_entry_usable_locked("
)
EVICT_UNTIL_FITS_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "bool unified_cache::onednn_graph_scratch_evict_pool_until_fits_locked("
)
CLEAR_POOL_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "void unified_cache::onednn_graph_scratch_clear_pool_locked("
)
# llama.cpp-c6ah: the choke point's OWN body, extracted
# separately from the three caller bodies above -- those checks assert what
# the CALLERS query (RELEASE_COMPLETE_CALL, never a bare event_complete());
# this one asserts what the choke point ITSELF does, specifically that its
# only remaining event_complete() call is the flag_slot == -1 fallback
# (the armed predicate was renamed from a release_done
# std::shared_ptr<std::atomic<bool>> to a flag_slot/flag_generation pair
# into a host-USM slab; the shape of this check -- exactly one call, no
# hook branch -- is unchanged), not a second call reintroduced by a RED-arm
# hook branch (the earlier, incorrect design this redesign replaced -- see
# the function's own comment for why a check-time hook branch could not
# reproduce pre-fix behaviour).
RELEASE_COMPLETE_BODY_CODE = extract_function_body(
    CACHE_CPP_CODE, "bool unified_cache::onednn_graph_scratch_pool_entry_release_complete("
)


def test_onednn_graph_allocator_source_contract() -> None:
    checks = {}

    # All four of these are about make_engine()'s own behavior, so all four
    # are scoped to its extracted body (llama.cpp-gwno spec-review round 3,
    # nit 9) rather than searched across the whole ~7000-line common.hpp,
    # where a same-named call or string could exist elsewhere by coincidence.
    #
    # The engine must actually be built WITH an allocator, not the bare
    # dnnl::sycl_interop::make_engine() this whole change exists to stop
    # using as the default path.
    checks["engine built with allocator"] = (
        "dnnl::graph::sycl_interop::make_engine_with_allocator(dev, ctx, alloc)" in MAKE_ENGINE_BODY_CODE
    )
    # Whitespace-insensitive: this call is the one most likely to get
    # re-wrapped by clang-format when an unrelated edit changes the
    # surrounding line lengths, which would otherwise silently break an
    # exact-text check without the call itself having changed.
    checks["allocator constructed from the two callbacks"] = (
        "dnnl::graph::sycl_interop::make_allocator( ggml_sycl::onednn_graph_sycl_malloc, "
        "ggml_sycl::onednn_graph_sycl_free)" in normalize_ws(MAKE_ENGINE_BODY_CODE)
    )
    checks["opt-out env var gates it"] = "onednn_graph_allocator_enabled()" in MAKE_ENGINE_BODY_CODE
    checks["raw allocator engine kept as the fallback"] = (
        "dnnl::sycl_interop::make_engine(dev, ctx)" in MAKE_ENGINE_BODY_CODE
    )

    # Callback signatures declared as free functions (no user-data slot in
    # the oneDNN C API to carry a `this` -- see the header comment).
    checks["malloc callback declared"] = (
        "void * onednn_graph_sycl_malloc(size_t size, size_t alignment, const void * dev, const void * ctx);"
        in CACHE_HPP_CODE
    )
    checks["free callback declared"] = (
        "onednn_graph_sycl_free(void * buf, const void * dev, const void * ctx, void * event);" in CACHE_HPP_CODE
    )
    checks["opt-out env var declared"] = "bool onednn_graph_allocator_enabled();" in CACHE_HPP_CODE

    # Backing storage: served from the same ONEDNN VRAM zone the primitive-API
    # scratchpad (reserve_onednn_scratch) already uses, not a fresh raw
    # allocation on every call. Scoped to each function's own body rather
    # than the whole file, so this can't accidentally match an unrelated
    # zone_alloc/zone_free call site elsewhere.
    checks["malloc method declared"] = (
        "void * onednn_graph_scratch_alloc(size_t size, size_t alignment, sycl::queue * q);" in CACHE_HPP_CODE
    )
    checks["free method declared"] = (
        "onednn_graph_scratch_free(void * ptr, const sycl::event * event);" in CACHE_HPP_CODE
    )
    # llama.cpp-0oxf round-6 finding H5: the local this call passes as the
    # third argument was renamed from `align` to `alignment` (normalized in
    # place from the parameter of the same name) so onednn_graph_scratch_alloc()
    # matches the rest of the family's parameter naming.
    checks["malloc routes through the ONEDNN zone"] = (
        "zone_alloc(vram_zone_id::ONEDNN, size, alignment)" in ALLOC_BODY_CODE
    )
    checks["free routes through the ONEDNN zone"] = "zone_free(vram_zone_id::ONEDNN, ptr)" in FREE_BODY_CODE

    # Zone-backed reclaim is IMMEDIATE, deliberately with no event wait --
    # device-side reuse safety comes from every Graph-scratch consumer
    # submitting on the SAME in-order compute queue (ctx.stream()), not from
    # a host-side completion check. This must stay documented (the
    # assumption it rests on, and what to do if it ever stops holding), not
    # just implemented silently. Deliberately checked against the FULL
    # (comment-bearing) text -- this check is about the prose existing, not
    # about code behavior, so stripping comments would make it un-satisfiable.
    checks["zone reclaim documents the in-order-queue assumption"] = (
        "ctx.stream()" in CACHE_CPP and "in-order compute queue" in CACHE_CPP
    )

    # POSITIVE structural check, not a spelling denylist: extract
    # onednn_graph_scratch_free()'s actual body (brace-balanced, comments
    # stripped) and assert that NONE of the tokens that would make it block
    # appear anywhere in it. An earlier version of this check enumerated
    # three exact call spellings (event_complete(*event), event_complete(it->...,
    # event_complete(event)) -- a denylist that silently stops covering a 4th
    # spelling, a helper wrapping the same blocking call, or a plain .wait().
    # event_complete() itself calls get_info<command_execution_status>(),
    # which is what actually blocks on this queue (see get_dma_queue()'s
    # comment) -- included directly so a rewritten check that skips the
    # event_complete() name but calls get_info directly still gets caught.
    checks["no blocking token anywhere in the free path body"] = _has_no_blocking_token(FREE_BODY_CODE)

    # llama.cpp-0oxf changed WHEN this call fires: it used to be the DIRECT
    # branch's only release mechanism, unconditionally. Since the pool
    # redesign it fires only when a size bucket is already at its bounded
    # depth (onednn_graph_scratch_pool_depth_per_size()) -- the common case
    # instead parks the buffer in onednn_graph_scratch_reuse_pool_ (checked
    # below). The call text itself is unchanged, so this stays a valid
    # (if now narrower) structural check: the overflow release path must
    # still exist and still be event-gated, not silently dropped.
    checks["direct fallback overflow release deferred via retain_handles_until_event"] = normalize_ws(
        "retain_handles_until_event({ std::move(entry.owner) }, *event);"
    ) in normalize_ws(FREE_BODY_CODE)

    # llama.cpp-0oxf pool redesign: a freed DIRECT buffer's PRIMARY fate is
    # the size-bucketed reuse pool, not the shared drain worker -- verified
    # here as a source contract because the whole point of the redesign (the
    # zone floor cannot actually be reached at runtime, so the DIRECT path
    # is many models' steady state, not an occasional fallback) depends on
    # this, not on the overflow path above alone.
    checks["free path parks into the reuse pool"] = (
        "onednn_graph_scratch_reuse_pool_[freed_size]" in normalize_ws(FREE_BODY_CODE)
        and "bucket.push_back(" in normalize_ws(FREE_BODY_CODE)
    )
    # Ordering, not just presence: concatenate the two bodies in the order
    # they run at call time (onednn_graph_scratch_alloc() calls into
    # onednn_graph_scratch_alloc_direct_locked() only once neither the zone
    # nor the pool could serve the request) and assert the FIRST pool-lookup
    # call appears before the FIRST unified_alloc() call -- a check that
    # only asserted presence of the pool-lookup call would still pass if the
    # ordering regressed (a fresh allocation attempted first, the pool only
    # consulted afterward). strip_literals() first (llama.cpp-0oxf round-4
    # finding F3): CACHE_CPP_CODE is comment-stripped but not
    # literal-stripped, and the give-up-waiting GGML_LOG_ERROR just before
    # the real unified_alloc() call NAMES "unified_alloc(" in its own
    # message text -- read code, not messages, the same rule this file
    # already applies to the negative blocking-token/sycl::free checks.
    alloc_path_code = strip_literals(ALLOC_BODY_CODE + DIRECT_BODY_CODE)
    # .find() (not .index()): an absent token must fail THIS named check, not
    # raise an unguarded ValueError that pytest would report as a collection
    # error on a check that never ran, masking which assertion actually failed.
    pool_probe_pos = alloc_path_code.find("onednn_graph_scratch_try_pool_locked(")
    fresh_alloc_pos = alloc_path_code.find("unified_alloc(")
    checks["alloc path tries the reuse pool before a fresh allocation"] = (
        pool_probe_pos != -1 and fresh_alloc_pos != -1 and pool_probe_pos < fresh_alloc_pos
    )
    # Narrower than the concatenated check above: that one only proves the
    # alloc()-level pool probe precedes SOME fresh allocation somewhere
    # across either body -- it would not catch a re-check inside
    # onednn_graph_scratch_alloc_direct_locked() itself moved to after ITS
    # OWN unified_alloc() call, since the earlier alloc()-level probe in
    # ALLOC_BODY_CODE would still make the concatenated check pass. Assert
    # the same ordering again scoped to DIRECT_BODY_CODE alone.
    direct_code_no_literals = strip_literals(DIRECT_BODY_CODE)
    direct_probe_pos = direct_code_no_literals.find("onednn_graph_scratch_try_pool_locked(")
    direct_alloc_pos = direct_code_no_literals.find("unified_alloc(")
    checks["DIRECT body's own re-check precedes its own fresh allocation"] = (
        direct_probe_pos != -1 and direct_alloc_pos != -1 and direct_probe_pos < direct_alloc_pos
    )
    # llama.cpp-pqgl: the size>cap early-out must run AFTER the eviction sweep,
    # not before it -- an earlier version of this fix returned before ever
    # calling the sweep for an oversized request, which skips a real VRAM
    # release (the sweep evicts event-complete pooled entries unconditionally,
    # even for a request it can never make fit) right before the caller's
    # fresh unified_alloc(). Ordering, not just presence, mirrors the
    # pool-before-fresh-allocation checks above: a presence-only check would
    # still pass if the early-out moved back ahead of the sweep.
    wait_headroom_code = strip_literals(WAIT_HEADROOM_BODY_CODE)
    evict_sweep_pos = wait_headroom_code.find("onednn_graph_scratch_evict_pool_until_fits_locked(")
    size_cap_match = re.search(r"size\s*>\s*cap", wait_headroom_code)
    size_cap_pos = size_cap_match.start() if size_cap_match else -1
    checks["oversized early-out runs after the eviction sweep, not before it"] = (
        evict_sweep_pos != -1 and size_cap_pos != -1 and evict_sweep_pos < size_cap_pos
    )
    # llama.cpp-0oxf round-5 finding G2: the pop-and-reuse attempt and the
    # wait loop's peek must not be free to diverge again on what counts as
    # "usable" -- assert both actually route through the one shared
    # predicate, and that the peek's own declaration still accepts the
    # alignment/device_id it needs to apply that predicate (a peek that lost
    # those parameters back to a size-only signature would silently regress
    # to the old, laxer test even with the predicate call still present
    # elsewhere).
    checks["try_reuse_pool_locked() body calls the shared usability predicate"] = (
        "onednn_graph_scratch_entry_usable_locked(" in TRY_REUSE_POOL_BODY_CODE
    )
    checks["pool_size_ready_locked() body calls the shared usability predicate"] = (
        "onednn_graph_scratch_entry_usable_locked(" in POOL_SIZE_READY_BODY_CODE
    )
    checks["pool_size_ready_locked() declaration still takes alignment and device_id"] = (
        "onednn_graph_scratch_pool_size_ready_locked(size_t size, size_t alignment, int device_id) const;"
        in normalize_ws(CACHE_HPP_CODE)
    )
    # llama.cpp-0oxf round-6 finding H3: onednn_graph_scratch_entry_usable_locked()'s
    # own declaration now says callers must NOT call event_complete() on an
    # entry themselves before calling it (the predicate already performs
    # that query internally) -- both call sites must actually honor that,
    # not just call the predicate (checked above) while ALSO keeping a
    # redundant, potentially-blocking event_complete() call of their own.
    # Reuses this file's existing _has_no_blocking_token() negative-check
    # machinery (already proven non-vacuous below, and again for these two
    # bodies specifically in test_pool_predicate_callers_have_no_blocking_token_witness).
    checks["try_reuse_pool_locked() body has no blocking token of its own"] = _has_no_blocking_token(
        TRY_REUSE_POOL_BODY_CODE
    )
    checks["pool_size_ready_locked() body has no blocking token of its own"] = _has_no_blocking_token(
        POOL_SIZE_READY_BODY_CODE
    )
    # llama.cpp-c6ah: the single choke point every pool-entry completion
    # check must route through, instead of a direct event_complete() query
    # on release_event -- see onednn_graph_scratch_pool_entry_release_complete()'s
    # own comment for why (event_complete()'s bare command_execution_status
    # query BLOCKS rather than polls on the profiling-enabled queue
    # release_event lives on). Three call sites: the shared usability
    # predicate both try_reuse_pool_locked() and pool_size_ready_locked()
    # apply (checked above to call the predicate; this asserts what the
    # PREDICATE ITSELF queries), the eviction sweep, and the pool-clear
    # reclaim path (checked again, alongside its own retain_handles_until_event()
    # requirement, below). Reuses _routes_through_release_complete() -- proven
    # non-vacuous in test_pool_entry_release_complete_is_the_single_choke_point.
    checks["entry_usable_locked() calls the release-complete choke point, not event_complete directly"] = (
        _routes_through_release_complete(ENTRY_USABLE_BODY_CODE)
    )
    checks["evict_pool_until_fits_locked() calls the release-complete choke point, not event_complete directly"] = (
        _routes_through_release_complete(EVICT_UNTIL_FITS_BODY_CODE)
    )
    checks["clear_pool_locked() calls the release-complete choke point, not event_complete directly"] = (
        _routes_through_release_complete(CLEAR_POOL_BODY_CODE)
    )
    # Bounded per-size depth (lead's constraint 3, ticket follow-up after the
    # pool redesign): without this, a workload that walks many distinct
    # sizes (a pp8192 run touches ~16 distinct ne11-derived shapes) could
    # grow the pool's footprint without limit even while each individual
    # size stays under the byte cap. Asserts the COMPARISON, not just that
    # the depth getter is called somewhere in the body -- a call present but
    # compared with the wrong operator (or not compared at all) would still
    # pass a presence-only check.
    checks["pool bounded per size, not just by total bytes"] = bool(
        re.search(r">=\s*onednn_graph_scratch_pool_depth_per_size\s*\(\s*\)", FREE_BODY_CODE)
    )
    # Teardown/context-reclaim/runtime-update must actually release pooled
    # buffers, not just stop tracking them -- checked structurally (the
    # reclaim entry point is called from all three sites the lead
    # specified: cache teardown, arena_reserve()'s context-reclaim branch,
    # and ggml_backend_sycl_set_runtime_context(), which does NOT call
    # arena_reserve() at all and so needs its own call site) rather than by
    # re-deriving what "correct" teardown means from scratch here.
    #
    # Scoped to shutdown_resources()'s own body, matching the context-reclaim
    # and runtime-context-update checks right below this one -- searching the
    # whole ~27000-line file cannot tell "called at teardown" apart from "the
    # literal string happens to appear somewhere else in the file" (a comment
    # quoting it, for instance).
    #
    # Two separate calls, not one onednn_graph_scratch_reclaim_pool("teardown"):
    # the summary log logs earlier in this function's body (right after the
    # high-water WARN, before either "shutting down" early return) so it
    # always prints on the common process-exit path, while the actual
    # release happens at the later, post-drain call site -- so both the log
    # call and the clear call must be present in this body, not the combined
    # helper.
    shutdown_resources_body_code = extract_function_body(CACHE_CPP_CODE, "bool unified_cache::shutdown_resources(")
    checks["pool reclaimed at cache teardown"] = (
        'onednn_graph_scratch_log_pool_summary_locked("teardown"' in shutdown_resources_body_code
        and "onednn_graph_scratch_clear_pool_locked();" in shutdown_resources_body_code
    )

    # BLOCKING: clear_pool_locked() must not destruct an entry whose release
    # event has not completed -- it must hand that one to
    # retain_handles_until_event() instead, since two of the four production
    # reclaim call sites (arena_reserve()'s context-reclaim branch,
    # ggml_backend_sycl_set_runtime_context()) do not drain the queue first.
    # Structural regression guard alongside the GPU test's own behavioral
    # coverage of the same property (test_pending_event_reclaim_does_not_destruct_in_flight,
    # whose own comment notes that coverage is itself a behavioural proxy --
    # its assertions also hold pre-fix). Tolerant of formatting: matches the
    # two calls' PRESENCE (not their exact argument text), which is what
    # actually survives clang-format re-wrapping or an unrelated rename of
    # the loop variable. llama.cpp-c6ah: the completion query this body makes
    # is RELEASE_COMPLETE_CALL, not a direct event_complete() -- see the
    # dedicated check above (a direct query here would BLOCK, same as it
    # used to for the other two pool-lookup call sites); this check adds the
    # retain_handles_until_event() half that one does not cover.
    checks["pool clear defers an incomplete-event entry instead of destructing it unconditionally"] = (
        RELEASE_COMPLETE_CALL in normalize_ws(CLEAR_POOL_BODY_CODE)
        and "retain_handles_until_event(" in normalize_ws(CLEAR_POOL_BODY_CODE)
    )
    # Scoped to arena_reserve()'s own body, not a same-file coincidence: the
    # reclaim call must be co-located with the KV/RUNTIME reclaim it is meant
    # to accompany, not merely present somewhere in a ~27000-line file.
    arena_reserve_body_code = extract_function_body(CACHE_CPP_CODE, "bool unified_cache::arena_reserve(")
    checks["pool reclaimed at context reclaim (same point KV/RUNTIME are reclaimed)"] = (
        normalize_ws("zone_reclaim(vram_zone_id::KV); zone_reclaim(vram_zone_id::RUNTIME);")
        in normalize_ws(arena_reserve_body_code)
        and 'onednn_graph_scratch_reclaim_pool("context reclaim")' in arena_reserve_body_code
    )
    # arena_reserve() is NOT called by the runtime-context-update path, so
    # the check above cannot cover it -- read ggml-sycl.cpp directly (plain
    # file I/O, not the codescout index, which is documented as blind inside
    # this specific file) and scope to
    # ggml_sycl_run_runtime_context_transaction()'s own body -- llama.cpp-tsfl
    # split ggml_backend_sycl_set_runtime_context() into a thin wrapper that
    # forwards into this shared body (also called by the new probe entry
    # point), and the reclaim call lives in the shared body, not the
    # wrapper. Comments are already stripped in GGML_SYCL_CPP_CODE (module
    # scope, same "strip first, extract second" rule as every other
    # extraction in this file -- extracting from raw GGML_SYCL_CPP first and
    # stripping the result after would let a stray brace inside a comment
    # desync extract_function_body()'s naive brace count before
    # strip_comments() ever ran).
    runtime_context_body = extract_function_body(
        GGML_SYCL_CPP_CODE, "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
    )
    checks["pool reclaimed at runtime context update"] = (
        'unified_cache_reclaim_onednn_graph_scratch_pool(ctx->device, "runtime context update")'
        in runtime_context_body
    )

    # TP fail-closed branch (gwno spec-review round 2, finding 4): under TP,
    # ctx.stream() returns the TP shared-context queue ahead of the cache's
    # own queue, so the in-order-queue argument above no longer names the
    # real consumer and immediate zone reclaim would race a TP consumer.
    checks["TP-active free path checked and deferred"] = (
        "ggml_sycl_get_tp_queue(" in FREE_BODY_CODE and "enqueue_deferred_zone_free(" in FREE_BODY_CODE
    )

    # reserve_onednn_scratch's growth guard must compare total_needed (the
    # primitive-API pair's own requirement, never including the Graph floor)
    # against the STORED getter, not the with-floor one -- comparing against
    # the with-floor reading lets the floor silently absorb the pair's growth
    # signal (llama.cpp-gwno spec-review round 3, finding 3).
    checks["growth guard reads the stored (bare) getter"] = (
        "unified_cache_get_planned_onednn_scratchpad_bytes_stored(dev_id) < total_needed" in CACHE_CPP_CODE
    )
    checks["growth guard does not read the with-floor getter"] = (
        "unified_cache_get_planned_onednn_scratchpad_bytes(dev_id) < total_needed" not in CACHE_CPP_CODE
    )

    # Env-tunable floor for the concurrent within-ubatch demand, additive on
    # top of the primitive-API pair (see unified_cache_get_planned_onednn_scratchpad_bytes).
    # Whitespace-insensitive: the call passes five struct-member arguments
    # (llama.cpp-o3a0 split the single n_head into n_head_ctx_max/
    # n_head_swa_max/n_swa), which clang-format is more likely to re-wrap
    # across lines than a shorter call ever was.
    checks["graph scratch zone floor is additive"] = normalize_ws(
        "bytes += onednn_graph_scratch_zone_floor_bytes_swa(shape.n_head_ctx_max, shape.n_head_swa_max, "
        "shape.n_swa, shape.n_ubatch, shape.n_ctx);"
    ) in normalize_ws(CACHE_CPP_CODE)
    # llama.cpp-o3a0: mutation witness for the caller actually passing the
    # new SWA-class fields through, not just the pre-existing ubatch/ctx pair
    # -- narrower than the full-call check above (which a clang-format
    # rewrap could still satisfy after a careless partial revert of one
    # argument, since normalize_ws would still find SOME five-argument call
    # matching the full string only if every token survives; this check
    # isolates the two fields the full check could not easily localize a
    # failure to). Scoped to WITH_FLOOR_GETTER_BODY_CODE, not the whole
    # file -- the
    # DIRECT-allocation-failure error log elsewhere in this file passes the
    # identical field names to a DIFFERENT call, so a file-wide substring
    # search here would still pass after a partial revert of THIS caller
    # specifically (the actual bug this check exists to catch).
    checks["graph scratch zone floor caller passes the swa fields"] = (
        "shape.n_head_swa_max" in WITH_FLOOR_GETTER_BODY_CODE and "shape.n_swa" in WITH_FLOOR_GETTER_BODY_CODE
    )
    checks["zone floor env var"] = "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB" in CACHE_CPP_CODE
    checks["allocator opt-out env var name"] = "GGML_SYCL_ONEDNN_CACHE_ALLOCATOR" in CACHE_CPP_CODE
    # llama.cpp-o3a0: the 3-arg overload must actually delegate into the
    # window-aware formula (not, say, keep a stale parallel copy of the
    # formula around) -- whitespace-insensitive for the same clang-format
    # re-wrap reason as the call above.
    checks["3-arg overload delegates to the swa formula"] = normalize_ws(
        "return onednn_graph_scratch_zone_floor_bytes_swa(n_head, 0, 0, n_ubatch, n_ctx);"
    ) in normalize_ws(FLOOR_BODY_CODE)
    # Token-anchored default value (llama.cpp-gwno spec-review round 2,
    # finding: this used to be untested, so a 512->64 regression or typo
    # would pass silently) -- not a bare "64" substring search, which would
    # match line numbers, byte counts, or anything else in the file.
    # llama.cpp-0oxf replaced the old `mb = 64` lambda-local variable with a
    # `kFloorMinBytes` constexpr once the floor became a formula
    # (max(64 MiB, 1.5 x n_head x n_ubatch x n_ctx x 4 B)) rather than a bare
    # env-overridable constant -- anchor on that constant's own definition,
    # not a value that could coincidentally appear elsewhere in the formula
    # (768 MiB, 1.5, sizeof(f32) as 4, etc. are all also just numbers).
    # llama.cpp-o3a0: reads FLOOR_SWA_BODY_CODE, not FLOOR_BODY_CODE -- the
    # constant moved into the window-aware formula when the 3-arg overload
    # became a thin wrapper (see the comment above FLOOR_BODY_CODE's
    # extraction).
    checks["default floor is 64 MiB"] = bool(
        re.search(r"kFloorMinBytes\s*=\s*64ull\s*\*\s*1024ull\s*\*\s*1024ull", FLOOR_SWA_BODY_CODE)
    )
    # llama.cpp-o3a0: mutation witness for the two-class window logic itself
    # -- if either the per-class min(n_ctx, n_swa + n_ubatch) window or the
    # max(ctx_term, swa_term) class selection is dropped (e.g. a careless
    # "simplification" back to a single term or to n_swa alone -- the
    # LATTER is a real regression this ticket shipped once, GPU-verified:
    # the SWA window is n_swa + n_ubatch, not n_swa alone), this fails even
    # though the Mistral (SWA-free) test rows in
    # test-sycl-onednn-graph-floor.cpp would not catch it, since
    # n_head_swa_max=0 makes the SWA term vanish there regardless of
    # whether min/max or the +n_ubatch term are present at all.
    checks["swa formula windows each class independently"] = bool(
        re.search(r"std::min\s*\(\s*static_cast<uint64_t>\(n_ctx\)\s*,\s*static_cast<uint64_t>\(n_swa\)\s*\+\s*"
                 r"static_cast<uint64_t>\(n_ubatch\)\s*\)", FLOOR_SWA_BODY_CODE)
    )
    checks["swa formula takes the max across classes, not the sum"] = bool(
        re.search(r"std::max\s*\(\s*ctx_term\s*,\s*swa_term\s*\)", FLOOR_SWA_BODY_CODE)
    )

    # High-water byte counter (llama.cpp-gwno perf follow-up): peak
    # concurrently-outstanding bytes, exposed and logged once at teardown so
    # a finished run's log answers "did the zone floor actually cover the
    # concurrent demand" without a special env var.
    checks["high-water getter declared"] = (
        "size_t onednn_graph_scratch_high_water_bytes() const { return onednn_graph_scratch_high_water_bytes_; }"
        in CACHE_HPP_CODE
    )
    checks["high-water logged once at teardown"] = "oneDNN Graph scratch high-water" in CACHE_CPP_CODE

    # No-cache fallback must fail loudly, not allocate/free cache-external
    # memory (llama.cpp-gwno spec-review round 2, findings 1-2): the malloc
    # callback's no-cache branch must return nullptr rather than calling
    # sycl::aligned_alloc_device(), and the free callback's no-cache branch
    # must not call sycl::free() on a pointer it cannot prove is a
    # standalone USM allocation.
    # strip_literals too: both branches log a WARN that NAMES the raw call they
    # refuse to make, and a negative check must read code, not messages.
    # Extracted from the already comment-stripped CACHE_CPP_CODE (finding 4;
    # see the module-level comment above the other extractions).
    malloc_fn_code = strip_literals(extract_function_body(CACHE_CPP_CODE, "void * onednn_graph_sycl_malloc("))
    free_fn_code = strip_literals(extract_function_body(CACHE_CPP_CODE, "void onednn_graph_sycl_free("))
    checks["malloc no-cache branch does not allocate cache-external memory"] = (
        "sycl::aligned_alloc_device" not in malloc_fn_code
    )
    checks["free no-cache branch does not call sycl::free"] = "sycl::free(" not in free_fn_code

    # llama.cpp-c6ah: the choke point itself must call event_complete() exactly once -- the
    # flag_slot == -1 fallback -- and never branch on the RED-arm test hook.
    # A branch there would still be querying a WATCHED event whenever
    # flag_slot was armed, so the RED arm can only be correct if this
    # function stays this simple and the hook instead lives in
    # onednn_graph_scratch_free()'s park site (checked below), which
    # decides whether flag_slot gets armed in the first place.
    checks["release_complete()'s only event_complete() call is the flag_slot == -1 fallback"] = (
        strip_literals(RELEASE_COMPLETE_BODY_CODE).count("event_complete(") == 1
    )
    checks["release_complete() does not branch on the RED-arm test hook"] = (
        "g_onednn_graph_scratch_test_force_blocking_pool_check" not in RELEASE_COMPLETE_BODY_CODE
    )
    checks["free() park site consults the RED-arm test hook before arming the flag"] = (
        "g_onednn_graph_scratch_test_force_blocking_pool_check" in FREE_BODY_CODE
    )
    # llama.cpp-c6ah: the pool's completion flag is armed by a
    # DEVICE marker kernel, never a host_task -- a host_task whose
    # depends_on() names an event from another queue (which the release
    # event always is here) was measured, both cards, to block the
    # SUBMITTING thread until that event completes, i.e. it would stall
    # this very function for the parked kernel's whole duration on every
    # single park. strip_literals first: the WARN messages in this body
    # name "host_task" in prose (describing what NOT to do), and a negative
    # check must read code, not messages.
    checks["free() body submits no host_task (the finding-31 marker-kernel design, not a watcher)"] = (
        "host_task(" not in strip_literals(FREE_BODY_CODE)
    )

    # llama.cpp-c6ah: CACHE_BACKING's two-site allowlist -- see
    # the module-level comment above ADOPT_MINT_HELPER. A third
    # cache_backing=true call site (a new bootstrap mint added without
    # review), a missing cohort tag, or a cohort tag that does not match
    # either allowlisted string must all fail this check.
    checks["cache_backing=true mints are exactly the two-site allowlist"] = sorted(
        _adopt_cache_backing_cohorts(CACHE_CPP_CODE)
    ) == sorted(ADOPT_CACHE_BACKING_ALLOWLIST)

    # llama.cpp-c6ah: the withdrawn "bare query lies" reading must
    # stay history-framed everywhere it is still mentioned -- checked against
    # the RAW (comment-bearing) source, since this is a check on comment
    # prose, not on code behavior (see the module docstring's stated
    # exception for the in-order-queue check, which this joins).
    checks["every withdrawn 'query lies' mention is history-framed"] = _withdrawn_lies_phrasing_is_history_framed(
        CACHE_CPP
    ) and _withdrawn_lies_phrasing_is_history_framed(CACHE_HPP)

    # llama.cpp-c6ah: the design doc must still state the two
    # MEASURED facts that replaced the withdrawn reading, not just avoid the
    # withdrawn wording.
    checks["design doc states the two measured completion-check facts"] = _design_doc_states_measured_facts()

    # llama.cpp-me60 (F1): the completion-flag slab is a bootstrap
    # CACHE_BACKING control (see ADOPT_CACHE_BACKING_ALLOWLIST above), but
    # shutdown_unified_cache()'s LATER registry sweep is class-blind -- it
    # never consults allocation_control_class, only pinned-pool
    # containment -- so the slab must be released explicitly here, on the
    # normal teardown path, not left to member destruction after that sweep
    # has already run. shutdown_resources_body_code was already extracted
    # above, for the teardown pool-clear check above.
    checks["oneDNN Graph-scratch flag slab owner is released inside shutdown_resources(), after the drain call"] = (
        _flag_slab_owner_released_after_drain(shutdown_resources_body_code)
    )

    # llama.cpp-me60: a SEPARATE, earlier defect than F1 (predates this
    # ticket -- traced to llama.cpp-0oxf) -- a parked DIRECT Graph-scratch
    # buffer's own EXTERNAL_EXACT allocation control must already be
    # reclaimed by the time shutdown_unified_cache()'s pre-teardown census
    # runs, or that census refuses shutdown on every process that ever
    # parked one.
    shutdown_unified_cache_body_code = extract_function_body(CACHE_CPP_CODE, "bool shutdown_unified_cache(")
    checks["oneDNN Graph-scratch pool is reclaimed before the pre-teardown census in shutdown_unified_cache()"] = (
        _pool_reclaimed_before_preteardown_census(shutdown_unified_cache_body_code)
    )

    # llama.cpp-3lgu (F7): nothing previously gated the guard that skips the
    # whole pre-census drain+reclaim pass once SYCL is already shutting
    # down -- add source coverage for it directly.
    checks["pre-census drain/reclaim loop is skipped once SYCL is already shutting down"] = (
        _preteardown_pool_loop_skipped_once_shutting_down(shutdown_unified_cache_body_code)
    )

    # llama.cpp-3lgu (F2): the pre-census pass had no queue-context validity
    # probe of its own -- a context torn down while g_sycl_shutting_down is
    # still false used to reach drain_all_queues_noexcept() (swallows the
    # throw) and then clear_pool_locked(), whose mem_handle destructors read
    # the same false flag and attempt real frees against an already-invalid
    # context.
    checks["pre-census drain/reclaim loop probes queue validity before its drain call"] = (
        _preteardown_loop_probes_queue_validity_before_drain(shutdown_unified_cache_body_code)
    )

    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, "onednn graph allocator source contract failed: " + ", ".join(failed)


def test_no_blocking_wait_check_has_a_mutation_witness() -> None:
    """Mutation witness for the structural check above (llama.cpp-gwno
    spec-review round 2, finding 5): proves _has_no_blocking_token() would
    actually catch a realistic regression -- event_complete() reintroduced
    as an inlined direct .wait() call -- rather than only ever passing on
    the current, correct source."""
    target = "zone_free(vram_zone_id::ONEDNN, ptr);"
    assert target in FREE_BODY_CODE, "mutation target string not found -- update this witness"
    mutated = FREE_BODY_CODE.replace(
        target,
        "sycl::event e = *event; e.wait(); " + target,
        1,
    )
    assert mutated != FREE_BODY_CODE
    assert _has_no_blocking_token(FREE_BODY_CODE), "the real, unmutated free path body should have no blocking token"
    assert not _has_no_blocking_token(mutated), (
        "mutation witness is broken: the injected .wait() was not detected by _has_no_blocking_token()"
    )


def test_pool_predicate_callers_have_no_blocking_token_witness() -> None:
    """Mutation witness for the two checks added in llama.cpp-0oxf round-6
    finding H3 -- proves _has_no_blocking_token() would actually catch a
    reintroduced, redundant event_complete() call in either
    onednn_graph_scratch_try_reuse_pool_locked() or
    onednn_graph_scratch_pool_size_ready_locked(), the specific regression
    those checks exist to catch (both used to call event_complete() on their
    own ahead of the shared predicate -- llama.cpp-c6ah), rather than only
    ever passing on the current, correct source."""
    # Whitespace-insensitive (llama.cpp-c6ah): this declaration sits next to
    # a comment block whose length determines whether clang-format pulls it
    # into a multi-line alignment group with a sibling declaration further
    # down, which changes the exact spacing between `mem_handle` and
    # `owner` -- normalize before matching so an unrelated comment edit
    # nearby cannot silently break this witness.
    reuse_target = "mem_handle owner = std::move(bucket[i].owner);"
    reuse_body_normalized = normalize_ws(TRY_REUSE_POOL_BODY_CODE)
    assert reuse_target in reuse_body_normalized, "mutation target string not found -- update this witness"
    reuse_mutated = reuse_body_normalized.replace(
        reuse_target,
        "event_complete(bucket[i].release_event); " + reuse_target,
        1,
    )
    assert reuse_mutated != reuse_body_normalized
    assert _has_no_blocking_token(TRY_REUSE_POOL_BODY_CODE), (
        "the real, unmutated try_reuse_pool_locked() body should have no blocking token"
    )
    assert not _has_no_blocking_token(reuse_mutated), (
        "mutation witness is broken: the injected event_complete() call was not detected by "
        "_has_no_blocking_token()"
    )

    ready_target = "return true;"
    ready_body_normalized = normalize_ws(POOL_SIZE_READY_BODY_CODE)
    assert ready_target in ready_body_normalized, "mutation target string not found -- update this witness"
    ready_mutated = ready_body_normalized.replace(
        ready_target,
        "event_complete(entry.release_event); " + ready_target,
        1,
    )
    assert ready_mutated != ready_body_normalized
    assert _has_no_blocking_token(POOL_SIZE_READY_BODY_CODE), (
        "the real, unmutated pool_size_ready_locked() body should have no blocking token"
    )
    assert not _has_no_blocking_token(ready_mutated), (
        "mutation witness is broken: the injected event_complete() call was not detected by "
        "_has_no_blocking_token()"
    )


def test_pool_entry_release_complete_is_the_single_choke_point() -> None:
    """Mutation witness for llama.cpp-c6ah: proves the three
    "calls RELEASE_COMPLETE_CALL, never a bare event_complete()" checks in
    the main contract test above would actually catch a reintroduced direct
    event_complete() query on a pool entry's release_event -- the specific
    regression this ticket fixes (querying that event directly BLOCKS rather
    than polls on the profiling-enabled queue it lives on, turning "skip an
    in-flight entry" back into "wait for it"). Without this witness, a
    check that always reports the CURRENT (correct) source as passing could
    just as easily never fire on the mutation it claims to guard against."""

    def assert_catches_direct_event_complete(body: str, label: str, target: str) -> None:
        assert target in body, f"{label}: mutation target string not found -- update this witness"
        mutated = body.replace(target, "event_complete(entry.release_event); " + target, 1)
        assert mutated != body
        assert _routes_through_release_complete(body), (
            f"{label}: the real, unmutated body should call {RELEASE_COMPLETE_CALL!r} and nothing named "
            "event_complete("
        )
        assert not _routes_through_release_complete(mutated), (
            f"{label}: mutation witness is broken -- the injected direct event_complete() call was not "
            "detected"
        )

    assert_catches_direct_event_complete(
        ENTRY_USABLE_BODY_CODE,
        "onednn_graph_scratch_entry_usable_locked()",
        "return onednn_graph_scratch_entry_fit::EVENT_PENDING;",
    )
    assert_catches_direct_event_complete(
        EVICT_UNTIL_FITS_BODY_CODE,
        "onednn_graph_scratch_evict_pool_until_fits_locked()",
        "for (auto bucket_it = onednn_graph_scratch_reuse_pool_.begin();",
    )
    assert_catches_direct_event_complete(
        CLEAR_POOL_BODY_CODE,
        "onednn_graph_scratch_clear_pool_locked()",
        "for (auto & bucket_kv : onednn_graph_scratch_reuse_pool_) {",
    )


def test_release_complete_has_no_red_arm_branch_of_its_own_mutation_witness() -> None:
    """Mutation witness for llama.cpp-c6ah (anchor tracks the
    flag_slot/flag_generation redesign, same property this witness always
    checked): proves the two checks added for it in the main contract test
    above would actually catch the SPECIFIC regression they exist to
    prevent -- a RED-arm test hook branch reintroduced inside
    onednn_graph_scratch_pool_entry_release_complete() itself, querying
    release_event directly whenever the hook is forced, regardless of
    flag_slot. That design was tried and found incorrect (it would still
    query a WATCHED event, a separate hazard documented in the function's
    own comment), which is why the fix moved the hook to
    onednn_graph_scratch_free()'s park site instead; this witness proves a
    regression back to the check-time branch would be caught, not merely
    that the current source happens to pass."""
    anchor = "if (entry.flag_slot >= 0) {"
    assert anchor in RELEASE_COMPLETE_BODY_CODE, "mutation target string not found -- update this witness"
    reintroduced_hook_branch = (
        "if (onednn_graph_scratch_test_hooks_enabled() && "
        "g_onednn_graph_scratch_test_force_blocking_pool_check.load(std::memory_order_acquire)) { "
        "return event_complete(entry.release_event); } "
    )
    mutated = RELEASE_COMPLETE_BODY_CODE.replace(anchor, reintroduced_hook_branch + anchor, 1)
    assert mutated != RELEASE_COMPLETE_BODY_CODE

    real_count = strip_literals(RELEASE_COMPLETE_BODY_CODE).count("event_complete(")
    mutated_count = strip_literals(mutated).count("event_complete(")
    assert real_count == 1, (
        "the real, unmutated release_complete() body should call event_complete() exactly once "
        f"(the flag_slot == -1 fallback); found {real_count}"
    )
    assert mutated_count != 1, (
        "mutation witness is broken: the reintroduced hook branch's extra event_complete() call was not "
        "detected by the call-count check"
    )

    assert "g_onednn_graph_scratch_test_force_blocking_pool_check" not in RELEASE_COMPLETE_BODY_CODE, (
        "the real, unmutated release_complete() body should not reference the RED-arm hook at all"
    )
    assert "g_onednn_graph_scratch_test_force_blocking_pool_check" in mutated, (
        "mutation witness is broken: the reintroduced hook reference was not present in the mutated text"
    )


def test_free_body_has_no_host_task_mutation_witness() -> None:
    """Mutation witness for llama.cpp-c6ah: proves the "free()
    body submits no host_task" check in the main contract test above would
    actually catch the SPECIFIC regression it exists to prevent -- a
    host_task reintroduced into onednn_graph_scratch_free()'s completion-
    flag arm, the exact design this fork replaced after measuring (both
    cards, 2026-09-09) that submitting a host_task with a cross-queue
    dependency blocks the SUBMITTING thread -- i.e. this very function,
    called from oneDNN's free callback -- until that dependency
    completes."""
    anchor = "depends_on(*event);"
    stripped_free_body = strip_literals(FREE_BODY_CODE)
    assert anchor in stripped_free_body, "mutation target string not found -- update this witness"
    mutated = stripped_free_body.replace(
        anchor,
        anchor + " h.host_task([]() {});",
        1,
    )
    assert mutated != stripped_free_body
    assert "host_task(" not in stripped_free_body, "the real, unmutated free() body should submit no host_task"
    assert "host_task(" in mutated, (
        "mutation witness is broken: the reintroduced host_task() call was not present in the mutated text"
    )


def test_cache_backing_allowlist_has_a_mutation_witness() -> None:
    """Mutation witness for llama.cpp-c6ah: proves the
    "cache_backing=true mints are exactly the two-site allowlist" check in
    the main contract test above would actually catch a reintroduced THIRD
    bootstrap mint -- a new unified_cache_adopt_raw_host_allocation() call
    site passing cache_backing=true with a cohort tag outside the two
    reviewed ones, added without the review this gate exists to force."""
    real_cohorts = _adopt_cache_backing_cohorts(CACHE_CPP_CODE)
    assert sorted(real_cohorts) == sorted(ADOPT_CACHE_BACKING_ALLOWLIST), (
        "the real, unmutated source should carry exactly the two allowlisted cache_backing=true cohorts -- "
        f"found {real_cohorts}"
    )

    third_call_probe = (
        '\nstatic alloc_handle _c6ah_finding41_mutation_probe(sycl::queue & q) {\n'
        '    return %s(nullptr, 0, q, alloc_role::OTHER, runtime_category::OTHER, '
        '"unified_cache:not_allowlisted", true);\n'
        '}\n'
    ) % ADOPT_MINT_HELPER
    mutated = CACHE_CPP_CODE + third_call_probe
    mutated_cohorts = _adopt_cache_backing_cohorts(mutated)
    assert sorted(mutated_cohorts) != sorted(ADOPT_CACHE_BACKING_ALLOWLIST), (
        "mutation witness is broken: the injected third cache_backing=true call site was not detected"
    )


def test_withdrawn_lies_phrasing_check_has_a_mutation_witness() -> None:
    """Mutation witness for llama.cpp-c6ah: proves the "every
    withdrawn 'query lies' mention is history-framed" check above would
    catch the SPECIFIC regression it exists to prevent -- the withdrawn
    reading restated as if it were still the current explanation, with no
    "earlier"/"misdiagnos" framing anywhere nearby."""
    assert _withdrawn_lies_phrasing_is_history_framed(CACHE_HPP), (
        "the real, unmutated unified-cache.hpp text should already pass this check"
    )
    assert _withdrawn_lies_phrasing_is_history_framed(CACHE_CPP), (
        "the real, unmutated unified-cache.cpp text should already pass this check"
    )
    unframed_reintroduction = (
        "\n// The bare event query lies about completion once a watcher is attached.\n"
    )
    mutated = CACHE_HPP + unframed_reintroduction
    assert not _withdrawn_lies_phrasing_is_history_framed(mutated), (
        "mutation witness is broken: the reintroduced, unframed 'lies' mention was not detected"
    )
    # A positive control the other way: the same sentence WITH framing must
    # still pass, proving this isn't just "reject any new occurrence of the
    # word".
    framed_reintroduction = (
        "\n// An earlier draft of this comment claimed the bare event query lies about completion once a "
        "watcher is attached; that was a misdiagnosis.\n"
    )
    assert _withdrawn_lies_phrasing_is_history_framed(CACHE_HPP + framed_reintroduction), (
        "the check should not reject a NEW mention that is properly history-framed"
    )
    # llama.cpp-c6ah: the specific gap a fixed-size character window left --
    # an UNFRAMED sentence sitting in the same paragraph as, but not the
    # same sentence as, a genuinely framed one used to inherit that framing
    # merely by being nearby. Two sentences, same paragraph (no blank
    # comment line between them): the first is framed and would pass on its
    # own; the second restates the withdrawn claim with no framing of its
    # own and must fail even though "misdiagnosis" appears a few words
    # earlier in the same paragraph.
    adjacent_unframed_reintroduction = (
        "\n// An earlier draft mishandled this timing. That was a misdiagnosis of a different effect.\n"
        "// The bare event query lies about completion once a watcher is attached.\n"
    )
    assert not _withdrawn_lies_phrasing_is_history_framed(CACHE_HPP + adjacent_unframed_reintroduction), (
        "mutation witness is broken: an unframed 'lies' sentence merely ADJACENT (same paragraph, different "
        "sentence) to an unrelated framed one was not detected -- framing must be in the same sentence, not "
        "merely nearby"
    )
    # llama.cpp-c6ah: the OTHER half of this ticket's own regression -- the
    # "lying" spelling (used in this codebase's own comments: "there was no
    # lying query") must be caught by the same mechanism, unframed.
    unframed_lying_reintroduction = "\n// There is no lying query here, this function tells the truth.\n"
    assert not _withdrawn_lies_phrasing_is_history_framed(CACHE_HPP + unframed_lying_reintroduction), (
        "mutation witness is broken: an unframed 'lying' mention was not detected -- only the 'lies' "
        "spelling was being matched"
    )
    # llama.cpp-c6ah: an ORDINARY English "lies"/"lying" sentence with
    # nothing to do with a query must never be flagged, unframed or not --
    # "queue" deliberately does NOT contain "quer" as a substring, so this
    # is a genuine negative case, not an accidental match.
    unrelated_lies_sentence = "\n// The slab lies before the queue in the class body.\n"
    assert _withdrawn_lies_phrasing_is_history_framed(CACHE_HPP + unrelated_lies_sentence), (
        "the check should not flag an ordinary English use of 'lies' that has nothing to do with a query"
    )


def test_design_doc_measured_facts_check_has_a_mutation_witness() -> None:
    """Mutation witness for llama.cpp-c6ah: proves the "design
    doc states the two measured completion-check facts" check above would
    catch either fact being edited or removed from
    docs/backend/sycl-memory-design.md -- by calling
    _design_doc_states_measured_facts() itself against the mutated text
    (not a separately re-derived local check that could silently drift
    from what the real check actually tests)."""
    assert _design_doc_states_measured_facts(), "the real, unmutated design doc should already pass this check"
    mutated_missing_first_fact = normalize_ws(MEMORY_DESIGN_MD).replace(
        normalize_ws(DESIGN_DOC_MEASURED_FACT_QUERY_BLOCKS), "", 1
    )
    assert normalize_ws(DESIGN_DOC_MEASURED_FACT_QUERY_BLOCKS) not in mutated_missing_first_fact
    assert normalize_ws(DESIGN_DOC_MEASURED_FACT_HOST_TASK_BLOCKS_SUBMITTER) in mutated_missing_first_fact
    assert not _design_doc_states_measured_facts(mutated_missing_first_fact), (
        "mutation witness is broken: removing the first measured fact was not detected by the real check "
        "function"
    )
    mutated_missing_second_fact = normalize_ws(MEMORY_DESIGN_MD).replace(
        normalize_ws(DESIGN_DOC_MEASURED_FACT_HOST_TASK_BLOCKS_SUBMITTER), "", 1
    )
    assert normalize_ws(DESIGN_DOC_MEASURED_FACT_HOST_TASK_BLOCKS_SUBMITTER) not in mutated_missing_second_fact
    assert normalize_ws(DESIGN_DOC_MEASURED_FACT_QUERY_BLOCKS) in mutated_missing_second_fact
    assert not _design_doc_states_measured_facts(mutated_missing_second_fact), (
        "mutation witness is broken: removing the second measured fact was not detected by the real check "
        "function"
    )


def test_flag_slab_owner_release_after_drain_check_has_a_mutation_witness() -> None:
    """llama.cpp-me60 (F1): proves _flag_slab_owner_released_after_drain()
    is a real positional check, not a same-body coincidence -- the same
    reset statement also appears TWICE earlier in shutdown_resources()'s
    body (its two early-return branches, both before the drain call), so a
    naive "does this string appear anywhere in the body" check would pass
    even with the normal-path release entirely absent. Removes exactly the
    POST-drain occurrence (the last one in the body -- both early-return
    occurrences sit before the drain call, so this is unambiguous) and
    confirms the real check function then reports it missing."""
    shutdown_resources_body_code = extract_function_body(CACHE_CPP_CODE, "bool unified_cache::shutdown_resources(")
    assert _flag_slab_owner_released_after_drain(shutdown_resources_body_code), (
        "the real, unmutated shutdown_resources() body should already pass this check"
    )
    normalized = normalize_ws(shutdown_resources_body_code)
    drain_idx = normalized.find(DRAIN_CALL)
    assert drain_idx != -1, "the real, unmutated body should call drain_all_queues_noexcept()"
    post_drain_reset_idx = normalized.rfind(FLAG_SLAB_OWNER_RESET_STMT)
    assert post_drain_reset_idx != -1 and post_drain_reset_idx > drain_idx, (
        "the real, unmutated body's LAST flag-slab-owner reset should be the post-drain one this check targets"
    )
    mutated = (
        normalized[:post_drain_reset_idx]
        + normalized[post_drain_reset_idx + len(FLAG_SLAB_OWNER_RESET_STMT) :]
    )
    assert mutated != normalized
    assert not _flag_slab_owner_released_after_drain(mutated), (
        "mutation witness is broken: removing the post-drain flag-slab-owner reset was not detected -- the "
        "check may be matching one of the two earlier, pre-drain occurrences instead"
    )


def test_pool_reclaimed_before_census_check_has_a_mutation_witness() -> None:
    """llama.cpp-me60: proves _pool_reclaimed_before_preteardown_census()
    checks the FIRST occurrence of each anchor (str.find(), not merely "the
    reclaim call occurs somewhere before A census call") -- injects an
    EARLIER copy of the census anchor ahead of the real reclaim call
    (approximating the historical ordering bug, where the census ran before
    any reclaim pass existed at all) and confirms the real check function
    then reports the ordering as wrong, even though the real (later) pair
    of calls is still present and still correctly ordered."""
    shutdown_unified_cache_body_code = extract_function_body(CACHE_CPP_CODE, "bool shutdown_unified_cache(")
    assert _pool_reclaimed_before_preteardown_census(shutdown_unified_cache_body_code), (
        "the real, unmutated shutdown_unified_cache() body should already pass this check"
    )
    normalized = normalize_ws(shutdown_unified_cache_body_code)
    reclaim_idx = normalized.find(POOL_RECLAIM_CALL)
    census_idx = normalized.find(PRETEARDOWN_CENSUS_CALL)
    assert reclaim_idx != -1 and census_idx != -1 and reclaim_idx < census_idx
    mutated = normalized[:reclaim_idx] + PRETEARDOWN_CENSUS_CALL + " " + normalized[reclaim_idx:]
    assert mutated != normalized
    assert not _pool_reclaimed_before_preteardown_census(mutated), (
        "mutation witness is broken: an earlier census call injected ahead of the reclaim call was not detected"
    )


def test_preteardown_pool_loop_shutdown_guard_check_has_a_mutation_witness() -> None:
    """llama.cpp-3lgu (F7): proves _preteardown_pool_loop_skipped_once_shutting_down()
    is a real structural check -- the guard text occurs exactly once in
    shutdown_unified_cache() and the drain+reclaim loop text is the FIRST of
    its two occurrences, so a naive "does the loop text occur anywhere in
    the body" check could not tell a guarded loop from an unguarded one
    sitting right after an unrelated guard. Removes the guard's own
    `if (!ggml_sycl_is_shutting_down())` text (leaving its block body and
    braces in place, exactly as an accidental de-guarding would) and
    confirms the real check function then reports the loop as unguarded."""
    shutdown_unified_cache_body_code = extract_function_body(CACHE_CPP_CODE, "bool shutdown_unified_cache(")
    assert _preteardown_pool_loop_skipped_once_shutting_down(shutdown_unified_cache_body_code), (
        "the real, unmutated shutdown_unified_cache() body should already pass this check"
    )

    # A first mutant that DE-NESTS without deleting anything: closes the
    # guard block immediately (empty `{ }` body), leaving the loop text
    # fully present in the function body but no longer inside the guard's
    # braces. This is the property the deletion mutant below cannot prove --
    # that check alone only shows the guard text must be PRESENT somewhere,
    # not that the loop must be NESTED inside it.
    normalized_body = normalize_ws(shutdown_unified_cache_body_code)
    denested = normalized_body.replace(
        PRETEARDOWN_SHUTTING_DOWN_GUARD + " {", PRETEARDOWN_SHUTTING_DOWN_GUARD + " { }", 1
    )
    assert denested != normalized_body
    assert CACHES_LOOP_STMT in denested, "sanity: this mutant must leave the loop text in place"
    assert not _preteardown_pool_loop_skipped_once_shutting_down(denested), (
        "mutation witness is broken: closing the guard block early, leaving the loop outside it, was not detected"
    )

    assert PRETEARDOWN_SHUTTING_DOWN_GUARD in shutdown_unified_cache_body_code
    mutated = shutdown_unified_cache_body_code.replace(PRETEARDOWN_SHUTTING_DOWN_GUARD, "", 1)
    assert mutated != shutdown_unified_cache_body_code
    assert not _preteardown_pool_loop_skipped_once_shutting_down(mutated), (
        "mutation witness is broken: removing the shutting-down guard was not detected -- the check may be "
        "matching the drain+reclaim loop's text unconditionally instead of requiring it inside the guard block"
    )


def test_preteardown_loop_queue_probe_check_has_a_mutation_witness() -> None:
    """llama.cpp-3lgu (F2): proves _preteardown_loop_probes_queue_validity_before_drain()
    is a real positional check on the probe's own get_context() call, not a
    same-body coincidence -- removes exactly that call from the real, fixed
    loop body (leaving its surrounding try {} catch (...) {} skeleton and
    the drain call untouched) and confirms the real check function then
    reports the probe as missing, even though the try/catch structure and
    the drain call are both still present and still correctly ordered.

    A second mutant proves the store is checked too, not just the try/catch
    shape: removes ONLY the g_sycl_shutting_down.store(true, ...) statement
    from the catch body (leaving get_context(), the try {}/catch (...) {}
    skeleton, and the drain call all untouched and still correctly ordered)
    and confirms the real check function reports it missing -- a catch
    clause that merely swallows the exception without recording the flag
    would otherwise pass this check even though this cache's own later
    shutdown_resources() call would then see a still-false flag.

    A third mutant proves the catch body's own `continue;` is checked too,
    not just the store: removes ONLY the `continue;` that follows the store
    (leaving get_context(), the store statement, the try {}/catch (...) {}
    skeleton, and the drain call all untouched and still correctly ordered)
    and confirms the real check function reports it missing. This is
    distinct from the loop's earlier `if (!item.second) { continue; }`,
    which sits before the try and must stay untouched by this mutant --
    without the catch body's own continue, the probe would still record the
    flag but the invalid cache would then fall through to
    drain_all_queues_noexcept() and the reclaim call instead of being
    skipped, which is exactly what the continue exists to prevent -- the
    store alone protects only the caches probed after this one.

    A fourth mutant proves the continue anchor is scoped to the catch's own
    braces, not merely positional between the store and the drain call: it
    starts from the third mutant's body (the catch's own continue already
    removed) and inserts an unrelated `if (x) { continue; }` immediately
    before drain_all_queues_noexcept() -- a continue that has nothing to do
    with the probe, sitting outside the catch entirely. A check that only
    asks "is there a continue somewhere between the store and the drain"
    would be fooled by this: it reports the probe as present even though
    the catch clause itself no longer contains a continue and this cache's
    own drain+reclaim would no longer be skipped.

    A fifth mutant proves the check's OTHER half -- the positional search
    from the store -- is still load-bearing even with containment in
    place: it reorders the two statements inside the real catch body to
    `{ continue; g_sycl_shutting_down.store(...); }`, leaving get_context(),
    the try {}/catch (...) {} skeleton, and the drain call all untouched
    and still correctly ordered. The continue is still inside the catch's
    own braces, so containment alone would accept it; but the continue no
    longer follows the store, so the reordered catch's store is
    unreachable -- the continue exits the loop iteration before it can
    ever run -- and this cache's own probe no longer actually sets the
    flag before skipping. The real check function must still report this
    mutant as missing."""
    shutdown_unified_cache_body_code = extract_function_body(CACHE_CPP_CODE, "bool shutdown_unified_cache(")
    assert _preteardown_loop_probes_queue_validity_before_drain(shutdown_unified_cache_body_code), (
        "the real, fixed shutdown_unified_cache() body should already pass this check (llama.cpp-3lgu F2)"
    )
    assert QUEUE_CONTEXT_PROBE_CALL in shutdown_unified_cache_body_code
    mutated = shutdown_unified_cache_body_code.replace(QUEUE_CONTEXT_PROBE_CALL, "", 1)
    assert mutated != shutdown_unified_cache_body_code
    assert not _preteardown_loop_probes_queue_validity_before_drain(mutated), (
        "mutation witness is broken: removing the queue-context probe call was not detected"
    )

    store_stmt = "g_sycl_shutting_down.store(true, std::memory_order_release);"
    assert store_stmt in shutdown_unified_cache_body_code, "sanity: the real store statement text must be present"
    mutated_store = shutdown_unified_cache_body_code.replace(store_stmt, "", 1)
    assert mutated_store != shutdown_unified_cache_body_code
    assert QUEUE_CONTEXT_PROBE_CALL in mutated_store, "sanity: this mutant must leave get_context() untouched"
    assert DRAIN_CALL in mutated_store, "sanity: this mutant must leave the drain call untouched"
    assert not _preteardown_loop_probes_queue_validity_before_drain(mutated_store), (
        "mutation witness is broken: removing only the g_sycl_shutting_down.store(true, ...) statement from the "
        "catch body was not detected -- the check may be validating the try/catch shape without checking what "
        "the catch body actually does"
    )

    store_idx_raw = shutdown_unified_cache_body_code.find(store_stmt)
    after_store_text = shutdown_unified_cache_body_code[store_idx_raw:]
    assert CATCH_CONTINUE_STMT in after_store_text, "sanity: the catch body's own continue; must follow the store"
    mutated_after_store = after_store_text.replace(CATCH_CONTINUE_STMT, "", 1)
    assert mutated_after_store != after_store_text
    mutated_continue = shutdown_unified_cache_body_code[:store_idx_raw] + mutated_after_store
    assert QUEUE_CONTEXT_PROBE_CALL in mutated_continue, "sanity: this mutant must leave get_context() untouched"
    assert store_stmt in mutated_continue, "sanity: this mutant must leave the store statement untouched"
    assert DRAIN_CALL in mutated_continue, "sanity: this mutant must leave the drain call untouched"
    assert not _preteardown_loop_probes_queue_validity_before_drain(mutated_continue), (
        "mutation witness is broken: removing only the catch body's own continue; (the one after the store, not "
        "the loop's earlier `if (!item.second) { continue; }`, which this mutant must leave untouched) was not "
        "detected -- without it, the probe would still record the flag but the invalid cache would then fall "
        "through to drain_all_queues_noexcept() and the reclaim call instead of being skipped"
    )

    drain_idx_raw = mutated_continue.find(DRAIN_CALL)
    assert drain_idx_raw != -1, "sanity: the third mutant's body must still contain the drain call"
    fail_open_mutant = (
        mutated_continue[:drain_idx_raw] + "if (x) { continue; } " + mutated_continue[drain_idx_raw:]
    )
    assert QUEUE_CONTEXT_PROBE_CALL in fail_open_mutant, "sanity: this mutant must leave get_context() untouched"
    assert store_stmt in fail_open_mutant, "sanity: this mutant must leave the store statement untouched"
    assert DRAIN_CALL in fail_open_mutant, "sanity: this mutant must leave the drain call untouched"
    assert fail_open_mutant.count(CATCH_ALL_STMT) == 1, "sanity: the catch-scope assert below assumes one catch in the body"
    assert CATCH_CONTINUE_STMT not in extract_function_body(fail_open_mutant, CATCH_ALL_STMT), (
        "sanity: this mutant's own catch braces must no longer contain a continue; -- the inserted "
        "`if (x) { continue; }` must land after the catch closes, immediately before the drain call"
    )
    assert not _preteardown_loop_probes_queue_validity_before_drain(fail_open_mutant), (
        "mutation witness is broken: the continue anchor is purely positional -- an unrelated "
        "`if (x) { continue; }` inserted immediately before drain_all_queues_noexcept() (after removing the "
        "catch body's own continue;) satisfies a check that only looks for *some* continue; between the store "
        "and the drain, even though the catch clause itself no longer contains one"
    )

    continue_idx_after_store = after_store_text.find(CATCH_CONTINUE_STMT)
    reordered_after_store = (
        CATCH_CONTINUE_STMT
        + after_store_text[len(store_stmt):continue_idx_after_store]
        + store_stmt
        + after_store_text[continue_idx_after_store + len(CATCH_CONTINUE_STMT):]
    )
    reordered_mutant = shutdown_unified_cache_body_code[:store_idx_raw] + reordered_after_store
    assert QUEUE_CONTEXT_PROBE_CALL in reordered_mutant, "sanity: this mutant must leave get_context() untouched"
    assert store_stmt in reordered_mutant, "sanity: this mutant must leave the store statement untouched"
    assert DRAIN_CALL in reordered_mutant, "sanity: this mutant must leave the drain call untouched"
    assert reordered_mutant.count(CATCH_ALL_STMT) == 1, "sanity: the catch-scope assert below assumes one catch in the body"
    assert CATCH_CONTINUE_STMT in extract_function_body(reordered_mutant, CATCH_ALL_STMT), (
        "sanity: the reordered catch body must still contain the continue; -- containment alone would accept "
        "this mutant, which is exactly what this mutant is meant to prove is not enough on its own"
    )
    assert not _preteardown_loop_probes_queue_validity_before_drain(reordered_mutant), (
        "mutation witness is broken: containment alone would accept this reordered catch body -- "
        "`{ continue; g_sycl_shutting_down.store(...); }`, whose store is now unreachable after the continue -- "
        "because the continue is still inside the catch's own braces; the positional search from the store is "
        "what requires the continue to FOLLOW the store, and it is what actually catches this mutant"
    )
