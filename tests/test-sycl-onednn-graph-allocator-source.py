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


COMMON_HPP_CODE = strip_comments(COMMON_HPP)
CACHE_HPP_CODE = strip_comments(CACHE_HPP)
CACHE_CPP_CODE = strip_comments(CACHE_CPP)

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
FLOOR_BODY_CODE = extract_function_body(CACHE_CPP_CODE, "static size_t onednn_graph_scratch_zone_floor_bytes()")
MAKE_ENGINE_BODY_CODE = extract_function_body(COMMON_HPP_CODE, "dnnl::engine make_engine(sycl::queue * q) {")


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
    checks["malloc routes through the ONEDNN zone"] = "zone_alloc(vram_zone_id::ONEDNN, size, align)" in ALLOC_BODY_CODE
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

    checks["direct fallback deferred via retain_handles_until_event"] = normalize_ws(
        "retain_handles_until_event({ std::move(owner) }, *event);"
    ) in normalize_ws(FREE_BODY_CODE)

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
    checks["graph scratch zone floor is additive"] = "bytes += onednn_graph_scratch_zone_floor_bytes()" in CACHE_CPP_CODE
    checks["zone floor env var"] = "GGML_SYCL_ONEDNN_GRAPH_ZONE_MB" in CACHE_CPP_CODE
    checks["allocator opt-out env var name"] = "GGML_SYCL_ONEDNN_CACHE_ALLOCATOR" in CACHE_CPP_CODE
    # Token-anchored default value (llama.cpp-gwno spec-review round 2,
    # finding: this used to be untested, so a 512->64 regression or typo
    # would pass silently) -- not a bare "64" substring search, which would
    # match line numbers, byte counts, or anything else in the file.
    # Specifically the `mb` initializer inside the floor helper's lambda.
    checks["default floor is 64 MiB"] = bool(re.search(r"\bmb\s*=\s*64\s*;", FLOOR_BODY_CODE))

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
