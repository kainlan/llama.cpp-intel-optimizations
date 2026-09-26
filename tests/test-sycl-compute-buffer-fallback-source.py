"""Source contract for llama.cpp-tsfl (nphx Task 2, track A; depends on Task 1
= llama.cpp-ibj0):

(A) Compute-buffer host-fallback visibility. The two host-pinned fallback
paths inside ggml_backend_sycl_buffer_type_alloc_buffer() -- a single
allocation exceeding the safe device-alloc limit, and a failed device alloc
retried host-pinned -- must log at GGML_LOG_WARN (GGML_LOG_INFO is dropped at
default verbosity, so a compute buffer silently landing in host memory looks
identical to a healthy run otherwise) and increment a per-device
g_compute_buffer_host_fallbacks[GGML_SYCL_MAX_DEVICES] counter, exposed via
ggml_backend_sycl_compute_buffer_host_fallbacks(int device) and reset to 0 by
the runtime-context transaction's own successful (publishing) path.

(B) A NON-PUBLISHING PROBE of the runtime-context transaction
(ggml_backend_sycl_probe_runtime_context_for_model(), nphx comment c-wgxn):
KV demotion is sticky and the transaction's own device-state side effects
(PP MoE oneDNN ring release/reserve) are real, so a probe that evaluates a
candidate (n_ctx, n_ubatch, ...) must never reach the transaction's
publication CAS (ggml_sycl::lifecycle_replace_placement_plan) and must leave
no physical trace behind. The probe and the publishing entry point
(ggml_backend_sycl_set_runtime_context()) share ONE static body,
ggml_sycl_run_runtime_context_transaction(), up to (not including) that CAS;
candidate refusals inside it log at GGML_LOG_INFO, not GGML_LOG_ERROR, in
probe mode.

Host-only, pure text assertions -- no SYCL device required, matching
test-sycl-ubatch-ring-replan-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Checks run against COMMENT-STRIPPED text so a positive structural check
cannot be fooled by prose that quotes a call the code does not actually
make.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()
# llama.cpp-oyfl / CLAUDE.md: plain file I/O, not the codescout index --
# ggml-sycl.cpp is ~100k lines and that index (and search_text's live scan)
# is documented blind/oversized for this specific file, so a tool-assisted
# search here would silently miss real occurrences.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()


# ---------------------------------------------------------------------------
# Small generic helpers, copied verbatim (not imported) from
# test-sycl-ubatch-ring-replan-source.py, which itself copied them from
# test-sycl-nonfa-attn-scratch-guard-source.py -- the established convention
# in this file family, stated explicitly in that file's own docstring. Each
# source-gate test file is registered and collected standalone by pytest
# (llama_test_pytest / CMakeLists.txt), with no shared conftest.py or
# importable helper module in tests/, so copying with attribution (rather
# than a cross-file import that would need sys.path surgery to survive
# standalone collection) is the working, already-proven pattern here.
# ---------------------------------------------------------------------------

# Single left-to-right alternation, not two sequential passes -- see
# test-sycl-onednn-graph-allocator-source.py's comment on why a block-comment
# pass run first can swallow a `/*` that appears inside a `//` comment's own
# prose (e.g. a glob like `weight-reclaim/*`).
_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments; keep string/char literals verbatim."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


GGML_SYCL_H_CODE = strip_comments(GGML_SYCL_H)
GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP)


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space, so a call or
    declaration re-wrapped by clang-format across lines still matches a
    single-line pattern."""
    return re.sub(r"\s+", " ", text)


def _bounded_body(code: str, start_marker: str, end_marker: str, *, after: int = 0) -> str:
    """Slice `code` from `start_marker` to the next `end_marker` after it,
    raising a clear assertion if either is not found -- shared by every check
    below so a match cannot silently come from some unrelated, later call
    site elsewhere in the (large) file."""
    start = code.find(start_marker, after)
    assert start != -1, f"{start_marker!r} not found"
    end = code.find(end_marker, start + 1)
    assert end != -1, f"could not bound {start_marker!r}'s body (looked for {end_marker!r})"
    return code[start:end]


def _body_of(raw: str, start_marker: str, end_marker: str) -> str:
    """Comment-strip `raw`, bound the result from `start_marker` to the next
    `end_marker`, and whitespace-normalize -- the pipeline every mutation
    witness needs to re-derive a checkable body from its freshly mutated raw
    string."""
    return _normalize_ws(_bounded_body(strip_comments(raw), start_marker, end_marker))


# Boundary literals named once, used by both the plain body-extraction
# helpers below and every mutation witness's _body_of() call.
_ALLOC_BUFFER_START = "static ggml_backend_buffer_t ggml_backend_sycl_buffer_type_alloc_buffer("
_ALLOC_BUFFER_END = "} catch (const sycl::exception & exc) {"
# llama.cpp-tsfl round 1 F6: return type changed from bool to the internal
# ggml_sycl_txn_result enum.
_TRANSACTION_START = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
_SET_RUNTIME_CONTEXT_START = "void ggml_backend_sycl_set_runtime_context("
# llama.cpp-jumy: return type changed from bool to the internal
# ggml_sycl_ring_replan_result enum (OK / RELEASE_REFUSED / DOES_NOT_FIT).
_REPLAN_START = "static ggml_sycl_ring_replan_result ggml_sycl_replan_pp_moe_onednn_ring("
_PROBE_START = "ggml_sycl_lifecycle_result ggml_backend_sycl_probe_runtime_context_for_model("
_AUTO_UBATCH_ENABLED_START = "bool ggml_backend_sycl_auto_ubatch_enabled("


def _alloc_buffer_body() -> str:
    return _bounded_body(GGML_SYCL_CPP_CODE, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)


def _transaction_body() -> str:
    # Bounded by the NEXT function's start (the thin wrapper this shared body
    # was extracted from) -- everything from _TRANSACTION_START ("static
    # ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction(") up to
    # (not including) "void ggml_backend_sycl_set_runtime_context(".
    return _bounded_body(GGML_SYCL_CPP_CODE, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)


def _probe_body() -> str:
    return _bounded_body(GGML_SYCL_CPP_CODE, _PROBE_START, _AUTO_UBATCH_ENABLED_START)


# ---------------------------------------------------------------------------
# (A) Compute-buffer host-fallback visibility
# ---------------------------------------------------------------------------


def test_large_buffer_fallback_is_warn_not_info():
    """A single allocation exceeding the safe device-alloc limit must log at
    WARN -- GGML_LOG_INFO is dropped at default verbosity, so a compute
    buffer silently landing in host memory would look identical to a healthy
    run."""
    body_norm = _normalize_ws(_alloc_buffer_body())
    assert re.search(
        r'GGML_LOG_WARN\(\s*"SYCL: Large buffer \(%zu MB\) exceeds safe alloc \(%zu MB\), '
        r'using host-pinned fallback',
        body_norm,
    ), "the oversize-request fallback must log at GGML_LOG_WARN, with its message text unchanged"
    assert not re.search(
        r'GGML_LOG_INFO\(\s*"SYCL: Large buffer \(%zu MB\) exceeds safe alloc', body_norm
    ), "the oversize-request fallback must no longer log at GGML_LOG_INFO"


def test_alloc_failed_fallback_is_warn_not_info():
    """A failed device allocation retried host-pinned must also log at WARN,
    for the same reason as the sibling check above."""
    body_norm = _normalize_ws(_alloc_buffer_body())
    assert re.search(
        r'GGML_LOG_WARN\(\s*"SYCL: Alloc failed \(%zu MB\), retrying with host-pinned fallback',
        body_norm,
    ), "the failed-alloc-retry fallback must log at GGML_LOG_WARN, with its message text unchanged"
    assert not re.search(
        r'GGML_LOG_INFO\(\s*"SYCL: Alloc failed \(%zu MB\), retrying with host-pinned fallback', body_norm
    ), "the failed-alloc-retry fallback must no longer log at GGML_LOG_INFO"


def test_kv_zone_fallback_stays_info():
    """The KV-zone fallback ("Arena RUNTIME zone full, runtime buffer ...
    allocated from KV zone") must stay GGML_LOG_INFO -- it is still device
    memory, and this task's own scope explicitly excludes it (llama.cpp-tsfl
    description, and llama.cpp-ibj0's sibling task's own gotcha)."""
    body_norm = _normalize_ws(_alloc_buffer_body())
    assert re.search(
        r'GGML_LOG_INFO\(\s*"\[SYCL\] Arena RUNTIME zone full, runtime buffer \(%.1f MB\) allocated from KV zone',
        body_norm,
    ), "the KV-zone fallback must remain GGML_LOG_INFO -- it must not have been promoted alongside the other two"


def test_fallback_level_changes_have_a_mutation_witness():
    """Mutation witness for the two WARN checks above: proves they would
    actually catch a level regressing back to INFO, rather than only ever
    passing on the current, correct source."""
    raw = GGML_SYCL_CPP
    warn_line_1 = (
        'GGML_LOG_WARN("SYCL: Large buffer (%zu MB) exceeds safe alloc (%zu MB), using host-pinned fallback\\n",'
    )
    warn_line_2 = 'GGML_LOG_WARN("SYCL: Alloc failed (%zu MB), retrying with host-pinned fallback\\n", size / (1024 * 1024));'
    assert warn_line_1 in raw, "mutation target not found -- update this witness to match the real source"
    assert warn_line_2 in raw, "mutation target not found -- update this witness to match the real source"

    mutated_raw = raw.replace(warn_line_1, warn_line_1.replace("GGML_LOG_WARN", "GGML_LOG_INFO"), 1)
    mutated_raw = mutated_raw.replace(warn_line_2, warn_line_2.replace("GGML_LOG_WARN", "GGML_LOG_INFO"), 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)
    assert not re.search(r'GGML_LOG_WARN\(\s*"SYCL: Large buffer', mutated_body_norm), (
        "mutation witness is broken: the regressed (INFO) source should no longer match the WARN check"
    )
    assert not re.search(r'GGML_LOG_WARN\(\s*"SYCL: Alloc failed', mutated_body_norm), (
        "mutation witness is broken: the regressed (INFO) source should no longer match the WARN check"
    )


def test_fallback_counter_declared_as_per_device_atomic():
    """g_compute_buffer_host_fallbacks must be a per-device
    std::atomic<uint64_t> array, sized GGML_SYCL_MAX_DEVICES, matching the
    other per-device atomics in this file (e.g. g_sycl_force_kernel_readback)."""
    assert re.search(
        r"std::atomic<uint64_t>\s+g_compute_buffer_host_fallbacks\[GGML_SYCL_MAX_DEVICES\]",
        GGML_SYCL_CPP_CODE,
    ), "g_compute_buffer_host_fallbacks must be declared as std::atomic<uint64_t>[GGML_SYCL_MAX_DEVICES]"


def _assert_oversize_increment_after_alloc_succeeded(body_norm: str) -> None:
    """llama.cpp-tsfl round 4 Q1 / round 5 R3: the oversize-request
    fallback's landing must be counted strictly between alloc_succeeded:'s
    own `if (forced_host_fallback)` guard and the `dev_ptr = main_alloc.ptr`
    statement that follows it -- not inline in the oversize branch itself,
    where the allocation attempt that decides success or failure has not
    yet run. Factored out so the real test below and its round 5 R3
    mutation witnesses call the IDENTICAL check, rather than each
    re-implementing a copy that could silently drift from it (the bug R3
    itself found: the previous witness's own re-implementation degenerated
    to a vacuous empty-range comparison on one of its two mutants instead of
    ever calling this logic)."""
    increments = [
        m.start()
        for m in re.finditer(r"g_compute_buffer_host_fallbacks\[buft_ctx->device\]\.fetch_add\(\s*1", body_norm)
    ]
    alloc_succeeded_idx = body_norm.find("alloc_succeeded:")
    assert alloc_succeeded_idx != -1, "could not find the alloc_succeeded: label"
    guard_idx = body_norm.find("if (forced_host_fallback)", alloc_succeeded_idx)
    assert guard_idx != -1, "could not find the forced_host_fallback guard after alloc_succeeded:"
    dev_ptr_idx = body_norm.find("dev_ptr = main_alloc.ptr", alloc_succeeded_idx)
    assert dev_ptr_idx != -1, "could not bound the forced_host_fallback guard block"
    assert any(guard_idx < idx < dev_ptr_idx for idx in increments), (
        "the oversize-request fallback must increment the counter AFTER the alloc_succeeded: label, INSIDE "
        "the forced_host_fallback guard -- not inline in the oversize branch itself"
    )


def test_both_fallback_sites_increment_the_counter():
    """Both host-pinned fallback paths inside
    ggml_backend_sycl_buffer_type_alloc_buffer()'s bounded body must land on
    a fetch_add(1, ...) of the per-device counter, each only once a
    SUCCESSFUL host-pinned allocation is confirmed. llama.cpp-tsfl round 4
    Q1: the oversize site no longer counts inline, at the point it merely
    DECIDES to force host-pinned (before the allocation attempt that
    decides success or failure has even run) -- it sets forced_host_
    fallback there instead, and is counted, guarded by that flag, at the
    shared alloc_succeeded: label, the same success point the retry site's
    own fetch_add already used (round 1 F10 / round 2 G1)."""
    body_norm = _normalize_ws(_alloc_buffer_body())
    increments = [
        m.start() for m in re.finditer(r"g_compute_buffer_host_fallbacks\[buft_ctx->device\]\.fetch_add\(\s*1", body_norm)
    ]
    assert len(increments) == 2, (
        f"expected exactly two fetch_add(1, ...) increments inside the alloc function's body -- found {len(increments)}"
    )

    warn1_idx = body_norm.find('"SYCL: Large buffer (%zu MB) exceeds safe alloc')
    warn2_idx = body_norm.find('"SYCL: Alloc failed (%zu MB), retrying with host-pinned fallback')
    assert warn1_idx != -1 and warn2_idx != -1

    forced_flag_idx = body_norm.find("forced_host_fallback = true")
    assert forced_flag_idx != -1 and forced_flag_idx < warn1_idx, (
        "the oversize branch must set forced_host_fallback = true, at or before its own WARN log"
    )

    _assert_oversize_increment_after_alloc_succeeded(body_norm)

    # llama.cpp-tsfl round 2 G1: the retry-path increment sits AFTER its own
    # WARN (which only announces the retry attempt, before the outcome is
    # known), inside the unified_alloc() success branch, immediately before
    # `goto alloc_succeeded` -- round 1 F10 moved it there deliberately so
    # only a retry that actually succeeds is counted, not merely attempted.
    # The PREVIOUS version of this check asserted the opposite (`idx <
    # warn2_idx`) and passed anyway, vacuously, on the OTHER increment (the
    # oversize-request one, which legitimately precedes warn1_idx and thus
    # also precedes warn2_idx).
    goto_idx = body_norm.find("goto alloc_succeeded", warn2_idx)
    assert goto_idx != -1, "could not find the `goto alloc_succeeded` that follows the retry WARN"
    assert any(warn2_idx < idx < goto_idx for idx in increments), (
        "the failed-alloc-retry fallback must increment the counter AFTER its own WARN log and BEFORE "
        "`goto alloc_succeeded` -- counted only once the retry itself succeeds, not merely attempted"
    )


def test_retry_increment_after_warn_has_a_mutation_witness():
    """Mutation witness for the ordering check above: proves it would
    actually catch the failed-alloc-retry counter increment moving back to
    BEFORE its own WARN, inside the retry's success branch, rather than
    only ever passing on the current, correct source. The check must
    FORBID `idx < warn2_idx`, not merely fail to require it -- a check that
    only required the ordering (rather than forbidding its opposite) would
    still pass on this mutant, satisfied by the OTHER (oversize-request)
    increment, which legitimately precedes both WARN calls."""
    raw = GGML_SYCL_CPP
    original_block = (
        '            GGML_LOG_WARN("SYCL: Alloc failed (%zu MB), retrying with host-pinned fallback\\n", '
        "size / (1024 * 1024));\n"
        "            req.intent.constraints.must_device      = false;\n"
        "            req.intent.constraints.must_host_pinned = true;\n"
        "            if (ggml_sycl::unified_alloc(req, &main_alloc) && main_alloc.ptr != nullptr) {\n"
        "                g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n"
        "                goto alloc_succeeded;\n"
        "            }\n"
    )
    assert original_block in raw, "mutation target block not found -- update this witness to match the real source"
    mutated_block = (
        "            g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n"
        '            GGML_LOG_WARN("SYCL: Alloc failed (%zu MB), retrying with host-pinned fallback\\n", '
        "size / (1024 * 1024));\n"
        "            req.intent.constraints.must_device      = false;\n"
        "            req.intent.constraints.must_host_pinned = true;\n"
        "            if (ggml_sycl::unified_alloc(req, &main_alloc) && main_alloc.ptr != nullptr) {\n"
        "                goto alloc_succeeded;\n"
        "            }\n"
    )
    mutated_raw = raw.replace(original_block, mutated_block, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)
    mutated_increments = [
        m.start()
        for m in re.finditer(
            r"g_compute_buffer_host_fallbacks\[buft_ctx->device\]\.fetch_add\(\s*1", mutated_body_norm
        )
    ]
    mutated_warn2_idx = mutated_body_norm.find('"SYCL: Alloc failed (%zu MB), retrying with host-pinned fallback')
    mutated_goto_idx = mutated_body_norm.find("goto alloc_succeeded", mutated_warn2_idx)
    assert mutated_warn2_idx != -1 and mutated_goto_idx != -1

    assert not any(mutated_warn2_idx < idx < mutated_goto_idx for idx in mutated_increments), (
        "mutation witness is broken: the mutated (pre-F10-shape, increment-before-WARN) source should FAIL "
        "the 'increment after WARN, before goto' check, but it did not"
    )


def test_oversize_guard_deleted_has_a_mutation_witness():
    """Mutation witness A for llama.cpp-tsfl round 5 R3: proves
    _assert_oversize_increment_after_alloc_succeeded() would actually catch
    the forced_host_fallback guard block being deleted entirely.

    R3 (reviewer finding): the PREVIOUS version of this witness deleted the
    whole `if (forced_host_fallback) { ... }` block, then re-implemented its
    own (buggy) copy of the check instead of calling the real one -- with
    guard_idx now -1, its `guard_bound_start = mutated_dev_ptr_idx` fallback
    collapsed the comparison range to (dev_ptr_idx, dev_ptr_idx), and
    `not any(idx in an EMPTY range)` is vacuously True regardless of the
    mutant. It never actually exercised the real check's own
    `assert guard_idx != -1` failure at all. Calling the shared helper
    directly, inside pytest.raises, fixes this: the helper's own assertion
    is what fires, not a re-implemented approximation of it."""
    raw = GGML_SYCL_CPP
    guarded_increment = (
        "    if (forced_host_fallback) {\n"
        "        g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n"
        "    }\n"
    )
    assert guarded_increment in raw, "mutation target (guarded increment) not found -- update this witness"
    mutated_raw = raw.replace(guarded_increment, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)
    with pytest.raises(AssertionError, match="could not find the forced_host_fallback guard"):
        _assert_oversize_increment_after_alloc_succeeded(mutated_body_norm)


def test_oversize_increment_moved_above_warn_has_a_mutation_witness():
    """Mutation witness B for llama.cpp-tsfl round 5 R3: proves the shared
    check's POSITIONAL clause (the increment must fall strictly between the
    guard and dev_ptr_idx) is exercised too, not just the guard's mere
    existence -- keeps the `if (forced_host_fallback) { ... }` block (now
    empty) and moves the fetch_add back inline in the oversize branch,
    right after forced_host_fallback is set -- exactly the pre-Q1 shape,
    just with the (now useless) empty guard left behind."""
    raw = GGML_SYCL_CPP
    guarded_increment = (
        "    if (forced_host_fallback) {\n"
        "        g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n"
        "    }\n"
    )
    assert guarded_increment in raw, "mutation target (guarded increment) not found -- update this witness"
    emptied_guard = "    if (forced_host_fallback) {\n    }\n"

    oversize_flag_set = "            forced_host_fallback = true;\n"
    assert raw.count(oversize_flag_set) == 1, (
        f"mutation target (oversize flag-set) not unique -- found {raw.count(oversize_flag_set)}"
    )

    mutated_raw = raw.replace(guarded_increment, emptied_guard, 1)
    mutated_raw = mutated_raw.replace(
        oversize_flag_set,
        oversize_flag_set
        + "            g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n",
        1,
    )
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)
    with pytest.raises(AssertionError, match="must increment the counter AFTER the alloc_succeeded"):
        _assert_oversize_increment_after_alloc_succeeded(mutated_body_norm)


def test_increment_sites_have_a_mutation_witness():
    """Mutation witness for the increment-count check above: proves it would
    actually catch one of the two increments being deleted."""
    raw = GGML_SYCL_CPP
    # llama.cpp-tsfl round 2 (companion to G1) / round 4 Q1 (indentation
    # shifted again by Q1's move of the oversize site's increment to the
    # alloc_succeeded: label): the two increment lines have DIFFERENT
    # indentation -- 16 spaces for the retry site (nested inside its own
    # success branch), 8 spaces for the oversize site (nested inside the
    # `if (forced_host_fallback)` guard at alloc_succeeded:, round 4 Q1). An
    # unanchored 8-space literal would also match as a trailing substring of
    # the 16-space line (its last 8 spaces + the rest coincide), so this
    # anchors with a leading "\n" to restrict the match to a line that
    # starts with EXACTLY this indentation -- the oversize site.
    increment_line = "\n        g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n"
    count = raw.count(increment_line)
    assert count == 1, f"expected exactly one line at this exact indentation (the oversize site) -- found {count}"
    mutated_raw = raw.replace(increment_line, "\n", 1)
    assert mutated_raw != raw

    def _increment_count(raw_source: str) -> int:
        body_norm = _body_of(raw_source, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)
        return len(re.findall(r"g_compute_buffer_host_fallbacks\[buft_ctx->device\]\.fetch_add\(\s*1", body_norm))

    original_count = _increment_count(raw)
    mutated_count = _increment_count(mutated_raw)
    assert original_count == 2
    assert mutated_count == original_count - 1, (
        f"mutation witness is broken: deleting one increment line must drop the count by exactly one -- found "
        f"{mutated_count} (expected {original_count - 1})"
    )


def test_transaction_resets_the_counter_on_success_after_the_cas():
    """llama.cpp-tsfl round 1 F4: ggml_sycl_run_runtime_context_transaction()'s
    PUBLISH path must reset g_compute_buffer_host_fallbacks[ctx->device] to 0
    on the SUCCESS TAIL, strictly AFTER the CAS has actually succeeded --
    not any earlier, where the publication-ID check, MMID materialization,
    or the CAS itself could still refuse and this transaction never
    publishes anything. ggml_backend_sycl_compute_buffer_host_fallbacks()
    (ggml-sycl.h) promises "since the last SUCCESSFUL runtime-context
    transaction"; resetting before those three could-still-fail steps broke
    that promise for every one of their refusal paths. (This test used to be
    named ...before_publish and pinned the OLD, wrong position -- before the
    CAS; F4 corrected the code and this test.)"""
    body_norm = _normalize_ws(_transaction_body())
    assert re.search(
        r"g_compute_buffer_host_fallbacks\[ctx->device\]\.store\(\s*0", body_norm
    ), "the transaction body must reset the counter with .store(0, ...)"

    reset_idx = body_norm.find("g_compute_buffer_host_fallbacks[ctx->device].store(0")
    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert reset_idx != -1 and cas_idx != -1
    assert cas_idx < reset_idx < len(body_norm), (
        "the counter reset must sit AFTER the CAS (ggml_sycl::lifecycle_replace_placement_plan) -- resetting "
        "any earlier risks resetting the counter for a transaction that still goes on to refuse"
    )


def test_reset_site_has_a_mutation_witness():
    """Mutation witness for the reset-site check above: proves it would
    actually catch the reset call being deleted."""
    raw = GGML_SYCL_CPP
    reset_line = "    g_compute_buffer_host_fallbacks[ctx->device].store(0, std::memory_order_relaxed);\n"
    assert raw.count(reset_line) == 1, f"expected exactly one reset line -- found {raw.count(reset_line)}"
    mutated_raw = raw.replace(reset_line, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)
    assert not re.search(r"g_compute_buffer_host_fallbacks\[ctx->device\]\.store\(\s*0", mutated_body_norm), (
        "mutation witness is broken: deleting the reset line left a reference to it behind"
    )


def test_reset_after_cas_has_a_mutation_witness():
    """Mutation witness for llama.cpp-tsfl round 1 F4 itself: proves the
    ordering check above would actually catch the reset moving back to its
    OLD (pre-F4) position, immediately before `auto next = ...` and
    therefore before the publication-ID check, MMID materialization, and
    the CAS -- exactly the bug F4 fixed."""
    raw = GGML_SYCL_CPP
    reset_line = "    g_compute_buffer_host_fallbacks[ctx->device].store(0, std::memory_order_relaxed);\n"
    assert raw.count(reset_line) == 1, f"expected exactly one reset line -- found {raw.count(reset_line)}"
    next_snapshot_line = "    auto next     = std::make_shared<ggml_sycl::lifecycle_plan_snapshot>(*current);\n"
    assert next_snapshot_line in raw, "mutation target (next snapshot line) not found -- update this witness"

    mutated_raw = raw.replace(reset_line, "", 1)
    mutated_raw = mutated_raw.replace(next_snapshot_line, reset_line + next_snapshot_line, 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)
    mutated_reset_idx = mutated_body_norm.find("g_compute_buffer_host_fallbacks[ctx->device].store(0")
    mutated_cas_idx = mutated_body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert mutated_reset_idx != -1 and mutated_cas_idx != -1
    assert not (mutated_cas_idx < mutated_reset_idx < len(mutated_body_norm)), (
        "mutation witness is broken: the mutated (pre-F4, reset-before-CAS) source should FAIL the "
        "cas_idx < reset_idx ordering check, but it did not"
    )


def test_header_declares_the_fallback_accessor():
    """ggml_backend_sycl_compute_buffer_host_fallbacks(int device) must be
    declared in ggml-sycl.h, next to ggml_backend_sycl_set_runtime_context()."""
    assert re.search(
        r"GGML_BACKEND_API\s+uint64_t\s+ggml_backend_sycl_compute_buffer_host_fallbacks\(\s*int\s+device\s*\)\s*;",
        GGML_SYCL_H_CODE,
    ), "ggml_backend_sycl_compute_buffer_host_fallbacks(int device) must be declared in ggml-sycl.h"

    idx_accessor = GGML_SYCL_H_CODE.find("ggml_backend_sycl_compute_buffer_host_fallbacks")
    idx_set_runtime_context = GGML_SYCL_H_CODE.find("ggml_backend_sycl_set_runtime_context(ggml_backend_t backend")
    assert idx_accessor != -1 and idx_set_runtime_context != -1
    assert 0 < idx_accessor - idx_set_runtime_context < 2000, (
        "the accessor should be declared close to ggml_backend_sycl_set_runtime_context(), per the task spec"
    )


def test_accessor_is_defined_and_not_file_static():
    """The accessor function must actually be defined in ggml-sycl.cpp, and
    exported (not file-static) so llama-context.cpp (a different translation
    unit) can call it."""
    assert re.search(
        r"uint64_t\s+ggml_backend_sycl_compute_buffer_host_fallbacks\(\s*int\s+device\s*\)\s*\{",
        GGML_SYCL_CPP_CODE,
    ), "ggml_backend_sycl_compute_buffer_host_fallbacks() must be defined in ggml-sycl.cpp"
    assert not re.search(
        r"static\s+uint64_t\s+ggml_backend_sycl_compute_buffer_host_fallbacks\(", GGML_SYCL_CPP_CODE
    ), "the accessor must not be file-static"


# ---------------------------------------------------------------------------
# (B) The non-publishing runtime-context probe
# ---------------------------------------------------------------------------


def test_header_declares_the_probe_struct_and_function():
    """ggml_sycl_runtime_context_probe (with its four documented fields) and
    ggml_backend_sycl_probe_runtime_context_for_model() must both be declared
    in ggml-sycl.h, next to ggml_backend_sycl_set_runtime_context_for_model()."""
    struct_match = re.search(
        r"struct\s+ggml_sycl_runtime_context_probe\s*\{([^}]*)\}\s*;", GGML_SYCL_H_CODE, flags=re.DOTALL
    )
    assert struct_match is not None, "struct ggml_sycl_runtime_context_probe must be declared in ggml-sycl.h"
    fields_norm = _normalize_ws(struct_match.group(1))
    for field in ("accepted", "would_demote_kv", "host_kv_bytes", "reason"):
        assert re.search(r"\b" + field + r"\b", fields_norm), (
            f"ggml_sycl_runtime_context_probe must declare a `{field}` field"
        )

    assert re.search(
        r"GGML_BACKEND_API\s+enum\s+ggml_sycl_lifecycle_result\s+ggml_backend_sycl_probe_runtime_context_for_model\s*\(",
        GGML_SYCL_H_CODE,
    ), "ggml_backend_sycl_probe_runtime_context_for_model() must be declared in ggml-sycl.h"

    idx_probe_decl = GGML_SYCL_H_CODE.find("GGML_BACKEND_API enum ggml_sycl_lifecycle_result "
                                           "ggml_backend_sycl_probe_runtime_context_for_model(")
    idx_publish_decl = GGML_SYCL_H_CODE.find("ggml_backend_sycl_set_runtime_context_for_model(\n    ggml_backend_t")
    assert idx_probe_decl != -1 and idx_publish_decl != -1
    assert 0 < idx_probe_decl - idx_publish_decl < 2500, (
        "the probe declaration should sit close to ggml_backend_sycl_set_runtime_context_for_model(), per the "
        "task spec"
    )


def test_probe_and_publisher_share_one_static_body():
    """Both ggml_backend_sycl_set_runtime_context() (the publisher) and
    ggml_backend_sycl_probe_runtime_context_for_model() (the probe) must call
    the SAME shared static function,
    ggml_sycl_run_runtime_context_transaction() -- not two independent
    admission implementations that could silently drift apart."""
    # llama.cpp-tsfl round 1 F6: return type is now the internal
    # ggml_sycl_txn_result enum, not bool.
    assert re.search(
        r"static\s+ggml_sycl_txn_result\s+ggml_sycl_run_runtime_context_transaction\s*\(", GGML_SYCL_CPP_CODE
    ), "ggml_sycl_run_runtime_context_transaction() must exist and be file-static"

    publisher_body_norm = _normalize_ws(
        _bounded_body(GGML_SYCL_CPP_CODE, _SET_RUNTIME_CONTEXT_START, _PROBE_START)
    )
    assert "ggml_sycl_run_runtime_context_transaction(" in publisher_body_norm, (
        "ggml_backend_sycl_set_runtime_context() must call ggml_sycl_run_runtime_context_transaction()"
    )
    # Comment-stripped text has already dropped any /*probe_mode=*/-style
    # inline comment (strip_comments() removes /* */ blocks) -- match the
    # positional call itself: (..., flash_attn_enabled, false, nullptr).
    # llama.cpp-3aos: tolerate additional bare arguments inserted between
    # n_seq_max and flash_attn_enabled (e.g. kv_unified) -- the intent is
    # "probe_mode=false, out=nullptr reach the shared body", not that these
    # two parameters are adjacent.
    assert re.search(
        r"ggml_sycl_run_runtime_context_transaction\(\s*backend,\s*n_ctx,\s*n_ubatch,\s*n_seq_max,(?:\s*\w+,)*\s*"
        r"flash_attn_enabled,\s*false,\s*nullptr\s*\)",
        publisher_body_norm,
    ), "the publishing wrapper must call the shared body with probe_mode=false, out=nullptr"

    probe_body_norm = _normalize_ws(_probe_body())
    assert "ggml_sycl_run_runtime_context_transaction(" in probe_body_norm, (
        "ggml_backend_sycl_probe_runtime_context_for_model() must call ggml_sycl_run_runtime_context_transaction()"
    )
    assert re.search(
        r"ggml_sycl_run_runtime_context_transaction\(\s*backend,\s*n_ctx,\s*n_ubatch,\s*n_seq_max,(?:\s*\w+,)*\s*"
        r"flash_attn_enabled,\s*true,\s*out\s*\)",
        probe_body_norm,
    ), "the probe must call the shared body with probe_mode=true, out=out"


def test_shared_body_call_sites_have_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    the probe silently going back to calling the OLD, non-shared, direct
    logic instead of the shared transaction function."""
    raw = GGML_SYCL_CPP
    # llama.cpp-tsfl round 1 F6: the probe now dispatches the shared body's
    # ggml_sycl_txn_result through a switch, not a bare bool.
    # llama.cpp-uajm: swa_full sits between kv_unified and flash_attn_enabled.
    probe_call = (
        "    const ggml_sycl_txn_result result = ggml_sycl_run_runtime_context_transaction(\n"
        "        backend, n_ctx, n_ubatch, n_seq_max, kv_unified, swa_full, flash_attn_enabled, /*probe_mode=*/true, out);\n"
    )
    assert probe_call in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(probe_call, "", 1)
    assert mutated_raw != raw

    mutated_probe_body_norm = _body_of(mutated_raw, _PROBE_START, _AUTO_UBATCH_ENABLED_START)
    assert "ggml_sycl_run_runtime_context_transaction(" not in mutated_probe_body_norm, (
        "mutation witness is broken: deleting the call left a reference to it behind"
    )


# llama.cpp-tsfl round 1 F2/F8: after F2 added a SECOND `if (probe_mode) {`
# inside the transaction body (the KV-demotion WARN/INFO branch, much
# earlier in the function than the probe's own exit branch), a plain
# re.search() for the bare `if (probe_mode) {` pattern finds THAT one
# first, not the exit branch this section's checks actually care about.
# Identify the exit branch specifically by content unique to it: it is the
# only `if (probe_mode)` block that captures a rollback result into
# `rollback_ok` (round 1 F1).
def _find_probe_exit_branch(body_norm: str):
    for m in re.finditer(r"if\s*\(\s*probe_mode\s*\)\s*\{", body_norm):
        if "rollback_ok" in body_norm[m.start() : m.start() + 800]:
            return m
    return None


def test_probe_mode_branch_returns_before_the_cas():
    """The shared transaction body's probe EXIT branch (`if (probe_mode) {
    ... return ggml_sycl_txn_result::ACCEPTED; }`, identified by
    _find_probe_exit_branch() above -- NOT the separate F2 KV-demotion
    `if (probe_mode)` branch) must appear BEFORE the CAS
    (ggml_sycl::lifecycle_replace_placement_plan) -- both bounds asserted:
    the branch must exist, the CAS call must exist, and the branch's own
    success return must precede the CAS textually (the shared body is
    straight-line code with early returns, so "precedes textually" means
    "cannot execute past this point when probe_mode is true, because every
    branch between here and the CAS is either this one's own return or an
    earlier, unconditional refusal")."""
    body_norm = _normalize_ws(_transaction_body())

    probe_branch = _find_probe_exit_branch(body_norm)
    assert probe_branch is not None, (
        "the shared body must have an `if (probe_mode) { ... }` EXIT branch, identified by its own "
        "rollback_ok capture"
    )

    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert cas_idx != -1, "could not find the CAS call (ggml_sycl::lifecycle_replace_placement_plan)"

    probe_return_idx = body_norm.find("return ggml_sycl_txn_result::ACCEPTED", probe_branch.start())
    assert probe_return_idx != -1 and probe_return_idx < cas_idx, (
        "probe_mode is not the anonymous case -- the branch's own success return must precede the CAS"
    )
    assert probe_branch.start() < cas_idx, "the probe_mode branch itself must precede the CAS"

    # The probe_mode branch must be the ONLY branch between the ring re-plan
    # call and the CAS that can return true without having built `next`
    # (the shared_ptr snapshot the CAS operates on) -- i.e. `auto next =`
    # must appear strictly between the probe_mode branch and the CAS, never
    # before it, so a probe genuinely cannot reach the CAS through any path.
    next_snapshot_idx = body_norm.find("auto next = std::make_shared<ggml_sycl::lifecycle_plan_snapshot>(*current)")
    assert next_snapshot_idx != -1, "could not find the `next` snapshot construction"
    assert probe_branch.start() < next_snapshot_idx < cas_idx, (
        "the `next` snapshot (which the CAS operates on) must be built strictly AFTER the probe_mode branch and "
        "BEFORE the CAS -- a probe must never reach a point where `next` exists"
    )


def test_probe_branch_ordering_has_a_mutation_witness():
    """Mutation witness for llama.cpp-tsfl round 1 F8: proves the ordering
    check above would actually catch the probe's EXIT branch being moved to
    AFTER the CAS (so a probe could reach it), by actually MOVING the
    branch's text, not merely neutering its condition.

    F8 (reviewer finding): the PREVIOUS version of this witness rewrote the
    guard to `if (false && probe_mode)`, which only stops that literal
    regex from matching -- an EXISTENCE mutation. It does not test the
    ORDERING assertion at all: a probe_mode branch genuinely relocated to
    after the CAS, verbatim, would still satisfy "the branch exists
    somewhere" while failing exactly the property the ordering check exists
    to catch. This version cuts the exit branch's own text out of its
    original position and re-inserts it, byte-identical, immediately after
    the publish call -- a real reordering."""
    raw = GGML_SYCL_CPP
    # Anchored with a leading "\n" so this matches only a LINE that starts
    # with exactly 4 spaces then "if (probe_mode) {" -- plain substring
    # search without the anchor also matches as a SUFFIX of the two more
    # deeply indented if(probe_mode) blocks (F2's demotion branch at 12
    # spaces, F3's ring-replan success branch at 8), since "    if
    # (probe_mode) {\n" is itself a trailing substring of both.
    branch_start_marker = "\n    if (probe_mode) {\n"
    branch_end_marker = "        return ggml_sycl_txn_result::ACCEPTED;\n    }\n"
    publish_marker = "    ggml_sycl_publish_prepared_plan_locked(prepared_publication);\n"

    assert raw.count(branch_start_marker) == 1, (
        f"mutation target (branch start) not unique -- found {raw.count(branch_start_marker)}; the exit "
        "branch is identified by exactly 4-space indentation at the start of a line, distinct from the two "
        "other, more deeply indented if(probe_mode) blocks (F2's demotion branch, F3's ring-replan success "
        "branch)"
    )
    start = raw.find(branch_start_marker) + 1  # skip the leading "\n" itself -- it belongs to the previous line
    end = raw.find(branch_end_marker, start)
    assert end != -1, "mutation target (branch end) not found after the branch start"
    end += len(branch_end_marker)
    branch_block = raw[start:end]

    # llama.cpp-tsfl: search from `end`, not index 0 -- an unrelated,
    # earlier occurrence of this exact call (a different function, the
    # model-load path) exists elsewhere in this ~100k-line file.
    publish_idx = raw.find(publish_marker, end)
    assert publish_idx != -1, "mutation target (publish call) not found after the branch"

    mutated_raw = raw[:start] + raw[end:]
    # Same "search from `start`, not 0" reasoning as above -- the cut did
    # not move `start`'s own position in the file, so the unrelated earlier
    # occurrence is still there ahead of it.
    insert_at = mutated_raw.find(publish_marker, start) + len(publish_marker)
    mutated_raw = mutated_raw[:insert_at] + branch_block + mutated_raw[insert_at:]
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)
    mutated_probe_branch = _find_probe_exit_branch(mutated_body_norm)
    assert mutated_probe_branch is not None, "mutation witness is broken: could not re-find the moved branch"
    mutated_cas_idx = mutated_body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert mutated_cas_idx != -1

    # The REAL (fixed) check requires probe_branch.start() < cas_idx; on this
    # genuinely-reordered mutant that must now be FALSE.
    assert not (mutated_probe_branch.start() < mutated_cas_idx), (
        "mutation witness is broken: the F8 ordering check should FAIL on this genuinely-reordered mutant "
        "(branch moved after the CAS), but the branch still appears to precede it"
    )


def test_probe_rolls_back_the_ring_before_returning():
    """The probe_mode branch must roll the PP MoE oneDNN ring back to its
    pre-transaction n_ubatch (with probe_mode=true, so the rollback's own
    logging is also quiet) before returning -- the ring re-plan call just
    above it already performed real device-state side effects (release +
    reserve), which a probe must not leave behind."""
    body_norm = _normalize_ws(_transaction_body())
    # llama.cpp-tsfl round 1 F2/F8: identify the EXIT branch specifically
    # (see _find_probe_exit_branch's own comment) -- a plain re.search()
    # would find the earlier F2 KV-demotion `if (probe_mode)` branch
    # instead, and this test happened to still pass on that wrong anchor
    # only because the resulting slice was a SUPERSET that also contained
    # the real exit branch (the two are adjacent, with nothing excluded in
    # between) -- a coincidental pass, not a correct one.
    probe_branch = _find_probe_exit_branch(body_norm)
    assert probe_branch is not None
    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert cas_idx != -1, "could not find the CAS call (ggml_sycl::lifecycle_replace_placement_plan)"
    probe_block = body_norm[probe_branch.start():cas_idx]

    # Comment-stripped text drops any /*probe_mode=*/-style inline comment --
    # match the positional call itself.
    assert re.search(
        r"ggml_sycl_replan_pp_moe_onednn_ring\(\s*ctx->device,\s*pre_replan_pp_moe_ring_n_ubatch,\s*true\s*\)",
        probe_block,
    ), "the probe_mode branch must roll the ring back with probe_mode=true"

    out_fill_idx = probe_block.find('out->accepted = true')
    rollback_idx = probe_block.find("ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, pre_replan_pp_moe_ring_n_ubatch")
    assert out_fill_idx != -1 and rollback_idx != -1
    assert rollback_idx < out_fill_idx, (
        "the ring rollback must happen BEFORE out->accepted is filled and the branch returns -- not after, "
        "which would leave a window where the ring is still in its (unrolled-back) probed state"
    )


def test_ring_release_refused_is_classified_busy():
    """llama.cpp-jumy: ggml_sycl_replan_pp_moe_onednn_ring() reports
    RELEASE_REFUSED when release_pp_moe_onednn_scratch_ring() refuses
    because the ring is still claimed by an in-flight dispatch -- a
    TRANSIENT condition unrelated to whether the requested size fits. The
    shared transaction body must classify only that shape as BUSY (a caller
    may retry); every other non-OK value (the sizing overflow, the F13
    pre-check, or a failed Step 3 reserve) must stay the deterministic
    REFUSED "does not fit" answer, unchanged from before this task."""
    body_norm = _normalize_ws(_transaction_body())
    busy_match = re.search(
        r"if\s*\(\s*ring_replan_result\s*==\s*ggml_sycl_ring_replan_result::RELEASE_REFUSED\s*\)\s*\{\s*"
        r'return\s+busy\(\s*"busy \(PP MoE oneDNN scratch ring claimed by an in-flight dispatch\)"\s*\)',
        body_norm,
    )
    assert busy_match is not None, (
        "a RELEASE_REFUSED ring re-plan result must return busy(...) with a distinct, transient-sounding reason"
    )
    refused_match = re.search(
        r"if\s*\(\s*ring_replan_result\s*!=\s*ggml_sycl_ring_replan_result::OK\s*\)\s*\{\s*"
        r'return\s+refuse\(\s*"PP MoE oneDNN scratch ring does not fit"\s*\)',
        body_norm,
    )
    assert refused_match is not None, (
        "every other non-OK ring re-plan result must still refuse() -- the deterministic 'does not fit' answer"
    )
    assert busy_match.start() < refused_match.start(), (
        "the RELEASE_REFUSED (BUSY) check must run BEFORE the general non-OK (REFUSED) check, or "
        "RELEASE_REFUSED would be caught by the broader != OK check first and misclassified as REFUSED"
    )


def test_ring_release_refused_busy_classification_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    the RELEASE_REFUSED branch being deleted -- collapsing back to the
    pre-task shape where every ring re-plan failure, including a transient
    release refusal, was classified REFUSED."""
    raw = GGML_SYCL_CPP
    busy_branch = (
        "    if (ring_replan_result == ggml_sycl_ring_replan_result::RELEASE_REFUSED) {\n"
        "        // refusal already logged (ERROR normally, INFO in probe mode)\n"
        '        return busy("busy (PP MoE oneDNN scratch ring claimed by an in-flight dispatch)");\n'
        "    }\n"
    )
    assert busy_branch in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(busy_branch, "", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)
    assert not re.search(
        r"ring_replan_result\s*==\s*ggml_sycl_ring_replan_result::RELEASE_REFUSED", mutated_body_norm
    ), "mutation witness is broken: deleting the branch left a reference to it behind"


def test_probe_rollback_failure_logs_error_not_warn():
    """llama.cpp-jumy: the probe's OWN rollback failure (its attempt to roll
    the PP MoE oneDNN scratch ring back to the pre-transaction n_ubatch
    itself fails) is an anomaly, not a candidate refusal --
    GGML_SYCL_RUNTIME_TXN_REFUSAL's own policy comment says an anomaly stays
    GGML_LOG_ERROR unconditionally, probe or not (matching the sibling
    "restore FAILED" anomaly a few lines above it in the same function).
    This log must be ERROR, not WARN."""
    body_norm = _normalize_ws(_transaction_body())
    probe_branch = _find_probe_exit_branch(body_norm)
    assert probe_branch is not None
    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert cas_idx != -1, "could not find the CAS call (ggml_sycl::lifecycle_replace_placement_plan)"
    probe_block = body_norm[probe_branch.start():cas_idx]

    assert "probe rollback of the PP MoE oneDNN scratch ring" in probe_block, (
        "could not find the probe's own rollback-failure message"
    )
    assert re.search(
        r'GGML_LOG_ERROR\(\s*"\[SYCL-PLAN\] probe rollback of the PP MoE oneDNN scratch ring',
        probe_block,
    ), "the probe's own rollback failure must log at GGML_LOG_ERROR, not GGML_LOG_WARN -- it is an anomaly"
    assert not re.search(
        r'GGML_LOG_WARN\(\s*"\[SYCL-PLAN\] probe rollback of the PP MoE oneDNN scratch ring',
        probe_block,
    ), "the probe's own rollback failure must not log at GGML_LOG_WARN"


def test_probe_rollback_failure_log_level_has_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    the rollback-failure log reverting to GGML_LOG_WARN."""
    raw = GGML_SYCL_CPP
    error_call = (
        "            GGML_LOG_ERROR(\n"
        '                "[SYCL-PLAN] probe rollback of the PP MoE oneDNN scratch ring to n_ubatch=%u failed; ring left "\n'
    )
    assert error_call in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(error_call, error_call.replace("GGML_LOG_ERROR", "GGML_LOG_WARN"), 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)
    assert re.search(
        r'GGML_LOG_WARN\(\s*"\[SYCL-PLAN\] probe rollback of the PP MoE oneDNN scratch ring',
        mutated_body_norm,
    ), "mutation witness is broken: could not find the mutated (WARN) log line"


def test_refusal_macro_downgrades_to_info_in_probe_mode():
    """GGML_SYCL_RUNTIME_TXN_REFUSAL(probe_mode, ...) must dispatch to
    GGML_LOG_INFO when probe_mode is true and GGML_LOG_ERROR otherwise -- the
    single mechanism candidate refusals (ring re-plan, non-FA scratch, KV
    budget) use to stay quiet under a probe while remaining loud on the
    publishing path."""
    macro_match = re.search(
        r"#define\s+GGML_SYCL_RUNTIME_TXN_REFUSAL\(probe_mode,\s*\.\.\.\)", GGML_SYCL_CPP_CODE
    )
    assert macro_match is not None, "GGML_SYCL_RUNTIME_TXN_REFUSAL(probe_mode, ...) must be defined"

    # The macro body is the next ~10 lines of raw (non-comment-stripped)
    # text, since a macro definition must not be comment-stripped for its
    # backslash line continuations to be trustworthy; re-search the RAW
    # source directly, bounded by the macro's own #define and its `while (0)`
    # close.
    raw_start = GGML_SYCL_CPP.find("#define GGML_SYCL_RUNTIME_TXN_REFUSAL(probe_mode, ...)")
    assert raw_start != -1
    raw_end = GGML_SYCL_CPP.find("while (0)", raw_start)
    assert raw_end != -1
    macro_body = GGML_SYCL_CPP[raw_start:raw_end]
    # Drop line-continuation backslashes before whitespace-normalizing --
    # a macro body is a sequence of `... \` -newline pairs, and
    # _normalize_ws() only collapses whitespace RUNS, leaving the backslash
    # itself as a literal character between tokens.
    macro_body_norm = _normalize_ws(macro_body.replace("\\\n", " "))

    assert re.search(r"if\s*\(\s*probe_mode\s*\)\s*\{\s*GGML_LOG_INFO\(__VA_ARGS__\)", macro_body_norm), (
        "the macro must call GGML_LOG_INFO when probe_mode is true"
    )
    assert re.search(r"else\s*\{\s*GGML_LOG_ERROR\(__VA_ARGS__\)", macro_body_norm), (
        "the macro must call GGML_LOG_ERROR when probe_mode is false"
    )


def test_refusal_macro_used_at_the_kv_budget_and_ring_and_nonfa_sites():
    """The macro must actually be used at the candidate-refusal sites the
    task spec names (ring re-plan, non-FA attention scratch, KV budget) --
    declaring it and never using it would satisfy the check above while
    changing nothing about what a probe actually prints."""
    # clang-format may or may not break `probe_mode,` onto its own
    # continuation line depending on the call's overall width -- match
    # loosely on the paren/argument boundary, not on exact wrapping.
    macro_call_re = re.compile(r"GGML_SYCL_RUNTIME_TXN_REFUSAL\(\s*probe_mode\s*,")

    transaction_body_norm = _normalize_ws(_transaction_body())
    assert macro_call_re.search(transaction_body_norm), (
        "the shared transaction body must use the refusal macro for its own KV/MMID budget refusals"
    )

    ring_fn_body_norm = _normalize_ws(_bounded_body(GGML_SYCL_CPP_CODE, _REPLAN_START, _TRANSACTION_START))
    assert macro_call_re.search(ring_fn_body_norm), (
        "ggml_sycl_replan_pp_moe_onednn_ring() must use the refusal macro for its own candidate refusals"
    )

    nonfa_fn_body_norm = _normalize_ws(
        _bounded_body(GGML_SYCL_CPP_CODE, "static bool ggml_sycl_check_nonfa_attn_scratch(", _REPLAN_START)
    )
    assert macro_call_re.search(nonfa_fn_body_norm), (
        "ggml_sycl_check_nonfa_attn_scratch() must use the refusal macro for its own candidate refusals"
    )

    # And every one of those three functions must actually receive a
    # probe_mode parameter for the macro call above to even compile. The ring
    # re-plan takes its KV-zone inputs after it (llama.cpp-u1bb).
    assert re.search(r"ggml_sycl_replan_pp_moe_onednn_ring\([^)]*bool\s+probe_mode\s*=\s*false\s*[,)]",
                     GGML_SYCL_CPP_CODE), "ggml_sycl_replan_pp_moe_onednn_ring() must take a probe_mode parameter"
    assert re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^)]*bool\s+probe_mode\s*=\s*false\s*\)",
                     GGML_SYCL_CPP_CODE), "ggml_sycl_check_nonfa_attn_scratch() must take a probe_mode parameter"


def test_refusal_macro_has_a_mutation_witness():
    """Mutation witness for the macro-usage check above: proves it would
    actually catch the ring function's refusal macro call reverting to a
    bare GGML_LOG_ERROR (which would print on every probe refusal, not just
    every publish refusal)."""
    raw = GGML_SYCL_CPP
    macro_call = (
        "        GGML_SYCL_RUNTIME_TXN_REFUSAL(\n"
        "            probe_mode,\n"
        "            \"[SYCL-PLAN] runtime context update rejected: PP MoE oneDNN scratch ring re-plan for "
        "n_ubatch=%u could \"\n"
    )
    assert macro_call in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(
        macro_call, macro_call.replace("GGML_SYCL_RUNTIME_TXN_REFUSAL(\n            probe_mode,", "GGML_LOG_ERROR("), 1
    )
    assert mutated_raw != raw

    macro_call_re = re.compile(r"GGML_SYCL_RUNTIME_TXN_REFUSAL\(\s*probe_mode\s*,")
    original_ring_fn_body_norm = _body_of(raw, _REPLAN_START, _TRANSACTION_START)
    original_count = len(macro_call_re.findall(original_ring_fn_body_norm))
    assert original_count == 3, (
        "expected exactly three macro uses in the unmutated ring function (the slot-sizing-overflow refusal, "
        f"the release-failure refusal, and refuse_and_restore()'s own refusal) -- found {original_count}; "
        "update this witness to match the real source"
    )

    mutated_ring_fn_body_norm = _body_of(mutated_raw, _REPLAN_START, _TRANSACTION_START)
    matches = macro_call_re.findall(mutated_ring_fn_body_norm)
    assert len(matches) == original_count - 1, (
        "mutation witness is broken: reverting one call site to a bare GGML_LOG_ERROR must drop the count by "
        f"exactly one -- found {len(matches)} (expected {original_count - 1})"
    )


def test_probe_refuses_when_model_is_not_currently_published():
    """The probe must check the given model token against the currently
    published plan's identity BEFORE calling the shared transaction body --
    unlike ggml_backend_sycl_set_runtime_context_for_model(), it does not
    itself select or publish a different model's plan."""
    body_norm = _normalize_ws(_probe_body())
    assert re.search(
        r"current->model_id\s*!=\s*model\.model_id\s*\|\|\s*current->load_txn_id\s*!=\s*model\.load_txn_id",
        body_norm,
    ), "the probe must compare the given model token against the currently published plan's identity"
    assert "GGML_SYCL_LIFECYCLE_STALE_IDENTITY" in body_norm, (
        "a model-identity mismatch must refuse with GGML_SYCL_LIFECYCLE_STALE_IDENTITY"
    )

    identity_check_idx = body_norm.find("current->model_id != model.model_id")
    transaction_call_idx = body_norm.find("ggml_sycl_run_runtime_context_transaction(")
    assert identity_check_idx != -1 and transaction_call_idx != -1
    assert identity_check_idx < transaction_call_idx, (
        "the identity check must run BEFORE the shared transaction body is ever called"
    )


def test_probe_does_not_publish_or_select_a_different_model_plan():
    """The probe must not call the plan-selection/publication primitives
    ggml_backend_sycl_set_runtime_context_for_model() itself uses
    (lifecycle_select_placement_plan, ggml_sycl_publish_plan_locked) -- it
    only evaluates candidates against whichever plan is already current."""
    body_norm = _normalize_ws(_probe_body())
    assert "lifecycle_select_placement_plan(" not in body_norm, (
        "the probe must not select a different model's plan -- it evaluates against the currently published one"
    )
    assert "ggml_sycl_publish_plan_locked(" not in body_norm, (
        "the probe must not itself publish a plan snapshot as current"
    )


def test_probe_out_is_never_null_and_zero_initialized():
    """`out` must not be NULL (refused with GGML_SYCL_LIFECYCLE_NULL_OUTPUT
    when it is) and must be zero-initialized before any field is filled, so
    a caller cannot read stale/uninitialized fields on an early refusal
    that does not reach every fill site."""
    body_norm = _normalize_ws(_probe_body())
    null_check_match = re.search(r"if\s*\(\s*!\s*out\s*\)\s*\{\s*return\s+GGML_SYCL_LIFECYCLE_NULL_OUTPUT\s*;",
                                 body_norm)
    assert null_check_match is not None, (
        "the probe must refuse with GGML_SYCL_LIFECYCLE_NULL_OUTPUT when out is NULL"
    )
    zero_init_match = re.search(r"\*out\s*=\s*ggml_sycl_runtime_context_probe\s*\{\s*\}\s*;", body_norm)
    assert zero_init_match is not None, "the probe must zero-initialize *out before any field is filled"
    assert null_check_match.start() < zero_init_match.start(), (
        "the NULL check must precede the zero-init (dereferencing a NULL out to zero-init it would be the "
        "exact bug the NULL check exists to prevent)"
    )
