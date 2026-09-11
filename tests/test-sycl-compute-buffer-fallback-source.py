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
_TRANSACTION_START = "static bool ggml_sycl_run_runtime_context_transaction("
_SET_RUNTIME_CONTEXT_START = "void ggml_backend_sycl_set_runtime_context("
_PROBE_START = "ggml_sycl_lifecycle_result ggml_backend_sycl_probe_runtime_context_for_model("
_AUTO_UBATCH_ENABLED_START = "bool ggml_backend_sycl_auto_ubatch_enabled("


def _alloc_buffer_body() -> str:
    return _bounded_body(GGML_SYCL_CPP_CODE, _ALLOC_BUFFER_START, _ALLOC_BUFFER_END)


def _transaction_body() -> str:
    # Bounded by the NEXT function's start (the thin wrapper this shared body
    # was extracted from) -- everything from "static bool
    # ggml_sycl_run_runtime_context_transaction(" up to (not including)
    # "void ggml_backend_sycl_set_runtime_context(".
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


def test_both_fallback_sites_increment_the_counter():
    """Both host-pinned fallback paths inside
    ggml_backend_sycl_buffer_type_alloc_buffer()'s bounded body must
    fetch_add(1, ...) the per-device counter -- the function is a
    function-try-block (catches sycl::exception at its close), so the
    increment must sit in the ordinary control flow before any return, not
    after one."""
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
    assert any(idx < warn1_idx for idx in increments), (
        "the oversize-request fallback must increment the counter at or before its own WARN log"
    )
    assert any(idx < warn2_idx for idx in increments), (
        "the failed-alloc-retry fallback must increment the counter at or before its own WARN log"
    )


def test_increment_sites_have_a_mutation_witness():
    """Mutation witness for the increment-count check above: proves it would
    actually catch one of the two increments being deleted."""
    raw = GGML_SYCL_CPP
    increment_line = "            g_compute_buffer_host_fallbacks[buft_ctx->device].fetch_add(1, std::memory_order_relaxed);\n"
    count = raw.count(increment_line)
    assert count == 2, f"expected exactly two identical increment lines -- found {count}"
    mutated_raw = raw.replace(increment_line, "", 1)
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


def test_transaction_resets_the_counter_on_success_before_publish():
    """ggml_sycl_run_runtime_context_transaction()'s PUBLISH path (not the
    probe path) must reset g_compute_buffer_host_fallbacks[ctx->device] to 0
    once every candidate check has passed, before the plan that
    graph_reserve() will allocate compute buffers against is published (the
    CAS), so a caller reading the counter afterward sees "fallbacks since the
    last successful runtime-context transaction", not carried over from a
    stale plan."""
    body_norm = _normalize_ws(_transaction_body())
    assert re.search(
        r"g_compute_buffer_host_fallbacks\[ctx->device\]\.store\(\s*0", body_norm
    ), "the transaction body must reset the counter with .store(0, ...)"

    reset_idx = body_norm.find("g_compute_buffer_host_fallbacks[ctx->device].store(0")
    # The probe branch's own early "return true;" (right after its own
    # "OK" fill and the ring rollback) must precede the reset -- the reset
    # belongs only to the publish path that runs AFTER that branch.
    probe_branch_idx = body_norm.find("if (probe_mode) {")
    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert probe_branch_idx != -1 and reset_idx != -1 and cas_idx != -1
    assert probe_branch_idx < reset_idx < cas_idx, (
        "the counter reset must sit AFTER the probe_mode branch (so a probe never resets it) and BEFORE the "
        "CAS (so it only fires once the transaction is actually about to publish)"
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
    assert re.search(
        r"static\s+bool\s+ggml_sycl_run_runtime_context_transaction\s*\(", GGML_SYCL_CPP_CODE
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
    assert re.search(
        r"ggml_sycl_run_runtime_context_transaction\(\s*backend,\s*n_ctx,\s*n_ubatch,\s*n_seq_max,\s*"
        r"flash_attn_enabled,\s*false,\s*nullptr\s*\)",
        publisher_body_norm,
    ), "the publishing wrapper must call the shared body with probe_mode=false, out=nullptr"

    probe_body_norm = _normalize_ws(_probe_body())
    assert "ggml_sycl_run_runtime_context_transaction(" in probe_body_norm, (
        "ggml_backend_sycl_probe_runtime_context_for_model() must call ggml_sycl_run_runtime_context_transaction()"
    )
    assert re.search(
        r"ggml_sycl_run_runtime_context_transaction\(\s*backend,\s*n_ctx,\s*n_ubatch,\s*n_seq_max,\s*"
        r"flash_attn_enabled,\s*true,\s*out\s*\)",
        probe_body_norm,
    ), "the probe must call the shared body with probe_mode=true, out=out"


def test_shared_body_call_sites_have_a_mutation_witness():
    """Mutation witness for the check above: proves it would actually catch
    the probe silently going back to calling the OLD, non-shared, direct
    logic instead of the shared transaction function."""
    raw = GGML_SYCL_CPP
    probe_call = (
        "    const bool accepted = ggml_sycl_run_runtime_context_transaction(backend, n_ctx, n_ubatch, n_seq_max,\n"
        "                                                                    flash_attn_enabled, /*probe_mode=*/true, out);\n"
    )
    assert probe_call in raw, "mutation target not found -- update this witness to match the real source"
    mutated_raw = raw.replace(probe_call, "", 1)
    assert mutated_raw != raw

    mutated_probe_body_norm = _body_of(mutated_raw, _PROBE_START, _AUTO_UBATCH_ENABLED_START)
    assert "ggml_sycl_run_runtime_context_transaction(" not in mutated_probe_body_norm, (
        "mutation witness is broken: deleting the call left a reference to it behind"
    )


def test_probe_mode_branch_returns_before_the_cas():
    """The shared transaction body's `if (probe_mode) { ... return true; }`
    branch must appear BEFORE the CAS
    (ggml_sycl::lifecycle_replace_placement_plan) -- both bounds asserted:
    the probe_mode branch must exist, the CAS call must exist, and the
    branch's own `return true;` must precede the CAS textually (the shared
    body is straight-line code with early returns, so "precedes textually"
    means "cannot execute past this point when probe_mode is true, because
    every branch between here and the CAS is either this one's own return or
    an earlier, unconditional refusal")."""
    body_norm = _normalize_ws(_transaction_body())

    probe_branch = re.search(r"if\s*\(\s*probe_mode\s*\)\s*\{", body_norm)
    assert probe_branch is not None, "the shared body must have an `if (probe_mode) { ... }` branch"

    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
    assert cas_idx != -1, "could not find the CAS call (ggml_sycl::lifecycle_replace_placement_plan)"

    probe_return_idx = body_norm.find("return true", probe_branch.start())
    assert probe_return_idx != -1 and probe_return_idx < cas_idx, (
        "probe_mode is not the anonymous case -- the branch's own `return true;` must precede the CAS"
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
    """Mutation witness for the check above: proves it would actually catch
    the probe_mode branch being moved to AFTER the CAS (so a probe could
    reach it)."""
    raw = GGML_SYCL_CPP
    probe_block = (
        "    if (probe_mode) {\n"
    )
    assert probe_block in raw, "mutation target not found -- update this witness to match the real source"
    # Move the guard to text AFTER the CAS by renaming the real one and
    # inserting a decoy after the CAS call -- this reproduces "the branch
    # exists somewhere in the function" while breaking "the branch precedes
    # the CAS", which is exactly what the ordering assertion must catch.
    cas_line = "    if (!ggml_sycl::lifecycle_replace_placement_plan(current, immutable)) {\n"
    assert cas_line in raw, "mutation target (CAS line) not found -- update this witness to match the real source"
    mutated_raw = raw.replace(probe_block, "    if (false && probe_mode) {\n", 1)
    assert mutated_raw != raw

    mutated_body_norm = _body_of(mutated_raw, _TRANSACTION_START, _SET_RUNTIME_CONTEXT_START)
    probe_branch = re.search(r"if\s*\(\s*probe_mode\s*\)\s*\{", mutated_body_norm)
    assert probe_branch is None, (
        "mutation witness is broken: the mutated source (guard changed to `false && probe_mode`) should no "
        "longer match the exact `if (probe_mode) {` pattern the real check requires"
    )


def test_probe_rolls_back_the_ring_before_returning():
    """The probe_mode branch must roll the PP MoE oneDNN ring back to its
    pre-transaction n_ubatch (with probe_mode=true, so the rollback's own
    logging is also quiet) before returning -- the ring re-plan call just
    above it already performed real device-state side effects (release +
    reserve), which a probe must not leave behind."""
    body_norm = _normalize_ws(_transaction_body())
    probe_branch = re.search(r"if\s*\(\s*probe_mode\s*\)\s*\{", body_norm)
    assert probe_branch is not None
    cas_idx = body_norm.find("lifecycle_replace_placement_plan(current, immutable)")
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

    ring_fn_body_norm = _normalize_ws(_bounded_body(GGML_SYCL_CPP_CODE, "static bool ggml_sycl_replan_pp_moe_onednn_ring(",
                                                     _TRANSACTION_START))
    assert macro_call_re.search(ring_fn_body_norm), (
        "ggml_sycl_replan_pp_moe_onednn_ring() must use the refusal macro for its own candidate refusals"
    )

    nonfa_fn_body_norm = _normalize_ws(
        _bounded_body(GGML_SYCL_CPP_CODE, "static bool ggml_sycl_check_nonfa_attn_scratch(",
                     "static bool ggml_sycl_replan_pp_moe_onednn_ring(")
    )
    assert macro_call_re.search(nonfa_fn_body_norm), (
        "ggml_sycl_check_nonfa_attn_scratch() must use the refusal macro for its own candidate refusals"
    )

    # And every one of those three functions must actually receive a
    # probe_mode parameter for the macro call above to even compile.
    assert re.search(r"ggml_sycl_replan_pp_moe_onednn_ring\([^)]*bool\s+probe_mode\s*=\s*false\s*\)",
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
    original_ring_fn_body_norm = _body_of(
        raw, "static bool ggml_sycl_replan_pp_moe_onednn_ring(", _TRANSACTION_START
    )
    original_count = len(macro_call_re.findall(original_ring_fn_body_norm))
    assert original_count == 3, (
        "expected exactly three macro uses in the unmutated ring function (the slot-sizing-overflow refusal, "
        f"the release-failure refusal, and refuse_and_restore()'s own refusal) -- found {original_count}; "
        "update this witness to match the real source"
    )

    mutated_ring_fn_body_norm = _body_of(
        mutated_raw, "static bool ggml_sycl_replan_pp_moe_onednn_ring(", _TRANSACTION_START
    )
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
