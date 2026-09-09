"""Source contract for llama.cpp-oyfl: the runtime-context-update guard that
refuses a context whose non-flash-attention batched mul_mat scratch demand
exceeds the SCRATCH zone the arena already reserved, instead of letting
ggml_sycl_mul_mat_batched_sycl() abort mid-prefill (the "batched F16 mul_mat
failed -- no recovery path available" GGML_ABORT in ggml-sycl.cpp).

This predicate is EMPIRICAL, not a modeled worst case: an earlier revision
attempted to model everything that can spill outside the fixed SYCL arena
(compute-buffer regrowth, SCRATCH overflow, oneDNN scratchpad fragmentation)
and compare it against live free VRAM. Hardware measurement falsified that
model -- a non-FA prefill was measured consuming several GB more outside the
arena than any term the model accounted for, on both discrete cards, and no
zone size or headroom override closed the gap (llama.cpp-k1ev, filed for that
unexplained consumer). Until k1ev is closed, "does the demand exceed the
SCRATCH zone" is the check with real hardware support, so that is what this
file gates -- do not resurrect the live-free-VRAM predicate without new
evidence that k1ev's consumer is understood and bounded.

Host-only, pure text assertions -- no SYCL device required, matching
test-sycl-onednn-graph-allocator-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Checks run against COMMENT-STRIPPED text (see strip_comments()) so a
positive structural check cannot be fooled by prose that quotes a call the
code does not actually make -- a comment can describe a call site
accurately without the code actually containing it, and a substring check
against raw (comment-bearing) text cannot tell the difference; stripping
comments first closes that gap. test-sycl-onednn-graph-allocator-source.py
applies the same hardening for the same reason.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL_H = (ROOT / "ggml/include/ggml-sycl.h").read_text()
CACHE_HPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
CACHE_CPP = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
LLAMA_CONTEXT_CPP = (ROOT / "src/llama-context.cpp").read_text()
# llama.cpp-oyfl: plain file I/O, not the codescout index -- CLAUDE.md
# documents that index (and search_text's live scan) as blind/oversized for
# this specific ~60k-line file, so a tool-assisted search here would
# silently miss real occurrences.
GGML_SYCL_CPP = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()


# Single left-to-right alternation, not two sequential passes -- see
# test-sycl-onednn-graph-allocator-source.py's comment on why a block-comment
# pass run first can swallow a `/*` that appears inside a `//` comment's own
# prose (e.g. a glob like `weight-reclaim/*`). Order matters here for the
# same reason even though none of these four files are known to hit it today.
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
CACHE_HPP_CODE = strip_comments(CACHE_HPP)
CACHE_CPP_CODE = strip_comments(CACHE_CPP)
LLAMA_CONTEXT_CPP_CODE = strip_comments(LLAMA_CONTEXT_CPP)
GGML_SYCL_CPP_CODE = strip_comments(GGML_SYCL_CPP)


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space, so a call or
    declaration re-wrapped by clang-format across lines still matches a
    single-line pattern."""
    return re.sub(r"\s+", " ", text)


def test_api_carries_flash_attn_enabled():
    """Both public entry points must declare the new parameter -- without it
    ggml_backend_sycl_set_runtime_context() has no way to know whether flash
    attention is on for this context, and the guard below cannot be gated on
    the real per-context state."""
    header_norm = _normalize_ws(GGML_SYCL_H_CODE)
    assert "ggml_backend_sycl_set_runtime_context(ggml_backend_t backend" in header_norm
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context\(ggml_backend_t backend,"
        r".*?bool\s+flash_attn_enabled\);",
        header_norm,
    ), "ggml_backend_sycl_set_runtime_context() declaration must carry a bool flash_attn_enabled parameter"
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context_for_model\(.*?bool\s+flash_attn_enabled\);",
        header_norm,
    ), "ggml_backend_sycl_set_runtime_context_for_model() declaration must carry a bool flash_attn_enabled parameter"


def test_llama_context_threads_real_flash_attn_state():
    """llama-context.cpp must pass the RESOLVED cparams.flash_attn (not the
    raw llama_flash_attn_type, and not a hardcoded true/false) into the
    runtime-context call -- otherwise the guard below is gated on nothing
    real for the one caller that matters."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    matches = re.findall(
        r"runtime_context_fn\([^;]*?cparams\.n_seq_max,\s*cparams\.flash_attn\)", ctx_norm
    )
    assert len(matches) >= 2, (
        "expected at least two runtime_context_fn(...) call sites in llama-context.cpp "
        "(the initial call and the BUSY-retry loop) to pass cparams.flash_attn -- found "
        f"{len(matches)}"
    )


def test_guard_consults_the_demand_formula():
    """ggml_sycl_check_nonfa_attn_scratch() (ggml-sycl.cpp) -- the shared
    helper both ggml_backend_sycl_set_runtime_context() and the narrow
    ggml_backend_sycl_recheck_runtime_context_flash_attn() re-check funnel
    into -- must actually call the exported demand formula and gate it on
    flash_attn_enabled -- declaring the parameter without consulting it
    would be a guard that never fires."""
    func_start = GGML_SYCL_CPP_CODE.find("static bool ggml_sycl_check_nonfa_attn_scratch(")
    assert func_start != -1, "ggml_sycl_check_nonfa_attn_scratch() definition not found in ggml-sycl.cpp"
    # Bound the search to this function's body: from the definition to the
    # next top-level function definition after it
    # (ggml_backend_sycl_set_runtime_context, the first of its two callers),
    # so a match cannot come from some unrelated later call site.
    next_func = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_context(", func_start + 1
    )
    assert next_func != -1, "could not bound ggml_sycl_check_nonfa_attn_scratch()'s body"
    body = GGML_SYCL_CPP_CODE[func_start:next_func]
    body_norm = _normalize_ws(body)

    assert "if (flash_attn_enabled) { return true; }" in body_norm, (
        "ggml_sycl_check_nonfa_attn_scratch() must gate the non-FA scratch guard on "
        "flash_attn_enabled being false"
    )
    assert "unified_cache_nonfa_attn_scratch_demand_bytes(" in body_norm, (
        "ggml_sycl_check_nonfa_attn_scratch() must call "
        "unified_cache_nonfa_attn_scratch_demand_bytes() to size the guard"
    )
    assert "zone_capacity(" in body_norm and "vram_zone_id::SCRATCH" in body_norm, (
        "the guard must compare the demand against the SCRATCH zone's actual reserved capacity "
        "(cache->zone_capacity(vram_zone_id::SCRATCH)), not a re-derived or assumed size"
    )
    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in body_norm, (
        "a refusal must report the largest fitting n_ctx (same style as the KV budget refusal's "
        "ggml_sycl_largest_fitting_n_ctx()), not just an error with no remediation"
    )
    assert "runtime context update rejected" in body_norm, (
        "this guard's own refusals must say \"runtime context update rejected\", matching the "
        "pre-existing KV budget refusal's \"runtime KV update rejected\" family of wording (both are "
        "runtime-context-update refusals, just for different checks)"
    )
    assert "GGML_SYCL_NONFA_ATTN_SCRATCH_MB" not in body_norm, (
        "the runtime refusal must NOT advertise GGML_SYCL_NONFA_ATTN_SCRATCH_MB as a remediation -- "
        "a hardware sweep (768/1536 MiB, raised headroom to 3-4 GB) still aborted on both cards "
        "(llama.cpp-k1ev); -fa 1/auto or a smaller -c are the only remediations with hardware support"
    )
    assert "scratch-limited" in body_norm, (
        "the largest-fitting-n_ctx remediation must be labeled \"scratch-limited\" -- it is the "
        "SCRATCH zone's own capacity limit, not a whole-device guarantee (llama.cpp-k1ev's "
        "unexplained outside-arena consumer is not reflected in this figure)"
    )
    assert "unified_cache_ensure_planned_arena_zones(" in body_norm, (
        "the guard must make the opportunistic re-plan attempt (same call reserve_onednn_scratch() "
        "already documents as succeeding only while the arena is still unused) before checking"
    )
    assert "unified_cache_set_planned_nonfa_attn_scratch_shape(" in body_norm, (
        "the opportunistic re-plan must record the REAL runtime shape first, not the load-time one"
    )
    assert "scratch_capacity > scratch_capacity_before" in body_norm, (
        "the re-plan's own log line must be gated on the zone actually having grown -- logging "
        "\"raised\" when unified_cache_ensure_planned_arena_zones() could not rebuild (live weight "
        "leases) would misreport a no-op as a success"
    )


def test_both_callers_wire_into_the_shared_guard():
    """Both ggml_backend_sycl_set_runtime_context() (the full transaction)
    and ggml_backend_sycl_recheck_runtime_context_flash_attn() (the narrow
    re-check) must actually call the shared ggml_sycl_check_nonfa_attn_scratch()
    helper -- a helper that exists and is correct but is never called by one
    of its two intended entry points would leave that path's contexts
    unguarded."""
    full_start = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_context(")
    assert full_start != -1, "ggml_backend_sycl_set_runtime_context() definition not found"
    full_next = GGML_SYCL_CPP_CODE.find(
        "ggml_backend_sycl_set_runtime_context_for_model(", full_start + 1
    )
    assert full_next != -1, "could not bound ggml_backend_sycl_set_runtime_context()'s body"
    full_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[full_start:full_next])
    assert "ggml_sycl_check_nonfa_attn_scratch(" in full_body_norm, (
        "ggml_backend_sycl_set_runtime_context() must call the shared guard helper"
    )

    recheck_start = GGML_SYCL_CPP_CODE.find(
        "ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn("
    )
    assert recheck_start != -1, "ggml_backend_sycl_recheck_runtime_context_flash_attn() definition not found"
    recheck_next = GGML_SYCL_CPP_CODE.find(
        "void ggml_backend_sycl_set_runtime_n_ctx(", recheck_start + 1
    )
    assert recheck_next != -1, "could not bound ggml_backend_sycl_recheck_runtime_context_flash_attn()'s body"
    recheck_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[recheck_start:recheck_next])
    assert "ggml_sycl_check_nonfa_attn_scratch(" in recheck_body_norm, (
        "ggml_backend_sycl_recheck_runtime_context_flash_attn() must call the shared guard helper -- "
        "this is the entry point resolve_fused_ops() uses for an AUTO context resolving to non-FA, "
        "so a missing call here would leave that specific context unguarded"
    )

    # The header declaration for the narrow re-check entry point must also
    # exist, so llama-context.cpp compiles against a real exported symbol.
    header_norm = _normalize_ws(GGML_SYCL_H_CODE)
    assert (
        "ggml_backend_sycl_recheck_runtime_context_flash_attn(" in header_norm
    ), "ggml_backend_sycl_recheck_runtime_context_flash_attn() must be declared in ggml-sycl.h"


def test_callers_pass_the_all_layers_head_count():
    """llama.cpp-rqak: both guard call sites -- the full transaction's
    next_plan and the narrow re-check's current->plan -- must pass the
    ALL-LAYERS head-count field into ggml_sycl_check_nonfa_attn_scratch().
    The non-FA attention path runs on EVERY attention layer, so its demand
    model (max(16 MiB, n_head x n_ubatch x n_ctx x 2 B x 3)) needs the
    maximum query-head count over all layers -- not llama.cpp-o3a0's
    n_head_ctx_max/n_head_swa_max, which are maxima over oneDNN-ELIGIBLE
    layers only and both read 0 for a model where every layer is
    ineligible (e.g. a DeepSeek-V3-class model, D=576), which would push
    this guard into its "could not evaluate" WARN path and leave that
    exact shape unguarded. And never the pre-o3a0 planner_n_head name,
    which o3a0 removed -- git's clean auto-merge of o3a0 (per-class
    maxima) with llama.cpp-oyfl (this guard, developed against the
    pre-o3a0 single field) left the guard referencing a member that no
    longer exists."""
    full_start = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_context(")
    assert full_start != -1
    full_next = GGML_SYCL_CPP_CODE.find(
        "ggml_backend_sycl_set_runtime_context_for_model(", full_start + 1
    )
    assert full_next != -1
    full_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[full_start:full_next])

    recheck_start = GGML_SYCL_CPP_CODE.find(
        "ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn("
    )
    assert recheck_start != -1
    recheck_next = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_n_ctx(", recheck_start + 1)
    assert recheck_next != -1
    recheck_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[recheck_start:recheck_next])

    for caller_name, body in (
        ("the full transaction (ggml_backend_sycl_set_runtime_context)", full_body_norm),
        ("the narrow re-check (ggml_backend_sycl_recheck_runtime_context_flash_attn)", recheck_body_norm),
    ):
        assert "planner_n_head_all" in body, (
            f"{caller_name} must pass the all-layers head-count field (planner_n_head_all) into "
            "ggml_sycl_check_nonfa_attn_scratch() -- see llama.cpp-rqak"
        )
        # \b requires a non-word char on both sides, and "_" is a word char,
        # so this cannot match inside planner_n_head_all/planner_n_head_ctx_max/
        # planner_n_head_swa_max -- it only matches the bare, pre-o3a0 name.
        assert not re.search(r"\bplanner_n_head\b", body), (
            f"{caller_name} must not reference the removed planner_n_head field (llama.cpp-o3a0 replaced "
            "it with the per-class n_head_ctx_max/n_head_swa_max maxima; this guard needs its own "
            "all-layers field, planner_n_head_all, not either of those)"
        )
        assert "n_head_ctx_max" not in body and "n_head_swa_max" not in body, (
            f"{caller_name} must not pass the o3a0 per-class oneDNN-eligible-only maxima "
            "(n_head_ctx_max / n_head_swa_max) into the non-FA attention guard -- those exclude "
            "oneDNN-ineligible layers and can both be 0 for a model where every layer is ineligible, "
            "which would silently leave that model's non-FA scratch demand unguarded"
        )


def test_narrow_recheck_forbids_replan_and_takes_the_lock():
    """The narrow re-check must call the shared guard with allow_replan=false
    (never recording, re-planning, or restoring the plan-time SCRATCH-zone
    shape it does not own), and must take the same module-admission guard
    and tensor-inventory lock the full transaction serializes its own
    mutating work under, confirming under that lock that the plan snapshot
    read before the lock is still the live one."""
    full_start = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_context(")
    assert full_start != -1
    full_next = GGML_SYCL_CPP_CODE.find("ggml_backend_sycl_set_runtime_context_for_model(", full_start + 1)
    assert full_next != -1
    full_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[full_start:full_next])
    # allow_replan is passed positionally as the call's 6th (final) argument
    # -- matched by position, not by the inline `/*allow_replan=*/` comment
    # naming it, because that comment is exactly the kind of text
    # strip_comments() (deliberately) removes before this check ever sees
    # the source.
    assert re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^()]*flash_attn_enabled,\s*true\)", full_body_norm), (
        "the full transaction must call the shared guard with allow_replan=true (the final positional "
        "argument, after flash_attn_enabled)"
    )

    recheck_start = GGML_SYCL_CPP_CODE.find(
        "ggml_sycl_lifecycle_result ggml_backend_sycl_recheck_runtime_context_flash_attn("
    )
    assert recheck_start != -1
    recheck_next = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_n_ctx(", recheck_start + 1)
    assert recheck_next != -1
    recheck_body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[recheck_start:recheck_next])

    assert re.search(r"ggml_sycl_check_nonfa_attn_scratch\([^()]*flash_attn_enabled,\s*false\)", recheck_body_norm), (
        "the narrow re-check must call the shared guard with allow_replan=false (the final positional "
        "argument, after flash_attn_enabled) -- it must not record, re-plan, or restore the plan-time "
        "SCRATCH-zone shape"
    )
    assert "sycl_module_mutation_guard module_guard;" in recheck_body_norm, (
        "the narrow re-check must take sycl_module_mutation_guard, the same admission guard the full "
        "transaction's callers rely on, so it cannot run past a module shutdown"
    )
    assert "std::lock_guard<std::mutex> lock(g_tensor_inventory_mutex);" in recheck_body_norm, (
        "the narrow re-check must take g_tensor_inventory_mutex, the same mutex the full transaction "
        "serializes its own mutating work under, before confirming the plan snapshot is still live -- "
        "not because some specific accessor this then reads requires the lock (zone_capacity() is an "
        "unsynchronized array read with no lock contract of its own)"
    )
    assert "ggml_sycl_global_plan_snapshot().get() != current.get()" in recheck_body_norm, (
        "the narrow re-check must confirm, under the lock, that the plan snapshot read before the lock "
        "is still the live one -- a read-then-confirm-under-lock construction of its own, guarding "
        "against acting on a plan a concurrent model load/unload or runtime-context call has already "
        "superseded"
    )


def test_for_model_forwards_flash_attn_enabled():
    """ggml_backend_sycl_set_runtime_context_for_model() must forward its new
    parameter into the inner call rather than dropping it on the floor."""
    func_start = GGML_SYCL_CPP_CODE.find("ggml_sycl_lifecycle_result ggml_backend_sycl_set_runtime_context_for_model(")
    assert func_start != -1, "ggml_backend_sycl_set_runtime_context_for_model() definition not found"
    body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[func_start : func_start + 8000])
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context\(backend, n_ctx, n_ubatch, n_seq_max,\s*flash_attn_enabled\)",
        body_norm,
    ), "ggml_backend_sycl_set_runtime_context_for_model() must forward flash_attn_enabled to the inner call"


def test_formula_and_inverse_are_declared_and_defined():
    """The demand formula and its inverse must be declared (header) and
    defined (implementation) -- both exported (not file-static), unlike the
    oneDNN Graph-scratch floor sibling, because ggml-sycl.cpp (a different
    translation unit) must call them directly."""
    hpp_norm = _normalize_ws(CACHE_HPP_CODE)
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)

    assert "size_t unified_cache_nonfa_attn_scratch_demand_bytes(uint32_t n_head, uint32_t n_ubatch, uint32_t n_ctx);" in hpp_norm
    assert "size_t unified_cache_nonfa_attn_scratch_demand_bytes(uint32_t n_head, uint32_t n_ubatch, uint32_t n_ctx) {" in cpp_norm

    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in hpp_norm
    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in cpp_norm

    # Neither may be declared `static` -- in the .cpp that would make them
    # unreachable from ggml-sycl.cpp, silently turning the guard above into
    # a compile error this text-only test cannot otherwise catch (a build is
    # a stronger check, but this test runs without one); in the .hpp a
    # `static` on the DECLARATION would give each translation unit that
    # includes the header its own internal-linkage copy, which is just as
    # wrong even though it happens to still compile. Checked in both files
    # for both functions (four checks), with a "static ... name(" bridge
    # that does not name a return type at all -- `\bstatic\b[^;{}]*?\bname\(`
    # -- so it is not fooled by a return-type spelling change ("static
    # std::size_t", say) or a storage-class keyword or attribute stacked
    # between `static` and the type ("static inline", "static
    # __attribute__((used)) size_t"); bounded to `[^;{}]` so it cannot
    # cross a statement or scope boundary and match some unrelated earlier
    # `static` against this name's own later, unrelated appearance. Each
    # match is reduced to a bool BEFORE the assert -- asserting directly on
    # a `re.search()` result (or on the huge normalized-source string
    # itself) would make a failing pytest try to render that whole
    # multi-hundred-KB string as part of the diff.
    def _has_static_before(text: str, name: str) -> bool:
        return bool(re.search(r"\bstatic\b[^;{}]*?\b" + re.escape(name) + r"\s*\(", text))

    demand_name  = "unified_cache_nonfa_attn_scratch_demand_bytes"
    inverse_name = "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch"

    assert not _has_static_before(cpp_norm, demand_name), f"{demand_name}() must not be file-static (.cpp)"
    assert not _has_static_before(hpp_norm, demand_name), f"{demand_name}() must not be declared static (.hpp)"
    assert not _has_static_before(cpp_norm, inverse_name), f"{inverse_name}() must not be file-static (.cpp)"
    assert not _has_static_before(hpp_norm, inverse_name), f"{inverse_name}() must not be declared static (.hpp)"


def test_auto_flash_attn_resolution_rechecks_the_guard():
    """cparams.flash_attn defaults true for AUTO before resolve_fused_ops()
    resolves it (llama-context.cpp's cparams init runs before the
    constructor's runtime-context call), so an AUTO context that resolves
    to OFF must re-trigger a SYCL guard re-check with the now-resolved
    value, or the guard above never sees it for that context.

    Two DIFFERENT methods are involved, not one shared between both call
    sites: sycl_resync_runtime_context_flash_attn() (the FULL
    runtime-context transaction: KV replan, MoE MMID reaccount/materialize,
    plan republish) is the constructor's OWN initial call, made before any
    AUTO resolution; sycl_recheck_runtime_context_flash_attn() (a NARROW
    re-check of only the non-FA attention scratch guard) is what
    resolve_fused_ops() calls once AUTO actually resolves. Re-running the
    full transaction from resolve_fused_ops() would needlessly redo KV/MMID
    work that has no reason to change just because flash_attn_enabled did."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    assert "void llama_context::sycl_resync_runtime_context_flash_attn()" in ctx_norm, (
        "expected the constructor's full-transaction helper to still exist"
    )
    assert "void llama_context::sycl_recheck_runtime_context_flash_attn()" in ctx_norm, (
        "expected a separate narrow re-check helper for resolve_fused_ops() to call"
    )

    # The constructor calls the FULL-transaction helper, not the narrow one.
    # Bounded to the destructor that immediately follows it (not all the way
    # to resolve_fused_ops(), which would also swallow the two helper
    # methods' own definitions sitting in between and make this check
    # imprecise about what "the constructor" actually means).
    ctor_start = ctx_norm.find("llama_context::llama_context(")
    assert ctor_start != -1, "llama_context::llama_context() definition not found"
    dtor_start = ctx_norm.find("llama_context::~llama_context(", ctor_start + 1)
    assert dtor_start != -1, "~llama_context() definition not found"
    resolve_start = ctx_norm.find("void llama_context::resolve_fused_ops(")
    assert resolve_start != -1, "resolve_fused_ops() definition not found"
    assert ctor_start < dtor_start < resolve_start, (
        "expected the constructor, then the destructor, then resolve_fused_ops(), in that order"
    )
    ctor_body = ctx_norm[ctor_start:dtor_start]
    assert "sycl_resync_runtime_context_flash_attn();" in ctor_body, (
        "the constructor must call the FULL-transaction sycl_resync_runtime_context_flash_attn() -- "
        "it is where n_ctx/n_ubatch are established for the first time"
    )
    assert "sycl_recheck_runtime_context_flash_attn();" not in ctor_body, (
        "the constructor must NOT call the narrow re-check -- that would skip the KV replan/MMID "
        "reaccount a first-time call needs"
    )

    # resolve_fused_ops() calls the NARROW re-check, inside the same
    # if (cparams.auto_fa) block that resolves flash_attn, so it fires
    # exactly once, right when the real value becomes known -- and must NOT
    # call the full-transaction helper (that would needlessly re-run KV/MMID
    # work resolve_fused_ops() has no reason to touch).
    resolve_body = ctx_norm[resolve_start : resolve_start + 4000]
    assert re.search(r"if \(cparams\.auto_fa\) \{[^}]*resolve\([^;]*flash_attn[^;]*;[^}]*"
                      r"sycl_recheck_runtime_context_flash_attn\(\);", resolve_body), (
        "resolve_fused_ops() must call sycl_recheck_runtime_context_flash_attn() inside the same "
        "if (cparams.auto_fa) block that resolves flash_attn, so it fires exactly once, right when "
        "the real value becomes known"
    )
    assert "sycl_resync_runtime_context_flash_attn();" not in resolve_body, (
        "resolve_fused_ops() must NOT call the full-transaction helper -- only the narrow re-check"
    )

    # Both helpers' own header declarations exist too, so both callers
    # compile against real class members, not undeclared symbols this test
    # alone would not catch.
    ctx_h = (ROOT / "src/llama-context.h").read_text()
    ctx_h_norm = _normalize_ws(strip_comments(ctx_h))
    assert "void sycl_resync_runtime_context_flash_attn();" in ctx_h_norm
    assert "void sycl_recheck_runtime_context_flash_attn();" in ctx_h_norm


_PREPROC_OPEN_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
_PREPROC_ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
_PREPROC_DNNL_IF_RE = re.compile(r"^\s*#\s*if\s+GGML_SYCL_DNNL\b")


def _is_inside_if_dnnl(raw: str, position: int) -> bool:
    """True if `position` falls lexically inside an #if GGML_SYCL_DNNL /
    #endif block, tracking full nesting depth rather than the nearest
    preceding #if/#endif tokens -- a last-match scan gives the wrong answer
    as soon as an unrelated nested #if/#endif (or #ifdef/#ifndef) sits
    between the #if GGML_SYCL_DNNL and the position being checked, because
    the innermost #endif it finds may close that unrelated nested block
    rather than the GGML_SYCL_DNNL one."""
    depth = 0
    dnnl_depth = None
    for line in raw[:position].splitlines():
        stripped = line.strip()
        if _PREPROC_OPEN_RE.match(stripped):
            depth += 1
            if dnnl_depth is None and _PREPROC_DNNL_IF_RE.match(stripped):
                dnnl_depth = depth
        elif _PREPROC_ENDIF_RE.match(stripped):
            if dnnl_depth is not None and depth == dnnl_depth:
                dnnl_depth = None
            depth = max(0, depth - 1)
    return dnnl_depth is not None


def test_plan_time_shape_is_recorded_unconditionally():
    """populate_host_zone_sizing() must record the non-FA shape
    UNCONDITIONALLY (not gated behind #if GGML_SYCL_DNNL like the oneDNN
    sibling) -- the path this sizes for is independent of oneDNN."""
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)
    assert "unified_cache_set_planned_nonfa_attn_scratch_shape(plan.device_id, plan.planner_n_head_all, plan.planner_n_ubatch, plan.planner_n_ctx)" in cpp_norm

    # Comment-stripped text drops "#if GGML_SYCL_DNNL" lines too (they are
    # preprocessor directives, not comments -- re-check against the RAW,
    # comment-bearing source): the call site must not sit inside that guard.
    # Tracked by nesting depth (see _is_inside_if_dnnl above), not by a
    # nearest-preceding-token scan, which can misjudge across an unrelated
    # nested #if/#endif sitting between the guard and the call site.
    raw = CACHE_CPP
    call_idx = raw.find("unified_cache_set_planned_nonfa_attn_scratch_shape(plan.device_id")
    assert call_idx != -1
    assert not _is_inside_if_dnnl(raw, call_idx), (
        "unified_cache_set_planned_nonfa_attn_scratch_shape() call site must not be inside "
        "an #if GGML_SYCL_DNNL block"
    )
