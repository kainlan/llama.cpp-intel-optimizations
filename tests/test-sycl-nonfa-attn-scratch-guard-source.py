"""Source contract for llama.cpp-oyfl: the runtime-context-update guard that
refuses a context whose non-flash-attention batched mul_mat path would
exceed the VRAM available outside the SYCL backend's fixed arena, instead of
letting ggml_sycl_mul_mat_batched_sycl() abort mid-prefill (the "batched F16
mul_mat failed -- no recovery path available" GGML_ABORT in ggml-sycl.cpp).
A SCRATCH-zone-capacity check alone is not sufficient -- a zone overflow
spills to a raw device allocation OUTSIDE the arena instead of failing
outright (unified-cache.cpp's unified_alloc()), and the SYCL scheduler's own
compute buffer draws on that same outside-arena headroom when it regrows to
a real ubatch's actual n_kv -- so the guard compares the sum of everything
that can spill outside the arena (compute-buffer regrowth, SCRATCH overflow,
oneDNN scratchpad) against the VRAM actually available outside it, queried
live.

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
    """Both public entry points must declare flash_attn_enabled AND
    reserved_compute_buffer_bytes -- without the first, the guard cannot
    know whether flash attention is on for this context; without the
    second, it cannot know the SYCL scheduler's actual compute-buffer size,
    which the authoritative (post-sched_reserve) call needs to predict
    regrowth."""
    header_norm = _normalize_ws(GGML_SYCL_H_CODE)
    assert "ggml_backend_sycl_set_runtime_context(ggml_backend_t backend" in header_norm
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context\(ggml_backend_t backend,"
        r".*?bool\s+flash_attn_enabled,\s*size_t\s+reserved_compute_buffer_bytes\);",
        header_norm,
    ), (
        "ggml_backend_sycl_set_runtime_context() declaration must carry bool flash_attn_enabled "
        "followed by size_t reserved_compute_buffer_bytes"
    )
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context_for_model\(.*?bool\s+flash_attn_enabled,\s*"
        r"size_t\s+reserved_compute_buffer_bytes\);",
        header_norm,
    ), (
        "ggml_backend_sycl_set_runtime_context_for_model() declaration must carry bool "
        "flash_attn_enabled followed by size_t reserved_compute_buffer_bytes"
    )


def test_llama_context_threads_real_flash_attn_state():
    """llama-context.cpp must pass the RESOLVED cparams.flash_attn (not the
    raw llama_flash_attn_type, and not a hardcoded true/false) into the
    runtime-context call -- otherwise the guard below is gated on nothing
    real for the one caller that matters."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    matches = re.findall(
        r"runtime_context_fn\([^;]*?cparams\.n_seq_max,\s*cparams\.flash_attn,\s*reserved_compute_buffer_bytes\)",
        ctx_norm,
    )
    assert len(matches) >= 2, (
        "expected at least two runtime_context_fn(...) call sites in llama-context.cpp "
        "(the initial call and the BUSY-retry loop) to pass cparams.flash_attn -- found "
        f"{len(matches)}"
    )


def test_guard_consults_the_demand_formula():
    """ggml_backend_sycl_set_runtime_context() (ggml-sycl.cpp) must actually
    call the exported demand formula and gate it on flash_attn_enabled --
    declaring the parameter without consulting it would be a guard that
    never fires."""
    func_start = GGML_SYCL_CPP_CODE.find("void ggml_backend_sycl_set_runtime_context(")
    assert func_start != -1, "ggml_backend_sycl_set_runtime_context() definition not found in ggml-sycl.cpp"
    # Bound the search to this function's body: from the definition to the
    # next top-level function definition after it
    # (ggml_backend_sycl_set_runtime_context_for_model), so a match cannot
    # come from some unrelated later call site.
    next_func = GGML_SYCL_CPP_CODE.find(
        "ggml_backend_sycl_set_runtime_context_for_model(", func_start + 1
    )
    assert next_func != -1, "could not bound ggml_backend_sycl_set_runtime_context()'s body"
    body = GGML_SYCL_CPP_CODE[func_start:next_func]
    body_norm = _normalize_ws(body)

    assert "!flash_attn_enabled" in body_norm, (
        "ggml_backend_sycl_set_runtime_context() must gate the non-FA scratch guard on "
        "flash_attn_enabled being false"
    )
    assert "unified_cache_nonfa_attn_scratch_demand_bytes(" in body_norm, (
        "ggml_backend_sycl_set_runtime_context() must call "
        "unified_cache_nonfa_attn_scratch_demand_bytes() to size the guard"
    )
    assert "zone_capacity(" in body_norm and "vram_zone_id::SCRATCH" in body_norm, (
        "the guard must read the SCRATCH zone's actual reserved capacity "
        "(cache->zone_capacity(vram_zone_id::SCRATCH)) to compute the scratch-overflow term, "
        "not a re-derived or assumed size"
    )
    assert "unified_cache_largest_fitting_n_ctx_for_nonfa_attn_scratch(" in body_norm, (
        "a refusal must report the largest fitting n_ctx (same style as the KV budget refusal's "
        "ggml_sycl_largest_fitting_n_ctx()), not just an error with no remediation"
    )
    assert "runtime context rejected" in body_norm, (
        "this guard's own refusals must say \"runtime context rejected\", not reuse the pre-existing "
        "KV budget refusal's \"runtime KV update rejected\" wording -- they are different checks"
    )
    assert "GGML_SYCL_NONFA_ATTN_SCRATCH_MB" in body_norm, (
        "the SCRATCH-zone refusal must name the env var and the exact MB value that would let this "
        "shape proceed, since that is the only automatic-at-load-time remediation available "
        "(llama.cpp-fkpg gates piping the real n_ctx into pre-load sizing)"
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

    # The outside-arena predicate: a zone-capacity-only comparison is not
    # sufficient because overflow spills outside the fixed arena instead of
    # failing outright, and the SYCL scheduler's own compute buffer draws on
    # that same headroom when it regrows to the real n_kv.
    assert "ggml_backend_sycl_get_device_memory(" in body_norm, (
        "the guard must query live free VRAM (not a static planned budget) to know how much "
        "headroom exists outside the fixed arena"
    )
    assert "reserved_compute_buffer_bytes" in body_norm, (
        "the guard must fold in the SYCL scheduler's actual compute-buffer size, forwarded from "
        "the new API parameter"
    )
    assert "unified_cache_get_planned_onednn_scratchpad_bytes(" in body_norm, (
        "the guard must fold in the oneDNN scratchpad's planned estimate -- its own failure mode "
        "in the repro was fragmentation, which a capacity-only comparison cannot see"
    )
    assert re.search(r"margin\s*=\s*std::max<uint64_t>\(", body_norm), (
        "the guard must apply a calibrated safety margin (documented as max(256 MiB, 10% of "
        "live_free), covering driver/runtime allocations neither modeled term accounts for)"
    )
    assert "total_outside > available" in body_norm, (
        "the primary check must compare the summed outside-arena demand against the summed "
        "available headroom, not compare demand against zone capacity alone"
    )

    # The complementary whole-plan check: weights+KV (already tracked in
    # next_plan.vram_bytes) plus this consumer's own transient footprint,
    # against next_plan.vram_budget.
    assert "vram_bytes" in body_norm and "vram_budget" in body_norm, (
        "the complementary whole-plan check must compare against next_plan.vram_bytes/vram_budget, "
        "the same totals the KV budget refusal already tracks"
    )
    assert "sizeof(float)" in body_norm, (
        "the whole-plan check must add the scheduler-owned KQ/KQV f32 buffer's bytes "
        "(n_head * n_ctx * n_ubatch * sizeof(float))"
    )
    assert "ggml_sycl_largest_fitting_n_ctx(next_plan, next_kv_info)" in body_norm, (
        "the whole-plan check's remediation must include the KV-based largest-fitting figure, "
        "not only the SCRATCH-zone-based one"
    )
    assert "std::min(fits_kv, fits_scr)" in body_norm, (
        "the whole-plan check's remediation must be the MIN of the KV-based and SCRATCH-zone-based "
        "largest-fitting figures, per the reviewed design -- not either alone"
    )

    # next_plan.vram_bytes covers only weights + KV + the MoE MMID pool; it
    # omits the ONEDNN/RUNTIME/SCRATCH zone reservations that the arena is
    # separately sized from. That omission must be documented, including
    # which direction the resulting error runs (this check can UNDER
    # -estimate total device usage), not left implicit -- checked against
    # RAW (comment-bearing) text, since this is prose inside a comment that
    # strip_comments() would otherwise remove before this test ever saw it.
    raw_func_start = GGML_SYCL_CPP.find("void ggml_backend_sycl_set_runtime_context(")
    assert raw_func_start != -1
    raw_next_func = GGML_SYCL_CPP.find(
        "ggml_backend_sycl_set_runtime_context_for_model(", raw_func_start + 1
    )
    assert raw_next_func != -1
    raw_body = GGML_SYCL_CPP[raw_func_start:raw_next_func]
    assert "ONEDNN/RUNTIME/SCRATCH" in raw_body, (
        "the whole-plan check's comment must name which zone reservations next_plan.vram_bytes "
        "omits (ONEDNN/RUNTIME/SCRATCH), so a future reader does not assume it covers everything"
    )
    assert "UNDER" in raw_body, (
        "the whole-plan check's comment must state the omission's error direction (it can "
        "under-estimate total device usage), not just that terms are missing"
    )


def test_for_model_forwards_flash_attn_enabled():
    """ggml_backend_sycl_set_runtime_context_for_model() must forward both new
    parameters into the inner call rather than dropping either on the floor."""
    func_start = GGML_SYCL_CPP_CODE.find("ggml_sycl_lifecycle_result ggml_backend_sycl_set_runtime_context_for_model(")
    assert func_start != -1, "ggml_backend_sycl_set_runtime_context_for_model() definition not found"
    body_norm = _normalize_ws(GGML_SYCL_CPP_CODE[func_start : func_start + 8000])
    assert re.search(
        r"ggml_backend_sycl_set_runtime_context\(backend, n_ctx, n_ubatch, n_seq_max,\s*"
        r"flash_attn_enabled,\s*reserved_compute_buffer_bytes\)",
        body_norm,
    ), (
        "ggml_backend_sycl_set_runtime_context_for_model() must forward flash_attn_enabled and "
        "reserved_compute_buffer_bytes to the inner call, in that order"
    )


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

    # Neither may be declared `static` in the .cpp -- that would make them
    # unreachable from ggml-sycl.cpp, silently turning the guard above into
    # a compile error this text-only test cannot otherwise catch (a build is
    # a stronger check, but this test runs without one).
    demand_def = cpp_norm.find("size_t unified_cache_nonfa_attn_scratch_demand_bytes(")
    assert demand_def != -1
    preceding = cpp_norm[max(0, demand_def - 40) : demand_def]
    assert "static" not in preceding, "unified_cache_nonfa_attn_scratch_demand_bytes() must not be file-static"


def test_auto_flash_attn_resolution_rechecks_the_guard():
    """cparams.flash_attn defaults true for AUTO before resolve_fused_ops()
    resolves it (llama-context.cpp's cparams init runs before the
    constructor's runtime-context call), so an AUTO context that resolves
    to OFF must re-trigger the SYCL runtime-context call with the
    now-resolved value, or the guard above never sees it for that context."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    assert re.search(
        r"void llama_context::sycl_resync_runtime_context_flash_attn\(bool\s+query_reserved_compute_buffer\)",
        ctx_norm,
    ), (
        "expected a shared helper re-running the runtime-context call, callable from the "
        "constructor, resolve_fused_ops(), and sched_reserve(), parameterized on whether to "
        "query the real compute-buffer size"
    )

    resolve_start = ctx_norm.find("void llama_context::resolve_fused_ops(")
    assert resolve_start != -1, "resolve_fused_ops() definition not found"
    resolve_body = ctx_norm[resolve_start : resolve_start + 4000]
    assert re.search(r"if \(cparams\.auto_fa\) \{[^}]*resolve\([^;]*flash_attn[^;]*;[^}]*"
                      r"sycl_resync_runtime_context_flash_attn\(\);", resolve_body), (
        "resolve_fused_ops() must call sycl_resync_runtime_context_flash_attn() (with the default "
        "query_reserved_compute_buffer=false, since sched_reserve() has not run yet at this point) "
        "inside the same if (cparams.auto_fa) block that resolves flash_attn, so it fires exactly "
        "once, right when the real value becomes known"
    )

    # The helper's own header declaration exists too, so both callers compile
    # against a real class member, not an undeclared symbol this test alone
    # would not catch.
    ctx_h = (ROOT / "src/llama-context.h").read_text()
    ctx_h_norm = _normalize_ws(strip_comments(ctx_h))
    assert re.search(
        r"void sycl_resync_runtime_context_flash_attn\(bool\s+query_reserved_compute_buffer\s*=\s*false\);",
        ctx_h_norm,
    )


def test_sched_reserve_makes_the_authoritative_post_reserve_call():
    """The constructor's and resolve_fused_ops()'s calls both necessarily run
    before sched_reserve()'s graph-reserve passes allocate the SYCL
    scheduler's real compute buffer, so neither can know its actual size.
    sched_reserve() itself must make one more call, with
    query_reserved_compute_buffer=true, once that size is known -- this is
    the authoritative call the guard's compute-buffer-regrowth term depends
    on."""
    ctx_norm = _normalize_ws(LLAMA_CONTEXT_CPP_CODE)
    reserve_sig = "void llama_context::sched_reserve()"
    reserve_start = ctx_norm.find(reserve_sig)
    assert reserve_start != -1, "sched_reserve() definition not found"
    # Bound the search to sched_reserve()'s own body via the next top-level
    # method definition after it, so a match cannot come from some unrelated
    # later call site. Search starts past this function's OWN signature --
    # searching from reserve_start + 1 would immediately re-match sched
    # _reserve() itself, collapsing the body to nothing.
    body_start = reserve_start + len(reserve_sig)
    next_method = re.search(r"void llama_context::\w+\(", ctx_norm[body_start:])
    assert next_method is not None, "could not bound sched_reserve()'s body"
    reserve_body = ctx_norm[reserve_start : body_start + next_method.start()]

    # The /*query_reserved_compute_buffer=*/ inline comment is itself
    # stripped by strip_comments() before this text is seen, so the pattern
    # below matches the call as it appears AFTER stripping: a bare `true`.
    assert re.search(
        r"sycl_resync_runtime_context_flash_attn\(\s*true\s*\)",
        reserve_body,
    ), (
        "sched_reserve() must call sycl_resync_runtime_context_flash_attn(true) after its "
        "graph-reserve passes complete, so the guard's authoritative check runs with the real "
        "reserved compute-buffer size"
    )

    # The size query itself lives inside the shared helper (queried per
    # backend there, not passed in as one externally-computed value), not
    # literally inside sched_reserve()'s own text.
    helper_start = ctx_norm.find("void llama_context::sycl_resync_runtime_context_flash_attn(")
    assert helper_start != -1, "sycl_resync_runtime_context_flash_attn() definition not found"
    helper_body = ctx_norm[helper_start : helper_start + 4000]
    assert "ggml_backend_sched_get_buffer_size(sched.get(), backend.get())" in helper_body, (
        "the shared helper must query the real per-backend compute-buffer size when asked, not "
        "guess it or take it as an externally-precomputed single value"
    )


def test_plan_time_shape_is_recorded_unconditionally():
    """populate_host_zone_sizing() must record the non-FA shape
    UNCONDITIONALLY (not gated behind #if GGML_SYCL_DNNL like the oneDNN
    sibling) -- the path this sizes for is independent of oneDNN."""
    cpp_norm = _normalize_ws(CACHE_CPP_CODE)
    assert "unified_cache_set_planned_nonfa_attn_scratch_shape(plan.device_id, plan.planner_n_head, plan.planner_n_ubatch, plan.planner_n_ctx)" in cpp_norm

    # Comment-stripped text drops "#if GGML_SYCL_DNNL" lines too (they are
    # preprocessor directives, not comments -- re-check against the RAW,
    # comment-bearing source): the call site must not sit inside that guard.
    raw = CACHE_CPP
    call_idx = raw.find("unified_cache_set_planned_nonfa_attn_scratch_shape(plan.device_id")
    assert call_idx != -1
    # Find the nearest enclosing #if/#endif pair by scanning backward for the
    # last #if GGML_SYCL_DNNL and the last #endif before the call; if the
    # #endif comes after the #if (i.e. no #endif closed it before our call),
    # the call is still inside that block.
    before = raw[:call_idx]
    last_if = before.rfind("#if GGML_SYCL_DNNL")
    last_endif = before.rfind("#endif")
    assert not (last_if != -1 and last_if > last_endif), (
        "unified_cache_set_planned_nonfa_attn_scratch_shape() call site must not be inside "
        "an #if GGML_SYCL_DNNL block"
    )
