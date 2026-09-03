#!/usr/bin/env python3
"""Source gate for llama.cpp-nz1k (prefill L2b phase 1), extended by
llama.cpp-pktr for the ONEDNN_SOA kernel-choice eligibility split.

Two SEPARATE gates now apply to the Q8_0 SOA oneDNN path, and this file
checks both:

  1. KERNEL-CHOICE ELIGIBILITY (llama.cpp-pktr): whether ONEDNN_SOA may be
     selected at all is gated by ggml_sycl_q8_0_onednn_soa_enabled()
     (GGML_SYCL_Q8_ONEDNN_SOA), which defaults ON -- the pktr per-weight
     layout rule now materializes tile-misaligned dense Q8_0 rows SOA at
     default env (e.g. gemma4-E4B's K=2560 head), and those PP batches need a
     route to oneDNN GEMM parity, not MMQ_SOA at ~3 TFLOPS.
  2. WOQ-INT8 EXECUTE (llama.cpp-nz1k, unchanged): whether the int8-plane
     WoQ primitive itself runs, inside ggml_sycl_op_mul_mat_sycl, is still
     gated by ggml_sycl_onednn_woq_q8_enabled() (GGML_SYCL_ONEDNN_WOQ_Q8),
     default OFF (ruling 9 / nz1k c-c4jp: phase 1 stages the scale plane per
     call, an opt-in interim only; the default is decided on llama.cpp-2zsc
     with the A/B numbers). With it off, an ONEDNN_SOA-selected batch still
     reaches oneDNN PP through the SOA-aware f16 dequant + row_gemm fallback
     in the same function -- so the SOA-plane LOOKUP (writing q8_0_soa_ptr)
     must run whenever the arm is reached, unconditional of the WoQ env; only
     the prepare/stage/execute of the WoQ primitive is gated by it.

The lookup must apply the codebase's standard eligibility predicate
(ggml_sycl_can_use_layout_for_kernel) before its single lookup, that lookup
must be the only writer of q8_0_soa_ptr, and both env vars must default as
stated above.

This file exists so a quiet default flip, or a new eligibility site for
ONEDNN_SOA that forgets the correct env gate, fails a test instead of
shipping.

Runs under pytest (llama_test_pytest registration) and as a plain script.
Point it at alternate copies (to exercise the RED path with a deliberately
broken tree) via GGML_SYCL_WOQ_Q8_BACKEND_SOURCE / GGML_SYCL_WOQ_Q8_COMMON_SOURCE;
defaults are the in-tree files. No SYCL device or build is touched.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("GGML_SYCL_WOQ_Q8_BACKEND_SOURCE", str(ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp")))
COMMON = Path(os.environ.get("GGML_SYCL_WOQ_Q8_COMMON_SOURCE", str(ROOT / "ggml/src/ggml-sycl/common.hpp")))

backend = BACKEND.read_text()
common = COMMON.read_text()

ARM_SIG = "static bool ggml_sycl_onednn_woq_q8_enabled() {"
SOA_ELIGIBILITY_SIG = "static bool ggml_sycl_q8_0_onednn_soa_enabled() {"
HELPER_SIG = "static layout_mode ggml_sycl_q8_0_dense_planned_layout(const ggml_tensor * src0, int device) {"
OP_SIG = "inline void ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context & ctx,"
PICK_SIG = "auto pick_kernel_for_layout = [&](layout_mode layout) -> std::optional<ggml_sycl_mul_mat_kernel> {"
GET_OPTIMAL_SIG = "static layout_mode get_optimal(ggml_type qtype, tensor_usage usage, int device_id = -1) {"


def matching_brace(text, open_idx):
    """Comment/string-aware brace match (self-contained copy, per this fork's
    one-file-per-gate convention)."""
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
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "chr"
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
                i += 2
                continue
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
        i += 1
    raise AssertionError("unbalanced braces")


def function_body(text, signature):
    # Whitespace-flexible (spec review nit 4, rev-pktr-spec-4): a reflowed
    # signature (e.g. a long parameter list clang-format wraps differently)
    # must not read as "missing definition".
    idx = ws_find(text, signature)
    assert idx >= 0, f"missing definition: {signature}"
    open_idx = text.find("{", idx)
    return text[open_idx : matching_brace(text, open_idx) + 1]


def ws_pattern(needle):
    """Compile a regex matching `needle` with every run of whitespace
    treated as flexible (`\\s+`), so a pure clang-format line-wrap in the
    source (demonstrably real: git-clang-format wrapped
    `q8_0_dense_planned_layout == GGML_LAYOUT_SOA &&` across two lines in
    this exact function during development) does not break an exact-string
    match. Positions returned by .search()/.finditer() are real offsets
    into the ORIGINAL, un-normalized text (spec review finding 3,
    rev-pktr-spec-3)."""
    tokens = needle.split()
    assert tokens, "empty needle"
    return re.compile(r"\s+".join(re.escape(t) for t in tokens))


def ws_find(text, needle, start=0):
    """First match position of `needle` in `text`, whitespace-flexible, or -1."""
    m = ws_pattern(needle).search(text, start)
    return m.start() if m else -1


def ws_rfind(text, needle):
    """Last match position of `needle` in `text`, whitespace-flexible, or -1."""
    matches = list(ws_pattern(needle).finditer(text))
    return matches[-1].start() if matches else -1


def ws_in(text, needle):
    """Whitespace-flexible containment check: True iff `needle` (as a
    sequence of tokens) appears in `text` with any whitespace between
    tokens, including a line wrap. Argument order (text, needle) matches
    ws_find/ws_rfind/ws_count in this same helper block (spec review nit,
    rev-pktr-spec-6: this function alone used to take (needle, text))."""
    return ws_pattern(needle).search(text) is not None


def ws_count(text, needle):
    """Number of whitespace-flexible matches of `needle` in `text` -- for
    "defined exactly once" checks on a signature (e.g. HELPER_SIG,
    ARM_SIG, SOA_ELIGIBILITY_SIG) that a reflowed parameter list must not
    false-fail (spec review should-fix 1, rev-pktr-spec-5)."""
    return len(list(ws_pattern(needle).finditer(text)))


def enclosing_if_condition_text(text, pos):
    """Full condition text of the innermost `if (...)` whose CONDITION
    (not body) contains `pos` -- complements enclosing_if_conditions, which
    only sees ifs whose body contains the target position. Used to extract
    a guard's complete condition regardless of a line wrap inside it, so a
    substring check across the whole condition is safe rather than a
    fragile single-physical-line slice."""
    if_idx = text.rfind("if (", 0, pos)
    if if_idx < 0:
        if_idx = text.rfind("if(", 0, pos)
    if if_idx < 0:
        return None
    paren_start = text.find("(", if_idx)
    depth = 1
    j = paren_start + 1
    n = len(text)
    while depth > 0 and j < n:
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        j += 1
    cond_end = j - 1
    if not (paren_start < pos < cond_end):
        return None
    return text[paren_start + 1 : cond_end]


def enclosing_if_conditions(text, start, target):
    """List of condition strings for every currently-open `if (...) {` block at
    `target`, scanning code (not comments/strings) from `start`. Used to check
    that a call site is actually gated by a condition possibly several nested
    `if`s above it, not just its immediate parent (spec review finding 2,
    rev-pktr-spec-2: the SOA lookup's planned-layout check sits on the OUTER
    `if`, while its own immediate guard is an unrelated inner predicate)."""
    stack = []
    i = start
    state = "code"
    n = len(text)
    while i < target:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch == '"':
                state = "str"
                i += 1
                continue
            if ch == "'":
                state = "chr"
                i += 1
                continue
            if text.startswith("if (", i) or text.startswith("if(", i):
                paren_start = text.find("(", i)
                depth = 1
                j = paren_start + 1
                while depth > 0 and j < n:
                    c = text[j]
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                    j += 1
                cond = text[paren_start + 1 : j - 1]
                k = j
                while k < n and text[k] in " \t\n":
                    k += 1
                if k < n and text[k] == "{":
                    stack.append(("if", cond))
                    i = k + 1
                    continue
                i = j
                continue
            if ch == "{":
                stack.append(("block", None))
                i += 1
                continue
            if ch == "}":
                if stack:
                    stack.pop()
                i += 1
                continue
            i += 1
        elif state == "line":
            if ch == "\n":
                state = "code"
            i += 1
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            i += 1
        elif state == "str":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "code"
            i += 1
        elif state == "chr":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "code"
            i += 1
    return [cond for kind, cond in stack if kind == "if"]


def test_arm_env_accessor_exists_once_and_defaults_off():
    assert ws_count(backend, ARM_SIG) == 1, "ggml_sycl_onednn_woq_q8_enabled must be defined exactly once"
    body = function_body(backend, ARM_SIG)
    assert 'std::getenv("GGML_SYCL_ONEDNN_WOQ_Q8")' in body, "arm accessor must read GGML_SYCL_ONEDNN_WOQ_Q8"
    assert (
        "(env != nullptr && std::atoi(env) != 0) ? 1 : 0" in body
    ), "GGML_SYCL_ONEDNN_WOQ_Q8 must default OFF (unset => 0); ruling 9 / nz1k c-c4jp"


def test_soa_eligibility_accessor_exists_once_and_defaults_on():
    # llama.cpp-pktr: separate accessor for whether ONEDNN_SOA may be chosen
    # at all, distinct from the WoQ-execute accessor above. Mirrors
    # ggml_sycl_q8_0_onednn_coalesced_enabled's default-ON pattern.
    assert ws_count(backend, SOA_ELIGIBILITY_SIG) == 1, "ggml_sycl_q8_0_onednn_soa_enabled must be defined exactly once"
    body = function_body(backend, SOA_ELIGIBILITY_SIG)
    assert 'std::getenv("GGML_SYCL_Q8_ONEDNN_SOA")' in body, "SOA eligibility accessor must read GGML_SYCL_Q8_ONEDNN_SOA"
    assert (
        "(env == nullptr || std::atoi(env) != 0) ? 1 : 0" in body
    ), "GGML_SYCL_Q8_ONEDNN_SOA must default ON (unset => 1; env==nullptr short-circuits atoi)"


def test_layout_env_defaults_coalesced_and_only_touches_dense_projections():
    assert 'std::getenv("GGML_SYCL_Q8_DENSE_LAYOUT")' in common, "common.hpp must read GGML_SYCL_Q8_DENSE_LAYOUT"
    assert (
        'q8_dense_soa_cached = (env && std::strcmp(env, "soa") == 0) ? 1 : 0;' in common
    ), 'GGML_SYCL_Q8_DENSE_LAYOUT must default to coalesced (only "soa" selects SOA)'
    # Whitespace-flexible: the original exact-string form (with a literal
    # newline and hardcoded indentation) false-failed on a pure reindent of
    # this block in common.hpp (spec review nit 3, rev-pktr-spec-4).
    soa_returns = len(
        list(ws_pattern("if (qtype == GGML_TYPE_Q8_0 && q8_dense_soa_cached) { return GGML_LAYOUT_SOA;").finditer(common))
    )
    assert soa_returns == 2, f"expected the SOA return in the ATTENTION/FFN and OUTPUT blocks (2), found {soa_returns}"
    get_optimal = function_body(common, GET_OPTIMAL_SIG)
    emb_idx = get_optimal.find("if (usage == tensor_usage::EMBEDDING) {")
    assert emb_idx > 0, "get_optimal EMBEDDING block not found"
    emb_block = get_optimal[emb_idx : matching_brace(get_optimal, get_optimal.find("{", emb_idx)) + 1]
    assert "q8_dense_soa_cached" not in emb_block, "GGML_SYCL_Q8_DENSE_LAYOUT must not touch EMBEDDING (nz1k scope)"


def test_every_onednn_soa_selection_site_consults_the_env_gate():
    # Four `case ONEDNN_SOA:` sites: kernel-name table, override check,
    # preferred-kernel selection, dispatch. llama.cpp-pktr: the two SELECTION
    # sites must consult the KERNEL-CHOICE ELIGIBILITY gate
    # (ggml_sycl_q8_0_onednn_soa_enabled, default ON) -- NOT the WoQ-execute
    # gate, which they must no longer reference at all; the name table and the
    # dispatch arm (which executes an already-selected kernel) consult neither.
    case_sites = [m.start() for m in re.finditer(r"case ggml_sycl_mul_mat_kernel::ONEDNN_SOA:", backend)]
    assert len(case_sites) == 4, (
        f"expected 4 `case ONEDNN_SOA:` sites (name table, override check, preferred selection, dispatch), "
        f"found {len(case_sites)}"
    )
    kinds = {"gated": 0, "dispatch": 0, "name": 0, "unknown": 0}
    for start in case_sites:
        end = re.search(r"\n\s*(case |default:)", backend[start + 10 :])
        body = backend[start : start + 10 + (end.start() if end else 800)]
        if "ggml_sycl_q8_0_onednn_soa_enabled()" in body:
            assert (
                "ggml_sycl_onednn_woq_q8_enabled()" not in body
            ), "a kernel-choice ELIGIBILITY site must not also reference the WoQ-EXECUTE gate"
            kinds["gated"] += 1
        elif "ggml_sycl_op_mul_mat<no_quantize_q8_1>" in body:
            kinds["dispatch"] += 1
        elif 'return "ONEDNN_SOA";' in body:
            kinds["name"] += 1
        else:
            kinds["unknown"] += 1
    assert kinds == {"gated": 2, "dispatch": 1, "name": 1, "unknown": 0}, f"ONEDNN_SOA case census: {kinds}"


def test_pick_kernel_for_layout_soa_case_is_gated():
    pick = function_body(backend, PICK_SIG)
    soa_case = pick[pick.find("case GGML_LAYOUT_SOA:") : pick.find("case GGML_LAYOUT_AOS:")]
    assign = soa_case.find("layout_kernel = ggml_sycl_mul_mat_kernel::ONEDNN_SOA;")
    assert assign > 0, "pick_kernel_for_layout SOA case must be able to select ONEDNN_SOA"
    guard = soa_case[:assign]
    assert ws_in(guard, "src0->type == GGML_TYPE_Q8_0") and "ggml_sycl_q8_0_onednn_soa_enabled()" in guard, (
        "pick_kernel_for_layout's ONEDNN_SOA choice must be guarded by Q8_0 && the eligibility gate"
    )
    assert (
        "ggml_sycl_onednn_woq_q8_enabled()" not in guard
    ), "pick_kernel_for_layout's ONEDNN_SOA choice must not also require the WoQ-execute gate"
    # Qualified enum reference, not bare "ONEDNN_SOA": llama.cpp-pktr's env
    # var GGML_SYCL_Q8_ONEDNN_SOA also contains that substring, so a comment
    # naming the env var would otherwise double-count and false-fail this.
    assert (
        soa_case.count("ggml_sycl_mul_mat_kernel::ONEDNN_SOA") == 1
    ), "ONEDNN_SOA must be selected from exactly one branch of the SOA case"


def test_planned_layout_decided_once_and_gates_both_lookups():
    # llama.cpp-pktr: ggml_sycl_get_weight_layout_ptr does not verify that
    # what it returns matches the layout it was asked for, so a lookup's
    # own success/failure must never stand in for "the planner chose this
    # layout" -- in EITHER direction (this caused a real B50 regression,
    # `1, 2, 3, 4, 5,###############` on Mistral Q8_0 at default env, when
    # the planned-layout signal was read from get_effective_layout_mode
    # instead -- see the shared helper's own comment for why). The tensor's
    # actual materialized layout is decided ONCE, by the standalone helper
    # ggml_sycl_q8_0_dense_planned_layout, and used to gate both lookups.
    assert (
        ws_count(backend, HELPER_SIG) == 1
    ), "ggml_sycl_q8_0_dense_planned_layout must be defined exactly once"
    helper_body = function_body(backend, HELPER_SIG)
    assert (
        "get_effective_layout_mode(" not in helper_body
    ), "the planned layout must NOT be read from get_effective_layout_mode (rev-pktr-spec-2 regression source)"
    assert (
        helper_body.count("ggml_sycl_resolve(") == 1
    ), "the helper must call ggml_sycl_resolve() exactly once"
    assert ws_in(helper_body, "resolved.on_device"), "the helper must require on_device before trusting .layout"
    op_body = function_body(backend, OP_SIG)
    call_pattern = re.compile(r"q8_0_dense_planned_layout\s*=\s*ggml_sycl_q8_0_dense_planned_layout\(")
    calls = list(call_pattern.finditer(op_body))
    assert len(calls) == 1, (
        f"ggml_sycl_op_mul_mat_sycl must assign q8_0_dense_planned_layout from the shared helper exactly once, "
        f"found {len(calls)}"
    )
    decl_idx = calls[0].start()
    soa_guard_idx = ws_find(op_body, "q8_0_dense_planned_layout == GGML_LAYOUT_SOA")
    coalesced_guard_idx = ws_find(op_body, "q8_0_dense_planned_layout == GGML_LAYOUT_COALESCED")
    assert 0 < decl_idx < soa_guard_idx, "the planned layout must be declared before the SOA lookup consults it"
    assert 0 < decl_idx < coalesced_guard_idx, "the planned layout must be declared before the COALESCED lookup consults it"
    # Neither lookup's guard may use the OTHER lookup's own pointer result as
    # a proxy for what the planner chose. Extract the FULL enclosing `if`
    # condition (not a single physical line -- a line wrap inside the
    # condition would otherwise truncate this check) via
    # enclosing_if_condition_text.
    soa_guard_text = enclosing_if_condition_text(op_body, soa_guard_idx)
    coalesced_guard_text = enclosing_if_condition_text(op_body, coalesced_guard_idx)
    assert soa_guard_text is not None, "no enclosing `if (...)` condition contains the SOA equality"
    assert coalesced_guard_text is not None, "no enclosing `if (...)` condition contains the COALESCED equality"
    assert (
        "q8_0_coalesced_ptr" not in soa_guard_text
    ), "the SOA lookup's guard must not reference q8_0_coalesced_ptr"
    assert "q8_0_soa_ptr" not in coalesced_guard_text, (
        "the COALESCED lookup's guard must not reference q8_0_soa_ptr -- a planned-SOA weight whose SOA lookup "
        "declined must not fall through to a COALESCED lookup"
    )


def test_exactly_three_q8_0_layout_lookups_all_gated_on_planned_layout():
    # llama.cpp-pktr spec-review finding 2 (rev-pktr-spec-2): a THIRD Q8_0
    # layout lookup exists in this function -- the llama.cpp-dkw0 fp32
    # sibling's coalesced dequant lookup (reachable when the f16 GEMM branch
    # above declines: GGML_SYCL_F16=OFF, a row-split dispatch, or
    # GGML_PREC_F32) -- and it requested GGML_LAYOUT_COALESCED for ANY Q8_0
    # weight with no planned-layout gate at all, so a tile-misaligned Q8_0
    # weight the pktr rule resolves to SOA could have its SOA plane handed
    # back and read as coalesced. Census every
    # ggml_sycl_get_weight_layout_ptr(src0, ctx.device, GGML_LAYOUT_*) call
    # site in the function and require each one sit behind an enclosing
    # `if` (possibly several levels up, not just its immediate parent --
    # the SOA lookup's own immediate guard is an unrelated full_rows/
    # k_blocked predicate) that references q8_0_dense_planned_layout.
    # rev-pktr-spec-3 finding 3: checking only that some enclosing `if`
    # CONTAINS the variable name is directionally blind -- a mutant gating
    # the fp32 lookup on `q8_0_dense_planned_layout != GGML_LAYOUT_SOA`
    # (wrong operator, wrong constant) passed the earlier form of this
    # test. Require the SPECIFIC equality each lookup needs: the target
    # layout it requests must equal the requested constant in an enclosing
    # condition, via `==` (not `!=`, `<`, or anything else).
    op_body = function_body(backend, OP_SIG)
    lookup_pattern = re.compile(r"ggml_sycl_get_weight_layout_ptr\(src0, ctx\.device, GGML_LAYOUT_(SOA|COALESCED)\)")
    matches = list(lookup_pattern.finditer(op_body))
    assert len(matches) == 3, (
        f"expected exactly 3 Q8_0 layout lookups in ggml_sycl_op_mul_mat_sycl (SOA, the f16-branch COALESCED, "
        f"the dkw0 fp32-branch COALESCED), found {len(matches)}"
    )
    for m in matches:
        requested_layout = m.group(1)  # "SOA" or "COALESCED"
        conds = enclosing_if_conditions(op_body, 0, m.start())
        assert conds, f"lookup {m.group(0)!r} at offset {m.start()} has no enclosing `if` guard at all"
        required = f"q8_0_dense_planned_layout == GGML_LAYOUT_{requested_layout}"
        assert any(ws_in(c, required) for c in conds), (
            f"lookup {m.group(0)!r} at offset {m.start()} requests {requested_layout} but no enclosing `if` "
            f"contains the exact equality {required!r} (found {len(conds)} enclosing guard(s): {conds!r})"
        )


def test_single_gemm_call_site_consumes_the_soa_plane_and_declines_safely():
    assert backend.count("DnnlGemmWrapper::woq_gemm_q8_0(") == 1, "woq_gemm_q8_0 must be called from exactly one site"
    op_body = function_body(backend, OP_SIG)
    call = op_body.find("DnnlGemmWrapper::woq_gemm_q8_0(")
    assert call > 0, "the woq_gemm_q8_0 call must live in ggml_sycl_op_mul_mat_sycl"
    before = op_body[:call]
    # llama.cpp-pktr: the outer guard no longer requires the WoQ-execute env --
    # the SOA-plane lookup must run whenever the arm is reached (Q8_0,
    # planned SOA, row_diff > 0, contiguous), so the fallback dequant further
    # down this function can address q8_0_soa_ptr correctly regardless of
    # whether the WoQ int8-plane primitive itself is opted into.
    guard_idx = ws_rfind(
        before, "if (!used_woq && src0->type == GGML_TYPE_Q8_0 && q8_0_dense_planned_layout == GGML_LAYOUT_SOA &&"
    )
    assert guard_idx > 0, (
        "the arm must be guarded by `!used_woq && Q8_0 && q8_0_dense_planned_layout == GGML_LAYOUT_SOA && ...` "
        "(no WoQ-execute env in the outer guard)"
    )
    outer_guard_text = enclosing_if_condition_text(before, guard_idx + 5)
    assert outer_guard_text is not None, "the outer guard's own condition could not be extracted"
    assert (
        "ggml_sycl_onednn_woq_q8_enabled()" not in outer_guard_text
    ), "the WoQ-execute env must not gate the outer guard (only the decline reasoning may reference it)"
    arm = before[guard_idx:]
    assert ws_in(
        arm, "ggml_sycl_get_weight_layout_ptr(src0, ctx.device, GGML_LAYOUT_SOA)"
    ), "the arm must consume the SOA-materialized plane (GGML_LAYOUT_SOA lookup)"
    assert 'decline = "soa_plane_not_resident"' in arm, "a missing SOA plane must decline, not proceed"
    assert 'decline = "no_pp_scratch"' in arm, "scales must be staged only into the existing PP scratch (ruling 1)"
    # llama.cpp-pktr: with the WoQ-execute env off but a resident SOA plane,
    # the arm must decline with a reason that says so -- NOT
    # "soa_plane_not_resident" (the plane WAS found) and not silent success.
    assert 'decline = "woq_disabled"' in arm, "WoQ execute disabled with a resident SOA plane must decline distinctly"
    assert (
        arm.find('decline = "soa_plane_not_resident"') < arm.find('decline = "woq_disabled"') < arm.find('decline = "no_pp_scratch"')
    ), "decline reasons must be checked in order: plane residency, then WoQ opt-in, then scratch availability"
    assert (
        "ggml_sycl_onednn_woq_q8_enabled()" in arm
    ), "the WoQ-execute gate must still be consulted somewhere in the arm (the woq_disabled decline)"
    # Structural invariant: the SOA plane is looked up only behind the
    # non-materializing predicate AND the full-rows / K-blocked tests, so a
    # non-null q8_0_soa_ptr implies the fallback dequant's addressing is valid
    # and the arm can never mint a second stored layout.
    assert ws_in(
        arm,
        "if (full_rows && k_blocked && ggml_sycl_can_use_layout_for_kernel(src0, GGML_LAYOUT_SOA, ctx.device)) {",
    ), "the SOA lookup must sit behind full_rows && k_blocked && ggml_sycl_can_use_layout_for_kernel"
    lookup = ws_find(arm, "ggml_sycl_get_weight_layout_ptr(src0, ctx.device, GGML_LAYOUT_SOA)")
    guard = ws_find(arm, "if (full_rows && k_blocked && ggml_sycl_can_use_layout_for_kernel(")
    assert 0 < guard < lookup, "the predicate guard must precede the SOA lookup"
    # The guarded lookup must be the ONLY writer of q8_0_soa_ptr and the ONLY
    # SOA getter call: a second, ungated assignment would reinstate both the
    # partial-row mis-addressing and a dispatch-time layout request.
    writers = re.findall(r"\bq8_0_soa_ptr\s*=[^=]", op_body)
    assert len(writers) == 2, f"expected exactly the declaration + one guarded writer of q8_0_soa_ptr, found {len(writers)}"
    assert op_body.count("q8_0_soa_ptr = candidate;") == 1, "the guarded lookup must be the single writer"
    assert (
        len(list(ws_pattern("ggml_sycl_get_weight_layout_ptr(src0, ctx.device, GGML_LAYOUT_SOA)").finditer(op_body)))
        == 1
    ), "exactly one SOA getter call is allowed, inside the guarded block"
    # The primitive must be prepared (known good) ONCE, before the staging kernel
    # is submitted, and that same plan is what executes.
    ready = arm.find("DnnlGemmWrapper::woq_q8_0_prepare(")
    stage = arm.find("q8_0_soa_scale_plane_to_kbn_sycl(")
    assert 0 < ready < stage, "woq_q8_0_prepare must run before the scale staging is submitted"
    assert arm.count("DnnlGemmWrapper::woq_q8_0_prepare(") == 1, "the arm must prepare exactly once per dispatch"
    assert op_body[call:].startswith("DnnlGemmWrapper::woq_gemm_q8_0(ctx, plan,"), "the execute must consume the prepared plan"
    after = op_body[call:]
    # The decline path must dequantize from the SOA plane, and that branch must
    # be guarded by the pointer itself (an `if (false)`-style dead branch would
    # send the SOA plane to the AOS dispatcher).
    assert re.search(
        r"if \(q8_0_soa_ptr\) \{[^{}]*dequantize_row_q8_0_soa_to_fp16_rowmajor\(q8_0_soa_ptr", after
    ), "the SOA dequant fallback must be the body of `if (q8_0_soa_ptr) {`"
    coalesced_guard_start = ws_find(after, "if (q8_0_dense_planned_layout == GGML_LAYOUT_COALESCED")
    assert coalesced_guard_start > 0, "the COALESCED lookup must be gated by the planned layout"
    coalesced_guard_text = enclosing_if_condition_text(after, coalesced_guard_start + 5)
    assert coalesced_guard_text is not None, "the COALESCED lookup's own condition could not be extracted"
    assert (
        "ggml_sycl_q8_0_onednn_coalesced_enabled()" in coalesced_guard_text
    ), "the COALESCED lookup must also still consult its own opt-out env"
    assert "q8_0_soa_ptr" not in coalesced_guard_text, (
        "the COALESCED lookup must not be gated by whether the SOA lookup declined -- not by whether the SOA "
        "lookup succeeded either"
    )


def test_scale_staging_has_exactly_one_producer():
    # The [N][K/32] SOA order must never reach oneDNN directly (probe V5 is
    # accepted-but-garbage); the transpose into PP scratch is the only producer.
    assert (
        backend.count("q8_0_soa_scale_plane_to_kbn_sycl(") == 1
    ), "exactly one scale-staging call (the transpose into PP scratch) is expected"


if __name__ == "__main__":
    import sys

    failures = 0
    for fn_name, fn in sorted(list(globals().items())):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {fn_name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}: {exc}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("all tests passed")
