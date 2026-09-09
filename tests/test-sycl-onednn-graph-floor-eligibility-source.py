"""Source-gate for llama.cpp-o3a0: the head-dim eligibility rule the oneDNN
Graph-scratch zone floor uses to decide which layers count toward
n_head_ctx_max/n_head_swa_max (src/llama-model.cpp) must stay identical to
the real oneDNN SDPA eligibility gate it mirrors
(ggml_sycl_flash_attn_ext_onednn_plan(), ggml/src/ggml-sycl/fattn-onednn.cpp).

src/llama-model.cpp is a llama-layer file and cannot include the SYCL-only
fattn-onednn.cpp/.hpp to share the real predicate directly (see the header
comment on llama_model_sycl_onednn_head_dim_eligible()), so the rule is
REPLICATED rather than shared. A replicated rule can silently drift from its
source without either side's own tests noticing -- each half looks correct
in isolation. This is a pure text/regex check, no build required, matching
test-sycl-onednn-graph-allocator-source.py's pytest-collectible pattern
(llama_test_pytest hands this file to pytest.main(), so checks must live
inside a test_*() function or pytest's "no tests collected" (exit 5) is
scored as a failure by design, not a skip).

Scope: this only checks the two copies stay IN SYNC with each other, not that
either is independently "correct" -- that is what the two source files' own
review and tests are for.

spec-review round 1 (F1) hardening: the real gate's D>256 branch is a
DISJUNCTION -- eligible when the D512 hatch is set OR the layer's own
attention scale is already canonical (1/sqrt(D)) -- not the hatch alone. An
earlier revision of the replicated rule required the hatch unconditionally,
which is STRICTER than the real gate and would silently under-count
eligibility (under-provisioning the floor) for a D>256 layer whose scale
happens to be canonical without the hatch. The checks below pin the full
disjunction, and test_removing_the_scale_clause_is_caught() proves the
disjunction check is a real mutation witness, not a check that would pass
regardless of whether the scale clause exists.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_MODEL_CPP = (ROOT / "src/llama-model.cpp").read_text(encoding="utf-8")
FATTN_ONEDNN_CPP = (ROOT / "ggml/src/ggml-sycl/fattn-onednn.cpp").read_text(encoding="utf-8")

ENV_VAR_NAME = "GGML_SYCL_FA_ONEDNN_D512_SCALE"


def extract_function_body(src: str, signature_anchor: str) -> str:
    """Return the brace-balanced body (including the braces) of the function
    whose signature contains signature_anchor, scanning from the first '{'
    at or after the anchor to its matching '}'. Mirrors
    test-sycl-onednn-graph-allocator-source.py's helper of the same name."""
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


LLAMA_MODEL_ELIGIBLE_BODY = extract_function_body(
    LLAMA_MODEL_CPP,
    "static bool llama_model_sycl_onednn_head_dim_eligible(uint32_t head_dim, float f_attention_scale) {",
)
FATTN_ONEDNN_D512_SCALE_RELAXED_BODY = extract_function_body(
    FATTN_ONEDNN_CPP, "bool ggml_sycl_fa_onednn_d512_scale_relaxed() {"
)


def test_both_copies_read_the_identical_env_var() -> None:
    # The single point of truth this test exists to protect: if either side
    # renames its env var (or one is fixed and the other forgotten), the
    # planner-side floor and the SYCL-side SDPA gate silently stop agreeing
    # on which layers are eligible.
    assert f'std::getenv("{ENV_VAR_NAME}")' in LLAMA_MODEL_ELIGIBLE_BODY, (
        f"llama_model_sycl_onednn_head_dim_eligible() must read {ENV_VAR_NAME} via std::getenv"
    )
    assert f'std::getenv("{ENV_VAR_NAME}")' in FATTN_ONEDNN_D512_SCALE_RELAXED_BODY, (
        f"ggml_sycl_fa_onednn_d512_scale_relaxed() must read {ENV_VAR_NAME} via std::getenv"
    )


def test_both_copies_parse_the_env_var_the_same_way() -> None:
    # Different C control-flow shape (an early `if (...) { return true; }`
    # on the llama-model.cpp side vs. a ternary on the SYCL side -- the
    # llama-model.cpp copy needs an early return so the scale-check clause
    # can follow it), but the SAME truthiness rule: nonzero via atoi(), unset
    # treated as not-relaxed. A "0"-valued or unset var must not be read
    # differently by either side.
    assert re.search(r"env\s*&&\s*std::atoi\(\s*env\s*\)\s*!=\s*0", LLAMA_MODEL_ELIGIBLE_BODY), (
        "llama_model_sycl_onednn_head_dim_eligible() must treat the hatch as set only when "
        "std::atoi(env) != 0, matching the SYCL-side truthiness rule"
    )
    assert re.search(
        r"\?\s*\(\s*std::atoi\(\s*e\s*\)\s*!=\s*0\s*\)\s*:\s*false", FATTN_ONEDNN_D512_SCALE_RELAXED_BODY
    ), "ggml_sycl_fa_onednn_d512_scale_relaxed() must parse the env var as `e ? (std::atoi(e) != 0) : false`"


def test_llama_model_copy_uses_the_same_d_thresholds_as_the_sycl_gate() -> None:
    # ggml_sycl_flash_attn_ext_onednn_plan() rejects D > 512 unconditionally
    # (UNSUPPORTED_D) and requires the D512 hatch only for D > 256
    # (params.ne00 > 256). The replicated rule must use the identical
    # thresholds, not values that happen to produce the same answer for
    # today's in-tree models but diverge for some future head dim.
    assert re.search(r"head_dim\s*>\s*512", LLAMA_MODEL_ELIGIBLE_BODY), (
        "the D > 512 reject threshold must be replicated exactly (UNSUPPORTED_D in fattn-onednn.cpp)"
    )
    assert re.search(r"head_dim\s*<=\s*256", LLAMA_MODEL_ELIGIBLE_BODY), (
        "the D <= 256 unconditional-eligibility threshold must be replicated exactly "
        "(the `params.ne00 > 256` gate in fattn-onednn.cpp)"
    )
    assert re.search(r"ne00\s*>\s*512", FATTN_ONEDNN_CPP), "fattn-onednn.cpp's own D > 512 reject moved or was renamed"
    assert re.search(
        r"ne00\s*>\s*256\s*&&\s*!onednn_d512_scale_relaxed", FATTN_ONEDNN_CPP
    ), "fattn-onednn.cpp's own D > 256 scale-relaxed gate moved or was renamed"


def eligibility_disjunction_checks(body: str) -> dict:
    """The structural pieces of the D>256 disjunction: hatch-eligible OR
    scale-eligible. Factored into a function (rather than inlined in one
    test) so the mutation-witness test below can run it against a MUTATED
    copy of the body and confirm the scale-clause check specifically goes
    false, not just that "some check somewhere" fails.

    Note: `body` is the BODY ONLY (from extract_function_body(), starting
    at the opening brace) -- the two-parameter signature is not part of it,
    so it is not re-checked here. It is still enforced: the module-level
    LLAMA_MODEL_ELIGIBLE_BODY extraction itself uses the full
    `(uint32_t head_dim, float f_attention_scale)` signature as its anchor,
    so a missing or renamed parameter fails loudly at collection time
    (str.index() raising ValueError), before any test in this file runs."""
    return {
        "hatch branch returns eligible before the scale check": bool(
            re.search(r"std::atoi\(\s*env\s*\)\s*!=\s*0\s*\)\s*\{\s*return true;", body)
        ),
        "canonical scale derived from f_attention_scale the same way build_*.cpp does": bool(
            re.search(r"f_attention_scale\s*==\s*0\.0f\s*\?\s*1\.0f\s*/\s*sqrtf\(", body)
        ),
        "scale checked against the real gate's own tolerance (1e-3f)": bool(
            re.search(r"1\.0f\s*/\s*kq_scale\s*-\s*sqrtf\(.*?\)\s*<\s*1e-3f", body)
        ),
    }


def test_eligibility_mirrors_the_full_scale_or_hatch_disjunction() -> None:
    checks = eligibility_disjunction_checks(LLAMA_MODEL_ELIGIBLE_BODY)
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, "eligibility helper does not mirror the scale-or-hatch disjunction: " + ", ".join(failed)


def test_removing_the_scale_clause_is_caught() -> None:
    """Mutation witness (spec-review round 1, F1): delete the final
    tolerance-check return statement -- the clause this fix added on top of
    the hatch-only check -- and confirm the disjunction check above
    specifically goes false for THAT clause, proving it is not vacuously
    true regardless of whether the scale clause is present."""
    mutated = LLAMA_MODEL_ELIGIBLE_BODY.replace(
        "return std::fabs(1.0f / kq_scale - sqrtf(static_cast<float>(head_dim))) < 1e-3f;",
        "return false;",
    )
    assert mutated != LLAMA_MODEL_ELIGIBLE_BODY, (
        "the exact return-statement text this mutation targets was not found -- "
        "update the mutation string to match the current implementation"
    )
    mutated_checks = eligibility_disjunction_checks(mutated)
    assert not mutated_checks["scale checked against the real gate's own tolerance (1e-3f)"], (
        "the scale-tolerance check still passes after removing the clause -- "
        "the check itself is not a real mutation witness"
    )
    # The OTHER checks (hatch branch, scale derivation, D thresholds) must
    # still hold on the mutated body -- this mutation removes only the
    # tolerance-check return, so a check that also breaks here would be
    # entangled with the wrong clause.
    other_checks = {k: v for k, v in mutated_checks.items() if k != "scale checked against the real gate's own tolerance (1e-3f)"}
    still_failed = [name for name, ok in other_checks.items() if not ok]
    assert not still_failed, "unrelated checks broke on this mutation, entangled with the wrong clause: " + ", ".join(
        still_failed
    )
