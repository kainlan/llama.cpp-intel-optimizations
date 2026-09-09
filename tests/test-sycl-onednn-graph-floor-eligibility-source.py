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
    LLAMA_MODEL_CPP, "static bool llama_model_sycl_onednn_head_dim_eligible(uint32_t head_dim) {"
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
    # Same truthiness rule (nonzero via atoi), not merely the same variable
    # name -- "unset or \"0\"" vs "unset only" would silently disagree on a
    # value like "0" even though both read the identical variable.
    pattern = re.compile(r"\?\s*\(\s*std::atoi\(\s*e(?:nv)?\s*\)\s*!=\s*0\s*\)\s*:\s*false")
    assert pattern.search(LLAMA_MODEL_ELIGIBLE_BODY), (
        "llama_model_sycl_onednn_head_dim_eligible() must parse the env var as "
        "`env ? (std::atoi(env) != 0) : false`, matching the SYCL-side parse exactly"
    )
    assert pattern.search(FATTN_ONEDNN_D512_SCALE_RELAXED_BODY), (
        "ggml_sycl_fa_onednn_d512_scale_relaxed() must parse the env var as "
        "`e ? (std::atoi(e) != 0) : false`"
    )


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
