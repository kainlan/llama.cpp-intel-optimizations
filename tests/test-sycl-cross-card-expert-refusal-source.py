"""Source contract for the stage's refusal of a cross-card expert plan when the cards have no peer path.

With GGML_SYCL_MOE_MULTI_GPU=1 the planner may put expert copies on a second card. The cards on this host have no
peer path, and a two-card run on that path faulted the first card, so ggml_backend_sycl_stage_load_inventory_plan
refuses such a plan before any weight is placed. Both of its refusal sites ask: the late pass per device, before it
configures that device's host zones, and the early pass on the candidate, before it is staged as the probe plan. The
behaviour of the predicate and of the peer-path read runs in test-sycl-measure-plan-override; this file pins the
wiring that test cannot reach without a device:

  * at each site the cross-card refusal is asked before the lazy-plan refusal;
  * at each site it returns GGML_SYCL_LIFECYCLE_PLAN_REJECTED, the lazy-plan refusal's return;
  * the predicate's body is exactly "wanted and not peer path": it takes the peer fact as its argument and calls no
    device query. It reads the opt-in through the wanted check, so it is pure only in the device fact;
  * the stage helper passes the predicate the peer fact read from the peer-link records of the plan's cards;
  * the refusal is one ERROR line that names the remedy.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions; run as a script it calls pytest itself. Checks read comment-stripped, whitespace-normalised
text with adjacent string literals joined. Each check has one mutation witness that applies the regression it guards:
the witness must make its own check fail and leave every other check passing.
"""

import functools
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)
# A string literal, a char literal, a run of anything else, or one stray quote (a digit separator, say).
_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"' r"|'(?:\\.|[^'\\])*'" r"|[^\"']+" r"|.", flags=re.DOTALL)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


@functools.lru_cache(maxsize=32)
def view(src: str) -> str:
    """Comment-stripped text, every run of whitespace one space, adjacent string literals joined. Idempotent."""
    out = []
    for m in _TOKEN_RE.finditer(re.sub(r"\s+", " ", strip_comments(src))):
        tok = m.group(0)
        if tok[0] == '"' and len(out) >= 2 and out[-1] == " " and out[-2][0] == '"':
            out.pop()
        if tok[0] == '"' and out and out[-1][0] == '"':
            out[-1] = out[-1][:-1] + tok[1:]
        else:
            out.append(tok)
    return "".join(out)


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


def fn_body(text: str, sig: str) -> str:
    """The body of the definition `sig {`, never a forward declaration `sig;`."""
    s = text.find(sig + " {")
    return block_at(text, s + len(sig)) if s != -1 else ""


def ordered(text: str, *parts: str) -> bool:
    pos = 0
    for p in parts:
        k = text.find(p, pos)
        if k == -1:
            return False
        pos = k + len(p)
    return bool(text)


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as view() with the one `old` inside the function `sig` replaced by `new`."""
    v = view(src)
    body = fn_body(v, sig)
    assert body and body.count(old) == 1, f"witness anchor {old!r} is not once in {sig!r}"
    s = v.find(body, v.find(sig + " {"))
    return v[:s] + body.replace(old, new, 1) + v[s + len(body) :]


STAGE_SIG = (
    "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_load_inventory_plan(const ggml_sycl_tensor_inventory * "
    "inventory, const ggml_sycl_placement_envelope * envelope, bool early)"
)
PRED_SIG = (
    "static bool ggml_sycl_cross_card_expert_plan_refused(const ggml_sycl::placement_plan & plan, bool peer_path)"
)
HELPER_SIG = "static bool ggml_sycl_stage_refuse_cross_card_expert_plan(const ggml_sycl::placement_plan * plan)"

REJECTED = "return GGML_SYCL_LIFECYCLE_PLAN_REJECTED; }"

# The late pass: per device, after the candidate is checked current and before that device's host zones.
LATE_AFTER = 'throw std::runtime_error("stale host provisioning candidate before allocation");'
LATE_CROSS = "if (ggml_sycl_stage_refuse_cross_card_expert_plan(snapshot->plan.get())) {"
LATE_LAZY = "if (ggml_sycl_stage_refuse_lazy_plan(snapshot->plan.get(), early)) {"
LATE_BEFORE = "ggml_sycl_configure_host_zones_for_plan("

# The early pass: on the candidate every device planned, before it is staged as the probe plan.
EARLY_AFTER = "lifecycle_find_candidate_placement_plan(effect.owner.load.value) : nullptr;"
EARLY_CROSS = "if (early_candidate && ggml_sycl_stage_refuse_cross_card_expert_plan(early_candidate->plan.get())) {"
EARLY_LAZY = "if (early_candidate && ggml_sycl_stage_refuse_lazy_plan(early_candidate->plan.get(), early)) {"
EARLY_BEFORE = "lifecycle_stage_probe_placement_plan("

PRED_BODY = "{ return ggml_sycl_moe_multi_gpu_wanted(plan) && !peer_path; }"
# The helper reads the peer fact for the cards the plan uses and hands it to the predicate.
PEER_FACT = "const bool peer_path = !plan->multi_device || ggml_sycl_cards_have_peer_path(plan->devices);"
PEER_FACT_ASKED = "if (!ggml_sycl_cross_card_expert_plan_refused(*plan, peer_path)) {"
REFUSAL_LINE = (
    '"[MOE-MULTI-GPU] load refused: the plan puts experts on a second card and the cards have no peer path; unset '
    'GGML_SYCL_MOE_MULTI_GPU to load with the single-card plan\\n"'
)
# The ERROR call with the line as its only argument; the formatter may break after the open parenthesis.
REFUSAL_CALL_RE = re.compile(r"GGML_LOG_ERROR\( ?" + re.escape(REFUSAL_LINE) + r" ?\);")
REMEDY = "; unset GGML_SYCL_MOE_MULTI_GPU to load with the single-card plan"


def claim_late_cross_card_before_lazy(src: str) -> bool:
    return ordered(fn_body(view(src), STAGE_SIG), LATE_AFTER, LATE_CROSS, LATE_LAZY, LATE_BEFORE)


def claim_early_cross_card_before_lazy(src: str) -> bool:
    return ordered(fn_body(view(src), STAGE_SIG), EARLY_AFTER, EARLY_CROSS, EARLY_LAZY, EARLY_BEFORE)


def claim_late_returns_plan_rejected(src: str) -> bool:
    return (LATE_CROSS + " " + REJECTED) in fn_body(view(src), STAGE_SIG)


def claim_early_returns_plan_rejected(src: str) -> bool:
    return (EARLY_CROSS + " " + REJECTED) in fn_body(view(src), STAGE_SIG)


def claim_predicate_takes_the_peer_fact(src: str) -> bool:
    return fn_body(view(src), PRED_SIG) == PRED_BODY


def claim_helper_passes_the_peer_fact(src: str) -> bool:
    return ordered(fn_body(view(src), HELPER_SIG), PEER_FACT, PEER_FACT_ASKED)


def claim_refusal_line_names_the_remedy(src: str) -> bool:
    body = fn_body(view(src), HELPER_SIG)
    return body.count("GGML_LOG_") == 1 and REFUSAL_CALL_RE.search(body) is not None


CLAIMS = {
    "late_cross_card_before_lazy": claim_late_cross_card_before_lazy,
    "early_cross_card_before_lazy": claim_early_cross_card_before_lazy,
    "late_returns_plan_rejected": claim_late_returns_plan_rejected,
    "early_returns_plan_rejected": claim_early_returns_plan_rejected,
    "predicate_takes_the_peer_fact": claim_predicate_takes_the_peer_fact,
    "helper_passes_the_peer_fact": claim_helper_passes_the_peer_fact,
    "refusal_line_names_the_remedy": claim_refusal_line_names_the_remedy,
}


def assert_only_claim_fails(claim: str, mutated: str) -> None:
    assert not CLAIMS[claim](mutated), f"the witness for {claim} does not make it fail"
    others = [name for name, fn in CLAIMS.items() if name != claim and not fn(mutated)]
    assert not others, f"the witness for {claim} also fails {others}"


def _block(cross_or_lazy: str) -> str:
    return cross_or_lazy + " " + REJECTED


# ---- the checks ----


def test_late_cross_card_before_lazy():
    assert claim_late_cross_card_before_lazy(SYCL), (
        "the late pass does not ask the cross-card refusal after the candidate check and before the lazy-plan "
        "refusal and the host-zone configuration"
    )


def test_early_cross_card_before_lazy():
    assert claim_early_cross_card_before_lazy(SYCL), (
        "the early pass does not ask the cross-card refusal on the candidate before the lazy-plan refusal and before "
        "the candidate is staged as the probe plan"
    )


def test_late_returns_plan_rejected():
    assert claim_late_returns_plan_rejected(SYCL), (
        "the late pass's cross-card refusal does not return GGML_SYCL_LIFECYCLE_PLAN_REJECTED"
    )


def test_early_returns_plan_rejected():
    assert claim_early_returns_plan_rejected(SYCL), (
        "the early pass's cross-card refusal does not return GGML_SYCL_LIFECYCLE_PLAN_REJECTED"
    )


def test_predicate_takes_the_peer_fact():
    assert claim_predicate_takes_the_peer_fact(SYCL), (
        f"ggml_sycl_cross_card_expert_plan_refused's body is not exactly {PRED_BODY!r}: it must take the peer fact "
        "as its argument and call no device query itself"
    )


def test_helper_passes_the_peer_fact():
    assert claim_helper_passes_the_peer_fact(SYCL), (
        f"the stage helper does not read {PEER_FACT!r} and pass it to the predicate, so the refusal does not read "
        "the peer-link records of the plan's cards"
    )


def test_refusal_line_names_the_remedy():
    assert claim_refusal_line_names_the_remedy(SYCL), (
        "the stage helper does not log exactly one line, the ERROR that names the refusal and its remedy"
    )


# ---- one witness per check ----


def test_witness_late_lazy_asked_first():
    assert_only_claim_fails(
        "late_cross_card_before_lazy",
        mutate_in(
            SYCL,
            STAGE_SIG,
            _block(LATE_CROSS) + " " + _block(LATE_LAZY),
            _block(LATE_LAZY) + " " + _block(LATE_CROSS),
        ),
    )


def test_witness_early_lazy_asked_first():
    assert_only_claim_fails(
        "early_cross_card_before_lazy",
        mutate_in(
            SYCL,
            STAGE_SIG,
            _block(EARLY_CROSS) + " " + _block(EARLY_LAZY),
            _block(EARLY_LAZY) + " " + _block(EARLY_CROSS),
        ),
    )


def test_witness_late_returns_another_code():
    assert_only_claim_fails(
        "late_returns_plan_rejected",
        mutate_in(
            SYCL,
            STAGE_SIG,
            _block(LATE_CROSS),
            LATE_CROSS + " return GGML_SYCL_LIFECYCLE_EFFECT_FAILED; }",
        ),
    )


def test_witness_early_returns_another_code():
    assert_only_claim_fails(
        "early_returns_plan_rejected",
        mutate_in(
            SYCL,
            STAGE_SIG,
            _block(EARLY_CROSS),
            EARLY_CROSS + " return GGML_SYCL_LIFECYCLE_EFFECT_FAILED; }",
        ),
    )


def test_witness_predicate_queries_the_cards():
    assert_only_claim_fails(
        "predicate_takes_the_peer_fact",
        mutate_in(SYCL, PRED_SIG, "&& !peer_path;", "&& !peer_path && !ggml_sycl_cards_have_peer_path(plan.devices);"),
    )


def test_witness_helper_assumes_a_peer_path():
    assert_only_claim_fails(
        "helper_passes_the_peer_fact",
        mutate_in(SYCL, HELPER_SIG, "ggml_sycl_cards_have_peer_path(plan->devices)", "true"),
    )


def test_witness_refusal_line_without_the_remedy():
    assert_only_claim_fails(
        "refusal_line_names_the_remedy",
        mutate_in(SYCL, HELPER_SIG, REMEDY, ""),
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
