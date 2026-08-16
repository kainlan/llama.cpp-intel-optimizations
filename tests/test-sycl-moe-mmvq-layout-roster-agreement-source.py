#!/usr/bin/env python3
"""Source contract: MoE MMVQ-ID layout advertisement/refusal must be roster-derived.

llama.cpp-zoly / llama.cpp-nkfc (R2, producer half): a route must not advertise a
(type, layout) pair for which no indexed kernel exists, anchored on the launcher
roster in moe-mmvq-tables.hpp -- never a hand-maintained copy of it. R2's
complementary form (refuse a still-named unsupported pair rather than downgrade
it) is the same roster consulted in the opposite direction; see nkfc c-32rt /
c-vcn5 for why the two forms are not interchangeable.

This gate does NOT re-implement the roster or re-derive per-(type, layout)
admissibility -- that is tests/test-sycl-moe-mmvq-tables.cpp, a compiled gate
over the header itself. This gate checks a different, STRUCTURAL property: that
the advertisement sites (ggml_sycl_select_moe_mmvq_layout,
ggml_sycl_select_moe_expert_cache_layout) and the refusal site
(ggml_sycl_mul_mat_id_vec_q's AoS-only guard) each consult the roster by CALLING
it, rather than re-spelling a {Q4_0, Q8_0}-shaped type list by hand -- and that
no OTHER hand-maintained copy of that predicate has crept in anywhere else in
the SYCL backend.

Per nkfc c-gga1 / c-3x8d: the anchor is (type, layout) ONLY, never device.
Kernel existence is a compile-time, device-agnostic fact (every predicate in
moe-mmvq-tables.hpp takes no device parameter); per-device optimality is a
different layer (ggml_sycl_moe_query_route_capability / layout_policy) and is
not this gate's job to assert.

Sites are located by CONTENT, not a frozen line list or a frozen count of four:
a regex enumerates every remaining hand-maintained instance of the carve-out
shape at commit time (zoly found a 4th, unaudited site -- mmvq.cpp:20376 --by
searching the pattern rather than re-checking known locations; see nkfc
c-32rt). A future copy-paste instance is caught by the same regex, not missed
because it wasn't on a list.

Scope note (owner ruling, "layout follows residency" --
docs/backend/sycl-memory-design.md): ggml/src/ggml-sycl/common.hpp's
layout_policy::get_optimal / get_with_override also carries a {Q4_0, Q8_0} pin,
and is DELIBERATELY left alone by this ticket's fix (nkfc c-vcn5). It feeds
STORAGE materialization as well as advertisement, and is upstream of the two
advertisement sites this gate does cover; a downstream AOS decision cannot be
widened back out (ggml_sycl_adjust_layout_for_tensor only ever moves AOS ->
non-AOS under a non-AOS-target guard). Constraining storage by kernel
availability would invert the ruling that weights are materialized in the
layout optimal for the residency they are in -- kernel availability is the
right authority for ADVERTISEMENT, not for STORAGE. Do not "fix" that site by
widening this gate to common.hpp; the endgame item (a layout-aware Q6_K _id
launcher, generalized) is tracked separately, post-merge, per the ruling.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL_DIR = ROOT / "ggml/src/ggml-sycl"

# The pre-fix hand-maintained shape: a boolean OR-chain of type comparisons
# whose full set of compared types is EXACTLY {Q4_0, Q8_0} -- no more, no
# fewer. Matched as a full maximal chain (not a 2-of-N substring) so a
# DIFFERENT, legitimate invariant that happens to mention the same two types
# alongside others (e.g. "Q4_0 || Q8_0 || Q6_K", a reorder-capability check
# used by dmmv.cpp/getrows.cpp/mmq.cpp) is not misread as this defect. See
# test_regex_does_not_over_match_longer_or_chains below -- those three files
# are the reason this needs to be a full-chain match, not a substring one.
# Deliberately anchored on "==" so it cannot match an unrelated `case
# GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0:` switch label pair, which uses ":" and
# is a different (legitimate) shape entirely.
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*(?:(?:->|\.)[A-Za-z_][A-Za-z0-9_]*)*"
_TYPE_CMP = rf"{_IDENT}\s*==\s*GGML_TYPE_[A-Z0-9_]+"
OR_CHAIN_RE = re.compile(rf"(?:{_TYPE_CMP}\s*\|\|\s*)+{_TYPE_CMP}")


def chain_types(chain_text: str):
    return re.findall(r"GGML_TYPE_([A-Z0-9_]+)", chain_text)


def find_carveouts(text: str):
    """Full OR-chains whose type set is EXACTLY {Q4_0, Q8_0}."""
    matches = []
    for m in OR_CHAIN_RE.finditer(text):
        types = chain_types(m.group(0))
        if len(types) == 2 and set(types) == {"Q4_0", "Q8_0"}:
            matches.append(m)
    return matches

# nkfc c-vcn5: deliberately not fixed by this ticket. See the module docstring.
DELIBERATELY_UNFIXED = {SYCL_DIR / "common.hpp"}

ROSTER_CALLS = (
    "moe_mmvq_any_dispatch_supports_layout(",
    "moe_mmvq_batched_dispatch_supports_type(",
    "moe_mmvq_batched_dispatch_supports_layout(",
)


def sycl_sources():
    for path in sorted(SYCL_DIR.glob("*.cpp")) + sorted(SYCL_DIR.glob("*.hpp")):
        yield path


def strip_comments(text: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_block)


def function_body(text: str, signature: str) -> str:
    """Extract a full function body by brace matching, not a guessed byte span --
    a span rots the moment the function grows past it and reports a false
    "roster call missing" instead of a real regression."""
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    raise ValueError(f"unclosed function body: {signature}")


def test_positive_control_regex_is_sensitive():
    """A regex that has never been observed to fire is not evidence it works."""
    injected = "if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0) { resolved = GGML_LAYOUT_AOS; }"
    assert find_carveouts(injected), "control premise broken: the carve-out regex must fire on its own target shape"

    injected_reordered = "if (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) { return GGML_LAYOUT_AOS; }"
    assert find_carveouts(injected_reordered), "control premise broken: type-order variant must also fire"

    injected_bare_ident = "if (qtype == GGML_TYPE_Q4_0 || qtype == GGML_TYPE_Q8_0) { return GGML_LAYOUT_AOS; }"
    assert find_carveouts(injected_bare_ident), "control premise broken: non-arrow identifiers must also fire"

    # Negative companion: a switch-case pair is a different, legitimate shape
    # (no "=="), and must NOT be flagged by the same regex.
    case_pair = "case GGML_TYPE_Q4_0:\ncase GGML_TYPE_Q8_0:\n    return true;"
    assert not find_carveouts(case_pair), "regex over-matches a switch-case label pair, which is not this defect"


def test_regex_does_not_over_match_longer_or_chains():
    """A 2-of-N substring of a longer chain is a DIFFERENT invariant, not this one.

    dmmv.cpp, getrows.cpp, ggml-sycl.cpp (onednn scratch + CPU-dispatch reorder)
    and mmq.cpp all legitimately OR Q4_0/Q8_0 together with one or more OTHER
    types to answer "is reorder/coalesced capable", a broader and already-
    generalized question unrelated to this ticket's AoS-only pin. A matcher
    that flags the Q4_0/Q8_0 substring inside those chains produces false
    positives on healthy code -- this is a negative control against that,
    built from the real shapes found in-tree rather than an invented example.
    """
    three_way = "if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q6_K) {"
    assert not find_carveouts(three_way), "a 3-type OR-chain must not be read as the 2-type carve-out"

    five_way = (
        "bool type_ok = (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q8_0 || "
        "tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q6_K || tensor->type == GGML_TYPE_MXFP4);"
    )
    assert not find_carveouts(five_way), "a 5-type OR-chain must not be read as the 2-type carve-out"


def test_no_hand_maintained_carveout_survives_outside_the_named_exception():
    """Enumerate every remaining instance at commit time -- not a frozen list of four."""
    offenders = []
    for path in sycl_sources():
        if path in DELIBERATELY_UNFIXED:
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        for match in find_carveouts(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)}")
    assert not offenders, (
        "hand-maintained {Q4_0, Q8_0}-style layout carve-out found outside "
        "common.hpp (which is deliberately left unfixed by llama.cpp-zoly per "
        "the layout-follows-residency ruling; see this file's module "
        "docstring) -- replace it with a roster call "
        "(moe_mmvq_any_dispatch_supports_layout / "
        "moe_mmvq_batched_dispatch_supports_type):\n" + "\n".join(offenders)
    )


def test_deliberately_unfixed_site_still_exists_and_is_named_correctly():
    """If common.hpp's pin is ever removed or moved, this gate's exception is stale."""
    common_hpp = SYCL_DIR / "common.hpp"
    text = strip_comments(common_hpp.read_text(encoding="utf-8"))
    assert find_carveouts(text), (
        "common.hpp no longer carries a {Q4_0, Q8_0}-shaped layout carve-out -- "
        "if it was fixed, removed, or reworded, retire this gate's "
        "DELIBERATELY_UNFIXED exception (and its citation of the "
        "layout-follows-residency ruling) rather than leaving a stale carve-out"
    )


def test_advertisement_and_refusal_sites_consult_the_roster():
    """The three sites this ticket fixed must call the roster, not re-list types."""
    ggml_sycl_cpp = (SYCL_DIR / "ggml-sycl.cpp").read_text(encoding="utf-8")
    mmvq_cpp = (SYCL_DIR / "mmvq.cpp").read_text(encoding="utf-8")

    mmvq_layout_fn = function_body(ggml_sycl_cpp, "layout_mode ggml_sycl_select_moe_mmvq_layout(")
    expert_cache_fn = function_body(ggml_sycl_cpp, "static layout_mode ggml_sycl_select_moe_expert_cache_layout(")
    mmvq_dispatch_fn = function_body(mmvq_cpp, "bool ggml_sycl_mul_mat_id_vec_q(")

    for name, body in (
        ("ggml_sycl_select_moe_mmvq_layout", mmvq_layout_fn),
        ("ggml_sycl_select_moe_expert_cache_layout", expert_cache_fn),
        ("ggml_sycl_mul_mat_id_vec_q", mmvq_dispatch_fn),
    ):
        assert any(call in body for call in ROSTER_CALLS), (
            f"{name} no longer calls a moe-mmvq-tables.hpp roster predicate -- "
            "did an edit revert it to a hand-maintained type list?"
        )
        assert not find_carveouts(strip_comments(body)), (
            f"{name} still contains a hand-maintained {{Q4_0, Q8_0}}-shaped "
            "carve-out alongside (or instead of) the roster call"
        )


if __name__ == "__main__":
    test_positive_control_regex_is_sensitive()
    test_regex_does_not_over_match_longer_or_chains()
    test_no_hand_maintained_carveout_survives_outside_the_named_exception()
    test_deliberately_unfixed_site_still_exists_and_is_named_correctly()
    test_advertisement_and_refusal_sites_consult_the_roster()
    print("test-sycl-moe-mmvq-layout-roster-agreement-source: OK")
