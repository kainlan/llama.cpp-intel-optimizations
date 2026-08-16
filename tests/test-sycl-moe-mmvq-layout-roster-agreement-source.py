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

The broad enumeration check above only recognizes the exact {Q4_0, Q8_0}
shape. It is deliberately paired with a second, narrower check
(test_advertisement_and_refusal_sites_decide_layout_via_roster_only) that
asserts the SPECIFIC decision expression at each of the three fixed sites
contains no literal GGML_TYPE_* comparison at all -- catching a revert to a
hand-maintained list of ANY length or shape, not only the exact pre-fix one.
The two checks answer different questions ("does this exact old shape exist
anywhere" vs. "do these three decisions still delegate entirely to the
roster") and neither subsumes the other.

HONEST SCOPE STATEMENT for the narrower check (team-lead review, second
round): it inspects the TEXT of each site's if-condition -- recursively,
preferring the innermost nested if, so a hardcoded list nested under an
outer roster-derived if is found (this is what closes finding #2's original
gap: previously a NON-recursive scan returned the outer, misleadingly-clean
condition and missed a hardcoded list hiding one level deeper). It does NOT
resolve what an opaque identifier in that condition refers to. Two bypasses
follow directly from that limit and are accepted as out of scope, because
closing them needs real data-flow analysis, not a regex/brace scanner:
  (1) a hand-maintained list hoisted into a variable declared just before the
      if, with the if referencing the variable (e.g.
      `const bool aos_only = (t==Q4_0||t==Q8_0||t==Q6_K); if (roster_call(...)
      && aos_only) {...}`);
  (2) the same list hidden behind a helper function call in the condition.
Both require deliberately restructuring the decision, not merely editing it,
which is why they are accepted rather than chased. The bound: neither bypass
can hide the exact {Q4_0, Q8_0} shape from the BROAD enumeration check above,
which scans raw file text irrespective of if-statement scoping -- so a
2-type hoisted variable or helper is still an offender there even though
this narrower check cannot see through the indirection to it. Only a
hoisted/wrapped list that is NOT exactly {Q4_0, Q8_0} escapes both checks.

Not in scope, and deliberately so: llama.cpp-mn70's sibling consumer-side
guard (R1, landed as 011064e2b) is a RUNTIME comparison against the observed
storage layout (`get_effective_layout_mode(src0_extra) != layout`), not a
hand-maintained TYPE list -- it is not exposed to the "reverts to a stale
type roster" failure mode this gate exists to catch, so it is not asserted
here. Stating this so a later reader does not "unify" the two checks; they
protect different invariants (R1: consume what is materialized; R2: advertise
only what has kernels -- nkfc c-gga1).

Two in-tree exemplars show the R1/R2 shape this gate enforces, worth reading
before touching either checked function (zoly c-8e6f/c-pqrc): the
`!use_expert_cache` host-routing branch (ggml-sycl.cpp, near the expert-cache
dispatch) resets a non-AOS route layout back to AOS rather than carrying it
when the fallback path cannot honor it; and `ggml_sycl_select_moe_mmvq_layout`
itself, twelve lines below its own roster-derived carve-out, already does
the *consumer* half correctly for device-resident weights -- `final_layout =
get_effective_layout_mode(extra)` under `!host_weights` discards whatever the
selector chose and uses the storage layout instead. Neither exemplar is
itself part of this gate's assertions; they are cited so a maintainer reading
the three checked sites understands what "correct" already looks like
elsewhere in the same file.

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

The exemption is SITE-scoped (the get_optimal function body), not FILE-scoped
-- common.hpp is otherwise fully covered by the broad enumeration check, so a
future, unrelated hand-maintained carve-out landing anywhere else in that file
is still caught rather than silently inheriting a whole-file pass. (get_optimal
is the only site: get_with_override delegates to it and its own qtype
comparison is single-type Q4_0, not a {Q4_0, Q8_0} chain, so it never matches
find_carveouts on its own -- verified empirically below.)
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
# SITE-scoped (the get_optimal function body), not FILE-scoped -- see docstring.
COMMON_HPP = SYCL_DIR / "common.hpp"
DELIBERATELY_UNFIXED_SIGNATURE = "static layout_mode get_optimal("

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


def function_span(text: str, signature: str):
    """(start, end) of a full function body by brace matching, not a guessed
    byte span -- a span rots the moment the function grows past it and reports
    a false "roster call missing" instead of a real regression."""
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return start, position + 1
    raise ValueError(f"unclosed function body: {signature}")


def function_body(text: str, signature: str) -> str:
    start, end = function_span(text, signature)
    return text[start:end]


def matching_delimiter(text: str, opening: int, open_char: str, close_char: str) -> int:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == open_char:
            depth += 1
        elif text[position] == close_char:
            depth -= 1
            if depth == 0:
                return position
    raise ValueError(f"unclosed delimiter: {open_char}")


def if_statements(body: str):
    """Yield (condition_text, consequent_text) for every `if (...) { ... }` or
    `if (...) stmt;` in body, RECURSIVELY -- both top-level ifs and every if
    nested inside another if's consequent -- in a depth-first, parent-before-
    child order. A minimal brace/paren-balanced scanner in the same spirit as
    this repo's own test-sycl-supports-op-indexed-moe-source.py (braced_body /
    matching_delimiter), not a full C++ parser.

    Recursion is load-bearing, not cosmetic: an outer if's consequent_text
    contains the FULL TEXT of everything nested inside it, so a marker that
    actually lives inside a nested if also appears (as a substring) in every
    enclosing if's consequent. Without recursion, only the outer if is ever
    seen, and its (roster-derived, clean) condition is what a caller matching
    on "consequent contains marker" would wrongly return -- exactly the
    fail-open a team-lead review caught: a hand-maintained type list nested
    under an outer if that itself carries a roster call. find_decision_condition
    below is what turns "also visible" into "correctly selected": among all
    matches, it prefers the one with the SHORTEST consequent, which is always
    the innermost (a nested consequent is a strict substring of its parents').
    """
    position = 0
    while True:
        idx = body.find("if", position)
        if idx == -1:
            return
        before_ok = idx == 0 or not (body[idx - 1].isalnum() or body[idx - 1] == "_")
        after = idx + 2
        while after < len(body) and body[after].isspace():
            after += 1
        if not before_ok or after >= len(body) or body[after] != "(":
            position = idx + 2
            continue
        cond_close = matching_delimiter(body, after, "(", ")")
        condition = body[after + 1 : cond_close]
        rest = cond_close + 1
        while rest < len(body) and body[rest].isspace():
            rest += 1
        if rest < len(body) and body[rest] == "{":
            brace_close = matching_delimiter(body, rest, "{", "}")
            consequent = body[rest : brace_close + 1]
            position = brace_close + 1
        else:
            stmt_end = body.index(";", rest)
            consequent = body[rest : stmt_end + 1]
            position = stmt_end + 1
        yield condition, consequent
        # Recurse into this if's own consequent to find anything nested
        # inside it. This does not re-visit or double-yield: the recursive
        # call scans a separately-extracted substring, while the outer loop's
        # `position` above already advanced past this whole consequent in the
        # ORIGINAL body, so the two scans cover disjoint textual roles even
        # though one is a physical substring of the other.
        yield from if_statements(consequent)


def find_decision_condition(body: str, consequent_marker: str) -> str:
    """The condition of the INNERMOST if-statement whose consequent contains
    consequent_marker (searched recursively -- see if_statements). "Innermost"
    is the match with the shortest consequent text: a nested if's consequent
    is always a strict substring of every ancestor if's consequent that also
    contains the marker, so the shortest matching consequent is unambiguously
    the most deeply nested one, and picking it is what closes the nested-if
    fail-open (a hand-maintained list nested under an outer roster-derived
    if) rather than returning the outer, misleadingly-clean condition.

    Raising on zero matches is deliberate: no marker occurrence at all means
    the extraction itself is unsound for this body, not that the invariant
    holds."""
    matches = [(cond, cons) for cond, cons in if_statements(body) if consequent_marker in cons]
    if not matches:
        raise ValueError(f"expected at least one if-statement with {consequent_marker!r} in its consequent, found 0")
    innermost_condition, _ = min(matches, key=lambda pair: len(pair[1]))
    return innermost_condition


LITERAL_TYPE_RE = re.compile(r"\bGGML_TYPE_[A-Z0-9_]+\b")


def test_find_decision_condition_prefers_innermost_nested_if():
    """Unit-level proof that the nested-if fail-open is closed, independent of
    the real source. Constructs the exact hazard shape a team-lead review
    caught (a hardcoded list nested under an outer if that itself carries a
    roster call) and asserts find_decision_condition returns the INNER
    condition, not the outer, misleadingly-clean one."""
    nested = (
        "if (moe_mmvq_batched_dispatch_supports_type(src0->type)) {\n"
        "    if (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0 || "
        "src0->type == GGML_TYPE_Q6_K) {\n"
        "        resolved = GGML_LAYOUT_AOS;\n"
        "    }\n"
        "}\n"
    )
    condition = find_decision_condition(nested, "GGML_LAYOUT_AOS")
    assert "GGML_TYPE_Q6_K" in condition, (
        f"expected the INNER hardcoded-list condition, got the outer roster-derived one: {condition!r}"
    )
    assert not any(call in condition for call in ROSTER_CALLS), (
        f"the selected condition should be the inner hand-maintained list, not the outer roster call: {condition!r}"
    )

    # Negative companion: when the marker-bearing if is NOT nested (the normal
    # shape at all three real sites today), the single top-level match is
    # still returned correctly -- recursion finding zero additional matches
    # must not change behavior for the common case.
    flat = "if (moe_mmvq_any_dispatch_supports_layout(src0->type, resolved)) {\n    resolved = GGML_LAYOUT_AOS;\n}\n"
    flat_condition = find_decision_condition(flat, "GGML_LAYOUT_AOS")
    assert "moe_mmvq_any_dispatch_supports_layout" in flat_condition


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
    """Enumerate every remaining instance at commit time -- not a frozen list of four.

    The common.hpp exception is SITE-scoped (the get_optimal function body),
    not FILE-scoped: any carve-out elsewhere in that file is still an offender.
    """
    offenders = []
    for path in sycl_sources():
        text = strip_comments(path.read_text(encoding="utf-8"))
        exempt_span = None
        if path == COMMON_HPP:
            exempt_span = function_span(text, DELIBERATELY_UNFIXED_SIGNATURE)
        for match in find_carveouts(text):
            if exempt_span and exempt_span[0] <= match.start() < exempt_span[1]:
                continue
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)}")
    assert not offenders, (
        "hand-maintained {Q4_0, Q8_0}-style layout carve-out found outside "
        "layout_policy::get_optimal in common.hpp (which is deliberately left "
        "unfixed by llama.cpp-zoly per the layout-follows-residency ruling; "
        "see this file's module docstring) -- replace it with a roster call "
        "(moe_mmvq_any_dispatch_supports_layout / "
        "moe_mmvq_batched_dispatch_supports_type):\n" + "\n".join(offenders)
    )


def test_deliberately_unfixed_site_still_exists_and_is_named_correctly():
    """If get_optimal's pin is ever removed or moved, this gate's exception is stale."""
    text = strip_comments(COMMON_HPP.read_text(encoding="utf-8"))
    get_optimal_body = function_body(text, DELIBERATELY_UNFIXED_SIGNATURE)
    assert find_carveouts(get_optimal_body), (
        "layout_policy::get_optimal no longer carries a {Q4_0, Q8_0}-shaped "
        "layout carve-out -- if it was fixed, removed, or reworded, retire "
        "this gate's DELIBERATELY_UNFIXED_SIGNATURE exception (and its "
        "citation of the layout-follows-residency ruling) rather than leaving "
        "a stale carve-out"
    )

    # get_with_override delegates to get_optimal and must NOT independently
    # carry the {Q4_0, Q8_0} shape -- if it grew one, the site-scoped
    # exemption above would miss it (get_with_override is a separate function,
    # outside the exempted span) and it would correctly show up as an offender
    # in the broad enumeration test; this just makes that expectation explicit
    # rather than relying on the other test to notice by accident.
    override_body = function_body(text, "static layout_mode get_with_override(")
    assert not find_carveouts(override_body), (
        "get_with_override now independently carries a {Q4_0, Q8_0}-shaped "
        "carve-out -- it is outside the site-scoped exemption "
        "(DELIBERATELY_UNFIXED_SIGNATURE covers only get_optimal) and must "
        "either be removed or the exemption widened deliberately, with the "
        "docstring updated to say why"
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


def test_advertisement_and_refusal_sites_decide_layout_via_roster_only():
    """Stronger than 'a roster call appears somewhere in the function' (which
    the widened logging gate call alone would satisfy without discriminating
    anything -- team-lead review finding #2 on ccbd76ae5, the follow-up round
    to that commit): the SPECIFIC decision expression governing the
    carve-out/refusal must itself consult the roster and must contain no
    literal GGML_TYPE_* comparison. A revert to a hand-maintained list of ANY
    length or shape -- not only the exact {Q4_0, Q8_0} pair the broad
    enumeration check above recognizes -- is caught here, because this check
    is scoped to the decision itself rather than to "does a roster call exist
    anywhere in this function". c-gga1's extended-stopgap-becomes-permanent
    hazard is exactly a longer hardcoded list surviving unnoticed; that is
    what this test is for.
    """
    ggml_sycl_cpp = (SYCL_DIR / "ggml-sycl.cpp").read_text(encoding="utf-8")
    mmvq_cpp = (SYCL_DIR / "mmvq.cpp").read_text(encoding="utf-8")

    mmvq_layout_fn = strip_comments(function_body(ggml_sycl_cpp, "layout_mode ggml_sycl_select_moe_mmvq_layout("))
    expert_cache_fn = strip_comments(
        function_body(ggml_sycl_cpp, "static layout_mode ggml_sycl_select_moe_expert_cache_layout(")
    )
    mmvq_dispatch_fn = strip_comments(function_body(mmvq_cpp, "bool ggml_sycl_mul_mat_id_vec_q("))

    # (name, function body, the string marking the decision's consequent).
    # ggml_sycl_select_moe_mmvq_layout / ggml_sycl_select_moe_expert_cache_layout
    # both assign/return GGML_LAYOUT_AOS ONLY from their carve-out decision
    # (verified: no other branch in either function does); the mmvq.cpp
    # refusal is keyed on its trace-tag string, unique to that guard.
    sites = (
        ("ggml_sycl_select_moe_mmvq_layout", mmvq_layout_fn, "GGML_LAYOUT_AOS"),
        ("ggml_sycl_select_moe_expert_cache_layout", expert_cache_fn, "GGML_LAYOUT_AOS"),
        ("ggml_sycl_mul_mat_id_vec_q", mmvq_dispatch_fn, "quant_layout_unsupported"),
    )
    for name, body, marker in sites:
        condition = find_decision_condition(body, marker)
        assert any(call in condition for call in ROSTER_CALLS), (
            f"{name}'s layout decision no longer calls a moe-mmvq-tables.hpp "
            f"roster predicate in its condition -- did an edit revert it to a "
            f"hand-maintained type list? condition was: {condition!r}"
        )
        literal_types = LITERAL_TYPE_RE.findall(condition)
        assert not literal_types, (
            f"{name}'s layout decision condition still contains a literal "
            f"type comparison ({literal_types}) alongside (or instead of) the "
            f"roster call -- ANY hand-maintained type list here defeats the "
            f"fix, not just the exact {{Q4_0, Q8_0}} shape. condition was: "
            f"{condition!r}"
        )


if __name__ == "__main__":
    test_find_decision_condition_prefers_innermost_nested_if()
    test_positive_control_regex_is_sensitive()
    test_regex_does_not_over_match_longer_or_chains()
    test_no_hand_maintained_carveout_survives_outside_the_named_exception()
    test_deliberately_unfixed_site_still_exists_and_is_named_correctly()
    test_advertisement_and_refusal_sites_consult_the_roster()
    test_advertisement_and_refusal_sites_decide_layout_via_roster_only()
    print("test-sycl-moe-mmvq-layout-roster-agreement-source: OK")
