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

HONEST SCOPE STATEMENT for the narrower check (team-lead review, now on its
fourth round; the earlier "false-safe" errors in this paragraph's second and
third drafts were themselves corrected by an independently-run 17-mutation
regression matrix, not by re-reading the code -- see the matrix's own
harness for the authoritative record). It inspects the TEXT of each site's
if-condition -- recursively into nested consequents, so a hardcoded list
nested under an outer roster-derived if is found (finding #2's fix: a
NON-recursive scan returned the outer, misleadingly-clean condition and
missed a hardcoded list hiding one level deeper). Among every marker-bearing
if found, it selects the innermost ONLY when all of them form a single
ancestor chain by INTERVAL containment (each match's [start, end) offset
strictly inside the next's, not merely textually a substring of it -- see
find_decision_condition's docstring for why offsets and not text); if two
matches are not in that relation -- e.g. two SIBLING ifs, one clean and one a
hand-maintained list, neither nested in the other, including two with
byte-identical or textually-overlapping consequents -- it raises rather than
picking one by a tiebreak (round 3 picked the shortest match unconditionally,
which silently resolved sibling ambiguity by comment-length coincidence; that
was fixed to a textual containment chain check, itself found unsound on the
identical/overlapping-text shapes and replaced with the interval form here).

The what-was-caught-when is itself informative and is recorded, not just
asserted: a 17-mutation matrix run against every prior gate revision confirms
each successive fix strictly added coverage over its predecessor and lost
none, with the sole exception below.

ONE limit category remains genuinely out of scope, for one reason:
IDENTIFIER OPACITY -- the check cannot resolve what an opaque identifier in a
condition refers to, so it cannot see through a hand-maintained list hoisted
into a variable referenced by the condition (`const bool aos_only =
(t==Q4_0||t==Q8_0||t==Q6_K); if (roster_call(...) && aos_only) {...}`), or
the same list hidden behind a helper function call in the condition. Both
require deliberately restructuring the decision, not merely editing it,
which is why they are accepted rather than chased -- closing them needs real
data-flow analysis, not a regex/brace scanner.

What is NOT in that category, corrected here because an earlier draft of
this paragraph got both wrong in the SAFE direction (understating the check,
which is the less dangerous mistake but still an inaccurate one): the check
is NOT `if`-shaped by architecture in the sense of refusing to look inside
other control-flow constructs. `if_statements` is a flat, structure-agnostic
scan for `if (` tokens anywhere in the text, so a marker-bearing decision
reached through an `else`/`else if` branch, or through a `for`/`while` loop
body, IS found and chain-checked exactly like a top-level one, AS LONG AS
the decision itself is shaped as an `if`. Verified, not assumed: the matrix
above scores `else if` (M5a), a nested `else { if (...) }` (M5b), and a
hardcoded array walked in a `for` loop guarded by an `if` (M5d) all RED.

The one thing a flat `if (` scan genuinely cannot find is a decision with NO
`if` token governing it at all -- a raw `switch` `case` label
(`case GGML_TYPE_Q4_0: resolved = GGML_LAYOUT_AOS; break;`, no "if"
anywhere) is the concrete instance the matrix caught (M5c, GREEN/blind on
every revision including this one), and it is DELIBERATELY not fixed here:
widening the scanner to walk `switch` bodies is a real, separate change that
deserves its own justification rather than riding this documentation and
offset fix. Filed as [ticket pending -- team lead to supply id]; reference it
here once filed. Until then, treat M5c/this gap as a KNOWN, NAMED exception,
not an oversight.

Consequence for the bound below: it holds for IDENTIFIER OPACITY, but does
NOT hold for the switch-case gap, and stating it as if it did would be the
exact kind of implied completeness this whole paragraph exists to avoid.
The bound: neither the hoisted-variable nor the helper-wrapper bypass can
hide the exact {Q4_0, Q8_0} shape from the BROAD enumeration check earlier in
this file, which scans raw file text irrespective of if-statement scoping or
identifiers -- so a 2-type hoisted/wrapped list is still an offender there
even though this narrower, decision-scoped check cannot see through the
indirection to it. The switch-case idiom is EXEMPT from that bound: `case
GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0:` uses `:`, not `==`, so OR_CHAIN_RE
(anchored on `==` specifically so it does not collide with switch-case
syntax -- see that regex's own comment) never matches it at ANY arity,
including the exact pre-fix shape. A switch-form carve-out therefore escapes
BOTH checks regardless of how many types it lists. This is not a hazard this
file quietly hopes goes unnoticed: it is the reason the ticket above exists.

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
# alongside others (e.g. "Q4_0 || Q8_0 || Q6_K") is not misread as this
# defect. See test_regex_does_not_over_match_longer_or_chains below for the
# full, current accounting of every such chain in-tree (by enclosing
# function, not a file list here that would just go stale the same way) --
# that accounting is the reason this needs to be a full-chain match, not a
# substring one. Deliberately anchored on "==" so it cannot match an unrelated `case
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


def if_statements(body: str, base_offset: int = 0):
    """Yield (condition_text, consequent_text, consequent_start, consequent_end)
    for every `if (...) { ... }` or `if (...) stmt;` in body, RECURSIVELY --
    both top-level ifs and every if nested inside another if's consequent --
    in a depth-first, parent-before-child order. consequent_start/end are
    ABSOLUTE offsets into the TOP-LEVEL body first passed to this function
    (base_offset threads that origin through recursion, since a recursive
    call operates on an extracted substring and would otherwise report
    positions relative to that substring instead). A minimal brace/paren-
    balanced scanner in the same spirit as this repo's own
    test-sycl-supports-op-indexed-moe-source.py (braced_body /
    matching_delimiter), not a full C++ parser.

    Recursion is load-bearing, not cosmetic: an outer if's consequent_text
    contains the FULL TEXT of everything nested inside it, so a marker that
    actually lives inside a nested if also appears (as a substring) in every
    enclosing if's consequent. Without recursion, only the outer if is ever
    seen, and its (roster-derived, clean) condition is what a caller matching
    on "consequent contains marker" would wrongly return -- exactly the
    fail-open a team-lead review caught: a hand-maintained type list nested
    under an outer if that itself carries a roster call.

    Offsets, not text, are what find_decision_condition below uses to decide
    "innermost" -- a second team-lead review round caught that TEXTUAL
    containment (`inner in outer`) is wrong on two adversarial shapes a
    textual check cannot tell apart from genuine nesting: two SIBLING
    consequents that happen to be byte-identical after comment-stripping
    (`a in b` is True when a == b, even though neither contains the other),
    and a short, braceless consequent that is an accidental TEXTUAL substring
    of an unrelated, differently-braced SIBLING consequent elsewhere in the
    body. INTERVAL containment (is [start,end) of one strictly inside
    [start,end) of the other) is exact where substring containment is not,
    because it is anchored to where the text actually IS, not merely what it
    reads as.
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
            consequent_start, consequent_end = rest, brace_close + 1
            position = brace_close + 1
        else:
            stmt_end = body.index(";", rest)
            consequent = body[rest : stmt_end + 1]
            consequent_start, consequent_end = rest, stmt_end + 1
            position = stmt_end + 1
        yield condition, consequent, base_offset + consequent_start, base_offset + consequent_end
        # Recurse into this if's own consequent to find anything nested
        # inside it. This does not re-visit or double-yield: the recursive
        # call scans a separately-extracted substring, while the outer loop's
        # `position` above already advanced past this whole consequent in the
        # ORIGINAL body, so the two scans cover disjoint textual roles even
        # though one is a physical substring of the other. base_offset keeps
        # the yielded positions anchored to the TOP-LEVEL body throughout.
        yield from if_statements(consequent, base_offset + consequent_start)


def find_decision_condition(body: str, consequent_marker: str) -> str:
    """The condition of the INNERMOST if-statement whose consequent contains
    consequent_marker (searched recursively -- see if_statements). "Innermost"
    is the match whose consequent INTERVAL -- not text -- is contained in
    every other match's interval, and that containment is verified explicitly
    before any match is trusted, rather than inferred from length.

    A nested if's consequent interval is always strictly inside every
    ANCESTOR if's consequent interval that also contains the marker; that is
    what makes "innermost = most contained" true, and it is a property of a
    chain of ancestors, not of an arbitrary set of matches. Two matches that
    stand in no containment relation at all -- e.g. two SIBLING if-statements,
    each independently containing the marker -- are not "close" and "far";
    they are simply ambiguous.

    A second team-lead review round is why this is INTERVAL containment
    (start/end offsets) and not TEXTUAL containment (`inner_text in
    outer_text`), which was this function's first fix-round-3 form and was
    itself found unsound on two adversarial shapes: two sibling consequents
    that are byte-IDENTICAL after comment-stripping (`a in b` is True when
    a == b, even though neither is nested in the other), and a short,
    braceless consequent that is an accidental textual substring of an
    unrelated, differently-shaped sibling elsewhere in the body. Offsets
    cannot be fooled by either: they are anchored to where the text actually
    lives, not to what it happens to read as.

    So: order every match by interval length (end - start), then verify each
    is INTERVAL-contained in the next-longer one (its [start, end) lies
    entirely within the next match's [start, end)). If that chain check
    fails, the ambiguity is real and must not be silently resolved by picking
    one -- raise, the same way zero matches raises. This guard existed as
    "raise on anything other than exactly one match" before nested-if support
    was added, and RELAXING it (many matches now allowed) to add nested-if
    coverage would have silently reopened a hole of the same class if it were
    not paired with this containment check: a hand-maintained list living in
    a SIBLING if next to the clean carve-out is not ambiguous-checked against
    the carve-out's condition unless this loop runs. Do not weaken this to
    "prefer the shortest" (by length OR by text) without re-deriving why that
    is still sound -- twice now, it wasn't.

    Raising on zero matches is unchanged and deliberate: no marker occurrence
    at all means the extraction itself is unsound for this body, not that the
    invariant holds."""
    matches = [
        (cond, start, end) for cond, cons, start, end in if_statements(body) if consequent_marker in cons
    ]
    if not matches:
        raise ValueError(f"expected at least one if-statement with {consequent_marker!r} in its consequent, found 0")
    ordered = sorted(matches, key=lambda m: m[2] - m[1])
    for (_, inner_start, inner_end), (_, outer_start, outer_end) in zip(ordered, ordered[1:]):
        if not (outer_start <= inner_start and inner_end <= outer_end):
            raise ValueError(
                f"{consequent_marker!r} occurs in two if-consequents whose intervals are not nested in "
                f"one another (offsets [{inner_start},{inner_end}) vs [{outer_start},{outer_end})) -- the "
                f"decision is ambiguous, so 'innermost' is undefined. This body needs a more specific "
                f"marker, or the two matching if-statements need to be reconciled by hand before this "
                f"check can trust either one."
            )
    innermost_condition, _, _ = ordered[0]
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


def test_find_decision_condition_raises_on_sibling_ambiguity():
    """Unit-level proof of the second team-lead finding: closing the nested-if
    hole by simply picking the shortest marker-bearing consequent (without a
    chain check) opens a SIBLING hole of the same severity. A hand-maintained
    list living in a second, sibling if -- next to the clean, roster-derived
    carve-out, not nested inside or around it -- must not be silently resolved
    by "shortest wins"; it must raise, because the two matches are genuinely
    ambiguous, not near-and-far.

    This is the reviewer's realistic mutation shape: a clean carve-out
    immediately followed by a second if, unrelated in nesting, that also
    assigns GGML_LAYOUT_AOS from a hardcoded list -- e.g. a well-intentioned
    "stopgap until the Q6_K launcher lands" someone adds next to, rather than
    inside, the roster-derived decision.
    """
    sibling = (
        "if (moe_mmvq_any_dispatch_supports_layout(src0->type, resolved)) {\n"
        "    resolved = GGML_LAYOUT_AOS;\n"
        "}\n"
        "if (src0->type == GGML_TYPE_Q6_K) {\n"
        "    // stopgap until the Q6_K launcher lands\n"
        "    resolved = GGML_LAYOUT_AOS;\n"
        "}\n"
    )
    try:
        find_decision_condition(sibling, "GGML_LAYOUT_AOS")
        raised = False
    except ValueError:
        raised = True
    assert raised, "two sibling if-statements both matching the marker must raise, not silently pick one"

    # Negative companion: the nested case (a genuine ancestor chain) must
    # still resolve, not raise -- the chain check must not over-fire on real
    # nesting, only on an actual sibling ambiguity.
    nested = (
        "if (moe_mmvq_batched_dispatch_supports_type(src0->type)) {\n"
        "    if (src0->type == GGML_TYPE_Q6_K) {\n"
        "        resolved = GGML_LAYOUT_AOS;\n"
        "    }\n"
        "}\n"
    )
    condition = find_decision_condition(nested, "GGML_LAYOUT_AOS")
    assert "GGML_TYPE_Q6_K" in condition


def test_find_decision_condition_uses_offsets_not_text_for_containment():
    """Unit-level proof of the third team-lead finding: TEXTUAL containment
    (`inner in outer`) is unsound on two adversarial shapes that INTERVAL
    containment (start/end offsets) resolves correctly. Both are constructed
    to defeat a textual check specifically, not just to exercise the sibling
    path generically."""
    # Adversarial shape 1: two SIBLING consequents that are byte-identical.
    # `"resolved = GGML_LAYOUT_AOS;" in "resolved = GGML_LAYOUT_AOS;"` is True
    # (a string is always "contained in" an identical copy of itself), so a
    # textual check would misread this as one nested inside the other and
    # silently pick either -- exactly the ambiguity the chain check exists to
    # catch, wearing a disguise a length/text comparison cannot see through.
    identical_siblings = (
        "if (moe_mmvq_any_dispatch_supports_layout(src0->type, resolved)) {\n"
        "    resolved = GGML_LAYOUT_AOS;\n"
        "}\n"
        "if (src0->type == GGML_TYPE_Q6_K) {\n"
        "    resolved = GGML_LAYOUT_AOS;\n"
        "}\n"
    )
    try:
        find_decision_condition(identical_siblings, "GGML_LAYOUT_AOS")
        raised = False
    except ValueError:
        raised = True
    assert raised, "byte-identical sibling consequents must raise, not be silently treated as nested"

    # Adversarial shape 2: a short, BRACELESS clean consequent whose text is
    # an accidental substring of an unrelated, differently-braced sibling's
    # consequent -- textually "contained" (substring), but not intervally
    # contained (disjoint byte ranges). A textual check would pick the clean
    # (braceless) one as "innermost" merely because it is a substring by
    # coincidence, not because it is structurally nested in anything.
    braceless_substring_hazard = (
        "if (moe_mmvq_any_dispatch_supports_layout(src0->type, resolved))\n"
        "    resolved = GGML_LAYOUT_AOS;\n"
        "if (src0->type == GGML_TYPE_Q6_K) {\n"
        "    resolved = GGML_LAYOUT_AOS;\n"
        "}\n"
    )
    try:
        find_decision_condition(braceless_substring_hazard, "GGML_LAYOUT_AOS")
        raised = False
    except ValueError:
        raised = True
    assert raised, "a braceless consequent that is a textual substring of an unrelated sibling must still raise"


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

    Nine such superset chains exist at HEAD (c1f4504c8): dmmv.cpp (2),
    getrows.cpp (1), mmq.cpp (1), and ggml-sycl.cpp (5) -- each legitimately
    ORs Q4_0/Q8_0 together with one or more OTHER types to answer a question
    broader than, and unrelated to, this ticket's AoS-only pin. Cited by
    enclosing function rather than line number, since line numbers here have
    already been observed to drift by dozens of lines between two commits in
    one day -- a frozen list would just manufacture the next miscount. The
    five in ggml-sycl.cpp, and they are NOT all the same question (do not
    over-generalize the shape): `onednn_pp_unified_scratch_enabled` (oneDNN
    PP scratch-buffer sizing heuristic), `ggml_backend_sycl_buffer_init_tensor`
    and `ggml_backend_sycl_buffer_set_tensor` (CPU-dispatch SoA-reorder-on-
    upload eligibility), `ggml_sycl_mul_mat` (dispatch routing), and
    `ggml_backend_sycl_device_supports_op` (capability query). A matcher that
    flags the Q4_0/Q8_0 substring inside any of these chains produces false
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
    test_find_decision_condition_raises_on_sibling_ambiguity()
    test_find_decision_condition_uses_offsets_not_text_for_containment()
    test_positive_control_regex_is_sensitive()
    test_regex_does_not_over_match_longer_or_chains()
    test_no_hand_maintained_carveout_survives_outside_the_named_exception()
    test_deliberately_unfixed_site_still_exists_and_is_named_correctly()
    test_advertisement_and_refusal_sites_consult_the_roster()
    test_advertisement_and_refusal_sites_decide_layout_via_roster_only()
    print("test-sycl-moe-mmvq-layout-roster-agreement-source: OK")
