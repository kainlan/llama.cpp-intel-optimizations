"""Source contract for keeping a short ONEDNN zone when a rebuild of the VRAM arena is blocked.

A second load whose oneDNN Graph scratch floor needs a larger ONEDNN zone than the live arena has used to be refused
at the load stage's entry ensure: the zones grow only by a rebuild, and nothing is rebuilt while anything points into
the arena. The Graph floor is a reservation, not a requirement the load cannot run without (what the zone cannot hold
of it is served on the DIRECT path under its cap, and a context needing more is refused by name at the context), so
the blocked rebuild keeps the zones when the floor is all they lack. These checks pin where and how:

- place: unified_cache::ensure_planned_arena_zones() decides the keep inside its rebuild-blocked branch, before any
  refusal line, over the zones it read (arena_zones) and the sizes it decided on, and returns true when it keeps;
- ring: a PP MoE ring that alone holds the arena keeps nothing while the caller asked to release it first
  (ring_release_first); the call refuses and marks the refusal keep_deferred;
- rule: one cache function decides and stores the keep, under the zone-plan mutex, through the pure rules
  zone_onednn_plan_keep_short and zone_arena_shortfall_is_reservation_only (test-zone-sizing Case 41), and nothing
  else calls those rules;
- entry: the load stage's entry ensure asks with ring_release_first first, and every call after its last resort asks
  without it; a ring the release kept is asked about again only when the refusal says the zones would be kept.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read comment-stripped, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must fail only that check.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
UC = (SYCL / "unified-cache.cpp").read_text()
GS = (SYCL / "ggml-sycl.cpp").read_text()

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (kept)
    r"|'(?:\\.|[^'\\\n])*'"  # char literal (kept)
    r"|//[^\n]*"  # line comment (dropped)
    r"|/\*.*?\*/",  # block comment (dropped)
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def code(src: str) -> str:
    """Comment-stripped text with every run of whitespace made one space. Idempotent."""
    return re.sub(r"\s+", " ", strip_comments(src))


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
    s = text.find(sig)
    return block_at(text, s + len(sig) - 1) if s != -1 else ""


def ordered(text: str, *parts: str) -> bool:
    pos = 0
    for p in parts:
        k = text.find(p, pos)
        if k == -1:
            return False
        pos = k + len(p)
    return bool(text)


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` as code() with the one `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and body.count(old) == 1, f"witness anchor {old!r} is not once in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


ENSURE_SIG = (
    "bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
    "const zone_arena_ensure_options & options) {"
)
BLOCKED = "if (!zone_arena_rebuild_allowed(holders)) {"
KEEP_FN = "onednn_zone_plan_keep_short_and_store"
KEEP_SIG = f"static bool {KEEP_FN}("
NEED = "const zone_arena_sizes need = { scratch_zone, onednn_zone, runtime_zone };"
DEFER = "const bool defer_keep = options.ring_release_first && zone_arena_only_ring_holds(true, holders);"
HELD_PAIR = "const size_t held_pair = arena_held_onednn_pair_bytes();"
KEEP_CALL = f"const bool reservation_only = {KEEP_FN}(dev_id, live_plan, arena_zones, need, held_pair, !defer_keep, "
KEPT_RETURN = "if (reservation_only && !defer_keep) {"
DEFERRED_MARK = "refusal->keep_deferred = reservation_only && defer_keep;"
PLAN_LOCK = "std::lock_guard<std::mutex> lock(g_onednn_zone_plan_mutex);"
SHORT_RULE = "zone_onednn_plan_keep_short(g_onednn_zone_plan[device_id], live, have.onednn, held_pair_bytes)"
ONLY_RULE = (
    "zone_arena_shortfall_is_reservation_only(have, need, zone_onednn_kept_pair_bytes(short_plan, held_pair_bytes))"
)
STORE = "if (store) { g_onednn_zone_plan[device_id] = short_plan; }"
ENTRY_SIG = "static bool ggml_sycl_ensure_load_arena_zones(int device, ggml_sycl::zone_arena_refusal & refusal) {"
ENTRY_FIRST = "options.ring_release_first = true;"
ENTRY_SPENT = "options.ring_release_first = false;"
ENTRY_CALL = "ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options)"
ENTRY_KEPT_RING = "if (!refusal.keep_deferred) { return false; }"


def _ensure(uc: str) -> str:
    return fn_body(code(uc), ENSURE_SIG)


def claim_keep_sits_in_the_blocked_branch(uc: str) -> bool:
    """The keep is decided once, inside the rebuild-blocked branch, over the zones read and the sizes decided, and its
    true return comes before the branch's first refusal line."""
    ensure = _ensure(uc)
    if not ensure or ensure.count(KEEP_FN + "(") != 1 or BLOCKED not in ensure:
        return False
    blocked = block_at(ensure, ensure.index(BLOCKED) + len(BLOCKED) - 1)
    if KEEP_CALL not in blocked:
        return False
    first_error = blocked.find("GGML_LOG_ERROR(")
    return first_error > 0 and ordered(blocked[:first_error], NEED, HELD_PAIR, KEEP_CALL, KEPT_RETURN, "return true;")


def claim_ring_alone_defers_the_keep(uc: str) -> bool:
    """A lone ring with ring_release_first stores nothing and keeps nothing; the refusal says the keep was deferred."""
    ensure = _ensure(uc)
    if BLOCKED not in ensure:
        return False
    blocked = block_at(ensure, ensure.index(BLOCKED) + len(BLOCKED) - 1)
    return ordered(blocked, DEFER, KEEP_CALL, KEPT_RETURN) and blocked.count(DEFERRED_MARK) == 1


def claim_one_rule_decides_the_keep(uc: str, gs: str) -> bool:
    """The keep function applies both pure rules under the zone-plan mutex and stores only when asked; no other site
    applies them."""
    c = code(uc)
    body = fn_body(c, KEEP_SIG)
    if not ordered(body, PLAN_LOCK, SHORT_RULE, ONLY_RULE, STORE):
        return False
    for rule in ("zone_onednn_plan_keep_short(", "zone_arena_shortfall_is_reservation_only("):
        if c.count(rule) != 1 or rule in code(gs):
            return False
    return True


def claim_entry_spends_its_last_resort_first(gs: str) -> bool:
    """The entry asks first with ring_release_first, then turns it off before its two later calls, and asks again
    over a kept ring only when the refusal says the zones would be kept."""
    body = fn_body(code(gs), ENTRY_SIG)
    return (
        body.count(ENTRY_CALL) == 3
        and ordered(body, ENTRY_FIRST, ENTRY_CALL, ENTRY_SPENT, ENTRY_KEPT_RING, ENTRY_CALL, ENTRY_CALL)
        and body.count(ENTRY_FIRST) == 1
    )


def keep_checks(uc: str, gs: str) -> dict:
    return {
        "place": claim_keep_sits_in_the_blocked_branch(uc),
        "ring": claim_ring_alone_defers_the_keep(uc),
        "rule": claim_one_rule_decides_the_keep(uc, gs),
        "entry": claim_entry_spends_its_last_resort_first(gs),
    }


def only_fails(uc: str, gs: str, failing: str) -> None:
    results = keep_checks(uc, gs)
    assert results == {name: name != failing for name in results}, f"witness for {failing!r} is broken: {results}"


def test_the_keep_sits_in_the_blocked_branch():
    assert claim_keep_sits_in_the_blocked_branch(UC), (
        "ensure_planned_arena_zones() must decide the keep once, inside `if (!zone_arena_rebuild_allowed(holders))`, "
        "over arena_zones and the sizes it decided on, and return true before the branch's first refusal line"
    )


def test_a_ring_alone_defers_the_keep():
    assert claim_ring_alone_defers_the_keep(UC), (
        "a PP MoE ring that alone holds the arena must keep nothing while the caller asked to release it first, and "
        "the refusal must say the zones would have been kept (keep_deferred)"
    )


def test_one_rule_decides_the_keep():
    assert claim_one_rule_decides_the_keep(UC, GS), (
        f"{KEEP_FN}() must apply zone_onednn_plan_keep_short and zone_arena_shortfall_is_reservation_only under the "
        "zone-plan mutex and store only when asked, and no other site may apply them"
    )


def test_the_entry_spends_its_last_resort_first():
    assert claim_entry_spends_its_last_resort_first(GS), (
        "ggml_sycl_ensure_load_arena_zones() must ask first with ring_release_first, ask after its last resort "
        "without it, and ask again over a kept ring only when the refusal's keep_deferred is set"
    )


def test_witnesses():
    assert all(keep_checks(UC, GS).values()), f"sanity: every check passes on the source: {keep_checks(UC, GS)}"

    # The refusal line logged before the keep is decided: a kept load would still print that it was refused.
    only_fails(mutate_in(UC, ENSURE_SIG, NEED, 'GGML_LOG_ERROR("refused\\n"); ' + NEED), GS, "place")
    # The keep decided over the zones as they are rather than the sizes decided on: nothing would ever be short.
    only_fails(mutate_in(UC, ENSURE_SIG, NEED, "const zone_arena_sizes need = arena_zones;"), GS, "place")
    # The lone ring kept short before its release.
    only_fails(mutate_in(UC, ENSURE_SIG, DEFER, "const bool defer_keep = false;"), GS, "ring")
    # A deferred keep the caller is never told about: the kept ring would refuse a load it could keep.
    only_fails(mutate_in(UC, ENSURE_SIG, DEFERRED_MARK, ""), GS, "ring")
    # The maximum floor stored for a kept zone: the Graph room would read larger than the zone holds.
    only_fails(
        mutate_in(UC, KEEP_SIG, SHORT_RULE, "zone_onednn_plan_keep(g_onednn_zone_plan[device_id], live)"), GS, "rule"
    )
    # The keep stored even when the caller defers it.
    only_fails(mutate_in(UC, KEEP_SIG, STORE, "g_onednn_zone_plan[device_id] = short_plan;"), GS, "rule")
    # The predicate applied a second time, in the entry.
    second_rule = (
        ENTRY_KEPT_RING + " if (!ggml_sycl::zone_arena_shortfall_is_reservation_only(refusal.have, refusal.need, 0)) "
        "{ return false; }"
    )
    only_fails(UC, mutate_in(GS, ENTRY_SIG, ENTRY_KEPT_RING, second_rule), "rule")
    # The entry's later calls still asking to release the ring first: a kept ring would never be kept short.
    only_fails(UC, mutate_in(GS, ENTRY_SIG, ENTRY_SPENT, ""), "entry")
    # A kept ring asked about again whatever the cache said.
    only_fails(UC, mutate_in(GS, ENTRY_SIG, ENTRY_KEPT_RING, ""), "entry")
    # The first call already past the last resort: a lone ring would be kept short instead of released.
    only_fails(UC, mutate_in(GS, ENTRY_SIG, ENTRY_FIRST, ""), "entry")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
