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
  zone_onednn_plan_keep_short and zone_arena_shortfall_is_reservation_only (test-zone-sizing Case 41); nothing else
  applies the predicate, and the only other site of zone_onednn_plan_keep_short is a failed load's put-back (restore);
- entry: the load stage's entry ensure asks with ring_release_first first, and every call after its last resort asks
  without it; a ring the release kept is asked about again only when the refusal says the zones would be kept;
- cap: the DIRECT Graph-scratch cap's budget snapshot does not rise while the arena is kept. The sufficient and keep
  paths store min(held, current) through one compare-exchange helper over zone_graph_direct_cap_snapshot_kept
  (test-zone-sizing Case 42), and only a rebuild stores the current budget. A snapshot that rose across tenants would
  grow the Graph floor of a kept zone and ask for a rebuild the zone does not need;
- warn: a keep logs one WARN naming the planned Graph floor, the zone kept and the Graph room it leaves, behind a
  latch (zone_kept_onednn_warn_due, Case 43) that exchanges the load transaction in: once per load and device, since
  the stage asks for the zones in two passes and again in its grow leg, and at every keep for a direct entry (0);
- restore: the zone-plan snapshot IS a term a failed load puts back. The keep and the sufficient path store a
  description (the pair plan and the Graph floor) and move no memory, so a refused load that left its own description
  would leave the live model a smaller Graph room for nothing. The published-terms snapshot reads it as its own group
  (differing() and take_groups() carry its arm), and the restore stores zone_onednn_plan_keep_short(prior, prior,
  capacity now, held pair now) under the zone-plan mutex only while the snapshot still holds what the load left
  (test-zone-sizing Case 44). The DIRECT cap snapshot is not put back (it describes the arena as it stands, and a
  failed load that rebuilt the arena leaves the rebuilt arena in place), so the published-terms snapshot does not read
  it, and the keep path writes no planner-published term;
- log: the DIRECT-failure ERROR and the plan's "[SYCL-PLAN] oneDNN Graph-scratch zone floor:" line keep their fields
  and append, through one helper, the Graph room the stored zone plan holds whenever it is below the formula floor, so
  neither claims a kept zone holds the floor.

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
SNAPSHOT = "onednn_graph_scratch_direct_cap_plan_snapshot_bytes_"
CAP_FN = "onednn_graph_direct_cap_snapshot_keep"
CAP_SIG = f"void unified_cache::{CAP_FN}() {{"
CAP_CURRENT = "const size_t current = available_budget();"
CAP_HELD = f"size_t held = {SNAPSHOT}.load(std::memory_order_acquire);"
CAP_RULE = "zone_graph_direct_cap_snapshot_kept(held, current)"
CAP_LOOP = (f"while (!{SNAPSHOT}.compare_exchange_weak( held, {CAP_RULE}, std::memory_order_acq_rel, "
            "std::memory_order_acquire)) {")
CAP_KEEP = f"{CAP_FN}();"
SUFFICIENT = "if (zones_sufficient) {"
REBUILD_STORE = f"{SNAPSHOT}.store(available_budget(), std::memory_order_release);"
REBUILD_TAIL = "onednn_zone_plan_store(dev_id, live_plan);"
KEPT_DECL = "zone_onednn_plan kept;"
KEEP_INTO_KEPT = KEEP_CALL + "&kept);"
LATCH_READ = "const uint64_t latched_txn = onednn_zone_kept_warn_txn_.exchange(options.load_txn);"
LATCH = "zone_kept_onednn_warn_due(latched_txn, options.load_txn)"
KEPT_WARN = "GGML_LOG_WARN("
KEPT_WARN_TEXT = '"[VRAM-ARENA] device %d keeps its ONEDNN zone of %.1f MB: '
KEPT_WARN_ARGS = ("dev_id, arena_zones.onednn / (1024.0 * 1024.0), live_plan.graph_floor_bytes / (1024.0 * 1024.0), "
                  "onednn_zone / (1024.0 * 1024.0), kept.graph_floor_bytes / (1024.0 * 1024.0));")
TERMS_SIG = "load_published_terms load_published_terms::snapshot(int device) {"
PUBLISHED_WRITES = ("unified_cache_set_planned_", "g_planned_")
ZONE_GROUP = "LOAD_PUBLISHED_GROUP_ZONE_PLAN"
TERMS_READ = "terms.onednn_zone_plan = onednn_zone_plan_load(device);"
DIFFERING_SIG = "uint32_t load_published_terms::differing(const load_published_terms & other) const {"
DIFFERING_ARM = f"bit(!zone_onednn_plan_same(onednn_zone_plan, other.onednn_zone_plan), {ZONE_GROUP})"
TAKE_SIG = "void load_published_terms::take_groups(const load_published_terms & from, uint32_t groups) {"
TAKE_ARM = f"if (groups & {ZONE_GROUP}) {{ onednn_zone_plan = from.onednn_zone_plan; }}"
RESTORE_SIG = "void load_published_record::restore(int device) const noexcept {"
RESTORE_ARM = f"if (back & {ZONE_GROUP}) {{"
RESTORE_CAPACITY = "cache->zone_capacity(vram_zone_id::ONEDNN)"
RESTORE_HELD = "cache->arena_held_onednn_pair_bytes()"
PUT_BACK_FN = "onednn_zone_plan_put_back"
PUT_BACK_SIG = f"static void {PUT_BACK_FN}("
PUT_BACK_CALL = f"{PUT_BACK_FN}(device, written.onednn_zone_plan, prior.onednn_zone_plan, capacity, held);"
PUT_BACK_TEST = "if (zone_onednn_plan_same(g_onednn_zone_plan[device_id], written)) {"
PUT_BACK_RULE = "zone_onednn_plan_keep_short(prior, prior, capacity_bytes, held_pair_bytes)"
PUT_BACK_STORE = f"g_onednn_zone_plan[device_id] = {PUT_BACK_RULE};"
SHORT_FN = "onednn_zone_plan_short_text"
SHORT_SIG = f"static std::string {SHORT_FN}(int device_id, size_t formula_bytes) {{"
SHORT_READ = "const zone_onednn_plan stored = onednn_zone_plan_load(device_id);"
SHORT_TEST = (
    "if (zone_onednn_plan_same(stored, zone_onednn_plan()) || stored.graph_floor_bytes >= formula_bytes) { "
    "return std::string(); }"
)
SHORT_PRINT = "stored.graph_floor_bytes / (1024.0 * 1024.0)"
DIRECT_SIG = "void * unified_cache::onednn_graph_scratch_alloc_direct_locked("
DIRECT_FLOOR = "const size_t formula_floor = onednn_graph_scratch_zone_floor_bytes_swa("
DIRECT_TEXT = "n_ctx=%u%s). Refusing"
DIRECT_ARG = f"{SHORT_FN}(req.device, formula_floor).c_str());"
POPULATE_SIG = "static void populate_host_zone_sizing(placement_plan & plan,"
FLOOR_VALUE = "const size_t graph_floor = onednn_graph_scratch_zone_floor_bytes_swa("
FLOOR_PREFIX = '"[SYCL-PLAN] oneDNN Graph-scratch zone floor: %.1f MB (n_head_ctx_max=%u n_head_swa_max=%u n_swa=%u "'
FLOOR_TEXT = '"n_ubatch=%u n_ctx=%u)%s\\n",'
FLOOR_ARG = f"{SHORT_FN}(plan.device_id, graph_floor).c_str());"


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
    """The keep function applies both pure rules under the zone-plan mutex and stores only when asked. No other site
    applies the predicate, and zone_onednn_plan_keep_short is applied elsewhere only by a failed load's put-back."""
    c = code(uc)
    body = fn_body(c, KEEP_SIG)
    if not ordered(body, PLAN_LOCK, SHORT_RULE, ONLY_RULE, STORE):
        return False
    only = "zone_arena_shortfall_is_reservation_only("
    if c.count(only) != 1 or only in code(gs):
        return False
    short = "zone_onednn_plan_keep_short("
    return body.count(short) == 1 and c.count(short) == 1 + fn_body(c, PUT_BACK_SIG).count(short) and short not in code(gs)


def claim_entry_spends_its_last_resort_first(gs: str) -> bool:
    """The entry asks first with ring_release_first, then turns it off before its two later calls, and asks again
    over a kept ring only when the refusal says the zones would be kept."""
    body = fn_body(code(gs), ENTRY_SIG)
    return (
        body.count(ENTRY_CALL) == 3
        and ordered(body, ENTRY_FIRST, ENTRY_CALL, ENTRY_SPENT, ENTRY_KEPT_RING, ENTRY_CALL, ENTRY_CALL)
        and body.count(ENTRY_FIRST) == 1
    )


def claim_cap_snapshot_rises_only_at_a_rebuild(uc: str) -> bool:
    """The DIRECT cap's budget snapshot has three stores, all in the ensure: the sufficient path and the keep path
    store min(held, current) through one compare-exchange helper (0 is unset), and the rebuild stores current. No
    other site stores it."""
    c = code(uc)
    ensure = _ensure(uc)
    helper = fn_body(c, CAP_SIG)
    if not ordered(helper, CAP_CURRENT, CAP_HELD, CAP_LOOP) or c.count("zone_graph_direct_cap_snapshot_kept(") != 1:
        return False
    if SUFFICIENT not in ensure or BLOCKED not in ensure:
        return False
    sufficient = block_at(ensure, ensure.index(SUFFICIENT))
    blocked = block_at(ensure, ensure.index(BLOCKED))
    kept = block_at(blocked, blocked.find(KEPT_RETURN)) if KEPT_RETURN in blocked else ""
    return (
        CAP_KEEP in sufficient
        and SNAPSHOT not in sufficient
        and ordered(kept, CAP_KEEP, "return true;")
        and SNAPSHOT not in kept
        and ensure.count(CAP_KEEP) == 2
        and ensure.count(SNAPSHOT) == 1
        and ordered(ensure, BLOCKED, REBUILD_STORE, REBUILD_TAIL)
        and c.count(SNAPSHOT + ".store(") == 1
    )


def _kept_block(uc: str) -> str:
    ensure = _ensure(uc)
    if BLOCKED not in ensure:
        return ""
    blocked = block_at(ensure, ensure.index(BLOCKED))
    return block_at(blocked, blocked.find(KEPT_RETURN)) if KEPT_RETURN in blocked else ""


def claim_a_keep_warns_once_per_load(uc: str) -> bool:
    """The keep fills `kept` and, when the latch says so, logs one WARN naming the planned floor, the zone kept and the
    Graph room it leaves (the kept floor), before the keep returns true. The latch exchanges the load transaction in,
    so the stage's two passes and its grow leg warn once per load and device, and a direct entry (0) at every keep."""
    ensure = _ensure(uc)
    kept = _kept_block(uc)
    return (
        ordered(ensure, KEPT_DECL, KEEP_INTO_KEPT)
        and ordered(kept, LATCH_READ, "if (" + LATCH + ") {", KEPT_WARN, KEPT_WARN_TEXT, KEPT_WARN_ARGS, "return true;")
        and kept.count(KEPT_WARN) == 1
        and code(uc).count("onednn_zone_kept_warn_txn_") == 1
    )


def claim_a_failed_load_puts_the_zone_plan_back(uc: str) -> bool:
    """The zone-plan snapshot is one group of a load's published record: the published-terms snapshot reads it,
    differing() and take_groups() carry its arm, and the restore puts it back through the put-back, which under the
    zone-plan mutex stores zone_onednn_plan_keep_short(prior, prior, capacity, held pair) only while the snapshot still
    holds what the load left. The DIRECT cap snapshot is not read into the record, and the keep path writes no
    planner-published term."""
    c = code(uc)
    terms = fn_body(c, TERMS_SIG)
    restore = fn_body(c, RESTORE_SIG)
    arm = block_at(restore, restore.find(RESTORE_ARM)) if RESTORE_ARM in restore else ""
    writers = (fn_body(c, KEEP_SIG), fn_body(c, CAP_SIG), _kept_block(uc))
    return (
        terms.count(TERMS_READ) == 1
        and SNAPSHOT not in terms
        and DIFFERING_ARM in fn_body(c, DIFFERING_SIG)
        and TAKE_ARM in fn_body(c, TAKE_SIG)
        and ordered(arm, RESTORE_CAPACITY, RESTORE_HELD, PUT_BACK_CALL)
        and c.count(PUT_BACK_FN + "(") == 2
        and ordered(fn_body(c, PUT_BACK_SIG), PLAN_LOCK, PUT_BACK_TEST, PUT_BACK_STORE)
        and all(writers)
        and not any(w in body for body in writers for w in PUBLISHED_WRITES)
    )


def claim_both_lines_name_the_stored_floor(uc: str) -> bool:
    """The DIRECT-failure ERROR and the plan's floor INFO line print the formula floor with their fields unchanged, then
    the Graph room the stored zone plan holds whenever it holds less than that floor, read through one helper."""
    c = code(uc)
    return (
        ordered(fn_body(c, SHORT_SIG), SHORT_READ, SHORT_TEST, SHORT_PRINT)
        and ordered(fn_body(c, DIRECT_SIG), DIRECT_FLOOR, "GGML_LOG_ERROR(", DIRECT_TEXT, DIRECT_ARG)
        and ordered(fn_body(c, POPULATE_SIG), FLOOR_VALUE, "GGML_LOG_INFO(", FLOOR_PREFIX, FLOOR_TEXT, FLOOR_ARG)
        and c.count(SHORT_FN + "(") == 3
    )


def keep_checks(uc: str, gs: str) -> dict:
    return {
        "place": claim_keep_sits_in_the_blocked_branch(uc),
        "ring": claim_ring_alone_defers_the_keep(uc),
        "rule": claim_one_rule_decides_the_keep(uc, gs),
        "entry": claim_entry_spends_its_last_resort_first(gs),
        "cap": claim_cap_snapshot_rises_only_at_a_rebuild(uc),
        "warn": claim_a_keep_warns_once_per_load(uc),
        "restore": claim_a_failed_load_puts_the_zone_plan_back(uc),
        "log": claim_both_lines_name_the_stored_floor(uc),
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


def test_the_cap_snapshot_rises_only_at_a_rebuild():
    assert claim_cap_snapshot_rises_only_at_a_rebuild(UC), (
        f"the sufficient and keep paths of ensure_planned_arena_zones() must store the DIRECT cap snapshot through "
        f"{CAP_FN}() (min of held and current, by compare-exchange), the rebuild must store current, and no other "
        "site may store it: a snapshot that rises while the arena is kept grows the Graph floor and asks for a rebuild"
    )


def test_a_keep_warns_once_per_load():
    assert claim_a_keep_warns_once_per_load(UC), (
        "a keep must log one WARN naming the planned Graph floor, the zone kept and the Graph room it leaves, behind a "
        "latch that exchanges the load transaction in (once per load and device; every keep for a direct entry)"
    )


def test_a_failed_load_puts_the_zone_plan_back():
    assert claim_a_failed_load_puts_the_zone_plan_back(UC), (
        "the zone-plan snapshot must be a group of the load's published record (read by snapshot(), with arms in "
        f"differing() and take_groups()), and the restore must put it back through {PUT_BACK_FN}(), which stores "
        "zone_onednn_plan_keep_short(prior, prior, capacity now, held pair now) under the zone-plan mutex only while "
        "the snapshot still holds what the load left; the DIRECT cap snapshot must not be read into the record, and "
        "the keep path must write no planner-published term"
    )


def test_both_lines_name_the_stored_floor():
    assert claim_both_lines_name_the_stored_floor(UC), (
        "the DIRECT-failure ERROR and the '[SYCL-PLAN] oneDNN Graph-scratch zone floor:' line must keep their fields "
        f"and append, through {SHORT_FN}(), the Graph room the stored zone plan holds whenever it is below the "
        "formula floor, so neither line claims the zone holds the formula floor"
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

    # The sufficient path storing the current budget again: a tenant's release raises the snapshot of a kept arena.
    only_fails(mutate_in(UC, ENSURE_SIG, CAP_KEEP + " #endif onednn_zone_plan_keep_and_store(",
                         REBUILD_STORE + " #endif onednn_zone_plan_keep_and_store("), GS, "cap")
    # The keep path leaving the snapshot as it was: the snapshot of a zone kept short is never bounded.
    only_fails(mutate_in(UC, ENSURE_SIG, CAP_KEEP + " #endif " + LATCH_READ, "#endif " + LATCH_READ), GS, "cap")
    # The rebuild keeping the lower snapshot: a rebuilt arena's cap would stay at a budget it was not sized against.
    only_fails(mutate_in(UC, ENSURE_SIG, REBUILD_STORE, CAP_KEEP), GS, "cap")
    # The helper storing the current budget: the rule is not applied.
    only_fails(mutate_in(UC, CAP_SIG, CAP_RULE, "current"), GS, "cap")

    # The latch read and never moved: the first load's WARN would be the last.
    only_fails(mutate_in(UC, ENSURE_SIG, "onednn_zone_kept_warn_txn_.exchange(options.load_txn)",
                         "onednn_zone_kept_warn_txn_.load()"), GS, "warn")
    # No latch: the stage's second pass and its grow leg warn again for one load.
    only_fails(mutate_in(UC, ENSURE_SIG, "if (" + LATCH + ") {", "{"), GS, "warn")
    # The room printed from the plan's floor rather than the kept one.
    only_fails(mutate_in(UC, ENSURE_SIG, "kept.graph_floor_bytes / (1024.0 * 1024.0));",
                         "live_plan.graph_floor_bytes / (1024.0 * 1024.0));"), GS, "warn")
    # The keep publishing a planner term a failed load would then have to put back.
    only_fails(mutate_in(UC, KEEP_SIG, STORE, STORE + " g_planned_onednn_scratchpad_bytes[device_id].store(0);"),
               GS, "restore")
    # The record not reading the zone plan: a refused load leaves its description, and a smaller Graph room, behind.
    only_fails(mutate_in(UC, TERMS_SIG, TERMS_READ, ""), GS, "restore")
    # The record reading the DIRECT cap snapshot, which describes the arena as it stands and is not put back.
    only_fails(mutate_in(UC, TERMS_SIG, TERMS_READ, TERMS_READ + f" (void) {SNAPSHOT};"), GS, "restore")
    # A changed zone plan that marks no group: the restore never sees it differ.
    only_fails(mutate_in(UC, DIFFERING_SIG, DIFFERING_ARM, f"bit(false, {ZONE_GROUP})"), GS, "restore")
    # The prior plan stored raw: a zone rebuilt smaller, or a pair reserved under the wider bound, is described wrong.
    only_fails(mutate_in(UC, PUT_BACK_SIG, PUT_BACK_RULE, "prior"), GS, "restore")
    # The prior plan stored over another publisher's later store.
    only_fails(mutate_in(UC, PUT_BACK_SIG, PUT_BACK_TEST, "{"), GS, "restore")
    # The put-back reading the zone as the failed load found it rather than as it is now.
    only_fails(mutate_in(UC, RESTORE_SIG, RESTORE_HELD, "0"), GS, "restore")

    # The helper that never names the stored figure: both lines claim the zone holds the formula floor.
    only_fails(mutate_in(UC, SHORT_SIG, SHORT_TEST, "return std::string();"), GS, "log")
    # The DIRECT-failure ERROR without the stored figure.
    only_fails(mutate_in(UC, DIRECT_SIG, DIRECT_ARG, '"");'), GS, "log")
    # The floor line without it.
    only_fails(mutate_in(UC, POPULATE_SIG, FLOOR_ARG, '"");'), GS, "log")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
