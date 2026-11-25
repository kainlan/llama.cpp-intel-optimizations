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
- cap: the DIRECT Graph-scratch cap's budget snapshot does not rise while the arena is kept. The sufficient path,
  both keep paths (a short ONEDNN zone, and a SCRATCH zone short of the MoE batched MMVQ bound alone) and both ends of
  a RUNTIME give-back (an arena something holds, kept as built, and an arena rebuilt with its SCRATCH and ONEDNN zones
  as they were: zone_arena_ensure_options::give_back) call one local lambda, keep_arena_as_built, which binds the
  compute arena when it is unbound and stores min(held, current) through one compare-exchange helper over
  zone_graph_direct_cap_snapshot_kept (test-zone-sizing Case 42), and only a rebuild for larger zones stores the
  current budget. A snapshot that rose across tenants would grow the Graph floor of a kept zone and ask for a rebuild
  the zone does not need;
- warn: a keep logs one WARN naming the planned Graph floor, the zone kept and the Graph room it leaves, behind a
  latch (zone_kept_onednn_warn_due, Case 43) that exchanges the load transaction in: once per load and device, since
  the stage asks for the zones in two passes and again in its grow leg, and at every keep for a direct entry (0);
- restore: the zone-plan snapshot IS a term a failed load puts back. The keep and the sufficient path store a
  description (the pair plan and the Graph floor) and move no memory, so a refused load that left its own description
  would leave the live model a smaller Graph room for nothing. The published-terms snapshot reads it as its own group
  (differing() and take_groups() carry its arm), and the restore stores zone_onednn_plan_keep_short(prior, prior,
  capacity now, held pair now) under the zone-plan mutex only while the snapshot still holds what the load left
  (test-zone-sizing Case 44), through the put-back of the device's cache, which reads that cache's zone itself before
  it takes the mutex. The DIRECT cap snapshot is not put back (it describes the arena as it stands, and a
  failed load that rebuilt the arena leaves the rebuilt arena in place), so the published-terms snapshot does not read
  it, and the keep path writes no planner-published term;
- log: the DIRECT-failure ERROR and the plan's "[SYCL-PLAN] oneDNN Graph-scratch zone floor:" line keep their fields
  and append, through one helper, the Graph room the stored zone plan holds whenever it is below the formula floor, so
  neither claims a kept zone holds the floor;
- home: the zone plan lives on the cache that owns the zone (a member of unified_cache and its own mutex), never in a
  per-device store, so a zone cannot be judged by a plan another cache stored on the same device. A cache whose arena
  was reserved by hand stored none, and test-unified-runtime-alloc's arena_owned_shutdown_and_lifecycle_serialization
  was refused its pair because the device's own cache had stored a plan whose Graph floor exceeded the hand-built zone.
  Every accessor is a member that takes that mutex, and nothing else touches the member;
- readers: the pair reserve and its refusal read their own cache's plan; the pair-bound and Graph-room accessors read
  the plan of the cache whose capacity they read; a reader that holds only a device id (the published-terms snapshot,
  the plan's floor line) asks that device's existing cache through one helper, which creates none.

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
HPP = (SYCL / "unified-cache.hpp").read_text()

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
KEEP_SIG = f"bool unified_cache::{KEEP_FN}("
NEED = "const zone_arena_sizes need = { scratch_zone, onednn_zone, runtime_zone };"
DEFER = "const bool defer_keep = options.ring_release_first && zone_arena_only_ring_holds(true, holders);"
HELD_PAIR = "const size_t held_pair = arena_held_onednn_pair_bytes();"
KEEP_CALL = f"const bool reservation_only = {KEEP_FN}(live_plan, arena_zones, need, held_pair, !defer_keep, "
KEPT_RETURN = "if (reservation_only && !defer_keep) {"
DEFERRED_MARK = "refusal->keep_deferred = reservation_only && defer_keep;"
PLAN_LOCK = "std::lock_guard<std::mutex> lock(onednn_zone_plan_mutex_);"
SHORT_RULE = "zone_onednn_plan_keep_short(onednn_zone_plan_, live, have.onednn, held_pair_bytes)"
ONLY_RULE = (
    "zone_arena_shortfall_is_reservation_only(have, need, zone_onednn_kept_pair_bytes(short_plan, held_pair_bytes))"
)
STORE = "if (store) { onednn_zone_plan_ = short_plan; }"
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
# The one lambda every keep calls: it binds the compute arena when it is unbound, then keeps the snapshot.
KEEP_ARENA = "keep_arena_as_built();"
KEEP_ARENA_DEF = "const auto keep_arena_as_built = [&]() {"
KEEP_ARENA_BIND = "if (!compute_arena_ptr_ || !vram_owns(compute_arena_ptr_)) { compute_arena_bind(); }"
# A give-back with no surplus to take keeps the arena as built too, where another zone would grow.
SUFFICIENT = "if (!surplus_asked && (zones_sufficient || options.give_back != nullptr)) {"
# The two ends of a RUNTIME give-back: an arena something holds is kept as built, and a rebuilt one keeps its SCRATCH and
# ONEDNN zones as they were, so both keep the snapshot as a kept arena does.
GIVE_BACK_HELD = "if (surplus_asked && !zone_arena_rebuild_allowed(holders)) {"
GIVE_BACK_TAIL = "if (giving_back) {"
# A blocked rebuild keeps a SCRATCH zone short of the MoE batched MMVQ bound alone before the ONEDNN keep is decided.
SCRATCH_BLOCKED = "if (scratch_kept < scratch_zone && !zone_arena_rebuild_allowed(holders)) {"
REBUILD_STORE = f"{SNAPSHOT}.store(available_budget(), std::memory_order_release);"
REBUILD_TAIL = "onednn_zone_plan_store(live_plan);"
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
TERMS_READ = "terms.onednn_zone_plan = onednn_zone_plan_load_of_device(device);"
DIFFERING_SIG = "uint32_t load_published_terms::differing(const load_published_terms & other) const {"
DIFFERING_ARM = f"bit(!zone_onednn_plan_same(onednn_zone_plan, other.onednn_zone_plan), {ZONE_GROUP})"
TAKE_SIG = "void load_published_terms::take_groups(const load_published_terms & from, uint32_t groups) {"
TAKE_ARM = f"if (groups & {ZONE_GROUP}) {{ onednn_zone_plan = from.onednn_zone_plan; }}"
RESTORE_SIG = "void load_published_record::restore(int device) const noexcept {"
RESTORE_ARM = f"if (back & {ZONE_GROUP}) {{"
PUT_BACK_FN = "onednn_zone_plan_put_back"
PUT_BACK_SIG = f"void unified_cache::{PUT_BACK_FN}("
PUT_BACK_CALL = f"cache->{PUT_BACK_FN}(written.onednn_zone_plan, prior.onednn_zone_plan);"
# The put-back reads its own cache's zone, before it takes the zone-plan mutex.
CAPACITY_READ = "zone_capacity(vram_zone_id::ONEDNN)"
HELD_READ = "arena_held_onednn_pair_bytes()"
PUT_BACK_CAPACITY = f"const size_t capacity = {CAPACITY_READ};"
PUT_BACK_HELD = f"const size_t held = {HELD_READ};"
PUT_BACK_TEST = "if (zone_onednn_plan_same(onednn_zone_plan_, written)) {"
PUT_BACK_RULE = "zone_onednn_plan_keep_short(prior, prior, capacity, held)"
PUT_BACK_STORE = f"onednn_zone_plan_ = {PUT_BACK_RULE};"
SHORT_FN = "onednn_zone_plan_short_text"
SHORT_SIG = f"static std::string {SHORT_FN}(const zone_onednn_plan & stored, size_t formula_bytes) {{"
SHORT_TEST = (
    "if (zone_onednn_plan_same(stored, zone_onednn_plan()) || stored.graph_floor_bytes >= formula_bytes) { "
    "return std::string(); }"
)
SHORT_PRINT = "stored.graph_floor_bytes / (1024.0 * 1024.0)"
DIRECT_SIG = "void * unified_cache::onednn_graph_scratch_alloc_direct_locked("
DIRECT_FLOOR = "const size_t formula_floor = onednn_graph_scratch_zone_floor_bytes_swa("
DIRECT_TEXT = "n_ctx=%u%s). Refusing"
DIRECT_ARG = f"{SHORT_FN}(onednn_zone_plan_load(), formula_floor).c_str());"
POPULATE_SIG = "static void populate_host_zone_sizing(placement_plan & plan,"
FLOOR_VALUE = "const size_t graph_floor = onednn_graph_scratch_zone_floor_bytes_swa("
FLOOR_PREFIX = '"[SYCL-PLAN] oneDNN Graph-scratch zone floor: %.1f MB (n_head_ctx_max=%u n_head_swa_max=%u n_swa=%u "'
FLOOR_TEXT = '"n_ubatch=%u n_ctx=%u)%s\\n",'
FLOOR_ARG = f"{SHORT_FN}(onednn_zone_plan_load_of_device(plan.device_id), graph_floor).c_str());"
CLASS_OPEN = "class unified_cache {"
HOME_MUTEX = "mutable std::mutex onednn_zone_plan_mutex_;"
HOME_MEMBER = "zone_onednn_plan onednn_zone_plan_{};"
MEMBER_LOCK = PLAN_LOCK
LOAD_SIG = "zone_onednn_plan unified_cache::onednn_zone_plan_load() const {"
ACCESSOR_SIGS = (
    LOAD_SIG,
    "void unified_cache::onednn_zone_plan_store(const zone_onednn_plan & plan) {",
    "void unified_cache::onednn_zone_plan_keep_and_store(const zone_onednn_plan & plan) {",
    "bool unified_cache::onednn_zone_plan_keep_short_and_store(",
    "void unified_cache::onednn_zone_plan_put_back(",
)
OF_DEVICE_FN = "onednn_zone_plan_load_of_device"
OF_DEVICE_SIG = f"static zone_onednn_plan {OF_DEVICE_FN}(int device_id) {{"
OF_DEVICE_LOOKUP = "const unified_cache * const cache = get_existing_unified_cache_for_device(device_id);"
OF_DEVICE_READ = "return cache != nullptr ? cache->onednn_zone_plan_load() : zone_onednn_plan();"
RESERVE_SIG = "bool unified_cache::reserve_onednn_scratch(size_t weights_size, size_t activations_size) {"
RESERVE_FIRST = "const zone_onednn_plan zone_plan = arena_on ? onednn_zone_plan_load() : zone_onednn_plan();"
RESERVE_REFUSAL = "const size_t pair_bound_now = onednn_pp_pair_bound_for(onednn_zone_plan_load(), zone_cap);"
PAIR_BOUND_SIG = ("bool unified_cache_get_onednn_pp_pair_bound(int device_id, size_t * bound, size_t * planned_weights, "
                  "size_t * planned_activations) {")
ROOM_SIG = "bool unified_cache_get_onednn_graph_room(int device_id, size_t * room) {"
BOUND_FN = "onednn_pp_pair_bound_for"
# The first argument every use of the bound helper may pass: its own parameter at the definition, the reserve's loaded
# plan, this cache's plan, or the plan of the cache whose capacity the accessor reads.
BOUND_FIRST_ARGS = {"const zone_onednn_plan & plan", "zone_plan", "onednn_zone_plan_load()",
                    "cache->onednn_zone_plan_load()"}


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
    """The DIRECT cap's budget snapshot has six stores, all in the ensure: the sufficient path, the keep of a SCRATCH
    zone short of the MoE batched MMVQ bound alone, the keep of a short ONEDNN zone and both ends of a RUNTIME give-back
    each call keep_arena_as_built, which stores min(held, current) through one compare-exchange helper (0 is unset), and
    the rebuild for larger zones stores current. No other site stores it."""
    c = code(uc)
    ensure = _ensure(uc)
    helper = fn_body(c, CAP_SIG)
    if not ordered(helper, CAP_CURRENT, CAP_HELD, CAP_LOOP) or c.count("zone_graph_direct_cap_snapshot_kept(") != 1:
        return False
    anchors = (SUFFICIENT, BLOCKED, SCRATCH_BLOCKED, KEEP_ARENA_DEF, GIVE_BACK_HELD, GIVE_BACK_TAIL)
    if any(a not in ensure for a in anchors):
        return False
    keep_arena = block_at(ensure, ensure.index(KEEP_ARENA_DEF) + len(KEEP_ARENA_DEF) - 1)
    sufficient = block_at(ensure, ensure.index(SUFFICIENT))
    scratch_kept = block_at(ensure, ensure.index(SCRATCH_BLOCKED))
    give_back_held = block_at(ensure, ensure.index(GIVE_BACK_HELD))
    give_back_tail = block_at(ensure, ensure.index(GIVE_BACK_TAIL))
    blocked = block_at(ensure, ensure.index(BLOCKED))
    kept = block_at(blocked, blocked.find(KEPT_RETURN)) if KEPT_RETURN in blocked else ""
    return (
        ordered(keep_arena, KEEP_ARENA_BIND, CAP_KEEP)
        and SNAPSHOT not in keep_arena
        and KEEP_ARENA in sufficient
        and SNAPSHOT not in sufficient
        and ordered(scratch_kept, KEEP_ARENA, "return true;")
        and SNAPSHOT not in scratch_kept
        and ordered(kept, KEEP_ARENA, "return true;")
        and SNAPSHOT not in kept
        and ordered(give_back_held, KEEP_ARENA, "return true;")
        and SNAPSHOT not in give_back_held
        and ordered(give_back_tail, KEEP_ARENA, "return true;")
        and SNAPSHOT not in give_back_tail
        and ensure.count(CAP_KEEP) == 1
        and ensure.count(KEEP_ARENA) == 5
        and ensure.count(SNAPSHOT) == 1
        and ordered(ensure, BLOCKED, GIVE_BACK_TAIL, REBUILD_STORE, REBUILD_TAIL)
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
    differing() and take_groups() carry its arm, and the restore puts it back through the put-back of the device's
    cache. The put-back reads that cache's ONEDNN zone capacity and the pair held in it, then under the zone-plan mutex
    stores zone_onednn_plan_keep_short(prior, prior, capacity, held pair) only while the snapshot still holds what the
    load left; it takes no figure from its caller, so the plan is never clamped to another cache's zone. The DIRECT cap
    snapshot is not read into the record, and the keep path writes no planner-published term."""
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
        and arm.count(PUT_BACK_CALL) == 1
        and c.count(PUT_BACK_FN + "(") == 2
        and ordered(fn_body(c, PUT_BACK_SIG), PUT_BACK_CAPACITY, PUT_BACK_HELD, PLAN_LOCK, PUT_BACK_TEST, PUT_BACK_STORE)
        and all(writers)
        and not any(w in body for body in writers for w in PUBLISHED_WRITES)
    )


def claim_both_lines_name_the_stored_floor(uc: str) -> bool:
    """The DIRECT-failure ERROR and the plan's floor INFO line print the formula floor with their fields unchanged, then
    the Graph room the stored zone plan holds whenever it holds less than that floor, through one helper: the ERROR
    passes its own cache's plan, the floor line the plan of the device's cache."""
    c = code(uc)
    return (
        ordered(fn_body(c, SHORT_SIG), SHORT_TEST, SHORT_PRINT)
        and ordered(fn_body(c, DIRECT_SIG), DIRECT_FLOOR, "GGML_LOG_ERROR(", DIRECT_TEXT, DIRECT_ARG)
        and ordered(fn_body(c, POPULATE_SIG), FLOOR_VALUE, "GGML_LOG_INFO(", FLOOR_PREFIX, FLOOR_TEXT, FLOOR_ARG)
        and c.count(SHORT_FN + "(") == 3
    )


def first_args(text: str, fn: str) -> list:
    """The first argument of every call or definition `fn(` in `text`, up to its first top-level comma or the closing
    parenthesis."""
    found = []
    for m in re.finditer(r"\b" + re.escape(fn) + r"\(", text):
        depth, i = 0, m.end()
        while i < len(text):
            ch = text[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            i += 1
        found.append(text[m.end() : i].strip())
    return found


def claim_the_zone_plan_lives_on_its_cache(uc: str, hpp: str) -> bool:
    """The plan is a member of the cache that owns the zone, under that cache's own mutex, and no per-device store of
    it exists. Each accessor takes the mutex once, no other site takes it or touches the member, and the load takes no
    argument a device could key it by."""
    c = code(uc)
    h = code(hpp)
    cls = block_at(h, h.find(CLASS_OPEN)) if CLASS_OPEN in h else ""
    bodies = [fn_body(c, sig) for sig in ACCESSOR_SIGS]
    member = r"\bonednn_zone_plan_(?!\w)"
    member_uses = len(re.findall(member, c))
    return (
        "g_onednn_zone_plan" not in c
        and re.search(r"\bzone_onednn_plan\s+\w+\s*\[", c) is None
        and cls.count(HOME_MUTEX) == 1
        and cls.count(HOME_MEMBER) == 1
        and all(body.count(MEMBER_LOCK) == 1 for body in bodies)
        and c.count("onednn_zone_plan_mutex_") == len(ACCESSOR_SIGS)
        and member_uses > 0
        and member_uses == sum(len(re.findall(member, body)) for body in bodies)
        and re.search(r"\bonednn_zone_plan_load\(\s*[^)\s]", c) is None
    )


def claim_readers_ask_the_cache_that_owns_the_zone(uc: str) -> bool:
    """Every use of the pair-bound helper passes a plan read from a cache it already holds: the reserve and its refusal
    their own cache's, the pair-bound and Graph-room accessors the cache whose capacity they read (the exact reads are
    pinned in test-sycl-onednn-pp-scratch-plan-source.py and test-sycl-onednn-graph-scratch-refusal-source.py). None of
    those three bodies reads a plan through the device-id helper, and the reserve looks no cache up. The readers that
    hold only a device id (the published-terms snapshot and the plan's floor line) ask that device's existing cache
    through that one helper, which creates none."""
    c = code(uc)
    reserve = fn_body(c, RESERVE_SIG)
    pair_bound = fn_body(c, PAIR_BOUND_SIG)
    room = fn_body(c, ROOM_SIG)
    of_device = OF_DEVICE_FN + "("
    # The helper's definition and the two readers that hold only a device id are its only sites.
    of_device_sites = 1 + fn_body(c, TERMS_SIG).count(of_device) + fn_body(c, POPULATE_SIG).count(of_device)
    return (
        set(first_args(c, BOUND_FN)) <= BOUND_FIRST_ARGS
        and all(BOUND_FN + "(" in body for body in (reserve, pair_bound, room))
        and ordered(fn_body(c, OF_DEVICE_SIG), OF_DEVICE_LOOKUP, OF_DEVICE_READ)
        and c.count(of_device) == of_device_sites
        and RESERVE_FIRST in reserve
        and RESERVE_REFUSAL in reserve
        and "get_existing_unified_cache_for_device(" not in reserve
        and not any(of_device in body for body in (reserve, pair_bound, room))
    )


def keep_checks(uc: str, gs: str, hpp: str = HPP) -> dict:
    return {
        "place": claim_keep_sits_in_the_blocked_branch(uc),
        "ring": claim_ring_alone_defers_the_keep(uc),
        "rule": claim_one_rule_decides_the_keep(uc, gs),
        "entry": claim_entry_spends_its_last_resort_first(gs),
        "cap": claim_cap_snapshot_rises_only_at_a_rebuild(uc),
        "warn": claim_a_keep_warns_once_per_load(uc),
        "restore": claim_a_failed_load_puts_the_zone_plan_back(uc),
        "log": claim_both_lines_name_the_stored_floor(uc),
        "home": claim_the_zone_plan_lives_on_its_cache(uc, hpp),
        "readers": claim_readers_ask_the_cache_that_owns_the_zone(uc),
    }


def only_fails(uc: str, gs: str, failing: str, hpp: str = HPP) -> None:
    results = keep_checks(uc, gs, hpp)
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
        "zone-plan mutex and store only when asked, and no other site may apply them, except a failed load's put-back "
        "(which applies zone_onednn_plan_keep_short)"
    )


def test_the_entry_spends_its_last_resort_first():
    assert claim_entry_spends_its_last_resort_first(GS), (
        "ggml_sycl_ensure_load_arena_zones() must ask first with ring_release_first, ask after its last resort "
        "without it, and ask again over a kept ring only when the refusal's keep_deferred is set"
    )


def test_the_cap_snapshot_rises_only_at_a_rebuild():
    assert claim_cap_snapshot_rises_only_at_a_rebuild(UC), (
        f"the sufficient and both keep paths of ensure_planned_arena_zones() must store the DIRECT cap snapshot "
        f"through {CAP_FN}() (min of held and current, by compare-exchange), the rebuild must store current, and no "
        "other site may store it: a snapshot that rises while the arena is kept grows the Graph floor and asks for a "
        "rebuild"
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


def test_the_zone_plan_lives_on_its_cache():
    assert claim_the_zone_plan_lives_on_its_cache(UC, HPP), (
        "the ONEDNN zone plan must be a member of the unified_cache that owns the zone, under its own mutex, with no "
        "per-device store, and every accessor a member that takes that mutex. A per-device store judges a zone by a "
        "plan another cache stored: a cache whose arena was reserved by hand was refused every oneDNN pair because the "
        "device's own cache had stored a larger Graph floor"
    )


def test_readers_ask_the_cache_that_owns_the_zone():
    assert claim_readers_ask_the_cache_that_owns_the_zone(UC), (
        "the pair reserve and its refusal must read their own cache's plan, the pair-bound and Graph-room accessors "
        f"the plan of the cache whose capacity they read, and a reader with only a device id must ask that device's "
        f"existing cache through {OF_DEVICE_FN}(), which no other reader calls"
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
    only_fails(mutate_in(UC, KEEP_SIG, SHORT_RULE, "zone_onednn_plan_keep(onednn_zone_plan_, live)"), GS, "rule")
    # The keep stored even when the caller defers it.
    only_fails(mutate_in(UC, KEEP_SIG, STORE, "onednn_zone_plan_ = short_plan;"), GS, "rule")
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

    # The sufficient path and the SCRATCH keep end alike, so their witnesses mutate the block the anchor opens.
    ensure = _ensure(UC)

    def in_ensure_block(anchor: str, old: str, new: str) -> str:
        block = block_at(ensure, ensure.index(anchor))
        return code(UC).replace(ensure, ensure.replace(block, block.replace(old, new, 1), 1), 1)

    # The sufficient path storing the current budget again: a tenant's release raises the snapshot of a kept arena.
    only_fails(in_ensure_block(SUFFICIENT, KEEP_ARENA, REBUILD_STORE), GS, "cap")
    # The keep path leaving the snapshot as it was: the snapshot of a zone kept short is never bounded.
    only_fails(mutate_in(UC, ENSURE_SIG, KEEP_ARENA + " " + LATCH_READ, LATCH_READ), GS, "cap")
    # The SCRATCH keep leaving the snapshot as it was: an arena kept for a short SCRATCH zone is not bounded either.
    only_fails(in_ensure_block(SCRATCH_BLOCKED, KEEP_ARENA, ""), GS, "cap")
    # A held give-back leaving the snapshot as it was: the arena it keeps is not bounded.
    only_fails(in_ensure_block(GIVE_BACK_HELD, KEEP_ARENA, ""), GS, "cap")
    # A give-back rebuild storing the current budget: its ONEDNN zone is the one it had, so the snapshot must not rise.
    only_fails(in_ensure_block(GIVE_BACK_TAIL, KEEP_ARENA, REBUILD_STORE), GS, "cap")
    # The lambda every keep calls leaving the snapshot as it was: no kept arena is bounded.
    only_fails(mutate_in(UC, ENSURE_SIG, CAP_KEEP + " #endif };", "#endif };"), GS, "cap")
    # The lambda leaving an unbound compute arena unbound: a kept arena would serve no compute scratch.
    only_fails(mutate_in(UC, ENSURE_SIG, KEEP_ARENA_BIND, ""), GS, "cap")
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
    only_fails(mutate_in(UC, KEEP_SIG, STORE, STORE + " g_planned_onednn_scratchpad_bytes[0].store(0);"), GS,
               "restore")
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
    # The put-back ignoring the pair held in the zone now (read as 0): the stored floor leaves no room for that pair.
    only_fails(mutate_in(UC, PUT_BACK_SIG, HELD_READ, "0"), GS, "restore")
    # The put-back reading the zone's capacity as 0: every put-back would store a Graph floor of 0.
    only_fails(mutate_in(UC, PUT_BACK_SIG, CAPACITY_READ, "0"), GS, "restore")
    # The zone read under the zone-plan mutex: the mutex is no longer a leaf.
    only_fails(mutate_in(UC, PUT_BACK_SIG, PUT_BACK_CAPACITY + " " + PUT_BACK_HELD + " " + PLAN_LOCK,
                         PLAN_LOCK + " " + PUT_BACK_CAPACITY + " " + PUT_BACK_HELD), GS, "restore")
    # The caller handing the put-back a zone's figures again: they can be another cache's.
    only_fails(mutate_in(UC, RESTORE_SIG, PUT_BACK_CALL,
                         f"cache->{PUT_BACK_FN}(written.onednn_zone_plan, prior.onednn_zone_plan, 0, 0);"), GS,
               "restore")
    # The take_groups arm dropped: the record never takes the zone plan as its prior, so a restore has nothing to put.
    only_fails(mutate_in(UC, TAKE_SIG, TAKE_ARM, ""), GS, "restore")

    # The helper that never names the stored figure: both lines claim the zone holds the formula floor.
    only_fails(mutate_in(UC, SHORT_SIG, SHORT_TEST, "return std::string();"), GS, "log")
    # The DIRECT-failure ERROR without the stored figure.
    only_fails(mutate_in(UC, DIRECT_SIG, DIRECT_ARG, '"");'), GS, "log")
    # The floor line without it.
    only_fails(mutate_in(UC, POPULATE_SIG, FLOOR_ARG, '"");'), GS, "log")

    # A per-device store of the plan again: a zone is judged by whichever cache on the device stored last.
    only_fails(code(UC) + " static zone_onednn_plan g_onednn_zone_plan[GGML_SYCL_MAX_DEVICES]{};", GS, "home")
    # The plan read without its mutex: a store and a read race.
    only_fails(mutate_in(UC, LOAD_SIG, MEMBER_LOCK, ""), GS, "home")
    # The member read outside the accessors, without the mutex.
    only_fails(code(UC) + " zone_onednn_plan peek(const unified_cache & c) { return c.onednn_zone_plan_; }", GS, "home")
    # The plan no longer a member of the cache.
    hpp_code = code(HPP)
    assert hpp_code.count(HOME_MEMBER) == 1, "witness anchor: the plan member is not declared once"
    only_fails(UC, GS, "home", hpp_code.replace(HOME_MEMBER, ""))

    # The reserve's refusal judged by the device's cache rather than its own: the hand-built arena's defect.
    only_fails(mutate_in(UC, RESERVE_SIG, "onednn_pp_pair_bound_for(onednn_zone_plan_load(), zone_cap)",
                         f"onednn_pp_pair_bound_for({OF_DEVICE_FN}(ggml_sycl_get_device_id_from_queue(queue_)), "
                         "zone_cap)"), GS, "readers")
    # The pair-bound accessor reading a plan through a second lookup, which can find another cache than the one
    # whose capacity it reads.
    only_fails(mutate_in(UC, PAIR_BOUND_SIG, "zone_plan = cache->onednn_zone_plan_load();",
                         f"zone_plan = {OF_DEVICE_FN}(device_id);"), GS, "readers")
    # The Graph-room accessor the same way.
    only_fails(mutate_in(UC, ROOM_SIG, "cache->onednn_zone_plan_load(), capacity",
                         f"{OF_DEVICE_FN}(device_id), capacity"), GS, "readers")
    # The device-id helper creating a cache for a reader that only asks.
    only_fails(mutate_in(UC, OF_DEVICE_SIG, "get_existing_unified_cache_for_device(device_id)",
                         "get_unified_cache_for_device(device_id)"), GS, "readers")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
