"""Source contract for a model load whose plan the VRAM arena's zones cannot hold: the load is refused, never aborted.

The arena's zones grow only by a rebuild, and the arena is not rebuilt while anything points into it (a live model's
allocations, or what a freed model's context left behind). A load whose planned terms need more than the zones hold
then cannot be placed. It used to abort the process; it is now refused by name and the models already on the device
keep running. These checks pin that:

No GGML_ABORT stays on the load's plan path: compute_and_store_plan_for_inventory, the two planner entries that call
it (ggml_sycl_compute_placement_plan_early_impl, ggml_sycl_set_tensor_inventory_impl) and the stage.

compute_and_store_plan_for_inventory returns bool. At both zone ensures (the entry ensure and the micro-batch
fallback's) a false ensure logs the refusal line, which zone_arena_refusal_text formats from what the ensure wrote, and
returns false.

Every caller propagates it: each planner entry reports a refused plan through its plan_refused out-flag, and the stage
returns GGML_SYCL_LIFECYCLE_PLAN_REJECTED from either pass, the late one before it checks the staged candidate (a
refused plan staged none, and that check would throw an anonymous EFFECT_FAILED). The exported direct entries keep their
void signatures across the dlopen boundary; they log the refusal and stage nothing.

The refusal names need and have: the ensure writes, at the holders refusal, the three zone sizes it decided on as the
need and the arena's zone capacities, read before either refusal, as the have; at a failed reserve, the same need, the
zones the device has after the failure as the have, and those capacities, read before the rebuild destroyed the arena,
as released; the device wrapper names a device with no cache. zone_arena_refusal_text prints both for every zone, in
MB, and a short zone whose shortfall is under 0.1 MB in KB, or in bytes under 1 KB, so no shortfall prints as 0.0 MB,
and a failed reserve's released arena. At the holders refusal the ensure also writes the holders its gate refused on,
and the text names them and chooses its remedy from them (test-zone-sizing Case 38 checks the text and each remedy).

A failed load puts back what it published: the stage captures the device-wide planned terms before its first device
plans (a load_published_record per device, kept for the one open load transaction), the abort of a load's owner
effects restores them for that transaction, and load_end forgets them once the load committed. Every device-wide
unified_cache_set_planned_* call on the load's publish path (ggml_sycl_inventory_publish_planner_inputs,
populate_inventory_globals, ggml_sycl_publish_load_ubatch_scratch), and each planner term, is restored by
load_published_record::restore; the oneDNN pair's setter by the halves' store and the sum's compare-exchange.
Owner-keyed terms leave with their owner instead, and the compute term's micro-batch is sticky by design
(EXEMPT_SETTERS).

The restore puts back only what still holds the load's value. A runtime context transaction of a live model publishes
some of the same terms while the load runs, and the oneDNN pair's reserve raises the stored sum. Both planner entries
construct a ggml_sycl_load_published_pass right after they take g_tensor_inventory_mutex and before they publish; it
notes into the record what the pass changed, and the record takes what another publisher wrote between passes as the
value to put back. The restore takes g_tensor_inventory_mutex before the record's mutex, puts a group back only while it
still holds what the load published (each setter under its group's bit), and puts the sum back by compare-exchange from
the pair's total the load published, so a reserve's raise outside the lock survives.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
GS = (SYCL / "ggml-sycl.cpp").read_text()
UC = (SYCL / "unified-cache.cpp").read_text()
ZS = (SYCL / "zone-sizing.cpp").read_text()

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
    """`src` as code() with the first `old` inside the function `sig` replaced by `new`."""
    c = code(src)
    body = fn_body(c, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = c.find(body, c.find(sig))
    return c[:s] + body.replace(old, new, 1) + c[s + len(body) :]


# ---- the load's plan path ----

# The definitions, by the part of each signature that names the return type: compute_and_store returns bool.
COMPUTE_SIG = "static bool compute_and_store_plan_for_inventory("
EARLY_SIG = "static int ggml_sycl_compute_placement_plan_early_impl("
LATE_SIG = "static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_set_tensor_inventory_impl("
STAGE_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_load_inventory_plan("
EXPORTED_EARLY_SIG = "void ggml_backend_sycl_compute_placement_plan_early(ggml_backend_t backend,"
EXPORTED_LATE_SIG = "void ggml_backend_sycl_set_tensor_inventory(ggml_backend_t backend,"
PATH_SIGS = (COMPUTE_SIG, EARLY_SIG, LATE_SIG, STAGE_SIG)

# The entry ensure goes through the load's last-resort ring release (test-sycl-load-stage-effect-pass-source pins it),
# which writes the same refusal.
ENTRY_ENSURE = ("if (ggml_sycl::get_unified_cache_for_device(ctx->device) && "
                "!ggml_sycl_ensure_load_arena_zones(ctx->device, refusal)) {")
FALLBACK_ENSURE = ("if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, &refusal, "
                   "ggml_sycl_load_arena_options())) {")
REFUSE = 'GGML_LOG_ERROR("%s\\n", ggml_sycl::zone_arena_refusal_text(refusal).c_str()); return false; }'
ABORT_LINE = 'GGML_ABORT("[SYCL-PLAN] failed to size VRAM arena zones from tensor inventory");'


def claim_no_abort_on_the_path(gs: str) -> bool:
    c = code(gs)
    bodies = [fn_body(c, sig) for sig in PATH_SIGS]
    return all(bodies) and not any("GGML_ABORT(" in b for b in bodies) and \
        "failed to size VRAM arena zones" not in c


def claim_both_ensures_refuse_by_name(gs: str) -> bool:
    body = fn_body(code(gs), COMPUTE_SIG)
    return (ordered(body, "ggml_sycl::zone_arena_refusal refusal;", ENTRY_ENSURE, REFUSE)
            and ordered(body, "published_before.restore(ctx->device);", FALLBACK_ENSURE, REFUSE)
            and body.count("return false;") == 2 and body.rstrip("} ").endswith("return true;"))


# The stage's per-device calls and what each does with a refused plan. The late refusal is read before the staged
# candidate is checked.
STAGE_EARLY = ("bool plan_refused = false; if (early) { const int planned = "
               "ggml_sycl_compute_placement_plan_early_impl(backend, inventory, &plan_refused); "
               "if (plan_refused) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED; }")
STAGE_LATE = ("decision.snapshot = ggml_sycl_set_tensor_inventory_impl(backend, inventory, &plan_refused, "
              "decision.effects); if (plan_refused) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED; }")
STAGE_CANDIDATE_CHECK = "if (!candidate_is_current(decision.snapshot)) {"
PLAN_CALL = "compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct)"
EARLY_PROPAGATE = "if (!" + PLAN_CALL + ") { if (plan_refused != nullptr) { *plan_refused = true; } return -1; }"
LATE_PROPAGATE = "if (!" + PLAN_CALL + ") { if (plan_refused != nullptr) { *plan_refused = true; } return {}; }"


def claim_every_caller_propagates(gs: str) -> bool:
    c = code(gs)
    stage = fn_body(c, STAGE_SIG)
    return (EARLY_PROPAGATE in fn_body(c, EARLY_SIG)
            and LATE_PROPAGATE in fn_body(c, LATE_SIG)
            and ordered(stage, STAGE_EARLY, STAGE_LATE, STAGE_CANDIDATE_CHECK)
            and "ggml_sycl_compute_placement_plan_early_impl(backend, inventory, nullptr);" in
            fn_body(c, EXPORTED_EARLY_SIG)
            and "(void) ggml_sycl_set_tensor_inventory_impl(backend, inventory, nullptr, effects);" in
            fn_body(c, EXPORTED_LATE_SIG))


# ---- the refusal names need and have ----

ENSURE_SIG = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
              "const zone_arena_ensure_options & options)")
WRAPPER_SIG = ("bool unified_cache_ensure_planned_arena_zones(int device_id, zone_arena_refusal * refusal, "
               "const zone_arena_ensure_options & options)")
# The need is the three sizes the ensure decided on; the have is the arena's zones, read once before either refusal
# (a failed reserve has destroyed the arena by the time it refuses).
NEED = "refusal->need = { scratch_zone, onednn_zone, runtime_zone };"
HAVE = "refusal->have = arena_zones;"
ZONES_READ = ("arena_zones = { zone_capacity(vram_zone_id::SCRATCH), zone_capacity(vram_zone_id::ONEDNN), "
              "zone_capacity(vram_zone_id::RUNTIME) };")
HOLDERS = "refusal->holders = holders;"
# Whether the zones would have been kept had the caller not asked to release a lone PP MoE ring first.
KEEP_DEFERRED = "refusal->keep_deferred = reservation_only && defer_keep;"
HELD_FILL = ("if (refusal != nullptr) { refusal->reason = zone_arena_refusal_reason::REBUILD_HELD; refusal->device = "
             "dev_id; " + NEED + " " + HAVE + " " + HOLDERS + " " + KEEP_DEFERRED + " } return false;")
# A SCRATCH zone short of the MoE batched MMVQ bound alone, in an arena a lone ring holds, is kept only after the ring's
# release is tried: that refusal names the same need, have and holders, and says the zones would have been kept.
SCRATCH_HELD_FILL = ("if (refusal != nullptr) { refusal->reason = zone_arena_refusal_reason::REBUILD_HELD; "
                     "refusal->device = dev_id; " + NEED + " " + HAVE + " " + HOLDERS +
                     " refusal->keep_deferred = true; } return false;")
# A failed reserve's have is what the device has after it; the arena a rebuild destroyed for it is named as released.
RESERVE_HAVE = ("refusal->have = { zone_capacity(vram_zone_id::SCRATCH), zone_capacity(vram_zone_id::ONEDNN), "
                "zone_capacity(vram_zone_id::RUNTIME) };")
RELEASED = "refusal->released = arena_zones;"
RESERVE_FILL = ("if (refusal != nullptr) { refusal->reason = zone_arena_refusal_reason::RESERVE_FAILED; "
                "refusal->device = dev_id; " + NEED + " " + RESERVE_HAVE + " " + RELEASED + " } return false;")
NO_CACHE_FILL = ("if (!cache) { if (refusal != nullptr) { refusal->reason = zone_arena_refusal_reason::NO_CACHE; "
                 "refusal->device = device_id; } return false; }")
TEXT_SIG = "std::string zone_arena_refusal_text(const zone_arena_refusal & refusal)"
TEXT_ZONES = ('"SCRATCH", refusal.need.scratch, refusal.have.scratch', '"ONEDNN", refusal.need.onednn, '
              'refusal.have.onednn', '"RUNTIME", refusal.need.runtime, refusal.have.runtime')
TEXT_FORMAT = '"%s%s need %.1f MB have %.1f MB"'
# A short zone whose shortfall would round to nothing in MB prints its need, have and shortfall in a smaller unit.
TEXT_SMALL = ('"%s%s need %.1f KB have %.1f KB (%.1f KB short)"',
              '"%s%s need %zu bytes have %zu bytes (%zu bytes short)"')
# The holders refusal's clause is composed from the holders the ensure wrote.
TEXT_HELD = "text += zone_arena_held_text(refusal.holders, refusal.device);"
TEXT_RELEASED = "refusal.released.scratch / MiB, refusal.released.onednn / MiB, refusal.released.runtime / MiB"


def claim_refusal_names_need_and_have(uc: str, zs: str) -> bool:
    c = code(uc)
    ensure = fn_body(c, ENSURE_SIG)
    text = fn_body(code(zs), TEXT_SIG)
    return (ordered(ensure, ZONES_READ, SCRATCH_HELD_FILL, "if (!zone_arena_rebuild_allowed(holders)) {", HELD_FILL,
                    "if (!arena_destroy()) {", "if (!arena_reserve(", RESERVE_FILL)
            and ensure.count("zone_arena_refusal_reason::REBUILD_HELD;") == 2
            and NO_CACHE_FILL in fn_body(c, WRAPPER_SIG)
            and all(z in text for z in TEXT_ZONES) and TEXT_FORMAT in text and all(f in text for f in TEXT_SMALL)
            and TEXT_HELD in text and TEXT_RELEASED in text)


# ---- a failed load puts back what it published ----

ABORT_SIG = "static bool ggml_sycl_abort_owner_effects_noexcept("
LOAD_END_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_end(ggml_sycl_load_txn txn,"
CAPTURE = "ggml_sycl_load_published_terms_capture(effect.owner.load.value);"
STAGE_LOOP = "for (int i = 0; i < n_devices; ++i) {"
# The restore runs outside every try stage of the abort, as the planned-load-terms drop does, so no failed stage skips it
# and the abort gains no catch handler; it is noexcept where it is defined.
RESTORE = "ggml_sycl_load_published_terms_restore(owner.load.value);"
RESTORE_DEF = "static void ggml_sycl_load_published_terms_restore(uint64_t txn) noexcept {"
FORGET = "ggml_sycl_load_published_terms_forget(result.token.load.value);"
COMMIT_CLEANUP = "if (result.cleanup_required) {"
COMMITTED = "if (result.committed) {"
RESTORE_SIG = "static void ggml_sycl_load_published_terms_restore(uint64_t txn) noexcept"
RECORD_RESTORE_SIG = "void load_published_record::restore(int device) const noexcept"
RECORD_NOTE_SIG = ("void load_published_record::note(const load_published_terms & before, "
                   "const load_published_terms & after)")
PUBLISH_SIGS = (
    "static ggml_sycl_inventory_planner_inputs ggml_sycl_inventory_publish_planner_inputs(",
    "static void populate_inventory_globals(ggml_backend_sycl_context * ctx, const ggml_sycl_tensor_inventory * "
    "inventory)",
    "static bool ggml_sycl_publish_load_ubatch_scratch(int device, uint32_t n_ubatch)",
)
# Not device-wide: the compute term's micro-batch is sticky by design and every load records the same fixed planning
# value (unified-cache.hpp says so at its setter).
EXEMPT_SETTERS = {"unified_cache_set_planned_compute_term_n_ubatch"}
SETTER_RE = re.compile(r"\b(unified_cache_set_planned_\w+)\(")
# Today's device-wide setters on the publish path, so a scan that reads the wrong bodies cannot pass by finding nothing.
PUBLISH_SETTERS_TODAY = {
    "unified_cache_set_planned_pp_pipeline_scratch_bytes",
    "unified_cache_set_planned_pp_moe_onednn_scratch",
    "unified_cache_set_planned_onednn_scratchpad_pair",
    "unified_cache_set_planned_pp_moe_onednn_kv_zone_slots",
    "unified_cache_set_planned_pp_moe_onednn_row_bytes",
    "unified_cache_set_planned_pp_moe_onednn_n_ubatch",
}


def publish_path_setters(gs: str) -> set:
    c = code(gs)
    found = set()
    for sig in PUBLISH_SIGS:
        found |= set(SETTER_RE.findall(fn_body(c, sig)))
    return found - EXEMPT_SETTERS


def depth_at(body: str, idx: int) -> int:
    """Brace depth at `idx` of a block that starts with its own '{': 1 is the block's top level."""
    return body.count("{", 0, idx) - body.count("}", 0, idx)


def restored_outside_every_try(c: str) -> bool:
    abort = fn_body(c, ABORT_SIG)
    k = abort.find(RESTORE)
    return k != -1 and abort.count(RESTORE) == 1 and depth_at(abort, k) == 1 and RESTORE_DEF in c


# The terms the planners publish while they compute a plan (planner_published_terms), which the record restores too.
PLANNER_SETTERS = {
    "unified_cache_set_planned_onednn_graph_scratch_shape",
    "unified_cache_set_planned_nonfa_attn_scratch_shape",
    "unified_cache_set_planned_moe_control_requirement",
    "unified_cache_set_planned_moe_host_scatter_scratch_bytes",
    "unified_cache_set_planned_moe_mmvq_scratch_bytes",
}
# The pair's setter stores the halves and then their total as the sum. The record puts the halves back on their own and
# the sum by compare-exchange, so a reserve's raise of the sum survives; together those restore what the setter wrote.
HALVES = "unified_cache_store_planned_onednn_pair_halves(device, prior.onednn_weights, prior.onednn_activations);"
SUM_CAS = "g_planned_onednn_scratchpad_bytes[device].compare_exchange_strong(published, prior.onednn_stored,"
PAIR_SETTER = "unified_cache_set_planned_onednn_scratchpad_pair"


def record_restored_setters(uc: str) -> set:
    body = fn_body(code(uc), RECORD_RESTORE_SIG)
    restored = set(SETTER_RE.findall(body))
    if HALVES in body and SUM_CAS in body:
        restored.add(PAIR_SETTER)
    return restored


def claim_failed_load_restores(gs: str, uc: str) -> bool:
    c = code(gs)
    stage = fn_body(c, STAGE_SIG)
    load_end = fn_body(c, LOAD_END_SIG)
    restored = record_restored_setters(uc)
    setters = publish_path_setters(gs)
    return (ordered(stage, CAPTURE, STAGE_LOOP)
            and restored_outside_every_try(c)
            and ordered(load_end, COMMIT_CLEANUP, FORGET, COMMITTED)
            and PUBLISH_SETTERS_TODAY <= setters and setters <= restored and PLANNER_SETTERS <= restored)


# ---- a failed load keeps what another publisher wrote since ----

# Each planner entry records its pass under the inventory lock it takes, before it publishes anything.
IMPL_LOCK = "std::lock_guard lock(g_tensor_inventory_mutex);"
PASS = "ggml_sycl_load_published_pass published(g_sycl_plan_load_effect_txn);"
IMPL_POPULATE = "populate_inventory_globals(ctx, inventory);"
PASS_END_SIG = "~ggml_sycl_load_published_pass()"
PASS_NOTE = "saved.devices[d].note(before[d], ggml_sycl::load_published_terms::snapshot(static_cast<int>(d)));"
# The restore takes the inventory lock before the record's mutex, the order a pass holds them in when it notes.
RESTORE_LOCKS = ("std::lock_guard inventory(g_tensor_inventory_mutex);",
                 "std::lock_guard<std::mutex> lock(saved.mutex);", "saved.devices[d].restore(static_cast<int>(d));")
# What a pass finds that this load did not write is another publisher's, and is what the restore puts back; what the load
# published of the sum is the pair's total, which is all the pair's setter writes to it.
NOTE_OTHER = "prior.take_groups(before, before.differing(written));"
NOTE_SUM = "written.onednn_stored = after.onednn_weights + after.onednn_activations;"
# A group goes back only while it holds what the load published and differs from what it held before.
HELD = ("const uint32_t back = ~load_published_terms::snapshot(device).differing(written) & "
        "prior.differing(written);")
GROUP_SETTERS = (
    ("LOAD_PUBLISHED_GROUP_PIPELINE", "unified_cache_set_planned_pp_pipeline_scratch_bytes("),
    ("LOAD_PUBLISHED_GROUP_RING", "unified_cache_set_planned_pp_moe_onednn_scratch("),
    ("LOAD_PUBLISHED_GROUP_RING", "unified_cache_set_planned_pp_moe_onednn_kv_zone_slots("),
    ("LOAD_PUBLISHED_GROUP_RING", "unified_cache_set_planned_pp_moe_onednn_n_ubatch("),
    ("LOAD_PUBLISHED_GROUP_RING_ROWS", "unified_cache_set_planned_pp_moe_onednn_row_bytes("),
    ("LOAD_PUBLISHED_GROUP_ONEDNN_PAIR", "unified_cache_store_planned_onednn_pair_halves("),
    ("LOAD_PUBLISHED_GROUP_GRAPH", "unified_cache_set_planned_onednn_graph_scratch_shape("),
    ("LOAD_PUBLISHED_GROUP_NONFA", "unified_cache_set_planned_nonfa_attn_scratch_shape("),
    ("LOAD_PUBLISHED_GROUP_CONTROL", "unified_cache_set_planned_moe_control_requirement("),
    ("LOAD_PUBLISHED_GROUP_SCATTER", "unified_cache_set_planned_moe_host_scatter_scratch_bytes("),
    ("LOAD_PUBLISHED_GROUP_ZONE_PLAN", "onednn_zone_plan_put_back("),
    ("LOAD_PUBLISHED_GROUP_MOE_MMVQ", "unified_cache_set_planned_moe_mmvq_scratch_bytes("),
)
# The sum is never stored plainly: a plain store would put back over a reserve's raise.
SUM_PLAIN = ("unified_cache_set_planned_onednn_scratchpad_bytes(", PAIR_SETTER + "(",
             "g_planned_onednn_scratchpad_bytes[device].store(")


def guarded(body: str, group: str, setter: str) -> bool:
    k = body.find("if (back & " + group + ") {")
    return k != -1 and setter in block_at(body, k) and body.count(setter) == 1


def claim_restore_keeps_a_later_publish(gs: str, uc: str) -> bool:
    c = code(gs)
    u = code(uc)
    note = fn_body(u, RECORD_NOTE_SIG)
    restore = fn_body(u, RECORD_RESTORE_SIG)
    entries = [fn_body(c, sig) for sig in (EARLY_SIG, LATE_SIG)]
    return (all(ordered(e, IMPL_LOCK, PASS, IMPL_POPULATE) and e.count(PASS) == 1 for e in entries)
            and PASS_NOTE in fn_body(c, PASS_END_SIG)
            and ordered(fn_body(c, RESTORE_SIG), *RESTORE_LOCKS)
            and NOTE_OTHER in note and NOTE_SUM in note
            and ordered(restore, HELD, *(s for _, s in GROUP_SETTERS))
            and all(guarded(restore, g, s) for g, s in GROUP_SETTERS)
            and SUM_CAS in restore and not any(p in restore for p in SUM_PLAIN))


# ---- the claims, and a witness per claim ----


def test_no_abort_on_the_load_plan_path():
    assert claim_no_abort_on_the_path(GS), (
        "a GGML_ABORT is on the load's plan path, or one of its functions is gone: a load whose zones cannot be sized "
        "must be refused, never abort the process")


def test_witness_entry_ensure_aborts():
    assert not claim_no_abort_on_the_path(mutate_in(GS, COMPUTE_SIG, "return false; }", ABORT_LINE + " }"))


def test_both_ensures_refuse_by_name():
    assert claim_both_ensures_refuse_by_name(GS), (
        "compute_and_store_plan_for_inventory does not return bool, or a zone ensure that fails does not log the "
        "refusal line and return false")


def test_witness_entry_ensure_fills_no_refusal():
    m = mutate_in(GS, COMPUTE_SIG, "!ggml_sycl_ensure_load_arena_zones(ctx->device, refusal)) { GGML_LOG",
                  "!ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device)) { GGML_LOG")
    assert not claim_both_ensures_refuse_by_name(m)


def test_witness_fallback_ensure_plans_on():
    c = code(GS)
    body = fn_body(c, COMPUTE_SIG)
    k = body.find(FALLBACK_ENSURE)
    assert k != -1, "witness anchor: the fallback ensure"
    tail = body[k:].replace("return false; }", "}", 1)
    m = c.replace(body, body[:k] + tail, 1)
    assert not claim_both_ensures_refuse_by_name(m)


def test_every_caller_propagates_the_refusal():
    assert claim_every_caller_propagates(GS), (
        "a caller drops the refused plan: each planner entry must report it through plan_refused, and the stage must "
        "return GGML_SYCL_LIFECYCLE_PLAN_REJECTED from both passes, the late one before it checks the candidate")


def test_witness_stage_early_ignores_the_refusal():
    m = mutate_in(GS, STAGE_SIG, "if (plan_refused) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED; } if (planned >= 0) {",
                  "if (planned >= 0) {")
    assert not claim_every_caller_propagates(m)


def test_witness_stage_late_checks_the_candidate_first():
    late_check = "if (plan_refused) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED; }"
    m = mutate_in(GS, STAGE_SIG, STAGE_LATE,
                  "decision.snapshot = ggml_sycl_set_tensor_inventory_impl(backend, inventory, &plan_refused, "
                  "decision.effects);")
    m = mutate_in(m, STAGE_SIG, "if (ggml_sycl_stage_refuse_lazy_plan(decision.snapshot->plan.get(), early)) {",
                  late_check + " if (ggml_sycl_stage_refuse_lazy_plan(decision.snapshot->plan.get(), early)) {")
    assert not claim_every_caller_propagates(m)


def test_witness_late_entry_drops_the_flag():
    m = mutate_in(GS, LATE_SIG, "if (plan_refused != nullptr) { *plan_refused = true; } return {}; }", "}")
    assert not claim_every_caller_propagates(m)


def test_witness_early_entry_drops_the_flag():
    m = mutate_in(GS, EARLY_SIG, EARLY_PROPAGATE, "(void) " + PLAN_CALL + ";")
    assert not claim_every_caller_propagates(m)


# The exported entries keep their void signatures across the dlopen boundary: one that asks for the refusal itself and
# aborts on it would take the process down where the stage refuses the load.
def test_witness_exported_early_entry_aborts_on_the_flag():
    m = mutate_in(GS, EXPORTED_EARLY_SIG, "ggml_sycl_compute_placement_plan_early_impl(backend, inventory, nullptr);",
                  "bool refused = false; ggml_sycl_compute_placement_plan_early_impl(backend, inventory, &refused); "
                  "GGML_ASSERT(!refused);")
    assert not claim_every_caller_propagates(m)


def test_witness_exported_late_entry_aborts_on_the_flag():
    m = mutate_in(GS, EXPORTED_LATE_SIG,
                  "(void) ggml_sycl_set_tensor_inventory_impl(backend, inventory, nullptr, effects);",
                  "bool refused = false; (void) ggml_sycl_set_tensor_inventory_impl(backend, inventory, &refused, "
                  "effects); GGML_ASSERT(!refused);")
    assert not claim_every_caller_propagates(m)


def test_refusal_names_need_and_have():
    assert claim_refusal_names_need_and_have(UC, ZS), (
        "the ensure does not write each zone's need (the sizes it decided on) and have (the arena's capacities) at "
        "every false exit, or zone_arena_refusal_text does not print both for every zone")


def test_witness_need_written_from_the_capacities():
    m = mutate_in(UC, ENSURE_SIG, "REBUILD_HELD; refusal->device = dev_id; " + NEED,
                  "REBUILD_HELD; refusal->device = dev_id; refusal->need = arena_zones;")
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_held_refusal_writes_its_need_from_the_capacities():
    m = mutate_in(UC, ENSURE_SIG, HELD_FILL, HELD_FILL.replace(NEED, "refusal->need = arena_zones;"))
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_held_refusal_drops_its_holders():
    m = mutate_in(UC, ENSURE_SIG, HELD_FILL, HELD_FILL.replace(HOLDERS, ""))
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_scratch_refusal_says_nothing_would_be_kept():
    m = mutate_in(UC, ENSURE_SIG, "refusal->keep_deferred = true;", "refusal->keep_deferred = false;")
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_a_third_held_refusal_that_names_nothing():
    m = mutate_in(UC, ENSURE_SIG, "if (!arena_destroy()) {", "if (refusal != nullptr) { refusal->reason = "
                  "zone_arena_refusal_reason::REBUILD_HELD; } if (!arena_destroy()) {")
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_reserve_failure_writes_nothing():
    m = mutate_in(UC, ENSURE_SIG, RESERVE_FILL, "return false;")
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_have_read_after_the_arena_is_destroyed():
    m = mutate_in(UC, ENSURE_SIG, ZONES_READ, "")
    m = mutate_in(m, ENSURE_SIG, "if (!arena_reserve(", ZONES_READ + " if (!arena_reserve(")
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_text_drops_the_have():
    m = mutate_in(ZS, TEXT_SIG, TEXT_FORMAT, '"%s%s need %.1f MB"')
    assert not claim_refusal_names_need_and_have(UC, m)


def test_witness_reserve_failure_names_the_destroyed_arena_as_have():
    m = mutate_in(UC, ENSURE_SIG, RESERVE_HAVE + " " + RELEASED, HAVE)
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_text_drops_the_released_arena():
    m = mutate_in(ZS, TEXT_SIG, TEXT_RELEASED, "0.0, 0.0, 0.0")
    assert not claim_refusal_names_need_and_have(UC, m)


def test_witness_held_refusal_writes_no_holders():
    m = mutate_in(UC, ENSURE_SIG, HOLDERS, "")
    assert not claim_refusal_names_need_and_have(m, ZS)


def test_witness_text_remedy_ignores_the_holders():
    m = mutate_in(ZS, TEXT_SIG, TEXT_HELD, "text += zone_arena_held_text(zone_arena_holders{}, refusal.device);")
    assert not claim_refusal_names_need_and_have(UC, m)


def test_witness_text_prints_a_small_shortfall_in_mb():
    m = mutate_in(ZS, TEXT_SIG, TEXT_SMALL[0], '"%s%s need %.1f MB have %.1f MB (%.1f MB short)"')
    assert not claim_refusal_names_need_and_have(UC, m)


def test_a_failed_load_puts_back_what_it_published():
    assert claim_failed_load_restores(GS, UC), (
        "a failed load does not put back the device-wide terms it published: the stage must capture them before its "
        "first device plans, the owner-effects abort must restore them, load_end must forget them once committed, "
        "and the restore must call every device-wide setter on the load's publish path")


def test_witness_abort_restores_nothing():
    m = mutate_in(GS, ABORT_SIG, RESTORE, "")
    assert not claim_failed_load_restores(m, UC)


def test_witness_restore_inside_a_try_stage():
    m = mutate_in(GS, ABORT_SIG, RESTORE, "try { " + RESTORE + " } catch (...) { clean = false; }")
    assert not claim_failed_load_restores(m, UC)


def test_witness_restore_declared_throwing():
    m = code(GS).replace(RESTORE_DEF, RESTORE_DEF.replace(" noexcept {", " {"), 1)
    assert not claim_failed_load_restores(m, UC)


def test_witness_capture_after_the_first_device():
    m = mutate_in(GS, STAGE_SIG, CAPTURE, "")
    m = mutate_in(m, STAGE_SIG, "rollback.started = true;", "rollback.started = true; " + CAPTURE)
    assert not claim_failed_load_restores(m, UC)


def test_witness_restore_drops_the_slot_zones():
    m = mutate_in(UC, RECORD_RESTORE_SIG, "unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(", "(void) (")
    assert not claim_failed_load_restores(GS, m)


def test_witness_restore_drops_a_planner_term():
    m = mutate_in(UC, RECORD_RESTORE_SIG, "unified_cache_set_planned_moe_host_scatter_scratch_bytes(", "(void) (")
    assert not claim_failed_load_restores(GS, m)


def test_witness_restore_drops_the_moe_mmvq_bound():
    m = mutate_in(UC, RECORD_RESTORE_SIG, "unified_cache_set_planned_moe_mmvq_scratch_bytes(", "(void) (")
    assert not claim_failed_load_restores(GS, m)


def test_witness_restore_drops_the_sum():
    m = mutate_in(UC, RECORD_RESTORE_SIG, SUM_CAS, "std::atomic<size_t>{}.compare_exchange_strong(published, "
                  "prior.onednn_stored,")
    assert not claim_failed_load_restores(GS, m)


def test_witness_publish_path_gains_an_unrestored_setter():
    m = mutate_in(GS, PUBLISH_SIGS[1], "g_model_n_layer = inventory->n_layer;",
                  "ggml_sycl::unified_cache_set_planned_scratch_hold(ctx->device, 0, 0); "
                  "g_model_n_layer = inventory->n_layer;")
    assert not claim_failed_load_restores(m, UC)


def test_witness_forget_before_the_commit_cleanup():
    m = mutate_in(GS, LOAD_END_SIG, FORGET, "")
    m = mutate_in(m, LOAD_END_SIG, COMMIT_CLEANUP, FORGET + " " + COMMIT_CLEANUP)
    assert not claim_failed_load_restores(m, UC)



def test_a_failed_load_keeps_what_another_publisher_wrote_since():
    assert claim_restore_keeps_a_later_publish(GS, UC), (
        "a failed load's restore can put back over a live context's publish: both planner entries must record their "
        "pass under the inventory lock before they publish, the restore must take that lock before the record's, a "
        "group must go back only while it holds what the load published, and the sum only by compare-exchange")


def test_witness_early_entry_records_no_pass():
    m = mutate_in(GS, EARLY_SIG, PASS, "")
    assert not claim_restore_keeps_a_later_publish(m, UC)


def test_witness_late_entry_records_after_it_publishes():
    m = mutate_in(GS, LATE_SIG, PASS, "")
    m = mutate_in(m, LATE_SIG, IMPL_POPULATE, IMPL_POPULATE + " " + PASS)
    assert not claim_restore_keeps_a_later_publish(m, UC)


def test_witness_pass_notes_nothing():
    m = mutate_in(GS, PASS_END_SIG, PASS_NOTE, "")
    assert not claim_restore_keeps_a_later_publish(m, UC)


def test_witness_restore_without_the_inventory_lock():
    m = mutate_in(GS, RESTORE_SIG, RESTORE_LOCKS[0], "")
    assert not claim_restore_keeps_a_later_publish(m, UC)


def test_witness_restore_takes_the_locks_in_the_other_order():
    m = mutate_in(GS, RESTORE_SIG, RESTORE_LOCKS[0], "")
    m = mutate_in(m, RESTORE_SIG, RESTORE_LOCKS[1], RESTORE_LOCKS[1] + " " + RESTORE_LOCKS[0])
    assert not claim_restore_keeps_a_later_publish(m, UC)


def test_witness_note_ignores_another_publisher():
    m = mutate_in(UC, RECORD_NOTE_SIG, NOTE_OTHER, "")
    assert not claim_restore_keeps_a_later_publish(GS, m)


def test_witness_note_takes_a_raised_sum_as_the_loads():
    m = mutate_in(UC, RECORD_NOTE_SIG, NOTE_SUM, "")
    assert not claim_restore_keeps_a_later_publish(GS, m)


def test_witness_restore_puts_back_every_group():
    m = mutate_in(UC, RECORD_RESTORE_SIG, HELD, "const uint32_t back = LOAD_PUBLISHED_GROUP_ALL;")
    assert not claim_restore_keeps_a_later_publish(GS, m)


def test_witness_restore_puts_back_the_ring_unguarded():
    m = mutate_in(UC, RECORD_RESTORE_SIG, "if (back & LOAD_PUBLISHED_GROUP_RING) {", "{")
    assert not claim_restore_keeps_a_later_publish(GS, m)


def test_witness_restore_puts_back_the_moe_mmvq_bound_unguarded():
    m = mutate_in(UC, RECORD_RESTORE_SIG, "if (back & LOAD_PUBLISHED_GROUP_MOE_MMVQ) {", "{")
    assert not claim_restore_keeps_a_later_publish(GS, m)


def test_witness_restore_stores_the_sum_plainly():
    m = mutate_in(UC, RECORD_RESTORE_SIG, SUM_CAS, "unified_cache_set_planned_onednn_scratchpad_bytes(device, "
                  "prior.onednn_stored); (void) (published,")
    assert not claim_restore_keeps_a_later_publish(GS, m)


# ---- a failed load puts back the two-card MoE flag its planner set ----

# The planner sets g_moe_multi_gpu_active in compute_and_store_plan_for_inventory, inside the pass of either entry, and
# live models' dispatch reads it, as does the next load's planner before it writes it. A failed load puts back what it
# held before, kept as a group is: the capture reads it, each pass notes it, and the restore puts the prior back by
# compare-exchange, so a write since by another writer survives.
COMPUTE_FLAG_WRITES = ("g_moe_multi_gpu_active.store(true, std::memory_order_release);",
                       "g_moe_multi_gpu_active.store(false, std::memory_order_release);",
                       "g_moe_multi_gpu_active.exchange(false, std::memory_order_acq_rel)")
CAPTURE_SIG = "static void ggml_sycl_load_published_terms_capture(uint64_t txn)"
PASS_START_SIG = "explicit ggml_sycl_load_published_pass(uint64_t load_txn)"
FLAG_CAPTURE = ("saved.moe_multi_gpu_prior = saved.moe_multi_gpu_written = "
                "g_moe_multi_gpu_active.load(std::memory_order_acquire);")
FLAG_BEFORE = "moe_multi_gpu_before = g_moe_multi_gpu_active.load(std::memory_order_acquire);"
FLAG_NOTE = ("if (moe_multi_gpu_before != saved.moe_multi_gpu_written) { saved.moe_multi_gpu_prior = "
             "moe_multi_gpu_before; } saved.moe_multi_gpu_written = "
             "g_moe_multi_gpu_active.load(std::memory_order_acquire);")
FLAG_RESTORE = ("bool published = saved.moe_multi_gpu_written; if (saved.moe_multi_gpu_prior != published) { (void) "
                "g_moe_multi_gpu_active.compare_exchange_strong(published, saved.moe_multi_gpu_prior, "
                "std::memory_order_acq_rel); }")
TXN_CHECK = "if (txn == 0 || saved.txn != txn) { return; }"
ENTRY_PLAN_CALL = "compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct)"


def claim_failed_load_restores_the_two_card_flag(gs: str) -> bool:
    c = code(gs)
    compute = fn_body(c, COMPUTE_SIG)
    restore = fn_body(c, RESTORE_SIG)
    entries = [fn_body(c, sig) for sig in (EARLY_SIG, LATE_SIG)]
    return (all(w in compute for w in COMPUTE_FLAG_WRITES)
            and all(ordered(e, PASS, ENTRY_PLAN_CALL) for e in entries)
            and FLAG_CAPTURE in fn_body(c, CAPTURE_SIG)
            and FLAG_BEFORE in fn_body(c, PASS_START_SIG)
            and FLAG_NOTE in fn_body(c, PASS_END_SIG)
            and ordered(restore, RESTORE_LOCKS[0], RESTORE_LOCKS[1], TXN_CHECK, FLAG_RESTORE)
            and "g_moe_multi_gpu_active.store(" not in restore)


def test_a_failed_load_puts_back_the_two_card_flag():
    assert claim_failed_load_restores_the_two_card_flag(GS), (
        "a failed load leaves the two-card MoE flag its planner set: the capture must read g_moe_multi_gpu_active, "
        "each planner pass must note it, and the restore must put the prior back by compare-exchange after the "
        "inventory lock and the transaction check")


def test_witness_capture_skips_the_flag():
    m = mutate_in(GS, CAPTURE_SIG, FLAG_CAPTURE, "")
    assert not claim_failed_load_restores_the_two_card_flag(m)


def test_witness_pass_does_not_note_the_flag():
    m = mutate_in(GS, PASS_END_SIG, FLAG_NOTE, "")
    assert not claim_failed_load_restores_the_two_card_flag(m)


def test_witness_restore_leaves_the_flag():
    m = mutate_in(GS, RESTORE_SIG, FLAG_RESTORE, "")
    assert not claim_failed_load_restores_the_two_card_flag(m)


def test_witness_restore_stores_the_flag_plainly():
    m = mutate_in(GS, RESTORE_SIG, FLAG_RESTORE,
                  "g_moe_multi_gpu_active.store(saved.moe_multi_gpu_prior, std::memory_order_release);")
    assert not claim_failed_load_restores_the_two_card_flag(m)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
