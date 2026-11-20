"""Source contract for the load stage's two passes: every refusal is decided before the stage's effects run.

ggml_backend_sycl_stage_load_inventory_plan plans every device first (the decide pass) and configures a device's host
zones, sets up its layer streaming and releases its stale ring only after the last refusal of the pass is decided (the
effect pass). A load runs the stage twice, an early pass and then a late one, and the early pass's effect step runs
before the late pass decides. A pass refused on any device therefore runs none of its own effects, and a load refused in
its late pass has already run its early pass's effect, the stale ring's release. The rollback guard's abort puts back
what the decide pass published. The decide pass still changes a device in the planner's zone decision: the entry
ensure's last resort below releases a ring, and an arena nothing holds may be rebuilt for the plan's zones, which the
refused load's abort tears down again when no model is live (pinned by test-sycl-last-release-source.py). These checks
pin the rest:

The decide pass releases the PP MoE oneDNN scratch ring in one place only: ggml_sycl_ensure_load_arena_zones, the
planner's entry ensure, as its last resort when a rebuild is needed and the ring's slot buffers are the arena's only
holders (zone_arena_only_ring_holds). Nothing else the decide pass runs (populate_inventory_globals, the VRAM budget,
compute_and_store_plan_for_inventory, the micro-batch publish, either planner entry) releases a ring in its own body.
populate keeps the planned KV-zone flag write, which the abort of a refused load puts back.

The planner entries run none of the stage's effects (layer streaming, host zones, the stale ring's release); the one
ring release they can reach is the entry ensure's last resort above. Layer streaming's build_layer_map,
register_host_ptr and allocate_buffers, and the stale KV-zone ring's release, run within the stage only in the effect
step (ggml_sycl_run_inventory_effects for a late pass, ggml_sycl_run_early_inventory_effect for an early one, both
through ggml_sycl_release_stale_kv_zone_ring), and the stage configures the host zones after them. The late entry hands
what it decided back in a ggml_sycl_inventory_effects record that carries its own copy of the streamed inventory; no
record lives outside a function. Each effect helper takes g_tensor_inventory_mutex for its own work and the stage never
takes it, so the stage provisions its retained snapshots outside the inventory lock.

In the stage, the decide loop runs no effect, and the per-device test failure sits inside it, so a failure after the
first device runs no effect for it. Every PLAN_REJECTED and EFFECT_FAILED return of the stage comes before its first
effect; after the first configure_host_zones_for_plan only the final status is returned. The two owner checks that
follow an effect throw: they detect a competing owner, which the rollback guard handles, and refuse nothing the load
decided.

Each device's effects run from the snapshot that device staged, under the check that the load's owner is still the
active token and the load's candidate is still the snapshot the last device staged, before and after that device's
effects. The leases of the candidates the devices replaced are taken from the slot scope's out-slot inside the decide
loop, kept in a stage-local list declared before the rollback guard, and dropped only after the effect loop.

The ring's release lines are WARN, so they survive default verbosity. A rebuild whose arena teardown refuses refuses the
ensure by its own reason, and no arena is reserved after it (unified-cache.cpp).

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live
inside test_*() functions. Checks read comment-stripped, whitespace-normalised text. Each check has a mutation witness
that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
GS = (SYCL / "ggml-sycl.cpp").read_text()
UC = (SYCL / "unified-cache.cpp").read_text()

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


# ---- the functions, by the part of each signature that names the return type ----

POPULATE_SIG = "static void populate_inventory_globals("
BUDGET_SIG = "static void compute_vram_budget_for_plan("
COMPUTE_SIG = "static bool compute_and_store_plan_for_inventory("
PUBLISH_SIG = "static bool ggml_sycl_publish_load_ubatch_scratch("
EARLY_SIG = "static int ggml_sycl_compute_placement_plan_early_impl("
LATE_SIG = "static std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> ggml_sycl_set_tensor_inventory_impl("
ENSURE_LOAD_SIG = "static bool ggml_sycl_ensure_load_arena_zones("
EFFECTS_SIG = "static void ggml_sycl_run_inventory_effects("
EARLY_EFFECT_SIG = "static void ggml_sycl_run_early_inventory_effect("
STALE_SIG = "static void ggml_sycl_release_stale_kv_zone_ring("
STAGE_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_load_inventory_plan("
EXPORTED_EARLY_SIG = "void ggml_backend_sycl_compute_placement_plan_early(ggml_backend_t backend,"
EXPORTED_LATE_SIG = "void ggml_backend_sycl_set_tensor_inventory(ggml_backend_t backend,"
UC_ENSURE_SIG = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
                 "const zone_arena_ensure_options & options)")
DECIDE_SIGS = (POPULATE_SIG, BUDGET_SIG, COMPUTE_SIG, PUBLISH_SIG, EARLY_SIG, LATE_SIG)

RING_RELEASE = "release_pp_moe_onednn_scratch_ring("
STALE_RELEASE = "ggml_sycl_release_stale_kv_zone_ring("
ENTRY_ENSURE = "ggml_sycl_ensure_load_arena_zones(ctx->device, refusal)"
ONLY_RING_GUARD = ("if (!ggml_sycl::zone_arena_only_ring_holds(refusal.reason == "
                   "ggml_sycl::zone_arena_refusal_reason::REBUILD_HELD, refusal.holders)) { return false; }")
FLAG_WRITE = "ggml_sycl::unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(ctx->device, false, false);"
STREAM_CALLS = ("build_layer_map(", "register_host_ptr(", "allocate_buffers(")
CONFIGURE = "ggml_sycl_configure_host_zones_for_plan("
EFFECT_CALLS = ("ggml_sycl_run_inventory_effects(", "ggml_sycl_run_early_inventory_effect(", CONFIGURE, STALE_RELEASE,
                RING_RELEASE, "lifecycle_stage_probe_placement_plan(") + STREAM_CALLS
INVENTORY_LOCK = "g_tensor_inventory_mutex"
DECIDE_LOOP = "for (int i = 0; i < n_devices; ++i) {"
EFFECT_LOOP = "for (const late_decision & decision : late_decisions) {"
TEST_FAILURE = 'throw std::runtime_error("deterministic stage_load_inventory_plan failure");'
REFUSAL_RETURNS = ("return GGML_SYCL_LIFECYCLE_PLAN_REJECTED;", "return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;")
FINAL_RETURN = "return n_devices == 0 ? GGML_SYCL_LIFECYCLE_NOT_FOUND : GGML_SYCL_LIFECYCLE_OK;"
LEASE_LIST = "std::vector<ggml_sycl::ring_lease> displaced_leases;"
LEASE_DROP = "for (ggml_sycl::ring_lease & lease : displaced_leases) { lazy_plan_slot_scope::drop_displaced(lease); }"


def stage_try(gs: str) -> str:
    """The stage's try block, up to its catch."""
    st = fn_body(code(gs), STAGE_SIG)
    t = st.find("try {")
    k = st.find("} catch (...) {", t)
    return st[t:k] if -1 not in (t, k) else ""


# ---- the decide pass's one ring release ----


def claim_decide_pass_releases_the_ring_only_at_the_entry_ensure(gs: str) -> bool:
    c = code(gs)
    bodies = [fn_body(c, sig) for sig in DECIDE_SIGS]
    if not all(bodies) or any(RING_RELEASE in b or STALE_RELEASE in b for b in bodies):
        return False
    ensure = fn_body(c, ENSURE_LOAD_SIG)
    compute = fn_body(c, COMPUTE_SIG)
    loop = block_at(stage_try(gs), stage_try(gs).find(DECIDE_LOOP))
    return (
        ensure.count(RING_RELEASE) == 1
        and ordered(ensure, "ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options)",
                    ONLY_RING_GUARD, "cache->" + RING_RELEASE, "refusal = {};",
                    "return ggml_sycl::unified_cache_ensure_planned_arena_zones(device, &refusal, options);")
        and compute.count("ggml_sycl_ensure_load_arena_zones(") == 1
        and ordered(compute, ENTRY_ENSURE, "compute_multi_device_plan(")
        and c.count("ggml_sycl_ensure_load_arena_zones(") == 2  # the definition and the entry ensure
        and bool(loop)
        and RING_RELEASE not in loop
        and STALE_RELEASE not in loop
    )


def claim_populate_keeps_the_flag_write(gs: str) -> bool:
    return FLAG_WRITE in fn_body(code(gs), POPULATE_SIG)


# ---- the planner entries run none of the stage's effects ----


def claim_entries_run_no_stage_effect(gs: str) -> bool:
    c = code(gs)
    early, late = fn_body(c, EARLY_SIG), fn_body(c, LATE_SIG)
    effects, stale = fn_body(c, EFFECTS_SIG), fn_body(c, STALE_SIG)
    early_effect = fn_body(c, EARLY_EFFECT_SIG)
    if not (early and late and effects and stale and early_effect):
        return False
    banned = STREAM_CALLS + (CONFIGURE, RING_RELEASE, STALE_RELEASE)
    if any(t in early or t in late for t in banned):
        return False
    return (
        ordered(effects, "if (effects.device < 0) { return; }", "std::lock_guard lock(g_tensor_inventory_mutex);",
                "ggml_sycl_release_stale_kv_zone_ring(effects.device);", "if (!effects.stream) { return; }",
                "mgr.build_layer_map(effects.stream_inventory.data(), effects.stream_inventory.size());",
                "mgr.register_host_ptr(", "mgr.allocate_buffers(q)")
        and "ring_cache->" + RING_RELEASE in stale
        and "decided.stream_inventory" in late
        and late.count("effects = std::move(decided);") == 1
        and ordered(late, "if (!compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct)) {",
                    "return {}; }", "effects = std::move(decided);")
        and ordered(early_effect, "std::lock_guard lock(g_tensor_inventory_mutex);",
                    "ggml_sycl_release_stale_kv_zone_ring(device);")
        and ordered(fn_body(c, EXPORTED_EARLY_SIG), "ggml_sycl_compute_placement_plan_early_impl(backend, inventory, "
                    "nullptr);", "if (device >= 0) {", "ggml_sycl_run_early_inventory_effect(device);")
        and ordered(fn_body(c, EXPORTED_LATE_SIG), "ggml_sycl_set_tensor_inventory_impl(backend, inventory, nullptr, "
                    "effects);", "ggml_sycl_run_inventory_effects(effects);")
    )


def claim_streamed_inventory_is_a_stage_local(gs: str) -> bool:
    c = code(gs)
    rec = block_at(c, c.find("struct ggml_sycl_inventory_effects {"))
    file_scope = re.compile(
        r"(?m)^(?:static\s+|thread_local\s+|inline\s+)*(?:std::vector<\s*)?"
        r"(?:ggml_sycl_inventory_effects|late_decision)\b[^(]*?\b\w+\s*(?:[;={]|\[)"
    )
    st = stage_try(gs)
    return (
        "std::vector<std::pair<std::string, size_t>> stream_inventory;" in rec
        and not file_scope.search(strip_comments(gs))
        and "std::vector<late_decision> late_decisions;" in st
        and "ggml_sycl_inventory_effects effects;" in block_at(st, st.find("struct late_decision {"))
    )


# ---- the stage: every refusal before the first effect ----


def claim_stage_decides_every_refusal_before_any_effect(gs: str) -> bool:
    st = stage_try(gs)
    loop = block_at(st, st.find(DECIDE_LOOP))
    if not loop or TEST_FAILURE not in loop or any(t in loop for t in EFFECT_CALLS):
        return False
    effect_at = [st.find(t) for t in EFFECT_CALLS if st.find(t) != -1]
    if not effect_at:
        return False
    first_effect = min(effect_at)
    if any(st.find(r, first_effect) != -1 for r in REFUSAL_RETURNS):
        return False
    after_configure = st[st.find(CONFIGURE) :]
    return after_configure.count("return ") == 1 and FINAL_RETURN in after_configure


def claim_stage_takes_no_inventory_lock(gs: str) -> bool:
    st = fn_body(code(gs), STAGE_SIG)
    return bool(st) and INVENTORY_LOCK not in st and "ggml_sycl_run_early_inventory_effect(device);" in st


def claim_effects_check_the_last_snapshot(gs: str) -> bool:
    st = stage_try(gs)
    loop = block_at(st, st.find(EFFECT_LOOP))
    current = block_at(loop, loop.find("const auto effects_are_current = [&]() {"))
    return (
        "late_decisions.empty() ? nullptr : late_decisions.back().snapshot;" in st
        and "decision.snapshot->load_txn_id == owner.load.value" in current
        and "candidate_is_current(staged)" in current
        and ordered(loop, 'if (!effects_are_current()) { throw std::runtime_error("stale host provisioning candidate '
                    'before allocation"); }', "ggml_sycl_run_inventory_effects(decision.effects);",
                    "ggml_sycl::get_unified_cache_for_device(decision.device);",
                    "ggml_sycl::ggml_sycl_configure_host_zones_for_plan(cache, decision.snapshot->plan);",
                    'if (!effects_are_current()) { throw std::runtime_error("stale host provisioning candidate after '
                    'allocation"); }')
    )


def claim_leases_outlive_the_effects(gs: str) -> bool:
    st = stage_try(gs)
    loop = block_at(st, st.find(DECIDE_LOOP))
    effects = block_at(st, st.find(EFFECT_LOOP))
    return (
        ordered(st, "lazy_plan_slot_scope coverage_slot(coverage);", LEASE_LIST, "struct owner_rollback_guard {",
                DECIDE_LOOP)
        and ordered(loop, "ggml_sycl::ring_lease replaced = coverage_slot.take_displaced();",
                    "displaced_leases.push_back(std::move(replaced));")
        and "drop_displaced(" not in loop
        and st.count("drop_displaced(") == 1
        and bool(effects)
        and ordered(st, effects, LEASE_DROP, "rollback.released = true;")
        and "std::shared_ptr<const ggml_sycl::lifecycle_plan_snapshot> snapshot;" in block_at(
            st, st.find("struct late_decision {"))
    )


def claim_ring_release_lines_warn(gs: str) -> bool:
    c = code(gs)
    bodies = (fn_body(c, STALE_SIG), fn_body(c, ENSURE_LOAD_SIG))
    return all(b and b.count("GGML_LOG_WARN(") == 2 and "GGML_LOG_INFO(" not in b and "GGML_LOG_ERROR(" not in b
               for b in bodies)


def claim_refused_teardown_reserves_nothing(uc: str) -> bool:
    ens = fn_body(code(uc), UC_ENSURE_SIG)
    destroy = ens.find("if (!arena_destroy()) {")
    on_refusal = block_at(ens, destroy)
    return (
        destroy != -1
        and "return refuse_teardown(cause, arena_zones);" in on_refusal
        and "arena_reserve(" not in on_refusal
        and ordered(ens, "const zone_arena_teardown_refusal earlier = arena_teardown_refusal();",
                    "zone_arena_teardown_left_closed(earlier)) { return refuse_teardown(earlier,", "if (arena_active()) {",
                    "if (!arena_destroy()) {", "if (!arena_reserve(")
        and "refusal->reason = zone_arena_refusal_reason::TEARDOWN_REFUSED;" in ens
    )


# ---- the checks ----


def test_decide_pass_releases_the_ring_only_at_the_entry_ensure():
    assert claim_decide_pass_releases_the_ring_only_at_the_entry_ensure(GS), (
        "a decide step of the load releases the PP MoE ring outside the entry ensure's last resort, or that resort "
        "lost its only-the-ring guard: a load refused after it would have released a ring a live context may use")


def test_witness_populate_releases_the_ring_again():
    mutated = mutate_in(GS, POPULATE_SIG, FLAG_WRITE,
                        "ggml_sycl::get_existing_unified_cache_for_device(ctx->device)->"
                        "release_pp_moe_onednn_scratch_ring(); " + FLAG_WRITE)
    assert not claim_decide_pass_releases_the_ring_only_at_the_entry_ensure(mutated)


def test_witness_last_resort_without_its_guard():
    mutated = mutate_in(GS, ENSURE_LOAD_SIG, "if (!ggml_sycl::zone_arena_only_ring_holds(",
                        "if (false && !ggml_sycl::zone_arena_only_ring_holds(")
    assert not claim_decide_pass_releases_the_ring_only_at_the_entry_ensure(mutated)


def test_witness_fallback_ensure_releases_the_ring_too():
    mutated = mutate_in(GS, COMPUTE_SIG, "if (!ggml_sycl::unified_cache_ensure_planned_arena_zones(ctx->device, "
                        "&refusal, ggml_sycl_load_arena_options())) {",
                        "if (!ggml_sycl_ensure_load_arena_zones(ctx->device, refusal)) {")
    assert not claim_decide_pass_releases_the_ring_only_at_the_entry_ensure(mutated)


def test_populate_keeps_the_flag_write():
    assert claim_populate_keeps_the_flag_write(GS), (
        "populate_inventory_globals no longer resets the planned KV-zone slot flags")


def test_witness_flag_write_gone():
    assert not claim_populate_keeps_the_flag_write(mutate_in(GS, POPULATE_SIG, FLAG_WRITE, ""))


def test_entries_run_no_stage_effect():
    assert claim_entries_run_no_stage_effect(GS), (
        "a planner entry runs one of the stage's effects (layer streaming, host zones, the stale ring's release) in "
        "its own body, or the effect step lost one of them: the effect would outlive a load refused on a later device")


def test_witness_late_entry_streams_again():
    mutated = mutate_in(GS, LATE_SIG, "effects = std::move(decided);",
                        "ggml_sycl::get_layer_stream_manager(ctx->device).build_layer_map(nullptr, 0); "
                        "effects = std::move(decided);")
    assert not claim_entries_run_no_stage_effect(mutated)


def test_witness_effects_before_the_refusal():
    mutated = mutate_in(GS, LATE_SIG, "if (!compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct)) {",
                        "effects = std::move(decided); "
                        "if (!compute_and_store_plan_for_inventory(ctx, vram_budget, budget_pct)) {")
    assert not claim_entries_run_no_stage_effect(mutated)


def test_streamed_inventory_is_a_stage_local():
    assert claim_streamed_inventory_is_a_stage_local(GS), (
        "the streamed inventory's copy is not the effect record's own, or a record lives outside a function")


def test_witness_record_at_file_scope():
    assert not claim_streamed_inventory_is_a_stage_local(
        GS + "\nstatic ggml_sycl_inventory_effects g_sycl_inventory_effects;\n")


def test_stage_decides_every_refusal_before_any_effect():
    assert claim_stage_decides_every_refusal_before_any_effect(GS), (
        "the stage changes a device before every refusal of the pass is decided (an effect in the decide loop, a "
        "refusal after the first effect, or the per-device test failure outside the decide loop)")


def test_witness_configure_inside_the_decide_loop():
    mutated = mutate_in(GS, STAGE_SIG, "late_decisions.push_back(std::move(decision));",
                        "ggml_sycl::ggml_sycl_configure_host_zones_for_plan(nullptr, decision.snapshot->plan); "
                        "late_decisions.push_back(std::move(decision));")
    assert not claim_stage_decides_every_refusal_before_any_effect(mutated)


def test_witness_refusal_after_the_effects():
    mutated = mutate_in(GS, STAGE_SIG, LEASE_DROP,
                        "if (late_decisions.empty()) { return GGML_SYCL_LIFECYCLE_PLAN_REJECTED; } " + LEASE_DROP)
    assert not claim_stage_decides_every_refusal_before_any_effect(mutated)


def test_witness_test_failure_after_the_effects():
    mutated = mutate_in(GS, STAGE_SIG, TEST_FAILURE, "")
    mutated = mutate_in(mutated, STAGE_SIG, "rollback.released = true;", TEST_FAILURE + " rollback.released = true;")
    assert not claim_stage_decides_every_refusal_before_any_effect(mutated)


def test_stage_takes_no_inventory_lock():
    assert claim_stage_takes_no_inventory_lock(GS), (
        "the stage takes g_tensor_inventory_mutex itself, or its early effect does not go through the helper that "
        "takes it: provisioning must run outside the inventory lock, which the effect helpers hold only for their work")


def test_witness_stage_locks_the_early_effect_inline():
    mutated = mutate_in(GS, STAGE_SIG, "ggml_sycl_run_early_inventory_effect(device);",
                        "std::lock_guard lock(g_tensor_inventory_mutex); ggml_sycl_release_stale_kv_zone_ring(device);")
    assert not claim_stage_takes_no_inventory_lock(mutated)


def test_effects_check_the_last_snapshot():
    assert claim_effects_check_the_last_snapshot(GS), (
        "a device's effects do not run from its own snapshot under the check that the owner is active and the "
        "candidate is the last device's snapshot, before and after the effects")


def test_witness_effects_check_their_own_snapshot():
    mutated = mutate_in(GS, STAGE_SIG, "candidate_is_current(staged)", "candidate_is_current(decision.snapshot)")
    assert not claim_effects_check_the_last_snapshot(mutated)


def test_witness_effects_configure_from_the_last_snapshot():
    mutated = mutate_in(GS, STAGE_SIG, "ggml_sycl_configure_host_zones_for_plan(cache, decision.snapshot->plan);",
                        "ggml_sycl_configure_host_zones_for_plan(cache, staged->plan);")
    assert not claim_effects_check_the_last_snapshot(mutated)


def test_leases_outlive_the_effects():
    assert claim_leases_outlive_the_effects(GS), (
        "a replaced candidate's lease drops before every device's effects have run, or before the rollback guard's "
        "abort on a failed stage")


def test_witness_lease_dropped_in_the_decide_loop():
    mutated = mutate_in(GS, STAGE_SIG, "displaced_leases.push_back(std::move(replaced));",
                        "lazy_plan_slot_scope::drop_displaced(replaced);")
    assert not claim_leases_outlive_the_effects(mutated)


def test_witness_lease_list_after_the_rollback_guard():
    c = code(GS)
    mutated = c.replace(LEASE_LIST + " ", "", 1).replace("host_budget_pass_scope host_budget; const int n_devices",
                                                          LEASE_LIST + " host_budget_pass_scope host_budget; "
                                                          "const int n_devices", 1)
    assert mutated != c
    assert not claim_leases_outlive_the_effects(mutated)


def test_ring_release_lines_warn():
    assert claim_ring_release_lines_warn(GS), "a ring release or keep line is not at WARN"


def test_witness_release_line_at_info():
    mutated = mutate_in(GS, STALE_SIG, "GGML_LOG_WARN(", "GGML_LOG_INFO(")
    assert not claim_ring_release_lines_warn(mutated)


def test_refused_teardown_reserves_nothing():
    assert claim_refused_teardown_reserves_nothing(UC), (
        "a rebuild whose arena teardown refused can still reserve an arena, or the refusal lost its own reason")


def test_witness_reserve_after_a_refused_teardown():
    mutated = mutate_in(UC, UC_ENSURE_SIG, "return refuse_teardown(cause, arena_zones);",
                        "(void) refuse_teardown(cause, arena_zones);")
    assert not claim_refused_teardown_reserves_nothing(mutated)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
