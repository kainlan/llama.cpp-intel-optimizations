"""Source contract for the release of the last model on the SYCL backend.

When the last model of the process is released, nothing sized for a model may remain: no zone holds a live byte, no
chunk lease is outstanding, and the terms a load published for each device are back to the values a device no load has
planned on reads. The next load then plans and builds the arena as the first load of a process does. A load that ends
refused gets the same release and reserves nothing; the next load's start reserves the early arena again.

These checks pin the code that does it:

The model release hook (ggml_sycl_release_model_slot_resources) and a failed load's abort
(ggml_sycl_abort_owner_effects_noexcept) each end in one call of the one step, ggml_sycl_release_last_model_effects:
the hook after its slot release and its terms' drop, the abort after the published terms' restore and the drop, outside
every try stage. The step is noexcept, skips static destruction, asks each existing cache once (GLOBAL mode shares one
cache between devices), does nothing more when a model is live, and puts a device's terms back only when its cache
emptied, through the one list (load_published_terms{}.restore) under the inventory lock, never through setters of its
own.

The cache member it calls, unified_cache::release_for_no_live_model, is noexcept and catches. In order: it returns when
a model is live; waits once for one barrier over the cache's queues; releases the PP MoE oneDNN ring, the oneDNN
scratchpad pair and the reorder temp buffer; drains the queued frees through the existing drain, once; reads the
arena's holders; unbinds the compute arena; tears the arena down; and only then marks the cache released. Each release
that refuses stops it with a line that names the refusal. A held arena is an ERROR with the holders named for a model
release, a WARN at a failed load's abort (the failed model's buffers still hold it then); the release is INFO. The pair
release refuses while a reservation holds the pair and frees outside the pair's lock. None of the new code aborts or
asserts, and none of the release path reserves: no ensure, reserve or allocation.

The load's start (ggml_backend_sycl_model_load_begin) reserves a released cache's early arena after the caches exist
and before the candidate is bound, and a reserve it cannot make refuses the load with PLAN_REJECTED after an ERROR line
that names it. The released flag clears only after the arena is back.

ji71: g_tiered_headroom_reserve, which nothing read, is gone with its INFO line and its static-storage census row. The
decide pass's last-resort ring release stays, and its comment says why: liveness is process-wide, so a model live on
another device keeps this device's ring.

Host-only text assertions, no SYCL device. llama_test_pytest hands this file to pytest.main(), so the checks live inside
test_*() functions. Checks read COMMENT-STRIPPED, whitespace-normalised text, except the comment check, which reads the
raw text. Each check has a mutation witness that applies the regression it guards and must make the check fail.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GS_RAW = (ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp").read_text()
UC_RAW = (ROOT / "ggml/src/ggml-sycl/unified-cache.cpp").read_text()
UH_RAW = (ROOT / "ggml/src/ggml-sycl/unified-cache.hpp").read_text()
CSV_RAW = (ROOT / "docs/backend/sycl-static-storage-inventory.csv").read_text()

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


GS = code(GS_RAW)
UC = code(UC_RAW)
UH = code(UH_RAW)


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


def brace_depth(text: str, end: int) -> int:
    """How many braces are open at `end` in `text`, skipping string and char literals."""
    depth, i = 0, 0
    while i < end:
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
        i += 1
    return depth


def mutate_in(src: str, sig: str, old: str, new: str) -> str:
    """`src` (already code()) with the first `old` inside the function `sig` replaced by `new`."""
    body = fn_body(src, sig)
    assert body and old in body, f"witness anchor {old!r} is not in {sig!r}"
    s = src.find(body, src.find(sig))
    return src[:s] + body.replace(old, new, 1) + src[s + len(body) :]


STEP_SIG = ("static void ggml_sycl_release_last_model_effects(const char * site, "
            "ggml_sycl::unified_cache_release_cause cause) noexcept")
HOOK_SIG = "static void ggml_sycl_release_model_slot_resources(ggml_sycl::lifecycle::ModelToken owner)"
ABORT_SIG = ('static bool ggml_sycl_abort_owner_effects_noexcept(ggml_sycl::lifecycle::ModelToken owner, '
             'const char * site = "(unknown)") noexcept')
BEGIN_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_model_load_begin(ggml_sycl_load_txn * txn)"
START_SIG = "static bool ggml_sycl_load_start_reserve_released_arenas()"
LAST_RESORT_SIG = "static bool ggml_sycl_ensure_load_arena_zones(int device, ggml_sycl::zone_arena_refusal & refusal)"

MEMBER_SIG = ("unified_cache_last_release unified_cache::release_for_no_live_model(const char * site, "
              "unified_cache_release_cause cause) noexcept")
MEMBER_DECL = ("unified_cache_last_release release_for_no_live_model(const char * site, "
               "unified_cache_release_cause cause) noexcept;")
PAIR_SIG = "bool unified_cache::release_onednn_scratch_pair()"
REORDER_SIG = "void unified_cache::release_reorder_temp()"
RESERVE_SIG = "bool unified_cache::reserve_released_arena(zone_arena_refusal * refusal)"
ENSURE_SIG = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
              "const zone_arena_ensure_options & options)")
TERMS_RESTORE_SIG = "void load_published_terms::restore(int device, uint32_t groups) const noexcept"

HOOK_CALL = ('ggml_sycl_release_last_model_effects("model-release", '
             'ggml_sycl::unified_cache_release_cause::MODEL_RELEASED);')
ABORT_CALL = "ggml_sycl_release_last_model_effects(site, ggml_sycl::unified_cache_release_cause::LOAD_ABORTED);"
STEP_NAME = "ggml_sycl_release_last_model_effects("
DROP = "(void) ggml_sycl::unified_cache_drop_planned_load_terms(owner.owner);"
ABORT_RESTORE = "ggml_sycl_load_published_terms_restore(owner.load.value);"
HOOK_WARN = 'ggml_sycl_diag_emit_warn("[SYCL] model slot %u released: '

EMPTIED = "ggml_sycl::unified_cache_last_release::EMPTIED"
MODEL_LIVE = "ggml_sycl::unified_cache_last_release::MODEL_LIVE"
TERMS_RESET = "ggml_sycl::load_published_terms{}.restore(d);"
INVENTORY_LOCK = "std::lock_guard inventory(g_tensor_inventory_mutex);"
DEDUPE = "released[d] = first == d ? caches[d]->release_for_no_live_model(site, cause) : released[first];"
SHUTDOWN_SKIP = "if (ggml_sycl::ggml_sycl_is_shutting_down()) { return; }"

MEMBER_ORDER = (
    "if (live_model_mask() != 0) { return unified_cache_last_release::MODEL_LIVE; }",
    "submit_barrier_all().wait();",
    "if (!release_pp_moe_onednn_scratch_ring()) {",
    "if (!release_onednn_scratch_pair()) {",
    "release_reorder_temp();",
    "process_deferred_frees_public();",
    "zone_arena_holders holders = arena_read_holders(live_zone_used);",
    "if (!zone_arena_rebuild_allowed(holders)) {",
    "compute_arena_unbind();",
    "if (!arena_destroy()) {",
    "arena_released_.store(true, std::memory_order_release);",
)

# What the release path must never do: it releases, and a reserve, an allocation or an ensure (which may reserve) would
# turn the release of a refused load into the allocation the design forbids there.
RESERVERS = (
    "arena_reserve(",
    "ensure_planned_arena_zones",
    "reserve_released_arena",
    "ggml_sycl_load_start_reserve_released_arenas",
    "zone_alloc(",
    "unified_alloc",
    "unified_allocate",
    "reserve_onednn_scratch",
    "reserve_reorder_temp",
)
# What the cache member must never do over what still holds the arena: it names it and keeps it.
FORCERS = ("zone_reclaim(", "zone_settle(", "zone_reset", "arena_reset(", "evict", "reap", "arena_abandon")
ABORTS = ("GGML_ABORT", "GGML_ASSERT", "abort(", "std::terminate", "throw")


def check_hook(gs: str) -> list:
    body = fn_body(gs, HOOK_SIG)
    if not body:
        return ["the model release hook is missing"]
    out = []
    if body.count(STEP_NAME) != 1 or HOOK_CALL not in body:
        out.append("the hook does not call the step exactly once with MODEL_RELEASED")
    if not ordered(body, DROP, HOOK_WARN, HOOK_CALL):
        out.append("the hook's step call is not after its terms' drop and its summary line")
    if not body.endswith(HOOK_CALL + " }"):
        out.append("the hook's step call is not its last statement")
    for direct in ("release_for_no_live_model", "arena_destroy", "release_pp_moe_onednn_scratch_ring",
                   "load_published_terms{}"):
        if direct in body:
            out.append(f"the hook does part of the step itself: {direct}")
    return out


def check_abort(gs: str) -> list:
    body = fn_body(gs, ABORT_SIG)
    if not body:
        return ["the owner-effects abort is missing"]
    out = []
    if body.count(STEP_NAME) != 1 or ABORT_CALL not in body:
        out.append("the abort does not call the step exactly once with LOAD_ABORTED")
    if not ordered(body, ABORT_RESTORE, DROP, ABORT_CALL):
        out.append("the abort's step call is not after the published terms' restore and the drop")
    if not body.endswith(ABORT_CALL + " return clean; }"):
        out.append("the abort's step call is not its last statement before the return")
    k = body.find(ABORT_CALL)
    if k != -1 and brace_depth(body, k) != 1:
        out.append("the abort's step call is inside a block (a try stage)")
    for reserve in RESERVERS:
        if reserve in body:
            out.append(f"the abort reserves: {reserve}")
    return out


def check_step(gs: str) -> list:
    body = fn_body(gs, STEP_SIG)
    if not body:
        return ["the step is missing or is not noexcept"]
    out = []
    if not body.startswith("{ try {") or "} catch (...) { GGML_LOG_ERROR(" not in body:
        out.append("the step does not catch every exception with an ERROR line")
    if SHUTDOWN_SKIP not in body:
        out.append("the step does not skip static destruction")
    if DEDUPE not in body or "if (caches[e] == caches[d]) {" not in body:
        out.append("the step asks a cache it already asked")
    if not ordered(body, DEDUPE, f"if (released[d] == {MODEL_LIVE}) {{ return; }}", f"if (released[d] == {EMPTIED}) {{",
                   INVENTORY_LOCK, TERMS_RESET):
        out.append("the step resets a device's terms while a model is live or when its cache did not empty")
    if body.count(TERMS_RESET) != 1:
        out.append("the step does not reset the terms through the one list exactly once")
    for setter in ("unified_cache_set_planned_", "unified_cache_store_planned_", "onednn_zone_plan_store",
                   "onednn_zone_plan_put_back"):
        if setter in body:
            out.append(f"the step sets a term itself: {setter}")
    if f"if (all_emptied) {{ g_moe_multi_gpu_active.store(false" not in body:
        out.append("the step does not put the two-card flag back once every cache emptied")
    for reserve in RESERVERS:
        if reserve in body:
            out.append(f"the step reserves: {reserve}")
    return out


def check_member(uc: str, uh: str) -> list:
    body = fn_body(uc, MEMBER_SIG)
    if not body or MEMBER_DECL not in uh:
        return ["the cache's last release member is missing or is not noexcept"]
    out = []
    if not ordered(body, *MEMBER_ORDER):
        out.append("the member's steps are out of order")
    if body.count("submit_barrier_all()") != 1 or body.count(".wait()") != 1:
        out.append("the member does not wait for exactly one barrier")
    if body.count("process_deferred_frees_public();") != 1:
        out.append("the member does not drain exactly once")
    if "catch (...) { GGML_LOG_ERROR(" not in body or not ordered(body, "} catch (...) {",
                                                                  "return unified_cache_last_release::FAILED; } }"):
        out.append("the member does not end an exception with an ERROR line and FAILED")
    if uc.count("arena_released_.store(true") != 1:
        out.append("the released flag is set somewhere other than after the teardown")
    for force in FORCERS:
        if force in body:
            out.append(f"the member forces what holds the arena: {force}")
    for reserve in RESERVERS:
        if reserve in body:
            out.append(f"the member reserves: {reserve}")
    return out


def _branch(body: str, head: str) -> str:
    k = body.find(head)
    return block_at(body, k + len(head) - 1) if k != -1 else ""


def check_refusals_named(uc: str) -> list:
    body = fn_body(uc, MEMBER_SIG)
    if not body:
        return ["the cache's last release member is missing"]
    out = []
    for head, result in (("if (!release_pp_moe_onednn_scratch_ring()) {", "RING_CLAIMED"),
                         ("if (!release_onednn_scratch_pair()) {", "PAIR_HELD"),
                         ("if (!arena_destroy()) {", "TEARDOWN_REFUSED")):
        branch = _branch(body, head)
        if "GGML_LOG_ERROR(" not in branch or f"return unified_cache_last_release::{result};" not in branch:
            out.append(f"the refusal {result} is not an ERROR line that stops the release")
    if "arena_teardown_cause_name(teardown)" not in _branch(body, "if (!arena_destroy()) {"):
        out.append("a refused teardown does not name its cause")
    held = _branch(body, "if (!zone_arena_rebuild_allowed(holders)) {")
    aborted = _branch(held, "if (cause == unified_cache_release_cause::LOAD_ABORTED) {")
    rest = held.replace(aborted, "", 1) if aborted else held
    if "GGML_LOG_WARN(" not in aborted or "GGML_LOG_ERROR(" in aborted or "arena_report_holders" in aborted:
        out.append("a held arena at a failed load's abort is not one WARN line")
    if "return unified_cache_last_release::ARENA_HELD;" not in aborted:
        out.append("a held arena at a failed load's abort does not stop the release")
    if not ordered(rest, "GGML_LOG_ERROR(", 'arena_report_holders(dev_id, live_zone_used, holders, "last-release");',
                   "return unified_cache_last_release::ARENA_HELD;"):
        out.append("a held arena at a model release is not an ERROR line with its holders named")
    if not ordered(body, "arena_released_.store(true, std::memory_order_release);", "GGML_LOG_INFO(",
                   "return unified_cache_last_release::EMPTIED;"):
        out.append("the release is not an INFO line")
    return out


def check_pair(uc: str) -> list:
    body = fn_body(uc, PAIR_SIG)
    if not body:
        return ["the pair release is missing"]
    out = []
    lock = "std::lock_guard<std::mutex> lock(onednn_scratch_mutex_);"
    k = body.find(lock)
    locked = block_at(body, body.rfind("{", 0, k)) if k != -1 else ""
    if not locked or not locked.startswith("{ " + lock + " if (onednn_scratch_refcount_ != 0) { return false; }"):
        out.append("the pair release does not refuse on a held reservation before it takes the pair")
    for moved in ("weights_owner = std::move(onednn_weights_scratch_owner_);",
                  "activations_owner = std::move(onednn_activations_scratch_owner_);",
                  "onednn_weights_scratch_ = nullptr;", "onednn_activations_scratch_ = nullptr;"):
        if moved not in locked:
            out.append(f"the pair is not taken out under its lock: {moved}")
    for free in ("zone_free(", "release(", "saturating_sub_used(", "owner = {}"):
        if free in locked.replace("_owner_ = {}", ""):
            out.append(f"the pair release frees under its lock: {free}")
    after = body[body.find(locked) + len(locked):] if locked else ""
    if not ordered(after, "zone_free(vram_zone_id::ONEDNN, ptr);", 'release(weights, weights_owner, "weights");',
                   'release(activations, activations_owner, "activations");', "saturating_sub_used(direct_bytes);"):
        out.append("the pair release does not free both halves after its lock")
    for reserve in RESERVERS:
        if reserve in body:
            out.append(f"the pair release reserves: {reserve}")
    return out


def check_no_abort(gs: str, uc: str) -> list:
    out = []
    for text, sig in ((gs, STEP_SIG), (gs, START_SIG), (uc, MEMBER_SIG), (uc, PAIR_SIG), (uc, REORDER_SIG),
                      (uc, RESERVE_SIG)):
        body = fn_body(text, sig)
        if not body:
            out.append(f"missing: {sig}")
            continue
        for a in ABORTS:
            if a in body:
                out.append(f"{sig} aborts: {a}")
    return out


def check_load_start(gs: str, uc: str) -> list:
    out = []
    begin = fn_body(gs, BEGIN_SIG)
    if not ordered(begin, "ggml_sycl_model_loading_effects(true, true);",
                   "if (ggml_sycl_load_start_reserve_released_arenas()) {", "registry->bind_candidate(result.txn);",
                   "return GGML_SYCL_LIFECYCLE_OK;"):
        out.append("the load's start does not reserve the released arenas after the caches exist and before binding")
    # One roll-back path below the try, for a refused reserve and an exception alike, so the abort keeps its one
    # literal site tag: a refused reserve leaves the try with PLAN_REJECTED, an exception with EFFECT_FAILED.
    if "ggml_sycl_lifecycle_result rollback_code = GGML_SYCL_LIFECYCLE_PLAN_REJECTED; try {" not in begin:
        out.append("a load whose start cannot reserve is not refused with PLAN_REJECTED")
    if "} catch (...) { rollback_code = GGML_SYCL_LIFECYCLE_EFFECT_FAILED; }" not in begin:
        out.append("the load's start no longer rolls an exception back as EFFECT_FAILED")
    if not ordered(begin, "} catch (...) { rollback_code = GGML_SYCL_LIFECYCLE_EFFECT_FAILED; }",
                   'ggml_sycl_abort_owner_effects_noexcept(ticket.token, "load_begin/rollback");',
                   "return rollback_code; }") or not begin.endswith("return rollback_code; }"):
        out.append("the load's start roll-back does not abort the owner effects and return its code")
    # Two writes of the code: its start value and the catch's.
    if begin.count("ggml_sycl_abort_owner_effects_noexcept(") != 1 or begin.count("rollback_code =") != 2 \
            or begin.count("return rollback_code;") != 1:
        out.append("the load's start has more than one roll-back path")
    start = fn_body(gs, START_SIG)
    if not ordered(start, "if (!cache->reserve_released_arena(&refusal)) {", "GGML_LOG_ERROR(",
                   '"[SYCL] model load refused at its start: %s\\n"', "zone_arena_refusal_text(refusal)",
                   "return false;"):
        out.append("a start reserve that fails is not refused by name")
    reserve = fn_body(uc, RESERVE_SIG)
    if not ordered(reserve, "if (!arena_released_.load(std::memory_order_acquire)) { return true; }",
                   "if (!arena_active() && !ensure_planned_arena_zones(refusal)) { return false; }",
                   "arena_released_.store(false, std::memory_order_release);"):
        out.append("the released flag does not clear only after the arena is back")
    if gs.count("ggml_sycl_load_start_reserve_released_arenas()") != 2:
        out.append("the start reserve is called somewhere other than the load's start")
    return out


def check_zone_plan_store_callers(uc: str, gs: str) -> list:
    out = []
    allowed = (fn_body(uc, ENSURE_SIG), fn_body(uc, TERMS_RESTORE_SIG))
    calls = uc.count("onednn_zone_plan_store(") - uc.count("void unified_cache::onednn_zone_plan_store(")
    inside = sum(body.count("onednn_zone_plan_store(") for body in allowed)
    if calls != inside or inside == 0:
        out.append("onednn_zone_plan_store is called outside the ensure and the published terms' restore")
    if "onednn_zone_plan_store(" in gs:
        out.append("ggml-sycl.cpp stores a zone plan itself")
    return out


def check_tiered_headroom_gone(gs_raw: str, csv_raw: str) -> list:
    out = []
    for gone in ("g_tiered_headroom_reserve", "Tiered headroom reserve"):
        if gone in gs_raw:
            out.append(f"still in ggml-sycl.cpp: {gone}")
    if "g_tiered_headroom_reserve" in csv_raw:
        out.append("the static-storage census still lists g_tiered_headroom_reserve")
    return out


def check_last_resort_comment(gs_raw: str) -> list:
    k = gs_raw.find(LAST_RESORT_SIG)
    body = block_at(gs_raw, k + len(LAST_RESORT_SIG)) if k != -1 else ""
    release = body.find("cache->release_pp_moe_onednn_scratch_ring()")
    if release == -1:
        return ["the decide pass's last-resort ring release is gone"]
    comment = body[:release]
    out = []
    for words in ("process-wide", "another device"):
        if words not in comment:
            out.append(f"the last resort's comment does not say why it is reachable: {words!r}")
    return out


# ---- the checks on the tree ----


def test_hook_ends_in_the_step():
    assert check_hook(GS) == []


def test_abort_ends_in_the_step_outside_every_try():
    assert check_abort(GS) == []


def test_step_resets_terms_through_one_list():
    assert check_step(GS) == []


def test_member_releases_in_order_and_reserves_nothing():
    assert check_member(UC, UH) == []


def test_member_names_what_holds_on():
    assert check_refusals_named(UC) == []


def test_pair_release_frees_outside_its_lock():
    assert check_pair(UC) == []


def test_new_code_adds_no_abort():
    assert check_no_abort(GS, UC) == []


def test_load_start_reserves_and_refuses_by_name():
    assert check_load_start(GS, UC) == []


def test_zone_plan_store_has_two_callers():
    assert check_zone_plan_store_callers(UC, GS) == []


def test_tiered_headroom_reserve_is_gone():
    assert check_tiered_headroom_gone(GS_RAW, CSV_RAW) == []


def test_last_resort_comment_says_why():
    assert check_last_resort_comment(GS_RAW) == []


# ---- mutation witnesses: each applies the regression its check guards ----


def test_witness_hook_without_the_step():
    assert check_hook(mutate_in(GS, HOOK_SIG, HOOK_CALL, ""))


def test_witness_hook_step_before_the_drop():
    m = mutate_in(GS, HOOK_SIG, DROP, HOOK_CALL + " " + DROP)
    assert check_hook(mutate_in(m, HOOK_SIG, HOOK_CALL + " }", "}"))


def test_witness_abort_step_inside_a_try():
    assert check_abort(mutate_in(GS, ABORT_SIG, ABORT_CALL, "try { " + ABORT_CALL + " } catch (...) { }"))


def test_witness_abort_step_before_the_restore():
    m = mutate_in(GS, ABORT_SIG, ABORT_CALL, "")
    assert check_abort(mutate_in(m, ABORT_SIG, ABORT_RESTORE, ABORT_CALL + " " + ABORT_RESTORE))


def test_witness_abort_reserves():
    reserve = " ggml_sycl_load_start_reserve_released_arenas();"
    assert check_abort(mutate_in(GS, ABORT_SIG, ABORT_CALL, ABORT_CALL + reserve))


def test_witness_step_resets_terms_by_setter():
    setter = "unified_cache_set_planned_pp_pipeline_scratch_bytes(d, 0);"
    assert check_step(mutate_in(GS, STEP_SIG, TERMS_RESET, setter))


def test_witness_step_resets_terms_of_a_cache_that_kept_its_arena():
    assert check_step(mutate_in(GS, STEP_SIG, f"if (released[d] == {EMPTIED}) {{", "if (true) {"))


def test_witness_step_resets_terms_while_a_model_is_live():
    assert check_step(mutate_in(GS, STEP_SIG, f"if (released[d] == {MODEL_LIVE}) {{ return; }}", ""))


def test_witness_step_asks_a_shared_cache_twice():
    every = "released[d] = caches[d]->release_for_no_live_model(site, cause);"
    assert check_step(mutate_in(GS, STEP_SIG, DEDUPE, every))


def test_witness_step_runs_at_static_destruction():
    assert check_step(mutate_in(GS, STEP_SIG, SHUTDOWN_SKIP, ""))


def test_witness_member_releases_before_the_barrier():
    m = mutate_in(UC, MEMBER_SIG, "submit_barrier_all().wait();", "")
    assert check_member(mutate_in(m, MEMBER_SIG, "release_reorder_temp();",
                                  "release_reorder_temp(); submit_barrier_all().wait();"), UH)


def test_witness_member_drains_after_the_teardown():
    m = mutate_in(UC, MEMBER_SIG, "process_deferred_frees_public();", "")
    assert check_member(mutate_in(m, MEMBER_SIG, "arena_released_.store(true, std::memory_order_release);",
                                  "process_deferred_frees_public(); "
                                  "arena_released_.store(true, std::memory_order_release);"), UH)


def test_witness_member_waits_twice():
    assert check_member(mutate_in(UC, MEMBER_SIG, "release_reorder_temp();",
                                  "release_reorder_temp(); queue_.wait();"), UH)


def test_witness_member_reserves_after_the_teardown():
    assert check_member(mutate_in(UC, MEMBER_SIG, "arena_released_.store(true, std::memory_order_release);",
                                  "arena_released_.store(true, std::memory_order_release); "
                                  "(void) ensure_planned_arena_zones();"), UH)


def test_witness_member_reclaims_what_holds_the_arena():
    assert check_member(mutate_in(UC, MEMBER_SIG, "compute_arena_unbind();",
                                  "zone_reclaim(vram_zone_id::RUNTIME); compute_arena_unbind();"), UH)


def test_witness_member_not_noexcept():
    assert check_member(UC, UH.replace(MEMBER_DECL, MEMBER_DECL.replace(" noexcept;", ";"), 1))


def test_witness_ring_refusal_goes_on():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, "return unified_cache_last_release::RING_CLAIMED;", ""))


def test_witness_held_arena_not_named():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG,
                                          'arena_report_holders(dev_id, live_zone_used, holders, "last-release");',
                                          ""))


def test_witness_held_arena_at_abort_is_an_error():
    m = mutate_in(UC, MEMBER_SIG, "if (cause == unified_cache_release_cause::LOAD_ABORTED) { GGML_LOG_WARN(",
                  "if (cause == unified_cache_release_cause::LOAD_ABORTED) { GGML_LOG_ERROR(")
    assert check_refusals_named(m)


def test_witness_teardown_refusal_unnamed():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, "arena_teardown_cause_name(teardown)", '"refused"'))


def test_witness_pair_frees_under_its_lock():
    assert check_pair(mutate_in(UC, PAIR_SIG, "onednn_weights_scratch_ = nullptr;",
                                "zone_free(vram_zone_id::ONEDNN, onednn_weights_scratch_); "
                                "onednn_weights_scratch_ = nullptr;"))


def test_witness_pair_ignores_a_held_reservation():
    assert check_pair(mutate_in(UC, PAIR_SIG, "if (onednn_scratch_refcount_ != 0) { return false; }", ""))


def test_witness_new_code_asserts():
    assert check_no_abort(GS, mutate_in(UC, PAIR_SIG, "return true;", "GGML_ASSERT(direct_bytes == 0); return true;"))


def test_witness_load_binds_before_the_reserve():
    m = mutate_in(GS, BEGIN_SIG, "if (ggml_sycl_load_start_reserve_released_arenas()) { registry->bind_candidate("
                                 "result.txn);",
                  "registry->bind_candidate(result.txn); if (ggml_sycl_load_start_reserve_released_arenas()) {")
    assert check_load_start(m, UC)


def test_witness_start_refusal_is_retried():
    assert check_load_start(mutate_in(GS, BEGIN_SIG, "rollback_code = GGML_SYCL_LIFECYCLE_PLAN_REJECTED; try {",
                                      "rollback_code = GGML_SYCL_LIFECYCLE_EFFECT_FAILED; try {"), UC)


def test_witness_start_refusal_skips_the_abort():
    assert check_load_start(mutate_in(GS, BEGIN_SIG, "return GGML_SYCL_LIFECYCLE_OK; }",
                                      "return GGML_SYCL_LIFECYCLE_OK; } txn->id = 0; return rollback_code;"), UC)


def test_witness_flag_clears_before_the_reserve():
    m = mutate_in(UC, RESERVE_SIG, "arena_released_.store(false, std::memory_order_release); return true;",
                  "return true;")
    assert check_load_start(GS, mutate_in(m, RESERVE_SIG, "if (!arena_active()",
                                          "arena_released_.store(false, std::memory_order_release); "
                                          "if (!arena_active()"))


def test_witness_zone_plan_stored_by_another_caller():
    assert check_zone_plan_store_callers(mutate_in(UC, RESERVE_SIG, "return true; }",
                                                   "onednn_zone_plan_store(zone_onednn_plan{}); return true; }"), GS)


def test_witness_tiered_headroom_reserve_back():
    assert check_tiered_headroom_gone(
        GS_RAW + "\nstatic std::array<size_t, GGML_SYCL_MAX_DEVICES> g_tiered_headroom_reserve = {};\n", CSV_RAW)


def test_witness_last_resort_comment_without_why():
    k = GS_RAW.find(LAST_RESORT_SIG)
    assert k != -1
    assert check_last_resort_comment(GS_RAW[:k] + GS_RAW[k:].replace("another device", "the device", 1))
