"""Source contract for the release of the last model on the SYCL backend.

When the last model of the process is released, nothing sized for a model may remain: no zone holds a live byte, no
chunk lease is outstanding, and the terms a load published for each device are back to the values a device no load has
planned on reads. The next load then plans and builds the arena as the first load of a process does. A load that ends
refused gets the same release and reserves nothing; the next load's start reserves the early arena again.

These checks pin the code that does it:

The model release hook (ggml_sycl_release_model_slot_resources) and a failed load's abort
(ggml_sycl_abort_owner_effects_noexcept) each end in one call of the one step, ggml_sycl_release_last_model_effects:
the hook after its slot release and its terms' drop, the abort after the published terms' restore and the drop, outside
every try stage, both with the scope EVERY_CACHE. The hook retires the model's MoE mmid workspaces just before its step,
once: the registry's reference to a workspace pool is what keeps its blob in the arena once no lease holds it, so a
retire after the step left that blob to be read as a holder of an arena no model uses. The hook runs only once the
teardown's fail-closed gates (the plan's republication and the MoE discovery release) have passed, as the last step of
the owner effects, so a teardown they refuse retires nothing; the unload hook and its failure branches retire nothing of
their own, and the other retire sites (a load's end, a runtime context transaction and a live bind) keep theirs. The
step is noexcept, skips static destruction, asks each existing
cache once (GLOBAL mode shares one cache between devices), does nothing more when a model is live, and puts a device's
terms back only when its cache was asked and emptied, through the one list (load_published_terms{}.restore) under the
inventory lock, never through setters of its own. With the scope DEFERRED_CACHES it asks only a cache whose release a
failed load's buffers deferred; a cache it leaves out keeps what it holds and its terms, and counts as emptied for the
two-card flag only when it holds no arena. A step that asked no cache leaves that flag as it is: it records that it
asked one only once the scope let it, before it asks.

The cache member it calls, unified_cache::release_for_no_live_model, is noexcept and catches. In order: it returns when
a model is live; waits once for one barrier over the cache's queues; releases the PP MoE oneDNN ring and the oneDNN
scratchpad pair; drains the queued frees through the existing drain, once; reads the
arena's holders; unbinds the compute arena; tears the arena down; and only then marks the cache released. Each release
that refuses stops it with a line that names the refusal. A held arena is an ERROR with the holders named for a model
release; one WARN line at a failed load's abort (the failed model's buffers still hold it then); and at a load's start,
which has its own cause (LOAD_START), a WARN that says a holder still there once the failed model is freed is a leak,
with the holders named at WARN too, through the holders report's level, which logs its holders line at the level its
caller passes. The release is INFO. At the abort the held arena also marks the release deferred
(last_release_deferred_); the start's cause never sets that mark, its held arena keeps it, and only a release that
empties the cache clears it. The pair release refuses while a reservation holds the pair and frees outside the pair's
lock. None of the new code aborts or asserts, and none of the release path reserves: no ensure, reserve or allocation.

The load's start (ggml_backend_sycl_model_load_begin), inside its transaction and once the caches exist, first runs
the step again for the deferred caches (LOAD_START, DEFERRED_CACHES), then reserves a released cache's early arena
before the candidate is bound, and a reserve it cannot make refuses the load with PLAN_REJECTED after an ERROR line
that names it. The released flag clears only after the arena is back.

g_tiered_headroom_reserve, which nothing read, is gone with its INFO line and its static-storage census row. The
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
    """The brace block that opens at the first '{' at or after `start`, skipping comments and string and char literals.

    Raw source reaches it too (a check that reads a comment), where an apostrophe in a comment is not a char literal.
    """
    o = text.find("{", start) if start >= 0 else -1
    if o == -1:
        return ""
    depth, i = 0, o
    while i < len(text):
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = len(text) if j == -1 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = len(text) if j == -1 else j + 2
            continue
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
            "ggml_sycl::unified_cache_release_cause cause, ggml_sycl_last_release_scope scope) noexcept")
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
RESERVE_SIG = "bool unified_cache::reserve_released_arena(zone_arena_refusal * refusal)"
ENSURE_SIG = ("bool unified_cache::ensure_planned_arena_zones(zone_arena_refusal * refusal, "
              "const zone_arena_ensure_options & options)")
TERMS_RESTORE_SIG = "void load_published_terms::restore(int device, uint32_t groups) const noexcept"

EVERY = "ggml_sycl_last_release_scope::EVERY_CACHE"
DEFERRED = "ggml_sycl_last_release_scope::DEFERRED_CACHES"
HOOK_CALL = ('ggml_sycl_release_last_model_effects("model-release", '
             f"ggml_sycl::unified_cache_release_cause::MODEL_RELEASED, {EVERY});")
ABORT_CALL = f"ggml_sycl_release_last_model_effects(site, ggml_sycl::unified_cache_release_cause::LOAD_ABORTED, {EVERY});"
START_CAUSE = "ggml_sycl::unified_cache_release_cause::LOAD_START"
START_CALL = f'ggml_sycl_release_last_model_effects("load-start", {START_CAUSE}, {DEFERRED});'
STEP_NAME = "ggml_sycl_release_last_model_effects("
DROP = "(void) ggml_sycl::unified_cache_drop_planned_load_terms(owner.owner);"
ABORT_RESTORE = "ggml_sycl_load_published_terms_restore(owner.load.value);"
HOOK_WARN = 'ggml_sycl_diag_emit_warn("[SYCL] model slot %u released: '

UNLOAD_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_model_unloaded_token(ggml_sycl_model_token token)"
# Declared before its definition, so it is found by name (fn_def), not by its first signature.
EFFECTS_NAME = "ggml_sycl_teardown_owner_effects"
RETIRE_NAME = "ggml_sycl::unified_cache_retire_moe_mmid_workspaces("
# The dying model's every plan identity (the default 0), named by its exact token. code() leaves one space after the
# parenthesis when the call breaks its line there.
HOOK_RETIRE = re.compile(r"\(void\) ggml_sycl::unified_cache_retire_moe_mmid_workspaces\( ?"
                         r"\{ owner\.model\.value, owner\.load\.value, owner\.owner\.generation \}\);")
EFFECTS_TAIL = "ggml_sycl_release_model_slot_resources(owner); return true; } catch (...) { return false; } }"
UNLOAD_ERASE = ("if (result == ggml_sycl::lifecycle::error::OK) { "
                "ggml_sycl::lifecycle_erase_placement_plan(owner.model.value, owner.load.value); }")
# The retire sites other than the hook's, by the function that holds them, with their counts: they keep what they do.
OTHER_RETIRES = {
    "ggml_backend_sycl_model_load_end": 2,
    "ggml_sycl_run_runtime_context_transaction": 2,
    "ggml_sycl_bind_materialize_live_mmid_workspaces": 1,
}

EMPTIED = "ggml_sycl::unified_cache_last_release::EMPTIED"
MODEL_LIVE = "ggml_sycl::unified_cache_last_release::MODEL_LIVE"
TERMS_RESET = "ggml_sycl::load_published_terms{}.restore(d);"
INVENTORY_LOCK = "std::lock_guard inventory(g_tensor_inventory_mutex);"
DEDUPE = "released[d] = first == d ? caches[d]->release_for_no_live_model(site, cause) : released[first];"
SHUTDOWN_SKIP = "if (ggml_sycl::ggml_sycl_is_shutting_down()) { return; }"
SCOPE_EVERY = f"const bool every = scope == {EVERY};"
ASK = "asked[d] = first == d ? (every || caches[d]->last_release_deferred()) : asked[first];"
LEFT_OUT = "if (!asked[d]) { all_emptied = all_emptied && !caches[d]->arena_active(); continue; }"
TERMS_RESET_IF = f"if (asked[d] && released[d] == {EMPTIED}) {{"
TWO_CARD = "if (all_emptied && (every || asked_any)) { g_moe_multi_gpu_active.store(false"
ASKED_ANY = "asked_any = true;"
DEFER_SET = "last_release_deferred_.store(true, std::memory_order_release);"
DEFER_CLEAR = "last_release_deferred_.store(false, std::memory_order_release);"
DEFER_MEMBER = "std::atomic<bool> last_release_deferred_{ false };"
DEFER_READ = "bool last_release_deferred() const { return last_release_deferred_.load(std::memory_order_acquire); }"
HELD_HEAD = "if (!zone_arena_rebuild_allowed(holders)) {"
ABORTED_HEAD = "if (cause == unified_cache_release_cause::LOAD_ABORTED) {"
STARTED_HEAD = "if (cause == unified_cache_release_cause::LOAD_START) {"
CAUSE_ENUM = "enum class unified_cache_release_cause { MODEL_RELEASED, LOAD_ABORTED, LOAD_START };"
ABORTED_TEXT = "(%s): the failed load's buffers still hold the arena \""
STARTED_TEXT = "(%s): a load is starting on an arena a failed load left \""
LEAK_TEXT = 'is a leak\\n"'
START_HOLDERS = "arena_report_holders(dev_id, live_zone_used, holders, site, GGML_LOG_LEVEL_WARN);"
RELEASE_HOLDERS = 'arena_report_holders(dev_id, live_zone_used, holders, "last-release", GGML_LOG_LEVEL_ERROR);'
REBUILD_HOLDERS = 'arena_report_holders(dev_id, live_zone_used, holders, "arena-rebuild", GGML_LOG_LEVEL_ERROR);'
REPORT_SIG = ("void unified_cache::arena_report_holders(int dev_id, const size_t * live_zone_used, "
              "zone_arena_holders & holders, const char * site, ggml_log_level level)")
HOLDERS_LINE = '"[VRAM-ARENA] live in the arena on device %d (MB): %s; "'
# The holders line goes out at the caller's level: ERROR for a refused rebuild and a model release, WARN at a load's
# start. code() leaves one space after the parenthesis when the call breaks its line there.
HOLDERS_AT_LEVEL = re.compile(r"ggml_log_internal\( ?level, " + re.escape(HOLDERS_LINE))

MEMBER_ORDER = (
    "if (live_model_mask() != 0) { return unified_cache_last_release::MODEL_LIVE; }",
    "submit_barrier_all().wait();",
    "if (!release_pp_moe_onednn_scratch_ring()) {",
    "if (!release_onednn_scratch_pair()) {",
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


def fn_def(text: str, name: str) -> str:
    """The body of the function defined as `name`, in code() text, wherever its signature breaks; "" when none is."""
    for m in re.finditer(r"(?<![\w:])(?!return\b)[\w:<>]+ " + re.escape(name) + r"\(", text):
        rest = text[m.end():]
        b, s = rest.find("{"), rest.find(";")
        if b != -1 and (s == -1 or b < s):
            return block_at(text, m.end())
    return ""


def check_retire_order(gs: str) -> list:
    hook = fn_body(gs, HOOK_SIG)
    effects = fn_def(gs, EFFECTS_NAME)
    unload = fn_body(gs, UNLOAD_SIG)
    if not hook or not effects or not unload:
        return ["the model release hook, the owner effects or the unload hook is missing"]
    out = []
    retire = HOOK_RETIRE.search(hook)
    if hook.count(RETIRE_NAME) != 1 or not retire:
        out.append("the hook does not retire the dying model's MoE mmid workspaces exactly once, by its token")
    elif not ordered(hook, DROP, HOOK_WARN, retire.group(0), HOOK_CALL):
        out.append("the hook's retire is not after its slot release and before its step")
    elif brace_depth(hook, hook.find(retire.group(0))) != 1:
        out.append("the hook's retire is inside a block")
    # The gates that keep a refused teardown's model alive come first; the hook is the last thing the effects do.
    if RETIRE_NAME in effects:
        out.append("the owner effects retire a workspace outside the hook")
    if not ordered(effects, "moe_discovery_release_retrying(", "return false;", "release_if_owner(dying);",
                   EFFECTS_TAIL) or not effects.endswith(EFFECTS_TAIL):
        out.append("the hook is not the owner effects' last step, after their fail-closed gates")
    if RETIRE_NAME in unload:
        out.append("the unload hook retires a workspace of its own (after the hook's step asked the arena)")
    if UNLOAD_ERASE not in unload:
        out.append("the unload hook does not erase the plan of a finalized teardown alone")
    for sig in (ABORT_SIG, BEGIN_SIG):
        if RETIRE_NAME in fn_body(gs, sig):
            out.append(f"a retire reached the release path of a load: {sig}")
    for name, n in OTHER_RETIRES.items():
        body = fn_def(gs, name)
        if not body or body.count(RETIRE_NAME) != n:
            out.append(f"{name} no longer retires as it did ({n})")
    if gs.count(RETIRE_NAME) != 1 + sum(OTHER_RETIRES.values()):
        out.append("a retire site was added or removed")
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
    if SCOPE_EVERY not in body or not ordered(body, "if (caches[e] == caches[d]) {", ASK, LEFT_OUT, DEDUPE):
        out.append("the step asks a cache its scope leaves out, or counts one it left out as emptied with an arena")
    # Without the record a start that asked a cache could never put the two-card flag back; set before the scope's
    # filter, it would put it back for a start that asked none.
    if body.count(ASKED_ANY) != 1 or not ordered(body, LEFT_OUT, ASKED_ANY, DEDUPE):
        out.append("the step does not record that it asked a cache, once, after the scope let it and before it asks")
    if not ordered(body, DEDUPE, f"if (released[d] == {MODEL_LIVE}) {{ return; }}", TERMS_RESET_IF, INVENTORY_LOCK,
                   TERMS_RESET):
        out.append("the step resets a device's terms while a model is live, or when its cache was not asked or did not "
                   "empty")
    if body.count(TERMS_RESET) != 1:
        out.append("the step does not reset the terms through the one list exactly once")
    for setter in ("unified_cache_set_planned_", "unified_cache_store_planned_", "onednn_zone_plan_store",
                   "onednn_zone_plan_put_back"):
        if setter in body:
            out.append(f"the step sets a term itself: {setter}")
    if TWO_CARD not in body:
        out.append("the step does not put the two-card flag back once every cache emptied, or does when it asked none")
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
    held = _branch(body, HELD_HEAD)
    aborted = _branch(held, ABORTED_HEAD)
    started = _branch(held, STARTED_HEAD)
    rest = held
    for branch in (aborted, started):
        rest = rest.replace(branch, "", 1) if branch else rest
    if "GGML_LOG_WARN(" not in aborted or "GGML_LOG_ERROR(" in aborted or "arena_report_holders" in aborted \
            or ABORTED_TEXT not in aborted:
        out.append("a held arena at a failed load's abort is not its one WARN line")
    if "return unified_cache_last_release::ARENA_HELD;" not in aborted:
        out.append("a held arena at a failed load's abort does not stop the release")
    # Between the abort and the free of the failed model its buffers hold the arena; after that free a holder is a
    # leak. A start can come in either window, so it is a WARN that says so and names the holders, at WARN.
    if not ordered(started, "GGML_LOG_WARN(", STARTED_TEXT, LEAK_TEXT, START_HOLDERS,
                   "return unified_cache_last_release::ARENA_HELD;") or "GGML_LOG_ERROR(" in started:
        out.append("a held arena at a load's start is not a WARN line with its holders named at WARN")
    if not ordered(rest, "GGML_LOG_ERROR(", RELEASE_HOLDERS, "return unified_cache_last_release::ARENA_HELD;"):
        out.append("a held arena at a model release is not an ERROR line with its holders named at ERROR")
    report = fn_body(uc, REPORT_SIG)
    if len(HOLDERS_AT_LEVEL.findall(report)) != 1 or "GGML_LOG_ERROR(" in report:
        out.append("the holders report does not log its holders line at the level its caller passes")
    # Its definition and three callers: the refused rebuild and the model release at ERROR, the load's start at WARN.
    if uc.count("arena_report_holders(") != 4 or REBUILD_HOLDERS not in uc:
        out.append("the holders report has a caller beyond the three, or a refused rebuild names its holders below "
                   "ERROR")
    if not ordered(body, "arena_released_.store(true, std::memory_order_release);", "GGML_LOG_INFO(",
                   "return unified_cache_last_release::EMPTIED;"):
        out.append("the release is not an INFO line")
    return out


def check_deferred_mark(uc: str, uh: str) -> list:
    body = fn_body(uc, MEMBER_SIG)
    if not body:
        return ["the cache's last release member is missing"]
    out = []
    aborted = _branch(_branch(body, HELD_HEAD), ABORTED_HEAD)
    started = _branch(_branch(body, HELD_HEAD), STARTED_HEAD)
    if not ordered(aborted, DEFER_SET, "return unified_cache_last_release::ARENA_HELD;"):
        out.append("a held arena at a failed load's abort does not mark the release deferred")
    # The start's cause only keeps the mark (a held arena) or clears it (a release that empties): only the abort sets it.
    if CAUSE_ENUM not in uh or not started or "last_release_deferred_" in started:
        out.append("a load's start has no cause of its own, or its held arena touches the deferred mark")
    if uc.count("last_release_deferred_.store(true") != 1:
        out.append("the deferred mark is set somewhere other than at a failed load's held arena")
    if f"if (!arena_active()) {{ {DEFER_CLEAR} return unified_cache_last_release::EMPTIED; }}" not in body \
            or not ordered(body, "arena_released_.store(true, std::memory_order_release);", DEFER_CLEAR,
                           "return unified_cache_last_release::EMPTIED;"):
        out.append("a release that empties the cache does not clear the deferred mark")
    if uc.count("last_release_deferred_.store(false") != 2 or body.count(DEFER_CLEAR) != 2:
        out.append("the deferred mark clears somewhere other than where the release empties the cache")
    if DEFER_MEMBER not in uh or DEFER_READ not in uh:
        out.append("the cache does not carry the deferred mark, or does not answer it")
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
    for text, sig in ((gs, STEP_SIG), (gs, START_SIG), (uc, MEMBER_SIG), (uc, PAIR_SIG), (uc, RESERVE_SIG)):
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
    # Inside the load's transaction, which no model's teardown can begin beside, and before the reserve, which then
    # reserves the arena of a cache the deferred release emptied.
    if not ordered(begin, "result = registry->begin_outer();", "ggml_sycl_model_loading_effects(true, true);",
                   START_CALL, "if (ggml_sycl_load_start_reserve_released_arenas()) {"):
        out.append("the load's start does not run the deferred release inside its transaction, once the caches "
                   "exist and before its reserve")
    if gs.count(DEFERRED) != 1 or begin.count(START_CALL) != 1:
        out.append("the deferred release runs somewhere other than once at the load's start")
    if gs.count(START_CAUSE) != 1:
        out.append("the start's cause is passed somewhere other than the load's start")
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


def test_hook_retires_the_mmid_workspaces_before_the_step():
    assert check_retire_order(GS) == []


def test_abort_ends_in_the_step_outside_every_try():
    assert check_abort(GS) == []


def test_step_resets_terms_through_one_list():
    assert check_step(GS) == []


def test_member_releases_in_order_and_reserves_nothing():
    assert check_member(UC, UH) == []


def test_member_names_what_holds_on():
    assert check_refusals_named(UC) == []


def test_member_marks_a_deferred_release():
    assert check_deferred_mark(UC, UH) == []


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


def test_block_at_reads_raw_source_past_a_comment_apostrophe():
    src = "void f() { // the loader's buffer\n    g(); /* it's { */ }\nint x = '}';"
    assert block_at(src, 0) == "{ // the loader's buffer\n    g(); /* it's { */ }"


# ---- mutation witnesses: each applies the regression its check guards ----


def test_witness_hook_without_the_step():
    assert check_hook(mutate_in(GS, HOOK_SIG, HOOK_CALL, ""))


def test_witness_hook_step_before_the_drop():
    m = mutate_in(GS, HOOK_SIG, DROP, HOOK_CALL + " " + DROP)
    assert check_hook(mutate_in(m, HOOK_SIG, HOOK_CALL + " }", "}"))


def _hook_retire(gs: str) -> str:
    m = HOOK_RETIRE.search(fn_body(gs, HOOK_SIG))
    assert m, "witness anchor: the hook's retire"
    return m.group(0)


def test_witness_hook_without_the_retire():
    assert check_retire_order(mutate_in(GS, HOOK_SIG, _hook_retire(GS), ""))


def test_witness_hook_retires_after_the_step():
    m = mutate_in(GS, HOOK_SIG, _hook_retire(GS), "")
    assert check_retire_order(mutate_in(m, HOOK_SIG, HOOK_CALL, HOOK_CALL + " " + _hook_retire(GS)))


def test_witness_retire_back_after_the_finalize():
    # The order this check exists to stop: the unload hook retires once the registry finalizes, after the step.
    m = mutate_in(GS, HOOK_SIG, _hook_retire(GS), "")
    old = "ggml_sycl::lifecycle_erase_placement_plan(owner.model.value, owner.load.value);"
    assert check_retire_order(mutate_in(m, UNLOAD_SIG, old, _hook_retire(GS) + " " + old))


def test_witness_retire_before_the_discovery_gate():
    m = mutate_in(GS, HOOK_SIG, _hook_retire(GS), "")
    anchor = "const ggml_sycl::moe_owner_key moe_owner = ggml_sycl_moe_owner_key(owner);"
    body = fn_def(m, EFFECTS_NAME)
    assert anchor in body, "witness anchor: the owner effects' discovery key"
    s = m.find(body)
    assert check_retire_order(m[:s] + body.replace(anchor, _hook_retire(GS) + " " + anchor, 1) + m[s + len(body):])


def test_witness_retire_in_the_unload_catch():
    anchor = "} catch (...) { (void) ggml_sycl_restore_latest_live_plan();"
    assert check_retire_order(mutate_in(GS, UNLOAD_SIG, anchor, anchor + " " + _hook_retire(GS)))


def test_witness_retire_in_a_failed_teardown_branch():
    anchor = "(void) registry->finalize_teardown(ticket, false); return GGML_SYCL_LIFECYCLE_EFFECT_FAILED;"
    assert check_retire_order(mutate_in(GS, UNLOAD_SIG, anchor, _hook_retire(GS) + " " + anchor))


def test_witness_retire_inside_a_branch_of_the_hook():
    r = _hook_retire(GS)
    assert check_retire_order(mutate_in(GS, HOOK_SIG, r, "if (owner.model.value != 0) { " + r + " }"))


def test_witness_load_end_loses_a_retire():
    body = fn_def(GS, "ggml_backend_sycl_model_load_end")
    k = body.find(RETIRE_NAME)
    assert k != -1, "witness anchor: the load end's retire"
    s = GS.find(body)
    assert check_retire_order(GS[:s] + body[:k] + "ggml_sycl::kept(" + body[k + len(RETIRE_NAME):] + GS[s + len(body):])


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
    assert check_step(mutate_in(GS, STEP_SIG, TERMS_RESET_IF, "if (true) {"))


def test_witness_step_resets_terms_of_a_cache_it_left_out():
    assert check_step(mutate_in(GS, STEP_SIG, TERMS_RESET_IF, f"if (released[d] == {EMPTIED}) {{"))


def test_witness_start_asks_every_cache():
    assert check_step(mutate_in(GS, STEP_SIG, ASK, "asked[d] = first == d ? true : asked[first];"))


def test_witness_left_out_cache_counts_as_emptied():
    assert check_step(mutate_in(GS, STEP_SIG, "all_emptied = all_emptied && !caches[d]->arena_active(); continue;",
                                "continue;"))


def test_witness_start_that_asked_none_puts_the_two_card_flag_back():
    assert check_step(mutate_in(GS, STEP_SIG, "if (all_emptied && (every || asked_any)) {", "if (all_emptied) {"))


def test_witness_start_that_asked_a_cache_never_puts_the_two_card_flag_back():
    assert check_step(mutate_in(GS, STEP_SIG, ASKED_ANY, ""))


def test_witness_step_records_a_cache_its_scope_left_out():
    m = mutate_in(GS, STEP_SIG, ASKED_ANY, "")
    assert check_step(mutate_in(m, STEP_SIG, LEFT_OUT, ASKED_ANY + " " + LEFT_OUT))


def test_witness_step_resets_terms_while_a_model_is_live():
    assert check_step(mutate_in(GS, STEP_SIG, f"if (released[d] == {MODEL_LIVE}) {{ return; }}", ""))


def test_witness_step_asks_a_shared_cache_twice():
    every = "released[d] = caches[d]->release_for_no_live_model(site, cause);"
    assert check_step(mutate_in(GS, STEP_SIG, DEDUPE, every))


def test_witness_step_runs_at_static_destruction():
    assert check_step(mutate_in(GS, STEP_SIG, SHUTDOWN_SKIP, ""))


def test_witness_member_releases_before_the_barrier():
    m = mutate_in(UC, MEMBER_SIG, "submit_barrier_all().wait();", "")
    assert check_member(mutate_in(m, MEMBER_SIG, "process_deferred_frees_public();",
                                  "submit_barrier_all().wait(); process_deferred_frees_public();"), UH)


def test_witness_member_drains_after_the_teardown():
    m = mutate_in(UC, MEMBER_SIG, "process_deferred_frees_public();", "")
    assert check_member(mutate_in(m, MEMBER_SIG, "arena_released_.store(true, std::memory_order_release);",
                                  "process_deferred_frees_public(); "
                                  "arena_released_.store(true, std::memory_order_release);"), UH)


def test_witness_member_waits_twice():
    assert check_member(mutate_in(UC, MEMBER_SIG, "process_deferred_frees_public();",
                                  "process_deferred_frees_public(); queue_.wait();"), UH)


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
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, RELEASE_HOLDERS, ""))


def test_witness_held_arena_named_below_error():
    m = mutate_in(UC, MEMBER_SIG, RELEASE_HOLDERS, RELEASE_HOLDERS.replace("_ERROR", "_WARN"))
    assert check_refusals_named(m)


def test_witness_refused_rebuild_names_its_holders_at_warn():
    assert check_refusals_named(mutate_in(UC, ENSURE_SIG, REBUILD_HOLDERS, REBUILD_HOLDERS.replace("_ERROR", "_WARN")))


def test_witness_held_arena_at_abort_is_an_error():
    m = mutate_in(UC, MEMBER_SIG, f"{ABORTED_HEAD} {DEFER_SET} GGML_LOG_WARN(",
                  f"{ABORTED_HEAD} {DEFER_SET} GGML_LOG_ERROR(")
    assert check_refusals_named(m)


def test_witness_abort_says_what_the_start_says():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, ABORTED_TEXT, STARTED_TEXT))


def test_witness_held_arena_at_start_is_an_error():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, f"{STARTED_HEAD} GGML_LOG_WARN(",
                                          f"{STARTED_HEAD} GGML_LOG_ERROR("))


def test_witness_held_arena_at_start_not_named():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, START_HOLDERS, ""))


def test_witness_held_arena_at_start_named_at_error():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, START_HOLDERS, START_HOLDERS.replace("_WARN", "_ERROR")))


def test_witness_held_arena_at_start_does_not_call_it_a_leak():
    assert check_refusals_named(mutate_in(UC, MEMBER_SIG, LEAK_TEXT, '"'))


def test_witness_held_arena_at_start_takes_the_model_release_path():
    started = _branch(_branch(fn_body(UC, MEMBER_SIG), HELD_HEAD), STARTED_HEAD)
    m = mutate_in(UC, MEMBER_SIG, STARTED_HEAD[:-1] + started, "")
    assert check_refusals_named(m) and check_deferred_mark(m, UH)


def test_witness_report_ignores_its_level():
    line = HOLDERS_AT_LEVEL.search(fn_body(UC, REPORT_SIG))
    assert line, "witness anchor: the holders report's line at its level"
    assert check_refusals_named(mutate_in(UC, REPORT_SIG, line.group(0), "GGML_LOG_ERROR(" + HOLDERS_LINE))


def test_witness_report_has_a_fourth_caller():
    m = mutate_in(UC, MEMBER_SIG, "compute_arena_unbind();",
                  "arena_report_holders(dev_id, live_zone_used, holders, site, GGML_LOG_LEVEL_INFO); "
                  "compute_arena_unbind();")
    assert check_refusals_named(m)


def test_witness_abort_does_not_mark_the_release_deferred():
    assert check_deferred_mark(mutate_in(UC, MEMBER_SIG, DEFER_SET, ""), UH)


def test_witness_held_arena_at_start_marks_the_release_deferred():
    m = mutate_in(UC, MEMBER_SIG, f"{STARTED_HEAD} GGML_LOG_WARN(", f"{STARTED_HEAD} {DEFER_SET} GGML_LOG_WARN(")
    assert check_deferred_mark(m, UH)


def test_witness_held_arena_at_start_clears_the_mark():
    m = mutate_in(UC, MEMBER_SIG, f"{STARTED_HEAD} GGML_LOG_WARN(", f"{STARTED_HEAD} {DEFER_CLEAR} GGML_LOG_WARN(")
    assert check_deferred_mark(m, UH)


def test_witness_start_has_no_cause_of_its_own():
    assert check_deferred_mark(UC, UH.replace(CAUSE_ENUM, CAUSE_ENUM.replace(", LOAD_START", ""), 1))


def test_witness_model_release_marks_the_release_deferred():
    m = mutate_in(UC, MEMBER_SIG, DEFER_SET, "")
    assert check_deferred_mark(mutate_in(m, MEMBER_SIG, HELD_HEAD, DEFER_SET + " " + HELD_HEAD), UH)


def test_witness_refused_release_clears_the_mark():
    assert check_deferred_mark(mutate_in(UC, MEMBER_SIG, "return unified_cache_last_release::RING_CLAIMED;",
                                         DEFER_CLEAR + " return unified_cache_last_release::RING_CLAIMED;"), UH)


def test_witness_emptied_cache_without_an_arena_keeps_the_mark():
    assert check_deferred_mark(mutate_in(UC, MEMBER_SIG, "if (!arena_active()) { " + DEFER_CLEAR,
                                         "if (!arena_active()) {"), UH)


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


def test_witness_load_start_skips_the_deferred_release():
    assert check_load_start(mutate_in(GS, BEGIN_SIG, START_CALL, ""), UC)


def test_witness_load_start_passes_the_aborts_cause():
    m = mutate_in(GS, BEGIN_SIG, START_CALL, START_CALL.replace(START_CAUSE, START_CAUSE.replace("_START", "_ABORTED")))
    assert check_load_start(m, UC)


def test_witness_abort_passes_the_starts_cause():
    m = mutate_in(GS, ABORT_SIG, ABORT_CALL, ABORT_CALL.replace("::LOAD_ABORTED", "::LOAD_START"))
    assert check_abort(m) and check_load_start(m, UC)


def test_witness_deferred_release_after_the_reserve():
    m = mutate_in(GS, BEGIN_SIG, START_CALL, "")
    assert check_load_start(mutate_in(m, BEGIN_SIG, "registry->bind_candidate(result.txn);",
                                      START_CALL + " registry->bind_candidate(result.txn);"), UC)


def test_witness_deferred_release_outside_the_transaction():
    m = mutate_in(GS, BEGIN_SIG, START_CALL, "")
    assert check_load_start(mutate_in(m, BEGIN_SIG, "registry = &ggml_sycl::lifecycle::global_registry();",
                                      START_CALL + " registry = &ggml_sycl::lifecycle::global_registry();"), UC)


def test_witness_abort_asks_only_the_deferred_caches():
    m = mutate_in(GS, ABORT_SIG, ABORT_CALL, ABORT_CALL.replace(EVERY, DEFERRED))
    assert check_abort(m) and check_load_start(m, UC)


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


def test_witness_last_resort_release_gone_while_another_function_releases():
    # later functions release the ring too, so only the function's own body can tell this release is gone
    k = GS_RAW.find(LAST_RESORT_SIG)
    assert k != -1
    gone = GS_RAW[:k] + GS_RAW[k:].replace("cache->release_pp_moe_onednn_scratch_ring()", "cache->keep_ring()", 1)
    assert "cache->release_pp_moe_onednn_scratch_ring()" in gone[k:]
    assert check_last_resort_comment(gone) == ["the decide pass's last-resort ring release is gone"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
