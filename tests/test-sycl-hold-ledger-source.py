"""Source contract for llama.cpp-kpjw review r7: the hold ledger's wiring.

The host-seam test (ggml/src/ggml-sycl/tests/test-hold-ledger.cpp) and test-zone-sizing execute the ledger's
arithmetic; this file pins the places that only text can: which caller reaches which entry, in which order, and which
allocation paths are covered by the compute scope. Every claim is checked on COMMENT-STRIPPED, whitespace-normalized
text, and every claim has a mutant that must make it fail (a claim that survives its own mutant pins nothing).

- C1 the settle releases the previous rung's compute buffers BEFORE its publish, through the same lambda the rungs use;
- I4 only a fit verdict (the dedicated exception type) sets rung_fit_refused for a reserve that threw;
- I3 a pinned -ub refreshes its epoch's KV room before it reserves; the refresh is an exported, registered entry;
- I1 the non-FA check takes its free memory from the ledger, not from a bare driver reading;
- I5 nothing allocates on a scheduler outside llama_context::sched_alloc_graph / sched_reserve_graph (K-shift included),
  and the projector's scheduler opens the same exported scope;
- I2 the ledger's unmodelled consumers and RELEASING rows are documented where the baseline is read;
- M5 truncating the per-rung record list is loud.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text: runs collapse to one space, and a re-wrap at a bracket or comma
    (`f(\\n a, b)`, `f(a ,b )`) reads the same as `f(a, b)`, so a clang-format run cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


CTX = read("src/llama-context.cpp")
CTX_H = read("src/llama-context.h")
KV = read("src/llama-kv-cache.cpp")
CLIP = read("tools/mtmd/clip.cpp")
UBATCH_H = read("src/llama-auto-ubatch.h")
SYCL_CPP = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
CACHE_CPP = read("ggml/src/ggml-sycl/unified-cache.cpp")
SYCL_H = read("ggml/include/ggml-sycl.h")
CACHE_H = read("ggml/src/ggml-sycl/unified-cache.hpp")


def _balanced(text: str, open_at: int) -> str:
    assert text[open_at] == "{"
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
    raise AssertionError("unbalanced braces")


# ---------------------------------------------------------------------------
# claims: each takes the RAW sources it reads, so a mutant is a text edit of the raw source
# ---------------------------------------------------------------------------


def claim_settle_releases_before_publish(ctx: str) -> bool:
    c = norm(ctx)
    if c.count("auto release_rung_buffers = [&]() {") != 1:
        return False
    # try_candidate and the settle both call the one lambda
    if c.count("release_rung_buffers();") != 2:
        return False
    settle_at = c.find("if (!sched_matches_last_good || cparams.n_ubatch != last_good) {")
    if settle_at == -1:
        return False
    settle = c[settle_at:]
    rel = settle.find("release_rung_buffers();")
    pub = settle.find("sycl_resync_runtime_context_flash_attn();")
    return 0 <= rel < pub


def claim_try_candidate_releases_first(ctx: str) -> bool:
    c = norm(ctx)
    return "auto try_candidate = [&](uint32_t c) -> const char * { rung_fit_refused = false; release_rung_buffers();" in c


def claim_fit_flag_needs_the_fit_exception(ctx: str, header: str) -> bool:
    c = norm(ctx)
    h = norm(header)
    if "struct llama_auto_ubatch_fit_refusal : public std::runtime_error" not in h:
        return False
    if (
        "rung_fit_refused = dynamic_cast<const llama_auto_ubatch_fit_refusal *>(&e) != nullptr;" not in c
    ):
        return False
    # the dedicated type is thrown from two places and nowhere else: the PLAN_REJECTED recheck, and sched_reserve()
    # for a transaction result that carries the fit verdict (the compute-buffer failures and the tenant section's
    # refusal return {status, reason, true}; the lifecycle failures return no verdict)
    if c.count("throw llama_auto_ubatch_fit_refusal(") != 2:
        return False
    if not re.search(
        r"if \(result\.fit_refusal\) \{ throw llama_auto_ubatch_fit_refusal\(result\.reason\); \} "
        r"throw std::runtime_error\(result\.reason\);",
        c,
    ):
        return False
    for what in ("pp", "tg"):
        if f'{{ sched_reserve_status::FAILED, "failed to allocate compute {what} buffers", true }}' not in c:
            return False
    # the tenant section's builder refuses a plan inconsistency, never a capacity shortfall: no fit verdict
    if "return { sched_reserve_status::REFUSED, tenant_reason };" not in c or "tenant_reason, true" in c:
        return False
    return bool(
        re.search(
            r"if \(rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED\) \{ throw llama_auto_ubatch_fit_refusal\(what\); \} "
            r"throw std::runtime_error\(what\);",
            c,
        )
    )


def claim_pinned_ub_refreshes_epoch_before_reserve(ctx: str) -> bool:
    c = norm(ctx)
    m = re.search(
        r"if \(llama_context_has_sycl_backend\(backends\)\) \{ llama_context_sycl_hold_epoch_refresh\(backends\); \} "
        r"#endif if \(sycl_auto_ubatch_trial\) \{ sycl_select_auto_ubatch\(params\.type_k, params\.type_v\); \} "
        r"else \{ sched_reserve\(\); \}",
        c,
    )
    return m is not None


def claim_refresh_entry_is_exported_and_registered(sycl_h: str, sycl_cpp: str, cache_cpp: str) -> bool:
    h = norm(sycl_h)
    s = norm(sycl_cpp)
    k = norm(cache_cpp)
    if "ggml_backend_sycl_planned_hold_epoch_refresh(ggml_backend_t backend);" not in h:
        return False
    if 'strcmp(name, "ggml_backend_sycl_planned_hold_epoch_refresh") == 0) { return (void *) ggml_backend_sycl_planned_hold_epoch_refresh; }' not in s:
        return False
    if "ggml_sycl::unified_cache_refresh_hold_epoch_kv_room( ctx->device, ctx->planned_scratch_owner, ggml_sycl_hold_kv_room(ctx->device, 0));" not in s.replace(
        "ggml_sycl::unified_cache_refresh_hold_epoch_kv_room(ctx->device,",
        "ggml_sycl::unified_cache_refresh_hold_epoch_kv_room( ctx->device,",
    ):
        return False
    # a refresh moves only the KV room of the owner's own live epoch
    return bool(
        re.search(
            r"void unified_cache_refresh_hold_epoch_kv_room\(int device_id, uint64_t owner, size_t kv_room\) \{.*?"
            r"if \(state\.owner != owner \|\| state\.epoch_n_ubatch == 0\) \{ return; \} state\.epoch_kv_room = kv_room; \}",
            k,
        )
    )


def claim_nonfa_free_comes_from_the_ledger(sycl_cpp: str) -> bool:
    s = norm(sycl_cpp)
    return bool(
        re.search(
            r"const size_t free_cmp = hold_query \? ggml_sycl::unified_cache_hold_free_before\(device, hold_query->owner, "
            r"free_mem, hold_query->rung_live\) : free_mem; "
            r"if \(ggml_sycl::unified_cache_nonfa_attn_scratch_fits_headroom\(nonfa_demand, free_cmp\)\)",
            s,
        )
    )


def claim_raw_rows_are_credited_by_origin(cache_cpp: str) -> bool:
    k = norm(cache_cpp)
    if (
        "rec.scheduler_compute = req.intent.constraints.spill_to_kv_zone_before_raw && "
        "!req.intent.constraints.forbid_vram_zone_spill;" not in k
    ):
        return False
    m = re.search(r"size_t unified_cache_raw_device_held_bytes\(int device_id, size_t \* compute_live\) \{(.*?)\n?\} static bool", k)
    body = m.group(1) if m else ""
    # held = every outside-arena device row INCLUDING releasing ones; credit = LIVE scheduler compute rows only
    return (
        "h.device != device_id || h.tier != alloc_tier::DEVICE_VRAM || h.zone_managed || row.from_arena" in body
        and "if (row.scheduler_compute && row.state == runtime_alloc_state::LIVE) {" in body
        and "RELEASING" not in body
    )


def claim_baseline_is_a_window_max(cache_cpp: str) -> bool:
    k = norm(cache_cpp)
    return (
        "state.baseline_cold = zone_hold_cold_update(state.baseline_owner == owner, state.baseline_cold, cand);" in k
        and "const size_t persistent = zone_hold_persistent_raw(raw_held, compute_live, rung_live);" in k
        and "return zone_hold_free_before(state.baseline_cold, persistent);" in k
    )


def claim_unmodelled_consumers_name_their_direction(cache_cpp: str) -> bool:
    # comments, so read the RAW text
    return (
        "each such consumer errs the same way for the check" in cache_cpp
        and "ADMITS what" in cache_cpp
        and "a stale-low first reading REFUSES" in cache_cpp
        and "A row that is RELEASING is still held" in cache_cpp
    )


def claim_rung_record_truncation_is_loud(cache_cpp: str) -> bool:
    k = norm(cache_cpp)
    m = re.search(r"if \(state\.rung_requests\.size\(\) >= kHoldRungRecordLimit\) \{(.*?)\} else \{", k)
    body = m.group(1) if m else ""
    # counted under the lock; the WARN itself is said after the lock is released (the mutex is a leaf)
    warn_at = k.find("GGML_LOG_WARN(", k.find("size_t unified_cache_note_runtime_request("))
    unlock_at = k.find("hold = state.hold; }", k.find("size_t unified_cache_note_runtime_request("))
    return (
        "state.rung_record_refusals++" in body
        and "GGML_LOG" not in body
        and 0 <= unlock_at < warn_at
        and "if (warn_refused) {" in k
    )


def claim_every_scheduler_allocation_goes_through_the_helpers(ctx: str, kv: str, ctx_h: str) -> bool:
    c = norm(ctx)
    # exactly one bare alloc call, inside its own helper; the one bare reserve besides the helper's is a MEASURE
    # state's, which reserves on a scheduler of its own that no plan scope or hold record covers
    if c.count("ggml_backend_sched_alloc_graph(") != 1 or c.count("ggml_backend_sched_reserve(") != 2:
        return False
    if (
        "const bool reserved = state.measure ? ggml_backend_sched_reserve(state.sched.get(), gf) : sched_reserve_graph(gf);"
        not in c
    ):
        return False
    if not re.search(
        r"bool llama_context::sched_alloc_graph\(ggml_cgraph \* gf\) \{ sycl_compute_scope_guard \w+\(sycl_compute_scope_fn\(\)\); "
        r"return ggml_backend_sched_alloc_graph\(sched\.get\(\), gf\); \}",
        c,
    ):
        return False
    if not re.search(
        r"bool llama_context::sched_reserve_graph\(ggml_cgraph \* gf\) \{ sycl_compute_scope_guard \w+\(sycl_compute_scope_fn\(\)\); "
        r"return ggml_backend_sched_reserve\(sched\.get\(\), gf\); \}",
        c,
    ):
        return False
    k = norm(kv)
    if "ggml_backend_sched_alloc_graph" in k or "ggml_backend_sched_reserve(" in k:
        return False
    if "lctx->sched_alloc_graph(gf)" not in k:
        return False
    h = norm(ctx_h)
    return "bool sched_alloc_graph(ggml_cgraph * gf);" in h and "bool sched_reserve_graph(ggml_cgraph * gf);" in h


def claim_clip_opens_the_exported_scope(clip: str) -> bool:
    c = norm(clip)
    return (
        '"ggml_backend_sycl_compute_alloc_scope"' in c
        and "{ clip_sycl_compute_scope sycl_scope(ctx_clip.backend); ggml_backend_sched_reserve(ctx_clip.sched.get(), gf); }" in c
        and "{ clip_sycl_compute_scope sycl_scope(ctx->backend); alloc_ok = ggml_backend_sched_alloc_graph(ctx->sched.get(), gf); }" in c
    )


def claim_fit_picks_the_kv_room_at_the_call_site(sycl_cpp: str) -> bool:
    """The fit's KV room: the realized check (rung live) judges with its epoch's snapshot, a transaction with the zone as
    it is now. The helper's arithmetic is host-tested; this pins that the fit calls it with the live-rung flag, the
    epoch lookup and the live room only when there is no epoch."""
    s = norm(sycl_cpp)
    return bool(
        re.search(
            r"const bool have_epoch = q\.rung_live && ggml_sycl::unified_cache_get_hold_epoch\(q\.device, q\.owner, "
            r"&epoch_n_ubatch, &epoch_kv_room\); "
            r"in\.kv_room = ggml_sycl::zone_hold_pick_kv_room\(q\.rung_live, have_epoch, epoch_kv_room, "
            r"have_epoch \? 0 : ggml_sycl_hold_kv_room\(q\.device, q\.kv_pending\)\);",
            s,
        )
    )


def claim_non_fa_check_counts_the_hold_term(sycl_cpp: str) -> bool:
    """The non-FA check's demand is the non-FA scratch PLUS the hold's worst-case spill bound, both compared with the
    ledger's free memory."""
    s = norm(sycl_cpp)
    return bool(
        re.search(
            r"const size_t hold_spill_bytes = hold_query \? ggml_sycl_planned_scratch_hold_spill_bound\(\*hold_query\) : 0; "
            r"const size_t nonfa_scratch_demand = ggml_sycl::unified_cache_nonfa_attn_scratch_demand_bytes\(n_head, n_ubatch, n_ctx\); "
            r"const size_t nonfa_demand = ggml_sycl::zone_hold_nonfa_demand\(nonfa_scratch_demand, hold_spill_bytes\);",
            s,
        )
        and "unified_cache_nonfa_attn_scratch_fits_headroom(nonfa_demand, free_cmp)" in s
    )


def claim_dl_refresh_lookup_names_the_exported_entry(ctx: str, sycl_cpp: str, sycl_h: str) -> bool:
    """Under GGML_BACKEND_DL the context finds the refresh by name. A typo there silently skips the refresh (the proc
    lookup returns null and the backend is skipped), so the string must be the one the backend registers and declares."""
    c = norm(ctx)
    m = re.search(
        r"llama_context_sycl_hold_epoch_refresh_proc\(ggml_backend_dev_t dev\) \{ return reinterpret_cast<decltype\(&ggml_backend_sycl_planned_hold_epoch_refresh\)>\("
        r'llama_context_sycl_proc_addr\(dev, "([A-Za-z0-9_]+)"\)\); \}',
        c,
    )
    if not m:
        return False
    name = m.group(1)
    s = norm(sycl_cpp)
    h = norm(sycl_h)
    return (
        name == "ggml_backend_sycl_planned_hold_epoch_refresh"
        and f'strcmp(name, "{name}") == 0) {{ return (void *) {name}; }}' in s
        and f"{name}(ggml_backend_t backend);" in h
    )


def claim_record_cap_is_one_constant_and_one_raw_filter_remains(sycl_cpp: str, cache_cpp: str, cache_h: str) -> bool:
    """The fit's rung buffer takes its size from the cache's record cap (one header constant, no second literal), and
    the raw-row filter exists once: the dead live-bytes duplicate is gone."""
    s = norm(sycl_cpp)
    return (
        "constexpr size_t kMaxRungs = ggml_sycl::kHoldRungRecordLimit;" in s
        and "zone_hold_rung_request rungs[kMaxRungs];" in s
        and "kMaxRungs = 32" not in s
        and "GGML_ASSERT(ggml_sycl::unified_cache_hold_rung_record_limit" not in s
        and "constexpr size_t kHoldRungRecordLimit = 32;" in norm(cache_h)
        and "unified_cache_raw_device_live_bytes" not in cache_cpp
        and "unified_cache_raw_device_live_bytes" not in cache_h
    )

def claim_no_bare_scheduler_allocation_elsewhere() -> bool:
    """Every other source that includes the ggml-backend scheduler API: no bare alloc/reserve outside the two helpers
    and the projector's scoped blocks."""
    allowed = {"src/llama-context.cpp", "tools/mtmd/clip.cpp"}
    for base in ("src", "tools", "common", "examples"):
        for p in sorted((ROOT / base).rglob("*.cpp")):
            rel = str(p.relative_to(ROOT))
            if rel in allowed:
                continue
            code = strip_comments(p.read_text(errors="replace"))
            if re.search(r"\bggml_backend_sched_(alloc_graph|reserve)\s*\(", code):
                return False
    return True


# ---------------------------------------------------------------------------
# the claims hold on the tree
# ---------------------------------------------------------------------------


def test_settle_releases_the_previous_rungs_buffers_before_its_publish():
    assert claim_settle_releases_before_publish(CTX)
    assert claim_try_candidate_releases_first(CTX)


def test_the_fit_flag_needs_the_dedicated_exception():
    assert claim_fit_flag_needs_the_fit_exception(CTX, UBATCH_H)


def test_a_pinned_ub_refreshes_its_epoch_before_reserving():
    assert claim_pinned_ub_refreshes_epoch_before_reserve(CTX)
    assert claim_refresh_entry_is_exported_and_registered(SYCL_H, SYCL_CPP, CACHE_CPP)


def test_the_non_fa_check_takes_free_memory_from_the_ledger():
    assert claim_nonfa_free_comes_from_the_ledger(SYCL_CPP)


def test_raw_rows_are_credited_by_origin_and_releasing_rows_stay_held():
    assert claim_raw_rows_are_credited_by_origin(CACHE_CPP)
    assert claim_baseline_is_a_window_max(CACHE_CPP)


def test_the_ledger_documents_which_way_each_unmodelled_consumer_errs():
    assert claim_unmodelled_consumers_name_their_direction(CACHE_CPP)


def test_truncating_the_rung_records_is_loud():
    assert claim_rung_record_truncation_is_loud(CACHE_CPP)


def test_every_scheduler_allocation_goes_through_the_scoped_helpers():
    assert claim_every_scheduler_allocation_goes_through_the_helpers(CTX, KV, CTX_H)
    assert claim_clip_opens_the_exported_scope(CLIP)
    assert claim_no_bare_scheduler_allocation_elsewhere()


def test_the_fit_picks_its_kv_room_at_the_call_site():
    assert claim_fit_picks_the_kv_room_at_the_call_site(SYCL_CPP)


def test_the_non_fa_check_counts_the_hold_term():
    assert claim_non_fa_check_counts_the_hold_term(SYCL_CPP)


def test_the_record_cap_is_one_constant_and_one_raw_filter_remains():
    assert claim_record_cap_is_one_constant_and_one_raw_filter_remains(SYCL_CPP, CACHE_CPP, CACHE_H)


def test_the_dl_refresh_lookup_names_the_exported_entry():
    assert claim_dl_refresh_lookup_names_the_exported_entry(CTX, SYCL_CPP, SYCL_H)


# ---------------------------------------------------------------------------
# mutants: each edit of the raw source must make its claim fail
# ---------------------------------------------------------------------------


def _once(raw: str, old: str, new: str) -> str:
    """Replace `old` once, ignoring every whitespace difference in the match: a clang-format run must not turn a
    mutant into a vacuous 'target not found' failure."""
    pat = re.compile(r"\s*".join(re.escape(ch) for ch in old if not ch.isspace()))
    found = len(pat.findall(raw))
    assert found == 1, f"mutation target not unique/found ({found}): {old[:60]!r}"
    return pat.sub(lambda _m: new, raw, count=1)


def _settle_release_line(raw: str) -> str:
    m = re.search(r"\n        release_rung_buffers\(\);\n        // The settle's publish is a fit check", raw)
    assert m, "settle release not found"
    return raw[: m.start()] + "\n        // The settle's publish is a fit check" + raw[m.end() :]


def test_mutant_settle_without_release_fails_the_claim():
    assert not claim_settle_releases_before_publish(_settle_release_line(CTX))


def test_mutant_settle_release_after_publish_fails_the_claim():
    moved = _settle_release_line(CTX)
    moved = _once(
        moved,
        "        std::exception_ptr settle_error;\n",
        "        std::exception_ptr settle_error;\n        std::string        settle_refusal_pre;\n",
    )
    # put the release after the publish try-block instead
    moved = _once(
        moved,
        "        if (settle_error) {\n            uint32_t largest_ub = 0;",
        "        release_rung_buffers();\n        if (settle_error) {\n            uint32_t largest_ub = 0;",
    )
    assert not claim_settle_releases_before_publish(moved)


def test_mutant_settle_with_its_own_release_copy_fails_the_claim():
    mutated = _once(
        CTX,
        "        release_rung_buffers();\n        // The settle's publish is a fit check",
        "        synchronize();\n        sched.reset();\n        // The settle's publish is a fit check",
    )
    assert not claim_settle_releases_before_publish(mutated)


def test_mutant_try_candidate_without_release_fails_the_claim():
    mutated = _once(CTX, "rung_fit_refused = false;\n        release_rung_buffers();", "rung_fit_refused = false;")
    assert not claim_try_candidate_releases_first(mutated)


def test_mutant_any_exception_is_a_fit_verdict_fails_the_claim():
    mutated = _once(
        CTX,
        "rung_fit_refused = dynamic_cast<const llama_auto_ubatch_fit_refusal *>(&e) != nullptr;",
        "rung_fit_refused = true;",
    )
    assert not claim_fit_flag_needs_the_fit_exception(mutated, UBATCH_H)


def test_mutant_busy_lifecycle_throws_the_fit_exception_fails_the_claim():
    mutated = _once(
        CTX,
        "                throw std::runtime_error(what);\n            }\n        }",
        "                throw llama_auto_ubatch_fit_refusal(what);\n            }\n        }",
    )
    assert not claim_fit_flag_needs_the_fit_exception(mutated, UBATCH_H)


def test_mutant_fit_exception_is_a_plain_runtime_error_fails_the_claim():
    mutated = _once(UBATCH_H, "struct llama_auto_ubatch_fit_refusal : public std::runtime_error", "struct llama_auto_ubatch_fit_refusal")
    assert not claim_fit_flag_needs_the_fit_exception(CTX, mutated)


def test_mutant_pinned_ub_without_refresh_fails_the_claim():
    mutated = _once(CTX, "            llama_context_sycl_hold_epoch_refresh(backends);\n", "")
    assert not claim_pinned_ub_refreshes_epoch_before_reserve(mutated)


def test_mutant_pinned_ub_refreshes_after_reserve_fails_the_claim():
    moved = _once(CTX, "            llama_context_sycl_hold_epoch_refresh(backends);\n", "")
    moved = _once(
        moved,
        "        if (sycl_auto_ubatch_trial) {\n            sycl_select_auto_ubatch(params.type_k, params.type_v);\n        } else {\n            sched_reserve();\n        }\n",
        "        if (sycl_auto_ubatch_trial) {\n            sycl_select_auto_ubatch(params.type_k, params.type_v);\n        } else {\n            sched_reserve();\n            llama_context_sycl_hold_epoch_refresh(backends);\n        }\n",
    )
    assert not claim_pinned_ub_refreshes_epoch_before_reserve(moved)


def test_mutant_refresh_not_registered_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        'if (strcmp(name, "ggml_backend_sycl_planned_hold_epoch_refresh") == 0) {',
        'if (strcmp(name, "ggml_backend_sycl_planned_hold_epoch_refresh_x") == 0) {',
    )
    assert not claim_refresh_entry_is_exported_and_registered(SYCL_H, mutated, CACHE_CPP)


def test_mutant_refresh_ignores_the_owner_check_fails_the_claim():
    mutated = _once(
        CACHE_CPP,
        "    if (state.owner != owner || state.epoch_n_ubatch == 0) {\n        return;\n    }\n    state.epoch_kv_room = kv_room;",
        "    state.epoch_kv_room = kv_room;",
    )
    assert not claim_refresh_entry_is_exported_and_registered(SYCL_H, SYCL_CPP, mutated)


def test_mutant_non_fa_reads_a_bare_driver_figure_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        "ggml_sycl::unified_cache_hold_free_before(device, hold_query->owner, free_mem, hold_query->rung_live) :\n            free_mem;",
        "free_mem :\n            free_mem;",
    )
    assert not claim_nonfa_free_comes_from_the_ledger(mutated)


def test_mutant_releasing_rows_not_held_fails_the_claim():
    mutated = _once(
        CACHE_CPP,
        "            held = h.size > SIZE_MAX - held ? SIZE_MAX : held + h.size;\n            // Only a LIVE scheduler",
        "            if (row.state == runtime_alloc_state::RELEASING) {\n                continue;\n            }\n            held = h.size > SIZE_MAX - held ? SIZE_MAX : held + h.size;\n            // Only a LIVE scheduler",
    )
    # the row is still counted for the credit below; the claim that nothing special-cases RELEASING in the held sum
    # is the documented rule, so the mutant must trip the text check on the RELEASING token
    assert not claim_raw_rows_are_credited_by_origin(mutated)


def test_mutant_credit_of_any_row_state_fails_the_claim():
    mutated = _once(
        CACHE_CPP,
        "if (row.scheduler_compute && row.state == runtime_alloc_state::LIVE) {",
        "if (row.scheduler_compute) {",
    )
    assert not claim_raw_rows_are_credited_by_origin(mutated)


def test_mutant_credit_by_timing_not_origin_fails_the_claim():
    mutated = _once(
        CACHE_CPP,
        "rec.scheduler_compute = req.intent.constraints.spill_to_kv_zone_before_raw &&\n                                !req.intent.constraints.forbid_vram_zone_spill;",
        "rec.scheduler_compute = true;",
    )
    assert not claim_raw_rows_are_credited_by_origin(mutated)


def test_mutant_baseline_is_the_latest_reading_fails_the_claim():
    mutated = _once(
        CACHE_CPP,
        "state.baseline_cold  = zone_hold_cold_update(state.baseline_owner == owner, state.baseline_cold, cand);",
        "state.baseline_cold  = cand;",
    )
    assert not claim_baseline_is_a_window_max(mutated)


def test_mutant_undocumented_direction_fails_the_claim():
    mutated = _once(CACHE_CPP, "a stale-low first reading REFUSES", "a first reading is read")
    assert not claim_unmodelled_consumers_name_their_direction(mutated)


def test_mutant_silent_truncation_fails_the_claim():
    mutated = _once(CACHE_CPP, "warn_refused = state.rung_record_refusals++ == 0;", "warn_refused = state.rung_record_refusals == 0;")
    assert not claim_rung_record_truncation_is_loud(mutated)


def test_mutant_warn_under_the_state_lock_fails_the_claim():
    mutated = _once(
        CACHE_CPP,
        "warn_refused = state.rung_record_refusals++ == 0;",
        'warn_refused = state.rung_record_refusals++ == 0; if (warn_refused) { GGML_LOG_WARN("x"); }',
    )
    assert not claim_rung_record_truncation_is_loud(mutated)


def test_mutant_a_bare_sched_alloc_in_the_context_fails_the_claim():
    mutated = _once(
        CTX,
        "bool llama_context::sched_reserve_graph(ggml_cgraph * gf) {",
        "static bool sched_stray(ggml_backend_sched_t s, ggml_cgraph * g) { return ggml_backend_sched_alloc_graph(s, g); }\n"
        "bool llama_context::sched_reserve_graph(ggml_cgraph * gf) {",
    )
    assert not claim_every_scheduler_allocation_goes_through_the_helpers(mutated, KV, CTX_H)


def test_mutant_helper_without_scope_fails_the_claim():
    mutated = _once(
        CTX,
        "bool llama_context::sched_alloc_graph(ggml_cgraph * gf) {\n    sycl_compute_scope_guard sycl_scope(sycl_compute_scope_fn());\n",
        "bool llama_context::sched_alloc_graph(ggml_cgraph * gf) {\n",
    )
    assert not claim_every_scheduler_allocation_goes_through_the_helpers(mutated, KV, CTX_H)


def test_mutant_k_shift_back_to_a_bare_call_fails_the_claim():
    mutated = _once(KV, "lctx->sched_alloc_graph(gf)", "ggml_backend_sched_alloc_graph(lctx->get_sched(), gf)")
    assert not claim_every_scheduler_allocation_goes_through_the_helpers(CTX, mutated, CTX_H)


def test_mutant_clip_alloc_outside_the_scope_fails_the_claim():
    mutated = _once(
        CLIP,
        "        clip_sycl_compute_scope sycl_scope(ctx->backend);\n        alloc_ok = ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);",
        "        alloc_ok = ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);",
    )
    assert not claim_clip_opens_the_exported_scope(mutated)


def test_mutant_clip_reserve_outside_the_scope_fails_the_claim():
    mutated = _once(
        CLIP,
        "            clip_sycl_compute_scope sycl_scope(ctx_clip.backend);\n            ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);",
        "            ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);",
    )
    assert not claim_clip_opens_the_exported_scope(mutated)


def test_mutant_fit_always_uses_the_live_kv_room_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        "in.kv_room      = ggml_sycl::zone_hold_pick_kv_room(q.rung_live, have_epoch, epoch_kv_room,\n                                                   have_epoch ? 0 : ggml_sycl_hold_kv_room(q.device, q.kv_pending));",
        "in.kv_room = ggml_sycl_hold_kv_room(q.device, q.kv_pending);",
    )
    assert not claim_fit_picks_the_kv_room_at_the_call_site(mutated)


def test_mutant_fit_ignores_the_epoch_flag_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        "q.rung_live && ggml_sycl::unified_cache_get_hold_epoch(q.device, q.owner, &epoch_n_ubatch, &epoch_kv_room);",
        "ggml_sycl::unified_cache_get_hold_epoch(q.device, q.owner, &epoch_n_ubatch, &epoch_kv_room);",
    )
    assert not claim_fit_picks_the_kv_room_at_the_call_site(mutated)


def test_mutant_non_fa_drops_the_hold_term_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        "ggml_sycl::zone_hold_nonfa_demand(nonfa_scratch_demand, hold_spill_bytes);",
        "nonfa_scratch_demand;",
    )
    assert not claim_non_fa_check_counts_the_hold_term(mutated)


def test_mutant_non_fa_hold_bound_is_always_zero_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        "hold_query ? ggml_sycl_planned_scratch_hold_spill_bound(*hold_query) : 0;",
        "0;",
    )
    assert not claim_non_fa_check_counts_the_hold_term(mutated)


def test_mutant_dl_refresh_lookup_typo_fails_the_claim():
    mutated = _once(
        CTX,
        'llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_planned_hold_epoch_refresh")',
        'llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_planned_hold_epoch_refesh")',
    )
    assert not claim_dl_refresh_lookup_names_the_exported_entry(mutated, SYCL_CPP, SYCL_H)


def test_mutant_dl_refresh_registration_typo_fails_the_claim():
    mutated = _once(
        SYCL_CPP,
        'if (strcmp(name, "ggml_backend_sycl_planned_hold_epoch_refresh") == 0) {',
        'if (strcmp(name, "ggml_backend_sycl_planned_hold_epoch_refresh_") == 0) {',
    )
    assert not claim_dl_refresh_lookup_names_the_exported_entry(CTX, mutated, SYCL_H)


def test_mutant_second_literal_for_the_record_cap_fails_the_claim():
    mutated = _once(SYCL_CPP, "kMaxRungs = ggml_sycl::kHoldRungRecordLimit;", "kMaxRungs = 32;")
    assert not claim_record_cap_is_one_constant_and_one_raw_filter_remains(mutated, CACHE_CPP, CACHE_H)


def test_mutant_header_cap_constant_gone_fails_the_claim():
    mutated = _once(CACHE_H, "constexpr size_t kHoldRungRecordLimit = 32;", "constexpr size_t kHoldRungRecordLimitX = 32;")
    assert not claim_record_cap_is_one_constant_and_one_raw_filter_remains(SYCL_CPP, CACHE_CPP, mutated)

def test_mutant_second_raw_filter_returns_fails_the_claim():
    mutated = CACHE_CPP + "\nsize_t unified_cache_raw_device_live_bytes(int device_id) { return 0; }\n"
    assert not claim_record_cap_is_one_constant_and_one_raw_filter_remains(SYCL_CPP, mutated, CACHE_H)


def _rewrap(raw: str, old: str, new: str) -> str:
    """Re-wrap `old` the way clang-format might (a break after every opening bracket and comma), without changing its
    meaning, so a claim over the result must still hold."""
    return _once(raw, old, new)


def test_a_rewrapped_dl_lookup_still_satisfies_its_claim():
    rewrapped = _rewrap(
        CTX,
        'llama_context_sycl_proc_addr(dev, "ggml_backend_sycl_planned_hold_epoch_refresh")',
        'llama_context_sycl_proc_addr(\n            dev,\n            "ggml_backend_sycl_planned_hold_epoch_refresh" )',
    )
    assert claim_dl_refresh_lookup_names_the_exported_entry(rewrapped, SYCL_CPP, SYCL_H)


def test_a_rewrapped_kv_room_call_still_satisfies_its_claim():
    rewrapped = _rewrap(
        SYCL_CPP,
        "ggml_sycl::zone_hold_pick_kv_room(q.rung_live, have_epoch, epoch_kv_room,",
        "ggml_sycl::zone_hold_pick_kv_room(\n        q.rung_live ,have_epoch,\n        epoch_kv_room,",
    )
    assert claim_fit_picks_the_kv_room_at_the_call_site(rewrapped)


def test_a_rewrapped_non_fa_demand_still_satisfies_its_claim():
    rewrapped = _rewrap(
        SYCL_CPP,
        "ggml_sycl::zone_hold_nonfa_demand(nonfa_scratch_demand, hold_spill_bytes);",
        "ggml_sycl::zone_hold_nonfa_demand(\n        nonfa_scratch_demand,\n        hold_spill_bytes );",
    )
    assert claim_non_fa_check_counts_the_hold_term(rewrapped)


def test_a_rewrapped_credit_filter_still_satisfies_its_claim():
    rewrapped = _rewrap(
        CACHE_CPP,
        "if (row.scheduler_compute && row.state == runtime_alloc_state::LIVE) {",
        "if (\n                row.scheduler_compute && row.state == runtime_alloc_state::LIVE ) {",
    )
    assert claim_raw_rows_are_credited_by_origin(rewrapped)


# --- the typed fit refusal is capacity only; the MEASURE bare reserve is that state's own scheduler ----------------


def test_mutant_tenant_builder_refusal_as_a_fit_verdict_fails_the_claim():
    mutated = _once(CTX, "return { sched_reserve_status::REFUSED, tenant_reason };", "return { sched_reserve_status::REFUSED, tenant_reason, true };")
    assert not claim_fit_flag_needs_the_fit_exception(mutated, UBATCH_H)


def test_mutant_a_publish_result_other_than_plan_rejected_as_a_fit_verdict_is_pinned_by_the_publish_gate():
    # the publish's fit flag is pinned in test-sycl-publish-status-source.py (rc == PLAN_REJECTED only); this gate
    # pins that a transaction result without the flag never throws the typed refusal
    mutated = _once(CTX, "if (result.fit_refusal) { throw llama_auto_ubatch_fit_refusal(result.reason); }", "throw llama_auto_ubatch_fit_refusal(result.reason);")
    assert not claim_fit_flag_needs_the_fit_exception(mutated, UBATCH_H)


def test_mutant_the_contexts_own_scheduler_reserves_bare_fails_the_claim():
    mutated = _once(CTX, ": sched_reserve_graph(gf);", ": ggml_backend_sched_reserve(sched.get(), gf);")
    assert not claim_every_scheduler_allocation_goes_through_the_helpers(mutated, KV, CTX_H)


def test_mutant_every_graph_reserve_bare_fails_the_claim():
    mutated = _once(CTX, "state.measure ? ggml_backend_sched_reserve(state.sched.get(), gf)", "true ? ggml_backend_sched_reserve(state.sched.get(), gf)")
    assert not claim_every_scheduler_allocation_goes_through_the_helpers(mutated, KV, CTX_H)


def test_mutant_a_measure_state_through_the_helper_fails_the_claim():
    # the helper reserves on the context's `sched`, never a MEASURE state's own: routing it there would reserve the
    # wrong scheduler
    mutated = _once(CTX, "state.measure ? ggml_backend_sched_reserve(state.sched.get(), gf) : sched_reserve_graph(gf)", "sched_reserve_graph(gf)")
    assert not claim_every_scheduler_allocation_goes_through_the_helpers(mutated, KV, CTX_H)
