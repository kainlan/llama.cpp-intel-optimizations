"""Source contract for llama.cpp-p6i0: the load's planned compute term is a named RUNTIME consumer.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, -ub 512, no -c): the scheduler's compute buffer was planned by nobody. The weight
pack filled the shared zone, the RUNTIME zone held the PP MoE ring, and the buffer's 1543.1 MB chunk landed raw,
outside the arena, in the external headroom, while its 512 MB chunk fit no tier and the context was refused.

The fix this gate pins: the load measures the compute buffer at the probe placement and hands its per-chunk peaks to
ggml_backend_sycl_load_reserve_compute_term, which sizes a planned RUNTIME term from them at the RUNTIME TLSF's grain
(zone_compute_term_bytes, no headroom) and stores it beside the dense scratch terms. The RUNTIME zone requirement
folds it in, so the zone is sized for the buffer before the weight pack; the ring re-plan leaves it that room; and the
landing line names it. Around it: the early inventory plan stages the probe plan the PROBE measure installs, the
admitted check's units export answers the reserve's own sizing rule, and the measure override's install proc answers
only the name of its kv_shape form.

The state term beside it: the context memory the measure placed on a device's plain buffer type (Qwen3.8's recurrent
state, 112.6 MiB on the B70) is allocated RUNTIME-first before the compute buffer, so the term alone did not hold the
buffer (the 512 MB chunk landed raw). ggml_backend_sycl_load_reserve_state_term sizes it at the same grain and stores
it as its own planned RUNTIME term, which the requirement and the ring count beside the compute term.

Owner-keyed (llama.cpp-ihk6, canonical contract 12.4): both terms are each owner's own contribution, keyed by the model
slot and slot generation of the load that measured them; the device reads the merge over its owners. The reserve keys
its store by the open load's owner under that load's effect; a load's rollback and a slot's release each drop only
their owner's contribution and republish the merge; and no production code clears the whole table. The table itself
is unit-tested in ggml/src/ggml-sycl/tests/test-planned-load-terms.cpp; this gate pins how the backend reaches it.

Every claim is checked on COMMENT-STRIPPED, whitespace-normalized text and has a mutant that must make it fail.

Pytest-style (module-level test_* functions): register with llama_test_pytest.
"""

import re
from pathlib import Path

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
    """Comment-stripped, whitespace-canonical text, so a clang-format re-wrap cannot turn a claim red."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    return t


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
LLAMA_CTX = read("src/llama-context.cpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose normalized text starts with `signature` (string
    literals are skipped while balancing). Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


def _skip_literal(text: str, i: int) -> int:
    """The index just past the string or character literal that opens at `i`."""
    quote, j, n = text[i], i + 1, len(text)
    while j < n and text[j] != quote:
        j += 2 if text[j] == "\\" else 1
    return j + 1


def depth_at(text: str, pos: int) -> int:
    """The brace depth at `pos` (literals skipped). In a body() result a statement of the function itself is at 1."""
    depth, i = 0, 0
    while 0 <= i < pos:
        if text[i] in "\"'":
            i = _skip_literal(text, i)
            continue
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return depth


def block_end(text: str, pos: int) -> int:
    """The index of the brace closing the innermost block that contains `pos` (literals skipped); -1 when none."""
    depth, i = 0, pos
    while 0 <= i < len(text):
        if text[i] in "\"'":
            i = _skip_literal(text, i)
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return i
            depth -= 1
        i += 1
    return -1


RESERVE_SIG = "bool ggml_backend_sycl_load_reserve_compute_term(ggml_sycl_load_txn txn,"
OPEN_EFFECT = "const auto effect = ggml_sycl::lifecycle::global_registry().acquire_load_effect({ txn.id });"
SIZE_SIG = "static bool ggml_sycl_load_compute_term_size(const uint64_t * chunk_bytes, uint32_t n_chunks, size_t * out)"
UNITS_SIG = "bool ggml_backend_sycl_load_compute_term_bytes(const uint64_t * chunk_bytes, uint32_t n_chunks, uint64_t * out)"
SETTER_SIG = "bool unified_cache_set_planned_compute_term(int device_id, lifecycle::SlotToken owner, size_t bytes)"
SET_TERM_SIG = "static bool unified_cache_set_planned_load_term(int device_id, lifecycle::SlotToken owner,"
PUBLISH_SIG = "static void planned_load_terms_publish_locked()"
NAME_REPLACED_SIG = "static void planned_load_terms_name_replaced(lifecycle::SlotToken owner, int device_id,"
DROP_SIG = "bool unified_cache_drop_planned_load_terms(lifecycle::SlotToken owner)"
REQ_SIG = "bool unified_cache_get_planned_runtime_zone_requirement(int device_id, size_t * out)"
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
LANDING_SIG = "static void ggml_sycl_log_compute_buffer_landing(int device, const std::string & name, size_t size,"
LOAD_SIG = "static void ggml_sycl_model_loading_effects(bool loading, bool outer)"
TEARDOWN_SIG = "static void ggml_sycl_release_model_slot_resources(ggml_sycl::lifecycle::ModelToken owner)"
ABORT_SIG = "static bool ggml_sycl_abort_owner_effects_noexcept(ggml_sycl::lifecycle::ModelToken owner,"
DROP_CALL = "(void) ggml_sycl::unified_cache_drop_planned_load_terms(owner.owner);"
DROP_DECL = "bool unified_cache_drop_planned_load_terms(lifecycle::SlotToken owner) noexcept;"
DROP_DEF = "bool unified_cache_drop_planned_load_terms(lifecycle::SlotToken owner) noexcept {"
STALE_LOOP = "for (int stale_device = 0; stale_device < GGML_SYCL_MAX_DEVICES; ++stale_device) {"
# The owner store's shape, with its local names captured rather than pinned.
GUARD_RE = re.compile(r"std::lock_guard<std::mutex> \w+\(g_planned_load_terms\.mutex\);")
OWNER_SET_RE = re.compile(
    r"if \(!g_planned_load_terms\.owners\.set\(device_id, owner, term, bytes, &(\w+)\)\) \{ return false; \}")
PUBLISH_CALL = "planned_load_terms_publish_locked();"
DEVICE_LOOP_RE = re.compile(r"for \(int (\w+) = 0; \1 < GGML_SYCL_MAX_DEVICES; \+\+\1\) \{")
# Every way to write the merged atomics: the atomic writers, assignment and compound assignment, and increments.
MERGE_WRITE_RE = re.compile(
    r"(?:\+\+|--)\s*g_planned_(?:compute|state)_term_bytes\b"
    r"|\bg_planned_(?:compute|state)_term_bytes\s*\[[^\]]*\]\s*"
    r"(?:\.\s*(?:store|exchange|fetch_\w+|compare_exchange_\w+)\s*\(|(?:<<|>>|[-+*/%&|^])?=(?!=)|\+\+|--)")
HEADER = read("ggml/src/ggml-sycl/unified-cache.hpp")
STAGE_SIG = "ggml_sycl_lifecycle_result ggml_backend_sycl_stage_inventory_plan(const ggml_sycl_tensor_inventory * inventory,"
STAGE_PROBE = ("if (early) { const uint64_t txn = effect.owner.load.value; "
               "const auto candidate = ggml_sycl::lifecycle_find_candidate_placement_plan(txn); "
               "if (candidate && candidate->plan) { ggml_sycl::lifecycle_stage_probe_placement_plan(txn, "
               "ggml_sycl::placement_plan(*candidate->plan), candidate->kv_info, candidate->model_n_layer); } }")


# ---- (a) the reservation: refusals first, the allocator's grain, the planner term only ----------------------------


def claim_the_units_are_the_grain_at_the_buffer_alignment(sycl: str) -> bool:
    """One rule for the reservation's units: zone_compute_term_bytes at the compute buffer type's alignment."""
    return body(norm(sycl), SIZE_SIG) == (
        "{ return ggml_sycl::zone_compute_term_bytes(chunk_bytes, n_chunks, GGML_SYCL_BUFFER_BASE_ALIGNMENT, out); }")


def claim_reserve_sizes_at_the_grain_and_writes_only_the_term(sycl: str) -> bool:
    """The export admits the mutation, refuses a closed txn, n_ctx 0, a bad device and a device with no arena, then
    stores exactly what the reservation's units answered as the open load's own contribution: no headroom, no ledger.
    The load effect is the one source for both "is this the open load" and "who owns it" (llama.cpp-ihk6): no second
    open-transaction read, no live-model mask."""
    b = body(norm(sycl), RESERVE_SIG)
    return (claim_the_units_are_the_grain_at_the_buffer_alignment(sycl) and bool(b) and ordered(
        b, "sycl_module_mutation_guard module_guard;", "if (!module_guard) { return false; }", OPEN_EFFECT,
        "if (!effect) {", "else if (n_ctx == 0)",
        "else if (device < 0 || device >= ggml_sycl_info().total_gpu_count || device >= GGML_SYCL_MAX_DEVICES)",
        "if (cache == nullptr || !cache->arena_active())",
        "else if (!ggml_sycl_load_compute_term_size(chunk_bytes, n_chunks, &bytes))",
        "else if (!ggml_sycl::unified_cache_set_planned_compute_term(device, effect.owner.owner, bytes))",
        "if (why != nullptr)", "return false;", "return true;")
        and b.count("unified_cache_set_planned_compute_term(") == 1 and b.count("acquire_load_effect(") == 1
        and "ggml_sycl_load_txn_is_open" not in b and "live_mask" not in b
        and "ledger" not in b and "ggml_sycl_load_record_compute_term" not in b)


def claim_the_units_export_is_the_reserves_rule_and_stateless(sycl: str) -> bool:
    """The loader's admitted check sizes C-hat and c(P) through this export, so it must answer the reserve's own rule,
    and read or write nothing: no module admission, no planner term, no ledger."""
    b = body(norm(sycl), UNITS_SIG)
    return (claim_the_units_are_the_grain_at_the_buffer_alignment(sycl) and bool(b)
            and "if (out == nullptr || !ggml_sycl_load_compute_term_size(chunk_bytes, n_chunks, &bytes)) { return false; }" in b
            and "*out = static_cast<uint64_t>(bytes);" in b
            and "zone_compute_term_bytes" not in b and "unified_cache" not in b and "ledger" not in b
            and "module_guard" not in b
            and ('strcmp(name, "ggml_backend_sycl_load_compute_term_bytes") == 0) { '
                 "return (void *) ggml_backend_sycl_load_compute_term_bytes; }") in norm(sycl))


def claim_owner_store_republishes_under_the_lock(cache: str) -> bool:
    """One owner's term (llama.cpp-ihk6): the owner table replaces that owner's value, and the merge the readers load is
    republished for every device in the same locked step, so a merge computed before a later contribution cannot be
    published over it, and a device the store swept a stale contribution off is republished too. Each swept device is
    named after the lock is released, by the helper every owner store shares (llama.cpp-45qo), which walks every
    device. No device-global merge input and no live-model flag remain."""
    t = norm(cache)
    b = body(t, SET_TERM_SIG)
    guard = GUARD_RE.search(b) if b else None
    if guard is None or b.count(PUBLISH_CALL) != 1:
        return False
    scope_end = block_end(b, guard.end())
    stored = OWNER_SET_RE.search(b, guard.end())
    published = b.find(PUBLISH_CALL, guard.end())
    if stored is None or scope_end < 0 or not (stored.end() <= published < scope_end):
        return False
    named = b.find("planned_load_terms_name_replaced(owner, device_id, %s);" % stored.group(1), scope_end)
    helper = body(t, NAME_REPLACED_SIG)
    loop = DEVICE_LOOP_RE.search(helper) if helper else None
    if named < 0 or loop is None:
        return False
    loop_body = helper[loop.end():block_end(helper, loop.end())]
    swept = "replaced.generation[%s]" % loop.group(1)
    return ("GGML_LOG_WARN(" in loop_body and "were still standing on device %d" in loop_body and swept in loop_body
            and "zone_dense_scratch_merge_input" not in b and "other_model_live" not in b)


def claim_publish_is_the_merge_with_release(cache: str) -> bool:
    """The device-visible terms are the owner table's merge on every device, both stored with release semantics, and
    nothing else writes them: no whole-table clear, no second writer, whether by store, exchange, read-modify-write,
    assignment or increment."""
    t = norm(cache)
    b = body(t, PUBLISH_SIG)
    return (bool(b) and ordered(
        b, "for (int device_id = 0; device_id < GGML_SYCL_MAX_DEVICES; ++device_id) {",
        "g_planned_compute_term_bytes[device_id].store("
        "g_planned_load_terms.owners.merged(device_id, planned_load_term::COMPUTE), std::memory_order_release);",
        "g_planned_state_term_bytes[device_id].store(g_planned_load_terms.owners.merged(device_id, "
        "planned_load_term::STATE), std::memory_order_release);")
            and len(MERGE_WRITE_RE.findall(t)) == 2 and len(MERGE_WRITE_RE.findall(b)) == 2)


def claim_setter_stores_the_owners_term(cache: str) -> bool:
    """The compute setter is the owner store for the compute term, and carries the owner instead of a live flag."""
    return (body(norm(cache), SETTER_SIG)
            == "{ return unified_cache_set_planned_load_term(device_id, owner, planned_load_term::COMPUTE, bytes); }"
            and claim_owner_store_republishes_under_the_lock(cache) and claim_publish_is_the_merge_with_release(cache))


# ---- (b) the term is drawn against: requirement, ring admission, landing line --------------------------------------


def claim_requirement_folds_in_the_term(cache: str) -> bool:
    """The RUNTIME zone requirement adds the term with the overflow checked, as it does the dense plans."""
    req = body(norm(cache), REQ_SIG)
    return (bool(req) and ordered(
        req, "const size_t compute = unified_cache_get_planned_compute_term_bytes(device_id);",
        "compute > SIZE_MAX - base - mmq_src1 - dequant_f16 ||",
        "*out = base + mmq_src1 + dequant_f16 + compute + state;"))


def claim_ring_leaves_the_term_room(sycl: str) -> bool:
    """The ring re-plan counts the term as pending RUNTIME demand, before the ring is re-planned."""
    txn = body(norm(sycl), TXN_SIG)
    return bool(txn) and ordered(
        txn, "ring_kv_zone.runtime_pending_bytes += dense_scratch_runtime_bytes;",
        "ring_kv_zone.runtime_pending_bytes += ggml_sycl::unified_cache_get_planned_compute_term_bytes(ctx->device);",
        "ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch, probe_mode, &ring_kv_zone)")


def claim_landing_line_names_the_term(sycl: str) -> bool:
    """The compute buffer's landing line names the zone and the planned term it drew against."""
    b = body(norm(sycl), LANDING_SIG)
    return (bool(b) and "GGML_LOG_WARN(" in b and "zone=%s planned_compute_term=%.1f MiB" in b
            and "ggml_sycl::unified_cache_get_planned_compute_term_bytes(device)" in b)


# ---- (c) the terms leave by their owner's key (llama.cpp-ihk6, canonical contract 12.4) ---------------------------


def claim_drop_removes_the_owner_and_republishes(cache: str) -> bool:
    """A drop removes one owner's contributions and republishes every device's merge under the same lock."""
    return body(norm(cache), DROP_SIG) == (
        "{ std::lock_guard<std::mutex> lock(g_planned_load_terms.mutex); "
        "if (!g_planned_load_terms.owners.drop(owner)) { return false; } "
        "planned_load_terms_publish_locked(); return true; }")


def claim_the_drop_cannot_throw(cache: str, header: str) -> bool:
    """The drop is noexcept where it is declared and where it is defined, which is what lets the rollback call it
    outside every try stage (and outside the frozen catch-handler debt of that function)."""
    return DROP_DECL in norm(header) and DROP_DEF in norm(cache)


def claim_teardown_drops_only_its_slot(sycl: str) -> bool:
    """The released slot drops its own contribution, unconditionally: whether another model is live or a load is in
    flight decides nothing, because neither owns this slot's term."""
    b = body(norm(sycl), TEARDOWN_SIG)
    return (bool(b) and b.count(DROP_CALL) == 1 and b.count("unified_cache_drop_planned_load_terms(") == 1
            and depth_at(b, b.find(DROP_CALL)) == 1 and "live_mask" not in b and "g_sycl_in_model_load" not in b)


def claim_abort_drops_only_its_owner(sycl: str) -> bool:
    """The aborted load drops its own contribution after the last try stage, whichever stage that is, unconditionally
    and outside any try, so no failed stage before it skips it and no catch handler is added for it; whatever is live
    is left alone."""
    b = body(norm(sycl), ABORT_SIG)
    return (bool(b) and b.endswith("} catch (...) { clean = false; } " + DROP_CALL + " return clean; }")
            and b.count("unified_cache_drop_planned_load_terms(") == 1 and depth_at(b, b.find(DROP_CALL)) == 1
            and "live_mask" not in b)


def claim_load_entry_leaves_the_terms_alone(sycl: str) -> bool:
    """A new load's entry is no reset point for anybody's terms: a contribution without a live owner cannot exist,
    so there is nothing for it to clear."""
    b = body(norm(sycl), LOAD_SIG)
    return (bool(b) and "if (outer) {" in b and "planned_load_terms" not in b
            and "planned_compute_term" not in b and "planned_state_term" not in b)


def claim_only_the_owner_ends_drop(sycl: str, cache: str, header: str) -> bool:
    """No whole-table clear survives anywhere, and the only drops are the two owner ends: the slot release and the
    rollback."""
    t = norm(sycl)
    calls = [m.start() for m in re.finditer(r"\bunified_cache_drop_planned_load_terms\(", t)]
    spans = []
    for sig in (TEARDOWN_SIG, ABORT_SIG):
        b = body(t, sig)
        at = t.find(b) if b else -1
        spans.append((at, at + len(b)) if at >= 0 else (-1, -1))
    return (len(calls) == 2 and all(any(a <= c < z for a, z in spans) for c in calls)
            and all(sum(1 for c in calls if a <= c < z) == 1 for a, z in spans)
            and all("clear_planned_load_terms" not in norm(x) for x in (sycl, cache, header))
            and len(re.findall(r"\bunified_cache_drop_planned_load_terms\(", norm(cache))) == 1)


# ---- (b0) the probe plan is staged by the early inventory plan -----------------------------------------------------


def claim_early_plan_stages_the_probe_plan(sycl: str) -> bool:
    """The PROBE-stage measure installs the probe plan, and only the early arm of the inventory plan stages it: a copy
    of the load's early candidate, after every device planned it. Without it every SYCL load's probe measure refuses
    (no plan staged), and nothing on the host would notice."""
    b = body(norm(sycl), STAGE_SIG)
    return (bool(b) and ordered(b, "if (early) { ggml_backend_sycl_compute_placement_plan_early(backend, inventory); }",
                                STAGE_PROBE)
            and b.count("lifecycle_stage_probe_placement_plan(") == 1)


def test_the_early_plan_stages_the_probe_plan():
    assert claim_early_plan_stages_the_probe_plan(SYCL)


# ---- (b) the measure's plan override is reached only under the name of its kv_shape form ---------------------------

INSTALL_OLD = '"ggml_backend_sycl_measure_plan_override_install"'
INSTALL_KV = '"ggml_backend_sycl_measure_plan_override_install_kv"'


def claim_install_proc_answers_only_its_kv_name(sycl: str, llama: str) -> bool:
    """The install entry gained `kv_shape`, so it is exported, and looked up, under a new name only: a libllama built
    for the two-argument form then finds no proc and its measure refuses by name instead of passing garbage."""
    return (("strcmp(name, " + INSTALL_KV + ") == 0") in norm(sycl) and INSTALL_OLD not in sycl
            and INSTALL_KV in llama and INSTALL_OLD not in llama)


def test_the_install_proc_answers_only_its_kv_name():
    assert claim_install_proc_answers_only_its_kv_name(SYCL, LLAMA_CTX)


def test_the_reservation_sizes_at_the_grain_and_writes_only_the_term():
    assert claim_reserve_sizes_at_the_grain_and_writes_only_the_term(SYCL)


def test_the_units_export_is_the_reserves_rule_and_stateless():
    assert claim_the_units_export_is_the_reserves_rule_and_stateless(SYCL)


def test_the_setter_stores_the_owners_term():
    assert claim_setter_stores_the_owners_term(CACHE)


def test_the_requirement_folds_in_the_term():
    assert claim_requirement_folds_in_the_term(CACHE)


def test_the_ring_leaves_the_term_room():
    assert claim_ring_leaves_the_term_room(SYCL)


def test_the_landing_line_names_the_term():
    assert claim_landing_line_names_the_term(SYCL)


def test_the_drop_removes_the_owner_and_republishes():
    assert claim_drop_removes_the_owner_and_republishes(CACHE)


def test_the_slot_release_drops_only_its_slot():
    assert claim_teardown_drops_only_its_slot(SYCL)


def test_the_abort_drops_only_its_owner():
    assert claim_abort_drops_only_its_owner(SYCL)


def test_the_drop_cannot_throw():
    assert claim_the_drop_cannot_throw(CACHE, HEADER)


def test_the_load_entry_leaves_the_terms_alone():
    assert claim_load_entry_leaves_the_terms_alone(SYCL)


def test_only_the_owner_ends_drop_and_nothing_clears_the_table():
    assert claim_only_the_owner_ends_drop(SYCL, CACHE, HEADER)


# ---- mutants: every claim must fail against the thing it forbids --------------------------------------------------


def _once(raw: str, old: str, new: str, within: str = "") -> str:
    """The mutant: `old` replaced by `new` in the source, all three compared as norm() text so a clang-format re-wrap
    of the source moves no anchor. The anchor must match exactly once, in the body of `within` when it is given (for
    an anchor the file repeats), else in the whole file. The claims norm() their input, so they read the result."""
    text, o, n = norm(raw), norm(old), norm(new)
    scope = body(text, within) if within else text
    assert scope.count(o) == 1, f"mutant anchor must match exactly once: {old!r} x{scope.count(o)}"
    return text.replace(scope, scope.replace(o, n, 1), 1) if within else text.replace(o, n, 1)


_TERM_SET = "} else if (!ggml_sycl::unified_cache_set_planned_compute_term(device, effect.owner.owner, bytes)) {"


def test_mutant_reserve_with_headroom_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET, _TERM_SET.replace("effect.owner.owner, bytes)", "effect.owner.owner, bytes + bytes / 8)")))


_SIZE_CALL = "return ggml_sycl::zone_compute_term_bytes(chunk_bytes, n_chunks, GGML_SYCL_BUFFER_BASE_ALIGNMENT, out);"


def test_mutant_reserve_without_the_grain_fails():
    mutant = _once(SYCL, _SIZE_CALL, _SIZE_CALL.replace("GGML_SYCL_BUFFER_BASE_ALIGNMENT", "1"))
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(mutant)
    assert not claim_the_units_export_is_the_reserves_rule_and_stateless(mutant)


def test_mutant_reserve_sizing_past_the_shared_rule_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "} else if (!ggml_sycl_load_compute_term_size(chunk_bytes, n_chunks, &bytes)) {",
              "} else if (!ggml_sycl::zone_compute_term_bytes(chunk_bytes, n_chunks, 1, &bytes)) {"))


def test_mutant_units_export_with_its_own_rule_fails():
    assert not claim_the_units_export_is_the_reserves_rule_and_stateless(
        _once(SYCL, "if (out == nullptr || !ggml_sycl_load_compute_term_size(chunk_bytes, n_chunks, &bytes)) {",
              "if (out == nullptr || !ggml_sycl::zone_compute_term_bytes(chunk_bytes, n_chunks, 1, &bytes)) {"))


def test_mutant_units_export_writing_the_planner_term_fails():
    assert not claim_the_units_export_is_the_reserves_rule_and_stateless(
        _once(SYCL, "*out = static_cast<uint64_t>(bytes);",
              "*out = static_cast<uint64_t>(bytes); (void) ggml_sycl::unified_cache_set_planned_compute_term("
              "0, ggml_sycl::lifecycle::SlotToken{ 0, 1 }, bytes);"))


def test_mutant_units_proc_not_answered_fails():
    assert not claim_the_units_export_is_the_reserves_rule_and_stateless(
        _once(SYCL, 'if (strcmp(name, "ggml_backend_sycl_load_compute_term_bytes") == 0) {',
              'if (strcmp(name, "ggml_backend_sycl_load_compute_term_bytes_x") == 0) {'))


def test_mutant_reserve_on_a_closed_txn_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "if (!effect) {", "if (false) {", within=RESERVE_SIG))


def test_mutant_reserve_asking_the_open_txn_twice_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "if (!effect) {", "if (!effect || !ggml_sycl_load_txn_is_open(txn.id)) {", within=RESERVE_SIG))


def test_mutant_reserve_without_the_arena_check_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "if (cache == nullptr || !cache->arena_active()) {", "if (cache == nullptr) {", within=RESERVE_SIG))


def test_mutant_reserve_writing_the_ledger_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET,
              _TERM_SET + " (void) ggml_sycl_load_record_compute_term(txn.id, device, bytes, n_ctx);"))


def test_mutant_reserve_keyed_by_a_fixed_owner_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET, _TERM_SET.replace("effect.owner.owner", "ggml_sycl::lifecycle::SlotToken{ 0, 1 }")))


def test_mutant_reserve_keyed_by_the_live_mask_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET, _TERM_SET.replace(
            "effect.owner.owner", "ggml_sycl::lifecycle::SlotToken{ ggml_sycl::lifecycle::global_registry().live_mask(), 1 }")))


_OWNER_SET = "if (!g_planned_load_terms.owners.set(device_id, owner, term, bytes, &replaced)) { return false; }"
_PUBLISH_ONE = "planned_load_terms_publish_locked(); }"


def test_mutant_setter_back_to_a_device_global_merge_fails():
    assert not claim_setter_stores_the_owners_term(
        _once(CACHE, _OWNER_SET,
              "g_planned_compute_term_bytes[device_id].store(zone_dense_scratch_merge_input("
              "g_planned_compute_term_bytes[device_id].load(std::memory_order_acquire), bytes, other_model_live), "
              "std::memory_order_release);", within=SET_TERM_SIG))


def test_mutant_setter_not_republishing_fails():
    assert not claim_setter_stores_the_owners_term(_once(CACHE, _PUBLISH_ONE, "}", within=SET_TERM_SIG))


def test_mutant_setter_republishing_outside_the_lock_fails():
    assert not claim_setter_stores_the_owners_term(
        _once(CACHE, _OWNER_SET + " " + _PUBLISH_ONE, _OWNER_SET + " } planned_load_terms_publish_locked();",
              within=SET_TERM_SIG))


def _rename(raw: str, within: str, old: str, new: str) -> str:
    """Every whole-word `old` renamed to `new` in the body of `within`, on norm() text: a refactor, not a mutant."""
    text = norm(raw)
    scope = body(text, within)
    assert re.search(r"\b%s\b" % re.escape(old), scope), f"rename target {old!r} absent"
    return text.replace(scope, re.sub(r"\b%s\b" % re.escape(old), new, scope), 1)


def test_control_setter_with_its_locals_renamed_passes():
    """Property, not spelling: the guard, the swept record and the loop index may take any name."""
    renamed = _rename(_rename(_rename(CACHE, SET_TERM_SIG, "lock", "guard"), SET_TERM_SIG, "replaced", "swept"),
                      NAME_REPLACED_SIG, "stale_device", "d")
    assert claim_owner_store_republishes_under_the_lock(renamed)
    assert claim_setter_stores_the_owners_term(renamed)


def test_mutant_setter_naming_only_the_storing_device_fails():
    assert not claim_setter_stores_the_owners_term(
        _once(CACHE, STALE_LOOP, "for (int stale_device = device_id; stale_device == device_id; ++stale_device) {",
              within=NAME_REPLACED_SIG))


def test_mutant_setter_never_naming_the_sweep_fails():
    assert not claim_setter_stores_the_owners_term(
        _once(CACHE, "planned_load_terms_name_replaced(owner, device_id, replaced);", "", within=SET_TERM_SIG))


def test_mutant_compute_setter_storing_the_state_term_fails():
    assert not claim_setter_stores_the_owners_term(
        _once(CACHE, "return unified_cache_set_planned_load_term(device_id, owner, planned_load_term::COMPUTE, bytes);",
              "return unified_cache_set_planned_load_term(device_id, owner, planned_load_term::STATE, bytes);"))


def test_mutant_setter_carrying_the_live_flag_fails():
    assert not claim_setter_stores_the_owners_term(
        _once(CACHE, SETTER_SIG, "bool unified_cache_set_planned_compute_term(int device_id, lifecycle::SlotToken owner, "
                                 "size_t bytes, bool other_model_live)"))


def test_mutant_publish_relaxed_fails():
    assert not claim_publish_is_the_merge_with_release(
        _once(CACHE, "g_planned_load_terms.owners.merged(device_id, planned_load_term::COMPUTE), std::memory_order_release);",
              "g_planned_load_terms.owners.merged(device_id, planned_load_term::COMPUTE), std::memory_order_relaxed);"))


def test_mutant_a_second_writer_of_the_merge_fails():
    assert not claim_publish_is_the_merge_with_release(
        _once(CACHE, "size_t unified_cache_get_planned_state_term_bytes(int device_id) {",
              "void unified_cache_zero_planned_state_terms() { for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) { "
              "g_planned_state_term_bytes[d].store(0, std::memory_order_release); } } "
              "size_t unified_cache_get_planned_state_term_bytes(int device_id) {"))


def test_mutant_an_exchange_writer_of_the_merge_fails():
    assert not claim_publish_is_the_merge_with_release(
        _once(CACHE, "size_t unified_cache_get_planned_state_term_bytes(int device_id) {",
              "void unified_cache_zero_planned_state_terms() { for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) { "
              "(void) g_planned_state_term_bytes[d].exchange(0, std::memory_order_acq_rel); } } "
              "size_t unified_cache_get_planned_state_term_bytes(int device_id) {"))


def test_mutant_an_assignment_writer_of_the_merge_fails():
    assert not claim_publish_is_the_merge_with_release(
        _once(CACHE, "size_t unified_cache_get_planned_compute_term_bytes(int device_id) {",
              "void unified_cache_zero_planned_compute_terms() { for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) { "
              "g_planned_compute_term_bytes[d] = 0; } } "
              "size_t unified_cache_get_planned_compute_term_bytes(int device_id) {"))


def test_mutant_publish_for_one_device_only_fails():
    assert not claim_publish_is_the_merge_with_release(
        _once(CACHE, "for (int device_id = 0; device_id < GGML_SYCL_MAX_DEVICES; ++device_id) {",
              "for (int device_id = 0; device_id < 1; ++device_id) {", within=PUBLISH_SIG))


def test_mutant_requirement_without_the_term_fails():
    assert not claim_requirement_folds_in_the_term(
        _once(CACHE, "*out = base + mmq_src1 + dequant_f16 + compute + state;", "*out = base + mmq_src1 + dequant_f16 + state;"))


def test_mutant_requirement_unchecked_fails():
    assert not claim_requirement_folds_in_the_term(
        _once(CACHE, "compute > SIZE_MAX - base - mmq_src1 - dequant_f16", "false"))


def test_mutant_ring_not_charged_the_term_fails():
    assert not claim_ring_leaves_the_term_room(
        _once(SYCL, "ring_kv_zone.runtime_pending_bytes += ggml_sycl::unified_cache_get_planned_compute_term_bytes(ctx->device);",
              ";"))


def test_mutant_landing_line_without_the_term_fails():
    assert not claim_landing_line_names_the_term(
        _once(SYCL, "zone=%s planned_compute_term=%.1f MiB\\n", "zone=%s\\n"))


_RELEASE_CALLS = "g_sycl_lifecycle_release_slot_calls.fetch_add(1, std::memory_order_relaxed);"
_RESET_STAGE_END = "ggml_sycl_reset_model_load_scratch_state(true); } catch (...) { clean = false; }"


def test_mutant_drop_not_republishing_fails():
    assert not claim_drop_removes_the_owner_and_republishes(
        _once(CACHE, "planned_load_terms_publish_locked(); return true;", "return true;", within=DROP_SIG))


def test_mutant_drop_declared_throwing_fails():
    assert not claim_the_drop_cannot_throw(CACHE, _once(HEADER, DROP_DECL, DROP_DECL.replace(" noexcept;", ";")))


def test_mutant_drop_defined_throwing_fails():
    assert not claim_the_drop_cannot_throw(_once(CACHE, DROP_DEF, DROP_DEF.replace(" noexcept {", " {")), HEADER)


def test_mutant_drop_clearing_every_owner_fails():
    assert not claim_drop_removes_the_owner_and_republishes(
        _once(CACHE, "if (!g_planned_load_terms.owners.drop(owner)) { return false; }",
              "g_planned_load_terms.owners = planned_load_terms{};", within=DROP_SIG))


def test_control_slot_release_dropping_first_passes():
    """Property, not position: the release may drop its slot's terms anywhere at its top level."""
    moved = _once(_once(SYCL, DROP_CALL, "", within=TEARDOWN_SIG), "const uint32_t slot = owner.owner.slot;",
                  DROP_CALL + " const uint32_t slot = owner.owner.slot;", within=TEARDOWN_SIG)
    assert claim_teardown_drops_only_its_slot(moved)
    assert claim_only_the_owner_ends_drop(moved, CACHE, HEADER)


def test_mutant_slot_release_not_dropping_fails():
    assert not claim_teardown_drops_only_its_slot(_once(SYCL, DROP_CALL, "", within=TEARDOWN_SIG))


def test_mutant_slot_release_dropping_only_when_alone_fails():
    assert not claim_teardown_drops_only_its_slot(
        _once(SYCL, DROP_CALL, "if (ggml_sycl::lifecycle::global_registry().live_mask() == 0) { " + DROP_CALL + " }",
              within=TEARDOWN_SIG))


def test_mutant_slot_release_dropping_the_load_in_flight_fails():
    assert not claim_teardown_drops_only_its_slot(
        _once(SYCL, DROP_CALL,
              "(void) ggml_sycl::unified_cache_drop_planned_load_terms("
              "ggml_sycl::lifecycle::global_registry().current_active_slot());", within=TEARDOWN_SIG))


def test_control_abort_with_a_stage_added_before_the_drop_passes():
    """Property, not the stage's name: a new last try stage leaves the drop after it, still outside any try."""
    assert claim_abort_drops_only_its_owner(
        _once(SYCL, "} catch (...) { clean = false; } " + DROP_CALL,
              "} catch (...) { clean = false; } try { (void) owner; } catch (...) { clean = false; } " + DROP_CALL,
              within=ABORT_SIG))


def test_mutant_abort_not_dropping_fails():
    assert not claim_abort_drops_only_its_owner(_once(SYCL, DROP_CALL, "", within=ABORT_SIG))


def test_mutant_abort_dropping_only_when_alone_fails():
    assert not claim_abort_drops_only_its_owner(
        _once(SYCL, DROP_CALL, "if (ggml_sycl::lifecycle::global_registry().live_mask() == 0) { " + DROP_CALL + " }",
              within=ABORT_SIG))


def test_mutant_abort_drop_skipped_by_a_failed_reset_fails():
    assert not claim_abort_drops_only_its_owner(
        _once(SYCL, _RESET_STAGE_END + " " + DROP_CALL,
              "ggml_sycl_reset_model_load_scratch_state(true); " + DROP_CALL + " } catch (...) { clean = false; }",
              within=ABORT_SIG))


def test_mutant_abort_drop_in_a_try_stage_of_its_own_fails():
    assert not claim_abort_drops_only_its_owner(
        _once(SYCL, DROP_CALL, "try { " + DROP_CALL + " } catch (...) { clean = false; }", within=ABORT_SIG))


def test_mutant_abort_drop_before_the_reset_fails():
    assert not claim_abort_drops_only_its_owner(
        _once(SYCL, "try { " + _RESET_STAGE_END + " " + DROP_CALL, DROP_CALL + " try { " + _RESET_STAGE_END,
              within=ABORT_SIG))


_LIVE_MASK_READ = "const uint32_t live_mask = ggml_sycl::lifecycle::global_registry().live_mask();"


def test_mutant_load_entry_clearing_again_fails():
    mutant = _once(SYCL, _LIVE_MASK_READ,
                   _LIVE_MASK_READ + " if (live_mask == 0) { ggml_sycl::unified_cache_clear_planned_load_terms(); }",
                   within=LOAD_SIG)
    assert not claim_load_entry_leaves_the_terms_alone(mutant)
    assert not claim_only_the_owner_ends_drop(mutant, CACHE, HEADER)


def test_mutant_a_third_drop_site_fails():
    assert not claim_only_the_owner_ends_drop(
        _once(SYCL, _LIVE_MASK_READ,
              _LIVE_MASK_READ + " (void) ggml_sycl::unified_cache_drop_planned_load_terms("
              "ggml_sycl::lifecycle::global_registry().current_active_slot());", within=LOAD_SIG),
        CACHE, HEADER)


def test_mutant_a_whole_table_clear_declared_fails():
    assert not claim_only_the_owner_ends_drop(
        SYCL, CACHE, _once(HEADER, DROP_DECL, DROP_DECL + " void unified_cache_clear_planned_load_terms();"))



def test_mutant_backend_answering_the_old_install_name_fails():
    arm = "if (strcmp(name, " + INSTALL_KV + ") == 0) {"
    assert not claim_install_proc_answers_only_its_kv_name(
        _once(SYCL, arm, "if (strcmp(name, " + INSTALL_OLD + ") == 0) { return nullptr; } " + arm),
        LLAMA_CTX)


def test_mutant_llama_asking_the_old_install_name_fails():
    assert not claim_install_proc_answers_only_its_kv_name(SYCL, _once(LLAMA_CTX, INSTALL_KV, INSTALL_OLD))



_STAGE_CALL = ("ggml_sycl::lifecycle_stage_probe_placement_plan(txn, ggml_sycl::placement_plan(*candidate->plan), "
               "candidate->kv_info, candidate->model_n_layer);")


def test_mutant_probe_plan_not_staged_fails():
    assert not claim_early_plan_stages_the_probe_plan(_once(SYCL, _STAGE_CALL, ""))


def test_mutant_probe_plan_staged_by_the_late_arm_fails():
    assert not claim_early_plan_stages_the_probe_plan(
        _once(SYCL, "if (early) { const uint64_t txn = effect.owner.load.value;",
              "if (!early) { const uint64_t txn = effect.owner.load.value;", within=STAGE_SIG))


def test_mutant_probe_plan_staged_for_another_load_fails():
    assert not claim_early_plan_stages_the_probe_plan(
        _once(SYCL, "const uint64_t txn = effect.owner.load.value;", "const uint64_t txn = effect.owner.load.value + 1;",
              within=STAGE_SIG))


# ---- (b1) the load-time logs say what ran: the stage by name, the pp trace's n_seqs as reserved ---------------------

INSTALL_SIG = "bool ggml_backend_sycl_measure_plan_override_install_kv(uint64_t load_txn,"
STAGE_NAME_SIG = "static const char * ggml_sycl_measure_stage_name(ggml_sycl_measure_stage stage)"
PP_N_SEQS = "uint32_t pp_n_seqs = n_seqs;"


def claim_install_logs_name_the_stage(sycl: str) -> bool:
    """The KV re-fit WARN and the placement trace name the stage as the loader's refusal text does, never as an
    integer."""
    t = norm(sycl)
    name = body(t, STAGE_NAME_SIG)
    b = body(t, INSTALL_SIG)
    return (bool(name) and bool(b)
            and 'case GGML_SYCL_MEASURE_STAGE_PROBE: return "probe";' in name
            and 'case GGML_SYCL_MEASURE_STAGE_CANDIDATE_B: return "admitted";' in name
            and 'case GGML_SYCL_MEASURE_STAGE_CANDIDATE_C: return "late";' in name
            and ordered(b, "the KV re-fit for load %llu at the %s stage refused: %s",
                        "(unsigned long long) load_txn, ggml_sycl_measure_stage_name(stage), why.c_str());")
            and ordered(b, "measure plan override: load %llu stage %s: weights",
                        "(unsigned long long) load_txn, ggml_sycl_measure_stage_name(stage),")
            and "re-fit for load %llu at stage %d" not in b and "load %llu stage %d" not in b)


def claim_pp_trace_reads_the_reserved_n_seqs(llama: str) -> bool:
    """The closing pp reserve's compute trace records the n_seqs the graph was reserved with, from the one variable
    both reserve calls pass, rather than re-deriving the architecture condition."""
    t = norm(llama)
    at = t.find(PP_N_SEQS)
    blk = t[at:at + 2000] if at >= 0 else ""
    return (t.count(PP_N_SEQS) == 1 and ordered(
        blk, PP_N_SEQS, "case LLM_ARCH_KIMI_LINEAR: case LLM_ARCH_MINIMAX_01:", "pp_n_seqs = 1;",
        "gf = graph_reserve(state, n_tokens, pp_n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);",
        "default: gf = graph_reserve(state, n_tokens, pp_n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);",
        "g.kind = LLAMA_MEASURE_KIND_PP_AGAIN;", "g.n_seqs = pp_n_seqs;")
        and "? 1 : n_seqs" not in blk)


def test_the_install_logs_name_the_stage():
    assert claim_install_logs_name_the_stage(SYCL)


def test_the_pp_trace_reads_the_reserved_n_seqs():
    assert claim_pp_trace_reads_the_reserved_n_seqs(LLAMA_CTX)


def test_mutant_refit_warn_with_an_integer_stage_fails():
    assert not claim_install_logs_name_the_stage(
        _once(SYCL, "load %llu at the %s stage refused: %s\\n\", (unsigned long long) load_txn, "
                    "ggml_sycl_measure_stage_name(stage), why.c_str());",
              "load %llu at stage %d refused: %s\\n\", (unsigned long long) load_txn, (int) stage, why.c_str());",
              within=INSTALL_SIG))


def test_mutant_stage_names_swapped_fails():
    assert not claim_install_logs_name_the_stage(
        _once(SYCL, 'case GGML_SYCL_MEASURE_STAGE_CANDIDATE_B: return "admitted";',
              'case GGML_SYCL_MEASURE_STAGE_CANDIDATE_B: return "late";', within=STAGE_NAME_SIG))


def test_mutant_pp_trace_rederiving_the_arch_fails():
    assert not claim_pp_trace_reads_the_reserved_n_seqs(
        _once(LLAMA_CTX, "g.n_seqs = pp_n_seqs;",
              "g.n_seqs = (model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01) ? 1 : n_seqs;"))


def test_mutant_pp_reserve_not_passing_its_n_seqs_fails():
    assert not claim_pp_trace_reads_the_reserved_n_seqs(
        _once(LLAMA_CTX, "pp_n_seqs = 1; gf = graph_reserve(state, n_tokens, pp_n_seqs,",
              "pp_n_seqs = 1; gf = graph_reserve(state, n_tokens, 1,"))


# ---- (d) the state term: the context memory the RUNTIME zone takes before the compute buffer ----------------------

RESERVE_STATE_SIG = "bool ggml_backend_sycl_load_reserve_state_term(ggml_sycl_load_txn txn, int32_t device, uint64_t state_bytes)"
STATE_SETTER_SIG = "bool unified_cache_set_planned_state_term(int device_id, lifecycle::SlotToken owner, size_t bytes)"


def claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(sycl: str) -> bool:
    """The state export admits the mutation, refuses a closed txn, a bad device and a device with no arena, sizes the
    one buffer at the reservation's grain and stores that as the open load's own state term: no headroom, no ledger,
    no compute term; and the backend answers its proc name."""
    b = body(norm(sycl), RESERVE_STATE_SIG)
    return (claim_the_units_are_the_grain_at_the_buffer_alignment(sycl) and bool(b) and ordered(
        b, "sycl_module_mutation_guard module_guard;", "if (!module_guard) { return false; }", OPEN_EFFECT,
        "if (!effect) {",
        "else if (device < 0 || device >= ggml_sycl_info().total_gpu_count || device >= GGML_SYCL_MAX_DEVICES)",
        "if (cache == nullptr || !cache->arena_active())",
        "else if (!ggml_sycl_load_compute_term_size(&state_bytes, 1, &bytes))",
        "else if (!ggml_sycl::unified_cache_set_planned_state_term(device, effect.owner.owner, bytes))",
        "if (why != nullptr)", "return false;", "return true;")
        and b.count("unified_cache_set_planned_state_term(") == 1 and b.count("acquire_load_effect(") == 1
        and "ggml_sycl_load_txn_is_open" not in b and "live_mask" not in b
        and "unified_cache_set_planned_compute_term" not in b
        and "ledger" not in b and "ggml_sycl_load_record_compute_term" not in b
        and ('strcmp(name, "ggml_backend_sycl_load_reserve_state_term") == 0) { '
             "return (void *) ggml_backend_sycl_load_reserve_state_term; }") in norm(sycl))


def claim_state_setter_stores_the_owners_term(cache: str) -> bool:
    """The state setter is the same owner store, for the state term."""
    return (body(norm(cache), STATE_SETTER_SIG)
            == "{ return unified_cache_set_planned_load_term(device_id, owner, planned_load_term::STATE, bytes); }"
            and claim_owner_store_republishes_under_the_lock(cache) and claim_publish_is_the_merge_with_release(cache))


def claim_requirement_folds_in_the_state_term(cache: str) -> bool:
    """The RUNTIME zone requirement adds the state term after the compute term, with the overflow checked."""
    req = body(norm(cache), REQ_SIG)
    return (bool(req) and ordered(
        req, "const size_t compute = unified_cache_get_planned_compute_term_bytes(device_id);",
        "const size_t state = unified_cache_get_planned_state_term_bytes(device_id);",
        "state > SIZE_MAX - base - mmq_src1 - dequant_f16 - compute) { return false; }",
        "*out = base + mmq_src1 + dequant_f16 + compute + state;"))


def claim_ring_leaves_the_state_room(sycl: str) -> bool:
    txn = body(norm(sycl), TXN_SIG)
    return bool(txn) and ordered(
        txn, "ring_kv_zone.runtime_pending_bytes += ggml_sycl::unified_cache_get_planned_compute_term_bytes(ctx->device);",
        "ring_kv_zone.runtime_pending_bytes += ggml_sycl::unified_cache_get_planned_state_term_bytes(ctx->device);",
        "ggml_sycl_replan_pp_moe_onednn_ring(ctx->device, next_kv_info.n_ubatch, probe_mode, &ring_kv_zone)")


def test_the_state_reservation_sizes_at_the_grain_and_writes_only_the_term():
    assert claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(SYCL)


def test_the_state_setter_stores_the_owners_term():
    assert claim_state_setter_stores_the_owners_term(CACHE)


def test_the_requirement_folds_in_the_state_term():
    assert claim_requirement_folds_in_the_state_term(CACHE)


def test_the_ring_leaves_the_state_room():
    assert claim_ring_leaves_the_state_room(SYCL)


_STATE_SET = "} else if (!ggml_sycl::unified_cache_set_planned_state_term(device, effect.owner.owner, bytes)) {"


def test_mutant_state_reserve_with_headroom_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _STATE_SET, _STATE_SET.replace("effect.owner.owner, bytes)", "effect.owner.owner, bytes + bytes / 8)")))


def test_mutant_state_reserve_into_the_compute_term_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _STATE_SET, _STATE_SET.replace("unified_cache_set_planned_state_term(", "unified_cache_set_planned_compute_term(")))


def test_mutant_state_reserve_without_the_grain_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "} else if (!ggml_sycl_load_compute_term_size(&state_bytes, 1, &bytes)) {",
              "} else if (!(bytes = state_bytes, true)) {"))


def test_mutant_state_reserve_on_a_closed_txn_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "if (!effect) {", "if (false) {", within=RESERVE_STATE_SIG))


def test_mutant_state_reserve_keyed_by_a_fixed_owner_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _STATE_SET, _STATE_SET.replace("effect.owner.owner", "ggml_sycl::lifecycle::SlotToken{ 0, 1 }")))


def test_mutant_state_proc_not_answered_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, 'if (strcmp(name, "ggml_backend_sycl_load_reserve_state_term") == 0) {',
              'if (strcmp(name, "ggml_backend_sycl_load_reserve_state_term_unused") == 0) {'))


def test_mutant_state_setter_storing_the_compute_term_fails():
    assert not claim_state_setter_stores_the_owners_term(
        _once(CACHE, "return unified_cache_set_planned_load_term(device_id, owner, planned_load_term::STATE, bytes);",
              "return unified_cache_set_planned_load_term(device_id, owner, planned_load_term::COMPUTE, bytes);"))


def test_mutant_state_publish_not_the_merge_fails():
    assert not claim_state_setter_stores_the_owners_term(
        _once(CACHE, "g_planned_state_term_bytes[device_id].store(g_planned_load_terms.owners.merged(device_id, "
                     "planned_load_term::STATE), std::memory_order_release);",
              "g_planned_state_term_bytes[device_id].store(0, std::memory_order_release);"))


def test_mutant_requirement_without_the_state_term_fails():
    assert not claim_requirement_folds_in_the_state_term(
        _once(CACHE, "*out = base + mmq_src1 + dequant_f16 + compute + state;", "*out = base + mmq_src1 + dequant_f16 + compute;"))


def test_mutant_requirement_state_unchecked_fails():
    assert not claim_requirement_folds_in_the_state_term(
        _once(CACHE, "state > SIZE_MAX - base - mmq_src1 - dequant_f16 - compute", "false"))


def test_mutant_ring_not_charged_the_state_fails():
    assert not claim_ring_leaves_the_state_room(
        _once(SYCL, "ring_kv_zone.runtime_pending_bytes += ggml_sycl::unified_cache_get_planned_state_term_bytes(ctx->device);",
              ";"))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
