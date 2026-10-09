"""Source contract for llama.cpp-p6i0: the load's planned compute term is a named RUNTIME consumer.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, -ub 512, no -c): the scheduler's compute buffer was planned by nobody. The weight
pack filled the shared zone, the RUNTIME zone held the PP MoE ring, and the buffer's 1543.1 MB chunk landed raw,
outside the arena, in the external headroom, while its 512 MB chunk fit no tier and the context was refused.

The fix this gate pins: the load measures the compute buffer at the probe placement and hands its per-chunk peaks to
ggml_backend_sycl_load_reserve_compute_term, which sizes a planned RUNTIME term from them at the RUNTIME TLSF's grain
(zone_compute_term_bytes, no headroom) and stores it beside the dense scratch terms. The RUNTIME zone requirement
folds it in, so the zone is sized for the buffer before the weight pack; the ring re-plan leaves it that room; the
landing line names it; and it is dropped where no model is live, so a next load never inherits it. Around it: the
early inventory plan stages the probe plan the PROBE measure installs, the admitted check's units export answers the
reserve's own sizing rule, and the measure override's install proc answers only the name of its kv_shape form.

The state term beside it: the context memory the measure placed on a device's plain buffer type (Qwen3.8's recurrent
state, 112.6 MiB on the B70) is allocated RUNTIME-first before the compute buffer, so the term alone did not hold the
buffer (the 512 MB chunk landed raw). ggml_backend_sycl_load_reserve_state_term sizes it at the same grain and stores
it as its own planned RUNTIME term, which the requirement and the ring count beside the compute term and the clear
drops with it.

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


RESERVE_SIG = "bool ggml_backend_sycl_load_reserve_compute_term(ggml_sycl_load_txn txn,"
SIZE_SIG = "static bool ggml_sycl_load_compute_term_size(const uint64_t * chunk_bytes, uint32_t n_chunks, size_t * out)"
UNITS_SIG = "bool ggml_backend_sycl_load_compute_term_bytes(const uint64_t * chunk_bytes, uint32_t n_chunks, uint64_t * out)"
SETTER_SIG = "bool unified_cache_set_planned_compute_term(int device_id, size_t bytes, bool other_model_live)"
REQ_SIG = "bool unified_cache_get_planned_runtime_zone_requirement(int device_id, size_t * out)"
TXN_SIG = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction("
LANDING_SIG = "static void ggml_sycl_log_compute_buffer_landing(int device, const std::string & name, size_t size,"
LOAD_SIG = "static void ggml_sycl_model_loading_effects(bool loading, bool outer)"
TEARDOWN_SIG = "static void ggml_sycl_release_model_slot_resources(ggml_sycl::lifecycle::ModelToken owner)"
ABORT_SIG = "static bool ggml_sycl_abort_owner_effects_noexcept(ggml_sycl::lifecycle::ModelToken owner,"
CLEAR = "ggml_sycl::unified_cache_clear_planned_load_terms();"
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
    stores exactly what the reservation's units answered: no headroom, no ledger."""
    b = body(norm(sycl), RESERVE_SIG)
    return (claim_the_units_are_the_grain_at_the_buffer_alignment(sycl) and bool(b) and ordered(
        b, "sycl_module_mutation_guard module_guard;", "if (!module_guard) { return false; }",
        "if (!ggml_sycl_load_txn_is_open(txn.id))", "else if (n_ctx == 0)",
        "else if (device < 0 || device >= ggml_sycl_info().total_gpu_count || device >= GGML_SYCL_MAX_DEVICES)",
        "if (cache == nullptr || !cache->arena_active())",
        "else if (!ggml_sycl_load_compute_term_size(chunk_bytes, n_chunks, &bytes))",
        "else if (!ggml_sycl::unified_cache_set_planned_compute_term(device, bytes, "
        "ggml_sycl::lifecycle::global_registry().live_mask() != 0))",
        "if (why != nullptr)", "return false;", "return true;")
        and b.count("unified_cache_set_planned_compute_term(") == 1
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


def claim_setter_merges_like_the_dense_terms(cache: str) -> bool:
    """Device-global term: while another model is live the larger of the stored and the new term stays."""
    b = body(norm(cache), SETTER_SIG)
    return (bool(b) and "if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) { return false; }" in b
            and "zone_dense_scratch_merge_input(g_planned_compute_term_bytes[device_id].load(std::memory_order_acquire), "
                "bytes, other_model_live)" in b)


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


# ---- (c) the term is dropped where no model is live ----------------------------------------------------------------


def claim_load_entry_clears_with_no_model_live(sycl: str) -> bool:
    b = body(norm(sycl), LOAD_SIG)
    return bool(b) and ordered(
        b, "if (outer) {", "const uint32_t live_mask = ggml_sycl::lifecycle::global_registry().live_mask();",
        "if (live_mask == 0) { " + CLEAR + " }")


def claim_teardown_clears_with_no_other_model_and_no_load(sycl: str) -> bool:
    b = body(norm(sycl), TEARDOWN_SIG)
    return bool(b) and ordered(
        b, "const size_t reclaimed = ggml_sycl::unified_cache_release_model_slot(slot);",
        "const uint32_t others_live = ggml_sycl::lifecycle::global_registry().live_mask() & "
        "(slot < 32 ? ~(1u << slot) : ~0u);",
        "if (others_live == 0 && !g_sycl_in_model_load.load(std::memory_order_acquire)) { " + CLEAR + " }")


def claim_abort_clears_with_no_model_live(sycl: str) -> bool:
    b = body(norm(sycl), ABORT_SIG)
    return bool(b) and ordered(
        b, "ggml_sycl_reset_model_load_scratch_state(true);",
        "if (ggml_sycl::lifecycle::global_registry().live_mask() == 0) { " + CLEAR + " }")


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


def test_the_setter_merges_like_the_dense_terms():
    assert claim_setter_merges_like_the_dense_terms(CACHE)


def test_the_requirement_folds_in_the_term():
    assert claim_requirement_folds_in_the_term(CACHE)


def test_the_ring_leaves_the_term_room():
    assert claim_ring_leaves_the_term_room(SYCL)


def test_the_landing_line_names_the_term():
    assert claim_landing_line_names_the_term(SYCL)


def test_the_term_is_dropped_where_no_model_is_live():
    assert claim_load_entry_clears_with_no_model_live(SYCL)
    assert claim_teardown_clears_with_no_other_model_and_no_load(SYCL)
    assert claim_abort_clears_with_no_model_live(SYCL)


# ---- mutants: every claim must fail against the thing it forbids --------------------------------------------------


def _once(raw: str, old: str, new: str) -> str:
    assert raw.count(old) >= 1, f"mutant anchor not found: {old!r}"
    return raw.replace(old, new, 1)


_TERM_SET = ("            } else if (!ggml_sycl::unified_cache_set_planned_compute_term(\n"
             "                           device, bytes, ggml_sycl::lifecycle::global_registry().live_mask() != 0)) {")


def test_mutant_reserve_with_headroom_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET, _TERM_SET.replace("device, bytes,", "device, bytes + bytes / 8,")))


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
        _once(SYCL, "    *out = static_cast<uint64_t>(bytes);\n",
              "    *out = static_cast<uint64_t>(bytes);\n    (void) ggml_sycl::unified_cache_set_planned_compute_term(0, bytes);\n"))


def test_mutant_units_proc_not_answered_fails():
    assert not claim_the_units_export_is_the_reserves_rule_and_stateless(
        _once(SYCL, 'if (strcmp(name, "ggml_backend_sycl_load_compute_term_bytes") == 0) {',
              'if (strcmp(name, "ggml_backend_sycl_load_compute_term_bytes_x") == 0) {'))


def test_mutant_reserve_on_a_closed_txn_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "if (!ggml_sycl_load_txn_is_open(txn.id)) {", "if (false) {"))


def test_mutant_reserve_without_the_arena_check_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "if (cache == nullptr || !cache->arena_active()) {", "if (cache == nullptr) {"))


def test_mutant_reserve_writing_the_ledger_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET,
              _TERM_SET + "\n                (void) ggml_sycl_load_record_compute_term(txn.id, device, bytes, n_ctx);"))


def test_mutant_reserve_replacing_a_live_models_term_fails():
    assert not claim_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _TERM_SET, _TERM_SET.replace("live_mask() != 0", "live_mask() != 0 && false")))


def test_mutant_setter_overwriting_fails():
    assert not claim_setter_merges_like_the_dense_terms(
        _once(CACHE, "zone_dense_scratch_merge_input(g_planned_compute_term_bytes[device_id].load(std::memory_order_acquire), bytes,\n"
                     "                                       other_model_live)",
              "bytes"))


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


def test_mutant_load_entry_clear_dropped_fails():
    assert not claim_load_entry_clears_with_no_model_live(
        _once(SYCL, "            if (live_mask == 0) {\n                ggml_sycl::unified_cache_clear_planned_load_terms();",
              "            if (live_mask == 0) {\n                ;"))


def test_mutant_load_entry_clears_beside_a_live_model_fails():
    assert not claim_load_entry_clears_with_no_model_live(
        _once(SYCL, "            if (live_mask == 0) {\n                ggml_sycl::unified_cache_clear_planned_load_terms();",
              "            if (true) {\n                ggml_sycl::unified_cache_clear_planned_load_terms();"))


def test_mutant_teardown_clears_during_a_load_fails():
    assert not claim_teardown_clears_with_no_other_model_and_no_load(
        _once(SYCL, "if (others_live == 0 && !g_sycl_in_model_load.load(std::memory_order_acquire)) {",
              "if (others_live == 0) {"))


def test_mutant_abort_clear_dropped_fails():
    assert not claim_abort_clears_with_no_model_live(
        _once(SYCL, "        if (ggml_sycl::lifecycle::global_registry().live_mask() == 0) {\n"
                    "            ggml_sycl::unified_cache_clear_planned_load_terms();",
              "        if (ggml_sycl::lifecycle::global_registry().live_mask() == 0) {\n            ;"))



def test_mutant_backend_answering_the_old_install_name_fails():
    arm = "    if (strcmp(name, " + INSTALL_KV + ") == 0) {"
    assert not claim_install_proc_answers_only_its_kv_name(
        _once(SYCL, arm, "    if (strcmp(name, " + INSTALL_OLD + ") == 0) {\n        return nullptr;\n    }\n" + arm),
        LLAMA_CTX)


def test_mutant_llama_asking_the_old_install_name_fails():
    assert not claim_install_proc_answers_only_its_kv_name(SYCL, _once(LLAMA_CTX, INSTALL_KV, INSTALL_OLD))



_STAGE_RAW = ("                ggml_sycl::lifecycle_stage_probe_placement_plan(txn, ggml_sycl::placement_plan(*candidate->plan),\n"
              "                                                                candidate->kv_info, candidate->model_n_layer);\n")


def test_mutant_probe_plan_not_staged_fails():
    assert not claim_early_plan_stages_the_probe_plan(_once(SYCL, _STAGE_RAW, ""))


def test_mutant_probe_plan_staged_by_the_late_arm_fails():
    assert not claim_early_plan_stages_the_probe_plan(
        _once(SYCL, "        if (early) {\n            // llama.cpp-p6i0: the early plan is the load's probe placement.",
              "        if (!early) {\n            // llama.cpp-p6i0: the early plan is the load's probe placement."))


def test_mutant_probe_plan_staged_for_another_load_fails():
    assert not claim_early_plan_stages_the_probe_plan(
        _once(SYCL, "            const uint64_t txn       = effect.owner.load.value;",
              "            const uint64_t txn       = effect.owner.load.value + 1;"))


# ---- (d) the state term: the context memory the RUNTIME zone takes before the compute buffer ----------------------

RESERVE_STATE_SIG = "bool ggml_backend_sycl_load_reserve_state_term(ggml_sycl_load_txn txn, int32_t device, uint64_t state_bytes)"
STATE_SETTER_SIG = "bool unified_cache_set_planned_state_term(int device_id, size_t bytes, bool other_model_live)"
CLEAR_SIG = "void unified_cache_clear_planned_load_terms()"


def claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(sycl: str) -> bool:
    """The state export admits the mutation, refuses a closed txn, a bad device and a device with no arena, sizes the
    one buffer at the reservation's grain and stores that as the state term: no headroom, no ledger, no compute
    term; and the backend answers its proc name."""
    b = body(norm(sycl), RESERVE_STATE_SIG)
    return (claim_the_units_are_the_grain_at_the_buffer_alignment(sycl) and bool(b) and ordered(
        b, "sycl_module_mutation_guard module_guard;", "if (!module_guard) { return false; }",
        "if (!ggml_sycl_load_txn_is_open(txn.id))",
        "else if (device < 0 || device >= ggml_sycl_info().total_gpu_count || device >= GGML_SYCL_MAX_DEVICES)",
        "if (cache == nullptr || !cache->arena_active())",
        "else if (!ggml_sycl_load_compute_term_size(&state_bytes, 1, &bytes))",
        "else if (!ggml_sycl::unified_cache_set_planned_state_term(device, bytes, "
        "ggml_sycl::lifecycle::global_registry().live_mask() != 0))",
        "if (why != nullptr)", "return false;", "return true;")
        and b.count("unified_cache_set_planned_state_term(") == 1
        and "unified_cache_set_planned_compute_term" not in b
        and "ledger" not in b and "ggml_sycl_load_record_compute_term" not in b
        and ('strcmp(name, "ggml_backend_sycl_load_reserve_state_term") == 0) { '
             "return (void *) ggml_backend_sycl_load_reserve_state_term; }") in norm(sycl))


def claim_state_setter_merges_like_the_compute_term(cache: str) -> bool:
    b = body(norm(cache), STATE_SETTER_SIG)
    return (bool(b) and "if (device_id < 0 || device_id >= GGML_SYCL_MAX_DEVICES) { return false; }" in b
            and "zone_dense_scratch_merge_input(g_planned_state_term_bytes[device_id].load(std::memory_order_acquire), "
                "bytes, other_model_live)" in b)


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


def claim_clear_drops_both_terms(cache: str) -> bool:
    b = body(norm(cache), CLEAR_SIG)
    return (bool(b) and "g_planned_compute_term_bytes[device_id].store(0, std::memory_order_release);" in b
            and "g_planned_state_term_bytes[device_id].store(0, std::memory_order_release);" in b
            and "unified_cache_clear_planned_compute_terms" not in cache)


def test_the_state_reservation_sizes_at_the_grain_and_writes_only_the_term():
    assert claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(SYCL)


def test_the_state_setter_merges_like_the_compute_term():
    assert claim_state_setter_merges_like_the_compute_term(CACHE)


def test_the_requirement_folds_in_the_state_term():
    assert claim_requirement_folds_in_the_state_term(CACHE)


def test_the_ring_leaves_the_state_room():
    assert claim_ring_leaves_the_state_room(SYCL)


def test_the_clear_drops_both_terms():
    assert claim_clear_drops_both_terms(CACHE)


_STATE_SET = ("            } else if (!ggml_sycl::unified_cache_set_planned_state_term(\n"
              "                           device, bytes, ggml_sycl::lifecycle::global_registry().live_mask() != 0)) {")


def test_mutant_state_reserve_with_headroom_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _STATE_SET, _STATE_SET.replace("device, bytes,", "device, bytes + bytes / 8,")))


def test_mutant_state_reserve_into_the_compute_term_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, _STATE_SET, _STATE_SET.replace("unified_cache_set_planned_state_term(", "unified_cache_set_planned_compute_term(")))


def test_mutant_state_reserve_without_the_grain_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, "            } else if (!ggml_sycl_load_compute_term_size(&state_bytes, 1, &bytes)) {",
              "            } else if (!(bytes = state_bytes, true)) {"))


def test_mutant_state_reserve_on_a_closed_txn_fails():
    b = body(SYCL, RESERVE_STATE_SIG)
    assert b and not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        SYCL.replace(b, b.replace("if (!ggml_sycl_load_txn_is_open(txn.id)) {", "if (false) {", 1), 1))


def test_mutant_state_proc_not_answered_fails():
    assert not claim_state_reserve_sizes_at_the_grain_and_writes_only_the_term(
        _once(SYCL, 'if (strcmp(name, "ggml_backend_sycl_load_reserve_state_term") == 0) {',
              'if (strcmp(name, "ggml_backend_sycl_load_reserve_state_term_unused") == 0) {'))


def test_mutant_state_setter_overwriting_fails():
    assert not claim_state_setter_merges_like_the_compute_term(
        _once(CACHE, "zone_dense_scratch_merge_input(g_planned_state_term_bytes[device_id].load(std::memory_order_acquire), bytes,\n"
                     "                                       other_model_live)",
              "bytes"))


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


def test_mutant_clear_keeping_the_state_fails():
    assert not claim_clear_drops_both_terms(
        _once(CACHE, "        g_planned_state_term_bytes[device_id].store(0, std::memory_order_release);\n", ""))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
