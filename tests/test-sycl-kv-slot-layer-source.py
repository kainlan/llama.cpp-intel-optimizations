"""Source contract: a KV cache slot lives where the weight layer that executes it lives.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (Inkling 2 Mini, inkling2_loop): 24 weight layers run as 48 KV cache slots, the core layers 6..17 three
times, so slots 24..47 run layers 6..17 again. The planner keyed KV by weight layer id: slots 24..47 had no layer of
their own, fell to the host tier, and their attention ran on the CPU, while the plan charged 24 slots of KV, not 48.

The fix this gate pins:
  - libllama names the weight layer of each slot (llama_model::layer_phys, the identity unless a looped model
    overrides it) and hands the map to the backend in the inventory, one entry per slot; the struct grew from 192 to
    208 bytes, appended, so no existing offset moved, and both sides pin the layout;
  - the backend takes the map once, at intake: only with one entry per slot and every entry a weight layer id below
    the slot count, otherwise it plans every slot as its own layer and says so at WARN; an identity map is kept empty,
    so a model whose slots are its layers plans and prints exactly as before. The backend computes no map of its own,
    and the residency hook still answers for the slot libllama asks about;
  - the planner keys KV by slot: each layer is charged for every slot it executes, a slot's KV goes where its layer's
    weights are (the single-device KV phase decides slot by slot, in slot order, and a host layer's slots are host
    slots), the multi-device planner and its blocks move and count a layer's slots with the layer, and the block
    executor candidate diagnostic moves every slot its block's layers execute;
  - a line that names each slot prints only for a looped model.

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


CACHE = read("ggml/src/ggml-sycl/unified-cache.cpp")
CACHE_HPP = read("ggml/src/ggml-sycl/unified-cache.hpp")
SYCL = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
SYCL_H = read("ggml/include/ggml-sycl.h")
MODEL = read("src/llama-model.cpp")
MODEL_H = read("src/llama-model.h")
MODELS_H = read("src/models/models.h")
KV_CACHE = read("src/llama-kv-cache.cpp")


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals are
    skipped while balancing). Empty when there is none."""
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


# ---- (a) libllama names each slot's layer ---------------------------------------------------------------------------


BASE_LAYER_PHYS = "virtual int layer_phys(int il) const { return il; }"
LOOP_STRUCT = "struct llama_model_inkling2_loop : public llama_model_inkling2 {"
LOOP_OVERRIDE = "int layer_phys(int il) const override;"
ARRAYS_SIG = "static llama_model_sycl_kv_layer_arrays llama_model_sycl_build_kv_layer_arrays(const llama_model & model,"
ARRAYS_FILL = "out.phys[il] = model.layer_phys((int) il);"
POPULATE_SIG = "static void llama_model_sycl_populate_inventory(ggml_sycl_tensor_inventory & inventory,"
POPULATE_SET = ("inventory.kv_layer_phys = kv_layer_phys; "
                "inventory.kv_layer_phys_count = kv_layer_phys != nullptr ? n_layer : 0;")
CALLER_PASSES = "kv_layer_arrays.idx_k_width.get(), kv_layer_arrays.phys.get(), total_size,"


def claim_libllama_names_each_slots_layer(model_h: str, models_h: str, model: str) -> bool:
    """llama_model names the identity, inkling2_loop overrides it, the KV arrays carry it per slot, and both inventory
    builders hand it to the backend with one entry per slot."""
    mh = norm(models_h)
    at = mh.find(LOOP_STRUCT)
    loop = body(mh[at:], LOOP_STRUCT[:-2]) if at >= 0 else ""
    m = norm(model)
    return (BASE_LAYER_PHYS in norm(model_h) and LOOP_OVERRIDE in loop
            and ARRAYS_FILL in body(m, ARRAYS_SIG) and POPULATE_SET in body(m, POPULATE_SIG)
            and m.count(CALLER_PASSES) == 2)


# ---- (b) the layout -------------------------------------------------------------------------------------------------


INVENTORY_STRUCT = "struct ggml_sycl_tensor_inventory {"
INV_TAIL = "uint32_t n_ubatch_context; const int32_t * kv_layer_phys; uint32_t kv_layer_phys_count; }"
HEADER_PINS = (
    "static_assert(sizeof(ggml_sycl_tensor_inventory) == 208,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, n_ubatch_context) == 188,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_layer_phys) == 192,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_layer_phys_count) == 200,",
)
BACKEND_PINS = HEADER_PINS + (
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_layer_count) == 168,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, n_ctx_context) == 172,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_idx_k_width_per_layer) == 176,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, n_ctx_context_is_default) == 184,",
)


def claim_map_is_appended_and_pinned(sycl_h: str, sycl: str) -> bool:
    """The map ends the struct, after the named micro-batch, and both sides of the dlopen boundary pin the new size and
    every offset that existed before it, so growing the struct moved nothing."""
    h = norm(sycl_h)
    at = h.find(INVENTORY_STRUCT)
    inv = body(h[at:], "struct ggml_sycl_tensor_inventory") if at >= 0 else ""
    s = norm(sycl)
    return inv.endswith(INV_TAIL) and all(p in h for p in HEADER_PINS) and all(p in s for p in BACKEND_PINS)


# ---- (c) the backend takes the map once, and computes none --------------------------------------------------------


FILL_SIG = ("static void populate_inventory_globals(ggml_backend_sycl_context * ctx, "
            "const ggml_sycl_tensor_inventory * inventory)")
INTAKE_CALL = "ggml_sycl::placement_kv_layer_phys_from_inventory(*inventory);"
INTAKE_COPY = "g_placement_kv_info.layer_phys = phys.layer_phys;"
REFUSED_WARN = '"[SYCL-PLAN] KV slot map ignored: kv_layer_phys_count=%u kv_layer_count=%u%s;'
REFUSED_COST = "\"as its own layer (a looped model's repeated slots go to the host tier)\\n\""
INTAKE_SIG = ("placement_kv_layer_phys_intake placement_kv_layer_phys_from_inventory("
              "const ggml_sycl_tensor_inventory & inventory)")
NO_MAP = "if (inventory.kv_layer_phys_count == 0) { return intake; }"
COUNT_CHECK = ("if (inventory.kv_layer_phys == nullptr || inventory.kv_layer_phys_count != inventory.kv_layer_count) { "
               "intake.refused = true;")
IDENTITY_EMPTY = "if (kv_layer_phys_is_identity(layer_phys)) { return intake; }"
RANGE_CHECK = ("if (layer_phys[slot] < 0 || static_cast<uint32_t>(layer_phys[slot]) >= inventory.kv_layer_count) { "
               "intake.refused = true;")
ACCEPT = "intake.layer_phys = layer_phys; return intake; }"
HOOK_SIG = "bool ggml_backend_sycl_kv_layer_on_device_from_dev(ggml_backend_dev_t dev, int32_t il)"
HOOK_ANSWER = "return snapshot->plan->get_kv_device(static_cast<int>(il)) >= 0;"
HOOK_ASK = "!hooks.kv_layer_on_device(dev, (int32_t) il)"


def claim_backend_takes_the_map_once(sycl: str, cache: str, kv_cache: str) -> bool:
    """The backend reads the map in one place, through the intake: no count is no map, and a count is refused unless the
    map is there with one entry per slot and every entry is a weight layer id below the slot count; the identity is kept
    empty; a refusal is said at WARN with its cost. Nothing else in the backend writes a map, and the residency hook
    answers for the slot libllama asks about."""
    s = norm(sycl)
    fill = body(s, FILL_SIG)
    intake = body(norm(cache), INTAKE_SIG)
    hook = body(s, HOOK_SIG)
    return (bool(fill) and ordered(fill, INTAKE_CALL, INTAKE_COPY, REFUSED_WARN, REFUSED_COST)
            and s.count(INTAKE_CALL) == 1 and len(re.findall(r"\.layer_phys = ", s)) == 1
            and bool(intake) and ordered(intake, NO_MAP, COUNT_CHECK, IDENTITY_EMPTY, RANGE_CHECK, ACCEPT)
            and bool(hook) and HOOK_ANSWER in hook and HOOK_ASK in norm(kv_cache))


# ---- (d) the planner keys KV by slot -------------------------------------------------------------------------------


PLAN_MIRROR = "plan.layer_phys = kv_info.layer_phys;"
SLOTS_SIG = "static void plan_layer_kv_slots(placement_plan & plan,"
SLOTS_LOOP = "for (const int slot : kv_slots_of_layers(kv_info.layer_phys, layer_id, layer_id)) {"
SLOTS_SET = "plan.kv_device[slot] = kv_on_target && slot_has_kv ? target : -1;"
SLOTS_CALL = "plan_layer_kv_slots(plan, kv_info, layer_id, "
LAYER_KV_SIG = "static void plan_single_device_layer_kv(placement_plan & plan,"
SLOT_ORDER = "slot_layers.emplace_back(static_cast<int>(slot), kv_info.layer_phys[slot]);"
PHYS_CHARGE = "kv_info.kv_bytes_for_phys_layer(layer_id)"
BLOCK_CHARGE = "block.kv_bytes += plan.kv_size_for_phys_layer(layer);"
PHYS_SUM_SIG = "size_t kv_bytes_for_phys_layer(int layer_id) const"
PHYS_SUM = ("for (const int slot : kv_slots_of_layers(layer_phys, layer_id, layer_id)) { "
            "bytes += kv_bytes_for_layer(static_cast<uint32_t>(slot)); }")
RANGE_SIG = "int layer_range_kv_device(int start_layer, int end_layer) const"
RANGE_LOOP = "for (const int slot : kv_slots_of_layers(layer_phys, start_layer, end_layer)) {"
CANDIDATE_LOOP = ("for (const int slot : ggml_sycl::kv_slots_of_layers(runtime_kv_plan.layer_phys, block.start_layer, "
                  "block.end_layer)) {")
LAYER_KEYED = (
    "plan.kv_device[layer_id]",
    "kv_info.kv_bytes_for_layer(static_cast<uint32_t>(layer_id))",
    "plan.kv_size_for_layer(static_cast<uint32_t>(layer))",
)


def claim_planner_keys_kv_by_slot(cache: str, cache_hpp: str, sycl: str) -> bool:
    """Both planners mirror the map; no KV decision or charge is keyed by a weight layer id; each layer's slots go where
    the layer goes (three dense passes) and are charged with it (every charge site); the KV phase walks the slots in
    slot order; a block counts and owns every slot its layers execute; the candidate diagnostic moves them all."""
    c = norm(cache)
    h = norm(cache_hpp)
    slots = body(c, SLOTS_SIG)
    return (c.count(PLAN_MIRROR) == 2 and not any(k in c for k in LAYER_KEYED)
            and bool(slots) and ordered(slots, SLOTS_LOOP, SLOTS_SET) and c.count(SLOTS_CALL) == 3
            and SLOT_ORDER in body(c, LAYER_KV_SIG)
            and c.count(PHYS_CHARGE) == 7 and BLOCK_CHARGE in c
            and PHYS_SUM in body(h, PHYS_SUM_SIG) and RANGE_LOOP in body(h, RANGE_SIG)
            and CANDIDATE_LOOP in norm(sycl))


# ---- (e) only a looped model prints anything new --------------------------------------------------------------------


SLOT_LINE = '"[PLACEMENT] layer %3d kv slots: %s\\n"'
SLOT_LINE_GATE = "if (!plan.layer_phys.empty()) { std::string slot_text;"
TARGET_IDENTITY = ("plan.layer_phys.empty() ? plan.get_kv_device(layer_id) : "
                   "plan.layer_range_kv_device(layer_id, layer_id);")
ACCEPT_LINE_GATE = "} else if (!phys.layer_phys.empty()) {"
ACCEPT_LINE = '"[SYCL-PLAN] KV slot map: %zu KV slots run weight layers 0..%d;'


def claim_identity_prints_as_before(cache: str, sycl: str) -> bool:
    """The per-slot line and the accepted-map line print only for a non-identity map, and a layer's KV target is read
    as before when every slot is its own layer."""
    c = norm(cache)
    fill = body(norm(sycl), FILL_SIG)
    return (c.count(SLOT_LINE) == 1 and ordered(c, SLOT_LINE_GATE, SLOT_LINE) and TARGET_IDENTITY in c
            and ordered(fill, ACCEPT_LINE_GATE, ACCEPT_LINE))


# ---- the tree as it is ----------------------------------------------------------------------------------------------


def test_libllama_names_each_slots_layer():
    assert claim_libllama_names_each_slots_layer(MODEL_H, MODELS_H, MODEL)


def test_map_is_appended_and_pinned():
    assert claim_map_is_appended_and_pinned(SYCL_H, SYCL)


def test_backend_takes_the_map_once():
    assert claim_backend_takes_the_map_once(SYCL, CACHE, KV_CACHE)


def test_planner_keys_kv_by_slot():
    assert claim_planner_keys_kv_by_slot(CACHE, CACHE_HPP, SYCL)


def test_identity_prints_as_before():
    assert claim_identity_prints_as_before(CACHE, SYCL)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def test_mutant_producer_sends_the_slot_fails():
    """libllama passes the slot where the layer belongs: the map is the identity and slots 24..47 stay host."""
    mutant = _once(MODEL, ARRAYS_FILL, "out.phys[il] = (int32_t) il;")
    assert not claim_libllama_names_each_slots_layer(MODEL_H, MODELS_H, mutant)


def test_mutant_loop_model_names_no_layer_fails():
    mutant = _once(MODELS_H, LOOP_OVERRIDE, "")
    assert not claim_libllama_names_each_slots_layer(MODEL_H, mutant, MODEL)


def test_mutant_one_builder_drops_the_map_fails():
    """The late inventory without the map: the late plan, the one the residency hook answers from, keys by layer."""
    n = norm(MODEL)
    at = n.find("static void llama_model_sycl_set_late_inventory(")
    k = n.find(CALLER_PASSES, at)
    assert at >= 0 and k >= 0
    mutant = n[:k] + "kv_layer_arrays.idx_k_width.get(), nullptr, total_size," + n[k + len(CALLER_PASSES):]
    assert not claim_libllama_names_each_slots_layer(MODEL_H, MODELS_H, mutant)


def test_mutant_map_inserted_before_the_micro_batch_fails():
    """Inserted instead of appended: n_ubatch_context would move from the offset the consumer reads."""
    h = _once(SYCL_H, "uint32_t n_ubatch_context; const int32_t * kv_layer_phys; uint32_t kv_layer_phys_count; }",
              "const int32_t * kv_layer_phys; uint32_t kv_layer_phys_count; uint32_t n_ubatch_context; }")
    assert not claim_map_is_appended_and_pinned(h, SYCL)


def test_mutant_backend_size_pin_left_at_192_fails():
    mutant = _once(SYCL, BACKEND_PINS[0], "static_assert(sizeof(ggml_sycl_tensor_inventory) == 192,")
    assert not claim_map_is_appended_and_pinned(SYCL_H, mutant)


def test_mutant_header_offset_unpinned_fails():
    assert not claim_map_is_appended_and_pinned(_once(SYCL_H, HEADER_PINS[2], "static_assert(true,"), SYCL)


def test_mutant_backend_copies_the_raw_map_fails():
    """The map copied as given, past the intake: a map of the wrong length is read past its end."""
    mutant = _once(SYCL, INTAKE_COPY, "g_placement_kv_info.layer_phys.assign(inventory->kv_layer_phys, "
                   "inventory->kv_layer_phys + inventory->kv_layer_phys_count);")
    assert not claim_backend_takes_the_map_once(mutant, CACHE, KV_CACHE)


def test_mutant_intake_skips_the_count_fails():
    assert not claim_backend_takes_the_map_once(SYCL, _once(CACHE, COUNT_CHECK, "if (false) { intake.refused = true;"),
                                                KV_CACHE)


def test_mutant_null_map_with_a_count_is_no_map_fails():
    """A count with a null map read as no map: the slots are planned as layers and nothing is said."""
    mutant = _once(CACHE, NO_MAP, "if (inventory.kv_layer_phys == nullptr || inventory.kv_layer_phys_count == 0) { "
                   "return intake; }")
    assert not claim_backend_takes_the_map_once(SYCL, mutant, KV_CACHE)


def test_mutant_intake_skips_the_range_fails():
    mutant = _once(CACHE, RANGE_CHECK, "if (false) { intake.refused = true;")
    assert not claim_backend_takes_the_map_once(SYCL, mutant, KV_CACHE)


def test_mutant_identity_kept_as_a_map_fails():
    """The identity kept as a map: an identity model would print the per-slot lines."""
    assert not claim_backend_takes_the_map_once(SYCL, _once(CACHE, IDENTITY_EMPTY, ""), KV_CACHE)


def test_mutant_refusal_is_silent_fails():
    mutant = _once(SYCL, REFUSED_WARN, '"[SYCL-PLAN] quiet:')
    assert not claim_backend_takes_the_map_once(mutant, CACHE, KV_CACHE)


def test_mutant_refusal_hides_its_cost_fails():
    """The WARN without what the refusal costs a looped model."""
    mutant = _once(SYCL, REFUSED_COST, '"as its own layer\\n"')
    assert not claim_backend_takes_the_map_once(mutant, CACHE, KV_CACHE)


def test_mutant_backend_computes_its_own_map_fails():
    """A second writer: the backend deriving the map itself instead of reading what libllama names."""
    mutant = _once(SYCL, INTAKE_COPY, INTAKE_COPY + " g_placement_kv_info.layer_phys = {};")
    assert not claim_backend_takes_the_map_once(mutant, CACHE, KV_CACHE)


def test_mutant_hook_answers_for_a_physical_layer_fails():
    mutant = _once(SYCL, HOOK_ANSWER, "return snapshot->plan->get_layer_device(static_cast<int>(il)) >= 0;")
    assert not claim_backend_takes_the_map_once(mutant, CACHE, KV_CACHE)


def test_mutant_kv_phase_keyed_by_layer_fails():
    """Restores the defect in the KV phase: the slot written is the layer, so slots 24..47 get no device."""
    mutant = _once(CACHE, "plan.kv_device[slot] = kv_on_device ? device_id : -1;",
                   "plan.kv_device[layer_id] = kv_on_device ? device_id : -1;")
    assert not claim_planner_keys_kv_by_slot(mutant, CACHE_HPP, SYCL)


def test_mutant_one_charge_keyed_by_layer_fails():
    """One charge site back on the layer's own slot: that planner charges 24 slots of KV, not 48."""
    n = norm(CACHE)
    k = n.find(PHYS_CHARGE)
    assert k >= 0
    mutant = n[:k] + "kv_info.kv_bytes_for_layer(static_cast<uint32_t>(layer_id))" + n[k + len(PHYS_CHARGE):]
    assert not claim_planner_keys_kv_by_slot(mutant, CACHE_HPP, SYCL)


def test_mutant_multi_device_mirror_dropped_fails():
    n = norm(CACHE)
    k = n.rfind(PLAN_MIRROR)
    mutant = n[:k] + n[k + len(PLAN_MIRROR):]
    assert not claim_planner_keys_kv_by_slot(mutant, CACHE_HPP, SYCL)


def test_mutant_block_counts_one_slot_per_layer_fails():
    mutant = _once(CACHE, BLOCK_CHARGE, "block.kv_bytes += plan.kv_size_for_layer(static_cast<uint32_t>(layer));")
    assert not claim_planner_keys_kv_by_slot(mutant, CACHE_HPP, SYCL)


def test_mutant_block_owner_from_layer_ids_fails():
    mutant = _once(CACHE_HPP, RANGE_LOOP, "for (int slot = start_layer; slot <= end_layer; ++slot) {")
    assert not claim_planner_keys_kv_by_slot(CACHE, mutant, SYCL)


def test_mutant_candidate_moves_layer_ids_fails():
    mutant = _once(SYCL, CANDIDATE_LOOP, "for (int slot = block.start_layer; slot <= block.end_layer; ++slot) {")
    assert not claim_planner_keys_kv_by_slot(CACHE, CACHE_HPP, mutant)


def test_mutant_slot_line_for_every_model_fails():
    """The per-slot line printed for an identity model too: the Mistral and GPT-OSS plan lines would change."""
    mutant = _once(CACHE, SLOT_LINE_GATE, "if (true) { std::string slot_text;")
    assert not claim_identity_prints_as_before(mutant, SYCL)


def test_mutant_accept_line_for_every_model_fails():
    mutant = _once(SYCL, ACCEPT_LINE_GATE, "} else {")
    assert not claim_identity_prints_as_before(CACHE, mutant)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
