"""Source contract for llama.cpp-8ecj: the single-device planner places dense weights, then KV, then routed experts.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, no -c): the load planned KV at n_ctx 512, the routed experts filled the card, and
the runtime context transaction, which can only demote, re-placed all the KV to the host tier, so the 12
full-attention layers ran their attention on the CPU. (That it reported 48 layers, not 12, is the over-count
test-sycl-kv-layer-owners-source pins.)

The fix this gate pins:
  - compute_placement_plan packs in three phases, in order: the dense pass charges weights only; the KV phase charges
    each device layer's KV and then holds the room the context the model opens with needs on top of it; only then
    are the routed experts packed into what is left. Under the GGML_SYCL_KV_PIN_DEVICE diagnostic the dense pass
    charges each layer's KV itself and the KV phase does nothing: no per-layer charge and no room, so pinned mode
    keeps the behaviour it had before the room;
  - the room is capped at what is left and counted in the plan's device bytes, and the runtime transaction's KV
    re-derivation drops it;
  - the expert pack runs a second first-fit budget with the room added back, and the room's cost is NET: the device
    bytes of routed experts that budget places minus the ones the real pack places (a larger first-fit budget can
    take a larger triplet and then skip smaller ones the real pack did place, so counting only the triplets that fit
    it and not the real budget over-states the cost); the plan keeps that figure, and the room's line reports both
    packs in MiB after the pack, as a WARN when the room cost experts, naming its limit: the room is for n_ctx_train;
  - the loader hands the backend that context (n_ctx_train, the probe measure's own n_ctx, since the caller's -c
    does not reach the load, fkpg).

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


PLAN_SIG = "placement_plan compute_placement_plan(const std::vector<placement_tensor_info> & tensor_inventory,"
LAYER_KV_SIG = "static void plan_single_device_layer_kv(placement_plan & plan,"
ROOM_SIG = "static kv_context_room hold_kv_context_room(placement_plan & plan,"
ROOM_LOG_SIG = "static void log_kv_context_room(const kv_context_room & room,"
REFRESH_SIG = "void refresh_kv_byte_totals()"
POPULATE_SIG = "static void llama_model_sycl_populate_inventory(ggml_sycl_tensor_inventory & inventory,"
INVENTORY_STRUCT = "struct ggml_sycl_tensor_inventory {"

DENSE_LOOP = "for (const auto & [layer_id, indices] : dense_layer_indices) {"
LAYER_KV_CALL = "plan_single_device_layer_kv(plan, kv_info, layer_has_attention, device_id, remaining);"
ROOM_CALL = "room = hold_kv_context_room(plan, kv_info, device_id, remaining);"
EXPERT_PASS = "auto moe_groups = build_moe_triplet_groups(plan, moe_indices, &hotness_source);"
WEIGHTS_ONLY = "on_device = weight_charge <= remaining; target = on_device ? device_id : -1; kv_on_device = false;"


# ---- (a) the three phases, in order -------------------------------------------------------------------------------


KV_PHASE = ("kv_context_room room; if (!planner_kv_pin_device_enabled()) { " + LAYER_KV_CALL + " " + ROOM_CALL
            + " }")


def claim_phases_are_dense_kv_experts(cache: str) -> bool:
    """Dense pass, then the KV phase (the per-layer KV charge and the context's room, both outside the pin
    diagnostic, so pinned mode holds no room), then experts: each exactly once."""
    b = body(norm(cache), PLAN_SIG)
    return (bool(b) and ordered(b, DENSE_LOOP, KV_PHASE, EXPERT_PASS)
            and b.count(LAYER_KV_CALL) == 1 and b.count(ROOM_CALL) == 1 and b.count(EXPERT_PASS) == 1)


def _dense_loop(cache: str) -> str:
    b = body(norm(cache), PLAN_SIG)
    at = b.find(DENSE_LOOP)
    return body(b[at:], DENSE_LOOP[:-2]) if at >= 0 else ""


def claim_dense_pass_charges_weights_only(cache: str) -> bool:
    """Outside the pin diagnostic the dense pass decides a layer on its weights alone and charges no KV: the KV phase
    does, so KV that does not fit never moves a dense layer to the host."""
    loop = _dense_loop(cache)
    return bool(loop) and WEIGHTS_ONLY in loop and "weight_charge + kv_cost" not in loop and "total_cost" not in loop


def claim_layer_kv_follows_the_dense_layer(cache: str) -> bool:
    """A layer's KV is charged on the device only when its dense weights are there and it fits; otherwise host."""
    b = body(norm(cache), LAYER_KV_SIG)
    return (bool(b) and ordered(
        b, "for (const auto & [layer_id, has_attention] : layer_has_attention)",
        "const bool kv_on_device = plan.get_layer_device(layer_id) == device_id && kv_cost <= remaining;",
        "remaining -= kv_cost;", "plan.kv_vram_bytes += kv_cost;", "plan.kv_host_bytes += kv_cost;",
        "plan.kv_device[layer_id] = kv_on_device ? device_id : -1;"))


def claim_room_is_the_context_extra_capped_and_counted(cache: str) -> bool:
    """The room sums the context's extra KV over the layers whose KV is on the device, takes no more than is left,
    and counts in the plan's device bytes."""
    b = body(norm(cache), ROOM_SIG)
    return (bool(b) and ordered(
        b, "if (layer_id < 0 || owner != device_id) { continue; }",
        "kv_info.kv_context_extra_bytes_for_layer(static_cast<uint32_t>(layer_id));",
        "room.held = std::min(room.wanted, remaining);", "remaining -= room.held;",
        "plan.kv_context_reserve_bytes = room.held;", "plan.vram_bytes += room.held;"))


ROOM_PACK = ("if (group.charge_bytes <= without_room) { without_room -= group.charge_bytes; "
             "room.expert_bytes_without += group.bytes; room.expert_groups_without++; }")
ROOM_REAL = "room.expert_bytes = stats.device_bytes; room.expert_groups = stats.device_groups;"
ROOM_NET = ("size_t displaced_bytes() const { return expert_bytes_without > expert_bytes ? "
            "expert_bytes_without - expert_bytes : 0; }")
ROOM_STRUCT = "struct kv_context_room {"


def claim_room_displacement_is_counted_and_logged(cache: str) -> bool:
    """The expert pack runs a second first-fit budget with the room added back and records the device bytes each pack
    placed; the room's cost is the difference (net, never the triplets that fit the second budget alone). The plan
    keeps that figure and the room's line reports it after the pack."""
    n = norm(cache)
    b = body(n, PLAN_SIG)
    at = n.find(ROOM_STRUCT)
    room = body(n[at:], ROOM_STRUCT[:-2]) if at >= 0 else ""
    return (bool(b) and ROOM_NET in room
            and ordered(
                b, ROOM_CALL, "size_t without_room = remaining + room.held;",
                "const bool on_device = group.charge_bytes <= remaining;", ROOM_PACK, ROOM_REAL,
                "log_moe_triplet_pack_stats(\"PLACEMENT-MOE\", stats, remaining);",
                "plan.kv_context_room_displaced_bytes = room.displaced_bytes();",
                "log_kv_context_room(room, kv_info, device_id);")
            and b.count("room.expert_bytes_without +=") == 1 and b.count("room.expert_bytes =") == 1)


def claim_room_line_is_visible_and_names_its_limit(cache: str) -> bool:
    """The line is in MiB, says the room is for n_ctx_train because the load does not see -c, and is a WARN whenever
    the room cost experts or could not hold all it wanted, so a default run shows it."""
    b = body(norm(cache), ROOM_LOG_SIG)
    return (bool(b) and "const bool warn = room.displaced_bytes() > 0 || room.held < room.wanted;" in b
            and ordered(b, "if (warn) { GGML_LOG_WARN(fmt,", "} else { GGML_LOG_INFO(fmt,")
            and "held %.1f MiB of %.1f MiB" in b and "it cost %.1f MiB of device-resident routed experts" in b
            and "with the room added back" in b
            and "The room is for n_ctx_train: the load does not see -c." in b)


def claim_extra_is_the_context_minus_the_charge(hpp: str) -> bool:
    """The context's extra KV comes from the same per-layer rule as the charge, at n_ctx_context."""
    n = norm(hpp)
    return ("size_t kv_bytes_for_layer(uint32_t il) const { return kv_bytes_for_layer_at(il, n_ctx); }" in n
            and ordered(n, "size_t kv_context_extra_bytes_for_layer(uint32_t il) const {",
                        "const size_t at_context = kv_bytes_for_layer_at(il, n_ctx_context);",
                        "const size_t charged = kv_bytes_for_layer(il);",
                        "return at_context > charged ? at_context - charged : 0;"))


def claim_runtime_rederivation_drops_the_room(hpp: str) -> bool:
    """Once the runtime transaction re-derives the KV totals for its real shape, the room is no longer counted."""
    b = body(norm(hpp), REFRESH_SIG)
    return bool(b) and ordered(b, "kv_context_reserve_bytes = 0;", "vram_bytes = weight_vram_bytes + kv_vram_bytes;")


# ---- (b) the context comes from libllama ----------------------------------------------------------------------------


def claim_loader_hands_the_opening_context(model: str, sycl: str, header: str) -> bool:
    """The inventory carries the probe measure's own n_ctx (n_ctx_train for an unspecified -c), right after
    kv_layer_count and before the indexer widths (the KV tail ggml-sycl.cpp pins with static_asserts), and the backend
    copies it into its KV inputs."""
    pop = body(norm(model), POPULATE_SIG)
    h   = norm(header)
    at  = h.find(INVENTORY_STRUCT)
    inv = body(h[at:], "struct ggml_sycl_tensor_inventory") if at >= 0 else ""
    want = ("inventory.n_ctx_context = llama_load_measure_n_ctx(llama_model_sycl_make_placement_envelope()"
            ".n_ctx, hparams.n_ctx_train);")
    return (bool(pop) and want in pop
            and inv.endswith("uint32_t kv_layer_count; uint32_t n_ctx_context; "
                             "const uint32_t * kv_idx_k_width_per_layer; }")
            and "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;" in norm(sycl))


def test_phases_are_dense_kv_experts():
    assert claim_phases_are_dense_kv_experts(CACHE)


def test_dense_pass_charges_weights_only():
    assert claim_dense_pass_charges_weights_only(CACHE)


def test_layer_kv_follows_the_dense_layer():
    assert claim_layer_kv_follows_the_dense_layer(CACHE)


def test_room_is_the_context_extra_capped_and_counted():
    assert claim_room_is_the_context_extra_capped_and_counted(CACHE)


def test_room_displacement_is_counted_and_logged():
    assert claim_room_displacement_is_counted_and_logged(CACHE)


def test_room_line_is_visible_and_names_its_limit():
    assert claim_room_line_is_visible_and_names_its_limit(CACHE)


def test_extra_is_the_context_minus_the_charge():
    assert claim_extra_is_the_context_minus_the_charge(CACHE_HPP)


def test_runtime_rederivation_drops_the_room():
    assert claim_runtime_rederivation_drops_the_room(CACHE_HPP)


def test_loader_hands_the_opening_context():
    assert claim_loader_hands_the_opening_context(MODEL, SYCL, SYCL_H)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def test_mutant_experts_first_fails():
    """Restores master's order: the experts are packed before the context's KV room is held."""
    n = norm(CACHE)
    mutant = _once(n, ROOM_CALL + " ", "")
    mutant = mutant.replace("add_dense_woq_alternates(plan, remaining, device_id);",
                            ROOM_CALL + " add_dense_woq_alternates(plan, remaining, device_id);", 1)
    assert ROOM_CALL in mutant
    assert not claim_phases_are_dense_kv_experts(mutant)


def test_mutant_layer_kv_after_the_pack_fails():
    """The per-layer KV charge moved after the expert pack: the experts would take the KV's bytes first."""
    pack_end = "plan.kv_context_room_displaced_bytes = room.displaced_bytes();"
    mutant = _once(CACHE, LAYER_KV_CALL + " ", "")
    assert mutant.count(pack_end) == 1
    mutant = mutant.replace(pack_end, LAYER_KV_CALL + " " + pack_end, 1)
    assert mutant.count(LAYER_KV_CALL) == 1
    assert not claim_phases_are_dense_kv_experts(mutant)


def test_mutant_room_held_under_the_pin_fails():
    """The room held whatever the pin says: pinned mode would lose experts to a room it never had."""
    mutant = _once(CACHE, KV_PHASE, "kv_context_room room; if (!planner_kv_pin_device_enabled()) { " + LAYER_KV_CALL
                   + " } " + ROOM_CALL)
    assert not claim_phases_are_dense_kv_experts(mutant)


def test_mutant_no_room_fails():
    assert not claim_phases_are_dense_kv_experts(_once(CACHE, ROOM_CALL, ""))


def test_mutant_kv_charged_with_the_dense_layer_fails():
    """Restores master's coupled charge: a layer whose weights and KV do not fit together goes to the host whole."""
    mutant = _once(CACHE, WEIGHTS_ONLY,
                   "on_device = weight_charge + kv_cost <= remaining; target = on_device ? device_id : -1; "
                   "kv_on_device = on_device;")
    assert not claim_dense_pass_charges_weights_only(mutant)


def test_mutant_kv_on_a_host_layer_fails():
    assert not claim_layer_kv_follows_the_dense_layer(
        _once(CACHE, "plan.get_layer_device(layer_id) == device_id && kv_cost <= remaining",
              "kv_cost <= remaining"))


def test_mutant_room_uncapped_fails():
    assert not claim_room_is_the_context_extra_capped_and_counted(
        _once(CACHE, "room.held = std::min(room.wanted, remaining);", "room.held = room.wanted;"))


def test_mutant_room_not_counted_fails():
    assert not claim_room_is_the_context_extra_capped_and_counted(
        _once(CACHE, "plan.vram_bytes += room.held;", ""))


def test_mutant_displacement_without_the_room_fails():
    """A counterfactual budget without the room added back would count nothing: every triplet it fits, fits."""
    assert not claim_room_displacement_is_counted_and_logged(
        _once(CACHE, "size_t without_room = remaining + room.held;", "size_t without_room = remaining;"))


def test_mutant_displacement_gross_fails():
    """The gross count: the second budget counts only the triplets the real pack put on the host."""
    assert not claim_room_displacement_is_counted_and_logged(
        _once(CACHE, "room.expert_bytes_without += group.bytes; room.expert_groups_without++;",
              "if (!on_device) { room.expert_bytes_without += group.bytes; room.expert_groups_without++; }"))


def test_mutant_displacement_not_net_fails():
    """The figure is the second budget's device bytes alone, with the real pack's not subtracted."""
    assert not claim_room_displacement_is_counted_and_logged(
        _once(CACHE, "expert_bytes_without > expert_bytes ? expert_bytes_without - expert_bytes : 0;",
              "expert_bytes_without;"))


def test_mutant_displacement_misses_the_real_pack_fails():
    assert not claim_room_displacement_is_counted_and_logged(
        _once(CACHE, "room.expert_bytes = stats.device_bytes;", "room.expert_bytes = 0;"))


def test_mutant_room_logged_before_the_pack_fails():
    n = norm(CACHE)
    call = "log_kv_context_room(room, kv_info, device_id);"
    mutant = _once(n, call, "").replace(ROOM_CALL, ROOM_CALL + " " + call, 1)
    assert not claim_room_displacement_is_counted_and_logged(mutant)


def test_mutant_room_line_only_info_fails():
    assert not claim_room_line_is_visible_and_names_its_limit(
        _once(CACHE, "const bool warn = room.displaced_bytes() > 0 || room.held < room.wanted;",
              "const bool warn = false;"))


def test_mutant_extra_at_the_planning_context_fails():
    assert not claim_extra_is_the_context_minus_the_charge(
        _once(CACHE_HPP, "kv_bytes_for_layer_at(il, n_ctx_context);", "kv_bytes_for_layer_at(il, n_ctx);"))


def test_mutant_room_survives_the_runtime_shape_fails():
    assert not claim_runtime_rederivation_drops_the_room(
        _once(CACHE_HPP, "kv_context_reserve_bytes = 0; vram_bytes = weight_vram_bytes",
              "vram_bytes = weight_vram_bytes"))


def test_mutant_loader_keeps_the_planning_context_fails():
    assert not claim_loader_hands_the_opening_context(
        _once(MODEL, "llama_load_measure_n_ctx(llama_model_sycl_make_placement_envelope().n_ctx, hparams.n_ctx_train)",
              "inventory.n_ctx"), SYCL, SYCL_H)


def test_mutant_field_appended_after_padding_fails():
    """A field inserted into the KV tail moves the fields the consumer reads at pinned offsets."""
    mutant = _once(SYCL_H, "uint32_t n_ctx_context; const uint32_t * kv_idx_k_width_per_layer; };",
                   "uint32_t n_ctx_context; size_t spare; const uint32_t * kv_idx_k_width_per_layer; };")
    assert not claim_loader_hands_the_opening_context(MODEL, SYCL, mutant)


def test_mutant_backend_drops_the_context_fails():
    assert not claim_loader_hands_the_opening_context(
        MODEL, _once(SYCL, "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;", ""), SYCL_H)
