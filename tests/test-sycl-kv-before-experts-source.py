"""Source contract for llama.cpp-8ecj: the single-device planner places dense weights, then KV, then routed experts.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, no -c): the load planned KV at n_ctx 512, the routed experts filled the card, and
the runtime context transaction, which can only demote, re-placed all KV to the host tier ("KV overflow re-placed to
host tier: 48 layer(s) demoted ... the device has 283.8 MB free for KV"), so the 12 full-attention layers ran their
attention on the CPU. 48, not 12: the SYCL inventory charged KV for the 36 recurrent layers the KV cache never holds.

The fix this gate pins:
  - compute_placement_plan packs in three phases, in order: the dense pass charges weights only; the KV phase charges
    each device layer's KV and then holds the room the context the model opens with needs on top of it; only then
    are the routed experts packed into what is left;
  - the room is capped at what is left and counted in the plan's device bytes, and the runtime transaction's KV
    re-derivation drops it;
  - the loader hands the backend that context (n_ctx_train, the probe measure's own n_ctx, since the caller's -c
    does not reach the load), and the per-layer KV kind comes from the memory the default context builds, so a
    layer the KV cache does not hold costs nothing.

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
SHAPES = read("src/llama-layer-shapes.cpp")


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
ROOM_SIG = "static void hold_kv_context_room(placement_plan & plan,"
REFRESH_SIG = "void refresh_kv_byte_totals()"
POPULATE_SIG = "static void llama_model_sycl_populate_inventory(ggml_sycl_tensor_inventory & inventory,"
ARRAYS_SIG = "static llama_model_sycl_kv_layer_arrays llama_model_sycl_build_kv_layer_arrays(const llama_model & model,"
OWNERS_SIG = "llama_kv_layer_owners llama_kv_layer_owners_default(const llama_model & model)"
INVENTORY_STRUCT = "struct ggml_sycl_tensor_inventory {"

DENSE_LOOP = "for (const auto & [layer_id, indices] : dense_layer_indices) {"
LAYER_KV_CALL = "plan_single_device_layer_kv(plan, kv_info, layer_has_attention, device_id, remaining);"
ROOM_CALL = "hold_kv_context_room(plan, kv_info, device_id, remaining);"
EXPERT_PASS = "auto moe_groups = build_moe_triplet_groups(plan, moe_indices, &hotness_source);"
WEIGHTS_ONLY = "on_device = weight_charge <= remaining; target = on_device ? device_id : -1; kv_on_device = false;"


# ---- (a) the three phases, in order -------------------------------------------------------------------------------


def claim_phases_are_dense_kv_experts(cache: str) -> bool:
    """Dense pass, then the per-layer KV charge (outside the pin diagnostic), then the context's room, then experts:
    each exactly once."""
    b = body(norm(cache), PLAN_SIG)
    return (bool(b) and ordered(b, DENSE_LOOP, "if (!planner_kv_pin_device_enabled()) { " + LAYER_KV_CALL + " }",
                                ROOM_CALL, EXPERT_PASS)
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
        "const size_t held = std::min(wanted, remaining);", "remaining -= held;",
        "plan.kv_context_reserve_bytes = held;", "plan.vram_bytes += held;"))


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


# ---- (b) the context and the per-layer kind come from libllama ------------------------------------------------------


def claim_loader_hands_the_opening_context(model: str, sycl: str, header: str) -> bool:
    """The inventory carries the probe measure's own n_ctx (n_ctx_train for an unspecified -c), in the struct's tail
    padding, and the backend copies it into its KV inputs."""
    pop = body(norm(model), POPULATE_SIG)
    h   = norm(header)
    at  = h.find(INVENTORY_STRUCT)
    inv = body(h[at:], "struct ggml_sycl_tensor_inventory") if at >= 0 else ""
    want = ("inventory.n_ctx_context = llama_load_measure_n_ctx(llama_model_sycl_make_placement_envelope()"
            ".n_ctx, hparams.n_ctx_train);")
    return (bool(pop) and want in pop
            and inv.endswith("uint32_t kv_layer_count; uint32_t n_ctx_context; }")
            and "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;" in norm(sycl))


def claim_kind_comes_from_the_default_memory(model: str, shapes: str) -> bool:
    """A layer owns KV when the default context's memory holds it; has_kv() only for a kind the shapes do not model."""
    arrays = body(norm(model), ARRAYS_SIG)
    owners = body(norm(shapes), OWNERS_SIG)
    return (bool(arrays) and ordered(
        arrays, "const llama_kv_layer_owners owners = llama_kv_layer_owners_default(model);",
        "const bool owns_kv = owners.modelled ? il < owners.owns.size() && owners.owns[il] : hparams.has_kv(il);",
        "if (!owns_kv) {", "out.kind[il] = GGML_SYCL_KV_LAYER_SHARED;")
            and bool(owners) and ordered(
                owners, "const llama_kv_layer_shapes_result kv = llama_kv_layer_shapes(model, params_mem, cparams);",
                "if (!kv.unsupported.empty()) { return out; }", "out.modelled = true;",
                "out.owns[il] = kv.layers[il].has_kv;"))


def test_phases_are_dense_kv_experts():
    assert claim_phases_are_dense_kv_experts(CACHE)


def test_dense_pass_charges_weights_only():
    assert claim_dense_pass_charges_weights_only(CACHE)


def test_layer_kv_follows_the_dense_layer():
    assert claim_layer_kv_follows_the_dense_layer(CACHE)


def test_room_is_the_context_extra_capped_and_counted():
    assert claim_room_is_the_context_extra_capped_and_counted(CACHE)


def test_extra_is_the_context_minus_the_charge():
    assert claim_extra_is_the_context_minus_the_charge(CACHE_HPP)


def test_runtime_rederivation_drops_the_room():
    assert claim_runtime_rederivation_drops_the_room(CACHE_HPP)


def test_loader_hands_the_opening_context():
    assert claim_loader_hands_the_opening_context(MODEL, SYCL, SYCL_H)


def test_kind_comes_from_the_default_memory():
    assert claim_kind_comes_from_the_default_memory(MODEL, SHAPES)


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
        _once(CACHE, "const size_t held = std::min(wanted, remaining);", "const size_t held = wanted;"))


def test_mutant_room_not_counted_fails():
    assert not claim_room_is_the_context_extra_capped_and_counted(
        _once(CACHE, "plan.vram_bytes += held;", ""))


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
    """A field after a new member moves the struct's tail, which this layout rule forbids."""
    mutant = _once(SYCL_H, "uint32_t n_ctx_context; };", "uint32_t n_ctx_context; size_t spare; };")
    assert not claim_loader_hands_the_opening_context(MODEL, SYCL, mutant)


def test_mutant_backend_drops_the_context_fails():
    assert not claim_loader_hands_the_opening_context(
        MODEL, _once(SYCL, "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;", ""), SYCL_H)


def test_mutant_kind_from_has_kv_fails():
    """Restores master's rule: every has_kv() layer is charged KV, recurrent layers included."""
    assert not claim_kind_comes_from_the_default_memory(
        _once(MODEL, "const bool owns_kv = owners.modelled ? il < owners.owns.size() && owners.owns[il] : "
              "hparams.has_kv(il);", "const bool owns_kv = hparams.has_kv(il);"), SHAPES)


def test_mutant_owners_ignore_the_shapes_fails():
    assert not claim_kind_comes_from_the_default_memory(
        MODEL, _once(SHAPES, "out.owns[il] = kv.layers[il].has_kv;", "out.owns[il] = true;"))
