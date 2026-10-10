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
    packs in MB after the pack, as a WARN when the room cost experts; with no requested context it logs one INFO
    line saying no room is held;
  - the loader hands the backend that context (llama.cpp-ak0p): the one the caller is about to create
    (llama_model_params::n_ctx_hint, which common sets from -c and llama-bench from its test's context), padded the
    way llama_context pads n_ctx for a single sequence, or 0 when no request reached the load, so a run without -c
    holds no room and places as master does. The placement envelope's n_ctx stays 0, so the load measures and the
    planning shape keep their inputs.

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
CONTEXT = read("src/llama-context.cpp")
LLAMA_H = read("include/llama.h")
COMMON = read("common/common.cpp")
BENCH = read("tools/llama-bench/llama-bench.cpp")


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
ENVELOPE_SIG = "static ggml_sycl_placement_envelope llama_model_sycl_make_placement_envelope()"
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


NO_ROOM = ('if (kv_info.n_ctx_context == 0) { GGML_LOG_INFO("[PLACEMENT] no KV context room on device %d: no '
           'requested n_ctx reached the load \" \"(planner n_ctx=%u); KV beyond it is placed at context '
           'creation\\n\", device_id, kv_info.n_ctx); return; }')
ROOM_EMPTY = "if (room.wanted == 0) { return; }"


def claim_room_line_is_visible_and_names_its_source(cache: str) -> bool:
    """With no requested context the line says no room is held (INFO, one per device); otherwise it is in MB, says
    the context is the requested one, and is a WARN whenever the room cost experts or could not hold all it wanted, so
    a default run shows it."""
    b = body(norm(cache), ROOM_LOG_SIG)
    return (bool(b) and ordered(b, NO_ROOM, ROOM_EMPTY)
            and "const bool warn = room.displaced_bytes() > 0 || room.held < room.wanted;" in b
            and "ggml_log_internal(warn ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO," in b
            and "held %.1f MB of %.1f MB for n_ctx_context=%u over \" \"planner n_ctx=%u" in b
            and "it cost %.1f MB of \" \"device-resident routed experts" in b
            and "with the room added back. n_ctx_context is the requested n_ctx.\\n\"," in b
            and "n_ctx_train" not in b)


def claim_extra_is_the_context_minus_the_charge(hpp: str) -> bool:
    """The context's extra KV comes from the same per-layer rule as the charge, at n_ctx_context."""
    n = norm(hpp)
    return ("size_t kv_bytes_for_layer(uint32_t il) const { return kv_bytes_for_layer_at(il, n_ctx); }" in n
            and ordered(n, "size_t kv_context_extra_bytes_for_layer(uint32_t il) const {",
                        "const size_t at_context = kv_bytes_for_layer_at(il, n_ctx_context);",
                        "const size_t charged = kv_bytes_for_layer(il);",
                        "return at_context > charged ? at_context - charged : 0;"))


KV_AT_SIG = "size_t kv_bytes_for_layer_at(uint32_t il, uint32_t ctx) const"
UNIFORM_FALLBACK = "return is_swa_layer(static_cast<int>(il)) ? kv_bytes_per_swa_layer(ctx) : kv_bytes_per_layer(ctx);"


def claim_uniform_fallback_has_one_formula(hpp: str) -> bool:
    """Without per-layer truth, a layer at `ctx` is the representative full or SWA layer at `ctx`: the room and the
    charge forward to kv_bytes_per_layer(ctx)/kv_bytes_per_swa_layer(ctx) and never restate their formula."""
    n = norm(hpp)
    at = body(n, KV_AT_SIG)
    full = body(n, "size_t kv_bytes_per_layer(uint32_t ctx) const")
    return (bool(at) and UNIFORM_FALLBACK in at and "n_embd_k_gqa" not in at
            and "size_t kv_bytes_per_layer() const { return kv_bytes_per_layer(n_ctx); }" in n
            and "size_t kv_bytes_per_swa_layer() const { return kv_bytes_per_swa_layer(n_ctx); }" in n
            and "n_embd_k_gqa, n_embd_v_gqa, ctx," in full)


def claim_runtime_rederivation_drops_the_room(hpp: str) -> bool:
    """Once the runtime transaction re-derives the KV totals for its real shape, the room is no longer counted."""
    b = body(norm(hpp), REFRESH_SIG)
    return bool(b) and ordered(b, "kv_context_reserve_bytes = 0;", "vram_bytes = weight_vram_bytes + kv_vram_bytes;")


# ---- (b) the context comes from libllama ----------------------------------------------------------------------------


CONTEXT_PAD_RE = re.compile(r"cparams\.n_ctx = GGML_PAD\(cparams\.n_ctx, (\d+)\);")
HINT_PICK_RE = re.compile(r"inventory\.n_ctx_context = n_ctx_hint != 0 \? GGML_PAD\(n_ctx_hint, (\d+)\) : 0;")
HINT_PASS = "max_pp_pipeline_weight_bytes, hparams, model.get_n_ctx_hint());"
INV_TAIL = "uint32_t kv_layer_count; uint32_t n_ctx_context; const uint32_t * kv_idx_k_width_per_layer; }"
COPY_CTX = "g_placement_kv_info.n_ctx_context = inventory->n_ctx_context;"
TAIL_ASSERTS = (
    "static_assert(sizeof(ggml_sycl_tensor_inventory) == 184,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, n_ctx_context) == 172,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_idx_k_width_per_layer) == 176,",
)


def claim_loader_hands_the_requested_context(model: str, model_h: str, sycl: str, header: str) -> bool:
    """The inventory carries the context the caller is about to create (llama_model_params::n_ctx_hint), or 0 when
    no request reached the load, so no room is held; both plan builders pass the model's hint. The context sits right
    after kv_layer_count and before the indexer widths (the KV tail ggml-sycl.cpp pins with static_asserts), and the
    backend copies it into its KV inputs."""
    m   = norm(model)
    pop = body(m, POPULATE_SIG)
    h   = norm(header)
    at  = h.find(INVENTORY_STRUCT)
    inv = body(h[at:], "struct ggml_sycl_tensor_inventory") if at >= 0 else ""
    s   = norm(sycl)
    return (bool(pop) and len(HINT_PICK_RE.findall(pop)) == 1 and pop.count("inventory.n_ctx_context =") == 1
            and "n_ctx_train" not in pop.split("inventory.n_ctx_context =")[1].split(";")[0]
            and "llama_load_measure_n_ctx" not in pop
            and m.count(HINT_PASS) == 2
            and "uint32_t get_n_ctx_hint() const { return params.n_ctx_hint; }" in norm(model_h)
            and inv.endswith(INV_TAIL)
            and COPY_CTX in s and all(a in s for a in TAIL_ASSERTS))


def claim_hint_is_padded_like_the_context(model: str, context: str) -> bool:
    """The room's context is padded with the constant llama_context pads n_ctx with (src/llama-context.cpp,
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256)), so the room covers a single-sequence context of the requested size.
    A non-unified multi-slot context can exceed it (each sequence's share is padded, SWA cells grow with the streams);
    the runtime transaction re-places that overflow."""
    pad = CONTEXT_PAD_RE.findall(norm(context))
    pick = HINT_PICK_RE.findall(body(norm(model), POPULATE_SIG))
    return len(pad) == 1 and len(pick) == 1 and pad[0] == pick[0]


def claim_envelope_keeps_no_context(model: str) -> bool:
    """The hint reaches the room only: the placement envelope still carries n_ctx 0, so the probe, admitted and late
    measures (llama_load_measure_n_ctx of the envelope's n_ctx) and the planning shape keep n_ctx_train and 512."""
    env = body(norm(model), ENVELOPE_SIG)
    return bool(env) and "envelope.n_ctx = 0;" in env and "n_ctx_hint" not in env


def claim_api_carries_the_hint(llama_h: str, model: str) -> bool:
    """llama_model_params ends with the hint, after the booleans, and it defaults to 0 (no request)."""
    return ("bool load_mtp; uint32_t n_ctx_hint; };" in norm(llama_h)
            and re.search(r"/\*\.load_mtp\s*=\*/\s*false,\s*/\*\.n_ctx_hint\s*=\*/\s*0,\s*\};", model) is not None)


COMMON_SIG = "struct llama_model_params common_model_params_to_llama(common_params & params)"
COMMON_HINT = "mparams.n_ctx_hint = params.n_ctx > 0 ? (uint32_t) params.n_ctx : 0;"
COMMON_FIT = "common_fit_params(params.model.path.c_str(), &mparams, &cparams,"
FIT_REFRESH = "mparams.n_ctx_hint = cparams.n_ctx;"
COMMON_FIT_REFRESH = "if (params.n_ctx > 0) { mparams.n_ctx_hint = cparams.n_ctx; }"
COMMON_LOAD = "llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);"


def claim_common_hands_the_requested_context(common: str) -> bool:
    """common passes -c as the hint (0, the training context, stays 0). After a fit, which may shrink a requested
    context, the hint is the context the fit chose, before the model loads; without -c the fit resolves the training
    context and the hint stays 0, so no room is held (owner ruling, llama.cpp-ak0p)."""
    n = norm(common)
    b = body(n, COMMON_SIG)
    return (bool(b) and COMMON_HINT in b and ordered(n, COMMON_FIT, COMMON_FIT_REFRESH, COMMON_LOAD)
            and n.count(FIT_REFRESH) == 1)


BENCH_CTX = "uint32_t n_ctx() const { return n_prompt + n_gen + n_depth; }"
BENCH_FIT = ("cparams.n_ctx = std::max(cparams.n_ctx, inst.n_ctx());",
             "common_fit_params(inst.model.c_str(), &mparams, &cparams,", FIT_REFRESH,
             "lmodel = llama_model_load_from_file(inst.model.c_str(), mparams);")


def claim_bench_hint_is_its_context(bench: str) -> bool:
    """llama-bench's hint and its context's n_ctx come from one formula (prompt + generated + depth), and its fit
    path refreshes the hint from the context it fitted, before the model loads."""
    n = norm(bench)
    mp = body(n, "llama_model_params to_llama_mparams() const")
    cp = body(n, "llama_context_params to_llama_cparams() const")
    return (BENCH_CTX in n and n.count("n_prompt + n_gen + n_depth") == 1
            and bool(mp) and "mparams.n_ctx_hint = n_ctx();" in mp
            and bool(cp) and "cparams.n_ctx = n_ctx();" in cp
            and ordered(n, *BENCH_FIT))


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


def test_room_line_is_visible_and_names_its_source():
    assert claim_room_line_is_visible_and_names_its_source(CACHE)


def test_extra_is_the_context_minus_the_charge():
    assert claim_extra_is_the_context_minus_the_charge(CACHE_HPP)


def test_uniform_fallback_has_one_formula():
    assert claim_uniform_fallback_has_one_formula(CACHE_HPP)


def test_runtime_rederivation_drops_the_room():
    assert claim_runtime_rederivation_drops_the_room(CACHE_HPP)


def test_loader_hands_the_requested_context():
    assert claim_loader_hands_the_requested_context(MODEL, MODEL_H, SYCL, SYCL_H)


def test_hint_is_padded_like_the_context():
    assert claim_hint_is_padded_like_the_context(MODEL, CONTEXT)


def test_envelope_keeps_no_context():
    assert claim_envelope_keeps_no_context(MODEL)


def test_api_carries_the_hint():
    assert claim_api_carries_the_hint(LLAMA_H, MODEL)


def test_common_hands_the_requested_context():
    assert claim_common_hands_the_requested_context(COMMON)


def test_bench_hint_is_its_context():
    assert claim_bench_hint_is_its_context(BENCH)


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
    assert not claim_room_line_is_visible_and_names_its_source(
        _once(CACHE, "const bool warn = room.displaced_bytes() > 0 || room.held < room.wanted;",
              "const bool warn = false;"))


def test_mutant_no_room_line_missing_fails():
    """A run without -c says nothing, so the gate cannot tell it held no room."""
    assert not claim_room_line_is_visible_and_names_its_source(_once(CACHE, NO_ROOM, ""))


def test_mutant_room_line_keeps_the_old_limit_fails():
    assert not claim_room_line_is_visible_and_names_its_source(
        _once(CACHE, "n_ctx_context is the requested n_ctx.", "The room is for n_ctx_train: the load does not see -c."))


def test_mutant_extra_at_the_planning_context_fails():
    assert not claim_extra_is_the_context_minus_the_charge(
        _once(CACHE_HPP, "kv_bytes_for_layer_at(il, n_ctx_context);", "kv_bytes_for_layer_at(il, n_ctx);"))


def test_mutant_room_survives_the_runtime_shape_fails():
    assert not claim_runtime_rederivation_drops_the_room(
        _once(CACHE_HPP, "kv_context_reserve_bytes = 0; vram_bytes = weight_vram_bytes",
              "vram_bytes = weight_vram_bytes"))


HINT_PICK = "inventory.n_ctx_context = n_ctx_hint != 0 ? GGML_PAD(n_ctx_hint, 256) : 0;"


def test_mutant_loader_ignores_the_hint_fails():
    """The room for n_ctx_train whatever the caller asked for: the defect this ticket fixes."""
    assert not claim_loader_hands_the_requested_context(
        _once(MODEL, HINT_PICK, "inventory.n_ctx_context = hparams.n_ctx_train;"), MODEL_H, SYCL, SYCL_H)


def test_mutant_loader_falls_back_to_n_ctx_train_fails():
    """A run without -c holds the room for n_ctx_train again instead of placing as master does."""
    assert not claim_loader_hands_the_requested_context(
        _once(MODEL, HINT_PICK, "inventory.n_ctx_context = n_ctx_hint != 0 ? GGML_PAD(n_ctx_hint, 256) : "
              "hparams.n_ctx_train;"), MODEL_H, SYCL, SYCL_H)


def test_mutant_loader_rides_the_measure_context_fails():
    """The room keyed to the probe measure's n_ctx again."""
    assert not claim_loader_hands_the_requested_context(
        _once(MODEL, HINT_PICK, "inventory.n_ctx_context = llama_load_measure_n_ctx("
              "llama_model_sycl_make_placement_envelope().n_ctx, hparams.n_ctx_train);"), MODEL_H, SYCL, SYCL_H)


def test_mutant_hint_padded_differently_fails():
    """A padding of its own: the room falls short of the context llama_context creates."""
    assert not claim_hint_is_padded_like_the_context(
        _once(MODEL, "GGML_PAD(n_ctx_hint, 256)", "GGML_PAD(n_ctx_hint, 255)"), CONTEXT)


def test_mutant_context_pads_differently_fails():
    """The comparison is real: llama_context changing its padding turns the claim red until the room follows."""
    assert not claim_hint_is_padded_like_the_context(
        MODEL, _once(CONTEXT, "cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);",
                     "cparams.n_ctx = GGML_PAD(cparams.n_ctx, 512);"))


def test_mutant_hint_unpadded_fails():
    assert not claim_hint_is_padded_like_the_context(
        _once(MODEL, HINT_PICK, "inventory.n_ctx_context = n_ctx_hint;"), CONTEXT)


def test_mutant_late_builder_drops_the_hint_fails():
    """One builder passes no hint: the late plan would undo the early plan's room."""
    n = norm(MODEL)
    at = n.find("static void llama_model_sycl_set_late_inventory(")
    assert at >= 0 and n.find(HINT_PASS, at) >= 0
    k = n.find(HINT_PASS, at)
    mutant = n[:k] + "max_pp_pipeline_weight_bytes, hparams, 0);" + n[k + len(HINT_PASS):]
    assert not claim_loader_hands_the_requested_context(mutant, MODEL_H, SYCL, SYCL_H)


def test_mutant_field_inserted_before_the_indexer_widths_fails():
    """A field inserted into the KV tail moves the fields the consumer reads at pinned offsets."""
    mutant = _once(SYCL_H, "uint32_t n_ctx_context; const uint32_t * kv_idx_k_width_per_layer; };",
                   "uint32_t n_ctx_context; size_t spare; const uint32_t * kv_idx_k_width_per_layer; };")
    assert not claim_loader_hands_the_requested_context(MODEL, MODEL_H, SYCL, mutant)


def test_mutant_field_appended_fails():
    """A field appended after the indexer widths: the struct no longer ends where the consumer's size pin says."""
    mutant = _once(SYCL_H, "const uint32_t * kv_idx_k_width_per_layer; };",
                   "const uint32_t * kv_idx_k_width_per_layer; uint8_t spare; };")
    assert not claim_loader_hands_the_requested_context(MODEL, MODEL_H, SYCL, mutant)


def test_mutant_backend_drops_the_context_fails():
    assert not claim_loader_hands_the_requested_context(MODEL, MODEL_H, _once(SYCL, COPY_CTX, ""), SYCL_H)


def test_mutant_envelope_carries_the_hint_fails():
    """The hint routed through envelope.n_ctx would move the load measures' compute term too."""
    assert not claim_envelope_keeps_no_context(_once(MODEL, "envelope.n_ctx = 0;", "envelope.n_ctx = n_ctx_hint;"))


def test_mutant_api_hint_not_last_fails():
    assert not claim_api_carries_the_hint(
        _once(LLAMA_H, "bool load_mtp; uint32_t n_ctx_hint; };", "uint32_t n_ctx_hint; bool load_mtp; };"), MODEL)


def test_mutant_common_drops_the_hint_fails():
    assert not claim_common_hands_the_requested_context(_once(COMMON, COMMON_HINT, ""))


def test_mutant_common_fit_keeps_the_unfitted_hint_fails():
    assert not claim_common_hands_the_requested_context(_once(COMMON, COMMON_FIT_REFRESH, ""))


def test_mutant_common_fit_refresh_unguarded_fails():
    """Restores the defect: without -c the fit's training context becomes the hint and reinstalls its room."""
    assert not claim_common_hands_the_requested_context(_once(COMMON, COMMON_FIT_REFRESH, FIT_REFRESH))


def test_mutant_bench_hint_drops_the_depth_fails():
    """A second formula for the bench's context, without the depth: the room would miss the depth's KV."""
    assert not claim_bench_hint_is_its_context(
        _once(BENCH, "mparams.n_ctx_hint = n_ctx();", "mparams.n_ctx_hint = n_prompt + n_gen;"))


def test_mutant_bench_drops_the_hint_fails():
    assert not claim_bench_hint_is_its_context(_once(BENCH, "mparams.n_ctx_hint = n_ctx();", ""))


def test_mutant_bench_fit_keeps_the_unfitted_hint_fails():
    assert not claim_bench_hint_is_its_context(_once(BENCH, FIT_REFRESH, ""))


def test_mutant_uniform_fallback_restated_fails():
    """The fallback's own copy of the full-attention formula, as before: two sources for one layer's KV."""
    assert not claim_uniform_fallback_has_one_formula(
        _once(CACHE_HPP, UNIFORM_FALLBACK,
              "return kv_layer_bytes_for_kind(GGML_SYCL_KV_LAYER_FULL, n_embd_k_gqa, n_embd_v_gqa, ctx, n_swa, "
              "n_ubatch, n_seq_max, kv_unified, swa_full);"))


def test_mutant_helper_ignores_ctx_fails():
    """The helper sized at the planning n_ctx whatever context it is asked about: the room would be 0."""
    n = norm(CACHE_HPP)
    full = body(n, "size_t kv_bytes_per_layer(uint32_t ctx) const")
    at_ctx = "n_embd_k_gqa, n_embd_v_gqa, ctx,"
    assert full.count(at_ctx) == 1
    mutant = n.replace(full, full.replace(at_ctx, "n_embd_k_gqa, n_embd_v_gqa, n_ctx,"), 1)
    assert not claim_uniform_fallback_has_one_formula(mutant)
