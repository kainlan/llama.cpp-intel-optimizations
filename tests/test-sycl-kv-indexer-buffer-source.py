"""Source contract for llama.cpp-8ecj: a memory that holds one layer in two KV buffers is budgeted and allocated
buffer by buffer.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8 IQ3_XXS, -c 10240, -ub 256): the attention K/V landed on the device, then the indexer key
cache of llama_memory_hybrid_idx asked for its 30 MiB and was refused with
  "[KV-TIER] device 0: device-planned KV 240.0 MB exceeds the 27.6 MB free for KV"
and no context. (At the default context the fit moves all 12 attention layers' KV to the host tier, so neither buffer
has a device layer and the backstop has nothing to refuse.)

The tiered allocator's backstop summed the plan's per-layer KV, which is the layer's KV and not one buffer's, for
every device layer of whichever buffer it was allocating, so the indexer buffer was charged the attention buffer's
bytes. The fix this gate pins: the backstop runs after the tier manager is configured for this buffer
and counts each member layer at the size this buffer allocates for it (kv_tier_manager::kv_layer_size), through the
one helper kv_buffer_device_bytes(). test-kv-runtime-demotion runs that helper on the Qwen3.8 two-buffer shape.

Nothing budgeted the indexer cache either: the inventory published the attention widths only, so the planner, the
runtime transaction and the largest-fitting -c hint all left it out, and at -c 10240 the fit kept all 12 attention
layers on the device with 27.6 MiB left for the 30 MiB indexer buffer. The fix this gate pins: the indexer key width travels as its own per-layer field
(ggml_sycl_tensor_inventory::kv_idx_k_width_per_layer, appended, sizeof 184 pinned at the consumer), every budget
(placement_kv_info::kv_bytes_for_layer, placement_plan::kv_size_for_layer, the -c hint) adds the indexer keys to
the layer's K/V, and the tier manager compares each buffer with its own cache's sum, never with the layers' total,
which no single buffer holds. test-sycl-kv-layer-sizing runs the two buffers through configure_from_plan.

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


SYCL_CPP = read("ggml/src/ggml-sycl/ggml-sycl.cpp")
DEMOTION_HPP = read("ggml/src/ggml-sycl/kv-runtime-demotion.hpp")
HEADER = read("ggml/include/ggml-sycl.h")
CACHE_HPP = read("ggml/src/ggml-sycl/unified-cache.hpp")
CACHE_CPP = read("ggml/src/ggml-sycl/unified-cache.cpp")
TIER_CPP = read("ggml/src/ggml-sycl/kv-tier-manager.cpp")


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


ALLOC_SIG = "static ggml_backend_buffer_t tiered_kv_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size)"
HELPER_SIG = "inline size_t kv_buffer_device_bytes("

CONFIGURE = "staged.configure_from_plan(device, *kv_plan, n_layers, kv_slice, &buffer_layer_mask);"
BUFFER_SIZE = "layer_bytes[l] = staged.kv_layer_size(l);"
COUNT = "ggml_sycl::kv_buffer_device_bytes(layer_owner, buffer_layer_mask, layer_bytes, device);"
REFUSE = "if (ggml_sycl::kv_admission_mismatch(planned_device_bytes, kv_vram_cap)) {"
HELPER_SUM = "if (l < member.size() && member[l] != 0 && layer_owner[l] == device) { total += layer_bytes[l]; }"


FIELD = "uint32_t kv_layer_count; const uint32_t * kv_idx_k_width_per_layer; };"
READ = ("g_placement_kv_info.layer_idx_k_width.assign(inventory->kv_idx_k_width_per_layer, "
        "inventory->kv_idx_k_width_per_layer + inventory->kv_layer_count);")
LAYOUT = (
    "static_assert(sizeof(ggml_sycl_tensor_inventory) == 184,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_layer_count) == 168,",
    "static_assert(offsetof(ggml_sycl_tensor_inventory, kv_idx_k_width_per_layer) == 176,",
)
KV_AT_SIG = "size_t kv_bytes_for_layer(uint32_t il) const"
KV_SIZE_SIG = "size_t kv_size_for_layer(uint32_t layer_id) const"
AT_ADDS = "+ kv_idx_bytes_for_layer_at(il, n_ctx);"
SIZE_ADDS = "+ kv_idx_size_for_layer(layer_id);"
PLAN_COPY = "plan.layer_idx_k_width = kv_info.layer_idx_k_width;"
HINT = ("const uint32_t widths = kv_info.layer_k_width[layer] + kv_info.layer_v_width[layer] + "
        "kv_info.idx_k_width(layer);")
CONFIGURE_SIG = "void kv_tier_manager::configure_from_plan("
MAIN_SUM = "main_sum += plan.kv_main_size_for_layer(l);"
IDX_SUM = "idx_sum += plan.kv_idx_size_for_layer(l);"
PICK = "const size_t truth_sum = is_idx_buffer ? idx_sum : main_sum;"
PER_LAYER = "per_layer_kv_bytes_[l] = is_idx_buffer ? plan.kv_idx_size_for_layer(l) : plan.kv_main_size_for_layer(l);"


def claim_inventory_carries_the_indexer_width(header: str, sycl_cpp: str) -> bool:
    """The field is appended after kv_layer_count, the backend copies it, and the consumer pins the layout."""
    n = norm(sycl_cpp)
    return FIELD in norm(header) and READ in n and all(a in n for a in LAYOUT)


def claim_budgets_add_the_indexer(cache_hpp: str, cache_cpp: str, sycl_cpp: str) -> bool:
    """Every budget adds the indexer keys to the layer's K/V: the planner's per-layer charge, the plan's
    per-layer size (the transaction, the host zone, the per-device charges), and the -c hint's bytes per cell."""
    n = norm(cache_hpp)
    at = body(n, KV_AT_SIG)
    size = body(n, KV_SIZE_SIG)
    return (bool(at) and AT_ADDS in at and bool(size) and SIZE_ADDS in size
            and norm(cache_cpp).count(PLAN_COPY) == 2 and HINT in norm(sycl_cpp))


def claim_tier_manager_sizes_each_buffer_from_its_cache(tier_cpp: str) -> bool:
    """A buffer is compared with the sum of the one cache it holds, so neither of a hybrid_idx memory's two buffers
    is measured against the total and refused into the slice with a WARN on every load."""
    cfg = body(norm(tier_cpp), CONFIGURE_SIG)
    return (bool(cfg) and ordered(cfg, MAIN_SUM, IDX_SUM, PICK, PER_LAYER)
            and "plan.kv_size_for_layer(" not in cfg)


def claim_backstop_counts_this_buffer(sycl_cpp: str) -> bool:
    """The backstop counts this buffer's own layer sizes, from the tier manager configured for it, and refuses only
    after that count; the plan's per-layer KV is no longer what it sums."""
    alloc = body(norm(sycl_cpp), ALLOC_SIG)
    return (bool(alloc) and ordered(alloc, CONFIGURE, BUFFER_SIZE, COUNT, REFUSE)
            and "planned_device_bytes += runtime_kv_plan.kv_size_for_layer(" not in alloc)


def claim_helper_sums_member_device_layers(demotion_hpp: str) -> bool:
    """The helper adds a layer's bytes only when the buffer holds it and the device owns it."""
    helper = body(norm(demotion_hpp), HELPER_SIG)
    return bool(helper) and HELPER_SUM in helper and helper.rstrip().endswith("return total; }")


def test_backstop_counts_this_buffer():
    assert claim_backstop_counts_this_buffer(SYCL_CPP)


def test_helper_sums_member_device_layers():
    assert claim_helper_sums_member_device_layers(DEMOTION_HPP)


def test_inventory_carries_the_indexer_width():
    assert claim_inventory_carries_the_indexer_width(HEADER, SYCL_CPP)


def test_budgets_add_the_indexer():
    assert claim_budgets_add_the_indexer(CACHE_HPP, CACHE_CPP, SYCL_CPP)


def test_tier_manager_sizes_each_buffer_from_its_cache():
    assert claim_tier_manager_sizes_each_buffer_from_its_cache(TIER_CPP)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def test_mutant_backstop_charges_the_plans_layer_kv_fails():
    """Restores the defect: every buffer charged the plan's per-layer KV (the 6144 MiB the indexer buffer read as)."""
    assert not claim_backstop_counts_this_buffer(
        _once(SYCL_CPP, BUFFER_SIZE, "layer_bytes[l] = runtime_kv_plan.kv_size_for_layer(l);"))


def test_mutant_backstop_before_configure_fails():
    """A count taken before the tier manager knows this buffer cannot be this buffer's."""
    moved = _once(SYCL_CPP, CONFIGURE, "")
    at = moved.find(REFUSE)
    assert at >= 0
    assert not claim_backstop_counts_this_buffer(moved[:at] + REFUSE + " } " + CONFIGURE + moved[at + len(REFUSE):])


def test_mutant_helper_ignores_membership_fails():
    assert not claim_helper_sums_member_device_layers(
        _once(DEMOTION_HPP, HELPER_SUM, "if (layer_owner[l] == device) { total += layer_bytes[l]; }"))


def test_mutant_backend_ignores_the_field_fails():
    assert not claim_inventory_carries_the_indexer_width(HEADER, _once(SYCL_CPP, READ, ""))


def test_mutant_layout_unpinned_fails():
    assert not claim_inventory_carries_the_indexer_width(HEADER, _once(SYCL_CPP, LAYOUT[0], "static_assert(true,"))


def test_mutant_planner_charge_without_the_indexer_fails():
    """Restores the defect: the per-layer charge leaves the indexer keys out."""
    assert not claim_budgets_add_the_indexer(_once(CACHE_HPP, AT_ADDS, ";"), CACHE_CPP, SYCL_CPP)


def test_mutant_plan_size_without_the_indexer_fails():
    assert not claim_budgets_add_the_indexer(_once(CACHE_HPP, SIZE_ADDS, ";"), CACHE_CPP, SYCL_CPP)


def test_mutant_one_plan_drops_the_indexer_fails():
    n = norm(CACHE_CPP)
    at = n.find(PLAN_COPY)
    assert at >= 0
    assert not claim_budgets_add_the_indexer(CACHE_HPP, n[:at] + n[at + len(PLAN_COPY):], SYCL_CPP)


def test_mutant_hint_without_the_indexer_fails():
    assert not claim_budgets_add_the_indexer(
        CACHE_HPP, CACHE_CPP,
        _once(SYCL_CPP, HINT, "const uint32_t widths = kv_info.layer_k_width[layer] + kv_info.layer_v_width[layer];"))


def test_mutant_tier_manager_measures_the_total_fails():
    """Folding the caches back together: each buffer compared with both caches' sum, the WARN on every load."""
    assert not claim_tier_manager_sizes_each_buffer_from_its_cache(
        _once(TIER_CPP, PICK, "const size_t truth_sum = main_sum + idx_sum;"))
