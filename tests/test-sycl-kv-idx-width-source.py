"""Source contract for llama.cpp-5efe C4 (part b): the indexer width reaches the demotion's per-layer bytes.

The width travels per layer: master's ggml_sycl_tensor_inventory::kv_idx_k_width_per_layer (llama.cpp-8ecj) is the one
source, and placement_plan::kv_size_for_layer already charges it, so the demotion's demand is that one figure. The
runtime half is the pure host test `test-sycl-kv-demand-idx` (the accessor's arithmetic and the Qwen 6912.0 MB figure).
This gate pins the plumbing that test cannot see, end to end, so that no hop can be dropped and leave the printed
host-KV figure short by the indexer, and so that no second source of the width can come back:

1. `ggml_sycl_tensor_inventory` carries the per-layer `kv_idx_k_width_per_layer` array and no scalar `kv_idx_width`;
2. the llama layer fills the array in ONE place, `llama_model_sycl_populate_inventory`, from its parameter, and BOTH of
   its callers (early plan and late inventory) pass `kv_layer_arrays.idx_k_width.get()`, which the memory's layer
   owners fill; no helper derives the width from the architecture a second time;
3. `ggml_backend_sycl_stage_load_inventory_plan`'s apply step copies the array into the planner's
   `g_placement_kv_info.layer_idx_k_width`, and both plan builders mirror it into the persisted plan, beside the
   per-layer truth;
4. `placement_plan::kv_size_for_layer` adds `kv_idx_size_for_layer` on its per-layer branch, and
   `kv_demand_for_layer` returns `kv_size_for_layer(layer_id)` and reads no width itself (a second sum would count the
   indexer twice);
5. both demotion sites (the runtime demotion input and the residency re-fit input) read `kv_demand_for_layer`, and no
   `layer_kv_bytes` input reads `kv_main_size_for_layer` or `kv_idx_size_for_layer`, the figures that leave part of the
   layer out.

Plain file I/O, comment-stripped text, with planted-mutation witnesses.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces")


def function_body(text: str, signature_re: str) -> str:
    hits = list(re.finditer(signature_re, text))
    assert len(hits) == 1, f"{signature_re!r} matched {len(hits)} times"
    return balanced(text, text.index("{", hits[0].end()))


def read(rel: str) -> str:
    return strip_comments((ROOT / rel).read_text())


INVENTORY_HEADER = "ggml/include/ggml-sycl.h"
LLAMA_MODEL = "src/llama-model.cpp"
BACKEND = "ggml/src/ggml-sycl/ggml-sycl.cpp"
CACHE_HPP = "ggml/src/ggml-sycl/unified-cache.hpp"
CACHE_CPP = "ggml/src/ggml-sycl/unified-cache.cpp"

POPULATE = r"static\s+void\s+llama_model_sycl_populate_inventory\s*\("
EARLY = r"static\s+void\s+llama_model_sycl_compute_early_plan\s*\("
LATE = r"static\s+void\s+llama_model_sycl_set_late_inventory\s*\("
DEMAND = r"size_t\s+kv_demand_for_layer\s*\(\s*uint32_t\s+layer_id\s*\)\s*const"
SIZE = r"size_t\s+kv_size_for_layer\s*\(\s*uint32_t\s+layer_id\s*\)\s*const"


# ---- 1. the inventory field ----


def inventory_members() -> list:
    text = read(INVENTORY_HEADER)
    hit = re.search(r"struct\s+ggml_sycl_tensor_inventory\s*\{", text)
    assert hit, "ggml_sycl_tensor_inventory is missing"
    body = balanced(text, hit.end() - 1)
    return [m.strip() for m in body.split(";") if m.strip()]


def test_the_inventory_carries_the_per_layer_width_and_no_scalar():
    members = inventory_members()
    names = [re.split(r"\s+", m.replace("*", " ").strip())[-1] for m in members]
    assert "kv_idx_k_width_per_layer" in names, "ggml_sycl_tensor_inventory has no kv_idx_k_width_per_layer"
    decl = members[names.index("kv_idx_k_width_per_layer")]
    assert re.fullmatch(r"const\s+uint32_t\s*\*\s*kv_idx_k_width_per_layer", decl), f"not a per-layer array: {decl!r}"
    assert "kv_idx_width" not in names, "a scalar kv_idx_width is a second source of the indexer width"


# ---- 2. the llama layer ----


def test_no_helper_derives_the_width_from_the_architecture():
    text = read(LLAMA_MODEL)
    assert not re.search(r"\bllama_model_sycl_kv_idx_width\b", text), (
        "the width comes from the memory's layer owners (kv_layer_arrays.idx_k_width); an arch-keyed helper is a "
        "second source"
    )


def test_populate_inventory_sets_the_array_once_from_its_parameter():
    text = read(LLAMA_MODEL)
    body = function_body(text, POPULATE)
    sets = re.findall(r"inventory\s*\.\s*kv_idx_k_width_per_layer\s*=\s*([^;]+);", body)
    assert len(sets) == 1, f"populate_inventory sets kv_idx_k_width_per_layer {len(sets)} times"
    assert sets[0].strip() == "kv_idx_k_width_per_layer", f"the array is set from {sets[0].strip()!r}"
    assert not re.search(r"\binventory\s*\.\s*kv_idx_width\b", body), "populate_inventory sets a scalar width"
    sig = re.search(POPULATE + r"[^{]*\{", text, flags=re.DOTALL).group(0)
    assert re.search(r"const\s+uint32_t\s*\*\s*kv_idx_k_width_per_layer", sig), (
        "populate_inventory takes no kv_idx_k_width_per_layer parameter"
    )
    assert not re.search(r"uint32_t\s+kv_idx_width\b", sig), "populate_inventory still takes the scalar width"


def callers_pass_the_owner_array(text: str) -> list:
    bad = []
    for name, sig in (("early plan", EARLY), ("late inventory", LATE)):
        body = function_body(text, sig)
        call = re.search(r"llama_model_sycl_populate_inventory\s*\(([^;]*)\)\s*;", body, flags=re.DOTALL)
        if not call:
            bad.append(f"{name}: no populate_inventory call")
            continue
        if not re.search(r"kv_layer_arrays\s*\.\s*idx_k_width\s*\.\s*get\s*\(\s*\)", call.group(1)):
            bad.append(f"{name}: the call does not pass kv_layer_arrays.idx_k_width.get()")
    return bad


def test_both_callers_pass_the_owner_array():
    assert callers_pass_the_owner_array(read(LLAMA_MODEL)) == []


def test_mutation_witness_a_caller_passing_null_is_caught():
    code = read(LLAMA_MODEL)
    mutated = code.replace("kv_layer_arrays.idx_k_width.get()", "nullptr", 1)
    assert mutated != code, "mutation did not apply"
    assert len(callers_pass_the_owner_array(mutated)) == 1


# ---- 3. the planner ----


def test_the_apply_step_copies_the_array_into_the_planner():
    text = read(BACKEND)
    assert re.search(
        r"g_placement_kv_info\s*\.\s*layer_idx_k_width\s*\.\s*assign\s*\("
        r"\s*inventory\s*->\s*kv_idx_k_width_per_layer\s*,"
        r"\s*inventory\s*->\s*kv_idx_k_width_per_layer\s*\+\s*inventory\s*->\s*kv_layer_count\s*\)",
        text,
    ), "the apply step does not copy inventory->kv_idx_k_width_per_layer into g_placement_kv_info.layer_idx_k_width"
    assert not re.search(r"g_placement_kv_info\s*\.\s*kv_idx_width\b", text), "the planner still carries a scalar width"


def test_both_plan_builders_mirror_the_width_beside_the_per_layer_truth():
    text = read(CACHE_CPP)
    mirrors = re.findall(r"plan\s*\.\s*layer_idx_k_width\s*=\s*kv_info\s*\.\s*layer_idx_k_width\s*;", text)
    assert len(mirrors) == 2, (
        f"{len(mirrors)} plan builders mirror layer_idx_k_width, expected the 2 that mirror layer_kind"
    )
    kinds = re.findall(r"plan\s*\.\s*layer_kind\s*=\s*kv_info\s*\.\s*layer_kind\s*;", text)
    assert len(kinds) == 2, "the plan builders that mirror layer_kind changed"
    assert not re.search(r"\bplan\s*\.\s*kv_idx_width\b", text), "a plan builder mirrors a scalar width"


# ---- 4. the accessors ----


SIZE_WITHOUT_INDEXER = "kv_size_for_layer does not add kv_idx_size_for_layer: the budgets would leave the indexer out"
DEMAND_SECOND_SUM = "kv_demand_for_layer is not exactly kv_size_for_layer(layer_id): a second sum counts the indexer twice"


def accessor_problems(text: str) -> list:
    bad = []
    size = function_body(text, SIZE)
    if not re.search(r"\+\s*kv_idx_size_for_layer\s*\(\s*layer_id\s*\)", size):
        bad.append(SIZE_WITHOUT_INDEXER)
    demand = function_body(text, DEMAND)
    if not re.fullmatch(r"\s*return\s+kv_size_for_layer\s*\(\s*layer_id\s*\)\s*;\s*", demand):
        bad.append(DEMAND_SECOND_SUM)
    return bad


def test_the_demand_is_the_size_and_the_size_charges_the_indexer():
    assert accessor_problems(read(CACHE_HPP)) == []


def test_mutation_witness_a_second_indexer_sum_is_caught():
    code = read(CACHE_HPP)
    good = "return kv_size_for_layer(layer_id); }"
    assert code.count(good) == 1, "the delegating body moved"
    mutated = code.replace(good, "return kv_size_for_layer(layer_id) + kv_idx_size_for_layer(layer_id); }", 1)
    assert accessor_problems(mutated) == [DEMAND_SECOND_SUM]


def test_mutation_witness_a_size_without_the_indexer_is_caught():
    code = read(CACHE_HPP)
    size = function_body(code, SIZE)
    mutated_size = re.sub(r"\s*\+\s*kv_idx_size_for_layer\s*\(\s*layer_id\s*\)", "", size, count=1)
    assert mutated_size != size, "mutation did not apply"
    mutated = code.replace(size, mutated_size, 1)
    assert accessor_problems(mutated) == [SIZE_WITHOUT_INDEXER]


# ---- 5. the two demotion sites ----

SITE_RUNTIME = r"kv_demotion_in\s*\.\s*layer_kv_bytes\s*\[\s*l\s*\]\s*=\s*([^;]+);"
SITE_REFIT = r"\bin\s*\.\s*layer_kv_bytes\s*\[\s*l\s*\]\s*=\s*([^;]+);"
PARTIAL = re.compile(r"\bkv_(main|idx)_size_for_layer\b")


def test_both_demotion_sites_read_the_demand_accessor():
    text = read(BACKEND)
    for name, pat, plan in (("runtime demotion", SITE_RUNTIME, "plan"), ("residency re-fit", SITE_REFIT, "next_plan")):
        hits = re.findall(pat, text)
        assert len(hits) == 1, f"{name}: {len(hits)} layer_kv_bytes assignments, expected 1"
        assert re.fullmatch(
            plan + r"\s*\.\s*kv_demand_for_layer\s*\(\s*static_cast<uint32_t>\s*\(\s*l\s*\)\s*\)", hits[0].strip()
        ), f"{name}: layer_kv_bytes is read from {hits[0].strip()!r}, not {plan}.kv_demand_for_layer(l)"


def test_no_layer_kv_bytes_input_reads_a_partial_figure():
    text = read(BACKEND)
    for m in re.finditer(r"layer_kv_bytes\s*\[[^\]]*\]\s*=\s*([^;]+);", text):
        assert not PARTIAL.search(m.group(1)), f"a demotion input reads part of the layer: {m.group(0).strip()!r}"


# ---- the readers have a positive control ----


def test_planted_partial_reader_is_seen():
    planted = "kv_demotion_in.layer_kv_bytes[l] = plan.kv_main_size_for_layer(static_cast<uint32_t>(l));"
    hits = re.findall(SITE_RUNTIME, planted)
    assert len(hits) == 1 and PARTIAL.search(hits[0]), (
        "the site reader cannot see a planted kv_main_size_for_layer read"
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
