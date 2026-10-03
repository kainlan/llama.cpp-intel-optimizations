"""Source contract for llama.cpp-5efe C4 (part b): the indexer width reaches the demotion's per-layer bytes.

The runtime half is the pure host test `test-sycl-kv-demand-idx` (the accessor's arithmetic, the Qwen 6912.0 MB figure,
the inventory's size and offsets). This gate pins the plumbing that test cannot see, end to end, so that no hop can be
dropped and leave the printed host-KV figure short by the indexer:

1. `ggml_sycl_tensor_inventory` declares `uint32_t kv_idx_width` as its last member, right after `kv_layer_count`
   (the dlopen boundary: the field lands in tail padding, never between two existing fields);
2. the llama layer fills it in ONE place, `llama_model_sycl_populate_inventory`, from a helper that answers
   `hparams.indexer_head_size` for the architecture whose memory is `llama_memory_hybrid_idx` (QWEN4EXP, without SWA)
   and 0 for every other model, and BOTH of its callers (early plan and late inventory) pass that helper's answer;
3. `ggml_backend_sycl_stage_inventory_plan`'s apply step copies it into the planner's `g_placement_kv_info`
   unconditionally (a model without per-layer arrays still carries an indexer width), and both plan builders mirror it
   into the persisted plan, beside the per-layer truth;
4. `placement_plan::kv_demand_for_layer` is `kv_size_for_layer` plus the indexer, and `kv_size_for_layer` itself, which
   sizes the device per-layer slabs, does not read the width;
5. both demotion sites (the runtime demotion input and the residency re-fit input) read `kv_demand_for_layer`, and
   neither reads `kv_size_for_layer` into `layer_kv_bytes`.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

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

HELPER = r"static\s+uint32_t\s+llama_model_sycl_kv_idx_width\s*\("
POPULATE = r"static\s+void\s+llama_model_sycl_populate_inventory\s*\("
EARLY = r"static\s+void\s+llama_model_sycl_compute_early_plan\s*\("
LATE = r"static\s+void\s+llama_model_sycl_set_late_inventory\s*\("


# ---- 1. the inventory field ----


def inventory_members() -> list:
    text = read(INVENTORY_HEADER)
    hit = re.search(r"struct\s+ggml_sycl_tensor_inventory\s*\{", text)
    assert hit, "ggml_sycl_tensor_inventory is missing"
    body = balanced(text, hit.end() - 1)
    return [m.strip() for m in body.split(";") if m.strip()]


def test_the_inventory_declares_kv_idx_width_last_after_kv_layer_count():
    members = inventory_members()
    names = [re.split(r"\s+", m.replace("*", " ").strip())[-1] for m in members]
    assert "kv_idx_width" in names, "ggml_sycl_tensor_inventory has no kv_idx_width"
    assert names[-1] == "kv_idx_width", f"kv_idx_width is not the last member (last is {names[-1]!r})"
    assert names[-2] == "kv_layer_count", "kv_idx_width must follow kv_layer_count (tail padding, no field moves)"
    decl = members[-1]
    assert re.match(r"uint32_t\s+kv_idx_width$", decl), f"kv_idx_width is not a plain uint32_t: {decl!r}"


# ---- 2. the llama layer ----


def test_the_helper_answers_the_indexer_width_only_for_the_idx_memory_architecture():
    body = function_body(read(LLAMA_MODEL), HELPER)
    assert "LLM_ARCH_QWEN4EXP" in body, "the helper does not name the architecture whose memory is hybrid_idx"
    assert re.search(
        r"swa_type\s*==\s*LLAMA_SWA_TYPE_NONE", body
    ), "the hybrid_idx branch is taken only without SWA (the iswa memory wins otherwise); the helper must match"
    assert re.search(r"hparams\s*\.\s*indexer_head_size", body), "the helper does not read hparams.indexer_head_size"
    assert re.search(
        r"indexer_head_size\s*>\s*0", body
    ), "a null filter_idx (no indexer tensors) builds no idx cache: the helper must answer 0 for width 0"
    assert re.search(r"return\s+0\s*;", body), "the helper has no 0 answer for the other models"


def test_populate_inventory_sets_the_field_once_from_its_parameter():
    body = function_body(read(LLAMA_MODEL), POPULATE)
    sets = re.findall(r"inventory\s*\.\s*kv_idx_width\s*=\s*([^;]+);", body)
    assert len(sets) == 1, f"populate_inventory sets kv_idx_width {len(sets)} times"
    assert re.fullmatch(r"kv_idx_width", sets[0].strip()), f"kv_idx_width is set from {sets[0].strip()!r}"
    sig = re.search(POPULATE + r"[^{]*\{", read(LLAMA_MODEL), flags=re.DOTALL).group(0)
    assert re.search(r"uint32_t\s+kv_idx_width", sig), "populate_inventory takes no kv_idx_width parameter"


def test_both_callers_pass_the_helpers_answer():
    text = read(LLAMA_MODEL)
    for name, sig in (("early plan", EARLY), ("late inventory", LATE)):
        body = function_body(text, sig)
        call = re.search(r"llama_model_sycl_populate_inventory\s*\(([^;]*)\)\s*;", body, flags=re.DOTALL)
        assert call, f"{name}: no populate_inventory call"
        assert re.search(
            r"llama_model_sycl_kv_idx_width\s*\(\s*ml\s*\.\s*get_arch\s*\(\s*\)\s*,\s*hparams\s*\)", call.group(1)
        ), f"{name}: the call does not pass llama_model_sycl_kv_idx_width(ml.get_arch(), hparams)"


# ---- 3. the planner ----


def test_the_apply_step_copies_the_width_unconditionally():
    text = read(BACKEND)
    hit = re.search(r"g_placement_kv_info\s*\.\s*kv_idx_width\s*=\s*inventory\s*->\s*kv_idx_width\s*;", text)
    assert hit, "the apply step does not copy inventory->kv_idx_width into g_placement_kv_info"
    # not inside the per-layer-arrays branch: a model without arrays still carries an indexer width
    arrays = re.search(r"if\s*\(\s*inventory\s*->\s*kv_layer_count\s*>\s*0[^{]*\{", text)
    assert arrays, "the per-layer arrays branch moved"
    assert not (
        arrays.end() < hit.start() < arrays.end() + len(balanced(text, arrays.end() - 1))
    ), "the width is copied only when the per-layer arrays are present"


def test_both_plan_builders_mirror_the_width_beside_the_per_layer_truth():
    text = read(CACHE_CPP)
    mirrors = re.findall(r"plan\s*\.\s*kv_idx_width\s*=\s*kv_info\s*\.\s*kv_idx_width\s*;", text)
    assert len(mirrors) == 2, f"{len(mirrors)} plan builders mirror kv_idx_width, expected the 2 that mirror layer_kind"
    kinds = re.findall(r"plan\s*\.\s*layer_kind\s*=\s*kv_info\s*\.\s*layer_kind\s*;", text)
    assert len(kinds) == 2, "the plan builders that mirror layer_kind changed"


# ---- 4. the accessor ----


def test_kv_demand_for_layer_adds_the_indexer_to_kv_size_for_layer():
    text = read(CACHE_HPP)
    demand = function_body(text, r"size_t\s+kv_demand_for_layer\s*\(\s*uint32_t\s+layer_id\s*\)\s*const")
    assert re.search(
        r"kv_size_for_layer\s*\(\s*layer_id\s*\)", demand
    ), "the accessor does not start from kv_size_for_layer"
    assert "kv_idx_width" in demand, "the accessor does not read the indexer width"
    assert (
        "planner_n_ctx" in demand
    ), "the indexer cache holds every cell of the context; the accessor must use planner_n_ctx"
    assert re.search(r"sizeof\s*\(\s*ggml_fp16_t\s*\)", demand), "the K element size is the planner's fp16"
    assert re.search(
        r"==\s*0", demand
    ), "a layer that carries no KV has no indexer either: the accessor needs a 0 early out"


def test_kv_size_for_layer_does_not_read_the_width():
    text = read(CACHE_HPP)
    size = function_body(text, r"size_t\s+kv_size_for_layer\s*\(\s*uint32_t\s+layer_id\s*\)\s*const")
    assert "kv_idx_width" not in size, "kv_size_for_layer sizes the device slabs; the indexer is not remapped into one"


# ---- 5. the two demotion sites ----

SITE_RUNTIME = r"kv_demotion_in\s*\.\s*layer_kv_bytes\s*\[\s*l\s*\]\s*=\s*([^;]+);"
SITE_REFIT = r"\bin\s*\.\s*layer_kv_bytes\s*\[\s*l\s*\]\s*=\s*([^;]+);"


def test_both_demotion_sites_read_the_demand_accessor():
    text = read(BACKEND)
    for name, pat, plan in (("runtime demotion", SITE_RUNTIME, "plan"), ("residency re-fit", SITE_REFIT, "next_plan")):
        hits = re.findall(pat, text)
        assert len(hits) == 1, f"{name}: {len(hits)} layer_kv_bytes assignments, expected 1"
        assert re.fullmatch(
            plan + r"\s*\.\s*kv_demand_for_layer\s*\(\s*static_cast<uint32_t>\s*\(\s*l\s*\)\s*\)", hits[0].strip()
        ), f"{name}: layer_kv_bytes is read from {hits[0].strip()!r}, not {plan}.kv_demand_for_layer(l)"


def test_no_layer_kv_bytes_input_reads_kv_size_for_layer():
    text = read(BACKEND)
    for m in re.finditer(r"layer_kv_bytes\s*\[[^\]]*\]\s*=\s*([^;]+);", text):
        assert "kv_size_for_layer" not in m.group(
            1
        ), f"a demotion input reads kv_size_for_layer: {m.group(0).strip()!r}"


# ---- the readers have a positive control ----


def test_planted_size_reader_is_seen():
    planted = "kv_demotion_in.layer_kv_bytes[l] = plan.kv_size_for_layer(static_cast<uint32_t>(l));"
    hits = re.findall(SITE_RUNTIME, planted)
    assert (
        len(hits) == 1 and "kv_size_for_layer" in hits[0]
    ), "the site reader cannot see a planted kv_size_for_layer read"
