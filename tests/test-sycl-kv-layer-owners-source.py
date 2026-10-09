"""Source contract for llama.cpp-8ecj part 1: the SYCL inventory charges KV only for the layers the KV cache holds.

Host-only: reads sources, builds nothing, loads no model and touches no device.

The defect (B70, Qwen3.8, no -c): "KV overflow re-placed to host tier: 48 layer(s) demoted ... 24576 MB host KV".
Qwen3-Next's KV cache holds 12 layers, every 4th; the other 36 are recurrent and hold r/s state, not K/V. The SYCL
inventory built each layer's KV kind from hparams.has_kv(il), which is true for every layer of that model, so the
planner charged and the runtime transaction demoted 4x the KV the cache allocates.

The fix this gate pins:
  - the inventory's per-layer KV kind comes from llama_kv_layer_owners_default(): the layers the memory a default
    context builds holds K/V for, from llama_kv_layer_shapes(), the source the KV cache is sized from. A layer it
    does not hold is SHARED, with zero widths;
  - a memory kind the shapes do not model reports itself unmodelled, and only then does the inventory keep has_kv();
  - both inventory builders (the early plan and the late inventory) derive the arrays from the model;
  - a SHARED layer costs zero cells, so the planner charges it nothing and the demotion walk skips it.

test-layer-shapes checks the behaviour on built memories: the owners equal the layers each memory created K/V for,
in every config, including a memory where has_kv() over-counts. test-kv-runtime-demotion pins the numbers the
re-placement log reports for the Qwen3-Next shape: 12 layers and 6144 MiB, against 48 and 24576.

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


MODEL = read("src/llama-model.cpp")
SHAPES = read("src/llama-layer-shapes.cpp")
DEMOTION_HPP = read("ggml/src/ggml-sycl/kv-runtime-demotion.hpp")


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


ARRAYS_SIG = "static llama_model_sycl_kv_layer_arrays llama_model_sycl_build_kv_layer_arrays(const llama_model & model,"
OWNERS_SIG = "llama_kv_layer_owners llama_kv_layer_owners_default(const llama_model & model)"
CELLS_SIG = "inline size_t kv_layer_cells(uint8_t kind,"

OWNS_KV = "const bool owns_kv = owners.modelled ? il < owners.owns.size() && owners.owns[il] : hparams.has_kv(il);"
UNMODELLED = "if (!kv.unsupported.empty()) { return out; }"
BUILD_FROM_MODEL = "llama_model_sycl_build_kv_layer_arrays(model, n_layer);"
SHARED_CELLS = "if (kind == KV_CELLS_SHARED) { return 0; }"


def claim_kind_comes_from_the_default_memory(model: str, shapes: str) -> bool:
    """A layer owns KV when the default context's memory holds it; one it does not hold is SHARED with no width."""
    arrays = body(norm(model), ARRAYS_SIG)
    owners = body(norm(shapes), OWNERS_SIG)
    return (bool(arrays) and ordered(
        arrays, "const llama_kv_layer_owners owners = llama_kv_layer_owners_default(model);", OWNS_KV,
        "if (!owns_kv) {", "out.kind[il] = GGML_SYCL_KV_LAYER_SHARED;", "out.k_width[il] = 0;", "out.v_width[il] = 0;")
            and bool(owners) and ordered(
                owners, "const llama_kv_layer_shapes_result kv = llama_kv_layer_shapes(model, params_mem, cparams);",
                "out.owns[il] = kv.layers[il].has_kv;"))


def claim_unmodelled_kind_keeps_has_kv(shapes: str) -> bool:
    """A kind the shapes do not model returns before it is marked modelled, so the builder falls back to has_kv()."""
    owners = body(norm(shapes), OWNERS_SIG)
    return bool(owners) and ordered(owners, UNMODELLED, "out.modelled = true;", "out.owns.assign(")


def claim_both_builders_use_the_model(model: str) -> bool:
    """The early plan and the late inventory both build the arrays from the model, never from its hparams alone."""
    n = norm(model)
    return n.count(BUILD_FROM_MODEL) == 2 and "llama_model_sycl_build_kv_layer_arrays(hparams" not in n


def claim_shared_layer_costs_nothing(demotion_hpp: str) -> bool:
    """SHARED is the first branch of the one cell function: zero cells, so zero bytes for planner and demotion."""
    cells = body(norm(demotion_hpp), CELLS_SIG)
    return bool(cells) and cells.startswith("{ " + SHARED_CELLS)


def test_kind_comes_from_the_default_memory():
    assert claim_kind_comes_from_the_default_memory(MODEL, SHAPES)


def test_unmodelled_kind_keeps_has_kv():
    assert claim_unmodelled_kind_keeps_has_kv(SHAPES)


def test_both_builders_use_the_model():
    assert claim_both_builders_use_the_model(MODEL)


def test_shared_layer_costs_nothing():
    assert claim_shared_layer_costs_nothing(DEMOTION_HPP)


# ---- mutants: each must turn its claim red --------------------------------------------------------------------------


def _once(text: str, old: str, new: str) -> str:
    """`text` with the one occurrence of `old` (matched on normalized text) replaced by `new`."""
    n = norm(text)
    assert n.count(old) == 1, f"mutant anchor must match exactly once: {old!r} x{n.count(old)}"
    return n.replace(old, new, 1)


def test_mutant_kind_from_has_kv_fails():
    """Restores master's rule: every has_kv() layer is charged KV, recurrent layers included."""
    assert not claim_kind_comes_from_the_default_memory(
        _once(MODEL, OWNS_KV, "const bool owns_kv = hparams.has_kv(il);"), SHAPES)


def test_mutant_owners_ignore_the_shapes_fails():
    assert not claim_kind_comes_from_the_default_memory(
        MODEL, _once(SHAPES, "out.owns[il] = kv.layers[il].has_kv;", "out.owns[il] = true;"))


def test_mutant_unowned_layer_keeps_its_width_fails():
    assert not claim_kind_comes_from_the_default_memory(
        _once(MODEL, "out.kind[il] = GGML_SYCL_KV_LAYER_SHARED; out.k_width[il] = 0;",
              "out.kind[il] = GGML_SYCL_KV_LAYER_SHARED; out.k_width[il] = hparams.n_embd_k_gqa(il);"), SHAPES)


def test_mutant_unmodelled_kind_counted_as_modelled_fails():
    """An unmodelled kind published as modelled with no owners would charge every layer 0."""
    assert not claim_unmodelled_kind_keeps_has_kv(_once(SHAPES, UNMODELLED, ""))


def test_mutant_one_builder_from_hparams_fails():
    n = norm(MODEL)
    at = n.find(BUILD_FROM_MODEL)
    assert at >= 0
    mutated = n[:at] + "llama_model_sycl_build_kv_layer_arrays(hparams, n_layer);" + n[at + len(BUILD_FROM_MODEL):]
    assert not claim_both_builders_use_the_model(mutated)


def test_mutant_shared_layer_charged_fails():
    assert not claim_shared_layer_costs_nothing(
        _once(DEMOTION_HPP, SHARED_CELLS, "if (kind == KV_CELLS_SHARED) { return n_ctx; }"))
