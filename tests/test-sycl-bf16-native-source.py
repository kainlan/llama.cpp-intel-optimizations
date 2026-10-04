#!/usr/bin/env python3
"""A BF16 dense weight must run from the BF16 bytes the planner placed, with no
second device copy (llama.cpp-9qjy).

Before: ggml_sycl_mul_mat materialized an F32 copy of every BF16 weight at its
first dispatch (unified_allocate, role WEIGHT, must_device), cached for the
model's lifetime. The planner placed the raw BF16 at 1x and never saw the F32
copy, so on Qwen3.8 IQ3 the planner filled the B70 and the first
materialization aborted the load of blk.14.hc_attn_down.weight. That is the
fork's "one fact, two sources" defect: the planner sized the weight as BF16
while the executor consumed F32.

After: the weight is consumed in its materialized BF16 layout by the native
executor in mul-mat-bf16.hpp. This gate pins, on the SOURCE, that

  * the lazy dispatch-time materialization (function, caches, per-key locks,
    owner-teardown hook, synthesized "<name>.bf16_materialized_f32" alias, host
    conversion) is gone;
  * ggml_sycl_mul_mat hands a BF16 weight to ggml_sycl_mul_mat_bf16_weight and
    does not re-enter itself with a retyped view;
  * that executor allocates nothing and waits on nothing -- it resolves the
    weight where the planner put it and submits one kernel;
  * supports_op admits a BF16 weight through the same predicate the executor
    asserts, ggml_sycl_bf16_weight_native_route_available.

Source-level on purpose: none of this is observable without a SYCL device. The
kernel arithmetic itself is checked by test-sycl-bf16-mul-mat (host device).
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
COMMON = ROOT / "ggml/src/ggml-sycl/common.hpp"
KERNEL = ROOT / "ggml/src/ggml-sycl/mul-mat-bf16.hpp"

# Names that only the removed lazy-materialization route used. Each must be absent
# from the code (comments excluded: the history is allowed to be discussed).
REMOVED = [
    "ggml_sycl_bf16_weight_materialize_f32",
    "ggml_sycl_bf16_weight_materialize_route_available",
    "ggml_sycl_bf16_materialize_cache_erase_for_owner",
    "ggml_sycl_bf16_materialize_key",
    "g_sycl_bf16_materialize_cache",
    "g_sycl_bf16_materialize_locks",
    "g_sycl_bf16_materialize_mutex",
    "bf16_materialized_f32",
    "ggml_bf16_to_fp32_row",
]


def strip_comments(text: str) -> str:
    """Remove // and /* */ comments, leaving string and char literals intact."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                i += 1
        elif two == "/*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n and text[i] != q:
                if text[i] == "\\":
                    out.append(text[i])
                    i += 1
                out.append(text[i])
                i += 1
            out.append(q)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def matching_brace(text: str, open_idx: int) -> int:
    assert text[open_idx] == "{"
    depth = 0
    i, n = open_idx, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            q = c
            i += 1
            while i < n and text[i] != q:
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def body_after(text: str, anchor: str) -> str:
    """Body of the block opened by the first '{' after the first occurrence of
    anchor that is followed by a definition (not a ';' declaration)."""
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx < 0:
            raise ValueError(f"anchor not found: {anchor!r}")
        brace = text.find("{", idx)
        semi = text.find(";", idx)
        if brace >= 0 and (semi < 0 or brace < semi):
            return text[brace + 1:matching_brace(text, brace)]
        start = idx + len(anchor)


_CACHE = {}


def _code() -> str:
    if "code" not in _CACHE:
        _CACHE["code"] = strip_comments(SOURCE.read_text())
    return _CACHE["code"]


def test_lazy_materialization_is_gone():
    code = _code()
    present = [name for name in REMOVED if name in code]
    assert not present, f"ggml-sycl.cpp still contains removed lazy-materialization names: {present}"


def test_mul_mat_hands_bf16_to_the_native_executor_without_recursing():
    mul_mat = body_after(
        _code(),
        "static void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx,\n"
        "                              const ggml_tensor *         src0,\n"
        "                              const ggml_tensor *         src1,\n"
        "                              ggml_tensor *               dst,\n"
        "                              const layout_mode *         forced_layout = nullptr)")
    m = re.search(r"if\s*\(\s*src0\s*&&\s*src0->type\s*==\s*GGML_TYPE_BF16\s*\)\s*\{", mul_mat)
    assert m is not None, "ggml_sycl_mul_mat has no BF16 weight branch"
    branch = mul_mat[m.end() - 1:]
    branch = branch[1:matching_brace(branch, 0)]
    assert "ggml_sycl_mul_mat_bf16_weight(ctx, src0, src1, dst)" in re.sub(r"\s+", " ", branch), \
        "the BF16 branch must call ggml_sycl_mul_mat_bf16_weight(ctx, src0, src1, dst)"
    assert "ggml_sycl_mul_mat(" not in branch, "the BF16 branch must not re-enter ggml_sycl_mul_mat with a retyped view"
    assert "src0_f32" not in branch, "the BF16 branch must not build a retyped F32 view of the weight"


def test_executor_allocates_nothing_and_waits_on_nothing():
    executor = body_after(_code(), "static void ggml_sycl_mul_mat_bf16_weight(")
    assert "ggml_sycl_bf16::mul_mat_bf16_f32(" in executor, "the executor must launch ggml_sycl_bf16::mul_mat_bf16_f32"
    assert "ggml_sycl_bf16_weight_native_route_available(" in executor, \
        "the executor must assert the same predicate supports_op admitted the op through"
    assert "on_device" in executor, \
        "the executor must refuse a weight that is not device-resident (placement decides the executor)"
    for forbidden in ("unified_allocate", "unified_alloc(", "malloc_device", "malloc_host", "ggml_sycl_pool_alloc",
                      "ggml_sycl_bf16_materialize", "mem_copy", ".wait()", "->wait()"):
        assert forbidden not in executor, f"the BF16 executor must not use {forbidden!r} (no second copy, no host wait)"


def test_supports_op_admits_bf16_through_the_executors_predicate():
    assert re.search(
        r"if\s*\(\s*ggml_sycl_bf16_weight_native_route_available\(\s*op->src\[0\]\s*,\s*op->src\[1\]\s*,\s*op\s*,"
        r"\s*device\s*\)\s*\)", _code()) is not None, \
        "supports_op must admit a BF16 weight via ggml_sycl_bf16_weight_native_route_available(op->src[0], op->src[1], op, device)"


def test_route_predicate_composes_weight_shape_and_buffer_class():
    route = body_after(
        _code(),
        "static bool ggml_sycl_bf16_weight_native_route_available(const ggml_tensor * src0,\n"
        "                                                         const ggml_tensor * src1,\n"
        "                                                         const ggml_tensor * dst,\n"
        "                                                         int                 device)")
    assert "ggml_sycl_bf16_weight_dispatch_available(" in route and "mul_mat_shape_supported(" in route, \
        "the native route predicate must compose the weight predicate and the shape contract"
    assert "ggml_backend_buffer_is_sycl_split(" in route and "ggml_backend_buffer_is_sycl_tp(" in route, \
        "the native route predicate must still decline split and TP buffers"
    assert "mul_mat_shape_supported" in strip_comments(KERNEL.read_text()), \
        "mul-mat-bf16.hpp must define mul_mat_shape_supported"
    assert "ggml_sycl_bf16_weight_dispatch_available" in strip_comments(COMMON.read_text()), \
        "common.hpp must keep the pure weight predicate host tests call"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except (AssertionError, ValueError) as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
