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

Above SKINNY_MAX_COLS columns the executor first offers the op to its oneDNN
route (ggml_sycl_mul_mat_bf16_onednn): a bf16 copy of the activations, rounded
to nearest even, in the planned dense src1 buffer, then one bf16 x bf16 -> f32
GEMM. The gate pins that the route allocates nothing and waits on nothing, that
it declines (and the tiled kernel runs) while a graph is recorded, on a short
buffer and on a primitive oneDNN cannot create, that the graph-entry walk counts
its copy with the executor's own questions, and that the planner and the zone
adapter plan that copy under the same gate.

Source-level on purpose: none of this is observable without a SYCL device. The
kernel arithmetic itself is checked by test-sycl-bf16-mul-mat (host device), the
route's by test-sycl-bf16-onednn-mul-mat-device (a GPU).
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
COMMON = ROOT / "ggml/src/ggml-sycl/common.hpp"
KERNEL = ROOT / "ggml/src/ggml-sycl/mul-mat-bf16.hpp"
CACHE = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"
PROBE_DIR = ROOT / "docs/backend/data/inkling2"
PROBES = ("t22-probe-b70-4a0b8b6d2.txt", "t22-probe-b50-4a0b8b6d2.txt")

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
    assert "ggml_sycl_resolve_no_materialize(" in executor, \
        "the executor must resolve the weight without materializing (ggml_sycl_resolve_no_materialize)"
    # ggml_sycl_resolve( can fall through to ggml_sycl_get_data_ptr_slow, which stages and allocates.
    # Source ready events are chained by graph_compute (ggml_sycl_should_add_dependency), so the
    # executor takes no event of its own either.
    for forbidden in ("ggml_sycl_resolve(", "unified_allocate", "unified_alloc(", "malloc_device", "malloc_host",
                      "ggml_sycl_pool_alloc", "ggml_sycl_bf16_materialize", "mem_copy", ".wait()", "->wait()",
                      "ready_event", "depends_on"):
        assert forbidden not in executor, f"the BF16 executor must not use {forbidden!r} (no second copy, no host wait)"
    assert "max_work_group_sizes[ctx.device]" in executor, \
        "the executor must pass the device's max work-group size so the split work-group fits it"
    assert "weight.extent < weight_bytes" in executor and "weight.extent != 0" not in executor, \
        "an unknown (0) extent must be rejected, not accepted"


def test_executor_launch_is_a_kernel_profiler_row():
    # A decoded Inkling 2 Mini token runs about 628 of these launches; without a row the kernel profiler
    # (GGML_SYCL_KERNEL_PROFILE) cannot say what they cost. The launch goes through the profiler's submit wrapper,
    # which times the event the kernel returns, under a name of its own and with the shape in the metadata.
    executor = body_after(_code(), "static void ggml_sycl_mul_mat_bf16_weight(")
    wrapped = executor.find("ggml_sycl_profile_submit(")
    launch = executor.find("ggml_sycl_bf16::mul_mat_bf16_f32(")
    assert 0 <= wrapped < launch, "the kernel must be launched inside ggml_sycl_profile_submit"
    assert '"mulmat.bf16_native"' in executor, "the profiler row is named mulmat.bf16_native"
    for key in ('"ncols="', '";nrows="', '";n_vecs="'):
        assert key in executor, f"the row's metadata must carry {key}"
    assert "ggml_sycl_kernel_profile_enabled()" in executor, \
        "the metadata string is built only when the profiler is on (628 launches per token)"


def test_kernel_takes_no_event_and_no_atomics():
    kernel = strip_comments(KERNEL.read_text())
    for forbidden in ("depends_on", "sycl::event * ready", "atomic_ref", "atomic_fetch", "malloc_device", "malloc_host"):
        assert forbidden not in kernel, f"mul-mat-bf16.hpp must not use {forbidden!r}"


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
    for placement in ("ggml_sycl_weight_residency_is_observable(", "ggml_sycl_weight_executes_on_host(",
                      "ggml_sycl_weight_is_planned_on_other_device("):
        assert placement in route, \
            f"the native route predicate must decline what the executor aborts on: missing {placement}"
    assert "mul_mat_shape_supported" in strip_comments(KERNEL.read_text()), \
        "mul-mat-bf16.hpp must define mul_mat_shape_supported"
    assert "ggml_sycl_bf16_weight_dispatch_available" in strip_comments(COMMON.read_text()), \
        "common.hpp must keep the pure weight predicate host tests call"


def nows(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _route() -> str:
    return body_after(_code(), "static bool ggml_sycl_mul_mat_bf16_onednn(ggml_backend_sycl_context & ctx,")


def test_executor_offers_the_gemm_route_before_the_native_launch():
    executor = body_after(_code(), "static void ggml_sycl_mul_mat_bf16_weight(")
    offer = nows(executor).find(
        "if (ggml_sycl_mul_mat_bf16_onednn_takes(src0, src1, &act_bytes) && ggml_sycl_mul_mat_bf16_onednn(ctx, src0, "
        "static_cast<const uint16_t *>(weight.ptr), src1_ddf, dst_ddf, n_vecs, ldx, ldy, act_bytes)) { return; }")
    assert offer >= 0, "the executor must offer the op to the oneDNN route (takes, then the route) and return on success"
    assert offer < nows(executor).find("ggml_sycl_profile_submit("), \
        "the route is offered before the native kernel is submitted, so a declined route leaves the tiled kernel to run"
    assert offer > nows(executor).find("f32_columns_uniform(dst, &ldy)"), \
        "the route is offered after the executor has checked the weight and the column strides it passes"


def test_route_takes_asks_the_column_count_first():
    takes = body_after(_code(), "static bool ggml_sycl_mul_mat_bf16_onednn_takes(")
    assert "if constexpr (!ggml_sycl_bf16_onednn_scratch_drawable())" in takes, \
        "a build without oneDNN never takes the route"
    cols = takes.find("n_vecs <= ggml_sycl_bf16::SKINNY_MAX_COLS")
    gate = takes.find("ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16)")
    assert 0 <= cols < gate, \
        "the column count is asked before the admission, so decode leaves on one comparison; the gate is the BF16 " \
        "oneDNN PP admission (GGML_SYCL_ONEDNN_PP=0 is the opt-out)"
    assert "zone_dequant_f16_region_bytes(src0->ne[0] * n_vecs, act_bytes)" in nows(takes), \
        "the copy's bytes are the dense dequant src1 buffer's own region arithmetic"


def test_route_allocates_nothing_and_waits_on_nothing():
    route = _route()
    for forbidden in ("ensure_buffer", "ggml_sycl_resolve(", "unified_allocate", "unified_alloc(", "malloc_device",
                      "malloc_host", "ggml_sycl_pool_alloc", "mem_copy", ".wait()", "->wait()", "ready_event",
                      "depends_on", "catch"):
        assert forbidden not in route, \
            f"the BF16 oneDNN route must not use {forbidden!r} (nothing allocated, no host wait, gemm() declines)"
    assert "ctx.dequant_f16_src1_scratch" in route, "the copy goes into the planned dense src1 buffer"
    recording = nows(route).find(
        "ggml_sycl_graph_recording_active() || stream->ext_oneapi_get_state() == "
        "sycl::ext::oneapi::experimental::queue_state::recording")
    short = route.find("buffer.capacity(ctx.device) < act_bytes")
    draw = route.find("backing_handle.resolve(ctx.device)")
    assert recording >= 0, "the route declines while a command graph is recorded (a new primitive would be created in it)"
    assert 0 <= short < draw, "a buffer shorter than the copy is a decline, asked before the buffer is drawn"
    assert "!act.on_device" in route, "a buffer that does not resolve on the device is a decline"


def test_route_rounds_the_copy_to_nearest_even_in_a_profiler_row():
    route = _route()
    submit = route.find("ggml_sycl_profile_submit(")
    rne = route.find("ggml_sycl_bf16::f32_to_bf16_rne(")
    gemm = route.find("DnnlGemmWrapper::gemm(")
    assert 0 <= submit < rne < gemm, \
        "the conversion is launched inside ggml_sycl_profile_submit, rounds with f32_to_bf16_rne, and precedes the GEMM"
    assert '"mulmat.bf16_onednn.act_to_bf16"' in route, "the conversion's profiler row is mulmat.bf16_onednn.act_to_bf16"
    kernel = strip_comments(KERNEL.read_text())
    assert "inline uint16_t f32_to_bf16_rne(float f)" in kernel, "mul-mat-bf16.hpp defines the device rounding"


def test_route_gemm_is_strict_bf16_and_declines_on_a_failed_creation():
    route = nows(_route())
    call = route.find("const std::optional<sycl::event> bf16_gemm = DnnlGemmWrapper::gemm(")
    assert call >= 0, "the route keeps gemm()'s optional, so a decline reaches the executor"
    args = route[call:route.find(";", call)]
    for arg in ("weight, bf16, 1, k,", "act_bf16, bf16, 1, k,", "dst_ddf, f32, stream, 1, 1,", '"bf16_weight"',
                "GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT", "GGML_SYCL_DNNL_CREATE_FAILURE_DECLINE"):
        assert arg in args, f"the route's gemm() call must carry {arg!r}"
    assert "return bf16_gemm.has_value();" in route, "the route returns whether gemm() ran"


def test_walk_counts_the_route_with_the_executors_questions():
    walk = body_after(_code(), "static bool ggml_sycl_dequant_f16_ensure_for_graph(")
    assert walk.lstrip().startswith("#    if GGML_SYCL_DNNL\n"), \
        "the walk is compiled wherever the route is (oneDNN), not only where the f16 arm is"
    flat = nows(walk)
    counted = flat.find(
        "if (ggml_sycl_bf16_onednn_walk_counted() && ggml_sycl_mul_mat_bf16_onednn_takes(src0, src1, &act_bytes) && "
        "ggml_sycl_bf16_weight_native_route_available(src0, src1, node, ctx.device) && "
        "act_bytes > demand[ctx.device].src1_bytes) { demand[ctx.device].src1_bytes = act_bytes; "
        "demand[ctx.device].src1_node = node; } continue;")
    assert counted >= 0, \
        "the walk counts a BF16 node's copy with the executor's predicate and the route's own takes, on this device"
    loop = flat.find("for (int i = 0; i < cgraph->n_nodes; i++)")
    arm = flat.find("# if GGML_SYCL_DEQUANT_F16_ARM", loop)
    assert 0 <= loop < counted < arm, "the BF16 branch sits in the node loop, outside the f16 arm's compile condition"


def test_planner_plans_the_wider_src1_row():
    planner = body_after(_code(), "static ggml_sycl_dense_scratch_planned ggml_sycl_plan_load_dense_scratch(")
    assert ("zone_dense_src1_planned_bytes_per_token(f16_drawable, maxima.dequant_f16_src1_bytes_per_token, "
            "ggml_sycl_bf16_onednn_scratch_drawable(), maxima.bf16_onednn_src1_bytes_per_token)") in nows(planner), \
        "the planned src1 row is the wider of the f16 arm's and the BF16 route's, each where its consumer is compiled in"


def test_adapter_marks_the_bf16_copy_under_the_routes_gate():
    adapter = nows(body_after(strip_comments(CACHE.read_text()),
                              "std::vector<zone_tensor_desc> unified_cache_adapt_zone_inventory("))
    mark = adapter.find("if (shaped && item.type == GGML_TYPE_BF16 && !is_expert && is_matrix && "
                        "ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16)) {")
    assert mark >= 0, "the adapter marks a dense BF16 matrix under the route's own gate"
    assert "desc.bf16_onednn_src1_bytes_per_token = src1_bytes;" in adapter[mark:], \
        "the mark is the copy's bytes per token"


def test_threshold_comment_cites_the_committed_probe():
    raw = SOURCE.read_text()
    takes_at = raw.find("static bool ggml_sycl_mul_mat_bf16_onednn_takes(")
    comment = raw[max(0, takes_at - 2500):takes_at]
    for probe in PROBES:
        assert probe in comment, f"the route's threshold comment must cite {probe}"
        assert (PROBE_DIR / probe).is_file(), f"the cited probe output must be committed at {PROBE_DIR / probe}"


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
