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
GEMM. The gate pins that the route allocates nothing and waits on nothing; that
it declines (and the tiled kernel runs) while a graph is recorded, before it
reads its buffer; that a short or unresolved buffer is a loud decline (one WARN
per device slot, an under-estimate on the buffer's cohort); that it asks oneDNN
for the primitive before it converts anything and runs the GEMM on the
conversion's event; that the graph-entry walk counts its copy with the
executor's own questions, apart from the f16 arm's, and caps the buffer rather
than refuse a graph over the route's share; and that the planner and the zone
adapter plan that copy under the same gate. Every route check is also run
against in-memory mutants of the source, each of which it must reject.

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


def body_span(text: str, anchor: str) -> tuple:
    """(start, end) of the body of the block opened by the first '{' after the
    first occurrence of anchor that is followed by a definition (not a ';'
    declaration)."""
    start = 0
    while True:
        idx = text.find(anchor, start)
        if idx < 0:
            raise ValueError(f"anchor not found: {anchor!r}")
        brace = text.find("{", idx)
        semi = text.find(";", idx)
        if brace >= 0 and (semi < 0 or brace < semi):
            return brace + 1, matching_brace(text, brace)
        start = idx + len(anchor)


def body_after(text: str, anchor: str) -> str:
    """Body of the block opened by the first '{' after the first occurrence of
    anchor that is followed by a definition (not a ';' declaration)."""
    start, end = body_span(text, anchor)
    return text[start:end]


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


def _raw() -> str:
    if "raw" not in _CACHE:
        _CACHE["raw"] = SOURCE.read_text()
    return _CACHE["raw"]


def _cache_code() -> str:
    if "cache" not in _CACHE:
        _CACHE["cache"] = strip_comments(CACHE.read_text())
    return _CACHE["cache"]


def _kernel() -> str:
    if "kernel" not in _CACHE:
        _CACHE["kernel"] = strip_comments(KERNEL.read_text())
    return _CACHE["kernel"]


ROUTE = "static bool ggml_sycl_mul_mat_bf16_onednn(ggml_backend_sycl_context & ctx,"
EXECUTOR = "static void ggml_sycl_mul_mat_bf16_weight("
TAKES = "static bool ggml_sycl_mul_mat_bf16_onednn_takes("
WALK = "static bool ggml_sycl_dequant_f16_ensure_for_graph("
PLANNER = "static ggml_sycl_dense_scratch_planned ggml_sycl_plan_load_dense_scratch("
ADAPTER = "std::vector<zone_tensor_desc> unified_cache_adapt_zone_inventory("

RECORDING = ("ggml_sycl_graph_recording_active() || stream->ext_oneapi_get_state() == "
             "sycl::ext::oneapi::experimental::queue_state::recording;")
RECORDING_DECLINE = "if (recording) { return false; }"
QUERY = "if (!bf16_gemm(true, {})) { return false; }"
CONVERSION = "const sycl::event converted = ggml_sycl_profile_submit("
RUN = "return bf16_gemm(false, { converted }).has_value();"
GEMM_LAMBDA = ("const auto bf16_gemm = [&](bool query_only, const std::vector<sycl::event> & deps) { "
               "return DnnlGemmWrapper::gemm(")


def _route() -> str:
    return body_after(_code(), ROUTE)


def _branch(flat: str, head: str) -> str:
    """The body of the one block `head` opens in the flattened text `flat` (head ends with '{')."""
    at = flat.find(head)
    assert at >= 0, f"missing: {head!r}"
    assert flat.find(head, at + 1) < 0, f"more than one: {head!r}"
    brace = at + len(head) - 1
    return flat[brace + 1:matching_brace(flat, brace)]


def test_executor_offers_the_gemm_route_before_the_native_launch():
    executor = body_after(_code(), EXECUTOR)
    offer = nows(executor).find(
        "if (ggml_sycl_mul_mat_bf16_onednn_takes(src0, src1, &act_bytes) && ggml_sycl_mul_mat_bf16_onednn(ctx, src0, "
        "static_cast<const uint16_t *>(weight.ptr), src1_ddf, dst_ddf, n_vecs, ldx, ldy, act_bytes)) { return; }")
    assert offer >= 0, "the executor must offer the op to the oneDNN route (takes, then the route) and return on success"
    assert offer < nows(executor).find("ggml_sycl_profile_submit("), \
        "the route is offered before the native kernel is submitted, so a declined route leaves the tiled kernel to run"
    assert offer > nows(executor).find("f32_columns_uniform(dst, &ldy)"), \
        "the route is offered after the executor has checked the weight and the column strides it passes"


def test_route_takes_asks_the_column_count_first():
    takes = body_after(_code(), TAKES)
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


def test_route_declines_while_recording_before_anything_else():
    # The route creates a oneDNN primitive at its query; inside a recording that creation would be recorded too. So the
    # recording decline comes before the route reads the buffer, asks gemm() or submits the conversion.
    flat = nows(_route())
    predicate = flat.find(RECORDING)
    assert predicate >= 0, "the route asks whether a command graph records (the batched F32 route's condition)"
    decline = flat.find(RECORDING_DECLINE)
    assert decline > predicate, "the recording predicate is followed by the route's decline"
    for later in ("buffer.capacity(", "backing_handle.resolve(", "bf16_gemm(true", "ggml_sycl_profile_submit("):
        at = flat.find(later)
        assert at >= 0, f"the route no longer has {later!r}"
        assert decline < at, f"the recording decline must come before {later!r}"


def test_route_declines_on_a_short_or_unresolved_buffer_loudly():
    flat = nows(_route())
    short = _branch(flat, "if (capacity < act_bytes) {")
    assert "ggml_sycl::zone_sizing_record_underestimate(buffer.cohort_id, act_bytes, planned);" in short, \
        "a buffer shorter than the copy counts the plan's under-estimate under the buffer's cohort"
    assert "if (!stats.warned) { stats.warned = true; GGML_LOG_WARN(" in short, \
        "a buffer shorter than the copy warns once per device slot"
    assert short.rstrip().endswith("return false;"), "a buffer shorter than the copy is a decline"
    unresolved = _branch(flat, "if (!act || !act.on_device) {")
    assert "if (!stats.warned) { stats.warned = true; GGML_LOG_WARN(" in unresolved, \
        "a buffer that does not resolve on the device warns once per device slot"
    assert unresolved.rstrip().endswith("return false;"), "a buffer that does not resolve on the device is a decline"
    assert "const size_t capacity = buffer.capacity(ctx.device);" in flat, "the decline reads the buffer's capacity"
    assert flat.find("if (capacity < act_bytes) {") < flat.find("backing_handle.resolve(ctx.device)"), \
        "a buffer shorter than the copy is a decline, asked before the buffer is drawn"


def test_route_rounds_the_copy_to_nearest_even_in_a_profiler_row():
    route = _route()
    submit = route.find("ggml_sycl_profile_submit(")
    rne = route.find("ggml_sycl_bf16::f32_to_bf16_rne(")
    run = nows(route).find(RUN)
    assert 0 <= submit < rne and nows(route).find("ggml_sycl_bf16::f32_to_bf16_rne(") < run, \
        "the conversion is launched inside ggml_sycl_profile_submit, rounds with f32_to_bf16_rne, and precedes the GEMM"
    assert '"mulmat.bf16_onednn.act_to_bf16"' in route, "the conversion's profiler row is mulmat.bf16_onednn.act_to_bf16"
    assert "inline uint16_t f32_to_bf16_rne(float f)" in _kernel(), "mul-mat-bf16.hpp defines the device rounding"


def test_route_asks_gemm_before_converting_and_runs_on_the_conversion():
    # A primitive oneDNN cannot create, or a declined scratchpad, ends the route before anything is written; the GEMM
    # then waits on the conversion's event rather than on the queue order alone.
    flat = nows(_route())
    used = flat.find("stats.note_use(act_bytes);")
    query = flat.find(QUERY)
    convert = flat.find(CONVERSION)
    run = flat.find(RUN)
    assert 0 <= used < query, "gemm() is asked once the buffer is drawn"
    assert 0 <= query < convert < run, \
        "gemm() is asked with query_only before the conversion, and run on the conversion's event after it"
    assert flat.count("bf16_gemm(") == 2, "the descriptor is asked exactly twice: the query and the run"
    lambda_at = flat.find(GEMM_LAMBDA)
    assert 0 <= lambda_at < query, "one gemm lambda serves both asks, so they describe the same primitive"
    call = flat[lambda_at:flat.find(";", lambda_at)]
    assert 'static_cast<int>(ldy), deps, "bf16_weight", query_only,' in call, \
        "the lambda hands its deps and query_only to gemm()"


def test_route_gemm_is_strict_bf16_and_declines_on_a_failed_creation():
    route = nows(_route())
    call = route.find(GEMM_LAMBDA)
    assert call >= 0, "the route keeps gemm()'s optional, so a decline reaches the executor"
    args = route[call:route.find(";", call)]
    for arg in ("weight, bf16, 1, k,", "act_bf16, bf16, 1, k,", "dst_ddf, f32, stream, 1, 1,", '"bf16_weight"',
                "GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT", "GGML_SYCL_DNNL_CREATE_FAILURE_DECLINE"):
        assert arg in args, f"the route's gemm() call must carry {arg!r}"
    assert RUN in route, "the route returns whether gemm() ran"


def test_walk_counts_the_route_with_the_executors_questions():
    walk = body_after(_code(), WALK)
    assert walk.lstrip().startswith("#    if GGML_SYCL_DNNL\n"), \
        "the walk is compiled wherever the route is (oneDNN), not only where the f16 arm is"
    flat = nows(walk)
    counted = flat.find(
        "if (ggml_sycl_bf16_onednn_walk_counted() && ggml_sycl_mul_mat_bf16_onednn_takes(src0, src1, &act_bytes) && "
        "ggml_sycl_bf16_weight_native_route_available(src0, src1, node, ctx.device) && "
        "act_bytes > demand[ctx.device].bf16_src1_bytes) { demand[ctx.device].bf16_src1_bytes = act_bytes; "
        "demand[ctx.device].bf16_src1_node = node; } continue;")
    assert counted >= 0, \
        "the walk counts a BF16 node's copy with the executor's predicate and the route's own takes, on this device, " \
        "apart from the f16 arm's demand"
    loop = flat.find("for (int i = 0; i < cgraph->n_nodes; i++)")
    arm = flat.find("# if GGML_SYCL_DEQUANT_F16_ARM", loop)
    assert 0 <= loop < counted < arm, "the BF16 branch sits in the node loop, outside the f16 arm's compile condition"


def test_walk_caps_a_route_demand_and_refuses_only_an_f16_demand():
    # The route has a next path (it declines on a short buffer), the f16 arm has none: a route demand the RUNTIME zone
    # cannot hold caps the buffer, and only an f16 demand refuses the graph (zone_walk_ensure_dense_buffer; its cases
    # are test-zone-sizing Case 53).
    # Compared with no whitespace at all: these statements wrap differently as their names change.
    walk = re.sub(r"\s+", "", body_after(_code(), WALK))

    def at(text: str) -> int:
        return walk.find(re.sub(r"\s+", "", text))

    assert at('{ ctx.dequant_f16_src0_scratch, demand[d].src0_bytes, demand[d].src0_bytes, demand[d].node, "src0", '
              'false },') >= 0, "every src0 demand is the f16 arm's: all of it is required"
    assert at('{ ctx.dequant_f16_src1_scratch, std::max(demand[d].src1_bytes, demand[d].bf16_src1_bytes), '
              'demand[d].src1_bytes, demand[d].src1_node, "src1", true },') >= 0, \
        "the src1 buffer is asked for the wider demand and requires only the f16 arm's"
    decide = at("ggml_sycl::zone_walk_ensure_dense_buffer(b.buffer.capacity(d), planned, b.demand, b.required, "
                "[&](size_t bytes) { return bytes <= ggml_sycl_dequant_f16_walk_limit() && "
                "b.buffer.ensure_buffer(bytes, d, queue) != nullptr; });")
    assert decide >= 0, "the walk grows the buffer only through the decision, which may cap it"
    assert walk.count("ensure_buffer(") == 1, "the decision's grow is the walk's only one"
    refused = at("if (held != ggml_sycl::ZONE_WALK_BUFFER_REFUSED) { continue; }")
    error = at('GGML_LOG_ERROR("[DEQUANT-F16] graph refused before submission')
    assert 0 <= decide < refused < error, "a held or capped buffer goes on; only a refusal reaches the graph's refusal"


def test_planner_plans_the_wider_src1_row():
    planner = body_after(_code(), PLANNER)
    assert ("zone_dense_src1_planned_bytes_per_token(f16_drawable, maxima.dequant_f16_src1_bytes_per_token, "
            "ggml_sycl_bf16_onednn_scratch_drawable(), maxima.bf16_onednn_src1_bytes_per_token)") in nows(planner), \
        "the planned src1 row is the wider of the f16 arm's and the BF16 route's, each where its consumer is compiled in"


def test_adapter_marks_the_bf16_copy_under_the_routes_gate():
    adapter = nows(body_after(_cache_code(), ADAPTER))
    mark = adapter.find("if (shaped && item.type == GGML_TYPE_BF16 && !is_expert && is_matrix && "
                        "ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16)) {")
    assert mark >= 0, "the adapter marks a dense BF16 matrix under the route's own gate"
    assert "desc.bf16_onednn_src1_bytes_per_token = src1_bytes;" in adapter[mark:], \
        "the mark is the copy's bytes per token"


def test_threshold_comment_cites_the_committed_probe():
    raw = _raw()
    takes_at = raw.find(TAKES)
    comment = raw[max(0, takes_at - 2500):takes_at]
    for probe in PROBES:
        assert probe in comment, f"the route's threshold comment must cite {probe}"
        assert (PROBE_DIR / probe).is_file(), f"the cited probe output must be committed at {PROBE_DIR / probe}"


# ---- each route check above fails on a mutant of the code it reads ----------------------------------------------

def ws_pattern(text: str) -> str:
    """A regex for `text` however it is wrapped: no whitespace is significant."""
    return r"\s*".join(re.escape(c) for c in re.sub(r"\s+", "", text))


def mutate(text: str, scope, old: str, new: str) -> str:
    """`text` with the one whitespace-insensitive match of `old` inside the body of `scope` (the whole text when
    `scope` is None) replaced by `new`."""
    start, end = body_span(text, scope) if scope else (0, len(text))
    region = text[start:end]
    found = list(re.finditer(ws_pattern(old), region))
    if len(found) != 1:
        raise LookupError(f"mutant anchor {old!r} matches {len(found)} times in {scope or 'the file'}")
    m = found[0]
    return text[:start] + region[:m.start()] + new + region[m.end():] + text[end:]


# (check, the text it reads, [(scope, old, new), ...] applied in order, what the mutant does)
MUTANTS = [
    (test_executor_offers_the_gemm_route_before_the_native_launch, "code",
     [(EXECUTOR, "&& ggml_sycl_mul_mat_bf16_onednn(ctx, src0,", "&& !ggml_sycl_mul_mat_bf16_onednn(ctx, src0,")],
     "the executor returns on a declined route"),
    (test_route_takes_asks_the_column_count_first, "code",
     [(TAKES, "n_vecs <= ggml_sycl_bf16::SKINNY_MAX_COLS || !ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16)",
       "!ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16) || n_vecs <= ggml_sycl_bf16::SKINNY_MAX_COLS")],
     "the admission is asked before the column count"),
    (test_route_allocates_nothing_and_waits_on_nothing, "code",
     [(ROUTE, "stats.note_use(act_bytes);",
       "(void) buffer.ensure_buffer(act_bytes, ctx.device, *stream); stats.note_use(act_bytes);")],
     "the route grows the buffer"),
    (test_route_allocates_nothing_and_waits_on_nothing, "code",
     [(ROUTE, RUN, "stream->wait(); " + RUN)],
     "the route waits on the host"),
    (test_route_declines_while_recording_before_anything_else, "code",
     [(ROUTE, RECORDING_DECLINE, "")],
     "the recording decline is deleted"),
    (test_route_declines_while_recording_before_anything_else, "code",
     [(ROUTE, RECORDING_DECLINE, ""), (ROUTE, RUN, RECORDING_DECLINE + " " + RUN)],
     "the recording decline moves after the conversion"),
    (test_route_declines_while_recording_before_anything_else, "code",
     [(ROUTE, RECORDING_DECLINE, ""), (ROUTE, "std::string profile_metadata;",
                                       RECORDING_DECLINE + " std::string profile_metadata;")],
     "the recording decline moves between the query and the conversion"),
    (test_route_declines_while_recording_before_anything_else, "code",
     [(ROUTE, RECORDING_DECLINE, ""), (ROUTE, "const ggml_sycl::resolved_ptr act =",
                                       RECORDING_DECLINE + " const ggml_sycl::resolved_ptr act =")],
     "the recording decline moves after the buffer's capacity is asked"),
    (test_route_declines_while_recording_before_anything_else, "code",
     [(ROUTE, "ggml_sycl_graph_recording_active() ||", "")],
     "the recording predicate loses the process-wide recording flag"),
    (test_route_declines_on_a_short_or_unresolved_buffer_loudly, "code",
     [(ROUTE, "ggml_sycl::zone_sizing_record_underestimate(buffer.cohort_id, act_bytes, planned);", "")],
     "a short buffer is not counted against the plan"),
    (test_route_declines_on_a_short_or_unresolved_buffer_loudly, "code",
     [(ROUTE, "planned); if (!stats.warned) {", "planned); if (false) {")],
     "a short buffer's decline is silent"),
    (test_route_declines_on_a_short_or_unresolved_buffer_loudly, "code",
     [(ROUTE, "if (!act || !act.on_device) { if (!stats.warned) {", "if (!act || !act.on_device) { if (false) {")],
     "an unresolved buffer's decline is silent"),
    (test_route_rounds_the_copy_to_nearest_even_in_a_profiler_row, "code",
     [(ROUTE, "ggml_sycl_bf16::f32_to_bf16_rne(", "ggml_sycl_bf16::f32_to_bf16_trunc(")],
     "the copy truncates"),
    (test_route_asks_gemm_before_converting_and_runs_on_the_conversion, "code",
     [(ROUTE, QUERY, ""), (ROUTE, RUN, QUERY + " " + RUN)],
     "the query moves after the conversion"),
    (test_route_asks_gemm_before_converting_and_runs_on_the_conversion, "code",
     [(ROUTE, RUN, "return bf16_gemm(false, {}).has_value();")],
     "the GEMM does not wait on the conversion's event"),
    (test_route_asks_gemm_before_converting_and_runs_on_the_conversion, "code",
     [(ROUTE, 'static_cast<int>(ldy), deps, "bf16_weight", query_only,',
       'static_cast<int>(ldy), {}, "bf16_weight", query_only,')],
     "the lambda drops its deps"),
    (test_route_asks_gemm_before_converting_and_runs_on_the_conversion, "code",
     [(ROUTE, '"bf16_weight", query_only,', '"bf16_weight", false,')],
     "the lambda runs the GEMM on the query"),
    (test_route_gemm_is_strict_bf16_and_declines_on_a_failed_creation, "code",
     [(ROUTE, "GGML_SYCL_DNNL_CREATE_FAILURE_DECLINE", "GGML_SYCL_DNNL_CREATE_FAILURE_THROW")],
     "a failed creation throws"),
    (test_route_gemm_is_strict_bf16_and_declines_on_a_failed_creation, "code",
     [(ROUTE, "GGML_SYCL_DNNL_FPMATH_NO_DOWNCONVERT", "GGML_SYCL_DNNL_FPMATH_BUILD")],
     "the GEMM takes the build's float math"),
    (test_walk_counts_the_route_with_the_executors_questions, "code",
     [(WALK, "ggml_sycl_bf16_weight_native_route_available(src0, src1, node, ctx.device) &&", "")],
     "the walk counts a node the executor does not serve"),
    (test_walk_counts_the_route_with_the_executors_questions, "code",
     [(WALK, "demand[ctx.device].bf16_src1_bytes = act_bytes;", "demand[ctx.device].src1_bytes = act_bytes;")],
     "the walk counts the route's copy as the f16 arm's"),
    (test_walk_caps_a_route_demand_and_refuses_only_an_f16_demand, "code",
     [(WALK, 'demand[d].src1_bytes, demand[d].src1_node, "src1", true },',
       'std::max(demand[d].src1_bytes, demand[d].bf16_src1_bytes), demand[d].src1_node, "src1", true },')],
     "the route's demand is required"),
    (test_walk_caps_a_route_demand_and_refuses_only_an_f16_demand, "code",
     [(WALK, "if (held != ggml_sycl::ZONE_WALK_BUFFER_REFUSED) {", "if (held == ggml_sycl::ZONE_WALK_BUFFER_HELD) {")],
     "a capped buffer refuses the graph"),
    (test_walk_caps_a_route_demand_and_refuses_only_an_f16_demand, "code",
     [(WALK, "bytes <= ggml_sycl_dequant_f16_walk_limit() &&", "")],
     "the walk's grows ignore the test limit"),
    (test_planner_plans_the_wider_src1_row, "code",
     [(PLANNER, "ggml_sycl_bf16_onednn_scratch_drawable(), maxima.bf16_onednn_src1_bytes_per_token)",
       "ggml_sycl_bf16_onednn_scratch_drawable(), 0)")],
     "the planner drops the BF16 row"),
    (test_adapter_marks_the_bf16_copy_under_the_routes_gate, "cache",
     [(ADAPTER, "&& ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16)", "")],
     "the adapter marks without the route's gate"),
    (test_threshold_comment_cites_the_committed_probe, "raw",
     [(None, PROBES[0], "t22-probe-b70-0000000000.txt")],
     "the comment cites a probe that is not committed"),
]

ROUTE_CHECKS = (
    test_executor_offers_the_gemm_route_before_the_native_launch,
    test_route_takes_asks_the_column_count_first,
    test_route_allocates_nothing_and_waits_on_nothing,
    test_route_declines_while_recording_before_anything_else,
    test_route_declines_on_a_short_or_unresolved_buffer_loudly,
    test_route_rounds_the_copy_to_nearest_even_in_a_profiler_row,
    test_route_asks_gemm_before_converting_and_runs_on_the_conversion,
    test_route_gemm_is_strict_bf16_and_declines_on_a_failed_creation,
    test_walk_counts_the_route_with_the_executors_questions,
    test_walk_caps_a_route_demand_and_refuses_only_an_f16_demand,
    test_planner_plans_the_wider_src1_row,
    test_adapter_marks_the_bf16_copy_under_the_routes_gate,
    test_threshold_comment_cites_the_committed_probe,
)

_READERS = {"code": _code, "cache": _cache_code, "raw": _raw}


def test_every_route_check_fails_on_its_mutants():
    missing = [check.__name__ for check in ROUTE_CHECKS if not any(m[0] is check for m in MUTANTS)]
    assert not missing, f"route checks with no mutant: {missing}"
    survived = []
    for check, key, edits, what in MUTANTS:
        original = _READERS[key]()
        mutated = original
        for scope, old, new in edits:
            mutated = mutate(mutated, scope, old, new)
        assert mutated != original, f"mutant {what!r} changed nothing"
        _CACHE[key] = mutated
        try:
            check()
            survived.append(f"{check.__name__}: {what}")
        except AssertionError:
            pass
        except ValueError as exc:
            survived.append(f"{check.__name__}: {what} (ended in {exc!r}, not in the check's assertion)")
        finally:
            _CACHE[key] = original
    assert not survived, f"mutants the checks did not reject: {survived}"


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
