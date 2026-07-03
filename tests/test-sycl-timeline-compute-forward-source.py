from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


def read_source() -> str:
    return GGML_SYCL.read_text(encoding="utf-8")


def matching_brace(source: str, open_brace: int) -> int:
    depth = 0
    for index in range(open_brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("no matching brace")


def graph_compute_impl_body(src: str) -> str:
    begin = src.index("static void ggml_backend_sycl_graph_compute_impl")
    open_brace = src.index("{", begin)
    close_brace = matching_brace(src, open_brace)
    return src[open_brace:close_brace]


def assert_no_waits(source: str) -> None:
    assert ".wait(" not in source
    assert "wait_and_throw" not in source


def test_graph_compute_impl_has_timeline_scope_after_reentry_guard() -> None:
    src = read_source()
    assert '#include "sycl-timeline.hpp"' in src

    body = graph_compute_impl_body(src)

    guard = body.index("compute_impl_guard _reentry_guard")
    scope = body.index('GGML_SYCL_TIMELINE_SCOPE("ggml.graph", "graph_compute_impl"', guard)
    compute_start = body.index("auto t_compute_start", guard)

    assert guard < scope < compute_start
    scope_setup = body[guard:compute_start]
    assert_no_waits(scope_setup)


def test_compute_forward_early_handled_route_has_timeline_scope_metadata() -> None:
    src = read_source()
    begin = src.index("auto e2e_record_early_handled_route")
    helper_open = src.index("{", begin)
    helper_close = matching_brace(src, helper_open)
    helper = src[begin:helper_close]

    assert "g_sycl_timeline_graph_spans_enabled" in src
    gate = helper.index("if (g_sycl_timeline_graph_spans_enabled)")
    gate_close = matching_brace(helper, helper.index("{", gate))
    metadata = helper.index("ggml_sycl_timeline_tensor_metadata", gate)
    scope = helper.index('"ggml.op", "early_handled_route"', metadata)
    e2e_record = helper.index("ggml_sycl::e2e_tg_profile_record", gate_close)

    assert gate < metadata < scope < gate_close < e2e_record
    assert "std::optional<ggml_sycl::sycl_timeline_scope>" in helper[:gate]
    assert "ctx.device" in helper[gate:gate_close]
    assert "dst ? dst->name : nullptr" in helper[gate:gate_close]
    assert "dst ? ggml_op_name(dst->op) : nullptr" in helper[gate:gate_close]
    assert_no_waits(helper[gate:gate_close])


def test_compute_forward_switch_has_timeline_scope_metadata() -> None:
    src = read_source()
    begin = src.index("static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {")
    switch = src.index("switch (dst->op)", begin)
    preceding_window = src[max(0, switch - 2500) : switch]

    assert "timeline_graph_span_flag_guard" in src
    gate = preceding_window.index("if (g_sycl_timeline_graph_spans_enabled)")
    gate_close = matching_brace(preceding_window, preceding_window.index("{", gate))
    metadata = preceding_window.index("ggml_sycl_timeline_tensor_metadata", gate)
    scope = preceding_window.index('"ggml.op", "compute_forward"', metadata)

    assert gate < metadata < scope < gate_close
    assert "std::optional<ggml_sycl::sycl_timeline_scope>" in preceding_window
    assert "ggml_sycl_timeline_tensor_metadata(dst, ctx.device, dst->name" in preceding_window
    assert "ggml_op_name(dst->op)" in preceding_window
    assert_no_waits(preceding_window)


def test_node_loop_compute_forward_has_timeline_scope_metadata() -> None:
    src = read_source()
    body = graph_compute_impl_body(src)

    loop_marker = "impl_phase_log(\"pre_loop\");"
    loop_start = body.index("for (int i = 0; i < cgraph->n_nodes; i++) {", body.index(loop_marker))
    loop_open = body.index("{", loop_start)
    loop_close = matching_brace(body, loop_open)
    loop_body = body[loop_open:loop_close]

    compute_line = "bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);"
    positions = []
    search_from = 0
    while True:
        try:
            position = loop_body.index(compute_line, search_from)
        except ValueError:
            break
        positions.append(position)
        search_from = position + len(compute_line)

    assert positions
    for position in positions:
        preceding_window = loop_body[max(0, position - 1000) : position]
        assert "std::optional<ggml_sycl::sycl_timeline_scope>" in preceding_window
        assert "timeline_spans_enabled" in preceding_window
        assert '"ggml.op", "compute_forward_node"' in preceding_window
        assert "ggml_sycl_timeline_node_metadata" in preceding_window
        assert "node->name" in preceding_window
        assert "ggml_op_name(node->op)" in preceding_window
        assert_no_waits(preceding_window)
