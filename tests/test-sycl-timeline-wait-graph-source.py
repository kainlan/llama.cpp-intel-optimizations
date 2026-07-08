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


def function_body(source: str, signature: str) -> str:
    begin = source.index(signature)
    open_brace = source.index("{", begin)
    close_brace = matching_brace(source, open_brace)
    return source[open_brace:close_brace]


def compute_impl_guard_body(source: str) -> str:
    graph_body = function_body(source, "static void ggml_backend_sycl_graph_compute_impl")
    begin = graph_body.index("struct compute_impl_guard")
    open_brace = graph_body.index("{", begin)
    close_brace = matching_brace(graph_body, open_brace)
    return graph_body[open_brace:close_brace]


def test_trace_queue_wait_records_timeline_scope_around_queue_waits() -> None:
    src = read_source()
    body = function_body(src, "static void ggml_sycl_trace_queue_wait")

    assert "std::optional<ggml_sycl::sycl_timeline_scope> wait_timeline_scope" in body
    assert "timeline_wait_span_enabled = g_sycl_timeline_graph_spans_enabled" in body
    gate = body.index("if (timeline_wait_span_enabled)")
    gate_close = matching_brace(body, body.index("{", gate))
    gate_body = body[gate:gate_close]

    reason = gate_body.index('wait_timeline_metadata = "reason="')
    node_metadata = gate_body.index("ggml_sycl_timeline_node_metadata", reason)
    scope = gate_body.index('"sycl.wait", "queue_wait"', node_metadata)
    first_wait = body.index("q->wait();", gate_close)

    assert reason < node_metadata < scope
    assert gate < first_wait
    assert "safe_reason" in gate_body
    assert "node_idx" in gate_body
    assert "device" in gate_body
    assert "node_name" in gate_body
    assert "op_name" in gate_body

    wait_positions = []
    search_from = 0
    while True:
        index = body.find("q->wait();", search_from)
        if index < 0:
            break
        wait_positions.append(index)
        search_from = index + len("q->wait();")
    assert wait_positions
    assert all(gate < wait for wait in wait_positions)


def test_compute_impl_guard_records_timeline_scopes_around_bcs_and_dma_drains() -> None:
    src = read_source()
    assert "timeline_graph_span_flag_guard_(timeline_spans_enabled)" in src
    guard = compute_impl_guard_body(src)

    for queue_name, span_name in (
        ("bcs", "bcs_queue_drain"),
        ("dma", "dma_queue_drain"),
    ):
        wait = guard.index(f"cache->get_{queue_name}_queue().wait();")
        preceding = guard[max(0, wait - 900) : wait]
        following = guard[wait : wait + 120]

        assert f"std::optional<ggml_sycl::sycl_timeline_scope> {span_name}_timeline_scope" in preceding
        assert "if (g_sycl_timeline_graph_spans_enabled)" in preceding
        assert f'"sycl.wait", "{span_name}"' in preceding
        assert f"{span_name}_timeline_scope.emplace" in preceding
        assert f"queue={queue_name}" in preceding
        assert "catch (...)" in following


def test_moe_sequence_graphlet_records_timeline_scopes_around_refresh_record_replay() -> None:
    src = read_source()
    begin = src.index("const char * ptr_table_reject = nullptr;")
    invalidate_call = "sycl_ctx->invalidate_moe_sequence_graphs();"
    end = src.index(invalidate_call, begin) + len(invalidate_call)
    sequence_graphlet = src[begin:end]

    assert 'GGML_SYCL_TIMELINE_SCOPE("sycl.graph", "moe_sequence_pointer_table_refresh"' in sequence_graphlet
    assert 'GGML_SYCL_TIMELINE_SCOPE("sycl.graph", "moe_sequence_graphlet_record"' in sequence_graphlet
    assert 'GGML_SYCL_TIMELINE_SCOPE("sycl.graph", "moe_sequence_graphlet_replay"' in sequence_graphlet
    assert "ggml_sycl_timeline_node_metadata" in sequence_graphlet
    assert "sycl_ctx->device" in sequence_graphlet
    assert "node_idx" in sequence_graphlet
    assert "cgraph->n_nodes" in sequence_graphlet
    assert "node->name" in sequence_graphlet
    assert "ggml_op_name(node->op)" in sequence_graphlet
    assert "moe_sequence_graphlet_prepare_pointer_tables" in sequence_graphlet
    assert "moe_graph_record_moe_dispatch_graph" in sequence_graphlet
    assert "ext_oneapi_graph(*exec_graph)" in sequence_graphlet
