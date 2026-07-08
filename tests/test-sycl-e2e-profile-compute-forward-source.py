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


def assert_no_waits(source: str) -> None:
    assert ".wait(" not in source
    assert "wait_and_throw" not in source


def test_compute_forward_has_e2e_profile_scope_and_flush() -> None:
    src = read_source()
    assert '#include "e2e-profile.hpp"' in src
    begin = src.index("static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {")
    end = src.index("// WEDGE-T4: GGML_SYCL_SAFE_MODE", begin)
    body = src[begin:end]

    safe_dst = body.index("ggml_sycl::sycl_tensor")
    scope_decl = body.index("std::optional<ggml_sycl::e2e_tg_scope> e2e_scope", safe_dst)
    scope_gate = body.index("if (ggml_sycl::e2e_tg_profile_enabled())", scope_decl)
    scope_gate_close = matching_brace(body, body.index("{", scope_gate))
    stage = body.index("ggml_sycl::e2e_tg_stage_from_op(dst->op, dst->name", scope_gate)
    op_name = body.index("ggml_op_name(dst->op), true", stage)
    switch = body.index("switch (dst->op)", scope_gate_close)
    reset = body.index("e2e_scope.reset();", switch)
    flush = body.index("ggml_sycl::e2e_tg_profile_flush_if_ready(stderr);", reset)

    assert safe_dst < scope_decl < scope_gate < stage < op_name < scope_gate_close < switch < reset < flush
    assert_no_waits(body[scope_decl:scope_gate_close])
    assert_no_waits(body[reset:flush])


def test_compute_forward_records_cpu_dispatch_success() -> None:
    src = read_source()
    begin = src.index("if (!ggml_sycl_graph_dispatch_recording_active(&ctx) && should_dispatch_to_cpu")
    end = src.index("if (dst->src[0] != nullptr", begin)
    branch = src[begin:end]
    assert "ggml_sycl_compute_forward_cpu(ctx, dst)" in branch
    record = branch.index("ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::CPU_DISPATCH")
    gate = branch.rindex("if (ggml_sycl::e2e_tg_profile_enabled())", 0, record)
    gate_close = matching_brace(branch, branch.index("{", gate))
    flush = branch.index("ggml_sycl::e2e_tg_profile_flush_if_ready(stderr);", record)
    ret = branch.index("return true;", flush)
    assert gate < record < flush < gate_close < ret
    assert_no_waits(branch[gate:gate_close])


def test_graph_diag_records_e2e_graph_stage() -> None:
    src = read_source()
    begin = src.index("static void ggml_sycl_graph_diag_report")
    end = src.index("static void ggml_sycl_moe_aggregation_diag", begin)
    body = src[begin:end]
    record = body.index("ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::GRAPH")
    gate = body.rindex("if (ggml_sycl::e2e_tg_profile_enabled())", 0, record)
    gate_close = matching_brace(body, body.index("{", gate))
    assert gate < record < gate_close
    assert "use_graph ? \"use_graph_1\" : \"use_graph_0\"" in body[record:gate_close]
    assert_no_waits(body[gate:gate_close])


def test_compute_forward_records_and_flushes_early_handled_routes() -> None:
    src = read_source()
    begin = src.index("auto e2e_record_early_handled_route")
    helper_open = src.index("{", begin)
    helper_close = matching_brace(src, helper_open)
    helper = src[begin:helper_close]
    gate = helper.index("if (ggml_sycl::e2e_tg_profile_enabled())")
    gate_close = matching_brace(helper, helper.index("{", gate))
    record = helper.index("ggml_sycl::e2e_tg_profile_record", gate)
    flush = helper.index("ggml_sycl::e2e_tg_profile_flush_if_ready(stderr);", record)
    assert gate < record < flush < gate_close
    assert "ggml_sycl::e2e_tg_stage_from_op(dst->op, dst->name)" in helper[record:gate_close]
    assert_no_waits(helper[gate:gate_close])

    end = src.index("std::optional<ggml_sycl::e2e_tg_scope> e2e_scope", helper_close)
    body = src[helper_close:end]
    tp_skip_pos = body.index("// Skip ALL ops except TP MUL_MAT on secondary devices")
    tp_return_pos = body.index("return true;  // Op \"succeeded\" by being skipped", tp_skip_pos)
    assert "e2e_record_early_handled_route();" in body[tp_skip_pos:tp_return_pos]
    for route in (
        "ggml_sycl_try_route_simple_consumer",
        "ggml_sycl_try_route_flash_attn_ext",
        "ggml_sycl_try_route_mul_mat_weight_owner",
        "ggml_sycl_try_route_mul_mat_activation",
    ):
        route_pos = body.index(route)
        return_pos = body.index("return true;", route_pos)
        route_block = body[route_pos:return_pos]
        assert "e2e_record_early_handled_route();" in route_block
