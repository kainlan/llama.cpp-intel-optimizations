from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGML_SYCL = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


def read_source() -> str:
    return GGML_SYCL.read_text(encoding="utf-8")


def test_compute_forward_has_e2e_profile_scope_and_flush() -> None:
    src = read_source()
    assert '#include "e2e-profile.hpp"' in src
    begin = src.index("static bool ggml_sycl_compute_forward")
    end = src.index("// WEDGE-T4: GGML_SYCL_SAFE_MODE", begin)
    body = src[begin:end]
    assert "std::optional<ggml_sycl::e2e_tg_scope> e2e_scope" in body
    assert "e2e_scope.reset();" in body
    assert "ggml_sycl::e2e_tg_stage_from_op(dst->op, dst->name" in body
    assert "ggml_sycl::e2e_tg_profile_flush_if_ready(stderr);" in body


def test_compute_forward_records_cpu_dispatch_success() -> None:
    src = read_source()
    begin = src.index("if (!ggml_sycl_graph_dispatch_recording_active(&ctx) && should_dispatch_to_cpu")
    end = src.index("if (dst->src[0] != nullptr", begin)
    branch = src[begin:end]
    assert "ggml_sycl_compute_forward_cpu(ctx, dst)" in branch
    assert "ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::CPU_DISPATCH" in branch


def test_graph_diag_records_e2e_graph_stage() -> None:
    src = read_source()
    begin = src.index("static void ggml_sycl_graph_diag_report")
    end = src.index("static void ggml_sycl_moe_aggregation_diag", begin)
    body = src[begin:end]
    assert "ggml_sycl::e2e_tg_profile_record(ggml_sycl::e2e_tg_stage::GRAPH" in body
    assert "use_graph ? \"use_graph_1\" : \"use_graph_0\"" in body


def test_compute_forward_records_and_flushes_early_handled_routes() -> None:
    src = read_source()
    begin = src.index("auto e2e_record_early_handled_route")
    end = src.index("ggml_sycl::sycl_tensor safe_dst", begin)
    body = src[begin:end]
    assert "ggml_sycl::e2e_tg_stage_from_op(dst->op, dst->name)" in body
    assert "ggml_sycl::e2e_tg_profile_flush_if_ready(stderr);" in body
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
