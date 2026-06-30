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
    assert "ggml_sycl::e2e_tg_scope e2e_scope" in body
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
