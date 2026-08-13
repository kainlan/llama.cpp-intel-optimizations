#!/usr/bin/env python3
"""Central retained prompt-fusion dataflow and fallback contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


def prompt_region() -> str:
    text = SOURCE.read_text()
    start = text.index("// Prompt pair/fusion is admitted only")
    end = text.index("auto record_moe_gpu_path", start)
    return text[start:end]


def test_central_prompt_fusion_hook_and_enable_predicate() -> None:
    region = prompt_region()
    for witness in (
        "prompt_pair_retained_roles_validated && src0->type == GGML_TYPE_MXFP4",
        "src0 == pair.gate_weight && dst == pair.gate_dst",
        "!pair.gate_bias && !pair.up_bias &&",
        "!pair.down_bias",
        "pair.down_dst->src[1] == pair.glu_dst",
        "mmvq_submit_retained_prompt_fusion(",
        "mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(",
        "mmvq_moe_batched_dispatch_down_from_cached_q8_mxfp4(",
        "gate_layout == GGML_LAYOUT_SOA && up_layout == gate_layout",
        "down_layout == GGML_LAYOUT_SOA || down_layout == GGML_LAYOUT_MXFP4_I8",
        "&glu_event, &glu_event_set, &recorder, &ids_deps",
        "recorder.write_started() ?",
        "fused::ErrorCode::submit_failed_no_write",
    ):
        assert witness in region, witness
    assert "q1_nvfp4_direct_b70_validated" not in region


def test_exact_retained_escrow_and_atomic_publication_contract() -> None:
    region = prompt_region()
    for role in (
        "gate", "gate_table", "up", "up_table", "activation", "ids",
        "glu", "intermediate", "down", "down_table",
    ):
        assert f"fused::OwnerRole::{role}" in region, role
    for witness in (
        "operand.expert_id, operand.occurrence",
        "operand.token_index, operand.slot_index",
        "identity(operand.lease)",
        "fusion_bundle.ids_identity          = identity(ids_device_handle)",
        "ids_resolved.ptr == ids_device",
        "std::vector<sycl::event>{ ids_ready_event }",
        "std::vector<sycl::event> ids_deps{ ids_ready_event }",
        "static thread_local fused::PublicationStore publication_store",
        "stage_skip(g_moe_precomputed_mmid_skip, pair.gate_dst)",
        "stage_skip(g_moe_precomputed_mmid_skip, pair.up_dst)",
        "stage_skip(g_moe_precomputed_node_skip, pair.glu_dst)",
        "stage_skip(g_moe_precomputed_mmid_skip, pair.down_dst)",
        "ggml_sycl_set_tensor_ready_event(pair.glu_dst",
        "ggml_sycl_set_tensor_ready_event(pair.down_dst",
    ):
        assert witness in region, witness


def test_fallback_is_prewrite_only() -> None:
    region = prompt_region()
    hook = region.index("mmvq_submit_retained_prompt_fusion(")
    fallback = region.index("if (!result.fallback_allowed)", hook)
    quarantine = region.index("throw std::runtime_error(result.status.detail)", fallback)
    retained = region.index("Pre-write refusal", quarantine)
    assert hook < fallback < quarantine < retained
    assert "post-write quarantine: never enter the unfused path" in region


def test_unsupported_layouts_and_shape_mismatches_refuse_before_artifacts() -> None:
    region = prompt_region()
    admission = region.index("const bool executor_layouts_ok")
    shapes = region.index("const bool executor_shapes_ok", admission)
    table_upload = region.index("ggml_sycl_upload_moe_retained_ptr_table_from_batch", shapes)
    ids_stage = region.index("ggml_sycl_get_moe_ids_device_ptr", table_upload)
    assert admission < shapes < table_upload < ids_stage
    matrix = region[admission:table_upload]
    assert "gate_layout == GGML_LAYOUT_XMX_TILED" not in matrix
    assert "down_layout == GGML_LAYOUT_MXFP4_DPAS" not in matrix
    assert "pair.ids->ne[1] == pair.src1->ne[2]" in region[shapes:table_upload]


def test_production_write_boundary_is_inside_mmvq_before_first_submit() -> None:
    mmvq = (ROOT / "ggml/src/ggml-sycl/mmvq.cpp").read_text()
    start = mmvq.index("bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(")
    end = mmvq.index("// Debug A/B", start)
    body = mmvq[start:end]
    boundary = body.index("write_recorder->mark_write_started()")
    first_write = body.index("mmvq_profile_submit_quantize_activation_q8_soa(")
    assert boundary < first_write
    assert "return false;" in body[:boundary]  # allocation/shape refusal remains fallback-safe


def test_prompt_q8_preflight_covers_cold_cache_artifact_and_strides() -> None:
    mmvq = (ROOT / "ggml/src/ggml-sycl/mmvq.cpp").read_text()
    preflight_start = mmvq.index("bool mmvq_moe_prompt_q8_preflight(")
    pair_start = mmvq.index("bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(", preflight_start)
    preflight = mmvq[preflight_start:pair_start]
    for witness in (
        "mxfp4_moe_tg_q8_artifact_enabled()",
        "GGML_SYCL_MOE_PROMPT_Q8_PREFLIGHT_FAIL_ALLOC",
        "gate_layer == up_layer && gate_layer == down_layer",
        "glu_dst->nb[0] == sizeof(float)",
        "glu_dst->nb[1] == static_cast<size_t>(glu_dst->ne[0]) * sizeof(float)",
        "glu_dst->nb[2] == static_cast<size_t>(glu_dst->ne[1]) * glu_dst->nb[1]",
        "mxfp4_moe_prompt_q8_bytes(src1->ne[0]",
        "mxfp4_moe_prompt_q8_bytes(glu_dst->ne[0]",
        "std::max(activation_bytes, glu_bytes)",
        "mxfp4_moe_tg_reuse_get_or_alloc_q8",
        "out->q8_owner",
    ):
        assert witness in preflight, witness


def test_prompt_receipt_prevents_post_boundary_q8_growth_and_retains_owner() -> None:
    mmvq = (ROOT / "ggml/src/ggml-sycl/mmvq.cpp").read_text()
    start = mmvq.index("bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(")
    end = mmvq.index("// Debug A/B", start)
    body = mmvq[start:end]
    boundary = body.index("write_recorder->mark_write_started()")
    receipt = body.index("prompt_q8_preflight->q8_owner.resolve", 0, boundary)
    publish = body.index("prompt_q8_preflight ? &prompt_q8_preflight->q8_owner", boundary)
    assert receipt < boundary < publish
    # The reserved-owner publication arm resolves the receipt and never calls
    # the growing allocator; the allocator is confined to the ordinary arm.
    publish_fn_start = mmvq.index("static bool mxfp4_moe_tg_publish_q8_soa(")
    publish_fn_end = mmvq.index("// Returns true if the tensor's weights", publish_fn_start)
    publish_fn = mmvq[publish_fn_start:publish_fn_end]
    reserved_arm = publish_fn[publish_fn.index("if (reserved_q8_owner)"):
                              publish_fn.index("} else {", publish_fn.index("if (reserved_q8_owner)"))]
    assert "mxfp4_moe_tg_reuse_get_or_alloc_q8" not in reserved_arm
    region = prompt_region()
    assert "activation, q8_preflight.q8_owner" in region
    assert "&recorder, &ids_deps, &q8_preflight" in region
    allocator_start = mmvq.index("static void * mxfp4_moe_tg_reuse_get_or_alloc_q8(")
    allocator_end = mmvq.index("static void * mxfp4_moe_tg_reuse_get_or_alloc_device_scratch", allocator_start)
    allocator = mmvq[allocator_start:allocator_end]
    assert "retired_owner = cache.q8_handle" in allocator
    assert "retain_handles_until_event" in allocator
