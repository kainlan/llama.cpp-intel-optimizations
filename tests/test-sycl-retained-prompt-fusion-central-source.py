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
