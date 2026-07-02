#!/usr/bin/env python3
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp"
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
GGML_SYCL = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"
UNIFIED_CACHE = ROOT / "ggml" / "src" / "ggml-sycl" / "unified-cache.cpp"
COMMON_HPP = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
GGML_H = ROOT / "ggml" / "include" / "ggml.h"
BENCH_ARGS = ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp"
REFERENCE_HEADER = ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "reference_kernels.hpp"
HARNESS = ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
MAIN = ROOT / "tools" / "sycl-kernel-bench" / "main.cpp"


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def test_bundle4_layout_formula_preserves_baseline_bytes_and_aligns_payloads() -> None:
    tile_n_total = 16
    packed_bytes = 16
    bundle_groups = 4
    baseline_group_bytes = tile_n_total * (1 + packed_bytes)
    payload_group_bytes = tile_n_total * packed_bytes
    payload_slab_bytes = bundle_groups * payload_group_bytes
    scale_slab_bytes = bundle_groups * tile_n_total
    bundle_bytes = payload_slab_bytes + scale_slab_bytes
    assert baseline_group_bytes == 272
    assert bundle_bytes == 1088
    assert bundle_bytes == bundle_groups * baseline_group_bytes
    assert [g * payload_group_bytes for g in range(bundle_groups)] == [0, 256, 512, 768]
    assert all((g * payload_group_bytes) % 64 == 0 for g in range(bundle_groups))
    assert payload_slab_bytes == 1024


def test_reference_bundle4_layout_helper_exists() -> None:
    reference = REFERENCE.read_text(encoding="utf-8")
    helper = slice_between(reference, "make_xmx_tiled_bundle4_payload_layout", "static int normalize_supported_xmx_tiles_n")
    assert "constexpr size_t bundle_groups" in helper
    assert "const size_t     payload_group_bytes" in helper
    assert "const size_t     payload_slab_bytes" in helper
    assert "const size_t     scale_slab_bytes" in helper
    assert "const size_t     bundle_bytes" in helper
    assert "bundle_bytes == bundle_groups * old_group_bytes" in helper
    assert "group_n / bundle_groups" in helper
    assert "group_n % bundle_groups" in helper
    normalized = " ".join(helper.split())
    assert "new_payload = new_bundle + group_in_bundle * payload_group_bytes" in normalized
    assert "new_scale = new_bundle + payload_slab_bytes + group_in_bundle * tile_n_total" in normalized


def test_bundle4_benchmark_route_is_registered_and_parsed() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")
    harness = HARNESS.read_text(encoding="utf-8")
    bench_args = BENCH_ARGS.read_text(encoding="utf-8")
    reference_header = REFERENCE_HEADER.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")
    route = "mxfp4_pair_glu_xmx_tiled_bundle4_packed_r8_m2_sparse32_bias"
    assert route in registry
    assert route in main
    assert "parse_moe_xmx_tiled_bundle4" in harness
    assert "const bool xmx_tiled_bundle4             = parse_moe_xmx_tiled_bundle4(config.kernel_name)" in harness
    assert "const int  xmx_tiled_bundle4_group_bytes = xmx_tiled_bundle4 ? 1088 : 0" in harness
    assert "bool  xmx_tiled_bundle4" in bench_args
    assert "int   xmx_tiled_bundle4_group_bytes" in bench_args
    assert "bool                         xmx_tiled_bundle4" in reference_header
    assert "int                          xmx_tiled_bundle4_group_bytes" in reference_header
    assert "xmx_tiled_bundle4, xmx_tiled_bundle4_group_bytes" in reference


def test_reference_bundle4_route_is_fail_closed_before_layout_allocation() -> None:
    reference = REFERENCE.read_text(encoding="utf-8")
    invalid = slice_between(
        reference,
        "if (xmx_tiled_bundle4 &&",
        "const size_t         selected_count",
    )
    assert "!xmx_tiled" in invalid
    assert "!xmx_tiled_pack_q8" in invalid
    assert "xmx_tiled_grouped" in invalid
    assert "xmx_tiled_prefetch" in invalid
    assert "xmx_tiled_m_tiles != 2" in invalid
    assert "rows_per_wg != 8" in invalid
    assert "xmx_tiles_n != 1" in invalid
    assert "xmx_tiled_bundle4_group_bytes != 1088" in invalid
    assert "mxfp4_pair_glu XMX_TILED_BUNDLE4 requires packed XMX_TILED r8 m2 and 1088-byte bundles." in invalid
    layout_block = slice_between(reference, "if (xmx_tiled_v2)", "const size_t  expert_bytes")
    assert "if (xmx_tiled_bundle4)" in layout_block
    assert "make_xmx_tiled_bundle4_payload_layout(launch_layout, m, k)" in layout_block


def test_sycl_bundle4_loader_and_kernel_route_exist() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "mxfp4_xmx_tiled_bundle4_load_a_vec_from_bundle" in mmvq
    assert "struct mxfp4_pair_glu_xmx_tiled_bundle4_dpas_m2_kernel" in mmvq
    assert "mxfp4_pair_glu_xmx_tiled_bundle4_dpas_m2_sycl" in mmvq
    helper = slice_between(
        mmvq,
        "mxfp4_xmx_tiled_bundle4_load_a_vec_from_bundle",
        "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_v2_load_a_vec_from_group",
    )
    assert "constexpr int bundle_groups" in helper
    assert "payload_group_bytes" in helper
    assert "payload_slab_bytes" in helper
    assert "scale_slab_bytes" in helper
    assert "group_in_bundle * payload_group_bytes" in helper
    assert "payload_slab_bytes + group_in_bundle * scale_slab_bytes" in helper


def test_sycl_bundle4_launch_is_fail_closed_before_pack_enqueue() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    validation = slice_between(mmvq, "if (args.xmx_tiled_bundle4)", "if (args.xmx_tiled_v2)")
    assert "args.xmx_tiled_bundle4_group_bytes == 1088" in validation
    assert "args.xmx_tiled_m_tiles == 2" in validation
    assert "args.rows_per_wg == 8" in validation
    assert "args.xmx_tiles_n == 1" in validation
    assert "!args.xmx_tiled_v2" in validation
    launch = slice_between(
        mmvq,
        "if (args.xmx_tiled_pack_q8)",
        "if (args.xmx_tiled_m_tiles == 4)",
    )
    bundle4_pos = launch.index("if (args.xmx_tiled_bundle4)")
    pack_pos = launch.index("mxfp4_dpas_pack_q8_single_col_groups_sycl", bundle4_pos)
    submit_pos = launch.index("mxfp4_pair_glu_xmx_tiled_bundle4_dpas_m2_submit<8", bundle4_pos)
    dispatch_validation = launch[bundle4_pos:pack_pos]
    assert "!args.xmx_tiled_v2" in dispatch_validation
    assert bundle4_pos < pack_pos < submit_pos
    assert "return true" in launch[submit_pos:]


def test_runtime_bundle4_layout_has_separate_enum_name_and_cache_key() -> None:
    ggml_h = GGML_H.read_text(encoding="utf-8")
    common = COMMON_HPP.read_text(encoding="utf-8")
    cache = UNIFIED_CACHE.read_text(encoding="utf-8")
    sycl = GGML_SYCL.read_text(encoding="utf-8")

    enum_block = slice_between(ggml_h, "enum ggml_layout_mode", "struct ggml_tensor_layout")
    assert "GGML_LAYOUT_XMX_TILED," in enum_block
    assert "GGML_LAYOUT_XMX_TILED_BUNDLE4" in enum_block
    assert enum_block.index("GGML_LAYOUT_XMX_TILED,") < enum_block.index("GGML_LAYOUT_XMX_TILED_BUNDLE4")

    assert "layout-ready-xmx-tiled-bundle4" in common
    assert "case GGML_LAYOUT_XMX_TILED_BUNDLE4" in common
    assert "ggml_sycl_layout_is_tiled" in common
    assert "mode == GGML_LAYOUT_XMX_TILED || mode == GGML_LAYOUT_XMX_TILED_BUNDLE4" in common

    assert "planner_moe_gateup_bundle4_enabled" in cache
    assert "GGML_SYCL_MOE_GATEUP_BUNDLE4" in cache
    assert "planner_layout_bytes_xmx_tiled_bundle4_for_dims" in cache
    assert "GGML_LAYOUT_XMX_TILED_BUNDLE4" in cache
    assert "make_direct_stage_key(cache_entry_type::MOE_EXPERT, key, layout)" in cache
    assert "ggml_sycl_layout_specific_moe_expert_cache_key(base_key, target_layout)" in sycl


def test_runtime_bundle4_materializer_is_env_gated_and_fail_closed() -> None:
    sycl = GGML_SYCL.read_text(encoding="utf-8")
    cache = UNIFIED_CACHE.read_text(encoding="utf-8")

    helper = slice_between(sycl, "static bool ggml_sycl_moe_gateup_bundle4_enabled", "static size_t ggml_sycl_xmx_tiled_bundle4_bytes_for_dims")
    assert "GGML_SYCL_MOE_GATEUP_BUNDLE4" in helper
    assert "return env && std::atoi(env) != 0" in helper

    support = slice_between(sycl, "static bool ggml_sycl_moe_gateup_bundle4_supported", "static layout_mode ggml_sycl_select_moe_gpu_layout")
    assert "moe_kind != MOE_TENSOR_GATE && moe_kind != MOE_TENSOR_UP" in support
    assert "ggml_sycl_xmx_tiled_bundle4_bytes_for_dims" in support

    metadata_guard = slice_between(
        sycl,
        "if (ctx->n_experts <= 0 || ctx->info.total_bytes == 0)",
        "const size_t expert_dst_bytes = ggml_sycl_xmx_tiled_bundle4_bytes_for_info(ctx->info)",
    )
    assert "if (ctx->bundle4)" in metadata_guard
    assert "XMX tiled bundle4 fill invalid metadata" in metadata_guard
    assert "throw std::runtime_error(\"XMX tiled bundle4 invalid metadata\")" in metadata_guard

    fill = slice_between(sycl, "if (ctx->bundle4)", "const size_t expert_bytes = ctx.bundle4")
    assert "ggml_sycl_xmx_tiled_bundle4_bytes_for_info" in fill
    assert "XMX tiled bundle4 fill invalid shape/size" in fill
    assert "throw std::runtime_error(\"XMX tiled bundle4 invalid shape/size\")" in fill
    assert "XMX tiled bundle4 host materialization failed" in fill
    assert "throw std::runtime_error(\"XMX tiled bundle4 host materialization failed\")" in fill
    assert "XMX tiled bundle4 fill failed" in fill
    assert "throw;" in fill
    assert "ggml_sycl_reorder_mxfp4_to_xmx_bundle4_layout" in fill
    assert "ctx->source_layout == GGML_LAYOUT_SOA" in fill

    planner = slice_between(cache, "static bool planner_moe_gateup_bundle4_enabled", "static size_t planner_layout_bytes_xmx_tiled_for_dims")
    assert "GGML_SYCL_MOE_GATEUP_BUNDLE4" in planner
    assert "return env && std::atoi(env) != 0" in planner


def test_runtime_bundle4_pp_safe_tg_only_routing() -> None:
    sycl = GGML_SYCL.read_text(encoding="utf-8")
    cache = UNIFIED_CACHE.read_text(encoding="utf-8")
    mmvq = MMVQ.read_text(encoding="utf-8")

    pp_support = slice_between(
        cache,
        "static bool planner_moe_primary_executor_supports_pp_layout_on_device",
        "static bool planner_moe_primary_executor_supports_pp_layout(",
    )
    assert "layout == GGML_LAYOUT_XMX_TILED_BUNDLE4" in pp_support
    assert "return false" in pp_support

    pp_soa = slice_between(
        cache,
        "static bool planner_moe_layout_needs_pp_soa_on_device",
        "static bool planner_moe_layout_needs_pp_soa(",
    )
    assert "layout == GGML_LAYOUT_XMX_TILED || layout == GGML_LAYOUT_XMX_TILED_BUNDLE4" in pp_soa
    assert "if (layout == GGML_LAYOUT_XMX_TILED_BUNDLE4)" in pp_soa
    assert "return true" in pp_soa

    selected_rows = slice_between(sycl, "static layout_mode ggml_sycl_moe_layout_for_selected_rows", "// Prepare MoE pointer tables")
    assert "layout == GGML_LAYOUT_XMX_TILED_BUNDLE4" in selected_rows
    assert "n_tokens > 1 || selected_rows == 0" in selected_rows
    assert "return n_tokens > 1 ? GGML_LAYOUT_SOA" in selected_rows

    phase = slice_between(sycl, "static layout_mode ggml_sycl_moe_phase_target_layout", "static bool   ggml_sycl_release_moe_tensor_layout")
    assert "planned_layout == GGML_LAYOUT_XMX_TILED_BUNDLE4" in phase
    assert "planned_layout == GGML_LAYOUT_SOA && ggml_sycl_moe_plan_pp_soa_promoted(device)" in phase
    assert "return GGML_LAYOUT_XMX_TILED_BUNDLE4" in phase

    runtime_phase = slice_between(sycl, "static bool ggml_sycl_moe_runtime_phase_materialization_enabled", "static bool ggml_sycl_moe_phase_materialization_needed")
    assert "ggml_sycl_moe_gateup_bundle4_enabled()" in runtime_phase

    pair_coordinator = slice_between(
        sycl,
        "if ((pair_layout != GGML_LAYOUT_SOA && pair_layout != GGML_LAYOUT_MXFP4_I8 &&",
        "return reject_pair(\"layout\")",
    )
    assert "pair_layout != GGML_LAYOUT_XMX_TILED_BUNDLE4" in pair_coordinator

    pair_ids_bridge = slice_between(
        sycl,
        "const bool use_device_grouped_moe_decode =",
        "trace_pair_stage(\"pair-glu-submit-begin\")",
    )
    assert "pair_layout == GGML_LAYOUT_XMX_TILED_BUNDLE4" in pair_ids_bridge
    assert "const int32_t * pair_ids_host_arg = use_device_ids_for_pair_glu ? nullptr : ids_data" in pair_ids_bridge
    assert "use_device_ids_for_pair_glu ? 0 : static_cast<int64_t>(ids_n_elem)" in pair_ids_bridge

    dispatch = slice_between(
        mmvq,
        "if (!used_direct_xmx && weight_layout == GGML_LAYOUT_XMX_TILED_BUNDLE4)",
        "if (!used_direct_xmx && !used_xmx_tiled_dpas && weight_layout == GGML_LAYOUT_XMX_TILED)",
    )
    assert "total_batches > 0" in dispatch
    assert "ne12 == 1" in dispatch
    assert "n_gpu_entries == total_batches" in dispatch
    assert "!ids_host" in dispatch
    assert "GGML_LAYOUT_XMX_TILED_BUNDLE4" in dispatch
    assert "mxfp4_pair_glu_xmx_tiled_bundle4_dpas_m2_submit" in dispatch
