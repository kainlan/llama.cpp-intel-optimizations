#!/usr/bin/env python3
"""Every device submit on the prompt (window) path is either wrapped by the kernel profiler or listed here.

A submit the profiler cannot see lands in "device idle" or in nothing, so the attribution table of a prompt ubatch is
only as complete as this list is short. The list shrinks to empty; it never grows without a reason beside the entry.

A raw submit is a `->submit(` or `.submit(` call that is not inside the argument list of a profiler wrapper call, at
any nesting depth: `ggml_sycl_profile_submit(`, or a `*_profile_submit` helper whose body forwards to it (found from
the source, not listed by hand). Comments, string and character literals are blanked first, so a submit named in prose
is not counted. The census is keyed by `file:function` with a count per key: a line moving does not change it, and a
second raw submit added to a function that is already listed does. A submit inside a helper that its callers wrap is
still raw here, because the helper's text cannot know its callers; its entry says which caller wraps it.

Only `submit(` is censused. A kernel launched with `queue.parallel_for(` or a copy with `queue.memcpy(` bypasses
`submit(` and is not seen by this pin.

Run directly or under pytest; `-s` prints the `file:line function` list, and a failing census prints it too.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml" / "src" / "ggml-sycl"
PROFILER_HEADER = "sycl-kernel-profiler.hpp"

# The files a prompt ubatch of the window path runs through. The graph is the qwen4exp one (src/models/qwen4exp.cpp,
# src/models/delta-net-base.cpp); each op it builds is dispatched by ggml_sycl_compute_forward in ggml-sycl.cpp, and
# the file named is where that op's forward function, or a callee it reaches with more than one row, is defined. A
# header counts as a file of its own: its submits belong to every translation unit that includes it.
FILES = [
    # the dispatcher, and the forward functions of MUL_MAT, MUL_MAT_ID (the window executor), GET_ROWS, RMS_NORM,
    # SCALE, SUM_ROWS, ARGSORT and TOP_K
    "ggml-sycl.cpp",
    # ggml_sycl_submit_marker: the window's timing marks (moe-stream-plan.cpp) and the GEMM span marks of the executor
    "common.hpp",
    "binbcast.cpp",  # ADD, MUL, SUB, DIV, REPEAT
    "element_wise.cpp",  # UNARY (sigmoid, silu, softplus, relu, abs, sgn, sqrt, exp, neg), CLAMP, GLU (swiglu)
    "concat.cpp",  # CONCAT
    "cpy.cpp",  # CPY, CONT
    "dsv4-hc.cpp",  # DSV4_HC_PRE (ggml_dsv4_hc_pre_gated), DSV4_HC_POST
    "fill.cpp",  # FILL
    "rope.cpp",  # ROPE (ggml_rope_multi)
    "set_rows.cpp",  # SET_ROWS (KV writes)
    "ssm_conv.cpp",  # SSM_CONV
    "cumsum.cpp",  # CUMSUM
    "gated_delta_net.cpp",  # GATED_DELTA_NET
    "pad.cpp",  # PAD
    "set.cpp",  # SET (ggml_set_inplace)
    "solve_tri.cpp",  # SOLVE_TRI
    "softmax.cpp",  # SOFT_MAX (the MoE gate's softmax)
    "norm.cpp",  # ggml_sycl_op_rms_norm, called by ggml_sycl_rms_norm
    "getrows.cpp",  # ggml_sycl_op_get_rows, called by ggml_sycl_get_rows
    # FLASH_ATTN_EXT: fattn.cpp picks the kernel per shape, so every kernel file of its translation unit is counted
    "fattn.cpp",
    "fattn-onednn.cpp",  # ggml_sycl_flash_attn_ext_onednn, the default prompt route
    "fattn-dpas.cpp",
    "fattn-tile.cpp",
    "fattn-esimd-f16.hpp",
    "fattn-mma.hpp",
    "fattn-tile-f16.hpp",
    "fattn-tile.hpp",
    "fattn-v2-esimd.hpp",
    "fattn-v2-partition.hpp",
    "fattn-vec-f16.hpp",
    "fattn-vec.hpp",
    "fattn-xmx-f16-v2.hpp",
    "fattn-xmx-f16.hpp",
    # MUL_MAT with more than one row: the unified kernel (dispatch.hpp ggml_sycl_mul_mat_unified_default), MMQ and its
    # XMX arm (ggml_sycl_op_mul_mat_q, ggml_sycl_op_mul_mat_q_xmx), the src1 quantize they read, and the dequantize to
    # f16 that the oneDNN GEMM reads
    "unified-kernel.cpp",
    "mmq.cpp",
    "mmq_xmx.cpp",
    "quantize.hpp",
    "convert.cpp",  # also the window executor's expert dequant (to_fp16, the Q8_0 SOA dequant)
    "gemm.hpp",  # DnnlGemmWrapper, the window executor's GEMMs
    "mmvq.cpp",  # a staged ring slot is served by the plain MMVQ_COMPAT kernel
    # the window itself: the ring's plan and copies, the copy helper it calls (mem_copy_async), and the executor's
    # scratch admission
    "moe-stream-plan.cpp",
    "moe-stream-ring.cpp",
    "moe-scratch-admission.cpp",
    "mem-ops.cpp",
]

# "file:function" -> (raw submits in it, why they are not wrapped yet). A named lambda is `function/lambda`.
ALLOW: dict[str, tuple[int, str]] = {
    "common.hpp:ggml_sycl_submit_marker": (1, "empty ordering marker on every caller's behalf; no work time"),
    "convert.cpp:dequantize_row_iq1_m_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq1_s_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq2_s_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq2_xs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq2_xxs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq3_s_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq3_xxs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq4_nl_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_iq4_xs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_q4_K_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "convert.cpp:dequantize_row_q4_K_sycl_reorder":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "convert.cpp:reorder_mxfp4_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "convert.cpp:reorder_q4_0_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "convert.cpp:reorder_q6_k_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "convert.cpp:reorder_q8_0_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "cumsum.cpp:ggml_sycl_op_cumsum": (1, "not yet wrapped"),
    "fattn.cpp:ggml_sycl_fattn_xmx_materialize_packed_k":
        (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn.cpp:ggml_sycl_fattn_xmx_submit_set_rows_update":
        (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn-mma.hpp:launch_fattn_mma": (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn-tile-f16.hpp:launch_fattn_tile_f16": (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn-v2-partition.hpp:launch_paged_attention_v2":
        (2, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn-vec-f16.hpp:launch_fattn_vec_f16": (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn-vec.hpp:launch_fattn_vec": (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "fattn-xmx-f16-v2.hpp:launch_fattn_xmx_v2_decode_gqa_split_leaf":
        (2, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "ggml-sycl.cpp:argmax_f32_i32_sycl": (1, "not yet wrapped"),
    "ggml-sycl.cpp:argsort_f32_i32_sycl": (2, "not yet wrapped"),
    "ggml-sycl.cpp:dispatch_experts_secondary_gpu_impl": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_add_f32": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_copy_ids_to_host": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_dispatch_host_flash_attn_async": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_dispatch_host_flash_attn_sync": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_moe_down_weighted_sum_sycl": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_mul_mat_batched_sycl": (2, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_mul_mat_id": (4, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_mul_mat_id/copy_src1_current_f32": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_mul_mat_tp_row_parallel_post": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_router_f32_bias_argsort_sycl": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_moe_gather_activation_rows": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_moe_scatter_rows": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_mul_mat_unpin_event": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_q8_1_soa_repeat": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_q8_1_soa_selected": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_queue_sync_event": (2, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_secondary_moe_glu": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_submit_secondary_moe_weighted_reduce": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_update_moe_ptr_table": (1, "not yet wrapped"),
    "ggml-sycl.cpp:ggml_sycl_upload_moe_transient_ptr_table": (1, "not yet wrapped"),
    "ggml-sycl.cpp:moe_cpu_job_join_on_queue": (1, "not yet wrapped"),
    "ggml-sycl.cpp:reorder_mxfp4_gpu": (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "ggml-sycl.cpp:reorder_q4_0_gpu": (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "ggml-sycl.cpp:reorder_q4_k_gpu": (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "ggml-sycl.cpp:reorder_q6_k_gpu": (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "ggml-sycl.cpp:reorder_q8_0_gpu": (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "ggml-sycl.cpp:sycl_reorder_parallel_for":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "ggml-sycl.cpp:test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing":
        (1, "test or benchmark entry point; graph compute does not call it"),
    "ggml-sycl.cpp:try_xmx_sorted_moe": (3, "not yet wrapped"),
    "mem-ops.cpp:mem_copy_ptr_async": (1, "not yet wrapped"),
    "mem-ops.cpp:mem_fill_direct_submit": (1, "not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q2_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q3_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_colfused_4tile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_colfused_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_largetile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_multitile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_1_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q4_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q5_0_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q5_1_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q5_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q6_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q8_0_q8_1_xmx_largetile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:ggml_mul_mat_q8_0_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "mmq_xmx.cpp:xmx_test_kernel": (1, "test or benchmark entry point; graph compute does not call it"),
    "mmvq.cpp:coalesced_mul_mat_vec_mxfp4_q8_1_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:convert_mxfp4_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:convert_q4_0_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:convert_q6_k_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:convert_q8_0_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:ggml_sycl_build_moe_compact_list":
        (2, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mmvq_submit_aos_impl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mmvq_submit_mxfp4_soa": (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mmvq_submit_q4_0_soa":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mmvq_submit_q6_k_soa":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mmvq_submit_quantize_q8_1_aos_after":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mmvq_submit_quantize_q8_1_soa":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq1_m_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq1_s_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq2_s_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq2_xs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq2_xxs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq3_s_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq3_xxs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq4_nl_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_iq4_xs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_mxfp4_q8_1_sycl": (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mul_mat_vec_q2_0_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q2_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q3_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q4_1_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q4_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q5_0_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q5_1_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q5_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mul_mat_vec_q6_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "mmvq.cpp:mxfp4_build_grouped_metadata_from_ids_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_down_sum_direct_final_zero_dst_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_atomic_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_scratch_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_scratch_reduce_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_dpas_down_single_col_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_dpas_i8_rowmajor_single_col_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_i8_grouped_direct_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_i8_grouped_direct_q8_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_i8_grouped_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_i8_grouped_packed_q8_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_moe_gateup_prepack_selected_rows_submit":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_gateup_prepack_dpas_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_multirhs_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_singlecol_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_soa_dpas_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_split_q8_submit/submit_glu_q8":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_split_submit/submit_glu":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_m2_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_packed_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_xmx_tiled_grouped_direct_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_xmx_tiled_grouped_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:mxfp4_xmx_tiled_grouped_weighted_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "mmvq.cpp:reorder_mul_mat_vec_mxfp4_i8_q8_1_id_sycl_rows/submit_mmv":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_xmx_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_xmx_tiled_id_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "mmvq.cpp:variable_tile_mul_mat_vec_q6_k_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "norm.cpp:ggml_sycl_op_rms_norm_back": (1, "not yet wrapped"),
    "norm.cpp:l2_norm_f32_sycl": (2, "not yet wrapped"),
    "quantize.hpp:quantize_row_q8_0_sycl": (1, "src1 quantize helper, not yet wrapped"),
    "quantize.hpp:quantize_row_q8_1_sycl": (1, "src1 quantize helper, not yet wrapped"),
    "solve_tri.cpp:ggml_sycl_op_solve_tri": (1, "not yet wrapped"),
    "ssm_conv.cpp:kernel_ssm_conv": (1, "not yet wrapped"),
    "unified-kernel.cpp:dispatch_xmx_kernel": (3, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:launch_unified_matmul": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:micro_submit_add": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:micro_submit_attention": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:micro_submit_rms_norm": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:micro_submit_silu_mul": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:micro_submit_strided_copy": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:mmvq_submit_silu_mul": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:UnifiedKernel::benchmark_graph_overhead":
        (7, "test or benchmark entry point; graph compute does not call it"),
    "unified-kernel.cpp:UnifiedKernel::execute_persistent_phased": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:UnifiedKernel::launch_persistent_kernel_async": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:UnifiedKernel::launch_persistent_kernel/run_persistent_kernel":
        (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:UnifiedKernel::record_micro_graph": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:UnifiedKernel::rms_norm": (1, "unified kernel launch, not yet wrapped"),
    "unified-kernel.cpp:UnifiedKernel::silu_mul": (1, "unified kernel launch, not yet wrapped"),
}

BASE_WRAPPER = "ggml_sycl_profile_submit"

SUBMIT = re.compile(r"(?:->|\.)\s*submit\s*\(")

# A comment, a raw string, a string or a character literal. Each is blanked to spaces with its newlines kept, so
# offsets and line numbers survive and nothing inside one is read as code, a parenthesis included.
NOT_CODE = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r"|\bR\"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)\""
    r"|\"(?:\\.|[^\"\\\n])*\""
    r"|'(?:\\.|[^'\\\n])+'",
    re.S,
)
NAME_BEFORE_CALL = re.compile(r"(\w+)\s*(?:<[^<>;{}()]*>\s*)?$")
NAME_BEFORE_PARAMS = re.compile(r"((?:~?\w+\s*::\s*)*~?\w+)\s*(?:<[^<>;{}()]*>\s*)?$")
OPERATOR_BEFORE_PARAMS = re.compile(r"((?:\w+\s*::\s*)*operator\s*(?:\(\s*\)|[^\w\s(]+))\s*$")
# what may sit between a parameter list's ')' and its body's '{'
AFTER_PARAMS = re.compile(
    r"\)\s*(?:(?:const|volatile|noexcept|override|final|mutable|try|&&|&)\s*|noexcept\s*\([^()]*\)\s*"
    r"|->\s*[\w:<>,\s\*&]+?\s*)*$"
)
NOT_A_FUNCTION = {"if", "for", "while", "switch", "catch", "return", "sizeof", "decltype", "alignof", "noexcept",
                  "static_assert", "defined", "__attribute__"}
CLASS_HEAD = re.compile(r"\b(?:struct|class|union)\s+(?:alignas\s*\([^()]*\)\s*)?(\w+)[^;{}()]*$")
NAMED_LAMBDA = re.compile(r"\b(\w+)\s*=\s*$")


CONDITIONAL = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b")


def blank_alternative_branches(code: str) -> str:
    """Blanks the #elif / #else branches of each conditional group, so brackets are counted over one branch: code like
    `#ifdef A for (...) { #else for (...) { #endif` opens one block, not two. A branch that holds a submit is kept, so
    nothing is hidden; if it also unbalances the brackets, the balance test names the file."""
    lines = code.split("\n")
    groups: list[list[list[int]]] = []  # per open group, its branches as lists of line indices
    drop: set[int] = set()
    for i, text in enumerate(lines):
        m = CONDITIONAL.match(text)
        if m and m.group(1) in ("if", "ifdef", "ifndef"):
            groups.append([[]])
        elif m and m.group(1) in ("elif", "else") and groups:
            groups[-1].append([])
        elif m and m.group(1) == "endif" and groups:
            for branch in groups.pop()[1:]:
                if not any(SUBMIT.search(lines[j]) for j in branch):
                    drop.update(branch)
        elif groups:
            groups[-1][-1].append(i)
            for outer in groups[:-1]:
                outer[-1].append(i)
    return "\n".join(re.sub(r"[^\n]", " ", t) if i in drop else t for i, t in enumerate(lines))


def blank(src: str) -> str:
    return blank_alternative_branches(NOT_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src))


class Code:
    """Blanked source with its bracket structure: which '(' each ')' closes, and the open '(' and '{' at each submit."""

    def __init__(self, src: str):
        self.code = blank(src)
        self.paren_open: dict[int, int] = {}
        self.brace_close: dict[int, int] = {}
        self.unbalanced: list[str] = []
        self.submits: list[tuple[int, list[int], list[int]]] = []
        starts = [m.start() for m in SUBMIT.finditer(self.code)]
        parens: list[int] = []
        braces: list[int] = []
        si = 0
        for m in re.finditer(r"[(){}]", self.code):
            pos, c = m.start(), m.group(0)
            while si < len(starts) and starts[si] < pos:
                self.submits.append((starts[si], list(parens), list(braces)))
                si += 1
            if c == "(":
                parens.append(pos)
            elif c == "{":
                braces.append(pos)
            elif c == ")":
                if parens:
                    self.paren_open[pos] = parens.pop()
                else:
                    self.unbalanced.append(f"')' at line {self.line(pos)} closes nothing")
            elif braces:
                self.brace_close[braces.pop()] = pos
            else:
                self.unbalanced.append(f"'}}' at line {self.line(pos)} closes nothing")
        while si < len(starts):
            self.submits.append((starts[si], list(parens), list(braces)))
            si += 1
        self.unbalanced += [f"'(' at line {self.line(p)} is never closed" for p in parens]
        self.unbalanced += [f"'{{' at line {self.line(p)} is never closed" for p in braces]

    def line(self, pos: int) -> int:
        return self.code.count("\n", 0, pos) + 1

    def callee(self, open_paren: int) -> str | None:
        m = NAME_BEFORE_CALL.search(self.code, max(0, open_paren - 300), open_paren)
        return m.group(1) if m else None

    def lambda_name(self, bracket_close: int) -> str | None:
        """`auto name = [...]`: the name, else None. bracket_close is the ']' that ends the capture list."""
        depth = 0
        i = bracket_close
        while i >= 0:
            c = self.code[i]
            if c == "]":
                depth += 1
            elif c == "[":
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        m = NAMED_LAMBDA.search(self.code, max(0, i - 200), max(0, i))
        return m.group(1) if m else None

    def brace_kind(self, pos: int) -> tuple[str, str | None]:
        """What the '{' at pos opens: function, lambda or class, with its name, or a plain block."""
        lo = max(0, pos - 2000)
        head = self.code[lo:pos].rstrip()
        end = lo + len(head)
        if not head:
            return "block", None
        m = CLASS_HEAD.search(head)
        if m:
            return "class", m.group(1)
        if head.endswith("]"):
            return "lambda", self.lambda_name(end - 1)
        t = AFTER_PARAMS.search(head)
        if not t:
            return "block", None
        close = lo + t.start()
        while True:
            open_ = self.paren_open.get(close)
            if open_ is None:
                return "block", None
            before_lo = max(0, open_ - 400)
            before = self.code[before_lo:open_].rstrip()
            # `[...](` or a template lambda `[...]<T>(`
            if before.endswith("]") or re.search(r"\]\s*<[^<>]*>$", before):
                return "lambda", self.lambda_name(before_lo + before.rfind("]"))
            op = OPERATOR_BEFORE_PARAMS.search(before)
            if op:
                return "function", re.sub(r"\s+", "", op.group(1))
            name = NAME_BEFORE_PARAMS.search(before)
            if not name:
                return "block", None
            ident = re.sub(r"\s+", "", name.group(1))
            if ident.split("::")[-1] in NOT_A_FUNCTION:
                return "block", None
            # a member initializer, `: x(y)` or `, x(y)`: the constructor's own parameter list is further back
            pre = before[:name.start()].rstrip()
            if (pre.endswith(":") and not pre.endswith("::")) or pre.endswith(","):
                prev = pre[:-1].rstrip()
                if prev.endswith(")"):
                    close = before_lo + len(prev) - 1
                    continue
                return "block", None
            return "function", ident

    def enclosing(self, braces: list[int]) -> str:
        """The innermost named function around a submit, `Class::` added for a member defined in its class, and
        `/name` for each named lambda between them."""
        func = None
        lambdas: list[str] = []
        for b in reversed(braces):
            kind, name = self.brace_kind(b)
            if func is None:
                if kind == "lambda" and name:
                    lambdas.append(name)
                elif kind == "function":
                    func = name
            elif kind == "class" and "::" not in func:
                func = f"{name}::{func}"
                break
            elif kind == "function":
                break
        if func is None:
            return "?"
        return "/".join([func] + list(reversed(lambdas)))


_PARSED: dict[str, Code] = {}


def parse(src: str) -> Code:
    """Code for a text, kept per text: the controls census the tree again with one file changed, and the other files,
    ggml-sycl.cpp above all, need not be parsed a second time."""
    if src not in _PARSED:
        _PARSED[src] = Code(src)
    return _PARSED[src]


def derive_wrappers(texts: dict[str, str]) -> set[str]:
    """ggml_sycl_profile_submit, and every `*_profile_submit` function defined in the texts whose body calls it."""
    wrappers = {BASE_WRAPPER}
    for src in texts.values():
        code = parse(src)
        for m in re.finditer(r"\b(\w+_profile_submit)\s*\(", code.code):
            if m.group(1) == BASE_WRAPPER:
                continue
            close = next((c for c, o in code.paren_open.items() if o == m.end() - 1), None)
            if close is None:
                continue
            body = re.match(r"\s*(?:const\s*)?\{", code.code[close + 1:])
            if not body:
                continue
            start = close + body.end()
            end = code.brace_close.get(start)
            if end is not None and re.search(rf"\b{BASE_WRAPPER}\s*\(", code.code[start:end]):
                wrappers.add(m.group(1))
    return wrappers


def census(texts: dict[str, str], wrappers: set[str] | None = None) -> dict[str, list[tuple[int, str]]]:
    """The raw submits of each named text, as (line, function) pairs. A submit is wrapped when any '(' still open at
    it, however far back and however deeply nested, is the argument list of a wrapper call."""
    if wrappers is None:
        wrappers = derive_wrappers(texts)
    found: dict[str, list[tuple[int, str]]] = {}
    for name, src in texts.items():
        code = parse(src)
        for pos, parens, braces in code.submits:
            if any(code.callee(p) in wrappers for p in parens):
                continue
            found.setdefault(name, []).append((code.line(pos), code.enclosing(braces)))
    return found


def census_keys(found: dict[str, list[tuple[int, str]]]) -> dict[str, int]:
    keys: dict[str, int] = {}
    for name, sites in found.items():
        for _, func in sites:
            key = f"{name}:{func}"
            keys[key] = keys.get(key, 0) + 1
    return keys


def diff_against_allow(keys: dict[str, int], allow: dict[str, tuple[int, str]]) -> list[str]:
    problems = []
    for key in sorted(set(keys) | set(allow)):
        have = keys.get(key, 0)
        want = allow[key][0] if key in allow else 0
        if have != want:
            problems.append(f"{key}: {have} raw submit(s) in the tree, {want} listed")
    return problems


def read_tree() -> dict[str, str]:
    texts = {}
    for name in FILES + [PROFILER_HEADER]:
        path = SYCL / name
        assert path.exists(), f"{name} is gone: update FILES"
        texts[name] = path.read_text(encoding="utf-8")
    return texts


def census_tree(texts: dict[str, str]) -> dict[str, list[tuple[int, str]]]:
    wrappers = derive_wrappers(texts)
    return census({name: texts[name] for name in FILES}, wrappers)


def format_sites(found: dict[str, list[tuple[int, str]]]) -> str:
    return "\n".join(f"{name}:{line} {func}" for name in sorted(found) for line, func in found[name])


def raw_functions(src: str, wrappers: set[str] | None = None) -> list[str]:
    return [func for _, func in census({"x.cpp": src}, wrappers).get("x.cpp", [])]


def plan_window_detector_says_wrapped(src: str) -> bool:
    """The detector the plan first proposed: wrapped when the nearest unclosed '(' within 400 characters before the
    submit is the profiler's. Kept only to show the case below is one it gets wrong."""
    code = blank(src)
    m = SUBMIT.search(code)
    head = code[max(0, m.start() - 400):m.start()]
    depth, i = 0, len(head) - 1
    while i >= 0:
        c = head[i]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth == 0:
                return head[:i].rstrip().endswith(BASE_WRAPPER)
            depth -= 1
        i -= 1
    return False


def test_one_wrapped_and_one_raw_submit_yield_exactly_the_raw_one():
    sample = (
        "sycl::event a(sycl::queue * q) {\n"
        "    return ggml_sycl_profile_submit(*q, label, [&](sycl::queue & pq) {\n"
        "        return pq.submit([&](sycl::handler & cgh) { cgh.single_task([] {}); });\n"
        "    });\n"
        "}\n"
        "void b(sycl::queue * q) { q->submit([](sycl::handler &) {}); }\n"
    )
    assert raw_functions(sample) == ["b"]


def test_a_submit_in_a_comment_or_a_literal_is_not_counted():
    sample = (
        "void a(sycl::queue * q) {\n"
        "    // q->submit([&](sycl::handler & cgh) {});\n"
        "    /* q.submit( */\n"
        "    log(\"q->submit(\");\n"
        "    log(R\"x(q.submit()x\");\n"
        "    char open = '(';\n"
        "    ggml_sycl_profile_submit(*q, label(\"(\"), [&](sycl::queue & pq) { return pq.submit(cgf); });\n"
        "}\n"
        "void b(sycl::queue * q) { log(\")\"); q->submit(cgf); }\n"
    )
    # the '(' in the character literal and the parentheses in strings must not move a wrapper boundary either
    assert raw_functions(sample) == ["b"]


def test_a_wrapped_submit_behind_a_long_lambda_is_still_wrapped():
    filler = "".join(f"        int v{i} = f{i}(x, g(y));\n" for i in range(160))
    sample = (
        "sycl::event a(sycl::queue * q) {\n"
        "    return ggml_sycl_profile_submit(*q, label, [&](sycl::queue & pq) {\n"
        + filler
        + "        return finish(pq, record(pq.submit([&](sycl::handler & cgh) { cgh.single_task([] {}); })));\n"
        "    });\n"
        "}\n"
    )
    assert len(filler) > 4000
    # the nearest unclosed call is record(, so a detector that stops there, or looks back a fixed window, misreads it
    assert not plan_window_detector_says_wrapped(sample)
    assert raw_functions(sample) == []
    # and the same text under a call that is not a wrapper is raw, so the empty answer above is not a blind scanner
    assert raw_functions(sample.replace(BASE_WRAPPER, "not_a_wrapper")) == ["a"]


def test_a_helper_that_forwards_to_the_profiler_is_a_wrapper_and_one_that_does_not_is_not():
    helpers = (
        "template <typename F> static sycl::event x_profile_submit(sycl::queue & q, F && f) {\n"
        "    if (!enabled()) { return f(q); }\n"
        "    return ggml_sycl_profile_submit(q, label, static_cast<F &&>(f));\n"
        "}\n"
        "template <typename F> static sycl::event y_profile_submit(sycl::queue & q, F && f) { return f(q); }\n"
        "void a(sycl::queue & q) { x_profile_submit(q, [&](sycl::queue & pq) { return pq.submit(cgf); }); }\n"
        "void b(sycl::queue & q) { y_profile_submit(q, [&](sycl::queue & pq) { return pq.submit(cgf); }); }\n"
    )
    wrappers = derive_wrappers({"x.cpp": helpers})
    assert wrappers == {BASE_WRAPPER, "x_profile_submit"}
    assert raw_functions(helpers, wrappers) == ["b"]


def test_the_tree_derives_the_known_wrappers():
    assert derive_wrappers(read_tree()) == {
        BASE_WRAPPER,
        "ggml_sycl_memcpy_profile_submit",
        "ggml_sycl_set_rows_profile_submit",
    }


def test_the_enclosing_function_is_named_for_every_definition_shape():
    sample = (
        "template <typename T, int N>\n"
        "static sycl::event templated(sycl::queue & q,\n"
        "                             const T *     x) {\n"
        "    return q.submit(cgf);\n"
        "}\n"
        "foo::foo(sycl::queue & q) : q_(q), n_(f(1)) { q.submit(cgf); }\n"
        "auto trailing(sycl::queue & q) -> sycl::event { return q.submit(cgf); }\n"
        "void tried(sycl::queue & q) try { q.submit(cgf); } catch (...) {}\n"
        "struct holder : base<int> {\n"
        "    sycl::event member(sycl::queue & q) const noexcept { return q.submit(cgf); }\n"
        "    sycl::event operator()(sycl::queue & q) { return q.submit(cgf); }\n"
        "};\n"
        "void holder::outside(sycl::queue & q) { if (x) { for (;;) { q.submit(cgf); } } }\n"
        "void outer(sycl::queue & q) {\n"
        "    auto named = [&](int i) -> bool { q.submit(cgf); return true; };\n"
        "    run([&] { q.submit(cgf); });\n"
        "}\n"
    )
    assert raw_functions(sample) == [
        "templated",
        "foo::foo",
        "trailing",
        "tried",
        "holder::member",
        "holder::operator()",
        "holder::outside",
        "outer/named",
        "outer",
    ]


def test_every_file_of_the_path_has_balanced_brackets_and_a_name_for_every_site():
    texts = read_tree()
    for name in FILES:
        assert not parse(texts[name]).unbalanced, f"{name}: {parse(texts[name]).unbalanced[:3]}"
    found = census_tree(texts)
    unnamed = [f"{name}:{line}" for name, sites in found.items() for line, func in sites if func == "?"]
    assert found and not unnamed, f"no census, or sites with no enclosing function: {unnamed}"


def test_a_raw_submit_added_to_a_real_file_is_reported():
    texts = read_tree()
    probe = "\nstatic void census_probe_added(sycl::queue * q) { q->submit([&](sycl::handler & cgh) {}); }\n"
    texts["norm.cpp"] = texts["norm.cpp"] + probe
    problems = diff_against_allow(census_keys(census_tree(texts)), ALLOW)
    assert problems == ["norm.cpp:census_probe_added: 1 raw submit(s) in the tree, 0 listed"]


def test_a_second_raw_submit_in_a_listed_function_is_reported():
    texts = read_tree()
    found = census_tree(texts)
    assert found, "no raw submit found in the tree"
    name, (line, func) = sorted(found.items())[0][0], sorted(found.items())[0][1][0]
    lines = texts[name].split("\n")
    lines.insert(line, "    q->submit([&](sycl::handler & cgh) {});")
    texts[name] = "\n".join(lines)
    keys = census_keys(census_tree(texts))
    allow = {key: (count, "listed") for key, count in census_keys(found).items()}
    listed = allow[f"{name}:{func}"][0]
    expected = f"{name}:{func}: {listed + 1} raw submit(s) in the tree, {listed} listed"
    assert diff_against_allow(keys, allow) == [expected]


def test_a_listed_function_whose_submits_are_gone_fails():
    texts = read_tree()
    found = census_tree(texts)
    assert found, "no raw submit found in the tree"
    allow = {key: (count, "listed") for key, count in census_keys(found).items()}
    name = sorted(found)[0]
    func = found[name][0][1]
    lines = texts[name].split("\n")
    for line, f in found[name]:
        if f == func:
            lines[line - 1] = SUBMIT.sub(".submitted_elsewhere(", lines[line - 1])
    texts[name] = "\n".join(lines)
    problems = diff_against_allow(census_keys(census_tree(texts)), allow)
    assert problems == [f"{name}:{func}: 0 raw submit(s) in the tree, {allow[f'{name}:{func}'][0]} listed"]


def test_every_allowlist_entry_has_a_reason():
    assert all(count > 0 and reason.strip() for count, reason in ALLOW.values())


def test_census_matches_the_allowlist():
    t0 = time.monotonic()
    texts = read_tree()
    found = census_tree(texts)
    elapsed = time.monotonic() - t0
    print(format_sites(found))
    print(f"{sum(len(s) for s in found.values())} raw submit(s) in {len(census_keys(found))} function(s) "
          f"over {len(FILES)} file(s), census took {elapsed:.2f} s")
    problems = diff_against_allow(census_keys(found), ALLOW)
    assert not problems, "\n".join(problems) + "\n\nraw submits found:\n" + format_sites(found)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
