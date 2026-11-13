#!/usr/bin/env python3
"""Every device launch on the prompt (window) path is either wrapped by the kernel profiler or listed here.

A launch the profiler cannot see lands in "device idle" or in nothing, so the attribution table of a prompt ubatch is
only as complete as this list is short. The list shrinks to empty; it never grows without a reason beside the entry.

Two kinds of launch are censused, and kept apart in the output and in the allowlist keys:
- submit: a `->submit(` or `.submit(` call. A submit whose command group is only a `host_task` counts too: it occupies
  the in-order device queue, and the device sits idle while it runs, so the profiler must see it like a kernel.
- shortcut: a queue member that enqueues a command without submit(): `parallel_for`, `single_task`, `memcpy`, `memset`,
  `fill`, `copy`, `prefetch`, `mem_advise`, in member-call form only (`->name(` / `.name(`), so `std::memcpy(` and free
  functions never match. Two rules decide whether a member call is a queue launch:
  1. Inside a submit's parentheses it is the command group's handler call, and the submit is the launch: not counted.
  2. Elsewhere, the receiver's declared type decides. The receiver is the name before the `.`/`->` (`stream`), or a
     call (`ctx.stream()`). Its type is read from the nearest declaration before the call in the same function,
     lambda parameters included (`T name`, `T & name`, `T * const name`; `T name(` for a call); failing that, from
     the file; failing that, from any file read here (a member declared in the class header beside a .cpp). A queue
     is `sycl::queue`, `queue_ptr` or `dpct::queue_ptr`, by value, reference or pointer: counted. `sycl::handler`
     (a helper that takes the command group's handler) and any other type (`std::array::fill`) are not. A receiver
     with no declaration found, or declared `auto`, fails the pin, so a queue cannot hide behind an unread type.

A launch is profiled, and not a site, in two cases:
- Wrapped: any '(' still open at it, at any nesting depth and any distance, is the argument list of a wrapper call.
  The wrappers are `ggml_sycl_profile_submit` and every function that hands one of its own forwarding-reference
  parameters (`F && f`) on as the command group: as the wrapper call's command-group argument (`f`,
  `static_cast<F &&>(f)`, `std::forward<F>(f)`), or as the whole argument of a `submit(` inside the call's arguments,
  a lambda body included (mmq_submit). A forward anywhere else (a label, a log call) makes nothing a wrapper. Found
  from the source and closed transitively, whatever the name.
- Recorded: its function is a row of RECORDED_ON_THE_PATH, and the launch is lexically followed in the same
  function body by a call to `ggml_sycl_kernel_profile_record_event` or `ggml_sycl_profile_record_returned_event`
  that receives its event (Code.recorded): the launch is that call's argument, or it is assigned to a plain local
  `ev` that a later record call is handed (or `std::move(ev)`) as its event. A record call under a condition (the
  profiler being on) is accepted. The rule does not follow paths; the table row is what vouches for them, so a launch
  in a function with no row is reported whatever the code around it looks like, and a row excuses only when the rule
  matches exactly its count of launches in that function (a new launch, or a stale row, fails). Adding a row is the
  reviewed act: read the function and confirm the event each record call receives is that launch's event first.
Comments, string and character literals are blanked first, so a launch named in prose is not counted. The #elif and
#else branches of a conditional are blanked only when they hold no call of either class, so that braces opened in
both branches count once; a branch holding a launch is kept whatever its condition, a dead one included (its entry
says so). The census is keyed by `kind:file:function` with a count per key: a line moving does not change it, and a
second raw launch added to a function that is already listed does. A launch inside a helper that its callers wrap is
still raw here, because the helper's text cannot know its callers; its entry says which caller wraps it.

FILES membership: a file is in FILES when a call from the qwen4exp prompt graph reaches it with more than one row on a
default route, and the comment beside it cites the dispatch site. An opt-in route may also be listed when its comment
and its entries name it as one (the XMX sorted MoE route), so that it stays censused. Every other source under
ggml/src/ggml-sycl that holds a launch-like call is in EXCLUDED_FILES (or under an EXCLUDED_DIRS prefix) with its
reason; that table is the source of truth, and test_every_launching_file_is_on_the_path_or_excluded fails on a file in
neither. Challenge an exclusion by citing the call that reaches the file; then move it to FILES and its launches to
ALLOW.

Run directly or under pytest; `-s` prints the `file:line function kind` list and the counts per kind and file, and a
failing census prints the list too.
"""
from __future__ import annotations

import bisect
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
    # SCALE, SUM_ROWS and ARGSORT; TOP_K's forward function here only calls topk.cpp
    "ggml-sycl.cpp",
    # TOP_K: ggml_top_k (qwen4exp.cpp:813) -> ggml_sycl_top_k (ggml-sycl.cpp) -> ggml_sycl_op_top_k
    "topk.cpp",
    "diag.cpp",  # DIAG: ggml_diag (delta-net-base.cpp:158) -> ggml_sycl_op_diag
    # hc_pre_launch, hc_comb_launch, hc_post_launch, called by dsv4-hc.cpp for DSV4_HC_PRE / DSV4_HC_POST
    # (qwen4exp.cpp:319 ggml_dsv4_hc_pre_gated, :364 ggml_dsv4_hc_post)
    "dsv4-hc-kernels.hpp",
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
    # an opt-in route, kept on the list so it stays censused: try_xmx_sorted_moe (ggml_sycl_mul_mat_id, ahead of the
    # window executor) refuses every batch unless GGML_SYCL_XMX_MOE_SORTED=1 ("diagnostic-only-current-sorted-wrapper"),
    # before its first submit; these are its token sort, XMX GEMMs and fused MXFP4 GEMMs (try_fused_xmx_moe_mxfp4_*)
    "moe-sort.hpp",
    "moe-xmx.hpp",
    "moe-xmx-fused.hpp",
    # ggml_sycl_mul_mat_id_fused, the batched-prefill fused ESIMD MoE (off only for GGML_SYCL_DISABLE_FUSED_MOE)
    "fused-moe-esimd.hpp",
    # moe_cpu_job_prologue (ggml-sycl.cpp) arms a host expert job's input marker through
    # unified_cache_cpu_job_input_marker_arm, which submits the marker on the device queue
    "unified-cache.cpp",
    # ExpertPredictor::predict_multi_layer, called by ggml_sycl_mul_mat_id while the predictor is active
    "expert-prefetch.cpp",
    # the CPU executor: should_dispatch_to_cpu (ggml-sycl.cpp) routes a MUL_MAT whose weight resolves to host, with no
    # opt-in, and the ops of a CPU-bound layer, to ggml_sycl_compute_forward_cpu; its submits to the device queue are
    # host_task bodies (a host_task on the device queue counts, see the docstring)
    "cpu-dispatch.cpp",
]

# Every other source under ggml/src/ggml-sycl that holds a launch-like call (either class, before any receiver or
# wrapper test), with the one reason it is not on the window path. test_every_launching_file_is_on_the_path_or_excluded
# fails on a file in neither list, so a new file, or a file a new call reaches, cannot go unread.
NOT_BUILT = "an op the qwen4exp graph does not build (qwen4exp.cpp, delta-net-base.cpp and llama-graph.cpp name none)"
NOT_INCLUDED = "no backend source includes it"
EXCLUDED_DIRS = {
    "tests/": "device tests and benchmarks; not compiled into the backend",
}
EXCLUDED_FILES = {
    "add-id.cpp": "ADD_ID: built only for expert biases (llama-graph.cpp build_moe_ffn *_exps_b); qwen4exp passes none",
    "col2im-1d.cpp": "COL2IM_1D: " + NOT_BUILT,
    "conv.cpp": "CONV_TRANSPOSE_1D: " + NOT_BUILT,
    "conv2d-dw.cpp": "CONV_2D_DW: " + NOT_BUILT,
    "conv2d-transpose.cpp": "CONV_TRANSPOSE_2D: " + NOT_BUILT,
    "conv2d.cpp": "CONV_2D: " + NOT_BUILT,
    "conv3d.cpp": "CONV_3D: " + NOT_BUILT,
    "count-equal.cpp": "COUNT_EQUAL: " + NOT_BUILT,
    "cross_entropy_loss.cpp": "CROSS_ENTROPY_LOSS (training): " + NOT_BUILT,
    "gla.cpp": "GATED_LINEAR_ATTN: " + NOT_BUILT,
    "im2col.cpp": "IM2COL: " + NOT_BUILT,
    "outprod.cpp": "OUT_PROD: " + NOT_BUILT,
    "pad_reflect_1d.cpp": "PAD_REFLECT_1D: " + NOT_BUILT,
    "pool.cpp": "POOL_1D, POOL_2D: " + NOT_BUILT,
    "repeat_back.cpp": "REPEAT_BACK: " + NOT_BUILT,
    "roll.cpp": "ROLL: " + NOT_BUILT,
    "ssm_scan.cpp": "SSM_SCAN: " + NOT_BUILT,
    "tsembd.cpp": "TIMESTEP_EMBEDDING: " + NOT_BUILT,
    "upscale.cpp": "UPSCALE: " + NOT_BUILT,
    "wkv.cpp": "RWKV_WKV6, RWKV_WKV7: " + NOT_BUILT,
    "lightning-indexer-kernel.hpp": "the lightning indexer (lightning-indexer.cpp): " + NOT_BUILT,
    "set_rows_paged.hpp": "SET_ROWS_PAGED; " + NOT_INCLUDED,
    "dmmv.cpp": "dequantize-mul-mat-vec, the one-row (decode) MUL_MAT route",
    "dmmv-esimd.hpp": "dequantize-mul-mat-vec in ESIMD, the one-row (decode) MUL_MAT route",
    "gpu-sampler.hpp": "sampling (ggml_sycl_sample_token*), after the graph",
    "cont-batching.hpp": "multi-sequence sampling (ggml_sycl_multi_seq_sample*), after the graph",
    "persistent-tg-kernel.cpp": "the persistent token-generation kernel, decode only",
    "fused-ffn.hpp": "opt-in GGML_SYCL_FFN_FUSION=1 (read before execute_ffn_fusion in graph compute); "
                     "can_fuse_ffn then admits Q4_0 gate/up pairs of at most 8 rows",
    "mmq-esimd.hpp": "opt-in GGML_SYCL_MMQ_ESIMD: mmq_esimd_version() is 0 when it is unset",
    "mmq-xmx.hpp": "opt-in GGML_SYCL_XMX_MMQ=1 (use_xmx_mmq_kernel, read by can_use_xmx_q8_0)",
    "convert-esimd.hpp": "opt-in GGML_SYCL_ESIMD_DEQUANT=1 (g_esimd_dequant_enabled in convert.cpp)",
    "dnnl-ops.hpp": "the oneDNN softmax, opt-in GGML_SYCL_ONEDNN_SOFTMAX (read in softmax.cpp)",
    "kv-cache-quant.hpp": "included by fattn.cpp, which calls neither launch_quantize_kv_fp8 nor launch_attention_fp8",
    "dpct/helper.hpp": "dpct_memcpy, dpct_memset and host_buffer: no backend source outside dpct/ calls them",
    "lazy-io-pool.cpp": "its submit is the read queue of the I/O pool (lazy_io_engine::queue), not a SYCL queue",
    "moe-fused-transaction.cpp": "on the path (mmvq_submit_retained_prompt_fusion -> submit_prompt_fusion), but its "
                                 "two submits are Transaction::submit and the submitter's submit, not SYCL queues; "
                                 "the prompt fusion's device work is submitted from mmvq.cpp",
    "moe-tile-convert.cpp": "MXFP4 to XMX-tiled weight conversion, run when a weight is materialized "
                            "(ggml_sycl_fill_xmx_tiled, convert_tensor_layout), not by an op of the graph",
    "moe-mmid-workspace.cpp": "no launch: its .fill( calls are std::array::fill in moe_mmid_workspace_registry::admit",
    "mxfp4-stored-gemm.cpp": "its header mxfp4-stored-gemm.hpp is included by no other backend source",
    "fused-norm-gemm.hpp": "included only by mmvq-rmsnorm.hpp, which " + NOT_INCLUDED,
    "mmvq-rmsnorm.hpp": NOT_INCLUDED,
    "l2-prefetch.hpp": NOT_INCLUDED,
    "mul-mat-bf16.hpp": NOT_INCLUDED,
    "quantized-comm.hpp": NOT_INCLUDED,
}

# shared reasons for the shortcut entries
SHORTCUT_KERNEL = "kernel launched by a queue shortcut (parallel_for / single_task), not yet labelled"
SHORTCUT_USM = "USM copy or fill launched by a queue shortcut, not yet labelled"
SHORTCUT_KERNEL_QK_K = (SHORTCUT_KERNEL + "; one per branch of #if QK_K == 256, and the #else branch is dead code "
                        "(ggml-common.h defines QK_K as 256)")
MOE_XMX = "opt-in GGML_SYCL_XMX_MOE_SORTED=1 (try_xmx_sorted_moe); kept on the list so the opt-in route stays censused"
CPU_HOST_TASK = ("a host-placed op run on the CPU as a host_task on the device queue (the in-order queue orders it "
                 "with the device kernels), not yet labelled")
FUSED_ESIMD = "the fused ESIMD MoE (ggml_sycl_mul_mat_id_fused, on by default), not yet wrapped"
FUSED_ESIMD_CASES = ("; one launch per case: the direct f32 kernel under graph recording or without Q8 scratch, "
                     "else DP4A")

# "kind:file:function" -> (raw sites of that kind in it, why they are not wrapped yet). A named lambda is
# `function/lambda`. The kind is `submit` or `shortcut`, so the two classes are driven down separately.
ALLOW: dict[str, tuple[int, str]] = {
    "submit:common.hpp:ggml_sycl_submit_marker": (1, "empty ordering marker on every caller's behalf; no work time"),
    "submit:convert.cpp:dequantize_row_iq1_m_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq1_s_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq2_s_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq2_xs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq2_xxs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq3_s_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq3_xxs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq4_nl_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_iq4_xs_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_q4_K_sycl":
        (1, "dequant helper; the window executor wraps its call (dequant_expert_f16), other callers do not"),
    "submit:convert.cpp:dequantize_row_q4_K_sycl_reorder":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:convert.cpp:reorder_mxfp4_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:convert.cpp:reorder_q4_0_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:convert.cpp:reorder_q6_k_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:convert.cpp:reorder_q8_0_aos_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:cumsum.cpp:ggml_sycl_op_cumsum": (1, "not yet wrapped"),
    "submit:fattn-mma.hpp:launch_fattn_mma": (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "submit:fattn-tile-f16.hpp:launch_fattn_tile_f16":
        (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "submit:fattn-v2-partition.hpp:launch_paged_attention_v2":
        (2, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "submit:fattn-vec-f16.hpp:launch_fattn_vec_f16":
        (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "submit:fattn-vec.hpp:launch_fattn_vec": (1, "attention kernel, not yet wrapped; fattn.cpp picks it per shape"),
    "submit:ggml-sycl.cpp:argmax_f32_i32_sycl": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:argsort_f32_i32_sycl": (2, "not yet wrapped"),
    "submit:ggml-sycl.cpp:dispatch_experts_secondary_gpu_impl": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_add_f32": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_copy_ids_to_host": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_dispatch_host_flash_attn_async": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_dispatch_host_flash_attn_sync": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_moe_down_weighted_sum_sycl": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_batched_sycl": (2, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_id": (
        4,
        "two are host_task submits of the Q1/NVFP4 exact route, ordering and completion rather than device kernels: "
        "the injected terminal failure (test-only, under GGML_SYCL_Q1_NVFP4_ROUTE_TESTING) and the completion that "
        "follows terminal->arm(final); two are prompt row copies, not yet wrapped: the MXFP4 SOA f16 activation "
        "staging (f16_staging_event) and the expert output scatter",
    ),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_id/copy_src1_current_f32": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_tp_row_parallel_post": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_router_f32_bias_argsort_sycl": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_moe_gather_activation_rows": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_moe_scatter_rows": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_mul_mat_unpin_event": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_q8_1_soa_repeat": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_q8_1_soa_selected": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_queue_sync_event": (2, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_secondary_moe_glu": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_secondary_moe_weighted_reduce": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_update_moe_ptr_table": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:ggml_sycl_upload_moe_transient_ptr_table": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:moe_cpu_job_join_on_queue": (1, "not yet wrapped"),
    "submit:ggml-sycl.cpp:reorder_mxfp4_gpu":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:ggml-sycl.cpp:reorder_q4_0_gpu":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:ggml-sycl.cpp:reorder_q4_k_gpu":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:ggml-sycl.cpp:reorder_q6_k_gpu":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:ggml-sycl.cpp:reorder_q8_0_gpu":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:ggml-sycl.cpp:sycl_reorder_parallel_for":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:ggml-sycl.cpp:test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing":
        (1, "test or benchmark entry point; graph compute does not call it"),
    "submit:ggml-sycl.cpp:try_xmx_sorted_moe": (
        3,
        MOE_XMX + ": the sort's dependency barrier and the two sorted_token_ids extracts",
    ),
    "submit:mem-ops.cpp:mem_copy_ptr_async": (1, "not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q2_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q3_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_colfused_4tile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_colfused_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_largetile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_multitile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_1_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q5_0_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q5_1_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q5_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q6_K_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q8_0_q8_1_xmx_largetile_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q8_0_q8_1_xmx_sycl": (1, "MMQ XMX kernel, not yet wrapped"),
    "submit:mmq_xmx.cpp:xmx_test_kernel": (1, "test or benchmark entry point; graph compute does not call it"),
    "submit:mmvq.cpp:coalesced_mul_mat_vec_mxfp4_q8_1_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:convert_mxfp4_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:convert_q4_0_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:convert_q6_k_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:convert_q8_0_to_coalesced_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:ggml_sycl_build_moe_compact_list":
        (2, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mmvq_submit_aos_impl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mmvq_submit_mxfp4_soa":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mmvq_submit_q4_0_soa":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mmvq_submit_q6_k_soa":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mmvq_submit_quantize_q8_1_aos_after":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mmvq_submit_quantize_q8_1_soa":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq1_m_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq1_s_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq2_s_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq2_xs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq2_xxs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq3_s_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq3_xxs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq4_nl_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_iq4_xs_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_mxfp4_q8_1_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mul_mat_vec_q2_0_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q2_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q3_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q4_1_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q4_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q5_0_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q5_1_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q5_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mul_mat_vec_q6_K_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:mmvq.cpp:mxfp4_build_grouped_metadata_from_ids_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_down_sum_direct_final_zero_dst_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_atomic_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_scratch_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_scratch_reduce_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_dpas_down_single_col_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_dpas_i8_rowmajor_single_col_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_i8_grouped_direct_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_i8_grouped_direct_q8_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_i8_grouped_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_i8_grouped_packed_q8_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_moe_gateup_prepack_selected_rows_submit":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_gateup_prepack_dpas_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_multirhs_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_singlecol_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_soa_dpas_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_split_q8_submit/submit_glu_q8":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_split_submit/submit_glu":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_m2_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_m4_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_packed_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_xmx_tiled_grouped_direct_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_xmx_tiled_grouped_direct_q8_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:mxfp4_xmx_tiled_grouped_weighted_q8_m2_sycl":
        (1, "MXFP4 kernel, not yet wrapped; the target packs hold no MXFP4 tensor"),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_i8_q8_1_id_sycl_rows/submit_mmv":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_xmx_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_xmx_tiled_id_sycl":
        (1, "weight layout conversion; runs when a weight is placed, not per ubatch"),
    "submit:mmvq.cpp:variable_tile_mul_mat_vec_q6_k_q8_1_sycl":
        (1, "vector kernel, not yet wrapped"),
    "submit:norm.cpp:ggml_sycl_op_rms_norm_back": (1, "not yet wrapped"),
    "submit:norm.cpp:l2_norm_f32_sycl": (2, "not yet wrapped"),
    "submit:quantize.hpp:quantize_row_q8_0_sycl": (1, "src1 quantize helper, not yet wrapped"),
    "submit:quantize.hpp:quantize_row_q8_1_sycl": (1, "src1 quantize helper, not yet wrapped"),
    "submit:solve_tri.cpp:ggml_sycl_op_solve_tri": (1, "not yet wrapped"),
    "submit:ssm_conv.cpp:kernel_ssm_conv": (1, "not yet wrapped"),
    "submit:unified-kernel.cpp:dispatch_xmx_kernel": (
        3,
        "debug only: the GGML_SYCL_XMX_DETAIL timing probes (real, empty and SLM-only kernels between q.wait() "
        "calls); the production launch below them in the same function is wrapped (ggml_sycl_profile_submit)",
    ),
    "submit:unified-kernel.cpp:launch_unified_matmul": (1, "unified kernel launch, not yet wrapped"),
    "submit:cpu-dispatch.cpp:cpu_submit_async":
        (1, "the CPU executor's submit helper: a kernel on the CPU queue, or a host_task on the device queue for the "
            "CPU matmul and fused paths (run_mul_mat, run_fused); not yet labelled"),
    "submit:cpu-dispatch.cpp:cpu_cpy": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_glu": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_norm": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_rms_norm": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_rope": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_scale": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_silu": (1, CPU_HOST_TASK),
    "submit:expert-prefetch.cpp:ExpertPredictor::predict_pregate":
        (1, "the expert predictor's gate GEMV, run while the predictor is active; not yet wrapped"),
    "submit:fused-moe-esimd.hpp:launch_fused_moe_mxfp4_impl": (3, FUSED_ESIMD + FUSED_ESIMD_CASES),
    "submit:fused-moe-esimd.hpp:launch_fused_moe_mxfp4_soa":
        (3, FUSED_ESIMD + "; one launch per hidden-size case (2880, 4096, any other)"),
    "submit:fused-moe-esimd.hpp:launch_fused_moe_q8_0": (1, FUSED_ESIMD),
    "submit:fused-moe-esimd.hpp:launch_persistent_moe_mxfp4_impl":
        (3, FUSED_ESIMD + "; the persistent variant, for at most 8 tokens" + FUSED_ESIMD_CASES),
    "submit:moe-sort.hpp:moe_compute_expert_offsets_gpu": (1, MOE_XMX),
    "submit:moe-sort.hpp:moe_compute_expert_offsets_gpu_simple": (1, MOE_XMX),
    "submit:moe-sort.hpp:moe_compute_tile_mapping": (1, MOE_XMX),
    "submit:moe-sort.hpp:moe_count_tokens_per_expert_async": (1, MOE_XMX),
    "submit:moe-sort.hpp:moe_scatter_results_f16_to_f32_async": (1, MOE_XMX),
    "submit:moe-sort.hpp:moe_sort_tokens_by_expert_async": (1, MOE_XMX),
    "submit:moe-xmx-fused.hpp:fused_xmx_moe_gemm_mxfp4_soa": (1, MOE_XMX),
    "submit:moe-xmx-fused.hpp:fused_xmx_moe_gemm_mxfp4_tiled": (1, MOE_XMX),
    "submit:moe-xmx-fused.hpp:fused_xmx_moe_gemm_q8_0": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_fused_xmx_moe_q8_0_soa": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_xmx_moe_gemm_mxfp4": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_xmx_moe_gemm_mxfp4_coalesced": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_xmx_moe_gemm_mxfp4_soa": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_xmx_moe_gemm_q8_0": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_xmx_moe_gemm_q8_0_coalesced": (1, MOE_XMX),
    "submit:moe-xmx.hpp:launch_xmx_moe_gemm_q8_0_soa": (1, MOE_XMX),
    "submit:moe-xmx.hpp:preprocess_tokens_q8_async": (1, MOE_XMX),
    "submit:unified-cache.cpp:cpu_job_input_marker_submit":
        (1, "the input marker of a host expert job (a single_task that publishes a serial), not yet wrapped"),
    "submit:unified-cache.cpp:unified_cache::onednn_graph_scratch_free":
        (1, "the completion marker of a freed oneDNN Graph scratch slot on the watch queue, not yet wrapped"),
    "submit:unified-kernel.cpp:micro_submit_add": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:micro_submit_attention": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:micro_submit_rms_norm": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:micro_submit_silu_mul": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:micro_submit_strided_copy": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:mmvq_submit_silu_mul": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:UnifiedKernel::benchmark_graph_overhead":
        (7, "test or benchmark entry point; graph compute does not call it"),
    "submit:unified-kernel.cpp:UnifiedKernel::execute_persistent_phased": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:UnifiedKernel::launch_persistent_kernel_async":
        (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:UnifiedKernel::launch_persistent_kernel/run_persistent_kernel":
        (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:UnifiedKernel::record_micro_graph": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:UnifiedKernel::rms_norm": (1, "unified kernel launch, not yet wrapped"),
    "submit:unified-kernel.cpp:UnifiedKernel::silu_mul": (1, "unified kernel launch, not yet wrapped"),
    "shortcut:concat.cpp:concat_T_sycl": (3, SHORTCUT_KERNEL),
    "shortcut:diag.cpp:ggml_sycl_op_diag": (1, SHORTCUT_KERNEL),
    "shortcut:dsv4-hc-kernels.hpp:hc_comb_launch": (1, SHORTCUT_KERNEL),
    "shortcut:dsv4-hc-kernels.hpp:hc_post_launch": (1, SHORTCUT_KERNEL),
    "shortcut:dsv4-hc-kernels.hpp:hc_pre_launch": (1, SHORTCUT_KERNEL),
    "shortcut:moe-sort.hpp:moe_convert_f32_to_f16": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_convert_f32_to_f16_async": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_count_tokens_per_expert": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_scatter_results": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_scatter_results_f16_to_f32": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_sort_tokens_by_expert": (1, MOE_XMX),
    "shortcut:moe-xmx-fused.hpp:fused_xmx_moe_gemm_q8_0": (1, MOE_XMX),
    "shortcut:moe-xmx.hpp:extract_q8_0_scales": (1, MOE_XMX),
    "shortcut:concat.cpp:concat_T_sycl_non_cont": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:convert_unary_nc_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_block_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_mxfp4_soa_to_fp16_rowmajor": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_mxfp4_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_nvfp4_fp16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q2_K_sycl": (2, SHORTCUT_KERNEL_QK_K),
    "shortcut:convert.cpp:dequantize_row_q3_K_sycl": (2, SHORTCUT_KERNEL_QK_K),
    "shortcut:convert.cpp:dequantize_row_q4_0_soa_to_fp16_rowmajor": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl_coalesced": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl_coalesced_rowmajor": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl_reorder": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q4_1_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q5_K_sycl": (2, SHORTCUT_KERNEL_QK_K),
    "shortcut:convert.cpp:dequantize_row_q6_K_sycl": (2, SHORTCUT_KERNEL_QK_K),
    "shortcut:convert.cpp:dequantize_row_q6_K_sycl_reorder": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q8_0_coalesced_rowmajor": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:dequantize_row_q8_0_soa_to_fp16_rowmajor": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:reorder_mxfp4_aos_to_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:reorder_q4_0_aos_to_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:reorder_q4_0_coalesced_to_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:reorder_q4_k_aos_to_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:reorder_q6_k_aos_to_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:reorder_q8_0_aos_to_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:repack_mxfp4_soa_to_woq": (2, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:repack_mxfp4_soa_to_woq_batched": (2, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:repack_mxfp4_xmx_tiled_to_woq": (2, SHORTCUT_KERNEL),
    "shortcut:convert.cpp:repack_mxfp4_xmx_tiled_to_woq_batched": (2, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_bf16_bf16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_bf16_f16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_bf16_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f16_bf16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f16_f16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f16_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_bf16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_f16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_i32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_iq4_nl_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_q4_0_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_q4_1_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_q5_0_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_q5_1_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_f32_q8_0_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_i16_i16_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_i32_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_i32_i32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q4_0_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q4_0_q4_0": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q4_1_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q4_1_q4_1": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q5_0_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q5_0_q5_0": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q5_1_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q5_1_q5_1": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q8_0_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_q8_0_q8_0": (1, SHORTCUT_KERNEL),
    "shortcut:cpy.cpp:ggml_cpy_row_contig_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:acc_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_arange": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_ceil": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_clamp": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_cos": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_floor": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_geglu": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_geglu_erf": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_geglu_quick": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_leaky_relu": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_log": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_reglu": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_round": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_sin": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_sqr": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_sqrt": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_swiglu": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_trunc": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:ggml_sycl_op_unary": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:swiglu_oai_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:element_wise.cpp:upscale_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:fattn-onednn.cpp:ggml_sycl_flash_attn_ext_onednn_materialize_q_f32": (1, SHORTCUT_KERNEL),
    "shortcut:fattn-onednn.cpp:ggml_sycl_onednn_fa_materialize_one": (2, SHORTCUT_KERNEL),
    "shortcut:fattn.cpp:ggml_sycl_flash_attn_ext": (2, SHORTCUT_KERNEL),
    "shortcut:fattn.cpp:ggml_sycl_v2_pre_allocate_buffers": (1, SHORTCUT_KERNEL),
    "shortcut:fill.cpp:ggml_sycl_op_fill": (2, SHORTCUT_KERNEL),
    "shortcut:gated_delta_net.cpp:launch_gated_delta_net": (4, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q4_0_coalesced_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q4_k_aos_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q6_k_aos_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q6_k_coalesced_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q6_k_coalesced_variable_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q6_k_soa_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q8_0_aos_pair_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q8_0_aos_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_q8_0_coalesced_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_sycl_float": (1, SHORTCUT_KERNEL),
    "shortcut:getrows.cpp:get_rows_sycl_reorder": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:diag_mask_inf_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_mul_mat_p021_f16_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_mul_mat_vec_nc_f16_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_copy_ids_to_host": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_cpy_tensor_2d": (3, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_mul_mat_batched_f16_fallback": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_mul_mat_tp_row_parallel_post": (8, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_op_mean": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_op_mul_mat": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_op_pool2d": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_readback_via_shared_kernel": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:ggml_sycl_tp_launch_async_ffn": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:scale_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:sum_rows_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:ggml-sycl.cpp:tp_device1_worker_thread_func": (1, SHORTCUT_KERNEL),
    "shortcut:mem-ops.cpp:mem_copy_ptr_async": (1, SHORTCUT_USM),
    "shortcut:pad.cpp:pad_f32_sycl": (1, SHORTCUT_KERNEL),
    "shortcut:set.cpp:ggml_sycl_op_set": (1, SHORTCUT_KERNEL),
    "shortcut:unified-kernel.cpp:UnifiedKernel::execute_deferred_copies": (1, SHORTCUT_USM),
    "shortcut:unified-kernel.cpp:UnifiedKernel::reset_dag_counters": (4, SHORTCUT_USM),
}

# "file:function" -> the launches in it that are profiled by an explicit record call. The record rule (Code.recorded)
# excuses a launch only in a function listed here, and only when the number of launches it matches there equals the
# count. Adding a row is the reviewed act, not the rule: read the function and confirm that the event each record
# call receives is that launch's event, on the paths that matter, before adding or raising a row.
RECORDED_ON_THE_PATH = {
    # copy_event, assigned by each of three width branches, recorded once after them while profiling is on
    "common.hpp:ggml_sycl_graph_safe_memcpy": 3,
    # first_event and merge_event, each recorded right after its launch
    "fattn-xmx-f16-v2.hpp:launch_fattn_xmx_v2_decode_gqa_split_leaf": 2,
    "fattn.cpp:ggml_sycl_fattn_xmx_materialize_packed_k": 1,  # pack_event
    "fattn.cpp:ggml_sycl_fattn_xmx_submit_set_rows_update": 1,  # event
    "mem-ops.cpp:mem_fill_direct_submit": 1,  # event
}

BASE_WRAPPER = "ggml_sycl_profile_submit"
# record a launch's returned event with the profiler; a launch whose event reaches one is profiled (Code.recorded)
RECORDERS = ("ggml_sycl_kernel_profile_record_event", "ggml_sycl_profile_record_returned_event")
RECORD_CALL = re.compile(r"\b(?:" + "|".join(RECORDERS) + r")\s*\(")

SUBMIT = re.compile(r"(?:->|\.)\s*submit\s*\(")
# a queue member that enqueues a command without submit(); the receiver decides whether it is one (receiver_kind)
SHORTCUT = re.compile(
    r"(?:->|\.)\s*(parallel_for|single_task|memcpy|memset|fill|copy|prefetch|mem_advise)\s*(?:<[^;{}()]*>\s*)?\("
)
QUEUE_TYPES = {"sycl::queue", "queue", "queue_ptr", "dpct::queue_ptr"}
HANDLER_TYPES = {"sycl::handler", "handler"}
NOT_A_TYPE = {"return", "else", "case", "new", "delete", "throw", "goto", "sizeof", "co_return", "using", "typedef",
              "operator", "const", "static", "inline", "constexpr", "auto", "do"}

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
    """Blanks the #elif / #else branches of each conditional group that hold no launch-like call, so brackets are
    counted over one branch: code like `#ifdef A for (...) { #else for (...) { #endif` opens one block, not two. A
    branch that holds a call of either class (a submit or a shortcut name) is kept whatever its condition, so no
    launch is hidden by the blanking; if a kept branch also unbalances the brackets, the balance test names the
    file. The first branch of each group is never blanked."""
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
                text = "\n".join(lines[j] for j in branch)
                if not (SUBMIT.search(text) or SHORTCUT.search(text)):
                    drop.update(branch)
        elif groups:
            groups[-1][-1].append(i)
            for outer in groups[:-1]:
                outer[-1].append(i)
    return "\n".join(re.sub(r"[^\n]", " ", t) if i in drop else t for i, t in enumerate(lines))


def blank(src: str) -> str:
    return blank_alternative_branches(NOT_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src))


class Code:
    """Blanked source with its bracket structure: which '(' each ')' closes, and the open '(' and '{' at each site."""

    def __init__(self, src: str):
        self.code = blank(src)
        self.paren_open: dict[int, int] = {}
        self.paren_close: dict[int, int] = {}
        # for a '{' that opens a function: the '(' and ')' of its parameter list
        self.params: dict[int, tuple[int, int]] = {}
        self._brace_starts: list[int] | None = None
        self.brace_close: dict[int, int] = {}
        self.unbalanced: list[str] = []
        # (position, kind, method, open '(' positions, open '{' positions) for each submit or shortcut call
        self.sites: list[tuple[int, str, str, list[int], list[int]]] = []
        starts = sorted([(m.start(), "submit", "submit") for m in SUBMIT.finditer(self.code)]
                        + [(m.start(), "shortcut", m.group(1)) for m in SHORTCUT.finditer(self.code)])
        parens: list[int] = []
        braces: list[int] = []
        si = 0
        for m in re.finditer(r"[(){}]", self.code):
            pos, c = m.start(), m.group(0)
            while si < len(starts) and starts[si][0] < pos:
                self.sites.append((*starts[si], list(parens), list(braces)))
                si += 1
            if c == "(":
                parens.append(pos)
            elif c == "{":
                braces.append(pos)
            elif c == ")":
                if parens:
                    self.paren_open[pos] = parens.pop()
                    self.paren_close[self.paren_open[pos]] = pos
                else:
                    self.unbalanced.append(f"')' at line {self.line(pos)} closes nothing")
            elif braces:
                self.brace_close[braces.pop()] = pos
            else:
                self.unbalanced.append(f"'}}' at line {self.line(pos)} closes nothing")
        while si < len(starts):
            self.sites.append((*starts[si], list(parens), list(braces)))
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
                self.params[pos] = (open_, close)
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
            self.params[pos] = (open_, close)
            return "function", ident

    def function_brace(self, braces: list[int]) -> int:
        """The '{' of the innermost named function around a site, or 0."""
        for b in reversed(braces):
            if self.brace_kind(b)[0] == "function":
                return b
        return 0

    def function_brace_at(self, pos: int) -> int | None:
        """The '{' of the innermost named function whose body holds pos, for a position that is not a site."""
        if self._brace_starts is None:
            self._brace_starts = sorted(self.brace_close)
        i = bisect.bisect_left(self._brace_starts, pos) - 1
        while i >= 0:
            b = self._brace_starts[i]
            if self.brace_close[b] > pos and self.brace_kind(b)[0] == "function":
                return b
            i -= 1
        return None

    def innermost_brace(self, pos: int) -> int | None:
        """The '{' of the innermost block of any kind whose body holds pos."""
        if self._brace_starts is None:
            self._brace_starts = sorted(self.brace_close)
        i = bisect.bisect_left(self._brace_starts, pos) - 1
        while i >= 0:
            b = self._brace_starts[i]
            if self.brace_close[b] > pos:
                return b
            i -= 1
        return None

    def braces_at(self, pos: int) -> set[int]:
        """The '{' of every block whose body holds pos."""
        found = set()
        b = self.innermost_brace(pos)
        while b is not None:
            found.add(b)
            b = self.innermost_brace(b)
        return found

    def is_loop(self, brace: int) -> bool:
        head = self.code[max(0, brace - 2000):brace].rstrip()
        if re.search(r"\bdo$", head):
            return True
        open_ = self.paren_open.get(max(0, brace - 2000) + len(head) - 1) if head.endswith(")") else None
        return open_ is not None and self.callee(open_) in ("for", "while")

    def is_else_arm(self, brace: int) -> bool:
        """The '{' opens an `else` or `else if (...)` arm."""
        lo = max(0, brace - 2000)
        head = self.code[lo:brace].rstrip()
        if re.search(r"\belse$", head):
            return True
        if not head.endswith(")"):
            return False
        open_ = self.paren_open.get(lo + len(head) - 1)
        return (open_ is not None and self.callee(open_) == "if"
                and re.search(r"\belse\s*if\s*$", self.code[max(0, open_ - 200):open_]) is not None)

    def body_brace(self, braces: list[int]) -> int | None:
        """The '{' of the innermost function, lambda or class body among braces (any order)."""
        for b in sorted(braces, reverse=True):
            if self.brace_kind(b)[0] in ("function", "lambda", "class"):
                return b
        return None

    def assigned_event(self, pos: int) -> tuple[str, int] | None:
        """(`ev`, where its statement starts) when the launch whose '.'/'->' is at pos is the whole right-hand side of
        `ev = ...` (a declaration `sycl::event ev = ...` included) and `ev` is a plain name, not a member (`s.ev`,
        `this->ev`); else None."""
        i = pos
        while True:  # back over the receiver expression: names, `::`, `.`, `->` and bracketed calls or indexes
            head = self.code[max(0, i - 300):i]
            stripped = head.rstrip()
            j = max(0, i - 300) + len(stripped)
            if stripped.endswith(")") and (j - 1) in self.paren_open:
                i = self.paren_open[j - 1]
                continue
            m = re.search(r"(?:->|\.|::)?\s*\w+$", stripped)
            if not m:
                break
            i = max(0, i - 300) + m.start()
            if not re.match(r"\s*(?:->|\.|::)", self.code[i:]):
                break
        lo = max(0, i - 300)
        m = re.search(r"\b(\w+)\s*=\s*$", self.code[lo:i])
        if not m or re.search(r"(?:\.|->|::)\s*$", self.code[max(0, lo + m.start(1) - 50):lo + m.start(1)]):
            return None
        # the statement starts after the type of a declaration, at the name of an assignment
        decl = re.search(r"([A-Za-z_][\w:]*(?:\s*<[^;{}()]*>)?[\s&*]+)$", self.code[max(0, lo + m.start(1) - 300):
                                                                               lo + m.start(1)])
        start = lo + m.start(1)
        if decl and decl.group(1).split()[0].split("<")[0] not in NOT_A_TYPE:
            start -= len(decl.group(1))
        return m.group(1), start

    def under_braceless_loop(self, start: int) -> bool:
        """The statement starting at start is, through brace-less control prefixes only, the body of a `for (...)`,
        `while (...)` or `do`: walking back over `if (...)` heads and `else` (with the statement its `if` controls),
        to the previous `;`, `{` or `}`, meets a loop head."""
        end = start
        while True:
            lo = max(0, end - 2000)  # a window, not the whole file: this runs for every assigned launch
            head = self.code[lo:end].rstrip()
            if re.search(r"\bdo$", head):
                return True
            other = re.search(r"\belse$", head)
            if other:
                end = self.controlled_statement_head(lo + other.start())
                if end is None:
                    return False
                continue
            open_ = self.paren_open.get(lo + len(head) - 1) if head.endswith(")") else None
            word = self.callee(open_) if open_ is not None else None
            if word in ("for", "while"):
                return True
            if word != "if":
                return False
            keyword_lo = max(0, open_ - 300)
            end = keyword_lo + re.search(r"\bif\s*$", self.code[keyword_lo:open_]).start()

    def controlled_statement_head(self, else_pos: int) -> int | None:
        """For the `else` at else_pos: the end of the `if (...)` head whose statement it follows, or None when that
        statement is not found behind a control head."""
        i = else_pos - 1
        while i >= 0 and self.code[i].isspace():
            i -= 1
        if i >= 0 and self.code[i] == "}":
            opens = [b for b, c in self.brace_close.items() if c == i]
            if not opens:
                return None
            return opens[0]
        i -= 1  # past the statement's ';'
        while i >= 0:
            c = self.code[i]
            if c == ")" and i in self.paren_open:
                if self.callee(self.paren_open[i]) in ("if", "for", "while"):
                    return i + 1
                i = self.paren_open[i] - 1
                continue
            if c in ";{}":
                return None
            i -= 1
        return None

    def writes(self, name: str, lo: int, hi: int) -> list[int]:
        """Positions in [lo, hi) where name is assigned (`name =`) or declared (`T name`, `T & name = ...`)."""
        ident = re.escape(name)
        found = [m.start() for m in re.finditer(r"(?<![.>:\w])" + ident + r"\s*=(?!=)", self.code[lo:hi])]
        for m in re.finditer(r"\b([A-Za-z_]\w*)(?:\s*<[^;{}()]*>)?[ \t&*]+" + ident + r"\s*[=;({,]",
                             self.code[lo:hi]):
            if m.group(1) not in NOT_A_TYPE:
                found.append(m.start())
        return [lo + p for p in found]

    def recorded(self, pos: int, parens: list[int], braces: list[int]) -> bool:
        """The launch at pos is lexically followed, in the same function or lambda body, by a record call (RECORDERS)
        that receives its event. This is a candidate only: census excuses it only for a function that is a row of
        RECORDED_ON_THE_PATH with a matching count, because the rule does not follow paths and the row vouches for them.
        - its event is the record call's argument, the launch not inside a lambda written in that call; or
        - it is assigned to a plain name `ev` (`ev = <launch>`, a declaration included) and a later record call in the
          same body is handed `ev`, or `std::move(ev)`, as its event (second) argument; a record call under a
          condition (the profiler being on) is accepted; unless
          - the assignment is the body of a loop with no braces, through any brace-less `if` / `else` heads;
          - a loop around the launch, braced or not, does not also hold the record call;
          - the outermost block that holds the record call but not the launch is an `else` / `else if` arm;
          - `ev` is assigned or declared between them in the launch's block, a block around it, or a block that holds
            the record call (a shadowing declaration included)."""
        if parens and self.callee(parens[-1]) in RECORDERS and not any(b > parens[-1] for b in braces):
            return True
        target = self.assigned_event(pos)
        body = self.body_brace(braces)
        if target is None or body is None:
            return False
        name, start = target
        if self.under_braceless_loop(start):
            return False
        launch_braces = set(braces)
        call_open = self.code.find("(", pos)
        after_launch = self.paren_close.get(call_open, pos)
        for m in RECORD_CALL.finditer(self.code, after_launch, self.brace_close.get(body, len(self.code))):
            open_ = m.end() - 1
            close = self.paren_close.get(open_)
            args = top_level_args(self.code[open_ + 1:close]) if close is not None else []
            event = re.sub(r"^std::move\s*\(\s*(\w+)\s*\)$", r"\1", args[1]) if len(args) >= 2 else None
            if event != name:
                continue
            around_record = self.braces_at(m.start())
            if self.body_brace(list(around_record)) != body:
                continue
            if any(self.is_loop(b) for b in launch_braces - around_record):
                continue
            record_only = around_record - launch_braces
            if record_only and launch_braces - around_record and self.is_else_arm(min(record_only)):
                continue
            if any(self.innermost_brace(w) in launch_braces or self.innermost_brace(w) in around_record
                   for w in self.writes(name, after_launch, m.start())):
                continue
            return True
        return False

    def receiver(self, pos: int) -> str | None:
        """The object a member call at pos (the '.' or '->') is made on: `q` for `q->f(`, `stream()` for
        `ctx.stream()->f(`, None for anything else (`(*q).f(`, `qs[i].f(`)."""
        head = self.code[max(0, pos - 300):pos].rstrip()
        if head.endswith(")"):
            open_ = self.paren_open.get(max(0, pos - 300) + len(head) - 1)
            name = self.callee(open_) if open_ is not None else None
            return f"{name}()" if name else None
        m = re.search(r"(\w+)$", head)
        return m.group(1) if m else None

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


def top_level_args(text: str) -> list[str]:
    args, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
    args.append(text[start:].strip())
    return args


# an argument that passes a parameter on: `f`, `static_cast<F &&>(f)` or `std::forward<F>(f)`
FORWARDED_ARG = re.compile(r"(?:static_cast\s*<[^()]*>|std::forward\s*<[^()]*>)\s*\(\s*(\w+)\s*\)|(\w+)")
# a parameter forwarded as the whole argument of a submit inside a wrapper call's arguments, a lambda body included
FORWARDED_TO_SUBMIT = re.compile(r"(?:->|\.)\s*submit\s*\(\s*"
                                 r"(?:\bstatic_cast\s*<[^()]*&&\s*>|\bstd::forward\s*<[^()]*>)"
                                 r"\s*\(\s*(\w+)\s*\)\s*\)")


def derive_wrappers(texts: dict[str, str]) -> set[str]:
    """ggml_sycl_profile_submit, and every function that hands one of its own forwarding-reference parameters
    (`F && f`) on as the command group of a wrapper call: as the argument in the wrapper's command-group position
    (`f`, `static_cast<F &&>(f)`, `std::forward<F>(f)`), or as the whole argument of a `submit(` inside the call's
    arguments, as mmq_submit does from inside the profiler's lambda
    (`ggml_sycl_profile_submit(q, label, [&](sycl::queue &) { return stream->submit(std::forward<F>(fn)); })`).
    A derived wrapper's command-group position is that parameter's. Found by name-free structure and closed
    transitively, so a helper is a wrapper whatever it is called (ggml_sycl_submit_binbcast_kernel is one)."""
    key = tuple(sorted((name, id(parse(src))) for name, src in texts.items()))
    if key not in _WRAPPERS:
        _WRAPPERS[key] = derive_wrappers_uncached(texts)
    return set(_WRAPPERS[key])


_WRAPPERS: dict[tuple, set[str]] = {}


# the command-group parameter of ggml_sycl_profile_submit(q, label, submit_fn, file, line, function)
BASE_WRAPPER_COMMAND_GROUP = 2


def derive_wrappers_uncached(texts: dict[str, str]) -> set[str]:
    # wrapper -> the index of its command-group parameter, the one a caller's command group is passed in
    wrappers = {BASE_WRAPPER: BASE_WRAPPER_COMMAND_GROUP}
    changed = True
    while changed:
        changed = False
        call = re.compile(r"\b(" + "|".join(sorted(wrappers)) + r")\s*(?:<[^;{}()]*>\s*)?\(")
        for src in texts.values():
            code = parse(src)
            for m in call.finditer(code.code):
                open_ = m.end() - 1
                close = code.paren_close.get(open_)
                fb = code.function_brace_at(open_)
                if close is None or fb is None or fb not in code.params:
                    continue
                kind, name = code.brace_kind(fb)
                p_open, p_close = code.params[fb]
                params = top_level_args(code.code[p_open + 1:p_close])
                forwarding = {}
                for index, param in enumerate(params):
                    f = re.search(r"&&\s*(\w+)\s*(?:=.*)?$", param, re.S)
                    if f:
                        forwarding[f.group(1)] = index
                inside = code.code[open_ + 1:close]
                # forwarded as the whole argument of a submit inside the call (mmq_submit's lambda), or passed as the
                # call's own command-group argument
                passed_on = [g.group(1) for g in FORWARDED_TO_SUBMIT.finditer(inside)]
                args = top_level_args(inside)
                if wrappers[m.group(1)] < len(args):
                    inner = FORWARDED_ARG.fullmatch(args[wrappers[m.group(1)]])
                    if inner:
                        passed_on.append(inner.group(1) or inner.group(2))
                for passed in passed_on:
                    if passed in forwarding:
                        short = name.split("::")[-1]
                        if short not in wrappers:
                            wrappers[short] = forwarding[passed]
                            changed = True
                        break
    return set(wrappers)


def declared_type(text: str, name: str, last: bool) -> str | None:
    """The type in a declaration of name (`T name`, `T & name`, `T * const name`, `T name(` for a function) in text:
    the last one when last, else a queue or handler declaration first and any other after."""
    call = name.endswith("()")
    ident = re.escape(name[:-2] if call else name)
    after = r"\s*\(" if call else r"\s*(?=[;=,)\[{(]|$)"
    # a type is a (qualified) name with optional template arguments, on the declaration's own line; an argument may
    # hold one level of brackets (`std::array<size_t, static_cast<size_t>(zone_id::COUNT)>`)
    targs = r"<(?:[^;{}()<>\n]|\([^;{}()\n]*\)|<[^;{}()<>\n]*>)*>"
    pattern = re.compile(r"\b([A-Za-z_]\w*(?:::\w+)*)(?:[ \t]*" + targs + r")?(?:[ \t]*(?:const|&&|&|\*))*[ \t]+"
                         + ident + r"\b" + after, re.M)
    types = [m.group(1) for m in pattern.finditer(text) if m.group(1) not in NOT_A_TYPE]
    if not types:
        return None
    if last:
        return types[-1]
    typed = [t for t in types if t in QUEUE_TYPES or t in HANDLER_TYPES]
    return typed[0] if typed else types[0]


_FILE_TYPES: dict[tuple[int, str], str | None] = {}


def file_declared_type(text: str, name: str) -> str | None:
    """declared_type over a whole file, kept per file and name: a file's receivers repeat, and a whole-file scan of
    ggml-sycl.cpp is the census's most expensive step."""
    key = (id(parse(text)), name)
    if key not in _FILE_TYPES:
        _FILE_TYPES[key] = declared_type(parse(text).code, name, last=False)
    return _FILE_TYPES[key]


def receiver_kind(code: Code, pos: int, braces: list[int], file_text: str, decl_texts: dict[str, str]) -> str:
    """queue, handler, other:<type> or unresolved: the declared type of the receiver of the shortcut at pos, from the
    nearest declaration before it in its function (parameters and lambda parameters included), else from its file,
    else from any file the census reads (a member declared in a class header)."""
    name = code.receiver(pos)
    if name is None:
        return "unresolved"
    lo = max(0, code.function_brace(braces) - 1500)
    found = declared_type(code.code[lo:pos], name, last=True)
    if found is None:
        found = file_declared_type(file_text, name)
    if found is None:
        for other in decl_texts.values():
            found = file_declared_type(other, name)
            if found is not None:
                break
    if found is None:
        return "unresolved"
    if found in QUEUE_TYPES:
        return "queue"
    if found in HANDLER_TYPES:
        return "handler"
    return f"other:{found}"


def census(texts: dict[str, str], wrappers: set[str] | None = None, decl_texts: dict[str, str] | None = None,
           skipped: list[str] | None = None, recorded: list[str] | None = None,
           table: dict[str, int] | None = None) -> dict[str, list[tuple[int, str, str]]]:
    """The raw sites of each named text, as (line, function, kind) triples. A site is wrapped when any '(' still open
    at it, however far back and however deeply nested, is the argument list of a wrapper call. A launch the record
    rule matches (Code.recorded) is excused only when its "file:function" is a row of table (RECORDED_ON_THE_PATH by
    default) and the rule matches exactly that many launches in the function; each one excused is appended to
    recorded. A shortcut is a site only when its receiver is a queue and it is not inside a submit's parentheses
    (there it is the command group's handler call, and the submit is the site); each shortcut set aside is appended to
    skipped with its reason."""
    if wrappers is None:
        wrappers = derive_wrappers(texts)
    if decl_texts is None:
        decl_texts = texts
    if table is None:
        table = RECORDED_ON_THE_PATH
    found: dict[str, list[tuple[int, str, str]]] = {}
    for name, src in texts.items():
        code = parse(src)
        unwrapped = []
        matched: dict[str, int] = {}
        for pos, kind, method, parens, braces in code.sites:
            if any(code.callee(p) in wrappers for p in parens):
                continue
            key = f"{name}:{code.enclosing(braces)}" if code.recorded(pos, parens, braces) else None
            if key is not None:
                matched[key] = matched.get(key, 0) + 1
            unwrapped.append((pos, kind, method, parens, braces, key))
        for pos, kind, method, parens, braces, key in unwrapped:
            if key is not None and table.get(key) == matched[key]:
                if recorded is not None:
                    recorded.append(key)
                continue
            callees = [code.callee(p) for p in parens]
            if kind == "shortcut":
                why = None
                if "submit" in callees:
                    why = "inside a submit"
                else:
                    receiver = receiver_kind(code, pos, braces, src, decl_texts)
                    if receiver != "queue":
                        why = f"receiver {code.receiver(pos)} is {receiver}"
                if why is not None:
                    if skipped is not None:
                        skipped.append(f"{name}:{code.enclosing(braces)} .{method}( {why}")
                    continue
            found.setdefault(name, []).append((code.line(pos), code.enclosing(braces), kind))
    return found


def census_keys(found: dict[str, list[tuple[int, str, str]]]) -> dict[str, int]:
    keys: dict[str, int] = {}
    for name, sites in found.items():
        for _, func, kind in sites:
            key = f"{kind}:{name}:{func}"
            keys[key] = keys.get(key, 0) + 1
    return keys


def diff_against_allow(keys: dict[str, int], allow: dict[str, tuple[int, str]]) -> list[str]:
    problems = []
    for key in sorted(set(keys) | set(allow)):
        have = keys.get(key, 0)
        want = allow[key][0] if key in allow else 0
        if have != want:
            problems.append(f"{key}: {have} raw site(s) in the tree, {want} listed")
    return problems


def read_tree() -> dict[str, str]:
    """FILES, the profiler header (wrapper definitions) and the header beside each .cpp of FILES (its class members'
    declarations, which a receiver's type is read from)."""
    texts = {}
    for name in FILES + [PROFILER_HEADER]:
        path = SYCL / name
        assert path.exists(), f"{name} is gone: update FILES"
        texts[name] = path.read_text(encoding="utf-8")
    for name in FILES:
        header = SYCL / (name[:-4] + ".hpp")
        if name.endswith(".cpp") and header.exists() and header.name not in texts:
            texts[header.name] = header.read_text(encoding="utf-8")
    return texts


def census_tree(texts: dict[str, str], skipped: list[str] | None = None) -> dict[str, list[tuple[int, str, str]]]:
    wrappers = derive_wrappers(texts)
    return census({name: texts[name] for name in FILES}, wrappers, texts, skipped)


def format_sites(found: dict[str, list[tuple[int, str, str]]]) -> str:
    return "\n".join(f"{name}:{line} {func} {kind}" for name in sorted(found) for line, func, kind in found[name])


def count_by_kind_and_file(found: dict[str, list[tuple[int, str, str]]]) -> str:
    rows = []
    for kind in ("submit", "shortcut"):
        per_file = {name: sum(1 for site in sites if site[2] == kind) for name, sites in found.items()}
        per_file = {name: n for name, n in per_file.items() if n}
        listed = ", ".join(f"{name} {n}" for name, n in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0])))
        rows.append(f"{kind}: {sum(per_file.values())} ({listed})")
    return "\n".join(rows)


def raw_functions(src: str, wrappers: set[str] | None = None) -> list[str]:
    return [func for _, func, _ in census({"x.cpp": src}, wrappers).get("x.cpp", [])]


def raw_sites(src: str, skipped: list[str] | None = None, table: dict[str, int] | None = None) -> list[tuple[str, str]]:
    return [(func, kind) for _, func, kind in census({"x.cpp": src}, skipped=skipped, table=table).get("x.cpp", [])]


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
        "ggml_sycl_submit_binbcast_kernel",
        "mmq_submit",
    }


def test_a_helper_that_forwards_its_command_group_inside_the_profilers_lambda_is_a_wrapper():
    helpers = (
        "template <typename F> sycl::event via_lambda(sycl::queue * q, F && fn) {\n"
        "    return ggml_sycl_profile_submit(label, [&]() { return q->submit(std::forward<F>(fn)); });\n"
        "}\n"
        "template <typename F> sycl::event outside(sycl::queue * q, F && fn) {\n"
        "    sycl::event ev = q->submit(std::forward<F>(fn));\n"
        "    return ggml_sycl_profile_submit(label, [&]() { return ev; });\n"
        "}\n"
    )
    wrappers = derive_wrappers({"x.cpp": helpers})
    assert wrappers == {BASE_WRAPPER, "via_lambda"}
    assert raw_functions(helpers, wrappers) == ["outside"]


def test_a_launch_in_an_alternative_branch_is_counted_whatever_its_kind():
    for launch, kind in (("q->parallel_for(r, k);", "shortcut"), ("q->submit(cgf);", "submit")):
        sample = f"void a(sycl::queue * q) {{\n#ifdef FOO\n    log();\n#else\n    {launch}\n#endif\n}}\n"
        assert raw_sites(sample) == [("a", kind)], sample
    # a branch with no launch is still blanked, so a block opened in both branches is counted once
    sample = "void a(sycl::queue * q) {\n#ifdef FOO\n    for (;;) {\n#else\n    while (x) {\n#endif\n" \
             "        q->submit(cgf);\n    }\n}\n"
    assert not parse(sample).unbalanced
    assert raw_sites(sample) == [("a", "submit")]


RECORD = "ggml_sycl_kernel_profile_record_event"


H = "void f(sycl::queue * q, bool c) {\n"
L1 = "q->submit([&](sycl::handler & h) { h.single_task(k1); })"
L2 = "q->submit([&](sycl::handler & h) { h.single_task(k2); })"
REC2 = "ggml_sycl_profile_record_returned_event"


def test_a_launch_in_a_function_not_in_the_table_is_reported_whatever_the_shape():
    # (shape, launches in it): with no table row for x.cpp:f, every launch is reported, the plain shape included
    cases = {
        "plain": (H + f"sycl::event ev = {L1};\n{RECORD}(lab, ev);\n}}\n", 1),
        "direct": (H + f"{REC2}(lab, {L1});\n}}\n", 1),
        "opposite arm": (H + f"sycl::event ev;\nif (c) {{ ev = {L1}; }} else {{ ev = {L2}; {RECORD}(lab, ev); }}\n}}\n",
                         2),
        "shadow": (H + f"sycl::event ev = {L1};\n{{ sycl::event ev = other(); {RECORD}(lab, ev); }}\n}}\n", 1),
        "brace-less for": (H + f"sycl::event ev;\nfor (int i = 0; i < n; ++i) ev = {L1};\n{RECORD}(lab, ev);\n}}\n", 1),
        "brace-less do": (H + f"sycl::event ev;\ndo ev = {L1}; while (c);\n{RECORD}(lab, ev);\n}}\n", 1),
        "brace-less loop and if": (H + f"sycl::event ev;\nwhile (c) if (x) ev = {L1};\n{RECORD}(lab, ev);\n}}\n", 1),
        "brace-less loop and else": (H + f"sycl::event ev;\nwhile (c) if (x) g(); else ev = {L1};\n"
                                     f"{RECORD}(lab, ev);\n}}\n", 1),
        "brace-less else record": (H + f"sycl::event ev = {L1};\nif (c) g(); else {RECORD}(lab, ev);\n}}\n", 1),
        "sibling blocks": (H + f"sycl::event ev;\n{{ ev = {L1}; }} {{ ev = {L2}; }}\n{RECORD}(lab, ev);\n}}\n", 2),
        "two ifs": (H + f"sycl::event ev;\nif (a) {{ ev = {L1}; }} if (b) {{ ev = {L2}; }}\n"
                    f"{RECORD}(lab, ev);\n}}\n", 2),
        "switch case": (H + f"sycl::event ev;\nswitch (k) {{ case 0: ev = {L1}; break; "
                        f"case 1: {RECORD}(lab, ev); }}\n}}\n", 1),
        "catch handler": (H + f"sycl::event ev = {L1};\ntry {{ g(); }} catch (...) {{ {RECORD}(lab, ev); }}\n}}\n", 1),
        "range-for shadow": (H + f"sycl::event ev = {L1};\nfor (auto & ev : evs) {{ {RECORD}(lab, ev); }}\n}}\n", 1),
        "structured binding": (H + f"sycl::event ev = {L1};\nauto [ev, n] = pair();\n{RECORD}(lab, ev);\n}}\n", 1),
        "record in a define": (H + f"sycl::event ev = {L1};\n#define REC_IT {RECORD}(lab, ev)\n}}\n", 1),
        "lambda argument": (H + f"sycl::event ev;\n{RECORD}(lab, ev, [&] {{ {L1}; }});\n}}\n", 1),
        "member": (H + f"sycl::event ev;\ns.ev = {L1};\n{RECORD}(lab, ev);\n}}\n", 1),
        "uncalled lambda": (H + f"sycl::event ev = {L1};\nauto g = [&] {{ {RECORD}(lab, ev); }};\n}}\n", 1),
        "ternary": (H + f"sycl::event ev = c ? {L1} : {L2};\n{RECORD}(lab, ev);\n}}\n", 2),
    }
    for what, (sample, launches) in cases.items():
        assert [func for func, _ in raw_sites(sample)] == ["f"] * launches, what


def test_a_listed_function_whose_launches_the_rule_matches_is_excused():
    cases = {
        "plain": (H + f"sycl::event ev = {L1};\n{RECORD}(lab, ev, site, t0, t1);\n}}\n", 1),
        "direct": (H + f"return {REC2}(lab, {L1});\n}}\n", 1),
        "join": (H + f"sycl::event ev;\nif (c) {{ ev = {L1}; }} else {{ ev = {L2}; }}\n{RECORD}(lab, ev);\n}}\n", 2),
        "moved": (H + f"sycl::event ev = {L1};\n{REC2}(lab, std::move(ev));\n}}\n", 1),
        "record under profiling on": (H + f"sycl::event ev = {L1};\nif (on) {{ {RECORD}(lab, ev); }}\n}}\n", 1),
    }
    for what, (sample, count) in cases.items():
        assert raw_sites(sample, table={"x.cpp:f": count}) == [], what


def test_a_listed_function_whose_match_count_differs_from_its_row_is_reported():
    two = H + f"sycl::event a = {L1};\n{RECORD}(lab, a);\nsycl::event b = {L2};\n{RECORD}(lab, b);\n}}\n"
    # a new recorded launch in a listed function: N listed, N + 1 matched
    assert [func for func, _ in raw_sites(two, table={"x.cpp:f": 1})] == ["f", "f"]
    # a stale row: N listed, fewer matched
    one = H + f"sycl::event ev = {L1};\n{RECORD}(lab, ev);\n}}\n"
    assert [func for func, _ in raw_sites(one, table={"x.cpp:f": 2})] == ["f"]


def test_inside_a_listed_function_the_rule_still_refuses_what_it_can_see():
    table = {"x.cpp:f": 1}
    cases = {
        # one matched launch is listed; each refused shape leaves the row matching nothing, so the launch is reported
        "brace-less do": H + f"sycl::event ev;\ndo ev = {L1}; while (c);\n{RECORD}(lab, ev);\n}}\n",
        "brace-less loop and if": H + f"sycl::event ev;\nwhile (c) if (x) ev = {L1};\n{RECORD}(lab, ev);\n}}\n",
        "brace-less loop and else": H + f"sycl::event ev;\nwhile (c) if (x) g(); else ev = {L1};\n"
                                        f"{RECORD}(lab, ev);\n}}\n",
        "label slot": H + f"sycl::event ev = {L1};\n{RECORD}(ev, other);\n}}\n",
        "record before": H + f"sycl::event ev;\n{RECORD}(lab, ev);\nev = {L1};\n}}\n",
        "overwritten": H + f"sycl::event ev = {L1};\nev = other();\n{RECORD}(lab, ev);\n}}\n",
    }
    for what, sample in cases.items():
        assert [func for func, _ in raw_sites(sample, table=table)] == ["f"], what


def test_the_record_rule_excuses_exactly_the_tables_launches_of_the_tree():
    texts = read_tree()
    excused: list[str] = []
    census({name: texts[name] for name in FILES}, derive_wrappers(texts), texts, recorded=excused)
    counted: dict[str, int] = {}
    for key in excused:
        counted[key] = counted.get(key, 0) + 1
    assert counted == RECORDED_ON_THE_PATH
    assert sum(RECORDED_ON_THE_PATH.values()) == 8


def test_a_forward_that_is_not_the_command_group_does_not_make_a_wrapper():
    helpers = (
        "template <typename F> sycl::event logs(sycl::queue * q, F && f) {\n"
        "    return ggml_sycl_profile_submit(q, label, [&](sycl::queue & pq) { log(std::forward<F>(f)); "
        "return pq.submit(cgf); });\n"
        "}\n"
        "template <typename F> sycl::event labels(sycl::queue * q, F && f) {\n"
        "    return ggml_sycl_profile_submit(q, make_label(std::forward<F>(f)), cgf);\n"
        "}\n"
    )
    assert derive_wrappers({"x.cpp": helpers}) == {BASE_WRAPPER}


def test_a_launch_split_over_lines_in_an_alternative_branch_is_counted():
    for launch, kind in (("q->\n        parallel_for(r, k);", "shortcut"),
                         ("q->parallel_for<\n        name>(r, k);", "shortcut"),
                         ("q\n        ->submit(cgf);", "submit")):
        sample = f"void a(sycl::queue * q) {{\n#ifdef FOO\n    log();\n#else\n    {launch}\n#endif\n}}\n"
        assert raw_sites(sample) == [("a", kind)], sample


def test_a_helper_that_forwards_under_any_name_is_a_wrapper():
    helpers = (
        "template <typename F> static sycl::event launch_named(sycl::queue & q, const char * n, F && f) {\n"
        "    return ggml_sycl_profile_submit(q, label(n), static_cast<F &&>(f));\n"
        "}\n"
        "template <typename F> static sycl::event outer(sycl::queue & q, F && f) {\n"
        "    return launch_named(q, \"x\", std::forward<F>(f));\n"
        "}\n"
        "void a(sycl::queue & q) { outer(q, [&](sycl::queue & pq) { return pq.parallel_for(r, k); }); }\n"
        "void b(sycl::queue & q) { other(q, [&](sycl::queue & pq) { return pq.parallel_for(r, k); }); }\n"
    )
    wrappers = derive_wrappers({"x.cpp": helpers})
    assert wrappers == {BASE_WRAPPER, "launch_named", "outer"}
    assert raw_functions(helpers, wrappers) == ["b"]


def test_a_raw_queue_shortcut_is_reported_and_the_same_shortcut_in_a_wrapper_is_not():
    sample = (
        "void a(sycl::queue * stream) { stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { f(i); }); }\n"
        "void b(sycl::queue & q) {\n"
        "    ggml_sycl_profile_submit(q, label, [&](sycl::queue & pq) { return pq.parallel_for(r, k); });\n"
        "}\n"
        "void c(dpct::queue_ptr s) { s->memcpy(dst, src, n); }\n"
        "void d(ggml_backend_sycl_context & ctx) { ctx.stream()->single_task(k); }\n"
        "struct ggml_backend_sycl_context { queue_ptr stream(); };\n"
    )
    assert raw_sites(sample) == [("a", "shortcut"), ("c", "shortcut"), ("d", "shortcut")]


def test_a_handler_call_inside_a_raw_submit_yields_one_site_the_submit():
    sample = (
        "void a(sycl::queue * q) {\n"
        "    q->submit([&](sycl::handler & cgh) { cgh.parallel_for(r, k); cgh.memcpy(d, s, n); });\n"
        "}\n"
        "static void launch(sycl::handler & cgh) { cgh.parallel_for(r, k); }\n"
    )
    skipped: list[str] = []
    assert raw_sites(sample, skipped) == [("a", "submit")]
    assert skipped == [
        "x.cpp:a .parallel_for( inside a submit",
        "x.cpp:a .memcpy( inside a submit",
        "x.cpp:launch .parallel_for( receiver cgh is handler",
    ]


def test_std_memcpy_and_a_member_call_on_a_non_queue_object_are_not_reported_and_a_queue_fill_is():
    sample = (
        "void a(sycl::queue * q, char * d, const char * s, size_t n) {\n"
        "    std::memcpy(d, s, n);\n"
        "    memcpy(d, s, n);\n"
        "    std::array<int, 4> arr;\n"
        "    arr.fill(0);\n"
        "    staging_buffer buf;\n"
        "    buf.copy(d, n);\n"
        "}\n"
        "void b(sycl::queue & q2) { q2.fill(ptr, 0, n); }\n"
    )
    skipped: list[str] = []
    assert raw_sites(sample, skipped) == [("b", "shortcut")]
    assert skipped == [
        "x.cpp:a .fill( receiver arr is other:std::array",
        "x.cpp:a .copy( receiver buf is other:staging_buffer",
    ]


def test_a_receiver_of_unknown_type_is_named_unresolved():
    sample = "void a() { auto q = make_queue(); q.parallel_for(r, k); }\n"
    skipped: list[str] = []
    assert raw_sites(sample, skipped) == []
    assert skipped == ["x.cpp:a .parallel_for( receiver q is unresolved"]


# every shortcut on the path set aside by its receiver's type, as "file:function .method( receiver type"
EXCLUDED_SHORTCUTS = [
    # the command groups the CPU executor hands to cpu_submit_async; that submit is the site
    "cpu-dispatch.cpp:cpu_binary_op .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_glu .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_norm .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_rms_norm .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_rope .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_scale .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_silu .parallel_for( receiver cgh is handler",
    "cpu-dispatch.cpp:cpu_soft_max .parallel_for( receiver cgh is handler",
    "ggml-sycl.cpp:ggml_sycl_plan_dry_run_plans .fill( receiver ready is other:std::array",
    "ggml-sycl.cpp:ggml_sycl_reset_model_load_scratch_state .fill( "
    "receiver g_moe_expert_vram_reserve is other:std::array",
    "ggml-sycl.cpp:populate_inventory_globals .fill( receiver g_moe_expert_vram_reserve is other:std::array",
    # the GRF-mode helper takes the command group's handler; its callers submit
    "mmq.cpp:ggml_sycl_parallel_for_grf .parallel_for( receiver cgh is handler",
    "mmq.cpp:ggml_sycl_parallel_for_grf .parallel_for( receiver cgh is handler",
    "mmq.cpp:ggml_sycl_parallel_for_grf .parallel_for( receiver cgh is handler",
    "unified-cache.cpp:arena_pp_profile_snapshot_zones .fill( receiver cap_out is other:std::array",
    "unified-cache.cpp:arena_pp_profile_snapshot_zones .fill( receiver used_out is other:std::array",
    "unified-cache.cpp:unified_cache::configure_host_zones .fill( "
    "receiver host_zone_reserved_bytes_ is other:std::array",
    "unified-cache.cpp:unified_cache::configure_host_zones .fill( "
    "receiver host_zone_reserved_bytes_ is other:std::array",
]


def test_every_shortcut_set_aside_on_the_path_is_a_known_non_queue_and_none_is_unresolved():
    skipped: list[str] = []
    census_tree(read_tree(), skipped)
    by_type = sorted(entry for entry in skipped if "inside a submit" not in entry)
    assert not [entry for entry in by_type if entry.endswith("unresolved")], by_type
    assert by_type == EXCLUDED_SHORTCUTS


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
    unnamed = [f"{name}:{line}" for name, sites in found.items() for line, func, _ in sites if func == "?"]
    assert found and not unnamed, f"no census, or sites with no enclosing function: {unnamed}"


def test_a_raw_queue_shortcut_added_to_a_real_file_is_reported():
    texts = read_tree()
    probe = "\nstatic void census_probe_added(sycl::queue * q) { q->parallel_for(r, [=](sycl::id<1> i) {}); }\n"
    texts["norm.cpp"] = texts["norm.cpp"] + probe
    problems = diff_against_allow(census_keys(census_tree(texts)), ALLOW)
    assert problems == ["shortcut:norm.cpp:census_probe_added: 1 raw site(s) in the tree, 0 listed"]


def test_a_raw_submit_added_to_a_real_file_is_reported():
    texts = read_tree()
    probe = "\nstatic void census_probe_added(sycl::queue * q) { q->submit([&](sycl::handler & cgh) {}); }\n"
    texts["norm.cpp"] = texts["norm.cpp"] + probe
    problems = diff_against_allow(census_keys(census_tree(texts)), ALLOW)
    assert problems == ["submit:norm.cpp:census_probe_added: 1 raw site(s) in the tree, 0 listed"]


def test_a_second_raw_submit_in_a_listed_function_is_reported():
    texts = read_tree()
    found = census_tree(texts)
    assert found, "no raw submit found in the tree"
    name = sorted(found)[0]
    line, func, kind = next(site for site in found[name] if site[2] == "submit")
    lines = texts[name].split("\n")
    lines.insert(line, "    q->submit([&](sycl::handler & cgh) {});")
    texts[name] = "\n".join(lines)
    keys = census_keys(census_tree(texts))
    allow = {key: (count, "listed") for key, count in census_keys(found).items()}
    listed = allow[f"submit:{name}:{func}"][0]
    expected = f"submit:{name}:{func}: {listed + 1} raw site(s) in the tree, {listed} listed"
    assert diff_against_allow(keys, allow) == [expected]


def test_a_listed_function_whose_submits_are_gone_fails():
    texts = read_tree()
    found = census_tree(texts)
    assert found, "no raw submit found in the tree"
    allow = {key: (count, "listed") for key, count in census_keys(found).items()}
    name = sorted(found)[0]
    func = next(site[1] for site in found[name] if site[2] == "submit")
    lines = texts[name].split("\n")
    for line, f, kind in found[name]:
        if f == func and kind == "submit":
            lines[line - 1] = SUBMIT.sub(".submitted_elsewhere(", lines[line - 1])
    texts[name] = "\n".join(lines)
    problems = diff_against_allow(census_keys(census_tree(texts)), allow)
    key = f"submit:{name}:{func}"
    assert problems == [f"{key}: 0 raw site(s) in the tree, {allow[key][0]} listed"]


def launching_sources() -> list[str]:
    """Every .cpp/.hpp under ggml/src/ggml-sycl whose code (comments and literals blanked) holds a launch-like call of
    either class, before any receiver or wrapper test, as a path relative to it."""
    found = []
    for path in sorted(SYCL.rglob("*")):
        if path.suffix not in (".cpp", ".hpp") or not path.is_file():
            continue
        code = blank(path.read_text(encoding="utf-8", errors="replace"))
        if SUBMIT.search(code) or SHORTCUT.search(code):
            found.append(path.relative_to(SYCL).as_posix())
    return found


def test_every_launching_file_is_on_the_path_or_excluded():
    on_path = set(FILES)
    unread = [name for name in launching_sources()
              if name not in on_path and name not in EXCLUDED_FILES
              and not any(name.startswith(d) for d in EXCLUDED_DIRS)]
    assert not unread, f"in neither FILES nor EXCLUDED_FILES: {unread}"
    assert not on_path & set(EXCLUDED_FILES), "a file is both on the path and excluded"
    assert all(reason.strip() for reason in EXCLUDED_FILES.values())
    stale = [name for name in EXCLUDED_FILES if not (SYCL / name).exists()]
    assert not stale, f"excluded files that are gone: {stale}"


def test_every_allowlist_entry_has_a_reason():
    assert all(count > 0 and reason.strip() for count, reason in ALLOW.values())


def test_census_matches_the_allowlist():
    t0 = time.monotonic()
    texts = read_tree()
    found = census_tree(texts)
    elapsed = time.monotonic() - t0
    print(format_sites(found))
    print(count_by_kind_and_file(found))
    print(f"{sum(len(s) for s in found.values())} raw site(s) in {len(census_keys(found))} key(s) "
          f"over {len(FILES)} file(s), census took {elapsed:.2f} s")
    problems = diff_against_allow(census_keys(found), ALLOW)
    assert not problems, "\n".join(problems) + "\n\nraw submits found:\n" + format_sites(found)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
