#!/usr/bin/env python3
"""Every device launch on the prompt (window) path is either wrapped by the kernel profiler or listed here.

A launch the profiler cannot see lands in "device idle" or in nothing, so the attribution table of a prompt ubatch is
only as complete as this list is short. The list shrinks to empty; it never grows without a reason beside the entry.

To update (`python3 tests/test-sycl-window-path-submit-census-source.py --print` lists every key with its count and
file:line, the ALLOW lines to paste, what the record rule matched per function, and, for the files on the path,
what each SET_ASIDE_BY_RECEIVER row set aside and which rows do not match):
  a. You wrapped a site: its key's tree count dropped. Lower that ALLOW count, or delete the entry at 0.
  b. You added a launch on the path: wrap it. If it cannot be wrapped yet, add or raise its ALLOW entry with what it
     is and what a wrap task must do (the WRAP_ texts).
  c. A new source file holds a launch: test_every_launching_file_is_on_the_path_or_excluded names it. Add it to FILES
     citing the call that reaches it from the prompt graph, or to EXCLUDED_FILES with why nothing on the path does;
     a NOT_INCLUDED reason is checked against the #include lines, and a receiver-rule reason against the census.
  d. A launch is profiled through its event (a record call is handed what the launch returned): read the function,
     confirm each record call receives that launch's event, and set the function's RECORDED_ON_THE_PATH row to the
     number of such launches. Nothing is excused without a row.
  e. A SET_ASIDE_BY_RECEIVER row does not match, or you want to add one (set_aside_problems names a row's calls by
     file:line, or why it was refused). Adding or raising a row is the reviewed act: read the declaration of each
     receiver by hand. If it is a queue, delete the row: its calls are launches. If it is no queue, set the row's
     count to the calls on that receiver in its one function, with a reason saying what the receiver is, where it is
     declared, and where any device work its call leads to is counted. The pin proves only that the set of set-aside
     receiver expressions and their counts did not change, never that a receiver is no queue.

Two kinds of launch are censused, and kept apart in the output and in the allowlist keys:
- submit: a `->submit(` or `.submit(` call. A submit whose command group is only a `host_task` counts too: it occupies
  the in-order device queue, and the device sits idle while it runs, so the profiler must see it like a kernel.
- shortcut: a queue member that enqueues a command without submit(): `parallel_for`, `single_task`, `memcpy`, `memset`,
  `fill`, `copy`, `prefetch`, `mem_advise`, in member-call form only (`->name(` / `.name(`), so `std::memcpy(` and free
  functions never match. Inside a submit's parentheses, the innermost enclosing submit decides: if it counts, the
  shortcut is its command group's handler call and the submit is the launch, so the shortcut is not counted; if a
  row sets it aside (a host pool's job), the shortcut counts like any other, whatever encloses that submit.
Every call of either kind counts unless a reviewed row sets it aside. No type is read anywhere: a row of
SET_ASIDE_BY_RECEIVER, keyed `kind:file:function:receiver`, names the receiver expression exactly as written at the
call, prefix included and whitespace removed, and sets aside the calls of that kind on that expression in that function
only when there are exactly as many as the row's count, all in one definition of it (a row naming an overload pair, or
one name in two namespaces, is refused), and a named one (a row for calls outside every named function is refused). A
receiver a row can name is a plain chain of names joined by `.`, `->` or `::` (`pool`, `engine->queue`, `this->pool_`,
`ns::g_pool`); a call, an index, a cast, a template argument, a parenthesised or dereferenced operand, a `*` or `&`
before it, a line continuation and a keyword joined by `::` make it nameless, so its call counts. A call on `c->q` is
not one on `q`. Adding a row is the reviewed act: the reviewer reads the receiver's declaration by hand, and the row's
reason says what it is and where it is declared. The pin proves only that the set of set-aside expressions and their
counts did not change; that a receiver is no queue is the reviewer's claim. queue_tripwire is a lexical tripwire on top,
not a resolver: it refuses a row whose last name the function plainly binds next to the token `queue` (`sycl::queue &
ready`, `for (sycl::queue * ready : qs)`), and proves nothing when it stays quiet.

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
default route, and the comment beside it cites the dispatch site by function. An opt-in route may also be listed when
its comment and its entries name it as one (the XMX sorted MoE route, tensor parallelism), so that it stays censused.
Every other source under ggml/src/ggml-sycl that holds a launch-like call is in EXCLUDED_FILES (or under an
EXCLUDED_DIRS prefix) with its reason; that table is the source of truth, and
test_every_launching_file_is_on_the_path_or_excluded fails on a file in neither. Challenge an exclusion by citing the
call that reaches the file; then move it to FILES and its launches to ALLOW.

Run directly or under pytest; a failing comparison names each key, its launches and what to do, and `--print` writes
the full report.
"""
from __future__ import annotations

import bisect
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sycl_gate import ROOT  # noqa: E402

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
    # TOP_K: ggml_top_k (the qwen4exp graph's expert selection) -> ggml_sycl_top_k (ggml-sycl.cpp) -> ggml_sycl_op_top_k
    "topk.cpp",
    "diag.cpp",  # DIAG: ggml_diag (delta-net-base.cpp, the chunked delta rule) -> ggml_sycl_op_diag
    # hc_pre_launch, hc_comb_launch, hc_post_launch, called by dsv4-hc.cpp for DSV4_HC_PRE / DSV4_HC_POST
    # (the qwen4exp graph's ggml_dsv4_hc_pre_gated and ggml_dsv4_hc_post)
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
    # the BF16 dense-weight GEMM: ggml_sycl_mul_mat (ggml-sycl.cpp) hands every MUL_MAT whose weight is BF16 to
    # ggml_sycl_mul_mat_bf16_weight -> mul_mat_bf16_f32; Qwen3.8's routers, hyper-connection projections, q_proj,
    # k_proj, ssm_alpha and ssm_beta are BF16 (the comment above ggml_sycl_mul_mat_bf16_weight names them)
    "mul-mat-bf16.hpp",
    # the fused RMS_NORM + MUL + MUL_MAT: graph compute's fusion detection calls ggml_sycl_mul_mat_with_rmsnorm ->
    # fused_rmsnorm_quantize_q8_1_sycl on the default route (fusion site bit 3) when ggml_sycl_can_fuse_rmsnorm_mulmat
    # admits it: at least 8 rows, a Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K weight, and a shape the oneDNN prompt GEMM declines
    "fused-norm-gemm.hpp",
    # the prompt fusion's transaction (mmvq_submit_retained_prompt_fusion -> submit_prompt_fusion): on the path, and
    # its two submits, on transaction and submitter, are SET_ASIDE_BY_RECEIVER rows
    # (SET_ASIDE_BY_RECEIVER); its device work is submitted from mmvq.cpp
    "moe-fused-transaction.cpp",
    # quantized_allreduce, called by ggml_sycl_mul_mat_tp_row_parallel_post (ggml-sycl.cpp), which ggml_sycl_mul_mat
    # calls on every MUL_MAT; it runs for a TP-sharded row-parallel weight, where the quantized all-reduce is on by
    # default (GGML_SYCL_QUANT_ALLREDUCE unset reads as 1). Listed by the rule that already censuses that function's
    # own launches: tensor parallelism is opt-in, so like the XMX sorted MoE route it is kept so it stays censused
    "quantized-comm.hpp",
    # the CPU executor: should_dispatch_to_cpu (ggml-sycl.cpp) routes a MUL_MAT whose weight resolves to host, with no
    # opt-in, and the ops of a CPU-bound layer, to ggml_sycl_compute_forward_cpu; its submits to the device queue are
    # host_task bodies (a host_task on the device queue counts, see the docstring)
    "cpu-dispatch.cpp",
]

# Every other source under ggml/src/ggml-sycl that holds a launch-like call (either class, before any receiver or
# wrapper test), with the one reason it is not on the window path. test_every_launching_file_is_on_the_path_or_excluded
# fails on a file in neither list, so a new file, or a file a new call reaches, cannot go unread.
NOT_BUILT = "an op the qwen4exp graph does not build (qwen4exp.cpp, delta-net-base.cpp and llama-graph.cpp name none)"
# test_every_nothing_includes_it_reason_is_true checks every reason built on this against the #include lines
NOT_INCLUDED = "no source outside tests/ includes it"
# test_every_file_excused_by_the_receiver_rule_censuses_to_nothing checks every reason that cites this
BY_RECEIVER_CHECKED = "censused to no site by test_every_file_excused_by_the_receiver_rule_censuses_to_nothing"
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
    "set_rows_paged.hpp": "SET_ROWS_PAGED (included by ggml-sycl.cpp, which dispatches GGML_OP_SET_ROWS_PAGED to "
                          "ggml_sycl_op_set_rows_paged): " + NOT_BUILT,
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
    "lazy-io-pool.cpp": "the weight loader's I/O pool, not an op of the graph; and its one submit is "
                        "lazy_io_queue::submit on engine->queue, a SET_ASIDE_BY_RECEIVER row ("
                        + BY_RECEIVER_CHECKED + ")",
    "moe-tile-convert.cpp": "MXFP4 to XMX-tiled weight conversion, run when a weight is materialized "
                            "(ggml_sycl_fill_xmx_tiled, convert_tensor_layout), not by an op of the graph",
    "moe-mmid-workspace.cpp": "no launch: its .fill( calls are std::array::fill in moe_mmid_workspace_registry::admit",
    "mxfp4-stored-gemm.cpp": "its header mxfp4-stored-gemm.hpp (besides this file): " + NOT_INCLUDED,
    "mmvq-rmsnorm.hpp": NOT_INCLUDED,
    "l2-prefetch.hpp": "the L2 prefetch manager (ggml-sycl.cpp makes one only when is_l2_prefetch_enabled(), opt-in "
                       "GGML_SYCL_L2_PREFETCH=1, and calls it only for a one-row src1): decode only",
}

# The packs this census serves: /models/Qwen3.8-Flash-Next-GGUF/, UD-Q4_K_XL and exps-IQ4_NL.
TARGET_PACKS = "the target packs (Qwen3.8-Flash-Next UD-Q4_K_XL and exps-IQ4_NL)"

# What a wrap task does, by launch form. Every reason that asks for a wrap ends with one of these.
WRAP_SUBMIT = "to wrap: hand this command group to ggml_sycl_profile_submit with a label"
WRAP_SHORTCUT = "to wrap: make the same call on the handler, in a command group handed to ggml_sycl_profile_submit"
WRAP_USM = "to wrap: issue it as cgh.memcpy / cgh.memset in a command group handed to ggml_sycl_profile_submit"
WRAP_HOST_TASK = ("to wrap: hand the host_task's command group to ggml_sycl_profile_submit, which then times the slot "
                  "the host work holds on the device queue")


def kernel(what: str) -> str:
    """The reason of a raw submit of a kernel on the path."""
    return f"{what}; {WRAP_SUBMIT}"


def shortcut(what: str) -> str:
    """The reason of a queue shortcut (parallel_for / single_task) that launches a kernel on the path."""
    return f"{what}, launched by a queue shortcut; {WRAP_SHORTCUT}"


def host_task(what: str) -> str:
    return f"a host_task on the device queue that {what}; {WRAP_HOST_TASK}"


def fused_esimd(cases: str = "") -> str:
    return f"the fused ESIMD MoE (ggml_sycl_mul_mat_id_fused, on by default){cases}; {WRAP_SUBMIT}"


def bf16(which: str) -> str:
    return ("the BF16 dense-weight GEMM: ggml_sycl_mul_mat_bf16_weight (ggml-sycl.cpp) calls mul_mat_bf16_f32, which "
            f"picks {which}; the caller takes a profile scope (GGML_SYCL_PROFILE_SCOPE_GEMM) and discards the event, "
            f"which times the host call but does not wrap the launch; {WRAP_SUBMIT}")


# Reasons a wrap task leaves the launch.
MARKER = "an empty marker kernel with no work time, which a wrap task leaves: "
WEIGHT_LAYOUT = "a weight layout conversion: runs when a weight is placed, not per ubatch, so a wrap task leaves it"
TEST_ENTRY = "a test or benchmark entry point that graph compute does not call; a wrap task leaves it"
NO_MXFP4 = TARGET_PACKS + " hold no MXFP4 tensor, so a wrap task for them may leave it"
MXFP4 = "an MXFP4 kernel; " + NO_MXFP4
MOE_XMX = ("opt-in GGML_SYCL_XMX_MOE_SORTED=1 (try_xmx_sorted_moe), kept on the list so the opt-in route stays "
           "censused; a wrap task leaves it while the route is opt-in")
MOE_XMX_SORT = MOE_XMX + ": the sort's dependency barrier and the two sorted_token_ids extracts"
XMX_DETAIL_PROBES = ("debug only: the GGML_SYCL_XMX_DETAIL timing probes (real, empty and SLM-only kernels between "
                     "q.wait() calls), which a wrap task leaves; the production launch below them in the same "
                     "function is wrapped (ggml_sycl_profile_submit)")

# Shared descriptions, completed by kernel() / shortcut() / a WRAP_ text.
TP = "tensor parallelism only (a TP-sharded row-parallel weight): "
ATTENTION = "an attention kernel that fattn.cpp picks per shape"
VECTOR = "an MMVQ kernel (a staged ring slot is served by the plain MMVQ_COMPAT kernel)"
DEQUANT = ("a dequantize-to-f16 kernel of convert.cpp's to_fp16 tables (the oneDNN GEMM's and the window executor's "
           "weight dequant)")
DEQUANT_QK_K = (DEQUANT + "; one per branch of #if QK_K == 256, and the #else branch is dead code (ggml-common.h "
                "defines QK_K as 256)")
DEQUANT_HELPER = ("a dequantize-to-f16 helper: the window executor wraps its call (dequant_expert_f16) and other "
                  "callers do not; to wrap: wrap those callers' calls, so the executor's spans are not counted twice")
FUSED_NORM = "the fused RMS_NORM + MUL + MUL_MAT (ggml_sycl_mul_mat_with_rmsnorm): "
FUSED_ESIMD_CASES = ("; one launch per case: the direct f32 kernel under graph recording or without Q8 scratch, "
                     "else DP4A")
CPU_HOST_TASK = ("a host-placed op run on the CPU as a host_task on the device queue (the in-order queue orders it "
                 "with the device kernels); " + WRAP_HOST_TASK)
CPU_SUBMIT_HELPER = ("the CPU executor's submit helper: a kernel on the CPU queue, or a host_task on the device queue "
                     "for the CPU matmul and fused paths (run_mul_mat, run_fused); " + WRAP_HOST_TASK)
MUL_MAT_ID_COPIES = (
    "two are host_task submits of the Q1/NVFP4 exact route, ordering and completion rather than device kernels, "
    "which a wrap task leaves: the injected terminal failure (test-only, under GGML_SYCL_Q1_NVFP4_ROUTE_TESTING) and "
    "the completion that follows terminal->arm(final); two are prompt row copies: the MXFP4 SOA f16 activation "
    "staging (f16_staging_event) and the expert output scatter; for those two, " + WRAP_SUBMIT
)

# "kind:file:function" -> (raw sites of that kind in it, what they are and what a wrap task does with them). A named
# lambda is `function/lambda`. The kind is `submit` or `shortcut`, so the two classes are driven down separately.
# Kept sorted by key (test_allow_is_written_sorted_by_key_with_no_duplicate); `--print` writes the lines to paste.
ALLOW: dict[str, tuple[int, str]] = {
    "shortcut:concat.cpp:concat_T_sycl": (3, shortcut("CONCAT of contiguous sources, three launches")),
    "shortcut:concat.cpp:concat_T_sycl_non_cont": (1, shortcut("CONCAT of non-contiguous sources")),
    "shortcut:convert.cpp:convert_unary_nc_sycl": (1, shortcut("a non-contiguous type conversion (to_fp16_nc)")),
    "shortcut:convert.cpp:dequantize_block_sycl": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_mxfp4_soa_to_fp16_rowmajor":
        (1, shortcut("an MXFP4 dequantize to f16; " + NO_MXFP4)),
    "shortcut:convert.cpp:dequantize_row_mxfp4_sycl": (1, shortcut("an MXFP4 dequantize to f16; " + NO_MXFP4)),
    "shortcut:convert.cpp:dequantize_row_nvfp4_fp16_sycl": (1, shortcut("the NVFP4 dequantize to f16")),
    "shortcut:convert.cpp:dequantize_row_q2_K_sycl": (2, shortcut(DEQUANT_QK_K)),
    "shortcut:convert.cpp:dequantize_row_q3_K_sycl": (2, shortcut(DEQUANT_QK_K)),
    "shortcut:convert.cpp:dequantize_row_q4_0_soa_to_fp16_rowmajor": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl_coalesced": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl_coalesced_rowmajor": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q4_0_sycl_reorder": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q4_1_sycl": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q5_K_sycl": (2, shortcut(DEQUANT_QK_K)),
    "shortcut:convert.cpp:dequantize_row_q6_K_sycl": (2, shortcut(DEQUANT_QK_K)),
    "shortcut:convert.cpp:dequantize_row_q6_K_sycl_reorder": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q8_0_coalesced_rowmajor": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:dequantize_row_q8_0_soa_to_fp16_rowmajor": (1, shortcut(DEQUANT)),
    "shortcut:convert.cpp:reorder_mxfp4_aos_to_soa_sycl":
        (1, "an AOS to SOA weight reorder (reorder_aos_to_soa_device); " + WEIGHT_LAYOUT),
    "shortcut:convert.cpp:reorder_q4_0_aos_to_soa_sycl":
        (1, "an AOS to SOA weight reorder (reorder_aos_to_soa_device); " + WEIGHT_LAYOUT),
    "shortcut:convert.cpp:reorder_q4_0_coalesced_to_soa_sycl":
        (1, "an AOS to SOA weight reorder (reorder_aos_to_soa_device); " + WEIGHT_LAYOUT),
    "shortcut:convert.cpp:reorder_q4_k_aos_to_soa_sycl":
        (1, "an AOS to SOA weight reorder (reorder_aos_to_soa_device); " + WEIGHT_LAYOUT),
    "shortcut:convert.cpp:reorder_q6_k_aos_to_soa_sycl":
        (1, "an AOS to SOA weight reorder (reorder_aos_to_soa_device); " + WEIGHT_LAYOUT),
    "shortcut:convert.cpp:reorder_q8_0_aos_to_soa_sycl":
        (1, "an AOS to SOA weight reorder (reorder_aos_to_soa_device); " + WEIGHT_LAYOUT),
    "shortcut:convert.cpp:repack_mxfp4_soa_to_woq":
        (2, shortcut("an MXFP4 repack to the oneDNN WOQ layout; " + NO_MXFP4)),
    "shortcut:convert.cpp:repack_mxfp4_soa_to_woq_batched":
        (2, shortcut("an MXFP4 repack to the oneDNN WOQ layout; " + NO_MXFP4)),
    "shortcut:convert.cpp:repack_mxfp4_xmx_tiled_to_woq":
        (2, shortcut("an MXFP4 repack to the oneDNN WOQ layout; " + NO_MXFP4)),
    "shortcut:convert.cpp:repack_mxfp4_xmx_tiled_to_woq_batched":
        (2, shortcut("an MXFP4 repack to the oneDNN WOQ layout; " + NO_MXFP4)),
    "shortcut:cpy.cpp:ggml_cpy_bf16_bf16_sycl": (1, shortcut("CPY bf16 to bf16")),
    "shortcut:cpy.cpp:ggml_cpy_bf16_f16_sycl": (1, shortcut("CPY bf16 to f16")),
    "shortcut:cpy.cpp:ggml_cpy_bf16_f32_sycl": (1, shortcut("CPY bf16 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_f16_bf16_sycl": (1, shortcut("CPY f16 to bf16")),
    "shortcut:cpy.cpp:ggml_cpy_f16_f16_sycl": (1, shortcut("CPY f16 to f16")),
    "shortcut:cpy.cpp:ggml_cpy_f16_f32_sycl": (1, shortcut("CPY f16 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_f32_bf16_sycl": (1, shortcut("CPY f32 to bf16")),
    "shortcut:cpy.cpp:ggml_cpy_f32_f16_sycl": (1, shortcut("CPY f32 to f16")),
    "shortcut:cpy.cpp:ggml_cpy_f32_f32_sycl": (1, shortcut("CPY f32 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_f32_i32_sycl": (1, shortcut("CPY f32 to i32")),
    "shortcut:cpy.cpp:ggml_cpy_f32_iq4_nl_sycl": (1, shortcut("CPY f32 to iq4_nl")),
    "shortcut:cpy.cpp:ggml_cpy_f32_q4_0_sycl": (1, shortcut("CPY f32 to q4_0")),
    "shortcut:cpy.cpp:ggml_cpy_f32_q4_1_sycl": (1, shortcut("CPY f32 to q4_1")),
    "shortcut:cpy.cpp:ggml_cpy_f32_q5_0_sycl": (1, shortcut("CPY f32 to q5_0")),
    "shortcut:cpy.cpp:ggml_cpy_f32_q5_1_sycl": (1, shortcut("CPY f32 to q5_1")),
    "shortcut:cpy.cpp:ggml_cpy_f32_q8_0_sycl": (1, shortcut("CPY f32 to q8_0")),
    "shortcut:cpy.cpp:ggml_cpy_i16_i16_sycl": (1, shortcut("CPY i16 to i16")),
    "shortcut:cpy.cpp:ggml_cpy_i32_f32_sycl": (1, shortcut("CPY i32 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_i32_i32_sycl": (1, shortcut("CPY i32 to i32")),
    "shortcut:cpy.cpp:ggml_cpy_q4_0_f32_sycl": (1, shortcut("CPY q4_0 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_q4_0_q4_0": (1, shortcut("CPY q4_0 to q4_0")),
    "shortcut:cpy.cpp:ggml_cpy_q4_1_f32_sycl": (1, shortcut("CPY q4_1 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_q4_1_q4_1": (1, shortcut("CPY q4_1 to q4_1")),
    "shortcut:cpy.cpp:ggml_cpy_q5_0_f32_sycl": (1, shortcut("CPY q5_0 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_q5_0_q5_0": (1, shortcut("CPY q5_0 to q5_0")),
    "shortcut:cpy.cpp:ggml_cpy_q5_1_f32_sycl": (1, shortcut("CPY q5_1 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_q5_1_q5_1": (1, shortcut("CPY q5_1 to q5_1")),
    "shortcut:cpy.cpp:ggml_cpy_q8_0_f32_sycl": (1, shortcut("CPY q8_0 to f32")),
    "shortcut:cpy.cpp:ggml_cpy_q8_0_q8_0": (1, shortcut("CPY q8_0 to q8_0")),
    "shortcut:cpy.cpp:ggml_cpy_row_contig_sycl": (1, shortcut("CPY of contiguous rows")),
    "shortcut:diag.cpp:ggml_sycl_op_diag": (1, shortcut("DIAG")),
    "shortcut:dsv4-hc-kernels.hpp:hc_comb_launch": (1, shortcut("the DSV4 hyper-connection comb kernel")),
    "shortcut:dsv4-hc-kernels.hpp:hc_post_launch": (1, shortcut("the DSV4 hyper-connection post kernel")),
    "shortcut:dsv4-hc-kernels.hpp:hc_pre_launch": (1, shortcut("the DSV4 hyper-connection pre kernel")),
    "shortcut:element_wise.cpp:acc_f32_sycl": (1, shortcut("ACC")),
    "shortcut:element_wise.cpp:ggml_sycl_op_arange": (1, shortcut("ARANGE")),
    "shortcut:element_wise.cpp:ggml_sycl_op_ceil": (1, shortcut("CEIL")),
    "shortcut:element_wise.cpp:ggml_sycl_op_clamp": (1, shortcut("CLAMP")),
    "shortcut:element_wise.cpp:ggml_sycl_op_cos": (1, shortcut("COS")),
    "shortcut:element_wise.cpp:ggml_sycl_op_floor": (1, shortcut("FLOOR")),
    "shortcut:element_wise.cpp:ggml_sycl_op_geglu": (1, shortcut("GEGLU")),
    "shortcut:element_wise.cpp:ggml_sycl_op_geglu_erf": (1, shortcut("GEGLU_ERF")),
    "shortcut:element_wise.cpp:ggml_sycl_op_geglu_quick": (1, shortcut("GEGLU_QUICK")),
    "shortcut:element_wise.cpp:ggml_sycl_op_leaky_relu": (1, shortcut("LEAKY_RELU")),
    "shortcut:element_wise.cpp:ggml_sycl_op_log": (1, shortcut("LOG")),
    "shortcut:element_wise.cpp:ggml_sycl_op_reglu": (1, shortcut("REGLU")),
    "shortcut:element_wise.cpp:ggml_sycl_op_round": (1, shortcut("ROUND")),
    "shortcut:element_wise.cpp:ggml_sycl_op_sin": (1, shortcut("SIN")),
    "shortcut:element_wise.cpp:ggml_sycl_op_sqr": (1, shortcut("SQR")),
    "shortcut:element_wise.cpp:ggml_sycl_op_sqrt": (1, shortcut("SQRT")),
    "shortcut:element_wise.cpp:ggml_sycl_op_swiglu": (1, shortcut("SWIGLU")),
    "shortcut:element_wise.cpp:ggml_sycl_op_trunc": (1, shortcut("TRUNC")),
    "shortcut:element_wise.cpp:ggml_sycl_op_unary": (1, shortcut("UNARY (the generic table)")),
    "shortcut:element_wise.cpp:swiglu_oai_sycl": (1, shortcut("SWIGLU_OAI")),
    "shortcut:element_wise.cpp:upscale_sycl": (1, shortcut("UPSCALE")),
    "shortcut:fattn-onednn.cpp:ggml_sycl_flash_attn_ext_onednn_materialize_q_f32":
        (1, shortcut("materializes Q as dense f16 for the oneDNN SDPA")),
    "shortcut:fattn-onednn.cpp:ggml_sycl_onednn_fa_materialize_one":
        (2, shortcut("materializes K and V as dense f16 for the oneDNN SDPA (MATERIALIZE_REQUIRED)")),
    "shortcut:fattn.cpp:ggml_sycl_flash_attn_ext":
        (2, shortcut("fills the paged-attention v2 block table and sequence lengths before each call")),
    "shortcut:fattn.cpp:ggml_sycl_v2_pre_allocate_buffers":
        (1, shortcut("fills the paged-attention v2 block table when its buffers are allocated")),
    "shortcut:fill.cpp:ggml_sycl_op_fill": (2, shortcut("FILL, two launches")),
    "shortcut:fused-norm-gemm.hpp:quantize_with_rmsnorm_q8_1_sycl":
        (1, shortcut(FUSED_NORM + "its normalize-and-quantize to Q8_1")),
    "shortcut:gated_delta_net.cpp:launch_gated_delta_net": (4, shortcut("GATED_DELTA_NET, four launches")),
    "shortcut:getrows.cpp:get_rows_q4_0_coalesced_sycl": (1, shortcut("GET_ROWS of q4_0, coalesced")),
    "shortcut:getrows.cpp:get_rows_q4_k_aos_sycl": (1, shortcut("GET_ROWS of q4_k, aos")),
    "shortcut:getrows.cpp:get_rows_q6_k_aos_sycl": (1, shortcut("GET_ROWS of q6_k, aos")),
    "shortcut:getrows.cpp:get_rows_q6_k_coalesced_sycl": (1, shortcut("GET_ROWS of q6_k, coalesced")),
    "shortcut:getrows.cpp:get_rows_q6_k_coalesced_variable_sycl": (1, shortcut("GET_ROWS of q6_k, coalesced variable")),
    "shortcut:getrows.cpp:get_rows_q6_k_soa_sycl": (1, shortcut("GET_ROWS of q6_k, soa")),
    "shortcut:getrows.cpp:get_rows_q8_0_aos_pair_sycl": (1, shortcut("GET_ROWS of q8_0, aos pair")),
    "shortcut:getrows.cpp:get_rows_q8_0_aos_sycl": (1, shortcut("GET_ROWS of q8_0, aos")),
    "shortcut:getrows.cpp:get_rows_q8_0_coalesced_sycl": (1, shortcut("GET_ROWS of q8_0, coalesced")),
    "shortcut:getrows.cpp:get_rows_sycl": (1, shortcut("GET_ROWS of a quantized type, AOS")),
    "shortcut:getrows.cpp:get_rows_sycl_float": (1, shortcut("GET_ROWS of f32 / f16")),
    "shortcut:getrows.cpp:get_rows_sycl_reorder": (1, shortcut("GET_ROWS of a reordered (SOA) quantized type")),
    "shortcut:ggml-sycl.cpp:diag_mask_inf_f32_sycl": (1, shortcut("DIAG_MASK_INF")),
    "shortcut:ggml-sycl.cpp:ggml_mul_mat_p021_f16_f32_sycl": (1, shortcut("the permuted f16 x f32 MUL_MAT (p021)")),
    "shortcut:ggml-sycl.cpp:ggml_mul_mat_vec_nc_f16_f32_sycl":
        (1, shortcut("the non-contiguous f16 x f32 vector MUL_MAT")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_copy_ids_to_host":
        (1, shortcut("packs the MUL_MAT_ID ids for their copy to the host, the second route")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_cpy_tensor_2d": (3, shortcut("a 2D tensor copy as int32 words, three launches")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_mul_mat_batched_f16_fallback":
        (1, shortcut("the batched f16 x f32 fallback GEMM")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_mul_mat_tp_row_parallel_post":
        (8, shortcut(TP + "the row-parallel post-processing kernels, eight launches")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_op_mean": (1, shortcut("MEAN's divide by the row length")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_op_mul_mat":
        (1, shortcut("copies an f32 MUL_MAT result into its f16 destination")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_op_pool2d": (1, shortcut("POOL_2D")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_readback_via_shared_kernel":
        (1, shortcut("a device readback through shared memory")),
    "shortcut:ggml-sycl.cpp:ggml_sycl_tp_launch_async_ffn": (1, shortcut(TP + "the async FFN's GLU")),
    "shortcut:ggml-sycl.cpp:scale_f32_sycl": (1, shortcut("SCALE")),
    "shortcut:ggml-sycl.cpp:sum_rows_f32_sycl": (1, shortcut("SUM_ROWS")),
    "shortcut:ggml-sycl.cpp:tp_device1_worker_thread_func": (1, shortcut(TP + "device 1's worker GLU")),
    "shortcut:mem-ops.cpp:mem_copy_ptr_async":
        (1, "the window ring's copy helper (mem_copy_async), no dependencies: queue.memcpy; " + WRAP_USM),
    "shortcut:moe-sort.hpp:moe_convert_f32_to_f16": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_convert_f32_to_f16_async": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_count_tokens_per_expert": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_scatter_results": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_scatter_results_f16_to_f32": (1, MOE_XMX),
    "shortcut:moe-sort.hpp:moe_sort_tokens_by_expert": (1, MOE_XMX),
    "shortcut:moe-xmx-fused.hpp:fused_xmx_moe_gemm_q8_0": (1, MOE_XMX),
    "shortcut:moe-xmx.hpp:extract_q8_0_scales": (1, MOE_XMX),
    "shortcut:pad.cpp:pad_f32_sycl": (1, shortcut("PAD")),
    "shortcut:set.cpp:ggml_sycl_op_set": (1, shortcut("SET")),
    "shortcut:unified-kernel.cpp:UnifiedKernel::execute_deferred_copies":
        (1, "the unified kernel's deferred copies, queue.memcpy; " + WRAP_USM),
    "shortcut:unified-kernel.cpp:UnifiedKernel::reset_dag_counters":
        (4, "the unified kernel's DAG counter reset, a queue.memcpy and three queue.memset; " + WRAP_USM),
    "submit:common.hpp:ggml_sycl_submit_marker":
        (1, MARKER + "the window's timing marks and the executor's GEMM span marks, on every caller's behalf"),
    "submit:convert.cpp:dequantize_row_iq1_m_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq1_s_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq2_s_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq2_xs_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq2_xxs_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq3_s_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq3_xxs_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq4_nl_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_iq4_xs_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_q4_K_sycl": (1, DEQUANT_HELPER),
    "submit:convert.cpp:dequantize_row_q4_K_sycl_reorder": (1, WEIGHT_LAYOUT),
    "submit:convert.cpp:reorder_mxfp4_aos_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:convert.cpp:reorder_q4_0_aos_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:convert.cpp:reorder_q6_k_aos_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:convert.cpp:reorder_q8_0_aos_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:cpu-dispatch.cpp:cpu_cpy": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_glu": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_norm": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_rms_norm": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_rope": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_scale": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_silu": (1, CPU_HOST_TASK),
    "submit:cpu-dispatch.cpp:cpu_submit_async": (1, CPU_SUBMIT_HELPER),
    "submit:cumsum.cpp:ggml_sycl_op_cumsum": (1, kernel("CUMSUM")),
    "submit:expert-prefetch.cpp:ExpertPredictor::predict_pregate":
        (1, kernel("the expert predictor's gate GEMV, run while the predictor is active")),
    "submit:fattn-mma.hpp:launch_fattn_mma": (1, kernel(ATTENTION)),
    "submit:fattn-tile-f16.hpp:launch_fattn_tile_f16": (1, kernel(ATTENTION)),
    "submit:fattn-v2-partition.hpp:launch_paged_attention_v2": (2, kernel(ATTENTION)),
    "submit:fattn-vec-f16.hpp:launch_fattn_vec_f16": (1, kernel(ATTENTION)),
    "submit:fattn-vec.hpp:launch_fattn_vec": (1, kernel(ATTENTION)),
    "submit:fused-moe-esimd.hpp:launch_fused_moe_mxfp4_impl": (3, fused_esimd(FUSED_ESIMD_CASES)),
    "submit:fused-moe-esimd.hpp:launch_fused_moe_mxfp4_soa":
        (3, fused_esimd("; one launch per hidden-size case (2880, 4096, any other)")),
    "submit:fused-moe-esimd.hpp:launch_fused_moe_q8_0": (1, fused_esimd()),
    "submit:fused-moe-esimd.hpp:launch_persistent_moe_mxfp4_impl":
        (3, fused_esimd("; the persistent variant, for at most 8 tokens" + FUSED_ESIMD_CASES)),
    "submit:fused-norm-gemm.hpp:compute_rms_scales_sycl":
        (2, kernel(FUSED_NORM + "its RMS scales, one launch per block-size branch")),
    "submit:ggml-sycl.cpp:argmax_f32_i32_sycl": (1, kernel("ARGMAX")),
    "submit:ggml-sycl.cpp:argsort_f32_i32_sycl": (2, kernel("ARGSORT, one launch per sort order")),
    "submit:ggml-sycl.cpp:dispatch_experts_secondary_gpu_impl":
        (1, kernel("the secondary GPU's expert-output reduce (reduce_event)")),
    "submit:ggml-sycl.cpp:ggml_sycl_add_f32": (1, kernel("an in-place f32 add (dst += src), waited on at once")),
    "submit:ggml-sycl.cpp:ggml_sycl_copy_ids_to_host":
        (1, kernel("packs the MUL_MAT_ID ids for their copy to the host")),
    "submit:ggml-sycl.cpp:ggml_sycl_moe_down_weighted_sum_sycl":
        (1, kernel("the MoE down projection's weighted sum over the selected experts")),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_batched_sycl":
        (2, kernel("fills the batched GEMM's pointer tables, one launch per src1 layout branch")),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_id": (4, MUL_MAT_ID_COPIES),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_id/copy_src1_current_f32":
        (1, kernel("gathers the current expert's src1 rows into a contiguous f32 buffer")),
    "submit:ggml-sycl.cpp:ggml_sycl_mul_mat_tp_row_parallel_post":
        (1, kernel(TP + "adds device 1's partial result into dst")),
    "submit:ggml-sycl.cpp:ggml_sycl_router_f32_bias_argsort_sycl": (1, kernel("the MoE router's biased top-k argsort")),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_moe_gather_activation_rows":
        (1, kernel("gathers the activation rows of each selected expert")),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_moe_scatter_rows": (1, kernel("scatters expert output rows back to tokens")),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_mul_mat_unpin_event":
        (1, MARKER + "a MUL_MAT's unpin point (on an out-of-order queue a barrier is used instead)"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_q8_1_soa_repeat": (1, kernel("repeats Q8_1 SOA activation rows per expert")),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_q8_1_soa_selected":
        (1, kernel("selects Q8_1 SOA activation rows per expert")),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_queue_sync_event":
        (2, MARKER + "a queue sync point, one launch for an in-order queue and one after a barrier otherwise"),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_secondary_moe_glu": (1, kernel("the secondary GPU's MoE GLU")),
    "submit:ggml-sycl.cpp:ggml_sycl_submit_secondary_moe_weighted_reduce":
        (1, kernel("the secondary GPU's MoE weighted reduce")),
    "submit:ggml-sycl.cpp:ggml_sycl_update_moe_ptr_table":
        (1, MARKER + "joins the MoE pointer table's dependencies into one event"),
    "submit:ggml-sycl.cpp:ggml_sycl_upload_moe_transient_ptr_table":
        (1, MARKER + "joins the transient MoE pointer table's dependencies into one event"),
    "submit:ggml-sycl.cpp:moe_cpu_job_join_on_queue":
        (1, host_task("joins a host expert job back onto the device queue")),
    "submit:ggml-sycl.cpp:reorder_mxfp4_gpu": (1, WEIGHT_LAYOUT),
    "submit:ggml-sycl.cpp:reorder_q4_0_gpu": (1, WEIGHT_LAYOUT),
    "submit:ggml-sycl.cpp:reorder_q4_k_gpu": (1, WEIGHT_LAYOUT),
    "submit:ggml-sycl.cpp:reorder_q6_k_gpu": (1, WEIGHT_LAYOUT),
    "submit:ggml-sycl.cpp:reorder_q8_0_gpu": (1, WEIGHT_LAYOUT),
    "submit:ggml-sycl.cpp:sycl_reorder_parallel_for": (1, WEIGHT_LAYOUT),
    "submit:ggml-sycl.cpp:test_moe_ptr_table_dispatch_bundle_retains_table_compact_missing": (1, TEST_ENTRY),
    "submit:ggml-sycl.cpp:try_xmx_sorted_moe": (3, MOE_XMX_SORT),
    "submit:mem-ops.cpp:mem_copy_ptr_async":
        (1, "the window ring's copy helper (mem_copy_async), with dependencies: cgh.memcpy; " + WRAP_SUBMIT),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q2_K_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q3_K_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_colfused_4tile_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_colfused_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_largetile_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_multitile_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_0_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_1_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q4_K_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q5_0_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q5_1_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q5_K_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q6_K_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q8_0_q8_1_xmx_largetile_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:ggml_mul_mat_q8_0_q8_1_xmx_sycl": (1, kernel("an MMQ XMX GEMM")),
    "submit:mmq_xmx.cpp:xmx_test_kernel": (1, TEST_ENTRY),
    "submit:mmvq.cpp:coalesced_mul_mat_vec_mxfp4_q8_1_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:convert_mxfp4_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:convert_q4_0_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:convert_q6_k_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:convert_q8_0_to_coalesced_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:ggml_sycl_build_moe_compact_list": (2, kernel("builds the MoE compact expert list, two launches")),
    "submit:mmvq.cpp:mmvq_submit_aos_impl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mmvq_submit_mxfp4_soa": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mmvq_submit_q4_0_soa": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mmvq_submit_q6_k_soa": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mmvq_submit_quantize_q8_1_aos_after": (1, kernel("quantizes src1 to Q8_1 (AOS) for MMVQ")),
    "submit:mmvq.cpp:mmvq_submit_quantize_q8_1_soa": (1, kernel("quantizes src1 to Q8_1 (SOA) for MMVQ")),
    "submit:mmvq.cpp:mul_mat_vec_iq1_m_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq1_s_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq2_s_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq2_xs_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq2_xxs_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq3_s_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq3_xxs_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq4_nl_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_iq4_xs_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_mxfp4_q8_1_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mul_mat_vec_q2_0_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q2_K_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q3_K_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q4_1_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q4_K_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q5_0_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q5_1_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q5_K_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mul_mat_vec_q6_K_q8_1_sycl": (1, kernel(VECTOR)),
    "submit:mmvq.cpp:mxfp4_build_grouped_metadata_from_ids_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_down_sum_direct_final_zero_dst_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_atomic_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_scratch_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_down_sum_i8_direct_final_rank_parallel_scratch_reduce_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_dpas_down_single_col_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_dpas_i8_rowmajor_single_col_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_i8_grouped_direct_q8_m2_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_i8_grouped_direct_q8_m4_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_i8_grouped_direct_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_i8_grouped_packed_q8_m4_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_moe_gateup_prepack_selected_rows_submit": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_gateup_prepack_dpas_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_multirhs_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_singlecol_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_soa_dpas_m4_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_split_q8_submit/submit_glu_q8": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_split_submit/submit_glu": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_direct_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_m2_direct_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_dpas_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_m2_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_m4_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_direct_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_grouped_packed_q8_m2_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_xmx_tiled_grouped_direct_q8_m2_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_xmx_tiled_grouped_direct_q8_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:mxfp4_xmx_tiled_grouped_weighted_q8_m2_sycl": (1, kernel(MXFP4)),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_i8_q8_1_id_sycl_rows/submit_mmv": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_xmx_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:reorder_mul_mat_vec_mxfp4_q8_1_xmx_tiled_id_sycl": (1, WEIGHT_LAYOUT),
    "submit:mmvq.cpp:variable_tile_mul_mat_vec_q6_k_q8_1_sycl": (1, kernel(VECTOR)),
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
    "submit:mul-mat-bf16.hpp:launch_skinny": (1, bf16("launch_skinny, for at most SKINNY_MAX_COLS columns")),
    "submit:mul-mat-bf16.hpp:launch_tiled": (1, bf16("launch_tiled, for more columns")),
    "submit:norm.cpp:ggml_sycl_op_rms_norm_back":
        (1, "RMS_NORM_BACK, which shares norm.cpp with RMS_NORM: " + NOT_BUILT + "; a wrap task may leave it"),
    "submit:norm.cpp:l2_norm_f32_sycl": (2, kernel("L2_NORM, one launch per work-group size branch")),
    "submit:quantize.hpp:quantize_row_q8_0_sycl": (1, kernel("quantizes src1 to Q8_0 for MMQ")),
    "submit:quantize.hpp:quantize_row_q8_1_sycl": (1, kernel("quantizes src1 to Q8_1 for MMQ")),
    "submit:quantized-comm.hpp:quantize_and_copy_to_host":
        (2, kernel(TP + "the quantized all-reduce's min/max and INT16 quantize, two launches")),
    "submit:solve_tri.cpp:ggml_sycl_op_solve_tri": (1, kernel("SOLVE_TRI")),
    "submit:ssm_conv.cpp:kernel_ssm_conv": (1, kernel("SSM_CONV")),
    "submit:unified-cache.cpp:cpu_job_input_marker_submit":
        (1, kernel("the input marker of a host expert job (a single_task that publishes a serial)")),
    "submit:unified-cache.cpp:unified_cache::onednn_graph_scratch_free":
        (1, MARKER + "the completion marker of a freed oneDNN Graph scratch slot, on the watch queue"),
    "submit:unified-kernel.cpp:UnifiedKernel::benchmark_graph_overhead": (7, TEST_ENTRY),
    "submit:unified-kernel.cpp:UnifiedKernel::execute_persistent_phased": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:UnifiedKernel::launch_persistent_kernel/run_persistent_kernel":
        (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:UnifiedKernel::launch_persistent_kernel_async": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:UnifiedKernel::record_micro_graph": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:UnifiedKernel::rms_norm": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:UnifiedKernel::silu_mul": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:dispatch_xmx_kernel": (3, XMX_DETAIL_PROBES),
    "submit:unified-kernel.cpp:launch_unified_matmul": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:micro_submit_add": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:micro_submit_attention": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:micro_submit_rms_norm": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:micro_submit_silu_mul": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:micro_submit_strided_copy": (1, kernel("a unified-kernel launch")),
    "submit:unified-kernel.cpp:mmvq_submit_silu_mul": (1, kernel("a unified-kernel launch")),
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

# "kind:file:function:receiver" -> (count, reason): the calls set aside by their receiver. receiver is the receiver
# expression exactly as written at the call, a plain chain of names (Code.receiver), prefix included and whitespace
# removed (`engine->queue`, `pool`, `cgh`). A call is set aside only when its exact key has a row and the one definition
# of the function holds exactly count calls of that kind on that expression; every other call counts, and no type is
# read anywhere. Adding a row is the reviewed act: read the receiver's declaration by hand, and its call, first. If the
# receiver is a queue, there is no row to add. The pin proves only that the set of set-aside expressions and their
# counts did not change; the reason is the reviewer's claim, so it says what the receiver is and where it is declared.
# queue_tripwire refuses a row whose name the function plainly binds next to `queue`, as a tripwire, not a type check.
SET_ASIDE_BY_RECEIVER = {
    # the CPU executor's command groups, handed to cpu_submit_async, whose submit is the site; cgh is the lambda's
    # `sycl::handler & cgh` parameter at each line
    "shortcut:cpu-dispatch.cpp:cpu_binary_op:cgh": (1, "the command group's handler (cpu-dispatch.cpp:5116)"),
    "shortcut:cpu-dispatch.cpp:cpu_glu:cgh": (1, "the command group's handler (cpu-dispatch.cpp:5364)"),
    "shortcut:cpu-dispatch.cpp:cpu_norm:cgh": (1, "the command group's handler (cpu-dispatch.cpp:5835)"),
    "shortcut:cpu-dispatch.cpp:cpu_rms_norm:cgh": (1, "the command group's handler (cpu-dispatch.cpp:4978)"),
    "shortcut:cpu-dispatch.cpp:cpu_rope:cgh": (1, "the command group's handler (cpu-dispatch.cpp:6183)"),
    "shortcut:cpu-dispatch.cpp:cpu_scale:cgh": (1, "the command group's handler (cpu-dispatch.cpp:5929)"),
    "shortcut:cpu-dispatch.cpp:cpu_silu:cgh": (1, "the command group's handler (cpu-dispatch.cpp:5224)"),
    "shortcut:cpu-dispatch.cpp:cpu_soft_max:cgh": (1, "the command group's handler (cpu-dispatch.cpp:5709)"),
    "shortcut:ggml-sycl.cpp:ggml_sycl_plan_dry_run_plans:ready": (
        1, "a std::array<bool, GGML_SYCL_MAX_DEVICES> (ggml-sycl.cpp:19187): a host fill"),
    "shortcut:ggml-sycl.cpp:ggml_sycl_reset_model_load_scratch_state:g_moe_expert_vram_reserve": (
        1, "a file-scope std::array<size_t, GGML_SYCL_MAX_DEVICES> (ggml-sycl.cpp:4208): a host fill"),
    "shortcut:ggml-sycl.cpp:populate_inventory_globals:g_moe_expert_vram_reserve": (
        1, "a file-scope std::array<size_t, GGML_SYCL_MAX_DEVICES> (ggml-sycl.cpp:4208): a host fill"),
    "shortcut:mmq.cpp:ggml_sycl_parallel_for_grf:cgh": (
        3, "the `sycl::handler & cgh` parameter (mmq.cpp:321), one launch per GRF mode; the callers submit"),
    "shortcut:unified-cache.cpp:arena_pp_profile_snapshot_zones:cap_out": (
        1, "a std::array & parameter (unified-cache.cpp:5443): a host fill"),
    "shortcut:unified-cache.cpp:arena_pp_profile_snapshot_zones:used_out": (
        1, "a std::array & parameter (unified-cache.cpp:5442): a host fill"),
    "shortcut:unified-cache.cpp:unified_cache::configure_host_zones:host_zone_reserved_bytes_": (
        2, "a std::array member of unified_cache (unified-cache.hpp:6063): a host fill"),
    "submit:ggml-sycl.cpp:ggml_sycl_dispatch_host_flash_attn_async:pool": (
        1, "`auto & pool = g_attn_host_pools[device]` (ggml-sycl.cpp:97223) over a plain array of "
           "ggml_sycl::AttnHostPool (ggml-sycl.cpp:2729): a CPU job; its D2H feeds are mem_copy_ptr_async, counted in "
           "mem-ops.cpp"),
    "submit:ggml-sycl.cpp:ggml_sycl_dispatch_host_flash_attn_sync:pool": (
        1, "`auto & pool = g_attn_host_pools[device]` (ggml-sycl.cpp:97441), as in the async dispatch"),
    "submit:lazy-io-pool.cpp:lazy_io_submit:engine->queue": (
        1, "the lazy_io_queue member of lazy_io_engine (lazy-io-pool.cpp:53), engine declared at "
           "lazy-io-pool.cpp:1869: queues an O_DIRECT file read, no device command"),
    "submit:moe-fused-transaction.cpp:Transaction::submit:submitter": (
        1, "the `Submitter & submitter` parameter (moe-fused-transaction.cpp:213): PromptSubmitter::submit calls the "
           "executor's submit_writes, whose launches are counted in ggml-sycl.cpp"),
    "submit:moe-fused-transaction.cpp:submit_prompt_fusion:transaction": (
        1, "a local Transaction (moe-fused-transaction.cpp:391): its submit checks its state and calls "
           "Submitter::submit, set aside above"),
}

# Synthetic source the record-rule tests are built from: a function head, two launches, the two record calls.
FN_HEAD = "void f(sycl::queue * q, bool c) {\n"
LAUNCH_K1 = "q->submit([&](sycl::handler & h) { h.single_task(k1); })"
LAUNCH_K2 = "q->submit([&](sycl::handler & h) { h.single_task(k2); })"
RECORD_EVENT = "ggml_sycl_kernel_profile_record_event"
RECORD_RETURNED = "ggml_sycl_profile_record_returned_event"

BASE_WRAPPER = "ggml_sycl_profile_submit"
# record a launch's returned event with the profiler; a launch whose event reaches one is profiled (Code.recorded)
RECORDERS = ("ggml_sycl_kernel_profile_record_event", "ggml_sycl_profile_record_returned_event")
# `ggml_sycl_kernel_profile_record_event(`
RECORD_CALL = re.compile(r"\b(?:" + "|".join(RECORDERS) + r")\s*\(")

# `q->submit(`, `stream()\n    .submit(`
SUBMIT = re.compile(r"(?:->|\.)\s*submit\s*\(")
# a queue member that enqueues a command without submit(); only a SET_ASIDE_BY_RECEIVER row sets one aside:
# `stream->parallel_for(`, `q.memcpy(`, `q.parallel_for<class k>(`
SHORTCUT = re.compile(
    r"(?:->|\.)\s*(parallel_for|single_task|memcpy|memset|fill|copy|prefetch|mem_advise)\s*(?:<[^;{}()]*>\s*)?\("
)
# Lookback windows, in characters of blanked source. Each bounds a backward search so a scan stays linear in the
# file; beside each, what a miss does and which way it fails.
# The text before a '{' or a statement: a signature, a class head, a loop / if / else head. A signature longer than
# this reads as a plain block, so its sites are named `?` and the no-`?` test fails; a loop or else head beyond it
# is not seen, which can only let the record rule match more, and a match counts only through a reviewed row.
HEAD_WINDOW = 2000
# A callee name before '(', a receiver expression before '.' / '->', the declaration before `ev =`. A callee missed
# is no wrapper, so its launch is reported; a receiver cut short names no row, which keeps its call counted.
NAME_WINDOW = 300
# A function's name before its parameter list. A miss reads the body as a plain block, named `?`: the test fails.
PARAMS_NAME_WINDOW = 400
# `auto name =` before a lambda's capture list, and `else` before `if (`. A lambda name missed drops `/name` from
# the key, which then differs from ALLOW; an `else if` missed can only let the record rule match more.
LAMBDA_NAME_WINDOW = 200
# The `.` / `->` / `::` just before `ev` in `s.ev = launch`: a miss reads a member as a local, which can only let the
# record rule match more.
MEMBER_WINDOW = 50
NOT_A_TYPE = {"return", "else", "case", "new", "delete", "throw", "goto", "sizeof", "co_return", "using", "typedef",
              "operator", "const", "static", "inline", "constexpr", "auto", "do"}

# A comment, a raw string, a string or a character literal (`// x`, `/* x */`, `R"x(...)x"`, `"..."`, `'('`). Each
# is blanked to spaces with its newlines kept, so
# offsets and line numbers survive and nothing inside one is read as code, a parenthesis included.
NOT_CODE = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r"|\bR\"(?P<delim>[^()\\\s]{0,16})\(.*?\)(?P=delim)\""
    r"|\"(?:\\.|[^\"\\\n])*\""
    r"|'(?:\\.|[^'\\\n])+'",
    re.S,
)
# the callee before a '(': `f` in `f(`, `parallel_for` in `parallel_for<class k>(`
NAME_BEFORE_CALL = re.compile(r"(\w+)\s*(?:<[^<>;{}()]*>\s*)?$")
# a function's name before its parameter list: `templated`, `holder::outside`, `foo::~foo`
NAME_BEFORE_PARAMS = re.compile(r"((?:~?\w+\s*::\s*)*~?\w+)\s*(?:<[^<>;{}()]*>\s*)?$")
# an operator's name before its parameter list: `operator()`, `holder::operator+=`
OPERATOR_BEFORE_PARAMS = re.compile(r"((?:\w+\s*::\s*)*operator\s*(?:\(\s*\)|[^\w\s(]+))\s*$")
# what may sit between a parameter list's ')' and its body's '{': `) const noexcept`, `) -> sycl::event`, `) try`
AFTER_PARAMS = re.compile(
    r"\)\s*(?:(?:const|volatile|noexcept|override|final|mutable|try|&&|&)\s*|noexcept\s*\([^()]*\)\s*"
    r"|->\s*[\w:<>,\s\*&]+?\s*)*$"
)
# a name before a '(' that does not make the '{' after it a function body: `if constexpr (X) {` is a block. A
# brace-init constructor `T x{...}` names no function either; a site under one is named `?`, which
# test_every_file_of_the_path_has_balanced_brackets_and_a_name_for_every_site fails on
NOT_A_FUNCTION = {"if", "for", "while", "switch", "catch", "return", "sizeof", "decltype", "alignof", "noexcept",
                  "static_assert", "defined", "__attribute__", "constexpr", "consteval"}
# a class head before its '{': `struct holder : base<int>`, `class alignas(64) slot`
CLASS_HEAD = re.compile(r"\b(?:struct|class|union)\s+(?:alignas\s*\([^()]*\)\s*)?(\w+)[^;{}()]*$")
# the name a lambda is assigned to, before its capture list: `named` in `auto named = [&]`
NAMED_LAMBDA = re.compile(r"\b(\w+)\s*=\s*$")
# a conditional directive: `#if`, `#  ifdef X`, `#elif`, `#else`, `#endif`
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


# This pin keeps its own blanker rather than sycl_gate.views(): views() has no raw string literal (`R"x(...)x"`), whose
# quotes and parentheses would otherwise be read as code, and the census needs every offset and newline kept.
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
        # the '{' each '{' opened inside, so the innermost block around a position is a climb, not a scan
        self.brace_parent: dict[int, int | None] = {}
        self.unbalanced: list[str] = []
        self._newlines = [m.start() for m in re.finditer("\n", self.code)]
        self._brace_kinds: dict[int, tuple[str, str | None]] = {}
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
                self.brace_parent[pos] = braces[-1] if braces else None
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
        return bisect.bisect_left(self._newlines, pos) + 1

    def callee(self, open_paren: int) -> str | None:
        m = NAME_BEFORE_CALL.search(self.code, max(0, open_paren - NAME_WINDOW), open_paren)
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
        m = NAMED_LAMBDA.search(self.code, max(0, i - LAMBDA_NAME_WINDOW), max(0, i))
        return m.group(1) if m else None

    def brace_kind(self, pos: int) -> tuple[str, str | None]:
        """What the '{' at pos opens: function, lambda or class, with its name, or a plain block. Kept per brace: the
        census asks it again for every site under the brace."""
        if pos not in self._brace_kinds:
            self._brace_kinds[pos] = self.brace_kind_uncached(pos)
        return self._brace_kinds[pos]

    def brace_kind_uncached(self, pos: int) -> tuple[str, str | None]:
        lo = max(0, pos - HEAD_WINDOW)
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
            before_lo = max(0, open_ - PARAMS_NAME_WINDOW)
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
        b = self.innermost_brace(pos)
        while b is not None:
            if self.brace_kind(b)[0] == "function":
                return b
            b = self.brace_parent.get(b)
        return None

    def innermost_brace(self, pos: int) -> int | None:
        """The '{' of the innermost block of any kind whose body holds pos. The last '{' before pos either holds it
        or closed before it, and then the block that holds pos (if any) is among its ancestors."""
        if self._brace_starts is None:
            self._brace_starts = sorted(self.brace_parent)
        i = bisect.bisect_left(self._brace_starts, pos) - 1
        b = self._brace_starts[i] if i >= 0 else None
        while b is not None and self.brace_close.get(b, len(self.code)) < pos:
            b = self.brace_parent.get(b)
        return b

    def braces_at(self, pos: int) -> set[int]:
        """The '{' of every block whose body holds pos."""
        found = set()
        b = self.innermost_brace(pos)
        while b is not None:
            found.add(b)
            b = self.innermost_brace(b)
        return found

    def is_loop(self, brace: int) -> bool:
        head = self.code[max(0, brace - HEAD_WINDOW):brace].rstrip()
        if re.search(r"\bdo$", head):
            return True
        open_ = self.paren_open.get(max(0, brace - HEAD_WINDOW) + len(head) - 1) if head.endswith(")") else None
        return open_ is not None and self.callee(open_) in ("for", "while")

    def is_else_arm(self, brace: int) -> bool:
        """The '{' opens an `else` or `else if (...)` arm."""
        lo = max(0, brace - HEAD_WINDOW)
        head = self.code[lo:brace].rstrip()
        if re.search(r"\belse$", head):
            return True
        if not head.endswith(")"):
            return False
        open_ = self.paren_open.get(lo + len(head) - 1)
        return (open_ is not None and self.callee(open_) == "if"
                and re.search(r"\belse\s*if\s*$", self.code[max(0, open_ - LAMBDA_NAME_WINDOW):open_]) is not None)

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
            head = self.code[max(0, i - NAME_WINDOW):i]
            stripped = head.rstrip()
            j = max(0, i - NAME_WINDOW) + len(stripped)
            if stripped.endswith(")") and (j - 1) in self.paren_open:
                i = self.paren_open[j - 1]
                continue
            m = re.search(r"(?:->|\.|::)?\s*\w+$", stripped)
            if not m:
                break
            i = max(0, i - NAME_WINDOW) + m.start()
            if not re.match(r"\s*(?:->|\.|::)", self.code[i:]):
                break
        lo = max(0, i - NAME_WINDOW)
        m = re.search(r"\b(\w+)\s*=\s*$", self.code[lo:i])
        if not m or re.search(r"(?:\.|->|::)\s*$", self.code[max(0, lo + m.start(1) - MEMBER_WINDOW):lo + m.start(1)]):
            return None
        # the statement starts at the type of a declaration (`sycl::event ` in `sycl::event ev =`), at the name of an
        # assignment
        decl = re.search(r"([A-Za-z_][\w:]*(?:\s*<[^;{}()]*>)?[\s&*]+)$",
                         self.code[max(0, lo + m.start(1) - NAME_WINDOW):lo + m.start(1)])
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
            lo = max(0, end - HEAD_WINDOW)  # a window, not the whole file: this runs for every assigned launch
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
            keyword_lo = max(0, open_ - NAME_WINDOW)
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
        # `ev = x` (not `ev == x`, not `s.ev = x`), then a declaration `sycl::event ev;` / `auto & ev = x` / `T ev(x)`
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
        """The receiver expression of the member call at pos (the '.' or '->') when it is a plain chain of names
        joined by `.`, `->` or `::` (`q`, `engine->queue`, `this->pool_`, `ns::g_pool`), written with its whitespace
        removed; else None, which no row names, so the call counts. A call, an index, a cast, a template argument,
        a parenthesised or dereferenced operand (`foo.stream()`, `g_pools[d]`, `static_cast<T &>(g)`, `(*q)`), a
        `*` or `&` right before the chain, a backslash line continuation, and a keyword joined by `::` all give
        None."""
        code = self.code
        lo = max(0, pos - NAME_WINDOW)
        i = pos
        operand = True
        while True:
            while i > lo and code[i - 1] in " \t\n":
                i -= 1
            if i <= lo:
                return None
            if operand:
                m = re.search(r"\w+$", code[lo:i])
                if m is None:
                    return None
                i = lo + m.start()
                operand = False
            elif code[i - 2:i] in ("->", "::") and code[i - 3:i] != ":::":
                i -= 2
                operand = True
            elif code[i - 1] == ".":
                i -= 1
                operand = True
            else:
                break
        if code[i - 1] in "\\*&":
            return None
        text = re.sub(r"\s+", "", code[i:pos])
        if text[:1].isdigit() or set(re.findall(r"\w+", text)) & CHAIN_KEYWORDS:
            return None
        return text

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


_WRAPPERS: dict[tuple, set[str]] = {}


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


def census(texts: dict[str, str], wrappers: set[str] | None = None, skipped: list[str] | None = None,
           recorded: list[str] | None = None, table: dict[str, int] | None = None,
           rule_matches: dict[str, list[int]] | None = None,
           set_aside_table: dict[str, tuple[int, str]] | None = None,
           receiver_matches: dict[str, list[str]] | None = None,
           tripped: dict[str, str] | None = None) -> dict[str, list[tuple[int, str, str]]]:
    """The raw sites of each named text, as (line, function, kind) triples. A site is wrapped when any '(' still open
    at it, however far back and however deeply nested, is the argument list of a wrapper call. A launch the record
    rule matches (Code.recorded) is excused only when its "file:function" is a row of table (RECORDED_ON_THE_PATH by
    default) and the rule matches exactly that many launches in the function; each one excused is appended to
    recorded, and the line of every launch the rule matches goes to rule_matches under its "file:function".
    A call is set aside by its receiver only when its "kind:file:function:receiver" is a row of set_aside_table
    (SET_ASIDE_BY_RECEIVER), the function holds exactly the row's count of such calls, and queue_tripwire does not
    refuse the row; the file:line of each call with a row goes to receiver_matches under its key, and each refusal to
    tripped. A shortcut inside a submit's parentheses is that submit's command group's call, not a site, unless the
    submit is set aside (a host pool's job may launch on a queue). Each call set aside is appended to skipped with
    its reason."""
    if wrappers is None:
        wrappers = derive_wrappers(texts)
    if table is None:
        table = RECORDED_ON_THE_PATH
    if set_aside_table is None:
        set_aside_table = SET_ASIDE_BY_RECEIVER
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
                if rule_matches is not None:
                    rule_matches.setdefault(key, []).append(code.line(pos))
            unwrapped.append((pos, kind, method, parens, braces, key))
        launches = []
        for pos, kind, method, parens, braces, key in unwrapped:
            if key is not None and table.get(key) == matched[key]:
                if recorded is not None:
                    recorded.append(key)
                continue
            launches.append((pos, kind, method, parens, braces))
        set_aside: set[int] = set()
        # submits first: whether a submit is set aside decides whether the shortcuts inside it count
        for which in ("submit", "shortcut"):
            calls: dict[str, list[tuple[int, list[int], str]]] = {}
            for pos, kind, method, parens, braces in launches:
                if kind != which or (kind == "shortcut" and inside_counted_submit(code, parens, set_aside)):
                    continue
                receiver = code.receiver(pos)
                key = f"{kind}:{name}:{code.enclosing(braces)}:{receiver}"
                if receiver is not None and key in set_aside_table:
                    calls.setdefault(key, []).append((pos, braces, receiver))
            for key, held in calls.items():
                lines = [f"{name}:{code.line(pos)}" for pos, _, _ in held]
                if receiver_matches is not None:
                    receiver_matches.setdefault(key, []).extend(lines)
                # one row covers one definition: overloads, or functions of one name in two namespaces, share a key
                definitions = {code.function_brace(braces) for _, braces, _ in held}
                refusal = None
                if code.enclosing(held[0][1]) == "?":
                    refusal = (f"a row needs a named function, and the calls at {', '.join(lines)} have none, so they "
                               "count (To update, e)")
                elif len(definitions) > 1:
                    refusal = (f"the row names {len(definitions)} definitions of {code.enclosing(held[0][1])}, with "
                               f"calls at {', '.join(lines)}; a row covers one definition, so these calls count "
                               "(To update, e)")
                elif set_aside_table[key][0] != len(held):
                    continue
                else:
                    receiver = held[0][2]
                    bound = queue_tripwire(code, held[0][1], receiver)
                    if bound:
                        refusal = (f"{', '.join(f'{name}:{line}' for line in bound)} binds {last_name(receiver)} next "
                                   "to `queue`; read its declaration, and if the receiver is a queue, delete the row "
                                   "(To update, e)")
                if refusal is not None:
                    if tripped is not None:
                        tripped[key] = refusal
                    continue
                set_aside.update(pos for pos, _, _ in held)
        for pos, kind, method, parens, braces in launches:
            why = None
            if pos in set_aside:
                why = f"on {code.receiver(pos)}, a SET_ASIDE_BY_RECEIVER row"
            elif kind == "shortcut" and inside_counted_submit(code, parens, set_aside):
                why = "inside a submit"
            if why is not None:
                if skipped is not None:
                    skipped.append(f"{name}:{code.enclosing(braces)} .{method}( {why}")
                continue
            found.setdefault(name, []).append((code.line(pos), code.enclosing(braces), kind))
    return found


# keywords the receiver walk can join to a chain through `::` (`return ::q.fill(p)` reads `return::q`): no expression
CHAIN_KEYWORDS = {"return", "co_await", "co_yield", "co_return", "throw", "else", "case", "new", "delete", "sizeof",
                  "typename", "do"}

# `queue` or `queue_ptr` as a whole token: `sycl::queue`, `dpct::queue_ptr`, but not `lazy_io_queue`
QUEUE_TOKEN = re.compile(r"\bqueue(?:_ptr)?\b")


def last_name(receiver: str) -> str:
    """The last name of a receiver chain: `queue` for `engine->queue`, `g_pool` for `ns::g_pool`."""
    return re.findall(r"\w+", receiver)[-1]


def queue_tripwire(code: Code, braces: list[int], receiver: str) -> list[int]:
    """The lines on which the one function around a row's calls plainly binds the receiver's last name next to the
    token `queue`, in one statement (text between `;`, `{` and `}`). A tripwire, not a resolver: it reads no type and
    follows no alias, so a row it does not refuse is still only as good as the reviewer who read the receiver's
    declaration. It looks from the function's parameter list to its end."""
    ident = re.escape(last_name(receiver))
    binds = [
        # a reference or a pointer: `sycl::queue & ready`, `sycl::queue * ready` (a range-for's variable too)
        re.compile(r"\bqueue(?:_ptr)?\b\s*(?:<[^;{}]*?>)?\s*(?:const\s*)?[&*]+\s*(?:const\s+)?" + ident + r"\b"),
        # a value: `sycl::queue ready{dev}`, `dpct::queue_ptr ready`
        re.compile(r"\bqueue(?:_ptr)?\s+(?:const\s+)?" + ident + r"\b"),
        # an assignment or an initialization from an expression naming a queue: `auto & ready = ctx->queue()`
        re.compile(r"\b" + ident + r"\s*=(?!=)"),
        # a structured binding or a capture: `auto [ready, n] = ...queue...`, `[ready = ctx.queue]`
        re.compile(r"\[\s*&?\s*" + ident + r"\b"),
    ]
    fb = code.function_brace(braces)
    lo = code.params[fb][0] if fb in code.params else fb
    hi = code.brace_close.get(fb, len(code.code)) if fb else len(code.code)
    lines = []
    for statement in re.finditer(r"[^;{}]+", code.code[lo:hi]):
        text = statement.group(0)
        if not QUEUE_TOKEN.search(text):
            continue
        for bind in binds:
            m = bind.search(text)
            if m:
                lines.append(code.line(lo + statement.start() + m.start()))
                break
    return lines


def inside_counted_submit(code: Code, parens: list[int], set_aside: set[int]) -> bool:
    """Whether the innermost enclosing submit of a call counts: a call there is that submit's command group's
    (`h.memcpy(` in `q.submit([&](sycl::handler & h) { ... })`). set_aside holds the position of each submit set aside
    by its receiver, whose job is no command group; what encloses that submit does not matter."""
    for p in reversed(parens):
        if code.callee(p) == "submit":
            dot = re.search(r"(?:->|\.)\s*submit\s*$", code.code[max(0, p - NAME_WINDOW):p])
            # a free `submit(` is no submit site and no command group: look further out
            if dot is not None:
                return max(0, p - NAME_WINDOW) + dot.start() not in set_aside
    return False


def census_keys(found: dict[str, list[tuple[int, str, str]]]) -> dict[str, int]:
    keys: dict[str, int] = {}
    for name, sites in found.items():
        for _, func, kind in sites:
            key = f"{kind}:{name}:{func}"
            keys[key] = keys.get(key, 0) + 1
    return keys


def sites_by_key(found: dict[str, list[tuple[int, str, str]]]) -> dict[str, list[str]]:
    """"kind:file:function" -> the file:line of each of its raw launches."""
    by_key: dict[str, list[str]] = {}
    for name, sites in found.items():
        for line, func, kind in sites:
            by_key.setdefault(f"{kind}:{name}:{func}", []).append(f"{name}:{line}")
    return by_key


def diff_against_allow(found: dict[str, list[tuple[int, str, str]]], allow: dict[str, tuple[int, str]],
                       rule_matches: dict[str, list[int]] | None = None,
                       record_table: dict[str, int] | None = None) -> list[str]:
    """One line per key whose tree count differs from its ALLOW count: the key, the launches with their file:line,
    the listed count, and what to do (the "To update" step of the module docstring). A function the record rule
    matches launches in, with no RECORDED_ON_THE_PATH row, is pointed at that table too."""
    if record_table is None:
        record_table = RECORDED_ON_THE_PATH
    by_key = sites_by_key(found)
    problems = []
    for key in sorted(set(by_key) | set(allow)):
        where = by_key.get(key, [])
        want = allow[key][0] if key in allow else 0
        if len(where) > want:
            function = key.split(":", 1)[1]
            record = ""
            if rule_matches and function in rule_matches and function not in record_table:
                record = "; or, if a record call receives its event, add a RECORDED_ON_THE_PATH row (To update, d)"
            problems.append(f"{key}: {len(where)} raw launch(es) in the tree ({', '.join(where)}), {want} listed: "
                            f"wrap them, or list them in ALLOW with why they are not wrapped (To update, b){record}")
        elif len(where) < want:
            at = f" ({', '.join(where)})" if where else ""
            problems.append(f'{key}: {len(where)} raw launch(es) in the tree{at}, {want} listed ("{allow[key][1]}"): '
                            "lower the count or delete the entry (To update, a)")
    return problems


def record_table_problems(rule_matches: dict[str, list[int]], table: dict[str, int]) -> list[str]:
    """One line per RECORDED_ON_THE_PATH row whose count is not what the record rule matches in that function."""
    problems = []
    for key in sorted(table):
        lines = rule_matches.get(key, [])
        if len(lines) != table[key]:
            name = key.split(":", 1)[0]
            at = f" ({', '.join(f'{name}:{line}' for line in lines)})" if lines else ""
            problems.append(f"{key}: RECORDED_ON_THE_PATH says {table[key]}, the record rule matches {len(lines)}{at}: "
                            "read the function and set the row to the launches whose event it records (To update, d)")
    return problems


def set_aside_problems(receiver_matches: dict[str, list[str]], table: dict[str, tuple[int, str]],
                       tripped: dict[str, str] | None = None) -> list[str]:
    """One line per SET_ASIDE_BY_RECEIVER row that does not set its calls aside: it was refused (tripped holds why),
    its count is not the number of calls on that receiver in the function, or no such call is left; each says what to
    do."""
    tripped = tripped or {}
    problems = []
    for key in sorted(set(receiver_matches) | set(table)):
        calls = receiver_matches.get(key, [])
        row = table[key][0] if key in table else 0
        if key in tripped:
            problems.append(f"{key}: refused: {tripped[key]}")
        elif calls and len(calls) != row:
            problems.append(f"{key}: {len(calls)} call(s) on this receiver in the function ({', '.join(calls)}), "
                            f"SET_ASIDE_BY_RECEIVER says {row}: read the declaration of each receiver; if it is a "
                            f"queue, delete the row; otherwise set the count to {len(calls)} (To update, e)")
        elif not calls:
            problems.append(f"{key}: SET_ASIDE_BY_RECEIVER says {row}, no call in the function has this receiver: "
                            "delete the row (To update, e)")
    return problems


def receiver_sections(skipped: list[str], problems: list[str]) -> str:
    """The two receiver sections of `--print`: each call set aside by a row, and each row that does not match."""
    out = ["# set aside by a SET_ASIDE_BY_RECEIVER row, in the files on the path"]
    out += sorted(entry for entry in skipped if "SET_ASIDE_BY_RECEIVER" in entry) or ["(none)"]
    out.append("# SET_ASIDE_BY_RECEIVER rows that do not match the files on the path (excluded files included)")
    out += problems or ["(none)"]
    return "\n".join(out)


def paste_line(key: str, count: int, reason: str) -> str:
    """An ALLOW line, written as the literal (json.dumps gives a double-quoted string Python reads back)."""
    return f"    {json.dumps(key)}: ({count}, {json.dumps(reason)}),"


def report(found: dict[str, list[tuple[int, str, str]]], rule_matches: dict[str, list[int]],
           allow: dict[str, tuple[int, str]], table: dict[str, int]) -> str:
    """What `--print` writes: every key with its count and launches, the ALLOW changes as lines to paste (or to
    delete), and what the record rule matched per function against RECORDED_ON_THE_PATH."""
    by_key = sites_by_key(found)
    out = ["# raw launches: key, count, file:line of each"]
    out += [f"{key} {len(where)} ({', '.join(where)})" for key, where in sorted(by_key.items())]
    out.append("# ALLOW changes, to paste (a new key's reason is a placeholder to replace)")
    for key in sorted(set(by_key) | set(allow)):
        have = len(by_key.get(key, []))
        want, reason = allow.get(key, (0, "TODO: what it is, and why it is not wrapped"))
        if have == want:
            continue
        if have:
            out.append(paste_line(key, have, reason))
        else:
            out.append("#" + paste_line(key, want, reason)[1:] + "  <- delete: 0 in the tree")
    out.append("# the record rule's matches per function (only a RECORDED_ON_THE_PATH row excuses them)")
    for key in sorted(set(rule_matches) | set(table)):
        lines = rule_matches.get(key, [])
        name = key.split(":", 1)[0]
        at = f" ({', '.join(f'{name}:{line}' for line in lines)})" if lines else ""
        row = table.get(key, "-")
        out.append(f"{key}: the record rule matches {len(lines)}{at}, RECORDED_ON_THE_PATH says {row}")
    return "\n".join(out)


def print_report() -> int:
    texts = read_tree()
    skipped: list[str] = []
    matches: dict[str, list[int]] = {}
    found = census({name: texts[name] for name in FILES}, derive_wrappers(texts), skipped,
                   rule_matches=matches)
    print(report(found, matches, ALLOW, RECORDED_ON_THE_PATH))
    print(receiver_sections(skipped, tree_set_aside_problems(texts)))
    print(count_by_kind_and_file(found))
    print(f"{sum(len(sites) for sites in found.values())} raw launch(es) in {len(census_keys(found))} key(s) over "
          f"{len(FILES)} file(s)")
    return 0


def read_tree() -> dict[str, str]:
    """FILES, the profiler header and the header beside each .cpp of FILES: every text a wrapper may be defined in."""
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


def receiver_rule_files() -> list[str]:
    """The excluded files whose reason is a SET_ASIDE_BY_RECEIVER row: their set-asides are rows of that table."""
    return [name for name, reason in EXCLUDED_FILES.items() if BY_RECEIVER_CHECKED in reason]


def tree_set_aside_problems(texts: dict[str, str]) -> list[str]:
    """set_aside_problems over FILES and the files excluded by a receiver row, whose rows live in the same table."""
    matches: dict[str, list[str]] = {}
    tripped: dict[str, str] = {}
    census({name: texts[name] for name in FILES}, derive_wrappers(texts), receiver_matches=matches,
           tripped=tripped)
    for name in receiver_rule_files():
        census({name: (SYCL / name).read_text(encoding="utf-8")}, derive_wrappers(texts), receiver_matches=matches,
               tripped=tripped)
    return set_aside_problems(matches, SET_ASIDE_BY_RECEIVER, tripped)


def census_tree(texts: dict[str, str], skipped: list[str] | None = None,
                rule_matches: dict[str, list[int]] | None = None) -> dict[str, list[tuple[int, str, str]]]:
    wrappers = derive_wrappers(texts)
    return census({name: texts[name] for name in FILES}, wrappers, skipped, rule_matches=rule_matches)


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


def raw_sites(src: str, skipped: list[str] | None = None, table: dict[str, int] | None = None,
              wrappers: set[str] | None = None) -> list[tuple[str, str]]:
    """(function, kind) of each raw launch of src, censused as a file named x.cpp."""
    found = census({"x.cpp": src}, wrappers, skipped=skipped, table=table)
    return [(func, kind) for _, func, kind in found.get("x.cpp", [])]


def raw_functions(src: str, wrappers: set[str] | None = None) -> list[str]:
    return [func for func, _ in raw_sites(src, wrappers=wrappers)]


def rejected_nearest_paren_design_says_wrapped(src: str) -> bool:
    """A rejected design, kept as a foil: wrapped when the nearest unclosed '(' within a fixed window before the
    submit is the profiler's. test_a_wrapped_submit_behind_a_long_lambda_is_still_wrapped shows a case it gets
    wrong and the census gets right."""
    window = 400
    code = blank(src)
    m = SUBMIT.search(code)
    head = code[max(0, m.start() - window):m.start()]
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
    assert not rejected_nearest_paren_design_says_wrapped(sample)
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
    known = {BASE_WRAPPER, "ggml_sycl_memcpy_profile_submit", "ggml_sycl_set_rows_profile_submit",
             "ggml_sycl_submit_binbcast_kernel", "mmq_submit"}
    derived = derive_wrappers(read_tree())
    assert derived == known, f"derived and not known: {sorted(derived - known)}; known and not derived: " \
                             f"{sorted(known - derived)}"


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


def test_a_launch_in_a_function_not_in_the_table_is_reported_whatever_the_shape():
    # (shape, launches in it): with no table row for x.cpp:f, every launch is reported, the plain shape included
    cases = {
        "plain": (FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_EVENT}(lab, ev);\n}}\n", 1),
        "direct": (FN_HEAD + f"{RECORD_RETURNED}(lab, {LAUNCH_K1});\n}}\n", 1),
        "opposite arm": (
            FN_HEAD + f"sycl::event ev;\n"
            f"if (c) {{ ev = {LAUNCH_K1}; }} else {{ ev = {LAUNCH_K2}; {RECORD_EVENT}(lab, ev); }}\n}}\n",
            2,
        ),
        "shadow": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"{{ sycl::event ev = other(); {RECORD_EVENT}(lab, ev); }}\n}}\n",
            1,
        ),
        "brace-less for": (
            FN_HEAD + f"sycl::event ev;\n"
            f"for (int i = 0; i < n; ++i) ev = {LAUNCH_K1};\n"
            f"{RECORD_EVENT}(lab, ev);\n}}\n",
            1,
        ),
        "brace-less do": (
            FN_HEAD + f"sycl::event ev;\n"
            f"do ev = {LAUNCH_K1}; while (c);\n"
            f"{RECORD_EVENT}(lab, ev);\n}}\n",
            1,
        ),
        "brace-less loop and if": (
            FN_HEAD + f"sycl::event ev;\n"
            f"while (c) if (x) ev = {LAUNCH_K1};\n"
            f"{RECORD_EVENT}(lab, ev);\n}}\n",
            1,
        ),
        "brace-less loop and else": (FN_HEAD + f"sycl::event ev;\nwhile (c) if (x) g(); else ev = {LAUNCH_K1};\n"
                                     f"{RECORD_EVENT}(lab, ev);\n}}\n", 1),
        "brace-less else record": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"if (c) g(); else {RECORD_EVENT}(lab, ev);\n}}\n",
            1,
        ),
        "sibling blocks": (
            FN_HEAD + f"sycl::event ev;\n"
            f"{{ ev = {LAUNCH_K1}; }} {{ ev = {LAUNCH_K2}; }}\n"
            f"{RECORD_EVENT}(lab, ev);\n}}\n",
            2,
        ),
        "two ifs": (FN_HEAD + f"sycl::event ev;\nif (a) {{ ev = {LAUNCH_K1}; }} if (b) {{ ev = {LAUNCH_K2}; }}\n"
                    f"{RECORD_EVENT}(lab, ev);\n}}\n", 2),
        "switch case": (FN_HEAD + f"sycl::event ev;\nswitch (k) {{ case 0: ev = {LAUNCH_K1}; break; "
                        f"case 1: {RECORD_EVENT}(lab, ev); }}\n}}\n", 1),
        "catch handler": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"try {{ g(); }} catch (...) {{ {RECORD_EVENT}(lab, ev); }}\n}}\n",
            1,
        ),
        "range-for shadow": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"for (auto & ev : evs) {{ {RECORD_EVENT}(lab, ev); }}\n}}\n",
            1,
        ),
        "structured binding": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"auto [ev, n] = pair();\n"
            f"{RECORD_EVENT}(lab, ev);\n}}\n",
            1,
        ),
        "record in a define": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"#define REC_IT {RECORD_EVENT}(lab, ev)\n}}\n",
            1,
        ),
        "lambda argument": (FN_HEAD + f"sycl::event ev;\n{RECORD_EVENT}(lab, ev, [&] {{ {LAUNCH_K1}; }});\n}}\n", 1),
        "member": (FN_HEAD + f"sycl::event ev;\ns.ev = {LAUNCH_K1};\n{RECORD_EVENT}(lab, ev);\n}}\n", 1),
        "uncalled lambda": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"auto g = [&] {{ {RECORD_EVENT}(lab, ev); }};\n}}\n",
            1,
        ),
        "ternary": (FN_HEAD + f"sycl::event ev = c ? {LAUNCH_K1} : {LAUNCH_K2};\n{RECORD_EVENT}(lab, ev);\n}}\n", 2),
    }
    for what, (sample, launches) in cases.items():
        assert [func for func, _ in raw_sites(sample)] == ["f"] * launches, what


def test_a_listed_function_whose_launches_the_rule_matches_is_excused():
    cases = {
        "plain": (FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_EVENT}(lab, ev, site, t0, t1);\n}}\n", 1),
        "direct": (FN_HEAD + f"return {RECORD_RETURNED}(lab, {LAUNCH_K1});\n}}\n", 1),
        "join": (
            FN_HEAD + f"sycl::event ev;\n"
            f"if (c) {{ ev = {LAUNCH_K1}; }} else {{ ev = {LAUNCH_K2}; }}\n"
            f"{RECORD_EVENT}(lab, ev);\n}}\n",
            2,
        ),
        "moved": (FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_RETURNED}(lab, std::move(ev));\n}}\n", 1),
        "record under profiling on": (
            FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n"
            f"if (on) {{ {RECORD_EVENT}(lab, ev); }}\n}}\n",
            1,
        ),
    }
    for what, (sample, count) in cases.items():
        assert raw_sites(sample, table={"x.cpp:f": count}) == [], what


def test_a_listed_function_whose_match_count_differs_from_its_row_is_reported():
    two = (FN_HEAD + f"sycl::event a = {LAUNCH_K1};\n"
           f"{RECORD_EVENT}(lab, a);\n"
           f"sycl::event b = {LAUNCH_K2};\n"
           f"{RECORD_EVENT}(lab, b);\n}}\n")
    # a new recorded launch in a listed function: N listed, N + 1 matched
    assert [func for func, _ in raw_sites(two, table={"x.cpp:f": 1})] == ["f", "f"]
    # a stale row: N listed, fewer matched
    one = FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_EVENT}(lab, ev);\n}}\n"
    assert [func for func, _ in raw_sites(one, table={"x.cpp:f": 2})] == ["f"]


def test_inside_a_listed_function_the_rule_still_refuses_what_it_can_see():
    table = {"x.cpp:f": 1}
    cases = {
        # one matched launch is listed; each refused shape leaves the row matching nothing, so the launch is reported
        "brace-less do": FN_HEAD + f"sycl::event ev;\ndo ev = {LAUNCH_K1}; while (c);\n{RECORD_EVENT}(lab, ev);\n}}\n",
        "brace-less loop and if": (FN_HEAD + f"sycl::event ev;\n"
                                   f"while (c) if (x) ev = {LAUNCH_K1};\n"
                                   f"{RECORD_EVENT}(lab, ev);\n}}\n"),
        "brace-less loop and else": (FN_HEAD + f"sycl::event ev;\n"
                                     f"while (c) if (x) g(); else ev = {LAUNCH_K1};\n"
                                     f"{RECORD_EVENT}(lab, ev);\n}}\n"),
        "label slot": FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_EVENT}(ev, other);\n}}\n",
        "record before": FN_HEAD + f"sycl::event ev;\n{RECORD_EVENT}(lab, ev);\nev = {LAUNCH_K1};\n}}\n",
        "overwritten": FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\nev = other();\n{RECORD_EVENT}(lab, ev);\n}}\n",
    }
    for what, sample in cases.items():
        assert [func for func, _ in raw_sites(sample, table=table)] == ["f"], what


def test_the_record_rule_excuses_exactly_the_tables_launches_of_the_tree():
    texts = read_tree()
    excused: list[str] = []
    matches: dict[str, list[int]] = {}
    census({name: texts[name] for name in FILES}, derive_wrappers(texts), recorded=excused,
           rule_matches=matches)
    wrong = record_table_problems(matches, RECORDED_ON_THE_PATH)
    assert not wrong, "\n".join(wrong)
    counted: dict[str, int] = {}
    for key in excused:
        counted[key] = counted.get(key, 0) + 1
    differ = sorted(key for key in set(counted) | set(RECORDED_ON_THE_PATH)
                    if counted.get(key, 0) != RECORDED_ON_THE_PATH.get(key, 0))
    assert not differ, f"excused launches differ from RECORDED_ON_THE_PATH at {differ}"


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
    assert receiver_census(sample, {"shortcut:x.cpp:launch:cgh": (1, "a handler")}, skipped=skipped) == [
        ("a", "submit")]
    assert skipped == [
        "x.cpp:a .parallel_for( inside a submit",
        "x.cpp:a .memcpy( inside a submit",
        "x.cpp:launch .parallel_for( on cgh, a SET_ASIDE_BY_RECEIVER row",
    ]


def test_std_memcpy_is_no_launch_a_listed_std_array_fill_is_set_aside_and_an_unlisted_receivers_copy_counts():
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
    # buf has no row, so its copy counts: only a reviewed row sets a call aside
    assert receiver_census(sample, {"shortcut:x.cpp:a:arr": (1, "a std::array")}, skipped=skipped) == [
        ("a", "shortcut"), ("b", "shortcut")]
    assert skipped == ["x.cpp:a .fill( on arr, a SET_ASIDE_BY_RECEIVER row"]


def test_a_shortcut_with_no_row_for_its_receiver_counts():
    sample = "void a() { auto q = make_queue(); q.parallel_for(r, k); }\n"
    skipped: list[str] = []
    assert receiver_census(sample, {"shortcut:x.cpp:a:p": (1, "another receiver")}, skipped=skipped) == [
        ("a", "shortcut")]
    assert skipped == []


def test_every_set_aside_row_is_what_the_receiver_rule_finds_in_the_tree():
    problems = tree_set_aside_problems(read_tree())
    assert not problems, "SET_ASIDE_BY_RECEIVER is wrong:\n" + "\n".join(problems)


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


def problems_added_by(name: str, probe: str) -> tuple[list[str], int]:
    """The ALLOW problems that appending probe to the real file name adds to the tree's own (so the control fails
    only on its own cause, whatever else has drifted), and the line the probe's function is on."""
    texts = read_tree()
    before = diff_against_allow(census_tree(texts), ALLOW)
    texts[name] = texts[name] + probe
    line = texts[name].split("\n").index(probe.strip()) + 1
    return [problem for problem in diff_against_allow(census_tree(texts), ALLOW) if problem not in before], line


def test_a_raw_queue_shortcut_added_to_a_real_file_is_reported():
    probe = "\nstatic void census_probe_added(sycl::queue * q) { q->parallel_for(r, [=](sycl::id<1> i) {}); }\n"
    added, line = problems_added_by("norm.cpp", probe)
    assert added == [f"shortcut:norm.cpp:census_probe_added: 1 raw launch(es) in the tree (norm.cpp:{line}), 0 "
                     "listed: wrap them, or list them in ALLOW with why they are not wrapped (To update, b)"]


def test_a_raw_submit_added_to_a_real_file_is_reported():
    probe = "\nstatic void census_probe_added(sycl::queue * q) { q->submit([&](sycl::handler & cgh) {}); }\n"
    added, line = problems_added_by("norm.cpp", probe)
    assert added == [f"submit:norm.cpp:census_probe_added: 1 raw launch(es) in the tree (norm.cpp:{line}), 0 "
                     "listed: wrap them, or list them in ALLOW with why they are not wrapped (To update, b)"]


LISTED_ONE_SUBMIT = "void b(sycl::queue & q) {\n    q.submit(c);\n}\n"


def test_a_second_raw_submit_in_a_listed_function_is_reported():
    allow = {"submit:x.cpp:b": (1, "one copy")}
    assert diff_against_allow(census({"x.cpp": LISTED_ONE_SUBMIT}), allow) == []
    grown = LISTED_ONE_SUBMIT.replace("    q.submit(c);\n", "    q.submit(c);\n    q.submit(c);\n")
    assert diff_against_allow(census({"x.cpp": grown}), allow) == [
        "submit:x.cpp:b: 2 raw launch(es) in the tree (x.cpp:2, x.cpp:3), 1 listed: wrap them, or list them in "
        "ALLOW with why they are not wrapped (To update, b)"
    ]


def test_a_listed_function_whose_submits_are_gone_fails():
    allow = {"submit:x.cpp:b": (1, "one copy")}
    gone = SUBMIT.sub(".submitted_elsewhere(", LISTED_ONE_SUBMIT)
    assert diff_against_allow(census({"x.cpp": gone}), allow) == [
        'submit:x.cpp:b: 0 raw launch(es) in the tree, 1 listed ("one copy"): lower the count or delete the entry '
        "(To update, a)"
    ]


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
    assert not unread, f"in neither FILES nor EXCLUDED_FILES (To update, c): {unread}"
    both = sorted(on_path & set(EXCLUDED_FILES))
    assert not both, f"both on the path and excluded: {both}"
    blank = [name for name, reason in EXCLUDED_FILES.items() if not reason.strip()]
    assert not blank, f"excluded with no reason: {blank}"
    stale = [name for name in EXCLUDED_FILES if not (SYCL / name).exists()]
    assert not stale, f"excluded files that are gone: {stale}"


def test_every_file_excused_by_the_receiver_rule_censuses_to_nothing():
    texts = read_tree()
    cited = receiver_rule_files()
    assert cited, "no exclusion cites the receiver rule, so this test checks nothing"
    for name in cited:
        skipped: list[str] = []
        found = census({name: (SYCL / name).read_text(encoding="utf-8")}, derive_wrappers(texts), skipped)
        assert not found, f"{name} is excluded as explained by the receiver rule, but holds:\n{format_sites(found)}"
        assert any(".submit( on " in entry and "SET_ASIDE_BY_RECEIVER" in entry for entry in skipped), \
            f"{name}: no SET_ASIDE_BY_RECEIVER row set anything aside"


def test_every_allowlist_entry_has_a_reason():
    bad = [key for key, (count, reason) in ALLOW.items() if count <= 0 or not reason.strip()]
    assert not bad, f"ALLOW entries with no count or no reason: {bad}"


def test_census_matches_the_allowlist():
    t0 = time.monotonic()
    texts = read_tree()
    matches: dict[str, list[int]] = {}
    found = census_tree(texts, rule_matches=matches)
    print(count_by_kind_and_file(found))
    print(f"{sum(len(s) for s in found.values())} raw launch(es) in {len(census_keys(found))} key(s) "
          f"over {len(FILES)} file(s), census took {time.monotonic() - t0:.2f} s")
    # a wrong row turns its launches into unlisted ones: name the row, not the ALLOW lines it would cause
    assert not record_table_problems(matches, RECORDED_ON_THE_PATH), \
        "RECORDED_ON_THE_PATH is wrong (test_the_record_rule_excuses_exactly_the_tables_launches_of_the_tree names " \
        "the row); the ALLOW comparison waits for it"
    assert not [p for p in tree_set_aside_problems(texts) if p.split(":")[1] in FILES], \
        "SET_ASIDE_BY_RECEIVER is wrong (test_every_set_aside_row_is_what_the_receiver_rule_finds_in_the_tree names " \
        "the row); the ALLOW comparison waits for it"
    problems = diff_against_allow(found, ALLOW, matches)
    assert not problems, "\n".join(problems) + "\n\n`python3 tests/" + Path(__file__).name + \
        " --print` lists every key and writes the ALLOW lines to paste"


def test_a_submit_on_a_receiver_with_a_row_is_set_aside_and_every_other_counts():
    sample = (
        "void a(thread_pool & pool) { pool.submit([] { work(); }); }\n"
        "void b(sycl::queue & q) { q.submit(cgf); }\n"
        "void c() { auto q = make_queue(); q.submit(cgf); }\n"
    )
    skipped: list[str] = []
    # no type is read: the row for `pool` in a is what sets that submit aside, and nothing else is
    assert receiver_census(sample, {"submit:x.cpp:a:pool": (1, "a host pool")}, skipped=skipped) == [
        ("b", "submit"), ("c", "submit")]
    assert skipped == ["x.cpp:a .submit( on pool, a SET_ASIDE_BY_RECEIVER row"]
    assert receiver_census(sample, {}) == [("a", "submit"), ("b", "submit"), ("c", "submit")]


def test_an_if_constexpr_block_is_not_named_as_a_function():
    sample = "void a(sycl::queue * q) {\n    if constexpr (X) { q->submit(cgf); }\n}\n"
    assert raw_functions(sample) == ["a"]


def test_a_mismatch_names_its_key_its_launches_and_what_to_do_in_both_directions():
    found = census({"x.cpp": "void b(sycl::queue & q) {\n    q.submit(c);\n    q.submit(c);\n}\n"})
    allow = {"submit:x.cpp:gone": (1, "was the old copy")}
    assert diff_against_allow(found, allow) == [
        "submit:x.cpp:b: 2 raw launch(es) in the tree (x.cpp:2, x.cpp:3), 0 listed: wrap them, or list them in "
        "ALLOW with why they are not wrapped (To update, b)",
        'submit:x.cpp:gone: 0 raw launch(es) in the tree, 1 listed ("was the old copy"): lower the count or delete '
        "the entry (To update, a)",
    ]
    assert diff_against_allow(found, {"submit:x.cpp:b": (2, "two copies")}) == []


def test_a_wrong_record_row_names_its_count_and_the_rules_matches():
    one = FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_EVENT}(lab, ev);\n}}\n"
    table = {"x.cpp:f": 2, "x.cpp:g": 1}
    matches: dict[str, list[int]] = {}
    census({"x.cpp": one}, table=table, rule_matches=matches)
    assert matches == {"x.cpp:f": [2]}
    assert record_table_problems(matches, table) == [
        "x.cpp:f: RECORDED_ON_THE_PATH says 2, the record rule matches 1 (x.cpp:2): read the function and set the "
        "row to the launches whose event it records (To update, d)",
        "x.cpp:g: RECORDED_ON_THE_PATH says 1, the record rule matches 0: read the function and set the row to the "
        "launches whose event it records (To update, d)",
    ]
    assert record_table_problems(matches, {"x.cpp:f": 1}) == []


def test_the_print_report_lists_keys_the_paste_lines_and_the_record_matches():
    found = census({"x.cpp": "void b(sycl::queue & q) {\n    q.submit(c);\n}\n"
                             "void k(sycl::queue & q) {\n    q.submit(c);\n    q.submit(c);\n}\n"})
    allow = {"submit:x.cpp:gone": (1, "old"), "submit:x.cpp:k": (1, "a copy")}
    lines = report(found, {"x.cpp:f": [3]}, allow, {"x.cpp:f": 1}).split("\n")
    for line in [
        "submit:x.cpp:b 1 (x.cpp:2)",
        "submit:x.cpp:k 2 (x.cpp:5, x.cpp:6)",
        '    "submit:x.cpp:b": (1, "TODO: what it is, and why it is not wrapped"),',
        '    "submit:x.cpp:k": (2, "a copy"),',
        '#   "submit:x.cpp:gone": (1, "old"),  <- delete: 0 in the tree',
        "x.cpp:f: the record rule matches 1 (x.cpp:3), RECORDED_ON_THE_PATH says 1",
    ]:
        assert line in lines, f"{line!r} missing from the report:\n" + "\n".join(lines)


def test_allow_is_written_sorted_by_key_with_no_duplicate():
    text = Path(__file__).read_text(encoding="utf-8")
    block = text[text.index("\nALLOW: dict"):]
    block = block[:block.index("\n}\n")]
    keys = re.findall(r'^    "((?:submit|shortcut):[^"]+)":', block, re.M)
    duplicated = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicated, f"ALLOW lists these keys twice (a dict literal keeps only the last): {duplicated}"
    unsorted = [b for a, b in zip(keys, keys[1:]) if b < a]
    assert not unsorted, f"ALLOW keys out of order, first at {unsorted[0]}"
    assert len(keys) == len(ALLOW), "an ALLOW key is not at the start of its own line"


def includers_outside_tests() -> dict[str, list[str]]:
    """basename -> the sources outside any tests/ directory that #include a file of that name, over the trees a
    backend or tool is built from."""
    found: dict[str, list[str]] = {}
    for top in ("ggml", "src", "common", "tools", "examples"):
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix not in (".c", ".cpp", ".h", ".hpp") or not path.is_file() or "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r'^\s*#\s*include\s*[<"](?:[^">]*/)?([^/">]+)[">]', text, re.M):
                found.setdefault(m.group(1), []).append(path.relative_to(ROOT).as_posix())
    return found


def test_every_nothing_includes_it_reason_is_true():
    includers = includers_outside_tests()
    assert includers.get("norm.hpp"), "the include scan finds nothing, so its silence below would mean nothing"
    claimed = {}
    for name, reason in EXCLUDED_FILES.items():
        header = re.fullmatch(r"its header (\S+) \(besides this file\): " + re.escape(NOT_INCLUDED), reason)
        if header or reason.endswith(NOT_INCLUDED):
            claimed[name] = header.group(1) if header else name.split("/")[-1]
    assert claimed, "no reason claims that nothing includes a file"
    false = {}
    for name, checked in claimed.items():
        others = [path for path in includers.get(checked, []) if path != f"ggml/src/ggml-sycl/{name}"]
        if others:
            false[name] = others
    assert not false, f"excluded as {NOT_INCLUDED!r}, but included by: {false}"


def receiver_census(src: str, rows: dict[str, tuple[int, str]], skipped: list[str] | None = None,
                    matches: dict[str, list[str]] | None = None,
                    tripped: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """(function, kind) of each raw launch of src as x.cpp, with rows standing in for SET_ASIDE_BY_RECEIVER."""
    found = census({"x.cpp": src}, skipped=skipped, set_aside_table=rows, receiver_matches=matches,
                   tripped=tripped)
    return [(func, kind) for _, func, kind in found.get("x.cpp", [])]


def test_a_row_names_the_receiver_expression_as_written_prefix_included():
    sample = (
        "void f(Ctx * c, Ctx & d, Pool & q, lazy_engine * engine) {\n"
        "    c->q.submit(job);\n"
        "    d.q.submit(job);\n"
        "    q.submit(job);\n"
        "    engine -> queue . submit(job);\n"
        "    this->pool_.submit(job);\n"
        "}\n"
    )
    rows = {"submit:x.cpp:f:q": (1, "test"), "submit:x.cpp:f:engine->queue": (1, "test"),
            "submit:x.cpp:f:this->pool_": (1, "test")}
    skipped: list[str] = []
    assert receiver_census(sample, rows, skipped=skipped) == [("f", "submit"), ("f", "submit")]
    assert skipped == [
        "x.cpp:f .submit( on q, a SET_ASIDE_BY_RECEIVER row",
        "x.cpp:f .submit( on engine->queue, a SET_ASIDE_BY_RECEIVER row",
        "x.cpp:f .submit( on this->pool_, a SET_ASIDE_BY_RECEIVER row",
    ]
    found = census({"x.cpp": sample}, set_aside_table=rows)
    assert [line for line, _, _ in found.get("x.cpp", [])] == [2, 3]


def test_a_receiver_that_is_not_a_chain_of_names_matches_no_row():
    # receivers no row may name, each with a row for the text it would otherwise be keyed by
    for call, text in [
        ("static_cast<sycl::queue &>(g).fill(p, 0, n);", "(g)"),
        ("std::get<0>(t).fill(p, 0, n);", "(t)"),
        ("F<T>(x).fill(p, 0, n);", "(x)"),
        ("(g).fill(p, 0, n);", "(g)"),
        ("(*ready).fill(p, 0, n);", "(*ready)"),
        ("foo.stream().fill(p, 0, n);", "foo.stream()"),
        ("g_pools[d].fill(p, 0, n);", "g_pools[d]"),
        ("a.\\\n    b.fill(p, 0, n);", "b"),
        # the character right before the chain: a dereference, an address, a product, a conjunction, a continuation
        ("*q.fill(p);", "q"),
        ("auto z = &q.fill(p);", "q"),
        ("x = a * q.fill(p);", "q"),
        ("x = a && q.fill(p);", "q"),
        ("\\\n    q.fill(p);", "q"),
        # a keyword joined to the chain by `::` is no part of an expression
        ("return ::q.fill(p);", "return::q"),
    ]:
        src = "void f() {\n    " + call + "\n}\n"
        assert receiver_census(src, {f"shortcut:x.cpp:f:{text}": (1, "test")}) == [("f", "shortcut")], src


def test_a_row_covering_two_definitions_of_one_function_name_is_refused():
    one = "void f(int n) {\n    std::array<int, 4> ready;\n    ready.fill(0);\n}\n"
    two = "void f(sycl::queue & ready) {\n    ready.fill(p, 0, n);\n}\n"
    for src, lines in [(one + two, "x.cpp:3, x.cpp:6"), (two + one, "x.cpp:2, x.cpp:6")]:
        rows = {"shortcut:x.cpp:f:ready": (2, "a std::array")}
        matches: dict[str, list[str]] = {}
        tripped: dict[str, str] = {}
        assert receiver_census(src, rows, matches=matches, tripped=tripped) == [("f", "shortcut")] * 2, src
        assert set_aside_problems(matches, rows, tripped) == [
            f"shortcut:x.cpp:f:ready: refused: the row names 2 definitions of f, with calls at {lines}; a row covers "
            "one definition, so these calls count (To update, e)"], src
    function = "void f(Pool & p) { p.submit(job); }\n"
    src = "namespace a {\n" + function + "}\nnamespace b {\n" + function + "}\n"
    assert receiver_census(src, {"submit:x.cpp:f:p": (2, "a host pool")}) == [("f", "submit")] * 2


def test_a_row_for_calls_in_no_named_function_is_refused():
    src = "auto a = [](Pool & p) { p.submit(job); };\nauto b = [](Pool & p) { p.submit(job); };\n"
    rows = {"submit:x.cpp:?:p": (2, "a host pool")}
    matches: dict[str, list[str]] = {}
    tripped: dict[str, str] = {}
    assert receiver_census(src, rows, matches=matches, tripped=tripped) == [("?", "submit")] * 2
    assert set_aside_problems(matches, rows, tripped) == [
        "submit:x.cpp:?:p: refused: a row needs a named function, and the calls at x.cpp:1, x.cpp:2 have none, so "
        "they count (To update, e)"]


def test_the_innermost_enclosing_submit_decides_for_a_shortcut_inside_nested_submits():
    sample = ("void f(sycl::queue & q, Pool & pool) {\n"
              "    q.submit([&] { pool.submit([&] { x.memcpy(a, b, n); }); });\n}\n")
    # pool's submit is set aside by its row, so the memcpy inside it counts, though q's counted submit encloses both
    assert receiver_census(sample, {"submit:x.cpp:f:pool": (1, "a host pool")}) == [("f", "submit"),
                                                                                   ("f", "shortcut")]


def test_a_free_submit_call_is_no_submit_site_so_a_queue_call_inside_it_counts():
    sample = "void f(sycl::queue & q) {\n    submit([&] { q.memcpy(a, b, n); });\n}\n"
    assert receiver_census(sample, {}) == [("f", "shortcut")]


def test_a_row_whose_receiver_holds_a_scope_operator_is_keyed_and_named_whole():
    rows = {"submit:x.cpp:g:ns::g_pool": (1, "a host pool")}
    assert receiver_census("void g() {\n    ns::g_pool.submit(job);\n}\n", rows) == []
    src = "void f(sycl::queue & g_pool) {\n    ns::g_pool.fill(p, 0, n);\n}\n"
    rows = {"shortcut:x.cpp:f:ns::g_pool": (1, "a std::array")}
    matches: dict[str, list[str]] = {}
    tripped: dict[str, str] = {}
    assert receiver_census(src, rows, matches=matches, tripped=tripped) == [("f", "shortcut")]
    assert set_aside_problems(matches, rows, tripped) == [
        "shortcut:x.cpp:f:ns::g_pool: refused: x.cpp:1 binds g_pool next to `queue`; read its declaration, and if "
        "the receiver is a queue, delete the row (To update, e)"]


def test_a_row_whose_count_differs_or_whose_call_is_gone_names_what_to_do():
    sample = (
        "void a(ThreadPool & pool) {\n    pool.submit(job);\n    pool.submit(job);\n}\n"
        "void b(ThreadPool & pool) {\n    pool.submit(job);\n}\n"
    )
    rows = {"submit:x.cpp:a:pool": (1, "a host pool"), "submit:x.cpp:gone:pool": (2, "a host pool")}
    matches: dict[str, list[str]] = {}
    # b has no row, so its call counts and is no problem of the table
    assert receiver_census(sample, rows, matches=matches) == [("a", "submit"), ("a", "submit"), ("b", "submit")]
    assert set_aside_problems(matches, rows) == [
        "submit:x.cpp:a:pool: 2 call(s) on this receiver in the function (x.cpp:2, x.cpp:3), SET_ASIDE_BY_RECEIVER "
        "says 1: read the declaration of each receiver; if it is a queue, delete the row; otherwise set the count to "
        "2 (To update, e)",
        "submit:x.cpp:gone:pool: SET_ASIDE_BY_RECEIVER says 2, no call in the function has this receiver: delete the "
        "row (To update, e)",
    ]
    rows["submit:x.cpp:a:pool"] = (2, "a host pool")
    del rows["submit:x.cpp:gone:pool"]
    matches.clear()
    assert receiver_census(sample, rows, matches=matches) == [("b", "submit")]
    assert set_aside_problems(matches, rows) == []


def test_the_tripwire_refuses_a_row_whose_name_the_function_binds_to_a_queue():
    for src, line, func in [
        ("void f(sycl::queue & ready) {\n    ready.fill(x, 0, n);\n}\n", 1, "f"),
        ("void f() {\n    sycl::queue * ready = get();\n    ready->fill(x, 0, n);\n}\n", 2, "f"),
        ("void f(std::vector<sycl::queue *> & qs) {\n    for (sycl::queue * ready : qs) {\n"
         "        ready->fill(x, 0, n);\n    }\n}\n", 2, "f"),
        # each of the following only one binding pattern catches: by value, by assignment, a structured binding
        ("void f(sycl::device dev) {\n    sycl::queue ready{dev};\n    ready.fill(x, 0, n);\n}\n", 2, "f"),
        ("void f(Ctx * ctx) {\n    auto & ready = ctx->queue();\n    ready.fill(x, 0, n);\n}\n", 2, "f"),
        ("void f(sycl::queue * q) {\n    auto [ready, n] = std::pair<sycl::queue *, int>(q, 0);\n"
         "    ready->fill(x, 0, n);\n}\n", 2, "f"),
        # an init-capture
        ("void f(Ctx & ctx) {\n    auto g = [ready = ctx.queue]() {\n        ready.fill(x, 0, n);\n    };\n}\n", 2,
         "f/g"),
    ]:
        rows = {f"shortcut:x.cpp:{func}:ready": (1, "a std::array")}
        matches: dict[str, list[str]] = {}
        tripped: dict[str, str] = {}
        assert receiver_census(src, rows, matches=matches, tripped=tripped) == [(func, "shortcut")], src
        assert set_aside_problems(matches, rows, tripped) == [
            f"shortcut:x.cpp:{func}:ready: refused: x.cpp:{line} binds ready next to `queue`; read its declaration, "
            "and if the receiver is a queue, delete the row (To update, e)"], src
    # a std::array of that name in a function that also takes a queue is not refused
    src = "void f(sycl::queue & q) {\n    std::array<int, 4> ready;\n    ready.fill(0);\n    q.memset(p, 0, n);\n}\n"
    assert receiver_census(src, {"shortcut:x.cpp:f:ready": (1, "a std::array")}) == [("f", "shortcut")]


def test_a_queue_shortcut_inside_a_submit_set_aside_by_a_row_counts():
    sample = "void f(ThreadPool & pool, sycl::queue & q) {\n    pool.submit([&] { q.memcpy(a, b, n); });\n}\n"
    skipped: list[str] = []
    assert receiver_census(sample, {"submit:x.cpp:f:pool": (1, "a host pool")}, skipped=skipped) == [
        ("f", "shortcut")]
    assert skipped == ["x.cpp:f .submit( on pool, a SET_ASIDE_BY_RECEIVER row"]
    # inside a counted submit, the handler's call is that submit's command, not a second launch
    sample = "void f(sycl::queue & q) {\n    q.submit([&](sycl::handler & h) { h.memcpy(a, b, n); });\n}\n"
    assert receiver_census(sample, {"submit:x.cpp:f:pool": (1, "a host pool")}) == [("f", "submit")]


def test_the_print_report_lists_the_set_aside_calls_and_the_rows_that_do_not_match():
    sample = "void a(ThreadPool & pool) {\n    pool.submit(job);\n}\nvoid b(Pool & p) {\n    p.submit(job);\n}\n"
    rows = {"submit:x.cpp:a:pool": (1, "a host pool"), "submit:x.cpp:b:p": (2, "a host pool")}
    skipped: list[str] = []
    matches: dict[str, list[str]] = {}
    receiver_census(sample, rows, skipped=skipped, matches=matches)
    assert receiver_sections(skipped, set_aside_problems(matches, rows)).split("\n") == [
        "# set aside by a SET_ASIDE_BY_RECEIVER row, in the files on the path",
        "x.cpp:a .submit( on pool, a SET_ASIDE_BY_RECEIVER row",
        "# SET_ASIDE_BY_RECEIVER rows that do not match the files on the path (excluded files included)",
        "submit:x.cpp:b:p: 1 call(s) on this receiver in the function (x.cpp:5), SET_ASIDE_BY_RECEIVER says 2: read "
        "the declaration of each receiver; if it is a queue, delete the row; otherwise set the count to 1 "
        "(To update, e)",
    ]
    assert receiver_sections([], []).split("\n") == [
        "# set aside by a SET_ASIDE_BY_RECEIVER row, in the files on the path", "(none)",
        "# SET_ASIDE_BY_RECEIVER rows that do not match the files on the path (excluded files included)", "(none)",
    ]


def test_an_unlisted_launch_the_record_rule_matches_also_names_the_record_table():
    src = FN_HEAD + f"sycl::event ev = {LAUNCH_K1};\n{RECORD_EVENT}(lab, ev);\n}}\n"
    matches: dict[str, list[int]] = {}
    found = census({"x.cpp": src}, table={}, rule_matches=matches)
    assert diff_against_allow(found, {}, matches) == [
        "submit:x.cpp:f: 1 raw launch(es) in the tree (x.cpp:2), 0 listed: wrap them, or list them in ALLOW with "
        "why they are not wrapped (To update, b); or, if a record call receives its event, add a "
        "RECORDED_ON_THE_PATH row (To update, d)",
    ]


if __name__ == "__main__":
    import sys

    import pytest

    if "--print" in sys.argv[1:]:
        sys.exit(print_report())
    sys.exit(pytest.main([__file__, "-q"]))
