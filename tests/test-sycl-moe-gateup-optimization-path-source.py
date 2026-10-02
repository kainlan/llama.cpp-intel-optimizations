#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"
COMMON = ROOT / "ggml" / "src" / "ggml-sycl" / "common.hpp"
REGISTRY = ROOT / "tools" / "sycl-kernel-bench" / "kernel_registry.hpp"
PLAN = ROOT / "docs" / "plans" / "2026-06-30-sycl-mxfp4-multirhs-gateup-dpas-work-reduction.md"
SYCL_DOC = ROOT / "docs" / "backend" / "SYCL.md"


def slice_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start + len(start_marker))
    return text[start:end]


def blank_comments_and_strings(text: str) -> str:
    """Same length as text, with comments and string/char literal bodies turned into spaces (newlines kept), so a
    brace or a marker inside either is invisible and every offset still indexes the original."""
    out, i, n = [], 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            end = text.find("\n", i)
            end = n if end < 0 else end
        elif two == "/*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
        elif text[i] in "\"'":
            quote, end = text[i], i + 1
            while end < n and text[end] != quote:
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, n)
            out.append(quote + "".join("\n" if c == "\n" else " " for c in text[i + 1:end - 1]) + quote)
            i = end
            continue
        else:
            out.append(text[i])
            i += 1
            continue
        out.append("".join("\n" if c == "\n" else " " for c in text[i:end]))
        i = end
    return "".join(out)


def function_slice(text: str, signature: str) -> str:
    """The function whose definition starts at `signature`, from the signature to its MATCHING closing brace.

    Ending the slice at the next definition's marker lets a decoy function placed in between (or any later
    text up to that marker) satisfy a check about this one; the brace match cannot be fooled by what follows.
    """
    code = blank_comments_and_strings(text)
    start = code.index(signature)
    depth, i = 0, start
    while True:
        char = code[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "{" and depth == 0:
            break
        elif char == ";" and depth == 0:
            raise AssertionError("%s is a declaration, not a definition" % signature)
        i += 1
    braces = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            braces += 1
        elif code[j] == "}":
            braces -= 1
            if braces == 0:
                return text[start:j + 1]
    raise AssertionError("no matching brace for %s" % signature)


S1_SIGNATURE = "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl("
K_REDUCE_SIGNATURE = "SYCL_ESIMD_FUNCTION inline void mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce("


def test_current_xmx_tiled_a_load_layout_is_scale_prefix_plus_payload() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    helper = slice_between(
        mmvq,
        "mxfp4_xmx_tiled_load_a_vec_from_group",
        "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec(",
    )
    assert "constexpr int packed_bytes  = k_per / 2" in helper
    assert "const uint8_t * scale_ptr  = group + xmx_row_in_group" in helper
    assert "const uint8_t * packed_ptr = group + tile_n_total + xmx_row_in_group * packed_bytes" in helper
    assert "block_load<uint8_t, Repeat>(scale_ptr)" in helper
    assert "block_load<uint8_t, compact_bytes>(packed_ptr)" in helper


def test_current_packed_m2_route_uses_272_byte_groups_for_tile_n_16() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    # The S=1 kernel by its matching brace: the K-split kernel that follows also calls the shared helper, and so
    # would any decoy function in between, so ending the slice at "the next definition" lets that call satisfy
    # the check when the S=1 kernel runs a private copy of the loop.
    body = function_slice(mmvq, S1_SIGNATURE)
    assert "const int64_t group_bytes     = tile_n_total * (1 + k_per / 2)" in body
    assert "const int64_t kt_group_stride = n_tile_groups_n * group_bytes" in body
    # e8484d7a7 (llama.cpp-lis9) hoisted the K-tile loop out of this kernel into a helper shared with the
    # K-split variant, so the 272-byte-group A loads now live in the helper. Require the kernel to still drive
    # that helper (not a private copy of the loop) and score the loads where they are.
    # Comment-blind: the kernel's own comments name the helper, which is not a call to it.
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.DOTALL)
    assert re.search(r"mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce<\s*Repeat\s*,\s*Prefetch\s*>\s*\(", code)
    k_reduce = function_slice(mmvq, K_REDUCE_SIGNATURE)
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0" in k_reduce
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(up_group0" in k_reduce
    registry = REGISTRY.read_text(encoding="utf-8")
    assert "mxfp4_pair_glu_xmx_tiled_packed_r8_m2_sparse32_bias" in registry


def test_a_decoy_function_between_the_kernels_cannot_satisfy_the_s1_kernel_check() -> None:
    """The S=1 kernel's call to the shared K-reduce helper must be the kernel's own: move it into a decoy
    function placed after the kernel and the slice must not see it; same for the helper's A loads."""
    mmvq = MMVQ.read_text(encoding="utf-8")
    call = re.compile(r"mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce<\s*Repeat\s*,\s*Prefetch\s*>\s*\(")

    def s1_code(text: str) -> str:
        return blank_comments_and_strings(function_slice(text, S1_SIGNATURE))

    assert call.search(s1_code(mmvq))
    ksplit = mmvq.index("static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_ksplit_sycl(")
    body = function_slice(mmvq, S1_SIGNATURE)
    s1_at = mmvq.index(body)
    # (a) rename the S=1 kernel's own call and plant a decoy carrying the original spelling before the K-split kernel
    own = call.search(body)
    renamed_body = body[:own.start()] + "mxfp4_pair_glu_removed_private_loop<Repeat, Prefetch>(" + body[own.end():]
    decoy = "static void decoy_for_the_s1_check() { mxfp4_pair_glu_xmx_tiled_dpas_m2_k_reduce<Repeat, Prefetch>(); }\n"
    mutant = mmvq[:s1_at] + renamed_body + "\n" + decoy + mmvq[s1_at + len(body):]
    assert mutant != mmvq and mutant.index("decoy_for_the_s1_check") < mutant.index(
        "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_ksplit_sycl(")
    assert not call.search(s1_code(mutant)), "the S=1 slice reached a decoy function after the kernel"
    # (b) the K-reduce helper's A loads, moved to a decoy that follows the helper
    reduce_body = function_slice(mmvq, K_REDUCE_SIGNATURE)
    reduce_at = mmvq.index(reduce_body)
    emptied = reduce_body.replace("mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0", "removed_a_load(gate_group0")
    decoy = "static void decoy_for_the_k_reduce_check() { mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0, up_group0); }\n"
    mutant = mmvq[:reduce_at] + emptied + "\n" + decoy + mmvq[reduce_at + len(reduce_body):]
    assert mutant != mmvq
    assert "mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0" not in function_slice(mutant, K_REDUCE_SIGNATURE)
    assert ksplit > s1_at


def test_single_stream_tg_cannot_use_existing_grouped_gateup_without_more_rows() -> None:
    common = COMMON.read_text(encoding="utf-8")
    assert "static constexpr size_t GGML_SYCL_MXFP4_MOE_XMX_N  = 16" in common
    mmvq = MMVQ.read_text(encoding="utf-8")
    dispatch = slice_between(
        mmvq,
        "const bool grouped_decode_shape = xmx_tiled_grouped_eligible",
        "if (device_grouped_shape)",
    )
    assert "total_batches >= exec_n" in dispatch
    assert "ids_host_count == total_batches" in dispatch


def test_rejected_dpas_column_candidates_stay_rejected() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    assert "Decision: `runtime-rejected`" in plan
    assert "235.588515" in plan
    assert "605.034755" in plan
    assert "1323.299220" in plan
    sycl_doc = SYCL_DOC.read_text(encoding="utf-8")
    assert "No production" in sycl_doc
    assert "GGML_SYCL_MOE_GATEUP_MULTIRHS" in sycl_doc
    assert "route is authorized or wired" in sycl_doc


def v2_payload_offset(row: int, packed_bytes: int = 16) -> int:
    return row * packed_bytes


def v2_scale_offset(row: int, tile_n_total: int = 16, packed_bytes: int = 16) -> int:
    return tile_n_total * packed_bytes + row


def v2_group_bytes(tile_n_total: int = 16, packed_bytes: int = 16) -> int:
    payload_bytes = tile_n_total * packed_bytes
    scale_slab_bytes = 64
    return payload_bytes + scale_slab_bytes


def test_v2_aligned_payload_layout_contract() -> None:
    assert v2_group_bytes() == 320
    assert v2_payload_offset(0) == 0
    assert v2_payload_offset(8) == 128
    assert v2_payload_offset(0) % 64 == 0
    assert v2_payload_offset(8) % 64 == 0
    assert v2_scale_offset(0) == 256
    assert v2_scale_offset(15) == 271
    assert v2_scale_offset(0) >= 256


def test_v2_benchmark_cli_scaffolding_exists() -> None:
    bench = (ROOT / "ggml" / "src" / "ggml-sycl" / "ggml-sycl-bench.hpp").read_text(encoding="utf-8")
    harness = (ROOT / "tools" / "sycl-kernel-bench" / "benchmark_harness.hpp").read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    main = (ROOT / "tools" / "sycl-kernel-bench" / "main.cpp").read_text(encoding="utf-8")
    reference = (ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp").read_text(
        encoding="utf-8"
    )
    assert re.search(r"\bbool\s+xmx_tiled_v2\s*=\s*false\s*;", bench)
    assert re.search(r"\bint\s+xmx_tiled_v2_group_bytes\s*=\s*320\s*;", bench)
    assert "parse_moe_xmx_tiled_v2" in harness
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" in registry
    assert "mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias" in main
    assert "args.xmx_tiled_v2" in reference
    assert "args.xmx_tiled_v2_group_bytes" in reference


def test_v2_reference_generator_transforms_current_xmx_layout() -> None:
    reference = (ROOT / "tools" / "sycl-kernel-bench" / "kernels" / "reference" / "mxfp4_inline_dot.cpp").read_text(
        encoding="utf-8"
    )
    assert "make_xmx_tiled_v2_aligned_payload_layout" in reference
    assert re.search(r"\bold_group_bytes\s*=\s*tile_n_total\s*\*\s*\(1\s*\+\s*packed_bytes\)", reference)
    assert re.search(r"\bnew_group_bytes\s*=\s*tile_n_total\s*\*\s*packed_bytes\s*\+\s*64", reference)
    assert "new_group + row * packed_bytes" in reference
    assert "new_group + tile_n_total * packed_bytes + row" in reference
    assert "make_xmx_tiled_v2_aligned_payload_layout(launch_layout, m, k)" in reference
    assert "logical_expert_bytes = predecoded_i8 ? launch_layout.size() : weights.layout.size()" in reference


def test_v2_mmvq_load_helper_and_bench_branch_exist() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert "mxfp4_xmx_tiled_v2_load_a_vec_from_group" in mmvq
    assert "struct mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_kernel" in mmvq
    assert "mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl" in mmvq
    helper = slice_between(
        mmvq,
        "mxfp4_xmx_tiled_v2_load_a_vec_from_group",
        "template <int Repeat>\nSYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec_from_group",
    )
    assert "const uint8_t * packed_ptr = group + xmx_row_in_group * packed_bytes" in helper
    assert "const uint8_t * scale_ptr  = group + tile_n_total * packed_bytes + xmx_row_in_group" in helper
    body = slice_between(
        mmvq,
        "mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_sycl",
        "static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl",
    )
    assert "const int64_t group_bytes     = 320" in body
    assert "mxfp4_xmx_tiled_v2_load_a_vec_from_group<Repeat>(gate_group0" in body
    assert "mxfp4_xmx_tiled_v2_load_a_vec_from_group<Repeat>(up_group0" in body
    assert "parallel_for<mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_kernel" in body
    assert "gate_part0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>" in body
    assert "up_part0   = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>" in body
    launch = slice_between(mmvq, "case 8:\n                if (args.xmx_tiled_grouped)", "if (args.xmx_tiled_m_tiles == 4)")
    assert "mxfp4_dpas_pack_q8_single_col_groups_sycl" in launch
    assert "if (args.xmx_tiled_v2)" in launch
    assert "args.xmx_tiled_v2_group_bytes == 320" in launch
    assert "mxfp4_pair_glu_xmx_tiled_v2_dpas_m2_submit<8" in launch


def test_launch_timing_instrumentation_is_default_off_when_implemented() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    if "GGML_SYCL_MOE_TG_PROFILE_LAUNCH" not in mmvq:
        return
    assert "mxfp4_moe_tg_profile_launch_enabled" in mmvq
    assert "std::getenv(\"GGML_SYCL_MOE_TG_PROFILE_LAUNCH\")" in mmvq
    assert "launch_submit_us" in mmvq
    assert "launch_device_us" in mmvq
    assert "launch_wait_us" in mmvq


def test_expert_histogram_instrumentation_is_default_off_when_implemented() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    if "GGML_SYCL_MOE_EXPERT_HIST" not in mmvq:
        return
    assert "mxfp4_moe_expert_hist_enabled" in mmvq
    assert "std::getenv(\"GGML_SYCL_MOE_EXPERT_HIST\")" in mmvq
    assert "[MOE-EXPERT-HIST]" in mmvq


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
