#!/usr/bin/env python3
"""A hybrid MXFP4 xmx_tiled expert must be decoded by a kernel that reads xmx_tiled (llama.cpp-4hg7).

THE DEFECT.  GPT-OSS 20B on the B50 at GGML_SYCL_VRAM_BUDGET_PCT=60 keeps blk.12 as the one MIXED
tensor: 12 experts on the device (layout xmx_tiled), 20 on the host.  At decode a token routes
4 slots, 1 to a device expert and 3 to host experts.  The device slot's gate/up output was garbage
(+-1e33, NaN) while the three host slots matched the reference exactly.

Two defects, one chain (the GPU run of 2026-10-01 shows the first, the code shows the second):

  1. mmvq_moe_batched_dispatch's MXFP4 XMX_TILED branch -- the only grouped kernel that decodes
     xmx_tiled -- declined with reason=coverage whenever the device entries did not cover EVERY
     (id, token) slot (`full_gpu_cover`), although the kernel is driven by per-slot route arrays
     and never needed an all-slots cover.  Hybrid decode is exactly partial cover.
  2. The decline dropped the op onto the per-expert fallback, ggml_sycl_mul_mat(..., &entry_layout)
     with entry_layout = xmx_tiled.  Its MXFP4 direct path mapped any forced layout it did not know
     onto AOS (the `default:` of a switch), and its reconcile guard compares the STORED layout to
     the ADVERTISED one, which agree -- so it then decoded xmx_tiled bytes as AOS.

THE CONTRACT.
  A. (mmvq.cpp, XMX_TILED branch) the coverage decision is `moe_mmvq_xmx_tiled_grouped_accepts_cover`
     (a pure predicate in moe-mmvq-tables.hpp, unit-tested by test-sycl-moe-mmvq-tables); the old
     `grouped_decode_shape = full_gpu_cover` and the second `!full_gpu_cover ||` caps test are gone;
     the "slot-missing" scan runs only for full cover; the group row count is compared with
     n_gpu_entries (== total_batches only for full cover); a sparse (rows-per-expert / occupancy)
     batch is accepted for any cover, not only full.
  B. (ggml-sycl.cpp, MXFP4 direct path of ggml_sycl_mul_mat) a forced layout the direct path cannot
     decode (`!moe_mmvq_mxfp4_direct_reads_layout(*forced_layout)`) is refused with GGML_ABORT, and
     that refusal sits BEFORE the `switch (*forced_layout)` that would map it onto AOS.

WHAT THIS DOES NOT PROVE.  It reads source text.  It does not show the kernel's numbers are right on
a partial cover, nor that decode is coherent -- that is the GPU repro (PCT=60, -c 1024, -ub 1 -b 1,
eval-callback), run by the lead.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each
to FAIL, then requires the unmodified tree to pass.

THE RED, against 4d70bdea4:
  FAIL: ... grouped_decode_shape ... (llama.cpp-4hg7)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml/src/ggml-sycl/mmvq.cpp"
BACKEND = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"


class ContractError(AssertionError):
    pass


def blank_comments(src: str) -> str:
    out = []
    i = 0
    n = len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def squash(text: str) -> str:
    return re.sub(r"\(\s+", "(", re.sub(r"\s+", " ", text))


def region(code: str, start_marker: str, end_marker: str) -> str:
    a = code.find(start_marker)
    if a < 0:
        raise ContractError(f"FAIL: anchor not found: {start_marker!r} (llama.cpp-4hg7)")
    b = code.find(end_marker, a)
    if b < 0:
        raise ContractError(f"FAIL: end anchor not found after {start_marker!r}: {end_marker!r} (llama.cpp-4hg7)")
    return squash(code[a:b])


def check_mmvq(src: str) -> None:
    code = blank_comments(src)
    # The XMX_TILED branch of the MXFP4 case, up to the next layout branch.
    xmx = region(code, "if (layout == GGML_LAYOUT_XMX_TILED) {\n                constexpr int repeat",
                 "} else if (layout == GGML_LAYOUT_MXFP4_DPAS)")

    if "grouped_decode_shape = moe_mmvq_xmx_tiled_grouped_accepts_cover(full_gpu_cover, n_gpu_entries, xmx_route_arrays_ok);" not in xmx:
        raise ContractError(
            "FAIL: the xmx_tiled grouped_decode_shape is not decided by moe_mmvq_xmx_tiled_grouped_accepts_cover("
            "full_gpu_cover, n_gpu_entries, xmx_route_arrays_ok) -- partial cover (hybrid decode) is declined and "
            "the op falls to a per-expert path that cannot read xmx_tiled (llama.cpp-4hg7)"
        )
    if re.search(r"grouped_decode_shape\s*=\s*full_gpu_cover\s*;", xmx):
        raise ContractError("FAIL: grouped_decode_shape = full_gpu_cover has come back (llama.cpp-4hg7)")
    if re.search(r"!\s*full_gpu_cover\s*\|\|", xmx):
        raise ContractError("FAIL: a `!full_gpu_cover ||` rejection survives in the xmx_tiled branch (llama.cpp-4hg7)")

    scan = re.search(r"if \(full_gpu_cover\) \{ for \(uint8_t seen : seen_slots\) \{ if \(!seen\) \{ log_xmx_reject\(\"slot-missing\"\);", xmx)
    if not scan:
        raise ContractError(
            "FAIL: the 'slot-missing' scan must run only for full cover; on partial cover the unseen slots belong "
            "to the host experts (llama.cpp-4hg7)"
        )
    if "grouped_rows_host.size() != static_cast<size_t>(n_gpu_entries)" not in xmx:
        raise ContractError(
            "FAIL: the grouped row count must be compared with n_gpu_entries, not total_batches (equal only for "
            "full cover) (llama.cpp-4hg7)"
        )
    sparse = re.search(r"const bool sparse_xmx_batch = ([^;]*);", xmx)
    if not sparse or "full_gpu_cover" in sparse.group(1):
        raise ContractError(
            "FAIL: sparse_xmx_batch is conditioned on full_gpu_cover; a one-row partial decode batch would be "
            "declined by the occupancy policy (llama.cpp-4hg7)"
        )


def check_backend(src: str) -> None:
    code = blank_comments(src)
    a = code.find("if (src0->type == GGML_TYPE_MXFP4 && !src0_planned_host && (src0_on_device")
    if a < 0:
        raise ContractError("FAIL: MXFP4 direct path anchor not found in ggml_sycl_mul_mat (llama.cpp-4hg7)")
    body = squash(code[a : a + 12000])
    guard = re.search(
        r"if \(forced_layout && !moe_mmvq_mxfp4_direct_reads_layout\(\*forced_layout\)\) \{ GGML_ABORT\(", body)
    sw = body.find("switch (*forced_layout)")
    if not guard:
        raise ContractError(
            "FAIL: the MXFP4 direct path does not refuse a forced layout it cannot decode "
            "(`!moe_mmvq_mxfp4_direct_reads_layout(*forced_layout)` -> GGML_ABORT); xmx_tiled would be read as "
            "AOS (llama.cpp-4hg7)"
        )
    if sw < 0 or guard.start() > sw:
        raise ContractError(
            "FAIL: the refusal must precede the `switch (*forced_layout)` whose default arm maps the layout onto "
            "AOS (llama.cpp-4hg7)"
        )


def check(mmvq_src: str, backend_src: str) -> None:
    check_mmvq(mmvq_src)
    check_backend(backend_src)


def self_test(mmvq_src: str, backend_src: str) -> int:
    failures: list[str] = []

    def expect_fail(name: str, m: str | None, b: str | None) -> None:
        mm = mmvq_src if m is None else m
        bb = backend_src if b is None else b
        if mm == mmvq_src and bb == backend_src:
            failures.append(f"{name}: mutation did not change the source (anchor stale)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        try:
            check(mm, bb)
        except ContractError:
            print(f"  mutant {name}: caught")
            return
        failures.append(f"{name}: mutant survived")
        print(f"  mutant {name}: SURVIVED")

    def sub(src: str, old: str, new: str) -> str:
        return src.replace(old, new, 1)

    cover_call = "moe_mmvq_xmx_tiled_grouped_accepts_cover(full_gpu_cover, n_gpu_entries, xmx_route_arrays_ok);"
    expect_fail("cover-back-to-full-only", sub(mmvq_src, cover_call, "full_gpu_cover;"), None)
    expect_fail("cover-ignores-route-arrays", sub(mmvq_src, cover_call,
                "moe_mmvq_xmx_tiled_grouped_accepts_cover(full_gpu_cover, n_gpu_entries, true);"), None)
    expect_fail("caps-test-regains-full-cover", sub(mmvq_src,
                "if (!xmx_capabilities_match_int8_tile(caps, repeat, exec_n, k_per) ||\n                    !xmx_capabilities_support_sub_group(caps, GGML_SYCL_MXFP4_MOE_XMX_SG) ||\n                    caps.optimal_tiles_n <= 0) {\n                    log_xmx_reject(\"caps\");",
                "if (!full_gpu_cover || !xmx_capabilities_match_int8_tile(caps, repeat, exec_n, k_per) ||\n                    !xmx_capabilities_support_sub_group(caps, GGML_SYCL_MXFP4_MOE_XMX_SG) ||\n                    caps.optimal_tiles_n <= 0) {\n                    log_xmx_reject(\"caps\");"), None)
    expect_fail("slot-scan-unconditional", sub(mmvq_src,
                "                if (full_gpu_cover) {\n                    for (uint8_t seen : seen_slots) {",
                "                {\n                    for (uint8_t seen : seen_slots) {"), None)
    expect_fail("rows-compared-with-total-batches", sub(mmvq_src,
                "grouped_rows_host.size() != static_cast<size_t>(n_gpu_entries)",
                "grouped_rows_host.size() != static_cast<size_t>(total_batches)"), None)
    expect_fail("sparse-batch-full-cover-only", sub(mmvq_src,
                "const bool sparse_xmx_batch = std::strcmp(",
                "const bool sparse_xmx_batch = full_gpu_cover && std::strcmp("), None)
    guard = "if (forced_layout && !moe_mmvq_mxfp4_direct_reads_layout(*forced_layout)) {"
    expect_fail("direct-guard-removed", None, sub(backend_src, guard, "if (false) {"))
    expect_fail("direct-guard-not-abort", None, sub(backend_src, 'GGML_ABORT(\n                "[MXFP4-DIRECT] %s: no decode',
                'GGML_LOG_WARN(\n                "[MXFP4-DIRECT] %s: no decode'))
    # guard after the switch: move the guard text below the switch by renaming the first switch anchor.
    at = backend_src.find(guard)
    sw = backend_src.find("switch (*forced_layout)", at)
    if at >= 0 and sw > at:
        # a second, earlier switch makes the guard come AFTER a switch (*forced_layout)
        earlier = backend_src[:at] + "switch (*forced_layout) { default: break; }\n        " + backend_src[at:]
        expect_fail("direct-guard-after-switch", None, earlier)

    try:
        check(mmvq_src, backend_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASS: 9 mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    mmvq_src = MMVQ.read_text()
    backend_src = BACKEND.read_text()
    if "--self-test" in argv:
        return self_test(mmvq_src, backend_src)
    try:
        check(mmvq_src, backend_src)
    except ContractError as e:
        print(e)
        return 1
    print(
        "PASS: the grouped xmx_tiled executor accepts partial cover, and the per-expert MXFP4 direct path "
        "refuses layouts it cannot decode instead of reading them as AOS (llama.cpp-4hg7)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
