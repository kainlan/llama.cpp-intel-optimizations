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
  B. (ggml-sycl.cpp, MXFP4 direct path of ggml_sycl_mul_mat) the loaded layout is the answer.  The
     reconcile guard (advertised vs stored) comes FIRST and still falls through on a mismatch.  The
     EFFECTIVE layout after it -- the forced one, else the tensor's own (src0->extra) -- is then
     checked with `moe_mmvq_mxfp4_direct_reads_layout` and a layout the direct kernels cannot decode
     is refused with GGML_ABORT, before any src0_data is used.  The same predicate guards the
     resolve_weight fallback.  Pins: I1a (abort decided on the effective layout, after the
     reconcile), I1b (the non-forced src0->extra branch is part of the effective layout), I1c (the
     resolve fallback), I1d (ONE fact, ONE source: the effective layout is derived once, and both the
     kernel-ABI mapping `data_layout` and the refusal read that single value, not a second copy).  A pre-reconcile abort is the defect this ordering replaces: it turned a
     forced-vs-stored mismatch that used to warn and fall through into a crash.
     Reachable on GPT-OSS at full budget only if the grouped xmx_tiled / I8 kernel declines for a
     reason other than coverage (caps, shape, tile-n-total, route arrays, kernel-row-limit with the
     chunker off); no path was found by reading, and none was run.

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
            "FAIL [pin A1]: the xmx_tiled grouped_decode_shape is not decided by moe_mmvq_xmx_tiled_grouped_accepts_cover("
            "full_gpu_cover, n_gpu_entries, xmx_route_arrays_ok) -- partial cover (hybrid decode) is declined and "
            "the op falls to a per-expert path that cannot read xmx_tiled (llama.cpp-4hg7)"
        )
    if re.search(r"grouped_decode_shape\s*=\s*full_gpu_cover\s*;", xmx):
        raise ContractError("FAIL [pin A2]: grouped_decode_shape = full_gpu_cover has come back (llama.cpp-4hg7)")
    if re.search(r"!\s*full_gpu_cover\s*\|\|", xmx):
        raise ContractError("FAIL [pin A3]: a `!full_gpu_cover ||` rejection survives in the xmx_tiled branch (llama.cpp-4hg7)")
    # M3: the capability test lives in ONE place (xmx_caps_ok); a second copy after the
    # coverage test was dead code, and a place for a coverage conjunct to creep back in.
    if len(re.findall(r"xmx_capabilities_match_int8_tile\(", xmx)) != 1:
        raise ContractError(
            "FAIL [pin M3a]: the xmx_tiled branch tests the XMX capabilities more than once (or not at all); "
            "the single xmx_caps_ok test must be the only one (llama.cpp-4hg7)"
        )

    scan = re.search(r"if \(full_gpu_cover\) \{ for \(uint8_t seen : seen_slots\) \{ if \(!seen\) \{ log_xmx_reject\(\"slot-missing\"\);", xmx)
    if not scan:
        raise ContractError(
            "FAIL [pin A4]: the 'slot-missing' scan must run only for full cover; on partial cover the unseen slots belong "
            "to the host experts (llama.cpp-4hg7)"
        )
    if not re.search(r"GGML_ASSERT\(grouped_rows_host\.size\(\) == static_cast<size_t>\(n_gpu_entries\) &&", xmx):
        raise ContractError(
            "FAIL [pin A5]: the grouped row count must be asserted against n_gpu_entries (one row per device "
            "entry), not total_batches (equal only for full cover) (llama.cpp-4hg7)"
        )
    if re.search(r"grouped_rows_host\.size\(\)\s*!=", xmx):
        raise ContractError("FAIL [pin A5b]: a runtime row-count rejection has come back; the assert replaces it (llama.cpp-4hg7)")
    sparse = re.search(r"const bool sparse_xmx_batch = ([^;]*);", xmx)
    if not sparse or "full_gpu_cover" in sparse.group(1):
        raise ContractError(
            "FAIL [pin A6]: sparse_xmx_batch is conditioned on full_gpu_cover; a one-row partial decode batch would be "
            "declined by the occupancy policy (llama.cpp-4hg7)"
        )


def check_backend(src: str) -> None:
    code = blank_comments(src)
    a = code.find("if (src0->type == GGML_TYPE_MXFP4 && !src0_planned_host && (src0_on_device")
    if a < 0:
        raise ContractError("FAIL: MXFP4 direct path anchor not found in ggml_sycl_mul_mat (llama.cpp-4hg7)")
    body = squash(code[a : a + 14000])

    # The reconcile guard: advertised vs stored.  It comes first and falls through on a mismatch.
    rec_at = body.find("bool layout_reconciled = true;")
    rec_false = body.find("layout_reconciled = false;")
    if rec_at < 0 or rec_false < rec_at:
        raise ContractError("FAIL: the reconcile guard (layout_reconciled) was not found in the MXFP4 direct path (llama.cpp-4hg7)")

    # I1.  The effective layout is the forced one, else the tensor's own -- and the refusal is
    # decided on THAT, after the reconcile.
    eff = re.search(
        r"const layout_mode direct_effective_layout = forced_layout \? \*forced_layout : src0->extra \? "
        r"get_effective_layout_mode\(static_cast<const ggml_tensor_extra_gpu \*>\(src0->extra\)\) : GGML_LAYOUT_AOS;",
        body,
    )
    if not eff:
        raise ContractError(
            "FAIL [pin I1b]: the effective layout of the MXFP4 direct path no longer covers the NON-forced "
            "branch (src0->extra): an effective xmx_tiled would still be mapped onto AOS (llama.cpp-4hg7)"
        )
    guard = re.search(
        r"if \(layout_reconciled && !moe_mmvq_mxfp4_direct_reads_layout\(direct_effective_layout\)\) \{ GGML_ABORT\(", body)
    if not guard:
        raise ContractError(
            "FAIL [pin I1a]: the MXFP4 direct path does not abort on an EFFECTIVE layout it cannot decode, decided "
            "after the reconcile (`layout_reconciled && !moe_mmvq_mxfp4_direct_reads_layout(direct_effective_layout)`): "
            "either xmx_tiled would be read as AOS, or a forced-vs-stored mismatch that used to fall through "
            "now aborts (llama.cpp-4hg7)"
        )
    if not (rec_false < guard.start() and eff.start() < guard.start()):
        raise ContractError(
            "FAIL [pin I1a]: the abort precedes the reconcile guard; the loaded layout is the answer, so reconcile "
            "FIRST and abort only on the effective layout (llama.cpp-4hg7)"
        )
    # I1d.  One derivation.  Up to the first use of src0_data the layout is derived exactly once;
    # the kernel-ABI mapping (data_layout) is a switch on that value, and nothing re-derives it from
    # the forced layout or the tensor's extra.
    upto = body[: body.find("const void * src0_data = nullptr;")]
    if (
        len(re.findall(r"const layout_mode direct_effective_layout =", upto)) != 1
        or len(re.findall(r"get_effective_layout_mode\(static_cast<const ggml_tensor_extra_gpu \*>\(src0->extra\)\)", upto)) != 1
        or "switch (*forced_layout)" in upto
        or not re.search(r"direct_effective_layout = .*? switch \(direct_effective_layout\) \{ case GGML_LAYOUT_SOA:", upto)
    ):
        raise ContractError(
            "FAIL [pin I1d]: the MXFP4 direct path derives the effective layout more than once (one fact, two "
            "sources): data_layout must be a switch on the single direct_effective_layout that the refusal also "
            "reads (llama.cpp-4hg7)"
        )
    src0_data_at = body.find("const void * src0_data = nullptr;")
    if src0_data_at < guard.start():
        raise ContractError("FAIL [pin I1a]: the abort must precede the first use of src0_data (llama.cpp-4hg7)")
    if re.search(r"if \(forced_layout && !moe_mmvq_mxfp4_direct_reads_layout\(\*forced_layout\)\)", body):
        raise ContractError("FAIL [pin I1a]: a pre-reconcile abort on the forced layout survives (llama.cpp-4hg7)")
    fallback = re.search(
        r"auto resolved = ggml_sycl_resolve\(src0, ctx\.device\); if \(resolved\) \{ "
        r"if \(!moe_mmvq_mxfp4_direct_reads_layout\(resolved\.layout\)\) \{ GGML_ABORT\(",
        body,
    )
    if not fallback:
        raise ContractError(
            "FAIL [pin I1c]: the resolve_weight fallback of the MXFP4 direct path maps a resolved layout it cannot "
            "decode onto AOS instead of refusing (llama.cpp-4hg7)"
        )


def check(mmvq_src: str, backend_src: str) -> None:
    check_mmvq(mmvq_src)
    check_backend(backend_src)


def self_test(mmvq_src: str, backend_src: str) -> int:
    failures: list[str] = []

    def expect_fail(name: str, m: str | None, b: str | None, pin: str | None = None) -> None:
        mm = mmvq_src if m is None else m
        bb = backend_src if b is None else b
        if mm == mmvq_src and bb == backend_src:
            failures.append(f"{name}: mutation did not change the source (anchor stale)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        try:
            check(mm, bb)
        except ContractError as e:
            if pin is not None and f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: failed, but not on pin {pin}: {str(e)[:120]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:60]}...)")
                return
            print(f"  mutant {name}: caught" + (f" (pin {pin})" if pin else ""))
            return
        failures.append(f"{name}: mutant survived")
        print(f"  mutant {name}: SURVIVED")

    def sub(src: str, old: str, new: str) -> str:
        return src.replace(old, new, 1)

    cover_call = "moe_mmvq_xmx_tiled_grouped_accepts_cover(full_gpu_cover, n_gpu_entries, xmx_route_arrays_ok);"
    expect_fail("cover-back-to-full-only", sub(mmvq_src, cover_call, "full_gpu_cover;"), None, "A1")
    expect_fail("cover-ignores-route-arrays", sub(mmvq_src, cover_call,
                "moe_mmvq_xmx_tiled_grouped_accepts_cover(full_gpu_cover, n_gpu_entries, true);"), None, "A1")
    # a second capability test creeps back in after the coverage test
    expect_fail("caps-test-duplicated", sub(mmvq_src,
                "                if (device_grouped_xmx_shape && !xmx_route_arrays_ok) {",
                "                if (!xmx_capabilities_match_int8_tile(caps, repeat, exec_n, k_per)) {\n"
                "                    log_xmx_reject(\"caps\");\n                    return false;\n                }\n"
                "                if (device_grouped_xmx_shape && !xmx_route_arrays_ok) {"), None, "M3a")
    expect_fail("slot-scan-unconditional", sub(mmvq_src,
                "                if (full_gpu_cover) {\n                    for (uint8_t seen : seen_slots) {",
                "                {\n                    for (uint8_t seen : seen_slots) {"), None, "A4")
    expect_fail("rows-asserted-against-total-batches", sub(mmvq_src,
                "GGML_ASSERT(grouped_rows_host.size() == static_cast<size_t>(n_gpu_entries) &&",
                "GGML_ASSERT(grouped_rows_host.size() == static_cast<size_t>(total_batches) &&"), None, "A5")
    expect_fail("rows-runtime-rejection-returns", sub(mmvq_src,
                "if (grouped_n_groups <= 0 || grouped_n_chunks <= 0) {",
                "if (grouped_n_groups <= 0 || grouped_n_chunks <= 0 ||\n                    grouped_rows_host.size() != static_cast<size_t>(n_gpu_entries)) {"), None, "A5b")
    expect_fail("sparse-batch-full-cover-only", sub(mmvq_src,
                "const bool sparse_xmx_batch = std::strcmp(",
                "const bool sparse_xmx_batch = full_gpu_cover && std::strcmp("), None, "A6")

    abort_after = ("        if (layout_reconciled && !moe_mmvq_mxfp4_direct_reads_layout(direct_effective_layout)) {")
    # I1(a): the very same abort block, textually moved in front of the reconcile guard.
    blk_start = backend_src.find(abort_after)
    blk_end = backend_src.find("\n        }\n", blk_start) + len("\n        }\n")
    rec_start = backend_src.find("        bool layout_reconciled = true;\n")
    if 0 <= rec_start < blk_start < blk_end:
        block = backend_src[blk_start:blk_end]
        moved = backend_src[:rec_start] + block + backend_src[rec_start:blk_start] + backend_src[blk_end:]
    else:
        moved = backend_src
    expect_fail("abort-before-reconcile", None, moved, "I1a")
    # I1(a'): the post-reconcile abort is simply gone.
    expect_fail("abort-removed", None, sub(backend_src, abort_after, "        if (false) {"), "I1a")
    # I1(a''): the abort judges only the forced layout, not the effective one.
    expect_fail("abort-judges-forced-only", None, sub(backend_src,
                "!moe_mmvq_mxfp4_direct_reads_layout(direct_effective_layout)) {\n            GGML_ABORT(",
                "forced_layout && !moe_mmvq_mxfp4_direct_reads_layout(*forced_layout)) {\n            GGML_ABORT("), "I1a")
    # I1(b): the non-forced branch (src0->extra) drops out of the effective layout.
    expect_fail("effective-layout-ignores-extra", None, sub(backend_src,
                "src0->extra   ? get_effective_layout_mode(static_cast<const ggml_tensor_extra_gpu *>(src0->extra)) :\n                            GGML_LAYOUT_AOS;",
                "GGML_LAYOUT_AOS;"), "I1b")
    # I1(d): the data_layout mapping re-derives from the forced layout instead of reading the single value.
    expect_fail("data-layout-rederives-from-forced", None, sub(backend_src,
                "switch (direct_effective_layout) {\n            case GGML_LAYOUT_SOA:",
                "switch (forced_layout ? *forced_layout : GGML_LAYOUT_AOS) {\n            case GGML_LAYOUT_SOA:"), "I1d")
    # I1(d'): ... or from the tensor's extra.
    expect_fail("data-layout-rederives-from-extra", None, sub(backend_src,
                "switch (direct_effective_layout) {\n            case GGML_LAYOUT_SOA:",
                "switch (get_effective_layout_mode(static_cast<const ggml_tensor_extra_gpu *>(src0->extra))) {\n            case GGML_LAYOUT_SOA:"), "I1d")
    # I1(c): the resolve_weight fallback maps a resolved unreadable layout onto AOS again.
    expect_fail("resolved-fallback-unguarded", None, sub(backend_src,
                "if (!moe_mmvq_mxfp4_direct_reads_layout(resolved.layout)) {", "if (false) {"), "I1c")

    try:
        check(mmvq_src, backend_src)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print("SELF-TEST PASS: 14 mutants caught, unmodified tree passes")
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
