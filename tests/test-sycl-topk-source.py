#!/usr/bin/env python3
"""SYCL GGML_OP_TOP_K runs on the device for every k, chosen by the loaded shape (llama.cpp-1b6u).

THE DEFECT.  ggml_sycl_device_supports_op_impl accepted TOP_K only for op->ne[0] <= SYCL_TOPK_MAX_K
(32), because the only kernel was the MoE-gating one (BLOCK x MAX_K candidates in SLM).  Qwen3.8's
sparse-attention indexer builds ggml_top_k(indexer_score_tokens, min(n_kv, indexer_top_k + r - 1)),
k in the hundreds to thousands, so each of its 12 attention layers ran its top-k on the CPU and cut
the SYCL graph 12 times per decode token.

THE CONTRACT.
  S. (topk.hpp) ONE pure selector, ggml_sycl_top_k_select_path(ncols, k, smpbo), decides the path from
     the shape alone: NONE when k < 1, k > ncols or ncols does not fit an int32 index; GATING for
     k <= ggml_sycl_top_k_gating_max_k; BITONIC when the padded row's keys and indices
     (ggml_sycl_top_k_bitonic_slm_bytes) fit smpbo; RADIX otherwise.  SYCL_TOPK_MAX_K, which the
     gating kernel sizes its arrays by, is that same constant (one fact, one source).
  P. (ggml-sycl.cpp supports_op) TOP_K is accepted iff src0 is a contiguous F32 row, dst is I32, and
     the selector returns a path other than NONE for (src0->ne[0], op->ne[0], devices[device].smpbo).
     No tensor-name test, no surviving `<= SYCL_TOPK_MAX_K` bound.
  D. (ggml-sycl.cpp ggml_sycl_op_top_k) dispatch asks the SAME selector with the SAME inputs
     (src0.ne(0), dst.ne(0), devices[ctx.device].smpbo) and switches on the answer: GATING ->
     topk_f32_i32_sycl, BITONIC and RADIX -> ggml_sycl_top_k_large, NONE -> GGML_ABORT.  No name test.
  M. (topk.cpp) the memory contract: scratch only through ggml_sycl_pool_alloc on ctx.pool() (the
     unified-cache SCRATCH zone, mem_handle-owned, released behind a marker event); no sycl::malloc_*,
     sycl::free, host wait (.wait / wait_and_throw); the bitonic SLM request is the selector's
     ggml_sycl_top_k_bitonic_slm_bytes, so the bound supports_op checked is the one dispatch uses; the
     radix work-group is one work-item per digit bin (static_assert block == bins) and is bounded by
     the device's max work-group size at launch, as the bitonic path bounds its own.
  O. (topk.cpp) the radix gather is an ordered compaction (exclusive_scan_over_group, no atomics), so
     the index set it returns is a pure function of the input: the k largest ordered keys, ties
     broken by the lower column -- the same set the bitonic path returns.  Both halves of that tie
     rule are pinned, because test-backend-ops scores tie rows as value multisets and passes any
     tie rule: the bitonic comparator ranks an equal key with the lower column first (O3), and the
     gather takes the `rank` equal-key columns lowest column first -- earlier ranges before later
     ones, lower lanes before higher ones within a round, and none past rank (O4).
  L. (topk.cpp) every kernel submit goes through ggml_sycl_profile_submit under a sycl.top_k.* label,
     and so does the k <= 32 gating kernel in ggml-sycl.cpp (sycl.top_k.gating), through the same
     label helper (ggml_sycl_top_k_profile_label, declared in topk.hpp).
  T. (test-backend-ops) the TOP_K cases cover k in {64, 256, 2048} x ncols in {256, 4096, 262144}
     plus a non-power-of-two ncols, with ties and with fewer than k finite values.

WHAT THIS DOES NOT PROVE.  It reads source text.  It does not show the kernels return the right
indices -- that is test-backend-ops -o TOP_K, run by the lead on the device.

NOT VACUOUS.  --self-test runs the gate on in-memory mutants of the real sources and requires each
to FAIL on its own pin, then requires the unmodified tree to pass.

THE RED, against 2b6da473e (the unmodified base):
  FAIL [pin S0]: ggml/src/ggml-sycl/topk.hpp is missing (llama.cpp-1b6u)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
HPP = SYCL / "topk.hpp"
CPP = SYCL / "topk.cpp"
BACKEND = SYCL / "ggml-sycl.cpp"
TBO = ROOT / "tests/test-backend-ops.cpp"
TAG = "(llama.cpp-1b6u)"


class ContractError(AssertionError):
    pass


def fail(pin: str, msg: str) -> None:
    raise ContractError(f"FAIL [pin {pin}]: {msg} {TAG}")


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
    return re.sub(r"\s*([(),;{}<>=!&|*+\-?:\[\]])\s*", r"\1", re.sub(r"\s+", " ", text))


def body_of(code: str, signature: str, pin: str) -> str:
    """The brace-balanced body that follows the first occurrence of `signature`."""
    a = code.find(signature)
    if a < 0:
        fail(pin, f"anchor not found: {signature!r}")
    b = code.find("{", a)
    depth = 0
    for i in range(b, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[b : i + 1]
    fail(pin, f"unbalanced body after {signature!r}")
    return ""


def case_body(code: str, case: str, fn_sig: str, pin: str) -> str:
    fn = body_of(code, fn_sig, pin)
    a = fn.find(case)
    if a < 0:
        fail(pin, f"{case} not found in {fn_sig}")
    b = re.search(r"\n\s*case GGML_OP_|\n\s*default:", fn[a + len(case) :])
    return fn[a : a + len(case) + (b.start() if b else len(fn))]


# ---------------------------------------------------------------------------------------------------
# S: the selector


def check_selector(hpp: str | None, backend: str) -> None:
    if hpp is None:
        fail("S0", "ggml/src/ggml-sycl/topk.hpp is missing")
    code = blank_comments(hpp)
    if not re.search(r"constexpr int ggml_sycl_top_k_gating_max_k = 32;", code):
        fail("S1", "topk.hpp does not define constexpr int ggml_sycl_top_k_gating_max_k = 32")
    slm = squash(body_of(code, "ggml_sycl_top_k_bitonic_slm_bytes(int64_t ncols)", "S2"))
    if "ncols_pad*(sizeof(uint32_t)+sizeof(int32_t))" not in slm or "while(ncols_pad<(size_t)ncols)" not in slm:
        fail("S2", "ggml_sycl_top_k_bitonic_slm_bytes is not next_pow2(ncols) * (sizeof(uint32_t) + sizeof(int32_t)) "
                   "-- the SLM the bitonic kernel holds is one key and one index per padded column")
    sel = squash(body_of(code, "ggml_sycl_top_k_select_path(int64_t ncols, int64_t k, size_t smpbo)", "S3"))
    steps = [
        ("S3", "if(k<1||k>ncols||ncols>INT32_MAX){return GGML_SYCL_TOP_K_PATH_NONE;}",
         "the selector must refuse k < 1, k > ncols (ggml_top_k's ne00 >= k) and ncols beyond an int32 index first"),
        ("S4", "if(k<=ggml_sycl_top_k_gating_max_k){return GGML_SYCL_TOP_K_PATH_GATING;}",
         "k <= ggml_sycl_top_k_gating_max_k must keep the MoE-gating kernel"),
        ("S5", "if(ggml_sycl_top_k_bitonic_slm_bytes(ncols)<=smpbo){return GGML_SYCL_TOP_K_PATH_BITONIC;}",
         "the bitonic path must be taken exactly when its SLM fits smpbo"),
        ("S6", "return GGML_SYCL_TOP_K_PATH_RADIX;",
         "everything else must go to the multi-workgroup radix select, not back to the CPU"),
    ]
    at = -1
    for pin, needle, why in steps:
        pos = sel.find(needle)
        if pos < 0 or pos < at:
            fail(pin, why)
        at = pos
    if re.search(r"name|strstr|strcmp", sel):
        fail("S7", "the selector reads a name; the loaded shape is the answer")
    bcode = blank_comments(backend)
    if not re.search(r"#define SYCL_TOPK_MAX_K\s+ggml_sycl_top_k_gating_max_k\b", bcode):
        fail("S8", "SYCL_TOPK_MAX_K is not ggml_sycl_top_k_gating_max_k: the gating kernel's bound and the selector's "
                   "would be two sources for one fact")


# ---------------------------------------------------------------------------------------------------
# P and D: supports_op and dispatch


SUPPORTS_SIG = ("static bool ggml_sycl_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op, "
                "bool placement_declines) {")
TOPK_OP_SIG = "inline void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst)"


def check_supports_op(backend: str) -> None:
    code = blank_comments(backend)
    case = squash(case_body(code, "case GGML_OP_TOP_K:", SUPPORTS_SIG, "P0"))
    want = ("return ggml_is_contiguous(op->src[0])&&op->src[0]->type==GGML_TYPE_F32&&op->type==GGML_TYPE_I32&&"
            "ggml_sycl_top_k_select_path(op->src[0]->ne[0],op->ne[0],ggml_sycl_info().devices[device].smpbo)!="
            "GGML_SYCL_TOP_K_PATH_NONE;")
    if want not in case:
        fail("P1", "supports_op TOP_K is not `contiguous F32 src0 && I32 dst && ggml_sycl_top_k_select_path("
                   "src0->ne[0], op->ne[0], devices[device].smpbo) != NONE`")
    if "SYCL_TOPK_MAX_K" in case:
        fail("P2", "supports_op TOP_K still bounds k by SYCL_TOPK_MAX_K; k > 32 would go back to the CPU")
    if re.search(r"->name|strstr|strcmp|_name\(", case):
        fail("P3", "supports_op TOP_K tests a tensor name; the loaded shape is the answer")


def check_dispatch(backend: str) -> None:
    code = blank_comments(backend)
    fn = squash(body_of(code, TOPK_OP_SIG, "D0"))
    want = ("ggml_sycl_top_k_select_path(ncols,k,ggml_sycl_info().devices[ctx.device].smpbo)")
    if want not in fn:
        fail("D1", "ggml_sycl_op_top_k does not ask the selector with (ncols, k, devices[ctx.device].smpbo)")
    if "const int64_t ncols=src0.ne(0);" not in fn or "const int64_t k=dst.ne(0);" not in fn:
        fail("D2", "ggml_sycl_op_top_k's ncols/k are not src0.ne(0)/dst.ne(0): dispatch and supports_op would ask "
                   "the selector about different shapes")
    if not re.search(r"case GGML_SYCL_TOP_K_PATH_GATING:topk_f32_i32_sycl\(", fn):
        fail("D3", "the GATING path does not dispatch topk_f32_i32_sycl")
    if not re.search(r"case GGML_SYCL_TOP_K_PATH_BITONIC:case GGML_SYCL_TOP_K_PATH_RADIX:ggml_sycl_top_k_large\("
                     r"ctx,src0_dd,dst_dd,ncols,nrows,k,path\);", fn):
        fail("D4", "the BITONIC and RADIX paths do not dispatch ggml_sycl_top_k_large(ctx, ..., path)")
    if not re.search(r"case GGML_SYCL_TOP_K_PATH_NONE:[^}]*GGML_ABORT\(", fn) and not re.search(
            r"default:[^}]*GGML_ABORT\(", fn):
        fail("D5", "a NONE path must abort -- supports_op declined it, so reaching dispatch is a defect")
    if re.search(r"\.name\(\)|strstr|strcmp", fn):
        fail("D6", "ggml_sycl_op_top_k tests a tensor name; the loaded shape is the answer")
    if fn.count("topk_f32_i32_sycl(") != 1:
        fail("D3", "topk_f32_i32_sycl is reached outside the GATING case")


# ---------------------------------------------------------------------------------------------------
# M, O, L: the kernels TU


FORBIDDEN_ALLOC = [r"\bmalloc_device\b", r"\bmalloc_host\b", r"\bmalloc_shared\b", r"sycl::free\b",
                   r"\baligned_alloc", r"\bunified_alloc\b", r"\bunified_allocate", r"\bnew\s+\w+\["]
LABELS = ["sycl.top_k.bitonic", "sycl.top_k.radix_histogram", "sycl.top_k.radix_select", "sycl.top_k.radix_count",
          "sycl.top_k.radix_gather"]


def check_kernels(cpp: str | None) -> None:
    if cpp is None:
        fail("M0", "ggml/src/ggml-sycl/topk.cpp is missing")
    code = blank_comments(cpp)
    for pat in FORBIDDEN_ALLOC:
        if re.search(pat, code):
            fail("M1", f"topk.cpp allocates outside the op pool ({pat}); scratch must come from ctx.pool()")
    if re.search(r"(\.|->)\s*wait\s*\(|wait_and_throw|get_profiling_info", code):
        fail("M2", "topk.cpp waits on the host; ordering is the in-order queue")
    large = squash(body_of(code, "void ggml_sycl_top_k_large(", "M3"))
    if not re.search(r"ggml_sycl_pool_alloc<uint32_t>scratch\(ctx\.pool\(\),", large):
        fail("M3", "the radix scratch is not a ggml_sycl_pool_alloc<uint32_t> on ctx.pool()")
    bit = squash(body_of(code, "static void top_k_bitonic_f32_i32_sycl(", "M4"))
    if "const size_t slm_bytes=ggml_sycl_top_k_bitonic_slm_bytes(ncols);" not in bit or \
            "GGML_ASSERT(slm_bytes<=ggml_sycl_info().devices[device].smpbo);" not in bit:
        fail("M4", "the bitonic kernel's SLM is not the selector's ggml_sycl_top_k_bitonic_slm_bytes(ncols), "
                   "asserted against smpbo")
    if not re.search(r"static_assert\(top_k_radix_block==top_k_radix_bins,", squash(code)):
        fail("M5", "nothing couples the radix work-group width to the digit bin count: the histogram and "
                   "select kernels index one bin per work-item")
    if "GGML_ASSERT(top_k_radix_block<=ggml_sycl_info().max_work_group_sizes[ctx.device]);" not in large:
        fail("M6", "the radix launch is not bounded by the device's max work-group size")
    gather = squash(body_of(code, "static void top_k_radix_gather(", "O1"))
    if "atomic" in gather:
        fail("O1", "the radix gather uses an atomic: the tie-broken index set would depend on scheduling")
    if gather.count("sycl::exclusive_scan_over_group(") < 2:
        fail("O2", "the radix gather is not an ordered compaction (exclusive_scan_over_group for both the "
                   "greater and the equal flags)")
    # The loads are pinned with the comparator: swapping only the index loads (ia from ixj, ib from col)
    # inverts the tiebreak to the higher column while the comparator text stays intact, and
    # test-backend-ops scores tie rows as multisets, so nothing else would notice.
    for needle in ["keys[col]=col<ncols?top_k_ordered_key(x_row[col]):0u;idxs[col]=col;",
                   "const uint32_t ka=keys[col];const uint32_t kb=keys[ixj];const int32_t ia=idxs[col];"
                   "const int32_t ib=idxs[ixj];",
                   "const bool b_first=kb>ka||(kb==ka&&ib<ia);",
                   "if(((col&kk)==0)==b_first){"]:
        if needle not in bit:
            fail("O3", f"the bitonic comparator does not rank an equal key with the lower column first "
                       f"(missing {needle!r}): a tie would pick a different set than the radix gather")
    for needle in ["const uint32_t n_gt=(uint32_t)k-rank;",
                   "for(int64_t b2=tid;b2<b;b2+=top_k_radix_block){prev_gt+=cnt[(row*bpr+b2)*2+0];"
                   "prev_eq+=cnt[(row*bpr+b2)*2+1];}",
                   "const int64_t c=c0+tid;",
                   "const uint32_t key=c<end?top_k_ordered_key(x_row[c]):0u;",
                   "const uint32_t pre_eq=sycl::exclusive_scan_over_group(it.get_group(),is_eq,"
                   "sycl::plus<uint32_t>());",
                   "if(is_eq&&base_eq+pre_eq<rank){dst_row[n_gt+base_eq+pre_eq]=(int32_t)c;}",
                   "base_eq+=sycl::reduce_over_group(it.get_group(),is_eq,sycl::plus<uint32_t>());"]:
        if needle not in gather:
            fail("O4", f"the radix gather does not take the first `rank` equal-key columns in column order "
                       f"(missing {needle!r}): a tie would pick a different set than the bitonic sort")
    if '#include "sycl-kernel-profiler.hpp"' not in cpp:
        fail("L0", "topk.cpp does not include the kernel profiler")
    if "ggml_sycl_profile_label ggml_sycl_top_k_profile_label(sycl::queue&q,const char*name," not in squash(code):
        fail("L0", "topk.cpp does not define the shared label helper ggml_sycl_top_k_profile_label")
    for label in LABELS:
        if f'"{label}"' not in code:
            fail("L1", f"kernel label {label} is missing")
    submits = len(re.findall(r"(\.|->)\s*submit\s*\(", code))
    profiled = len(re.findall(r"ggml_sycl_profile_submit\s*\(", code))
    if submits == 0 or profiled < submits:
        fail("L2", f"{submits} queue submits but {profiled} ggml_sycl_profile_submit wrappers: a top-k kernel "
                   f"runs unlabelled")


GATING_SIG = "static void topk_f32_i32_sycl(const float * x,"


def check_gating_label(hpp: str | None, backend: str) -> None:
    if hpp is None or "ggml_sycl_profile_label ggml_sycl_top_k_profile_label(" not in blank_comments(hpp):
        fail("L3", "topk.hpp does not declare ggml_sycl_top_k_profile_label for the gating kernel")
    gating = squash(body_of(blank_comments(backend), GATING_SIG, "L3"))
    if 'const ggml_sycl_profile_label label=ggml_sycl_top_k_profile_label(*stream,"sycl.top_k.gating",' \
            not in gating or "ggml_sycl_profile_submit(*stream,label,[&](sycl::queue&q){return q.submit(" \
            not in gating:
        fail("L3", "the k <= 32 gating kernel is not submitted through ggml_sycl_profile_submit under "
                   "sycl.top_k.gating")
    if gating.count("submit(") != gating.count("ggml_sycl_profile_submit(") * 2:
        fail("L3", "the gating kernel has a queue submit outside ggml_sycl_profile_submit")


# ---------------------------------------------------------------------------------------------------
# T: test-backend-ops shapes


def check_tbo(tbo: str) -> None:
    a = tbo.find("llama.cpp-1b6u")
    if a < 0:
        fail("T0", "test-backend-ops has no llama.cpp-1b6u TOP_K block")
    blk = squash(tbo[a : a + 3000])
    for needle, why in [
        ("for(int k:{64,256,2048})", "k in {64, 256, 2048}"),
        ("for(int64_t ncols:{256,4096,262144,262144+11})", "ncols in {256, 4096, 262144} and a non-power-of-two"),
        ("if(k>ncols){continue;}", "skip k > ncols, which ggml_top_k forbids"),
        ("new test_top_k(GGML_TYPE_F32,{ncols,1,1,1},k)", "the decode shape (one row)"),
        ("new test_top_k(GGML_TYPE_F32,{ncols,3,1,1},k,true)", "ties, several rows"),
        ("new test_top_k(GGML_TYPE_F32,{262144,1,1,1},2051,false,1000)", "the radix path with fewer finite than k"),
    ]:
        if needle not in blk:
            fail("T1", f"the 1b6u TOP_K block misses {why}")


# ---------------------------------------------------------------------------------------------------


def read_sources() -> dict[str, str | None]:
    return {
        "hpp": HPP.read_text() if HPP.exists() else None,
        "cpp": CPP.read_text() if CPP.exists() else None,
        "backend": BACKEND.read_text(),
        "tbo": TBO.read_text(),
    }


def check(s: dict[str, str | None]) -> None:
    check_selector(s["hpp"], s["backend"])
    check_supports_op(s["backend"])
    check_dispatch(s["backend"])
    check_kernels(s["cpp"])
    check_gating_label(s["hpp"], s["backend"])
    check_tbo(s["tbo"])


# pytest entry points (one per contract letter) ------------------------------------------------------


def test_selector() -> None:
    s = read_sources()
    check_selector(s["hpp"], s["backend"])


def test_supports_op() -> None:
    check_supports_op(read_sources()["backend"])


def test_dispatch() -> None:
    check_dispatch(read_sources()["backend"])


def test_kernels() -> None:
    s = read_sources()
    check_kernels(s["cpp"])
    check_gating_label(s["hpp"], s["backend"])


def test_backend_ops_shapes() -> None:
    check_tbo(read_sources()["tbo"])


def test_self_test() -> None:
    assert self_test(read_sources()) == 0


# --self-test ---------------------------------------------------------------------------------------


def self_test(s: dict[str, str | None]) -> int:
    failures: list[str] = []
    caught = 0

    def expect_fail(name: str, key: str, old: str, new: str, pin: str) -> None:
        nonlocal caught
        src = s[key]
        if src is None or old not in src:
            failures.append(f"{name}: anchor not found (stale mutant)")
            print(f"  mutant {name}: NOT APPLIED")
            return
        m = dict(s)
        m[key] = src.replace(old, new, 1)
        try:
            check(m)
        except ContractError as e:
            if f"[pin {pin}]" not in str(e):
                failures.append(f"{name}: failed on the wrong pin: {str(e)[:140]}")
                print(f"  mutant {name}: caught by the WRONG pin ({str(e)[:70]}...)")
                return
            caught += 1
            print(f"  mutant {name}: caught (pin {pin})")
            return
        failures.append(f"{name}: survived")
        print(f"  mutant {name}: SURVIVED")

    # S
    expect_fail("gating-max-literal", "hpp", "constexpr int ggml_sycl_top_k_gating_max_k = 32;",
                "constexpr int ggml_sycl_top_k_gating_max_k = 64;", "S1")
    expect_fail("slm-ints-only", "hpp", "ncols_pad * (sizeof(uint32_t) + sizeof(int32_t))",
                "ncols_pad * sizeof(int32_t)", "S2")
    expect_fail("selector-allows-k-above-ncols", "hpp", "k < 1 || k > ncols || ncols > INT32_MAX",
                "k < 1 || ncols > INT32_MAX", "S3")
    expect_fail("selector-drops-gating", "hpp", "if (k <= ggml_sycl_top_k_gating_max_k) {",
                "if (false) {", "S4")
    expect_fail("selector-bitonic-ignores-slm", "hpp", "if (ggml_sycl_top_k_bitonic_slm_bytes(ncols) <= smpbo) {",
                "if (ncols <= 16384) {", "S5")
    expect_fail("selector-falls-back-to-cpu", "hpp", "return GGML_SYCL_TOP_K_PATH_RADIX;",
                "return GGML_SYCL_TOP_K_PATH_NONE;", "S6")
    expect_fail("gating-bound-literal", "backend", "#define SYCL_TOPK_MAX_K      ggml_sycl_top_k_gating_max_k",
                "#define SYCL_TOPK_MAX_K      32", "S8")
    # P
    expect_fail("supports-op-old-bound", "backend",
                "ggml_sycl_top_k_select_path(op->src[0]->ne[0], op->ne[0], ggml_sycl_info().devices[device].smpbo) !=\n"
                "                       GGML_SYCL_TOP_K_PATH_NONE;",
                "op->ne[0] <= SYCL_TOPK_MAX_K;", "P1")
    expect_fail("supports-op-wrong-smpbo-device", "backend",
                "ggml_sycl_top_k_select_path(op->src[0]->ne[0], op->ne[0], ggml_sycl_info().devices[device].smpbo)",
                "ggml_sycl_top_k_select_path(op->src[0]->ne[0], op->ne[0], ggml_sycl_info().devices[0].smpbo)", "P1")
    expect_fail("supports-op-name-test", "backend",
                "        case GGML_OP_TOP_K:\n            // The selector",
                "        case GGML_OP_TOP_K:\n"
                "            if (op->name[0] == 'x') { return false; }\n"
                "            // The selector",
                "P3")
    # D
    expect_fail("dispatch-without-selector", "backend",
                "ggml_sycl_top_k_select_path(ncols, k, ggml_sycl_info().devices[ctx.device].smpbo)",
                "(k <= SYCL_TOPK_MAX_K ? GGML_SYCL_TOP_K_PATH_GATING : GGML_SYCL_TOP_K_PATH_RADIX)", "D1")
    expect_fail("dispatch-k-from-src", "backend", "const int64_t k       = dst.ne(0);",
                "const int64_t k       = src0.ne(1);", "D2")
    expect_fail("dispatch-radix-to-gating", "backend",
                "        case GGML_SYCL_TOP_K_PATH_BITONIC:\n        case GGML_SYCL_TOP_K_PATH_RADIX:\n"
                "            ggml_sycl_top_k_large(",
                "        case GGML_SYCL_TOP_K_PATH_BITONIC:\n        case GGML_SYCL_TOP_K_PATH_RADIX:\n"
                "            topk_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, k, main_stream);\n"
                "            break;\n            ggml_sycl_top_k_large(", "D4")
    expect_fail("gating-reached-outside-its-case", "backend",
                "        case GGML_SYCL_TOP_K_PATH_NONE:\n        default:\n",
                "        case GGML_SYCL_TOP_K_PATH_NONE:\n        default:\n"
                "            topk_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, k, main_stream);\n", "D3")
    expect_fail("dispatch-none-silent", "backend",
                "            GGML_ABORT(\"[TOP_K] no SYCL path",
                "            GGML_LOG_WARN(\"[TOP_K] no SYCL path", "D5")
    # M, O, L
    expect_fail("kernels-raw-malloc", "cpp", "ggml_sycl_pool_alloc<uint32_t> scratch(ctx.pool(), scratch_elems);",
                "uint32_t * scratch_ptr = sycl::malloc_device<uint32_t>(scratch_elems, *stream);\n"
                "    ggml_sycl_pool_alloc<uint32_t> scratch(ctx.pool(), scratch_elems);", "M1")
    expect_fail("kernels-host-wait", "cpp", "    top_k_radix_gather(",
                "    stream->wait();\n    top_k_radix_gather(", "M2")
    expect_fail("bitonic-slm-own-size", "cpp",
                "const size_t slm_bytes = ggml_sycl_top_k_bitonic_slm_bytes(ncols);",
                "const size_t slm_bytes = (size_t) ncols * 8;", "M4")
    expect_fail("gather-atomic", "cpp", "const uint32_t pre_gt =",
                "sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> a(cnt[0]);\n"
                "                const uint32_t pre_gt =", "O1")
    expect_fail("bitonic-tie-to-higher-column", "cpp", "(kb == ka && ib < ia)", "(kb == ka && ib > ia)", "O3")
    expect_fail("bitonic-index-loads-swapped", "cpp",
                "const int32_t  ia      = idxs[col];\n"
                "                                const int32_t  ib      = idxs[ixj];",
                "const int32_t  ia      = idxs[ixj];\n"
                "                                const int32_t  ib      = idxs[col];",
                "O3")
    expect_fail("bitonic-no-index-tiebreak", "cpp", "kb > ka || (kb == ka && ib < ia)", "kb > ka", "O3")
    expect_fail("gather-eq-unbounded", "cpp", "if (is_eq && base_eq + pre_eq < rank) {", "if (is_eq) {", "O4")
    expect_fail("gather-eq-later-ranges-first", "cpp", "for (int64_t b2 = tid; b2 < b; b2 += top_k_radix_block) {",
                "for (int64_t b2 = b + 1 + tid; b2 < bpr; b2 += top_k_radix_block) {", "O4")
    expect_fail("gather-eq-higher-lanes-first", "cpp", "const int64_t  c     = c0 + tid;",
                "const int64_t  c     = c0 + (top_k_radix_block - 1 - tid);", "O4")
    expect_fail("radix-block-unbound-to-bins", "cpp",
                "static_assert(top_k_radix_block == top_k_radix_bins,", "static_assert(true,", "M5")
    expect_fail("radix-launch-unbounded", "cpp",
                "GGML_ASSERT(top_k_radix_block <= ggml_sycl_info().max_work_group_sizes[ctx.device]);", "", "M6")
    expect_fail("gating-unlabelled", "backend",
                "ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {\n        return q.submit(",
                "stream->submit([&](sycl::handler & h0) {});\n    ggml_sycl_profile_submit(*stream, label, "
                "[&](sycl::queue & q) {\n        return q.submit(", "L3")
    expect_fail("label-missing", "cpp", '"sycl.top_k.radix_count"', '"sycl.top_k.count"', "L1")
    expect_fail("unlabelled-submit", "cpp", "    top_k_radix_gather(",
                "    stream->submit([&](sycl::handler & cgh) { cgh.single_task([=]() {}); });\n    top_k_radix_gather(",
                "L2")
    # T
    expect_fail("tbo-no-radix-shape", "tbo", "for (int64_t ncols : { 256, 4096, 262144, 262144 + 11 })",
                "for (int64_t ncols : { 256, 4096 })", "T1")

    try:
        check(s)
        print("  self-test: unmodified tree passes")
    except ContractError as e:
        print(f"  self-test: unmodified tree FAILS: {e}")
        failures.append("unmodified")

    if failures:
        print(f"SELF-TEST FAIL: {', '.join(failures)}")
        return 1
    print(f"SELF-TEST PASS: {caught} mutants caught, unmodified tree passes")
    return 0


def main(argv: list[str]) -> int:
    s = read_sources()
    if "--self-test" in argv:
        rc = self_test(s)
        if rc != 0:
            return rc
    try:
        check(s)
    except ContractError as e:
        print(e)
        return 1
    print("PASS: TOP_K is accepted for every shape a SYCL kernel implements, and dispatch selects the kernel "
          "from that same shape rule " + TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
