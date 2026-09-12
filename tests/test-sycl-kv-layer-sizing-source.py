#!/usr/bin/env python3
"""Pin the shape of the SYCL KV planner's per-layer sizing fix (llama.cpp-3aos).

Two bugs, one root, described in the ticket:

  Bug 1: placement_kv_info::kv_bytes_per_swa_layer() hardcoded n_seq_max=1
  in its SWA cell-count formula, while llama_kv_cache_iswa allocates
  GGML_PAD(min(kv_size, n_swa * n_seq_max + n_ubatch), 256) cells -- any SWA
  model served with `--parallel > 1` was under-budgeted.

  Bug 2: a single global (n_embd_k_gqa, n_embd_v_gqa) width pair and one
  SWA/non-SWA mask cannot represent a heterogeneous model (Gemma 4 E4B: full
  layers wider than SWA layers, and a trailing block with no K/V of their
  own at all).

The fix threads a shared per-layer formula (kv_layer_bytes_for_kind(), a
three-way FULL/SWA/SHARED switch that consumes n_seq_max) through both the
planner (placement_kv_info::kv_bytes_for_layer()) and the persisted plan
(placement_plan::kv_size_for_layer()), and the runtime transaction body
(ggml_backend_sycl_set_runtime_context_for_model's shared body) threads its
caller's n_seq_max into next_kv_info before the KV budget is recomputed from
it.

This is a regex/structural check on the SOURCE, not a build+run test --
see test-sycl-kv-layer-sizing.cpp for the numeric RED/GREEN behavioral
check. This file exists because the numeric test can pass for the wrong
reason (e.g. a hand test that never exercises the real n_seq_max plumbing);
pinning the shape independently closes that gap.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
UNIFIED_CACHE_HPP = ROOT / "ggml/src/ggml-sycl/unified-cache.hpp"
UNIFIED_CACHE_CPP = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"
GGML_SYCL_CPP = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"

KV_LAYER_BYTES_FOR_KIND = "kv_layer_bytes_for_kind"
# The actual shared transaction body -- ggml_backend_sycl_set_runtime_context_
# for_model() (the public entry point) and the probe path both call into this
# one function; it is where next_kv_info/next_plan are built and consumed.
TRANSACTION_SIGNATURE = "static ggml_sycl_txn_result ggml_sycl_run_runtime_context_transaction"


def function(text: str, signature: str) -> str:
    """Extract one brace-delimited function body starting at `signature`.

    Copied from test-sycl-buffer-base-alignment-source.py's helper of the
    same name -- brace/string/comment-aware so a `{`/`}` inside a string
    literal or comment does not end the scan early.
    """
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    state = "code"
    i = brace
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 1
            elif ch == "/" and nxt == "*":
                state = "block"
                i += 1
            elif ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        elif state == "line":
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 1
        else:
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                i += 1
            elif ch == quote:
                state = "code"
        i += 1
    raise AssertionError(f"unclosed function {signature}")


def function_or_none(text: str, signature: str) -> str | None:
    if signature not in text:
        return None
    return function(text, signature)


def kv_layer_bytes_for_kind_violations(source: str) -> list[str]:
    body = function_or_none(source, KV_LAYER_BYTES_FOR_KIND)
    if body is None:
        return [f"{KV_LAYER_BYTES_FOR_KIND} is missing"]

    found: list[str] = []

    # Bug 1: n_seq_max must actually be consumed by the SWA cell formula --
    # not just accepted as a parameter and ignored. A caller that always
    # forces seqs=1 (the pre-fix behavior) would still compile against this
    # signature, so the check is on the formula itself, not merely the
    # parameter list.
    if not re.search(r"n_seq_max\s*>\s*0\s*\?\s*n_seq_max\s*:\s*1", body):
        found.append(f"{KV_LAYER_BYTES_FOR_KIND} does not derive seqs from n_seq_max (with a >0 guard)")
    if not re.search(r"n_swa\s*\*\s*seqs\s*\+\s*n_ubatch", body):
        found.append(f"{KV_LAYER_BYTES_FOR_KIND} does not multiply n_swa by seqs in the window formula")

    # Bug 2: the three-way switch, SHARED charged 0.
    if "GGML_SYCL_KV_LAYER_SHARED" not in body:
        found.append(f"{KV_LAYER_BYTES_FOR_KIND} does not branch on GGML_SYCL_KV_LAYER_SHARED")
    if not re.search(r"kind\s*==\s*GGML_SYCL_KV_LAYER_SHARED\s*\)\s*\{\s*return\s+0\s*;", body):
        found.append(f"{KV_LAYER_BYTES_FOR_KIND} does not return 0 for a SHARED layer")
    if "GGML_SYCL_KV_LAYER_SWA" not in body:
        found.append(f"{KV_LAYER_BYTES_FOR_KIND} does not branch on GGML_SYCL_KV_LAYER_SWA")

    return found


def kv_bytes_for_layer_violations(source: str) -> list[str]:
    """placement_kv_info::kv_bytes_for_layer() must route through the shared
    per-layer formula when per-layer truth is populated, not re-derive its
    own copy -- the exact "two independent formulas can drift" shape this
    ticket fixes.
    """
    body = function_or_none(source, "size_t kv_bytes_for_layer(uint32_t il) const")
    if body is None:
        return ["placement_kv_info::kv_bytes_for_layer is missing"]
    found: list[str] = []
    # A real CALL, not merely a comment mentioning the function's name --
    # `kv_layer_bytes_for_kind()` also appears in this function's own prose.
    if not re.search(re.escape(KV_LAYER_BYTES_FOR_KIND) + r"\(\s*layer_kind\[", body):
        found.append("kv_bytes_for_layer does not call kv_layer_bytes_for_kind with the per-layer arrays")
    if "has_per_layer_kv_truth" not in body:
        found.append("kv_bytes_for_layer does not check has_per_layer_kv_truth before using per-layer arrays")
    return found


def kv_size_for_layer_violations(source: str) -> list[str]:
    """placement_plan::kv_size_for_layer() -- the function the runtime
    transaction body's refresh_kv_byte_totals()/rebuild_runtime_per_device_
    vram() actually query -- must ALSO route through kv_layer_bytes_for_kind,
    not just placement_kv_info's copy. Two call sites computing the same
    per-layer byte count from two different formulas is exactly how the
    planner and the allocator disagreed before this ticket.
    """
    body = function_or_none(source, "size_t kv_size_for_layer(uint32_t layer_id) const")
    if body is None:
        return ["placement_plan::kv_size_for_layer is missing"]
    found: list[str] = []
    # A real CALL, not merely a comment mentioning the function's name --
    # `kv_layer_bytes_for_kind()` also appears in this function's own prose.
    if not re.search(re.escape(KV_LAYER_BYTES_FOR_KIND) + r"\(\s*layer_kind\[", body):
        found.append("kv_size_for_layer does not call kv_layer_bytes_for_kind with the per-layer arrays")
    if "layer_kind" not in body or "layer_k_width" not in body or "layer_v_width" not in body:
        found.append("kv_size_for_layer does not consult the plan's per-layer kind/width arrays")
    return found


def transaction_body_violations(source: str) -> list[str]:
    """The shared runtime-context transaction body must thread the caller's
    n_seq_max into next_kv_info BEFORE the KV budget is (re)computed from
    it -- setting it only afterward (as this file's own git history shows
    for planner_n_ctx/n_ubatch/n_seq_max on next_plan) silently sizes every
    layer's zone from the STALE previous n_seq_max.
    """
    body = function_or_none(source, TRANSACTION_SIGNATURE)
    if body is None:
        return [f"{TRANSACTION_SIGNATURE} is missing"]

    # Whitespace-tolerant: clang-format column-aligns adjacent `=` signs, so
    # the exact spacing here drifts with whatever sibling assignment lines
    # sit next to it.
    kv_info_re = re.compile(r"next_kv_info\.n_seq_max\s*=\s*n_seq_max;")
    plan_re    = re.compile(r"next_plan\.planner_n_seq_max\s*=\s*n_seq_max;")

    found: list[str] = []
    kv_info_match = kv_info_re.search(body)
    if kv_info_match is None:
        found.append("the transaction body never sets next_kv_info.n_seq_max")
    plan_match = plan_re.search(body)
    if plan_match is None:
        found.append("the transaction body never sets next_plan.planner_n_seq_max")

    # Ordering: both next_kv_info.n_seq_max and next_plan.planner_n_seq_max
    # must be set before next_plan.update_runtime_kv_sizes() -- that call's
    # refresh_kv_byte_totals() walks kv_size_for_layer() per layer, which
    # reads next_plan.planner_n_seq_max/n_ubatch/n_ctx directly, and the call
    # also computes next_kv_info.kv_bytes_per_layer()/kv_bytes_per_swa_layer()
    # as its own arguments. Setting either only afterward (as this file's own
    # git history shows for planner_n_ctx/n_ubatch/n_seq_max, which used to
    # be set only after the per-device budget accounting that follows this
    # call) sizes every layer's zone from the STALE previous n_seq_max.
    # Anchored to the qualified CALL (`next_plan.foo(`), not a bare mention --
    # this file's own comments name the function in prose too.
    kv_info_pos = kv_info_match.start() if kv_info_match else -1
    plan_pos    = plan_match.start() if plan_match else -1
    update_pos  = body.find("next_plan.update_runtime_kv_sizes(")
    if kv_info_pos == -1 or update_pos == -1 or kv_info_pos > update_pos:
        found.append("next_kv_info.n_seq_max is not set before next_plan.update_runtime_kv_sizes() consumes it")
    if plan_pos == -1 or update_pos == -1 or plan_pos > update_pos:
        found.append("next_plan.planner_n_seq_max is not set before next_plan.update_runtime_kv_sizes() consumes it")

    return found


def stale_comment_violations(hpp_source: str, cpp_source: str) -> list[str]:
    found: list[str] = []
    if "with n_seq_max=1" in hpp_source:
        found.append("unified-cache.hpp still carries the stale 'with n_seq_max=1' comment")
    if "n_seq_max=1-only" in cpp_source:
        found.append("unified-cache.cpp still carries the stale 'n_seq_max=1-only' deferral rationale")
    return found


def test_kv_layer_bytes_for_kind_consumes_n_seq_max_and_the_three_kinds() -> None:
    assert kv_layer_bytes_for_kind_violations(UNIFIED_CACHE_HPP.read_text()) == []


def test_kv_bytes_for_layer_shares_the_formula() -> None:
    assert kv_bytes_for_layer_violations(UNIFIED_CACHE_HPP.read_text()) == []


def test_kv_size_for_layer_shares_the_formula() -> None:
    assert kv_size_for_layer_violations(UNIFIED_CACHE_HPP.read_text()) == []


def test_transaction_body_threads_n_seq_max_before_use() -> None:
    assert transaction_body_violations(GGML_SYCL_CPP.read_text()) == []


def test_no_stale_n_seq_max_one_comment_remains() -> None:
    assert stale_comment_violations(UNIFIED_CACHE_HPP.read_text(), UNIFIED_CACHE_CPP.read_text()) == []


def test_mutations_are_witnessed() -> None:
    """Every check above must actually fire on the defect it claims to catch."""
    hpp = UNIFIED_CACHE_HPP.read_text()
    hpp_mutations = [
        # Bug 1 reinstated: hardcode seqs=1, dropping the n_seq_max> 0 guard.
        hpp.replace("const uint32_t seqs      = n_seq_max > 0 ? n_seq_max : 1;",
                    "const uint32_t seqs      = 1;", 1),
        # Bug 1 reinstated a different way: seqs computed but not multiplied in.
        hpp.replace("n_swa * seqs + n_ubatch", "n_swa + n_ubatch", 1),
        # Bug 2 reinstated: drop the SHARED arm entirely.
        re.sub(r"if \(kind == GGML_SYCL_KV_LAYER_SHARED\) \{\s*\n\s*return 0;\s*\n\s*\}\s*\n", "", hpp, count=1),
        # kv_bytes_for_layer stops routing through the shared formula.
        hpp.replace(
            "return kv_layer_bytes_for_kind(layer_kind[il], layer_k_width[il], layer_v_width[il], n_ctx, n_swa, n_ubatch,\n"
            "                                           n_seq_max);",
            "return 0;", 1),
        # kv_size_for_layer stops routing through the shared formula.
        hpp.replace(
            "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
            "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max);",
            "return 0;", 1),
        # The stale comment reappears.
        hpp.replace(
            "// n_seq_max scales the window -- see the field comment above; it\n",
            "// with n_seq_max=1. n_seq_max scales the window -- see the field comment above; it\n", 1),
    ]
    for index, mutated in enumerate(hpp_mutations):
        assert mutated != hpp, f"unified-cache.hpp mutation {index} did not change the source"
        violated = (kv_layer_bytes_for_kind_violations(mutated) or kv_bytes_for_layer_violations(mutated) or
                    kv_size_for_layer_violations(mutated) or stale_comment_violations(mutated, ""))
        assert violated, f"unified-cache.hpp mutation {index} was not witnessed"

    cpp = GGML_SYCL_CPP.read_text()
    cpp_mutations = [
        # The line the ticket names explicitly as the fix. Whitespace-tolerant
        # (clang-format column-aligns the `=`) via re.sub rather than a
        # literal .replace().
        re.sub(r"[ \t]*next_kv_info\.n_seq_max\s*=\s*n_seq_max;\n", "", cpp, count=1),
        # planner_n_seq_max still set, but only after the calls that need it.
        cpp.replace(
            "    next_plan.planner_n_ctx     = n_ctx;\n"
            "    next_plan.planner_n_ubatch  = next_kv_info.n_ubatch;\n"
            "    next_plan.planner_n_seq_max = n_seq_max;\n"
            "    next_plan.update_runtime_kv_sizes(n_ctx, next_kv_info.kv_bytes_per_layer(), next_kv_info.kv_bytes_per_swa_layer());\n",
            "    next_plan.update_runtime_kv_sizes(n_ctx, next_kv_info.kv_bytes_per_layer(), next_kv_info.kv_bytes_per_swa_layer());\n"
            "    next_plan.planner_n_ctx     = n_ctx;\n"
            "    next_plan.planner_n_ubatch  = next_kv_info.n_ubatch;\n"
            "    next_plan.planner_n_seq_max = n_seq_max;\n", 1),
    ]
    for index, mutated in enumerate(cpp_mutations):
        assert mutated != cpp, f"ggml-sycl.cpp mutation {index} did not change the source"
        assert transaction_body_violations(mutated), f"ggml-sycl.cpp mutation {index} was not witnessed"
