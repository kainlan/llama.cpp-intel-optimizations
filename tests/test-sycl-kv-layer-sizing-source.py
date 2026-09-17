#!/usr/bin/env python3
"""Pin the shape of the SYCL KV planner's per-layer sizing fix (llama.cpp-3aos).

Three bugs, one root, described in the ticket:

  Bug 1: placement_kv_info::kv_bytes_per_swa_layer() hardcoded n_seq_max=1
  in its SWA cell-count formula. Any SWA model served with `--parallel > 1`
  was under-budgeted.

  Bug 2: a single global (n_embd_k_gqa, n_embd_v_gqa) width pair and one
  SWA/non-SWA mask cannot represent a heterogeneous model (Gemma 4 E4B: full
  layers wider than SWA layers, and a trailing block with no K/V of their
  own at all) -- both in the per-layer inventory (ggml-sycl.h/llama-model.cpp)
  AND in the aggregate fallback width src/llama-model.cpp assigns
  (n_embd_k_gqa_max()/n_embd_v_gqa_max(), not layer 0's width).

  Bug 3 (found on a live GPU acceptance run after Bugs 1-2 were fixed):
  llama_context splits its KV cache into n_seq_max independent STREAMS
  unless kv_unified==true (the default for llama-completion/llama-bench is
  kv_unified==false); each stream's SWA window is then capped independently
  (n_swa + n_ubatch, not n_swa * n_seq_max + n_ubatch), and there are
  n_seq_max such streams. The Bug-1 fix alone assumed the OPPOSITE (a single
  shared stream whose window scales with n_seq_max) -- correct only when
  kv_unified==true.

The fix threads a shared per-layer formula (kv_layer_bytes_for_kind(), a
three-way FULL/SWA/SHARED switch that consumes n_seq_max AND kv_unified)
through both the planner (placement_kv_info::kv_bytes_for_layer()) and the
persisted plan (placement_plan::kv_size_for_layer()), and the runtime
transaction body (ggml_sycl_run_runtime_context_transaction) threads its
caller's n_seq_max and kv_unified into next_kv_info/next_plan before the KV
budget is recomputed from them.

This is a regex/structural check on the SOURCE, not a build+run test --
see test-sycl-kv-layer-sizing.cpp for the numeric RED/GREEN behavioral
check. This file exists because the numeric test can pass for the wrong
reason (e.g. a hand test that never exercises the real n_seq_max/kv_unified
plumbing); pinning the shape independently closes that gap.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
UNIFIED_CACHE_HPP = ROOT / "ggml/src/ggml-sycl/unified-cache.hpp"
UNIFIED_CACHE_CPP = ROOT / "ggml/src/ggml-sycl/unified-cache.cpp"
GGML_SYCL_CPP = ROOT / "ggml/src/ggml-sycl/ggml-sycl.cpp"
LLAMA_MODEL_CPP = ROOT / "src/llama-model.cpp"

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
    body = function_or_none(source, "inline size_t " + KV_LAYER_BYTES_FOR_KIND)
    if body is None:
        return [f"{KV_LAYER_BYTES_FOR_KIND} is missing"]

    found: list[str] = []

    # Bug 1: n_seq_max must actually be consumed by the SWA cell formula --
    # not just accepted as a parameter and ignored.
    if not re.search(r"seqs\s*=\s*n_seq_max\s*>\s*0\s*\?\s*n_seq_max\s*:\s*1", body):
        found.append("does not derive seqs from n_seq_max (with a >0 guard)")

    # Bug 2: the three-way switch, SHARED charged 0.
    if not re.search(r"kind\s*==\s*GGML_SYCL_KV_LAYER_SHARED\s*\)\s*\{\s*return\s+0\s*;", body):
        found.append("does not return 0 for a SHARED layer")
    if "GGML_SYCL_KV_LAYER_SWA" not in body:
        found.append("does not branch on GGML_SYCL_KV_LAYER_SWA")

    # Bug 3: the kv_unified two-mode split.
    if not re.search(r"bool\s+kv_unified\b",
                     function_signature_params(source, "inline size_t " + KV_LAYER_BYTES_FOR_KIND)):
        found.append("has no bool kv_unified parameter")
    if not re.search(r"if\s*\(\s*kv_unified\s*\)\s*\{", body):
        found.append("does not branch on kv_unified")
    # Unified mode: single stream, window scales with n_seq_max.
    if not re.search(r"n_ctx_seq\s*=\s*n_ctx\s*;\s*\n\s*n_stream\s*=\s*1\s*;\s*\n\s*window_seqs\s*=\s*seqs\s*;", body):
        found.append("kv_unified==true branch does not set n_ctx_seq=n_ctx, n_stream=1, window_seqs=seqs")
    # Non-unified mode: n_seq_max independent streams, each windowed at 1 sequence.
    if not re.search(r"n_ctx_seq\s*=\s*GGML_PAD\(n_ctx\s*/\s*seqs,\s*256\)", body):
        found.append("kv_unified==false branch does not derive n_ctx_seq = GGML_PAD(n_ctx / seqs, 256)")
    if not re.search(r"n_stream\s*=\s*seqs\s*;\s*\n\s*window_seqs\s*=\s*1\s*;", body):
        found.append("kv_unified==false branch does not set n_stream=seqs, window_seqs=1")
    # The per-stream cap and the stream multiplication must both survive
    # into the final formula, in EITHER mode.
    if not re.search(r"n_swa\s*\*\s*window_seqs\s*\+\s*n_ubatch", body):
        found.append("does not multiply n_swa by window_seqs in the per-stream window formula")
    if not re.search(r"swa_cells\s*=\s*swa_cells_per_stream\s*\*\s*n_stream", body):
        found.append("does not multiply the per-stream cell count by n_stream")

    return found


def function_signature_params(source: str, signature: str) -> str:
    """The parameter list text between a function signature and its opening
    brace -- used only to check a parameter TYPE/NAME exists, not to parse
    the body."""
    start = source.index(signature)
    brace = source.index("{", start)
    return source[start:brace]


def kv_bytes_for_layer_violations(source: str) -> list[str]:
    """placement_kv_info::kv_bytes_for_layer() must route through the shared
    per-layer formula when per-layer truth is populated, not re-derive its
    own copy -- the exact "two independent formulas can drift" shape this
    ticket fixes. Must also forward kv_unified, not just n_seq_max.
    """
    body = function_or_none(source, "size_t kv_bytes_for_layer(uint32_t il) const")
    if body is None:
        return ["placement_kv_info::kv_bytes_for_layer is missing"]
    found: list[str] = []
    # A real CALL, not merely a comment mentioning the function's name --
    # `kv_layer_bytes_for_kind()` also appears in this function's own prose.
    call = re.search(re.escape(KV_LAYER_BYTES_FOR_KIND) + r"\(\s*layer_kind\[il\][^;]*\)", body, re.S)
    if call is None:
        found.append("does not call kv_layer_bytes_for_kind with the per-layer arrays")
    elif "kv_unified" not in call.group(0):
        found.append("calls kv_layer_bytes_for_kind without forwarding kv_unified")
    if "has_per_layer_kv_truth" not in body:
        found.append("does not check has_per_layer_kv_truth before using per-layer arrays")
    return found


def kv_size_for_layer_violations(source: str) -> list[str]:
    """placement_plan::kv_size_for_layer() -- the function the runtime
    transaction body's refresh_kv_byte_totals()/rebuild_runtime_per_device_
    vram() actually query -- must ALSO route through kv_layer_bytes_for_kind
    (forwarding planner_kv_unified), not just placement_kv_info's copy. Two
    call sites computing the same per-layer byte count from two different
    formulas -- or the same formula fed a stale kv_unified -- is exactly how
    the planner and the allocator disagreed before this ticket.
    """
    body = function_or_none(source, "size_t kv_size_for_layer(uint32_t layer_id) const")
    if body is None:
        return ["placement_plan::kv_size_for_layer is missing"]
    found: list[str] = []
    call = re.search(re.escape(KV_LAYER_BYTES_FOR_KIND) + r"\(\s*layer_kind\[layer_id\][^;]*\)", body, re.S)
    if call is None:
        found.append("does not call kv_layer_bytes_for_kind with the per-layer arrays")
    elif "planner_kv_unified" not in call.group(0):
        found.append("calls kv_layer_bytes_for_kind without forwarding planner_kv_unified")
    if "layer_kind" not in body or "layer_k_width" not in body or "layer_v_width" not in body:
        found.append("does not consult the plan's per-layer kind/width arrays")
    return found


def transaction_body_violations(source: str) -> list[str]:
    """The shared runtime-context transaction body must thread the caller's
    n_seq_max AND kv_unified into next_kv_info/next_plan BEFORE the KV
    budget is (re)computed from them -- setting either only afterward
    (as this file's own git history shows for planner_n_ctx/n_ubatch/
    n_seq_max) silently sizes every layer's zone from the STALE previous
    runtime context.
    """
    body = function_or_none(source, TRANSACTION_SIGNATURE)
    if body is None:
        return [f"{TRANSACTION_SIGNATURE} is missing"]

    # Whitespace-tolerant: clang-format column-aligns adjacent `=` signs, so
    # the exact spacing here drifts with whatever sibling assignment lines
    # sit next to it.
    patterns = {
        "next_kv_info.n_seq_max":    r"next_kv_info\.n_seq_max\s*=\s*n_seq_max;",
        "next_kv_info.kv_unified":   r"next_kv_info\.kv_unified\s*=\s*kv_unified;",
        "next_plan.planner_n_seq_max":   r"next_plan\.planner_n_seq_max\s*=\s*n_seq_max;",
        "next_plan.planner_kv_unified":  r"next_plan\.planner_kv_unified\s*=\s*kv_unified;",
    }
    positions: dict[str, int] = {}
    found: list[str] = []
    for name, pattern in patterns.items():
        m = re.search(pattern, body)
        if m is None:
            found.append(f"the transaction body never sets {name}")
        else:
            positions[name] = m.start()

    # Ordering: all four must be set before next_plan.update_runtime_kv_sizes()
    # -- that call's refresh_kv_byte_totals() walks kv_size_for_layer() per
    # layer, which reads next_plan.planner_n_seq_max/n_ubatch/n_ctx/
    # kv_unified directly, and the call also computes
    # next_kv_info.kv_bytes_per_layer()/kv_bytes_per_swa_layer() as its own
    # arguments. Anchored to the qualified CALL (`next_plan.foo(`), not a
    # bare mention -- this file's own comments name the function in prose
    # too.
    update_pos = body.find("next_plan.update_runtime_kv_sizes(")
    if update_pos == -1:
        found.append("next_plan.update_runtime_kv_sizes( is missing")
    else:
        for name, pos in positions.items():
            if pos > update_pos:
                found.append(f"{name} is not set before next_plan.update_runtime_kv_sizes() consumes it")

    return found


def n_embd_gqa_max_violations(source: str) -> list[str]:
    """llama.cpp-3aos: the inventory's FULL-attention width
    fallback fields must be the model-wide MAXIMUM per-layer width
    (hparams.n_embd_k_gqa_max()/n_embd_v_gqa_max()), not layer 0's width
    (hparams.n_embd_k_gqa()/n_embd_v_gqa() with no explicit layer index) --
    on Gemma 4 E4B layer 0 is a SWA layer, so the il=0 form silently
    under-widens the FULL-attention fallback (512 instead of 1024),
    corrupting kv_bytes_per_layer() and everything that reads it
    (ggml_sycl_largest_fitting_n_ctx(), the KV-buffer-kind heuristic).
    """
    found: list[str] = []
    if not re.search(r"inventory\.n_embd_k_gqa\s*=\s*hparams\.n_embd_k_gqa_max\(\)\s*;", source):
        found.append("inventory.n_embd_k_gqa is not assigned from hparams.n_embd_k_gqa_max()")
    if not re.search(r"inventory\.n_embd_v_gqa\s*=\s*hparams\.n_embd_v_gqa_max\(\)\s*;", source):
        found.append("inventory.n_embd_v_gqa is not assigned from hparams.n_embd_v_gqa_max()")
    return found


def stale_comment_violations(hpp_source: str, cpp_source: str) -> list[str]:
    found: list[str] = []
    if "with n_seq_max=1" in hpp_source:
        found.append("unified-cache.hpp still carries the stale 'with n_seq_max=1' comment")
    if "n_seq_max=1-only" in cpp_source:
        found.append("unified-cache.cpp still carries the stale 'n_seq_max=1-only' deferral rationale")
    return found


def test_kv_layer_bytes_for_kind_consumes_n_seq_max_and_kv_unified() -> None:
    assert kv_layer_bytes_for_kind_violations(UNIFIED_CACHE_HPP.read_text()) == []


def test_kv_bytes_for_layer_shares_the_formula() -> None:
    assert kv_bytes_for_layer_violations(UNIFIED_CACHE_HPP.read_text()) == []


def test_kv_size_for_layer_shares_the_formula() -> None:
    assert kv_size_for_layer_violations(UNIFIED_CACHE_HPP.read_text()) == []


def test_transaction_body_threads_n_seq_max_and_kv_unified_before_use() -> None:
    assert transaction_body_violations(GGML_SYCL_CPP.read_text()) == []


def test_llama_model_uses_n_embd_gqa_max() -> None:
    assert n_embd_gqa_max_violations(LLAMA_MODEL_CPP.read_text()) == []


def test_no_stale_n_seq_max_one_comment_remains() -> None:
    assert stale_comment_violations(UNIFIED_CACHE_HPP.read_text(), UNIFIED_CACHE_CPP.read_text()) == []


# ---------------------------------------------------------------------------
# llama.cpp-3aos: each mutation below names ONE checker and ONE
# expected violation substring, and the assertion is that THAT specific
# check fires with THAT specific message -- not "any of several checkers
# fired something". A check that has gone dead (matches nothing, or matches
# the wrong thing) is caught here even if some OTHER checker happens to
# still notice the same mutated text.
# ---------------------------------------------------------------------------

def _assert_witnessed(original: str, mutated: str, checker, expected_substring: str, label: str) -> None:
    assert mutated != original, f"{label}: mutation did not change the source"
    violations = checker(mutated)
    assert any(expected_substring in v for v in violations), (
        f"{label}: expected a violation containing {expected_substring!r}, got {violations!r}")


def test_mutation_seqs_hardcoded_to_1_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace("const uint32_t seqs = n_seq_max > 0 ? n_seq_max : 1;", "const uint32_t seqs = 1;", 1)
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations, "does not derive seqs from n_seq_max",
                      "seqs hardcoded to 1")


def test_mutation_shared_arm_dropped_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = re.sub(r"if \(kind == GGML_SYCL_KV_LAYER_SHARED\) \{\s*\n\s*return 0;\s*\n\s*\}\s*\n", "", hpp, count=1)
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations, "does not return 0 for a SHARED layer",
                      "SHARED arm dropped")


def test_mutation_kv_unified_param_renamed_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    # Renaming the parameter (not removing "bool") is the exact gap the old
    # `"bool" not in ...` check missed -- it would still find "bool" in the
    # signature and pass. This mutation keeps a bool parameter present
    # under a different name, so only a check anchored on "bool kv_unified"
    # as a pair (not "bool" alone) can catch it.
    mutated = hpp.replace(
        "                                      bool     kv_unified) {",
        "                                      bool     kv_unified_flag) {", 1)
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations, "has no bool kv_unified parameter",
                      "kv_unified parameter renamed away")


def test_mutation_kv_unified_branch_dropped_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    # Collapse the two-mode split to the (kv_unified==true) formula
    # unconditionally -- the exact Bug 3 shape (assumes a single shared
    # stream regardless of the caller's real kv_unified). Skips any
    # comment lines between the brace and the assignments (a (?:...)*
    # non-capturing group, not the literal comment text) so the witness
    # anchors on code only and does not go stale when the comment's
    # wording changes.
    mutated, n = re.subn(
        r"if \(kv_unified\) \{\n"
        r"(?:[ \t]*//[^\n]*\n)*"
        r"            n_ctx_seq   = n_ctx;\n"
        r"            n_stream    = 1;\n"
        r"            window_seqs = seqs;\n"
        r"        \} else \{",
        "if (true) {\n"
        "            n_ctx_seq   = n_ctx;\n"
        "            n_stream    = 1;\n"
        "            window_seqs = seqs;\n"
        "        } else if (false) {",
        hpp, count=1)
    assert n == 1, "kv_unified branch pattern did not match the current source"
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations,
                      "does not branch on kv_unified", "kv_unified branch collapsed")


def test_mutation_kv_bytes_for_layer_stops_forwarding_kv_unified_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace(
        "return kv_layer_bytes_for_kind(layer_kind[il], layer_k_width[il], layer_v_width[il], n_ctx, n_swa, n_ubatch,\n"
        "                                           n_seq_max, kv_unified);",
        "return kv_layer_bytes_for_kind(layer_kind[il], layer_k_width[il], layer_v_width[il], n_ctx, n_swa, n_ubatch,\n"
        "                                           n_seq_max, false);", 1)
    _assert_witnessed(hpp, mutated, kv_bytes_for_layer_violations, "without forwarding kv_unified",
                      "kv_bytes_for_layer stops forwarding kv_unified")


def test_mutation_kv_size_for_layer_stops_forwarding_kv_unified_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace(
        "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
        "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max,\n"
        "                                           planner_kv_unified);",
        "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
        "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max,\n"
        "                                           false);", 1)
    _assert_witnessed(hpp, mutated, kv_size_for_layer_violations, "without forwarding planner_kv_unified",
                      "kv_size_for_layer stops forwarding planner_kv_unified")


def test_mutation_stale_comment_reappears_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace(
        "// n_seq_max scales the window -- see the field comment above; it\n",
        "// with n_seq_max=1. n_seq_max scales the window -- see the field comment above; it\n", 1)
    _assert_witnessed(hpp, mutated, lambda s: stale_comment_violations(s, ""),
                      "stale 'with n_seq_max=1' comment", "stale comment reappears")


def test_mutation_stale_cpp_comment_reappears_is_witnessed() -> None:
    # Mirrors the .hpp witness above for the .cpp half of
    # stale_comment_violations() -- only the .hpp half had a mutation
    # witness before this commit, so the .cpp half's marker string could
    # have gone dead (matched nothing) unnoticed.
    cpp = UNIFIED_CACHE_CPP.read_text()
    mutated = cpp.replace(
        "// llama.cpp-3aos: kv_bytes_for_layer() already returns 0 for a\n",
        "// n_seq_max=1-only deferral rationale, superseded.\n"
        "// llama.cpp-3aos: kv_bytes_for_layer() already returns 0 for a\n", 1)
    _assert_witnessed(cpp, mutated, lambda s: stale_comment_violations("", s),
                      "stale 'n_seq_max=1-only' deferral rationale", "stale cpp comment reappears")


def test_mutation_next_kv_info_n_seq_max_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"[ \t]*next_kv_info\.n_seq_max\s*=\s*n_seq_max;\n", "", cpp, count=1)
    _assert_witnessed(cpp, mutated, transaction_body_violations, "never sets next_kv_info.n_seq_max",
                      "next_kv_info.n_seq_max dropped")


def test_mutation_next_kv_info_kv_unified_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"[ \t]*next_kv_info\.kv_unified\s*=\s*kv_unified;\n", "", cpp, count=1)
    _assert_witnessed(cpp, mutated, transaction_body_violations, "never sets next_kv_info.kv_unified",
                      "next_kv_info.kv_unified dropped")


def test_mutation_next_plan_kv_unified_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"[ \t]*next_plan\.planner_kv_unified\s*=\s*kv_unified;\n", "", cpp, count=1)
    _assert_witnessed(cpp, mutated, transaction_body_violations, "never sets next_plan.planner_kv_unified",
                      "next_plan.planner_kv_unified dropped")


def test_mutation_planner_assignments_reordered_after_update_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    before = (
        "    next_plan.planner_n_ctx      = n_ctx;\n"
        "    next_plan.planner_n_ubatch   = next_kv_info.n_ubatch;\n"
        "    next_plan.planner_n_seq_max  = n_seq_max;\n"
        "    next_plan.planner_kv_unified = kv_unified;\n"
        "    next_plan.update_runtime_kv_sizes(n_ctx, next_kv_info.kv_bytes_per_layer(), next_kv_info.kv_bytes_per_swa_layer());\n"
    )
    after = (
        "    next_plan.update_runtime_kv_sizes(n_ctx, next_kv_info.kv_bytes_per_layer(), next_kv_info.kv_bytes_per_swa_layer());\n"
        "    next_plan.planner_n_ctx      = n_ctx;\n"
        "    next_plan.planner_n_ubatch   = next_kv_info.n_ubatch;\n"
        "    next_plan.planner_n_seq_max  = n_seq_max;\n"
        "    next_plan.planner_kv_unified = kv_unified;\n"
    )
    mutated = cpp.replace(before, after, 1)
    _assert_witnessed(cpp, mutated, transaction_body_violations,
                      "is not set before next_plan.update_runtime_kv_sizes() consumes it",
                      "planner_* assignments moved after update_runtime_kv_sizes()")


def test_mutation_n_embd_k_gqa_reverts_to_layer0_form_is_witnessed() -> None:
    src = LLAMA_MODEL_CPP.read_text()
    # Whitespace-tolerant: clang-format column-aligns adjacent `=` signs, so
    # the exact spacing here drifts with whatever sibling assignment sits
    # next to it.
    mutated = re.sub(r"inventory\.n_embd_k_gqa(\s*)=\s*hparams\.n_embd_k_gqa_max\(\);",
                     r"inventory.n_embd_k_gqa\1= hparams.n_embd_k_gqa();", src, count=1)
    _assert_witnessed(src, mutated, n_embd_gqa_max_violations,
                      "inventory.n_embd_k_gqa is not assigned from hparams.n_embd_k_gqa_max()",
                      "n_embd_k_gqa reverts to the layer-0 (non-_max) accessor")


def test_mutation_n_embd_v_gqa_reverts_to_layer0_form_is_witnessed() -> None:
    src = LLAMA_MODEL_CPP.read_text()
    mutated = re.sub(r"inventory\.n_embd_v_gqa(\s*)=\s*hparams\.n_embd_v_gqa_max\(\);",
                     r"inventory.n_embd_v_gqa\1= hparams.n_embd_v_gqa();", src, count=1)
    _assert_witnessed(src, mutated, n_embd_gqa_max_violations,
                      "inventory.n_embd_v_gqa is not assigned from hparams.n_embd_v_gqa_max()",
                      "n_embd_v_gqa reverts to the layer-0 (non-_max) accessor")
