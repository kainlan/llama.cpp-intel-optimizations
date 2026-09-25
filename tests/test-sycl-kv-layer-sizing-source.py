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

  Bug 4 (llama.cpp-uajm, found by a raw-API harness): llama_context_default_
  params() sets swa_full=true while common/common.h defaults it false, so no
  CLI tool ever ran the planner against the raw default. With swa_full,
  llama_kv_cache_iswa sets size_swa = size_base (src/llama-kv-cache-iswa.cpp)
  -- an SWA layer then holds n_ctx cells, byte-identical to a FULL layer --
  but the planner sized SWA layers by the window regardless, and the
  full-context K tensor overflowed the planned slab at context init
  ("[KV-REMAP] ERROR: cache_k_l0 overflows layer alloc!"). The plan must
  describe what llama allocates: swa_full is threaded from cparams through
  the same plumbing as kv_unified, and kv_layer_bytes_for_kind() sizes an
  SWA layer as FULL when it is set.

The fix threads a shared per-layer formula (kv_layer_bytes_for_kind(), a
three-way FULL/SWA/SHARED switch that consumes n_seq_max, kv_unified AND
swa_full)
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
LLAMA_CONTEXT_CPP = ROOT / "src/llama-context.cpp"

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

    # Bug 4 (llama.cpp-uajm): swa_full must be a real parameter AND must be
    # consulted before the windowed SWA formula -- with swa_full llama
    # allocates an SWA layer at size_base (n_ctx_seq per stream), so the
    # windowed arithmetic above must not run for it.
    if not re.search(r"bool\s+swa_full\b",
                     function_signature_params(source, "inline size_t " + KV_LAYER_BYTES_FOR_KIND)):
        found.append("has no bool swa_full parameter")
    if not re.search(r"kind\s*==\s*GGML_SYCL_KV_LAYER_SWA\s*&&\s*!\s*swa_full", body):
        found.append("does not branch on swa_full")

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
    elif not re.search(r"\bswa_full\s*\)", call.group(0)):
        # llama.cpp-uajm: and swa_full, as the LAST argument (a literal
        # false there is exactly the "size by the window regardless" defect).
        found.append("calls kv_layer_bytes_for_kind without forwarding swa_full")
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
    elif not re.search(r"\bplanner_swa_full\s*\)", call.group(0)):
        # llama.cpp-uajm: the plan's own copy of swa_full, last argument.
        found.append("calls kv_layer_bytes_for_kind without forwarding planner_swa_full")
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
        # llama.cpp-uajm: swa_full, same treatment as kv_unified.
        "next_kv_info.swa_full":         r"next_kv_info\.swa_full\s*=\s*swa_full;",
        "next_plan.planner_swa_full":    r"next_plan\.planner_swa_full\s*=\s*swa_full;",
    }
    positions: dict[str, int] = {}
    found: list[str] = []
    for name, pattern in patterns.items():
        m = re.search(pattern, body)
        if m is None:
            found.append(f"the transaction body never sets {name}")
        else:
            positions[name] = m.start()

    # Ordering: all six must be set before next_plan.update_runtime_kv_sizes()
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


def llama_context_swa_full_violations(source: str) -> list[str]:
    """llama.cpp-uajm: llama-context.cpp must (1) record params.swa_full on
    cparams (llama_cparams::swa_full -- the memory module already consumes
    params.swa_full directly, but the SYCL calls below read cparams), (2)
    forward cparams.swa_full on EVERY runtime_context_fn(...) and
    probe_fn(...) call -- the probe must be given the same swa_full the
    candidate would publish with, or its accept/reject answers for the wrong
    KV shape -- and (3) key the persisted auto-n_ubatch cache on it, since
    swa_full changes the plan's KV bytes and so which candidates fit (a
    cached plan from a CLI run, swa_full=false, must not be reused by a
    raw-API context, swa_full=true).
    """
    found: list[str] = []
    if not re.search(r"cparams\.swa_full\s*=\s*params\.swa_full\s*;", source):
        found.append("cparams.swa_full is not assigned from params.swa_full")
    # Every runtime_context_fn(...) / probe_fn(...) call must carry
    # cparams.swa_full somewhere in its argument list; `[^;]*?` bounds the
    # search to one statement so a later call cannot satisfy an earlier one.
    for fn in ("runtime_context_fn", "probe_fn"):
        calls = re.findall(fn + r"\([^;]*?\)\s*;", source)
        if len(calls) < 2:
            found.append(f"expected at least two {fn}(...) call sites, found {len(calls)}")
        for call in calls:
            if "cparams.swa_full" not in call:
                found.append(f"cparams.swa_full is not forwarded to {fn}")
                break
    if not re.search(r"cache_key\.swa_full\s*=\s*cparams\.swa_full\s*;", source):
        found.append("cache_key.swa_full is not assigned from cparams.swa_full")
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


def test_llama_context_threads_swa_full() -> None:
    assert llama_context_swa_full_violations(LLAMA_CONTEXT_CPP.read_text()) == []


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
    # llama.cpp-uajm: kv_unified is no longer the last parameter (swa_full
    # follows it), so the rename is anchored on `bool <ws> kv_unified,`.
    mutated, n = re.subn(r"(bool\s+)kv_unified,", r"\1kv_unified_flag,", hpp, count=1)
    assert n == 1, "kv_unified parameter pattern did not match the current source"
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations, "has no bool kv_unified parameter",
                      "kv_unified parameter renamed away")


def test_mutation_swa_full_param_renamed_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated, n = re.subn(r"(bool\s+)swa_full\)", r"\1swa_full_flag)", hpp, count=1)
    assert n == 1, "swa_full parameter pattern did not match the current source"
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations, "has no bool swa_full parameter",
                      "swa_full parameter renamed away")


def test_mutation_swa_full_branch_dropped_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    # Back to "size by the window regardless of swa_full" -- the exact
    # llama.cpp-uajm defect.
    mutated, n = re.subn(r"kind == GGML_SYCL_KV_LAYER_SWA && !swa_full", "kind == GGML_SYCL_KV_LAYER_SWA", hpp,
                         count=1)
    assert n == 1, "swa_full branch pattern did not match the current source"
    _assert_witnessed(hpp, mutated, kv_layer_bytes_for_kind_violations, "does not branch on swa_full",
                      "swa_full branch dropped")


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
        "                                           n_seq_max, kv_unified, swa_full);",
        "return kv_layer_bytes_for_kind(layer_kind[il], layer_k_width[il], layer_v_width[il], n_ctx, n_swa, n_ubatch,\n"
        "                                           n_seq_max, false, swa_full);", 1)
    _assert_witnessed(hpp, mutated, kv_bytes_for_layer_violations, "without forwarding kv_unified",
                      "kv_bytes_for_layer stops forwarding kv_unified")


def test_mutation_kv_bytes_for_layer_stops_forwarding_swa_full_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace(
        "return kv_layer_bytes_for_kind(layer_kind[il], layer_k_width[il], layer_v_width[il], n_ctx, n_swa, n_ubatch,\n"
        "                                           n_seq_max, kv_unified, swa_full);",
        "return kv_layer_bytes_for_kind(layer_kind[il], layer_k_width[il], layer_v_width[il], n_ctx, n_swa, n_ubatch,\n"
        "                                           n_seq_max, kv_unified, false);", 1)
    _assert_witnessed(hpp, mutated, kv_bytes_for_layer_violations, "without forwarding swa_full",
                      "kv_bytes_for_layer stops forwarding swa_full")


def test_mutation_kv_size_for_layer_stops_forwarding_kv_unified_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace(
        "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
        "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max,\n"
        "                                           planner_kv_unified, planner_swa_full);",
        "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
        "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max,\n"
        "                                           false, planner_swa_full);", 1)
    _assert_witnessed(hpp, mutated, kv_size_for_layer_violations, "without forwarding planner_kv_unified",
                      "kv_size_for_layer stops forwarding planner_kv_unified")


def test_mutation_kv_size_for_layer_stops_forwarding_planner_swa_full_is_witnessed() -> None:
    hpp = UNIFIED_CACHE_HPP.read_text()
    mutated = hpp.replace(
        "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
        "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max,\n"
        "                                           planner_kv_unified, planner_swa_full);",
        "return kv_layer_bytes_for_kind(layer_kind[layer_id], layer_k_width[layer_id], layer_v_width[layer_id],\n"
        "                                           planner_n_ctx, planner_n_swa, planner_n_ubatch, planner_n_seq_max,\n"
        "                                           planner_kv_unified, false);", 1)
    _assert_witnessed(hpp, mutated, kv_size_for_layer_violations, "without forwarding planner_swa_full",
                      "kv_size_for_layer stops forwarding planner_swa_full")


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


def test_mutation_next_kv_info_swa_full_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"[ \t]*next_kv_info\.swa_full\s*=\s*swa_full;\n", "", cpp, count=1)
    _assert_witnessed(cpp, mutated, transaction_body_violations, "never sets next_kv_info.swa_full",
                      "next_kv_info.swa_full dropped")


def test_mutation_next_plan_swa_full_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"[ \t]*next_plan\.planner_swa_full\s*=\s*swa_full;\n", "", cpp, count=1)
    _assert_witnessed(cpp, mutated, transaction_body_violations, "never sets next_plan.planner_swa_full",
                      "next_plan.planner_swa_full dropped")


def test_mutation_cparams_swa_full_not_recorded_is_witnessed() -> None:
    src = LLAMA_CONTEXT_CPP.read_text()
    mutated = re.sub(r"[ \t]*cparams\.swa_full\s*=\s*params\.swa_full;\n", "", src, count=1)
    _assert_witnessed(src, mutated, llama_context_swa_full_violations,
                      "cparams.swa_full is not assigned from params.swa_full", "cparams.swa_full assignment dropped")


def test_mutation_runtime_context_fn_drops_swa_full_is_witnessed() -> None:
    src = LLAMA_CONTEXT_CPP.read_text()
    # Replace the forwarded field with a literal at ONE call site -- the
    # check must fail if ANY site stops forwarding the real value.
    mutated = re.sub(r"(runtime_context_fn\([^;]*?)cparams\.swa_full", r"\1false", src, count=1)
    _assert_witnessed(src, mutated, llama_context_swa_full_violations,
                      "cparams.swa_full is not forwarded to runtime_context_fn",
                      "runtime_context_fn call site stops forwarding cparams.swa_full")


def test_mutation_probe_fn_drops_swa_full_is_witnessed() -> None:
    src = LLAMA_CONTEXT_CPP.read_text()
    mutated = re.sub(r"(probe_fn\([^;]*?)cparams\.swa_full", r"\1false", src, count=1)
    _assert_witnessed(src, mutated, llama_context_swa_full_violations,
                      "cparams.swa_full is not forwarded to probe_fn",
                      "probe_fn call site stops forwarding cparams.swa_full")


def test_mutation_cache_key_swa_full_dropped_is_witnessed() -> None:
    src = LLAMA_CONTEXT_CPP.read_text()
    mutated = re.sub(r"[ \t]*cache_key\.swa_full\s*=\s*cparams\.swa_full;\n", "", src, count=1)
    _assert_witnessed(src, mutated, llama_context_swa_full_violations,
                      "cache_key.swa_full is not assigned from cparams.swa_full", "cache_key.swa_full dropped")


def test_mutation_planner_assignments_reordered_after_update_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    before = (
        "    next_plan.planner_n_ctx      = n_ctx;\n"
        "    next_plan.planner_n_ubatch   = next_kv_info.n_ubatch;\n"
        "    next_plan.planner_n_seq_max  = n_seq_max;\n"
        "    next_plan.planner_kv_unified = kv_unified;\n"
        "    next_plan.planner_swa_full   = swa_full;\n"
        "    next_plan.update_runtime_kv_sizes(n_ctx, next_kv_info.kv_bytes_per_layer(), next_kv_info.kv_bytes_per_swa_layer());\n"
    )
    after = (
        "    next_plan.update_runtime_kv_sizes(n_ctx, next_kv_info.kv_bytes_per_layer(), next_kv_info.kv_bytes_per_swa_layer());\n"
        "    next_plan.planner_n_ctx      = n_ctx;\n"
        "    next_plan.planner_n_ubatch   = next_kv_info.n_ubatch;\n"
        "    next_plan.planner_n_seq_max  = n_seq_max;\n"
        "    next_plan.planner_kv_unified = kv_unified;\n"
        "    next_plan.planner_swa_full   = swa_full;\n"
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


# ---------------------------------------------------------------------------
# llama.cpp-17ea: runtime KV admission wiring. The residency decisions are
# pinned numerically by test-kv-runtime-demotion; what that host test cannot
# see is whether the transaction and the tiered KV allocator are wired to
# them, so these pin the wiring on the source.
# ---------------------------------------------------------------------------

TIERED_KV_ALLOC_SIGNATURE = "static ggml_backend_buffer_t tiered_kv_buft_alloc_buffer"
HOST_KV_REFUSAL = "and be read by device kernels over PCIe; refusing."


def strip_comments(text: str) -> str:
    """Drop // and /* */ comments, keeping string and char literals intact."""
    out: list[str] = []
    state = "code"
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch in "\"'":
                state = "string" if ch == '"' else "char"
            out.append(ch)
        elif state == "line":
            if ch == "\n":
                state = "code"
                out.append(ch)
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        else:
            out.append(ch)
            quote = '"' if state == "string" else "'"
            if ch == "\\":
                out.append(nxt)
                i += 2
                continue
            if ch == quote:
                state = "code"
        i += 1
    return "".join(out)


def runtime_kv_admission_violations(source: str) -> list[str]:
    body = function_or_none(source, TRANSACTION_SIGNATURE)
    if body is None:
        return ["the runtime-context transaction is missing"]
    code = strip_comments(body)
    found: list[str] = []

    if not re.search(r"kv_residency_needs_refit\(\s*published_shape\s*,\s*next_shape\s*,\s*ctx->runtime_kv_admitted\s*\)",
                     code):
        found.append("the re-fit trigger does not receive ctx->runtime_kv_admitted")

    sets = [m.start() for m in re.finditer(r"runtime_kv_admitted\s*=\s*true\s*;", strip_comments(source))]
    tail = re.search(r"ctx->runtime_kv_admitted\s*=\s*true\s*;\s*return\s+ggml_sycl_txn_result::ACCEPTED\s*;\s*}\s*$",
                     code)
    if not sets:
        found.append("runtime_kv_admitted is never set true")
    elif len(sets) != 1 or tail is None:
        found.append("runtime_kv_admitted is set true somewhere other than the publish tail")

    refit = re.search(r"plan_runtime_kv_residency\(in\)(.*?)rebuild_runtime_per_device_vram\(\)", code, re.S)
    if refit is None or "next_plan.refresh_layer_block_kv_devices();" not in refit.group(1):
        found.append("the re-fit does not refresh the layer blocks' KV owners")
    demoted = re.search(r"next_plan\s*=\s*std::move\(demoted_plan\);(.*?)replan_ok\s*=\s*true;", code, re.S)
    if demoted is None or "next_plan.refresh_layer_block_kv_devices();" not in demoted.group(1):
        found.append("the budget-path demotion does not refresh the layer blocks' KV owners")

    if re.search(r"ggml_sycl_largest_fitting_n_ctx\(", code):
        found.append("a -c hint is not derived from the live KV headroom (ggml_sycl_largest_fitting_n_ctx_live)")
    return found


def tiered_kv_refusal_order_violations(source: str) -> list[str]:
    body = function_or_none(source, TIERED_KV_ALLOC_SIGNATURE)
    if body is None:
        return ["the tiered KV allocator is missing"]
    code = strip_comments(body)
    found: list[str] = []
    refusal = code.find(HOST_KV_REFUSAL)
    commit = code.find("mgr = staged;")
    if refusal < 0 or commit < 0:
        return ["the host-KV refusal or the tier manager commit is missing"]
    before = code[:commit]
    if re.search(r"\bmgr\.(configure_from_plan|configure_with_weights|compute_region_layout)\(", before):
        found.append("the device's tier manager is configured before the host-KV refusal")
    for side_effect in ("host_zone_grow(", "ggml_sycl_log_load_summary(device, planned_kv_device"):
        at = code.find(side_effect)
        if at < 0 or at < refusal or at < commit:
            found.append(f"{side_effect} runs before the host-KV refusal decides")
    if refusal > commit:
        found.append("the tier manager is committed before the host-KV refusal decides")
    return found


def test_runtime_kv_admission_wiring() -> None:
    assert runtime_kv_admission_violations(GGML_SYCL_CPP.read_text()) == []


def test_tiered_kv_refusal_has_no_side_effects() -> None:
    assert tiered_kv_refusal_order_violations(GGML_SYCL_CPP.read_text()) == []


def test_mutation_admitted_flag_never_set_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"\n\s*ctx->runtime_kv_admitted\s*=\s*true;", "", cpp, count=1)
    _assert_witnessed(cpp, mutated, runtime_kv_admission_violations, "runtime_kv_admitted is never set true",
                      "the publish tail no longer sets runtime_kv_admitted")


def test_mutation_admitted_flag_set_early_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"\n(\s*)ctx->runtime_kv_admitted\s*=\s*true;", "", cpp, count=1)
    mutated = mutated.replace("    bool   kv_was_demoted        = false;\n",
                              "    bool   kv_was_demoted        = false;\n    ctx->runtime_kv_admitted = true;\n", 1)
    _assert_witnessed(cpp, mutated, runtime_kv_admission_violations,
                      "runtime_kv_admitted is set true somewhere other than the publish tail",
                      "runtime_kv_admitted set before the transaction can still refuse")


def test_mutation_refit_ignores_admission_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = re.sub(r"(kv_residency_needs_refit\(\s*published_shape\s*,\s*next_shape\s*,\s*)ctx->runtime_kv_admitted",
                     r"\1false", cpp, count=1)
    _assert_witnessed(cpp, mutated, runtime_kv_admission_violations,
                      "the re-fit trigger does not receive ctx->runtime_kv_admitted", "trigger passed a constant")


def test_mutation_refit_refresh_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    body = function(cpp, TRANSACTION_SIGNATURE)
    at = body.index("plan_runtime_kv_residency(in)")
    new_body = body[:at] + body[at:].replace("next_plan.refresh_layer_block_kv_devices();\n", "", 1)
    _assert_witnessed(cpp, cpp.replace(body, new_body, 1), runtime_kv_admission_violations,
                      "the re-fit does not refresh the layer blocks' KV owners", "re-fit refresh dropped")


def test_mutation_budget_path_refresh_dropped_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    body = function(cpp, TRANSACTION_SIGNATURE)
    at = body.index("next_plan = std::move(demoted_plan);")
    new_body = body[:at] + body[at:].replace("next_plan.refresh_layer_block_kv_devices();\n", "", 1)
    _assert_witnessed(cpp, cpp.replace(body, new_body, 1), runtime_kv_admission_violations,
                      "the budget-path demotion does not refresh the layer blocks' KV owners",
                      "budget-path refresh dropped")


def test_mutation_budget_hint_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("ggml_sycl_largest_fitting_n_ctx_live(next_plan, next_kv_info, -1, admitted_kv)",
                          "ggml_sycl_largest_fitting_n_ctx(next_plan, next_kv_info, -1, 0, 0)", 1)
    _assert_witnessed(cpp, mutated, runtime_kv_admission_violations,
                      "a -c hint is not derived from the live KV headroom", "a hint bypasses the live headroom")


def test_mutation_tier_manager_configured_in_place_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("staged.configure_from_plan(", "mgr.configure_from_plan(", 1)
    _assert_witnessed(cpp, mutated, tiered_kv_refusal_order_violations,
                      "the device's tier manager is configured before the host-KV refusal",
                      "tier manager configured in place")


def test_mutation_host_zone_grown_before_refusal_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    body = function(cpp, TIERED_KV_ALLOC_SIGNATURE)
    grow_at = body.index("    if (host_kv_bytes > 0 && plan_cache && plan_cache->host_zones_configured()) {")
    grow_end = body.index("\n    }\n", body.index("host_zone_grow(", grow_at)) + len("\n    }\n")
    grow = body[grow_at:grow_end]
    rest = body[:grow_at] + body[grow_end:]
    refusal_at = rest.index("    // Demoted KV belongs in the SYCL_KV_Host buffer type")
    new_body = rest[:refusal_at] + grow + "\n" + rest[refusal_at:]
    _assert_witnessed(cpp, cpp.replace(body, new_body, 1), tiered_kv_refusal_order_violations,
                      "host_zone_grow( runs before the host-KV refusal decides", "host KV zone grown before refusal")


# ---------------------------------------------------------------------------
# llama.cpp-17ea: one live KV headroom. test-kv-runtime-demotion case 21 pins
# kv_vram_available() itself, but its inputs are constants; what makes it the
# admission number is that the transaction, the allocator and the -c hints all
# reach it through unified_cache_kv_vram_available(), and that function keeps
# the has-arena branch. Those call sites are pinned here.
# ---------------------------------------------------------------------------

KV_DEMOTION_HPP = ROOT / "ggml/src/ggml-sycl/kv-runtime-demotion.hpp"
KV_HEADROOM_SIGNATURE = "size_t unified_cache_kv_vram_available("
KV_WEIGHT_CAPACITY_SIGNATURE = "size_t unified_cache_kv_weight_capacity("
LIVE_HINT_SIGNATURE = "static uint32_t ggml_sycl_largest_fitting_n_ctx_live"
TRY_DEMOTE_SIGNATURE = "static bool ggml_sycl_try_demote_runtime_kv"
CTX_HINT_SIGNATURE = "static std::string ggml_sycl_all_vram_ctx_hint"
LLAMA_CONTEXT_CTOR_SIGNATURE = "llama_context::llama_context("
RESYNC_SIGNATURE = "void llama_context::sycl_resync_runtime_context_flash_attn()"
PLAN_OWNED_BLOCK = "if (kv_plan && kv_geometry.valid()) {"


def kv_headroom_wiring_violations(sycl_cpp: str, cache_cpp: str) -> list[str]:
    found: list[str] = []
    txn = strip_comments(function(sycl_cpp, TRANSACTION_SIGNATURE))
    if not re.search(r"in\.available\.push_back\(\s*ggml_sycl::unified_cache_kv_vram_available\(", txn):
        found.append("the transaction's in.available is not the live KV headroom")

    alloc = strip_comments(function(sycl_cpp, TIERED_KV_ALLOC_SIGNATURE))
    caps = re.findall(r"\bkv_vram_cap\s*=(?!=)([^;]*);", alloc)
    if len(caps) != 1 or "ggml_sycl::unified_cache_kv_vram_available(" not in caps[0]:
        found.append("the allocator's kv_vram_cap is not the live KV headroom")
    backstop = re.search(r"if\s*\(\s*ggml_sycl::kv_admission_mismatch\(\s*planned_device_bytes\s*,\s*kv_vram_cap\s*\)"
                         r"\s*\)\s*\{[^{}]*return\s+nullptr\s*;", alloc)
    block = alloc.find(PLAN_OWNED_BLOCK)
    staged = alloc.find("ggml_sycl::kv_tier_manager staged = mgr;")
    if backstop is None:
        found.append("the allocator's kv_admission_mismatch backstop is missing or does not refuse")
    elif block < 0 or not block < backstop.start() < staged or re.search(r"\breturn\s+nullptr\b|\bgoto\b",
                                                                          alloc[block:backstop.start()]):
        found.append("the allocator's backstop is not reached on every planned allocation")
    if re.search(r"runtime_kv_plan\.kv_device\[[^\]]*\]\s*=\s*-1", alloc):
        found.append("the allocator demotes device-planned KV itself")

    live = strip_comments(function(sycl_cpp, LIVE_HINT_SIGNATURE))
    if "ggml_sycl::unified_cache_kv_vram_available(" not in live:
        found.append("the -c hint does not read the live KV headroom")

    probe_calls = re.findall(r"ggml_sycl_run_runtime_context_transaction\(([^;]*)\);", strip_comments(sycl_cpp))
    if not any(re.search(r",\s*true\s*,\s*out\s*$", c) for c in probe_calls):
        found.append("the probe does not run the transaction body")

    body = function_or_none(cache_cpp, KV_HEADROOM_SIGNATURE)
    if body is None:
        return found + ["unified_cache_kv_vram_available is missing"]
    code = strip_comments(body)
    if not re.search(r"return\s+kv_vram_available\(\s*has_arena\s*,\s*has_arena\s*\?\s*cache->zone_available\("
                     r"\s*vram_zone_id::KV\s*\)\s*:\s*0\s*,\s*has_arena\s*\?\s*0\s*:", code):
        found.append("unified_cache_kv_vram_available lost its has-arena branch")
    if re.search(r"==\s*0\s*\)", code):
        found.append("unified_cache_kv_vram_available falls back to the budget when the KV zone reads 0")
    if "get_unified_cache_for_device(" in code:
        found.append("unified_cache_kv_vram_available can create a cache (under g_tensor_inventory_mutex)")
    for sig, name in ((KV_HEADROOM_SIGNATURE, "unified_cache_kv_vram_available"),
                      (KV_WEIGHT_CAPACITY_SIGNATURE, "unified_cache_kv_weight_capacity")):
        fn = function_or_none(cache_cpp, sig)
        if fn is None or not re.search(r"kv_reads_device_arena\(\s*multi_device\s*,", strip_comments(fn)):
            found.append(f"{name} does not apply the multi-device GLOBAL special case")
    return found


def kv_refit_announcement_violations(sycl_cpp: str) -> list[str]:
    found: list[str] = []
    txn = strip_comments(function(sycl_cpp, TRANSACTION_SIGNATURE))
    refit = re.search(r"plan_runtime_kv_residency\(in\)(.*?)rebuild_runtime_per_device_vram\(\)", txn, re.S)
    if refit is None:
        return ["the re-fit is missing"]
    if not re.search(r"load_it\s*=\s*next_plan\.load_kv_device\.find\(", txn) or \
            "next_plan.kv_device = next_plan.load_kv_device;" not in refit.group(1):
        found.append("the re-fit does not restart from the load residency")
    announce = re.search(r"const\s+bool\s+announce\s*=\s*!probe_mode\s*&&\s*next_plan\.kv_device\s*!=\s*"
                         r"current->plan->kv_device\s*;", refit.group(1))
    warn = refit.group(1).find("KV overflow re-placed to host tier")
    gate = refit.group(1).find("if (!announce)")
    if announce is None or gate < 0 or not announce.start() < gate < warn:
        found.append("the re-fit WARN is not gated on a change to the published residency")
    refused = refit.group(1).find("if (!residency.fits)")
    if refused < 0 or (0 <= warn < refused):
        found.append("a refused re-fit can log KV it re-placed")
    return found


def kv_publish_order_violations(ctx_cpp: str) -> list[str]:
    found: list[str] = []
    ctor = strip_comments(function(ctx_cpp, LLAMA_CONTEXT_CTOR_SIGNATURE))
    publish = ctor.find("sycl_resync_runtime_context_flash_attn();")
    memory = ctor.find("memory.reset(model.create_memory(")
    ladder = ctor.find("sycl_select_auto_ubatch(")
    if min(publish, memory, ladder) < 0 or not publish < memory < ladder:
        found.append("a backend's first publish can run after the context's KV is allocated")
    resync = strip_comments(function(ctx_cpp, RESYNC_SIGNATURE))
    skips = re.findall(r"if\s*\(([^)]*)\)\s*\{\s*continue;", resync)
    if skips != ["owner.model_id == 0 || owner.load_txn_id == 0"] or resync.count("continue;") != 1:
        found.append("the constructor's publish can skip one SYCL backend of a context")
    return found


def kv_demotion_message_violations(sycl_cpp: str) -> list[str]:
    found: list[str] = []
    demote = strip_comments(function(sycl_cpp, TRY_DEMOTE_SIGNATURE))
    if not re.search(r"kv_demotion_in\.demote_swa\s*=\s*true\s*;", demote):
        found.append("the budget-path demotion never demotes SWA")
    txn = strip_comments(function(sycl_cpp, TRANSACTION_SIGNATURE))
    if "Largest all-VRAM context is about -c %u" in txn:
        found.append("a demotion WARN prints the -c hint unguarded")
    hint = function_or_none(sycl_cpp, CTX_HINT_SIGNATURE)
    if hint is None or "fits >= 256" not in strip_comments(hint):
        found.append("the -c hint helper does not drop an answer of 0")
    over = re.search(r"const\s+bool\s+plan_over_budget\s*=\s*replan_reason\s*==\s*"
                     r"ggml_sycl::moe_mmid_runtime_reason::BUDGET_EXCEEDED\s*;", txn)
    if over is None or not re.search(r"if\s*\(\s*plan_over_budget\s*\)\s*\{\s*const\s+uint32_t\s+fits\s*=", txn):
        found.append("the budget refusal quotes -c for a constraint -c does not cure")
    if not re.search(r"runtime KV update rejected: %s \(%s\)", txn):
        found.append("the budget refusal does not name its constraint")
    alloc = strip_comments(function(sycl_cpp, TIERED_KV_ALLOC_SIGNATURE))
    if "Reduce -c" in alloc:
        found.append("the allocator backstop blames the context size")
    override = re.search(r"const\s+bool\s+host_kv_override\s*=([^;]*);", alloc)
    if override is None or "kv_hot_layers_override_active(" not in override.group(1) or \
            "!kv_plan" not in override.group(1) or "!= nullptr ||" in override.group(1).split("!kv_plan")[0]:
        found.append("the HOT_* overrides are read by presence or license host layers under a plan")
    return found


def test_kv_headroom_wiring() -> None:
    assert kv_headroom_wiring_violations(GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()) == []


def test_kv_refit_announcement() -> None:
    assert kv_refit_announcement_violations(GGML_SYCL_CPP.read_text()) == []


def test_kv_publish_order() -> None:
    assert kv_publish_order_violations(LLAMA_CONTEXT_CPP.read_text()) == []


def test_kv_demotion_messages() -> None:
    assert kv_demotion_message_violations(GGML_SYCL_CPP.read_text()) == []


def _headroom_checker(cache_cpp: str):
    return lambda sycl_cpp: kv_headroom_wiring_violations(sycl_cpp, cache_cpp)


def _cache_checker(sycl_cpp: str):
    return lambda cache_cpp: kv_headroom_wiring_violations(sycl_cpp, cache_cpp)


def test_mutation_headroom_reverts_to_zero_means_no_arena_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    body = function(cache, KV_HEADROOM_SIGNATURE)
    old = body[:body.index("{") + 1] + """
    size_t available = 0;
    if (vram_arena_enabled()) {
        auto * cache = get_unified_cache_for_device(device_id);
        if (cache && cache->arena_active()) {
            available = cache->zone_available(vram_zone_id::KV);
        }
    }
    if (available == 0) {
        available = unified_cache_available_for_compute(device_id);
    }
    return available;
}"""
    _assert_witnessed(cache, cache.replace(body, old, 1), _cache_checker(cpp),
                      "falls back to the budget when the KV zone reads 0", "cc1381a71's body reverted")


def test_mutation_allocator_inline_cap_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    mutated = re.sub(r"const size_t kv_vram_cap =\s*kv_host_val != 1 \? ggml_sycl::unified_cache_kv_vram_available\("
                     r"device, kv_multi_device\) : 0;", """size_t kv_vram_cap = 0;
    if (kv_host_val != 1) {
        if (ggml_sycl::vram_arena_enabled()) {
            auto * cache = ggml_sycl::get_unified_cache_for_device(device);
            if (cache && cache->arena_active()) {
                kv_vram_cap = cache->zone_available(ggml_sycl::vram_zone_id::KV);
            }
        }
        if (kv_vram_cap == 0) {
            kv_vram_cap = ggml_sycl::unified_cache_available_for_compute(device);
        }
    }""", cpp, count=1)
    _assert_witnessed(cpp, mutated, _headroom_checker(cache), "the allocator's kv_vram_cap is not the live KV headroom",
                      "the old inline cap restored")


def test_mutation_available_from_budgets_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    mutated = cpp.replace("in.available.push_back(ggml_sycl::unified_cache_kv_vram_available(device, "
                          "next_plan.multi_device));",
                          "in.available.push_back(next_plan.per_device_vram_budgets[device]);", 1)
    _assert_witnessed(cpp, mutated, _headroom_checker(cache), "the transaction's in.available is not the live KV headroom",
                      "in.available pointed at per_device_vram_budgets")


def test_mutation_backstop_removed_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    mutated = cpp.replace("if (ggml_sycl::kv_admission_mismatch(planned_device_bytes, kv_vram_cap)) {",
                          "if (false) {", 1)
    _assert_witnessed(cpp, mutated, _headroom_checker(cache), "backstop is missing or does not refuse",
                      "backstop disabled")


def test_mutation_backstop_bypassed_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    body = function(cpp, TIERED_KV_ALLOC_SIGNATURE)
    at = body.index("        // Counted by the allocation loop's owner rule")
    new_body = body[:at] + "        if (kv_vram_cap > 0) {\n            return nullptr;\n        }\n" + body[at:]
    _assert_witnessed(cpp, cpp.replace(body, new_body, 1), _headroom_checker(cache),
                      "backstop is not reached on every planned allocation", "an early return ahead of the backstop")


def test_mutation_allocator_resize_restored_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    mutated = cpp.replace("        kv_plan = &runtime_kv_plan;",
                          "        runtime_kv_plan.kv_device[0] = -1;\n        kv_plan = &runtime_kv_plan;", 1)
    _assert_witnessed(cpp, mutated, _headroom_checker(cache), "the allocator demotes device-planned KV itself",
                      "the allocator's own resize restored")


def test_mutation_headroom_creates_cache_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    body = function(cache, KV_HEADROOM_SIGNATURE)
    new_body = body.replace("get_existing_cache_for_device(device_id)", "get_unified_cache_for_device(device_id)", 1)
    _assert_witnessed(cache, cache.replace(body, new_body, 1), _cache_checker(cpp), "can create a cache",
                      "the creating lookup restored")


def test_mutation_headroom_ignores_global_mode_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    body = function(cache, KV_HEADROOM_SIGNATURE)
    new_body = re.sub(r"kv_reads_device_arena\(multi_device, get_effective_mode\(\) == unified_cache_mode::GLOBAL\)",
                      "true", body, count=1)
    _assert_witnessed(cache, cache.replace(body, new_body, 1), _cache_checker(cpp),
                      "unified_cache_kv_vram_available does not apply the multi-device GLOBAL special case",
                      "the GLOBAL special case dropped")


def test_mutation_refit_warn_ungated_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("const bool announce = !probe_mode && next_plan.kv_device != current->plan->kv_device;",
                          "const bool announce = !probe_mode;", 1)
    _assert_witnessed(cpp, mutated, kv_refit_announcement_violations,
                      "the re-fit WARN is not gated on a change to the published residency",
                      "every re-fit announced")


def test_mutation_refit_sticky_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("next_plan.kv_device = next_plan.load_kv_device;\n        for (size_t l = 0; l < n_kv_layers",
                          "for (size_t l = 0; l < n_kv_layers", 1)
    _assert_witnessed(cpp, mutated, kv_refit_announcement_violations, "does not restart from the load residency",
                      "the re-fit starts from the published residency")


def test_mutation_refit_logs_before_refusal_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    body = function(cpp, TRANSACTION_SIGNATURE)
    refusal_at = body.index("        if (!residency.fits) {")
    refusal_end = body.index("        }\n", body.index("return refuse(", refusal_at)) + len("        }\n")
    refusal = body[refusal_at:refusal_end]
    rest = body[:refusal_at] + body[refusal_end:]
    at = rest.index("        // A second backend of a context re-fits")
    loop_end = rest.index("\n    }\n", rest.index("KV overflow re-placed to host tier", at))
    new_body = rest[:loop_end] + "\n" + refusal + rest[loop_end:]
    _assert_witnessed(cpp, cpp.replace(body, new_body, 1), kv_refit_announcement_violations,
                      "a refused re-fit can log KV it re-placed", "refusal checked after the WARNs")


def test_mutation_publish_after_kv_allocation_is_witnessed() -> None:
    ctx = LLAMA_CONTEXT_CPP.read_text()
    ctor = function(ctx, LLAMA_CONTEXT_CTOR_SIGNATURE)
    first = ctor.index("        sycl_resync_runtime_context_flash_attn();\n")
    moved = ctor[:first] + ctor[first + len("        sycl_resync_runtime_context_flash_attn();\n"):]
    at = moved.index("        memory.reset(model.create_memory(params_mem, cparams));\n")
    at += len("        memory.reset(model.create_memory(params_mem, cparams));\n")
    moved = moved[:at] + "        sycl_resync_runtime_context_flash_attn();\n" + moved[at:]
    _assert_witnessed(ctx, ctx.replace(ctor, moved, 1), kv_publish_order_violations,
                      "a backend's first publish can run after the context's KV is allocated",
                      "the constructor publishes after creating the KV cache")


def test_mutation_publish_skips_a_backend_is_witnessed() -> None:
    ctx = LLAMA_CONTEXT_CPP.read_text()
    mutated = ctx.replace("        if (runtime_context_fn) {\n            const auto & owner = model.get_sycl_model_token();",
                          "        if (runtime_context_fn) {\n            if (backend != backends.front()) {\n"
                          "                continue;\n            }\n"
                          "            const auto & owner = model.get_sycl_model_token();", 1)
    _assert_witnessed(ctx, mutated, kv_publish_order_violations, "can skip one SYCL backend of a context",
                      "one backend skipped")


def test_mutation_budget_path_swa_off_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("    kv_demotion_in.demote_swa  = true;\n", "", 1)
    _assert_witnessed(cpp, mutated, kv_demotion_message_violations, "never demotes SWA", "budget path SWA off")


def test_mutation_unguarded_hint_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace('"in host memory.%s\\n",', '"in host memory. Largest all-VRAM context is about -c %u\\n",', 1)
    _assert_witnessed(cpp, mutated, kv_demotion_message_violations, "prints the -c hint unguarded",
                      "the old unguarded -c print")


def test_mutation_hint_quotes_zero_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("return fits >= 256 ?", "return fits >= 0 ?", 1)
    _assert_witnessed(cpp, mutated, kv_demotion_message_violations, "does not drop an answer of 0", "0 quoted")


def test_mutation_mmid_refusal_quotes_c_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("        if (plan_over_budget) {\n            const uint32_t fits",
                          "        if (true) {\n            const uint32_t fits", 1)
    _assert_witnessed(cpp, mutated, kv_demotion_message_violations, "quotes -c for a constraint -c does not cure",
                      "-c quoted for MMID growth")


def test_mutation_backstop_blames_context_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace("device-planned KV in host memory.\\n\",", "device-planned KV in host memory. Reduce -c.\\n\",", 1)
    _assert_witnessed(cpp, mutated, kv_demotion_message_violations, "blames the context size",
                      "the old Reduce -c text")


def test_mutation_hot_layers_by_presence_is_witnessed() -> None:
    cpp = GGML_SYCL_CPP.read_text()
    mutated = cpp.replace('ggml_sycl::kv_hot_layers_override_active(std::getenv("GGML_SYCL_KV_HOT_LAYERS"))',
                          'std::getenv("GGML_SYCL_KV_HOT_LAYERS") != nullptr', 1)
    _assert_witnessed(cpp, mutated, kv_demotion_message_violations, "read by presence", "HOT_LAYERS by presence")


def test_mutation_probe_skips_transaction_is_witnessed() -> None:
    cpp, cache = GGML_SYCL_CPP.read_text(), UNIFIED_CACHE_CPP.read_text()
    mutated = cpp.replace("flash_attn_enabled, /*probe_mode=*/true, out);", "flash_attn_enabled, /*probe_mode=*/false, out);",
                          1)
    _assert_witnessed(cpp, mutated, _headroom_checker(cache), "the probe does not run the transaction body",
                      "the probe runs the publishing path")
