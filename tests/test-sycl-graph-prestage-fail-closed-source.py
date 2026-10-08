#!/usr/bin/env python3
"""Graph input pre-stage fails closed, and handles view inputs (llama.cpp-qhfp, review fold B1/B2).

Host-only: reads sources, runs no build, loads no model and touches no device.

B1. Once CONCAT stopped keeping qwen35 out of recording, `llama-bench` on Qwen3.6-27B aborted:

    [GRAPH-PRESTAGE] Failed to stage node_src tensor rs_s_copy (view) (data=..., 4 bytes)
    [GET_ROWS] graph recording needs pre-staged input indices for tensor=rs_s_copy (view) bytes=4
    [GET_ROWS] Falling back to CPU get_rows (device index staging failed)
    [SYCL] CPU direct get_rows failed (get_rows index staging): wait method cannot be used for an event
    associated with a command graph.

Two defects, one chain:
  (a) rs_s_copy is a recurrent-state copy index: `ggml_view_1d(s_copy, ...)` of a host-buffer INPUT.
      ggml_view_tensor() does not copy flags, so the view has no GGML_TENSOR_FLAG_INPUT; the pre-stage
      pass only stages INPUT tensors to a stable device buffer, and a non-weight non-INPUT host tensor
      then fails the unified-cache and staging-cache paths. The same chain feeds the replay refresh,
      which also keys on the flag, so a staged view would never be refreshed. A view of an INPUT is that
      input, and a zero-byte view (s_copy_extra with one sequence) has nothing to stage.
  (b) A pre-stage failure did not stop the recording. A consumer that finds no staged copy falls back to a
      host path whose wait is illegal inside a recording. A graph whose inputs cannot be staged must decline
      recording BEFORE it starts and run on the direct path.

Every recorder honours it (review fold I2). The two full-graph sites were not the only ones that pre-stage and
then record or replay: the dense split recorder, the MoE block graphlets and the MoE segment replay and record
all called the pre-stage as a statement and carried on. They are the same hazard (a recorded graph reading a
host-resident input), so each now goes through graph_prestage_or_decline and, on a decline, runs the graph
directly. The gate enumerates every call of graph_prestage_leaf_tensors: the only legal caller is
graph_prestage_or_decline itself, so a new recording site that pre-stages without consuming the result fails.

The staging handle and every recorder that baked its pointer (review r5, r6, r7). graph_input_stage used to drop an
input's staging handle BEFORE allocating the replacement. The drop frees the buffer with no event gate while a
replay from the previous token may still be reading it (the pre-stage does not drain), and a failed allocation
left no entry at all. The staging map owns the handle. A recorded graph keeps a copy only when its consumer
resolves it through a retention path (GET_ROWS indices do, via terminal_retention_ticket::prepare); a consumer
that takes the raw pointer does not, and the map cannot know which consumers baked it. So the fix is in parts,
each pinned below:
  * graph_input_stage allocates the replacement into LOCALS and publishes into the map as its last act, after
    every failure return, so a failure leaves the old entry live;
  * the displaced handle is RETAINED until the last work submitted so far completes
    (retain_handles_until_event on a ggml_sycl_submit_marker event, taken before the handle is touched), never freed
    and never released by a host wait.
    The swap passes a COPY of the handle to retain and only then publishes the replacement, so a throwing retain
    leaves the old entry intact. It sets graph_input_staging_swapped (the one record of the fact) and bumps the staging
    generation. Reuse requires `capacity >= nbytes`;
  * a staging failure for an INPUT tensor is terminal for the pass (all_staged = false). It no longer falls
    through to the cache or staging-cache paths, which could succeed and let the pre-stage report true with the
    entry already displaced. Pre-existing exception, NOT closed here: an INPUT with an empty name does not take
    the INPUT arm (the `tensor->name[0] != '\\0'` guard) and still falls through to the cache paths;
  * graph_prestage_or_decline, the single gateway every recorder passes, consumes the swapped flag, whether the
    pass succeeded or declined, and retires every recorder that may have baked the old pointer
    (graph_staging_swap_retire): dense range graphs (drop_graphs drains), the MoE segment, block, direct-dispatch
    and sequence epochs (each retire waits for its terminals), and a LIVE exec graph. The staging map is not
    touched, so there is no second pre-stage. The retire reports success; a failed MoE retire fails the gateway closed
    (it declines), and what makes a failure a failure is pinned: the retire's catch and non-OK paths return false, each
    invalidator disables its own recorder, and each replay gate honors that disabled flag.
The gateway runs INSIDE a graph_compute that has already pinned weights and experts (graph_preload_weights,
graph_preload_moe_experts), so it must NOT call sycl_exec_graph_clear_active: that unpins the leases, clears the
CPU staging cache and the MoE layout cache mid-compute (its own header cites a measured gemma regression). The gate
forbids clear_active, the staging-map release, the unpins and those caches in the gateway and the swap retire.
The per-site retires at a DECLINE (dense drop_graphs, MoE invalidators, clear_active at the re-record decline)
are conservative and are NOT claimed to cover the pointer.

The decline memo (review fold M2) lives on the context, not in a thread_local slot: a (ctx pointer, hash) slot
was evicted by every other signature (PP and decode alternate) and was keyed by an address a new context can
reuse. A staging failure is also not permanent for a signature: the memo forgets it after a bounded number of
skipped tokens, so a transient failure (a staging allocation) recovers while a structural one (a
host-resident input) re-declines at the cost of one pass.

Run with --self-test to prove every check fires against a mutant of the thing it forbids.
"""
import argparse
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sycl = root / "ggml/src/ggml-sycl"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=str(sycl / "ggml-sycl.cpp"))
parser.add_argument("--common", default=str(sycl / "common.hpp"))
parser.add_argument("--memo", default=str(sycl / "graph-prestage-decline-memo.hpp"))
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()


def strip_comments(source):
    """Remove C/C++ comments, keeping string literals and the line count."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\":
                    if i + 1 < n:
                        out.append(source[i + 1])
                        i += 2
                        continue
                elif source[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("\n" * source.count("\n", i, end))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def mask_strings(source):
    """Replace the CONTENT of string and char literals with 'x' (same length), so a brace, a semicolon or an
    assignment inside a message cannot satisfy or confuse a structural check."""
    out = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ('"', "'"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\" and i + 1 < n:
                    out.append("xx")
                    i += 2
                    continue
                out.append("x" if source[i] != "\n" else "\n")
                i += 1
            if i < n:
                out.append(quote)
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def read(path):
    return strip_comments(Path(path).read_text())


def function_body(source, signature_regex):
    """Return the brace-balanced body of the first function matching the regex."""
    m = re.search(signature_regex, source)
    if not m:
        return None
    start = source.find("{", m.end() - 1 if source[m.end() - 1] == "{" else m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return None


def entry_blocks_graphs(compute):
    """Returns "" when the compute entry turns graphs off after a failed retire, else the reason it does not.

    Some `if (<cond>) { use_sycl_graph = false; }` block must satisfy ALL of:
      (a) its condition is exactly an `||` chain of `sycl_ctx->` members that includes `sycl_ctx->moe_graphs_disabled`
          (no `&&`, negation, comparison, bit-operator or ternary can change what the flag means);
      (b) it is an unguarded top-level statement of the compute body: brace depth 1, and the previous significant
          character (skipping blanks and preprocessor lines) is `;`, `{` or `}`, so no braceless `if`, no `else`
          and no `else if` governs it;
      (c) the only preprocessor conditional open ahead of it is the function's `#ifdef GGML_SYCL_GRAPH`, and no
          `#endif`/`#else`/`#elif` precedes it (conservative: a block hoisted ahead of that `#ifdef` also fails, which
          is the intended pin, not an accident);
      (d) its body is the assignment as a direct statement, plus an optional debug line either side;
      (e) it precedes every consumer of use_sycl_graph;
      (f) from it to the LAST recording nothing assigns use_sycl_graph except `= false` and `&=` (a later `= true`,
          `|= true`, `^=`, `%=`, `<<=`, `++`, `= x || y` or a reference/pointer alias would re-enable it).
    Strings are masked: an assignment in a message is text, not code. Dead code (`if (0)`, a dead wrapper) is rejected
    by (a), (b) and (d); only the shapes above are accepted."""
    masked = mask_strings(compute or "")
    if not masked:
        return "the compute function body was not found"
    consumers = {"moe_graphlet_replay_probe =": masked.find("moe_graphlet_replay_probe ="),
                 "descriptor_moe_graph_candidates": masked.find("descriptor_moe_graph_candidates"),
                 "model_sycl_graph.begin_recording(": masked.rfind("model_sycl_graph.begin_recording(")}
    for name, at in consumers.items():
        if at < 0:
            return f"consumer anchor missing from the compute body: {name}"
    first_consumer = min(consumers.values())
    last_consumer = max(consumers.values())
    shape = r"\{\s*(GGML_SYCL_DEBUG\([^;{}]*\);\s*)?use_sycl_graph\s*=\s*false\s*;\s*(GGML_SYCL_DEBUG\([^;{}]*\);\s*)?\}"
    reason = "no `if (... moe_graphs_disabled ...)` block in the compute body"
    for m in re.finditer(r"\bif\s*\(([^(){};]*)\)\s*\{", masked):
        if re.fullmatch(r"\s*(?:sycl_ctx->\w+\s*\|\|\s*)*sycl_ctx->moe_graphs_disabled(?:\s*\|\|\s*sycl_ctx->\w+)*\s*", m.group(1)) is None:
            continue
        # Previous significant character: skip blanks and preprocessor lines.
        prior = [ln for ln in masked[: m.start()].splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        prev_char = prior[-1].rstrip()[-1:] if prior else ""
        if prev_char not in (";", "{", "}"):
            reason = "the moe_graphs_disabled block is guarded (a braceless if, else or else-if precedes it)"
            continue
        # The graph section of the function lives under the one `#ifdef GGML_SYCL_GRAPH`; any other conditional ahead of
        # the block (an `#if 0` around it, an `#else` arm) could compile it away.
        conditionals = [re.sub(r"#\s*", "#", re.sub(r"\s+", " ", x.strip()))
                        for x in re.findall(r"^\s*#\s*(?:if|ifdef|ifndef|else|elif|endif)\b.*$", masked[: m.start()], re.M)]
        if conditionals != ["#ifdef GGML_SYCL_GRAPH"]:
            reason = "a preprocessor conditional other than `#ifdef GGML_SYCL_GRAPH` sits ahead of the block: " + repr(conditionals)
            continue
        depth = masked.count("{", 0, m.start()) - masked.count("}", 0, m.start())
        if depth != 1:
            reason = "the moe_graphs_disabled block is not a top-level statement of the compute body"
            continue
        end = masked.find("}", m.end())
        if re.fullmatch(shape, masked[m.end() - 1 : end + 1]) is None:
            reason = "the moe_graphs_disabled block does not directly assign use_sycl_graph = false"
            continue
        if end >= first_consumer:
            reason = "the moe_graphs_disabled block does not precede the first consumer"
            continue
        window = masked[end + 1 : last_consumer]
        if re.search(r"(?<!&)&\s*(\w+\s*=\s*)?use_sycl_graph\b", window):
            reason = "use_sycl_graph is aliased by reference after the moe_graphs_disabled block"
            continue
        bad = [x for x in re.finditer(r"\buse_sycl_graph\s*(<<|>>|[|^+\-*/&%]?)=(?!=)|\+\+\s*use_sycl_graph|\buse_sycl_graph\s*\+\+|--\s*use_sycl_graph|\buse_sycl_graph\s*--|\buse_sycl_graph\s*\)\s*=(?!=)", window)
               if not (x.group(1) == "&" or (x.group(1) == "" and re.match(r"\s*false\s*;", window[x.end():])))]
        if bad:
            reason = "use_sycl_graph is re-enabled after the moe_graphs_disabled block"
            continue
        return ""
    return reason


def evaluate(backend, common, memo_hdr):
    results = {}

    prestage = function_body(backend, r"static bool graph_prestage_leaf_tensors\([^)]*\)\s*\{")
    refresh = function_body(backend, r"static void graph_refresh_input_tensors\([^)]*\)\s*\{")
    is_input = function_body(backend, r"static bool graph_tensor_is_input\([^)]*\)\s*\{")
    decline = function_body(backend, r"static bool graph_prestage_or_decline\([^)]*\)\s*\{")
    declined = function_body(backend, r"static bool graph_prestage_skip_declined\([^)]*\)\s*\{")
    compute = function_body(backend, r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{")
    results["anchor: prestage returns bool"] = prestage is not None
    results["anchor: graph_refresh_input_tensors exists"] = refresh is not None
    results["anchor: graph_tensor_is_input exists"] = is_input is not None
    results["anchor: graph_prestage_or_decline exists"] = decline is not None
    results["anchor: graph_prestage_skip_declined exists"] = declined is not None
    results["anchor: graph_compute_unchecked exists"] = compute is not None
    if None in (prestage, refresh, is_input, decline, declined, compute):
        return results

    # --- (a) a view of an INPUT is that input -------------------------------------------------
    results["the INPUT test walks view_src"] = "view_src" in is_input and "GGML_TENSOR_FLAG_INPUT" in is_input
    results["pre-stage asks the view-aware INPUT test"] = "graph_tensor_is_input(" in prestage
    results["pre-stage has no flag-only INPUT test left"] = "flags & GGML_TENSOR_FLAG_INPUT" not in prestage
    results["refresh discovery asks the view-aware INPUT test"] = "graph_tensor_is_input(" in refresh
    results["refresh discovery has no flag-only INPUT test left"] = "flags & GGML_TENSOR_FLAG_INPUT" not in refresh
    # A zero-byte tensor has nothing to stage; reporting it as a failure is what made fail-closed unusable.
    failed_at = prestage.find("Failed to stage")
    zero_at = re.search(r"ggml_nbytes\(tensor\)\s*==\s*0", prestage)
    results["a zero-byte tensor is a no-op, not a staging failure"] = \
        zero_at is not None and 0 <= zero_at.start() < failed_at

    # --- (b) fail closed ----------------------------------------------------------------------
    results["a staging failure is reported to the caller"] = \
        re.search(r"all_staged\s*=\s*false", prestage) is not None and "return all_staged" in prestage
    results["the decline names itself at WARN"] = "GGML_LOG_WARN" in decline and "not recording" in decline
    results["the decline returns false on failure"] = "return false" in decline
    results["a declined graph is remembered on the context, so the pass is not repeated per token"] = \
        "prestage_decline_memo.remember(graph_hash, ctx->graph_compute_seq)" in decline and \
        re.search(r"prestage_decline_memo\.skip\(graph_hash,\s*ctx->graph_compute_seq\)", declined) is not None
    results["the decline memo is not a thread-local or process-wide slot"] = \
        "thread_local" not in decline and "thread_local" not in declined and "g_graph_prestage_skip_declined" not in backend
    results["the context owns the decline memo"] = re.search(r"graph_prestage_decline_memo\s+prestage_decline_memo\s*;", common) is not None
    results["the memo forgets a decline after a bounded number of skips"] = \
        re.search(r"retry_after\s*=\s*\d+", memo_hdr) is not None and "erase(" in memo_hdr
    results["the memo holds several signatures at once"] = \
        "std::vector" in memo_hdr and re.search(r"max_entries\s*=\s*\d+", memo_hdr) is not None
    results["the memo counts a signature once per token, however many sites ask"] = \
        "last_seq" in memo_hdr and re.search(r"bool\s+skip\(uint64_t\s+hash,\s*uint64_t\s+seq\)", memo_hdr) is not None
    results["the context numbers its graph computes, so the memo can tell tokens apart"] = \
        re.search(r"graph_compute_seq\s*(=\s*0)?\s*;", common) is not None and \
        re.search(r"\+\+sycl_ctx->graph_compute_seq;", compute) is not None and \
        0 <= re.search(r"graph_compute_seq", compute).start() < compute.find("graph_prestage_skip_declined(")
    # The bump is an UNCONDITIONAL top-level statement ahead of every early return: a bump under an `if`, inside a
    # nested block, or after a return does not number every token, and the memo then miscounts.
    bump = re.search(r"\+\+sycl_ctx->graph_compute_seq;", compute)
    ctx_def = re.search(r"auto \* sycl_ctx = static_cast<ggml_backend_sycl_context \*>\(backend->context\);", compute)
    results["the token number is bumped unconditionally, at top level, before any early return"] = \
        bool(bump and ctx_def and bump.start() > ctx_def.end() and
             compute[:bump.start()].count("{") - compute[:bump.start()].count("}") == 1 and
             compute[:bump.start()].rstrip()[-1:] in (";", "{", "}") and
             re.search(r"\breturn\b", compute[ctx_def.end():bump.start()]) is None)
    results["the dense split key is namespaced, so one memo never mixes two key spaces"] = \
        re.search(r"static\s+uint64_t\s+dense_split_key\(", memo_hdr) is not None
    results["a successful pre-stage forgets an earlier decline"] = "prestage_decline_memo.forget(graph_hash)" in decline
    sites = [m.start() for m in re.finditer(r"model_sycl_graph\.begin_recording\(", compute)]
    results["both full-graph recording sites exist"] = len(sites) == 2
    ok_sites = True
    for at in sites:
        before = compute[:at]
        call = before.rfind("graph_prestage_or_decline(")
        if call < 0:
            ok_sites = False
            continue
        between = before[call - 20:]
        # the decline must leave the function (direct path) before recording begins, and no void pre-stage
        # may sit between the check and the recording.
        if re.match(r"\s*if\s*\(\s*!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\s*\)\s*\{\s*"
                    r"(sycl_exec_graph_clear_active\(sycl_ctx, \"prestage-declined\"\);\s*)?"
                    r"compute_impl_unlocked\(\);\s*record_completion\(false\);\s*return GGML_STATUS_SUCCESS;",
                    between[between.find("if"):]) is None:
            ok_sites = False
        if "graph_prestage_leaf_tensors(" in between:
            ok_sites = False
    results["every full-graph recording is preceded by a pre-stage that can decline it"] = len(sites) == 2 and ok_sites
    results["an already-declined graph goes direct before any recording state is built"] = \
        "graph_prestage_skip_declined(" in compute and \
        0 <= compute.find("graph_prestage_skip_declined(") < compute.find("model_sycl_graph.begin_recording(")

    # --- every pre-stage consumer honours the result ------------------------------------------------------
    def calls(name):
        # a call, not the declaration or definition ("static bool name(")
        return [m.start() for m in re.finditer(r"(?<!bool )\b%s\(" % name, backend)]

    decl_span = (backend.find(decline), backend.find(decline) + len(decline))
    stray = [at for at in calls("graph_prestage_leaf_tensors") if not decl_span[0] <= at < decl_span[1]]
    results["no code pre-stages as a statement: graph_prestage_or_decline is the only caller of the pre-stage"] = \
        not stray
    consumers = calls("graph_prestage_or_decline")
    consumed = [at for at in consumers
                if re.search(r"if\s*\([^;{]*!\s*$", backend[max(0, at - 160):at]) is not None]
    results["every pre-stage consumer tests the result (void-style calls ignore a failure)"] = \
        len(consumers) > 0 and len(consumers) == len(consumed)
    results["no unnamed pre-stage consumer: the seven recording sites below are all there are"] = len(consumers) == 7

    # Each recording site, named, with the exact shape of its decline: the condition is the bare negated call (a
    # `&& false` or `|| true` tail would make the decline dead or the pre-stage unconditional), and the body leaves
    # for the direct path (a dense decline that only sets graphs_off_ and falls through still records).
    call = r"graph_prestage_or_decline\(%s\)"
    dense = function_body(backend, r"void prepare_graphs\(\)\s*\{") or ""
    graphlets = function_body(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{") or ""
    results["site 1, dense split recorder: declines with STAGE_FAILED, drops its graphs and returns"] = \
        re.search(r"if\s*\(any_missing\s*&&\s*!" + call % r"&ctx_,\s*cgraph_,\s*graph_prestage_decline_memo::dense_split_key\(key\)" +
                  r"\)\s*\{\s*graphs_off_\s*=\s*ggml_sycl::DENSE_GRAPH_OFF_STAGE_FAILED;\s*st\.drop_graphs\(ctx_\);\s*return;\s*\}",
                  dense) is not None
    results["site 2, MoE block graphlets: declines, invalidates them and returns false to the direct fallback"] = re.search(
        r"if\s*\(!" + call % r"sycl_ctx,\s*cgraph,\s*graph_hash" + r"\)\s*\{[^{}]*;\s*sycl_ctx->invalidate_moe_block_graphs\(\);\s*return false;\s*\}",
        graphlets) is not None
    site = r"if\s*\(!" + call % r"sycl_ctx,\s*cgraph,\s*graph_hash" + r"\)\s*\{\s*"
    results["site 3, MoE segment replay: declines, invalidates the segments, runs direct, else replays"] = re.search(
        site + r"sycl_ctx->invalidate_moe_segments\(\);\s*compute_impl_unlocked\(\);\s*\}\s*else\s*"
        r"(if\s*\(!sycl_ctx->moe_segments_valid\)\s*\{\s*compute_impl_unlocked\(\);\s*\}\s*else\s*)?\{\s*"
        r"graph_refresh_input_tensors\(sycl_ctx, cgraph\);\s*moe_graph_replay_segments\(", compute) is not None
    results["site 3: segments a staging swap retired run the token direct instead of replaying nothing"] = re.search(
        site + r"sycl_ctx->invalidate_moe_segments\(\);\s*compute_impl_unlocked\(\);\s*\}\s*else if\s*\(!sycl_ctx->moe_segments_valid\)\s*\{\s*"
        r"compute_impl_unlocked\(\);\s*\}\s*else\s*\{\s*graph_refresh_input_tensors\(sycl_ctx, cgraph\);\s*moe_graph_replay_segments\(", compute) is not None
    results["site 4, MoE segment record: declines and runs direct, else records"] = re.search(
        r"\}\s*else\s*" + site + r"compute_impl_unlocked\(\);\s*\}\s*else\s*\{\s*graph_refresh_input_tensors\(sycl_ctx, cgraph\);\s*"
        r"sycl_ctx->invalidate_moe_segments\(\);", compute) is not None
    exit_direct = r"compute_impl_unlocked\(\);\s*record_completion\(false\);\s*return GGML_STATUS_SUCCESS;\s*\}"
    rerecord_at = compute.find("re-record + update (%s)")
    # the FIRST pre-stage after the re-record anchor, matched from its own `if` (a later site must not satisfy it)
    rr_call = compute.find("graph_prestage_or_decline(", rerecord_at) if rerecord_at >= 0 else -1
    rr_decline = re.match(site + r"sycl_exec_graph_clear_active\(sycl_ctx, \"prestage-declined\"\);\s*" + exit_direct,
                          compute[compute.rfind("if", 0, rr_call):]) if rr_call >= 0 else None
    results["site 5, full re-record: declines, clears the live graph (after a wait) and leaves for the direct path"] = \
        rr_decline is not None
    # llama.cpp-7pm2 B2: a keyed decode split's slot. Warmup and direct keys record nothing and run before the
    # decline; a replay or a record runs only past it, and each re-reads the slot, since a staging swap at the
    # gateway retires every slot after begin() chose the action. A replay whose inputs now stage to other buffers
    # than the slot's graphs read forgets the slot and runs direct.
    keyed = re.search(
        r"\}\s*else if\s*\(slot_action == gsc::action::DIRECT\)\s*\{\s*compute_impl_unlocked\(\);\s*\}\s*else\s*" + site +
        r"compute_impl_unlocked\(\);\s*\}\s*else if\s*\(slot_action == gsc::action::REPLAY\)\s*\{\s*"
        r"const auto \* slot = sycl_ctx->moe_segment_slots\.payload\(slot_key\);\s*if\s*\(!slot\)\s*\{\s*"
        r"compute_impl_unlocked\(\);\s*\}\s*"
        r"else if\s*\(!moe_segment_slot_staging_matches\(sycl_ctx, cgraph, \*slot\)\)\s*\{[^{}]*?"
        r"sycl_ctx->moe_segment_slots\.forget\(slot_key\);\s*compute_impl_unlocked\(\);\s*\}\s*"
        r"else\s*\{\s*moe_segment_slot_refresh_inputs\(sycl_ctx, cgraph, \*slot\);\s*"
        r"moe_graph_replay_segment_slot\(sycl_ctx, cgraph, \*slot\);\s*graph_executed = true;\s*\}\s*\}\s*"
        r"else if\s*\(!sycl_ctx->moe_segment_slots\.state\(slot_key\)\)\s*\{\s*compute_impl_unlocked\(\);\s*\}\s*else\s*\{",
        compute)
    results["site 7, keyed segment slot: declines and runs direct, else replays or records a slot that still exists"] = \
        keyed is not None and compute.find("moe_graph_record_segment_slot(", keyed.end()) > keyed.end()
    first_at = compute.find("Pre-staging leaf tensors before recording")
    results["site 6, full first record: declines and leaves for the direct path"] = \
        first_at >= 0 and re.match(r"[^;]*;\s*" + site + exit_direct, compute[first_at:]) is not None

    # Re-record tears the live exec graph down (reset, release the retained pool, mark inactive). The decline is
    # decided BEFORE every one of them: a transient staging failure must not destroy a graph the last replay may
    # still have in flight, nor leave its pins and hash stale, and the attempt is only counted once the pre-stage
    # let it proceed.
    def first_after(needle):
        return compute.find(needle, rerecord_at) if rerecord_at >= 0 else -1

    actions = {"exec_graph.reset()": first_after("exec_graph.reset()"),
               "release_pool_retained(": first_after("release_pool_retained("),
               "active_exec_graph.valid = false": first_after("active_exec_graph.valid = false"),
               "rerecord_attempts": first_after("rerecord_attempts.fetch_add")}
    results["re-record decides the decline before every teardown action and before counting the attempt"] = \
        rr_decline is not None and all(at >= 0 and rr_call < at for at in actions.values())

    # The six sites appear in this order, each decline strictly between its neighbours' anchors. A decline block
    # that moved (with its debug string) past a begin_recording, or into another site's region, still satisfies
    # every per-site shape above; the order is what pins where it sits.
    begin = [m.start() for m in re.finditer(r"model_sycl_graph\.begin_recording\(", compute)]
    s3 = re.search(r"moe_graph_replay_segments\(", compute)
    s4 = re.search(r"moe_graph_record_segments\(", compute)
    s7 = re.search(r"slot_action == gsc::action::DIRECT", compute)
    s7_call = compute.find("graph_prestage_or_decline(", s7.end()) if s7 else -1
    s7_replay = compute.find("moe_graph_replay_segment_slot(")
    s7_record = compute.find("moe_graph_record_segment_slot(")
    first_call = compute.find("graph_prestage_or_decline(", first_at) if first_at >= 0 else -1
    chain = [("keyed decline", s7_call), ("keyed replay", s7_replay), ("keyed record", s7_record),
             ("segment replay", s3.start() if s3 else -1), ("segment record", s4.start() if s4 else -1),
             ("re-record anchor", rerecord_at), ("re-record decline", rr_call),
             ("re-record begin_recording", begin[0] if len(begin) == 2 else -1),
             ("first-record anchor", first_at), ("first-record decline", first_call),
             ("first-record begin_recording", begin[1] if len(begin) == 2 else -1)]
    results["the recording sites keep their order, each decline between its neighbours (and before its recording)"] = \
        all(at >= 0 for _, at in chain) and all(chain[i][1] < chain[i + 1][1] for i in range(len(chain) - 1))

    # --- staging handle swap (review r6, r7) ----------------------------------------------------------------------
    stage_fn = function_body(common, r"void \* graph_input_stage\(const ggml_tensor \* owner,[^)]*\)\s*\{") or ""
    clear_fn = function_body(common, r"void graph_input_staging_clear\(sycl::queue & q\)\s*\{") or ""
    swap_fn = function_body(backend, r"static bool graph_staging_swap_retire\(ggml_backend_sycl_context \* ctx\)\s*\{") or ""

    def stmt_at(body, token):
        """Position of `token` where it STARTS a statement (after `;`, `{` or `}`), else -1: a token under an `if`
        or inside an expression does not count."""
        m = re.search(r"(?:^|[;{}])\s*" + re.escape(token), body)
        return m.end() - len(token) if m else -1

    results["graph_input_stage allocates the replacement before it touches the entry (no drop-then-allocate)"] = \
        bool(stage_fn) and "unified_allocate_owner(" in stage_fn and \
        re.search(r"it->second\.handle\s*=\s*ggml_sycl::mem_handle\s*\{\s*\}", stage_fn) is None and \
        all(m.start() > stage_fn.find("unified_allocate_owner(") for m in re.finditer(r"graph_input_staging\[owner\]", stage_fn)) and \
        re.search(r"graph_input_staging\[owner\]", stage_fn) is not None
    # The map entry is the PUBLISH point: nothing may be assigned into it before the last failure return, or a
    # failed allocation, resolve or copy leaves an unvalidated entry behind.
    last_fail = max((m.start() for m in re.finditer(r"return nullptr;", stage_fn)), default=-1)
    results["graph_input_stage publishes into the map only after its last failure return"] = \
        last_fail >= 0 and len(re.findall(r"graph_input_staging\[owner\]", stage_fn)) == 1 and \
        stage_fn.find("graph_input_staging[owner]") > last_fail
    results["the displaced staging handle is retained until a marker event, flagged, and the generation bumped"] = \
        re.search(r"if\s*\(\s*slot\.handle\.valid\(\)\s*\)\s*\{\s*"
                  r"sycl::event\s+retire\s*=\s*ggml_sycl_submit_marker<graph_input_staging_retire_marker>\(q\);\s*"
                  r"graph_input_staging_swapped\s*=\s*true;\s*"
                  r"ggml_sycl::retain_handles_until_event\(\s*\{\s*slot\.handle\s*\}\s*,\s*retire\s*\);\s*\}\s*"
                  r"slot\.handle\s*=\s*std::move\(handle\);\s*slot\.capacity\s*=\s*nbytes;\s*graph_input_staging_generation\+\+;",
                  stage_fn) is not None
    # Review r8 M1/M2: the event is taken FIRST, before the slot is touched (an argument-evaluation order that moved the
    # handle and then threw from the barrier would free the buffer with no event gate), and it comes from the
    # documented marker helper, not a bare ext_oneapi_submit_barrier (the L0 barrier-event corruption it avoids).
    results["the staging swap takes its event from the marker helper before it moves the handle, never a bare barrier"] = \
        re.search(r"struct\s+graph_input_staging_retire_marker\s*\{\s*\}\s*;", common) is not None and \
        "ext_oneapi_submit_barrier" not in stage_fn and \
        stage_fn.find("ggml_sycl_submit_marker<graph_input_staging_retire_marker>(q)") >= 0 and \
        stage_fn.find("ggml_sycl_submit_marker<graph_input_staging_retire_marker>(q)") < stage_fn.find("retain_handles_until_event")
    # One fact, one source, and the only mutation sites: the swap itself publishes and retains; nothing else resets,
    # erases or releases an entry, and the swap waits on nothing (retention is by event, not by host wait).
    results["the staging map's only mutation sites are the swap and the clear, and the swap waits on nothing"] = \
        len(re.findall(r"\.handle\s*=[^=]", stage_fn)) == 1 and \
        len(re.findall(r"graph_input_staging_swapped\s*=\s*true", stage_fn)) == 1 and \
        re.search(r"\.wait\(|wait_and_throw|trace_queue_wait|sycl::event\s*\{\s*\}", stage_fn) is None and \
        len(re.findall(r"graph_input_staging\.clear\(\)", common)) == 1 and \
        re.search(r"graph_input_staging\.erase\(", common) is None and \
        re.search(r"graph_input_staging\.(clear|erase)|graph_input_staging_retired", backend) is None and \
        "graph_input_staging_retired" not in common
    results["the context owns the swapped flag, and clear resets it with the map"] = \
        re.search(r"bool\s+graph_input_staging_swapped\s*=\s*false\s*;", common) is not None and \
        re.search(r"graph_input_staging\.clear\(\);\s*graph_input_staging_swapped\s*=\s*false;", clear_fn) is not None
    results["a staging failure for an INPUT tensor is terminal for the pass (no fallthrough to the cache paths)"] = \
        re.search(r"graph_input_stage\([^;]*;\s*if\s*\(\s*!dev_ptr\s*\)\s*\{[^{}]*\ball_staged\s*=\s*false;\s*return;\s*\}", prestage) is not None
    results["the INPUT arm tests exactly 'is input, named' and its success path returns"] = \
        re.search(r"if\s*\(graph_tensor_is_input\(tensor\)\s*&&\s*tensor->name\s*&&\s*tensor->name\[0\]\s*!=\s*'\\0'\)\s*\{[^{}]*"
                  r"graph_input_stage\([^;]*;\s*if\s*\(\s*!dev_ptr\s*\)\s*\{[^{}]*\}\s*staged_count\+\+;\s*mark_staged\(tensor\);\s*"
                  r"GGML_SYCL_DEBUG\([^;]*;\s*return;\s*\}", prestage) is not None
    results["the gateway consumes the swapped flag and retires the recorders, with no second pass"] = \
        re.search(r"const bool staged\s*=\s*graph_prestage_leaf_tensors\(ctx, cgraph\);\s*bool\s+retired\s*=\s*true;\s*"
                  r"if\s*\(ctx->graph_input_staging_swapped\)\s*\{\s*retired = graph_staging_swap_retire\(ctx\);\s*\}\s*"
                  r"if\s*\(staged && retired\)\s*\{", decline) is not None and \
        len(re.findall(r"graph_prestage_leaf_tensors\(", decline)) == 1
    # I1 (r7): the gateway and the swap retire run INSIDE a graph_compute that has already pinned weights and experts,
    # so they must not call the whole clear_active (it unpins leases, clears the CPU staging cache and the MoE layout
    # cache) and must not release the staging map. They leave leases, pins and those caches alone.
    forbidden = ["sycl_exec_graph_clear_active", "graph_input_staging_clear", "graph_unpin_", "graph_weight_leases",
                 "graph_moe_expert_leases", "ggml_sycl_cpu_staging_cache_clear", "invalidate_moe_phase_layout_cache",
                 "input_tensors_cached", "cached_input_tensors"]
    results["the gateway and the swap retire leave leases, pins, the CPU staging cache and the MoE layout cache alone"] = \
        bool(swap_fn) and all(t not in swap_fn and t not in decline for t in forbidden)
    # Review r8 M3: statement ORDER is not enough (a brace-wrapped dead call, a host wait, a stray release or a stray
    # signature-cache reset all keep every anchored statement in order). The retire is pinned as its exact token
    # sequence, so nothing may be added, wrapped, reordered or dropped; the only conditional is the live exec block.
    swap_expected = " ".join([
        "{ ctx->graph_input_staging_swapped = false;",
        'if (ggml_sycl_graph_diag_enabled()) { GGML_LOG_WARN("[GRAPH-DIAG] staging swap at graph compute %llu\\n", '
        "(unsigned long long) ctx->graph_compute_seq); }",
        "ggml_sycl_block_exec_dense_drop_graphs(ctx);",
        "const bool segments_retired = ctx->invalidate_moe_segments();",
        "const bool block_graphs_retired = ctx->invalidate_moe_block_graphs();",
        "const bool direct_dispatch_retired = ctx->invalidate_moe_direct_dispatch_graphs();",
        "const bool sequence_graphs_retired = ctx->invalidate_moe_sequence_graphs();",
        'if (ctx->exec_graph) { ggml_sycl_trace_queue_wait(ctx->stream(), "staging-swapped", ctx->device, -1, nullptr); '
        "ctx->exec_graph.reset(); sycl_exec_graph_release_pool_retained(ctx); ctx->active_exec_graph.valid = false; "
        "ctx->exec_graph_n_nodes = 0; ctx->exec_graph_hash = 0; }",
        "return segments_retired && block_graphs_retired && direct_dispatch_retired && sequence_graphs_retired; }",
    ])
    results["the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction"] = \
        bool(swap_fn) and re.sub(r"\s+", " ", swap_fn).strip() == swap_expected
    # Review r8 M4: a failed retire must fail the gateway closed. Each invalidator reports whether it retired (a
    # failure leaves the recorded graphs valid, and sets only its own disabled flag), the retire ANDs all four without
    # short-circuiting, and the callers that match a recorded graph after the gateway also honor the disabled flag.
    invalidators = {"invalidate_moe_segments": "moe_graphs_disabled", "invalidate_moe_direct_dispatch_graphs": "moe_direct_dispatch_graphs_disabled",
                    "invalidate_moe_block_graphs": "moe_block_graphs_disabled", "invalidate_moe_sequence_graphs": "moe_sequence_graphs_disabled"}
    inv_ok = True
    for name, flag in invalidators.items():
        body = function_body(common, r"\bbool\s+" + name + r"\(\)\s*\{") or ""
        inv_ok = inv_ok and bool(re.match(r"\{\s*if\s*\(!ggml_sycl_retire_moe_graph_epoch\(this\)\)\s*\{\s*" +
                                          flag + r"\s*=\s*true;\s*return false;\s*\}", body)) and \
            re.search(r"return true;\s*\}\s*$", body) is not None
    results["each MoE invalidator reports whether it retired the epoch (false on failure)"] = inv_ok
    results["the gateway declines when the staging swap could not retire every recorder"] = \
        re.search(r"if\s*\(staged && retired\)\s*\{\s*ctx->prestage_decline_memo\.forget\(graph_hash\);\s*return true;\s*\}\s*"
                  r"ctx->prestage_decline_memo\.remember\(graph_hash, ctx->graph_compute_seq\);", decline) is not None
    results["graph_input_stage reuses an entry only when it is large enough (capacity >= nbytes)"] = \
        re.search(r"auto it = graph_input_staging\.find\(owner\);\s*if\s*\(it != graph_input_staging\.end\(\) && it->second\.capacity >= nbytes\)\s*\{", stage_fn) is not None
    # Review r9: what makes a failure a failure. The retire's drain and registry paths report false, and every replay
    # gate honors the disabled flag the failure sets, so a failed retire cannot be walked past.
    retire_exact_fn = function_body(backend, r"static bool moe_graph_retention_retire_exact\(ggml_backend_sycl_context \* ctx\)\s*\{") or ""
    epoch_fn = function_body(backend, r"bool ggml_sycl_retire_moe_graph_epoch\(ggml_backend_sycl_context \* ctx\) noexcept\s*\{") or ""
    results["a failed drain or registry retire makes the epoch retire report false"] = \
        re.search(r"wait_and_throw\(\);\s*\}\s*\}\s*catch\s*\(\.\.\.\)\s*\{\s*return false;\s*\}", retire_exact_fn) is not None and \
        re.search(r"if\s*\(rc != ggml_sycl::moe::retention_error::OK && rc != ggml_sycl::moe::retention_error::STALE\)\s*\{\s*return false;\s*\}", retire_exact_fn) is not None and \
        re.fullmatch(r"\{\s*try\s*\{\s*return moe_graph_retention_retire_exact\(ctx\);\s*\}\s*catch\s*\(\.\.\.\)\s*\{\s*return false;\s*\}\s*\}", epoch_fn) is not None
    # Whitespace-flexible on purpose: a clang-format rewrap of these conditions must not turn the gate red.
    results["every replay gate honors the disabled flag a failed retire sets"] = \
        re.search(r"!sycl_ctx->moe_graphs_disabled\s*&&\s*!sycl_ctx->moe_direct_dispatch_graphs_disabled\s*&&", backend) is not None and \
        re.search(r"sycl_ctx->moe_graphs_disabled\s*\|\|\s*sycl_ctx->moe_sequence_graphs_disabled\s*\|\|\s*node->op\s*!=\s*GGML_OP_MUL_MAT_ID", backend) is not None and \
        re.search(r"if\s*\(\s*sycl_ctx->moe_block_graphs_disabled\s*\|\|[^{};]*\bsycl_ctx->moe_graphs_disabled\s*\)\s*\{", graphlets) is not None
    # With the post-gateway checks gone, this entry block is what stops a segment replay after a failed retire
    # (the segments stay marked valid): it must turn graphs off for the whole compute.
    entry_reason = entry_blocks_graphs(compute)
    if entry_reason:
        print("NOTE: compute-entry check: " + entry_reason)
    results["the compute entry turns graphs off when MoE graphs were disabled by a failed retire"] = entry_reason == ""
    results["the INPUT arm stages on the backend's own queue"] = \
        re.search(r"if\s*\(graph_tensor_is_input\(tensor\)\s*&&\s*tensor->name\s*&&\s*tensor->name\[0\]\s*!=\s*'\\0'\)\s*\{\s*"
                  r"sycl::queue\s*&\s*q\s*=\s*\*ctx->stream\(\);\s*void\s*\*\s*dev_ptr\s*=\s*ctx->graph_input_stage\(tensor, tensor->data, nbytes, q\);",
                  prestage) is not None
    dense_drop = function_body(backend, r"static void ggml_sycl_block_exec_dense_drop_graphs\(ggml_backend_sycl_context \* ctx\)\s*\{") or ""
    results["the dense drop helper drops a state that exists and does not create one"] = \
        "drop_graphs(*ctx)" in dense_drop and "find(ctx)" in dense_drop and "state_for(" not in dense_drop
    gw_at = dense.find("graph_prestage_or_decline(")
    results["the dense recorder re-sizes its range graphs after the gateway may have retired them"] = \
        gw_at >= 0 and re.search(r"if\s*\(st\.graphs\.size\(\)\s*!=\s*ranges_\.size\(\)\)\s*\{\s*st\.graphs\.resize\(ranges_\.size\(\)\);\s*st\.graphs_key\s*=\s*key;\s*ctx_\.input_tensors_cached\s*=\s*false;", dense[gw_at:]) is not None

    # Every release of recorded state waits first, under the guard that says there is something in flight (review r6).
    clear_active = function_body(backend, r"static void sycl_exec_graph_clear_active\(ggml_backend_sycl_context \* ctx, const char \* reason\)\s*\{") or ""
    wait_at = re.search(r"if\s*\(\s*ctx->exec_graph\s*\)\s*\{\s*ggml_sycl_trace_queue_wait\(ctx->stream\(\)", clear_active)
    results["clear_active waits on the queue before it resets the exec graph or releases the staging"] = \
        bool(wait_at) and 0 <= wait_at.start() < clear_active.find("ctx->exec_graph.reset()") < \
        clear_active.find("graph_input_staging_clear(") and clear_active.count("graph_input_staging_clear(") == 1
    drop_fn = function_body(backend, r"void ggml_sycl_block_exec_dense_state::drop_graphs\(ggml_backend_sycl_context & ctx\)\s*\{") or ""
    results["drop_graphs drains each used device before it clears the range graphs"] = \
        re.search(r"if\s*\(!used\[d\]\)\s*\{\s*continue;\s*\}\s*try\s*\{\s*ggml_sycl_block_exec_dense_queue\(ctx, d\)->wait_and_throw\(\);", drop_fn) is not None \
        and drop_fn.find("wait_and_throw()") < drop_fn.find("graphs.clear()")
    retire_fn = function_body(backend, r"static bool moe_graph_retention_retire_exact\(ggml_backend_sycl_context \* ctx\)\s*\{") or ""
    results["retire_exact drains every terminal's queue before it retires the epoch"] = \
        re.search(r"for\s*\(const auto & \[device, _\] : ctx->moe_retention_terminals\)\s*\{\s*ctx->stream\(device, 0\)->wait_and_throw\(\);", retire_fn) is not None \
        and retire_fn.find("wait_and_throw()") < retire_fn.find("retire_exact(ctx->moe_retention_epoch)")
    return results


def run(label, sources, expect_fail=None):
    results = evaluate(*sources)
    failed = sorted(k for k, v in results.items() if not v)
    if expect_fail is None:
        for k in sorted(results):
            print(("PASS: " if results[k] else "FAIL: ") + k)
        return failed
    fired = expect_fail in failed
    print(("PASS" if fired else "FAIL") + f": mutant '{label}' fires '{expect_fail}'")
    return [] if fired else [label]


backend, common = read(args.backend), read(args.common)
memo_hdr = read(args.memo) if Path(args.memo).exists() else ""
failed = run("tree", (backend, common, memo_hdr))

if args.self_test:
    def mutate(src, old, new, count=1):
        if old not in src:
            print(f"FAIL: self-test anchor missing: {old!r}")
            failed.append("self-test anchor " + old)
            return src
        return src.replace(old, new, count)

    def mutate_in_func(src, sig_regex, old, new):
        m = re.search(sig_regex, src)
        if not m or src.find(old, m.end()) < 0:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {old!r}")
            failed.append("self-test anchor " + old)
            return src
        k = src.find(old, m.end())
        return src[:k] + new + src[k + len(old):]

    def mutate_re(src, sig_regex, pattern, repl):
        """Replace the first match of `pattern` after the first match of `sig_regex`."""
        m = re.search(sig_regex, src)
        pm = re.compile(pattern).search(src, m.end()) if m else None
        if not pm:
            print(f"FAIL: self-test anchor missing: {sig_regex!r} .. {pattern!r}")
            failed.append("self-test anchor " + pattern)
            return src
        return src[:pm.start()] + repl + src[pm.end():]

    pre_sig = r"static bool graph_prestage_leaf_tensors\([^)]*\)\s*\{"
    ref_sig = r"static void graph_refresh_input_tensors\([^)]*\)\s*\{"
    dec_sig = r"static bool graph_prestage_or_decline\([^)]*\)\s*\{"
    cmp_sig = r"static ggml_status ggml_backend_sycl_graph_compute_unchecked\([^)]*\)\s*\{"
    # Sites 3 and 4 sit in the legacy (prompt-phase) segmented branch, after the keyed slot's own decline (site 7).
    legacy_seg_sig = r"\}\s*else if \(segments_match\)\s*\{"
    # the first full-graph recording site (the debug line just above it is the anchor)
    full_sig = r"Pre-staging leaf tensors before recording[^;]*;"
    stage_sig = r"void \* graph_input_stage\(const ggml_tensor \* owner,[^)]*\)\s*\{"
    swap_sig = r"static bool graph_staging_swap_retire\(ggml_backend_sycl_context \* ctx\)\s*\{"
    dense_drop_sig = r"static void ggml_sycl_block_exec_dense_drop_graphs\(ggml_backend_sycl_context \* ctx\)\s*\{"
    clear_sig = r"static void sycl_exec_graph_clear_active\(ggml_backend_sycl_context \* ctx, const char \* reason\)\s*\{"
    drop_sig = r"void ggml_sycl_block_exec_dense_state::drop_graphs\(ggml_backend_sycl_context & ctx\)\s*\{"
    retire_sig = r"static bool moe_graph_retention_retire_exact\(ggml_backend_sycl_context \* ctx\)\s*\{"
    def move_first_record_decline(src):
        """Move the first-record decline block, with its debug string, to just after that recording begins."""
        m = re.search(r"GGML_SYCL_DEBUG\(\"\[SYCL-GRAPH\] Pre-staging leaf tensors before recording[^;]*;\s*"
                      r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{[^{}]*\}", src)
        if not m:
            print("FAIL: self-test anchor missing: first-record decline block")
            failed.append("self-test anchor first-record decline block")
            return src
        block = m.group(0)
        rest = src[:m.start()] + src[m.end():]
        b = re.compile(r"model_sycl_graph\.begin_recording\([^;]*;").search(rest, m.start())
        return rest[:b.end()] + "\n" + block + "\n" + rest[b.end():]

    def move_entry_block_into_later_if(src):
        """Move the compute entry's moe_graphs_disabled block inside the later `if (use_sycl_graph) {` consumer."""
        m = re.search(r"if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\}", src)
        if not m:
            print("FAIL: self-test anchor missing: compute entry block")
            failed.append("self-test anchor compute entry block")
            return src
        block = m.group(0)
        rest = src[:m.start()] + src[m.end():]
        k = rest.find("\n    if (use_sycl_graph) {\n", m.start())
        k2 = k + len("\n    if (use_sycl_graph) {\n")
        return rest[:k2] + block + "\n" + rest[k2:]

    mem_ = memo_hdr
    mutants = [
        # review r13
        ("the entry condition compares the flag to false", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"if \(sycl_ctx->moe_graphs_disabled\) \{(\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;)", r"if (sycl_ctx->moe_graphs_disabled == false) {\1"), common, mem_)),
        ("the entry condition is a less-than", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"if \(sycl_ctx->moe_graphs_disabled\) \{(\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;)", r"if (sycl_ctx->moe_graphs_disabled < 0) {\1"), common, mem_)),
        ("the entry block is the else arm of another if", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(\n    )(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG)", r'\1if (sycl_ctx->graphs_disabled) { GGML_SYCL_DEBUG("x\\n"); } else \2'), common, mem_)),
        ("the entry block is guarded by a braceless if", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(\n    )(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG)", r"\1if (cached_is_decode)\n    \2"), common, mem_)),
        ("the entry block is guarded by a braceless if (false)", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(\n    )(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG)", r"\1if (false)\n    \2"), common, mem_)),
        ("an #endif closes the graph section before the block", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"((if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\}))", r"#endif\n\1\n#ifdef GGML_SYCL_GRAPH"), common, mem_)),
        ("use_sycl_graph is assigned through std::tie", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"((if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\}))", r"\1\n    std::tie(use_sycl_graph) = std::make_tuple(true);"), common, mem_)),
        # review r12
        ("use_sycl_graph is re-enabled just before the descriptor candidates", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(\n\s*const int descriptor_moe_graph_candidates =)", r"\n    use_sycl_graph = true;\1"), common, mem_)),
        ("use_sycl_graph is or-assigned true after the block", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\})", r"\1\n    use_sycl_graph |= true;"), common, mem_)),
        ("the entry block is compiled away by #if 0", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\})", r"#if 0\n\1\n#endif"), common, mem_)),
        ("use_sycl_graph is re-enabled through a reference", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\})", r"\1\n    { bool & g = use_sycl_graph; g = true; }"), common, mem_)),
        # review r11
        ("the entry block is moved under the later use_sycl_graph test", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (move_entry_block_into_later_if(backend), common, mem_)),
        ("the entry block is followed by use_sycl_graph = true", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\})", r"\1\n    use_sycl_graph = true;"), common, mem_)),
        ("the entry block is followed by an or-assignment", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\})", r"\1\n    use_sycl_graph = use_sycl_graph || g_split_config.enabled;"), common, mem_)),
        ("the entry assignment is under a dead if", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"if \(sycl_ctx->moe_graphs_disabled\) \{(\s*GGML_SYCL_DEBUG\([^;]*;\s*)use_sycl_graph = false;", r"if (sycl_ctx->moe_graphs_disabled) {\1if (0) use_sycl_graph = false;"), common, mem_)),
        ("the entry assignment exists only inside a message", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;",
                    'if (sycl_ctx->moe_graphs_disabled) { GGML_SYCL_DEBUG("use_sycl_graph = false;\\n");'), common, mem_)),
        ("the entry block is wrapped in a dead if", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"(if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;\s*\})", r"if (false) { \1 }"), common, mem_)),
        # review r10
        ("the compute entry no longer turns graphs off", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"if \(sycl_ctx->moe_graphs_disabled\) \{\s*GGML_SYCL_DEBUG\([^;]*;\s*use_sycl_graph = false;",
                    'if (sycl_ctx->moe_graphs_disabled) {\n        GGML_SYCL_DEBUG("x\\n");'), common, mem_)),
        ("the compute entry's disabled block is dead", "the compute entry turns graphs off when MoE graphs were disabled by a failed retire",
         (mutate_re(backend, cmp_sig, r"if \(sycl_ctx->moe_graphs_disabled\) \{", "if (false) {"), common, mem_)),
        # review r9
        ("the displaced handle is moved into retain", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig, "retain_handles_until_event({ slot.handle }, retire)", "retain_handles_until_event({ std::move(slot.handle) }, retire)"), mem_)),
        ("the reuse test loses its size bound", "graph_input_stage reuses an entry only when it is large enough (capacity >= nbytes)",
         (backend, mutate_in_func(common, stage_sig, "it->second.capacity >= nbytes", "it->second.capacity > 0"), mem_)),
        ("the retire drain's catch reports success", "a failed drain or registry retire makes the epoch retire report false",
         (mutate_in_func(backend, retire_sig, "} catch (...) {\n        return false;", "} catch (...) {\n        return true;"), common, mem_)),
        ("the registry retire's failure is ignored", "a failed drain or registry retire makes the epoch retire report false",
         (mutate_in_func(backend, retire_sig, "rc != ggml_sycl::moe::retention_error::OK && rc != ggml_sycl::moe::retention_error::STALE", "false"), common, mem_)),
        ("the epoch wrapper's catch reports success", "a failed drain or registry retire makes the epoch retire report false",
         (mutate_in_func(backend, r"bool ggml_sycl_retire_moe_graph_epoch\([^)]*\) noexcept\s*\{", "} catch (...) {\n        return false;", "} catch (...) {\n        return true;"), common, mem_)),
        ("the direct-dispatch gate ignores its disabled flag", "every replay gate honors the disabled flag a failed retire sets",
         (mutate_re(backend, r"direct_moe_graphlet_probe\s*=", r"!sycl_ctx->moe_direct_dispatch_graphs_disabled\s*&&\s*", ""), common, mem_)),
        ("the sequence gate ignores moe_graphs_disabled", "every replay gate honors the disabled flag a failed retire sets",
         (mutate_re(backend, r"moe_graph_try_sequence_graphlet_for_node\(", r"sycl_ctx->moe_graphs_disabled\s*\|\|\s*(?=sycl_ctx->moe_sequence_graphs_disabled)", ""), common, mem_)),
        ("the block graphlet gate ignores moe_graphs_disabled", "every replay gate honors the disabled flag a failed retire sets",
         (mutate_re(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{", r"\|\|\s*sycl_ctx->moe_graphs_disabled\)\s*\{", ") {"), common, mem_)),
        # review r8
        ("the handle is moved before the event is taken", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig,
                                  "sycl::event retire = ggml_sycl_submit_marker<graph_input_staging_retire_marker>(q);\n"
                                  "            graph_input_staging_swapped = true;\n"
                                  "            ggml_sycl::retain_handles_until_event({ slot.handle }, retire);",
                                  "graph_input_staging_swapped = true;\n"
                                  "            ggml_sycl::retain_handles_until_event({ slot.handle }, ggml_sycl_submit_marker<graph_input_staging_retire_marker>(q));"), mem_)),
        ("the swap uses a bare barrier", "the staging swap takes its event from the marker helper before it moves the handle, never a bare barrier",
         (backend, mutate_in_func(common, stage_sig, "ggml_sycl_submit_marker<graph_input_staging_retire_marker>(q)", "q.ext_oneapi_submit_barrier()"), mem_)),
        ("the swap flags after it retains", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig,
                                  "graph_input_staging_swapped = true;\n"
                                  "            ggml_sycl::retain_handles_until_event({ slot.handle }, retire);",
                                  "ggml_sycl::retain_handles_until_event({ slot.handle }, retire);\n"
                                  "            graph_input_staging_swapped = true;"), mem_)),
        ("the dense drop hides in a dead brace", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ggml_sycl_block_exec_dense_drop_graphs(ctx);",
                         "if (false) { ggml_sycl_block_exec_dense_drop_graphs(ctx); }"), common, mem_)),
        ("the sequence retire hides in a conditional brace", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_sequence_graphs();",
                         "false; if (ctx->exec_graph) { ctx->invalidate_moe_sequence_graphs(); }"), common, mem_)),
        ("the swap retire waits on the host", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ggml_sycl_block_exec_dense_drop_graphs(ctx);",
                         "ggml_sycl_block_exec_dense_drop_graphs(ctx);\n    ctx->stream()->wait_and_throw();"), common, mem_)),
        ("the swap retire drains unconditionally", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "if (ctx->exec_graph) {",
                         "ggml_sycl_trace_queue_wait(ctx->stream(), \"staging-swapped\", ctx->device, -1, nullptr);\n    if (ctx->exec_graph) {"), common, mem_)),
        ("the swap retire releases the pool unconditionally", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "if (ctx->exec_graph) {", "sycl_exec_graph_release_pool_retained(ctx);\n    if (ctx->exec_graph) {"), common, mem_)),
        ("the swap retire resets the signature cache", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "if (ctx->exec_graph) {", "ctx->cached_graph_sig_n_nodes = -1;\n    if (ctx->exec_graph) {"), common, mem_)),
        ("the swap retire short-circuits its retires", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_block_graphs();",
                         "segments_retired && ctx->invalidate_moe_block_graphs();"), common, mem_)),
        ("the swap retire returns true unconditionally", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "return segments_retired", "return true || segments_retired"), common, mem_)),
        ("the INPUT arm stages on another queue", "the INPUT arm stages on the backend's own queue",
         (mutate_re(backend, pre_sig, r"sycl::queue & q\s*= \*ctx->stream\(\);", "sycl::queue & q = *ctx->stream(ctx->device, 1);"), common, mem_)),
        ("the gateway ignores a failed retire", "the gateway declines when the staging swap could not retire every recorder",
         (mutate_in_func(backend, dec_sig, "if (staged && retired) {", "if (staged) {"), common, mem_)),
        ("the gateway drops the retire result", "the gateway consumes the swapped flag and retires the recorders, with no second pass",
         (mutate_in_func(backend, dec_sig, "retired = graph_staging_swap_retire(ctx);", "graph_staging_swap_retire(ctx);"), common, mem_)),
        ("an invalidator stops reporting failure", "each MoE invalidator reports whether it retired the epoch (false on failure)",
         (backend, mutate_in_func(common, r"bool invalidate_moe_segments\(\)\s*\{", "return false;\n        }", "return true;\n        }"), mem_)),
        ("an invalidator stops reporting success", "each MoE invalidator reports whether it retired the epoch (false on failure)",
         (backend, mutate_in_func(common, r"bool invalidate_moe_sequence_graphs\(\)\s*\{", "return true;\n    }", "return false;\n    }"), mem_)),
        ("dense post-gateway resize forgets the cached-input reset", "the dense recorder re-sizes its range graphs after the gateway may have retired them",
         (mutate_in_func(backend, r"graph_prestage_decline_memo::dense_split_key\(key\)\)\)\s*\{[^}]*\}\s*if \(st\.graphs\.size\(\) != ranges_\.size\(\)\) \{",
                         "ctx_.input_tensors_cached = false;", "(void) 0;"), common, mem_)),
        ("flag-only INPUT test in pre-stage", "pre-stage has no flag-only INPUT test left",
         (mutate_in_func(backend, pre_sig, "graph_tensor_is_input(tensor)",
                         "(tensor->flags & GGML_TENSOR_FLAG_INPUT)"), common, mem_)),
        ("flag-only INPUT test in refresh", "refresh discovery has no flag-only INPUT test left",
         (mutate_in_func(backend, ref_sig, "graph_tensor_is_input(tensor)",
                         "(tensor->flags & GGML_TENSOR_FLAG_INPUT)"), common, mem_)),
        ("view walk dropped", "the INPUT test walks view_src",
         (mutate_in_func(backend, r"static bool graph_tensor_is_input\([^)]*\)\s*\{", "view_src", "view_XXXX"),
          common, mem_)),
        ("zero-byte failure restored", "a zero-byte tensor is a no-op, not a staging failure",
         (mutate_in_func(backend, pre_sig, "ggml_nbytes(tensor) == 0", "ggml_nbytes(tensor) == 12345"), common, mem_)),
        ("failure swallowed", "a staging failure is reported to the caller",
         (mutate_in_func(backend, pre_sig, "return all_staged", "return true"), common, mem_)),
        ("decline is silent", "the decline names itself at WARN",
         (mutate_in_func(backend, dec_sig, "GGML_LOG_WARN", "GGML_SYCL_DEBUG"), common, mem_)),
        ("decline not remembered", "a declined graph is remembered on the context, so the pass is not repeated per token",
         (mutate_in_func(backend, dec_sig, "prestage_decline_memo.remember(graph_hash, ctx->graph_compute_seq)", "(void) graph_hash"),
          common, mem_)),
        ("memo made thread-local", "the decline memo is not a thread-local or process-wide slot",
         (mutate_in_func(backend, dec_sig, "prestage_decline_memo.remember(graph_hash, ctx->graph_compute_seq)",
                         "static thread_local int slot; slot = (int) graph_hash"), common, mem_)),
        ("context loses the memo", "the context owns the decline memo",
         (backend, common.replace("graph_prestage_decline_memo prestage_decline_memo;", ""), mem_)),
        ("memo never forgets", "the memo forgets a decline after a bounded number of skips",
         (backend, common, re.sub(r"retry_after\s*=\s*\d+", "retry_never = 0", mem_))),
        ("memo holds one signature", "the memo holds several signatures at once",
         (backend, common, re.sub(r"max_entries\s*=\s*\d+", "one_slot = 1", mem_))),
        ("success keeps an old decline", "a successful pre-stage forgets an earlier decline",
         (mutate_in_func(backend, dec_sig, "prestage_decline_memo.forget(graph_hash)", "(void) graph_hash"),
          common, mem_)),
        ("recording site loses its decline", "every full-graph recording is preceded by a pre-stage that can decline it",
         (mutate_in_func(backend, full_sig, "graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)",
                         "graph_prestage_leaf_tensors(sycl_ctx, cgraph)"), common, mem_)),
        ("decline does not leave", "every full-graph recording is preceded by a pre-stage that can decline it",
         (mutate_in_func(backend, full_sig, "graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)",
                         "graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash) || true"), common, mem_)),
        ("block graphlets pre-stage as a statement",
         "no code pre-stages as a statement: graph_prestage_or_decline is the only caller of the pre-stage",
         (mutate_in_func(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{",
                         "graph_prestage_or_decline(", "graph_prestage_leaf_tensors("), common, mem_)),
        ("a new recorder pre-stages as a statement",
         "no code pre-stages as a statement: graph_prestage_or_decline is the only caller of the pre-stage",
         (backend + "\nstatic void new_recorder(ggml_backend_sycl_context * c, const ggml_cgraph * g) "
                    "{ graph_prestage_leaf_tensors(c, g); }\n", common, mem_)),
        ("segment replay ignores the result",
         "every pre-stage consumer tests the result (void-style calls ignore a failure)",
         (mutate_in_func(backend, cmp_sig, "if (!graph_prestage_or_decline(", "(void) (graph_prestage_or_decline("),
          common, mem_)),
        ("dense decline without its return",
         "site 1, dense split recorder: declines with STAGE_FAILED, drops its graphs and returns",
         (mutate_re(backend, r"void prepare_graphs\(\)\s*\{", r"st\.drop_graphs\(ctx_\);\s*return;",
                    "st.drop_graphs(ctx_);"), common, mem_)),
        ("dense key not namespaced",
         "site 1, dense split recorder: declines with STAGE_FAILED, drops its graphs and returns",
         (mutate_in_func(backend, r"void prepare_graphs\(\)\s*\{", "graph_prestage_decline_memo::dense_split_key(key)",
                         "key"), common, mem_)),
        ("memo drops its dense key helper", "the dense split key is namespaced, so one memo never mixes two key spaces",
         (backend, common, mem_.replace("dense_split_key", "dense_split_keyX"))),
        ("block graphlets drop their decline", "site 2, MoE block graphlets: declines, invalidates them and returns false to the direct fallback",
         (mutate_in_func(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{",
                         "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash))", "if (false)"), common, mem_)),
        ("block graphlet decline falls through",
         "site 2, MoE block graphlets: declines, invalidates them and returns false to the direct fallback",
         (mutate_in_func(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{",
                         "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) {",
                         "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { (void) 0;"
                         " } if (false) {"), common, mem_)),
        ("segment replay decline is dead", "site 3, MoE segment replay: declines, invalidates the segments, runs direct, else replays",
         (mutate_in_func(backend, legacy_seg_sig, "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) {",
                         "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash) && false) {"), common, mem_)),
        ("segment record decline loses its direct run", "site 4, MoE segment record: declines and runs direct, else records",
         (mutate_re(backend, legacy_seg_sig, r"\}\s*else if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{\s*compute_impl_unlocked\(\);",
                    "} else if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { (void) 0;"), common, mem_)),
        ("re-record decline is unconditional-tail", "site 5, full re-record: declines, clears the live graph (after a wait) and leaves for the direct path",
         (mutate_re(backend, r"re-record \+ update \(%s\)", r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{",
                    "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash) || true) {"), common, mem_)),
        ("first record decline does not leave", "site 6, full first record: declines and leaves for the direct path",
         (mutate_re(backend, full_sig, r"compute_impl_unlocked\(\);\s*record_completion\(false\);\s*return GGML_STATUS_SUCCESS;",
                    "compute_impl_unlocked();"), common, mem_)),
        ("re-record resets the exec graph before it decides",
         "re-record decides the decline before every teardown action and before counting the attempt",
         (mutate_re(backend, r"re-record \+ update \(%s\)", r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{\s*sycl_exec_graph_clear_active",
                    "sycl_ctx->exec_graph.reset(); if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { "
                    "sycl_exec_graph_clear_active"), common, mem_)),
        ("re-record releases the retained pool before it decides",
         "re-record decides the decline before every teardown action and before counting the attempt",
         (mutate_re(backend, r"re-record \+ update \(%s\)", r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{\s*sycl_exec_graph_clear_active",
                    "sycl_exec_graph_release_pool_retained(sycl_ctx); if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { "
                    "sycl_exec_graph_clear_active"), common, mem_)),
        ("re-record marks the graph inactive before it decides",
         "re-record decides the decline before every teardown action and before counting the attempt",
         (mutate_re(backend, r"re-record \+ update \(%s\)", r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{\s*sycl_exec_graph_clear_active",
                    "sycl_ctx->active_exec_graph.valid = false; if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { "
                    "sycl_exec_graph_clear_active"), common, mem_)),
        ("re-record counts the attempt before it decides",
         "re-record decides the decline before every teardown action and before counting the attempt",
         (mutate_re(backend, r"re-record \+ update \(%s\)", r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{\s*sycl_exec_graph_clear_active",
                    "g_graph_diag_counters.rerecord_attempts.fetch_add(1, std::memory_order_relaxed); "
                    "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { sycl_exec_graph_clear_active"),
          common, mem_)),
        ("re-record decline stops clearing the graph", "site 5, full re-record: declines, clears the live graph (after a wait) and leaves for the direct path",
         (mutate_in_func(backend, r"re-record \+ update \(%s\)", "sycl_exec_graph_clear_active(sycl_ctx, \"prestage-declined\");",
                         ""), common, mem_)),
        ("dense decline keeps its graphs", "site 1, dense split recorder: declines with STAGE_FAILED, drops its graphs and returns",
         (mutate_re(backend, r"void prepare_graphs\(\)\s*\{", r"st\.drop_graphs\(ctx_\);\s*return;", "return;"),
          common, mem_)),
        ("graphlet decline keeps its graphs", "site 2, MoE block graphlets: declines, invalidates them and returns false to the direct fallback",
         (mutate_re(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{",
                    r"if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{[^{}]*\}",
                    "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) { return false; }"), common, mem_)),
        ("segment replay decline keeps its segments", "site 3, MoE segment replay: declines, invalidates the segments, runs direct, else replays",
         (mutate_re(backend, cmp_sig, r"sycl_ctx->invalidate_moe_segments\(\);\s*compute_impl_unlocked\(\);\s*\} else if \(!sycl_ctx->moe_segments_valid\)",
                    "compute_impl_unlocked();\n } else if (!sycl_ctx->moe_segments_valid)"), common, mem_)),
        ("first-record decline moves past begin_recording",
         "the recording sites keep their order, each decline between its neighbours (and before its recording)",
         (move_first_record_decline(backend), common, mem_)),
        ("memo ignores the token number", "the memo counts a signature once per token, however many sites ask",
         (backend, common, mem_.replace("last_seq", "last_sXq"))),
        ("context stops numbering its computes", "the context numbers its graph computes, so the memo can tell tokens apart",
         (mutate_in_func(backend, cmp_sig, "++sycl_ctx->graph_compute_seq", "(void) sycl_ctx"), common, mem_)),
        # staging swap (review r6)
        ("drop-then-allocate returns", "graph_input_stage allocates the replacement before it touches the entry (no drop-then-allocate)",
         (backend, mutate_in_func(common, stage_sig, "ggml_sycl::alloc_request req{};",
                                  "if (it != graph_input_staging.end()) { it->second.handle = ggml_sycl::mem_handle{}; }\n"
                                  "        ggml_sycl::alloc_request req{};"), mem_)),
        ("an INPUT staging failure falls through", "a staging failure for an INPUT tensor is terminal for the pass (no fallthrough to the cache paths)",
         (mutate_re(backend, pre_sig, r"if \(!dev_ptr\) \{", "if (false) {"), common, mem_)),
        ("the gateway ignores the swapped flag", "the gateway consumes the swapped flag and retires the recorders, with no second pass",
         (mutate_in_func(backend, dec_sig, "if (ctx->graph_input_staging_swapped) {", "if (false) {"), common, mem_)),
        ("the swap retire forgets dense", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ggml_sycl_block_exec_dense_drop_graphs(ctx);", ""), common, mem_)),
        ("the swap retire forgets the direct-dispatch graphs", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_direct_dispatch_graphs();", ""), common, mem_)),
        ("the swap retire forgets the sequence graphs", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_sequence_graphs();", ""), common, mem_)),
        ("the dense drop helper creates a state", "the dense drop helper drops a state that exists and does not create one",
         (mutate_in_func(backend, dense_drop_sig, "find(ctx)", "find(ctx); (void) ggml_sycl_block_exec_dense_state_for(*ctx)"), common, mem_)),
        ("dense does not re-size after the gateway", "the dense recorder re-sizes its range graphs after the gateway may have retired them",
         (mutate(backend, "if (st.graphs.size() != ranges_.size()) {", "if (false) {"), common, mem_)),
        # staging swap (review r6, r7)
        ("the displaced handle is freed at the swap", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig, "ggml_sycl::retain_handles_until_event({ slot.handle }, retire);",
                                  "slot.handle = ggml_sycl::mem_handle{};"), mem_)),
        ("the displaced handle is retained on an empty event", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig, "ggml_sycl_submit_marker<graph_input_staging_retire_marker>(q)", "sycl::event{}"), mem_)),
        ("the swap is not flagged", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig, "graph_input_staging_swapped = true;", "(void) slot;"), mem_)),
        ("the swap is flagged for a new entry too", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig, "if (slot.handle.valid()) {", "if (true) {"), mem_)),
        ("the swap flags a first publication", "the staging map's only mutation sites are the swap and the clear, and the swap waits on nothing",
         (backend, mutate_in_func(common, stage_sig, "slot.capacity = nbytes;", "slot.capacity = nbytes;\n        graph_input_staging_swapped = true;"), mem_)),
        ("the generation is not bumped at the swap", "the displaced staging handle is retained until a marker event, flagged, and the generation bumped",
         (backend, mutate_in_func(common, stage_sig, "slot.capacity = nbytes;\n        graph_input_staging_generation++;", "slot.capacity = nbytes;"), mem_)),
        ("the swap waits on the host", "the staging map's only mutation sites are the swap and the clear, and the swap waits on nothing",
         (backend, mutate_in_func(common, stage_sig, "slot.capacity = nbytes;", "slot.capacity = nbytes;\n        q.wait();"), mem_)),
        ("the entry is published before the resolve check", "graph_input_stage publishes into the map only after its last failure return",
         (backend, mutate_re(common, stage_sig, r"if \(!resolved\.ptr \|\| !resolved\.on_device\) \{\s*return nullptr;",
                             "graph_input_staging[owner].capacity = nbytes;\n        if (!resolved.ptr || !resolved.on_device) {\n            return nullptr;"), mem_)),
        ("clear keeps the swapped flag", "the context owns the swapped flag, and clear resets it with the map",
         (backend, mutate(common, "graph_input_staging_swapped = false;\n        graph_input_staging_generation++;", "graph_input_staging_generation++;"), mem_)),
        ("the staging map is released before the wait", "clear_active waits on the queue before it resets the exec graph or releases the staging",
         (mutate_in_func(backend, clear_sig, "if (ctx->exec_graph) {", "ctx->graph_input_staging_clear(*ctx->stream());\n    if (ctx->exec_graph) {"), common, mem_)),
        ("the INPUT condition is dead", "the INPUT arm tests exactly 'is input, named' and its success path returns",
         (mutate_re(backend, pre_sig, r"if \(graph_tensor_is_input\(tensor\) && tensor->name && tensor->name\[0\] != '\\0'\) \{",
                    "if (graph_tensor_is_input(tensor) && tensor->name && tensor->name[0] != '\\0' && false) {"), common, mem_)),
        ("the INPUT success falls through", "the INPUT arm tests exactly 'is input, named' and its success path returns",
         (mutate_re(backend, pre_sig, r"(\(long long\) tensor->name, tensor->data, nbytes, dev_ptr\);|tensor->name, tensor->data, nbytes, dev_ptr\);)\s*return;",
                    "tensor->name, tensor->data, nbytes, dev_ptr);"), common, mem_)),
        ("the gateway retires only when the pass staged", "the gateway consumes the swapped flag and retires the recorders, with no second pass",
         (mutate_in_func(backend, dec_sig, "if (ctx->graph_input_staging_swapped) {", "if (staged && ctx->graph_input_staging_swapped) {"), common, mem_)),
        ("the gateway pre-stages a second time", "the gateway consumes the swapped flag and retires the recorders, with no second pass",
         (mutate_in_func(backend, dec_sig, "graph_staging_swap_retire(ctx);", "graph_staging_swap_retire(ctx);\n        staged = staged && graph_prestage_leaf_tensors(ctx, cgraph);"), common, mem_)),
        ("the swap retire calls the whole clear_active", "the gateway and the swap retire leave leases, pins, the CPU staging cache and the MoE layout cache alone",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_sequence_graphs();", "ctx->invalidate_moe_sequence_graphs();\n    sycl_exec_graph_clear_active(ctx, \"staging-swapped\");"), common, mem_)),
        ("the swap retire unpins the weights", "the gateway and the swap retire leave leases, pins, the CPU staging cache and the MoE layout cache alone",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_sequence_graphs();", "ctx->invalidate_moe_sequence_graphs();\n    graph_unpin_weights(ctx);"), common, mem_)),
        ("the swap retire releases the staging map", "the gateway and the swap retire leave leases, pins, the CPU staging cache and the MoE layout cache alone",
         (mutate_in_func(backend, swap_sig, "ctx->graph_input_staging_swapped = false;", "ctx->graph_input_staging_swapped = false;\n    ctx->graph_input_staging_clear(*ctx->stream());"), common, mem_)),
        ("the swap retire's exec-graph reset loses its drain", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ggml_sycl_trace_queue_wait(ctx->stream(), \"staging-swapped\", ctx->device, -1, nullptr);", ""), common, mem_)),
        ("the swap retire's exec-graph reset is unconditional", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "if (ctx->exec_graph) {", "if (true) {"), common, mem_)),
        ("the swap retire's segments invalidate is dead", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ctx->invalidate_moe_segments();", "if (false) ctx->invalidate_moe_segments();"), common, mem_)),
        ("the swap retire's dense drop is dead", "the swap retire is exactly: flag, dense drop, four MoE retires, a drained live-exec reset, and their conjunction",
         (mutate_in_func(backend, swap_sig, "ggml_sycl_block_exec_dense_drop_graphs(ctx);", "if (false) ggml_sycl_block_exec_dense_drop_graphs(ctx);"), common, mem_)),
        ("clear_active's wait guard goes dead", "clear_active waits on the queue before it resets the exec graph or releases the staging",
         (mutate_in_func(backend, clear_sig, "if (ctx->exec_graph) {", "if (false) {"), common, mem_)),
        # waits before releases (review r6)
        ("clear_active loses its wait", "clear_active waits on the queue before it resets the exec graph or releases the staging",
         (mutate_in_func(backend, clear_sig, "ggml_sycl_trace_queue_wait(ctx->stream(), reason ? reason : \"exec-graph-clear\", ctx->device, -1, nullptr);", ""), common, mem_)),
        ("drop_graphs loses its drain", "drop_graphs drains each used device before it clears the range graphs",
         (mutate_in_func(backend, drop_sig, "ggml_sycl_block_exec_dense_queue(ctx, d)->wait_and_throw();", "(void) d;"), common, mem_)),
        ("drop_graphs' drain guard goes dead", "drop_graphs drains each used device before it clears the range graphs",
         (mutate_in_func(backend, drop_sig, "if (!used[d]) {", "if (true) {"), common, mem_)),
        ("retire_exact loses its drain", "retire_exact drains every terminal's queue before it retires the epoch",
         (mutate_in_func(backend, retire_sig, "ctx->stream(device, 0)->wait_and_throw();", "(void) device;"), common, mem_)),
        ("retire_exact's drain loop goes empty", "retire_exact drains every terminal's queue before it retires the epoch",
         (mutate_in_func(backend, retire_sig, "for (const auto & [device, _] : ctx->moe_retention_terminals)",
                         "for (const auto & [device, _] : decltype(ctx->moe_retention_terminals){})"), common, mem_)),
        # review r6 minors
        ("the graphlet invalidate sits under a dead if", "site 2, MoE block graphlets: declines, invalidates them and returns false to the direct fallback",
         (mutate_re(backend, r"static bool moe_graph_try_block_graphlets\([^)]*\)\s*\{", r"\"stage-failed\"\);\s*sycl_ctx->invalidate_moe_block_graphs\(\);",
                    "\"stage-failed\"); if (false) sycl_ctx->invalidate_moe_block_graphs();"), common, mem_)),
        ("the token number is bumped under a condition", "the token number is bumped unconditionally, at top level, before any early return",
         (mutate_in_func(backend, cmp_sig, "++sycl_ctx->graph_compute_seq;", "if (sycl_ctx->exec_graph) ++sycl_ctx->graph_compute_seq;"), common, mem_)),
        ("the token number is bumped in a nested block", "the token number is bumped unconditionally, at top level, before any early return",
         (mutate_in_func(backend, cmp_sig, "++sycl_ctx->graph_compute_seq;", "if (sycl_ctx->device >= 0) {\n ++sycl_ctx->graph_compute_seq;\n }"), common, mem_)),
        ("the token number is bumped after an early return", "the token number is bumped unconditionally, at top level, before any early return",
         (mutate_in_func(backend, cmp_sig, "++sycl_ctx->graph_compute_seq;", "if (cgraph->n_nodes == 0) { return GGML_STATUS_SUCCESS; }\n ++sycl_ctx->graph_compute_seq;"), common, mem_)),
        ("segments retired by a swap are replayed", "site 3: segments a staging swap retired run the token direct instead of replaying nothing",
         (mutate_in_func(backend, cmp_sig, "else if (!sycl_ctx->moe_segments_valid) {", "else if (false) {"), common, mem_)),
        ("the retired-segments branch is gone", "site 3: segments a staging swap retired run the token direct instead of replaying nothing",
         (mutate_re(backend, cmp_sig, r"\s*else if \(!sycl_ctx->moe_segments_valid\) \{\s*compute_impl_unlocked\(\);\s*\}", ""), common, mem_)),
        ("the keyed slot replays a slot a staging swap retired", "site 7, keyed segment slot: declines and runs direct, else replays or records a slot that still exists",
         (mutate_in_func(backend, cmp_sig, "if (!slot) {", "if (false) {"), common, mem_)),
        ("the keyed slot records a slot a staging swap retired", "site 7, keyed segment slot: declines and runs direct, else replays or records a slot that still exists",
         (mutate_in_func(backend, cmp_sig, "} else if (!sycl_ctx->moe_segment_slots.state(slot_key)) {", "} else if (false) {"), common, mem_)),
        ("the keyed slot records past a decline", "site 7, keyed segment slot: declines and runs direct, else replays or records a slot that still exists",
         (mutate_in_func(backend, r"slot_action == gsc::action::DIRECT\)", "} else if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) {",
                         "} else if (false) {"), common, mem_)),
        ("an eighth recorder appears", "no unnamed pre-stage consumer: the seven recording sites below are all there are",
         (backend + "\nstatic bool new_recorder(ggml_backend_sycl_context * c, const ggml_cgraph * g) "
                    "{ if (!graph_prestage_or_decline(c, g, 1)) { return false; } return true; }\n", common, mem_)),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
