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

A declined pre-stage must not leave a recorded graph that points at a staging buffer it may have replaced
(review r5). graph_input_stage drops an input's staging handle before it allocates the replacement, so a
failure partway through can leave a live exec graph, MoE segments, block graphlets or dense range graphs
with a baked pointer to freed memory. The only owner of that handle is the per-context staging map (it is not
in graph_retained_handles); it is cleared by sycl_exec_graph_clear_active after a queue wait. So each decline
drops the recorded state it could have invalidated: clear_active (which waits) for the exec graph, and the
epoch-retiring invalidators the MoE and dense paths already use.

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
        "prestage_decline_memo.remember(graph_hash)" in decline and \
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
        re.search(r"graph_compute_seq\s*;", common) is not None and \
        re.search(r"(\+\+\s*sycl_ctx->graph_compute_seq|sycl_ctx->graph_compute_seq\s*\+\+)", compute) is not None and \
        0 <= re.search(r"graph_compute_seq", compute).start() < compute.find("graph_prestage_skip_declined(")
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
    results["no unnamed pre-stage consumer: the six recording sites below are all there are"] = len(consumers) == 6

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
        r"if\s*\(!" + call % r"sycl_ctx,\s*cgraph,\s*graph_hash" + r"\)\s*\{[^{}]*invalidate_moe_block_graphs\(\);[^{}]*\breturn false;\s*\}",
        graphlets) is not None
    site = r"if\s*\(!" + call % r"sycl_ctx,\s*cgraph,\s*graph_hash" + r"\)\s*\{\s*"
    results["site 3, MoE segment replay: declines, invalidates the segments, runs direct, else replays"] = re.search(
        site + r"sycl_ctx->invalidate_moe_segments\(\);\s*compute_impl_unlocked\(\);\s*\}\s*else\s*\{\s*"
        r"graph_refresh_input_tensors\(sycl_ctx, cgraph\);\s*moe_graph_replay_segments\(", compute) is not None
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
    first_call = compute.find("graph_prestage_or_decline(", first_at) if first_at >= 0 else -1
    chain = [("segment replay", s3.start() if s3 else -1), ("segment record", s4.start() if s4 else -1),
             ("re-record anchor", rerecord_at), ("re-record decline", rr_call),
             ("re-record begin_recording", begin[0] if len(begin) == 2 else -1),
             ("first-record anchor", first_at), ("first-record decline", first_call),
             ("first-record begin_recording", begin[1] if len(begin) == 2 else -1)]
    results["the recording sites keep their order, each decline between its neighbours (and before its recording)"] = \
        all(at >= 0 for _, at in chain) and all(chain[i][1] < chain[i + 1][1] for i in range(len(chain) - 1))
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
    # the first full-graph recording site (the debug line just above it is the anchor)
    full_sig = r"Pre-staging leaf tensors before recording[^;]*;"
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

    mem_ = memo_hdr
    mutants = [
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
         (mutate_in_func(backend, dec_sig, "prestage_decline_memo.remember(graph_hash)", "(void) graph_hash"),
          common, mem_)),
        ("memo made thread-local", "the decline memo is not a thread-local or process-wide slot",
         (mutate_in_func(backend, dec_sig, "prestage_decline_memo.remember(graph_hash)",
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
         (mutate_re(backend, r"void prepare_graphs\(\)\s*\{", r"DENSE_GRAPH_OFF_STAGE_FAILED;\s*return;",
                    "DENSE_GRAPH_OFF_STAGE_FAILED;"), common, mem_)),
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
         (mutate_in_func(backend, cmp_sig, "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash)) {",
                         "if (!graph_prestage_or_decline(sycl_ctx, cgraph, graph_hash) && false) {"), common, mem_)),
        ("segment record decline loses its direct run", "site 4, MoE segment record: declines and runs direct, else records",
         (mutate_re(backend, cmp_sig, r"\}\s*else if \(!graph_prestage_or_decline\(sycl_ctx, cgraph, graph_hash\)\) \{\s*compute_impl_unlocked\(\);",
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
         (mutate_re(backend, cmp_sig, r"sycl_ctx->invalidate_moe_segments\(\);\s*compute_impl_unlocked\(\);\s*\} else \{\s*graph_refresh_input_tensors",
                    "compute_impl_unlocked();\n } else {\n graph_refresh_input_tensors"), common, mem_)),
        ("first-record decline moves past begin_recording",
         "the recording sites keep their order, each decline between its neighbours (and before its recording)",
         (move_first_record_decline(backend), common, mem_)),
        ("memo ignores the token number", "the memo counts a signature once per token, however many sites ask",
         (backend, common, mem_.replace("last_seq", "last_sXq"))),
        ("context stops numbering its computes", "the context numbers its graph computes, so the memo can tell tokens apart",
         (mutate_in_func(backend, cmp_sig, "++sycl_ctx->graph_compute_seq", "(void) sycl_ctx"), common, mem_)),
        ("a seventh recorder appears", "no unnamed pre-stage consumer: the six recording sites below are all there are",
         (backend + "\nstatic bool new_recorder(ggml_backend_sycl_context * c, const ggml_cgraph * g) "
                    "{ if (!graph_prestage_or_decline(c, g, 1)) { return false; } return true; }\n", common, mem_)),
    ]
    for label, expect, sources in mutants:
        failed += run(label, sources, expect)

if failed:
    print("\nFAILED: " + ", ".join(failed))
    sys.exit(1)
print("\nALL CHECKS PASSED")
