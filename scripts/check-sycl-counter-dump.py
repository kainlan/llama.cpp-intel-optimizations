#!/usr/bin/env python3
"""H13: the SYCL counter dump exists in the shipped library.

GGML_SYCL_PRIVATE_TESTING is defined only on test targets, never on the ggml-sycl
target. A counter, a table or a printer behind that macro therefore never runs in
llama-cli, llama-completion, llama-server or llama-bench, and a vehicle arm that
scores a [SYCL-COUNTER] line from one of those tools is VOID by construction. This
gate keeps the dump out of that state. It is pure Python over source text and
allocates nothing, so it is safe at any ctest parallelism.

It checks, with comments stripped, over ggml/src/ggml-sycl excluding tests/:

  * unified_cache_test_counter_dump is defined once and registered with std::atexit
    once, and prints the three format strings of the dump contract;
  * the table is the per-tree expected set: the registered counters and snapshot
    entries equal the lists below, in order, minus the entries retired on that tree;
  * neither the printer, nor its atexit registration, nor the table, nor a registered
    counter's definition or increment, nor a _for_testing accessor that reads one,
    lies in a branch that is not compiled when GGML_SYCL_PRIVATE_TESTING is undefined;
  * the ggml-sycl target is given no GGML_SYCL_PRIVATE_TESTING definition;
  * the arena-active mirror the raw exit reads is set where an arena is created and
    cleared where it is destroyed and where it is abandoned;
  * the zone chokepoint, and every call of it, reads none of the zone allocator's
    free-space figures (zone_available, zone_largest_free), which need a mutex a
    refusal does not hold.

The region scanner has a positive control: it must find at least one
GGML_SYCL_PRIVATE_TESTING region in unified-cache.cpp, else the gate is VOID, because
a scanner that finds no region passes everything.

Usage:
  check-sycl-counter-dump.py [--root DIR]     check the tree, then run the mutation matrix
  check-sycl-counter-dump.py --no-mutations   check the tree only
  check-sycl-counter-dump.py --mutations-only run the mutation matrix only

The mutation matrix applies each RED to the tree's text in memory and requires the
gate to fail with that RED's message, so a check that can no longer fail is caught.
"""

import argparse
import os
import re
import sys

MACRO = "GGML_SYCL_PRIVATE_TESTING"

# The dump's fixed field list, in print order, with the step that lands each entry's
# producer. A field is registered from step 0 whether or not its producer has landed;
# "lands" only decides whether a reading of it is VOID. "retired" is the step after
# which the field is absent (none of the fields themselves retire; the interim keys of
# onednn_graph_route_declined do, see ROUTE_KEYS).
#
# refusal_unattributed lands at step 3, so it has no producer before then and a reading of
# it on a tree earlier than step 3 is VOID, not 0: its definition (a refusal on an unmarked
# thread during compute) needs the per-context marker that lands with step 3, and a step-0
# producer would count every refusal.
FIELDS = [
    ("ext_alloc_count", "step0", None),
    ("ext_alloc_arena", "step0", None),
    ("zone_cascade_miss", "step0", None),
    ("zone_unconverted_miss", "step0", None),
    ("zone_plan_refusal", "step0", None),
    ("refusal_unattributed", "step3", None),
    ("refusal_late", "step3", None),
    ("onednn_scratchpad_over_plan_declined", "step3", None),
    ("late_term_shrink_admitted", "step4", None),
    ("arena_policy_refusal", "step5", None),
    ("stream_dma_non_device_arrivals", "step0", None),
    ("onednn_pp_record_mode_acquires", "step0", None),
    ("set_rows_stage_arrivals", "step0", None),
    ("set_rows_stage_record_mode_acquires", "step0", None),
    ("load_row_op_time_arrivals", "step0", None),
    ("onednn_sdpa_admitted", "step0", None),
    ("onednn_sdpa_executed", "step0", None),
    ("onednn_sdpa_fallback_after_admit", "step0", None),
    ("moe_table_reach_zero_gpu_expert", "step0", None),
    ("onednn_graph_compile_live_draws", "step0", None),
    ("onednn_graph_callback_unmarked_mallocs", "step3", None),
    ("onednn_fa_plan_calls", "b1", None),
    ("onednn_graph_mask_declined", "b1", None),
    ("onednn_graph_route_declined", "b1", None),
    ("onednn_graph_decline_at_entry", "b1", None),
    ("onednn_graph_scratch_barrier_failed", "b2", None),
]

# The snapshot list: (id, printed name, lands, retired).
SNAPSHOTS = [
    ("zone_available_weight_first_decode", "zone_available{WEIGHT}@first_decode", "step0", None),
    ("zone_largest_free_weight_first_decode", "zone_largest_free{WEIGHT}@first_decode", "step0", None),
    ("zone_available_runtime_context_txn", "zone_available{RUNTIME}@context_txn", "step0", None),
    ("zone_largest_free_runtime_context_txn", "zone_largest_free{RUNTIME}@context_txn", "step0", None),
    ("zone_capacity_onednn_context_txn", "zone_capacity{ONEDNN}@context_txn", "step0", None),
    ("onednn_pp_a_bytes_context_txn", "onednn_pp_a_bytes@context_txn", "step4", None),
    ("weight_host_tiered_bytes_load_1", "weight_host_tiered_bytes{load_1}@last_load_end", "step0", None),
    ("weight_host_tiered_bytes_load_2", "weight_host_tiered_bytes{load_2}@last_load_end", "step0", None),
    ("weight_planned_device_bytes_load_1", "weight_planned_device_bytes{load_1}@last_load_end", "step0", None),
    ("weight_planned_device_bytes_load_2", "weight_planned_device_bytes{load_2}@last_load_end", "step0", None),
    ("weight_live_bytes_last_load_end", "weight_live_bytes@last_load_end", "step0", None),
]

# The keys of onednn_graph_route_declined: the three interim keys land in (b1) and retire
# in (b2), which deletes their producer; the four (b2) keys land in (b2).
ROUTE_COUNTER = "onednn_graph_route_declined"
ROUTE_INTERIM_KEYS = ["interim_tp", "interim_capped", "interim_capacity"]

# Counters whose producer is on the tree at step 0 and must have a compiled increment.
REQUIRED_PRODUCERS = ["ext_alloc_count", "ext_alloc_arena", "zone_cascade_miss", "zone_unconverted_miss",
                      "zone_plan_refusal", "stream_dma_non_device_arrivals", "onednn_pp_record_mode_acquires",
                      "set_rows_stage_arrivals", "set_rows_stage_record_mode_acquires", "onednn_sdpa_admitted",
                      "onednn_sdpa_executed", "onednn_sdpa_fallback_after_admit", "moe_table_reach_zero_gpu_expert",
                      "onednn_graph_compile_live_draws"]

# The G0 counters whose increment must sit in a named function, so a producer that drifted into an
# unrelated function (or a second copy of the site) fails the gate rather than reading as live:
# (counter, file under ggml/src/ggml-sycl, qualified function name).
PRODUCER_SITES = [
    ("stream_dma_non_device_arrivals", "unified-cache.cpp", "unified_cache::stream_dma"),
    ("onednn_pp_record_mode_acquires", "ggml-sycl.cpp", "acquire_onednn_pp_scratch"),
    ("set_rows_stage_arrivals", "set_rows.cpp", "ggml_sycl_set_rows_stage_ptr"),
    ("set_rows_stage_record_mode_acquires", "set_rows.cpp", "ggml_sycl_set_rows_stage_ptr"),
    ("onednn_sdpa_admitted", "fattn-onednn.cpp", "ggml_sycl_flash_attn_ext_onednn"),
    ("onednn_sdpa_executed", "fattn-onednn.cpp", "ggml_sycl_flash_attn_ext_onednn"),
    ("onednn_sdpa_fallback_after_admit", "fattn-onednn.cpp", "ggml_sycl_flash_attn_ext_onednn"),
    ("moe_table_reach_zero_gpu_expert", "ggml-sycl.cpp", "ggml_sycl_note_moe_table_reach"),
    ("onednn_graph_compile_live_draws", "fattn-onednn.cpp", "build_and_compile_sdpa"),
]

# The one counter whose evaluation is armed-only: its zero test walks the placement plan once per expert,
# per MoE table call, so it runs only in a run that arms the dump (the one place the counter is read).
# Every other producer counts unconditionally (§M179: a counter behind the environment made S0a's raw-exit
# count read 0 on a run that set neither variable).
ARMED_ONLY_COUNTERS = ["moe_table_reach_zero_gpu_expert"]

# What makes an `if` an environment or build test, for the unconditional-counting rule.
ENV_TEST = re.compile(r"getenv|counter_dump_requested|dump_report_enabled|dump_report_armed|PRIVATE_TESTING"
                      r"|g_ggml_sycl_debug|trace_enabled")

# G0's row-134 reach counts: every production call of ggml_sycl_ensure_moe_ptr_table, and every caller of
# ggml_sycl_update_moe_ptr_table / ggml_sycl_upload_moe_ptr_table_from_batch, notes its site before the
# call: (function that holds it, the site label it passes). A label with a `/role` suffix names a call
# inside that function; the function is what precedes the slash.
REACH_SITES = [
    ("ggml_sycl_upload_moe_transient_ptr_table", "ensure:ggml_sycl_upload_moe_transient_ptr_table"),
    ("ggml_sycl_update_moe_ptr_table", "ensure:ggml_sycl_update_moe_ptr_table"),
    # 7pm2 Stage A moved the preload's body into graph_preload_moe_experts_impl (the wrapper reports a failure);
    # the notes moved with it and keep their labels, which the dump keys and the docs name.
    ("graph_preload_moe_experts_impl", "ensure:graph_preload_moe_experts"),
    ("graph_preload_moe_experts_impl", "update:graph_preload_moe_experts"),
    ("ggml_sycl_upload_moe_retained_ptr_table_from_batch", "upload:ggml_sycl_upload_moe_retained_ptr_table_from_batch"),
    ("try_xmx_sorted_moe", "upload:try_xmx_sorted_moe"),
    ("ggml_sycl_mul_mat_id", "upload:ggml_sycl_mul_mat_id/prompt"),
    ("ggml_sycl_mul_mat_id", "upload:ggml_sycl_mul_mat_id/retained"),
    ("ggml_sycl_mul_mat_id", "upload:ggml_sycl_mul_mat_id/retained_local"),
    ("ggml_sycl_mul_mat_id", "upload:ggml_sycl_mul_mat_id/retained_tail"),
    ("ggml_sycl_mul_mat_id", "update:ggml_sycl_mul_mat_id/pair_gate"),
    ("ggml_sycl_mul_mat_id", "update:ggml_sycl_mul_mat_id/pair_up"),
    ("ggml_sycl_mul_mat_id", "update:ggml_sycl_mul_mat_id/pair_down"),
]

# The functions that emit a G0 report or the dump: each writes with std::fprintf(stderr, ...) and calls no
# GGML_LOG_* macro. The log drops GGML_LOG_INFO at default verbosity in every tool, so a report that went
# through it would vanish from a plain llama-completion run and G0 would read as vacuous.
REPORT_EMITTERS = [
    ("unified-cache.cpp", "unified_cache_dump_report"),
    ("unified-cache.cpp", "unified_cache_dump_report_once"),
    ("unified-cache.cpp", "dump_report_zone_figures_for"),
    ("unified-cache.cpp", "unified_cache_test_counter_dump"),
    ("unified-cache.cpp", "unified_cache_dump_capture_load_end"),
]

# The snapshot entries' producers: id -> (file, function that holds the capture). The two zone-figure
# points share one capture function that takes the point as a parameter and so names every entry; the
# load-end entries share the load-end capture. onednn_pp_a_bytes lands at step 4 and has no producer.
SNAPSHOT_SITES = {
    "zone_available_weight_first_decode": ("unified-cache.cpp", "unified_cache_dump_capture_zone_figures"),
    "zone_largest_free_weight_first_decode": ("unified-cache.cpp", "unified_cache_dump_capture_zone_figures"),
    "zone_available_runtime_context_txn": ("unified-cache.cpp", "unified_cache_dump_capture_zone_figures"),
    "zone_largest_free_runtime_context_txn": ("unified-cache.cpp", "unified_cache_dump_capture_zone_figures"),
    "zone_capacity_onednn_context_txn": ("unified-cache.cpp", "unified_cache_dump_capture_zone_figures"),
    "weight_host_tiered_bytes_load_1": ("unified-cache.cpp", "unified_cache_dump_capture_load_end"),
    "weight_host_tiered_bytes_load_2": ("unified-cache.cpp", "unified_cache_dump_capture_load_end"),
    "weight_planned_device_bytes_load_1": ("unified-cache.cpp", "unified_cache_dump_capture_load_end"),
    "weight_planned_device_bytes_load_2": ("unified-cache.cpp", "unified_cache_dump_capture_load_end"),
    "weight_live_bytes_last_load_end": ("unified-cache.cpp", "unified_cache_dump_capture_load_end"),
}

# Where each capture function is called from: (file, enclosing function, text the call carries). A
# capture that nothing calls prints not_captured on every run, which G0 would read as VOID, not as a bug.
CAPTURE_CALLERS = [
    ("ggml-sycl.cpp", "ggml_sycl_run_runtime_context_transaction", "dump_point::CONTEXT_TXN"),
    ("ggml-sycl.cpp", "ggml_backend_sycl_graph_compute_unchecked", "dump_point::FIRST_DECODE"),
    ("unified-cache.cpp", "unified_cache_note_model_load_end", "unified_cache_dump_capture_load_end"),
]

# The G0 report lines: (kind, file, function, text that must appear in the function). A report is a
# line at its instant, not a table entry, so the gate pins that its producer exists and says its kind.
REPORT_SITES = [
    ("landing", "ggml-sycl.cpp", "ggml_backend_sycl_buffer_publish", '"landing dev='),
    ("landing", "ggml-sycl.cpp", "ggml_backend_sycl_buffer_publish", "vram_zone_id::RUNTIME"),
    ("landing", "ggml-sycl.cpp", "ggml_backend_sycl_buffer_publish", "vram_zone_id::SCRATCH"),
    ("zone_figures", "unified-cache.cpp", "dump_report_zone_figures_for", "[SYCL-REPORT] zone_figures dev="),
    ("zone_figures", "unified-cache.cpp", "unified_cache_dump_capture_load_end", "vram_zone_id::ONEDNN"),
    ("planned_host", "unified-cache.cpp", "unified_cache_dump_capture_load_end", '"planned_host load_txn='),
    ("moe_zero_gpu_expert_tensors", "unified-cache.cpp", "unified_cache_dump_capture_load_end",
     "[SYCL-REPORT] moe_zero_gpu_expert_tensors dev="),
    ("arm_a_kernel", "ggml-sycl.cpp", "bool ggml_sycl_dispatch_mul_mat_kernel", "layout_mismatch="),
    ("arm_a_kernel", "ggml-sycl.cpp", "bool ggml_sycl_dispatch_mul_mat_kernel", "src0_layout="),
    ("row73_own_alloc", "ggml-sycl.cpp", "bool ggml_sycl_ensure_moe_ptr_table", '"row73_own_alloc dev='),
    ("arm_a_kernel", "ggml-sycl.cpp", "bool ggml_sycl_dispatch_mul_mat_kernel", '"arm_a_kernel dev='),
]

# A field whose producer cannot land yet prints value=not_captured, never a zero: its definition
# needs the request's pending_owner (a load's request always carries its hold's owner), which lands
# with moua L4L6 / 23mk S4a. The printer reads one predicate for the set, and the gate pins both.
NOT_CAPTURED_FIELDS = ["load_row_op_time_arrivals", "onednn_graph_callback_unmarked_mallocs"]

# The tree's own facts: which landing steps are on it.
B2_SENTINEL = re.compile(r"\bggml_sycl_onednn_graph_scratch_range_miss\s*\([^;{]*\)\s*(?:noexcept\s*)?\{")
B1_SENTINEL = re.compile(r"\bggml_sycl_fattn_onednn_dispatch_routed\s*\([^;{]*\)\s*(?:const\s*)?\{")

FORMAT_STRINGS = [
    "[SYCL-COUNTER] dev=%d name=%s value=%llu",
    "[SYCL-COUNTER] dev=%d name=%s value=not_captured",
    "[SYCL-COUNTER] end devices=%d",
]

# The zone allocator's free-space figures need the zone's group mutex, which a refusal does not hold.
ZONE_ALLOCATOR_FIGURES = ["zone_available", "zone_largest_free"]
CHOKEPOINT_CACHE_READS = ["zone_used", "zone_capacity"]

PRINTER = "unified_cache_test_counter_dump"
SRC_EXT = (".cpp", ".hpp", ".h", ".c", ".inc")


def strip_comments(text):
    """Blank out comments, keeping every newline (so line numbers survive) and string/char literals."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                if text[i] == "\\" and i + 1 < n and text[i + 1] == "\n":
                    out.append(" \n")
                    i += 2
                    continue
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        elif c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i])
                    i += 1
                if text[i] == "\n":
                    break
                out.append(text[i])
                i += 1
            if i < n and text[i] == quote:
                out.append(quote)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def macro_polarity(expr):
    """+1 if the macro appears positively in a conditional expression, -1 if only negated, 0 if absent."""
    if MACRO not in expr:
        return 0
    positive = False
    negative = False
    for m in re.finditer(MACRO, expr):
        before = expr[: m.start()].rstrip()
        # strip a `defined`, `defined(` or `defined (` prefix
        probe = before
        probe = re.sub(r"\(\s*$", "", probe).rstrip()
        probe = re.sub(r"defined\s*$", "", probe).rstrip()
        if probe.endswith("!"):
            negative = True
        else:
            positive = True
    if positive:
        return 1
    return -1 if negative else 0


def compiled_lines(text):
    """For each line, (compiled, cause) with the macro undefined. cause names why a line is gated."""
    lines = text.split("\n")
    flags = [(True, "")] * len(lines)
    # a frame: [branch_gated, branch_cause, first_branch_negated, any_branch_positive_seen]
    stack = []
    result = []
    continued = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        is_directive = stripped.startswith("#") and not continued
        if is_directive:
            body = stripped[1:].strip()
            m = re.match(r"(ifdef|ifndef|if|elif|else|endif)\b(.*)", body)
            if m:
                kind, rest = m.group(1), m.group(2).strip()
                if kind == "ifdef":
                    pol = 1 if rest.split()[:1] == [MACRO] else 0
                    stack.append([pol == 1, "if", False])
                elif kind == "ifndef":
                    pol = -1 if rest.split()[:1] == [MACRO] else 0
                    stack.append([False, "", pol == -1])
                elif kind == "if":
                    pol = macro_polarity(rest)
                    stack.append([pol == 1, "if", pol == -1])
                elif kind == "elif" and stack:
                    pol = macro_polarity(rest)
                    stack[-1][0] = pol == 1
                    stack[-1][1] = "if"
                elif kind == "else" and stack:
                    # the #else of a negated test is not compiled; the #else of a positive test is
                    negated_first = stack[-1][2]
                    stack[-1][0] = negated_first
                    stack[-1][1] = "else-of-negated" if negated_first else ""
                elif kind == "endif" and stack:
                    stack.pop()
        gated = [f for f in stack if f[0]]
        if gated:
            result.append((False, gated[-1][1] or "if"))
        else:
            result.append((True, ""))
        continued = line.rstrip().endswith("\\")
    return result


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def read_tree(root):
    """The scanned sources (comment-stripped) and the CMakeLists text."""
    base = os.path.join(root, "ggml", "src", "ggml-sycl")
    files = {}
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = os.path.relpath(dirpath, base)
        if rel_dir == "tests" or rel_dir.startswith("tests" + os.sep):
            dirnames[:] = []
            continue
        for name in filenames:
            if name.endswith(SRC_EXT):
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, base)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    files[rel] = fh.read()
    cmake_path = os.path.join(base, "CMakeLists.txt")
    with open(cmake_path, encoding="utf-8", errors="replace") as fh:
        cmake = fh.read()
    return files, cmake


def macro_list(text, define):
    """(text, first line) of a continued `#define NAME(X)` list, or (None, None)."""
    lines = text.split("\n")
    pat = re.compile(r"^[ \t]*#[ \t]*define[ \t]+" + re.escape(define) + r"\(X\)")
    for idx, line in enumerate(lines):
        if pat.match(line):
            body = [line]
            while body[-1].rstrip().endswith("\\") and idx + len(body) < len(lines):
                body.append(lines[idx + len(body)])
            return "\n".join(body), idx + 1
    return None, None


def function_body_range(text, name):
    """(start_line, end_line) of the function definition `name(...) {...}` in text, or None."""
    m = re.search(r"\b" + re.escape(name) + r"\s*\(\s*\)\s*(?:noexcept\s*)?\{", text)
    if not m:
        return None
    depth = 0
    i = m.end() - 1
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return line_of(text, m.start()), line_of(text, i)


def function_text(text, qualified_name):
    """The text of the function definition `qualified_name(...) ... { ... }`, or None."""
    for m in re.finditer(r"\b" + re.escape(qualified_name) + r"\s*\(", text):
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        j = i
        while j < len(text) and text[j] not in "{;":
            j += 1
        if j < len(text) and text[j] == "{":
            depth = 0
            k = j
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[m.start():k + 1]
                k += 1
    return None


def cmake_targets_with_macro(cmake):
    """Targets that receive the macro through target_compile_definitions/options."""
    targets = set()
    other_forms = []
    for m in re.finditer(r"(target_compile_definitions|target_compile_options)\s*\(\s*(\S+)([^)]*)\)", cmake):
        if MACRO in m.group(3):
            targets.add(m.group(2))
    for m in re.finditer(r"(add_compile_definitions|add_definitions|set_source_files_properties|CMAKE_CXX_FLAGS"
                         r"|CMAKE_C_FLAGS|set_property)\b[^\n]*", cmake):
        if MACRO in m.group(0):
            other_forms.append(m.group(0).strip())
    return targets, other_forms


def enclosing_blocks(text, pos):
    """[(open_brace, close_brace)] of every `{...}` pair in `text` that contains `pos`, innermost first.
    String and char literals are skipped (comments are already stripped)."""
    stack = []
    pairs = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            stack.append(i)
        elif c == "}" and stack:
            start = stack.pop()
            if start < pos < i:
                pairs.append((start, i))
        i += 1
    pairs.sort(key=lambda se: se[1] - se[0])
    return pairs


def block_condition(text, brace):
    """The parenthesised condition of the `if (...)`/`while (...)` whose `{` is at `brace`, or ''."""
    j = brace - 1
    while j >= 0 and text[j] in " \t\n":
        j -= 1
    if j < 0 or text[j] != ")":
        return ""
    depth = 0
    k = j
    while k >= 0:
        if text[k] == ")":
            depth += 1
        elif text[k] == "(":
            depth -= 1
            if depth == 0:
                break
        k -= 1
    head = text[max(0, k - 12):k].rstrip()
    if not re.search(r"\b(?:if|while)$", head):
        return ""
    return text[k + 1:j]


def env_gated_before(body, pos):
    """The environment/build test that gates `body[pos]`, or None: an enclosing `if (...)` whose condition
    names one, or an earlier `if (<test>) return` in the same function (an early exit)."""
    for start, _end in enclosing_blocks(body, pos):
        cond = block_condition(body, start)
        if cond and ENV_TEST.search(cond):
            return cond.strip()
    for m in re.finditer(r"\bif\s*\(([^;{}]*)\)\s*\{?\s*return\b", body[:pos]):
        if ENV_TEST.search(m.group(1)):
            return m.group(1).strip()
    return None


def check(files, cmake):
    """Return the list of failures for one tree."""
    fails = []
    stripped = {rel: strip_comments(text) for rel, text in files.items()}
    comp = {rel: compiled_lines(text) for rel, text in stripped.items()}

    # Positive control: the scanner must see a macro region in unified-cache.cpp.
    cache_comp = comp.get("unified-cache.cpp")
    if cache_comp is None:
        return ["H13: unified-cache.cpp not found"]
    if not any(not c for c, _ in cache_comp):
        return ["H13 VOID: the region scanner found no GGML_SYCL_PRIVATE_TESTING region in unified-cache.cpp"]

    cache_text = stripped["unified-cache.cpp"]

    # The printer: defined once, registered with atexit once.
    defs = []
    for rel, text in stripped.items():
        for m in re.finditer(r"\bvoid\s+" + PRINTER + r"\s*\(\s*\)\s*\{", text):
            defs.append((rel, m))
    if len(defs) != 1:
        fails.append(f"H13 M74: {PRINTER} is defined {len(defs)} times, expected once")
    else:
        rel, m = defs[0]
        rng = function_body_range(stripped[rel], PRINTER)
        if rng:
            bad = [(i, comp[rel][i - 1]) for i in range(rng[0], rng[1] + 1) if not comp[rel][i - 1][0]]
            if bad:
                cause = bad[0][1][1]
                if cause == "else-of-negated":
                    fails.append("H13 M74: counter dump in the #else of an #ifndef")
                else:
                    fails.append("H13 M74: counter dump gated by PRIVATE_TESTING")
        body = stripped[rel]
        start = m.start()
        # the three format strings live in the printer's body
        body_text = body[start : body.find("\n}", start) + 2] if rng else ""
        for fmt in FORMAT_STRINGS:
            if fmt not in body_text:
                fails.append(f"H13 M74: format string missing from the printer: {fmt}")

    registrations = []
    for rel, text in stripped.items():
        for m in re.finditer(r"std::atexit\s*\(\s*" + PRINTER + r"\s*\)", text):
            registrations.append((rel, line_of(text, m.start())))
    if len(registrations) != 1:
        fails.append("H13 M74(g): dump not registered at exit" if not registrations else
                     "H13 M74(g): dump registered at exit more than once")
    else:
        rel, ln = registrations[0]
        ok, cause = comp[rel][ln - 1]
        if not ok:
            fails.append("H13 M74: dump registration in the #else of an #ifndef" if cause == "else-of-negated" else
                         "H13 M74: dump registration gated by PRIVATE_TESTING")

    # Which landing steps are on this tree.
    b2_landed = bool(B2_SENTINEL.search(cache_text))
    b1_landed = any(B1_SENTINEL.search(t) for t in stripped.values())

    # The table: the counter and snapshot macros.
    where = None
    counter_text = counter_line = None
    for rel, text in stripped.items():
        t, ln = macro_list(text, "GGML_SYCL_DUMP_COUNTERS")
        if t is not None:
            if where is not None:
                fails.append("H13 M74: GGML_SYCL_DUMP_COUNTERS is defined more than once")
            where, counter_text, counter_line = rel, t, ln
    if where is None:
        fails.append("H13 M74: the counter table (GGML_SYCL_DUMP_COUNTERS) is not defined")
        registered = []
    else:
        registered = re.findall(r"\bX\(\s*(\w+)\s*\)", counter_text)
        span = counter_text.count("\n") + 1
        for off in range(span):
            ok, cause = comp[where][counter_line - 1 + off]
            if not ok:
                fails.append("H13 M74: counter table gated by PRIVATE_TESTING")
                break

    expected = [name for name, lands, retired in FIELDS if not (retired == "b2" and b2_landed)]
    if registered != expected:
        missing = [n for n in expected if n not in registered]
        extra = [n for n in registered if n not in expected]
        for n in missing:
            lands = next(l for nm, l, _ in FIELDS if nm == n)
            if n == "onednn_graph_mask_declined":
                fails.append("H13 M74(i): mask_declined field missing")
            elif lands != "step0":
                fails.append(f"H13 M79(d): field absent before its lands step: {n} (lands {lands})")
            else:
                fails.append(f"H13 M74: field list differs from §5.1: {n} missing")
        for n in extra:
            fails.append(f"H13 M74: field list differs from §5.1: {n} is not in the list")
        if not missing and not extra:
            fails.append("H13 M74: field list differs from §5.1: order differs")

    # The printed names must come from the table, so a registered counter cannot go unprinted.
    if "GGML_SYCL_DUMP_COUNTERS(" not in cache_text.replace("#define", ""):
        fails.append("H13 M74: unified-cache.cpp builds the printed names without the table macro")

    # Snapshot entries.
    snap_text = snap_line = snap_where = None
    for rel, text in stripped.items():
        t, ln = macro_list(text, "GGML_SYCL_DUMP_SNAPSHOTS")
        if t is not None:
            snap_where, snap_text, snap_line = rel, t, ln
    snap_registered = []
    if snap_where is None:
        fails.append("H13 M79(e): snapshot entry missing or misspelled: the snapshot table is not defined")
    else:
        snap_registered = re.findall(r'\bX\(\s*(\w+)\s*,\s*"([^"]*)"\s*\)', snap_text)
        span = snap_text.count("\n") + 1
        for off in range(span):
            ok, cause = comp[snap_where][snap_line - 1 + off]
            if not ok:
                fails.append("H13 M74: snapshot table gated by PRIVATE_TESTING")
                break
    snap_expected = [(sid, printed) for sid, printed, lands, retired in SNAPSHOTS]
    if snap_registered != snap_expected:
        reg_set = set(snap_registered)
        missing = [e for e in snap_expected if e not in reg_set]
        extra = [e for e in snap_registered if e not in set(snap_expected)]
        for sid, printed in missing:
            lands = next(l for i, _, l, _ in SNAPSHOTS if i == sid)
            if extra:
                fails.append(f"H13 M79(e): snapshot entry missing or misspelled: {printed}")
            else:
                fails.append(f"H13 M79(e): snapshot entry absent before its lands step: {printed} (lands {lands})")
        for sid, printed in extra:
            fails.append(f"H13 M79(e): snapshot entry missing or misspelled: unlisted {printed}")
        if not missing and not extra:
            fails.append("H13 M79(e): snapshot entry missing or misspelled: order differs")
    if "value=not_captured" not in cache_text:
        fails.append("H13 M79(e): the printer carries no value=not_captured sentinel")

    # Every occurrence of a registered counter (an increment, a definition, a _for_testing accessor
    # that reads one) lies in a compiled region.
    for name, _, _ in FIELDS:
        tok = re.compile(r"\bdump_counter::" + re.escape(name) + r"\b")
        for rel, text in stripped.items():
            for m in tok.finditer(text):
                ln = line_of(text, m.start())
                ok, cause = comp[rel][ln - 1]
                if not ok:
                    fails.append(f"H13 M74: counter {name} gated ({rel}:{ln})")
    for name in REQUIRED_PRODUCERS:
        tok = re.compile(r"\bdump_counter::" + re.escape(name) + r"\b")
        live = 0
        for rel, text in stripped.items():
            for m in tok.finditer(text):
                if comp[rel][line_of(text, m.start()) - 1][0]:
                    live += 1
        if live == 0:
            fails.append(f"H13 M74: counter {name} gated: no compiled use on this tree")

    # Every step-0 snapshot entry has a compiled producer, in the function that owns its capture point.
    for sid, printed, lands, retired in SNAPSHOTS:
        if lands != "step0":
            continue
        rel, fn = SNAPSHOT_SITES[sid]
        body = function_text(stripped.get(rel, ""), fn)
        if body is None:
            fails.append(f"H13 G0: {fn} not found in {rel}, the capture site of {printed}")
            continue
        m = re.search(r"\bdump_snapshot::" + sid + r"\b", body)
        if not m:
            fails.append(f"H13 G0: snapshot {printed} has no producer in {fn} ({rel})")
        else:
            ln = line_of(stripped[rel], stripped[rel].find(body) + m.start())
            if not comp[rel][ln - 1][0]:
                fails.append(f"H13 G0: snapshot {printed} producer is gated by PRIVATE_TESTING ({rel}:{ln})")

    for name, rel, fn in PRODUCER_SITES:
        body = function_text(stripped.get(rel, ""), fn)
        if body is None:
            fails.append(f"H13 G0: {fn} not found in {rel}, the producer site of {name}")
        elif not re.search(r"dump_counter_add(?:_key)?\s*\(\s*(?:ggml_sycl::)?dump_counter::" + name + r"\b", body):
            fails.append(f"H13 G0: counter {name} has no increment in {fn} ({rel})")

    for kind, rel, fn, needle in REPORT_SITES:
        body = function_text(stripped.get(rel, ""), fn)
        if body is None:
            fails.append(f"H13 G0: {fn} not found in {rel}, the producer of the {kind} report")
        elif needle not in body:
            fails.append(f"H13 G0: {fn} ({rel}) lost the {kind} report (`{needle}`)")

    # A report is written with std::fprintf(stderr, ...) and never through the ggml log: the log drops INFO
    # at default verbosity. The emitters' whole bodies, and the block of each report in ggml-sycl.cpp.
    for rel, fn in REPORT_EMITTERS:
        body = function_text(stripped.get(rel, ""), fn)
        if body is None:
            fails.append(f"H13 G0: {fn} not found in {rel}, a report emitter")
            continue
        if "GGML_LOG_" in body:
            fails.append(f"H13 G0: report emitter {fn} ({rel}) calls GGML_LOG_*, which default verbosity drops")
        if not re.search(r"\bfprintf\s*\(\s*stderr\b", body):
            fails.append(f"H13 G0: report emitter {fn} ({rel}) does not write with fprintf(stderr, ...)")
    for kind, rel, fn, needle in REPORT_SITES:
        if rel != "ggml-sycl.cpp":
            continue
        body = function_text(stripped.get(rel, ""), fn)
        at = body.find(needle) if body else -1
        if at < 0:
            continue
        blocks = enclosing_blocks(body, at)
        scope = body[blocks[0][0]:blocks[0][1]] if blocks else body
        if "GGML_LOG_" in scope:
            fails.append(f"H13 G0: the {kind} report block in {fn} ({rel}) calls GGML_LOG_*, which default "
                         f"verbosity drops")
        if not re.search(r"unified_cache_dump_report(?:_once|_zone_figures)?\s*\(", scope):
            fails.append(f"H13 G0: the {kind} report block in {fn} ({rel}) does not emit through the dump report API")

    # Each row-134 reach site notes itself, in the function that holds the call.
    for fn, label in REACH_SITES:
        body = function_text(stripped.get("ggml-sycl.cpp", ""), fn)
        if body is None:
            fails.append(f"H13 G0: {fn} not found in ggml-sycl.cpp, the holder of reach site {label}")
        elif f'ggml_sycl_note_moe_table_reach("{label}"' not in body:
            fails.append(f"H13 G0: {fn} (ggml-sycl.cpp) lost its row-134 reach note `{label}`")

    # Counted unconditionally (§M179): no producer increment sits under an environment or build test,
    # inside an enclosing `if` or after an early exit on one. ARMED_ONLY_COUNTERS name the exception.
    for name, rel, fn in PRODUCER_SITES:
        if name in ARMED_ONLY_COUNTERS:
            continue
        body = function_text(stripped.get(rel, ""), fn)
        if body is None:
            continue
        for m in re.finditer(r"dump_counter_add(?:_key)?\s*\(\s*(?:ggml_sycl::)?dump_counter::" + name + r"\b", body):
            cond = env_gated_before(body, m.start())
            if cond is not None:
                fails.append(f"H13 M179: counter {name} is counted conditionally in {fn} ({rel}), under `{cond}`")

    for rel, fn, needle in CAPTURE_CALLERS:
        body = function_text(stripped.get(rel, ""), fn)
        if body is None:
            fails.append(f"H13 G0: {fn} not found in {rel}, the caller of a snapshot capture")
        elif needle not in body:
            fails.append(f"H13 G0: {fn} ({rel}) does not call the snapshot capture `{needle}`")

    # A not-captured field has no increment anywhere and the printer reads the one predicate.
    pred = function_text(cache_text, "dump_counter_captured")
    for name in NOT_CAPTURED_FIELDS:
        if pred is None or not re.search(r"\bdump_counter::" + name + r"\b", pred):
            fails.append(f"H13 G0: {name} is not named by dump_counter_captured, so the printer would print a zero")
        for rel, text in stripped.items():
            for m in re.finditer(r"dump_counter_add(?:_key)?\s*\(\s*(?:ggml_sycl::)?dump_counter::" + name + r"\b", text):
                fails.append(f"H13 G0: {name} has an increment ({rel}:{line_of(text, m.start())}) "
                             f"but is declared not captured")

    # An armed-only counter prints not_captured unless the report sites' armed flag read armed, and that flag
    # records its reading for the printer, so an unarmed or never-evaluated 0 cannot read as a measured 0.
    for name in ARMED_ONLY_COUNTERS:
        if pred is None or not (re.search(r"\bdump_counter::" + name + r"\b", pred) and "g_dump_armed_state" in pred):
            fails.append(f"H13 G0: armed-only counter {name} is not tied to the armed reading in dump_counter_captured")
    armed_fn = function_text(stripped.get("ggml-sycl.cpp", ""), "ggml_sycl_dump_report_armed")
    if armed_fn is None or "unified_cache_dump_note_armed" not in armed_fn:
        fails.append("H13 G0: ggml_sycl_dump_report_armed does not record its reading for the dump "
                     "(unified_cache_dump_note_armed)")

    # The report sites read the dump flag once per process (a getenv per op is a libc scan on a hot path):
    # the helper holds a function-local static, and nothing outside unified-cache.cpp, the helper aside,
    # calls the live accessor.
    if armed_fn is not None and not re.search(r"\bstatic\s+const\s+bool\b", armed_fn):
        fails.append("H13 I-2: ggml_sycl_dump_report_armed is not a function-local static const bool, so every "
                     "call reads the environment")
    for rel, text in stripped.items():
        if rel in ("unified-cache.cpp", "unified-cache.hpp"):
            continue
        scan = text.replace(armed_fn, "") if (rel == "ggml-sycl.cpp" and armed_fn) else text
        for m in re.finditer(r"\bunified_cache_dump_report_enabled\s*\(", scan):
            fails.append(f"H13 I-2: the live dump accessor is called outside ggml_sycl_dump_report_armed "
                         f"({rel}:{line_of(scan, m.start())})")

    # The stream_dma per-caller key names the file and the function, so two callers in one file stay apart.
    sd = function_text(cache_text, "unified_cache::stream_dma")
    if sd is None or "caller_func" not in sd or '"%s:%s"' not in sd:
        fails.append("H13 M-3: unified_cache::stream_dma's non-device-arrival key does not carry the caller's "
                     "function (`<file>:<function>`)")

    # The load-end report asks the plan's own predicate for "on device", as the runtime reach counter does,
    # and the planned device bytes are the one per-entry sum (never the single-device recorded figure).
    le = function_text(cache_text, "unified_cache_dump_capture_load_end")
    if le is not None:
        # D2 (source-pinned only: a behavioural case needs an arena cache; the behavioural check is the lead's
        # G0 rerun): weight_live_bytes reads cache->weight_bytes() only without an arena (an arena weight is not
        # in that figure), and prints not_captured, never a zero that means "not counted", with one.
        m = re.search(r"if\s*\(\s*cache->arena_active\(\)\s*\)\s*\{([^{}]*)\}\s*else\s*\{([^{}]*)\}", le)
        if m is None or "snapshot_clear(dump_snapshot::weight_live_bytes_last_load_end" not in m.group(1) or \
                "snapshot_set(dump_snapshot::weight_live_bytes_last_load_end" not in m.group(2) or \
                "weight_bytes()" not in m.group(2):
            fails.append("H13 D2: weight_live_bytes@last_load_end is not not_captured under an active arena "
                         "(cache->weight_bytes() does not count arena-placed weights)")
        if "expert_on_device(" not in le:
            fails.append("H13 M-5: unified_cache_dump_capture_load_end does not use expert_on_device, the "
                         "predicate the reach counter uses")
        if "weight_vram_bytes" in le:
            fails.append("H13 M-4: unified_cache_dump_capture_load_end reads weight_vram_bytes, a second "
                         "meaning for weight_planned_device_bytes")
    gc = function_text(stripped.get("ggml-sycl.cpp", ""), "ggml_sycl_moe_plan_gpu_expert_count")
    if gc is None or "expert_on_device(" not in gc:
        fails.append("H13 M-5: ggml_sycl_moe_plan_gpu_expert_count does not use expert_on_device")

    # The interim keys of onednn_graph_route_declined.
    key_re = re.compile(r"dump_counter_add_key\s*\(\s*dump_counter::" + ROUTE_COUNTER + r"\s*,[^;]*?\"(\w+)\"",
                        re.S)
    route_keys = set()
    for text in stripped.values():
        route_keys.update(key_re.findall(text))
    if b2_landed:
        for key in ROUTE_INTERIM_KEYS:
            if key in route_keys:
                fails.append(f"H13 M79(d): retired field still registered: {ROUTE_COUNTER}{{{key}}}")
    elif b1_landed:
        for key in ROUTE_INTERIM_KEYS:
            if key not in route_keys:
                fails.append(f"H13 M79(d): field dropped before its retiring step: {ROUTE_COUNTER}{{{key}}}")

    # The arena-active mirror: set where an arena is created, cleared where it is destroyed and
    # where it is abandoned. arena_base_ is a plain pointer the raw layer cannot read lock-free.
    note_re = re.compile(r"\bext_alloc_note_arena_state\s*\(([^;]*?),\s*(true|false)\s*\)\s*;")
    sets = len([m for m in note_re.finditer(cache_text) if m.group(2) == "true"])
    if sets < 2:
        fails.append(f"H13 M74(m): arena-active mirror set at {sets} site(s), expected the single-chunk and N-chunk reserve")
    for fn in ("unified_cache::arena_destroy", "unified_cache::arena_abandon"):
        body = function_text(cache_text, fn)
        if body is None:
            fails.append(f"H13 M74(m): {fn} not found")
        elif not any(m.group(2) == "false" for m in note_re.finditer(body)):
            fails.append(f"H13 M74(m): {fn} does not clear the arena-active mirror")

    # The raw exit counts at every raw exit: nothing returns before the first counter increment.
    note = function_text(cache_text, "unified_cache_note_raw_exit")
    if note is None:
        fails.append("H13 M74(r): unified_cache_note_raw_exit not found")
    else:
        count_at = note.find("dump_counter::ext_alloc_count")
        ret_at = note.find("return")
        if count_at < 0:
            fails.append("H13 M74(r): unified_cache_note_raw_exit does not count ext_alloc_count")
        elif 0 <= ret_at < count_at:
            fails.append("H13 M74(r): unified_cache_note_raw_exit returns before counting, so ext_alloc_count "
                         "reads 0 on a run that sets neither environment variable")

    # The chokepoint reads no allocator free-space figure, in its body or at any call.
    choke = function_text(cache_text, "unified_cache_zone_refusal")
    if choke is None:
        fails.append("H13 M74(c): unified_cache_zone_refusal not found")
    else:
        # An allow-list, so a figure nobody thought to name cannot slip in: the body may read the cache
        # only through zone_used and zone_capacity (an atomic and a field fixed at arena creation), and
        # only after the trace gate.
        gate = re.search(r"if\s*\(\s*!\s*ext_alloc_trace_enabled\s*\(\s*\)\s*\)", choke)
        for m in re.finditer(r"\bcache\s*->\s*(\w+)", choke):
            if m.group(1) not in CHOKEPOINT_CACHE_READS:
                fails.append(f"H13 M74(c): the chokepoint reads cache->{m.group(1)}, outside its allow-list "
                             f"{CHOKEPOINT_CACHE_READS}")
            elif gate is None or m.start() < gate.start():
                fails.append(f"H13 M74(c): the chokepoint reads cache->{m.group(1)} before the trace gate")
        if re.search(r"\ballocator\b|\blargest_free", choke):
            fails.append("H13 M74(c): the chokepoint reaches a zone allocator")
    for rel, text in stripped.items():
        for m in re.finditer(r"\bunified_cache_zone_refusal\s*\(([^;]*)\)\s*;", text):
            for fig in ZONE_ALLOCATOR_FIGURES:
                if re.search(r"\b" + fig + r"\b", m.group(1)):
                    fails.append(f"H13 M74(c): a call of the chokepoint passes {fig} ({rel}:{line_of(text, m.start())})")

    # The ggml-sycl target is given no macro definition.
    targets, other_forms = cmake_targets_with_macro(cmake)
    if "ggml-sycl" in targets:
        fails.append("H13 M74: the ggml-sycl target defines GGML_SYCL_PRIVATE_TESTING")
    for line in other_forms:
        fails.append(f"H13 M74: GGML_SYCL_PRIVATE_TESTING set outside a test target: {line}")
    for m in re.finditer(r"target_link_libraries\s*\(\s*ggml-sycl(?![\w-])([^)]*)\)", cmake):
        for t in targets:
            if re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", m.group(1)):
                fails.append(f"H13 M74: ggml-sycl links {t}, which defines GGML_SYCL_PRIVATE_TESTING")
    return fails


# ---------------------------------------------------------------------------
# Scanner fixture and the mutation matrix
# ---------------------------------------------------------------------------

def scanner_fixture():
    """The region scanner on the negated shape: an #ifndef with an #else."""
    fixture = "\n".join([
        "int a;",
        "#ifndef GGML_SYCL_PRIVATE_TESTING",
        "int compiled_branch;",
        "#else",
        "int gated_branch;",
        "#endif",
        "#if defined(GGML_SYCL_PRIVATE_TESTING)",
        "int gated_positive;",
        "#else",
        "int compiled_else;",
        "#endif",
        "#if !defined(GGML_SYCL_PRIVATE_TESTING)",
        "int compiled_negated;",
        "#else",
        "int gated_else_of_negated;",
        "#endif",
        "#if defined(GGML_SYCL_DNNL) && defined(GGML_SYCL_PRIVATE_TESTING)",
        "int gated_mixed;",
        "#endif",
        "#if defined(GGML_SYCL_DNNL)",
        "int compiled_other_macro;",
        "#endif",
    ])
    flags = compiled_lines(fixture)
    want = {
        "int compiled_branch;": True,
        "int gated_branch;": False,
        "int gated_positive;": False,
        "int compiled_else;": True,
        "int compiled_negated;": True,
        "int gated_else_of_negated;": False,
        "int gated_mixed;": False,
        "int compiled_other_macro;": True,
    }
    lines = fixture.split("\n")
    fails = []
    for text, expect in want.items():
        got = flags[lines.index(text)][0]
        if got != expect:
            fails.append(f"H13 scanner fixture: `{text}` classified compiled={got}, expected {expect}")
    return fails, len(want)


def wrap_in_macro(text, start_line, end_line, negate=False):
    """Wrap lines [start_line, end_line] (1-based) in an #if of the macro."""
    lines = text.split("\n")
    head = f"#ifndef {MACRO}" if negate else f"#if defined({MACRO})"
    lines.insert(end_line, "#endif")
    lines.insert(start_line - 1, head)
    return "\n".join(lines)


def mutation_matrix(files, cmake):
    """(label, expected message, mutated files, mutated cmake)."""
    stripped_cache = strip_comments(files["unified-cache.cpp"])
    muts = []

    def clone():
        return dict(files)

    # The printer's body wrapped in #if defined(M).
    rng = function_body_range(stripped_cache, PRINTER)
    f = clone()
    f["unified-cache.cpp"] = wrap_in_macro(files["unified-cache.cpp"], rng[0], rng[1])
    muts.append(("printer under #if defined(M)", "H13 M74: counter dump gated by PRIVATE_TESTING", f, cmake))

    # The printer in the #else of an #ifndef.
    f = clone()
    lines = files["unified-cache.cpp"].split("\n")
    lines.insert(rng[1], "#endif")
    lines.insert(rng[0] - 1, "#else")
    lines.insert(rng[0] - 1, "#ifndef " + MACRO)
    f["unified-cache.cpp"] = "\n".join(lines)
    muts.append(("printer in the #else of an #ifndef", "H13 M74: counter dump in the #else of an #ifndef", f, cmake))

    # ext_alloc_count's increment moved under the macro.
    f = clone()
    text = files["unified-cache.cpp"]
    m = re.search(r"^[^\n]*dump_counter::ext_alloc_count[^\n]*$", text, re.M)
    ln = line_of(text, m.start())
    f["unified-cache.cpp"] = wrap_in_macro(text, ln, ln)
    muts.append(("ext_alloc_count gated", "H13 M74: counter ext_alloc_count gated", f, cmake))

    # A field dropped, renamed, reordered.
    where = next(rel for rel, t in files.items() if "define GGML_SYCL_DUMP_COUNTERS" in t)
    ctext = files[where]
    f = clone()
    f[where] = re.sub(r"^[ \t]*X\(zone_plan_refusal\)[ \t]*\\\n", "", ctext, count=1, flags=re.M)
    muts.append(("field dropped", "H13 M74: field list differs from §5.1", f, cmake))
    f = clone()
    f[where] = ctext.replace("X(zone_plan_refusal)", "X(zone_plan_refusals)", 1)
    muts.append(("field renamed", "H13 M74: field list differs from §5.1", f, cmake))
    f = clone()
    f[where] = ctext.replace("X(zone_cascade_miss)", "X(@@A)").replace("X(zone_unconverted_miss)",
                                                                            "X(zone_cascade_miss)").replace(
        "X(@@A)", "X(zone_unconverted_miss)")
    muts.append(("fields reordered", "H13 M74: field list differs from §5.1", f, cmake))

    # A step-3 field dropped from a step-0 tree.
    f = clone()
    f[where] = re.sub(r"^[ \t]*X\(refusal_late\)[ \t]*\\\n", "", ctext, count=1, flags=re.M)
    muts.append(("refusal_late dropped", "H13 M79(d): field absent before its lands step", f, cmake))

    f = clone()
    f[where] = re.sub(r"^[ \t]*X\(onednn_graph_mask_declined\)[ \t]*\\\n", "", ctext, count=1, flags=re.M)
    muts.append(("mask_declined dropped", "H13 M74(i): mask_declined field missing", f, cmake))

    # A snapshot entry dropped, and one misspelled.
    swhere = next(rel for rel, t in files.items() if "define GGML_SYCL_DUMP_SNAPSHOTS" in t)
    stext = files[swhere]
    f = clone()
    f[swhere] = re.sub(r"^[ \t]*X\(weight_live_bytes_last_load_end,[^\n]*\n", "", stext, count=1, flags=re.M)
    f[swhere] = f[swhere].replace('"weight_planned_device_bytes{load_2}@last_load_end") \\', '"weight_planned_device_bytes{load_2}@last_load_end")')
    muts.append(("snapshot entry dropped", "H13 M79(e): snapshot entry absent before its lands step", f, cmake))
    f = clone()
    f[swhere] = stext.replace("@first_decode", "@first_decod", 1)
    muts.append(("snapshot point misspelled", "H13 M79(e): snapshot entry missing or misspelled", f, cmake))

    # The atexit registration removed.
    f = clone()
    f["unified-cache.cpp"] = re.sub(r"std::atexit\(\s*" + PRINTER + r"\s*\)\s*;", "", files["unified-cache.cpp"])
    muts.append(("atexit registration removed", "H13 M74(g): dump not registered at exit", f, cmake))

    # (b2)-shaped: interim_capped still registered.
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"] + (
        "\nvoid ggml_sycl_onednn_graph_scratch_range_miss(int dev) {\n}\n"
        "void route_probe() { unified_cache_dump_counter_add_key(dump_counter::onednn_graph_route_declined,"
        " 0, \"interim_capped\"); }\n")
    muts.append(("(b2) with interim_capped", "H13 M79(d): retired field still registered", f, cmake))

    # (b1)-shaped: interim_tp missing.
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"] + (
        "\nbool ggml_sycl_fattn_onednn_dispatch_routed(int x) {\n return true;\n}\n"
        "void route_probe() { unified_cache_dump_counter_add_key(dump_counter::onednn_graph_route_declined,"
        " 0, \"interim_capped\");\n unified_cache_dump_counter_add_key(dump_counter::onednn_graph_route_declined,"
        " 0, \"interim_capacity\"); }\n")
    muts.append(("(b1) with interim_tp missing", "H13 M79(d): field dropped before its retiring step", f, cmake))

    cache_raw = files["unified-cache.cpp"]

    def with_cache(text):
        g = clone()
        g["unified-cache.cpp"] = text
        return g

    # The registration under the macro.
    m = re.search(r"^[^\n]*std::atexit\(\s*" + PRINTER + r"\s*\)[^\n]*$", cache_raw, re.M)
    ln = line_of(cache_raw, m.start())
    muts.append(("atexit registration gated", "H13 M74: dump registration gated by PRIVATE_TESTING",
                 with_cache(wrap_in_macro(cache_raw, ln, ln)), cmake))

    # Each table's whole span under the macro.
    for rel, text in files.items():
        for define, label, message in (("GGML_SYCL_DUMP_COUNTERS", "counter table gated",
                                        "H13 M74: counter table gated by PRIVATE_TESTING"),
                                       ("GGML_SYCL_DUMP_SNAPSHOTS", "snapshot table gated",
                                        "H13 M74: snapshot table gated by PRIVATE_TESTING")):
            t, first = macro_list(strip_comments(text), define)
            if t is None:
                continue
            g = clone()
            g[rel] = wrap_in_macro(text, first, first + t.count("\n"))
            muts.append((label, message, g, cmake))

    # Every compiled use of a step-0 producer under the macro: no compiled use is left.
    lines = cache_raw.split("\n")
    hits = [i + 1 for i, l in enumerate(lines) if "dump_counter::zone_plan_refusal" in l]
    mutated = cache_raw
    for ln in reversed(hits):
        mutated = wrap_in_macro(mutated, ln, ln)
    g = with_cache(mutated)
    for rel, text in files.items():
        if rel != "unified-cache.cpp" and "dump_counter::zone_plan_refusal" in text:
            tl = text.split("\n")
            mm = text
            for ln in reversed([i + 1 for i, l in enumerate(tl) if "dump_counter::zone_plan_refusal" in l]):
                mm = wrap_in_macro(mm, ln, ln)
            g[rel] = mm
    muts.append(("zone_plan_refusal has no compiled use", "H13 M74: counter zone_plan_refusal gated: no compiled use",
                 g, cmake))

    # Each of the printer's three format strings, and the sentinel.
    muts.append(("value format changed", "format string missing from the printer: [SYCL-COUNTER] dev=%d name=%s value=%llu",
                 with_cache(cache_raw.replace("name=%s value=%llu", "name=%s val=%llu")), cmake))
    muts.append(("not_captured format changed",
                 "format string missing from the printer: [SYCL-COUNTER] dev=%d name=%s value=not_captured",
                 with_cache(cache_raw.replace("value=not_captured", "value=none")), cmake))
    muts.append(("end line format changed", "format string missing from the printer: [SYCL-COUNTER] end devices=%d",
                 with_cache(cache_raw.replace("[SYCL-COUNTER] end devices=%d", "[SYCL-COUNTER] done devices=%d")), cmake))
    muts.append(("not_captured sentinel gone", "the printer carries no value=not_captured sentinel",
                 with_cache(cache_raw.replace("value=not_captured", "value=none")), cmake))

    # The printed names built without the table macro.
    muts.append(("printed names without the table macro",
                 "unified-cache.cpp builds the printed names without the table macro",
                 with_cache(cache_raw.replace("GGML_SYCL_DUMP_COUNTERS(", "GGML_SYCL_DUMP_COUNTERS_BY_HAND(")), cmake))

    # The retired-at-step-3 field has no producer on this tree, so dropping it is caught by its lands step.
    f = clone()
    f[where] = re.sub(r"^[ \t]*X\(refusal_unattributed\)[ \t]*\\\n", "", ctext, count=1, flags=re.M)
    muts.append(("refusal_unattributed dropped", "H13 M79(d): field absent before its lands step: refusal_unattributed",
                 f, cmake))

    # The arena-active mirror: set sites and the two clears.
    note_call = "ext_alloc_note_arena_state(queue, true);"
    first = cache_raw.index(note_call)
    muts.append(("arena mirror set once", "H13 M74(m): arena-active mirror set at 1 site(s)",
                 with_cache(cache_raw[:first] + cache_raw[first + len(note_call):]), cmake))
    for fn, label in (("void unified_cache::arena_abandon()", "arena_abandon leaves the mirror set"),
                      ("bool unified_cache::arena_destroy()", "arena_destroy leaves the mirror set")):
        at = cache_raw.find(fn)
        if at < 0:
            at = cache_raw.find(fn.replace("bool ", "void ", 1))
        clear = "ext_alloc_note_arena_state(queue_, false);"
        k = cache_raw.index(clear, at)
        qualified = "unified_cache::" + fn.split("::")[1].split("(")[0]
        muts.append((label, f"H13 M74(m): {qualified} does not clear the arena-active mirror",
                     with_cache(cache_raw[:k] + cache_raw[k + len(clear):]), cmake))

    # The raw exit's accounting gated on the environment again.
    arena_read = "const bool arena = dump_dev_valid(dev) && dump_tbl().arena_active[dev]"
    muts.append(("raw exit gated on the trace", "H13 M74(r): unified_cache_note_raw_exit returns before counting",
                 with_cache(cache_raw.replace(arena_read, "if (!ext_alloc_trace_enabled()) {\n        return;\n    }\n    " + arena_read, 1)),
                 cmake))

    # The chokepoint reads an allocator figure, in its body and at a call.
    muts.append(("chokepoint reads zone_largest_free", "H13 M74(c): the chokepoint reads cache->zone_largest_free",
                 with_cache(cache_raw.replace("cache->zone_used(zone)", "cache->zone_largest_free(zone)", 1)), cmake))
    muts.append(("chokepoint reads an unnamed figure", "H13 M74(c): the chokepoint reads cache->zone_free_bytes",
                 with_cache(cache_raw.replace("cache->zone_used(zone)", "cache->zone_free_bytes(zone)", 1)), cmake))
    anchor = "    const alloc_constraints & constraints = req.intent.constraints;\n    if (constraints.cascade_step) {"
    muts.append(("chokepoint reads the cache before the trace gate",
                 "H13 M74(c): the chokepoint reads cache->zone_used before the trace gate",
                 with_cache(cache_raw.replace(anchor, "    (void) cache->zone_used(zone);\n" + anchor, 1)), cmake))
    muts.append(("call passes zone_available", "H13 M74(c): a call of the chokepoint passes zone_available",
                 with_cache(cache_raw.replace("unified_cache_zone_refusal(req, zid, alloc_size, cache);",
                                              "unified_cache_zone_refusal(req, zid, alloc_size, cache->zone_available(zid));",
                                              1)), cmake))

    # The G0 producers: each counter's increment dropped from its named function, one counter's
    # increment moved into another function of the file, a not-captured field given an increment, and
    # the printer's predicate no longer naming it.

    for name, rel, fn in PRODUCER_SITES:
        text = files[rel]
        stripped_text = strip_comments(text)
        body = function_text(stripped_text, fn)
        # Blank every occurrence of the counter's name inside the function body, in the raw text.
        raw_at = text.find(body.split("\n", 1)[0])
        raw_body_end = raw_at + len(body)
        mutated_body = re.sub(r"dump_counter::" + name + r"\b", "dump_counter::COUNT", text[raw_at:raw_body_end])
        f = clone()
        f[rel] = text[:raw_at] + mutated_body + text[raw_body_end:]
        muts.append((f"{name} increment dropped", f"H13 G0: counter {name} has no increment in {fn}", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace(
        "dump_counter::onednn_pp_record_mode_acquires", "dump_counter::COUNT", 1) + (
        "\nstatic void h13_moved_producer(int d) { ggml_sycl::unified_cache_dump_counter_add("
        "ggml_sycl::dump_counter::onednn_pp_record_mode_acquires, d); }\n")
    muts.append(("record-mode increment moved to another function",
                 "H13 G0: counter onednn_pp_record_mode_acquires has no increment in acquire_onednn_pp_scratch",
                 f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"] + (
        "\nstatic void h13_extra(int d) { ggml_sycl::unified_cache_dump_counter_add("
        "ggml_sycl::dump_counter::load_row_op_time_arrivals, d); }\n")
    muts.append(("not-captured field given an increment",
                 "H13 G0: load_row_op_time_arrivals has an increment", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "c != static_cast<size_t>(dump_counter::load_row_op_time_arrivals) &&", "true &&", 1)
    muts.append(("predicate no longer names the not-captured field",
                 "H13 G0: load_row_op_time_arrivals is not named by dump_counter_captured", f, cmake))

    # The snapshot producers and their callers, and the report lines.
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "dump_snapshot::weight_live_bytes_last_load_end, dev, cache->weight_bytes()",
        "dump_snapshot::COUNT, dev, cache->weight_bytes()", 1)
    # The clear call under an arena also names the snapshot, so the generic producer check cannot see the set
    # call go; the D2 pin does (its else arm must set it from cache->weight_bytes()).
    muts.append(("load-end live-bytes snapshot producer dropped",
                 "H13 D2: weight_live_bytes@last_load_end is not not_captured under an active arena", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace("if (cache->arena_active()) {\n        unified_cache_dump_snapshot_clear(dump_snapshot::weight_live_bytes_last_load_end",
                                                                "if (false) {\n        unified_cache_dump_snapshot_clear(dump_snapshot::weight_live_bytes_last_load_end", 1)
    muts.append(("live-bytes snapshot reads a zero-valued figure under an arena",
                 "H13 D2: weight_live_bytes@last_load_end is not not_captured under an active arena", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "dump_snapshot::zone_capacity_onednn_context_txn, dev,", "dump_snapshot::COUNT, dev,", 1)
    muts.append(("onednn capacity snapshot producer dropped",
                 "H13 G0: snapshot zone_capacity{ONEDNN}@context_txn has no producer", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace(
        "ctx->device, ggml_sycl::dump_point::CONTEXT_TXN", "ctx->device, ggml_sycl::dump_point::FIRST_DECODE", 1)
    muts.append(("context transaction no longer captures",
                 "H13 G0: ggml_sycl_run_runtime_context_transaction (ggml-sycl.cpp) does not call the snapshot capture",
                 f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace(
        "sycl_ctx->device, ggml_sycl::dump_point::FIRST_DECODE", "sycl_ctx->device, ggml_sycl::dump_point::CONTEXT_TXN", 1)
    muts.append(("first decode no longer captures",
                 "H13 G0: ggml_backend_sycl_graph_compute_unchecked (ggml-sycl.cpp) does not call the snapshot capture",
                 f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "unified_cache_dump_capture_load_end(device_id, cache.get(), slot, load_txn_id);", "", 1)
    muts.append(("load end no longer captures",
                 "H13 G0: unified_cache_note_model_load_end (unified-cache.cpp) does not call the snapshot capture",
                 f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace('"row73_own_alloc dev=', '"row73 dev=', 1)
    muts.append(("row 73 report dropped", "H13 G0: bool ggml_sycl_ensure_moe_ptr_table", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace('"arm_a_kernel dev=', '"arm_a dev=', 1)
    muts.append(("arm A kernel report dropped", "H13 G0: bool ggml_sycl_dispatch_mul_mat_kernel", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace(
        'ggml_sycl::unified_cache_dump_report_zone_figures(ctx->device, "after_backend_buffer",\n'
        '                                                          ggml_sycl::vram_zone_id::SCRATCH);', "", 1)
    muts.append(("landing no longer reports SCRATCH", "H13 G0: ggml_backend_sycl_buffer_publish", f, cmake))

    # The fold's rules: reports never go through the ggml log, producers count unconditionally, the
    # row-134 reach sites note themselves, and the new not-captured field stays not captured.
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        'std::fprintf(stderr, "[SYCL-REPORT] %s\\n", text);', 'GGML_LOG_INFO("[SYCL-REPORT] %s\\n", text);', 1)
    muts.append(("report emitter swapped to GGML_LOG_INFO",
                 "H13 G0: report emitter unified_cache_dump_report (unified-cache.cpp) calls GGML_LOG_*", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        '    std::fprintf(stderr,\n                 "[SYCL-REPORT] zone_figures dev=%d',
        '    GGML_LOG_INFO(\n                 "[SYCL-REPORT] zone_figures dev=%d', 1)
    muts.append(("zone_figures emitter swapped to GGML_LOG_INFO",
                 "H13 G0: report emitter dump_report_zone_figures_for (unified-cache.cpp) calls GGML_LOG_*", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace(
        "        ggml_sycl::unified_cache_dump_report(line);\n        ggml_sycl::unified_cache_dump_report_zone_figures(",
        "        GGML_LOG_INFO(\"%s\\n\", line);\n        ggml_sycl::unified_cache_dump_report_zone_figures(", 1)
    muts.append(("landing report swapped to GGML_LOG_INFO",
                 "H13 G0: the landing report block in ggml_backend_sycl_buffer_publish (ggml-sycl.cpp) calls GGML_LOG_*",
                 f, cmake))
    f = clone()
    f["set_rows.cpp"] = files["set_rows.cpp"].replace(
        '    {\n        char key[32];\n        std::snprintf(key, sizeof(key), "bytes=%zu", bytes);',
        '    if (ggml_sycl::unified_cache_dump_report_enabled()) {\n        char key[32];\n'
        '        std::snprintf(key, sizeof(key), "bytes=%zu", bytes);', 1)
    muts.append(("set_rows stage arrival counted only while armed",
                 "H13 M179: counter set_rows_stage_arrivals is counted conditionally in ggml_sycl_set_rows_stage_ptr", f,
                 cmake))
    f = clone()
    f["set_rows.cpp"] = files["set_rows.cpp"].replace(
        "    // The stage is predicted never reached on a single card",
        "    if (!ggml_sycl::unified_cache_dump_report_enabled()) {\n        return ptr;\n    }\n"
        "    // The stage is predicted never reached on a single card", 1)
    muts.append(("set_rows stage returns early unless armed",
                 "H13 M179: counter set_rows_stage_arrivals is counted conditionally in ggml_sycl_set_rows_stage_ptr", f,
                 cmake))
    f = clone()
    f["ggml-sycl.cpp"] = re.sub(r'ggml_sycl_note_moe_table_reach\("update:ggml_sycl_mul_mat_id/pair_up",[^;]*;', "",
                                files["ggml-sycl.cpp"], count=1)
    muts.append(("a row-134 reach note dropped",
                 "H13 G0: ggml_sycl_mul_mat_id (ggml-sycl.cpp) lost its row-134 reach note `update:ggml_sycl_mul_mat_id/pair_up`",
                 f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = re.sub(r'ggml_sycl_note_moe_table_reach\("ensure:graph_preload_moe_experts",[^;]*;', "",
                                files["ggml-sycl.cpp"], count=1)
    muts.append(("the preload's reach note dropped",
                 "H13 G0: graph_preload_moe_experts_impl (ggml-sycl.cpp) lost its row-134 reach note `ensure:graph_preload_moe_experts`",
                 f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"] + (
        "\nstatic void h13_extra2(int d) { ggml_sycl::unified_cache_dump_counter_add("
        "ggml_sycl::dump_counter::onednn_graph_callback_unmarked_mallocs, d); }\n")
    muts.append(("callback unmarked-malloc field given an increment",
                 "H13 G0: onednn_graph_callback_unmarked_mallocs has an increment", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "return g_dump_armed_state.load(std::memory_order_relaxed) == 1;", "return true;", 1)
    muts.append(("armed-only counter no longer tied to the armed reading",
                 "H13 G0: armed-only counter moe_table_reach_zero_gpu_expert is not tied to the armed reading", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace(
        "ggml_sycl::unified_cache_dump_note_armed(a);", "(void) a;", 1)
    muts.append(("armed flag no longer records its reading",
                 "H13 G0: ggml_sycl_dump_report_armed does not record its reading", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "           c != static_cast<size_t>(dump_counter::onednn_graph_callback_unmarked_mallocs);", "           true;", 1)
    muts.append(("predicate no longer names the callback field",
                 "H13 G0: onednn_graph_callback_unmarked_mallocs is not named by dump_counter_captured", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = files["ggml-sycl.cpp"].replace("layout_mismatch=%d ne11", "ne11", 1)
    muts.append(("arm A report drops the mismatch flag", "H13 G0: bool ggml_sycl_dispatch_mul_mat_kernel", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace('"planned_host load_txn=', '"planned load_txn=', 1)
    muts.append(("planned_host report dropped", "H13 G0: unified_cache_dump_capture_load_end", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace(
        "[SYCL-REPORT] moe_zero_gpu_expert_tensors dev=", "[SYCL-REPORT] moe_zero dev=", 1)
    muts.append(("zero-GPU-expert report dropped", "H13 G0: unified_cache_dump_capture_load_end", f, cmake))

    # §M205 r2: the load-end zero-tensor report is a report emitter; the I-2 split is pinned; the stream_dma
    # key carries the function; the load-end figures read the plan's predicate and one per-entry sum.
    f = clone()
    f["unified-cache.cpp"] = re.sub(r'std::fprintf\(stderr,\s*"\[SYCL-REPORT\] moe_zero_gpu_expert_tensors',
                                    'GGML_LOG_INFO("[SYCL-REPORT] moe_zero_gpu_expert_tensors',
                                    files["unified-cache.cpp"], count=1)
    muts.append(("zero-tensor report swapped to GGML_LOG_INFO",
                 "H13 G0: report emitter unified_cache_dump_capture_load_end (unified-cache.cpp) calls GGML_LOG_*", f,
                 cmake))
    live = "ggml_sycl::unified_cache_dump_report_enabled()"
    gtext = files["ggml-sycl.cpp"]
    f = clone()
    f["ggml-sycl.cpp"] = gtext.replace("if (is_lm_head && ggml_sycl_dump_report_armed()) {",
                                       "if (is_lm_head && " + live + ") {", 1)
    muts.append(("arm-A site reads the live accessor", "H13 I-2: the live dump accessor is called outside", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = gtext.replace("    if (ggml_sycl_dump_report_armed()) {\n        // Row 73's own-allocation fallback",
                                       "    if (" + live + ") {\n        // Row 73's own-allocation fallback", 1)
    muts.append(("row-73 site reads the live accessor", "H13 I-2: the live dump accessor is called outside", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = gtext.replace("    if (!ggml_sycl_dump_report_armed()) {\n        return;\n    }\n    int64_t gpu_experts",
                                       "    if (!" + live + ") {\n        return;\n    }\n    int64_t gpu_experts", 1)
    muts.append(("reach note reads the live accessor", "H13 I-2: the live dump accessor is called outside", f, cmake))
    f = clone()
    f["ggml-sycl.cpp"] = gtext.replace("    static const bool armed = [] {", "    const bool armed = [] {", 1)
    muts.append(("armed helper no longer a function-local static",
                 "H13 I-2: ggml_sycl_dump_report_armed is not a function-local static const bool", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = re.sub(r'std::snprintf\(caller_key, sizeof\(caller_key\), "%s:%s",\s*dump_site_basename\(caller_file\),\s*'
                                    r'caller_func != nullptr \? caller_func : "-"\);',
                                    'std::snprintf(caller_key, sizeof(caller_key), "%s", dump_site_basename(caller_file));',
                                    files["unified-cache.cpp"], count=1)
    muts.append(("stream_dma key reduced to the file basename",
                 "H13 M-3: unified_cache::stream_dma's non-device-arrival key does not carry the caller's function", f,
                 cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace("any = plan.expert_on_device(kv.first, e, dev);",
                                                                "any = e == 0;", 1)
    muts.append(("zero-tensor report re-derives on-device by hand",
                 "H13 M-5: unified_cache_dump_capture_load_end does not use expert_on_device", f, cmake))
    f = clone()
    f["unified-cache.cpp"] = files["unified-cache.cpp"].replace("host = plan.weight_host_bytes;",
                                                                "host = plan.weight_host_bytes;\n                planned = plan.weight_vram_bytes;", 1)
    muts.append(("planned device bytes read the recorded single-device figure",
                 "H13 M-4: unified_cache_dump_capture_load_end reads weight_vram_bytes", f, cmake))

    # The ggml-sycl target given the macro.
    muts.append(("ggml-sycl defines the macro", "H13 M74: the ggml-sycl target defines GGML_SYCL_PRIVATE_TESTING",
                 files, cmake + "\ntarget_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_PRIVATE_TESTING=1)\n"))
    return muts


def run_mutations(files, cmake):
    fails = []
    matrix = mutation_matrix(files, cmake)
    for label, message, mfiles, mcmake in matrix:
        got = check(mfiles, mcmake)
        if not any(message in g for g in got):
            fails.append(f"H13 mutation `{label}` was not caught: expected `{message}`, gate said {got or 'PASS'}")
    return fails, len(matrix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--no-mutations", action="store_true")
    ap.add_argument("--mutations-only", action="store_true")
    args = ap.parse_args()

    files, cmake = read_tree(os.path.abspath(args.root))
    failures, fixture_cases = scanner_fixture()
    if not args.mutations_only:
        failures += check(files, cmake)
    mutation_count = 0
    if not args.no_mutations and not failures:
        mutation_fails, mutation_count = run_mutations(files, cmake)
        failures += mutation_fails
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: the counter dump is compiled into the shipped library and its table matches the contract"
          f" ({mutation_count} mutations caught, {fixture_cases} scanner-fixture cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
