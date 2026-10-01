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
                      "zone_plan_refusal"]

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
