"""Source contract for the host-only plan dry run (th32 §4.2 "One source for the number").

The dry run, the stage's host_planned_bytes and the loader's expected_host are one computation, and "the dry-run and
the candidates do not store". The device test `test-sycl-plan-dry-run` scores the figures; this gate pins what a run
cannot see:

1. the stage's planner calls are DERIVED from the stage's own source: the calls in
   `compute_and_store_plan_for_inventory` to a function unified-cache.hpp declares returning `placement_plan`, and the
   calls in `ggml_sycl_configure_host_zones_for_plan` to one declared returning `host_zone_commit_terms`; every planner
   so found records its host terms through `populate_host_zone_sizing` and `record_host_ledger_plan`;
2. the dry run (the export and every `ggml_sycl_plan_dry_run*` helper) calls each of those functions, takes the WEIGHT
   terms from `placement_plan_host_weight_zone_terms`, reports expected_host as the plan's `host_planned_bytes`, sizes
   the commit terms from the plan's own `host_zone_terms` inputs, and shares the stage's inventory and budget helpers
   rather than restating them; it prints the plan's lazy tables, each one's residency, and the plan's refusal on
   [PLAN-DRY-RUN] LAZY lines from the plan's own fields, through the two formatters the planner's lines print with,
   and its host_planned_bytes check counts the lazy terms through the helper the plan's sum uses;
3. the dry run stores nothing: no `host_ledger_store_budget`, no `host_budget_pass_scope`, no zone configuration, no
   cache construction, no backend or queue initialisation, no write of the inventory globals; the shared helpers store
   nothing either;
4. the export refuses an open load or a live model before it samples, samples the host and computes the budget before
   it takes the inventory lock, and is declared, versioned and reachable through the proc table; the plans say, on
   one [PLAN-DRY-RUN] line gated on a device whose existing cache has no active arena, that the arena's zone
   reservations are not modelled;
5. libllama shares the early plan's inventory builder and the lazy-record pre-fill between the load and the header
   read, and the header read loads no tensor and opens no load; the tool resolves the export through the registry and
   includes no backend internals;
6. the extraction changes no behaviour: the load's early plan is the builder and the apply, the builder takes the
   steps the early plan took in the order it took them, and the load's pre-fill helper calls the record pre-fill;
7. the dry run leaves no trace: every per-device planner global the stage planners write (in their recording
   definitions and the static functions those reach within two calls) or the shared planner-input publish writes is
   written again by the destructor of the dry run's saved state, so a later load reads what it would have read
   without the dry run;
8. the residency decision's §10 plan lines, which the dry run's planner call prints, carry every field T3 can print,
   each from the plan's own terms: a STORAGE [PLACEMENT] line its cache's index bytes, a HOST one the host budget,
   the planned bytes and the leftover, and the [HOST-PLAN] lazy line the budget's source, the host weights, the WEIGHT
   zone's named terms, the host KV and the caches carving the zones. The budget's source is a planner input beside the
   budget: the load copies the ledger's flag stored with its budget, the dry run its own sample's, and the decision
   reads neither the ledger nor the environment. Every lazy line prints a table's fields through
   placement_lazy_table_fields and a refusal through placement_lazy_refusal_text, so no line builds the consumer's
   device name or the refusal text itself.

Plain file I/O, comment-stripped text.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

# One lexer for every pass: a raw string, a string, a char literal, a line comment, a block comment.
_LEXEME_RE = re.compile(
    r'\b(?:u8|u|U|L)?R"(?P<delim>[^ ()\\\t\n]{0,16})\((?P<raw>.*?)\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if m.group("delim") is not None or tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def balanced(text: str, open_idx: int) -> str:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise AssertionError("unbalanced braces")


def definitions(text: str, name: str) -> list:
    """The bodies of every definition of `name` (overloads included): a name, its parameter list, then a brace."""
    bodies = []
    for m in re.finditer(rf"(?<![\w.])(?<!->){re.escape(name)}\s*\(", text):
        depth, i = 0, m.end() - 1
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", text[i + 1 :])
        if tail:
            bodies.append(balanced(text, i + 1 + tail.end() - 1))
    return bodies


def body(text: str, name: str) -> str:
    found = definitions(text, name)
    assert found, f"no definition of {name}"
    return "\n".join(found)


def calls(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.])(?<!->){re.escape(name)}\s*\(", text) is not None


def member_calls(text: str, name: str) -> bool:
    """A call of `name` through an object or a pointer, which `calls` deliberately does not match."""
    return re.search(rf"(?:\.|->)\s*{re.escape(name)}\s*\(", text) is not None


def read(path: Path) -> str:
    return strip_comments(path.read_text())


def sycl_cpp() -> str:
    return read(SYCL / "ggml-sycl.cpp")


def declared_returning(hpp: str, rtype: str) -> set:
    return set(re.findall(rf"(?m)^\s*(?:(?:inline|static|constexpr)\s+)*{rtype}\s+(\w+)\s*\(", hpp))


def called_from(text: str, names: set) -> set:
    return {n for n in names if calls(text, n)}


EXPORT = "ggml_backend_sycl_plan_dry_run"

# The stage helpers the dry run shares instead of restating: the inventory's planner detail and KV shape, the
# per-device planner inputs it publishes, the single- versus multi-device choice, and the budgets each plan packs to.
SHARED = (
    "ggml_sycl_inventory_planner_detail",
    "ggml_sycl_inventory_fill_kv_info",
    "ggml_sycl_inventory_publish_planner_inputs",
    "ggml_sycl_plan_multi_device",
    "ggml_sycl_plan_single_device_budget",
    "ggml_sycl_plan_device_budgets",
)

# What a dry run must never reach: a store, a storing scope, zone configuration on a live cache, cache construction
# (the get-or-create accessors), backend or queue initialisation, the stage's own storing entries.
FORBIDDEN = (
    "host_ledger_store_budget",
    "host_budget_pass_scope",
    "ggml_sycl_configure_host_zones_for_plan",
    "configure_host_zones",
    "ensure_planned_arena_zones",
    "get_unified_cache_for_device",
    "ggml_sycl_device_budget_authority",
    "register_for_queue",
    "init_shared_context_queues",
    "ggml_backend_sycl_init",
    "populate_inventory_globals",
    "compute_and_store_plan_for_inventory",
    "ggml_sycl_set_tensor_inventory_impl",
    "unified_cache_set_plan",
)


def dry_run_closure(cpp: str) -> str:
    """The export and every `ggml_sycl_plan_dry_run*` helper it is built from."""
    parts = [body(cpp, EXPORT)]
    for name in sorted(set(re.findall(r"\b(ggml_sycl_plan_dry_run\w*)\s*\(", cpp))):
        parts.extend(definitions(cpp, name))
    return "\n".join(parts)


# ---- 1. the stage's calls, derived ----


def stage_planners() -> set:
    hpp = read(SYCL / "unified-cache.hpp")
    stage = body(sycl_cpp(), "compute_and_store_plan_for_inventory")
    found = called_from(stage, declared_returning(hpp, "placement_plan"))
    assert found, "compute_and_store_plan_for_inventory calls no planner unified-cache.hpp declares"
    return found


def stage_sizing() -> set:
    hpp = read(SYCL / "unified-cache.hpp")
    found = called_from(
        body(sycl_cpp(), "ggml_sycl_configure_host_zones_for_plan"), declared_returning(hpp, "host_zone_commit_terms")
    )
    assert found, "ggml_sycl_configure_host_zones_for_plan calls no sizing unified-cache.hpp declares"
    return found


def test_the_derivation_finds_the_stage_planners():
    # Engagement for the derivation: the two planners and the commit-terms sizing this tree is known to call.
    assert stage_planners() == {"compute_placement_plan", "compute_multi_device_plan"}, stage_planners()
    assert stage_sizing() == {"placement_plan_host_zone_commit_terms"}, stage_sizing()


def test_the_derivation_sees_a_planner_call_and_ignores_a_comment():
    hpp = "placement_plan compute_a(int x);\ninline placement_plan compute_b(int x);\nsize_t other(int);\n"
    src = strip_comments("void f() { auto p = compute_a(1); // compute_b(2)\n other(3); }")
    assert called_from(body(src, "f"), declared_returning(hpp, "placement_plan")) == {"compute_a"}


def test_every_stage_planner_records_its_host_terms():
    cpp = read(SYCL / "unified-cache.cpp")
    for planner in stage_planners():
        bodies = definitions(cpp, planner)
        assert bodies, f"no definition of {planner}"
        recording = [b for b in bodies if calls(b, "populate_host_zone_sizing") and calls(b, "record_host_ledger_plan")]
        assert recording, f"no definition of {planner} calls populate_host_zone_sizing and record_host_ledger_plan"


# ---- 2. the dry run calls the same functions ----


def test_the_dry_run_calls_every_stage_planner_and_the_sizing():
    closure = dry_run_closure(sycl_cpp())
    for name in sorted(stage_planners() | stage_sizing() | {"placement_plan_host_weight_zone_terms"}):
        assert calls(closure, name), f"the dry run does not call {name}"


def test_expected_host_is_the_plans_figure():
    closure = dry_run_closure(sycl_cpp())
    assert re.search(r"\bexpected_host\s*=\s*[\w.\->\[\]]*\bhost_planned_bytes\s*;", closure), (
        "the dry run's expected_host is not assigned the plan's host_planned_bytes"
    )


def test_the_commit_terms_take_the_plans_own_inputs():
    closure = dry_run_closure(sycl_cpp())
    hit = re.search(r"placement_plan_host_zone_commit_terms\s*\(([^;]*)\)\s*;", closure)
    assert hit, "the dry run does not size the commit terms"
    args = hit.group(1)
    own = re.search(r"host_zone_terms\s*\.\s*onednn_live", args) and re.search(r"host_zone_terms\s*\.\s*chunk", args)
    assert own, (
        f"the commit terms are sized from ({args}), not the plan's own host_zone_terms.onednn_live and .chunk"
    )
    assert "CHUNK_SIZE" not in closure, "the dry run restates the chunk"


def test_the_dry_run_shares_the_stage_helpers():
    cpp = sycl_cpp()
    stage = body(cpp, "populate_inventory_globals") + body(cpp, "compute_and_store_plan_for_inventory")
    closure = dry_run_closure(cpp)
    for name in SHARED:
        assert definitions(cpp, name), f"{name} is not defined"
        assert calls(stage, name), f"the stage does not call {name}"
        assert calls(closure, name), f"the dry run does not call {name}"


def test_the_residency_seam_is_named():
    # The weight entries' residency is reported per host backing from one named function; the lazy tables' residency
    # is the plan's per-table decision (th32 §4.3 step 4), which the LAZY lines below print.
    cpp = sycl_cpp()
    seam = body(cpp, "ggml_sycl_plan_dry_run_residency")
    assert calls(dry_run_closure(cpp).replace(seam, ""), "ggml_sycl_plan_dry_run_residency"), (
        "nothing in the dry run calls the residency seam"
    )


# The lazy tables (th32 §4.3 step 4, §10): the plan decides each table's residency and records a refusal, and the dry
# run prints what the plan holds, never a second computation: one LAZY line per table with its residency and resident
# bytes, and one for the plan carrying its refusal and the figures behind it. Both print through the formatters the
# planner's own lines print with, so the dry run and the planner name every field alike.
LAZY_FIELDS = ("lazy_tables", "lazy_refusal", "lazy_refusal_detail", "residency", "host_resident_bytes")
LAZY_FORMATTERS = ("placement_lazy_table_fields", "placement_lazy_refusal_text")
# What only the formatters build: the consumer's device name and the refusal with its optional detail.
LAZY_RESTATED = (
    (r'"CPU"', "builds the consumer's device name itself"),
    (r"refusal=%s%s%s", "builds the refusal text itself"),
)
# The fields the table formatter prints, as host-ledger-test-harness.hpp's printed() reads them, found in its string
# literals joined, so a formatter built by concatenation and one built from one format string both show them.
# locked= is the resident records' tier property (§3.1, §10): yes when the table holds resident bytes, none at 0.
# yieldable= needs the table's row geometry, so the formatter prints it when it is handed the table's info (§6.4).
LAZY_TABLE_FIELDS = (
    "table=",
    " resident=",
    " locked=",
    " cache_slots=",
    " floor_slots=",
    " consumer_device=",
    " yieldable=",
)
# A table formatter call with no table info: its line drops yieldable= and the consumer fields.
LAZY_FIELDS_WITHOUT_INFO = re.compile(r"\bplacement_lazy_table_fields\s*\([^;]*?,\s*nullptr\s*\)")


def log_formats(text: str) -> list:
    """The format string of every GGML_LOG_INFO and GGML_LOG_WARN call, adjacent literals joined."""
    return [
        "".join(re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1)))
        for m in re.finditer(r"GGML_LOG_(?:INFO|WARN)\s*\(\s*((?:\"(?:\\.|[^\"\\])*\"\s*)+)", text)
    ]


def formatter_bodies(uc: str) -> str:
    return "\n".join(body(uc, name) for name in LAZY_FORMATTERS)


def formatter_gaps(formatters: str) -> list:
    """What the two formatters lack; empty when they print every field from the plan's own record."""
    literals = "".join(re.findall(r'"((?:\\.|[^"\\])*)"', formatters))
    gaps = [
        f"the table formatter does not print {field.strip()}" for field in LAZY_TABLE_FIELDS if field not in literals
    ]
    if not re.search(r"\.\s*host_resident_bytes\b", formatters):
        gaps.append("the table formatter does not read the table's host_resident_bytes")
    if not re.search(r"\.\s*lazy_refusal_detail\b", formatters):
        gaps.append("the refusal formatter drops the refusal's detail")
    if '"none"' not in formatters:
        gaps.append("the refusal formatter prints no 'none' for a plan that did not refuse")
    if '"yes"' not in formatters:
        gaps.append("the table formatter prints no 'yes' for a table holding locked resident bytes")
    return gaps


def restated_gaps(text: str, where: str) -> list:
    return [f"{where} {gap}" for pattern, gap in LAZY_RESTATED if re.search(pattern, text)]


def lazy_report_gaps(closure: str, formatters: str) -> list:
    """What the dry run's lazy report lacks; empty when it holds. `formatters` is the bodies of LAZY_FORMATTERS."""
    gaps = []
    lazy = [f for f in log_formats(closure) if f.startswith("[PLAN-DRY-RUN]") and " LAZY " in f]
    if not any(" residency=" in f for f in lazy) or not calls(closure, "placement_lazy_table_fields"):
        gaps.append("no [PLAN-DRY-RUN] LAZY line prints a table's residency and its fields through the table formatter")
    if not any(" refusal=" in f for f in lazy) or not calls(closure, "placement_lazy_refusal_text"):
        gaps.append("no [PLAN-DRY-RUN] LAZY line prints the plan's refusal through the refusal formatter")
    if LAZY_FIELDS_WITHOUT_INFO.search(closure):
        gaps.append("the dry run hands the table formatter no table info, so its LAZY lines drop yieldable=")
    for field in LAZY_FIELDS:
        if not re.search(rf"(?:\.|->)\s*{field}\b", closure + formatters):
            gaps.append(f"the dry run never reads the plan's {field}")
    gaps += restated_gaps(closure, "the dry run")
    for decider in ("placement_plan_decide_lazy_residency", "record_host_ledger_plan"):
        if calls(closure, decider):
            gaps.append(f"the dry run calls {decider}: a second computation of the decision")
    return gaps


def lazy_sum_gaps(closure: str, hpp: str) -> list:
    """host_planned_bytes is the zone terms plus the lazy terms (th32 §4.2): the dry run's [PLAN-DRY-RUN-BUG] check
    must count the lazy terms through the plan's own helper, or every plan holding a table logs a false bug."""
    gaps = []
    if re.search(r"\.\s*total\s*\(\s*\)\s*!=\s*[\w.\->]*\bhost_planned_bytes\b", closure):
        gaps.append("the dry run compares the zone terms alone with host_planned_bytes")
    if not calls(closure, "placement_plan_lazy_host_bytes"):
        gaps.append("the dry run does not count the lazy terms through placement_plan_lazy_host_bytes")
    if not calls(body(hpp, "placement_plan_host_planned_bytes"), "placement_plan_lazy_host_bytes"):
        gaps.append("placement_plan_host_planned_bytes does not sum the lazy terms through their helper")
    return gaps


def test_the_dry_run_prints_the_lazy_decision():
    gaps = lazy_report_gaps(dry_run_closure(sycl_cpp()), formatter_bodies(read(SYCL / "unified-cache.cpp")))
    assert not gaps, "; ".join(gaps)


def test_the_dry_run_counts_the_lazy_terms():
    gaps = lazy_sum_gaps(dry_run_closure(sycl_cpp()), read(SYCL / "unified-cache.hpp"))
    assert not gaps, "; ".join(gaps)


def test_the_lazy_checks_see_each_dropped_piece():
    # Positive controls on a synthetic report shaped like the real one: each check passes it and fails each mutant.
    report = strip_comments(
        "void r(const placement_plan & plan) {\n"
        "  for (const placement_lazy_table & t : plan.lazy_tables) {\n"
        '    GGML_LOG_INFO("[PLAN-DRY-RUN] plan=%u LAZY %s residency=%s %s\\n", 0, t.name.c_str(),\n'
        '                  t.residency == S ? "a" : "b", placement_lazy_table_fields(t, info).c_str());\n'
        "  }\n"
        '  GGML_LOG_INFO("[PLAN-DRY-RUN] plan=%u LAZY tables=%zu refusal=%s\\n", 0, plan.lazy_tables.size(),\n'
        "                placement_lazy_refusal_text(plan).c_str());\n"
        "  const size_t lazy = placement_plan_lazy_host_bytes(plan);\n"
        "  if (terms.total() + lazy != plan.host_planned_bytes) {}\n"
        "}\n"
    )
    formatters = strip_comments(
        "std::string placement_lazy_table_fields(const placement_lazy_table & t,\n"
        "                                        const placement_lazy_table_info * i) {\n"
        '  return "table=" + std::to_string(t.table_bytes) + " resident=" + std::to_string(t.host_resident_bytes) +\n'
        '         " locked=" + (t.host_resident_bytes != 0 ? "yes" : "none") +\n'
        '         " cache_slots=" + std::to_string(t.cache_slots) +\n'
        '         " floor_slots=" + std::to_string(t.cache_floor_slots) +\n'
        '         " consumer_device=" + (t.consumer_device >= 0 ? std::to_string(t.consumer_device) : "CPU") +\n'
        '         (i != nullptr ? " yieldable=" + std::to_string(yieldable(t, i->row_size)) : "");\n'
        "}\n"
        "std::string placement_lazy_refusal_text(const placement_plan & plan) {\n"
        '  return plan.lazy_refusal.empty() ? "none" : plan.lazy_refusal + " " + plan.lazy_refusal_detail;\n'
        "}\n"
    )
    hpp = (
        "inline size_t placement_plan_host_planned_bytes(const placement_plan & p) {\n"
        "  return placement_plan_lazy_host_bytes(p);\n"
        "}\n"
    )
    assert lazy_report_gaps(report, formatters) == [], lazy_report_gaps(report, formatters)
    assert formatter_gaps(formatters) == [], formatter_gaps(formatters)
    assert lazy_sum_gaps(report, hpp) == [], lazy_sum_gaps(report, hpp)
    mutants = {
        "the refusal dropped from its line": (report.replace(" refusal=%s", ""), formatters),
        "the per-table line dropped": (report.replace("residency=%s ", ""), formatters),
        "the fields not through the formatter": (
            report.replace("placement_lazy_table_fields(t, info).c_str()", "t.name.c_str()"), formatters
        ),
        "no table info handed to the formatter": (
            report.replace("placement_lazy_table_fields(t, info)", "placement_lazy_table_fields(t, nullptr)"),
            formatters,
        ),
        "the refusal not through its formatter": (
            report.replace("placement_lazy_refusal_text(plan).c_str()", "plan.lazy_refusal.c_str()"), formatters
        ),
        "the detail not read": (report, formatters.replace(" + plan.lazy_refusal_detail", "")),
        "the resident bytes not read": (report, formatters.replace("t.host_resident_bytes", "0")),
        "the device name restated": (report.replace("t.name.c_str(),", 't.name.c_str(), "CPU",'), formatters),
        "the refusal text restated": (report.replace(" refusal=%s", " refusal=%s%s%s"), formatters),
        "a second decision": (
            report.replace("const size_t lazy", "record_host_ledger_plan(plan); const size_t lazy"), formatters
        ),
    }
    for name, (mutant, mutant_formatters) in mutants.items():
        assert (mutant, mutant_formatters) != (report, formatters), f"mutant '{name}' changed nothing"
        assert lazy_report_gaps(mutant, mutant_formatters), f"the lazy report check passes with {name}"
    formatter_mutants = {
        "the resident bytes not printed": formatters.replace('" resident="', '" r="'),
        "the device not printed": formatters.replace('" consumer_device="', '" d="'),
        "the locked state not printed": formatters.replace('" locked="', '" l="'),
        "the yieldable bytes not printed": formatters.replace('" yieldable="', '" y="'),
        "no yes": formatters.replace('"yes"', '"y"'),
        "no none": formatters.replace('"none"', '""'),
    }
    for name, mutant in formatter_mutants.items():
        assert mutant != formatters, f"mutant '{name}' changed nothing"
        assert formatter_gaps(mutant), f"the formatter check passes with {name}"
    bare = report.replace("terms.total() + lazy != plan.host_planned_bytes", "terms.total() != plan.host_planned_bytes")
    assert bare != report and lazy_sum_gaps(bare, hpp), "the sum check passes a bare zone-terms comparison"
    assert lazy_sum_gaps(report, hpp.replace("placement_plan_lazy_host_bytes(p)", "0")), (
        "the sum check passes a host_planned_bytes that does not count the lazy terms"
    )


# ---- 8. the residency decision's plan lines ----

# The fields each line must carry, keyed as host-ledger-test-harness.hpp's printed() reads them (" name="). locked=
# comes through the table formatter (LAZY_TABLE_FIELDS); output_slot, landing and cold_set wait for the rows that add
# them.
PLAN_LINE_FIELDS = {
    "STORAGE": (" index=",),
    "HOST": (" host_budget=", " host_planned=", " leftover="),
    "PLAN": (" source=", " weights=", " named_terms=", " kv=", " caches_carving="),
}

# Where each value comes from: the one helper or plan term that holds it.
PLAN_LINE_SOURCES = (
    (r"(?<![\w.])lazy_row_cache_index_bytes\s*\(", "the index bytes are not lazy_row_cache_index_bytes"),
    (r"(?<![\w.])placement_plan_host_weight_zone_terms\s*\(", "the weights are not placement_plan_host_weight_zone_terms"),
    (r"host_zone_terms\s*\.\s*kv\b", "the KV is not the plan's host_zone_terms.kv"),
    (r"kv_info\s*\.\s*host_budget_overridden\b", "the source is not the planner input kv_info.host_budget_overridden"),
    (r"\bdevices\s*\.\s*size\s*\(", "the caches carving is not counted from the plan's devices"),
)


def plan_line_gaps(decide: str) -> list:
    """What the residency decision's plan lines lack; empty when they hold."""
    gaps = []
    formats = log_formats(decide)
    groups = {
        "STORAGE": [f for f in formats if f.startswith("[PLACEMENT] lazy ") and " residency=STORAGE " in f],
        "HOST": [f for f in formats if f.startswith("[PLACEMENT] lazy ") and " residency=HOST " in f],
        "PLAN": [f for f in formats if f.startswith("[HOST-PLAN] lazy ")],
    }
    for group, lines in groups.items():
        if not lines:
            gaps.append(f"no {group} plan line with a literal residency or prefix")
        for line in lines:
            for field in PLAN_LINE_FIELDS[group]:
                if line.count(field) != 1:
                    gaps.append(f"the {group} line prints {field.strip()} {line.count(field)} times, want once")
            tail = line.find(" refusal=")
            for field in PLAN_LINE_FIELDS[group]:
                if tail >= 0 and line.find(field) > tail:
                    gaps.append(f"the {group} line prints {field.strip()} after its refusal")
    for pattern, gap in PLAN_LINE_SOURCES:
        if not re.search(pattern, decide):
            gaps.append(gap)
    for reader in ("host_ledger_budget_overridden", "host_ledger_budget", "getenv"):
        if calls(decide, reader):
            gaps.append(f"the decision calls {reader}: the budget's source is an input, not read here")
    for name in LAZY_FORMATTERS:
        if not calls(decide, name):
            gaps.append(f"the decision does not print through {name}")
    gaps += restated_gaps(decide, "the decision")
    return gaps


def budget_source_gaps(store: str, accessor: str, load: str, closure: str) -> list:
    """The budget's source travels with the budget: stored beside it from the same sample, copied by the load, and
    taken by the dry run from its own sample."""
    gaps = []
    if not re.search(r"g_host_ledger_budget_overridden\s*\.\s*store\s*\(", store) or "OVERRIDE" not in store:
        gaps.append("host_ledger_store_budget does not store the override flag beside the budget")
    if not re.search(r"g_host_ledger_budget_overridden\s*\.\s*load\s*\(", accessor):
        gaps.append("host_ledger_budget_overridden does not read the stored flag")
    if not re.search(
        r"g_placement_kv_info\s*\.\s*host_budget_overridden\s*=\s*(?:ggml_sycl::)?host_ledger_budget_overridden\s*\(",
        load,
    ):
        gaps.append("the load does not copy the stored flag into its planner input")
    if not re.search(r"\bkv\s*\.\s*host_budget_overridden\s*=", closure):
        gaps.append("the dry run does not set its planner input's budget source")
    if calls(closure, "host_ledger_budget_overridden"):
        gaps.append("the dry run reads the ledger's stored flag, not its own sample's")
    # An accepted override above the formula is not the budget (design 5efe §4.5: it composes), so the source is the
    # dry run's own budget's binding term, as the store's is.
    if not re.search(r"\.\s*override_binds\b", closure):
        gaps.append("the dry run's budget source is not its own budget's binding term (override_binds)")
    return gaps


def test_the_plan_lines_carry_every_printable_field():
    uc = read(SYCL / "unified-cache.cpp")
    gaps = plan_line_gaps(body(uc, "placement_plan_decide_lazy_residency")) + formatter_gaps(formatter_bodies(uc))
    assert not gaps, "; ".join(gaps)


def test_the_budget_source_is_a_planner_input():
    uc = read(SYCL / "unified-cache.cpp")
    cpp = sycl_cpp()
    gaps = budget_source_gaps(
        body(uc, "host_ledger_store_budget"),
        body(uc, "host_ledger_budget_overridden"),
        body(cpp, "compute_and_store_plan_for_inventory"),
        dry_run_closure(cpp),
    )
    assert not gaps, "; ".join(gaps)


def test_the_plan_line_checks_see_each_dropped_piece():
    # Positive controls on synthetic text shaped like the real one: each check passes it and fails each mutant.
    decide = strip_comments(
        "void d(placement_plan & plan, const placement_kv_info & kv_info) {\n"
        "  const host_weight_zone_terms wt = placement_plan_host_weight_zone_terms(plan);\n"
        "  const size_t carving = plan.multi_device ? plan.devices.size() : 1;\n"
        "  const std::string fields = placement_lazy_table_fields(t, &info);\n"
        "  if (storage) {\n"
        '    GGML_LOG_WARN("[PLACEMENT] lazy %s residency=STORAGE %s index=%zu\\n", n, fields.c_str(),\n'
        "                  lazy_row_cache_index_bytes(s));\n"
        "  } else {\n"
        '    GGML_LOG_WARN("[PLACEMENT] lazy %s residency=HOST %s host_budget=%zu "\n'
        '                  "host_planned=%zu leftover=%zu\\n", n, fields.c_str(), b, p, l);\n'
        "  }\n"
        "  const std::string refusal = placement_lazy_refusal_text(plan);\n"
        '  GGML_LOG_WARN("[HOST-PLAN] lazy device=%d budget=%zu source=%s planned=%zu weights=%zu named_terms=%zu "\n'
        '                "kv=%zu zones=%zu caches_carving=%zu refusal=%s\\n", d, b,\n'
        '                kv_info.host_budget_overridden ? "GGML_SYCL_HOST_RESERVE_MB" : "computed", p,\n'
        "                wt.host_weights, wt.tlsf_bound, plan.host_zone_terms.kv, z, carving, refusal.c_str());\n"
        "}\n"
    )
    assert plan_line_gaps(decide) == [], plan_line_gaps(decide)
    decide_mutants = {
        "the index dropped": decide.replace(" index=%zu", ""),
        "the host budget dropped": decide.replace(" host_budget=%zu", ""),
        "the leftover dropped": decide.replace(" leftover=%zu", ""),
        "the source dropped": decide.replace(" source=%s", ""),
        "the weights dropped": decide.replace(" weights=%zu", ""),
        "the named terms dropped": decide.replace(" named_terms=%zu", ""),
        "the KV dropped": decide.replace('"kv=%zu ', '"'),
        "the carving dropped": decide.replace(" caches_carving=%zu", ""),
        "a field after the refusal": decide.replace(" caches_carving=%zu refusal=%s", " refusal=%s caches_carving=%zu"),
        "a field printed twice": decide.replace(" zones=%zu", " zones=%zu kv=%zu"),
        "the residency a conversion": decide.replace("residency=STORAGE", "residency=%s"),
        "the index from elsewhere": decide.replace("lazy_row_cache_index_bytes(s)", "4 * s"),
        "the weights from elsewhere": decide.replace("placement_plan_host_weight_zone_terms(plan)", "{}"),
        "the KV from elsewhere": decide.replace("plan.host_zone_terms.kv", "plan.host_zone_kv_bytes"),
        "the source from the ledger": decide.replace(
            "kv_info.host_budget_overridden", "host_ledger_budget_overridden()"
        ),
        "the source from the environment": decide.replace(
            "kv_info.host_budget_overridden", 'getenv("GGML_SYCL_HOST_RESERVE_MB")'
        ),
        "the carving a constant": decide.replace("plan.multi_device ? plan.devices.size() : 1", "1"),
        "the table fields not through the formatter": decide.replace("placement_lazy_table_fields(t, &info)", '""'),
        "the refusal not through its formatter": decide.replace("placement_lazy_refusal_text(plan)", "r"),
        "the device name restated": decide.replace('"computed", p,', '"computed", "CPU", p,'),
    }
    for name, mutant in decide_mutants.items():
        assert mutant != decide, f"mutant '{name}' changed nothing"
        assert plan_line_gaps(mutant), f"the plan line check passes with {name}"

    store = (
        "g_host_ledger_mem_available.store(sample.available_bytes);\n"
        "g_host_ledger_budget_overridden.store(sample.reserve.status == detail::host_reserve_parse_status::OVERRIDE);\n"
    )
    accessor = "return g_host_ledger_budget_overridden.load(std::memory_order_acquire);\n"
    load = "g_placement_kv_info.host_budget_overridden = ggml_sycl::host_ledger_budget_overridden();\n"
    closure = (
        "const bool budget_overridden = budget.override_binds;\n"
        "kv.host_mem_available_pre = mem_available;\nkv.host_budget_overridden = budget_overridden;\n"
    )
    assert budget_source_gaps(store, accessor, load, closure) == [], budget_source_gaps(store, accessor, load, closure)
    source_mutants = {
        "the flag not stored": (store.replace("g_host_ledger_budget_overridden.store", "(void)"), accessor, load,
                                closure),
        "the flag stored from no override test": (store.replace("::OVERRIDE", "::AUTO").replace("OVERRIDE", "AUTO"),
                                                  accessor, load, closure),
        "the accessor reads nothing": (store, "return false;\n", load, closure),
        "the load copies nothing": (store, accessor, "", closure),
        "the dry run sets nothing": (store, accessor, load, "kv.host_mem_available_pre = mem_available;\n"),
        "the dry run's source is the accepted override": (
            store, accessor, load,
            closure.replace("budget.override_binds", "reserve.status == detail::host_reserve_parse_status::OVERRIDE"),
        ),
        "the dry run reads the ledger": (
            store, accessor, load, closure.replace("budget_overridden;", "ggml_sycl::host_ledger_budget_overridden();")
        ),
    }
    for name, args in source_mutants.items():
        assert args != (store, accessor, load, closure), f"mutant '{name}' changed nothing"
        assert budget_source_gaps(*args), f"the budget source check passes with {name}"


# The headroom refusal a plan's would-refuse line names (design 5efe §4.5) is its budget's own: the load copies the
# stored sample's figures into the planner input, the dry run its own sample's, and the line reads the planner input,
# never the stored values a dry run does not write.
REFUSAL_FIELDS = (
    ("host_headroom_exhausted", "host_ledger_headroom_exhausted"),
    ("host_mem_total", "host_ledger_mem_total"),
    ("host_headroom", "host_ledger_headroom"),
)


def refusal_source_gaps(record: str, load: str, closure: str) -> list:
    """Where the plan line's refusal reason comes from; empty when it is the budget's own sample."""
    gaps = []
    if not calls(record, "host_headroom_refusal_reason"):
        gaps.append("record_host_ledger_plan does not format the reason through host_headroom_refusal_reason")
    stored_readers = tuple(accessor for _, accessor in REFUSAL_FIELDS)
    stored_readers += ("host_ledger_mem_available", "host_ledger_budget")
    for reader in stored_readers:
        if calls(record, reader):
            gaps.append(f"record_host_ledger_plan calls {reader}: the reason is the stored sample's, not the plan's")
    if re.search(r"\bg_host_ledger_\w+", record):
        gaps.append("record_host_ledger_plan reads a stored ledger atomic")
    for field, accessor in REFUSAL_FIELDS + (("host_mem_available_pre", "host_ledger_mem_available"),):
        if not re.search(rf"kv_info\s*\.\s*{field}\b", record):
            gaps.append(f"the reason does not read kv_info.{field}")
    for field, accessor in REFUSAL_FIELDS:
        if not re.search(rf"g_placement_kv_info\s*\.\s*{field}\s*=\s*(?:ggml_sycl::)?{accessor}\s*\(", load):
            gaps.append(f"the load does not copy {accessor}() into its planner input's {field}")
        if not re.search(rf"\bkv\s*\.\s*{field}\s*=", closure):
            gaps.append(f"the dry run does not set its planner input's {field}")
        if calls(closure, accessor):
            gaps.append(f"the dry run calls {accessor}: the stored sample's, not its own")
    return gaps


def test_the_plan_refusal_reason_is_the_budgets_own_sample():
    uc = read(SYCL / "unified-cache.cpp")
    cpp = sycl_cpp()
    gaps = refusal_source_gaps(
        body(uc, "record_host_ledger_plan"), body(cpp, "compute_and_store_plan_for_inventory"), dry_run_closure(cpp)
    )
    assert not gaps, "; ".join(gaps)


def test_the_refusal_source_check_sees_each_dropped_piece():
    record = (
        "const std::string reason = host_headroom_refusal_reason(kv_info.host_headroom_exhausted,\n"
        "    kv_info.host_mem_available_pre, kv_info.host_mem_total, kv_info.host_headroom);\n"
    )
    load = (
        "g_placement_kv_info.host_headroom_exhausted = ggml_sycl::host_ledger_headroom_exhausted();\n"
        "g_placement_kv_info.host_mem_total = ggml_sycl::host_ledger_mem_total();\n"
        "g_placement_kv_info.host_headroom = ggml_sycl::host_ledger_headroom();\n"
    )
    closure = (
        "kv.host_headroom_exhausted = budget.status == ggml_sycl::host_budget_status::HEADROOM_EXHAUSTED;\n"
        "kv.host_mem_total = mem_total;\nkv.host_headroom = budget.headroom;\n"
    )
    assert refusal_source_gaps(record, load, closure) == [], refusal_source_gaps(record, load, closure)
    mutants = {
        "the line reads the stored flag": (
            record.replace("kv_info.host_headroom_exhausted", "host_ledger_headroom_exhausted()"), load, closure),
        "the line reads a stored atomic": (
            record.replace("kv_info.host_mem_total", "g_host_ledger_mem_total.load()"), load, closure),
        "the load copies no exhausted flag": (
            record, load.replace("host_ledger_headroom_exhausted()", "false"), closure),
        "the dry run sets no headroom": (record, load, closure.replace("kv.host_headroom = budget.headroom;\n", "")),
        "the dry run reads the stored MemTotal": (
            record, load, closure.replace("= mem_total;", "= ggml_sycl::host_ledger_mem_total();")),
    }
    for name, args in mutants.items():
        assert args != (record, load, closure), f"mutant '{name}' changed nothing"
        assert refusal_source_gaps(*args), f"the refusal source check passes with {name}"


# ---- 3. nothing stored ----


def test_the_dry_run_stores_nothing():
    closure = dry_run_closure(sycl_cpp())
    for name in FORBIDDEN:
        assert not calls(closure, name), f"the dry run calls {name}"
    assert not re.search(r"\bhost_budget_pass_scope\b", closure), "the dry run names a host_budget_pass_scope"


def test_the_shared_helpers_store_nothing():
    cpp = sycl_cpp()
    for name in SHARED:
        b = body(cpp, name)
        for forbidden in ("host_ledger_store_budget", "host_budget_pass_scope", "get_unified_cache_for_device",
                          "ggml_sycl_device_budget_authority"):
            assert not calls(b, forbidden) and not re.search(rf"\b{forbidden}\b", b), f"{name} reaches {forbidden}"


def test_member_calls_sees_a_member_call_only():
    assert member_calls("auto t = registry.current_active_token();", "current_active_token")
    assert member_calls("auto * w = ml->get_weight(n);", "get_weight")
    assert not member_calls("current_active_token();", "current_active_token")
    assert not calls("auto t = registry.current_active_token();", "current_active_token")


def test_the_forbidden_scan_sees_a_store():
    src = strip_comments("int ggml_backend_sycl_plan_dry_run(int a) { host_budget_pass_scope s(1); return a; }")
    assert re.search(r"\bhost_budget_pass_scope\b", body(src, EXPORT))
    assert not calls(body(src, EXPORT), "get_unified_cache_for_device")
    assert not calls("x = get_existing_unified_cache_for_device(d);", "get_unified_cache_for_device")


# ---- 4. the export ----


def test_the_export_refuses_then_samples_then_locks():
    b = body(sycl_cpp(), EXPORT)
    for name in ("current_active_token", "live_mask"):
        assert member_calls(b, name), f"the export does not consult {name}"
    busy = b.find("GGML_SYCL_LIFECYCLE_LOAD_BUSY")
    sample = b.find("host_ledger_sample(")
    budget = b.find("host_ledger_compute_budget(")
    lock = b.find("g_tensor_inventory_mutex")
    assert busy >= 0, "the export never refuses with LOAD_BUSY"
    assert 0 <= busy < sample, "the export samples before it refuses an open load"
    assert 0 <= sample < budget, "the export does not compute the budget from its sample"
    assert lock < 0 or budget < lock, "the export samples the host while it holds the inventory lock"


# The planners subtract the arena's zone reservations only on a device whose cache has an active arena, and the dry
# run creates no cache, so where none is active its device share is an upper bound and its host terms a lower bound.
ARENA_CAVEAT = "arena zone reservations not modelled: device share is an upper bound, host terms a lower bound"


def arena_caveat_printed(text: str) -> bool:
    """A log call whose format, its adjacent literals joined, starts with the caveat, after the two questions it is
    gated on, each asked of the planner's own rule: a single-device plan's, ggml_sycl_plan_single_device_arena_cache,
    with no inline restatement of that predicate (a member call of arena_active); and a multi-device plan's,
    multi_device_plan_subtracts_arena asked of the device_id of each budget the plan packs (every physical device), not
    of a scheduler-visible index."""
    joined = re.sub(r'"\s+"', "", text)
    m = re.search(r'GGML_LOG_(?:INFO|WARN)\s*\(\s*"\[PLAN-DRY-RUN\] ' + re.escape(ARENA_CAVEAT), joined)
    if m is None:
        return False
    head = joined[: m.start()]
    single = calls(head, "ggml_sycl_plan_single_device_arena_cache") and not member_calls(head, "arena_active")
    loop = re.search(r"for\s*\(\s*(?:const\s+)?(?:ggml_sycl::)?device_budget\s*&\s*(\w+)\s*:\s*budgets\s*\)", head)
    multi = loop is not None and re.search(
        rf"(?<![\w.])(?:ggml_sycl::)?multi_device_plan_subtracts_arena\s*\(\s*{loop.group(1)}\s*\.\s*device_id\s*\)",
        head[loop.end():]) is not None
    return single and multi


def test_the_single_device_budget_and_the_caveat_ask_one_rule():
    cpp = sycl_cpp()
    assert calls(body(cpp, "ggml_sycl_plan_single_device_budget"), "ggml_sycl_plan_single_device_arena_cache"), (
        "ggml_sycl_plan_single_device_budget decides its arena without ggml_sycl_plan_single_device_arena_cache, the "
        "rule the dry run's caveat asks"
    )
    rule = body(cpp, "ggml_sycl_plan_single_device_arena_cache")
    assert member_calls(rule, "arena_active") and calls(rule, "get_existing_unified_cache_for_device"), (
        "ggml_sycl_plan_single_device_arena_cache does not ask the existing cache whether its arena is active"
    )


def test_the_multi_device_planner_and_the_caveat_ask_one_rule():
    ucc = read(SYCL / "unified-cache.cpp")
    planner = body(ucc, "compute_multi_device_plan")
    assert calls(planner, "multi_device_plan_arena_cache"), (
        "compute_multi_device_plan decides its arena subtraction without multi_device_plan_arena_cache, the rule the "
        "dry run's caveat asks through multi_device_plan_subtracts_arena"
    )
    assert calls(body(ucc, "multi_device_plan_subtracts_arena"), "multi_device_plan_arena_cache"), (
        "multi_device_plan_subtracts_arena answers from another rule than the planner's"
    )


def test_the_dry_run_says_when_arena_reservations_are_not_modelled():
    plans = body(sycl_cpp(), "ggml_sycl_plan_dry_run_plans")
    assert arena_caveat_printed(plans), (
        f"ggml_sycl_plan_dry_run_plans does not print '[PLAN-DRY-RUN] {ARENA_CAVEAT}' for a device with no active arena"
    )


def test_the_arena_caveat_check_sees_a_dropped_reworded_or_ungated_line():
    printed = strip_comments(
        "static int ggml_sycl_plan_dry_run_plans(int n) {\n"
        "    std::string no_arena;\n"
        "    for (int i = 0; i < n; ++i) {\n"
        "        if (multi) {\n"
        "            for (ggml_sycl::device_budget & b : budgets) {\n"
        "                if (!ggml_sycl::multi_device_plan_subtracts_arena(b.device_id)) { no_arena += \" x\"; }\n"
        "            }\n"
        "        } else {\n"
        "            if (ggml_sycl_plan_single_device_arena_cache(i) == nullptr) { no_arena += \" x\"; }\n"
        "        }\n"
        "    }\n"
        "    if (!no_arena.empty()) {\n"
        '        GGML_LOG_INFO("[PLAN-DRY-RUN] arena zone reservations not modelled: "\n'
        '                      "device share is an upper bound, "\n'
        '                      "host terms a lower bound (no active arena on device%s)\\n", no_arena.c_str());\n'
        "    }\n"
        "    return n;\n"
        "}\n"
    )
    assert arena_caveat_printed(body(printed, "ggml_sycl_plan_dry_run_plans"))
    dropped = re.sub(r"GGML_LOG_INFO\(.*?\);", "", printed, flags=re.DOTALL)
    assert not arena_caveat_printed(body(dropped, "ggml_sycl_plan_dry_run_plans"))
    reworded = printed.replace("an upper bound", "a bound")
    assert not arena_caveat_printed(body(reworded, "ggml_sycl_plan_dry_run_plans"))
    ungated = printed.replace("ggml_sycl_plan_single_device_arena_cache(i) == nullptr", "false")
    assert ungated != printed
    assert not arena_caveat_printed(body(ungated, "ggml_sycl_plan_dry_run_plans"))
    # The single-device predicate restated inline: a second copy of the rule ggml_sycl_plan_single_device_budget asks.
    restated = printed.replace(
        "ggml_sycl_plan_single_device_arena_cache(i) == nullptr",
        "!ggml_sycl::get_existing_unified_cache_for_device(i) || "
        "!ggml_sycl::get_existing_unified_cache_for_device(i)->arena_active()",
    )
    assert restated != printed
    assert not arena_caveat_printed(body(restated, "ggml_sycl_plan_dry_run_plans"))
    # A multi-device plan asked about the scheduler-visible index: on a collapsed layout it names device 0 alone.
    exposed = printed.replace("multi_device_plan_subtracts_arena(b.device_id)", "multi_device_plan_subtracts_arena(i)")
    assert exposed != printed
    assert not arena_caveat_printed(body(exposed, "ggml_sycl_plan_dry_run_plans"))
    unasked = printed.replace("!ggml_sycl::multi_device_plan_subtracts_arena(b.device_id)", "false")
    assert unasked != printed
    assert not arena_caveat_printed(body(unasked, "ggml_sycl_plan_dry_run_plans"))


def test_the_export_is_declared_versioned_and_in_the_proc_table():
    hdr = (ROOT / "ggml/include/ggml-sycl.h").read_text()
    assert re.search(rf"//\s*Proc name:\s*\"{EXPORT}\"", hdr), "the header carries no proc name line"
    assert re.search(rf"GGML_BACKEND_API\s+enum\s+ggml_sycl_lifecycle_result\s+{EXPORT}\s*\(", hdr), (
        "the export is not a GGML_BACKEND_API returning ggml_sycl_lifecycle_result"
    )
    assert re.search(r"#define\s+GGML_SYCL_PLAN_DRY_RUN_VERSION\s+\d+", hdr), "no result version"
    for s in ("ggml_sycl_plan_dry_run_opts", "ggml_sycl_plan_dry_run_plan", "ggml_sycl_plan_dry_run_result"):
        assert re.search(rf"GGML_SYCL_ABI_ASSERT\s*\(\s*sizeof\s*\(\s*(struct\s+)?{s}\s*\)", hdr), (
            f"{s} has no size assert"
        )
    cpp = sycl_cpp()
    assert re.search(rf'strcmp\s*\(\s*name\s*,\s*"{EXPORT}"\s*\)\s*==\s*0', cpp), "the proc table has no arm"
    assert re.search(rf"return\s*\(\s*void\s*\*\s*\)\s*{EXPORT}\s*;", cpp), "the proc table arm returns another symbol"


# ---- 5. libllama and the tool ----


def test_libllama_shares_the_builder_and_the_prefill():
    src = read(ROOT / "src/llama-model.cpp")
    header = body(src, "llama_model_sycl_header_inventory")
    early = body(src, "llama_model_sycl_compute_early_plan")
    load = body(src, "load_tensors")
    assert calls(early, "llama_model_sycl_build_early_inventory"), "the early plan does not use the shared builder"
    assert calls(header, "llama_model_sycl_build_early_inventory"), "the header read does not use the shared builder"
    assert calls(load, "llama_model_prefill_lazy_record"), "the load does not use the shared pre-fill"
    assert calls(header, "llama_model_prefill_lazy_record"), "the header read does not use the shared pre-fill"
    assert not calls(load, "llama_lazy_range_record_prefill"), "the load restates the pre-fill"
    builder = body(src, "llama_model_sycl_build_early_inventory")
    assert calls(builder, "llama_model_sycl_populate_inventory"), "the shared builder does not populate the inventory"
    for name in ("load_all_data", "llama_model_sycl_apply_inventory", "ggml_backend_sycl_model_load_begin",
                 "create_tensor", "load_tensors"):
        assert not calls(header, name), f"the header read calls {name}"


def test_the_tool_resolves_the_export_through_the_registry():
    tool = read(ROOT / "tools/sycl-plan-dry-run/sycl-plan-dry-run.cpp")
    assert calls(tool, "ggml_backend_reg_get_proc_address"), "the tool does not resolve the export"
    assert f'"{EXPORT}"' in tool, "the tool resolves another name"
    assert not calls(tool, EXPORT), "the tool links the export directly"
    for inc in re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', tool):
        assert "ggml-sycl/" not in inc and not inc.endswith(".hpp"), f"the tool includes backend internals: {inc}"
    for name in ("llama_model_load_from_file", "llama_init_from_model", "ggml_backend_sycl_init"):
        assert not calls(tool, name), f"the tool calls {name}"


def prefill_log_marker() -> str:
    """The text every line the lazy-range record's pre-fill logs starts with, after its function-name prefix: the
    common prefix of the format strings in llama_lazy_range_record_prefill, so a rewording there moves it here."""
    b = body(read(ROOT / "src/llama-lazy-range-record.cpp"), "llama_lazy_range_record_prefill")
    fmts = re.findall(r'LLAMA_LOG_INFO\s*\(\s*"((?:\\.|[^"\\])*)"', b)
    assert len(fmts) >= 2, f"the pre-fill logs {len(fmts)} INFO lines, want the mode line and the row line"
    texts = [f[len("%s: "):] if f.startswith("%s: ") else f for f in fmts]
    return os.path.commonprefix(texts).rstrip()


def callback_routes_to_stdout(callback: str, marker: str) -> bool:
    """A branch of `callback` whose condition finds `marker` in the text with strstr and whose body writes the text,
    unchanged, to stdout."""
    for m in re.finditer(r"if\s*\((.*?)\)\s*\{", callback, flags=re.DOTALL):
        lits = re.findall(r'strstr\s*\(\s*\w+\s*,\s*"((?:\\.|[^"\\])*)"\s*\)', m.group(1))
        if marker in lits and re.search(r"fputs\s*\(\s*text\s*,\s*stdout\s*\)", balanced(callback, m.end() - 1)):
            return True
    return False


def test_the_tool_prints_the_lazy_range_record():
    # A table the record defers to the storage tier is left out of the plan; the tool shows which, in the pre-fill's
    # own words, beside the plan.
    marker = prefill_log_marker()
    assert marker == "lazy-range record:", f"the pre-fill's lines start {marker!r}"
    callback = body(read(ROOT / "tools/sycl-plan-dry-run/sycl-plan-dry-run.cpp"), "log_callback")
    assert callback_routes_to_stdout(callback, marker), (
        f"the tool's log_callback does not write the pre-fill's '{marker}' lines to stdout unchanged"
    )


def test_the_route_check_sees_a_dropped_route():
    routed = strip_comments(
        'void log_callback(int level, const char * text) {\n'
        '    if (std::strstr(text, "[PLAN-DRY-RUN") != nullptr'
        ' || std::strstr(text, "lazy-range record:") != nullptr) {\n'
        '        std::fputs(text, stdout);\n    } else if (level > 2) {\n        std::fputs(text, stderr);\n    }\n}\n'
    )
    assert callback_routes_to_stdout(body(routed, "log_callback"), "lazy-range record:")
    dropped = routed.replace(' || std::strstr(text, "lazy-range record:") != nullptr', "")
    assert not callback_routes_to_stdout(body(dropped, "log_callback"), "lazy-range record:")
    reworded = routed.replace("std::fputs(text, stdout);", 'std::printf("table deferred\\n");')
    assert not callback_routes_to_stdout(body(reworded, "log_callback"), "lazy-range record:")


# ---- 6. the extraction changes no behaviour ----

# The early plan's steps, in the order the load took them before the builder was extracted.
EARLY_STEPS = (
    "llama_lazy_early_plan_excludes",
    "llama_model_sycl_add_tensor_info",
    "llama_model_sycl_mark_get_rows_only",
    "llama_model_sycl_build_kv_layer_arrays",
    "llama_model_sycl_populate_inventory",
)


def first_call(text: str, name: str) -> int:
    m = re.search(rf"(?<![\w.])(?<!->){re.escape(name)}\s*\(", text)
    return m.start() if m else -1


def test_the_early_plan_is_the_builder_and_the_apply():
    src = read(ROOT / "src/llama-model.cpp")
    early = body(src, "llama_model_sycl_compute_early_plan")
    assert calls(early, "llama_model_sycl_build_early_inventory"), "the early plan does not call the builder"
    assert calls(early, "llama_model_sycl_apply_inventory"), "the early plan does not apply the inventory"
    for step in EARLY_STEPS:
        assert not calls(early, step), f"the early plan restates {step} beside the builder"


def test_the_builder_takes_the_early_plans_steps_in_order():
    builder = body(read(ROOT / "src/llama-model.cpp"), "llama_model_sycl_build_early_inventory")
    at = [first_call(builder, step) for step in EARLY_STEPS]
    missing = [step for step, pos in zip(EARLY_STEPS, at) if pos < 0]
    assert not missing, f"the builder does not call {missing}"
    assert at == sorted(at), f"the builder calls the steps out of order: {dict(zip(EARLY_STEPS, at))}"
    hit = re.search(r"llama_model_sycl_populate_inventory\s*\(([^;]*)\)\s*;", builder)
    assert hit and "get_n_ctx_hint()" in hit.group(1) and "get_n_ubatch_hint()" in hit.group(1), (
        "the builder does not plan at the model's context and micro-batch hints"
    )


def test_the_prefill_helper_is_the_record_prefill():
    helper = body(read(ROOT / "src/llama-model.cpp"), "llama_model_prefill_lazy_record")
    assert calls(helper, "llama_lazy_range_record_prefill"), "the pre-fill helper does not call the record pre-fill"
    assert member_calls(helper, "get_weight"), (
        "the pre-fill helper does not resolve the record's sources from the file"
    )


# ---- 7. no trace ----


def setters_in(text: str) -> set:
    return set(re.findall(r"\b(unified_cache_set_planned_\w+)\s*\(", text))


def static_callees(cpp: str, text: str) -> list:
    names = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", text))
    return [n for n in sorted(names) if re.search(rf"(?m)^static\s+[^;{{]*\b{n}\s*\(", cpp)]


def planner_written_setters() -> set:
    """The setters the stage planners call while they plan: in their recording definitions and in the file's static
    functions those reach within two calls (the host sizing, the host scatter term and what they call)."""
    cpp = read(SYCL / "unified-cache.cpp")
    texts, seen = [], set()
    frontier = []
    for planner in stage_planners():
        frontier.extend(b for b in definitions(cpp, planner) if calls(b, "populate_host_zone_sizing"))
    texts.extend(frontier)
    for _ in range(2):
        nxt = []
        for text in frontier:
            for name in static_callees(cpp, text):
                if name not in seen:
                    seen.add(name)
                    nxt.extend(definitions(cpp, name))
        texts.extend(nxt)
        frontier = nxt
    found = setters_in("\n".join(texts))
    assert found, "the stage planners write no per-device planner global"
    return found


def test_the_planner_writes_the_known_globals():
    # Engagement for the derivation: the shapes the sizing is known to publish.
    found = planner_written_setters()
    for name in ("unified_cache_set_planned_nonfa_attn_scratch_shape",
                 "unified_cache_set_planned_onednn_graph_scratch_shape",
                 "unified_cache_set_planned_moe_control_requirement",
                 "unified_cache_set_planned_moe_host_scatter_scratch_bytes"):
        assert name in found, f"the derivation misses {name}: {sorted(found)}"


def test_the_saved_state_restores_every_written_global():
    cpp = sycl_cpp()
    hit = re.search(r"struct\s+(ggml_sycl_plan_dry_run_\w*state\w*)\s*\{", cpp)
    assert hit, "the dry run has no saved-state struct"
    name = hit.group(1)
    struct_body = balanced(cpp, hit.end() - 1)
    dtor = re.search(rf"~\s*{name}\s*\(\s*\)\s*(?:noexcept\s*)?\{{", struct_body)
    assert dtor, f"{name} has no destructor"
    restored = setters_in(balanced(struct_body, dtor.end() - 1))
    written = planner_written_setters() | setters_in(body(cpp, "ggml_sycl_inventory_publish_planner_inputs"))
    missing = sorted(written - restored)
    assert not missing, f"the dry run's saved state does not restore {missing}"
    assert re.search(rf"\b{name}\s+\w+\s*[;({{]", body(cpp, EXPORT) + dry_run_closure(cpp)), (
        f"the dry run never holds a {name}"
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
