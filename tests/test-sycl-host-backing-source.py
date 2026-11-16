"""Source contract for llama.cpp-5efe C4 (part c): the planner's host-weight accounting is split by host backing.

The runtime half is the pure host test `test-sycl-host-backing-accounting` (each backing's term, placed against charged,
the per-entry group split, the refresh, the mapped-source recount, an expert moving between tiers). This gate pins what
that test cannot see: that every writer of the host terms goes through the one helper that tests the backing, and that
every reader reads the meaning design 5efe §4.4 assigns it.

1. `placement_entry::host_backing` is a `placement_host_backing` defaulting to PINNED, the enum has the four values, and
   `placement_plan` carries `host_cpu_owned_bytes`, `host_mapped_bytes` and `host_mapped_source_bytes`;
2. nothing writes `weight_host_bytes`, `host_cpu_owned_bytes` or `host_mapped_bytes` except the plan's own
   `charge_host_weight` / `uncharge_host_weight` (and the `= 0` initialisation of a new plan): every accumulation site in
   the planner (the single-device and multi-device builders, the layout-change delta, `update_expert_placement`,
   `placement_move_host_expert_backing`) calls one of the helpers, and the sites are the pinned list below; the move
   helper uncharges each expert from the backing it leaves and charges it to the one it takes, at its `src_size`,
   around the flip;
3. the readers: the plan snapshot's `planned_host_bytes` (staged candidate and published) is the PLACED figure; the
   lifecycle publication carries `charged_host_bytes` beside it into `ModelState`; the load summary prints placed and
   charged; both `Totals` lines print all four terms; `refresh_kv_byte_totals` counts the cpu-owned bytes and not the
   mapped ones.
4. the opt-out of the planner's pin of routed host experts has one reader, `placement_pin_host_experts_enabled()`:
   unified-cache.hpp names the variable once, the reader is the only place either source reads it from the
   environment, the planner's pin asks the reader, and `ggml_check_sycl`'s settings table carries a row for it whose
   value is overwritten from the reader after the parse, so the non-default settings line reports a `=0`.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

# One lexer for every pass: a raw string (whose body may hold quotes and newlines), a string, a char literal, a line
# comment, a block comment. Scanning left to right, a quote inside a char literal or a raw string opens nothing.
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


def function_body(text: str, signature_re: str) -> str:
    hits = list(re.finditer(signature_re, text))
    assert len(hits) == 1, f"{signature_re!r} matched {len(hits)} times"
    return balanced(text, text.index("{", hits[0].end()))


def read(rel: str) -> str:
    return strip_comments((SYCL / rel).read_text())


HPP = "unified-cache.hpp"
CPP = "unified-cache.cpp"
HOST_TERMS = ("weight_host_bytes", "host_cpu_owned_bytes", "host_mapped_bytes")


# ---- 1. the entry field, the enum and the plan terms ----


def test_the_enum_has_the_four_backings():
    hpp = read(HPP)
    hit = re.search(r"enum\s+class\s+placement_host_backing\b[^{]*\{", hpp)
    assert hit, "placement_host_backing is missing"
    body = balanced(hpp, hit.end() - 1)
    names = [n.strip() for n in body.split(",") if n.strip()]
    assert names == ["PINNED", "FILE_MAPPED", "HOST_CPU", "CPU_OWNED"], f"the backings are {names}"


def test_the_entry_carries_a_backing_defaulting_to_pinned():
    hpp = read(HPP)
    hit = re.search(r"struct\s+placement_entry\s*\{", hpp)
    body = balanced(hpp, hit.end() - 1)
    assert re.search(
        r"placement_host_backing\s+host_backing\s*=\s*placement_host_backing\s*::\s*PINNED\s*;", body
    ), "placement_entry has no `placement_host_backing host_backing = PINNED`"


def test_the_plan_carries_the_three_new_terms_at_zero():
    hpp = read(HPP)
    hit = re.search(r"struct\s+placement_plan\s*\{", hpp)
    body = balanced(hpp, hit.end() - 1)
    for term in ("host_cpu_owned_bytes", "host_mapped_bytes", "host_mapped_source_bytes"):
        assert re.search(rf"size_t\s+{term}\s*=\s*0\s*;", body), f"placement_plan has no `size_t {term} = 0`"


# ---- 2. the writers ----


def helper_bodies(hpp: str) -> list:
    bodies = []
    for name in ("charge_host_weight", "uncharge_host_weight"):
        bodies.append(function_body(hpp, rf"void\s+{name}\s*\(\s*placement_host_backing\s+\w+\s*,\s*size_t\s+\w+\s*\)"))
    return bodies


def without(text: str, bodies: list) -> str:
    for b in bodies:
        text = text.replace(b, "")
    return text


def blank_string_literals(text: str) -> str:
    """Keeps each string literal's delimiters and blanks its inside, so lengths and line numbers are unchanged.

    Built on the same lexer as strip_comments, so a char literal (even `'"'`) is left alone and a raw string's body is
    blanked whole."""

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if m.group("delim") is not None:
            start, end = m.start("raw") - m.start(), m.end("raw") - m.start()
            return tok[:start] + re.sub(r"[^\n]", " ", tok[start:end]) + tok[end:]
        if tok[0] == '"':
            return '"' + " " * (len(tok) - 2) + '"'
        return tok

    return _LEXEME_RE.sub(repl, text)


def host_term_writes(text: str) -> list:
    """Every write to one of the host terms: `+=`, `-=`, a plain `=` and saturating_sub on it."""
    pat = rf"\b({'|'.join(HOST_TERMS)})\b\s*(\+=|-=|=(?!=))\s*([^;]*);"
    writes = [(m.group(1), m.group(2), m.group(3).strip()) for m in re.finditer(pat, text)]
    for m in re.finditer(rf"saturating_sub\s*\(\s*({'|'.join(HOST_TERMS)})\b", text):
        writes.append((m.group(1), "saturating_sub", ""))
    return writes


def test_only_the_helpers_write_the_host_terms():
    hpp = read(HPP)
    cpp = read(CPP)
    for rel, text in ((HPP, without(hpp, helper_bodies(hpp))), (CPP, cpp)):
        # A printf label is not a write: the write scan blanks string-literal contents first.
        stray = [w for w in host_term_writes(blank_string_literals(text)) if not (w[1] == "=" and w[2] == "0")]
        assert not stray, f"{rel}: a host term is written outside charge_host_weight/uncharge_host_weight: {stray}"


def test_the_helpers_test_the_backing():
    hpp = read(HPP)
    charge, uncharge = helper_bodies(hpp)
    for name, body in (("charge_host_weight", charge), ("uncharge_host_weight", uncharge)):
        assert re.search(r"\bswitch\s*\(\s*backing\s*\)|\bbacking\s*==", body), f"{name} does not test the backing"
        for term in HOST_TERMS:
            assert term in body, f"{name} never touches {term}"
    assert re.search(
        r"placement_host_backing\s*::\s*FILE_MAPPED", charge
    ), "charge_host_weight does not single FILE_MAPPED out"
    assert "host_bytes" in charge, "charge_host_weight does not charge host_bytes"


MOVE_HELPER = r"static\s+size_t\s+placement_move_host_expert_backing\s*\("
# The old backing is e.host_backing before the flip or `from`, which the loop matched it against; the new one is
# e.host_backing after the flip or `to`.
MOVE_FLIP = (
    r"\bplan\s*\.\s*uncharge_host_weight\s*\(\s*(?:e\s*\.\s*host_backing|from)\s*,\s*e\s*\.\s*src_size\s*\)\s*;\s*"
    r"e\s*\.\s*host_backing\s*=\s*to\s*;\s*"
    r"plan\s*\.\s*charge_host_weight\s*\(\s*(?:e\s*\.\s*host_backing|to)\s*,\s*e\s*\.\s*src_size\s*\)\s*;"
)


def planner_site_gaps(cpp: str) -> list:
    gaps = []
    singles = len(re.findall(r"\bplan\s*\.\s*charge_host_weight\s*\(", cpp))
    groups = len(re.findall(r"\bplan\s*\.\s*charge_host_weight_entries\s*\(", cpp))
    undo = len(re.findall(r"\bplan\s*\.\s*uncharge_host_weight\s*\(", cpp))
    # single entry: the standalone dense host branch, the layout-change delta (extra),
    # placement_move_host_expert_backing's charge to the new backing; groups: the per-layer dense unit (both KV arms),
    # the MoE host group, the multi-device commit (its standalone dense, dense layer and MoE group all reach it), the
    # cohesive host layer (dense + MoE); undo: the layout-change delta's freed bytes,
    # placement_move_host_expert_backing's uncharge of the old backing
    if singles < 3:
        gaps.append(f"{singles} charge_host_weight call sites in the planner, expected the pinned single-entry sites")
    if groups < 6:
        gaps.append(f"{groups} charge_host_weight_entries call sites in the planner, expected the pinned group sites")
    if undo < 2:
        gaps.append(f"{undo} uncharge_host_weight call sites, expected the layout-change delta's and the move helper's")
    hits = list(re.finditer(MOVE_HELPER, cpp))
    if len(hits) != 1:
        return gaps + [f"{len(hits)} definitions of placement_move_host_expert_backing, want 1"]
    move = balanced(cpp, cpp.index("{", hits[0].end()))
    if not re.search(MOVE_FLIP, move):
        gaps.append("placement_move_host_expert_backing does not uncharge the old backing, flip it and charge the new "
                    "one, each at e.src_size")
    return gaps


def test_the_planner_sites_are_the_pinned_list():
    gaps = planner_site_gaps(read(CPP))
    assert not gaps, "\n".join(gaps)


def test_the_planner_site_check_sees_each_dropped_piece():
    cpp = read(CPP)
    charge = r"(plan\s*\.\s*)charge_host_weight(\s*\(\s*e\s*\.\s*host_backing\s*,\s*e\s*\.\s*src_size\s*\)\s*;)"
    uncharge = r"(plan\s*\.\s*)uncharge_host_weight(\s*\(\s*e\s*\.\s*host_backing\s*,\s*e\s*\.\s*src_size\s*\))"
    mutants = {
        "the move helper's charge is removed": re.sub(charge, r"\1charge_unused\2", cpp, count=1),
        "the move helper's uncharge is removed": re.sub(uncharge, r"\1uncharge_unused\2", cpp, count=1),
        "the move helper charges the old backing": re.sub(
            r"(e\s*\.\s*host_backing\s*=\s*to\s*;\s*plan\s*\.\s*charge_host_weight\s*\(\s*)e\s*\.\s*host_backing",
            r"\1from", cpp, count=1),
        "the move helper charges the device size": re.sub(
            r"(e\s*\.\s*host_backing\s*=\s*to\s*;\s*plan\s*\.\s*charge_host_weight\s*\(\s*e\s*\.\s*host_backing\s*,"
            r"\s*e\s*\.\s*)src_size", r"\1dst_size", cpp, count=1),
    }
    for label, mutated in mutants.items():
        assert mutated != cpp, f"the mutant '{label}' changed nothing"
        assert planner_site_gaps(mutated), f"the mutant '{label}' is not caught"
    # The same flip spelled with the loop's own names is the same accounting, so the check accepts it.
    named = re.sub(uncharge, r"\1uncharge_host_weight(from, e.src_size)", cpp, count=1)
    named = re.sub(charge, r"\1charge_host_weight(to, e.src_size);", named, count=1)
    assert named != cpp, "the from/to spelling changed nothing"
    assert not planner_site_gaps(named), "the flip spelled with from and to is refused"


def test_update_expert_placement_moves_bytes_through_the_helpers():
    hpp = read(HPP)
    body = function_body(hpp, r"bool\s+update_expert_placement\s*\(")
    assert re.search(r"\bcharge_host_weight\s*\(\s*e\s*\.\s*host_backing", body), "demotion does not charge by backing"
    assert re.search(r"\buncharge_host_weight\s*\(\s*e\s*\.\s*host_backing", body), "promotion does not uncharge by backing"
    # The expert loan/refill transaction calls this per move, so it stays O(1): only e changes, and a FILE_MAPPED
    # entry's file bytes leave the staging sources on demotion and return on promotion. The builders' recount before
    # the Totals stays the authority.
    assert not re.search(r"recount_host_mapped_source_bytes\s*\(", body), "a per-move recount walks every entry"
    assert re.search(
        r"saturating_sub\s*\(\s*host_mapped_source_bytes\s*,\s*e\s*\.\s*src_size\s*\)", body
    ), "demotion does not take the mapped expert's file bytes out of the staging sources"
    assert re.search(
        r"host_mapped_source_bytes\s*\+=\s*e\s*\.\s*src_size\s*;", body
    ), "promotion does not return the mapped expert's file bytes to the staging sources"


def test_both_builders_recount_the_mapped_sources_before_the_totals():
    cpp = read(CPP)
    for tag in ("[PLACEMENT] Totals", "[PLACEMENT-MULTI] Totals"):
        at = cpp.index(tag)
        before = cpp[:at]
        last = before.rfind("recount_host_mapped_source_bytes")
        assert last >= 0, f"{tag}: no recount before the print"
        assert at - last < 2500, f"{tag}: the recount is too far above the print to be this plan's"


# ---- 3. the readers ----


MAKE_CANDIDATE_SIGNATURE = r"std::shared_ptr<const\s+lifecycle_plan_snapshot>\s+lifecycle_make_candidate_snapshot\s*\("


def staged_candidate_problems(make_body: str) -> list:
    """What is wrong with the candidate builder's host figures. The stage and the probe both build their candidate
    through lifecycle_make_candidate_snapshot, so that helper is the one writer of a candidate's host bytes."""
    problems = []
    if not re.search(
        r"snapshot\s*->\s*planned_host_bytes\s*=\s*snapshot\s*->\s*plan\s*->\s*placed_host_weight_bytes\s*\(\s*\)\s*;",
        make_body,
    ):
        problems.append("the staged candidate's planned_host_bytes is not the placed figure")
    if not re.search(
        r"snapshot\s*->\s*charged_host_bytes\s*=\s*snapshot\s*->\s*plan\s*->\s*charged_host_weight_bytes\s*\(\s*\)\s*;",
        make_body,
    ):
        problems.append("the staged candidate does not carry the charged figure")
    if "weight_host_bytes" in make_body:
        problems.append("the candidate builder still reads weight_host_bytes")
    return problems


def test_the_snapshot_planned_host_bytes_is_the_placed_figure():
    cpp = read(CPP)
    stage = function_body(cpp, r"void\s+lifecycle_stage_placement_plan\s*\(")
    probe = function_body(cpp, r"void\s+lifecycle_stage_probe_placement_plan\s*\(")
    make = function_body(cpp, MAKE_CANDIDATE_SIGNATURE)
    problems = staged_candidate_problems(make)
    assert not problems, problems
    # one source: the stage and the load-time measure's probe both build their candidate through the helper
    # (tests/test-sycl-tiered-verdict-contract.py reads the same helper for the same fact)
    for name, body in (("lifecycle_stage_placement_plan", stage), ("lifecycle_stage_probe_placement_plan", probe)):
        assert re.search(r"\blifecycle_make_candidate_snapshot\s*\(", body), (
            f"{name} does not build its candidate through lifecycle_make_candidate_snapshot"
        )
    publish = function_body(cpp, r"bool\s+lifecycle_publish_placement_plan\s*\(")
    assert re.search(
        r"published\s*->\s*planned_host_bytes\s*=\s*published\s*->\s*plan\s*->\s*placed_host_weight_bytes\s*\(\s*\)\s*;",
        publish,
    ), "the published snapshot's planned_host_bytes is not the placed figure"
    assert re.search(
        r"published\s*->\s*charged_host_bytes\s*=\s*published\s*->\s*plan\s*->\s*charged_host_weight_bytes\s*\(\s*\)\s*;",
        publish,
    ), "the published snapshot does not carry the charged figure"
    assert "weight_host_bytes" not in stage and "weight_host_bytes" not in publish, "a snapshot still reads weight_host_bytes"


def test_the_lifecycle_state_carries_the_charged_figure():
    hpp = strip_comments((SYCL / "model-lifecycle.hpp").read_text())
    for struct in ("ModelState", "publication_data"):
        hit = re.search(rf"struct\s+{struct}\s*\{{", hpp)
        assert hit, f"{struct} is missing"
        assert "charged_host_bytes" in balanced(hpp, hit.end() - 1), f"{struct} has no charged_host_bytes"
    life = strip_comments((SYCL / "model-lifecycle.cpp").read_text())
    assert re.search(r"publication\s*\.\s*charged_host_bytes", life), "finalize_end does not publish charged_host_bytes"
    gs = strip_comments((SYCL / "ggml-sycl.cpp").read_text())
    assert re.search(
        r"data\s*\.\s*charged_host_bytes\s*=\s*snapshot\s*\.\s*charged_host_bytes\s*;", gs
    ), "the publication is not built from the snapshot's charged figure"
    assert re.search(
        r"publication\s*\.\s*charged_host_bytes", gs
    ), "the prepared ModelState does not carry the publication's charged figure"


def test_the_load_summary_prints_placed_and_charged():
    gs = strip_comments((SYCL / "ggml-sycl.cpp").read_text())
    body = function_body(gs, r"static\s+void\s+ggml_sycl_log_load_summary\s*\(")
    assert re.search(r"placed_host_weight_bytes\s*\(\s*\)", body), "the load summary does not read the placed figure"
    assert re.search(r"charged_host_weight_bytes\s*\(\s*\)", body), "the load summary does not read the charged figure"
    assert "host charged" in body, "the load summary does not label the charged figure"
    assert "host placed" in body, "the load summary does not label the placed figure"


def test_both_totals_lines_print_all_four_terms():
    cpp = read(CPP)
    for tag in ("[PLACEMENT] Totals", "[PLACEMENT-MULTI] Totals"):
        at = cpp.index(tag)
        call = cpp[at : at + 1500]
        for term in ("weight_host_bytes", "host_cpu_owned_bytes", "host_mapped_bytes", "host_mapped_source_bytes"):
            assert term in call, f"{tag}: the Totals line does not print {term}"


def test_the_refresh_counts_cpu_owned_and_not_mapped():
    hpp = read(HPP)
    body = function_body(hpp, r"void\s+refresh_kv_byte_totals\s*\(\s*\)")
    hit = re.search(r"\bhost_bytes\s*=\s*([^;]+);", body)
    assert hit, "refresh_kv_byte_totals does not set host_bytes"
    expr = hit.group(1)
    assert "weight_host_bytes" in expr and "host_cpu_owned_bytes" in expr and "kv_host_bytes" in expr, expr
    assert "host_mapped_bytes" not in expr, "mapped bytes are not charged to host_bytes until locked (design 5efe §4.5)"


# ---- 4. the preload ----


def test_the_preload_serves_in_place_only_what_the_plan_left_mapped():
    src = read("ggml-sycl.cpp")
    start = src.index("auto register_host_aos_expert = [&](")
    body = balanced(src, src.index("{", start))
    cond = body[body.index("served_in_place ="):]
    cond = cond[:cond.index(";")]
    assert "host_backing_sizes_weight_zone(planned_backing)" in cond, cond
    # In place only when the backing does NOT size the WEIGHT zone: a dropped negation serves pinned experts in place.
    assert re.search(
        r"!\s*(?:ggml_sycl\s*::\s*placement_plan\s*::\s*)?host_backing_sizes_weight_zone\s*\(\s*planned_backing\s*\)",
        cond,
    ), cond
    assert cond.index("host_backing_sizes_weight_zone") < cond.index("ggml_sycl_mapped_buffer_owner"), cond
    assert "getenv" not in body

    # A failed pinned allocation over a mapping registers the mapping with its owner, and warns once per tensor.
    fb_at = re.search(r"\}\s*else\s+if\s*\(\s*!\s*tensor_source_is_device\s*\)\s*\{", body)
    assert fb_at, "the lambda has no fallback branch for a host source without pinned room"
    fallback = balanced(body, fb_at.end() - 1)
    owner = re.search(r"ggml_sycl_mapped_buffer_owner\s*\(\s*tensor\s*->\s*buffer\s*,\s*&\s*host_owner\s*\)", fallback)
    assert owner, fallback
    gate = re.search(
        r"\bif\s*\(\s*ggml_sycl_mapped_buffer_owner\s*\(\s*tensor\s*->\s*buffer\s*,\s*&\s*host_owner\s*\)\s*\)\s*\{",
        fallback,
    )
    assert gate, "the owner call does not gate the warning"
    gated = balanced(fallback, gate.end() - 1)
    assert re.search(r"\bGGML_LOG_WARN\s*\(", gated), "the warning is not inside the mapping-owner branch"
    assert re.search(r"\blogged_pin_alloc_failed\s*=\s*true\s*;", gated), "the once-flag is not set inside that branch"
    clear = re.search(r"\bhost_owner\s*=\s*\{\s*\}\s*;", fallback)
    assert clear and clear.start() < owner.start(), "host_owner is not cleared before the owner call"

    # The caller hands the plan's backing over, not a constant.
    assert re.search(r"register_host_aos_expert\s*\(\s*e\s*,\s*expert_aos\s*,\s*placement\s*\.\s*host_backing\s*,", src)


# ---- positive controls ----


def test_the_candidate_clause_rejects_the_old_figure():
    make = function_body(read(CPP), MAKE_CANDIDATE_SIGNATURE)
    placed = re.search(
        r"snapshot\s*->\s*planned_host_bytes\s*=\s*snapshot\s*->\s*plan\s*->\s*placed_host_weight_bytes\s*\(\s*\)\s*;",
        make,
    )
    assert placed, "the control has no placed-figure assignment to replace"
    old = make.replace(placed.group(0), "snapshot->planned_host_bytes = snapshot->plan->weight_host_bytes;")
    problems = staged_candidate_problems(old)
    assert "the staged candidate's planned_host_bytes is not the placed figure" in problems, problems
    assert "the candidate builder still reads weight_host_bytes" in problems, problems


def test_the_write_scanner_sees_a_planted_write():
    planted = "plan.weight_host_bytes += entry.dst_size; saturating_sub(host_mapped_bytes, 4); host_cpu_owned_bytes = 7;"
    found = host_term_writes(planted)
    assert ("weight_host_bytes", "+=", "entry.dst_size") in found, found
    assert ("host_mapped_bytes", "saturating_sub", "") in found, found
    assert ("host_cpu_owned_bytes", "=", "7") in found, found
    assert host_term_writes("plan.weight_host_bytes = 0;") == [("weight_host_bytes", "=", "0")]
    assert host_term_writes("if (a.weight_host_bytes == 0) {}") == []
    # the blanking the real scan applies does not hide a write in code
    assert host_term_writes(blank_string_literals(planted)) == found


def test_the_write_scanner_ignores_a_label_inside_a_string():
    quoted = 'snprintf(text, n, "weight_host_bytes=%zu", x); log("host_cpu_owned_bytes = 7");'
    assert host_term_writes(blank_string_literals(quoted)) == []
    bare = "weight_host_bytes=%zu; host_cpu_owned_bytes = 7;"
    found = host_term_writes(blank_string_literals(bare))
    assert ("weight_host_bytes", "=", "%zu") in found, found
    assert ("host_cpu_owned_bytes", "=", "7") in found, found
    assert blank_string_literals('a("x\\n")\nb').count("\n") == 1


def test_the_write_scanner_sees_a_write_after_a_quote_char_or_a_raw_string():
    # A quote character inside a char literal opens no string, so the write after it stays visible.
    after_char = "if (c == '\"') weight_host_bytes = 5; log(\"x\");"
    assert ("weight_host_bytes", "=", "5") in host_term_writes(blank_string_literals(after_char))
    # A quote inside a raw string closes nothing, so the write after the raw string stays visible. The trailing string
    # is what a quote-pairing blanker would pair the stray quote with, hiding the write between them.
    after_raw = 'auto s = R"(a"b)"; weight_host_bytes = 5; log("x");'
    assert ("weight_host_bytes", "=", "5") in host_term_writes(blank_string_literals(after_raw))
    # A label inside a raw string is not a write, and blanking a raw string keeps its line count.
    raw_label = 'log(R"x(host_cpu_owned_bytes = 7;\n)x");'
    assert host_term_writes(blank_string_literals(raw_label)) == []
    assert blank_string_literals('R"(a\nb)"\nc').count("\n") == 2


# ---- 4. the pin's opt-out ----

GS = "ggml-sycl.cpp"
PIN_ENV = "MOE_PIN_HOST_EXPERTS_ENV"
PIN_READER = "placement_pin_host_experts_enabled"
PIN_GLOBAL = "g_ggml_sycl_moe_pin_host_experts"


def pin_opt_out_gaps(hpp: str, cpp: str, gs: str) -> list:
    gaps = []
    if not re.search(rf'\bconstexpr\s+const\s+char\s*\*\s*{PIN_ENV}\s*=\s*"GGML_SYCL_MOE_PIN_HOST_EXPERTS"\s*;', hpp):
        gaps.append(f"unified-cache.hpp does not name the variable as {PIN_ENV}")
    if not re.search(rf"\bbool\s+{PIN_READER}\s*\(\s*\)\s*;", hpp):
        gaps.append(f"unified-cache.hpp does not declare {PIN_READER}()")
    readers = list(re.finditer(rf"\bbool\s+{PIN_READER}\s*\(\s*\)\s*\{{", cpp))
    if len(readers) != 1:
        return gaps + [f"{len(readers)} definitions of {PIN_READER}, want 1"]
    reader = balanced(cpp, readers[0].end() - 1)
    if not re.search(rf"\bgetenv\s*\(\s*{PIN_ENV}\s*\)", reader):
        gaps.append(f"{PIN_READER} does not read {PIN_ENV}")
    for name, text in (("unified-cache.cpp", cpp.replace(reader, "")), (GS, gs)):
        if re.search(rf"\bgetenv\s*\(\s*(?:ggml_sycl\s*::\s*)?{PIN_ENV}\b|GGML_SYCL_MOE_PIN_HOST_EXPERTS", text):
            gaps.append(f"{name} reads the variable outside {PIN_READER}")
    pin = function_body(cpp, r"static\s+bool\s+placement_pin_routed_experts_for_ring\s*\(")
    if not re.search(rf"\b{PIN_READER}\s*\(\s*\)", pin):
        gaps.append(f"placement_pin_routed_experts_for_ring does not ask {PIN_READER}()")
    table = re.search(r"\bsycl_env_settings\s*\[\s*\]\s*=\s*\{", gs)
    if not table:
        return gaps + ["ggml-sycl.cpp has no sycl_env_settings table"]
    rows = balanced(gs, table.end() - 1)
    if not re.search(rf"\{{\s*ggml_sycl\s*::\s*{PIN_ENV}\s*,\s*&\s*{PIN_GLOBAL}\s*,\s*1\s*\}}", rows):
        gaps.append(f"the settings table has no row {{ ggml_sycl::{PIN_ENV}, &{PIN_GLOBAL}, 1 }}")
    after = gs[table.end() :]
    loop = re.search(r"for\s*\(\s*const\s+sycl_env_setting\s*&\s*\w+\s*:\s*sycl_env_settings\s*\)", after)
    overwrite = re.search(rf"\b{PIN_GLOBAL}\s*=\s*ggml_sycl\s*::\s*{PIN_READER}\s*\(\s*\)", after)
    if not overwrite:
        gaps.append(f"{PIN_GLOBAL} is not overwritten from ggml_sycl::{PIN_READER}()")
    elif not loop or overwrite.start() < loop.start():
        gaps.append(f"{PIN_GLOBAL} is overwritten from the reader before the table parse, which then replaces it")
    return gaps


def pin_opt_out_sources() -> tuple:
    return read(HPP), read(CPP), read(GS)


def test_the_pin_opt_out_has_one_reader_and_a_settings_row():
    gaps = pin_opt_out_gaps(*pin_opt_out_sources())
    assert not gaps, "\n".join(gaps)


def test_the_pin_opt_out_check_sees_each_dropped_piece():
    hpp, cpp, gs = pin_opt_out_sources()
    row = rf"\{{\s*ggml_sycl\s*::\s*{PIN_ENV}\s*,\s*&\s*{PIN_GLOBAL}\s*,\s*1\s*\}}\s*,"
    overwrite = rf"\b{PIN_GLOBAL}\s*=\s*ggml_sycl\s*::\s*{PIN_READER}\s*\(\s*\)[^;]*;"
    mutants = {
        "the settings row is removed": (hpp, cpp, re.sub(row, "", gs, count=1)),
        "the overwrite from the reader is removed": (hpp, cpp, re.sub(overwrite, ";", gs, count=1)),
        "the planner reads the environment itself": (
            hpp,
            re.sub(rf"!\s*{PIN_READER}\s*\(\s*\)", f"!std::getenv({PIN_ENV})", cpp, count=1),
            gs,
        ),
    }
    for label, (m_hpp, m_cpp, m_gs) in mutants.items():
        assert (m_hpp, m_cpp, m_gs) != (hpp, cpp, gs), f"the mutant '{label}' changed nothing"
        assert pin_opt_out_gaps(m_hpp, m_cpp, m_gs), f"the mutant '{label}' is not caught"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
