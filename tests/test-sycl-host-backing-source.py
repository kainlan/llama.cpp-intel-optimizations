"""Source contract for llama.cpp-5efe C4 (part c): the planner's host-weight accounting is split by host backing.

The runtime half is the pure host test `test-sycl-host-backing-accounting` (each backing's term, placed against charged,
the per-entry group split, the refresh, the mapped-source recount, an expert moving between tiers). This gate pins what
that test cannot see: that every writer of the host terms goes through the one helper that tests the backing, and that
every reader reads the meaning design 5efe §4.4 assigns it.

1. `placement_entry::host_backing` is a `placement_host_backing` defaulting to PINNED, the enum has the four values, and
   `placement_plan` carries `host_cpu_owned_bytes`, `host_mapped_bytes` and `host_mapped_source_bytes`;
2. nothing writes `weight_host_bytes`, `host_cpu_owned_bytes` or `host_mapped_bytes` except the plan's own
   `charge_host_weight` / `uncharge_host_weight` (and the `= 0` initialisation of a new plan): every accumulation site in
   the planner (the single-device and multi-device builders, the layout-change delta, `update_expert_placement`) calls
   one of the helpers, and the sites are the pinned list below;
3. the readers: the plan snapshot's `planned_host_bytes` (staged candidate and published) is the PLACED figure; the
   lifecycle publication carries `charged_host_bytes` beside it into `ModelState`; the load summary prints placed and
   charged; both `Totals` lines print all four terms; `refresh_kv_byte_totals` counts the cpu-owned bytes and not the
   mapped ones.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

_LEXEME_RE = re.compile(
    r'"(?:\\.|[^"\\\n])*"' r"|'(?:\\.|[^'\\\n])*'" r"|//[^\n]*" r"|/\*.*?\*/",
    flags=re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
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
        stray = [w for w in host_term_writes(text) if not (w[1] == "=" and w[2] == "0")]
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


def test_the_planner_sites_are_the_pinned_list():
    cpp = read(CPP)
    singles = len(re.findall(r"\bplan\s*\.\s*charge_host_weight\s*\(", cpp))
    groups = len(re.findall(r"\bplan\s*\.\s*charge_host_weight_entries\s*\(", cpp))
    undo = len(re.findall(r"\bplan\s*\.\s*uncharge_host_weight\s*\(", cpp))
    # single entry: the standalone dense host branch, the layout-change delta (extra), the multi-device standalone dense
    # via commit; groups: the per-layer dense unit (both KV arms), the MoE host group, the multi-device commit
    # (dense layer, MoE group), the cohesive host layer (dense + MoE); undo: the layout-change delta's freed bytes
    assert singles >= 2, f"{singles} charge_host_weight call sites in the planner, expected the pinned single-entry sites"
    assert groups >= 6, f"{groups} charge_host_weight_entries call sites in the planner, expected the pinned group sites"
    assert undo >= 1, "the layout-change delta does not uncharge through the helper"


def test_update_expert_placement_moves_bytes_through_the_helpers():
    hpp = read(HPP)
    body = function_body(hpp, r"bool\s+update_expert_placement\s*\(")
    assert re.search(r"\bcharge_host_weight\s*\(\s*e\s*\.\s*host_backing", body), "demotion does not charge by backing"
    assert re.search(r"\buncharge_host_weight\s*\(\s*e\s*\.\s*host_backing", body), "promotion does not uncharge by backing"
    assert re.search(r"recount_host_mapped_source_bytes\s*\(", body), "the mapped-source term is not re-derived"


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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
