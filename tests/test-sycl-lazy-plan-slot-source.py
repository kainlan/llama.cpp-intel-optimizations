"""Source contract for the lazy tables' coverage pin on its way to the planner (th32 §3.2, §4.3 step 4).

A plan may place a lazy table STORAGE only when its caller pinned the IO pool, and the pin reaches the planner as the
last parameter, `coverage_pin`, of the four planner entry points. A dropped forward compiles: the parameter defaults to
nullptr, so the plan silently refuses every STORAGE table as lazy-table-ring-unpinned. The device tests score the
forwards that a run takes; this gate pins every one of them, including the paths a given host never takes:

1. both planner calls in ggml-sycl.cpp's `compute_and_store_plan_for_inventory`, the single-device one and the
   multi-device one, pass the thread's plan slot `g_sycl_lazy_plan_slot` as their last argument;
2. both pair overloads in unified-cache.cpp, `compute_placement_plan` and `compute_multi_device_plan` taking
   `std::pair<std::string, size_t>` rows, forward `coverage_pin` as the last argument of the planner they call, and
   neither discards it;
3. the multi-device planner's one-device delegation forwards `coverage_pin` to `compute_placement_plan`;
4. both planners record their host terms with `coverage_pin`, and the record hands it to the residency decision;
5. only `lazy_plan_slot_scope` writes the slot, so a pin on it lives exactly as long as the scope that set it, and the
   slot is thread-local.

The self-test applies one text mutant per forward to the unmutated tree, which must pass first, and requires each
mutant to be caught.

Plain file I/O, comment-stripped text.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"

SLOT = "g_sycl_lazy_plan_slot"
PIN = "coverage_pin"
PAIR_ROW = re.compile(r"std::pair\s*<\s*std::string\s*,\s*size_t\s*>")

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


def read(path: Path) -> str:
    return strip_comments(path.read_text())


def close_of(text: str, open_idx: int) -> int:
    """The index of the bracket that closes the one at `open_idx`, counting (), [] and {} together."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] in "([{":
            depth += 1
        elif text[i] in ")]}":
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError("unbalanced brackets")


def name_re(name: str) -> re.Pattern:
    # A call or definition of `name`, qualified or not, never a member call or a longer identifier.
    return re.compile(rf"(?<![\w.])(?<!->){re.escape(name)}\s*\(")


def definitions(text: str, name: str) -> list:
    """(parameter text, body start, body end) for every definition of `name`: its parameter list, then a brace."""
    found = []
    for m in name_re(name).finditer(text):
        open_paren = m.end() - 1
        close_paren = close_of(text, open_paren)
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", text[close_paren + 1 :])
        if tail:
            brace = close_paren + 1 + tail.end() - 1
            found.append((text[open_paren + 1 : close_paren], brace + 1, close_of(text, brace)))
    return found


def split_args(args: str) -> list:
    """The top-level arguments of an argument list, stripped."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(args):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
    tail = args[start:].strip()
    if tail or out:
        out.append(tail)
    return out


def calls_in(text: str, start: int, end: int, callee: str) -> list:
    """(argument-list start, argument-list end) of every call of `callee` between `start` and `end`."""
    spans = []
    for m in name_re(callee).finditer(text, start, end):
        open_paren = m.end() - 1
        spans.append((open_paren + 1, close_of(text, open_paren)))
    return spans


def last_args(text: str, start: int, end: int, callee: str) -> list:
    return [(split_args(text[a:b]) or [""])[-1] for a, b in calls_in(text, start, end, callee)]


def the_definition(text: str, name: str, pair: bool) -> tuple:
    """The one definition of `name` whose rows are pairs (`pair`) or tensor infos."""
    found = [d for d in definitions(text, name) if bool(PAIR_ROW.search(d[0])) == pair]
    assert len(found) == 1, f"{len(found)} {'pair' if pair else 'tensor-info'} definitions of {name}, want 1"
    return found[0]


# --- the contract ---


def stage_violations(sycl: str) -> list:
    out = []
    defs = definitions(sycl, "compute_and_store_plan_for_inventory")
    if len(defs) != 1:
        return [f"{len(defs)} definitions of compute_and_store_plan_for_inventory, want 1"]
    _, start, end = defs[0]
    for callee in ("compute_placement_plan", "compute_multi_device_plan"):
        lasts = last_args(sycl, start, end, callee)
        if not lasts:
            out.append(f"compute_and_store_plan_for_inventory calls no {callee}: the gate lost the path it pins")
        for last in lasts:
            if last != SLOT:
                out.append(f"compute_and_store_plan_for_inventory calls {callee} with last argument '{last}', "
                           f"not the thread's plan slot {SLOT}: the pin never reaches that path's planner")
    return out


def overload_violations(uc: str) -> list:
    out = []
    for name in ("compute_placement_plan", "compute_multi_device_plan"):
        _, start, end = the_definition(uc, name, pair=True)
        if re.search(rf"\bGGML_UNUSED\s*\(\s*{PIN}\s*\)", uc[start:end]):
            out.append(f"the pair overload of {name} discards {PIN}")
        lasts = last_args(uc, start, end, name)
        if lasts != [PIN]:
            out.append(f"the pair overload of {name} calls {name} with last arguments {lasts}, want ['{PIN}']")
    return out


def delegation_violations(uc: str) -> list:
    _, start, end = the_definition(uc, "compute_multi_device_plan", pair=False)
    lasts = last_args(uc, start, end, "compute_placement_plan")
    if not lasts:
        return ["compute_multi_device_plan has no one-device delegation to compute_placement_plan"]
    return [f"compute_multi_device_plan's one-device delegation passes '{last}' as its last argument, not {PIN}"
            for last in lasts if last != PIN]


def record_violations(uc: str) -> list:
    out = []
    for name in ("compute_placement_plan", "compute_multi_device_plan"):
        _, start, end = the_definition(uc, name, pair=False)
        lasts = last_args(uc, start, end, "record_host_ledger_plan")
        if not lasts or any(last != PIN for last in lasts):
            out.append(f"{name} records its host terms with last arguments {lasts}, want {PIN} at every record")
    defs = definitions(uc, "record_host_ledger_plan")
    if len(defs) != 1:
        return out + [f"{len(defs)} definitions of record_host_ledger_plan, want 1"]
    _, start, end = defs[0]
    lasts = last_args(uc, start, end, "placement_plan_decide_lazy_residency")
    if lasts != [PIN]:
        out.append(f"record_host_ledger_plan hands the residency decision {lasts}, want ['{PIN}']")
    return out


def slot_violations(sycl: str) -> list:
    out = []
    if not re.search(rf"\bthread_local\s+const\s+ggml_sycl::ring_lease\s*\*\s*{SLOT}\s*=\s*nullptr\s*;", sycl):
        out.append(f"{SLOT} is not a thread_local const ggml_sycl::ring_lease * initialised to nullptr")
    m = re.search(r"\bclass\s+lazy_plan_slot_scope\s*\{", sycl)
    if not m:
        return out + ["no class lazy_plan_slot_scope"]
    scope_start, scope_end = m.end() - 1, close_of(sycl, m.end() - 1)
    writes = [w.start() for w in re.finditer(rf"(?<![\w*])(?<!\*\s){SLOT}\s*=(?!=)", sycl)]
    if not any(scope_start < w < scope_end for w in writes):
        out.append(f"lazy_plan_slot_scope never writes {SLOT}")
    stray = [w for w in writes if not scope_start < w < scope_end]
    if stray:
        lines = sorted({sycl.count("\n", 0, w) + 1 for w in stray})
        out.append(f"{SLOT} is written outside lazy_plan_slot_scope at line(s) {lines}")
    return out


def violations(sycl: str, uc: str) -> list:
    found = stage_violations(sycl) + overload_violations(uc) + delegation_violations(uc)
    return found + record_violations(uc) + slot_violations(sycl)


# --- the mutants: each drops one forward from a passing tree ---


def edit_last_arg(text: str, start: int, end: int, callee: str, replacement) -> str:
    """Replaces the last argument of the first call of `callee` in [start, end); None removes the argument."""
    spans = calls_in(text, start, end, callee)
    assert spans, f"no call of {callee} to mutate"
    a, b = spans[0]
    args = split_args(text[a:b])
    assert args, f"the call of {callee} has no argument to mutate"
    args = args[:-1] + ([] if replacement is None else [replacement])
    return text[:a] + ", ".join(args) + text[b:]


def in_stage(callee: str, replacement):
    def mutate(sycl: str, uc: str) -> tuple:
        _, start, end = definitions(sycl, "compute_and_store_plan_for_inventory")[0]
        return edit_last_arg(sycl, start, end, callee, replacement), uc

    return mutate


def in_uc(name: str, pair: bool, callee: str, replacement):
    def mutate(sycl: str, uc: str) -> tuple:
        _, start, end = the_definition(uc, name, pair)
        return sycl, edit_last_arg(uc, start, end, callee, replacement)

    return mutate


def in_record(replacement):
    def mutate(sycl: str, uc: str) -> tuple:
        _, start, end = definitions(uc, "record_host_ledger_plan")[0]
        return sycl, edit_last_arg(uc, start, end, "placement_plan_decide_lazy_residency", replacement)

    return mutate


def stray_writer(sycl: str, uc: str) -> tuple:
    return sycl + f"\nstatic void lazy_slot_stray() {{ {SLOT} = nullptr; }}\n", uc


CPP, MULTI, RECORD = "compute_placement_plan", "compute_multi_device_plan", "record_host_ledger_plan"

MUTANTS = {
    "the single-device stage call drops the slot": in_stage(CPP, None),
    "the single-device stage call passes nullptr": in_stage(CPP, "nullptr"),
    "the multi-device stage call drops the slot": in_stage(MULTI, None),
    "the multi-device stage call passes nullptr": in_stage(MULTI, "nullptr"),
    "the pair compute_placement_plan drops the pin": in_uc(CPP, True, CPP, None),
    "the pair compute_multi_device_plan drops the pin": in_uc(MULTI, True, MULTI, None),
    "the one-device delegation drops the pin": in_uc(MULTI, False, CPP, None),
    "compute_placement_plan records with nullptr": in_uc(CPP, False, RECORD, "nullptr"),
    "compute_multi_device_plan records with nullptr": in_uc(MULTI, False, RECORD, "nullptr"),
    "the record hands the decision nullptr": in_record("nullptr"),
    "a second writer of the slot": stray_writer,
}


def sources() -> tuple:
    return read(SYCL / "ggml-sycl.cpp"), read(SYCL / "unified-cache.cpp")


def test_stage_planner_calls_pass_the_slot():
    sycl, _ = sources()
    found = stage_violations(sycl)
    assert not found, "\n".join(found)


def test_pair_overloads_forward_the_pin():
    _, uc = sources()
    found = overload_violations(uc)
    assert not found, "\n".join(found)


def test_one_device_delegation_forwards_the_pin():
    _, uc = sources()
    found = delegation_violations(uc)
    assert not found, "\n".join(found)


def test_planners_record_with_the_pin():
    _, uc = sources()
    found = record_violations(uc)
    assert not found, "\n".join(found)


def test_only_the_scope_writes_the_slot():
    sycl, _ = sources()
    found = slot_violations(sycl)
    assert not found, "\n".join(found)


def test_every_mutant_is_caught():
    sycl, uc = sources()
    base = violations(sycl, uc)
    assert not base, "the unmutated tree fails, so no mutant can be scored:\n" + "\n".join(base)
    missed = []
    for label, mutate in MUTANTS.items():
        mutated_sycl, mutated_uc = mutate(sycl, uc)
        assert (mutated_sycl, mutated_uc) != (sycl, uc), f"the mutant '{label}' changed nothing"
        if not violations(mutated_sycl, mutated_uc):
            missed.append(label)
    assert not missed, "mutants the gate does not catch: " + ", ".join(missed)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
