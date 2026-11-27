#!/usr/bin/env python3
"""Every ggml_sycl::execution::error maps to a C result and a name.

With comments blanked, in ggml-sycl.cpp:
  C. ggml_sycl_execution_c_result has a `case E::<X>:` that returns a value for every enumerator of
     ggml_sycl::execution::error (execution-lifecycle.hpp). Before this pin ALLOCATION_FAILED and
     LOCK_HELD_ALLOCATION fell through to GGML_SYCL_EXECUTION_BUSY, which a C caller may retry.
  M. the mapping is the one the enumerator's meaning decides: ALLOCATION_FAILED is
     GGML_SYCL_EXECUTION_ALLOCATION_FAILURE (the value ggml_sycl_execution_caught_bad_alloc returns for the same
     condition), LOCK_HELD_ALLOCATION is GGML_SYCL_EXECUTION_INTERNAL_ERROR (a persistent allocation about to run
     under the registry mutex is a code defect, not memory pressure, and must not be retried), and a value outside
     the enum also ends in INTERNAL_ERROR, never BUSY.
  N. ggml_sycl_execution_error_name has a case for every enumerator and returns the enumerator's own name.
  E. the enum body and both functions are found and non-empty, and every enumerator parses, so a rename fails the
     test instead of leaving the coverage pins with nothing to check.
  D. neither switch has a `default:` label, so the compiler sees a missing enumerator.
  P. each of the two functions lies wholly inside a `#pragma clang diagnostic` push/pop region in which -Wswitch is
     an error, and no pragma naming -Wswitch inside or before it relaxes that (ignored, warning, remark, or a pop
     that ends the region early). A new enumerator then breaks the build instead of printing a warning that is
     read past. Whether the two functions share one region or each has its own does not matter.

What this does not prove:
  - It reads source text; it never executes the mapping. The clang -Wswitch error pragma is the executable guard:
    the compiler, not this test, rejects a switch with a missing enumerator.
  - It does not see enumerators produced by a macro, or a `case` that returns through a helper or a ternary: the
    case pattern is `case E::X: return <expr>;`, and a different shape reads as a missing case (the test fails
    closed, it does not pass).
  - Preprocessor conditionals are not evaluated: a pragma or an enumerator under `#if` counts as present. Lines
    starting with `#` inside the enum body are skipped.
  - GCC ignores clang pragmas; the pragma guard holds only for clang/icpx, the compilers of this translation unit.
  - ggml_sycl_lifecycle_c_result and other switches are out of scope.
Run with --self-test so each pin is proven to catch its mutant (and only the pins it should), and each positive
control is proven to leave every pin green.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
C_RESULT = "static ggml_sycl_execution_result ggml_sycl_execution_c_result(ggml_sycl::execution::error e) {"
NAME_FN = "static const char * ggml_sycl_execution_error_name(ggml_sycl::execution::error e) {"
FUNCTIONS = (("ggml_sycl_execution_c_result", C_RESULT), ("ggml_sycl_execution_error_name", NAME_FN))
PUSH = "#pragma clang diagnostic push"
ERROR_SWITCH = '#pragma clang diagnostic error "-Wswitch"'
POP = "#pragma clang diagnostic pop"
IGNORE_SWITCH = '#pragma clang diagnostic ignored "-Wswitch"'
WARN_SWITCH = '#pragma clang diagnostic warning "-Wswitch"'
EXPECTED_C = {
    "ALLOCATION_FAILED": "GGML_SYCL_EXECUTION_ALLOCATION_FAILURE",
    "LOCK_HELD_ALLOCATION": "GGML_SYCL_EXECUTION_INTERNAL_ERROR",
}
OUT_OF_RANGE = "GGML_SYCL_EXECUTION_INTERNAL_ERROR"
# clang honours `#pragma GCC diagnostic` on the same stack, so both spellings count.
PRAGMA = re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+(?:clang|GCC)[ \t]+diagnostic[ \t]+'
                    r'(push|pop|(error|fatal|warning|ignored|remark)[ \t]+"([^"]*)")', re.M)

C_PIN = "C every enumerator has a c_result case"
N_PIN = "N every enumerator has a name case"
P_PIN = "P -Wswitch is an error across both functions"
E_ENUM = "E the enum body is found and every enumerator parses"
E_C = "E ggml_sycl_execution_c_result is found with cases"
E_N = "E ggml_sycl_execution_error_name is found with cases"
M_ALLOC = "M ALLOCATION_FAILED maps to ALLOCATION_FAILURE"
M_LOCK = "M LOCK_HELD_ALLOCATION maps to INTERNAL_ERROR"
M_TAIL = "M out of range maps to INTERNAL_ERROR"
N_OWN = "N each name is the enumerator's own"
D_C = "D no default in c_result"
D_N = "D no default in error_name"


class MutantNotApplied(Exception):
    pass


def blank_comments(text):
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def body(text, signature):
    """The brace-balanced body after the first occurrence of `signature`, or '' if absent."""
    start = text.find(signature)
    if start < 0:
        return ""
    open_at = text.find("{", start + len(signature) - 1)
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[open_at:i + 1]
    return ""


def enumerators(header):
    """(names, unparsed): the identifier of each enumerator, and the text of each item that has none."""
    inner = body(header, "enum class error {")[1:-1]
    inner = "\n".join(line for line in inner.split("\n") if not line.lstrip().startswith("#"))
    names, unparsed = [], []
    for item in (item.strip() for item in inner.split(",")):
        if not item:
            continue
        m = re.match(r"[A-Za-z_]\w*", item)
        (names if m else unparsed).append(m.group(0) if m else item)
    return names, unparsed


def cases(fn_body):
    """{enumerator: returned expression} for every `case E::X: return <expr>;` in the function body."""
    return {m.group(1): " ".join(m.group(2).split())
            for m in re.finditer(r"case\s+E::(\w+)\s*:\s*return\s+([^;]+);", fn_body)}


def trailing_return(fn_body):
    """The expression of the first `return` after the switch's closing brace (the out-of-range tail)."""
    sw = body(fn_body, "switch (e) {")
    tail = fn_body[fn_body.find(sw) + len(sw):] if sw else ""
    m = re.search(r"return\s+([^;]+);", tail)
    return " ".join(m.group(1).split()) if m else None


def switch_states(text):
    """[(position, -Wswitch is an error from here on)] for every diagnostic pragma, in file order."""
    events, stack, state = [(0, False)], [], False
    for m in PRAGMA.finditer(text):
        if m.group(1) == "push":
            stack.append(state)
        elif m.group(1) == "pop":
            state = stack.pop() if stack else False
        elif m.group(3) == "-Wswitch":
            state = m.group(2) in ("error", "fatal")
        events.append((m.start(), state))
    return events


def uncovered(text):
    """Names of the functions whose body is not wholly under a -Wswitch error (or that are missing)."""
    events = switch_states(text)
    out = []
    for name, signature in FUNCTIONS:
        fn = body(text, signature)
        start = text.find(signature)
        if not fn or start < 0:
            out.append(name)
            continue
        end = start + len(signature) - 1 + len(fn)
        at_start = [state for pos, state in events if pos < start][-1]
        inside = [state for pos, state in events if start < pos < end]
        if not (at_start and all(inside)):
            out.append(name)
    return out


def claims(header, sycl):
    """{pin: '' when it holds, else what is wrong}."""
    names, unparsed = enumerators(header)
    c_fn, n_fn = body(sycl, C_RESULT), body(sycl, NAME_FN)
    c_cases, n_cases = cases(c_fn), cases(n_fn)
    c_missing = [n for n in names if n not in c_cases]
    n_missing = [n for n in names if n not in n_cases]
    wrong_names = [n for n in names if n in n_cases and n_cases[n] != f'"{n}"']
    not_error = uncovered(sycl)

    def verdict(ok, why):
        return "" if ok else why

    def mapping(name):
        got = c_cases.get(name)
        return verdict(got == EXPECTED_C[name], f"c_result maps {name} to {got!r}, want {EXPECTED_C[name]}")

    return {
        E_ENUM: verdict(names and "OK" in names and not unparsed,
                        f"could not parse enumerator {unparsed[0]!r}" if unparsed else "enum class error not found"),
        E_C: verdict(c_fn and c_cases, "ggml_sycl_execution_c_result not found, or it has no `case E::X: return`"),
        E_N: verdict(n_fn and n_cases, "ggml_sycl_execution_error_name not found, or it has no `case E::X: return`"),
        C_PIN: verdict(names and c_fn and not c_missing, f"missing: {', '.join(c_missing) or 'everything'}"),
        M_ALLOC: mapping("ALLOCATION_FAILED"),
        M_LOCK: mapping("LOCK_HELD_ALLOCATION"),
        M_TAIL: verdict(trailing_return(c_fn) == OUT_OF_RANGE,
                        f"the tail returns {trailing_return(c_fn)!r}, want {OUT_OF_RANGE}"),
        N_PIN: verdict(names and n_fn and not n_missing, f"missing: {', '.join(n_missing) or 'everything'}"),
        N_OWN: verdict(names and not wrong_names, f"wrong name for: {', '.join(wrong_names) or 'everything'}"),
        D_C: verdict(c_fn and re.search(r"\bdefault\s*:", c_fn) is None, "c_result has a default label, or is missing"),
        D_N: verdict(n_fn and re.search(r"\bdefault\s*:", n_fn) is None,
                     "error_name has a default label, or is missing"),
        P_PIN: verdict(not not_error, f"-Wswitch is not an error across: {', '.join(not_error)}"),
    }


def sub(text, pattern, repl):
    """`text` with the one match of `pattern` replaced; any other count means the mutant is stale."""
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise MutantNotApplied(f"NOT APPLIED: anchor matches {len(found)} times, not once: {pattern[:70]!r}")
    return re.sub(pattern, lambda m: m.expand(repl), text, count=1)


def case(name, value):
    return r"case\s+E::" + name + r":\s*return\s+" + value + ";"


def enum_edit(header, items):
    return sub(header, r"LOCK_HELD_ALLOCATION,\n", "LOCK_HELD_ALLOCATION,\n" + items)


def mutants():
    """(name, {pins that must fail, exactly}, builder(header, sycl) -> (header, sycl))."""
    def on_sycl(*edit):
        return lambda h, s: (h, sub(s, *edit))

    def on_header(*edit):
        return lambda h, s: (sub(h, *edit), s)

    return [
        ("drop the ALLOCATION_FAILED case of c_result", {C_PIN, M_ALLOC},
         on_sycl(case("ALLOCATION_FAILED", "GGML_SYCL_EXECUTION_ALLOCATION_FAILURE"), "")),
        ("ALLOCATION_FAILED maps to BUSY", {M_ALLOC},
         on_sycl(case("ALLOCATION_FAILED", "GGML_SYCL_EXECUTION_ALLOCATION_FAILURE"),
                 "case E::ALLOCATION_FAILED: return GGML_SYCL_EXECUTION_BUSY;")),
        ("LOCK_HELD_ALLOCATION maps to ALLOCATION_FAILURE", {M_LOCK},
         on_sycl(case("LOCK_HELD_ALLOCATION", "GGML_SYCL_EXECUTION_INTERNAL_ERROR"),
                 "case E::LOCK_HELD_ALLOCATION: return GGML_SYCL_EXECUTION_ALLOCATION_FAILURE;")),
        ("the out-of-range tail returns BUSY", {M_TAIL},
         on_sycl(r"(\}\s*)return GGML_SYCL_EXECUTION_INTERNAL_ERROR;(\s*\}\s*static const char \*)",
                 r"\1return GGML_SYCL_EXECUTION_BUSY;\2")),
        ("drop the LOCK_HELD_ALLOCATION case of error_name", {N_PIN},
         on_sycl(case("LOCK_HELD_ALLOCATION", '"LOCK_HELD_ALLOCATION"'), "")),
        ("error_name names ALLOCATION_FAILED UNKNOWN", {N_OWN},
         on_sycl(case("ALLOCATION_FAILED", '"ALLOCATION_FAILED"'), 'case E::ALLOCATION_FAILED: return "UNKNOWN";')),
        ("c_result gets a default in place of the NOT_FOUND case", {D_C, C_PIN},
         on_sycl(case("NOT_FOUND", "GGML_SYCL_EXECUTION_NOT_FOUND"), "default: return GGML_SYCL_EXECUTION_NOT_FOUND;")),
        ("error_name gets a default in place of the NOT_FOUND case", {D_N, N_PIN}, on_sycl(case("NOT_FOUND", '"NOT_FOUND"'), 'default: return "NOT_FOUND";')),
        ("the error pragma is removed", {P_PIN}, on_sycl(re.escape(ERROR_SWITCH), "")),
        ("the pop moves before error_name", {P_PIN}, on_sycl(re.escape(NAME_FN), POP + "\n" + NAME_FN)),
        ("an inner ignored pragma relaxes c_result", {P_PIN},
         on_sycl(r"(" + re.escape(C_RESULT) + r"\s*using E = [^;]*;\s*)switch \(e\) \{", r"\1" + IGNORE_SWITCH + "\nswitch (e) {")),
        ("an ignored pragma between the functions relaxes error_name", {P_PIN},
         on_sycl(re.escape(NAME_FN), IGNORE_SWITCH + "\n" + NAME_FN)),
        ("an inner warning pragma relaxes error_name", {P_PIN},
         on_sycl(re.escape(NAME_FN), WARN_SWITCH + "\n" + NAME_FN)),
        ("the enum is renamed", {E_ENUM, C_PIN, N_PIN, N_OWN}, on_header(r"enum class error \{", "enum class errors {")),
        ("an enumerator is not an identifier", {E_ENUM},
         lambda h, s: (enum_edit(h, "    42,\n"), s)),
        ("c_result is renamed", {E_C, C_PIN, M_ALLOC, M_LOCK, M_TAIL, D_C, P_PIN},
         on_sycl(r"ggml_sycl_execution_c_result\(ggml_sycl::execution::error e\) \{",
                 "ggml_sycl_execution_c_result_renamed(ggml_sycl::execution::error e) {")),
        ("error_name is renamed", {E_N, N_PIN, D_N, P_PIN},
         on_sycl(r"ggml_sycl_execution_error_name\(ggml_sycl::execution::error e\) \{",
                 "ggml_sycl_execution_error_name_renamed(ggml_sycl::execution::error e) {")),
        ("a new enumerator", {C_PIN, N_PIN}, lambda h, s: (enum_edit(h, "    NEW_VALUE,\n"), s)),
        ("a new last enumerator without a trailing comma", {C_PIN, N_PIN},
         lambda h, s: (enum_edit(h, "    NEW_VALUE\n"), s)),
        ("a new enumerator with an initializer", {C_PIN, N_PIN},
         lambda h, s: (enum_edit(h, "    NEW_VALUE = 11,\n"), s)),
    ]


def controls():
    """(name, builder) for legal edits after which every pin must still hold."""
    def widen(h, s):
        moved = PUSH + "\n" + ERROR_SWITCH + "\n"
        anchor = "static ggml_sycl::lifecycle::ModelToken ggml_sycl_cpp_token("
        return h, sub(sub(s, re.escape(moved), ""), re.escape(anchor), moved + anchor)

    def own_regions(h, s):
        return h, sub(s, re.escape(NAME_FN), POP + "\n" + PUSH + "\n" + ERROR_SWITCH + "\n" + NAME_FN)

    return [
        ("one region widened to the function above c_result", widen),
        ("each function in a region of its own", own_regions),
        ("a preprocessor line inside the enum", lambda h, s: (enum_edit(h, "#if 1\n#endif\n"), s)),
    ]


def self_test(header, sycl):
    failed = []
    for name, expected, build in mutants():
        try:
            h, s = build(header, sycl)
        except MutantNotApplied as e:
            print(f"FAIL: mutant '{name}': {e}")
            failed.append(name)
            continue
        got = {pin for pin, why in claims(h, s).items() if why}
        if got != expected:
            print(f"FAIL: mutant '{name}': pins that failed {sorted(got)}, expected {sorted(expected)}")
            failed.append(name)
    for name, build in controls():
        try:
            h, s = build(header, sycl)
        except MutantNotApplied as e:
            print(f"FAIL: control '{name}': {e}")
            failed.append(name)
            continue
        broke = [f"{pin} ({why})" for pin, why in claims(h, s).items() if why]
        if broke:
            print(f"FAIL: control '{name}' is legal but broke: {'; '.join(broke)}")
            failed.append(name)
    try:
        sub("nothing here", "absent anchor", "")
        print("FAIL: a stale mutant anchor is not reported NOT APPLIED")
        failed.append("stale-anchor")
    except MutantNotApplied:
        pass
    return failed


def main():
    header = blank_comments((SYCL / "execution-lifecycle.hpp").read_text())
    sycl = blank_comments((SYCL / "ggml-sycl.cpp").read_text())
    failed = []
    for pin, why in claims(header, sycl).items():
        if why:
            print(f"FAIL: {pin} ({why})")
            failed.append(pin)
    if "--self-test" in sys.argv and not failed:
        failed += self_test(header, sycl)
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
