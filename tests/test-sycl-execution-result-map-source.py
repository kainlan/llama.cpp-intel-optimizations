#!/usr/bin/env python3
"""Every ggml_sycl::execution::error maps to a C result and a name (llama.cpp-ppjj).

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
  D. neither switch has a `default:` label, so the compiler sees a missing enumerator.
  P. both functions sit inside one `#pragma clang diagnostic` push/pop region that makes -Wswitch an error, so a
     new enumerator breaks the build instead of printing a warning that is read past.
Run with --self-test so each pin is proven to catch its mutant.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYCL = ROOT / "ggml/src/ggml-sycl"
C_RESULT = "static ggml_sycl_execution_result ggml_sycl_execution_c_result(ggml_sycl::execution::error e) {"
NAME_FN = "static const char * ggml_sycl_execution_error_name(ggml_sycl::execution::error e) {"
PUSH = "#pragma clang diagnostic push"
ERROR_SWITCH = '#pragma clang diagnostic error "-Wswitch"'
POP = "#pragma clang diagnostic pop"
EXPECTED_C = {
    "ALLOCATION_FAILED": "GGML_SYCL_EXECUTION_ALLOCATION_FAILURE",
    "LOCK_HELD_ALLOCATION": "GGML_SYCL_EXECUTION_INTERNAL_ERROR",
}
OUT_OF_RANGE = "GGML_SYCL_EXECUTION_INTERNAL_ERROR"


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
    inner = body(header, "enum class error {")
    return [n for n in re.findall(r"\b([A-Z][A-Z0-9_]*)\b\s*,", inner)]


def cases(fn_body):
    """{enumerator: returned expression} for every `case E::X: return <expr>;` in the function body."""
    return {m.group(1): " ".join(m.group(2).split())
            for m in re.finditer(r"case\s+E::(\w+)\s*:\s*return\s+([^;]+);", fn_body)}


def trailing_return(fn_body):
    """The expression of the last `return` after the switch's closing brace."""
    sw = body(fn_body, "switch (e) {")
    tail = fn_body[fn_body.find(sw) + len(sw):] if sw else ""
    m = re.search(r"return\s+([^;]+);", tail)
    return " ".join(m.group(1).split()) if m else None


def claims(header, sycl):
    names = enumerators(header)
    c_fn = body(sycl, C_RESULT)
    n_fn = body(sycl, NAME_FN)
    c_cases, n_cases = cases(c_fn), cases(n_fn)
    err = sycl.find(ERROR_SWITCH)
    push = sycl.rfind(PUSH, 0, err)
    pop = sycl.find(POP, err)
    region = sycl[err:pop] if 0 <= push < err < pop else ""
    return {
        "enum parsed": len(names) >= 11 and "OK" in names,
        "C every enumerator has a c_result case": bool(c_fn) and all(n in c_cases for n in names),
        "M ALLOCATION_FAILED maps to ALLOCATION_FAILURE": c_cases.get("ALLOCATION_FAILED") ==
            EXPECTED_C["ALLOCATION_FAILED"],
        "M LOCK_HELD_ALLOCATION maps to INTERNAL_ERROR": c_cases.get("LOCK_HELD_ALLOCATION") ==
            EXPECTED_C["LOCK_HELD_ALLOCATION"],
        "M out of range maps to INTERNAL_ERROR": trailing_return(c_fn) == OUT_OF_RANGE,
        "N every enumerator has a name case": bool(n_fn) and all(n in n_cases for n in names),
        "N each name is the enumerator's own": all(n_cases.get(n) == f'"{n}"' for n in names),
        "D no default in c_result": c_fn != "" and re.search(r"\bdefault\s*:", c_fn) is None,
        "D no default in error_name": n_fn != "" and re.search(r"\bdefault\s*:", n_fn) is None,
        "P -Wswitch is an error across both functions": C_RESULT in region and NAME_FN in region and
            region.count("\nstatic ") == 2 and pop >= sycl.find(NAME_FN) + len(NAME_FN) + len(n_fn) - 1,
    }


def mutants(header, sycl):
    """(claim that must fail, header, sycl) for each single mutation of the real text."""
    def sub(text, pattern, repl):
        out, n = re.subn(pattern, repl, text, count=1)
        assert n == 1, pattern
        return out

    def case(name, value):
        return r"case\s+E::" + name + r":\s*return\s+" + value + ";"

    extra = lambda h: sub(h, r"LOCK_HELD_ALLOCATION,\n", "LOCK_HELD_ALLOCATION,\n    NEW_VALUE,\n")
    return [
        ("C every enumerator has a c_result case", header,
         sub(sycl, case("ALLOCATION_FAILED", "GGML_SYCL_EXECUTION_ALLOCATION_FAILURE"), "")),
        ("M ALLOCATION_FAILED maps to ALLOCATION_FAILURE", header,
         sub(sycl, case("ALLOCATION_FAILED", "GGML_SYCL_EXECUTION_ALLOCATION_FAILURE"),
             "case E::ALLOCATION_FAILED: return GGML_SYCL_EXECUTION_BUSY;")),
        ("M LOCK_HELD_ALLOCATION maps to INTERNAL_ERROR", header,
         sub(sycl, case("LOCK_HELD_ALLOCATION", "GGML_SYCL_EXECUTION_INTERNAL_ERROR"),
             "case E::LOCK_HELD_ALLOCATION: return GGML_SYCL_EXECUTION_ALLOCATION_FAILURE;")),
        ("M out of range maps to INTERNAL_ERROR", header,
         sub(sycl, r"\}\s*return GGML_SYCL_EXECUTION_INTERNAL_ERROR;\s*\}\s*static const char \*",
             "} return GGML_SYCL_EXECUTION_BUSY; } static const char *")),
        ("N every enumerator has a name case", header,
         sub(sycl, case("LOCK_HELD_ALLOCATION", '"LOCK_HELD_ALLOCATION"'), "")),
        ("N each name is the enumerator's own", header,
         sub(sycl, case("ALLOCATION_FAILED", '"ALLOCATION_FAILED"'), 'case E::ALLOCATION_FAILED: return "UNKNOWN";')),
        ("D no default in c_result", header,
         sub(sycl, case("NOT_FOUND", "GGML_SYCL_EXECUTION_NOT_FOUND"), "default: return GGML_SYCL_EXECUTION_NOT_FOUND;")),
        ("D no default in error_name", header,
         sub(sycl, case("NOT_FOUND", '"NOT_FOUND"'), 'default: return "NOT_FOUND";')),
        ("P -Wswitch is an error across both functions", header, sub(sycl, re.escape(ERROR_SWITCH), "")),
        ("P -Wswitch is an error across both functions", header, sub(sycl, re.escape(NAME_FN), POP + "\n" + NAME_FN)),
        ("C every enumerator has a c_result case", extra(header), sycl),
        ("N every enumerator has a name case", extra(header), sycl),
    ]


def main():
    header = blank_comments((SYCL / "execution-lifecycle.hpp").read_text())
    sycl = blank_comments((SYCL / "ggml-sycl.cpp").read_text())
    failed = [name for name, ok in claims(header, sycl).items() if not ok]
    for name in failed:
        print(f"FAIL: {name}")
    if "--self-test" in sys.argv and not failed:
        for claim, h, s in mutants(header, sycl):
            if claims(h, s)[claim]:
                print(f"FAIL: the pin '{claim}' did not catch its mutant")
                failed.append(claim)
    print("FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
