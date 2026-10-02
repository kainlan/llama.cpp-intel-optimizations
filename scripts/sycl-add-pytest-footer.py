#!/usr/bin/env python3
"""Append the pytest footer to every pytest-style tests/test-sycl-*.py gate that has no `__main__` guard.

A pytest-style gate (something pytest or unittest collects: module-level `def test...`, a Test* class, a
TestCase subclass, `test_x = lambda ...`) run as `python3 tests/test-sycl-x.py` defines its functions and exits 0
having asserted nothing: the vacuous pass that made a python3 loop read a red tree green (llama.cpp-qeld). The
footer makes a direct run execute the file's tests under pytest and exit with pytest's status instead.

    python3 scripts/sycl-add-pytest-footer.py            # append the footer where it is missing
    python3 scripts/sycl-add-pytest-footer.py --check    # list the files that need work, exit 1 if any
    python3 scripts/sycl-add-pytest-footer.py --self-test  # planted cases against the logic (run by ctest)

Idempotent, and it only ever appends: a gate that already has a real top-level `if __name__ == "__main__":`
statement (a manual runner, or its own pytest.main call) is left alone, since a second block would run after the
first; whether that block really runs every test is the audit's R3, not this script's call. A `__main__` that
appears only in a comment, docstring or string is not a guard and does not stop the footer. A file that already
ends with this footer but with the spacing of an older version (one blank line before the guard, flake8 E305) is
normalised to the current one, so the output is the same from either starting point.

The classifier and the guard test are the ones tests/test-sycl-gate-registration-audit.py uses, loaded from it so
the two cannot drift.
"""
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_audit():
    spec = importlib.util.spec_from_file_location("sycl_gate_registration_audit",
                                                  ROOT / "tests" / "test-sycl-gate-registration-audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()
FOOTER = AUDIT.FOOTER
FOOTER_AT_END = re.compile(r"\n*" + re.escape(FOOTER.lstrip("\n")) + r"\Z")


def needs_footer(text):
    return AUDIT.is_pytest_style(text) and not AUDIT.has_main_guard(text)


def with_footer(text):
    """The text this script wants the file to have: the footer appended where it is missing, an existing footer
    normalised to the current spacing, anything else unchanged."""
    if needs_footer(text):
        return text.rstrip("\n") + "\n" + FOOTER
    m = FOOTER_AT_END.search(text)
    if m:
        return text[:m.start()].rstrip("\n") + "\n" + FOOTER
    return text


def self_test():
    """Planted inputs: each shape that must be footered, left alone or normalised, checked through with_footer()."""
    pytest_gate = "def test_x():\n    assert True\n"
    unittest_gate = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        pass\n"
    legacy = "\n" + FOOTER.lstrip("\n")
    footered = [
        ("main-only-in-a-comment", "# never run as __main__\n" + pytest_gate),
        ("main-only-in-a-docstring", '"""never run as __main__"""\n' + pytest_gate),
        ("main-only-in-a-string", pytest_gate + 'x = """\nif __name__ == "__main__":\n    pass\n"""\n'),
        ("guard-inside-a-function", pytest_gate + 'def f():\n    if __name__ == "__main__":\n        pass\n'),
        ("unittest-class", unittest_gate),
        ("test-class", "class TestX:\n    def test_x(self):\n        pass\n"),
        ("def-test", "def test():\n    pass\n"),
        ("def-testFoo", "def testFoo():\n    pass\n"),
        ("nested-under-if", "if True:\n    def test_x():\n        pass\n"),
        ("test-lambda", "test_x = lambda: None\n"),
        ("no-trailing-newline", "def test_x():\n    pass"),
    ]
    untouched = [
        ("script-style", "print('ok')\n"),
        ("def-only-in-a-docstring", '"""\ndef test_x():\n    pass\n"""\nprint(1)\n'),
        ("data-named-test", "test_cases = [1, 2]\n"),
        ("own-real-guard", pytest_gate + '\nif __name__ == "__main__":\n    test_x()\n'),
        ("own-pass-guard", pytest_gate + '\nif __name__ == "__main__":\n    pass\n'),
        ("current-footer", pytest_gate + FOOTER),
    ]
    failures = []
    for label, text in footered:
        wanted = with_footer(text)
        if not wanted.endswith(FOOTER) or wanted == text:
            failures.append("%s: footer not appended" % label)
        elif not AUDIT.runs_itself(wanted):
            failures.append("%s: the appended footer does not satisfy the audit's R3" % label)
        elif with_footer(wanted) != wanted:
            failures.append("%s: second pass changed the file (not idempotent)" % label)
    for label, text in untouched:
        if with_footer(text) != text:
            failures.append("%s: file that must be left alone was changed" % label)
    one_blank = pytest_gate + legacy
    if with_footer(one_blank) != pytest_gate + FOOTER:
        failures.append("legacy-spacing: one-blank-line footer not normalised to the current spacing")
    if with_footer(pytest_gate + FOOTER.lstrip("\n")) != pytest_gate + FOOTER:
        failures.append("legacy-spacing: footer with the guard right after the code not normalised")
    if with_footer(pytest_gate) != with_footer(one_blank):
        failures.append("reproducibility: the output differs between a pre-footer and a legacy-footer starting point")
    for failure in failures:
        print("SELF-TEST FAIL " + failure)
    if not failures:
        print("sycl-add-pytest-footer self-test: PASS (%d footered, %d untouched, normalisation, reproducibility)"
              % (len(footered), len(untouched)))
    return 1 if failures else 0


def main(argv):
    if "--self-test" in argv:
        return self_test()
    check = "--check" in argv
    pending = []
    for path in sorted((ROOT / "tests").glob("test-sycl-*.py")):
        text = path.read_text()
        wanted = with_footer(text)
        if wanted == text:
            continue
        pending.append(("needs footer: " if needs_footer(text) else "normalise footer: ", path.name))
        if not check:
            path.write_text(wanted)
    for what, name in pending:
        print((what if check else what.replace("needs footer", "appended footer").replace("normalise", "normalised")) + name)
    print("%d file(s) %s" % (len(pending), "need the footer" if check else "updated"))
    return 1 if (check and pending) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
