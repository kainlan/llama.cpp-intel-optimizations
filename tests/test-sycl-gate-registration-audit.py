#!/usr/bin/env python3
"""Every tests/test-sycl-*.py gate is registered with ctest, and in a form that cannot pass vacuously.

A gate nobody registered never runs in any sweep, so it can go stale unnoticed (llama.cpp-qeld found seven:
six that sat green and unregistered, and a family that was red for weeks). And a pytest-style module (module-level
`test_*` functions) registered with plain `python3 file.py` executes nothing at all: importing it defines functions
and exits 0. That is the vacuous-pass shape inside ctest itself, so such a module must be registered through
`llama_test_pytest`, which hands it to pytest and treats "no tests collected" as a failure.

Rules, per tests/test-sycl-*.py:
  R1  it is named by an add_test / llama_test_pytest / llama_test_cmd in a CMakeLists.txt (comments do not count),
      or it is in UNREGISTERED_ALLOWLIST with a reason;
  R2  if it has module-level `def test_...`, every such registration is llama_test_pytest;
  R3  if it has module-level `def test_...`, it can run itself: it ends with the pytest footer that
      scripts/sycl-add-pytest-footer.py appends, or it carries its own `__main__` block, so running it directly
      (`python3 tests/test-sycl-x.py`) executes its tests or fails loudly instead of passing vacuously;
  R4  UNREGISTERED_ALLOWLIST holds no stale entry (a file that is registered, or no longer exists);
  R5  every test-sycl-*.py a registration names exists under tests/ (a registration of a file that was never
      committed fails at ctest time with "file not found", or worse, only passes on the one checkout that has the
      file), unless it is in MISSING_FILE_ALLOWLIST with a reason; an allowlisted name whose file now exists is
      stale.

`--self-test` also proves the audit can fail: it plants an unregistered gate, a pytest-style module registered with
add_test, and a pytest-style module without the footer into temp trees (synthetic, and a copy of the real tree) and
requires each to be reported, and a clean synthetic tree to report nothing.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REGISTRARS = ("add_test", "llama_test_pytest", "llama_test_cmd")
SKIP_DIRS = {".git", ".llm-wiki", "artifacts", "media", "models", "node_modules", "build"}
# name -> reason. A gate listed here is deliberately not in ctest; empty means every gate runs.
UNREGISTERED_ALLOWLIST = {}
# name -> reason. A registered gate whose file is not in the tree. Empty: every registered gate is committed.
# An entry goes stale (and is reported) the moment its file appears.
MISSING_FILE_ALLOWLIST = {}
REQUIRE_PYTEST_FOOTER = True
FOOTER_RE = re.compile(r'if __name__ == "__main__":\s*\n(?:\s+import [A-Za-z_.]+\s*\n)*\s+sys\.exit\(pytest\.main\(\[__file__, "-q"\]\)\)\s*$')


def cmake_commands(text):
    """(lowercased name, argument text) for every command, skipping comments; paren-balanced, quote- and
    bracket-argument-aware."""
    i, n, out = 0, len(text), []
    ws = re.compile(r"\s*")
    ident = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
    while i < n:
        i = ws.match(text, i).end()
        if i >= n:
            break
        if text[i] == "#":
            bracket = re.match(r"#\[(=*)\[", text[i:])
            if bracket:
                end = text.find("]" + bracket.group(1) + "]", i)
                i = n if end < 0 else end + len(bracket.group(1)) + 2
            else:
                end = text.find("\n", i)
                i = n if end < 0 else end + 1
            continue
        m = ident.match(text, i)
        if not m:
            i += 1
            continue
        j = m.end()
        k = re.compile(r"[ \t]*").match(text, j).end()
        if k >= n or text[k] != "(":
            i = j
            continue
        depth, p = 0, k
        while p < n:
            c = text[p]
            if c == "#":
                bracket = re.match(r"#\[(=*)\[", text[p:])
                if bracket:
                    end = text.find("]" + bracket.group(1) + "]", p)
                    p = n if end < 0 else end + len(bracket.group(1)) + 2
                    continue
                end = text.find("\n", p)
                p = n if end < 0 else end
                continue
            if c == '"':
                p += 1
                while p < n and text[p] != '"':
                    p += 2 if text[p] == "\\" else 1
            elif c == "[":
                bracket = re.match(r"\[(=*)\[", text[p:])
                if bracket:
                    end = text.find("]" + bracket.group(1) + "]", p)
                    p = n if end < 0 else end + len(bracket.group(1)) + 2
                    continue
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            p += 1
        out.append((m.group(0).lower(), text[k + 1:p]))
        i = p + 1
    return out


def registrations(root):
    """gate file name -> sorted list of the registering command names that mention it."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith("build")]
        if "CMakeLists.txt" not in filenames:
            continue
        text = (Path(dirpath) / "CMakeLists.txt").read_text(errors="replace")
        for name, args in cmake_commands(text):
            if name not in REGISTRARS:
                continue
            for gate in set(re.findall(r"test-sycl-[A-Za-z0-9_-]+\.py", args)):
                found.setdefault(gate, []).append(name)
    return {gate: sorted(forms) for gate, forms in found.items()}


def is_pytest_style(text):
    return re.search(r"^(?:async )?def test_", text, re.M) is not None


def audit(root, require_footer=REQUIRE_PYTEST_FOOTER, allowlist=None, missing_allowlist=None):
    root = Path(root)
    allowlist = UNREGISTERED_ALLOWLIST if allowlist is None else allowlist
    missing_allowlist = MISSING_FILE_ALLOWLIST if missing_allowlist is None else missing_allowlist
    registered = registrations(root)
    problems = []
    gates = sorted((root / "tests").glob("test-sycl-*.py"))
    names = {gate.name for gate in gates}
    for gate in gates:
        text = gate.read_text(errors="replace")
        forms = registered.get(gate.name, [])
        if not forms:
            if gate.name not in allowlist:
                problems.append("R1 %s is not registered with ctest (no add_test/llama_test_pytest/llama_test_cmd names it)"
                                % gate.name)
            continue
        if gate.name in allowlist:
            problems.append("R4 %s is allowlisted as unregistered but is registered" % gate.name)
        if is_pytest_style(text):
            if set(forms) != {"llama_test_pytest"}:
                problems.append("R2 %s is pytest-style but registered via %s; plain python3 runs none of its tests"
                                % (gate.name, "/".join(forms)))
            if require_footer and not FOOTER_RE.search(text) and "__main__" not in text:
                problems.append("R3 %s is pytest-style but cannot run itself: no pytest footer and no __main__ block "
                                "(run scripts/sycl-add-pytest-footer.py)" % gate.name)
    for name in sorted(allowlist):
        if name not in names:
            problems.append("R4 %s is allowlisted but does not exist" % name)
    for name in sorted(registered):
        if name not in names and name not in missing_allowlist:
            problems.append("R5 %s is named by a registration (%s) but is not in tests/" % (name, "/".join(registered[name])))
    for name in sorted(missing_allowlist):
        if name in names:
            problems.append("R5 %s is allowlisted as missing but exists; drop the entry" % name)
        elif name not in registered:
            problems.append("R5 %s is allowlisted as missing but nothing registers it" % name)
    return problems


# --- positive controls --------------------------------------------------------------------------------------------

FOOTER = '\nif __name__ == "__main__":\n    import sys\n\n    import pytest\n\n    sys.exit(pytest.main([__file__, "-q"]))\n'
OWN_MAIN = '\nif __name__ == "__main__":\n    test_x()\n'
SCRIPT_GATE = "print('ok')\n"
PYTEST_GATE = "def test_x():\n    assert True\n"


def write_tree(tmp, gates, cmake):
    (tmp / "tests").mkdir(parents=True)
    for name, text in gates.items():
        (tmp / "tests" / name).write_text(text)
    (tmp / "tests" / "CMakeLists.txt").write_text(cmake)


def self_test():
    failures = []
    clean_cmake = """
# test-sycl-commented-out.py is only mentioned in a comment
add_test(NAME a COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-a.py)
llama_test_pytest(${Python3_EXECUTABLE}
    SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-b.py
    LABEL "sycl")
"""
    clean_gates = {"test-sycl-a.py": SCRIPT_GATE, "test-sycl-b.py": PYTEST_GATE + FOOTER}
    with tempfile.TemporaryDirectory(prefix="gate-audit-") as raw:
        base = Path(raw)

        def case(label, gates, cmake, expect, require_footer=True):
            tree = base / label
            write_tree(tree, gates, cmake)
            problems = audit(tree, require_footer=require_footer, allowlist={}, missing_allowlist={})
            if expect is None and problems:
                failures.append("%s: clean tree reported %s" % (label, problems))
            if expect is not None and not any(expect in problem for problem in problems):
                failures.append("%s: expected a problem containing %r, got %s" % (label, expect, problems))

        case("clean", clean_gates, clean_cmake, None)
        case("unregistered", dict(clean_gates, **{"test-sycl-c.py": SCRIPT_GATE}), clean_cmake,
             "R1 test-sycl-c.py")
        case("comment-only-mention", dict(clean_gates, **{"test-sycl-commented-out.py": SCRIPT_GATE}), clean_cmake,
             "R1 test-sycl-commented-out.py")
        case("pytest-via-add_test", dict(clean_gates, **{"test-sycl-d.py": PYTEST_GATE + FOOTER}),
             clean_cmake + "add_test(NAME d COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-d.py)\n",
             "R2 test-sycl-d.py")
        case("pytest-with-own-main", dict(clean_gates, **{"test-sycl-f.py": PYTEST_GATE + OWN_MAIN}),
             clean_cmake + "llama_test_pytest(${Python3_EXECUTABLE} SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-f.py)\n",
             None)
        case("pytest-without-footer", dict(clean_gates, **{"test-sycl-e.py": PYTEST_GATE}),
             clean_cmake + "llama_test_pytest(${Python3_EXECUTABLE} SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-e.py)\n",
             "R3 test-sycl-e.py")
        case("pytest-without-footer-not-required", dict(clean_gates, **{"test-sycl-e.py": PYTEST_GATE}),
             clean_cmake + "llama_test_pytest(${Python3_EXECUTABLE} SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-e.py)\n",
             None, require_footer=False)
        # A registration naming a file that is not in tests/ (R5), and the matching allowlist rules.
        tree = base / "missing-file"
        write_tree(tree, clean_gates, clean_cmake + "add_test(NAME g COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-ghost.py)\n")
        problems = audit(tree, allowlist={}, missing_allowlist={})
        if not any("R5 test-sycl-ghost.py" in p for p in problems):
            failures.append("missing-file: registration of an absent file not reported: %s" % problems)
        problems = audit(tree, allowlist={}, missing_allowlist={"test-sycl-ghost.py": "reason"})
        if problems:
            failures.append("missing-file: allowlisted ghost still reported: %s" % problems)
        problems = audit(tree, allowlist={}, missing_allowlist={"test-sycl-ghost.py": "r", "test-sycl-a.py": "r"})
        if not any("R5 test-sycl-a.py is allowlisted as missing but exists" in p for p in problems):
            failures.append("missing-file: stale missing-allowlist entry not reported: %s" % problems)
        # Stale allowlist entries are reported both ways.
        tree = base / "allowlist"
        write_tree(tree, clean_gates, clean_cmake)
        problems = audit(tree, allowlist={"test-sycl-a.py": "registered", "test-sycl-gone.py": "missing"},
                         missing_allowlist={})
        if not (any("R4 test-sycl-a.py" in p for p in problems) and any("R4 test-sycl-gone.py" in p for p in problems)):
            failures.append("allowlist: stale entries not reported: %s" % problems)

        # The same plants against a copy of the real tree, so a CMake spelling the synthetic cases do not cover
        # (bracket arguments, nested parentheses, generator expressions) cannot make the audit blind.
        root = Path(__file__).resolve().parents[1]
        real = base / "real"
        (real / "tests").mkdir(parents=True)
        for path in (root / "tests").glob("test-sycl-*.py"):
            shutil.copy(path, real / "tests" / path.name)
        for rel in ("tests/CMakeLists.txt", "ggml/src/ggml-sycl/CMakeLists.txt"):
            (real / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(root / rel, real / rel)
        baseline = audit(real)
        if baseline:
            failures.append("real-tree copy is not clean before planting: %s" % baseline)
        (real / "tests" / "test-sycl-zz-planted-unregistered.py").write_text(SCRIPT_GATE)
        planted = audit(real)
        if not any("R1 test-sycl-zz-planted-unregistered.py" in p for p in planted):
            failures.append("real-tree copy: planted unregistered gate not reported: %s" % planted)
        (real / "tests" / "test-sycl-zz-planted-unregistered.py").unlink()
        (real / "tests" / "test-sycl-zz-planted-pytest.py").write_text(PYTEST_GATE)
        with open(real / "tests" / "CMakeLists.txt", "a") as handle:
            handle.write("\nadd_test(NAME zz COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-zz-planted-pytest.py)\n")
        planted = audit(real)
        if not any("R2 test-sycl-zz-planted-pytest.py" in p for p in planted):
            failures.append("real-tree copy: planted add_test pytest module not reported: %s" % planted)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="also prove the audit fails on planted violations")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = audit(root)
    for problem in problems:
        print("FAIL " + problem)
    if args.self_test:
        failures = self_test()
        for failure in failures:
            print("SELF-TEST FAIL " + failure)
        problems = problems + failures
    if problems:
        sys.exit(1)
    gates = len(list((root / "tests").glob("test-sycl-*.py")))
    print("sycl gate registration audit: PASS (%d gates%s)" % (gates, ", self-test PASS" if args.self_test else ""))


if __name__ == "__main__":
    main()
