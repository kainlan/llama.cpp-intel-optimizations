#!/usr/bin/env python3
"""Every tests/test-sycl-*.py gate is registered with ctest, and in a form that cannot pass vacuously.

A gate nobody registered never runs in any sweep, so it can go stale unnoticed (llama.cpp-qeld found seven:
six that sat green and unregistered, and a family that was red for weeks). And a pytest-style module (module-level
`test*` functions) registered with plain `python3 file.py` executes nothing at all: importing it defines functions
and exits 0. That is the vacuous-pass shape inside ctest itself, so such a module must be registered through
`llama_test_pytest`, which hands it to pytest and treats "no tests collected" as a failure.

Scope: tests/test-sycl-*.py, deliberately. The tests/test-sycl-*.sh gates are not audited here; four of them
(test-sycl-lifecycle-g1-aba.sh, test-sycl-lifecycle-mutations.sh, test-sycl-mistral-attn-correctness.sh,
test-sycl-offload-regression.sh) are named by no CMakeLists.txt on purpose: they are manual GPU runners, and GPU work
is serialised through the lead session, not run by a sweep.

Rules, per tests/test-sycl-*.py:
  R1  it is named, in COMMAND position (add_test COMMAND, llama_test_pytest SCRIPT, llama_test_cmd ARGS), by a
      registration in a CMakeLists.txt, or it is in UNREGISTERED_ALLOWLIST with a reason. Comments, an argument of
      some other script, a longer file name that ends in the gate's name, a `.pyc`, and a registration inside
      `if(FALSE)` / `if(0)` / the dead branch of `if(TRUE)` do not count;
  R2  if pytest or unittest would collect something from it (a module-level `test*` function, a `Test*` class, a
      unittest.TestCase subclass, `test_x = lambda ...`, also nested under `if`/`try`), every registration is
      llama_test_pytest. Exception: a file holding only unittest.TestCase classes whose own `__main__` calls
      unittest.main() really runs under plain python3;
  R3  such a gate can run itself, so `python3 tests/test-sycl-x.py` executes its tests or fails loudly instead of
      passing vacuously. That takes a real top-level `if __name__ == "__main__":` statement (not text in a comment,
      docstring or string, not a guard inside a function) whose block either exits with
      pytest.main([__file__, ...]) (no -k/-m/--deselect/--ignore/--co selection, status not dropped), or calls
      unittest.main() for a file of TestCase classes, or loops over every `test_` name in globals() and exits
      non-zero on failure with no test called by hand outside the loop. The footer scripts/sycl-add-pytest-footer.py
      appends is the first form;
  R4  UNREGISTERED_ALLOWLIST holds no stale entry (a file that is registered, or no longer exists);
  R5  every test-sycl-*.py a registration names exists under tests/ (a registration of a file that was never
      committed fails at ctest time with "file not found", or worse, only passes on the one checkout that has the
      file), unless it is in MISSING_FILE_ALLOWLIST with a reason; an allowlisted name whose file now exists is
      stale.

Known limit: this reads every CMakeLists.txt under the tree statically, so it does not know whether CMake reaches a
registration. A registration in an uncalled function()/macro(), behind a non-constant if(), or in a directory no
add_subdirectory() visits still counts as registered. `ctest -N` in a configured tree is the reachability check.

`--self-test` also proves the audit can fail: it plants each escape above (and the shapes that must stay clean) into
temp trees, synthetic and a copy of the real tree (every CMakeLists.txt this audit scans), and requires each to be
reported, and a clean synthetic tree to report nothing.
"""
import argparse
import ast
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REGISTRARS = ("add_test", "llama_test_pytest", "llama_test_cmd")
# The keyword after which a registrar names the file it runs; the gate is the first .py file from there on
# (llama_test_pytest: the token right after SCRIPT).
COMMAND_KEYWORD = {"add_test": "COMMAND", "llama_test_pytest": "SCRIPT", "llama_test_cmd": "ARGS"}
SKIP_DIRS = {".git", ".llm-wiki", "artifacts", "media", "models", "node_modules", "build"}
# name -> reason. A gate listed here is deliberately not in ctest; empty means every gate runs.
UNREGISTERED_ALLOWLIST = {}
# name -> reason. A registered gate whose file is not in the tree. Empty: every registered gate is committed.
# An entry goes stale (and is reported) the moment its file appears.
MISSING_FILE_ALLOWLIST = {}
REQUIRE_PYTEST_FOOTER = True
# What scripts/sycl-add-pytest-footer.py appends (two blank lines before the guard: flake8 E305).
FOOTER = '\n\nif __name__ == "__main__":\n    import sys\n\n    import pytest\n\n    sys.exit(pytest.main([__file__, "-q"]))\n'
EXIT_CALLS = {"sys.exit", "exit", "quit", "SystemExit", "os._exit"}
# pytest.main arguments that select a subset (or none) of the module's tests.
PYTEST_SELECTION_FLAGS = ("--co", "--deselect", "--ignore", "--lf", "--last-failed", "--sw", "--stepwise")
STATIC_TRUE = {"1", "on", "yes", "y", "true"}
STATIC_FALSE = {"0", "off", "no", "n", "false", "ignore", "notfound"}


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


def cmake_tokens(args):
    """The arguments of one command as a list of strings: comments dropped, quoted and bracket arguments unwrapped,
    `;`-lists split."""
    tokens, i, n = [], 0, len(args)
    while i < n:
        c = args[i]
        if c.isspace():
            i += 1
        elif c == "#":
            bracket = re.match(r"#\[(=*)\[", args[i:])
            if bracket:
                end = args.find("]" + bracket.group(1) + "]", i)
                i = n if end < 0 else end + len(bracket.group(1)) + 2
            else:
                end = args.find("\n", i)
                i = n if end < 0 else end
        elif c == '"':
            j = i + 1
            while j < n and args[j] != '"':
                j += 2 if args[j] == "\\" else 1
            tokens.append(args[i + 1:j])
            i = j + 1
        elif re.match(r"\[=*\[", args[i:]):
            bracket = re.match(r"\[(=*)\[", args[i:])
            end = args.find("]" + bracket.group(1) + "]", i)
            stop = n if end < 0 else end
            tokens.append(args[i + len(bracket.group(0)):stop])
            i = n if end < 0 else end + len(bracket.group(1)) + 2
        else:
            j = i
            while j < n and not args[j].isspace():
                j += 1
            tokens.extend(part for part in args[i:j].split(";") if part)
            i = j
    return tokens


def registered_gate(command, args):
    """The tests/test-sycl-*.py file a registrar command runs, or None.

    Only the COMMAND/SCRIPT/ARGS position counts: the first .py file from that keyword on (for add_test without a
    COMMAND keyword, the first .py file anywhere). A gate named as an argument of some other script, a longer
    file name ending in the gate's, or a .pyc does not register it."""
    tokens = cmake_tokens(args)
    keyword = COMMAND_KEYWORD[command]
    if keyword in tokens:
        tokens = tokens[tokens.index(keyword) + 1:]
    elif command != "add_test":
        return None
    if command == "llama_test_pytest":
        candidate = tokens[0] if tokens else ""
    else:
        candidate = next((tok for tok in tokens if tok.endswith(".py")), "")
    m = re.fullmatch(r"(?:.*/)?(test-sycl-[A-Za-z0-9_.-]+\.py)", candidate)
    return m.group(1) if m else None


def static_condition(args):
    """True / False for a constant if() condition, None when it is not one (then both branches are live)."""
    tokens = cmake_tokens(args)
    if len(tokens) == 1:
        value = tokens[0].lower()
        if value in STATIC_TRUE:
            return True
        if value in STATIC_FALSE or value.endswith("-notfound"):
            return False
    return None


def live_commands(text):
    """cmake_commands(text) minus the commands inside an if()/elseif()/else() branch that a constant condition
    makes dead (if(FALSE), if(0), the else() of if(TRUE))."""
    stack, out = [], []
    for name, args in cmake_commands(text):
        outer = stack[-1]["live"] if stack else True
        if name == "if":
            cond = static_condition(args)
            stack.append({"outer": outer, "taken": cond is True, "live": outer and cond is not False})
        elif name == "elseif" and stack:
            frame = stack[-1]
            cond = static_condition(args)
            frame["live"] = frame["outer"] and not frame["taken"] and cond is not False
            frame["taken"] = frame["taken"] or cond is True
        elif name == "else" and stack:
            frame = stack[-1]
            frame["live"] = frame["outer"] and not frame["taken"]
        elif name == "endif" and stack:
            stack.pop()
        elif outer:
            out.append((name, args))
    return out


def cmake_files(root):
    """Every CMakeLists.txt the audit reads under root (the one walk, so the self-test's tree copy matches it)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith("build")]
        if "CMakeLists.txt" in filenames:
            yield Path(dirpath) / "CMakeLists.txt"


def registrations(root):
    """gate file name -> sorted list of the registering command names that run it."""
    found = {}
    for path in cmake_files(root):
        for name, args in live_commands(path.read_text(errors="replace")):
            if name not in REGISTRARS:
                continue
            gate = registered_gate(name, args)
            if gate:
                found.setdefault(gate, []).append(name)
    return {gate: sorted(forms) for gate, forms in found.items()}


def parse(text):
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def is_main_guard(node):
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left] + test.comparators
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def _collect(body, kinds):
    for node in body:
        if is_main_guard(node):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                kinds.add("function")
        elif isinstance(node, ast.ClassDef):
            if any("TestCase" in ast.unparse(base) for base in node.bases):
                kinds.add("unittest")
            elif node.name.startswith("Test"):
                kinds.add("class")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if (any(isinstance(t, ast.Name) and t.id.startswith("test") for t in targets)
                    and isinstance(node.value, (ast.Lambda, ast.Name, ast.Attribute))):
                kinds.add("function")
        else:
            for field in ("body", "orelse", "finalbody"):
                _collect(getattr(node, field, None) or [], kinds)
            for handler in getattr(node, "handlers", None) or []:
                _collect(handler.body, kinds)


def collect_kinds(text):
    """What pytest/unittest would collect from the module: a subset of {"function", "class", "unittest"}."""
    tree = parse(text)
    if tree is None:
        found = re.search(r"^\s*(?:async\s+)?def\s+test|^\s*class\s+\w*Test\w*|^\s*test\w*\s*=\s*lambda", text, re.M)
        return {"function"} if found else set()
    kinds = set()
    _collect(tree.body, kinds)
    return kinds


def is_pytest_style(text):
    return bool(collect_kinds(text))


def _call_name(call):
    return ast.unparse(call.func)


def _pytest_main_runs_the_file(call):
    """pytest.main([...]) naming __file__ and selecting nothing out."""
    if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
        return False
    items = call.args[0].elts
    if not any(isinstance(item, ast.Name) and item.id == "__file__" for item in items):
        return False
    for item in items:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            flag = item.value
            if flag[:2] in ("-k", "-m") or flag.startswith(PYTEST_SELECTION_FLAGS):
                return False
    return True


def _guard_runs_every_test(guard, kinds):
    nodes = list(ast.walk(guard))
    calls = [node for node in nodes if isinstance(node, ast.Call)]
    for call in calls:
        if (_call_name(call) in EXIT_CALLS and call.args and isinstance(call.args[0], ast.Call)
                and _call_name(call.args[0]) == "pytest.main" and _pytest_main_runs_the_file(call.args[0])):
            return True
    # unittest.main() only runs TestCase classes: next to module-level test_ functions it would run nothing of them.
    if kinds <= {"unittest"}:
        for call in calls:
            if _call_name(call) == "unittest.main" and not any(kw.arg == "exit" for kw in call.keywords):
                return True
    exits = any(_call_name(call) in EXIT_CALLS for call in calls)
    for loop in (node for node in nodes if isinstance(node, ast.For) and "globals()" in ast.unparse(node.iter)):
        inside = {id(node) for node in ast.walk(loop)}
        selects_tests = any(
            isinstance(node, ast.Call) and _call_name(node).endswith(".startswith") and node.args
            and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("test")
            for node in ast.walk(loop))
        by_hand = any(isinstance(call.func, ast.Name) and call.func.id.startswith("test")
                      for call in calls if id(call) not in inside)
        if selects_tests and exits and not by_hand:
            return True
    return False


def runs_itself(text):
    """True when `python3 file.py` runs every test the module defines (or fails): see R3."""
    tree = parse(text)
    if tree is None:
        return False
    kinds = collect_kinds(text)
    return any(is_main_guard(node) and _guard_runs_every_test(node, kinds) for node in tree.body)


def has_main_guard(text):
    tree = parse(text)
    return tree is not None and any(is_main_guard(node) for node in tree.body)


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
                problems.append("R1 %s is not registered with ctest (no add_test/llama_test_pytest/llama_test_cmd runs it)"
                                % gate.name)
            continue
        if gate.name in allowlist:
            problems.append("R4 %s is allowlisted as unregistered but is registered" % gate.name)
        kinds = collect_kinds(text)
        if kinds:
            self_running = runs_itself(text)
            if set(forms) != {"llama_test_pytest"} and not (kinds <= {"unittest"} and self_running):
                problems.append("R2 %s is pytest-style but registered via %s; plain python3 runs none of its tests"
                                % (gate.name, "/".join(forms)))
            if require_footer and not self_running:
                problems.append("R3 %s is pytest-style but cannot run itself: no top-level __main__ block that runs every "
                                "test and exits with its status (run scripts/sycl-add-pytest-footer.py)" % gate.name)
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

SCRIPT_GATE = "print('ok')\n"
PYTEST_GATE = "def test_x():\n    assert True\n"
OWN_MAIN = (
    '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n'
    "    for name, fn in list(globals().items()):\n"
    '        if name.startswith("test_") and callable(fn):\n'
    "            try:\n                fn()\n            except AssertionError:\n                failures += 1\n"
    "    sys.exit(1 if failures else 0)\n"
)
HAND_LISTED = '\nif __name__ == "__main__":\n    test_x()\n    print("ok")\n'
UNITTEST_GATE = "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertTrue(True)\n"
UNITTEST_MAIN = '\n\nif __name__ == "__main__":\n    unittest.main()\n'
REG_P_PYTEST = "llama_test_pytest(${Python3_EXECUTABLE} SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-p.py)\n"
REG_P_ADD = "add_test(NAME p COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-p.py)\n"
# (label, text of test-sycl-p.py, registration, expected problem or None). Each is one way a gate could look runnable
# without being so, or a way a runnable one could be wrongly refused.
R3_CASES = [
    ("r3-main-in-comment", "# never run as __main__\n" + PYTEST_GATE, REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-main-in-docstring", '"""never run as __main__"""\n' + PYTEST_GATE, REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-guard-pass", PYTEST_GATE + '\nif __name__ == "__main__":\n    pass\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-guard-print-only", PYTEST_GATE + '\nif __name__ == "__main__":\n    print("ok")\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-hand-listed-calls", PYTEST_GATE + HAND_LISTED, REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-main-status-dropped", PYTEST_GATE + FOOTER.replace('sys.exit(pytest.main([__file__, "-q"]))', 'pytest.main([__file__, "-q"])'),
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-main-k-nomatch", PYTEST_GATE + FOOTER.replace('"-q"', '"-q", "-k", "nomatch"'), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-main-without-file", PYTEST_GATE + FOOTER.replace("[__file__, \"-q\"]", '["-q"]'), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-guard-inside-string",
     PYTEST_GATE + 'x = """\nif __name__ == "__main__":\n    sys.exit(pytest.main([__file__, "-q"]))\n"""\n',
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-guard-inside-function",
     PYTEST_GATE + '\n\ndef f():\n    if __name__ == "__main__":\n        sys.exit(pytest.main([__file__, "-q"]))\n',
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-runs-no-module-level-test", PYTEST_GATE + UNITTEST_MAIN, REG_P_PYTEST, "R3 test-sycl-p.py"),
    # The exit must be a direct statement of the guard body, with nothing exiting (or swallowing) before it.
    ("r3-exit-0-before-pytest", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n    import pytest\n    sys.exit(0)\n'
     '    sys.exit(pytest.main([__file__, "-q"]))\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-exit-under-if-false", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n    import pytest\n    if False:\n'
     '        sys.exit(pytest.main([__file__, "-q"]))\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-exit-under-argv-condition", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n    import pytest\n'
     '    if len(sys.argv) > 5:\n        sys.exit(pytest.main([__file__, "-q"]))\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-exit-in-try-swallowing-systemexit", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n    import pytest\n'
     '    try:\n        sys.exit(pytest.main([__file__, "-q"]))\n    except SystemExit:\n        pass\n', REG_P_PYTEST,
     "R3 test-sycl-p.py"),
    ("r3-loop-swallows-assertions-then-exits-0", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_"):\n            try:\n'
     '                fn()\n            except AssertionError:\n                pass\n    sys.exit(0)\n', REG_P_PYTEST,
     "R3 test-sycl-p.py"),
    ("r3-loop-never-records-a-failure", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n    n = 0\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_") and False:\n            fn()\n'
     '    sys.exit(1 if n else 0)\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-main-o-addopts", PYTEST_GATE + FOOTER.replace('"-q"', '"-o", "addopts=-k nomatch"'), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-main-p-disables-plugin", PYTEST_GATE + FOOTER.replace('"-q"', '"-p", "no:python"'), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-pytest-main-pyargs", PYTEST_GATE + FOOTER.replace('"-q"', '"--pyargs"'), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-argv-selector", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(argv=["x", "nomatch"])\n',
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-default-test", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(defaultTest="T.test_x")\n',
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-ok-loop-exits-only-on-failure", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_") and callable(fn):\n'
     '            try:\n                fn()\n            except AssertionError:\n                failures += 1\n'
     '    if failures:\n        sys.exit(1)\n    print("ok")\n', REG_P_PYTEST, None),
    ("r3-ok-loop-collects-failures-in-a-list", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failed = []\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_") and callable(fn):\n'
     '            try:\n                fn()\n            except AssertionError as exc:\n                failed.append(name)\n'
     '    sys.exit(1 if failed else 0)\n', REG_P_PYTEST, None),
    ("r3-ok-pytest-or-fallback-runner", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    try:\n        import pytest\n'
     '    except ImportError:\n        pytest = None\n    if pytest is not None:\n        sys.exit(pytest.main(["-q", __file__]))\n'
     '    failed = []\n    for name, fn in sorted(globals().items()):\n        if name.startswith("test_") and callable(fn):\n'
     '            try:\n                fn()\n            except AssertionError as exc:\n                failed.append(name)\n'
     '    sys.exit(1 if failed else 0)\n', REG_P_PYTEST, None),
    ("r3-ok-unittest-main-forwarding-cli-args", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    import sys\n\n'
     '    rest = sys.argv[1:]\n    unittest.main(argv=[sys.argv[0], *rest])\n', REG_P_ADD, None),
    ("r3-ok-footer", PYTEST_GATE + FOOTER, REG_P_PYTEST, None),
    ("r3-ok-footer-trailing-comment", PYTEST_GATE + FOOTER + "# end\n", REG_P_PYTEST, None),
    ("r3-ok-single-quoted-guard", PYTEST_GATE + FOOTER.replace('"__main__"', "'__main__'"), REG_P_PYTEST, None),
    ("r3-ok-raise-systemexit", PYTEST_GATE + FOOTER.replace("sys.exit(", "raise SystemExit("), REG_P_PYTEST, None),
    ("r3-ok-globals-loop", PYTEST_GATE + OWN_MAIN, REG_P_PYTEST, None),
]
# Shapes pytest or unittest collect that a `^def test_` match misses: all must count as pytest-style, so registering
# one with plain python3 (which collects nothing) is reported.
CLASSIFIER_CASES = [
    ("cls-test-class", "class TestX:\n    def test_x(self):\n        assert True\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-unittest-subclass", UNITTEST_GATE, REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-def-test", "def test():\n    assert True\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-def-testFoo", "def testFoo():\n    assert True\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-double-space", "def  test_x():\n    assert True\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-nested-under-if", "if True:\n    def test_x():\n        assert True\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-lambda-assignment", "test_x = lambda: None\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-async-def", "async def test_x():\n    assert True\n", REG_P_ADD, "R2 test-sycl-p.py"),
    ("cls-crlf", "def test_x():\r\n    assert True\r\n", REG_P_ADD, "R2 test-sycl-p.py"),
    # and the shapes that must NOT count: data named test_*, a def only inside a docstring, a helper called by __main__
    ("cls-ok-test-data-variable", "test_cases = [1, 2]\nprint(test_cases)\n", REG_P_ADD, None),
    ("cls-ok-def-inside-docstring", '"""\ndef test_x():\n    pass\n"""\nprint(1)\n', REG_P_ADD, None),
    ("cls-ok-unittest-with-main", UNITTEST_GATE + UNITTEST_MAIN, REG_P_ADD, None),
    ("cls-unittest-without-main-under-pytest", UNITTEST_GATE, REG_P_PYTEST, "R3 test-sycl-p.py"),
]
# Registration matching is by command position and whole file name: a gate named as an ARGUMENT of another script,
# a longer file name that merely ends in a gate's name, a .pyc, or a block that is never configured does not register it.
MATCH_CASES = [
    ("reg-argument-of-other-script", "add_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-n.py test-sycl-m.py)\n"),
    ("reg-longer-file-name", "add_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/old-test-sycl-m.py)\n"),
    ("reg-pyc", "add_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.pyc)\n"),
    ("reg-cmd-argument-of-other-script", "llama_test_cmd(python3 NAME o ARGS ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-n.py test-sycl-m.py)\n"),
    ("reg-in-if-false", "if(FALSE)\nadd_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py)\nendif()\n"),
    ("reg-in-if-0", "if(0)\nadd_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py)\nendif()\n"),
    ("reg-in-else-of-if-true", "if(TRUE)\nelse()\nadd_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py)\nendif()\n"),
]
_M = "${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py"
# A registration that exists but does not run the gate as a test: another program, or a property that disables it,
# inverts its result, or turns a pass into a skip. (SKIP_RETURN_CODE and the other properties are fine.)
MATCH_CASES += [
    ("reg-not-an-interpreter-echo", "add_test(NAME o COMMAND ${CMAKE_COMMAND} -E echo " + _M + ")\n"),
    ("reg-not-an-interpreter-cat", "add_test(NAME o COMMAND cat " + _M + ")\n"),
    ("reg-cmd-not-an-interpreter", "llama_test_cmd(bash NAME o ARGS " + _M + ")\n"),
    ("reg-disabled", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES DISABLED TRUE)\n"),
    ("reg-will-fail", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES WILL_FAIL TRUE)\n"),
    ("reg-pass-regex", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES PASS_REGULAR_EXPRESSION \"ok|FAIL\")\n"),
    ("reg-skip-regex", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES SKIP_REGULAR_EXPRESSION \".\")\n"),
    ("reg-disabled-via-set-property", "add_test(NAME o COMMAND python3 " + _M + ")\nset_property(TEST o PROPERTY DISABLED ON)\n"),
    ("reg-pytest-disabled-by-name", "llama_test_pytest(python3 NAME o SCRIPT " + _M + ")\nset_tests_properties(o PROPERTIES DISABLED 1)\n"),
    ("reg-pytest-disabled-by-default-name", "llama_test_pytest(python3 SCRIPT " + _M + ")\nset_tests_properties(test-sycl-m PROPERTIES DISABLED TRUE)\n"),
    ("reg-in-if-not-true", "if(NOT TRUE)\nadd_test(NAME o COMMAND python3 " + _M + ")\nendif()\n"),
    ("reg-in-if-not-1", "if(NOT 1)\nadd_test(NAME o COMMAND python3 " + _M + ")\nendif()\n"),
]
MATCH_OK_CASES = [
    ("reg-ok-else-of-if-false", "if(FALSE)\nelse()\nadd_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py)\nendif()\n"),
    ("reg-ok-if-variable", "if(SOME_OPTION)\nadd_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py)\nendif()\n"),
    ("reg-ok-skip-return-code", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES LABELS \"x\" SKIP_RETURN_CODE 77 TIMEOUT 30)\n"),
    ("reg-ok-disabled-false", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES DISABLED FALSE)\n"),
    ("reg-ok-other-test-disabled", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(some-other-test PROPERTIES DISABLED TRUE)\n"),
    ("reg-ok-if-not-false", "if(NOT FALSE)\nadd_test(NAME o COMMAND python3 " + _M + ")\nendif()\n"),
    ("reg-ok-python-variable", "add_test(NAME o COMMAND ${LLAMA_PYTHON3} " + _M + ")\n"),
    ("reg-ok-interpreter-flag", "add_test(NAME o COMMAND python3 -B " + _M + ")\n"),
    ("reg-ok-command-then-args", "add_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py --self-test)\n"),
]


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
        for label, text, registration, expect in R3_CASES + CLASSIFIER_CASES:
            case(label, dict(clean_gates, **{"test-sycl-p.py": text}), clean_cmake + registration, expect)
        for label, registration in MATCH_CASES:
            case(label, dict(clean_gates, **{"test-sycl-m.py": SCRIPT_GATE, "test-sycl-n.py": SCRIPT_GATE}),
                 clean_cmake + "add_test(NAME n COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-n.py)\n" + registration
                 if "test-sycl-n.py" not in registration else clean_cmake + registration,
                 "R1 test-sycl-m.py")
        case("reg-ok-dotted-gate-name", dict(clean_gates, **{"test-sycl-m.v2.py": SCRIPT_GATE}),
             clean_cmake + "add_test(NAME m COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.v2.py)\n", None)
        for label, registration in MATCH_OK_CASES:
            case(label, dict(clean_gates, **{"test-sycl-m.py": SCRIPT_GATE}), clean_cmake + registration, None)
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
        for path in cmake_files(root):
            rel = path.relative_to(root)
            (real / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, real / rel)
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
