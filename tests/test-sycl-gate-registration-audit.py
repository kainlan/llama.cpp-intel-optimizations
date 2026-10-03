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
      `if(FALSE)` / `if(0)` / `if(NOT TRUE)` / the dead branch of `if(TRUE)` do not count, nor does a registration
      whose program is not an interpreter (`cat gate.py`, `cmake -E echo gate.py` name the file and never run it), nor
      one that a test property turns off or reinterprets (DISABLED, WILL_FAIL, PASS_REGULAR_EXPRESSION,
      SKIP_REGULAR_EXPRESSION; SKIP_RETURN_CODE, LABELS and TIMEOUT are fine);
  R2  if pytest or unittest would collect something from it (a module-level `test*` function, a `Test*` class, a
      unittest.TestCase subclass, `test_x = lambda ...`, also nested under `if`/`try`), every registration is
      llama_test_pytest. Exception: a file holding only unittest.TestCase classes whose own `__main__` calls
      unittest.main() really runs under plain python3;
  R3  such a gate can run itself, so `python3 tests/test-sycl-x.py` executes its tests or fails loudly instead of
      passing vacuously. That takes a real top-level `if __name__ == "__main__":` statement (not text in a comment,
      docstring or string, not a guard inside a function) whose block either exits with
      pytest.main([__file__, ...]) (no -k/-m/-o/-p/--pyargs/--deselect/--ignore/--co selection, status not dropped), or
      calls unittest.main() for a file of TestCase classes (no defaultTest, no argv selector, no exit=False), or
      loops over every `test_` name in globals() and then exits non-zero when the loop recorded a failure, with no
      test called by hand outside the loop. The exit (or the loop) must be a DIRECT statement of the guard body, with
      no exit before it: an exit under `if False`, behind a CLI condition, inside a `try` that swallows SystemExit,
      or after an unconditional `sys.exit(0)` does not count. The footer scripts/sycl-add-pytest-footer.py appends is
      the first form;
  R4  UNREGISTERED_ALLOWLIST holds no stale entry (a file that is registered, or no longer exists);
  R5  every test-sycl-*.py a registration names exists under tests/ (a registration of a file that was never
      committed fails at ctest time with "file not found", or worse, only passes on the one checkout that has the
      file), unless it is in MISSING_FILE_ALLOWLIST with a reason; an allowlisted name whose file now exists is
      stale;
  R6  at least MIN_GATES gates exist (a tests/ that moved would otherwise be audited as empty and pass).

  R7  (only with --census BUILD_DIR) every registered gate is in the ctest files of that configured build, so a
      registration behind a configuration guard, in an uncalled function()/macro(), in an empty foreach() or in an
      unvisited directory cannot hide. Gates a default configuration legitimately lacks are in
      CENSUS_ABSENT_ALLOWLIST with a reason;
  R8  (only with --census BUILD_DIR) every ctest registration of that build whose program is a python interpreter (or a
      .py file) carries the `python` label, as a whole member of its LABELS list, whether it runs a .py file,
      `-c <inline>` or `-m pytest <dir>`; a .py in the arguments of a non-python program does not count. `ctest -L
      python` is the python census, and a registration without the label is invisible to it (llama.cpp-gogg: 67 were,
      one of them red on master). This reads the build's ctest files, so it covers .py files that are not
      tests/test-sycl-*.py and every registrar. PYTHON_LABEL_EXEMPT lists the registrations that stay unlabelled
      because they drive device work (`-L python` must stay free of it), each with its reason; an entry is reported
      once the build no longer registers the test or the test carries `python` after all.

Known limit: without --census this reads every CMakeLists.txt under the tree statically, so it does not know whether
CMake reaches a registration. A registration in an uncalled function()/macro(), in a foreach() over an empty list,
behind a non-constant if(), or in a directory no add_subdirectory() visits still counts as registered, and a
test property reaching the test through a CMake variable (`set_tests_properties(${NAME} ...)`) is not resolved. That is
exactly how test-sycl-module-nodelete-source.py went unrun in a default build (registered inside the
GGML_BACKEND_DL block): only the census, run against a configured build, catches that class. The ctest registration of
the census covers GGML_SYCL=ON with GGML_BACKEND_DL=OFF only; other configurations are not censused. The census
requires the gate's path to resolve into the source tree's tests/ directory, and `-c` to be exactly the pytest stub
llama_test_pytest writes.

R3's selector rule: a globals() loop is "plain" only when its
selecting and skipping conditions are built from `.startswith(...)`, callable()/isinstance()/inspect.isfunction-style
calls, the loop's own names and string constants. Any comparison, non-string constant, other call (a helper that
always returns False included) or global name makes the loop not plain, and the gate is reported as unable to run
itself. R3's reading of an exit is what stays bounded. dropped_status_calls follows a main()-style status through
direct statements, `if`/`with`/`try` nesting, assignments, `rc = main(); print(rc)`, `main() and 0` and `if rc:` (which
counts only when a branch exits with a failure status). It does not follow it through a helper (`report(main())`), a
conditional expression with a constant on both sides (`0 if main() else 0`), arithmetic that zeroes it
(`main() * 0`), a walrus, asyncio.run, a method call (`G().main()`), or a name reassigned before the exit
(`rc = main(); rc = 0; sys.exit(rc)`); a gate like that needs a reviewer.

`--self-test` also proves the audit can fail: it plants each escape above (and the shapes that must stay clean) into
temp trees, synthetic and a copy of the real tree (every CMakeLists.txt this audit scans), and requires each to be
reported, and a clean synthetic tree to report nothing.
"""
import argparse
import ast
import os
import re
import shutil
import subprocess
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
# --census: gates the CMakeLists.txt files register that a configured build (GGML_SYCL=ON, GGML_BACKEND_DL=OFF, the default
# configuration) still does not contain, name -> reason. Each is a registration behind a configuration guard; an entry
# goes stale (and is reported) the moment the build does contain the gate.
CENSUS_ABSENT_ALLOWLIST = {
    "test-sycl-module-dependencies.py": "needs the GGML_BACKEND_DL module (registered inside the DL-only block of ggml-sycl)",
}
# --census, R8: ctest registrations that run a .py file and deliberately do NOT carry the `python` label, name -> reason.
# `ctest -L python` is a host-only census; these two are python drivers of device work (they load models or run a GPU
# binary), so labelling them python would put GPU work into a sweep that is otherwise safe to run. An entry goes stale
# (and is reported) when the build no longer registers the test, or the test carries `python` after all.
PYTHON_LABEL_EXEMPT = {
    "sycl-lifecycle-gpu-sequential": "loads the models its G1 fixture names (labelled `model`); `-L python` must not load models",
    "mem-handle-eviction-b70-repeat-clean-exit": "runs test-mem-handle-eviction on the B70 three times; `-L python` must not touch a device",
}
# The audit refuses a tests/ that has lost most of its gates (a moved directory would otherwise audit nothing and PASS).
MIN_GATES = 160
# The programs a registration may run a gate with: python itself, or a CMake variable that names it.
INTERPRETER = re.compile(r"(?:.*/)?python[0-9.]*|\$\{\w*python\w*\}", re.I)
# Test properties that make a registered gate not count: it never runs, its exit status is inverted, or its output
# decides the verdict instead of the exit code. SKIP_RETURN_CODE, LABELS, TIMEOUT, FAIL_REGULAR_EXPRESSION are fine.
DEFECT_PROPERTIES = {"DISABLED", "WILL_FAIL", "PASS_REGULAR_EXPRESSION", "SKIP_REGULAR_EXPRESSION"}
# SKIP_RETURN_CODE is allowed only as 77, the status the gates use for "could not run": 0 would report a pass as a skip
# and 1 a failure as one.
SKIP_RETURN_CODE_OK = "77"
# Interpreter flags that leave "python runs this file" intact. Anything else in front of the gate (-m py_compile,
# -c pass, -h, -V, ...) makes python do something other than run it.
SAFE_INTERPRETER_FLAGS = {"-B", "-u", "-s", "-E", "-I", "-O", "-OO", "-P"}
# What llama_test_pytest expands to in a configured build: `python -c <stub> gate.py`, the stub running the gate under
# pytest and returning its status (tests/CMakeLists.txt). It is the one `-c` that still runs the gate; compare
# `python -c pass gate.py`.
PYTEST_STUB = "sys.exit(pytest.main(['-q', sys.argv[1]]))"
PYTEST_STUB_TEXT = ("import importlib.util\nimport sys\nif importlib.util.find_spec('pytest') is None:\n    sys.exit(77)\n"
                    "import pytest\n" + PYTEST_STUB + "\n")
# What scripts/sycl-add-pytest-footer.py appends (two blank lines before the guard: flake8 E305).
FOOTER = '\n\nif __name__ == "__main__":\n    import sys\n\n    import pytest\n\n    sys.exit(pytest.main([__file__, "-q"]))\n'
EXIT_CALLS = {"sys.exit", "exit", "quit", "SystemExit", "os._exit"}
# pytest.main arguments that select a subset (or none) of the module's tests.
PYTEST_SELECTION_FLAGS = ("--co", "--deselect", "--ignore", "--lf", "--last-failed", "--sw", "--stepwise", "--pyargs")
# Short pytest options (matched as a prefix, since `-kexpr` is legal) that select tests out, or whose value can:
# -k/-m select, -o addopts=... injects options, -p no:python switches the collector off.
PYTEST_SELECTION_SHORT = ("-k", "-m", "-o", "-p")
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


def _is_pytest_stub(text):
    """The `-c` script llama_test_pytest writes, line for line (trailing whitespace aside): a script that merely contains
    its last line, or has a statement in front of it, is something else."""
    return [line.rstrip() for line in text.strip().splitlines()] == [line.rstrip() for line in PYTEST_STUB_TEXT.strip().splitlines()]


def command_gate_path(command_tokens):
    """The path token of the tests/test-sycl-*.py file a test command [program, args...] runs, or None.

    The program must be an interpreter (python3, ${Python3_EXECUTABLE}, ${LLAMA_PYTHON3}): `cat gate.py` or
    `cmake -E echo gate.py` name the file without running it. The gate is the first .py argument, and only the flags in
    SAFE_INTERPRETER_FLAGS may sit in front of it (`-m py_compile`, `-c pass`, `-h`, `-V` make python do something
    else). A gate that is an argument of another script, a longer file name ending in the gate's, or a .pyc is not it.
    Shared by the static audit (CMakeLists.txt) and the census (a configured build's CTestTestfile.cmake)."""
    if not command_tokens or not INTERPRETER.fullmatch(command_tokens[0]):
        return None
    rest = command_tokens[1:]
    if len(rest) > 2 and rest[0] == "-c" and _is_pytest_stub(rest[1]):
        rest = rest[2:]
    at = next((i for i, tok in enumerate(rest) if tok.endswith(".py")), None)
    if at is None or any(tok not in SAFE_INTERPRETER_FLAGS for tok in rest[:at]):
        return None
    return rest[at] if re.fullmatch(r"(?:.*/)?test-sycl-[A-Za-z0-9_.-]+\.py", rest[at]) else None


def command_gate(command_tokens):
    """The gate's file name: command_gate_path without the directory."""
    path = command_gate_path(command_tokens)
    return path.rsplit("/", 1)[-1] if path else None


def registration_target(command, args):
    """(gate file name, registered test name) for a registrar command that runs a tests/test-sycl-*.py gate, or
    (None, None). Only the COMMAND/SCRIPT/ARGS position counts (see command_gate); for add_test without a COMMAND
    keyword, the arguments after the test name."""
    tokens = cmake_tokens(args)
    keyword = COMMAND_KEYWORD[command]
    whole = tokens
    if keyword in tokens:
        tokens = tokens[tokens.index(keyword) + 1:]
    elif command != "add_test":
        return None, None
    else:
        tokens = tokens[1:]
    if command == "llama_test_pytest":
        command_tokens = [whole[0], tokens[0]] if whole and tokens else []
    elif command == "llama_test_cmd":
        command_tokens = [whole[0]] + tokens if whole else []
    else:
        command_tokens = tokens
    gate = command_gate(command_tokens)
    if gate is None:
        return None, None
    if command == "add_test" and "CONFIGURATIONS" in whole:
        return None, None  # only runs in the named build configurations
    name = None
    if command == "add_test":
        if "NAME" in whole and whole.index("NAME") + 1 < len(whole):
            name = whole[whole.index("NAME") + 1]
        elif whole:
            name = whole[0]
    elif "NAME" in whole and whole.index("NAME") + 1 < len(whole):
        name = whole[whole.index("NAME") + 1]
    elif command == "llama_test_pytest":
        name = gate.split(".")[0]  # get_filename_component(... NAME_WE): everything before the first dot
    else:
        name = whole[0] if whole else None
    return gate, name


def registered_gate(command, args):
    return registration_target(command, args)[0]


def static_condition(args):
    """True / False for a constant if() condition, None when it is not one (then both branches are live)."""
    tokens = cmake_tokens(args)
    negate = len(tokens) == 2 and tokens[0].upper() == "NOT"
    if negate:
        tokens = tokens[1:]
    if len(tokens) == 1:
        value = tokens[0].lower()
        if value in STATIC_TRUE:
            return not negate
        if value in STATIC_FALSE or value.endswith("-notfound"):
            return negate
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


def _property_defects(name, args):
    """(test names, [(property, value)]) of a live set_tests_properties()/set_property(TEST ...) command."""
    tokens = cmake_tokens(args)
    if name == "set_tests_properties" and "PROPERTIES" in tokens:
        at = tokens.index("PROPERTIES")
        names, rest = tokens[:at], tokens[at + 1:]
        return names, list(zip(rest[0::2], rest[1::2]))
    if name == "set_property" and tokens[:1] == ["TEST"] and "PROPERTY" in tokens:
        at = tokens.index("PROPERTY")
        names = [tok for tok in tokens[1:at] if tok not in ("APPEND", "APPEND_STRING")]
        return names, [(tokens[at + 1], tokens[at + 2])] if len(tokens) > at + 2 else []
    return [], []


def property_defects(tests, properties, rule):
    """Problems for registered tests (name -> gates) that a property turns off, inverts or reinterprets."""
    defects = []
    for names, pairs in properties:
        for test in names:
            for gate in sorted(tests.get(test, ())):
                for key, value in pairs:
                    key = key.upper()
                    if key in DEFECT_PROPERTIES and value.lower() not in STATIC_FALSE:
                        defects.append("%s %s is registered (as test %s) but not run as a test: %s is set on it"
                                       % (rule, gate, test, key))
                    if key == "SKIP_RETURN_CODE" and value != SKIP_RETURN_CODE_OK:
                        defects.append("%s %s is registered (as test %s) but not run as a test: SKIP_RETURN_CODE %s "
                                       "reports exit status %s as a skip (only %s is allowed)"
                                       % (rule, gate, test, value, value, SKIP_RETURN_CODE_OK))
    return defects


def scan_registrations(root):
    """(gate file name -> sorted list of the registering command names that run it,
        problems for gates whose registration a test property turns off, inverts or reinterprets)."""
    found, tests, properties = {}, {}, []
    for path in cmake_files(root):
        for name, args in live_commands(path.read_text(errors="replace")):
            if name in REGISTRARS:
                gate, test = registration_target(name, args)
                if gate:
                    found.setdefault(gate, []).append(name)
                    tests.setdefault(test, set()).add(gate)
            elif name in ("set_tests_properties", "set_property"):
                properties.append(_property_defects(name, args))
    return {gate: sorted(forms) for gate, forms in found.items()}, property_defects(tests, properties, "R1")


def registrations(root):
    """gate file name -> sorted list of the registering command names that run it."""
    return scan_registrations(root)[0]


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
            if flag.startswith(PYTEST_SELECTION_SHORT) or flag.startswith(PYTEST_SELECTION_FLAGS):
                return False
    return True


def _exit_statement(stmt):
    """The exit call when stmt is itself `sys.exit(...)` / `raise SystemExit(...)`, else None."""
    call = None
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
    elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
        call = stmt.exc
    return call if call is not None and _call_name(call) in EXIT_CALLS else None


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _recorded_names(loop):
    """Names the loop body writes: assigned, augmented, or grown with .append()/.add()/... (the failure record)."""
    out = set()
    for node in ast.walk(loop):
        if isinstance(node, ast.AugAssign):
            out |= _names(node.target)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                out |= _names(target)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("append", "add", "extend", "update", "insert", "setdefault")):
            out |= _names(node.func.value)
    return out


def _nonzero_exit(call):
    if not call.args:
        return False
    arg = call.args[0]
    return not (isinstance(arg, ast.Constant) and arg.value in (0, None))


def _loop_reports_failure(body, loop_at, recorded):
    """After the loop: a direct exit that depends on what the loop recorded, or `if <recorded>:` whose body exits
    non-zero. `sys.exit(0)`, or an exit that never reads the record, swallows every failure."""
    for stmt in body[loop_at + 1:]:
        call = _exit_statement(stmt)
        if call is not None:
            return bool(call.args) and bool(_names(call.args[0]) & recorded)
        if isinstance(stmt, ast.If) and _names(stmt.test) & recorded:
            if any(_exit_statement(inner) is not None and _nonzero_exit(_exit_statement(inner)) for inner in stmt.body):
                return True
    return False


UNITTEST_MAIN_KEYWORDS = {"argv", "verbosity", "failfast", "catchbreak", "buffer", "warnings"}


def _unittest_main_selects_nothing_out(call):
    """unittest.main() over this module: no defaultTest, exit=, module=, testLoader=, testRunner= or other keyword that
    changes what runs or whether the process exits with the result; the module (when named) is this one; no argv selector."""
    if any(kw.arg not in UNITTEST_MAIN_KEYWORDS for kw in call.keywords):
        return False
    if len(call.args) > 1:
        return False
    if call.args:
        module = call.args[0]
        if not ((isinstance(module, ast.Name) and module.id == "__name__")
                or (isinstance(module, ast.Constant) and module.value in ("__main__", None))):
            return False
    for kw in call.keywords:
        if kw.arg == "argv":
            argv = kw.value
            ok = isinstance(argv, ast.List) and argv.elts and ast.unparse(argv.elts[0]) == "sys.argv[0]" and all(
                isinstance(e, ast.Starred) for e in argv.elts[1:])
            if not ok and ast.unparse(argv) not in ("sys.argv", "sys.argv[:]"):
                return False
    return True


_PLAIN_CALLS = {"callable", "isinstance"}
_PLAIN_NAMES = {"callable", "isinstance", "type", "types", "inspect", "FunctionType", "isfunction", "isroutine"}


def _plain_test(test, loop_names):
    """A selector condition built only from `.startswith(...)`, callable()/isinstance(), and/or/not over the loop's own
    names: no comparison, no constant False, no length/endswith/membership narrowing."""
    for node in ast.walk(test):
        if isinstance(node, ast.Compare):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, str):
            return False
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if not (name.endswith(".startswith") or name in _PLAIN_CALLS or name.endswith(("isfunction", "isroutine"))):
                return False
        if isinstance(node, ast.Name) and node.id not in loop_names and node.id not in _PLAIN_NAMES:
            return False
    return True


def _selector_is_plain(loop):
    """The loop runs every test_ name: it iterates the whole globals() (no slice, no islice), never breaks out, and every
    condition that selects (`name.startswith("test...")`) or skips (an `if` around a `continue`) is a plain one."""
    if any(isinstance(sub, ast.Slice) for sub in ast.walk(loop.iter)):
        return False
    if any(isinstance(sub, ast.Call) and _call_name(sub).endswith("islice") for sub in ast.walk(loop.iter)):
        return False
    loop_names = _names(loop.target)
    nested = [n for n in ast.walk(loop) if n is not loop and isinstance(n, (ast.For, ast.While))]
    inside_nested = {id(sub) for n in nested for sub in ast.walk(n)}
    for node in ast.walk(loop):
        if isinstance(node, ast.Break) and id(node) not in inside_nested:
            return False
        if isinstance(node, ast.If):
            selects = any(isinstance(sub, ast.Call) and _call_name(sub).endswith(".startswith") for sub in ast.walk(node.test))
            skips = any(isinstance(sub, (ast.Continue, ast.Break)) for stmt in node.body + node.orelse for sub in ast.walk(stmt))
            if (selects or skips) and not _plain_test(node.test, loop_names):
                return False
    return True


def _handlers_record_or_stop(loop, recorded):
    """Every `except` in the loop either records the failure (writes a name the exit reads), re-raises, or exits:
    one that just passes swallows whatever it catches. (`except unittest.SkipTest` is a skip, not a failure.)"""
    for node in ast.walk(loop):
        if isinstance(node, ast.ExceptHandler):
            if node.type is not None and ast.unparse(node.type).endswith("SkipTest"):
                continue  # a deliberate skip is not a swallowed failure
            writes, stops = set(), False
            for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(sub, ast.Raise):
                    stops = True
                if isinstance(sub, ast.Call) and _call_name(sub) in EXIT_CALLS:
                    stops = True
            writes = _recorded_names(ast.Module(body=node.body, type_ignores=[]))
            if not (stops or writes & recorded):
                return False
    return True


def _guard_runs_every_test(guard, kinds):
    """The guard body, statement by statement: the exit has to be one of its direct statements, so nothing
    conditional, `try`-wrapped or already-exited can stand in for it."""
    body = guard.body
    calls = [node for node in ast.walk(guard) if isinstance(node, ast.Call)]
    for at, stmt in enumerate(body):
        call = _exit_statement(stmt)
        if call is not None:
            # A direct exit that is not the one we are looking for ends the block: nothing after it runs.
            return (bool(call.args) and isinstance(call.args[0], ast.Call) and _call_name(call.args[0]) == "pytest.main"
                    and _pytest_main_runs_the_file(call.args[0]))
        # unittest.main() only runs TestCase classes: next to module-level test_ functions it would run nothing of them.
        if (kinds <= {"unittest"} and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and _call_name(stmt.value) == "unittest.main" and _unittest_main_selects_nothing_out(stmt.value)):
            return True
        if isinstance(stmt, ast.For) and "globals()" in ast.unparse(stmt.iter):
            inside = {id(node) for node in ast.walk(stmt)}
            selects_tests = any(
                isinstance(node, ast.Call) and _call_name(node).endswith(".startswith") and node.args
                and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("test")
                for node in ast.walk(stmt))
            by_hand = any(isinstance(call.func, ast.Name) and call.func.id.startswith("test")
                          for call in calls if id(call) not in inside)
            recorded = _recorded_names(stmt)
            read_after = set().union(*(_names(later) for later in body[at + 1:])) if body[at + 1:] else set()
            if (selects_tests and not by_hand and _selector_is_plain(stmt) and _loop_reports_failure(body, at, recorded)
                    and _handlers_record_or_stop(stmt, recorded & read_after)):
                return True
    return False


def _returns_a_status(func):
    """True when the function body (not nested defs) returns something other than None / a constant 0."""
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value in (0, None)):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def _exits_with_failure(node):
    """An exit call with a non-constant-0 status, or a raise that is not SystemExit(0)/SystemExit()/SystemExit(None).
    `if rc: sys.exit(0)` decides on rc and still reports success, so it does not vindicate the status."""
    if isinstance(node, ast.Call):
        return _call_name(node) in EXIT_CALLS and _nonzero_exit(node)
    if isinstance(node, ast.Raise):
        exc = node.exc
        if isinstance(exc, ast.Call) and _call_name(exc) == "SystemExit":
            return _nonzero_exit(exc)
        if isinstance(exc, ast.Name) and exc.id == "SystemExit":
            return False  # a bare `raise SystemExit` exits 0
        return True
    return False


def dropped_status_calls(text):
    """Names f such that a top-level `__main__` guard (anywhere inside it: under if/with/try too) calls `f()` although f
    returns a status, and does not pass that status to an exit: as a bare statement, assigned to a name that neither an
    exit call nor the condition of an `if` that exits reads, or exit(f() and 0)-style with a constant that discards it. `main()` instead of `sys.exit(main())` turns every
    failure the function reports into exit 0. Applies to any gate, script-style included."""
    tree = parse(text)
    if tree is None:
        return []
    returning = {node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _returns_a_status(node)}
    dropped = []
    for guard in (node for node in tree.body if is_main_guard(node)):
        exit_args = [call.args[0] for call in ast.walk(guard)
                     if isinstance(call, ast.Call) and _call_name(call) in EXIT_CALLS and call.args]
        exit_names = set().union(*(_names(arg) for arg in exit_args)) if exit_args else set()
        for branch in (n for n in ast.walk(guard) if isinstance(n, ast.If)):
            if any(_exits_with_failure(c) for stmt in branch.body + branch.orelse for c in ast.walk(stmt)):
                exit_names |= _names(branch.test)  # `if rc: sys.exit(rc)` / `if missing: raise SystemExit(77)` decide on it
        for node in ast.walk(guard):
            call, name = None, None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and isinstance(node.value, ast.Call)):
                call, name = node.value, node.targets[0].id
            if call is None or not isinstance(call.func, ast.Name) or call.func.id not in returning:
                continue
            if name is None or name not in exit_names:
                dropped.append(call.func.id)
        for arg in exit_args:
            for boolop in (n for n in ast.walk(arg) if isinstance(n, ast.BoolOp)):
                calls_f = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in returning
                              for v in boolop.values for c in ast.walk(v))
                discards = any(isinstance(v, ast.Constant) and not v.value for v in boolop.values)
                if calls_f and discards:
                    dropped.append("(status discarded in an exit argument)")
    return dropped


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


def audit(root, require_footer=REQUIRE_PYTEST_FOOTER, allowlist=None, missing_allowlist=None, min_gates=0):
    root = Path(root)
    allowlist = UNREGISTERED_ALLOWLIST if allowlist is None else allowlist
    missing_allowlist = MISSING_FILE_ALLOWLIST if missing_allowlist is None else missing_allowlist
    registered, problems = scan_registrations(root)
    gates = sorted((root / "tests").glob("test-sycl-*.py"))
    if len(gates) < min_gates:
        problems.append("R6 only %d tests/test-sycl-*.py gates found under %s (floor %d): the tests directory moved, or "
                        "gates were deleted, and the audit would otherwise pass over nothing (a deliberate removal: lower "
                        "MIN_GATES, suggested floor %d)" % (len(gates), root, min_gates, suggested_min_gates(len(gates))))
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
        if require_footer:
            for func in dropped_status_calls(text):
                problems.append("R3 %s calls %s() from its __main__ guard and drops the status it returns; every failure "
                                "it reports would exit 0 (use sys.exit(%s()))" % (gate.name, func, func))
        if kinds:
            self_running = runs_itself(text)
            if set(forms) != {"llama_test_pytest"} and not (kinds <= {"unittest"} and self_running):
                problems.append("R2 %s is pytest-style but registered via %s; plain python3 runs none of its tests"
                                % (gate.name, "/".join(forms)))
            if require_footer and not self_running:
                problems.append("R3 %s is pytest-style but cannot run itself: no top-level __main__ block that runs every "
                                "test and exits with its status (run scripts/sycl-add-pytest-footer.py, or fix the "
                                "existing __main__ block)" % gate.name)
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
RUN_AND_RECORD = ["try:", "    fn()", "except AssertionError:", "    failures += 1"]
RUN_AND_RECORD_IN_IF = ["    " + line for line in RUN_AND_RECORD]


def loop_main(iterable, body, tail="sys.exit(1 if failures else 0)"):
    """A __main__ guard whose globals() loop has this body (a list of lines) and then this tail statement."""
    lines = ["", 'if __name__ == "__main__":', "    import sys", "", "    failures = 0", "    for name, fn in %s:" % iterable]
    lines += ["        " + line for line in body] + ["    " + tail, ""]
    return "\n".join(lines)


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
MAIN_FN = "import sys\n\n\ndef main():\n    return 1 if len(sys.argv) > 99 else 0\n\n"
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
    ("r3-loop-failure-branch-exits-0", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_") and callable(fn):\n'
     '            try:\n                fn()\n            except AssertionError:\n                failures += 1\n'
     '    if failures:\n        sys.exit(0)\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-failure-branch-bare-exit", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_") and callable(fn):\n'
     '            try:\n                fn()\n            except AssertionError:\n                failures += 1\n'
     '    if failures:\n        sys.exit()\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    # unittest.main(module=...) / unittest.main("othermod") runs another module's tests, not this file's.
    ("r3-unittest-main-module-keyword", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(module="othermod")\n',
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-positional-module", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main("othermod")\n',
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-ok-unittest-main-name-module", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(__name__)\n',
     REG_P_ADD, None),
    # A handler in the loop that swallows what the recording handler does not catch, and a loop narrowed to nothing.
    ("r3-loop-extra-swallow-except", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n    for name, fn in list(globals().items()):\n        if name.startswith("test_") and callable(fn):\n            try:\n                fn()\n            except AssertionError:\n                failures += 1\n            except Exception:\n                pass\n    sys.exit(1 if failures else 0)\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-name-narrowed", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n    for name, fn in list(globals().items()):\n        if name.startswith("test_") and name == "test_nothing" and callable(fn):\n            try:\n                fn()\n            except AssertionError:\n                failures += 1\n    sys.exit(1 if failures else 0)\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-narrowed-by-and-false", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n    for name, fn in list(globals().items()):\n        if name.startswith("test_") and False:\n            try:\n                fn()\n            except AssertionError:\n                failures += 1\n    sys.exit(1 if failures else 0)\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-narrowed-by-membership", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n\n    failures = 0\n    for name, fn in list(globals().items()):\n        if name.startswith("test_") and name in ("test_nothing",):\n            try:\n                fn()\n            except AssertionError:\n                failures += 1\n    sys.exit(1 if failures else 0)\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    # A script-style gate whose main() returns a status that the guard drops: `main()` instead of `sys.exit(main())`.
    ("r3-script-main-status-dropped", MAIN_FN + '\nif __name__ == "__main__":\n    main()\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-status-assigned-and-dropped", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    print("done")\n',
     REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-ok-script-main-exit", MAIN_FN + '\nif __name__ == "__main__":\n    sys.exit(main())\n', REG_P_ADD, None),
    ("r3-ok-script-main-raise-systemexit", MAIN_FN + '\nif __name__ == "__main__":\n    raise SystemExit(main())\n', REG_P_ADD, None),
    ("r3-ok-script-main-status-forwarded", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    sys.exit(rc)\n', REG_P_ADD, None),
    ("r3-ok-script-main-exits-itself", "import sys\n\n\ndef main():\n    sys.exit(1)\n\n\nif __name__ == \"__main__\":\n    main()\n",
     REG_P_ADD, None),
    ("r3-ok-script-main-returns-nothing", "def main():\n    print('ok')\n    return\n\n\nif __name__ == \"__main__\":\n    main()\n",
     REG_P_ADD, None),
    ("r3-ok-loop-skiptest-handler", PYTEST_GATE + '\nif __name__ == "__main__":\n    import sys\n    import unittest\n\n    failures = 0\n'
     '    for name, fn in list(globals().items()):\n        if name.startswith("test_") and callable(fn):\n'
     '            try:\n                fn()\n            except unittest.SkipTest:\n                print("skip")\n'
     '            except AssertionError:\n                failures += 1\n    sys.exit(1 if failures else 0)\n', REG_P_PYTEST, None),
    # More ways to run fewer tests than the loop appears to: a nested skip, a slice, an early break, a name-shape filter,
    # a custom loader, and a loop that records failures and then exits 0 anyway.
    ("r3-loop-nested-if-continue", PYTEST_GATE + loop_main("list(globals().items())", [
        'if name.startswith("test_"):', "    if len(name) > 0:", "        continue", *RUN_AND_RECORD_IN_IF]),
     REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-sliced-to-nothing", PYTEST_GATE + loop_main("list(globals().items())[:0]", [
        'if name.startswith("test_"):', *RUN_AND_RECORD_IN_IF]), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-breaks-after-the-first-test", PYTEST_GATE + loop_main("list(globals().items())", [
        'if name.startswith("test_"):', *RUN_AND_RECORD_IN_IF, "    break"]), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-endswith-narrowing", PYTEST_GATE + loop_main("list(globals().items())", [
        'if name.startswith("test_") and name.endswith("_never"):', *RUN_AND_RECORD_IN_IF]), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-records-then-exits-0", PYTEST_GATE + loop_main("list(globals().items())", [
        'if name.startswith("test_"):', *RUN_AND_RECORD_IN_IF], tail="sys.exit(0)"), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-records-then-exits-a-constant", PYTEST_GATE + loop_main("list(globals().items())", [
        'if name.startswith("test_"):', *RUN_AND_RECORD_IN_IF], tail="status = 0\n    sys.exit(status)"), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-records-prints-then-exits-0", PYTEST_GATE + loop_main("list(globals().items())", [
        'if name.startswith("test_"):', *RUN_AND_RECORD_IN_IF], tail="print(failures)\n    sys.exit(0)"), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-islice-to-nothing", PYTEST_GATE + "import itertools\n" + loop_main("itertools.islice(list(globals().items()), 0)", [
        'if name.startswith("test_"):', *RUN_AND_RECORD_IN_IF]), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-loop-selector-reads-a-flag", PYTEST_GATE + "ENABLED = False\n" + loop_main("list(globals().items())", [
        'if name.startswith("test_") and ENABLED:', *RUN_AND_RECORD_IN_IF]), REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-test-loader", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(testLoader=object())\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-test-runner", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(testRunner=object())\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-kwargs-splat", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(**{"defaultTest": "T.test_nothing"})\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-unittest-main-second-positional", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(__name__, "T.test_nothing")\n', REG_P_PYTEST, "R3 test-sycl-p.py"),
    ("r3-ok-unittest-main-verbosity", UNITTEST_GATE + '\n\nif __name__ == "__main__":\n    unittest.main(verbosity=2, failfast=True)\n', REG_P_ADD, None),
    ("r3-ok-loop-callable-and-isinstance", PYTEST_GATE + loop_main("sorted(globals().items())", [
        'if name.startswith("test_") and callable(fn) and not isinstance(fn, type):', *RUN_AND_RECORD_IN_IF]), REG_P_PYTEST, None),
    ("r3-ok-loop-continue-for-non-tests", PYTEST_GATE + loop_main("sorted(globals().items())", [
        'if not name.startswith("test_") or not callable(fn):', "    continue", *RUN_AND_RECORD]), REG_P_PYTEST, None),
    # The status a main() returns, dropped where the first version of the check did not look.
    ("r3-script-main-dropped-under-if", MAIN_FN + '\nif __name__ == "__main__":\n    if len(sys.argv) > 0:\n        main()\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-dropped-under-try", MAIN_FN + '\nif __name__ == "__main__":\n    try:\n        main()\n    finally:\n        print("done")\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-dropped-under-with", MAIN_FN + '\nif __name__ == "__main__":\n    with open(__file__) as handle:\n        main()\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-assigned-and-printed", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    print(rc)\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-and-zero", MAIN_FN + '\nif __name__ == "__main__":\n    sys.exit(main() and 0)\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-ok-script-main-nested-forwarded", MAIN_FN + '\nif __name__ == "__main__":\n    if len(sys.argv) > 0:\n        sys.exit(main())\n', REG_P_ADD, None),
    # `if rc:` only vindicates the status when the branch really exits with a failure.
    ("r3-script-main-if-rc-exits-0", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        sys.exit(0)\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-if-rc-prints-else-exits-0", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        print("failed")\n    else:\n        sys.exit(0)\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-if-rc-only-prints", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        print("failed")\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-if-rc-raises-system-exit-0", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        raise SystemExit(0)\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-main-if-rc-raises-bare-system-exit", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        raise SystemExit\n', REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-script-other-name-dropped", "import sys\n\n\ndef run():\n    return 1\n\n\nif __name__ == \"__main__\":\n    run()\n", REG_P_ADD, "R3 test-sycl-p.py"),
    ("r3-ok-script-main-if-not-rc-else-raises", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if not rc:\n        print("ok")\n    else:\n        raise SystemExit(77)\n', REG_P_ADD, None),
    ("r3-ok-script-main-if-not-rc-else-exit", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if not rc:\n        print("ok")\n    else:\n        sys.exit(rc)\n', REG_P_ADD, None),
    ("r3-ok-script-main-if-rc-raises-failure", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        raise SystemExit(77)\n', REG_P_ADD, None),
    ("r3-ok-script-main-if-rc-exits-nonzero-else-exits-0", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        sys.exit(rc)\n    else:\n        sys.exit(0)\n', REG_P_ADD, None),
    ("r3-ok-script-main-if-rc-exit", MAIN_FN + '\nif __name__ == "__main__":\n    rc = main()\n    if rc:\n        sys.exit(rc)\n', REG_P_ADD, None),
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
    # CMake property names are case-sensitive, but a lowercase spelling is not worth betting a gate on: fail closed.
    ("reg-disabled-lowercase-key", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES disabled TRUE)\n"),
    ("reg-skip-return-code-lowercase-key", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES skip_return_code 1)\n"),
    ("reg-disabled-via-set-property", "add_test(NAME o COMMAND python3 " + _M + ")\nset_property(TEST o PROPERTY DISABLED ON)\n"),
    ("reg-pytest-disabled-by-name", "llama_test_pytest(python3 NAME o SCRIPT " + _M + ")\nset_tests_properties(o PROPERTIES DISABLED 1)\n"),
    ("reg-pytest-disabled-by-default-name", "llama_test_pytest(python3 SCRIPT " + _M + ")\nset_tests_properties(test-sycl-m PROPERTIES DISABLED TRUE)\n"),
    # A SKIP_RETURN_CODE other than 77 turns a failing (1) or passing (0) exit into a skip.
    ("reg-skip-return-code-1", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES SKIP_RETURN_CODE 1)\n"),
    ("reg-skip-return-code-0", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES SKIP_RETURN_CODE 0)\n"),
    # An interpreter flag in front of the gate makes python do something other than run it.
    ("reg-interp-m-py-compile", "add_test(NAME o COMMAND python3 -m py_compile " + _M + ")\n"),
    # A program whose name only STARTS like an interpreter is a wrapper, not python (INTERPRETER must fullmatch).
    ("reg-interp-name-only-starts-like-python", "add_test(NAME o COMMAND python3-wrapper " + _M + ")\n"),
    ("reg-interp-path-only-starts-like-python", "add_test(NAME o COMMAND /usr/bin/python3.sh " + _M + ")\n"),
    ("reg-interp-c-pass", "add_test(NAME o COMMAND python3 -c pass " + _M + ")\n"),
    ("reg-interp-help", "add_test(NAME o COMMAND python3 -h " + _M + ")\n"),
    ("reg-interp-version", "add_test(NAME o COMMAND python3 -V " + _M + ")\n"),
    ("reg-cmd-interp-m", "llama_test_cmd(python3 NAME o ARGS -m py_compile " + _M + ")\n"),
    # The test only exists in one build configuration.
    ("reg-configurations", "add_test(NAME o COMMAND python3 " + _M + " CONFIGURATIONS NeverBuilt)\n"),
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
    ("reg-ok-skip-return-code-77-string", "add_test(NAME o COMMAND python3 " + _M + ")\nset_tests_properties(o PROPERTIES SKIP_RETURN_CODE \"77\")\n"),
    ("reg-ok-interpreter-flags-b-u", "add_test(NAME o COMMAND python3 -B -u " + _M + ")\n"),
    ("reg-ok-cmd-interpreter-flag", "llama_test_cmd(python3 NAME o ARGS -B " + _M + ")\n"),
    ("reg-ok-command-then-args", "add_test(NAME o COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-m.py --self-test)\n"),
]


def write_tree(tmp, gates, cmake):
    (tmp / "tests").mkdir(parents=True)
    for name, text in gates.items():
        (tmp / "tests" / name).write_text(text)
    (tmp / "tests" / "CMakeLists.txt").write_text(cmake)


def suggested_min_gates(real_count):
    """A floor inside the band: the real count rounded down to a multiple of 5 (never below 90% of it)."""
    return min(real_count, max(-(-9 * real_count // 10), real_count // 5 * 5))


def min_gates_band_problem(min_gates, real_count):
    """The R6 floor must track the real gate count: far below it and many gates could vanish silently, above it and the
    audit would fail on a healthy tree. Returns the problem text, or None when MIN_GATES is within 10% below the count."""
    if 0.9 * real_count <= min_gates <= real_count:
        return None
    return ("MIN_GATES is %d but %d gates exist: keep the floor within 10%% below the real count (suggested floor %d)"
            % (min_gates, real_count, suggested_min_gates(real_count)))


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
        # The floor: a tree with (almost) no gates must not pass, and one at the floor must.
        tree = base / "floor"
        write_tree(tree, clean_gates, clean_cmake)
        if not any("R6" in p for p in audit(tree, allowlist={}, missing_allowlist={}, min_gates=3)):
            failures.append("floor: a tree below the gate floor was not reported")
        if not any("R6" in p and "suggested floor 2" in p for p in audit(tree, allowlist={}, missing_allowlist={}, min_gates=3)):
            failures.append("floor: the R6 message does not name the suggested floor")
        if any("R6" in p for p in audit(tree, allowlist={}, missing_allowlist={}, min_gates=2)):
            failures.append("floor: a tree at the gate floor was reported")
        tree = base / "floor-moved"
        (tree / "tests").mkdir(parents=True)
        (tree / "tests" / "CMakeLists.txt").write_text(clean_cmake)
        if not any("R6" in p for p in audit(tree, allowlist={}, missing_allowlist={}, min_gates=MIN_GATES)):
            failures.append("floor-moved: an empty tests/ passed the audit")
        # The floor tracks the real count: a floor far below it would let many gates vanish silently.
        real_count = len(list((Path(__file__).resolve().parents[1] / "tests").glob("test-sycl-*.py")))
        if min_gates_band_problem(MIN_GATES, real_count):
            failures.append(min_gates_band_problem(MIN_GATES, real_count))
        message = min_gates_band_problem(100, 168) or ""
        if "168" not in message or "100" not in message or "suggested floor 165" not in message:
            failures.append("the MIN_GATES message does not name the count and the suggested floor: %r" % message)
        for floor, count, wants_problem in ((100, 168, True), (168, 168, False), (152, 168, False), (151, 168, True), (135, 150, False), (134, 150, True), (170, 168, True)):
            if bool(min_gates_band_problem(floor, count)) != wants_problem:
                failures.append("min_gates_band_problem(%d, %d) %s a problem" % (floor, count, "missed" if wants_problem else "invented"))
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
        baseline = audit(real, min_gates=MIN_GATES)
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


def configured_gates(build_dir, tests_dir):
    """(gates a configured build RUNS, problems for ones it only names).

    Every add_test of every CTestTestfile.cmake under the build is parsed: the program must be an interpreter and the gate
    the first .py argument behind safe flags (command_gate), and no DISABLED / WILL_FAIL / regex / SKIP_RETURN_CODE-not-77
    property may sit on the test, and the gate's path must resolve into `tests_dir` (a same-named file elsewhere, such as
    a stale copy, is not the gate). A substring match would count `old-gate.py`, a `.orig`, an echo of the path or a
    py_compile of it as the gate being configured."""
    tests_real = os.path.realpath(str(tests_dir))
    tests, properties, seen = {}, [], False
    for dirpath, dirnames, filenames in os.walk(build_dir):
        dirnames[:] = [d for d in dirnames if d != "CMakeFiles"]
        if "CTestTestfile.cmake" not in filenames:
            continue
        seen = True
        for name, args in cmake_commands((Path(dirpath) / "CTestTestfile.cmake").read_text(errors="replace")):
            if name == "add_test":
                tokens = cmake_tokens(args)
                path = command_gate_path(tokens[1:]) if tokens else None
                gate = None
                if path and os.path.realpath(os.path.dirname(path) or ".") == tests_real:
                    gate = os.path.basename(path)
                if gate:
                    tests.setdefault(tokens[0], set()).add(gate)
            elif name == "set_tests_properties":
                properties.append(_property_defects(name, args))
    if not seen:
        return None, []
    ran = {gate for gates in tests.values() for gate in gates}
    return ran, property_defects(tests, properties, "R7")


def census(root, build_dir, absent_allowlist=None, tests_dir=None):
    """R7: every gate the CMakeLists.txt files register (R1's static view) is in the configured build's ctest files.

    The static audit cannot tell whether CMake reaches a registration: one behind a configuration guard, in an uncalled
    function() or in an unvisited directory counts as registered there and never runs (llama.cpp-qeld found
    test-sycl-module-nodelete-source.py registered inside the GGML_BACKEND_DL block, so a default build never ran it).
    This compares against what the build actually configured, with no GPU and no ctest run."""
    root = Path(root)
    absent_allowlist = CENSUS_ABSENT_ALLOWLIST if absent_allowlist is None else absent_allowlist
    configured = configured_gates(build_dir, root / "tests" if tests_dir is None else tests_dir)
    if configured[0] is None:
        return ["R7 no CTestTestfile.cmake under %s: not a configured build, so the census proves nothing" % build_dir]
    ran, problems = configured
    problems = list(problems)
    registered, _ = scan_registrations(root)
    for gate in sorted(registered):
        present = gate in ran
        if not present and gate not in absent_allowlist:
            problems.append("R7 %s is registered in a CMakeLists.txt but is not in the configured build %s: behind a "
                            "configuration guard or in an uncalled function, so it never runs here (add it to "
                            "CENSUS_ABSENT_ALLOWLIST with the reason if that is intended)" % (gate, build_dir))
        if present and gate in absent_allowlist:
            problems.append("R7 %s is allowlisted as absent from the configured build but is there; drop the entry" % gate)
    for gate in sorted(absent_allowlist):
        if gate not in registered:
            problems.append("R7 %s is in CENSUS_ABSENT_ALLOWLIST but nothing registers it" % gate)
    return problems


def python_command(command_tokens):
    """What a test command [program, args...] runs under python, or None: the first .py argument, `-c <inline>` or
    `-m <module>` of a python interpreter, or the .py file that is itself the program. The program decides: a `.py` in
    the arguments of cmake or a test binary (`-DGATE=a.py`) is not a python test, and a python command with no .py file
    (`python -c <inline>`, `python -m pytest <dir>`) is. The .py match ignores case."""
    if not command_tokens:
        return None
    program, rest = command_tokens[0], command_tokens[1:]
    if program.lower().endswith(".py"):
        return program
    if not INTERPRETER.fullmatch(program):
        return None
    script = next((tok for tok in rest if tok.lower().endswith(".py")), None)
    if script:
        return script
    for flag, arg in zip(rest, rest[1:] + [""]):
        if flag in ("-c", "-m"):
            return "python %s %s" % (flag, arg.strip().splitlines()[0][:40] if arg.strip() else "")
    return "python"


def label_census(build_dir, exempt=None):
    """R8: every ctest registration of a configured build whose command runs a .py file carries the `python` label.

    `ctest -L python` is the census of the host-only python gates; a registration without the label is invisible to it
    (llama.cpp-gogg: 67 were, among them sycl-lifecycle-source-contract, which sat red on master while a merge claimed
    "ctest -L python 116/116"). Read from the configured build's ctest files, so every registrar counts (add_test,
    llama_test_pytest, llama_test_cmd, a function that calls one) and not only the spellings a CMakeLists.txt scan knows.
    A python command is any test whose program is a python interpreter (or is a .py file), whether it runs a .py file,
    `-c <inline>` or `-m pytest <dir>` (python_command); the label must be a whole member of the LABELS list
    (`python-extra` is not `python`), and APPENDed or set_property labels count. PYTHON_LABEL_EXEMPT names the
    registrations that drive device work and must stay out of `-L python`."""
    exempt = PYTHON_LABEL_EXEMPT if exempt is None else exempt
    scripts, labels, seen = {}, {}, False
    for dirpath, dirnames, filenames in os.walk(build_dir):
        dirnames[:] = [d for d in dirnames if d != "CMakeFiles"]
        if "CTestTestfile.cmake" not in filenames:
            continue
        seen = True
        for name, args in cmake_commands((Path(dirpath) / "CTestTestfile.cmake").read_text(errors="replace")):
            if name == "add_test":
                tokens = cmake_tokens(args)
                what = python_command(tokens[1:])
                if tokens and what:
                    scripts[tokens[0]] = what
            elif name in ("set_tests_properties", "set_property"):
                names, pairs = _property_defects(name, args)
                for test in names:
                    for key, value in pairs:
                        if key.upper() == "LABELS":
                            labels.setdefault(test, set()).update(part for part in value.split(";") if part)
    if not seen:
        return ["R8 no CTestTestfile.cmake under %s: not a configured build, so the label census proves nothing" % build_dir]
    problems = ["R8 %s runs %s but does not carry the `python` label (labels: %s): `ctest -L python` skips it, so a red "
                "there is invisible to the python census (add python to its LABELS, or to PYTHON_LABEL_EXEMPT with the "
                "reason if it drives device work)"
                % (test, script.rsplit("/", 1)[-1] if script.endswith(".py") else script, ";".join(sorted(labels.get(test, ()))) or "none")
                for test, script in sorted(scripts.items()) if "python" not in labels.get(test, ()) and test not in exempt]
    for test in sorted(exempt):
        if test not in scripts:
            problems.append("R8 %s is in PYTHON_LABEL_EXEMPT but the build registers no .py test of that name; drop the entry" % test)
        elif "python" in labels.get(test, ()):
            problems.append("R8 %s is in PYTHON_LABEL_EXEMPT but carries the `python` label; drop the entry" % test)
    return problems


def _b(text):
    return 'add_test([=[b]=] ' + text + ')\n'


CENSUS_BAD_ENTRIES = (
    ("a .pyc", _b('"/usr/bin/python3" "/x/tests/test-sycl-b.pyc"')),
    ("a .orig backup", _b('"/usr/bin/python3" "/x/tests/test-sycl-b.py.orig"')),
    ("a longer file name ending in the gate's", _b('"/usr/bin/python3" "/x/tests/old-test-sycl-b.py"')),
    ("the gate as another script's argument", _b('"/usr/bin/python3" "/x/tests/test-sycl-a.py" "/x/tests/test-sycl-b.py"')),
    ("echo", _b('"/usr/bin/cmake" "-E" "echo" "/x/tests/test-sycl-b.py"')),
    ("python -m py_compile", _b('"/usr/bin/python3" "-m" "py_compile" "/x/tests/test-sycl-b.py"')),
    ("python -c", _b('"/usr/bin/python3" "-c" "pass" "/x/tests/test-sycl-b.py"')),
    ("a DISABLED test", _b('"/usr/bin/python3" "/x/tests/test-sycl-b.py"') + 'set_tests_properties([=[b]=] PROPERTIES  DISABLED "TRUE")\n'),
    ("a WILL_FAIL test", _b('"/usr/bin/python3" "/x/tests/test-sycl-b.py"') + 'set_tests_properties([=[b]=] PROPERTIES  WILL_FAIL "TRUE")\n'),
    ("DISABLED set through a plain name for a bracketed test name", _b('"/usr/bin/python3" "/x/tests/test-sycl-b.py"') + 'set_tests_properties(b PROPERTIES  DISABLED "TRUE")\n'),
    ("SKIP_RETURN_CODE 1", _b('"/usr/bin/python3" "/x/tests/test-sycl-b.py"') + 'set_tests_properties([=[b]=] PROPERTIES  SKIP_RETURN_CODE "1")\n'),
)


def census_self_test(base):
    """The census reports an unconfigured registration, a stale allowlist entry and an empty build dir.

    The ctest files here name their gates under /x/tests, so that is the tests directory the census is told to expect
    (census(..., tests_dir=)); the real run derives it from the source tree."""
    failures = []

    def run(tree, build, allowlist):
        return census(tree, build, allowlist, tests_dir="/x/tests")

    tree = base / "census-tree"
    write_tree(tree, {"test-sycl-a.py": SCRIPT_GATE, "test-sycl-b.py": SCRIPT_GATE},
               "add_test(NAME a COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-a.py)\n"
               "if(SOME_GUARD)\nadd_test(NAME b COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-sycl-b.py)\nendif()\n")
    build = base / "census-build"
    (build / "tests").mkdir(parents=True)
    (build / "tests" / "CTestTestfile.cmake").write_text('add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n')
    problems = run(tree, build, {})
    if not any("R7 test-sycl-b.py" in p for p in problems) or any("R7 test-sycl-a.py" in p for p in problems):
        failures.append("census: a registration the build did not configure was not (only) reported: %s" % problems)
    if run(tree, build, {"test-sycl-b.py": "guarded"}):
        failures.append("census: an allowlisted absent gate was still reported")
    if not any("drop the entry" in p for p in run(tree, build, {"test-sycl-a.py": "reason", "test-sycl-b.py": "r"})):
        failures.append("census: a stale absent-allowlist entry was not reported")
    if not any("nothing registers it" in p for p in run(tree, build, {"test-sycl-b.py": "r", "test-sycl-ghost.py": "r"})):
        failures.append("census: an allowlist entry nothing registers was not reported")
    empty = base / "census-empty"
    empty.mkdir()
    if not any("not a configured build" in p for p in run(tree, empty, {})):
        failures.append("census: an empty build directory passed")
    # a gate that is only in a nested directory's ctest file is still configured
    (build / "tests" / "CTestTestfile.cmake").write_text('add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n')
    nested = build / "ggml" / "src" / "ggml-sycl"
    nested.mkdir(parents=True)
    (nested / "CTestTestfile.cmake").write_text('add_test([=[b]=] "/usr/bin/python3" "/x/tests/test-sycl-b.py")\n')
    if run(tree, build, {}):
        failures.append("census: a gate present only in a nested ctest file was reported: %s" % run(tree, build, {}))
    (nested / "CTestTestfile.cmake").unlink()
    # What the build contains must RUN the gate: each of these names the file without running it, or runs it disabled.
    for label, text in CENSUS_BAD_ENTRIES:
        (build / "tests" / "CTestTestfile.cmake").write_text(
            'add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n' + text)
        if not any("R7 test-sycl-b.py" in p for p in run(tree, build, {})):
            failures.append("census: %s counted as the gate being configured" % label)
    (build / "tests" / "CTestTestfile.cmake").write_text(
        'add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n'
        'add_test([=[b]=] "/usr/bin/python3" "-B" "/x/tests/test-sycl-b.py" "--self-test")\n'
        'set_tests_properties([=[b]=] PROPERTIES  LABELS "x" SKIP_RETURN_CODE "77" TIMEOUT "30")\n')
    if run(tree, build, {}):
        failures.append("census: a configured gate with flags, arguments and harmless properties was reported: %s"
                        % run(tree, build, {}))
    (build / "tests" / "CTestTestfile.cmake").write_text(
        'add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n'
        'add_test([=[b]=] "/usr/bin/python3" "-c" "' + PYTEST_STUB_TEXT + '" "/x/tests/test-sycl-b.py")\n')
    if run(tree, build, {}):
        failures.append("census: llama_test_pytest's `-c <pytest stub>` form was reported: %s" % run(tree, build, {}))
    # A path is not a name: the gate must be the file in the tests directory, not a same-named file elsewhere.
    for label, entry in (("a same-named file in another directory", '/x/stale/test-sycl-b.py'),
                         ("a same-named file in a subdirectory of tests", '/x/tests/old/test-sycl-b.py'),
                         ("a relative path that resolves elsewhere", '/x/tests/../stale/test-sycl-b.py')):
        (build / "tests" / "CTestTestfile.cmake").write_text(
            'add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n'
            'add_test([=[b]=] "/usr/bin/python3" "%s")\n' % entry)
        if not any("R7 test-sycl-b.py" in p for p in run(tree, build, {})):
            failures.append("census: %s counted as the gate being configured" % label)
    (build / "tests" / "CTestTestfile.cmake").write_text(
        'add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n'
        'add_test([=[b]=] "/usr/bin/python3" "/x/stale/../tests/test-sycl-b.py")\n')
    if run(tree, build, {}):
        failures.append("census: a path that normalises into the tests directory was reported: %s" % run(tree, build, {}))
    # The pytest stub must be exactly what llama_test_pytest writes: a script that merely contains its last line is not it.
    for label, stub in (("the stub preceded by a statement that exits 0", "import os\nos._exit(0)\n" + PYTEST_STUB_TEXT),
                        ("only the stub's last line", PYTEST_STUB),
                        ("the stub with its skip guard removed", PYTEST_STUB_TEXT.replace("    sys.exit(77)", "    pass"))):
        (build / "tests" / "CTestTestfile.cmake").write_text(
            'add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n'
            'add_test([=[b]=] "/usr/bin/python3" "-c" "%s" "/x/tests/test-sycl-b.py")\n' % stub)
        if not any("R7 test-sycl-b.py" in p for p in run(tree, build, {})):
            failures.append("census: `-c` with %s counted as the pytest stub" % label)
    # A gate in a directory below one that has its own CTestTestfile.cmake is still found (the walk must not stop at
    # the first directory with a ctest file).
    (build / "tests" / "CTestTestfile.cmake").write_text('add_test([=[a]=] "/usr/bin/python3" "/x/tests/test-sycl-a.py")\n')
    below = build / "tests" / "deeper" / "still"
    below.mkdir(parents=True)
    (build / "tests" / "deeper" / "CTestTestfile.cmake").write_text("# no tests of its own\n")
    (below / "CTestTestfile.cmake").write_text('add_test([=[b]=] "/usr/bin/python3" "/x/tests/test-sycl-b.py")\n')
    if run(tree, build, {}):
        failures.append("census: a gate below a directory that has its own ctest file was reported: %s" % run(tree, build, {}))
    shutil.rmtree(build / "tests" / "deeper")
    # main() must act on the census: run the script itself against a build dir that cannot pass.
    done = subprocess.run([sys.executable, os.path.abspath(__file__), "--census", str(empty)], capture_output=True, text=True,
                          timeout=120)
    if done.returncode != 1 or "R7" not in done.stdout:
        failures.append("census: `--census <empty dir>` exited %d (want 1) with output %r" % (done.returncode, done.stdout[:120]))
    return failures


def label_census_self_test(base):
    """The label census reports every shape of a .py registration without the `python` label, and nothing else."""
    failures = []
    build = base / "label-build"
    ctest_file = build / "tests" / "CTestTestfile.cmake"
    ctest_file.parent.mkdir(parents=True)

    def run(text):
        ctest_file.write_text(text)
        return label_census(build, exempt={})

    py = '"/usr/bin/python3" "/x/tests/test-sycl-b.py"'
    stub = '"/usr/bin/python3" "-c" "' + PYTEST_STUB_TEXT + '" "/x/tests/test-sycl-b.py"'
    # Each of these runs a .py file and must be named.
    for label, text in (
            ("no LABELS at all", 'add_test([=[b]=] %s)\n' % py),
            ("LABELS without python", 'add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "sycl;host")\n' % py),
            ("a label that merely contains python", 'add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "python-extra;mypython")\n' % py),
            ("llama_test_pytest's default `main` label", 'add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "main" SKIP_RETURN_CODE "77")\n' % stub),
            ("python labelled on another test only", 'add_test([=[b]=] %s)\nadd_test([=[c]=] %s)\nset_tests_properties([=[c]=] PROPERTIES  LABELS "python")\n' % (py, py.replace("sycl-b", "sycl-c"))),
            ("a script run with a flag and an argument", 'add_test([=[b]=] "/usr/bin/python3" "-B" "/x/tests/test-sycl-b.py" "--self-test")\n'),
            ("a script that is not a test-sycl gate", 'add_test([=[b]=] "/usr/bin/python3" "/x/scripts/some-audit.py")\n'),
            ("python -c with an inline script and no .py file", 'add_test([=[b]=] "/usr/bin/python3" "-c" "import sys; sys.exit(0)" "/x/bin/libx.so")\n'),
            ("python -m pytest over a directory", 'add_test([=[b]=] "/usr/bin/python3" "-m" "pytest" "-q" "/x/tests")\n'),
            ("a .PY file name in capitals", 'add_test([=[b]=] "/usr/bin/python3" "/x/tests/test-sycl-B.PY")\n'),
            ("an executable .py file run as the program itself", 'add_test([=[b]=] "/x/scripts/run-gate.py" "--self-test")\n'),
            ("an unversioned interpreter path", 'add_test([=[b]=] "/opt/py/bin/python" "-c" "pass")\n')):
        problems = run(text)
        if not any(p.startswith("R8 b ") for p in problems):
            failures.append("label census: %s was not reported: %s" % (label, problems))
    # And these must stay clean.
    for label, text in (
            ("LABELS python", 'add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "python")\n' % py),
            ("python among other labels", 'add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "sycl;python;cache" TIMEOUT "60")\n' % py),
            ("the pytest stub, python-labelled", 'add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "sycl;python" SKIP_RETURN_CODE "77")\n' % stub),
            ("python through set_property APPEND", 'add_test([=[b]=] %s)\nset_property(TEST b APPEND PROPERTY LABELS python)\n' % py),
            ("a plain name for a bracketed test name", 'add_test([=[b]=] %s)\nset_tests_properties(b PROPERTIES  LABELS "python")\n' % py),
            ("a test that runs no .py file", 'add_test([=[b]=] "/x/bin/test-mem-ops")\nadd_test([=[c]=] "/usr/bin/bash" "/x/tests/test-sycl-c.sh")\n'),
            ("a non-python program with a .py in an argument", 'add_test([=[b]=] "/usr/bin/cmake" "-DGATE=/x/tests/test-sycl-b.py" "-P" "/x/run.cmake")\n'),
            ("a non-python test binary handed a .py path", 'add_test([=[b]=] "/x/bin/test-thing" "--script" "/x/tests/test-sycl-b.py")\n'),
            ("python -c, labelled", 'add_test([=[b]=] "/usr/bin/python3" "-c" "pass")\nset_tests_properties([=[b]=] PROPERTIES  LABELS "python")\n'),
            ("python -m pytest, labelled", 'add_test([=[b]=] "/usr/bin/python3" "-m" "pytest" "/x/tests")\nset_tests_properties([=[b]=] PROPERTIES  LABELS "sycl;python")\n'),
            ("a program whose name merely starts with python", 'add_test([=[b]=] "/x/bin/python-like-tool" "-c" "x")\n')):
        problems = run(text)
        if problems:
            failures.append("label census: %s was reported: %s" % (label, problems))
    # Each unlabelled registration is named, and only those.
    problems = run('add_test([=[a]=] %s)\nset_tests_properties([=[a]=] PROPERTIES  LABELS "python")\n'
                   'add_test([=[b]=] %s)\nadd_test([=[c]=] %s)\n' % (py.replace("sycl-b", "sycl-a"), py, py.replace("sycl-b", "sycl-c")))
    if len(problems) != 2 or any(p.startswith("R8 a ") for p in problems):
        failures.append("label census: wanted exactly b and c reported, got %s" % problems)
    # A registration in a nested directory's ctest file is read as well.
    ctest_file.write_text('add_test([=[a]=] %s)\nset_tests_properties([=[a]=] PROPERTIES  LABELS "python")\n' % py.replace("sycl-b", "sycl-a"))
    nested = build / "ggml" / "src"
    nested.mkdir(parents=True)
    (nested / "CTestTestfile.cmake").write_text('add_test([=[n]=] %s)\n' % py)
    if not any(p.startswith("R8 n ") for p in label_census(build, exempt={})):
        failures.append("label census: an unlabelled registration in a nested ctest file was not reported")
    (nested / "CTestTestfile.cmake").unlink()
    # An exempt registration (device work) may stay unlabelled; a stale exemption is reported.
    ctest_file.write_text('add_test([=[b]=] %s)\nadd_test([=[n]=] %s)\n' % (py, py))
    if label_census(build, exempt={"b": "r", "n": "r"}):
        failures.append("label census: exempt registrations were reported: %s" % label_census(build, exempt={"b": "r", "n": "r"}))
    if not any(p.startswith("R8 b ") for p in label_census(build, exempt={"n": "r"})):
        failures.append("label census: a registration not in the exemption list was not reported")
    if not any("R8 ghost is in PYTHON_LABEL_EXEMPT" in p for p in label_census(build, exempt={"b": "r", "n": "r", "ghost": "r"})):
        failures.append("label census: an exemption naming no registration was not reported")
    ctest_file.write_text('add_test([=[b]=] %s)\nset_tests_properties([=[b]=] PROPERTIES  LABELS "python")\n'
                          'add_test([=[n]=] %s)\nset_tests_properties([=[n]=] PROPERTIES  LABELS "python")\n' % (py, py))
    if not any("R8 b is in PYTHON_LABEL_EXEMPT but carries" in p for p in label_census(build, exempt={"b": "r"})):
        failures.append("label census: an exemption on a python-labelled test was not reported")
    ctest_file.write_text('add_test([=[a]=] %s)\nset_tests_properties([=[a]=] PROPERTIES  LABELS "python")\n'
                          'add_test([=[n]=] %s)\n' % (py.replace("sycl-b", "sycl-a"), py))
    empty = base / "label-empty"
    empty.mkdir()
    if not any("not a configured build" in p for p in label_census(empty, exempt={})):
        failures.append("label census: an empty build directory passed")
    # main() must act on it: the script itself exits 1 on a build whose registration lacks the label.
    done = subprocess.run([sys.executable, os.path.abspath(__file__), "--census", str(build)], capture_output=True, text=True,
                          timeout=120)
    if done.returncode != 1 or "R8 n " not in done.stdout:
        failures.append("label census: `--census <build with an unlabelled .py test>` exited %d (want 1) with output %r"
                        % (done.returncode, done.stdout[-200:]))
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="also prove the audit fails on planted violations")
    parser.add_argument("--census", metavar="BUILD_DIR",
                        help="also require every registered gate to be in this configured build's CTestTestfile.cmake files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    problems = audit(root, min_gates=MIN_GATES)
    if args.census:
        problems = problems + census(root, args.census) + label_census(args.census)
    for problem in problems:
        print("FAIL " + problem)
    if args.self_test:
        failures = self_test()
        with tempfile.TemporaryDirectory(prefix="gate-census-") as raw:
            failures += census_self_test(Path(raw))
        with tempfile.TemporaryDirectory(prefix="gate-labels-") as raw:
            failures += label_census_self_test(Path(raw))
        for failure in failures:
            print("SELF-TEST FAIL " + failure)
        problems = problems + failures
    if problems:
        sys.exit(1)
    gates = len(list((root / "tests").glob("test-sycl-*.py")))
    print("sycl gate registration audit: PASS (%d gates%s%s)" % (
        gates, ", self-test PASS" if args.self_test else "", ", census PASS" if args.census else ""))


if __name__ == "__main__":
    main()
