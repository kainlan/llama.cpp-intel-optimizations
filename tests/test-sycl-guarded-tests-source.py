"""Source contract: a test that skips itself when GGML_USE_SYCL is not defined is compiled with GGML_USE_SYCL, its skip
exits 77, and every ctest registration of it reads 77 as a skip.

Many test sources open with a stub of this shape:

    #if !defined(GGML_USE_SYCL)
    int main() { fprintf(stderr, "... skipping test.\\n"); return LLAMA_TEST_EXIT_SKIP; }
    #else
    ... the real test ...
    #endif

GGML_USE_SYCL is a compile definition, not something the source can see for itself. A target receives it from ggml
(ggml_add_backend() publishes it), from ggml-private-test-registry through ggml-sycl-private-fixtures, or from its own
target_compile_definitions(). A target that gets it from none of them compiles only the stub, and if the stub returns 0
ctest reports a pass for a binary that ran nothing. Six device tests were in that state, unnoticed, because both halves
failed together. This gate checks, from the CMake and source text alone (no build directory, no binary):

  - every target that compiles a source with such a stub receives GGML_USE_SYCL, through its own PRIVATE or PUBLIC
    definitions or what it links (INTERFACE reaches only its consumers), and no CMake file undefines it (`-U`);
  - every way out of a stub `main` is `return LLAMA_TEST_EXIT_SKIP;` (an exit(), std::exit(), _Exit() or
    quick_exit() counts as a way out), a stub `main` has one, and the header defines the macro as 77;
  - no `main` holds a GGML_USE_SYCL guard of its own, outside GUARDS_IN_MAIN_ALLOWED: compiled without the define,
    such a test passes with part of its body gone;
  - every ctest registration whose command names such a target in any word (as the target, as $<TARGET_FILE:target>,
    or by a path ending in it; behind a wrapper, inside `sh -c "..."`, or as an argument) carries SKIP_RETURN_CODE 77,
    by its own properties in its own directory or by the helper that registered it, or is listed with its reason in
    REGISTRATIONS_ALLOWED.

A stub is a way out of `main` that is compiled only when GGML_USE_SYCL is not defined: in the branch of
`#if !defined(GGML_USE_SYCL)` / `#ifndef GGML_USE_SYCL` / `#if !GGML_USE_SYCL` that holds when it is undefined, or in
the `#else` / `#elif` of `#if defined(GGML_USE_SYCL)` / `#ifdef GGML_USE_SYCL` / `#if GGML_USE_SYCL`. A condition that
adds other terms (`|| !GGML_SYCL_DNNL`) still counts, because its branch is taken when GGML_USE_SYCL is undefined.

The gate reads every CMake file, every C/C++ source under a `tests` directory, and every source an executable in those
CMake files names, wherever it lives. The CMake reader follows set(), unset(), list(APPEND), cmake_parse_arguments(),
foreach() over literal items and lists, and every function() or macro() at its call sites, with its parameters and ARGN
bound; llama_build, llama_build_and_test, llama_test and llama_test_cmd are read by their own rules instead (link
libraries and skip code from their bodies; only llama_build takes a caller's LLAMA_TEST_NAME), and so is the
ggml_add_backend() definition of GGML_USE_<backend>. It ignores if() conditions: a definition or property in any branch
counts. A variable it does not compute (list(GET), string(), math(), a foreach(RANGE) variable) stays a pattern, and
the reader fails closed on it: a source pattern that could name a stub source, and a test command word that could name
a stub target, fail. A target whose name is such a pattern is matched by the same token in a test command.

Every CMake command that names a stub source (its file name, a path ending in it, or its stem in set(), list(),
foreach() or a replayed function call) must be one whose text the reader followed into a source list of a target, or
into a test command that names a target compiling that source; any other mention fails.

What the gate does NOT model:
  - a chain whose first branch tests another macro: in `#if X ... #elif defined(GGML_USE_SYCL) ... #else <stub>` the
    #else is not read as a stub, and in `#if X ... #elif !defined(GGML_USE_SYCL) <stub>` the `#if X` branch, also
    compiled without the define when X holds, is not;
  - a preprocessor directive continued onto a second line with a backslash: only its first line is read;
  - a stub in a header that a test source includes: only `main` in the source itself is read;
  - a variable that the reader follows into a test command and that also reaches a source through an expression it
    does not compute (string(CONCAT), list(GET), ...): the mention counts as attributed, the second use is not seen;
  - variables crossing files (PARENT_SCOPE, include(), add_subdirectory()): each file is replayed with its own
    variables, which is why a mention that reaches no target in its own file fails.

Each check is proven to catch its mutant below, by the failure text of that check.
"""

import functools
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_HEADER = "tests/test-skip.h"
SKIP_MACRO = "LLAMA_TEST_EXIT_SKIP"
DEFINE = "GGML_USE_SYCL"
SOURCE_SUFFIXES = (".c", ".cc", ".cpp")
ROOT_MARK = "@ROOT@"


# ---------------------------------------------------------------------------------------------------------------------
# The tree: repo-relative path -> text, for the files select_tree() names.
# ---------------------------------------------------------------------------------------------------------------------


def load_tree(root: Path) -> dict:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [d for d in dirnames if not (d.startswith(".") or d == "build" or d.startswith("build-"))]
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        files.extend("/".join(parts + [name]) for name in filenames)
    return select_tree(files, lambda rel: (root / rel).read_text(errors="replace"))


def select_tree(files, read) -> dict:
    """The files the gate reads, from a listing of repo-relative paths and a reader: every CMake file, every C/C++
    source under a `tests` directory, the skip header, and every source an executable in those CMake files names,
    wherever it lives (a stub-carrying source outside `tests` is still a test that can compile out)."""
    tree = {}
    for rel in files:
        parts = rel.split("/")
        name = parts[-1]
        is_cmake = name == "CMakeLists.txt" or name.endswith(".cmake")
        is_test_source = "tests" in parts[:-1] and name.endswith(SOURCE_SUFFIXES)
        if is_cmake or is_test_source or rel == SKIP_HEADER:
            tree[rel] = read(rel)
    listed = set(files)
    for target, entries in Model(tree).executables.items():
        for token, cm, _, _ in entries:
            got = Model.resolve(token, cm)
            if got and got not in tree and got in listed and got.endswith(SOURCE_SUFFIXES):
                tree[got] = read(got)
    return tree


# ---------------------------------------------------------------------------------------------------------------------
# CMake reading
# ---------------------------------------------------------------------------------------------------------------------

_BRACKET_OPEN = re.compile(r"\[(=*)\[")


_PARSED = {}


def cmake_commands(text: str) -> list:
    """[(name_lower, [args], line)] for every command invocation, comments dropped, quotes and brackets unwrapped.
    Memoised by text: a mutant changes one or two files and the rest parse once."""
    if text not in _PARSED:
        _PARSED[text] = _parse_cmake(text)
    return _PARSED[text]


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _comment_end(text, i):
    """The offset just past the comment that starts with the `#` at `i`: a bracket comment `#[[...]]` or a line."""
    n = len(text)
    m = _BRACKET_OPEN.match(text, i + 1)
    if m:
        end = text.find("]" + m.group(1) + "]", m.end())
        return n if end < 0 else end + len(m.group(1)) + 2
    end = text.find("\n", i)
    return n if end < 0 else end


def _parse_cmake(text: str) -> list:
    cmds = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "#":
            end = _comment_end(text, i)
            line += text.count("\n", i, end)
            i = end
            continue
        m = _IDENTIFIER.match(text, i)
        if not m:
            i += 1
            continue
        j = m.end()
        while j < n and text[j] in " \t":
            j += 1
        if j >= n or text[j] != "(":
            i = m.end()
            continue
        name, start_line = m.group(0).lower(), line
        args, depth, k, cur = [], 1, j + 1, None
        while k < n and depth:
            ch = text[k]
            if ch == "\n":
                line += 1
            if ch == '"':
                end = k + 1
                buf = []
                while end < n and text[end] != '"':
                    if text[end] == "\\" and end + 1 < n:
                        buf.append(text[end + 1])
                        end += 2
                        continue
                    buf.append(text[end])
                    end += 1
                line += text.count("\n", k + 1, end)
                cur = (cur or "") + "".join(buf)
                k = end + 1
                continue
            if ch == "[" and cur is None:
                bm = _BRACKET_OPEN.match(text, k)
                if bm:
                    end = text.find("]" + bm.group(1) + "]", bm.end())
                    end = n if end < 0 else end
                    line += text.count("\n", k, end)
                    args.append(text[bm.end():end])
                    k = end + len(bm.group(1)) + 2
                    continue
            if ch == "#":
                end = _comment_end(text, k)
                line += text.count("\n", k, end)
                k = end
                continue
            if ch in " \t\r\n" or ch in "()":
                if cur is not None:
                    args.append(cur)
                    cur = None
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                k += 1
                continue
            cur = (cur or "") + ch
            k += 1
        if cur is not None:
            args.append(cur)
        cmds.append((name, args, start_line))
        i = k
    return cmds


_VAR_REF = re.compile(r"\$\{([A-Za-z0-9_.+-]+)\}")


def substitute(token: str, env: dict) -> str:
    for _ in range(8):
        new = _VAR_REF.sub(lambda m: env[m.group(1)] if m.group(1) in env else m.group(0), token)
        if new == token:
            break
        token = new
    return token


def split_list(values) -> list:
    out = []
    for v in values:
        out.extend(x for x in v.split(";") if x != "")
    return out


def _block_end(cmds, start, open_name, close_name):
    depth = 0
    for idx in range(start, len(cmds)):
        if cmds[idx][0] == open_name:
            depth += 1
        elif cmds[idx][0] == close_name:
            depth -= 1
            if depth == 0:
                return idx
    return len(cmds)


# Helpers whose registrations Model._take reads by their own rules; every other function or macro is replayed at its
# call sites.
HELPERS = frozenset(("llama_build", "llama_build_and_test", "llama_test", "llama_test_cmd"))

# env key under which the reader keeps, for a variable, the (file, line) of every command whose text reached its value.
_ORIGIN = "\0origin:"


def parse_arguments(items, options=(), ones=(), multis=()):
    """({keyword: [values]}, unparsed) as cmake_parse_arguments() splits `items`. A keyword that is absent is absent
    from the result: cmake_parse_arguments() leaves its variable undefined."""
    values, current, unparsed = {}, None, []
    for item in items:
        if item in options:
            values[item], current = ["TRUE"], None
        elif item in ones or item in multis:
            values[item], current = [], item
        elif current is not None:
            values[current].append(item)
            if current in ones:
                current = None
        else:
            unparsed.append(item)
    return values, unparsed


def _parse_arguments(sargs, env, origins):
    """cmake_parse_arguments(<prefix> <options> <one_value> <multi_value> <args>...) into `env`."""
    if len(sargs) < 4 or sargs[0] == "PARSE_ARGV":
        return
    prefix, options, ones, multis = sargs[0], split_list([sargs[1]]), split_list([sargs[2]]), split_list([sargs[3]])
    values, unparsed = parse_arguments(split_list(sargs[4:]), options, ones, multis)
    for key in options:
        env[f"{prefix}_{key}"] = ";".join(values.get(key, ["FALSE"]))
    for key in ones + multis:
        if key in values:
            env[f"{prefix}_{key}"] = ";".join(values[key])
            env[_ORIGIN + f"{prefix}_{key}"] = origins
        else:
            env.pop(f"{prefix}_{key}", None)
    env[f"{prefix}_UNPARSED_ARGUMENTS"] = ";".join(unparsed)
    env[_ORIGIN + f"{prefix}_UNPARSED_ARGUMENTS"] = origins


def _origins(args, env) -> frozenset:
    """The (file, line) origins of every variable the raw `args` refer to."""
    found = set()
    for a in args:
        for m in _VAR_REF.finditer(a):
            found |= env.get(_ORIGIN + m.group(1), frozenset())
    return frozenset(found)


def _computed(name, sargs) -> list:
    """The variables a command other than set()/list(APPEND) assigns: the reader leaves them unset."""
    def at(i):
        return [sargs[i]] if -len(sargs) <= i < len(sargs) else []

    op = sargs[0] if sargs else ""
    if name == "list":
        if op in ("GET", "LENGTH", "FIND", "JOIN", "SUBLIST"):
            return at(-1)
        if op in ("POP_BACK", "POP_FRONT"):
            return sargs[1:]
        return at(1)
    if name == "string":
        if op == "REGEX":
            return at(3) if sargs[1:2] in (["MATCH"], ["MATCHALL"]) else at(4)
        positions = {"REPLACE": 3, "FIND": 3, "SUBSTRING": 4, "JOIN": 2, "CONCAT": 1, "APPEND": 1, "PREPEND": 1}
        return at(positions.get(op, -1))
    if name == "math" or (name == "file" and op in ("GLOB", "GLOB_RECURSE", "RELATIVE_PATH", "TO_CMAKE_PATH")):
        return at(1)
    if name in ("get_filename_component", "get_target_property", "get_property", "find_program", "find_file",
                "find_library", "find_path"):
        return at(0)
    return []


def _unset(env, var):
    env.pop(var, None)
    env.pop(_ORIGIN + var, None)


def expand(cmds, env, functions, out, path, depth=0):
    """Replay the commands of the CMake file `path` in order: set()/unset()/list(APPEND)/cmake_parse_arguments()
    update `env`, foreach() bodies are replayed per item, function()/macro() bodies are recorded in `functions` and
    replayed at each call site (the HELPERS excepted) with their parameters and ARGN bound. A foreach() whose items
    cannot be listed (RANGE, or a list the reader never saw set) is replayed once with its variable left unresolved,
    so what its body registers is still seen, as a pattern. Every other command is appended to `out` as (name,
    substituted args, line, origins): origins are the (file, line) of the command and of every command whose text
    reached its arguments through a variable."""
    idx = 0
    while idx < len(cmds):
        name, args, line = cmds[idx]
        here = frozenset({(path, line)}) | _origins(args, env)
        if name in ("function", "macro"):
            end = _block_end(cmds, idx, name, "end" + name)
            if args:
                functions[args[0].lower()] = (args[1:], cmds[idx + 1:end], path)
            idx = end + 1
            continue
        if name in functions and name not in HELPERS and depth < 8:
            params, body, def_path = functions[name]
            argv = split_list([substitute(a, env) for a in args])
            scope = dict(env)
            for k, param in enumerate(params):
                if k < len(argv):
                    scope[param], scope[_ORIGIN + param] = argv[k], here
                else:
                    _unset(scope, param)
            for var, value in (("ARGN", argv[len(params):]), ("ARGV", argv)):
                scope[var], scope[_ORIGIN + var] = ";".join(value), here
            scope["ARGC"] = str(len(argv))
            for k, a in enumerate(argv):
                scope[f"ARGV{k}"], scope[_ORIGIN + f"ARGV{k}"] = a, here
            expand(body, scope, functions, out, def_path, depth + 1)
            out.append((name, [substitute(a, env) for a in args], line, here))
            idx += 1
            continue
        if name == "foreach":
            end = _block_end(cmds, idx, "foreach", "endforeach")
            body = cmds[idx + 1:end]
            sargs = [substitute(a, env) for a in args]
            var, items = (sargs[0] if sargs else ""), []
            rest = sargs[1:]
            if rest and rest[0] in ("IN",):
                mode = None
                for a in rest[1:]:
                    if a in ("ITEMS", "LISTS", "ZIP_LISTS"):
                        mode = a
                    elif mode == "ITEMS":
                        items.extend(split_list([a]))
                    elif mode == "LISTS":
                        items.extend(split_list([env.get(a, "${" + a + "}")]))
                        here |= env.get(_ORIGIN + a, frozenset())
            elif rest and rest[0] == "RANGE":
                items = [None]
            else:
                items = split_list(rest)
            if any(item is not None and _VAR_REF.search(item) for item in items):
                items = [None]
            saved = (env.get(var), env.get(_ORIGIN + var))
            for item in items:
                if item is None:
                    _unset(env, var)  # the body is still read, with its variable left as a pattern
                else:
                    env[var], env[_ORIGIN + var] = item, here
                expand(body, env, functions, out, path, depth)
            _unset(env, var)
            if saved[0] is not None:
                env[var] = saved[0]
            if saved[1] is not None:
                env[_ORIGIN + var] = saved[1]
            idx = end + 1
            continue
        sargs = [substitute(a, env) for a in args]
        if name == "set" and sargs:
            vals = [a for a in sargs[1:] if a not in ("PARENT_SCOPE",)]
            if "CACHE" in vals:
                vals = vals[:vals.index("CACHE")]
            env[sargs[0]], env[_ORIGIN + sargs[0]] = ";".join(vals), here
        elif name == "unset" and sargs:
            _unset(env, sargs[0])
        elif name == "list" and len(sargs) >= 2 and sargs[0] == "APPEND":
            cur = env.get(sargs[1], "")
            env[sargs[1]] = ";".join([x for x in [cur] if x] + sargs[2:])
            env[_ORIGIN + sargs[1]] = env.get(_ORIGIN + sargs[1], frozenset()) | here
        elif name == "cmake_parse_arguments":
            _parse_arguments(sargs, env, here)
        else:
            # A variable a command computes (list(GET), string(), math(), ...) is not evaluated: it is unset, so what
            # uses it reads as a pattern rather than as a stale value from an earlier command.
            for var in _computed(name, sargs):
                _unset(env, var)
        if name == "llama_build" and env.get("LLAMA_TEST_NAME"):
            # Only llama_build honours a caller's LLAMA_TEST_NAME: cmake_parse_arguments undefines it in the others.
            sargs = sargs + ["NAME", env["LLAMA_TEST_NAME"]]
            here |= env.get(_ORIGIN + "LLAMA_TEST_NAME", frozenset())
        out.append((name, sargs, line, here))
        idx += 1


def scoped(args):
    """[(scope, item)] for a target_* argument list after the target name. An item before any keyword is PUBLIC (the
    plain signature of target_link_libraries is transitive)."""
    scope, out = "PUBLIC", []
    for a in args:
        if a in ("PRIVATE", "PUBLIC", "INTERFACE"):
            scope = a
            continue
        if a in ("LINK_PRIVATE", "LINK_PUBLIC", "LINK_INTERFACE_LIBRARIES"):
            scope = {"LINK_PRIVATE": "PRIVATE"}.get(a, "PUBLIC")
            continue
        for item in split_list([a]):
            out.append((scope, item))
    return out


def is_define(item: str) -> bool:
    return re.fullmatch(DEFINE + r"(=1)?", item) is not None


class Model:
    """What the CMake files say about executables, their definitions and links, and the tests that run them."""

    def __init__(self, tree: dict):
        self.executables = {}  # target -> [(source token, cmake file, line, origins)]
        self.defs = {}  # target -> [(scope, item)]
        self.links = {}  # target -> [(scope, item)]
        # (cmake directory, test name) -> (command, cmake file, line, inherent skip code or None, origins)
        self.tests = {}
        self.test_props = {}  # (cmake directory, test name) -> {prop: value}
        self.function_publishes = set()  # libraries a function body gives GGML_USE_SYCL
        self.compiling = {}  # source -> compiling_targets() result, read by both CMake checks
        self.helpers = {}  # helper name -> {"links": [...], "skip": value or None}
        cmake_files = sorted(p for p in tree if p.endswith(("CMakeLists.txt", ".cmake")))
        functions = {}
        per_file = {}
        for path in cmake_files:
            d = os.path.dirname(path)
            env = {
                "CMAKE_CURRENT_SOURCE_DIR": ROOT_MARK + ("/" + d if d else ""),
                "CMAKE_CURRENT_LIST_DIR": ROOT_MARK + ("/" + d if d else ""),
                "CMAKE_SOURCE_DIR": ROOT_MARK,
                "PROJECT_SOURCE_DIR": ROOT_MARK + ("/ggml" if path.startswith("ggml/") else ""),
            }
            out = []
            expand(cmake_commands(tree[path]), env, functions, out, path)
            per_file[path] = out
        self.functions = set(functions)  # every function() and macro() name, lower case
        self._read_functions(functions, per_file)
        for path, out in per_file.items():
            for name, args, line, origins in out:
                self._take(path, name, args, line, origins)

    # The helpers in tests/CMakeLists.txt build an executable named after the source and link what their body links.
    def _read_functions(self, functions, per_file):
        backend_calls = set()
        for out in per_file.values():
            for name, args, _, _ in out:
                if name == "ggml_add_backend" and args:
                    backend_calls.add(args[0].upper())
        for fname, (params, body, _) in functions.items():
            # Read as written: a body's set() calls depend on its arguments, so substituting them would rename the
            # very variables (`${TEST_TARGET}`, `${backend_use}`) the reading keys on.
            # ggml_add_backend(): string(TOUPPER "GGML_USE_${backend}" v) then target_compile_definitions(ggml PUBLIC
            # ${v}) gives every called backend's GGML_USE_<backend> to ggml's consumers.
            upper_vars = {}
            for name, args, _ in body:
                if name == "string" and len(args) == 3 and args[0] == "TOUPPER":
                    upper_vars[args[2]] = args[1]
            for name, args, _ in body:
                if name != "target_compile_definitions" or not args:
                    continue
                for scope, item in scoped(args[1:]):
                    if scope == "PRIVATE":
                        continue
                    m = _VAR_REF.fullmatch(item)
                    tmpl = upper_vars.get(m.group(1)) if m else None
                    if tmpl and params:
                        for backend in backend_calls:
                            if substitute(tmpl, {params[0]: backend}).upper() == DEFINE:
                                self.function_publishes.add(args[0])
                    elif is_define(item):
                        self.function_publishes.add(args[0])
            if fname in ("llama_build", "llama_build_and_test"):
                links, skip = [], None
                for name, args, _ in body:
                    if name == "target_link_libraries" and args and args[0] == "${TEST_TARGET}":
                        links.extend(item for _, item in scoped(args[1:]))
                    if name == "set_tests_properties" and "PROPERTIES" in args:
                        kv = args[args.index("PROPERTIES") + 1:]
                        for k, v in zip(kv[::2], kv[1::2]):
                            if k == "SKIP_RETURN_CODE":
                                skip = v
                self.helpers[fname] = {"links": links, "skip": skip}

    def _add_source(self, target, token, path, line, origins):
        self.executables.setdefault(target, []).append((token, path, line, origins))

    def _take(self, path, name, args, line, origins):
        d = os.path.dirname(path)  # a test's name and properties belong to the directory that registers it
        if name == "add_executable" and args:
            if any(a in ("IMPORTED", "ALIAS") for a in args[1:]):
                return
            for token in args[1:]:
                if token not in ("WIN32", "MACOSX_BUNDLE", "EXCLUDE_FROM_ALL"):
                    self._add_source(args[0], token, path, line, origins)
        elif name == "target_sources" and args:
            for _, token in scoped(args[1:]):
                self._add_source(args[0], token, path, line, origins)
        elif name in ("llama_build", "llama_build_and_test") and args:
            # llama_build(<source> [<source>...]) takes NAME only from the caller's LLAMA_TEST_NAME (added by expand);
            # llama_build_and_test(<source> [NAME n] [LABEL l] [WORKING_DIRECTORY d] [ARGS ...] [<source>...]).
            and_test = name == "llama_build_and_test"
            values, extra = parse_arguments(args[1:], (), ("NAME", "LABEL", "WORKING_DIRECTORY") if and_test else
                                            ("NAME",), ("ARGS",) if and_test else ())
            target = values.get("NAME", [re.sub(r"\.[^.]*$", "", os.path.basename(args[0]))])[0]
            for token in [args[0]] + extra:
                self._add_source(target, token, path, line, origins)
            helper = self.helpers.get(name, {"links": [], "skip": None})
            self.links.setdefault(target, []).extend(("PRIVATE", lib) for lib in helper["links"])
            if and_test:
                self.tests[(d, target)] = ((f"$<TARGET_FILE:{target}>", *values.get("ARGS", [])), path, line,
                                           helper["skip"], origins)
        elif name == "target_compile_definitions" and args:
            self.defs.setdefault(args[0], []).extend(scoped(args[1:]))
        elif name == "target_link_libraries" and args:
            self.links.setdefault(args[0], []).extend(scoped(args[1:]))
        elif name == "add_test" and args:
            if "NAME" in args and "COMMAND" in args:
                test = args[args.index("NAME") + 1]
                command = args[args.index("COMMAND") + 1:]
            else:  # the old signature: add_test(<name> <command> [<arg>...])
                test, command = args[0], args[1:]
            self.tests[(d, test)] = (tuple(command), path, line, None, origins)
        elif name in ("llama_test", "llama_test_cmd") and args:
            # llama_test runs $<TARGET_FILE:<target>>, llama_test_cmd runs <target> itself; both append ARGS.
            values, _ = parse_arguments(args[1:], (), ("NAME", "LABEL", "WORKING_DIRECTORY"), ("ARGS",))
            test = values.get("NAME", [args[0]])[0]
            exe = f"$<TARGET_FILE:{args[0]}>" if name == "llama_test" else args[0]
            self.tests[(d, test)] = ((exe, *values.get("ARGS", [])), path, line, None, origins)
        elif name == "set_tests_properties" and "PROPERTIES" in args:
            at = args.index("PROPERTIES")
            kv = args[at + 1:]
            for test in args[:at]:
                props = self.test_props.setdefault((d, test), {})
                for k, v in zip(kv[::2], kv[1::2]):
                    props[k] = v
        elif name == "set_property" and len(args) >= 4 and args[0] == "TEST" and "PROPERTY" in args:
            at = args.index("PROPERTY")
            if at + 1 < len(args):
                key, vals = args[at + 1], args[at + 2:]
                for test in args[1:at]:
                    self.test_props.setdefault((d, test), {})[key] = ";".join(vals)

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def resolve(token, cmake_path):
        """The repo-relative path a source token names, or None when a variable or generator expression is left in it
        (the token is then returned as a regex by `pattern`)."""
        if "${" in token or "$<" in token:
            return None
        if token.startswith(ROOT_MARK):
            rel = token[len(ROOT_MARK):].lstrip("/")
        else:
            rel = os.path.join(os.path.dirname(cmake_path), token)
        return os.path.normpath(rel).replace(os.sep, "/")

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def pattern(token, cmake_path):
        tok = token
        if not tok.startswith(ROOT_MARK) and "${" not in tok.split("/")[0]:
            tok = ROOT_MARK + "/" + os.path.dirname(cmake_path) + "/" + tok
        parts = re.split(r"(\$\{[^}]*\}|\$<[^>]*>)", tok)
        rx = "".join(".*" if p.startswith("$") else re.escape(p) for p in parts)
        rx = rx.replace(re.escape(ROOT_MARK) + "/", "(?:.*/)?").replace(re.escape(ROOT_MARK), "")
        return re.compile(rx)

    def publishers(self):
        pub = set(self.function_publishes)
        for lib, entries in self.defs.items():
            if any(scope != "PRIVATE" and is_define(item) for scope, item in entries):
                pub.add(lib)
        changed = True
        while changed:
            changed = False
            for lib, entries in self.links.items():
                if lib not in pub and any(scope != "PRIVATE" and item in pub for scope, item in entries):
                    pub.add(lib)
                    changed = True
        return pub

    def receives(self, target, pub):
        """A target compiles with what it gives itself PRIVATE or PUBLIC; INTERFACE reaches only its consumers."""
        if any(scope != "INTERFACE" and is_define(item) for scope, item in self.defs.get(target, [])):
            return True
        return any(scope != "INTERFACE" and item in pub for scope, item in self.links.get(target, []))


# ---------------------------------------------------------------------------------------------------------------------
# C/C++ reading
# ---------------------------------------------------------------------------------------------------------------------

_LITERAL = r'R"([^(\s]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
_LEXEME_RE = re.compile(_LITERAL + r"|//[^\n]*|/\*.*?\*/", flags=re.DOTALL)


@functools.lru_cache(maxsize=None)  # a mutant changes one or two sources; the rest are read once
def code_only(src: str) -> str:
    """The source with comments and literal text blanked to spaces, lines and offsets kept."""

    return _LEXEME_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)


# Undefined, GGML_USE_SYCL reads as 0 in an #if: `!GGML_USE_SYCL` and `GGML_USE_SYCL == 0` hold exactly when
# `!defined(GGML_USE_SYCL)` does, so the value form is a stub guard too.
_NEG = re.compile(r"!\s*\(*\s*(?:defined\s*\(?\s*)?" + DEFINE + r"\b|\b" + DEFINE + r"\s*==\s*0\b")
_POS = re.compile(r"\b" + DEFINE + r"\b")
_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def _branch_kind(directive, expr):
    """'neg': the branch holds when GGML_USE_SYCL is undefined; 'pos': it holds only when it is defined; None: no
    GGML_USE_SYCL term."""
    expr = expr.strip()
    if directive == "ifdef":
        return "pos" if re.match(DEFINE + r"\b", expr) else None
    if directive == "ifndef":
        return "neg" if re.match(DEFINE + r"\b", expr) else None
    if _NEG.search(expr):
        return "neg"
    if _POS.search(expr):
        return "pos"
    return None


def off_lines(code: str) -> set:
    """1-based numbers of the lines compiled only when GGML_USE_SYCL is undefined."""
    off, stack = set(), []  # stack of [kind_of_first_branch, current_branch_is_off]
    for n, raw in enumerate(code.split("\n"), 1):
        m = _DIRECTIVE.match(raw)
        if m:
            d, expr = m.group(1), m.group(2)
            if d in ("if", "ifdef", "ifndef"):
                kind = _branch_kind(d, expr)
                stack.append([kind, kind == "neg"])
            elif d == "elif" and stack:
                kind = _branch_kind("if", expr)
                stack[-1][1] = stack[-1][0] == "pos" or kind == "neg"
            elif d == "else" and stack:
                stack[-1][1] = stack[-1][0] == "pos"
            elif d == "endif" and stack:
                stack.pop()
            continue
        if any(frame[1] for frame in stack):
            off.add(n)
    return off


# `int main(...)`, then any specifier (`noexcept`) before the body.
_MAIN = re.compile(r"\bint\s+main\s*\([^;{)]*\)[^;{}]*\{")
# The ways out of main with a status: `return x;`, and exit(x) / std::exit(x) / _Exit(x) / quick_exit(x).
_EXIT = re.compile(r"\breturn\b([^;]*);|\b(?:std\s*::\s*)?(?:exit|_Exit|quick_exit)\s*\(([^;]*)\)\s*;")


def _mains(code):
    """[(line of `main`, body start offset, body end offset)] for every main definition in `code`."""
    found = []
    for m in _MAIN.finditer(code):
        depth, i = 1, m.end()
        while i < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            i += 1
        found.append((code.count("\n", 0, m.start()) + 1, m.end(), i))
    return found


@functools.lru_cache(maxsize=None)
def stubs(src: str) -> list:
    """[(line, verb, status expression or None)] for each stub: a return or exit call in `main` on a GGML_USE_SYCL-off
    line, and a `main` defined on such a line that has neither (None)."""
    code = code_only(src)
    off = off_lines(code)
    found = []
    for main_line, start, end in _mains(code):
        exits = []
        for r in _EXIT.finditer(code, start, end):
            line = code.count("\n", 0, r.start()) + 1
            if line in off:
                verb = "returns" if r.group(1) is not None else "exits with"
                exits.append((line, verb, " ".join((r.group(1) if r.group(1) is not None else r.group(2)).split())))
        found.extend(exits)
        if main_line in off and not exits:
            found.append((main_line, "returns", None))
    return found


@functools.lru_cache(maxsize=None)
def guards_in_main(src: str) -> list:
    """Lines of the #if / #ifdef / #ifndef / #elif directives with a GGML_USE_SYCL term inside a `main` body."""
    code = code_only(src)
    lines = code.split("\n")
    found = []
    for _, start, end in _mains(code):
        first, last = code.count("\n", 0, start) + 1, code.count("\n", 0, end) + 1
        for n in range(first, last + 1):
            m = _DIRECTIVE.match(lines[n - 1])
            if m and m.group(1) in ("if", "ifdef", "ifndef", "elif") and _branch_kind(m.group(1), m.group(2)):
                found.append(n)
    return found


# ---------------------------------------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------------------------------------


def stub_sources(tree):
    """{source: stubs} for every source in the tree with at least one stub."""
    return {p: stubs(t) for p, t in sorted(tree.items()) if p.endswith(SOURCE_SUFFIXES) and stubs(t)}


# Sources with a GGML_USE_SYCL guard inside main, each with the reason the part outside the guard is a test of its own.
GUARDS_IN_MAIN_ALLOWED = {
    "tests/test-mmq-soa.cpp":
        "the guard's #else is the skip stub, returning LLAMA_TEST_EXIT_SKIP; nothing else in main but a banner runs "
        "without the define (and no target builds the file today)",
    "tests/test-sycl-moe-q8-scratch.cpp":
        "the two scratch-sizing cases are host arithmetic and run without the define; only the pool-reset case needs "
        "the backend",
}


def _stem(path):
    return re.sub(r"\.[^.]*$", "", os.path.basename(path))


def compiling_targets(model, src):
    """(targets, unresolved, origins): the targets whose resolved sources include `src`; the unresolved source tokens
    that could name it (a token is a candidate only when it spells the file's stem: a bare `${files}` could name
    anything, and a stub source compiled that way is caught by `unattributed_mentions` instead); and the (file, line)
    of every CMake command whose text reached either."""
    if src in model.compiling:
        targets, unresolved, origins = model.compiling[src]
        return set(targets), list(unresolved), set(origins)
    stem = _stem(src)
    targets, unresolved, origins = set(), [], set()
    for target, entries in model.executables.items():
        target_unresolved = "${" in target or "$<" in target
        for tok, cm, line, came_from in entries:
            got = model.resolve(tok, cm)
            if got == src and not target_unresolved:
                targets.add(target)
                origins |= came_from
            elif (got == src or (got is None and model.pattern(tok, cm).fullmatch(src))) and stem in tok:
                unresolved.append((tok if not target_unresolved else f"{target}: {tok}", cm, line))
                origins |= came_from
    model.compiling[src] = (frozenset(targets), tuple(unresolved), frozenset(origins))
    return targets, unresolved, origins


# Commands whose arguments can carry a bare stem into a source name later (`set(n test-x)` ... `${n}.cpp`); in any
# other command a stem names a target or a test, which shares it by convention.
_STEM_CARRIERS = frozenset(("set", "list", "foreach"))


def unattributed_mentions(tree, model, src, attributed):
    """(file, line) of every CMake command that names `src` (its file name, a path ending in it, or its stem where a
    stem can become a source) but is not among the `attributed` origins of a target's sources."""
    base, stem = os.path.basename(src), _stem(src)
    carriers = _STEM_CARRIERS | (model.functions - HELPERS)
    hits = set()
    for cm in sorted(p for p in tree if p.endswith(("CMakeLists.txt", ".cmake"))):
        index = _arguments_by_last_component(tree[cm])
        for _, a, line in index.get(base, []):
            literal = "${" not in a and "$<" not in a
            if (cm, line) not in attributed and not (literal and model.resolve(a, cm) != src):
                hits.add((cm, line))
        for name, a, line in index.get(stem, []):
            if a == stem and name in carriers and (cm, line) not in attributed:
                hits.add((cm, line))
    return sorted(hits)


@functools.lru_cache(maxsize=None)
def _arguments_by_last_component(text):
    """{last path component: [(command, argument, line)]} for every argument of every command in a CMake text."""
    index = {}
    for name, args, line in cmake_commands(text):
        for a in split_list(args):
            index.setdefault(a.rsplit("/", 1)[-1], []).append((name, a, line))
    return index


# `-UGGML_USE_SYCL` (or `/U`), as one argument or two, in a compile option, flag variable or source property.
_UNDEFINE = re.compile(r"(?:^|[\s;])[-/]U\s*" + DEFINE + r"\b")


def define_failures(tree, model=None):
    model = model or Model(tree)
    pub = model.publishers()
    failures = []
    for cm in sorted(p for p in tree if p.endswith(("CMakeLists.txt", ".cmake"))):
        for _, args, line in cmake_commands(tree[cm]):
            if _UNDEFINE.search(" ".join(args)):
                failures.append(f"{cm}:{line}: undefines GGML_USE_SYCL: what it applies to compiles its stubs")
    for src in stub_sources(tree):
        targets, unresolved, attributed = compiling_targets(model, src)
        for tok, cm, line in unresolved:
            failures.append(f"{cm}:{line}: cannot tell whether `{tok}` compiles {src}, which carries a GGML_USE_SYCL "
                            "stub; name its sources literally")
        hits = unattributed_mentions(tree, model, src, attributed)
        if hits:
            # A stem on its way to a test command that names a target compiling `src` is that target's name, followed
            # to where it is used: attributed too.
            for command, _, _, _, came_from in model.tests.values():
                words = command_words(command)
                if any(names_target(w, t) for w in words for t in targets):
                    attributed |= came_from
            hits = [h for h in hits if h not in attributed]
        for cm, line in hits:
            failures.append(f"{cm}:{line}: names {src}, which carries a GGML_USE_SYCL stub, in a form the gate "
                            "cannot attribute to a target")
        for target in sorted(targets):
            if not model.receives(target, pub):
                failures.append(f"target {target} compiles {src}, which carries a GGML_USE_SYCL stub, and does not "
                                "receive GGML_USE_SYCL: its test body compiles out")
    return failures


def stub_failures(tree):
    failures = []
    header = tree.get(SKIP_HEADER, "")
    if not re.search(r"^\s*#\s*define\s+" + SKIP_MACRO + r"\s+77\s*$", header, flags=re.MULTILINE):
        failures.append(f"{SKIP_HEADER} does not define {SKIP_MACRO} as 77")
    for src, found in stub_sources(tree).items():
        for line, verb, expr in found:
            if expr is None:
                failures.append(f"{src}:{line}: the GGML_USE_SYCL-off stub main has no return, so it exits 0")
            elif expr != SKIP_MACRO:
                failures.append(f"{src}:{line}: the GGML_USE_SYCL-off stub {verb} `{expr}`, not {SKIP_MACRO}")
    for src, text in sorted(tree.items()):
        if src.endswith(SOURCE_SUFFIXES) and src not in GUARDS_IN_MAIN_ALLOWED:
            for line in guards_in_main(text):
                failures.append(f"{src}:{line}: a GGML_USE_SYCL guard inside main: compiled without the define, the "
                                "test passes with part of its body gone; guard the whole of main, or list the file "
                                "and its reason in GUARDS_IN_MAIN_ALLOWED")
    return failures


# Registrations that name a stub target without SKIP_RETURN_CODE 77, keyed by (CMake directory, test name), each with
# the reason the test's exit code is not the stub's.
REGISTRATIONS_ALLOWED = {
    ("ggml/src/ggml-sycl", "sycl-private-carrier-linkage"):
        "the linkage audit reads each fixture binary with ldd and nm and never runs it",
}


def command_words(command) -> list:
    """The words of a test command: its arguments split at list separators and at whitespace, so a binary inside
    `sh -c "..."` or a `--flag=<path>` is a word of its own or ends one."""
    return [w for a in command for w in re.split(r"[;\s]+", a) if w]


def names_target(word, target):
    """True when `word` names `target`'s binary: by name, by $<TARGET_FILE:...>, or by a path ending in it."""
    return (word == target or word.endswith("/" + target) or f"$<TARGET_FILE:{target}>" in word or
            f"$<TARGET_FILE_NAME:{target}>" in word)


_TOOL_VAR = re.compile(r"\$\{(CMAKE_[A-Za-z0-9_]*|[A-Za-z0-9_]*(?:EXECUTABLE|COMMAND|PROGRAM))\}")


def could_name_target(word, target, first):
    """True when `word` keeps a variable the reader could not resolve and, read as a pattern, could name `target`.
    A tool variable on its own (`${Python3_EXECUTABLE}`, `${CMAKE_COMMAND}`) names an interpreter or tool, and a
    bare variable after the first word is an argument value: neither is a candidate. `$<TARGET_FILE:${t}>` always is."""
    m = re.search(r"\$<TARGET_FILE(?:_NAME)?:(.*)>", word)
    name = m.group(1) if m else word
    if not re.search(r"\$\{|\$<", name):
        return False
    if not m and (_TOOL_VAR.fullmatch(name) or (not first and re.fullmatch(r"\$\{[^}]*\}", name))):
        return False
    last = name.rsplit("/", 1)[-1]
    rx = "".join(".*" if p.startswith("$") else re.escape(p) for p in re.split(r"(\$\{[^}]*\}|\$<[^>]*>)", last))
    return re.fullmatch(rx, target) is not None


def registration_failures(tree, model=None):
    model = model or Model(tree)
    stub_targets = {}
    for src in stub_sources(tree):
        for target in compiling_targets(model, src)[0]:
            stub_targets.setdefault(target, src)
        # An executable whose name keeps a variable (a target built in a loop the reader could not list) is matched
        # by the same token in a test command.
        for target, entries in model.executables.items():
            if _VAR_REF.search(target) and any(model.resolve(tok, cm) == src for tok, cm, _, _ in entries):
                stub_targets.setdefault(target, src)
    pattern_targets = [t for t in model.executables if _VAR_REF.search(t)]
    failures = []
    # Fail closed: wherever a stub binary appears in a command (run directly, behind a wrapper, through a shell, or
    # handed to another program), the registration reads 77 as a skip unless REGISTRATIONS_ALLOWED says why not.
    for (d, test), (command, cm, line, inherent, _) in sorted(model.tests.items()):
        skip = model.test_props.get((d, test), {}).get("SKIP_RETURN_CODE", inherent)
        if skip == "77" or (d, test) in REGISTRATIONS_ALLOWED:
            continue
        words = command_words(command)
        # A word with a variable left in it that names a target built under that same pattern is that target (a stub
        # one is in stub_targets), not a pattern that could name any stub target.
        open_words = [(k, w) for k, w in enumerate(words) if ("${" in w or "$<" in w) and
                      not any(names_target(w, t) for t in pattern_targets)]
        for target in sorted(stub_targets):
            if any(names_target(w, target) for w in words):
                failures.append(f"{cm}:{line}: test {test} names {target} (GGML_USE_SYCL stub in "
                                f"{stub_targets[target]}) without SKIP_RETURN_CODE 77")
                continue
            for k, w in open_words:
                if could_name_target(w, target, k == 0):
                    failures.append(f"{cm}:{line}: cannot tell whether test {test} (word `{w}`) names {target} "
                                    f"(GGML_USE_SYCL stub in {stub_targets[target]}); name the binary literally")
                    break
    return failures


def all_failures(tree):
    model = Model(tree)
    return stub_failures(tree) + define_failures(tree, model) + registration_failures(tree, model)


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------

_TREE = None


def tree():
    global _TREE
    if _TREE is None:
        _TREE = load_tree(ROOT)
    return _TREE


def test_the_reader_sees_the_stubs_and_their_targets():
    t = tree()
    found = stub_sources(t)
    assert "tests/test-unified-cache-bugs.cpp" in found, "the stub reader no longer sees a known stub"
    model = Model(t)
    targets, _, _ = compiling_targets(model, "tests/test-unified-cache-bugs.cpp")
    assert targets == {"test-unified-cache-bugs"}, targets
    targets, _, _ = compiling_targets(model, "tests/test-sycl-zone-reset-live-refusal.cpp")
    assert targets == {"test-sycl-zone-reset-live-refusal"}, targets  # through llama_build()
    assert "ggml" in model.publishers() and "ggml-sycl-private-fixtures" in model.publishers()
    assert model.tests, "the reader found no ctest registration"


def test_every_stub_exits_77():
    failures = stub_failures(tree())
    assert not failures, "\n".join(failures)


def test_every_target_compiling_a_stub_receives_the_define():
    failures = define_failures(tree())
    assert not failures, "\n".join(failures)


def test_every_registration_of_a_stub_reads_77_as_a_skip():
    failures = registration_failures(tree())
    assert not failures, "\n".join(failures)


UCB = "tests/test-unified-cache-bugs.cpp"
SYCL_CMAKE = "ggml/src/ggml-sycl/CMakeLists.txt"
GGML_CMAKE = "ggml/src/CMakeLists.txt"
TESTS_CMAKE = "tests/CMakeLists.txt"


def _edit(path, old, new, count=1):
    def apply(t):
        assert old in t[path], f"mutant anchor missing from {path}: {old!r}"
        t = dict(t)
        t[path] = t[path].replace(old, new, count)
        return t

    return apply


def _sub(path, pattern, repl):
    def apply(t):
        new, n = re.subn(pattern, repl, t[path], count=1)
        assert n == 1, f"mutant pattern missing from {path}: {pattern!r}"
        t = dict(t)
        t[path] = new
        return t

    return apply


def _add(path, text):
    def apply(t):
        t = dict(t)
        t[path] = t.get(path, "") + text
        return t

    return apply


def _both(*edits):
    def apply(t):
        for e in edits:
            t = e(t)
        return t

    return apply


def _relist(files):
    """Put `files` on the listing the tree is selected from, and select again: what the gate reads of them is what
    load_tree would read had they been on disk."""
    def apply(t):
        disk = dict(t, **files)
        return select_tree(sorted(disk), disk.__getitem__)
    return apply


SYNTHETIC_ELSE_STUB = """
#include "test-skip.h"
int main() {
#ifdef GGML_USE_SYCL
    return run_everything();
#else
    return 0;
#endif
}
"""

SYNTHETIC_MAIN_GUARD = """
#include "test-skip.h"
int main() {
    int failed = 0;
#ifdef GGML_USE_SYCL
    failed += run_sycl_cases();
#endif
    return failed;
}
"""

MUTANTS = {
    # The registry keeps the define to itself: the fixture executables lose it.
    "registry-define-private": (
        _edit(GGML_CMAKE, "target_compile_definitions(ggml-private-test-registry PUBLIC",
              "target_compile_definitions(ggml-private-test-registry PRIVATE"),
        "target test-unified-cache-bugs compiles tests/test-unified-cache-bugs.cpp, which carries a GGML_USE_SYCL "
        "stub, and does not receive GGML_USE_SYCL"),
    # ggml keeps GGML_USE_<backend> to itself: every test that links ggml loses it.
    "ggml-backend-define-private": (
        _edit(GGML_CMAKE, "target_compile_definitions(ggml PUBLIC ${backend_use})",
              "target_compile_definitions(ggml PRIVATE ${backend_use})"),
        "target test-sycl-zone-reset-live-refusal compiles tests/test-sycl-zone-reset-live-refusal.cpp"),
    # A new target compiles a stub source and links neither ggml nor the fixtures; it is reached through foreach().
    "foreach-target-without-define": (
        _add(SYCL_CMAKE, "\nforeach(_t IN ITEMS test-ucb-copy)\n"
             "    add_executable(${_t} ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-cache-bugs.cpp)\n"
             "    target_link_libraries(${_t} PRIVATE ggml-base)\nendforeach()\n"),
        "target test-ucb-copy compiles tests/test-unified-cache-bugs.cpp"),
    # A source path the reader cannot resolve could name a stub source.
    "unresolved-source-path": (
        _add(SYCL_CMAKE, "\nadd_executable(test-ucb-unknown ${SOME_UNSET_DIR}/test-unified-cache-bugs.cpp)\n"),
        "cannot tell whether `${SOME_UNSET_DIR}/test-unified-cache-bugs.cpp` compiles tests/test-unified-cache-bugs.cpp"),
    # The stub exits 0 again.
    "stub-returns-0": (
        _edit(UCB, f"return {SKIP_MACRO};", "return 0;"),
        (f"{UCB}:", f"the GGML_USE_SYCL-off stub returns `0`, not {SKIP_MACRO}")),
    # The stub falls off the end of main.
    "stub-without-return": (
        _edit(UCB, f"return {SKIP_MACRO};", ";"),
        "stub main has no return, so it exits 0"),
    # The #ifndef spelling of the same stub.
    "ifndef-stub-returns-0": (
        _both(_edit(UCB, "#if !defined(GGML_USE_SYCL)", "#ifndef GGML_USE_SYCL"),
              _edit(UCB, f"return {SKIP_MACRO};", "return 0;")),
        "returns `0`, not LLAMA_TEST_EXIT_SKIP"),
    # A stub in the #else of #ifdef GGML_USE_SYCL, inside main.
    "else-branch-stub-in-main": (
        _add("tests/test-synthetic-else-stub.cpp", SYNTHETIC_ELSE_STUB),
        "tests/test-synthetic-else-stub.cpp:7: the GGML_USE_SYCL-off stub returns `0`"),
    # The skip macro stops meaning 77.
    "skip-macro-not-77": (
        _edit(SKIP_HEADER, f"#define {SKIP_MACRO} 77", f"#define {SKIP_MACRO} 0"),
        f"does not define {SKIP_MACRO} as 77"),
    # The registration loses its skip code.
    "registration-without-skip-code": (
        _sub(SYCL_CMAKE, r"(set_tests_properties\(unified-cache-bugs PROPERTIES[^)]*?)\n    SKIP_RETURN_CODE 77", r"\1"),
        "test unified-cache-bugs names test-unified-cache-bugs"),
    # A registration through llama_test() (which sets no skip code of its own) loses its skip code.
    "llama-test-registration-without-skip-code": (
        _sub(TESTS_CMAKE,
             r"(set_tests_properties\(test-sycl-zone-reset-live-refusal PROPERTIES[^)]*?)\n *SKIP_RETURN_CODE 77", r"\1"),
        "test test-sycl-zone-reset-live-refusal names test-sycl-zone-reset-live-refusal"),
    # A second registration names the binary through $<TARGET_FILE:...> and sets no skip code.
    "target-file-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-again COMMAND $<TARGET_FILE:test-unified-cache-bugs> --again)\n"),
        "test ucb-again names test-unified-cache-bugs"),
    # A stub registered through llama_build_and_test(), whose body no longer sets the skip code.
    "helper-without-skip-code": (
        _both(_edit(TESTS_CMAKE, "        LABELS \"${LLAMA_TEST_LABEL}\"\n        SKIP_RETURN_CODE 77)\nendfunction()",
                    "        LABELS \"${LLAMA_TEST_LABEL}\")\nendfunction()"),
              _add(TESTS_CMAKE, "\nllama_build_and_test(test-sycl-pointer-types.cpp)\n")),
        "test test-sycl-pointer-types names test-sycl-pointer-types"),
    # The value spelling of the guard: undefined, GGML_USE_SYCL reads as 0, so `#if !GGML_USE_SYCL` is a stub too.
    "value-form-stub-returns-0": (
        _both(_edit(UCB, "#if !defined(GGML_USE_SYCL)", "#if !GGML_USE_SYCL"),
              _edit(UCB, f"return {SKIP_MACRO};", "return 0;")),
        (f"{UCB}:", f"the GGML_USE_SYCL-off stub returns `0`, not {SKIP_MACRO}")),
    # The value spelling of the positive guard, with the stub in its #else.
    "value-form-else-stub": (
        _add("tests/test-synthetic-else-stub.cpp", SYNTHETIC_ELSE_STUB.replace("#ifdef GGML_USE_SYCL",
                                                                              "#if GGML_USE_SYCL")),
        "tests/test-synthetic-else-stub.cpp:7: the GGML_USE_SYCL-off stub returns `0`"),
    # A wrapper in front of the binary passes its exit code through: the registration still runs the stub.
    "timeout-wrapped-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-timeout COMMAND timeout -k 5 60 $<TARGET_FILE:test-unified-cache-bugs>)\n"),
        "test ucb-timeout names test-unified-cache-bugs"),
    # The binary named by its path in the build tree.
    "binary-path-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-path COMMAND ${CMAKE_BINARY_DIR}/bin/test-unified-cache-bugs)\n"),
        "test ucb-path names test-unified-cache-bugs"),
    # A registration inside a function the gate has no rule for: its call sites are replayed.
    "function-registration-without-skip-code": (
        _add(SYCL_CMAKE, "\nfunction(ucb_again_test suffix)\n"
             "    add_test(NAME ucb-${suffix} COMMAND test-unified-cache-bugs ${ARGN})\nendfunction()\n"
             "ucb_again_test(fn --again)\n"),
        "test ucb-fn names test-unified-cache-bugs"),
    # A function that builds a stub source into a target of its own, without the define.
    "function-target-without-define": (
        _add(SYCL_CMAKE, "\nfunction(ucb_copy name)\n"
             "    cmake_parse_arguments(UC \"\" \"\" \"LIBS\" ${ARGN})\n"
             "    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-cache-bugs.cpp)\n"
             "    target_link_libraries(${name} PRIVATE ${UC_LIBS})\nendfunction()\n"
             "ucb_copy(test-ucb-fn-copy LIBS ggml-base)\n"),
        "target test-ucb-fn-copy compiles tests/test-unified-cache-bugs.cpp"),
    # A loop the reader cannot list (RANGE): its registration is read as a pattern that could run the stub binary.
    "unlisted-loop-registration": (
        _add(SYCL_CMAKE, "\nforeach(_i RANGE 1 3)\n"
             "    add_test(NAME ucb-r${_i} COMMAND test-unified-cache-${_i})\nendforeach()\n"),
        ("cannot tell whether test ucb-r${_i}", "names test-unified-cache-bugs")),
    # A stub source outside any `tests` directory, compiled by a target that does not receive the define.
    "stub-source-outside-tests": (
        _relist({"tools/stubbed/stubbed.cpp": SYNTHETIC_ELSE_STUB,
                 "tools/stubbed/CMakeLists.txt": "add_executable(stubbed-tool stubbed.cpp)\n"}),
        "target stubbed-tool compiles tools/stubbed/stubbed.cpp, which carries a GGML_USE_SYCL stub"),
    # A second target compiles a stub source through a variable set for the parent scope, which the reader, replaying
    # each file on its own, never sees reach a source list: the set() is a mention it cannot attribute.
    "parent-scope-second-target": (
        _both(_add(TESTS_CMAKE, "\nset(UCB_COPY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test-unified-cache-bugs.cpp "
                   "PARENT_SCOPE)\n"),
              _add("CMakeLists.txt", "\nadd_executable(test-ucb-parent ${UCB_COPY_SOURCE})\n")),
        (f"{TESTS_CMAKE}:", f"names {UCB}", "cannot attribute to a target")),
    # A stub's stem in a variable that reaches neither a test command nor an attributed source.
    "stem-reaching-no-test": (
        _add(SYCL_CMAKE, "\nset(_ucb_names test-unified-cache-bugs)\n"
             "target_link_libraries(${_ucb_names} PRIVATE ggml)\n"),
        (f"{SYCL_CMAKE}:", f"names {UCB}", "cannot attribute to a target")),
    # The stub exits 0 through exit() before its return.
    "stub-exit-call": (
        _edit(UCB, f"    return {SKIP_MACRO};", f"    std::exit(EXIT_SUCCESS);\n    return {SKIP_MACRO};"),
        (f"{UCB}:", f"the GGML_USE_SYCL-off stub exits with `EXIT_SUCCESS`, not {SKIP_MACRO}")),
    # A specifier between the parameter list and the body still makes a main.
    "noexcept-main-stub-returns-0": (
        _both(_edit(UCB, "#if !defined(GGML_USE_SYCL)\nint main() {",
                    "#if !defined(GGML_USE_SYCL)\nint main() noexcept {"),
              _edit(UCB, f"return {SKIP_MACRO};", "return 0;")),
        (f"{UCB}:", f"the GGML_USE_SYCL-off stub returns `0`, not {SKIP_MACRO}")),
    # Part of main compiled out, the rest passing.
    "guard-inside-main": (
        _add("tests/test-synthetic-main-guard.cpp", SYNTHETIC_MAIN_GUARD),
        "tests/test-synthetic-main-guard.cpp:5: a GGML_USE_SYCL guard inside main"),
    # A target that gives itself the define INTERFACE only: its consumers compile with it, it does not.
    "interface-define-on-target": (
        _add(SYCL_CMAKE, "\nadd_executable(test-ucb-iface\n"
             "    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/test-unified-cache-bugs.cpp)\n"
             "target_compile_definitions(test-ucb-iface INTERFACE GGML_USE_SYCL)\n"),
        "target test-ucb-iface compiles tests/test-unified-cache-bugs.cpp"),
    # A compile option undefines the define the target receives.
    "undefine-option": (
        _add(SYCL_CMAKE, "\ntarget_compile_options(test-unified-cache-bugs PRIVATE -UGGML_USE_SYCL)\n"),
        (f"{SYCL_CMAKE}:", "undefines GGML_USE_SYCL")),
    # The binary behind `cmake -E chdir`, a wrapper the gate has no rule for.
    "chdir-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-chdir COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR}\n"
             "    $<TARGET_FILE:test-unified-cache-bugs>)\n"),
        "test ucb-chdir names test-unified-cache-bugs"),
    # The binary inside a shell command string.
    "shell-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-sh COMMAND sh -c \"$<TARGET_FILE:test-unified-cache-bugs>\")\n"),
        "test ucb-sh names test-unified-cache-bugs"),
    # llama_build() names its target after the caller's LLAMA_TEST_NAME, and the test registers that name.
    "llama-test-name-registration": (
        _add(TESTS_CMAKE, "\nset(LLAMA_TEST_NAME test-ucb-named)\nllama_build(test-unified-cache-bugs.cpp)\n"
             "llama_test(test-ucb-named)\nunset(LLAMA_TEST_NAME)\n"),
        "test test-ucb-named names test-ucb-named"),
    # A stub source passed to llama_build() after the first source.
    "llama-build-extra-source": (
        _add(TESTS_CMAKE, "\nllama_build(test-ucb-extra-main.cpp test-unified-cache-bugs.cpp)\n"
             "add_test(NAME ucb-extra COMMAND test-ucb-extra-main)\n"),
        "test ucb-extra names test-ucb-extra-main"),
    # A test of the same name in another directory, where the skip code set for the first does not reach it.
    "same-test-name-two-directories": (
        _add("CMakeLists.txt", "\nadd_test(NAME unified-cache-bugs COMMAND $<TARGET_FILE:test-unified-cache-bugs>)\n"),
        "test unified-cache-bugs names test-unified-cache-bugs"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_each_check_catches_its_mutant(name):
    edit, fragment = MUTANTS[name]
    base = tree()
    mutant = edit(base)
    assert mutant != base, f"the mutant '{name}' did not change the tree"
    lines = all_failures(mutant)
    parts = fragment if isinstance(fragment, tuple) else (fragment,)  # every part on one failure line
    failures = "\n".join(lines)
    assert any(all(p in line for p in parts) for line in lines), (
        f"the mutant '{name}' was not caught by its check; failures:\n{failures}")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
