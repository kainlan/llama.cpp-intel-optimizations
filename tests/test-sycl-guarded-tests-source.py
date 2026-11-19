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

  - every target that compiles a source with such a stub receives GGML_USE_SYCL;
  - every return in a stub is `return LLAMA_TEST_EXIT_SKIP;`, a stub `main` has one, and the header defines the macro
    as 77;
  - every ctest registration that runs such a target carries SKIP_RETURN_CODE 77, by its own properties or by the
    helper function that registered it. The program a command runs is its first word after any pass-through wrapper
    (`timeout`, `env`, `${CMAKE_COMMAND} -E env`, ...), named as the target, as $<TARGET_FILE:target>, or by a path
    ending in it. A command that hands the binary to another program as an argument, such as the private carrier
    linkage audit, has that program's exit code, not the stub's.

A stub is a `return` inside `main` that is compiled only when GGML_USE_SYCL is not defined: in the branch of
`#if !defined(GGML_USE_SYCL)` / `#ifndef GGML_USE_SYCL` / `#if !GGML_USE_SYCL` that holds when it is undefined, or in
the `#else` / `#elif` of `#if defined(GGML_USE_SYCL)` / `#ifdef GGML_USE_SYCL` / `#if GGML_USE_SYCL`. A condition that
adds other terms (`|| !GGML_SYCL_DNNL`) still counts, because its branch is taken when GGML_USE_SYCL is undefined.

The gate reads every CMake file, every C/C++ source under a `tests` directory, and every source an executable in those
CMake files names, wherever it lives. The CMake reader follows set(), list(APPEND), cmake_parse_arguments(), foreach()
over literal items and lists, and every function() or macro() at its call sites, with its parameters and ARGN bound;
the `llama_*` test helpers are read by their own rules instead (link libraries and skip code from their bodies), and so
is the ggml_add_backend() definition of GGML_USE_<backend>. It ignores if() conditions: a definition or property in any
branch counts. A variable it does not compute (list(GET), string(), math(), a foreach(RANGE) variable) stays a pattern,
and the reader fails closed on it: a source pattern that could name a stub source, a test command pattern that could
run a stub target, and a stub source that a CMake file names in a form the reader cannot attribute to a target. A
target whose name is such a pattern is matched by the same token in a test command.

Each check is proven to catch its mutant below, by the failure text of that check.
"""

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
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "build"))]
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
        for token, cm, _ in entries:
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
            m = _BRACKET_OPEN.match(text, i + 1)
            if m:
                end = text.find("]" + m.group(1) + "]", m.end())
                end = n if end < 0 else end + len(m.group(1)) + 2
            else:
                end = text.find("\n", i)
                end = n if end < 0 else end
            line += text.count("\n", i, end)
            i = end
            continue
        m = re.compile(r"[A-Za-z_][A-Za-z0-9_]*").match(text, i)
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
                bm = _BRACKET_OPEN.match(text, k + 1)
                if bm:
                    end = text.find("]" + bm.group(1) + "]", bm.end())
                    end = n if end < 0 else end + len(bm.group(1)) + 2
                else:
                    end = text.find("\n", k)
                    end = n if end < 0 else end
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
HELPERS = frozenset(("llama_build", "llama_build_and_test", "llama_test", "llama_test_cmd", "llama_test_pytest"))


def _parse_arguments(sargs, env):
    """cmake_parse_arguments(<prefix> <options> <one_value> <multi_value> <args>...)."""
    if len(sargs) < 4 or sargs[0] == "PARSE_ARGV":
        return
    prefix, options, ones, multis = sargs[0], split_list([sargs[1]]), split_list([sargs[2]]), split_list([sargs[3]])
    values, current, unparsed = {}, None, []
    for item in split_list(sargs[4:]):
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
    for key in options:
        env[f"{prefix}_{key}"] = ";".join(values.get(key, ["FALSE"]))
    for key in ones + multis:
        if key in values:
            env[f"{prefix}_{key}"] = ";".join(values[key])
        else:
            env.pop(f"{prefix}_{key}", None)
    env[f"{prefix}_UNPARSED_ARGUMENTS"] = ";".join(unparsed)


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


def expand(cmds, env, functions, out, depth=0):
    """Replay the commands in order: set()/list(APPEND)/cmake_parse_arguments() update `env`, foreach() bodies are
    replayed per item, function()/macro() bodies are recorded in `functions` and replayed at each call site (the
    HELPERS excepted) with their parameters and ARGN bound. A foreach() whose items cannot be listed (RANGE, or a
    list the reader never saw set) is replayed once with its variable left unresolved, so what its body registers is
    still seen, as a pattern. Every other command is appended to `out` with its variables substituted."""
    idx = 0
    while idx < len(cmds):
        name, args, line = cmds[idx]
        if name in ("function", "macro"):
            end = _block_end(cmds, idx, name, "end" + name)
            if args:
                functions[args[0].lower()] = (args[1:], cmds[idx + 1:end])
            idx = end + 1
            continue
        if name in functions and name not in HELPERS and depth < 8:
            params, body = functions[name]
            argv = split_list([substitute(a, env) for a in args])
            scope = dict(env)
            for k, param in enumerate(params):
                if k < len(argv):
                    scope[param] = argv[k]
                else:
                    scope.pop(param, None)
            scope["ARGN"] = ";".join(argv[len(params):])
            scope["ARGV"] = ";".join(argv)
            scope["ARGC"] = str(len(argv))
            for k, a in enumerate(argv):
                scope[f"ARGV{k}"] = a
            expand(body, scope, functions, out, depth + 1)
            out.append((name, [substitute(a, env) for a in args], line))
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
            elif rest and rest[0] == "RANGE":
                items = [None]
            else:
                items = split_list(rest)
            if any(item is not None and _VAR_REF.search(item) for item in items):
                items = [None]
            saved = env.get(var)
            for item in items:
                if item is None:
                    env.pop(var, None)  # the body is still read, with its variable left as a pattern
                else:
                    env[var] = item
                expand(body, env, functions, out, depth)
            if saved is None:
                env.pop(var, None)
            else:
                env[var] = saved
            idx = end + 1
            continue
        sargs = [substitute(a, env) for a in args]
        if name == "set" and sargs:
            vals = [a for a in sargs[1:] if a not in ("PARENT_SCOPE",)]
            if "CACHE" in vals:
                vals = vals[:vals.index("CACHE")]
            env[sargs[0]] = ";".join(vals)
        elif name == "list" and len(sargs) >= 2 and sargs[0] == "APPEND":
            cur = env.get(sargs[1], "")
            env[sargs[1]] = ";".join([x for x in [cur] if x] + sargs[2:])
        elif name == "cmake_parse_arguments":
            _parse_arguments(sargs, env)
        else:
            # A variable a command computes (list(GET), string(), math(), ...) is not evaluated: it is unset, so what
            # uses it reads as a pattern rather than as a stale value from an earlier command.
            for var in _computed(name, sargs):
                env.pop(var, None)
        out.append((name, sargs, line))
        idx += 1


def scoped(args, keywords=("PRIVATE", "PUBLIC", "INTERFACE")):
    """[(scope, item)] for a target_* argument list after the target name. An item before any keyword is PUBLIC (the
    plain signature of target_link_libraries is transitive)."""
    scope, out = "PUBLIC", []
    for a in args:
        if a in keywords:
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
        self.executables = {}  # target -> [(source token, cmake file, line)]
        self.defs = {}  # target -> [(scope, item)]
        self.links = {}  # target -> [(scope, item)]
        self.tests = {}  # test name -> (target, cmake file, line, inherent skip code or None)
        self.test_props = {}  # test name -> {prop: value}
        self.unresolved = []  # (pattern, cmake file, line): a source token with a variable left in it
        self.function_publishes = set()  # libraries a function body gives GGML_USE_SYCL
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
            expand(cmake_commands(tree[path]), env, functions, out)
            per_file[path] = out
        self._read_functions(functions, per_file)
        for path, out in per_file.items():
            for name, args, line in out:
                self._take(path, name, args, line)

    # The helpers in tests/CMakeLists.txt build an executable named after the source and link what their body links.
    def _read_functions(self, functions, per_file):
        backend_calls = set()
        for out in per_file.values():
            for name, args, _ in out:
                if name == "ggml_add_backend" and args:
                    backend_calls.add(args[0].upper())
        for fname, (params, body) in functions.items():
            # Read as written: a body's set() calls depend on its arguments, so substituting them would rename the
            # very variables (`${TEST_TARGET}`, `${backend_use}`) the reading keys on.
            body_flat = body
            # ggml_add_backend(): string(TOUPPER "GGML_USE_${backend}" v) then target_compile_definitions(ggml PUBLIC
            # ${v}) gives every called backend's GGML_USE_<backend> to ggml's consumers.
            upper_vars = {}
            for name, args, _ in body_flat:
                if name == "string" and len(args) == 3 and args[0] == "TOUPPER":
                    upper_vars[args[2]] = args[1]
            for name, args, _ in body_flat:
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
                for name, args, _ in body_flat:
                    if name == "target_link_libraries" and args and args[0] == "${TEST_TARGET}":
                        links.extend(item for _, item in scoped(args[1:]))
                    if name == "set_tests_properties" and "PROPERTIES" in args:
                        kv = args[args.index("PROPERTIES") + 1:]
                        for k, v in zip(kv[::2], kv[1::2]):
                            if k == "SKIP_RETURN_CODE":
                                skip = v
                self.helpers[fname] = {"links": links, "skip": skip}

    def _add_source(self, target, token, path, line):
        self.executables.setdefault(target, []).append((token, path, line))

    def _take(self, path, name, args, line):
        if name == "add_executable" and args:
            if any(a in ("IMPORTED", "ALIAS") for a in args[1:]):
                return
            for token in args[1:]:
                if token not in ("WIN32", "MACOSX_BUNDLE", "EXCLUDE_FROM_ALL"):
                    self._add_source(args[0], token, path, line)
        elif name == "target_sources" and args:
            for _, token in scoped(args[1:]):
                self._add_source(args[0], token, path, line)
        elif name in ("llama_build", "llama_build_and_test") and args:
            target = re.sub(r"\.[^.]*$", "", os.path.basename(args[0]))
            if name == "llama_build_and_test" and "NAME" in args:
                target = args[args.index("NAME") + 1]
            self._add_source(target, args[0], path, line)
            helper = self.helpers.get(name, {"links": [], "skip": None})
            self.links.setdefault(target, []).extend(("PRIVATE", lib) for lib in helper["links"])
            if name == "llama_build_and_test":
                self.tests[target] = ((target,), path, line, helper["skip"])
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
            self.tests[test] = (tuple(command), path, line, None)
        elif name == "llama_test" and args:
            test = args[args.index("NAME") + 1] if "NAME" in args else args[0]
            self.tests[test] = ((args[0],), path, line, None)
        elif name == "llama_test_cmd" and args:  # runs its first argument with ARGS; llama_test_pytest runs pytest
            test = args[args.index("NAME") + 1] if "NAME" in args else args[0]
            rest = args[args.index("ARGS") + 1:] if "ARGS" in args else []
            for stop in ("NAME", "LABEL", "WORKING_DIRECTORY"):
                rest = rest[:rest.index(stop)] if stop in rest else rest
            self.tests[test] = ((args[0], *rest), path, line, None)
        elif name == "set_tests_properties" and "PROPERTIES" in args:
            at = args.index("PROPERTIES")
            kv = args[at + 1:]
            for test in args[:at]:
                props = self.test_props.setdefault(test, {})
                for k, v in zip(kv[::2], kv[1::2]):
                    props[k] = v
        elif name == "set_property" and len(args) >= 4 and args[0] == "TEST" and "PROPERTY" in args:
            at = args.index("PROPERTY")
            if at + 1 < len(args):
                key, vals = args[at + 1], args[at + 2:]
                for test in args[1:at]:
                    self.test_props.setdefault(test, {})[key] = ";".join(vals)

    # -----------------------------------------------------------------------------------------------------------------

    @staticmethod
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
        if any(is_define(item) for _, item in self.defs.get(target, [])):
            return True
        return any(item in pub for _, item in self.links.get(target, []))


# ---------------------------------------------------------------------------------------------------------------------
# C/C++ reading
# ---------------------------------------------------------------------------------------------------------------------

_LITERAL = r'R"([^(\s]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
_LEXEME_RE = re.compile(_LITERAL + r"|//[^\n]*|/\*.*?\*/", flags=re.DOTALL)


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


_MAIN = re.compile(r"\bint\s+main\s*\([^;{)]*\)\s*\{")
_RETURN = re.compile(r"\breturn\b([^;]*);")


def stubs(src: str) -> list:
    """[(line, returned expression or None)] for each stub: a return in `main` on a GGML_USE_SYCL-off line, and a
    `main` defined on such a line that has no return (None)."""
    code = code_only(src)
    off = off_lines(code)
    found = []
    for m in _MAIN.finditer(code):
        start = m.end()
        depth, i = 1, start
        while i < len(code) and depth:
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            i += 1
        main_line = code.count("\n", 0, m.start()) + 1
        returns = []
        for r in _RETURN.finditer(code, start, i):
            line = code.count("\n", 0, r.start()) + 1
            if line in off:
                returns.append((line, " ".join(r.group(1).split())))
        found.extend(returns)
        if main_line in off and not returns:
            found.append((main_line, None))
    return found


# ---------------------------------------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------------------------------------


_STUBS = {}


def stub_sources(tree):
    """{source: stubs} for every source in the tree with at least one stub (memoised by text)."""
    found = {}
    for p, t in sorted(tree.items()):
        if p.endswith(SOURCE_SUFFIXES):
            if t not in _STUBS:
                _STUBS[t] = stubs(t)
            if _STUBS[t]:
                found[p] = _STUBS[t]
    return found


def compiling_targets(model, src):
    """(targets, unresolved): the targets whose resolved sources include `src`, and the unresolved source tokens that
    could name it. A token is a candidate only when it spells the file's stem: a bare `${files}` could name anything,
    and a stub source compiled that way is caught by `named_but_unattributed` instead."""
    stem = re.sub(r"\.[^.]*$", "", os.path.basename(src))
    targets, unresolved = set(), []
    for target, entries in model.executables.items():
        target_unresolved = "${" in target or "$<" in target
        for tok, cm, line in entries:
            got = model.resolve(tok, cm)
            if got == src and not target_unresolved:
                targets.add(target)
            elif (got == src or (got is None and model.pattern(tok, cm).fullmatch(src))) and stem in tok:
                unresolved.append((tok if not target_unresolved else f"{target}: {tok}", cm, line))
    return targets, unresolved


def named_but_unattributed(tree, src):
    """CMake files that name `src` (its file name, a path ending in it, or its stem as a whole argument) although no
    target was found to compile it."""
    base = os.path.basename(src)
    stem = re.sub(r"\.[^.]*$", "", base)
    hits = []
    for cm in sorted(p for p in tree if p.endswith(("CMakeLists.txt", ".cmake"))):
        for _, args, line in cmake_commands(tree[cm]):
            for a in split_list(args):
                if a in (base, stem) or a.endswith("/" + base):
                    hits.append((cm, line))
                    break
    return hits


def define_failures(tree, model=None):
    model = model or Model(tree)
    pub = model.publishers()
    failures = []
    for src in stub_sources(tree):
        targets, unresolved = compiling_targets(model, src)
        for tok, cm, line in unresolved:
            failures.append(f"{cm}:{line}: cannot tell whether `{tok}` compiles {src}, which carries a GGML_USE_SYCL "
                            "stub; name its sources literally")
        if not targets and not unresolved:
            for cm, line in named_but_unattributed(tree, src):
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
        for line, expr in found:
            if expr is None:
                failures.append(f"{src}:{line}: the GGML_USE_SYCL-off stub main has no return, so it exits 0")
            elif expr != SKIP_MACRO:
                failures.append(f"{src}:{line}: the GGML_USE_SYCL-off stub returns `{expr}`, not {SKIP_MACRO}")
    return failures


# Programs that run the rest of their command line and exit with its code: {name: (options taking a value, number of
# positional arguments before the command)}. `env` also takes NAME=VALUE assignments.
WRAPPERS = {
    "timeout": ({"-s", "--signal", "-k", "--kill-after"}, 1),
    "env": ({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}, 0),
    "nice": ({"-n", "--adjustment"}, 0),
    "setsid": (set(), 0),
    "stdbuf": ({"-i", "-o", "-e"}, 0),
}


def program(command) -> str:
    """The program a test command runs, after any WRAPPERS (and `${CMAKE_COMMAND} -E env`) in front of it: the test's
    exit code is that program's."""
    cmd = list(command)
    while cmd:
        if cmd[0] == "${CMAKE_COMMAND}" and cmd[1:3] == ["-E", "env"]:
            cmd = ["env"] + cmd[3:]
        base = cmd[0].rsplit("/", 1)[-1]
        if base not in WRAPPERS:
            return cmd[0]
        takes_value, positional = WRAPPERS[base]
        i = 1
        while i < len(cmd) and (cmd[i].startswith("-") or (base == "env" and "=" in cmd[i])):
            i += 2 if cmd[i] in takes_value else 1
        cmd = cmd[i + positional:]
    return ""


def runs_target(exe, target):
    """True when the program token `exe` names `target`'s binary: by name, by $<TARGET_FILE:...>, or by a path that
    ends in it (`${CMAKE_BINARY_DIR}/bin/<target>`)."""
    return exe in (target, f"$<TARGET_FILE:{target}>") or ("/" in exe and exe.rsplit("/", 1)[1] == target)


_TOOL_VAR = re.compile(r"\$\{(CMAKE_[A-Za-z0-9_]*|[A-Za-z0-9_]*(?:EXECUTABLE|COMMAND|PROGRAM))\}")


def could_run_target(exe, target):
    """True when `exe` keeps a variable the reader could not resolve and, read as a pattern, could name `target`.
    A tool variable on its own (`${Python3_EXECUTABLE}`, `${CMAKE_COMMAND}`) names an interpreter or tool, not a test
    binary, and is not a candidate."""
    m = re.fullmatch(r"\$<TARGET_FILE:(.*)>", exe)
    name = m.group(1) if m else exe
    if not re.search(r"\$\{|\$<", name):
        return False
    if not m and _TOOL_VAR.fullmatch(name):
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
            if _VAR_REF.search(target) and any(model.resolve(tok, cm) == src for tok, cm, _ in entries):
                stub_targets.setdefault(target, src)
    failures = []
    for test, (command, cm, line, inherent) in sorted(model.tests.items()):
        # The test's exit code is its program's: the binary when the command runs it, directly or through a wrapper
        # such as timeout. A command that hands the binary to another program as an argument (the carrier linkage
        # audit reads the files) answers for itself.
        exe = program(command)
        paired = any(runs_target(exe, t) for t in model.executables if _VAR_REF.search(t))
        for target in sorted(stub_targets):
            if runs_target(exe, target):
                skip = model.test_props.get(test, {}).get("SKIP_RETURN_CODE", inherent)
                if skip != "77":
                    failures.append(f"{cm}:{line}: test {test} runs {target} (GGML_USE_SYCL stub in "
                                    f"{stub_targets[target]}) without SKIP_RETURN_CODE 77")
            elif not paired and could_run_target(exe, target):
                failures.append(f"{cm}:{line}: cannot tell whether test {test} (command `{exe}`) runs {target} "
                                f"(GGML_USE_SYCL stub in {stub_targets[target]}); name the binary literally")
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
    targets, _ = compiling_targets(model, "tests/test-unified-cache-bugs.cpp")
    assert targets == {"test-unified-cache-bugs"}, targets
    targets, _ = compiling_targets(model, "tests/test-sycl-zone-reset-live-refusal.cpp")
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
        "test unified-cache-bugs runs test-unified-cache-bugs"),
    # A registration through llama_test() (which sets no skip code of its own) loses its skip code.
    "llama-test-registration-without-skip-code": (
        _sub(TESTS_CMAKE,
             r"(set_tests_properties\(test-sycl-zone-reset-live-refusal PROPERTIES[^)]*?)\n *SKIP_RETURN_CODE 77", r"\1"),
        "test test-sycl-zone-reset-live-refusal runs test-sycl-zone-reset-live-refusal"),
    # A second registration names the binary through $<TARGET_FILE:...> and sets no skip code.
    "target-file-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-again COMMAND $<TARGET_FILE:test-unified-cache-bugs> --again)\n"),
        "test ucb-again runs test-unified-cache-bugs"),
    # A stub registered through llama_build_and_test(), whose body no longer sets the skip code.
    "helper-without-skip-code": (
        _both(_edit(TESTS_CMAKE, "        LABELS \"${LLAMA_TEST_LABEL}\"\n        SKIP_RETURN_CODE 77)\nendfunction()",
                    "        LABELS \"${LLAMA_TEST_LABEL}\")\nendfunction()"),
              _add(TESTS_CMAKE, "\nllama_build_and_test(test-sycl-pointer-types.cpp)\n")),
        "test test-sycl-pointer-types runs test-sycl-pointer-types"),
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
        "test ucb-timeout runs test-unified-cache-bugs"),
    # The binary named by its path in the build tree.
    "binary-path-registration": (
        _add(SYCL_CMAKE, "\nadd_test(NAME ucb-path COMMAND ${CMAKE_BINARY_DIR}/bin/test-unified-cache-bugs)\n"),
        "test ucb-path runs test-unified-cache-bugs"),
    # A registration inside a function the gate has no rule for: its call sites are replayed.
    "function-registration-without-skip-code": (
        _add(SYCL_CMAKE, "\nfunction(ucb_again_test suffix)\n"
             "    add_test(NAME ucb-${suffix} COMMAND test-unified-cache-bugs ${ARGN})\nendfunction()\n"
             "ucb_again_test(fn --again)\n"),
        "test ucb-fn runs test-unified-cache-bugs"),
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
        ("cannot tell whether test ucb-r${_i}", "runs test-unified-cache-bugs")),
    # A stub source outside any `tests` directory, compiled by a target that does not receive the define.
    "stub-source-outside-tests": (
        _relist({"tools/stubbed/stubbed.cpp": SYNTHETIC_ELSE_STUB,
                 "tools/stubbed/CMakeLists.txt": "add_executable(stubbed-tool stubbed.cpp)\n"}),
        "target stubbed-tool compiles tools/stubbed/stubbed.cpp, which carries a GGML_USE_SYCL stub"),
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
