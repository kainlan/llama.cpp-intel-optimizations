"""Independent checks for the pure-Python SYCL source gates.

A gate written as a chain of top-level `assert` statements stops at the first
stale pin, so every check after it silently never runs and one failure hides
the rest (llama.cpp-gsb9, llama.cpp-qeld). Wrap each check in `with gate(name):`
instead: a failure is recorded with its source line, the next check still runs,
and `finish()` turns the record into the process exit status.

    from sycl_gate import gate, finish

    with gate("the reserve path allocates owner-first"):
        assert "unified_allocate_owner(" in block
    ...
    finish("my gate: PASS")

Not a pytest plugin: pytest-style gates (module-level test_* functions) are
already independent, one result per function; they use two kinds of helper
from this module. First, the message helpers: a claim that returns a bare
bool says nothing about WHICH token failed, so pass first_missing (names the
first anchor the claim needs that its region lacks) or first_present (names
the first forbidden text its region holds) as the message.

    from sycl_gate import first_missing

    assert claim_x(SYCL), first_missing(
        "file: region", region, ANCHOR_A, ANCHOR_B)

Second, the text helpers for source-contract pins, after the message helpers:
ROOT (the repo root), strip_comments (source text with comments blanked, line
breaks kept), norm (comment-stripped, whitespace-canonical text), read (a repo
file's text, or empty when it is missing), body (the brace-balanced body of a
definition), ordered (needles in order), mutate (a normalised text with one
replacement, for positive controls) and resolve_mutants (source text with every
`#if` over a MUTANT macro resolved as the production build compiles it).

Third, last in the module, what the pins of the device REDs registered through one
CMake function share: tree_texts and missing (the pinned files' texts, and the ones
absent), the CMake reader (cmake_strip, cmake_args, registration, arms_of,
if_stack_at), the registering function's contract (function_problems), rejects (a
positive control's check) and tree_tests (the three tree tests each such pin binds).
"""
import contextlib
import linecache
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

_RESULTS = []


@contextlib.contextmanager
def gate(name: str):
    try:
        yield
    except Exception as error:  # noqa: BLE001 -- any failure, including a missing anchor, fails this check
        frame = traceback.extract_tb(error.__traceback__)[-1]
        location = "line %d: %s" % (frame.lineno, (frame.line or "").strip())
        detail = str(error) or type(error).__name__
        _RESULTS.append((name, "%s -- %s: %s%s" % (location, type(error).__name__, detail, _operands(error))))
    else:
        _RESULTS.append((name, None))


def _operands(error: BaseException) -> str:
    """A bare `assert token in text` says nothing about WHICH token failed inside a loop, so list the
    short str/int/bool names the failing statement mentions.

    Names bound by a comprehension inside that statement (`all(needle in text for needle in ...)`), or as the
    parameters of a lambda/def written in it, are NOT
    resolvable: the comprehension has finished by the time the assert fails, and looking the name up in the
    module scope would report whatever a stale module-level variable of the same name last held, blaming the
    wrong needle. Those names are left out rather than guessed.
    """
    tb = error.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    summary = traceback.extract_tb(error.__traceback__)[-1]
    lines = [linecache.getline(summary.filename, number)
             for number in range(summary.lineno, (getattr(summary, "end_lineno", None) or summary.lineno) + 1)]
    statement = "".join(lines) or summary.line or ""
    frame = tb.tb_frame
    scope = dict(frame.f_globals)
    scope.update(frame.f_locals)
    bound_here = set()
    for targets in re.findall(r"\bfor\s+([A-Za-z_][A-Za-z_0-9]*(?:\s*,\s*[A-Za-z_][A-Za-z_0-9]*)*)\s+in\b", statement):
        bound_here.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", targets))
    # Parameters of a lambda (or a def) written in the statement are bound inside it too.
    for params in re.findall(r"\blambda\b([^:]*):", statement) + re.findall(r"\bdef\s+\w+\s*\(([^)]*)\)", statement):
        for piece in params.split(","):
            name = re.match(r"\s*\**\s*([A-Za-z_][A-Za-z_0-9]*)", piece)
            if name:
                bound_here.add(name.group(1))
    if frame.f_code.co_name.startswith("<") and frame.f_code.co_name != "<module>":
        # A generator-expression / lambda frame: names missing from its own locals live in a scope we cannot
        # see from here, so do not fall back to module globals.
        scope = dict(frame.f_locals)
    shown = []
    for word in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", statement)):
        value = scope.get(word)
        if word in bound_here:
            continue
        if isinstance(value, (str, int, bool)) and len(repr(value)) <= 120 and word not in ("True", "False"):
            shown.append("%s=%r" % (word, value))
    return "  [%s]" % ", ".join(shown) if shown else ""


def failures() -> list:
    return [(name, why) for name, why in _RESULTS if why is not None]


def finish(pass_message: str = "", min_checks: int = 1) -> None:
    """Report every failed check, then exit non-zero if there was one.

    A gate whose `with gate(...)` blocks never ran (an early return, a guard that skips them, a rename that
    left none registered) would otherwise print "0/0 checks passed" and exit 0, which proves nothing. Require
    at least `min_checks` recorded checks; pass the real count to also catch a gate that silently lost some.
    """
    failed = failures()
    if len(_RESULTS) < min_checks:
        print("FAIL too few checks ran: %d recorded, at least %d required" % (len(_RESULTS), min_checks),
              file=sys.stderr)
        sys.exit(1)
    for name, why in failed:
        print("FAIL %s\n     %s" % (name, why.replace("\n", "\n     ")), file=sys.stderr)
    print("%d/%d checks passed" % (len(_RESULTS) - len(failed), len(_RESULTS)))
    if failed:
        sys.exit(1)
    if pass_message:
        print(pass_message)


def run_without_source(script: str):
    """Run a copy of the gate `script` in a tree that lacks the sources it pins: (exit status, stdout + stderr).

    The copy sits in tmp/tests/ with this module beside it, so `../ggml/...` does not exist for it. A gate that pins a
    source file must FAIL there (and say the file was not found); exiting 77 is ctest's skip, which reads as green,
    and a moved or renamed source is exactly what such a gate exists to notice. Callers check both the status and
    the text, so a gate that dies on something else (an ImportError) does not pass for a missing-source failure.
    """
    with tempfile.TemporaryDirectory(prefix="gate-missing-source-") as tmp:
        tests = os.path.join(tmp, "tests")
        os.makedirs(tests)
        copy = os.path.join(tests, os.path.basename(script))
        shutil.copy(script, copy)
        shutil.copy(os.path.abspath(__file__), os.path.join(tests, "sycl_gate.py"))
        done = subprocess.run([sys.executable, copy], capture_output=True, text=True, timeout=60)
        return done.returncode, done.stdout + done.stderr


def _shown(text: str) -> str:
    """`text` for a failure message: whole when short, else cut to 100 characters with a trailing "..."."""
    return text if len(text) <= 100 else text[:97] + "..."


def first_missing(where: str, text: str, *anchors: str, signature: str = "") -> str:
    """The message for `assert claim(...), first_missing(...)`: names the first anchor the claim needs that `text`
    does not contain, and where it looked, so a stale pin says what it expected instead of a bare AssertionError.
    When every anchor is present the claim failed on an order, count or depth condition, and the message says so.
    `signature`, when given, is the text that opens the region `text` was cut from (a function body); an empty
    `text` then means the signature moved, and the message names the signature. Never empty, so the assert always
    prints it."""
    if signature and not text:
        return "%s: missing %r" % (where, _shown(signature))
    for anchor in anchors:
        if anchor not in text:
            return "%s: missing %r" % (where, _shown(anchor))
    if len(anchors) == 1:
        return "%s: the anchor is present; an order, count or depth condition failed" % where
    return "%s: all %d anchors present; an order, count or depth condition failed" % (where, len(anchors))


def first_present(where: str, text: str, *forbidden: str) -> str:
    """The message for an absence claim: names the first forbidden text that `text` does contain, and where, or says
    none of them is present (so the claim failed on an emptiness or count condition). Never empty."""
    for found in forbidden:
        if found in text:
            return "%s: found %r" % (where, _shown(found))
    if len(forbidden) == 1:
        return "%s: the forbidden text is absent; an emptiness or count condition failed" % where
    return "%s: none of %d forbidden texts present; an emptiness or count condition failed" % (where, len(forbidden))


# ---- text helpers for source-contract pins ----------------------------------------------------------------------------

# The repo root, resolved the way the pins resolved it from their own path: this file sits in tests/, so parents[1] of
# it is the same directory as parents[1] of any pin beside it. The pins import it rather than define their own.
ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    """`src` with comments blanked (their line breaks kept) and string and character literals left whole."""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


def _spaces(s: str) -> str:
    return re.sub(r"[^\n]", " ", s)


def views(raw: str):
    """`raw` as (code, bare): `code` blanks comments and keeps string and character literals, and `bare` also blanks
    each literal's inside, keeping its quotes. Unlike strip_comments, which drops a comment's columns, both keep every
    offset and line break of `raw`, so a position means the same thing in all three."""
    code, bare, at = [], [], 0
    for m in _LEXEME_RE.finditer(raw):
        code.append(raw[at:m.start()])
        bare.append(raw[at:m.start()])
        tok = m.group(0)
        if tok[0] in "\"'":
            code.append(tok)
            bare.append(tok[0] + _spaces(tok[1:-1]) + tok[-1])
        else:
            code.append(_spaces(tok))
            bare.append(_spaces(tok))
        at = m.end()
    code.append(raw[at:])
    bare.append(raw[at:])
    return "".join(code), "".join(bare)


_BRACKET_RE = {"(": re.compile(r"[()]"), "{": re.compile(r"[{}]"), "[": re.compile(r"[\[\]]")}


def match_close(text: str, open_at: int) -> int:
    """The offset of the bracket closing the one at `open_at`, or -1. Only that bracket kind is counted, so pass a bare
    view, where no literal holds a bracket."""
    opener = text[open_at]
    depth = 0
    for m in _BRACKET_RE[opener].finditer(text, open_at):
        if m.group(0) == opener:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.start()
    return -1


def norm(text: str) -> str:
    """Comment-stripped, whitespace-canonical text with adjacent string literals joined, so neither a clang-format
    re-wrap nor a re-split format string can turn a claim red. Idempotent, so a mutant built on normalized text is
    read the same way as the source."""
    t = re.sub(r"\s+", " ", strip_comments(text))
    t = re.sub(r"\( ", "(", t)
    t = re.sub(r" \)", ")", t)
    t = re.sub(r" ,", ",", t)
    t = re.sub(r",(?! )", ", ", t)
    t = re.sub(r'(?<!\\)" "', "", t)
    return t


def read(rel: str) -> str:
    """The file's text, or empty when it does not exist yet, so a missing file fails a presence claim instead of
    erroring. An absence claim passes vacuously on "", so it needs a non-empty region (or a check that this read is)."""
    p = ROOT / rel
    return p.read_text(errors="replace") if p.exists() else ""


def body(text: str, signature: str) -> str:
    """The brace-balanced body of the first definition whose text starts with `signature` (string literals skipped
    while balancing). A signature ending in "{" opens the body itself. Empty when there is none."""
    at = text.find(signature)
    if at < 0:
        return ""
    open_at = at + len(signature) - 1 if signature.endswith("{") else text.find("{", at + len(signature))
    if open_at < 0:
        return ""
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    return ""


def ordered(text: str, *needles: str) -> bool:
    """Every needle is present, each after the one before it."""
    at = 0
    for needle in needles:
        at = text.find(needle, at)
        if at < 0:
            return False
        at += len(needle)
    return True


def mutate(text: str, old: str, new: str) -> str:
    """`text` (normalized) with `old` replaced once; the mutant must exist, or it would prove nothing."""
    n = norm(text)
    assert old in n, f"mutant precondition: {old!r} is not in the text"
    return n.replace(old, new, 1)


# A planted fault lives in a block whose `#if` names a MUTANT macro (a CMake variant's -D, or the macro a private-
# fixture header defines for its runtime seams). The production build compiles none of those blocks, so a gate that
# scores production code reads the text with them resolved.
_DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def _mutant_condition(kw: str, rest: str) -> bool:
    """A directive's value with every mutant macro undefined. Only a condition over mutant macros is evaluated."""
    rest = re.sub(r"//.*$", "", rest).strip()
    if kw == "ifdef":
        return False
    if kw == "ifndef":
        return True
    expr = re.sub(r"\bdefined\s*\(\s*\w*MUTANT\w*\s*\)", " False ", rest)
    expr = re.sub(r"\bdefined\s+\w*MUTANT\w*", " False ", expr)
    expr = expr.replace("||", " or ").replace("&&", " and ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    assert re.fullmatch(r"[\s()notandrFalse]*", expr), f"mutant directive not understood: #{kw} {rest}"
    return bool(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 -- the text is checked to be boolean literals


def resolve_mutants(raw: str) -> str:
    """`raw` with each mutant block resolved; dropped lines become empty, so line numbers and offsets per line hold."""
    out = []
    stack = []

    def active(entries) -> bool:
        return all(e["taking"] for e in entries if e["mutant"])

    for line in raw.split("\n"):
        m = _DIRECTIVE_RE.match(line)
        if m:
            kw, rest = m.group(1), m.group(2)
            if kw in ("if", "ifdef", "ifndef"):
                if "MUTANT" in rest:
                    val = _mutant_condition(kw, rest)
                    stack.append({"mutant": True, "taking": val, "taken": val})
                    out.append("")
                else:
                    out.append(line if active(stack) else "")
                    stack.append({"mutant": False})
                continue
            top = stack[-1] if stack else {"mutant": False}
            if kw == "elif" and top["mutant"]:
                val = (not top["taken"]) and _mutant_condition("if", rest)
                top["taking"], top["taken"] = val, top["taken"] or val
                out.append("")
                continue
            if kw == "else" and top["mutant"]:
                top["taking"], top["taken"] = not top["taken"], True
                out.append("")
                continue
            if kw == "endif":
                stack.pop()
                out.append("" if top["mutant"] else (line if active(stack) else ""))
                continue
            out.append(line if active(stack[:-1]) else "")
            continue
        out.append(line if active(stack) else "")
    return "\n".join(out)


# ---- the pins of the device REDs one CMake function registers --------------------------------------------------------
#
# The SCRATCH return's three device REDs are registered through one function, with their planted faults as MUTANTS
# triples, and each has a pin that reads that registration, the function's contract and the sources the test needs.
# What the three pins share lives here, so a fix to the CMake reader or to the function's contract is made once: each
# pin passes its own file list, function name, guarding block, CMake path and clause table.


def tree_texts(files) -> dict:
    """Each file's raw text; a missing file is absent from the dict, so the clauses name it."""
    out = {}
    for rel in files:
        p = ROOT / rel
        if p.exists():
            out[rel] = p.read_text(errors="replace")
    return out


def missing(texts: dict, *rels: str) -> list:
    return [f"{rel} not found" for rel in rels if rel not in texts]


def cmake_strip(text: str) -> str:
    """CMake text with `#` comments blanked (their line breaks kept); a `#` inside a quoted argument stays."""
    out, i, n, quoted = [], 0, len(text), False
    while i < n:
        c = text[i]
        if quoted:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                quoted = False
        elif c == '"':
            quoted = True
            out.append(c)
        elif c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def cmake_args(text: str, open_at: int) -> list:
    """The arguments of the command whose `(` is at `open_at`: quoted ones as written between the quotes (escapes
    kept), unquoted ones as tokens. None when the command is not closed. The function re-expands its arguments as
    ${ARGN}, which splits a quoted argument at each unescaped `;`, so a quoted one is split there too: a semicolon in
    an expected text adds a list element and shifts every later triple, as CMake reads it."""
    args, i, n = [], open_at + 1, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ")":
            return args
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            args += re.split(r"(?<!\\);", text[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            args.append(text[i:j])
            i = j
    return None


def registration(cmake: str, func: str, stem: str):
    """(offset, {"TIMEOUT": [...], "ENV": [...], "MUTANTS": [...]}) of the `func(<stem> ...)` call, or None."""
    m = re.search(r"^[ \t]*" + func + r"\s*\(\s*" + re.escape(stem) + r"\s", cmake, re.M)
    if not m:
        return None
    args = cmake_args(cmake, cmake.index("(", m.start()))
    if args is None:
        return None
    parsed, key = {"TIMEOUT": [], "ENV": [], "MUTANTS": []}, None
    for a in args[1:]:
        if a in parsed:
            key = a
        elif key is not None:
            parsed[key].append(a)
    return m.start(), parsed


def arms_of(parsed: dict):
    """The MUTANTS triples as {names: (cases, expected text)}, or None when the list is not whole triples."""
    flat = parsed["MUTANTS"]
    if len(flat) % 3:
        return None
    return {flat[i]: (flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)}


def if_stack_at(cmake: str, offset: int) -> list:
    """The `if (...)` lines open at `offset`, outermost first."""
    stack = []
    for m in re.finditer(r"^[ \t]*(if|endif)\s*\((.*)$", cmake[:offset], re.M):
        if m.group(1) == "if":
            stack.append(("if (" + m.group(2)).strip())
        elif stack:
            stack.pop()
    return stack


def function_problems(cmake: str, func: str, block: str, cmake_rel: str) -> list:
    """The contract of `func`, the function that registers the SCRATCH return's device REDs (§M82 case (2), §M88 (2)),
    in the comment-stripped text of `cmake_rel`: defined inside `block`, linked against the private fixtures and never
    libllama, pinned to one device and 64 MiB chunks, and every arm scored by its expected text."""
    m = re.search(r"^[ \t]*function\s*\(\s*" + func + r"\s+stem\s*\)(.*?)^[ \t]*endfunction\s*\(\s*\)", cmake,
                  re.M | re.S)
    if not m:
        return [f"{cmake_rel}: no function({func} stem)"]
    f = m.group(1)
    fails = []
    if block not in if_stack_at(cmake, m.start()):
        fails.append(f"{func} is defined outside `{block}`")
    need = {
        "target_link_libraries(${stem} PRIVATE ggml-sycl-private-fixtures": "links the private fixtures",
        '"ONEAPI_DEVICE_SELECTOR=level_zero:1"': "pins the selector to level_zero:1",
        '"GGML_SYCL_PINNED_CHUNK_MB=64"': "sets 64 MiB chunks",
        'LABELS "sycl;cache;mem-handle;gpu-serial"': "labels the base ctest",
        'LABELS "sycl;cache;mem-handle;gpu-serial;mutant"': "labels each arm",
        "COMMAND ${stem} --mutant ${_mutants} --cases ${_cases}": "runs each arm on its mutants and cases",
        'PASS_REGULAR_EXPRESSION "${_regex}"': "scores each arm by its expected text",
    }
    for text, what in need.items():
        if text not in f:
            fails.append(f"{func} no longer {what}: missing {text!r}")
    for text, n in (("RUN_SERIAL TRUE", 2), ("SKIP_RETURN_CODE 77", 2), ('ENVIRONMENT "${_env}"', 2)):
        if f.count(text) != n:
            fails.append(f"{func} carries {text!r} {f.count(text)} time(s), expected {n} (base and arm)")
    if re.search(r"target_link_libraries\([^)]*\bllama\b", f):
        fails.append(f"{func} links libllama")
    return fails


def rejects(clauses: dict, clause: str, texts: dict, reason: str) -> None:
    """A positive control: `clauses[clause]` must reject the planted `texts` with a message containing `reason`."""
    fails = clauses[clause](texts)
    assert any(reason in f for f in fails), f"{clause} did not reject the control with {reason!r}: {fails}"


def tree_tests(files, clauses: dict, script: str):
    """The three tests each such pin runs on its own tree, as (test_sources_exist, test_clause_holds_on_the_tree,
    test_missing_sources_fail_and_say_so). The pin binds them under those names at module level, where pytest collects
    them; `script` is the pin's own path, which the last test copies into a tree without the sources."""
    import pytest  # here, not at the top: the script-style gates import this module and need no pytest

    def test_sources_exist():
        texts = tree_texts(files)
        assert not missing(texts, *files), missing(texts, *files)

    @pytest.mark.parametrize("clause", sorted(clauses))
    def test_clause_holds_on_the_tree(clause):
        fails = clauses[clause](tree_texts(files))
        assert not fails, "\n".join(fails)

    def test_missing_sources_fail_and_say_so():
        if missing(tree_texts(files), *files):
            # The control's own copy runs in a tree without the sources, and must not run the control again.
            pytest.skip("the pinned sources are absent here: test_sources_exist reports them")
        rc, out = run_without_source(script)
        assert rc != 0, out
        assert "not found" in out, out

    return test_sources_exist, test_clause_holds_on_the_tree, test_missing_sources_fail_and_say_so
