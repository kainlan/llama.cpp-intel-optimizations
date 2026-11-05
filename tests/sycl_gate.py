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

Second, the text helpers for source-contract pins, at the end of the module:
ROOT (the repo root), strip_comments (source text with comments blanked, line
breaks kept), norm (comment-stripped, whitespace-canonical text), read (a repo
file's text, or empty when it is missing), body (the brace-balanced body of a
definition), ordered (needles in order), mutate (a normalised text with one
replacement, for positive controls) and resolve_mutants (source text with every
`#if` over a MUTANT macro resolved as the production build compiles it).
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
