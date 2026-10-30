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
already independent, one result per function. This module serves them two kinds of helper. The message helpers:
a claim that returns a bare bool says nothing about WHICH token failed, so pass
first_missing (names the first anchor the claim needs that its region lacks) or
first_present (names the first forbidden text its region holds) as the message.

    from sycl_gate import first_missing

    assert claim_x(SYCL), first_missing("file: region", region, ANCHOR_A, ANCHOR_B)

And the text helpers the MoE source-contract pins share, at the end of the module: norm (comment-stripped,
whitespace-canonical text), read (a repo file's text, or empty when it is missing), body (the brace-balanced body of a
definition), ordered (needles in order) and mutate (a normalised text with one replacement, for positive controls).
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


# ---- text helpers shared by the source-contract pins (exec-window, staged-ring-route, stream-enqueue) -------------------

# The repo root, resolved the way the pins resolved it from their own path: this file sits in tests/, so parents[1] of
# it is the same directory as parents[1] of any pin beside it.
_REPO_ROOT = Path(__file__).resolve().parents[1]

_LEXEME_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'",
    re.DOTALL,
)


def strip_comments(src: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0] in "\"'":
            return tok
        return "\n" * tok.count("\n")

    return _LEXEME_RE.sub(repl, src)


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
    """The file's text, or empty when it does not exist yet, so a missing file fails the claims instead of erroring."""
    p = _REPO_ROOT / rel
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
