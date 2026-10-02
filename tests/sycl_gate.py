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
already independent, one result per function.
"""
import contextlib
import linecache
import re
import sys
import traceback

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

    Names bound by a comprehension inside that statement (`all(needle in text for needle in ...)`) are NOT
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
