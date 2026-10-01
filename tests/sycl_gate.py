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
    short str/int/bool names the failing line mentions."""
    tb = error.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    line = traceback.extract_tb(error.__traceback__)[-1].line or ""
    scope = dict(tb.tb_frame.f_globals)
    scope.update(tb.tb_frame.f_locals)
    shown = []
    for word in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", line)):
        value = scope.get(word)
        if isinstance(value, (str, int, bool)) and len(repr(value)) <= 120 and word not in ("True", "False"):
            shown.append("%s=%r" % (word, value))
    return "  [%s]" % ", ".join(shown) if shown else ""


def failures() -> list:
    return [(name, why) for name, why in _RESULTS if why is not None]


def finish(pass_message: str = "") -> None:
    """Report every failed check, then exit non-zero if there was one."""
    failed = failures()
    for name, why in failed:
        print("FAIL %s\n     %s" % (name, why.replace("\n", "\n     ")), file=sys.stderr)
    print("%d/%d checks passed" % (len(_RESULTS) - len(failed), len(_RESULTS)))
    if failed:
        sys.exit(1)
    if pass_message:
        print(pass_message)
