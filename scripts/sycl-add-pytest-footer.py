#!/usr/bin/env python3
"""Append the pytest footer to every pytest-style tests/test-sycl-*.py gate that cannot run itself.

A pytest-style gate (module-level `def test_...`, no `__main__` block) run as `python3 tests/test-sycl-x.py`
defines its functions and exits 0 having asserted nothing: the vacuous pass that made a python3 loop read a
red tree green (llama.cpp-qeld). The footer makes a direct run execute the file's tests under pytest and exit
with pytest's status instead.

    python3 scripts/sycl-add-pytest-footer.py            # append the footer where it is missing
    python3 scripts/sycl-add-pytest-footer.py --check    # list the files that need it, exit 1 if any

Idempotent, and it only ever appends: a gate that already has its own `__main__` block (a manual runner, or its
own pytest.main call) is left alone, since a second block would run after the first. The footer text is the one
tests/test-sycl-gate-registration-audit.py (R3) recognises.
"""
import re
import sys
from pathlib import Path

FOOTER = '\nif __name__ == "__main__":\n    import sys\n\n    import pytest\n\n    sys.exit(pytest.main([__file__, "-q"]))\n'


def needs_footer(text):
    return re.search(r"^(?:async )?def test_", text, re.M) is not None and "__main__" not in text


def main(argv):
    check = "--check" in argv
    root = Path(__file__).resolve().parents[1]
    pending = []
    for path in sorted((root / "tests").glob("test-sycl-*.py")):
        text = path.read_text()
        if not needs_footer(text):
            continue
        pending.append(path.name)
        if not check:
            path.write_text(text.rstrip("\n") + "\n" + FOOTER)
    for name in pending:
        print(("needs footer: " if check else "appended footer: ") + name)
    print("%d file(s) %s" % (len(pending), "need the footer" if check else "updated"))
    return 1 if (check and pending) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
