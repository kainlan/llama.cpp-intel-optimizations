"""Behaviour of tests/sycl_gate.py, the helper the independent-check source gates are built on.

A helper that can pass vacuously or blame the wrong needle undermines every gate that uses it, so pin both:
zero recorded checks must fail, and a failing generator expression must be annotated without naming a stale
module-level variable that shares a comprehension variable's name.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def run(script: str):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=TESTS, capture_output=True,
                          text=True, timeout=60)


def test_finish_fails_when_no_check_ran() -> None:
    result = run("""
        from sycl_gate import finish
        finish("PASS")
    """)
    assert result.returncode == 1, result.stdout
    assert "too few checks ran" in result.stderr
    assert "PASS" not in result.stdout


def test_finish_honours_min_checks() -> None:
    script = """
        from sycl_gate import gate, finish
        with gate("one"):
            assert True
        finish("PASS", min_checks=%d)
    """
    assert run(script % 1).returncode == 0
    assert run(script % 2).returncode == 1


def test_generator_failure_never_blames_a_stale_global() -> None:
    result = run("""
        from sycl_gate import gate, finish
        text = "abc"
        needle = "STALE-MODULE-LEVEL-NEEDLE"
        with gate("every needle is present"):
            assert all(needle in text for needle in ("a", "zzz"))
        finish()
    """)
    assert result.returncode == 1
    assert "STALE-MODULE-LEVEL-NEEDLE" not in result.stderr
    # The comprehension-bound name is left unannotated, but the failing statement and the real operand named
    # in it are still reported.
    assert "text='abc'" in result.stderr
    assert "needle=" not in result.stderr


def test_plain_frame_failure_still_names_its_operands() -> None:
    result = run("""
        from sycl_gate import gate, finish
        def check():
            token = "missing"
            assert token in "abc"
        with gate("a plain check"):
            check()
        finish()
    """)
    assert result.returncode == 1
    assert "token='missing'" in result.stderr

if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
