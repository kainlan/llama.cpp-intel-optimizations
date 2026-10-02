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


def test_a_non_assertion_failure_is_recorded_and_the_next_check_still_runs() -> None:
    """A missing anchor surfaces as ValueError (str.index), not AssertionError: that must fail its own check and
    leave the checks after it running, or one stale pin hides the rest again."""
    result = run("""
        from sycl_gate import gate, finish
        with gate("first block raises ValueError"):
            "abc".index("zzz")
        with gate("second block still runs"):
            assert 1 + 1 == 3
        with gate("third block passes"):
            assert True
        finish("PASS", min_checks=3)
    """)
    assert result.returncode == 1, result.stdout
    assert "FAIL first block raises ValueError" in result.stderr
    assert "ValueError" in result.stderr
    assert "FAIL second block still runs" in result.stderr
    assert "1/3 checks passed" in result.stdout
    assert "PASS" not in result.stdout.replace("checks passed", "")


def test_run_without_source_runs_a_copy_in_a_tree_that_lacks_the_source(tmp_path) -> None:
    """The missing-source arm of the gates that pin a source file: a copy of the gate, with sycl_gate beside it, run
    where ../ggml does not exist; the status and the output come back so the caller can check both."""
    script = tmp_path / "gate.py"
    script.write_text("import pathlib\nimport sys\n\nimport sycl_gate\n\nroot = pathlib.Path(__file__).resolve().parent.parent\n"
                      "print('SOURCE_PRESENT=%s' % (root / 'ggml').exists())\nprint('not found: x', file=sys.stderr)\nsys.exit(3)\n")
    (tmp_path / "ggml").mkdir()  # beside the ORIGINAL gate; the copy must not see it
    result = run("""
        from sycl_gate import run_without_source
        status, output = run_without_source(%r)
        print(status)
        print(output)
    """ % str(script))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "3", result.stdout
    assert "SOURCE_PRESENT=False" in result.stdout
    assert "not found: x" in result.stdout


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


def test_lambda_parameter_never_blames_a_stale_global() -> None:
    """A lambda's parameter is bound inside the lambda, so the module-level name it shadows must not be reported
    (the assert fails in the caller's frame, where the global of the same name still holds its old value)."""
    result = run("""
        from sycl_gate import gate, finish
        text = "abc"
        needle = "STALE-LAMBDA-NEEDLE"
        flag = "STALE-LAMBDA-FLAG"
        with gate("a lambda check"):
            assert (lambda needle, *, flag=0: needle in text)("zzz")
        finish()
    """)
    assert result.returncode == 1
    assert "STALE-LAMBDA-NEEDLE" not in result.stderr
    assert "STALE-LAMBDA-FLAG" not in result.stderr
    assert "text='abc'" in result.stderr


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
