"""Behaviour of tests/sycl_gate.py, the helper the independent-check source gates are built on.

A helper that can pass vacuously or blame the wrong needle undermines every gate that uses it, so pin both:
zero recorded checks must fail, and a failing generator expression must be annotated without naming a stale
module-level variable that shares a comprehension variable's name. The message helpers the pytest-style pins pass
as assert messages, first_missing and first_present, are pinned here too: what each names, the singular and plural
wording, and the 100-character cut.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

from sycl_gate import first_missing, first_present

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


def test_first_missing_names_the_absent_anchor_and_where_it_looked() -> None:
    assert first_missing("W", "a", "a", "b") == "W: missing 'b'"


def test_first_missing_names_the_first_of_two_missing_anchors() -> None:
    assert first_missing("W", "zzz", "a", "b") == "W: missing 'a'"


def test_first_missing_with_every_anchor_present_says_an_order_count_or_depth_condition_failed() -> None:
    message = first_missing("W", "a b c", "a", "b", "c")
    assert message == "W: all 3 anchors present; an order, count or depth condition failed"


def test_first_missing_with_one_anchor_present_reads_in_the_singular() -> None:
    assert first_missing("W", "a", "a") == "W: the anchor is present; an order, count or depth condition failed"


def test_first_missing_shows_an_anchor_of_exactly_100_characters_whole() -> None:
    assert first_missing("W", "", "x" * 100) == "W: missing %r" % ("x" * 100)


def test_first_missing_cuts_an_anchor_of_101_characters_to_97_and_dots() -> None:
    assert first_missing("W", "", "x" * 101) == "W: missing %r" % ("x" * 97 + "...")
    assert len(first_missing("W", "", "x" * 300)) < 160


def test_first_missing_is_never_empty() -> None:
    assert first_missing("W", "")
    assert first_missing("W", "t")
    assert first_missing("", "", "a")


def test_first_missing_with_a_signature_and_empty_text_names_the_signature() -> None:
    assert first_missing("W", "", "a", "b", signature="void f()") == "W: missing 'void f()'"


def test_first_missing_with_a_signature_and_text_behaves_as_without_it() -> None:
    assert first_missing("W", "a", "a", "b", signature="void f()") == first_missing("W", "a", "a", "b")
    assert first_missing("W", "a b", "a", "b", signature="void f()") == first_missing("W", "a b", "a", "b")


def test_first_missing_shows_a_signature_of_exactly_100_characters_whole() -> None:
    assert first_missing("W", "", "a", signature="x" * 100) == "W: missing %r" % ("x" * 100)


def test_first_missing_cuts_a_signature_of_101_characters_to_97_and_dots() -> None:
    assert first_missing("W", "", "a", signature="x" * 101) == "W: missing %r" % ("x" * 97 + "...")


def test_first_present_names_the_forbidden_text_and_where_it_looked() -> None:
    assert first_present("W", "x bad y", "good", "bad") == "W: found 'bad'"


def test_first_present_names_the_first_of_two_forbidden_texts() -> None:
    assert first_present("W", "b a", "a", "b") == "W: found 'a'"


def test_first_present_with_none_present_names_how_many_were_searched() -> None:
    message = first_present("W", "xyz", "a", "b", "c")
    assert message == "W: none of 3 forbidden texts present; an emptiness or count condition failed"


def test_first_present_with_one_forbidden_text_reads_in_the_singular() -> None:
    assert first_present("W", "xyz", "a") == "W: the forbidden text is absent; an emptiness or count condition failed"


def test_first_present_shows_a_text_of_exactly_100_characters_whole() -> None:
    assert first_present("W", "y" * 100, "y" * 100) == "W: found %r" % ("y" * 100)


def test_first_present_cuts_a_text_of_101_characters_to_97_and_dots() -> None:
    assert first_present("W", "y" * 101, "y" * 101) == "W: found %r" % ("y" * 97 + "...")


def test_first_present_is_never_empty() -> None:
    assert first_present("W", "")
    assert first_present("W", "t", "t")
    assert first_present("", "t", "t")


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
