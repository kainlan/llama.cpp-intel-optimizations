"""Unit tests for scripts/sycl-logit-oracle.py (pure python, no GPU, no numpy).

Run with the system interpreter (the conda python has a broken numpy):

    /usr/bin/python3 -m pytest tests/test-sycl-logit-oracle.py -v
"""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sycl-logit-oracle.py"

# The script has a dash in its name, so it cannot be a normal import.
oracle = SourceFileLoader("oracle", str(SCRIPT)).load_module()


# --- comparison core (spec tests) ------------------------------------------


def test_identical_captures_match():
    a = [{"token": 5, "logits": [1.0, 2.0, 3.0]}]
    assert oracle.compare(a, a, tol=1e-6).ok


def test_divergence_detected_with_index():
    a = [{"token": 5, "logits": [1.0, 2.0, 3.0]}]
    b = [{"token": 5, "logits": [1.0, 2.0, 3.5]}]
    r = oracle.compare(a, b, tol=1e-6)
    assert not r.ok and r.first_divergent_index == 0 and r.max_abs > 0.4


def test_token_mismatch_is_failure():
    a = [{"token": 5, "logits": [1.0]}]
    b = [{"token": 6, "logits": [1.0]}]
    assert not oracle.compare(a, b, tol=1e-6).ok


# --- comparison core (edge cases) ------------------------------------------


def test_result_fields_present():
    r = oracle.compare([{"token": 1}], [{"token": 1}], tol=1e-6)
    for field in ("ok", "max_abs", "max_rel", "first_divergent_index", "reason"):
        assert hasattr(r, field), field


def test_token_only_entries_compare():
    a = [{"token": 1}, {"token": 2}, {"token": 3}]
    assert oracle.compare(a, list(a), tol=0.0).ok


def test_token_only_divergence_reports_index():
    a = [{"token": 1}, {"token": 2}, {"token": 3}]
    b = [{"token": 1}, {"token": 2}, {"token": 9}]
    r = oracle.compare(a, b, tol=0.0)
    assert not r.ok and r.first_divergent_index == 2


def test_difference_within_tolerance_passes():
    a = [{"token": 5, "logits": [1.0, 2.0]}]
    b = [{"token": 5, "logits": [1.0, 2.0 + 1e-9]}]
    assert oracle.compare(a, b, tol=1e-6).ok


def test_max_abs_and_rel_reported_even_when_ok():
    a = [{"token": 5, "logits": [10.0]}]
    b = [{"token": 5, "logits": [10.5]}]
    r = oracle.compare(a, b, tol=1.0)
    assert r.ok
    assert abs(r.max_abs - 0.5) < 1e-9
    assert abs(r.max_rel - 0.5 / 10.5) < 1e-9


def test_zero_magnitude_logits_do_not_divide_by_zero():
    a = [{"token": 5, "logits": [0.0]}]
    b = [{"token": 5, "logits": [0.0]}]
    r = oracle.compare(a, b, tol=1e-6)
    assert r.ok and r.max_rel == 0.0


def test_length_mismatch_is_failure_at_first_extra_index():
    a = [{"token": 1}, {"token": 2}]
    b = [{"token": 1}]
    r = oracle.compare(a, b, tol=1e-6)
    assert not r.ok and r.first_divergent_index == 1
    assert "length" in r.reason.lower()


def test_logit_vector_length_mismatch_is_failure():
    a = [{"token": 5, "logits": [1.0, 2.0]}]
    b = [{"token": 5, "logits": [1.0]}]
    r = oracle.compare(a, b, tol=1e-6)
    assert not r.ok and r.first_divergent_index == 0


def test_empty_captures_are_a_failure():
    r = oracle.compare([], [], tol=1e-6)
    assert not r.ok and "empty" in r.reason.lower()


def test_first_divergent_index_is_the_earliest_one():
    a = [{"token": 1, "logits": [1.0]}, {"token": 2, "logits": [2.0]}, {"token": 3, "logits": [3.0]}]
    b = [{"token": 1, "logits": [1.0]}, {"token": 2, "logits": [9.0]}, {"token": 3, "logits": [9.0]}]
    r = oracle.compare(a, b, tol=1e-6)
    assert not r.ok and r.first_divergent_index == 1


def test_reason_is_empty_when_ok():
    r = oracle.compare([{"token": 1}], [{"token": 1}], tol=1e-6)
    assert r.reason == ""


# --- capture file loading ---------------------------------------------------


def _write_json(tmpdir, name, obj):
    path = pathlib.Path(tmpdir) / name
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_capture_accepts_bare_list():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_json(tmp, "a.json", [{"token": 1}])
        assert oracle.load_capture(path) == [{"token": 1}]


def test_load_capture_accepts_object_with_entries():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_json(tmp, "a.json", {"mode": "token", "entries": [{"token": 7}]})
        assert oracle.load_capture(path) == [{"token": 7}]


def test_load_capture_rejects_unknown_shape():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_json(tmp, "a.json", {"nope": 1})
        try:
            oracle.load_capture(path)
        except ValueError:
            return
        raise AssertionError("expected ValueError for an unrecognized capture shape")


def test_entries_from_token_ids_builds_entries():
    entries = oracle.entries_from_token_ids([11, 22], pieces=["a", "b"])
    assert [e["token"] for e in entries] == [11, 22]
    assert [e["piece"] for e in entries] == ["a", "b"]


def test_parse_token_ids_reads_llama_tokenize_output():
    assert oracle.parse_token_ids("[1, 2, 3]\n") == [1, 2, 3]


def test_parse_token_ids_ignores_surrounding_log_noise():
    text = "load: some log line\n[10, 20]\nllama_perf: done\n"
    assert oracle.parse_token_ids(text) == [10, 20]


def test_parse_token_ids_rejects_output_without_a_list():
    try:
        oracle.parse_token_ids("no ids here\n")
    except ValueError:
        return
    raise AssertionError("expected ValueError when no token id list is present")


# --- capture plumbing (stub binaries, no GPU and no model) -------------------


def _write_stub(tmpdir, name, body):
    path = pathlib.Path(tmpdir) / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _capture_args(tmp, completion, tokenize, **overrides):
    args = argparse.Namespace(
        model=str(pathlib.Path(tmp) / "model.gguf"),
        prompt="1, 2, 3, 4, 5,",
        n_predict=4,
        seed=42,
        device="level_zero:1",
        n_gpu_layers=None,
        extra=[],
        completion_bin=str(completion),
        tokenize_bin=str(tokenize),
        timeout=30.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_capture_builds_entries_from_stub_binaries():
    with tempfile.TemporaryDirectory() as tmp:
        completion = _write_stub(tmp, "completion", 'printf " 6, 7, 8"\n')
        tokenize = _write_stub(tmp, "tokenize", 'cat > /dev/null; echo "[28705, 28784]"\n')
        obj = oracle.capture(_capture_args(tmp, completion, tokenize))
        assert obj["mode"] == "token"
        assert [e["token"] for e in obj["entries"]] == [28705, 28784]
        assert obj["meta"]["seed"] == 42 and obj["meta"]["temp"] == 0.0
        assert obj["meta"]["text"] == " 6, 7, 8"


def test_capture_forwards_the_device_selector_to_the_completion_binary():
    with tempfile.TemporaryDirectory() as tmp:
        completion = _write_stub(tmp, "completion", 'printf "%s" "$ONEAPI_DEVICE_SELECTOR"\n')
        tokenize = _write_stub(tmp, "tokenize", 'cat > /dev/null; echo "[1]"\n')
        obj = oracle.capture(_capture_args(tmp, completion, tokenize, device="level_zero:0"))
        assert obj["meta"]["text"] == "level_zero:0"


def test_capture_passes_deterministic_sampling_flags():
    with tempfile.TemporaryDirectory() as tmp:
        completion = _write_stub(tmp, "completion", 'printf "%s" "$*"\n')
        tokenize = _write_stub(tmp, "tokenize", 'cat > /dev/null; echo "[1]"\n')
        argv = oracle.capture(_capture_args(tmp, completion, tokenize))["meta"]["text"]
        assert "--temp 0" in argv and "--seed 42" in argv


def test_capture_raises_when_the_completion_binary_fails():
    with tempfile.TemporaryDirectory() as tmp:
        completion = _write_stub(tmp, "completion", "echo boom >&2\nexit 3\n")
        tokenize = _write_stub(tmp, "tokenize", 'cat > /dev/null; echo "[1]"\n')
        try:
            oracle.capture(_capture_args(tmp, completion, tokenize))
        except RuntimeError as exc:
            assert "boom" in str(exc)
            return
        raise AssertionError("expected RuntimeError when llama-completion exits non-zero")


def test_capture_reports_a_missing_binary():
    with tempfile.TemporaryDirectory() as tmp:
        tokenize = _write_stub(tmp, "tokenize", 'cat > /dev/null; echo "[1]"\n')
        missing = pathlib.Path(tmp) / "does-not-exist"
        try:
            oracle.capture(_capture_args(tmp, missing, tokenize))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError for a missing llama-completion")


# --- CLI --------------------------------------------------------------------


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_compare_exits_zero_on_match():
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_json(tmp, "a.json", [{"token": 1, "logits": [1.0]}])
        b = _write_json(tmp, "b.json", [{"token": 1, "logits": [1.0]}])
        proc = _run_cli("compare", str(a), str(b))
        assert proc.returncode == 0, proc.stderr


def test_cli_compare_exits_nonzero_on_divergence():
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_json(tmp, "a.json", [{"token": 1, "logits": [1.0]}])
        b = _write_json(tmp, "b.json", [{"token": 1, "logits": [2.0]}])
        proc = _run_cli("compare", str(a), str(b))
        assert proc.returncode != 0
        assert "diverge" in (proc.stdout + proc.stderr).lower()


def test_cli_compare_honours_tolerance():
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_json(tmp, "a.json", [{"token": 1, "logits": [1.0]}])
        b = _write_json(tmp, "b.json", [{"token": 1, "logits": [1.4]}])
        assert _run_cli("compare", str(a), str(b), "--tol", "0.5").returncode == 0
        assert _run_cli("compare", str(a), str(b), "--tol", "0.1").returncode != 0
