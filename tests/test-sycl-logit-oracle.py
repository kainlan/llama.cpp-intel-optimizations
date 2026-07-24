"""Unit tests for scripts/sycl-logit-oracle.py (pure python, no GPU, no numpy).

Run with the system interpreter (the conda python has a broken numpy):

    /usr/bin/python3 -m pytest tests/test-sycl-logit-oracle.py -v
"""

import argparse
import array
import json
import pathlib
import struct
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


def test_top_token_ranking_mismatch_is_a_failure():
    a = [{"token": 5, "top_tokens": [5, 7], "logits": [1.0, 0.5]}]
    b = [{"token": 5, "top_tokens": [5, 9], "logits": [1.0, 0.5]}]
    r = oracle.compare(a, b, tol=1e-6)
    assert not r.ok and r.first_divergent_index == 0
    assert "top-k" in r.reason


def test_matching_top_tokens_do_not_diverge():
    a = [{"token": 5, "top_tokens": [5, 7], "logits": [1.0, 0.5]}]
    assert oracle.compare(a, [dict(a[0])], tol=1e-6).ok


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


# --- llama-perplexity .kld binary parsing ------------------------------------
#
# The fixture builder below mirrors the writer in tools/perplexity/perplexity.cpp
# (header at :460 and :521, rows at :94-105) so the parser is tested against the
# real layout rather than against its own assumptions.


def _kld_bytes(n_ctx, n_vocab, n_chunk, rows, tokens=None):
    nv = 2 * ((n_vocab + 1) // 2) + 4
    if tokens is None:
        tokens = list(range(n_chunk * n_ctx))
    buf = bytearray(b"_logits_")
    buf += struct.pack("<Iii", n_ctx, n_vocab, n_chunk)
    buf += struct.pack(f"<{len(tokens)}i", *tokens)
    for scale, min_log_prob, quantized in rows:
        padded = list(quantized) + [0] * (nv - 4 - len(quantized))
        buf += struct.pack("<2f", scale, min_log_prob)
        buf += array.array("H", padded).tobytes()
    return bytes(buf)


def _write_kld(tmpdir, name, n_ctx, n_vocab, n_chunk, rows, tokens=None):
    path = pathlib.Path(tmpdir) / name
    path.write_bytes(_kld_bytes(n_ctx, n_vocab, n_chunk, rows, tokens))
    return path


# n_ctx=6 -> first=3, rows_per_chunk = 6-1-3 = 2; one chunk, so two scored rows.
FIXTURE_ROWS = [
    (0.001, -4.0, [65535, 100, 0, 7]),
    (0.002, -8.0, [10, 65535, 20, 30]),
]


def _fixture(tmpdir, name="capture.kld"):
    return _write_kld(tmpdir, name, n_ctx=6, n_vocab=4, n_chunk=1, rows=FIXTURE_ROWS)


def test_read_kld_header_parses_the_real_layout():
    with tempfile.TemporaryDirectory() as tmp:
        with open(_fixture(tmp), "rb") as fh:
            header = oracle.read_kld_header(fh)
        assert (header.n_ctx, header.n_vocab, header.n_chunk) == (6, 4, 1)
        assert header.tokens == list(range(6))


def test_kld_header_derived_geometry():
    header = oracle.KldHeader(n_ctx=6, n_vocab=4, n_chunk=3, tokens=[])
    assert header.nv == 2 * ((4 + 1) // 2) + 4 == 8
    assert header.rows_per_chunk == 6 - 1 - 3
    assert header.n_rows == 3 * 2


def test_kld_header_pads_an_odd_vocabulary_to_even():
    # nv rounds the vocabulary up to an even count and adds the 4 header slots.
    assert oracle.KldHeader(n_ctx=4, n_vocab=3, n_chunk=1, tokens=[]).nv == 8
    assert oracle.KldHeader(n_ctx=4, n_vocab=5, n_chunk=1, tokens=[]).nv == 10


def test_read_kld_header_rejects_a_bad_magic():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "bad.kld"
        path.write_bytes(b"_wrong__" + struct.pack("<Iii", 6, 4, 1))
        with open(path, "rb") as fh:
            try:
                oracle.read_kld_header(fh)
            except ValueError as exc:
                assert "magic" in str(exc)
                return
        raise AssertionError("expected ValueError for a bad magic")


def test_read_kld_header_rejects_a_truncated_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "short.kld"
        path.write_bytes(b"_logits_" + struct.pack("<I", 6))
        with open(path, "rb") as fh:
            try:
                oracle.read_kld_header(fh)
            except ValueError:
                return
        raise AssertionError("expected ValueError for a truncated header")


def test_read_kld_header_rejects_a_short_token_array():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "short-tokens.kld"
        path.write_bytes(b"_logits_" + struct.pack("<Iii", 6, 4, 1) + struct.pack("<2i", 0, 1))
        with open(path, "rb") as fh:
            try:
                oracle.read_kld_header(fh)
            except ValueError as exc:
                assert "token" in str(exc)
                return
        raise AssertionError("expected ValueError for a short token array")


def test_iter_kld_rows_rejects_a_truncated_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "cut.kld"
        path.write_bytes(_kld_bytes(6, 4, 1, FIXTURE_ROWS)[:-6])
        try:
            list(oracle.iter_kld_rows(path))
        except ValueError as exc:
            assert "truncated" in str(exc)
            return
        raise AssertionError("expected ValueError for a truncated row")


def test_decode_kld_row_splits_scale_min_and_quantized_values():
    raw = struct.pack("<2f", 0.5, -3.0) + array.array("H", [1, 2, 3, 4]).tobytes()
    scale, min_log_prob, quantized = oracle.decode_kld_row(raw, n_vocab=4)
    assert abs(scale - 0.5) < 1e-9 and abs(min_log_prob + 3.0) < 1e-6
    assert list(quantized) == [1, 2, 3, 4]


def test_decode_kld_row_ignores_the_even_padding_slot():
    # n_vocab=3 pads to 4 uint16; only the first 3 belong to the vocabulary.
    raw = struct.pack("<2f", 0.5, -3.0) + array.array("H", [1, 2, 3, 999]).tobytes()
    _, _, quantized = oracle.decode_kld_row(raw, n_vocab=3)
    assert list(quantized) == [1, 2, 3]


def test_dequantize_kld_row_applies_the_writer_formula():
    # log_prob[i] = min_log_prob + scale*q[i]  (perplexity.cpp log_softmax)
    values = oracle.dequantize_kld_row(0.001, -4.0, array.array("H", [0, 1000, 65535]))
    assert abs(values[0] + 4.0) < 1e-9
    assert abs(values[1] - (-4.0 + 1.0)) < 1e-9
    assert abs(values[2] - (-4.0 + 65.535)) < 1e-6


def test_top_k_from_kld_row_ranks_by_log_probability():
    tokens, logits = oracle.top_k_from_kld_row(0.001, -4.0, array.array("H", [10, 65535, 20, 30]), k=2)
    assert tokens == [1, 3]
    assert abs(logits[0] - (-4.0 + 65.535)) < 1e-6
    assert abs(logits[1] - (-4.0 + 0.030)) < 1e-9


def test_top_k_from_kld_row_breaks_ties_towards_the_lower_token_id():
    tokens, _ = oracle.top_k_from_kld_row(1.0, 0.0, array.array("H", [7, 7, 7]), k=2)
    assert tokens == [0, 1]


def test_top_k_from_kld_row_clamps_k_to_the_vocabulary():
    tokens, logits = oracle.top_k_from_kld_row(1.0, 0.0, array.array("H", [1, 2]), k=99)
    assert len(tokens) == 2 and len(logits) == 2


def test_entries_from_kld_reduces_each_row_to_top_k():
    with tempfile.TemporaryDirectory() as tmp:
        entries, header = oracle.entries_from_kld(_fixture(tmp), top_k=2)
        assert header.n_rows == 2 and len(entries) == 2
        assert [e["token"] for e in entries] == [0, 1]
        assert entries[0]["top_tokens"] == [0, 1]
        assert len(entries[0]["logits"]) == 2
        assert entries[0]["index"] == 0 and entries[1]["index"] == 1


def test_entries_from_kld_records_the_evaluated_target_token():
    # n_ctx=6 -> first=3, so rows 0 and 1 score tokens at positions 4 and 5.
    with tempfile.TemporaryDirectory() as tmp:
        entries, _ = oracle.entries_from_kld(_fixture(tmp), top_k=1)
        assert [e["target"] for e in entries] == [4, 5]


def test_entries_from_kld_honours_max_rows():
    with tempfile.TemporaryDirectory() as tmp:
        entries, _ = oracle.entries_from_kld(_fixture(tmp), top_k=2, max_rows=1)
        assert len(entries) == 1


def test_entries_from_kld_round_trips_through_compare():
    with tempfile.TemporaryDirectory() as tmp:
        entries, _ = oracle.entries_from_kld(_fixture(tmp), top_k=2)
        assert oracle.compare(entries, list(entries), tol=1e-9).ok


def test_entries_from_kld_detects_a_single_quantization_step():
    # One step of 0.001 must be visible at the default tolerance.
    bumped = [(0.001, -4.0, [65535, 101, 0, 7]), FIXTURE_ROWS[1]]
    with tempfile.TemporaryDirectory() as tmp:
        base, _ = oracle.entries_from_kld(_fixture(tmp), top_k=2)
        path = _write_kld(tmp, "bumped.kld", n_ctx=6, n_vocab=4, n_chunk=1, rows=bumped)
        candidate, _ = oracle.entries_from_kld(path, top_k=2)
        r = oracle.compare(base, candidate, tol=oracle.DEFAULT_TOL)
        assert not r.ok and r.first_divergent_index == 0
        assert abs(r.max_abs - 0.001) < 1e-6


# --- capture plumbing (stub binaries, no GPU and no model) -------------------


def _write_stub(tmpdir, name, body):
    path = pathlib.Path(tmpdir) / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _capture_args(tmp, completion, tokenize, **overrides):
    args = argparse.Namespace(
        mode="tokens",
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
        assert obj["mode"] == "tokens"
        assert [e["token"] for e in obj["entries"]] == [28705, 28784]
        assert obj["meta"]["seed"] == 42 and obj["meta"]["temp"] == 0.0
        assert obj["meta"]["text"] == " 6, 7, 8"


def test_capture_rejects_an_unknown_mode():
    with tempfile.TemporaryDirectory() as tmp:
        completion = _write_stub(tmp, "completion", 'printf ""\n')
        tokenize = _write_stub(tmp, "tokenize", 'cat > /dev/null; echo "[1]"\n')
        try:
            oracle.capture(_capture_args(tmp, completion, tokenize, mode="nonsense"))
        except ValueError:
            return
        raise AssertionError("expected ValueError for an unknown capture mode")


# --- logits-mode capture plumbing (stub llama-perplexity) --------------------


def _logits_capture_args(tmp, perplexity_bin, kld_out, **overrides):
    args = argparse.Namespace(
        mode="logits",
        model=str(pathlib.Path(tmp) / "model.gguf"),
        input_file=None,
        n_ctx=6,
        n_chunk=1,
        top_k=2,
        max_rows=None,
        kld_out=str(kld_out),
        seed=42,
        device="level_zero:1",
        n_gpu_layers=None,
        extra=[],
        perplexity_bin=str(perplexity_bin),
        timeout=30.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_capture_logits_parses_the_file_the_tool_wrote():
    with tempfile.TemporaryDirectory() as tmp:
        fixture = _fixture(tmp, "fixture.kld")
        kld_out = pathlib.Path(tmp) / "out.kld"
        stub = _write_stub(tmp, "perplexity", f'cp "{fixture}" "{kld_out}"\n')
        obj = oracle.capture(_logits_capture_args(tmp, stub, kld_out))
        assert obj["mode"] == "logits"
        assert [e["token"] for e in obj["entries"]] == [0, 1]
        assert obj["meta"]["n_vocab"] == 4 and obj["meta"]["n_ctx"] == 6
        assert obj["meta"]["top_k"] == 2 and obj["meta"]["seed"] == 42


def test_capture_logits_passes_the_deterministic_geometry_flags():
    with tempfile.TemporaryDirectory() as tmp:
        fixture = _fixture(tmp, "fixture.kld")
        kld_out = pathlib.Path(tmp) / "out.kld"
        argv_log = pathlib.Path(tmp) / "argv.txt"
        stub = _write_stub(tmp, "perplexity", f'printf "%s" "$*" > "{argv_log}"\ncp "{fixture}" "{kld_out}"\n')
        oracle.capture(_logits_capture_args(tmp, stub, kld_out))
        argv = argv_log.read_text(encoding="utf-8")
        assert "--kl-divergence-base" in argv
        assert "--seed 42" in argv
        # -b must equal -c so llama-perplexity keeps n_parallel at 1.
        assert "-c 6" in argv and "-b 6" in argv


def test_capture_logits_generates_a_deterministic_input_file():
    with tempfile.TemporaryDirectory() as tmp:
        fixture = _fixture(tmp, "fixture.kld")
        kld_out = pathlib.Path(tmp) / "out.kld"
        copies = [pathlib.Path(tmp) / f"input{i}.txt" for i in (0, 1)]
        texts = []
        for copy in copies:
            stub = _write_stub(tmp, "perplexity", f'cp "$4" "{copy}"\ncp "{fixture}" "{kld_out}"\n')
            oracle.capture(_logits_capture_args(tmp, stub, kld_out))
            texts.append(copy.read_text(encoding="utf-8"))
        assert texts[0] == texts[1] and len(texts[0]) > 0


def test_default_input_text_is_long_enough_for_the_chunks():
    text = oracle.default_input_text(n_ctx=64, n_chunk=2)
    assert len(text.split()) >= 4 * 64 * 2


def test_capture_logits_reports_a_missing_logits_file():
    with tempfile.TemporaryDirectory() as tmp:
        kld_out = pathlib.Path(tmp) / "never-written.kld"
        stub = _write_stub(tmp, "perplexity", "exit 0\n")
        try:
            oracle.capture(_logits_capture_args(tmp, stub, kld_out))
        except RuntimeError as exc:
            assert "2*n_ctx" in str(exc)
            return
        raise AssertionError("expected RuntimeError when no logits file is produced")


def test_capture_logits_reports_a_failing_tool():
    with tempfile.TemporaryDirectory() as tmp:
        kld_out = pathlib.Path(tmp) / "out.kld"
        stub = _write_stub(tmp, "perplexity", "echo kaboom >&2\nexit 1\n")
        try:
            oracle.capture(_logits_capture_args(tmp, stub, kld_out))
        except RuntimeError as exc:
            assert "kaboom" in str(exc)
            return
        raise AssertionError("expected RuntimeError when llama-perplexity exits non-zero")


def test_capture_logits_reports_a_missing_input_file():
    with tempfile.TemporaryDirectory() as tmp:
        kld_out = pathlib.Path(tmp) / "out.kld"
        stub = _write_stub(tmp, "perplexity", "exit 0\n")
        missing = str(pathlib.Path(tmp) / "no-such-input.txt")
        try:
            oracle.capture(_logits_capture_args(tmp, stub, kld_out, input_file=missing))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError for a missing --input-file")


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


def test_cli_compare_rejects_a_mode_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_json(tmp, "a.json", {"mode": "logits", "entries": [{"token": 1}]})
        b = _write_json(tmp, "b.json", {"mode": "tokens", "entries": [{"token": 1}]})
        proc = _run_cli("compare", str(a), str(b))
        assert proc.returncode == 2
        assert "modes differ" in (proc.stdout + proc.stderr)


def test_capture_defaults_to_logits_mode():
    args = oracle.build_parser().parse_args(["capture", "--model", "m.gguf", "--out", "o.json"])
    assert args.mode == "logits"
    assert args.n_ctx == oracle.DEFAULT_CTX and args.top_k == oracle.DEFAULT_TOP_K


def test_capture_rejects_an_unknown_mode_at_the_cli():
    proc = _run_cli("capture", "--model", "m.gguf", "--out", "o.json", "--mode", "nonsense")
    assert proc.returncode != 0
    assert "nonsense" in proc.stderr


def test_cli_compare_honours_tolerance():
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_json(tmp, "a.json", [{"token": 1, "logits": [1.0]}])
        b = _write_json(tmp, "b.json", [{"token": 1, "logits": [1.4]}])
        assert _run_cli("compare", str(a), str(b), "--tol", "0.5").returncode == 0
        assert _run_cli("compare", str(a), str(b), "--tol", "0.1").returncode != 0
