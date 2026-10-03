#!/usr/bin/env python3
"""Gate for scripts/parse-qwen4exp-mtp-acceptance.py.

The parser turns a llama-speculative-simple log into the numbers the MTP
feasibility measurement (llama.cpp-0rhb) is about: draft acceptance and mean
tokens per round.  The failure that matters is a run where speculation was
never active -- it still prints a summary, with n_drafted = 0 -- so the tests
carry a negative control for exactly that.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-qwen4exp-mtp-acceptance.py"

STATS_LINE = (
    "0.31.337.771 T spec    statistics: statistics       draft-mtp: #calls(b,g,a) =    1    90     90, "
    "#gen drafts =     90, #acc drafts =    80, #gen tokens =    270, #acc tokens =   170, "
    "#mean acc len = 2.89, #acc rate/pos = (0.889, 0.600, 0.400)"
)


def log_text(n_draft: int = 3, n_predict: int = 260, n_drafted: int = 270, n_accept: int = 170,
             stats: bool = True) -> str:
    lines = [
        "0.00.001.000 I common_init_from_params: warming up the model with an empty run",
        "encoded   81 tokens in    3.210 seconds, speed:   25.234 t/s",
        "decoded  %d tokens in   41.000 seconds, speed:    6.341 t/s" % n_predict,
        "",
        "n_draft   = %d" % n_draft,
        "n_predict = %d" % n_predict,
        "n_drafted = %d" % n_drafted,
        "n_accept  = %d" % n_accept,
        "accept    = %.3f%%" % (100.0 * n_accept / n_drafted if n_drafted else float("nan")),
        "",
        "draft:",
        "",
    ]
    if stats:
        lines.append(STATS_LINE)
    lines += ["", "target:", ""]
    return "\n".join(lines) + "\n"


def run(*paths: pathlib.Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PARSER), *[str(p) for p in paths], *(extra or [])],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def test_parses_the_summary_and_the_per_position_stats() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q8_n3_code.log"
        log.write_text(log_text(), encoding="utf-8")
        result = run(log, extra=["--json"])
        assert result.returncode == 0, result.stdout
        rec = json.loads(result.stdout)["arms"][0]
        assert rec["arm"] == "q8_n3_code"
        assert rec["n_draft"] == 3
        assert rec["n_predict"] == 260
        assert rec["n_drafted"] == 270
        assert rec["n_accept"] == 170
        assert abs(rec["accept_rate"] - 170 / 270) < 1e-9
        assert rec["mean_len"] == 2.89
        assert rec["mean_len_source"] == "stats"
        assert rec["acc_rate_per_pos"] == [0.889, 0.6, 0.4]
        assert abs(rec["decode_tps"] - 6.341) < 1e-9


def test_without_the_stats_line_the_mean_length_is_derived_and_labelled() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q4_n3_chat.log"
        log.write_text(log_text(stats=False), encoding="utf-8")
        result = run(log, extra=["--json"])
        assert result.returncode == 0, result.stdout
        rec = json.loads(result.stdout)["arms"][0]
        # rounds = n_drafted / n_draft = 90; tokens per round = n_predict / rounds
        assert rec["mean_len_source"] == "derived"
        assert abs(rec["mean_len"] - 260 / 90) < 1e-9
        assert rec["acc_rate_per_pos"] is None


def test_a_run_that_never_drafted_is_rejected_not_reported_as_zero_acceptance() -> None:
    # negative control: speculation off still prints a summary, with n_drafted = 0
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q8_n3_code.log"
        log.write_text(log_text(n_drafted=0, n_accept=0, stats=False), encoding="utf-8")
        result = run(log)
        assert result.returncode == 2
        assert "no draft tokens were generated" in result.stdout
        assert "Traceback" not in result.stdout


def test_a_log_without_the_summary_is_rejected_without_a_traceback() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q8_n2_reasoning.log"
        log.write_text("0.00.001.000 E llama_model_load: error loading model\n", encoding="utf-8")
        result = run(log)
        assert result.returncode == 2
        assert "did not reach the speculative summary" in result.stdout
        assert "Traceback" not in result.stdout


def test_one_bad_arm_fails_the_whole_run_but_the_good_arms_are_still_listed() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        good = tmp / "q8_n2_code.log"
        bad = tmp / "q8_n2_chat.log"
        good.write_text(log_text(n_draft=2, n_drafted=180, n_accept=120, n_predict=250), encoding="utf-8")
        bad.write_text(log_text(n_drafted=0, n_accept=0, stats=False), encoding="utf-8")
        result = run(good, bad)
        assert result.returncode == 2
        assert "q8_n2_code" in result.stdout
        assert "q8_n2_chat" in result.stdout


def test_table_aggregates_per_head_and_n_max_over_the_prompts() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        paths = []
        for prompt, (drafted, accept) in {"code": (200, 150), "chat": (100, 50), "reasoning": (100, 50)}.items():
            p = tmp / ("q8_n2_%s.log" % prompt)
            p.write_text(log_text(n_draft=2, n_drafted=drafted, n_accept=accept, n_predict=256), encoding="utf-8")
            paths.append(p)
        result = run(*paths, extra=["--json"])
        assert result.returncode == 0, result.stdout
        agg = json.loads(result.stdout)["groups"]
        assert len(agg) == 1
        g = agg[0]
        assert g["head"] == "q8" and g["n_draft"] == 2 and g["arms"] == 3
        # token-weighted: (150 + 50 + 50) / (200 + 100 + 100)
        assert abs(g["accept_rate"] - 250 / 400) < 1e-9


def test_text_table_names_every_arm() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q4_n3_code.log"
        log.write_text(log_text(), encoding="utf-8")
        result = run(log)
        assert result.returncode == 0, result.stdout
        assert "q4_n3_code" in result.stdout
        assert "62.96" in result.stdout or "63.0" in result.stdout  # 170/270 as a percentage


def test_p_min_arms_are_kept_apart_from_the_unfiltered_arms_of_the_same_head_and_n_max() -> None:
    # same head and n-max, different draft filter: rolling them up together would hide the filter's effect
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        a = tmp / "q8_n3_code.log"
        b = tmp / "q8_n3_p05_code.log"
        a.write_text(log_text(n_draft=3, n_drafted=300, n_accept=150), encoding="utf-8")
        b.write_text(log_text(n_draft=3, n_drafted=200, n_accept=160), encoding="utf-8")
        result = run(a, b, extra=["--json"])
        assert result.returncode == 0, result.stdout
        groups = json.loads(result.stdout)["groups"]
        assert len(groups) == 2
        by_tag = {g["p_min"]: g for g in groups}
        assert set(by_tag) == {"none", "p05"}
        assert abs(by_tag["none"]["accept_rate"] - 0.5) < 1e-9
        assert abs(by_tag["p05"]["accept_rate"] - 0.8) < 1e-9


def test_a_p_min_arm_without_the_stats_line_is_rejected_not_given_a_derived_length() -> None:
    # the derived length assumes every round drafts n_draft tokens, which a p-min filter breaks
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q4_n3_p05_chat.log"
        log.write_text(log_text(stats=False), encoding="utf-8")
        result = run(log)
        assert result.returncode == 2
        assert "needs the '#mean acc len' statistics line" in result.stdout
        assert "Traceback" not in result.stdout
