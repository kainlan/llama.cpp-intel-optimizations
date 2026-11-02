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


# Verbatim from the first real run (q4_n3_code, IQ3_XXS target, Q4_0 head, -lv 4): at that
# verbosity every summary line carries a "<timestamp> I " log prefix, which the hand-written
# fixture above does not have.  A parser that only accepts bare lines passes the fixture and
# rejects every real log.
REAL_PREFIXED_LOG = """\
20.06.639.658 I encoded   58 tokens in   29.268 seconds, speed:    1.982 t/s
20.06.639.660 I decoded  260 tokens in 1153.960 seconds, speed:    0.225 t/s
20.06.639.660 I 
20.06.639.660 I n_draft   = 3
20.06.639.660 I n_predict = 260
20.06.639.660 I n_drafted = 211
20.06.639.660 I n_accept  = 189
20.06.639.661 I accept    = 89.573%
20.06.639.661 I 
20.06.639.661 I draft:

20.06.639.712 I spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     71     71, #gen drafts =     71, #acc drafts =    68, #gen tokens =    211, #acc tokens =   189, #mean acc len = 3.66, #acc rate/pos = (0.958, 0.887, 0.817), dur(b,g,a) = 0.010, 11291.186, 0.491 ms
"""


def test_parses_a_log_whose_lines_carry_the_timestamp_log_prefix() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        log = pathlib.Path(tmp_raw) / "q4_n3_code.log"
        log.write_text(REAL_PREFIXED_LOG, encoding="utf-8")
        result = run(log, extra=["--json"])
        assert result.returncode == 0, result.stdout
        rec = json.loads(result.stdout)["arms"][0]
        assert (rec["n_draft"], rec["n_predict"], rec["n_drafted"], rec["n_accept"]) == (3, 260, 211, 189)
        assert rec["mean_len"] == 3.66
        assert rec["mean_len_source"] == "stats"
        assert rec["acc_rate_per_pos"] == [0.958, 0.887, 0.817]
        assert abs(rec["decode_tps"] - 0.225) < 1e-9


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


# ---- baseline-vs-MTP pairs (parser --pairs) and the script's arm validation -----------------------------
# Verbatim lines from the first interleaved pair (code prompt): llama-completion reports the baseline
# decode rate in its common_perf_print "eval time" line; llama-speculative-simple reports "decoded".
REAL_BASELINE_LOG = """\
4.20.332.943 I common_perf_print: prompt eval time =   10280.11 ms /    58 tokens (  177.24 ms per token,     5.64 tokens per second)
4.20.333.048 I common_perf_print:        eval time =  199613.05 ms /   255 runs   (  782.80 ms per token,     1.28 tokens per second)
"""
REAL_MTP_LOG = """\
2.58.860.660 I encoded   58 tokens in    6.941 seconds, speed:    8.356 t/s
2.58.860.660 I decoded  259 tokens in  115.649 seconds, speed:    2.240 t/s
2.58.860.660 I 
2.58.860.661 I n_draft   = 2
2.58.860.661 I n_predict = 259
2.58.860.661 I n_drafted = 178
2.58.860.661 I n_accept  = 170
2.58.860.662 I accept    = 95.506%
2.58.860.703 I spec common_specu: statistics        draft-mtp: #calls(b,g,a) =    1     89     89, #gen drafts =     89, #acc drafts =    87, #gen tokens =    178, #acc tokens =   170, #mean acc len = 2.91, #acc rate/pos = (0.978, 0.933), dur(b,g,a) = 0.009, 7596.425, 0.512 ms
2.58.860.711 I common_perf_print: prompt eval time =  104222.67 ms /   324 tokens (  321.67 ms per token,     3.11 tokens per second)
2.58.860.712 I common_perf_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
"""

SCRIPT = ROOT / "scripts" / "qwen4exp-mtp-acceptance.sh"


def write_pair(tmp: pathlib.Path, base_text: str = REAL_BASELINE_LOG, mtp_text: str = REAL_MTP_LOG,
               prompt: str = "code") -> pathlib.Path:
    d = tmp / "pairs"
    d.mkdir(exist_ok=True)
    if base_text is not None:
        (d / ("base_%s_1.log" % prompt)).write_text(base_text, encoding="utf-8")
    if mtp_text is not None:
        (d / ("mtp_%s_1.log" % prompt)).write_text(mtp_text, encoding="utf-8")
    return d


def test_pairs_mode_reports_the_speedup_of_the_mtp_run_over_the_baseline() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        d = write_pair(pathlib.Path(tmp_raw))
        result = run(extra=["--pairs", str(d), "--json"])
        assert result.returncode == 0, result.stdout
        pair = json.loads(result.stdout)["pairs"][0]
        assert pair["prompt"] == "code"
        # baseline rate = runs / seconds = 255 / 199.61305, not the rounded 1.28 the log prints
        assert abs(pair["base_tps"] - 255 / 199.61305) < 1e-9
        assert abs(pair["mtp_tps"] - 2.240) < 1e-9
        assert abs(pair["speedup"] - 2.240 / (255 / 199.61305)) < 1e-9
        assert abs(pair["accept_rate"] - 170 / 178) < 1e-9


def test_pairs_text_table_names_the_prompt_and_prints_the_speedup() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        d = write_pair(pathlib.Path(tmp_raw))
        result = run(extra=["--pairs", str(d)])
        assert result.returncode == 0, result.stdout
        assert "code" in result.stdout and "1.75x" in result.stdout


def test_pairs_mode_rejects_a_speculative_log_passed_as_the_baseline() -> None:
    # negative control: a speculative run also prints an "eval time" line, for 1 run and 0.00 ms
    with tempfile.TemporaryDirectory() as tmp_raw:
        d = write_pair(pathlib.Path(tmp_raw), base_text=REAL_MTP_LOG)
        result = run(extra=["--pairs", str(d)])
        assert result.returncode == 2
        assert "not a no-MTP baseline" in result.stdout
        assert "Traceback" not in result.stdout


def test_pairs_mode_rejects_a_baseline_log_with_no_eval_time_line() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        d = write_pair(pathlib.Path(tmp_raw), base_text="0.00.001.000 E llama_model_load: error loading model\n")
        result = run(extra=["--pairs", str(d)])
        assert result.returncode == 2
        assert "no 'eval time' line" in result.stdout


def test_pairs_mode_rejects_a_baseline_without_its_mtp_partner() -> None:
    with tempfile.TemporaryDirectory() as tmp_raw:
        d = write_pair(pathlib.Path(tmp_raw), mtp_text=None)
        result = run(extra=["--pairs", str(d)])
        assert result.returncode == 2
        assert "mtp_code_1.log" in result.stdout


def run_script(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    import os
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run(["bash", str(SCRIPT), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False, env=full_env)


def test_script_dry_run_refuses_an_arm_with_an_unknown_head_instead_of_printing_an_empty_md() -> None:
    result = run_script("--dry-run", env={"ARMS": "q9_n2_code"})
    assert result.returncode != 0
    assert "bad arm" in result.stderr
    assert "-md ''" not in result.stdout


def test_script_dry_run_refuses_a_malformed_arm_name() -> None:
    for arm in ("q4_n2", "q4_x2_code", "q4_n2_p07_code", "q4_n2_poetry"):
        result = run_script("--dry-run", env={"ARMS": arm})
        assert result.returncode != 0, arm
        assert "bad arm" in result.stderr, arm


def test_script_pairs_dry_run_prints_the_interleaved_order_and_the_baseline_has_no_speculation() -> None:
    result = run_script("--pairs", "--dry-run", env={"BIN": "/x/spec", "BASE_BIN": "/x/completion", "OUT": "/o"})
    assert result.returncode == 0, result.stderr
    headers = [ln[2:] for ln in result.stdout.splitlines() if ln.startswith("# ")]
    assert headers == ["base_code_1", "mtp_code_1", "mtp_chat_1", "base_chat_1", "base_reasoning_1", "mtp_reasoning_1"]
    cmds = [ln for ln in result.stdout.splitlines() if not ln.startswith("# ")]
    assert len(cmds) == 6
    for header, cmd in zip(headers, cmds):
        if header.startswith("base_"):
            assert cmd.startswith("/x/completion ") and "--spec-type" not in cmd and "-no-cnv" in cmd, cmd
        else:
            assert cmd.startswith("/x/spec ") and "--spec-type draft-mtp" in cmd and "-no-cnv" not in cmd, cmd
        assert "/o/pairs/%s.log" % header in cmd


def test_script_pairs_baseline_and_mtp_share_the_same_run_flags() -> None:
    result = run_script("--pairs", "--dry-run", env={"BIN": "/x/spec", "BASE_BIN": "/x/completion"})
    cmds = [ln for ln in result.stdout.splitlines() if not ln.startswith("# ")]
    base = cmds[0].split()
    mtp = cmds[1].split()
    for flag in ("-n", "--seed", "--temp", "-c", "-ub", "-ngl", "-lm", "-lzm", "-t", "-tb"):
        assert flag in base and flag in mtp, flag
        assert base[base.index(flag) + 1] == mtp[mtp.index(flag) + 1], flag
