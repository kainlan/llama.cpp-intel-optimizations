#!/usr/bin/env python3
"""Fail-closed reader for the SYCL bench matrices (merge-certification and long-prompt).

Plan: docs/plans/2026-08-02-sycl-merge-readiness.md step 6 (merge-cert) and
docs/plans/2026-09-04-sycl-perf-followups.md task L4 (long-prompt). Gate
definitions live in docs/backend/sycl-perf-baselines.md.

Two independent matrices, selected with --matrix (default merge-cert):

  merge-cert   Four arms (B70/B50 x Mistral/GPT-OSS). Each arm is FIVE separate
               `llama-bench -v ... -p 512 -n 128 -fa 1 -r 5` PROCESSES, one log
               each. Every arm carries a numeric floor (B50) or band (B70) --
               see MERGE_CERT_ARMS below -- and a miss is exit 1, not exit 2.

  long-prompt  Twelve arms (B70/B50 x Mistral/GPT-OSS/gemma4 x pp2048/pp8192).
               Same five-process-per-arm shape, but REPORT-ONLY: no floor or
               band has been declared for it yet (a new guardrail needs an
               owner ruling per CLAUDE.md), so it can only ever exit 0 (input
               clean) or 2 (input gap) -- exit 1 is unreachable for this
               matrix. `--table` additionally prints the markdown rows for
               docs/backend/sycl-perf-baselines.md, derived from the same
               parsed samples as the report -- never re-parsed.

The `+/-` llama-bench prints is the spread WITHIN one process and is
deliberately ignored; every verdict here is the mean ACROSS the five
processes (long-prompt also reports the sample standard deviation across
them).

Why this script exists at all: reading the logs by eye fails OPEN. A missing
arm, a truncated log, and a run that emitted no `-fa 1` row all look exactly
like a clean pass. So every uncertainty here is an error, never a smaller
sample and never a skipped check.

    exit 0  every declared arm was fully present and parseable (and, for
            merge-cert, within its gate)
    exit 1  VERDICT FAIL - merge-cert only: everything parsed, and a mean
            missed its floor/band. Unreachable for long-prompt.
    exit 2  INPUT/PARSE FAILURE - a verdict could not be computed at all

1 and 2 are deliberately distinct: "the branch is slow" and "I could not
measure the branch" are different facts and must not share an exit code.

Usage:
    parse-sycl-bench-matrix.py --dir artifacts/perf-final
    parse-sycl-bench-matrix.py --arm b50-mistral=log1,log2,log3,log4,log5
    parse-sycl-bench-matrix.py --matrix long-prompt --dir artifacts/perf-<sha>-longprompt --table
    parse-sycl-bench-matrix.py --self-test

`--dir` expects `<arm>-<n>.log` (or `.txt`) for n in 1..runs, e.g.
`b70-mistral-1.log` (merge-cert) or `b70-mistral-pp2048-1.log` (long-prompt).
Arms named below.

Stdlib only, per the project's minimal-dependency rule.
"""

import argparse
import glob
import io
import os
import re
import statistics
import sys

# --- Gate constants -------------------------------------------------------
#
# Source: docs/backend/sycl-perf-baselines.md ("Merge-certification performance
# gate"), which restates plan step 6. Do NOT edit these to make a run pass --
# the plan is explicit that a B70 miss opens an owner-reviewed baseline task
# rather than authorising a lowered guardrail.
#
# B50 arms carry explicit FLOORS (mean must be >=).
# B70 arms carry BANDS (mean must fall inside the recorded historical min-max);
# this document does not authorise a numeric B70 merge floor.
# Long-prompt arms carry neither -- REPORT means "no gate exists yet".

FLOOR = "floor"
BAND = "band"
REPORT = "report"

MERGE_CERT_ARMS = {
    "b50-gptoss": {
        "selector": "level_zero:1",
        "kind": FLOOR,
        "tests": ("pp512", "tg128"),
        "pp512": 849.0,
        "tg128": 30.4,
    },
    "b50-mistral": {
        "selector": "level_zero:1",
        "kind": FLOOR,
        "tests": ("pp512", "tg128"),
        "pp512": 1128.0,
        "tg128": 44.6,
    },
    # B70 bands refreshed 2026-08-26 under owner ruling (llama.cpp-kzug):
    # the pre-refresh rows predated the oneDNN-batched MoE PP default-ON flip
    # (llama.cpp-iikr) and a master pp improvement, so both binaries of the
    # Phase C paired certification EXCEEDED the pp bands by ~20% and missed
    # the mistral tg band identically. New bands anchor on the Phase C
    # perf-final paired data at 410433dd5 (pre and cand agree in-pair),
    # widened to the documented B70 noise (tg cv 3.3%, +-10% single-run).
    "b70-gptoss": {
        "selector": "level_zero:0",
        "kind": BAND,
        "tests": ("pp512", "tg128"),
        "pp512": (1600.0, 1810.0),
        "tg128": (38.0, 46.5),
    },
    "b70-mistral": {
        "selector": "level_zero:0",
        "kind": BAND,
        "tests": ("pp512", "tg128"),
        "pp512": (2850.0, 3200.0),
        "tg128": (95.0, 112.0),
    },
}

# --- Long-prompt matrix (report-only; llama.cpp-z0wt / plan task L4) ------
#
# Twelve arms: {b70,b50} x {mistral,gptoss,gemma4} x {pp2048,pp8192}. No
# floor/band declared -- a new guardrail needs an owner ruling per CLAUDE.md
# -- so every arm is kind REPORT and the verdict can only ever be "input was
# clean" (exit 0) or "input had a gap" (exit 2). check_gate() always returns
# ok=True for REPORT arms, so VERDICT FAIL (exit 1) is structurally
# unreachable for this matrix.
_LONG_PROMPT_SELECTORS = {"b70": "level_zero:0", "b50": "level_zero:1"}
_LONG_PROMPT_MODELS = ("mistral", "gptoss", "gemma4")
_LONG_PROMPT_PPS = (2048, 8192)

MODEL_LABELS = {
    "mistral": "Mistral 7B Q4_0",
    "gptoss": "GPT-OSS 20B MXFP4",
    "gemma4": "gemma4 E4B Q8_0",
}
CARD_LABELS = {"b70": "B70", "b50": "B50"}


def _build_long_prompt_arms():
    arms = {}
    for card, selector in _LONG_PROMPT_SELECTORS.items():
        for model in _LONG_PROMPT_MODELS:
            for pp in _LONG_PROMPT_PPS:
                arms["%s-%s-pp%d" % (card, model, pp)] = {
                    "selector": selector,
                    "kind": REPORT,
                    "tests": ("pp%d" % pp, "tg128"),
                }
    return arms


LONG_PROMPT_ARMS = _build_long_prompt_arms()

MATRICES = {
    "merge-cert": MERGE_CERT_ARMS,
    "long-prompt": LONG_PROMPT_ARMS,
}
DEFAULT_MATRIX = "merge-cert"

# --- Free-VRAM contamination floors ---------------------------------------
#
# Purpose per docs/backend/sycl-perf-baselines.md: catch a run that measured
# under another tenant's memory pressure ("a B70 run reporting ~13.8 GB instead
# of 32.6 GB is confounded -- discard it"). This is a CONTAMINATION DETECTOR,
# not a precision instrument, and it is deliberately loose for a reason:
#
#   The perf document states "~16.2 GB free on the B50", but the only two
#   committed real B50 `-v` captures (captures/16-soak-mistral-bench-r30.err
#   and captures/17-soak-gptoss-bench-r15.err) both report 14677/14679 MiB
#   free on a healthy card. A floor set at the documented figure would fail
#   every healthy B50 run. The discrepancy is unresolved -- 16250 is close to
#   the card's ~16304 MB TOTAL, so the documented number may be a total rather
#   than an observed free -- and is recorded as a gap rather than guessed at.
#
#   The B70 figure has NO committed `-v` capture to check against; 30000 is
#   derived from the documented ~32602 MiB and is set well above the ~14131 MiB
#   (13.8 GB) contamination case the document names. Tighten with
#   --min-free-mib once a real B70 capture is committed.
#
# Shared by both matrices -- the selectors (and their contamination floors)
# don't change with the matrix, only the arms measured on them.
# Override per invocation with --min-free-mib b70=NNNNN.
MIN_FREE_MIB = {
    "level_zero:0": 30000,  # B70, derived from docs (no committed capture yet)
    "level_zero:1": 14000,  # B50, below the 14677/14679 observed on a clean card
}

DEFAULT_RUNS = 5

# --- Log format -----------------------------------------------------------
#
# Free-VRAM line: verbatim shape from captures/16-soak-mistral-bench-r30.err
#   "llama_prepare_model_devices: using device SYCL0 (...) - 14677 MiB free"
FREE_VRAM_RE = re.compile(r"-\s+(\d+)\s+MiB free")

# n_ctx: llama_context's own init log, printed once per llama_context BUILT --
# i.e. once per llama-bench TEST CASE (the pp-only and tg-only tests each
# build their own context), and it reflects whatever n_ctx the planner
# actually settled on, not merely what was requested (verified 2026-09-07:
# a `-p 8192` run reported "n_ctx = 8192", not the naive 8192+128). Anchored
# to "llama_context:" so it never matches the model-metadata
# "n_ctx_train"/"n_ctx_orig_yarn" lines or the SYCL planner's own "n_ctx="
# diagnostics; the `\s+=` after "n_ctx" excludes "n_ctx_seq" (its "_seq" is
# not whitespace, so `\s+` can't bridge to "=").
N_CTX_RE = re.compile(r"llama_context:\s*n_ctx\s+=\s*(\d+)")

# Result table: markdown_printer in tools/llama-bench/llama-bench.cpp.
#   header  -> "| model | size | params | backend | ngl | fa | test | t/s |"
#              (print_header; get_field_display_name maps flash_attn -> "fa")
#   test    -> "pp%d" / "tg%d" / "pp%d+tg%d"          (print_test)
#   t/s     -> "%.2f +/- %.2f" with a UTF-8 +/- sign  (print_test)
#   fa cell -> the INT enum: LLAMA_FLASH_ATTN_TYPE_ENABLED == 1 (llama.h),
#              so `-fa 1` renders as "1". Rows that are not fa=1 are ignored,
#              and an arm with no fa=1 rows is an ERROR, not an empty result.
TS_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\u00b1|\+/-)")
SEPARATOR_RE = re.compile(r"^[\s:\-|]+$")


class ParseError(Exception):
    """Any condition under which a verdict cannot be computed. Always exit 2."""


def split_row(line):
    """Split a markdown table row into stripped cells."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def parse_log(path, min_free_mib, wanted_tests):
    """Return {test: float for test in wanted_tests} plus "free_mib" and
    "n_ctx" for one process.

    Raises ParseError on anything that makes the file unusable. There is no
    partial success: a file either yields every wanted test or it fails.

    "n_ctx" is the last `llama_context: n_ctx = N` value seen before the row
    matching wanted_tests[0] (the "pp" test) -- i.e. the achieved context
    size for THAT test, not for whatever runs after it (llama-bench builds a
    separate llama_context per test, so a later test's own n_ctx line, e.g.
    tg128's, must not leak into this figure). None if no such line appears
    before that row. Only the long-prompt --table path consults this; every
    other caller ignores it.
    """
    if not os.path.isfile(path):
        raise ParseError("%s: no such file" % path)

    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    if not text.strip():
        raise ParseError("%s: file is empty (an empty capture is VOID, not clean)" % path)

    free_matches = FREE_VRAM_RE.findall(text)
    if not free_matches:
        raise ParseError(
            "%s: no '- NNNNN MiB free' line. Was llama-bench run WITHOUT -v? "
            "Without it the log callback is nulled and free VRAM is never printed, "
            "so the run cannot be shown to be uncontaminated." % path
        )
    free_mib = min(int(m) for m in free_matches)
    if free_mib < min_free_mib:
        raise ParseError(
            "%s: only %d MiB free, below the %d MiB contamination floor. "
            "Another tenant held memory; discard this run, do not rationalise it."
            % (path, free_mib, min_free_mib)
        )

    pp_test = wanted_tests[0]
    ctx_before_pp = []
    pp_seen = False
    header_cols = None
    found = {}
    for line in text.splitlines():
        if not pp_seen:
            m_ctx = N_CTX_RE.search(line)
            if m_ctx:
                ctx_before_pp.append(int(m_ctx.group(1)))
        if not line.lstrip().startswith("|"):
            continue
        cells = split_row(line)
        if header_cols is None:
            if "test" in cells and "t/s" in cells:
                header_cols = cells
            continue
        if SEPARATOR_RE.match(line):
            continue
        if len(cells) != len(header_cols):
            continue  # a device-info table or an interleaved log line, not a result row
        row = dict(zip(header_cols, cells))
        if "fa" not in row:
            raise ParseError(
                "%s: result table has no 'fa' column, so these rows cannot be shown "
                "to be the -fa 1 matrix. llama-bench emits that column only when "
                "flash_attn differs from its default." % path
            )
        if row["fa"] != "1":
            continue
        test = row["test"]
        if test == pp_test:
            pp_seen = True  # stop attributing later n_ctx lines to this test
        if test not in wanted_tests:
            continue
        m = TS_RE.match(row["t/s"])
        if not m:
            raise ParseError("%s: unparseable t/s cell for %s: %r" % (path, test, row["t/s"]))
        if test in found:
            raise ParseError(
                "%s: more than one fa=1 %s row. Ambiguous -- one process must "
                "contribute exactly one sample per test." % (path, test)
            )
        found[test] = float(m.group(1))

    if header_cols is None:
        raise ParseError("%s: no llama-bench result table found (no header with 'test' and 't/s')" % path)

    missing = [t for t in wanted_tests if t not in found]
    if missing:
        raise ParseError(
            "%s: no fa=1 row for %s. The run did not produce the required "
            "measurement; this is not a zero." % (path, ", ".join(missing))
        )

    result = {t: found[t] for t in wanted_tests}
    result["free_mib"] = free_mib
    result["n_ctx"] = ctx_before_pp[-1] if ctx_before_pp else None
    return result


def check_gate(spec, test, mean):
    """Return (ok, description). Pure comparison -- no side effects."""
    kind = spec["kind"]
    if kind == REPORT:
        return True, "report-only"
    limit = spec[test]
    if kind == FLOOR:
        return mean >= limit, "floor >= %.2f" % limit
    lo, hi = limit
    return lo <= mean <= hi, "band %.2f-%.2f" % (lo, hi)


def _stdev(values):
    """Sample standard deviation (n-1). 0.0 for a single sample -- stdev needs
    at least two points and a lone sample has no spread to report."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


_TABLE_HEADER = "| card | model | PP2048 | TG128 | PP8192 | TG128 | ctx achieved | notes |"
_TABLE_SEPARATOR = "|---|---|---|---|---|---|---|---|"


def _build_table_rows(arms, results):
    """Return (rows, None) or (None, error_message).

    Reuses the samples already parsed into `results` -- never re-parses a
    log -- so the table and the report above it are the same numbers.
    """
    rows = [_TABLE_HEADER, _TABLE_SEPARATOR]
    for card in ("b70", "b50"):
        for model in _LONG_PROMPT_MODELS:
            cell = {}
            ctx_achieved = None
            for pp in _LONG_PROMPT_PPS:
                arm = "%s-%s-pp%d" % (card, model, pp)
                samples = results[arm]
                pp_test = "pp%d" % pp
                for test in (pp_test, "tg128"):
                    values = [s[test] for s in samples]
                    mean = sum(values) / len(values)
                    sd = _stdev(values)
                    cell[(pp, test)] = "%.2f \u00b1 %.2f" % (mean, sd)
                if pp == 8192:
                    default_ctx = pp + 128  # n_prompt + n_gen, the naive expectation
                    effective = [
                        s["n_ctx"] if s["n_ctx"] is not None else default_ctx
                        for s in samples
                    ]
                    if len(set(effective)) != 1:
                        return None, (
                            "INPUT ERROR: %s: processes disagree on achieved n_ctx: %s\n"
                            % (arm, effective)
                        )
                    ctx_achieved = effective[0]
            rows.append(
                "| %s | %s | %s | %s | %s | %s | %d | - |"
                % (
                    CARD_LABELS[card],
                    MODEL_LABELS[model],
                    cell[(2048, "pp2048")],
                    cell[(2048, "tg128")],
                    cell[(8192, "pp8192")],
                    cell[(8192, "tg128")],
                    ctx_achieved,
                )
            )
    return rows, None


def evaluate(matrix, arm_files, runs, min_free_overrides, out, err, want_table=False):
    """Evaluate one matrix. Returns an exit code (0, 1 or 2).

    matrix: "merge-cert" or "long-prompt" (a key into MATRICES).
    arm_files: {arm_name: [path, ...]}. Every arm in MATRICES[matrix] must be
    present.

    Stream contract, which the self-test asserts rather than assumes:
      out (stdout) carries RESULTS -- the per-arm means table, the markdown
      table rows when --table is given, and a PASS/report-only verdict.
      err (stderr) carries every DIAGNOSTIC that accompanies a non-zero exit.
    So exit 2 writes nothing to stdout at all (no verdict was computable, so
    there is no result to report), and exit 1 (merge-cert only) writes the
    table to stdout with the failure explanation on stderr. A caller
    redirecting only stdout can still never mistake an error for a result.
    """
    arms = MATRICES[matrix]

    if want_table and matrix != "long-prompt":
        err.write(
            "INPUT ERROR: --table is long-prompt only, got --matrix %s.\n" % matrix
        )
        return 2

    missing_arms = [a for a in sorted(arms) if a not in arm_files]
    if missing_arms:
        err.write("INPUT ERROR: arm(s) entirely absent: %s\n" % ", ".join(missing_arms))
        err.write("A missing arm is not a pass. All %d arms are required.\n" % len(arms))
        return 2

    unknown = [a for a in sorted(arm_files) if a not in arms]
    if unknown:
        err.write("INPUT ERROR: unknown arm(s): %s\n" % ", ".join(unknown))
        err.write("Known arms: %s\n" % ", ".join(sorted(arms)))
        return 2

    results = {}
    for arm in sorted(arms):
        paths = arm_files[arm]
        if len(paths) != runs:
            err.write(
                "INPUT ERROR: %s has %d sample(s), required %d. A short arm is an "
                "error, not a smaller sample.\n" % (arm, len(paths), runs)
            )
            return 2
        selector = arms[arm]["selector"]
        floor_mib = min_free_overrides.get(selector, MIN_FREE_MIB[selector])
        wanted = arms[arm]["tests"]
        samples = []
        for path in paths:
            try:
                samples.append(parse_log(path, floor_mib, wanted))
            except ParseError as exc:
                err.write("INPUT ERROR: %s: %s\n" % (arm, exc))
                return 2
        results[arm] = samples

    table_rows = None
    if want_table:
        table_rows, table_err = _build_table_rows(arms, results)
        if table_err:
            err.write(table_err)
            return 2

    failures = []
    if matrix == "merge-cert":
        out.write("Merge-certification performance matrix -- %d processes per arm\n\n" % runs)
    else:
        out.write(
            "Long-prompt performance matrix -- %d processes per arm "
            "(report-only: no gate declared for the long-prompt matrix)\n\n" % runs
        )

    for arm in sorted(arms):
        samples = results[arm]
        spec = arms[arm]
        out.write("%s (%s)\n" % (arm, spec["selector"]))
        out.write("  free VRAM: %s MiB\n" % " ".join(str(s["free_mib"]) for s in samples))
        for test in spec["tests"]:
            values = [s[test] for s in samples]
            mean = sum(values) / len(values)
            ok, desc = check_gate(spec, test, mean)
            if spec["kind"] == REPORT:
                sd = _stdev(values)
                out.write(
                    "  %-6s mean %9.2f  sd %6.2f  [%s]  %s\n"
                    % (test, mean, sd, " ".join("%.2f" % v for v in values), desc)
                )
            else:
                out.write(
                    "  %-6s mean %9.2f  [%s]  %s  %s\n"
                    % (
                        test,
                        mean,
                        " ".join("%.2f" % v for v in values),
                        desc,
                        "OK" if ok else "FAIL",
                    )
                )
            if not ok:
                failures.append("%s %s mean %.2f misses %s" % (arm, test, mean, desc))
        out.write("\n")

    if failures:
        err.write("VERDICT: FAIL\n")
        for f in failures:
            err.write("  %s\n" % f)
        err.write(
            "\nPer the plan, a B70 miss opens an owner-reviewed baseline/regression\n"
            "task; it does not authorise inventing or lowering a guardrail. Before\n"
            "concluding regression, discriminate load with an interleaved paired A/B\n"
            "against a known-good comparator on the same host state.\n"
        )
        return 1

    if matrix == "merge-cert":
        out.write("VERDICT: PASS -- all %d arms present, parseable, and within gate\n" % len(arms))
    else:
        out.write(
            "VERDICT: report-only: no gate declared for the long-prompt matrix -- "
            "all %d arms present and parseable\n" % len(arms)
        )

    if table_rows is not None:
        out.write("\n")
        for row in table_rows:
            out.write(row + "\n")

    return 0


def collect_from_dir(directory, runs, arm_names):
    """Map arm -> sorted list of its run logs. Accepts .log and .txt."""
    arm_files = {}
    for arm in arm_names:
        paths = []
        for n in range(1, runs + 1):
            hits = sorted(
                glob.glob(os.path.join(directory, "%s-%d.log" % (arm, n)))
                + glob.glob(os.path.join(directory, "%s-%d.txt" % (arm, n)))
            )
            paths.extend(hits)
        if paths:
            arm_files[arm] = paths
    return arm_files


# --- Self-test ------------------------------------------------------------
#
# The parser guards against vacuous passes, so it needs its own positive
# control or it repeats the trap it exists to prevent. These cases prove the
# parser can return each of its three exit codes, using COMMITTED fixtures.

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "task18-parser-fixtures",
)


def _fx(name):
    return os.path.join(FIXTURE_DIR, name)


def _fixture_for_long_prompt_arm(arm):
    """'b70-mistral-pp2048' -> 'b70-pp2048-good.txt'.

    One fixture per (card, pp) is committed, reused across all three models
    (llama.cpp-z0wt): the parser doesn't care which model a result table
    names, only whether the table and free-VRAM line parse.
    """
    m = re.match(r"^(b70|b50)-(?:mistral|gptoss|gemma4)-(pp\d+)$", arm)
    if not m:
        raise ValueError("not a long-prompt arm: %r" % arm)
    card, pp = m.groups()
    return _fx("%s-%s-good.txt" % (card, pp))


def _all_good(matrix):
    """Every arm of `matrix` at DEFAULT_RUNS in-gate (or, for long-prompt,
    simply parseable) samples. The baseline every case perturbs."""
    arms = MATRICES[matrix]
    if matrix == "merge-cert":
        return {arm: [_fx("%s-good.txt" % arm)] * DEFAULT_RUNS for arm in arms}
    return {arm: [_fixture_for_long_prompt_arm(arm)] * DEFAULT_RUNS for arm in arms}


def _with(matrix, **overrides):
    arms = _all_good(matrix)
    arms.update(overrides)
    return arms


def self_test(out):
    good = _fx("b50-mistral-good.txt")
    lp_pp8192_good = _fx("b70-pp8192-good.txt")

    # (name, matrix, arm_files, want_table, expected exit code)
    cases = [
        ("all arms good -> PASS", "merge-cert", _all_good("merge-cert"), False, 0),
        ("one arm has 4 samples -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4}), False, 2),
        ("one sample empty -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4 + [_fx("empty.txt")]}), False, 2),
        ("one sample missing the free-VRAM line -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4 + [_fx("no-vram-line.txt")]}), False, 2),
        ("one sample has an unparseable t/s cell -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4 + [_fx("unparseable-ts.txt")]}), False, 2),
        ("one sample has no fa column -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4 + [_fx("no-fa-column.txt")]}), False, 2),
        ("one sample contaminated (low free VRAM) -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4 + [_fx("low-free-vram.txt")]}), False, 2),
        ("a missing file -> parse error", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [good] * 4 + [_fx("does-not-exist.txt")]}), False, 2),
        ("an arm entirely absent -> parse error", "merge-cert",
         {a: v for a, v in _all_good("merge-cert").items() if a != "b70-gptoss"}, False, 2),
        ("below-floor throughput -> VERDICT FAIL (not a parse error)", "merge-cert",
         _with("merge-cert", **{"b50-mistral": [_fx("b50-mistral-below-floor.txt")] * DEFAULT_RUNS}), False, 1),

        ("long-prompt all arms good -> PASS (report-only)", "long-prompt",
         _all_good("long-prompt"), False, 0),
        ("long-prompt one arm entirely absent -> parse error", "long-prompt",
         {a: v for a, v in _all_good("long-prompt").items() if a != "b70-gemma4-pp8192"}, False, 2),
        ("long-prompt one sample missing its pp8192 row -> parse error", "long-prompt",
         _with("long-prompt", **{
             "b70-mistral-pp8192": [lp_pp8192_good] * 4 + [_fx("pp8192-missing-pprow.txt")],
         }), False, 2),
        ("long-prompt n_ctx disagreement across a cell's processes -> parse error", "long-prompt",
         _with("long-prompt", **{
             "b70-mistral-pp8192": [lp_pp8192_good] * 4 + [_fx("b70-pp8192-ctx-mismatch.txt")],
         }), True, 2),
        ("--table with merge-cert -> parse error", "merge-cert",
         _all_good("merge-cert"), True, 2),
        ("long-prompt --table all good -> PASS with markdown rows", "long-prompt",
         _all_good("long-prompt"), True, 0),
    ]

    # Expected stream occupancy per exit code. Checking this is the point:
    # without it, "diagnostics go to stderr" is a claim about the source rather
    # than a property of the program.
    #                 exit: (stdout non-empty?, stderr non-empty?)
    stream_contract = {0: (True, False), 1: (True, True), 2: (False, True)}

    failures = 0
    for name, matrix, arms, want_table, expected in cases:
        cap_out, cap_err = io.StringIO(), io.StringIO()
        got = evaluate(matrix, arms, DEFAULT_RUNS, {}, cap_out, cap_err, want_table)
        want_out, want_err = stream_contract[expected]
        streams_ok = (bool(cap_out.getvalue()) == want_out
                      and bool(cap_err.getvalue()) == want_err)
        ok = got == expected and streams_ok
        note = ""
        if got != expected:
            note = "  <- wrong exit code"
        elif not streams_ok:
            note = "  <- wrong stream: stdout=%d bytes stderr=%d bytes" % (
                len(cap_out.getvalue()), len(cap_err.getvalue()))
        out.write("  %-62s expected %d got %d  %s%s\n"
                  % (name, expected, got, "OK" if ok else "FAIL", note))
        if not ok:
            failures += 1

    out.write("\n%d/%d self-test cases passed\n" % (len(cases) - failures, len(cases)))
    if failures:
        out.write("SELF-TEST FAILED -- do not trust this parser's verdicts.\n")
        return 2
    out.write(
        "SELF-TEST PASSED -- the parser demonstrably returns 0, 1 and 2, so a PASS\n"
        "from it is a measurement rather than the only outcome it can produce.\n"
        "Stream routing is checked too: exit 2 writes NOTHING to stdout, so a caller\n"
        "capturing only stdout cannot mistake an unmeasurable run for a result.\n"
    )
    return 0


def parse_min_free(values):
    """--min-free-mib b70=31000 -> {"level_zero:0": 31000}"""
    alias = {"b70": "level_zero:0", "b50": "level_zero:1"}
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ParseError("--min-free-mib expects <b70|b50>=<mib>, got %r" % item)
        key, _, raw = item.partition("=")
        key = key.strip().lower()
        if key not in alias:
            raise ParseError("--min-free-mib key must be b70 or b50, got %r" % key)
        try:
            out[alias[key]] = int(raw)
        except ValueError:
            raise ParseError("--min-free-mib value must be an integer, got %r" % raw)
    return out


EPILOG = """
exit codes -- the contract this parser exists to provide:

  0  PASS                   every arm present and parseable (and, for
                            merge-cert, within its floor/band). Results on
                            stdout.
  1  VERDICT FAIL           merge-cert only: everything parsed; a mean
                            missed its floor/band. Means table on stdout,
                            explanation on stderr. Unreachable for
                            long-prompt (report-only, no gate declared).
                            Discriminate load from regression with an
                            interleaved paired A/B before concluding
                            regression. A B70 miss opens an owner-reviewed
                            baseline task -- it never authorises lowering a
                            guardrail.
  2  INPUT/PARSE FAILURE    no verdict could be computed. NOTHING on stdout;
                            diagnosis on stderr. This is neither a pass nor a
                            fail -- fix the inputs and re-run the matrix.

Exit 2 covers: missing file, empty file, an arm with fewer than the required
samples, an arm entirely absent, a results directory that does not exist, a
directory that exists but is empty, an unparseable t/s cell, a table with no
'fa' column (not the -fa 1 matrix), a log with no '- NNNNN MiB free' line
(llama-bench run without -v), free VRAM below the contamination floor,
--table combined with --matrix merge-cert, and (long-prompt --table only) the
five processes of one arm disagreeing on the achieved n_ctx.

1 and 2 are deliberately distinct: "the branch is slow" and "I could not
measure the branch" are different facts and must not share an exit code.

Matrices: merge-cert (four arms, numeric floor/band, gates merges) and
long-prompt (twelve arms, report-only -- no floor/band exists yet).

Verify the parser before trusting its verdict:  --self-test  (expects 16/16)
Gate definition: docs/backend/sycl-perf-baselines.md
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--matrix", choices=sorted(MATRICES), default=DEFAULT_MATRIX,
                     help="which matrix to evaluate (default: %(default)s)")
    ap.add_argument("--dir", help="directory holding <arm>-<n>.log|.txt")
    ap.add_argument("--arm", action="append", default=[],
                    help="<arm>=<log1>,<log2>,... (repeatable)")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                    help="required processes per arm (default %d)" % DEFAULT_RUNS)
    ap.add_argument("--min-free-mib", action="append", default=[],
                    help="override the free-VRAM floor, e.g. b70=31000")
    ap.add_argument("--table", action="store_true",
                    help="long-prompt only: also print the sycl-perf-baselines.md "
                         "markdown rows (report-only; error with --matrix merge-cert)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the parser against its committed fixtures and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(sys.stdout)

    if not args.dir and not args.arm:
        sys.stderr.write("INPUT ERROR: pass --dir, --arm, or --self-test\n")
        return 2

    try:
        overrides = parse_min_free(args.min_free_mib)
    except ParseError as exc:
        sys.stderr.write("INPUT ERROR: %s\n" % exc)
        return 2

    arms = MATRICES[args.matrix]

    arm_files = {}
    if args.dir:
        if not os.path.isdir(args.dir):
            sys.stderr.write(
                "INPUT ERROR: %s does not exist. A missing results directory means "
                "NOT MEASURED, never 'nothing observed'.\n" % args.dir
            )
            return 2
        arm_files.update(collect_from_dir(args.dir, args.runs, arms))
    for item in args.arm:
        if "=" not in item:
            sys.stderr.write("INPUT ERROR: --arm expects <arm>=<log,...>, got %r\n" % item)
            return 2
        name, _, paths = item.partition("=")
        arm_files[name.strip()] = [p for p in paths.split(",") if p]

    if not arm_files:
        sys.stderr.write(
            "INPUT ERROR: no arm logs found. Expected <arm>-<n>.log for arms: %s\n"
            % ", ".join(sorted(arms))
        )
        return 2

    return evaluate(args.matrix, arm_files, args.runs, overrides, sys.stdout, sys.stderr, args.table)


if __name__ == "__main__":
    sys.exit(main())
