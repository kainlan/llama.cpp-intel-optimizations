#!/usr/bin/env python3
"""Fail-closed reader for the Task 20 merge-certification performance matrix.

Plan: docs/plans/2026-08-02-sycl-merge-readiness.md step 6. Gate definition:
docs/backend/sycl-perf-baselines.md, "Merge-certification performance gate".

The matrix is four arms (B70/B50 x Mistral/GPT-OSS). Each arm is FIVE separate
`llama-bench -v ... -p 512 -n 128 -fa 1 -r 5` PROCESSES, one log each. The `+/-`
llama-bench prints is the spread WITHIN one process and is deliberately ignored;
the gate is the mean ACROSS the five processes.

Why this script exists at all: reading the logs by eye fails OPEN. A missing
arm, a truncated log, and a run that emitted no `-fa 1` row all look exactly
like a clean pass. So every uncertainty here is an error, never a smaller
sample and never a skipped check.

    exit 0  every declared arm was fully present, parseable, and within gate
    exit 1  VERDICT FAIL - everything parsed, and a mean missed its floor/band
    exit 2  INPUT/PARSE FAILURE - a verdict could not be computed at all

1 and 2 are deliberately distinct: "the branch is slow" and "I could not
measure the branch" are different facts and must not share an exit code.

Usage:
    parse-sycl-bench-matrix.py --dir artifacts/perf-final
    parse-sycl-bench-matrix.py --arm b50-mistral=log1,log2,log3,log4,log5
    parse-sycl-bench-matrix.py --self-test

`--dir` expects `<arm>-<n>.log` (or `.txt`) for n in 1..runs, e.g.
`b70-mistral-1.log`. Arms named below.

Stdlib only, per the project's minimal-dependency rule.
"""

import argparse
import glob
import io
import os
import re
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

FLOOR = "floor"
BAND = "band"

ARMS = {
    "b50-gptoss": {
        "selector": "level_zero:1",
        "kind": FLOOR,
        "pp512": 849.0,
        "tg128": 30.4,
    },
    "b50-mistral": {
        "selector": "level_zero:1",
        "kind": FLOOR,
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
        "pp512": (1600.0, 1810.0),
        "tg128": (38.0, 46.5),
    },
    "b70-mistral": {
        "selector": "level_zero:0",
        "kind": BAND,
        "pp512": (2850.0, 3200.0),
        "tg128": (95.0, 112.0),
    },
}

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

WANTED_TESTS = ("pp512", "tg128")


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


def parse_log(path, min_free_mib):
    """Return {"pp512": float, "tg128": float, "free_mib": int} for one process.

    Raises ParseError on anything that makes the file unusable. There is no
    partial success: a file either yields both measurements or it fails.
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

    header_cols = None
    found = {}
    for line in text.splitlines():
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
        if test not in WANTED_TESTS:
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

    missing = [t for t in WANTED_TESTS if t not in found]
    if missing:
        raise ParseError(
            "%s: no fa=1 row for %s. The run did not produce the required "
            "measurement; this is not a zero." % (path, ", ".join(missing))
        )

    return {"pp512": found["pp512"], "tg128": found["tg128"], "free_mib": free_mib}


def check_gate(arm, test, mean):
    """Return (ok, description). Pure comparison -- no side effects."""
    spec = ARMS[arm]
    limit = spec[test]
    if spec["kind"] == FLOOR:
        return mean >= limit, "floor >= %.2f" % limit
    lo, hi = limit
    return lo <= mean <= hi, "band %.2f-%.2f" % (lo, hi)


def evaluate(arm_files, runs, min_free_overrides, out, err):
    """Evaluate the whole matrix. Returns an exit code (0, 1 or 2).

    arm_files: {arm_name: [path, ...]}. Every arm in ARMS must be present.

    Stream contract, which the self-test asserts rather than assumes:
      out (stdout) carries RESULTS -- the per-arm means table and a PASS verdict.
      err (stderr) carries every DIAGNOSTIC that accompanies a non-zero exit.
    So exit 2 writes nothing to stdout at all (no verdict was computable, so
    there is no result to report), and exit 1 writes the table to stdout with
    the failure explanation on stderr. A caller redirecting only stdout can
    still never mistake an error for a result.
    """
    missing_arms = [a for a in sorted(ARMS) if a not in arm_files]
    if missing_arms:
        err.write("INPUT ERROR: arm(s) entirely absent: %s\n" % ", ".join(missing_arms))
        err.write("A missing arm is not a pass. All %d arms are required.\n" % len(ARMS))
        return 2

    unknown = [a for a in sorted(arm_files) if a not in ARMS]
    if unknown:
        err.write("INPUT ERROR: unknown arm(s): %s\n" % ", ".join(unknown))
        err.write("Known arms: %s\n" % ", ".join(sorted(ARMS)))
        return 2

    results = {}
    for arm in sorted(ARMS):
        paths = arm_files[arm]
        if len(paths) != runs:
            err.write(
                "INPUT ERROR: %s has %d sample(s), required %d. A short arm is an "
                "error, not a smaller sample.\n" % (arm, len(paths), runs)
            )
            return 2
        selector = ARMS[arm]["selector"]
        floor_mib = min_free_overrides.get(selector, MIN_FREE_MIB[selector])
        samples = []
        for path in paths:
            try:
                samples.append(parse_log(path, floor_mib))
            except ParseError as exc:
                err.write("INPUT ERROR: %s: %s\n" % (arm, exc))
                return 2
        results[arm] = samples

    failures = []
    out.write("Merge-certification performance matrix -- %d processes per arm\n\n" % runs)
    for arm in sorted(ARMS):
        samples = results[arm]
        spec = ARMS[arm]
        out.write("%s (%s)\n" % (arm, spec["selector"]))
        out.write("  free VRAM: %s MiB\n" % " ".join(str(s["free_mib"]) for s in samples))
        for test in WANTED_TESTS:
            values = [s[test] for s in samples]
            mean = sum(values) / len(values)
            ok, desc = check_gate(arm, test, mean)
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

    out.write("VERDICT: PASS -- all %d arms present, parseable, and within gate\n" % len(ARMS))
    return 0


def collect_from_dir(directory, runs):
    """Map arm -> sorted list of its run logs. Accepts .log and .txt."""
    arm_files = {}
    for arm in ARMS:
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


def _all_good():
    """Every arm at five in-gate samples. The baseline every case perturbs."""
    return {arm: [_fx("%s-good.txt" % arm)] * DEFAULT_RUNS for arm in ARMS}


def _with(**overrides):
    arms = _all_good()
    arms.update(overrides)
    return arms


def self_test(out):
    good = _fx("b50-mistral-good.txt")
    cases = [
        ("all arms good -> PASS", _all_good(), 0),
        ("one arm has 4 samples -> parse error",
         _with(**{"b50-mistral": [good] * 4}), 2),
        ("one sample empty -> parse error",
         _with(**{"b50-mistral": [good] * 4 + [_fx("empty.txt")]}), 2),
        ("one sample missing the free-VRAM line -> parse error",
         _with(**{"b50-mistral": [good] * 4 + [_fx("no-vram-line.txt")]}), 2),
        ("one sample has an unparseable t/s cell -> parse error",
         _with(**{"b50-mistral": [good] * 4 + [_fx("unparseable-ts.txt")]}), 2),
        ("one sample has no fa column -> parse error",
         _with(**{"b50-mistral": [good] * 4 + [_fx("no-fa-column.txt")]}), 2),
        ("one sample contaminated (low free VRAM) -> parse error",
         _with(**{"b50-mistral": [good] * 4 + [_fx("low-free-vram.txt")]}), 2),
        ("a missing file -> parse error",
         _with(**{"b50-mistral": [good] * 4 + [_fx("does-not-exist.txt")]}), 2),
        ("an arm entirely absent -> parse error",
         {a: v for a, v in _all_good().items() if a != "b70-gptoss"}, 2),
        ("below-floor throughput -> VERDICT FAIL (not a parse error)",
         _with(**{"b50-mistral": [_fx("b50-mistral-below-floor.txt")] * DEFAULT_RUNS}), 1),
    ]

    # Expected stream occupancy per exit code. Checking this is the point:
    # without it, "diagnostics go to stderr" is a claim about the source rather
    # than a property of the program.
    #                 exit: (stdout non-empty?, stderr non-empty?)
    stream_contract = {0: (True, False), 1: (True, True), 2: (False, True)}

    failures = 0
    for name, arms, expected in cases:
        cap_out, cap_err = io.StringIO(), io.StringIO()
        got = evaluate(arms, DEFAULT_RUNS, {}, cap_out, cap_err)
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

  0  PASS                   every arm present, parseable, and within its
                            floor/band. Results on stdout.
  1  VERDICT FAIL           everything parsed; a mean missed its floor/band.
                            Means table on stdout, explanation on stderr.
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
(llama-bench run without -v), and free VRAM below the contamination floor.

1 and 2 are deliberately distinct: "the branch is slow" and "I could not
measure the branch" are different facts and must not share an exit code.

Verify the parser before trusting its verdict:  --self-test  (expects 10/10)
Gate definition: docs/backend/sycl-perf-baselines.md
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir", help="directory holding <arm>-<n>.log|.txt")
    ap.add_argument("--arm", action="append", default=[],
                    help="<arm>=<log1>,<log2>,... (repeatable)")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                    help="required processes per arm (default %d)" % DEFAULT_RUNS)
    ap.add_argument("--min-free-mib", action="append", default=[],
                    help="override the free-VRAM floor, e.g. b70=31000")
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

    arm_files = {}
    if args.dir:
        if not os.path.isdir(args.dir):
            sys.stderr.write(
                "INPUT ERROR: %s does not exist. A missing results directory means "
                "NOT MEASURED, never 'nothing observed'.\n" % args.dir
            )
            return 2
        arm_files.update(collect_from_dir(args.dir, args.runs))
    for item in args.arm:
        if "=" not in item:
            sys.stderr.write("INPUT ERROR: --arm expects <arm>=<log,...>, got %r\n" % item)
            return 2
        name, _, paths = item.partition("=")
        arm_files[name.strip()] = [p for p in paths.split(",") if p]

    if not arm_files:
        sys.stderr.write(
            "INPUT ERROR: no arm logs found. Expected <arm>-<n>.log for arms: %s\n"
            % ", ".join(sorted(ARMS))
        )
        return 2

    return evaluate(arm_files, args.runs, overrides, sys.stdout, sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
