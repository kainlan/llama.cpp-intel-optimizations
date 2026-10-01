#!/usr/bin/env python3
"""scripts/sycl-graph-diag-score.sh must read the GRAPH_DIAG final line the backend prints.

ggml_backend_sycl_free prints the whole GRAPH_DIAG summary once, as
`[GRAPH-DIAG] phase=final ...`, and the GW7 and GRP gates read every count from
that one line. The scorer takes the LAST final line whose `calls=` is >= 1 (a
temporary backend that a model load creates and frees prints a final line with
every counter at 0), anchors each key on its left, and reports VOID, never a
zero, when the line or the key is missing or the key matches twice.

This test runs the scorer itself, so there is one authority. It reads the
format literal out of ggml-sycl.cpp, formats it into a synthetic log with a
distinct sentinel for each conversion, and checks that the script returns each
scored key's own sentinel from the last qualifying line:

  * one load-time final line (`calls=0`), then two lines with `calls` >= 1;
  * a permanent colliding-key line: `xstage_host_returns_tg=<other>` beside the
    real key. The anchored scorer returns the real value; a scorer without its
    left anchor matches both and reports VOID, which fails here;
  * VOID, exit status 3, for a log with only the load-time line, for a key the
    line does not carry, and for a literal in the shape this literal replaced
    (`stage_host_returns pp=... tg=...`), which carries no key of its own.

Usage: test-sycl-graph-diag-final-scorer.py [scorer.sh [ggml-sycl.cpp]]
"""

import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The keys present in the literal today. `micro_replay` joins them with the hook
# that increments it.
SCORED_KEYS = [
    "stage_host_returns_pp",
    "stage_host_returns_tg",
    "stage_host_returns_other",
    "calls",
    "full_replay",
    "block_replay",
]

STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
CONVERSION = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?(?:ll|l|z)?[sdufg]")
KEYED_CONVERSION = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=(%[-+ #0]*\d*(?:\.\d+)?(?:ll|l|z)?[sdufg])")


def read_format_literal(source_path):
    """The concatenated format literal of the `[GRAPH-DIAG] phase=` fprintf."""
    with open(source_path, encoding="utf-8") as f:
        src = f.read()
    start = src.find('"[GRAPH-DIAG] phase=%s')
    if start < 0:
        raise AssertionError("no `[GRAPH-DIAG] phase=%s` format literal in %s" % (source_path, source_path))
    parts = []
    pos = start
    while True:
        m = STRING_LITERAL.match(src, pos)
        if not m:
            break
        parts.append(m.group(1))
        pos = m.end()
        while pos < len(src) and src[pos].isspace():
            pos += 1
    return "".join(parts).replace("\\n", "\n")


def format_line(literal, line_no, zero):
    """Fills every conversion with a sentinel unique to (line_no, conversion)."""
    values = {}
    count = [0]

    def sub(m):
        conv = m.group(0)
        count[0] += 1
        if conv.endswith("s"):
            return "final"
        if conv.endswith("f"):
            return "0.000"
        return "0" if zero else str((line_no + 1) * 1000000 + count[0])

    out = CONVERSION.sub(sub, literal).rstrip("\n")
    return out


def sentinel_of(line, key):
    m = re.search(r"(?:^| )%s=([0-9]+)(?: |$)" % re.escape(key), line)
    return int(m.group(1)) if m else None


def run_scorer(script, log_path, key):
    p = subprocess.run(["bash", script, log_path, key], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def main():
    script = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "scripts", "sycl-graph-diag-score.sh")
    source = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "ggml", "src", "ggml-sycl", "ggml-sycl.cpp")
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    literal = read_format_literal(source)
    keyed = {}
    for m in KEYED_CONVERSION.finditer(literal):
        keyed.setdefault(m.group(1), []).append(m.group(2))

    # Each scored key must be a whole key of the literal, exactly once.
    for key in SCORED_KEYS:
        n = len(re.findall(r"(?:^| )%s=%%" % re.escape(key), literal))
        check(n == 1, "key `%s` occurs %d times as a whole key in the format literal (want 1)" % (key, n))
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1

    with tempfile.TemporaryDirectory() as tmp:

        def write_log(name, lines):
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write("noise before\n")
                for line in lines:
                    f.write(line + "\n")
                    f.write("[GRAPH-DIAG] phase=TG calls=999 pp=1 tg=2 stage_host_returns_tg=88888888\n")
                f.write("noise after\n")
            return path

        load_time = format_line(literal, 0, zero=True)
        first = format_line(literal, 1, zero=False)
        last = format_line(literal, 2, zero=False)

        # The per-frame TG line in the log above is not a final line and must
        # never be read; the load-time line has calls=0 and must never be read.
        log = write_log("sentinels.log", [load_time, first, last])
        for key in SCORED_KEYS:
            want = sentinel_of(last, key)
            check(want is not None and want >= 1, "synthetic last line has no sentinel for `%s`" % key)
            rc, out = run_scorer(script, log, key)
            check(rc == 0 and out == str(want), "key `%s`: want %s from the last qualifying line, got rc=%d %r" %
                  (key, want, rc, out))

        # A trailing final line with calls=0 (a temporary backend freed after
        # the context) must not displace the context's own line.
        log = write_log("trailing-load-time.log", [first, last, load_time])
        rc, out = run_scorer(script, log, "stage_host_returns_tg")
        check(rc == 0 and out == str(sentinel_of(last, "stage_host_returns_tg")),
              "a trailing calls=0 final line displaced the last qualifying line: rc=%d %r" % (rc, out))

        # Colliding key: a longer key that ends with the scored name.
        collided = last.replace(" stage_host_returns_tg=", " xstage_host_returns_tg=5555555 stage_host_returns_tg=", 1)
        check(collided != last, "could not build the colliding-key line")
        log = write_log("colliding.log", [load_time, collided])
        rc, out = run_scorer(script, log, "stage_host_returns_tg")
        check(rc == 0 and out == str(sentinel_of(last, "stage_host_returns_tg")),
              "colliding key `xstage_host_returns_tg=` was read: rc=%d %r" % (rc, out))

        # VOID: only the load-time final line.
        log = write_log("only-load-time.log", [load_time])
        rc, out = run_scorer(script, log, "stage_host_returns_tg")
        check(rc == 3 and out == "VOID", "a log with only a calls=0 final line must be VOID: rc=%d %r" % (rc, out))

        # VOID: no final line at all.
        path = os.path.join(tmp, "no-final.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[GRAPH-DIAG] phase=TG calls=5 pp=1 tg=4 stage_host_returns_tg=3\n")
        rc, out = run_scorer(script, path, "stage_host_returns_tg")
        check(rc == 3 and out == "VOID", "a log with no final line must be VOID: rc=%d %r" % (rc, out))

        # VOID: a key the line does not carry.
        log = write_log("missing-key.log", [load_time, last])
        rc, out = run_scorer(script, log, "micro_replay_not_a_key")
        check(rc == 3 and out == "VOID", "a key the line does not carry must be VOID: rc=%d %r" % (rc, out))

        # VOID: a key that matches twice.
        twice = last + " stage_host_returns_tg=42"
        log = write_log("twice.log", [load_time, twice])
        rc, out = run_scorer(script, log, "stage_host_returns_tg")
        check(rc == 3 and out == "VOID", "a key matching twice must be VOID: rc=%d %r" % (rc, out))

        # VOID: a key that matches twice in ADJACENT tokens. A scanner that
        # consumes the separator after each match reads "k=42 k=N" as one match.
        adjacent = last.replace(" stage_host_returns_tg=", " stage_host_returns_tg=42 stage_host_returns_tg=", 1)
        check(adjacent != last, "could not build the adjacent-duplicate line")
        log = write_log("adjacent.log", [load_time, adjacent])
        rc, out = run_scorer(script, log, "stage_host_returns_tg")
        check(rc == 3 and out == "VOID", "an adjacent duplicate key must be VOID: rc=%d %r" % (rc, out))

        # Mutant: the separator-consuming scanner the scorer used to carry must
        # be fooled by the adjacent case, or the case does not discriminate.
        with open(script, encoding="utf-8") as f:
            text = f.read()
        body = re.search(r"diag_key\(\) \{.*?\n\}\n", text, re.S)
        check(body is not None, "could not locate diag_key in the scorer")
        if body:
            old_scan = ("diag_key() {\n    printf '%s\\n' \"$1\" | "
                        "grep -oE \"(^| )$2=[0-9]+( |\\$)\" | grep -oE '[0-9]+'\n}\n")
            mutant = os.path.join(tmp, "mutant-score.sh")
            with open(mutant, "w", encoding="utf-8") as f:
                f.write(text.replace(body.group(0), old_scan, 1))
            rc, out = run_scorer(mutant, log, "stage_host_returns_tg")
            check(rc == 0, "mutant scanner was not fooled by the adjacent duplicate (the case does not "
                  "discriminate): rc=%d %r" % (rc, out))

        # VOID: the literal this one replaced carried no key of its own.
        legacy = last.replace(
            "stage_host_returns_pp=", "stage_host_returns pp=", 1).replace(
            "stage_host_returns_tg=", "tg=", 1).replace(
            "stage_host_returns_other=", "other=", 1)
        log = write_log("legacy.log", [load_time, legacy])
        rc, out = run_scorer(script, log, "stage_host_returns_tg")
        check(rc == 3 and out == "VOID", "the legacy key-less literal must be VOID: rc=%d %r" % (rc, out))

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("PASS: %d scored keys, last-qualifying-line, collision, adjacent-duplicate and VOID cases" % len(SCORED_KEYS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
