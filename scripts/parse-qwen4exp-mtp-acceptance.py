#!/usr/bin/env python3
"""Summarise llama-speculative-simple logs for the qwen4exp MTP measurement.

Input: one log per arm, named `<head>_n<k>[_p05]_<prompt>.log` (`_p05` = --spec-draft-p-min 0.5) (see
scripts/qwen4exp-mtp-acceptance.sh).  Output: per-arm draft acceptance and mean
tokens per round, plus a token-weighted roll-up per (head, n-max).

What the numbers are:
  accept_rate  n_accept / n_drafted, as llama-speculative-simple prints it.
  mean_len     tokens produced per verify round (1 + accepted drafts).  Taken
               from the `#mean acc len` statistics line when the log has it
               (run with -lv 4); otherwise derived as n_predict / (n_drafted /
               n_draft), which assumes every round drafted exactly n_draft
               tokens, and labelled `derived`.

--pairs DIR reads interleaved baseline-vs-MTP pairs, `base_<prompt>_<n>.log` (llama-completion, no
speculation) next to `mtp_<prompt>_<n>.log` (llama-speculative-simple), and prints the MTP decode rate over
the baseline's.  The baseline rate is runs / seconds from its `common_perf_print` "eval time" line (the
log's own t/s is rounded to 2 places); the MTP rate is the `decoded ... speed` figure.  The two timers
cover 255 and 259 tokens respectively (the prompt pass emits the first token in both), a 0.4% asymmetry.

A log whose summary shows n_drafted == 0 is rejected rather than reported as
0% acceptance: that is speculation that never ran, not speculation that
failed, and the two must not share a row.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

INT_FIELDS = ("n_draft", "n_predict", "n_drafted", "n_accept")
# at -lv 4 every line carries a "<s>.<ms>.<us>.<ns> <level> " log prefix; at default verbosity none does
LOG_PREFIX = r"^(?:\d+\.\d+\.\d+\.\d+ [A-Z] )?"
STATS_RE = re.compile(r"#mean acc len = ([0-9.]+)(?:, #acc rate/pos = \(([^)]*)\))?")
BASE_EVAL_RE = re.compile(r"common_perf_print:\s+eval time =\s+([0-9.]+) ms /\s+(\d+) runs")
PAIR_NAME_RE = re.compile(r"^base_(?P<prompt>[a-z]+)_(?P<idx>\d+)\.log$")
DECODE_RE = re.compile(r"decoded\s+(\d+) tokens in\s+([0-9.]+) seconds, speed:\s+([0-9.]+) t/s")


class LogError(Exception):
    pass


def parse_log(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = path.stem.split("_")
    has_p_min = len(parts) >= 4 and re.fullmatch(r"p\d+", parts[2]) is not None
    rec: dict = {"arm": path.stem}
    for name in INT_FIELDS:
        m = re.search(LOG_PREFIX + r"%s\s*= (\d+)\s*$" % name, text, re.M)
        if not m:
            raise LogError("%s: no '%s' line (the run did not reach the speculative summary)" % (path.name, name))
        rec[name] = int(m.group(1))
    if rec["n_drafted"] == 0:
        raise LogError("%s: no draft tokens were generated (n_drafted = 0); speculation was not active" % path.name)
    if rec["n_draft"] <= 0:
        raise LogError("%s: n_draft = %d" % (path.name, rec["n_draft"]))

    rec["accept_rate"] = rec["n_accept"] / rec["n_drafted"]

    stats = None
    for m in STATS_RE.finditer(text):
        stats = m
    if stats:
        rec["mean_len"] = float(stats.group(1))
        rec["mean_len_source"] = "stats"
        pos = stats.group(2)
        rec["acc_rate_per_pos"] = [float(x) for x in pos.split(",")] if pos else None
    elif has_p_min:
        # a p-min filter ends drafts early, so rounds no longer draft exactly n_draft tokens
        raise LogError("%s: a p-min arm needs the '#mean acc len' statistics line (run with -lv 4)" % path.name)
    else:
        rounds = rec["n_drafted"] / rec["n_draft"]
        rec["mean_len"] = rec["n_predict"] / rounds
        rec["mean_len_source"] = "derived"
        rec["acc_rate_per_pos"] = None

    dec = DECODE_RE.search(text)
    rec["decode_tps"] = float(dec.group(3)) if dec else None

    rec["head"] = parts[0]
    rec["p_min"] = parts[2] if has_p_min else "none"
    return rec


def parse_baseline(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in BASE_EVAL_RE.finditer(text):
        pass
    if m is None:
        raise LogError("%s: no 'eval time' line (the run did not finish)" % path.name)
    ms, runs = float(m.group(1)), int(m.group(2))
    # a speculative run prints the same line for 1 run and 0.00 ms; that is not a decode measurement
    if runs <= 1 or ms <= 0.0 or re.search(LOG_PREFIX + r"n_drafted\s*= \d+", text, re.M):
        raise LogError("%s: not a no-MTP baseline (eval time covers %d run(s) in %.2f ms; "
                       "pass the llama-completion log)" % (path.name, runs, ms))
    return {"runs": runs, "eval_ms": ms, "tps": runs / (ms / 1000.0)}


def parse_pairs(directory: pathlib.Path) -> tuple[list[dict], list[str]]:
    pairs, errors = [], []
    bases = sorted(p for p in directory.glob("base_*.log") if PAIR_NAME_RE.match(p.name))
    if not bases:
        errors.append("%s: no base_<prompt>_<n>.log files" % directory)
    for base_path in bases:
        name = PAIR_NAME_RE.match(base_path.name)
        mtp_path = directory / ("mtp_%s_%s.log" % (name.group("prompt"), name.group("idx")))
        try:
            if not mtp_path.exists():
                raise LogError("%s: no %s to pair with" % (base_path.name, mtp_path.name))
            base = parse_baseline(base_path)
            mtp = parse_log(mtp_path)
            if mtp["decode_tps"] is None:
                raise LogError("%s: no 'decoded ... speed' line" % mtp_path.name)
        except LogError as e:
            errors.append(str(e))
            continue
        pairs.append({
            "prompt": name.group("prompt"),
            "pair": int(name.group("idx")),
            "base_tps": base["tps"],
            "mtp_tps": mtp["decode_tps"],
            "speedup": mtp["decode_tps"] / base["tps"],
            "n_draft": mtp["n_draft"],
            "accept_rate": mtp["accept_rate"],
            "mean_len": mtp["mean_len"],
        })
    return pairs, errors


def render_pairs(pairs: list[dict]) -> str:
    rows = ["%-10s %5s %10s %8s %9s %8s %8s" %
            ("prompt", "pair", "base_t/s", "mtp_t/s", "speedup", "accept%", "mean_len")]
    for r in pairs:
        rows.append("%-10s %5d %10.3f %8.3f %8.2fx %8.2f %8.2f" %
                    (r["prompt"], r["pair"], r["base_tps"], r["mtp_tps"], r["speedup"],
                     100.0 * r["accept_rate"], r["mean_len"]))
    return "\n".join(rows)


def aggregate(arms: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for rec in arms:
        groups.setdefault((rec["head"], rec["n_draft"], rec["p_min"]), []).append(rec)
    out = []
    for (head, n_draft, p_min), recs in sorted(groups.items()):
        drafted = sum(r["n_drafted"] for r in recs)
        accepted = sum(r["n_accept"] for r in recs)
        out.append({
            "head": head,
            "n_draft": n_draft,
            "p_min": p_min,
            "arms": len(recs),
            "accept_rate": accepted / drafted,
            "mean_len": sum(r["mean_len"] for r in recs) / len(recs),
        })
    return out


def render(arms: list[dict], groups: list[dict]) -> str:
    rows = ["%-24s %5s %9s %9s %8s %8s %-8s %-22s %9s" %
            ("arm", "n_max", "n_drafted", "n_accept", "accept%", "mean_len", "source", "acc/pos", "decode_ts")]
    for r in arms:
        pos = ",".join("%.2f" % x for x in r["acc_rate_per_pos"]) if r["acc_rate_per_pos"] else "-"
        tps = "%.2f" % r["decode_tps"] if r["decode_tps"] is not None else "-"
        rows.append("%-24s %5d %9d %9d %8.2f %8.2f %-8s %-22s %9s" %
                    (r["arm"], r["n_draft"], r["n_drafted"], r["n_accept"], 100.0 * r["accept_rate"],
                     r["mean_len"], r["mean_len_source"], pos, tps))
    rows.append("")
    rows.append("%-24s %5s %6s %5s %8s %8s" % ("head", "n_max", "p_min", "arms", "accept%", "mean_len"))
    for g in groups:
        rows.append("%-24s %5d %6s %5d %8.2f %8.2f" %
                    (g["head"], g["n_draft"], g["p_min"], g["arms"], 100.0 * g["accept_rate"], g["mean_len"]))
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", type=pathlib.Path)
    ap.add_argument("--pairs", type=pathlib.Path, metavar="DIR",
                    help="directory of base_<prompt>_<n>.log / mtp_<prompt>_<n>.log pairs")
    ap.add_argument("--json", action="store_true", help="emit {arms, groups, pairs, errors} as JSON")
    args = ap.parse_args(argv)
    if not args.logs and args.pairs is None:
        ap.error("give arm logs, --pairs DIR, or both")

    arms, errors = [], []
    for path in args.logs:
        try:
            arms.append(parse_log(path))
        except (LogError, OSError) as e:
            errors.append(str(e))
    pairs: list[dict] = []
    if args.pairs is not None:
        pairs, pair_errors = parse_pairs(args.pairs)
        errors += pair_errors

    groups = aggregate(arms) if arms else []
    if args.json:
        print(json.dumps({"arms": arms, "groups": groups, "pairs": pairs, "errors": errors}, indent=2))
    else:
        if arms:
            print(render(arms, groups))
        if pairs:
            if arms:
                print()
            print(render_pairs(pairs))
        for e in errors:
            print("ERROR: %s" % e)
    if errors:
        if args.json:
            for e in errors:
                print("ERROR: %s" % e, file=sys.stderr)
        # the arms that did parse are still listed above; the run as a whole failed
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
