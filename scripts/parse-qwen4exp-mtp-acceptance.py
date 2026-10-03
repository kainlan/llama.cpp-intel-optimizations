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
STATS_RE = re.compile(r"#mean acc len = ([0-9.]+)(?:, #acc rate/pos = \(([^)]*)\))?")
DECODE_RE = re.compile(r"decoded\s+(\d+) tokens in\s+([0-9.]+) seconds, speed:\s+([0-9.]+) t/s")


class LogError(Exception):
    pass


def parse_log(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    rec: dict = {"arm": path.stem}
    for name in INT_FIELDS:
        m = re.search(r"^%s\s*= (\d+)\s*$" % name, text, re.M)
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
    elif path.stem.split("_")[2:3] and re.fullmatch(r"p\d+", path.stem.split("_")[2]) and len(path.stem.split("_")) >= 4:
        # a p-min filter ends drafts early, so rounds no longer draft exactly n_draft tokens
        raise LogError("%s: a p-min arm needs the '#mean acc len' statistics line (run with -lv 4)" % path.name)
    else:
        rounds = rec["n_drafted"] / rec["n_draft"]
        rec["mean_len"] = rec["n_predict"] / rounds
        rec["mean_len_source"] = "derived"
        rec["acc_rate_per_pos"] = None

    dec = DECODE_RE.search(text)
    rec["decode_tps"] = float(dec.group(3)) if dec else None

    parts = path.stem.split("_")
    rec["head"] = parts[0]
    rec["p_min"] = parts[2] if len(parts) >= 4 and re.fullmatch(r"p\d+", parts[2]) else "none"
    return rec


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
    ap.add_argument("logs", nargs="+", type=pathlib.Path)
    ap.add_argument("--json", action="store_true", help="emit {arms, groups} as JSON")
    args = ap.parse_args(argv)

    arms, errors = [], []
    for path in args.logs:
        try:
            arms.append(parse_log(path))
        except (LogError, OSError) as e:
            errors.append(str(e))

    groups = aggregate(arms) if arms else []
    if args.json:
        print(json.dumps({"arms": arms, "groups": groups, "errors": errors}, indent=2))
    else:
        if arms:
            print(render(arms, groups))
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
