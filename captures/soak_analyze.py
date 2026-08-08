#!/usr/bin/env python3
"""Pair ZONE-RESET-AUDIT site rows with their largest_free lines and compare
captures. Usage: soak_analyze.py <capture> [<capture> ...]

Prints, per capture, one row per site: visits, graphs span, and
largest_free first/last/min. With >1 capture, also prints a delta table of
`min` against the FIRST capture given (the baseline), flagging any site whose
min drops more than the pre-registered 5% tolerance.
"""
import re
import sys

SITE = re.compile(r"^\[ZONE-RESET-AUDIT\] site=(\S+) visits=(\d+).*?graphs=\[([^\]]*)\]")
LF = re.compile(r"^\[ZONE-RESET-AUDIT\]\s+largest_free first=([\d.]+) MB last=([\d.]+) MB min=([\d.]+) MB")
TOLERANCE = 0.05


def parse(path):
    """site -> dict; largest_free belongs to the most recent site row."""
    out, current = {}, None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = SITE.match(line)
            if m:
                current = m.group(1)
                out.setdefault(current, {"visits": int(m.group(2)), "graphs": m.group(3)})
                continue
            m = LF.match(line)
            if m and current:
                out[current].update(first=float(m.group(1)), last=float(m.group(2)), min=float(m.group(3)))
    return out


def main(paths):
    parsed = [(p, parse(p)) for p in paths]
    for path, sites in parsed:
        print(f"\n=== {path}")
        if not sites:
            print("  NO SITE ROWS -- capture is VOID, not clean (see pre-registration)")
            continue
        for name, d in sorted(sites.items()):
            lf = ""
            if "min" in d:
                lf = f"  largest_free first={d['first']:.2f} last={d['last']:.2f} min={d['min']:.2f} MB"
            else:
                lf = "  (no largest_free reported)"
            print(f"  {name:<42} visits={d['visits']:<6} graphs=[{d['graphs']}]{lf}")

    if len(parsed) < 2:
        return
    base_path, base = parsed[0]
    print(f"\n=== min-delta vs baseline {base_path} (tolerance {TOLERANCE:.0%})")
    verdict_lines = []
    for path, sites in parsed[1:]:
        print(f"\n-- {path}")
        for name, d in sorted(sites.items()):
            if "min" not in d or name not in base or "min" not in base[name]:
                continue
            b, s = base[name]["min"], d["min"]
            drop = (b - s) / b if b else 0.0
            flag = "FLAG" if drop > TOLERANCE else "ok"
            if flag == "FLAG":
                verdict_lines.append(f"{path}: {name} min {b:.2f} -> {s:.2f} ({drop:.1%} drop)")
            print(f"  {name:<42} min {b:>10.2f} -> {s:>10.2f}  ({drop:+.1%})  {flag}")
    print("\n=== VERDICT INPUT")
    if verdict_lines:
        print("  sites exceeding tolerance:")
        for v in verdict_lines:
            print("   ", v)
    else:
        print("  no site min dropped more than tolerance vs baseline")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
