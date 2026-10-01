#!/usr/bin/env python3
"""Regenerate appendix-rows.json (clause (m) of check-sycl-alloc-zone-contract.py) from the 23mk design's census table.

    python3 scripts/sycl-alloc-zone-contract/gen-appendix-rows.py --design DESIGN.md --master 2c4f5e45d --rev 4.16 \\
        [--out scripts/sycl-alloc-zone-contract/appendix-rows.json]

The table is the appendix whose header row starts `| # | site (master`. Columns read: row, site, function (AST), zone today,
code, core zone, exact term. Backticks and `**` are stripped. The file is rewritten whole; never hand-edit it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

DOC = ("Read by scripts/check-sycl-alloc-zone-contract.py clause (m) (the SCRATCH floor list). The census table of the 23mk "
       "design's appendix (master %s, rev %s), reduced to the columns the clause reads: row, site (as at that master), function, "
       "zone today, code (the reach class: R, F, T, B, H, X, D, REF, ZH...), core zone, exact term. The design is not in the "
       "repository, so this table is the clause's input; regenerate it with scripts/sycl-alloc-zone-contract/gen-appendix-rows.py, "
       "with the SCRATCH_FLOOR_CONSUMERS table in the gate, in the commit that changes a row's zone or term.")


def cell(s):
    return re.sub(r"\*\*", "", s.strip().strip("`").strip()).strip("`").strip().replace("\\|", "|")


def parse(text):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("| # | site (master")), None)
    if start is None:
        sys.exit("gen-appendix-rows: no census table header found")
    rows = []
    for l in lines[start + 2:]:
        if not l.startswith("|"):
            break
        c = [x for x in re.split(r"(?<!\\)\|", l.strip())[1:-1]]
        rows.append({"row": int(c[0]), "site": cell(c[1]), "function": cell(c[2]), "zone": cell(c[3]), "code": cell(c[5]),
                     "core": cell(c[8]), "term": cell(c[9])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "appendix-rows.json"))
    a = ap.parse_args()
    rows = parse(Path(a.design).read_text())
    body = ",\n".join("    " + json.dumps(r) for r in rows)
    Path(a.out).write_text('{\n  "schema": 1,\n  "_doc": %s,\n  "rows": [\n%s\n  ]\n}\n' % (json.dumps(DOC % (a.master, a.rev)), body))
    print("wrote %d rows to %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
