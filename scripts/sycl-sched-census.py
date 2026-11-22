#!/usr/bin/env python3
"""Count scheduler splits and ops per backend in a GGML_SCHED_DEBUG=2 log (llama-completion needs -v to print it).

A graph starts at its "SPLIT #0" line. "largest" is by op-node count, which excludes view ops because the scheduler
does not print them. Node lines carry no backend, so each node counts for the backend of the split it sits in. A split
index that neither restarts at 0 nor follows the previous one is reported as an anomaly (a torn line).
"""
import argparse
import collections
import json
import re
import sys

# ggml-backend.cpp prints "## SPLIT #<i>: <backend> # <n> inputs" and "node #%3d (%10.10s): ...": the op name is
# right-aligned and cut to 10 characters, so FLASH_ATTN_EXT appears as FLASH_ATTN. Threads interleave and tear lines
# in the middle, so a token can sit mid-line: scan the whole text and keep the tokens in position order. The node
# pattern pads with spaces only, so it cannot join a torn line to the next one across a newline.
TOKEN_RE = re.compile(r"## SPLIT #(?P<split>\d+): (?P<backend>\S+) # \d+ inputs|node # *\d+ \( *(?P<op>[A-Z0-9_]+)\)")


def parse(text):
    """(graphs, anomalies): each graph is {"splits": [backend, ...], "ops": Counter((backend, op))}."""
    graphs = []
    backend = None
    prev = None
    anomalies = 0
    for m in TOKEN_RE.finditer(text):
        if m.group("split") is not None:
            idx = int(m.group("split"))
            if prev is not None and idx != 0 and idx != prev + 1:
                anomalies += 1
            prev = idx
            if idx == 0 or not graphs:
                graphs.append({"splits": [], "ops": collections.Counter()})
            backend = m.group("backend")
            graphs[-1]["splits"].append(backend)
        elif graphs and backend is not None:
            graphs[-1]["ops"][(backend, m.group("op"))] += 1
    return graphs, anomalies


def summarize(graph):
    per_backend = collections.Counter(graph["splits"])
    return {
        "nodes": sum(graph["ops"].values()),
        "splits": len(graph["splits"]),
        "splits_by_backend": dict(per_backend),
        "ops": [{"backend": b, "op": op, "count": n}
                for (b, op), n in sorted(graph["ops"].items(), key=lambda kv: (-kv[1], kv[0]))],
    }


def render(title, s):
    lines = [f"{title}: {s['nodes']} nodes, {s['splits']} splits "
             f"({', '.join(f'{b} {n}' for b, n in sorted(s['splits_by_backend'].items()))})"]
    width = max((len(o["backend"]) for o in s["ops"]), default=0)
    lines += [f"  {o['count']:6d}  {o['backend']:<{width}} {o['op']}" for o in s["ops"]]
    return "\n".join(lines)


def self_test():
    text = ("## SPLIT #0: SYCL0 # 1 inputs\nnode #  0 (   MUL_MAT): a\nno## SPLIT #1: CPU # 2 inputsde #  1 (     SCALE): b\n"
            "node #  2 (FLASH_ATTN): c\n## SPLIT #2: SYCL0 # 1 inputs\nnode #  3 (       ADD): d\n"
            "## SPLIT #0: CPU # 0 inputs\nnode #  0 (  GET_ROWS): e\n")
    graphs, anomalies = parse(text)
    assert anomalies == 0, anomalies
    assert len(graphs) == 2, graphs
    first = summarize(graphs[0])
    assert first["nodes"] == 3 and first["splits"] == 3, first
    assert first["splits_by_backend"] == {"SYCL0": 2, "CPU": 1}, first
    assert {"backend": "CPU", "op": "FLASH_ATTN", "count": 1} in first["ops"], first
    assert summarize(graphs[1])["splits_by_backend"] == {"CPU": 1}
    assert parse("nothing here") == ([], 0)
    # A torn SPLIT #0 leaves only the next split of that graph, whose index does not follow the previous one.
    torn = text.replace("## SPLIT #0: CPU # 0 inputs", "## SPL garbage") + "## SPLIT #1: CPU # 0 inputs\n"
    torn_graphs, torn_anomalies = parse(torn)
    assert torn_anomalies == 1 and len(torn_graphs) == 1, (torn_anomalies, torn_graphs)
    assert parse("node #  4 (\n   MUL_MAT): x\n") == ([], 0)
    print("self-test: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test and args.json:
        ap.error("--self-test cannot be combined with --json")
    if args.self_test:
        self_test()
        return 0
    if not args.log:
        ap.error("a log file is required")
    try:
        with open(args.log, errors="replace") as fh:
            graphs, anomalies = parse(fh.read())
    except OSError as e:
        print(f"{args.log}: {e.strerror}", file=sys.stderr)
        return 2
    if not graphs:
        print("no scheduler splits found (was GGML_SCHED_DEBUG=2 set and -v passed?)", file=sys.stderr)
        return 2
    if anomalies:
        print(f"WARNING: {anomalies} split-index discontinuities (torn lines?); counts may be wrong", file=sys.stderr)
    largest = summarize(max(graphs, key=lambda g: sum(g["ops"].values())))
    last = summarize(graphs[-1])
    if args.json:
        print(json.dumps({"graphs": len(graphs), "anomalies": anomalies, "largest": largest, "last": last}, indent=1))
    else:
        print(f"{len(graphs)} graphs")
        print(render("largest graph", largest))
        print(render("last graph", last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
