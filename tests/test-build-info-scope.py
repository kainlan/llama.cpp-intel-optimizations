#!/usr/bin/env python3
"""A commit must change the command of exactly one compile edge.

The commit id reaches the build as a -D define. As a target-wide definition of
ggml-base (GGML_COMMIT) and llama (LLAMA_COMMIT) it was part of every compile
command of those targets, so each commit re-ran ~194 compiles and, through
ccache, missed on all of them (llama.cpp-vuy0). Ninja reruns an edge when its
command changes, and an edge whose command does not contain the id cannot
change when the id does -- so counting the edges that contain it counts the
compiles a commit costs.

Reads build.ninja, takes the configured commit from the GGML_COMMIT define, and
requires that exactly one edge mentions it and that the edge is a compile.
(common/build-info.cpp carries the commit in its generated contents, not its
command, so it is one more recompile per commit by design.)
Exit 77 when there is no build.ninja or the build has no git commit.
"""

import os
import re
import sys

COMMIT_DEFINE = re.compile(r'-DGGML_COMMIT=\\"([^"\\]+)\\"')
COMPILE_RULE = re.compile(r"^C(XX)?_COMPILER__")


def parse_edges(path):
    """Return a list of (outputs, rule, text) with text = the edge's full line and variables."""
    edges = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("build "):
                outputs, _, rest = line[6:].partition(": ")
                rule = rest.split()[0] if rest.split() else ""
                current = [outputs.split("|")[0].split(), rule, [line]]
                edges.append(current)
            elif line.startswith("  ") and current is not None:
                current[2].append(line)
            elif not line.startswith(" "):
                current = None
    return [(outs, rule, "\n".join(text)) for outs, rule, text in edges]


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    build_ninja = os.path.join(build_dir, "build.ninja")
    if not os.path.isfile(build_ninja):
        print(f"SKIP: no {build_ninja}")
        return 77

    edges = parse_edges(build_ninja)
    commits = sorted({m.group(1) for _, _, text in edges for m in COMMIT_DEFINE.finditer(text)})
    if not commits:
        print("FAIL: no edge defines GGML_COMMIT; ggml_commit() would not build")
        return 1
    if len(commits) > 1:
        print(f"FAIL: GGML_COMMIT has several values: {commits}")
        return 1
    commit = commits[0]
    if commit.endswith("-dirty"):
        commit = commit[: -len("-dirty")]
    if commit == "unknown":
        print("SKIP: build configured without a git commit")
        return 77

    carriers = [(outs, rule) for outs, rule, text in edges if commit in text]
    for outs, rule in carriers:
        print(f"  {commit} on {' '.join(outs)} ({rule})")
    compiles = [c for c in carriers if COMPILE_RULE.match(c[1])]
    if len(carriers) != 1 or len(compiles) != 1:
        print(f"FAIL: commit {commit} is on {len(carriers)} edges ({len(compiles)} compiles); "
              "want exactly one compile edge")
        return 1
    print("test-build-info-scope: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
