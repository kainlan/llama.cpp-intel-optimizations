#!/usr/bin/env python3
"""Per-commit values must stay off every edge command that does not need them.

Ninja reruns an edge when its command changes, so an edge whose command carries
a value that moves with every commit reruns at every commit. The commit id used
to be a target-wide definition of ggml-base (GGML_COMMIT) and llama
(LLAMA_COMMIT), so it was on ~194 compile commands, each of which reran -- and,
through ccache, missed -- at every commit (llama.cpp-vuy0).

This checks edge COMMANDS only; it does not count every per-commit rebuild.
Values that reach a file's generated contents rebuild it without touching any
command: common/build-info.cpp carries the commit id and build number that way,
and the UI asset step's output (ui.cpp, ui.h) changes with the build number it
is given. Measured after one commit (ninja -n, dev targets): five compile or
generate steps -- ggml.c, build-info.cpp, the UI asset step, ui.cpp and
server-http.cpp -- plus the relinks behind them.

Reads build.ninja and requires:
  1. the commit id, from the GGML_COMMIT define, on exactly one edge, which is a
     compile (ggml.c);
  2. the build number, from the LLAMA_BUILD_NUMBER the UI asset step is given
     (else from common/build-info.cpp), on no edge but that step.
Exit 77 when there is no build.ninja, when the build has no git commit, or when
it does not compile ggml-base (LLAMA_USE_SYSTEM_GGML).
"""

import os
import re
import sys

COMMIT_DEFINE = re.compile(r'-DGGML_COMMIT=\\"([^"\\]+)\\"')
COMPILE_RULE = re.compile(r"^C(XX)?_COMPILER__")
BUILD_NUMBER_DEFINE = re.compile(r"-DLLAMA_BUILD_NUMBER=([0-9]+)")
BUILD_NUMBER_SOURCE = re.compile(r"^int LLAMA_BUILD_NUMBER = ([0-9]+);", re.M)
# The one edge that needs the build number: it names the UI assets to fetch.
BUILD_NUMBER_EDGES = ("tools/ui/ui-assets.stamp",)


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
    if not any(COMPILE_RULE.match(rule) and any("/ggml-base.dir/" in o for o in outs)
               for outs, rule, _ in edges):
        print("SKIP: the build does not compile ggml-base (LLAMA_USE_SYSTEM_GGML?)")
        return 77
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
    ok = True
    if len(carriers) != 1 or len(compiles) != 1:
        print(f"FAIL: commit {commit} is on {len(carriers)} edges ({len(compiles)} compiles); "
              "want exactly one compile edge")
        ok = False

    numbers = {m.group(1) for _, _, text in edges for m in BUILD_NUMBER_DEFINE.finditer(text)}
    if not numbers:
        build_info = os.path.join(build_dir, "common", "build-info.cpp")
        if os.path.isfile(build_info):
            with open(build_info, encoding="utf-8", errors="replace") as f:
                numbers = set(BUILD_NUMBER_SOURCE.findall(f.read()))
    numbers.discard("0")  # no git: nothing moves per commit
    if not numbers:
        print("  build number: not rendered anywhere, not checked")
    for number in sorted(numbers):
        token = re.compile(r"(?<![0-9])" + number + r"(?![0-9])")
        stray = [outs for outs, _, text in edges
                 if token.search(text) and not any(o in BUILD_NUMBER_EDGES for o in outs)]
        for outs in stray:
            print(f"  build number {number} on {' '.join(outs)}")
        if stray:
            print(f"FAIL: build number {number} is on {len(stray)} edges besides {BUILD_NUMBER_EDGES}")
            ok = False
    if not ok:
        return 1
    print("test-build-info-scope: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
