#!/usr/bin/env python3
"""Source-reading tests must not depend on an absolute __FILE__.

scripts/sycl-build.sh runs ccache with base_dir (llama.cpp-vuy0), which
rewrites each source path -- and so __FILE__ -- relative to the build
directory. A test that recovers the repo root from __FILE__ then finds it only
when run from a directory that happens to line up, and these binaries are run
from anywhere (some are install()ed). Each such locator therefore also takes
the absolute LLAMA_CPP_SOURCE_ROOT the build defines; ccache leaves -D values
alone.

Checks, for every C++ source whose candidate_roots() anchors on __FILE__:
  1. the locator consults LLAMA_CPP_SOURCE_ROOT before the __FILE__ anchor;
  2. given a Ninja build directory, every compile of that source defines
     LLAMA_CPP_SOURCE_ROOT as the absolute repo root.
Check 2 is skipped for sources the build does not compile (for example under
GGML_BACKEND_DL); exit 77 if the build compiles none of them.
"""

import os
import re
import subprocess
import sys

ANCHOR = "const std::string source_file = __FILE__;"
GUARD = re.compile(r"^#\s*ifdef\s+LLAMA_CPP_SOURCE_ROOT\b", re.MULTILINE)


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def anchored_sources(root):
    out = subprocess.run(["git", "-C", root, "grep", "-l", "-F", ANCHOR, "--", "*.cpp", "*.hpp", "*.h"],
                         capture_output=True, text=True, check=False)
    if out.returncode not in (0, 1):
        raise SystemExit(f"git grep failed: {out.stderr.strip()}")
    return sorted(line for line in out.stdout.splitlines() if line)


def check_source(root, rel):
    text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
    match = GUARD.search(text)
    guard = match.start() if match else -1
    anchor = text.find(ANCHOR)
    if guard < 0:
        return f"{rel}: candidate_roots() anchors on __FILE__ but never consults LLAMA_CPP_SOURCE_ROOT"
    if guard > anchor:
        return f"{rel}: LLAMA_CPP_SOURCE_ROOT is consulted after the __FILE__ anchor"
    if text.count(ANCHOR) != 1:
        return f"{rel}: expected one __FILE__ anchor, found {text.count(ANCHOR)}"
    return None


def compile_defines(build_dir):
    """Map absolute source path -> list of DEFINES strings of its compile edges."""
    result = {}
    current = None
    with open(os.path.join(build_dir, "build.ninja"), encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("build "):
                current = None
                _, _, rest = line[6:].partition(": ")
                parts = rest.split()
                if len(parts) >= 2 and "COMPILER" in parts[0]:
                    current = os.path.normpath(os.path.join(build_dir, parts[1].replace("$:", ":")))
                    result.setdefault(current, [])
            elif current and line.startswith("  DEFINES = "):
                result[current].append(line[len("  DEFINES = "):])
    return result


def main():
    root = repo_root()
    sources = anchored_sources(root)
    if not sources:
        print("FAIL: found no __FILE__-anchored locator -- this check has lost its anchor")
        return 1

    problems = [p for p in (check_source(root, rel) for rel in sources) if p]

    checked = 0
    if len(sys.argv) > 1 and os.path.isfile(os.path.join(sys.argv[1], "build.ninja")):
        defines = compile_defines(sys.argv[1])
        want = re.compile(r'-DLLAMA_CPP_SOURCE_ROOT=\\"' + re.escape(root) + r'/?\\"(\s|$)')
        for rel in sources:
            edges = defines.get(os.path.join(root, rel))
            if edges is None:
                continue
            for d in edges:
                checked += 1
                if not want.search(d):
                    problems.append(f"{rel}: compiled without -DLLAMA_CPP_SOURCE_ROOT=\"{root}\"")
        if checked == 0 and not problems:
            print(f"SKIP: {sys.argv[1]} compiles none of the {len(sources)} anchored sources")
            return 77

    print(f"{len(sources)} anchored sources, {checked} compile edges checked")
    if problems:
        for p in problems:
            print("FAIL: " + p)
        return 1
    print("test-sycl-source-root-anchor: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
