#!/usr/bin/env python3
"""Every SYCL device link must run under the device-link launcher and pool.

llama.cpp-vuy0 routes libggml-sycl and every executable that embeds the
backend objects through ggml/src/ggml-sycl/sycl-device-link.sh (private
TMPDIR, toolchain-keyed ocloc cache) and the `ggml_sycl_device_link` Ninja
pool. Test executables reach the backend objects through an INTERFACE target,
so CMake routes them in a deferred pass; this reads the generated Ninja files
and fails if any device-linking edge escaped that pass -- for instance a new
consumer that links the objects some other way.

A link edge is device-linking when its link flags carry -fsycl-targets or its
inputs include ggml-sycl backend objects. Each such edge must also carry the
link options the speed-up rests on:
  - -fsycl-max-parallel-link-jobs=<GGML_SYCL_DEVICE_LINK_JOBS> when that is > 1;
  - -allow_caching for both BMG targets exactly when its launcher runs with
    --ocloc-cache. Without the launcher's NEO environment the flag caches
    nothing and leaves an empty ./ocloc_cache behind; without the flag the
    launcher's cache is never consulted.
Exit 77 when the build directory is not a Ninja SYCL build.
"""

import os
import re
import sys

POOL = "ggml_sycl_device_link"
ALLOW_CACHING = tuple(f"-Xsycl-target-backend=intel_gpu_bmg_{t} -allow_caching" for t in ("g21", "g31"))
LAUNCHER = "sycl-device-link.sh"
BACKEND_OBJECT_DIRS = ("/ggml-sycl.dir/", "/ggml-sycl-q1-route-test-objects.dir/")


def parse_ninja(path):
    """Return (rules, edges): rule name -> command, and a list of edges."""
    rules = {}
    edges = []
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("rule "):
                current = {"kind": "rule", "name": line[5:].strip(), "vars": {}}
                rules[current["name"]] = current
            elif line.startswith("build "):
                head = line[6:]
                outputs, _, rest = head.partition(": ")
                parts = rest.split()
                current = {
                    "kind": "build",
                    "outputs": outputs.split(),
                    "rule": parts[0] if parts else "",
                    "inputs": parts[1:],
                    "vars": {},
                }
                edges.append(current)
            elif line.startswith("  ") and current is not None and " = " in line:
                key, _, value = line.strip().partition(" = ")
                current["vars"][key] = value
            elif not line.startswith(" "):
                current = None
    return {name: r["vars"].get("command", "") for name, r in rules.items()}, edges


def is_device_link(edge):
    if "LINKER" not in edge["rule"]:
        return False
    if "-fsycl-targets=" in edge["vars"].get("LINK_FLAGS", ""):
        return True
    return any(d in inp for inp in edge["inputs"] for d in BACKEND_OBJECT_DIRS)


def cmake_cache(build_dir):
    values = {}
    path = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", line.rstrip("\n"))
                if m:
                    values[m.group(1)] = m.group(2)
    return values


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    build_ninja = os.path.join(build_dir, "build.ninja")
    rules_ninja = os.path.join(build_dir, "CMakeFiles", "rules.ninja")
    if not (os.path.isfile(build_ninja) and os.path.isfile(rules_ninja)):
        print(f"SKIP: {build_dir} is not a Ninja build directory")
        return 77

    rules, edges = parse_ninja(rules_ninja)
    more_rules, edges = parse_ninja(build_ninja)
    rules.update(more_rules)

    device_links = [e for e in edges if is_device_link(e)]
    if not device_links:
        print(f"SKIP: no SYCL device-link edges in {build_ninja}")
        return 77

    cache = cmake_cache(build_dir)
    try:
        jobs = int(cache.get("GGML_SYCL_DEVICE_LINK_JOBS", "1"))
    except ValueError:
        jobs = 1

    bad = []
    # The object-directory criterion fails open on a rename, so each directory
    # it names must exist. The Q1 route objects are built for tests only, and
    # not under GGML_BACKEND_DL.
    def on(name, default):
        return cache.get(name, default).upper() in ("ON", "1", "TRUE", "YES", "Y")
    with open(build_ninja, encoding="utf-8", errors="replace") as f:
        ninja_text = f.read()
    for obj_dir in BACKEND_OBJECT_DIRS:
        if obj_dir == "/ggml-sycl-q1-route-test-objects.dir/" and (
                not on("BUILD_TESTING", "ON") or on("GGML_BACKEND_DL", "OFF")):
            continue
        if obj_dir not in ninja_text:
            bad.append(f"no {obj_dir} in build.ninja: the backend-object criterion names a directory that does not exist")
    for edge in device_links:
        target = edge["outputs"][0]
        if edge["vars"].get("pool") != POOL:
            bad.append(f"{target}: pool is {edge['vars'].get('pool', '<none>')!r}, want {POOL!r}")
        command = rules.get(edge["rule"], "")
        if not re.search(r"\S*" + re.escape(LAUNCHER) + r"\b", command):
            bad.append(f"{target}: rule {edge['rule']} does not run {LAUNCHER}")
        flags = edge["vars"].get("LINK_FLAGS", "")
        if jobs > 1 and f"-fsycl-max-parallel-link-jobs={jobs}" not in flags:
            bad.append(f"{target}: missing -fsycl-max-parallel-link-jobs={jobs}")
        cached = "--ocloc-cache" in command
        caching_flags = [f for f in ALLOW_CACHING if f in flags]
        if cached and len(caching_flags) != len(ALLOW_CACHING):
            bad.append(f"{target}: launcher runs --ocloc-cache but the link lacks -allow_caching for both BMG targets")
        if caching_flags and not cached:
            bad.append(f"{target}: -allow_caching without the launcher's --ocloc-cache environment")

    pooled = [e for e in edges if e["vars"].get("pool") == POOL]
    stray = [e["outputs"][0] for e in pooled if not is_device_link(e)]
    for target in stray:
        bad.append(f"{target}: in pool {POOL} but is not a SYCL device link")

    print(f"{len(device_links)} device-link edges, {len(pooled)} pooled")
    if bad:
        for line in bad:
            print("FAIL: " + line)
        return 1
    print("test-sycl-device-link-routing: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
