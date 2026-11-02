#!/usr/bin/env python3
"""Resolve the module+0xOFFSET frames of hostmem-bigalloc-probe output.

usage: hostmem-bigalloc-resolve.py run.err [--frames N]

Prints one block per [BIGALLOC] record with function names (addr2line -f -C)
for the first N frames (default 10), then a summary of how many allocations and
how many bytes each distinct "first project frame" accounts for. A frame is a
"project frame" when its module is not libc/libstdc++/libgcc/ld.

Run it on the same host and tree as the probed run: offsets are looked up with
addr2line in the libraries actually loaded. See the caveats in
hostmem-bigalloc-probe.cpp (frame skipping, no valloc/pvalloc/mmap coverage,
non-PIE offsets); a negative result there is not proof of absence.
"""
import re, subprocess, sys, collections

args = sys.argv[1:]
nframes = 10
if "--frames" in args:
    i = args.index("--frames")
    nframes = int(args[i + 1])
    del args[i:i + 2]
if not args:
    sys.exit(__doc__)

rec_re = re.compile(r"^\[BIGALLOC\] fn=(\S+) size=(\d+) MiB=([\d.]+) align=(\d+) tid=(\d+)")
frm_re = re.compile(r"^\s+#(\d+) (\S+)\+0x([0-9a-f]+)$")
records = []
cur = None
for line in open(args[0], errors="replace"):
    m = rec_re.match(line)
    if m:
        cur = {"fn": m.group(1), "size": int(m.group(2)), "frames": []}
        continue
    if line.startswith("[BIGALLOC-END]"):
        if cur:
            records.append(cur)
        cur = None
        continue
    m = frm_re.match(line)
    if m and cur is not None:
        cur["frames"].append((m.group(2), int(m.group(3), 16)))

cache = {}
def resolve(mod, off):
    key = (mod, off)
    if key not in cache:
        try:
            # every frame is a RETURN address: look up the call instruction, not the one after it
            out = subprocess.run(["addr2line", "-f", "-C", "-e", mod, hex(off - 1)], capture_output=True,
                                 text=True, timeout=60).stdout.split("\n")
            cache[key] = out[0] + (" " + out[1] if len(out) > 1 and not out[1].startswith("??") else "")
        except Exception as e:  # missing module on this host, etc.
            cache[key] = "?? (%s)" % e
    return cache[key]

system = re.compile(r"(/libc\.|/libstdc\+\+|/libgcc_s|/ld-linux|/libm\.|hostmem_bigalloc)")
summary = collections.OrderedDict()
for r in records:
    print("[BIGALLOC] %s %.1f MiB" % (r["fn"], r["size"] / 1048576.0))
    project = []
    for i, (mod, off) in enumerate(r["frames"]):
        name = resolve(mod, off)
        if i < nframes:
            print("   #%d %s  [%s+0x%x]" % (i, name, mod.rsplit("/", 1)[-1], off))
        short = name.split(" /")[0]
        # container plumbing (std::allocator, operator new) names no owner
        if not system.search(mod) and not short.startswith(("std::", "operator new", "__gnu_cxx::")):
            project.append(short)
    key = " <- ".join(project[:2]) or "?"
    cnt, tot = summary.get(key, (0, 0))
    summary[key] = (cnt + 1, tot + r["size"])
print("\nSUMMARY by first two project frames (count, total GiB):")
for k, (c, t) in sorted(summary.items(), key=lambda kv: -kv[1][1]):
    print("  %4d  %7.2f GiB  %s" % (c, t / 2**30, k))
print("  total records: %d, %.2f GiB" % (len(records), sum(r["size"] for r in records) / 2**30))
