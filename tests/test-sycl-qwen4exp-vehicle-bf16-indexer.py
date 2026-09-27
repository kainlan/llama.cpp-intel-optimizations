#!/usr/bin/env python3
"""Gate for scripts/sycl-qwen4exp-vehicle-bf16-indexer.py (census doc, step b2).

The rewrite feeds the qwen4exp census vehicle, and the census cannot see tensor
types: a vehicle with F32 where BF16 or Q8_0 belongs scores its rows as agreeing.
So this plants tiny GGUFs and requires:
- the rewrite converts exactly the indexer projections, bit-exact to ggml's
  round-to-nearest-even, and leaves every other byte alone;
- it refuses (rc 1, no output, input untouched) an input that is its own output
  (the same path or a symlink), an all-F32 step-a file, an already-BF16 indexer
  and a file with no indexer;
- --verify accepts the result and refuses F32 indexers or F32 experts;
- a failure mid-write leaves an existing output untouched and no temp file.

Standard library only: the script runs through its own /usr/bin/python3 shebang,
because the ctest interpreter on this host (miniconda) cannot load numpy.
Exits 77 (skip) when /usr/bin/python3 has no numpy.
"""

import hashlib
import os
import resource
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "sycl-qwen4exp-vehicle-bf16-indexer.py"
PY     = "/usr/bin/python3"

F32, Q8_0, BF16 = 0, 8, 30
ALIGN = 32

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"FAIL: {msg}")


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors):
    """tensors: [(name, type, ne list, raw bytes, or an int size for a sparse zero hole)]"""
    kv = gguf_str("general.architecture") + struct.pack("<I", 8) + gguf_str("qwen4exp")
    # an empty string array, as the test-llama-archs fixture writes tokenizer.ggml.merges
    kv += gguf_str("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0)
    infos = b""
    offset = 0
    for name, qtype, ne, raw in tensors:
        n = raw if isinstance(raw, int) else len(raw)
        infos += gguf_str(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", qtype, offset)
        offset += n + (-n % ALIGN)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 2) + kv + infos)
        f.write(b"\0" * (-f.tell() % ALIGN))
        for _, _, _, raw in tensors:
            if isinstance(raw, int):
                f.seek(raw + (-raw % ALIGN), os.SEEK_CUR)
            else:
                f.write(raw + b"\0" * (-len(raw) % ALIGN))
        f.truncate()


def read_gguf(path):
    """{name: (type, ne, raw bytes)} for a file written like write_gguf's"""
    b = Path(path).read_bytes()
    _, _, n_t, n_kv = struct.unpack_from("<IIQQ", b, 0)
    o = 24

    def rstr(o):
        n, = struct.unpack_from("<Q", b, o)
        return b[o + 8:o + 8 + n].decode(), o + 8 + n

    for _ in range(n_kv):
        _, o = rstr(o)
        vt, = struct.unpack_from("<I", b, o)
        o += 4
        if vt == 8:
            _, o = rstr(o)
        else:
            assert vt == 9, vt
            et, n = struct.unpack_from("<IQ", b, o)
            assert et == 8 and n == 0
            o += 12
    infos = []
    for _ in range(n_t):
        name, o = rstr(o)
        nd, = struct.unpack_from("<I", b, o)
        ne = list(struct.unpack_from(f"<{nd}Q", b, o + 4))
        qt, off = struct.unpack_from("<IQ", b, o + 4 + 8 * nd)
        o += 4 + 8 * nd + 12
        infos.append((name, qt, ne, off))
    base = o + (-o % ALIGN)
    out = {}
    for name, qt, ne, off in infos:
        n = 1
        for d in ne:
            n *= d
        size = {F32: 4 * n, BF16: 2 * n, Q8_0: 34 * n // 32}[qt]
        out[name] = (qt, ne, b[base + off:base + off + size])
    return out


def f32_bits(vals):
    return b"".join(struct.pack("<I", v) for v in vals)


def bf16_ref(u):
    # ggml_compute_fp32_to_bf16 (ggml-impl.h): quiet NaN, else round to nearest even
    if (u & 0x7fffffff) > 0x7f800000:
        return (u >> 16) | 64
    return (u + (0x7fff + ((u >> 16) & 1))) >> 16


# ties to even both ways, NaN, inf, -0, a subnormal, max finite
Q_BITS = [0x3f808000, 0x3f818000, 0x3f808001, 0x3f807fff, 0x7f800001, 0xff800001,
          0x7f800000, 0x80000000, 0x007fffff, 0x7f7fffff] + [0x3f800000 + 977 * i for i in range(22)]
K_BITS = [0x40490fdb + 131 * i for i in range(8)]
EXPERT_Q8 = bytes(range(68))                 # 2 Q8_0 blocks; content is never read
EXPERT_F32 = f32_bits([0x3f800000] * 64)


def vehicle(indexer_type=F32, expert_type=Q8_0, with_indexer=True):
    t = [("blk.0.ffn_down_exps.weight", expert_type, [32, 2],
          EXPERT_Q8 if expert_type == Q8_0 else EXPERT_F32),
         ("blk.0.ffn_down_exps.scale", F32, [2], f32_bits([0x3f800000] * 2))]
    if with_indexer:
        def idx(bits):
            if indexer_type == F32:
                return f32_bits(bits)
            return b"".join(struct.pack("<H", bf16_ref(u)) for u in bits)
        t += [("blk.1.indexer.q_proj.weight", indexer_type, [8, 4], idx(Q_BITS)),
              ("blk.1.indexer.q_norm.weight", F32, [8], f32_bits(K_BITS)),
              ("blk.1.indexer.k_proj.weight", indexer_type, [8, 1], idx(K_BITS))]
    return t


def run(*args):
    return subprocess.run([str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def temps(d):
    return [p for p in os.listdir(d) if p.startswith(".bf16-indexer-")]


def refused(tag, r, src, before, dst=None):
    check(r.returncode == 1, f"{tag}: expected rc 1, got {r.returncode} ({r.stderr.strip()[-200:]})")
    check(sha(src) == before, f"{tag}: input was modified")
    if dst is not None:
        check(not os.path.lexists(dst) or os.path.islink(dst), f"{tag}: output written")
    check(not temps(os.path.dirname(src)), f"{tag}: temp file left behind")


def main():
    if subprocess.run([PY, "-c", "import numpy"], capture_output=True).returncode != 0:
        print(f"SKIP: {PY} cannot import numpy, so the script under test cannot run")
        return 77

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "q8.gguf")
        write_gguf(src, vehicle())
        before = sha(src)

        # GREEN: the rewrite
        out = os.path.join(d, "out.gguf")
        r = run(src, out)
        check(r.returncode == 0, f"rewrite: rc {r.returncode} ({r.stderr.strip()[-300:]})")
        check(sha(src) == before, "rewrite: input was modified")
        a, b = read_gguf(src), read_gguf(out)
        check(list(a) == list(b), "rewrite: tensor order or names changed")
        for name, (qt, ne, raw) in a.items():
            bq, bne, braw = b[name]
            check(bne == ne, f"{name}: shape changed")
            if name.endswith(("q_proj.weight", "k_proj.weight")):
                want = b"".join(struct.pack("<H", bf16_ref(u))
                                for u in struct.unpack(f"<{len(raw) // 4}I", raw))
                check(bq == BF16, f"{name}: type {bq}, expected BF16")
                check(braw == want, f"{name}: BF16 bits differ from ggml's rounding")
            else:
                check(bq == qt and braw == raw, f"{name}: not copied unchanged")
        check(run("--verify", out).returncode == 0, "--verify refused the rewrite's output")
        check(not temps(d), "rewrite: temp file left behind")

        # every case below gets its own directory and a freshly planted input,
        # so one defect cannot contaminate the next case's evidence
        def case(tag, **kw):
            cd = os.path.join(d, tag.replace(" ", "-"))
            os.mkdir(cd)
            p = os.path.join(cd, "in.gguf")
            write_gguf(p, vehicle(**kw))
            return p, sha(p), os.path.join(cd, "out.gguf")

        # RED: the output path is the input
        p, h, _ = case("dst == src")
        refused("dst == src", run(p, p), p, h)
        p, h, _ = case("dst symlinks to src")
        link = os.path.join(os.path.dirname(p), "link.gguf")
        os.symlink(p, link)
        refused("dst symlinks to src", run(p, link), p, h, link)

        # RED: wrongly typed inputs, each refused before any output exists
        for tag, kw in (("all-F32 step-a file", dict(expert_type=F32)),
                        ("already-BF16 indexer", dict(indexer_type=BF16)),
                        ("no indexer", dict(with_indexer=False))):
            p, h, o = case(tag, **kw)
            refused(tag, run(p, o), p, h, o)

        # RED: --verify on the step-b file and on BF16 indexers over F32 experts
        p, _, _ = case("verify step-b file")
        check(run("--verify", p).returncode == 1, "--verify accepted F32 indexers")
        p, _, _ = case("verify f32 experts", indexer_type=BF16, expert_type=F32)
        r = run("--verify", p)
        check(r.returncode == 1 and "ffn_down_exps" in r.stderr, "--verify accepted F32 experts")

        # a failure mid-write (a file-size limit makes write() fail with EFBIG
        # partway through, as ENOSPC would) keeps an existing output, no temp file
        p, _, keep = case("mid-write")
        Path(keep).write_bytes(b"previous output")

        def small_files():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (256, 256))

        check(os.path.getsize(out) > 256, "mid-write: the output is too small to hit the limit")
        r = subprocess.run([str(SCRIPT), p, keep], capture_output=True, text=True, preexec_fn=small_files,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        check(r.returncode != 0 and "File too large" in r.stderr, f"mid-write: write did not fail ({r.stderr[-200:]})")
        check(Path(keep).read_bytes() == b"previous output", "mid-write: existing output replaced")
        check(not temps(os.path.dirname(keep)), "mid-write: temp file left behind")

        # RED: a wrongly typed input is refused before any tensor data is read.
        # A 256 MiB sparse tensor ahead of a BF16 indexer costs nothing unless
        # the script copies it, so the child's peak RSS shows when it refused.
        cd = os.path.join(d, "early-refusal")
        os.mkdir(cd)
        p = os.path.join(cd, "in.gguf")
        big = 256 << 20
        write_gguf(p, [("blk.0.ffn_down_exps.weight", Q8_0, [32, big // 34], big // 34 * 34)]
                   + vehicle(indexer_type=BF16)[2:])
        code = ("import resource, subprocess, sys\n"
                "r = subprocess.run(sys.argv[1:], capture_output=True)\n"
                "print(r.returncode, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n")
        r = subprocess.run([sys.executable, "-c", code, str(SCRIPT), p, os.path.join(cd, "out.gguf")],
                           capture_output=True, text=True)
        rc, rss_kib = map(int, r.stdout.split())
        check(rc == 1, f"early refusal: expected rc 1, got {rc}")
        check(rss_kib < (big >> 10) // 2, f"early refusal: peak RSS {rss_kib} KiB, so tensor data was read first")

    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("PASS: qwen4exp vehicle BF16 indexer rewrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
