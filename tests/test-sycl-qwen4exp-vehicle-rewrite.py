#!/usr/bin/env python3
"""Gate for scripts/sycl-qwen4exp-vehicle-rewrite.py (census doc, step b2).

The rewrite feeds the qwen4exp census vehicle, and the census cannot see tensor
types or which tensors exist: a vehicle whose tensors differ from the real
model's scores its rows as agreeing on paths the real model never takes. So this
plants tiny GGUFs -- a two-shard "real model", a step-a file (all F32) and a
step-b file (quantized, reordered) -- and requires:
- --derive to read the real shards into the expected map, and to refuse a
  class with two types;
- the rewrite to give, in step b's order and with the metadata unchanged, each
  tensor's real type from the step that has it: step b's bytes, step a's F32
  bytes where quantize changed an F32 tensor, BF16 bit-exact to ggml's
  round-to-nearest-even from step a's F32, the allowlisted F16; to split the
  fused ffn_gate_up_exps into gate (rows [0, n_ff) of each expert) and up (rows
  [n_ff, 2 n_ff)); to drop .scale, .input_scale and attn_{q,k,v}.bias; and to
  create the output with the umask's permissions;
- rc 1, no output, inputs untouched, each for its own reason: an output that is
  an input (same path or symlink), a step b not quantized from step a, an
  all-F32 step b, no indexer, no gate/up, and an extra tensor the loader may
  require;
- --verify to accept the result and refuse, each for its own reason, the step-b
  file, type flips (Q8_0 or F16 where the real model has F32, F32 experts, F32
  indexer, F32 token_embd, F16 outside the allowlist, and F16 in the allowlist
  where ne0 = 32 lets Q8_0 hold it), an extra tensor, the fused gate_up (in both
  layers, and in the second layer only), and a missing global tensor;
- a failure mid-write to leave an existing output and no temp file, and a
  wrongly shaped input to be refused before its data is read.

Standard library only: the script runs through its own /usr/bin/python3 shebang,
because the ctest interpreter on this host (miniconda) cannot load numpy.
Exits 77 (skip) when /usr/bin/python3 is missing or cannot import what the
script imports.
"""

import hashlib
import json
import os
import resource
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "sycl-qwen4exp-vehicle-rewrite.py"
PY     = "/usr/bin/python3"

F32, F16, Q8_0, BF16 = 0, 1, 8, 30
TYPE_NAME = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16"}
ALIGN = 32

N_EMBD, N_FF, N_EXP = 64, 2, 2
Q8_ROW = N_EMBD // 32 * 34      # two Q8_0 blocks per row, so a one-block row size is wrong

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"FAIL: {msg}")


def size(qtype, ne):
    n = 1
    for d in ne:
        n *= d
    return {F32: 4 * n, F16: 2 * n, BF16: 2 * n, Q8_0: 34 * n // 32}[qtype]


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors):
    """tensors: [(name, type, ne list, raw bytes, or None for a sparse zero hole)]"""
    kv = gguf_str("general.architecture") + struct.pack("<I", 8) + gguf_str("qwen4exp")
    # an empty string array, as the test-llama-archs fixture writes tokenizer.ggml.merges
    kv += gguf_str("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0)
    infos = b""
    offset = 0
    for name, qtype, ne, _ in tensors:
        n = size(qtype, ne)
        infos += gguf_str(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", qtype, offset)
        offset += n + (-n % ALIGN)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 2) + kv + infos)
        f.write(b"\0" * (-f.tell() % ALIGN))
        for _, qtype, ne, raw in tensors:
            n = size(qtype, ne)
            if raw is None:
                f.seek(n + (-n % ALIGN), os.SEEK_CUR)
            else:
                assert len(raw) == n, (qtype, ne, len(raw))
                f.write(raw + b"\0" * (-n % ALIGN))
        f.truncate()


def read_gguf(path):
    """([(name, type, ne, raw bytes)], metadata bytes) for a file written like write_gguf's"""
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
    kv = b[24:o]
    infos = []
    for _ in range(n_t):
        name, o = rstr(o)
        nd, = struct.unpack_from("<I", b, o)
        ne = list(struct.unpack_from(f"<{nd}Q", b, o + 4))
        qt, off = struct.unpack_from("<IQ", b, o + 4 + 8 * nd)
        o += 4 + 8 * nd + 12
        infos.append((name, qt, ne, off))
    base = o + (-o % ALIGN)
    return [(name, qt, ne, b[base + off:base + off + size(qt, ne)]) for name, qt, ne, off in infos], kv


def f32_bits(vals):
    return b"".join(struct.pack("<I", v) for v in vals)


def bf16_ref(u):
    # ggml_compute_fp32_to_bf16 (ggml-impl.h): quiet NaN, else round to nearest even
    if (u & 0x7fffffff) > 0x7f800000:
        return (u >> 16) | 64
    return (u + (0x7fff + ((u >> 16) & 1))) >> 16


def to_bf16(raw):
    return b"".join(struct.pack("<H", bf16_ref(u)) for u in struct.unpack(f"<{len(raw) // 4}I", raw))


def filler(tag, n):
    # distinct bytes per tensor and step, so a copy from the wrong source cannot match
    out = b""
    while len(out) < n:
        out += hashlib.sha256(f"{tag}/{len(out)}".encode()).digest()
    return out[:n]


# ties to even both ways, NaN, inf, -0, a subnormal, max finite
Q_BITS = [0x3f808000, 0x3f818000, 0x3f808001, 0x3f807fff, 0x7f800001, 0xff800001,
          0x7f800000, 0x80000000, 0x007fffff, 0x7f7fffff] + [0x3f800000 + 977 * i for i in range(22)]
K_BITS = [0x40490fdb + 131 * i for i in range(8)]

GATE_UP_NE = [N_EMBD, 2 * N_FF, N_EXP]
SPLIT_NE   = [N_EMBD, N_FF, N_EXP]

# the fixture's tensors: name, ne, step b's type, the real model's type, and
# where the rewrite must take it from: "b", "a" (step a's F32), "bf16" (from
# step a's F32), "split", or "drop"
SPEC = [
    ("token_embd.weight",             [N_EMBD, 4],   Q8_0, Q8_0, "b"),
    ("output_hc_up.weight",           [8, N_EMBD],   F16,  Q8_0, "b"),      # allowlisted
    ("blk.0.attn_qkv.weight",         [N_EMBD, 2],   Q8_0, Q8_0, "b"),
    ("blk.0.attn_qkv.scale",          [1],           F32,  None, "drop"),
    ("blk.0.attn_qkv.input_scale",    [1],           F32,  None, "drop"),
    ("blk.0.hc_attn_up.weight",       [8, N_EMBD],   F16,  Q8_0, "b"),      # allowlisted
    ("blk.0.ple_norm_key.weight",     [N_EMBD, 4],   Q8_0, F32,  "a"),
    ("blk.0.ple_conv1d.weight",       [4, N_EMBD],   F16,  F32,  "a"),
    ("blk.0.ffn_down_exps.weight",    SPLIT_NE,      Q8_0, Q8_0, "b"),
    ("blk.0.ffn_down_exps.scale",     [N_EXP],       F32,  None, "drop"),
    ("blk.0.ffn_gate_up_exps.weight", GATE_UP_NE,    Q8_0, None, "split"),
    ("blk.1.attn_q.weight",           [N_EMBD, 2],   Q8_0, Q8_0, "b"),
    ("blk.1.attn_q.bias",             [2],           F32,  None, "drop"),
    ("blk.1.indexer.q_proj.weight",   [8, 4],        F32,  BF16, "bf16"),
    ("blk.1.indexer.q_norm.weight",   [8],           F32,  F32,  "b"),
    ("blk.1.indexer.k_proj.weight",   [8, 1],        F32,  BF16, "bf16"),
    ("blk.1.hc_attn_up.weight",       [8, N_EMBD],   F16,  Q8_0, "b"),
    ("blk.1.ffn_down_exps.weight",    SPLIT_NE,      Q8_0, Q8_0, "b"),
    ("blk.1.ffn_gate_up_exps.weight", GATE_UP_NE,    Q8_0, None, "split"),
]
INDEXER = {"blk.1.indexer.q_proj.weight": Q_BITS, "blk.1.indexer.k_proj.weight": K_BITS}


def split_names(name):
    blk = name.split(".ffn_")[0]
    return [f"{blk}.ffn_gate_exps.weight", f"{blk}.ffn_up_exps.weight"]


def real_tensors(spec=SPEC):
    """the real model's (name, type, ne), fused gate_up replaced by gate and up"""
    out = []
    for name, ne, _, real, source in spec:
        if source == "split":
            out += [(n, Q8_0, SPLIT_NE) for n in split_names(name)]
        elif real is not None:
            # the fixture's HC low rank is 8, the real model's 320: Q8_0 needs ne0 % 32 == 0
            out.append((name, real, [32] + ne[1:] if real == Q8_0 and ne[0] % 32 else ne))
    return out


def step_a(spec=SPEC, holes=()):
    return [(name, F32, ne, None if name in holes else
             f32_bits(INDEXER[name]) if name in INDEXER else filler(f"a/{name}", size(F32, ne)))
            for name, ne, _, _, _ in spec]


def step_b(spec=SPEC, a=None, holes=()):
    # quantize copies F32 tensors unchanged and reorders the rest; reversed here
    a = {t[0]: t for t in (a or step_a(spec, holes))}
    return [(name, bt, ne, None if name in holes else a[name][3] if bt == F32 else filler(f"b/{name}", size(bt, ne)))
            for name, ne, bt, _, _ in reversed(spec)]


def run(*args, **kw):
    return subprocess.run([str(SCRIPT), *map(str, args)], capture_output=True, text=True, **kw)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def temps(d):
    return [p for p in os.listdir(d) if p.startswith(".vehicle-rewrite-")]


def skip_reason():
    # probe exactly what the script imports: gguf-py pulls in numpy and yaml
    try:
        r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0, 'gguf-py'); import gguf"],
                           cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        return f"{PY} does not exist"
    if r.returncode != 0:
        return f"{PY} cannot import gguf-py ({r.stderr.strip().splitlines()[-1:]})"
    return None


def main():
    reason = skip_reason()
    if reason:
        print(f"SKIP: {reason}, so the script under test cannot run")
        return 77

    with tempfile.TemporaryDirectory() as d:
        # --derive: two real shards, one layer kind each
        real = real_tensors()
        shards = [os.path.join(d, "real-1.gguf"), os.path.join(d, "real-2.gguf")]
        write_gguf(shards[0], [(n, t, ne, None) for n, t, ne in real if not n.startswith("blk.1.")])
        write_gguf(shards[1], [(n, t, ne, None) for n, t, ne in real if n.startswith("blk.1.")])
        tmap = os.path.join(d, "map.json")
        r = run("--derive", *shards)
        check(r.returncode == 0, f"--derive: rc {r.returncode} ({r.stderr.strip()[-300:]})")
        Path(tmap).write_text(r.stdout)
        layers = {}
        for n, _, _ in real:
            if n.startswith("blk."):
                layers.setdefault(n.split(".")[1], set()).add(n.split(".", 2)[2])
        want_map = {"source": ["real-1.gguf", "real-2.gguf"],
                    "types": {n.replace("blk.0.", "blk.#.").replace("blk.1.", "blk.#."): TYPE_NAME[t]
                              for n, t, _ in real},
                    "layers": sorted(sorted(s) for s in layers.values())}
        check(json.loads(r.stdout or "{}") == want_map, f"--derive: map {r.stdout[:300]}")

        # RED: a class with two types has no single real type, so --derive refuses it
        two = [os.path.join(d, "two-types-1.gguf"), os.path.join(d, "two-types-2.gguf")]
        write_gguf(two[0], [(n, t, ne, None) for n, t, ne in real if not n.startswith("blk.1.")])
        write_gguf(two[1], [(n, F16 if n == "blk.1.hc_attn_up.weight" else t, ne, None)
                            for n, t, ne in real if n.startswith("blk.1.")])
        r = run("--derive", *two)
        check(r.returncode != 0 and "blk.1.hc_attn_up.weight is F16" in r.stderr,
              f"--derive with two types in one class: rc {r.returncode} ({r.stderr.strip()[-200:]})")

        def rewrite(*args, **kw):
            return run("--map", tmap, *args, **kw)

        def verify(path):
            return run("--map", tmap, "--verify", path)

        # every case gets its own directory and freshly planted inputs, so one
        # defect cannot contaminate the next case's evidence
        def case(tag, a=None, b=None):
            cd = os.path.join(d, tag.replace(" ", "-"))
            os.mkdir(cd)
            pa, pb = os.path.join(cd, "a.gguf"), os.path.join(cd, "b.gguf")
            a = a if a is not None else step_a()
            write_gguf(pa, a)
            write_gguf(pb, b if b is not None else step_b(a=a))
            return pa, pb, (sha(pa), sha(pb)), os.path.join(cd, "out.gguf")

        # GREEN: the rewrite
        pa, pb, _, out = case("green")
        umask = os.umask(0)
        os.umask(umask)
        r = rewrite(pa, pb, out)
        check(r.returncode == 0, f"rewrite: rc {r.returncode} ({r.stderr.strip()[-300:]})")
        if not os.path.exists(out):
            print(f"{len(failures)} failure(s), and no output to check further")
            return 1
        (ta, _), (tb, kv_b) = read_gguf(pa), read_gguf(pb)
        got, kv_out = read_gguf(out)
        a_by = {t[0]: t for t in ta}
        spec_by = {s[0]: s for s in SPEC}
        want = []
        for name, bt, ne, braw in tb:                       # step b's order
            source = spec_by[name][4]
            if source == "b":
                want.append((name, bt, ne, braw))
            elif source == "a":
                want.append((name, F32, ne, a_by[name][3]))
            elif source == "bf16":
                want.append((name, BF16, ne, to_bf16(a_by[name][3])))
            elif source == "split":
                for i, n in enumerate(split_names(name)):
                    # per expert e, gate is rows [0, n_ff) and up rows [n_ff, 2 n_ff) of its 2 n_ff rows
                    rows = [braw[(e * 2 * N_FF + i * N_FF) * Q8_ROW:][:N_FF * Q8_ROW] for e in range(N_EXP)]
                    want.append((n, Q8_0, SPLIT_NE, b"".join(rows)))
        check([t[0] for t in got] == [t[0] for t in want],
              f"rewrite: tensors {[t[0] for t in got]}, expected {[t[0] for t in want]}")
        got_by = {t[0]: t for t in got}
        for name, qt, ne, raw in want:
            g = got_by.get(name, (None, None, None, None))
            check(g[1:3] == (qt, ne), f"{name}: type {g[1]} ne {g[2]}, expected {qt} {ne}")
            check(g[3] == raw, f"{name}: bytes are not {spec_by.get(name, (0, 0, 0, 0, 'split'))[4]}'s")
        check(kv_out == kv_b, "rewrite: metadata changed")
        mode = os.stat(out).st_mode & 0o777
        check(mode == 0o666 & ~umask, f"rewrite: mode {oct(mode)}, expected {oct(0o666 & ~umask)}")
        check(verify(out).returncode == 0, "--verify refused the rewrite's output")
        check(not temps(os.path.dirname(out)), "rewrite: temp file left behind")
        real_shape = got

        def refused(tag, r, hashes, pa, pb, dst, why):
            check(r.returncode == 1, f"{tag}: expected rc 1, got {r.returncode} ({r.stderr.strip()[-200:]})")
            check((sha(pa), sha(pb)) == hashes, f"{tag}: an input was modified")
            check(not os.path.lexists(dst) or os.path.islink(dst), f"{tag}: output written")
            check(not temps(os.path.dirname(pa)), f"{tag}: temp file left behind")
            check(why in r.stderr, f"{tag}: refused for another reason ({r.stderr.strip()[-300:]})")

        # RED: the output path is an input
        for tag, which in (("dst is step a", 0), ("dst is step b", 1)):
            pa, pb, h, _ = case(tag)
            dst = (pa, pb)[which]
            r = rewrite(pa, pb, dst)
            check(r.returncode == 1 and "refusing to overwrite" in r.stderr, f"{tag}: rc {r.returncode}")
            check((sha(pa), sha(pb)) == h, f"{tag}: an input was modified")
        pa, pb, h, _ = case("dst symlinks to step b")
        link = os.path.join(os.path.dirname(pa), "link.gguf")
        os.symlink(pb, link)
        refused("dst symlinks to step b", rewrite(pa, pb, link), h, pa, pb, link, "refusing to overwrite")

        # RED: inputs that cannot give the real model's tensor set, each refused
        # before any output exists and for its own reason
        def without(*names):
            return [s for s in SPEC if s[0] not in names]
        all_f32 = [(n, ne, F32, rt, src) for n, ne, _, rt, src in SPEC]
        extra = SPEC + [("blk.0.ssm_mystery.weight", [N_EMBD, 2], F32, None, "drop")]
        for tag, a, b, why in (
                ("b not quantized from a", step_a(), step_b()[1:], "b is not quantized from a"),
                ("all-F32 step b", step_a(), step_b(all_f32),
                 "blk.0.attn_qkv.weight: the real model's is Q8_0, step b has F32 and step a F32"),
                ("no indexer", step_a(without(*INDEXER, "blk.1.indexer.q_norm.weight")),
                 step_b(without(*INDEXER, "blk.1.indexer.q_norm.weight")), "blk.1 is no real layer"),
                ("no gate or up", step_a(without("blk.1.ffn_gate_up_exps.weight")),
                 step_b(without("blk.1.ffn_gate_up_exps.weight")), "blk.1 is no real layer"),
                ("undroppable extra", step_a(extra), step_b(extra),
                 "blk.0.ssm_mystery.weight is not in the real model, and the loader may require it")):
            pa, pb, h, o = case(tag, a, b)
            refused(tag, rewrite(pa, pb, o), h, pa, pb, o, why)

        # RED: --verify on files that differ from the real model, each for its own reason
        def vehicle(tag, tensors):
            p = os.path.join(d, tag.replace(" ", "-") + ".gguf")
            write_gguf(p, [(n, t, ne, None) for n, t, ne, *_ in tensors])
            return p

        def retype(name, qtype):
            return [(n, qtype if n == name else t, ne) for n, t, ne, _ in real_shape]

        def fused(*blks):
            split = {n for b in blks for n in split_names(f"{b}.ffn_gate_up_exps.weight")}
            return ([(n, t, ne) for n, t, ne, _ in real_shape if n not in split]
                    + [(f"{b}.ffn_gate_up_exps.weight", Q8_0, GATE_UP_NE) for b in blks])

        p = vehicle("verify real shape", real_shape)
        r = verify(p)
        check(r.returncode == 0, f"verify real shape: rc {r.returncode} ({r.stderr.strip()[-300:]})")
        for tag, tensors, why in (
                ("verify step-b file", tb, "blk.0.attn_qkv.scale is not in the real model"),
                ("verify Q8_0 ple_norm", retype("blk.0.ple_norm_key.weight", Q8_0),
                 "blk.0.ple_norm_key.weight is Q8_0, the real model's is F32"),
                ("verify F16 ple_conv1d", retype("blk.0.ple_conv1d.weight", F16),
                 "blk.0.ple_conv1d.weight is F16, the real model's is F32"),
                ("verify F32 experts", retype("blk.1.ffn_down_exps.weight", F32),
                 "blk.1.ffn_down_exps.weight is F32, the real model's is Q8_0"),
                ("verify F32 indexer", retype("blk.1.indexer.k_proj.weight", F32),
                 "blk.1.indexer.k_proj.weight is F32, the real model's is BF16"),
                ("verify F16 outside the allowlist", retype("blk.0.attn_qkv.weight", F16),
                 "blk.0.attn_qkv.weight is F16, the real model's is Q8_0"),
                ("verify F16 where Q8_0 fits", [(n, t, [32] + ne[1:] if n == "blk.0.hc_attn_up.weight" else ne)
                                                 for n, t, ne, _ in real_shape],
                 "blk.0.hc_attn_up.weight is F16, the real model's is Q8_0"),
                ("verify F32 token_embd", retype("token_embd.weight", F32),
                 "token_embd.weight is F32, the real model's is Q8_0"),
                ("verify extra scale", [t[:3] for t in real_shape] + [("blk.0.ffn_down_exps.scale", F32, [N_EXP])],
                 "blk.0.ffn_down_exps.scale is not in the real model"),
                ("verify fused gate_up", fused("blk.0", "blk.1"), "blk.0.ffn_gate_up_exps.weight is not in the real"),
                ("verify fused second layer", fused("blk.1"), "blk.1.ffn_gate_up_exps.weight is not in the real"),
                ("verify fused second layer shape", fused("blk.1"), "blk.1 is no real layer"),
                ("verify no token_embd", [t[:3] for t in real_shape if t[0] != "token_embd.weight"],
                 "global tensors: lacks ['token_embd.weight']")):
            r = verify(vehicle(tag, tensors))
            check(r.returncode == 1 and why in r.stderr, f"{tag}: rc {r.returncode} ({r.stderr.strip()[-300:]})")

        # a failure mid-write (a file-size limit makes write() fail with EFBIG
        # partway through, as ENOSPC would) keeps an existing output, no temp file
        pa, pb, _, keep = case("mid-write")
        Path(keep).write_bytes(b"previous output")

        def small_files():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (256, 256))

        check(os.path.getsize(out) > 256, "mid-write: the output is too small to hit the limit")
        r = rewrite(pa, pb, keep, preexec_fn=small_files, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        check(r.returncode != 0 and "File too large" in r.stderr, f"mid-write: write did not fail ({r.stderr[-200:]})")
        check(Path(keep).read_bytes() == b"previous output", "mid-write: existing output replaced")
        check(not temps(os.path.dirname(keep)), "mid-write: temp file left behind")

        # RED: a wrong input is refused before any tensor data is read. A
        # 256 MiB sparse tensor ahead of an undroppable extra costs nothing
        # unless the script copies it, so the child's peak RSS shows when it refused.
        big = 256 << 20
        big_ne = [N_EMBD, big // Q8_ROW]
        spec = [("token_embd.weight", big_ne, Q8_0, Q8_0, "b")] + extra[1:]
        a = step_a(spec, holes={"token_embd.weight"})
        pa, pb, _, o = case("early refusal", a, step_b(spec, a, holes={"token_embd.weight"}))
        code = ("import resource, subprocess, sys\n"
                "r = subprocess.run(sys.argv[1:], capture_output=True)\n"
                "print(r.returncode, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)\n")
        r = subprocess.run([sys.executable, "-c", code, str(SCRIPT), "--map", tmap, pa, pb, o],
                           capture_output=True, text=True)
        rc, rss_kib = map(int, r.stdout.split())
        check(rc == 1, f"early refusal: expected rc 1, got {rc}")
        check(rss_kib < (big >> 10) // 2, f"early refusal: peak RSS {rss_kib} KiB, so tensor data was read first")

        # the committed map is well formed: derived from the six shards, and a
        # type for every tensor its layers name
        committed = json.loads((ROOT / "scripts" / "sycl-qwen4exp-real-tensors.json").read_text())
        check(len(committed["source"]) == 6, f"committed map: sources {committed['source']}")
        check(all({f"blk.#.{s}" for s in layer} <= set(committed["types"]) for layer in committed["layers"]),
              "committed map: a layer names a tensor without a type")

    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("PASS: qwen4exp vehicle rewrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
