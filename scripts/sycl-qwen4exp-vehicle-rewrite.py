#!/usr/bin/python3
"""Rewrite the qwen4exp census vehicle into the real model's tensor set.

The census reads the vehicle's sched dump, which prints no tensor types, so the
vehicle has to carry the real Qwen3.8-Flash-Next's tensors and types or its rows
score the wrong path as agreeing (docs/backend/sycl-qwen4exp-op-census.md).
What the real model has is not written down here by hand: --derive reads it from
the real shards' headers into scripts/sycl-qwen4exp-real-tensors.json (a type
per name class, the per-layer tensor sets, the global tensors), and the rewrite
and --verify both diff against that file.

Step a (test-llama-archs -o) and step b (llama-quantize Q8_0) differ from the
real model in four ways, and the rewrite fixes each:

- types quantize cannot produce: the indexer projections are BF16 in the real
  model, and quantize never converts them (src/llama-quant.cpp:327-329), so they
  are converted from step a's F32. Where step b quantized what the real model
  keeps F32 (ple_norm_*) or made F16 (ple_conv1d), step a's F32 tensor is
  copied unchanged;
- the fused blk.*.ffn_gate_up_exps.weight: the real model has separate
  ffn_gate_exps and ffn_up_exps, and the two reach different MUL_MAT_ID paths.
  Split along ne1 in the order build_moe_ffn views the fused result
  (src/llama-graph.cpp:2195-2198): rows [0, n_ff) are gate, rows [n_ff, 2 n_ff)
  are up, per expert. Rows are whole Q8_0 blocks. qwen4exp loads the separate
  form whenever the fused tensor is absent (src/llama-model.cpp:4262-4266);
- tensors the real model lacks: every .scale and .input_scale, which the loader
  creates TENSOR_NOT_REQUIRED in its generic pass (src/llama-model.cpp:
  2366-2499), and attn_{q,k,v}.bias, TENSOR_NOT_REQUIRED in create_tensor_qkv
  (src/llama-model.cpp:4302-4304). They are dropped; any other tensor the real
  model lacks is refused;
- what cannot be fixed is listed in ALLOWED with its reason.

Everything else, and the metadata, is copied byte for byte from step b, the
metadata as raw bytes: the fixture carries empty arrays (tokenizer.ggml.merges,
classifier.output_labels), which GGUFWriter refuses and GGUFReader reports
without their element type.

Host-only: it reads and writes files and touches no device. The shebang names
/usr/bin/python3 on purpose: `python3` on the fork's host is miniconda, whose
numpy cannot load (libmkl_intel_lp64.so.2 missing).

    scripts/sycl-qwen4exp-vehicle-rewrite.py <step-a.gguf> <step-b.gguf> <out.gguf>
    scripts/sycl-qwen4exp-vehicle-rewrite.py --verify <file.gguf>
    scripts/sycl-qwen4exp-vehicle-rewrite.py --derive <real shard.gguf>... > <map.json>

--map <map.json> replaces the committed map. The rewrite plans every output
tensor from the headers alone and diffs the plan against the map before it reads
any tensor data; --verify runs the same diff on a finished file and exits 0 only
when it is clean. sycl-qwen4exp-census-run.sh runs --verify, so a vehicle that
differs from the real model cannot score its rows as agreeing.
"""

import argparse
import collections
import json
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402

MAP = Path(__file__).resolve().parent / "sycl-qwen4exp-real-tensors.json"

LAYER     = re.compile(r"^blk\.(\d+)\.(.+)$")
GATE_UP   = re.compile(r"^(blk\.\d+)\.ffn_gate_up_exps\.weight$")
DROPPABLE = re.compile(r"\.(scale|input_scale)$|^blk\.\d+\.attn_[qkv]\.bias$")

BF16 = gguf.GGMLQuantizationType.BF16
F32  = gguf.GGMLQuantizationType.F32

# name class -> (the vehicle's type, why it cannot be the real model's)
ALLOWED = {
    c: ("F16", "ne0 is the HC low rank, 320 in the real model and 8 in the fixture; "
               "Q8_0 needs ne0 % 32 == 0, so quantize falls back to F16")
    for c in ("blk.#.hc_attn_up.weight", "blk.#.hc_ffn_up.weight", "output_hc_up.weight")
}


def name_class(name: str) -> str:
    m = LAYER.match(name)
    return f"blk.#.{m.group(2)}" if m else name


def pad(n: int, align: int) -> int:
    return (align - n % align) % align


def derive(shards: list[str]) -> dict:
    types = {}
    layers = collections.defaultdict(set)
    for path in shards:
        for t in gguf.GGUFReader(path).tensors:
            c = name_class(t.name)
            if types.setdefault(c, t.tensor_type.name) != t.tensor_type.name:
                raise SystemExit(f"{t.name} is {t.tensor_type.name}, but {c} is {types[c]} elsewhere")
            m = LAYER.match(t.name)
            if m:
                layers[int(m.group(1))].add(m.group(2))
    return {"source": sorted(os.path.basename(p) for p in shards),
            "types": dict(sorted(types.items())),
            "layers": sorted(sorted(s) for s in {frozenset(s) for s in layers.values()})}


def diff(tensors: list[tuple[str, str]], tmap: dict) -> list[str]:
    """what separates [(name, type name)] from the real model's tensor set"""
    problems = []
    types = tmap["types"]
    by_layer = collections.defaultdict(set)
    found_globals = set()
    for name, qtype in tensors:
        c = name_class(name)
        if c not in types:
            problems.append(f"{name} is not in the real model")
            continue
        if qtype != types[c] and qtype != ALLOWED.get(c, (None,))[0]:
            problems.append(f"{name} is {qtype}, the real model's is {types[c]}")
        m = LAYER.match(name)
        if m:
            by_layer[int(m.group(1))].add(m.group(2))
        else:
            found_globals.add(name)
    real_layers = [set(s) for s in tmap["layers"]]
    for il, s in sorted(by_layer.items()):
        if s not in real_layers:
            near = min(real_layers, key=lambda r: len(r ^ s))
            problems.append(f"blk.{il} is no real layer: next to the nearest it lacks {sorted(near - s)} "
                            f"and has {sorted(s - near)}")
    real_globals = {c for c in types if not c.startswith("blk.#.")}
    if found_globals != real_globals:
        problems.append(f"global tensors: lacks {sorted(real_globals - found_globals)}, "
                        f"has {sorted(found_globals - real_globals)}")
    return problems


def refuse(path: str, problems: list[str]) -> int:
    print(f"{path}: refusing: " + "; ".join(problems), file=sys.stderr)
    return 1


def verify(path: str, tmap: dict) -> int:
    problems = diff([(t.name, t.tensor_type.name) for t in gguf.GGUFReader(path).tensors], tmap)
    return refuse(path, problems) if problems else 0


def pick(ta, tb, types: list[str]):
    """(what, the tensor to read) for the first of types that step b, step a, or
    a BF16 conversion of step a's F32 provides; None if none does"""
    for want in types:
        if tb.tensor_type.name == want:
            return "b", tb
        if ta.tensor_type.name == want:
            return "a", ta
        if want == BF16.name and ta.tensor_type == F32:
            return "bf16", ta
    return None


def plan(ra: gguf.GGUFReader, rb: gguf.GGUFReader, tmap: dict):
    """([(name, type, ne, n_bytes, produce)], [dropped names], [problems]) from
    the headers alone; produce() returns the bytes and is only called while writing"""
    # quantize reorders the tensors, so compare the sets; the output keeps step b's order
    def shapes(r):
        return sorted((t.name, [int(d) for d in t.shape]) for t in r.tensors)
    if shapes(ra) != shapes(rb):
        return [], [], ["step a and step b have different tensor names or shapes, so b is not quantized from a"]
    ta_by_name = {t.name: t for t in ra.tensors}
    out, dropped, problems = [], [], []
    for tb in rb.tensors:
        ta = ta_by_name[tb.name]
        ne = [int(d) for d in tb.shape]
        c = name_class(tb.name)
        m = GATE_UP.match(tb.name)
        if m:
            parts = [f"{m.group(1)}.ffn_gate_exps.weight", f"{m.group(1)}.ffn_up_exps.weight"]
            want = [tmap["types"].get(name_class(p)) for p in parts]
            src = pick(ta, tb, want[:1]) if want[0] == want[1] else None
            if src is None or src[0] == "bf16":
                problems.append(f"{tb.name}: neither step has it as {want}")
                continue
            t = src[1]
            n_embd, n_ff2, n_exp = ne
            half = t.n_bytes // (n_ff2 * n_exp) * (n_ff2 // 2)

            # the data is expert-major: expert e's rows start at e * 2 * half
            def produce(t=t, first=True, half=half, n_exp=n_exp):
                b = t.data.tobytes()
                starts = (e * 2 * half + (0 if first else half) for e in range(n_exp))
                return b"".join(b[s:s + half] for s in starts)
            for i, p in enumerate(parts):
                out.append((p, t.tensor_type, [n_embd, n_ff2 // 2, n_exp], half * n_exp,
                            lambda p=produce, first=(i == 0): p(first=first)))
            continue
        if c not in tmap["types"]:
            if DROPPABLE.search(tb.name):
                dropped.append(tb.name)
            else:
                problems.append(f"{tb.name} is not in the real model, and the loader may require it")
            continue
        src = pick(ta, tb, [tmap["types"][c]] + ([ALLOWED[c][0]] if c in ALLOWED else []))
        if src is None:
            problems.append(f"{tb.name}: the real model's is {tmap['types'][c]}, step b has "
                            f"{tb.tensor_type.name} and step a {ta.tensor_type.name}")
            continue
        what, t = src
        if what == "bf16":
            out.append((t.name, BF16, ne, t.n_elements * 2,
                        lambda t=t: gguf.quants.quantize(t.data, BF16).tobytes()))
        else:
            out.append((t.name, t.tensor_type, ne, t.n_bytes, lambda t=t: t.data.tobytes()))
    problems += diff([(name, qtype.name) for name, qtype, _, _, _ in out], tmap)
    return out, dropped, problems


def rewrite(step_a: str, step_b: str, dst: str, tmap: dict) -> int:
    # a same-file rewrite would destroy an input on any failure mid-write
    for src in (step_a, step_b):
        if os.path.realpath(src) == os.path.realpath(dst) or (
                os.path.exists(dst) and os.path.samefile(src, dst)):
            print(f"{dst} is the input {src}; refusing to overwrite it", file=sys.stderr)
            return 1

    ra, rb = gguf.GGUFReader(step_a), gguf.GGUFReader(step_b)
    if ra.endianess != gguf.GGUFEndian.LITTLE or rb.endianess != gguf.GGUFEndian.LITTLE:
        print("only little-endian files are handled", file=sys.stderr)
        return 1
    tensors, dropped, problems = plan(ra, rb, tmap)
    if problems:
        return refuse(step_b, problems)
    align = rb.alignment

    kv = [f for f in rb.fields.values() if not f.name.startswith("GGUF.")]
    kv_bytes = b"".join(p.tobytes() for f in kv for p in f.parts)

    # tensor infos first, from types and shapes alone; the data is streamed below
    infos = bytearray()
    offset = 0
    for name, qtype, ne, n_bytes, _ in tensors:
        name_b = name.encode("utf-8")
        infos += struct.pack("<Q", len(name_b)) + name_b
        infos += struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", int(qtype), offset)
        offset += n_bytes + pad(n_bytes, align)

    version = int(rb.fields["GGUF.version"].parts[-1][0])
    head = struct.pack("<IIQQ", gguf.GGUF_MAGIC, version, len(tensors), len(kv))

    # write beside dst and rename, so dst is either the old file or a complete new one
    fd, tmp = tempfile.mkstemp(prefix=".vehicle-rewrite-", dir=os.path.dirname(os.path.abspath(dst)))
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.fchmod(fd, 0o666 & ~umask)  # mkstemp's 0600 would otherwise survive the rename
        with os.fdopen(fd, "wb") as out:
            out.write(head + kv_bytes + infos)
            out.write(b"\0" * pad(out.tell(), align))
            for name, _, _, n_bytes, produce in tensors:
                blob = produce()
                assert len(blob) == n_bytes, name
                out.write(blob)
                out.write(b"\0" * pad(len(blob), align))
        os.replace(tmp, dst)
    except BaseException:
        os.unlink(tmp)
        raise

    b_types = {t.name: t.tensor_type for t in rb.tensors}
    for name, qtype, _, _, _ in tensors:
        if b_types.get(name, qtype) != qtype:
            print(f"{b_types[name].name} -> {qtype.name}: {name}")
    for t in rb.tensors:
        if GATE_UP.match(t.name):
            print(f"split: {t.name} -> ffn_gate_exps + ffn_up_exps")
    for name in dropped:
        print(f"dropped: {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default=str(MAP), help="the real model's tensor map (default: %(default)s)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", metavar="FILE", help="exit 0 only if FILE has the real model's tensor set")
    mode.add_argument("--derive", nargs="+", metavar="SHARD", help="print the tensor map of the real model's shards")
    ap.add_argument("files", nargs="*", metavar="step-a step-b out", help="rewrite step b into out")
    args = ap.parse_args()
    if args.derive:
        json.dump(derive(args.derive), sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    tmap = json.loads(Path(args.map).read_text())
    if args.verify:
        return verify(args.verify, tmap)
    if len(args.files) != 3:
        ap.error("the rewrite takes <step-a.gguf> <step-b.gguf> <out.gguf>")
    return rewrite(*args.files, tmap)


if __name__ == "__main__":
    sys.exit(main())
