#!/usr/bin/python3
"""Rewrite the qwen4exp census vehicle into the real model's tensor shapes.

The census reads the vehicle's sched dump, which prints no tensor types, so the
vehicle has to carry the real Qwen3.8-Flash-Next's types and layout or its rows
score the wrong path as agreeing (docs/backend/sycl-qwen4exp-op-census.md).
Two things differ after test-llama-archs + llama-quantize, and this fixes both:

- blk.*.indexer.{q,k}_proj.weight: the real model ships BF16, but
  llama-quantize can never produce that (src/llama-quant.cpp:327-329 exempts
  both names before any type is chosen). Converted F32 -> BF16.
- blk.*.ffn_gate_up_exps.weight: the fixture writes the fused tensor, the real
  model separate ffn_gate_exps and ffn_up_exps, and the two reach different
  MUL_MAT_ID paths. Split along ne1 in the order build_moe_ffn views the fused
  result (src/llama-graph.cpp:2195-2198): rows [0, n_ff) are gate, rows
  [n_ff, 2 n_ff) are up, per expert. Q8_0 blocks run along ne0, so each half
  is whole rows. qwen4exp loads the separate form whenever the fused tensor is
  absent (create_tensor_gate_up_exps, src/llama-model.cpp:4262-4266).

Everything else is copied byte for byte, the metadata as raw bytes: the
fixture carries empty arrays (tokenizer.ggml.merges, classifier.output_labels),
which GGUFWriter refuses and GGUFReader reports without their element type.

Host-only: it reads and writes files and touches no device. The shebang names
/usr/bin/python3 on purpose: `python3` on the fork's host is miniconda, whose
numpy cannot load (libmkl_intel_lp64.so.2 missing).

    scripts/sycl-qwen4exp-vehicle-rewrite.py <in.gguf> <out.gguf>
    scripts/sycl-qwen4exp-vehicle-rewrite.py --verify <file.gguf>

Both check every tensor type before any data is read: the input must be the
step-b Q8_0 vehicle (Q8_0 experts, fused gate_up, F32 indexer projections), and
--verify's file the real model's shape (Q8_0 experts, separate gate and up, no
fused tensor, BF16 indexer projections). --verify exits 0 only then;
sycl-qwen4exp-census-run.sh runs it so a vehicle in the wrong shape cannot
score its rows as agreeing.
"""

import os
import re
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402

INDEXER_PROJ = re.compile(r"^blk\.\d+\.indexer\.[qk]_proj\.weight$")
EXPERTS      = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.weight$")
GATE_UP      = re.compile(r"^(blk\.\d+)\.ffn_gate_up_exps\.weight$")
DOWN         = re.compile(r"^(blk\.\d+)\.ffn_down_exps\.weight$")

Q8_0 = gguf.GGMLQuantizationType.Q8_0
BF16 = gguf.GGMLQuantizationType.BF16
F32  = gguf.GGMLQuantizationType.F32


def pad(n: int, align: int) -> int:
    return (align - n % align) % align


def type_problems(reader: gguf.GGUFReader, indexer_type: gguf.GGMLQuantizationType, fused: bool) -> list[str]:
    # IDX-PROJ-BF16 and MOE-MMID score these types and this layout; the sched dump shows neither
    problems = []
    names = {t.name for t in reader.tensors}
    for pattern, want, what in ((INDEXER_PROJ, indexer_type, "indexer.{q,k}_proj"),
                                (EXPERTS,      Q8_0,         "ffn_*_exps")):
        found = [t for t in reader.tensors if pattern.match(t.name)]
        if not found:
            problems.append(f"no {what} tensors")
        problems += [f"{t.name} is {t.tensor_type.name}, expected {want.name}"
                     for t in found if t.tensor_type != want]
    for blk in (m.group(1) for m in map(DOWN.match, sorted(names)) if m):
        has_fused = f"{blk}.ffn_gate_up_exps.weight" in names
        has_split = {f"{blk}.ffn_gate_exps.weight", f"{blk}.ffn_up_exps.weight"} <= names
        if fused and not (has_fused and not has_split):
            problems.append(f"{blk}: expected the fused ffn_gate_up_exps the fixture writes")
        if not fused and not (has_split and not has_fused):
            problems.append(f"{blk}: expected separate ffn_gate_exps and ffn_up_exps and no fused "
                            "ffn_gate_up_exps, as the real model has")
    return problems


def refuse(path: str, problems: list[str]) -> int:
    print(f"{path}: refusing: " + "; ".join(problems), file=sys.stderr)
    return 1


def verify(path: str) -> int:
    problems = type_problems(gguf.GGUFReader(path), BF16, fused=False)
    return refuse(path, problems) if problems else 0


def out_tensors(reader: gguf.GGUFReader):
    """(name, type, ne, n_bytes, produce) per output tensor, in input order;
    produce() returns the bytes and is only called while writing"""
    for t in reader.tensors:
        ne = [int(d) for d in t.shape]
        m = GATE_UP.match(t.name)
        if m:
            n_embd, n_ff2, n_exp = ne
            assert n_ff2 % 2 == 0, t.name
            row = t.n_bytes // (n_ff2 * n_exp)
            half = n_ff2 // 2 * row

            # the data is expert-major: expert e's rows start at e * 2 * half
            def produce(t=t, first=True, half=half, n_exp=n_exp):
                b = t.data.tobytes()
                starts = (e * 2 * half + (0 if first else half) for e in range(n_exp))
                return b"".join(b[s:s + half] for s in starts)
            yield (f"{m.group(1)}.ffn_gate_exps.weight", Q8_0, [n_embd, n_ff2 // 2, n_exp],
                   half * n_exp, lambda p=produce: p(first=True))
            yield (f"{m.group(1)}.ffn_up_exps.weight", Q8_0, [n_embd, n_ff2 // 2, n_exp],
                   half * n_exp, lambda p=produce: p(first=False))
        elif INDEXER_PROJ.match(t.name):
            yield (t.name, BF16, ne, t.n_elements * 2,
                   lambda t=t: gguf.quants.quantize(t.data, BF16).tobytes())
        else:
            yield (t.name, t.tensor_type, ne, t.n_bytes, lambda t=t: t.data.tobytes())


def rewrite(src: str, dst: str) -> int:
    # a same-file rewrite would destroy the step-b input on any failure mid-write
    if os.path.realpath(src) == os.path.realpath(dst) or (
            os.path.exists(dst) and os.path.samefile(src, dst)):
        print(f"{dst} is the input {src}; refusing to overwrite it", file=sys.stderr)
        return 1

    reader = gguf.GGUFReader(src)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        print("only little-endian files are handled", file=sys.stderr)
        return 1
    problems = type_problems(reader, F32, fused=True)
    if problems:
        return refuse(src, problems)
    align = reader.alignment

    kv = [f for f in reader.fields.values() if not f.name.startswith("GGUF.")]
    kv_bytes = b"".join(p.tobytes() for f in kv for p in f.parts)

    # tensor infos first, from types and shapes alone; the data is streamed below
    tensors = list(out_tensors(reader))
    infos = bytearray()
    offset = 0
    for name, qtype, ne, n_bytes, _ in tensors:
        name_b = name.encode("utf-8")
        infos += struct.pack("<Q", len(name_b)) + name_b
        infos += struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", int(qtype), offset)
        offset += n_bytes + pad(n_bytes, align)

    version = int(reader.fields["GGUF.version"].parts[-1][0])
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

    for t in reader.tensors:
        if INDEXER_PROJ.match(t.name):
            print(f"BF16: {t.name}")
        elif GATE_UP.match(t.name):
            print(f"split: {t.name} -> ffn_gate_exps + ffn_up_exps")
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--verify":
        return verify(sys.argv[2])
    if len(sys.argv) != 3:
        print(__doc__.split("\n\n")[-2].strip(), file=sys.stderr)
        return 1
    return rewrite(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
