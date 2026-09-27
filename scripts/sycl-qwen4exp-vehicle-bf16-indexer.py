#!/usr/bin/python3
"""Rewrite the qwen4exp census vehicle's indexer projections to BF16.

The real Qwen3.8-Flash-Next ships blk.*.indexer.{q,k}_proj.weight as BF16, but
llama-quantize can never produce that: src/llama-quant.cpp:327-329 exempts both
names from quantization before any type is chosen, so a --tensor-type override
does not reach them and they stay F32. This copies a GGUF unchanged except for
those tensors, which are converted F32 -> BF16 (docs/backend/sycl-qwen4exp-op-census.md).

The metadata is copied as raw bytes, not through GGUFWriter: the test-llama-archs
fixture carries empty arrays (tokenizer.ggml.merges, classifier.output_labels),
which GGUFWriter refuses and GGUFReader reports without their element type.

Host-only: it reads and writes files and touches no device. The shebang names
/usr/bin/python3 on purpose: `python3` on the fork's host is miniconda, whose
numpy cannot load (libmkl_intel_lp64.so.2 missing).

    scripts/sycl-qwen4exp-vehicle-bf16-indexer.py <in.gguf> <out.gguf>
    scripts/sycl-qwen4exp-vehicle-bf16-indexer.py --verify <file.gguf>

--verify exits 0 only if the file has indexer projections and all of them are
BF16; sycl-qwen4exp-census-run.sh runs it so an F32 indexer cannot pass as BF16.
"""

import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402

INDEXER_PROJ = re.compile(r"^blk\.\d+\.indexer\.[qk]_proj\.weight$")


def pad(n: int, align: int) -> int:
    return (align - n % align) % align


def verify(path: str) -> int:
    reader = gguf.GGUFReader(path)
    found = [(t.name, t.tensor_type) for t in reader.tensors if INDEXER_PROJ.match(t.name)]
    bad = [f"{n} is {q.name}" for n, q in found if q != gguf.GGMLQuantizationType.BF16]
    if not found or bad:
        print(f"{path}: indexer projections are not all BF16 ({'; '.join(bad) or 'none found'}); "
              "run this script's rewrite on it first", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--verify":
        return verify(sys.argv[2])
    if len(sys.argv) != 3:
        print(__doc__.split("\n\n")[-2].strip(), file=sys.stderr)
        return 1
    src, dst = sys.argv[1], sys.argv[2]

    reader = gguf.GGUFReader(src)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        print("only little-endian files are handled", file=sys.stderr)
        return 1
    align = reader.alignment

    kv = [f for f in reader.fields.values() if not f.name.startswith("GGUF.")]
    kv_bytes = b"".join(p.tobytes() for f in kv for p in f.parts)

    converted = []
    infos = bytearray()
    blobs = []
    offset = 0
    for t in reader.tensors:
        data, qtype = t.data, t.tensor_type
        if INDEXER_PROJ.match(t.name):
            if qtype != gguf.GGMLQuantizationType.F32:
                print(f"{t.name} is {qtype.name}, expected F32; refusing", file=sys.stderr)
                return 1
            data = gguf.quants.quantize(data, gguf.GGMLQuantizationType.BF16)
            qtype = gguf.GGMLQuantizationType.BF16
            converted.append(t.name)
        blob = data.tobytes()
        name = t.name.encode("utf-8")
        ne = [int(d) for d in t.shape]
        infos += struct.pack("<Q", len(name)) + name
        infos += struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", int(qtype), offset)
        blobs.append(blob)
        offset += len(blob) + pad(len(blob), align)

    if not converted:
        # a vehicle without an indexer would silently lose IDX-PROJ-BF16 coverage
        print("no indexer.{q,k}_proj tensors found; refusing", file=sys.stderr)
        return 1

    version = int(reader.fields["GGUF.version"].parts[-1][0])
    head = struct.pack("<IIQQ", gguf.GGUF_MAGIC, version, len(reader.tensors), len(kv))
    with open(dst, "wb") as out:
        out.write(head + kv_bytes + infos)
        out.write(b"\0" * pad(out.tell(), align))
        for blob in blobs:
            out.write(blob)
            out.write(b"\0" * pad(len(blob), align))

    for name in converted:
        print(f"BF16: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
