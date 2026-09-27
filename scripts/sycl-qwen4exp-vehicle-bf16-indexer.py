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

Both check every tensor type before any data is read: the input must be the
step-b Q8_0 vehicle (Q8_0 experts, F32 indexer projections), and --verify's file
the step-b2 result (Q8_0 experts, BF16 indexer projections). --verify exits 0
only then; sycl-qwen4exp-census-run.sh runs it so a wrongly typed vehicle cannot
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

Q8_0 = gguf.GGMLQuantizationType.Q8_0
BF16 = gguf.GGMLQuantizationType.BF16
F32  = gguf.GGMLQuantizationType.F32


def pad(n: int, align: int) -> int:
    return (align - n % align) % align


def type_problems(reader: gguf.GGUFReader, indexer_type: gguf.GGMLQuantizationType) -> list[str]:
    # MOE-MMID and IDX-PROJ-BF16 score these types, and the sched dump prints none
    problems = []
    for pattern, want, what in ((INDEXER_PROJ, indexer_type, "indexer.{q,k}_proj"),
                                (EXPERTS,      Q8_0,         "ffn_*_exps")):
        found = [t for t in reader.tensors if pattern.match(t.name)]
        if not found:
            problems.append(f"no {what} tensors")
        problems += [f"{t.name} is {t.tensor_type.name}, expected {want.name}"
                     for t in found if t.tensor_type != want]
    return problems


def refuse(path: str, problems: list[str]) -> int:
    print(f"{path}: refusing: " + "; ".join(problems), file=sys.stderr)
    return 1


def verify(path: str) -> int:
    problems = type_problems(gguf.GGUFReader(path), BF16)
    return refuse(path, problems) if problems else 0


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
    problems = type_problems(reader, F32)
    if problems:
        return refuse(src, problems)
    align = reader.alignment

    kv = [f for f in reader.fields.values() if not f.name.startswith("GGUF.")]
    kv_bytes = b"".join(p.tobytes() for f in kv for p in f.parts)

    # tensor infos first, from types and shapes alone; the data is streamed below
    infos = bytearray()
    offset = 0
    for t in reader.tensors:
        convert = INDEXER_PROJ.match(t.name) is not None
        qtype   = BF16 if convert else t.tensor_type
        n_bytes = t.n_elements * 2 if convert else t.n_bytes
        name = t.name.encode("utf-8")
        ne = [int(d) for d in t.shape]
        infos += struct.pack("<Q", len(name)) + name
        infos += struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", int(qtype), offset)
        offset += n_bytes + pad(n_bytes, align)

    version = int(reader.fields["GGUF.version"].parts[-1][0])
    head = struct.pack("<IIQQ", gguf.GGUF_MAGIC, version, len(reader.tensors), len(kv))

    # write beside dst and rename, so dst is either the old file or a complete new one
    fd, tmp = tempfile.mkstemp(prefix=".bf16-indexer-", dir=os.path.dirname(os.path.abspath(dst)))
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(head + kv_bytes + infos)
            out.write(b"\0" * pad(out.tell(), align))
            for t in reader.tensors:
                data = t.data
                if INDEXER_PROJ.match(t.name):
                    data = gguf.quants.quantize(data, BF16)
                blob = data.tobytes()
                out.write(blob)
                out.write(b"\0" * pad(len(blob), align))
        os.replace(tmp, dst)
    except BaseException:
        os.unlink(tmp)
        raise

    for t in reader.tensors:
        if INDEXER_PROJ.match(t.name):
            print(f"BF16: {t.name}")
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
