#!/usr/bin/env python3
"""Deterministic output-equivalence oracle for SYCL dispatch/layout changes.

Why this exists
---------------
`llama-bench` measures tok/s only. A change to kernel dispatch, weight staging,
graph replay, or allocation routing can raise throughput by silently skipping or
mis-staging work and still emit garbage tokens (CLAUDE.md records a fake +19.6%
PP "win" that shipped past a throughput-only check). The existing GPT-OSS gate
only checks that a short count appears in the output, which cannot see a small
numerical drift. This script captures a deterministic generation and compares
two captures element by element, so a change is provably output-identical -- or
provably not.

Two capture modes
-----------------

**`--mode logits` (primary, the default).** `llama-perplexity --kl-divergence-base
FNAME` (alias `--save-all-logits`) records the log-probabilities of every
evaluated token to a binary file; the repo already uses this as a before/after
comparison workflow in `examples/model-conversion/scripts/utils/perplexity-gen.sh`
and `perplexity-run.sh`. `capture` drives that tool over a short deterministic
input, parses the binary directly, and reduces each row to its top-k
(token id, log-probability) pairs in the capture JSON. `compare` then diffs the
two captures in-process with the abs/rel tolerance -- no second model run, no
scraping of printed statistics.

The binary layout, read off the writer in `tools/perplexity/perplexity.cpp`
(`log_softmax` at :79, `process_logits` at :144, the header writes at :460 and
:521, the row count at :614)::

    "_logits_"                        8 bytes, ASCII magic
    n_ctx                             int32    (per-chunk context, not params.n_ctx)
    n_vocab                           int32
    n_chunk                           int32
    tokens[n_chunk * n_ctx]           int32 each, the evaluated token ids
    rows                              n_chunk * (n_ctx - 1 - n_ctx//2) of them

Each row is `nv = 2*((n_vocab + 1)//2) + 4` uint16 values. The first four
uint16 are two float32, `scale` and `min_log_prob`; the quantized values start
at uint16 index 4, one per vocabulary entry. A row decodes to a
softmax-normalized log-probability vector::

    log_prob[i] = min_log_prob + scale * q[i]

which is what this oracle compares -- shift-invariant, unlike raw logits. The
values are quantized: `scale = (max_logit - min_logit) / 65535` over a range
clamped to 16 nats, so one quantization step is at most ~2.4e-4. Two runs that
agree exactly produce byte-identical files and a divergence of exactly 0; the
default tolerance is therefore strict, and `--tol 3e-4` is the threshold that
tolerates a single-step wobble.

Written natively, so the file is little-endian on this host and the parser
assumes that.

The format carries an 8-byte magic and **no version field**, which is upstream's
design. If a future `perplexity.cpp` change altered the row layout without
touching the magic, a parser that trusted the magic alone would misparse rather
than fail -- and a silently wrong "no divergence" verdict is the worst thing
this tool could produce, since it would green-light exactly the class of broken
optimization the oracle exists to catch. The mitigation is `open_kld`, which
computes the exact file size the header implies and refuses anything else. That
converts most drift into a loud error, but it cannot catch a change that
preserves the total size (say, reordering fields within a row). If the layout
ever changes, this parser must be updated deliberately.

**`--mode tokens` (secondary fallback).** Records the greedy token id sequence
at `--temp 0 --seed 42`: `llama-completion` generates, `llama-tokenize --ids`
turns the text back into ids. `llama-completion` has no logit-dump flag of its
own, so this mode cannot see numerical drift that does not flip a greedy
argmax, and its ids come from re-tokenizing the output text rather than from the
sampler, which can shift `first_divergent_index` by a token. Use it when the
model or input is too large for a perplexity pass, when only the user-visible
output matters, or as a fast smoke check -- it needs no `-f` input file and
generates far less data.

Capture format
--------------
JSON, either a bare list of entries or an object with an ``entries`` key::

    {
      "mode": "logits",
      "meta": {"model": "...", "n_vocab": 32000, "top_k": 8, ...},
      "entries": [
        {"index": 0, "token": 28740, "target": 28705,
         "top_tokens": [28740, 28750], "logits": [-0.31, -2.44]}
      ]
    }

`token` is the row's argmax (the greedy prediction) in logit mode and the
generated token in token mode, so both modes agree on what `token` means.
`target` is the token the input actually continued with -- informational, and
not compared, since it is fixed by the input rather than by the backend. Entries
may carry a `logits` list, a matching `top_tokens` list, and a `piece` string.
Extra keys are ignored by the comparison.

Usage
-----
Capture a reference and a candidate, then compare them::

    ./scripts/sycl-logit-oracle.py capture \\
        --model /models/mistral-7b-v0.1.Q4_0.gguf \\
        --device level_zero:1 --out /tmp/before.json

    # ... rebuild with the change ...

    ./scripts/sycl-logit-oracle.py capture \\
        --model /models/mistral-7b-v0.1.Q4_0.gguf \\
        --device level_zero:1 --out /tmp/after.json

    ./scripts/sycl-logit-oracle.py compare /tmp/before.json /tmp/after.json

`compare` exits non-zero when the divergence exceeds the tolerance, so it can
gate a commit directly. Both captures must be produced in the same mode with the
same model, input, seed and geometry; `compare` fails outright on a mode
mismatch and warns when other recorded metadata differs.

Pure stdlib on purpose: run it with `/usr/bin/python3` (the conda python on this
host has a broken numpy), and do not add third-party imports.
"""

from __future__ import annotations

import argparse
import array
import dataclasses
import heapq
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
from typing import Any, Iterator

DEFAULT_PROMPT = "1, 2, 3, 4, 5,"
DEFAULT_N_PREDICT = 15
DEFAULT_SEED = 42
DEFAULT_TOL = 1e-6
DEFAULT_TOP_K = 8
DEFAULT_CTX = 64
DEFAULT_CHUNKS = 2

# Magic at the head of a llama-perplexity --kl-divergence-base file.
KLD_MAGIC = b"_logits_"

# Sanity ceilings on the .kld header. These are far above any real model, and
# exist only so a corrupted-but-plausible header cannot drive an enormous read.
MAX_KLD_N_CTX = 1 << 22
MAX_KLD_N_VOCAB = 1 << 24
MAX_KLD_N_CHUNK = 1 << 20

# Metadata keys that must agree for a comparison to be meaningful.
META_KEYS_COMPARED = ("model", "prompt", "n_predict", "seed", "temp", "n_ctx", "n_vocab", "top_k")

# Deterministic filler for the perplexity input file. llama-perplexity needs at
# least 2*n_ctx tokens, so this is repeated until the word count comfortably
# exceeds that; the text itself is arbitrary but must never change, or captures
# taken before and after a change stop being comparable.
DEFAULT_INPUT_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the parser reads every "
    "token in order and the backend dispatches each matrix multiplication to "
    "the accelerator without reordering the work. "
)


@dataclasses.dataclass
class Result:
    """Outcome of comparing two captures.

    `first_divergent_index` is -1 when the captures agree. `max_abs` / `max_rel`
    are reported even for a passing comparison so a near-miss is visible.
    """

    ok: bool
    max_abs: float
    max_rel: float
    first_divergent_index: int
    reason: str


def _relative(diff: float, x: float, y: float) -> float:
    scale = max(abs(x), abs(y))
    return diff / scale if scale > 0.0 else 0.0


def compare(a: list[dict[str, Any]], b: list[dict[str, Any]], tol: float = DEFAULT_TOL) -> Result:
    """Compare two capture entry lists, returning the worst divergence found.

    An index diverges when the token ids differ, when the top-k token rankings
    differ, when the logit vectors differ in length or presence, or when any
    logit differs by more than `tol` in absolute value. Scanning continues past
    the first divergence so `max_abs` and `max_rel` cover the whole overlap.

    Entries need only the keys they have: a token-mode capture carries `token`
    alone, a logit-mode capture adds `top_tokens` and `logits`.
    """
    if not a and not b:
        return Result(False, 0.0, 0.0, -1, "both captures are empty")

    max_abs = 0.0
    max_rel = 0.0
    first = -1
    reason = ""

    def diverge(index: int, why: str) -> None:
        nonlocal first, reason
        if first < 0:
            first = index
            reason = why

    for i in range(min(len(a), len(b))):
        entry_a = a[i]
        entry_b = b[i]

        token_a = entry_a.get("token")
        token_b = entry_b.get("token")
        if token_a != token_b:
            diverge(i, f"token id mismatch at index {i}: {token_a} != {token_b}")

        top_a = entry_a.get("top_tokens")
        top_b = entry_b.get("top_tokens")
        if top_a != top_b:
            diverge(i, f"top-k token ranking differs at index {i}: {top_a} != {top_b}")

        logits_a = entry_a.get("logits")
        logits_b = entry_b.get("logits")
        if (logits_a is None) != (logits_b is None):
            diverge(i, f"logits present in only one capture at index {i}")
            continue
        if logits_a is None:
            continue
        if len(logits_a) != len(logits_b):
            diverge(i, f"logit vector length mismatch at index {i}: {len(logits_a)} != {len(logits_b)}")
            continue

        for x, y in zip(logits_a, logits_b):
            diff = abs(float(x) - float(y))
            if diff > max_abs:
                max_abs = diff
            rel = _relative(diff, float(x), float(y))
            if rel > max_rel:
                max_rel = rel
            if diff > tol:
                diverge(i, f"logit divergence at index {i}: |{x} - {y}| = {diff:.6g} > tol {tol:.6g}")

    if len(a) != len(b):
        diverge(min(len(a), len(b)), f"capture length mismatch: {len(a)} vs {len(b)} entries")

    return Result(first < 0, max_abs, max_rel, first, reason)


def load_capture(path: pathlib.Path | str) -> list[dict[str, Any]]:
    """Read a capture file, accepting a bare entry list or an `entries` object."""
    obj = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
        return obj["entries"]
    raise ValueError(f"{path}: expected a list of entries or an object with an 'entries' list")


def load_meta(path: pathlib.Path | str) -> dict[str, Any]:
    """Read a capture's metadata, or an empty dict for a bare entry list."""
    obj = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("meta"), dict):
        return obj["meta"]
    return {}


def load_mode(path: pathlib.Path | str) -> str | None:
    """Read a capture's mode, or None for a bare entry list."""
    obj = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return obj.get("mode") if isinstance(obj, dict) else None


# --- llama-perplexity logits (.kld) binary format ---------------------------


@dataclasses.dataclass
class KldHeader:
    """Header of a `--kl-divergence-base` file. See the module docstring."""

    n_ctx: int
    n_vocab: int
    n_chunk: int
    tokens: list[int]

    @property
    def nv(self) -> int:
        """uint16 values per row: the vocabulary padded to even, plus 4 header slots."""
        return 2 * ((self.n_vocab + 1) // 2) + 4

    @property
    def rows_per_chunk(self) -> int:
        """Rows written per chunk -- perplexity scores only the second half of the window."""
        return self.n_ctx - 1 - self.n_ctx // 2

    @property
    def n_rows(self) -> int:
        return self.n_chunk * self.rows_per_chunk

    @property
    def row_bytes(self) -> int:
        return self.nv * 2

    @property
    def header_bytes(self) -> int:
        return len(KLD_MAGIC) + 12 + 4 * self.n_chunk * self.n_ctx

    @property
    def expected_bytes(self) -> int:
        """Exact size the file must have if it matches the layout this parser knows."""
        return self.header_bytes + self.n_rows * self.row_bytes


def read_kld_header(fh) -> KldHeader:
    """Read and validate the header of an open binary `.kld` stream."""
    magic = fh.read(len(KLD_MAGIC))
    if magic != KLD_MAGIC:
        raise ValueError(f"not a llama-perplexity logits file: magic is {magic!r}, expected {KLD_MAGIC!r}")

    fields = fh.read(12)
    if len(fields) != 12:
        raise ValueError("truncated logits file: header ends before n_ctx/n_vocab/n_chunk")
    n_ctx, n_vocab, n_chunk = struct.unpack("<3i", fields)
    if n_ctx <= 0 or n_vocab <= 0 or n_chunk <= 0:
        raise ValueError(f"implausible logits header: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}")
    if n_ctx > MAX_KLD_N_CTX or n_vocab > MAX_KLD_N_VOCAB or n_chunk > MAX_KLD_N_CHUNK:
        raise ValueError(
            f"implausibly large logits header: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk} "
            f"(ceilings {MAX_KLD_N_CTX}/{MAX_KLD_N_VOCAB}/{MAX_KLD_N_CHUNK}); the file is corrupt "
            "or is not a llama-perplexity logits file"
        )

    n_tokens = n_chunk * n_ctx
    raw = fh.read(n_tokens * 4)
    if len(raw) != n_tokens * 4:
        raise ValueError("truncated logits file: token array is short")
    tokens = list(struct.unpack(f"<{n_tokens}i", raw))

    return KldHeader(n_ctx=n_ctx, n_vocab=n_vocab, n_chunk=n_chunk, tokens=tokens)


def open_kld(path: pathlib.Path | str) -> KldHeader:
    """Read a `.kld` header and require the file to be exactly the size it implies.

    The format carries no version field, only an 8-byte magic, so a future
    change to the row layout in perplexity.cpp would otherwise be misparsed
    silently. A silently wrong "no divergence" verdict is the worst failure this
    tool can produce -- it would green-light a broken optimization -- so an
    unexpected size is a hard error, not a warning.
    """
    path = pathlib.Path(path)
    with open(path, "rb") as fh:
        header = read_kld_header(fh)

    actual = path.stat().st_size
    if actual != header.expected_bytes:
        raise ValueError(
            f"{path}: expected {header.expected_bytes} bytes for n_ctx={header.n_ctx} "
            f"n_vocab={header.n_vocab} n_chunk={header.n_chunk} "
            f"({header.header_bytes} header + {header.n_rows} rows x {header.row_bytes}), "
            f"but the file is {actual}. The file is truncated or corrupt, or the layout in "
            "tools/perplexity/perplexity.cpp has drifted from what this parser expects"
        )
    return header


def decode_kld_row(raw: bytes, n_vocab: int) -> tuple[float, float, array.array]:
    """Split one raw row into (scale, min_log_prob, quantized values)."""
    scale, min_log_prob = struct.unpack_from("<2f", raw, 0)
    quantized = array.array("H")
    quantized.frombytes(raw[8:8 + 2 * n_vocab])
    if sys.byteorder == "big":
        # The writer memcpy's native uint16; this host's files are little-endian.
        quantized.byteswap()
    return scale, min_log_prob, quantized


def dequantize_kld_row(scale: float, min_log_prob: float, quantized: array.array) -> list[float]:
    """Reconstruct a full log-probability vector: log_prob[i] = min_log_prob + scale*q[i]."""
    return [min_log_prob + scale * q for q in quantized]


def top_k_from_kld_row(scale: float, min_log_prob: float, quantized: array.array, k: int) -> tuple[list[int], list[float]]:
    """Return the k highest-probability (token ids, log-probabilities) of one row.

    Ranking happens in the quantized domain, which is monotonic in the
    log-probability, so it needs no dequantization. Ties break towards the lower
    token id, which keeps the ordering deterministic across runs.
    """
    k = max(1, min(k, len(quantized)))
    tokens = heapq.nlargest(k, range(len(quantized)), key=quantized.__getitem__)
    return tokens, [min_log_prob + scale * quantized[t] for t in tokens]


def iter_kld_rows(path: pathlib.Path | str) -> Iterator[tuple[int, float, float, array.array]]:
    """Yield `(row_index, scale, min_log_prob, quantized)` for each row in a `.kld` file."""
    header = open_kld(path)
    with open(path, "rb") as fh:
        fh.seek(header.header_bytes)
        for row in range(header.n_rows):
            raw = fh.read(header.row_bytes)
            if len(raw) != header.row_bytes:
                raise ValueError(f"{path}: truncated at row {row} of {header.n_rows}")
            yield (row, *decode_kld_row(raw, header.n_vocab))


def entries_from_kld(
    path: pathlib.Path | str,
    top_k: int = DEFAULT_TOP_K,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], KldHeader]:
    """Reduce a `.kld` file to top-k capture entries, one per scored row.

    Only the top k of each row is kept: the whole vector is ~64 KB per row for a
    32k vocabulary, which is far too much to carry in a JSON artifact, and a
    divergence large enough to matter shows up in the leading entries.
    """
    header = open_kld(path)

    # Row r of chunk c scores the token at position `first + 1 + r` of that chunk.
    first = header.n_ctx // 2
    entries: list[dict[str, Any]] = []
    for row, scale, min_log_prob, quantized in iter_kld_rows(path):
        if max_rows is not None and row >= max_rows:
            break
        tokens, logits = top_k_from_kld_row(scale, min_log_prob, quantized, top_k)
        chunk, offset = divmod(row, header.rows_per_chunk)
        target_index = chunk * header.n_ctx + first + offset + 1
        entries.append({
            "index": row,
            "token": tokens[0],
            "target": header.tokens[target_index] if target_index < len(header.tokens) else None,
            "top_tokens": tokens,
            "logits": logits,
        })
    return entries, header


def entries_from_token_ids(ids: list[int], pieces: list[str] | None = None) -> list[dict[str, Any]]:
    """Build capture entries from a greedy token id sequence."""
    entries: list[dict[str, Any]] = []
    for i, token in enumerate(ids):
        entry: dict[str, Any] = {"index": i, "token": int(token)}
        if pieces is not None and i < len(pieces):
            entry["piece"] = pieces[i]
        entries.append(entry)
    return entries


def parse_token_ids(text: str) -> list[int]:
    """Extract the id list printed by `llama-tokenize --ids`, ignoring log noise."""
    match = re.search(r"\[\s*(?:-?\d+\s*(?:,\s*-?\d+\s*)*)?\]", text)
    if match is None:
        raise ValueError("no token id list found in llama-tokenize output")
    return [int(tok) for tok in re.findall(r"-?\d+", match.group(0))]


def _env_with_device(device: str | None) -> dict[str, str]:
    """Process environment, with ONEAPI_DEVICE_SELECTOR set when a device was asked for."""
    env = dict(os.environ)
    if device:
        env["ONEAPI_DEVICE_SELECTOR"] = device
    return env


def run_completion(
    binary: pathlib.Path,
    model: pathlib.Path,
    prompt: str,
    n_predict: int,
    seed: int,
    device: str | None,
    n_gpu_layers: int | None,
    extra_args: list[str],
    timeout: float,
) -> str:
    """Run a deterministic greedy completion and return the generated text."""
    cmd = [
        str(binary),
        "-m", str(model),
        "-p", prompt,
        "-n", str(n_predict),
        "--seed", str(seed),
        "--temp", "0",
        "--no-display-prompt",
        "--simple-io",
    ]
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    cmd += extra_args

    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_device(device),
                          timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-completion failed ({proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def run_tokenize(binary: pathlib.Path, model: pathlib.Path, text: str, timeout: float) -> list[int]:
    """Re-tokenize generated text into token ids with `llama-tokenize --ids`."""
    cmd = [str(binary), "-m", str(model), "--stdin", "--ids", "--no-bos", "--no-escape", "--log-disable"]
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed ({proc.returncode}):\n{proc.stderr}")
    return parse_token_ids(proc.stdout)


def default_input_text(n_ctx: int, n_chunk: int) -> str:
    """Deterministic filler long enough for `n_chunk` perplexity chunks of `n_ctx`."""
    # llama-perplexity refuses inputs shorter than 2*n_ctx tokens, and needs
    # n_chunk*n_ctx to fill the requested chunks. One word is well under two
    # tokens, so four words per required token leaves ample headroom.
    words_needed = 4 * n_ctx * max(2, n_chunk)
    sentence_words = len(DEFAULT_INPUT_SENTENCE.split())
    repeats = -(-words_needed // sentence_words)
    return DEFAULT_INPUT_SENTENCE * repeats


def run_perplexity(
    binary: pathlib.Path,
    model: pathlib.Path,
    input_file: pathlib.Path,
    kld_out: pathlib.Path,
    n_ctx: int,
    n_chunk: int,
    seed: int,
    device: str | None,
    n_gpu_layers: int | None,
    extra_args: list[str],
    timeout: float,
) -> str:
    """Write a `--kl-divergence-base` logits file and return the tool's stderr log."""
    cmd = [
        str(binary),
        "-m", str(model),
        "-f", str(input_file),
        # -b == -c keeps n_parallel at 1, so the file's n_ctx is the context we asked
        # for rather than params.n_ctx/n_parallel (perplexity.cpp:2036).
        "-c", str(n_ctx),
        "-b", str(n_ctx),
        "--chunks", str(n_chunk),
        "--seed", str(seed),
        "--kl-divergence-base", str(kld_out),
    ]
    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    cmd += extra_args

    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_device(device),
                          timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed ({proc.returncode}):\n{proc.stderr}")
    if not kld_out.exists() or kld_out.stat().st_size == 0:
        raise RuntimeError(
            f"llama-perplexity wrote no logits to {kld_out}. The input usually tokenizes to fewer "
            f"than the required 2*n_ctx={2 * n_ctx} tokens -- use a longer --input-file.\n{proc.stderr}"
        )
    return proc.stderr


def capture_logits(args: argparse.Namespace) -> dict[str, Any]:
    """Capture top-k log-probabilities per scored token via llama-perplexity."""
    perplexity_bin = pathlib.Path(args.perplexity_bin)
    if not perplexity_bin.exists():
        raise FileNotFoundError(f"{perplexity_bin}: build llama-perplexity first (./scripts/sycl-build.sh)")

    with tempfile.TemporaryDirectory(prefix="sycl-logit-oracle-") as scratch:
        if args.input_file:
            input_file = pathlib.Path(args.input_file)
            if not input_file.exists():
                raise FileNotFoundError(f"{input_file}: --input-file does not exist")
        else:
            input_file = pathlib.Path(scratch) / "input.txt"
            input_file.write_text(default_input_text(args.n_ctx, args.n_chunk), encoding="utf-8")

        kld_out = pathlib.Path(args.kld_out) if args.kld_out else pathlib.Path(scratch) / "capture.kld"
        run_perplexity(
            perplexity_bin,
            pathlib.Path(args.model),
            input_file,
            kld_out,
            args.n_ctx,
            args.n_chunk,
            args.seed,
            args.device,
            args.n_gpu_layers,
            args.extra,
            args.timeout,
        )
        entries, header = entries_from_kld(kld_out, top_k=args.top_k, max_rows=args.max_rows)

    return {
        "mode": "logits",
        "meta": {
            "model": str(args.model),
            "input_file": str(input_file) if args.input_file else "<generated>",
            "n_ctx": header.n_ctx,
            "n_vocab": header.n_vocab,
            "n_chunk": header.n_chunk,
            "top_k": args.top_k,
            "seed": args.seed,
            "device": args.device,
            "n_gpu_layers": args.n_gpu_layers,
            "extra_args": args.extra,
        },
        "entries": entries,
    }


def capture_tokens(args: argparse.Namespace) -> dict[str, Any]:
    """Capture the greedy token id sequence via llama-completion + llama-tokenize."""
    completion_bin = pathlib.Path(args.completion_bin)
    tokenize_bin = pathlib.Path(args.tokenize_bin)
    if not completion_bin.exists():
        raise FileNotFoundError(f"{completion_bin}: build llama-completion first (./scripts/sycl-build.sh)")
    if not tokenize_bin.exists():
        raise FileNotFoundError(f"{tokenize_bin}: build llama-tokenize first (./scripts/sycl-build.sh)")

    text = run_completion(
        completion_bin,
        pathlib.Path(args.model),
        args.prompt,
        args.n_predict,
        args.seed,
        args.device,
        args.n_gpu_layers,
        args.extra,
        args.timeout,
    )
    ids = run_tokenize(tokenize_bin, pathlib.Path(args.model), text, args.timeout)

    return {
        "mode": "tokens",
        "meta": {
            "model": str(args.model),
            "prompt": args.prompt,
            "n_predict": args.n_predict,
            "seed": args.seed,
            "temp": 0.0,
            "device": args.device,
            "n_gpu_layers": args.n_gpu_layers,
            "extra_args": args.extra,
            "text": text,
        },
        "entries": entries_from_token_ids(ids),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    """Produce a capture object in the requested mode."""
    mode = getattr(args, "mode", "logits")
    if mode == "logits":
        return capture_logits(args)
    if mode == "tokens":
        return capture_tokens(args)
    raise ValueError(f"unknown capture mode {mode!r}: expected 'logits' or 'tokens'")


def cmd_capture(args: argparse.Namespace) -> int:
    obj = capture(args)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"captured {len(obj['entries'])} entries ({obj['mode']} mode) -> {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    entries_a = load_capture(args.a)
    entries_b = load_capture(args.b)

    mode_a = load_mode(args.a)
    mode_b = load_mode(args.b)
    if mode_a is not None and mode_b is not None and mode_a != mode_b:
        print(f"ERROR: capture modes differ ({mode_a} vs {mode_b}); they are not comparable", file=sys.stderr)
        return 2

    meta_a = load_meta(args.a)
    meta_b = load_meta(args.b)
    for key in META_KEYS_COMPARED:
        if key in meta_a and key in meta_b and meta_a[key] != meta_b[key]:
            print(f"warning: captures differ in {key}: {meta_a[key]!r} vs {meta_b[key]!r}", file=sys.stderr)

    result = compare(entries_a, entries_b, tol=args.tol)

    if args.json:
        print(json.dumps(dataclasses.asdict(result), indent=2))
    elif result.ok:
        print(
            f"OK: {len(entries_a)} entries identical within tol {args.tol:.6g} "
            f"(max_abs={result.max_abs:.6g}, max_rel={result.max_rel:.6g})"
        )
    else:
        print(f"DIVERGED at index {result.first_divergent_index}: {result.reason}")
        print(f"  max_abs={result.max_abs:.6g} max_rel={result.max_rel:.6g} tol={args.tol:.6g}")

    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic token/logit comparison oracle for SYCL changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Default mode is logit-level via llama-perplexity --kl-divergence-base (see module docstring).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="run a deterministic generation and write a capture JSON")
    cap.add_argument("--model", required=True, help="path to the GGUF model")
    cap.add_argument("--out", required=True, help="capture JSON output path")
    cap.add_argument("--mode", choices=("logits", "tokens"), default="logits",
                     help="logits: top-k log-probabilities via llama-perplexity (default). "
                          "tokens: greedy token ids via llama-completion (fallback)")
    cap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (kept fixed for determinism)")
    cap.add_argument("--device", default=None, help="value for ONEAPI_DEVICE_SELECTOR, e.g. level_zero:1")
    cap.add_argument("-ngl", "--n-gpu-layers", type=int, default=None, help="layers to offload to the GPU")
    cap.add_argument("--timeout", type=float, default=600.0, help="per-subprocess timeout in seconds")

    logit_group = cap.add_argument_group("logits mode")
    logit_group.add_argument("-f", "--input-file", default=None,
                             help="text to evaluate; must tokenize to >= 2*--n-ctx tokens "
                                  "(default: a fixed generated filler)")
    logit_group.add_argument("-c", "--n-ctx", type=int, default=DEFAULT_CTX, help="per-chunk context size")
    logit_group.add_argument("--n-chunk", type=int, default=DEFAULT_CHUNKS, help="chunks to evaluate")
    logit_group.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="log-probabilities kept per row")
    logit_group.add_argument("--max-rows", type=int, default=None, help="stop after this many scored rows")
    logit_group.add_argument("--kld-out", default=None, help="keep the raw .kld binary at this path")
    logit_group.add_argument("--perplexity-bin", default="./build/bin/llama-perplexity", help="llama-perplexity path")

    token_group = cap.add_argument_group("tokens mode")
    token_group.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"prompt (default: {DEFAULT_PROMPT!r})")
    token_group.add_argument("-n", "--n-predict", type=int, default=DEFAULT_N_PREDICT, help="tokens to generate")
    token_group.add_argument("--completion-bin", default="./build/bin/llama-completion", help="llama-completion path")
    token_group.add_argument("--tokenize-bin", default="./build/bin/llama-tokenize", help="llama-tokenize path")

    cap.add_argument("extra", nargs="*", help="extra args forwarded to the underlying llama.cpp tool")
    cap.set_defaults(func=cmd_capture)

    cmp_ = sub.add_parser("compare", help="compare two capture JSON files")
    cmp_.add_argument("a", help="reference capture JSON")
    cmp_.add_argument("b", help="candidate capture JSON")
    cmp_.add_argument("--tol", type=float, default=DEFAULT_TOL,
                      help=f"absolute tolerance (default {DEFAULT_TOL}, which demands an exact match). "
                           "The .kld quantization step is up to ~2.4e-4, so use --tol 3e-4 to allow a "
                           "one-step wobble when comparing real hardware runs")
    cmp_.add_argument("--json", action="store_true", help="emit the result as JSON")
    cmp_.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
