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

Granularity: TOKEN-LEVEL, not logit-level
-----------------------------------------
The plan asked for a top-k logit capture via `llama-completion --logits-file`.
That flag does **not** exist in this tree -- `./build/bin/llama-completion --help`
exposes no logit-dump option of any kind (only `--logit-bias`, a sampler input).
Per the plan's documented fallback, `capture` therefore records the **greedy
token id sequence** at `--temp 0 --seed 42`: the completion is generated with
`llama-completion` and the emitted text is turned back into token ids with
`llama-tokenize --ids`. This is an exact-equality oracle on the generated token
stream, not a numerical-drift oracle on logits.

Two consequences of the fallback, stated plainly:

* Token ids come from re-tokenizing the generated text, so they are a
  deterministic function of that text rather than the ids the sampler actually
  picked. Equal text always yields equal ids; a divergence in the text shows up
  as a divergence in the ids, but `first_divergent_index` is the index in the
  re-tokenized stream and may be off by a token from the true sampling step.
* Sub-threshold numerical drift that does not flip any greedy argmax is
  invisible here. A logit-level capture would see it; this cannot.

The comparison core (`compare`) is already logit-aware: any capture entry may
carry a `"logits"` list and it will be compared with an absolute tolerance. If a
logit-dump flag lands in `llama-completion` later, only `capture` needs to
change -- write `"logits"` into each entry and the oracle becomes logit-level
with no change to the comparison, the tolerance handling, or the exit codes.

Capture format
--------------
JSON, either a bare list of entries or an object with an ``entries`` key::

    {
      "mode": "token",
      "meta": {"model": "...", "prompt": "...", "seed": 42, ...},
      "entries": [{"index": 0, "token": 28740}, {"index": 1, "token": 28725}]
    }

Each entry has a ``token`` id and may have a ``logits`` list and a ``piece``
string. Extra keys are ignored by the comparison.

Usage
-----
Capture a reference and a candidate, then compare them::

    ./scripts/sycl-logit-oracle.py capture \\
        --model /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \\
        --device level_zero:1 --out /tmp/before.json

    # ... rebuild with the change ...

    ./scripts/sycl-logit-oracle.py capture \\
        --model /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \\
        --device level_zero:1 --out /tmp/after.json

    ./scripts/sycl-logit-oracle.py compare /tmp/before.json /tmp/after.json

`compare` exits non-zero when the divergence exceeds the tolerance, so it can
gate a commit directly. Both captures must be produced with the same model,
prompt, seed and token count; `compare` warns when the recorded metadata
differs.

Pure stdlib on purpose: run it with `/usr/bin/python3` (the conda python on this
host has a broken numpy), and do not add third-party imports.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

DEFAULT_PROMPT = "1, 2, 3, 4, 5,"
DEFAULT_N_PREDICT = 15
DEFAULT_SEED = 42
DEFAULT_TOL = 1e-6

# Metadata keys that must agree for a comparison to be meaningful.
META_KEYS_COMPARED = ("model", "prompt", "n_predict", "seed", "temp")


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

    An index diverges when the token ids differ, when the logit vectors differ
    in length or presence, or when any logit differs by more than `tol` in
    absolute value. Scanning continues past the first divergence so `max_abs`
    and `max_rel` cover the whole overlap.
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

    env = dict(os.environ)
    if device:
        env["ONEAPI_DEVICE_SELECTOR"] = device

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, check=False)
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


def capture(args: argparse.Namespace) -> dict[str, Any]:
    """Produce a capture object for one deterministic generation."""
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
        "mode": "token",
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


def cmd_capture(args: argparse.Namespace) -> int:
    obj = capture(args)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"captured {len(obj['entries'])} tokens ({obj['mode']}-level) -> {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    entries_a = load_capture(args.a)
    entries_b = load_capture(args.b)

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
        epilog="Captures are token-level: llama-completion has no logit-dump flag (see module docstring).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="run a deterministic generation and write a capture JSON")
    cap.add_argument("--model", required=True, help="path to the GGUF model")
    cap.add_argument("--out", required=True, help="capture JSON output path")
    cap.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"prompt (default: {DEFAULT_PROMPT!r})")
    cap.add_argument("-n", "--n-predict", type=int, default=DEFAULT_N_PREDICT, help="tokens to generate")
    cap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (kept fixed for determinism)")
    cap.add_argument("--device", default=None, help="value for ONEAPI_DEVICE_SELECTOR, e.g. level_zero:1")
    cap.add_argument("-ngl", "--n-gpu-layers", type=int, default=None, help="layers to offload to the GPU")
    cap.add_argument("--completion-bin", default="./build/bin/llama-completion", help="llama-completion path")
    cap.add_argument("--tokenize-bin", default="./build/bin/llama-tokenize", help="llama-tokenize path")
    cap.add_argument("--timeout", type=float, default=600.0, help="per-subprocess timeout in seconds")
    cap.add_argument("extra", nargs="*", help="extra args forwarded to llama-completion")
    cap.set_defaults(func=cmd_capture)

    cmp_ = sub.add_parser("compare", help="compare two capture JSON files")
    cmp_.add_argument("a", help="reference capture JSON")
    cmp_.add_argument("b", help="candidate capture JSON")
    cmp_.add_argument("--tol", type=float, default=DEFAULT_TOL, help=f"absolute tolerance (default {DEFAULT_TOL})")
    cmp_.add_argument("--json", action="store_true", help="emit the result as JSON")
    cmp_.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
