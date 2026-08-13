#!/usr/bin/env python3
"""Reject ordinary-SYCL linkage and duplicate backend globals in private carriers."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


def run(*args: str) -> str:
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed ({completed.returncode}):\n{completed.stdout}")
    return completed.stdout


def audit(binary_arg: str) -> None:
    binary = pathlib.Path(binary_arg)
    if not binary.is_file():
        raise RuntimeError(f"private carrier binary does not exist: {binary}")

    dependencies = run("ldd", str(binary))
    if re.search(r"(?:^|[/\s])libggml-sycl(?:\.so(?:\.[^\s]+)?)?(?:\s|$)", dependencies):
        raise RuntimeError(
            f"{binary} loads the ordinary libggml-sycl beside its embedded backend:\n{dependencies}")

    symbols = run("nm", "-C", "--defined-only", str(binary))
    definitions = [line for line in symbols.splitlines()
                   if re.search(r"\b[A-Z]\s+g_tp_device1_worker$", line)]
    if len(definitions) != 1:
        raise RuntimeError(
            f"{binary} must contain exactly one g_tp_device1_worker definition; "
            f"found {len(definitions)}:\n" + "\n".join(definitions))

    print(f"PASS {binary}: no libggml-sycl DT_NEEDED; one g_tp_device1_worker")


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} BINARY...", file=sys.stderr)
        return 2
    try:
        for binary in sys.argv[1:]:
            audit(binary)
    except (OSError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
