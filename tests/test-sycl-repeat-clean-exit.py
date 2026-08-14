#!/usr/bin/env python3
"""Run a device fixture repeatedly and require every process to exit cleanly."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} REPEATS BINARY", file=sys.stderr)
        return 2

    repeats = int(sys.argv[1])
    binary = sys.argv[2]
    if repeats < 1:
        print("REPEATS must be positive", file=sys.stderr)
        return 2

    for run in range(1, repeats + 1):
        completed = subprocess.run([binary], text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
        sys.stdout.write(completed.stdout)
        if completed.returncode == 77:
            print(f"SKIP {binary}: device unavailable on repeat {run}", file=sys.stderr)
            return 77
        if completed.returncode != 0:
            detail = (f"signal {-completed.returncode}" if completed.returncode < 0
                      else f"exit {completed.returncode}")
            print(f"FAIL {binary}: repeat {run}/{repeats} ended with {detail}", file=sys.stderr)
            return 1
        print(f"PASS {binary}: clean exit {run}/{repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
