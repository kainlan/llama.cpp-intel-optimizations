#!/usr/bin/env python3
"""Portable deterministic M1-M3 lifecycle mutation runner."""

import subprocess
import sys
from typing import Optional

CASES = {
    "M1": ("stale-generation",),
    "M2": ("nested-success",),
    "M3": ("inner-failure", "cancel", "wrong-txn", "depth-overflow"),
}
MARKERS = {
    "M1": "stale slot generation accepted\n",
    "M2": "nested load committed\n",
    "M3": "poisoned transaction published LIVE\n",
}


def run(binary: str, case: str, mutation: Optional[str] = None) -> subprocess.CompletedProcess:
    command = [binary, "--case", case]
    if mutation is not None:
        command += ["--mutation", mutation]
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] not in CASES:
        print(f"usage: {sys.argv[0]} TEST_BINARY M1|M2|M3", file=sys.stderr)
        return 2
    binary, mutation = sys.argv[1:]
    for case in CASES[mutation]:
        baseline = run(binary, case)
        if baseline.returncode != 0:
            sys.stderr.write(baseline.stdout)
            return baseline.returncode

        mutant = run(binary, case, mutation)
        if mutant.returncode == 0:
            print(f"mutant unexpectedly passed: {mutation} case={case}", file=sys.stderr)
            return 1
        lines = [line + "\n" for line in mutant.stdout.splitlines()]
        if MARKERS[mutation] not in lines:
            sys.stderr.write(mutant.stdout)
            print(f"{mutation} case lacked exact mutation marker: {case}", file=sys.stderr)
            return 1

        replay = run(binary, case)
        if replay.returncode != 0:
            sys.stderr.write(replay.stdout)
            return replay.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
