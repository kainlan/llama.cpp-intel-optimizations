#!/usr/bin/env python3
"""Integration checks for the static-storage census: the committed inventory
must match the current sources, and --check must detect drift.

Pass --check to run only the committed-inventory drift gate
(CommittedInventoryDriftTest); with no flag every test runs, which is what the
registered ctest target does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/audit-sycl-static-storage.py"
INPUTS = (
    "ggml/src/ggml-sycl/ggml-sycl.cpp",
    "ggml/src/ggml-sycl/unified-cache.cpp",
    "ggml/src/ggml-sycl/unified-cache.hpp",
    "ggml/src/ggml-sycl/fattn.cpp",
    "ggml/src/ggml-sycl/layer-streaming.cpp",
)
INVENTORY = "docs/backend/sycl-static-storage-inventory.csv"
STALE_MESSAGE = "is stale; regenerate without --check"


def audit_command(*args: str) -> list[str]:
    return [sys.executable, str(SCRIPT), *args]


def plant_stale_row(inventory: str) -> str:
    """Return the inventory with the first data row's line number shifted by one,
    re-serialised exactly as the generator writes it, so that row is the only
    byte difference."""
    lines = inventory.splitlines(keepends=True)
    if len(lines) < 2:
        raise AssertionError(f"{INVENTORY} has no data row to plant a stale copy of")
    row = next(csv.reader([lines[1]]))

    def render(fields: list[str]) -> str:
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerow(fields)
        return buffer.getvalue()

    if render(row) != lines[1]:
        raise AssertionError("re-serialising the first data row changed its bytes; the planted diff would not be one row")
    row[1] = str(int(row[1]) + 1)
    lines[1] = render(row)
    return "".join(lines)


class CommittedInventoryDriftTest(unittest.TestCase):
    """The gate: the committed inventory must be exactly what --check regenerates
    from the current sources, and a stale row in it must make --check fail."""

    def test_committed_inventory_matches_current_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            planted_path = Path(directory) / "planted-inventory.csv"
            planted_path.write_text(
                plant_stale_row((REPO / INVENTORY).read_text(encoding="utf-8")),
                encoding="utf-8",
            )
            # The gate and its positive control parse the same sources and differ
            # only in the one planted row, so run the two parses concurrently.
            # An absolute --output replaces the repo-relative default.
            gate = subprocess.Popen(
                audit_command("--check"), cwd=REPO, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            control = subprocess.Popen(
                audit_command("--check", "--output", str(planted_path)), cwd=REPO, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _, gate_stderr = gate.communicate()
            _, control_stderr = control.communicate()

        self.assertEqual(control.returncode, 1, control_stderr)
        self.assertIn(STALE_MESSAGE, control_stderr)
        self.assertEqual(
            gate.returncode, 0,
            f"{INVENTORY} does not match the current sources; regenerate it with "
            f"`python3 scripts/audit-sycl-static-storage.py` and commit the result.\n{gate_stderr}",
        )


class StaticStorageAuditIntegrationTest(unittest.TestCase):
    def run_audit(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            audit_command(*args),
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_self_test_ignores_live_source_line_drift_but_check_detects_it(self) -> None:
        current = self.run_audit("--self-test")
        self.assertEqual(current.returncode, 0, current.stderr)
        self.assertIn("synthetic-file-tail-scopes", current.stdout)

        with tempfile.TemporaryDirectory() as empty_directory:
            empty_root = self.run_audit("--self-test", "--repo", empty_directory)
            self.assertEqual(empty_root.returncode, 0, empty_root.stderr)

        with tempfile.TemporaryDirectory() as directory:
            shifted_repo = Path(directory)
            for relative in INPUTS:
                destination = shifted_repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / relative, destination)

            # Establish a known-matching baseline instead of assuming the
            # repository's audited snapshot happens to match integration HEAD.
            generated = self.run_audit("--repo", str(shifted_repo))
            self.assertEqual(generated.returncode, 0, generated.stderr)
            inventory_path = shifted_repo / INVENTORY
            baseline_checksum = hashlib.sha256(inventory_path.read_bytes()).hexdigest()

            baseline_check = self.run_audit("--check", "--repo", str(shifted_repo))
            self.assertEqual(baseline_check.returncode, 0, baseline_check.stderr)

            source_path = shifted_repo / INPUTS[0]
            source_path.write_text(
                "// unrelated prepended line\n" + source_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            shifted_check = self.run_audit("--check", "--repo", str(shifted_repo))
            self.assertEqual(shifted_check.returncode, 1, shifted_check.stderr)
            self.assertIn(f"ERROR: {INVENTORY} {STALE_MESSAGE}", shifted_check.stderr)
            self.assertEqual(
                hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                baseline_checksum,
                "--check must diagnose drift without rewriting the inventory",
            )


def missing_runtime_dependency() -> str | None:
    if shutil.which("g++") is None:
        return "g++ is required by the census compiler fixture"
    try:
        version("tree-sitter")
        version("tree-sitter-language-pack")
    except PackageNotFoundError as exc:
        return f"missing pinned parser dependency: {exc}"
    return None


if __name__ == "__main__":
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--check", action="store_true")
    options, unittest_args = cli.parse_known_args()
    missing = missing_runtime_dependency()
    if missing:
        print(f"SKIP: {missing}", file=sys.stderr)
        raise SystemExit(77)
    if options.check:
        unittest_args.append(CommittedInventoryDriftTest.__name__)
    unittest.main(argv=[sys.argv[0], *unittest_args])
