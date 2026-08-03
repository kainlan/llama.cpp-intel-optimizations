#!/usr/bin/env python3
"""Integration checks for repository-independent census self-tests and drift checks."""

from __future__ import annotations

import hashlib
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


class StaticStorageAuditIntegrationTest(unittest.TestCase):
    def run_audit(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
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
            self.assertIn(f"ERROR: {INVENTORY} is stale", shifted_check.stderr)
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
    missing = missing_runtime_dependency()
    if missing:
        print(f"SKIP: {missing}", file=sys.stderr)
        raise SystemExit(77)
    unittest.main()
