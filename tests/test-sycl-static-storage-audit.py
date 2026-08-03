#!/usr/bin/env python3
"""Integration checks for repository-independent census self-tests and drift checks."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
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

        with tempfile.TemporaryDirectory() as directory:
            shifted_repo = Path(directory)
            for relative in (*INPUTS, INVENTORY):
                destination = shifted_repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if relative == INPUTS[0]:
                    source = (REPO / relative).read_text(encoding="utf-8")
                    destination.write_text("// unrelated prepended line\n" + source, encoding="utf-8")
                else:
                    shutil.copyfile(REPO / relative, destination)

            shifted_self_test = self.run_audit("--self-test", "--repo", str(shifted_repo))
            self.assertEqual(shifted_self_test.returncode, 0, shifted_self_test.stderr)

            shifted_check = self.run_audit("--check", "--repo", str(shifted_repo))
            self.assertEqual(shifted_check.returncode, 1, shifted_check.stderr)
            self.assertIn(f"ERROR: {INVENTORY} is stale", shifted_check.stderr)


if __name__ == "__main__":
    unittest.main()
