#!/usr/bin/env python3
"""Integration checks for the static-storage census: the committed inventory
must classify the same static objects as the current sources, and both
--check-classification and the byte-exact --check must detect what they gate.

Pass --check to run only the committed-inventory drift gate
(CommittedInventoryDriftTest), which runs the script's --check-classification,
not its byte-exact --check; with no flag every test runs, which is what the
registered ctest target does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
DRIFT_MESSAGE = "classification drift"
LINE_ONLY_NOTE = "line/evidence columns differ"
PLANTED_DISPOSITION = "planted: classification control"
PLANTED_STATIC = "g_static_storage_audit_planted"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_sycl_static_storage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_command(*args: str) -> list[str]:
    return [sys.executable, str(SCRIPT), *args]


def render(fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(fields)
    return buffer.getvalue()


def plant_first_row(inventory: str, column: int, value) -> str:
    """Return the inventory with one column of the first data row replaced by
    value(old), re-serialised exactly as the generator writes it, so that row
    is the only byte difference."""
    lines = inventory.splitlines(keepends=True)
    if len(lines) < 2:
        raise AssertionError(f"{INVENTORY} has no data row to plant a stale copy of")
    row = next(csv.reader([lines[1]]))
    if render(row) != lines[1]:
        raise AssertionError("re-serialising the first data row changed its bytes; the planted diff would not be one row")
    row[column] = value(row[column])
    lines[1] = render(row)
    return "".join(lines)


def plant_line_shift(inventory: str) -> str:
    return plant_first_row(inventory, 1, lambda line: str(int(line) + 1))


def plant_reclassification(inventory: str) -> str:
    return plant_first_row(inventory, 10, lambda _: PLANTED_DISPOSITION)


def read_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


class ClassificationProjectionTest(unittest.TestCase):
    """The comparator the ctest gate uses, driven by the committed CSV and
    mutated copies of it -- no census parse."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = load_audit_module()
        cls.committed = read_rows((REPO / INVENTORY).read_text(encoding="utf-8"))

    def drift(self, current: list[dict[str, str]]):
        return self.audit.classification_drift(self.committed, current)

    def test_classification_columns_are_every_column_but_the_informational_ones(self) -> None:
        self.assertEqual(
            self.audit.CLASSIFICATION_COLUMNS,
            ("file", "symbol", "type", "scope", "mutability", "synchronization",
             "owner_identity", "reset_teardown_disposition"),
        )
        self.assertEqual(self.audit.CLASSIFICATION_COLUMNS, self.audit.classification_columns(self.audit.COLUMNS))
        # A column added to COLUMNS later is gated unless it is deliberately
        # declared informational.
        self.assertIn("new_column", self.audit.classification_columns((*self.audit.COLUMNS, "new_column")))

    def test_committed_inventory_has_no_drift_against_itself(self) -> None:
        self.assertEqual(self.drift([dict(row) for row in self.committed]), ({}, {}))

    def test_line_and_evidence_shift_is_not_drift(self) -> None:
        # What a blank line near the top of an input does: every line number and
        # every L<n>: cite in the evidence and disposition columns moves.
        def shift(value: str) -> str:
            return self.audit.re.sub(r"\bL(\d+):", lambda m: f"L{int(m.group(1)) + 1}:", value)

        shifted = []
        for row in self.committed:
            moved = {column: shift(value) for column, value in row.items()}
            moved["line"] = str(int(row["line"]) + 1)
            moved["writer_evidence"] = "none found by unscoped lexical scan"
            shifted.append(moved)
        self.assertNotEqual(
            [row["reset_teardown_disposition"] for row in shifted],
            [row["reset_teardown_disposition"] for row in self.committed],
            "the shift must reach the disposition cites, or this test proves nothing about them",
        )
        self.assertEqual(self.drift(shifted), ({}, {}))

    def test_added_namespace_static_is_drift_and_named(self) -> None:
        planted = dict(self.committed[0])
        planted.update(symbol=PLANTED_STATIC, scope="namespace:ggml_sycl", type="int", line="1")
        added, removed = self.drift([*self.committed, planted])
        self.assertEqual(removed, {})
        self.assertEqual(list(added.values()), [1])
        self.assertIn(PLANTED_STATIC, self.audit.format_classification_drift(added, removed))

    def test_changed_classification_column_is_drift(self) -> None:
        # Spelled out rather than read from the key, so dropping a column from
        # the key fails here instead of silently skipping its subtest.
        for column in ("file", "symbol", "type", "scope", "mutability", "synchronization",
                       "owner_identity", "reset_teardown_disposition"):
            with self.subTest(column=column):
                current = [dict(row) for row in self.committed]
                current[0][column] = "planted"
                added, removed = self.drift(current)
                self.assertEqual((list(added.values()), list(removed.values())), ([1], [1]))
                report = self.audit.format_classification_drift(added, removed)
                self.assertIn(self.committed[0]["symbol"], report)
                self.assertIn("added", report)
                self.assertIn("removed", report)

    def test_duplicate_rows_compare_as_a_multiset(self) -> None:
        key = self.audit.classification_key
        counts = self.audit.Counter(map(key, self.committed))
        duplicated = next(row for row in self.committed if counts[key(row)] > 1)
        current = [dict(row) for row in self.committed]
        current.remove(duplicated)
        added, removed = self.drift(current)
        self.assertEqual(added, {})
        self.assertEqual(removed, {key(duplicated): 1})
        self.assertIn("removed x1", self.audit.format_classification_drift(added, removed))


class CommittedInventoryDriftTest(unittest.TestCase):
    """The gate: the committed inventory must classify the same static objects
    as a fresh census of the current sources. A reclassified row planted into it
    must fail the gate, and a line number shifted in it must not."""

    def test_committed_inventory_matches_current_sources(self) -> None:
        committed = (REPO / INVENTORY).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            reclassified_path = Path(directory) / "reclassified-inventory.csv"
            reclassified_path.write_text(plant_reclassification(committed), encoding="utf-8")
            shifted_path = Path(directory) / "line-shifted-inventory.csv"
            shifted_path.write_text(plant_line_shift(committed), encoding="utf-8")
            # The gate and its two controls parse the same sources and differ
            # only in the one planted row, so run the three parses concurrently.
            # That attribution relies on the census being deterministic: two
            # parses of the same bytes must render the same CSV, otherwise a
            # control's verdict could come from parse variance, not the planted
            # row. An absolute --output replaces the repo-relative default.
            runs = {
                name: subprocess.Popen(
                    audit_command("--check-classification", *output), cwd=REPO, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                for name, output in (
                    ("gate", ()),
                    ("reclassified", ("--output", str(reclassified_path))),
                    ("line-shifted", ("--output", str(shifted_path))),
                )
            }
            results = {name: (run, *run.communicate()) for name, run in runs.items()}

        reclassified, _, reclassified_stderr = results["reclassified"]
        self.assertEqual(reclassified.returncode, 1, reclassified_stderr)
        self.assertIn(DRIFT_MESSAGE, reclassified_stderr)
        self.assertIn(PLANTED_DISPOSITION, reclassified_stderr)

        shifted, shifted_stdout, shifted_stderr = results["line-shifted"]
        self.assertEqual(shifted.returncode, 0, shifted_stderr)
        self.assertIn(LINE_ONLY_NOTE, shifted_stdout)

        gate, _, gate_stderr = results["gate"]
        self.assertEqual(
            gate.returncode, 0,
            f"{INVENTORY} does not classify the same static objects as the current "
            f"sources; regenerate it with `python3 scripts/audit-sycl-static-storage.py`, "
            f"review the reported rows, and commit the result.\n{gate_stderr}",
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

    def test_self_test_ignores_live_source_line_drift_and_the_two_checks_split_it(self) -> None:
        current = self.run_audit("--self-test")
        self.assertEqual(current.returncode, 0, current.stderr)
        self.assertIn("synthetic-file-tail-scopes", current.stdout)

        with tempfile.TemporaryDirectory() as empty_directory:
            empty_root = self.run_audit("--self-test", "--repo", empty_directory)
            self.assertEqual(empty_root.returncode, 0, empty_root.stderr)

        with tempfile.TemporaryDirectory() as directory:
            baseline_repo, shifted_repo, added_repo = (Path(directory) / name for name in ("baseline", "shifted", "added"))
            for relative in INPUTS:
                destination = baseline_repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / relative, destination)

            # Establish a known-matching baseline instead of assuming the
            # repository's audited snapshot happens to match integration HEAD.
            generated = self.run_audit("--repo", str(baseline_repo))
            self.assertEqual(generated.returncode, 0, generated.stderr)
            baseline_checksum = hashlib.sha256((baseline_repo / INVENTORY).read_bytes()).hexdigest()
            shutil.copytree(baseline_repo, shifted_repo)
            shutil.copytree(baseline_repo, added_repo)

            # A blank line near the top moves every later row: byte drift only.
            shifted_source = shifted_repo / INPUTS[0]
            shifted_source.write_text("\n" + shifted_source.read_text(encoding="utf-8"), encoding="utf-8")
            # A new namespace-scope static is a new object: classification drift.
            added_source = added_repo / INPUTS[-1]
            added_source.write_text(
                added_source.read_text(encoding="utf-8")
                + f"\nnamespace ggml_sycl {{\nstatic int {PLANTED_STATIC} = 0;\n}}\n",
                encoding="utf-8",
            )

            # Five parses of four trees, independent of each other: run them
            # concurrently so the test costs one more parse of wall time, not five.
            runs = {
                name: subprocess.Popen(
                    audit_command(*args, "--repo", str(repo)), cwd=REPO, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                for name, args, repo in (
                    ("baseline-check", ("--check",), baseline_repo),
                    ("baseline-classification", ("--check-classification",), baseline_repo),
                    ("shifted-check", ("--check",), shifted_repo),
                    ("shifted-classification", ("--check-classification",), shifted_repo),
                    ("added-classification", ("--check-classification",), added_repo),
                )
            }
            results = {name: (run, *run.communicate()) for name, run in runs.items()}

            for name in ("baseline-check", "baseline-classification"):
                run, stdout, stderr = results[name]
                self.assertEqual(run.returncode, 0, f"{name}: {stderr}")
                self.assertNotIn(LINE_ONLY_NOTE, stdout, name)

            run, _, stderr = results["shifted-check"]
            self.assertEqual(run.returncode, 1, stderr)
            self.assertIn(f"ERROR: {INVENTORY} {STALE_MESSAGE}", stderr)

            run, stdout, stderr = results["shifted-classification"]
            self.assertEqual(run.returncode, 0, stderr)
            self.assertIn(LINE_ONLY_NOTE, stdout)

            run, _, stderr = results["added-classification"]
            self.assertEqual(run.returncode, 1, stderr)
            self.assertIn(DRIFT_MESSAGE, stderr)
            self.assertIn(f"symbol={PLANTED_STATIC}", stderr)
            self.assertIn("scope=namespace:ggml_sycl", stderr)

            for repo in (baseline_repo, shifted_repo, added_repo):
                self.assertEqual(
                    hashlib.sha256((repo / INVENTORY).read_bytes()).hexdigest(),
                    baseline_checksum,
                    "--check and --check-classification must diagnose drift without rewriting the inventory",
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
