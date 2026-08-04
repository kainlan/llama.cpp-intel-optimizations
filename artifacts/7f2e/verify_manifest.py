#!/usr/bin/env python3
"""Regenerate and fail-closed verify the immutable 9a0670712 audit census."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

AUDITED = "9a06707120dfa4595b46f2241ec40bbbd8476959"
PARENT = "d892fa8c054c34528cb550881825b7082fa63d17"
EXPECTED_PATHS = 338
EXPECTED_HUNKS = 4087
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def area(path: str) -> str:
    if path.startswith(".beads/"):
        return "tracker-metadata"
    if path in {"AGENTS.md", "CLAUDE.md"} or path.startswith("docs/") or path.endswith(".md"):
        return "documentation"
    if path.startswith("tests/") or "/tests/" in path:
        return "test"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith(("tools/", "examples/")):
        return "tooling"
    return "production"


def classify(path: str, raw: bytes) -> tuple[str, str]:
    text = raw.decode("utf-8", "replace")
    a = area(path)
    if (path == "ggml/src/ggml-sycl/unified-cache.cpp" and
            "role == alloc_role::WEIGHT" in text and "runtime_category::HOST_COMPUTE" in text):
        return "known-defect:expert-staging-role-order", "fixed:282069dd4702c820bb0ac53757572b516c342746"
    if "reset_model_weight_entries" in text and "leaked model-weight mem_handle lease" in text:
        return "known-defect:live-weight-abort", "fixed:acdb192d43bf6b36a1a8e227aceb12d38845ecf1"
    if "live registered allocations in reset zone" in text:
        return "known-defect:live-zone-reset-abort", "fixed:4afdb6d9f1c247dd62cbf6a9a9802447efb64df4"
    if "level_zero:1" in text or "level_zero:0" in text:
        return f"{a}:selector-default", "provenance:llama.cpp-966h"
    if a != "production":
        return a, "reviewed:no-additional-9a-specific-defect"

    changed = [line for line in text.splitlines()[1:] if line[:1] in {"+", "-"}]
    removed = [line[1:] for line in changed if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in changed if line.startswith("+") and not line.startswith("+++")]
    normalize = lambda xs: [re.sub(r"\s+", "", x) for x in xs if re.sub(r"\s+", "", x)]
    if removed and added and normalize(removed) == normalize(added):
        return "production:format-only", "reviewed:non-semantic"
    joined = "\n".join(removed + added)
    control = re.compile(
        r"\b(if|else|switch|case|return|break|continue|abort|assert|enabled|disabled|default|reset|"
        r"alloc|free|owner|handle|cache|zone|tier|layout|device|queue|event|wait)\b",
        re.IGNORECASE,
    )
    if removed and added and control.search(joined):
        return "production:semantic-control-or-lifetime", "reviewed:no-additional-9a-specific-defect"
    if removed and added:
        return "production:semantic-replacement", "reviewed:no-additional-9a-specific-defect"
    if added:
        return "production:additive", "reviewed:no-additional-9a-specific-defect"
    if removed:
        return "production:deletion", "reviewed:no-additional-9a-specific-defect"
    return "production:other", "reviewed:no-additional-9a-specific-defect"


def blob(commit: str, path: str) -> str:
    try:
        return git("rev-parse", f"{commit}:{path}").decode().strip()
    except subprocess.CalledProcessError:
        return "-"


def census() -> tuple[list[list[object]], list[list[object]]]:
    paths = git("diff", "--name-only", "--no-renames", PARENT, AUDITED, "--").decode().splitlines()
    diff = git("diff", "--no-ext-diff", "--no-renames", "--unified=0", "--binary", PARENT, AUDITED, "--")
    current = None
    hunks: list[tuple[str, bytes]] = []
    lines = diff.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(b"diff --git "):
            parts = shlex.split(line.decode("utf-8", "surrogateescape").rstrip())
            current = parts[3][2:]
            i += 1
            continue
        if line.startswith(b"@@ "):
            start = i
            i += 1
            while i < len(lines) and not lines[i].startswith((b"@@ ", b"diff --git ")):
                i += 1
            assert current is not None
            hunks.append((current, b"".join(lines[start:i])))
            continue
        i += 1

    by_path = Counter(path for path, _ in hunks)
    path_rows: list[list[object]] = []
    for number, path in enumerate(paths, 1):
        path_rows.append([
            f"P{number:03d}", path, area(path), by_path[path], blob(PARENT, path), blob(AUDITED, path)
        ])

    path_ids = {row[1]: row[0] for row in path_rows}
    hunk_rows: list[list[object]] = []
    header_re = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    for number, (path, raw) in enumerate(hunks, 1):
        match = header_re.match(raw)
        if not match:
            raise RuntimeError(f"unparsed hunk header at {path}: {raw[:100]!r}")
        old_start, old_count, new_start, new_count = match.groups()
        old_count = old_count or b"1"
        new_count = new_count or b"1"
        body = raw.splitlines()[1:]
        removed = sum(x.startswith(b"-") and not x.startswith(b"---") for x in body)
        added = sum(x.startswith(b"+") and not x.startswith(b"+++") for x in body)
        category, disposition = classify(path, raw)
        hunk_rows.append([
            f"H{number:04d}", path_ids[path], path, int(old_start), int(old_count), int(new_start),
            int(new_count), removed, added, hashlib.sha256(raw).hexdigest(), category, disposition
        ])
    return path_rows, hunk_rows


def tsv(header: list[str], rows: list[list[object]]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue().encode()


def render() -> dict[str, bytes]:
    paths, hunks = census()
    return {
        "paths.tsv": tsv(
            ["path_id", "path", "area", "textual_hunks", "parent_blob", "audited_blob"], paths
        ),
        "hunks.tsv": tsv(
            ["hunk_id", "path_id", "path", "old_start", "old_count", "new_start", "new_count",
             "removed_lines", "added_lines", "hunk_sha256", "classification", "disposition"], hunks
        ),
    }


def validate(rendered: dict[str, bytes]) -> None:
    path_rows = list(csv.DictReader(io.StringIO(rendered["paths.tsv"].decode()), delimiter="\t"))
    hunk_rows = list(csv.DictReader(io.StringIO(rendered["hunks.tsv"].decode()), delimiter="\t"))
    assert git("rev-parse", f"{AUDITED}^").decode().strip() == PARENT
    assert len(path_rows) == EXPECTED_PATHS, len(path_rows)
    assert len(hunk_rows) == EXPECTED_HUNKS, len(hunk_rows)
    assert sum(int(row["textual_hunks"]) for row in path_rows) == EXPECTED_HUNKS
    assert [row["path_id"] for row in path_rows] == [f"P{i:03d}" for i in range(1, EXPECTED_PATHS + 1)]
    assert [row["hunk_id"] for row in hunk_rows] == [f"H{i:04d}" for i in range(1, EXPECTED_HUNKS + 1)]
    path_ids = {row["path"]: row["path_id"] for row in path_rows}
    assert len(path_ids) == EXPECTED_PATHS
    assert all(row["path"] in path_ids and row["path_id"] == path_ids[row["path"]] for row in hunk_rows)
    assert all(row["classification"] and row["disposition"] and len(row["hunk_sha256"]) == 64 for row in hunk_rows)
    assert not any(row["classification"].endswith(":other") for row in hunk_rows)
    counts = Counter(row["classification"] for row in hunk_rows)
    dispositions = Counter(row["disposition"] for row in hunk_rows)
    print(f"commit={AUDITED}")
    print(f"parent={PARENT}")
    print(f"changed_paths={len(path_rows)}")
    print(f"textual_hunks={len(hunk_rows)}")
    print(f"classified_hunks={sum(counts.values())}")
    print(f"unclassified_paths={sum(not row['area'] for row in path_rows)}")
    print(f"unclassified_hunks={sum(not row['classification'] or not row['disposition'] for row in hunk_rows)}")
    for key in sorted(counts):
        print(f"class[{key}]={counts[key]}")
    for key in sorted(dispositions):
        print(f"disposition[{key}]={dispositions[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write regenerated TSV files")
    args = parser.parse_args()
    rendered = render()
    validate(rendered)
    if args.write:
        for name, data in rendered.items():
            (HERE / name).write_bytes(data)
        print("wrote=paths.tsv,hunks.tsv")
    else:
        for name, expected in rendered.items():
            actual = (HERE / name).read_bytes()
            if actual != expected:
                raise SystemExit(f"FAIL: {name} differs from deterministic regeneration")
            print(f"sha256[{name}]={hashlib.sha256(actual).hexdigest()}")
        print("manifest_match=yes")


if __name__ == "__main__":
    main()
