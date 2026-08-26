#!/usr/bin/env python3
"""Deterministic census of separately counted backend registry getters."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm"}
CALL_RE = re.compile(r"ggml_backend_(?:dev|reg)_get\s*\([^)]*\)")
DEFINITION_RE = re.compile(
    r"ggml_backend_(?:dev|reg)_t\s+ggml_backend_(?:dev|reg)_get"
)
EXPECTED_BY_FILE = {
    "common/arg.cpp": 2,
    "common/common.cpp": 1,
    "examples/llama.android/lib/src/main/cpp/ai_chat.cpp": 1,
    "src/llama-context.cpp": 1,
    "src/llama-model.cpp": 3,
    "src/llama.cpp": 3,
    "tests/test-backend-ops.cpp": 1,
    "tests/test-gguf.cpp": 1,
    "tests/test-llama-archs.cpp": 1,
    "tests/test-opt.cpp": 1,
    "tests/test-planner-canary-cpy-visibility.cpp": 2,
    "tests/test-sycl-lifecycle-gpu-sequential.cpp": 1,
    "tests/test-sycl-lifecycle-runtime-wrapper.cpp": 8,
    "tests/test-thread-safety.cpp": 1,
    "tools/llama-bench/llama-bench.cpp": 4,
    "tools/rpc/rpc-server.cpp": 2,
    "tools/server/server-context.cpp": 1,
    "tools/tuning/main.cpp": 1,
}
COMPARISON_ONLY = {
    "tests/test-sycl-lifecycle-runtime-wrapper.cpp": 8,
    "tools/server/server-context.cpp": 1,
}
# Upstream b10630 removed the server spec-fit tgt_devices block (draft-model
# memory measurement moved into common_speculative_init_from_params), so the
# index-aligned placeholder class is retired. The literal is kept as a tripwire:
# if the pattern reappears anywhere it must be re-classified deliberately.
PLACEHOLDER = "tgt_devices.push_back(ggml_backend_dev_get(i));"


def source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in {".git", "build", "vendor"} for part in path.parts):
            continue
        yield path


def fail(message: str) -> None:
    print(f"backend count/get census failed: {message}", file=sys.stderr)
    raise SystemExit(1)


calls = []
texts = {}
for path in source_files():
    text = path.read_text(errors="replace")
    relative = path.relative_to(ROOT).as_posix()
    texts[relative] = text
    for match in CALL_RE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end]
        if "GGML_API" in line or DEFINITION_RE.search(line):
            continue
        calls.append((relative, text, match, line))

actual_by_file = Counter(relative for relative, _, _, _ in calls)
if dict(sorted(actual_by_file.items())) != EXPECTED_BY_FILE:
    fail(f"call-site inventory drifted: {dict(sorted(actual_by_file.items()))}")

classified = Counter()
unsafe = []
for relative, text, match, line in calls:
    invocation = match.group(0)
    if PLACEHOLDER in line:
        fail(f"retired index placeholder reappeared in {relative}")

    # Calls used only as comparison operands are nullable by construction.
    if re.search(re.escape(invocation) + r"\s*(?:==|!=)", line):
        classified["comparison-only"] += 1
        continue

    line_start = text.rfind("\n", 0, match.start()) + 1
    assignment = re.search(r"([A-Za-z_]\w*)\s*=\s*$", text[line_start:match.start()])
    if assignment:
        variable = assignment.group(1)
        after = text[match.end():match.end() + 300]
        null_witness = re.search(
            rf"(?:if\s*\(\s*!\s*{variable}\b|"
            rf"if\s*\(\s*{variable}\s*(?:&&|\?)|"
            rf"\b{variable}\s*\?)",
            after,
        )
        first_use = re.search(rf"\b{variable}\b", after)
        if null_witness and first_use and null_witness.start() <= first_use.start():
            classified["null-checked-consumer"] += 1
            continue

    line_number = text.count("\n", 0, match.start()) + 1
    unsafe.append(f"{relative}:{line_number}: {line.strip()}")

if unsafe:
    fail("unsafe consumers remain:\n" + "\n".join(unsafe))

expected_classes = Counter({
    "null-checked-consumer": 26,
    "comparison-only": 9,
})
if classified != expected_classes:
    fail(f"classification drifted: {dict(classified)}")
if sum(classified.values()) != 35:
    fail(f"expected 35 classified calls, found {sum(classified.values())}")
if Counter({path: count for path, count in COMPARISON_ONLY.items()}) != Counter(
    relative
    for relative, text, match, line in calls
    if PLACEHOLDER not in line and re.search(re.escape(match.group(0)) + r"\s*(?:==|!=)", line)
):
    fail("comparison-only allowlist drifted")

print(
    "backend count/get census: PASS "
    "(35/35 classified: 26 null-checked, 9 comparison-only; unsafe 0)"
)
