#!/usr/bin/env python3
"""Compile the public lifecycle ABI contract instead of matching source text."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
compiler = os.environ.get("CXX", "c++")
with tempfile.TemporaryDirectory(prefix="sycl-lifecycle-api-") as tmp:
    output = Path(tmp) / "public-api"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{root / 'ggml/include'}",
            str(root / "tests/test-sycl-lifecycle-public-api.cpp"),
            "-o",
            str(output),
        ],
        check=True,
    )
    subprocess.run([str(output)], check=True)
print("public lifecycle ABI executable contract: PASS")
