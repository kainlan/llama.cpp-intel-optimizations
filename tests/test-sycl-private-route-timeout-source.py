#!/usr/bin/env python3
"""Source contract for the private Q1/NVFP4 production-route watchdog."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "ggml/src/ggml-sycl/CMakeLists.txt"
TEST_NAME = "sycl-q1-nvfp4-admitted-device"
WATCHDOG_ENV = "GGML_SYCL_OP_TIMEOUT_MS"
EXPECTED_WATCHDOG_MS = 110_000
MIN_CTEST_MARGIN_MS = 10_000


def private_route_properties(cmake: str) -> str:
    match = re.search(
        rf"set_tests_properties\(\s*{re.escape(TEST_NAME)}\s+PROPERTIES(?P<body>.*?)\)",
        cmake,
        re.DOTALL,
    )
    assert match, f"missing private production-route properties for {TEST_NAME}"
    return match.group("body")


def test_private_route_watchdog_is_explicit_bounded_and_isolated() -> None:
    cmake = CMAKE.read_text()
    properties = private_route_properties(cmake)

    watchdog = re.search(rf'ENVIRONMENT\s+"{WATCHDOG_ENV}=(\d+)"', properties)
    outer = re.search(r"TIMEOUT\s+(\d+)", properties)
    assert watchdog, "private route must explicitly set the backend watchdog"
    assert outer, "private route must retain an outer CTest timeout"

    watchdog_ms = int(watchdog.group(1))
    outer_ms = int(outer.group(1)) * 1000
    assert watchdog_ms == EXPECTED_WATCHDOG_MS
    assert watchdog_ms > 30_000, "cold AOT must not inherit the 30 s failure mode"
    assert outer_ms - watchdog_ms >= MIN_CTEST_MARGIN_MS

    # Production defaults and unrelated tests must remain untouched.
    assert cmake.count(WATCHDOG_ENV) == 1
