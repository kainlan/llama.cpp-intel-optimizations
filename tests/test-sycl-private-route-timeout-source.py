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


def strip_cmake_comments(cmake: str) -> str:
    """Drop every `#` comment, whole-line or trailing, keeping quoted arguments (a `#` or `)` inside one is text)."""
    out = []
    in_quote = False
    i = 0
    while i < len(cmake):
        ch = cmake[i]
        if in_quote:
            out.append(ch)
            if ch == "\\" and i + 1 < len(cmake):
                out.append(cmake[i + 1])
                i += 1
            elif ch == '"':
                in_quote = False
        elif ch == '"':
            in_quote = True
            out.append(ch)
        elif ch == "#":
            while i < len(cmake) and cmake[i] != "\n":
                i += 1
            continue  # keep the newline itself
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def private_route_properties(cmake: str) -> str:
    # Comments are removed first, and the body then ends at the first ")" that is outside a quoted argument: the
    # registration's own comments contain parentheses (the "(level_zero:1 selects the B50 for a second run)" in its
    # comment would otherwise cut the body off before ENVIRONMENT and TIMEOUT), and so could a trailing comment or a
    # quoted value.
    code = strip_cmake_comments(cmake)
    match = re.search(rf"set_tests_properties\(\s*{re.escape(TEST_NAME)}\s+PROPERTIES", code)
    assert match, f"missing private production-route properties for {TEST_NAME}"
    in_quote = False
    for i in range(match.end(), len(code)):
        ch = code[i]
        if ch == '"' and code[i - 1] != "\\":
            in_quote = not in_quote
        elif ch == ")" and not in_quote:
            return code[match.end() : i]
    raise AssertionError(f"unterminated set_tests_properties for {TEST_NAME}")


def test_private_route_watchdog_is_explicit_bounded_and_isolated() -> None:
    cmake = CMAKE.read_text()
    properties = private_route_properties(cmake)

    # ENVIRONMENT is one quoted ";"-list: the registration deliberately pins ONEAPI_DEVICE_SELECTOR beside the
    # watchdog (commit 414856e27), so the watchdog is one entry of that list, not the whole string.
    environment = re.search(r'ENVIRONMENT\s+"([^"]*)"', properties)
    assert environment, "private route must set an ENVIRONMENT"
    watchdog_entries = [
        re.fullmatch(rf"{WATCHDOG_ENV}=(\d+)", entry)
        for entry in environment.group(1).split(";")
        if entry.startswith(f"{WATCHDOG_ENV}=")
    ]
    assert len(watchdog_entries) == 1 and watchdog_entries[0], (
        f"private route must explicitly set the backend watchdog as exactly one well-formed {WATCHDOG_ENV}=<digits> "
        f"entry of its ';'-separated ENVIRONMENT; the ENVIRONMENT value was {environment.group(1)!r}"
    )
    outer = re.search(r"TIMEOUT\s+(\d+)", properties)
    assert outer, "private route must retain an outer CTest timeout"

    watchdog_ms = int(watchdog_entries[0].group(1))
    outer_ms = int(outer.group(1)) * 1000
    assert watchdog_ms == EXPECTED_WATCHDOG_MS
    assert watchdog_ms > 30_000, "cold AOT must not inherit the 30 s failure mode"
    assert outer_ms - watchdog_ms >= MIN_CTEST_MARGIN_MS

    # Production defaults and unrelated tests must remain untouched: this is the only place the file SETS the
    # watchdog. Counted over code with the CMake comments removed -- 4432f195d (llama.cpp-u1pn) explains the
    # AOT choice in a comment that names the variable, and that prose is not a setting.
    code = strip_cmake_comments(cmake)
    assert code.count(WATCHDOG_ENV) == 1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
