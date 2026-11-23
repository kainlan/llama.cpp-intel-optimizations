#!/usr/bin/env python3
"""Source contract for the watchdog of the private production-route device tests.

Each listed test enters a production route cold, and cold B70 AOT/submit can outlast the backend's 30 s default op
watchdog. So each registration sets GGML_SYCL_OP_TIMEOUT_MS explicitly, 10 s or more below its own CTest timeout so a
real hang is still diagnosed by the backend, and nothing else in the file sets it.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMAKE = ROOT / "ggml/src/ggml-sycl/CMakeLists.txt"
# The registered production-route device tests that set the watchdog. Membership is by name, not by label: a new
# user is added here on purpose, never by acquiring a label.
TEST_NAMES = ("sycl-q1-nvfp4-admitted-device", "sycl-moe-bf16-direct-device")
WATCHDOG_ENV = "GGML_SYCL_OP_TIMEOUT_MS"
EXPECTED_WATCHDOG_MS = 110_000
MIN_CTEST_MARGIN_MS = 10_000
ROUTE_LABEL = "production-route"


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


def registration_properties(code: str, name: str) -> str:
    # `code` has its comments removed, and the body then ends at the first ")" that is outside a quoted argument: a
    # registration's own comments contain parentheses (the "(level_zero:1 selects the B50 for a second run)" in the
    # q1 comment would otherwise cut the body off before ENVIRONMENT and TIMEOUT), and so could a trailing comment or
    # a quoted value.
    match = re.search(rf"set_tests_properties\(\s*{re.escape(name)}\s+PROPERTIES", code)
    assert match, f"{name}: missing its set_tests_properties(... PROPERTIES"
    in_quote = False
    for i in range(match.end(), len(code)):
        ch = code[i]
        if ch == '"' and code[i - 1] != "\\":
            in_quote = not in_quote
        elif ch == ")" and not in_quote:
            return code[match.end() : i]
    raise AssertionError(f"{name}: unterminated set_tests_properties")


@pytest.mark.parametrize("name", TEST_NAMES)
def test_private_route_watchdog_is_explicit_and_bounded(name: str) -> None:
    properties = registration_properties(strip_cmake_comments(CMAKE.read_text()), name)

    # ENVIRONMENT is one quoted ";"-list: each registration pins ONEAPI_DEVICE_SELECTOR beside the watchdog (commit
    # 414856e27 for q1), so the watchdog is one entry of that list, not the whole string.
    environment = re.search(r'ENVIRONMENT\s+"([^"]*)"', properties)
    assert environment, f"{name}: a private production-route test must set an ENVIRONMENT"
    watchdog_entries = [
        re.fullmatch(rf"{WATCHDOG_ENV}=(\d+)", entry)
        for entry in environment.group(1).split(";")
        if entry.startswith(f"{WATCHDOG_ENV}=")
    ]
    assert len(watchdog_entries) == 1 and watchdog_entries[0], (
        f"{name}: must set the backend watchdog as exactly one well-formed {WATCHDOG_ENV}=<digits> entry of its "
        f"';'-separated ENVIRONMENT; the ENVIRONMENT value was {environment.group(1)!r}"
    )
    outer = re.search(r"\bTIMEOUT\s+(\d+)", properties)
    assert outer, f"{name}: a private production-route test must keep an outer CTest TIMEOUT"

    watchdog_ms = int(watchdog_entries[0].group(1))
    outer_ms = int(outer.group(1)) * 1000
    assert watchdog_ms == EXPECTED_WATCHDOG_MS, f"{name}: watchdog {watchdog_ms} ms, want {EXPECTED_WATCHDOG_MS}"
    assert watchdog_ms > 30_000, f"{name}: cold AOT must not inherit the 30 s failure mode"
    assert outer_ms - watchdog_ms >= MIN_CTEST_MARGIN_MS, (
        f"{name}: CTest TIMEOUT {outer_ms} ms leaves {outer_ms - watchdog_ms} ms above the {watchdog_ms} ms watchdog, "
        f"want >= {MIN_CTEST_MARGIN_MS}"
    )

    # A consequence of being on the list, not the rule for getting on it: every listed test is a production-route
    # device test.
    labels = re.search(r'LABELS\s+"([^"]*)"', properties)
    assert labels and ROUTE_LABEL in labels.group(1).split(";"), (
        f"{name}: a listed registration must carry the {ROUTE_LABEL!r} label; its LABELS were "
        f"{labels.group(1) if labels else None!r}"
    )


def test_watchdog_is_set_only_by_the_listed_registrations() -> None:
    # Production defaults and unrelated tests stay untouched: the listed registrations are the only places the file
    # sets the watchdog, and a new user is added to TEST_NAMES on purpose. Counted over code with the CMake comments
    # removed -- 4432f195d (llama.cpp-u1pn) explains the AOT choice in a comment that names the variable, and that
    # prose is not a setting.
    code = strip_cmake_comments(CMAKE.read_text())
    count = code.count(WATCHDOG_ENV)
    assert count == len(TEST_NAMES), (
        f"{WATCHDOG_ENV} appears {count} times in {CMAKE.name} outside comments, want {len(TEST_NAMES)}: one per "
        f"listed registration ({', '.join(TEST_NAMES)})"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
