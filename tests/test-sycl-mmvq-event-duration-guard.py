#!/usr/bin/env python3
"""Source gate: every call to mmvq_sycl_event_duration_us() must be reachable
only when a profiling diagnostic is actually enabled.

mmvq_sycl_event_duration_us() calls sycl::event::get_profiling_info(), which
BLOCKS the calling host thread until the event completes (see the comment on
the helper itself). Every backend queue is created with
sycl::property::queue::enable_profiling, so the call always succeeds and
always blocks -- there is no way to observe "profiling info unavailable" as
an escape hatch on this backend. An unconditional call on the MoE decode hot
path is therefore an unconditional host wait per submit (ruling 6: no host
waits in dispatch). This test scans the source text (no GPU, no build)
rather than measuring the wait, because the wait cannot be observed without
hardware.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MMVQ = ROOT / "ggml" / "src" / "ggml-sycl" / "mmvq.cpp"

CALL = "mmvq_sycl_event_duration_us("
DEFINITION_MARKER = "static double mmvq_sycl_event_duration_us"

# Any one of these appearing on the call's own statement, or on an
# enclosing `if (...)` line within the preceding-scope window below, counts
# as a profile guard. These are exactly the predicates the profiler paths in
# mmvq.cpp already consume kernel_event_us / kernel_profile_us under.
GUARD_TOKENS = (
    "detail_profile",
    "tg_profile",
    "pp_profile",
    "profile_launch",
    "mmvq_moe_tg_profile_enabled()",
    "mmvq_moe_pp_profile_enabled()",
)

# How far back (in lines) to look for an enclosing `if (guard)` before giving
# up. Every real guard in this file sits within a handful of lines of its
# call site; a generous window still catches drift without matching across
# unrelated functions.
ENCLOSING_WINDOW = 60


def _iter_lines() -> list[str]:
    return MMVQ.read_text(encoding="utf-8").split("\n")


def _same_statement_guarded(line: str, call_pos: int) -> bool:
    # The guard for a ternary read (`cond ? mmvq_sycl_event_duration_us(...) : -1.0`)
    # sits earlier on the same line, before the call itself.
    prefix = line[:call_pos]
    return any(tok in prefix for tok in GUARD_TOKENS)


def _enclosing_guarded(lines: list[str], call_idx: int) -> tuple[bool, int | None]:
    """Walk backward from the call line tracking brace balance. A line where
    the running balance goes negative is the nearest line that OPENS a scope
    still active at the call site (an ancestor block) -- skip over any fully
    balanced sibling block along the way. If that opening line is an `if`
    whose condition contains a guard token, the call is guarded; otherwise
    keep climbing to the next ancestor, up to ENCLOSING_WINDOW lines back.
    """
    depth = 0
    steps = 0
    i = call_idx - 1
    while i >= 0 and steps < ENCLOSING_WINDOW:
        line = lines[i]
        depth += line.count("}") - line.count("{")
        if depth < 0:
            if any(tok in line for tok in GUARD_TOKENS):
                return True, i + 1
            depth = 0  # this ancestor wasn't a guard; keep climbing
        i -= 1
        steps += 1
    return False, None


def _call_sites(lines: list[str]) -> list[tuple[int, str, bool, str]]:
    """Returns (1-based line number, stripped source text, guarded, how) for
    every call of mmvq_sycl_event_duration_us(...) other than its own
    definition."""
    sites = []
    for idx, line in enumerate(lines):
        if DEFINITION_MARKER in line:
            continue
        call_pos = line.find(CALL)
        if call_pos == -1:
            continue
        guarded = _same_statement_guarded(line, call_pos)
        how = "same-statement"
        if not guarded:
            guarded, encl = _enclosing_guarded(lines, idx)
            how = f"enclosing if @ L{encl}" if guarded else "UNGUARDED"
        sites.append((idx + 1, line.strip(), guarded, how))
    return sites


def test_helper_definition_documents_that_it_blocks() -> None:
    mmvq = MMVQ.read_text(encoding="utf-8")
    assert DEFINITION_MARKER in mmvq
    def_idx = mmvq.index(DEFINITION_MARKER)
    # A comment must precede the definition (within a small window) stating
    # that the helper blocks and must only be called under a profile guard.
    window = mmvq[max(0, def_idx - 800):def_idx]
    assert "block" in window.lower(), (
        "mmvq_sycl_event_duration_us needs a comment stating it BLOCKS the host "
        "(get_profiling_info waits for event completion)"
    )
    assert "profile" in window.lower() and "guard" in window.lower(), (
        "mmvq_sycl_event_duration_us needs a comment stating it must only be "
        "called under a profile guard"
    )


def test_call_site_census_is_nonempty_and_stable() -> None:
    # A positive control: if this drops to zero, the scan below would pass
    # vacuously. mmvq.cpp is known (2026-09) to carry 8 non-definition call
    # sites of mmvq_sycl_event_duration_us; require at least that many so a
    # broken scan (or a mass deletion) cannot silently pass.
    lines = _iter_lines()
    sites = _call_sites(lines)
    assert len(sites) >= 8, (
        f"expected >= 8 mmvq_sycl_event_duration_us call sites, found {len(sites)}; "
        "either the scan regressed or call sites were removed -- update this test "
        "deliberately if the latter"
    )


def test_every_mmvq_event_duration_call_is_profile_guarded() -> None:
    lines = _iter_lines()
    sites = _call_sites(lines)
    unguarded = [(n, text, how) for n, text, guarded, how in sites if not guarded]
    assert not unguarded, "unguarded mmvq_sycl_event_duration_us() call(s) -- " + (
        "each must only run when a profiling diagnostic (detail_profile / "
        "tg_profile / pp_profile / profile_launch / "
        "mmvq_moe_tg_profile_enabled() / mmvq_moe_pp_profile_enabled()) is "
        "actually enabled, because get_profiling_info() BLOCKS the host "
        "until the event completes:\n"
    ) + "\n".join(f"  L{n}: {text}" for n, text, _how in unguarded)


if __name__ == "__main__":
    import sys

    try:
        import pytest
    except ImportError:
        pytest = None

    if pytest is not None:
        sys.exit(pytest.main(["-q", __file__]))

    # Minimal fallback runner so a direct `python3` invocation is still
    # self-checking on a machine without pytest installed.
    failures = []
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
            except AssertionError as exc:
                failures.append((_name, str(exc)))
    for _name, _msg in failures:
        print(f"FAILED {_name}\n{_msg}\n")
    print(f"{len(failures)} failed" if failures else "OK")
    sys.exit(1 if failures else 0)
