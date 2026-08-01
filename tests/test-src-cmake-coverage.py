#!/usr/bin/env python3
"""Every tracked src/*.cpp must appear in src/CMakeLists.txt.

A source that exists but is not in the build is compiled into nothing: its
symbols vanish, and every test of it fails at LINK time with an error that
names the test, not the missing module. That misdirection cost a full triage
cycle (llama.cpp-4hvq blamed test rot; the module was simply not built).

History: an upstream merge (`ff333a570`, "Merge remote-tracking branch
'upstream/master' into feature/sycl-coalescing") pulled in upstream's
`089dd41fe` ("cmake: use glob to collect src/models sources", #22005), which
replaced the explicit `models/*.cpp` list with `file(GLOB ...)`. Conflict
resolution on `src/CMakeLists.txt` took the incoming block wholesale and, with
it, silently dropped four fork-local top-level entries that the glob does not
cover (the glob only reaches `models/*.cpp`): `llama-kv-block.cpp` (added by
`eff644999`), `llama-tensor-class.cpp` (added by `1bb021aed`), and
`llama-pp-scheduler.cpp` (added by `7c0a45544`). `llama-moe-profile.cpp`
(added by `71075d62e`) was never wired in at all -- that commit added the
`.cpp`/`.h` pair without touching `src/CMakeLists.txt` in the first place, so
it has no "removal" commit to blame, only a missing addition.
"""
from __future__ import annotations

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKE = ROOT / "src" / "CMakeLists.txt"

# Sources deliberately excluded from the library build. Each entry MUST carry a
# reason; an empty reason is not an exemption.
EXCLUDED: dict[str, str] = {}


def tracked_src_sources() -> list[str]:
    # Top-level src/*.cpp only. src/models/*.cpp is deliberately excluded here:
    # since 089dd41fe ("cmake: use glob to collect src/models sources",
    # #22005), src/CMakeLists.txt pulls that whole directory in via
    # `file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")`, so a models/*.cpp file
    # is covered without ever appearing by name in the CMakeLists text --
    # asserting its name were present would be checking the wrong thing.
    # test_models_glob_is_present below guards the glob itself instead.
    out = subprocess.run(
        ["git", "ls-files", "src/*.cpp"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [
        pathlib.Path(p).name for p in out
        if pathlib.Path(p).parent == pathlib.Path("src")
    ]


def test_every_tracked_src_cpp_is_in_the_build() -> None:
    cmake = CMAKE.read_text(encoding="utf-8")
    missing = [
        name for name in tracked_src_sources()
        if name not in cmake and name not in EXCLUDED
    ]
    assert not missing, (
        f"tracked src/*.cpp absent from src/CMakeLists.txt: {sorted(missing)}. "
        "Either add them to the build or delete them; a source that exists but "
        "is not compiled fails at link time with a misleading error."
    )


def test_models_glob_is_present() -> None:
    # Guards the mechanism test_every_tracked_src_cpp_is_in_the_build relies on
    # to exempt src/models/*.cpp: if this glob is ever removed (e.g. reverted
    # back to an explicit list, or dropped outright), every file under
    # src/models/ silently stops being exempted from nothing -- it stops being
    # BUILT -- while this suite's other test would keep passing, since it never
    # looks at models/*.cpp names at all.
    cmake = CMAKE.read_text(encoding="utf-8")
    assert 'file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")' in cmake, (
        "src/CMakeLists.txt no longer globs models/*.cpp -- files under "
        "src/models/ may no longer be compiled into the library."
    )
    assert "${LLAMA_MODELS_SOURCES}" in cmake, (
        "LLAMA_MODELS_SOURCES is no longer referenced in add_library(llama ...)"
    )


def test_exclusions_carry_a_reason() -> None:
    blank = [k for k, v in EXCLUDED.items() if not v.strip()]
    assert not blank, f"EXCLUDED entries without a reason: {blank}"
