"""Every directory holding an icpx-linked target must disable linker depfiles.

Intel's SYCL link step records transient icpx/ocloc objects (written under
$TMPDIR) in the linker depfile and then deletes them. Ninja sees a dependency
that no longer exists and relinks the target on every build, forever -- the tree
never reaches a no-op build. The remedy is one line,
`set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)`, and it has been in
ggml/src/ggml-sycl/CMakeLists.txt for a long time.

It was there and nowhere else, which is the whole point of this file.
`CMAKE_CXX_LINK_DEPENDS_USE_LINKER` initialises the `LINK_DEPENDS_USE_LINKER`
target property at target-creation time, so it is **directory-scoped**: it
reaches the directory that sets it and the subdirectories that directory adds,
and nothing else. Nine targets across four other directories quietly went
without it. Nothing failed, nothing warned -- builds were simply never
incremental. See llama.cpp-9ubg.

This is a pure source assertion: no build, no GPU, no allocation. It exists so
that directory five fails loudly instead of silently.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Markers that mean "a target created in this directory links with icpx".
SYCL_LINK_MARKERS = (
    "-fsycl",
    "IntelSYCL::SYCL_CXX",
    "ggml_add_backend_library(ggml-sycl",
)

SETTING = "CMAKE_CXX_LINK_DEPENDS_USE_LINKER"

# Directories known to still lack the setting. Each entry is a debt, not an
# exemption -- test_pending_entries_retire_themselves below fails the moment an
# entry is fixed, so the list cannot outlive the problem it records.
#
# Empty is the good state, and it is the state this list retired itself into on
# first use: it held tests/CMakeLists.txt only for as long as that directory's
# four icpx-linked targets went uncovered. Keep the set and this comment. The
# mechanism is what makes a future gap recordable without either blocking the
# fix that finds it or becoming a permanent exemption.
PENDING: set[str] = set()


def _cmakelists() -> list[Path]:
    found = []
    for path in ROOT.rglob("CMakeLists.txt"):
        rel = path.relative_to(ROOT)
        if any(part.startswith("build") or part in ("node_modules", ".git") for part in rel.parts):
            continue
        found.append(path)
    return sorted(found)


def _links_with_icpx(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(marker in text for marker in SYCL_LINK_MARKERS)


def _sets_the_variable(path: Path) -> bool:
    return SETTING in path.read_text(encoding="utf-8", errors="replace")


def _status(path: Path) -> str:
    """One of: covered, uncovered, unreachable.

    The variable is directory-scoped and inherited by subdirectories, so a
    directory is covered when it sets it itself or when some ancestor that
    add_subdirectory()s it does. A directory no ancestor adds is `unreachable`:
    CMake never processes it, it contributes no targets, and it cannot be dirty
    -- flagging it would be a false positive (tools/onednn-f4-probe is such a
    directory today).
    """
    if _sets_the_variable(path):
        return "covered"
    directory = path.parent
    while directory != ROOT:
        child = directory.name
        directory = directory.parent
        parent_cmake = directory / "CMakeLists.txt"
        if not parent_cmake.exists():
            return "unreachable"
        parent_text = parent_cmake.read_text(encoding="utf-8", errors="replace")
        if f"add_subdirectory({child})" not in parent_text:
            return "unreachable"
        if SETTING in parent_text:
            return "covered"
    return "uncovered"


def _uncovered() -> set[str]:
    return {
        str(path.relative_to(ROOT))
        for path in _cmakelists()
        if _links_with_icpx(path) and _status(path) == "uncovered"
    }


def test_at_least_one_directory_links_with_icpx() -> None:
    # Guards against the scan silently matching nothing (a rename of the markers,
    # a moved tree), which would make every assertion below pass vacuously.
    linking = [str(p.relative_to(ROOT)) for p in _cmakelists() if _links_with_icpx(p)]
    assert "ggml/src/ggml-sycl/CMakeLists.txt" in linking
    assert len(linking) >= 5, linking


def test_every_icpx_linking_directory_disables_linker_depfiles() -> None:
    uncovered = _uncovered()
    assert uncovered == PENDING, (
        "directories with an icpx-linked target but no "
        f"{SETTING} in scope: {sorted(uncovered - PENDING)}; "
        f"recorded-but-now-fixed: {sorted(PENDING - uncovered)}. "
        "Add `set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)` to that directory's "
        "CMakeLists.txt (guarded on Ninja + IntelLLVM), or to a parent that "
        "add_subdirectory()s it. See llama.cpp-9ubg."
    )


def test_pending_entries_retire_themselves() -> None:
    # PENDING is a debt list. The moment an entry is fixed this fails, so the
    # entry has to be deleted rather than quietly becoming a permanent exemption.
    still_broken = _uncovered()
    for entry in sorted(PENDING):
        assert (ROOT / entry).exists(), f"PENDING names a file that no longer exists: {entry}"
        assert entry in still_broken, (
            f"{entry} now sets {SETTING} -- remove it from PENDING in this file."
        )
