# Wave-1 conflict brief: build system

Pre-merge analysis of every both-touched build-system file, so wave 1 (T15) can
resolve them mechanically. Base for both diffs is `81ff7abe5` (fork/upstream
merge-base); fork side is `master`, upstream side is tag `b10630`.

## Derivation

```bash
comm -12 <(git diff --name-only 81ff7abe5 master | LC_ALL=C sort) \
         <(git diff --name-only 81ff7abe5 b10630 | LC_ALL=C sort) \
  | grep -E 'CMakeLists\.txt|\.cmake$|^scripts/|^\.github/'
```

Raw output (12 files):

```
.github/workflows/build-sycl.yml
.github/workflows/build-wasm.yml
.github/workflows/build-webgpu.yml
CMakeLists.txt
ggml/CMakeLists.txt
ggml/src/CMakeLists.txt
ggml/src/ggml-sycl/CMakeLists.txt
scripts/ui-assets.cmake
src/CMakeLists.txt
tests/CMakeLists.txt
tools/CMakeLists.txt
tools/ui/CMakeLists.txt
```

12 entries below, one per file.

**Conflict shapes verified against `git merge-tree --write-tree master b10630`**
(read-only — simulates the exact merge T15 will do, without touching the
working tree). Of these 12 files, only **four** actually produce a merge
conflict in git's own attempt: `ggml/src/ggml-sycl/CMakeLists.txt`,
`tests/CMakeLists.txt`, `.github/workflows/build-wasm.yml`, and
`.github/workflows/build-webgpu.yml`. The other eight — `CMakeLists.txt`,
`ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `scripts/ui-assets.cmake`,
`src/CMakeLists.txt`, `tests/CMakeLists.txt`'s sibling `tools/CMakeLists.txt`,
`tools/ui/CMakeLists.txt`, and `.github/workflows/build-sycl.yml` — auto-merge
cleanly despite adjacent or nearby hunks; their entries below describe the
apparent adjacency for context and end in a sanity-check, not a conflict
resolution. Confirm the four-file list yourself before trusting any of this:

```bash
git merge-tree --write-tree master b10630 > /tmp/mt.txt
grep CONFLICT /tmp/mt.txt
```

---

## `CMakeLists.txt` (root)

**Fork intent:** hoists `include(CTest)` to run *before* `add_subdirectory(ggml)`
(new block guarded by `LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT
CMAKE_JS_VERSION`, right after the `LLAMA_USE_SYSTEM_GGML` alias block), and
removes the old inline `include(CTest)` that used to sit immediately before
`add_subdirectory(tests)`. Reason given: backend `add_test()` calls need CTest
enabled before ggml's subdirectory is processed, or they're invisible to a
fresh top-level `ctest -N`.

**Upstream intent:** three independent changes in the same file: (1) adds a
`LLAMA_VERSION_MAJOR/MINOR/PATCH` + `LLAMA_BUILD_IS_DEV` version scheme,
replacing the old `LLAMA_INSTALL_VERSION = 0.0.${LLAMA_BUILD_NUMBER})` line and
renaming the generated `llama-version.cmake` to `llama-config-version.cmake`;
(2) adds `LLAMA_SUBPROCESS` option + `LLAMA_SUBPROCESS_DEFAULT` platform gate
(OFF for iOS/Android/Emscripten); (3) restructures vendoring —
`add_subdirectory(vendor)` unconditionally, dropping the old
`add_subdirectory(vendor/cpp-httplib)` call that lived inside the
`LLAMA_BUILD_COMMON` guard.

**Interaction:** no direct line overlap — fork's hunk touches the
`LLAMA_BUILD_COMMON`/`LLAMA_BUILD_TESTS` blocks (old lines ~182–213); upstream's
hunks touch the top of the file (~2–101) and the `add_subdirectory(vendor)`
region (~199–224) plus the `write_basic_package_version_file` block (~266–292).
Both a fork-only `include(CTest)` hoist and upstream's `LLAMA_VERSION`/vendor
changes must land.

**RESOLVE:** interleave — take fork's early `include(CTest)` hoist and removal
of the late one, plus upstream's version-scheme rewrite (`LLAMA_VERSION`,
`llama-config-version.cmake`) and vendor restructuring, verbatim. Post-merge,
grep for `include(CTest)` and confirm exactly one occurrence (the hoisted one)
and that `LLAMA_INSTALL_VERSION` has no remaining reader (see `src/CMakeLists.txt`
below, which also references it).

---

## `ggml/CMakeLists.txt`

**Fork intent:** adds `GGML_SYCL_OUT_OF_ORDER` option; and restructures the
tests-enablement tail — replaces `if (GGML_BUILD_TESTS) enable_testing()
add_subdirectory(tests) endif()` with an early, broader `enable_testing()` gate
(`if (BUILD_TESTING OR GGML_BUILD_TESTS OR GGML_SYCL_BUILD_XMX_TESTS)`) placed
*before* `add_subdirectory(src)`, plus a narrower late guard
(`GGML_BUILD_TESTS AND EXISTS ".../tests"`) around `add_subdirectory(tests)`
alone (no `enable_testing()` there anymore).

**Upstream intent:** version bump (`GGML_VERSION_MINOR` 15→22, `PATCH` 3→0);
drops the dead `GGML_HIP_ROCWMMA_FATTN` option; adds `GGML_OPENMP_FETCH`,
`GGML_ET`, `GGML_ET_SYSEMU` options; removes a commented-out dead
`GGML_METAL`/`RESOURCE` block; renames `ggml-version.cmake` →
`ggml-config-version.cmake` (mirrors the root file's rename).

**Interaction:** disjoint line ranges (fork: ~251, ~297–303; upstream: ~4–8,
~216, ~243–257, ~340–342, ~400–413). No textual overlap.

**RESOLVE:** interleave — apply both sides independently; no hunk-level
conflict expected. Verify after merge that `enable_testing()` appears exactly
once (the fork's early, broadened gate) and that `add_subdirectory(tests)` uses
the fork's `EXISTS` guard.

---

## `ggml/src/CMakeLists.txt`

**Fork intent:** (1) adds an early `enable_testing()` under
`GGML_SYCL_BUILD_XMX_TESTS` so SYCL unit tests are discoverable without
`ggml/tests`; (2) adds a `ggml-private-test-registry` STATIC library (gated
`BUILD_TESTING AND NOT GGML_BACKEND_DL`) that embeds `ggml-backend-dl.cpp` +
`ggml-backend-reg.cpp` with `GGML_USE_CPU`/`GGML_USE_SYCL` compile defs, so
full-backend test fixtures get their own static registry closure instead of
double-registering backend globals via libggml's DSO; (3) inside
`ggml_add_backend_library()`, adds a `RUNTIME DESTINATION` clause alongside
each existing `LIBRARY DESTINATION` install() call (both the `GGML_BACKEND_DIR`
and `CMAKE_INSTALL_BINDIR` branches).

**Upstream intent:** replaces the `if (GGML_OPENMP) find_package(OpenMP) ...`
block with a much larger `if (GGML_OPENMP_FETCH) ... elseif (GGML_OPENMP)
...  endif()` — a from-scratch LLVM OpenMP fetch/extract path for Windows/Clang
(downloads/verifies/extracts `libomp` via 7-Zip, sets up `ggml-openmp-c`/
`ggml-openmp-cxx` INTERFACE targets) that funnels into new
`GGML_OPENMP_TARGET_C`/`GGML_OPENMP_TARGET_CXX` variables consumed a few lines
later where `target_link_libraries(ggml-base PRIVATE OpenMP::OpenMP_C
OpenMP::OpenMP_CXX)` becomes `target_link_libraries(ggml-base PRIVATE
${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})`. Separately: PowerPC/AIX
support in `GGML_CPU_ALL_VARIANTS` (`Linux` → `Linux|AIX`), and adds
`ggml_add_backend(ET)` to the backend list.

**Interaction:** disjoint — fork's hunks are at old lines 3–11 and 262–299;
upstream's are at old lines 222–236, 430, 473. No overlap.

**RESOLVE:** interleave — apply both independently. No textual conflict
expected; verify with a configure (not a full build, per this task's no-build
rule — leave to T15/wave-1 execution) that `ggml-private-test-registry` still
builds against whichever `OpenMP` target variables end up defined (it does not
itself touch OpenMP, so this is a low-risk sequencing check only).

---

## `ggml/src/ggml-sycl/CMakeLists.txt`

**This is the GLOB trap file** — by far the largest and highest-risk entry.
Upstream's whole diff against base is **11 lines added, 0 removed**
(`git diff --numstat 81ff7abe5..b10630 -- ggml/src/ggml-sycl/CMakeLists.txt`),
confined to the `GGML_SYCL_DEVICE_ARCH` AOT block; the fork's diff is ~4,900 lines and
registers the bulk of the fork's SYCL test suite directly in this file, far
beyond upstream's two source `file(GLOB ...)` calls (base file, unmodified by
either side elsewhere: `GGML_SOURCES_SYCL` glob at base line 26 and the
`fattn-tile*`/`fattn-vec*` template-instance globs at base lines 27–30 — the
fork's hunk rewrites this exact region, see below).

**Fork intent (selected highlights, not exhaustive — see file for full text):**
- Rewrites the `GGML_SYCL_TARGET` validation at the top (was a hard
  `FATAL_ERROR` if unset/invalid; fork instead defaults it to `"INTEL"` if
  undefined, then prints the status line).
- Adds a directory-scoped `CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE` workaround
  for icpx/ocloc depfile churn under Ninja (llama.cpp-9ubg); sibling copies live
  in `tools/CMakeLists.txt` and `tests/CMakeLists.txt` (see below), and
  `test-sycl-link-depends-coverage.py` gates a 5th directory never needing one.
- Adds `kv-tier-manager.cpp` to the `ggml_add_backend_library(ggml-sycl ...)`
  source list.
- Guards `target_link_libraries(ggml-sycl PRIVATE ggml-cpu)` under `NOT
  GGML_BACKEND_DL`.
- Adds options: `GGML_SYCL_MMQ_XMX`, `GGML_SYCL_XMX_GEMM` (with a
  `FATAL_ERROR` cross-check that XMX_GEMM requires MMQ_XMX — mmq_xmx.hpp is
  only included under the latter), `GGML_SYCL_DEBUG_SOA_LAYOUT`,
  `GGML_SYCL_QUANTIZER_DEBUG`, `GGML_SYCL_ASAN_PARTIAL`,
  `GGML_SYCL_PROFILING_DEBUG`.
- Rewrites the source glob: `file(GLOB GGML_SOURCES_SYCL "*.cpp")` followed by
  `list(FILTER ... EXCLUDE REGEX "test_joint_matrix.*\\.cpp$")` and
  `"test_tensor_parallel\\.cpp$"` (upstream's separate `fattn-tile*`/
  `fattn-vec*` template-instance globs are gone from the fork's version of this
  region — confirm at merge time whether the fork already folds those into the
  main `*.cpp` glob or has moved them elsewhere; do not silently drop them).
- Adds `option(GGML_SYCL_BUILD_XMX_TESTS "Build XMX ESIMD unit tests" OFF)` and,
  under `if (GGML_SYCL_BUILD_XMX_TESTS AND GGML_SYCL_TARGET STREQUAL "INTEL"
  AND NOT GGML_BACKEND_DL)`, registers a large block of `add_executable(...)`
  test targets. The condition is deliberately extended (not nested) so a
  `GGML_BACKEND_DL=ON` configure (the nightly Intel Docker image) falls to the
  `else()` branch, which registers `DISABLED (requires ...)` ctest
  placeholders instead of silently omitting the tests (the "visible-absence"
  property, llama.cpp-u2mz).
- **123 `add_executable(...)` registrations total** in this file (verified:
  `git diff 81ff7abe5..master -- ggml/src/ggml-sycl/CMakeLists.txt | grep -c
  '^+.*add_executable('`), spanning both the `GGML_SYCL_BUILD_XMX_TESTS`-gated
  block and a large tail of targets registered unconditionally (outside that
  guard, so they always build) — e.g. `test-sycl-dispatch-tuning`,
  `test-kv-slice-sizing`, `test-mem-handle-*`, `test-unified-cache-*`,
  `test-sycl-moe-*`, `test-tensor-classification`, `test-cold-start`,
  `test-dmmv-*`, `test-mmq-*`, and dozens more (grep the file for
  `^add_executable(` and `^    add_executable(` to enumerate the full,
  current list — it will have drifted further by merge time).

**Upstream intent:** inside the existing `if (GGML_SYCL_DEVICE_ARCH) ...
endif()` AOT block (base lines ~195–213, `spir64_gen` target flags), adds
`include(ProcessorCount)` + `ProcessorCount(_ggml_sycl_nproc)` and a new cache
variable `GGML_SYCL_MAX_PARALLEL_LINK_JOBS`, then appends
`-fsycl-max-parallel-link-jobs=${GGML_SYCL_MAX_PARALLEL_LINK_JOBS}` to the
existing `target_link_options(...)` call — parallelizes the `llvm-foreach --
ocloc` AOT device-image lowering step. Nothing else in this file changes
upstream.

**⚠️ Real conflict, not just adjacency:** the fork's master has *already
rewritten* this exact `if (GGML_SYCL_DEVICE_ARCH)` block to a different form —
current `master` (line 836 in the live file) reads:
```
if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
    target_link_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
endif()
```
i.e. the fork dropped the `spir64_gen`/AOT multi-line `target_compile_options`
+ `target_link_options` form entirely in favor of a single-line
`--offload-arch=` flag on each. Upstream's `GGML_SYCL_MAX_PARALLEL_LINK_JOBS`
patch is written against the *old* `spir64_gen` form and cannot be applied
verbatim — the fork's version has neither the `-fsycl-targets=spir64_gen` line
nor a bare `target_link_options` block for it to extend. This needs a manual
decision, not a mechanical carry-forward: either (a) port the parallel-link-jobs
cache variable onto the fork's `--offload-arch=` line (append
`-fsycl-max-parallel-link-jobs=${GGML_SYCL_MAX_PARALLEL_LINK_JOBS}` there
instead), or (b) restore upstream's `spir64_gen` form and re-verify the fork's
`--offload-arch=` change was not load-bearing for something else. Do not
default to "ours" here without confirming which AOT flag form the fork
actually needs on hardware — this is a functional compiler-flag choice, not a
cosmetic one.

**⚠️ The conflict git's own merge actually reports is at the WRONG location —
do not trust its line numbers.** Ran for real (`git merge-file -L ours -L base
-L theirs` against the three blobs `git merge-tree` names for this path):
git's xdiff-based 3-way merge places the conflict markers around
`add_executable(test-xmx-config ...)`'s `target_include_directories(...)`
block (~line 1197 of the merged text), splicing upstream's
`GGML_SYCL_MAX_PARALLEL_LINK_JOBS` patch in there instead of anywhere near
`GGML_SYCL_DEVICE_ARCH`. Worse: the *real* semantic site — master's lines
836–838, the `--offload-arch=` block quoted above — passes through this merge
completely untouched, with **no conflict marker at all**, silently keeping
master's form and silently dropping upstream's `-fsycl-max-parallel-link-jobs`
feature. The file's ~4,900-line rewrite is large enough that git's LCS-based
diff loses track of which block corresponds to which; the merge tool's output
for this file cannot be used as a checklist. Wave 1 must (1) not assume the
absence of a conflict marker at `GGML_SYCL_DEVICE_ARCH` means it's resolved —
it isn't, it's silently wrong — and (2) not act on the spurious marker it
does produce near `test-xmx-config` beyond taking "ours" there (upstream's
content landed in the wrong place and carries no meaning at that location).

**RESOLVE:** interleave, with a flagged manual decision — carry forward all
123 fork test registrations and the option/glob/link-workaround additions
verbatim (upstream touches none of that region); for the `GGML_SYCL_DEVICE_ARCH`
block specifically, do NOT trust git's merge output either way (see above) —
manually decide between option (a)/(b) and edit master's lines 836–838
directly, then manually revert whatever git spliced into the `test-xmx-config`
block back to master's original `target_include_directories(...)` form.
Flagging this as the **highest-risk single hunk in the whole build-system
group.**

---

## `scripts/ui-assets.cmake`

**Fork intent:** in `hf_download()`, after "archive verified and extracted",
adds a `message(WARNING ...)` when the resolved version came from the floating
`"latest"` bucket pointer rather than a pinned version — flags that the
embedded UI bundle is unpinned and can silently change contents without a
commit to attribute a break to (llama.cpp-fdm1).

**Upstream intent:** in `npm_build()`, replaces `npm install` with `npm ci`
(and updates both the `message(STATUS ...)` before and the failure message
after to match) — deterministic installs from the lockfile instead of a
mutating install.

**Interaction:** different functions entirely (`hf_download` vs `npm_build`),
no line overlap.

**RESOLVE:** interleave — trivial, apply both independently.

---

## `src/CMakeLists.txt`

**Fork intent:** adds four fork-only source files to the `add_library(llama
...)` list: `llama-kv-block.cpp` (immediately after `llama-kv-cache-dsv4.cpp`),
`llama-pp-scheduler.cpp` (after `llama-memory-recurrent.cpp`, before
`llama-mmap.cpp`), `llama-moe-profile.cpp` (between `llama-model.cpp` and
`llama-quant.cpp`), `llama-tensor-class.cpp` (after `llama-sampler.cpp`,
before `llama-vocab.cpp`). No changes to `set_target_properties` in this diff.

**Upstream intent:** adds two new upstream source files,
`llama-kv-cache-dsa-iswa.cpp` and `llama-kv-cache-msa.cpp`, immediately after
`llama-kv-cache-dsa.cpp` and before `llama-kv-cache-dsv4.cpp`. Also rewrites
`set_target_properties(llama ...)`: `VERSION ${LLAMA_INSTALL_VERSION}` /
`SOVERSION 0` become `VERSION ${LLAMA_VERSION_BASE}` / `SOVERSION
${LLAMA_VERSION_MAJOR}` (follows the root `CMakeLists.txt` version-scheme
rename), and adds a `target_compile_definitions(llama PRIVATE LLAMA_VERSION="..."
LLAMA_COMMIT="...")` block.

**Auto-merges cleanly.** The two sides' insertion points are NOT the same line
— upstream's two files land right after `llama-kv-cache-dsa.cpp`, the fork's
`llama-kv-block.cpp` lands after `llama-kv-cache-dsv4.cpp`, one line further
down — so git's 3-way merge resolves this without a conflict marker. Confirmed
against `git merge-tree --write-tree master b10630` (this path does not appear
in its `CONFLICT` output) and by reading the resulting merged blob directly,
which contains all six new sources in one unbroken run, in this order:
```
llama-kv-cache-dsa.cpp
llama-kv-cache-dsa-iswa.cpp     <- upstream
llama-kv-cache-msa.cpp          <- upstream
llama-kv-cache-dsv4.cpp
llama-kv-block.cpp              <- fork
```
The `set_target_properties`/version rewrite likewise has no fork-side edit to
conflict with — it merges as a clean "theirs" take, but it depends on the root
`CMakeLists.txt` `LLAMA_VERSION_BASE`/`LLAMA_VERSION_MAJOR` variables existing
post-merge (see root entry above) — do not take this hunk before that one
lands, or the build breaks on an undefined variable where
`LLAMA_INSTALL_VERSION` used to be.

**RESOLVE:** this file needs no manual hunk merge. Sanity-check only — after
T15's merge, grep `add_library(llama` in the resulting file and confirm all
six new sources are present (`llama-kv-cache-dsa-iswa.cpp`,
`llama-kv-cache-msa.cpp`, `llama-kv-block.cpp`, `llama-pp-scheduler.cpp`,
`llama-moe-profile.cpp`, `llama-tensor-class.cpp`) and that
`set_target_properties`/`target_compile_definitions` took upstream's form,
sequenced after the root `CMakeLists.txt` version-scheme resolution lands.

---

## `tools/CMakeLists.txt`

**Fork intent:** (1) adds the same directory-scoped
`CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE` icpx-depfile workaround as
`ggml/src/ggml-sycl/CMakeLists.txt` and `tests/CMakeLists.txt` (comment cites
the four affected tool targets: `sycl-source-line-probe`,
`sycl-kernel-bench`/`sycl-mxfp4-source-line-probe`, `sycl-mxfp4-moe-bench`,
`onednn-woq-repro`); (2) adds a `if (GGML_SYCL AND NOT GGML_BACKEND_DL)` block
registering four SYCL dev-benchmark subdirectories
(`sycl-kernel-bench`, `sycl-source-line-probe`, `sycl-mxfp4-moe-bench`,
`onednn-woq-repro`) — guarded as a whole block, not per-target, because these
consume symbols internal to the `ggml-sycl` MODULE that are unavailable under
`GGML_BACKEND_DL` at link time (llama.cpp-o6tf; breaks the nightly Intel Docker
image otherwise). This block is inserted immediately before
`add_subdirectory(fit-params)`.

**Upstream intent:** removes `add_subdirectory(parser)` (dead subdirectory);
adds a `if (GGML_METAL) add_subdirectory(tuning) endif()` block, inserted
immediately *after* `add_subdirectory(fit-params)` and before
`add_subdirectory(results)`.

**Interaction:** the two insertions are adjacent (fork's SYCL block goes
before `fit-params`, upstream's METAL/`tuning` block goes after it, both
between the same two anchor lines `add_subdirectory(export-lora)`/`endif()` and
`add_subdirectory(results)`), which is close enough that a 3-line-context diff
may still conflict-mark depending on merge driver — treat as needing a manual
check even if it applies cleanly. No overlapping *content* (different guard
variables, different subdirectories, different insertion points relative to
`fit-params`).

**RESOLVE:** interleave, order: `add_subdirectory(export-lora)` → `endif()` →
[fork's `GGML_SYCL AND NOT GGML_BACKEND_DL` block, 4 subdirs] →
`add_subdirectory(fit-params)` → [upstream's `GGML_METAL` → `tuning` block] →
`add_subdirectory(results)`. Drop `add_subdirectory(parser)` per upstream. Take
the icpx-depfile workaround (fork, first hunk) verbatim — no upstream touch
there.

---

## `tools/ui/CMakeLists.txt`

**Fork intent:** replaces the entire `add_custom_target(llama-ui-assets ALL
BYPRODUCTS ...)` provisioning block (base lines ~69–90) with a much larger
mechanism: computes a `UI_ASSET_DEPENDS` list sourced from `sources.cmake`
globs, `dist/` contents (with a `dist/_gzip` exclusion pattern shared between a
`list(FILTER ... EXCLUDE REGEX)` and a `foreach` directory-property loop —
comment stresses these must stay byte-identical, gated by
`test-ui-gzip-exclude.cmake`), and per-directory `CMAKE_CONFIGURE_DEPENDS`
properties on `tools/ui` and `tools/ui/dist` so a `dist/` appearing/vanishing
triggers reconfiguration. Converts the custom target from an always-dirty
`add_custom_target(... ALL ...)` (no real OUTPUT) to a real
`add_custom_command(OUTPUT "${UI_ASSETS_STAMP}" ...)` + a thin
`add_custom_target(llama-ui-assets ALL DEPENDS "${UI_ASSETS_STAMP}")` wrapper,
so a clean tree gets a genuine Ninja no-op (llama.cpp-9ubg). Extensive comments
document an "axis inventory" of what was tested (create/delete of `dist/`,
file-vs-directory granularity, content correctness, depth, the gzip exclusion,
dotfiles, symlinks) plus two OPEN items (`sources.cmake` staleness —
llama.cpp-3658 — and npm cold-start).

**Upstream intent:** in the `else()` branch of the `CMAKE_CROSSCOMPILING`
check (base lines ~61–69, just before the fork's replaced block), wraps
`add_executable(llama-ui-embed embed.cpp)` in a save/exclude/restore of
directory-scoped `COMPILE_OPTIONS`/`LINK_LIBRARIES` properties — strips any
`-fsanitize=...` flags for this build-time-only tool (fixes a TSan "memory
layout is incompatible" CI failure), then restores the original directory
properties afterward so the real `llama-ui` library below keeps sanitizer
instrumentation.

**Interaction:** upstream's hunk ends exactly where the fork's replaced block
begins (base line 69, the comment `# Run the provisioning script every build
...`) — adjacent, not overlapping. Low conflict risk but verify the merge
driver doesn't mis-splice the `endif()` that closes upstream's `else()` branch
right before the fork's stamp-based rewrite begins.

**RESOLVE:** interleave — take upstream's sanitizer-exclusion wrapper around
`add_executable(llama-ui-embed ...)` verbatim (it precedes the fork's
replacement region and touches nothing the fork changed), then take the fork's
full stamp/`UI_ASSET_DEPENDS`/axis-inventory rewrite verbatim immediately
after.

---

## `.github/workflows/build-sycl.yml`

**Fork intent:** adds `continue-on-error: true` to the `windows-latest-sycl`
job, immediately after `runs-on: windows-2022` — Windows SYCL is best-effort
because the backend's memory core uses POSIX `mmap` with no ported Windows
equivalent, so this leg is expected to fail and should report without
blocking.

**Upstream intent:** adds a `ccache-clear` step (using the new
`./.github/actions/ccache-clear` composite action) to both the
`ubuntu-24-sycl` job (after the build step) and the `windows-latest-sycl` job
(after the `win-build-sycl.bat` step) — periodic ccache pruning keyed per job,
dry-run unless on a `push` to `master`.

**Interaction:** no line overlap — fork's insert is right after `runs-on:
windows-2022` (top of the windows job); upstream's inserts are at the *end* of
each job's step list. Both land inside the same `windows-latest-sycl` job block
but at opposite ends of it.

**RESOLVE:** interleave, content-only — keep fork's `continue-on-error: true`
and add both of upstream's `ccache-clear` steps. This requires
`.github/actions/ccache-clear` to exist post-merge (an upstream-added composite
action outside this file group; confirm it lands as part of the same wave or
a dependency, not silently dropped). Per this task's scope, runtime CI
enablement (whether these workflows actually execute) is a separate landing
step — see T24's disable loop; this brief covers content resolution only.

---

## `.github/workflows/build-wasm.yml`

**Fork intent:** none beyond what upstream already did — `diff <(git show
master:.github/workflows/build-wasm.yml) <(git show
b10630:.github/workflows/build-wasm.yml)` shows the fork's 90-line version is a
**byte-for-byte prefix** of upstream's 100-line version; the only delta is
upstream's added trailing `ccache-clear` step (key
`webgpu-ubuntu-24.04-arm-wasm`). This file did not exist at the merge-base
(`81ff7abe5`); both sides independently created it as part of splitting the
`ubuntu-wasm` job out of `build-webgpu.yml` (see next entry) — the fork's
version already reflects an earlier upstream restructuring the fork had picked
up before `b10630`.

**Upstream intent:** same split, plus the `ccache-clear` step.

**Interaction:** none — no fork-local content exists to preserve.

**RESOLVE:** theirs — take `b10630`'s version verbatim. Content-only; runtime
CI enablement is T24's concern, not this brief's.

---

## `.github/workflows/build-webgpu.yml`

**Fork intent:** none beyond what upstream already did — same relationship as
`build-wasm.yml` above. `diff <(git show master:...) <(git show b10630:...)`
shows the fork's version is upstream's version *minus* two `ccache-clear`
steps (jobs `webgpu-macos-latest` and `webgpu-ubuntu-24.04`) — every other
change (the `**/*.tmpl` + `embed_wgsl.py` path-trigger addition, and the
`ubuntu-wasm` job removal in favor of the standalone `build-wasm.yml`) is
already identical between the two sides.

**Upstream intent:** same restructuring, plus the two `ccache-clear` steps.

**Interaction:** none — no fork-local content exists to preserve.

**RESOLVE:** theirs — take `b10630`'s version verbatim. Content-only; runtime
CI enablement is T24's concern, not this brief's.

---

## Summary table

| file | conflict shape | RESOLVE |
|---|---|---|
| `CMakeLists.txt` | disjoint hunks, no textual overlap | interleave |
| `ggml/CMakeLists.txt` | disjoint hunks, no textual overlap | interleave |
| `ggml/src/CMakeLists.txt` | disjoint hunks, no textual overlap | interleave |
| `ggml/src/ggml-sycl/CMakeLists.txt` | **real conflict**, and git's own merge marks the WRONG location (spurious hit near `test-xmx-config`; the real `GGML_SYCL_DEVICE_ARCH` site at master lines 836–838 shows no marker and silently loses upstream's feature); 123 fork test registrations elsewhere, untouched by upstream | interleave + **manual flag decision**, ignore git's conflict markers (highest risk) |
| `scripts/ui-assets.cmake` | different functions, no overlap | interleave |
| `src/CMakeLists.txt` | auto-merges cleanly — insertion points differ by one line (`llama-kv-cache-dsa.cpp` vs `llama-kv-cache-dsv4.cpp`); confirmed absent from `git merge-tree`'s CONFLICT output | none needed; sanity-check all six new sources + version rewrite present |
| `tests/CMakeLists.txt` | **real conflict** at merged lines 296–564 (fork's `test-archs-exclude-cli` block vs upstream's `test-generate-models`/`FIXTURES_SETUP` restructuring); the `test-alloc.cpp`/`LLAMA_USE_SYSTEM_GGML` guard auto-merges | interleave, manual splice at lines 296–564 only |
| `tools/CMakeLists.txt` | adjacent (not overlapping) insertions around `fit-params` | interleave |
| `tools/ui/CMakeLists.txt` | adjacent (not overlapping), upstream ends exactly where fork's block begins | interleave |
| `.github/workflows/build-sycl.yml` | disjoint insertions in same job | interleave (content-only) |
| `.github/workflows/build-wasm.yml` | fork is a strict subset of upstream | theirs (content-only) |
| `.github/workflows/build-webgpu.yml` | fork is a strict subset of upstream | theirs (content-only) |

---

## `tests/CMakeLists.txt` — fork-local registrations that must survive

This file carries the largest and most heavily labeled test surface in the
repo. Wave 1 must not lose any of the following when resolving upstream's
edits to the same file.

**Function-level changes (both sides touch `llama_build_and_test`, at
non-overlapping lines — interleaves cleanly):**
- Fork: replaces `set_property(TEST ${TEST_TARGET} PROPERTY LABELS
  ${LLAMA_TEST_LABEL})` with `set_tests_properties(${TEST_TARGET} PROPERTIES
  LABELS "${LLAMA_TEST_LABEL}" SKIP_RETURN_CODE 77)` — bakes the skip-exit-code
  contract into *every* registered test via the shared helper, so a
  model-requiring test that hits `test_skip_no_model()` reports ctest "Skipped"
  instead of "Passed" (llama.cpp-nwip). Also adds a new `llama_test_pytest()`
  helper function (pytest-style Python test registration with the same
  `SKIP_RETURN_CODE 77` contract) and a top-of-file
  `CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE` icpx-depfile workaround (sibling of
  the one in `ggml/src/ggml-sycl/CMakeLists.txt` and `tools/CMakeLists.txt`;
  names the four SYCL-linking tests it fixes:
  `test-sycl-fattn-onednn-gates`, `test-sycl-kernel-profiler`,
  `test-unified-cache-unpin-event`, `test-sycl-zone-reset-live-refusal`).
- Upstream: removes `get-model.cpp` from the default `TEST_SOURCES` list in
  `llama_build_and_test` (a different line in the same function — no overlap).

**LABELS taxonomy actually in use (grep `tests/CMakeLists.txt` for `LABELS`
before touching this file — the list below is a snapshot, not exhaustive, and
will drift):** `sycl`, `moe`, `source`, `hostonly`, `route-table`,
`vram-arena`, `headroom`, `onednn`, `woq`, `mxfp4`, `tdd`, `backend`,
`lifecycle`, `g1`, `python`, `hash-pinned`, `build`, `policy`,
`source-contract`, `audit`, `execution`, `profiling`, `cache`, `mem-handle`,
`env`, `lease`, `bugfix`, `layout`, `model-load`, `gpu-serial`, `fattn`,
`gates`, `main;quant`, `main;tensor`, `main;sycl;shapes`, `main;kv;paged`. The
`main;<x>` compound-label pattern is used specifically because
`llama_build_and_test`'s `LABEL` parameter is a `oneValueArgs` (a `;`-joined
string does not survive `cmake_parse_arguments` there — it explodes into
`UNPARSED_ARGUMENTS`), so those tests set the compound label via a *second*
`set_tests_properties(<target> PROPERTIES LABELS "...")` call after
registration, not via the `LABEL` argument.

**`SKIP_RETURN_CODE 77` sites (beyond the blanket one now baked into
`llama_build_and_test` itself, above):** individually re-affirmed or added at
~20 call sites throughout the file (`test-archs-exclude-cli`,
`test-sycl-alloc-policy`, `test-bench-guard`, the `sycl;policy` family,
`test-sycl-fattn-onednn-gates` in the "verbatim apart from LABELS" restored
block, `test-tensor-class`, and others) — grep `SKIP_RETURN_CODE` in the
current file for the full, current set before editing. This is the mechanism
CLAUDE.md's "Verification Commands" section documents as making a skip
*visible* as a skip rather than a false "Passed".

**`test-gguf.cpp`/`test-alloc.cpp`/`LLAMA_USE_SYSTEM_GGML` anchor auto-merges
cleanly** — verified by running the real 3-way merge (`git merge-file` against
the three blobs `git merge-tree --write-tree master b10630` names for this
path): no conflict marker appears anywhere near this guard. Upstream moves
both `test-gguf.cpp` and `test-alloc.cpp` under a new `if (NOT
LLAMA_USE_SYSTEM_GGML)` guard (dropping `test-alloc`'s previously-standalone
`target_include_directories(... ${PROJECT_SOURCE_DIR}/ggml/src)` line in the
process) and removes the older, separately-located standalone
`llama_build_and_test(test-gguf.cpp)` call that used to sit next to
`test-backend-ops.cpp`; the fork's five restored tests, described below, land
right after this guard in the merged file with no interaction. Fork's `master`
never touches `LLAMA_USE_SYSTEM_GGML` or moves `test-gguf.cpp`, so there is
nothing for these two edits to collide over.

**⚠️ The real conflict is a ~270-line span the earlier draft of this brief
missed entirely: merged-file lines 296–564** (per `git merge-file -L ours -L
base -L theirs`), immediately after `llama_build_and_test(test-llama-archs.cpp)`
inside the `NOT WIN32 OR NOT BUILD_SHARED_LIBS` guard. Both sides insert new
content at that exact point and each side's insertion runs through to the
guard's closing `endif()`:
- **Fork ("ours")** inserts the `test-archs-exclude-cli` block: a
  `llama_build_and_test(test-archs-exclude.cpp)` call plus, under `if (NOT
  WIN32)`, a `llama_test_cmd(bash NAME test-archs-exclude-cli ...)`
  registration (`SKIP_RETURN_CODE 77` when the driven binary wasn't built) —
  gates `-x/--exclude` of `test-llama-archs`, deliberately not named
  `test-llama-archs-exclude` because four CI workflows run `ctest -E
  "test-llama-archs"` (an unanchored regex) that would otherwise swallow it.
- **Upstream ("theirs")** inserts the `test-generate-models` /
  `test-recurrent-state-rollback` fixture restructuring at the same point: a
  `MODEL_DIR` + `llama_test(test-llama-archs NAME test-generate-models ...)`
  with `FIXTURES_SETUP generate-models`, then three
  `test-recurrent-state-rollback*` variants (dense, `-nemotron-h`, `-dsv4`)
  each with `FIXTURES_REQUIRED generate-models`, replacing the old
  `llama_build_and_test(test-recurrent-state-rollback.cpp LABEL "model" ARGS
  -m "${MODEL_DEST}")` registration that lived elsewhere in the base file.

Both blocks are real and both-sides-needed; the fix is additive, not
exclusive. Also restores five host-only C++ tests, unrelated to and *not*
overlapping this span — `test-tensor-data`, `test-q8-0-soa`,
`test-mmq-soa-q4-0`, `test-sycl-model-shapes`, and `test-tensor-class` (the
fifth; links the classifier from libllama per llama.cpp-habh rather than
compiling `src/llama-tensor-class.cpp` a second time into the test binary) —
plus `test-q4-0-q8-0-vec-dot-regression` (with its own
`target_include_directories(... ${PROJECT_SOURCE_DIR}/ggml/src)`, needed
because the restored source includes `ggml-cpu/quants.h` → `ggml-common.h`),
citing llama.cpp-0igs and a specific pre-fork commit (`3c8f296fd`) as the
provenance for "byte-identical, restored, not reinvented" — do not re-derive
these from scratch during merge; keep the restored blocks intact. These land
after the (auto-merging) `LLAMA_USE_SYSTEM_GGML` guard described above, well
past the real conflict span.

Other upstream additions elsewhere in the file that fork doesn't touch and
must simply be carried forward: `test-unicode.cpp`, `test-batch-alloc.cpp`
(inside the `NOT WIN32 OR NOT BUILD_SHARED_LIBS` guard), `test-chat-analysis.cpp`
(build only, not registered as a test), `test-model-resolution.cpp` (+
`target_link_libraries(... PRIVATE cpp-httplib)`), `test-mtmd-impl.cpp` (+
`target_link_libraries(... PRIVATE mtmd)`), and `test-rset-release.cpp` under
`if (APPLE)`.

**RESOLVE (tests/CMakeLists.txt as a whole):** interleave, with manual care at
exactly one point — the ~270-line span at merged lines 296–564 described
above: keep the fork's `test-archs-exclude-cli` block, then insert upstream's
`test-generate-models`/`test-recurrent-state-rollback` restructuring
immediately after it, then a single closing `endif()` (git's raw merge
duplicates or drops this incorrectly when the span is this size — verify by
hand that exactly one `endif()` closes the `NOT WIN32 OR NOT
BUILD_SHARED_LIBS` guard afterward). The `LLAMA_USE_SYSTEM_GGML`/`test-alloc`
anchor needs no manual attention — sanity-check only. Every other hunk in this
file is additive on one side only and carries forward mechanically.
