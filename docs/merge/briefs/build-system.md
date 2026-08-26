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
`src/CMakeLists.txt`, `tools/CMakeLists.txt`,
`tools/ui/CMakeLists.txt`, and `.github/workflows/build-sycl.yml` — auto-merge
cleanly despite adjacent or nearby hunks; their entries below describe the
apparent adjacency for context and end in a sanity-check, not a conflict
resolution. Confirm the four-file list yourself before trusting any of this:

```bash
git merge-tree --write-tree master b10630 > /tmp/mt.txt
grep CONFLICT /tmp/mt.txt
```

Merged-file line numbers cited below (e.g. "merged lines 296–564") are read
directly out of the blob `git merge-tree --write-tree` produces — this is the
merge-ort algorithm, the one `git merge`/wave-1's actual merge uses. They were
re-derived at fork commit `bf03d2031` (`master` at that point; `b10630` is a
fixed tag) via:

```bash
tree=$(git merge-tree --write-tree master b10630)   # writes conflict markers into the blob for a conflicted path
git show ${tree}:<path> | grep -n '^<<<<<<<\|^=======\|^>>>>>>>'
```

This is a different algorithm from `git merge-file` (plain 3-way, xdiff/diff3)
run directly against the three blobs `git merge-tree` names for a path — the
two agree on `tests/CMakeLists.txt` (both land at 296/523/564) but **diverge**
on `ggml/src/ggml-sycl/CMakeLists.txt`, where `merge-file` reports a conflict
at a location `merge-tree`/merge-ort does not touch at all. Agreement on one
file is not evidence for another. Treat `git merge-tree --write-tree` as the
sole authority for conflict location in this brief; do not extrapolate from a
`merge-file` experiment on a different file, however similar it looks.

To reproduce the `merge-file` cross-check for any path: `git merge-tree`'s own
plumbing output (without `--write-tree`) lists the base/ours/theirs blob OIDs
for every conflicted path in `<mode> <oid> <stage> <path>` form (stage 1 =
base, 2 = ours, 3 = theirs) — extract the three blobs by OID and diff them
directly, no worktree or checkout needed:

```bash
git merge-tree master b10630 | awk -v p='tests/CMakeLists.txt' \
  '$0 ~ "\t"p"$" {print $2, $3}'                # -> <oid> <stage> per line, 3 lines for a conflict
git show <stage-1-oid> > base.txt   # repeat for stage 2 (ours) -> ours.txt, stage 3 (theirs) -> theirs.txt
cp ours.txt merged.txt
git merge-file -L ours -L base -L theirs merged.txt base.txt theirs.txt
grep -n '^<<<<<<<\|^=======\|^>>>>>>>' merged.txt
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

**Actual merge-ort conflict location** (the authority — `git merge-tree
--write-tree master b10630`, then `git show <tree>:ggml/src/ggml-sycl/CMakeLists.txt`,
re-derived at fork commit `bf03d2031`): **one conflict; markers at merged
lines 866 / 3617 / 3632.** Line numbers below are marker-exclusive — content
starts the line *after* an opening marker and ends the line *before* the
next one, since an executor resolving this keeps the content, never a marker
line. This is NOT a small, localized hunk:

- **OURS (master) content, lines 867–3616 — 2,750 lines, containing 73
  `add_executable(...)` registrations** (counted directly:
  `sed -n '867,3616p' <merged-blob> | grep -c 'add_executable('`). It opens
  immediately after `set(XMX_TEST_SYCL_OPTIONS ...)` with the
  `# ⚠️ UNIFIED_KERNEL_TEST_STANDALONE IS RETIRED. Do not reintroduce it.`
  comment, and closes right after `test-cpu-gpu-soa-interaction`'s
  `RUN_SERIAL TRUE` property line — i.e. it spans nearly the entire
  `GGML_SYCL_BUILD_XMX_TESTS`-gated block described above. `test-xmx-config`'s
  full registration (merged lines 1194–1215: the `add_executable(...)`
  through its `set_tests_properties(xmx-config PROPERTIES ... TIMEOUT 60)`
  call) sits fully **inside** this span, completely intact — it is not
  touched, spliced, or altered by the merge in any way. A plain
  `git merge-file` 3-way merge (xdiff/diff3) against the same three blobs
  reports a conflict there instead — a different algorithm producing a
  different, wrong answer, not a nuance of the real one (see the method note
  below). There is no `test-xmx-config` splice to revert.
- **THEIRS (b10630) content, lines 3618–3631** — upstream's entire patch to
  this file reproduced in full (the
  `ProcessorCount`/`GGML_SYCL_MAX_PARALLEL_LINK_JOBS` addition to the
  `GGML_SYCL_DEVICE_ARCH` AOT block, shown above under "Upstream intent").
- A trailing `)` + `endif()` is shared context immediately after the closing
  marker, closing both sides' surrounding blocks.

**The core problem stands, just relocated:** master's real
`GGML_SYCL_DEVICE_ARCH` site — lines 836–838, the `--offload-arch=` form
quoted above — passes through this merge with **no conflict marker anywhere
near it**, and `-fsycl-max-parallel-link-jobs`/`ProcessorCount` appear nowhere
in the merged file except inside the THEIRS content of this conflict. So
resolving the conflict as "ours" — i.e. keeping the 2,750-line block
wholesale, which is what a shallow "just take ours, it's bigger and it's all
our tests" instinct would do — silently discards upstream's
parallel-link-jobs feature exactly as before; it does not appear anywhere
else to be recovered from. The option (a)/(b) decision from "Upstream intent"
above still stands and still requires a manual edit to lines 836–838
specifically, independent of how this conflict itself is resolved.

**Method note:** a plain `git merge-file` 3-way merge against the same three
blobs agrees with this `git merge-tree` result for `tests/CMakeLists.txt`
(both land at marker lines 296/523/564) but **disagrees** for this file,
placing its conflict at the wrong location entirely (see above). One file's
agreement is not evidence for another's — this file's ~4,900-line rewrite is
large enough that a plain xdiff-based 3-way merge loses track of which block
corresponds to which. `git merge-tree --write-tree` (merge-ort) is the tool
wave-1's actual merge runs, and is the sole authority used for every
coordinate in this brief.

**RESOLVE:** for this conflict, take the OURS content in full (867–3616, all
2,750 lines / 73 registrations — this is the fork's entire XMX test suite and
upstream contributes nothing inside this span). Separately and additionally,
manually resolve the `GGML_SYCL_DEVICE_ARCH` AOT flag question at lines
836–838 — this is NOT satisfied by resolving the conflict above, since
upstream's patch to that block never appears as a marked conflict there.
Manually decide between option (a) (port `-fsycl-max-parallel-link-jobs`
onto the fork's `--offload-arch=` lines) or (b) (restore upstream's
`spir64_gen` form and re-verify `--offload-arch=` wasn't load-bearing) before
landing. Also carry forward the option/glob/link-workaround additions outside
the 867–3616 span verbatim (upstream touches none of that region). Flagging
this as the **highest-risk single hunk in the whole build-system group.**

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
in its `CONFLICT` output). Reading the resulting merged blob directly: only
three of the six new sources land contiguously — upstream's two plus the
fork's `llama-kv-block.cpp` — interspersed with the two pre-existing anchor
lines each side inserted next to:
```
llama-kv-cache-dsa.cpp          (pre-existing anchor)
llama-kv-cache-dsa-iswa.cpp     <- upstream, new
llama-kv-cache-msa.cpp          <- upstream, new
llama-kv-cache-dsv4.cpp         (pre-existing anchor)
llama-kv-block.cpp              <- fork, new
```
The other three fork-only sources are NOT part of this run — they sit
scattered elsewhere in the same `add_library(llama ...)` list, at their
already-described anchor points (`llama-pp-scheduler.cpp` after
`llama-memory-recurrent.cpp`, `llama-moe-profile.cpp` between
`llama-model.cpp` and `llama-quant.cpp`, `llama-tensor-class.cpp` after
`llama-sampler.cpp`). All six are present in the merged file — they are just
not contiguous with each other.

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
`add_subdirectory(results)`) but do not overlap. **Auto-merges cleanly** —
confirmed absent from `git merge-tree --write-tree master b10630`'s
`CONFLICT` output.

**RESOLVE:** none needed; sanity-check the resulting order: `add_subdirectory(export-lora)` → `endif()` →
[fork's `GGML_SYCL AND NOT GGML_BACKEND_DL` block, 4 subdirs] →
`add_subdirectory(fit-params)` → [upstream's `GGML_METAL` → `tuning` block] →
`add_subdirectory(results)`, that `add_subdirectory(parser)` is gone per
upstream, and that the icpx-depfile workaround (fork, first hunk) is present
verbatim.

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
...`) — adjacent, not overlapping. **Auto-merges cleanly** — confirmed absent
from `git merge-tree --write-tree master b10630`'s `CONFLICT` output.

**RESOLVE:** none needed; sanity-check that the resulting file has upstream's
sanitizer-exclusion wrapper around `add_executable(llama-ui-embed ...)`
followed immediately by the fork's full stamp/`UI_ASSET_DEPENDS`/axis-inventory
rewrite, with exactly one `endif()` closing upstream's `else()` branch between
them (not swallowed or duplicated).

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

**RESOLVE:** theirs — take `b10630`'s version verbatim:

```bash
git show b10630:.github/workflows/build-wasm.yml > .github/workflows/build-wasm.yml
git add .github/workflows/build-wasm.yml
```

Content-only; runtime CI enablement is T24's concern, not this brief's.

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

**RESOLVE:** theirs — take `b10630`'s version verbatim:

```bash
git show b10630:.github/workflows/build-webgpu.yml > .github/workflows/build-webgpu.yml
git add .github/workflows/build-webgpu.yml
```

Content-only; runtime CI enablement is T24's concern, not this brief's.

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

**The real conflict spans marker-exclusive content lines 297–522 (OURS) and
524–563 (THEIRS)**, markers at 296/523/564 (per `git merge-tree --write-tree
master b10630`, cross-checked with `git merge-file -L ours -L base -L
theirs` against the same three blobs — the two agree here, see the method
note above), immediately after `llama_build_and_test(test-llama-archs.cpp)`
inside the `NOT WIN32 OR NOT BUILD_SHARED_LIBS` guard.

**OURS is NOT just the `test-archs-exclude-cli` block — it is 226 lines
registering 12 tests across 8 separate `if (GGML_SYCL AND NOT
GGML_BACKEND_DL)` guards**, verified by walking the merged blob line by
line. In order: `test-archs-exclude.cpp` plus the `if (NOT WIN32)`-gated
`test-archs-exclude-cli` (the pair described just below), then
unconditionally `test-archs-table.cpp` (line 337) and
`test-sycl-moe-mmvq-tables.cpp` (line 345 — deliberately NOT SYCL-guarded per
its own comment, since it's host-only), then eight independently-gated
registrations, each in its own `if (GGML_SYCL AND NOT GGML_BACKEND_DL)`
block: `test-sycl-route-table-stamp` (360), `test-sycl-vram-headroom` (381),
`test-onednn-woq-mxfp4` (401), `test-sycl-mxfp4-woq-repack` (426),
`test-sycl-mxfp4-woq-tiled-repack` (453), `test-sycl-mxfp4-woq-repack-bench`
(477, `add_executable` only, not a ctest), `test-sycl-mxfp4-woq-gemm-bench`
(496, likewise), and `test-sycl-moe-routing-device-oracle` (521, likewise).
- The archs-exclude pair: a `llama_build_and_test(test-archs-exclude.cpp)`
  call plus, under `if (NOT WIN32)`, a `llama_test_cmd(bash NAME
  test-archs-exclude-cli ...)` registration (`SKIP_RETURN_CODE 77` when the
  driven binary wasn't built) — gates `-x/--exclude` of `test-llama-archs`,
  deliberately not named `test-llama-archs-exclude` because four CI
  workflows run `ctest -E "test-llama-archs"` (an unanchored regex) that
  would otherwise swallow it.

**THEIRS is upstream's `test-generate-models`/`test-recurrent-state-rollback`
fixture restructuring**, 40 lines: a `MODEL_DIR` + `llama_test(test-llama-archs
NAME test-generate-models ...)` with `FIXTURES_SETUP generate-models`, then
three `test-recurrent-state-rollback*` variants (dense, `-nemotron-h`,
`-dsv4`) each with `FIXTURES_REQUIRED generate-models`, replacing the old
`llama_build_and_test(test-recurrent-state-rollback.cpp LABEL "model" ARGS
-m "${MODEL_DEST}")` registration that lived elsewhere in the base file.

**⚠️ Why a naive OURS+THEIRS+shared-`endif()` concatenation is wrong — the
true mechanism, not a guess about size:** the single `endif()` immediately
after THEIRS's content (the line right after the closing conflict marker) is
textually unchanged from the base file, so merge-ort factored it as shared
trailing context instead of duplicating it per side. But it closes a
*different construct* on each side. Verified from the raw
`git diff 81ff7abe5..master -- tests/CMakeLists.txt`: master closes the outer
`NOT WIN32 OR NOT BUILD_SHARED_LIBS` guard *early*, with a **new** `endif()`
added immediately after the `if (NOT WIN32) ... endif()` sub-block for
`test-archs-exclude-cli` — so every registration after that point, including
the very last one (`test-sycl-moe-routing-device-oracle`, opening at line
521), is unconditional or independently SYCL-gated, and it is *that* last
block whose closing `endif()` is the base's original line, reused. In
upstream's file the same original `endif()` still closes the outer `NOT
WIN32 OR NOT BUILD_SHARED_LIBS` guard, unchanged, because upstream never
restructured it — its four new fixture tests are inserted immediately before
that same line, still inside the guard. So concatenating OURS in full, then
THEIRS in full, then the one shared `endif()`, produces CMake that is
syntactically balanced (every `if` still has a matching `endif`) but
semantically wrong: upstream's `MODEL_DIR`/`test-generate-models`/rollback
fixtures end up nested inside `if (GGML_SYCL AND NOT GGML_BACKEND_DL)` (the
guard `test-sycl-moe-routing-device-oracle` opened, since nothing ever closes
it within OURS's own content) — so those four tests silently do not register
at all on a non-SYCL build, which is the common, default configuration. This
is not "git duplicates or drops an endif because the span is large" — it is
one specific, identifiable `endif()` correctly recognized as textually
identical on both sides while playing two different structural roles.

Also restores five host-only C++ tests, unrelated to and *not* overlapping
this span — `test-tensor-data`, `test-q8-0-soa`, `test-mmq-soa-q4-0`,
`test-sycl-model-shapes`, and `test-tensor-class` (the fifth; links the
classifier from libllama per llama.cpp-habh rather than compiling
`src/llama-tensor-class.cpp` a second time into the test binary) — plus
`test-q4-0-q8-0-vec-dot-regression` (with its own
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

**RESOLVE (tests/CMakeLists.txt as a whole):** every hunk in this file except
one is additive on one side only and carries forward mechanically (including
the auto-merging `LLAMA_USE_SYSTEM_GGML`/`test-alloc` anchor — no manual
attention needed there). The one exception, the 297–522/524–563 conflict, is
manual and has three parts:
1. Keep every line of OURS (297–522) in order — all 12 registrations across
   the 8 SYCL guards, unchanged.
2. Insert THEIRS's 40 lines (524–563) immediately after the `if (NOT WIN32)
   ... endif()` sub-block for `test-archs-exclude-cli`, and *before* the very
   next `endif()` (the one that closes the outer `NOT WIN32 OR NOT
   BUILD_SHARED_LIBS` guard early). This is the one placement that preserves
   upstream's own original nesting intent — upstream inserted this content
   immediately after `test-llama-archs.cpp` and before that guard's `endif()`
   in their own diff — without perturbing any of OURS's already-correct
   structure.
3. Add a **new** `endif()` immediately after
   `target_link_libraries(test-sycl-moe-routing-device-oracle PRIVATE
   llama-common)`, closing the one guard that no longer has a closer once
   THEIRS's content moves to step 2: the pre-existing trailing `endif()` (that
   used to sit right after THEIRS in the raw conflict) now correctly closes
   only the outer guard, immediately following THEIRS's newly-relocated
   content, and this new one closes `test-sycl-moe-routing-device-oracle`'s
   own guard where it was previously left open.

Verify by constructing the resolved region and hand-checking guard nesting —
every `if` opened in this span must close before its next sibling
registration begins, and `test-generate-models`/`test-recurrent-state-rollback*`
must NOT end up inside any `if (GGML_SYCL ...)` block.

---

## Summary table

| file | conflict shape | RESOLVE |
|---|---|---|
| `CMakeLists.txt` | disjoint hunks, no textual overlap. **AUTO-MERGES** | none needed; sanity-check exactly one `include(CTest)` (the early one) and the vendor/version-scheme hunks landed |
| `ggml/CMakeLists.txt` | disjoint hunks, no textual overlap. **AUTO-MERGES** | none needed; sanity-check `enable_testing()` appears exactly once (the fork's early, broadened gate) |
| `ggml/src/CMakeLists.txt` | disjoint hunks, no textual overlap. **AUTO-MERGES** | none needed; sanity-check `ggml-private-test-registry` and the OpenMP-fetch rewrite both landed |
| `ggml/src/ggml-sycl/CMakeLists.txt` | **real conflict**, markers at 866/3617/3632 (OURS content 867–3616, 2,750 lines / 73 registrations, vs upstream's whole patch at 3618–3631); the real `GGML_SYCL_DEVICE_ARCH` site at master lines 836–838 carries no marker at all and silently loses upstream's feature regardless of how the conflict is resolved | take OURS content 867–3616 + **separate manual flag decision** at 836–838 (highest risk) |
| `scripts/ui-assets.cmake` | different functions, no overlap. **AUTO-MERGES** | none needed; sanity-check both `hf_download()` and `npm_build()` changes landed |
| `src/CMakeLists.txt` | insertion points differ by one line (`llama-kv-cache-dsa.cpp` vs `llama-kv-cache-dsv4.cpp`). **AUTO-MERGES** — confirmed absent from `git merge-tree`'s `CONFLICT` output | none needed; sanity-check all six new sources + version rewrite present |
| `tests/CMakeLists.txt` | **real conflict**, markers at 296/523/564 (OURS content 297–522, 226 lines / 12 registrations across 8 SYCL guards, vs upstream's `test-generate-models`/`FIXTURES_SETUP` restructuring at 524–563); a naive ours+theirs+shared-`endif()` concatenation nests upstream's 4 new tests inside the last SYCL guard, silently dropping them on non-SYCL builds. The `test-alloc.cpp`/`LLAMA_USE_SYSTEM_GGML` guard elsewhere in the file auto-merges | manual splice — see the full `tests/CMakeLists.txt` entry above (highest-risk region in this file) |
| `tools/CMakeLists.txt` | adjacent (not overlapping) insertions around `fit-params`. **AUTO-MERGES** — confirmed absent from `CONFLICT` output | none needed; sanity-check ordering and that `add_subdirectory(parser)` is gone |
| `tools/ui/CMakeLists.txt` | adjacent (not overlapping), upstream ends exactly where fork's block begins. **AUTO-MERGES** — confirmed absent from `CONFLICT` output | none needed; sanity-check exactly one `endif()` between upstream's and the fork's blocks |
| `.github/workflows/build-sycl.yml` | disjoint insertions in same job. **AUTO-MERGES** | none needed; sanity-check `continue-on-error: true` plus both `ccache-clear` steps landed |
| `.github/workflows/build-wasm.yml` | fork is a strict subset of upstream. **Real conflict (add/add)** per `git merge-tree`, but trivial — no fork-local content | `git show b10630:.github/workflows/build-wasm.yml > .github/workflows/build-wasm.yml && git add .github/workflows/build-wasm.yml` (theirs, content-only) |
| `.github/workflows/build-webgpu.yml` | fork is a strict subset of upstream. **Real conflict (content)** per `git merge-tree`, but trivial — no fork-local content | `git show b10630:.github/workflows/build-webgpu.yml > .github/workflows/build-webgpu.yml && git add .github/workflows/build-webgpu.yml` (theirs, content-only) |
