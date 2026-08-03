# SYCL test trustworthiness inventory (llama.cpp-u2mz)

**You cannot certify a merge on tests that cannot fail.** This is a first pass at
proving, for SYCL-related tests, that they can actually go red — not just that
they exist and print `PASS`. Method: mutate the property the test claims to
check, then ask whether the test would notice. For any test that needs a build
or a GPU, that mutation was *reasoned through by code inspection* (real
function bodies traced by hand against the test's assertions) rather than
executed, because this pass ran without `BUILD.lock` or device access — those
still need the lead (or a follow-up) to actually run the described mutation and
confirm the reasoning.

## ⚠️ TOP-LINE ANSWER: the `llama.cpp-0igs` merge-blocker question

**Of the still-unregistered test sources, 64 cover code this branch changed.**
Redirected here by the lead: exhaustively mutation-verifying the whole
remaining population (~65–70 more registered C++ tests, ~90 Python/shell
gates) has falling marginal value and is not what gates *this* merge — the
question that matters is how much of `llama.cpp-0igs`'s ~147-file backlog is
actually merge-relevant.

**Method:**
1. Changed surface: `git diff --name-only master...HEAD -- ggml/src/ggml-sycl/ src/`
   → 44 files (`fattn*`, `unified-cache`/`unified-kernel`, `mem-handle`,
   `kv-tier-manager`, `zone-sizing`, `gpu-arch`/`cold-start`, `dispatch.hpp`,
   `xmx-dispatch-gate.hpp`, `common.hpp/cpp`, `binbcast.cpp`, plus
   `src/llama-model.cpp/h` and `src/llama-moe-profile.cpp` on the MoE side).
2. Registration status for all 291 unique `tests/*.cpp` +
   `ggml/src/ggml-sycl/tests/*.cpp` source names: literal-name grep against
   both `CMakeLists.txt` files (catches `foreach`-loop registrations, where
   the name appears in the loop list even though `add_executable` uses a
   variable) → **166 unregistered** (125 registered, matching the lead's
   171-registration figure once Python/shell registrations are added back in).
3. Intersected the 166 against the changed surface two ways: does the source
   `#include` one of the 12 changed headers, or does its filename contain one
   of 21 keyword stems drawn from the changed subsystems (`fattn`, `onednn`,
   `xmx`, `unified-cache`, `unified-kernel`, `mem-handle`, `mem-ops`,
   `kv-tier`, `zone-sizing`, `gpu-arch`, `cold-start`, `binbcast`, `dispatch`,
   `graph-replay`, `cross-model`, `moe`, `tensor-usage`, `tensor-placement`,
   `tensor-class`, `expert`, `grovemoe`/`chexps`) → **65 hits**.
4. Split those 65 by whether they existed at `3c8f296fd` (the pre-wipe
   commit) — this matters because the lead's own rule is explicit: **a file
   registered at `3c8f296fd` belongs on the `0igs` restoration list even if
   currently vacuous; only a never-registered file is a delete candidate.**
   **64 of 65 existed at `3c8f296fd`** (genuine wipe casualties). The one
   exception, `tests/bench-sycl-fattn-gptoss.cpp`, is new to this branch and
   named like the other `bench-*`/`bench-dnnl-ops.cpp` files that were never
   meant to be ctest targets — almost certainly not a real restoration
   candidate, though not individually confirmed.

**So: 64, not 147.** Full sorted list (all under `tests/`, all pre-existing
at `3c8f296fd`): `mini-context-prototype`, `test-cold-start`,
`test-cpu-gpu-soa-interaction`, `test-dmmv-q4-0-coalesced`,
`test-dmmv-q6k-coalesced`, `test-expert-cache`,
`test-expert-routing-roundtrip`, `test-fattn-thread-local`,
`test-ggml-sycl-soa`, `test-layout-bytes`, `test-mmq-q6k-gpu`,
`test-mmvq-q8-0-streaming-bench`, `test-moe-expert-placement`,
`test-moe-mini-graph`, `test-moe-mul-mat-id`, `test-moe-mul-mat-id-q4q8`,
`test-mul-mat-host-streaming`, `test-mxfp4-xmx-tiled`,
`test-onednn-fallback`, `test-onednn-woq`, `test-pinned-chunk-pool`,
`test-planner-canary-cpy-visibility`, `test-planner-canary-direct-load`,
`test-planner-canary-pp-tg-union`,
`test-planner-canary-skeleton-determinism`, `test-q6k-56block-debug`,
`test-q6k-dispatch`, `test-q6k-layout-debug`, `test-q6k-reorder-dispatch`,
`test-q6k-variable-reorder`, `test-q8-0-layout-cache-path`,
`test-q8-0-layout-cache-path-mmvq`, `test-sycl-cpu-dispatch`,
`test-sycl-expert-cache-bandwidth`, `test-sycl-expert-prefetch`,
`test-sycl-fattn-onednn-descriptors`,
`test-sycl-fattn-onednn-materialization`, `test-sycl-fattn-xmx-policy`,
`test-sycl-kernel-selection`, `test-sycl-kv-planned-device-materialization`,
`test-sycl-moe-expert-parallelism`, `test-sycl-moe-handle-resolution`,
`test-sycl-moe-identity-hash`, `test-sycl-moe-q8-scratch`,
`test-sycl-onednn-packed-cache`, `test-sycl-orchestrator`,
`test-sycl-prestage-routed-experts`, `test-sycl-race-conditions`,
`test-sycl-set-rows-owner-routing`, `test-sycl-unified-cache`,
`test-sycl-unified-memory-e2e`, `test-sycl-weight-key-stability`,
`test-sycl-weight-key-uniqueness`, `test-sycl-xmx-unified-correctness`,
`test-tensor-classification`, `test-tiered-dispatch`,
`test-tile-decomposition`, `test-unified-cache-concurrent`,
`test-unified-cache-integrity`, `test-unified-dispatch-integration`,
`test-xmx-host-streaming`, `test-xmx-kernel-config`,
`test-xmx-quant-loaders`, `test-xmx-unified-kernel`.

Aggregate size, as a build-cost proxy: **~26,000 lines, ~347
`CHECK`/`assert`/`TEST_FAIL`/`EXPECT_`-style call sites** across the 64. At
the ticket's own ~15-min-per-file `ocloc` estimate, restoring this subset is
**~16 hours** of serialised build time against ~35.5 hours for the full
~142-file backlog — a real reduction, not a rounding difference.

**Caveats, stated plainly rather than glossed over:**
- This is a *topical-relevance* filter (header includes + filename
  keywords), not a read of each file. Two spot-checks (`test-moe-expert-placement.cpp`,
  `test-sycl-unified-cache.cpp`) confirmed genuine, substantive relevance —
  but the filter can't distinguish a real regression test from another
  `test-graph-replay`-shaped exploration script that happens to touch a
  changed subsystem. `test-moe-expert-placement.cpp` in particular opens with
  "Micro-benchmark" framing and manual build instructions, the same shape as
  the two files already deleted this pass — it is on the *coverage-relevant*
  list, not a confirmed-good list. Per the lead's own rule this is moot for
  the disposition question (registered-at-`3c8f296fd` → restore, not
  delete), but it matters for how much of the resulting coverage is real
  once restored.
- The keyword list is mine, not exhaustive-by-construction; a false negative
  (a relevant file missed because it uses vocabulary outside the 12
  headers/21 keywords) is more likely than a false positive here, so 64
  should be read as a floor, not a ceiling.
- None of the 64 were mutation-tested this pass — that is the natural next
  step once the lead has the restoration-vs-defer decision this number is
  for.

## Task 4a: registration provenance (static audit)

Scope is exactly the 64 merge-relevant names in the top-line 0igs list: the handoff's 41 restored rows, its remaining 22-file set, and the separately recorded `test-expert-cache` pre-wipe exception. This is registration provenance only; hazard classes and final dispositions are intentionally deferred to Tasks 4b/4c.

Evidence keys: historical references are `git show 3c8f296fd:tests/CMakeLists.txt` line numbers; live references are `ggml/src/ggml-sycl/CMakeLists.txt` line numbers at `c27ba2292`. A full literal-name audit found **none of these 64 names in live `tests/CMakeLists.txt`**. “Registration unguarded” means no surrounding CMake `if()` controls the live `add_executable`/`add_test`; target-local link guards do not hide the test. GREEN means an active live `add_test` exists; RED means absent, target-only/manual, commented out, or hidden by a false guard.

### Handoff-restored 41 — all GREEN

| source row | historical registration / guard evidence | live SYCL CMake registration / guard evidence | provenance |
|---|---|---|---|
| `tests/test-cold-start.cpp` | `target 2091; add_test 2099; guard GGML_SYCL` | `target 2010; add_test 2029; registration unguarded` | **GREEN** |
| `tests/test-dmmv-q4-0-coalesced.cpp` | `target 1211; add_test 1219; guard GGML_SYCL` | `target 2099; add_test 2118; registration unguarded` | **GREEN** |
| `tests/test-dmmv-q6k-coalesced.cpp` | `target 1224; add_test 1232; guard GGML_SYCL` | `target 2125; add_test 2144; registration unguarded` | **GREEN** |
| `tests/test-fattn-thread-local.cpp` | `target 1146; add_test 1154; guard GGML_SYCL` | `target 2152; add_test 2171; registration unguarded` | **GREEN** |
| `tests/test-ggml-sycl-soa.cpp` | `target 1431; add_test 1441; guard GGML_SYCL` | `target 2178; add_test 2197; registration unguarded` | **GREEN** |
| `tests/test-layout-bytes.cpp` | `target 1104; add_test 1113; guard GGML_SYCL` | `target 2204; add_test 2223; registration unguarded` | **GREEN** |
| `tests/test-mmq-q6k-gpu.cpp` | `target 559; add_test 568; guard GGML_SYCL` | `target 2230; add_test 2249; registration unguarded` | **GREEN** |
| `tests/test-moe-mini-graph.cpp` | `target 878; add_test 886; guard GGML_SYCL` | `target 2256; add_test 2275; registration unguarded` | **GREEN** |
| `tests/test-moe-mul-mat-id.cpp` | `target 806; add_test 814; guard GGML_SYCL` | `target 2282; add_test 2301; registration unguarded` | **GREEN** |
| `tests/test-moe-mul-mat-id-q4q8.cpp` | `target 852; add_test 860; guard GGML_SYCL` | `target 2308; add_test 2327; registration unguarded` | **GREEN** |
| `tests/test-mul-mat-host-streaming.cpp` | `target 819; add_test 827; guard GGML_SYCL` | `target 2334; add_test 2353; registration unguarded` | **GREEN** |
| `tests/test-onednn-fallback.cpp` | `target 2127; add_test 2131; guard GGML_SYCL` | `target 2360; add_test 2379; registration unguarded` | **GREEN** |
| `tests/test-onednn-woq.cpp` | `target 301; add_test 315; unguarded` | `target 2386; add_test 2408; registration unguarded` | **GREEN** |
| `tests/test-q6k-dispatch.cpp` | `target 1655; add_test 1663; guard GGML_SYCL` | `target 2415; add_test 2434; registration unguarded` | **GREEN** |
| `tests/test-q8-0-layout-cache-path.cpp` | `target 1371; add_test 1381; guard GGML_SYCL` | `target 2441; add_test 2460; registration unguarded` | **GREEN** |
| `tests/test-q8-0-layout-cache-path-mmvq.cpp` | `target 1386; add_test 1396; guard GGML_SYCL` | `target 2467; add_test 2486; registration unguarded` | **GREEN** |
| `tests/test-sycl-cpu-dispatch.cpp` | `target 1838; add_test 1847; guard GGML_SYCL` | `target 2493; add_test 2512; registration unguarded` | **GREEN** |
| `tests/test-sycl-fattn-onednn-materialization.cpp` | `target 1007; add_test 1017; guard GGML_SYCL` | `target 1932; add_test 1951; registration unguarded` | **GREEN** |
| `tests/test-sycl-fattn-xmx-policy.cpp` | `target 1037; add_test 1045; guard GGML_SYCL` | `target 1958; add_test 1977; registration unguarded` | **GREEN** |
| `tests/test-sycl-kernel-selection.cpp` | `target 1538; add_test 1547; guard GGML_SYCL` | `target 1984; add_test 2003; registration unguarded` | **GREEN** |
| `tests/test-sycl-kv-planned-device-materialization.cpp` | `target 1050; add_test 1058; guard GGML_SYCL` | `target 1802; add_test 1821; registration unguarded` | **GREEN** |
| `tests/test-sycl-moe-expert-parallelism.cpp` | `target 1598; add_test 1607; guard GGML_SYCL` | `target 2519; add_test 2538; registration unguarded` | **GREEN** |
| `tests/test-sycl-moe-handle-resolution.cpp` | `target 966; add_test 974; guard GGML_SYCL` | `target 2062; add_test 2081; registration unguarded` | **GREEN** |
| `tests/test-sycl-moe-identity-hash.cpp` | `target 1586; add_test 1595; guard GGML_SYCL` | `target 2545; add_test 2564; registration unguarded` | **GREEN** |
| `tests/test-sycl-moe-q8-scratch.cpp` | `target 979; add_test 987; guard GGML_SYCL` | `target 1828; add_test 1847; registration unguarded` | **GREEN** |
| `tests/test-sycl-onednn-packed-cache.cpp` | `target 1526; add_test 1535; guard GGML_SYCL` | `target 1854; add_test 1873; registration unguarded` | **GREEN** |
| `tests/test-sycl-orchestrator.cpp` | `target 1550; add_test 1559; guard GGML_SYCL` | `target 2571; add_test 2590; registration unguarded` | **GREEN** |
| `tests/test-sycl-prestage-routed-experts.cpp` | `target 2080; add_test 2086; guard GGML_SYCL` | `target 2597; add_test 2616; registration unguarded` | **GREEN** |
| `tests/test-sycl-unified-cache.cpp` | `target 1979; add_test 1990; guard GGML_SYCL` | `target 2623; add_test 2642; registration unguarded` | **GREEN** |
| `tests/test-sycl-unified-memory-e2e.cpp` | `target 2031; add_test 2042; guard GGML_SYCL` | `target 2649; add_test 2668; registration unguarded` | **GREEN** |
| `tests/test-sycl-weight-key-stability.cpp` | `target 1562; add_test 1571; guard GGML_SYCL` | `target 1750; add_test 1769; registration unguarded` | **GREEN** |
| `tests/test-sycl-weight-key-uniqueness.cpp` | `target 1574; add_test 1583; guard GGML_SYCL` | `target 1880; add_test 1899; registration unguarded` | **GREEN** |
| `tests/test-sycl-xmx-unified-correctness.cpp` | `target 904; add_test 912; guard GGML_SYCL` | `target 2675; add_test 2694; registration unguarded` | **GREEN** |
| `tests/test-tensor-classification.cpp` | `target 1874; add_test 1876; guard GGML_SYCL` | `target 1776; add_test 1795; registration unguarded` | **GREEN** |
| `tests/test-tiered-dispatch.cpp` | `target 2006; add_test 2013; guard GGML_SYCL` | `target 2701; add_test 2720; registration unguarded` | **GREEN** |
| `tests/test-unified-cache-concurrent.cpp` | `target 1133; add_test 1141; guard GGML_SYCL` | `target 2036; add_test 2055; registration unguarded` | **GREEN** |
| `tests/test-unified-cache-integrity.cpp` | `target 1089; add_test 1098; guard GGML_SYCL` | `target 1906; add_test 1925; registration unguarded` | **GREEN** |
| `tests/test-xmx-host-streaming.cpp` | `target 832; add_test 847; guard GGML_SYCL` | `target 2727; add_test 2746; registration unguarded` | **GREEN** |
| `tests/test-xmx-kernel-config.cpp` | `target 2168; add_test 2171; guard GGML_SYCL` | `target 2753; add_test 2772; registration unguarded` | **GREEN** |
| `tests/test-xmx-quant-loaders.cpp` | `target 498; add_test 504; unguarded` | `target 2779; add_test 2798; registration unguarded` | **GREEN** |
| `tests/test-xmx-unified-kernel.cpp` | `target 2177; add_test 2183; guard GGML_SYCL` | `target 2805; add_test 2824; registration unguarded` | **GREEN** |

### Handoff remaining 22 — all RED

| source row | historical registration / guard evidence | both live CMake files | provenance |
|---|---|---|---|
| `tests/mini-context-prototype.cpp` | `target 425; guard GGML_SYCL; no add_test (manual-only comment 388–394)` | `absent from both live files` | **RED** |
| `tests/test-cpu-gpu-soa-interaction.cpp` | `target 1417; add_test 1425; guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-expert-routing-roundtrip.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-mmvq-q8-0-streaming-bench.cpp` | `target 715; add_test 723–730 (4 names); guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-moe-expert-placement.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-mxfp4-xmx-tiled.cpp` | `target 792; add_test 801; guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-pinned-chunk-pool.cpp` | `DISABLED - broken test` at 1784–1785; target/test lines 1786–1794 commented out inside GGML_SYCL | `absent from both live files` | **RED** |
| `tests/test-planner-canary-cpy-visibility.cpp` | `target 411; guard GGML_SYCL; no add_test (manual-only comment 388–394)` | `absent from both live files` | **RED** |
| `tests/test-planner-canary-direct-load.cpp` | `target 419; guard GGML_SYCL; no add_test (manual-only comment 388–394)` | `absent from both live files` | **RED** |
| `tests/test-planner-canary-pp-tg-union.cpp` | `target 405; guard GGML_SYCL; no add_test (manual-only comment 388–394)` | `absent from both live files` | **RED** |
| `tests/test-planner-canary-skeleton-determinism.cpp` | `target 397; guard GGML_SYCL; no add_test (manual-only comment 388–394)` | `absent from both live files` | **RED** |
| `tests/test-q6k-56block-debug.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-q6k-layout-debug.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-q6k-reorder-dispatch.cpp` | `target 1667; add_test 1676; guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-q6k-variable-reorder.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-sycl-expert-cache-bandwidth.cpp` | `target 1644; add_test 1652; guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-sycl-expert-prefetch.cpp` | `target 1925; add_test 1935; guards GGML_SYCL + FALSE` | `absent from both live files` | **RED** |
| `tests/test-sycl-fattn-onednn-descriptors.cpp` | `target 1022; add_test 1032; guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-sycl-race-conditions.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-sycl-set-rows-owner-routing.cpp` | `target 1076; add_test 1084; guard GGML_SYCL` | `absent from both live files` | **RED** |
| `tests/test-tile-decomposition.cpp` | `absent (no target/add_test)` | `absent from both live files` | **RED** |
| `tests/test-unified-dispatch-integration.cpp` | `target 2148; add_test 2162; guard GGML_SYCL` | `absent from both live files` | **RED** |

### Separate pre-wipe guard-hidden exception — RED

| source row | historical registration / guard evidence | both live CMake files | provenance |
|---|---|---|---|
| `tests/test-expert-cache.cpp` | `target 1825; add_test 1833; guards GGML_SYCL + FALSE`; preceded by `DISABLED: expert-cache.hpp removed` at lines 1822–1824 | `absent from both live files` | **RED** |

**Completeness check:** 41 GREEN + 22 RED + 1 guard-hidden RED = 64/64 rows. Historical shape within the 22: eight active ctest registrations under `GGML_SYCL`, five `GGML_SYCL` targets explicitly not wired to ctest, one commented-out block, one `GGML_SYCL && FALSE` registration, and seven names absent entirely. The separate exception is another `GGML_SYCL && FALSE` registration. No status in this section asserts safety, usefulness, or a future disposition.


## Two instances named in the ticket

### 1. `tests/test-sycl-tensor-placement.cpp` — FIXED, was a mock

The original file defined its own `test_tensor_type` enum and its own
`is_expert_tensor()`, then asserted that reimplementation against itself. It
never called either real classifier
(`infer_tensor_usage()` in `ggml-sycl/common.hpp`,
`expert_tensor_role_from_tensor_name()` in `ggml-sycl/unified-cache.hpp`), so it
stayed green the entire time the real `infer_tensor_usage()` did not recognize
grovemoe's `ffn_*_chexps` tensors as expert weights — exactly the property its
name claims to cover. That bug was found by a SIGABRT in an unrelated sweep,
months later.

**Fix:** rewritten to `#include "common.hpp"` and call the two real classifiers
directly on realistic tensor names (attention/FFN/norm/embedding, the plain MoE
trio, the fused `gate_up` tensor, grovemoe's `_chexps` trio, arctic's
`ffn_norm_exps`, and the dense `_shexp` family) — 21 checks total. Every
expected value was traced by hand against the current `infer_tensor_usage()`
and `expert_tensor_role_from_tensor_name()` source (both header-only, pure
`strstr`/`strcmp` classifiers — no device needed). **Mutation:** delete the
`ffn_gate_chexps`/`ffn_up_chexps`/`ffn_down_chexps` literals (or the
`ffn_norm_exps` false-positive guard) from either real classifier; the
corresponding `check_usage`/`check_role` call must fail. It was previously
**unregistered** (not present in any `CMakeLists.txt` — `search_text` for
`tensor-placement`/`tensor_placement` across the whole repo returned zero
matches before this change), so in addition to being a mock it also never ran.
Registered in `ggml/src/ggml-sycl/CMakeLists.txt` alongside the other
CPU-only-by-construction tests (`test-sycl-graph-retention-scope`,
`test-sycl-transient-alloc-intent-scope`).

### 2. `tests/test-sycl-graph-retention-scope.cpp` — the ticket's premise was wrong

The ticket's check (`grep -c 'graph-retention-scope' tests/CMakeLists.txt`
returns 0) is true but the file is registered in
**`ggml/src/ggml-sycl/CMakeLists.txt`** (the "un-guarded SYCL tests" section),
not `tests/CMakeLists.txt`. Confirmed multiple ways: it appears in `ctest -N`
(test #14 in this build), has a `CTestCostData.txt` entry (it has actually run:
0.236s), and `build/ggml/src/ggml-sycl/CTestTestfile.cmake` carries its
`add_test`. **This is a live registration, not a lost one** — the ticket's grep
checked the wrong file. (Generalizable trap: `grep <name> tests/CMakeLists.txt`
returning 0 tells you nothing about `ggml/src/ggml-sycl/CMakeLists.txt`, which
independently registers dozens of `tests/test-sycl-*.cpp` files by relative
path.)

On code review it is a well-built, non-decorative test: it calls real
production functions (`ggml_sycl::retain_handles_until_event`,
`ggml_sycl_graph_recording_active`, `ggml_sycl::graph_retained_handle_count`,
`ggml_sycl::release_graph_retained_handles`), asserts on real counters with a
distinct `FAIL:` message per property, and property 2
(`test_own_recording_without_sink_retains_for_graph_lifetime`) is explicitly a
**positive control** with a built-in self-check
(`"FAIL: test setup is inert"`) against the setup itself being vacuous. No
mutation was run (no build access this pass), but nothing here needs one to
see that it is not a mock — it exercises the real thread-local/global-atomic
guard the regression (`llama.cpp-oze0`) was about. No fix needed.

## The 19-test residency/MoE `foreach` loop (`ggml/src/ggml-sycl/CMakeLists.txt:1691`)

**Ownership note:** `ggml/src/ggml-sycl/CMakeLists.txt` is `impl-w1`'s file as of
2026-08-02 (corrected mid-ticket after a shared-checkout `git add` swept up
their in-progress `test-cross-model-weight-usage` registration into this
ticket's commit — see the task comment log). The 19 `.cpp` sources this loop
registers are still `tests/` and in scope here; the CMake registration itself
is not.

Spot-checked 2 of 19 (`test-sycl-residency-diagnostics.cpp`,
`test-sycl-moe-same-expert-grouping.cpp`) rather than reading all 19 in full —
recording this as a partial result, not a clearance of the whole loop.
Cross-referenced all 19 against the mechanical bare-`assert()` scan below:
none flagged. Both spot-checked files call real production code, not a
reimplementation:

- `test-sycl-residency-diagnostics.cpp` calls real `ggml_sycl::` `_for_test`
  instrumentation hooks (`residency_diagnostics_record_accept_for_test`,
  `test_cache_replacement_allowed_for_test`, etc.) exposed by
  `residency-plan.hpp`, uses an NDEBUG-proof `CHECK()`/`return 1` macro, and
  includes a specific, plausible mutation target: `test_replacement_guard_refuses_live_or_retired_entries`
  asserts that a live or retired cache entry refuses replacement (`reject_live_lease_pressure == 1`)
  — inverting that guard in the real replacement-allowed logic should fail it.
- `test-sycl-moe-same-expert-grouping.cpp` is a hybrid: it calls real
  `ggml_sycl::test_moe_token_major_metadata_entry`/`_input` types AND reads the
  actual `ggml-sycl.cpp` source text to assert against literal patterns (the
  `read_required_file`/`contains` pair). The source-text half is the fragile
  style `llama.cpp-0igs` already flagged elsewhere (a reformat or rewording
  breaks it without the underlying behavior changing) — worth checking what
  literal strings it greps for if it ever goes red for a non-behavioral reason.

The other 17 in this loop (`test-sycl-mem-handle-lifetime`,
`test-sycl-residency-reservation`, `test-sycl-descriptor-retention`,
`test-sycl-moe-residency-preflight`, `test-sycl-moe-fused-down-sum-policy`,
`test-sycl-moe-token-major-metadata`,
`test-sycl-moe-direct-final-token-major-bridge`,
`test-sycl-moe-direct-final-scratch-plan`, `test-sycl-moe-glu-q8-artifact-policy`,
`test-sycl-moe-glu-q8-fused-store-policy`, `test-sycl-moe-gateup-prepack-policy`,
`test-sycl-moe-gateup-prepack-scratch`,
`test-sycl-moe-xmx-tiled-single-layout-policy`,
`test-sycl-moe-xmx-tiled-single-layout-planner`,
`test-sycl-moe-xmx-tiled-materialization`, `test-sycl-moe-fusion-noactivation`,
`test-sycl-moe-sequence-graphlet-policy`) were confirmed to include a real
production or shared-test header (`unified-cache.hpp`, `moe-layer-plan.hpp`,
`mem-handle.hpp`, or `ggml-sycl/ggml-sycl-test.hpp`) rather than a
self-contained mock — **that rules out the instance-1 defect class
specifically, it is not a mutation-tested clearance.** Listing them here,
un-cleared, rather than silently treating "includes a real header" as "can
fail."

## `ggml/src/ggml-sycl/tests/*.cpp` — two deleted, several confirmed good

Header/mechanism-checked the remaining `.cpp` files directly under
`ggml/src/ggml-sycl/tests/` (distinct from the `tests/` files the CMake loop
above pulls in by relative path).

**Deleted: `test-graph-replay.cpp` and `test-graph-replay-chain.cpp`.** Both
are genuinely "cannot fail," confirmed by reading `main()`, not inferred:
each per-scenario `check()`/inline comparison returns a `bool` or prints
`OK`/`FAIL` as a **string label only** — the caller (a `void`-returning
per-test function) never propagates it, and `main()` unconditionally
`return 0`s unless a `sycl::exception` escapes. A reader has to notice the
printed word "FAIL" themselves; `ctest` (or any script checking exit code)
cannot. Both were **never registered, at any point in this repo's history**
(`git show 3c8f296fd:tests/CMakeLists.txt` has zero hits for either name, and
`search_text` across the live tree, including `build/`, returns zero) —
confirmed via `git log`, the introducing commit (`54b5e6f1e`) is titled "add
**standalone** L0 graph replay behavior tests" and the file's own header
comment gives manual `icpx -fsycl -o test-graph-replay …` build instructions,
consistent with "always meant as an ad-hoc diagnostic, not a ctest test."
Neither exercises any `ggml-sycl` production code — both are pure raw
SYCL/L0 command-graph API exploration (does L0 graph replay re-read updated
`malloc_device`/`malloc_host` memory). This is the `llama.cpp-0igs`
"~30 never registered even at `3c8f296fd`" population, where that ticket's
own text names deletion (with a stated reason) as the correct action rather
than restoration. Deleted here rather than left inert, per this ticket's own
acceptance criteria ("any test that cannot be made to fail is deleted").

**Confirmed good, properly wired:** `test-kernel-dispatch.cpp` and
`test-mmq-xmx-dispatch.cpp` (both currently registered and active — ctest
names `kernel-dispatch` / `mmq-xmx-dispatch`) and `test-xmx-hardware-detect.cpp`,
`test-xmx-esimd-basic.cpp`, `test-esimd-vectorized-dequant.cpp` (registered
but gated behind `GGML_SYCL_BUILD_XMX_TESTS`, OFF by default — same
lower-live-risk caveat as `test-xmx-config.cpp` above). All five use the
`all_passed &= test_...()` accumulator pattern and end with
`return all_passed ? 0 : 1;` (with an explicit, narrow graceful-skip `return 0`
only when hardware genuinely lacks the feature being tested, e.g. no XMX/ESIMD
support) — real failure propagation, not discarded like the two deleted
files above. Not individually mutation-tested this pass, but the exit-code
mechanism itself is sound, which is the property the two deletions above
were missing.

## ⚠️ Correction: the earlier "all 72 Python gates pass" baseline was checking the wrong thing

An earlier draft of this doc reported running every `tests/test-sycl-*.py`
file directly (`python3 tests/test-sycl-X.py`) and getting rc=0 across all 72.
**That result is close to meaningless and is retracted.** 68 of the 72 are
**pytest-style**: every check lives inside a bare `def test_*() -> None:`
function with no `if __name__ == "__main__":` block. Invoking such a file with
plain `python3` runs only module-level code (imports, constant tables) and
**calls none of the `test_*` functions** — it cannot observe a failure no
matter what the code under test does, and rc=0 says nothing. Confirmed the
gap is real: rerunning the same 68 files correctly via
`python3 -m pytest -q <files>` actually executes the assertions.

**This is not a defect in the tests — the project already engineered around
it.** `tests/CMakeLists.txt`'s `llama_test_pytest()` macro (used for every one
of these 68) registers the ctest command as
`python3 -c "import pytest; sys.exit(pytest.main(['-q', script]))"`, not a bare
script invocation, and the macro's neighboring comment names the exact trap:
*"an unregistered pytest-style file exits 0 for any state of the code under
test, which would make a gate against vacuous passes itself a vacuous pass."*
Confirmed against the generated registration
(`build/tests/CTestTestfile.cmake`): `test-sycl-link-depends-coverage` really
is invoked through `pytest.main()`. So the registered gates were never
vacuous; **my first verification pass was.** Recording this prominently
because the same mistake is easy to repeat: `python3 <file>.py` is the wrong
command for any file in this population without its own `__main__` block, and
the file itself gives no visual cue — it still parses, still imports cleanly,
still exits 0.

**Corrected baseline**, all 68 pytest-style files run together via
`python3 -m pytest -q`: **470 passed, 3 failed.** The 3 failures are exactly
the two `llama.cpp-0igs` already found and explicitly declined to restore
(`test-sycl-end-to-end-profiling-docs.py::test_research_artifact_is_present`,
`test-sycl-mxfp4-tg-speedup-docs.py::test_final_review_records_rejection_and_copied_validation_evidence`
and `::test_final_review_does_not_present_tiles_as_default_on_or_recommended`
— both assert against files, `research.md` and
`activation/sycl-mxfp4-tg-speedup-final-review-20260707.md`, that have never
existed in this repo). Confirmed both files are **not currently registered**
in `ctest -N`, consistent with 0igs's "declined" disposition — this is not a
live gap, just the same already-triaged finding reproduced independently.

**Two files verified with a real, executed mutation** (not just
code-reading), using a shadow-root technique for the one that reads real repo
files — symlink every top-level entry to the real tree except a writable copy
of the exact file(s) under test, confirmed by checking `Path.resolve()`
doesn't silently collapse back to the real tree (it does if any ancestor in
the path is itself a symlink — a second trap worth recording: make the
*script's own* directory a real copy too, not just the target file's):

- **`test-sycl-gap-causes.py`** (script-style, has `__main__`): subprocess-
  invokes the real `scripts/parse-sycl-gap-causes.py` against a checked-in
  fixture trace and asserts specific classification counts. Copied script +
  fixture + full `scripts/` tree to a scratch dir (needed — the parser
  dynamically loads a second module, `parse-sycl-timeline.py`). Control:
  passes (`rc=0`) with a detailed PASS message naming every count. Mutation:
  one line in the copied parser
  (`if device_gap_has_implicit_serialization(...)` → `if False and …`) —
  `rc=1`, 5 distinct `FAIL:` lines, exactly matching the predicted count shift
  (`implicit_queue_serialization` count 1→0, `truly_idle` count 3→4).
- **`test-sycl-link-depends-coverage.py`** (pytest-style): walks every
  `CMakeLists.txt` in the repo checking that any directory linking with icpx
  also sets `CMAKE_CXX_LINK_DEPENDS_USE_LINKER`. Control against the live
  repo (`python3 -m pytest -q tests/test-sycl-link-depends-coverage.py`):
  5 passed, `rc=0`. Mutation: removed the `set(CMAKE_CXX_LINK_DEPENDS_USE_LINKER FALSE)`
  line from a shadow copy of `ggml/src/ggml-sycl/CMakeLists.txt` (the one
  directory that carries it) — `rc=1`, 3 of 5 checks fail
  (`test_at_least_one_directory_links_with_icpx`,
  `test_every_icpx_linking_directory_disables_linker_depfiles`,
  `test_ggml_sycl_self_declares`), 2 unrelated checks (`NOT_IN_BUILD`,
  `PENDING` bookkeeping) correctly stay green — specificity, not just
  sensitivity, same property the `test-kv-slice-sizing.cpp` model
  demonstrates in C++.

The remaining 66 pytest-style files and the other 3 script-style files
(`test-sycl-gap-device-split.py`, `test-sycl-moe-profile-parser.py`,
`test-sycl-timeline-gap-class-conservation.py`) were run correctly (via
pytest / direct invocation as appropriate) but not individually
mutation-tested this pass.

## Coordination finding that changes the shape of this ticket

**`llama.cpp-0igs`** (P1, currently `open`, owner `impl-0igs` died in the
2026-08-01 host reboot — lease released, task ownerless) already tracks a much
larger version of this exact problem: **141 `add_test()` registrations in
`tests/CMakeLists.txt` were wiped wholesale** near commit `4974bf53c` (the
earlier attribution to `d3dce4e0a` was retracted in that ticket as an unsound
topological-walk artifact — see its own METHOD WARNING). `3c8f296fd` had 147
registrations; HEAD had 6 before that ticket's batches 1–3 restored 68. **~147
C++ restorations remain deferred** in that ticket's own notes (~142 raw
`add_executable` blocks each needing an `ocloc` device link against the shared
`build/`).

This is not a coincidence with this ticket's two "confirmed instances" — a test
that isn't registered at all is the most extreme case of "cannot fail" there
is. **The bulk of that problem is `llama.cpp-0igs`'s scope, not re-litigated
file-by-file here**, but this pass independently found and fixed two directly
on-theme members of that same 186-file dead-source population (both call real
production code, neither is a mock):

- **`tests/test-sycl-tensor-usage.cpp`** — calls `ggml_sycl_get_tensor_usage()`
  (`ggml-sycl.cpp:9525`, the cached wrapper around `infer_tensor_usage()`) on
  real `ggml_tensor` objects built through the CPU backend. Was registered at
  `3c8f296fd`, lost with the rest. Registered here in
  `ggml/src/ggml-sycl/CMakeLists.txt` (needs `ggml-cpu` linked in addition to
  `ggml-base ggml ggml-sycl` for `ggml_backend_cpu_init`). **Mutation:** same
  as instance 1 above, one level up the stack — corrupt the cache-population
  path or `infer_tensor_usage()` itself and a `check_usage` call fails.
- **`tests/test-tensor-placement.cpp`** — calls `ggml_sycl_classify_tensor()`
  (`ggml-sycl.cpp:8115`, a *separate*, simpler dense/expert/other classifier
  used for tiered-memory buffer routing — distinct from `tensor_usage`).
  Registered at `3c8f296fd` with bare `assert()` only; `50723136a` already
  applied the `#undef NDEBUG` fix noted in `llama.cpp-0igs`'s comment, but the
  registration itself was still missing. Registered here. **Mutation:** flip
  any of the 9 `assert()` expectations against the real classifier logic
  (verified by hand: expert pattern is `_exps.`/`_exps_`, dense is
  `blk.` + `attn_`/`ffn_`, everything else falls to "other" including the
  router gate, which the test explicitly documents as "treated as dense").

**Recommendation to the lead:** either reassign `llama.cpp-0igs` to a live
owner or fold its remaining ~147-file backlog into this epic's scope
explicitly — right now it sits ownerless and directly overlaps this ticket's
mandate.

## Mechanical scan: bare `assert()` without `#undef NDEBUG`

This project builds Release with `-O3 -DNDEBUG`, which compiles every bare
`assert()` to nothing (proven by `llama.cpp-0igs`'s own mutation matrix: with
every condition forced false, the binary still exits 0 under `-DNDEBUG`, and
134 without it). A test whose *only* failure signal is a bare `assert()` is
mechanically "cannot fail" without reading a line of test logic. Scanned every
`.cpp` under `tests/` and `ggml/src/ggml-sycl/tests/` (positive-controlled:
this is a plain grep for the literal token, not `search_text`'s index, so it
does not inherit the `ggml-sycl.cpp` blind spot) for `assert(` present without
a preceding `#undef NDEBUG` or a gtest include. 28 files matched; triage:

| file | assert hits | disposition |
|---|---:|---|
| `ggml/src/ggml-sycl/tests/test-esimd-dpas-gate.cpp` | 1 | **false positive** — the word appears only in a comment explaining the file deliberately uses a `CHECK()` macro that always runs, not bare `assert()` |
| `ggml/src/ggml-sycl/tests/test-gpu-arch.cpp` | 1 | same false positive, same `CHECK()` pattern |
| `ggml/src/ggml-sycl/tests/test-zone-sizing.cpp` | 1 | same false positive, same `CHECK()` pattern |
| `ggml/src/ggml-sycl/tests/test-xmx-config.cpp` | 27 | **real, fixed here** — registered but gated behind `GGML_SYCL_BUILD_XMX_TESTS` (OFF by default), so lower live-risk, but genuinely vacuous under `-DNDEBUG` if that option is ever turned on. Fixed with `#undef NDEBUG` before `<cassert>`, same pattern as `test-tensor-placement.cpp` |
| `ggml/src/ggml-sycl/tests/test-moe-mxfp4-dp4a.cpp` | 2 | **not a defect** — the 2 bare asserts are minor struct-size sanity checks (`sizeof(block_mxfp4)==17` etc.); the actual correctness checks (DP4A vs scalar, 5 subtests) use a `TEST_FAIL`/`return false` macro and an `all_passed` accumulator that are NDEBUG-proof and drive the real exit code |
| `tests/test-archs-exclude.cpp`, `test-archs-table.cpp`, `test-chat.cpp`, `test-rope.cpp` | 1 each | not SYCL-specific (general llama.cpp tests, out of this ticket's `tests/`+`ggml/src/ggml-sycl/tests/` SYCL-behaviour scope); not individually re-verified this pass |
| `tests/test-grammar-llguidance.cpp` | 4 | registered via `llama_build_and_test`; general (non-SYCL), not re-verified |
| `tests/test-quantize-stats.cpp` | 1 | registered via `llama_build` only (no `llama_test`/`add_test` found for it — may not even run as a test); general, not SYCL, not re-verified |
| 17 remaining files (`test-ab-validation`, `test-crossover-discovery`, `test-dense-scheduler`, `test-edge-cases`, `test-expert-cache.cpp`, `test-ggml-flash-attn-ext`, `test-kv-cache-coordinator`, `test-pinned-chunk-pool`, `test-prefetch-scheduler`, `test-q6k-56block-debug`, `test-q6k-variable-reorder`, `test-sycl-prestage-routed-experts`, `test-tensor-classification`, `test-tile-decomposition`, `test-transfer-learning`, `test-vram-pool`, `test-xmx-unified-kernel`) | 1–77 | **unregistered anywhere** (confirmed against both `CMakeLists.txt` files) — part of `llama.cpp-0igs`'s 186-file dead-source population. Several are clearly SYCL/MoE-backend tests despite lacking a `sycl-` prefix (`tile-decomposition`, `vram-pool`, `pinned-chunk-pool`, `xmx-unified-kernel`, `test-sycl-prestage-routed-experts` itself). Would need **both** registration and the same `#undef NDEBUG` fix if restored — noting this so `llama.cpp-0igs` doesn't rediscover the NDEBUG half of the problem file-by-file the way `llama.cpp-4jlv`/`c17b010af`/`d4cccba5e`/`d4608ec7e` each rediscovered the registration half in isolation |

## Four more ctest/grep traps, found during `llama.cpp-0igs` restoration batches

Extending the trap list already in this doc (`ctest -R` on zero matches,
`--output-on-failure` hiding a passing test's output, `SKIP` needing exit 77):

- **`ctest -N` prints test names, not labels — grepping it for a label
  returns 0 no matter what.** The lead's own first proof attempt for batch 1
  was `ctest -N | grep -c '0igs-batch1'`, which came back 0 and nearly read
  as "all 13 excluded." The actual selector is `-L <label>`
  (`ctest -N -L 0igs-batch1`). Same failure shape as the other two: a clean,
  wrong-question answer that looks like evidence.
- **A registration grep must carry guard context, not just the line.**
  `test-expert-cache.cpp` looked like a genuine `3c8f296fd` restoration
  candidate — `grep -c "^\s*add_executable(test-expert-cache "` returned 1 —
  but that one line sat inside an `if(FALSE) ... endif()` block with the
  comment *"DISABLED: expert-cache.hpp removed (replaced by unified cache +
  placement table)"*. It was disabled **before** the registration wipe, for a
  real architectural reason (the header it includes no longer exists —
  confirmed by `-fsyntax-only`), not a casualty of it. A line-level match is
  not the same claim as "this target builds when CMake processes this file."
- **`-LE` is an unanchored regex over LABELS too, not just `-E` over test
  names.** 9 of batch 1's 13 restored tests carried LABELS containing the
  bare words `cache` or `mem-handle`, colliding with CLAUDE.md's
  throttled-sweep denylist (`-LE 'residency|mem-handle|cache'`) despite being
  confirmed host-only and deserving none of the GPU-memory hazard the
  denylist exists for. The first fix (`32a70dcf3`) renamed them to
  `cache-hostonly` / `mem-handle-hostonly` and was verified by checking that
  no label *equalled* a denylisted word — the right property, checked with
  the wrong operator. `ctest -LE` matches by regex/substring exactly the way
  `-R`/`-E` already do over test names (this doc's own earlier trap: `-E
  "test-llama-archs"` also excludes `test-llama-archs-table`), so `cache`
  still matched inside `cache-hostonly` and 11 of 41 restored tests stayed
  silently excluded. Nobody had written that `-L`/`-LE` share that semantic
  with `-R`/`-E`, so the fix repeated the exact defect class it was fixing.
  Corrected in `b7555cc73` by dropping the hazard word entirely rather than
  suffixing it (bare `hostonly` + subsystem word, no shared substring with
  the denylist at all), and verified with `re.search` over every label
  string instead of equality — 0 of 41 collide. `llama.cpp-0igs` itself
  documents the underlying failure mode (`impl-5exz`: 5 of 9 un-guarded
  tests similarly affected); batch 1's original rate was worse, 9 of 13.
- **`-fsyntax-only` proves compilation, not linking.** `test-onednn-woq.cpp`
  compiled clean but failed to link (`undefined reference to
  dnnl_primitive_destroy`, `DSO missing from command line`): it includes two
  header-only files (`ggml-sycl/gemm.hpp`, `ggml-sycl/onednn-woq.hpp`) that
  call the oneDNN C++ API inline, so those calls compile directly into the
  test's own object file and need `DNNL::dnnl` linked into the test binary
  itself — linking `ggml-sycl` (a `PRIVATE` link, so its own oneDNN link
  doesn't propagate) isn't enough. `-fsyntax-only` stops before the link
  step by design, so it cannot catch this class of failure; only an actual
  build (or a link-only step) does. Fixed in `b7555cc73` with the same
  `if (GGML_SYCL_DNNL)` / `DNNL::dnnl` guard `test-sycl-onednn-mxfp4-
  feasibility` already used — copied from an existing oneDNN-linking target
  rather than guessed.

## What this pass did NOT cover

Being explicit rather than implying completeness: this pass verified the two
named instances, the tensor-usage-family bonus fixes, ran the mechanical
NDEBUG scan across the whole `tests/`+`ggml/src/ggml-sycl/tests/` SYCL
population, and header-checked (not mutation-tested) the 19-test
residency/MoE `foreach` loop, spot-reading 2 of those 19 in full. It did
**not** individually mutation-test:

- The other 17 of the 19 residency/MoE-loop tests (header-checked only, see
  above — real production headers confirmed, but that only rules out the
  instance-1 mock defect class, nothing more).
- The remaining ~65–70 currently-*registered* SYCL C++ tests outside that
  loop (the guarded XMX/ESIMD suite behind `GGML_SYCL_BUILD_XMX_TESTS`, the
  kernel-profiler/timeline/vtune/zebin parser `.cpp` gates, etc.).
- The ~90 registered Python/shell parser and doc-assertion gates. `llama.cpp-0igs`'s
  own report already found and fixed several of these (a vertical-alignment
  break, a stale wording match, two gates asserting on files that never
  existed in this repo) using a shadow-root mutation harness with a
  no-op-mutation negative control — that harness is worth reusing here rather
  than re-deriving one.

`test-kv-slice-sizing.cpp` remains the model worth copying (87 checks; a
one-word `assign`→`resize` mutation fires exactly 8 and leaves 79 green —
specificity, not just sensitivity). A follow-up pass should work through the
remaining population the same way: read the assertions, trace them against
the real function being called (not a local reimplementation, and not merely
"a real header is included"), and where a build is available, run the
described mutation rather than reasoning through it.
