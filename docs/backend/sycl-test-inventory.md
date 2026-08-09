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

> ⚠️ **This document is cumulative and parts of it are HISTORY.** A remediation
> pass on 2026-08-08 deleted ten tests, repaired or reclassified six more, and
> changed the population from 127 to 117 — see **REMEDIATION PASS** below, which
> is the current state. Sections above it (the `llama.cpp-0igs` restoration
> audit, Task 16/17d/19) are point-in-time records and still name files and
> registrations that no longer exist: `test-onednn-fallback`,
> `test-xmx-kernel-config`, `test-xmx-quant-loaders`, `test-xmx-unified-kernel`,
> `test-kernel-dispatch`, `test-sycl-pp-moe-scratch-lifecycle` and the four
> `test-sycl-moe-*-policy`/`-metadata`/`-bridge` mocks. They are left as written
> rather than back-edited, because rewriting an executed audit's record makes it
> unciteable. **Read the dispositions table before acting on any row above.**

## ⚠️ TOP-LINE ANSWER: the `llama.cpp-0igs` merge-blocker question

**At the original inventory baseline on 2026-08-02 (`333a8d7b2`), 64
still-unregistered test sources covered code this branch changed.** At the
current Task 4a audit point (`c27ba2292`), those 64 source rows split into 41
active registrations and 23 inactive registrations. Redirected here by the
lead: exhaustively mutation-verifying the whole
remaining population (~65–70 more registered C++ tests, ~90 Python/shell
gates) has falling marginal value and is not what gates *this* merge — the
question that matters is how many of `llama.cpp-0igs`'s **approximately 147
pending C++ source rows** are actually merge-relevant.

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
   commit) — this matters because the lead's initial-scope rule is explicit:
   **a file registered at `3c8f296fd` belongs in the `0igs` restoration-review
   scope even if currently vacuous; a never-registered file begins as a delete
   candidate.** This is scope inclusion, not a mandatory restore disposition:
   Task 16 may decline a vacuous current source. **64 of 65 existed at
   `3c8f296fd`** (genuine wipe casualties). The one
   exception, `tests/bench-sycl-fattn-gptoss.cpp`, is new to this branch and
   named like the other `bench-*`/`bench-dnnl-ops.cpp` files that were never
   meant to be ctest targets — almost certainly not a real restoration
   candidate, though not individually confirmed.

**So: 64 of approximately 147 pending C++ source rows.** The exact 147
historical `add_test()` calls are supporting endpoint evidence, not a
one-to-one denominator equivalent to source rows. Full sorted list (all under
`tests/`, all pre-existing at `3c8f296fd`): `mini-context-prototype`,
`test-cold-start`,
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
  list, not a confirmed-good list. Pre-wipe registration therefore included it
  in the initial restoration-review scope; it did not compel restoration.
  Task 16 may decline a vacuous or manual-only current source, as it does
  below.
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

**Completeness check:** 41 GREEN + 22 RED + 1 guard-hidden RED = 64/64 rows. Historical shape within the 22: eight source rows with active ctest registrations under `GGML_SYCL` (11 CTest names, because `test-mmvq-q8-0-streaming-bench.cpp` registered four), five `GGML_SYCL` targets explicitly not wired to ctest, one commented-out block, one `GGML_SYCL && FALSE` registration, and seven names absent entirely. The separate exception is another `GGML_SYCL && FALSE` registration. No status in this section asserts safety, usefulness, or a future disposition.

## Task 4b: execution-hazard classification (static audit)

This classification is an execution characteristic, **not** Task 4c's final restore/manual/delete disposition. In particular, `never-test` below means that the current source is obsolete or a self-contained debug/mock rather than an executable production regression test; Task 4c still owns what happens to the file. `host-only` means the source neither initializes a SYCL backend/queue nor allocates/submits device work. Linking SYCL code does not alone change that class when the tested path is CPU-side policy, bookkeeping, layout, or compile-time behavior. `GPU serial` means SYCL backend/queue initialization, device allocation, submission, a backend graph compute, or an indirect call that initializes device-manager state. **Normatively, every `GPU serial` row is lead-only and must execute serially, one at a time.** This is a required scheduling contract, not a claim that live CTest metadata currently enforces it. `manual` identifies an opt-in benchmark, standalone hardware experiment, or special-instrumentation run. `parser` is reserved for tests whose subject is parsing text/artifacts; none of these 64 sources has that execution shape.

Classes are mutually exclusive. Apply this precedence when a source has more than one characteristic: **model-loading > manual > never-test > GPU serial > host-only > parser**. The higher operational constraint wins: model hazards dominate all other execution details; an intentionally manual benchmark remains manual even when it uses GPUs; a current obsolete/mock source remains never-test even if it constructs SYCL objects; device reachability dominates host-only; and parser is the final text/artifact-only fallback. This makes the aggregate counts deterministic.

**Evidence pin:** Task 4a provenance is pinned at merge `e015e1e0c`. Task 4b source and live-CMake evidence was audited through branch tip `284a78bee` immediately before this documentation-only coherence correction; because the correction changes only this inventory, the audited source/CMake content is identical at the resulting branch tip.

At that pinned live snapshot, **31 Task 4a GREEN rows are classified `GPU serial`, and all 31 lack CTest `RUN_SERIAL` enforcement**. Their live registration does not waive the lead-only/one-at-a-time contract. Existing `hostonly` labels are also stale where noted below. Task 17 owns label and scheduling remediation; Task 4b records the hazard and does not edit CMake.

Evidence keys used in the table:

- **H:** a literal static scan of the 9 host-only sources found zero occurrences of the direct/runtime markers `ggml_backend_sycl_init`, `ggml_sycl_get_device`, `sycl::queue`, `ggml_backend_graph_compute`, `ggml_backend_sycl_buffer_type`, `ggml_backend_sycl_kv_buffer_type`, `gpu_selector`, `malloc_device`, `parallel_for`, `single_task`, or `.submit(` **and** the indirect device/cache APIs `ggml_backend_sycl_get_device_memory`, `get_unified_cache_for_device`, `ensure_cached_alloc`, or `ggml_backend_sycl_get_weight_cache_key`. The indirect list is load-bearing: the two unified-memory sources do not spell a queue or kernel launch but require a real device cache and allocate against its VRAM budget; likewise, the three weight-key sources enter the SYCL registry/device path under default evictability. This deliberately uses a stricter execution test than restoration commits `f87b6f410`/`d27a6fe19`, whose no-direct-kernel filter also admitted sources that initialize a backend, execute a graph, or reach the device through cache/key APIs.
- **G:** the cited source line directly initializes a SYCL backend/queue, obtains a device buffer type, allocates/submits device work, executes a backend graph, or enters one of those device paths indirectly. For weight keys, `ggml_backend_sycl_get_weight_cache_key` at `ggml-sycl.cpp:9563–9589` calls `ggml_backend_sycl_reg` for a default-evictable host tensor; registry initialization at `ggml-sycl.cpp:94710–94739` enters `ggml_sycl_info`, enumerates devices, sets each device, and obtains its SYCL device object/manager state.
- **M:** the cited source header makes the run opt-in/manual, a benchmark, a standalone hardware experiment, or dependent on a special instrumentation build.
- **N:** the cited source either reimplements the purported production helper locally or includes the removed `expert-cache.hpp`; it is not a production-path test in its current form.
- **L:** model/model-file loading evidence is expanded in the dedicated five-row hazard table below.

| source row | hazard class | static source evidence |
|---|---|---|
| `tests/test-cold-start.cpp` | **host-only** | H |
| `tests/test-dmmv-q4-0-coalesced.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 308; `ggml_backend_sycl_init` at line 739 |
| `tests/test-dmmv-q6k-coalesced.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 81; `ggml_backend_sycl_init` at line 171 |
| `tests/test-fattn-thread-local.cpp` | **GPU serial** | G: lines 27–46 construct SYCL queues; lines 49–78 touch and clean up thread-local device buffers |
| `tests/test-ggml-sycl-soa.cpp` | **GPU serial** | G: backend graph computes begin at lines 115/315; SYCL backend initialization begins at lines 174/244 |
| `tests/test-layout-bytes.cpp` | **GPU serial** | G: lines 32–38 pin a Level Zero selector and initialize the SYCL backend |
| `tests/test-mmq-q6k-gpu.cpp` | **GPU serial** | G: `ggml_backend_sycl_init` at line 89; `ggml_backend_graph_compute` at line 384 |
| `tests/test-moe-mini-graph.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 234; `ggml_backend_sycl_init` at line 269 |
| `tests/test-moe-mul-mat-id.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 142; `ggml_backend_sycl_init` at line 244 |
| `tests/test-moe-mul-mat-id-q4q8.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 175; `ggml_backend_sycl_init` at line 285 |
| `tests/test-mul-mat-host-streaming.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 119; `ggml_backend_sycl_init` at line 144 |
| `tests/test-onednn-fallback.cpp` | **host-only** | H |
| `tests/test-onednn-woq.cpp` | **GPU serial** | G: line 193 initializes the SYCL backend and lines 200–212 require its SYCL/oneDNN queue |
| `tests/test-q6k-dispatch.cpp` | **GPU serial** | G: SYCL backend initialization at lines 95/234; graph compute at lines 177/306 |
| `tests/test-q8-0-layout-cache-path.cpp` | **GPU serial** | G: `ggml_backend_sycl_init` at line 77; `ggml_backend_graph_compute` at line 168 |
| `tests/test-q8-0-layout-cache-path-mmvq.cpp` | **GPU serial** | G: `ggml_backend_sycl_init` at line 80; `ggml_backend_graph_compute` at line 171 |
| `tests/test-sycl-cpu-dispatch.cpp` | **host-only** | H |
| `tests/test-sycl-fattn-onednn-materialization.cpp` | **GPU serial** | G: line 132 obtains device 0's default queue for the materialization case |
| `tests/test-sycl-fattn-xmx-policy.cpp` | **host-only** | H |
| `tests/test-sycl-kernel-selection.cpp` | **GPU serial** | G: lines 49/103 initialize the SYCL backend for selection cases |
| `tests/test-sycl-kv-planned-device-materialization.cpp` | **GPU serial** | G: lines 28–37 pin two Level Zero devices; line 102 obtains the device KV buffer type and line 103 allocates it |
| `tests/test-sycl-moe-expert-parallelism.cpp` | **host-only** | H; source lines 3–8 explicitly call these CPU-side data-structure tests with no GPU dependency |
| `tests/test-sycl-moe-handle-resolution.cpp` | **GPU serial** | G: queue-driven resolution cases begin at lines 50/94/123/184 and main constructs a queue at lines 376–381 |
| `tests/test-sycl-moe-identity-hash.cpp` | **GPU serial** | G: lines 119/178/253/324/385 call `ggml_backend_sycl_get_weight_cache_key`, entering the default-evictability SYCL registry/device initialization chain documented above |
| `tests/test-sycl-moe-q8-scratch.cpp` | **GPU serial** | G: lines 73–78 pin Level Zero and initialize the SYCL backend before device-VRAM scratch allocation |
| `tests/test-sycl-onednn-packed-cache.cpp` | **GPU serial** | G: line 18 initializes the SYCL backend; line 152 obtains its device queue |
| `tests/test-sycl-orchestrator.cpp` | **GPU serial** | G: graph computes at lines 271/342; SYCL backend initialization at line 373 |
| `tests/test-sycl-prestage-routed-experts.cpp` | **never-test** | N: the source locally mimics routed-expert prestaging, includes no production header or production path, and relies on NDEBUG-vulnerable bare `assert()` checks |
| `tests/test-sycl-unified-cache.cpp` | **GPU serial** | G: lines 815–820 require device-0 memory; lines 90–180 obtain its unified cache and allocate weight/expert entries into VRAM with `ensure_cached_alloc` |
| `tests/test-sycl-unified-memory-e2e.cpp` | **GPU serial** | G: lines 870–875 require device-0 memory; lines 94–201 obtain its unified cache and allocate attention/KV/expert entries against VRAM (the model workload is simulated, the allocations are not) |
| `tests/test-sycl-weight-key-stability.cpp` | **GPU serial** | G: lines 67/85 call `ggml_backend_sycl_get_weight_cache_key`, entering the default-evictability SYCL registry/device initialization chain documented above |
| `tests/test-sycl-weight-key-uniqueness.cpp` | **GPU serial** | G: lines 109/133/165–166 call `ggml_backend_sycl_get_weight_cache_key`, entering the default-evictability SYCL registry/device initialization chain documented above |
| `tests/test-sycl-xmx-unified-correctness.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 204; `ggml_backend_sycl_init` at line 257 |
| `tests/test-tensor-classification.cpp` | **host-only** | H |
| `tests/test-tiered-dispatch.cpp` | **GPU serial** | G: the dispatch cases repeatedly initialize the SYCL backend, beginning at lines 50/96/142/186 |
| `tests/test-unified-cache-concurrent.cpp` | **GPU serial** | G: queue-driven cache stress cases begin at lines 31/86/144 and main constructs the queue at line 222 |
| `tests/test-unified-cache-integrity.cpp` | **GPU serial** | G: line 98 obtains the device queue and line 157 initializes the SYCL backend |
| `tests/test-xmx-host-streaming.cpp` | **GPU serial** | G: `ggml_backend_graph_compute` at line 133; `ggml_backend_sycl_init` at line 218 |
| `tests/test-xmx-kernel-config.cpp` | **host-only** | H |
| `tests/test-xmx-quant-loaders.cpp` | **host-only** | H |
| `tests/test-xmx-unified-kernel.cpp` | **host-only** | H |
| `tests/mini-context-prototype.cpp` | **model-loading** | L1 |
| `tests/test-cpu-gpu-soa-interaction.cpp` | **GPU serial** | G: lines 176, 243, 308, 408, and 501 select a GPU queue; lines 71–166 define direct SYCL reorder/DMMV work |
| `tests/test-expert-routing-roundtrip.cpp` | **manual** | M: lines 7–12 give standalone `icpx` build and two-GPU run instructions; lines 29–54 require two GPUs/queues |
| `tests/test-mmvq-q8-0-streaming-bench.cpp` | **manual** | M: lines 1–19 say opt-in benchmark and define benchmark-only environment knobs/modes |
| `tests/test-moe-expert-placement.cpp` | **manual** | M: lines 1–19 identify a micro-benchmark with standalone manual build/run instructions |
| `tests/test-mxfp4-xmx-tiled.cpp` | **GPU serial** | G: lines 1–8 require GPU-vs-CPU conversion on one SYCL device; line 119 selects the GPU queue |
| `tests/test-pinned-chunk-pool.cpp` | **GPU serial** | G: lines 17–24 construct a SYCL queue and a pool whose first case allocates 1 GiB (the file repeats queue-backed allocation cases) |
| `tests/test-planner-canary-cpy-visibility.cpp` | **model-loading** | L2 |
| `tests/test-planner-canary-direct-load.cpp` | **model-loading** | L3 |
| `tests/test-planner-canary-pp-tg-union.cpp` | **model-loading** | L4 |
| `tests/test-planner-canary-skeleton-determinism.cpp` | **model-loading** | L5 |
| `tests/test-q6k-56block-debug.cpp` | **never-test** | N: lines 1–3 call it a debug test; lines 11–24 locally redefine the block and tile helper instead of calling production code |
| `tests/test-q6k-layout-debug.cpp` | **never-test** | N: lines 1–25 locally redefine the Q6_K block and “same as common.hpp” tile helpers |
| `tests/test-q6k-reorder-dispatch.cpp` | **GPU serial** | G: lines 245/293 initialize the SYCL backend and lines 479/613 select GPU queues |
| `tests/test-q6k-variable-reorder.cpp` | **never-test** | N: lines 7–24 locally redefine the Q6_K block and inline the helper under test |
| `tests/test-sycl-expert-cache-bandwidth.cpp` | **manual** | M: lines 1–14 identify a two-device bandwidth/latency microbenchmark and give its manual selector |
| `tests/test-sycl-expert-prefetch.cpp` | **GPU serial** | G: lines 1–6 require real asynchronous H2D DMA through a SYCL device; lines 27–30 select a GPU queue |
| `tests/test-sycl-fattn-onednn-descriptors.cpp` | **GPU serial** | G: lines 127–151 allocate/copy device data through the oneDNN/SYCL stream |
| `tests/test-sycl-race-conditions.cpp` | **manual** | M: lines 7–14 require a separate ThreadSanitizer configure/build to observe the race property |
| `tests/test-sycl-set-rows-owner-routing.cpp` | **GPU serial** | G: line 284 obtains the device-0 queue and the following cases allocate/execute owner-routed SET_ROWS work |
| `tests/test-tile-decomposition.cpp` | **never-test** | N: lines 5–24 explicitly inline/reimplement the helpers purportedly under test |
| `tests/test-unified-dispatch-integration.cpp` | **GPU serial** | G: lines 128, 203, 331, 447, 573, and 714 construct SYCL queues for dispatch cases |
| `tests/test-expert-cache.cpp` | **never-test** | N: line 11 includes removed `expert-cache.hpp`; Task 4a records the pre-wipe block was already under `FALSE` with “removed (replaced by unified cache + placement table)” |

### The five model-loading hazards — no ordinary parallel CTest registration

These are the **exact five** model/model-file-loading hazards. They **must not enter ordinary parallel CTest registration**. Any eventual execution remains lead-only and serial, with the repository's model-loading safeguards; Task 4c decides the final disposition and Task 4b does not register them.

| evidence key / source | exact source evidence |
|---|---|
| **L1** `tests/mini-context-prototype.cpp` | Lines 12–16 define two full-weight worker loads plus a metadata-only load; line 169 calls `llama_model_load_from_file`, line 183 calls `llama_init_from_model`, and line 231 `execl`s each worker. |
| **L2** `tests/test-planner-canary-cpy-visibility.cpp` | Lines 22–24 say one process loads Mistral 7B and decodes repeatedly; lines 391–392 log and call `llama_model_load_from_file`, and line 411 calls `llama_init_from_model`. |
| **L3** `tests/test-planner-canary-direct-load.cpp` | This deliberately bypasses llama's loader but still consumes a real model file: lines 38–45 require the Mistral fixture, lines 69–78 `mmap` it and take source bytes from that mapping, and lines 83–158 initialize the SYCL backend and transfer/verify those bytes in a device tensor. Default fixture paths are in `tests/test-planner-canary-common.hpp:105–112`. |
| **L4** `tests/test-planner-canary-pp-tg-union.cpp` | Lines 16–22 specify fork/exec workers that each load a model/context; line 175 calls `llama_model_load_from_file`, line 204 calls `llama_new_context_with_model`, and lines 311/381–382 run workers across available model fixtures/shapes. |
| **L5** `tests/test-planner-canary-skeleton-determinism.cpp` | Lines 3–10 delegate to the mini-context proof with `real-A`/`real-B` full weight loads; lines 46–55 fork/exec that binary, and lines 82–103 locate it and iterate available model fixtures. This inherits L1's full-weight loads. |

**Task 4b counts:** 9 host-only + 39 GPU serial + 5 model-loading + 5 manual + 6 never-test + 0 parser = **64/64**. This independently reconciles with Task 4a's 41 GREEN + 23 RED = 64 source rows: the hazard and provenance axes are complete but intentionally do not imply one another.

**Static completeness check:** the classification table contains each Task 4a source row exactly once (64 unique names; no missing names and no extras), and the six class counts sum to 64. The dedicated hazard table contains exactly five unique names, all five are classified `model-loading` in the complete table, and no other row has that class. No CMake or disposition change is made here.


## Task 4c: restoration dispositions (actionability audit, not registration acceptance)

This table disposes the exact 64-row changed-surface population identified for
`llama.cpp-0igs`; it does **not** enlarge or finally accept that scope. The Task
4 filter intersected the pending C++ source population with this branch's
changed surface and found 65 topical hits, exactly 64 of which existed before
the wipe. Therefore this audit covers **64 of approximately 147 pending C++
source rows** (a topical floor), with the other roughly 83 assigned to `lead`
under `llama.cpp-0igs` as post-merge debt. Separately, the exact endpoint
evidence is 147 historical `add_test()` calls at `3c8f296fd` versus 6 at the
then-HEAD, a loss of 141 registrations. Those call counts support the wipe
finding; they are not asserted to be a one-to-one denominator equivalent to
source rows. This does not claim that 64 registrations are accepted or that the
83 are irrelevant forever.

The disposition is a source-level recommendation. **Task 16
(`llama.cpp-o2hp`) is the authority that records the exact accepted and
declined candidates**. Downstream, Task 17 owns registration metadata,
`llama.cpp-8u22` owns mutation proofs, and Task 19 owns only the final clean
build and accepted-set runtime. At the Task 4c review point, the recommendation in
`llama.cpp-awcp` to retain the 64-row scope had not yet been explicitly
owner-accepted; the Task 16 section below now records that decision. Likewise,
a currently live row is accepted only if it appears in Task 16's accepted
set.

For all `GPU serial` restore candidates, including the 31 already live,
execution remains lead-only and one-at-a-time; Task 17 must repair missing
`RUN_SERIAL`/labels. The five `model-loading` rows are manual-only and retain
the stronger rule: no ordinary parallel CTest registration, and any eventual
run is lead-only, serial, once-only, and uses the repository model-loading
safeguards.

Task 14/15 source outcomes are incorporated rather than treated as runtime
proof: `llama.cpp-x9r0`, `-1qij`, `-xz8x`, `-zmvu`, `-fehs`, and `-xvdd` are
closed and merged, making their six rows source-ready restore candidates.
Their tracker closures explicitly defer CTest metadata to Task 17, mutation
proofs to `llama.cpp-8u22`, and final build/runtime verification to Task 19.
In particular, the opt-in streaming benchmark is a
restore candidate because it historically exposed four CTest modes and Task
14 supplied the missing skip/failure contract; that does not turn other
standalone benchmarks into ordinary tests.

The action keys below record the historical Task 4c recommendation and keep
that review point compact. They are referential, not a normative downstream
handoff: the Task 16 accepted/declined sets and contracts below are the sole
normative handoff for all later tasks.

- **RC-L (live restore candidate):** `llama.cpp-o2hp` explicitly accepts or
  declines the row. If accepted, preserve its live target and process it through
  Task 17's exact sequential chain: `llama.cpp-m2ke` (17a target/link topology)
  → `llama.cpp-vohe` (17b oneDNN guards) → `llama.cpp-cidw` (17c skip code,
  safe labels, and serial properties) → `llama.cpp-kdfh` (17d final
  registration/selection audit). Then `llama.cpp-8kyi` (Task 19) performs the
  lead-only serial runtime acceptance.
- **RC-I (inactive restore candidate):** `llama.cpp-o2hp` explicitly accepts or
  declines the row. If accepted, the same
  `llama.cpp-m2ke` → `llama.cpp-vohe` → `llama.cpp-cidw` → `llama.cpp-kdfh`
  chain registers and audits it, then `llama.cpp-8kyi` performs lead-only
  serial runtime acceptance.
- **M-MODEL (manual-only model load):** `lead`, assigned through
  `llama.cpp-0igs`, keeps the row out of ordinary CTest and documents one
  lead-only, serial, safeguarded model run; `llama.cpp-o2hp` records the
  ordinary-registration decline.
- **M-OPTIN (manual-only diagnostic/benchmark):** `lead`, assigned through
  `llama.cpp-0igs`, retains an opt-in manual procedure and does not add ordinary
  CTest registration; `llama.cpp-o2hp` records the registration decline.
- **D-LOCAL (deleted/never-test local reimplementation):** historically,
  `lead` through `llama.cpp-0igs` deletes the self-contained local-helper
  reimplementation with its Task 4b per-file reason; Task 16 supersedes this
  generic routing where it names a dedicated child owner.
- **D-OBSOLETE (deleted/never-test obsolete source):** `lead`, assigned through
  `llama.cpp-0igs`, deletes the source that includes removed
  `expert-cache.hpp`, preserving that reason; `llama.cpp-o2hp` records the
  decline.
- **P-PIN (named pinned-pool task):** `lead`, assigned through
  `llama.cpp-32dg8.20`, updates the source to the canonical pinned-pool API
  while preserving allocation, reuse, capacity, and failure checks; it then
  returns to `llama.cpp-o2hp`, and the lead performs the task's build/GPU proof.

| source row | Task 4a state | Task 4b hazard | exactly one disposition | action key |
|---|---|---|---|---|
| `tests/test-cold-start.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-dmmv-q4-0-coalesced.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-dmmv-q6k-coalesced.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-fattn-thread-local.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-ggml-sycl-soa.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-layout-bytes.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-mmq-q6k-gpu.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-moe-mini-graph.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-moe-mul-mat-id.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-moe-mul-mat-id-q4q8.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-mul-mat-host-streaming.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-onednn-fallback.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-onednn-woq.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-q6k-dispatch.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-q8-0-layout-cache-path.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-q8-0-layout-cache-path-mmvq.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-cpu-dispatch.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-sycl-fattn-onednn-materialization.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-fattn-xmx-policy.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-sycl-kernel-selection.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-kv-planned-device-materialization.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-moe-expert-parallelism.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-sycl-moe-handle-resolution.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-moe-identity-hash.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-moe-q8-scratch.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-onednn-packed-cache.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-orchestrator.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-prestage-routed-experts.cpp` | GREEN/live | never-test | **deleted/never-test** | **D-LOCAL** |
| `tests/test-sycl-unified-cache.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-unified-memory-e2e.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-weight-key-stability.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-weight-key-uniqueness.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-sycl-xmx-unified-correctness.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-tensor-classification.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-tiered-dispatch.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-unified-cache-concurrent.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-unified-cache-integrity.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-xmx-host-streaming.cpp` | GREEN/live | GPU serial | **restore candidate** | **RC-L** |
| `tests/test-xmx-kernel-config.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-xmx-quant-loaders.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/test-xmx-unified-kernel.cpp` | GREEN/live | host-only | **restore candidate** | **RC-L** |
| `tests/mini-context-prototype.cpp` | RED/inactive | model-loading | **manual-only** | **M-MODEL** |
| `tests/test-cpu-gpu-soa-interaction.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-expert-routing-roundtrip.cpp` | RED/inactive | manual | **manual-only** | **M-OPTIN** |
| `tests/test-mmvq-q8-0-streaming-bench.cpp` | RED/inactive | manual | **restore candidate** | **RC-I** |
| `tests/test-moe-expert-placement.cpp` | RED/inactive | manual | **manual-only** | **M-OPTIN** |
| `tests/test-mxfp4-xmx-tiled.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-pinned-chunk-pool.cpp` | RED/inactive | GPU serial | **named tracker task: `llama.cpp-32dg8.20`** | **P-PIN** |
| `tests/test-planner-canary-cpy-visibility.cpp` | RED/inactive | model-loading | **manual-only** | **M-MODEL** |
| `tests/test-planner-canary-direct-load.cpp` | RED/inactive | model-loading | **manual-only** | **M-MODEL** |
| `tests/test-planner-canary-pp-tg-union.cpp` | RED/inactive | model-loading | **manual-only** | **M-MODEL** |
| `tests/test-planner-canary-skeleton-determinism.cpp` | RED/inactive | model-loading | **manual-only** | **M-MODEL** |
| `tests/test-q6k-56block-debug.cpp` | RED/inactive | never-test | **deleted/never-test** | **D-LOCAL** |
| `tests/test-q6k-layout-debug.cpp` | RED/inactive | never-test | **deleted/never-test** | **D-LOCAL** |
| `tests/test-q6k-reorder-dispatch.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-q6k-variable-reorder.cpp` | RED/inactive | never-test | **deleted/never-test** | **D-LOCAL** |
| `tests/test-sycl-expert-cache-bandwidth.cpp` | RED/inactive | manual | **manual-only** | **M-OPTIN** |
| `tests/test-sycl-expert-prefetch.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-sycl-fattn-onednn-descriptors.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-sycl-race-conditions.cpp` | RED/inactive | manual | **manual-only** | **M-OPTIN** |
| `tests/test-sycl-set-rows-owner-routing.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-tile-decomposition.cpp` | RED/inactive | never-test | **deleted/never-test** | **D-LOCAL** |
| `tests/test-unified-dispatch-integration.cpp` | RED/inactive | GPU serial | **restore candidate** | **RC-I** |
| `tests/test-expert-cache.cpp` | RED/inactive | never-test | **deleted/never-test** | **D-OBSOLETE** |

**Disposition counts:** 48 restore candidates + 9 manual-only + 6
deleted/never-test + 1 named tracker task = **64/64 unique rows**. The action
keys independently reconcile as 40 RC-L + 8 RC-I + 5 M-MODEL + 4 M-OPTIN + 5
D-LOCAL + 1 D-OBSOLETE + 1 P-PIN = **64/64**. “Live” is provenance, not Task
16 acceptance. This legend preserves the Task 4c review record only; Task 16
below resolves the then-open `llama.cpp-awcp` owner-policy decision and is the
single source for downstream ownership and sequencing.

## Task 16: owner-accepted restoration set

The owner accepts the 64-row changed-surface audit scope and leaves the roughly
83 other pending C++ source rows with `lead` under `llama.cpp-0igs` as
post-merge debt. Acceptance here means **accepted for source-faithful,
non-vacuous registration work**, not runtime acceptance: Task 17 must satisfy
the conditions below before a row is enabled; `llama.cpp-8u22` must supply the
lead-only mutation proofs; and Task 19 (`llama.cpp-8kyi`) owns only the final
clean build and accepted-set runtime. No Task 14/15 source review is
represented as a green build, CTest, GPU, or mutation result.

**Normative registration contract.** `CMD(name)` means exactly
`ctest --test-dir build -R '^name$' --output-on-failure`. `CHAIN` means the
sequential owner/action chain `lead`: `llama.cpp-m2ke` (target/link topology) →
`llama.cpp-vohe` (oneDNN guards, including a no-op pass for non-oneDNN rows) →
`llama.cpp-cidw` (skip code, safe labels, and `RUN_SERIAL`) →
`llama.cpp-kdfh` (final registration/selection audit) → `llama.cpp-8u22`
(lead-only mutation-proof gate) → `llama.cpp-8kyi` (Task 19 final clean build
and accepted-set runtime only). `NV-77` is the exact accepted skip policy: an
unavailable required capability exits 77 and the CTest has `SKIP_RETURN_CODE
77`; a failed property exits nonzero; success requires at least one claimed
assertion/path (and, for a GPU row, the claimed device work) to execute. A
current success-on-unavailable or all-subcases-skipped path must be repaired by
`llama.cpp-cidw` before enablement. `RUN_SERIAL TRUE` is required for every
`GPU serial` row; all GPU execution in `8u22` and `8kyi` is lead-only and
one-at-a-time. Host-only rows do not acquire `RUN_SERIAL` merely by linking
SYCL.

**Normative declined-row contract and exact tracker order.** The prestage
D-LOCAL source deletion is owned by `llama.cpp-3ygx`. After that child,
`llama.cpp-m2ke` removes the row's active target, `add_test`, and test
properties; the Task 17 sequence continues through `llama.cpp-vohe` and
`llama.cpp-cidw`, and `llama.cpp-kdfh` verifies that no declined Task 16 row
remains registered. The dependency order is exactly `llama.cpp-3ygx` →
`llama.cpp-m2ke` → `llama.cpp-vohe` → `llama.cpp-cidw` → `llama.cpp-kdfh` →
`llama.cpp-8u22` → `llama.cpp-8kyi`; separately, `llama.cpp-0igs` depends on
`llama.cpp-3ygx` and retains the other declined model/manual/deletion actions
assigned to `lead`. Neither `8u22` nor `8kyi` takes ownership of declined-row
manual procedures.

### Accepted — 48/64

Every row below is accepted with owner/action `CHAIN` and skip policy `NV-77`.
Target is the exact CMake target to preserve or restore; command is the exact
lead invocation after Task 17 registration. The five source rows with exit-77
notes already have that source branch; Task 17 must still add/audit CTest's
matching property. `source-ready` on the two Task 15 rows means only
that the reviewed production-path source was merged. All other rows require
Task 17c's explicit soft-skip/non-vacuity audit and repair where necessary.

| source | target | exact test command | hazard | skip / acceptance condition | owner / Task 17 action |
|---|---|---|---|---|---|
| `tests/test-cold-start.cpp` | `test-cold-start` | `CMD(test-cold-start)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-dmmv-q4-0-coalesced.cpp` | `test-dmmv-q4-0-coalesced` | `CMD(test-dmmv-q4-0-coalesced)` | GPU serial | NV-77; repair soft subcase skips | `lead` / `CHAIN` |
| `tests/test-dmmv-q6k-coalesced.cpp` | `test-dmmv-q6k-coalesced` | `CMD(test-dmmv-q6k-coalesced)` | GPU serial | NV-77; repair soft subcase skips | `lead` / `CHAIN` |
| `tests/test-fattn-thread-local.cpp` | `test-fattn-thread-local` | `CMD(test-fattn-thread-local)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-ggml-sycl-soa.cpp` | `test-ggml-sycl-soa` | `CMD(test-ggml-sycl-soa)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-layout-bytes.cpp` | `test-layout-bytes` | `CMD(test-layout-bytes)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-mmq-q6k-gpu.cpp` | `test-mmq-q6k-gpu` | `CMD(test-mmq-q6k-gpu)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-moe-mini-graph.cpp` | `test-moe-mini-graph` | `CMD(test-moe-mini-graph)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-moe-mul-mat-id.cpp` | `test-moe-mul-mat-id` | `CMD(test-moe-mul-mat-id)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-moe-mul-mat-id-q4q8.cpp` | `test-moe-mul-mat-id-q4q8` | `CMD(test-moe-mul-mat-id-q4q8)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-mul-mat-host-streaming.cpp` | `test-mul-mat-host-streaming` | `CMD(test-mul-mat-host-streaming)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-onednn-fallback.cpp` | `test-onednn-fallback` | `CMD(test-onednn-fallback)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-onednn-woq.cpp` | `test-onednn-woq` | `CMD(test-onednn-woq)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-q6k-dispatch.cpp` | `test-q6k-dispatch` | `CMD(test-q6k-dispatch)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-q8-0-layout-cache-path.cpp` | `test-q8-0-layout-cache-path` | `CMD(test-q8-0-layout-cache-path)` | GPU serial | NV-77 | `lead` / `CHAIN` |
| `tests/test-q8-0-layout-cache-path-mmvq.cpp` | `test-q8-0-layout-cache-path-mmvq` | `CMD(test-q8-0-layout-cache-path-mmvq)` | GPU serial | NV-77 | `lead` / `CHAIN` |
| `tests/test-sycl-cpu-dispatch.cpp` | `test-sycl-cpu-dispatch` | `CMD(test-sycl-cpu-dispatch)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-sycl-fattn-onednn-materialization.cpp` | `test-sycl-fattn-onednn-materialization` | `CMD(test-sycl-fattn-onednn-materialization)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-fattn-xmx-policy.cpp` | `test-sycl-fattn-xmx-policy` | `CMD(test-sycl-fattn-xmx-policy)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-sycl-kernel-selection.cpp` | `test-sycl-kernel-selection` | `CMD(test-sycl-kernel-selection)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-kv-planned-device-materialization.cpp` | `test-sycl-kv-planned-device-materialization` | `CMD(test-sycl-kv-planned-device-materialization)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-moe-expert-parallelism.cpp` | `test-sycl-moe-expert-parallelism` | `CMD(test-sycl-moe-expert-parallelism)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-sycl-moe-handle-resolution.cpp` | `test-sycl-moe-handle-resolution` | `CMD(test-sycl-moe-handle-resolution)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-moe-identity-hash.cpp` | `test-sycl-moe-identity-hash` | `CMD(test-sycl-moe-identity-hash)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-moe-q8-scratch.cpp` | `test-sycl-moe-q8-scratch` | `CMD(test-sycl-moe-q8-scratch)` | GPU serial | NV-77 | `lead` / `CHAIN` |
| `tests/test-sycl-onednn-packed-cache.cpp` | `test-sycl-onednn-packed-cache` | `CMD(test-sycl-onednn-packed-cache)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-orchestrator.cpp` | `test-sycl-orchestrator` | `CMD(test-sycl-orchestrator)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-unified-cache.cpp` | `test-sycl-unified-cache` | `CMD(test-sycl-unified-cache)` | GPU serial | NV-77; repair all-subcases-skipped success | `lead` / `CHAIN` |
| `tests/test-sycl-unified-memory-e2e.cpp` | `test-sycl-unified-memory-e2e` | `CMD(test-sycl-unified-memory-e2e)` | GPU serial | NV-77; repair all-subcases-skipped success | `lead` / `CHAIN` |
| `tests/test-sycl-weight-key-stability.cpp` | `test-sycl-weight-key-stability` | `CMD(test-sycl-weight-key-stability)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-weight-key-uniqueness.cpp` | `test-sycl-weight-key-uniqueness` | `CMD(test-sycl-weight-key-uniqueness)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-sycl-xmx-unified-correctness.cpp` | `test-sycl-xmx-unified-correctness` | `CMD(test-sycl-xmx-unified-correctness)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-tensor-classification.cpp` | `test-tensor-classification` | `CMD(test-tensor-classification)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-tiered-dispatch.cpp` | `test-tiered-dispatch` | `CMD(test-tiered-dispatch)` | GPU serial | NV-77; repair all-subcases-skipped success | `lead` / `CHAIN` |
| `tests/test-unified-cache-concurrent.cpp` | `test-unified-cache-concurrent` | `CMD(test-unified-cache-concurrent)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-unified-cache-integrity.cpp` | `test-unified-cache-integrity` | `CMD(test-unified-cache-integrity)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-xmx-host-streaming.cpp` | `test-xmx-host-streaming` | `CMD(test-xmx-host-streaming)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-xmx-kernel-config.cpp` | `test-xmx-kernel-config` | `CMD(test-xmx-kernel-config)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-xmx-quant-loaders.cpp` | `test-xmx-quant-loaders` | `CMD(test-xmx-quant-loaders)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-xmx-unified-kernel.cpp` | `test-xmx-unified-kernel` | `CMD(test-xmx-unified-kernel)` | host-only | NV-77 | `lead` / `CHAIN` |
| `tests/test-cpu-gpu-soa-interaction.cpp` | `test-cpu-gpu-soa-interaction` | `CMD(test-cpu-gpu-soa-interaction)` | GPU serial | NV-77; Task 15a source-ready; mutation deferred to `8u22`, runtime to `8kyi` | `lead` / `CHAIN` |
| `tests/test-mmvq-q8-0-streaming-bench.cpp` | `test-mmvq-q8-0-streaming-bench` | `CMD(test-mmvq-q8-0-streaming-bench)`; `CMD(test-mmvq-q8-0-streaming-smoke)`; `CMD(test-mmq-q8-0-streaming-smoke)`; `CMD(test-mmq-q8-0-streaming-forced)` | manual (opt-in CTest modes) | NV-77; 77-ready by Task 14a; all four historical modes required | `lead` / `CHAIN` |
| `tests/test-mxfp4-xmx-tiled.cpp` | `test-mxfp4-xmx-tiled` | `CMD(test-mxfp4-xmx-tiled)` | GPU serial | NV-77; 77-ready by Task 14b | `lead` / `CHAIN` |
| `tests/test-q6k-reorder-dispatch.cpp` | `test-q6k-reorder-dispatch` | `CMD(test-q6k-reorder-dispatch)` | GPU serial | NV-77; Task 15b source-ready; mutation deferred to `8u22`, runtime to `8kyi` | `lead` / `CHAIN` |
| `tests/test-sycl-fattn-onednn-descriptors.cpp` | `test-sycl-fattn-onednn-descriptors` | `CMD(test-sycl-fattn-onednn-descriptors)` | GPU serial | NV-77; 77-ready by Task 14c | `lead` / `CHAIN` |
| `tests/test-sycl-set-rows-owner-routing.cpp` | `test-sycl-set-rows-owner-routing` | `CMD(test-sycl-set-rows-owner-routing)` | GPU serial | NV-77; repair success-on-unavailable | `lead` / `CHAIN` |
| `tests/test-unified-dispatch-integration.cpp` | `test-unified-dispatch-integration` | `CMD(test-unified-dispatch-integration)` | GPU serial | NV-77; 77-ready by Task 14d | `lead` / `CHAIN` |

Accepted reconciliation (amended 2026-08-06): **40 live + 7 inactive = 47**;
hazards are **9 host-only + 37 GPU serial + 1 manual opt-in = 47**. The count
was 48 when Task 16 was decided; `tests/test-sycl-expert-prefetch.cpp` was
subsequently RE-DECLINED and its registration dropped by `a4791b7a9`
(2026-08-05): the source includes `expert-cache.hpp`, which no longer exists —
the same obsolete-header class as the declined `test-expert-cache.cpp` row. It
now sits in the declined table below. The Task 19 gate (`llama.cpp-8kyi`,
comment c-jx40) reconciled `ctest -N` against this amended set: 50 exact CTest
names, all registered, all executed at `772798e91`. A future rewrite against
the current prefetch API needs a new Task 16 acceptance decision. The source-ready review set
is exactly Task 14a–d plus Task 15a–b: streaming, MXFP4, oneDNN descriptors,
unified dispatch, CPU/GPU SoA, and Q6K reorder. Their required lead mutation
proofs remain conditions on `llama.cpp-8u22`; the final clean build and
accepted-set runtime remain conditions on `llama.cpp-8kyi`. None is a result
claimed here.

### Declined from ordinary restoration — 16/64

These are exact declines of the **current sources** from the accepted ordinary
registration set. Each retains an owner and next action; a future rewritten
source needs a new acceptance decision.

> The 9 manual-only rows (5 `M-MODEL` + 4 `M-OPTIN`) now have their replacement
> procedure written up under **"Lead runbook: the 9 manual-only declined rows"**
> below — exact command, selector, memory sampling, and expected pass/skip shape
> per row. Read it before running any of them; three do **not** take
> `level_zero:1` and one needs its own ThreadSanitizer build tree.

| declined source(s) | count | reason / hazard | exact owner and next action |
|---|---:|---|---|
| `tests/mini-context-prototype.cpp`; `tests/test-planner-canary-cpy-visibility.cpp`; `tests/test-planner-canary-direct-load.cpp`; `tests/test-planner-canary-pp-tg-union.cpp`; `tests/test-planner-canary-skeleton-determinism.cpp` | 5 | model-loading; no ordinary/parallel CTest registration | `lead` via `llama.cpp-0igs`: retain manual-only, lead-only, serial, once-only safeguarded procedure; any run remains `lead`/`0igs` work, outside Task 19 |
| `tests/test-expert-routing-roundtrip.cpp`; `tests/test-moe-expert-placement.cpp`; `tests/test-sycl-expert-cache-bandwidth.cpp`; `tests/test-sycl-race-conditions.cpp` | 4 | manual benchmark/diagnostic or special instrumentation | `lead` via `llama.cpp-0igs`: retain opt-in manual procedure; do not ordinarily register |
| `tests/test-q6k-56block-debug.cpp`; `tests/test-q6k-layout-debug.cpp`; `tests/test-q6k-variable-reorder.cpp`; `tests/test-tile-decomposition.cpp` | 4 | never-test local reimplementations; vacuous against production | `lead` via `llama.cpp-0igs`: delete current sources, preserving the Task 4b per-file reasons |
| `tests/test-sycl-prestage-routed-experts.cpp` | 1 | D-LOCAL: current source locally mimics routed-expert prestaging, includes no production header/path, and its bare `assert()` checks compile away under NDEBUG | `llama.cpp-3ygx`: delete the current source. Then `llama.cpp-m2ke` removes its active target, `add_test`, and properties; `llama.cpp-kdfh` verifies that no declined row remains registered. To seek reacceptance, rewrite against the production path with NDEBUG-proof checks and request a new Task 16 decision |
| `tests/test-expert-cache.cpp` | 1 | never-test obsolete include of removed `expert-cache.hpp` | `lead` via `llama.cpp-0igs`: delete current source, preserving the obsolete-header reason |
| `tests/test-sycl-expert-prefetch.cpp` | 1 | RE-DECLINED 2026-08-06 (was Task 16 accepted): includes removed `expert-cache.hpp`; registration dropped by `a4791b7a9` | `lead` via `llama.cpp-0igs`: delete or rewrite against the current prefetch API; a rewrite needs a new Task 16 acceptance decision |
| `tests/test-pinned-chunk-pool.cpp` | 1 | current source is unsafe/stale pending canonical pinned-pool API work | `lead` via `llama.cpp-32dg8.20`: preserve allocation/reuse/capacity/failure checks in a rewrite, perform its build/GPU proof, then request a new Task 16 acceptance decision |

Final reconciliation: **48 accepted + 16 declined = 64/64 unique rows** at the
Task 16 decision point; **47 accepted + 17 declined** after the 2026-08-06
`test-sycl-expert-prefetch` amendment above (registration dropped by
`a4791b7a9`; Task 19 ran the amended 50-name set at `772798e91`). The
five model-loading hazards are exactly the five declined model rows above; none
is accepted into ordinary CTest. The 38 accepted GPU rows remain lead-only and
serial, gated on Task 17 metadata, the `llama.cpp-8u22` mutation-proof gate,
and Task 19's final clean build plus accepted-set runtime. This is the owner
decision that removes `llama.cpp-awcp`'s policy blocker while
preserving the roughly 83-row post-merge assignment to `lead`.

**Deletion actions executed 2026-08-09** (`llama.cpp-0igs`), completing the
`D-LOCAL` / `D-OBSOLETE` rows above. All six sources are gone from `tests/`:

| commit | sources | verified before removal |
|---|---|---|
| `386b7ebb1` | `test-q6k-56block-debug`, `test-q6k-layout-debug`, `test-q6k-variable-reorder`, `test-tile-decomposition` | each has **zero** `#include "..."` of any production header, so nothing it asserts can reach backend code; two say so in their own opening comments ("Simplified Q6_K block for testing", "Inline the helpers for testing") |
| `54ebd8abe` | `test-expert-cache`, `test-sycl-expert-prefetch` | `ggml/src/ggml-sycl/expert-cache.hpp` does not exist and both include it, so neither compiles. Note `expert-prefetch.hpp` **does** still exist — it is the `expert-cache.hpp` include specifically that is dead |

`test-sycl-prestage-routed-experts` was already deleted earlier by
`llama.cpp-3ygx` (`b8c665363`). For each of the six, zero references were
confirmed across both live `CMakeLists.txt` and all of `tests/` immediately
before removal, with a positive control (the same check reports 11 references
for `test-cpu-gpu-soa-interaction`) so an absence could not be a probe that
never fires. Matches under `build-13qq/` are a stale generated build tree —
evidence these once compiled, not live references. Any rewrite needs a fresh
Task 16 acceptance decision.

## Task 17d: final restored-registration audit

This source-side audit is pinned at `fc606640e`, after the approved Task
17a–17c changes. It reconciles the normative Task 16 tables above against every
live `CMakeLists.txt`, rather than treating a source-name match or a zero-match
`ctest -R` as proof.

### Complete Task 16 reconciliation

| Task 16 set | source rows / targets | selectable CTest names in the canonical static configuration | audit result |
|---|---:|---:|---|
| accepted and already live before Task 17 | 40 | 40 | every target has exactly one `add_executable` and every exact Task 16 command name has exactly one `add_test` |
| accepted and restored by Task 17 | 8 | 11 | eight targets, including the four historical streaming modes from one executable; all are selectable |
| **accepted total** | **48** | **51** | **48/48 unique accepted targets active; 51/51 unique exact test names registered once** |
| declined | **16** | **0** | **no declined basename has an active `add_executable` or `add_test` in either CMake file** |

Thus the source-row decision remains **48 accepted + 16 declined = 64/64**.

> ⚠️ **The accepted/declined figures in this Task 17d section are pinned at
> `fc606640e`; the restored-registration figures below are corrected forward.**
> The `test-sycl-expert-prefetch` re-decline landed after the audit ran
> (`a4791b7a9`, 2026-08-05, reconciled into the Task 16 tables by `8518c9b70`),
> taking the accepted set to **47 rows / 50 CTest names / 17 declined**.
>
> The **48 / 51 / 16** counts in the reconciliation table above are left
> unedited deliberately: that table is the record of what the audit examined,
> and rewriting its numbers would make it a record of something that never
> happened. For the current accepted counts read the Task 16 reconciliation
> above; for what actually executed, the Task 19 section below.
>
> The **restored target/property table**, the **reproducible clean-configuration
> script**, and the **JSON reconciliation gate** that follow are a different
> kind of statement — they describe what a configure registers, and the script
> is meant to be run, so a stale figure there is a broken instruction rather
> than a misremembered record. Those are corrected to the post-`a4791b7a9`
> values (**10** restored CTest names, **6** `RUN_SERIAL`, **6**
> `SKIP_RETURN_CODE 77`), cited in place, with the `fc606640e` value kept
> alongside so the audit's own reading stays legible.

The difference between 48 accepted rows and 51 CTest names is intentional and
comes only from `test-mmvq-q8-0-streaming-bench.cpp`, whose one executable has
four required modes. The `sycl-restored` label identifies only the 11
registrations added by Task 17 — **9** of them at HEAD, after `a4791b7a9`
dropped `test-sycl-expert-prefetch` and `llama.cpp-u2mz` dropped the bare
streaming-bench name (see the correction under the restored table below); it is
not retroactively applied to the 40 registrations that were already live.

### Restored target/property table

`static` below means the target and its tests are inside
`if (NOT GGML_BACKEND_DL)`. The descriptor fixture has the additional
`GGML_SYCL_DNNL` guard and directly links `DNNL::dnnl`; no other restored row
acquires that oneDNN-only guard. Every target links the static
`ggml-base;ggml;ggml-cpu;ggml-sycl` topology, with `IntelSYCL::SYCL_CXX` added
when found.

| restored target | CTest names | guard | `sycl-restored` | `RUN_SERIAL` names | skip-77 names | manual names |
|---|---:|---|---:|---:|---:|---:|
| `test-cpu-gpu-soa-interaction` | 1 | static | 1 | 1 | 0 | 0 |
| `test-mmvq-q8-0-streaming-bench` | 3 | static | 3 | 0 | 2 | 3 |
| `test-mxfp4-xmx-tiled` | 1 | static | 1 | 1 | 1 | 0 |
| `test-q6k-reorder-dispatch` | 1 | static | 1 | 1 | 0 | 0 |
| `test-sycl-expert-prefetch` — **removed by `a4791b7a9`** | 0 | — | 0 | 0 | 0 | 0 |
| `test-sycl-fattn-onednn-descriptors` | 1 | static + oneDNN | 1 | 1 | 1 | 0 |
| `test-sycl-set-rows-owner-routing` | 1 | static | 1 | 1 | 0 | 0 |
| `test-unified-dispatch-integration` | 1 | static | 1 | 1 | 1 | 0 |
| **total at HEAD** (post-`u2mz` streaming drop) | **9** | **7 targets** | **9** | **6** | **5** | **3** |
| total after `a4791b7a9`, before that drop | 10 | 7 targets | 10 | 6 | 6 | 4 |
| total as audited at `fc606640e` | 11 | 8 targets | 11 | 7 | 7 | 4 |

⚠️ **Second forward correction, 2026-08-09 (`llama.cpp-24dl`).** The streaming row was
`4 / 4 / 0 / 3 / 4` and the HEAD total `10 / 10 / 6 / 6 / 4` until `llama.cpp-u2mz` removed the
bare `add_test(NAME test-mmvq-q8-0-streaming-bench COMMAND test-mmvq-q8-0-streaming-bench)`
registration. The reason is recorded in place at
`ggml/src/ggml-sycl/CMakeLists.txt:3351-3355`: that name carried no `ENVIRONMENT`, so it always
printed `SKIP: set GGML_SYCL_MMVQ_BENCH=1 to run` and exited 77 — **a registration that could
never do work**. The three surviving names drive the same executable with the env it needs.

The figures above are machine-derived from the live file (names carrying the `sycl-restored`
label, grouped by their `COMMAND` target), not hand-counted.

The `test-sycl-expert-prefetch` row carried one `sycl-restored` name, one
`RUN_SERIAL` and one `SKIP_RETURN_CODE 77`, which is the whole of the
11→10 / 7→6 / 7→6 delta; `a4791b7a9` deleted its `add_executable`, `add_test`
and property block together, so nothing else in the table moved.

The streaming names share one label property call. The MMQ smoke and forced-MMQ
modes have `SKIP_RETURN_CODE 77`; the cache smoke mode is deliberately not
included because its configured path has no exit-77 branch (unavailable
cache/device work is a failure there), and the bare benchmark mode that was the
fourth member of this group no longer exists. The other three skip-77
registrations match explicit source exits of 77 (four before `a4791b7a9`). The
six ordinary GPU fixtures are `RUN_SERIAL` (seven before `a4791b7a9`); the
opt-in streaming modes instead carry the `manual` label as a group. A
substring-regex scan of every restored label found zero matches for the
throttled-sweep denylist `residency|mem-handle|cache`.

### Generated CTest evidence and lead handoff

The lead supplied generated evidence at the same `fc606640e` snapshot; Task
17d did not configure, build, or run a GPU. The supplied
`cmake -S . -B build` completed with rc=0 **against an existing configured
build tree**. That is valid provenance for the generated evidence below, but it
is only a cache-preserving reconfigure command: in a fresh directory it would
leave the default `GGML_SYCL=OFF` and would not reproduce these registrations.
The supplied CTest JSON contained **250** tests total and exactly **11**
`sycl-restored` names, with **7** `RUN_SERIAL`, **7** `SKIP_RETURN_CODE 77`,
**4** `manual`, and **0** forbidden-label matches. `ctest -N -L
sycl-restored` selected 11 and `ctest -N -L manual` selected 4, so neither
selector proof is a zero-match pass.

The reproducible clean configuration below expands
`scripts/sycl-build.sh`'s `configure_args` and explicitly pins the required
source/direct-link backend topology (`GGML_BACKEND_DL=OFF`) instead of relying
on a pre-existing cache. It leaves `BUILD_SHARED_LIBS` at the script/project
default because that setting does not govern direct backend linkage. The ccache
launchers are included under the same availability check as the script.

```sh
set -euo pipefail
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u

rm -rf build
configure_args=(
  -S .
  -B build
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG'
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'
  -DGGML_SYCL=ON
  -DGGML_SYCL_TARGET=INTEL
  -DGGML_SYCL_ONECCL=ON
  -DGGML_SYCL_F16=ON
  -DGGML_BACKEND_DL=OFF
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
  -DCMAKE_C_COMPILER=icx
  -DCMAKE_CXX_COMPILER=icpx
)
if command -v ccache >/dev/null 2>&1; then
  configure_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi
cmake "${configure_args[@]}"

# test-sycl-expert-prefetch is deliberately absent: a4791b7a9 deleted the
# target, so naming it here would fail the build.
cmake --build build --config Release --target \
  test-cpu-gpu-soa-interaction test-mmvq-q8-0-streaming-bench \
  test-mxfp4-xmx-tiled test-q6k-reorder-dispatch \
  test-sycl-fattn-onednn-descriptors \
  test-sycl-set-rows-owner-routing test-unified-dispatch-integration -j 1

# Registration/property inspection only; these commands execute no tests.
ctest --test-dir build -N -L '^sycl-restored$'  # expected: 10 tests (11 before a4791b7a9)
ctest --test-dir build -N -L '^manual$'         # expected: 4 tests
ctest --test-dir build --show-only=json-v1 > /tmp/kdfh-ctest.json
```

There is intentionally no `-DGGML_SYCL_DNNL=ON` option to add: for
`GGML_SYCL_TARGET=INTEL`, the SYCL CMake requires `find_package(DNNL REQUIRED)`
and sets its internal `GGML_SYCL_DNNL=1` only after finding an Intel oneDNN
package. A clean configure that cannot find that package fails rather than
silently omitting oneDNN. This is the oneDNN-enabled assumption under which
`test-sycl-fattn-onednn-descriptors` contributes the tenth restored name.

The JSON proof must remain nonzero and reconcile to `sycl-restored=10`,
`RUN_SERIAL=6`, `SKIP_RETURN_CODE 77=6`, `manual=4`, and forbidden
restored-label matches=0 before runtime delegation — the post-`a4791b7a9`
figures. The audit at `fc606640e` reconciled the pre-`a4791b7a9` set (total 250,
`sycl-restored=11`, `RUN_SERIAL=7`, `SKIP_RETURN_CODE 77=7`); the total is not
restated as a current gate because commits after `fc606640e` add and remove
unrelated registrations, so only the restored-label figures are stable enough to
assert. Mutation proofs remain with `llama.cpp-8u22`, and clean accepted-set GPU
runtime remains with Task 19 (`llama.cpp-8kyi`); this audit claims neither.


## Task 19: accepted-set lead gate — executed results

The single recorded runtime execution of the accepted set. Source:
`llama.cpp-8kyi` (closed 2026-08-06), its comment `c-jx40`, and the committed
evidence in `artifacts/task19/`. Everything below is copied from that record;
nothing here is inferred.

### Build

| item | value |
|---|---|
| build SHA | `772798e91814340d07f20a4e9e3969427759ed2d` |
| command | `run_build restored-build ./scripts/sycl-build.sh` |
| rc | 0 |
| backend present | `grep -E '^GGML_SYCL:' build/CMakeCache.txt` → `BOOL=ON`; `ldd build/bin/llama-completion \| grep -cE 'libggml-sycl\|libsycl'` → 2 |

The backend check is not ceremonial. A reconfigure can silently reset
`GGML_SYCL` to `OFF`, the build still succeeds, and every token-level gate still
passes — on the CPU, ~13x slower. A green test run proves nothing about which
backend produced it.

### Registration reconciliation (before any execution)

`ctest -N` at `772798e91` found **50 of the 51** accepted CTest names registered
exactly once. The missing name, `test-sycl-expert-prefetch`, was deliberately
dropped by `a4791b7a9` — its source includes the removed `expert-cache.hpp`. The
Task 16 tables above carry that re-decline (accepted set **47 rows / 50 CTest
names**). Task 19 ran the amended 50.

### Runtime

Serial, one `ctest` invocation per name, under `GPU.lock`, on the B70:

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0          # selector, pinned
timeout 300 ctest --test-dir build -R "^${t}\$" --output-on-failure \
    > "task19-logs/$t.log" 2>&1
```

Per name the runner recorded rc, `Shmem` before and after (2 s settle), and a
**non-vacuity control**: `grep -cE '^\s*Start ' "$log"`, the count of tests the
regex actually selected. A count of 0 would have written a `ZERO_MATCH` row. All
50 rows report `matched=1` and **no `ZERO_MATCH` row exists** — so no result
below is a zero-match filter reading as a pass. The runner also aborted the whole
sweep if `Shmem` exceeded 100 GB after any test; it never fired.

| outcome | count |
|---|---:|
| pass (rc 0) | 38 |
| skip (rc 77, `***Skipped`) | 1 — `test-mmvq-q8-0-streaming-bench`, the opt-in benchmark mode |
| fail (ctest rc 8) | 11 |
| **total** | **50** |

Elapsed 2026-08-06 16:24:14 → 16:29:05 (−04:00), 4 min 51 s. Memory stayed flat
across the whole sweep: `Shmem` 5602656 kB at the first sample → 5623064 kB at
the last (5.60 → 5.62 GB), final `MemAvailable` 207398448 kB (~207 GB). Nowhere
near the hazard this document's model-loading rules guard against — these are
unit fixtures, not model loads.

### The 11 failures, with the exact printed signal

Every signal below is a string the binary actually prints, taken from that
test's captured log. None is a probe that could not have fired.

| test | printed signal | RCA ticket | ticket status |
|---|---|---|---|
| `test-dmmv-q4-0-coalesced` | `errors=88 max_diff=0.427313 max_rel=0.309105 FAIL` | `llama.cpp-szv8` | closed |
| `test-dmmv-q6k-coalesced` | `SYCL error: CHECK_TRY_ERROR(op(...))` at `ggml-sycl.cpp:38688` in `ggml_sycl_op_mul_mat`; subprocess aborted | `llama.cpp-99ke` | closed |
| `test-ggml-sycl-soa` | same `ggml-sycl.cpp:38688` abort (its own 8 subtests passed first) | `llama.cpp-99ke` | closed |
| `test-q8-0-layout-cache-path` | `Failed to resolve SoA layout pointer (source=wrong_layout)` | `llama.cpp-43uy` | closed |
| `test-q8-0-layout-cache-path-mmvq` | same `wrong_layout` resolve failure | `llama.cpp-43uy` | closed |
| `test-sycl-moe-identity-hash` | `FAIL: model_id mismatch` / `FAIL: test_single_weight_name_hash` (6 of 7 passed) | `llama.cpp-qvid` | closed |
| `test-sycl-unified-memory-e2e` | `[UNIFIED-CACHE] alloc malloc_device returned nullptr` | `llama.cpp-mequ` | closed |
| `test-sycl-weight-key-stability` | `FAIL: cache key missing GGUF identity fields` | `llama.cpp-n3pw` | closed |
| `test-sycl-weight-key-uniqueness` | `FAIL: tied weights did not share cache identity` | `llama.cpp-n3pw` | closed |
| `test-sycl-xmx-unified-correctness` | `SKIP: no graph-pinned entries, cannot validate graph path` then `FAIL: SYCL backend run failed` | `llama.cpp-sfe9` | closed |
| `test-tiered-dispatch` | `FAIL: all-device-first has no planner target for all-device-first.blk.0.attn_q.weight` | `llama.cpp-wmc2` | closed |

⚠️ **Ten of those eleven tickets have since closed, and that is NOT the same as
ten of those eleven tests now passing.** The closures are RCA-and-fix closures
landed at later SHAs (`llama.cpp-sfe9` at `ab6cbe970`, `llama.cpp-mequ`,
`llama.cpp-n3pw`, `llama.cpp-wmc2`, `llama.cpp-43uy`, `llama.cpp-99ke`,
`llama.cpp-qvid`), each verified against its own gate. The owed re-run has now
been executed — see the next section.

### Re-run at `d5ac9cc50` (2026-08-09): the set IS green

Executed by the lead with the Task 19 runner shape and ONE deliberate change:
**no global `ONEAPI_DEVICE_SELECTOR` export** — the 38 per-test pins landed by
`llama.cpp-24dl` (`da1d9c96f`) govern instead. (Task 19's global
`level_zero:0` export overrode every per-test pin; this run is the first where
the pins were actually exercised.) Evidence:
`artifacts/task19/rerun-d5ac9cc50-results.tsv`.

| outcome | count |
|---|---:|
| pass (rc 0, matched=1) | 44 |
| designed skip (77) | 1 — `test-xmx-host-streaming` (its body gates on `GGML_SYCL_XMX_GEMM && GGML_SYCL_MMQ_XMX`, off in the default config; see the registration comment) |
| fail | **0** |
| `ZERO_MATCH` (name no longer registered) | 5 — all documented deliberate removals, see below |
| **total names swept** | **50** |

**Every one of Task 19's 11 failures now passes**, including
`test-dmmv-q4-0-coalesced` (the szv8 instrument). Elapsed 2 min 51 s; `Shmem`
flat 2.69→2.72 GB; final `MemAvailable` ~220 GB. The non-vacuity control fired
correctly on exactly the five removed names:

- `test-mmvq-q8-0-streaming-bench` — bare name dropped by `llama.cpp-u2mz`
  (see the second forward-correction above).
- `test-onednn-fallback`, `test-xmx-kernel-config`, `test-xmx-quant-loaders`,
  `test-xmx-unified-kernel` — deleted by `llama.cpp-u2mz` `059d28670`
  ("ten decorative tests and one dead registration"); each carries an in-file
  tombstone comment at its former registration site explaining why.

So the live accepted set is **45 names: 44 green + 1 designed skip**.
`artifacts/task19/task19-names.txt` remains the historical 50-name list for the
`772798e91` run and is deliberately unedited.

`llama.cpp-szv8` has since CLOSED (2026-08-08, merged a0026c257, final GREEN at
0b73dfdad) and the once-"unexplained coincidence" is fully resolved: the
near-bit-identical failures across allegedly different kernels happened because
**neither run executed a DMMV kernel at all** — the TG fast-path
(ggml-sycl.cpp:~55303) claims every batch=1 quantized mul_mat with a non-AoS
layout and dispatches MMVQ with q8_1 activations upstream of both
GGML_SYCL_FORCE_DMMV sites. The 12–31% "wrong answer" was MMVQ's by-design
8-bit-activation accuracy scored against a full-precision oracle; the Q4_0
coalesced DMMV kernel is exonerated (one-hot probes clean, q8_1-MMVQ host
oracle matched the GPU to seven significant figures). The test now binds the
forcing (`GGML_SYCL_TG_FAST=0`) and asserts kernel identity with a
tolerance-proof 100x ratio gate; its row in this sweep is a genuine green.
Production-reachability question carried by `llama.cpp-erf1`.

### Known-failing, and not a regression — separate from the 11 above

⚠️ **`test-sycl-moe-sequence-graphlet-policy` is NOT one of the 11 Task 19
failures.** It carries a `mem-handle` label, so the throttled-sweep denylist
excludes it and the Task 19 name list never contained it — which is precisely
why it went unnoticed for so long. Every one of the 11 above maps to an RCA
ticket in that table; none of them is this. Do not conflate the two sets when
tallying merge-readiness reds.

`test-sycl-moe-sequence-graphlet-policy` fails for a cause that predates the
C-series base `9674a390b` and is not a production defect. Two sub-tests call
`read_required_file` on `docs/plans/2026-06-24-sycl-moe-aggregation-decision.md`
and `docs/plans/2026-06-24-sycl-moe-default-fast-path-decision.md`, which
`std::exit(1)` when absent — and **neither document was ever committed, at any
point in git history**, despite the test reading them since its first commit
`b08e732cd` (2026-06-29). So this test has never passed end-to-end in its
existence; the `mem-handle` label exclusion kept that invisible. Two further
items in the same file are equally unfixable as text drift: both region markers
for the `xmx_branch` sub-tests were removed from `ggml-sycl.cpp` by
`d87d54cdd`/`9402d151e`, and the `grouped_decode_runtime_uses_device_ids_contract`
gating flag was renamed into a broader flag. Tracked as `llama.cpp-26ak`
(**open**). Treat a red here as this known wall, not as a new failure — and do
not "fix" it by writing plausible-looking decision docs, since the two sub-tests
would then assert against invented content.

## Lead runbook: the 9 manual-only declined rows (`llama.cpp-0igs`)

These are the 5 `M-MODEL` + 4 `M-OPTIN` rows from the Task 16 declined table. Task 16's
ruling is that **none of them may enter ordinary CTest registration** — not even with hazard
labels and `RUN_SERIAL`, because a registered test is reachable by a sweep and these are not
safe to reach that way. What follows is the procedure that replaces registration.

**None of these has a CMake target.** Every one needs a manual build, and the four `M-OPTIN`
rows carry their own `icpx` line in the source header. **Do not add targets for them** — that
is the registration Task 16 declined.

### Rules that apply to every row here

- **Lead-only, serial, once-only.** One at a time, never concurrently with any other GPU
  work, never in a subagent or background task. Take `GPU.lock`.
- **Never loop one.** Per CLAUDE.md's never-loop rule, the property that matters is "loads a
  model onto a GPU", and the 5 `M-MODEL` rows all do. A single `test-llama-archs` run peaks
  at 195–206 GB of 255 GB unpinned; a second run starting before the first releases does not
  fit. There is no gradual approach — the first overlap is fatal.
- **Sample memory around every run**, not `free -g`, and let it settle:
  ```sh
  grep -E '^(MemAvailable|Shmem):' /proc/meminfo     # before
  <run the binary>
  sleep 5; grep -E '^(MemAvailable|Shmem):' /proc/meminfo   # after
  ```
  Abort the session if `Shmem` climbs past ~100 GB. A reading taken in the same command as the
  exiting process is measuring the wrong instant and manufactures a false alarm.
- **Pin the selector**, but ⚠️ **not all of these want `level_zero:1`** — see the per-row
  column. Two of them deliberately run on the **CPU** backend, and the two-GPU rows need both
  cards. A blanket `level_zero:1` would silently change what three of these rows test.
- **Check the GPU after any crash or forced stop** before trusting a later benchmark:
  `journalctl -k --since "1 hour ago" --no-pager | grep -iE 'GT reset|guc_id|CAT error'`.
- Model fixtures resolve via `MISTRAL_PATH` / `GPTOSS_PATH`, defaulting to
  `/models/mistral-7b-v0.1.Q4_0.gguf` and `/models/gpt-oss-20b-mxfp4.gguf`
  (`tests/test-planner-canary-common.hpp`).

### The 5 model-loading rows (`M-MODEL`)

| row | selector | what it does | expected shape |
|---|---|---|---|
| `mini-context-prototype` | **`opencl:cpu`** | Task 5 proof. Fork/execs three workers: `real-A` and `real-B` full weight loads plus a metadata-only (`no_alloc=true`) mini-context. | PASS iff `real-A == real-B` **and** `mini == real-A` for compute-buffer sizes and the FA auto-detect verdict. |
| `test-planner-canary-skeleton-determinism` | **`opencl:cpu`** | Canary D0.1. A thin orchestrator that fork/execs the prototype above — it **inherits L1's full-weight loads**, so it is the same hazard, not a lighter one. | Same PASS condition as the prototype; it reports the prototype's verdict. |
| `test-planner-canary-cpy-visibility` | **`opencl:cpu`** | Canary D0.3. One process loads Mistral 7B and runs `graph_reserve` five times, checking `(op_id, op_type, name)` sequences are identical. | PASS iff all five sequences match. The multi-device scenario is gated behind `D0_3_MULTIDEVICE` and is **unavailable on this host** — leave it unset. |
| `test-planner-canary-pp-tg-union` | `level_zero:1` | Canary D0.2. Fork/exec workers build PP-shape (ubatch=max) and TG-shape (ubatch=1) graphs for one model and check the union covers every op either executes. | ⚠️ Its own header states the completeness caveat: if a split aborts, later splits never fire their callbacks, so **an op set captured under abort is partial**. A PASS is only meaningful on a clean run. |
| `test-planner-canary-direct-load` | `level_zero:1` | Canary D0.4. Deliberately bypasses `llama_model_load_from_file`; `mmap`s the Mistral fixture and moves bytes into a pre-allocated device tensor via one `ggml_backend_tensor_set`. | PASS iff exactly one copy lands the bytes. Lightest of the five — it consumes the model **file** but does not build a llama context. |

⚠️ **The `opencl:cpu` selector on three rows is deliberate and load-bearing.** **Two** of them
record why in their own headers: it sidesteps host-side wedges — `zhzbp` for
`mini-context-prototype`, `m09zb` for D0.3 (`test-planner-canary-cpy-visibility`) — that block
these proofs on the GPU. The third, `test-planner-canary-skeleton-determinism`, says nothing
about the backend at all: it is a thin orchestrator that `execl`s the
`test-mini-context-prototype` binary, so it **inherits** that selector along with the protocol
rather than choosing it. Running any of the three on `level_zero:*` does not make them
stricter — it reintroduces the wedge they were written to route around.

### The 4 opt-in diagnostic/benchmark rows (`M-OPTIN`)

| row | build + selector | what it measures | expected shape |
|---|---|---|---|
| `test-expert-routing-roundtrip` | `icpx -fsycl -O2 -o test-expert-routing-roundtrip tests/test-expert-routing-roundtrip.cpp`; `ONEAPI_DEVICE_SELECTOR='level_zero:0;level_zero:1'` | P4.5 MoE routing across 2 GPUs: GPU0 gating → route → GPU1 expert compute → merge back. | **Exits 77 when two devices are unavailable** (source line 49) — a 77 here is a legitimate skip, not a pass. ⚠️ There is **no P2P between these cards** (different CPU root ports); all traffic host-bounces, so treat any throughput figure as host-bounce, never as peer DMA. |
| `test-sycl-expert-cache-bandwidth` | `ONEAPI_DEVICE_SELECTOR="level_zero:0,1"` | VRAM read BW, H2D BW, and MMVQ latency, local vs streamed. | Prints numbers; there is no pass/fail oracle — it is a measurement, so it must not be read as a gate. ⚠️ Its header still says **B580**, which is no longer in this machine; the second card is a **B70**. Do not compare its output against any B580-era figure. |
| `test-moe-expert-placement` | `icpx -fsycl -O2 -pthread -o test-moe-expert-placement tests/test-moe-expert-placement.cpp`; `ONEAPI_DEVICE_SELECTOR=level_zero:0` | Micro-benchmark of static-CPU vs dynamic host↔VRAM expert placement, incl. a DMA break-even point. | Measurement, not a gate — it opens with "Micro-benchmark" framing. ⚠️ It targets `level_zero:0`, the **B70 measurement card**: run it only when no capture or benchmark is in progress, and do not treat its absolute numbers as baselines under this host's permanent ambient load. |
| `test-sycl-race-conditions` | **needs its own TSan tree**: `cmake -B build-tsan -G Ninja -DGGML_SYCL=ON -DCMAKE_CXX_FLAGS="-fsanitize=thread" -DCMAKE_C_FLAGS="-fsanitize=thread" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"` | Spawns many threads calling lazily-initialized SYCL config functions concurrently. | The verdict is **ThreadSanitizer's report**, not the exit status — without proper atomics TSan reports a data race. A green run in a non-TSan build proves nothing, which is why this cannot be an ordinary registration. Build it in a separate tree; do not reconfigure the shared `build/` with sanitizer flags. |

**Why none of these becomes a registration.** Three need a non-default selector, one needs a
differently-configured build tree, two have no pass/fail oracle at all, and five load models
under a rule that forbids unattended repetition. Registering any of them would make it
reachable by a sweep that satisfies none of those preconditions — the failure mode this
document exists to prevent.

## `test-q6k-dispatch` — the oracle was the bug (`llama.cpp-2ln5`)

Closed 2026-08-04. Worth recording as a *contract* outcome rather than a bug
fix, because the disposition is the interesting part: a ~1% GPU-vs-CPU deviation
turned out not to justify a production change, so the correction landed in the
**reference**, not in the kernel and not in a widened tolerance.

The test compares a Q6_K GPU dispatch against a CPU reference. Its old reference
computed in F32 while the GPU path quantizes activations to Q8_1 — so it scored
the right answer against the wrong oracle, and the residual ~1.44% was the oracle
disagreeing with itself. The contract is now "match `CPU_Q8_1`
(`abs <= max(1e-3, 1e-4*abs(CPU_Q8_1))`)", with the F32 column retained as a
reported diagnostic rather than a gate.

Landed as `91c95398b` + `4178587d2` (from `a21067805` + `c004cb659`). Lead-verified
on hardware, build rc=0, under `GPU.lock`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`
— **one default green and two expected-RED controls**, which is what makes the
green mean anything:

| mode | rc | signal |
|---|---:|---|
| default | 0 | every Test 1/2 row satisfies the `CPU_Q8_1` contract — row 100 reports `GPU-REF 0.000%` while `Q8-F32` is `1.444%`; Tests 3/4 PASS |
| `--reference=f32` | 1 | the legacy mismatched oracle restored: Test 1 fails 2 of 16, Test 2 rows 100/1000 at 1.444%/1.440%; Tests 3/4 still PASS |
| `--corrupt-q8-reference` | 1 | the matching oracle perturbed: 16/16 and 6/6 specific `CPU_Q8` contract failures; Tests 3/4 still PASS |

The two controls answer different questions. `--reference=f32` proves the new
contract is the thing that changed the verdict; `--corrupt-q8-reference` proves
the new contract can still fail. Tests 3 and 4 pass in all three modes, which
demonstrates the change is scoped to the oracle rather than disabling the suite.
`MATCH_REF`/`GPU-REF` diagnostics were confirmed truthful in every mode. Memory
recovered afterwards (`MemAvailable` ~200 GiB, `Shmem` ~4.2 GiB).

The general lesson, which cost this project time elsewhere: **if the exact answer
also fails the test, the oracle is the bug.** A CPU reference is not
automatically a reference for a GPU kernel that takes a different numeric path.

---

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

## Historical coordination context for `llama.cpp-0igs`

At the 2026-08-01 snapshot, the prior `llama.cpp-0igs` implementer's lease had
ended after a host reboot. That historical owner gap explained why this audit
originally called for reassignment; it is no longer current. The ticket remains
open and is now assigned to `lead` (without an active worker lease).

The historical endpoint evidence remains: **141 `add_test()` registrations in
`tests/CMakeLists.txt` were wiped wholesale** near commit `4974bf53c` (the
earlier attribution to `d3dce4e0a` was retracted as an unsound topological-walk
artifact; see `llama.cpp-0igs`'s METHOD WARNING). `3c8f296fd` had exactly 147
`add_test()` calls and the then-HEAD had 6 before batches 1–3 restored 68. Those
registration-call counts support the wipe finding; they are not a one-to-one
source-row denominator. The pending work is approximately 147 C++ source rows,
of which Task 4c disposes 64 as merge-relevant and leaves roughly 83 as
post-merge debt.

Current ownership is concrete: `lead` owns, through `llama.cpp-0igs`, the 9
manual-only actions, 6 deletion actions, and roughly 83 post-merge source rows.
The pinned-pool row is sequenced separately through open task
`llama.cpp-32dg8.20`, also assigned to `lead`, before returning to Task 16.

This overlap explains this ticket's two "confirmed instances" — a test that is
not registered at all is the most extreme case of "cannot fail." **The bulk of
that problem remains `llama.cpp-0igs`'s scope, not re-litigated file-by-file
here**, but this pass independently found and fixed two directly on-theme
members of that same 186-file dead-source population (both call real production
code, neither is a mock):

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

**Current handoff:** no reassignment recommendation remains. `lead` is the
assigned owner for `llama.cpp-0igs` and `llama.cpp-32dg8.20`; the exact split and
sequencing are recorded above and in the Task 4c action-key legend.

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
| 17 files in the mechanical scan's **pre-restoration historical snapshot** (`test-ab-validation`, `test-crossover-discovery`, `test-dense-scheduler`, `test-edge-cases`, `test-expert-cache.cpp`, `test-ggml-flash-attn-ext`, `test-kv-cache-coordinator`, `test-pinned-chunk-pool`, `test-prefetch-scheduler`, `test-q6k-56block-debug`, `test-q6k-variable-reorder`, `test-sycl-prestage-routed-experts`, `test-tensor-classification`, `test-tile-decomposition`, `test-transfer-learning`, `test-vram-pool`, `test-xmx-unified-kernel`) | 1–77 | At that historical scan point all 17 were unregistered. At the Task 4b pinned live snapshot, `test-sycl-prestage-routed-experts`, `test-tensor-classification`, and `test-xmx-unified-kernel` are now actively registered; the other 14 remain absent from live registration (including the false-guarded/obsolete `test-expert-cache`). This row records the original NDEBUG finding, not a current registration audit or Task 4c disposition. Any remaining bare-assert source would need an NDEBUG-proof failure path before a future registration. |

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
  names.** Historically, 9 of batch 1's 13 restored tests carried LABELS
  containing the bare words `cache` or `mem-handle`, colliding with
  CLAUDE.md's throttled-sweep denylist (`-LE
  'residency|mem-handle|cache'`). The restoration pass called those nine
  host-only; Task 4b's stricter indirect-device audit reclassifies **8 of the
  9 as GPU serial** (only `test-tensor-classification` remains host-only).
  The first fix (`32a70dcf3`) renamed the labels to `cache-hostonly` /
  `mem-handle-hostonly` and was verified by checking that no label *equalled*
  a denylisted word — the right property, checked with the wrong operator.
  `ctest -LE` matches by regex/substring exactly the way `-R`/`-E` already do
  over test names (this doc's own earlier trap: `-E "test-llama-archs"` also
  excludes `test-llama-archs-table`), so `cache` still matched inside
  `cache-hostonly` and 11 of 41 restored tests stayed silently excluded.
  Nobody had written that `-L`/`-LE` share that semantic with `-R`/`-E`, so
  the fix repeated the exact defect class it was fixing. Corrected in
  `b7555cc73` by dropping the hazard word entirely rather than suffixing it
  (bare `hostonly` + subsystem word, no shared substring with the denylist at
  all), and verified with `re.search` over every label string instead of
  equality — 0 of 41 collide. At the pinned live snapshot, the `hostonly`
  labels retained by those eight GPU rows — and by the two GPU-serial unified
  memory rows restored in batch 2 — are stale hazard metadata; Task 17 owns
  correcting labels and adding serial scheduling enforcement. This Task 4b
  audit does not edit CMake. `llama.cpp-0igs` itself documents the underlying
  failure mode (`impl-5exz`: 5 of 9 un-guarded tests similarly affected);
  batch 1's original rate was worse, 9 of 13.
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

---

# Phase A: the batched mutation-verification plan (owner ruling 2026-08-08, `ona8` c-ntbb)

Written by `impl-u2mz-plan` as **analysis only** — no build, no GPU, no ctest, no
binary was run producing this section, and no production or test code was
touched. Everything below is a *prediction* the lead executes in Phase B. Where
a row says "fires exactly N of M checks", that is a reading of the source, not a
transcript.

## The population, and why it is 131 and not "~65–70"

The earlier estimate in *What this pass did NOT cover* ("the remaining ~65–70
currently-registered SYCL C++ tests outside that loop") **undercounts by roughly
half.** Derived from the CMake sources rather than a directory listing or a
possibly-stale generated `build/` tree:

```
ggml/src/ggml-sycl/CMakeLists.txt
  literal add_test(...) calls .................................. 156
  after expanding the five foreach loops
    (_case h2 x36, _case h10 x5, _mutation x3,
     _residency_test x19, _packed_k_checkpoint x5) ............. 219 ctest names
  of those, naming a C++ executable target ..................... 199
  of those, naming ${Python3_EXECUTABLE} or a script ...........  20
  DISTINCT C++ executable targets .............................. 131

minus already mutation-proven (excluded from this plan) ........   4
  test-kv-slice-sizing            (ticket description: assign->resize, 8 of 87)
  test-sycl-retained-handoff-contended  (llama.cpp-fbj5)
  test-sycl-set-rows-bounds             (llama.cpp-nhip)
  test-dmmv-q4-0-coalesced              (llama.cpp-szv8, ratio gate)
                                        ------------------------------------
PLANNED IN THIS SECTION ....................................... 127
```

⚠️ **127 is the Phase A figure and the derivation above is its history — it is
not the current population.** The remediation pass deleted ten of the tests it
counted, so the live number is **117**; see *REMEDIATION PASS* below and the
appendix, which carries the arithmetic. Re-running the `grep -c 'add_test('`
command below will now return a smaller number than the 156 recorded here, for
the same reason.

Reproduce the arithmetic (registration-only; executes nothing):

```bash
grep -c 'add_test(' ggml/src/ggml-sycl/CMakeLists.txt          # 156
grep -c 'llama_build_and_test(' tests/CMakeLists.txt           # 45 (2 are inside comments)
```

⚠️ **The `-N` cross-check must come from the source, not `build/`.** The main
checkout's generated tree can be stale, and 21 of the 131 targets sit behind
CMake guards that are OFF in the current cache — so `ctest -N` in `build/`
lists *fewer* tests than the sources register, and the difference is not
drift, it is the guards (see *Blocked* below). `ctest -R` on a guarded-off
name **exits 0 by matching zero tests**, which is the trap the ticket already
names.

The 127 rows are the SYCL-side population. Tests registered in
`tests/CMakeLists.txt` that are not SYCL-specific (`test-chat`,
`test-tokenizer-*`, `test-grammar-*`, the 71 `llama_test_pytest` gates already
cleared in c-2qqk) are out of scope for this ticket and are not planned here.

## How to run a batch (this is the whole cost model)

The saving is **not** parallel mutation — one mutation is live at a time,
always. The saving is that mutations landing in the **same file** share one
build sequence:

```
apply mutation 1 -> build -> run test 1 -> REPLACE with mutation 2 (same file)
 -> build -> run test 2 -> ... -> restore file -> build once
```

That is **N+1 builds for N tests** instead of 2N, and every build after the
first is a ccache-warm rebuild of one TU plus a link. Restoring between tests
is what costs money; don't.

Build-cost classes used below (from CLAUDE.md's recorded figures: ~10 min warm
full build, ~15 min device links, ~50 min for `ggml-sycl.cpp` at a *cold* path):

| class | what it dirties | est. per build |
|---|---|---|
| `NO-REBUILD` | nothing (source-text gate, or a shipped `--flag` control) | 0 |
| `STANDALONE` | a small .cpp compiled directly into the test target only | ~2 min |
| `SMALL-TU` | one backend .cpp inside `libggml-sycl` (compile + device link) | ~12 min |
| `HEADER-WIDE` | a widely-included .hpp — most of the backend recompiles | ~25 min |
| `MEGA-TU` | `ggml-sycl.cpp` (100k lines) or a header only it includes | ~20 min |

These are estimates from recorded figures, not measurements taken by this pass.

---

## Batch A — NO REBUILD (2 tests + 2 control runs, ~10 min total)

Run these first. Two are source-text contract gates that read the worktree
`.cpp` **at runtime**, so editing the file turns them red with no compile at
all. The other two rows are *control runs* of shipped fault injectors — they
cost nothing and prove the harness reaches its target before you pay for the
real mutation, which for both lives in a later batch (`test-q6k-dispatch` in H,
`test-sycl-transient-alloc-intent-scope` in G). Only the two gates count toward
the population total.

| test | mutation | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-q6k-dispatch` | none — run `./build/bin/test-q6k-dispatch --corrupt-q8-reference` (also `--reference=f32`, documented delta 1.444%) | GPU | COUNT `Results: %d passed, %d failed, %d skipped` | test 1 `FAIL: 16/16 rows violate the enforced CPU_Q8_1 (abs <= max(1e-3, 1e-4*abs(CPU_Q8_1))) contract`, test 2 `FAIL: 6 sample rows violate the enforced CPU_Q8_1 (...) contract`; `Results: 2 passed, 2 failed, 0 skipped`; rc 1 | Built-in positive control. Do this before spending the header-wide `vecdotq.hpp` rebuild in Batch H. **Only tests 1 and 2 can fire**: `corrupt_q8_positive_control` is read at `:413` and `:590` and nowhere else, and the contract string is printed only at `:470`/`:646`. Test 3 compares two GPU runs to each other, and test 4 compares GPU to `cpu_dot_q6k_f32` under its own fixed 1% gate (`:830`) — neither consults the Q8 oracle, so no oracle mutation can reach them. Non-vacuity control: the banner prints `Q8 reference mutation: ENABLED (expected RED)`. Device-less: every subtest prints `SKIP: Could not initialize SYCL backend` and **returns true** — exit 0 having tested nothing. |
| `test-sycl-moe-fusion-noactivation` | `ggml/src/ggml-sycl/mmvq.cpp:418` `return unsafe && std::atoi(unsafe) != 0;` → `return true;` | CPU-ONLY | PASSFAIL, `PASS: MoE fusion probes no-activation guard` | `FAIL: tests/test-sycl-moe-fusion-noactivation.cpp:134: GGML_SYCL_MOE_GLU_Q8_FUSED_XMX alone must not activate runtime fused store`; rc 1 | Fires 1 of ~40 CHECKs. ⚠️ **The line anchor moves with the compiler and with any edit above the CHECK — grep the message, not the number.** Batch A's ctest binary printed `:124` while the CHECK's macro sat at source line 123 and its message at 124 (it anchors on the closing line); a host `g++` build of the same source anchors on the opening line. After `llama.cpp-ax4r` the CHECK sits at 133/134, so `:134` is the expected ctest anchor (`:133` observed from `g++`; not re-verified against the ctest binary, which needs a rebuild). It greps `mmvq.cpp`/`ggml-sycl.cpp` as text, so it can only ever see *string* edits — a behavioural regression that leaves the text intact is invisible to it. `read_required_file` `std::exit(1)`s if run from a tree without the sources. |
| `test-sycl-moe-direct-final-scratch-plan` | `ggml/src/ggml-sycl/mmvq.cpp:19808` delete the `n_ids > max_i64 / n_tokens ||` term | CPU-ONLY | PASSFAIL, `PASS: direct-final scratch plan` | `FAIL: tests/test-sycl-moe-direct-final-scratch-plan.cpp:82: direct-final total_batches overflow check must be present`; rc 1 | Same source-grep mechanism. Its other two cases assert `test_moe_direct_final_scratch_plan` (`ggml-sycl.cpp:24368`), a test-only reimplementation with no production caller — see *Mocks*. |
| `test-sycl-transient-alloc-intent-scope` | none for the control run: the file header (lines 36–47) tabulates exit codes for two wrong-fix predicate variants; run `./build/bin/test-sycl-transient-alloc-intent-scope own-thread` / `other-thread` | DEVICE-FREE-THREADS | PASSFAIL, 3 progress `PASS:` lines + final | see Batch G for the real mutation | Best-instrumented test in the population; property 2 is explicitly the positive control for property 1. |

### Batch A — executed results (2026-08-09, `llama.cpp-u2mz` Phase B)

Run lead-serial. The two source-text gates and the two control runs were all
executed; logs in the session scratchpad as `batchA-*.log`.

| row | executed | verdict |
|---|---|---|
| `test-sycl-moe-direct-final-scratch-plan` | GREEN → RED → GREEN | **clean**. RED printed the pre-registered string exactly: `FAIL: .../test-sycl-moe-direct-final-scratch-plan.cpp:82: direct-final total_batches overflow check must be present`. Both bookends `PASS: direct-final scratch plan`. |
| `test-sycl-moe-fusion-noactivation` | mutation RED fired; **baseline was already red** | **harness sound, oracle stale**. The mutation moved the first failing CHECK from `:164` to `:124` and back, so the gate demonstrably discriminates — but the binary's rc was 1 in all three runs, including the two logged as "green". The standing failure was `:164`, which grepped `ggml-sycl.cpp` for `direct_final_probe.fused_q8_quarantined` / `.q8_capacity_ok` / `.device_xmx_ok`: zero hits, and `git log -S` finds no commit that ever added them. Filed and fixed as `llama.cpp-ax4r` (see below). |
| `test-q6k-dispatch --corrupt-q8-reference` | injector functional | **shape corrected**. Real output is 2 failed / 2 passed, not 3/1 — see the row above for why tests 3 and 4 are structurally out of the injector's reach. The banner line `Q8 reference mutation: ENABLED (expected RED)` confirms the flag engaged, so this is a real result and not a no-op run. |
| `test-sycl-transient-alloc-intent-scope own-thread` / `other-thread` | rc 0, rc 0 | **matches the shipped predicate row.** The file's own exit-code table (header lines 36–47) gives `per-thread (shipped)` as `other-thread 0 / own-thread 0`; both wrong-fix variants red exactly one of the two, so a 0/0 pair is only reachable from the shipped predicate. |

`llama.cpp-ax4r` is fixed in `3fb01d4ce`: the three greps now name the local
bools production actually uses (`direct_final_fused_q8_quarantined` /
`direct_final_q8_capacity_ok` / `direct_final_device_xmx_ok`,
`ggml-sycl.cpp:66776-66784`), and the CHECK now asserts the *ordering* its own
message claimed — those reasons are returned at `:66804`/`:66807`/`:66810`,
before `metadata` at `:66818`. Three existence greps could never see "before".
The remaining literals in that file were audited one `grep -c -F` per term
against their target file; those three were the only zero-hit positives.

## Batch B — STANDALONE test-target TUs (19 tests, ~1.5 h)

### B1 — executed results (2026-08-09, `llama.cpp-u2mz` Phase B)

All four `model-lifecycle.cpp` rows: **GREEN → RED → GREEN**, each RED with its
pre-registered string, each restore verified (`git diff` empty, source line
byte-identical), final full convergence build rc=0 and all four names re-green.
ctest wraps the binary failure as **rc 8** (the c-hwhw wrap), scored on the
printed strings per this doc's own rule:

| row | RED string observed | specificity |
|---|---|---|
| load-txn `:668` | `rollback was not idempotent` | `sycl-lifecycle-h2-rollback-idempotence` red; **M1/M2/M3 injector tests stayed green** (the row's mandated constraint) |
| owner-reset `:817` | `H14 repeat teardown is not OK_ALREADY_DEAD` | `sycl-lifecycle-h14` red |
| wrapper-overlap `:169` | `BUSY reaper/destructor overlap lost wrapper semantics` | its own name red |
| runtime-host `:100` | `BUSY destructor/reaper overlap lost or duplicated token` | its own name red |

Not run: the other 41/8 sibling names per binary (the plan's full-binary
specificity claims remain source-derived; the executed check was the targeted
name + the injector-green constraint). Logs: session b6a72a8d scratchpad
`b1-*.log`. The final build also refreshed `build/bin`'s fusion-noactivation
binary to the `3fb01d4ce` oracle fix — green via ctest and direct invocation.

### B2 chunk 1 — executed results (2026-08-09): 5/5 GREEN → RED → GREEN

All five RED strings byte-matched their pre-registered rows (gpu-arch anchors
one line later, `:85` — the known compiler/line-drift class):

| row | RED observed |
|---|---|
| gpu-arch `:89` | `FAIL: ...test-gpu-arch.cpp:85: B70 name must fall back to ARC_BATTLEMAGE, not ARC_ALCHEMIST` |
| gpu-arch `:63` | `FAIL: ...test-esimd-dpas-gate.cpp:112: Battlemage architecture must classify as ARC_BATTLEMAGE` |
| cold-start `:134` | `FAIL: A580 should use tile_m=32` + `Results: 4 passed, 1 failed` |
| moe-scratch-admission `:79` | `FAIL: over-plan-request-is-refused: ... (allowed=1 reason=allowed)` |
| mmvq-launch-geometry `:58` | `FAIL: already-aligned-shapes-are-unchanged: aligned row count 16 must pad to itself` |

⚠️ **The first attempt VOIDED four of these rows on this ticket's own first
trap.** A fresh hand-written runner queried ctest by BINARY name; four of the
five register under different ctest names, `ctest -R` exited 0 on zero matches,
and rc=0 on the mutated builds read as "mutation survived" — while nothing ran,
baselines included. The recompile-and-relink evidence was checked and healthy,
which made the void look like four decorative tests. Caught by reading one run
log ("No tests were found!!!"). The re-run runner carries the task19
`matched=` counter and a ZERO_MATCH tripwire; every future batch runner must
be adapted from a hardened runner, never written from recall.

**Binary → ctest name map for this chunk** (the plan tables list binary
names; use these with `-R`): `test-gpu-arch` → `gpu-arch`;
`test-esimd-dpas-gate` → `esimd-dpas-gate`; `test-cold-start` →
`test-cold-start` (1:1); `test-moe-scratch-admission` →
`sycl-moe-scratch-admission` (+ `sycl-pp-moe-scratch-admission-contract`);
`test-mmvq-launch-geometry` → `sycl-mmvq-launch-geometry`
(+ `sycl-mmvq-launch-geometry-contract`). The `-contract` siblings ran red
under mutation with their primaries (matched=2); final re-green executed the
primaries, the siblings' green following from binary identity after the
restore build. Logs: `b2a-*.log` (first attempt, void rows retained as the
record of the trap) and `b2b-*.log` (the counting re-run).

None of these relink `libggml-sycl`; the mutated .cpp/.hpp is compiled straight
into the test binary. Sub-batch by file — the `model-lifecycle.cpp` group of
four is the single best value in the whole plan.

### B1 `ggml/src/ggml-sycl/model-lifecycle.cpp` — 4 tests, 5 builds

| test | mutation | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-sycl-lifecycle-load-txn` | `:668` `++rollbacks_;` → `rollbacks_ += 2;` | DEVICE-FREE-THREADS | PASSFAIL — silent on success | `rollback was not idempotent` on stderr, rc 1 | **Outstanding specificity: this binary backs 42 ctest names** (36 `h2-*`, 5 `h10-*`, `multi-model`) and only `sycl-lifecycle-h2-rollback-idempotence` goes red. Ships built-in fault injectors `--mutation M1/M2/M3` driven by `tests/test-sycl-lifecycle-mutations.py` (ctest `sycl-lifecycle-mutation-M1..M3`) — **your mutation must leave those green**. `publication-concurrency` is the slow case (20k publishes x 4 readers). |
| `test-sycl-lifecycle-owner-reset` | `:817` `return { dead->second.first == token ? error::OK_ALREADY_DEAD : error::STALE_IDENTITY, ... };` → `return { error::STALE_IDENTITY, ... };` | CPU-ONLY | PASSFAIL — silent on success | `H14 repeat teardown is not OK_ALREADY_DEAD`, rc 1 | 9 ctest names share this binary; only `sycl-lifecycle-h14` (+ the unqualified name) goes red. ⚠️ The source comment at lines 301–307 warns `prepare_teardown`'s two stale-identity guards are individually **null mutations** — do not pick those. |
| `test-sycl-lifecycle-wrapper-overlap` | `:169` drop `|| active_txn_ != 0` from the LOAD_BUSY guard | DEVICE-FREE-THREADS | PASSFAIL — silent on success | `BUSY reaper/destructor overlap lost wrapper semantics`, rc 1 | Fail-fast, so the 6 later scenarios are masked. RUN_SERIAL. |
| `test-sycl-lifecycle-runtime-host` | `:100` in `quarantine_queue::enqueue` `if (tokens_[i] == token) { return true; }` → `if (false) { ... }` | DEVICE-FREE-THREADS | PASSFAIL — silent on success | `BUSY destructor/reaper overlap lost or duplicated token`, rc 1 | `require()` is a real function, not `assert()` — NDEBUG-safe. |

### B2 — one test per file (15 tests, 30 builds)

| test | mutation | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-gpu-arch` | `gpu-arch.cpp:89` `name_contains(name, "B70")` → `"B71"` | CPU-ONLY | PASSFAIL, `=== All gpu-arch tests passed ===` | `FAIL: tests/test-gpu-arch.cpp:84: B70 name must fall back to ARC_BATTLEMAGE, not ARC_ALCHEMIST`, rc 1 | Uses a CHECK macro, not `assert()` — deliberate, the build is `-DNDEBUG`. Fires 4 assertions but CHECK returns on the first. |
| `test-esimd-dpas-gate` | `gpu-arch.cpp:63` `return sycl_gpu_family::ARC_BATTLEMAGE;` → `ARC_ALCHEMIST` | CPU-ONLY | PASSFAIL, `=== All esimd-dpas-gate tests passed ===` | 6 `ok: ... (ESIMD dpas enabled)` lines, then `FAIL: tests/test-esimd-dpas-gate.cpp:112: Battlemage architecture must classify as ARC_BATTLEMAGE`, rc 1 | Same file as the row above — **batch these two together**. Its link seams stub `XMXConfig::from_device`, a documented coverage gap. |
| `test-cold-start` | `cold-start.cpp:134` `else if (caps.eu_count >= 256)` → `>= 257` | CPU-ONLY | COUNT `Results: %d passed, %d failed` | `FAIL: A580 should use tile_m=32` / `Expected: 32, Got: 16` / `Results: 4 passed, 1 failed`, rc 1 | `test-gpu-arch` also compiles `cold-start.cpp`; expect a collateral red there. |
| `test-moe-scratch-admission` | `moe-scratch-admission.cpp:79` `if (aligned.ring_depth > cap.ring_depth)` → `> cap.ring_depth + 1` | CPU-ONLY | COUNT `test-moe-scratch-admission: PASS (9 cases)` + per-case `ok:` | 5 `ok:` lines then `FAIL: over-plan-request-is-refused: a deeper ring than planned is refused (allowed=1 reason=allowed)`, rc 1 | Narrowest clause available; a `>`→`>=` on any cap line detonates 5 cases at once. Has its own positive control (`refusal-reaches-no-side-effect`). |
| `test-mmvq-launch-geometry` | `mmvq-launch-geometry.hpp:58` `(rows + subgroups_per_workgroup - 1) / n` → `(rows + subgroups_per_workgroup) / n` | CPU-ONLY | COUNT `test-mmvq-launch-geometry: PASS (6 cases)` | `ok: slice-of-one-pads-to-one-workgroup` then `FAIL: already-aligned-shapes-are-unchanged: aligned row count 16 must pad to itself`, rc 1 | Only `mmvq.cpp` includes the header in production, and the test target compiles it directly — so this is standalone. Has a built-in positive control (`the-uniformity-predicate-can-fail`). |
| `test-moe-control-plan` | `moe-control-plan.cpp:188` drop `&& std::strstr(name, "_chexps") == nullptr` | CPU-ONLY | fixed-count only — `moe control plan: PASS (25 cases)` where 25 is `sizeof(cases)/sizeof(...)`, **not** a running count | `FAIL: grovemoe-chexps-family-parses: chunked gate must be a matrix`, rc 1 | The "25" cannot be diffed for partial breakage — treat as PASSFAIL. `moe_control_reset_all_state()` brackets every ledger case. |
| `test-zone-sizing` | `zone-sizing.cpp:139` `if (zone_is_moe_expert_tensor(tensor))` → `if (false && ...)` | CPU-ONLY | PASSFAIL, `PASS: zone-sizing structural path-scoped maxima` | `FAIL: tests/test-zone-sizing.cpp:145: gpt-oss onednn_eligible must skip the expert family and fall to the dense attention family`, rc 1 | Sibling predicates keep expert tensors, so they stay green — good specificity. Case 11 exists to distinguish `&&` from `||`; do not "simplify" it. |
| `test-sycl-dispatch-tuning` | `dispatch-tuning.cpp:242` `return ...MMVQ_COALESCED;` → `MMVQ_SOA` | CPU-ONLY | PASSFAIL, `PASS` | `FAILED: mmvq kernel mismatch`, rc 1 | 1 of 3 assertions. Writes `/tmp/dispatch_tuning_test.json` (tmpfs) and removes it on success. |
| `test-sycl-e2e-profile` | `e2e-profile.cpp:135` `g_e2e_tg_profile.tokens += 1;` → `+= 0;` | CPU-ONLY | PASSFAIL via `std::abort()` | stderr `test-sycl-e2e-profile: [SYCL-E2E-TG-PROFILE] tokens=1 ops=3 moe_calls=2 ...` then **SIGABRT, rc 134** | Failure mode is abort, not exit 1 — do not grep for "FAIL". Chosen so the RED string is quotable; the `GGML_OP_MUL_MAT_ID → MOE` alternative gives only `requirement failed`. |
| `test-sycl-cpu-traits-parity` | `cpu-traits-support.cpp:158` `index >= 0 && index < GGML_TYPE_COUNT` → `index <= GGML_TYPE_COUNT` | CPU-ONLY | PASSFAIL — prints **nothing** on success | `bounds check failed`, rc 1 | Cheapest in the plan (1 TU, links only ggml-base/ggml-cpu). Despite the name it calls no SYCL symbol. Returns an out-of-bounds pointer but never dereferences it. |
| `test-sycl-timeline` | `sycl-timeline.cpp:468` drop `|| state.successful_file_flushes > 0` | CPU-ONLY | PASSFAIL via `std::abort()` | `test-sycl-timeline: second flush must not clobber the first trace file`, **rc 134** | ~45 preceding `require()`s stay green. Targets an invariant the source comment states (lines 483–484). |
| `test-onednn-woq` | `onednn-woq.cpp:33` `out.scales_mask = (1<<0)|(1<<1);` → `(1<<0)` | GPU (test 4 only) | PASSFAIL | `FAILED: expected scales/zp mask=3 got 1/3`, rc 1 | 89-line TU — the cheapest backend file in the tree. ⚠️ `test_woq_gemm_q4_0` **fails open in five places** (`SKIP: ...` → `return true`). |
| `test-mmq-xmx-dispatch` | `xmx-dispatch-gate.hpp:46` `batch >= 1 && batch < threshold` → `batch <= threshold` | GPU-gated, launches nothing | COUNT `Tests run: %d, Passed: %d, Skipped: %d, Failed: %d`; 77 on no device | `FAILED: threshold <= 1 must accept no batch at all`, `Tests run: 5, Passed: 4, ... Failed: 1`, rc 1 | Test target links no `ggml-sycl` at all — build only this target and the mega-TU is untouched. ⚠️ Tests 1/2/4/5 are near-vacuous (they exercise the file's own CPU reference GEMM); test 3 is the only production-bound one, and its header records that test 3 *itself* used to be vacuous (`llama.cpp-cwev`). |
| `test-sycl-lifecycle-runtime-wrapper` | `ggml/src/ggml-backend-reg.cpp:1146` `case REMOVED: return "REMOVED";` → `"REMOVED_"` | DEVICE-FREE-THREADS in this build | PASSFAIL, `[sycl-runtime-wrapper] <phase>` markers | `[sycl-runtime-wrapper] assert failed: failed first publication registry tombstone state != REMOVED` + `state=REMOVED_ durable_owners=0 lookup_visible=0` + `generic registry lifecycle fixture failed`, rc 1 | ⚠️ **Big time-saver: in the default (non-DL) build this test touches no SYCL code at all** — its whole SYCL half is under `#if defined(GGML_SYCL_RUNTIME_MODULE)`, defined only when `GGML_BACKEND_DL=ON` (default OFF). The mutation lives in core ggml, so it is red in both builds. RUN_SERIAL. |
| `test-sycl-lifecycle-event-lease` | `execution-lifecycle.cpp:270` in `release_invocation_locked` `if (validate_root(graph.token_root, root) != error::OK) return error::MISMATCH;` → prefix `false &&` | CPU-ONLY (links only `Threads::Threads`, never includes `sycl/sycl.hpp`) | PASSFAIL — prints **nothing** on success | stderr `H13b release accepted wrong root`, rc 1, under `--case H13b` and under the bare all-cases run | **2-of-11 specificity**: 11 ctest names share this binary and only `sycl-lifecycle-h13b` (+ the unqualified name) goes red — H13b is the only case passing a deliberately wrong root. The binary is already mutation-aware: `test_mutation::M7_SUBMIT_RELEASES_DEVICES_EARLY` (`execution-lifecycle.cpp:245`) is a compiled-in fault injector that cases M7/M7b assert against. ⚠️ `execution-lifecycle.cpp` is also compiled into `test-sycl-lifecycle-owner-reset` — expect collateral reds there. ⚠️ `g6c()` runs only in the all-cases path; there is **no** `add_test(NAME sycl-lifecycle-g6c ...)`, unlike its siblings. |

---

## Batch C — `mem-handle.cpp` (5 tests, 6 builds, ~1.4 h)

| test | mutation | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-sycl-mem-handle-lifetime` | `:786` drop the trailing `&& arena_gen_ == other.arena_gen_` from `stable_identity_equal` | CPU-ONLY | PASSFAIL, `PASS: mem_handle lifetime diagnostics` | `FAIL: different arena generations must not compare stable-equal`, rc 1 | Fires exactly 1 of 15 CHECKs; the hash at `:761` still folds `arena_gen_`, so only the equality predicate moves. |
| `test-sycl-graph-retention-scope` | `:91` `... || g_ggml_sycl_graph_recording;` → `... || ggml_sycl_graph_recording_active();` (re-widens the guard; reintroduces `llama.cpp-oze0`) | DEVICE-FREE-THREADS | PASSFAIL, 3 `PASS:` + final | `FAIL: a non-recording thread's handle was parked for command-graph lifetime because ANOTHER thread was recording: count 0 -> 1.`, rc 1 | Best-designed test in the population: property 2 **is** the anti-mutation ("narrowing all the way to never-retain also makes property 1 pass"), and it carries a setup-inertness check. This is the ticket's premise-was-wrong instance 2 (c-g0zx) — it is registered, and this row upgrades the code-review verdict to a real mutation. |
| `test-sycl-retained-handoff-barrier` | `:1227` in `begin_retained_handle_publish` `++state.publishers;` → `state.publishers += 2;` | DEVICE-FREE-THREADS | PASSFAIL, three `PASS:` lines | `FAIL: drain did not clear after the publisher handed off its handle`, rc 1, after the ~1000 ms drain timeout | ⚠️ Do **not** instead delete `state.publishers == 0` from the drain predicate at `:1272` — that fires the very first check *and* every later one, so it tells you nothing about which half of the barrier works (and it is the mutation `llama.cpp-fbj5` already executed on the *contended* sibling). ⚠️ Removing `cv.notify_all()` from `retained_handle_publish_ticket::reset()` is a **void** mutation — `cv.wait_for` re-evaluates its predicate at timeout. |
| `test-mem-handle-wrong-device` | `:816` `self.ptr == theirs.ptr && size_ == other.size_;` → `size_ == other.size_;` | GPU | COUNT `Tests: %d run, %d passed, %d skipped` | `FAILED: CHUNK_LEASE stable identity must distinguish different ptrs inside the same leased chunk`, `Tests: 6 run, 5 passed, 2 skipped`, rc 1 | ⚠️ The obvious target — the wrong-device guard at `:504` — is a **null mutation under this registration**: ctest pins `level_zero:1`, so `total_gpu_count < 2` and both wrong-device cases `TEST_SKIP`. |
| `test-sycl-mem-handle-concurrent-resolve` | `:1011` in `operator=(const mem_handle &)` delete `new_entry->in_use_count.fetch_add(1);` | DEVICE-FREE-THREADS | PASSFAIL, one `PASS:` per subtest | `PASS: concurrent resolve() never observes a half-written state` then `FAIL: lease refcounts drifted: entry_a=4294967295 entry_b=4294967295 (expected 1 and 1)`, rc 1 | Deterministic, not race-dependent (the uint32 underflows). Selectable: `./build/bin/test-sycl-mem-handle-concurrent-resolve lease-once`. ⚠️ Do not mutate `take_lease_state_locked`'s `leased_entry_ = nullptr` — `store_lease_state_locked` overwrites it immediately, a silent no-op. |

## Batch D — `unified-cache.cpp` (14 tests, 15 builds, ~3.5 h) — the largest single-file group

23,238 lines, so each build is at the slow end of `SMALL-TU`. All 14 sites are
independent; sequence them in one file-open.

| test | mutation (`ggml/src/ggml-sycl/unified-cache.cpp:`) | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-sycl-residency-reservation` | `:390` `plan.bytes_reserved = plan.bytes_requested;` → `= 0;` | CPU-ONLY | PASSFAIL, `PASS: residency reservation policy` | `FAIL: reserved bytes must match accepted request`, rc 1 | The most surgical row in the plan: **1 of 22 CHECKs**. Alternative for the forced-budget path: delete `:359`'s `std::min(..., forced_bytes)` → fires only subtest 5. |
| `test-sycl-moe-residency-preflight` | `:383` `plan.reason = residency_reject_reason::FRAGMENTATION;` → `::BUDGET;` | CPU-ONLY | PASSFAIL, `PASS: MoE residency preflight policy` | `FAIL: enough total bytes but too-small largest block must reject as fragmentation`, rc 1 | 1 of 21 CHECKs — exactly the "does this distinguish exhaustion from fragmentation" question the subtest was written for. Each subtest `unsetenv`s the force-budget var first: a real built-in control. |
| `test-sycl-residency-diagnostics` | `:3746` `if (live == 0 && !old.retired)` → `if (live == 0)` | CPU-ONLY | PASSFAIL, `PASS: residency diagnostics` | `FAIL: retired entry replacement must be refused`, rc 1 | 1 of ~17 CHECKs; the live-lease refusal in the *same* subtest stays green — strong specificity. |
| `test-sycl-moe-handle-resolution` | `:4800` `if (location == cache_location::HOST_MMAP && !req.allow_mmap_host)` → `if (false)` | GPU (32 MB WEIGHT arena) | PASSFAIL, `SYCL MoE handle resolution tests: PASS/FAIL` | `  FAIL: plain host expert should honor allow_mmap_host=false`, rc 1 | ⚠️ **Mislabelled `hostonly`** — it allocates GPU memory, and because its labels omit `cache`/`residency` the throttled sweep *runs* it. ⚠️ **No `ONEAPI_DEVICE_SELECTOR` in the registration** and the source self-defaults to `level_zero:0` — the B70. Pin `level_zero:1` manually or it perturbs any benchmark in flight. |
| `test-unified-cache-bugs` | `:5621-5625` delete the `direct_expert_it` probe block in `is_cached` | GPU | PASSFAIL, `Unified cache bug tests: PASS/FAIL` | a failure inside `=== Test: is_cached layout/type coverage ===` case (c), then `... FAIL`, rc 1 | Reproduces the exact pre-fix `llama.cpp-5pvn` bug, and the subtest is written as a discriminating control (its comment predicts which cases a *blanket* break would also take down). ⚠️ CLAUDE.md never-loop family, ~8.5 GB RSS, one run only. One subtest self-skips unless `GGML_SYCL_TEST_UNIFIED_CACHE_GRAPH=1` and prints its SKIP without affecting `ok` — so a green run has always left that property unverified. |
| `test-unified-cache-fast-path` | `:5755` `if (entry.retired \|\| entry.layout != layout)` → `== layout` | GPU | COUNT `Tests: %d run, %d passed, %d skipped` | `FAILED: try_get_cached_fast should return non-null for cached entry`, `Tests: 2 run, 1 passed`, rc 1 | Only 2 tests exist, so 1-of-2 is the maximum specificity available; test 1 is a near-tautology. Side observation: `try_get_cached_fast` takes a `unique_lock` despite the "fast path / shared_lock" naming — pre-existing, untested. |
| `test-mem-handle-eviction` | `:6766` `if (have_mapped && !(mapped == ckey))` → `if (false && ...)` (kills the `id_to_key_` fallback in `acquire_weight_lease`) | GPU | COUNT `Tests: %d run, %d passed` | `[TEST] lease_and_plain_lookup_agree ... FAILED: acquire_weight_lease must resolve whatever get_weight_ptr resolves`, `Tests: 6 run, 5 passed`, rc 1 | **The test's own source certifies this discriminator** (lines 227–228). Tests 1c/2/3/4 carry explicit positive preconditions so they cannot pass vacuously. |
| `test-unified-cache-concurrent` | `:7589` in `evict_one` `if (entry.pinned)` → `if (false)` | GPU + 8/4 threads | PASSFAIL, `Unified cache concurrency tests: PASS/FAIL` | `Pinned entries evicted unexpectedly (used_before=4096 used_after=0)`, rc 1 | Deliberately does **not** touch the sibling `in_use_count > 0` lease guard 12 lines below — that is the canonical-contract guard and a much broader break. Tiny budgets (2 MiB), no memory hazard. Defaults to `level_zero:0` (B70) if unset. |
| `test-sycl-reset-model-weight-lease-preserve` | `:8643` `if (!entry.owner_tagged && (mode == MODEL_TEARDOWN \|\| live_mask != 0))` → drop the `\|\| live_mask != 0` | GPU (64 KB budgets) | PASSFAIL + `GREEN:`/`RED (control):` progress lines; exits 77 correctly | `FAIL: an unattributed entry was freed at a load boundary while a model is still live. ...`, rc 1 | **Already carries paired mutation controls** — `set_live_model_mask(0)` re-runs each preserve assertion in the state where reclaim *must* happen, so a no-op mutation cannot masquerade as a pass. Peak RSS 317 MB. Do not instead mutate `:8632` — that fires four test functions at once. |
| `test-mmvq-q8-0-streaming-bench` | `:9529` `copy_to_device_async(..., src.ptr + offset, cur, ...)` → `+ offset + 1` | GPU | PASSFAIL + timing lines | on `test-mmvq-q8-0-streaming-smoke`: `FAIL: GPU(cache) mismatch (nmse>2.0e-04 or max_abs>5.00e-02)`, rc 1 | **Four ctest names share this binary and only two are live.** The bare `test-mmvq-q8-0-streaming-bench` registration has no ENVIRONMENT, so it prints `SKIP: set GGML_SYCL_MMVQ_BENCH=1 to run` and exits 77 **always** — it is not a test (see *Undeletable/unmutatable*). The mutation fires only the `-smoke` (cache) name; the two `mmq` names take the graph path and stay green — useful cross-registration specificity. ⚠️ `test-mmvq-q8-0-streaming-smoke` is the one registration deliberately left out of the `SKIP_RETURN_CODE 77` list, so a 77 from it reads as FAILED. |
| `test-sycl-runtime-alloc` | `:11279` `if (expected_device >= 0 && expected_device != it->second.handle.device)` → `expected_device < -1 && ...` | GPU | COUNT `Tests: %d run, %d passed` (27) | `[TEST] strict_device_mismatch_fails ... FAILED: device mismatch free should fail`, `Tests: 27 run, 26 passed`, rc 1 | `ok &= f(q)` so all 27 still run after the failure — ideal for count-diffing. |
| `test-sycl-moe-q8-scratch` | `:15815` `demand.total_bytes = graph_op_count * demand.aligned_bytes_per_buffer;` → `* demand.bytes_per_buffer` | GPU (subtest 3) | PASSFAIL, `PASS: sycl MoE Q8_1 scratch sizing` | `FAIL: total scratch should reserve every graph Q8_1 buffer`, rc 1 | Only the totals assertion moves (12960 → 13056 alignment). `failed += !test(...)` with no early exit, so subtests 2/3 still run and pass. Subtest 3 prints `SKIP: SYCL backend unavailable` and **returns true** — a false green. |
| `test-sycl-moe-xmx-tiled-single-layout-planner` | `:18093-18095` delete the `if (legacy_env) { return std::atoi(legacy_env) != 0; }` branch | CPU-ONLY | PASSFAIL, `single-layout XMX_TILED planner tests passed` | `FAIL: tests/test-sycl-moe-xmx-tiled-single-layout-planner.cpp:84: legacy unsafe PP knob must still be honored with forced prompt XMX`, rc 1 | Fires the last CHECK of the last of 4 subtests. Not a mock: the planner helper delegates to the same `mxfp4_moe_single_gateup_layout_policy` production calls at `:18615`. |
| `test-sycl-layout-choice` | `:621` in `arena_default_external_headroom` delete the `std::max(..., arena_min_safe_external_headroom(...))` clamp | **CPU-ONLY** (hint was wrong) | 12 `PASS: <policy>` lines, no totals | 11 `PASS:` then `FAIL: arena headroom should raise undersized caller slack to the safe floor, got 592445440 expected 603979776`, rc 1 | Its device half is gated behind `GGML_SYCL_TEST_LAYOUT_CHOICE_BACKEND=1`, which ctest never sets — the registered run prints `SKIP: backend layout choice purge requires ...` and returns 0. The `ONEAPI_DEVICE_SELECTOR` in its ENVIRONMENT is vestigial. Fail-fast, so mutate late in the sequence, never in `run_fused_gate_up_role_test`. |

## Batch E — `mmvq.cpp` (5 tests, 6 builds, ~1.4 h)

| test | mutation (`ggml/src/ggml-sycl/mmvq.cpp:`) | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-sycl-moe-gateup-prepack-policy` | `:1018` `result.reason = "down-sum";` → `"capacity"` | CPU-ONLY | PASSFAIL, `PASS: MoE gate/up prepack policy`; CHECK prints `FAIL: %s:%d: %s` with file+line | `FAIL: tests/test-sycl-moe-gateup-prepack-policy.cpp:184: missing direct down-sum compatibility must reject before route selection`, rc 1 | 1 CHECK; the whole reject *ladder* is preserved, only one label moves. A sibling policy exists in `ggml-sycl.cpp:24474` — prefer this site, same evidence, ~20 min cheaper. |
| `test-sycl-moe-fused-down-sum-policy` | `:250-251` `if (std::strcmp(env,"tile4")==0) { return 4; }` → `return 2;` | CPU-ONLY | PASSFAIL, `PASS: MoE XMX down-sum direct-final policy` | `FAIL: tests/test-sycl-moe-fused-down-sum-policy.cpp:335: tile4 down Q8 DPAS tile env must parse`, rc 1 | 1 CHECK in the 13th of 17 subtests. 3 subtests source-grep `mmvq.cpp` and `std::exit(1)` (not CHECK) if the file is not found — a confusing hard exit from an unexpected CWD. Editing env-name *strings* would trip those greps as collateral; this mutation touches none. ⚠️ **THE MUTATION ABOVE IS TEST-LOCAL AND DOES NOT SATISFY THE ADJUDICATION** (see *Deferred dead code* → ADJUDICATED): only **5 of the 17** subtests reach production, via the memoized wrappers in `mmvq.cpp` ~`:214-364` to the dispatch sites `:19184-19208`. Pin the Phase B mutation to one of those five before spending the build slot; the row's `:250-251` edit reddens an env parser inside the test file. |
| `test-moe-mul-mat-id-q4q8` | `:2751` in `mul_mat_vec_q_id` swap the ids strides: `iid1 * ids_nb1 + id * ids_nb0` → `iid1 * ids_nb0 + id * ids_nb1` | GPU | PASSFAIL, per-case `nmse=%.6e max_diff=%.6f` | `MoE MUL_MAT_ID Q4_0 (base): nmse=<large> ...` + `FAIL: ... mismatch beyond tolerance (nmse>1.0e-03 or max_diff>2.00)`, rc 1 | Route verified on-path (n_experts=8, top_k=4, ne12=4 → threshold 32 ≥ batch 4, so MMVQ). Swapped strides stay in-bounds (max byte 108 of 128) so it corrupts rather than faults. Low specificity is unavoidable — main `break`s on the first failing case. |
| `test-moe-mul-mat-id` | `:4102` `get_int_from_table_16(v0, kvalues_mxfp4)` → `(v1, ...)` | GPU | PASSFAIL, per-case `nmse=... max_diff=...` | `MoE MUL_MAT_ID MXFP4 (base): nmse=<large> ...` + `FAIL: ... (nmse>5.0e-04 or max_diff>1.00)`, rc 1 | ⚠️ **Route uncertainty — confirm before building.** The test registers weights on the *host* buffer type, and CLAUDE.md routes host-resident weights to CPU dispatch; if so the live MXFP4 kernel is `ggml_sycl_cpu_expert_mul_mat_batched` in `cpu-dispatch.cpp`, not `mmvq.cpp`. Run once with `GGML_SYCL_DEBUG=1` and read the dispatch line first. ⚠️ Do **not** mutate `kvalues_mxfp4` in `ggml-common.h` — it is shared with the ggml-cpu reference, so both sides move and the test stays green. |
| `test-q8-0-layout-cache-path-mmvq` | `:21361` pass `src0_dd_i` (AoS base) instead of `soa_base` to `reorder_mul_mat_vec_q8_0_q8_1_sycl` | GPU | PASSFAIL, `Max diff: ... Result: PASS/FAIL` | `Max diff: 2.560000e+02, max rel: 1.000000e+00, min abs: 0.000000` then `Result: FAIL`, rc 1 | The test memsets `weight->data` to zero after caching, so an AoS read yields exactly 0 — unambiguous. The widest-blast mutation here, acceptable because the test's whole premise is "did the SoA pointer get used". Do not simplify the load-transaction bracket (RCA `llama.cpp-43uy`, lines 137–146). |

## Batch F — remaining single-file backend TUs (11 tests across 9 files, ~4 h)

| test | mutation | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-q8-0-layout-cache-path` | `dmmv.cpp:3705` drop `\|\| src0->type == GGML_TYPE_Q8_0` from `allow_layout` | GPU | PASSFAIL, `Result: PASS/FAIL` | `Max diff: 2.560000e+02, max rel: 1.000000e+00, min abs: 0.000000` / `Result: FAIL`, rc 1 | Same zeroed-AoS trick as its mmvq sibling. Backend-init failure returns 1, not 0 — no false green here. |
| `test-dmmv-q6k-coalesced` | `dmmv.cpp:1775` `tile_byte_offset += ts * (128 + 64 + 16);` → `ts * (128 + 64)` | GPU | COUNT — per-case `errors=%d max_diff=... max_rel=... PASS/FAIL` + `=== Summary: %s ===` | four green `ncols=8192` lines then `ncols=16384 nrows=8: errors=<n> ... FAIL` and `=== Summary: FAIL ===`, rc 1 | Maximum specificity: at ncols=8192 `num_tiles=1` so warp 0 never enters the loop. **Same file as the row above — batch them.** Confirm the run hits the COALESCED branch (`dmmv.cpp:3429`, KTRACE `dmmv_q6_k_coalesced_variable`) or nothing fires. False green: init failure prints `SKIP:` and returns 0. |
| `test-sycl-fattn-onednn-descriptors` | `fattn-onednn.cpp:182` `if ((H_q != H_kv) && (k_nc_stride != D \|\| v_nc_stride != D))` → prefix `false &&` | GPU | PASSFAIL, `SYCL oneDNN FA descriptor tests: %s` + `Descriptor child WIFEXITED status=%d` | `FAIL: GQA-5D-materialized max_abs=<big>` + 16 `[i] actual= expected=` lines, same for `MQA-5D-materialized`; the two `-direct` cases stay tiny; rc 1 | The test forks a child specifically so a numeric failure is provably `WIFEXITED status=1` and never a signal. Self-sets `level_zero:0` (B70) if unset. |
| `test-sycl-fattn-onednn-materialization` | `fattn-onednn.cpp:263` `const int64_t nb1 = value_tensor ? desc.v_src_nb1 : desc.k_src_nb1;` → `= desc.k_src_nb1;` | GPU (test 4) | PASSFAIL, one summary line | `FAIL: GQA V source byte offset mismatch` + `FAIL: non-monotonic V source byte offset mismatch` + `SYCL oneDNN FA materialization tests: FAIL`, rc 1 | **Same file as the row above — batch them.** Every K-side assertion and all of test 4 stay green. ⚠️ `#if !GGML_SYCL_DNNL` prints a skip and returns **0**, not 77. |
| `test-fattn-thread-local` | `fattn.cpp:908` in `tl_seq_id_buffers::free_all()` `if (g_fattn_shutting_down.load(acquire)) return;` → `if (false) return;` | GPU | PASSFAIL, `Thread-local buffers cleaned up successfully` | `Shutdown guard freed buffers unexpectedly (before=<N> after=<N-2>)`, rc 1 | 1 of 4 checks. ⚠️ Its ctest ENVIRONMENT sets only `LD_LIBRARY_PATH` — **no selector** — and the test self-defaults only if the var is unset, so under ctest it enumerates the iGPU. Pin manually. Two sibling `free_all()` overloads at `:851`/`:962` are not exercised. |
| `test-sycl-set-rows-owner-routing` | `set_rows.cpp:434` prefix `false &&` to the untracked-device-USM fail-closed clause | GPU (case 6 only) | PASSFAIL, `SYCL SET_ROWS owner routing tests: %s` | `  FAIL: untracked device USM must fail closed instead of being treated as host-stageable`, rc 1 | ⚠️ The mutated case is the **only** device-touching one; a device-less run SKIPs it and reports green. Confirm `=== Test: untracked device USM fails closed ===` is not followed by `SKIP:`. A device-free alternative exists in the dst-root-owner resolution (fires cases 1 and 5). |
| `test-sycl-cpu-dispatch` | `cpu-dispatch.cpp:1510` `const int n_full = std::min(threshold, n_tasks);` → `= n_tasks;` | CPU-ONLY | COUNT `%d passed, %d failed` | `test_int4_kernel_correctness ... FAIL (INT4 output identical to baseline, mean_diff=0.000000)` + `test_adaptive_split ... FAIL (expected 2 INT4-different tasks, got 0)`, `2 passed, 2 failed`, rc 1 | ⚠️ **Finding: the test is insensitive to the threshold constant it appears to guard** — every expectation is derived from `ggml_sycl_expert_miss_burst_threshold()` itself, so mutating that constant fires nothing. INT4 path is `#if defined(__AVX2__)`. This binary aborted in a past sweep (`artifacts/verify/restored41.txt:1207`) — confirm a green baseline first. |
| `test-mem-ops` | `mem-ops.cpp:255` both-host fast path `std::memcpy(dst_ptr, src_ptr, size);` → `size / 2` | GPU | `FAIL: %d mem_ops checks failed` on red; `PASS: ...` on green; exits 77 correctly | `FAIL: H2H copy byte 2048 expected 0x11 got 0x00` + `FAIL: 1 mem_ops checks failed`, rc 1 | 1 of 5 checks (the other four have a device endpoint). ⚠️ The "temporary handle event lifetime" check is **weak**: all four handles are `from_direct` raw views owning no lease, so flipping `retain_until_event` at `:507`/`:517` would **not** redden it. |
| `test-mxfp4-xmx-tiled` | `moe-tile-convert.cpp:153` out-of-range scale padding `*dst_ptr++ = 0;` → `= 1;` (AoS kernel only) | GPU | PASSFAIL | `AoS conversion mismatch at byte <N>: expected=0 got=1`, rc 1 (the SoA comparison passes first) | ⚠️ Do **not** mutate `MXFPXMXLayoutInfo::compute` or `reorder_mxfp4_to_xmx_layout` — the CPU reference and the GPU kernels are compared against each other, so shared layout math moves both sides and **cannot fail**. Exits 77 properly on no-device/non-XMX. |
| `test-q6k-reorder-dispatch` | `convert.cpp:1021` `scales_ptr[ib*(QK_K/16)+j] = x[ib].scales[j];` → `= x[ib].scales[0];` | GPU | COUNT `=== All Tests Complete: %d failure(s) ===` + per-subtest `Result: PASS/FAIL` | `=== Test 1: Production Q6_K Reorder Layout Verification ===` / `Result: FAIL`, later subtests still PASS, `1 failure(s)`, rc 1 | ⚠️ **Only 1 of 8 subtests reaches production.** Tests 2/3/5/6/7/8 re-derive the `quants.hpp` offsets and transcribe `vec_dot_q6_K_q8_1_impl_mmvq` inline — mark them MOCK. That is also why this mutation is clean (`failures` accumulates, no early exit). No 77 path: a device-less host prints `FAIL: SYCL error:` and exits 1. |
| `test-mmq-q6k-gpu` | `mmq.cpp:2007` swap the two scale indices in `sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x() + sc[i0/2+1]*sumi_d.y())` | GPU | COUNT `Results: %d passed, %d failed` + `Max relative error: %.4f%%` | `FAIL: <N> mismatches in 4096 values (...%)` for batch 2/8/16, `Results: 0 passed, 3 failed`, rc 1 | **Not specific** — the Q6_K MMQ launch macro selects on compute capability, never on ncols, so all three batches fire together. `vec_dot_q6_K_q8_1_impl_mmq` is shared by the AoS/SoA/coalesced kernels. ~200 lines of layout noise precede the verdict; grep `Results:`. |

## Batch G — `common.hpp` (5 tests, 6 builds, ~2.5 h)

The most widely included backend header — every build here is a full backend
rebuild including the mega-TU. **Land this batch when no other agent is
building** ([[land-header-changes-first-in-multi-agent-waves]]).

| test | mutation (`ggml/src/ggml-sycl/common.hpp:`) | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-sycl-tensor-placement` | `:1502` `strstr(name, "ffn_down_chexps");` → `false;` in `infer_tensor_usage`'s `moe_exps_name` | DEVICE-FREE-THREADS | COUNT `%d/%d tests passed` then `OK`/`FAILED`; checks accumulate rather than early-return | `FAIL usage(blk.0.ffn_down_chexps.weight): expected <MOE_EXPERT_WEIGHT>, got <FFN_WEIGHT>`, count one short, `FAILED`, rc 1 | **The ticket's instance 1, now the best-designed test in the population.** Its header states the exact mutation contract, and `check_role(...)` stays green because the role classifier is an independent literal list in `unified-cache.hpp:203` — one literal fires exactly one assertion of ~40. Real negative controls (`ffn_norm_exps`→NORM, `ffn_gate_shexp`→UNKNOWN). |
| `test-sycl-tensor-usage` | `:1544` `if (strstr(name, "_norm"))` → `"_normx"` | CPU-ONLY | PASSFAIL, `Tensor usage test: PASS/FAIL` | `FAIL: norm expected=5 got=0` then `Tensor usage test: FAIL`, rc 1 | Checks are `ok = ok && ...` so they **short-circuit** — mutating an early classifier hides everything after it. `_norm` is the 6th of 7 deliberately: five green checks run first. The 7th is skipped by the short-circuit; a missing 7th line is not a second failure. |
| `test-sycl-transient-alloc-intent-scope` | `:369` `intent.constraints.use_pinned_pool = !graph_lifetime;` → `= true;` | DEVICE-FREE-THREADS | PASSFAIL, 3 progress `PASS:` + final; `FAIL [%s]: ...` + `^ %d field(s) ...` | `FAIL [this thread recording]: host transient use_pinned_pool is 1, expected 0` + `^ 1 field(s) denied graph-lifetime treatment to a thread that IS recording.`, rc 1 | Ships a documented mutation matrix (header lines 36–47) tabulating exit codes for two wrong-fix variants — cross-check against it. Subtests selectable by argv. |
| `test-sycl-kv-view-resolution` | `:3871` `void * ptr = static_cast<char *>(base_ptr) + view_offs;` → drop `+ view_offs` | **CPU-ONLY** (hint was wrong — mocked topology, host stack "device" pointers) | PASSFAIL, `SYCL KV view resolution tests: PASS/FAIL`, per-subtest `FAIL:` lines | `FAIL: device-1 view should resolve as root device-1 pointer plus view_offs` (192), the permuted/flattened/slow equivalents, then `... FAIL`, rc 1 | Fires ~4 of 20; the 16 routing/ownership subtests stay green, and `ok &= ...` means all 20 run. Its one real-USM subtest is gated behind `GGML_SYCL_TEST_KV_VIEW_RUNTIME=1`, unset by ctest. ⚠️ Do not instead delete the fail-closed clause at `:3866` — it falls through to `..._slow()`, which may also return nullptr, so it can silently produce no RED. |
| `test-sycl-xmx-unified-correctness` | `:4552` `return resolved.ptr != nullptr && resolved.on_device;` → `&& !resolved.on_device;` | GPU | PASSFAIL; tri-state exit, 77 is a genuine skip | `FAIL: weight 'weight' is not device-resident on device 0 before compute; MUL_MAT would be routed to CPU and the comparison would be CPU-vs-CPU`, rc 1 | ⚠️ **Null-mutation trap stated by the test itself**: it does not assert which GPU kernel variant ran, so mutating `can_use_xmx()` thresholds or any single XMX kernel can silently reroute to a correct non-XMX path and stay green. The residency predicate is the only production property it hard-asserts. |

## Batch H — other shared headers (14 tests across 9 headers, ~6 h)

| test | mutation | radius | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|---|
| `test-sycl-unified-memory-e2e` | `unified-cache.hpp:2183` `budget_ - used` → `budget_ - used + 1` | HEADER-WIDE | GPU (hundreds of MB) | COUNT `E2E Test Results: %d/%d passed, %d failed, %d skipped`; exits 77 for skip *and* partial-skip | `  FAIL: available (<X>) != budget (<Y>) - used (<Z>)` then `E2E Test Results: 9/10 passed, 1 failed`, rc 1 | Off-by-one is too small to alter any other subcase's sizing. ⚠️ 77 here is neither pass nor fail (header lines 63–65). If the `base_budget()` pre-flight trips (another test created the device-0 cache in-process) you get 77 before any subcase runs and the mutation proves nothing — verify the run reached `Cache budget pinned:`. |
| `test-sycl-unified-cache` | `unified-cache.hpp:2183` `budget_ - used` → `budget_ + used` | HEADER-WIDE | **GPU** (hint was wrong) | COUNT `Test Results: %d/%d passed, %d failed` (13) | `FAIL: available=<budget+used>, expected <budget-used>` + `FAIL: available > budget (...)`, `Test Results: 11/13 passed, 2 failed`, rc 1 | **Same line as the row above — one build covers both** (apply the `+ used` form, run both, they are distinguishable by their own messages). Device-less: prints `ERROR: No SYCL device found` and returns **1**, never 77. |
| `test-sycl-moe-xmx-tiled-single-layout-policy` | `unified-cache.hpp:274` `if (in.pp_rows > 1 && !in.pp_supported)` → `> 4096` | HEADER-WIDE | CPU-ONLY | PASSFAIL, `single-layout XMX_TILED gate/up policy tests passed` | `FAIL: tests/test-sycl-moe-xmx-tiled-single-layout-policy.cpp:80: missing PP proof must reject`, rc 1 | Fixture uses pp_rows=2048, so raising the threshold silently admits a request with no PP proof — the exact fail-open the clause prevents. Fires 2 of 18 CHECKs. **Same header as the two rows above — batch all three.** |
| `test-sycl-weight-key-uniqueness` | `unified-cache-key.hpp:61` `const bool compare_logical = !a.has_gguf;` → `= true;` | HEADER-WIDE | GPU-lite (reaches `ggml_backend_sycl_reg()` despite the `hostonly` label) | PASSFAIL | `FAIL: tied weights did not share cache identity`, rc 1 | 🚨 **CHECK THE BASELINE FIRST — this test appears pre-existing RED at exactly the assertion this mutation targets** (`artifacts/task19/failing-logs/test-sycl-weight-key-uniqueness.txt`, `0% tests passed`). If still red the experiment is void and the finding is "fix this first". Fallback if phase 3 is already red: `ggml-sycl.cpp:~12005` `id.valid = true;` → `false`, firing phase 1's `FAIL: gguf_keys contains null key`. |
| `test-sycl-moe-identity-hash` | `unified-cache-key.hpp:61` — **the same line** | HEADER-WIDE | CPU-ONLY | COUNT `Tests run: %d, Passed: %d, Failed: %d` | `FAIL: two models mapping the same GGUF bytes must share one key` (tests 2/4, likely 6), `Tests run: 7, Passed: 4, Failed: 3`, `FAILED: the weight cache key no longer implements the n3pw physical-identity ruling.`, rc 1 | **One build covers this and the row above.** Exits 77 correctly when `GGML_USE_SYCL` is undefined. Narrower alternative: delete `\|\| a.aux_id != b.aux_id` (lines 66–67) → fires only test 7. Do not mutate `cache_id_hash` at `:83` as well — it must stay consistent with `cache_id_equal`. |
| `test-sycl-moe-gateup-prepack-scratch` | `moe-layer-plan.hpp:411` `if (!handle.has_stable_owner_identity())` → `if (false)` | HEADER-WIDE (via common.hpp) | CPU-ONLY | PASSFAIL, `PASS: MoE gate/up prepack scratch descriptor`; `FAIL: %s:%d: %s` | `FAIL: tests/test-sycl-moe-gateup-prepack-scratch.cpp:152: raw direct pointer handle without stable owner identity must be rejected`, rc 1 | This is the canonical-memory-contract property ("raw pointers are not ownership tokens") stated as a test — the right thing to mutate. ~35 earlier CHECKs pass first. Sibling guards at `:154`/`:504` — mutate **only** `:411`. |
| `test-sycl-descriptor-retention` | `moe-layer-plan.hpp:185` `if (resolved.layout != entry.expected_layout)` → `!= resolved.layout` | HEADER-WIDE | CPU-ONLY | PASSFAIL, `PASS: descriptor retention and replay validation` | `FAIL: layout change must reject replay`, rc 1 | Fires the 9th and last check; the other 8 stay green. **Same header as the row above — batch them.** Labelled `residency|mem-handle|cache`, so the throttled sweep excludes it; run by name. |
| `test-ggml-sycl-soa` | `quants.hpp:59` `d_offset = total_qs_bytes + block_index*sizeof(ggml_half)` → `total_qs_bytes - sizeof(ggml_half) + ...` | HEADER-WIDE | GPU | COUNT `Results: %d passed, %d failed` (~24) + per-subtest `Subtests: %d passed, %d failed` | `Test 11: SoA layout byte-level verification` → four `Block N d at offset ...: 0x0000 (expected 0x3C00) MISMATCH` + `FAIL: SoA layout mismatch (qs_correct=1, d_correct=0)`, `Results: 23 passed, 1 failed`, rc 1 | Writer and reader both go through `get_d_offset`, so numerics stay consistent and only the byte-level test sees it — but the 2-byte overlap clobbers the last qs block's tail, so **1–2 failures is the expected RED**. ⚠️ 19 separate `SKIP: Could not initialize SYCL backend` sites that `return true` — confirm real passes in `Results:` before trusting green. |
| `test-unified-dispatch-integration` | `dispatch.hpp:91` `return false; // FP16, BF16, F32, Q6_K ... legacy` → `return type == GGML_TYPE_Q8_0;` | HEADER-WIDE | GPU | COUNT `=== Results: %d/%d tests passed ===`; exits 77 if `GGML_SYCL_UNIFIED_DISPATCH=0` | `FAILED: should_use_unified(Q8_0) returned true, expected false`, `=== Results: 8/9 tests passed ===`, rc 1 | ⚠️ **Memory hazard**: no selector in the registration and the test uses `sycl::default_selector_v` — it will enumerate the iGPU (`llama.cpp-403s`, 231.7 GB of "VRAM"). Run as `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`. Do not leave the mutation applied across a benchmark. |
| `test-q6k-dispatch` | `vecdotq.hpp:105` `byte_sub_4(vil1 \| vih1, 0x20202020)` → `0x21212121` | HEADER-WIDE | GPU | COUNT `Results: %d passed, %d failed, %d skipped` | tests 1/2 `FAIL: <n>/<rows> rows violate the enforced CPU_Q8_1 (abs <= max(1e-3, 1e-4*abs(CPU_Q8_1))) contract`; test 4, if the corrupted kernel reaches it, prints its **own** string `FAIL: <n>/<rows> rows have >1% error`; test 3 stays green either way. Count not pre-registered — never executed | **Try Batch A's `--corrupt-q8-reference` first** — it needs no rebuild. Only spend this header build if you want the production-side proof too. ⚠️ This row said `Q8_1 contract` in tests 1/2/4 with `Results: 1 passed, 3 failed`. Corrected 2026-08-09 (`llama.cpp-u2mz`): the printed contract name is `CPU_Q8_1 (abs <= max(1e-3, 1e-4*abs(CPU_Q8_1)))`, and **test 4 cannot print that string at all** — it never calls `reference_contract_match`. The old count came from the Batch A row and does not transfer: that mutation corrupts the CPU **oracle** (tests 1/2 only), this one corrupts the production **kernel**, so their failure sets differ by construction. |
| `test-moe-mini-graph` | `dequantize.hpp:294` `v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;` → `* 0.55f` | HEADER-WIDE | GPU | PASSFAIL + `MoE mini-graph: nmse=%.6e max_diff=%.6f` | `FAIL: mismatch beyond tolerance (nmse>5.0e-04 or max_diff>1.00)`, rc 1 | ⚠️ **Reach unverified** — the test forces XMX-tiled MoE, which may decode MXFP4 without `dequantize_mxfp4`. Guaranteed-reach fallback (MEGA-TU): `ggml-sycl.cpp:37186` `dst[i] = scale*x[i] + bias;` → `* 1.01f` — the graph chains 6 `ggml_scale` nodes, 1.01^6 ≈ 1.062 → nmse ≈ 3.8e-3. ⚠️ **Vacuous-pass trap**: any failure inside the SYCL leg is reported as `SKIP: SYCL graph path unavailable or disabled` with **rc 0**. |
| `test-fattn-packed-k-lifecycle` | `fattn.hpp:126` `address >= begin && address < begin + size;` → `<= begin + size` | HEADER-WIDE | GPU nominally — **but the mutated check runs in `verify_host_boundaries()` before `preflight_device()`, so the RED reproduces with no GPU** | PASSFAIL, `PASS checkpoint=%s shape=... async_wait_failures=%d`; 77 on no device | `FAIL host packed-K boundary checks: host end-exclusive range arithmetic included its end`, rc 1, for **all five** registered checkpoints | 5 ctest names, one mutation. Cheapest GPU-family RED in the plan because the host prologue precedes the device gate. |
| `test-sycl-fattn-xmx-policy` | `fattn-xmx-f16.hpp:178` `return XMX_BATCH_KV_LARGE;` → `/ 2` | narrow header (only `fattn.cpp` + a bench + this test) | CPU-ONLY | PASSFAIL, `SYCL fattn policy tests: %s`; per-check `FAIL: <name> got %d want %d` | `FAIL: D128 ncols8 uses larger batch when local memory permits got 24 want 48` then `SYCL fattn policy tests: FAIL`, rc 1 | Fires 1 of ~35; `ok &=` so nothing is masked. Good "some green, some red" demonstration. |
| `test-dmmv-coalesced-q4-0-oracle` | `dmmv-coalesced-q4-0-layout.hpp:64` `blocks_per_row * DMMV_COALESCED_Q4_0_BLOCK_BYTES` → `* 32` (the literal Q8_0-stride defect the file is named for) | narrow header (convert.cpp + dmmv.cpp + 2 tests) | CPU-ONLY | COUNT — per-check `PASS`/`FAIL` lines + `%d check(s) failed` / `all checks passed` | exactly 4 of 20: `FAIL q8-row-stride-is-twice-the-q4-0-row q8=2048 q4_0=2048`, `... writes-past-the-allocation oob_writes=0`, `... produces-a-different-layout first_bad=-1`, `FAIL device-soa-writer-matches-contract first_bad=...`; `4 check(s) failed`, rc 1 | Round-trip and all 12 oracle checks stay green because writer and reader share the header — that is the point. Built-in positive control (`contract-oracle-catches-corrupted-kernel`): a fully-red run means you over-mutated. |

## Batch I — MEGA-TU `ggml-sycl.cpp` and headers only it includes (23 tests, 28 builds, ~9 h)

Do this last. Every build here is the 100k-line TU. One row is blocked
(`test-sycl-module-dlopen`, kept here as its build slot is real once the DL
config exists).

⚠️ **Four rows were removed from this batch on adjudication** — see *The
build-slot rule* below. `test-sycl-moe-glu-q8-artifact-policy`,
`test-sycl-moe-glu-q8-fused-store-policy`,
`test-sycl-moe-direct-final-token-major-bridge` and
`test-sycl-moe-token-major-metadata` each mutate a **test-only mirror with zero
production callers**, so a ~20-minute mega-TU build would prove only that a
mock can fail. They moved to the mock/deletion bucket.

| test | mutation (`ggml/src/ggml-sycl/ggml-sycl.cpp:` unless noted) | runtime | verifiability | expected RED | notes |
|---|---|---|---|---|---|
| `test-sycl-moe-xmx-tiled-materialization` | `:2734` drop `&& target_layout == GGML_LAYOUT_XMX_TILED` from `..._single_xmx_chunked_fallback_policy` | CPU-ONLY | PASSFAIL | `FAIL: tests/test-sycl-moe-xmx-tiled-materialization.cpp:161: non-XMX target must not use chunked fallback`, rc 1 | ⚠️ **Partial mock**: `test_moe_xmx_tiled_materialization_invariants` (`:24024`) has **zero production callers**, so 3 of its 5 test functions prove nothing. Only the two policy delegations are real — mutate within those. |
| `test-sycl-moe-expert-parallelism` | `:2820` `(int64_t(layer_id) << 32) \| int64_t(uint32_t(expert_id))` → `int64_t(layer_id) + int64_t(expert_id)` | DEVICE-FREE-THREADS | COUNT `=== Results: %d/%d passed, %d failed ===` | `FAIL: rank should be unique per entry` + `FAIL: test_popularity_key_uniqueness`, `=== Results: 6/7 passed, 1 failed ===`, rc 1 | Sound test of a real production structure. Its thread-safety subtest is weak (the reader only range-checks), so a green there is not evidence about the shared_mutex. |
| `test-tensor-placement` | `:8855` `if (strstr(name,"attn_") \|\| strstr(name,"ffn_"))` → drop the `ffn_` term | CPU-ONLY | bare `assert()` — aborts on the first | `Assertion 'ggml_sycl_classify_tensor("blk.10.ffn_down.weight") == 0' failed.`, **rc 134** | Fires 2 of 9 asserts but only one line prints. The file `#undef NDEBUG`s at line 14 *before* re-including `<cassert>` and records a verified mutation in its header — **do not remove that `#undef`**. |
| `test-cross-model-weight-usage` | `:11833` `it->second = tensor_usage::UNKNOWN;` → `it->second = mapped;` | **GPU** (hint was wrong — the plan stage inits a backend per device) | COUNT `=== %d checks, %d failures ===` (14) | `=== 14 checks, 2 failures ===` + `FAIL: check 2: model A's own tied-weight case forces UNKNOWN (precondition for check 4)` + `FAIL: check 5: A->B->reactivate-A lookup restores A exact same-name usage`, rc 1 | Has a built-in positive control (check 1) and negative control (check 3); this mutation leaves both green deliberately. The alternative "key by bare name" at `:11822` breaks 4 of 6 and destroys the controls. |
| `test-sycl-weight-key-stability` | `:11996` `id.file_offs = has_gguf_identity ? identity.file_offs : 0;` → `= 0;` | CPU-ONLY | PASSFAIL, `PASS: weight cache key remained stable` | `FAIL: cache key missing GGUF identity fields`, rc 1 | Only 3 checks total, so no partial-red outcome is available. Cheap to evaluate, expensive to build. |
| `test-sycl-moe-same-expert-grouping` | `:24339` `has_lane_filled_group \|\| count > 1` → `count >= 1` | CPU-ONLY | PASSFAIL | `FAIL: tests/...-same-expert-grouping.cpp:187: all-singleton grouping must fail closed so runtime falls back`, rc 1 | ⚠️ Its case 4 is a **source-text grep** of `mmvq.cpp` for three literals — it cannot fail behaviourally and passes even if the surrounding code is deleted. Needs the repo tree present at runtime. ⚠️ **WEAKEST-COVERAGE ROW IN BATCH I** (adjudicated 2026-08-08, KEEP): exactly one thin production-reaching function, evidenced only by those 3 `contains()` checks. Revisit when the token-major bridge integration wires its mocks to production; first candidate to drop if it does not. |
| `test-xmx-moe-mxfp4` | `:24489` in `test_moe_gateup_singlecol_policy` delete the `return out;` in the `graph_recording` arm | CPU for the 3 reachable subtests (they run *before* the device gate) | COUNT `Tests run: %d, Passed: %d, Skipped: %d, Failed: %d`; 77 after the host tests | `FAILED: graph recording must reject first implementation`, `Tests run: 3, Passed: 2, Skipped: 0, Failed: 1`, rc 1 | **BLOCKED** by `GGML_SYCL_BUILD_XMX_TESTS` (see below). Note a 77 here still means 3 real assertions ran. |
| `test-sycl-moe-sequence-graphlet-policy` | `:24797` `if (unsafe_fused_q8_requested)` → `if (false && ...)` in `moe_default_fast_path_policy_from_flags` | CPU-ONLY | PASSFAIL | `FAIL: tests/...-sequence-graphlet-policy.cpp:939: known unsafe fused-Q8 request must quarantine default sequence replay`, rc 1 | 47 of its 49 assertions are source-text greps; the two truth-table cases are the only behavioural coverage — but they **do** reach production (`:24775`), unlike the glu-q8 sibling. Because most cases read source at runtime, a text-only mutation reddens it with no rebuild. |
| `test-unified-cache-integrity` | `:29105` `if (!host_buffer && !sycl_buffer)` → `if (!host_buffer)` | GPU (768 staged weights, 128 MB ctx) | PASSFAIL, `Unified cache integrity test: %s` | `Failed to cache layout for type=2 idx=0` then `... FAIL`, rc 1 | Clean phase-1-green / phase-2-red split. ⚠️ Do **not** mutate `unified_cache::validate()` — that mutates the checker, not the property. The "on-topic" `id_to_key_[key] =` deletion at `unified-cache.cpp:4034` is unreliable (five other write sites can repopulate) — demand a positive control if you use it. |
| `test-sycl-kv-planned-device-materialization` | `:31532` `la.on_device && la.owner_device >= 0 ? la.owner_device : ctx->device` → `ctx->device` | **GPU, needs ≥2 physical devices** (hint was wrong) | PASSFAIL, `SYCL planned-device KV materialization test: PASS` | `  FAIL: cache_k_l1 smart handle for planned device 1 must resolve`, rc 1 | The host-KV-zone rollback half and the registry-owner checks stay green. ⚠️ Do not mutate `planned_owner` at `:32218/:32260` — `valid_device_kv_handle` then refuses and the run collapses into a generic assert. With <2 devices it prints `SKIP: need at least two physical SYCL devices` and returns **0**, not 77. |
| `test-mul-mat-host-streaming` | `:38382` `float * dst_ptr = ctx->dst_dd_i + row_start;` → `= ctx->dst_dd_i;` | GPU | PASSFAIL + `mul_mat host streaming: nmse=%.6e max_diff=%.6f` | `FAIL: mismatch beyond tolerance (nmse>5.0e-05 or max_diff>0.01)`, rc 1 | ⚠️ **Control first**: confirm the sliced path is taken (`GGML_SYCL_MUL_MAT_STREAM_DEBUG=1` → `[MUL_MAT_STREAM] row_start=... row_count=...`, a DEBUG-level line) or the mutation is inert. False green: `SKIP: Could not initialize SYCL backend` returns 0. |
| `test-cpu-gpu-soa-interaction` | `:46423` `if (cols % block_elements != 0)` → `if (false)` in `ggml_sycl_reorder_expected_size` | GPU | COUNT `Results: %d passed, %d failed` + per-test `Results: %d/10` | `FAIL: reorder geometry accepted non-representable dimensions or unsupported type` then `Results: 5 passed, 1 failed`, rc 1 | **Has a built-in positive control**: `--corrupt-post-copy` emits `[SOA-REACH]`/`[SOA-POSITIVE-CONTROL]` and rc 1 — run that first to prove reach. RUN_SERIAL with `GGML_SYCL_VRAM_ARENA=0;GGML_SYCL_ASYNC_MEM=1`. |
| `test-layout-bytes` | `:47060` MXFP4 arm `const size_t scale_bytes = nblocks;` → `nblocks * 2;` | GPU | PASSFAIL, `Layout bytes test: PASS/FAIL` | `MXFP4 coalesced bytes: expected 340, got 360` then `Layout bytes test: FAIL`, rc 1 | Q4_0/Q8_0/Q6_K and the XMX-tiled case stay green (`ok &=`). ⚠️ `SKIP: SYCL backend unavailable` returns **0**. Do **not** pick the Q8_0 case: it calls the production `ggml_sycl_q8_0_coalesced_row_quants_bytes()` on both sides, so a mutation there moves both and cannot fail. |
| `test-sycl-kernel-selection` | `:52428` `{ "MMQ_COALESCED", ...MMQ_COALESCED }` → `...MMQ_SOA` | GPU-optional (4 of 6 subtests device-free) | COUNT `%d/%d tests passed` (hint said PASSFAIL — wrong) | `FAIL (parsed wrong kernel)` from `test_parse_force_kernel_valid`, `5/6 tests passed`, rc 1 | ⚠️ Device-less, the two backend subtests print `SKIP (SYCL backend unavailable)` and **return true**, counting as passes — a 6/6 green may have exercised only 4. |
| `test-sycl-orchestrator` | `:53055` `if (kernel == ggml_sycl_mul_mat_kernel::ONEDNN_AOS)` → `if (false && ...)` | GPU | COUNT `%d/%d tests passed` (total hardcoded 5) | `  test_tuning_prefers_onednn: FAIL (expected onednn path selection)`, `4/5 tests passed`, rc 1 | It sets `GGML_SYCL_UNIFIED_DISPATCH=0` before init, so it exercises the **legacy** selection arm — do not mutate the unified early-return at `:52991`, it is unreachable here. `SKIP (SYCL backend unavailable)` returns **0**. |
| `test-rms-norm-mul-add-broadcast` | `:77372` in `ggml_sycl_fusion_operand_view_offset_safe`, the `view_src != nullptr && accumulated_view_offs != 0` arm `return false;` → `return true;` | GPU | COUNT `=== %d checks, %d failures ===` + per-case detail; 77 on no device | `  [FAIL] ncols=64 view-at-nonzero-offset (root cause) (... max_err=<O(1)>, tol=1.0e-03)` + the ncols=256 twin, `=== 6 checks, 2 failures ===`, `SOME CHECKS FAILED`, rc 1 | **Pre-validated**: the test's own comment (lines 371–378) prescribes exactly this A/B. Cases A/B are its declared positive control. |
| `test-sycl-device-uuid-api` | `:97917` inside `if (dev == nullptr \|\| uuid == nullptr)` `return false;` → `return true;` | CPU-ONLY for the assertions (but `ggml_backend_sycl_reg()` runs discovery — source oneAPI) | PASSFAIL — silent on success | `UUID registry procedure missing or accepted null device`, rc 1 | ⚠️ Do not mutate the `ggml_backend_dev_backend_reg(dev) != ggml_backend_sycl_reg()` identity guard instead — it would `static_cast` a CPU device context and read `ctx->device`, i.e. UB/segfault rather than a clean assertion. |
| `test-sycl-module-dlopen` | `:100006` `return 1u; // GGML_BACKEND_LIFETIME_POLICY_PROCESS` → `return 2u;` | CPU-ONLY | PASSFAIL — silent on success | `missing PROCESS_LIFETIME policy export`, rc 1 | **BLOCKED** — registered under `BUILD_TESTING AND GGML_BACKEND_DL`, and `GGML_BACKEND_DL` is OFF. Needs a separate `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` tree. Deprioritise. |
| `test-sycl-onednn-packed-cache` | `:26922` `... && onednn_pack_m > 0) ?` → `< 0) ?` | GPU | PASSFAIL, `PASS: onednn packed layout cached and AoS evicted` | `FAIL: expected onednn pack_m=32, got=0`, rc 1 | ⚠️ **Worst false-green in the population: six `SKIP:` paths all `return true` → exit 0**, including `SKIP: SYCL backend unavailable` and `SKIP: unified cache unavailable`. Grep for the `PASS:` line, never for rc 0. |
| `test-moe-bias-owner-state` | `moe-bias-state.hpp:193` in `clear()` `act_alpha = 0.0f;` → `act_alpha = act_alpha;` | MEGA (only `ggml-sycl.cpp` includes it) | CPU-ONLY | COUNT on pass only — `moe bias owner state: PASS (12 cases)` | `FAIL: activation-variants: the second model inherited the first model's swiglu alpha/limit`, rc 1 | Its CMake comment states outright that it never enumerates a device. Contains a real positive control and four `static_assert`s no runtime mutation can reach. |
| `test-moe-layer-ids-worker-reset` | `moe-layer-ids-cache.hpp:32` `entry.ids_host.clear();` → `entry.ids_host = std::vector<int32_t>();` | MEGA (only `ggml-sycl.cpp` includes it) | DEVICE-FREE-THREADS | PASSFAIL, `two-worker TLS graph-boundary reset passed 1000 iterations` | `FAIL: IDs vector capacity was released`, rc 1 | Fires 1 of 11 per-entry checks — move-assign frees the buffer so capacity drops while `empty()` still holds, leaving all four emptiness checks green. Do not use `cache.clear()` — it blows up size/bucket/address checks at once. |
| `test-tiered-dispatch` | `tiered-plan-clear.hpp:13` `if (!has_plan \|\| !multi_device)` → `\|\| multi_device` | MEGA (sole includer is `ggml-sycl.cpp:320`) | GPU (3 load-bracket transactions, no GGUF) | COUNT `%d tiered-dispatch check(s) failed` / `All tiered-dispatch checks passed`; returns **77** for both skips | `FAIL: single-device plan was reported as cleared`, `FAIL: single-device plan was cleared`, `FAIL: completed multi-device plan was not cleared`, the three `PASS: <label>` placement lines, `3 tiered-dispatch check(s) failed`, rc 1 | 3 of 5 contract checks fire and the whole (expensive) placement half stays green — a precise count signal. Every alternative target lives in `ggml-sycl.cpp` anyway, so **batch this with the KV-materialization row**. |
| `test-moe-discovery-owner-state` | `moe-discovery-state.hpp:57` `key.slot < moe_discovery_max_owners && key.slot_generation != 0;` → drop the generation clause | header is MEGA-reaching, **but this test target links only `Threads::Threads`** — `ninja test-moe-discovery-owner-state` rebuilds one file | DEVICE-FREE-THREADS | COUNT `moe discovery owner state: PASS (15 cases)` | `FAIL: invalid-owner-is-inert: a slot with no generation was accepted as an owner identity`, rc 1, PASS line absent | **Build only the test target** and this becomes a Batch-B-class cost. Its fixture is a recording stand-in for the process globals, but the registry under test is production. |

---

# REMEDIATION PASS (2026-08-08) — what was executed against the three trailing groups

Phase A's headline finding was that a fifth of the registered SYCL C++ suite
either cannot go red, tests a copy of itself, or is not built. This pass carried
out the spec's own remedy — *fix it or delete it* — on all of it. Nothing below
was built, run on a GPU, or ctest-ed; see **Pre-registered for the lead** at the
end of this section for what still has to be executed.

⚠️ **The set is 29, not 25.** The "25 findings" figure comes from the first
revision of the Phase A comment (7 cannot-fail + 6 mocks + 12 blocked). The fix
commit `4cd2145bb` moved four tests from Batch I into the mock bucket, taking it
to 6 → 10, so the correct total against the merged plan is
**7 + 10 + 12 = 29**. The doc body and appendix were already consistent at 29;
only the summary sentence lagged.

## Dispositions executed

| test | Phase A group | disposition |
|---|---|---|
| `test-tensor-classification` | CANNOT-FAIL | **FIXED** — `#undef NDEBUG` + re-include, 41 assertions now live |
| `test-sycl-esimd-float-atomic-compile` | CANNOT-FAIL | **FIXED** — reads back the atomic (`*ptr == 1.0f`); skips exit 77 |
| `test-sycl-onednn-mxfp4-feasibility` | CANNOT-FAIL | **RECLASSIFIED** `PROBE` — exits 77 when it could not probe |
| `test-xmx-host-streaming` | CANNOT-FAIL | **FIXED** — registration propagates the two options; skips exit 77 |
| `test-sycl-lifecycle-public-api` | CANNOT-FAIL | **RECLASSIFIED** `COMPILE-GATE` — source + registration documented |
| `test-unified-kernel-persistent` | CANNOT-FAIL | **REPAIRED** (unverified) — now `XMX-BLOCKED`; it was the suite's blocker |
| `test-xmx-kernel-config` | CANNOT-FAIL | **DELETED** — all `static_assert`, and over a dead header chain |
| `test-kernel-dispatch` | MOCK | **DELETED** — drifted mirror of `select_kernel_path` |
| `test-onednn-fallback` | MOCK | **DELETED** — `onednn-fallback.hpp` reaches no production TU |
| `test-sycl-pp-moe-scratch-lifecycle` | MOCK | **DELETED** — zero project headers; one case asserts `++` increments |
| `test-xmx-quant-loaders` | MOCK | **DELETED** — terminated include chain |
| `test-xmx-unified-kernel` | MOCK | **DELETED** — compares a duplicate of itself |
| `test-sycl-moe-glu-q8-artifact-policy` | MOCK | **DELETED** — condemned four, production-unreachable helper |
| `test-sycl-moe-glu-q8-fused-store-policy` | MOCK | **DELETED** — same |
| `test-sycl-moe-direct-final-token-major-bridge` | MOCK | **DELETED** — same |
| `test-sycl-moe-token-major-metadata` | MOCK | **DELETED** — same |
| `test-sycl-level-zero-vmem-feasibility` | MOCK | **KEPT**, reclassified `PROBE` — see the correction below |
| the 12 `GGML_SYCL_BUILD_XMX_TESTS` targets | XMX-BLOCKED | **UNBLOCKED (route a) + made visible (route b)** |
| `test-mmvq-q8-0-streaming-bench` (bare ctest name) | — | **DELETED** — no ENVIRONMENT, so it could only ever skip |

Commits: `b453f9a8d` (fixes), `059d28670` (deletions), `2414b5948` (XMX).

## The XMX group — both routes taken

**(a) The blocker is repaired.** `test-unified-kernel-persistent` was the single
target that stopped `GGML_SYCL_BUILD_XMX_TESTS=ON` building, so one rotted file
held 13 siblings hostage. The drift turned out to be five call sites and one
struct's fields, and the CMake comment describing it was wrong in two ways — see
*Corrections* below. ⚠️ **The repair is UNVERIFIED**: this lane cannot build.

**(b) The absence is now visible.** `ctest -R` cannot be made to fail on a
zero-match, so the fix is to make the match non-zero. With the option OFF, all 14
guarded names are registered as `DISABLED` placeholders, so `ctest -N -R
xmx-config` prints the test and a run reports it under *"The following tests did
not run: … (Disabled)"* rather than *"No tests were found"* and exit 0. A
`DISABLED` test's COMMAND never executes, so it costs nothing and needs no
target. The two name lists are machine-checkable against each other from the
CMake file itself; the check is written into the comment beside them.

## Corrections to the Phase A plan, found while executing it

1. **`test-tensor-classification` has 41 bare assertions, not 33** — 20 + 8 + 5 +
   3 + 5 across its five cases. The disposition is unchanged.
2. **`StridedCopyMeta` vs `SetRowsMeta` were conflated.** The inherited claim was
   that the test writes `meta.pad` on a struct with no such member at lines
   1365/1474/1952. **`SetRowsMeta` does have `pad`**, so 1365 and 1952 were always
   legal; only 1474 was not. The compile-failure disposition survives on
   `type_size` and the five pointer-vs-reference mismatches alone.
3. **The `add_set_rows`/`add_strided_copy` fix is the opposite of the obvious
   one.** The meta is no longer read through a device pointer — it is *copied
   into* the `OperationDescriptor` (`op.has_embedded_meta = true`,
   `unified-kernel.cpp:8600` and `:8624`). So the device staging buffers are
   deleted rather than dereferenced.
4. **`test-sycl-level-zero-vmem-feasibility` already exits 77 and its
   registration already carries `SKIP_RETURN_CODE 77`.** The `llama.cpp-k208`
   note repeated here ("skips render as FAILED") is stale — it was fixed, with a
   comment distinguishing genuine absence (77) from a Level Zero call that should
   have worked and did not (1). Nothing to do; the test is kept and reclassified
   as a `PROBE` because it contains no llama.cpp code.
5. **The `ggml_sycl_test_extract_layer_index` extern claim is dead, settled
   against the artifact.** `nm -D --defined-only build/bin/libggml-sycl.so`
   reports `T _Z34ggml_sycl_test_extract_layer_indexPKc` — global, exported, and
   mangled exactly as the test declares it.
6. **`tests/test-sycl-pp-moe-scratch-source.py` carried a tautology.** Its
   arena-generation check read
   `sycl_cpp.count("arena_generation_bump") >= 0 and cache_cpp.count(...) >= 3`.
   The first clause is true for every possible input, and that symbol does not
   occur in `ggml-sycl.cpp` at all (it lives in `unified-cache.hpp:2456` with 3
   call sites in `unified-cache.cpp`). Removed; the informative clause is kept.

## ⚠️ Deferred dead code — MUCH smaller than the handoff assumed

The handoff asked for the four condemned `test_moe_*` function bodies in
`ggml-sycl.cpp` to be recorded as "dead code pending lane pass". **Three of the
four are not dead**, because tests this pass KEEPS still call them. Deleting them
would break the build or redden a live test:

| helper | definition | still consumed by |
|---|---|---|
| `test_moe_down_sum_direct_final_policy` | `ggml-sycl.cpp:24071`, decl `ggml-sycl-test.hpp:202` | `tests/test-sycl-moe-fused-down-sum-policy.cpp` (**26** call sites), `tests/test-sycl-moe-same-expert-grouping.cpp:215` |
| `test_moe_build_token_major_metadata` | `ggml-sycl.cpp:24191`, decl `ggml-sycl-test.hpp:238` | `tests/test-sycl-moe-same-expert-grouping.cpp:81` |
| `test_moe_glu_q8_artifact_policy` | `ggml-sycl.cpp:24403`, decl `ggml-sycl-test.hpp:323` | `tests/test-sycl-moe-fusion-noactivation.cpp:77` — a **source-text** assertion, `contains(sycl, "test_moe_glu_q8_artifact_policy")`, "helper must exist" |
| `test_moe_glu_q8_fused_store_policy` | `ggml-sycl.cpp:24502`, decl `ggml-sycl-test.hpp:378` | **nothing** — genuinely unreferenced now |

So the lane pass has exactly **one** mechanical deletion available:
`test_moe_glu_q8_fused_store_policy`, together with its declaration and the
`test_moe_glu_q8_fused_store_*` input/result/counter types and the
`test_moe_glu_q8_kernel_path` enum around `ggml-sycl-test.hpp:352-380`. Confirm
the exact bounds at deletion time rather than trusting these line numbers, which
drift with every commit to that file.

**The wider finding this exposes, for the lead rather than for a lane owner:**
two tests that KEEP a Phase B build slot drive production-unreachable helpers —
`test-sycl-moe-fused-down-sum-policy` (Batch E) is **26** calls into
`test_moe_down_sum_direct_final_policy`, and `test-sycl-moe-same-expert-grouping`
(Batch I) calls two of them. The build-slot rule that condemned the four deleted
tests applies to those calls too.

**ADJUDICATED 2026-08-08 (spec review of the remediation): BOTH TESTS KEEP.** The
answer is mixed rather than clean, which is why it needed adjudicating:

- **`test-sycl-moe-fused-down-sum-policy` — KEEP, with its mutation pinned.**
  5 of its 17 test functions are production-reaching, traced through the
  memoized wrappers in `mmvq.cpp` (~`:214-364`) to the dispatch sites at
  `:19184-19208`. Its Batch E mutation **must target those five**; a mutation
  aimed at any of the other twelve would redden a mock and prove nothing. The 12
  pure-mock functions are scope-reduction candidates for a later pass, not a
  deletion this ticket makes.
- **`test-sycl-moe-same-expert-grouping` — KEEP, marginal.** Only one thin
  production-reaching function, and its evidence is 3 source-text `contains()`
  checks. Recorded as **the weakest-coverage row in Batch I**. Revisit when the
  token-major bridge integration wires its mocks to production; if that work
  does not land, this row is the first candidate to drop.

## Orphaned by the deletions — not this lane's files

Deleting the tests left three headers with no includer anywhere in the tree.
They are backend sources, outside this lane's scope:

- `ggml/src/ggml-sycl/onednn-fallback.hpp`
- `ggml/src/ggml-sycl/xmx-unified-q4-kernel.hpp` — and, through it,
  `xmx-kernel-config.hpp` and `xmx-quant-loaders.hpp`, whose only remaining
  includer is that header
- `ggml/src/ggml-sycl/xmx-esimd-gemm.hpp` and `xmx-esimd-gemm-q4.hpp` were
  *already* included by nothing before this pass; `xmx-esimd-common.hpp` is
  reachable only through them

## Pre-registered for the lead

Nothing in this pass was built or run. In order:

```bash
source /opt/intel/oneapi/setvars.sh --force

# 1. CONFIGURE + BUILD -- the decisive check. Ten deleted sources and six removed
#    CMake blocks are exactly what breaks a configure.
./scripts/sycl-build.sh -r

# 2. The deletions must have left no dangling ctest names.
ctest --test-dir build -N | grep -cE 'kernel-dispatch|onednn-fallback|xmx-kernel-config|xmx-quant-loaders|xmx-unified-kernel|pp-moe-scratch-lifecycle'   # want 0
ctest --test-dir build -N -R 'xmx-config'      # want 1 test, shown as Disabled -- NOT "No tests were found"

# 3. The route-(a) claim, which is UNVERIFIED and the most likely to fail.
cmake -B build-xmx -G Ninja -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_BUILD_XMX_TESTS=ON
ninja -C build-xmx test-unified-kernel-persistent
#    If it still does not compile: fix it forward. Do NOT comment the target out.

# 4. RED/GREEN on the two repaired tests. GREEN first, then the mutation.
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ctest --test-dir build \
  -R '^(test-tensor-classification|test-sycl-esimd-float-atomic-compile)$' --output-on-failure
```

**RED controls to run for step 4** — a fix that makes a test *able* to fail is
worth nothing until something has made it fail:

| test | mutation | expected |
|---|---|---|
| `test-tensor-classification` | `tests/test-tensor-classification.cpp:17` `== tensor_class::EMBEDDING` → `== tensor_class::OUTPUT` | abort, rc **134**, no `All tensor classification tests PASSED! (41 live assertions)` line. Before this pass the same edit exited **0**. |
| `test-sycl-esimd-float-atomic-compile` | in the kernel, `simd<float, 1> value = 1.0f;` → `= 0.0f` | `FAIL: ESIMD fadd atomic did not apply: expected 1.0, observed 0.000000`, rc **1**. Before this pass: rc 0, `PASS:` printed. |

Grep the printed `PASS:`/`FAIL:` strings, never `rc 0` — both of these exit 77 on
a missing device, and `test-sycl-esimd-float-atomic-compile` uses
`default_selector_v`, so **pin the selector** or it will pick the iGPU
(`llama.cpp-403s`).

`test-xmx-host-streaming` has no RED control here: it needs
`-DGGML_SYCL_XMX_GEMM=ON -DGGML_SYCL_MMQ_XMX=ON`, and confirming it now SKIPs
with 77 rather than passing with 0 in the default build is the check that
matters.

Format: `clang-format-19 --dry-run -Werror` was run on every touched source. All
report pre-existing whole-file drift on lines this pass did not touch; **no line
added by this pass is reformatted** by clang-format — verified per file by
diffing the formatted output and grepping for the added text.

Structure of `ggml/src/ggml-sycl/CMakeLists.txt`: **balanced under a
comment-aware `if`/`foreach` stack check at every commit in the range**
(`fe48225ba`, `b453f9a8d`, `059d28670`, `2414b5948`, `b752806ce` — depth 0 at
EOF, zero mismatched closers in each). **Naive `grep -c` tallies are unreliable
in this file** and no specific pair of counts should be quoted: comment prose
contains `if (` tokens, and the file is long enough that a stale or mid-edit
sample is easy to take without noticing. Check the *property* (balanced), not a
*number*.

⚠️ **THIS DOC SUPERSEDES TWO FIGURES IN THE COMMIT MESSAGES.** Both are
write-up errors; no code or logic depends on either, and the commits are not
amended because re-writing mid-range messages churns SHAs the review already
cites.

| figure | where it is wrong | correct value |
|---|---|---|
| "25 call sites" / "25 calls" into `test_moe_down_sum_direct_final_policy` | `059d28670` message | **26** — `grep -c 'test_moe_down_sum_direct_final_policy(' tests/test-sycl-moe-fused-down-sum-policy.cpp` |
| "if/endif 219/219 (was 226/226, -7 for the removed blocks)" | `059d28670` message | **unreproducible; do not quote.** The same commands now yield 224→220. The substantive claim — that the file is balanced — is independently confirmed above, and is the claim that matters. |

---

## Tests that CANNOT be made to fail — deletion candidates per the spec

> **STATUS: REMEDIATED.** All seven are dispositioned in the table above — four
> fixed or reclassified, one repaired, one deleted, and
> `test-sycl-lifecycle-public-api` documented as a compile-time gate. The
> analysis below is retained as the evidence for those decisions.

Seven registrations, in three failure modes. Each is a *permanently green*
result that a merge certification would otherwise count as coverage.

**1. Every exit path returns 0.**

- **`test-sycl-esimd-float-atomic-compile`** — prints `PASS:` on success and
  `SKIP: ...` + `return 0` from all four catch blocks. It never reads back the
  value it atomically adds, so the atomic could add nothing and it still
  passes. Touches **zero** fork code (pure oneAPI/ESIMD). Its only real value
  is that it compiles, which a build gate already provides. Also uses
  `default_selector_v` with no selector in its registration → iGPU.
  *Fix: assert `*ptr == 1.0f` and exit 77 on skips; otherwise delete.*
- **`test-sycl-onednn-mxfp4-feasibility`** — a capability **probe**, not a test.
  Three `probe_*` functions each print "supported"/"unsupported" and return; main
  returns 0 on every path including engine-creation failure. Touches no fork code.
  *Fix: convert to a recorded-expectation gate (assert the observed support
  matrix so a oneDNN upgrade that changes the answer goes red), or unregister.*
- **`test-mmvq-q8-0-streaming-bench`** (the bare registration only) — has no
  ENVIRONMENT, so it always prints `SKIP: set GGML_SYCL_MMVQ_BENCH=1 to run`
  and exits 77. Permanently skipped. The three sibling names on the same binary
  are live. *Fix: give it the env or drop the registration.*

**2. Assertions erased at compile time.**

- **`test-tensor-classification`** — 33 bare `assert()`s and **zero**
  occurrences of `NDEBUG` in the file. The build is Release `-O3 -DNDEBUG`
  (`scripts/sycl-build.sh:150-151`), so every assertion is preprocessed away
  and `main`'s only reachable exit is `return 0`; the `catch` block is dead.
  ⚠️ The comment at `ggml/src/ggml-sycl/CMakeLists.txt:2155-2161` claims
  `50723136a` already added the `#undef NDEBUG` fix — **that claim is false for
  this file** (it appears to describe `test-tensor-placement.cpp`, registered
  directly below). *Fix: one line, `#undef NDEBUG` after the includes.*
- **`test-xmx-kernel-config`** — every check is a `static_assert` and every
  subtest is `return true;` after them, so `main` has exactly one reachable
  exit. Worse, a mutation makes the **build** fail while ctest then runs the
  **stale binary**, which prints `All tests PASSED!` rc 0. Two of eight subtests
  static_assert arithmetic on literals (`8*32==256`). *Fix: it is a compile-time
  gate — either accept that and unregister it as a runtime test, or give it
  runtime checks.*
- **`test-sycl-lifecycle-public-api`** — `int main() { return 0; }` plus 15
  `static_assert`s. This is legitimate (it pins public ABI signatures), but the
  mutation protocol differs: **build the target and read the compiler**, do not
  run ctest. Not a deletion candidate; a documentation candidate.

**3. The body is unreachable.**

- **`test-xmx-host-streaming`** — `main` opens with
  `#if !defined(GGML_SYCL_XMX_GEMM) || !defined(GGML_SYCL_MMQ_XMX)`. Both
  options default OFF **and even with both ON they are
  `target_compile_definitions(ggml-sycl PRIVATE ...)`, so they never reach this
  test target.** There is no `-D` combination a user can pass that executes one
  line of its body; fixing it requires a CMake change. Compounding it, the live
  body's failure paths print `SKIP:` and `return 0` rather than 77.
- **`test-unified-kernel-persistent`** — cannot be built at all. Behind
  `GGML_SYCL_BUILD_XMX_TESTS` (OFF), and with the guard ON it **fails to
  compile** against the current API: `StridedCopyMeta`
  (`unified-kernel.hpp:295-302`) has no `type_size` and no `pad`, yet the test
  writes `meta.type_size` (line 1473) and `meta.pad` (1365/1474/1952);
  `add_set_rows`/`add_strided_copy` now take `const SetRowsMeta &` /
  `const StridedCopyMeta &` while the test passes device pointers.
  `CMakeLists.txt:1216-1231` documents this as OPEN, and the mismatch is
  independently checkable against `unified-kernel.hpp`.
  ⚠️ **Correction to this plan's first revision:** it also repeated that
  comment's claim that the test declares an extern
  `ggml_sycl_test_extract_layer_index` "that exists nowhere in the tree".
  **That is false** — the symbol is defined at
  `ggml/src/ggml-sycl/ggml-sycl.cpp:89939` (added by `2c49f9e13`), so the
  extern at `test-unified-kernel-persistent.cpp:42` would resolve fine. The
  disposition is unchanged because it never depended on that claim; the stale
  CMake comment is logged as its own micro-finding under *Blocked*.
  ***This single rotted target is what blocks the entire XMX suite*** — see below.

## Tests that reach no production code (mocks) — repair or delete

> **STATUS: REMEDIATED.** Nine of the ten scored `MOCK` were deleted; the tenth,
> `test-sycl-level-zero-vmem-feasibility`, was kept and reclassified `PROBE`
> (its `SKIP_RETURN_CODE 77` defect had already been fixed — see *Corrections*).
> The `MOCK` group is now empty. Three of the four condemned helpers turned out
> **not** to be dead code; see *Deferred dead code* above before deleting
> anything from `ggml-sycl.cpp`.

These *can* be made to fail, but only by mutating the test's own copy of the
logic. A green from them says nothing about the shipped backend. Listed loudest
first.

⚠️ **This table is descriptive, not a scoring bucket** — the same convention as
*Blocked*, and it matters because **12 rows appear here while only 10 are
scored `MOCK`** in the appendix. A test can be both a mock and unbuildable;
`test-esimd-vectorized-dequant`, `test-moe-mxfp4-dp4a` and
`test-xmx-hardware-detect` are all three, and are scored `XMX-BLOCKED`.

**Scoring precedence, applied uniformly:** `XMX-BLOCKED` > `CANNOT-FAIL` >
`MOCK` > batch. The blocker wins because it is the operative fact — you cannot
build the target to confirm the mock diagnosis, so "repair the guard" precedes
"repair the test". `test-xmx-quant-loaders` is scored `MOCK` rather than
`XMX-BLOCKED` because it is registered **outside** the guard
(`ggml/src/ggml-sycl/CMakeLists.txt:3654`) and does build — it merely exercises
a header whose include chain terminates.

| test | what it actually exercises |
|---|---|
| `test-sycl-pp-moe-scratch-lifecycle` | Includes **zero** project headers; the CMake target has one source and `Threads::Threads`. Every type (`slot_backing`, `local_slot_state`, `model`) is defined in an anonymous namespace in the test. `test_arena_reset_destroy_generation_bump()` is literally `uint64_t g = 7; const uint64_t stale = g; g++; require(stale != g, ...)` — it asserts that `++` increments. Seven "cases" named after PP MoE scratch lifecycle, none of which can observe any change under `ggml/src/ggml-sycl/`. |
| `test-onednn-fallback` | `onednn-fallback.hpp` is included by **nothing** outside its own test. `execute_cached()` has every real parameter commented out and never calls oneDNN; `init()` takes an `int*` as the "queue". ~90 assertions over a class no shipped path can reach. |
| `test-kernel-dispatch` | Defines its own `namespace ggml_sycl_unified_test` copy of `KernelPath`, `XMXConfig`, `get_esimd_min_batch()` and `select_kernel_path()` ("mirror the definitions… without SYCL dependencies"), includes no project header, links nothing. **The mirror has already drifted** — production `select_kernel_path` has grown `ESIMD_LARGE_TILE` (`unified-kernel.cpp:4148`) that the copy does not model. |
| `test-xmx-unified-kernel` | Titled "matches outputs of all 7 original kernel variants", but `compute_reference_q4_0_q8_1<Config>` is a host loop whose own comment says it "computes the same result regardless of Config" — the template parameter is unused. It compares that against a line-for-line duplicate in the test file. No SYCL kernel, no `joint_matrix`, no `UnifiedXMXKernel::operator()` is ever invoked. |
| `test-esimd-vectorized-dequant` | Re-implements every dequant inline in its own ESIMD kernels and scores them against its own `reference_dequant_q4_0/q8_0`; its only non-system include is `sycl-test-skip.hpp`. |
| `test-moe-mxfp4-dp4a` | Redefines `block_mxfp4`, `block_q8_1`, `kvalues_mxfp4`, `dp4a_cpu`, `mxfp4_dot_*` locally; `target_link_libraries` is `IntelSYCL::SYCL_CXX` only — no `ggml-sycl`. |
| `test-sycl-level-zero-vmem-feasibility` | Contains no llama.cpp code at all — a Level Zero driver capability probe. Belongs in a driver preflight script. (Its skip/fail split is well done, but its registration lacks `SKIP_RETURN_CODE 77`, so skips render as FAILED — `llama.cpp-k208`.) |
| `test-xmx-hardware-detect` / `test-xmx-quant-loaders` | Both exercise `xmx-esimd-common.hpp` / `xmx-quant-loaders.hpp`, whose include chains **terminate** — no production TU pulls them in. The former additionally tests a `#ifdef XMX_TEST_STANDALONE` shim for its "hardware detection" half, and its `add_test` has no `set_tests_properties` at all: no `SKIP_RETURN_CODE 77`, no `LD_LIBRARY_PATH`. |
| `test-sycl-moe-glu-q8-artifact-policy` | `test_moe_glu_q8_artifact_policy` (`ggml-sycl.cpp:24403`) — definition plus a declaration in `ggml-sycl-test.hpp:323`, and **no caller anywhere**. Its body is a self-contained ladder over a `test_moe_glu_q8_artifact_input` struct of bools, incrementing `g_test_moe_glu_q8_*` counters; it calls no production function. Same shape `test-sycl-tensor-placement` was rewritten to escape. |
| `test-sycl-moe-glu-q8-fused-store-policy` | `test_moe_glu_q8_fused_store_policy` (`ggml-sycl.cpp:24502`, declared `ggml-sycl-test.hpp:378`) — same: definition + declaration, zero callers, and `test_moe_glu_q8_kernel_path_supports_fused_store` switches over a `test_moe_glu_q8_kernel_path` enum that exists only in the test block. |
| `test-sycl-moe-direct-final-token-major-bridge` | `test_moe_down_sum_direct_final_policy` (`ggml-sycl.cpp:24071`, declared `ggml-sycl-test.hpp:202`) — zero callers; the real direct-final gate is re-implemented inline at `:66656-66706` and shares no code with it. Compounding: `test_moe_token_major_metadata_is_complete` sits behind `GGML_SYCL_TEST_MOE_XMX_FUSED_HELPERS`, whose only definition in the tree is line 2 of this test file. |
| `test-sycl-moe-token-major-metadata` | `test_moe_build_token_major_metadata` (`ggml-sycl.cpp:24191`, declared `ggml-sycl-test.hpp:238`) — zero callers. Its overflow arithmetic is genuinely good and genuinely untested elsewhere, which argues for **promoting it to production** rather than deleting the test. |

### The build-slot rule (adjudication, `ona8` spec review 2026-08-08)

Four of these were originally given a Batch I build slot *and* flagged as mocks
— two instructions, one of which had to win. The rule that resolves it:

> **A build slot is only justified when the mutated line is production-reachable.**
> Mutating a mirror proves the mirror can fail, which nobody doubted. The u2mz
> spec's own remedy for a decorative test is to fix or delete it, not to spend
> ~20 minutes of mega-TU rebuild demonstrating its decorativeness.

Verified per function rather than assumed — each of the four is a definition in
`ggml-sycl.cpp` plus a declaration in `ggml-sycl-test.hpp`, with **no third
occurrence** in the 100k-line TU:

```bash
cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -c test_moe_glu_q8_artifact_policy       # 1 (the definition)
cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -c test_moe_glu_q8_fused_store_policy    # 1
cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -c test_moe_down_sum_direct_final_policy # 1
cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -c test_moe_build_token_major_metadata   # 1
```

⚠️ **The same check clears two neighbours that look identical and are not** —
name similarity is not evidence here, so run the grep rather than pattern-match
on the `test_` prefix:

- `ggml_sycl_moe_single_xmx_chunked_fallback_policy` (`:2734`, **3** occurrences)
  is production, called by `..._chunked_fallback_allowed`. So
  `test-sycl-moe-xmx-tiled-materialization` **keeps** its Batch I slot — with
  the mutation pinned to that line, not to the test-only
  `test_moe_xmx_tiled_materialization_invariants` (1 occurrence) that 3 of its
  5 test functions drive.
- `moe_default_fast_path_policy_from_flags` (`:24775`, **3** occurrences) is
  production, so `test-sycl-moe-sequence-graphlet-policy` keeps its slot too,
  even though 47 of its 49 assertions are source-text greps.

The lead's steer covered `test-sycl-moe-glu-q8-artifact-policy`; this pass
extended it to the three siblings **by the same rule**, because applying it to
one and not the others would reproduce exactly the inconsistency the review
caught. Push back if the three should keep their slots.
| partial: `test-q6k-reorder-dispatch` (1 of 8 subtests production), `test-sycl-moe-xmx-tiled-materialization` (2 of 5), `test-sycl-moe-direct-final-scratch-plan` (1 of 3, and that one is a text grep), `test-mmq-xmx-dispatch` (1 of 5), `test-sycl-moe-sequence-graphlet-policy` (2 of 49) | see the batch rows |

## Blocked: targets that do not exist in the current configuration

> **STATUS: REMEDIATED both ways.** The build break that made
> `GGML_SYCL_BUILD_XMX_TESTS=ON` fail is repaired (unverified — it has not been
> built), and all 14 guarded names are now registered as `DISABLED` placeholders
> when the option is off, so `ctest -R` on them can no longer exit 0 by matching
> zero tests. The group is 13 in the appendix rather than 12, because
> `test-unified-kernel-persistent` moved here out of `CANNOT-FAIL` once repaired.
> `test-xmx-kernel-config` did not: it was deleted, and it was never in this
> group anyway (it is registered outside the guard).

⚠️ **Read the counting convention first.** This section is a **cross-reference,
not an additive bucket.** Only the `GGML_SYCL_BUILD_XMX_TESTS` group of **12**
is scored here in the partition; the `GGML_BACKEND_DL` and pre-existing-RED
entries below name tests that are *already counted in Batch I and Batch H
respectively* and appear here only so a reader looking for blockers finds them.
Adding this section's items to the batch totals double-counts. The flat
appendix at the end is the authority: every test appears in it exactly once.

### Behind `GGML_SYCL_BUILD_XMX_TESTS` — 14 guarded, 12 scored here

⚠️ **Nobody in this chain has previously had the right number**, so it is
recounted here against the literal `if`/`endif` bounds rather than by eye. The
guard opens at `ggml/src/ggml-sycl/CMakeLists.txt:761`
(`if (GGML_SYCL_BUILD_XMX_TESTS AND GGML_SYCL_TARGET STREQUAL "INTEL")`, the
`option(... OFF)` being at `:759`) and its matching `endif()` is at `:1287`. It
contains **14** `add_executable` calls:

```bash
awk 'NR>=761 && NR<=1287 && /add_executable\(/' ggml/src/ggml-sycl/CMakeLists.txt | wc -l   # 14
```

The 14, with the disposition that keeps label = list = sum:

| # | target | scored under |
|---|---|---|
| 1 | `test-xmx-hardware-detect` | XMX-blocked |
| 2 | `test-xmx-esimd-basic` | XMX-blocked |
| 3 | `test-xmx-moe-mxfp4` | **Batch I** (its mutation is in `ggml-sycl.cpp`) |
| 4 | `test-moe-mxfp4-dp4a` | XMX-blocked |
| 5 | `test-mxfp4-vector-dequant` | XMX-blocked |
| 6 | `test-unified-kernel` | XMX-blocked |
| 7 | `test-xmx-compute` | XMX-blocked |
| 8 | `test-xmx-default-enable` | XMX-blocked |
| 9 | `test-xmx-optimization` | XMX-blocked |
| 10 | `test-xmx-config` | XMX-blocked |
| 11 | `test-esimd-vectorized-dequant` | XMX-blocked |
| 12 | `test-esimd-prefetch` | XMX-blocked |
| 13 | `test-unified-kernel-ops` | XMX-blocked |
| 14 | `test-unified-kernel-persistent` | **cannot be made to fail** (does not compile) |

**14 guarded = 12 XMX-blocked + 1 Batch I + 1 cannot-fail.** The prior revision
of this section said "Ten" in its label, enumerated 13 names, and summed 12 in
its table — three different wrong numbers for one set.

⚠️ **Turning the flag ON does not fix it — the build then fails** on the rotted
`test-unified-kernel-persistent` (see its entry above). Until that one target is
repaired or excluded, build **single targets** (`ninja test-xmx-esimd-basic`),
never `all` with the flag on.

⚠️ And right now, `ctest -R xmx-config` (or any of those 14 names) **exits 0 by
matching zero tests.** That is the ticket's own first trap, live, on fourteen
registrations.

⚠️ **The pre-existing CMake comment at `:1216-1231` is itself miscounted and
partly false** — a micro-finding in its own right, because it is the first
thing the next reader will consult:

- It says *"Twelve of the thirteen targets build."* There are **14** targets in
  the guard, so both figures are wrong (13 build, 1 does not).
- It says `test-unified-kernel-persistent` *"declares an EXTERNAL
  `ggml_sycl_test_extract_layer_index` in its non-standalone branch, and no
  such symbol exists anywhere in the tree."* **That symbol does exist**, at
  `ggml/src/ggml-sycl/ggml-sycl.cpp:89939`, added by `2c49f9e13` ("sycl: expand
  unified persistent TG graph integration") — confirmed with
  `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n ggml_sycl_test_extract_layer_index`
  (codescout is blind in that file, so the pipe-grep is the honest query). The
  comment simply predates the commit.

The **compile-failure disposition survives untouched** — it rests on the
`StridedCopyMeta`/`SetRowsMeta` mismatch, which is independently verifiable and
unaffected. Only the supporting extern claim was wrong, and this plan's first
revision carried it forward from the comment without checking it. Correcting the
CMake comment belongs to whoever owns
`ggml/src/ggml-sycl/CMakeLists.txt` (`impl-w1`).

### Cross-references (already counted elsewhere — do not add)

- **Behind `GGML_BACKEND_DL`** (OFF): `test-sycl-module-dlopen`. Needs a
  separate `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` build tree. **Counted
  in Batch I.**
- **Pre-existing RED**: `test-sycl-weight-key-uniqueness` — see its Batch H
  row; resolve the baseline before running the mutation or the experiment is
  void. **Counted in Batch H.**

## Cross-task file conflicts — flag, do not route around silently

Three mutation sites land in files another live task owns. **Do not apply
these while that task is in flight; coordinate through `main`.**

| file | tests affected | owner note |
|---|---|---|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | all 23 Batch I rows | W1 owns it concurrently per the u2mz ticket's own constraints section. This is the largest conflict in the plan. |
| `ggml/src/ggml-sycl/unified-cache.cpp` | all 14 Batch D rows | Same constraints section names W1 as the owner. |
| `ggml/src/ggml-sycl/CMakeLists.txt` | none of the mutations, but every *fix* implied by the Blocked and Deletion sections | `impl-w1`'s as of 2026-08-02 (c-2qqk ownership note). This pass is scoped to `docs/` only and touched nothing. |

The `mem-handle.cpp`, `mmvq.cpp`, `dmmv.cpp`, `fattn*.cpp` and standalone-TU
batches have no recorded concurrent owner.

## Execution order and total cost

Cheapest-per-test first, which also front-loads the tests most likely to
produce a clean, unambiguous RED:

**The partition exactly — machine-checked, no test in two groups and none
unassigned. 127 rows as planned in Phase A; 117 after the remediation pass,
whose deletions all came from the three trailing groups:**

| # | group | tests | builds | est. |
|---|---|---|---|---|
| 1 | A — no rebuild | 2 | 0 | ~10 min |
| 2 | B — standalone TUs | 19 | 34 | ~1.5 h |
| 3 | C — `mem-handle.cpp` | 5 | 6 | ~1.4 h |
| 4 | E — `mmvq.cpp` | 5 | 6 | ~1.4 h |
| 5 | D — `unified-cache.cpp` | 14 | 15 | ~3.5 h |
| 6 | F — other backend TUs | 11 | 20 | ~4 h |
| 7 | G — `common.hpp` | 5 | 6 | ~2.5 h |
| 8 | H — other shared headers | 14 | 24 | ~7 h |
| 9 | I — `ggml-sycl.cpp` + its private headers | 23 | 28 | ~9 h |
| — | ~~cannot be made to fail~~ → `REMEDIATED-B` | 2 | — | fixed; Phase B rows still to write |
| — | ~~cannot be made to fail~~ → `PROBE` / `COMPILE-GATE` / `XMX-OPT-GATED` | 4 | — | reclassified; not batch work |
| — | blocked by `GGML_SYCL_BUILD_XMX_TESTS` | 13 | — | not built by default; blocker repaired, unverified |
| — | ~~mocks outside a batch~~ | 0 | — | 9 deleted, 1 reclassified `PROBE` |
| | **total** | **117** | **~139** | **~30 h** |

⚠️ **These twelve rows changed in the remediation pass; the batch rows A–I did
not.** The 98 batch tests, their ~139 builds and the ~30 h estimate are
untouched — every deletion came from the three trailing groups, which were never
scheduled for Phase B. The trailing groups went 29 → 19 (ten deletions), so the
total goes 127 → 117. Arithmetic and the full group breakdown are in the
appendix.

Cost is dominated by builds, not runs: ~139 builds are the whole number, and 53
of the tests are CPU-only or device-free — those can be *run* by anyone.
Only the builds (shared `build/`) and the GPU runs are lead-serialised.

⚠️ **Counting convention, because the first revision of this plan got it
wrong.** These twelve rows are a **partition**: every one of the 117 tests
appears in exactly one, and the *Blocked* section further down is a
cross-reference view over them, not an additional bucket. Two entries there
(`test-sycl-module-dlopen`, `test-sycl-weight-key-uniqueness`) name tests
already scored in Batch I and Batch H; adding them again is the double-count
the `ona8` spec review caught. **The flat appendix at the end of this section is
the authority** — 117 lines, one test each, checkable with the command given
there (which includes a positive control, because a mis-anchored `sed` range
makes the duplicate check pass vacuously).

Of the 23 Batch I rows, **19 mutate `ggml-sycl.cpp` itself** and 4 mutate
headers that only `ggml-sycl.cpp` includes (`moe-bias-state.hpp`,
`moe-layer-ids-cache.hpp`, `tiered-plan-clear.hpp`, `moe-discovery-state.hpp`).
The last of those is cheap in practice — its test target links only
`Threads::Threads`, so `ninja test-moe-discovery-owner-state` rebuilds one file
and never touches the mega-TU; it is scored in I because the header's other
consumer is the mega-TU, but budget it as Batch-B-class work.

⚠️ **The three trailing groups were 29 of 127 — nearly one test in four.** That
is the headline finding of Phase A, and it is not an execution cost, it is a
coverage result: nearly a quarter of the registered SYCL C++ suite either could
not go red, tested a copy of itself, or was not built at all. (The often-quoted
"25" is from the first revision of the Phase A comment and predates the fix
commit that moved four tests from Batch I into the mock bucket.)

**All 29 have now been dispositioned** — see *REMEDIATION PASS* above. Ten were
deleted, six fixed or repaired, four reclassified, and the `ctest -R` exit-0
trap that was live on fourteen registrations is closed by `DISABLED`
placeholders.

⚠️ Two standing rules apply throughout, from CLAUDE.md:

- **Pin the selector on every run**: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1`
  (or `:1` for B50-only). Six of these tests use `default_selector_v` or
  self-default to device 0 with no selector in their registration, and the iGPU
  reports 231.7 GB of "VRAM" (`llama.cpp-403s`). `test-unified-cache-bugs` is
  additionally in the never-loop family (~8.5 GB RSS).
- **A green is not evidence until you know work happened.** Fourteen tests in
  this population print `SKIP: ...` and `return 0` on a missing device or
  backend. For each, grep for the test's own `PASS:` string, never for `rc 0`
  — and treat an implausibly short runtime as a signal.

## Appendix: the flat partition (117 lines, one test each)

**N changed from 127 to 117 in the remediation pass (2026-08-08).** The
arithmetic, so it is checkable rather than asserted:

```
127  Phase A population
- 10  deleted (9 MOCK + 1 CANNOT-FAIL) -- see "REMEDIATED" below
= 117
```

The ten removed are `test-kernel-dispatch`, `test-onednn-fallback`,
`test-sycl-moe-direct-final-token-major-bridge`,
`test-sycl-moe-glu-q8-artifact-policy`,
`test-sycl-moe-glu-q8-fused-store-policy`,
`test-sycl-moe-token-major-metadata`, `test-sycl-pp-moe-scratch-lifecycle`,
`test-xmx-quant-loaders`, `test-xmx-unified-kernel` (all `MOCK`) and
`test-xmx-kernel-config` (`CANNOT-FAIL`). One row is **renamed** rather than
removed: `test-mmvq-q8-0-streaming-bench` → `test-mmvq-q8-0-streaming-smoke`,
because the bare registration under the old name was dropped and the binary's
live ctest name is the smoke one. The binary itself is unchanged, so Batch D is
still 14.

Group counts after the pass, summing to 117:

```
A 2   B 19   C 5   D 14   E 5   F 11   G 5   H 14   I 23      = 98  (Phase B batches)
REMEDIATED-B 2   PROBE 2   COMPILE-GATE 1   XMX-OPT-GATED 1
XMX-BLOCKED 13                                                = 19  (not batch work)
                                                                117
```

`MOCK` and `CANNOT-FAIL` are now **empty** — every member was either deleted or
retagged. `XMX-BLOCKED` grew 12 → 13 because `test-unified-kernel-persistent`
moved out of `CANNOT-FAIL`: it was repaired, so it is no longer a test that
cannot fail, merely one that is not built by default.

Every registered SYCL C++ test in scope appears **exactly once**. Verify with:

```bash
# The table header is `| group | test |` -- NO backticks around the two words.
# Anchoring on a backticked header matches nothing and the check passes
# vacuously; that mistake was made and caught while writing this line.
sed -n '/^| group | test |/,$p' docs/backend/sycl-test-inventory.md \
  | grep -oP '^\| `[A-Z-]+` \| `\K[a-z0-9-]+' > /tmp/u2mz-names
wc -l < /tmp/u2mz-names        # must print 117
sort /tmp/u2mz-names | uniq -d # must print nothing
# positive control -- the duplicate check must be able to fire:
sort /tmp/u2mz-names <(echo test-cold-start) | uniq -d   # must print test-cold-start
```

| group | test |
|---|---|
| `B` | `test-cold-start` |
| `I` | `test-cpu-gpu-soa-interaction` |
| `I` | `test-cross-model-weight-usage` |
| `H` | `test-dmmv-coalesced-q4-0-oracle` |
| `F` | `test-dmmv-q6k-coalesced` |
| `B` | `test-esimd-dpas-gate` |
| `XMX-BLOCKED` | `test-esimd-prefetch` |
| `XMX-BLOCKED` | `test-esimd-vectorized-dequant` |
| `H` | `test-fattn-packed-k-lifecycle` |
| `F` | `test-fattn-thread-local` |
| `H` | `test-ggml-sycl-soa` |
| `B` | `test-gpu-arch` |
| `I` | `test-layout-bytes` |
| `D` | `test-mem-handle-eviction` |
| `C` | `test-mem-handle-wrong-device` |
| `F` | `test-mem-ops` |
| `F` | `test-mmq-q6k-gpu` |
| `B` | `test-mmq-xmx-dispatch` |
| `B` | `test-mmvq-launch-geometry` |
| `D` | `test-mmvq-q8-0-streaming-smoke` |
| `I` | `test-moe-bias-owner-state` |
| `B` | `test-moe-control-plan` |
| `I` | `test-moe-discovery-owner-state` |
| `I` | `test-moe-layer-ids-worker-reset` |
| `H` | `test-moe-mini-graph` |
| `E` | `test-moe-mul-mat-id` |
| `E` | `test-moe-mul-mat-id-q4q8` |
| `XMX-BLOCKED` | `test-moe-mxfp4-dp4a` |
| `B` | `test-moe-scratch-admission` |
| `I` | `test-mul-mat-host-streaming` |
| `XMX-BLOCKED` | `test-mxfp4-vector-dequant` |
| `F` | `test-mxfp4-xmx-tiled` |
| `B` | `test-onednn-woq` |
| `H` | `test-q6k-dispatch` |
| `F` | `test-q6k-reorder-dispatch` |
| `F` | `test-q8-0-layout-cache-path` |
| `E` | `test-q8-0-layout-cache-path-mmvq` |
| `I` | `test-rms-norm-mul-add-broadcast` |
| `F` | `test-sycl-cpu-dispatch` |
| `B` | `test-sycl-cpu-traits-parity` |
| `H` | `test-sycl-descriptor-retention` |
| `I` | `test-sycl-device-uuid-api` |
| `B` | `test-sycl-dispatch-tuning` |
| `B` | `test-sycl-e2e-profile` |
| `REMEDIATED-B` | `test-sycl-esimd-float-atomic-compile` |
| `F` | `test-sycl-fattn-onednn-descriptors` |
| `F` | `test-sycl-fattn-onednn-materialization` |
| `H` | `test-sycl-fattn-xmx-policy` |
| `C` | `test-sycl-graph-retention-scope` |
| `I` | `test-sycl-kernel-selection` |
| `I` | `test-sycl-kv-planned-device-materialization` |
| `G` | `test-sycl-kv-view-resolution` |
| `D` | `test-sycl-layout-choice` |
| `PROBE` | `test-sycl-level-zero-vmem-feasibility` |
| `B` | `test-sycl-lifecycle-event-lease` |
| `B` | `test-sycl-lifecycle-load-txn` |
| `B` | `test-sycl-lifecycle-owner-reset` |
| `COMPILE-GATE` | `test-sycl-lifecycle-public-api` |
| `B` | `test-sycl-lifecycle-runtime-host` |
| `B` | `test-sycl-lifecycle-runtime-wrapper` |
| `B` | `test-sycl-lifecycle-wrapper-overlap` |
| `C` | `test-sycl-mem-handle-concurrent-resolve` |
| `C` | `test-sycl-mem-handle-lifetime` |
| `I` | `test-sycl-module-dlopen` |
| `A` | `test-sycl-moe-direct-final-scratch-plan` |
| `I` | `test-sycl-moe-expert-parallelism` |
| `E` | `test-sycl-moe-fused-down-sum-policy` |
| `A` | `test-sycl-moe-fusion-noactivation` |
| `E` | `test-sycl-moe-gateup-prepack-policy` |
| `H` | `test-sycl-moe-gateup-prepack-scratch` |
| `D` | `test-sycl-moe-handle-resolution` |
| `H` | `test-sycl-moe-identity-hash` |
| `D` | `test-sycl-moe-q8-scratch` |
| `D` | `test-sycl-moe-residency-preflight` |
| `I` | `test-sycl-moe-same-expert-grouping` |
| `I` | `test-sycl-moe-sequence-graphlet-policy` |
| `I` | `test-sycl-moe-xmx-tiled-materialization` |
| `D` | `test-sycl-moe-xmx-tiled-single-layout-planner` |
| `H` | `test-sycl-moe-xmx-tiled-single-layout-policy` |
| `PROBE` | `test-sycl-onednn-mxfp4-feasibility` |
| `I` | `test-sycl-onednn-packed-cache` |
| `I` | `test-sycl-orchestrator` |
| `D` | `test-sycl-reset-model-weight-lease-preserve` |
| `D` | `test-sycl-residency-diagnostics` |
| `D` | `test-sycl-residency-reservation` |
| `C` | `test-sycl-retained-handoff-barrier` |
| `D` | `test-sycl-runtime-alloc` |
| `F` | `test-sycl-set-rows-owner-routing` |
| `G` | `test-sycl-tensor-placement` |
| `G` | `test-sycl-tensor-usage` |
| `B` | `test-sycl-timeline` |
| `G` | `test-sycl-transient-alloc-intent-scope` |
| `H` | `test-sycl-unified-cache` |
| `H` | `test-sycl-unified-memory-e2e` |
| `I` | `test-sycl-weight-key-stability` |
| `H` | `test-sycl-weight-key-uniqueness` |
| `G` | `test-sycl-xmx-unified-correctness` |
| `REMEDIATED-B` | `test-tensor-classification` |
| `I` | `test-tensor-placement` |
| `I` | `test-tiered-dispatch` |
| `D` | `test-unified-cache-bugs` |
| `D` | `test-unified-cache-concurrent` |
| `D` | `test-unified-cache-fast-path` |
| `I` | `test-unified-cache-integrity` |
| `H` | `test-unified-dispatch-integration` |
| `XMX-BLOCKED` | `test-unified-kernel` |
| `XMX-BLOCKED` | `test-unified-kernel-ops` |
| `XMX-BLOCKED` | `test-unified-kernel-persistent` |
| `XMX-BLOCKED` | `test-xmx-compute` |
| `XMX-BLOCKED` | `test-xmx-config` |
| `XMX-BLOCKED` | `test-xmx-default-enable` |
| `XMX-BLOCKED` | `test-xmx-esimd-basic` |
| `XMX-BLOCKED` | `test-xmx-hardware-detect` |
| `XMX-OPT-GATED` | `test-xmx-host-streaming` |
| `I` | `test-xmx-moe-mxfp4` |
| `XMX-BLOCKED` | `test-xmx-optimization` |
| `B` | `test-zone-sizing` |

Group keys. `A`–`I` are the execution batches in order and are the only Phase B
work. The rest are dispositions, not schedule:

- `REMEDIATED-B` — was permanently green, has been fixed, and is now a Phase B
  candidate whose mutation row still has to be written and run.
- `PROBE` — a capability probe rather than a gate: it reports what the driver or
  oneDNN supports, and no observed answer is "wrong". The only failure it can
  express is "the probe did not happen", which it now does by exiting 77.
- `COMPILE-GATE` — checked by the compiler, not by ctest. Mutating and running
  it proves nothing; build the target and read the compiler.
- `XMX-OPT-GATED` — real test, body reachable only with `GGML_SYCL_XMX_GEMM=ON`
  and `GGML_SYCL_MMQ_XMX=ON`. Those definitions now reach the test target;
  before this pass no `-D` could execute one line of it.
- `XMX-BLOCKED` — behind `GGML_SYCL_BUILD_XMX_TESTS` (default OFF). The build
  break that stopped `=ON` from working has been repaired but **not yet built**,
  and each of the 14 names is now also registered as a DISABLED placeholder when
  the option is off, so `ctest -R` on them can no longer match zero silently.
- `MOCK`, `CANNOT-FAIL` — both empty; retained as key definitions only.
