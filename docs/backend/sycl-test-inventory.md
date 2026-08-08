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
registrations added by Task 17 — **10** of them at HEAD, after `a4791b7a9`
dropped `test-sycl-expert-prefetch`; it is not retroactively applied to the 40
registrations that were already live.

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
| `test-mmvq-q8-0-streaming-bench` | 4 | static | 4 | 0 | 3 | 4 |
| `test-mxfp4-xmx-tiled` | 1 | static | 1 | 1 | 1 | 0 |
| `test-q6k-reorder-dispatch` | 1 | static | 1 | 1 | 0 | 0 |
| `test-sycl-expert-prefetch` — **removed by `a4791b7a9`** | 0 | — | 0 | 0 | 0 | 0 |
| `test-sycl-fattn-onednn-descriptors` | 1 | static + oneDNN | 1 | 1 | 1 | 0 |
| `test-sycl-set-rows-owner-routing` | 1 | static | 1 | 1 | 0 | 0 |
| `test-unified-dispatch-integration` | 1 | static | 1 | 1 | 1 | 0 |
| **total at HEAD** (post-`a4791b7a9`) | **10** | **7 targets** | **10** | **6** | **6** | **4** |
| total as audited at `fc606640e` | 11 | 8 targets | 11 | 7 | 7 | 4 |

The `test-sycl-expert-prefetch` row carried one `sycl-restored` name, one
`RUN_SERIAL` and one `SKIP_RETURN_CODE 77`, which is the whole of the
11→10 / 7→6 / 7→6 delta; `a4791b7a9` deleted its `add_executable`, `add_test`
and property block together, so nothing else in the table moved.

The four streaming names share one label property call. Their bare benchmark,
MMQ smoke, and forced-MMQ modes have `SKIP_RETURN_CODE 77`; the cache smoke
mode is deliberately not included because its configured path has no exit-77
branch (unavailable cache/device work is a failure there). The other three
skip-77 registrations match explicit source exits of 77 (four before
`a4791b7a9`). The six ordinary GPU fixtures are `RUN_SERIAL` (seven before
`a4791b7a9`); the four opt-in streaming modes instead carry the `manual` label
as a group. A substring-regex scan of every restored label found zero matches
for the throttled-sweep denylist `residency|mem-handle|cache`.

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
| `test-dmmv-q4-0-coalesced` | `errors=88 max_diff=0.427313 max_rel=0.309105 FAIL` | `llama.cpp-szv8` | **in_progress** |
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
`llama.cpp-qvid`), each verified against its own gate. **No re-run of the
accepted set has been recorded since `772798e91`.** The 38/1/11 split above
therefore remains the only executed result for this set, and a fresh serial
sweep is owed before certification claims the set is green. Do not read the
closed column as a pass column.

`llama.cpp-szv8` is the one still open. Its own investigation established
something worth carrying: `test-dmmv-q4-0-coalesced` **did not exercise the
kernel it names** at the time of this sweep — the readback fell back to the
tensor's SoA storage (`source=tensor-storage-fallback`), so the numeric rows
were measuring an unintended path. `szv8c` (`7be63714e`) rewired it through the
real staging bracket, and the first attributable run then reported
`source=layout_ptr` with `Coalesced layout check: PASSED` and *near-bit-identical*
numeric failures (`errors=55 max_rel=0.165479` against the pre-fix
`55 / 0.165480`). That coincidence is unexplained and is the live question on
that ticket.

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
