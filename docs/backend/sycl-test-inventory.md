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
