# SYCL Backend Environment Variables (fork tuning)

This is a curated catalog of the `GGML_SYCL_*` tuning and debugging variables
most useful for this fork; it is not an exhaustive inventory. `CLAUDE.md` keeps
only the handful of load-bearing performance opt-outs.

To inventory names in the current source without depending on which environment
accessor reads them, search string literals and allow digits:

```bash
find ggml/src/ggml-sycl -type f \
  \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -print0 \
  | xargs -0 cat | grep -oE '"GGML_SYCL[A-Z0-9_]*"' | sort -u
```

The output includes every matching literal, including names retained only for
compatibility, diagnostics, comments, or removal notices; confirm a live read
before treating any result as an active setting.

## Performance-critical (all default ON, opt-out)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable MMVQ fast-path (slower TG) |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (minimal TG impact ~3%, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (skip unified kernel) |

`GGML_SYCL_DISABLE_GRAPH` controls graph replay, not graph-compute concurrency.
The graph-compute mutex is process-global, but does not universally serialize
submission. Direct/fallback paths release it before `compute_impl` submission.
In contrast, persistent-TG/deferred-copy paths and command-graph record/replay
paths submit while the process-global mutex remains held. Completion may still
outlive the lock where a path permits deferred exit. Thus host submission can
overlap across calls on direct/fallback paths, and device execution may overlap
across calls and devices; pure-GPU decode may also return with kernels still in
flight. Do not infer supported concurrent inference or cache safety from either
overlap. The process-global `unified_cache_set_graph_compute_active(bool)` flag
is an eviction guard, not a per-device concurrency control. Same-device
concurrent inference also remains unsupported for the distinct context/arena
ownership reasons documented in the canonical contract §5.

⚠️ **`GGML_SYCL_DISABLE_GRAPH=1` is currently load-bearing for multi-context
workloads, and that is an open question rather than a setting to recommend.**
Measured at `98deb46ed` with replay ON (the default): the first context to begin
a tracked graph on a device holds its invocation indefinitely — a replay-active
graph never reaches a terminal state, so `release_invocation` silently no-ops —
and every other context's claim on that device is refused `DEVICE_BUSY` exactly
once, with no path by which it later succeeds. The discriminator is clean:
`test-thread-safety` (3 models × 4 contexts, decodes serialized per device)
fails at 3.4 s with replay on and passes at 6.95 s with replay off, both pinned
`level_zero:0,1`.

Consequences worth knowing before you reach for this flag:

- The `test-thread-safety` **ctest registration sets it** (the
  `llama.cpp-b16a`/`llama.cpp-cnre` disposition). That makes the test pass; it
  does not resolve the production question. Do not read that green result as
  evidence that multi-context works on the default configuration.
- Whether winner-holds-forever is intended exclusivity or a gap is being
  adjudicated on `llama.cpp-u7vj` (open, P2), with the analysis in the canonical
  contract §5.3.
- The blast radius is larger than the device count suggests: on current Level
  Zero multi-GPU runtimes the scheduler-visible device set collapses to **one**
  device, so all contexts in a process share device 0's registry slot regardless
  of how many physical GPUs are installed.

## Experimental (opt-in, off by default)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PP_PIPELINE=1` | OFF | Enable double-buffered FP16 weight dequant prefetch. B50 GPT-OSS `llama-bench` PP improves to ~1030-1043 tok/s, but GPT-OSS chat correctness currently fails with repeated `isNaN`, so keep this opt-in until fixed. |

## Kernel dispatch tuning

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_FORCE_MMVQ=1` | Force MMVQ kernels for all batch sizes |
| `GGML_SYCL_FORCE_ESIMD=1` | Force ESIMD kernels |
| `GGML_SYCL_FORCE_MMQ=1` | Force MMQ kernels |
| `GGML_SYCL_FORCE_DMMV=1` | Force DMMV kernels |
| `GGML_SYCL_ESIMD_MIN_BATCH=N` | Min batch size for ESIMD dispatch |
| `GGML_SYCL_ONEDNN_PP_MIN_BATCH=N` | Min batch for oneDNN PP path |
| `GGML_SYCL_ONEDNN_MUL=1` | Enable oneDNN for element-wise MUL (default OFF, SYCL kernel is 2.3x faster) |
| `GGML_SYCL_BATCH_EXPERTS=0` | Disable batched expert kernel launches (default ON) |
| `GGML_SYCL_ESIMD_DEQUANT=1` | Opt-in retest hatch for ESIMD small-block dequant; standard SYCL is the default. ⚠️ The 1.9x-slower figure behind that default was measured on an **Arc B580 + oneAPI 2025.3** — that card is no longer in this machine (replaced by the B70) and the toolchain has moved on, so treat it as *historical justification*, not a current measurement. The conclusion is still believed to hold (block granularity too small to amortize LSC loads), but it has not been re-measured on Battlemage G31. Same caveat applies to the copy of this claim in `CLAUDE.md`. |
| `GGML_SYCL_LAYOUT_OVERRIDE=<mode>` | Force a weight layout: `aos`, `soa`, `coalesced`, or `xmx_tiled`. Overrides the layout policy's own choice — use for A/B isolation, not as a default. (Migrated from AGENTS.md 2026-07-25, which was its only documentation.) |
| `GGML_SYCL_USE_XMX_GEMM=1` | Route quantized MUL_MAT through the experimental XMX GEMM kernels (measured 5–11x **slower** for quantized models). Needs a build carrying **both** `GGML_SYCL_XMX_GEMM` and `GGML_SYCL_MMQ_XMX`; in a default build it does nothing. `=0` disables it on both dispatch paths (it did not until `llama.cpp-wvbw`; see below). |
| `GGML_SYCL_XMX_THRESHOLD=N` | Upper batch bound for the XMX GEMM path; the gate is `batch >= 1 && batch < N`. Default **64**, stated only by the settings table in `ggml_check_sycl()` — not by the global's initializer. Same build requirement as above. See below. |

### `GGML_SYCL_USE_XMX_GEMM` / `GGML_SYCL_XMX_THRESHOLD` — the XMX GEMM path

**Both are compile-gated.** The globals `g_ggml_sycl_use_xmx_gemm` and
`g_ggml_sycl_xmx_threshold`, their `sycl_env_settings` rows in
`ggml_check_sycl()`, and both dispatch sites all sit inside
`#ifdef GGML_SYCL_XMX_GEMM`, which is `option(... OFF)` in
`ggml/src/ggml-sycl/CMakeLists.txt`. In a stock build the variables are not read
at all — setting them measures nothing.

⚠️ **`-DGGML_SYCL_XMX_GEMM` alone does not compile.** The GEMM blocks call
`ggml_sycl_xmx_available()` and `ggml_sycl_xmx_supports_type()`, declared in
`mmq_xmx.hpp`, which `ggml-sycl.cpp` includes only under `GGML_SYCL_MMQ_XMX` — a
second, independently-defaulted CMake option. **Configure both**, e.g.
`-DGGML_SYCL_XMX_GEMM=ON -DGGML_SYCL_MMQ_XMX=ON`.

The requirement is now enforced in two places (`llama.cpp-d6d6`, fixed
2026-07-31), because one of them cannot cover the other:

- `ggml/src/ggml-sycl/CMakeLists.txt` fails the configure with a
  `message(FATAL_ERROR)` naming `GGML_SYCL_MMQ_XMX`. It is a hard error rather
  than an implicit force-ON so that nothing rewrites your cache behind your
  back — `-DGGML_SYCL_XMX_GEMM=ON` will *not* silently turn the other option on
  for you, and `GGML_SYCL_MMQ_XMX` in `CMakeCache.txt` always means what it says.
- `ggml-sycl.cpp` carries the same condition as an `#error` next to the
  conditional include, because the original reproducer was a direct
  `icpx -fsyntax-only -DGGML_SYCL_XMX_GEMM` that never runs CMake at all.

⚠️ **The four undeclared-identifier errors still appear** — the `#error` is the
*first* diagnostic and names the missing flag, but clang does not stop at
`#error`, so it goes on to hit the two dispatch sites and re-emit the old
cascade below it. Read the top of the output, not the bottom: a compile that
ends in `use of undeclared identifier 'ggml_sycl_xmx_available'` is still the
missing-option failure, not a broken XMX path.

**Where the threshold's default comes from — read this before quoting a number.**
The authoritative default is the `GGML_SYCL_XMX_THRESHOLD` row of the
`sycl_env_settings` table in `ggml_check_sycl()`, currently **64**. That parse
writes the global *unconditionally* at backend init, before any `mul_mat`
dispatch can read it, so it wins in every run. Do **not** take the default from
the declaration of `g_ggml_sycl_xmx_threshold`: that initializer is a deliberate
fail-closed `0` covering only the pre-parse window, and
`scripts/check-sycl-xmx-threshold-default.sh` (ctest
`test-sycl-xmx-threshold-policy`) enforces that it stays `0` so a second,
competing literal cannot reappear.

That guard exists because the two literals disagreed for months. `a0ede18b3`
introduced both as `64`; `05519d18f` (*"Increase XMX threshold to 1024 (was 64)
for broader XMX usage"*) changed **only** the initializer, so the increase never
took effect. `43d04b327` made the table row the single source
(`llama.cpp-d5h0`).

⚠️ **So 64 is the value that has always been *effective*, not the value that won
a measurement.** The 1024 hypothesis was never evaluated — the code silently
ignored it, including in that commit's own B50 numbers, which were taken at 64.
`llama.cpp-eju9` is open to actually measure 64 vs 1024 (interleaved paired A/B,
both cards, plus the Mistral completion gate, since a threshold change reroutes
kernel dispatch). If you are tuning this you are in unmeasured territory, not
second-guessing a benchmarked choice.

**The gate**, identical in `ggml_sycl_select_preferred_kernel` and
`ggml_sycl_mul_mat`:

```c
use_xmx = batch >= 1 && batch < g_ggml_sycl_xmx_threshold;
```

so `GGML_SYCL_XMX_THRESHOLD=0` or `=1` disables the XMX path outright, and
"XMX for every batch" is only reachable by naming a large `N`.

✅ **`GGML_SYCL_USE_XMX_GEMM=0` disables XMX everywhere** — fixed 2026-08-01,
`llama.cpp-wvbw`. Both dispatch sites now read the parsed global
`g_ggml_sycl_use_xmx_gemm`, so the value is honoured and the startup report
cannot disagree with the behaviour.

⚠️ **The advice this section carried until then was a workaround for a bug, and
if you find it repeated anywhere else it is now wrong.** It read: *"one of them
ignores the value … `ggml_sycl_select_preferred_kernel` tests
`std::getenv(…) != nullptr` **or** the global … To disable, leave the variable
unset; do not set it to `0`."* That was accurate at the time —
`select_preferred_kernel` tested mere *presence*, so `=0` enabled XMX there while
`ggml_sycl_mul_mat` read the same `0` as disabled, giving one process two
opposite dispatch configurations and a report that printed `0` regardless. The
presence form had been copied from the `GGML_SYCL_FORCE_MMQ` /
`GGML_SYCL_FORCE_DMMV` idiom directly below it, where set-to-anything *is* the
intent because those have no settings row, no global and no report line.

Regression gate: `scripts/check-sycl-xmx-enable-single-source.sh` (textual — the
block is `#ifdef GGML_SYCL_XMX_GEMM`, so an ordinary green build compiles zero
lines of it and certifies nothing). It carries its own fixture suite:
`scripts/check-sycl-xmx-enable-single-source.sh --self-test`.

`ggml_check_sycl()` reports the threshold only when the enable flag is on, and
that report is `GGML_LOG_INFO` — see CLAUDE.md on why those lines need raised
verbosity to appear at all.

## Binbcast completion event

`ggml_sycl_op_bin_bcast` (ADD / SUB / MUL / DIV / REPEAT) publishes one completion
event to two consumers: `ggml_sycl_set_tensor_ready_event(dst, ...)` for
`GGML_OP_MUL`, and `unified_cache::unpin_on_event`, which parks the weight-cache
lease release on it. `GGML_SYCL_BINBCAST_EVENT_MODE` selects where that event
comes from.

| Value | Default | Effect |
|-------|---------|--------|
| `barrier` | **yes** (also the fallback for any unrecognised value) | Manufacture the event with `ext_oneapi_submit_barrier()`. On an in-order queue this is silently promoted to `safe`. |
| `safe` | | Manufacture the event with an empty `single_task` marker kernel. |
| `reuse` | | Return the binbcast kernel's own submission event — no extra submission. Falls back to a real `safe` submission if no kernel event was captured. |

`reuse` exists to remove the empty marker submissions from the decode path (72 per
token on GPT-OSS 20B). The kernel event carries the same guarantee both consumers
need: the kernel is what writes `dst` and what reads the pinned weights, so its
completion means `dst` is ready and the pins are safe to release.

⚠️ **There is no "no event" option, and adding one is not an optimisation.** A
default-constructed `sycl::event` reads as **already complete**, so handing one to
`unpin_on_event` releases the weight-cache lease while the GPU is still reading the
weight — `DEVICE_LOST` or silent corruption. That failure mode presents as a
**speedup**, so a faster run is not evidence that it is correct. Every mode, on
every path, must yield a real completion event.

Debug: with `GGML_SYCL_DEBUG=1` the unpin path logs
`[SYCL-BINBCAST] unpin event mode=<mode> source=<kernel|submission> pins=<n>`.
`source` is what actually happened — a configured `reuse` that fell back still
reports `submission`. `GGML_SYCL_BINBCAST_TRACE=1` adds `[BINBCAST]` staging and
launch traces from the same file.

Gate (all three modes, plus the host-only event-source policy check):

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-unpin-event --mode=compare
```

## Persistent TG kernel (experimental, opt-in)

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_PERSISTENT_TG=1` | OFF | **Required** to enable persistent TG kernel. Without this, all other PERSISTENT_TG_* vars are ignored |
| `GGML_SYCL_PERSISTENT_TG_PHASE=0` | ON | Disable phase-based scheduling (falls back to DAG or legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_DAG=0` | ON | Disable DAG scheduling (falls back to legacy barrier) |
| `GGML_SYCL_PERSISTENT_TG_N_WGS=N` | auto | Override work-group count (auto: max_compute_units/4, clamped 4-64) |
| `GGML_SYCL_PERSISTENT_TG_LOG_POLICY=1` | OFF | Print kernel dispatch mode (phase/dag/split/n_wgs) on each launch |
| `GGML_SYCL_MOE_BLOCK_GRAPHLETS=1` | OFF | Enable experimental MoE block command graphlets |
| `GGML_SYCL_PERSISTENT_SPLIT=1` | OFF | Enable persistent kernel for multi-device row-split |

Testing persistent TG modes:

```bash
# Phase mode (default when persistent TG enabled)
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -n 128

# DAG mode (disable phase, enable DAG)
GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_PHASE=0 GGML_SYCL_PERSISTENT_TG_DAG=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /models/mistral-7b-v0.1.Q4_0.gguf -n 128

# Correctness check — must pass the Mistral completion gate.
# Output is "1, 2, 3, 4, 5, 6, 7, 8, 9, 10" and STOPS at 10 (EOS).
# This line used to say "ends 6..15" -- unreachable at -n 15, so a passing
# gate reads as a failure and sends you hunting a nonexistent TG bug.
GGML_SYCL_PERSISTENT_TG=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

## Memory budget and pressure hierarchy

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_VRAM_BUDGET_PCT=N` | **100** | VRAM budget as % of total (triggers CPU offload when model exceeds). ⚠️ This row said **90** until 2026-07-30; that was correct when written but went stale at `9b0fe06aa` (2026-04-02, *"budget = actual free VRAM, default 100% — remove artificial headroom"*), which moved the two load-bearing defaults to 100 and removed the 256 MB/10% headroom deduction. 100% is **not** an unconditional whole-card guarantee: the budget is still capped to actual free VRAM at cache init, plus a structural runtime-slack reservation. Two further copies of this literal (`unified-cache.cpp:9770`, `:9926`) were missed by that commit and still read 90 — see `llama.cpp-ytr7`; one is log-only, the other a pre-init fallback. |
| `GGML_SYCL_KV_HOST=1` | OFF | Force KV cache to host pinned memory (Level 1 offload) |
| `GGML_SYCL_KV_HOT_LAYERS=N` | auto | Hot layer count for per-layer KV hot/cold tiering |
| `GGML_SYCL_KV_HOT_PCT=N` | auto | Hot window as % of total KV buffer |
| `GGML_SYCL_FORCE_STREAMING=1` | OFF | Enable GPU weight streaming (Level 5, last resort) |
| `GGML_SYCL_HOST_COMPUTE=1` | OFF | Use host-pinned compute buffers (eliminates staging for CPU-dispatched layers) |
| ~~`GGML_SYCL_PIPELINE_MOE=1`~~ | — | ⚠️ **DEAD — does nothing.** No code reads this name (comments only), and its gate `ggml_sycl_pipeline_moe_enabled()` (`ggml-sycl.cpp:14178`) is a hardcoded `return false;`. Added `ae5eae507`, getenv removed `3cf9b5fdf` (2026-05-09, listed under "Removed (hardcoded to defaults)"), then **this row was created `f3c36987f` two months after the removal** — it was never accurate. Setting it measures nothing; do not conclude multi-GPU MoE overlap "doesn't help" from it. `GGML_SYCL_PIPELINE_CPU` below **is** live. |
| `GGML_SYCL_PIPELINE_CPU=1` | OFF | Pipeline CPU expert compute with GPU attention across MoE layers: CPU experts from layer N run during layer N+1 attention |

## Cache and memory

| Variable | Effect |
|----------|--------|
| ~~`GGML_SYCL_UNIFIED_CACHE=0`~~ | **Removed by `9a0670712` with optional cache enablement and its enable/disable branches.** Setting this name no longer disables anything; there is no replacement opt-out. |
| `GGML_SYCL_UNIFIED_CACHE_MODE=<mode>` | Select cache topology (`auto`, `global`, or `per_device`) only; it cannot disable the cache. |
| `GGML_SYCL_NO_PINNED=1` | Disable pinned host memory |
| `GGML_SYCL_WEIGHTS_EVICTABLE=1` | Allow weight eviction under memory pressure |
| `GGML_SYCL_MEM_BUDGET=<MB>` | Set VRAM budget in MB |

For the architectural contract and migration history behind these two rows, see
§1.2 and §9.3 of `docs/design/sycl-canonical-memory-architecture.md`.

## Debugging

| Variable | Effect |
|----------|--------|
| `GGML_SYCL_DEBUG=1` | Enable detailed kernel dispatch logging (MASSIVE output) |
| `GGML_SYCL_UNIFIED_DEBUG=1` | Debug unified kernel dispatch |
| `GGML_SYCL_NAN_CHECK=1` | Enable NaN detection in outputs |
| `GGML_SYCL_VALIDATE=1` | Enable A/B validation between kernel paths |
| `GGML_SYCL_GRAPH_RERECORD=1` | Use graph re-record instead of replay (very slow, diagnostic only) |
| `GGML_SYCL_OP_TIMEOUT_MS=<N>` | Abort with diagnostic if no inference progress for N ms (default 30000, set to 0 to disable). Fires before the xe driver's 10s GT reset cascade. Effective detection latency is `timeout + ~500 ms`. |
| `GGML_SYCL_SAFE_MODE=1` | Drain the SYCL queue after every op submit so a fault surfaces at the op that caused it (2-3x slowdown, implies `GGML_SYCL_DISABLE_GRAPH=1`). Useful for CI canaries and correlating intermittent hangs 1:1 with their triggering op. |
| `GGML_SYCL_FUSION_BIT1_REACH_DEBUG=1` | Default OFF (`ggml-sycl.cpp:83907`). Logs one `[FUSION-BIT1-GUARD]` line per bit1 fusion candidate: `is_weight`, the accumulated `view_offs`, `nb[0]`, element size, the per-operand `safe` verdict, and the full gate chain. Built for the gemma3n cross-model investigation (`llama.cpp-8t4s`), and still the way to answer "did the 81gx view-offset guard classify this operand correctly?" — in the healthy 2-arch run at HEAD, all 44 of gemma3n's unsafe views log `is_weight=0` and are refused. Diagnostic only; the block is guarded by the flag. |
| `GGML_SYCL_HANDLE_STRICT=1` | Default OFF; reports `ggml_tensor_extra_gpu` `data_handle`/`data_device` divergence (first 16 only) without needing `GGML_SYCL_DEBUG=1`. Diagnostic only, no perf effect. Plan: `docs/plans/2026-07-30-extra-device-indexed-handle-storage.md`. |
| `GGML_SYCL_MOE_LAYOUT_DEBUG=1` | Emit the `[MOE-LAYOUT]` per-pass summary unconditionally. The down-i8 / gateup-i8 lines already fire on ANY decline without this; the variable adds the lines a fully-successful pass would otherwise not print. |
| `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS=<N>` | Hard cap on how many down tensors the MoE I8 layout pass upgrades. Unset (or negative) = no cap, the shipping behaviour; `0` disables the upgrade. **Diagnostic only — do not set in production.** See the measured cost below. |
| `GGML_SYCL_ARENA_PP_PROFILE=1` | Emit `[ARENA-PP-*]` counters, including `[ARENA-PP-ONEDNN] … reserve_req_mb=W/A` — the summed oneDNN weights/activations reservation requests. This is the **only** log that reports what was actually asked for, as opposed to what was planned. |
| `GGML_SYCL_ZONE_RESET_AUDIT=1\|2` | Default OFF. Phase 0 of the retire-zone-reset epic (llama.cpp-iiff): every reset/drain site reports what is still live in its zone as `[ZONE-RESET-AUDIT]` lines, with the allocation's own `alloc_id`/`cohort`/`role`/`category` attribution. `=1` changes no behaviour and is safe across the whole gate set. `=2` additionally suppresses the reset even when the zone is clean — host SCRATCH/STAGING are reset-only by design, so that leaks without bound. See below. ⚠️ **The variable name and its site strings (`host-zone-reset/*`, `scratch-pool-reset/bump`, `device-zone-reset/*`, `weight-reclaim/*`) were deliberately NOT renamed when `llama.cpp-37ba` renamed the underlying API to `*_boundary_check` / `*_reclaim` / `scratch_pool_epoch_boundary`.** They are stable historical identifiers that keep captures 01–17 baseline-comparable. They look stale against the current code; they are not. Do not "fix" them. |

### `GGML_SYCL_ZONE_RESET_AUDIT` — reading a capture

Four sites report: `device-zone-reset/<zone>:devN`, `host-zone-reset/<zone>`,
`scratch-pool-reset/bump:devN`, and `weight-reclaim/<mode>:devN` (the mode being
`load-boundary`, `mid-load-replan` or `model-teardown`).

Read `visits_with_live` against `visits` first. They exist because an **empty
capture is not evidence of zero escapes** — it is equally consistent with the
site never having been reached. `visits=0` for a site means it does not appear at
all; a run that reached no site at all says so outright
(`NO RESET SITE WAS VISITED … this run proves NOTHING`).

A full inventory is re-emitted every 256 site visits, at process exit, from the
SYCL watchdog before its `_Exit(1)`, and from a `SIGSEGV`/`SIGABRT` handler — the
runs with the confirmed escapes are the runs that crash. The signal handler
chains to whatever handler it displaced (the planner canaries install their own),
falling back to `SIG_DFL` only when there was none.

#### Every line is printed twice — do not `grep -c`

Output goes to WARN **plus** a raw `stderr` copy. The raw copy exists because
`GGML_LOG_INFO` is dropped at default verbosity in every tool, so an INFO-level
audit line would produce an empty capture indistinguishable from a clean one —
but WARN is *not* dropped, so in the normal case **both copies survive and every
count is doubled**.

| regime | factor | how to dedup |
|---|---|---|
| any tool calling `common_init()` — `llama-cli`, `llama-completion`, `test-thread-safety`, `test-llama-archs` (**all four captures in the protocol**) | x2 | `common_init()` enables the log prefix, so the copies differ: the WARN copy is preceded by a timestamp and a `W ` marker, the raw copy starts at column 0. **`grep '^\[ZONE-RESET-AUDIT\]'` selects exactly the raw copy.** |
| `llama-bench` without `-v` (null log callback) | x1 | nothing to do — only the raw copy exists |
| a binary left on ggml's default log sink (itself a bare `fputs`) | x2 | byte-identical copies, **no discriminator — expect x2 and halve** |

Two practical consequences:

- Confirm the factor empirically before counting anything:
  `grep -c 'ZONE-RESET-AUDIT] ==== end inventory'` against the number of reports
  you expect.
- For a true escape count use `grep 'NEW-ESCAPE' <capture> | sort -u | wc -l` —
  those lines are unique per `(site, cohort, size)`, so `sort -u` collapses the
  duplication without discarding real repeats.

The two writes are separate calls and `common_log` is asynchronous, so their
order is not guaranteed and they can interleave with other output. This is the
same hazard that corrupts `test-llama-archs`' results table; prefer the prose
lines over anything column-aligned.

#### `role` / `category` / `tier` mean different things at the weight-reclaim site

At the three zone sites those columns are read straight off the allocation's
`alloc_handle` and mean what they say. A weight **cache entry** has no such
handle, so `weight-reclaim/*` rows substitute unrelated fields into the same
column names:

| column | zone sites | `weight-reclaim/*` rows |
|---|---|---|
| `alloc_id` | allocation id | weight `name_hash` |
| `role` | `alloc_role` | `cache_layout` |
| `category` | `runtime_category` | **lease count** |
| `tier` | `alloc_tier` | `cache_location` |

This is kept for parity with c-jec1's published inventory, which was read this
way — its "`role=4` = the MoE expert weights" decoding is a `cache_layout`, not
an `alloc_role` — and matching that format is what lets the new capture diff
against the old one. Decoding the columns to names is tracked as
`llama.cpp-u7vi`.

**Read the site name before reading these columns, and never compare them across
sites.**

#### `weight-reclaim/*` cohort names

`unified-cache.cpp`'s `reclaim_weight_entries()` labels every preserved entry
with one of four cohorts, computed from `live` (lease count, i.e. `category`
above) and `owned_by_live` (owner-mask overlap with the current live-model
mask):

| cohort | condition | meaning |
|---|---|---|
| `weight:leaked_lease` | `live != 0`, no live owner, entry was tagged | a real leak — `entries_leaked` counts it, `GGML_SYCL_STRICT_LEASES=1` aborts on it |
| `weight:leased` | `live != 0`, owned by a live model OR never tagged | benign — either correct concurrent ownership, or a lease on an entry the code never claimed to attribute |
| `weight:owned_by_live_model` | `live == 0`, owned by a live model | correct: another live model's idle weight |
| `weight:unattributed` | `live == 0`, not owned by a live model | preserved because it was never tagged (`!entry.owner_tagged`) — see below, never a leak |

`GGML_SYCL_STRICT_LEASES=1` can abort **only** on `weight:leaked_lease` —
`entries_leaked` increments exclusively inside the `live != 0` branch of
`reclaim_weight_entries()`, so `weight:unattributed` (`live == 0` by
definition) can never reach it. This was adjudicated from source in
`llama.cpp-zjz6` after the Phase-0 recapture reclassified GPT-OSS's 1536
MoE-expert `weight-reclaim/model-teardown` entries from `weight:leased`
(pre-`2wv5`) to `weight:unattributed` (post-`2wv5`): the change is that their
lease dropped to zero earlier, not that their ownership tag changed — they
were untagged in both captures. They stay untagged because they are
materialized by `ggml_sycl_materialize_moe_tensor_phase_layout()`'s bulk-XMX
branch (role/`cache_layout` 4 = `GGML_LAYOUT_XMX_TILED`), reached from the
graph-compute path's `GGML_OP_MUL_MAT_ID` scan — a runtime path with no load
transaction bound, so the load-transaction-scoped `stamp_pending_owner()` /
`note_model_load_end()` tagging never reaches them. (`moe_prestage_popular_experts()`
is a different function; it stages only `GGML_LAYOUT_SOA`/`GGML_LAYOUT_AOS`
and merely *pins* an already-existing `XMX_TILED` entry, never creates one.)
`MODEL_TEARDOWN` preserves untagged entries unconditionally, so they are not
reclaimed until the next `MID_LOAD_REPLAN` (immediately, regardless of
`live_mask` — untagged means `owner_mask == 0`, which never overlaps
`live_mask`) or `LOAD_BOUNDARY` (only once `live_mask == 0`). See
`docs/design/sycl-canonical-memory-architecture.md` §1.2 for the full chain
and the practical consequence (delayed, not denied, reclaim).

The attribution is read from the allocation, never re-derived. It became
trustworthy only with llama.cpp-f9tg (`85eb63dcb` / `810ae7fef`); captures taken
before that fix mislabel COMPUTE/CONTROL allocations as GRAPH.

The audit frees nothing and resets nothing at `=1`. Everything it reports stays
owned by whoever already holds its handle.

### `GGML_SYCL_MOE_DOWN_I8_MAX_TENSORS` — what it costs to cap

Measured on GPT-OSS 20B MXFP4, B50, `-p 512 -n 128 -fa 1`, 6 interleaved rounds
per level (`docs/plans/2026-07-25-moe-down-i8-dose-response-findings.md`):

| granted | pp512 | tg128 |
|--------:|------:|------:|
| 0 | 605.51 | 34.14 |
| 2 | 913.50 | 35.13 |
| 5 | 907.13 | 35.56 |

**Setting this to 0 costs 33.7% of pp512.** The layout upgrade is load-bearing
for the ~894 pp512 baseline, not a marginal tuning knob. Beyond 2 layers the
trade is small and roughly symmetric: −0.70% pp512 for +1.24% tg128 going 2→5.
The variable exists to isolate the layout from the VRAM reclaim that pays for
it; `GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB` cannot do that job because it
moves the arena, the zones and the resident set at the same time.
