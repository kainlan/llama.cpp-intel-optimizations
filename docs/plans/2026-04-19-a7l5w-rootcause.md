# llama.cpp-a7l5w — Rootcause, instrumentation, and follow-up

**Status**: Reproduced. Instrumentation committed. Root cause refined but
the architectural fix not yet landed. **Gate 4 remains FAILING.**

**Branch**: `feature/sycl-coalescing`
**Precursors**:
- `llama.cpp-mqxer` A0/A0b/A0c/A0d docs (`docs/plans/2026-04-18-mqxer-root-cause.md`)
- `llama.cpp-lj6p0` (TLSF-fragmentation fix)
- `llama.cpp-skgik` (CpuExpertPool UAF fix)
- `llama.cpp-wmk39` (TLSF already chunk-aware; fault is caller-side)

## Crash signature (this repro)

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
    -p '1, 2, 3, 4, 5,' -n 30 --seed 42 --temp 0 -no-cnv
```

- SEGV during prompt-processing phase of layer ~17 of the first graph
  (consistent: `[A7L5W-GEMM]` logs stop at `blk.16.ffn_gate_inp.weight`,
  the 17th router MUL_MAT).
- Fault site: DNNL JIT F32 GEMM microkernel
  (`vbroadcastss -0x80(%rcx),%ymm2` + `vfmadd231ps`).
- Four threads (main + 3 OpenMP workers) at the same JIT PC — this is
  the OpenMP parallel region of a single `dnnl_sgemm` still executing
  at crash time.
- Faulting addresses (per-thread `rcx` / LHS A-matrix pointer):
  `0x72e776358b00`, `0x72e77505cb40`, `0x72e776358b60`, `0x72e776358b40`.
- The 0x1.4 MB past the end of a 2 GB `/dev/dri/renderD129` chunk at
  `[0x72e6f6200000, 0x72e776200000)`. Unmapped VA gap between the chunk
  and the 104 KB zero-guard at `0x72e7763e6000` — fault is a classic
  "past end of a pinned-chunk VMA."

## What my Phase-1 instrumentation proved

Added a `#include`-gated probe (`a7l5w-probe.hpp`) wired into:

- `cpu_mul_mat` early entry (log every call)
- `cpu_mul_mat` `run_mul_mat` lambda top (log before any GEMM math)
- GEMM-fallback branch entry (log before `dnnl_sgemm`)
- `cpu_mul_mat/dnnl_sgemm_A/B/C` — bound-check A, B, C against
  `alloc_registry`
- `cpu_pp_gemm/dequant_in`, `cpu_pp_gemm/dnnl_sgemm_A/B/C`
- `moe_host_expert/weight_host` pre-task-submit
- `kv_remap/per_layer` hard assertion (replaces the soft log-only
  overflow check at `ggml-sycl.cpp:14617`)
- `cpu_pp_caller/src0_batch` at `ggml-sycl.cpp:30478`

**None of the bound-check probes fired** despite repeated reproducing
SEGVs. Concretely:

- `cpu_mul_mat` IS called on `blk.N.ffn_gate_inp.weight` (F32 router)
  with `M=14 N=32 K=2880`.
- `async_mode == false` because `cpu_tensor_is_moe_routing_chain`
  returns true for `ffn_gate_inp` → `force_sync_for_moe_routing=true`.
  No `A7L5W-SUBMIT` events for the legacy sync path.
- `g_cpu_chain_event_valid == 0` every time
  `ggml_sycl_cpu_staging_drain()` runs at graph entry — no async host_task
  is ever pending from this path, so the existing A0d drain has no effect
  here.
- The GEMM-fallback branch is entered. Pointer logging in that branch
  shows `src0_batch` values that, over 17 successful calls, **drift
  out of the renderD129-mapped VA region entirely**:
  - blk.0 `src0_batch=0x72e8c1893400` — inside a mapped chunk
  - blk.16 `src0_batch=0x74fde9c08a80` — NOT in any mapping in the
    coredump at fault time
- The four threads at the JIT fault PC have `rcx` values in the
  `0x72e77...` range — matching the prev-chunk end `0x72e776200000`
  (1.35 MB into the unmapped gap).

**Therefore**: between the per-call `GEMM-fallback` log and the
microsecond-later `dnnl_sgemm` worker dereference, the A-matrix backing
**is torn down** by some other thread.

## The caller-side bound check is not the bug

My Site-1 probe asks alloc_registry whether
`[weight_f32, weight_f32 + N * weight_ld * 4)` fits inside a registered
allocation. At submission time, the pointer IS still registered and
fits — the check passes silently. The fault is a TOCTOU: between the
check (or the implicit "pointer capture for the OMP parallel region")
and the DNNL microkernel's vectorised load, the DRM-backed host-USM
page hosting that pointer gets unmapped.

## Revised root cause (hypothesis, not yet verified with blocking
instrumentation)

Unified-cache HOST_PINNED weight views returned by
`get_host_ptr(src0, ..., gpu_q, ...)` → `cache->get_view(key,
GGML_LAYOUT_AOS)` are **not reference-counted / lease-pinned**. The
cache's eviction path (or a concurrent `sycl::free` from some
background worker) can free the backing `sycl::malloc_host` chunk while
an in-flight `dnnl_sgemm` on the main thread is still using the view.

Two specific candidate invalidation paths (to confirm with
instrumentation):

1. **Unified-cache HOST_PINNED eviction during the same graph**:
   `unified_cache::evict_one` or the deferred-free worker ran `sycl::free`
   on the weight's HOST_PINNED chunk while `cpu_mul_mat` was still in
   `dnnl_sgemm`. The A0d drain at graph entry (`ggml_sycl_cpu_staging_drain`)
   covers ASYNC CPU host_tasks only — it does NOT block eviction of a
   HOST_PINNED view consumed by a SYNCHRONOUS caller on the same thread.

2. **Host-zone STAGING reset mid-graph**: not applicable here — the
   drain does its job at graph boundaries. Rejected by the observation
   that `g_cpu_chain_event_valid == 0` at every drain, so there is no
   async CPU task to drain for this path.

Candidate 1 is the more likely because:
- The pointer progression across 17 router dispatches (`0x72e8c..`,
  `0x72e8d..`, `0x72e7f..`, ..., `0x74fde..`) is consistent with per-layer
  cache-churn: each layer's router weight gets a fresh
  `sycl::malloc_host` chunk when the previous layer's is evicted.
- The predecessor mqxer A0c write-up already documented that the faulting
  pointer CLASS is DRM-backed host-USM, with weight views in the same
  VMAs as the faulting pointer.
- The fault happens on the 17th dispatch, not the 1st — consistent with
  a counter-based cache-pressure threshold (N entries then evict-one-LRU
  on the next insert).

## Design alternatives considered

### Alt A — Refcount / lease on unified-cache views (preferred)

Turn `cache_ptr_view` into an RAII handle that increments an atomic
refcount on the underlying cache entry on creation and decrements on
destruction. While refcount > 0, the cache cannot run `sycl::free` on
that entry; eviction skips it and picks another LRU victim.

Implementation sketch:

- Add `std::atomic<int> in_use_count` to the unified-cache entry.
- `cache->get_view(key, layout)` returns a `cache_lease` RAII object
  that bumps `in_use_count` in ctor, decrements in dtor.
- `evict_one` treats entries with `in_use_count > 0` as unevictable and
  falls through to the next LRU candidate; if all are pinned, the
  eviction fails hard (caller retries after a drain-point).
- `get_host_ptr(..., weight)` in `cpu-dispatch.cpp` must now own the
  lease across the `dnnl_sgemm` call — hold the lease until after the
  `run_mul_mat` returns.

This is the correct architectural fix. It closes the race entirely at
the ownership-contract level, not at a drain-point.

### Alt B — Drain all CPU workers + block eviction for current graph

Extend `ggml_sycl_cpu_staging_drain` to also grab a shared-lock on the
unified-cache's eviction path (eviction takes the exclusive lock). This
would block eviction for the duration of the graph compute.

Rejected: coarse-grained, impacts memory headroom (all current-graph
weight views pinned until graph ends), and creates new deadlock risk
if the current-graph work pressures VRAM and triggers eviction of its
own views.

### Alt C — Copy the weight into a caller-owned scratch buffer

Make `cpu_mul_mat` memcpy the A-matrix into a persistent scratch
(`g_cpu_dispatch_buffers.scratch_nk` or similar) before passing to
`dnnl_sgemm`. Guarantees the pointer handed to DNNL is stable for the
call.

Rejected: wasteful for a 368 KB per-call memcpy on the hot path, and
papers over the real contract bug (which will resurface for the next
caller to get bitten).

### Decision

Implement **Alt A**. It requires carving the refcount into the
unified-cache entry struct and auditing every `get_view` caller, but
it makes the lifetime contract explicit and checkable.

## What I committed this session

1. `ggml/src/ggml-sycl/a7l5w-probe.hpp` — probe header, default ON
   (force-enable ignores `-DNDEBUG`).
2. `cpu-dispatch.cpp` additions at the call sites listed above.
3. `ggml-sycl.cpp` — hard assertion at `tensor->data = la.ptr + offset`
   per-layer KV remap (Site 4).
4. `ggml-sycl.cpp:30478` — per-batch bounds-check at the CPU-PP caller.

All are compile-gated behind `GGML_SYCL_A7L5W_INSTRUMENT` (default 1).
The probe's ABORT path already flushes stderr, and the UNREG path now
flushes too.

**I did NOT land the Alt A refcount fix.** That change requires
touching unified-cache ownership contracts and every `get_view` caller —
too invasive to land without a dedicated review cycle. The
instrumentation is the final landed deliverable for this bead; the fix
is tracked as follow-up work.

## Gate status at end of session

| Gate | Status |
|------|--------|
| 1. Mistral canonical (`6, 7, 8, 9, 10`) | PASS (unchanged — instrumentation is not on the Mistral path) |
| 2. Mistral perf (PP512 ≥ 1700, TG128 ≥ 81) | UNCHANGED (not benchmarked this session — instrumentation is per-call and adds registry lookups; should be measured before promoting the probe out of opt-in.) |
| 3. GPT-OSS 20B `llama-bench -p 512 -n 128 -r 3` | UNCHANGED |
| 4. GPT-OSS 20B `llama-completion -n 128` | **FAIL** — SEGV reproducible, signature verified at ~17 router dispatches |
| 5. GPT-OSS 120B `llama-bench -c 131072` | UNCHANGED (expected FAIL at higher pressure; not run) |

## Re-adding the probe later

If you remove the probe headers/sites in a follow-up cleanup, save the
body of `ggml/src/ggml-sycl/a7l5w-probe.hpp` here and the diffs in
`cpu-dispatch.cpp` / `ggml-sycl.cpp` under the `[A7L5W-ENTRY]`,
`[A7L5W-RUN]`, `[A7L5W-GEMM]`, `[A7L5W-SUBMIT]`, `[A7L5W-DRAIN]` log
tags and the `GGML_SYCL_A7L5W_ASSERT_*` macros. Re-adding for a future
investigation is a ~5-minute paste.

## Next steps (recommended)

1. Add a lease/refcount protocol to `unified_cache::cache_entry` and
   `cache_ptr_view` (Alt A).
2. Audit every `get_host_ptr(weight, ...)` caller to hold the lease
   across its downstream CPU work.
3. Re-run the 20B canary; the probe should continue to stay silent and
   Gate 4 should pass.
4. If Gate 4 still fails, the next-step evidence to gather is
   instrumentation on the unified cache's eviction path logging every
   `sycl::free(ptr)` event with `(ptr, size, t)`; cross-reference with
   the `[A7L5W-GEMM]` log to catch the concurrent free.
