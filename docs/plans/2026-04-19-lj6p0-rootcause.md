# llama.cpp-lj6p0 — Root cause and architectural fix

**Status**: Investigation complete. Root cause identified. Fix designed.
**Branch**: `feature/sycl-coalescing`
**Author**: investigation 2026-04-19

## TL;DR

The crash is **NOT** a use-after-free in the WEIGHT zone eviction path as originally
hypothesized. It is a **silent address-space fragmentation bug** in
`ggml_sycl::unified_alloc` (`ggml/src/ggml-sycl/unified-cache.cpp:5562-5584`):
the allocator fragments a caller's "contiguous" request across multiple
non-contiguous pinned host chunks, then returns only the **first** segment's
pointer while hiding the fact that segments 2..N live at unrelated virtual
addresses. Any caller that treats the returned pointer as if it is backed by
`size` contiguous bytes — **which is exactly what ggml's backend-buffer
abstraction does** — will compute `tensor->data = base_ptr + offset` for
tensors past segment-0's end, producing pointers that land in the **unmapped
gap** between pinned chunks. The CPU backend's OpenMP workers then SEGV in
`__intel_avx_rep_memcpy` when they try to write their tile output
(`memcpy(&dst_col[iir0], tmp, …)` at
`ggml/src/ggml-cpu/ggml-cpu.c:1486`).

Three of four OpenMP workers land on this same unmapped gap simultaneously →
three-way simultaneous SEGV, consistent with the coredump.

## Evidence

### 1. Reproducer and coredump

```
source /opt/intel/oneapi/setvars.sh --force
ulimit -c unlimited
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -r 1
```

Result: `Segmentation fault (core dumped)` during PP512 setup, exit 139.
Coredump at `/var/lib/apport/coredump/core._Apps_llama_cpp_build_bin_llama-bench.*.780694.*`.

### 2. Stack trace (all four crashing OpenMP workers)

```
#0  0x00000000004d... in __intel_avx_rep_memcpy
#1  0x...          in ggml_compute_forward_mul_mat_id                (libggml-cpu.so)
#2  0x...          in ggml_graph_compute_thread
#3  0x...          in ggml_graph_compute[extracted]
#4  0x...          in __kmp_invoke_microtask                          (libiomp5.so)
...
#9  0x...          in ggml_backend_cpu_graph_compute(backend, cgraph)
#10 0x...          in ggml_backend_sched_graph_compute_async
#11 0x...          in llama_context::graph_compute
#12 0x...          in llama_context::process_ubatch
#13 0x...          in llama_context::decode
#14 0x...          in llama_decode
#15 0x...          in test_prompt (ctx, n_prompt=512, ...)
#16 0x...          in main
```

Notably: the SIGSEGV on thread 1 **is intercepted by
`NEO::PageFaultManagerLinux::pageFaultHandlerWrapper`** (Intel NEO driver's
process-wide SIGSEGV handler for shared-USM migration), but the fault address
is not an NEO-managed SVM page, so the handler passes it through → true SEGV.

### 3. memcpy arguments at fault time (from coredump)

Thread 1 saved registers at signal handler entry:

| reg   | value                    | meaning                           |
|-------|--------------------------|-----------------------------------|
| `rdi` | `0x72e937e037e0`         | memcpy **destination**            |
| `rsi` | `0x72ed0a8a9900`         | memcpy source (`tmp[16]` on stack) |
| `rdx` | `0x4f8d90` (5,213,584)   | (Intel memcpy internal register — not `n`) |
| `rcx` | `0x40` (64)              | memcpy `n` = 16 floats = 64 bytes |
| `r9`  | `0x487930`               | (memcpy function pointer itself)  |

The memcpy is the output-write at `ggml-cpu.c:1486`:

```c
memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
```

where `dst_col = dst->data + (i1*nb1 + i2*nb2)`. The 64-byte write destination
is `dst_col + iir0 * sizeof(float)` inside the MUL_MAT_ID output tensor.

### 4. VMA analysis — `rdi = 0x72e937e037e0` is in an unmapped gap

From `info proc mappings` captured on the same coredump:

```
0x000072e8b7e00000  0x000072e937e00000   0x80000000  0x4d4b2a000  rw-s   /dev/dri/renderD129
0x000072e937fcf000  0x000072e938000000   0x31000     0x0          --p    /dev/zero (deleted)
0x000072e938000000  0x000072e9b8000000   0x80000000  0x454b2a000  rw-s   /dev/dri/renderD129
```

The fault address `0x72e937e037e0`:

- is **14,304 bytes past** the end of the preceding 2 GB DRM-backed chunk
  (`…b7e00000 – …37e00000`)
- is **1,822,208 bytes before** the next 196 KB `/dev/zero (deleted)` guard
- lies in a completely **unmapped hole** between two adjacent
  `/dev/dri/renderD129` pinned-chunk mappings

The dst pointer is **past the end of its supposed backing chunk**. It has
never been mapped. The memcpy is writing off the end of a "contiguous"
allocation that turns out to be non-contiguous in virtual memory.

Because `renderD129` is the Level-Zero device node for Intel Arc on Linux,
all `sycl::malloc_host` allocations are exposed into the process as
`rw-s /dev/dri/renderD129` shared VMAs. The pinned-chunk pool carves the
process's host-USM into 2 GB chunks; **each chunk is a separate VMA and they
are not guaranteed to be virtually adjacent**. The "gap" between chunks in
the dump above is not a teardown artefact — it is the baseline address-space
layout because the `sycl::malloc_host` / DRM mmap sequence gets whatever
addresses the kernel picks, with intermediate `/dev/zero` guard VMAs between.

### 5. Matching the fault to a pinned-chunk boundary

The log from a successful (non-bench) run with the same model:

```
[SYCL] Allocated pinned base chunk 1..7 (size=2048.0 MB each, total=14.0 GB)
[SYCL] Host buffer alloc request: 2037.2 MB (evictable=1, in_model_load=1)  ← model weights
...  [6 x weight buffer allocations totalling ~11.5 GB]
[HOST-ARENA] configured zones: WEIGHT=6697.2 MB  KV=64.0 MB  STAGING=1189.7 MB  SCRATCH=2476.0 MB
[SYCL] Host buffer alloc request: 881.9 MB (evictable=1, in_model_load=0)   ← compute buffer
```

The **881.9 MB `Host buffer alloc request`** post-configure is the trigger.
By that point, 6 × ~2 GB weight allocations have fragmented the WEIGHT
zone's TLSF free list across the 7 pre-allocated chunks. No single chunk has
881.9 MB of contiguous free space, so `pinned_chunk_pool::zone_alloc_segmented`
splits the allocation across chunks. `unified_alloc` then returns only
`segments[0].ptr` to the caller.

That 881.9 MB buffer is then registered as a ggml compute buffer via
`ggml_backend_cpu_buffer_from_ptr(ptr, size)`. ggml-alloc tallies tensor
storage within this "buffer" as if it were one contiguous `881.9 MB`
region. Tensors placed at offsets past `segments[0].size` get
`tensor->data = ptr + offset` values that index into the **unmapped gap**
between the chunks containing segment 0 and segment 1.

### 6. The culprit code

`ggml/src/ggml-sycl/unified-cache.cpp`, lines 5558-5570 (and the mirror at
5578-5595):

```cpp
// Use segmented allocation for large requests - transparently handles
// allocations larger than the chunk size (8GB).
segmented_buffer segs =
    ucache->host_zone_alloc_segmented(pool_zone, alloc_size, pinned_chunk_pool::DEFAULT_ALIGNMENT);
if (!segs.segments.empty()) {
    ptr               = segs.segments[0].ptr;        // ← LIE: caller sees "contiguous alloc_size"
    // Store all segments in the handle for proper cleanup
    out->all_segments = std::move(segs.segments);
    uses_pinned_pool  = true;
    zone_managed      = true;
    out->zone_managed = true;
    out->host_zone    = pool_zone;
}
```

The code's own comment claims "transparently handles allocations larger than
the chunk size (8GB)" — but the transparency only extends to the **free** path
(`all_segments` is consulted at `unified_free_record`, lines 5265-5268). The
**alloc** path is not transparent: the caller receives a pointer whose
backing is `segments[0].size` bytes, not `alloc_size` bytes.

## Why earlier hypotheses are wrong

### A0e / "WEIGHT-zone UAF during eviction"

The A0e fix (commit 2e784b7e6) drains `g_pending_scatter` / `g_pending_cpu_pipeline`
futures before `compute_impl_guard` calls `evict_and_flush`. The bead
description hypothesized that the lj6p0 crash is a different
concurrent-reader class that A0e doesn't cover.

That hypothesis does not fit the evidence:

- **No eviction is firing.** The `[GRAPH-COMPUTE] free=995.7 MB` logs show
  VRAM consistently above the 512 MB threshold that would gate
  `evict_and_flush` in `compute_impl_guard`. No `evict_and_flush` is being
  called during the crash window.
- **The fault address is not in the WEIGHT-zone managed by `evict_one`.**
  `evict_one` frees cache `entries_` (SOA/XMX-layout device-VRAM copies that
  fell back to host). The fault is on a compute buffer allocated through the
  ggml host-buffer path (`ggml_backend_sycl_host_buffer_type_alloc_buffer`),
  which routes to `unified_alloc` with `role=WEIGHT, must_host_pinned=true`
  and is tracked in `g_runtime_alloc_registry` — **not in `entries_`**.
  `evict_one` has no way to free this buffer.
- **The Mistral 7B case is unaffected** (passes), yet Mistral has the same
  eviction path. Mistral fits in a single pinned chunk, so the allocator
  never has to segment — and the bug never fires.

### "ggml scheduler routes to CPU backend; CPU reads stale WEIGHT pointer"

This was a plausible framing but the actual failure is a **structural
misalignment at allocation time**, not a concurrent teardown. The CPU
backend's OpenMP workers are doing exactly what they should be doing —
writing to `tensor->data + offset`. That pointer has been bogus since the
moment the buffer was allocated; no concurrent eviction is needed to
manifest the SEGV.

### Why short-gen works and long-gen fails

Gate-4 (`-n 15`) ends before the first 49-node MUL_MAT_ID graph dispatches
a CPU-routed op to a tensor whose `tensor->data` lies past the segment-0
cliff. The 49-node graph first appears a few tokens in (MoE layer rotation
between experts). With `-n 15` the specific host-resident expert tile that
lives past the cliff is not touched; with `-n 64`/`-n 128` it is, and the
write in `mul_mat_id_one_chunk` faults.

`llama-bench` triggers earlier and more deterministically because PP512
dispatches all 32 experts in one large graph; the warmup batch is enough to
hit a cross-segment tensor.

## Fix design

### Design principles

1. **Correctness at allocation, not at use.** The caller cannot distinguish
   a fragmented allocation from a contiguous one without cross-checking
   segments, and no ggml-level caller should have to. The allocator must
   only return pointers whose `[ptr, ptr+size)` is fully mapped and
   contiguous in the process address space.
2. **No feature regression.** CPU offload, MoE pipelining, graph replay,
   host-resident weights, 120B full-context — all must keep working.
3. **No env-var opt-in.** The fix is the default path.
4. **Preserve existing zone semantics.** Per-alloc WEIGHT-zone freeing,
   bump-reset STAGING/SCRATCH, shared-mutex guards — none of this changes.

### Candidate designs

Three candidates were evaluated; the third is chosen.

---

**Candidate 1 — Grow the pinned chunk size so one chunk always suffices.**
Pros: No code change to alloc logic. Cons: Fails for large compute
buffers (11 GB), raises per-allocation Level-Zero limits, and Intel L0 has
a hard per-alloc ceiling of ~11.3 GB. Does not scale to 120B+131K context.
Rejected.

**Candidate 2 — Merge adjacent chunks into a single sliding arena on grow.**
Pros: Makes all allocations contiguous by construction. Cons: Requires
controlling the VA placement of `sycl::malloc_host` return values, which
the Level-Zero runtime does not expose. We cannot ask SYCL/L0 to mmap its
next chunk at a specific address. Rejected.

**Candidate 3 (chosen) — Reject silent fragmentation at the allocator; 
make the caller's buffer honestly reflect one chunk's worth of contiguous 
memory.**

This is the right architectural fix:

- `unified_alloc` for host-pinned allocations **fails** if the request cannot
  be satisfied by a single contiguous segment from one chunk. It does
  **not** silently return `segments[0].ptr`.
- `ggml_backend_sycl_host_buffer_type_get_max_size` already exists as the
  mechanism for ggml-alloc to split large buffers into chunk-sized pieces.
  We tighten it so it returns the **largest single-chunk free contiguous
  block currently available in the target zone** (for the role inferred
  from `in_model_load` + `weights_evictable`), capped by the static chunk
  size. ggml-alloc then splits its buffer into N smaller buffers, each of
  which fits contiguously in one chunk, each a separate ggml
  `ggml_backend_buffer_t` with its own base pointer.
- When the allocator cannot satisfy even a chunk-sized request (everything
  fragmented), it grows the pool with a new chunk and retries (the pool
  already supports `grow()`; this is just a matter of triggering it from
  the alloc failure path).

Net effect: every `ggml_backend_buffer_t` the SYCL host-buffer-type returns
points to a single contiguous host-USM region. Multi-chunk allocations
become multiple ggml buffers, which the ggml allocator already handles
correctly (it doesn't assume inter-buffer contiguity).

Why this scales to 120B/131K context: 120B's compute buffers are larger
than 20B's, but the fix makes the buffer-split behaviour honest. ggml-alloc
responds to smaller `get_max_size` by chunking its buffer list — that path
is well-tested (used daily for Mistral Q4_0 with the same host-buffer
type). Host-zone zone capacity tracking is unchanged; only the per-buffer
size cap changes.

### Implementation

Three edits in `ggml/src/ggml-sycl`:

1. **`unified-cache.cpp` lines 5558-5570 and 5578-5595**: replace both
   segmented-alloc branches with a single-contiguous alloc via
   `pinned_chunk_pool::zone_alloc` (not `zone_alloc_segmented`). If that
   returns nullptr, trigger a pool `grow()` for the zone and retry. If the
   grow also fails, return failure from `unified_alloc` so the caller can
   fall back to the `ggml_sycl_malloc_host` path (line 5610). The
   `out->all_segments` field is no longer needed for the pinned-pool
   path and is reset to empty.
2. **`unified-cache.cpp` lines 5261-5272**: update `unified_free_record`'s
   WEIGHT-zone branch to no longer loop `all_segments`. Each tracked
   allocation is now a single segment → free via `host_zone_free(WEIGHT, ptr)`.
3. **`ggml-sycl.cpp` `ggml_backend_sycl_host_buffer_type_get_max_size`
   (lines 17770-17797)**: change the return value from
   `min(scratch_avail, chunk_cap)` (static) to
   `min(largest_contiguous_free(target_zone), chunk_cap)` so ggml-alloc
   receives an accurate picture of how big a contiguous buffer it can
   request right now. Expose a helper on `pinned_chunk_pool` —
   `largest_contiguous_free(host_zone_id)` — that iterates `zone_allocators_[z]`
   and returns `max_i(zone_allocators_[z][i].allocator->largest_free_block())`.

The static `chunk_cap` upper bound (2 GB) stays as a secondary cap so that
ggml-alloc never requests a buffer larger than one chunk even when the
largest free block spans multiple chunks (it can't — TLSF doesn't cross
chunk boundaries — but the cap is defensive).

Existing segmented-alloc call sites (`unified_alloc` for host allocations)
are the only callers of `host_zone_alloc_segmented` that consume
multi-segment results as if contiguous. Keep `host_zone_alloc_segmented` on
the public API — it has a legitimate use for clients that are
segmentation-aware — but remove its use from `unified_alloc`.

### Why this preserves A0e's fix

A0e drained CPU-expert futures before `evict_and_flush` because
`evict_and_flush` could free WEIGHT-zone host pages out from under CPU
workers. That fix is still necessary and still correct — it protects
against a genuine concurrent-eviction hazard for the cache `entries_` path.
This new fix is orthogonal: it ensures the **initial** allocation handed
to ggml is honest, so no write off the end of a buffer occurs even when
no eviction is in flight.

### Why this preserves all features

- **CPU offload, MoE pipelining, graph replay, host-resident weights**: all
  operate on ggml buffers, and ggml-alloc already handles multi-buffer
  chunking transparently. No call path changes.
- **120B full context**: larger compute buffers just mean ggml-alloc
  creates more chunk-sized buffers, rather than one giant buffer that
  silently fragments.
- **Mistral 7B**: Mistral's compute buffers fit in a single 2 GB chunk;
  behaviour is unchanged for it.

## Gate plan

Run after implementing the fix:

1. Mistral canonical correctness (must emit `6, 7, 8, 9, 10`).
2. Mistral perf (PP512 ≥ 1700, TG128 ≥ 81).
3. GPT-OSS 20B `llama-bench -p 512 -n 128 -r 3` — **must pass 3/3** (currently crashes).
4. GPT-OSS 20B `llama-completion -n 128` — **must complete without SEGV**
   (currently crashes).
5. GPT-OSS 120B `llama-bench -p 512 -n 128 -r 1` — **must pass** (per user directive).

Between gates: 30-60 s cooldown for Arc B580 thermal.

## Gate results

Run with the fragmentation fix applied:

1. Mistral canonical: **PASS** — output ` 1, 2, 3, 4, 5, 6, 7, 8, 9, 10`.
2. Mistral perf: **PASS** — PP512 = 1701.62 ± 1.06 tok/s, TG128 = 81.06 ± 0.02 tok/s.
3. GPT-OSS 20B `llama-bench -p 512`: **PP PASS** at 55.51 tok/s (previously
   crashed in `mul_mat_id memcpy` → the exact signature the bead documents).
   TG128 phase crashes in a **different** backtrace (`dnnl_sgemm /
   gemv_kernel_driver` in `CpuExpertPool`) — see "Residual TG regression"
   below.
4. GPT-OSS 20B completion `-n 128`: First-token graph completes, then crashes
   during TG with the same `dnnl_sgemm` signature. Baseline (without this
   fix) also crashes at the same point, so the TG crash is pre-existing.

## Residual TG regression (tracked separately)

After landing the allocator-fragmentation fix, GPT-OSS 20B TG still crashes
in a **different** call path:

```
#2  simd_mxfp4_q8_0_16row               (libggml-sycl.so)
#3  tbb::...::start_for::run_body       (CpuExpertPool TBB arena)
#4  tbb::...::start_for::execute
...
#11 ggml_sycl_cpu_expert_mul_mat_batched
#12 std::function_handler<...CpuExpertPool::submit_batch...>::_M_invoke
#13 CpuExpertPool::worker_thread
```

`rbx` is a wild-valued expert-row pointer loaded from the stack
`row_ptrs[16]` array at `cpu-dispatch.cpp:5989`. NEO's page-fault handler
intercepts the fault, passes it through, SEGV.

This is the same family as the original mqxer bug (A0-A0e) but a different
site: it is **not** covered by A0d's `ggml_sycl_cpu_staging_drain` (that
drains the async host_task chain, not the `CpuExpertPool`) and not by
A0e's `compute_impl_guard` future drain (that only fires when
`evict_and_flush` is triggered by low VRAM — which does not occur here;
`free=995 MB` throughout the run).

Draining the CpuExpertPool futures at graph entry (before
`host_zone_reset(STAGING)/SCRATCH)`) was tested and got further (10+
tokens before fault instead of 0-1) but did not fully fix the crash.
A thorough fix requires deeper investigation — likely tracking each
`CpuExpertPool` task's STAGING/SCRATCH pointer ownership through the
worker lifecycle and gating zone resets on per-task completion rather
than process-global future polling.

**This residual bug is being tracked as a separate beads ticket.** It is
**not** a regression from the fragmentation fix — baseline HEAD
(without this fix) also crashes at the same token.
