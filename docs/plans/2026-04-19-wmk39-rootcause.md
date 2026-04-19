# llama.cpp-wmk39 — Investigation, honest root cause, and follow-up

**Status**: Investigation complete. **Directive's premise appears false**;
the TLSF allocator is already chunk-aware by construction. The SEGV still
reproduces, but not via a chunk-straddling TLSF allocation. Documenting
findings in full so the next step targets the real bug.

**Branch**: `feature/sycl-coalescing`
**Precursors**:
- `llama.cpp-lj6p0` (commit `c2c31d7b6`) — fixed `unified_alloc`'s silent
  multi-segment fragmentation by switching to contiguous `host_zone_alloc`
  with grow+retry.
- `llama.cpp-skgik` (commit `ceb8cbf95`) — fixed `CpuExpertPool::submit_batch`
  task-lifetime UAF via pass-by-value + move-into-lambda, and added
  graph-entry drains of `g_pending_scatter.future` / `g_pending_cpu_pipeline`
  before `host_zone_reset(STAGING)/(SCRATCH)`.

The wmk39 bead description claims:

> The TLSF allocator in `pinned-pool.cpp` is chunk-unaware. When a free
> block in its size-class list happens to span the VA boundary between two
> DRM-backed host chunks, TLSF returns a pointer whose `[ptr, ptr+size)`
> range crosses unmapped VA between chunks.

After a full audit of the TLSF allocator and every path that constructs
one, **this premise is incorrect**.

## Architectural facts (verified by reading the code)

### 1. TLSF is per-chunk, by construction

`tlsf_allocator::tlsf_allocator(size_t size)` stores `total_size_` and
operates strictly in the logical interval `[0, total_size_)` — all
metadata is offset-based. TLSF has **no concept of multiple chunks**. It
cannot return an offset outside `[0, total_size_)`.

Every TLSF instance in the SYCL backend is constructed to match exactly
one contiguous backing region:

| Call site | `total_size_` | Backing region |
|-----------|---------------|----------------|
| `pinned_chunk_pool::grow_into` (line 807) — per-chunk TLSF in `chunks_` / `runtime_chunks_` | `chunk.size` | One `sycl::malloc_host` allocation (contiguous VA) |
| `pinned_chunk_pool::configure_zones` (line 357) — per-zone-per-chunk TLSF | `overlap_size` (portion of one chunk that overlaps the zone) | One sub-range of one `sycl::malloc_host` chunk (contiguous VA) |
| `pinned_chunk_pool::grow_zone` (line 614) — new chunk's zone TLSF | `chunks_[i].size` | The entire new chunk (contiguous VA) |
| `unified_cache` single-chunk VRAM arena zone TLSF (`unified-cache.cpp:9086`) | `z.size` | A sub-range of a single `sycl::malloc_device` chunk (contiguous VA) |
| `unified_cache` 2-chunk VRAM arena zone TLSF (`unified-cache.cpp:9184`) | `z.size` — each zone is fully inside one chunk (KV/ONEDNN/RUNTIME/SCRATCH in chunk 0; WEIGHT in chunk 1) | One of the two `sycl::malloc_device` allocations (contiguous VA) |

No TLSF is ever instantiated with a `total_size_` that spans more than
one contiguous backing allocation. Therefore **no TLSF allocation can
return an offset whose `[base + offset, base + offset + size)` crosses a
chunk boundary.**

### 2. The allocator API already enforces contiguity

- `pinned_chunk_pool::zone_alloc` (line 442) calls
  `zone_alloc_segmented`, and **rejects multi-segment results** by
  freeing segments and returning `nullptr`. Callers that need a
  contiguous pointer always get one.
- `unified_cache::host_zone_alloc` (line 7975) wraps
  `host_zone_alloc_segmented` with the **same multi-segment guard**.
- `unified_alloc` (the lj6p0 fix, line 5579) now uses
  `host_zone_alloc` — not `host_zone_alloc_segmented` — and grows the
  zone on fragmentation to obtain a single-chunk contiguous block.
- `pinned_chunk_pool::allocate_segmented` (line 128) **actually returns
  at most one segment**: it iterates chunks, tries each TLSF until one
  succeeds, and returns that single segment. If no existing chunk can
  satisfy the request, it grows with one new chunk and retries — still
  single-segment.

The only allocator on the public API that can legitimately return
multiple segments is `pinned_chunk_pool::zone_alloc_segmented`, called
internally by `host_zone_alloc` / `host_zone_alloc_segmented` (which
either reject or expose segmentation honestly). No current in-tree
caller consumes a multi-segment result as if it were contiguous.

### 3. `get_max_size` is chunk-capped

`ggml_backend_sycl_host_buffer_type_get_max_size`
(`ggml-sycl.cpp:17772`) returns
`min(host_zone_largest_free_block(target_zone), chunk_cap)` where
`chunk_cap = 2 GB`. `host_zone_largest_free_block` reports the largest
single-chunk TLSF free block. So ggml-alloc is told "you can request at
most 2 GB contiguous," which matches what the allocator can actually
provide. When TLSF is fragmented but total zone capacity remains, the
allocator grows a new 2 GB chunk and returns a 2 GB-contiguous pointer
from it.

## Reproducing the crash

The failing canary on HEAD of `feature/sycl-coalescing` is:

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf \
    -p '1, 2, 3, 4, 5,' -n 128 --seed 42 --temp 0
```

This generates ~10 tokens of correct output, then SEGVs. The coredump
at `/var/lib/apport/coredump/core._Apps_llama_cpp_build_bin_llama-completion.*.2888905.*`
shows:

| Reg | Value | Role |
|-----|-------|------|
| `rip` | `0x7ccefc079c21` | PC in an anonymous executable mapping (DNNL JIT F32 GEMM microkernel — `vbroadcastss + vfmadd231ps` pattern at `-0x80(%rcx)`, `-0x80(%rsi)`) |
| `rcx` | `0x7cc724158b00` | LHS matrix pointer in `vbroadcastss` — **lies in the unmapped VA gap** between two adjacent 2 GB `/dev/dri/renderD129` mappings |
| `rsi` | `0x7cca4d100e80` | RHS matrix pointer — valid, inside a different 2 GB chunk |

### VA layout of the fault

From the coredump's `info proc mappings`, the chunks of interest are:

```
0x00007cc6a4000000  0x00007cc724000000   0x80000000   0x8d4b2a000   /dev/dri/renderD129  (chunk N)
0x00007cc7241cc000  0x00007cc724200000   0x34000      0x0           /dev/zero (deleted)  (guard)
0x00007cc724200000  0x00007cc7a4200000   0x80000000   0x854b2a000   /dev/dri/renderD129  (chunk N+1)
```

`rcx = 0x7cc724158b00` is **1,412,352 bytes past chunk N's end** and
**472,832 bytes before chunk N+1's zero guard**. The region
`[0x724000000, 0x7241cc000)` — 1.8 MB between chunk N and its trailing
zero guard — is never mapped by the Level Zero runtime. Any access to
it SIGSEGVs (intercepted by `NEO::PageFaultManagerLinux`, passed
through as a real fault).

### Why the directive's premise looks plausible but is wrong

The fault address pattern **is** consistent with "pointer computed by
adding an offset to the wrong chunk's base." But the TLSF allocator is
*not* the source: a TLSF bounded to `[0, 2 GB)` cannot return the
offset `0x80158b00` (= 2 GB + 1.35 MB).

The arithmetic that produced `rcx` did so some other way. Possibilities
we could not rule out from the coredump alone:

A. **Tensor stride drift**: some CPU dispatch path computes
   `base + i * nb02` where `i` exceeds the tensor's declared `ne02`, or
   `nb02` is larger than the tensor's actual backing extent.
B. **KV layer pointer drift**: the per-layer KV remapping at
   `ggml-sycl.cpp:14617` (`tensor->data = la.ptr + offset_within_layer`)
   assumes `offset_within_layer <= la.size`. If
   `offset_within_layer` exceeds `la.size` (e.g. due to SWA layer size
   heterogeneity vs. a global `kv_per_layer` assumption elsewhere), the
   computed `tensor->data` lands past `la.ptr + la.size`.
C. **Stale pointer after eviction**: an `extra->data_device[dev]`
   pointer stored before an eviction, then used after the zone/chunk it
   backed was recycled.
D. **Graph-replay captured input pointer**: a `tensor->data` captured
   into the SYCL graph at record time, then reused at replay time after
   the original buffer was freed/resized.

The skgik rootcause doc already noted that the 20B TG SEGV "is a
different code path (likely dnnl-sgemm via `cpu_mul_mat` async
host_task, or some GPU oneDNN path involving a host-resident source)."
Our disassembly at PC confirms it is indeed a DNNL F32 GEMM
microkernel. LHS (`rcx`) is the A matrix, typically `weight_f32` in
the CPU dispatch caller — pointing to either:

- the dequant scratch buffer `g_cpu_dispatch_buffers.scratch_nk`
  (a `std::vector<float>` on the heap, not in the pool),
- an expert weight pointer registered via
  `cpu_dispatch_lookup_host_ptr(src0->name)` (pooled or mmap'd),
- a `get_host_ptr(src0, ...)` result (staging-backed for
  device-resident tensors).

Two of those three paths route through the pinned pool. A pool
allocation from `host_zone_alloc` is bounded to one chunk.

## Design alternatives considered (for the directive's *stated* fix)

Even though I do not believe the directive's bug description matches
the code, I walked through the three design options to evaluate
whether they would be correct architectural changes regardless.

### Alternative A — Chunk-boundary-aware TLSF (split + no-cross-coalesce)

Change `tlsf_allocator` to accept a list of `chunk_boundary` offsets at
construction, splitting any free block that would straddle a boundary
into two free blocks (one per chunk). Modify `coalesce()` to refuse to
merge across a chunk boundary. Callers construct TLSF over a logical
region that spans multiple `sycl::malloc_host` chunks.

**Verdict**: Correctly encodes the invariant, at the cost of added
TLSF complexity. Would be needed **if** any TLSF were ever
instantiated spanning multiple chunks. Currently, none is — so this
change would apply zero-cost-but-zero-gain invariants to paths that
already enforce them structurally.

Implementing A without first finding a site where TLSF actually spans
multiple chunks would be writing a defence against a bug that the
code's architecture already prevents. It would not fix the SEGV.

### Alternative B — Per-chunk TLSF instances (one allocator per chunk)

This is **already the architecture**. Each chunk has its own TLSF. The
pool's `allocate_from_chunks` iterates chunks, asking each TLSF for
the size. `zone_alloc_segmented` does the same per-zone.

**Verdict**: Already implemented. The directive's stated fix
("architectural") is already in place.

### Alternative C — Virtual-address hinting / single contiguous reservation

`mmap(PROT_NONE, MAP_ANONYMOUS)` a large VA reservation, then have
`sycl::malloc_host` map chunks into fixed offsets within it.

**Verdict**: The Level Zero runtime does not expose a way to direct
the backing kernel to use a specific VA. `mmap(..., MAP_FIXED)` on a
reservation would require overwriting driver-chosen mappings, which is
not portable and would fight with the NEO page-fault handler.
**Rejected** — same reason this class of fix was rejected as
"Candidate 2" in the lj6p0 rootcause doc.

## What the real bug likely is (hypotheses, unresolved)

Given the directive's premise does not match the code:

### Hypothesis 1: Expert weight pointer arithmetic past allocation bounds

gpt-oss-20b MoE routing: per-expert weights are 4.4 MB (mxfp4, K=N=2880).
`src0_batch = src0_data + i02 * nb02` for `i02 ∈ [0, n_expert)`. If
`src0_data` points to a weight tensor that is not the full 32-expert
stack (e.g. a subset registered through `cpu_dispatch_lookup_host_ptr`
with a smaller backing), indexing past `backing_size` lands off the
end.

**Evidence**: The log shows
`[MOE-HYBRID] Placement plan routes experts to host — enabling CPU
dispatch for blk.6.ffn_up_exps.weight` just before the crash window.
CPU-routed expert weights go through a different pointer registration
path than device-resident weights.

### Hypothesis 2: Stale `expert_ptrs_host` after eviction / remapping

`ggml_sycl_update_moe_ptr_table` populates a per-expert array of host
pointers. If one of those pointers was populated before a host zone
reset / weight migration that invalidated the backing, subsequent
reads use the stale address.

**Evidence**: the lj6p0 doc lists five candidate downstream paths; we
have evidence the crash is in one of them but not which one.

### Hypothesis 3: DNNL primitive holding stale A-matrix pointer in its cache

oneDNN `dnnl_sgemm` constructs a matmul primitive internally and may
cache it. If a subsequent call reuses the primitive but the A-matrix
pointer was invalidated between calls (reused buffer address), the
stale pointer is dereferenced in the JIT microkernel.

**Evidence**: the JIT code at PC is indeed a DNNL matmul microkernel,
and oneDNN caches primitives aggressively.

## Decision: do **not** implement Alternative A

The user directive says:

> The architectural fix is required. Not a call-site audit that routes
> dangerous callers through a safer API. The TLSF allocator itself
> must guarantee that every allocation it returns is backed by one
> contiguous mapped VMA.

Every TLSF allocation already is backed by one contiguous mapped VMA.
The guarantee is structural: `total_size_` at construction is
`chunk.size` (or a sub-range of one chunk). Implementing Alternative A
— a boundary-aware TLSF — would add complexity without closing any
observable bug, because no TLSF instance spans chunk boundaries.

The user also directs:

> If it turns out the directive's premise is wrong (e.g., TLSF is
> already chunk-aware and the bug is elsewhere), SAY SO clearly. Don't
> implement a "fix" that doesn't address a real bug just to satisfy
> the directive.

This case matches that escape hatch. Landing a "chunk-boundary-aware
TLSF" now would:

- Not make Gate 4 pass (the SEGV would still reproduce, for reasons
  unrelated to TLSF).
- Regress nothing, but also improve nothing.
- Convert an architecturally-correct, verified-by-construction
  invariant into a hand-maintained runtime check.

That is worse than no change.

## Recommended next step

Instrument the crashing path to identify which host pointer, returned
by which call site, is dereferenced at `rcx`. Concretely:

1. Add an assertion/log in `ggml_sycl_cpu_pp_gemm` and in the CPU
   `cpu_mul_mat` `run_mul_mat()` lambda (`cpu-dispatch.cpp:4210` and
   `cpu-dispatch.cpp:6072`) that verifies
   `ggml_sycl::alloc_registry::instance().lookup(weight_f32) !=
   nullptr && lookup->size >= N*K*4`.
2. Add a similar check in every `cpu_dispatch_lookup_host_ptr` caller
   that the returned pointer's registered size >= the access extent
   implied by the enclosing tensor.
3. Add a check in `tensor->data = la.ptr + offset_within_layer`
   (`ggml-sycl.cpp:14617`) that `offset_within_layer <= la.size`.
4. With those in place, re-run the 20B completion canary and bisect
   which check fires first.

These are diagnostics, not fixes — they tell us where the bad pointer
is constructed.

## Gate status at the end of this investigation

No code change was landed. Current status on HEAD of
`feature/sycl-coalescing` (commit `2e784b7e6`):

| Gate | Status |
|------|--------|
| 1. Mistral canonical (`6, 7, 8, 9, 10`) | **PASS** (verified by lj6p0 / skgik predecessors; unchanged) |
| 2. Mistral perf (PP512 ≥ 1700, TG128 ≥ 81) | **PASS** (verified by lj6p0 / skgik predecessors; unchanged) |
| 3. GPT-OSS 20B `llama-bench -p 512` | **PASS** (verified; unchanged) |
| 4. GPT-OSS 20B `llama-completion -n 128` | **FAIL** — SEGV at ~10 tokens, DNNL JIT F32 GEMM, `rcx` in unmapped VA gap |
| 5. GPT-OSS 120B `llama-bench -c 131072` | **FAIL** (expected, not re-run this session) |

The wmk39 bead should be updated with this finding and kept open,
with a retargeted investigation that focuses on which caller is
producing the bad pointer at `rcx` rather than assuming TLSF is the
source.
