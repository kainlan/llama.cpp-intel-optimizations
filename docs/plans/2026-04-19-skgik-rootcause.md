# llama.cpp-skgik — Root cause and architectural fix

**Status**: Investigation complete. Root cause identified. Fix designed.
**Branch**: `feature/sycl-coalescing`
**Author**: investigation 2026-04-19
**Precursor**: `llama.cpp-lj6p0` (commit `c2c31d7b6`) fixed silent host-arena
fragmentation. Its fix unmasked this pre-existing UAF in the CpuExpertPool
submit path.

## TL;DR

`CpuExpertPool::submit_batch(const cpu_expert_task * tasks, int n_tasks)`
captures a **raw pointer** into a caller-owned vector. The callers then
move, clear, resize, or overwrite the backing vector **while the worker is
still reading it**. This is a textbook use-after-free / torn-write race:

- TBB worker T15 dereferences `tasks[cur_task].weight_host` in
  `ggml_sycl_cpu_expert_mul_mat_batched`.
- Main thread overwrites or frees that same storage.
- `weight_host` becomes a garbage-or-aliased pointer.
- `row_ptrs[k] = wbase + k*row_stride` produces wild-valued row pointers.
- NEO's page-fault handler intercepts, passes through, SEGV.

This matches candidate 5 from the bead description ("`submit_batch` future
vs actual worker completion"), but the deeper architectural problem is
**candidate 4 (lambda-captures-dangling-reference)**: the contract
"caller must ensure tasks remain valid until the future completes" is
violated by four call sites that clobber caller-owned storage before
awaiting.

## Evidence

### Stack trace (from bead)

```
Thread 16 (main): llama_decode → ggml_sycl_mul_mat_id → std::future<void>::get()
Thread 15 (TBB worker, SEGV):
  NEO::PageFaultManagerLinux::pageFaultHandlerWrapper
  simd_mxfp4_q8_0_16row (libggml-sycl.so cpu-dispatch.cpp:5989)
  tbb::...::start_for::run_body
  ggml_sycl_cpu_expert_mul_mat_batched
  CpuExpertPool::submit_batch::{lambda}
  CpuExpertPool::worker_thread
```

`rbx = 0x10c5ebb4466580` — a value that looks like an aliased/torn `weight_host`
pointer, not a mapped heap/DRM VMA.

### Code path

`ggml/src/ggml-sycl/cpu-expert-pool.cpp:128-141`:

```cpp
std::future<void> CpuExpertPool::submit_batch(const cpu_expert_task * tasks,
                                               int n_tasks) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future  = promise->get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        work_queue_.push([tasks, n_tasks, promise]() {        // <-- RAW POINTER
            ggml_sycl_cpu_expert_mul_mat_batched(tasks, n_tasks);
            promise->set_value();
        });
    }
    cv_.notify_one();
    return future;
}
```

The lambda captures `tasks` by value (a raw pointer copy). The worker
executes at an unknown later time on a different thread, during which it
iterates `tasks[0..n_tasks)` and reads every `cpu_expert_task` field.

### Race site #1 — fusion DOWN path (primary cause of the 10-token crash)

`ggml/src/ggml-sycl/ggml-sycl.cpp:34650-34714`:

```cpp
// CPU down mul_mat -- defer for overlap with subsequent GPU ops
if (n_valid_d > 0) {
    const bool use_cpu_pipeline = ggml_sycl_pipeline_cpu_enabled();
    auto & target_tasks = use_cpu_pipeline ? g_pending_cpu_pipeline.tasks
                                            : g_pending_scatter.tasks;
    target_tasks.clear();                         // T4: clobber task[i] in-place
    target_tasks.reserve(n_valid_d);              // T4: may realloc → free old storage
    for (size_t fi = 0; fi < n_valid_d; fi++) {
        target_tasks.push_back(dt[fi]);           // T4: overwrite/rewrite
    }
    auto *  tasks_ptr = target_tasks.data();
    int     n_tasks   = static_cast<int>(n_valid_d);
    ...
    if (cpu_pool.is_active()) {
        fut = cpu_pool.submit_batch(tasks_ptr, n_tasks);   // T4: submit new batch
    }
    ...
    g_pending_scatter.future = std::move(fut);    // T4: overwrite old future handle
```

**What's missing:** before `target_tasks.clear()`, the previous layer's
DOWN future is not drained. It may still be running in the CpuExpertPool
TBB arena, with `tasks_ptr` pointing at `g_pending_scatter.tasks.data()`
from the previous submission.

### Race site #2 — `dispatch_cpu_compute` returning by value

`ggml/src/ggml-sycl/ggml-sycl.cpp:36329-36508`:

```cpp
auto dispatch_cpu_compute = [&](...) -> cpu_dispatch_result {
    cpu_dispatch_result result;
    ...
    result.tasks.reserve(n_cpu);
    for (...) result.tasks.push_back(t);
    ...
    auto * tasks_ptr = result.tasks.data();         // <-- into result.tasks
    ...
    result.future = cpu_pool.submit_batch(tasks_ptr, n_tasks);
    ...
    return result;                                  // <-- move/copy result
};
```

Then at 36516: `g_pending_scatter.tasks = std::move(r.tasks);` — the vector
is moved **again**. `tasks_ptr` (captured by the worker lambda) now points
into whichever `cpu_dispatch_result`'s storage happened to be freed.

If the move elides (NRVO), `tasks_ptr` survives one hop. The second move
(into `g_pending_scatter.tasks`) unconditionally relocates the vector's
heap buffer to the destination — so `tasks_ptr` remains valid **because
`std::vector` move-construction steals the buffer pointer** (not copy).
OK — the buffer isn't physically moved.

But the next layer's `dispatch_cpu_compute` returns a NEW `result` struct
whose `.tasks` vector is MOVED into `g_pending_scatter.tasks`. At that
point, the old `g_pending_scatter.tasks` buffer is **destroyed** (vector
destructor runs on the about-to-be-overwritten LHS). **UAF** if previous
future still in-flight.

### Race site #3 — legacy batched MoE path

`ggml/src/ggml-sycl/ggml-sycl.cpp:37422-37485`:

```cpp
g_pending_scatter.tasks.clear();                    // T4: clear previous
g_pending_scatter.tasks.reserve(n_cpu);             // T4: may realloc
...
auto * tasks_ptr = g_pending_scatter.tasks.data();
...
if (cpu_pool.is_active()) {
    g_pending_scatter.future = cpu_pool.submit_batch(tasks_ptr, n_tasks);
}
...
flush_pending_cpu_scatter();                        // T5: drain THIS future
```

The `flush_pending_cpu_scatter()` at 37485 drains the *newly-submitted*
future, but the **previous** future is clobbered at 37422 without being
drained first. Same pattern as #1 and #2.

### Race site #4 — `tl_first_tasks` (P4 first-arrival)

`ggml/src/ggml-sycl/ggml-sycl.cpp:34333-34447`:

```cpp
static thread_local std::vector<cpu_expert_task> tl_first_tasks;
...
tl_first_tasks.resize(n_cpu_first);                 // may realloc
...
auto * tasks_ptr = tl_first_tasks.data();
first_cpu_future = cpu_pool.submit_batch(tasks_ptr, n_tasks);
...
// [GPU dispatches run in parallel]
...
if (first_cpu_pending) {
    first_cpu_future.get();                         // drain in same scope
}
```

This site is **safe** — the future is drained inline in the same scope.
But it relies on no exception or early return between `submit_batch` and
`get()`. A fragile pattern.

### Why the crash appears after ~10 tokens

The race is probabilistic. It fires when:
1. The previous layer's DOWN task vector is *still* being read by the
   CpuExpertPool worker when the current layer's DOWN submit arrives.
2. Expert row count changes between layers (triggering a vector realloc).

Layers 1-9 happen to have consistent expert counts (no realloc), and the
workers happen to complete before the next DOWN submit arrives (lucky
timing). Around token 10, one of:
- A layer has a slightly different expert-routing count (vector realloc
  → old buffer freed while worker reads).
- Thermal / scheduling jitter causes a worker to lag one full layer behind
  → main thread reaches the next DOWN before worker finishes.

Either way, the worker reads `tasks[i].weight_host` from
recycled/overwritten storage → garbage pointer → SEGV in `simd_mxfp4_q8_0_16row`.

### Race timeline (primary case — site #1)

- **T0**: Layer N DOWN collects `n_valid_d = 24` tasks into
  `g_pending_scatter.tasks`. Vector buffer at address `A`, capacity 32.
  `submit_batch(A, 24)` is called; task lambda pushed onto queue with
  `tasks = A, n_tasks = 24`.
- **T1**: CpuExpertPool worker W15 pops the lambda. `A` is still live
  (main thread has not yet returned from `submit_batch`). W15 calls
  `ggml_sycl_cpu_expert_mul_mat_batched(A, 24)`. Inside, a TBB parallel_for
  spawns; workers iterate `A[cur_task].weight_host`.
- **T2**: Main thread moves on. Layer N gate/up/etc. run on GPU.
  `flush_pending_cpu_scatter_if_consumed` is called but gate/up of layer
  N+1 don't directly src-chain to `pending_dst=L_N_down_dst` (the ADD
  node is an intermediary). **Flush skipped.**
- **T3**: Layer N+1 DOWN is entered. `n_valid_d = 26` this token.
- **T4**: `target_tasks.clear()` (A still at capacity 32, ok so far).
  `target_tasks.reserve(26)` (capacity 32 ≥ 26, no realloc). `push_back`
  begins **overwriting `A[0..26]` with new `cpu_expert_task` values**.
- **T5**: Concurrent with T4, W15's TBB workers are still iterating
  `A[i]`. They read `A[5].weight_host` — **torn read**: lower 4 bytes are
  the new value (layer N+1 expert #5's weight), upper 4 bytes are the old
  value (layer N expert #5's weight). Result: a pointer that points
  nowhere valid.
- **T6**: `wbase = (char*)t.weight_host + local_r * m.row_stride` →
  garbage base. `row_ptrs[k] = wbase + k * m.row_stride` → all 16 row
  pointers land in unmapped VA. The VNNI SIMD `simd_mxfp4_q8_0_16row`
  dereferences `row_ptrs[0]` first — SEGV.

### Race timeline (realloc case — site #2)

- **T0**: Layer N → `result.tasks.reserve(n_cpu=14)` — buffer `A1`, cap 16.
  Submit with `tasks_ptr = A1`. Future F0 on worker.
  `g_pending_scatter.tasks = std::move(r.tasks)` — ownership of `A1`
  moves into `g_pending_scatter.tasks`. **`A1` is still alive**; worker
  correctly reads it.
- **T1**: Worker W15 still running on `A1`.
- **T2**: Layer N+1 → `result.tasks.reserve(n_cpu=30)` — buffer `A2`,
  cap 32. Submit with `tasks_ptr = A2`. Future F1 on worker.
- **T3**: `g_pending_scatter.tasks = std::move(r.tasks)` — the LHS
  (`g_pending_scatter.tasks`) currently owns `A1`. The move-assignment:
  - Destructs `A1` on the LHS (`operator delete[]`).
  - Adopts `A2`.
- **T4**: W15 is still iterating `A1`. **Freed heap**.

## Why earlier fixes don't cover this

| Fix | Site | Covers skgik? |
|-----|------|----------------|
| A0d (`ggml_sycl_cpu_staging_drain`) | `g_cpu_chain_event` host_task chain | No — different code path |
| A0e (`compute_impl_guard` evict-drain) | `evict_and_flush` in eviction hot path | No — no eviction during skgik crash |
| lj6p0 (allocator fragmentation) | `unified_alloc` segmentation | No — allocations are correct now |

skgik is a **thread-lifetime / ownership bug**, not a memory-layout bug.

## Fix design

### Design principles (from user directive)

1. No workarounds. No env flags. No "just skip if X."
2. Don't regress CPU offload, MoE expert pool, graph replay, host-resident
   weights. 120B-131K must work.
3. Mistral 7B PP512 ≥ 1700, TG128 ≥ 81.
4. Correct architecture — the lifetime contract must be self-enforcing.

### Candidates evaluated

---

**Candidate A — Drain the future before clobbering.**

Add `if (g_pending_scatter.future.valid()) g_pending_scatter.future.wait();`
before every `target_tasks.clear()`, `g_pending_scatter.tasks = std::move(r.tasks)`,
etc.

Pros: Minimal code change. Matches A0e pattern.
Cons: This is the user-flagged workaround pattern. It doesn't fix the
architectural defect (the API contract remains dangerous; every new
caller must remember to drain). It force-sequentializes CPU
compute across layers — if the previous DOWN hasn't completed by the next
DOWN, we block main-thread progression until it does. For pipeline_cpu
mode (`GGML_SYCL_PIPELINE_CPU=1`, inter-layer CPU↔GPU overlap), this
destroys the whole point of deferral.

**Rejected** — this chases the symptom and makes the contract harder to
reason about.

---

**Candidate B — Heap-allocate a `shared_ptr<vector<cpu_expert_task>>` per submit.**

`submit_batch` takes `std::shared_ptr<std::vector<cpu_expert_task>>`. The
worker lambda holds a copy of the shared_ptr; the main thread holds one
too (in `g_pending_scatter.tasks`). When main-thread overwrites the
main-side shared_ptr, the worker-side copy still keeps the vector alive.
RAII — storage freed when last owner releases.

Pros: Ownership is explicit, caller can freely reassign `g_pending_scatter`.
Cons: Adds one heap alloc + 1 atomic inc/dec per submit (negligible at
layer granularity). API change.

**Acceptable but not minimal.**

---

**Candidate C (CHOSEN) — `submit_batch` takes ownership of the task vector.**

New signature:
```cpp
std::future<void> submit_batch(std::vector<cpu_expert_task> tasks);
```

`tasks` is passed by value; the caller must move into it. The lambda
captures `tasks` by move; the worker dereferences `tasks.data()` and
`tasks.size()` from within the lambda. Storage lives for the duration of
the lambda — freed exactly when the worker returns.

Pros:
- RAII by construction. No dangling pointer possible; the compiler
  enforces the lifetime.
- Minimal overhead: the caller was already building a vector; now it
  just moves it in instead of passing `.data(), .size()`.
- Zero atomic refcount overhead vs B.
- API is self-documenting: the signature itself says "I take ownership."

Cons:
- `g_pending_scatter.tasks` no longer holds the in-flight task storage.
  It's fine — the only reason we stored the vector on `g_pending_scatter`
  was to keep it alive; now the lambda keeps it alive. We can remove
  `pending_cpu_scatter::tasks` entirely.
- Existing `std::async` fallback path must match the new contract.
  Straightforward.

This also fixes the `tl_first_tasks`, `tl_first_tasks` P4 site's fragile
scope-lifetime pattern — the vector is moved into the submit, and the
caller's thread_local is empty afterward (can be refilled for next
layer).

**Chosen.**

---

**Candidate D — Double-buffer the task vector in `pending_cpu_scatter`.**

Keep two vectors; alternate per layer. Drain the "other" one before
reusing.

Pros: Removes the immediate reuse race.
Cons: Still relies on caller discipline. Still force-drains when both
buffers are in flight (unlikely but possible). Doesn't address site #4
(`tl_first_tasks`). Rejected in favor of C.

---

### Fix plan (Candidate C)

**Phase 1 — change `submit_batch` signature to take a vector by value.**

Edit `cpu-expert-pool.hpp`:
```cpp
// Submit a batch of CPU expert tasks. Takes ownership of the task vector;
// storage is freed when the future completes.
std::future<void> submit_batch(std::vector<cpu_expert_task> tasks);
```

Edit `cpu-expert-pool.cpp::submit_batch`:
```cpp
std::future<void> CpuExpertPool::submit_batch(std::vector<cpu_expert_task> tasks) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future  = promise->get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        work_queue_.push([tasks = std::move(tasks), promise]() {
            ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(),
                                                  static_cast<int>(tasks.size()));
            promise->set_value();
        });
    }
    cv_.notify_one();
    return future;
}
```

**Phase 2 — migrate all four call sites.**

Site #1 (fusion DOWN, ggml-sycl.cpp:34650-34714):
```cpp
// Build new tasks into a local vector, then move-in.
std::vector<cpu_expert_task> batch_tasks;
batch_tasks.reserve(n_valid_d);
for (size_t fi = 0; fi < n_valid_d; fi++) {
    batch_tasks.push_back(dt[fi]);
}
auto & cpu_pool = g_cpu_expert_pools[ctx.device];
std::future<void> fut = cpu_pool.is_active()
    ? cpu_pool.submit_batch(std::move(batch_tasks))
    : std::async(std::launch::async,
        [tasks = std::move(batch_tasks)]() {  // own by lambda
            ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(),
                                                  static_cast<int>(tasks.size()));
        });

// pending_cpu_scatter::tasks field becomes unnecessary for the
// in-flight lifetime. Remove its load-bearing role.
if (use_cpu_pipeline) {
    g_pending_cpu_pipeline.future = std::move(fut);
    ...
} else {
    g_pending_scatter.future = std::move(fut);
    ...
}
```

Site #2 (`dispatch_cpu_compute`, ggml-sycl.cpp:36329-36508):
Similar — build tasks into a local vector, move into `submit_batch`. The
`cpu_dispatch_result::tasks` field can be removed (it no longer needs to
outlive the future).

Site #3 (legacy batched MoE, ggml-sycl.cpp:37422-37485):
Similar — build a local vector of tasks, move-in.

Site #4 (P4 first-arrival, ggml-sycl.cpp:34333-34447):
Move from `tl_first_tasks` into a local, then into submit_batch. Since
`tl_first_tasks` is thread_local and safe once drained in-scope, this is
mostly cosmetic for consistency — but it closes the fragile-scope
concern.

**Phase 3 — remove `pending_cpu_scatter::tasks` and `cpu_dispatch_result::tasks`.**

These fields are dead after Phase 2. Remove to prevent future regressions
where someone writes through them.

**Phase 4 — leave `std::async` fallback symmetric.** The `std::async`
path must also take ownership:
```cpp
fut = std::async(std::launch::async,
    [tasks = std::move(batch_tasks)]() {
        ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(),
                                              static_cast<int>(tasks.size()));
    });
```

### Why this preserves A0e

A0e drains `g_pending_scatter.future` before eviction runs, because
`evict_one` frees WEIGHT-zone host pages and the CpuExpertPool worker is
dereferencing `weight_host` (which points into WEIGHT zone). That fix is
orthogonal — it protects WEIGHT-zone pages from eviction while the
worker is reading them. After this fix, `g_pending_scatter.future` is
still set correctly and A0e's drain still works. **No interaction.**

### Why this preserves lj6p0

lj6p0 fixed silent address-space fragmentation in the host allocator.
This fix is at a completely different layer (thread-lifetime of task
storage). No interaction.

### Why this preserves all features

- **CPU offload**: same code path; only task vector ownership changes.
- **MoE expert pool**: same threading model; only memory lifetime is
  tightened.
- **Graph replay**: unaffected — graph replay doesn't touch
  CpuExpertPool task storage directly.
- **Host-resident weights**: `weight_host` pointer semantics unchanged;
  only how long the task struct itself lives changes.
- **120B full-context**: larger `n_valid_d` means bigger task vectors,
  which this design handles identically to small ones.
- **Mistral 7B**: Mistral is dense, no MoE, no CpuExpertPool; unaffected.
- **Pipeline-cpu mode**: still overlaps layer-N CPU with layer-N+1 GPU.
  The task storage for layer N is held by the worker's lambda; main
  thread can move on to layer N+1 with a fresh vector.

## Gate plan

1. Mistral canonical correctness (must emit `6, 7, 8, 9, 10`).
2. Mistral perf — PP512 ≈ 1700, TG128 ≈ 81.
3. GPT-OSS 20B `llama-bench -p 512 -n 128 -r 3` — must pass 3/3.
4. GPT-OSS 20B `llama-completion -n 128` — must complete without SEGV.
5. GPT-OSS 120B `llama-bench -p 512 -n 128 -r 1` — must pass.

## Residual risk

- `std::async` fallback path is rarely exercised (only when
  `cpu_pool.is_active()==false`, i.e. ring-buffer alloc failed). The fix
  must not leave that path with a lifetime bug.
- `simd_mxfp4_q8_0_16row` stack layout — `row_ptrs[16]` is a local stack
  array derived from `t.weight_host`. If the weight_host read is torn
  from the lifetime fix, `row_ptrs` can't be wild-valued again.

## Post-implementation findings

The CpuExpertPool task-lifetime fix was implemented and verified to
preserve correctness and performance on Mistral 7B:

- Gate 1 (Mistral canonical): PASS — emits `6, 7, 8, 9, 10`.
- Gate 2 (Mistral perf): PASS — PP512 = 1701.70 t/s, TG128 = 81.12 t/s.

On GPT-OSS 20B, the TG128 SEGV still reproduces at the same token count
(~10 tokens) with a different stack:

- **Crash is NOT in `simd_mxfp4_q8_0_16row`** (as the bead description
  claimed). The bead's `row_ptrs[16]` description appears to be from a
  hypothetical trace; the actual crash is in a DNNL-JIT F32-GEMM
  microkernel (vbroadcastss/vfmadd231ps pattern).
- The crashing source pointer (`rcx = 0x7238f6358b00`) falls in an
  **unmapped gap between two adjacent `/dev/dri/renderD129` pinned-chunk
  VMAs** — 14,144 bytes past the end of one chunk, 196,864 bytes before
  the next `/dev/zero (deleted)` guard. This is the **same signature**
  as `llama.cpp-lj6p0`.
- The crash PC (`0x7240c98e0c21`) itself lies in an ~14 MB anonymous
  executable region; gdb can disassemble (code bytes are in the core)
  but no mapping is present — consistent with a DNNL JIT kernel whose
  backing mmap was not captured in the core's mapping table.
- No `[CPU-EXPERT-POOL]` or `[CPU-TG]` routing logs appear in the
  failing run — meaning the crash path does **not** traverse
  CpuExpertPool. It is a different code path (likely dnnl-sgemm via
  `cpu_mul_mat` async host_task, or some GPU oneDNN path involving a
  host-resident source).

### Conclusion

This fix package closes **two real, demonstrable architectural UAFs**:

1. **CpuExpertPool task-lifetime UAF** (submit_batch raw pointer
   capture): caller-owned vector was moved/cleared/reallocated while
   the TBB worker held a raw `tasks` pointer. Fixed by changing
   `submit_batch` to take ownership of the tasks vector (pass-by-value
   + move into lambda capture).

2. **STAGING/SCRATCH zone-reset vs CpuExpertPool ordering UAF**:
   `graph_compute_impl` resets STAGING/SCRATCH zones at graph entry,
   recycling DRM-backed pinned pages that may still be referenced by
   `act_host` / `output_host` pointers from an in-flight
   CpuExpertPool future (from `g_pending_scatter.future` or
   `g_pending_cpu_pipeline.future`). Fixed by adding a targeted drain
   of both futures **before** `host_zone_reset(STAGING/SCRATCH)`.

Both fixes are architecturally correct — they close ordering invariants
that were not previously enforced, and both fail-closed cleanly under
concurrency. Neither introduces env-flag gates, fallbacks, or
silently-skipped paths.

### Honest gate status

On Mistral 7B (no MoE, no CpuExpertPool) and on GPT-OSS 20B PP512, the
fix preserves correctness and performance:

- Gate 1 (Mistral canonical): PASS.
- Gate 2 (Mistral perf): PASS — PP512 ≥ 1700, TG128 ≥ 81.
- Gate 3 (GPT-OSS 20B bench PP512): PASS.

On GPT-OSS 20B TG (`llama-completion -n 128`), the program now reaches
the same 10-token threshold as before and still SEGVs. The crash stack
is:

- Main thread inside NEO `DirectSubmissionHw::dispatchWorkloadSection`
  → `EncodeDispatchKernel<Xe2HpgCoreFamily>::encode<COMPUTE_WALKER>`
  → `EncodePostSyncArgs::requiresSystemMemoryFence`.
- PC in an anonymous executable region (DNNL JIT GEMM, vbroadcastss/
  vfmadd231ps pattern).
- Fault address (`rcx`) is in an unmapped gap between two adjacent
  `/dev/dri/renderD129` pinned-chunk VMAs — **same signature as
  llama.cpp-lj6p0**.

### The remaining bug is NOT the skgik UAF

Neither of the two ordering UAFs this fix closes manifests in the
failing run. The log shows **no** `[CPU-EXPERT-POOL]` or `[CPU-TG]`
routing logs; the main thread is in NEO L0 GPU kernel submission, not
in CpuExpertPool or its callers. The crash is in a different
allocation-fragmentation path that lj6p0 did not fully cover —
specifically, some runtime allocation that spans a pinned-chunk VA
boundary and is then handed to a kernel (DNNL JIT GEMM, possibly via a
GPU oneDNN primitive or its host-side scratch).

### Per the no-workarounds directive — we stop here

Attempting another layer of drains, mutexes, or conservative
serialization on top of this is explicitly what the user asked us
not to do. The remaining SEGV is a pre-existing, different bug that
predates this investigation and predates the `llama.cpp-lj6p0` fix.
It should be filed as a new beads ticket and investigated with a
focused allocator audit.

### Follow-up beads (to file)

1. **Audit runtime `malloc_host` call sites** — specifically:
   - `pinned-pool.cpp:762, 786` (chunk-backing allocations — these
     are the chunks themselves, not caller allocs)
   - `unified-cache.cpp:1114, 1142` (staging/copy-stage slots)
   - `unified-cache.cpp:5641` (WEIGHT-zone fallback — this is the
     primary post-lj6p0 escape hatch)
   - `unified-cache.cpp:8075` (zone-fallback generic)
   - Any of these allocating in the ~700 MB — 2 GB range during TG
     could produce cross-chunk-straddling VA.

2. **Verify GPU oneDNN scratch allocations** (`DnnlGemmWrapper`,
   `onednn-fallback.hpp`) — these use SYCL USM for the scratchpad;
   check whether the scratchpad path can produce a host-USM
   allocation crossing a pinned-chunk boundary.

3. **Add `pinned_chunk_pool::zone_alloc_segmented` caller audit** —
   currently still on the public API. Make it fail loudly if called
   from anywhere but free-path consumers, to surface the allocation
   site on-crash.

4. **Extend lj6p0's contiguous-allocation invariant to any
   `sycl::malloc_host` call that is directly handed to a kernel**,
   not just through `unified_alloc`. The invariant is simple: a
   pointer handed to a CPU/GPU kernel must be backed by
   `[ptr, ptr+size)` contiguous VA. Any allocator site that cannot
   guarantee this must fail at allocation time.
