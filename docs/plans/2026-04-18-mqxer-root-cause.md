# mqxer root-cause evidence (A0 research packet)

`llama.cpp-1yw54` (A0) investigation, 2026-04-18, HEAD **56ead4cce** on
`feature/sycl-coalescing`.

## TL;DR

- `llama.cpp-mqxer`'s crash is a SIGSEGV inside oneDNN's JIT'd
  `gemv_kernel_driver<float,float,float>` CPU kernel, called via
  `dnnl_sgemm` at `ggml/src/ggml-sycl/cpu-dispatch.cpp:4191`
  (inside `cpu_mul_mat()` at line 3770).
- Workload: GPT-OSS 20B MoE **router / gate projection** during token
  generation. Shape `M=32 experts × N=1 token × K=n_embd=2880`.
- **Important correction vs original A0 fix sketch:** the fault address
  does NOT fall inside any of the three pointers we pass to
  `dnnl_sgemm` (`weight_f32`, `src1_batch`, `dst_batch`). The crashed
  thread is a oneDNN worker dereferencing a pointer that was valid at
  dispatch time but is no longer mapped by the time the worker reads
  it. The simple "pad the weight tensor" fix does NOT address the
  observed crash.

See **"Corrected root cause"** section below for the refined theory and
the updated fix direction.

## Reproduction

HEAD `56ead4cce`, build flags per `CLAUDE.md`.

```bash
source /opt/intel/oneapi/setvars.sh --force
ulimit -c unlimited
ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 180 gdb -batch \
  -x <(echo 'set pagination off
handle SIGSEGV stop print nopass
run
info registers
bt 15
quit') --args ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-mxfp4.gguf -p 512 -n 128 -r 1
```

Deterministic: three separate runs, all reproduce. PP512 completes
(~55 t/s); SIGSEGV in a non-main thread between the PP phase summary
and the TG128 line. `llama-bench` prints the `pp512` row, then dies.

## Backtrace at fault

```
Thread NN "llama-bench" received signal SIGSEGV, Segmentation fault.
0x00007ffe54869100 in ?? ()                                  ← JIT frame, no symbol
#0  0x00007ffe54869100 in ?? ()
#1  0x00007ffda7fcd450 in ?? ()
#2  0x00007ffda7fcd430 in ?? ()
#3  0x00007ffda7fcd440 in ?? ()
#4  0x0000000012c71780 in ?? ()
#5  0x0000000000000090 in ?? ()
#6  0x00007ffda7fcddb0 in ?? ()
#7  0x00007fffee338e34 in
    dnnl::impl::cpu::x64::gemv_kernel_driver<float,float,float>(...)
    from /opt/intel/oneapi/dnnl/2025.3/lib/libdnnl.so.3
Backtrace stopped: frame did not save the PC
```

Frames 0-6 are JIT-generated (xbyak-emitted) and have no DWARF, so
the real caller chain ends at frame 7. Anything higher requires
either compiled-with-FP oneDNN or a different diagnostic approach
(`libdnnl.so.3` as shipped by oneAPI 2025.3 is compiled without
frame pointers in this kernel).

### Faulting instruction

```
=> 0x7ffe5486922c:   vmovups -0x80(%rdx,%r8,2),%ymm0   ← fault here (1st repro)
   0x7ffe5486922c:   vfmadd231ps %ymm0,%ymm11,%ymm1
   0x7ffe54869238:   vmovups -0x60(%rdx,%r8,2),%ymm0
   0x7ffe5486923f:   vfmadd231ps %ymm0,%ymm11,%ymm2
```

AVX-256 vector load with indexed addressing. This is the hot inner loop
of gemv's SIMD accumulate (B[k] × a[k] FMA unrolled into 4 YMM regs).

### Register state (2nd instrumented repro)

```
rax            0x1                 1
rbx            0x90                144
rcx            0x7ff7cbb6f300
rdx            0x7ff7cbb63f00       ← base pointer for the faulting load
rsi            0x90                144
rdi            0x20                32
r8             0x2d00              11520  = K × sizeof(float) = 2880 × 4
r9             0x7ffaf4a94980
r10            0x2d00              11520
r11            0x12c71800
r12            0x12c71780
r13            0x7ff7cbb58b00       ← another suspicious heap ptr
r14            0x12c71800
r15            0x8700              34560
rip            0x7ffe54869100
```

Effective-address math for the faulting vmovups:
`rdx + r8*2 - 0x80 = 0x7ff7cbb63f00 + 0x5a00 - 0x80 = 0x7ff7cbb69880`.

## dnnl_sgemm-call trace (gdb breakpoint)

Running llama-bench under gdb with a breakpoint at `dnnl_sgemm` entry
and pretty-printing the first few integer/pointer args, we observe only
two distinct M/N/K shape combos across the run:

```
[dnnl_sgemm] M=32 N=512 K=2880   ← PP512 router gemm, M outputs × N tokens × K n_embd
[dnnl_sgemm] M=32 N=512 K=2880
...   (many of these during PP)
[dnnl_sgemm] M=32 N=1   K=2880   ← first TG router gemv
[dnnl_sgemm] M=32 N=1   K=2880
[dnnl_sgemm] M=32 N=1   K=2880
...
[dnnl_sgemm] M=32 N=1   K=2880   ← ~15 successful TG calls
Thread NN "llama-bench" received signal SIGSEGV, Segmentation fault.
```

N=32 matches GPT-OSS 20B's `hparams.n_expert == 32`; K=2880 matches
its `hparams.n_embd == 2880`. The crashed call is a TG (M=1 token)
router/gate projection: `32 logits = W^T [32×2880] × x [2880 floats]`.

## Instrumented pointer capture (Follow-up 1)

Temporarily added fprintf immediately before the `dnnl_sgemm` call at
cpu-dispatch.cpp:4191 to log the three pointer args + their
theoretical end addresses. (Instrumentation was backed out after the
capture — not in the committed tree.)

Snippet of the captured log (around the crash):

```
[A0-DEBUG] weight_f32=0x7ff84d8e5100 end_w=0x7ff84d93f100
           src1_batch=0x7ffaf312e100 end_s1=0x7ffaf3130e00
           dst_batch=0x7ffaf36ee100 end_dst=0x7ffaf36ee180
           M=1 N=32 K=2880 weight_ld=2880 ldc=32
[A0-DEBUG] weight_f32=0x7ff8688aa980 end_w=0x7ff868904980
           src1_batch=0x7ffaf36ee100 end_s1=0x7ffaf36f0e00
           dst_batch=0x7ffaf36ee100 end_dst=0x7ffaf36ee180
           M=1 N=32 K=2880 weight_ld=2880 ldc=32
[A0-DEBUG] weight_f32=0x7ff883870200 end_w=0x7ff8838ca200 ...
[A0-DEBUG] weight_f32=0x7ff89e835a80 end_w=0x7ff89e88fa80 ...   ← last logged call
Thread 20 "llama-bench" received signal SIGSEGV     (rdx=0x7ff7cbb63f00)
```

Full 65-call capture preserved at `/tmp/a0-fu/debug-calls.txt`
(on this box; ephemeral).

### Cross-check: fault address vs ALL logged pointer ranges

Scanned all 65 `[A0-DEBUG]` lines, checking whether the fault address
`rdx=0x7ff7cbb63f00` (or the neighbor `r13=0x7ff7cbb58b00`) falls within
any `[weight_f32, end_w)`, `[src1_batch, end_s1)`, or
`[dst_batch, end_dst)` range:

| ptr class | ranges scanned | hits for fault addr | hits for r13 |
|-----------|----------------|----------------------|--------------|
| weight_f32 range | 65 | 0 | 0 |
| src1_batch range | 65 | 0 | 0 |
| dst_batch range  | 65 | 0 | 0 |

**The fault address sits outside every argument buffer we passed to
`dnnl_sgemm`.** Nearest weight_f32 start `<= fault` is
`0x7ff7ca208a80` — 26 MB before the fault address, far past the end
of its 368 KB range.

### Interpretation

The faulting thread is NOT processing any of the submissions our
single-threaded instrumentation observed. Given oneDNN's threadpool
launches workers that run concurrently with the submitter thread, the
most consistent interpretation is:

1. The submitter calls `dnnl_sgemm(N)` which enqueues tile work into
   oneDNN's internal threadpool with pointers `weight_f32[N]`,
   `src1_batch[N]`, etc. Workers pick up the tiles.
2. Submitter proceeds to `dnnl_sgemm(N+1)`, `dnnl_sgemm(N+2)`, etc.
3. Between those calls, something in **our** code frees or recycles a
   buffer that a still-running worker's tile pointer references.
4. The worker reaches the next page boundary of its strided load,
   which is now unmapped — SIGSEGV.

The recycled buffer is most likely a **staging buffer** — specifically
one produced by `get_host_ptr` → `staging_ensure` for one of the
activation tensors. Staging banks in `cpu-dispatch.cpp` rotate via
`g_staging_bank = 1 - g_staging_bank` at `staging_begin_op()`
(cpu-dispatch.cpp:2297, 2323), and the pool can release the old lease
if the bank is reused before the worker has finished reading from it.

### What this means for the fix

The original A0 fix sketch (copy router F32 weights to `scratch_nk` to
provide tail slack) was predicated on the fault being an over-read
past the end of an F32 mmap'd weight tensor. **That predicate is
false.** The weight pointer is not in the fault range, and mmap'd
weights are stable for the process lifetime anyway.

The corrected fix direction: ensure **activation / output staging
buffers retain the ownership for the lifetime of the oneDNN task**,
not just until the submitter returns. Two viable shapes:

- **Synchronous**: call `dnnl_sgemm` synchronously (it's actually
  blocking on the caller thread — oneDNN_CPU_ENGINE sgemm is
  synchronous from the caller's POV — so this may already be the case,
  need to verify) AND ensure staging bank rotation happens
  AFTER the call returns. If staging is rotated mid-call by another
  cpu_mul_mat on a different thread, that's the bug.
- **Async with proper fencing**: if oneDNN returns asynchronously via
  a threadpool, we need to either drain oneDNN's pool before reusing
  any staging buffer, OR take a ref on the staging buffer held until
  the oneDNN call completes.

Further narrowing will come from:
- Reading the dnnl_sgemm implementation (is it blocking on the
  submitter thread? or does it hand to a pool and return?)
- Checking `g_cpu_dispatch_buffers` vs `g_staging_bank` interactions
  for any path that can mutate pool state mid-gemm.

Not investigated in this A0 pass; flagged as a NEW open question
superseding the original "pad the weight tensor" sketch.

## What this rules out

- **Not CPU-heap UAF of a buffer allocated by `cpu-dispatch.cpp`
  itself.** `g_cpu_dispatch_buffers.src1_q` / `scratch_nk` are
  thread-local `std::vector`s with stable heap storage after
  `resize()`. The prior investigation was correct that these are
  stably allocated.
- **Not an over-read past the end of an mmap'd F32 router weight.**
  The fault address is nowhere near the weight ranges.
- **Not the `g_moe_expert_biases` cache path (implementer-2's GAP-2).**
  That cache is consumed by the fused ADD_ID kernel, not by
  `dnnl_sgemm`'s gemv kernel. See my prior reply on GAP-2 for full
  reasoning.
- **Not a BCS CAT error class.** This is a pure CPU-side SIGSEGV;
  dmesg shows no `bcs` / `engine reset` / `FaultLevel` entries during
  repro on HEAD 56ead4cce. (The original mqxer bead mentioned those
  from a different branch state; on 56ead4cce the crash is CPU-only.)

## Evidence file manifest (ephemeral — on this box only)

`/tmp/a0-cores/` (first repro round, ABI-probing):
- `repro1.log` – initial bt showing gemv_kernel_driver frame #7
- `repro2.log` – bt + info threads dump
- `repro3.log` – stack-memory dump around rsp
- `trace3.log` – dnnl_sgemm BP trace with corrected ABI ordering
- `trace4.log`, `trace5.log` – deeper arg probes (inconclusive, noted)

`/tmp/a0-fu/` (follow-up instrumented build):
- `repro.log` – gdb run with `[A0-DEBUG]` lines + fault register dump
- `debug-calls.txt` – 65 `[A0-DEBUG]` call captures extracted

These are on tmpfs; this document preserves the key data.

## References

- llama.cpp-mqxer (bug, open): the crash itself.
- llama.cpp-1yw54 (A0 research, this task): investigation.
- llama.cpp-goegc.1 (latent, open): stale pointer after eviction,
  related class but orthogonal mechanism.
- commit 14f6f8347: A4 test for mem_handle eviction lifecycle
  (companion, orthogonal to this crash).
- `memory/bug_bcs_cat_prestage.md`: prior BCS CAT bug (resolved by
  9ug6y). Different fault class — mqxer is not a BCS fault on
  56ead4cce.

## A0c follow-up — 2026-04-18 afternoon

Two corrections + one major discovery.

### Correction 1: dnnl_sgemm IS synchronous (A0b/A0c confirmed via source)

Runtime libdnnl version reconciliation (from A0b quality-review polish):
the canonical repro command sources `/opt/intel/oneapi/setvars.sh`,
which prepends `/opt/intel/oneapi/dnnl/2025.3/lib` to
`LD_LIBRARY_PATH`. With that sourced,
`ldd $(which llama-bench) | grep libdnnl` shows
`libdnnl.so.3 → /opt/intel/oneapi/dnnl/2025.3/lib/libdnnl.so.3 →
libdnnl.so.3.9`. Without `setvars.sh`, it falls back to
`/usr/local/lib/libdnnl.so.3 → libdnnl.so.3.11`. Our repro always
sourced setvars, so runtime = v3.9 and the source tree cloned at
`/Apps/oneDNN` (v3.9 tag) matches 1:1.

Verified by reading oneDNN v3.9 source at `/Apps/oneDNN` (all paths
below are under that checkout):

- `/Apps/oneDNN/src/common/gemm.cpp:97-110` — `dnnl_sgemm` calls
  `cpu::extended_sgemm` synchronously and returns the status after
  completion.
- `/Apps/oneDNN/src/cpu/gemm/gemm.cpp:106-150` `extended_sgemm` →
  either calls CBLAS (synchronous) or `gemm_driver` (x86-64 path).
- `/Apps/oneDNN/src/cpu/x64/gemm/gemm_driver.cpp:2045-2092`
  `gemm_driver` → builds args and calls `gemm_threading_driver`
  synchronously.
- `/Apps/oneDNN/src/cpu/x64/gemm/gemm_driver.cpp:1911`
  `parallel(nthr_spawn, [&]{...})` — this is a fork-join call.
- **Build-config check**: the runtime libdnnl our repro loads is
  `/opt/intel/oneapi/dnnl/2025.3/lib/libdnnl.so.3.9`. `ldd
  llama-bench` under the canonical setvars-sourced environment
  shows `libiomp5.so` linked (Intel's LLVM OpenMP), confirming
  `DNNL_CPU_RUNTIME=OMP` for that build. This rules out any
  alternate runtime (TBB/THREADPOOL/SEQ) that `dnnl_thread.hpp`'s
  `#if` guards would steer into.
- With OMP, `parallel()` expands to `#pragma omp parallel
  num_threads(nthr)` at `/Apps/oneDNN/src/common/dnnl_thread.hpp:290`,
  which is a barrier/join region. All OMP workers complete their
  closure before the parallel region exits.

So dnnl_sgemm cannot leave OMP workers in flight past its return.
My A0 theory ("workers still running when submitter rotates staging")
is **wrong**.

### Correction 2: staging-bank rotation contract is sound

Re-traced `staging_begin_op()` (cpu-dispatch.cpp:2292-2324) +
`cpu_submit_async()` (cpu-dispatch.cpp:232-262): the async-mode
bank rotation correctly waits on the NEXT bank's prior compute
event before reuse, and the compute event tracked via
`staging_track_cpu_event()` is the SYCL event of the host_task
wrapping run_mul_mat (which wraps the synchronous dnnl_sgemm).
The waited-on event correctly subsumes the dnnl_sgemm completion.

There is NO mutex on the static `g_staging_bank` variable, but the
operations that mutate it all run on the main ggml-sycl thread
(ggml_backend_sycl_graph_compute dispatches sequentially). The
SYCL host_task runs on a different thread, but it captures pointer
VALUES (not the bank index) by lambda `[=]` capture — so bank
rotation on the main thread doesn't affect the in-flight host_task's
view of its input pointers.

### Discovery: the fault address is a DRM-mapped region

**This is the key evidence.** Captured `info proc mappings` at the
SIGSEGV point via gdb:

```
RDX 0x7ff7cb9fbf00 IS IN:
  0x00007ff74ba00000 0x00007ff7cba00000 0x80000000  0x8d4b2a000  rw-s  /dev/dri/renderD129
R13 0x7ff7cb9f0b00 IS IN: (same range)
```

The faulting pointer is mapped from `/dev/dri/renderD129` — the Intel
Arc GPU DRM render-node device — via a **shared (`rw-s`) 2 GB mapping
at `[0x7ff74ba00000, 0x7ff7cba00000)`**. This is how Level Zero /
SYCL expose host-accessible USM (from `sycl::malloc_host` or
`sycl::malloc_shared`) to the CPU process: as GEM objects shared
into the user's address space via the DRM driver.

Cross-reference with A0's instrumented pointer capture: many of the
`weight_f32` addresses from `[A0-DEBUG]` lines fall WITHIN this DRM
VMA range, e.g. (NOTE: the `info proc mappings` dump was captured in
a **different process** than A0's `[A0-DEBUG]` pointer trace, so ASLR
makes exact per-pointer VMA containment non-deterministic across runs.
The cross-reference below should be read as "same allocation CLASS —
DRM-backed SYCL host-USM" rather than "this exact pointer was in this
exact VMA"):

- `0x7ff7ca208a80` (logged weight_f32) — equivalent DRM-backed class
- `0x7ff79427d980` — equivalent DRM-backed class
- `0x7ff75e2f2880` — equivalent DRM-backed class
- `0x7ff7cb9fbf00` (fault) — in this run's DRM VMA

A0d's retrofit run captured BOTH the `[A0D-*]` pointer events AND
the `info proc mappings` within the **same pid** — see the A0d
follow-up section below for the in-process-confirmed match.

These weight pointers are NOT plain mmap'd GGUF file bytes. They are
**SYCL host-USM allocations** handed back by the unified cache via
`get_host_ptr()` → `cache->get_view(key, AOS)` → `view.ptr` where
`view.location != cache_location::DEVICE` (cpu-dispatch.cpp:2403-2406).
`HOST_PINNED` entries in our cache are `sycl::malloc_host`-style, and
the DRM backing is what makes them GPU-accessible.

### Real root cause (now supported by evidence)

`error 4` on `vmovups` means "user-space read from a page that is not
present in the page table." The VMA exists (rw-s permissions set),
but a specific page inside it is unmapped at the fault moment. For a
DRM-backed buffer, this happens when:

1. The DRM driver has **migrated** the backing (e.g. between VRAM
   and pinned host) and unmapped the CPU page(s) before remapping
   at a potentially-different virtual address (for the next migration
   state), OR
2. The buffer was **freed** on the SYCL side (`sycl::free`), the DRM
   object released, and the CPU VMA kept stale.

Option 2 fits our evidence pattern: `sycl::malloc_host` → register
as host_ptr → dnnl_sgemm reads → concurrently freed (by unified cache
eviction, staging bank rotation finalize, or model unload) →
dereference after free → SIGSEGV.

This is still a use-after-free, but the "source" of the dangling
pointer is the **unified cache's host-pinned weight view** — not a
staging buffer or scratch_nk. A concurrent path (eviction, pool
reclaim, runtime category cleanup) is releasing the host-USM backing
while dnnl_sgemm is still reading from it.

### Specific hypothesis to confirm/refute

The concurrent-free is likely coming from:

- **unified_cache::evict_one's deferred_free path**: line 3898 or
  finalize_evictions_locked at :3968 — `enqueue_deferred_free(ptr, size)`
  which eventually calls `sycl::free` via `process_deferred_frees` at a
  safe yield point. If that yield point coincides with the main graph
  compute's next CPU_MUL_MAT while the host-pinned backing is still
  being read by oneDNN's OMP workers inside dnnl_sgemm, pages get
  torn down.
- **host_arena / host-zone reset**: if the zone is reset after a
  phase transition (PP512 → TG128 is a phase boundary in llama-bench),
  zone reset releases all allocations in bulk. An in-flight dnnl_sgemm
  reading a HOST_PINNED view of a weight (whose backing is in the
  released zone) would fault.

### Recommended fix direction (now evidence-backed)

1. **First**: trace the exact call that frees the host-USM backing
   at/around the mqxer crash. Instrument `process_deferred_frees`,
   `host_zone_reset`, and `unified_cache::evict_one` with a log of
   `(time, ptr, size)`. Run the repro and check whether any of these
   fires DURING dnnl_sgemm (between the submitter's submit and the
   host_task's completion).
2. **If the deferred-free path is the culprit**: the same class as
   llama.cpp-9ug6y (BCS CAT). The fix pattern is known — defer frees
   until all queues (including the CPU-side oneDNN workers of
   host_task) have signaled completion. For a SYCL host_task, this
   means: do NOT process deferred frees on the main thread while any
   host_task is still in flight on any queue.
3. **If the host-zone-reset path is the culprit**: reset should be
   gated on all outstanding host_task / dnnl_sgemm completions, not
   just GPU queue drains.

None of these are the "pad the weight tensor" fix; both require
tracking host-task lifetime through the cache's free path.

### Confidence level

- **High**: dnnl_sgemm is synchronous; workers join before return.
- **High**: the fault address is backed by a DRM GEM (host-USM,
  not plain heap or mmap).
- **High**: the fault is a use-after-free of a host-USM page.
- **Medium**: the free is from unified_cache's deferred_free or
  zone-reset path. Other candidates (SYCL runtime migration) are
  plausible but less likely given the PP→TG phase boundary timing.
- **Low**: which specific free site — no direct evidence yet.
  Next research step is instrumenting process_deferred_frees.

### Experiments deferred (not needed)

Experiments 1 and 2 (bank race, thread_local uninit) were ruled out
by the dnnl_sgemm-sync finding + the observation that fault pointers
are DRM-backed, not heap.

### A0c evidence files (ephemeral — tmpfs)

- `/tmp/a0c-gdb.txt` — gdb script with python maps extractor
- `/tmp/a0c-maps.log` — `info proc mappings` + fault register dump

## A0d follow-up — 2026-04-18 afternoon — culprit site identified

Added instrumentation to log every `[A0D-READ-SGEMM]` (before
`dnnl_sgemm` at cpu-dispatch.cpp:4191), every `[A0D-FREE-DEFERRED]`
(inside `process_deferred_frees` device + host branches), and every
`[A0D-FREE-RESET]` (inside `host_zone_reset` and `free_pinned_runtime`).
Instrumentation in an instrumented build only — backed out before
commit, verified via `git diff`.

### Result: culprit is host_zone_reset at ggml-sycl.cpp:40709-40710

333 `[A0D-*]` events captured in the repro log. Pattern is strikingly
regular across the TG phase:

```
[A0D-READ-SGEMM] t=45331.269351385 weight=0x7ff7ca208a80  (last sgemm before fault)
[A0D-FREE-RESET] t=45331.???      host_zone_reset(zone=2)  ← STAGING
[A0D-FREE-RESET] t=45331.???      host_zone_reset(zone=3)  ← SCRATCH
[A0D-READ-SGEMM] t=...            (next token's sgemm)
  ... 65 total TG M=1 calls ...
SIGSEGV (fault rdx=0x7ff7cb9fbf00)
```

Zero `[A0D-FREE-DEFERRED]` events and zero `[A0D-FREE-RESET] ...
free_pinned_runtime` events. All `[A0D-FREE-RESET]` events are
`host_zone_reset(zone=STAGING)` and `host_zone_reset(zone=SCRATCH)`
paired. These are zone=2 (STAGING) and zone=3 (SCRATCH) per the
`host_zone_id` enum in `pinned-pool.hpp:24-29`.

Call site: **`ggml-sycl.cpp:40709-40710`** inside
`ggml_backend_sycl_graph_compute` (or equivalent — the
graph-compute entry point):

```cpp
ggml_sycl::unified_cache_reset_scratch_pool(sycl_ctx->device);
ggml_sycl::unified_cache_host_zone_reset(ggml_sycl::host_zone_id::STAGING);
ggml_sycl::unified_cache_host_zone_reset(ggml_sycl::host_zone_id::SCRATCH);
```

This executes at the start of EVERY graph_compute_impl. For TG,
llama-bench issues one token per graph, so these resets fire
per-token.

### How the race lands

1. Graph N (token N) enters `graph_compute_impl`. Host zones
   STAGING + SCRATCH are reset (bump allocator offsets go to 0;
   the TLSF / registry is purged).
2. Graph N's ops run. The MoE router MUL_MAT dispatches to
   `cpu_mul_mat` at cpu-dispatch.cpp:3770. For this op,
   `async_mode=true` (async_requested && gpu_q &&
   !force_sync_for_moe_routing — the router is NOT in the MoE
   routing chain, so the guard doesn't fire). Main thread stages
   src1 activation into STAGING (via `staging_ensure` ->
   `host_zone_alloc(STAGING, size)`), submits a host_task wrapping
   `dnnl_sgemm(weight, src1, dst)`, and RETURNS.
3. Main thread proceeds past cpu_mul_mat through the rest of
   graph N, eventually exiting `graph_compute_impl`.
4. llama-bench issues the next token → graph N+1 →
   `graph_compute_impl` entry → lines 40709-40710 reset STAGING
   + SCRATCH.
5. **Meanwhile**, the host_task from graph N is still running
   `dnnl_sgemm` on a SYCL-runtime worker thread, using the src1
   pointer that was handed out from STAGING in step 2. When N+1
   resets STAGING, the TLSF bump pointer and registry state are
   purged — subsequent N+1 allocations from STAGING will hand out
   overlapping bytes. Concurrent writes into the overlap region
   (or page-level TTM actions by the xe driver during re-use of
   this virtual range) produce the unmapped-page SIGSEGV observed.

The fault address `0x7ff7cb9fbf00` is inside the 2 GB DRM VMA
that backs this pinned-chunk pool's STAGING zone. Consistent.

### Why the "zone_reset doesn't free DRM pages" argument from A0c
was incomplete

A0c concluded "pinned chunks stay mapped for process lifetime, so
resetting the bump allocator shouldn't cause SEGV — fault must be
in a different mechanism." That was half right: the chunks stay
mapped, BUT resetting the bump allocator while a CONCURRENT host_task
holds a pointer into a now-recycled region creates a TTM-level
contention. The xe driver / L0 runtime may unmap or remap individual
pages within the chunk under that contention. The user-space `rw-s`
VMA remains; the specific 4KB page hosting the fault address is
not present at the moment of the `vmovups`. That's the `error 4`
the kernel reported.

The specific sequence that causes page unmap within a live chunk
remains unknown without kernel-side instrumentation — but it
doesn't have to be understood at the kernel level for the fix.
The fix is at the user-space contract: don't reset the zone while
a host_task still references it.

### Fix direction (evidence-locked now)

At `ggml-sycl.cpp:40709-40710`, before calling
`unified_cache_host_zone_reset(STAGING)` and
`unified_cache_host_zone_reset(SCRATCH)`:

**Wait for all in-flight host_tasks to complete.** Specifically:
- There is a global chain event `g_cpu_chain_event` (cpu-dispatch.cpp:253)
  that always points at the most recent async CPU host_task. Calling
  `g_cpu_chain_event.wait()` before the zone resets would drain
  the outstanding host_tasks.
- Alternately, track per-zone host_task events in the unified cache
  (similar to how `g_staging_compute_evt[]` per bank already tracks
  CPU events) and wait on them before the zone reset.

Either approach is 5-20 LOC in the right place. Risk: one-wait-point
before two zone resets (per token) adds a sync edge; if the last
async sgemm hasn't finished, we wait. For TG this is actually fine
because the very next graph's first op will consume the previous
token's MoE output anyway — the wait is on the critical path in
practice, not adding latency.

Recommended: add `ggml_sycl_cpu_offload_drain_chain()` helper (or
inline `if (g_cpu_chain_event_valid) { g_cpu_chain_event.wait();
g_cpu_chain_event_valid = false; }`) immediately before line 40709.
Measure PP/TG impact; should be <1% because the host_task usually
finishes before the next graph starts.

### Confidence level

- **Very high**: A0D instrumentation confirmed host_zone_reset is
  the only active free-class event during the TG phase.
- **Very high**: no deferred_free or free_pinned_runtime events fire
  during TG, so those paths are not the culprit.
- **High**: race window is between host_task submission in
  cpu_mul_mat's async_mode path and the next graph's host_zone_reset.
- **Medium**: the specific mechanism by which STAGING-zone reset
  unmaps a DRM page (as opposed to just recycling bytes) — I
  hypothesize TTM page-level contention, but don't have kernel-side
  proof. The user-space contract fix works regardless of that
  mechanism.

### A0d evidence files (ephemeral — tmpfs)

- `/tmp/a0d/gdb.txt` — gdb script with info proc mappings at fault
- `/tmp/a0d/run.log` — 333 [A0D-*] events + fault register dump +
  maps dump

### Bead updates

- `llama.cpp-bhp87` (A0d): completed with culprit site + fix sketch.
- `llama.cpp-mqxer`: updated with exact line number + fix pattern.

## A0d retrofit — 2026-04-18 late afternoon

Re-ran A0d with two additions per team-lead's polish:
- 4th instrumentation site: `[A0D-FREE-PREFETCH]` at
  `expert-prefetch.cpp:461-462` and `:809-811` covering both
  `unified_free(scores_alloc_)` paths.
- `info proc mappings` captured inside gdb in the **same pid** as
  the `[A0D-*]` events, so the VMA match is deterministic rather
  than ASLR-speculative (Minor 4).

### Result: culprit site + line numbers refined

Retrofit instrumented build exit, 65 sgemm + 268 zone-reset + 0
deferred + 0 prefetch events. Captured maps were in the same pid
as the events. Same-pid fault lookup:

```
[A0D-FAULT-LOOKUP] rdx=0x7ff7cbd63f00 r13=0x7ff7cbd58b00
[A0D-FAULT-LOOKUP] RDX in VMA: 0x00007ff7cbd4f000 0x00007ff7cbe00000
                   0xb1000  0x0  ---s  /dev/zero (deleted)
[A0D-FAULT-LOOKUP] R13 in VMA: (same VMA)
```

**New observation**: the fault VMA in this run is `/dev/zero
(deleted)` with permissions `---s` (no read, no write, no exec,
shared). `error 4` read → immediate SIGSEGV because no read
permission. Different VMA CLASS from A0c's `/dev/dri/renderD129`
finding, but both are **SYCL host-USM shared mappings whose
backing was torn down while still VMA-listed**. ASLR + the
moment the mmap teardown caught relative to the fault determine
which class we observe; the corrective action is identical.

### A0d fix — line numbers on HEAD

A0d initially cited `ggml-sycl.cpp:40709-40710` for the STAGING +
SCRATCH reset pair. On current HEAD the exact lines are
**`40708-40709`** (line numbers drift with small edits above that
point; team-lead's quality re-review caught this). The anchor-
robust reference:

> `ggml_backend_sycl_graph_compute` entry, the two
> `unified_cache_host_zone_reset(...)` calls immediately after
> `ggml_sycl_moe_layer_ids_cache_new_graph()` and
> `unified_cache_reset_scratch_pool()`.

### A0d fix — batched-mode coverage caveat

Proposed fix drains `g_cpu_chain_event` before the zone resets.
That event is set by `cpu_submit_async()` (cpu-dispatch.cpp:253)
— the **async-mode** CPU dispatch path. The **batched-mode** path
(cpu-dispatch.cpp:4197-4204) runs dnnl_sgemm directly inside the
caller's batched outer host_task on `gpu_q` and does NOT update
`g_cpu_chain_event`. A reader might ask why the fix doesn't cover
batched.

Answer (quality reviewer's clarification, recorded here for the
implementer): batched-mode host_tasks are implicitly covered by
`gpu_q` in-order semantics — the next graph's first GPU op on
`gpu_q` serialises after the batched outer host_task, and the
graph_compute_impl entry itself is reached only after the prior
graph's final SYCL work completes. The **legacy sync path** at
cpu-dispatch.cpp:4219-4228 already calls `cpu_wait_chain_event`
before run_mul_mat, so it is also safe.

Only the async-mode code path needs the explicit drain. The fix
site (graph_compute_impl entry) is the natural choke point —
draining once there is cheaper than per-op event tracking, and
correctness-equivalent for the async path that creates the race.

Recommended comment to include at the fix site:

```cpp
// Drain any in-flight async-mode CPU dispatch before resetting the
// host bump zones.  cpu_submit_async writes g_cpu_chain_event per
// call (cpu-dispatch.cpp:253); its host_task holds pointers into
// STAGING/SCRATCH that would be invalidated by the reset.  The
// batched- and legacy-sync paths are covered by gpu_q in-order
// semantics / cpu_wait_chain_event respectively.
if (ggml_sycl_cpu_chain_event_valid()) {
    ggml_sycl_cpu_chain_event_wait();
}
```

### Calibration note

Per team-lead's forward-looking feedback: A0 → A0b → A0c each
proposed a fix shape before verifying the underlying theory; A0c
had to correct A0b; A0d had to refine A0c's "DRM-backed UAF"
to "host_zone_reset racing with async host_task specifically."
Applying the calibration discipline here: A0d's "host_zone_reset
at graph start is the culprit" is **CONFIRMED** by the 333-event
instrumentation correlation with SIGSEGV. The specific VMA class
of the fault page (/dev/dri vs /dev/zero) is **observation-only**
— not a claim that the fix depends on.

### A0d-v2 evidence files (ephemeral — tmpfs)

- `/tmp/a0d-v2/gdb.txt` — gdb script with in-process maps lookup
- `/tmp/a0d-v2/run.log` — 333 + `[A0D-FAULT-LOOKUP]` entries

