# llama.cpp-skgik — Investigation findings

## Status
Architectural fixes committed (ceb8cbf95) but GPT-OSS 20B TG SEGV **not resolved**.
The visible crash is a **different bug** than the one described in the bead.

## What was fixed (correctly)

1. **CpuExpertPool::submit_batch raw pointer capture (UAF 1)**
   - Signature changed from `(const cpu_expert_task *, int n)` to
     `(std::vector<cpu_expert_task>)` — move ownership to worker lambda.
   - 4 call sites migrated: P4 first-arrival, P4 DOWN fusion,
     dispatch_cpu_compute, legacy batched MoE (37767 sync case).
   - `pending_cpu_scatter::tasks` / `pending_cpu_pipeline::tasks` fields
     no longer load-bearing.

2. **STAGING/SCRATCH reset vs CpuExpertPool ordering (UAF 2)**
   - Added targeted drain of `g_pending_scatter.future` and
     `g_pending_cpu_pipeline.future` before `host_zone_reset(STAGING/SCRATCH)`
     at graph-compute entry (ggml-sycl.cpp ~40826).

## What's still broken

- GPT-OSS 20B TG128 still SEGVs at ~10 tokens.
- Crash signature is the **same as lj6p0**: pointer in unmapped gap
  between adjacent `/dev/dri/renderD129` pinned-chunk VMAs.
- Main thread is in `NEO::DirectSubmissionHw::dispatchWorkloadSection`
  / `EncodeDispatchKernel<Xe2HpgCoreFamily>::encode<COMPUTE_WALKER>`.
- Crash PC in anonymous exec region (DNNL JIT GEMM: vbroadcastss/
  vfmadd231ps).
- No `[CPU-EXPERT-POOL]` / `[CPU-TG]` logs in the failing run —
  CpuExpertPool isn't even on the crash path.

## Suspected remaining bug

An allocation site that produces a pointer straddling a pinned-chunk
VA boundary, handed to DNNL JIT (GPU oneDNN primitive or its host-
pinned scratch). lj6p0 fixed the `unified_alloc` contract but some
other path (direct `sycl::malloc_host`, or zone fallback, or oneDNN
GPU-side host-scratch) still violates the single-segment invariant.

## Follow-up actions

Filed in docs/plans/2026-04-19-skgik-rootcause.md:

1. Audit remaining `sycl::malloc_host` call sites for 700 MB-2 GB
   TG-time allocations.
2. Verify GPU oneDNN scratch paths (DnnlGemmWrapper, onednn-fallback.hpp).
3. Add assert on pinned_chunk_pool::zone_alloc_segmented (catch the
   fragmenting caller).
4. Extend lj6p0's contiguous invariant to ALL pointer-to-kernel paths,
   not just unified_alloc.

## Gate results

- Gate 1 (Mistral canonical): PASS
- Gate 2 (Mistral perf, PP512=1701.42 / TG128=81.02): PASS no regression
- Gate 3 (GPT-OSS 20B PP512): PASS
- Gate 4 (GPT-OSS 20B TG completion -n 128): **FAIL** — same 10-token
  threshold, same lj6p0-class pointer-in-unmapped-gap signature
- Gate 5 (GPT-OSS 120B): not run (same class as Gate 4 likely fails)

## Key files touched

- ggml/src/ggml-sycl/cpu-expert-pool.{hpp,cpp} — submit_batch signature
- ggml/src/ggml-sycl/ggml-sycl.cpp — 4 call sites + STAGING drain
- docs/plans/2026-04-19-skgik-rootcause.md — full analysis

## Key learnings

- The bead description's `simd_mxfp4_q8_0_16row` / `row_ptrs[16]` claim
  did not match the actual crash. The previous agent may have been
  looking at a different run or hypothesizing a stack. Always verify
  the actual crash site first before designing a fix.
- lj6p0-class "pointer in unmapped DRM gap" is a FAMILY of bugs, not
  a single bug. Each new allocation path that crosses a pinned-chunk
  VA boundary surfaces the same symptom. Need a systematic audit.
