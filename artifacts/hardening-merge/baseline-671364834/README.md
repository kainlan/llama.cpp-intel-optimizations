# HM Task 1 baseline census — 671364834 (2026-08-14)

Canonical build + first honest full `test-backend-ops` run at HEAD with SYCL ON.
(The first attempt at 3d1598b3e was VOID: stale CMakeCache carried `GGML_SYCL=OFF`
from the previous session; caught by the backend-check gate, recovered by deleting
`CMakeCache.txt` only.)

## Build census (`build-k0.log`, ninja -k 0)

Library + all main binaries: GREEN. 9 fork-local test targets fail — all one class:
seam-consumer tests orphaned by the seam-isolation wave, never compiled since.
Tracked as llama.cpp-2lfg (HM Task 16); full list + symbols in that ticket.
(A 10th, test-expert-prefetch-runtime.cpp, was fixed directly at 671364834.)

## GPU census (`backend-ops.log`, B70 pinned, rc=134)

- 2,426 OK. All previously-known aborts (norm.cpp:1298 family, binbcast, mmvq:21548,
  Q1_0/NVFP4 to_fp16 assert reached at 39c76be7e) are GONE or not reached —
  the oin0/9wjb/6d67 fixes hold. Enumeration reached the MUL_MAT_ID region.
- NEW terminal defects, tracked as llama.cpp-f8ws (HM Task 17):
  1. Three numerical FAILs, all f16 MoE-hybrid MUL_MAT_ID (ERR 6.3–22.9 vs 5e-4):
     (n_mats=16,n_used=16,m=32,n=1024,k=16), (n_mats=2,n_used=2,m=32,n=8192,k=64),
     (n_mats=16,n_used=16,m=50,n=200,k=64).
  2. Abort ggml-alloc.c:83 during the third case: `ggml_tallocr_alloc: not enough
     space in the buffer to allocate sent_4 (needed 4096, available 4032)` —
     64-byte under-reservation (measure vs alloc alignment mismatch).
- Also present, diagnostic not fatal: repeated `[SYCL-ZERO-ALLOC-CHECK] +41.6 MB
  during pp phase` warnings in the MMID region.

Shmem flat (14.5→14.6 GB band), MemAvailable ~197-203 GB. Kernel health not yet
checked post-abort — checked at the next GPU session start.
