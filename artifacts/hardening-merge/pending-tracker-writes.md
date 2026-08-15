# Pending tracker writes (codescout daemon unreachable, 2026-08-15 ~02:20)

Queued for posting when the tracker returns. Each block names its target ticket.

## → llama.cpp-f8ws (assert-family sweep record + watch ruling)

SWEEP RESULT: the wall-3/4 species does NOT extend. 422 GGML_ASSERTs
(ggml-sycl.cpp 275, mmvq.cpp 89, unified-cache.cpp 55, mem-ops.cpp 3); ~40
validity-shaped. Class (i) CONVERT: 0 new (the signature — assert sitting on an
existing refusal path in a bool-returning function — had exactly one instance,
:51668, fixed in fb9fc070a; a collapse commit would be empty). Class (ii) KEEP
~30: ggml-sycl.cpp:18954,18957,19310,19313,21916,33189,33279,34098,34334,
37834-38290,59806,59824; unified-cache.cpp:3528,3546,3582,3595 — all
mem_handle/alloc_handle ownership invariants; these asserts ARE the memory
contract's enforcement.

FIVE CLASS-(iii) SITES ON WATCH (lead ruling: ALL STAY; no pre-emptive
conversion without reachability evidence):
- A1 ggml-sycl.cpp:57233 + A2 :57288 — GGML_ASSERT(src0_storage && "host-resident
  weight must have valid host pointer"), CPU host-matmul/PP-batch. Trigger: any
  wall here; FIRST SUSPECT after the provenance commit (provenance moves which
  tensors reach host-resident routing).
- B1 mmvq.cpp:21807 + B2 :21861 + B3 :21878 — GGML_ASSERT(dst_handle.valid() &&
  src_handle.valid()) on slice/staging/segmented copies; allocator-owned
  from_chunk_ptr pointers; functions return sycl::event (no refusal path).
  Trigger: a wall here, or a design giving these a refusal channel.

STANDING PRE-REGISTRATION: if census 5 (post-provenance) aborts, check
:57233/:57288 FIRST. A wall anywhere else is more likely a genuine invariant
violation — the cheap-conversion population is exhausted.

COVERAGE LIMIT: 422 filtered to ~40 by shape, ~8 read in depth; not a
certification of the whole assert surface.

WALL/COMMIT CHAIN on task/hm-f8ws: W1 tallocr 64-byte under-reservation →
9c995f34a (real cause bdcbf8e57); W2 guard refusal semantics (2764 misaligned →
~2782 fails) → bdcbf8e57; W3 mem-ops.cpp:57 extent-0 false positive →
c4a21dd47; W4 :51668 dead-lettered refusal → fb9fc070a.
