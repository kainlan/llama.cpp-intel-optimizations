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

## → llama.cpp-f8ws (lane handoff 2 + layout experiment, 2026-08-15 ~04:5x)

impl-f8ws2 stopped at context limit after: recipe_reason diagnostics + source_reason
sentinel fix (e2ff6b577); lease-fixture leak disposition + fix (f53c5c2c6 — fixture
held a raw lease result production always adopts; 582fee665's assert was RIGHT;
drain-asymmetry separate); staging design to implementation-readiness (710d35ebb,
brief in artifacts/f8ws2/pending-tracker-writes.md: register_device_expert with
REQUIRED allocation_owner; get_pointer_type validation; storage_owner = buffer's
managed_handle; free predicate :9863/:10358 skips storage_owner entries; hook =
pre-pass at MMID dispatch, resolver untouched).

LAYOUT EXPERIMENT (lead; GGML_SYCL_MOE_ROUTE_LOG=1 MMID subset,
scratchpad/f8ws-layout-exp.log): 437 [MOE-DIRECT] resolve-direct-miss lines with
buffer-owner TAGGED ids (0x8000000000000002 family) reaching direct resolution,
requested layout=0 (AOS) across the refusing f16 population, entries=0/ptr=nil.
VERDICT: AOS everywhere → non-owning primitive suffices; copying path NOT needed.

LEASE TEST 3 (f53c5c2c6): 9 cases ran (was 1); ruling-(b) case 1 PASSED; case 2
passed semantics then aborted unified-cache.cpp:9457 "deferred managed release
failed" — classifier/deferred-release finding open + 2 ownerless-lease warnings
to attribute. Successor: :9457 finding → register_device_expert → recipe split
ticket → census 6.

## → llama.cpp-f8ws (census 6 + lease verification, 2026-08-15 ~05:45)

LEASE GATE 9/9 PASS (verbose rerun proves suite = nine cases; the "11" bar was
arithmetic on an unverified baseline). Both ruling-(b) cases green; the formerly
:9457-aborting case passes: RULING (b) VERIFIED ON HARDWARE, 76041c5a5 confirmed.

CENSUS 6 at 1c548e15b (rc=134, truncated ~59.6k lines): route_unavailable 661→1
(prompt publication works; published_zero=1 to reconcile). Refusals moved
downstream: recipe_missing 616 (recipe_reason local 468 / direct 148 / recipe 1)
→ llama.cpp-gx30 population ~616, histogram measured. Wall probes all 0;
:57233/:57288 never fired. not_supported=11846 (truncated, = census 5's value).

WALL 5: mem-ops.cpp:393 "failed to allocate 32 byte host-pinned staging buffer"
with 192G free — pool-exhaustion/leak-shaped ("genuine defect" per the sweep's
prediction). RCA assigned to impl-f8ws3: attribution first (publication-path
accumulation vs pre-existing newly-reached), truncation-depth comparison c5 vs c6.
