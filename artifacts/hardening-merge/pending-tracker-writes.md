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

## Outage 5 (2026-08-15 ~10:26 UTC) — NOT a crash: foreign daemon holds port 47878

Port 47878 is bound by a codescout daemon running as user `gha-e2e-host` from
/opt/claude-team-toolkit-e2e-runner/.pi/agent/codescout-data/bin/codescout (pid 2994562
at time of writing). It reads our tasks.jsonl fine but cannot create
.codescout/.tasks.jsonl.lock (no write permission in kainlan's repo), so ALL tracker
WRITES fail with "cannot create lock file" while reads keep working. kainlan's own
daemon presumably died earlier and the foreign one bound the freed port. Unkillable from
this session (different uid). ESCALATED TO USER. Writes below to replay when a
kainlan-owned daemon holds the port again.

### PENDING: task_comment_add llama.cpp-09ts (author lead)
CLOSED — P0 complete. Fix ae1e64ae5 (per-layer owners, spec PASS zero findings) +
follow-up e0ab0f431 (three quality findings, delta PASS). GPU evidence: RED = two
deterministic aborts at 35d2aab08; GREEN = gate rc=0 with correct 1..10 at ae1e64ae5
(2/3 runs, the 1 failure being 480a's unrelated wall-5) and again at e0ab0f431 (rc=0,
binary-stability control, zero zone_id warnings as staleness control, no wall-5). The
finding-2 saga is memorialized in the session memory bank
(a-findings-rationale-can-contradict-its-prescription). Residuals tracked elsewhere:
wall-5 (llama.cpp-480a), fatal-vs-silent asymmetry (llama.cpp-jzeo). Mistral inference
is restored on the merge candidate.

### PENDING: task_close llama.cpp-09ts

### PENDING: task_create (found by impl-f8ws4 during 38d61b53c, mirrored by lead)
title: direct-authority admission diagnostic is optimizer-proven dead code — is the branch itself unreachable?
type: bug, priority: 2
body: Found 2026-08-15 while modernizing g_moe_direct_authority_candidates /
ggml_sycl_moe_mark_direct_authority_pre_admission (llama.cpp-f8ws c-jy9l, commit
38d61b53c — which did NOT cause it: the tag printed zero times across censuses 6/7/8,
all pre-change). Evidence: neither the diagnostic string nor the symbol survives into
ggml-sycl.cpp.o or the .so (grep -a .o = 0, .so = 0, strings = 0, nm = 0) against a
same-form control on the neighbouring tag = 1, so the zeros are measurements. No
preprocessor guard near the call site (nearest directive is an #endif before it) — the
OPTIMIZER is proving the call site unreachable. Question: is the direct-authority
admission branch itself dead (a whole admission path never taken — significant), or only
the diagnostic (benign)? Investigation belongs in the direct-authority admission
machinery, not the f8ws lane. Practical: do not look for this counter in any census.

### OUTAGE 5 RESOLVED (10:40 UTC) — all pending writes above REPLAYED, do not replay again
Corrected window: ~10:26 to 10:35:42 UTC only (impl-f8ws4's c-jy9l succeeded at 10:37:46,
proving writes were back; the foreign gha-e2e-host daemon pid 2994562 exited on its own
and a kainlan-owned daemon pid 3166589 bound the port at 10:35:42). Replay ledger:
09ts closure comment = c-jcb1, task_close 09ts = done, dead-code ticket = llama.cpp-vep2.
Nothing else was mirrored during the window. Any comment timestamped OUTSIDE that window
landed normally — do not re-mirror it (duplicate comments are harder to spot than missing
ones, per impl-f8ws4's warning).
