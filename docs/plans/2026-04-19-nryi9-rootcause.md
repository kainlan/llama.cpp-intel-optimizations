# llama.cpp-nryi9 — Root cause and architectural fix

**Status**: Investigation complete. Root cause identified. Fix designed and implemented.
**Branch**: `feature/sycl-coalescing`
**Author**: investigation 2026-04-19

## TL;DR

Mistral 7B Q4_0 at `GGML_SYCL_VRAM_BUDGET_PCT=30` (and smaller) intermittently
hangs/faults during inference with `level_zero backend failed with error: 20
(UR_RESULT_ERROR_DEVICE_LOST)` in `OP MUL_MAT`, followed by the watchdog
terminating the process 30 s later. The symptom the P3 agent described as
"KV divergence in `llama_context::decode`" is an effect, not the cause.

The real fault path is a **degenerate VRAM arena** produced by the cascade:

1. When this process starts after a prior SYCL process (or while another GPU
   consumer holds VRAM), `unified_cache` probes the pre-init free VRAM and
   caps the weight budget to whatever is actually free
   (`ggml/src/ggml-sycl/unified-cache.cpp:5154-5161`, "Capping budget from
   X MB to Y MB (pre-probe free VRAM)").
2. After the 512 MB generic-slack carve-out, the remaining budget can shrink
   to ~370 MB for a nominal 30%-of-11605 MB request.
3. `arena_reserve()` then subtracts a *fixed* 1024 MB tail for the SCRATCH
   (256) + ONEDNN (256) + RUNTIME (512) zones
   (`ggml/src/ggml-sycl/unified-cache.cpp:9204-9205`):
   `shared_bytes = alloc_size > tail_bytes ? alloc_size - tail_bytes : 0`.
   When the budget is smaller than the tail, **`shared_bytes` silently clamps
   to zero** — the KV+WEIGHT shared zone has zero capacity, but the arena is
   still logged as "Reserved single chunk: 370.1 MB … shared KV+weight=0.0 MB"
   and `arena_reserve()` returns **true**.
4. The unified cache continues as if the arena is healthy. Weight staging,
   KV allocations, and layer-streaming double-buffers then allocate outside
   the arena (via `sycl::malloc_device` per-entry), and eventually one of
   those allocations drives the device into an `UR_RESULT_ERROR_DEVICE_LOST`
   while a compute kernel is in flight.
5. The compute kernel's exception is caught and the catch site calls
   `std::exit(1)` (`ggml-sycl.cpp:39298`), but SYCL runtime teardown hangs
   holding the last event, so the process stalls until the watchdog fires.

The fix is to stop silently accepting a zero-width shared zone: the arena
reservation must return `false` when the requested tail zones exceed (or
consume all of) the available budget, letting the existing per-entry
fallback path handle the tight-VRAM case.

## Repro

Two shapes:

### Shape A — degenerate arena (fragmented VRAM, the bead's "crash")

Requires VRAM already partially consumed by another process or by a prior
run whose SYCL teardown hasn't released everything:

```bash
source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_VRAM_BUDGET_PCT=30 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -r 1
```

With ~882 MB free VRAM the run prints:

```
[UNIFIED-CACHE] Capping budget from 3481.6 MB to 882.1 MB (pre-probe free VRAM)
[VRAM-ARENA] Reserved single chunk: 370.1 MB (scratch=256.0, runtime=512.0,
              oneDNN=256.0, shared KV+weight=0.0 MB)
…
level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
Exception caught at file:/Apps/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp, line:39291
Error OP MUL_MAT
[SYCL-WATCHDOG] No GPU progress for 30006 ms (timeout 30000 ms, 1 devices known).
```

### Shape B — slow but correct (clean VRAM, same budget)

With all 11.6 GB of VRAM free at process start, the same command completes
with PP≈23 tok/s and TG≈5 tok/s. Output is correct. Layer streaming auto-
activates, F16 attention matmuls fall back to the CPU-graph helper because
the KV cache is host-resident, and the overall throughput is far below the
CLAUDE.md reference target (PP 269 / TG 14) because every attention op
round-trips through a whole-graph CPU fallback.

Shape B is *slow*, not crashing, and is bounded by a separate limitation
(F16 attention with host-resident KV has no in-place GPU path — see
`ggml-sycl.cpp:30968`). The target in CLAUDE.md was measured with
`GGML_SYCL_CPU_OFFLOAD=1` + OpenCL-CPU device, which currently cannot
initialise on this system because the OpenCL CPU build rejects
`-cl-intel-greater-than-4GB-buffer-required` (an independent bug outside
this bead).

## Root cause (detailed)

### Budget-cap → tail-subtract cascade

`unified_cache::unified_cache()` computes `budget_` along this path
(`unified-cache.cpp:5128-5175`):

```text
base_mem = device total bytes
pct      = GGML_SYCL_VRAM_BUDGET_PCT or 100
budget   = base_mem * pct / 100        // e.g. 30% of 11.6 GB = 3481 MB
clean_free = ggml_sycl_get_free_vram_at_init(device)
if clean_free > 0 and budget > clean_free:
    budget = clean_free                 // ← cap (logged)
budget -= 512 MB                        // generic slack
```

So a 30% nominal budget can become `min(3481 MB, clean_free) − 512 MB`. On a
system where another process holds 10.7 GB of VRAM, `clean_free ≈ 882 MB`
→ final budget = 370 MB.

That 370 MB is then passed to `arena_reserve()`. The zone layout code
computes:

```c++
const size_t tail_bytes   = onednn_bytes + runtime_bytes + scratch_bytes;
// defaults: 256 + 512 + 256 = 1024 MB
const size_t shared_bytes = alloc_size > tail_bytes ? alloc_size - tail_bytes : 0;
```

When `alloc_size == 370 MB` and `tail_bytes == 1024 MB`, `shared_bytes == 0`
by the ternary clamp. The three tail zones are then laid out starting at
offset 0 (since `shared_bytes == 0`), but each of them still *believes* it
owns its full requested capacity:

```
oz.size = onednn_bytes  = 256 MB   ← only 370 MB physical; this overlaps
rz.size = runtime_bytes = 512 MB   ← …with this, which in turn overlaps…
sz.size = scratch_bytes = 256 MB   ← …with the compute_arena scratch pointer
```

That's silent memory aliasing on top of the more obvious "weights can't live
anywhere" problem.

### Why it manifests as DEVICE_LOST on MUL_MAT

After arena reservation returns success (it shouldn't), the rest of the
system proceeds normally:

- Layer streaming allocates two device buffers (`~234 MB` total) via
  `unified_alloc`, which — given no weight zone — fall through to raw
  `sycl::malloc_device`.
- Placement plan decides most of the model belongs in VRAM (the plan is
  computed from `budget_pct * base_mem`, not from the post-cap budget, see
  `ggml-sycl.cpp:6328`; this is a secondary bug noted below).
- KV cache allocations and compute-scratch demands collide with the
  already-tight free VRAM.
- A subsequent `sycl::malloc_device` fails, and the L0 driver marks the
  context `DEVICE_LOST` because an in-flight compute kernel was targeting
  memory that is no longer valid.
- The MUL_MAT catch clause at line 39291 prints, calls `std::exit(1)`, but
  SYCL runtime cleanup holds the last event → the watchdog fires at +30 s.

### Why Shape B (clean VRAM) still misses the throughput target

Orthogonal to this bead but documented so the next reader isn't surprised:
Mistral at `VRAM_BUDGET_PCT=30` with clean VRAM runs correctly at
`PP≈23 / TG≈5`, not `PP≈269 / TG≈14`. The difference is that the CLAUDE.md
targets assumed the CPU-offload path (`GGML_SYCL_CPU_OFFLOAD=1` + OpenCL
CPU device for data-local compute), which cannot initialize on this box
today (`-cl-intel-greater-than-4GB-buffer-required` is rejected by the
OpenCL CPU compiler). Without CPU offload, F16 attention matmuls with
host-resident KV round-trip through `ggml_sycl_cpu_fallback_graph()` — a
whole-graph CPU dispatch per attention op (`ggml-sycl.cpp:30968-30977`).
That is a separate workstream; fixing it is outside nryi9's scope.

## Fix alternatives

**Alt 1 — Refuse degenerate arena (PICKED).**
In `arena_reserve()`, after computing `shared_bytes`, if the shared zone
would be empty (or implausibly small), free the probe allocation and
return `false`. Callers already handle arena reservation failure by
falling back to per-entry allocation (`unified-cache.cpp:1216`). This is
the minimal correctness fix — it removes the silent-aliasing zone layout
and lets the tight-VRAM case take the honest "no arena, per-entry alloc"
path which is known to be functional (or at least to fail more cleanly
when the system is truly out of VRAM).

*Tradeoff:* in pathological cases the per-entry path will also fail to
allocate, but that failure will be at an allocation call with a clear
sycl::exception message, not 30 s later in a watchdog cascade. If the
system genuinely doesn't have enough VRAM for the workload, the user
should see "out of memory" immediately and lower either the model size or
the budget.

**Alt 2 — Proportionally scale tail zones.**
When `budget_bytes < tail_bytes + min_shared`, scale each of SCRATCH,
ONEDNN, RUNTIME down proportionally so that `shared_bytes >= min_shared`
(e.g. at least 64 MB). This keeps the arena active but with shrunken
zones.

*Tradeoff:* complex invariants — shrunken SCRATCH can cause oneDNN matmul
scratch allocations to fail mid-graph, shrunken RUNTIME can make KV
allocation fail — and those failures would surface at inference time
rather than at arena-reserve time. We'd be trading one silent-failure
cascade for a different one. This is also a larger blast radius: the
256/256/512 defaults were chosen for a reason and shrinking them invites
regressions elsewhere.

**Alt 3 — Hard-cap `budget_pct` against `clean_free`.**
Move the "cap to pre-probe free VRAM" decision one level up so that
`set_tensor_inventory()` also sees the capped value and doesn't plan a
device placement that the arena can't fulfil. This partially addresses
the secondary bug noted above (placement plan uses un-capped budget) but
doesn't fix the zone layout aliasing itself.

*Tradeoff:* addresses a related bug but leaves the zero-shared-zone bomb
intact for any caller that passes a small-but-nonzero budget directly
(e.g. someone using `GGML_SYCL_MEM_BUDGET=300`).

### Pick: Alt 1

Alt 1 is the minimum architectural change that fixes the silent aliasing.
It doesn't hide the VRAM shortage (which is what the user actually needs
to know about), it takes the existing supported fallback path, and it
doesn't change any default zone sizes — so clean-VRAM runs are bit-for-bit
identical to today.

Alt 3 (clamping the un-capped budget in `set_tensor_inventory`) is worth
doing as a follow-up but is orthogonal. It deserves its own bead.

## Implementation

`ggml/src/ggml-sycl/unified-cache.cpp::arena_reserve()`:

- Single-chunk branch: after computing `shared_bytes`, if it is less than
  a small floor (16 MB, enough for handle metadata and a probe), free the
  arena chunk, clear `arena_chunks_`, and return `false`. Log a
  `GGML_LOG_WARN` with the budget/tail numbers so the user can see why.
- 2-chunk branch: analogous check on `kv_size` (the derived inner shared
  zone inside `chunk0`). The 2-chunk path is only taken after single-chunk
  fails, so both paths need the guard.

Callers in `unified_cache::unified_cache()` already log "[VRAM-ARENA]
Failed on device %d, falling back to per-entry allocation" when
`arena_reserve` returns false (line 1216), so no additional caller
plumbing is needed.

## Gate plan

1. Mistral canonical (default budget): completion produces "6, 7, 8, 9, 10".
2. Mistral perf default budget: PP ≥ 1700, TG ≥ 81.
3. Mistral `VRAM_BUDGET_PCT=30` with clean VRAM: **completes without
   crash**. Throughput remains limited by the CPU-fallback path (Shape B)
   — documented as out of scope.
4. 20B bench `-p 512 -n 128 -r 3`: PP ≥ 54, TG ≥ 15.
5. Mistral `VRAM_BUDGET_PCT=50` with clean VRAM: completes without crash.

The artificially-fragmented VRAM scenario (Shape A) is hard to reproduce
deterministically after the fix without explicitly bringing up another
VRAM-consuming process; we validate the guard via (a) the log line
"[VRAM-ARENA] Insufficient budget …" appearing in a synthetic low-budget
run (e.g. `GGML_SYCL_VRAM_BUDGET_PCT=5`), and (b) such a run aborting at
model-load allocation time with a clean sycl::exception rather than in a
watchdog cascade.
