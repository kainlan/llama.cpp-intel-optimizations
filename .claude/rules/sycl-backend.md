---
paths:
  - "ggml/src/ggml-sycl/**"
  - "tests/test-sycl-*"
  - "tests/test-unified-cache*"
  - "tests/test-mem-*"
---

# SYCL backend rules: memory ownership and architecture

This file loads when you open a file in the SYCL backend or one of its tests. `CLAUDE.md` carries
the short form of these rules. The two sections below are the full short form that `CLAUDE.md`
carried until 2026-11-16, moved here unchanged. The enforceable contract is
`docs/design/sycl-canonical-memory-architecture.md` and the narrative is
`docs/backend/sycl-memory-design.md`.

"Owner ruling" in this text means a decision the human made on the date given. In the moved text,
"this file" means `CLAUDE.md` as it stood before 2026-11-16.

## SYCL Memory Ownership

> **Full design & rationale:** `docs/backend/sycl-memory-design.md` (narrative —
> the three primitives, the single `unified_allocate` entry point, the
> allocation flow) and `docs/design/sycl-canonical-memory-architecture.md` (the
> in-force enforceable contract with allocator/pointer allowlists). This is the
> key design constraint of the fork — read those before touching allocation,
> dispatch, or eviction code. The rules below are the short form.

The unified cache is the memory allocator for the SYCL backend. Its authority is
not optional: `9a0670712` removed optional cache enablement and the enable/disable
branches, including the now-unread `GGML_SYCL_UNIFIED_CACHE` opt-out. The distinct
`GGML_SYCL_UNIFIED_CACHE_MODE` variable selects topology only; see the canonical
contract §1.2 and §9.3 for details. All SYCL backend GPU, host-pinned, staging,
scratch, graph-temporary, KV, oneDNN, and
weight-layout allocations must flow through the unified-cache allocation APIs
(`unified_alloc`, `unified_allocate`, cache materialization helpers, or wrappers
that return `mem_handle`). Do not introduce direct `sycl::malloc_device`,
`sycl::malloc_host`, `sycl::free`, raw TLSF allocation, or side caches outside
the unified-cache implementation. Any low-level allocation implementation detail
must remain inside unified-cache code and surface to the rest of the backend as
one of the **two sanctioned ownership surfaces**: a `mem_handle`, or an
`alloc_owner` returned by `unified_allocate_owner()` (owner-first allocation —
the intrusive control is minted before the physical allocation, so an allocation
cannot exist unowned; wrap it with `mem_handle::from_owned_alloc()` to hand it to
a consumer). The legacy `alloc_handle` from `unified_alloc()` remains allowlisted
as a compatibility form, but it is a migration state, not a third surface — a raw
pointer is never one at all.

Allocation **provenance** is an authority, not a hint: an owner's
`allocation_control_class` (`CACHE_BACKING` / `CACHE_SUBALLOCATION` /
`EXTERNAL_EXACT`) is fixed before coordinator admission and decides, among other
things, what survives destructive teardown. `CACHE_BACKING` must never become
requestable through a public field — it is mintable only via the private
`cache_backing_token` (pinned pool, compiler-enforced) or the TU-static bootstrap
adopt, whose two allowlisted sites are the cache's own staging buffer
(`unified_cache:staging`) and the oneDNN Graph-scratch pool's completion-flag slab
(`unified_cache:onednn_graph_scratch_flag_slab`); a third site fails the gate. See
the canonical contract §3.1.

`mem_handle` is the ownership and lifetime token. Code that uses an allocation
must hold a `mem_handle` (or an object that owns one) until the CPU thread,
SYCL event, command graph, or pointer table is finished with that allocation.
When the last handle/reference is released, the allocation is freed through the
unified cache (`unified_free`, `zone_free`, cache-entry lease release, etc.).
Do not add forced eviction, forced reap, or zone-reset logic to reclaim memory
that still has a live handle; a live allocation at cleanup means a leaked
reference or stale owner that must be fixed.

⚠️ **"At cleanup" is load-bearing, and was being read too broadly.** A live lease
is a defect only when its **owner is gone**. Several `llama_model` objects may
remain loaded at once, so **another live model's lease is correct, not leaked**,
and a *new model's load* is not a quiescent point for anybody else's weights.
This lifetime rule does not promise concurrent inference. The
`unified_cache_set_graph_compute_active(bool)` eviction guard and
`g_sycl_graph_compute_mutex` are process-global, but the mutex does not
universally serialize submission. Direct/fallback paths release it before
`compute_impl` submission. In contrast, persistent-TG/deferred-copy paths and
command-graph record/replay paths submit while the process-global mutex remains
held. Completion may still outlive the lock where a path permits deferred exit.
Thus host submission can overlap across calls on direct/fallback paths, and
device execution may overlap across calls and devices; pure-GPU decode may also
return with kernels still in flight. Do not infer supported concurrent inference
or cache safety from either overlap. Separately, same-device concurrent inference
remains unsupported because context-keyed KV/RUNTIME arena ownership is absent
(canonical contract §5).

Getting this backwards cost real time. `9a0670712` ("sycl: checkpoint unified
memory ownership work") replaced `reset_model_weight_entries`'s preserve-and-
continue with `GGML_ABORT`, plus the same in `host_zone_reset` and `zone_reset`.
That made `test-llama-archs` and `test-thread-safety` — both `main`-labelled —
fail deterministically, and each abort masked whatever came after it, so nobody
connected the failures back. `acdb192d4` restored preserve-and-continue for
leased weight entries; `4afdb6d9f` made whole-zone resets refuse the reset when
live registered allocations prevent safe partial preservation.

This does **not** license a general loosening. `mem_handle`'s destructor is still
the sole release point, and reclaiming memory that still has a live handle is
still forbidden. Current weight reclaim is ownership- and mode-aware and must use
`weight_entry_reclaimable()`: it preserves active leases, entries owned by a live
model even when `in_use_count == 0`, and unattributed entries when the reclaim
mode and live-model mask require preservation. Only entries that predicate
permits may be reclaimed.

Leak diagnosis remains separate from legitimate concurrency. Outside
`MID_LOAD_REPLAN`, an attributed entry that is still leased but has no live
model owner is reported as an ownerless leaked lease and may abort when
`GGML_SYCL_STRICT_LEASES=1`; ownerless classification, including that warning
and strict abort, is suppressed during `MID_LOAD_REPLAN`. Do not classify that
state as another model's valid ownership, and do not infer reclaimability from
`in_use_count` alone.

Raw pointers are not ownership tokens and must not model allocation state. They
are only transient ABI views resolved from `mem_handle` for immediate kernel
submission, oneDNN primitive calls, or tightly scoped CPU access. Do not store
raw pointers as the source of truth, use pointer addresses as cache keys, or
let pointers outlive their owning handle. Pointer tables and dispatch caches
must be derived from the stable identity/hash carried by `mem_handle`, not from
raw device addresses; if a table contains raw device pointers for a kernel ABI,
retain the corresponding handles for at least the lifetime of the queued work
or executable graph.

Weight cache entries are ref-counted through `mem_handle` leases. Eviction may
only remove entries whose in-use count is zero. If the cache cannot evict
because handles are still referenced, fix the missing release instead of
forcing eviction.

## Architecture

These are the architecture items of the old "Hard-Won Rules" chapter.

One item below has a newer amendment that the moved text does not carry. The item "PLACEMENT
DECIDES THE EXECUTOR" forbids weight streaming. The ruling of 2026-10-19 amended that for prompt
processing: a host-pinned expert may be staged to VRAM for a prompt when the executor cost model
says that is faster. The 2026-08-16 ruling still decides where weights live. Read "Placement
decides the executor" and "The executor cost model and the prefill ring" in
`docs/backend/sycl-memory-design.md` before you change either path.

- **The unified cache owns all GPU/host memory** (decision Feb 9, 2026). Weight placement, eviction (device→pinned host→mmap), and budget tracking all flow through it.
- **Use smart handles, never hold a raw `void*` from the cache.** A raw VRAM pointer becomes dangling the moment the cache evicts to host → DEVICE_LOST errors or corrupted results. Handles must resolve location on dereference so the cache can move data between tiers transparently.
- **A graph node without `GGML_TENSOR_FLAG_COMPUTE` is not executed:** a node walker that dispatches, fuses, records or plans work skips it through `ggml_sycl_is_noop`, or tests `ggml_sycl_graph_node_is_computed` (`prestage-class.hpp`, the one statement of the rule) when it must keep views, and `tests/test-sycl-compute-flag-source.py` holds the census.
- **PLACEMENT DECIDES THE EXECUTOR (owner ruling 2026-08-16; full statement in
  `docs/backend/sycl-memory-design.md` § "Placement decides the executor").** The
  planning pass puts data where it fits and where it runs best (dense weights fill
  VRAM before experts); inference then executes each op **where its data already
  is** — device-resident → that GPU runs it, host-pinned → the **CPU** runs it.
  Both escape hatches are forbidden: no GPU PCIe "zero-copy" reads of host memory
  (measured CPU AOS = 18–30 GB/s vs GPU zero-copy = 11.3 GB/s, 1.6–2.6x slower),
  and **no weight streaming** (copying host-resident weights to device scratch
  per dispatch). Host-resident experts go through the live CpuExpertPool path,
  parallelized with GPU work via `sycl::depends_on` (~9.7 µs cross-device
  latency). Format-conversion staging of *device-resident* data (on-device
  dequant into planned scratch zones) is layout work, not a placement violation.
  **Companion rule — LAYOUT FOLLOWS RESIDENCY** (owner ruling 2026-08-16, same
  doc section): weights are materialized in the optimal layout for the processor
  that executes them, per device (B50-optimal on the B50, B70-optimal on the
  B70, CPU-optimal for host-pinned); routes consume the materialized layout,
  never advertise past it, and only advertise (type, layout) pairs whose kernels
  exist. AOS fallback is a correctness stopgap, not the design.
  ⚠️ **Dispatch-side sharpening (owner ruling 2026-09-18): THE LOADED LAYOUT IS THE
  ANSWER, NOT A QUESTION TO ASK AGAIN.** The planner and the unified cache decide
  layout, per device, and what they materialized into VRAM/RAM **is correct by
  definition** — it is the fact, not a preference some later code may re-derive.
  So dispatch selects the **kernel** from the layout the operand is already in,
  exactly as it selects the **device** from where the operand already lives. Both
  come from the same authority: placement. Re-deriving layout at dispatch from a
  policy function is the defect, even when the policy is "right", because two
  sources for one fact will disagree eventually — which is precisely
  `llama.cpp-l9i1` (`route_layout=soa` vs `src0_layout=aos` on the same tensor;
  forcing them to agree turned NaN into a verified pass, measured 2026-09-18).
  The general shape is the `t98c`/`l9i1` family: **one fact, two sources.** If you
  find code asking a selector for something the operand already knows, delete the
  question, do not tune the answer.
  **Corollary — a missing kernel for the materialized layout is a SUPPORT GAP TO
  CLOSE, not a routing problem to work around.** When dispatch finds no kernel for
  the (type, layout) actually loaded, the fix is to write that kernel. It is *not*
  to re-route to another executor, silently convert the operand, refuse the op, or
  widen a policy so the planner picks a layout the kernel already has — all of
  which preserve the gap while hiding it. Advertising honestly (above) is how you
  avoid producing wrong numbers in the meantime; it is not the fix, and a
  capability refusal left standing is a ticket, not a resolution.
- **The VRAM budget calculation is correct by design for discrete cards**
  (`min(total*pct, free_at_init)`). Low free VRAM is a system problem. Fix the root cause at the
  system level, and do not "fix" it in the app by ignoring free VRAM. The full item, with the
  integrated-GPU case and its measurements, is in `docs/backend/sycl-host-safety.md`, section 8.
- **Small-block dequant (Q4_0/Q8_0/Q4_K) belongs on standard SYCL, not ESIMD.** ESIMD measured 1.9x SLOWER on Arc B580 + oneAPI 2025.3 (block granularity too small to amortize LSC loads). The real dequant lever is structural — fuse dequant into the matmul. Opt-in retest hatch: `GGML_SYCL_ESIMD_DEQUANT=1`.

## SYCL Environment Variables (the short table)

This is the table that `CLAUDE.md` carried until 2026-11-16. `CLAUDE.md` now lists the same
variables in a few lines.

The curated catalog (dispatch tuning, persistent-TG, memory hierarchy, cache,
and debugging) is **`docs/backend/sycl-env-vars.md`**. Its accessor-independent
literal-search command is the starting point for names not listed there. The
load-bearing performance opt-outs (all default ON — flip to disable) that you
most need to know:

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_SYCL_UNIFIED_SOA=0` | ON | Disable SOA memory layout (AOS fallback, ~4x slower TG) |
| `GGML_SYCL_TG_FAST=0` | ON | Disable MMVQ fast-path (slower TG) |
| `GGML_SYCL_DISABLE_GRAPH=1` | OFF | Disable SYCL graph replay (~3% TG, mainly helps PP) |
| `GGML_SYCL_ONEDNN_PP=0` | ON | Disable oneDNN for prompt processing |
| `GGML_SYCL_UNIFIED_FORCE_LEGACY=1` | OFF | Force legacy kernel dispatch (skip unified kernel) |
| `GGML_SYCL_QUEUE_PROFILING=<0\|1>` | OFF | Whether every queue is built with `enable_profiling`, which costs one device timestamp query per submitted command (off measured GPT-OSS 20B decode 12 to 26 % and Qwen prefill 34 % faster). Unset is off unless a profiling diagnostic is requested (`GGML_SYCL_KERNEL_PROFILE`, `GGML_SYCL_MXFP4_TG_PROFILE`, `GGML_SYCL_MXFP4_PP_PROFILE`, `GGML_SYCL_ATTN_SYNC_PROFILE`, `GGML_SYCL_MOE_EXEC_TRACE=2`): then on, with one WARN naming the variable. `1` is on. `0` is off and a requested diagnostic prints one WARN and no figure. Any other value: one WARN, treated as unset. Resolved once per process |
| `GGML_SYCL_LEASE_HOLDERS=1` | OFF | Diagnostic. A held arena (last release, load start, refused rebuild) names the code holding each chunk lease: one `chunk lease holder ... taken at libggml-sycl.so+0xOFF < ...` line per holder, at the level of the line that reported the arena. Each take then walks its call stack (slower run). Off costs one pointer store per take. Read once per process |

Common diagnostics: `GGML_SYCL_DEBUG=1` (verbose dispatch), `GGML_SYCL_NAN_CHECK=1`,
`GGML_SYCL_SAFE_MODE=1` (drain queue per op to localize faults), and
`GGML_SYCL_OP_TIMEOUT_MS=<N>` (abort before the xe GT-reset cascade).

## Logging from the backend

`GGML_LOG_INFO` is dropped at the default verbosity in every tool. If a diagnostic must be visible
in a normal run, emit it at WARN. Detail: `docs/backend/sycl-verification-traps.md`, section 8.

## Searching `ggml-sycl.cpp`

The codescout index omits real matches inside `ggml/src/ggml-sycl/ggml-sycl.cpp`. In that file,
verify with `cat ggml/src/ggml-sycl/ggml-sycl.cpp | grep -n '<pattern>'`.
