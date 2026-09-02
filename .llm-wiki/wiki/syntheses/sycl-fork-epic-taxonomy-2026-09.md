---
type: synthesis
title: SYCL fork epic taxonomy (2026-09-01 tracker triage)
description: The 2026-09-01 codescout tracker triage closed 163 of 581 open tasks with evidence, decided 8 more via GPU gates, and organized the 396 survivors into 18 design-first epics. This page is the durable record of that taxonomy — what each epic is for, what closes it, which owner rulings bind it, and what remains unresolved.
created: 2026-09-02
updated: 2026-09-02
sources:
  - id: SRC-2026-09-02-005
    resource: /sources/SRC-2026-09-02-005.md
  - id: SRC-2026-09-02-004
    resource: /sources/SRC-2026-09-02-004.md
  - id: SRC-2026-09-02-006
    resource: /sources/SRC-2026-09-02-006.md
  - id: SRC-2026-09-02-007
    resource: /sources/SRC-2026-09-02-007.md
  - id: SRC-2026-09-02-008
    resource: /sources/SRC-2026-09-02-008.md
  - id: SRC-2026-09-02-009
    resource: /sources/SRC-2026-09-02-009.md
---

# SYCL fork epic taxonomy (2026-09-01 tracker triage)

On 2026-09-01 the codescout tracker for this fork (581 open tasks) was triaged
against the current state of the repo (`master`, HEAD `fed0b58e2`, upstream
`b10630` merged 2026-08-26): 163 tasks closed with cited evidence (done,
superseded, obsolete-hardware, duplicate, or unverifiable), 8 more decided by
live GPU gates (4 closed, 1 new bug — `llama.cpp-mpps`), and the 396 survivors
assigned to exactly one of 18 epics — a merge of two independent taxonomy
proposals (subsystem-seam vs owner-outcome). Full taxonomy: [SRC-2026-09-02-005](/sources/SRC-2026-09-02-005.md).
Judging brief and rulings: [SRC-2026-09-02-004](/sources/SRC-2026-09-02-004.md).

## The 18 epics

| Tracker id | Title | Goal (one line) | Closing gate | Rulings that bind it |
|---|---|---|---|---|
| `llama.cpp-mubmt` | One allocator: every SYCL byte enters through the unified cache | Close every remaining direct `sycl::malloc_*`/`free` site, pool-alloc backlog, and bare `malloc_host` cohort so `unified_alloc`/`unified_allocate`/`unified_allocate_owner` are the only doors in. | `check-sycl-alloc-usage.sh` + `test-sycl-owner-allocation-migration.py` green with the allowlist reduced to unified-cache itself; zero `sycl::malloc_*`/`free` outside it; `GGML_SYCL_ZERO_ALLOC_CHECK` reports zero steady-state allocations on Mistral + GPT-OSS 20B tg128; Mistral gate + B70 baseline hold. | [Sole allocator](/concepts/single-allocator-invariant.md); iGPU never budgeted as VRAM; no wait added to make an allocation path safe. |
| `llama.cpp-rg2ft` | Zero raw-pointer escapes — mem_handle is the only ownership token | Finish the `data_device[]` phase-out; rekey every pointer table and dispatch cache onto the handle's stable identity hash. | `common.hpp` no longer declares `data_device[]`/`set_data_device()`/`data_device_ptr()`; `from_direct`/`from_chunk_ptr` census is zero or fully classified ABI-transient with a regression test. | [mem_handle sole ownership token](/concepts/weight-handle-leases.md); pointer tables derive from handle identity, never raw addresses. |
| *(new)* `refcount-reclaim-no-zone-resets` | Reclaim by refcount — finish eliminating zone reset and drain | Finish the `llama.cpp-iiff` programme: remove remaining reset/drain reclamation, give every transient zone epoch-refcounted ownership, fix reclaim predicates that free too much. | No caller of `zone_reset`/`host_zone_reset`/`arena_reset` remains on an inference or model-load path (grep-backed gate); the surviving-owner predicate exists once; a two-model residency test passes. | Zone resets are being **eliminated**, not extended; no forced eviction while a handle is live — a live lease is a defect only when its owner is gone. |
| `llama.cpp-792vn.5` | One VRAM ledger — arena zones are the only budget authority | Arena zones are the single source of truth for reserved vs free: no shadow counters, no TOCTOU window between a budget check and the allocation it authorizes. | Plan-vs-actual audit at end-of-context shows zone reserved == sum of live allocations for every zone kind on both cards, zero RUNTIME-zone overflows, arena `used_` decrements on eviction. | VRAM budget = `min(total*pct, free_at_init)`; iGPU host-unified memory never counted as VRAM; the cache owns the ledger. |
| `llama.cpp-3h5gm` | The planner plans every byte — weights, KV, compute, scratch | One placement pass, before anything is allocated, decides where every byte lives; nothing is placed by a runtime heuristic or an `-ngl` guess afterward. | `GGML_SYCL_PLAN_DUMP=1` at context creation prints a complete plan whose domains sum to the measured allocation total within 1% across Mistral, GPT-OSS 20B and 120B at three VRAM budgets, with no allocation outside a planned domain. | [Placement decides the executor](/concepts/placement-decides-the-executor.md); the plan names the layout, materialized once; planned destinations are unified-cache allocations. |
| `llama.cpp-sk1xz` | Model-max context always initializes — tiered KV placement, never a refusal | Default `n_ctx` is the model max; KV that doesn't fit VRAM is placed into pinned host tiers, never refused or silently shrunk. | `llama-cli`/`llama-completion` on GPT-OSS 20B with **no** `-c` flag reach generation on B50 and B70 (today's default 131072 path refuses); Mistral at model-max context passes. | [Never shrink context — place KV instead](/concepts/never-shrink-context-place-kv-instead.md); KV allocations are unified-cache allocations; host-tier KV runs where that data is. |
| *(new)* `event-ordered-graph-replay` | Ordered by events, replayed as a graph — no host waits in dispatch | Express dispatch-path ordering as SYCL event dependencies; make graph record/replay the steady-state path for decode and prompt. | Zero `wait()`/`wait_and_throw()`/queue-drain calls remain in dispatch-path functions (source-contract gate with a positive control); a GPT-OSS and a Mistral decode run report `exec_graph>0` with replay on ≥95% of decode steps. | [No host waits — event-chain everything](/concepts/no-host-waits-event-chain-everything.md); graph-retained handles are leases — a per-context path may not release another context's. |
| `llama.cpp-30ak7` | GPT-OSS decode at 80% of the memory-bandwidth roofline | Single-stream GPT-OSS 20B (and 120B) token generation must be limited by DRAM bandwidth, not kernel class, launch overhead, or layout shifts. | Achieved bytes/token × tok/s ≥ 0.8 × 224 GB/s on B50 and ≥ 0.8 × 608 GB/s on B70 for tg128, by interleaved paired A/B; hard floors while chased: tg128 ≥ 32 (B50) / ≥ 44 (B70). | [One layout per weight](/concepts/layout-follows-residency.md); no host waits; small-block dequant stays on standard SYCL, fused into the matmul; correctness before throughput. |
| `llama.cpp-xihy` | Prefill at INT8 XMX peak — one GEMM-shaped route per role | Prompt processing must run as real GEMMs on XMX, not entry-batched GEMV; gate/up still amplifies expert-weight traffic ~64x at pp512. | pp512 sustained INT8 utilization ≥ 80% of hardware peak (≥136 TOPS B50, ≥294 TOPS B70), computed from GEMMs actually issued; gate/up expert-weight bytes moved fall from ~413 GB to ≤~10 GB per pass. | Routes advertise only (type, layout) pairs whose kernels exist; oneDNN SDPA stays default-on for Mistral GQA; small-block dequant is not an ESIMD lever. |
| *(new)* `one-layout-honest-routes` | One layout per weight, and routes that only advertise kernels that exist | Each weight materialized exactly once in the layout optimal for its executor; `supports_op` advertises only (type, layout) pairs a real kernel covers. | A layout census at HEAD shows zero weights in two layouts and zero dispatch-time layout shifts for GPT-OSS 20B on both cards; every advertised route resolves to a compiled kernel (matrix test, fails closed on unadvertised pairs). | [Layout follows residency](/concepts/layout-follows-residency.md) / one layout per weight / all consumers support it; layout identity travels on the handle, never on a side table. |
| `llama.cpp-ic4f` | Every op runs where its data lives — MoE residency and the CPU expert pool | Device-resident experts run on that GPU; host-pinned experts run on the CPU via CpuExpertPool in a CPU-optimal layout; nothing streams host weights to device scratch per dispatch. | On B50 GPT-OSS 20B at three VRAM budgets, both canonical gates stay green with zero occurrences of weight streaming or per-token VRAM prestage in a verbose run; forced whole-layer host demotion no longer emits garbage tokens. | [Placement decides the executor](/concepts/placement-decides-the-executor.md) — GPU zero-copy reads of host memory and per-dispatch weight streaming are FORBIDDEN; overlap via `depends_on`, never a wait. |
| `llama.cpp-po3nd.2` | The second card earns its keep — dual-device PP and decode over a host bounce | B70+B50 together must beat the best single card on GPT-OSS PP and decode, with the planner owning contiguous layer-block placement. | Same-build interleaved paired A/B across B70-only, B50-only and dual: dual pp512 > max(1415, 894) and tg128 > max(44, 32) by a margin bigger than run noise, both canonical gates green throughout. | No PCIe P2P between the two discrete cards — every cross-device design here is a host bounce; placement decides the executor. |
| `llama.cpp-m0yo` | 120B loads fast and answers coherently | GPT-OSS 120B loads in bounded time by materializing weights straight into their planned unified-cache destinations — no persistent host double-copy. | 120B load wall time measured and improved against a recorded pre-change number on B70 with the iGPU excluded, Shmem flat, no host double-copy visible in a verbose load; completion matches a recorded token-for-token baseline. | No weight streaming, no per-token expert prestage; materialize into planned unified-cache storage, once, in the right layout. |
| *(new)* `canonical-gates-stay-green` | The two canonical gates stay green, on every path | The Mistral completion gate and GPT-OSS chat gate are the fork's definition of correct; they must pass on a fresh boot, through every dispatch path, behind a stable numeric oracle. | On a fresh boot with the discrete cards pinned, both gates pass on B70 and B50 in the same build, including across a PP→TG transition in one process, with zero aborts. | Correctness before throughput — the two named gates are the gates; opt-in flags with known GPT-OSS failures stay opt-in; oneDNN SDPA-on is correct for Mistral GQA. |
| *(new)* `tests-that-cannot-pass-vacuously` | Tests that cannot pass vacuously | A green test must mean work happened — today several PASS while skipping, or literal-match a string that legitimately changed. | The full documented-safe sweep (`-j 1 -LE 'residency\|mem-handle\|cache' -E '^test-backend-ops$'`) is green at HEAD, plus the excluded family serially; every skip exits 77, none reports PASS on a skip path. | Every gate needs a positive control that can fire; grep the string the code actually prints (`GGML_ABORT` never reaches a log). GPU/model-loading tests are lead-run only, single invocation, selector pinned. |
| `llama.cpp-1yr6` | Upstream absorbed — every op SYCL claims is implemented and dispatched | Land the Phase-C port queue from the b10630 backlog and close the honesty gap between what `supports_op` advertises, what's dispatched, and what `docs/ops/SYCL.csv` claims. | Every `[ports]` ticket lands with its upstream sha cited or closes as not-applicable; `test-backend-ops` (lead-run, selector pinned) shows no SYCL op failures for the named ops. | Verify a variable/knob still exists before restoring it; the iGPU-classification port must not parse or refuse `ONEAPI_DEVICE_SELECTOR`; allocation-touching ports still owe rulings 1–2. |
| `llama.cpp-p8sq` | Every microsecond of a decode step is attributed | Full device-event + host-submission accounting for a decode step and a pp512 pass, with the unrecorded remainder named rather than bucketed. | A single ranked host+device cost table for GPT-OSS 20B tg128/pp512 on B50 and B70 whose rows sum to ≥95% of measured wall time, unattributed remainder named and under 5%. | Baselines come from `docs/backend/sycl-perf-baselines.md`, never a remembered figure or a task's own stale acceptance criteria; absolute numbers under ambient load are not baselines. |
| *(new)* `build-docs-and-dev-loop` | A fast build, honest docs, working tooling | Fix the relink triggers and CMake dependency mistakes that force a full relink on every commit; correct docs/inventories describing code that no longer exists. | A no-op ninja after a commit does no relink and doesn't re-run CMake, completing in seconds; a fresh incremental build after a one-line edit finishes within the documented ~10-min ccache-warm window. | Env-var docs must match a grep of the source — a name no longer read is not a feature to restore; docs must not prescribe parsing or overriding `ONEAPI_DEVICE_SELECTOR`. |

*Full goal/bar/constraints/ordering text for every epic (longer than fits a table cell) is in the captured taxonomy source: [SRC-2026-09-02-005](/sources/SRC-2026-09-02-005.md).*

## Design principles the taxonomy encodes

The 18 epics are not an arbitrary partition of 396 tickets — they follow the
seams the owner's design rulings already cut into the codebase (full ruling
list: [SRC-2026-09-02-004](/sources/SRC-2026-09-02-004.md)):

1. **Ownership/lifetime stack** (`unified-cache-sole-allocator`,
   `handles-only-no-pointer-escapes`, `refcount-reclaim-no-zone-resets`,
   `budget-authority-one-ledger`) — the [single-allocator invariant](/concepts/single-allocator-invariant.md)
   and [mem_handle](/concepts/weight-handle-leases.md) as the only lifetime
   token are enforced bottom-up: one allocator, one ownership token, one
   reclamation mechanism (refcount, not reset), one budget ledger. A task that
   adds a second allocator, a raw-pointer cache key, a reset step, or a shadow
   counter is against design at the seam, not just against a style
   preference.
2. **Placement layer** (`planner-plans-every-byte`,
   `never-shrink-context-tiered-kv`) — [placement decides the executor](/concepts/placement-decides-the-executor.md)
   requires that placement happen once, in a single pass, before execution;
   these two epics own the planner itself and its highest-visibility failure
   mode (a context that refuses to initialize because nothing places KV to
   host tiers).
3. **Execution layer** (`event-ordered-graph-replay`, `one-layout-honest-routes`,
   `experts-run-where-they-live`) — once data is placed, [layout follows
   residency](/concepts/layout-follows-residency.md) and execution is
   [event-ordered, never wait-ordered](/concepts/no-host-waits-event-chain-everything.md).
   These epics remove the remaining places where the dispatcher still
   re-decides layout or ordering at op time instead of consuming what the
   planner already decided.
4. **Roofline targets** (`gptoss-decode-bandwidth`, `prefill-xmx-int8-peak`,
   `second-card-earns-its-keep`) — performance work is scoped against the
   2026-09-01 [efficiency ruling](/concepts/efficiency-goal-80pct-hardware-theoretical-peak.md):
   80% of *hardware* theoretical peak (bandwidth for decode, INT8 XMX for
   prefill), never 80% of another model's measured tok/s, and never a stale
   B580-era or pre-driver-pin guardrail.
5. **Correctness/quality gates** (`canonical-gates-stay-green`,
   `tests-that-cannot-pass-vacuously`) — verification infrastructure gets its
   own epics because the triage found the gates themselves were part of the
   problem: green results that didn't mean what they claimed.
6. **Scale and breadth** (`big-model-load-and-120b`,
   `op-coverage-and-upstream-ports`, `decode-cost-attribution`,
   `build-docs-and-dev-loop`) — work that either extends the above principles
   to a harder case (120B) or supports applying them (attribution, upstream
   parity, developer loop).

The merge rationale itself is a design decision worth recording: the taxonomy
is a merge of two independently generated proposals (subsystem-seam vs
owner-outcome framing) precisely because a single framing under-served either
"what code changes together" or "what bar the owner is judging against" —
the merged version keeps outcome-framed bars (stated in the units of the
2026-09-01 efficiency ruling) while borrowing the subsystem proposal's finer
split of reclamation from event-ordering/graph-replay (20 members would have
been one epic covering three orderable workstreams) and its un-crossed
epic-ticket reuse for the allocator and pointer-escape epics.

## Tensions / Caveats

These are open methodological risks the triage flagged but did not resolve —
they recur across the taxonomy rather than belonging to one epic, and they
match the shape of tensions already on record in this vault under
[bench-guard preflight refusal](/concepts/bench-guard-preflight-refusal.md)
and [known-stale baseline](/concepts/known-stale-baseline.md):

- **Unpinned `ONEAPI_DEVICE_SELECTOR` remains the single most repeated hazard
  in every epic's own gate-hygiene text.** Nearly every epic addendum in the
  taxonomy repeats, verbatim, "pin `ONEAPI_DEVICE_SELECTOR` on EVERY run — an
  unpinned run enumerates the iGPU and can OOM the host." Repeating the
  warning 18 times is not the same as enforcing it; no epic in this taxonomy
  makes selector-pinning a structural gate rather than a documentation
  reminder.
- **Vacuous ctest selections.** The `tests-that-cannot-pass-vacuously` epic
  exists specifically because label filters fail open (`test-backend-ops`
  carries no labels and a naive `-LE` sweep does not exclude it) and skip
  paths have historically exited 0. Its own bar depends on a prerequisite
  fail-closed CTest policy (`llama.cpp-ezfm`) landing first — until it does,
  every other epic's "run the suite" gate inherits the same vacuous-pass risk.
- **Tests not registered.** The same epic's scope includes tests that never
  reached the suite at all (dead CMake registrations, orphaned binaries) —
  a distinct failure mode from a mis-scoped label, and one no grep-based
  audit catches without enumerating the CMake target list against the
  binaries actually built.
- **B50 PL2-throttle caveat on today's GPT-OSS TG number.** 2026-09-01 22:07,
  `ONEAPI_DEVICE_SELECTOR=level_zero:1 llama-bench -m gpt-oss-20b-mxfp4 -p 512
  -n 128 -fa 1` on `master` `fed0b58e2` read pp512 843.35 ± 16.17 and tg128
  17.95 ± 0.71, against the 2026-08-26 certified 852.40 / 31.92
  (`docs/backend/sycl-perf-baselines.md`). Minutes later
  `/sys/class/drm/card2/device/tile0/gt0/freq0/throttle/status=1` with
  `reason_pl2=1`, `cur_freq` 1200 of `rp0` 2600, `act_freq` 0, clearing to
  `status=0` within ~15 minutes; `codescout.real` (pid 423422) held
  `/dev/dri/renderD130` (the B50) at the time. No in-run throttle sample was
  taken, so the tg figure is a **suspect measurement, not a regression
  finding** — the deciding test is one paired run after 5 minutes of
  `throttle=0`, with a 1 s sampler of `throttle/status` and `act_freq` running
  through it. (Unpublished session data — cite the session, not a SRC, until
  it's re-run and recorded.) The general protocol this vault documents
  ([PL2 throttling](/concepts/pl2-throttling.md)) predates this reading and
  was derived from `llama.cpp-dboi`/`llama.cpp-d0bp`.
- **A concrete instance surfaced during today's own GPU-gate decisions.** The
  gate for `llama.cpp-0np6` (promote persistent-TG to default) aborted
  instead of producing a number — `GGML_SYCL_PERSISTENT_TG=1` crashes in
  `extract_persistent_plan` on a zero-extent `mem_handle` source
  (`llama.cpp-mpps`, filed 2026-09-01/02, blocks `0np6`). This is exactly the
  shape of defect `handles-only-no-pointer-escapes` and
  `event-ordered-graph-replay` exist to close structurally, not just patch at
  the one call site — the producer must construct the handle with its real
  extent rather than the range guard being weakened.
- **Driver-version uncertainty affects any epic doing cross-device or
  compute-runtime-adjacent work.** The loaded Level Zero driver moved to
  stock 26.31 on 2026-08-18 (one-way PPA upgrade); whether it retains the
  patched 26.22 build's cross-device in-order dependency fixes and USM
  compression fix is unverified (see [Intel compute-runtime](/entities/intel-compute-runtime.md)).
  This bears directly on `second-card-earns-its-keep` and
  `event-ordered-graph-replay`.

## Open Questions

- **`never-shrink-context-tiered-kv`**: once tiered KV placement lands (part 3
  of `llama.cpp-uize`), should the canonical GPT-OSS chat gate still pin
  `-c 4096`, or should it validate the model-max default path directly? This
  was deliberately left to an owner decision rather than decided by the
  triage.
- **`second-card-earns-its-keep`**: upstream's `--split-mode tensor` (PR
  #19378, SYCL implementation PR #24152) already covers GPT-OSS among its
  supported MoE architectures and reports large PP/TG gains on a dual-B70
  self-benchmark. The epic added a new evaluation task
  (`llama.cpp-k2y8`) rather than resolving whether upstream TP should replace,
  complement, or be abandoned in favor of the fork's own block-pipeline
  design — unresolved at triage time.
- **`gptoss-decode-bandwidth` / `handles-only-no-pointer-escapes`**: is
  `llama.cpp-mpps`'s zero-extent `mem_handle` construction in
  `extract_persistent_plan` a one-site producer bug, or does it indicate a
  broader pattern of scalar device-copies built without a byte-contract
  extent elsewhere in the persistent-TG plan-extraction code? Not yet
  audited.
- **`one-layout-honest-routes`**: a 2026-09-01 web-research correction
  reversed an earlier claim that this fork's NVFP4/Q1_0 type IDs
  (`GGML_TYPE_NVFP4 = 40`, `GGML_TYPE_Q1_0 = 41`) were fork-original — they
  are now confirmed shared with upstream's own type enum. Whether upstream
  ships SYCL kernel coverage for either type (not established by the
  triage's search) is open, and bears on whether `sgox`/`zqoe`/`tjk4`'s
  device-decode certification work should reconcile against an upstream
  implementation before proceeding.
- **`prefill-xmx-int8-peak` / `one-layout-honest-routes`**: whether oneDNN's
  `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY` grouped-GEMM path (documented for
  MoE token routing in oneDNN 2026.x) should be evaluated in place of a
  hand-written dense per-expert GEMM kernel for the F16/F32/BF16 MMID gap
  (`llama.cpp-0yi9`) — flagged as directly actionable but not yet evaluated.

## Links

- [SRC-2026-09-02-005](/sources/SRC-2026-09-02-005.md) — the full merged taxonomy (`taxonomy/final.json`), 18 epics with goal/bar/constraints/ordering/members
- [SRC-2026-09-02-004](/sources/SRC-2026-09-02-004.md) — the triage design brief (hardware truths, numbered owner rulings, verdict vocabulary)
- [SRC-2026-09-02-006](/sources/SRC-2026-09-02-006.md), [SRC-2026-09-02-007](/sources/SRC-2026-09-02-007.md), [SRC-2026-09-02-008](/sources/SRC-2026-09-02-008.md), [SRC-2026-09-02-009](/sources/SRC-2026-09-02-009.md) — epic-level web-research findings cited above
- [Single-allocator invariant](/concepts/single-allocator-invariant.md)
- [WEIGHT handle leases](/concepts/weight-handle-leases.md)
- [Placement decides the executor](/concepts/placement-decides-the-executor.md)
- [Layout follows residency](/concepts/layout-follows-residency.md)
- [SYCL graph replay](/concepts/sycl-graph-replay.md)
- [Tiered memory hierarchy](/concepts/tiered-memory-hierarchy.md)
- [Never shrink context — place KV instead](/concepts/never-shrink-context-place-kv-instead.md)
- [No host waits — event-chain everything](/concepts/no-host-waits-event-chain-everything.md)
- [Efficiency goal is 80% of hardware theoretical peak](/concepts/efficiency-goal-80pct-hardware-theoretical-peak.md)
- [PL2 throttling](/concepts/pl2-throttling.md)
