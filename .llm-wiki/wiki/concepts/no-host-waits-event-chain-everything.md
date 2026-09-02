---
type: concept
title: No host waits — event-chain everything
description: Owner ruling that SYCL dispatch paths must express ordering as event dependencies (depends_on, oneDNN sycl_interop::execute deps), never a host wait/drain per op; waits are confined to teardown, error paths, and debug/init.
created: 2026-09-02
updated: 2026-09-02
sources:
  - id: SRC-2026-09-02-001
    resource: /sources/SRC-2026-09-02-001.md
  - id: SRC-2026-09-02-003
    resource: /sources/SRC-2026-09-02-003.md
---

# No host waits — event-chain everything

Owner ruling (2026-08-20): no `wait()` / `wait_and_throw()` / per-op drain in
SYCL dispatch paths. Ordering between GPU work, host-side compute, and
cross-device copies must be expressed as SYCL event dependencies —
`depends_on`, oneDNN's `sycl_interop::execute` dependency lists — not by
blocking the submitting host thread until a queue empties. A design that
proposes adding a wait as its fix is against design; a change that removes a
wait in favour of an event dependency is aligned.

The rule exists because in-order SYCL queues only order **submissions on that
queue**, not library-internal work, cross-device work, or CPU-side compute —
so a host wait is doing real synchronization work that a naive read of
"in-order queue" would say is unnecessary, and simply deleting waits without
replacing the ordering they provided reintroduces races.

## Where the exceptions are

The ruling is about the **hot dispatch path**, not "no `wait()` ever calls."
Documented exceptions, concretely (from the tiered-KV B2 overlapped
host-task-attention design):

- **Teardown**: draining any still-pending CPU-attention future at a
  graph-boundary or zone-reset point, alongside the existing
  `g_pending_scatter`/`g_pending_cpu_pipeline` drains.
- **The one bounded, unavoidable input-side join**: e.g. waiting on the
  `sycl::event` for a GPU RoPE kernel's async D2H copy immediately before a
  CPU task that consumes it is enqueued — but that wait is **deferred to
  right before the CPU task**, not issued at submission time, so it never
  blocks GPU submission for other graph regions in between (mirrors the
  existing `act_deferred_evt`/`act_deferred_pending` pattern).
- **A real consumer's blocking flush** — when a downstream op actually
  depends on pending host-side work, that IS the ordering edge, not a
  substitute for one. A non-consumer op reached first triggers only a
  non-blocking poll.
- Error paths and debug/init instrumentation (e.g. `GGML_SYCL_SAFE_MODE=1`
  deliberately drains per op to localize faults — that is a diagnostic mode,
  not the production dispatch path).

`sycl::queue::host_task` is explicitly avoided for compute dispatch too —
"Level Zero driver has issues with it that can corrupt event handles and
cause crashes" (`ggml-sycl.cpp` comment cited in the tiered-KV B2 addendum) —
which is a related but distinct constraint from the wait rule.

## Links

- [SRC-2026-09-02-001](/sources/SRC-2026-09-02-001.md) — CLAUDE.md, "No host waits" architecture ruling
- [SRC-2026-09-02-003](/sources/SRC-2026-09-02-003.md) — tiered-KV B2 addendum, event-topology design section
- [Never shrink context — place KV instead](/concepts/never-shrink-context-place-kv-instead.md) — the tiered-KV work this ruling was written against
- [Placement decides the executor](/concepts/placement-decides-the-executor.md)

**Tracked by:** `event-ordered-graph-replay` (new epic, no tracker id yet), `llama.cpp-ic4f` — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
