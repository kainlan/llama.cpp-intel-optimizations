---
type: concept
title: Never shrink context — place KV instead
description: Owner ruling that default n_ctx must stay at the model's max; when KV does not fit VRAM the fix is placement (tiered KV to pinned host), never auto-shrinking n_ctx and never refusing context init. Explicit -c is the only way to get a smaller context.
created: 2026-09-02
updated: 2026-09-02
sources:
  - id: SRC-2026-09-02-001
    resource: /sources/SRC-2026-09-02-001.md
  - id: SRC-2026-09-02-003
    resource: /sources/SRC-2026-09-02-003.md
---

# Never shrink context — place KV instead

Default `n_ctx` must equal the model's trained max. If the KV cache for that
context does not fit in VRAM, the fix is **placement** — moving KV to pinned
host memory via a tiered KV cache — never auto-shrinking `n_ctx`, and never
refusing context initialization outright. An explicit `-c <N>` from the user
is the only sanctioned way to get a smaller context.

## Why this needed a ruling

This fork disabled upstream llama.cpp's context fitter (`fit_params=false`
under SYCL) but the unified cache only **validates** whether KV fits — it
never lowers `n_ctx` itself. Disabling half of an upstream mechanism without
replacing the other half is its own general pitfall (CLAUDE.md records it as
"disabling an upstream mechanism transfers half the job"). That gap surfaced
concretely as `llama.cpp-uize` (2026-08-16): `llama-cli`
defaults to `n_ctx_train=131072`, and 131K of GPT-OSS KV (~3.2 GB) exceeded
the Arc Pro B50's placement headroom, so context init failed with a
misleading `[SYCL-PLAN] runtime KV update rejected: MMID workspace
demand/accounting failed` message immediately followed by a segfault — not a
chat-correctness or MMID regression, just an unfitted context. The refusal
message was later fixed to do the arithmetic and name a context that does
fit (e.g. "the largest context that fits is about `-c 56576`"), and the
workaround of pinning `-c 4096` on the canonical GPT-OSS chat gate was
verified to reach correct output. But a refusal was always a stopgap: three
parts were scoped for `llama.cpp-uize` — parts 1–2 (truthful refusal
message, `-c` pin as a workaround) landed; part 3 (re-place KV to host tiers
instead of refusing) was deliberately left unimplemented pending an owner
decision on whether the canonical gate should pin `-c` at all.

## The ruling

The owner decision landed: **never shrink, place instead.** This is the
design behind the tiered-KV work (`docs/plans/2026-08-26-tiered-kv-placement.md`
and its addenda, e.g. the B2 overlapped host-task-attention design in
`docs/plans/2026-08-27-tkv13-b2-addendum.md`) — demoted KV layers move to
host-pinned memory rather than the context being capped, and CPU-side
attention over host-resident KV is overlapped with GPU work via
[no-host-waits event chaining](/concepts/no-host-waits-event-chain-everything.md)
rather than serialized behind it.

## Consequence for tasks and gates

A task or design that proposes auto-shrinking `n_ctx`, or that treats a
context-init refusal as the fix rather than a stopgap, is against this
ruling. `-c` remains available as an explicit, user-requested override — it
is the *default-shrinking* and *silent-refusal* behaviors that are
forbidden, not a user asking for a smaller context on purpose.

## Links

- [SRC-2026-09-02-001](/sources/SRC-2026-09-02-001.md) — CLAUDE.md, GPT-OSS chat gate section (`llama.cpp-uize`) and architecture rulings
- [SRC-2026-09-02-003](/sources/SRC-2026-09-02-003.md) — tiered-KV B2 addendum (the placement implementation this ruling authorizes)
- [No host waits — event-chain everything](/concepts/no-host-waits-event-chain-everything.md)
- [Placement decides the executor](/concepts/placement-decides-the-executor.md)

**Tracked by:** `llama.cpp-sk1xz` — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
