---
type: concept
title: Placement decides the executor
description: The planning pass decides where data lives; inference then executes each op where its data already is — VRAM operand on that GPU, host-pinned operand on the CPU.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Placement decides the executor

The planning pass decides where data lives; inference then executes each op where its data already is — VRAM operand on that GPU, host-pinned operand on the CPU.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)

**Tracked by:** `llama.cpp-3h5gm`, `llama.cpp-ic4f`, `llama.cpp-po3nd.2` — see [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
