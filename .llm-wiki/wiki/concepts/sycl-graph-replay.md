---
type: concept
title: SYCL graph replay
description: Command-graph record/replay submission path; a replay-active graph currently never reaches a terminal state, so its invocation is held indefinitely (release_invocation no-ops), blocking other contexts with a one-shot DEVICE_BUSY.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# SYCL graph replay

Command-graph record/replay submission path; a replay-active graph currently never reaches a terminal state, so its invocation is held indefinitely (release_invocation no-ops), blocking other contexts with a one-shot DEVICE_BUSY.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
