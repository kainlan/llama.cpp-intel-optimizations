---
type: concept
title: Multi-GPU device collapse
description: On current Level Zero multi-GPU runtimes the scheduler-visible device set collapses to one device, so all contexts in a process share device 0's registry slot regardless of how many physical GPUs are installed.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# Multi-GPU device collapse

On current Level Zero multi-GPU runtimes the scheduler-visible device set collapses to one device, so all contexts in a process share device 0's registry slot regardless of how many physical GPUs are installed.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
