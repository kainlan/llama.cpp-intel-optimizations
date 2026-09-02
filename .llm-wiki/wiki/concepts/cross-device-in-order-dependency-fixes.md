---
type: concept
title: Cross-device in-order dependency fixes
description: Local compute-runtime fixes for ordering dependencies across devices, carried in the patched 26.22 build — a historical build no longer loaded on this machine.
created: 2026-08-29
updated: 2026-09-02
sources:
  - id: SRC-2026-08-29-003
    resource: /sources/SRC-2026-08-29-003.md
  - id: SRC-2026-09-02-002
    resource: /sources/SRC-2026-09-02-002.md
---

# Cross-device in-order dependency fixes

Local compute-runtime fixes for ordering dependencies across devices, carried in the patched 26.22 build.

## ⚠️ Currency caveat (2026-09-02)

The patched 26.22 build that carried this fix is **not the loaded driver**
since a 2026-08-18 PPA upgrade moved the system to stock 26.31 (see
[Intel compute-runtime](/entities/intel-compute-runtime.md)). Whether these
fixes are present in 26.31 is unverified — do not assume this fix is active
on the current driver without checking.

## Links

- [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md) · [SRC-2026-09-02-002](/sources/SRC-2026-09-02-002.md)
