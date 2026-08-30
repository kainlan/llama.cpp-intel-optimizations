---
type: entity
title: ONEAPI_DEVICE_SELECTOR
description: Environment variable keying device identity (level_zero:0 = B70, level_zero:1 = B50); the logged device=N is assigned after selector filtering.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-001
    resource: /sources/SRC-2026-08-29-001.md
  - id: SRC-2026-08-29-003
    resource: /sources/SRC-2026-08-29-003.md
---

# ONEAPI_DEVICE_SELECTOR

oneAPI environment variable keying device identity in this machine:
`level_zero:0` = [Arc Pro B70](/entities/intel-arc-pro-b70.md),
`level_zero:1` = [Arc Pro B50](/entities/intel-arc-pro-b50.md);
`level_zero:0,1` selects both cards (host-bounce multi-GPU path).

## Key facts

- **Key off the selector, never the logged `device=N`.** That index is
  assigned *after* selector filtering — a B50-only run and a B70-only run
  both print `device=0`.
- `level_zero:0,1` is the working multi-device configuration for host-bounce
  validation, since there is no direct P2P between the two cards.

## Merged from

Consolidates former duplicate page `oneapi-oneapideviceselector`.

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md) · [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md)
- [oneAPI](/entities/oneapi.md)
- [Host bounce](/concepts/host-bounce.md)
