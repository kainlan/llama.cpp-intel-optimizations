---
type: concept
title: iGPU VRAM budget must not count host RAM
description: The Arrow Lake-S iGPU reports ~231.7 GB global_mem_size, which is host RAM, not VRAM; the backend's VRAM budget calc (min(total*pct, free_at_init) at a 100% default) is correct for discrete cards but claims the whole host for an integrated GPU. Until fixed, pin ONEAPI_DEVICE_SELECTOR=level_zero:0,1 (llama.cpp-403s).
created: 2026-09-02
updated: 2026-09-02
sources:
  - id: SRC-2026-09-02-001
    resource: /sources/SRC-2026-09-02-001.md
  - id: SRC-2026-09-02-004
    resource: /sources/SRC-2026-09-02-004.md
---

# iGPU VRAM budget must not count host RAM

The VRAM budget calculation (`min(total * pct, free_at_init)`, default
`pct = 100%`) is correct **by design for discrete cards** — low free VRAM on
a discrete GPU is a real system constraint (other GPU tenants, driver
overhead) to fix at the system level, not an app bug to paper over. It is
**catastrophically wrong for an integrated GPU**, and that gap is the root
cause of this host's OOM history (`llama.cpp-403s`, measured 2026-08-01).

## The mechanism

The Arrow Lake-S iGPU reports `global_mem_size` = **231.7 GB** — 94% of the
host's 246.9 GB — because for an integrated GPU, "VRAM" *is* system RAM.
`ggml-sycl.cpp` feeds that number into the same budget path a discrete card
uses, at the same 100% default, and neither `ggml-sycl.cpp` nor
`unified-cache.cpp` checks `host_unified` or `is_integrated` anywhere. So the
backend can claim the whole machine's memory as "VRAM budget" for a device
that has no VRAM of its own.

`sycl::info::device::host_unified_memory` is queryable, and the device caps
struct already captures `global_mem_size` (`common.cpp:716`) — the
information needed to detect and exclude this case is present, it is simply
never consulted. Fixing it means treating "free VRAM" and "free host RAM" as
the same pool for an integrated device, and not budgeting 100% of a pool the
host also needs — not raising or lowering the percentage, but recognizing
the device class before applying the calculation at all.

## Measured impact and current workaround

Isolated with one variable — same 19 MB model, same single-threaded
`llama-completion`, only the device selector changed:

| `ONEAPI_DEVICE_SELECTOR` | peak `Shmem` |
|---|---:|
| `level_zero:0` (B70 only) | 2.4 GB |
| `level_zero:0,1` (B70 + B50) | 2.4 GB |
| unset (adds the iGPU) | 127.8 GB |

Until the budget path is fixed, **every local run must pin
`ONEAPI_DEVICE_SELECTOR` to the discrete cards** (`level_zero:0,1`, or a
single card as needed) to exclude the iGPU. An unpinned run is not a
theoretical risk — it is this host's primary global-OOM mechanism, and it
is the **default** state, since nothing in `test-llama-archs` or
`test-thread-safety`'s ctest registration sets a selector.

## Links

- [SRC-2026-09-02-001](/sources/SRC-2026-09-02-001.md) — CLAUDE.md, VRAM budget calc and "never-loop" sections
- [SRC-2026-09-02-004](/sources/SRC-2026-09-02-004.md) — design brief, owner rulings §10
- [VRAM budget](/concepts/vram-budget.md)
- [Shmem ceiling](/concepts/shmem-ceiling.md)
