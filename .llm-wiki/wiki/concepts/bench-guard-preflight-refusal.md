---
type: concept
title: Bench-guard preflight refusal
description: The three ordered host-health preflight checks (stale GPU tenant, Shmem ceiling, PL2 throttle/active card) that can refuse a benchmark run with exit 3.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-001
    resource: /sources/SRC-2026-08-29-001.md
---

# Bench-guard preflight refusal

Three preflight refusal classes in `scripts/bench-guard.sh`, checked in the
order the script runs them (tenants and Shmem are single checks; the
throttle/act_freq check is a poll loop, hence last):

1. **Stale GPU tenant** — any live `llama-cli|llama-bench|llama-completion`
   process.
2. **Shmem ceiling** — `Shmem` in `/proc/meminfo` above 10 GB (the
   TTM-shmem OOM signature this repo has hit repeatedly).
3. **PL2 throttle / active card** — `throttle/status != 0` or
   `act_freq != 0`, polled up to `--max-wait` (default 360 s) under
   `<card>/device/tile0/gt0/freq0/{throttle/status,act_freq}`. `<card>` is
   derived **live** from the PCI device symlink (`readlink -f
   /sys/class/drm/card*/device`) against `0000:03:00.0` (B70) or
   `0000:07:00.0` (B50) — never a static `cardN` index; DRM numbering moves
   across boots.

## Evidence (2026-08-21)

PL2 throttling alone decayed identical-binary readings 2–5× with a clean
kernel log (HEAD pp512 121.2 → 22.9). Separately, two 16-hour-stale hung
`llama-cli` tenants depressed every reading alongside them *and* masked a
real B70 defect until they were found and killed.

## Merged from

Consolidates former duplicate page `preflight-refusal-classes`.

## Links

- [SRC-2026-08-29-001](/sources/SRC-2026-08-29-001.md)
- [scripts/bench-guard.sh](/entities/scriptsbench-guardsh.md)
- [bench-guard VALID/SUSPECT verdict](/concepts/bench-guard-validsuspect-verdict.md)
- [Measurement masking (co-resident tenants)](/concepts/measurement-masking-co-resident-tenants.md)
