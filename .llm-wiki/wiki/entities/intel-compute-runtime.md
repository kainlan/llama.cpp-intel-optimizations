---
type: entity
title: Intel compute-runtime
description: The loaded Level Zero driver is stock 26.31 since a one-way PPA upgrade on 2026-08-18; the patched 26.22 Battlemage/BMG-only build (branch llama/26.22-cross-device) is preserved on disk but no longer loaded.
created: 2026-08-29
updated: 2026-09-02
sources:
  - id: SRC-2026-08-29-003
    resource: /sources/SRC-2026-08-29-003.md
  - id: SRC-2026-09-02-002
    resource: /sources/SRC-2026-09-02-002.md
  - id: SRC-2026-09-02-009
    resource: /sources/SRC-2026-09-02-009.md
---

# Intel compute-runtime

**Currently loaded: stock 26.31** (`libze_intel_gpu.so.1 -> libze_intel_gpu.so.1.17.39395`,
package `libze-intel-gpu1 26.31.39395.13`). A one-way PPA upgrade on
2026-08-18 moved the loader here; every measurement taken since that date ran
on 26.31, not on the patched build.

## History (accurate for 2026-05-30 through 2026-08-17 only)

Before the 2026-08-18 PPA upgrade, the system-default loader was a patched
26.22 Battlemage/BMG-only build (`libze_intel_gpu.so.1.15.38646`, branch
`llama/26.22-cross-device`, based on `upstream/releases/26.22`), carrying a
hung-i915 discovery fix, cross-device in-order dependency fixes, and the
upstream PR 930 USM compression fix. It was configured `SUPPORT_BMG=TRUE`
because the installed IGC/ocloc cannot recognize 26.22's future Xe3p/NVLP
built-ins.

## ⚠️ Loader state correction (2026-08-30, `llama.cpp-09um`)

The patched 26.22 build (`1.14.37435`) and the stock `.orig` remain on disk
in `/usr/lib/x86_64-linux-gnu/` but are **not loaded**. Whether 26.31 carries
the patched build's three fixes above is **unverified**; the 26.22-vs-26.27
GPT-OSS TG regression measured 2026-07-24 has not been re-measured on 26.31.
Any page or doc calling the patched 26.22 build "the system default" in the
present tense is describing history, not current state — verify with
`libze-intel-gpu1` package version before trusting a driver-version claim.

## 26.31.39395.13 release content (2026-09-01 research)

The currently-loaded 26.31.39395.13 is an enablement/maintenance release:
Crescent Island device-ID prep, Nova Lake Xe3P groundwork, Level Zero API
bumped to 1.17 for listed platforms; Battlemage (B70/B50, this host's cards)
retained at production status. No decode-bandwidth-relevant changelog entry
was found for this version — the driver pin is not stale relative to a
materially faster release currently available.
([github.com/intel/compute-runtime/releases](https://github.com/intel/compute-runtime/releases))

## Links

- [SRC-2026-08-29-003](/sources/SRC-2026-08-29-003.md) · [SRC-2026-09-02-002](/sources/SRC-2026-09-02-002.md) · [SRC-2026-09-02-009](/sources/SRC-2026-09-02-009.md)
- [SYCL fork epic taxonomy (2026-09-01 tracker triage)](/syntheses/sycl-fork-epic-taxonomy-2026-09.md)
- [Cross-device in-order dependency fixes](/concepts/cross-device-in-order-dependency-fixes.md) — carried by the patched build; presence on 26.31 unverified
- [IGC](/entities/igc.md)
- [ocloc](/entities/ocloc.md)
