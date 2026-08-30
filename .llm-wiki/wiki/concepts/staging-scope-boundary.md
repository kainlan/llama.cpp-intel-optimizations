---
type: concept
title: Staging scope boundary
description: Staging that converts the format of device-resident data (e.g., on-device MXFP4 dequant to f16 scratch) is not a placement violation — the rule governs residency, not layout conversion — but such conversion scratch belongs to planned zones.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Staging scope boundary

Staging that converts the format of device-resident data (e.g., on-device MXFP4 dequant to f16 scratch) is not a placement violation — the rule governs residency, not layout conversion — but such conversion scratch belongs to planned zones.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
