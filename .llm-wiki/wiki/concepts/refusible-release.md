---
type: concept
title: Refusible release
description: Dropping the last reference calls into the device's allocation_release_coordinator, which may return RETRY_SCHEDULED; the retry is queued via a retry_next_ pointer that allocates nothing, so it works under memory pressure.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# Refusible release

Dropping the last reference calls into the device's allocation_release_coordinator, which may return RETRY_SCHEDULED; the retry is queued via a retry_next_ pointer that allocates nothing, so it works under memory pressure.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
