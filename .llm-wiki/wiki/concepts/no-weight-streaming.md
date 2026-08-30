---
type: concept
title: No weight streaming
description: Copying host-resident weights into device scratch per dispatch is forbidden (a PCIe copy plus scratch pressure); the answer is CPU expert dispatch overlapped via sycl::depends_on.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-002
    resource: /sources/SRC-2026-08-29-002.md
---

# No weight streaming

Copying host-resident weights into device scratch per dispatch is forbidden (a PCIe copy plus scratch pressure); the answer is CPU expert dispatch overlapped via sycl::depends_on.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-002](/sources/SRC-2026-08-29-002.md)
