---
type: concept
title: Fail-closed admission contract
description: Once an XMX_TILED-claimed op reaches the batched MoE PP executor, every internal decline throws ggml_sycl_fallback_error rather than returning false, because the outer dispatch has already skipped every other route that could decode the claimed layout.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# Fail-closed admission contract

Once an XMX_TILED-claimed op reaches the batched MoE PP executor, every internal decline throws ggml_sycl_fallback_error rather than returning false, because the outer dispatch has already skipped every other route that could decode the claimed layout.

## Definition

[Clear explanation]

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
