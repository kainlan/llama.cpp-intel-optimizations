---
type: entity
title: GGML_SYCL_Q8_DENSE_AOS
description: Opt-in (default OFF) routing Q8_0 dense projections AOS so PP batches hit the oneDNN arm; +38% B70 pp512 but −34% B50 tg128, superseded for the PP win by GGML_SYCL_Q8_ONEDNN_COALESCED.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# GGML_SYCL_Q8_DENSE_AOS

Opt-in (default OFF) routing Q8_0 dense projections AOS so PP batches hit the oneDNN arm; +38% B70 pp512 but −34% B50 tg128, superseded for the PP win by GGML_SYCL_Q8_ONEDNN_COALESCED.

## Overview

[Key facts]

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
