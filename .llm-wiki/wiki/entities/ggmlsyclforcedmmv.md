---
type: entity
title: GGML_SYCL_FORCE_DMMV
description: Forces DMMV kernels, but has no effect at batch=1 on its own because the TG fast-path returns before the flag is read; pair with GGML_SYCL_TG_FAST=0.
created: 2026-08-29
updated: 2026-08-29
sources:
  - id: SRC-2026-08-29-004
    resource: /sources/SRC-2026-08-29-004.md
---

# GGML_SYCL_FORCE_DMMV

Forces DMMV kernels, but has no effect at batch=1 on its own because the TG fast-path returns before the flag is read; pair with GGML_SYCL_TG_FAST=0.

## Overview

[Key facts]

## Links

- [SRC-2026-08-29-004](/sources/SRC-2026-08-29-004.md)
