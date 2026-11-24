//
// The recoverable-fallback exception of the SYCL backend's graph compute. A refusal inside the segmented record and
// replay paths throws it, and graph_compute's catch drains, releases the cached graph state and returns
// GGML_STATUS_FAILED (docs/design/sycl-mmap-host-tier.md §4.4, "The six sites fail the graph by throwing").
//
// SYCL-free so the fail-closed helper (host-zone-plan.cpp) and its host-only test can throw and catch the same type
// the backend does. common.hpp includes it.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <exception>

// The type stores the pointer it is given, and the catch prints what() after the unwind, so the argument must be a
// string literal: a formatted local would dangle.
class ggml_sycl_fallback_error final : public std::exception {
  public:
    explicit ggml_sycl_fallback_error(const char * reason) noexcept :
        reason_(reason ? reason : "SYCL fallback failed") {}

    const char * what() const noexcept override { return reason_; }

  private:
    const char * reason_;
};
