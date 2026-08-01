//
// The XMX GEMM batch gate, stated once.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

// Whether a batch size falls in the range the XMX GEMM path handles.
//
// WHY THIS IS A FUNCTION AND NOT TWO COPIES OF AN EXPRESSION
//
// The rule lived inline at both dispatch sites -- ggml_sycl_select_preferred_kernel
// and ggml_sycl_mul_mat -- as `batch >= 1 && batch < g_ggml_sycl_xmx_threshold`,
// and a comment in tests/test-mmq-xmx-dispatch.cpp asserted the two were
// "identical". Nothing enforced that, and this variable has form: the threshold's
// DEFAULT was duplicated the same way and the copies silently disagreed 1024 vs
// 64 for months (llama.cpp-d5h0 / 43d04b327), while the ENABLE flag was read two
// incompatible ways at these very sites (llama.cpp-wvbw).
//
// It is also what lets the gate be TESTED. The test that claimed to cover this
// could not: it is a standalone binary that links SYCL and not ggml-sycl, so it
// could not reach the dispatch functions, and it settled for recomputing a
// hardcoded `batch < 8` locally and printing it (llama.cpp-cwev). A test that
// re-derives the rule it is checking cannot fail. Calling the same function the
// dispatch calls is the difference between asserting a shared fact and asserting
// a copy.
//
// Deliberately takes the threshold as a PARAMETER rather than reading
// g_ggml_sycl_xmx_threshold: that global lives inside #ifdef GGML_SYCL_XMX_GEMM
// in ggml-sycl.cpp, and depending on it here would put this header out of reach
// of exactly the standalone test it exists to serve.
//
// The threshold's default (64) is NOT stated here -- it is the
// GGML_SYCL_XMX_THRESHOLD row of the sycl_env_settings table in
// ggml_check_sycl(), guarded by scripts/check-sycl-xmx-threshold-default.sh. A
// second copy in this header would be the d5h0 bug over again.
//
// Note `threshold <= 1` disables XMX for every batch, which is what makes the
// global's fail-closed 0 initializer work.
static inline bool ggml_sycl_xmx_batch_in_range(int64_t batch, int threshold) {
    return batch >= 1 && batch < static_cast<int64_t>(threshold);
}
