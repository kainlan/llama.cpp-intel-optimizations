// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

namespace ggml_sycl {

// Clear a completed global plan only after multi-device consumers have copied
// it. Kept pure so contract tests can exercise local state without a mutating
// test seam in the installed backend API.
inline bool clear_completed_multi_device_global_plan(bool & has_plan, bool multi_device) {
    if (!has_plan || !multi_device) {
        return false;
    }
    has_plan = false;
    return true;
}

}  // namespace ggml_sycl
