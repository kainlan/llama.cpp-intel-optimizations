//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "attn-host-dispatch.hpp"

namespace ggml_sycl {

bool attn_tensor_depends_on(const ggml_tensor * tensor, const ggml_tensor * target, int depth) {
    if (!tensor || !target || depth > 32) {
        return false;
    }
    for (const ggml_tensor * t = tensor; t; t = t->view_src) {
        if (t == target) {
            return true;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (attn_tensor_depends_on(tensor->src[i], target, depth + 1)) {
            return true;
        }
    }
    return false;
}

bool attn_op_consumes_tensor(const ggml_tensor * consuming_dst, const ggml_tensor * pending_dst) {
    if (!consuming_dst || !pending_dst) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (attn_tensor_depends_on(consuming_dst->src[i], pending_dst, 0)) {
            return true;
        }
    }
    return false;
}

}  // namespace ggml_sycl
