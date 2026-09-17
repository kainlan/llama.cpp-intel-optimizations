//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// host-weight-alias.hpp -- classifies a host-weight registry NAME collision
// (llama.cpp-ttws).
//
// ggml_backend_sycl_register_host_weight_tensor keys its registry on
// owner + ggml_get_name(tensor), not on the ggml_tensor object. That is the
// right key for the unified cache (one cache entry per GGUF weight), but it
// means two distinct ggml_tensor objects with one name collide. The loader
// produces exactly that whenever a model ties its output head to token_embd:
// llama_model_loader::create_tensor's TENSOR_DUPLICATED path reuses the
// original object only when it already exists in the SAME buffer-type context,
// and Gemma 3n / Gemma 4 (src/models/gemma4.cpp) create the DUPLICATED output
// alias BEFORE the real tok_embd, so the lookup can never hit; llama-family
// archs route the two roles (MUL_MAT output vs GET_ROWS input) to different
// contexts and miss the same way. Per-layer rope_freqs duplicates do too.
//
// Both objects then describe the same GGUF bytes: same name (the GGUF key,
// hence the same weights_map entry, file and offset), same type, same ne[],
// same nbytes. The registry's reconcile merges them onto one extra -- that is
// the intended ownership outcome, not a defect -- so the WARN the registry used
// to print for it ("host weight registry mismatch for token_embd.weight") was
// noise on every tied-embedding load. This classifier separates that expected
// alias from the case the WARN exists for: one key, DIFFERENT metadata, which
// is a genuine conflict (two weights answering to one name, or a tensor whose
// shape/type changed between registrations).
//
// Pure function of ggml_tensor metadata: no SYCL headers, no device, no
// allocation. Kept in its own header so the decision is unit-testable without
// a GPU (ggml/src/ggml-sycl/tests/test-host-weight-alias.cpp) and so the print
// site's structure can be gated (tests/test-sycl-host-weight-alias-source.py).
//

#pragma once

#include "ggml.h"

#include <cstring>

namespace ggml_sycl {
namespace detail {

enum class host_weight_alias_kind {
    // The identical ggml_tensor object was registered again. The registry only
    // reaches its collision branch for this when the object's extra changed
    // underneath it, which is not an alias -- callers keep warning.
    SAME_TENSOR,
    // A second ggml_tensor object over the same GGUF weight: same name, type,
    // shape and byte count. Expected for tied token_embd/output and per-layer
    // rope_freqs duplicates. Not a defect; do not warn.
    ALIAS_SAME_BYTES,
    // Same registry key, different metadata (type, ne[], nbytes or name), or a
    // null side. Two weights answering to one name -- the case the WARN is for.
    DIVERGENT,
};

inline const char * host_weight_alias_kind_name(host_weight_alias_kind kind) {
    switch (kind) {
        case host_weight_alias_kind::SAME_TENSOR:
            return "same-tensor-new-extra";
        case host_weight_alias_kind::ALIAS_SAME_BYTES:
            return "tied-weight-alias";
        case host_weight_alias_kind::DIVERGENT:
            return "divergent";
    }
    return "unknown";
}

// `existing` is the tensor already in the registry row, `incoming` the one being
// registered under the same key. Either may be null (a row whose tensor was
// cleared); null is never an alias.
inline host_weight_alias_kind classify_host_weight_alias(const ggml_tensor * existing, const ggml_tensor * incoming) {
    if (existing == nullptr || incoming == nullptr) {
        return host_weight_alias_kind::DIVERGENT;
    }
    if (existing == incoming) {
        return host_weight_alias_kind::SAME_TENSOR;
    }
    if (existing->type != incoming->type) {
        return host_weight_alias_kind::DIVERGENT;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (existing->ne[d] != incoming->ne[d]) {
            return host_weight_alias_kind::DIVERGENT;
        }
    }
    if (ggml_nbytes(existing) != ggml_nbytes(incoming)) {
        return host_weight_alias_kind::DIVERGENT;
    }
    // The registry key already agreed on the name; re-check it here so the
    // classifier is a complete statement on its own, not one that trusts its
    // caller's key.
    if (std::strncmp(existing->name, incoming->name, GGML_MAX_NAME) != 0) {
        return host_weight_alias_kind::DIVERGENT;
    }
    return host_weight_alias_kind::ALIAS_SAME_BYTES;
}

}  // namespace detail
}  // namespace ggml_sycl
