//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// What a graph prestage does with one source tensor.
//
// The rule is a pure function of facts the backend gathers (ggml_sycl_prestage_classify in ggml-sycl.cpp), so it can
// be unit-tested on a host with no GPU, no SYCL runtime and no backend library
// (tests/test-sycl-prestage-class.cpp). The backend does not carry a second copy of it.

#pragma once

#include "ggml.h"

// The first three classes are decided without copying; COPY is the only class whose bytes
// ggml_sycl_prestage_needs_device_copy reports. INPUT and KV_HOST are not counted but still reach the prestage's copy
// path: INPUT has its own stable staging, and a KV-host source is staged as before.
enum class ggml_sycl_prestage_class {
    CONTROL,
    WEIGHT,
    DEVICE,
    INPUT,
    KV_HOST,
    COPY,
};

struct ggml_sycl_prestage_facts {
    bool is_control        = false;  // cross-device control storage
    bool is_weight         = false;
    bool is_device         = false;  // device USM
    bool is_input          = false;  // the tensor, or a tensor it views, carries GGML_TENSOR_FLAG_INPUT
    bool is_kv_host        = false;  // its bytes live in the KV-host buft
    bool root_is_op_output = false;  // ggml_sycl_prestage_root_is_op_output
};

// True when an op wrote the tensor's bytes: its view root is not a leaf. The root decides, because a view's own op is
// GGML_OP_VIEW whether it views a leaf or an op output.
static inline bool ggml_sycl_prestage_root_is_op_output(const ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    while (tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor->op != GGML_OP_NONE;
}

// A host-resident op output that no earlier class claims is another split's per-token activation: the CPU computed it
// into a host compute buffer the SYCL backend accepts in place, so the scheduler made no device copy. It is staged as
// an INPUT, to a stable buffer re-copied on every call. The rule sits after KV_HOST so that KV state never becomes
// per-call staging: a KV write (SET_ROWS, CPY) is an op whose result views the cache, and the cache is persistent.
static inline ggml_sycl_prestage_class ggml_sycl_prestage_class_of(const ggml_sycl_prestage_facts & facts) {
    if (facts.is_control) {
        return ggml_sycl_prestage_class::CONTROL;
    }
    if (facts.is_weight) {
        return ggml_sycl_prestage_class::WEIGHT;
    }
    if (facts.is_device) {
        return ggml_sycl_prestage_class::DEVICE;
    }
    if (facts.is_input) {
        return ggml_sycl_prestage_class::INPUT;
    }
    if (facts.is_kv_host) {
        return ggml_sycl_prestage_class::KV_HOST;
    }
    if (facts.root_is_op_output) {
        return ggml_sycl_prestage_class::INPUT;
    }
    return ggml_sycl_prestage_class::COPY;
}
