//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// What a graph prestage does with one source tensor.
//
// The rule is a pure function of facts the backend gathers (ggml_sycl_prestage_facts_of in ggml-sycl.cpp), so it can
// be unit-tested on a host with no GPU, no SYCL runtime and no backend library
// (tests/test-sycl-prestage-class.cpp). The backend does not carry a second copy of it.

#pragma once

#include "ggml.h"

// The first three classes are decided without copying. INPUT has its own stable staging, and a KV-host source is
// staged as before. Which classes ggml_sycl_prestage_needs_device_copy counts is ggml_sycl_prestage_reports_bytes.
enum class ggml_sycl_prestage_class {
    CONTROL,
    WEIGHT,
    DEVICE,
    INPUT,
    KV_HOST,
    COPY,
};

struct ggml_sycl_prestage_facts {
    bool is_control         = false;  // cross-device control storage
    bool is_weight          = false;
    bool is_device          = false;  // device USM
    bool is_input           = false;  // the tensor, or a tensor it views, carries GGML_TENSOR_FLAG_INPUT
    bool is_kv_host         = false;  // its bytes live in the KV-host buft
    bool root_is_op_output  = false;  // ggml_sycl_prestage_root_is_op_output
    bool root_is_graph_node = false;  // the view root is a node of the graph being staged (asked only when the
                                      // rule reads it: a host op output no earlier class claims)
};

// The tensor whose storage `tensor` views, or `tensor` itself.
static inline const ggml_tensor * ggml_sycl_prestage_view_root(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    while (tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor;
}

// A view of an INPUT tensor is that input. ggml_view_tensor() does not copy flags, so the INPUT flag lives on the root
// (llama's recurrent-state copy indices are `ggml_view_1d(s_copy, ...)`), and a flag-only test would treat the view as
// an ordinary host tensor: not staged to a stable device buffer, and never refreshed on replay.
//
// One rule for every reader: the prestage's INPUT fact, the replay refresh, the segment slot's input list, and a
// recorded kernel that reads a staged copy (bin-broadcast). A reader that accepted a tensor the prestage no longer
// stages or refreshes would read a staging entry left from an earlier token.
static inline bool ggml_sycl_prestage_is_input(const ggml_tensor * tensor) {
    for (; tensor; tensor = tensor->view_src) {
        if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
            return true;
        }
    }
    return false;
}

// True when the graph computes this node. ggml_build_forward_select adds every branch to the graph and flags only the
// selected one, so llama's unselected input-embedding branches (a GET_ROWS on inp_mixed_tokens, a SET_ROWS through
// inp_mixed_slots) are nodes too. The scheduler does not upload an input that only such a node reads, and ggml-alloc
// frees such a node's output at once when nothing consumes it, so running it reads indices nobody wrote and writes
// through them into memory another tensor owns. A leaf never carries the flag: ask this of nodes only.
//
// One rule for every walker that runs, stages or plans a node. The executors ask it through ggml_sycl_is_noop. The
// readers of a node's sources (the pre-stage, its refresh, the segment slot's input list, the layer-block pre-block
// copy), the walkers that choose nodes to run (the MoE topology scan and dispatch list, the persistent TG plan) and
// the fusions that also run a look-ahead node ask it directly. tests/test-sycl-graph-compute-flag-source.py names
// each one, and a new walker must ask it too. An input that only an uncomputed node reads is not staged, and no
// kernel asks for its staged copy, because only a computed node runs.
static inline bool ggml_sycl_graph_node_is_computed(const ggml_tensor * node) {
    return (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
}

// True when an op wrote the tensor's bytes: its view root is not a leaf. The root decides, because a view's own op is
// GGML_OP_VIEW whether it views a leaf or an op output.
static inline bool ggml_sycl_prestage_root_is_op_output(const ggml_tensor * tensor) {
    const ggml_tensor * root = ggml_sycl_prestage_view_root(tensor);
    return root && root->op != GGML_OP_NONE;
}

// A host-resident op output that no earlier class claims, and that the graph being staged does not produce, is another
// split's per-token activation: the CPU computed it into a host compute buffer the SYCL backend accepts in place, so
// the scheduler made no device copy. It is staged as an INPUT, to a stable buffer re-copied on every call.
//
// An op output this graph produces stays COPY. When the backend's own compute buffer is host memory
// (GGML_SYCL_HOST_COMPUTE=1) every intermediate is a host op output; staging them as inputs would copy each one before
// its producer has run, and a consumer that looks staging up by name while recording would bake that stale copy.
//
// The rule sits after KV_HOST so that KV state never becomes per-call staging, whatever the root of a KV write is.
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
    if (facts.root_is_op_output && !facts.root_is_graph_node) {
        return ggml_sycl_prestage_class::INPUT;
    }
    return ggml_sycl_prestage_class::COPY;
}

// Whether ggml_sycl_prestage_needs_device_copy reports the tensor's bytes: a COPY source, or a host op output another
// split produced. Ordinary kernels read both from host memory in place (only consumers that look staging up use the
// staged copy), so a resolver host return of either is a host read the counter must see. A flagged INPUT is not
// counted: its host storage is the stable buffer set_tensor writes and the replay refresh re-copies.
//
// The answer does not depend on root_is_graph_node: a host op output is COPY when the graph produces it and INPUT
// when it does not, and both are counted. That is what lets a caller with no graph in hand use it.
static inline bool ggml_sycl_prestage_reports_bytes(const ggml_sycl_prestage_facts & facts) {
    const ggml_sycl_prestage_class c = ggml_sycl_prestage_class_of(facts);
    return c == ggml_sycl_prestage_class::COPY || (c == ggml_sycl_prestage_class::INPUT && !facts.is_input);
}
