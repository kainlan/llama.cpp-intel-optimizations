//
// Source-staging decision for the simple-consumer route (llama.cpp-m1or).
//
// The simple-consumer route runs an elementwise-class op (ROPE, GET_ROWS, ADD,
// RMS_NORM, ...) on a chosen execution device. For every source it must answer
// one question: can the op read this operand where it already lives, or must
// the route copy it into a staging allocation on the execution device first?
// A "yes, stage" answer for any source makes the route fire, and a fired route
// allocates, copies synchronously, and drains the target queue.
//
// The policy is dependency-free so it can be unit-tested on a host with no GPU,
// no SYCL runtime and no backend library (tests/test-sycl-split-exec-policy.cpp).
// ggml-sycl.cpp gathers the facts about each source and asks this function; it
// does not carry a second copy of the rule.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

namespace ggml_sycl {

// What the planner knows about one source operand, relative to the execution
// device the plan has already chosen.
struct simple_consumer_src_facts {
    // Device whose storage owns the source, or -1 when no device owns it (host
    // storage, or storage the resolver could not attribute).
    int  owner                         = -1;
    // The source is a graph leaf (GGML_OP_NONE): an input or a weight, never an
    // intermediate produced earlier in this graph.
    bool is_leaf                       = false;
    // The leaf is a per-graph control input (GGML_TENSOR_FLAG_INPUT and not a
    // weight): token ids, positions, KV row indices, masks. Weights are never
    // control inputs -- placement forbids a GPU zero-copy read of host weights.
    bool is_control_input              = false;
    // The resolved pointer is usable as a device operand on the execution
    // device (device USM of that device, or shared USM).
    bool accessible_as_device_operand  = false;
    // The resolved pointer is host USM allocated in the execution device's own
    // SYCL context, so a kernel submitted to that device can read it in place.
    bool host_addressable_by_execution = false;
};

// True when the route must stage `src` onto `execution_device` before the op
// can run there. `current_device` is the device whose graph is executing (the
// backend context device); `active_source_seen` is true when some source of
// the same op is a device-owned tensor or an intermediate.
inline bool simple_consumer_src_needs_staging(const simple_consumer_src_facts & src,
                                              int                               execution_device,
                                              int                               current_device,
                                              bool                              active_source_seen) {
    if (execution_device < 0 || src.accessible_as_device_operand) {
        return false;
    }
    if (src.owner >= 0) {
        return src.owner != execution_device;
    }
    if (!src.is_leaf || !active_source_seen) {
        return false;
    }
    // A control input in the executing device's own host USM is read in place,
    // exactly as a single-device graph reads inp_pos and inp_tokens. Staging it
    // bought nothing but a synchronous copy and a queue drain per op (61 per
    // decoded token in a dense split). A cross-device target keeps staging:
    // host USM of one context is not addressable from another, and the cards
    // have no P2P.
    return !(execution_device == current_device && src.is_control_input && src.host_addressable_by_execution);
}

}  // namespace ggml_sycl
