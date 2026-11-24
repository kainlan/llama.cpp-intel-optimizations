//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MOE_RING_CONSUMER_HPP
#define GGML_SYCL_MOE_RING_CONSUMER_HPP

// Whether a load keeps a device's PP MoE oneDNN scratch ring, from the plan made with it.
//
// The loader sizes the ring for a model with an expert weight of a type a ring consumer runs
// (ggml_sycl_inventory_ring_has_consumer in ggml-sycl.cpp), and the ring's room comes out of the device's weight
// capacity. A ring whose weights reach the batched executor's resident claim and nothing else (moe_pp_ring_use: NVFP4
// today) serves a layer only when every expert of it is planned on the device, and stages no host expert. When the plan
// made with such a ring leaves an expert of those weights on the host, the ring has traded prompt room for a decode loss
// on every token, and measured on the B50 (Inkling 2 Mini) it lost on the prompt as well. The load then drops the ring
// and plans once more (ggml_sycl_stage_drop_displacing_ring in ggml-sycl.cpp). A ring with a weight that reaches the
// staged claim or the per-expert SOA staging serves host experts too, and the load keeps it as it always has.
//
// Host-only: the plan is data, so tests/test-sycl-moe-ring-consumer.cpp checks the answer on synthetic plans.

#include "moe-mmvq-tables.hpp"
#include "unified-cache.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// An expert weight of an inventory, as the ring decision reads it: its name, its expert count (ne[2]) and the ring uses
// its type reaches (moe_pp_ring_uses_of_type). The stage and the plan dry run build the list from the inventory with
// the filter the ring's sizing reads.
struct moe_pp_ring_weight {
    const char * name      = nullptr;
    int64_t      n_experts = 0;
    uint32_t     uses      = MOE_PP_RING_USE_NONE;
};

// What the plan made with a ring says about it.
struct moe_pp_ring_resident_check {
    size_t  n_weights     = 0;      // the weights with a ring use
    bool    resident_only = false;  // at least one, and every one reaches MOE_PP_RING_USE_RESIDENT alone
    int64_t host_experts  = 0;      // when resident_only: the experts of those weights the plan leaves on the host

    // Whether the load drops the ring: it serves resident layers only, and the plan made with it left an expert of
    // those layers' weights on the host.
    bool displaces_experts() const { return resident_only && host_experts > 0; }
};

// The check of `weights` under `plan`. A weight with no ring use is not a ring weight and is skipped. Any weight with a
// use beside the resident claim makes the ring one the load keeps as before, and nothing more is counted. Otherwise
// every expert in [0, n_experts) of each weight that the plan's semantic index places on the host is counted, on any
// device's plan: an expert the plan has no entry for is not on the host.
inline moe_pp_ring_resident_check moe_pp_ring_check_resident_only(const placement_plan &     plan,
                                                                  const moe_pp_ring_weight * weights,
                                                                  size_t                     n_weights) {
    moe_pp_ring_resident_check check;
    if (weights == nullptr) {
        return check;
    }
    bool resident_only = true;
    for (size_t i = 0; i < n_weights; ++i) {
        if (weights[i].uses == MOE_PP_RING_USE_NONE) {
            continue;
        }
        ++check.n_weights;
        if (weights[i].uses != MOE_PP_RING_USE_RESIDENT) {
            resident_only = false;
        }
    }
    check.resident_only = resident_only && check.n_weights > 0;
    if (!check.resident_only) {
        return check;
    }
    for (size_t i = 0; i < n_weights; ++i) {
        if (weights[i].uses == MOE_PP_RING_USE_NONE || weights[i].name == nullptr || weights[i].n_experts <= 0) {
            continue;
        }
        check.host_experts += plan.count_planned_experts(weights[i].name, weights[i].n_experts, -1).host;
    }
    return check;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_MOE_RING_CONSUMER_HPP
