#pragma once
// Per-(tensor, device) cached MoE expert route table (perf-recovery epic, track B).
// Built once per generation pair; consumed read-only by decode and PP dispatch.
// Ownership: entries hold real mem_handle leases (canonical contract: a live
// model's weights are non-evictable while leased — that is the desired
// semantics during inference). Invalidation releases the leases.
#include "mem-handle.hpp"

#include <cstdint>
#include <vector>

namespace ggml_sycl {

struct moe_route_table_stamp {
    uint64_t plan_generation           = 0;
    uint64_t expert_storage_generation = 0;
    bool     valid                     = false;
};

struct moe_route_entry {
    void *     ptr = nullptr;        // ABI view only; lease is the owner
    mem_handle lease;
    int        layout          = 0;  // layout_mode as int to keep this header light
    int        residency       = 0;  // moe_expert_route_kind as int
    bool       has_ready_event = false;
};

struct moe_route_table {
    moe_route_table_stamp        stamp;
    std::vector<moe_route_entry> experts;  // size n_expert when built

    void invalidate() {
        stamp = {};
        experts.clear();  // releases leases via mem_handle dtors
    }
};

}  // namespace ggml_sycl

static inline bool ggml_sycl_moe_route_table_current(const ggml_sycl::moe_route_table_stamp & stamp,
                                                     uint64_t                                 plan_generation,
                                                     uint64_t expert_storage_generation) {
    return stamp.valid && stamp.plan_generation == plan_generation &&
           stamp.expert_storage_generation == expert_storage_generation;
}
