#pragma once
// Per-(tensor, device) cached MoE expert route table (perf-recovery epic, track B).
// Built once per generation pair; consumed read-only by decode and PP dispatch.
// Ownership: entries hold real mem_handle leases (canonical contract: a live
// model's weights are non-evictable while leased — that is the desired
// semantics during inference). Invalidation releases the leases.
#include "mem-handle.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

// The "plan side" generation for route-table invalidation (perf-recovery
// track B, llama.cpp-1tjn). placement_plan (unified-cache.hpp) carries no
// generation counter of its own -- a MID_LOAD_REPLAN republishes the SAME
// plan snapshot object via ggml_sycl_republish_current_plan(), so its
// identity does not change even though reclaim_weight_entries() resets the
// cache's actual weight materialization underneath it. moe_expert_storage_
// generation (common.hpp) tracks per-tensor storage rewrites but is not
// itself touched by MID_LOAD_REPLAN's reclaim either. This counter is bumped
// explicitly by unified_cache::reclaim_weight_entries() for MID_LOAD_REPLAN
// so a cached route table is always invalidated across a replan.
inline std::atomic<uint64_t> & moe_route_table_replan_epoch() {
    static std::atomic<uint64_t> epoch{ 1 };
    return epoch;
}

inline uint64_t moe_route_table_bump_replan_epoch() {
    return moe_route_table_replan_epoch().fetch_add(1, std::memory_order_relaxed) + 1;
}

inline uint64_t moe_route_table_current_replan_epoch() {
    return moe_route_table_replan_epoch().load(std::memory_order_relaxed);
}

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
    // Hot-path cache of the cache-entry readiness bit: populated by B3 at
    // table build so decode/PP dispatch can check readiness without a
    // debug_info() call per dispatch.
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

static inline bool moe_route_table_current(const moe_route_table_stamp & stamp,
                                           uint64_t                      plan_generation,
                                           uint64_t                      expert_storage_generation) {
    return stamp.valid && stamp.plan_generation == plan_generation &&
           stamp.expert_storage_generation == expert_storage_generation;
}

}  // namespace ggml_sycl
