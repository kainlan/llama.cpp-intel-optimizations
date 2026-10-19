#pragma once

// What a prompt layer streams through the VRAM ring (llama.cpp-36j0, owner ruling 2026-10-19). The list is a pure
// function of the published plan, the regime and, for ROUTED_ONLY, the ids: no device, no queue. The enqueue that
// consumes it is moe_stream_phase (bodies in moe-stream-plan.cpp).

#include "moe-exec-cost.hpp"
#include "unified-cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ggml_sycl {

struct moe_stream_item {
    int                layer;
    int                expert_id;
    expert_tensor_role role;
    ggml_sycl_cache_id key;  // left default by moe_stream_plan_build: the entry carries no tensor to derive it from
    size_t             dst_bytes;  // the SOA size the device route wants
    size_t             src_bytes;  // the AOS size of the host copy
    ggml_type          type;       // 0 until the enqueue fills it from src0
    int64_t            ncols, nrows;
};

namespace detail {

// A gate/up/down entry of `layer` that is planned for the host. Host entries carry target_device -1 (the planner sets
// it at placement time), so they serve any device and the device argument of the callers is not a filter.
static inline bool moe_stream_host_triplet_entry(const placement_entry & e, int layer) {
    if (e.expert_id < 0 || e.layer_id != layer || e.on_device) {
        return false;
    }
    return e.expert_role == expert_tensor_role::GATE || e.expert_role == expert_tensor_role::UP ||
           e.expert_role == expert_tensor_role::DOWN;
}

static inline bool moe_stream_id_listed(const int32_t * ids, size_t n_ids, int expert) {
    for (size_t i = 0; i < n_ids; ++i) {
        if (ids[i] == expert) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

// STREAM_ALL: every host-planned gate/up/down entry of `layer`, in plan.entries order, ids ignored.
// ROUTED_ONLY: the same entries restricted to the experts named in ids, still in plan.entries order.
// Any other regime lists nothing. `device` is accepted for symmetry with the enqueue; see the detail helper.
static inline std::vector<moe_stream_item> moe_stream_plan_build(const placement_plan & plan,
                                                                 int                    device,
                                                                 int                    layer,
                                                                 moe_exec_regime        regime,
                                                                 const int32_t *        ids,
                                                                 size_t                 n_ids) {
    (void) device;
    std::vector<moe_stream_item> out;
    if (regime != moe_exec_regime::STREAM_ALL && regime != moe_exec_regime::ROUTED_ONLY) {
        return out;
    }
    for (const placement_entry & e : plan.entries) {
        if (!detail::moe_stream_host_triplet_entry(e, layer)) {
            continue;
        }
        if (regime == moe_exec_regime::ROUTED_ONLY && !detail::moe_stream_id_listed(ids, n_ids, e.expert_id)) {
            continue;
        }
        moe_stream_item item{};
        item.layer     = e.layer_id;
        item.expert_id = e.expert_id;
        item.role      = e.expert_role;
        item.dst_bytes = e.dst_size;
        item.src_bytes = e.src_size;
        out.push_back(item);
    }
    return out;
}

// True iff every distinct id in ids that is host-planned in `layer` has gate, up and down in `staged`.
static inline bool moe_stream_layer_fully_staged(const std::vector<moe_stream_item> & staged,
                                                 const placement_plan &               plan,
                                                 int                                  device,
                                                 int                                  layer,
                                                 const int32_t *                      ids,
                                                 size_t                               n_ids) {
    const std::vector<moe_stream_item> want =
        moe_stream_plan_build(plan, device, layer, moe_exec_regime::ROUTED_ONLY, ids, n_ids);
    for (const moe_stream_item & w : want) {
        bool found = false;
        for (const moe_stream_item & s : staged) {
            if (s.layer == layer && s.expert_id == w.expert_id && s.role == w.role) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

class moe_stream_ring;  // Task 6 defines it; held by pointer until then

struct moe_stream_staged {
    mem_handle  soa;
    mem_handle  aos;
    sycl::event ready;
};

class moe_stream_phase {  // one per device, owned by the backend context's extra_gpu
  public:
    moe_stream_phase();
    ~moe_stream_phase();

    void                      begin_ubatch(uint32_t n_tokens);
    // Called at layer `layer`'s first MoE op; weighs and enqueues layer+1 (and layer 0 itself at layer 0).
    moe_exec_decision         begin_layer(int                    layer,
                                          const placement_plan & plan,
                                          int                    device,
                                          unified_cache &        cache,
                                          sycl::queue &          ccs,
                                          const int32_t *        ids,
                                          size_t                 n_ids);
    const moe_stream_staged * find(int layer, int expert_id, expert_tensor_role role) const;
    void                      layer_done(int layer, sycl::event down_done);  // ring.epoch_boundary
    void                      end_prompt();                                  // first decode graph: clear

  private:
    std::unique_ptr<moe_stream_ring>                ring_;
    moe_exec_rates                                  rates_;
    std::unordered_map<uint64_t, moe_stream_staged> table_;  // key = (layer << 32) | (expert << 8) | role
};

}  // namespace ggml_sycl
