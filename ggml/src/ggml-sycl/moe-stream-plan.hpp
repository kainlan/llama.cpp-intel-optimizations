#pragma once

// What a prompt layer streams through the VRAM ring (llama.cpp-36j0, owner ruling 2026-10-19). The list is a pure
// function of the published plan, the regime and, for ROUTED_ONLY, the ids: no device, no queue. The enqueue that
// consumes it is moe_stream_phase (bodies in moe-stream-plan.cpp).

#include "moe-exec-cost.hpp"
#include "unified-cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

static inline bool moe_expert_role_is_triplet(expert_tensor_role r) {
    return r == expert_tensor_role::GATE || r == expert_tensor_role::UP || r == expert_tensor_role::DOWN;
}

// Fused GATE_UP and CHUNK_* host entries are deliberately NOT listed: the stream list is gate/up/down triplets, the only
// shape the ring stages. An empty list therefore does not mean "staged" for a fused-gate_up or chunked model.
// A gate/up/down entry of `layer` that is planned for the host. Host entries carry target_device -1 (the planner sets
// it at placement time), so they serve any device and the device argument of the callers is not a filter.
static inline bool moe_stream_host_triplet_entry(const placement_entry & e, int layer) {
    if (e.expert_id < 0 || e.layer_id != layer || e.on_device) {
        return false;
    }
    return moe_expert_role_is_triplet(e.expert_role);
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

// Only gate/up/down triplets are considered (see moe_stream_host_triplet_entry): a layer whose host experts are fused
// GATE_UP or CHUNK_* entries has an empty want list and so answers true without anything staged. Callers must not use
// this predicate to skip the CPU path for a fused-gate_up or chunked model.
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

// ---- Loan selection (llama.cpp-0zmf, plan Task 11a): which resident expert triplets a loan may take. -------------------
// Pure over the plan, like the stream list above: no device, no queue. The plan's build_index() must have run.

namespace detail {

// True iff a SOA-to-AOS inverse of ggml_sycl_reorder_weight_gpu exists for `type`. None exists yet: that function only
// runs AOS to SOA, so v1 lends AOS-layout entries only. Add a case here when an inverse kernel lands.
static inline bool moe_loan_soa_inverse_exists(ggml_type type) {
    (void) type;
    return false;
}

// AOS, or SOA with an inverse. The SOA arm stays dead until moe_loan_soa_inverse_exists has a case; no other layout has
// an inverse at all.
static inline bool moe_loan_layout_lendable(const placement_entry & e) {
    return e.layout == GGML_LAYOUT_AOS || (e.layout == GGML_LAYOUT_SOA && moe_loan_soa_inverse_exists(e.type));
}

// True iff the tensor `name` is bulk-staged on `device` at load, mirroring the per-expert loop of
// try_bulk_stage_moe_tensor (ggml-sycl.cpp) over the plan. Every expert must stage on `device` as one of:
//   (a) primary there (on_device and target_device == device) with no alternate resolving to `device`, or
//   (b) not primary there and carrying exactly one alternate that resolves to `device` (the "stage-alt-bulk" branch).
// An expert that is neither, and n_experts <= 1, mean the load path does not bulk-stage. A bulk tensor is views into ONE
// WEIGHT allocation, so no single expert can be handed back. The load path also compares layouts (after
// ggml_sycl_adjust_layout_for_tensor, which the plan cannot see); that comparison is not mirrored, so a disagreement
// still reads as bulk here: the fail-safe over-skip. Checks that need runtime state (tensor->data,
// GGML_SYCL_MOE_BULK_STAGE=0) are likewise not mirrored; each can only make the load path not bulk where this says bulk,
// which skips more, never less.
static inline bool moe_loan_tensor_bulk_staged(const placement_plan & plan, const std::string & name, int device) {
    int n_experts = 0;
    for (const placement_entry & e : plan.entries) {
        if (e.expert_id < 0 || e.name != name) {
            continue;
        }
        int n_alt = 0;
        for (const placement_alternate_layout & alt : e.alternate_layouts) {
            const int alt_target = alt.target_device >= 0 ? alt.target_device : e.target_device;
            if (alt_target == device) {
                ++n_alt;
            }
        }
        if (e.on_device && e.target_device == device) {
            if (n_alt != 0) {
                return false;
            }
        } else if (n_alt != 1) {
            return false;
        }
        ++n_experts;
    }
    return n_experts > 1;
}

// gate, up, down map to slots 0, 1, 2.
static inline int moe_loan_role_slot(expert_tensor_role r) {
    return static_cast<int>(r) - 1;
}

static inline uint64_t moe_loan_triplet_key(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) | static_cast<uint32_t>(expert);
}

// The tensor name is fixed by (layer, role), so (layer, role, device) names one bulk answer.
static inline uint64_t moe_loan_bulk_memo_key(const placement_entry & e) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(e.layer_id)) << 16) |
           (static_cast<uint64_t>(moe_loan_role_slot(e.expert_role)) << 8) |
           static_cast<uint64_t>(e.target_device & 0xff);
}

// The three entries of (layer, expert), by role slot; null when one is missing.
struct moe_loan_triplet {
    const placement_entry * e[3] = { nullptr, nullptr, nullptr };

    bool complete() const { return e[0] && e[1] && e[2]; }
};

static inline moe_loan_triplet moe_loan_triplet_of(const placement_plan & plan, int layer, int expert) {
    moe_loan_triplet t;
    for (int r = 0; r < 3; ++r) {
        t.e[r] = plan.find_expert_entry(layer, expert, static_cast<expert_tensor_role>(r + 1));
    }
    return t;
}

static inline bool moe_loan_triplet_is_lendable(const moe_loan_triplet &             t,
                                                const placement_plan &               plan,
                                                std::unordered_map<uint64_t, bool> & bulk_memo) {
    if (!t.complete()) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        const placement_entry & e = *t.e[i];
        if (!e.on_device || !moe_loan_layout_lendable(e)) {
            return false;
        }
        const uint64_t key = moe_loan_bulk_memo_key(e);
        auto           it  = bulk_memo.find(key);
        if (it == bulk_memo.end()) {
            it = bulk_memo.emplace(key, moe_loan_tensor_bulk_staged(plan, e.name, e.target_device)).first;
        }
        if (it->second) {
            return false;
        }
    }
    return true;
}

}  // namespace detail

// True iff the gate/up/down triplet of (layer, expert) can be lent: all three entries exist and are on a device, none is
// bulk-staged, and each is AOS or has a SOA-to-AOS inverse (v1: AOS only, see moe_loan_soa_inverse_exists).
// Precondition: plan.build_index() has run; find_expert_entry returns nullptr on an un-indexed plan, so selection then
// returns nothing (fail-safe, no signal).
static inline bool moe_expert_triplet_lendable(const placement_plan & plan, int layer, int expert) {
    std::unordered_map<uint64_t, bool> memo;
    return detail::moe_loan_triplet_is_lendable(detail::moe_loan_triplet_of(plan, layer, expert), plan, memo);
}

// The lendable tail of the resident set on `device`: whole (gate, up, down) triplets, walking plan.entries in REVERSE
// (pack-group order, popularity rank when ranks exist, so the tail is the least popular), until the summed
// vram_charge_size reaches bytes_wanted or nothing lendable is left. A triplet is never split. Keys come out in the
// order the triplets are met, gate/up/down within each. *bytes_selected (when non-null) receives the sum.
// Precondition: plan.build_index() has run; find_expert_entry returns nullptr on an un-indexed plan, so selection then
// returns nothing (fail-safe, no signal).
static inline std::vector<expert_placement_key> moe_expert_loan_select(const placement_plan & plan,
                                                                       int                    device,
                                                                       size_t                 bytes_wanted,
                                                                       size_t *               bytes_selected) {
    std::vector<expert_placement_key>  out;
    size_t                             sum = 0;
    std::unordered_map<uint64_t, bool> bulk_memo;
    std::unordered_set<uint64_t>       visited;
    for (size_t i = plan.entries.size(); i-- > 0 && sum < bytes_wanted;) {
        const placement_entry & e = plan.entries[i];
        if (e.expert_id < 0 || !detail::moe_expert_role_is_triplet(e.expert_role) || !e.on_device ||
            e.target_device != device) {
            continue;
        }
        if (!visited.insert(detail::moe_loan_triplet_key(e.layer_id, e.expert_id)).second) {
            continue;
        }
        const detail::moe_loan_triplet t = detail::moe_loan_triplet_of(plan, e.layer_id, e.expert_id);
        if (!t.complete() || t.e[0]->target_device != device || t.e[1]->target_device != device ||
            t.e[2]->target_device != device || !detail::moe_loan_triplet_is_lendable(t, plan, bulk_memo)) {
            continue;
        }
        for (int r = 0; r < 3; ++r) {
            out.push_back({ e.layer_id, e.expert_id, static_cast<expert_tensor_role>(r + 1) });
            sum += t.e[r]->vram_charge_size;
        }
    }
    if (bytes_selected) {
        *bytes_selected = sum;
    }
    return out;
}

}  // namespace ggml_sycl
