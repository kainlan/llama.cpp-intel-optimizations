#pragma once

// What a prompt layer streams through the VRAM ring (llama.cpp-36j0, owner ruling 2026-10-19). The list is a pure
// function of the published plan, the regime and, for ROUTED_ONLY, the ids: no device, no queue. The enqueue that
// consumes it is moe_stream_phase (bodies in moe-stream-plan.cpp).

#include "moe-exec-cost.hpp"
#include "moe-exec-policy.hpp"
#include "moe-stream-ring.hpp"
#include "soa-reorder-types.hpp"
#include "unified-cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// The AOS -> SOA reorder of one matrix on `queue` (ggml-sycl.cpp), shared by the load-time fill and the prompt ring:
// one kernel set, so the ring's SOA slot holds exactly what a load-time SOA weight holds. It handles the types of
// ggml_sycl_soa_reorder_supported_type; the ring never hands it another (moe_stream_item_reorderable).
sycl::event ggml_sycl_reorder_weight_gpu(sycl::queue &                    queue,
                                         const uint8_t *                  aos_device,
                                         uint8_t *                        soa_device,
                                         ggml_type                        type,
                                         int64_t                          ncols,
                                         int64_t                          nrows,
                                         size_t                           src_bytes,
                                         const std::vector<sycl::event> & deps);

namespace ggml_sycl {

struct moe_stream_item {
    int                layer;
    int                expert_id;
    expert_tensor_role role;
    ggml_sycl_cache_id key;  // left default by moe_stream_plan_build: the entry carries no tensor to derive it from
    size_t             dst_bytes;  // the SOA size the device route wants
    size_t             src_bytes;  // the AOS size of the host copy
    ggml_type          type;       // from the plan entry (moe_stream_plan_index_build); moe_stream_plan_build leaves 0
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

// One staged matrix: the SOA slot the device route reads, the AOS slot the copy landed in, and the reorder's event,
// after which `soa` holds the matrix. `layout` is the layout the SOA slot really holds (ruling C, c-3bde): a ring slice
// inherits the ring's AOS tag, so Task 8's resolver reads the layout from here, never from the slice.
struct moe_stream_staged {
    mem_handle       soa;
    mem_handle       aos;
    sycl::event      ready;
    ggml_layout_mode layout = GGML_LAYOUT_SOA;
};

// ---- The prompt phase's pure parts (llama.cpp-xod7, plan Task 7b) ----------------------------------------------------
// moe_stream_phase consults these before it touches a queue; test-moe-stream-phase runs them on hand-built plans.

// The stage table's key: (layer << 32) | (expert << 8) | role. A role fits in 8 bits and an expert id in 24.
static inline uint64_t moe_stream_table_key(int layer, int expert_id, expert_tensor_role role) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(expert_id) & 0xffffffu) << 8) |
           static_cast<uint64_t>(static_cast<uint8_t>(role));
}

// The micro-batch's ids marked once (c-7soh), sized n_expert: entry e is 1 iff some id is e. An id outside
// [0, n_expert) is ignored, so a bad id cannot index past the map.
static inline std::vector<uint8_t> moe_stream_ids_bitmap(const int32_t * ids, size_t n_ids, int n_expert) {
    std::vector<uint8_t> bitmap(n_expert > 0 ? static_cast<size_t>(n_expert) : 0, 0);
    for (size_t i = 0; ids != nullptr && i < n_ids; ++i) {
        if (ids[i] >= 0 && ids[i] < n_expert) {
            bitmap[static_cast<size_t>(ids[i])] = 1;
        }
    }
    return bitmap;
}

// One layer's host expert entries, read in the one scan of the plan a micro-batch makes (c-7soh).
struct moe_stream_layer_info {
    std::vector<moe_stream_item>              items;     // host gate/up/down entries in plan.entries order, typed
    std::vector<std::pair<ggml_type, size_t>> host_mix;  // (type, AOS bytes) over every host expert entry
    size_t  host_bytes       = 0;  // AOS bytes of every host expert entry of the layer, fused ones included
    int     n_host_experts   = 0;  // distinct expert ids with a host entry
    int     n_untriplet      = 0;  // host expert entries the ring cannot stage (fused GATE_UP, CHUNK_*)
    size_t  expert_bytes_max = 0;  // the largest one expert's host AOS bytes, summed over its roles
    int64_t K                = 0;  // the largest ne[0] over the host expert entries
    int64_t N                = 0;  // the largest ne[1] over the host expert entries
};

// The plan as the phase reads it for one micro-batch, and the inputs of the ring's geometry (ruling B, c-3bde).
struct moe_stream_plan_index {
    uint32_t host_matrices_per_layer_max = 0;  // the most items one layer has
    size_t   matrix_bytes_max            = 0;  // max(src, dst) over every item: one slot holds either side
    size_t   resident_expert_bytes       = 0;  // dst bytes of the expert entries resident on the device

    // Only layers with a host expert entry: the rest are skipped, never weighed.
    std::unordered_map<int, moe_stream_layer_info> layers;
};

namespace detail {

static inline void moe_stream_mix_add(std::vector<std::pair<ggml_type, size_t>> & mix, ggml_type type, size_t bytes) {
    for (std::pair<ggml_type, size_t> & tb : mix) {
        if (tb.first == type) {
            tb.second += bytes;
            return;
        }
    }
    mix.push_back({ type, bytes });
}

static inline bool moe_stream_routed(const std::vector<uint8_t> & bitmap, int expert_id) {
    return expert_id >= 0 && static_cast<size_t>(expert_id) < bitmap.size() && bitmap[static_cast<size_t>(expert_id)];
}

static inline bool moe_stream_rate_measured(double rate) {
    return std::isfinite(rate) && rate > 0.0;
}

}  // namespace detail

// One scan of plan.entries. Host entries carry target_device -1 and serve any device (moe_stream_host_triplet_entry);
// `device` selects only the resident bytes that bound the ring's size. The items take type and shape from the entry,
// which the planner copied from the tensor inventory (a zeroed shape fails moe_stream_item_reorderable).
static inline moe_stream_plan_index moe_stream_plan_index_build(const placement_plan & plan, int device) {
    moe_stream_plan_index                idx;
    std::unordered_map<uint64_t, size_t> expert_bytes;  // host AOS bytes per (layer, expert), keyed with role UNKNOWN
    for (const placement_entry & e : plan.entries) {
        if (e.expert_id < 0 || e.layer_id < 0) {
            continue;
        }
        if (e.on_device) {
            if (e.target_device == device) {
                idx.resident_expert_bytes += e.dst_size;
            }
            continue;
        }
        moe_stream_layer_info & li = idx.layers[e.layer_id];
        li.host_bytes += e.src_size;
        li.K = std::max(li.K, e.ne[0]);
        li.N = std::max(li.N, e.ne[1]);
        detail::moe_stream_mix_add(li.host_mix, e.type, e.src_size);
        auto ins = expert_bytes.emplace(moe_stream_table_key(e.layer_id, e.expert_id, expert_tensor_role::UNKNOWN), 0);
        if (ins.second) {
            ++li.n_host_experts;
        }
        ins.first->second += e.src_size;
        li.expert_bytes_max = std::max(li.expert_bytes_max, ins.first->second);
        if (!detail::moe_expert_role_is_triplet(e.expert_role)) {
            ++li.n_untriplet;
            continue;
        }
        moe_stream_item item{};
        item.layer     = e.layer_id;
        item.expert_id = e.expert_id;
        item.role      = e.expert_role;
        item.dst_bytes = e.dst_size;
        item.src_bytes = e.src_size;
        item.type      = e.type;
        item.ncols     = e.ne[0];
        item.nrows     = e.ne[1];
        li.items.push_back(item);
        idx.matrix_bytes_max = std::max(idx.matrix_bytes_max, std::max(e.src_size, e.dst_size));
    }
    for (const auto & kv : idx.layers) {
        idx.host_matrices_per_layer_max =
            std::max(idx.host_matrices_per_layer_max, static_cast<uint32_t>(kv.second.items.size()));
    }
    return idx;
}

// The items a decision stages: every item for STREAM_ALL, the experts the micro-batch routes to for ROUTED_ONLY, none
// for any other regime. plan.entries order either way.
static inline std::vector<moe_stream_item> moe_stream_select(const moe_stream_layer_info & li,
                                                             moe_exec_regime               regime,
                                                             const std::vector<uint8_t> &  bitmap) {
    std::vector<moe_stream_item> out;
    if (regime != moe_exec_regime::STREAM_ALL && regime != moe_exec_regime::ROUTED_ONLY) {
        return out;
    }
    for (const moe_stream_item & item : li.items) {
        if (regime == moe_exec_regime::STREAM_ALL || detail::moe_stream_routed(bitmap, item.expert_id)) {
            out.push_back(item);
        }
    }
    return out;
}

// True iff every host gate/up/down entry of the layer that the micro-batch routes to has a stage-table entry: the
// table answers, not a rebuilt list (c-7soh). A layer with a host entry the ring cannot stage (fused GATE_UP, CHUNK_*),
// or with host experts and no item, answers false: Task 10 relies on it. No info (the layer has no host entry) answers
// true. `staged` is the phase's table, or any set of its keys.
template <typename staged_keys>
static inline bool moe_stream_fully_staged(const moe_stream_layer_info * li,
                                           const std::vector<uint8_t> &  bitmap,
                                           const staged_keys &           staged) {
    if (li == nullptr) {
        return true;
    }
    if (li->n_untriplet > 0 || li->items.empty()) {
        return false;
    }
    for (const moe_stream_item & item : li->items) {
        if (detail::moe_stream_routed(bitmap, item.expert_id) &&
            staged.find(moe_stream_table_key(item.layer, item.expert_id, item.role)) == staged.end()) {
            return false;
        }
    }
    return true;
}

// True iff the reorder takes `item` into one ring slot of `slot_bytes`: its type has an AOS->SOA reorder (the one
// table, soa-reorder-types.hpp), its rows are whole blocks, and src_bytes is exactly its rows' bytes, which is also what
// the reorder writes. Anything else (an IQ type, a zeroed inventory shape) is not streamable in v1: counted, never
// copied as AOS.
static inline bool moe_stream_item_reorderable(const moe_stream_item & item, size_t slot_bytes) {
    if (!ggml_sycl_soa_reorder_supported_type(item.type) || item.ncols <= 0 || item.nrows <= 0 ||
        item.ncols % ggml_blck_size(item.type) != 0) {
        return false;
    }
    const size_t row = ggml_row_size(item.type, item.ncols);
    return row != 0 && static_cast<size_t>(item.nrows) <= SIZE_MAX / row &&
           row * static_cast<size_t>(item.nrows) == item.src_bytes &&
           std::max(item.src_bytes, item.dst_bytes) <= slot_bytes;
}

// True iff a lease's retained owner lets the ring copy from it directly (9y0p M4): a host-pinned WEIGHT allocation in
// one of this device's host zones, the shape mem-ops.cpp copies without a host bounce (a host zone of another device
// lives in another context). Any other lease would take the staged path, which the ring never does.
static inline bool moe_stream_owner_streamable(const alloc_metadata & owner, int device) {
    return owner.tier == alloc_tier::HOST_PINNED && owner.role == alloc_role::WEIGHT &&
           owner.host_zone != host_zone_id::COUNT && owner.device == device;
}

// A layer's rates: the per-type seeds over its host mix, then the device's tuning-cache record where it measured (a
// value is taken only when finite and positive), then the environment overrides.
static inline moe_exec_rates moe_stream_rates_assemble(const std::vector<std::pair<ggml_type, size_t>> & host_mix,
                                                       const ggml_sycl_moe_exec_rates_record *           record,
                                                       moe_exec_getenv_fn                                getenv_fn) {
    moe_exec_rates r = moe_exec_rates_seed(host_mix);
    if (record != nullptr) {
        // h2d: tuning-cache record, else seed; the T3 probe field is read here once it lands.
        if (detail::moe_stream_rate_measured(record->h2d_gbps)) {
            r.h2d_gbps = record->h2d_gbps;
        }
        if (detail::moe_stream_rate_measured(record->reorder_gbps)) {
            r.reorder_gbps = record->reorder_gbps;
        }
        // The device GEMM rate is the byte-weighted harmonic mean over the mix, as the seed's is, of each type's
        // recorded rate, else its seed.
        double bytes = 0.0;
        double us    = 0.0;
        for (const std::pair<ggml_type, size_t> & tb : host_mix) {
            double rate = moe_exec_dev_gflops_seed(tb.first);
            for (uint32_t i = 0; i < record->n_types && i < static_cast<uint32_t>(GGML_TYPE_COUNT); ++i) {
                if (record->types[i] == static_cast<int32_t>(tb.first) &&
                    detail::moe_stream_rate_measured(record->dev_gflops[i])) {
                    rate = record->dev_gflops[i];
                }
            }
            bytes += static_cast<double>(tb.second);
            us += static_cast<double>(tb.second) / rate;
        }
        if (bytes > 0.0) {
            r.dev_gflops = bytes / us;
        }
    }
    moe_exec_rates_apply_env(r, getenv_fn);
    return r;
}

static inline const char * moe_stream_regime_name(moe_exec_regime r) {
    switch (r) {
        case moe_exec_regime::STREAM_ALL:
            return "stream-all";
        case moe_exec_regime::ROUTED_ONLY:
            return "routed-only";
        default:
            return "none";
    }
}

// What begin_layer needs beyond its plan, cache and queue: the op's shape for the cost model, the backend's one policy
// and decision log (ggml_sycl_moe_exec_config(), ggml_sycl_moe_exec_log()), and two answers only the graph has, from
// the backend's MUL_MAT_ID scan: whether a layer's ops are in the graph being computed, and the host AOS lease of one
// of its experts (keyed from that layer's weight tensor, as the hybrid path's host route keys it).
struct moe_stream_op {
    uint32_t                n_tokens                   = 0;  // ids->ne[1]
    int                     n_expert                   = 0;  // src0->ne[2]
    int                     n_expert_used              = 0;  // ids->ne[0]
    uint64_t                graph_seq                  = 0;  // the context's graph_compute_seq
    moe_exec_policy         policy                     = moe_exec_policy::AUTO;
    moe_exec_decision_log * log                        = nullptr;
    void *                  graph_ctx                  = nullptr;
    bool (*layer_present)(void * graph_ctx, int layer) = nullptr;
    // The host lease of (layer, expert, role) and its bytes, or an invalid handle.
    mem_handle (*host_lease)(void * graph_ctx, int layer, int expert_id, expert_tensor_role role, size_t * bytes) =
        nullptr;
};

// The prompt phase of one device context (ggml_backend_sycl_context::moe_stream): the VRAM ring, the stage table and
// the current micro-batch's per-layer decisions.
//
// Ordering (ruling A, c-3bde): the device orders everything, the host never waits or asks. A copy waits for its ring
// region's gate, a reorder for its copy, and a region closes on an event after its last reorder. The SOA slots of a
// reused region are overwritten by reorders submitted to the same in-order compute queue after that region's readers,
// so the ring is offered only on an in-order queue and its readers (Task 8) must run on that queue.
//
// Lifetime (ruling D, c-fjx1): the ring's bytes go back only after the device's last use of them, on every exit path
// (end_prompt, a change of geometry, the destructor).
class moe_stream_phase {
  public:
    moe_stream_phase();
    ~moe_stream_phase();

    // A new micro-batch: the table and the decisions start over, and a refused ring reserve may be tried again; the
    // ring and its regions carry on.
    void                      begin_ubatch(uint32_t n_tokens);
    // Called at every prompt MUL_MAT_ID of `layer`; the first one per micro-batch decides. It stages the layer itself
    // when the layer was not weighed a layer ahead (layer 0 of the micro-batch: the one exposed copy) or its decision
    // is ROUTED_ONLY (which needs its own ids), then weighs layer + 1 and stages it now when that decision is a
    // STREAM_ALL ring. A layer below the last one, or the same layer in a new graph, starts a micro-batch. Returns the
    // layer's decision.
    moe_exec_decision         begin_layer(int                    layer,
                                          const placement_plan & plan,
                                          int                    device,
                                          unified_cache &        cache,
                                          sycl::queue &          ccs,
                                          const int32_t *        ids,
                                          size_t                 n_ids,
                                          const moe_stream_op &  op);
    const moe_stream_staged * find(int layer, int expert_id, expert_tensor_role role) const;
    // True iff every host expert of `layer` that ids route to is staged (moe_stream_fully_staged over the table).
    bool                      fully_staged(int layer, const int32_t * ids, size_t n_ids, int n_expert) const;
    // True iff slots were taken from the ring's current region since it last flipped.
    bool                      region_open() const;
    // After `layer`'s down op: an open region closes on `down_done`, a device event that ends the layer.
    void                      layer_done(int layer, sycl::event down_done);
    // The first decode graph after a prompt: the table, the decisions and the ring go.
    void                      end_prompt();

  private:
    struct layer_state {
        moe_exec_decision decision;
        double            window_us = 0.0;
        bool              decided   = false;
        bool              resolved  = false;  // staged when the decision is RING, logged either way
    };

    struct stage_counts {
        int    staged    = 0;  // experts with every selected matrix staged
        int    wanted    = 0;  // experts selected
        int    unowned   = 0;  // matrices skipped: no lease, or a lease the ring cannot copy directly
        int    untyped   = 0;  // matrices skipped: no AOS->SOA reorder for the type, or a shape that does not fit
        int    exhausted = 0;  // matrices skipped: no ring slot
        size_t bytes     = 0;  // AOS bytes copied
        bool   ring_unavailable = false;  // the ring could not be reserved
    };

    void              prepare(const placement_plan & plan, int device, sycl::queue & ccs);
    size_t            ring_bytes_free() const;
    moe_exec_decision weigh(const moe_stream_layer_info & li, const moe_stream_op & op, double window_us) const;
    moe_exec_decision decision_of(int layer) const;
    void              resolve(int                           layer,
                              const moe_stream_layer_info & li,
                              layer_state &                 st,
                              unified_cache &               cache,
                              const moe_stream_op &         op);
    stage_counts      stage(const moe_stream_layer_info & li,
                            moe_exec_regime               regime,
                            unified_cache &               cache,
                            const moe_stream_op &         op);
    bool              ring_ready();
    void              close_region(sycl::event region_done);
    sycl::event       compute_barrier();
    void              teardown(sycl::event last_use);
    void              report(int layer, const layer_state & st, const stage_counts & n, const moe_stream_op & op) const;

    std::unique_ptr<moe_stream_ring>                ring_;
    std::unordered_map<uint64_t, moe_stream_staged> table_;           // key = moe_stream_table_key(layer, expert, role)
    std::vector<uint64_t>                           region_keys_[2];  // the table keys whose slots each region holds
    int                                             cur_region_       = 0;      // flips with every epoch_boundary
    bool                                            region_submitted_ = false;  // a reorder went into it since
    sycl::event                                     region_last_reorder_;       // the last of them
    mem_handle                                      anchor_;         // one slot: any slice holds the whole reservation
    std::chrono::steady_clock::time_point           last_layer_tp_;  // the previous layer's first op, this micro-batch
    moe_stream_plan_index                           index_;
    std::unordered_map<int, layer_state>            layers_;
    std::vector<uint8_t>                            bitmap_;  // the current op's ids, marked
    std::optional<sycl::queue>                      ccs_;     // the compute queue the reorders run on
    moe_stream_ring_geometry                        geo_;
    ggml_sycl_moe_exec_rates_record                 record_              = {};
    int                                             device_              = -1;
    int                                             cur_layer_           = -1;
    uint64_t                                        cur_graph_seq_       = 0;
    uint32_t                                        n_tokens_            = 0;
    uint32_t                                        ubatch_seq_          = 0;
    bool                                            ubatch_live_         = false;
    bool                                            prepared_            = false;
    bool                                            in_order_            = false;
    bool                                            reserve_failed_      = false;
    bool                                            record_valid_        = false;
    bool                                            record_looked_up_    = false;
    bool                                            warned_not_in_order_ = false;
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
