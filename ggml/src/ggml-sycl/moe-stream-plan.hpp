#pragma once

// What a prompt layer streams through the VRAM ring (llama.cpp-36j0, owner ruling 2026-10-19). The ring serves the
// batched prompt executor only, one executor window at a time (W1, W2 of docs/plans/2026-10-24-prefill-window-ring.md):
// a window's items are a pure function of the published plan, the executor's claims and the phase's experts, with no
// device and no queue. The enqueue that consumes them is moe_stream_phase (bodies in moe-stream-plan.cpp).

#include "moe-exec-cost.hpp"
#include "moe-exec-policy.hpp"
#include "moe-stream-ring.hpp"
#include "unified-cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ggml_sycl {

// How the ring stages one matrix (W2 of docs/plans/2026-10-24-prefill-window-ring.md). AOS_ONLY: one slot the copy
// lands in, read in place by the batched prompt executor's dequant kernel, for a weight the executor claims. NONE: no
// slot, and no window stages it. The ring serves the executor only, so there is no SOA staging: no reorder into a
// second slot for a prompt kernel.
enum class moe_stream_staging : uint8_t {
    NONE,
    AOS_ONLY,
};

// Who reads a staged matrix's slot (llama.cpp-3460), with a plain SYCL kernel on the in-order compute queue (c-las5);
// neither hands the slot to a library. EXECUTOR: the batched prompt executor (llama.cpp-348c), which dequantizes the
// AOS slot into its f16 weight slot and gives only that slot to oneDNN; the route leaves such an entry to the route
// core, as a handoff when the executor takes the layer whole. Every matrix the ring stages is the executor's (W2).
// ROUTE: the resolver's STAGED_RING route, served as MMVQ_COMPAT. The ring stages nothing for it: an item staging NONE
// keeps it, and the route's device test inserts such entries itself (test_insert_staged).
enum class moe_stream_consumer : uint8_t {
    ROUTE,
    EXECUTOR,
};

// True iff the device runs a prompt MUL_MAT_ID over a device-resident expert of `type` in `layout`, ne0 x ne1
// (c-hmey). The backend answers from ggml_sycl_moe_query_route_capability. The ring no longer asks it (W2): it stages
// only what the library executor claims. The index build keeps the parameter for its callers' one signature.
using moe_stream_prompt_kernel_fn =
    bool (*)(void * graph_ctx, ggml_type type, ggml_layout_mode layout, int64_t ne0, int64_t ne1);

// True iff the prompt's library executor (the oneDNN batched one) claims `layer`'s `role` weight once the ring has
// staged its host experts (llama.cpp-3460). The ring then stages each of that weight's host matrices into one AOS slot
// whose consumer is the executor, and the staged route refuses the entry by that consumer: the index build asks the
// claim once, and the entry carries its answer (one fact, one source). A weight it does not claim stages nothing.
using moe_stream_library_claims_fn = bool (*)(void * graph_ctx, int layer, expert_tensor_role role);

// Records a request to borrow `bytes` of resident expert VRAM on `device` at its next graph boundary (llama.cpp-3460,
// c-0tmn), counted in ring slots of `slot_bytes` (llama.cpp-9x8t): the loan must free that many whole slots, not that
// many bytes, since a matrix smaller than a slot frees a hole no slot fits in. 0 counts bytes. The backend answers with
// ggml_sycl_request_expert_loan: lock-free, so the phase may call it while g_sycl_graph_compute_mutex is held, and the
// largest request since the last boundary wins.
using moe_stream_request_loan_fn = void (*)(int device, size_t bytes, size_t slot_bytes);

// How many loans the boundary runner has refused on `device` (REFUSED, never BUSY, which it re-posts itself); it only
// grows. The backend answers with ggml_sycl_expert_loan_refusals.
using moe_stream_loan_refusals_fn = uint64_t (*)(int device);

struct moe_stream_item {
    int                 layer;
    int                 expert_id;
    expert_tensor_role  role;
    size_t              src_bytes;  // the AOS size of the host copy
    ggml_type           type;       // from the plan entry (moe_stream_plan_index_build)
    int64_t             ncols, nrows;
    // From moe_stream_plan_index_build: AOS_ONLY for a weight the library executor claims, else NONE.
    moe_stream_staging  staging  = moe_stream_staging::NONE;
    // From moe_stream_plan_index_build: EXECUTOR for a weight the library executor claims once staged.
    moe_stream_consumer consumer = moe_stream_consumer::ROUTE;
};

namespace detail {

static inline bool moe_expert_role_is_triplet(expert_tensor_role r) {
    return r == expert_tensor_role::GATE || r == expert_tensor_role::UP || r == expert_tensor_role::DOWN;
}

}  // namespace detail

// One staged matrix: the slot its reader reads, the event after which that slot holds the matrix, and `layout`, the
// layout the slot really holds (ruling C, c-3bde), which the resolver's STAGED_RING route reads from here. Each slot is
// acquired with the layout written there (lead ruling c-o9uw-1), so the slice's own tag agrees, and the route refuses a
// slot where it does not. The ring stages AOS slots for the batched executor only (W2): `soa` is empty, `ready` is the
// copy's event, `layout` is AOS and `consumer` is the item's, EXECUTOR (llama.cpp-3460), by which the route refuses the
// entry. The SOA fields stay: the resolver still picks an entry's slot by `layout`.
struct moe_stream_staged {
    mem_handle          soa;
    mem_handle          aos;
    sycl::event         ready;
    ggml_layout_mode    layout   = GGML_LAYOUT_SOA;
    moe_stream_consumer consumer = moe_stream_consumer::ROUTE;
};

// ---- The prompt phase's pure parts (llama.cpp-xod7, plan Task 7b) ----------------------------------------------------
// moe_stream_phase consults these before it touches a queue; test-moe-stream-phase runs them on hand-built plans.

// The stage table's key: (layer << 32) | (expert << 8) | role. A role fits in 8 bits and an expert id in 24.
static inline uint64_t moe_stream_table_key(int layer, int expert_id, expert_tensor_role role) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(expert_id) & 0xffffffu) << 8) |
           static_cast<uint64_t>(static_cast<uint8_t>(role));
}

namespace detail {

// The bytes of `item`'s rows when its type is a known one with a block size, its rows are whole blocks and src_bytes is
// exactly those bytes; 0 otherwise (a zeroed inventory shape, a retired type). The one way of staging (AOS_ONLY) copies
// src_bytes and the executor's dequant then reads ncols x nrows of the type from the slot, so a mismatch is never
// staged.
static inline size_t moe_stream_item_rows_bytes(const moe_stream_item & item) {
    if (static_cast<int>(item.type) < 0 || static_cast<int>(item.type) >= GGML_TYPE_COUNT ||
        ggml_blck_size(item.type) <= 0 || item.ncols <= 0 || item.nrows <= 0 ||
        item.ncols % ggml_blck_size(item.type) != 0) {
        return 0;
    }
    const size_t row = ggml_row_size(item.type, item.ncols);
    if (row == 0 || static_cast<size_t>(item.nrows) > SIZE_MAX / row ||
        row * static_cast<size_t>(item.nrows) != item.src_bytes) {
        return 0;
    }
    return item.src_bytes;
}

}  // namespace detail

// True iff one ring slot of `slot_bytes` holds `item` as AOS, to be read in place (c-hmey): its rows are whole blocks,
// src_bytes is exactly its rows' bytes, and src_bytes fits. No SOA side is written.
static inline bool moe_stream_item_aos_fits(const moe_stream_item & item, size_t slot_bytes) {
    return detail::moe_stream_item_rows_bytes(item) != 0 && item.src_bytes <= slot_bytes;
}

// The slots one item takes in a region: one staged AOS-only, none otherwise.
static inline uint32_t moe_stream_staging_slots(moe_stream_staging staging) {
    return staging == moe_stream_staging::AOS_ONLY ? 1u : 0u;
}

// One layer's host expert entries, read in the one scan of the plan a micro-batch makes (c-7soh).
struct moe_stream_layer_info {
    std::vector<moe_stream_item>              items;     // host gate/up/down entries in plan.entries order, typed
    std::vector<std::pair<ggml_type, size_t>> host_mix;  // (type, AOS bytes) over every host expert entry
    size_t  host_bytes       = 0;  // AOS bytes of every host expert entry of the layer, fused ones included
    int     n_host_experts   = 0;  // distinct expert ids with a host entry
    int     n_untriplet      = 0;  // host expert entries the ring cannot stage (fused GATE_UP, CHUNK_*)
    int     n_streamable     = 0;  // items the ring gives slots: staging AOS_ONLY
    int     region_slots     = 0;  // the slots its streamable items take: one each
    size_t  soa_bytes        = 0;  // always 0: the ring reorders nothing (W2); the cost model's reorder input
    size_t  matrix_bytes_max = 0;  // the largest one streamable matrix's slot bytes: the unit the ring stages
    int64_t K                = 0;  // the largest ne[0] over the host expert entries
    int64_t N                = 0;  // the largest ne[1] over the host expert entries
    // Streamable items per role, GATE, UP, DOWN (detail::moe_stream_role_slot): one role is what an executor window
    // reads from, so the most items one role has bounds the window.
    int     role_items[3]    = { 0, 0, 0 };
};

// The plan as the phase reads it for one micro-batch, and the inputs of the ring's geometry (ruling B, c-3bde).
struct moe_stream_plan_index {
    uint32_t host_matrices_per_layer_max = 0;  // the most streamable items one layer has
    // The most slots one layer's streamable items take, one each. The ring is sized from window_slots, not from this;
    // the ring-unavailable WARN prints both.
    uint32_t region_slots_max            = 0;
    // The largest streamable item's src bytes: the AOS bytes one slot holds.
    size_t   matrix_bytes_max            = 0;
    size_t   resident_expert_bytes       = 0;  // dst bytes of the expert entries resident on the device
    uint32_t role_items_max              = 0;  // the most streamable items one role of one layer has
    uint32_t window_experts              = 0;  // G_plan, the experts one phase holds (moe_stream_window_experts)
    uint32_t window_slots                = 0;  // min(role_items_max, G_plan): a region's size, one executor window

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

static inline bool moe_stream_rate_measured(double rate) {
    return std::isfinite(rate) && rate > 0.0;
}

// The index of a gate/up/down role in moe_stream_layer_info::role_items. Only triplet items reach it.
static inline int moe_stream_role_slot(expert_tensor_role role) {
    switch (role) {
        case expert_tensor_role::GATE:
            return 0;
        case expert_tensor_role::UP:
            return 1;
        case expert_tensor_role::DOWN:
            return 2;
        default:
            GGML_ABORT("moe_stream_role_slot: role %d is not gate, up or down", static_cast<int>(role));
    }
}

}  // namespace detail

// What the ring's sizing needs from the batched executor (W3 of docs/plans/2026-10-24-prefill-window-ring.md): the
// planned f16 weight slot it dequantizes a phase into, and the group cap it honours. A phase holds
// G_plan = fit >= 2 ? fit / 2 : fit experts (fit = slot / one expert's f16 bytes), the executor's own two-buffer rule
// (moe_pp_batched_f16_plan_groups), capped. 0 slot bytes (a dense model, or before the load publishes the inventory
// globals) means no window and no ring.
struct moe_stream_window_hints {
    size_t weight_slot_bytes = 0;  // unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(device)
    size_t group_cap         = 0;  // ggml_sycl_moe_pp_f16_group_experts_max(); 0 = none
    // Device bytes of the triplets lent to the ring on this device, counted back as resident so the cap is
    // loan-invariant.
    size_t lent_bytes        = 0;
};

// The experts one executor phase holds for an expert of `expert_f16_bytes`: G_plan above, 0 when the slot holds none.
static inline uint32_t moe_stream_window_experts(const moe_stream_window_hints & h, size_t expert_f16_bytes) {
    if (h.weight_slot_bytes == 0 || expert_f16_bytes == 0) {
        return 0;
    }
    const size_t fit = h.weight_slot_bytes / expert_f16_bytes;
    size_t       g   = fit >= 2 ? fit / 2 : fit;
    if (h.group_cap != 0 && g > h.group_cap) {
        g = h.group_cap;
    }
    return g > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(g);
}

// One scan of plan.entries. Host entries carry target_device -1 and serve any device;
// `device` selects only the resident bytes that bound the ring's size. The items take type and shape from the entry,
// which the planner copied from the tensor inventory (a zeroed shape stages NONE). The ring serves the batched prompt
// executor only (W2 of docs/plans/2026-10-24-prefill-window-ring.md): a matrix one AOS slot holds, whose weight the
// library executor claims once staged, stages AOS_ONLY for the executor, whose own tables are the authority there
// (llama.cpp-3460), and every other item stages NONE. The claim is asked once per (layer, role). The device's prompt
// kernels are not asked, since no prompt kernel reads the ring: `prompt_kernel` stays for the callers' one signature.
// `hints` gives the executor's phase, which with the streamable items per role decides window_slots, the region the
// ring is sized to. `as_host`, when given, names (layer, expert) triplets, keyed moe_stream_table_key(layer, expert,
// UNKNOWN), whose gate/up/down entries resident on `device` are read as host entries: the plan as a loan of those
// triplets would leave it (the transaction's update_expert_placement flips on_device and the target only), without
// copying the plan (moe_stream_loan_size_post_loan).
static inline moe_stream_plan_index moe_stream_plan_index_build(
    const placement_plan &               plan,
    int                                  device,
    moe_stream_prompt_kernel_fn          prompt_kernel,
    void *                               graph_ctx,
    moe_stream_library_claims_fn         library_claims,
    const moe_stream_window_hints &      hints,
    const std::unordered_set<uint64_t> * as_host = nullptr) {
    GGML_UNUSED(prompt_kernel);  // kept for the signature; unused since the window ring (W2)
    moe_stream_plan_index        idx;
    // The (layer, expert) keys with a host entry, keyed with role UNKNOWN: counts each layer's distinct host experts.
    std::unordered_set<uint64_t> host_experts;

    std::unordered_map<uint64_t, bool> claims;  // (layer, role) keyed with expert 0 -> the executor claims it staged
    // One expert's f16 bytes, the executor's dequant unit: one matrix, ne0 x ne1 x 2 (the executor's f16_expert_bytes,
    // fed to moe_pp_batched_f16_plan_groups), the largest over the streamable items. Never the product of a layer's
    // per-dimension maxima: gate/up and down are transposed shapes, and that product is not one matrix.
    size_t                             expert_f16_bytes = 0;
    for (const placement_entry & e : plan.entries) {
        if (e.expert_id < 0 || e.layer_id < 0) {
            continue;
        }
        const bool lent =
            as_host != nullptr && e.on_device && e.target_device == device &&
            detail::moe_expert_role_is_triplet(e.expert_role) &&
            as_host->count(moe_stream_table_key(e.layer_id, e.expert_id, expert_tensor_role::UNKNOWN)) != 0;
        if (e.on_device && !lent) {
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
        if (host_experts.insert(moe_stream_table_key(e.layer_id, e.expert_id, expert_tensor_role::UNKNOWN)).second) {
            ++li.n_host_experts;
        }
        if (!detail::moe_expert_role_is_triplet(e.expert_role)) {
            ++li.n_untriplet;
            continue;
        }
        moe_stream_item item{};
        item.layer     = e.layer_id;
        item.expert_id = e.expert_id;
        item.role      = e.expert_role;
        item.src_bytes = e.src_size;
        item.type      = e.type;
        item.ncols     = e.ne[0];
        item.nrows     = e.ne[1];
        // A weight the library executor claims once staged is read by the executor's dequant kernel from one AOS
        // slot (llama.cpp-3460, resolving llama.cpp-v7np): no reorder and no second slot, and the route refuses the
        // entry by its consumer. Any other matrix stages NONE (W2), and a matrix no AOS slot can hold does not ask
        // for its weight.
        if (library_claims != nullptr && moe_stream_item_aos_fits(item, SIZE_MAX)) {
            const uint64_t ck = moe_stream_table_key(item.layer, 0, item.role);
            auto           c  = claims.find(ck);
            if (c == claims.end()) {
                c = claims.emplace(ck, library_claims(graph_ctx, item.layer, item.role)).first;
            }
            if (c->second) {
                item.staging  = moe_stream_staging::AOS_ONLY;
                item.consumer = moe_stream_consumer::EXECUTOR;
            }
        }
        li.items.push_back(item);
        // The ring is sized over the matrices it stages. Another one is still listed, so the layer's claim can see it
        // (moe_stream_layer_claimed), but it takes no slot and no window stages it.
        if (item.staging == moe_stream_staging::NONE) {
            continue;
        }
        ++li.n_streamable;
        li.region_slots += static_cast<int>(moe_stream_staging_slots(item.staging));
        li.role_items[detail::moe_stream_role_slot(item.role)] += 1;
        li.matrix_bytes_max  = std::max(li.matrix_bytes_max, e.src_size);
        idx.matrix_bytes_max = std::max(idx.matrix_bytes_max, e.src_size);
        // A streamable item has a non-zero shape (a zeroed one stages NONE), so this is never 0 where a window exists.
        expert_f16_bytes =
            std::max(expert_f16_bytes, static_cast<size_t>(item.ncols) * static_cast<size_t>(item.nrows) * 2);
    }
    for (const auto & kv : idx.layers) {
        idx.host_matrices_per_layer_max =
            std::max(idx.host_matrices_per_layer_max, static_cast<uint32_t>(kv.second.n_streamable));
        idx.region_slots_max = std::max(idx.region_slots_max, static_cast<uint32_t>(kv.second.region_slots));
        for (int r = 0; r < 3; ++r) {
            idx.role_items_max = std::max(idx.role_items_max, static_cast<uint32_t>(kv.second.role_items[r]));
        }
    }
    idx.window_experts = moe_stream_window_experts(hints, expert_f16_bytes);
    idx.window_slots   = std::min(idx.role_items_max, idx.window_experts);
    return idx;
}

// The ring geometry for `plan` on `device`: the one scan of the plan and the sizing over it. The stream phase sizes its
// ring with it (moe_stream_phase::prepare) and the auto-ubatch fit sizes the ring it charges with it
// (ggml_sycl_hold_ring_uncovered), so the two cannot disagree: what sizes the ring (the staging hooks, the index's
// window) changes here once. `index_out`, when given, receives the index the geometry was sized over. The ring is
// sized to one executor window per region, window_slots slots of the largest streamable matrix, not to a whole layer:
// the executor reads a layer a phase at a time, and the 10% bound refused every whole-layer ring on this host
// (llama.cpp-bcre).
// A plan with no host expert has no layer and no region slot, so the sizing answers the default geometry. A null
// `library_claims` claims nothing, so every item stages NONE and the geometry is the default too: no ring. The
// auto-ubatch fit, a caller with no live graph scan, passes ggml_sycl_moe_stream_claims_every_weight, claiming every
// weight, so the ring it charges bounds the phase's from above, but for the two edges ggml_sycl_hold_ring_uncovered
// names. `as_host` sizes the plan a loan of those triplets would leave (moe_stream_plan_index_build); the caller then
// counts their device bytes in hints.lent_bytes, as the published ledger counts a loan that ran.
static inline moe_stream_ring_geometry moe_stream_plan_ring_geometry(
    const placement_plan &               plan,
    int                                  device,
    moe_stream_prompt_kernel_fn          prompt_kernel,
    void *                               graph_ctx,
    moe_stream_library_claims_fn         library_claims,
    const moe_stream_window_hints &      hints,
    moe_stream_plan_index *              index_out = nullptr,
    const std::unordered_set<uint64_t> * as_host   = nullptr) {
    moe_stream_plan_index index =
        moe_stream_plan_index_build(plan, device, prompt_kernel, graph_ctx, library_claims, hints, as_host);
    // The cap reads the device's expert set with the ring's own loan counted back (llama.cpp-7999): a loan moves
    // hints.lent_bytes out of index.resident_expert_bytes, so the sum is the same before and after it. The add
    // saturates defensively: a real sum of device bytes cannot approach SIZE_MAX, so the clamp only keeps a corrupt
    // or sentinel lent_bytes from wrapping into a refusal.
    size_t resident_with_lent = index.resident_expert_bytes + hints.lent_bytes;
    if (resident_with_lent < index.resident_expert_bytes) {
        resident_with_lent = SIZE_MAX;
    }
    const moe_stream_ring_geometry geo =
        moe_stream_ring_size(index.window_slots, index.matrix_bytes_max, resident_with_lent);
    if (index_out != nullptr) {
        *index_out = std::move(index);
    }
    return geo;
}

// True iff two geometries are one ring: the fields prepare() compares before it keeps a ring, and the loan sizing
// compares to tell a converged loan (moe_stream_loan_size_post_loan).
static inline bool moe_stream_ring_geometry_same(const moe_stream_ring_geometry & a,
                                                 const moe_stream_ring_geometry & b) {
    return a.slot_bytes == b.slot_bytes && a.n_slots == b.n_slots && a.bytes == b.bytes &&
           a.region_slots == b.region_slots;
}

// True iff the batched executor takes every host matrix of the layer from the ring (W2): the layer has items, each one
// staged AOS_ONLY for the executor (the index build's record of the claim), and no host entry is one the ring cannot
// stage (fused GATE_UP, CHUNK_*). The geometry-free half of moe_stream_executor_claims, which the ring-unavailable rule
// asks first.
static inline bool moe_stream_layer_claimed(const moe_stream_layer_info & li) {
    if (li.n_untriplet != 0 || li.items.empty()) {
        return false;
    }
    for (const moe_stream_item & item : li.items) {
        if (item.staging == moe_stream_staging::NONE || item.consumer != moe_stream_consumer::EXECUTOR) {
            return false;
        }
    }
    return true;
}

// True iff GGML_SYCL_MOE_EXEC_DECLINE_LAYER (`knob`, -1 unset) names `layer`: begin_layer then turns the layer's
// decision to CPU with reason DECLINED, so executor_claims answers false and the layer stays hybrid (llama.cpp-ny6i).
// The claim itself (moe_stream_executor_claims) does not read the knob.
static inline bool moe_stream_layer_declined(int layer, int knob) {
    return knob >= 0 && knob == layer;
}

// True iff the batched executor takes the layer's weights whole from the ring: the layer is claimed
// (moe_stream_layer_claimed) and a ring of `geo` can exist. Anything less keeps the whole layer hybrid: the ring never
// serves some of a layer's host matrices while the rest stay on the hybrid path.
static inline bool moe_stream_executor_claims(const moe_stream_layer_info & li, const moe_stream_ring_geometry & geo) {
    return geo.n_slots != 0 && moe_stream_layer_claimed(li);
}

// Why a layer answered RING_UNAVAILABLE, for the phase's once-per-prompt WARN (lead ruling c-kcyf): the first rule that
// holds, in this order. A layer the executor does not claim comes first, since the ring serves the executor only (W2).
// geo is what moe_stream_ring_size gave for idx; its all-zero answer past the index's first three rules is the cap,
// an eleventh of the resident plus lent expert bytes (llama.cpp-7999; its overflow guards refuse sizes far above any
// resident expert set too). A window of 0 slots is no planned executor slot (before the load publishes the inventory
// globals), or a slot that holds no expert. The last name is unreachable while the cost model's stage unit is one
// streamable matrix (moe_stream_layer_info::matrix_bytes_max), since a region of window_slots >= 1 slots holds the
// largest streamable matrix in each slot; it is there so that no ring-unavailable line goes unexplained.
static inline const char * moe_stream_ring_unavailable_rule(const moe_stream_plan_index &    idx,
                                                            const moe_stream_layer_info &    li,
                                                            const moe_stream_ring_geometry & geo,
                                                            bool                             in_order,
                                                            bool                             reserve_failed) {
    if (!moe_stream_layer_claimed(li)) {
        return "executor does not claim";
    }
    if (idx.host_matrices_per_layer_max == 0) {
        return "no host matrices of a streamable type";
    }
    if (idx.matrix_bytes_max == 0) {
        return "no matrix bytes";
    }
    if (idx.window_slots == 0) {
        return "no executor window";
    }
    if (geo.n_slots == 0) {
        return "ring exceeds an eleventh of resident plus lent expert bytes";
    }
    if (!in_order) {
        return "queue not in order";
    }
    if (reserve_failed) {
        return "reserve refused";
    }
    return "region smaller than the largest streamable matrix";
}

// The items of `li` one window stages: the layer's items of `role` whose expert is in `experts`, in the order of
// `experts`. An expert with no host item of that role (device-resident) has nothing to stage, an item the executor does
// not claim (staging NONE) is never staged, and an expert listed twice is staged once.
static inline std::vector<moe_stream_item> moe_stream_window_items(const moe_stream_layer_info & li,
                                                                   expert_tensor_role            role,
                                                                   const int32_t *               experts,
                                                                   size_t                        n) {
    std::vector<moe_stream_item> out;
    for (size_t i = 0; experts != nullptr && i < n; ++i) {
        bool listed = false;
        for (const moe_stream_item & taken : out) {
            listed = listed || taken.expert_id == experts[i];
        }
        for (size_t j = 0; !listed && j < li.items.size(); ++j) {
            const moe_stream_item & item = li.items[j];
            if (item.role == role && item.expert_id == experts[i] && item.staging != moe_stream_staging::NONE) {
                out.push_back(item);
                listed = true;
            }
        }
    }
    return out;
}

// True iff a window is exposed: staged with no earlier window of its layer to hide under, which is only the layer's
// first window in the micro-batch. Every later window has the window before it to hide under, which prefetches the
// next one (see moe_stream_window_request), and no window prefetches across layers. begin_ubatch resets the last
// window's layer, so a micro-batch's first window is exposed even when it is the layer the last micro-batch ended at.
// `prev_layer` is the layer of the window before this one, -1 for none. The window's role and phase play no part, so
// the predicate takes neither.
static inline bool moe_stream_window_exposed(int layer, int prev_layer) {
    return layer != prev_layer;
}

// The id moe_stream_region_reads keys a region's window by: one per (layer, role, phase), never UINT32_MAX. A role has
// fewer than 4096 phases, and MAX_LAYERS layers x 3 roles x 4096 phases fit 32 bits.
static inline uint32_t moe_stream_window_id(int layer, expert_tensor_role role, size_t phase) {
    GGML_ASSERT(layer >= 0 && layer < moe_exec_decision_log::MAX_LAYERS && phase < 4096);
    return static_cast<uint32_t>(layer) * 3u * 4096u +
           static_cast<uint32_t>(detail::moe_stream_role_slot(role)) * 4096u + static_cast<uint32_t>(phase);
}

// True iff a lease's retained owner lets the ring copy from it directly (llama.cpp-9y0p): a host-pinned WEIGHT
// allocation in one of this device's host zones, the shape mem-ops.cpp copies without a host bounce (a host zone of
// another device lives in another context). Any other lease would take the staged path, which the ring never does.
static inline bool moe_stream_owner_streamable(const alloc_metadata & owner, int device) {
    return owner.tier == alloc_tier::HOST_PINNED && owner.role == alloc_role::WEIGHT &&
           owner.host_zone != host_zone_id::COUNT && owner.device == device;
}

// A layer's rates, one source per fact (llama.cpp-kp1l). Each starts at the per-type seeds over the layer's host mix.
//   h2d:     the device's tuning-cache record where it measured, else `probed_h2d_gbps` (the device's own resolved rate
//            from ggml_sycl_measure_h2d_pinned: 0 when the probe has not measured or read one back), else the seed. A
//            run with the tuning cache disabled, or a record the store refused, still prices with the probed rate.
//   reorder: the record's rate where it measured, else the seed. A record rate that is not positive means
//            "unmeasured": the probe measures the H2D rate only and stores reorder as 0, so a later change of the seed
//            is not masked by a copy of the old seed in the file. A record written before llama.cpp-kp1l (the
//            CACHE_VERSION did not change) holds the old 200 GB/s seed as `reorder_gbps` and is read as measured, and
//            GGML_SYCL_H2D_PROBE=1 over such a record carries it forward (the GGML_SYCL_MOE_EXEC_REORDER_GBPS catalog
//            row). Deleting the device's `*-moe-exec-rates.json` clears it.
//   dev:     each type's recorded rate (the record's per-type slot, which has no writer yet), else its seed, as a
//            byte-weighted harmonic mean over the mix.
// A value is taken only when finite and positive. The environment overrides apply last and win over all of them.
static inline moe_exec_rates moe_stream_rates_assemble(const std::vector<std::pair<ggml_type, size_t>> & host_mix,
                                                       const ggml_sycl_moe_exec_rates_record *           record,
                                                       double             probed_h2d_gbps,
                                                       moe_exec_getenv_fn getenv_fn) {
    moe_exec_rates r = moe_exec_rates_seed(host_mix);
    if (record != nullptr && detail::moe_stream_rate_measured(record->h2d_gbps)) {
        r.h2d_gbps = record->h2d_gbps;
    } else if (detail::moe_stream_rate_measured(probed_h2d_gbps)) {
        r.h2d_gbps = probed_h2d_gbps;
    }
    if (record != nullptr) {
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

// True iff a prompt MUL_MAT_ID of `layer` in graph `graph_seq` starts a micro-batch: none is live, the layer is below
// the current one, or it is the current layer again in a new graph. A new graph at a higher layer carries the
// micro-batch on, since a split micro-batch spans several graph_compute calls.
static inline bool moe_stream_ubatch_starts(bool     live,
                                            int      layer,
                                            int      cur_layer,
                                            uint64_t graph_seq,
                                            uint64_t cur_graph_seq) {
    return !live || layer < cur_layer || (layer == cur_layer && graph_seq != cur_graph_seq);
}

// True iff a layer's decision moves bytes: the decision is RING and GGML_SYCL_MOE_EXEC is not cpu. AUTO stages when the
// cost model chose the ring, and ring forces that choice in moe_exec_weigh (llama.cpp-3460, lifting lead ruling
// c-3skf). What the ring stages, the batched prompt executor (llama.cpp-3460) consumes (W2).
static inline bool moe_stream_stage_now(const moe_exec_decision & d, moe_exec_policy policy) {
    return d.choice == moe_exec_choice::RING && policy != moe_exec_policy::CPU;
}

// A prompt's loans for its short ring (llama.cpp-3460, c-0tmn). end_prompt resets it; a micro-batch does not.
struct moe_stream_loan_latch {
    bool     asked         = false;  // a loan was asked this prompt
    bool     refused       = false;  // one came back REFUSED, or none could be sized: ask for no other until end_prompt
    bool     unconverged   = false;  // the refusal was the sizing's: no ring of the plan a loan would leave fits
    uint64_t refusals_seen = 0;      // the device's REFUSED count at the last ask
};

// What sizes a short ring's loan from the plan the loan would leave (llama.cpp-8p5x): the plan and the device the ring
// was sized for, the geometry its reserve was short of, and the same hooks and executor window that sized it
// (moe_stream_plan_ring_geometry). Plain data, read during one call: `plan` is the caller's for that call only.
struct moe_stream_loan_sizer {
    const placement_plan *       plan           = nullptr;
    int                          device         = -1;
    moe_stream_ring_geometry     geo            = {};
    moe_stream_prompt_kernel_fn  prompt_kernel  = nullptr;
    void *                       graph_ctx      = nullptr;
    moe_stream_library_claims_fn library_claims = nullptr;
    moe_stream_window_hints      hints          = {};
};

// A loan sized from the plan it leaves (moe_stream_loan_size_post_loan).
struct moe_stream_loan_sizing {
    bool                     converged  = false;  // the geometry the request was sized for is the post-loan one
    size_t                   bytes      = 0;      // the request: the slots of `geo` not held, times geo.slot_bytes
    moe_stream_ring_geometry geo        = {};     // converged: the geometry the request funds; else the last post-loan
    uint32_t                 steps      = 0;      // post-loan geometries computed
    uint32_t                 n_triplets = 0;      // triplets the request selects
};

static inline moe_stream_loan_sizing moe_stream_loan_size_post_loan(const moe_stream_loan_sizer & sizer,
                                                                    size_t                        bytes_short);

enum class moe_stream_loan_step : uint8_t {
    NONE,     // nothing asked: no shortfall, no request hook, or a loan this prompt was refused
    REQUEST,  // a loan of the shortfall was asked for the next graph boundary
    REFUSED,  // the loan asked last came back REFUSED: the latch holds, and the caller WARNs once
};

// What a short ring reserve does about its shortfall (llama.cpp-3460, closing c-0tmn for production). ring_ready()
// calls this from its reserve-failed branch, so once per micro-batch (the reserve latch stops the later layers before
// the reserve, c-i8rz), with the bytes the WEIGHT zone was short of as moe_stream_ring::reserve reported them and the
// ring's slot size, which the loan counts them in (llama.cpp-9x8t). It asks for a loan at the next graph boundary
// (ggml_backend_sycl_synchronize runs it, c-tdra), and the next micro-batch's reserve tries again. A REFUSED count that
// moved since the last ask means that loan was refused: retrying would refuse again, so the latch holds until
// end_prompt. An ACCEPTED loan that still left the ring short, or a BUSY one, leaves the count alone, so the next short
// micro-batch asks again; a repeated request is harmless, since the largest since the last boundary wins.
// With a `sizer`, the loan is sized from the plan it leaves (moe_stream_loan_size_post_loan, llama.cpp-8p5x): lent
// triplets become host matrices, which can widen the window and so the ring the next micro-batch reserves. The request
// is that converged loan, in that geometry's slots. When no loan converges, none is asked: the latch holds as for a
// refused loan, and REFUSED is answered once. A null `sizer` has no plan to size against, and asks for the shortfall as
// reported, in `slot_bytes`.
static inline moe_stream_loan_step moe_stream_loan_on_short_reserve(moe_stream_loan_latch &       latch,
                                                                    int                           device,
                                                                    size_t                        bytes_short,
                                                                    size_t                        slot_bytes,
                                                                    moe_stream_request_loan_fn    request_loan,
                                                                    moe_stream_loan_refusals_fn   loan_refusals,
                                                                    const moe_stream_loan_sizer * sizer) {
    if (request_loan == nullptr || latch.refused || bytes_short == 0) {
        return moe_stream_loan_step::NONE;
    }
    const uint64_t refusals_now = loan_refusals != nullptr ? loan_refusals(device) : 0;
    if (latch.asked && refusals_now != latch.refusals_seen) {
        latch.refused = true;
        return moe_stream_loan_step::REFUSED;
    }
    size_t bytes = bytes_short;
    size_t slot  = slot_bytes;
    if (sizer != nullptr) {
        const moe_stream_loan_sizing sized = moe_stream_loan_size_post_loan(*sizer, bytes_short);
        if (!sized.converged) {
            latch.refused     = true;
            latch.unconverged = true;
            return moe_stream_loan_step::REFUSED;
        }
        bytes = sized.bytes;
        slot  = sized.geo.slot_bytes;
    }
    latch.asked         = true;
    latch.refusals_seen = refusals_now;
    request_loan(device, bytes, slot);
    return moe_stream_loan_step::REQUEST;
}

// What begin_layer and window_begin need beyond their plan, cache and queue: the op's shape for the cost model, the
// backend's one policy and decision log (ggml_sycl_moe_exec_config(), ggml_sycl_moe_exec_log()), and two answers only
// the graph has, from the backend's MUL_MAT_ID scan: the host AOS lease of one of a layer's experts (keyed from that
// layer's weight tensor, as the hybrid path's host route keys it), and whether the library executor claims a layer's
// weight (llama.cpp-3460), which decides what the ring stages (W2).
struct moe_stream_op {
    uint32_t                n_tokens      = 0;  // ids->ne[1]
    int                     n_expert      = 0;  // src0->ne[2]
    int                     n_expert_used = 0;  // ids->ne[0]
    uint64_t                graph_seq     = 0;  // the context's graph_compute_seq
    moe_exec_policy         policy        = moe_exec_policy::AUTO;
    moe_exec_decision_log * log           = nullptr;
    void *                  graph_ctx     = nullptr;
    // The host lease of (layer, expert, role) and its bytes, or an invalid handle.
    mem_handle (*host_lease)(void * graph_ctx, int layer, int expert_id, expert_tensor_role role, size_t * bytes) =
        nullptr;
    // Handed to the index build, which no longer asks it (moe_stream_prompt_kernel_fn).
    moe_stream_prompt_kernel_fn  prompt_kernel  = nullptr;
    // Unset, the library executor claims no weight and nothing stages for it (moe_stream_plan_index_build).
    moe_stream_library_claims_fn library_claims = nullptr;
    // Unset, a short ring reserve asks for no loan (moe_stream_loan_on_short_reserve).
    moe_stream_request_loan_fn   request_loan   = nullptr;
    // Unset, a requested loan is never seen refused, so every short micro-batch asks again.
    moe_stream_loan_refusals_fn  loan_refusals  = nullptr;
    // The layer GGML_SYCL_MOE_EXEC_DECLINE_LAYER forces off the executor (moe_stream_layer_declined); -1 is none.
    int                          decline_layer  = -1;

    // The executor's phase the ring is sized to: the device's planned f16 weight slot and the group cap, handed as is
    // to moe_stream_plan_ring_geometry. Unset (0 slot bytes), there is no window and no ring.
    moe_stream_window_hints window;
};

// Why the resolver refused a staged ring slot (llama.cpp-o9uw). The enumerators name the causes; the staged branch's
// lambda in ggml_sycl_build_moe_resolved_batch fixes the order they are asked in. RECIPE covers both the missing
// recipe, asked before the route, and the recipe validator's refusal, asked after it. NONE means served.
enum class moe_staged_route_refusal : uint8_t {
    NONE,
    CLAIMED,        // the entry's consumer is the batched prompt executor, which reads the slot itself
    UNSUPPORTED,    // the capability query answers no kernel for the type in the entry's layout
    KERNEL,         // it answers a kernel other than the plain MMVQ_COMPAT
    DIRECT_KERNEL,  // Q1_0/NVFP4: only their direct kernel runs them on a device (moe_type_needs_direct_kernel)
    VIEW,           // the slot resolves to no device pointer
    LAYOUT,         // the slot's resolved layout is not the entry's
    ROUTE,          // validate_moe_batch_route refuses the route
    RECIPE,         // the recipe is missing, or validate_moe_execution_recipe refuses it
};

inline const char * moe_staged_route_refusal_name(moe_staged_route_refusal refusal) {
    switch (refusal) {
        case moe_staged_route_refusal::NONE:
            return "served";
        case moe_staged_route_refusal::CLAIMED:
            return "claimed";
        case moe_staged_route_refusal::UNSUPPORTED:
            return "unsupported";
        case moe_staged_route_refusal::KERNEL:
            return "kernel";
        case moe_staged_route_refusal::DIRECT_KERNEL:
            return "direct-kernel";
        case moe_staged_route_refusal::VIEW:
            return "view";
        case moe_staged_route_refusal::LAYOUT:
            return "layout";
        case moe_staged_route_refusal::ROUTE:
            return "route";
        case moe_staged_route_refusal::RECIPE:
            return "recipe";
    }
    return "unknown";
}

// The ring regions' last uses, as reusing a region needs them (lead ruling c-o9uw-1), keyed by window (W4 of
// docs/plans/2026-10-24-prefill-window-ring.md): per region, the window its last fill holds (moe_stream_window_id;
// UINT32_MAX: none since the ring was reserved) and that window's read gate once the executor records one, an event
// behind every dequant that read the window. A region turns over several times per layer, so the gate is the window's,
// never a layer's. stage_window() refills a region, window_done() records a window's read gate and reset() starts both
// regions over, for a new ring or a torn-down one, so a gate recorded for one fill is never taken for another.
// Templated on the event so test-moe-stream-phase drives it without a SYCL object; moe_stream_phase holds it over
// sycl::event.
template <typename event_t> struct moe_stream_region_reads {
    uint32_t               window[2] = { UINT32_MAX, UINT32_MAX };
    std::optional<event_t> read_gate[2];

    // A copy went into `region`: it now holds window `w`, with no read gate until w's readers record one.
    void refill(int region, uint32_t w) {
        window[region] = w;
        read_gate[region].reset();
    }

    // `readers_done` follows every dequant that read window `w`, so it is the read gate of the region holding w. A
    // region refilled since holds another window and takes nothing of w's, and the no-window id names no region.
    void window_done(uint32_t w, const event_t & readers_done) {
        for (int r = 0; r < 2 && w != UINT32_MAX; ++r) {
            if (window[r] == w) {
                read_gate[r] = readers_done;
            }
        }
    }

    void reset() {
        for (int r = 0; r < 2; ++r) {
            window[r] = UINT32_MAX;
            read_gate[r].reset();
        }
    }
};

// The key a window's timing record is kept under (moe_stream_window_timeline): the micro-batch's sequence number and
// the window's id, since a window id names the same (layer, role, phase) again in every micro-batch.
static inline uint64_t moe_stream_window_timing_key(uint64_t ubatch_seq, uint32_t window) {
    return (ubatch_seq << 32) | window;
}

// The per-window timing GGML_SYCL_MOE_EXEC_TRACE=2 prints, one record per window, kept by the phase over sycl::event
// and flushed at the synchronize boundary, where the host has already waited for the queue, so reading an event's
// device times never waits on the compute path. Disabled (below trace level 2) it records nothing. Enabled, at() grows
// `records` on the submitting thread inside graph compute, a trace-only allocation path, and clear() keeps the
// vector's capacity, so later prompts of the same shape reuse it. The submitting thread is the only writer, so the
// records take no lock. Templated on the event so test-moe-stream-phase drives it without a SYCL object. Each figure
// and where it comes from:
//   bytes            the window's own copies' bytes: those staged for it, whichever window_begin call staged them
//                    (a prefetched window's copies go in during the window before it)
//   exposed          moe_stream_window_exposed: the layer's first window, which nothing staged ahead
//   submit_us        host steady_clock time inside window_begin for this window, its own copies and the prefetch of
//                    the next window included: the host-side submit overhead
//   copy_us          device time of the window's copies: the first copy's command_start to the last copy's
//                    command_end. The copy queue is in order, so the span covers every copy of the window (and any
//                    other submitter's work interleaved on that queue). -1 when a copy has no device times (the copy
//                    queue is built with profiling only under trace level 2) or the window copied nothing itself
//   readers_wait_us  how long the window's first copy waited FOR its read gate (the barrier behind the dequants of
//                    the window the region last held, or the compute-queue barrier that stands in for it): the gate's
//                    command_end minus the moment the copy could otherwise have started, the later of the copy's
//                    command_submit and the command_end of the copy queued before it on the copy queue (the last copy
//                    this timeline saw since the previous flush; none: the submit alone). Clamped at 0. A large value
//                    means the copy queue sat waiting for the compute queue to release the region. -1 when the copies
//                    waited on no read gate, an event has no device times, or the first copy's command_submit is 0
//                    (the runtime gave no submit time). Other submitters' work on the copy queue and the copy's other
//                    dependencies are not seen
//   start_lag_us     the first copy's command_start minus the read gate's command_end, clamped at 0: how late the copy
//                    began after the gate released it. The copy depends on the gate, so it never starts before the
//                    gate's end; a large value means the copy queue was busy or the copy was submitted late, i.e. the
//                    gate was NOT what held it. -1 when the copies waited on no read gate, or the first copy or the
//                    gate has no device times
//   launches         the dequant kernels and the GEMM groups the executor submitted for the window
template <typename event_t> struct moe_stream_window_timeline {
    struct record {
        uint64_t               key       = 0;
        bool                   begun     = false;  // window_begin named it; a prefetch never begun prints nothing
        int                    layer     = -1;
        expert_tensor_role     role      = expert_tensor_role::UNKNOWN;
        size_t                 phase     = 0;
        bool                   exposed   = false;
        double                 submit_us = 0.0;
        size_t                 bytes     = 0;
        uint32_t               launches  = 0;
        uint32_t               n_copies  = 0;
        event_t                first_copy{};
        event_t                last_copy{};
        std::optional<event_t> reuse_gate;  // the read gate the window's copies waited on
        std::optional<event_t> prev_copy;   // the copy queued just before the first copy, if this timeline saw one
    };

    bool                   enabled = false;  // trace level 2
    std::vector<record>    records;
    std::optional<event_t> queue_tail;       // the latest copy recorded since the previous flush

    void clear() {
        records.clear();
        queue_tail.reset();
    }

    // The latest record under `key`: a window's records are recent, so the search runs from the back.
    record * find(uint64_t key) {
        for (size_t i = records.size(); i-- > 0;) {
            if (records[i].key == key) {
                return &records[i];
            }
        }
        return nullptr;
    }

    record & at(uint64_t key) {
        record * r = find(key);
        if (r != nullptr) {
            return *r;
        }
        records.emplace_back();
        records.back().key = key;
        return records.back();
    }

    // stage_window() copied `bytes` for the window under `key`, as `copy`, after `gate`.
    void copied(uint64_t key, const event_t & copy, size_t bytes, const std::optional<event_t> & gate) {
        if (!enabled) {
            return;
        }
        record & r = at(key);
        if (r.n_copies == 0) {
            r.first_copy = copy;
            r.reuse_gate = gate;
            r.prev_copy  = queue_tail;
        }
        queue_tail  = copy;
        r.last_copy = copy;
        ++r.n_copies;
        r.bytes += bytes;
    }

    // window_begin() ran for the window under `key`.
    void begun(uint64_t key, int layer, expert_tensor_role role, size_t phase, bool exposed, double submit_us) {
        if (!enabled) {
            return;
        }
        record & r  = at(key);
        r.begun     = true;
        r.layer     = layer;
        r.role      = role;
        r.phase     = phase;
        r.exposed   = exposed;
        r.submit_us = submit_us;
    }

    // The executor submitted `n` more dequants or GEMM groups for the window under `key`.
    void launched(uint64_t key, uint32_t n) {
        if (!enabled) {
            return;
        }
        record * r = find(key);
        if (r != nullptr) {
            r->launches += n;
        }
    }

    // One line per begun window to `sink`, then the records go. `times(ev, &submit_ns, &start_ns, &end_ns)` answers
    // an event's device command_submit, command_start and command_end, false when it has none. Returns the lines
    // printed.
    template <typename times_fn, typename sink_fn> size_t flush(int device, times_fn times, sink_fn sink) {
        size_t printed = 0;
        for (const record & r : records) {
            if (!r.begun) {
                continue;
            }
            double     copy_us         = -1.0;
            double     readers_wait_us = -1.0;
            double     start_lag_us    = -1.0;
            uint64_t   first_submit = 0, first_start = 0, first_end = 0;
            const bool first_timed = r.n_copies != 0 && times(r.first_copy, &first_submit, &first_start, &first_end);
            uint64_t   last_submit = 0, last_start = 0, last_end = 0;
            if (first_timed && times(r.last_copy, &last_submit, &last_start, &last_end) && last_end >= first_start) {
                copy_us = static_cast<double>(last_end - first_start) / 1e3;
            }
            uint64_t   gate_submit = 0, gate_start = 0, gate_end = 0;
            const bool gate_timed =
                first_timed && r.reuse_gate && times(*r.reuse_gate, &gate_submit, &gate_start, &gate_end);
            // The moment the first copy could have started without its gate: its submit, or the end of the copy
            // ahead of it on the in-order copy queue when that is later.
            uint64_t       prev_submit = 0, prev_start = 0, prev_end = 0;
            const bool     prev_known = !r.prev_copy || times(*r.prev_copy, &prev_submit, &prev_start, &prev_end);
            const uint64_t ready      = std::max(first_submit, prev_end);
            // A zero submit is no time at all, not the device clock's origin: the wait would read as the gate's end.
            if (gate_timed && prev_known && first_submit != 0) {
                readers_wait_us = gate_end > ready ? static_cast<double>(gate_end - ready) / 1e3 : 0.0;
            }
            if (gate_timed) {
                start_lag_us = first_start > gate_end ? static_cast<double>(first_start - gate_end) / 1e3 : 0.0;
            }
            char line[288];
            std::snprintf(line, sizeof(line),
                          "[MOE-EXEC-TRACE] window device %d layer %d role %s phase %zu bytes=%zu exposed=%d "
                          "submit_us=%.0f copy_us=%.0f readers_wait_us=%.0f start_lag_us=%.0f launches=%u",
                          device, r.layer, expert_tensor_role_name(r.role), r.phase, r.bytes, r.exposed ? 1 : 0,
                          r.submit_us, copy_us, readers_wait_us, start_lag_us, r.launches);
            sink(line);
            ++printed;
        }
        clear();
        return printed;
    }
};

// What the batched prompt executor asks of the ring for one window (W1): one phase of one (layer, role), and the window
// to prefetch under it. The executor's phases split a role's active experts, and the graph dispatches a layer's roles
// in the order GATE, UP, DOWN (the gated activation takes the gate output as src[0]), so the window after a phase is
// the next phase of its role, or phase 0 of the next role at the role's last phase. There is no cross-layer prefetch.
struct moe_stream_window_request {
    int                layer     = -1;
    expert_tensor_role role      = expert_tensor_role::UNKNOWN;
    size_t             phase     = 0;        // this phase's index within the role (0 = first)
    const int32_t *    experts   = nullptr;  // the phase's active experts, host and device mixed
    size_t             n         = 0;
    // The role whose phase 0 is prefetched; UNKNOWN prefetches the next phase of `role`.
    expert_tensor_role next_role = expert_tensor_role::UNKNOWN;
    const int32_t *    next      = nullptr;  // the prefetched phase's experts; null: no prefetch
    size_t             n_next    = 0;
};

// The resolver's STAGED_RING lookups over one prompt (llama.cpp-o9uw): how many it served and refused, and the refusals
// by cause and detail, so the trace says why a staged slot went unserved (plan Task 8). A detail is a string with
// static storage (a capability reason, or moe_batch_reject_reason_name's), or null. An entry the batched prompt
// executor reads, on a weight it takes whole, is neither: the resolver hands it off (llama.cpp-3460), and handed_off
// counts it apart, so a refusal is always a slot nobody reads.
struct moe_staged_route_tally {
    struct refusal_count {
        moe_staged_route_refusal refusal = moe_staged_route_refusal::NONE;
        const char *             detail  = nullptr;
        size_t                   n       = 0;
    };

    size_t                     served     = 0;
    size_t                     handed_off = 0;
    size_t                     refused    = 0;
    std::vector<refusal_count> refusals;  // one per distinct (cause, detail), in the order first seen

    void        count(moe_staged_route_refusal refusal, const char * detail);
    void        count_handoff();
    // "cause(detail)=n", or "cause=n" with no detail, per distinct (cause, detail), space-separated; empty when none.
    std::string refusals_text() const;
};

// The prompt phase of one device context (ggml_backend_sycl_context::moe_stream): the VRAM ring, the stage table and
// the current micro-batch's per-layer decisions.
//
// The ring serves the batched prompt executor only, a window at a time (W1, W2 of
// docs/plans/2026-10-24-prefill-window-ring.md). begin_layer weighs and decides a layer and stages nothing. On a layer
// the executor claims whole (executor_claims), the executor asks window_begin before it dequantizes a phase, which
// copies the phase's missing host matrices into a region and prefetches the next window into the other one, and it
// calls window_done behind the phase's dequants.
//
// Ordering (ruling A, c-3bde): the device orders everything, the host never waits or asks. A copy waits for its ring
// region's gate, and a region closes on an event after its last copy. Every slot holds AOS and is read in place by the
// executor's dequant kernel on the in-order compute queue (c-las5), while the copy that rewrites it runs on the copy
// queue, which no queue order puts after the dequant. So a refill also waits for the region's read gate, the barrier
// the executor submits behind the dequants of the window the region last held (window_done). With no read gate
// recorded, a compute-queue barrier submitted at the refill stands in (see stage_window()). The ring is offered only on
// an in-order compute queue.
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
    // Called at every prompt MUL_MAT_ID of `layer`; the first one per micro-batch decides. The cost model weighs the
    // layer, and a RING decision stands only on a layer the executor claims whole (moe_stream_executor_claims) once the
    // ring reserves, at the first such layer; anything else is RING_UNAVAILABLE. The layer the op's decline_layer names
    // is CPU, DECLINED, whatever the cost model chose (moe_stream_layer_declined). Nothing is staged here, and no layer
    // is weighed ahead: the executor stages its windows (window_begin), whose experts and cache are their own, so
    // `cache` and the ids are not read. A layer below the last one, or the same layer in a new graph, starts a
    // micro-batch (moe_stream_ubatch_starts). Returns the layer's decision.
    moe_exec_decision         begin_layer(int                    layer,
                                          const placement_plan & plan,
                                          int                    device,
                                          unified_cache &        cache,
                                          sycl::queue &          ccs,
                                          const int32_t *        ids,
                                          size_t                 n_ids,
                                          const moe_stream_op &  op);
    const moe_stream_staged * find(int layer, int expert_id, expert_tensor_role role) const;
    // True iff slots were taken from the ring's current region since it last flipped.
    bool                      region_open() const;
    // After `layer`'s down op: an open region closes on `down_done`, a device event behind that op (see
    // ggml_sycl_moe_stream_layer_done). The read gates are the windows' (window_done), not the layer's.
    void                      layer_done(int layer, sycl::event down_done);
    // The first decode graph after a prompt: the table, the decisions and the ring go.
    void                      end_prompt();
    // Whether the last prompt end_prompt ended kept a whole ring reservation at any point (ring_ready()): a prompt
    // whose ring never engaged used none of the expert loan its short reserves asked for, so the backend returns that
    // loan at once instead of a triplet per decode graph (llama.cpp-i4ny). take_prompt_ended() answers true once after
    // each end_prompt that ended a prompt, so the backend reads the flag once per prompt.
    bool                      last_prompt_ring_engaged() const;
    bool                      take_prompt_ended();
    // The resolver's STAGED_RING lookups this prompt (llama.cpp-o9uw): NONE for an entry it served, else why it
    // refused the entry and left it to the route core, with the detail (moe_staged_route_tally). end_prompt reports and
    // clears them, or the destructor for a prompt no end_prompt ended: on the trace, and at WARN under an explicit ring
    // when any was refused.
    void                      count_staged_route(moe_staged_route_refusal refusal, const char * detail);
    // An entry the resolver handed to the batched prompt executor (llama.cpp-3460): the executor claims the layer
    // (executor_claims), so it reads the slot, and the lookup is no refusal.
    void                      count_executor_handoff();

    // True iff the batched prompt executor takes `layer`'s weights whole from the ring: the layer's decision is RING,
    // which begin_layer gives only to a layer the executor claims (moe_stream_executor_claims, asked again here), and
    // the ring is reserved (ring_ready()). A refused reserve answers false, and the layer stays hybrid.
    bool     executor_claims(int layer);
    // Before the executor dequantizes one phase of (layer, role), on a layer executor_claims answered true for: every
    // host matrix of the window that the ring can take is in the stage table after this returns. One not there yet is
    // copied now, with the rest of its window, into a region of its own, on the copy queue behind the region's read
    // gate, and the window after it (req.next) is copied into the other region. Returns the window's host experts that
    // are staged after the call; *misses, when given, is how many of them were not staged ahead, 0 for an exposed
    // window (moe_stream_window_exposed), the count the decision log's window counters take.
    uint32_t window_begin(const moe_stream_window_request & req,
                          unified_cache &                   cache,
                          const moe_stream_op &             op,
                          uint32_t *                        misses);
    // `readers_done` follows every dequant that read the window (layer, role, phase): it becomes the read gate of the
    // region holding that window. It frees nothing: the region's next refill waits for it.
    void     window_done(int layer, expert_tensor_role role, size_t phase, sycl::event readers_done);
    // The executor submitted `n` dequant kernels and GEMM groups for the window (layer, role, phase), for the window's
    // timing line (moe_stream_window_timeline). Records nothing below GGML_SYCL_MOE_EXEC_TRACE=2.
    void     window_launched(int layer, expert_tensor_role role, size_t phase, uint32_t n);
    // At the synchronize boundary, after the queue wait: one WARN line per window since the last flush, under
    // GGML_SYCL_MOE_EXEC_TRACE=2 (moe_stream_window_timeline), then the records go. Reading the events' device times
    // there never waits on the compute path. Prints nothing below trace level 2.
    void     flush_window_timing();

    // The host experts one window stages at most: a region's slots, one executor phase (W3). 0 with no ring geometry.
    uint32_t window_experts() const { return geo_.region_slots; }

    size_t staged_routes_served() const { return staged_routes_.served; }

    size_t staged_routes_refused() const { return staged_routes_.refused; }

    size_t staged_routes_handed_off() const { return staged_routes_.handed_off; }

    // This prompt's refusals by cause and detail (moe_staged_route_tally::refusals_text).
    std::string staged_route_refusals() const { return staged_routes_.refusals_text(); }

    // The times the ring was given back (teardown). The backend nulls the slot pointers its pointer tables copied out
    // of the stage table when this moves (llama.cpp-o9uw).
    uint64_t ring_releases() const { return ring_releases_; }
#if defined(GGML_SYCL_PRIVATE_TESTING)
    // Puts `entry` in the stage table under (layer, expert_id, role), as stage_window() does, for a device test of the
    // route.
    void       test_insert_staged(int layer, int expert_id, expert_tensor_role role, moe_stream_staged entry);
    // Reserves the ring with `geo` on `device`, its barriers on `ccs`, as prepare() and the first claimed layer's
    // ring_ready() do; test_acquire_slot takes a slot of it with `layout`, as stage_window() does. For a device test of
    // a staged slot's lifetime (llama.cpp-o9uw).
    bool       test_reserve_ring(int device, const moe_stream_ring_geometry & geo, sycl::queue & ccs);
    mem_handle test_acquire_slot(ggml_layout_mode layout);
#endif

  private:
    struct layer_state {
        moe_exec_decision decision;
        bool              decided = false;
    };

    // One window stage's matrices, each one expert's matrix of the window's role.
    struct stage_counts {
        int    staged    = 0;  // copied into a slot
        int    exhausted = 0;  // skipped: no slot left in the region
        size_t bytes     = 0;  // AOS bytes copied
    };

    void              prepare(const placement_plan & plan, int device, sycl::queue & ccs, const moe_stream_op & op);
    size_t            ring_bytes_free() const;
    moe_exec_decision weigh(const moe_stream_layer_info & li, const moe_stream_op & op, double window_us) const;
    moe_exec_decision decision_of(int layer) const;
    stage_counts      stage_window(const std::vector<moe_stream_item> & items,
                                   uint32_t                             window_id,
                                   unified_cache &                      cache,
                                   const moe_stream_op &                op);
    bool              ring_ready(const moe_stream_loan_sizer * sizer);
    void              close_region(sycl::event region_done);
    void              reset_regions();
    sycl::event       compute_barrier();
    void              teardown(sycl::event last_use);
    void              report(int layer, const layer_state & st) const;
    void              report_window(int layer, expert_tensor_role role, size_t phase, const stage_counts & n) const;
    void              report_staged_routes();
    void              warn_ring_unavailable(const moe_stream_op & op, const moe_stream_layer_info & li);
    void              warn_loan_refused(size_t bytes_short) const;
    void              warn_partial_released(uint32_t held) const;

    std::unique_ptr<moe_stream_ring>                ring_;
    // A refused reserve's ring and the slots it got, for the next micro-batch's ring_ready() to continue from while a
    // loan can still fund it. Only ring_ready() takes or stores it, end_prompt() checks it, teardown() drops it, and
    // nothing stages into it.
    std::unique_ptr<moe_stream_ring>                pending_ring_;
    std::unordered_map<uint64_t, moe_stream_staged> table_;           // key = moe_stream_table_key(layer, expert, role)
    std::vector<uint64_t>                           region_keys_[2];  // the table keys whose slots each region holds
    int                                             cur_region_       = 0;      // flips with every epoch_boundary
    bool                                            region_submitted_ = false;  // a copy went into it since
    sycl::event                                     region_last_;  // its last copy's event (see stage_window())
    moe_stream_plan_index                           index_;
    std::unordered_map<int, layer_state>            layers_;
    std::optional<sycl::queue>                      ccs_;  // the compute queue the readers and the barriers run on
    moe_stream_ring_geometry                        geo_;
    moe_stream_window_hints                         window_;  // the executor window prepare() sized geo_ with
    ggml_sycl_moe_exec_rates_record                 record_             = {};
    int                                             device_             = -1;
    int                                             cur_layer_          = -1;
    uint64_t                                        cur_graph_seq_      = 0;
    uint32_t                                        n_tokens_           = 0;
    bool                                            ubatch_live_        = false;
    bool                                            prepared_           = false;
    bool                                            in_order_           = false;
    bool                                            reserve_failed_     = false;
    bool                                            record_valid_       = false;
    bool                                            record_looked_up_   = false;
    bool                                            warned_unavailable_ = false;  // this prompt's diagnostic is out
    // Each region's last fill: its window and that window's read gate (stage_window(), window_done()).
    moe_stream_region_reads<sycl::event>            region_reads_;
    int      prev_window_layer_      = -1;          // the layer of this micro-batch's last window_begin, -1 for none
    uint32_t next_window_id_         = UINT32_MAX;  // the window the last window_begin prefetched, UINT32_MAX for none
    bool     next_window_whole_      = false;       // that prefetch staged every matrix of its window
    // The windows' timing under GGML_SYCL_MOE_EXEC_TRACE=2, keyed by micro-batch and window:
    // begin_layer enables it from the op's decision log, stage_window(), window_begin and window_launched() record,
    // and flush_window_timing() prints and clears it. Written by the submitting thread only, so it takes no lock.
    moe_stream_window_timeline<sycl::event> window_timing_;
    uint64_t                                ubatch_seq_    = 0;  // begin_ubatch counts micro-batches for the timing key
    // The last op's decision log and policy: report_staged_routes() reads both, from end_prompt and the destructor, and
    // window_begin feeds the log's window counters.
    moe_exec_decision_log *     log_ = nullptr;
    moe_exec_policy             policy_        = moe_exec_policy::AUTO;
    // The last op's loan hooks, which ring_ready() calls on a short reserve, and this prompt's loans (llama.cpp-3460).
    moe_stream_request_loan_fn  request_loan_  = nullptr;
    moe_stream_loan_refusals_fn loan_refusals_ = nullptr;
    moe_stream_loan_latch       loan_latch_;
    moe_staged_route_tally      staged_routes_;
    uint64_t                    ring_releases_            = 0;
    // This prompt kept a whole reservation (ring_ready()); end_prompt copies it into last_prompt_ring_engaged_ and
    // marks the prompt ended for take_prompt_ended() (llama.cpp-i4ny).
    bool                        ring_engaged_             = false;
    bool                        last_prompt_ring_engaged_ = false;
    bool                        prompt_ended_             = false;
};

// ---- Loan selection (llama.cpp-0zmf, plan Task 11a): which resident expert triplets a loan may take. -------------------
// Pure over the plan, like the stream list above: no device, no queue. The plan's build_index() must have run.

namespace detail {

// True iff a SOA-to-AOS inverse of ggml_sycl_reorder_weight_gpu (the load fill's reorder, ggml-sycl.cpp) exists for
// `type`. None exists yet: that function only runs AOS to SOA, so v1 lends AOS-layout entries only. Add a case here
// when an inverse kernel lands.
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

// The lendable tail of the resident set on `device`: whole (gate, up, down) triplets, walking plan.entries in REVERSE
// (pack-group order, popularity rank when ranks exist, so the tail is the least popular), until what they free covers
// bytes_wanted or nothing lendable is left. A triplet is never split. Keys come out in the order the triplets are met,
// gate/up/down within each. *bytes_selected (when non-null) receives the summed vram_charge_size.
// *slots_selected (when non-null) receives the ring slots the selected triplets fund, the figure the walk counted:
// 0 when slot_bytes is 0. It is not bytes_selected / slot_bytes: each matrix's remainder below a slot is in the byte
// sum and funds nothing, so a caller that answers in slots (the auto-ubatch fit's probe) reads this count.
// With slot_bytes 0, what they free is the summed vram_charge_size. With a ring slot of slot_bytes, it is ring slots
// (llama.cpp-9x8t): the walk wants bytes_wanted / slot_bytes slots, rounded up, and each matrix frees
// vram_charge_size / slot_bytes of them. A lent matrix leaves one hole in the WEIGHT zone, and the floors of three
// holes can sum to less than the floor of their sum, so a byte count would over-count and send the ring back for
// another loan at another boundary. Adjacent holes may coalesce, so the floor is an under-count, the safe direction. A
// triplet that frees no slot is skipped, not lent: it would cost the model three resident matrices for nothing. One
// whose per-matrix floors sum to one is lent whole and counts one.
// Precondition: plan.build_index() has run; find_expert_entry returns nullptr on an un-indexed plan, so selection then
// returns nothing (fail-safe, no signal).
static inline std::vector<expert_placement_key> moe_expert_loan_select(const placement_plan & plan,
                                                                       int                    device,
                                                                       size_t                 bytes_wanted,
                                                                       size_t                 slot_bytes,
                                                                       size_t *               bytes_selected,
                                                                       size_t * slots_selected = nullptr) {
    std::vector<expert_placement_key>  out;
    size_t                             sum = 0;
    size_t                             slots = 0;
    const size_t                       slots_wanted =
        slot_bytes != 0 ? bytes_wanted / slot_bytes + (bytes_wanted % slot_bytes != 0 ? 1 : 0) : 0;
    std::unordered_map<uint64_t, bool> bulk_memo;
    std::unordered_set<uint64_t>       visited;
    for (size_t i = plan.entries.size(); i-- > 0 && (slot_bytes != 0 ? slots < slots_wanted : sum < bytes_wanted);) {
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
        if (slot_bytes != 0) {
            size_t triplet_slots = 0;
            for (int r = 0; r < 3; ++r) {
                triplet_slots += t.e[r]->vram_charge_size / slot_bytes;
            }
            if (triplet_slots == 0) {
                continue;
            }
            slots += triplet_slots;
        }
        for (int r = 0; r < 3; ++r) {
            out.push_back({ e.layer_id, e.expert_id, static_cast<expert_tensor_role>(r + 1) });
            sum += t.e[r]->vram_charge_size;
        }
    }
    if (bytes_selected) {
        *bytes_selected = sum;
    }
    if (slots_selected) {
        *slots_selected = slots;
    }
    return out;
}

// ---- Loan sizing (llama.cpp-8p5x, owner ruling c-ifgn): a short ring's loan, sized from the plan the loan leaves. -----
//
// A loan flips its triplets to the host, and a host matrix the executor claims takes a ring slot, so the ring the next
// micro-batch rebuilds from the published plan (prepare()) can be wider than the one the reserve was short of: the
// window, min(role_items_max, G_plan), grows with the heaviest role's host matrices. The cap does not move with a loan
// (the lent bytes count back as resident, llama.cpp-7999), so a wider ring can also exceed it.
//
// Each step sizes the request for a target geometry, selects the triplets that request takes (moe_expert_loan_select,
// the walk the boundary's loan repeats over the same plan), and sizes the ring of the plan with those triplets read as
// host (moe_stream_plan_ring_geometry's as_host) and their device bytes added to hints.lent_bytes, as the ledger counts
// them once lent. The request converges when that post-loan geometry is the target: it funds the ring the next
// micro-batch reserves. The request for a target is its slots not held, times its slot size. The reserve's held slots,
// n_slots - bytes_short / slot_bytes, count while the slot size is the reserve's: prepare() gives them back on the new
// geometry, and each leaves a hole one such slot wide. With another slot size they count for nothing.
//
// Lending more only adds host matrices, so a step that does not converge widens the ring, by a region slot or a larger
// slot, and the window stops at G_plan: within G_plan + 1 post-loan geometries the request converges, or none does.
// The sizing refuses (converged false) when the post-loan ring cannot exist (n_slots 0: over the cap), when it differs
// from the target without growing (a lent matrix with a larger f16 expert can lower G_plan), when the bound is spent,
// and when it has nothing to size (no plan or no geometry). It reads the plan only: no device, no lock, nothing lent.
// The claims hook answers from its own plan, the pre-loan one (c-gi44); a staged claim reads the layout of the
// weight's device experts, which a loan of AOS triplets leaves as it was.
static inline moe_stream_loan_sizing moe_stream_loan_size_post_loan(const moe_stream_loan_sizer & sizer,
                                                                    size_t                        bytes_short) {
    moe_stream_loan_sizing           out;
    const moe_stream_ring_geometry & geo = sizer.geo;
    if (sizer.plan == nullptr || geo.n_slots == 0 || geo.slot_bytes == 0 || bytes_short == 0) {
        return out;
    }
    // Rounded up, so a shortfall that is not whole slots credits fewer held slots, never more.
    const size_t   short_slots = bytes_short / geo.slot_bytes + (bytes_short % geo.slot_bytes != 0 ? 1 : 0);
    const uint32_t held        = short_slots >= geo.n_slots ? 0 : geo.n_slots - static_cast<uint32_t>(short_slots);

    moe_stream_ring_geometry     target = geo;
    std::unordered_set<uint64_t> as_host;
    for (uint32_t step = 1;; ++step) {
        const uint32_t credit = target.slot_bytes == geo.slot_bytes ? std::min(held, target.n_slots) : 0;
        const size_t   bytes  = static_cast<size_t>(target.n_slots - credit) * target.slot_bytes;
        const std::vector<expert_placement_key> keys =
            moe_expert_loan_select(*sizer.plan, sizer.device, bytes, target.slot_bytes, nullptr);
        as_host.clear();
        size_t lent = 0;
        for (const expert_placement_key & k : keys) {
            as_host.insert(moe_stream_table_key(k.layer_id, k.expert_id, expert_tensor_role::UNKNOWN));
            const placement_entry * e = sizer.plan->find_expert_entry(k.layer_id, k.expert_id, k.role);
            lent += e != nullptr ? e->dst_size : 0;
        }
        moe_stream_window_hints hints = sizer.hints;
        hints.lent_bytes              = hints.lent_bytes + lent < hints.lent_bytes ? SIZE_MAX : hints.lent_bytes + lent;
        moe_stream_plan_index          idx;
        const moe_stream_ring_geometry post =
            moe_stream_plan_ring_geometry(*sizer.plan, sizer.device, sizer.prompt_kernel, sizer.graph_ctx,
                                          sizer.library_claims, hints, &idx, &as_host);
        out.steps      = step;
        out.bytes      = bytes;
        out.n_triplets = static_cast<uint32_t>(keys.size() / 3);
        out.geo        = post;
        if (moe_stream_ring_geometry_same(post, target)) {
            out.converged = true;
            return out;
        }
        if (post.n_slots == 0 || post.bytes <= target.bytes || step > idx.window_experts) {
            return out;
        }
        target = post;
    }
}

}  // namespace ggml_sycl
