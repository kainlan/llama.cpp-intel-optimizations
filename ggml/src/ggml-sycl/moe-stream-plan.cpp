// moe_stream_phase: the prompt phase's VRAM ring, stage table and per-layer decisions (llama.cpp-xod7, plan Task 7b of
// llama.cpp-36j0). The pure parts it consults are in moe-stream-plan.hpp; test-moe-stream-phase pins them.
//
// Nothing here waits on the host or asks an event for its status (ruling A, c-3bde): a copy waits for its region's
// gate on the device, a reorder for its copy, and a region closes on an event after its last use.

#include "moe-stream-plan.hpp"

#include "common.hpp"  // moe_stream_phase_deleter
#include "ggml-impl.h"
#include "mem-handle.hpp"
#include "mem-ops.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace ggml_sycl {

namespace {

const char * moe_stream_getenv(const char * name) {
    return std::getenv(name);
}

}  // namespace

moe_stream_phase::moe_stream_phase() = default;

// The context destructor resets this after draining its queues, so the barrier is already complete; it still orders the
// ring's return after every use for any other owner. As in moe_stream_ring's destructor, a throw from the barrier or
// from retain_handles_until_event() terminates: there is deliberately no catch.
moe_stream_phase::~moe_stream_phase() {
    teardown(ring_ && ccs_ ? ccs_->ext_oneapi_submit_barrier() : sycl::event());
}

void moe_stream_phase::begin_ubatch(uint32_t n_tokens) {
    table_.clear();
    region_keys_[0].clear();
    region_keys_[1].clear();
    layers_.clear();
    cur_layer_   = -1;
    ubatch_seq_  = ubatch_live_ ? ubatch_seq_ + 1 : 0;
    ubatch_live_ = true;
    n_tokens_    = n_tokens;
}

moe_exec_decision moe_stream_phase::begin_layer(int                    layer,
                                                const placement_plan & plan,
                                                int                    device,
                                                unified_cache &        cache,
                                                sycl::queue &          ccs,
                                                const int32_t *        ids,
                                                size_t                 n_ids,
                                                const moe_stream_op &  op) {
    const bool new_ubatch =
        !ubatch_live_ || layer < cur_layer_ || (layer == cur_layer_ && op.graph_seq != cur_graph_seq_);
    if (!new_ubatch && layer == cur_layer_) {
        return decision_of(layer);  // the layer's later ops: the first one decided
    }
    const std::chrono::steady_clock::time_point now       = std::chrono::steady_clock::now();
    double                                      window_us = 0.0;
    if (new_ubatch) {
        begin_ubatch(op.n_tokens);
        prepare(plan, device, ccs);
    } else {
        window_us = std::chrono::duration<double, std::micro>(now - last_layer_tp_).count();
    }
    last_layer_tp_ = now;
    cur_layer_     = layer;
    cur_graph_seq_ = op.graph_seq;
    bitmap_        = moe_stream_ids_bitmap(ids, n_ids, op.n_expert);

    // The layer itself: weighed here only when the previous layer did not weigh it (the micro-batch's first host
    // layer: its copy is exposed), staged here when it was not staged a layer ahead.
    const auto cur = index_.layers.find(layer);
    if (cur != index_.layers.end()) {
        layer_state & st = layers_[layer];
        if (!st.decided) {
            st.decision = weigh(cur->second, op, 0.0);
            st.decided  = true;
        }
        if (!st.resolved) {
            resolve(layer, cur->second, st, cache, op);
        }
    }

    // The next layer, under this layer's window. A STREAM_ALL ring knows its experts now and stages now; a
    // ROUTED_ONLY ring needs its own ids and stages at its own first op; a CPU decision is only logged.
    const int  next = layer + 1;
    const auto nxt  = index_.layers.find(next);
    if (nxt != index_.layers.end() && op.layer_present != nullptr && op.layer_present(op.graph_ctx, next)) {
        layer_state & ns = layers_[next];
        if (!ns.decided) {
            ns.decision  = weigh(nxt->second, op, window_us);
            ns.window_us = window_us;
            ns.decided   = true;
            const bool routed_ring =
                ns.decision.choice == moe_exec_choice::RING && ns.decision.regime == moe_exec_regime::ROUTED_ONLY;
            if (!routed_ring) {
                resolve(next, nxt->second, ns, cache, op);
            }
        }
    }
    return decision_of(layer);
}

const moe_stream_staged * moe_stream_phase::find(int layer, int expert_id, expert_tensor_role role) const {
    const auto it = table_.find(moe_stream_table_key(layer, expert_id, role));
    return it == table_.end() ? nullptr : &it->second;
}

bool moe_stream_phase::fully_staged(int layer, const int32_t * ids, size_t n_ids, int n_expert) const {
    const auto it = index_.layers.find(layer);
    return moe_stream_fully_staged(it == index_.layers.end() ? nullptr : &it->second,
                                   moe_stream_ids_bitmap(ids, n_ids, n_expert), table_);
}

bool moe_stream_phase::region_open() const {
    return ring_ && ring_->slots_free() < ring_->geometry().region_slots;
}

void moe_stream_phase::layer_done(int layer, sycl::event down_done) {
    if (layer != cur_layer_) {
        return;  // the down op of a layer this micro-batch has not begun: nothing of it is in the ring
    }
    close_region(down_done);
}

void moe_stream_phase::end_prompt() {
    if (!ubatch_live_ && !ring_ && table_.empty()) {
        return;  // every decode graph after the first: nothing to drop
    }
    teardown(ring_ ? compute_barrier() : sycl::event());
    layers_.clear();
    index_            = moe_stream_plan_index();
    ubatch_live_      = false;
    cur_layer_        = -1;
    ubatch_seq_       = 0;
    reserve_failed_   = false;
    record_valid_     = false;
    record_looked_up_ = false;
    prepared_         = false;
}

// One scan of the plan per micro-batch (c-7soh). A new geometry, device or compute queue gives the old ring back
// first (ruling D, c-fjx1), after a barrier on the old queue. The tuning-cache record is read once per prompt.
void moe_stream_phase::prepare(const placement_plan & plan, int device, sycl::queue & ccs) {
    index_ = moe_stream_plan_index_build(plan, device);
    const moe_stream_ring_geometry geo =
        moe_stream_ring_size(index_.host_matrices_per_layer_max, index_.matrix_bytes_max, index_.resident_expert_bytes);
    const bool same_queue = ccs_.has_value() && *ccs_ == ccs;
    const bool same_geo = geo.slot_bytes == geo_.slot_bytes && geo.n_slots == geo_.n_slots && geo.bytes == geo_.bytes &&
                          geo.region_slots == geo_.region_slots;
    if (prepared_ && (device != device_ || !same_queue || !same_geo)) {
        teardown(ring_ ? compute_barrier() : sycl::event());
    }
    if (!same_queue) {
        ccs_.emplace(ccs);
        in_order_ = ccs.is_in_order();
    }
    // The SOA slots of a reused region are overwritten after its readers only because both run in order on this
    // queue; on any other queue the ring is not offered.
    if (!in_order_ && !warned_not_in_order_) {
        GGML_LOG_WARN("[MOE-EXEC] device %d: the compute queue is not in order, so the stream ring is unavailable\n",
                      device);
        warned_not_in_order_ = true;
    }
    device_   = device;
    geo_      = geo;
    prepared_ = true;
    if (!record_looked_up_) {
        record_valid_     = ggml_backend_sycl_moe_exec_rates_lookup(device, &record_);
        record_looked_up_ = true;
    }
}

// The ring bytes one layer may use: one region, when a ring of this geometry can exist here.
size_t moe_stream_phase::ring_bytes_free() const {
    if (geo_.n_slots == 0 || !in_order_ || reserve_failed_) {
        return 0;
    }
    return static_cast<size_t>(geo_.region_slots) * geo_.slot_bytes;
}

moe_exec_decision moe_stream_phase::weigh(const moe_stream_layer_info & li,
                                          const moe_stream_op &         op,
                                          double                        window_us) const {
    moe_exec_layer_inputs in;
    in.n_tokens          = n_tokens_;  // the micro-batch's, from begin_ubatch
    in.n_expert          = op.n_expert;
    in.n_expert_used     = op.n_expert_used;
    in.host_expert_bytes = li.host_bytes;
    in.n_host_experts    = li.n_host_experts;
    in.K                 = li.K;
    in.N                 = li.N;
    in.window_us         = window_us;
    in.ring_bytes_free   = ring_bytes_free();
    in.triplet_bytes_max = li.expert_bytes_max;
    const moe_exec_rates r =
        moe_stream_rates_assemble(li.host_mix, record_valid_ ? &record_ : nullptr, moe_stream_getenv);
    return moe_exec_weigh(in, r, op.policy);
}

moe_exec_decision moe_stream_phase::decision_of(int layer) const {
    const auto it = layers_.find(layer);
    if (it != layers_.end() && it->second.decided) {
        return it->second.decision;
    }
    moe_exec_decision d;
    d.reason = moe_exec_reason::NO_HOST_EXPERTS;
    return d;
}

void moe_stream_phase::resolve(int                           layer,
                               const moe_stream_layer_info & li,
                               layer_state &                 st,
                               unified_cache &               cache,
                               const moe_stream_op &         op) {
    stage_counts n;
    // Only an explicit GGML_SYCL_MOE_EXEC=ring moves bytes or reserves the ring: nothing consumes a staged expert
    // until Task 8 (llama.cpp-o9uw) lands, so AUTO only records what the cost model would choose (lead ruling c-3skf).
    if (st.decision.choice == moe_exec_choice::RING && op.policy == moe_exec_policy::RING) {
        n = stage(li, st.decision.regime, cache, op);
        if (n.ring_unavailable) {
            st.decision.choice = moe_exec_choice::CPU;
            st.decision.reason = moe_exec_reason::RING_UNAVAILABLE;
        }
    }
    st.resolved = true;
    report(layer, st, n, op);
}

// Copies each selected matrix's AOS host bytes into one slot of the current region on the copy queue, after the
// region's gate, then reorders it into a second slot on the compute queue, after the copy. Each item the ring cannot
// take directly is counted, never copied another way.
moe_stream_phase::stage_counts moe_stream_phase::stage(const moe_stream_layer_info & li,
                                                       moe_exec_regime               regime,
                                                       unified_cache &               cache,
                                                       const moe_stream_op &         op) {
    stage_counts                       n;
    const std::vector<moe_stream_item> items = moe_stream_select(li, regime, bitmap_);
    if (items.empty()) {
        return n;
    }
    if (!ring_ready()) {
        n.ring_unavailable = true;
        return n;
    }
    // A region holds one layer: close the one in use, then drop the table entries of the layer this region last
    // held. Their slots are rewritten below, after that layer's reorders (the gate) and its readers (the in-order
    // compute queue).
    close_region(region_last_reorder_);
    for (uint64_t key : region_keys_[cur_region_]) {
        table_.erase(key);
    }
    region_keys_[cur_region_].clear();

    const int                                    device = device_;
    std::unordered_map<int, std::pair<int, int>> per_expert;  // expert -> (matrices selected, matrices staged)
    for (const moe_stream_item & item : items) {
        std::pair<int, int> & pe = per_expert[item.expert_id];
        ++pe.first;
        if (!moe_stream_item_reorderable(item, geo_.slot_bytes)) {
            ++n.untyped;
            continue;
        }
        size_t         host_bytes = 0;
        mem_handle     host       = op.host_lease != nullptr ?
                                        op.host_lease(op.graph_ctx, item.layer, item.expert_id, item.role, &host_bytes) :
                                        mem_handle();
        alloc_metadata owner{};
        if (!host.valid() || host_bytes < item.src_bytes ||
            !(host.owner_metadata(&owner) && moe_stream_owner_streamable(owner, device))) {
            ++n.unowned;
            continue;
        }
        mem_handle aos = ring_->acquire_slot();
        mem_handle soa = aos.valid() ? ring_->acquire_slot() : mem_handle();
        if (!soa.valid()) {
            ++n.exhausted;
            continue;
        }
        sycl::event reorder;
        try {
            const sycl::event copy =
                mem_copy_async(aos, host, item.src_bytes, cache.get_bcs_queue(), { ring_->current_region_gate() });
            const uint8_t * aos_ptr = static_cast<const uint8_t *>(aos.resolve(device).ptr);
            uint8_t *       soa_ptr = static_cast<uint8_t *>(soa.resolve(device).ptr);
            reorder = ggml_sycl_reorder_weight_gpu(*ccs_, aos_ptr, soa_ptr, item.type, item.ncols, item.nrows,
                                                   item.src_bytes, { copy });
            retain_handles_until_event({ aos, host }, reorder);
        } catch (const ggml_sycl_fallback_error &) {
            throw;
        } catch (const std::exception & e) {
            GGML_ABORT("[MOE-EXEC] stream enqueue failed: %s", e.what());
        }
        const uint64_t key   = moe_stream_table_key(item.layer, item.expert_id, item.role);
        region_submitted_    = true;
        region_last_reorder_ = reorder;
        anchor_              = soa;
        table_[key]          = { soa, aos, reorder, GGML_LAYOUT_SOA };
        region_keys_[cur_region_].push_back(key);
        n.bytes += item.src_bytes;
        ++pe.second;
    }
    for (const auto & kv : per_expert) {
        ++n.wanted;
        n.staged += kv.second.second == kv.second.first ? 1 : 0;
    }
    return n;
}

// The ring is reserved at the first layer that stages, so a prompt whose every layer runs on the CPU reserves
// nothing. A failed reserve stays failed for the rest of the prompt.
bool moe_stream_phase::ring_ready() {
    if (ring_) {
        return true;
    }
    if (reserve_failed_ || geo_.n_slots == 0 || !in_order_) {
        return false;
    }
    std::unique_ptr<moe_stream_ring> ring = std::make_unique<moe_stream_ring>();
    if (!ring->reserve(device_, geo_)) {
        reserve_failed_ = true;
        return false;
    }
    ring_                = std::move(ring);
    cur_region_          = 0;
    region_submitted_    = false;
    region_last_reorder_ = sycl::event();
    return true;
}

// Ends the current region's use when slots were taken from it. With nothing submitted into it the region keeps its
// older gate, which still guards that region's last use.
void moe_stream_phase::close_region(sycl::event region_done) {
    if (!region_open()) {
        return;
    }
    ring_->epoch_boundary(region_submitted_ ? region_done : ring_->current_region_gate());
    cur_region_       = 1 - cur_region_;
    region_submitted_ = false;
}

sycl::event moe_stream_phase::compute_barrier() {
    try {
        return ccs_->ext_oneapi_submit_barrier();
    } catch (const ggml_sycl_fallback_error &) {
        throw;
    } catch (const std::exception & e) {
        GGML_ABORT("[MOE-EXEC] stream teardown barrier failed: %s", e.what());
    }
}

// Gives the ring back after the device's last use of it (ruling D, c-fjx1): the current region closes on last_use,
// and one slot, which retains the whole reservation, is held until last_use too, because a region closed earlier
// carries a gate that precedes its SOA readers. The table's handles then go.
void moe_stream_phase::teardown(sycl::event last_use) {
    if (ring_) {
        close_region(last_use);
        if (anchor_.valid()) {
            retain_handles_until_event({ anchor_ }, last_use);
        }
        ring_->release();
        ring_.reset();
    }
    anchor_ = mem_handle();
    table_.clear();
    region_keys_[0].clear();
    region_keys_[1].clear();
    cur_region_          = 0;
    region_submitted_    = false;
    region_last_reorder_ = sycl::event();
    geo_                 = moe_stream_ring_geometry();
}

void moe_stream_phase::report(int                   layer,
                              const layer_state &   st,
                              const stage_counts &  n,
                              const moe_stream_op & op) const {
    if (st.decision.choice == moe_exec_choice::NOT_CONSULTED || op.log == nullptr) {
        return;
    }
    if (op.log->trace >= 1 && op.log->trace_sink != nullptr) {
        char line[256];
        std::snprintf(line, sizeof(line),
                      "[MOE-EXEC] trace layer %d ubatch %u: %s regime=%s p=%.2f t_cpu=%.0f t_ring=%.0f window=%.0f "
                      "staged=%d/%d bytes=%zu"
                      " unowned=%d untyped=%d exhausted=%d",
                      layer, ubatch_seq_, moe_exec_choice_name(st.decision.choice),
                      moe_stream_regime_name(st.decision.regime), st.decision.p_touch, st.decision.t_cpu_us,
                      st.decision.t_ring_us, st.window_us, n.staged, n.wanted, n.bytes, n.unowned, n.untyped,
                      n.exhausted);
        op.log->trace_sink(op.log->sink_ctx, line);
    }
    op.log->note(device_, layer, st.decision, n.bytes);
}

}  // namespace ggml_sycl

void ggml_sycl::moe_stream_phase_deleter::operator()(moe_stream_phase * p) const {
    delete p;
}
