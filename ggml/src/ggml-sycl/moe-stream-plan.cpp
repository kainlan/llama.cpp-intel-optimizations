// moe_stream_phase: the prompt phase's VRAM ring, stage table and per-layer decisions (llama.cpp-xod7, plan Task 7b of
// llama.cpp-36j0). The pure parts it consults are in moe-stream-plan.hpp; test-moe-stream-phase pins them.
//
// Nothing here waits on the host or asks an event for its status (ruling A, c-3bde): a copy waits for its region's
// gate on the device (and, around an AOS-only slot, for a compute-queue barrier), a reorder for its copy, and a region
// closes on an event after its last use.

#include "moe-stream-plan.hpp"

#include "common.hpp"  // moe_stream_phase_deleter
#include "ggml-impl.h"
#include "mem-handle.hpp"
#include "mem-ops.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // A reserve refused in the last micro-batch is tried once more in this one: an expert loan (ruling c-tdra,
    // llama.cpp-mwmr) may have made room in the WEIGHT zone since (lead ruling c-i8rz).
    reserve_failed_ = false;
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
    const bool new_ubatch = moe_stream_ubatch_starts(ubatch_live_, layer, cur_layer_, op.graph_seq, cur_graph_seq_);
    if (!new_ubatch && layer == cur_layer_) {
        return decision_of(layer);  // the layer's later ops: the first one decided
    }
    const std::chrono::steady_clock::time_point now       = std::chrono::steady_clock::now();
    double                                      window_us = 0.0;
    if (new_ubatch) {
        begin_ubatch(op.n_tokens);
        prepare(plan, device, ccs, op);
    } else {
        window_us = std::chrono::duration<double, std::micro>(now - last_layer_tp_).count();
    }
    last_layer_tp_ = now;
    cur_layer_     = layer;
    cur_graph_seq_ = op.graph_seq;
    log_           = op.log;
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
    // ROUTED_ONLY ring needs its own ids and stages at its own first op; a CPU decision is only logged. The decision log
    // holds layers below MAX_LAYERS, and ggml_sycl_moe_stream_prepare bounds only the current layer.
    const int  next = layer + 1;
    const auto nxt  = next < moe_exec_decision_log::MAX_LAYERS ? index_.layers.find(next) : index_.layers.end();
    if (nxt != index_.layers.end() && op.layer_present != nullptr && op.layer_present(op.graph_ctx, next)) {
        layer_state & ns = layers_[next];
        if (!ns.decided) {
            ns.decision  = weigh(nxt->second, op, window_us);
            ns.window_us = window_us;
            ns.decided   = true;
            if (moe_stream_resolve_ahead(ns.decision)) {
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

// `down_done` follows every kernel that read the slots of `layer`: they ran on this queue (c-las5) before its down op
// ended. So it becomes the read gate of the region holding `layer` (lead ruling c-o9uw-1), and an open region closes on
// it.
void moe_stream_phase::layer_done(int layer, sycl::event down_done) {
    if (layer != cur_layer_) {
        return;  // the down op of a layer this micro-batch has not begun: nothing of it is in the ring
    }
    for (int r = 0; r < 2; ++r) {
        if (region_layer_[r] == layer) {
            region_read_gate_[r] = down_done;
        }
    }
    close_region(down_done);
}

void moe_stream_phase::end_prompt() {
    report_staged_routes();
    if (!ubatch_live_ && !ring_ && table_.empty()) {
        return;  // every decode graph after the first: nothing to drop
    }
    // The next prompt explains its own unavailable ring.
    warned_unavailable_ = false;
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

// One scan of the plan per micro-batch (c-7soh), at its first op, which carries the device's prompt-kernel predicate
// (c-hmey) that decides each matrix's staging and so the ring's geometry. A new geometry, device or compute queue gives
// the old ring back first (ruling D, c-fjx1), after a barrier on the old queue. The tuning-cache record is read once
// per prompt.
void moe_stream_phase::prepare(const placement_plan & plan, int device, sycl::queue & ccs, const moe_stream_op & op) {
    index_ = moe_stream_plan_index_build(plan, device, op.prompt_kernel, op.graph_ctx, op.library_claims);
    const moe_stream_ring_geometry geo =
        moe_stream_ring_size(index_.region_slots_max, index_.matrix_bytes_max, index_.resident_expert_bytes);
    const bool same_queue = ccs_.has_value() && *ccs_ == ccs;
    const bool same_geo = geo.slot_bytes == geo_.slot_bytes && geo.n_slots == geo_.n_slots && geo.bytes == geo_.bytes &&
                          geo.region_slots == geo_.region_slots;
    if (prepared_ && (device != device_ || !same_queue || !same_geo)) {
        teardown(ring_ ? compute_barrier() : sycl::event());
    }
    if (!same_queue) {
        ccs_.emplace(ccs);
        // The SOA slots of a reused region are overwritten after its readers only because both run in order on this
        // queue; on any other queue the ring is not offered, and the ring-unavailable diagnostic says so.
        in_order_ = ccs.is_in_order();
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
    in.triplet_bytes_max = li.expert_bytes_max;  // the largest streamable triplet: a region holds no other kind
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
    // Only an explicit GGML_SYCL_MOE_EXEC=ring moves bytes or reserves the ring: AUTO only records what the cost model
    // would choose (lead ruling c-3skf).
    if (moe_stream_stage_now(st.decision, op.policy)) {
        n = stage(li, st.decision.regime, cache, op);
        if (n.ring_unavailable) {
            st.decision.choice = moe_exec_choice::CPU;
            st.decision.reason = moe_exec_reason::RING_UNAVAILABLE;
        }
    }
    if (st.decision.reason == moe_exec_reason::RING_UNAVAILABLE) {
        warn_ring_unavailable(op);
    }
    st.resolved = true;
    report(layer, st, n, op);
}

// Copies each selected matrix's AOS host bytes into a slot of the current region on the copy queue, after the
// region's gate. A matrix staged as SOA is then reordered into a second slot on the compute queue, after the copy; an
// AOS-only one (c-hmey) is read in place. Each item the ring cannot take directly is counted, never copied another way.
moe_stream_phase::stage_counts moe_stream_phase::stage(const moe_stream_layer_info & li,
                                                       moe_exec_regime               regime,
                                                       unified_cache &               cache,
                                                       const moe_stream_op &         op) {
    stage_counts                 n;
    std::vector<moe_stream_item> items = moe_stream_select(li, regime, bitmap_);
    if (items.empty()) {
        return n;
    }
    if (!ring_ready()) {
        n.ring_unavailable = true;
        return n;
    }
    // A region holds one layer: close the one in use, then drop the table entries of the layer this region last
    // held. Their slots are rewritten below, after that layer's reorders (the gate) and its readers (the in-order
    // compute queue, and the barrier below).
    close_region(region_last_);
    for (uint64_t key : region_keys_[cur_region_]) {
        table_.erase(key);
    }
    region_keys_[cur_region_].clear();

    // AOS-only items first. The region's last submission is then a reorder when it holds a SOA item, and that reorder
    // follows its own copy, which the in-order copy queue runs after every copy before it: the event the region closes
    // on follows every copy and every reorder into it. With no SOA item it is the last copy, which follows the rest.
    const auto aos_only_item = [](const moe_stream_item & it) {
        return it.staging == moe_stream_staging::AOS_ONLY;
    };
    std::stable_partition(items.begin(), items.end(), aos_only_item);
    // With an AOS-only item in the region's last use or in this one, a copy below may rewrite a slot that a kernel read
    // in the last use: an AOS-only slot is read in place, and a pair's SOA slot no longer sits at a fixed odd index.
    // Those readers ran on the compute queue (c-las5), and the gate does not follow them: under STREAM_ALL, layer_done
    // closes the region that holds the NEXT layer, so a region's gate precedes its own layer's readers. So the copies
    // also wait for the region's read gate (lead ruling c-o9uw-1): the barrier layer_done submitted behind the down op
    // of the layer the region last held, which follows every kernel that read the region.
    // The backend skips layer_done while no region is open. With no read gate recorded, the copies wait for a
    // compute-queue barrier submitted at the first of them. It follows every reader of the last use as well, because
    // no reader is submitted after it:
    // - each reader is a kernel of its own layer's MUL_MAT_ID, dispatched on this queue in node order (a prompt
    //   MUL_MAT_ID is never recorded into a command graph, and the backend reorders no nodes);
    // - the last use held a layer below the one whose first op runs this stage, or a layer of an earlier micro-batch.
    // The read gate is preferred: that barrier also waits for the work queued after the last use's down op, the next
    // layer's attention among it, so the copies would start later.
    const bool reuse_after_readers =
        region_aos_only_[cur_region_] || std::any_of(items.begin(), items.end(), aos_only_item);
    std::optional<sycl::event> reuse_gate;
    bool                       aos_only_staged = false;
    bool                       copied_in       = false;

    const int                                    device = device_;
    std::unordered_map<int, std::pair<int, int>> per_expert;  // expert -> (matrices selected, matrices staged)
    for (const moe_stream_item & item : items) {
        std::pair<int, int> & pe = per_expert[item.expert_id];
        ++pe.first;
        const bool aos_only = item.staging == moe_stream_staging::AOS_ONLY;
        if (item.staging == moe_stream_staging::NONE ||
            !(aos_only ? moe_stream_item_aos_fits(item, geo_.slot_bytes) :
                         moe_stream_item_reorderable(item, geo_.slot_bytes))) {
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
        // A SOA item takes two slots, AOS then SOA, an AOS-only item one. An item is taken whole or not at all, so an
        // exhausted region never leaves an AOS slot without its SOA partner.
        if (ring_->slots_free() < moe_stream_staging_slots(item.staging)) {
            ++n.exhausted;
            continue;
        }
        // Each slot is acquired with the layout written there, so its tag is the entry's layout (lead ruling c-o9uw-1).
        mem_handle aos = ring_->acquire_slot(GGML_LAYOUT_AOS);
        mem_handle soa = aos_only ? mem_handle() : ring_->acquire_slot(GGML_LAYOUT_SOA);
        GGML_ASSERT(aos.valid() && soa.valid() != aos_only);
        sycl::event reorder;
        sycl::event copied;
        try {
            // The lease admits a host entry still being filled (require_ready=false), so the copy waits on the device
            // for that entry's ready event as well as for the region's gate.
            const resolved_ptr       hr   = host.resolve(device);
            std::vector<sycl::event> deps = { ring_->current_region_gate() };
            if (hr.has_ready_event) {
                deps.push_back(hr.ready_event);
            }
            if (reuse_after_readers && !reuse_gate) {
                reuse_gate = region_read_gate_[cur_region_] ? *region_read_gate_[cur_region_] :
                                                              ccs_->ext_oneapi_submit_barrier();
            }
            if (reuse_gate) {
                deps.push_back(*reuse_gate);
            }

            const sycl::event copy = mem_copy_async(aos, host, item.src_bytes, cache.get_bcs_queue(), deps);
            if (aos_only) {
                // Read in place: the copy is this submission's last use of the lease and of the slot, and the table's
                // hold on the slot is not this submission's ownership.
                retain_handles_until_event({ aos, host }, copy);
                copied = copy;
            } else {
                const uint8_t * aos_ptr = static_cast<const uint8_t *>(aos.resolve(device).ptr);
                uint8_t *       soa_ptr = static_cast<uint8_t *>(soa.resolve(device).ptr);
                reorder = ggml_sycl_reorder_weight_gpu(*ccs_, aos_ptr, soa_ptr, item.type, item.ncols, item.nrows,
                                                       item.src_bytes, { copy });
                // Every handle the two submissions touch lives until the reorder: the table's hold on the SOA slot is
                // not this submission's ownership.
                retain_handles_until_event({ aos, host, soa }, reorder);
            }
        } catch (const ggml_sycl_fallback_error &) {
            throw;
        } catch (const std::exception & e) {
            GGML_ABORT("[MOE-EXEC] stream enqueue failed: %s", e.what());
        }
        const uint64_t key = moe_stream_table_key(item.layer, item.expert_id, item.role);
        region_submitted_  = true;
        copied_in          = true;
        anchor_            = aos;  // any slice holds the whole reservation
        if (aos_only) {
            // The entry's handle is the AOS slot, with the layout it really holds (ruling C, c-3bde).
            region_last_    = copied;
            aos_only_staged = true;
            table_[key]     = { mem_handle(), aos, copied, GGML_LAYOUT_AOS };
            ++n.aos_only;
        } else {
            region_last_ = reorder;
            table_[key]  = { soa, aos, reorder, GGML_LAYOUT_SOA };
            ++n.soa;
        }
        region_keys_[cur_region_].push_back(key);
        n.bytes += item.src_bytes;
        ++pe.second;
    }
    // A copy went in, so the region's last use is now this one: its layer, whether it left an AOS-only slot, and no
    // read gate until layer_done records one behind this layer's readers.
    if (copied_in) {
        region_aos_only_[cur_region_] = aos_only_staged;
        region_layer_[cur_region_]    = items.front().layer;
        region_read_gate_[cur_region_].reset();
    }
    for (const auto & kv : per_expert) {
        ++n.wanted;
        n.staged += kv.second.second == kv.second.first ? 1 : 0;
    }
    return n;
}

// The ring is reserved at the first layer that stages, so a prompt whose every layer runs on the CPU reserves
// nothing. A failed reserve stays failed for the rest of the micro-batch: one attempt per micro-batch.
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
    ring_ = std::move(ring);
    reset_regions();
    return true;
}

// Both regions start over: the first one current, nothing submitted into either and no slot held, and no last use, so
// no AOS-only slot, no layer and no read gate. A new ring and a torn-down one both start here, so a read gate recorded
// for one ring is never taken by the next one's copies (stage()).
void moe_stream_phase::reset_regions() {
    region_keys_[0].clear();
    region_keys_[1].clear();
    cur_region_         = 0;
    region_submitted_   = false;
    region_last_        = sycl::event();
    region_aos_only_[0] = false;
    region_aos_only_[1] = false;
    region_layer_[0]    = -1;
    region_layer_[1]    = -1;
    region_read_gate_[0].reset();
    region_read_gate_[1].reset();
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
        ++ring_releases_;
    }
    anchor_ = mem_handle();
    table_.clear();
    reset_regions();
    geo_ = moe_stream_ring_geometry();
}

void moe_stream_phase::report(int                   layer,
                              const layer_state &   st,
                              const stage_counts &  n,
                              const moe_stream_op & op) const {
    if (st.decision.choice == moe_exec_choice::NOT_CONSULTED || op.log == nullptr) {
        return;
    }
    // The stage line: the plan's fields under a prefix of its own, apart from note()'s "[MOE-EXEC-TRACE] dev" line
    // (plan amendment for Task 15, lead ruling c-xod7-q1).
    if (op.log->trace >= 1 && op.log->trace_sink != nullptr) {
        char line[256];
        std::snprintf(line, sizeof(line),
                      "[MOE-EXEC-TRACE] stage layer %d ubatch %u: %s regime=%s p=%.2f t_cpu=%.0f t_ring=%.0f "
                      "window=%.0f staged=%d/%d bytes=%zu"
                      " unowned=%d untyped=%d exhausted=%d",
                      layer, ubatch_seq_, moe_exec_choice_name(st.decision.choice),
                      moe_stream_regime_name(st.decision.regime), st.decision.p_touch, st.decision.t_cpu_us,
                      st.decision.t_ring_us, st.window_us, n.staged, n.wanted, n.bytes, n.unowned, n.untyped,
                      n.exhausted);
        op.log->trace_sink(op.log->sink_ctx, line);
        // The staged matrices split by the layout their slot holds, so a census can tell which path each layer took
        // (lead ruling c-o9uw-1, c-9wgo). A line of its own: the stage line keeps its bytes.
        char split[128];
        std::snprintf(split, sizeof(split), "[MOE-EXEC-TRACE] stage layouts layer %d ubatch %u: soa=%d aos_only=%d",
                      layer, ubatch_seq_, n.soa, n.aos_only);
        op.log->trace_sink(op.log->sink_ctx, split);
    }
    op.log->note(device_, layer, st.decision, n.bytes);
}

void moe_staged_route_tally::count(moe_staged_route_refusal refusal, const char * detail) {
    if (refusal == moe_staged_route_refusal::NONE) {
        ++served;
        return;
    }
    ++refused;
    for (refusal_count & c : refusals) {
        const bool same_detail =
            c.detail == detail || (c.detail != nullptr && detail != nullptr && std::strcmp(c.detail, detail) == 0);
        if (c.refusal == refusal && same_detail) {
            ++c.n;
            return;
        }
    }
    refusal_count c;
    c.refusal = refusal;
    c.detail  = detail;
    c.n       = 1;
    refusals.push_back(c);
}

std::string moe_staged_route_tally::refusals_text() const {
    std::string out;
    for (const refusal_count & c : refusals) {
        char item[128];
        if (c.detail != nullptr && c.detail[0] != '\0') {
            std::snprintf(item, sizeof(item), "%s(%s)=%zu", moe_staged_route_refusal_name(c.refusal), c.detail, c.n);
        } else {
            std::snprintf(item, sizeof(item), "%s=%zu", moe_staged_route_refusal_name(c.refusal), c.n);
        }
        if (!out.empty()) {
            out += ' ';
        }
        out += item;
    }
    return out;
}

void moe_stream_phase::count_staged_route(moe_staged_route_refusal refusal, const char * detail) {
    staged_routes_.count(refusal, detail);
}

// The resolver's STAGED_RING lookups for the prompt that ends, on the trace (llama.cpp-o9uw): without it a ring that
// stages and a route that serves none of it read alike in the stage lines. A prompt that refused any also gets a line
// naming each refusal's cause and detail (plan Task 8: the trace says why). Cleared for the next prompt.
void moe_stream_phase::report_staged_routes() {
    if ((staged_routes_.served != 0 || staged_routes_.refused != 0) && log_ != nullptr && log_->trace >= 1 &&
        log_->trace_sink != nullptr) {
        char line[160];
        std::snprintf(line, sizeof(line), "[MOE-EXEC-TRACE] staged route device %d: served=%zu refused=%zu", device_,
                      staged_routes_.served, staged_routes_.refused);
        log_->trace_sink(log_->sink_ctx, line);
        if (staged_routes_.refused != 0) {
            char head[64];
            std::snprintf(head, sizeof(head), "[MOE-EXEC-TRACE] staged route refusals device %d: ", device_);
            const std::string causes = head + staged_routes_.refusals_text();
            log_->trace_sink(log_->sink_ctx, causes.c_str());
        }
    }
    staged_routes_ = moe_staged_route_tally();
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void moe_stream_phase::test_insert_staged(int layer, int expert_id, expert_tensor_role role, moe_stream_staged entry) {
    table_[moe_stream_table_key(layer, expert_id, role)] = std::move(entry);
}
#endif

// One WARN per prompt saying why the ring is unavailable, at the first layer that answers RING_UNAVAILABLE, under an
// explicit ring or a trace (lead ruling c-kcyf): a ring-unavailable trace line must be explainable from the log. The
// slots and bytes are the ring the layer asked for, so when the 10% bound refused it they are the uncapped request.
void moe_stream_phase::warn_ring_unavailable(const moe_stream_op & op) {
    if (warned_unavailable_ || op.log == nullptr || op.log->sink == nullptr ||
        (op.policy != moe_exec_policy::RING && op.log->trace < 1)) {
        return;
    }
    warned_unavailable_ = true;
    const moe_stream_ring_geometry asked =
        geo_.n_slots != 0 ? geo_ : moe_stream_ring_size(index_.region_slots_max, index_.matrix_bytes_max, SIZE_MAX);
    char line[320];
    std::snprintf(line, sizeof(line),
                  "[MOE-EXEC] device %d: stream ring unavailable: %s (host_matrices_per_layer_max=%u "
                  "matrix_bytes_max=%zu resident_expert_bytes=%zu n_slots=%u bytes=%zu)",
                  device_, moe_stream_ring_unavailable_rule(index_, geo_, in_order_, reserve_failed_),
                  index_.host_matrices_per_layer_max, index_.matrix_bytes_max, index_.resident_expert_bytes,
                  asked.n_slots, asked.bytes);
    op.log->sink(op.log->sink_ctx, line);
}

}  // namespace ggml_sycl

void ggml_sycl::moe_stream_phase_deleter::operator()(moe_stream_phase * p) const {
    delete p;
}
