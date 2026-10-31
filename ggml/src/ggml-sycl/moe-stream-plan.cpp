// moe_stream_phase: the prompt phase's VRAM ring, stage table and per-layer decisions (llama.cpp-xod7, plan Task 7b of
// llama.cpp-36j0). The pure parts it consults are in moe-stream-plan.hpp; test-moe-stream-phase pins them.
//
// Nothing here waits on the host or asks an event for its status (ruling A, c-3bde): a copy waits on the device for its
// region's gate and its region's read gate (the executor's barrier behind the dequants of the window the region last
// held, or a compute-queue barrier), and a region closes on an event after its last copy.

#include "moe-stream-plan.hpp"

#include "common.hpp"  // moe_stream_phase_deleter
#include "ggml-impl.h"
#include "mem-handle.hpp"
#include "mem-ops.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

// The window timing's mark, an empty kernel (moe_stream_phase::mark_window_start).
class moe_stream_timing_mark;

namespace ggml_sycl {

namespace {

const char * moe_stream_getenv(const char * name) {
    return std::getenv(name);
}

// A mark's or a read gate's device command_end in ns, for the window timing line; false when the event has none (its
// queue was built without profiling, or the runtime refuses the query). Read only at the synchronize boundary, and
// never of a copy: the copy queue has no profiling.
bool moe_stream_event_end(const sycl::event & ev, uint64_t * end_ns) {
    uint64_t start_ns = 0;
    try {
        start_ns = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        *end_ns  = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (const sycl::exception &) {
        return false;
    }
    return start_ns != 0 && *end_ns >= start_ns;
}

}  // namespace

moe_stream_phase::moe_stream_phase() = default;

// The context destructor resets this after draining its queues, so the barrier is already complete; it still orders the
// ring's return after every use for any other owner. As in moe_stream_ring's destructor, a throw from the barrier or
// from retain_handles_until_event() terminates: there is deliberately no catch. A prompt no end_prompt ended (the
// context freed mid-prompt) is reported first; after an end_prompt the tally is empty and nothing prints again. The
// windows no synchronize flushed print too, their queues drained.
moe_stream_phase::~moe_stream_phase() {
    report_staged_routes();
    flush_window_timing();
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
    cur_layer_         = -1;
    ubatch_live_       = true;
    n_tokens_          = n_tokens;
    prev_window_layer_ = -1;
    next_window_id_    = UINT32_MAX;
    next_window_whole_ = false;
    ++ubatch_seq_;  // the window timing's key: a window id repeats in every micro-batch
}

// The decline knob's range (common.hpp) is the decision log's layers: a larger knob names a layer the log cannot
// record.
static_assert(GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX == moe_exec_decision_log::MAX_LAYERS - 1,
              "GGML_SYCL_MOE_EXEC_DECLINE_LAYER_MAX must be the decision log's last layer");

moe_exec_decision moe_stream_phase::begin_layer(int                    layer,
                                                const placement_plan & plan,
                                                int                    device,
                                                unified_cache &        cache,
                                                sycl::queue &          ccs,
                                                const int32_t *        ids,
                                                size_t                 n_ids,
                                                const moe_stream_op &  op) {
    GGML_UNUSED(cache);
    GGML_UNUSED(ids);
    GGML_UNUSED(n_ids);
    const bool new_ubatch = moe_stream_ubatch_starts(ubatch_live_, layer, cur_layer_, op.graph_seq, cur_graph_seq_);
    if (!new_ubatch && layer == cur_layer_) {
        return decision_of(layer);  // the layer's later ops: the first one decided
    }
    if (new_ubatch) {
        begin_ubatch(op.n_tokens);
        prepare(plan, device, ccs, op);
    }
    cur_layer_     = layer;
    cur_graph_seq_ = op.graph_seq;
    log_           = op.log;
    policy_        = op.policy;
    // The window timing, and the timing queue its marks go on, are trace level 2's, until a mark fails to submit.
    window_timing_.enabled = op.log != nullptr && op.log->trace >= 2 && !window_marks_failed_;
    request_loan_  = op.request_loan;
    loan_refusals_ = op.loan_refusals;

    // The layer is weighed at its own first op, never a layer ahead: the executor stages its own windows, and no copy
    // of a layer hides under the previous layer's (there is no cross-layer prefetch), so the cost model gets no window.
    const auto cur = index_.layers.find(layer);
    if (cur != index_.layers.end()) {
        layer_state & st = layers_[layer];
        if (!st.decided) {
            st.decision = weigh(cur->second, op, 0.0);
            st.decided  = true;
            // GGML_SYCL_MOE_EXEC_DECLINE_LAYER forces this layer off the executor (llama.cpp-ny6i): a CPU decision is
            // never staged, so the ring is neither reserved nor claimed for it and the layer runs the hybrid path.
            if (moe_stream_layer_declined(layer, op.decline_layer)) {
                st.decision.choice = moe_exec_choice::CPU;
                st.decision.reason = moe_exec_reason::DECLINED;
            }
            // A RING decision moves bytes under any policy but cpu (moe_stream_stage_now): AUTO when the cost model
            // chose the ring, ring always (llama.cpp-3460). It stands only on a layer the executor claims whole, once
            // the ring reserves at the first such layer: the ring serves the executor only (W2). Every decision is
            // reported.
            // A short reserve sizes its loan from this plan, with the hooks and window prepare() sized the ring with.
            moe_stream_loan_sizer sizer;
            sizer.plan           = &plan;
            sizer.device         = device_;
            sizer.geo            = geo_;
            sizer.prompt_kernel  = op.prompt_kernel;
            sizer.graph_ctx      = op.graph_ctx;
            sizer.library_claims = op.library_claims;
            sizer.hints          = window_;
            if (moe_stream_stage_now(st.decision, op.policy) &&
                !(moe_stream_executor_claims(cur->second, geo_) && ring_ready(&sizer))) {
                st.decision.choice = moe_exec_choice::CPU;
                st.decision.reason = moe_exec_reason::RING_UNAVAILABLE;
            }
            if (st.decision.reason == moe_exec_reason::RING_UNAVAILABLE) {
                warn_ring_unavailable(op, cur->second);
            }
            report(layer, st);
        }
    }
    return decision_of(layer);
}

const moe_stream_staged * moe_stream_phase::find(int layer, int expert_id, expert_tensor_role role) const {
    const auto it = table_.find(moe_stream_table_key(layer, expert_id, role));
    return it == table_.end() ? nullptr : &it->second;
}

// The index lookup comes first: a layer with no host entry answers without touching the ring. A RING decision stood
// only where begin_layer's ring_ready() kept a ring, so ring_ready() here answers from the ring held and never
// reaches a reserve, which is why it is handed no loan sizer.
bool moe_stream_phase::executor_claims(int layer) {
    const auto it = index_.layers.find(layer);
    return it != index_.layers.end() && decision_of(layer).choice == moe_exec_choice::RING &&
           moe_stream_executor_claims(it->second, geo_) && ring_ready(nullptr);
}

// A window the last call prefetched is staged already; any other is staged now, whole, into a region of its own. Then
// the next window goes into the other region, where its copies overlap this window's dequants.
uint32_t moe_stream_phase::window_begin(const moe_stream_window_request & req,
                                        unified_cache &                   cache,
                                        const moe_stream_op &             op,
                                        uint32_t *                        misses) {
    // The ring stages for the executor only: a layer whose RING decision stands and whose every host matrix it claims.
    if (req.layer != cur_layer_ || !executor_claims(req.layer)) {
        GGML_ABORT("[MOE-EXEC] window_begin on layer %d, which executor_claims did not answer true for", req.layer);
    }
    // The host time this call takes, for the window's timing line (trace level 2 only).
    const bool timed   = window_timing_.enabled;
    const auto t_begin = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    const auto li = index_.layers.find(req.layer);  // executor_claims found it

    // The bytes this call copies, the window's own and its prefetch's: note_window() counts them once, here.
    size_t     bytes_staged   = 0;
    // A window larger than its region breaks the ring's sizing: a region holds one executor window.
    const auto stage_or_abort = [&](const std::vector<moe_stream_item> & window, expert_tensor_role role, size_t p) {
        const stage_counts n = stage_window(window, moe_stream_window_id(req.layer, role, p), cache, op);
        report_window(req.layer, role, p, n);
        bytes_staged += n.bytes;
        if (n.exhausted != 0) {
            GGML_ABORT("[MOE-EXEC] window layer %d role %s phase %zu exceeds its region: %d of %zu without a slot",
                       req.layer, expert_tensor_role_name(role), p, n.exhausted, window.size());
        }
        return n;
    };
    const uint32_t                     wid     = moe_stream_window_id(req.layer, req.role, req.phase);
    const std::vector<moe_stream_item> items   = moe_stream_window_items(li->second, req.role, req.experts, req.n);
    uint32_t                           missing = 0;
    for (const moe_stream_item & item : items) {
        missing += find(item.layer, item.expert_id, item.role) == nullptr ? 1u : 0u;
    }
    if (wid == next_window_id_) {
        // Prefetched. A matrix its prefetch could not take (no streamable lease) would meet the same lease again, so it
        // is not asked again: the executor reads that expert another way. A prefetch that took its whole window lost
        // nothing since, because nothing stages between two windows.
        if (next_window_whole_ && missing != 0) {
            GGML_ABORT("[MOE-EXEC] window layer %d role %s phase %zu lost %u prefetched entries", req.layer,
                       expert_tensor_role_name(req.role), req.phase, missing);
        }
    } else if (missing != 0) {
        stage_or_abort(items, req.role, req.phase);  // whole, into the region the last prefetch did not take
    }
    const bool exposed = moe_stream_window_exposed(req.layer, prev_window_layer_);
    if (misses != nullptr) {
        *misses = exposed ? 0 : missing;
    }
    prev_window_layer_ = req.layer;

    // The next phase of this role, or phase 0 of next_role at this role's last phase.
    next_window_id_    = UINT32_MAX;
    next_window_whole_ = false;
    if (req.next != nullptr && req.n_next != 0) {
        const bool                         same_role = req.next_role == expert_tensor_role::UNKNOWN;
        const expert_tensor_role           nrole     = same_role ? req.role : req.next_role;
        const size_t                       nphase    = same_role ? req.phase + 1 : 0;
        const std::vector<moe_stream_item> nitems    = moe_stream_window_items(li->second, nrole, req.next, req.n_next);
        if (!nitems.empty()) {
            const stage_counts n = stage_or_abort(nitems, nrole, nphase);
            next_window_id_      = moe_stream_window_id(req.layer, nrole, nphase);
            next_window_whole_   = n.staged == static_cast<int>(nitems.size());
        }
    }
    if (log_ != nullptr) {
        log_->note_window(exposed, missing, bytes_staged);
    }
    uint32_t staged = 0;
    for (const moe_stream_item & item : items) {
        staged += find(item.layer, item.expert_id, item.role) != nullptr ? 1u : 0u;
    }
    if (timed) {
        const double submit_us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_begin).count();
        window_timing_.begun(moe_stream_window_timing_key(ubatch_seq_, wid), req.layer, req.role, req.phase, exposed,
                             submit_us);
    }
    return staged;
}

// The window's readers ran on the compute queue (c-las5) before `readers_done`, so it is the read gate of the region
// holding the window. A layer other than the current one has nothing in the ring this micro-batch.
void moe_stream_phase::window_done(int layer, expert_tensor_role role, size_t phase, sycl::event readers_done) {
    if (layer != cur_layer_) {
        return;
    }
    region_reads_.window_done(moe_stream_window_id(layer, role, phase), readers_done);
}

// As window_done: a layer other than the current one has no window this micro-batch.
void moe_stream_phase::window_launched(int layer, expert_tensor_role role, size_t phase, uint32_t n) {
    if (!window_timing_.enabled || layer != cur_layer_) {
        return;
    }
    window_timing_.launched(moe_stream_window_timing_key(ubatch_seq_, moe_stream_window_id(layer, role, phase)), n);
}

// The windows' lines go out at WARN through the decision log's sink: llama-bench drops INFO without -v, and the
// trace sink is INFO. The caller has waited for the compute queue, so every mark of a begun window follows only
// finished work, and reading the marks' device times does not wait on compute work.
void moe_stream_phase::flush_window_timing() {
    if (window_timing_.records.empty()) {
        return;
    }
    if (log_ == nullptr || log_->trace < 2 || log_->sink == nullptr) {
        window_timing_.clear();
        return;
    }
    moe_exec_decision_log * log = log_;
    window_timing_.flush(device_, moe_stream_event_end, [log](const char * line) { log->sink(log->sink_ctx, line); });
}

// The queue the window timing's marks go on: in order and profiling, on the compute queue's context and device, so a
// mark can wait on a copy and on a read gate and still carries device times. Not the copy queue, which has no
// profiling, and not the compute queue, whose readers would then wait for the prefetch copies.
sycl::queue & moe_stream_phase::timing_queue() {
    if (!timing_queue_) {
        timing_queue_.emplace(ccs_->get_context(), ccs_->get_device(), default_queue_properties());
    }
    return *timing_queue_;
}

// The window's free and start marks, submitted just before its first copy, so the window's own submit loop does not
// end them late. `deps` is everything that copy waits on except `gate`, its read gate. The free mark also waits for the
// copy queued before, since the copy queue is in order; the start mark adds the gate. A mark that fails to submit is
// caught here, not by stage_window()'s copy handler: drop_window_timing() turns the timing off and the copies go on.
void moe_stream_phase::mark_window_start(uint64_t                           key,
                                         std::vector<sycl::event>           deps,
                                         const std::optional<sycl::event> & gate) {
    try {
        if (window_timing_.queue_tail) {
            deps.push_back(*window_timing_.queue_tail);
        }
        sycl::queue &     q         = timing_queue();
        const sycl::event free_mark = ggml_sycl_submit_marker<moe_stream_timing_mark>(q, deps);
        if (gate) {
            deps.push_back(*gate);
        }
        const sycl::event start_mark = ggml_sycl_submit_marker<moe_stream_timing_mark>(q, deps);
        window_timing_.marked(key, free_mark, start_mark, gate);
    } catch (const sycl::exception & e) {
        drop_window_timing(e);
    }
}

// The window's end mark, behind its last copy. A failure to submit it is caught as the start marks' is.
void moe_stream_phase::mark_window_end(uint64_t key, const sycl::event & last_copy) {
    try {
        window_timing_.ended(key, ggml_sycl_submit_marker<moe_stream_timing_mark>(timing_queue(), { last_copy }));
    } catch (const sycl::exception & e) {
        drop_window_timing(e);
    }
}

// A timing mark, or the timing queue built for it, threw. The marks are a diagnostic that nothing the copies or the
// readers wait on depends on, so the window timing goes off for the rest of this phase, with one warning through the
// decision log's WARN sink, as the phase's other lines go, and the prompt goes on; begin_layer does not turn it back
// on. The windows recorded so far still print at the next flush, with -1 where a mark is missing.
void moe_stream_phase::drop_window_timing(const sycl::exception & e) {
    window_marks_failed_   = true;
    window_timing_.enabled = false;
    if (log_ == nullptr || log_->sink == nullptr) {
        return;
    }
    char line[384];
    std::snprintf(line, sizeof(line),
                  "[MOE-EXEC-TRACE] window timing marks failed on device %d: %s; no more window timing lines from "
                  "this phase",
                  device_, e.what());
    log_->sink(log_->sink_ctx, line);
}

bool moe_stream_phase::region_open() const {
    return ring_ && ring_->slots_free() < ring_->geometry().region_slots;
}

// `down_done` follows the layer's down op: an open region closes on it. The region's read gate is its window's
// (window_done): a region turns over several times per layer, so a layer's event says nothing about which of its
// windows a region holds.
void moe_stream_phase::layer_done(int layer, sycl::event down_done) {
    if (layer != cur_layer_) {
        return;  // the down op of a layer this micro-batch has not begun: nothing of it is in the ring
    }
    close_region(down_done);
}

void moe_stream_phase::end_prompt() {
    report_staged_routes();
    if (!ubatch_live_ && !ring_ && !pending_ring_ && table_.empty()) {
        return;  // every decode graph after the first: nothing to drop
    }
    // The next prompt explains its own unavailable ring, and asks for its own loans (a refusal held only this one).
    warned_unavailable_       = false;
    loan_latch_               = moe_stream_loan_latch();
    // Whether this prompt's ring engaged decides how the backend returns its loan (llama.cpp-i4ny).
    last_prompt_ring_engaged_ = ring_engaged_;
    ring_engaged_             = false;
    prompt_ended_             = true;
    teardown(ring_ ? compute_barrier() : sycl::event());
    layers_.clear();
    index_            = moe_stream_plan_index();
    ubatch_live_      = false;
    cur_layer_        = -1;
    reserve_failed_   = false;
    record_valid_     = false;
    record_looked_up_ = false;
    prepared_         = false;
}

bool moe_stream_phase::last_prompt_ring_engaged() const {
    return last_prompt_ring_engaged_;
}

bool moe_stream_phase::take_prompt_ended() {
    const bool ended = prompt_ended_;
    prompt_ended_    = false;
    return ended;
}

// One scan of the plan per micro-batch (c-7soh), at its first op, which carries the library executor's claim
// (llama.cpp-3460) that decides each matrix's staging and so the ring's geometry (W2). A new geometry, device or
// compute queue gives the old ring back first (ruling D, c-fjx1), after a barrier on the old queue. The tuning-cache
// record is read once per prompt.
void moe_stream_phase::prepare(const placement_plan & plan, int device, sycl::queue & ccs, const moe_stream_op & op) {
    const moe_stream_ring_geometry geo = moe_stream_plan_ring_geometry(plan, device, op.prompt_kernel, op.graph_ctx,
                                                                       op.library_claims, op.window, &index_);
    const bool                     same_queue = ccs_.has_value() && *ccs_ == ccs;
    const bool                     same_geo   = moe_stream_ring_geometry_same(geo, geo_);
    if (prepared_ && (device != device_ || !same_queue || !same_geo)) {
        teardown(ring_ ? compute_barrier() : sycl::event());
    }
    if (!same_queue) {
        ccs_.emplace(ccs);
        timing_queue_.reset();  // built anew at the next mark, on this queue's context and device
        // A region's readers, the executor's dequants, run in order on this queue, and its read gates are barriers on
        // it; on any other queue the ring is not offered, and the ring-unavailable diagnostic says so.
        in_order_ = ccs.is_in_order();
    }
    device_   = device;
    geo_      = geo;
    window_   = op.window;
    prepared_ = true;
    if (!record_looked_up_) {
        record_valid_     = ggml_backend_sycl_moe_exec_rates_lookup(device, &record_);
        record_looked_up_ = true;
    }
}

// The ring bytes one window may use: one region, when a ring of this geometry can exist here.
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
    in.n_tokens             = n_tokens_;  // the micro-batch's, from begin_ubatch
    in.n_expert             = op.n_expert;
    in.n_expert_used        = op.n_expert_used;
    in.host_expert_bytes    = li.host_bytes;
    in.n_host_experts       = li.n_host_experts;
    in.K                    = li.K;
    in.N                    = li.N;
    in.window_us            = window_us;
    in.ring_bytes_free      = ring_bytes_free();
    in.stage_unit_bytes_max = li.matrix_bytes_max;  // one slot: a window is staged a matrix at a time
    in.soa_bytes            = li.soa_bytes;         // 0: every slot is AOS, read in place, so no reorder is priced
    // The device's probed H2D rate (llama.cpp-kp1l), read live and under the probe's mutex: another model's load may
    // run the probe while this prompt weighs. 0 while the device is unresolved or keeps the seed.
    const double         probed_h2d_gbps = ggml_sycl_h2d_probed_gbps(device_);
    const moe_exec_rates r =
        moe_stream_rates_assemble(li.host_mix, record_valid_ ? &record_ : nullptr, probed_h2d_gbps, moe_stream_getenv);
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

// Copies each window matrix's AOS host bytes into a slot of a fresh region on the copy queue, after the region's gate
// and its read gate. Every slot holds AOS, read in place by the executor's dequant kernel (W2): one slot per matrix and
// no reorder. A matrix the ring cannot take directly is skipped, never copied another way: the executor reads it from
// where it lives.
moe_stream_phase::stage_counts moe_stream_phase::stage_window(const std::vector<moe_stream_item> & items,
                                                              uint32_t                             window_id,
                                                              unified_cache &                      cache,
                                                              const moe_stream_op &                op) {
    stage_counts n;
    if (items.empty()) {
        return n;
    }
    GGML_ASSERT(ring_ != nullptr);  // window_begin runs only where executor_claims reserved the ring
    // A region holds one window: close the one in use, then drop the table entries of the window this region last
    // held. Their slots are rewritten below, after that window's copies (the gate) and its dequants (the read gate).
    close_region(region_last_);
    for (uint64_t key : region_keys_[cur_region_]) {
        table_.erase(key);
    }
    region_keys_[cur_region_].clear();

    // A copy below may rewrite a slot that a dequant read in the region's last fill: the slot is read in place, and the
    // gate does not follow its readers, which ran on the compute queue (c-las5). So the copies also wait for the
    // region's read gate (lead ruling c-o9uw-1), the barrier the executor submits behind the dequants of the window the
    // region last held and hands to window_done(). With none recorded (that window never ran, or its window_done is
    // still to come), the copies wait for a compute-queue barrier submitted at the first of them. It follows every
    // reader of the last fill too, because no reader of it is submitted after it: the executor asks for a window
    // (window_begin) only after it has submitted every dequant of the window before, and the region refilled here holds
    // that window or an older one. The read gate is preferred: the barrier also waits for the work queued after the
    // last fill's dequants, so the copies would start later. A region no copy has gone into since the ring was reserved
    // has no reader to wait for.
    const bool                 refill_after_readers = region_reads_.window[cur_region_] != UINT32_MAX;
    std::optional<sycl::event> reuse_gate;
    const uint64_t             timing_key = moe_stream_window_timing_key(ubatch_seq_, window_id);

    const int               device     = device_;
    std::vector<uint64_t> & other_keys = region_keys_[1 - cur_region_];
    for (const moe_stream_item & item : items) {
        if (!moe_stream_item_aos_fits(item, geo_.slot_bytes)) {
            continue;  // a shape one slot does not hold
        }
        size_t         host_bytes = 0;
        mem_handle     host       = op.host_lease != nullptr ?
                                        op.host_lease(op.graph_ctx, item.layer, item.expert_id, item.role, &host_bytes) :
                                        mem_handle();
        alloc_metadata owner{};
        if (!host.valid() || host_bytes < item.src_bytes ||
            !(host.owner_metadata(&owner) && moe_stream_owner_streamable(owner, device))) {
            continue;  // no lease, or a lease the ring cannot copy directly
        }
        if (ring_->slots_free() == 0) {
            ++n.exhausted;
            continue;
        }
        // The slot is acquired with the layout written there, so its tag is the entry's layout (lead ruling c-o9uw-1).
        mem_handle aos = ring_->acquire_slot(MOE_STREAM_RING_SLOT_LAYOUT);
        GGML_ASSERT(aos.valid());
        sycl::event copied;
        try {
            // The lease admits a host entry still being filled (require_ready=false), so the copy waits on the device
            // for that entry's ready event as well as for the region's gate.
            const resolved_ptr       hr   = host.resolve(device);
            std::vector<sycl::event> deps = { ring_->current_region_gate() };
            if (hr.has_ready_event) {
                deps.push_back(hr.ready_event);
            }
            if (refill_after_readers && !reuse_gate) {
                reuse_gate = region_reads_.read_gate[cur_region_] ? *region_reads_.read_gate[cur_region_] :
                                                                    ccs_->ext_oneapi_submit_barrier();
            }
            // Trace level 2: the window's free and start marks, before the gate joins the copy's dependencies.
            if (window_timing_.enabled && n.staged == 0) {
                mark_window_start(timing_key, deps, reuse_gate);
            }
            if (reuse_gate) {
                deps.push_back(*reuse_gate);
            }
            copied = mem_copy_async(aos, host, item.src_bytes, cache.get_bcs_queue(), deps);
            // Read in place: the copy is this submission's last use of the lease and of the slot, and the table's hold
            // on the slot is not this submission's ownership.
            retain_handles_until_event({ aos, host }, copied);
        } catch (const ggml_sycl_fallback_error &) {
            throw;
        } catch (const std::exception & e) {
            GGML_ABORT("[MOE-EXEC] stream enqueue failed: %s", e.what());
        }
        const uint64_t key = moe_stream_table_key(item.layer, item.expert_id, item.role);
        if (table_.find(key) != table_.end()) {
            // The other region holds this matrix too, for a window the executor split another way: the entry moves
            // here, so that region's next refill does not drop it.
            other_keys.erase(std::remove(other_keys.begin(), other_keys.end(), key), other_keys.end());
        }
        window_timing_.copied(timing_key, copied, item.src_bytes);
        region_submitted_ = true;
        region_last_      = copied;
        // The entry's handle is the AOS slot, with the layout it really holds (ruling C, c-3bde), and the reader the
        // index build named for the item, the executor (llama.cpp-3460).
        table_[key]       = { mem_handle(), aos, copied, MOE_STREAM_RING_SLOT_LAYOUT, item.consumer };
        region_keys_[cur_region_].push_back(key);
        n.bytes += item.src_bytes;
        ++n.staged;
    }
    // Trace level 2: the window's end mark, behind its last copy.
    if (window_timing_.enabled && n.staged != 0) {
        mark_window_end(timing_key, region_last_);
    }
    // A copy went in, so the region's last fill is now this window, with no read gate until its window_done.
    if (n.staged != 0) {
        region_reads_.refill(cur_region_, window_id);
    }
    return n;
}

// The ring is reserved at the first layer whose RING decision stands (begin_layer, executor_claims), so a prompt whose
// every layer runs on the CPU reserves nothing. A failed reserve stays failed for the rest of the micro-batch: one
// attempt per micro-batch. A reserve the WEIGHT zone was short of asks the backend for an expert loan at the next graph
// boundary, so the next micro-batch's attempt may succeed, until a loan this prompt asked for comes back refused
// (llama.cpp-3460, c-0tmn). With begin_layer's `sizer` the loan is sized from the plan it leaves, whose ring can be
// wider, and none is asked when no loan converges (moe_stream_loan_size_post_loan). A refused reserve keeps the slots
// it got, and its ring waits in pending_ring_ for the next micro-batch's attempt, which continues from those slots, so
// the room a loan freed in holes a slot wide adds up across attempts instead of being asked for again as one block
// (llama.cpp-am87). The hold lives only while a loan can still fund it: once a loan this prompt asked for comes back
// refused, or the op wires no loan hook, the refused ring gives its slots back at once, because mid-prompt runtime
// allocations prefer the WEIGHT zone too (lead ruling, c-zgrd).
bool moe_stream_phase::ring_ready(const moe_stream_loan_sizer * sizer) {
    if (ring_) {
        return true;
    }
    if (reserve_failed_ || geo_.n_slots == 0 || !in_order_) {
        return false;
    }
    std::unique_ptr<moe_stream_ring> ring =
        pending_ring_ ? std::move(pending_ring_) : std::make_unique<moe_stream_ring>();
    size_t bytes_short = 0;
    if (!ring->reserve(device_, geo_, &bytes_short)) {
        reserve_failed_ = true;
        if (moe_stream_loan_on_short_reserve(loan_latch_, device_, bytes_short, geo_.slot_bytes, request_loan_,
                                             loan_refusals_, sizer) == moe_stream_loan_step::REFUSED) {
            warn_loan_refused(bytes_short);
        }
        // Decided after the loan step, which may latch a REFUSED loan in this very call. A ring not kept gives its
        // slots back as it leaves scope.
        const bool loan_can_follow = !loan_latch_.refused && request_loan_ != nullptr;
        if (loan_can_follow) {
            pending_ring_ = std::move(ring);
        } else {
            warn_partial_released(ring->slots_held());
        }
        return false;
    }
    ring_engaged_ = true;
    ring_         = std::move(ring);
    reset_regions();
    return true;
}

// Both regions start over: the first one current, nothing submitted into either and no slot held, and no last fill, so
// no window and no read gate. A new ring and a torn-down one both start here, so a read gate recorded for one ring is
// never taken by the next one's copies (stage_window()).
void moe_stream_phase::reset_regions() {
    region_keys_[0].clear();
    region_keys_[1].clear();
    cur_region_       = 0;
    region_submitted_ = false;
    region_last_      = sycl::event();
    region_reads_.reset();
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

// Gives the ring back after the device's last use of it (ruling D, c-fjx1): the current region closes on last_use, and
// the ring's release(last_use) holds every slot until last_use too. A region stage_window() closed is gated on its last
// copy, which precedes the region's readers, the executor's dequants, and each slot is its own allocation, so no one
// slot keeps the others (llama.cpp-am87). The table's handles then go. A partial reservation kept for the next attempt
// (ring_ready()) handed nothing out, so its slots go back at once.
void moe_stream_phase::teardown(sycl::event last_use) {
    if (ring_) {
        close_region(last_use);
        ring_->release(last_use);
        ring_.reset();
        ++ring_releases_;
    }
    pending_ring_.reset();
    table_.clear();
    reset_regions();
    geo_ = moe_stream_ring_geometry();
}

// The layer's decision goes to the decision log, its change line and its trace line (moe_exec_decision_log::note). The
// layer stages nothing itself: its bytes are its windows' (note_window).
void moe_stream_phase::report(int layer, const layer_state & st) const {
    if (log_ != nullptr) {
        log_->note(device_, layer, st.decision);
    }
}

// One trace line per window stage, apart from note()'s "[MOE-EXEC-TRACE] dev" line (lead ruling c-xod7-q1): the
// window, and the matrices it copied and their bytes.
void moe_stream_phase::report_window(int layer, expert_tensor_role role, size_t phase, const stage_counts & n) const {
    if (log_ == nullptr || log_->trace < 1 || log_->trace_sink == nullptr) {
        return;
    }
    char line[160];
    std::snprintf(line, sizeof(line), "[MOE-EXEC-TRACE] stage window layer %d role %s phase %zu: %d items %zu bytes",
                  layer, expert_tensor_role_name(role), phase, n.staged, n.bytes);
    log_->trace_sink(log_->sink_ctx, line);
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

void moe_staged_route_tally::count_handoff() {
    ++handed_off;
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

void moe_stream_phase::count_executor_handoff() {
    staged_routes_.count_handoff();
}

// The resolver's STAGED_RING lookups for the prompt that ends, on the trace (llama.cpp-o9uw): without it a ring that
// stages and a route that serves none of it read alike in the stage lines. The executor's handoffs (llama.cpp-3460)
// are counted apart from the refusals: the batched prompt executor reads those slots. A prompt that refused any also
// gets a line naming each refusal's cause and detail (plan Task 8: the trace says why). Under an explicit ring the
// refusals also go out at WARN: a refused expert runs on another route than the one asked for, and its copy is wasted.
// Cleared last, so the destructor after an end_prompt prints nothing.
void moe_stream_phase::report_staged_routes() {
    if ((staged_routes_.served != 0 || staged_routes_.handed_off != 0 || staged_routes_.refused != 0) &&
        log_ != nullptr && log_->trace >= 1 && log_->trace_sink != nullptr) {
        char line[160];
        std::snprintf(line, sizeof(line),
                      "[MOE-EXEC-TRACE] staged route device %d: served=%zu handed_off=%zu refused=%zu", device_,
                      staged_routes_.served, staged_routes_.handed_off, staged_routes_.refused);
        log_->trace_sink(log_->sink_ctx, line);
        if (staged_routes_.refused != 0) {
            char head[64];
            std::snprintf(head, sizeof(head), "[MOE-EXEC-TRACE] staged route refusals device %d: ", device_);
            const std::string causes = head + staged_routes_.refusals_text();
            log_->trace_sink(log_->sink_ctx, causes.c_str());
        }
    }
    if (staged_routes_.refused != 0 && policy_ == moe_exec_policy::RING && log_ != nullptr && log_->sink != nullptr) {
        char warn_head[160];
        std::snprintf(warn_head, sizeof(warn_head),
                      "[MOE-EXEC] staged route refusals device %d: served=%zu handed_off=%zu refused=%zu ", device_,
                      staged_routes_.served, staged_routes_.handed_off, staged_routes_.refused);
        const std::string warn_line = warn_head + staged_routes_.refusals_text();
        log_->sink(log_->sink_ctx, warn_line.c_str());
    }
    staged_routes_ = moe_staged_route_tally();
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
void moe_stream_phase::test_insert_staged(int layer, int expert_id, expert_tensor_role role, moe_stream_staged entry) {
    table_[moe_stream_table_key(layer, expert_id, role)] = std::move(entry);
}

bool moe_stream_phase::test_reserve_ring(int device, const moe_stream_ring_geometry & geo, sycl::queue & ccs) {
    device_         = device;
    geo_            = geo;
    reserve_failed_ = false;
    ccs_.emplace(ccs);
    in_order_ = ccs.is_in_order();
    return ring_ready(nullptr);
}

mem_handle moe_stream_phase::test_acquire_slot(ggml_layout_mode layout) {
    if (!ring_) {
        return mem_handle();
    }
    return ring_->acquire_slot(layout);
}
#endif

// One WARN per prompt saying why the ring is unavailable, at the first layer that answers RING_UNAVAILABLE, under an
// explicit ring or a trace (lead ruling c-kcyf): a ring-unavailable trace line must be explainable from the log. The
// rule is asked of that layer `li`, since whether the executor claims it is the layer's. region_slots_max is the most
// slots one layer's streamable matrices take; the ring holds one executor window (window_slots) per region instead, so
// n_slots is 2 x window_slots. The slots and bytes are the ring the layer asked for, so when the cap refused it
// they are the uncapped request.
void moe_stream_phase::warn_ring_unavailable(const moe_stream_op & op, const moe_stream_layer_info & li) {
    if (warned_unavailable_ || op.log == nullptr || op.log->sink == nullptr ||
        (op.policy != moe_exec_policy::RING && op.log->trace < 1)) {
        return;
    }
    warned_unavailable_ = true;
    const moe_stream_ring_geometry asked =
        geo_.n_slots != 0 ? geo_ : moe_stream_ring_size(index_.window_slots, index_.matrix_bytes_max, SIZE_MAX);
    char line[384];
    std::snprintf(line, sizeof(line),
                  "[MOE-EXEC] device %d: stream ring unavailable: %s (host_matrices_per_layer_max=%u "
                  "region_slots_max=%u window_slots=%u matrix_bytes_max=%zu resident_expert_bytes=%zu n_slots=%u "
                  "bytes=%zu)",
                  device_, moe_stream_ring_unavailable_rule(index_, li, geo_, in_order_, reserve_failed_),
                  index_.host_matrices_per_layer_max, index_.region_slots_max, index_.window_slots,
                  index_.matrix_bytes_max, index_.resident_expert_bytes, asked.n_slots, asked.bytes);
    op.log->sink(op.log->sink_ctx, line);
}

// One WARN per prompt when a loan the short ring asked for comes back REFUSED (llama.cpp-3460), or when no loan could
// be sized because the ring of the plan a loan would leave never settles or does not fit: the helper answers REFUSED
// once and then latches. A refused loan's line goes out under the same gate as the ring-unavailable line (lead ruling
// c-kcyf), since at default verbosity under AUTO the loan transaction's own "[MOE-EXEC] loan device %d refused: %s"
// line, which the latch makes print once per prompt, is already the one WARN for it. A loan never sized asks the
// transaction nothing, so nothing else would say why the prompt runs without its ring: that line goes out under the
// transaction line's condition instead, every policy and verbosity, once per prompt.
void moe_stream_phase::warn_loan_refused(size_t bytes_short) const {
    if (log_ == nullptr || log_->sink == nullptr) {
        return;
    }
    char line[320];
    if (loan_latch_.unconverged) {
        // The reason must hold in every state the latch can carry: a sizer with no plan or no ring takes no step and
        // leaves 0 slots without any cap having been met, so the step count is asked before the slot count.
        const char * why = loan_latch_.sizing_steps == 0 ? "no sizing step ran, no plan or no ring" :
                           loan_latch_.post_slots == 0   ? "over the cap" :
                                                           "it did not grow, or the sizing spent its bound";
        std::snprintf(line, sizeof(line),
                      "[MOE-EXEC] device %d: stream ring %.1f MiB short and no loan fits the ring it would leave after "
                      "%u sizing steps (post-loan ring %u slots, %zu bytes: %s): no loan this prompt",
                      device_, bytes_short / (1024.0 * 1024.0), loan_latch_.sizing_steps, loan_latch_.post_slots,
                      loan_latch_.post_bytes, why);
    } else {
        if (policy_ != moe_exec_policy::RING && log_->trace < 1) {
            return;
        }
        std::snprintf(line, sizeof(line),
                      "[MOE-EXEC] device %d: stream ring %.1f MiB short and its expert loan refused: no further loan "
                      "this prompt",
                      device_, bytes_short / (1024.0 * 1024.0));
    }
    log_->sink(log_->sink_ctx, line);
}

// A WARN when a refused ring gives its slots back because no loan can fund it any more (lead ruling, c-zgrd), so the
// ring's own "(holding K of N slots)" line before it is explained. It goes out under the same gate as the
// ring-unavailable line (lead ruling c-kcyf), and not for a ring that held no slot.
void moe_stream_phase::warn_partial_released(uint32_t held) const {
    if (held == 0 || log_ == nullptr || log_->sink == nullptr ||
        (policy_ != moe_exec_policy::RING && log_->trace < 1)) {
        return;
    }
    char line[160];
    std::snprintf(line, sizeof(line),
                  "[MOE-STREAM-RING] partial reservation released on device %d: no loan can follow (held %u of %u "
                  "slots)",
                  device_, held, geo_.n_slots);
    log_->sink(log_->sink_ctx, line);
}

}  // namespace ggml_sycl

void ggml_sycl::moe_stream_phase_deleter::operator()(moe_stream_phase * p) const {
    delete p;
}
