// llama.cpp-xod7 (plan Task 7b of llama.cpp-36j0): the prompt phase's pure parts, the ones moe_stream_phase consults
// before it touches a queue. Host-only: hand-built plans, no queue, no device, and no SYCL object is constructed.
//
// The plan: 5 layers x 8 experts, Q4_0 gate/up/down, resident on device 0 unless noted.
//   layers 0-2: experts 1, 5 and 6 on the host;
//   layer 1:    also a fused host GATE_UP entry for expert 1 (a shape the ring cannot stage);
//   layer 3:    no host expert (the phase skips it);
//   layer 4:    only a fused host GATE_UP for expert 2 (host experts present, no item).
//
// A second plan mixes in host experts of a type with no AOS->SOA reorder (IQ3_XXS).
//
// The ring serves the batched prompt executor only (W2 of docs/plans/2026-10-24-prefill-window-ring.md). A matrix whose
// weight the executor claims once its host experts are staged takes one AOS slot, which the executor's dequant kernel
// reads, and its entry names the executor as its consumer (llama.cpp-3460). A matrix the executor does not claim stages
// NONE, whatever the device's prompt kernels read, and the prompt-kernel predicate is no longer asked. The tests pass
// the claims as a predicate; the backend answers with ggml_sycl_moe_pp_onednn_batched_claims_staged_tensor.
//
// The phase stages a layer a window at a time: one executor phase of one role (moe_stream_window_items). The regions'
// read gates are kept by moe_stream_region_reads, keyed by the window each region holds, which the phase holds over
// sycl::event; the test drives it over int, in the order stage_window(), window_done() and reset_regions() call it.
//
// A short ring reserve asks for an expert loan (llama.cpp-3460, c-0tmn): the test plays ring_ready() over a reserve
// that refuses with a shortfall, with stub hooks for the backend's loan request and its boundary runner's REFUSED
// count, and drives the helper ring_ready() calls, moe_stream_loan_on_short_reserve.

#include "ggml-sycl/moe-stream-plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using ggml_sycl::alloc_metadata;
using ggml_sycl::alloc_role;
using ggml_sycl::alloc_tier;
using ggml_sycl::expert_tensor_role;
using ggml_sycl::host_zone_id;
using ggml_sycl::moe_exec_choice;
using ggml_sycl::moe_exec_decision;
using ggml_sycl::moe_exec_policy;
using ggml_sycl::moe_exec_rates;
using ggml_sycl::moe_exec_regime;
using ggml_sycl::moe_stream_consumer;
using ggml_sycl::moe_stream_item;
using ggml_sycl::moe_stream_layer_info;
using ggml_sycl::moe_stream_plan_index;
using ggml_sycl::moe_stream_staging;
using ggml_sycl::placement_entry;
using ggml_sycl::placement_plan;
using ggml_sycl::placement_priority;

int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr int     kLayers  = 5;
constexpr int     kExperts = 8;
constexpr int64_t kCols    = 256;  // 8 Q4_0 blocks per row
constexpr int64_t kRows    = 16;
constexpr size_t  kFused   = 8192;

const expert_tensor_role kRoles[3] = { expert_tensor_role::GATE, expert_tensor_role::UP, expert_tensor_role::DOWN };
const char *             kNames[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };

size_t src_bytes() {
    return ggml_row_size(GGML_TYPE_Q4_0, kCols) * static_cast<size_t>(kRows);
}

size_t dst_bytes() {
    return src_bytes() + 256;
}

bool triplet_on_host(int layer, int expert) {
    return layer <= 2 && (expert == 1 || expert == 5 || expert == 6);
}

placement_entry entry(int                layer,
                      int                expert,
                      expert_tensor_role role,
                      const char *       name,
                      bool               host,
                      int                device,
                      size_t             src,
                      size_t             dst) {
    placement_entry e("blk." + std::to_string(layer) + "." + name + ".weight", src, dst, 0, placement_priority::MOE_UP,
                      layer, expert, role, !host, host ? -1 : device);
    e.type  = GGML_TYPE_Q4_0;
    e.ne[0] = kCols;
    e.ne[1] = kRows;
    e.ne[2] = kExperts;
    return e;
}

placement_plan make_plan() {
    placement_plan plan{};
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            if (l == 4 && e == 2) {
                plan.entries.push_back(
                    entry(l, e, expert_tensor_role::GATE_UP, "ffn_gate_up_exps", true, 0, kFused, kFused));
                continue;
            }
            for (int r = 0; r < 3; ++r) {
                plan.entries.push_back(
                    entry(l, e, kRoles[r], kNames[r], triplet_on_host(l, e), 0, src_bytes(), dst_bytes()));
            }
        }
    }
    plan.entries.push_back(
        entry(1, 1, expert_tensor_role::GATE_UP, "ffn_gate_up_exps", true, 0, kFused, kFused));  // fused, layer 1
    // Never counted: a dense host tensor, and an expert resident on device 1.
    plan.entries.push_back(entry(1, -1, expert_tensor_role::UNKNOWN, "attn_q", true, 0, src_bytes(), dst_bytes()));
    plan.entries.push_back(entry(3, 0, expert_tensor_role::GATE, "ffn_gate_exps", false, 1, src_bytes(), 999));
    plan.build_index();
    return plan;
}

// Layer 0: experts 0 and 1 on the host as Q4_0 (streamable). Layer 1: experts 0-3 on the host as IQ3_XXS, which has no
// reorder, with a larger device size than any Q4_0 matrix. Experts 4-7 of both layers are resident Q4_0 on device 0.
placement_plan make_mixed_plan() {
    const size_t   iq_src = ggml_row_size(GGML_TYPE_IQ3_XXS, kCols) * static_cast<size_t>(kRows);
    placement_plan plan{};
    for (int e = 0; e < kExperts; ++e) {
        for (int r = 0; r < 3; ++r) {
            const bool q4_host = e < 2;
            plan.entries.push_back(entry(0, e, kRoles[r], kNames[r], q4_host, 0, src_bytes(), dst_bytes()));
            if (e < 4) {
                placement_entry iq = entry(1, e, kRoles[r], kNames[r], true, 0, iq_src, 4 * dst_bytes());
                iq.type            = GGML_TYPE_IQ3_XXS;
                plan.entries.push_back(iq);
            } else {
                plan.entries.push_back(entry(1, e, kRoles[r], kNames[r], false, 0, src_bytes(), dst_bytes()));
            }
        }
    }
    plan.build_index();
    return plan;
}

size_t iq_src_bytes() {
    return ggml_row_size(GGML_TYPE_IQ3_XXS, kCols) * static_cast<size_t>(kRows);
}

// A stand-in no real table matches: Q4_0 read as SOA and as AOS. The index build no longer asks it.
bool kernels_soa_q4(void *, ggml_type type, ggml_layout_mode layout, int64_t, int64_t) {
    return type == GGML_TYPE_Q4_0 && (layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_AOS);
}

// kernels_soa_q4, counting its calls in *graph_ctx.
bool kernels_soa_q4_counted(void * graph_ctx, ggml_type type, ggml_layout_mode layout, int64_t ne0, int64_t ne1) {
    ++*static_cast<int *>(graph_ctx);
    return kernels_soa_q4(nullptr, type, layout, ne0, ne1);
}

// A planned weight slot that holds every expert: the window is the most streamable items one role of a layer has.
const ggml_sycl::moe_stream_window_hints kWideWindow = { SIZE_MAX, 0 };

moe_stream_plan_index index_of(const placement_plan & plan, ggml_sycl::moe_stream_library_claims_fn claims) {
    return ggml_sycl::moe_stream_plan_index_build(plan, 0, kernels_soa_q4, nullptr, claims, kWideWindow);
}

// The library executor's claims, handed to the index build through graph_ctx as the backend's call is.
struct claims_probe {
    int calls = 0;
};

// Claims layer 0's GATE weight and every weight of layer 2, counting its calls.
bool claims_l0_gate_l2(void * graph_ctx, int layer, expert_tensor_role role) {
    ++static_cast<claims_probe *>(graph_ctx)->calls;
    return (layer == 0 && role == expert_tensor_role::GATE) || layer == 2;
}

bool claims_all(void *, int, expert_tensor_role) {
    return true;
}

bool claims_none(void *, int, expert_tensor_role) {
    return false;
}

const moe_stream_layer_info * layer_info(const moe_stream_plan_index & idx, int layer) {
    const auto it = idx.layers.find(layer);
    return it == idx.layers.end() ? nullptr : &it->second;
}

// No layer of `idx` has a byte to reorder: the ring stages AOS slots for the executor only, and moe_stream_staging has
// no SOA value. An index with no layers answers false: a check over nothing would pass vacuously, and every index this
// test builds has layers.
bool no_layer_reorders(const moe_stream_plan_index & idx) {
    for (const auto & kv : idx.layers) {
        if (kv.second.soa_bytes != 0) {
            return false;
        }
    }
    return !idx.layers.empty();
}

// Every item of `li` stages `staging` for `consumer`.
bool items_all(const moe_stream_layer_info & li, moe_stream_staging staging, moe_stream_consumer consumer) {
    for (const moe_stream_item & it : li.items) {
        if (it.staging != staging || it.consumer != consumer) {
            return false;
        }
    }
    return true;
}

const char * fake_env_h2d(const char * name) {
    return std::strcmp(name, "GGML_SYCL_MOE_EXEC_H2D_GBPS") == 0 ? "31" : nullptr;
}

bool near(double a, double b) {
    return std::fabs(a - b) <= 1e-9 * std::fmax(1.0, std::fabs(b));
}

// The backend's side of a short reserve's loan, as the phase's hooks see it: the requests made, and each device's count
// of loans its boundary runner refused, which a case moves to play a REFUSED outcome. Every access goes through a fresh
// read of a volatile pointer, so the compiler cannot prove a check true from the stores before it.
struct loan_stub_state {
    int      requests        = 0;
    int      last_device     = -1;
    size_t   last_bytes      = 0;
    size_t   last_slot_bytes = 0;
    uint64_t refusals[2]     = { 0, 0 };
};

loan_stub_state g_loan;

loan_stub_state * volatile g_loan_at = &g_loan;

void stub_request_loan(int device, size_t bytes, size_t slot_bytes) {
    loan_stub_state & s = *g_loan_at;
    ++s.requests;
    s.last_device     = device;
    s.last_bytes      = bytes;
    s.last_slot_bytes = slot_bytes;
}

uint64_t stub_loan_refusals(int device) {
    return (*g_loan_at).refusals[device];
}

// moe_stream_ring::reserve on a WEIGHT zone `short_by` bytes short of the ring: it refuses and reports the shortfall.
bool stub_reserve_refuses(size_t short_by, size_t * bytes_short) {
    *bytes_short = short_by;
    return false;
}

// The ring's slot, which ring_ready() hands the loan step from its geometry so the loan is counted in slots
// (llama.cpp-9x8t): a matrix_bytes_max that is not a power of two, as a real model's largest host matrix is.
constexpr size_t kLoanSlotBytes = 921600;

// One micro-batch of a prompt whose ring the WEIGHT zone cannot give, as moe_stream_phase::ring_ready() runs it at each
// of `n_layers` staging layers: begin_ubatch cleared the reserve latch, the first layer's reserve refuses and sets it,
// and its reserve-failed branch takes the loan step; the later layers stop at the latch (c-i8rz).
ggml_sycl::moe_stream_loan_step run_short_ubatch(ggml_sycl::moe_stream_loan_latch & latch,
                                                 int                                device,
                                                 size_t                             short_by,
                                                 int                                n_layers) {
    ggml_sycl::moe_stream_loan_step step           = ggml_sycl::moe_stream_loan_step::NONE;
    bool                            reserve_failed = false;
    for (int l = 0; l < n_layers; ++l) {
        if (reserve_failed) {
            continue;
        }
        size_t bytes_short = 0;
        if (!stub_reserve_refuses(short_by, &bytes_short)) {
            reserve_failed = true;
            // No loan sizer: these cases drive the latch alone, so the request is the shortfall as reported. The
            // sizing from the plan a loan leaves is test-moe-stream-plan's.
            step           = ggml_sycl::moe_stream_loan_on_short_reserve(latch, device, bytes_short, kLoanSlotBytes,
                                                                         stub_request_loan, stub_loan_refusals, nullptr);
        }
    }
    return step;
}

// GGML_SYCL_MOE_EXEC_DECLINE_LAYER (llama.cpp-ny6i): begin_layer turns the knob's layer to CPU with reason DECLINED, so
// executor_claims answers false and the layer stays hybrid. The helper is true for that layer only, and never with the
// knob unset (-1).
void test_declined_layer(const moe_stream_layer_info & claimed, const ggml_sycl::moe_stream_ring_geometry & geo) {
    CHECK(ggml_sycl::moe_stream_layer_declined(3, 3));
    CHECK(!ggml_sycl::moe_stream_layer_declined(4, 3));
    CHECK(!ggml_sycl::moe_stream_layer_declined(3, -1));
    CHECK(ggml_sycl::moe_stream_layer_declined(0, 0));
    CHECK(!ggml_sycl::moe_stream_layer_declined(-1, -1));
    // The decline lives in the layer's decision (begin_layer), not in the geometry check: the claim test takes no
    // knob, so a layer it accepts stays accepted whatever the knob names.
    CHECK(ggml_sycl::moe_stream_executor_claims(claimed, geo));
}

// The decision log prints a declined layer through the change line's fixed format, with the reason "declined", which is
// the line the gate's Q_decline arm scores; the summary counts it as a cpu layer-ubatch and not as a forced one.
void capture_line(void * ctx, const char * line) {
    *static_cast<std::string *>(ctx) = line;
}

void test_declined_line() {
    CHECK(std::string(ggml_sycl::moe_exec_reason_name(ggml_sycl::moe_exec_reason::DECLINED)) == "declined");
    std::string                      line;
    ggml_sycl::moe_exec_decision_log log(capture_line, &line);
    moe_exec_decision                d;
    d.choice    = moe_exec_choice::CPU;
    d.reason    = ggml_sycl::moe_exec_reason::DECLINED;
    d.p_touch   = 0.98;
    d.t_cpu_us  = 52000.0;
    d.t_ring_us = 9000.0;
    log.note(0, 3, d);
    CHECK(line == "[MOE-EXEC] layer 3: cpu (p=0.98 t_cpu=52000 us t_ring=9000 us declined)");
    char summary[256] = {};
    CHECK(log.take_summary(summary, sizeof(summary)));
    CHECK(std::strstr(summary, "summary: 0 layer-ubatches ring, 1 cpu, 0 forced,") != nullptr);
}

// The per-window timing the phase keeps under GGML_SYCL_MOE_EXEC_TRACE=2, over int events. The file
// has no fake cache or ring to drive window_begin with, so the test drives the timeline the phase holds over
// sycl::event, in the order the phase feeds it: stage_window() reports each copy (window_begin of phase 0 copies its
// own exposed window, then prefetches phase 1), window_begin reports the window, the executor reports its launches,
// and the synchronize boundary flushes one line per window. The fake profiler answers device submit, start and end
// times in ns; event 4 has none, as a copy on a queue without profiling.
bool fake_event_times(int ev, uint64_t * submit_ns, uint64_t * start_ns, uint64_t * end_ns) {
    switch (ev) {
        case 1:  // window 0's first copy: submitted at 0.5 ms, held by its gate until 0.9 ms, started at 1.0 ms
            *submit_ns = 500000;
            *start_ns  = 1000000;
            *end_ns    = 1400000;
            return true;
        case 2:  // window 0's last copy
            *submit_ns = 500000;
            *start_ns  = 1400000;
            *end_ns    = 2000000;
            return true;
        case 3:  // window 1's first copy: submitted at 1.5 ms, but the copy ahead of it on the queue ran to 2.0 ms
            *submit_ns = 1500000;
            *start_ns  = 2100000;
            *end_ns    = 2300000;
            return true;
        case 5:  // a copy the runtime gave device start and end but no submit time
            *submit_ns = 0;
            *start_ns  = 3000000;
            *end_ns    = 3100000;
            return true;
        case 10:  // window 0's read gate: the dequants of the window its region last held were done at 0.9 ms
            *submit_ns = 100000;
            *start_ns  = 800000;
            *end_ns    = 900000;
            return true;
        case 11:  // window 1's read gate
            *submit_ns = 1000000;
            *start_ns  = 1900000;
            *end_ns    = 2080000;
            return true;
        default:
            return false;
    }
}

void test_window_timing_records_one_record_per_window_only_under_trace() {
    using timeline_t = ggml_sycl::moe_stream_window_timeline<int>;
    std::vector<std::string> lines;
    const auto               sink = [&](const char * line) {
        lines.push_back(line);
    };
    const uint32_t w0 = ggml_sycl::moe_stream_window_id(4, expert_tensor_role::UP, 0);
    const uint32_t w1 = ggml_sycl::moe_stream_window_id(4, expert_tensor_role::UP, 1);
    const uint64_t k0 = ggml_sycl::moe_stream_window_timing_key(7, w0);
    const uint64_t k1 = ggml_sycl::moe_stream_window_timing_key(7, w1);

    // Below trace level 2 the phase leaves the timeline disabled: nothing is recorded and the flush prints nothing.
    // As in the region-reads case, every use goes through a volatile pointer so -O3 cannot fold the checks away.
    timeline_t off;
    timeline_t * volatile off_at = &off;
    const std::optional<int> gate10(10);
    (*off_at).copied(k0, 1, 1000, gate10);
    (*off_at).begun(k0, 4, expert_tensor_role::UP, 0, true, 12.4);
    (*off_at).launched(k0, 3);
    CHECK((*off_at).records.empty());
    CHECK((*off_at).flush(0, fake_event_times, sink) == 0);
    CHECK(lines.empty());

    timeline_t on;
    timeline_t * volatile on_at = &on;
    (*on_at).enabled            = true;
    // window_begin of phase 0: window 0 is not prefetched, so it is staged whole (exposed), then window 1 is prefetched
    // into the other region, behind its own read gate.
    (*on_at).copied(k0, 1, 1000, gate10);
    (*on_at).copied(k0, 2, 1000, gate10);
    (*on_at).copied(k1, 3, 500, std::optional<int>(11));
    (*on_at).copied(k1, 4, 500, std::optional<int>(11));
    (*on_at).begun(k0, 4, expert_tensor_role::UP, 0, true, 12.4);
    (*on_at).launched(k0, 3);  // two dequants and one GEMM group
    // window_begin of phase 1: prefetched, so it copies nothing of its own.
    (*on_at).begun(k1, 4, expert_tensor_role::UP, 1, false, 3.6);
    (*on_at).launched(k1, 2);
    // A prefetch whose window never began (the op fell back first) is recorded but prints no line.
    (*on_at).copied(ggml_sycl::moe_stream_window_timing_key(7, w1 + 1), 9, 500, std::optional<int>());
    CHECK((*on_at).records.size() == 3);

    CHECK((*on_at).flush(0, fake_event_times, sink) == 2);
    CHECK(lines.size() == 2);
    // copy_us spans the first copy's start to the last copy's end (1.0 ms). No copy was queued ahead of window 0, so
    // the first copy could have started at its submit (0.5 ms): readers_wait_us is the gate's end after that (0.4 ms),
    // and start_lag_us is the copy's start after the gate's end (0.1 ms).
    CHECK(lines.size() > 0 &&
          lines[0] ==
              "[MOE-EXEC-TRACE] window device 0 layer 4 role up phase 0 bytes=2000 exposed=1 submit_us=12 "
              "copy_us=1000 readers_wait_us=400 start_lag_us=100 launches=3");
    // The last copy has no device times, so the copy span is unmeasured. Window 0's last copy held the queue until
    // 2.0 ms, after window 1's submit (1.5 ms), so the gate's end (2.08 ms) cost 0.08 ms of waiting, not 0.58 ms; the
    // copy started 0.02 ms after the gate.
    CHECK(lines.size() > 1 &&
          lines[1] ==
              "[MOE-EXEC-TRACE] window device 0 layer 4 role up phase 1 bytes=1000 exposed=0 submit_us=4 "
              "copy_us=-1 readers_wait_us=80 start_lag_us=20 launches=2");
    // The flush empties the timeline: the next boundary prints only what came after it.
    CHECK((*on_at).records.empty());
    CHECK((*on_at).flush(0, fake_event_times, sink) == 0);

    // The same window in the next micro-batch is a record of its own: micro-batch 7's window 0, copied again after the
    // flush and never begun, prints nothing, and micro-batch 8's window 0 prints only its own bytes. A window that
    // copied nothing of its own measures neither span.
    const uint64_t k0_next = ggml_sycl::moe_stream_window_timing_key(8, w0);
    (*on_at).copied(k0, 1, 1000, gate10);
    (*on_at).copied(k0_next, 2, 700, std::optional<int>());
    CHECK((*on_at).records.size() == 2);
    (*on_at).begun(k0_next, 4, expert_tensor_role::UP, 0, true, 1.0);
    (*on_at).begun(ggml_sycl::moe_stream_window_timing_key(8, w1), 4, expert_tensor_role::UP, 1, false, 1.0);
    lines.clear();
    CHECK((*on_at).flush(1, fake_event_times, sink) == 2);
    // No read gate (a region's first fill since the ring was reserved): neither the wait nor the lag is measured.
    CHECK(lines.size() > 0 &&
          lines[0] ==
              "[MOE-EXEC-TRACE] window device 1 layer 4 role up phase 0 bytes=700 exposed=1 submit_us=1 "
              "copy_us=600 readers_wait_us=-1 start_lag_us=-1 launches=0");
    CHECK(lines.size() > 1 &&
          lines[1] ==
              "[MOE-EXEC-TRACE] window device 1 layer 4 role up phase 1 bytes=0 exposed=0 submit_us=1 "
              "copy_us=-1 readers_wait_us=-1 start_lag_us=-1 launches=0");

    // The copy queued ahead has no device times, so when the window's copy could have started is unknown: the wait
    // is unmeasured, while the lag after the gate still is (0.02 ms).
    const uint64_t k_a = ggml_sycl::moe_stream_window_timing_key(9, w0);
    const uint64_t k_b = ggml_sycl::moe_stream_window_timing_key(9, w1);
    (*on_at).copied(k_a, 4, 100, std::optional<int>());
    (*on_at).copied(k_b, 3, 200, std::optional<int>(11));
    (*on_at).begun(k_b, 4, expert_tensor_role::UP, 1, false, 2.0);
    lines.clear();
    CHECK((*on_at).flush(0, fake_event_times, sink) == 1);
    CHECK(lines.size() > 0 &&
          lines[0] ==
              "[MOE-EXEC-TRACE] window device 0 layer 4 role up phase 1 bytes=200 exposed=0 submit_us=2 "
              "copy_us=200 readers_wait_us=-1 start_lag_us=20 launches=0");
    // The flush also forgets the queue's last copy: a window after the boundary measures from its submit alone.
    CHECK(!(*on_at).queue_tail.has_value());

    // A copy with no submit time (0) leaves the wait unmeasured instead of reading as the gate's end (2080 us); the
    // copy span and the lag (0.92 ms) do not use the submit and are still measured.
    const uint64_t k_c = ggml_sycl::moe_stream_window_timing_key(10, w1);
    (*on_at).copied(k_c, 5, 300, std::optional<int>(11));
    (*on_at).begun(k_c, 4, expert_tensor_role::UP, 1, false, 2.0);
    lines.clear();
    CHECK((*on_at).flush(0, fake_event_times, sink) == 1);
    CHECK(lines.size() > 0 &&
          lines[0] ==
              "[MOE-EXEC-TRACE] window device 0 layer 4 role up phase 1 bytes=300 exposed=0 submit_us=2 "
              "copy_us=100 readers_wait_us=-1 start_lag_us=920 launches=0");
}

}  // namespace

int main() {
    // ---- the stage table's key: (layer << 32) | (expert << 8) | role ----
    const uint64_t k = ggml_sycl::moe_stream_table_key(1, 5, expert_tensor_role::UP);
    CHECK(k == ((1ull << 32) | (5ull << 8) | static_cast<uint64_t>(static_cast<uint8_t>(expert_tensor_role::UP))));
    CHECK(k != ggml_sycl::moe_stream_table_key(1, 5, expert_tensor_role::GATE));
    CHECK(k != ggml_sycl::moe_stream_table_key(1, 6, expert_tensor_role::UP));
    CHECK(k != ggml_sycl::moe_stream_table_key(2, 5, expert_tensor_role::UP));

    // ---- the per-micro-batch index: one scan of the plan, every weight claimed ----
    const placement_plan        plan = make_plan();
    const moe_stream_plan_index idx  = index_of(plan, claims_all);
    CHECK(idx.layers.size() == 4);
    CHECK(layer_info(idx, 3) == nullptr);  // no host expert: skipped, never weighed
    CHECK(no_layer_reorders(idx));

    const moe_stream_layer_info * l0 = layer_info(idx, 0);
    CHECK(l0 != nullptr);
    if (l0) {
        CHECK(l0->items.size() == 9);
        const int exp_e[3] = { 1, 5, 6 };
        for (size_t i = 0; i < l0->items.size() && i < 9; ++i) {
            const moe_stream_item & it = l0->items[i];
            CHECK(it.layer == 0 && it.expert_id == exp_e[i / 3] && it.role == kRoles[i % 3]);
            CHECK(it.type == GGML_TYPE_Q4_0 && it.ncols == kCols && it.nrows == kRows);
            CHECK(it.src_bytes == src_bytes());
        }
        // One AOS slot per matrix, read in place by the executor: nothing reorders.
        CHECK(items_all(*l0, moe_stream_staging::AOS_ONLY, moe_stream_consumer::EXECUTOR));
        CHECK(l0->n_host_experts == 3 && l0->n_untriplet == 0 && l0->n_streamable == 9);
        CHECK(l0->region_slots == 9);
        CHECK(l0->role_items[0] == 3 && l0->role_items[1] == 3 && l0->role_items[2] == 3);
        CHECK(l0->host_bytes == 9 * src_bytes());
        CHECK(l0->soa_bytes == 0);
        CHECK(l0->matrix_bytes_max == src_bytes());  // the AOS bytes a slot holds, not the plan's device size
        CHECK(l0->K == kCols && l0->N == kRows);
        CHECK(l0->host_mix.size() == 1 && l0->host_mix[0].first == GGML_TYPE_Q4_0 &&
              l0->host_mix[0].second == 9 * src_bytes());
    }
    const moe_stream_layer_info * l1 = layer_info(idx, 1);
    CHECK(l1 != nullptr);
    if (l1) {
        CHECK(l1->items.size() == 9 && l1->n_untriplet == 1);
        CHECK(l1->n_host_experts == 3);  // expert 1's fused entry is the same expert
        CHECK(l1->host_bytes == 9 * src_bytes() + kFused);
        CHECK(l1->soa_bytes == 0);       // the fused entry is never staged; the claimed items are read in place
    }
    const moe_stream_layer_info * l4 = layer_info(idx, 4);
    CHECK(l4 != nullptr);
    if (l4) {
        CHECK(l4->items.empty() && l4->n_host_experts == 1 && l4->n_untriplet == 1);
        CHECK(l4->soa_bytes == 0);
    }
    CHECK(idx.host_matrices_per_layer_max == 9);
    CHECK(idx.region_slots_max == 9);
    CHECK(idx.matrix_bytes_max == src_bytes());
    CHECK(idx.role_items_max == 3 && idx.window_slots == 3);
    // Resident on device 0: 24 triplet entries per layer for layers 3 and 4 minus the host ones, 15 for layers 0-2.
    const size_t resident_entries = 3 * (3 * 5) + 3 * 8 + 3 * 7;
    CHECK(idx.resident_expert_bytes == resident_entries * dst_bytes());

    // ---- an item the executor does not claim stages NONE, whatever the device's prompt kernels read (W2) ----
    {
        // No claim: every host item is still listed and the cost model still sees every host byte, but none takes a
        // slot, so no window and no ring.
        const moe_stream_plan_index unclaimed = index_of(plan, nullptr);
        CHECK(no_layer_reorders(unclaimed));
        const moe_stream_layer_info * u0 = layer_info(unclaimed, 0);
        CHECK(u0 != nullptr);
        if (u0) {
            CHECK(u0->items.size() == 9 && u0->n_streamable == 0 && u0->region_slots == 0);
            CHECK(items_all(*u0, moe_stream_staging::NONE, moe_stream_consumer::ROUTE));
            CHECK(u0->n_host_experts == 3 && u0->host_bytes == 9 * src_bytes());
        }
        CHECK(unclaimed.host_matrices_per_layer_max == 0 && unclaimed.region_slots_max == 0);
        CHECK(unclaimed.matrix_bytes_max == 0 && unclaimed.role_items_max == 0 && unclaimed.window_slots == 0);
        // A predicate that claims no weight builds the same index.
        const moe_stream_plan_index   refused = index_of(plan, claims_none);
        const moe_stream_layer_info * r0      = layer_info(refused, 0);
        CHECK(r0 != nullptr && items_all(*r0, moe_stream_staging::NONE, moe_stream_consumer::ROUTE));
        CHECK(refused.host_matrices_per_layer_max == 0 && refused.window_slots == 0);

        // The prompt kernels decide nothing and are never asked, claimed or not: the counted predicate reads Q4_0 as
        // SOA, yet a claimed matrix still takes one AOS slot.
        int                         calls = 0;
        const moe_stream_plan_index counted =
            ggml_sycl::moe_stream_plan_index_build(plan, 0, kernels_soa_q4_counted, &calls, claims_all, kWideWindow);
        const moe_stream_layer_info * k0 = layer_info(counted, 0);
        CHECK(k0 != nullptr && items_all(*k0, moe_stream_staging::AOS_ONLY, moe_stream_consumer::EXECUTOR));
        CHECK(counted.region_slots_max == 9 && counted.matrix_bytes_max == src_bytes());
        (void) ggml_sycl::moe_stream_plan_index_build(plan, 0, kernels_soa_q4_counted, &calls, nullptr, kWideWindow);
        CHECK(calls == 0);
        // No predicate at all: the claim still decides.
        const moe_stream_plan_index blind =
            ggml_sycl::moe_stream_plan_index_build(plan, 0, nullptr, nullptr, claims_all, kWideWindow);
        CHECK(blind.host_matrices_per_layer_max == 9 && blind.region_slots_max == 9 && blind.window_slots == 3);

        // The executor's table is the authority, not the MMVQ kernels: a claimed IQ3_XXS matrix, which no kernel of
        // kernels_soa_q4 reads, takes one AOS slot of its own bytes. Unclaimed it takes none.
        const moe_stream_plan_index   iq_free    = index_of(make_mixed_plan(), nullptr);
        const moe_stream_plan_index   iq_claimed = index_of(make_mixed_plan(), claims_all);
        const moe_stream_layer_info * i1         = layer_info(iq_claimed, 1);
        const moe_stream_layer_info * f1         = layer_info(iq_free, 1);
        CHECK(no_layer_reorders(iq_free) && no_layer_reorders(iq_claimed));
        CHECK(i1 != nullptr && f1 != nullptr);
        if (i1 && f1) {
            CHECK(f1->items.size() == 12 && f1->n_streamable == 0 && f1->region_slots == 0);
            CHECK(i1->items.size() == 12 && i1->n_streamable == 12 && i1->region_slots == 12);
            CHECK(items_all(*i1, moe_stream_staging::AOS_ONLY, moe_stream_consumer::EXECUTOR));
            CHECK(f1->n_host_experts == 4 && f1->host_bytes > 0);
        }
        // A slot holds the largest claimed matrix's AOS bytes: the IQ3_XXS device size the plan carries plays no part.
        CHECK(iq_claimed.matrix_bytes_max == std::max(src_bytes(), iq_src_bytes()));
        CHECK(iq_claimed.host_matrices_per_layer_max == 12 && iq_claimed.region_slots_max == 12);
        CHECK(iq_free.host_matrices_per_layer_max == 0 && iq_free.matrix_bytes_max == 0);
    }

    // ---- the claim is asked once per (layer, role), and an unclaimed role of a layer stages NONE ----
    {
        // claims_l0_gate_l2: layer 0's GATE weight and every weight of layer 2. Host experts are 1, 5 and 6.
        claims_probe                probe;
        const moe_stream_plan_index claimed = ggml_sycl::moe_stream_plan_index_build(
            make_plan(), 0, kernels_soa_q4, &probe, claims_l0_gate_l2, kWideWindow);
        CHECK(no_layer_reorders(claimed));
        const moe_stream_layer_info * c0 = layer_info(claimed, 0);
        const moe_stream_layer_info * c1 = layer_info(claimed, 1);
        const moe_stream_layer_info * c2 = layer_info(claimed, 2);
        CHECK(c0 != nullptr && c1 != nullptr && c2 != nullptr);
        if (c0 && c1 && c2) {
            CHECK(c0->items.size() == 9 && c1->items.size() == 9 && c2->items.size() == 9);
            for (const moe_stream_item & it : c0->items) {
                const bool gate = it.role == expert_tensor_role::GATE;
                CHECK(it.staging == (gate ? moe_stream_staging::AOS_ONLY : moe_stream_staging::NONE));
                CHECK(it.consumer == (gate ? moe_stream_consumer::EXECUTOR : moe_stream_consumer::ROUTE));
            }
            CHECK(items_all(*c1, moe_stream_staging::NONE, moe_stream_consumer::ROUTE));
            CHECK(items_all(*c2, moe_stream_staging::AOS_ONLY, moe_stream_consumer::EXECUTOR));
            // Layer 0: the three claimed gates, one slot each; layer 1: nothing; layer 2: all nine.
            CHECK(c0->n_streamable == 3 && c0->region_slots == 3);
            CHECK(c0->role_items[0] == 3 && c0->role_items[1] == 0 && c0->role_items[2] == 0);
            CHECK(c1->n_streamable == 0 && c1->region_slots == 0);
            CHECK(c2->n_streamable == 9 && c2->region_slots == 9);
        }
        CHECK(claimed.host_matrices_per_layer_max == 9 && claimed.region_slots_max == 9 && claimed.role_items_max == 3);
        // Asked once per (layer, role) with a matrix one AOS slot holds, not once per expert: layers 0-2, three roles.
        CHECK(probe.calls == 9);

        // A claimed matrix one AOS slot cannot hold (a zeroed inventory shape) stays NONE, and the claim is not asked
        // for its weight.
        placement_plan zeroed = make_plan();
        for (placement_entry & e : zeroed.entries) {
            if (e.layer_id == 2) {
                e.ne[0] = 0;
            }
        }
        zeroed.build_index();
        claims_probe                zprobe;
        const moe_stream_plan_index z =
            ggml_sycl::moe_stream_plan_index_build(zeroed, 0, kernels_soa_q4, &zprobe, claims_l0_gate_l2, kWideWindow);
        const moe_stream_layer_info * z2 = layer_info(z, 2);
        CHECK(z2 != nullptr && z2->items.size() == 9 && z2->n_streamable == 0 && z2->region_slots == 0);
        if (z2) {
            CHECK(items_all(*z2, moe_stream_staging::NONE, moe_stream_consumer::ROUTE));
        }
        CHECK(zprobe.calls == 6);
    }

    // A positive ring geometry, and the default one (no ring).
    ggml_sycl::moe_stream_ring_geometry no_ring;
    ggml_sycl::moe_stream_ring_geometry ring_geo;
    ring_geo.slot_bytes   = 4096;
    ring_geo.region_slots = 3;
    ring_geo.n_slots      = 6;
    ring_geo.bytes        = 6 * 4096;
    claims_probe                partial_probe;
    // Layer 0 claims only its gates, layer 2 every weight (claims_l0_gate_l2).
    const moe_stream_plan_index partial =
        ggml_sycl::moe_stream_plan_index_build(plan, 0, kernels_soa_q4, &partial_probe, claims_l0_gate_l2, kWideWindow);
    const moe_stream_plan_index unclaimed = index_of(plan, nullptr);

    // ---- the executor takes a layer's weights whole, or not at all (moe_stream_executor_claims) ----
    {
        const moe_stream_layer_info * c0 = layer_info(idx, 0);
        const moe_stream_layer_info * c1 = layer_info(idx, 1);
        const moe_stream_layer_info * c4 = layer_info(idx, 4);
        const moe_stream_layer_info * p0 = layer_info(partial, 0);
        const moe_stream_layer_info * p2 = layer_info(partial, 2);
        const moe_stream_layer_info * u0 = layer_info(unclaimed, 0);
        CHECK(c0 && c1 && c4 && p0 && p2 && u0);
        if (c0 && c1 && c4 && p0 && p2 && u0) {
            // Every host item claimed for the executor, no untriplet entry, and a ring that can exist.
            CHECK(ggml_sycl::moe_stream_executor_claims(*c0, ring_geo));
            CHECK(ggml_sycl::moe_stream_executor_claims(*p2, ring_geo));
            // No ring can exist: the claim is refused whatever the items say.
            CHECK(!ggml_sycl::moe_stream_executor_claims(*c0, no_ring));
            // A host entry the ring cannot stage (layer 1's fused GATE_UP) keeps the whole layer hybrid.
            CHECK(!ggml_sycl::moe_stream_executor_claims(*c1, ring_geo));
            // Host experts with no item (layer 4: only a fused entry).
            CHECK(!ggml_sycl::moe_stream_executor_claims(*c4, ring_geo));
            // One unclaimed role (layer 0's up and down) keeps the layer hybrid, and so does no claim at all.
            CHECK(!ggml_sycl::moe_stream_executor_claims(*p0, ring_geo));
            CHECK(!ggml_sycl::moe_stream_executor_claims(*u0, ring_geo));
            // The geometry-free half, which the ring-unavailable rule asks first.
            CHECK(ggml_sycl::moe_stream_layer_claimed(*c0) && ggml_sycl::moe_stream_layer_claimed(*p2));
            CHECK(!ggml_sycl::moe_stream_layer_claimed(*c1) && !ggml_sycl::moe_stream_layer_claimed(*c4));
            CHECK(!ggml_sycl::moe_stream_layer_claimed(*p0) && !ggml_sycl::moe_stream_layer_claimed(*u0));
            // Each half of an item's claim fails the layer alone: an unstaged item the executor would read, and a
            // staged item the route would read.
            moe_stream_layer_info unstaged = *c0;
            unstaged.items.back().staging  = moe_stream_staging::NONE;
            CHECK(!ggml_sycl::moe_stream_layer_claimed(unstaged));
            moe_stream_layer_info routed = *c0;
            routed.items.back().consumer = moe_stream_consumer::ROUTE;
            CHECK(!ggml_sycl::moe_stream_layer_claimed(routed));
        }
    }

    // ---- GGML_SYCL_MOE_EXEC_DECLINE_LAYER: one layer forced off the executor (llama.cpp-ny6i) ----
    {
        const moe_stream_layer_info * c0 = layer_info(idx, 0);
        CHECK(c0 != nullptr);
        if (c0 != nullptr) {
            test_declined_layer(*c0, ring_geo);
        }
        test_declined_line();
    }

    // ---- why a layer answered RING_UNAVAILABLE: the first rule that holds, in a fixed order ----
    {
        const auto rule = [](const moe_stream_plan_index & ix, const moe_stream_layer_info * li,
                             const ggml_sycl::moe_stream_ring_geometry & g, bool in_order, bool refused) {
            if (li == nullptr) {
                return std::string("<no layer>");
            }
            return std::string(ggml_sycl::moe_stream_ring_unavailable_rule(ix, *li, g, in_order, refused));
        };
        // A layer the executor does not claim wins over every later rule, the ring and the queue included: no claim at
        // all, a host entry the ring cannot stage, host experts with no item, one role unclaimed.
        CHECK(rule(unclaimed, layer_info(unclaimed, 0), no_ring, false, true) == "executor does not claim");
        CHECK(rule(idx, layer_info(idx, 1), ring_geo, true, false) == "executor does not claim");
        CHECK(rule(idx, layer_info(idx, 4), ring_geo, true, false) == "executor does not claim");
        CHECK(rule(partial, layer_info(partial, 0), ring_geo, true, false) == "executor does not claim");
        // A claimed layer goes on to the index's rules, each reached with one field of a copy cleared.
        const moe_stream_layer_info * c0      = layer_info(idx, 0);
        moe_stream_plan_index         no_host = idx;
        no_host.host_matrices_per_layer_max   = 0;
        CHECK(rule(no_host, c0, no_ring, false, true) == "no host matrices of a streamable type");
        moe_stream_plan_index no_bytes = idx;
        no_bytes.matrix_bytes_max      = 0;
        CHECK(rule(no_bytes, c0, no_ring, true, false) == "no matrix bytes");
        // No executor window (no planned weight slot, or one that holds no expert) sizes no ring either.
        moe_stream_plan_index no_window = idx;
        no_window.window_slots          = 0;
        CHECK(rule(no_window, c0, no_ring, true, false) == "no executor window");
        CHECK(rule(idx, c0, no_ring, true, false) == "ring exceeds an eleventh of resident plus lent expert bytes");
        CHECK(rule(idx, c0, ring_geo, false, false) == "queue not in order");
        CHECK(rule(idx, c0, ring_geo, true, true) == "reserve refused");
        // None of the seven: named anyway, so no ring-unavailable line goes unexplained.
        CHECK(rule(idx, c0, ring_geo, true, false) == "region smaller than the largest streamable matrix");
        CHECK(rule(partial, layer_info(partial, 2), ring_geo, true, false) ==
              "region smaller than the largest streamable matrix");
    }

    // ---- a window's items: one role's host items for the phase's experts, in the experts' order ----
    if (l0) {
        const expert_tensor_role up    = expert_tensor_role::UP;
        // Experts 0 and 7 are resident on device 0: they have no host item, so a window skips them.
        const int32_t            ex[4] = { 0, 1, 5, 7 };
        const auto               win   = ggml_sycl::moe_stream_window_items(*l0, up, ex, 4);
        CHECK(win.size() == 2);
        if (win.size() == 2) {
            CHECK(win[0].expert_id == 1 && win[0].role == up && win[0].layer == 0);
            CHECK(win[1].expert_id == 5 && win[1].role == up);
        }
        // The experts' order, not the plan's.
        const int32_t rev[2] = { 6, 1 };
        const auto    down   = ggml_sycl::moe_stream_window_items(*l0, expert_tensor_role::DOWN, rev, 2);
        CHECK(down.size() == 2 && down[0].expert_id == 6 && down[1].expert_id == 1);
        CHECK(down.size() == 2 && down[0].role == expert_tensor_role::DOWN);
        // A repeated expert is taken once: a window never stages one matrix into two slots.
        const int32_t dup[3] = { 5, 5, 5 };
        CHECK(ggml_sycl::moe_stream_window_items(*l0, up, dup, 3).size() == 1);
        // No experts, or a role with no host item, gives none.
        CHECK(ggml_sycl::moe_stream_window_items(*l0, up, nullptr, 0).empty());
        CHECK(ggml_sycl::moe_stream_window_items(*l0, expert_tensor_role::GATE_UP, ex, 4).empty());
        // An item the executor does not claim (staging NONE) is never in a window: layer 0's up weight is unclaimed
        // under claims_l0_gate_l2, its gate weight claimed.
        const moe_stream_layer_info * p0 = layer_info(partial, 0);
        CHECK(p0 != nullptr);
        if (p0) {
            CHECK(ggml_sycl::moe_stream_window_items(*p0, up, ex, 4).empty());
            CHECK(ggml_sycl::moe_stream_window_items(*p0, expert_tensor_role::GATE, ex, 4).size() == 2);
        }
    }

    // ---- which windows are exposed: copied with no earlier window of the layer to hide under ----
    {
        using ggml_sycl::moe_stream_window_exposed;
        // A layer's first window is exposed whatever its role or phase (GATE phase 0 in the dispatched order GATE, UP,
        // DOWN, or UP phase 0 in a model with no gate): no window prefetches across layers.
        CHECK(moe_stream_window_exposed(3, 2));
        // Every later window of the layer is prefetched under the one before it: the next phase of the role, or phase
        // 0 of the next role at the last phase. UP phase 0 is prefetched under GATE's last phase, so it is not exposed.
        CHECK(!moe_stream_window_exposed(3, 3));
        // A micro-batch's first window is exposed: begin_ubatch resets the last window's layer to -1.
        CHECK(moe_stream_window_exposed(0, -1));
    }

    // ---- a window's id: one per (layer, role, phase), never the no-window sentinel ----
    {
        using ggml_sycl::moe_stream_window_id;
        const expert_tensor_role     g = expert_tensor_role::GATE;
        const expert_tensor_role     u = expert_tensor_role::UP;
        const expert_tensor_role     d = expert_tensor_role::DOWN;
        std::unordered_set<uint32_t> ids_seen;
        size_t                       n_ids = 0;
        for (int l : { 0, 1, 47, 511 }) {
            for (expert_tensor_role r : { g, u, d }) {
                for (size_t p : { size_t(0), size_t(1), size_t(4095) }) {
                    const uint32_t w = moe_stream_window_id(l, r, p);
                    CHECK(w != UINT32_MAX);
                    ids_seen.insert(w);
                    ++n_ids;
                }
            }
        }
        CHECK(ids_seen.size() == n_ids);
        CHECK(moe_stream_window_id(2, u, 0) != moe_stream_window_id(2, g, 0));
        CHECK(moe_stream_window_id(2, u, 1) != moe_stream_window_id(3, u, 1));
    }

    // ---- an item one slot holds as AOS, read in place (c-hmey) ----
    moe_stream_item q{};
    q.type      = GGML_TYPE_Q4_0;
    q.ncols     = kCols;
    q.nrows     = kRows;
    q.src_bytes = src_bytes();
    CHECK(ggml_sycl::moe_stream_item_aos_fits(q, src_bytes()));  // the AOS bytes alone: no SOA side to hold
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(q, src_bytes() - 1));
    moe_stream_item bad = q;
    bad.type            = GGML_TYPE_COUNT;  // the plan's default type: none known
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    moe_stream_item iq = q;
    iq.type            = GGML_TYPE_IQ3_XXS;
    iq.src_bytes       = iq_src_bytes();
    CHECK(ggml_sycl::moe_stream_item_aos_fits(iq, 4096));
    bad      = iq;
    bad.type = GGML_TYPE_COUNT;
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    bad       = iq;
    bad.ncols = 100;  // not whole blocks
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    bad           = iq;
    bad.src_bytes = iq_src_bytes() + 1;  // not the rows' bytes
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    bad       = iq;
    bad.nrows = 0;
    CHECK(!ggml_sycl::moe_stream_item_aos_fits(bad, 4096));
    CHECK(ggml_sycl::moe_stream_staging_slots(moe_stream_staging::AOS_ONLY) == 1u);
    CHECK(ggml_sycl::moe_stream_staging_slots(moe_stream_staging::NONE) == 0u);

    // ---- a host lease the ring may copy from directly (llama.cpp-9y0p) ----
    alloc_metadata owner{};
    owner.tier      = alloc_tier::HOST_PINNED;
    owner.role      = alloc_role::WEIGHT;
    owner.host_zone = host_zone_id::WEIGHT;
    owner.device    = 0;
    CHECK(ggml_sycl::moe_stream_owner_streamable(owner, 0));
    CHECK(!ggml_sycl::moe_stream_owner_streamable(owner, 1));  // another device's host zone
    alloc_metadata o = owner;
    o.tier           = alloc_tier::DEVICE_VRAM;
    CHECK(!ggml_sycl::moe_stream_owner_streamable(o, 0));
    o           = owner;
    o.host_zone = host_zone_id::COUNT;
    CHECK(!ggml_sycl::moe_stream_owner_streamable(o, 0));
    o      = owner;
    o.role = alloc_role::STAGING;
    CHECK(!ggml_sycl::moe_stream_owner_streamable(o, 0));

    // ---- the rates: seed, then the tuning-cache record where it measured, then the environment ----
    const std::vector<std::pair<ggml_type, size_t>> q4 = {
        { GGML_TYPE_Q4_0, 1000 }
    };
    const moe_exec_rates seed      = ggml_sycl::moe_exec_rates_seed(q4);
    const moe_exec_rates no_record = ggml_sycl::moe_stream_rates_assemble(q4, nullptr, 0.0, nullptr);
    CHECK(near(no_record.h2d_gbps, seed.h2d_gbps) && near(no_record.reorder_gbps, seed.reorder_gbps));
    CHECK(near(no_record.dev_gflops, seed.dev_gflops) && near(no_record.cpu_gbps, seed.cpu_gbps));
    ggml_sycl_moe_exec_rates_record rec{};
    rec.h2d_gbps                  = 25.0;
    rec.reorder_gbps              = 300.0;
    rec.n_types                   = 1;
    rec.types[0]                  = GGML_TYPE_Q4_0;
    rec.dev_gflops[0]             = 1260.0;
    const moe_exec_rates measured = ggml_sycl::moe_stream_rates_assemble(q4, &rec, 0.0, nullptr);
    CHECK(near(measured.h2d_gbps, 25.0) && near(measured.reorder_gbps, 300.0) && near(measured.dev_gflops, 1260.0));
    const std::vector<std::pair<ggml_type, size_t>> mix = {
        { GGML_TYPE_Q4_0,  1000 },
        { GGML_TYPE_MXFP4, 1000 }
    };
    const moe_exec_rates mixed = ggml_sycl::moe_stream_rates_assemble(mix, &rec, 0.0, nullptr);
    CHECK(near(mixed.dev_gflops,
               2000.0 / (1000.0 / 1260.0 + 1000.0 / ggml_sycl::moe_exec_dev_gflops_seed(GGML_TYPE_MXFP4))));
    ggml_sycl_moe_exec_rates_record unmeasured = rec;
    unmeasured.h2d_gbps                        = 0.0;
    unmeasured.reorder_gbps                    = -1.0;
    const moe_exec_rates fallback              = ggml_sycl::moe_stream_rates_assemble(q4, &unmeasured, 0.0, nullptr);
    CHECK(near(fallback.h2d_gbps, seed.h2d_gbps) && near(fallback.reorder_gbps, seed.reorder_gbps));
    CHECK(near(ggml_sycl::moe_stream_rates_assemble(q4, &rec, 0.0, fake_env_h2d).h2d_gbps, 31.0));

    // llama.cpp-kp1l: h2d is the record's, else the device's probed rate, else the seed; the override wins over all.
    // No record at all (the tuning cache disabled): the probed rate prices, not the seed.
    const moe_exec_rates probed_only = ggml_sycl::moe_stream_rates_assemble(q4, nullptr, 26.5, nullptr);
    CHECK(near(probed_only.h2d_gbps, 26.5) && near(probed_only.reorder_gbps, seed.reorder_gbps));
    // A record that measured h2d wins over the probed rate.
    CHECK(near(ggml_sycl::moe_stream_rates_assemble(q4, &rec, 26.5, nullptr).h2d_gbps, 25.0));
    // A record without an h2d rate falls to the probed rate, then to the seed when that is not a rate either.
    CHECK(near(ggml_sycl::moe_stream_rates_assemble(q4, &unmeasured, 26.5, nullptr).h2d_gbps, 26.5));
    const double not_rates[] = { 0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity() };
    for (double bad : not_rates) {
        CHECK(near(ggml_sycl::moe_stream_rates_assemble(q4, nullptr, bad, nullptr).h2d_gbps, seed.h2d_gbps));
    }
    // The environment override is applied last: over the record and over the probed rate.
    CHECK(near(ggml_sycl::moe_stream_rates_assemble(q4, nullptr, 26.5, fake_env_h2d).h2d_gbps, 31.0));

    // A record that says reorder is unmeasured (0) takes the seed for it, and a later seed change is not masked.
    ggml_sycl_moe_exec_rates_record no_reorder = rec;
    no_reorder.reorder_gbps                    = 0.0;
    const moe_exec_rates no_reorder_rates      = ggml_sycl::moe_stream_rates_assemble(q4, &no_reorder, 0.0, nullptr);
    CHECK(near(no_reorder_rates.reorder_gbps, seed.reorder_gbps) && near(no_reorder_rates.h2d_gbps, 25.0));
    CHECK(ggml_sycl::moe_exec_rates_valid(measured) && ggml_sycl::moe_exec_rates_valid(fallback));

    // ---- the predicates begin_layer and executor_claims decide by ----
    // A micro-batch starts at the first op of a prompt, at a layer below the current one, or at the current layer in a
    // new graph. A new graph at a higher layer carries it on: a split micro-batch spans several graph_compute calls.
    CHECK(ggml_sycl::moe_stream_ubatch_starts(false, 3, 3, 7, 7));
    CHECK(ggml_sycl::moe_stream_ubatch_starts(true, 2, 3, 7, 7));
    CHECK(ggml_sycl::moe_stream_ubatch_starts(true, 0, 47, 8, 7));
    CHECK(ggml_sycl::moe_stream_ubatch_starts(true, 3, 3, 8, 7));
    // the layer's later ops
    CHECK(!ggml_sycl::moe_stream_ubatch_starts(true, 3, 3, 7, 7));
    // the next layer in the same graph
    CHECK(!ggml_sycl::moe_stream_ubatch_starts(true, 4, 3, 7, 7));
    // the next layer in the next graph of a split micro-batch
    CHECK(!ggml_sycl::moe_stream_ubatch_starts(true, 16, 15, 8, 7));

    // Bytes move when the layer's decision is RING under any policy but cpu: AUTO follows the cost model and ring
    // forces the choice in moe_exec_weigh (llama.cpp-3460, lifting lead ruling c-3skf).
    moe_exec_decision ring_all;
    ring_all.choice               = moe_exec_choice::RING;
    ring_all.regime               = moe_exec_regime::STREAM_ALL;
    moe_exec_decision ring_routed = ring_all;
    ring_routed.regime            = moe_exec_regime::ROUTED_ONLY;
    moe_exec_decision cpu_routed  = ring_routed;
    cpu_routed.choice             = moe_exec_choice::CPU;
    moe_exec_decision unconsulted;
    CHECK(ggml_sycl::moe_stream_stage_now(ring_all, moe_exec_policy::RING));
    CHECK(ggml_sycl::moe_stream_stage_now(ring_routed, moe_exec_policy::RING));
    CHECK(ggml_sycl::moe_stream_stage_now(ring_all, moe_exec_policy::AUTO));
    CHECK(ggml_sycl::moe_stream_stage_now(ring_routed, moe_exec_policy::AUTO));
    CHECK(!ggml_sycl::moe_stream_stage_now(ring_all, moe_exec_policy::CPU));
    CHECK(!ggml_sycl::moe_stream_stage_now(cpu_routed, moe_exec_policy::AUTO));
    CHECK(!ggml_sycl::moe_stream_stage_now(unconsulted, moe_exec_policy::AUTO));
    CHECK(!ggml_sycl::moe_stream_stage_now(cpu_routed, moe_exec_policy::RING));
    CHECK(!ggml_sycl::moe_stream_stage_now(unconsulted, moe_exec_policy::RING));

    // ---- the staged route's tally (llama.cpp-o9uw): served and refused lookups, and each refusal's cause and detail,
    // which end_prompt traces (plan Task 8: the trace says why) ----
    {
        using ggml_sycl::moe_staged_route_refusal;
        ggml_sycl::moe_staged_route_tally t;
        CHECK(t.served == 0 && t.handed_off == 0 && t.refused == 0 && t.refusals_text().empty());
        t.count(moe_staged_route_refusal::NONE, nullptr);
        t.count(moe_staged_route_refusal::LAYOUT, nullptr);
        t.count(moe_staged_route_refusal::UNSUPPORTED, "local-layout-unsupported");
        t.count(moe_staged_route_refusal::LAYOUT, nullptr);
        // The same detail at another address is the same refusal: details compare by text.
        const char same_reason[] = "local-layout-unsupported";
        t.count(moe_staged_route_refusal::UNSUPPORTED, same_reason);
        t.count(moe_staged_route_refusal::RECIPE, "RECIPE_MISMATCH");
        t.count(moe_staged_route_refusal::RECIPE, "RECIPE_MISSING");
        CHECK(t.served == 1 && t.refused == 6 && t.refusals.size() == 4);
        // An executor handoff (llama.cpp-3460) is neither served nor refused and names no cause: the executor reads
        // the slot itself, so the report's refusals and their WARN do not count it.
        CHECK(t.handed_off == 0);
        t.count_handoff();
        t.count_handoff();
        CHECK(t.handed_off == 2 && t.served == 1 && t.refused == 6 && t.refusals.size() == 4);
        CHECK(t.refusals_text() ==
              "layout=2 unsupported(local-layout-unsupported)=2 recipe(RECIPE_MISMATCH)=1 recipe(RECIPE_MISSING)=1");
        // Every cause has a name of its own.
        const moe_staged_route_refusal all[] = {
            moe_staged_route_refusal::NONE,          moe_staged_route_refusal::CLAIMED,
            moe_staged_route_refusal::UNSUPPORTED,   moe_staged_route_refusal::KERNEL,
            moe_staged_route_refusal::DIRECT_KERNEL, moe_staged_route_refusal::VIEW,
            moe_staged_route_refusal::LAYOUT,        moe_staged_route_refusal::ROUTE,
            moe_staged_route_refusal::RECIPE,
        };
        std::unordered_set<std::string> names;
        for (moe_staged_route_refusal r : all) {
            names.insert(ggml_sycl::moe_staged_route_refusal_name(r));
        }
        CHECK(names.size() == sizeof(all) / sizeof(all[0]) && names.count("unknown") == 0);
        CHECK(std::strcmp(ggml_sycl::moe_staged_route_refusal_name(moe_staged_route_refusal::DIRECT_KERNEL),
                          "direct-kernel") == 0);
    }

    // ---- the regions' last uses, by window: what stage_window() refills, window_done() gates and reset_regions()
    // forgets ----
    {
        // The struct is inline and every input below is a constant, so at -O3 the compiler would prove each check true
        // and drop it: a check that cannot fail proves nothing. Every use goes through a fresh read of a volatile
        // pointer instead, so no store before one read is known at the next, and each check runs (its text is in the
        // binary).
        ggml_sycl::moe_stream_region_reads<int> reads;
        ggml_sycl::moe_stream_region_reads<int> * volatile reads_at = &reads;
        const auto rr = [&]() -> ggml_sycl::moe_stream_region_reads<int> & {
            return *reads_at;
        };
        CHECK(rr().window[0] == UINT32_MAX && rr().window[1] == UINT32_MAX);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
        // stage_window(): region 0 holds window 7, region 1 the next window, 8, prefetched; neither has a read gate
        // until the executor's dequants of its window are behind a barrier.
        rr().refill(0, 7);
        rr().refill(1, 8);
        CHECK(rr().window[0] == 7 && rr().window[1] == 8);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
        // window_done(): the executor's barrier behind window 7's dequants gates the region holding window 7.
        rr().window_done(7, 41);
        CHECK(rr().read_gate[0] && *rr().read_gate[0] == 41);
        CHECK(!rr().read_gate[1]);
        // A window no region holds records nothing.
        rr().window_done(9, 42);
        CHECK(rr().read_gate[0] && *rr().read_gate[0] == 41 && !rr().read_gate[1]);
        // Refilled with window 9, region 0 forgets window 7's gate; region 1 keeps its window and takes its own gate.
        rr().refill(0, 9);
        CHECK(rr().window[0] == 9 && !rr().read_gate[0]);
        rr().window_done(8, 43);
        CHECK(rr().read_gate[1] && *rr().read_gate[1] == 43 && !rr().read_gate[0]);
        // A late done of a window no region holds any more records nothing.
        rr().window_done(7, 44);
        CHECK(!rr().read_gate[0] && rr().read_gate[1] && *rr().read_gate[1] == 43);
        // reset_regions(), for a new ring or a torn-down one: no window and no read gate in either region, so a later
        // window_done of a window the old ring held gates nothing.
        rr().reset();
        CHECK(rr().window[0] == UINT32_MAX && rr().window[1] == UINT32_MAX);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
        rr().window_done(9, 45);
        CHECK(!rr().read_gate[0] && !rr().read_gate[1]);
    }

    // ---- the per-window timing under GGML_SYCL_MOE_EXEC_TRACE=2 ----
    test_window_timing_records_one_record_per_window_only_under_trace();

    // ---- a short reserve asks for a loan of its shortfall, once per micro-batch, until one comes back REFUSED ----
    {
        using ggml_sycl::moe_stream_loan_latch;
        using ggml_sycl::moe_stream_loan_step;
        moe_stream_loan_latch latch;
        moe_stream_loan_latch * volatile latch_at = &latch;

        const auto lt = [&]() -> moe_stream_loan_latch & {
            return *latch_at;
        };

        const size_t      short_by = 3 * 1024 * 1024 + 4096;  // what the WEIGHT zone is short of
        loan_stub_state & st       = *g_loan_at;
        // Micro-batch 1: its four staging layers reach the reserve once, and the one request names the device and the
        // shortfall, not the ring's bytes, and the ring's slot the shortfall is counted in.
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::REQUEST);
        CHECK((*g_loan_at).requests == 1 && (*g_loan_at).last_device == 1 && (*g_loan_at).last_bytes == short_by);
        CHECK((*g_loan_at).last_slot_bytes == kLoanSlotBytes);
        // The boundary lent something but the ring is still short (ACCEPTED, the count did not move): micro-batch 2
        // asks again, once.
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::REQUEST);
        CHECK((*g_loan_at).requests == 2);
        // The boundary answered BUSY (the runner re-posts it and counts no refusal): micro-batch 3 asks again.
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::REQUEST);
        CHECK((*g_loan_at).requests == 3);
        // A refusal on another device is not this ring's.
        ++st.refusals[0];
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::REQUEST);
        CHECK((*g_loan_at).requests == 4);
        // The boundary REFUSED the loan: the next micro-batch answers REFUSED once (the caller's one WARN) and asks
        // for none, and neither does any later micro-batch of the prompt.
        ++st.refusals[1];
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::REFUSED);
        CHECK((*g_loan_at).requests == 4 && lt().refused);
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::NONE);
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::NONE);
        CHECK((*g_loan_at).requests == 4);
        // end_prompt clears the latch: the next prompt asks again, and a refusal counted before its first ask is not
        // its own.
        lt() = moe_stream_loan_latch();
        CHECK(run_short_ubatch(lt(), 1, short_by, 4) == moe_stream_loan_step::REQUEST);
        CHECK((*g_loan_at).requests == 5 && (*g_loan_at).last_bytes == short_by);
        // A reserve refused for another reason reports no shortfall: nothing to ask for, and the latch is untouched.
        lt() = moe_stream_loan_latch();
        CHECK(run_short_ubatch(lt(), 1, 0, 4) == moe_stream_loan_step::NONE);
        CHECK((*g_loan_at).requests == 5 && !lt().asked);
        // No request hook (a phase no backend op reached): nothing is asked.
        CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(lt(), 1, short_by, kLoanSlotBytes, nullptr,
                                                          stub_loan_refusals, nullptr) == moe_stream_loan_step::NONE);
        CHECK((*g_loan_at).requests == 5 && !lt().asked);
        // No refusal count: a loan is never seen refused, so every short micro-batch asks.
        for (int u = 0; u < 3; ++u) {
            CHECK(ggml_sycl::moe_stream_loan_on_short_reserve(lt(), 0, short_by, 0, stub_request_loan, nullptr,
                                                              nullptr) == moe_stream_loan_step::REQUEST);
        }
        CHECK((*g_loan_at).requests == 8 && (*g_loan_at).last_device == 0);
        CHECK((*g_loan_at).last_slot_bytes == 0);  // a slot size of 0 reaches the backend as is: it counts bytes
    }

    if (failures) {
        std::fprintf(stderr, "test-moe-stream-phase: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-moe-stream-phase: OK\n");
    return 0;
}
