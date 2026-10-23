// Device test of the STAGED_RING route (llama.cpp-o9uw, Task 8 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// The backend's stream phase is given stage-table entries whose slots are ring
// slots this test filled, and the resolver runs on the backend's own device and
// queue, as a prompt MUL_MAT_ID runs it, over a Q4_0 expert tensor with no
// cache entry (so the route core can only answer "unavailable"):
// 1. A prompt build (rows > 1) of an expert with an AOS entry is served: one
//    PRIMARY_DEVICE operand per occurrence, transfer STAGED_RING, layout AOS,
//    kernel MMVQ_COMPAT, carrying the entry's ready event; its lease resolves on
//    the device at the slot, and the bytes read back through it are the fill's.
//    The executor choice is PRIMARY_DEVICE, and the phase counts one served
//    lookup (the resolver memoizes per expert).
// 2. A decode build (rows == 1) never consults the entry: the route core
//    answers and the counts do not move.
// 3. An expert with no entry falls through to the route core, uncounted.
// 4. Each slot resolves with the layout it was acquired for (lead ruling
//    c-o9uw-1), and the tag check alone decides between two entries that differ
//    in nothing else. The SOA case is MXFP4, the one type a prompt kernel reads
//    as SOA (MMVQ_COMPAT): no plan is published, so the oneDNN batched executor
//    claims nothing. A MXFP4 SOA entry over the SOA slot is served from it, with
//    layout SOA. The same entry naming the AOS-tagged slot as its SOA slot is
//    refused and counted. So is a Q4_0 AOS entry over the SOA-tagged slot, which
//    the capability query alone would serve. A Q4_0 SOA entry over the
//    SOA-tagged slot agrees with its tag but is refused all the same: the only
//    prompt kernel that reads Q4_0 as SOA is the oneDNN batched prompt executor
//    (llama.cpp-348c), and a staged slot is served only with MMVQ_COMPAT
//    (c-las5), so the staged route refuses the kernel. The phase's refusals name
//    each cause, as the trace prints them: two layout refusals and one kernel
//    refusal, with the query's reason (plan Task 8).
// 5. Two micro-batches: after begin_ubatch the first entry is gone, and an
//    entry for the same expert in another slot is served from that slot, whose
//    bytes differ from the first one's by construction.
// 6. end_prompt clears the counts and the refusals.
// 7. A staged slot's pointer goes with its ring (c-8jtz). The backend's phase
//    reserves a ring of its own and a staged batch over one of its slots
//    uploads its pointer table, as a prompt MUL_MAT_ID does, for a weight with
//    an extra: the payload slot holds the slot's pointer and no handle, the
//    device table holds it too, no active lease is the slot, and the backend
//    holds one record for the weight, counted on its extra. end_prompt gives the
//    ring back: the payload slot is null, the device table is invalid (the
//    table pointer answers null), and no record is left, so releasing the extra
//    passes release_extra_gpu's assert.
//
// A test may wait on events; nothing here status-queries one.

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../moe-resolved-batch.hpp"
#include "../moe-stream-ring.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "sycl-test-skip.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace ggml_sycl;

static int g_failures = 0;

#define CHECK(cond, ...)                       \
    do {                                       \
        if (cond) {                            \
            std::printf("PASS: " __VA_ARGS__); \
        } else {                               \
            std::printf("FAIL: " __VA_ARGS__); \
            ++g_failures;                      \
        }                                      \
        std::printf("\n");                     \
    } while (0)

static constexpr int     kLayer  = 3;
static constexpr int32_t kExpert = 5;
static constexpr size_t  kSlot   = 1024 * 1024;

// Every byte of `n` at `ptr` equals `want`, read back behind `dep`.
static bool slot_holds(sycl::queue & q, const void * ptr, size_t n, uint8_t want, const sycl::event & dep) {
    if (ptr == nullptr) {
        return false;
    }
    std::vector<uint8_t> host(n, 0);
    mem_copy_ptr_async(host.data(), ptr, n, q, { dep }).wait();
    for (uint8_t b : host) {
        if (b != want) {
            return false;
        }
    }
    return true;
}

static void counts(ggml_backend_t backend, size_t * served, size_t * refused) {
    *served  = SIZE_MAX;
    *refused = SIZE_MAX;
    (void) test_moe_stream_staged_route_counts(backend, served, refused);
}

// The phase's refusals as the trace prints them, or "(no phase)".
static std::string refusals(ggml_backend_t backend) {
    std::string text = "(no phase)";
    (void) test_moe_stream_staged_route_refusals(backend, &text);
    return text;
}

// A served build of `expert` from `slot`, whose entry holds `layout`: every check of case 1, against the fill byte
// `want`.
static void check_served(sycl::queue &                     q,
                         int                               device,
                         const moe_resolved_batch_result & r,
                         int32_t                           expert,
                         ggml_layout_mode                  layout,
                         const mem_handle &                slot,
                         const sycl::event &               ready,
                         uint8_t                           want,
                         const char *                      what) {
    CHECK(r, "%s: the prompt build is admitted (reject %s)", what, moe_batch_reject_reason_name(r.reject));
    if (!r) {
        return;
    }
    CHECK(r.batch.operands.size() == 2, "%s: one operand per occurrence (got %zu)", what, r.batch.operands.size());
    const void * slot_ptr = slot.resolve(device).ptr;
    for (const moe_resolved_operand & op : r.batch.operands) {
        CHECK(op.expert_id == expert && op.residency() == moe_batch_residency::PRIMARY_DEVICE,
              "%s: the staged expert is a PRIMARY_DEVICE operand", what);
        CHECK(op.recipe().transfer == moe_recipe_transfer::STAGED_RING, "%s: the recipe's transfer is STAGED_RING",
              what);
        CHECK(op.recipe().layout == layout && op.actual_layout() == layout,
              "%s: the recipe's layout is the entry's, %s", what, layout == GGML_LAYOUT_SOA ? "SOA" : "AOS");
        CHECK(op.recipe().kernel == moe_route_kernel::MMVQ_COMPAT, "%s: the kernel is the plain MMVQ_COMPAT", what);
        CHECK(op.has_ready_event() && op.ready_event() == ready, "%s: the operand carries the entry's ready event",
              what);
        const resolved_ptr view = op.lease().resolve(device);
        CHECK(view.ptr != nullptr && view.on_device && view.ptr == slot_ptr,
              "%s: the lease resolves on the device at the slot (got %p, want %p)", what, view.ptr, slot_ptr);
        const moe_batch_executor_choice choice = choose_moe_batch_executor(op, device, false, 0);
        CHECK(choice && choice.executor == moe_batch_executor::PRIMARY_DEVICE,
              "%s: the executor choice is PRIMARY_DEVICE (reject %s)", what,
              moe_batch_reject_reason_name(choice.reject));
    }
    CHECK(slot_holds(q, r.batch.operands[0].lease().resolve(device).ptr, kSlot, want, ready),
          "%s: the bytes read through the lease are the fill's 0x%02x", what, want);
}

int main() {
    std::optional<sycl::device> dev_opt = sycl_test_require_gpu("the STAGED_RING route");
    if (!dev_opt) {
        return SYCL_TEST_SKIP;
    }
    const auto & sycl_info = ggml_sycl_info();
    if (sycl_info.total_gpu_count <= 0) {
        sycl_test_print_skip("the STAGED_RING route backend owner selection");
        return SYCL_TEST_SKIP;
    }
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend == nullptr) {
        std::printf("FAIL: ggml_backend_sycl_init(0) returned null\n");
        return 1;
    }
    // The backend owner's device and context, as test-moe-stream-ring does; the slots' copies run on this queue and
    // the resolver's barrier on the backend's queue waits for them.
    auto &        owner       = ggml_sycl_get_gpu_device(0);
    sycl::queue & owner_queue = owner.default_queue();
    sycl::queue   q(owner_queue.get_context(), owner_queue.get_device(), sycl::property::queue::in_order{});
    const int     device = ggml_sycl_get_device_id_from_queue(q);
    std::printf("Device: %s (logical %d)\n", q.get_device().get_info<sycl::info::device::name>().c_str(), device);
    if (device != 0) {
        std::printf("FAIL: backend owner queue did not round-trip to logical GPU 0\n");
        ggml_backend_free(backend);
        return 1;
    }
    (void) get_unified_cache_for_device(device);

    // A Q4_0 expert tensor of the stream phase's layer and role, with no data and no cache entry, and a MXFP4 one of
    // the same layer, role and shape for the SOA cases.
    ggml_init_params params  = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   tctx    = ggml_init(params);
    ggml_tensor *    src0    = ggml_new_tensor_3d(tctx, GGML_TYPE_Q4_0, 256, 64, 8);
    ggml_tensor *    src0_mx = ggml_new_tensor_3d(tctx, GGML_TYPE_MXFP4, 256, 64, 8);
    ggml_set_name(src0, "blk.3.ffn_gate_exps.weight");
    ggml_set_name(src0_mx, "blk.3.ffn_gate_exps.weight");
    const expert_tensor_role role = expert_tensor_role::GATE;

    {
        // Two regions of two 1 MiB slots: A and S in the first, B in the second.
        moe_stream_ring          ring;
        moe_stream_ring_geometry geo;
        geo.slot_bytes      = kSlot;
        geo.n_slots         = 4;
        geo.bytes           = 4 * kSlot;
        geo.region_slots    = 2;
        const bool reserved = ring.reserve(device, geo);
        CHECK(reserved, "reserve(4 slots x %zu bytes) on device %d", kSlot, device);
        if (!reserved) {
            ggml_free(tctx);
            ggml_backend_free(backend);
            std::printf("FAILED: %d case(s)\n", g_failures);
            return 1;
        }
        mem_handle        slot_a = ring.acquire_slot(GGML_LAYOUT_AOS);
        mem_handle        slot_s = ring.acquire_slot(GGML_LAYOUT_SOA);
        const sycl::event fill_a = mem_fill_async(slot_a, 0x11, kSlot, q, { ring.current_region_gate() });
        const sycl::event fill_s = mem_fill_async(slot_s, 0x33, kSlot, q, { ring.current_region_gate() });
        ring.epoch_boundary(fill_s);
        mem_handle        slot_b = ring.acquire_slot(GGML_LAYOUT_AOS);
        const sycl::event fill_b = mem_fill_async(slot_b, 0x22, kSlot, q, { ring.current_region_gate() });

        const int32_t prompt_ids[]  = { kExpert, kExpert };  // two tokens, one slot each: rows == 2
        const int32_t decode_ids[]  = { kExpert };           // one token: rows == 1
        const int32_t miss_ids[]    = { 6, 6 };
        const int32_t tag_ids[]     = { 7, 7 };
        const int32_t soa_tag_ids[] = { 2, 2 };
        const int32_t soa_q4_ids[]  = { 4, 4 };
        size_t        served = 0, refused = 0;

        // The build results hold slot leases; they all go before the ring is released below.
        {
            // Case 1: served. The insert makes the backend's phase.
            CHECK(test_moe_stream_insert_staged(backend, kLayer, kExpert, role, mem_handle(), slot_a, fill_a,
                                                GGML_LAYOUT_AOS),
                  "an AOS entry goes into the backend's stage table");
            const moe_resolved_batch_result first =
                test_moe_staged_ring_build(backend, src0, prompt_ids, 2, 1, GGML_LAYOUT_AOS);
            check_served(q, device, first, kExpert, GGML_LAYOUT_AOS, slot_a, fill_a, 0x11, "micro-batch 1");
            counts(backend, &served, &refused);
            CHECK(served == 1 && refused == 0, "one served lookup for one unique expert (served %zu, refused %zu)",
                  served, refused);

            // Case 2: decode never consults the entry.
            const moe_resolved_batch_result decode =
                test_moe_staged_ring_build(backend, src0, decode_ids, 1, 1, GGML_LAYOUT_AOS);
            CHECK(!decode && decode.reject == moe_batch_reject_reason::ROUTE_UNAVAILABLE,
                  "rows == 1: the route core answers, not the entry (reject %s)",
                  moe_batch_reject_reason_name(decode.reject));
            counts(backend, &served, &refused);
            CHECK(served == 1 && refused == 0, "rows == 1: the counts do not move (served %zu, refused %zu)", served,
                  refused);

            // Case 3: a miss falls through, uncounted.
            const moe_resolved_batch_result miss =
                test_moe_staged_ring_build(backend, src0, miss_ids, 2, 1, GGML_LAYOUT_AOS);
            CHECK(!miss && miss.reject == moe_batch_reject_reason::ROUTE_UNAVAILABLE,
                  "an expert with no entry falls through to the route core (reject %s)",
                  moe_batch_reject_reason_name(miss.reject));
            counts(backend, &served, &refused);
            CHECK(served == 1 && refused == 0, "a miss is not counted (served %zu, refused %zu)", served, refused);

            // Case 4: the slots carry the layouts they were acquired for.
            CHECK(slot_a.resolve(device).layout == GGML_LAYOUT_AOS && slot_s.resolve(device).layout == GGML_LAYOUT_SOA,
                  "slot A resolves as AOS and slot S as SOA, the layouts they were acquired for");
            // A MXFP4 SOA entry over the SOA slot: the tags agree, a prompt kernel reads MXFP4 as SOA and nothing
            // claims the weight, so it is served from the SOA slot.
            CHECK(test_moe_stream_insert_staged(backend, kLayer, 6, role, slot_s, slot_a, fill_s, GGML_LAYOUT_SOA),
                  "a SOA entry goes into the stage table");
            const moe_resolved_batch_result soa =
                test_moe_staged_ring_build(backend, src0_mx, miss_ids, 2, 1, GGML_LAYOUT_SOA);
            check_served(q, device, soa, 6, GGML_LAYOUT_SOA, slot_s, fill_s, 0x33, "MXFP4 SOA entry");
            counts(backend, &served, &refused);
            CHECK(served == 2 && refused == 0, "the SOA entry is served (served %zu, refused %zu)", served, refused);
            // The same entry naming the AOS-tagged slot as its SOA slot: type, kernel and claim are as above, so only
            // the tag check can refuse it.
            CHECK(test_moe_stream_insert_staged(backend, kLayer, 2, role, slot_a, slot_a, fill_a, GGML_LAYOUT_SOA),
                  "a SOA entry over an AOS-tagged slot goes into the stage table");
            const moe_resolved_batch_result soa_mistagged =
                test_moe_staged_ring_build(backend, src0_mx, soa_tag_ids, 2, 1, GGML_LAYOUT_SOA);
            CHECK(!soa_mistagged, "a SOA entry whose slot carries AOS is refused and falls through (reject %s)",
                  moe_batch_reject_reason_name(soa_mistagged.reject));
            counts(backend, &served, &refused);
            CHECK(served == 2 && refused == 1, "the SOA tag refusal is counted (served %zu, refused %zu)", served,
                  refused);
            // An AOS entry over the SOA-tagged slot: the query would serve Q4_0 as AOS, the tag check refuses it.
            CHECK(
                test_moe_stream_insert_staged(backend, kLayer, 7, role, mem_handle(), slot_s, fill_s, GGML_LAYOUT_AOS),
                "an AOS entry over a SOA-tagged slot goes into the stage table");
            const moe_resolved_batch_result mistagged =
                test_moe_staged_ring_build(backend, src0, tag_ids, 2, 1, GGML_LAYOUT_AOS);
            CHECK(!mistagged && mistagged.reject == moe_batch_reject_reason::ROUTE_UNAVAILABLE,
                  "an entry whose slot carries another layout is refused and falls through (reject %s)",
                  moe_batch_reject_reason_name(mistagged.reject));
            counts(backend, &served, &refused);
            CHECK(served == 2 && refused == 2, "the tag refusal is counted (served %zu, refused %zu)", served, refused);
            // A Q4_0 SOA entry over the SOA-tagged slot: the tags agree, but the only prompt kernel that reads Q4_0 as
            // SOA is the oneDNN batched executor (ONEDNN_BATCHED_PROMPT, llama.cpp-348c), and a staged slot is served
            // only with MMVQ_COMPAT, so the staged route refuses the kernel before the tag is looked at.
            CHECK(test_moe_stream_insert_staged(backend, kLayer, 4, role, slot_s, slot_a, fill_s, GGML_LAYOUT_SOA),
                  "a Q4_0 SOA entry over the SOA-tagged slot goes into the stage table");
            const moe_resolved_batch_result soa_q4 =
                test_moe_staged_ring_build(backend, src0, soa_q4_ids, 2, 1, GGML_LAYOUT_SOA);
            CHECK(!soa_q4, "a Q4_0 SOA entry is refused and falls through (reject %s)",
                  moe_batch_reject_reason_name(soa_q4.reject));
            counts(backend, &served, &refused);
            CHECK(served == 2 && refused == 3, "the Q4_0 SOA refusal is counted (served %zu, refused %zu)", served,
                  refused);
            const std::string causes = refusals(backend);
            CHECK(causes == "layout=2 kernel(local-onednn-batched-prompt)=1",
                  "the refusals name their causes: two layout, one kernel with the query's reason (got \"%s\")",
                  causes.c_str());

            // Case 5: the next micro-batch starts with an empty table and reads its own slot.
            CHECK(test_moe_stream_begin_ubatch(backend, 2), "the next micro-batch begins");
            const moe_resolved_batch_result stale =
                test_moe_staged_ring_build(backend, src0, prompt_ids, 2, 1, GGML_LAYOUT_AOS);
            CHECK(!stale && stale.reject == moe_batch_reject_reason::ROUTE_UNAVAILABLE,
                  "after begin_ubatch the first micro-batch's entry is gone (reject %s)",
                  moe_batch_reject_reason_name(stale.reject));
            (void) test_moe_stream_insert_staged(backend, kLayer, kExpert, role, mem_handle(), slot_b, fill_b,
                                                 GGML_LAYOUT_AOS);
            const moe_resolved_batch_result second =
                test_moe_staged_ring_build(backend, src0, prompt_ids, 2, 1, GGML_LAYOUT_AOS);
            check_served(q, device, second, kExpert, GGML_LAYOUT_AOS, slot_b, fill_b, 0x22, "micro-batch 2");
            if (second && first) {
                CHECK(second.batch.operands[0].lease().resolve(device).ptr !=
                          first.batch.operands[0].lease().resolve(device).ptr,
                      "the second micro-batch's lease is not the first one's slot");
            }

            // Case 6: end_prompt clears the counts and the refusals.
            CHECK(test_moe_stream_end_prompt(backend), "end_prompt runs");
            counts(backend, &served, &refused);
            CHECK(served == 0 && refused == 0, "end_prompt clears the counts (served %zu, refused %zu)", served,
                  refused);
            const std::string cleared = refusals(backend);
            CHECK(cleared.empty(), "end_prompt clears the refusals (got \"%s\")", cleared.c_str());
        }

        q.wait();
        ggml_backend_synchronize(backend);
        slot_a = mem_handle();
        slot_s = mem_handle();
        slot_b = mem_handle();
        ring.release();
    }

    {
        // Case 7: the phase's own ring, one slot per region, and an extra for the weight, which the upload needs.
        CHECK(test_moe_stream_reserve_ring(backend, kSlot, 2), "the backend's phase reserves a ring of 2 x %zu bytes",
              kSlot);
        mem_handle        slot     = test_moe_stream_acquire_slot(backend, GGML_LAYOUT_AOS);
        const void *      slot_ptr = slot.resolve(device).ptr;
        const sycl::event fill     = slot.valid() ? mem_fill_async(slot, 0x44, kSlot, q) : sycl::event();
        CHECK(slot.valid() && slot_ptr != nullptr && slot.resolve(device).layout == GGML_LAYOUT_AOS,
              "a slot of the phase's ring, acquired as AOS");
        auto * extra    = new ggml_tensor_extra_gpu();
        src0->extra     = extra;
        const int32_t n = static_cast<int32_t>(src0->ne[2]);
        {
            const int32_t prompt_ids[] = { kExpert, kExpert };
            CHECK(test_moe_stream_insert_staged(backend, kLayer, kExpert, role, mem_handle(), slot, fill,
                                                GGML_LAYOUT_AOS),
                  "an AOS entry over the phase's slot goes into the stage table");
            const moe_resolved_batch_result r =
                test_moe_staged_ring_build(backend, src0, prompt_ids, 2, 1, GGML_LAYOUT_AOS);
            CHECK(r, "the staged build is admitted (reject %s)", moe_batch_reject_reason_name(r.reject));
            const void * const * table =
                r ? test_moe_staged_ring_upload(backend, src0, r.batch, GGML_LAYOUT_AOS) : nullptr;
            CHECK(table != nullptr, "the staged batch's pointer table uploads");
            const std::vector<void *> &     payload = extra->weight().moe_expert_ptr_payload[device];
            const std::vector<mem_handle> & handles = extra->weight().moe_expert_handles[device];
            CHECK(payload.size() == static_cast<size_t>(n) && payload[kExpert] == slot_ptr,
                  "the payload slot holds the staged slot's pointer (got %p, want %p)",
                  payload.size() > static_cast<size_t>(kExpert) ? payload[kExpert] : nullptr, slot_ptr);
            CHECK(handles.size() == static_cast<size_t>(n) && !handles[kExpert].valid(),
                  "the payload slot's handle entry is empty: the slot is not the expert's storage");
            bool lease_is_slot = false;
            for (const mem_handle & lease : extra->weight().moe_expert_ptrs_leases[device]) {
                lease_is_slot = lease_is_slot || lease.resolve(device).ptr == slot_ptr;
            }
            CHECK(!lease_is_slot, "no active lease of the table is the staged slot");
            const uint32_t records_extra   = extra->weight().moe_stream_payload_records.load();
            const size_t   records_backend = test_moe_stream_payload_records(backend);
            CHECK(records_extra == 1 && records_backend == 1,
                  "one record for the weight, counted on its extra (extra %u, backend %zu)", records_extra,
                  records_backend);
            CHECK(extra->weight().moe_device_table_valid[device] && extra->moe_ptrs_ptr(device) == table,
                  "the device table is valid and is the one the upload returned");
            ggml_backend_synchronize(backend);
            std::vector<void *> device_table(static_cast<size_t>(n), nullptr);
            if (table != nullptr) {
                mem_copy_ptr_async(device_table.data(), table, device_table.size() * sizeof(void *), q).wait();
            }
            CHECK(device_table[kExpert] == slot_ptr, "the device table holds the staged slot's pointer (got %p)",
                  device_table[kExpert]);
        }

        // end_prompt gives the phase's ring back, so the recorded pointer goes.
        q.wait();
        ggml_backend_synchronize(backend);
        CHECK(test_moe_stream_end_prompt(backend), "end_prompt gives the phase's ring back");
        const std::vector<void *> & payload = extra->weight().moe_expert_ptr_payload[device];
        CHECK(payload.size() == static_cast<size_t>(n) && payload[kExpert] == nullptr,
              "the recorded payload slot is null once the ring is given back");
        CHECK(!extra->weight().moe_device_table_valid[device] && extra->moe_ptrs_ptr(device) == nullptr,
              "the device table is invalid, so the table pointer answers null until the next upload");
        CHECK(extra->weight().moe_stream_payload_records.load() == 0 && test_moe_stream_payload_records(backend) == 0,
              "no record is left (extra %u, backend %zu)", extra->weight().moe_stream_payload_records.load(),
              test_moe_stream_payload_records(backend));
        slot        = mem_handle();
        src0->extra = nullptr;
        release_extra_gpu(extra);  // asserts that no record names the extra
    }

    ggml_free(tctx);
    ggml_backend_free(backend);
    CHECK(drain_retained_handles(true), "the final retained-handle drain completes");

    if (g_failures != 0) {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
