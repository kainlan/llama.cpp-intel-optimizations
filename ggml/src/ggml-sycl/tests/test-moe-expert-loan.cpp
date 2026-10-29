//
// Test: the expert loan transaction (llama.cpp-mwmr, plan Task 11b of llama.cpp-36j0) and its refill
// (llama.cpp-sif1, plan Task 12).
//
// A prompt phase lends the device's least popular resident expert triplets to the prefill ring: each is copied to
// pinned host storage, the published plan flips it to the host, and its device entry is retired so the storage goes
// back. No MoE model fixture exists host-side, so the cases build synthetic 1 MiB experts through
// begin_planned_materialization + direct_stage_expert (the bulk case through the tensor-bulk stager) inside the
// backend, where the file-static transaction lives, and report here:
//
//   refuses_without_host_zones
//                           Lead ruling: before any case configures host zones, a loan of one staged triplet is
//                           refused "host zones not configured", the probe agrees, the boundary runner does not
//                           re-post it, and nothing is published, lent or moved off the device. Then, with the
//                           zones configured, the same request lends the triplet, from zone slices. Runs first.
//   lends_two_triplets      2 layers x 4 experts, expert 3 on the host: three requests leave the largest pending,
//                           the graph-boundary runner lends two triplets and clears it, the plan shows them on the
//                           host, resolve_expert finds host-pinned copies equal to the originals, the device entries
//                           are gone within 2 s and the device gets their storage back: the entries' sizes, which the
//                           loan reports, and which equal the requested charge only because a 1 MiB expert is
//                           256-byte aligned.
//   probe_changes_nothing   c-tdra first: inside a graph the transaction refuses "graph active" and a request waits.
//                           Then the same plan probed: the selection is reported and nothing is copied or published.
//   refuses_replacing_a_host_entry
//                           c-mwmr-3: a host entry planted under the selected expert's host id, checked live and idle
//                           (no lease, not retired) before the loan, is refused with "host entry exists"; nothing
//                           is copied or published and the planted entry keeps its storage.
//   refuses_shared_storage  c-sepd: a bulk-staged pair with expert 1 flipped to the host is refused with "shared
//                           storage"; nothing is copied or published.
//   direct_stage_outside_a_load_copies_through_staging
//                           The evidence the refill's design rests on: outside a model load, under a published
//                           plan, direct staging takes one expert from a pinned host copy (its gates pass), and its
//                           copy takes the staged branch, a synchronous bounce; so the refill copies owner to owner
//                           itself.
//   refill_readmits_one_triplet_per_call
//                           The two-triplet loan, then a decode boundary's refill and a direct one, one triplet each.
//                           Right after the first, while its withdrawn host entries are still in the cache, retired,
//                           the re-admitted ids resolve past them to the device entries, still IN_PROGRESS with the
//                           copy as their ready event, and each re-admitted handle resolves on the device and refuses
//                           a host caller (three wrong-device WARNs, by design).
//                           Each re-admitted triplet is on the device in the plan, resolves there with its original
//                           bytes, its extras name the device copy, and its host entries are gone; each refill call
//                           took no staged copy; a third call has nothing to refill; the raw device bytes and the
//                           WEIGHT zone room are what they were before the loan; the teardown counts read 2/2.
//   refill_cancelled_by_a_loan
//                           One refill, then a one-triplet loan: the next refill re-admits nothing (0 remaining) and
//                           publishes nothing, the triplet still lent stays on the host, a prompt boundary keeps the
//                           cursor disarmed and the next decode boundary re-admits again, the new loan's triplet (the
//                           ledger's back) while the first loan's stays lent; neither accepted refill took a staged
//                           copy.
//   refill_runs_once_per_decode_graph
//                           The two-triplet loan, then two boundaries with no graph between them: the first re-admits
//                           one triplet, the second answers "no decode graph since the last refill" and publishes
//                           nothing; the next decode graph's boundary re-admits the other.
//   refill_drops_an_unpreparable_triplet
//                           A four-triplet loan and a resolver that refuses one member. Refused at the back, the
//                           triplet leaves the ledger with nothing moved or published and its minted zone room handed
//                           back; refused second at two per call, the back moves and the refused one leaves. Each
//                           drop WARNs once ("stays on the host: no tensor for a lent expert", two in all) and resets
//                           the runner's count. Then a full WEIGHT zone answers "WEIGHT zone full" and keeps the last
//                           triplet (one "refill device 0 refused" WARN), which the next decode boundary re-admits
//                           once the room is back. The teardown counts read 2/4.
//   refill_moves_two_triplets_per_call
//                           A three-triplet loan, then one decode boundary at two per call: both move (2 triplets,
//                           six entries' bytes, 1 remaining, no staged copy), each re-admitted id resolves right away
//                           to a device entry IN_PROGRESS behind its copy, and the third stays lent until the next
//                           decode boundary re-admits it. The teardown counts read 3/3.
//   loan_sizes_its_copies_from_the_entry
//                           llama.cpp-pu8v: each device entry 72 bytes past the plan's dst_size, as S1-PRELOAD pads a
//                           Q4_0 expert whose rows are not a multiple of MATRIX_ROW_PADDING. A host copy minted at the
//                           plan's size is refused "host storage too small" and moves nothing; then the loan lends two
//                           triplets, each host copy the entry's size with its zero tail, reporting exactly the six
//                           entries' sizes freed, and one decode boundary re-admits both, reporting the same bytes.
//   loan_and_refill_stop_on_a_missing_entry
//                           llama.cpp-pu8v: a one-triplet loan whose down member has no device entry stops "not
//                           resident" (its refusal WARNed once): nothing lent, published or re-posted. After a
//                           two-triplet loan, the triplet behind the ledger's back loses its gate host copy: one
//                           decode boundary re-admits the back and keeps it, WARNing no drop, and the next stops "not
//                           lent" and drops it, WARNing "stays on the host: not lent" once, publishing nothing. Both
//                           WARNs are captured and asserted.
//   loan_reclaims_pointer_table_leases
//                           llama.cpp-6aj4: one layer's three tensors each get the pointer table a prompt graph's
//                           preload builds (ggml_sycl_update_moe_ptr_table over the routed device experts), and each
//                           table leases expert 2's device entry (checked first). Slot 0's table leases are then made
//                           the only view its extra holds of that entry, its storage records and slot handle dropped
//                           (checked, with no route-table lease and no hybrid record), so only the loan's lease path
//                           can mark that table stale; slots 1 and 2 keep every view. After the loan each table is
//                           stale, slot 0's checked by its stale mark alone and the others' also by a payload that no
//                           longer names expert 2. The one-triplet loan takes expert 2 and no other triplet; after the
//                           loan's own boundary drain and the one deferred-free pass synchronize runs after it, with no
//                           polling, its three device entries are gone and the WEIGHT zone has their room back, and the
//                           first decode boundary re-admits the triplet. A failure names each entry's lease count and
//                           the extra's handles on it per view, counted without resolving any.
//   loan_reclaims_after_a_timed_out_boundary_drain
//                           llama.cpp-6aj4: the same loan, slot 0 lease-only as above, with the drain worker held on
//                           the loan's first retained record, so the boundary's 100 ms drain times out (checked, with
//                           the worker parked). The three device entries survive that boundary's pass; once the worker
//                           goes and drains, the next pass erases them and returns their room, and the refill re-admits
//                           the triplet.
//   refill_returns_an_unengaged_prompts_loan_at_once
//                           llama.cpp-i4ny: a three-triplet loan, then a prompt whose ring never engaged ends. The
//                           first decode graph's boundary reads that from the phase, once, and at one triplet per
//                           call takes the whole loan back in one transaction: the ledger is empty, the three
//                           triplets are on the device with no staged copy, and exactly one "bulk refill" WARN names
//                           decode graph 1. The next decode boundary has nothing to refill and prints no second
//                           line. The teardown counts read 3/3.
//   refill_keeps_the_per_graph_refill_after_an_engaged_prompt
//                           llama.cpp-i4ny, the control: the same loan, then a prompt whose ring reserved two slots
//                           ends engaged. The first decode boundary is the per-graph refill alone: one triplet, two
//                           left, no bulk line.
//   refill_returns_a_loan_an_ended_context_owes
//                           llama.cpp-i4ny: a three-triplet loan, and a graph entry with nothing owed returns nothing.
//                           The lending context's free owes all three to the next context, moves nothing, and its
//                           report reads "0/3 triplets re-admitted (3 owed to the next context)". The next context's
//                           first graph entry returns the three in one transaction, with no staged copy and one
//                           "bulk refill ... between contexts" WARN; a second entry has nothing to return, and the
//                           next report reads "3/0" with no debt.
//   refill_returns_a_partial_bulk_refills_leftovers_per_graph
//                           llama.cpp-i4ny: a three-triplet loan, a prompt whose ring never engaged, and a WEIGHT
//                           zone with room for one triplet. The first decode boundary's bulk refill re-admits that
//                           one, accepted, with exactly one "1 of 3" bulk line; two stay lent and the debt is spent.
//                           With the room back, the next decode boundary re-admits one per graph, no second line.
//   refill_returns_a_refused_bulk_refills_loan_per_graph
//                           llama.cpp-i4ny: the same with no room. The bulk refill is refused (WEIGHT zone full)
//                           with no bulk line, all three stay lent and the debt is spent; with the room back, the
//                           next decode boundary re-admits one per graph, with no bulk line.
//
// The fixture configures the device's host zones, as a model load does before anything runs, so each lent copy is a
// slice of the WEIGHT host zone: the provenance the copy path reads as host USM, and so the one a refill can copy from
// without staging. Every lend in the refill cases checks it. The zones cannot be unconfigured, so the one case that
// needs them absent runs first and fails, rather than passing vacuously, if they are already there.
//
// The ctest registration sets GGML_SYCL_STRICT_LEASES=1, so an ownerless lease anywhere in the loan or the refill
// aborts the run. Each case puts back the plan publication it found and empties the lent ledger, so the next one stages
// under the same coherence. A setup failure names the cache's reason. The shared-storage case reserves a WEIGHT arena,
// so the loan cases before it stage raw device allocations and the refill cases after it, which mint their storage from
// the zone, stage into it.
//
// MIT license
// SPDX-License-Identifier: MIT
//

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "sycl-test-skip.hpp"

#include <cstdio>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>

static bool run_case(const char * name, std::string (*body)()) {
    fprintf(stderr, "[TEST] %s ... ", name);
    const std::string failure = body();
    if (!failure.empty()) {
        fprintf(stderr, "FAILED: %s\n", failure.c_str());
        return false;
    }
    fprintf(stderr, "PASSED\n");
    return true;
}

int main() {
    std::optional<sycl::device> device = sycl_test_require_gpu("expert loan transaction");
    if (!device) {
        return SYCL_TEST_SKIP;
    }
    if (ggml_backend_sycl_get_device_count() == 0 || ggml_sycl_info().total_gpu_count <= 0) {
        sycl_test_print_skip("expert loan transaction backend device");
        return SYCL_TEST_SKIP;
    }
    fprintf(stderr, "Device: %s\n", device->get_info<sycl::info::device::name>().c_str());

    bool ok = true;
    ok &= run_case("refuses_without_host_zones", ggml_sycl::test_expert_loan_refuses_without_host_zones);
    ok &= run_case("lends_two_triplets", ggml_sycl::test_expert_loan_lends_two_triplets);
    ok &= run_case("probe_changes_nothing", ggml_sycl::test_expert_loan_probe_changes_nothing);
    ok &= run_case("refuses_replacing_a_host_entry", ggml_sycl::test_expert_loan_refuses_replacing_a_host_entry);
    ok &= run_case("refuses_shared_storage", ggml_sycl::test_expert_loan_refuses_shared_storage);
    ok &= run_case("direct_stage_outside_a_load_copies_through_staging",
                   ggml_sycl::test_expert_direct_stage_outside_a_load_copies_through_staging);
    ok &= run_case("refill_readmits_one_triplet_per_call", ggml_sycl::test_expert_refill_readmits_one_triplet_per_call);
    ok &= run_case("refill_cancelled_by_a_loan", ggml_sycl::test_expert_refill_cancelled_by_a_loan);
    ok &= run_case("refill_runs_once_per_decode_graph", ggml_sycl::test_expert_refill_runs_once_per_decode_graph);
    ok &= run_case("refill_drops_an_unpreparable_triplet", ggml_sycl::test_expert_refill_drops_an_unpreparable_triplet);
    ok &= run_case("refill_moves_two_triplets_per_call", ggml_sycl::test_expert_refill_moves_two_triplets_per_call);
    ok &= run_case("loan_sizes_its_copies_from_the_entry", ggml_sycl::test_expert_loan_sizes_its_copies_from_the_entry);
    ok &= run_case("loan_and_refill_stop_on_a_missing_entry",
                   ggml_sycl::test_expert_loan_and_refill_stop_on_a_missing_entry);
    ok &= run_case("loan_reclaims_pointer_table_leases", ggml_sycl::test_expert_loan_reclaims_pointer_table_leases);
    ok &= run_case("loan_reclaims_after_a_timed_out_boundary_drain",
                   ggml_sycl::test_expert_loan_reclaims_after_a_timed_out_boundary_drain);
    ok &= run_case("refill_returns_an_unengaged_prompts_loan_at_once",
                   ggml_sycl::test_expert_refill_returns_an_unengaged_prompts_loan_at_once);
    ok &= run_case("refill_keeps_the_per_graph_refill_after_an_engaged_prompt",
                   ggml_sycl::test_expert_refill_keeps_the_per_graph_refill_after_an_engaged_prompt);
    ok &= run_case("refill_returns_a_loan_an_ended_context_owes",
                   ggml_sycl::test_expert_refill_returns_a_loan_an_ended_context_owes);
    ok &= run_case("refill_returns_a_partial_bulk_refills_leftovers_per_graph",
                   ggml_sycl::test_expert_refill_returns_a_partial_bulk_refills_leftovers_per_graph);
    ok &= run_case("refill_returns_a_refused_bulk_refills_loan_per_graph",
                   ggml_sycl::test_expert_refill_returns_a_refused_bulk_refills_loan_per_graph);

    fprintf(stderr, "%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
