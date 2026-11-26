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
//                           A host entry planted under the selected expert's host id, checked live and idle (no
//                           lease, not retired) before the loan, refuses the loan with "host entry exists" and a
//                           named evict of the same triplet with REFUSED "host entry exists", since a live entry
//                           never goes by itself; nothing is copied or published and the planted entry keeps its
//                           storage.
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
//   loan_and_refill_size_from_the_tensor
//                           Each device entry staged as a Q4_0 expert of 640 columns, 216 bytes past the tensor's
//                           bytes per expert. Host storage short of those bytes is refused "host storage too small",
//                           and an expert longer than its entry "device entry size", each moving nothing. The loan
//                           lends two triplets, each host copy and record the tensor's bytes, reporting the six
//                           entries' whole sizes freed. Over a poisoned WEIGHT zone, one decode boundary re-admits
//                           both at the staged size, records over the tensor's bytes, each tail in poisoned storage
//                           reading zero.
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
//   loan_on_a_mapping_plan_publishes_its_lent_members_as_pinned
//                           A plan that maps files, its device entries FILE_MAPPED as a mapping load leaves them: a
//                           one-triplet loan publishes the lent members off the device and PINNED, the pinned copy
//                           it made, and the refill publishes them on the device and FILE_MAPPED again. Each reads
//                           FILE_MAPPED on the device before the loan. Then the same on a plan that maps nothing,
//                           PINNED throughout.
//
// The named move's admit, through the same transaction, each host triplet copied to the pinned pool and registered
// as S1-PRELOAD registers an expert the plan keeps on the host:
//
//   admit_zero_fills_a_padded_tail
//                           The resolver reports Q4_0 rows of 640 columns, so each device copy is minted 216 bytes
//                           past the host bytes. With the free WEIGHT zone poisoned first (each tail checked to lie in
//                           poisoned storage), one admit moves the triplet: accepted with its outcome "OK", three
//                           staged sizes admitted, no staged copy, a new plan, no ledger or teardown count written.
//                           Each record covers the tensor's bytes from offset 0, each entry resolves on the device at
//                           the staged size, the payload equals the host bytes and the tail reads zero.
//   admit_refuses_a_file_mapped_member
//                           A triplet whose gate copy is served from a file mapping, named before a pinned triplet:
//                           the first is REFUSED "host copy not pinned" and stays on the host with its entries and
//                           records, the second moves, the call is accepted with the refusal as its reason, and the
//                           WEIGHT zone keeps nothing of the refused triplet.
//   admit_is_busy_while_lent_or_the_zone_is_full
//                           With a loan outstanding the admit is BUSY "loans outstanding"; after the refill, with room
//                           for one member, BUSY "WEIGHT zone full" and the room comes back. Neither publishes or
//                           moves anything. With the room back the admit moves the triplet, and the teardown counts
//                           stay the loan's and the refill's, 1/1.
//   admit_returns_the_host_zone_bytes
//                           The WEIGHT host zone counts the three pinned host copies, and within 2 s of the admit their
//                           host entries are gone and the zone uses what it did before they were pinned.
//   admit_completes_a_block_read_through_its_table
//                           The admit of a block's only host triplet leaves every expert of the block on the device.
//                           The three selected-ids pointer tables ggml_sycl_update_moe_ptr_table builds right after
//                           it, with no wait, for a ubatch routing that expert each chain a ready event and name the
//                           admitted device copy, and a kernel behind each upload on the same stream reads the
//                           original bytes through the table.
//   admit_reads_the_tensor_bytes_from_a_longer_host_copy
//                           Each host copy is one staged tail longer than the Q4_0 expert, its tail 0x5a. The admit
//                           mints the tensor's staged size, not the longer copy's, admits three of them, records the
//                           tensor's bytes from offset 0, reads the payload equal and the tail zero, and the WEIGHT
//                           host zone gets the whole longer copies' bytes back.
//   admit_refuses_a_host_copy_shorter_than_the_tensor
//                           The resolver reports a tensor 4 KiB longer per expert than the host copies. The cache
//                           refuses the triplet REFUSED "host copy size", nothing is published, the triplet stays on
//                           the host with its entries and records, and the minted storage's room comes back.
//   admit_completes_a_block_the_direct_route_serves
//                           With the tensors bound Q4_0 and the device info narrowed to device 0, the decode direct
//                           route (ggml_sycl_moe_decode_direct_table) refuses the block while expert 1 is on the host.
//                           After the admit it decides again and serves each tensor from AOS, its table names the
//                           admitted device copy, and a kernel behind the upload reads the bytes through the table.
//
// The named move's evict, through the same transaction:
//
//   evict_moves_a_resident_triplet
//                           One evict: accepted "OK", three staged sizes to give back, a new plan with each member on
//                           the host, no target device and PINNED backing, each member resolving to a pinned host copy
//                           of its original bytes, no ledger or teardown count written. The control: each member
//                           resolved on the device before the call. Its three device entries are all still there right
//                           after the call, and a deferred-free pass reaps them and returns their room.
//   evict_is_not_refilled
//                           An evict, then a one-triplet loan of another: one decode boundary's refill, allowed four,
//                           re-admits the lent triplet alone, the evicted one stays on the host, and the teardown
//                           counts read 1/1.
//   evict_refuses_shared_storage
//                           One call evicts expert 0 of a bulk-staged layer and of a layer staged per expert: the
//                           bulk one is REFUSED "shared storage" and left on the device, and the control, the one
//                           staged per expert, moves in the same call, which is accepted with "shared storage" as its
//                           reason.
//   evict_is_busy_right_after_an_admit
//                           Right after an admit the evict of that triplet is BUSY ("storage still referenced" or
//                           "host entry retiring") and changes nothing; once the copy's retention has drained it is
//                           BUSY "host entry retiring", whatever the timing, until the host entries are reaped; then
//                           it moves.
//   evict_is_busy_while_lent
//                           With a loan outstanding the evict is BUSY "loans outstanding" and changes nothing; after
//                           the refill it moves.
//   evict_and_admit_swap_in_one_call
//                           One call evicts expert 0 and admits expert 2: both "OK" with three staged sizes each, the
//                           block's resident count unchanged, and once the evicted entries are reaped the WEIGHT zone
//                           has the room it had before the call.
//   evict_and_admit_cycles_return_the_room
//                           Four evict and admit cycles of one triplet while a second triplet stays on the device, so
//                           the plan keeps using the card and its cache keeps the plan after every accepted evict and
//                           admit. Right after the first evict an admit is BUSY "device entry exists"; after each
//                           cycle's reaping the WEIGHT zone and the WEIGHT host zone are where they started, and the
//                           bytes at the end are the originals. The control, in every cycle: right after the evict the
//                           WEIGHT host zone holds the three pinned copies on top of where it started, and once the
//                           device entries are reaped the WEIGHT zone has their room on top of where it started.
//   evict_refusal_leaves_the_admit_list_alone
//                           An evict of a host triplet is REFUSED "not on device" in its list while the admit beside it
//                           moves: accepted with that refusal as the reason, none evicted, one admitted.
//   evict_room_is_not_an_admit_in_the_same_call
//                           With the WEIGHT zone full (the control: no room for one member), one call evicts expert 0
//                           and admits expert 2: the evict moves, the admit is BUSY "WEIGHT zone full", and right after
//                           the call the zone still has no member's room. Once expert 0's entries are reaped, a later
//                           call admits expert 2.
//   evict_takes_a_block_off_the_direct_route
//                           The control: once an admit completes the block, the decode direct route serves each tensor
//                           from AOS. The evict of expert 0 moves each tensor's storage generation, and the route then
//                           refuses the block rather than serve its old table.
//   evict_without_host_room_is_busy_and_keeps_the_zone
//                           The control: fillers that never grow the host WEIGHT zone fill it. The evict of expert 0 is
//                           then BUSY "host pinned pool exhausted", moves nothing and leaves the zone's capacity as it
//                           was; once the fillers go, the same evict moves, still without growing the zone.
//   evict_walks_no_entry_to_hand_over_its_id
//                           The control: before the evict each of expert 0's ids names its device entry. The evict and
//                           the reaping of its device entries leave the id remap's walk count where it was, and each
//                           id then names its host copy.
//   loan_rollback_hands_the_id_back_to_the_device_entry
//                           The control: expert 0 resolves on the device through its id. A loan of it whose first host
//                           registration fails is refused "host registration failed" and leaves no host copy; each id
//                           names its device entry again, each member resolves on the device, and the walk count is
//                           where it was.
//   moves_leave_the_eviction_latch_clear
//                           The control: a drop of expert 1's device entries outside any move sets the cache's eviction
//                           latch on their reap. A named evict of expert 0, the reap of its device entries, its admit
//                           back and the reap of its host copies, before it, leave the latch clear.
//   move_whose_swap_lost_sets_the_eviction_latch
//                           The control: the latch is clear after its reset. The plan is published as a model's
//                           lifecycle row, which the publication hook erases during the evict's host registration, so
//                           the evict's plan swap loses: it is REFUSED and the latch is set.
//   failed_moved_copy_hands_its_id_back
//                           The control: the same registration without the failure succeeds and maps the id to its host
//                           copy. With expert 0's device entry retired exactly (its id mapping gone, the extra's record
//                           keeping it), a host copy registered in its place whose mirror allocation fails is refused,
//                           leaves no host copy, and leaves the id naming the device entry.
//
// Who owns a moved copy. Each case claims some triplets for a model as a load claims what it staged, moves one, then
// releases the model as its teardown does, its tensors' records first; they run last, since a release scans the whole
// device cache. A case that fails after its claim leaves the slot to the fixture, which releases it, so the next case
// does not find it live:
//
//   admit_belongs_to_the_host_copys_model
//                           A model claims expert 1's host copies and expert 0's device entries. After the admit of
//                           expert 1 and the release, expert 1's device entries and expert 0's are gone and the WEIGHT
//                           zone has the admitted triplet's room back. Expert 1's host copies go too, but as retired
//                           entries the deferred-free pass removes whatever their tags. The control: expert 2, never
//                           claimed, keeps its device entries, since a release keeps an entry it cannot attribute.
//   evict_host_copy_belongs_to_the_device_entrys_model
//                           A model claims experts 0 and 1. After the evict of expert 0 and the reaping of its device
//                           entries, the release reclaims expert 0's host copies, with their room in the host WEIGHT
//                           zone, and expert 1's device entries. The control: expert 2, never claimed, stays.
//   evict_host_copy_takes_the_sources_pending_load
//                           A load marks expert 0's device entries and has not ended when expert 0 is evicted. Once
//                           the withdrawn device entries are reaped, each host copy is pending under that load; the
//                           load's end tags exactly the three copies, and the release reclaims them with their room in
//                           the host WEIGHT zone. The control: experts 1 and 2, never claimed, stay.
//   refill_belongs_to_the_lending_model
//                           A model claims experts 0 and 1, the only ones on the device; one is lent and refilled.
//                           The release reclaims both triplets' device entries. The refilled triplet's host copies go
//                           too, but as retired entries the deferred-free pass removes whatever their tags. No
//                           unclaimed control: the loan picks its own triplet, which must be a claimed one.
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
    const std::string failure   = body();
    // The fixture's teardown ran inside body(): a drain that timed out there fails the case it ended.
    const int         torn_down = ggml_sycl::test_expert_loan_take_teardown_failures();
    if (!failure.empty()) {
        fprintf(stderr, "FAILED: %s\n", failure.c_str());
        return false;
    }
    if (torn_down != 0) {
        fprintf(stderr, "FAILED: the fixture's teardown did not settle (%d drain timeout(s))\n", torn_down);
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
    ok &= run_case("loan_and_refill_size_from_the_tensor", ggml_sycl::test_expert_loan_and_refill_size_from_the_tensor);
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
    ok &= run_case("loan_on_a_mapping_plan_publishes_its_lent_members_as_pinned",
                   ggml_sycl::test_expert_loan_on_a_mapping_plan_publishes_its_lent_members_as_pinned);
    ok &= run_case("admit_zero_fills_a_padded_tail", ggml_sycl::test_expert_admit_zero_fills_a_padded_tail);
    ok &= run_case("admit_refuses_a_file_mapped_member", ggml_sycl::test_expert_admit_refuses_a_file_mapped_member);
    ok &= run_case("admit_is_busy_while_lent_or_the_zone_is_full",
                   ggml_sycl::test_expert_admit_is_busy_while_lent_or_the_zone_is_full);
    ok &= run_case("admit_returns_the_host_zone_bytes", ggml_sycl::test_expert_admit_returns_the_host_zone_bytes);
    ok &= run_case("admit_completes_a_block_read_through_its_table",
                   ggml_sycl::test_expert_admit_completes_a_block_read_through_its_table);
    ok &= run_case("admit_reads_the_tensor_bytes_from_a_longer_host_copy",
                   ggml_sycl::test_expert_admit_reads_the_tensor_bytes_from_a_longer_host_copy);
    ok &= run_case("admit_refuses_a_host_copy_shorter_than_the_tensor",
                   ggml_sycl::test_expert_admit_refuses_a_host_copy_shorter_than_the_tensor);
    ok &= run_case("admit_completes_a_block_the_direct_route_serves",
                   ggml_sycl::test_expert_admit_completes_a_block_the_direct_route_serves);
    ok &= run_case("evict_moves_a_resident_triplet", ggml_sycl::test_expert_evict_moves_a_resident_triplet);
    ok &= run_case("evict_is_not_refilled", ggml_sycl::test_expert_evict_is_not_refilled);
    ok &= run_case("evict_refuses_shared_storage", ggml_sycl::test_expert_evict_refuses_shared_storage);
    ok &= run_case("evict_is_busy_right_after_an_admit", ggml_sycl::test_expert_evict_is_busy_right_after_an_admit);
    ok &= run_case("evict_is_busy_while_lent", ggml_sycl::test_expert_evict_is_busy_while_lent);
    ok &= run_case("evict_and_admit_swap_in_one_call", ggml_sycl::test_expert_evict_and_admit_swap_in_one_call);
    ok &= run_case("evict_and_admit_cycles_return_the_room",
                   ggml_sycl::test_expert_evict_and_admit_cycles_return_the_room);
    ok &= run_case("evict_refusal_leaves_the_admit_list_alone",
                   ggml_sycl::test_expert_evict_refusal_leaves_the_admit_list_alone);
    ok &= run_case("evict_room_is_not_an_admit_in_the_same_call",
                   ggml_sycl::test_expert_evict_room_is_not_an_admit_in_the_same_call);
    ok &= run_case("evict_takes_a_block_off_the_direct_route",
                   ggml_sycl::test_expert_evict_takes_a_block_off_the_direct_route);
    ok &= run_case("evict_without_host_room_is_busy_and_keeps_the_zone",
                   ggml_sycl::test_expert_evict_without_host_room_is_busy_and_keeps_the_zone);
    ok &= run_case("evict_walks_no_entry_to_hand_over_its_id",
                   ggml_sycl::test_expert_evict_walks_no_entry_to_hand_over_its_id);
    ok &= run_case("loan_rollback_hands_the_id_back_to_the_device_entry",
                   ggml_sycl::test_expert_loan_rollback_hands_the_id_back_to_the_device_entry);
    ok &= run_case("moves_leave_the_eviction_latch_clear", ggml_sycl::test_expert_moves_leave_the_eviction_latch_clear);
    ok &= run_case("move_whose_swap_lost_sets_the_eviction_latch",
                   ggml_sycl::test_expert_move_whose_swap_lost_sets_the_eviction_latch);
    ok &= run_case("failed_moved_copy_hands_its_id_back", ggml_sycl::test_expert_failed_moved_copy_hands_its_id_back);
    // Last: each releases a model slot, a teardown scan of the whole device cache.
    ok &=
        run_case("admit_belongs_to_the_host_copys_model", ggml_sycl::test_expert_admit_belongs_to_the_host_copys_model);
    ok &= run_case("evict_host_copy_belongs_to_the_device_entrys_model",
                   ggml_sycl::test_expert_evict_host_copy_belongs_to_the_device_entrys_model);
    ok &= run_case("evict_host_copy_takes_the_sources_pending_load",
                   ggml_sycl::test_expert_evict_host_copy_takes_the_sources_pending_load);
    ok &= run_case("refill_belongs_to_the_lending_model", ggml_sycl::test_expert_refill_belongs_to_the_lending_model);

    fprintf(stderr, "%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
