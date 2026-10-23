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
//                           are gone within 2 s and the device gets their storage back: the entries' sizes, which
//                           equal the reported charge only because a 1 MiB expert is 256-byte aligned.
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
//                           6 charges of bytes, 1 remaining, no staged copy), each re-admitted id resolves right away
//                           to a device entry IN_PROGRESS behind its copy, and the third stays lent until the next
//                           decode boundary re-admits it. The teardown counts read 3/3.
//   loan_sizes_its_copies_from_the_entry
//                           llama.cpp-pu8v: each device entry 72 bytes past the plan's dst_size, as S1-PRELOAD pads a
//                           Q4_0 expert whose rows are not a multiple of MATRIX_ROW_PADDING. A host copy minted at the
//                           plan's size is refused "host storage too small" and moves nothing; then the loan lends two
//                           triplets, each host copy the entry's size with its zero tail, and one decode boundary
//                           re-admits both at that size.
//   loan_reclaims_pointer_table_leases
//                           llama.cpp-6aj4: one layer's three tensors each get the pointer table a prompt graph's
//                           preload builds (ggml_sycl_update_moe_ptr_table over the routed device experts), and each
//                           table leases expert 2's device entry (checked first). The one-triplet loan takes expert 2:
//                           within 2 s the boundary pass erases its three device entries and the WEIGHT zone gets
//                           their room back, and the first decode boundary re-admits the triplet. A failure names
//                           each entry's lease count and the extra's handles on it per view, counted without
//                           resolving any.
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
    ok &= run_case("loan_reclaims_pointer_table_leases", ggml_sycl::test_expert_loan_reclaims_pointer_table_leases);

    fprintf(stderr, "%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
