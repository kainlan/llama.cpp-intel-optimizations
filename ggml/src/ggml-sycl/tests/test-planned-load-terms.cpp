// Host-only gate for the owner-keyed planned load terms (planned-load-terms.hpp, llama.cpp-ihk6).
//
// The compute and state terms a load reserves were device-global scalars: a second model's reservation merged into
// the first's, and the only reset was a whole-table clear when no model was live. So a model that stayed live beside
// a larger one that had unloaded kept drawing the larger term, inflating every later context's RUNTIME zone and
// KV-ring admission on that device. Canonical contract 12.4 keys every reset by an explicit owner; here each owner
// (model slot, slot generation) keeps its own contribution, the device reads the merge over the owners on it, and an
// owner's contribution leaves only by its own key.
//
// Host-only, no SYCL: the header takes GGML_SYCL_MAX_DEVICES from ggml-sycl.h (the public include directory) and
// nothing else from the backend, and the table is a local value.

#include "planned-load-terms.hpp"

#include <cstdint>
#include <cstdio>

using ggml_sycl::planned_load_term;
using ggml_sycl::planned_load_terms;
using ggml_sycl::planned_load_terms_replaced;
using ggml_sycl::lifecycle::SlotToken;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        g_failures++;
    }
}

constexpr size_t kMiB = 1024ull * 1024ull;

// Two models live on one device: A in slot 0, B in slot 1, each at its first generation.
constexpr SlotToken kA{ 0, 1 };
constexpr SlotToken kB{ 1, 1 };

size_t compute(const planned_load_terms & t, int device) {
    return t.merged(device, planned_load_term::COMPUTE);
}

size_t state(const planned_load_terms & t, int device) {
    return t.merged(device, planned_load_term::STATE);
}

void test_a_re_measure_replaces_the_owners_own_term() {
    printf("a re-measure within one owner replaces its term (it does not merge with itself)\n");
    planned_load_terms t;
    check(t.set(0, kA, planned_load_term::COMPUTE, 100 * kMiB), "the first measure is stored");
    check(compute(t, 0) == 100 * kMiB, "the device reads 100");
    check(t.set(0, kA, planned_load_term::COMPUTE, 60 * kMiB), "the re-measure is stored");
    check(compute(t, 0) == 60 * kMiB, "the device reads 60, not max(100, 60)");
}

void test_the_device_reads_the_merge_of_its_live_owners() {
    printf("A=100 and B=60 live: the device reads 100; releasing A leaves 60; releasing B leaves 0\n");
    planned_load_terms t;
    check(t.set(0, kA, planned_load_term::COMPUTE, 100 * kMiB) && t.set(0, kB, planned_load_term::COMPUTE, 60 * kMiB),
          "both owners stored");
    check(compute(t, 0) == 100 * kMiB, "the merge is the larger, 100");
    check(t.drop(kA), "A's release drops A");
    check(compute(t, 0) == 60 * kMiB, "B stays live and the device reads B's 60, not A's leftover 100");
    check(t.drop(kB), "B's release drops B");
    check(compute(t, 0) == 0, "no owner left: 0");
}

void test_an_aborted_load_leaves_the_others_unchanged() {
    printf("a load that reserves and then aborts leaves the live owners' merge as it was\n");
    planned_load_terms t;
    check(t.set(0, kA, planned_load_term::COMPUTE, 60 * kMiB) && t.set(0, kA, planned_load_term::STATE, 10 * kMiB),
          "A is live with compute 60 and state 10");
    const SlotToken load{ 1, 1 };
    check(t.set(0, load, planned_load_term::COMPUTE, 200 * kMiB) && t.set(0, load, planned_load_term::STATE, 30 * kMiB),
          "the load in flight reserves compute 200 and state 30");
    check(compute(t, 0) == 200 * kMiB && state(t, 0) == 30 * kMiB, "while it is in flight the merge counts it");
    check(t.drop(load), "the rollback drops the load's contribution");
    check(compute(t, 0) == 60 * kMiB && state(t, 0) == 10 * kMiB, "A's 60 and 10 are what the device reads again");
}

void test_both_terms_are_one_owners_and_leave_together() {
    printf("an owner's compute and state terms are separate values that leave together\n");
    planned_load_terms t;
    check(t.set(0, kA, planned_load_term::STATE, 12 * kMiB), "a state term alone is stored");
    check(state(t, 0) == 12 * kMiB && compute(t, 0) == 0, "the state term does not read as a compute term");
    check(t.set(0, kA, planned_load_term::COMPUTE, 40 * kMiB), "the compute term is stored beside it");
    check(state(t, 0) == 12 * kMiB && compute(t, 0) == 40 * kMiB, "storing compute left state alone");
    check(t.drop(kA) && state(t, 0) == 0 && compute(t, 0) == 0, "one drop removes both");
}

void test_two_devices_are_independent() {
    printf("two devices are independent\n");
    planned_load_terms t;
    check(t.set(0, kA, planned_load_term::COMPUTE, 100 * kMiB), "A on device 0");
    check(t.set(1, kB, planned_load_term::COMPUTE, 300 * kMiB), "B on device 1");
    check(compute(t, 0) == 100 * kMiB && compute(t, 1) == 300 * kMiB, "each device reads only its own owners");
    check(t.set(1, kA, planned_load_term::COMPUTE, 20 * kMiB), "A also reserves on device 1");
    check(compute(t, 1) == 300 * kMiB, "device 1 reads max(300, 20)");
    check(t.drop(kA), "A's release");
    check(compute(t, 0) == 0 && compute(t, 1) == 300 * kMiB, "A leaves both devices; B's device 1 term stands");
    check(compute(t, GGML_SYCL_MAX_DEVICES - 1) == 0, "an untouched device reads 0");
}

void test_owners_that_are_not_model_slots_are_refused() {
    printf("an owner that is not a model slot is refused and stores nothing\n");
    planned_load_terms t;
    check(t.set(0, kA, planned_load_term::COMPUTE, 50 * kMiB), "A stores 50 as the baseline");
    const SlotToken buffer_owner{ ggml_sycl::lifecycle::buffer_owner_slot, 1 };
    const SlotToken no_model{ ggml_sycl::lifecycle::no_model_slot, 1 };
    const SlotToken past_the_table{ ggml_sycl::lifecycle::model_slot_count, 1 };
    const SlotToken unminted{ 2, 0 };
    check(!t.set(0, buffer_owner, planned_load_term::COMPUTE, 900 * kMiB), "buffer_owner_slot is refused");
    check(!t.set(0, no_model, planned_load_term::STATE, 900 * kMiB), "no_model_slot is refused");
    check(!t.set(0, past_the_table, planned_load_term::COMPUTE, 900 * kMiB), "slot model_slot_count is refused");
    check(!t.set(0, unminted, planned_load_term::COMPUTE, 900 * kMiB), "generation 0 (never minted) is refused");
    check(compute(t, 0) == 50 * kMiB && state(t, 0) == 0, "nothing was stored: the device still reads A's 50 only");
    check(!t.drop(buffer_owner) && !t.drop(no_model) && !t.drop(unminted), "and none of them can drop anything");
    check(compute(t, 0) == 50 * kMiB, "A's term survived the refused drops");
    check(!t.set(-1, kA, planned_load_term::COMPUTE, 1) &&
              !t.set(GGML_SYCL_MAX_DEVICES, kA, planned_load_term::COMPUTE, 1),
          "a device out of range is refused");
    check(compute(t, -1) == 0 && compute(t, GGML_SYCL_MAX_DEVICES) == 0, "and reads 0");
}

void test_a_slot_generation_is_its_own_owner() {
    printf("a slot's generations are distinct owners: a stale owner cannot write or drop the current one's term\n");
    planned_load_terms t;
    const SlotToken    old_gen{ 3, 4 };
    const SlotToken    new_gen{ 3, 5 };
    check(t.set(0, new_gen, planned_load_term::COMPUTE, 80 * kMiB), "generation 5 of slot 3 stores 80");
    check(!t.drop(old_gen), "generation 4's release drops nothing");
    check(compute(t, 0) == 80 * kMiB, "generation 5's term stands");
    check(!t.set(0, old_gen, planned_load_term::COMPUTE, 500 * kMiB), "generation 4 cannot write over it");
    check(compute(t, 0) == 80 * kMiB, "still 80");
}

bool replaced_none(const planned_load_terms_replaced & r) {
    for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
        if (r.generation[d] != 0) {
            return false;
        }
    }
    return true;
}

void test_a_newer_owner_replaces_a_stale_contribution_and_names_it() {
    printf("a newer generation of a slot replaces a contribution its predecessor left, and says whose it was\n");
    planned_load_terms          t;
    const SlotToken             old_gen{ 7, 1 };
    const SlotToken             new_gen{ 7, 2 };
    planned_load_terms_replaced replaced;
    replaced.generation[0] = 99;
    check(t.set(0, old_gen, planned_load_term::COMPUTE, 300 * kMiB, &replaced) && replaced_none(replaced),
          "a first store replaces nothing");
    check(t.set(0, old_gen, planned_load_term::STATE, 40 * kMiB, &replaced) && replaced_none(replaced),
          "the same owner's second term replaces nothing");
    check(t.set(0, new_gen, planned_load_term::COMPUTE, 100 * kMiB, &replaced) && replaced.generation[0] == 1,
          "the newer owner's store reports generation 1 replaced on device 0");
    check(compute(t, 0) == 100 * kMiB && state(t, 0) == 0,
          "the predecessor's 300 and 40 are gone whole, not merged into the newer owner's value");
    check(!t.drop(old_gen) && t.drop(new_gen) && compute(t, 0) == 0, "only the newer owner's key drops it now");
}

void test_a_newer_owner_sweeps_the_slot_on_every_device() {
    printf("a newer generation that stores on one device removes its predecessor's leftovers on every device\n");
    planned_load_terms          t;
    const SlotToken             old_gen{ 9, 1 };
    const SlotToken             new_gen{ 9, 2 };
    planned_load_terms_replaced replaced;
    check(t.set(0, old_gen, planned_load_term::COMPUTE, 300 * kMiB) &&
              t.set(1, old_gen, planned_load_term::COMPUTE, 200 * kMiB) &&
              t.set(1, old_gen, planned_load_term::STATE, 30 * kMiB),
          "generation 1 left terms on devices 0 and 1 without a drop");
    check(t.set(1, kB, planned_load_term::COMPUTE, 50 * kMiB), "B, another slot, is live on device 1 with 50");
    check(t.set(0, new_gen, planned_load_term::COMPUTE, 100 * kMiB, &replaced), "generation 2 stores on device 0 only");
    check(replaced.generation[0] == 1 && replaced.generation[1] == 1, "the store names generation 1 on both devices");
    bool others_untouched = true;
    for (int d = 2; d < GGML_SYCL_MAX_DEVICES; ++d) {
        others_untouched = others_untouched && replaced.generation[d] == 0;
    }
    check(others_untouched, "and on no device it held nothing on");
    check(compute(t, 0) == 100 * kMiB, "device 0 reads generation 2's 100");
    check(compute(t, 1) == 50 * kMiB && state(t, 1) == 0,
          "device 1 reads B's 50, not generation 1's leftover 200, and its leftover state is gone");
    check(!t.drop(old_gen), "nothing of generation 1 is left to drop");
    const SlotToken stale_gen{ 9, 1 };
    check(!t.set(3, stale_gen, planned_load_term::COMPUTE, 900 * kMiB),
          "generation 1 cannot store on a device the slot holds nothing on, once generation 2 exists anywhere");
    check(compute(t, 3) == 0, "device 3 still reads 0");
}

}  // namespace

int main() {
    test_a_re_measure_replaces_the_owners_own_term();
    test_the_device_reads_the_merge_of_its_live_owners();
    test_an_aborted_load_leaves_the_others_unchanged();
    test_both_terms_are_one_owners_and_leave_together();
    test_two_devices_are_independent();
    test_owners_that_are_not_model_slots_are_refused();
    test_a_slot_generation_is_its_own_owner();
    test_a_newer_owner_replaces_a_stale_contribution_and_names_it();
    test_a_newer_owner_sweeps_the_slot_on_every_device();

    printf("%s (%d failures)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
