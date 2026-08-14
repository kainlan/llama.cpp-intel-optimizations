// Deterministic failure-injection fixture for the transactional
// staging-replacement region of ggml-sycl.cpp (4f01993c9).
//
// The bar every scenario here is written against: with a seam armed, the staging
// path must take its drain/rollback branch, and every owner it was holding --
// source, staging, destination and the old escrow -- must still release exactly
// once. Release counts come from an injected release backend, so "exactly once"
// is measured rather than inferred from the absence of a crash.
//
// ⚠️ A green scenario with a reached-counter of 0 armed nothing and proved
// nothing. Every scenario asserts its seam actually fired before it scores the
// outcome, and the arithmetic scenarios additionally assert the unarmed
// traversal count so a seam site that fell off the path shows up as a failure
// rather than as a quiet pass.
//
// Coverage note, stated so it is not read as more than it is: four of the six
// seams (barrier submission, retention publication, arithmetic boundary,
// same-allocation host-vector) are driven end-to-end here without a device. The
// remaining two (forced allocation failure, resize while prior secondary work is
// in flight) sit in split_secondary_gpu_ensure() / convert_tensor_layout(),
// which need real device allocations to reach; this fixture proves only that
// they are armable and that nothing on the host path consumes them. Their
// end-to-end execution belongs to the hardware pass.

#include "staging-replacement-failpoint-seam.hpp"
#include "sycl-test-skip.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace ggml_sycl;

namespace {

struct counting_release_backend {
    std::atomic<int> attempts{ 0 };
    std::atomic<int> releases{ 0 };
};

release_attempt release_counting(const alloc_metadata &, void * opaque) noexcept {
    auto & backend = *static_cast<counting_release_backend *>(opaque);
    backend.attempts.fetch_add(1, std::memory_order_relaxed);
    backend.releases.fetch_add(1, std::memory_order_relaxed);
    return { release_attempt_status::RELEASED };
}

[[noreturn]] void fail(const char * message) {
    fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

alloc_metadata staging_metadata(uint64_t id) {
    alloc_metadata value;
    value.ptr        = reinterpret_cast<void *>(static_cast<uintptr_t>(0x20000 + id * 0x1000));
    value.size       = 4096;
    value.device     = 0;
    value.id         = id;
    value.generation = 3;
    value.offset     = 0;
    value.extent     = 4096;
    value.epoch_id   = 1;
    value.kind       = alloc_tier::HOST_PINNED;
    value.role       = alloc_role::STAGING;
    value.category   = runtime_category::STAGING;
    value.zone       = vram_zone_id::RUNTIME;
    value.host_zone  = host_zone_id::STAGING;
    return value;
}

// One escrow participant: a mem_handle over a control whose physical release is
// the counting backend, plus the coordinator that reports live controls.
struct counted_owner {
    mem_handle                                      handle;
    std::shared_ptr<allocation_release_coordinator> coordinator;
};

counted_owner make_counted_owner(counting_release_backend & backend, uint64_t id) {
    auto fixture = allocation_owner_test_create(staging_metadata(id), release_counting, &backend);
    check(static_cast<bool>(fixture.result), "escrow owner creation failed");
    counted_owner owner;
    owner.coordinator = fixture.coordinator;
    owner.handle      = mem_handle::from_owned_alloc(std::move(fixture.result.owner), GGML_LAYOUT_AOS);
    check(owner.handle.valid(), "escrow owner did not produce a valid handle");
    return owner;
}

void reset_seams() {
    ggml_sycl_staging_failpoint_reset();
    for (int fp = 0; fp < GGML_SYCL_STAGING_FAILPOINT_COUNT; ++fp) {
        const auto seam = static_cast<ggml_sycl_staging_failpoint>(fp);
        check(ggml_sycl_staging_failpoint_reached(seam) == 0, "reset left a reached counter set");
        check(ggml_sycl_staging_failpoint_traversals(seam) == 0, "reset left a traversal counter set");
        check(ggml_sycl_staging_failpoint_armed_remaining(seam) == 0, "reset left a seam armed");
    }
    check(ggml_sycl_staging_failpoint_retirement_drains() == 0, "reset left the rollback drain counter set");
}

// -------------------------------------------------------------------------
// Control: an armed seam that is never traversed must report zero, so a later
// non-zero reading cannot be an artefact of arming alone.
// -------------------------------------------------------------------------
void arming_alone_reaches_nothing() {
    reset_seams();
    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY, 2);
    check(ggml_sycl_staging_failpoint_armed_remaining(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 2,
          "arming did not record the requested fire count");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 0,
          "arming alone reported the seam as reached");
    check(ggml_sycl_staging_failpoint_traversals(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 0,
          "arming alone reported a traversal");
    reset_seams();
    printf("PASS arming-alone-reaches-nothing\n");
}

// -------------------------------------------------------------------------
// The checked arithmetic must reject real boundary operands with no seam
// involved. This is the acceptance criterion the seam cannot stand in for.
// -------------------------------------------------------------------------
void checked_arithmetic_rejects_real_boundaries() {
    reset_seams();
    constexpr size_t max = std::numeric_limits<size_t>::max();
    size_t           out = 0xdeadbeef;

    check(!ggml_sycl_staging_failpoint_checked_add(max, 1, &out), "checked add accepted an overflowing sum");
    check(out == 0xdeadbeef, "rejected checked add wrote its output");
    check(ggml_sycl_staging_failpoint_checked_add(max - 1, 1, &out) && out == max,
          "checked add rejected the largest representable sum");
    check(ggml_sycl_staging_failpoint_checked_add(max, 0, &out) && out == max, "checked add rejected a zero addend");
    check(!ggml_sycl_staging_failpoint_checked_add(1, 1, nullptr), "checked add accepted a null output");

    check(!ggml_sycl_staging_failpoint_checked_mul(max / 2 + 1, 2, &out),
          "checked mul accepted an overflowing product");
    check(ggml_sycl_staging_failpoint_checked_mul(max / 2, 2, &out) && out == (max / 2) * 2,
          "checked mul rejected a representable product");
    check(ggml_sycl_staging_failpoint_checked_mul(0, max, &out) && out == 0, "checked mul rejected a zero factor");
    check(!ggml_sycl_staging_failpoint_checked_mul(2, 2, nullptr), "checked mul accepted a null output");

    check(!ggml_sycl_staging_failpoint_checked_round_up(16, 0, &out), "round-up accepted a zero alignment");
    check(!ggml_sycl_staging_failpoint_checked_round_up(max, 2, &out),
          "round-up accepted an operand whose padding overflows");
    check(ggml_sycl_staging_failpoint_checked_round_up(16, 8, &out) && out == 16,
          "round-up altered an already aligned value");
    check(ggml_sycl_staging_failpoint_checked_round_up(17, 8, &out) && out == 24, "round-up produced a wrong result");
    check(!ggml_sycl_staging_failpoint_checked_round_up(17, 8, nullptr), "round-up accepted a null output");

    check(ggml_sycl_staging_failpoint_traversals(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) > 0,
          "the arithmetic seam site was never traversed -- it has fallen off the path under test");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 0,
          "an unarmed arithmetic seam fired");
    reset_seams();
    printf("PASS checked-arithmetic-rejects-real-boundaries\n");
}

// -------------------------------------------------------------------------
// Armed, the same helpers must take the rejection branch on operands that are
// nowhere near a boundary -- that is what lets a caller deep in the staging path
// be driven into rejection without synthesising a 2^64-byte tensor.
// -------------------------------------------------------------------------
void armed_arithmetic_forces_rejection() {
    reset_seams();
    size_t out = 7;

    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY, 1);
    check(!ggml_sycl_staging_failpoint_checked_add(1, 1, &out), "armed checked add still accepted its operands");
    check(out == 7, "armed checked add wrote its output before rejecting");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 1,
          "armed checked add did not record a fire");
    check(ggml_sycl_staging_failpoint_armed_remaining(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 0,
          "a single-fire arm survived its fire");
    check(ggml_sycl_staging_failpoint_checked_add(2, 2, &out) && out == 4,
          "the seam stayed armed past its requested fire count");

    reset_seams();
    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY, 1);
    check(!ggml_sycl_staging_failpoint_checked_mul(2, 2, &out), "armed checked mul still accepted its operands");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 1,
          "armed checked mul did not record a fire");

    // Round-up reaches the seam through its own call to the checked addition,
    // so only the unaligned case fires -- the aligned case never adds.
    reset_seams();
    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY, 1);
    check(!ggml_sycl_staging_failpoint_checked_round_up(17, 8, &out), "armed round-up still accepted its operands");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_ARITHMETIC_BOUNDARY) == 1,
          "armed round-up did not record a fire");

    reset_seams();
    printf("PASS armed-arithmetic-forces-rejection\n");
}

// -------------------------------------------------------------------------
// The two pointer-table views are built as one unit. A failure building them
// must leave the caller's existing views byte-for-byte intact, because a
// half-rebuilt pair reaching publication is the bug this shape exists to avoid.
// -------------------------------------------------------------------------
void host_vector_failure_leaves_prior_views_intact() {
    reset_seams();
    std::vector<mem_handle> handles;
    std::vector<void *>     payload;

    check(ggml_sycl_staging_failpoint_build_table_views(4, &handles, &payload),
          "unarmed table-view construction failed");
    check(handles.size() == 4 && payload.size() == 4, "table-view construction produced the wrong extents");
    check(payload[0] == nullptr && payload[3] == nullptr, "table-view payload was not zero initialised");
    check(ggml_sycl_staging_failpoint_traversals(GGML_SYCL_STAGING_FAILPOINT_HOST_VECTOR) == 1,
          "the host-vector seam site was never traversed");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_HOST_VECTOR) == 0,
          "an unarmed host-vector seam fired");

    // Mark the published views so a partial rebuild would be visible.
    void * const sentinel = reinterpret_cast<void *>(static_cast<uintptr_t>(0xfeedfaceu));
    payload[2]            = sentinel;

    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_HOST_VECTOR, 1);
    check(!ggml_sycl_staging_failpoint_build_table_views(9, &handles, &payload),
          "armed table-view construction reported success");
    check(ggml_sycl_staging_failpoint_reached(GGML_SYCL_STAGING_FAILPOINT_HOST_VECTOR) == 1,
          "armed table-view construction did not record a fire");
    check(handles.size() == 4 && payload.size() == 4, "a failed rebuild resized the published views");
    check(payload[2] == sentinel, "a failed rebuild overwrote the published payload");

    reset_seams();
    printf("PASS host-vector-failure-leaves-prior-views-intact\n");
}

// -------------------------------------------------------------------------
// An empty escrow secures without submitting anything, so no drain is owed.
// This is also the only success control that is safe without a device.
// -------------------------------------------------------------------------
void empty_escrow_secures_without_submission() {
    reset_seams();
    const std::vector<mem_handle> none;
    check(ggml_sycl_staging_failpoint_run_retirement(nullptr, none), "an empty retirement failed to secure");
    check(ggml_sycl_staging_failpoint_traversals(GGML_SYCL_STAGING_FAILPOINT_BARRIER_SUBMIT) == 0,
          "an empty retirement reached the barrier");
    check(ggml_sycl_staging_failpoint_retirement_drains() == 0, "an empty retirement drained");
    reset_seams();
    printf("PASS empty-escrow-secures-without-submission\n");
}

// -------------------------------------------------------------------------
// The core claim. With the seam armed the transaction must roll back, drain
// once, and leave every one of the four escrow participants releasable exactly
// once by its real owner -- not released early by the escrow, and not leaked.
// -------------------------------------------------------------------------
void retirement_rollback_releases_every_owner_exactly_once(ggml_sycl_staging_failpoint seam,
                                                           sycl::queue *               queue,
                                                           const char *                scenario) {
    reset_seams();
    counting_release_backend backend;

    // Named for the four owners the acceptance criterion enumerates.
    counted_owner source      = make_counted_owner(backend, 1);
    counted_owner staging     = make_counted_owner(backend, 2);
    counted_owner destination = make_counted_owner(backend, 3);
    counted_owner old_escrow  = make_counted_owner(backend, 4);

    std::vector<mem_handle> owners{ source.handle, staging.handle, destination.handle, old_escrow.handle };

    ggml_sycl_staging_failpoint_arm(seam, 1);
    const bool secured = ggml_sycl_staging_failpoint_run_retirement(queue, owners);

    check(!secured, "an armed retirement reported success");
    check(ggml_sycl_staging_failpoint_reached(seam) == 1,
          "the armed retirement seam never fired -- this run proves nothing");
    check(ggml_sycl_staging_failpoint_retirement_drains() == 1,
          "the rollback did not take the drain branch exactly once");
    check(backend.attempts.load() == 0, "the failed transaction released an owner its caller still holds");

    // Drop the caller-side copies. Only now may each allocation be released,
    // and each exactly once.
    owners.clear();
    check(backend.attempts.load() == 0, "dropping the transaction's copies released a live allocation");

    source.handle      = mem_handle{};
    staging.handle     = mem_handle{};
    destination.handle = mem_handle{};
    old_escrow.handle  = mem_handle{};

    check(backend.attempts.load() == 4 && backend.releases.load() == 4,
          "the four escrow owners did not release exactly once each");
    check(source.coordinator->live_controls() == 0 && staging.coordinator->live_controls() == 0 &&
              destination.coordinator->live_controls() == 0 && old_escrow.coordinator->live_controls() == 0,
          "a rolled-back transaction leaked a live control");

    reset_seams();
    printf("PASS retirement-rollback-releases-exactly-once (%s)\n", scenario);
}

// -------------------------------------------------------------------------
// The two device-sited seams. Host-side this can only establish that they are
// armable and that nothing on the host path consumes them; say exactly that
// rather than letting a green line imply the failure paths were executed.
// -------------------------------------------------------------------------
void device_sited_seams_are_armable_and_unconsumed() {
    reset_seams();
    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_ALLOCATION, 1);
    ggml_sycl_staging_failpoint_arm(GGML_SYCL_STAGING_FAILPOINT_RESIZE_IN_FLIGHT, 3);
    check(ggml_sycl_staging_failpoint_armed_remaining(GGML_SYCL_STAGING_FAILPOINT_ALLOCATION) == 1 &&
              ggml_sycl_staging_failpoint_armed_remaining(GGML_SYCL_STAGING_FAILPOINT_RESIZE_IN_FLIGHT) == 3,
          "a device-sited seam refused to arm");
    check(ggml_sycl_staging_failpoint_traversals(GGML_SYCL_STAGING_FAILPOINT_ALLOCATION) == 0 &&
              ggml_sycl_staging_failpoint_traversals(GGML_SYCL_STAGING_FAILPOINT_RESIZE_IN_FLIGHT) == 0,
          "a device-sited seam was consumed by the host path");
    reset_seams();
    printf(
        "PASS device-sited-seams-armable "
        "(mechanism only; forced allocation failure and in-flight resize execute on hardware)\n");
}

}  // namespace

int main() {
    arming_alone_reaches_nothing();
    checked_arithmetic_rejects_real_boundaries();
    armed_arithmetic_forces_rejection();
    host_vector_failure_leaves_prior_views_intact();
    empty_escrow_secures_without_submission();
    retirement_rollback_releases_every_owner_exactly_once(GGML_SYCL_STAGING_FAILPOINT_BARRIER_SUBMIT, nullptr,
                                                          "barrier submission, no queue");
    retirement_rollback_releases_every_owner_exactly_once(GGML_SYCL_STAGING_FAILPOINT_RETENTION_PUBLISH, nullptr,
                                                          "retention publication, no queue");
    device_sited_seams_are_armable_and_unconsumed();

    // Everything above ran without a device. The remaining scenario repeats the
    // rollback against a live queue, which is where the drain has something to
    // wait on; without a device there is nothing to prove and the run must say
    // so rather than exit 0 on a vacuous pass.
    auto device = sycl_test_require_gpu("the staging-replacement rollback against a live queue");
    if (!device) {
        return SYCL_TEST_SKIP;
    }

    sycl::queue queue(*device, sycl::property::queue::in_order{});
    retirement_rollback_releases_every_owner_exactly_once(GGML_SYCL_STAGING_FAILPOINT_BARRIER_SUBMIT, &queue,
                                                          "barrier submission, live queue");
    retirement_rollback_releases_every_owner_exactly_once(GGML_SYCL_STAGING_FAILPOINT_RETENTION_PUBLISH, &queue,
                                                          "retention publication, live queue");

    printf("All staging-replacement failpoint scenarios passed\n");
    return 0;
}
