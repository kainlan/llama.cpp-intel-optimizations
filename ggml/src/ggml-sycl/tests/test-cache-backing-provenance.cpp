//
// Adversarial cache-backing provenance tests
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// CACHE_BACKING is an authority, not a label: shutdown_unified_cache() exempts
// that class from destructive-teardown refusal (unified-cache.cpp,
// snapshot_allocation_controls(): `preteardown && ownership_class ==
// CACHE_BACKING`). Everything here tries to obtain that class from outside the
// two private mint paths described in allocation-provenance.hpp, and asserts
// that it cannot be reached.
//
// This fixture deliberately never includes allocation-provenance.hpp. That
// header is restricted to unified-cache.cpp and pinned-pool.cpp by
// tests/test-sycl-owner-allocation-migration.py, and a forgery test that had to
// be let inside the restriction would be testing the wrong surface. The
// direct-token forgery is therefore NOT here: it is a COMPILE-TIME guarantee,
// and a running program cannot witness one. It lives in
// tests/test-cache-backing-forgery-compile.py, which compiles probe TUs and
// requires them to be REJECTED.
//
// Case ordering in main() is load-bearing, and the cases below are defined in
// that RUN order (1, 5, 3, 4, 2) rather than numeric order; see the comments in
// main() for why each has to sit where it does.

#include "../unified-cache.hpp"
#include "sycl-test-skip.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

using namespace ggml_sycl;

namespace {

[[noreturn]] void fail(const char * message) {
    fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

// ---------------------------------------------------------------------------
// Case 1 (partial): the public request structs cannot even NAME backing.
//
// The strongest form of "a caller cannot forge this" is that the field does not
// exist on any type a caller can fill in. Detecting the ABSENCE of a member is
// the classic vacuous check -- a detector with a typo reports "absent" for
// everything -- so the detector is scored against a struct that certainly has
// the member before it is trusted against the real one.
// ---------------------------------------------------------------------------
template <typename T, typename = void> struct has_cache_backing_member : std::false_type {};

template <typename T>
struct has_cache_backing_member<T, decltype(void(std::declval<T &>().cache_backing))> : std::true_type {};

// Positive control for the detector above. If this ever reads false the
// detector is broken, and every "absent" result below would be meaningless.
struct forgeable_constraints_probe {
    bool cache_backing = false;
};

static_assert(has_cache_backing_member<forgeable_constraints_probe>::value,
              "detector is broken: it cannot see a cache_backing member that is definitely present, so its "
              "negative results below prove nothing");

static_assert(!has_cache_backing_member<alloc_constraints>::value,
              "alloc_constraints exposes a caller-writable cache_backing again: any caller could mint CACHE_BACKING");
static_assert(!has_cache_backing_member<alloc_intent>::value, "alloc_intent exposes a caller-writable cache_backing");
static_assert(!has_cache_backing_member<alloc_request>::value, "alloc_request exposes a caller-writable cache_backing");

// alloc_metadata KEEPS the field on purpose -- registry rows, diagnostics and
// legacy promotion consume it. Asserting it is still there keeps the three
// assertions above honest: they must be reporting a real difference between the
// request structs and the metadata row, not a detector that fails on everything.
static_assert(has_cache_backing_member<alloc_metadata>::value,
              "alloc_metadata lost cache_backing: the request-struct assertions above are no longer a contrast");

// The metadata row is the only writable carrier of the bit, so the remaining
// question is whether a caller can turn a metadata value back into an owner.
// It cannot: minting is private. (unified-cache.hpp asserts the same thing at
// its definition site; repeated here because the promotion case below depends
// on it -- see legacy_promotion_from_public_request_never_backs().)
static_assert(!std::is_constructible<alloc_handle, alloc_metadata>::value,
              "alloc_metadata can mint an alloc_handle: a caller could forge promotion provenance");

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Deterministic fake release backend: no physical memory is ever involved in
// the host-only cases, so the pointers below are identities, never dereferenced.
release_attempt release_noop(const alloc_metadata &, void * opaque) noexcept {
    if (opaque) {
        ++*static_cast<unsigned *>(opaque);
    }
    return { release_attempt_status::RELEASED };
}

alloc_metadata host_metadata(uint64_t id, int device) {
    alloc_metadata value;
    value.ptr        = reinterpret_cast<void *>(static_cast<uintptr_t>(0x5eed0000u + id * 0x100u));
    value.size       = 256;
    value.device     = device;
    value.id         = id;
    value.generation = 3;
    value.offset     = 0;
    value.extent     = 256;
    value.epoch_id   = 1;
    value.kind       = alloc_tier::HOST_PINNED;
    value.role       = alloc_role::STAGING;
    value.category   = runtime_category::STAGING;
    value.zone       = vram_zone_id::COUNT;
    value.host_zone  = host_zone_id::STAGING;
    return value;
}

// Ownership class of the control publishing `ptr`, read from the coordinator's
// observational snapshot. Returns nullopt when no control carries that pointer.
std::optional<allocation_control_class> class_of(const std::shared_ptr<allocation_release_coordinator> & coordinator,
                                                 const void *                                            ptr) {
    if (!coordinator) {
        return std::nullopt;
    }
    const auto controls = coordinator->snapshot_controls();
    const auto it       = std::find_if(controls.begin(), controls.end(),
                                       [&](const allocation_control_snapshot & c) { return c.metadata.ptr == ptr; });
    if (it == controls.end()) {
        return std::nullopt;
    }
    return it->ownership_class;
}

size_t count_class(const std::shared_ptr<allocation_release_coordinator> & coordinator,
                   allocation_control_class                                wanted) {
    if (!coordinator) {
        return 0;
    }
    const auto controls = coordinator->snapshot_controls();
    return static_cast<size_t>(
        std::count_if(controls.begin(), controls.end(),
                      [&](const allocation_control_snapshot & c) { return c.ownership_class == wanted; }));
}

// ---------------------------------------------------------------------------
// Case 5: two-device coordinator reset.
//
// Runs FIRST, on a pristine coordinator map -- see main(). Resetting device 0
// must not touch device 1, and device 0 must come back afterwards.
// ---------------------------------------------------------------------------
void two_device_coordinator_reset_isolates_the_other_device() {
    check(allocation_coordinator_test_count() == 0,
          "coordinators already existed before this case ran; it must run first, on a pristine map, or its "
          "detach assertions below would be reporting another subsystem's owners");

    auto first  = unified_allocation_release_coordinator(0);
    auto second = unified_allocation_release_coordinator(1);
    check(first && second, "per-device coordinator creation failed");
    check(first != second && first->device() == 0 && second->device() == 1, "the two devices share one coordinator");
    check(allocation_coordinator_test_count() == 2, "coordinator map did not record both devices");

    // These are REAL coordinators from the process-wide map, so they carry no
    // injected release backend and their release path goes to the runtime
    // registry. Publishing a matching intrusive row first is what keeps that
    // path off physical memory: the row carries test_no_physical_release, so
    // release resolves to RELEASED without ever freeing these synthetic
    // pointers. Without the row the release would find nothing, return
    // RETRY_SCHEDULED, and park the control on the retry list -- which would
    // then block the very detach this case is about, for the wrong reason.
    const alloc_metadata first_row  = host_metadata(1, 0);
    const alloc_metadata second_row = host_metadata(2, 1);
    check(allocation_registry_test_publish(first_row, true) && allocation_registry_test_publish(second_row, true),
          "per-device registry row publication failed");

    allocation_result on_first  = allocation_owner_test_create_on_coordinator(first_row, first);
    allocation_result on_second = allocation_owner_test_create_on_coordinator(second_row, second);
    check(on_first && on_second, "per-device owner creation failed");
    check(first->live_controls() == 1 && second->live_controls() == 1, "owners were not counted per device");
    const alloc_metadata second_before = on_second.owner.metadata();

    // Reset device 0 while its owner is live. Detach must REFUSE rather than
    // strand a live control, and the refusal leaves device 0 closed.
    check(!unified_allocation_release_coordinator_detach(0), "detach abandoned a live allocation control");
    check(!allocation_coordinator_test_try_register(first), "device 0 accepted a registration after its reset");

    // The whole point of the case: device 1 is untouched by all of the above.
    check(second->live_controls() == 1, "resetting device 0 disturbed device 1's live census");
    check(on_second.owner.metadata() == second_before, "resetting device 0 mutated device 1's owner identity");
    check(allocation_coordinator_test_try_register(second), "resetting device 0 closed device 1's admission");
    check(second->live_controls() == 1, "device 1's probe registration leaked a control");
    check(class_of(second, second_row.ptr).value_or(allocation_control_class::CACHE_BACKING) ==
              allocation_control_class::EXTERNAL_EXACT,
          "device 1's surviving owner lost or changed its ownership class");

    // Release device 0's owner, then the reset completes.
    check(on_first.owner.reset().released(), "device 0 owner release failed");
    check(first->live_controls() == 0, "device 0 owner release did not clear its census");
    // The row is gone, so the release really travelled the registry path rather
    // than being short-circuited somewhere that would leave detach unblocked for
    // an unrelated reason.
    check(!allocation_registry_test_contains(first_row.ptr), "device 0 release left its registry row behind");
    check(unified_allocation_release_coordinator_detach(0), "clean coordinator refused to detach after reset");
    check(allocation_coordinator_test_count() == 1, "detach did not remove device 0 from the coordinator map");

    // Retry on the reset device: a fresh coordinator, accepting work again. The
    // pointer comparison is what distinguishes a genuine replacement from the
    // old closed object being handed back.
    auto retried = unified_allocation_release_coordinator(0);
    check(retried && retried != first, "device 0 retry returned the closed coordinator instead of a fresh one");
    const alloc_metadata retry_row = host_metadata(3, 0);
    check(allocation_registry_test_publish(retry_row, true), "retry registry row publication failed");
    allocation_result after_reset = allocation_owner_test_create_on_coordinator(retry_row, retried);
    check(static_cast<bool>(after_reset), "device 0 could not allocate after its coordinator reset");
    check(retried->live_controls() == 1, "device 0 retry was not counted");
    check(second->live_controls() == 1, "device 0 retry disturbed device 1");

    check(after_reset.owner.reset().released(), "device 0 retry owner release failed");
    check(on_second.owner.reset().released(), "device 1 owner release failed");
    check(unified_allocation_release_coordinator_detach(0), "device 0 teardown failed");
    check(unified_allocation_release_coordinator_detach(1), "device 1 teardown failed");
    check(allocation_coordinator_test_count() == 0, "coordinator map was not restored for the cases that follow");
    // Leave no synthetic rows behind for the device cases to trip over.
    for (const alloc_metadata & row : { first_row, second_row, retry_row }) {
        allocation_registry_test_erase(row.ptr);
    }

    printf("PASS two-device-reset-refuses-live-control\n");
    printf("PASS two-device-reset-leaves-other-device-untouched\n");
    printf("PASS two-device-reset-retry-succeeds\n");
}

// ---------------------------------------------------------------------------
// Case 3: legacy-promotion forgery.
//
// promote_legacy_alloc_owner_with_coordinator() classifies from the metadata
// carried by the alloc_handle being promoted. A row created by a public request
// carries cache_backing=false -- unified-cache.cpp writes that field only from
// the thread_local g_allocating_cache_backing, which is set only beneath the
// private backing mint -- so promotion can never reach CACHE_BACKING.
// ---------------------------------------------------------------------------
void legacy_promotion_from_public_request_never_backs() {
    unsigned releases = 0;

    // (a) The forgery attempt: a row shaped exactly as a public request produces
    //     one. Nothing about it may promote to CACHE_BACKING.
    const alloc_metadata public_row = host_metadata(10, 0);
    check(!public_row.cache_backing, "fixture error: the public-request row already claims backing provenance");
    check(allocation_registry_test_publish(public_row, false), "legacy row publication failed");
    auto coordinator = std::make_shared<allocation_release_coordinator>(public_row.device, release_noop, &releases);
    allocation_result promoted = allocation_registry_test_promote(public_row, coordinator);
    check(static_cast<bool>(promoted), "public legacy row failed to promote at all");
    const auto promoted_class = class_of(coordinator, public_row.ptr);
    check(promoted_class.has_value(), "promoted owner published no control to classify");
    check(*promoted_class != allocation_control_class::CACHE_BACKING,
          "a legacy row from a public request promoted to CACHE_BACKING");
    check(*promoted_class == allocation_control_class::EXTERNAL_EXACT,
          "an unzoned public legacy row promoted to something other than EXTERNAL_EXACT");
    check(promoted.owner.reset().released(), "promoted owner release failed");
    allocation_registry_test_erase(public_row.ptr);

    // (b) SENSITIVITY CONTROL for (a). Without this, (a) would also pass if
    //     promotion had stopped consulting provenance at all and hardcoded
    //     EXTERNAL_EXACT -- a green assertion that had stopped testing anything.
    //     The mutation is asserted to have applied before it is relied on.
    alloc_metadata backing_row = host_metadata(11, 0);
    backing_row.cache_backing  = true;
    check(backing_row.cache_backing != host_metadata(11, 0).cache_backing,
          "control is void: the cache_backing mutation did not apply");
    check(allocation_registry_test_publish(backing_row, false), "control row publication failed");
    allocation_result backing = allocation_registry_test_promote(backing_row, coordinator);
    check(static_cast<bool>(backing), "control row failed to promote");
    check(class_of(coordinator, backing_row.ptr) == allocation_control_class::CACHE_BACKING,
          "control is void: promotion no longer produces CACHE_BACKING for a backing row, so case (a) above could "
          "not have failed and proves nothing");
    check(backing.owner.reset().released(), "control owner release failed");
    allocation_registry_test_erase(backing_row.ptr);

    // (b) is reachable ONLY through this private test seam, which mints an
    // alloc_handle from arbitrary metadata. A caller cannot do that: alloc_handle
    // is not constructible from alloc_metadata (static_assert above), exposes no
    // cache_backing reference among its public members, and every public route to
    // one goes through unified_alloc(), which writes the field from the
    // thread_local rather than from the request. So (b) is a sensitivity probe,
    // not a reachable forgery.

    // (c) The other non-backing branch, so (a) is not passing merely because
    //     every promotion happens to land on EXTERNAL_EXACT.
    alloc_metadata zoned_row = host_metadata(12, 0);
    zoned_row.zone_managed   = true;
    check(allocation_registry_test_publish(zoned_row, false), "zoned row publication failed");
    allocation_result zoned = allocation_registry_test_promote(zoned_row, coordinator);
    check(static_cast<bool>(zoned), "zoned row failed to promote");
    check(class_of(coordinator, zoned_row.ptr) == allocation_control_class::CACHE_SUBALLOCATION,
          "a zone-managed public row did not promote to CACHE_SUBALLOCATION");
    check(zoned.owner.reset().released(), "zoned owner release failed");
    allocation_registry_test_erase(zoned_row.ptr);

    check(coordinator->live_controls() == 0, "promotion cases leaked an allocation control");
    check(releases == 3, "promotion cases did not release exactly once each");
    printf("PASS legacy-promotion-public-row-never-cache-backing\n");
    printf("PASS legacy-promotion-classification-is-sensitive\n");
    printf("PASS legacy-promotion-zoned-row-is-suballocation\n");
}

// ---------------------------------------------------------------------------
// Case 4: pinned-pool positive control.
//
// If the Task-2 fix over-rotated, CACHE_BACKING would be unreachable even for
// the pinned pool's own chunks, and every "never CACHE_BACKING" assertion above
// would be passing for the wrong reason.
// ---------------------------------------------------------------------------
bool pinned_pool_backing_is_still_cache_backing(sycl::queue & q) {
    constexpr size_t mib   = 1024ull * 1024ull;
    unified_cache *  cache = get_unified_cache(q);
    if (!cache) {
        fail("unified cache unavailable for the pinned-pool positive control");
    }
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(2 * mib, 2 * mib, 4 * mib, 2 * mib);
    }
    check(cache->host_zones_configured(), "host zones unavailable for the pinned-pool positive control");

    alloc_request req{};
    req.queue                               = &q;
    req.device                              = 0;  // ONEAPI_DEVICE_SELECTOR remaps the selected card to runtime 0.
    req.size                                = 4096;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;

    auto coordinator = allocation_coordinator_test_lookup(req.device);
    check(coordinator != nullptr, "allocation coordinator unavailable");
    const size_t backing_before = count_class(coordinator, allocation_control_class::CACHE_BACKING);

    mem_handle   slice    = unified_allocate(req);
    resolved_ptr resolved = slice.resolve();
    check(slice.valid() && resolved.ptr != nullptr, "pinned-pool suballocation failed");
    check(!resolved.on_device, "pinned-pool suballocation resolved as device memory");

    const size_t backing_after = count_class(coordinator, allocation_control_class::CACHE_BACKING);
    printf("      pinned-pool CACHE_BACKING controls: %zu before, %zu after\n", backing_before, backing_after);
    check(backing_after >= 1,
          "no CACHE_BACKING control exists after a real pinned-pool allocation: the private mint has been broken "
          "and the forgery assertions elsewhere in this fixture would pass vacuously");

    // The slice itself is an interior suballocation, NOT backing. Backing is the
    // chunk underneath it, which the pool minted through its private token.
    check(class_of(coordinator, resolved.ptr) == allocation_control_class::CACHE_SUBALLOCATION,
          "a pinned-pool slice was classified as backing rather than as a suballocation");

    slice = {};
    printf("PASS pinned-pool-backing-is-cache-backing\n");
    printf("PASS pinned-pool-slice-is-suballocation\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 2: public-surface forgery.
//
// must_host_pinned + require_host_usm_base is the request shape the pinned pool
// itself uses for a backing chunk. Asked for through the PUBLIC entry point,
// with no token, it must come back EXTERNAL_EXACT -- and shutdown must not
// exempt it.
//
// Runs LAST on the device path: its final assertion is that shutdown REFUSES,
// and a refused shutdown leaves every coordinator closed (shutdown_unified_cache
// closes them all before the census), so no allocation afterwards would succeed.
// ---------------------------------------------------------------------------
bool public_standalone_host_usm_base_is_external_exact(sycl::queue & q) {
    uint64_t census_before[3]{};
    check(unified_cache_shutdown_allocation_control_census_for_test(census_before),
          "allocation-control census unavailable");

    alloc_request req{};
    req.queue                                    = &q;
    req.device                                   = 0;
    req.size                                     = 4096;
    req.intent.role                              = alloc_role::STAGING;
    req.intent.category                          = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned      = true;
    req.intent.constraints.require_host_usm_base = true;

    auto coordinator = allocation_coordinator_test_lookup(req.device);
    check(coordinator != nullptr, "allocation coordinator unavailable");
    const size_t backing_before = count_class(coordinator, allocation_control_class::CACHE_BACKING);

    allocation_result forged = unified_allocate_owner(req);
    check(static_cast<bool>(forged), "the public standalone host-USM request did not allocate at all");
    void * const forged_ptr = forged.owner.metadata().ptr;

    const auto forged_class = class_of(coordinator, forged_ptr);
    check(forged_class.has_value(), "the forged allocation published no control to classify");
    check(*forged_class != allocation_control_class::CACHE_BACKING,
          "a PUBLIC must_host_pinned + require_host_usm_base request minted CACHE_BACKING: the forgery hole is open");
    check(*forged_class == allocation_control_class::EXTERNAL_EXACT,
          "the forged allocation was not classified EXTERNAL_EXACT");
    check(count_class(coordinator, allocation_control_class::CACHE_BACKING) == backing_before,
          "the forged allocation added a CACHE_BACKING control somewhere other than its own");

    // Not exempted: it is counted as a live control like any external owner.
    uint64_t census_held[3]{};
    check(unified_cache_shutdown_allocation_control_census_for_test(census_held), "census unavailable while held");
    check(census_held[1] == census_before[1] + 1, "the forged allocation was not counted as a live control");

    // End-to-end: shutdown must REFUSE while it is held. Guard against passing
    // for a foreign reason -- the refusal is attributable to this owner only if
    // it is the process's one and only non-exempt live control. census[1] is the
    // sum over every coordinator, so comparing it with this coordinator's own
    // snapshot proves no other coordinator holds anything.
    // One snapshot for both figures. Deriving them from two separate snapshots
    // would compare counts taken at different instants, which is the shape of
    // the bug this whole fixture is about.
    const auto   controls = coordinator->snapshot_controls();
    const size_t non_exempt =
        static_cast<size_t>(std::count_if(controls.begin(), controls.end(), [](const allocation_control_snapshot & c) {
            return c.ownership_class != allocation_control_class::CACHE_BACKING;
        }));
    if (census_held[1] == controls.size() && non_exempt == 1) {
        check(!shutdown_unified_cache(),
              "shutdown exempted a live EXTERNAL_EXACT owner from destructive-teardown refusal");
        printf("PASS public-forgery-not-exempt-from-shutdown-refusal\n");
    } else {
        // Never claim a pass here. A refusal observed with other non-exempt
        // controls live would not be evidence about this owner.
        printf(
            "NOTE public-forgery-shutdown-refusal NOT EXERCISED: %zu non-exempt controls on this coordinator, "
            "%llu live process-wide -- refusal would not be attributable to the forged owner\n",
            non_exempt, static_cast<unsigned long long>(census_held[1]));
    }

    check(forged.owner.reset().released(), "forged owner release failed");
    printf("PASS public-standalone-host-usm-base-is-external-exact\n");
    printf("PASS public-forgery-counted-live-not-exempt\n");
    return true;
}

}  // namespace

int main() {
    // Case results go to stdout and the banner/skip diagnostics to stderr. With
    // stdout block-buffered the two interleave wrongly: the skip line lands
    // BEFORE the PASS lines of cases that already ran, so a run that verified
    // six things reads as a run that verified nothing. Unbuffer stdout so the
    // transcript matches the order of execution.
    setvbuf(stdout, nullptr, _IONBF, 0);

    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "Cache-Backing Provenance (adversarial)\n");
    fprintf(stderr, "===========================================\n");

    // Bound the pinned pool so the positive control cannot claim a large chunk.
    if (std::getenv("GGML_SYCL_PINNED_CHUNK_MB") == nullptr) {
        setenv("GGML_SYCL_PINNED_CHUNK_MB", "16", 0);
    }

    // Case 5 runs FIRST and on purpose: it asserts an empty coordinator map, and
    // detaching a device's coordinator is only meaningful before the cache has
    // created controls of its own. Running it after the device cases would make
    // its detach assertions report the cache's pinned chunks instead.
    two_device_coordinator_reset_isolates_the_other_device();
    legacy_promotion_from_public_request_never_backs();

    std::optional<sycl::device> dev_opt = sycl_test_require_gpu(
        "the public-surface forgery case and the pinned-pool positive control (the host-only cases above DID run "
        "and passed)");
    if (!dev_opt) {
        return SYCL_TEST_SKIP;
    }
    sycl::queue q(*dev_opt, sycl::property::queue::in_order{});

    // Positive control before the forgery case: the forgery case ends by proving
    // shutdown refuses, which closes every coordinator for the rest of the run.
    bool ok = pinned_pool_backing_is_still_cache_backing(q);
    ok &= public_standalone_host_usm_base_is_external_exact(q);

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "%s\n", ok ? "All cases passed." : "FAILED");
    return ok ? 0 : 1;
}
