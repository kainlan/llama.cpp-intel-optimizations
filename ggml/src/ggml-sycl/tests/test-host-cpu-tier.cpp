// The HOST_CPU tier of the unified cache through the real owner-first front door (llama.cpp-th32 T1b), in the
// private-fixture build of the backend.
//
// Host-only: the tier needs no device, queue or cache. unified_allocate_owner mints the intrusive control for device
// 0's release coordinator, the allocation is one locked anonymous mapping, the registries hold one record for it, and
// the release is exactly one munmap through neither ::free nor sycl::free. A test that cannot get its mapping locked
// (a finite RLIMIT_MEMLOCK below the request) exits 77, which ctest reports as skipped: it proved nothing.

#include "ggml.h"
#include "host-cpu-mapping.hpp"
#include "unified-cache.hpp"

#include <sys/resource.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace ggml_sycl;

namespace {

[[noreturn]] void fail(const char * message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

// VmLck of this process in kB: what mlock has pinned. 0 when the line is missing.
uint64_t vm_locked_kb() {
    std::ifstream in("/proc/self/status");
    std::string   line;
    while (std::getline(in, line)) {
        if (line.rfind("VmLck:", 0) == 0) {
            return std::strtoull(line.c_str() + 6, nullptr, 10);
        }
    }
    return 0;
}

alloc_request host_cpu_request(size_t bytes, alloc_role role) {
    alloc_request req;
    req.device                           = 0;
    req.size                             = bytes;
    req.intent.role                      = role;
    req.intent.constraints.must_host_cpu = true;
    return req;
}

// A refusal because the lock limit is below the request is the environment's, not the tier's.
void skip_if_unlockable(const allocation_result & allocation, size_t bytes) {
    if (allocation) {
        return;
    }
    rlimit lim{};
    if (getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY && lim.rlim_cur < bytes) {
        std::cout << "SKIP: RLIMIT_MEMLOCK is " << lim.rlim_cur << " bytes, below the request\n";
        std::exit(77);
    }
}

uint64_t runtime_census_all() {
    uint64_t out[8] = {};
    check(unified_cache_shutdown_runtime_alloc_census_for_test(out), "the shutdown census is unavailable");
    return out[0];
}

uint64_t runtime_census_host_pinned() {
    uint64_t out[8] = {};
    check(unified_cache_shutdown_runtime_alloc_census_for_test(out), "the shutdown census is unavailable");
    return out[1];
}

void tier_identity_registry_census_and_release() {
    const size_t   size   = 3u * 1024u * 1024u + 1u;  // two huge pages and a byte: the mapping rounds up to 4 MiB
    const size_t   mapped = host_cpu_mapped_size(size);
    const uint64_t locked_before   = vm_locked_kb();
    const size_t   rows_before     = allocation_registry_test_size();
    const uint64_t census_before   = runtime_census_all();
    const uint64_t pinned_before   = runtime_census_host_pinned();
    const uint64_t stats_before    = offload_stats_get().alloc_count_host_cpu;
    const uint64_t munmaps_before  = unified_cache_host_cpu_munmap_count_for_test();
    size_t         host_before     = 0;
    size_t         file_before     = 0;
    size_t         host_cpu_before = 0;
    unified_alloc_registry_sums_for_test(&host_before, &file_before);
    unified_alloc_registry_host_cpu_sum_for_test(&host_cpu_before);

    uint64_t free_before = 0;
    uint64_t sycl_before = 0;
    unified_cache_test_free_branch_counts(&free_before, &sycl_before);

    auto coordinator = unified_allocation_release_coordinator(0);
    check(coordinator != nullptr, "no release coordinator for device 0");
    check(coordinator->live_controls() == 0, "a control was live before the test");

    allocation_result allocation = unified_allocate_owner(host_cpu_request(size, alloc_role::LAZY_ROW_CACHE));
    skip_if_unlockable(allocation, mapped);
    check(static_cast<bool>(allocation), "the front door refused a valid HOST_CPU request");
    check(coordinator->live_controls() == 1, "the owner was not minted as exactly one control");

    const alloc_metadata meta = allocation.owner.metadata();
    check(meta.tier == alloc_tier::HOST_CPU, "the owner's tier is not HOST_CPU");
    check(meta.size == size, "the record is not the requested size");
    check(meta.device == 0, "the record names the wrong device");
    check(meta.role == alloc_role::LAZY_ROW_CACHE, "the record lost its role");
    void * const base = meta.ptr;
    check(base != nullptr, "no pointer");

    // exact, independently released, never cache backing: the CACHE_BACKING allowlist is untouched
    const auto snapshots = coordinator->snapshot_controls();
    check(snapshots.size() == 1 && snapshots[0].ownership_class == allocation_control_class::EXTERNAL_EXACT,
          "the owner is not EXTERNAL_EXACT");

    // the registry holds exactly one record for it, and the shutdown census counts it, as a HOST_CPU and not a
    // HOST_PINNED allocation
    check(allocation_registry_test_contains(base), "no registry record");
    check(allocation_registry_test_size() == rows_before + 1, "the allocation is not exactly one record");
    check(runtime_census_all() == census_before + 1, "the shutdown census does not count the allocation");
    check(runtime_census_host_pinned() == pinned_before, "the shutdown census counts a HOST_CPU allocation as pinned");
    const auto * info = alloc_registry::instance().lookup(base);
    check(info != nullptr && info->type == alloc_type::HOST_CPU, "alloc_registry does not type it HOST_CPU");
    check(alloc_registry::instance().get_alloc_type((char *) base + size - 1) == alloc_type::HOST_CPU,
          "an interior pointer is not typed HOST_CPU");
    check(offload_stats_get().alloc_count_host_cpu == stats_before + 1, "the allocation was not counted as HOST_CPU");

    // the registry reports the bytes apart from the host-pinned figure the validator compares with the host ledger
    size_t host_during     = 0;
    size_t file_during     = 0;
    size_t host_cpu_during = 0;
    unified_alloc_registry_sums_for_test(&host_during, &file_during);
    unified_alloc_registry_host_cpu_sum_for_test(&host_cpu_during);
    check(host_during == host_before, "a HOST_CPU allocation was charged to the host-pinned sum");
    check(host_cpu_during == host_cpu_before + size, "the HOST_CPU sum is not the allocation's size");

    // query_location: HOST_CPU, read by the CPU, not device-mappable
    const memory_location at_base     = query_location(base);
    const memory_location at_interior = query_location((char *) base + size / 2);
    check(at_base.tier == alloc_tier::HOST_CPU && at_interior.tier == alloc_tier::HOST_CPU,
          "query_location does not answer HOST_CPU");
    check(at_base.host_accessible() && !at_base.device_mappable() && !at_base.on_device(),
          "a HOST_CPU pointer must be CPU-read and not device-mappable");

    // locked in full before the front door returned, and writable end to end
    check(vm_locked_kb() >= locked_before + mapped / 1024, "the mapping is not mlocked in full");
    std::memset(base, 0xA5, size);
    check(((const unsigned char *) base)[0] == 0xA5 && ((const unsigned char *) base)[size - 1] == 0xA5,
          "the mapping is not writable");

    // a device may not be handed it: resolve() is the CPU's, resolve(device) is empty, slices too
    mem_handle handle = mem_handle::from_owned_alloc(std::move(allocation.owner));
    check(handle.valid(), "no handle for the owner");
    check(handle.resolve().ptr == base, "resolve() does not give the CPU the allocation");
    check(handle.resolve(0).ptr == nullptr, "resolve(device) handed a device a HOST_CPU allocation");
    mem_handle slice = handle.slice(4096, 4096);
    check(slice.valid() && slice.resolve().ptr == (char *) base + 4096, "the slice does not resolve for the CPU");
    check(slice.resolve(0).ptr == nullptr, "a slice handed a device a HOST_CPU allocation");

    // release: the last reference unmaps the whole mapping, in one munmap, through neither ::free nor sycl::free
    handle = {};
    check(coordinator->live_controls() == 1, "the slice dropped the owner early");
    slice = {};
    check(coordinator->live_controls() == 0, "the last reference did not release the control");
    check(!allocation_registry_test_contains(base), "the record survived the release");
    check(alloc_registry::instance().lookup(base) == nullptr, "alloc_registry still knows the released mapping");
    check(runtime_census_all() == census_before, "the shutdown census did not return");
    check(unified_cache_host_cpu_munmap_count_for_test() == munmaps_before + 1,
          "the release was not exactly one munmap of the whole mapping");
    check(vm_locked_kb() <= locked_before, "the lock outlived the release");

    uint64_t free_after = 0;
    uint64_t sycl_after = 0;
    unified_cache_test_free_branch_counts(&free_after, &sycl_after);
    check(free_after == free_before, "the release reached the ::free branch");
    check(sycl_after == sycl_before, "the release reached the sycl::free fallthrough");
    check(allocation_registry_test_size() == rows_before, "the registry did not return to its size");
    size_t host_after     = 0;
    size_t file_after     = 0;
    size_t host_cpu_after = 0;
    unified_alloc_registry_sums_for_test(&host_after, &file_after);
    unified_alloc_registry_host_cpu_sum_for_test(&host_cpu_after);
    check(host_after == host_before && host_cpu_after == host_cpu_before, "the sums did not return after the release");
    std::cout << "PASS tier-identity-registry-census-and-release\n";
}

// A role that would make the request look like a cache suballocation does not change the owner class, and the
// lazy roles are carried through to the record.
void owner_class_does_not_follow_the_role() {
    const size_t size        = 64u * 1024u;
    auto         coordinator = unified_allocation_release_coordinator(0);
    check(coordinator != nullptr, "no release coordinator for device 0");
    for (alloc_role role : { alloc_role::KV, alloc_role::LAZY_STAGING, alloc_role::LAZY_OUTPUT }) {
        allocation_result allocation = unified_allocate_owner(host_cpu_request(size, role));
        skip_if_unlockable(allocation, host_cpu_mapped_size(size));
        check(static_cast<bool>(allocation), "the front door refused a valid HOST_CPU request");
        check(allocation.owner.metadata().tier == alloc_tier::HOST_CPU, "the tier is not HOST_CPU");
        check(allocation.owner.metadata().role == role, "the role was lost");
        const auto snapshots = coordinator->snapshot_controls();
        check(snapshots.size() == 1 && snapshots[0].ownership_class == allocation_control_class::EXTERNAL_EXACT,
              "a HOST_CPU owner is not EXTERNAL_EXACT");
    }
    check(coordinator->live_controls() == 0, "an owner outlived its scope");
    std::cout << "PASS owner-class-does-not-follow-the-role\n";
}

// must_host_cpu selects the tier and nothing else: the entry refuses it beside a routing constraint, and a refusal
// leaves no control, record or mapping behind.
void refusals_leave_nothing_behind() {
    const size_t   size           = 64u * 1024u;
    const size_t   rows_before    = allocation_registry_test_size();
    const uint64_t munmaps_before = unified_cache_host_cpu_munmap_count_for_test();
    const uint64_t stats_before   = offload_stats_get().alloc_count_host_cpu;
    auto           coordinator    = unified_allocation_release_coordinator(0);

    alloc_request with_device                  = host_cpu_request(size, alloc_role::LAZY_ROW_CACHE);
    with_device.intent.constraints.must_device = true;
    allocation_result refused                  = unified_allocate_owner(with_device);
    check(!refused && !refused.owner, "must_host_cpu with must_device was accepted");
    check(refused.error == allocation_error::INVALID_REQUEST, "that refusal is not INVALID_REQUEST");

    alloc_request with_pinned                       = host_cpu_request(size, alloc_role::LAZY_ROW_CACHE);
    with_pinned.intent.constraints.must_host_pinned = true;
    refused                                         = unified_allocate_owner(with_pinned);
    check(!refused && !refused.owner, "must_host_cpu with must_host_pinned was accepted");
    check(refused.error == allocation_error::INVALID_REQUEST, "that refusal is not INVALID_REQUEST");

    alloc_request no_device = host_cpu_request(size, alloc_role::LAZY_ROW_CACHE);
    no_device.device        = -1;
    check(!unified_allocate_owner(no_device), "a HOST_CPU request with no device was accepted");

    check(!unified_allocate_owner(host_cpu_request(0, alloc_role::LAZY_ROW_CACHE)), "a zero-byte request was accepted");

    check(coordinator->live_controls() == 0, "a refusal left a live control");
    check(allocation_registry_test_size() == rows_before, "a refusal left a registry record");
    check(unified_cache_host_cpu_munmap_count_for_test() == munmaps_before, "a refusal mapped something");
    check(offload_stats_get().alloc_count_host_cpu == stats_before, "a refusal was counted as an allocation");
    std::cout << "PASS refusals-leave-nothing-behind\n";
}

// unified_select_tier maps must_host_cpu to HOST_CPU, and nothing else does: no constraint, size or cohort falls
// into it. A pure function of the request (device -1 never touches a cache), so it runs before any allocation.
void tier_selection_maps_must_host_cpu() {
    alloc_request plain;
    plain.size = 1u << 20;
    check(unified_select_tier(plain) == alloc_tier::DEVICE_VRAM, "an unconstrained request left DEVICE_VRAM");

    alloc_request explicit_cpu                    = plain;
    explicit_cpu.intent.constraints.must_host_cpu = true;
    check(unified_select_tier(explicit_cpu) == alloc_tier::HOST_CPU, "must_host_cpu does not select HOST_CPU");
    std::cout << "PASS tier-selection-maps-must-host-cpu\n";
}

// A cohort cannot inherit HOST_CPU: unified_select_tier's cohort branch skips an entry that names it, so a
// prefer_same_tier_as_cohort request without must_host_cpu falls through to normal selection.
void a_cohort_cannot_inherit_host_cpu() {
    alloc_request plain;
    plain.size = 1u << 20;

    const size_t  size           = 64u * 1024u;
    alloc_request first          = host_cpu_request(size, alloc_role::LAZY_STAGING);
    first.intent.cohort_id       = "th32-t1b-cohort";
    allocation_result allocation = unified_allocate_owner(first);
    skip_if_unlockable(allocation, host_cpu_mapped_size(size));
    check(static_cast<bool>(allocation), "the front door refused a valid HOST_CPU request");

    alloc_request follower                                 = plain;
    follower.intent.cohort_id                              = "th32-t1b-cohort";
    follower.intent.constraints.prefer_same_tier_as_cohort = true;
    check(unified_select_tier(follower) != alloc_tier::HOST_CPU, "a cohort inherited HOST_CPU");
    check(unified_select_tier(follower) == alloc_tier::DEVICE_VRAM, "the follower did not fall through");

    follower.intent.constraints.must_host_cpu = true;
    check(unified_select_tier(follower) == alloc_tier::HOST_CPU, "an explicit request lost to the cohort branch");
    std::cout << "PASS a-cohort-cannot-inherit-host-cpu\n";
}

// The role-bytes reading (llama.cpp-th32 T1b): the live bytes the registry holds under a role on a device, read through
// the C++ accessor and through the exported wrapper that takes the role as an int. VOID (false, *out zero) when the
// device has no cache, the role is out of range or an argument is invalid.
void role_bytes_reading() {
    uint64_t out = 123;
    check(!unified_cache_role_bytes_for_testing(-1, alloc_role::LAZY_ROW_CACHE, &out) && out == 0,
          "a negative device gave a reading");
    out = 123;
    check(!unified_cache_role_bytes_for_testing(0, alloc_role::LAZY_ROW_CACHE, nullptr), "a null out gave a reading");
    out = 123;
    check(!ggml_backend_sycl_role_bytes_for_testing(0, -1, &out) && out == 0, "a negative role gave a reading");
    out = 123;
    check(!ggml_backend_sycl_role_bytes_for_testing(0, (int) alloc_role::LAZY_OUTPUT + 1, &out) && out == 0,
          "an out-of-range role gave a reading");

    // no cache has been created in this process, so a reading for a valid device and role is VOID
    uint64_t owners[4] = {};
    check(unified_cache_shutdown_owner_census_for_test(owners), "the owner census is unavailable");
    check(owners[0] == 0, "a device cache exists before the VOID arm: that arm would prove nothing");
    out = 123;
    check(!unified_cache_role_bytes_for_testing(0, alloc_role::LAZY_ROW_CACHE, &out) && out == 0,
          "a device with no cache gave a reading");
    out = 123;
    check(!ggml_backend_sycl_role_bytes_for_testing(0, (int) alloc_role::LAZY_ROW_CACHE, &out) && out == 0,
          "the export gave a reading for a device with no cache");

    // the export is in the backend's proc-address table, which is how a test that links libllama reaches it
    void * entry =
        ggml_backend_reg_get_proc_address(ggml_backend_sycl_reg(), "ggml_backend_sycl_role_bytes_for_testing");
    check(entry == (void *) ggml_backend_sycl_role_bytes_for_testing, "the proc-address table has no such entry");
    std::cout << "PASS role-bytes-reading\n";
}

// The positive reading. It needs a device cache, which a host with no SYCL device cannot create, so ctest does not
// run it: run the binary with --device-cache under a device selector (for example ONEAPI_DEVICE_SELECTOR=level_zero:1).
// HOST_CPU allocations in two roles read back, through the C++ accessor and the exported wrapper, as exactly their
// own sizes, a third role reads zero, and the readings return to where they were when the allocations are released.
void role_bytes_with_a_device_cache() {
    check(get_unified_cache_for_device(0) != nullptr, "no device cache for device 0");
    const size_t row_bytes     = 64u * 1024u;
    const size_t staging_bytes = 192u * 1024u;

    auto read = [](alloc_role role) {
        uint64_t cxx = 1;
        uint64_t c   = 2;
        check(unified_cache_role_bytes_for_testing(0, role, &cxx), "the reading is VOID with a device cache");
        check(ggml_backend_sycl_role_bytes_for_testing(0, (int) role, &c), "the export is VOID with a device cache");
        check(cxx == c, "the export and the C++ accessor disagree");
        return cxx;
    };
    const uint64_t row_before     = read(alloc_role::LAZY_ROW_CACHE);
    const uint64_t staging_before = read(alloc_role::LAZY_STAGING);
    const uint64_t output_before  = read(alloc_role::LAZY_OUTPUT);

    allocation_result row     = unified_allocate_owner(host_cpu_request(row_bytes, alloc_role::LAZY_ROW_CACHE));
    allocation_result staging = unified_allocate_owner(host_cpu_request(staging_bytes, alloc_role::LAZY_STAGING));
    skip_if_unlockable(row, host_cpu_mapped_size(row_bytes));
    check(static_cast<bool>(row) && static_cast<bool>(staging), "the front door refused a valid HOST_CPU request");
    check(read(alloc_role::LAZY_ROW_CACHE) == row_before + row_bytes,
          "the row-cache role is not the allocation's size");
    check(read(alloc_role::LAZY_STAGING) == staging_before + staging_bytes, "the staging role is not its size");
    check(read(alloc_role::LAZY_OUTPUT) == output_before, "a role nothing allocated under moved");

    row = {};
    check(read(alloc_role::LAZY_ROW_CACHE) == row_before, "the row-cache reading did not return after the release");
    check(read(alloc_role::LAZY_STAGING) == staging_before + staging_bytes, "the release moved another role");
    staging = {};
    check(read(alloc_role::LAZY_STAGING) == staging_before, "the staging reading did not return after the release");
    std::cout << "PASS role-bytes-with-a-device-cache\n";
}

void device_mappable_by_tier() {
    memory_location loc{};
    loc.ptr  = (void *) 0x1000;
    loc.tier = alloc_tier::HOST_CPU;
    check(!loc.device_mappable() && loc.host_accessible() && !loc.on_device(),
          "HOST_CPU is host-accessible and not device-mappable");
    loc.tier = alloc_tier::HOST_PINNED;
    check(loc.device_mappable() && loc.host_accessible(), "HOST_PINNED is device-mappable and host-accessible");
    loc.tier = alloc_tier::DEVICE_VRAM;
    check(loc.device_mappable() && !loc.host_accessible(), "DEVICE_VRAM is device-mappable and not host-accessible");
    std::cout << "PASS device-mappable-by-tier\n";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--device-cache") == 0) {
        role_bytes_with_a_device_cache();
        return 0;
    }
    tier_selection_maps_must_host_cpu();
    tier_identity_registry_census_and_release();
    owner_class_does_not_follow_the_role();
    refusals_leave_nothing_behind();
    a_cohort_cannot_inherit_host_cpu();
    device_mappable_by_tier();
    role_bytes_reading();
    return 0;
}
