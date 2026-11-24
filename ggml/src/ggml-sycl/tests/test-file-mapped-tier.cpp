// The FILE_MAPPED tier of the unified cache (llama.cpp-5efe C2), through the private-fixture build of the backend.
//
// Host-only: a file mapping needs no device, queue or cache. The owner-first front door mints the intrusive control
// for device 0's release coordinator, the registry holds one record per reservation, and the release is one munmap of
// the whole reservation. The private counters on unified_free_record's `::free` and `sycl::free` branches must both
// read 0 at a correct release: a FILE_MAPPED record that reached either branch was freed as something it is not.

#include "ggml.h"
#include "unified-cache.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

unsigned char pattern(uint64_t i) {
    return (unsigned char) (i * 131 + 7);
}

struct fixture_file {
    std::string path;
    int         fd   = -1;
    uint64_t    size = 0;

    explicit fixture_file(uint64_t pages) {
        char      tmpl[] = "/tmp/file-mapped-tier-XXXXXX";
        const int tmp    = mkstemp(tmpl);
        check(tmp >= 0, "mkstemp");
        path                = tmpl;
        const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
        size                = pages * page;
        std::vector<unsigned char> bytes(size);
        for (uint64_t i = 0; i < size; ++i) {
            bytes[i] = pattern(i);
        }
        check(write(tmp, bytes.data(), size) == (ssize_t) size, "write");
        close(tmp);
        fd = open(path.c_str(), O_RDONLY);
        check(fd >= 0, "open");
    }

    ~fixture_file() {
        if (fd >= 0) {
            close(fd);
        }
        unlink(path.c_str());
    }
};

int open_fd_count() {
    int n = 0;
    for (int fd = 0; fd < 1024; ++fd) {
        n += fcntl(fd, F_GETFD) != -1 ? 1 : 0;
    }
    return n;
}

// Is any VMA of this process mapping the file `path`?
bool maps_file(const std::string & path) {
    std::ifstream in("/proc/self/maps");
    std::string   line;
    while (std::getline(in, line)) {
        if (line.find(path) != std::string::npos) {
            return true;
        }
    }
    return false;
}

file_mapping_request make_request(const fixture_file & f, uint64_t page) {
    file_mapping_request req;
    req.device      = 0;
    req.fd          = f.fd;
    req.file_size   = f.size;
    req.reservation = { 1 * page, 6 * page };
    req.segments    = {
        { 1 * page, 1 * page },
        { 4 * page, 2 * page }
    };
    return req;
}

void tier_identity_registry_and_release() {
    const uint64_t    page = (uint64_t) sysconf(_SC_PAGESIZE);
    fixture_file      f(8);
    const std::string path        = f.path;
    const int         fds_before  = open_fd_count();
    const size_t      rows_before = allocation_registry_test_size();

    uint64_t free_before = 0;
    uint64_t sycl_before = 0;
    unified_cache_test_free_branch_counts(&free_before, &sycl_before);
    const uint64_t munmaps_before = unified_cache_file_mapping_munmap_count_for_test();

    auto coordinator = unified_allocation_release_coordinator(0);
    check(coordinator != nullptr, "no release coordinator for device 0");
    check(coordinator->live_controls() == 0, "a control was live before the test");

    allocation_result allocation = unified_allocate_file_mapping(make_request(f, page));
    check(static_cast<bool>(allocation), "the front door refused a valid request");
    check(coordinator->live_controls() == 1, "the owner was not minted as exactly one control");

    const alloc_metadata meta = allocation.owner.metadata();
    check(meta.tier == alloc_tier::FILE_MAPPED, "the owner's tier is not FILE_MAPPED");
    check(meta.size == 6 * page, "the record is not the whole reservation");
    check(meta.device == 0, "the record names the wrong device");
    void * const base = meta.ptr;

    // the owner class is EXTERNAL_EXACT: an exact, independently released allocation, never cache backing
    const auto snapshots = coordinator->snapshot_controls();
    check(snapshots.size() == 1 && snapshots[0].ownership_class == allocation_control_class::EXTERNAL_EXACT,
          "the owner is not EXTERNAL_EXACT");

    // one registry record for the reservation, none for its segments
    check(allocation_registry_test_contains(base), "no registry record for the reservation");
    check(!allocation_registry_test_contains((char *) base + page), "a segment has its own registry record");
    check(allocation_registry_test_size() == rows_before + 1, "the reservation is not exactly one record");

    // the pointer-type registry knows the reservation, and its interior, as FILE_MAPPED
    const auto * info = alloc_registry::instance().lookup(base);
    check(info != nullptr && info->type == alloc_type::FILE_MAPPED, "alloc_registry does not type it FILE_MAPPED");
    check(alloc_registry::instance().get_alloc_type((char *) base + 3 * page) == alloc_type::FILE_MAPPED,
          "an interior pointer is not typed FILE_MAPPED");

    // query_location answers FILE_MAPPED, never UNREGISTERED, for the base and for an interior pointer
    const memory_location at_base     = query_location(base);
    const memory_location at_interior = query_location((char *) base + 3 * page + 17);
    check(at_base.tier == alloc_tier::FILE_MAPPED, "query_location(base) is not FILE_MAPPED");
    check(at_interior.tier == alloc_tier::FILE_MAPPED, "query_location(interior) is not FILE_MAPPED");
    check(at_base.host_accessible() && !at_base.device_mappable(),
          "a file mapping must be CPU-read and not device-mappable");

    // the bytes are the file's
    check(((const unsigned char *) base)[0] == pattern(1 * page) &&
              ((const unsigned char *) base)[3 * page + 5] == pattern(4 * page + 5),
          "the mapped bytes are not the file's");

    // a device may not be handed the reservation: resolve() is the CPU's, resolve(device) is empty, slices too
    mem_handle handle = mem_handle::from_owned_alloc(std::move(allocation.owner));
    check(handle.valid(), "no handle for the owner");
    check(handle.resolve().ptr == base, "resolve() does not give the CPU the reservation");
    check(handle.resolve(0).ptr == nullptr, "resolve(device) handed a device a file mapping");
    mem_handle slice = handle.slice(page, page);
    check(slice.valid() && slice.resolve().ptr == (char *) base + page, "the slice does not resolve for the CPU");
    check(slice.resolve(0).ptr == nullptr, "a slice handed a device a file mapping");

    // release: the last reference unmaps the whole reservation, in one munmap, through neither ::free nor sycl::free
    handle = {};
    check(coordinator->live_controls() == 1, "the slice dropped the owner early");
    slice = {};
    check(coordinator->live_controls() == 0, "the last reference did not release the control");
    check(!allocation_registry_test_contains(base), "the record survived the release");
    check(alloc_registry::instance().lookup(base) == nullptr, "alloc_registry still knows the released reservation");
    check(!maps_file(path), "a file segment is still mapped after the release");
    check(unified_cache_file_mapping_munmap_count_for_test() == munmaps_before + 1,
          "the release was not exactly one munmap of the whole reservation");

    uint64_t free_after = 0;
    uint64_t sycl_after = 0;
    unified_cache_test_free_branch_counts(&free_after, &sycl_after);
    check(free_after == free_before, "the release reached the ::free branch");
    check(sycl_after == sycl_before, "the release reached the sycl::free fallthrough");

    check(allocation_registry_test_size() == rows_before, "the registry did not return to its size");
    check(open_fd_count() == fds_before, "an fd leaked");
    std::cout << "PASS tier-identity-registry-and-release\n";
}

void refusals_leave_nothing_behind() {
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    fixture_file   f(8);
    const int      fds_before  = open_fd_count();
    const size_t   rows_before = allocation_registry_test_size();
    auto           coordinator = unified_allocation_release_coordinator(0);

    file_mapping_request wrong_size = make_request(f, page);
    wrong_size.file_size            = f.size + 1;
    allocation_result refused       = unified_allocate_file_mapping(wrong_size);
    check(!refused && !refused.owner, "a file of the wrong size was accepted");
    check(refused.error != allocation_error::NONE, "a refusal carries no error");

    file_mapping_request bad_device = make_request(f, page);
    bad_device.device               = -1;
    check(!unified_allocate_file_mapping(bad_device), "a request with no device was accepted");

    file_mapping_request no_segments = make_request(f, page);
    no_segments.segments.clear();
    check(!unified_allocate_file_mapping(no_segments), "a request with no segments was accepted");

    check(coordinator->live_controls() == 0, "a refusal left a live control");
    check(allocation_registry_test_size() == rows_before, "a refusal left a registry record");
    check(open_fd_count() == fds_before, "a refusal leaked an fd");
    std::cout << "PASS refusals-leave-nothing-behind\n";
}

// A file mapping is page cache, not host memory the cache allocated: it is never charged to the host ledger
// (unified_alloc_validate_registry compares that sum with g_runtime_reserved_host_bytes), and the registry reports its
// bytes separately. The two sums come from the one helper the validator reads.
void host_ledger_reports_file_mapped_apart() {
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    fixture_file   f(8);

    size_t host_before = 0;
    size_t file_before = 0;
    unified_alloc_registry_sums_for_test(&host_before, &file_before);

    allocation_result allocation = unified_allocate_file_mapping(make_request(f, page));
    check(static_cast<bool>(allocation), "the front door refused a valid request");

    size_t host_during = 0;
    size_t file_during = 0;
    unified_alloc_registry_sums_for_test(&host_during, &file_during);
    check(host_during == host_before, "a file mapping was charged to the host-pinned sum");
    check(file_during == file_before + 6 * page, "the file-mapped sum is not the reservation's length");

    allocation        = {};
    size_t host_after = 0;
    size_t file_after = 0;
    unified_alloc_registry_sums_for_test(&host_after, &file_after);
    check(host_after == host_before && file_after == file_before, "the sums did not return after the release");
    std::cout << "PASS host-ledger-reports-file-mapped-apart\n";
}

void device_mappable_by_tier() {
    memory_location loc{};
    loc.ptr  = (void *) 0x1000;
    loc.tier = alloc_tier::DEVICE_VRAM;
    check(loc.device_mappable() && !loc.host_accessible(), "DEVICE_VRAM is device-mappable and not host-accessible");
    loc.tier = alloc_tier::HOST_PINNED;
    check(loc.device_mappable() && loc.host_accessible(), "HOST_PINNED is device-mappable and host-accessible");
    loc.tier = alloc_tier::FILE_MAPPED;
    check(!loc.device_mappable() && loc.host_accessible(), "FILE_MAPPED is host-accessible and not device-mappable");
    loc.tier = alloc_tier::UNREGISTERED;
    check(!loc.device_mappable(), "an UNREGISTERED pointer is not device-mappable");
    std::cout << "PASS device-mappable-by-tier\n";
}

}  // namespace

int main() {
    tier_identity_registry_and_release();
    refusals_leave_nothing_behind();
    host_ledger_reports_file_mapped_apart();
    device_mappable_by_tier();
    return 0;
}
