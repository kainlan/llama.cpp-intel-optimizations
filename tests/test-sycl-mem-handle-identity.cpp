// Host test for mem_handle_identity and the tenant cohort tag (zhcn-design §3.1.1,
// lead ruling §B.2).
//
// A cache that only COMPARES a source must not hold it.  mem_handle_identity is the
// plain value such a cache keeps instead of a mem_handle: built only from the
// allocator's monotonic id (never an address), valid iff the id is non-zero, and
// holding no control.  This test pins that contract on real owner controls:
//
//  * an owner-backed handle yields a valid identity naming exactly its slice;
//  * a slice, a slice of a slice and a copy name what they should;
//  * a handle with no allocator id (an ownerless DIRECT view) yields NO identity,
//    and an invalid identity equals nothing, itself included;
//  * the §B.2 same-address case: an allocation freed and a new one minted at the
//    SAME address misses, because the id differs;
//  * holding an identity does not keep the allocation alive;
//  * the tenant cohort tag lives on the shared control and reads the same through
//    every copy and slice, and is null for an untagged or ownerless handle;
//  * the tag is set once: a second set aborts (a forked child), and a null cohort
//    is ignored.
//
// Nothing here touches a GPU: owners come from the private fixture factory, with an
// injected release backend, and the registration pins the selector to the CPU.
//
// Usage:
//   ./build/bin/test-sycl-mem-handle-identity

#include "ggml.h"
#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32)
#    include <sys/wait.h>
#    include <unistd.h>
#endif

using namespace ggml_sycl;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

std::atomic<uint64_t> g_next_id{ 5000 };
std::atomic<int>      g_released{ 0 };

release_attempt count_release(const alloc_metadata &, void *) noexcept {
    g_released.fetch_add(1);
    return { release_attempt_status::RELEASED };
}

// An owner over `backing` of `size` bytes with a fresh allocator id.  `cohort`, when
// non-null, tags the owner as a tenant before it is shared.
mem_handle make_owner(unsigned char * backing, size_t size, const char * cohort = nullptr) {
    const uint64_t id = g_next_id.fetch_add(1);
    alloc_metadata metadata{};
    metadata.ptr      = backing;
    metadata.size     = size;
    metadata.device   = 0;
    metadata.id       = id;
    metadata.alloc_id = id;
    metadata.epoch_id = 1;
    metadata.tier     = alloc_tier::HOST_PINNED;
    auto fixture      = allocation_owner_test_create(metadata, count_release, nullptr);
    if (!fixture.result) {
        std::fprintf(stderr, "FAIL: allocation_owner_test_create failed\n");
        std::exit(1);
    }
    if (cohort) {
        fixture.result.owner.set_tenant_cohort(cohort);
    }
    return mem_handle::from_owned_alloc(std::move(fixture.result.owner));
}

mem_handle make_ownerless() {
    static unsigned char buf[64];
    return mem_handle::from_direct(buf, GGML_LAYOUT_AOS, false, mem_handle::HOST_DEVICE, sizeof(buf));
}

unsigned char g_block[4096];

void test_identity_of_an_owner() {
    mem_handle                h  = make_owner(g_block, 1024);
    const mem_handle_identity id = h.identity();
    CHECK(id.valid(), "an owner-backed handle has an identity");
    CHECK(id.allocation_id == h.debug_info().canonical_allocation_id, "the id is the allocator's, from the handle");
    CHECK(id.slice_offset == 0, "the root's slice offset is 0");
    CHECK(id.size == 1024, "the root's size is the allocation's");
    CHECK(h.identity_equal(id), "a handle equals its own identity");

    mem_handle copy = h;
    CHECK(copy.identity_equal(id), "a copy names the same identity");
    CHECK(copy.identity() == id, "a copy's identity compares equal as a value");

    mem_handle other = make_owner(g_block + 2048, 1024);
    CHECK(!other.identity_equal(id), "a different allocation is a different identity");
    CHECK(other.identity() != id, "a different allocation compares unequal as a value");
}

void test_slices() {
    mem_handle                root = make_owner(g_block, 1024);
    const mem_handle_identity rid  = root.identity();

    mem_handle                a  = root.slice(128, 256);
    const mem_handle_identity ia = a.identity();
    CHECK(ia.valid(), "a slice of an owner has an identity");
    CHECK(ia.allocation_id == rid.allocation_id, "a slice shares its root's allocation id");
    CHECK(ia.slice_offset == 128 && ia.size == 256, "a slice names its own range");
    CHECK(!root.identity_equal(ia), "the root is not its slice");
    CHECK(!a.identity_equal(rid), "a slice is not its root");

    mem_handle                b  = a.slice(64, 32);
    const mem_handle_identity ib = b.identity();
    CHECK(ib.slice_offset == 128 + 64 && ib.size == 32, "a slice of a slice composes its offsets");

    mem_handle same = root.slice(128, 256);
    CHECK(same.identity_equal(ia), "two slices of one range are one identity");
    mem_handle shifted = root.slice(129, 256);
    CHECK(!shifted.identity_equal(ia), "a one-byte shift is a different identity");
    mem_handle shorter = root.slice(128, 255);
    CHECK(!shorter.identity_equal(ia), "a one-byte shorter slice is a different identity");
}

void test_no_identity() {
    mem_handle                ownerless = make_ownerless();
    const mem_handle_identity id        = ownerless.identity();
    CHECK(!id.valid(), "a handle with no allocator id has no identity");
    CHECK(!ownerless.identity_equal(id), "an invalid identity is never equal to a handle");
    CHECK(!(id == id), "an invalid identity equals nothing, itself included");
    CHECK(id != id, "an invalid identity is unequal to itself");

    mem_handle empty;
    CHECK(!empty.identity().valid(), "an empty handle has no identity");

    mem_handle                owner = make_owner(g_block, 64);
    const mem_handle_identity real  = owner.identity();
    CHECK(!(real == id) && !(id == real), "a valid and an invalid identity are unequal both ways");
    CHECK(!ownerless.identity_equal(real), "an ownerless view never equals a real identity");
}

// §B.2: an identity is never an address.  A source freed and a new one minted at the
// SAME address must miss.
void test_same_address_misses() {
    mem_handle_identity first;
    {
        mem_handle h = make_owner(g_block, 512);
        first        = h.identity();
        CHECK(first.valid(), "the first allocation has an identity");
    }
    CHECK(g_released.load() >= 1, "the first allocation was released");

    // Same backing address, same size: only the allocator's id differs.
    mem_handle second = make_owner(g_block, 512);
    CHECK(second.identity().valid(), "the reallocation has an identity");
    CHECK(!second.identity_equal(first), "a reallocation at the same address misses");
    CHECK(second.identity() != first, "a reallocation at the same address compares unequal as a value");
}

// An identity holds no control: the allocation is released when its last HANDLE goes,
// whatever identities are still around.
void test_identity_holds_no_reference() {
    mem_handle                h        = make_owner(g_block, 256);
    const mem_handle_identity id       = h.identity();
    const int                 before   = g_released.load();
    const bool                released = h.reset_owned_allocation().released();
    CHECK(released, "resetting the only handle releases the allocation");
    CHECK(g_released.load() == before + 1, "the release ran once");
    CHECK(id.valid(), "the identity value outlives the allocation");

    mem_handle later = make_owner(g_block, 256);
    CHECK(!later.identity_equal(id), "a stale identity never names a later allocation");
}

void test_generation_is_part_of_the_identity() {
    mem_handle                a     = make_owner(g_block, 128);
    // Two identities of one allocation id but a different generation (a replaced
    // incarnation) are different sources.
    const mem_handle_identity ia    = a.identity();
    mem_handle_identity       later = ia;
    later.generation += 1;
    CHECK(later.valid() && later.allocation_id == ia.allocation_id, "the twin kept the allocation id");
    CHECK(!a.identity_equal(later), "a different generation is a different identity");
    CHECK(later != ia, "a different generation compares unequal as a value");
    CHECK(later.hash() != ia.hash(), "a different generation hashes differently");
}

void test_hash() {
    mem_handle                root = make_owner(g_block, 1024);
    const mem_handle_identity a    = root.slice(0, 64).identity();
    const mem_handle_identity b    = root.slice(0, 64).identity();
    const mem_handle_identity c    = root.slice(64, 64).identity();
    CHECK(a.hash() == b.hash(), "equal identities hash equal");
    CHECK(a.hash() != c.hash(), "a shifted slice hashes differently");

    std::unordered_set<size_t> seen;
    for (size_t off = 0; off < 1024; off += 16) {
        seen.insert(root.slice(off, 16).identity().hash());
    }
    CHECK(seen.size() > 60, "the hash spreads distinct slices");
}

void test_tenant_cohort() {
    static const char kCohort[] = "context-test-cohort";

    mem_handle plain = make_owner(g_block, 256);
    CHECK(plain.tenant_cohort() == nullptr, "an untagged owner is not a tenant");
    CHECK(plain.slice(0, 64).tenant_cohort() == nullptr, "a slice of an untagged owner is not a tenant");

    mem_handle tenant = make_owner(g_block + 1024, 256, kCohort);
    CHECK(tenant.tenant_cohort() == kCohort, "a tagged owner reads its cohort");
    mem_handle copy = tenant;
    CHECK(copy.tenant_cohort() == kCohort, "a copy reads the same tag");
    mem_handle slice = tenant.slice(64, 64);
    CHECK(slice.tenant_cohort() == kCohort, "a slice reads the same tag");
    mem_handle slice2 = slice.slice(8, 8);
    CHECK(slice2.tenant_cohort() == kCohort, "a slice of a slice reads the same tag");

    CHECK(make_ownerless().tenant_cohort() == nullptr, "an ownerless handle is not a tenant");
    mem_handle empty;
    CHECK(empty.tenant_cohort() == nullptr, "an empty handle is not a tenant");

    // The tag does not leak across owners.
    mem_handle other = make_owner(g_block + 2048, 256);
    CHECK(other.tenant_cohort() == nullptr, "a later untagged owner is not a tenant");
}

// A fresh owner for the set-once arms, left untagged and not yet shared.
alloc_owner make_untagged_owner() {
    const uint64_t id = g_next_id.fetch_add(1);
    alloc_metadata metadata{};
    metadata.ptr      = g_block + 2048;
    metadata.size     = 128;
    metadata.device   = 0;
    metadata.id       = id;
    metadata.alloc_id = id;
    metadata.epoch_id = 1;
    metadata.tier     = alloc_tier::HOST_PINNED;
    auto fixture      = allocation_owner_test_create(metadata, count_release, nullptr);
    if (!fixture.result) {
        std::fprintf(stderr, "FAIL: allocation_owner_test_create failed\n");
        std::exit(1);
    }
    return std::move(fixture.result.owner);
}

void test_tenant_cohort_is_set_once() {
    static const char kFirst[]  = "context-first";
    static const char kSecond[] = "context-second";
    {
        alloc_owner owner = make_untagged_owner();
        owner.set_tenant_cohort(nullptr);
        mem_handle untagged = mem_handle::from_owned_alloc(std::move(owner));
        CHECK(untagged.tenant_cohort() == nullptr, "a null cohort leaves the owner untagged");
    }
    {
        alloc_owner owner = make_untagged_owner();
        owner.set_tenant_cohort(kFirst);
        owner.set_tenant_cohort(nullptr);
        mem_handle tagged = mem_handle::from_owned_alloc(std::move(owner));
        CHECK(tagged.tenant_cohort() == kFirst, "a null cohort does not clear a tag");
    }
#if !defined(_WIN32)
    // The second set aborts, so it runs in a child; a positive control (one set
    // exits 0) keeps a child that dies for another reason from passing the arm.
    for (const bool twice : { false, true }) {
        int fds[2] = { -1, -1 };
        CHECK(pipe(fds) == 0, "pipe failed");
        const pid_t pid = fork();
        CHECK(pid >= 0, "fork failed");
        if (pid == 0) {
            setenv("GGML_NO_BACKTRACE", "1", 1);
            dup2(fds[1], STDERR_FILENO);
            close(fds[0]);
            close(fds[1]);
            alloc_owner owner = make_untagged_owner();
            owner.set_tenant_cohort(kFirst);
            if (twice) {
                owner.set_tenant_cohort(kSecond);
            }
            _exit(0);
        }
        close(fds[1]);
        std::string out;
        char        buf[512];
        ssize_t     n;
        while ((n = read(fds[0], buf, sizeof(buf))) > 0) {
            out.append(buf, static_cast<size_t>(n));
        }
        close(fds[0]);
        int status = 0;
        CHECK(waitpid(pid, &status, 0) == pid, "waitpid failed");
        if (twice) {
            CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "a second set of the cohort aborts");
            CHECK(out.find("[TENANT] allocation already tagged") != std::string::npos,
                  "the abort is the set-once message, not another death");
        } else {
            CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "one set of the cohort exits cleanly (the control)");
            CHECK(out.find("already tagged") == std::string::npos, "the control prints no tag complaint");
        }
    }
#endif
}

}  // namespace

int main() {
    test_tenant_cohort_is_set_once();
    test_identity_of_an_owner();
    test_slices();
    test_no_identity();
    test_same_address_misses();
    test_identity_holds_no_reference();
    test_generation_is_part_of_the_identity();
    test_hash();
    test_tenant_cohort();
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-mem-handle-identity: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-mem-handle-identity: all ok\n");
    return 0;
}
