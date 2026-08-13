#include "unified-cache.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace ggml_sycl;

namespace {
std::atomic<bool> fail_allocations{false};

struct fake_release_backend {
    std::atomic<int> attempts{0};
    std::atomic<int> releases{0};
    std::atomic<int> refusals_remaining{0};
};

release_attempt release_fake(const alloc_metadata &, void * opaque) noexcept {
    auto & backend = *static_cast<fake_release_backend *>(opaque);
    backend.attempts.fetch_add(1, std::memory_order_relaxed);
    int remaining = backend.refusals_remaining.load(std::memory_order_relaxed);
    while (remaining > 0 && !backend.refusals_remaining.compare_exchange_weak(
               remaining, remaining - 1, std::memory_order_relaxed)) {}
    if (remaining > 0) return {release_attempt_status::RETRY_SCHEDULED};
    backend.releases.fetch_add(1, std::memory_order_relaxed);
    return {release_attempt_status::RELEASED};
}

alloc_metadata metadata(uint64_t id) {
    alloc_metadata value;
    value.ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000 + id * 0x100));
    value.size = 256;
    value.device = 0;
    value.id = id;
    return value;
}

[[noreturn]] void fail(const char * message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) fail(message);
}

allocation_owner_test_fixture make_owner(fake_release_backend & backend, uint64_t id) {
    auto fixture = allocation_owner_test_create(metadata(id), release_fake, &backend);
    check(static_cast<bool>(fixture.result), "test owner creation failed");
    check(fixture.coordinator && fixture.coordinator->live_controls() == 1, "new owner was not counted live");
    return fixture;
}

void unique_to_shared_preserves_identity() {
    fake_release_backend backend;
    auto fixture = make_owner(backend, 1);
    const alloc_metadata before = fixture.result.owner.metadata();
    shared_alloc_owner shared = std::move(fixture.result.owner).into_shared();
    check(!fixture.result.owner, "unique owner retained control after into_shared");
    check(shared.metadata() == before && shared.use_count() == 1, "unique-to-shared changed identity or reference");
    check(shared.reset().released(), "shared final release did not complete");
    check(backend.releases == 1 && fixture.coordinator->live_controls() == 0, "identity conversion duplicated release");
    std::cout << "PASS unique-to-shared-identity\n";
}

void concurrent_final_release_exactly_once() {
    fake_release_backend backend;
    auto fixture = make_owner(backend, 2);
    shared_alloc_owner root = std::move(fixture.result.owner).into_shared();
    constexpr int thread_count = 24;
    std::vector<shared_alloc_owner> copies;
    copies.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) copies.push_back(root);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (auto & copy : copies) {
        threads.emplace_back([owner = std::move(copy), &ready, &go]() mutable {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            (void) owner.reset();
        });
    }
    while (ready.load(std::memory_order_acquire) != thread_count) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    (void) root.reset();
    for (auto & thread : threads) thread.join();

    check(backend.attempts == 1 && backend.releases == 1, "concurrent final release was not exactly once");
    check(fixture.coordinator->live_controls() == 0, "concurrent release leaked its control");
    std::cout << "PASS concurrent-exactly-once-final-release\n";
}

void refusal_is_allocation_free_and_retries_without_duplicates() {
    fake_release_backend backend;
    backend.refusals_remaining = 3;
    auto fixture = make_owner(backend, 3);

    fail_allocations.store(true, std::memory_order_release);
    const release_attempt first = fixture.result.owner.reset();
    check(first.retry_scheduled(), "refused final release was not scheduled");
    check(fixture.coordinator->retry_count() == 1, "refused control was not queued exactly once");
    check(fixture.coordinator->process_retries() == 0, "refused retry reported a release");
    check(fixture.coordinator->retry_count() == 1, "first refusal duplicated or lost embedded node");
    check(fixture.coordinator->process_retries() == 0, "second refused retry reported a release");
    check(fixture.coordinator->retry_count() == 1, "second refusal duplicated or cycled embedded node");
    fail_allocations.store(false, std::memory_order_release);

    check(fixture.coordinator->process_retries() == 1, "successful retry did not release one control");
    check(fixture.coordinator->retry_count() == 0 && fixture.coordinator->live_controls() == 0,
          "successful retry left queue or live control state");
    check(backend.attempts == 4 && backend.releases == 1, "retry backend call counts were not deterministic");
    check(fixture.coordinator->process_retries() == 0 && backend.attempts == 4,
          "empty retry processing revisited a released node");
    std::cout << "PASS refused-release-allocation-free\n"
                 "PASS repeated-refusal-single-embedded-node\n"
                 "PASS retry-success\n";
}

void late_shutdown_is_busy_then_clean() {
    fake_release_backend backend;
    backend.refusals_remaining = 1;
    auto fixture = make_owner(backend, 4);
    check(!fixture.coordinator->can_detach(), "shutdown detached a live control");
    check(fixture.result.owner.reset().retry_scheduled(), "shutdown fixture did not enter retry state");
    check(!fixture.coordinator->can_detach(), "shutdown detached a queued retry");
    check(fixture.coordinator->process_retries() == 1, "shutdown retry did not complete");
    check(!fixture.coordinator->can_detach(), "open coordinator reported detachable");
    allocation_coordinator_test_close(fixture.coordinator);
    check(fixture.coordinator->can_detach(), "shutdown remained busy after close and retry drain");
    std::cout << "PASS late-shutdown-busy-then-clean\n";
}

void coordinator_close_linearizes_with_registration() {
    constexpr int device = GGML_SYCL_MAX_DEVICES - 1;
    std::mutex mutex;
    std::condition_variable cv;
    bool looked_up = false;
    bool may_register = false;
    bool registered = true;
    std::shared_ptr<allocation_release_coordinator> looked_up_coordinator;

    std::thread lookup_then_register([&] {
        looked_up_coordinator = allocation_coordinator_test_lookup(device);
        {
            std::lock_guard<std::mutex> lock(mutex);
            looked_up = true;
        }
        cv.notify_one();
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return may_register; });
        lock.unlock();
        registered = allocation_coordinator_test_try_register(looked_up_coordinator);
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return looked_up; });
    }
    allocation_coordinator_test_close_all();
    {
        std::lock_guard<std::mutex> lock(mutex);
        may_register = true;
    }
    cv.notify_one();
    lookup_then_register.join();

    check(!registered, "registration succeeded after coordinator close");
    check(looked_up_coordinator && looked_up_coordinator->live_controls() == 0,
          "rejected registration changed live census");
    check(unified_allocation_release_coordinator_detach(device), "closed clean coordinator did not detach");
    std::cout << "PASS coordinator-close-register-linearization\n";
}

void intrusive_registry_row_rejects_legacy_claim() {
    const alloc_metadata exact = metadata(80);
    check(allocation_registry_test_publish(exact, true), "intrusive test row publication failed");
    check(allocation_registry_test_claim(exact, false) == registered_release_status::OWNERSHIP_MISMATCH,
          "legacy claim acquired intrusive row");
    check(allocation_registry_test_claim(exact, true) == registered_release_status::RELEASED,
          "intrusive owner could not claim its exact row");
    allocation_registry_test_erase(exact.ptr);
    std::cout << "PASS legacy-release-rejects-intrusive-row\n"
                 "PASS intrusive-exact-row-claim\n";
}

void invalid_request_has_zero_coordinator_census() {
    const size_t before = allocation_coordinator_test_count();
    alloc_request invalid{};
    invalid.device = -2;
    invalid.size = 256;
    const allocation_result result = unified_allocate_owner(invalid);
    check(result.error == allocation_error::INVALID_REQUEST, "invalid request returned wrong error");
    check(allocation_coordinator_test_count() == before, "invalid request acquired a coordinator");
    std::cout << "PASS invalid-request-zero-coordinator-census\n";
}

void failure_accounting_and_metadata_nonownership() {
    fake_release_backend backend;
    auto control_failure = allocation_owner_test_create(metadata(5), release_fake, &backend,
                                                         allocation_error::CONTROL_ALLOCATION_FAILED);
    check(control_failure.result.error == allocation_error::CONTROL_ALLOCATION_FAILED,
          "control failure returned wrong error");
    check(control_failure.coordinator && control_failure.coordinator->live_controls() == 0,
          "control allocation failure changed live count");

    auto physical_failure = allocation_owner_test_create(metadata(6), release_fake, &backend,
                                                          allocation_error::PHYSICAL_ALLOCATION_FAILED);
    check(physical_failure.result.error == allocation_error::PHYSICAL_ALLOCATION_FAILED,
          "physical failure returned wrong error");
    check(physical_failure.coordinator && physical_failure.coordinator->live_controls() == 0,
          "physical allocation failure leaked a live control");

    auto fixture = make_owner(backend, 7);
    alloc_metadata copy_a = fixture.result.owner.metadata();
    alloc_metadata copy_b = copy_a;
    check(backend.attempts == 0 && fixture.coordinator->live_controls() == 1,
          "copyable metadata acquired ownership");
    check(fixture.result.owner.reset().released(), "metadata test owner did not release");
    check(copy_a == copy_b && backend.releases == 1 && fixture.coordinator->live_controls() == 0,
          "metadata copies affected physical ownership");
    std::cout << "PASS allocation-failure-live-count-zero\n"
                 "PASS metadata-cannot-own\n";
}
} // namespace

void * operator new(std::size_t size) {
    if (fail_allocations.load(std::memory_order_acquire)) throw std::bad_alloc();
    if (void * ptr = std::malloc(size)) return ptr;
    throw std::bad_alloc();
}
void * operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void * ptr) noexcept { std::free(ptr); }
void operator delete[](void * ptr) noexcept { std::free(ptr); }
void operator delete(void * ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void * ptr, std::size_t) noexcept { std::free(ptr); }

static_assert(std::is_copy_constructible_v<alloc_metadata>);
static_assert(std::is_trivially_destructible_v<alloc_metadata>);
static_assert(!std::is_constructible_v<alloc_owner, alloc_metadata>);
static_assert(!std::is_constructible_v<shared_alloc_owner, alloc_metadata>);

int main() {
    unique_to_shared_preserves_identity();
    concurrent_final_release_exactly_once();
    refusal_is_allocation_free_and_retries_without_duplicates();
    late_shutdown_is_busy_then_clean();
    failure_accounting_and_metadata_nonownership();
    coordinator_close_linearizes_with_registration();
    intrusive_registry_row_rejects_legacy_claim();
    invalid_request_has_zero_coordinator_census();
    std::cout << "intrusive allocation owner deterministic runtime tests: PASS\n" << std::flush;
    std::_Exit(0);
}
