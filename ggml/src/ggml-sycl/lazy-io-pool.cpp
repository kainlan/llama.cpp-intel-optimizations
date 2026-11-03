#include "lazy-io-pool.hpp"

#include "ggml-impl.h"

#include <utility>

// The interface only: every take returns an empty lease, so no pool exists, nothing is parked and no reap counts.
// The registry, the lease counts, the park and the reap follow their fixtures.

namespace ggml_sycl {

ring_lease::~ring_lease() {
    reset();
}

ring_lease::ring_lease(ring_lease && other) noexcept : pool_(other.pool_) {
    other.pool_ = nullptr;
}

ring_lease & ring_lease::operator=(ring_lease && other) noexcept {
    if (this != &other) {
        reset();
        pool_       = other.pool_;
        other.pool_ = nullptr;
    }
    return *this;
}

ring_lease ring_lease::clone() const noexcept {
    return ring_lease();
}

uint64_t ring_lease::ring_bytes() const noexcept {
    return 0;
}

void ring_lease::reset() noexcept {
    pool_ = nullptr;
}

ring_lease lazy_io_pool_take() {
    return ring_lease();
}

void ggml_sycl_lazy_ring_reap(lazy_reap_scope scope) noexcept {
    GGML_UNUSED(scope);
}

void lazy_io_pool_mark_shut_down() noexcept {}

lazy_coverage_scope::reap_guard::reap_guard() noexcept {}

lazy_coverage_scope::reap_guard::~reap_guard() {}

bool lazy_coverage_scope::take_if(bool has_lazy_candidate) {
    GGML_UNUSED(has_lazy_candidate);
    return false;
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
bool ggml_sycl_lazy_registry_for_testing(lazy_registry_state * out) {
    if (out == nullptr) {
        return false;
    }
    *out = lazy_registry_state{};
    return true;
}

void ggml_sycl_test_lazy_reap_hold(lazy_reap_hold_fn hold, void * ctx) {
    GGML_UNUSED(hold);
    GGML_UNUSED(ctx);
}

lazy_test_coverage_pin::lazy_test_coverage_pin(ring_lease lease) noexcept : lease_(std::move(lease)) {}

lazy_test_coverage_pin::~lazy_test_coverage_pin() {}

lazy_test_coverage_pin ggml_sycl_test_make_coverage_pin(uint64_t ring_bytes) {
    if (ring_bytes != 0) {
        GGML_ABORT("ggml_sycl_test_make_coverage_pin: ring_bytes=%llu; only a ring-less pool (0) is accepted",
                   (unsigned long long) ring_bytes);
    }
    return lazy_test_coverage_pin(lazy_io_pool_take());
}
#endif

}  // namespace ggml_sycl
