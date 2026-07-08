#include "ggml-sycl-test.hpp"
#include "residency-plan.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond, msg)                             \
    do {                                             \
        if (!(cond)) {                               \
            std::fprintf(stderr, "FAIL: %s\n", msg); \
            return 1;                                \
        }                                            \
    } while (0)

static int test_accept_and_reject_counters() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    ggml_sycl::residency_diagnostics_record_accept_for_test(1024, 4096, 2048);
    ggml_sycl::residency_diagnostics_record_reject_for_test(ggml_sycl::residency_reject_reason::BUDGET, 8192, 4096,
                                                            2048);
    ggml_sycl::residency_diagnostics_record_reject_for_test(ggml_sycl::residency_reject_reason::FRAGMENTATION, 3072,
                                                            4096, 1024);

    ggml_sycl::residency_diagnostics_snapshot snap = ggml_sycl::residency_diagnostics_snapshot_for_test();
    CHECK(snap.accept_count == 1, "accept counter must increment");
    CHECK(snap.reject_budget == 1, "budget reject counter must increment");
    CHECK(snap.reject_fragmentation == 1, "fragmentation reject counter must increment");
    CHECK(snap.last_bytes_requested == 3072, "last reject bytes requested recorded");
    CHECK(snap.last_largest_free_block == 1024, "last largest free block recorded");
    return 0;
}

static int test_live_owner_and_stale_descriptor_counters() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    ggml_sycl::residency_diagnostics_record_live_handle_for_test("owner-a", "WEIGHT", 1234);
    ggml_sycl::residency_diagnostics_record_stale_descriptor_for_test();
    ggml_sycl::residency_diagnostics_record_stale_descriptor_invalid_handle_for_test();
    ggml_sycl::residency_diagnostics_record_stale_descriptor_identity_mismatch_for_test();
    ggml_sycl::residency_diagnostics_record_stale_descriptor_generation_mismatch_for_test();
    ggml_sycl::residency_diagnostics_record_stale_descriptor_layout_mismatch_for_test();
    ggml_sycl::residency_diagnostics_record_stale_descriptor_device_mismatch_for_test();

    ggml_sycl::residency_diagnostics_snapshot snap = ggml_sycl::residency_diagnostics_snapshot_for_test();
    CHECK(snap.live_handle_count == 1, "live handle counter must increment");
    CHECK(std::strcmp(snap.last_live_owner_tag, "owner-a") == 0, "live owner tag must be recorded");
    CHECK(std::strcmp(snap.last_live_allocation_class, "WEIGHT") == 0, "live allocation class must be recorded");
    CHECK(snap.stale_descriptor_rejects == 6, "stale descriptor aggregate counter must increment");
    CHECK(snap.stale_descriptor_invalid_handle == 1, "invalid-handle stale counter must increment");
    CHECK(snap.stale_descriptor_identity_mismatch == 1, "identity stale counter must increment");
    CHECK(snap.stale_descriptor_generation_mismatch == 1, "generation stale counter must increment");
    CHECK(snap.stale_descriptor_layout_mismatch == 1, "layout stale counter must increment");
    CHECK(snap.stale_descriptor_device_mismatch == 1, "device stale counter must increment");
    return 0;
}

static int test_reject_reason_names() {
    CHECK(
        std::strcmp(ggml_sycl::residency_reject_reason_name(ggml_sycl::residency_reject_reason::BUDGET), "budget") == 0,
        "budget reason name must be stable");
    CHECK(std::strcmp(ggml_sycl::residency_reject_reason_name(ggml_sycl::residency_reject_reason::LIVE_LEASE_PRESSURE),
                      "live-lease-pressure") == 0,
          "live lease reason name must be stable");
    return 0;
}

static int test_replacement_guard_refuses_live_or_retired_entries() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    CHECK(ggml_sycl::test_cache_replacement_allowed_for_test(0, false), "idle entry replacement must be allowed");

    ggml_sycl::residency_diagnostics_reset_for_test();
    CHECK(!ggml_sycl::test_cache_replacement_allowed_for_test(1, false), "live entry replacement must be refused");
    auto snap = ggml_sycl::residency_diagnostics_snapshot_for_test();
    CHECK(snap.live_handle_count == 1, "live replacement refusal must record owner");
    CHECK(snap.reject_live_lease_pressure == 1, "live replacement refusal must record LIVE_LEASE_PRESSURE");

    ggml_sycl::residency_diagnostics_reset_for_test();
    CHECK(!ggml_sycl::test_cache_replacement_allowed_for_test(0, true), "retired entry replacement must be refused");
    snap = ggml_sycl::residency_diagnostics_snapshot_for_test();
    CHECK(snap.live_handle_count == 1, "retired replacement refusal must record owner");
    CHECK(snap.reject_live_lease_pressure == 1, "retired replacement refusal must record LIVE_LEASE_PRESSURE");
    return 0;
}

int main() {
    if (int rc = test_accept_and_reject_counters()) {
        return rc;
    }
    if (int rc = test_live_owner_and_stale_descriptor_counters()) {
        return rc;
    }
    if (int rc = test_reject_reason_names()) {
        return rc;
    }
    if (int rc = test_replacement_guard_refuses_live_or_retired_entries()) {
        return rc;
    }
    std::puts("PASS: residency diagnostics");
    return 0;
}
