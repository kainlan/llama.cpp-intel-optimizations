#include "ggml-sycl-test.hpp"
#include "moe-layer-plan.hpp"
#include "unified-cache.hpp"

#include <cstdio>

#define CHECK(cond, msg)                             \
    do {                                             \
        if (!(cond)) {                               \
            std::fprintf(stderr, "FAIL: %s\n", msg); \
            return 1;                                \
        }                                            \
    } while (0)

struct stable_handle_fixture {
    alignas(64) float storage[16] = {};
    ggml_sycl::unified_cache_entry entry{};
    ggml_sycl::mem_handle          handle;

    stable_handle_fixture(ggml_layout_mode layout    = GGML_LAYOUT_AOS,
                          bool             on_device = true,
                          int              device    = 0,
                          const void *     tag       = nullptr) {
        entry.in_use_count.store(1);
        ggml_sycl_cache_id id = ggml_sycl::test_make_cache_id(tag ? tag : storage);
        handle = ggml_sycl::mem_handle::from_weight_lease(id, device, storage, layout, on_device, &entry);
        handle.set_debug_owner("descriptor-entry");
    }
};

static int test_descriptor_retains_and_replays_valid_handle() {
    stable_handle_fixture h;

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("gate", h.handle, GGML_LAYOUT_AOS, 0);

    CHECK(desc.retained_handle_count() == 1, "descriptor must retain one handle");
    CHECK(desc.entries()[0].handle.stable_identity_equal(h.handle), "retained handle identity must match source");
    CHECK(desc.validate_for_replay(), "fresh descriptor must validate");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::NONE, "fresh reject reason none");
    return 0;
}

static int test_descriptor_rejects_invalid_handle() {
    ggml_sycl::residency_diagnostics_reset_for_test();

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("missing", ggml_sycl::mem_handle{}, GGML_LAYOUT_AOS, ggml_sycl::mem_handle::HOST_DEVICE);

    CHECK(!desc.validate_for_replay(), "invalid handle must reject replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::INVALID_HANDLE,
          "reject reason must be invalid handle");
    const auto snap = ggml_sycl::residency_diagnostics_snapshot_for_test();
    CHECK(snap.stale_descriptor_rejects == 1, "invalid descriptor diagnostic must increment");
    CHECK(snap.stale_descriptor_invalid_handle == 1, "invalid-handle counter must increment");
    return 0;
}

static int test_descriptor_rejects_raw_direct_handle() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    alignas(64) float     storage[16] = {};
    ggml_sycl::mem_handle raw         = ggml_sycl::mem_handle::from_direct(storage, GGML_LAYOUT_AOS, false);

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("raw-direct", raw, GGML_LAYOUT_AOS, ggml_sycl::mem_handle::HOST_DEVICE);

    CHECK(!desc.validate_for_replay(), "raw DIRECT pointer handle must reject persistent replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::INVALID_HANDLE,
          "raw direct reject reason must be invalid handle");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().stale_descriptor_invalid_handle == 1,
          "raw direct invalid-handle counter must increment");
    return 0;
}

static int test_descriptor_rejects_unresolvable_arena_as_invalid() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    ggml_sycl::mem_handle h =
        ggml_sycl::mem_handle::from_arena_zone(static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 256, 0, 7);
    h.set_debug_owner("stale-arena-generation");

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("arena", h, GGML_LAYOUT_AOS, 0);

    CHECK(!desc.validate_for_replay(), "unresolvable arena handle must reject replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::INVALID_HANDLE,
          "unresolvable arena reject reason must be invalid handle");
    const auto snap = ggml_sycl::residency_diagnostics_snapshot_for_test();
    CHECK(snap.stale_descriptor_rejects == 1, "stale descriptor diagnostic must increment");
    CHECK(snap.stale_descriptor_invalid_handle == 1, "invalid-handle counter must increment");
    CHECK(snap.stale_descriptor_generation_mismatch == 0,
          "unresolvable arena must not be mislabeled as generation mismatch");
    return 0;
}

static int test_descriptor_ignores_weight_generation_metadata() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture h(GGML_LAYOUT_AOS);

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("weight", h.handle, GGML_LAYOUT_AOS, 0);
    desc.override_generation_for_test(0, h.handle.generation() + 1);

    CHECK(desc.validate_for_replay(), "weight handles must not use arena generation metadata");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::NONE,
          "weight generation metadata override must not reject replay");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().stale_descriptor_generation_mismatch == 0,
          "weight handles must not increment generation mismatch counter");
    return 0;
}

static int test_descriptor_rejects_identity_mismatch() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture a(GGML_LAYOUT_AOS, true, 0, reinterpret_cast<void *>(1));
    stable_handle_fixture b(GGML_LAYOUT_AOS, true, 0, reinterpret_cast<void *>(2));

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("gate", a.handle, GGML_LAYOUT_AOS, 0);
    desc.replace_retained_handle_for_test(0, b.handle);

    CHECK(!desc.validate_for_replay(), "identity change must reject replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::IDENTITY_MISMATCH,
          "reject reason must be identity mismatch");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().stale_descriptor_identity_mismatch == 1,
          "identity mismatch counter must increment");
    return 0;
}

static int test_descriptor_rejects_device_mismatch() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture h(GGML_LAYOUT_AOS, true, 1);

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("device", h.handle, GGML_LAYOUT_AOS, 0);

    CHECK(!desc.validate_for_replay(), "device mismatch must reject replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::DEVICE_MISMATCH,
          "reject reason must be device mismatch");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().stale_descriptor_device_mismatch == 1,
          "device mismatch counter must increment");
    return 0;
}

static int test_descriptor_rejects_host_handle_for_device() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture h(GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE);

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("host-for-device", h.handle, GGML_LAYOUT_AOS, 0);

    CHECK(!desc.validate_for_replay(), "host handle must not satisfy device descriptor replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::DEVICE_MISMATCH,
          "host-for-device reject reason must be device mismatch");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().stale_descriptor_device_mismatch == 1,
          "host-for-device mismatch counter must increment");
    return 0;
}

static int test_descriptor_rejects_layout_mismatch() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture h(GGML_LAYOUT_AOS);

    ggml_sycl::moe_residency_descriptor desc;
    desc.add_entry_for_test("up", h.handle, GGML_LAYOUT_AOS, 0);
    desc.override_layout_for_test(0, GGML_LAYOUT_SOA);

    CHECK(!desc.validate_for_replay(), "layout change must reject replay");
    CHECK(desc.last_reject_reason() == ggml_sycl::moe_descriptor_reject_reason::LAYOUT_MISMATCH,
          "reject reason must be layout mismatch");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().stale_descriptor_layout_mismatch == 1,
          "layout mismatch counter must increment");
    return 0;
}

int main() {
    if (int rc = test_descriptor_retains_and_replays_valid_handle()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_invalid_handle()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_raw_direct_handle()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_unresolvable_arena_as_invalid()) {
        return rc;
    }
    if (int rc = test_descriptor_ignores_weight_generation_metadata()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_identity_mismatch()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_device_mismatch()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_host_handle_for_device()) {
        return rc;
    }
    if (int rc = test_descriptor_rejects_layout_mismatch()) {
        return rc;
    }
    std::puts("PASS: descriptor retention and replay validation");
    return 0;
}
