//
// llama.cpp-kpjw: the hold-fit ledger, driven through the real allocation registry.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The hold-spill fit (zone_hold_fit) is a pure function of its inputs; what it is given is not. Its free memory is
// the cache's own ledger (unified_cache_hold_free_before): a baseline taken from the driver, less the raw bytes that
// stay live without the rung's own buffers. That is state in the allocation registry, so it is tested here against
// registry rows, not against the arithmetic alone (tests/test-zone-sizing.cpp Cases 26-30 cover that). No device is
// touched: the rows are host-only test rows (allocation_registry_test_publish_raw) and the free-memory reading a
// caller would take from the driver is passed in by the test.
//
// What each case pins, in the reviewer's words (r7):
//   I3  a raw allocation that is not the rung's own scheduler compute buffer is persistent however late it came
//       (the recurrent state, made after a pinned -ub's one publish, was credited back as "the rung's own");
//   I2  the baseline is the maximum reading within a window and a publish opens a new window; a row in RELEASING
//       is still held, so its bytes are not credited before the physical free;
//   M4  an owner takeover leaves no epoch of the previous owner behind;
//   M5  the per-rung records are bounded, and the bound is loud, not a silent truncation.

#include "../unified-cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace ggml_sycl;

namespace {

constexpr size_t MiB = 1024ull * 1024ull;

[[noreturn]] void fail(const char * message) {
    fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

void * fake(uintptr_t n) {
    return reinterpret_cast<void *>(0x100000 + n * 0x1000);
}

struct hold_owner {
    int      device;
    uint64_t owner;

    explicit hold_owner(int d) : device(d), owner(unified_cache_mint_planned_scratch_owner()) {
        unified_cache_set_planned_scratch_hold(device, 76 * MiB, owner);
    }

    ~hold_owner() { unified_cache_release_planned_scratch_hold(device, owner); }
};

}  // namespace

int main() {
    const int dev = 0;

    // ---- the baseline is the maximum reading of a window; a publish opens a new window ----------------------------
    {
        hold_owner o(dev);
        unified_cache_begin_planned_hold_epoch(dev, o.owner, 512, 100 * MiB);
        check(unified_cache_hold_free_before(dev, o.owner, 800 * MiB, false) == 800 * MiB,
              "the first reading of a window is the baseline");
        check(unified_cache_hold_free_before(dev, o.owner, 300 * MiB, false) == 800 * MiB,
              "a later lagged reading does not lower it (a baseline re-taken every call would read 300)");
        check(unified_cache_hold_free_before(dev, o.owner, 900 * MiB, false) == 900 * MiB,
              "a later healthier reading raises it");
        unified_cache_begin_planned_hold_epoch(dev, o.owner, 1024, 100 * MiB);
        check(unified_cache_hold_free_before(dev, o.owner, 300 * MiB, false) == 300 * MiB,
              "a publish opens a new window: the old maximum is forgotten (another tenant may have arrived)");
    }

    // ---- I3: what is credited back is the rung's own SCHEDULER COMPUTE rows, never a row that merely came later ----
    {
        hold_owner o(dev);
        unified_cache_begin_planned_hold_epoch(dev, o.owner, 1024, 100 * MiB);
        // After the epoch began (a pinned -ub publishes once, before the memory module exists): the 461 MB recurrent
        // state, a raw row nobody flagged as a scheduler compute buffer, then the rung's own 495 MB compute buffer.
        check(allocation_registry_test_publish_raw(fake(1), dev, 461 * MiB, false, false, false), "state row");
        check(allocation_registry_test_publish_raw(fake(2), dev, 495 * MiB, true, false, false), "compute row");
        // The driver shows 600 MB free with both live; cold = 600 + 956.
        check(unified_cache_hold_free_before(dev, o.owner, 600 * MiB, true) == 1095 * MiB,
              "a live rung: its own compute row is credited back, the state allocated after the epoch is NOT "
              "(credited as the rung's own it reads 1556)");
        check(unified_cache_hold_free_before(dev, o.owner, 600 * MiB, false) == 600 * MiB,
              "a transaction (no live rung): every raw row is persistent, the driver's reading");
        // The rung's compute row is released (the settle, the next rung): the ledger is not asked the driver again, and
        // the verdict for the same rung is the one it had live.
        allocation_registry_test_erase(fake(2));
        check(unified_cache_hold_free_before(dev, o.owner, 600 * MiB, true) == 1095 * MiB,
              "release-then-evaluate: the baseline stands (a lagged driver reading is not used) and the rung's bytes "
              "are simply no longer live");
        check(unified_cache_hold_free_before(dev, o.owner, 600 * MiB, false) == 1095 * MiB,
              "and a transaction sees the same card");
        allocation_registry_test_erase(fake(1));
    }

    // ---- I2: a row that is RELEASING is still held until its physical free happened ------------------------------
    {
        hold_owner o(dev);
        unified_cache_begin_planned_hold_epoch(dev, o.owner, 512, 100 * MiB);
        check(allocation_registry_test_publish_raw(fake(3), dev, 461 * MiB, false, false, false), "state row");
        check(allocation_registry_test_publish_raw(fake(4), dev, 200 * MiB, false, false, false), "other row");
        check(unified_cache_hold_free_before(dev, o.owner, 700 * MiB, false) == 700 * MiB,
              "window baseline, both live");
        // The 200 MB row starts its release; the driver has not credited it yet.
        allocation_registry_test_erase(fake(4));
        check(allocation_registry_test_publish_raw(fake(4), dev, 200 * MiB, false, false, true), "releasing row");
        check(unified_cache_hold_free_before(dev, o.owner, 700 * MiB, false) == 700 * MiB,
              "RELEASING is still held: its bytes are not credited before the physical free (excluded it reads 900)");
        // Its free lands: the row is gone and the driver now shows the bytes.
        allocation_registry_test_erase(fake(4));
        check(unified_cache_hold_free_before(dev, o.owner, 900 * MiB, false) == 900 * MiB,
              "after the free: the card's reading");
        allocation_registry_test_erase(fake(3));
    }

    // ---- an arena sub-allocation and another device's row are not this device's raw bytes -------------------------
    {
        hold_owner o(dev);
        unified_cache_begin_planned_hold_epoch(dev, o.owner, 512, 100 * MiB);
        check(allocation_registry_test_publish_raw(fake(5), dev, 300 * MiB, false, true, false), "arena row");
        check(allocation_registry_test_publish_raw(fake(6), dev + 1, 300 * MiB, false, false, false),
              "other device row");
        size_t compute_live = 1;
        check(unified_cache_raw_device_held_bytes(dev, &compute_live) == 0 && compute_live == 0,
              "neither counts as raw device memory on this device");
        check(unified_cache_hold_free_before(dev, o.owner, 500 * MiB, false) == 500 * MiB,
              "the ledger reads the driver");
        allocation_registry_test_erase(fake(5));
        allocation_registry_test_erase(fake(6));
    }

    // ---- the epoch carries the KV room it began with, and a refresh replaces only the room ------------------------
    {
        hold_owner o(dev);
        uint32_t   n_ubatch = 0;
        size_t     kv_room  = 0;
        check(!unified_cache_get_hold_epoch(dev, o.owner, &n_ubatch, &kv_room), "no epoch before the first publish");
        unified_cache_begin_planned_hold_epoch(dev, o.owner, 512, 123 * MiB);
        check(
            unified_cache_get_hold_epoch(dev, o.owner, &n_ubatch, &kv_room) && n_ubatch == 512 && kv_room == 123 * MiB,
            "the epoch is the n_ubatch and the room it began with");
        unified_cache_refresh_hold_epoch_kv_room(dev, o.owner, 40 * MiB);
        check(unified_cache_get_hold_epoch(dev, o.owner, &n_ubatch, &kv_room) && n_ubatch == 512 && kv_room == 40 * MiB,
              "a refresh replaces the room and nothing else (a pinned -ub's room, re-read once the memory module "
              "exists)");
        unified_cache_refresh_hold_epoch_kv_room(dev, o.owner + 1, 7 * MiB);
        check(unified_cache_get_hold_epoch(dev, o.owner, &n_ubatch, &kv_room) && kv_room == 40 * MiB,
              "another owner's refresh changes nothing");
    }

    // ---- M4: an owner takeover leaves no epoch of the previous owner behind ----------------------------------------
    {
        hold_owner a(dev);
        unified_cache_begin_planned_hold_epoch(dev, a.owner, 1024, 100 * MiB);
        (void) unified_cache_note_runtime_request(dev, 990 * MiB, true);
        const uint64_t b = unified_cache_mint_planned_scratch_owner();
        unified_cache_set_planned_scratch_hold(dev, 50 * MiB, b);
        uint32_t n_ubatch = 0;
        size_t   kv_room  = 0;
        check(!unified_cache_get_hold_epoch(dev, b, &n_ubatch, &kv_room),
              "the new owner has no epoch until its own first publish");
        (void) unified_cache_note_runtime_request(dev, 500 * MiB, true);
        zone_hold_rung_request recs[4];
        check(unified_cache_get_hold_rung_requests(dev, b, recs, 4) == 0,
              "a request before the new owner's first publish is a load-time one: recorded under nobody's n_ubatch");
        unified_cache_release_planned_scratch_hold(dev, b);
    }

    // ---- M5: the per-rung records are bounded and say so -----------------------------------------------------------
    {
        hold_owner o(dev);
        for (uint32_t i = 1; i <= 40; ++i) {
            unified_cache_begin_planned_hold_epoch(dev, o.owner, 32 * i, 100 * MiB);
            (void) unified_cache_note_runtime_request(dev, i * MiB, true);
        }
        zone_hold_rung_request recs[64];
        const size_t           n = unified_cache_get_hold_rung_requests(dev, o.owner, recs, 64);
        check(n == kHoldRungRecordLimit, "the records are capped at the documented limit");
        check(unified_cache_hold_rung_record_refusals(dev, o.owner) == 40 - n,
              "every call the cap refused is counted, so the truncation is never silent");
    }

    printf("PASS: hold ledger\n");
    return 0;
}
