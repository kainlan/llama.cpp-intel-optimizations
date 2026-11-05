//
// test-sycl-host-scratch-demand: the host SCRATCH demand ledger behind ggml_backend_sycl_notify_host_buffer_sizes,
// over fake zones that record their grow_zone calls (docs/design/sycl-mmap-host-tier.md §4.4, "What the SYCL side does
// with it"). Host-only: the zones are fakes, not unified caches.
//
// Positive controls, one compile-time macro and one ctest target each (ggml/src/ggml-sycl/CMakeLists.txt):
//   HOST_ZONE_PLAN_MUTANT_LEDGER_ACCUMULATE    (host-zone-plan.cpp) accumulate instead of replace
//   HOST_SCRATCH_DEMAND_MUTANT_PER_BACKEND     (this file) one notify, so one grow, per SYCL backend of a context
//   HOST_ZONE_PLAN_MUTANT_LEDGER_ADDITIVE_GROW (host-zone-plan.cpp) the additive grow_scratch_zone it replaced
//   HOST_ZONE_PLAN_MUTANT_NOTIFY_DEVICE0       (host-zone-plan.cpp) the device-0 resolve
//   HOST_ZONE_PLAN_MUTANT_NOTIFY_CTX_DEVICE    (host-zone-plan.cpp) the notifying backend's ctx->device
//   HOST_ZONE_PLAN_MUTANT_LEDGER_NO_ROLLBACK   (host-zone-plan.cpp) a failed grow keeps its reservation
//   HOST_ZONE_PLAN_MUTANT_LEDGER_NO_RESERVE    (host-zone-plan.cpp) the deficit is not reserved before the unlock
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "host-zone-plan-test-util.hpp"
#include "host-zone-plan.hpp"

#include <algorithm>
#include <memory>

using namespace ggml_sycl;
using namespace host_zone_plan_test;

namespace {

constexpr size_t MiB = (size_t) 1 << 20;

// A fake SCRATCH zone. Like the pinned pool, one grow_zone commits one chunk of max(2048 MB, align_up(request)).
struct fake_zone {
    size_t capacity     = 0;
    int    grows        = 0;
    size_t last_request = 0;
    bool   fail         = false;
};

size_t fake_capacity(void * user) {
    return static_cast<fake_zone *>(user)->capacity;
}

bool fake_grow(void * user, size_t request) {
    fake_zone & z = *static_cast<fake_zone *>(user);
    if (z.fail) {
        return false;
    }
    ++z.grows;
    z.last_request    = request;
    const size_t page = 2 * MiB;
    z.capacity += std::max<size_t>(2048 * MiB, (request + page - 1) / page * page);
    return true;
}

constexpr size_t plan_scratch = 4096 * MiB;
constexpr size_t chunk_cap    = 256 * MiB;
constexpr size_t alignment    = 64;  // pinned_chunk_pool::DEFAULT_ALIGNMENT

host_scratch_zone_ops ops_for(fake_zone & z) {
    host_scratch_zone_ops ops;
    ops.user               = &z;
    ops.capacity           = fake_capacity;
    ops.grow               = fake_grow;
    ops.plan_scratch_bytes = plan_scratch;
    ops.chunk_cap          = chunk_cap;
    ops.alignment          = alignment;
    return ops;
}

void test_replace_not_add() {
    fake_zone z;
    z.capacity = plan_scratch;
    host_scratch_ledger ledger;
    const auto          ops = ops_for(z);

    capture_begin();
    // (100, 10) MiB: the target is the plan's SCRATCH plus 110 MiB, above the capacity, so one grow by the deficit.
    const auto r1 = ledger.notify_context(1, 100 * MiB, 10 * MiB, ops);
    HZP_CHECK(r1.ok && z.grows == 1, "the first notify grew %d time(s), expected 1", z.grows);
    HZP_CHECK(r1.target == plan_scratch + 110 * MiB, "target %.1f MiB, expected the plan's SCRATCH plus 110.0 MiB",
              (double) r1.target / MiB);
    // The request is the deficit plus one DEFAULT_ALIGNMENT per buffer at the chunk cap (one for 100 MiB, one for 10).
    HZP_CHECK(z.last_request == 110 * MiB + 2 * alignment,
              "grow request %zu B, expected the deficit %zu B plus 2 x %zu B", z.last_request, 110 * MiB, alignment);
    HZP_CHECK(log_has("[HOST-ARENA] SCRATCH demand ctx=1 "), "no SCRATCH demand line for ctx=1");

    // (300, 10) replaces it: the demand is 310 MiB, not 410, and the zone already holds plan + 310.
    const int before = z.grows;
    ledger.notify_context(1, 300 * MiB, 10 * MiB, ops);
    HZP_CHECK(ledger.context_demand(1) == 310 * MiB,
              "demand %.1f MiB after (100, 10) then (300, 10), expected 310.0 MiB",
              (double) ledger.context_demand(1) / MiB);
    HZP_CHECK(z.grows == before,
              "a replacing notify within capacity grew again: grow_zone called %d times, expected %d", z.grows, before);
}

void test_capacity_at_target_grows_nothing() {
    fake_zone z;
    z.capacity = plan_scratch + 110 * MiB;  // exactly the target
    host_scratch_ledger ledger;
    ledger.notify_context(1, 100 * MiB, 10 * MiB, ops_for(z));
    HZP_CHECK(z.grows == 0, "a notify whose target equals the capacity grew %d time(s)", z.grows);
}

void test_two_contexts_and_repeats() {
    fake_zone z;
    z.capacity = plan_scratch;
    host_scratch_ledger ledger;
    const auto          ops = ops_for(z);

    // Context A: 1000 MiB; target plan + 1000 > capacity: one grow (capacity becomes plan + 2048).
    ledger.notify_context(1, 1000 * MiB, 0, ops);
    HZP_CHECK(z.grows == 1, "context A's notify grew %d time(s), expected 1", z.grows);
    // Context B: 1500 MiB; the two sum to 2500 > 2048: one more grow, for the deficit.
    ledger.notify_context(2, 1500 * MiB, 0, ops);
    HZP_CHECK(z.grows == 2, "context B's notify grew the summed demand %d time(s) in all, expected 2", z.grows);
    HZP_CHECK(ledger.total() == 2500 * MiB, "two contexts sum to %.1f MiB, expected 2500.0 MiB",
              (double) ledger.total() / MiB);

    // The same notify repeated ten times, as sched_reserve's sites repeat it, grows nothing more.
    for (int i = 0; i < 10; ++i) {
        ledger.notify_context(2, 1500 * MiB, 0, ops);
    }
    HZP_CHECK(z.grows == 2, "ten repeats of one notify left grow_zone called %d times, expected 2", z.grows);

    // A context's demand dropped at its destructor does not shrink the zone, and a later context of the same size
    // grows nothing.
    const size_t cap = z.capacity;
    ledger.drop_context(1);
    HZP_CHECK(z.capacity == cap, "dropping a context's demand shrank the zone");
    ledger.notify_context(3, 1000 * MiB, 0, ops);
    HZP_CHECK(z.grows == 2, "a later context of the dropped one's size grew: grow_zone called %d times, expected 2",
              z.grows);
}

void test_one_demand_per_context() {
    // A llama_context with two SYCL backends notifies once, keyed by one backend context: its host buffers are one
    // demand however many SYCL backends it has.
    fake_zone z;
    z.capacity = plan_scratch;
    host_scratch_ledger ledger;
    const auto          ops         = ops_for(z);
    const uint64_t      backends[2] = { 11, 12 };
#if defined(HOST_SCRATCH_DEMAND_MUTANT_PER_BACKEND)
    // Positive control: one notify per SYCL backend, each its own demand.
    for (uint64_t key : backends) {
        ledger.notify_context(key, 1200 * MiB, 0, ops);
    }
#else
    ledger.notify_context(backends[0], 1200 * MiB, 0, ops);
#endif
    HZP_CHECK(ledger.total() == 1200 * MiB && z.grows == 1,
              "a two-backend context's demand is %.1f MiB with grow_zone called %d times, expected 1200.0 MiB and 1",
              (double) ledger.total() / MiB, z.grows);
}

void test_request_alignment_and_holder() {
    fake_zone z;
    z.capacity = plan_scratch;
    host_scratch_ledger ledger;
    const auto          ops = ops_for(z);
    // 600 MiB splits into 3 buffers at the 256 MiB cap, 10 MiB into 1: four alignments of slack.
    ledger.notify_context(1, 600 * MiB, 10 * MiB, ops);
    HZP_CHECK(z.last_request == 610 * MiB + 4 * alignment, "grow request %zu B, expected %zu B (deficit + 4 x %zu)",
              z.last_request, 610 * MiB + 4 * alignment, alignment);
    // A DEVICE holder's demand is on the same ledger.
    ledger.holder_demand("pinned_buffer_pool", 0, 5 * MiB, ops);
    HZP_CHECK(ledger.total() == 615 * MiB, "with a DEVICE holder the ledger holds %.1f MiB, expected 615.0 MiB",
              (double) ledger.total() / MiB);
}

// A zone whose grow lets one more notify in before its chunk commits: the notify a second thread makes while the first
// notify's grow runs. The ledger holds no lock across the grow, so this nested call is that interleaving.
struct interleaved_zone {
    fake_zone             zone;
    host_scratch_ledger * ledger = nullptr;
    host_scratch_zone_ops ops;
    bool                  pending        = false;  // a notify waits to arrive during the next grow
    uint64_t              pending_key    = 0;
    size_t                pending_demand = 0;
};

size_t interleaved_capacity(void * user) {
    return static_cast<interleaved_zone *>(user)->zone.capacity;
}

bool interleaved_grow(void * user, size_t request) {
    interleaved_zone & z = *static_cast<interleaved_zone *>(user);
    if (z.pending) {
        z.pending = false;
        z.ledger->notify_context(z.pending_key, z.pending_demand, 0, z.ops);
    }
    return fake_grow(&z.zone, request);
}

void test_interleaved_notify_grows_once() {
    // Two llama_contexts on one device. B's 500 MiB is already covered. A's 1000 MiB needs a grow, and while it runs,
    // B's reserve repeats its notify. Both see the one deficit (the target is the plan's SCRATCH plus 1500 MiB either
    // way), so one grow covers it.
    interleaved_zone    z;
    host_scratch_ledger ledger;
    z.zone.capacity = plan_scratch + 500 * MiB;
    z.ledger        = &ledger;
    z.ops           = ops_for(z.zone);
    z.ops.user      = &z;
    z.ops.capacity  = interleaved_capacity;
    z.ops.grow      = interleaved_grow;
    ledger.notify_context(2, 500 * MiB, 0, z.ops);
    HZP_CHECK(z.zone.grows == 0, "B's covered notify grew %d time(s), expected 0", z.zone.grows);
    z.pending        = true;
    z.pending_key    = 2;
    z.pending_demand = 500 * MiB;
    ledger.notify_context(1, 1000 * MiB, 0, z.ops);
    HZP_CHECK(!z.pending, "B's notify did not arrive during A's grow");
    HZP_CHECK(z.zone.grows == 1, "two notifies for one deficit grew %d time(s), expected 1", z.zone.grows);
}

void test_failed_grow_returns_its_reservation() {
    // The deficit is reserved before the grow runs. A grow that fails gives the reservation back, so the next notify
    // for the same deficit grows again instead of finding it covered.
    fake_zone z;
    z.capacity = plan_scratch;
    z.fail     = true;
    host_scratch_ledger ledger;
    const auto          ops = ops_for(z);
    const auto          r1  = ledger.notify_context(1, 1000 * MiB, 0, ops);
    HZP_CHECK(!r1.ok && z.grows == 0, "the failing grow returned ok=%d after %d grow(s)", (int) r1.ok, z.grows);
    z.fail        = false;
    const auto r2 = ledger.notify_context(1, 1000 * MiB, 0, ops);
    HZP_CHECK(r2.ok && z.grows == 1,
              "after a failed grow, the next notify for the same deficit grew %d time(s), expected 1", z.grows);
}

// ---------------------------------------------------------------------------------------------------------------
// The notify's core over two fake caches.
// ---------------------------------------------------------------------------------------------------------------

// One ledger per zone, made fresh with it: a ledger's reservations describe its zone, whose capacity never falls.
fake_zone                            g_zone[2];
std::unique_ptr<host_scratch_ledger> g_ledger[2];

int                  g_buft_device   = 0;
int                  g_context_calls = 0;
char                 g_backend_storage[2];
const ggml_backend_t g_sycl_dev1 = reinterpret_cast<ggml_backend_t>(&g_backend_storage[0]);
const ggml_backend_t g_not_sycl  = reinterpret_cast<ggml_backend_t>(&g_backend_storage[1]);

bool fake_is_sycl(ggml_backend_t b) {
    return b == g_sycl_dev1;
}

int fake_host_buft_device(ggml_backend_t) {
    return g_buft_device;
}

host_scratch_context_info fake_context_of(ggml_backend_t) {
    ++g_context_calls;
    return { 42, 1 };  // the notifying backend context is on device 1
}

bool fake_ledger_of(int device, host_scratch_ledger ** ledger, host_scratch_zone_ops * ops) {
    if (device < 0 || device > 1) {
        return false;
    }
    *ledger = g_ledger[device].get();
    *ops    = ops_for(g_zone[device]);
    return true;
}

host_scratch_notify_env fake_env() {
    host_scratch_notify_env env;
    env.is_sycl          = fake_is_sycl;
    env.host_buft_device = fake_host_buft_device;
    env.context_of       = fake_context_of;
    env.ledger_of        = fake_ledger_of;
    return env;
}

void reset_caches() {
    for (int d = 0; d < 2; ++d) {
        g_zone[d]          = fake_zone{};
        g_zone[d].capacity = plan_scratch;
        g_ledger[d]        = std::make_unique<host_scratch_ledger>();
    }
    g_context_calls = 0;
}

void test_non_sycl_refused() {
    reset_caches();
    capture_begin();
    const bool ok = host_scratch_notify(fake_env(), g_not_sycl, 100 * MiB, 0);
    HZP_CHECK(!ok && log_has("[HOST-ARENA] SCRATCH notify refused: the backend is not a SYCL backend"),
              "a non-SYCL backend's notify was not refused with the WARN");
    HZP_CHECK(g_context_calls == 0 && g_ledger[0]->total() == 0 && g_ledger[1]->total() == 0,
              "a refused notify read the backend's context or recorded a demand");
}

void test_device_is_the_host_bufts() {
    // The fake host buft allocates from device 0: a notify through a device-1 backend grows device 0's SCRATCH.
    reset_caches();
    g_buft_device = 0;
    host_scratch_notify(fake_env(), g_sycl_dev1, 100 * MiB, 10 * MiB);
    HZP_CHECK(g_zone[0].grows == 1 && g_zone[1].grows == 0,
              "with the host buft on device 0, a device-1 backend's notify grew device 0 %d time(s) and device 1 %d",
              g_zone[0].grows, g_zone[1].grows);
    // With the buft's device set to 1, the same notify grows device 1's.
    reset_caches();
    g_buft_device = 1;
    host_scratch_notify(fake_env(), g_sycl_dev1, 100 * MiB, 10 * MiB);
    HZP_CHECK(g_zone[1].grows == 1 && g_zone[0].grows == 0,
              "with the host buft on device 1, the notify grew device 1 %d time(s) and device 0 %d", g_zone[1].grows,
              g_zone[0].grows);
    g_buft_device = 0;
}

void test_failed_grow_returns_false() {
    reset_caches();
    g_zone[0].fail = true;
    const bool ok  = host_scratch_notify(fake_env(), g_sycl_dev1, 100 * MiB, 0);
    HZP_CHECK(!ok, "a notify whose grow failed returned true");
}

}  // namespace

int main() {
    test_replace_not_add();
    test_capacity_at_target_grows_nothing();
    test_two_contexts_and_repeats();
    test_one_demand_per_context();
    test_request_alignment_and_holder();
    test_interleaved_notify_grows_once();
    test_failed_grow_returns_its_reservation();
    test_non_sycl_refused();
    test_device_is_the_host_bufts();
    test_failed_grow_returns_false();
    return finish("test-sycl-host-scratch-demand");
}
