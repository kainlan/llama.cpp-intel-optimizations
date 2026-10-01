// Host tests of the KV region registry, scope and guard (llama.cpp-moua L3c;
// H8 and the registry-level subset of H9).
//
// What this proves, and what it does not.  The header models the lock order, the
// thread_local scope, the residency answer, the two-phase guard and the release
// proc over opaque handles.  H8's registry arms (the scope, the residency answer
// under and outside a scope, a mixed device set, publish A / publish B / create
// A) and H9's lock-discipline arms that need no device are scored here.  The arms
// that need a lifecycle body, a ledger, MMID or the republish allowlist are L4+
// work and are not scored here.
//
// Every arm that asserts an absence (no violation, no drop under a lock, no
// block) is paired with a positive control that provokes the very thing it looks
// for, so a zero is a measurement.  The mutations that were run against this file
// are listed with each arm.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../kv-region-registry.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_EQ(got, want, msg)                                                                                       \
    do {                                                                                                               \
        const long long got_v_  = (long long) (got);                                                                   \
        const long long want_v_ = (long long) (want);                                                                  \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, got_v_, want_v_); \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using namespace ggml_sycl;

namespace {

// ---- the abort channel ----------------------------------------------------
std::mutex               g_abort_mutex;
std::vector<std::string> g_aborts;

void record_abort(const char * message) {
    std::lock_guard<std::mutex> g(g_abort_mutex);
    g_aborts.push_back(message);
}

size_t abort_count() {
    std::lock_guard<std::mutex> g(g_abort_mutex);
    return g_aborts.size();
}

std::string last_abort() {
    std::lock_guard<std::mutex> g(g_abort_mutex);
    return g_aborts.empty() ? std::string() : g_aborts.back();
}

struct abort_capture {
    abort_capture() {
        {
            std::lock_guard<std::mutex> g(g_abort_mutex);
            g_aborts.clear();
        }
        kv_region_abort_handler().store(&record_abort);
        kv_lock_witness::reset();
    }

    ~abort_capture() { kv_region_abort_handler().store(nullptr); }
};

// ---- handles that record where they were dropped --------------------------
struct drop_log {
    std::atomic<int> drops{ 0 };
    std::atomic<int> drops_with_lock{ 0 };
};

kv_region_handle make_handle(drop_log & log) {
    return kv_region_handle(new int(0), [&log](void * p) {
        log.drops.fetch_add(1);
        if (kv_lock_witness::held_count() != 0) {
            log.drops_with_lock.fetch_add(1);
        }
        delete static_cast<int *>(p);
    });
}

kv_region_entry make_entry(drop_log & log, std::initializer_list<uint32_t> layers, uint64_t shape_key = 1) {
    kv_region_entry e;
    e.extents.push_back(make_handle(log));
    e.shape_key = shape_key;
    for (uint32_t l : layers) {
        kv_layer_slice s;
        s.extent      = 0;
        s.slot_offset = 4096ull * l;
        s.slot_size   = 4096;
        e.layout[l]   = s;
    }
    return e;
}

// ---- H8: the scope is thread_local ----------------------------------------
int case_scope_is_thread_local() {
    abort_capture cap;
    kv_region_scope::begin(7);
    kv_context_id id = 0;
    CHECK(kv_region_scope::active(&id) && id == 7, "scope open on the opening thread");

    bool        other_active    = true;
    bool        other_nested_ok = false;
    std::thread t([&] {
        other_active = kv_region_scope::active();
        kv_region_scope::begin(9);  // not nested: this thread has none open
        kv_context_id oid = 0;
        other_nested_ok   = kv_region_scope::active(&oid) && oid == 9;
        kv_region_scope::end();
    });
    t.join();
    CHECK(!other_active, "a scope open on one thread is invisible to another");
    CHECK(other_nested_ok, "another thread opens its own scope without a nested-begin abort");
    CHECK(kv_region_scope::active(&id) && id == 7, "the other thread's scope did not disturb ours");
    CHECK_EQ(abort_count(), 0, "no abort for concurrent scopes on two threads");
    kv_region_scope::end();
    CHECK(!kv_region_scope::active(), "end() closes the scope");

    // Positive control: a process-global scope (the shape the thread_local
    // replaces) IS visible from another thread, so the probe above could fail.
    static std::atomic<bool> global_open{ false };
    global_open.store(true);
    bool        seen_by_other = false;
    std::thread t2([&] { seen_by_other = global_open.load(); });
    t2.join();
    global_open.store(false);
    CHECK(seen_by_other, "control: a global scope is seen by another thread");
    return 0;
}

int case_scope_nested_begin_aborts() {
    abort_capture cap;
    kv_region_scope::begin(11);
    kv_region_scope::begin(22);
    CHECK_EQ(abort_count(), 1, "a nested begin aborts once");
    const std::string m = last_abort();
    CHECK(m.find("context 22") != std::string::npos && m.find("context 11") != std::string::npos,
          "the abort names both context ids");
    kv_context_id id = 0;
    CHECK(kv_region_scope::active(&id) && id == 11, "the outer scope is unchanged by the refused nested begin");
    kv_region_scope::end();

    // The guard closes the scope on a throw (llama_context's create_memory can throw).
    try {
        kv_region_scope::guard g(5);
        CHECK(kv_region_scope::active(), "guard opens the scope");
        throw 1;
    } catch (int) {
    }
    CHECK(!kv_region_scope::active(), "the guard closed the scope on the throw");
    return 0;
}

// ---- H8: residency answers --------------------------------------------------
int case_residency_answer() {
    abort_capture      cap;
    drop_log           log;
    kv_region_registry reg_a;  // context A's device
    kv_region_entry    discard   = reg_a.publish(1, make_entry(log, { 0, 1, 2 }));
    kv_region_entry    discard_b = reg_a.publish(2, make_entry(log, { 1, 3 }));

    int  plan_calls    = 0;
    auto plan_all_host = [&](uint32_t) {
        ++plan_calls;
        return false;
    };
    auto plan_all_device = [&](uint32_t) {
        ++plan_calls;
        return true;
    };

    {
        kv_region_scope::guard g(1);
        CHECK(kv_region_layer_on_device(&reg_a, 0, plan_all_host),
              "under scope A, a slotted layer is resident though the plan says host");
        CHECK(kv_region_layer_on_device(&reg_a, 2, plan_all_host), "under scope A, layer 2 is resident");
        CHECK(!kv_region_layer_on_device(&reg_a, 3, plan_all_device),
              "under scope A, a layer with no slot is host though the plan says device");
        CHECK_EQ(plan_calls, 0, "the plan is not consulted for a device that reserves regions under a scope");

        // A device that reserves no regions answers from the plan (mixed set).
        CHECK(kv_region_layer_on_device(nullptr, 3, plan_all_device), "no registry: the plan answers (device)");
        CHECK(!kv_region_layer_on_device(nullptr, 0, plan_all_host), "no registry: the plan answers (host)");
        CHECK_EQ(plan_calls, 2, "the plan was consulted for the regionless device");
    }

    // No scope: the plan answers, even for a device that has a registry.
    plan_calls = 0;
    CHECK(!kv_region_layer_on_device(&reg_a, 0, plan_all_host), "outside a scope the plan answers (host)");
    CHECK(kv_region_layer_on_device(&reg_a, 3, plan_all_device), "outside a scope the plan answers (device)");
    CHECK_EQ(plan_calls, 2, "outside a scope the plan is consulted every time");

    // Publish A, publish B, create A with the shared plan mutated in between: A's
    // mask is A's slot table, not the plan B mutated it to.
    bool plan_is_b   = false;
    auto shared_plan = [&](uint32_t il) {
        const bool a = (il == 0 || il == 1 || il == 2);
        const bool b = (il == 1 || il == 3);
        return plan_is_b ? b : a;
    };
    plan_is_b = true;  // B published after A and mutated the shared plan
    {
        kv_region_scope::guard g(1);
        bool                   mask[4];
        for (uint32_t il = 0; il < 4; ++il) {
            mask[il] = kv_region_layer_on_device(&reg_a, il, shared_plan);
        }
        CHECK(mask[0] && mask[1] && mask[2] && !mask[3], "A's mask equals A's slot table after B mutated the plan");
        // Control: the plan-only answer for the same layers is B's, which differs.
        CHECK(!shared_plan(0) && shared_plan(3), "control: the mutated plan alone would have answered B's residency");
    }
    {
        kv_region_scope::guard g(2);
        CHECK(!kv_region_layer_on_device(&reg_a, 0, shared_plan) && kv_region_layer_on_device(&reg_a, 3, shared_plan),
              "scope B reads B's slots from the same registry");
    }
    return 0;
}

// ---- H8: attach ---------------------------------------------------------------
int case_attach_reads_the_owner_registry() {
    abort_capture      cap;
    drop_log           log;
    kv_region_registry dev0;
    kv_region_registry dev1;
    // Context 1: layers 0,1 on device 0's region, layers 2,3 on device 1's.  The
    // tiered buffer is built on device 0, yet takes slots for 2,3 from dev1.
    kv_region_entry    d0 = dev0.publish(1, make_entry(log, { 0, 1 }));
    kv_region_entry    d1 = dev1.publish(1, make_entry(log, { 2, 3 }));
    // A different context holds a conflicting layer 2 on device 0: must not be read.
    kv_region_entry    d2 = dev0.publish(2, make_entry(log, { 2 }));

    std::vector<const kv_region_registry *> regs = { &dev0, &dev1 };
    std::vector<kv_layer_plan>              plan = {
        { 0, 0 },
        { 1, 0 },
        { 2, 1 },
        { 3, 1 }
    };

    kv_attach_result none = kv_region_attach_buffer(regs, plan);
    CHECK(none.no_scope && !none.ok, "no scope: an arena device with device-planned layers is a contract violation");

    {
        kv_region_scope::guard g(1);
        kv_attach_result       r = kv_region_attach_buffer(regs, plan);
        CHECK(r.ok && r.missing.empty(), "every device-planned layer has a slot");
        CHECK_EQ(r.slots.size(), 4, "four slots");
        CHECK(r.slots[2].owner_device == 1 && r.slots[2].slice.slot_offset == 4096ull * 2,
              "layer 2 comes from the owner device's registry");
        CHECK(r.slots[2].context == 1, "slots carry the scope's context id");

        // iSWA: a second buffer attaches to the same region, with no pop and no count.
        kv_attach_result r2 = kv_region_attach_buffer(regs, plan);
        CHECK(r2.ok && r2.slots.size() == 4, "a second attach in the scope finds the same slots");
        for (size_t i = 0; i < r.slots.size(); ++i) {
            CHECK(r.slots[i].slice.slot_offset == r2.slots[i].slice.slot_offset, "second attach sees the same offsets");
        }

        // A layer the owner's region has no slot for is reported by index.
        std::vector<kv_layer_plan> wider = plan;
        wider.push_back({ 7, 1 });
        kv_attach_result r3 = kv_region_attach_buffer(regs, wider);
        CHECK(!r3.ok && r3.missing.size() == 1 && r3.missing[0] == 7, "a missing slot is named");

        // A layer whose owner has no registry is missing too (not read from another device).
        std::vector<kv_layer_plan> no_reg = {
            { 2, 2 }
        };
        kv_attach_result r4 = kv_region_attach_buffer(regs, no_reg);
        CHECK(!r4.ok && r4.missing.size() == 1, "an owner without a registry yields a missing slot");

        // Control: reading the wrong (buffer) device for layer 2 finds nothing.
        kv_layer_slice s;
        CHECK(!dev0.layer_slice(1, 2, s), "control: layer 2 is not in device 0's region for context 1");
    }

    // The attach seam: a thread parked mid-attach does not block another thread's scope.
    std::atomic<bool> in_attach{ false };
    std::atomic<bool> release{ false };
    std::atomic<bool> second_done{ false };
    std::thread       a([&] {
        kv_region_scope::guard g(1);
        kv_region_attach_buffer(regs, plan, [&] {
            in_attach.store(true);
            while (!release.load()) {
                std::this_thread::yield();
            }
        });
    });
    while (!in_attach.load()) {
        std::this_thread::yield();
    }
    std::thread b([&] {
        kv_region_scope::guard g(2);
        kv_attach_result       r = kv_region_attach_buffer(regs, {
                                                               { 2, 0 }
        });
        second_done.store(r.ok && r.context == 2);
    });
    b.join();
    CHECK(second_done.load(), "another context's attach completes while a first is parked mid-attach");
    release.store(true);
    a.join();
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violations in the attach arms");
    return 0;
}

// ---- H9: the lock witness, with positive controls --------------------------
int case_witness_positive_controls() {
    abort_capture      cap;
    kv_witnessed_mutex l1{ KV_LOCK_L1_INVENTORY, "L1" };
    kv_witnessed_mutex l3{ KV_LOCK_L3_REGISTRY, "kv_region_mutex_" };
    kv_witnessed_mutex l4{ KV_LOCK_L4_SLOT_STATE, "L4" };
    kv_witnessed_mutex l5{ KV_LOCK_L5_GROUP, "L5" };

    {
        std::lock_guard<kv_witnessed_mutex> a(l1);
        std::lock_guard<kv_witnessed_mutex> b(l4);
        std::lock_guard<kv_witnessed_mutex> c(l5);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "L1, L4, L5 in rank order is clean");

    {
        std::lock_guard<kv_witnessed_mutex> a(l3);
        std::lock_guard<kv_witnessed_mutex> b(l5);  // anything under the leaf is a violation
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: a lock taken under the registry lock is flagged");
    CHECK(kv_lock_witness::last().find("leaf") != std::string::npos, "the leaf violation says so");

    kv_lock_witness::reset();
    {
        std::lock_guard<kv_witnessed_mutex> a(l5);
        std::lock_guard<kv_witnessed_mutex> b(l1);
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: L1 after L5 is an order violation");
    CHECK(kv_lock_witness::held_count() == 0, "the witness stack is empty after the scopes");

    // A leaf lock after the registry released is clean (never co-held).
    kv_lock_witness::reset();
    {
        std::lock_guard<kv_witnessed_mutex> a(l3);
    }
    {
        std::lock_guard<kv_witnessed_mutex> a(l5);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "registry then group, one at a time, is clean");

    // The drop-under-lock detector fires when it should.
    drop_log log;
    {
        kv_region_handle                    h = make_handle(log);
        std::lock_guard<kv_witnessed_mutex> a(l5);
        h.reset();
    }
    CHECK_EQ(log.drops_with_lock.load(), 1, "control: a final drop under a lock is detected");
    return 0;
}

// ---- registry: copy-in, copy-out, drops outside the lock -------------------
int case_registry_drops_outside_the_lock() {
    abort_capture      cap;
    drop_log           log;
    kv_region_registry reg;

    kv_region_entry prev = reg.publish(1, make_entry(log, { 0 }));
    CHECK(prev.empty(), "first publish replaces nothing");
    CHECK_EQ(reg.size(), 1, "one entry");

    // Republish: the replaced entry comes back, undropped.
    prev = reg.publish(1, make_entry(log, { 0, 1 }, 2));
    CHECK(!prev.empty() && prev.shape_key == 1, "the replaced entry is handed back");
    CHECK_EQ(log.drops.load(), 0, "nothing dropped yet: the caller owns the old extents");
    prev = kv_region_entry();
    CHECK_EQ(log.drops.load(), 1, "dropping the returned entry frees the old extent");
    CHECK_EQ(log.drops_with_lock.load(), 0, "and with no lock held");

    // lookup copies handles out: a copy keeps the extent alive past an erase.
    kv_region_entry copy;
    CHECK(reg.lookup(1, copy) && copy.layout.size() == 2, "lookup copies the entry");
    kv_region_entry taken;
    CHECK(reg.take(1, taken), "take moves the entry out");
    CHECK(!reg.lookup(1, copy), "a lookup of an erased context fails");
    CHECK_EQ(reg.size(), 0, "the registry is empty after the take");
    taken = kv_region_entry();
    CHECK_EQ(log.drops.load(), 1, "the copy still holds the extent");
    copy = kv_region_entry();
    CHECK_EQ(log.drops.load(), 2, "the last copy's drop frees it");
    CHECK_EQ(log.drops_with_lock.load(), 0, "no drop ran under a lock");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation");

    // lookup into a populated `out` drops its old contents after the unlock.
    kv_region_entry e1     = reg.publish(2, make_entry(log, { 0 }));
    kv_region_entry out    = make_entry(log, { 5 });
    const int       before = log.drops.load();
    CHECK(reg.lookup(2, out), "lookup replaces out");
    CHECK_EQ(log.drops.load(), before + 1, "out's old extent dropped");
    CHECK_EQ(log.drops_with_lock.load(), 0, "...outside the registry lock");
    return 0;
}

int case_registry_concurrent_churn() {
    abort_capture            cap;
    drop_log                 log;
    kv_region_registry       reg;
    std::atomic<int>         errors{ 0 };
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 400; ++i) {
                const kv_context_id id   = (kv_context_id) (t % 3);  // contended ids
                kv_region_entry     prev = reg.publish(id, make_entry(log, { 0, 1 }, (uint64_t) i));
                prev                     = kv_region_entry();
                kv_region_entry c;
                reg.lookup(id, c);
                if (i % 3 == 0) {
                    kv_region_entry tk;
                    reg.take(id, tk);
                }
                kv_layer_slice s;
                reg.layer_slice(id, 1, s);
            }
        });
    }
    for (std::thread & th : threads) {
        th.join();
    }
    for (kv_context_id id = 0; id < 3; ++id) {
        kv_region_entry tk;
        reg.take(id, tk);
    }
    CHECK_EQ(errors.load(), 0, "no errors");
    CHECK_EQ(log.drops_with_lock.load(), 0, "no drop under a lock under contention");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation under contention");
    CHECK_EQ(log.drops.load(), 8 * 400, "every published extent was freed exactly once");
    return 0;
}

// ---- L0: the outermost-only re-plan token -----------------------------------
int case_replan_token_nesting() {
    abort_capture   cap;
    kv_replan_token l0;
    {
        kv_replan_scope outer(l0);
        kv_replan_scope inner(l0);
        kv_replan_scope innermost(l0);
        CHECK_EQ(l0.lock_count(), 1, "nested holds on one thread do not lock again");
        CHECK(l0.held_by_this_thread(), "held");
        bool        free_from_other = true;
        std::thread t([&] { free_from_other = l0.probe_free(); });
        t.join();
        CHECK(!free_from_other, "control: another thread cannot take a held token");
    }
    CHECK_EQ(l0.unlock_count(), 1, "the outermost release unlocks once");
    CHECK(!l0.held_by_this_thread(), "released");
    CHECK(l0.probe_free(), "free after the outermost release");
    CHECK_EQ(kv_lock_witness::violations(), 0, "nesting is not a violation");

    // acquire_outermost refuses a re-entry (the release proc's rule).
    {
        kv_replan_scope held(l0);
        l0.acquire_outermost();
        CHECK_EQ(abort_count(), 1, "outermost-only acquire aborts when already held");
        CHECK(last_abort().find("re-plan token") != std::string::npos, "and names the token");
        CHECK_EQ(l0.lock_count(), 2, "it did not lock again");
    }
    return 0;
}

// ---- the release proc -----------------------------------------------------
int case_release_proc() {
    abort_capture      cap;
    drop_log           log;
    kv_replan_token    l0;
    kv_region_registry dev0;
    kv_region_registry dev1;

    auto tenants = std::make_shared<kv_tenant_slots>();
    tenants->add("c", 0, make_handle(log), 1024);
    kv_region_entry e0 = make_entry(log, { 0 });
    e0.tenants         = tenants;
    kv_region_entry p0 = dev0.publish(1, std::move(e0));
    kv_region_entry p1 = dev1.publish(1, make_entry(log, { 1 }));
    kv_region_entry p2 = dev0.publish(2, make_entry(log, { 0 }));  // another context: untouched
    tenants.reset();

    std::vector<kv_region_registry *> regs             = { &dev0, &dev1 };
    size_t                            retained_handles = 0;
    bool                              retain_with_lock = false;
    auto                              retain           = [&](std::vector<kv_region_handle> && v) {
        retained_handles += v.size();
        retain_with_lock = retain_with_lock || kv_lock_witness::held_count() != 0;
        v.clear();  // the "event" has passed
    };

    kv_region_release(regs, 1, l0, retain);
    CHECK_EQ(l0.lock_count(), 1, "the release proc takes L0 once");
    CHECK_EQ(l0.unlock_count(), 1, "and releases it");
    CHECK(!dev0.has_layer(1, 0) && !dev1.has_layer(1, 1), "context 1 is gone from every device");
    CHECK(dev0.has_layer(2, 0), "another context's entry is untouched");
    CHECK_EQ(retained_handles, 3, "two extents and one tenant slot go to retain_until_event");
    CHECK(!retain_with_lock, "retain ran with no lock held");
    // The extents arrive at retain_until_event as handles (count above), so the
    // fencing event, not the release proc, decides when the last generation frees.
    CHECK_EQ(log.drops.load(), 3, "everything but context 2's extent was freed");
    CHECK_EQ(log.drops_with_lock.load(), 0, "no drop under a lock");

    // Idempotent: a second call finds nothing, aborts nothing.
    const size_t aborts_before   = abort_count();
    const size_t retained_before = retained_handles;
    kv_region_release(regs, 1, l0, retain);
    CHECK_EQ(abort_count(), aborts_before, "a second release does not abort");
    CHECK_EQ(retained_handles, retained_before, "and retains nothing");

    // It really takes L0: a parked holder blocks it until the holder lets go.
    kv_region_entry   again = dev0.publish(3, make_entry(log, { 0 }));
    std::atomic<bool> parked{ false };
    std::atomic<bool> let_go{ false };
    std::thread       holder([&] {
        kv_replan_scope s(l0);
        parked.store(true);
        while (!let_go.load()) {
            std::this_thread::yield();
        }
    });
    while (!parked.load()) {
        std::this_thread::yield();
    }
    std::atomic<bool> finished{ false };
    std::thread       releaser([&] {
        kv_region_release(regs, 3, l0, retain);
        finished.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CHECK(!finished.load(), "the release proc waits on a parked L0 holder");
    CHECK(dev0.has_layer(3, 0), "and has removed nothing meanwhile");
    let_go.store(true);
    holder.join();
    releaser.join();
    CHECK(finished.load() && !dev0.has_layer(3, 0), "it completes once L0 is free");

    // Entered under a held token: the debug abort, and nothing is torn down.
    kv_region_entry kept = dev0.publish(4, make_entry(log, { 0 }));
    {
        kv_replan_scope held(l0);
        kv_region_release(regs, 4, l0, retain);
    }
    CHECK(last_abort().find("re-plan token") != std::string::npos, "release under a held token aborts by name");
    CHECK(dev0.has_layer(4, 0), "and leaves the entry alone");

    // Under L1 the order is flagged: the proc takes L0, never L1.
    kv_witnessed_mutex l1{ KV_LOCK_L1_INVENTORY, "L1" };
    kv_lock_witness::reset();
    {
        std::lock_guard<kv_witnessed_mutex> g(l1);
        kv_region_release(regs, 4, l0, retain);
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: calling the release proc under L1 is an order violation");
    kv_lock_witness::reset();
    kv_region_entry kept2 = dev0.publish(5, make_entry(log, { 0 }));
    kv_region_release(regs, 5, l0, retain);
    CHECK_EQ(kv_lock_witness::violations(), 0, "called from a clean thread it violates nothing");

    // A lock failure aborts by name and does not propagate out of the noexcept proc.
    kv_region_entry kept3 = dev0.publish(6, make_entry(log, { 0 }));
    dev0.fail_next_lock();
    const size_t before = abort_count();
    kv_region_release(regs, 6, l0, retain);
    CHECK_EQ(abort_count(), before + 1, "a lock failure aborts once");
    CHECK(last_abort().find("kv_region_mutex_") != std::string::npos, "naming the registry lock");
    CHECK(l0.probe_free(), "and L0 was released on the failure path");
    return 0;
}

// ---- the transaction guard ---------------------------------------------------
int case_txn_guard_two_phases() {
    abort_capture      cap;
    drop_log           log;
    kv_witnessed_mutex group{ KV_LOCK_L5_GROUP, "group mutex" };

    // Abandon: phase 1 under the group mutex alone, phase 2 with no lock held.
    int    clears              = 0;
    bool   group_held_in_clear = false;
    size_t held_in_clear       = 0;
    {
        kv_txn_guard g;
        g.note_ranges(&group, [&] {
            ++clears;
            group_held_in_clear = kv_lock_witness::holds(KV_LOCK_L5_GROUP);
            held_in_clear       = kv_lock_witness::held_count();
        });
        g.note_handle(make_handle(log));
        g.note_handle(make_handle(log));
        g.note_unused_control(make_handle(log));
        CHECK_EQ(g.handles(), 3, "the guard owns three handles");
        CHECK_EQ(log.drops.load(), 0, "nothing is freed while the guard is armed");
    }
    CHECK_EQ(clears, 1, "the abandoned guard cleared its ranges");
    CHECK(group_held_in_clear && held_in_clear == 1, "phase 1 held the group mutex and nothing else");
    CHECK_EQ(log.drops.load(), 3, "phase 2 dropped every handle");
    CHECK_EQ(log.drops_with_lock.load(), 0, "with no lock held");
    CHECK_EQ(group.lock_count(), 1, "one group lock");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no violation");

    // Commit: no clear, and the handle published to the registry survives the guard.
    kv_region_registry reg;
    int                clears2 = 0;
    {
        kv_txn_guard g;
        g.note_ranges(&group, [&] { ++clears2; });
        kv_region_entry e = make_entry(log, { 0 });
        g.note_handle(e.extents[0]);
        kv_region_entry prev = reg.publish(1, std::move(e));
        g.commit();
        CHECK(!g.armed(), "commit disarms");
    }
    CHECK_EQ(clears2, 0, "a committed guard clears no ranges");
    CHECK(reg.has_layer(1, 0), "the published entry stays");
    CHECK_EQ(log.drops.load(), 3, "the extent the guard also referenced is still alive in the registry");

    // Exception path: the guard abandons as the stack unwinds.
    int clears3 = 0;
    try {
        kv_txn_guard g;
        g.note_ranges(&group, [&] { ++clears3; });
        g.note_handle(make_handle(log));
        throw 1;
    } catch (int) {
    }
    CHECK_EQ(clears3, 1, "an exception unwinds through the guard's rollback");
    CHECK_EQ(log.drops.load(), 4, "and drops its handle");

    // Control: phase 1 under L1 is flagged (the guard is declared before L1's lock).
    kv_witnessed_mutex l1{ KV_LOCK_L1_INVENTORY, "L1" };
    kv_lock_witness::reset();
    {
        std::lock_guard<kv_witnessed_mutex> hold(l1);
        kv_txn_guard                        g;
        g.note_handle(make_handle(log));
        g.abandon();
    }
    CHECK(kv_lock_witness::violations() >= 1, "control: rolling back with L1 held is flagged");

    // The guard declared BEFORE the lock runs after it is released (reverse order).
    kv_lock_witness::reset();
    {
        kv_txn_guard g;
        g.note_handle(make_handle(log));
        std::lock_guard<kv_witnessed_mutex> hold(l1);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "a guard declared before L1 rolls back after L1 is released");

    // A group-mutex failure aborts by name, without propagating out of the destructor.
    kv_witnessed_mutex fragile{ KV_LOCK_L5_GROUP, "fragile group" };
    fragile.fail_next_lock();
    const size_t before = abort_count();
    {
        kv_txn_guard g;
        g.note_ranges(&fragile, [] {});
        g.note_handle(make_handle(log));
    }
    CHECK_EQ(abort_count(), before + 1, "a rollback that cannot take the group mutex aborts");
    CHECK(last_abort().find("group mutex") != std::string::npos, "naming it");
    return 0;
}

// ---- tenant slots and the weight-slot store ---------------------------------------
int case_tenant_slots_and_weight_slots() {
    abort_capture   cap;
    drop_log        log;
    kv_tenant_slots slots;
    slots.add("rows", 0, make_handle(log), 1000);
    slots.add("rows", 1, make_handle(log), 500);
    slots.add("per-op", 0, make_handle(log), 64);  // a DEVICE-scope slot: its own key

    CHECK(slots.claim("rows", 0, 1000) == kv_claim_result::OK, "claim at the cap is OK");
    CHECK(slots.claim("rows", 0, 10) == kv_claim_result::ALREADY_CLAIMED, "a second claim of the slot is refused");
    CHECK(slots.claim("rows", 1, 501) == kv_claim_result::OVER_PLAN, "a claim above the cap is OVER_PLAN");
    CHECK(slots.claim("rows", 9, 1) == kv_claim_result::NO_SLOT, "a claim with no slot is NO_SLOT");
    CHECK(slots.claim("per-op", 0, 64) == kv_claim_result::OK, "the per-op slot is independent of rows/0");
    CHECK(slots.claimed("per-op", 0) && !slots.claimed("rows", 1), "claim state is per (cohort, index)");

    std::vector<kv_region_handle> out;
    CHECK(!slots.take_unclaimed(out) && out.empty(), "slots with a live claim are not taken, and nothing moves");
    CHECK_EQ(slots.size(), 3, "all three slots remain");
    slots.release_claim("rows", 0);
    slots.release_claim("per-op", 0);
    CHECK(!slots.any_claimed(), "no claim is live");
    CHECK(slots.take_unclaimed(out) && out.size() == 3, "the unclaimed slots move out");
    CHECK_EQ(log.drops.load(), 0, "moved out, not dropped: the caller drops with no lock held");
    out.clear();
    CHECK_EQ(log.drops.load(), 3, "dropped by the caller");
    CHECK_EQ(log.drops_with_lock.load(), 0, "outside every lock");

    // add() over a live key drops the old handle after the lock.
    kv_tenant_slots again;
    again.add("rows", 0, make_handle(log), 10);
    const int before = log.drops.load();
    again.add("rows", 0, make_handle(log), 20);
    CHECK_EQ(log.drops.load(), before + 1, "a replaced slot's handle is dropped");
    CHECK_EQ(log.drops_with_lock.load(), 0, "outside the slot-state lock");
    CHECK_EQ(again.cap("rows", 0), 20, "and the cap is the new one");

    // The weight slot: one per (model, device), shared by contexts of a model.
    kv_weight_slot_store store;
    CHECK(store.record(1, 0, make_handle(log), 111), "record the model-1 slot on device 0");
    CHECK(!store.record(1, 0, make_handle(log), 222), "no setter: a second record for the key is refused");
    CHECK(store.record(1, 1, make_handle(log), 333), "the same model on device 1 is its own key");
    CHECK(store.record(2, 0, make_handle(log), 444), "another model on device 0 is its own key");
    size_t b = 0;
    CHECK(store.find(1, 0, &b) && b == 111, "the first record stands");
    CHECK(store.find(1, 1, &b) && b == 333, "device 1's slot is distinct");
    CHECK(store.find(2, 0, &b) && b == 444, "model 2's slot is distinct");
    CHECK(!store.find(2, 1), "no slot recorded for (2, 1)");
    kv_region_handle shared_a = store.handle(1, 0);
    kv_region_handle shared_b = store.handle(1, 0);
    CHECK(shared_a && shared_a == shared_b, "two contexts of one model on one device share its slot");
    const int        drops0 = log.drops.load();
    kv_region_handle taken  = store.take(1, 0);
    CHECK(!store.find(1, 0), "taken");
    taken.reset();
    shared_a.reset();
    CHECK_EQ(log.drops.load(), drops0, "a claim's copy keeps the slot alive past the store's erase");
    shared_b.reset();
    CHECK_EQ(log.drops.load(), drops0 + 1, "the last holder's drop frees it");
    CHECK_EQ(log.drops_with_lock.load(), 0, "never under a lock");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation");
    return 0;
}

}  // namespace

int main() {
    struct test_case {
        const char * name;
        int (*fn)();
    };

    const test_case cases[] = {
        { "scope_is_thread_local",           case_scope_is_thread_local           },
        { "scope_nested_begin_aborts",       case_scope_nested_begin_aborts       },
        { "residency_answer",                case_residency_answer                },
        { "attach_reads_the_owner_registry", case_attach_reads_the_owner_registry },
        { "witness_positive_controls",       case_witness_positive_controls       },
        { "registry_drops_outside_the_lock", case_registry_drops_outside_the_lock },
        { "registry_concurrent_churn",       case_registry_concurrent_churn       },
        { "replan_token_nesting",            case_replan_token_nesting            },
        { "release_proc",                    case_release_proc                    },
        { "txn_guard_two_phases",            case_txn_guard_two_phases            },
        { "tenant_slots_and_weight_slots",   case_tenant_slots_and_weight_slots   },
    };
    for (const test_case & c : cases) {
        if (c.fn() != 0) {
            std::fprintf(stderr, "case %s failed\n", c.name);
            return 1;
        }
        std::printf("ok   %s\n", c.name);
    }
    std::printf("kv-region-registry: %zu cases passed\n", sizeof(cases) / sizeof(cases[0]));
    return 0;
}
