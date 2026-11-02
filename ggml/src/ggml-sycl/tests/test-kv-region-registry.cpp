// Host tests of the KV region registry, scope and guard (llama.cpp-moua L3c;
// H8 and the registry-level subset of H9).
//
// What this proves, and what it does not.  The header models the lock order, the
// thread_local scope, the residency answer, the two-phase guard and the release
// proc over opaque handles.  Scored here:
//   H8   the scope is thread_local; nested begin aborts and a refused guard does
//        not close the outer scope; the residency answer under, outside and
//        across scopes (mixed device set; publish A / publish B / create A); the
//        attach reads the owner device's registry; four contexts x two buffers on
//        four threads, every thread's slots checked, one thread parked mid-attach
//        and its own result read after the release, with a global-scope control;
//   H9   the lock witness and its controls; registry copy-in/copy-out and drops
//        outside the lock; the outermost-only token; the two-phase guard; the
//        release proc (steps 1-3 of §2.4.2) with its fence hand-off, idempotence,
//        L0 wait, re-entry, order and lock-failure arms.
//
// Deferred, by H9 sub-id, and not scored here (each needs a lifecycle body, the
// ledger, MMID or the republish allowlist, so it belongs to L4 or later):
//   H9 failpoint sweep  head-slot refusal, accounting refusal, non-FA refusal,
//                       yield relock, carve of device 0 of 2, MMID, CAS; each with
//                       "registry and slot table unchanged", "no yield before step
//                       5" and ring rows still in their original slots at MMID/CAS;
//   H9 first-context    the FIRST_CONTEXT range re-record in the guard (the guard
//                       here takes a generic callback);
//   H9 concurrency      serialized concurrent re-plan pairings (zero busy or lost
//                       CAS, with their REDs); A's transaction between B's early
//                       stage and load_end; A in a transaction then B loads (the
//                       probe and FA recheck);
//   H9 lifecycle        can_unload against a pending unload; reactivation and
//                       complete_unload holding L0 (runs 1 and 2); load-B-while-A-
//                       holds-rows (r7fz) fixtures (1), (2), (3);
//   H9 (4) ONEDNN, (5) SCRATCH, (5b), (5c)  the ledger arms, including the later-
//                       load weight-slot store;
//   H2  the H_A2 constant and the jehw-pipeline capacity RED (A2 runs at H = 0, 10
//                       and 11 MiB here; the RED is a leased top copy), and the
//                       MTP and assistant shapes.
//
// Every arm that asserts an absence (no violation, no drop under a lock, no
// block) is paired with a positive control that provokes the very thing it looks
// for, so a zero is a measurement.  The mutations run against the arms are
// recorded in the commit messages of this file, not here.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../kv-region-registry.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
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

// GCC cannot see that the replaced operator new above is malloc, and flags the free in the matching delete.
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

// The allocation detector: a replaced global operator new that counts on a thread that armed it.
// The TSan target defines KV_REGISTRY_TEST_NO_ALLOC_DETECTOR: the sanitizer runtime interposes
// operator new itself (a second definition is a link error) and must see every allocation, so
// the allocation case is skipped there and runs, unchanged, in the plain target.
#ifndef KV_REGISTRY_TEST_NO_ALLOC_DETECTOR
thread_local bool   g_count_allocations = false;
thread_local size_t g_allocations       = 0;

void * operator new(std::size_t n) {
    if (g_count_allocations) {
        ++g_allocations;
    }
    void * p = std::malloc(n != 0 ? n : 1);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

void operator delete(void * p) noexcept {
    std::free(p);
}

void operator delete(void * p, std::size_t) noexcept {
    std::free(p);
}
#endif  // KV_REGISTRY_TEST_NO_ALLOC_DETECTOR

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
    std::atomic<int> drops_with_l0{ 0 };  // drops that ran with the re-plan token held
};

kv_region_handle make_handle(drop_log & log) {
    return kv_region_handle(new int(0), [&log](void * p) {
        log.drops.fetch_add(1);
        if (kv_lock_witness::holds(KV_LOCK_L0_REPLAN)) {
            log.drops_with_l0.fetch_add(1);
        }
        // The release proc holds L0 to its end; no other lock may be held at a drop.
        if (kv_lock_witness::held_count_besides(KV_LOCK_L0_REPLAN) != 0) {
            log.drops_with_lock.fetch_add(1);
        }
        delete static_cast<int *>(p);
    });
}

// A slot added at a key that holds none: nothing is replaced, so nothing is returned.
void add_fresh(kv_tenant_slots & t, const std::string & cohort, uint32_t index, kv_region_handle h, size_t cap) {
    const kv_slot_retention previous = t.add(cohort, index, std::move(h), cap);
    if (previous.owner) {
        std::abort();
    }
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

    // A guard whose begin was refused must not close the scope it did not open.
    {
        kv_region_scope::guard outer(31);
        const size_t           before = abort_count();
        {
            kv_region_scope::guard inner(32);
            CHECK_EQ(abort_count(), before + 1, "the nested guard's begin aborts once");
        }
        CHECK(kv_region_scope::active(&id) && id == 31, "the refused nested guard left the outer scope open");
    }
    CHECK(!kv_region_scope::active(), "and the outer guard closed it");
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

// ---- H8: four contexts, two buffers each, on four threads --------------------
// A reusable barrier over the threads of one arm.
class arm_barrier {
  public:
    explicit arm_barrier(int n) : n_(n) {}

    void wait() {
        const int gen = gen_.load();
        if (arrived_.fetch_add(1) + 1 == n_) {
            arrived_.store(0);
            gen_.fetch_add(1);
            return;
        }
        while (gen_.load() == gen) {
            std::this_thread::yield();
        }
    }

  private:
    int              n_;
    std::atomic<int> arrived_{ 0 };
    std::atomic<int> gen_{ 0 };
};

// Context `c`'s region: its own offsets, so a slot read from the wrong context is visible.
kv_region_entry make_context_entry(drop_log & log, kv_context_id c, std::initializer_list<uint32_t> layers) {
    kv_region_entry e;
    e.extents.push_back(make_handle(log));
    for (uint32_t l : layers) {
        kv_layer_slice s;
        s.slot_offset = 1000000ull * c + 4096ull * l;
        s.slot_size   = 4096;
        e.layout[l]   = s;
    }
    return e;
}

bool wired_to(const kv_attach_result & r, kv_context_id c) {
    if (!r.ok || r.context != c || r.slots.empty()) {
        return false;
    }
    for (const kv_attached_slot & s : r.slots) {
        if (s.context != c || s.slice.slot_offset != 1000000ull * c + 4096ull * s.layer) {
            return false;
        }
    }
    return true;
}

int case_h8_four_contexts_two_buffers() {
    abort_capture                cap;
    drop_log                     log;
    kv_region_registry           dev0;
    kv_region_registry           dev1;
    constexpr int                N = 4;
    std::vector<kv_region_entry> keep;
    for (kv_context_id c = 1; c <= N; ++c) {
        keep.push_back(dev0.publish(c, make_context_entry(log, c, { 0, 1 })));
        keep.push_back(dev1.publish(c, make_context_entry(log, c, { 2, 3 })));
    }
    const std::vector<const kv_region_registry *> regs  = { &dev0, &dev1 };
    const std::vector<kv_layer_plan>              buf_a = {
        { 0, 0 },
        { 1, 0 },
        { 2, 1 }
    };
    const std::vector<kv_layer_plan> buf_b = {
        { 2, 1 },
        { 3, 1 },
        { 1, 0 }
    };

    arm_barrier       in_scope(N);
    std::atomic<bool> release_a{ false };
    std::atomic<int>  parked{ 0 };
    kv_attach_result  first[N + 1];
    kv_attach_result  second[N + 1];
    std::atomic<bool> done[N + 1];
    for (std::atomic<bool> & d : done) {
        d.store(false);
    }
    std::vector<std::thread> threads;
    for (int c = 1; c <= N; ++c) {
        threads.emplace_back([&, c] {
            kv_region_scope::guard g((kv_context_id) c);
            in_scope.wait();  // every context is inside its own scope at once
            first[c]  = kv_region_attach_buffer(regs, buf_a, [&] {
                if (c == 1) {
                    // Parked mid-attach, with the other three scopes open and attaching.
                    parked.store(1);
                    while (!release_a.load()) {
                        std::this_thread::yield();
                    }
                }
            });
            second[c] = kv_region_attach_buffer(regs, buf_b);
            done[c].store(true);
        });
    }
    while (parked.load() == 0) {
        std::this_thread::yield();
    }
    for (int spin = 0; spin < 2000 && !(done[2].load() && done[3].load() && done[4].load()); ++spin) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(done[2].load() && done[3].load() && done[4].load(),
          "three contexts finish both buffers while the first is parked mid-attach");
    CHECK(!done[1].load(), "control: the parked thread has not finished");
    release_a.store(true);
    for (std::thread & t : threads) {
        t.join();
    }
    for (int c = 1; c <= N; ++c) {
        CHECK(wired_to(first[c], (kv_context_id) c), "buffer 1 of every context attaches that context's slots");
        CHECK(wired_to(second[c], (kv_context_id) c), "buffer 2 of every context attaches the same region");
        CHECK(first[c].slots.size() == 3 && second[c].slots.size() == 3, "three slots per buffer");
    }
    CHECK(wired_to(first[1], 1) && wired_to(second[1], 1),
          "the parked thread's own results, read after its release, are its own");
    CHECK_EQ(abort_count(), 0, "no abort across four concurrent scopes");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation");

    // Positive control: the same arm with a process-global current context.  Every
    // thread then attaches from whichever begin ran last, so all but one are cross-wired.
    std::atomic<kv_context_id> global_id{ 0 };
    arm_barrier                begun(N);
    std::atomic<int>           correct{ 0 };
    std::vector<std::thread>   ctl;
    for (int c = 1; c <= N; ++c) {
        ctl.emplace_back([&, c] {
            global_id.store((kv_context_id) c);
            begun.wait();
            const kv_attach_result r = kv_region_attach_for(global_id.load(), regs, buf_a);
            correct.fetch_add(wired_to(r, (kv_context_id) c) ? 1 : 0);
        });
    }
    for (std::thread & t : ctl) {
        t.join();
    }
    CHECK_EQ(correct.load(), 1, "control: a global scope wires exactly one of four threads to its own context");
    return 0;
}

// ---- H9: the lock witness, with positive controls --------------------------
int case_witness_positive_controls() {
    abort_capture      cap;
    kv_witnessed_mutex l1{ KV_LOCK_L1_INVENTORY, "L1" };
    kv_witnessed_mutex l3{ KV_LOCK_L3_REGISTRY, "kv_region_mutex_" };
    kv_witnessed_mutex slot{ KV_LOCK_L5_SLOT_STATE, "slot-state lock", KV_L5_SLOT_STATE };
    kv_witnessed_mutex l5{ KV_LOCK_L5_GROUP, "L5" };

    {
        std::lock_guard<kv_witnessed_mutex> a(l1);
        std::lock_guard<kv_witnessed_mutex> b(slot);
    }
    {
        std::lock_guard<kv_witnessed_mutex> a(l1);
        std::lock_guard<kv_witnessed_mutex> b(l5);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "L1 then the slot-state lock, and L1 then a group mutex, are clean");

    // Two L5 peers are ordered by the tie-break: a group mutex before the slot-state lock is
    // clean, the other way round is an order violation.
    {
        std::lock_guard<kv_witnessed_mutex> a(l5);
        std::lock_guard<kv_witnessed_mutex> b(slot);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "a group mutex then the slot-state lock ascends the L5 tie-break");
    {
        std::lock_guard<kv_witnessed_mutex> a(slot);
        std::lock_guard<kv_witnessed_mutex> b(l5);
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: the slot-state lock then a group mutex is flagged");
    CHECK(kv_lock_witness::last().find("order") != std::string::npos, "as an order violation");
    kv_lock_witness::reset();

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
        CHECK(last_abort().find("[REPLAN-TOKEN] release proc entered with L0 held") != std::string::npos,
              "and says what the spec's witness says");
        CHECK_EQ(l0.lock_count(), 2, "it did not lock again");
    }
    return 0;
}

// ---- the release proc -----------------------------------------------------
// What the retention callback saw, standing in for retain_handles_until_event.
struct retain_probe {
    std::vector<uint64_t> events;
    size_t                handed        = 0;
    bool                  with_lock     = false;  // a lock other than L0 held at the call
    bool                  l0_held       = true;   // L0 held at every call (the proc holds it to the end)
    bool                  extents_alive = true;   // the batch was still intact at every call
};

int case_release_proc() {
    abort_capture      cap;
    drop_log           log;
    kv_replan_token    l0;
    kv_region_registry dev0;
    kv_region_registry dev1;

    // Context 1's tenant table holds two ring rows, each with a slot-state
    // retention of its last generation (owner + done event), and one slot with none.
    auto tenants = std::make_shared<kv_tenant_slots>();
    add_fresh(*tenants, "ring", 0, make_handle(log), 1024);
    add_fresh(*tenants, "ring", 1, make_handle(log), 1024);
    add_fresh(*tenants, "rows", 2, make_handle(log), 512);
    kv_slot_retention first = tenants->exchange_retention("ring", 0, { make_handle(log), 100 });
    CHECK(!first.owner, "the first generation of a row has no previous retention");
    first = tenants->exchange_retention("ring", 1, { make_handle(log), 101 });
    CHECK(!first.owner, "nor does the second row");

    kv_region_entry e0 = make_entry(log, { 0 });
    e0.tenants         = tenants;
    std::vector<std::weak_ptr<void>> extents;
    extents.push_back(e0.extents[0]);
    kv_region_entry p0 = dev0.publish(1, std::move(e0));
    kv_region_entry e1 = make_entry(log, { 1 });
    extents.push_back(e1.extents[0]);
    kv_region_entry p1 = dev1.publish(1, std::move(e1));
    kv_region_entry p2 = dev0.publish(2, make_entry(log, { 0 }));  // another context: untouched
    tenants.reset();

    std::vector<kv_region_registry *> regs = { &dev0, &dev1 };
    retain_probe                      probe;
    auto                              retain = [&](kv_region_handle && h, uint64_t event) {
        ++probe.handed;
        probe.events.push_back(event);
        probe.with_lock = probe.with_lock || kv_lock_witness::held_count_besides(KV_LOCK_L0_REPLAN) != 0;
        probe.l0_held = probe.l0_held && l0.held_by_this_thread();
        for (const std::weak_ptr<void> & w : extents) {
            probe.extents_alive = probe.extents_alive && !w.expired();
        }
        h.reset();  // the event has passed
    };

    kv_region_release(regs, 1, l0, retain);
    CHECK_EQ(l0.lock_count(), 1, "the release proc takes L0 once");
    CHECK_EQ(l0.unlock_count(), 1, "and releases it");
    CHECK(!dev0.has_layer(1, 0) && !dev1.has_layer(1, 1), "context 1 is gone from every device");
    CHECK(dev0.has_layer(2, 0), "another context's entry is untouched");
    CHECK_EQ(probe.handed, 2, "only the two ring rows' slot-state retentions are fenced");
    CHECK(probe.events.size() == 2 && ((probe.events[0] == 100 && probe.events[1] == 101) ||
                                       (probe.events[0] == 101 && probe.events[1] == 100)),
          "each retention is handed with its own done event");
    CHECK(!probe.with_lock, "retain ran with no lock held but L0");
    CHECK(probe.l0_held, "and with L0 held, which the proc holds to its end");
    CHECK(probe.extents_alive, "the extents are still in the batch when the retentions are handed over");
    CHECK(extents[0].expired() && extents[1].expired(), "the extents drop with the batch, not through the callback");
    // 2 extents + 3 slot handles + 2 retention owners; context 2's extent stays.
    CHECK_EQ(log.drops.load(), 7, "everything of context 1 was freed, nothing of context 2");
    CHECK_EQ(log.drops_with_lock.load(), 0, "no drop under a lock");
    CHECK_EQ(log.drops_with_l0.load(), 7, "and every one of them ran with L0 still held");

    // Idempotent: a second call finds nothing, aborts nothing.
    const size_t aborts_before = abort_count();
    const size_t handed_before = probe.handed;
    kv_region_release(regs, 1, l0, retain);
    CHECK_EQ(abort_count(), aborts_before, "a second release does not abort");
    CHECK_EQ(probe.handed, handed_before, "and retains nothing");

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
    CHECK(last_abort().find("[REPLAN-TOKEN] release proc entered with L0 held") != std::string::npos,
          "release under a held token aborts by name");
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

    // A retention with no fence callback aborts by name instead of freeing the row
    // under work that may still read it; no retention and no callback is quiet.
    {
        auto t = std::make_shared<kv_tenant_slots>();
        add_fresh(*t, "ring", 0, make_handle(log), 64);
        t->exchange_retention("ring", 0, { make_handle(log), 7 });
        kv_region_entry e = make_entry(log, { 0 });
        e.tenants         = t;
        t.reset();
        kv_region_entry held = dev0.publish(8, std::move(e));
        const size_t    n0   = abort_count();
        kv_region_release(regs, 8, l0, kv_retain_fn());
        CHECK_EQ(abort_count(), n0 + 1, "retentions and no callback abort once");
        CHECK(last_abort().find("no retain_handles_until_event callback") != std::string::npos, "naming the callback");
        CHECK(l0.probe_free(), "and L0 was released");

        kv_region_entry bare = dev0.publish(9, make_entry(log, { 0 }));
        const size_t    n1   = abort_count();
        kv_region_release(regs, 9, l0, kv_retain_fn());
        CHECK_EQ(abort_count(), n1, "no retention and no callback is not an error");
    }

    // A callback that throws aborts by name and does not stop the other retentions.
    {
        auto t = std::make_shared<kv_tenant_slots>();
        add_fresh(*t, "ring", 0, make_handle(log), 64);
        add_fresh(*t, "ring", 1, make_handle(log), 64);
        t->exchange_retention("ring", 0, { make_handle(log), 21 });
        t->exchange_retention("ring", 1, { make_handle(log), 22 });
        kv_region_entry e = make_entry(log, { 0 });
        e.tenants         = t;
        t.reset();
        kv_region_entry held  = dev0.publish(10, std::move(e));
        int             calls = 0;
        const size_t    n0    = abort_count();
        kv_region_release(regs, 10, l0, [&](kv_region_handle &&, uint64_t) {
            if (++calls == 1) {
                throw std::runtime_error("boom");
            }
        });
        CHECK_EQ(abort_count(), n0 + 1, "a throwing callback aborts once");
        CHECK(last_abort().find("retain_handles_until_event threw") != std::string::npos, "by name");
        CHECK_EQ(calls, 2, "and the second retention was still handed over");
        CHECK(l0.probe_free() && !dev0.has_layer(10, 0), "L0 released and the entry is gone");
    }
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

// ---- tenant slots and their slot-state retentions ---------------------------------------
int case_tenant_slots_and_retentions() {
    abort_capture   cap;
    drop_log        log;
    kv_tenant_slots slots;
    add_fresh(slots, "rows", 0, make_handle(log), 1000);
    add_fresh(slots, "rows", 1, make_handle(log), 500);
    add_fresh(slots, "per-op", 0, make_handle(log), 64);  // a DEVICE-scope slot: its own key

    const kv_claim rows0 = slots.claim("rows", 0, 1000);
    CHECK(rows0.result == kv_claim_result::OK, "claim at the cap is OK");
    CHECK(slots.claim("rows", 0, 10).result == kv_claim_result::ALREADY_CLAIMED,
          "a second claim of the slot is refused");
    CHECK(slots.claim("rows", 1, 501).result == kv_claim_result::OVER_PLAN, "a claim above the cap is OVER_PLAN");
    CHECK(slots.claim("rows", 9, 1).result == kv_claim_result::NO_SLOT, "a claim with no slot is NO_SLOT");
    const kv_claim perop = slots.claim("per-op", 0, 64);
    CHECK(perop.result == kv_claim_result::OK, "the per-op slot is independent of rows/0");
    CHECK(slots.claimed("per-op", 0) && !slots.claimed("rows", 1), "claim state is per (cohort, index)");

    std::vector<kv_region_handle> out;
    CHECK(!slots.take_unclaimed(out) && out.empty(), "slots with a live claim are not taken, and nothing moves");
    CHECK_EQ(slots.size(), 3, "all three slots remain");
    CHECK(slots.release_claim("rows", 0, rows0.generation), "the claim's own token releases it");
    CHECK(slots.release_claim("per-op", 0, perop.generation), "and so does the other");
    CHECK(!slots.any_claimed(), "no claim is live");
    CHECK(slots.take_unclaimed(out) && out.size() == 3, "the unclaimed slots move out");
    CHECK_EQ(log.drops.load(), 0, "moved out, not dropped: the caller drops with no lock held");
    out.clear();
    CHECK_EQ(log.drops.load(), 3, "dropped by the caller");
    CHECK_EQ(log.drops_with_lock.load(), 0, "outside every lock");

    // add() over a live key drops the old handle after the lock.
    kv_tenant_slots again;
    add_fresh(again, "rows", 0, make_handle(log), 10);
    const int before = log.drops.load();
    CHECK(!again.add("rows", 0, make_handle(log), 20).owner, "a replaced slot with no retention returns none");
    CHECK_EQ(log.drops.load(), before + 1, "a replaced slot's handle is dropped");
    CHECK_EQ(log.drops_with_lock.load(), 0, "outside the slot-state lock");
    CHECK_EQ(again.cap("rows", 0), 20, "and the cap is the new one");

    // Over a CLAIMED row that holds a live retention (a re-plan installing a new slot):
    // the retention comes back with its event for the caller to fence, the claim state
    // does not carry over, and nothing is freed unfenced.
    kv_tenant_slots over;
    add_fresh(over, "ring", 0, make_handle(log), 64);
    CHECK(over.claim("ring", 0, 10).result == kv_claim_result::OK, "claim the row");
    CHECK(!over.exchange_retention("ring", 0, { make_handle(log), 5 }).owner, "record its retention");
    const int         drops_over = log.drops.load();
    kv_slot_retention back       = over.add("ring", 0, make_handle(log), 128);
    CHECK(back.owner && back.done_event == 5, "the replaced row's retention comes back with its event");
    CHECK_EQ(log.drops.load(), drops_over + 1, "only the replaced slot's own handle dropped; the retention did not");
    CHECK(!over.claimed("ring", 0), "the new slot does not inherit the old claim");
    CHECK(over.claim("ring", 0, 100).result == kv_claim_result::OK, "and can be claimed");
    std::vector<kv_slot_retention> left;
    over.take_retentions(left);
    CHECK(left.empty(), "the old retention is not left in the table for the release proc to fence twice");
    back = kv_slot_retention();
    CHECK_EQ(log.drops.load(), drops_over + 2, "dropped by the caller after fencing");
    CHECK_EQ(log.drops_with_lock.load(), 0, "never under a lock");

    // Slot-state retentions: a re-claim hands the previous generation's back, with its
    // event, and a slot with no record keeps nothing.
    kv_tenant_slots ring;
    add_fresh(ring, "ring", 0, make_handle(log), 64);
    kv_slot_retention prev = ring.exchange_retention("ring", 0, { make_handle(log), 1 });
    CHECK(!prev.owner, "a row's first record has no previous retention");
    const int drops_r = log.drops.load();
    prev              = ring.exchange_retention("ring", 0, { make_handle(log), 2 });
    CHECK(prev.owner && prev.done_event == 1, "a re-record returns the previous generation with its own event");
    CHECK_EQ(log.drops.load(), drops_r, "moved out under the lock, not dropped");
    prev = kv_slot_retention();
    CHECK_EQ(log.drops.load(), drops_r + 1, "dropped by the caller after the lock");
    CHECK(!ring.exchange_retention("ring", 9, { make_handle(log), 3 }).owner,
          "a row with no slot returns nothing, and its offered owner is the caller's to drop");
    std::vector<kv_slot_retention> kept;
    ring.take_retentions(kept);
    CHECK(kept.size() == 1 && kept[0].done_event == 2, "take_retentions moves the live generation out");
    kept.clear();
    ring.take_retentions(kept);
    CHECK(kept.empty(), "and a second take finds nothing");
    CHECK_EQ(ring.size(), 1, "the slot itself stays: it drops with the entry");
    CHECK_EQ(log.drops_with_lock.load(), 0, "never under a lock");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation");
    return 0;
}

// ---- the L5 tie-break --------------------------------------------------------
int case_l5_tie_break() {
    abort_capture cap;

    const struct {
        int          ordinal;
        const char * name;
    } order[] = {
        { KV_L5_GROUP,           "group"           },
        { KV_L5_ARENA_AUTHORITY, "arena authority" },
        { KV_L5_RUNTIME_ALLOC,   "runtime alloc"   },
        { KV_L5_SLOT_STATE,      "slot state"      },
        { KV_L5_LEDGER_WRITER,   "ledger writer"   },
        { KV_L5_RETAINED_STORE,  "retained store"  },
    };

    const size_t n = sizeof(order) / sizeof(order[0]);
    // Every ascending pair is clean and every descending pair is flagged, each with its
    // own mutex object so the check reads the ordinal and not the instance.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            kv_lock_witness::reset();
            kv_witnessed_mutex first{ KV_LOCK_L5_GROUP, order[i].name, order[i].ordinal };
            kv_witnessed_mutex second{ KV_LOCK_L5_GROUP, order[j].name, order[j].ordinal };
            {
                std::lock_guard<kv_witnessed_mutex> a(first);
                std::lock_guard<kv_witnessed_mutex> b(second);
            }
            if (i < j) {
                CHECK_EQ(kv_lock_witness::violations(), 0, "an ascending ordinal pair is clean");
            } else {
                CHECK_EQ(kv_lock_witness::violations(), 1, "control: a descending ordinal pair is flagged");
            }
        }
    }

    // One ordinal: the instance key orders, and equal is flagged too.
    kv_lock_witness::reset();
    kv_witnessed_mutex g1{ KV_LOCK_L5_GROUP, "zone 1", KV_L5_GROUP, 1 };
    kv_witnessed_mutex g2{ KV_LOCK_L5_GROUP, "zone 2", KV_L5_GROUP, 2 };
    kv_witnessed_mutex g2b{ KV_LOCK_L5_GROUP, "zone 2 again", KV_L5_GROUP, 2 };
    {
        std::lock_guard<kv_witnessed_mutex> a(g1);
        std::lock_guard<kv_witnessed_mutex> b(g2);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "two group mutexes in ascending instance order are clean");
    {
        std::lock_guard<kv_witnessed_mutex> a(g2);
        std::lock_guard<kv_witnessed_mutex> b(g1);
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: descending instances are flagged");
    kv_lock_witness::reset();
    {
        std::lock_guard<kv_witnessed_mutex> a(g2);
        std::lock_guard<kv_witnessed_mutex> b(g2b);
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: equal ordinal and instance is flagged");
    CHECK(kv_lock_witness::last().find("instance collision") != std::string::npos &&
              kv_lock_witness::last().find("witness instance 2") != std::string::npos,
          "the equal-instance report names the possible collision and the instance");
    kv_lock_witness::reset();
    {
        std::lock_guard<kv_witnessed_mutex> a(g2);
        std::lock_guard<kv_witnessed_mutex> b(g1);
    }
    CHECK(kv_lock_witness::last().find("collision") == std::string::npos,
          "control: a plain descending pair is reported as an order violation, not a collision");
    kv_lock_witness::reset();

    // Two slots' spin locks are the same subsystem with their serials as instances.
    kv_lock_witness::reset();
    kv_slot_spin_lock s1(1), s2(2);
    {
        std::lock_guard<kv_slot_spin_lock> a(s1);
        std::lock_guard<kv_slot_spin_lock> b(s2);
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "two slot spin locks in serial order are clean");
    {
        std::lock_guard<kv_slot_spin_lock> a(s2);
        std::lock_guard<kv_slot_spin_lock> b(s1);
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: slot spin locks out of serial order are flagged");

    // A release in any order removes the lock it names, not the newest of its rank.
    kv_lock_witness::reset();
    kv_witnessed_mutex early{ KV_LOCK_L5_GROUP, "early", KV_L5_GROUP };
    kv_witnessed_mutex late{ KV_LOCK_L5_GROUP, "late", KV_L5_LEDGER_WRITER };
    early.lock();
    late.lock();
    early.unlock();  // the lower one first
    CHECK_EQ(kv_lock_witness::held_count(), 1, "one lock is still held");
    {
        kv_witnessed_mutex                  under{ KV_LOCK_L5_GROUP, "under", KV_L5_RETAINED_STORE };
        std::lock_guard<kv_witnessed_mutex> u(under);  // above the ledger writer: still clean
    }
    CHECK_EQ(kv_lock_witness::violations(), 0, "the lock still held is the later one");
    {
        kv_witnessed_mutex                  below{ KV_LOCK_L5_GROUP, "below", KV_L5_ARENA_AUTHORITY };
        std::lock_guard<kv_witnessed_mutex> u(
            below);  // below the ledger writer: flagged, so the writer is the one held
    }
    CHECK_EQ(kv_lock_witness::violations(), 1, "control: and a lock below it is flagged");
    late.unlock();
    CHECK_EQ(kv_lock_witness::held_count(), 0, "everything released");

    // The leaf rule is unchanged: nothing is taken under the registry lock, a claim included.
    kv_lock_witness::reset();
    drop_log        log;
    kv_tenant_slots slots;
    add_fresh(slots, "rows", 0, make_handle(log), 100);
    kv_witnessed_mutex registry{ KV_LOCK_L3_REGISTRY, "kv_region_mutex_" };
    {
        std::lock_guard<kv_witnessed_mutex> a(registry);
        (void) slots.claim("rows", 0, 10);
    }
    CHECK(kv_lock_witness::violations() >= 1, "control: a claim under the registry lock is flagged");
    CHECK(kv_lock_witness::last().find("leaf") != std::string::npos, "as a leaf violation");
    return 0;
}

// ---- claims: the generation token ---------------------------------------------
int case_claim_generation_token() {
    abort_capture   cap;
    drop_log        log;
    kv_tenant_slots slots;
    add_fresh(slots, "rows", 0, make_handle(log), 100);
    add_fresh(slots, "rows", 1, make_handle(log), 100);

    const kv_claim a = slots.claim("rows", 0, 10);
    CHECK(a.result == kv_claim_result::OK && a.generation != 0, "a claim returns a live token");
    CHECK_EQ(a.wait_event, 0, "a slot never released has no event to chain on");
    const kv_claim refused = slots.claim("rows", 0, 10);
    CHECK(refused.result == kv_claim_result::ALREADY_CLAIMED && refused.generation == 0,
          "a refused claim has no token");
    CHECK(slots.claim("rows", 0, 1000).result == kv_claim_result::OVER_PLAN,
          "the cap is checked before the claim state");
    CHECK(slots.claim("rows", 7, 1).generation == 0, "no slot, no token");

    // A token that is not the live claim's releases nothing.
    CHECK(!slots.release_claim("rows", 0, a.generation + 1000), "a made-up token releases nothing");
    CHECK(!slots.release_claim("rows", 0, 0), "token 0 releases nothing");
    CHECK(!slots.release_claim("rows", 1, a.generation), "another slot's key with this token releases nothing");
    CHECK(!slots.release_claim("nope", 0, a.generation), "no such slot, nothing released");
    CHECK(slots.claimed("rows", 0), "the claim is still live after every wrong release");

    // The stale holder: A releases, B claims, A's late release must not free B's claim.
    CHECK(slots.release_claim("rows", 0, a.generation, 7001), "the live token releases, with an event");
    CHECK(!slots.release_claim("rows", 0, a.generation), "a second release of the same token is a no-op");
    const kv_claim b = slots.claim("rows", 0, 10);
    CHECK(b.result == kv_claim_result::OK && b.generation != a.generation, "the next claim has a new token");
    CHECK_EQ(b.wait_event, 7001, "and receives the previous release's event to chain on");
    CHECK(!slots.release_claim("rows", 0, a.generation), "the stale holder's late release is refused");
    CHECK(slots.claimed("rows", 0), "so B's claim survives it");
    CHECK(slots.release_claim("rows", 0, b.generation), "B's own token releases");
    CHECK(!slots.claimed("rows", 0), "and the slot is free");
    CHECK_EQ(slots.claim("rows", 0, 10).wait_event, 0,
             "a release with no event leaves the next claim nothing to wait for");

    // A replaced slot: the old claimant's token never matches the new slot, even as its
    // first claim (tokens are unique across the table, not per slot).
    kv_tenant_slots over;
    add_fresh(over, "ring", 0, make_handle(log), 100);
    const kv_claim old_claim = over.claim("ring", 0, 10);
    CHECK(old_claim.result == kv_claim_result::OK, "claim the old slot");
    (void) over.add("ring", 0, make_handle(log), 100);
    const kv_claim new_claim = over.claim("ring", 0, 10);
    CHECK(new_claim.result == kv_claim_result::OK && new_claim.generation != old_claim.generation,
          "the replacement's first claim has a token the old claimant does not hold");
    CHECK(!over.release_claim("ring", 0, old_claim.generation), "the old claimant cannot release the new slot's claim");
    CHECK(over.claimed("ring", 0), "which is still live");
    CHECK(over.release_claim("ring", 0, new_claim.generation), "its own token releases it");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation");
    return 0;
}

// ---- claims: nothing allocates on the path ------------------------------------
// The claim path holds the slot's spin lock and the witness is reading the lock stack: neither
// may allocate (and a lookup must not copy the cohort string).  The detector is the replaced
// global operator new above, armed per thread; a control proves it counts.
#ifdef KV_REGISTRY_TEST_NO_ALLOC_DETECTOR
int case_claim_path_does_not_allocate() {
    std::printf("skip claim_path_does_not_allocate: the sanitizer runtime owns operator new in this target\n");
    return 0;
}
#else
int case_claim_path_does_not_allocate() {
    abort_capture     cap;
    drop_log          log;
    kv_tenant_slots   slots;
    const std::string cohort = "a-cohort-name-longer-than-the-small-string-buffer";
    add_fresh(slots, cohort, 0, make_handle(log), 100);
    add_fresh(slots, cohort, 1, make_handle(log), 100);
    const std::string missing = cohort + "-missing";

    g_allocations                       = 0;
    g_count_allocations                 = true;
    const kv_claim ok                   = slots.claim(cohort, 0, 10);
    const kv_claim again                = slots.claim(cohort, 0, 10);
    const kv_claim over                 = slots.claim(cohort, 1, 1000);
    const kv_claim none                 = slots.claim(missing, 0, 10);
    const bool     stale                = slots.release_claim(cohort, 0, ok.generation + 1000);
    const bool     held                 = slots.claimed(cohort, 0);
    const bool     any                  = slots.any_claimed();
    const bool     freed                = slots.release_claim(cohort, 0, ok.generation, 5);
    g_count_allocations                 = false;
    const size_t claim_path_allocations = g_allocations;

    CHECK(ok.result == kv_claim_result::OK && again.result == kv_claim_result::ALREADY_CLAIMED &&
              over.result == kv_claim_result::OVER_PLAN && none.result == kv_claim_result::NO_SLOT && !stale && held &&
              any && freed,
          "the armed region ran every claim-path outcome");
    CHECK_EQ(claim_path_allocations, 0, "the claim path allocates nothing");

    // Controls: the detector counts a plain allocation and a copied cohort string.
    g_allocations       = 0;
    g_count_allocations = true;
    int *             p = new int(1);
    const std::string copy(cohort.c_str());
    g_count_allocations = false;
    delete p;
    CHECK(g_allocations >= 2, "control: the detector counts an allocation and a copied long string");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation");
    return 0;
}
#endif  // KV_REGISTRY_TEST_NO_ALLOC_DETECTOR

// ---- claims: exclusion and concurrency ------------------------------------------
int case_claim_exclusion() {
    abort_capture   cap;
    drop_log        log;
    kv_tenant_slots slots;
    add_fresh(slots, "rows", 0, make_handle(log), 100);
    add_fresh(slots, "rows", 1, make_handle(log), 100);

    // Many threads race for each of two slots: at no instant do two hold the same one, and
    // every successful claim is released by its own token.
    constexpr int            n_threads  = 8;
    constexpr int            rounds     = 4000;
    std::atomic<int>         holders[2] = { { 0 }, { 0 } };
    std::atomic<int>         overlap{ 0 };
    std::atomic<int>         wins{ 0 };
    std::atomic<int>         bad_release{ 0 };
    std::atomic<int>         go{ 0 };
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            while (go.load() == 0) {
                std::this_thread::yield();
            }
            const uint32_t slot = t % 2;  // four threads race for each of the two slots
            for (int r = 0; r < rounds; ++r) {
                const kv_claim c = slots.claim("rows", slot, 10);
                if (c.result != kv_claim_result::OK) {
                    continue;
                }
                if (holders[slot].fetch_add(1) != 0) {
                    overlap.fetch_add(1);
                }
                wins.fetch_add(1);
                std::this_thread::yield();  // widen the window a second claimant would land in
                holders[slot].fetch_sub(1);
                if (!slots.release_claim("rows", slot, c.generation)) {
                    bad_release.fetch_add(1);
                }
            }
        });
    }
    go.store(1);
    for (std::thread & th : threads) {
        th.join();
    }
    CHECK_EQ(overlap.load(), 0, "two threads never hold the same slot at once, on either slot");
    CHECK_EQ(bad_release.load(), 0, "every winner's own token releases its claim");
    CHECK(wins.load() > 100, "the arm made progress: claims succeeded");
    CHECK(!slots.any_claimed(), "nothing is left claimed");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation under contention");
    return 0;
}

// ---- the slot table under contention ------------------------------------------------
// The retention fields and the slot map are touched from several threads at once on the
// dispatch path; these two cases are the ones a missing pin or a weakened spin lock
// shows up in under TSan (and, for the accounting checks, in a plain build).  Every handle
// made is dropped exactly once, so a lost or doubled retention moves the count.

// Retention exchange from three threads and take_retentions from a fourth, on one slot.
int case_retention_race() {
    drop_log         log;
    std::atomic<int> made{ 0 };
    {
        kv_tenant_slots slots;
        add_fresh(slots, "r", 0, make_handle(log), 100);
        made.fetch_add(1);
        constexpr int            rounds = 3000;
        std::vector<std::thread> threads;
        for (int t = 0; t < 3; ++t) {
            threads.emplace_back([&] {
                for (int i = 0; i < rounds; ++i) {
                    kv_slot_retention next;
                    next.owner      = make_handle(log);
                    next.done_event = (uint64_t) i + 1;
                    made.fetch_add(1);
                    const kv_slot_retention previous = slots.exchange_retention("r", 0, std::move(next));
                    (void) previous;  // dropped here, with no lock held
                }
            });
        }
        threads.emplace_back([&] {
            for (int i = 0; i < rounds; ++i) {
                std::vector<kv_slot_retention> out;
                slots.take_retentions(out);
            }
        });
        for (std::thread & th : threads) {
            th.join();
        }
        std::vector<kv_slot_retention> last;
        slots.take_retentions(last);
    }
    CHECK_EQ(log.drops.load(), made.load(), "every handle made is dropped exactly once");
    CHECK_EQ(log.drops_with_lock.load(), 0, "and none is dropped under a lock");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation under the retention race");
    return 0;
}

// Claim, handle and release from three threads against a replace (add over the live key)
// and take_unclaimed + re-add from two more.
int case_claim_vs_table_race() {
    drop_log         log;
    std::atomic<int> made{ 0 };
    {
        kv_tenant_slots slots;
        add_fresh(slots, "r", 0, make_handle(log), 100);
        made.fetch_add(1);
        std::vector<std::thread> threads;
        for (int t = 0; t < 3; ++t) {
            threads.emplace_back([&] {
                for (int i = 0; i < 6000; ++i) {
                    const kv_claim c = slots.claim("r", 0, 10);
                    if (c.result == kv_claim_result::OK) {
                        const kv_region_handle h = slots.handle("r", 0);
                        (void) slots.release_claim("r", 0, c.generation);
                    }
                }
            });
        }
        threads.emplace_back([&] {
            for (int i = 0; i < 1500; ++i) {
                made.fetch_add(1);
                kv_slot_retention previous = slots.add("r", 0, make_handle(log), 100);
                (void) previous;
            }
        });
        threads.emplace_back([&] {
            for (int i = 0; i < 1500; ++i) {
                std::vector<kv_region_handle> out;
                if (slots.take_unclaimed(out)) {
                    made.fetch_add(1);
                    kv_slot_retention previous = slots.add("r", 0, make_handle(log), 100);
                    (void) previous;
                }
            }
        });
        for (std::thread & th : threads) {
            th.join();
        }
        CHECK(!slots.any_claimed() || slots.size() > 0, "the table is coherent after the race");
    }
    CHECK_EQ(log.drops.load(), made.load(), "every slot handle made is dropped exactly once");
    CHECK_EQ(kv_lock_witness::violations(), 0, "no lock violation under the table race");
    return 0;
}

// A holder with a token that is not live (token 0, a made-up one) releasing in a loop while
// four threads claim and release the same slot: the stale release reads the slot's claim
// state, and must do it under the spin lock like every other reader.
int case_stale_release_race() {
    drop_log        log;
    kv_tenant_slots slots;
    add_fresh(slots, "r", 0, make_handle(log), 100);
    std::atomic<int>         overlap{ 0 };
    std::atomic<int>         bad_release{ 0 };
    std::atomic<int>         stale_won{ 0 };
    std::atomic<int>         holders{ 0 };
    std::atomic<bool>        stop{ false };
    std::vector<std::thread> threads;
    std::vector<std::thread> claimers;
    for (int t = 0; t < 4; ++t) {
        claimers.emplace_back([&] {
            for (int i = 0; i < 8000; ++i) {
                const kv_claim c = slots.claim("r", 0, 10);
                if (c.result != kv_claim_result::OK) {
                    continue;
                }
                if (holders.fetch_add(1) != 0) {
                    overlap.fetch_add(1);
                }
                holders.fetch_sub(1);
                if (!slots.release_claim("r", 0, c.generation)) {
                    bad_release.fetch_add(1);
                }
            }
        });
    }
    threads.emplace_back([&] {
        while (!stop.load()) {
            if (slots.release_claim("r", 0, 0) || slots.release_claim("r", 0, (uint64_t) 1 << 60)) {
                stale_won.fetch_add(1);
            }
        }
    });
    for (std::thread & th : claimers) {
        th.join();
    }
    stop.store(true);
    for (std::thread & th : threads) {
        th.join();
    }
    CHECK_EQ(stale_won.load(), 0, "a token that was never live releases nothing, under contention");
    CHECK_EQ(overlap.load(), 0, "the stale releaser never lets two claimants hold the slot");
    CHECK_EQ(bad_release.load(), 0, "every winner's own token still releases its claim");
    CHECK(!slots.any_claimed(), "nothing is left claimed");
    return 0;
}

// The two events a slot holds come from two different calls (kv_tenant_slots::slot): the
// claimant chains on what the RELEASING caller passed, the retained owner's drop is fenced
// on what the RECORDING caller passed.  The case gives them different values, so a registry
// that read one for the other fails.
int case_slot_events_are_two_facts() {
    drop_log        log;
    kv_tenant_slots slots;
    add_fresh(slots, "ring", 0, make_handle(log), 100);

    const kv_claim g1 = slots.claim("ring", 0, 10);
    (void) slots.exchange_retention("ring", 0, { make_handle(log), 5 });
    CHECK(slots.release_claim("ring", 0, g1.generation, 4), "generation 1 releases with event 4");

    const kv_claim g2 = slots.claim("ring", 0, 10);
    CHECK_EQ(g2.wait_event, 4,
             "the claimant chains on the event the releasing caller passed (4, not the retention's 5)");
    // Generation 2 records its retention with event 9 and then releases with event 8.
    const kv_slot_retention prev = slots.exchange_retention("ring", 0, { make_handle(log), 9 });
    CHECK(prev.owner && prev.done_event == 5,
          "the owner handed back carries the event its recorder passed (5), not the release's 4");
    CHECK(slots.claim("ring", 0, 10).result == kv_claim_result::ALREADY_CLAIMED, "the slot is held");
    CHECK(slots.release_claim("ring", 0, g2.generation, 8), "generation 2 releases with event 8");
    const kv_claim g3 = slots.claim("ring", 0, 10);
    CHECK_EQ(g3.wait_event, 8, "the next claimant chains on the release's event (8), not the retention's 9");
    std::vector<kv_slot_retention> kept;
    slots.take_retentions(kept);
    CHECK(kept.size() == 1 && kept[0].done_event == 9,
          "and the retained owner is fenced on the event its recorder passed (9), not the release's 8");
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
        { "h8_four_contexts_two_buffers",    case_h8_four_contexts_two_buffers    },
        { "witness_positive_controls",       case_witness_positive_controls       },
        { "registry_drops_outside_the_lock", case_registry_drops_outside_the_lock },
        { "registry_concurrent_churn",       case_registry_concurrent_churn       },
        { "replan_token_nesting",            case_replan_token_nesting            },
        { "release_proc",                    case_release_proc                    },
        { "txn_guard_two_phases",            case_txn_guard_two_phases            },
        { "tenant_slots_and_retentions",     case_tenant_slots_and_retentions     },
        { "l5_tie_break",                    case_l5_tie_break                    },
        { "claim_generation_token",          case_claim_generation_token          },
        { "claim_exclusion",                 case_claim_exclusion                 },
        { "claim_path_does_not_allocate",    case_claim_path_does_not_allocate    },
        { "retention_race",                  case_retention_race                  },
        { "claim_vs_table_race",             case_claim_vs_table_race             },
        { "stale_release_race",              case_stale_release_race              },
        { "slot_events_are_two_facts",       case_slot_events_are_two_facts       },
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
