// Host tests of the tenant claim scope and of the registry entry that holds a context's slot table
// (llama.cpp-moua L4 step 3c): tenant-claim-scope.hpp, and install_tenant_slots / take_tenant_slots
// in kv-region-registry.hpp.
//
//   (1) the scope is a thread_local: one per thread, a nested open is refused and changes nothing,
//       a close of anything but this thread's open scope is refused, and another thread never sees it;
//   (2) a claim takes the lowest free slot of its cohort, checks the request against that slot's cap,
//       and is refused by status (no scope, no slot, over plan) without taking a slot;
//   (3) a release frees only the slot the claim names, once, from any thread, and empties the record;
//   (4) the slot's handle is copied into the claim, so the memory outlives a table that is dropped
//       while a buffer still uses it;
//   (5) the registry installs a table once per entry, hands it back moved out, and never makes its
//       last drop under the registry's leaf lock.
//
// The build is -DNDEBUG, so CHECK is explicit and always runs.

#include "../kv-region-registry.hpp"
#include "../runtime-context-section.hpp"
#include "../tenant-claim-scope.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

using namespace ggml_sycl;

namespace {

const char * const HOST = "context-compute-host";

// A handle that records when it is dropped, and whether the registry's leaf lock was held then.
struct watched {
    bool * dropped;
    bool * under_lock;

    watched(bool * d, bool * u) : dropped(d), under_lock(u) {}

    ~watched() {
        *dropped    = true;
        *under_lock = kv_lock_witness::holds(KV_LOCK_L3_REGISTRY);
    }
};

kv_region_handle plain_handle() {
    return std::make_shared<int>(7);
}

// A table with host slots of the given caps at indices 0..n-1.
std::shared_ptr<kv_tenant_slots> table_of(const std::vector<size_t> & caps, const char * cohort = HOST) {
    auto t = std::make_shared<kv_tenant_slots>();
    for (size_t i = 0; i < caps.size(); ++i) {
        (void) t->add(cohort, (uint32_t) i, plain_handle(), caps[i]);
    }
    return t;
}

// The scope an open returned, null when it did not open.
tenant_claim_scope::state * open_scope(std::shared_ptr<kv_tenant_slots> t) {
    tenant_claim_scope::state * out = nullptr;
    (void) tenant_claim_scope::open(std::move(t), out);
    return out;
}

size_t claims_of(const tenant_claim_scope::state * s) {
    size_t n = 12345;
    CHECK(tenant_claim_scope::claims_made(s, n), "claims_made refused this thread's open scope");
    return n;
}

// A second record of a claim `a` holds: what a stale holder keeps after the slot was released and claimed again.
tenant_claim copy_of(const tenant_claim & a) {
    tenant_claim c;
    c.slots      = a.slots;
    c.cohort     = a.cohort;
    c.index      = a.index;
    c.generation = a.generation;
    c.wait_event = a.wait_event;
    c.owner      = a.owner;
    return c;
}

void case_scope_is_per_thread() {
    CHECK(!tenant_claim_scope::active(), "no scope before an open");
    CHECK(open_scope(nullptr) == nullptr && !tenant_claim_scope::active(), "a null table opens nothing");
    auto                        t = table_of({ 100 });
    tenant_claim_scope::state * s = open_scope(t);
    CHECK(s != nullptr && tenant_claim_scope::active(), "an open with a table opens the scope");
    CHECK(open_scope(t) == nullptr, "a nested open is refused");
    CHECK(tenant_claim_scope::active(), "and the outer scope stays open");

    bool other_thread_active = true, other_close = true, other_own_open = false, other_close_ours = true,
         other_own_closed = false;
    std::thread([&] {
        other_thread_active = tenant_claim_scope::active();
        other_close         = tenant_claim_scope::close(s);
        // a thread with a scope of its own still cannot close ours, and its own is untouched by the try
        auto * mine         = open_scope(table_of({ 5 }));
        other_own_open      = mine != nullptr;
        other_close_ours    = tenant_claim_scope::close(s);
        other_own_closed    = !tenant_claim_scope::active() ? false : tenant_claim_scope::close(mine);
    }).join();
    CHECK(!other_thread_active, "another thread sees no scope");
    CHECK(!other_close && tenant_claim_scope::active(), "another thread cannot close this thread's scope");
    CHECK(other_own_open && !other_close_ours && other_own_closed,
          "a thread with its own scope cannot close ours, and closes its own");
    CHECK(tenant_claim_scope::active(), "ours is still open");

    CHECK(!tenant_claim_scope::close(nullptr), "a close of null is refused");
    CHECK(tenant_claim_scope::close(s), "the opening thread closes it");
    CHECK(!tenant_claim_scope::active(), "and no scope is open after");
    CHECK(!tenant_claim_scope::close(s), "a second close is refused");

    tenant_claim claim;
    CHECK(tenant_claim_scope::claim(HOST, 1, claim).status == tenant_claim_status::NO_SCOPE && !claim.live(),
          "a claim with no scope answers NO_SCOPE and takes nothing");
    CHECK(t->claimed(HOST, 0) == false, "and no slot is marked");
    size_t none = 7;
    CHECK(!tenant_claim_scope::claims_made(nullptr, none) && none == 0, "claims_made of null is refused and 0");

    // a new open starts from zero claims, also after a scope that made some
    s = open_scope(t);
    CHECK(s != nullptr && claims_of(s) == 0, "an opened scope starts at zero claims");
    tenant_claim made;
    CHECK(tenant_claim_scope::claim(HOST, 1, made).status == tenant_claim_status::OK && claims_of(s) == 1,
          "a claim is counted");
    CHECK(tenant_claim_scope::release(made) && tenant_claim_scope::close(s), "release and close");
    s = open_scope(t);
    CHECK(s != nullptr && claims_of(s) == 0, "a reopened scope starts at zero claims again");
    CHECK(tenant_claim_scope::close(s), "and closes");
}

void case_open_status_names_why() {
    auto                        t   = table_of({ 10 });
    tenant_claim_scope::state * out = reinterpret_cast<tenant_claim_scope::state *>(0x1);
    CHECK(tenant_claim_scope::open(nullptr, out) == tenant_claim_scope::open_status::NO_TABLE && out == nullptr,
          "no table: NO_TABLE, and `out` is cleared");
    CHECK(tenant_claim_scope::open(t, out) == tenant_claim_scope::open_status::OPENED && out != nullptr, "OPENED");
    tenant_claim_scope::state * scope = out;
    tenant_claim_scope::state * again = reinterpret_cast<tenant_claim_scope::state *>(0x1);
    CHECK(tenant_claim_scope::open(t, again) == tenant_claim_scope::open_status::NESTED && again == nullptr,
          "a second open on the thread: NESTED, and `out` is cleared");
    CHECK(tenant_claim_scope::open(nullptr, again) == tenant_claim_scope::open_status::NESTED,
          "NESTED wins over a missing table: a caller inside a scope always learns it");
    CHECK(tenant_claim_scope::active() && tenant_claim_scope::close(scope), "the refusals left the first scope intact");
}

void case_claims_made_is_validated() {
    auto   t = table_of({ 10 });
    auto * s = open_scope(t);
    size_t n = 99;
    CHECK(tenant_claim_scope::claims_made(s, n) && n == 0, "this thread's open scope is read");
    bool   other_ok = true;
    size_t other_n  = 99;
    std::thread([&] { other_ok = tenant_claim_scope::claims_made(s, other_n); }).join();
    CHECK(!other_ok && other_n == 0, "another thread's scope is not read");
    CHECK(tenant_claim_scope::close(s), "close");
    CHECK(!tenant_claim_scope::claims_made(s, n) && n == 0, "a closed scope is not read");
}

void case_a_scope_left_open_is_counted() {
    const size_t before = tenant_claim_scope::leaked_scopes();
    std::thread([&] {
        auto * s = open_scope(table_of({ 4 }));
        CHECK(s != nullptr, "the thread's scope opens");
        // never closed: the thread exits with it open
    }).join();
    CHECK(tenant_claim_scope::leaked_scopes() == before + 1, "a scope open at thread exit is counted as a leak");
    std::thread([&] {
        auto * s = open_scope(table_of({ 4 }));
        CHECK(s != nullptr && tenant_claim_scope::close(s), "a closed scope opens and closes");
    }).join();
    CHECK(tenant_claim_scope::leaked_scopes() == before + 1, "a closed scope is not a leak");
}

void case_a_dropped_record_releases_its_slot() {
    auto   t = table_of({ 10, 10 });
    auto * s = open_scope(t);
    {
        tenant_claim a;
        CHECK(tenant_claim_scope::claim(HOST, 5, a).status == tenant_claim_status::OK && t->claimed(HOST, 0), "claim");
    }
    CHECK(!t->claimed(HOST, 0), "a record dropped while live gives its slot back");

    // a moved record is the one live record: the moved-from does not release, the destination does
    tenant_claim a;
    CHECK(tenant_claim_scope::claim(HOST, 5, a).status == tenant_claim_status::OK, "claim again");
    {
        tenant_claim moved = std::move(a);
        CHECK(!a.live() && moved.live(), "a move transfers the claim");
        CHECK(t->claimed(HOST, 0), "and the slot is still held");
    }
    CHECK(!t->claimed(HOST, 0), "the destination's drop released it");
    CHECK(!tenant_claim_scope::release(a), "the moved-from record releases nothing");

    // an assignment over a live record releases what that record held
    tenant_claim x, y;
    CHECK(tenant_claim_scope::claim(HOST, 5, x).status == tenant_claim_status::OK && x.index == 0, "x holds slot 0");
    CHECK(tenant_claim_scope::claim(HOST, 5, y).status == tenant_claim_status::OK && y.index == 1, "y holds slot 1");
    x = std::move(y);
    CHECK(!t->claimed(HOST, 0) && t->claimed(HOST, 1),
          "the assignment released the record it replaced, not the new one");

    // an explicit release is not repeated by the destructor
    CHECK(tenant_claim_scope::release(x, 4) && !t->claimed(HOST, 1), "explicit release");
    tenant_claim z;
    CHECK(tenant_claim_scope::claim(HOST, 5, z).status == tenant_claim_status::OK && z.wait_event == 0 && z.index == 0,
          "claim slot 0 again");
    tenant_claim w;
    CHECK(tenant_claim_scope::claim(HOST, 5, w).status == tenant_claim_status::OK && w.index == 1 && w.wait_event == 4,
          "slot 1 carries the explicit release's event: the destructor did not overwrite it with 0");
    CHECK(tenant_claim_scope::close(s), "close");
}

void case_out_of_order_is_served_not_reported() {
    // The design reports 0,2 / 1,0 as plan bugs; this table serves them at the lowest free slot (see the header).
    auto         t = table_of({ 10, 10, 10 });
    auto *       s = open_scope(t);
    tenant_claim a, b, c;
    CHECK(tenant_claim_scope::claim(HOST, 1, a).index == 0 && tenant_claim_scope::claim(HOST, 1, b).index == 1 &&
              tenant_claim_scope::claim(HOST, 1, c).index == 2,
          "in order");
    CHECK(tenant_claim_scope::release(a) && tenant_claim_scope::release(c), "free 0 then 2: out of order");
    tenant_claim d, e, f;
    auto         od = tenant_claim_scope::claim(HOST, 1, d);
    auto         oe = tenant_claim_scope::claim(HOST, 1, e);
    auto         of = tenant_claim_scope::claim(HOST, 1, f);
    CHECK(od.status == tenant_claim_status::OK && od.index == 0 && oe.status == tenant_claim_status::OK &&
              oe.index == 2 && of.status == tenant_claim_status::NO_SLOT,
          "the free slots are served lowest first and nothing is reported as a plan violation");
    CHECK(tenant_claim_scope::close(s), "close");
}

void case_a_sparse_slot_set_is_fully_reachable() {
    // llama makes no element for a zero cap, so a cohort can hold indices {0, 2}: both slots are claimable.
    auto t = std::make_shared<kv_tenant_slots>();
    (void) t->add(HOST, 0, plain_handle(), 10);
    (void) t->add(HOST, 2, plain_handle(), 30);
    (void) t->add("context-compute", 1, plain_handle(), 99);  // another cohort's slot is never this cohort's
    auto *       s = open_scope(t);
    tenant_claim a, b, c;
    auto         oa = tenant_claim_scope::claim(HOST, 10, a);
    auto         ob = tenant_claim_scope::claim(HOST, 30, b);
    CHECK(
        oa.status == tenant_claim_status::OK && oa.index == 0 && ob.status == tenant_claim_status::OK && ob.index == 2,
        "the second slot of {0, 2} is reached, at index 2");
    CHECK(tenant_claim_scope::claim(HOST, 1, c).status == tenant_claim_status::NO_SLOT && !c.live(),
          "with both claimed there is no slot, and the gap at 1 is not one");
    CHECK(tenant_claim_scope::release(a) && tenant_claim_scope::release(b), "release");
    tenant_claim d;
    auto         od = tenant_claim_scope::claim(HOST, 31, d);
    CHECK(od.status == tenant_claim_status::OVER_PLAN && od.index == 0 && od.cap == 10,
          "the lowest free slot is checked, as before");
    CHECK(tenant_claim_scope::close(s), "close");
}

void case_claim_takes_the_lowest_free_slot() {
    auto   t = table_of({ 100, 200, 300 });
    auto * s = open_scope(t);
    CHECK(s != nullptr, "scope opens");
    tenant_claim a, b, c, d;
    auto         oa = tenant_claim_scope::claim(HOST, 50, a);
    CHECK(oa.status == tenant_claim_status::OK && oa.index == 0 && a.index == 0 && a.live(),
          "first claim takes index 0");
    auto ob = tenant_claim_scope::claim(HOST, 150, b);
    CHECK(ob.status == tenant_claim_status::OK && ob.index == 1 && b.index == 1, "second claim takes index 1");
    CHECK(a.generation != b.generation, "each claim has its own generation");
    CHECK(t->claimed(HOST, 0) && t->claimed(HOST, 1) && !t->claimed(HOST, 2), "slots 0 and 1 are marked, 2 is not");
    CHECK(claims_of(s) == 2, "the scope counts its two claims");

    // free out of order: index 0 goes, index 1 stays; the next claim takes index 0 again, not 2
    CHECK(tenant_claim_scope::release(a, 11) && !a.live() && a.slots == nullptr, "release empties the record");
    CHECK(!t->claimed(HOST, 0) && t->claimed(HOST, 1), "only slot 0 is free");
    auto oc = tenant_claim_scope::claim(HOST, 60, c);
    CHECK(oc.status == tenant_claim_status::OK && oc.index == 0, "the lowest free index is taken again");
    CHECK(c.wait_event == 11, "and the claim carries the previous release's event");

    auto od = tenant_claim_scope::claim(HOST, 300, d);
    CHECK(od.status == tenant_claim_status::OK && od.index == 2, "a request equal to the cap fits");
    tenant_claim e;
    auto         oe = tenant_claim_scope::claim(HOST, 1, e);
    CHECK(oe.status == tenant_claim_status::NO_SLOT && !e.live(), "every slot claimed: NO_SLOT, nothing taken");
    CHECK(claims_of(s) == 4, "a refused claim is not counted");
    CHECK(tenant_claim_scope::release(b) && tenant_claim_scope::release(c) && tenant_claim_scope::release(d),
          "releases");
    CHECK(!t->any_claimed(), "nothing is claimed after every release");
    CHECK(tenant_claim_scope::close(s), "close");
}

void case_claim_refusals() {
    auto         t = table_of({ 100, 40 });
    auto *       s = open_scope(t);
    tenant_claim a, b, miss;
    CHECK(tenant_claim_scope::claim(HOST, 101, a).status == tenant_claim_status::OVER_PLAN,
          "a request over the cap is OVER_PLAN");
    auto over = tenant_claim_scope::claim(HOST, 101, a);
    CHECK(over.index == 0 && over.cap == 100, "and names the slot and its cap");
    CHECK(!a.live() && !t->claimed(HOST, 0), "an over-plan claim takes nothing");

    CHECK(tenant_claim_scope::claim(HOST, 100, a).status == tenant_claim_status::OK, "the cap itself fits");
    auto second = tenant_claim_scope::claim(HOST, 41, b);
    CHECK(second.status == tenant_claim_status::OVER_PLAN && second.index == 1 && second.cap == 40,
          "the lowest FREE slot is the one checked, not the largest");
    CHECK(!b.live(), "and nothing is taken");

    auto none = tenant_claim_scope::claim("context-compute", 1, miss);
    CHECK(none.status == tenant_claim_status::NO_SLOT && none.index == 0,
          "a cohort with no slot is NO_SLOT at index 0");
    CHECK(tenant_claim_scope::release(a), "release");
    CHECK(tenant_claim_scope::close(s), "close");

    // a claim that is refused leaves a previous record untouched
    auto         t2 = table_of({ 10 });
    auto *       s2 = open_scope(t2);
    tenant_claim keep;
    CHECK(tenant_claim_scope::claim(HOST, 5, keep).status == tenant_claim_status::OK, "claim");
    const uint64_t gen = keep.generation;
    CHECK(tenant_claim_scope::claim(HOST, 5, keep).status == tenant_claim_status::NO_SLOT,
          "second claim finds no slot");
    CHECK(keep.live() && keep.generation == gen, "the refused claim left the record as it was");
    CHECK(tenant_claim_scope::release(keep) && tenant_claim_scope::close(s2), "release and close");
}

void case_release_is_exact() {
    auto         t = table_of({ 10 });
    auto *       s = open_scope(t);
    tenant_claim a;
    CHECK(tenant_claim_scope::claim(HOST, 5, a).status == tenant_claim_status::OK, "claim");
    tenant_claim stale = copy_of(a);  // a second record of the same claim
    CHECK(tenant_claim_scope::release(a, 3), "release");
    CHECK(!tenant_claim_scope::release(a), "a released record releases nothing");
    CHECK(!tenant_claim_scope::release(stale), "a stale copy of the claim releases nothing");

    tenant_claim b;
    CHECK(tenant_claim_scope::claim(HOST, 5, b).status == tenant_claim_status::OK, "the slot is claimed again");
    CHECK(!tenant_claim_scope::release(stale) && t->claimed(HOST, 0), "the stale token does not free the later claim");

    tenant_claim empty;
    CHECK(!tenant_claim_scope::release(empty), "a record that never claimed releases nothing");
    tenant_claim no_table;
    no_table.generation = 5;
    CHECK(!tenant_claim_scope::release(no_table), "a record that names a claim but holds no table releases nothing");

    // from another thread, after the scope closed
    CHECK(tenant_claim_scope::close(s), "close");
    bool released = false;
    std::thread([&] { released = tenant_claim_scope::release(b, 9); }).join();
    CHECK(released && !t->claimed(HOST, 0), "a free from another thread, after the scope, releases the slot");
}

void case_claim_keeps_the_memory() {
    bool dropped = false, under = true;
    auto t = std::make_shared<kv_tenant_slots>();
    (void) t->add(HOST, 0, std::make_shared<watched>(&dropped, &under), 64);
    auto *       s = open_scope(t);
    tenant_claim a;
    CHECK(tenant_claim_scope::claim(HOST, 8, a).status == tenant_claim_status::OK && a.owner != nullptr,
          "a claim carries the slot's handle");
    CHECK(tenant_claim_scope::close(s), "close");
    t.reset();
    CHECK(!dropped, "the registry's hold of the table is gone and the memory is not: the claim holds both");
    CHECK(tenant_claim_scope::release(a), "release");
    CHECK(dropped, "the last holder's release drops the memory");
}

void case_registry_install_and_take() {
    kv_region_registry reg;
    auto               t = table_of({ 10 });
    CHECK(!reg.install_tenant_slots(1, nullptr, 5) && reg.size() == 0,
          "a null table installs nothing and creates no entry");
    CHECK(reg.install_tenant_slots(1, t, 77), "the first install succeeds");
    CHECK(reg.tenants(1) == t && reg.size() == 1, "the entry holds the table");
    kv_region_entry seen;
    CHECK(reg.lookup(1, seen) && seen.tenant_key == 77, "and the key it was built for");

    auto other = table_of({ 20 });
    CHECK(!reg.install_tenant_slots(1, other, 88), "a second install over a held table is refused");
    CHECK(reg.tenants(1) == t && reg.lookup(1, seen) && seen.tenant_key == 77, "and changes nothing");
    CHECK(reg.install_tenant_slots(2, other, 88) && reg.tenants(2) == other, "another context's entry is separate");

    auto taken = reg.take_tenant_slots(1);
    CHECK(taken == t && reg.tenants(1) == nullptr, "take moves the table out");
    CHECK(reg.size() == 1 && !reg.lookup(1, seen), "an entry left empty is erased");
    CHECK(reg.take_tenant_slots(1) == nullptr, "a second take finds nothing");
    CHECK(reg.take_tenant_slots(9) == nullptr, "a take of an unknown context finds nothing");

    // the key follows the table unless a published section still carries one
    runtime_context_section sec;
    sec.tenant_key = 5;
    CHECK(reg.install_tenant_slots(3, table_of({ 1 }), 5), "install");
    (void) reg.set_published_section(3, std::make_shared<const runtime_context_section>(sec));
    (void) reg.take_tenant_slots(3);
    CHECK(reg.lookup(3, seen) && seen.tenant_key == 5 && seen.published != nullptr && seen.tenants == nullptr,
          "a published section keeps its key when the table goes");
    (void) reg.drop_published_section(3);
    CHECK(reg.size() == 1, "and the entry goes with its last holder");

    // while a table is held its key stands over a section's, and when the table goes the key is the section's
    runtime_context_section other_sec;
    other_sec.tenant_key = 7;
    CHECK(reg.install_tenant_slots(14, table_of({ 1 }), 5), "install");
    (void) reg.set_published_section(14, std::make_shared<const runtime_context_section>(other_sec));
    CHECK(reg.lookup(14, seen) && seen.tenant_key == 5, "the held table's key stands over the section's");
    (void) reg.take_tenant_slots(14);
    CHECK(reg.lookup(14, seen) && seen.tenant_key == 7 && seen.tenants == nullptr,
          "after the take the key is the published section's own");
    (void) reg.drop_published_section(14);

    // a take from an entry that still holds other things clears the key it described
    kv_region_entry holds_extent;
    holds_extent.extents.push_back(plain_handle());
    (void) reg.publish(5, std::move(holds_extent));
    CHECK(reg.install_tenant_slots(5, table_of({ 1 }), 31), "install beside extents");
    (void) reg.take_tenant_slots(5);
    CHECK(reg.lookup(5, seen) && seen.extents.size() == 1 && seen.tenant_key == 0,
          "the take cleared the key and kept the extents");

    // a take where there is no table leaves the entry as it was, an empty one included
    (void) reg.publish(6, kv_region_entry{});
    const size_t before = reg.size();
    CHECK(reg.take_tenant_slots(6) == nullptr && reg.size() == before, "a take with no table changes nothing");

    // an entry that already holds extents is not recreated by an install
    kv_region_entry with_extent;
    with_extent.extents.push_back(plain_handle());
    (void) reg.publish(4, std::move(with_extent));
    CHECK(reg.install_tenant_slots(4, table_of({ 1 }), 9) && reg.lookup(4, seen) && seen.extents.size() == 1,
          "an install into an entry with extents keeps them");
    CHECK(kv_lock_witness::violations() == 0, "the witness saw no lock-order violation");
}

void case_registry_drops_outside_the_lock() {
    kv_region_registry reg;
    bool               dropped = false, under = true;
    auto               t = std::make_shared<kv_tenant_slots>();
    (void) t->add(HOST, 0, std::make_shared<watched>(&dropped, &under), 8);
    CHECK(reg.install_tenant_slots(1, std::move(t), 1), "install");
    CHECK(!dropped, "the registry keeps the table alive");
    {
        auto taken = reg.take_tenant_slots(1);
        CHECK(!dropped, "a take hands the table out alive: the drop is the caller's");
    }
    CHECK(dropped && !under, "the caller's drop ran with the leaf lock released");
}

}  // namespace

int main() {
    case_scope_is_per_thread();
    case_open_status_names_why();
    case_claims_made_is_validated();
    case_a_scope_left_open_is_counted();
    case_a_dropped_record_releases_its_slot();
    case_out_of_order_is_served_not_reported();
    case_a_sparse_slot_set_is_fully_reachable();
    case_claim_takes_the_lowest_free_slot();
    case_claim_refusals();
    case_release_is_exact();
    case_claim_keeps_the_memory();
    case_registry_install_and_take();
    case_registry_drops_outside_the_lock();
    std::printf("test-tenant-claim-scope: all cases passed\n");
    return 0;
}
