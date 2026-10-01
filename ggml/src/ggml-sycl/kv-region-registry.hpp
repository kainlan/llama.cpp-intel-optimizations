// MIT license
// SPDX-License-Identifier: MIT
//
// The KV region registry, the region scope and the transaction guard
// (llama.cpp-moua §2.5, §2.3.1, §2.4.2), factored into a SYCL-free header so
// the logic can be tested on the host (H8, H9) before the unified cache wires
// it.  Nothing here allocates device memory or reads a plan: a "handle" is a
// shared_ptr<void> whose last drop frees the extent, which is what a mem_handle
// is to the registry, and the plan is a callback.
//
// What the header models, and the section that states it:
//   * kv_region_registry   ContextId -> {extent handles, shape key, frozen
//                          n_ubatch, layout, tenant slots, tenant key}, under
//                          kv_region_mutex_, a strictly leaf lock (§2.3.1);
//   * kv_region_scope      the thread_local scope the llama-side constructor
//                          opens around create_memory (§2.5);
//   * kv_region_attach_buffer / kv_region_layer_on_device
//                          the attach and the residency answer (§2.5);
//   * kv_tenant_slots      the published slot table a context claims from (§2.3.2);
//   * kv_lock_witness, kv_witnessed_mutex, kv_replan_token
//                          the lock order the model checks on every path: L0 (the
//                          outermost-only re-plan token) before L1, then L3, then L5
//                          (the arena group mutexes and the slot-state lock);
//   * kv_txn_guard         the two-phase rollback of §2.4.2: phase 1 clears this
//                          call's pending ranges under the group mutex with no L1
//                          held; phase 2 drops the handles with no lock held;
//   * kv_region_release    the teardown proc: idempotent, L0 and no L1, noexcept.
//
// Left to later steps, not modelled here: the later-load weight-slot store
// (H9 (4)'s ledger step, which is L4+L6 work), the per-slot claim spin lock (the
// model's slot-state lock is one table-wide mutex), and any tie-break between two
// L5 peers (see kv_lock_witness).

#ifndef GGML_SYCL_KV_REGION_REGISTRY_HPP
#define GGML_SYCL_KV_REGION_REGISTRY_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ggml_sycl {

using kv_context_id    = uint64_t;
using kv_region_handle = std::shared_ptr<void>;  // the last drop releases the extent

// ---------------------------------------------------------------------------
// The abort channel.  A violation of a contract the caller must keep (a nested
// scope, a lock failure in a destructor) is a GGML_ABORT in the backend; here it
// goes through a handler so a test can record it and continue.  With no handler
// it prints and aborts.
// ---------------------------------------------------------------------------
using kv_region_abort_fn = void (*)(const char * message);

inline std::atomic<kv_region_abort_fn> & kv_region_abort_handler() {
    static std::atomic<kv_region_abort_fn> handler{ nullptr };
    return handler;
}

// Returns only when a handler is installed and returns itself.
inline void kv_region_abort(const std::string & message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    if (kv_region_abort_fn h = kv_region_abort_handler().load()) {
        h(message.c_str());
        return;
    }
    std::abort();
}

// ---------------------------------------------------------------------------
// The lock witness.  Ranks follow the contract's table (§2.3.1): the re-plan
// token L0, the tensor-inventory lock L1, the registry lock L3 (a leaf), then L5,
// which both the arena group mutexes and the slot-state locks hold (the release
// proc's step 2 text calls the slot-state lock L5).  Acquiring a lock whose rank
// is not above every lock the thread holds is an order violation; acquiring
// anything while holding the registry lock is a leaf violation.  Two L5 peers held
// together are therefore a violation here: the production witness's instance
// tie-break between L5 peers is not modelled, and no path in the model needs it.  The witness
// counts and remembers violations instead of aborting, so a test can assert on
// them, and a "no lock held" check reads the same stack.
// ---------------------------------------------------------------------------
enum kv_lock_rank : int {
    KV_LOCK_L0_REPLAN     = 0,
    KV_LOCK_L1_INVENTORY  = 1,
    KV_LOCK_L3_REGISTRY   = 3,
    KV_LOCK_L5_GROUP      = 5,
    KV_LOCK_L5_SLOT_STATE = 5,
};

class kv_lock_witness {
  public:
    static void acquired(int rank, const char * name) {
        std::vector<held> & s = stack();
        for (const held & h : s) {
            if (h.rank == KV_LOCK_L3_REGISTRY) {
                violation(std::string("leaf: ") + name + " acquired while holding " + h.name);
            } else if (h.rank >= rank) {
                violation(std::string("order: ") + name + " acquired while holding " + h.name);
            }
        }
        s.push_back({ rank, name });
    }

    static void released(int rank) {
        std::vector<held> & s = stack();
        for (size_t i = s.size(); i-- > 0;) {
            if (s[i].rank == rank) {
                s.erase(s.begin() + (long) i);
                return;
            }
        }
        violation("released a lock this thread does not hold");
    }

    static bool holds(int rank) {
        for (const held & h : stack()) {
            if (h.rank == rank) {
                return true;
            }
        }
        return false;
    }

    static size_t held_count() { return stack().size(); }

    // The locks held other than `rank`: the release proc holds L0 to its end, so
    // its "no lock held" checks (a drop, the retain call) read this.
    static size_t held_count_besides(int rank) {
        size_t n = 0;
        for (const held & h : stack()) {
            n += h.rank != rank ? 1 : 0;
        }
        return n;
    }

    static size_t violations() { return count().load(); }

    static std::string last() {
        std::lock_guard<std::mutex> g(text_mutex());
        return text();
    }

    static void reset() {
        std::lock_guard<std::mutex> g(text_mutex());
        count().store(0);
        text().clear();
    }

    // A check a caller makes that is not a lock acquire: records the same way.
    static void violation(const std::string & what) {
        {
            std::lock_guard<std::mutex> g(text_mutex());
            text() = what;
        }
        count().fetch_add(1);
    }

  private:
    struct held {
        int          rank;
        const char * name;
    };

    static std::vector<held> & stack() {
        thread_local std::vector<held> s;
        return s;
    }

    static std::atomic<size_t> & count() {
        static std::atomic<size_t> c{ 0 };
        return c;
    }

    static std::mutex & text_mutex() {
        static std::mutex m;
        return m;
    }

    static std::string & text() {
        static std::string t;
        return t;
    }
};

// A mutex that reports to the witness, and counts its lock and unlock calls.
class kv_witnessed_mutex {
  public:
    kv_witnessed_mutex(int rank, const char * name) : rank_(rank), name_(name) {}

    void lock() {
        if (fail_next_.exchange(false)) {
            // std::mutex::lock can throw; the destructors that take these locks
            // must name the failure and abort rather than propagate (r4 m14).
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
        }
        kv_lock_witness::acquired(rank_, name_);
        m_.lock();
        locks_.fetch_add(1);
    }

    bool try_lock() {
        if (!m_.try_lock()) {
            return false;
        }
        kv_lock_witness::acquired(rank_, name_);
        locks_.fetch_add(1);
        return true;
    }

    void unlock() {
        unlocks_.fetch_add(1);
        m_.unlock();
        kv_lock_witness::released(rank_);
    }

    size_t lock_count() const { return locks_.load(); }

    size_t unlock_count() const { return unlocks_.load(); }

    // Test seam: the next lock() throws, which is how a lock failure inside a
    // destructor or the release proc is forced.
    void fail_next_lock() { fail_next_.store(true); }

  private:
    int                 rank_;
    const char *        name_;
    std::mutex          m_;
    std::atomic<size_t> locks_{ 0 };
    std::atomic<size_t> unlocks_{ 0 };
    std::atomic<bool>   fail_next_{ false };
};

// L0, the process-global re-plan mutex (rulings §E.2, §L0R).  An outermost-only
// token: nested acquires on a thread that holds it do not lock, and the
// outermost release unlocks.  The teardown proc takes it with
// acquire_outermost(), which aborts on a same-thread re-entry instead of
// hanging.
class kv_replan_token {
  public:
    void acquire() {
        int & d = depth();
        if (d == 0) {
            kv_lock_witness::acquired(KV_LOCK_L0_REPLAN, "L0 replan token");
            m_.lock();
            locks_.fetch_add(1);
        }
        ++d;
    }

    // For a body that must never run nested: a re-entry is a bug, not a hold.
    // Returns false, after reporting, when this thread already holds the token.
    bool acquire_outermost() {
        if (depth() > 0) {
            kv_region_abort("[REPLAN-TOKEN] release proc entered with L0 held");
            return false;
        }
        acquire();
        return true;
    }

    void release() {
        int & d = depth();
        if (d == 0) {
            kv_lock_witness::violation("released the re-plan token this thread does not hold");
            return;
        }
        if (--d == 0) {
            unlocks_.fetch_add(1);
            m_.unlock();
            kv_lock_witness::released(KV_LOCK_L0_REPLAN);
        }
    }

    bool held_by_this_thread() { return depth() > 0; }

    // A probe from the calling thread: true when the token was free (and is
    // released again at once).  The positive control of every "parked holder" arm.
    bool probe_free() {
        if (!m_.try_lock()) {
            return false;
        }
        m_.unlock();
        return true;
    }

    size_t lock_count() const { return locks_.load(); }

    size_t unlock_count() const { return unlocks_.load(); }

  private:
    // The depth is per thread and per token.
    int & depth() {
        thread_local std::vector<std::pair<const kv_replan_token *, int>> depths;
        for (auto & d : depths) {
            if (d.first == this) {
                return d.second;
            }
        }
        depths.push_back({ this, 0 });
        return depths.back().second;
    }

    std::mutex          m_;
    std::atomic<size_t> locks_{ 0 };
    std::atomic<size_t> unlocks_{ 0 };
};

// RAII holder of the token.
class kv_replan_scope {
  public:
    explicit kv_replan_scope(kv_replan_token & t) : t_(t) { t_.acquire(); }

    ~kv_replan_scope() { t_.release(); }

    kv_replan_scope(const kv_replan_scope &)             = delete;
    kv_replan_scope & operator=(const kv_replan_scope &) = delete;

  private:
    kv_replan_token & t_;
};

// ---------------------------------------------------------------------------
// The tenant slot table (§2.3.2, §2.5): per (cohort, index), the reserved slot's
// handle, its cap and its claim state.  One shared_ptr per published region; each
// backend context caches it for claims.  A claim is a slice lease: it changes
// nothing in the zone, it only checks the size against the cap and marks the slot.
// ---------------------------------------------------------------------------
enum class kv_claim_result : uint8_t {
    OK,
    NO_SLOT,          // the plan reserved no slot at (cohort, index)
    OVER_PLAN,        // size above the slot's cap
    ALREADY_CLAIMED,  // a previous claim is still live
};

// The slot-state retention of one ring row's last generation: the owner handle
// the row's record kept (retained_owners[slot]) and the event that fences the
// work that read it (done_events[slot]).  A superseded or torn-down row frees
// only after that event, so it travels to retain_handles_until_event.
struct kv_slot_retention {
    kv_region_handle owner;
    uint64_t         done_event = 0;
};

class kv_tenant_slots {
  public:
    void add(const std::string & cohort, uint32_t index, kv_region_handle handle, size_t cap) {
        slot replaced;  // dropped after the lock: a replaced slot's handle may be a final drop
        {
            std::lock_guard<kv_witnessed_mutex> g(mu_);
            slot &                              s = slots_[{ cohort, index }];
            replaced                              = std::move(s);
            s.handle                              = std::move(handle);
            s.cap                                 = cap;
        }
    }

    kv_claim_result claim(const std::string & cohort, uint32_t index, size_t size) {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = slots_.find({ cohort, index });
        if (it == slots_.end()) {
            return kv_claim_result::NO_SLOT;
        }
        if (size > it->second.cap) {
            return kv_claim_result::OVER_PLAN;
        }
        if (it->second.claimed) {
            return kv_claim_result::ALREADY_CLAIMED;
        }
        it->second.claimed = true;
        return kv_claim_result::OK;
    }

    void release_claim(const std::string & cohort, uint32_t index) {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = slots_.find({ cohort, index });
        if (it != slots_.end()) {
            it->second.claimed = false;
        }
    }

    bool claimed(const std::string & cohort, uint32_t index) const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = slots_.find({ cohort, index });
        return it != slots_.end() && it->second.claimed;
    }

    bool any_claimed() const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        for (const auto & s : slots_) {
            if (s.second.claimed) {
                return true;
            }
        }
        return false;
    }

    size_t cap(const std::string & cohort, uint32_t index) const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = slots_.find({ cohort, index });
        return it == slots_.end() ? 0 : it->second.cap;
    }

    // The slot's handle, copied out (a refcount increment, never a final drop).
    kv_region_handle handle(const std::string & cohort, uint32_t index) const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = slots_.find({ cohort, index });
        return it == slots_.end() ? nullptr : it->second.handle;
    }

    // Record the row's retention for the generation just recorded and return the
    // previous generation's, moved out: the caller hands it to
    // retain_handles_until_event(previous.done_event) before the new record is
    // used, and never drops it under the lock.  A row with no slot keeps nothing.
    kv_slot_retention exchange_retention(const std::string & cohort, uint32_t index, kv_slot_retention next) {
        kv_slot_retention                   previous;
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = slots_.find({ cohort, index });
        if (it == slots_.end()) {
            return previous;
        }
        previous             = std::move(it->second.retention);
        it->second.retention = std::move(next);
        return previous;  // NRVO'd out: the caller's drop is after the guard's unlock
    }

    // The release proc's step 2: move out every row's slot-state retention, for
    // the caller to hand to retain_handles_until_event with no lock held.  The
    // slots themselves stay: they drop with the entry's batch.
    void take_retentions(std::vector<kv_slot_retention> & out) {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        for (auto & s : slots_) {
            if (s.second.retention.owner) {
                out.push_back(std::move(s.second.retention));
                s.second.retention = kv_slot_retention();
            }
        }
    }

    // Move out the slots no claim holds, for the caller to drop with no lock held
    // (the tenant-only path's step (i)).  Returns false and moves nothing when a
    // slot is still claimed: that is a [CONTEXT-PLAN-BUG] for the caller.
    bool take_unclaimed(std::vector<kv_region_handle> & out) {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        for (const auto & s : slots_) {
            if (s.second.claimed) {
                return false;
            }
        }
        for (auto & s : slots_) {
            out.push_back(std::move(s.second.handle));
        }
        slots_.clear();
        return true;
    }

    size_t size() const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        return slots_.size();
    }

  private:
    struct slot {
        kv_region_handle  handle;
        size_t            cap     = 0;
        bool              claimed = false;
        kv_slot_retention retention;
    };

    // One table-wide mutex.  The contract's claim path is a per-slot leaf spin
    // lock; the rank (L5) and the "move out under the lock, drop after it" rule are
    // the same, and the model has no concurrent-claim arm that needs the finer lock.
    mutable kv_witnessed_mutex                       mu_{ KV_LOCK_L5_SLOT_STATE, "slot-state lock" };
    std::map<std::pair<std::string, uint32_t>, slot> slots_;
};

// ---------------------------------------------------------------------------
// The registry entry of one context on one device (§2.5).
// ---------------------------------------------------------------------------
struct kv_layer_slice {
    size_t extent         = 0;
    size_t slot_offset    = 0;
    size_t slot_size      = 0;
    size_t sidecar_offset = SIZE_MAX;
    size_t sidecar_size   = 0;
};

using kv_region_layout = std::map<uint32_t, kv_layer_slice>;  // device layer -> its slot

struct kv_region_entry {
    std::vector<kv_region_handle>    extents;              // one owner-first handle per extent
    uint64_t                         shape_key       = 0;  // the KV shape and the head-slot demand it was planned for
    size_t                           frozen_n_ubatch = 0;
    kv_region_layout                 layout;
    std::shared_ptr<kv_tenant_slots> tenants;         // shared with every backend context that claims
    uint64_t                         tenant_key = 0;  // the tenant section's digest

    bool empty() const { return extents.empty() && layout.empty() && !tenants; }
};

// One registry per device, under kv_region_mutex_ (§2.3.1): rank L3, strictly
// leaf.  Every access is a short copy-in or copy-out: lookups copy handles out,
// inserts move them in, and erases move the entry out and leave the drop to the
// caller, who runs it with no lock held.
class kv_region_registry {
  public:
    // `out` is replaced after the lock is released: whatever it held before can
    // be a final drop, and a final drop never runs under kv_region_mutex_.
    bool lookup(kv_context_id ctx, kv_region_entry & out) const {
        kv_region_entry copy;
        {
            std::lock_guard<kv_witnessed_mutex> g(mu_);
            auto                                it = entries_.find(ctx);
            if (it == entries_.end()) {
                return false;
            }
            copy = it->second;
        }
        out = std::move(copy);
        return true;
    }

    bool has_layer(kv_context_id ctx, uint32_t layer) const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = entries_.find(ctx);
        return it != entries_.end() && it->second.layout.count(layer) != 0;
    }

    bool layer_slice(kv_context_id ctx, uint32_t layer, kv_layer_slice & out) const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = entries_.find(ctx);
        if (it == entries_.end()) {
            return false;
        }
        auto l = it->second.layout.find(layer);
        if (l == it->second.layout.end()) {
            return false;
        }
        out = l->second;
        return true;
    }

    std::shared_ptr<kv_tenant_slots> tenants(kv_context_id ctx) const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = entries_.find(ctx);
        return it == entries_.end() ? nullptr : it->second.tenants;
    }

    // Publish `entry` for `ctx`.  Returns the entry it replaces (empty when
    // there was none), moved out: the caller drops it after every lock is released.
    kv_region_entry publish(kv_context_id ctx, kv_region_entry && entry) {
        kv_region_entry                     previous;
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        auto                                it = entries_.find(ctx);
        if (it != entries_.end()) {
            previous   = std::move(it->second);
            it->second = std::move(entry);
        } else {
            entries_.emplace(ctx, std::move(entry));
        }
        return previous;
    }

    // Erase `ctx`'s entry, moved out.  False when there is none.
    bool take(kv_context_id ctx, kv_region_entry & out) {
        kv_region_entry moved;
        {
            std::lock_guard<kv_witnessed_mutex> g(mu_);
            auto                                it = entries_.find(ctx);
            if (it == entries_.end()) {
                return false;
            }
            moved = std::move(it->second);
            entries_.erase(it);
        }
        out = std::move(moved);
        return true;
    }

    size_t size() const {
        std::lock_guard<kv_witnessed_mutex> g(mu_);
        return entries_.size();
    }

    // Test seam: the next lock() on this registry throws.
    void fail_next_lock() { mu_.fail_next_lock(); }

  private:
    mutable kv_witnessed_mutex               mu_{ KV_LOCK_L3_REGISTRY, "kv_region_mutex_" };
    std::map<kv_context_id, kv_region_entry> entries_;
};

// ---------------------------------------------------------------------------
// The region scope (§2.5).  llama_context's constructor opens it around
// create_memory.  It is a thread_local: the memory is created synchronously on
// the constructing thread, so concurrent contexts on other threads cannot see it.
// A nested begin is an error.
// ---------------------------------------------------------------------------
class kv_region_scope {
  public:
    // Returns false, after reporting, when a scope is already open: the refused
    // begin changes nothing, so its caller must not end() the outer scope.
    static bool begin(kv_context_id id) {
        State & s = state();
        if (s.open) {
            kv_region_abort("kv region scope: nested begin for context " + std::to_string(id) +
                            " inside the scope of context " + std::to_string(s.id));
            return false;
        }
        s.open = true;
        s.id   = id;
        return true;
    }

    static void end() { state().open = false; }

    static bool active(kv_context_id * id = nullptr) {
        const State & s = state();
        if (s.open && id != nullptr) {
            *id = s.id;
        }
        return s.open;
    }

    // The RAII guard llama_context wraps create_memory in, so the scope ends on a throw too.
    class guard {
      public:
        explicit guard(kv_context_id id) : opened_(begin(id)) {}

        // Only a guard that opened the scope closes it.
        ~guard() {
            if (opened_) {
                end();
            }
        }

        guard(const guard &)             = delete;
        guard & operator=(const guard &) = delete;

      private:
        bool opened_;
    };

  private:
    struct State {
        bool          open = false;
        kv_context_id id   = 0;
    };

    static State & state() {
        thread_local State s;
        return s;
    }
};

// A device layer the plan puts on a device, and the device whose region holds its slot.
struct kv_layer_plan {
    uint32_t layer        = 0;
    int      owner_device = 0;
};

struct kv_attached_slot {
    uint32_t       layer        = 0;
    int            owner_device = 0;
    kv_context_id  context      = 0;
    kv_layer_slice slice;
};

struct kv_attach_result {
    bool                          ok       = false;
    bool                          no_scope = false;  // an arena device with device-planned layers and no open scope
    kv_context_id                 context  = 0;
    std::vector<kv_attached_slot> slots;
    std::vector<uint32_t>         missing;  // device-planned layers the owner's region has no slot for
};

// The attach of one tiered KV buffer inside the scope (§2.5): for each
// device-planned layer with planned owner `o`, take slot `l` from registry(o)
// [ContextId], even when the buffer's device differs.  No pop and no count, so
// iSWA's two buffers attach to the same region.  `registries[d]` is null for a
// device that reserves no regions.  `mid_attach` is the test seam that holds a
// thread inside the attach (g_test_block_next_region_attach).
//
// kv_region_attach_for is the body with the context id as a parameter, which the
// scoped entry point reads from the thread_local scope.  A test drives it with an
// id from a global to show what a process-global scope would do.
inline kv_attach_result kv_region_attach_for(kv_context_id                                   id,
                                             const std::vector<const kv_region_registry *> & registries,
                                             const std::vector<kv_layer_plan> &              layers,
                                             const std::function<void()> &                   mid_attach = {}) {
    kv_attach_result r;
    r.context = id;
    if (mid_attach) {
        mid_attach();
    }
    for (const kv_layer_plan & l : layers) {
        kv_attached_slot s;
        s.layer        = l.layer;
        s.owner_device = l.owner_device;
        s.context      = id;
        const kv_region_registry * reg =
            (l.owner_device >= 0 && (size_t) l.owner_device < registries.size()) ? registries[l.owner_device] : nullptr;
        if (reg == nullptr || !reg->layer_slice(id, l.layer, s.slice)) {
            r.missing.push_back(l.layer);
            continue;
        }
        r.slots.push_back(s);
    }
    r.ok = r.missing.empty();
    return r;
}

inline kv_attach_result kv_region_attach_buffer(const std::vector<const kv_region_registry *> & registries,
                                                const std::vector<kv_layer_plan> &              layers,
                                                const std::function<void()> &                   mid_attach = {}) {
    kv_context_id id = 0;
    if (!kv_region_scope::active(&id)) {
        kv_attach_result r;
        r.no_scope = true;
        return r;
    }
    return kv_region_attach_for(id, registries, layers, mid_attach);
}

// llama's residency answer for layer `il`, whose planned device is `d` (§2.5).
// Under an open scope and a device that reserves regions, the registry answers:
// device-resident iff registry(d)[ContextId] has a slot for `il`.  Otherwise (no
// scope, or `d` reserves none: no arena, an arena enabled but not active, or a
// mixed device set) the plan answers, as today.  `registry_d` is null for a
// device that reserves no regions.
template <typename PlanFn>
bool kv_region_layer_on_device(const kv_region_registry * registry_d, uint32_t il, PlanFn && plan) {
    kv_context_id id = 0;
    if (registry_d != nullptr && kv_region_scope::active(&id)) {
        return registry_d->has_layer(id, il);
    }
    return plan(il);
}

// ---------------------------------------------------------------------------
// The transaction guard (§2.4.2): a refusing step after any carve or pending
// range rolls back through it, in two phases.
//   phase 1  with no L1 held, take only the group mutex of each TLSF this call
//            recorded ranges on and clear them (re-recording a first context's
//            FIRST_CONTEXT range in the same section);
//   phase 2  drop every handle this call carved and every unused pre-minted
//            control, with no instrumented lock held.
// commit() disarms it.
// ---------------------------------------------------------------------------
class kv_txn_guard {
  public:
    struct range_clear {
        kv_witnessed_mutex *  group = nullptr;
        std::function<void()> clear;  // runs under the group mutex
    };

    ~kv_txn_guard() { abandon(); }

    kv_txn_guard()                                 = default;
    kv_txn_guard(const kv_txn_guard &)             = delete;
    kv_txn_guard & operator=(const kv_txn_guard &) = delete;

    void note_ranges(kv_witnessed_mutex * group, std::function<void()> clear) {
        ranges_.push_back({ group, std::move(clear) });
    }

    void note_handle(kv_region_handle h) { handles_.push_back(std::move(h)); }

    void note_unused_control(kv_region_handle h) { controls_.push_back(std::move(h)); }

    void commit() { armed_ = false; }

    bool armed() const { return armed_; }

    size_t handles() const { return handles_.size() + controls_.size(); }

    void abandon() {
        if (!armed_) {
            return;
        }
        armed_ = false;
        if (kv_lock_witness::holds(KV_LOCK_L1_INVENTORY)) {
            kv_lock_witness::violation("guard phase 1 ran with L1 held");
        }
        for (range_clear & r : ranges_) {
            try {
                std::lock_guard<kv_witnessed_mutex> g(*r.group);
                if (r.clear) {
                    r.clear();
                }
            } catch (const std::system_error &) {
                kv_region_abort("kv_region_txn rollback could not take the group mutex");
            }
        }
        ranges_.clear();
        if (kv_lock_witness::held_count() != 0) {
            kv_lock_witness::violation("guard phase 2 dropped handles with a lock held");
        }
        handles_.clear();
        controls_.clear();
    }

  private:
    bool                          armed_ = true;
    std::vector<range_clear>      ranges_;
    std::vector<kv_region_handle> handles_;
    std::vector<kv_region_handle> controls_;
};

// ---------------------------------------------------------------------------
// The teardown proc (§2.4.2 "Teardown"): run from the plan guard's destructor
// with the ContextId captured at create_exec.  Idempotent: a second call finds
// nothing.  It takes L0 (outermost-only) and no L1, holds L0 to the end, and runs
//   1. under kv_region_mutex_, move every (c, *) entry out into a local batch;
//   2. under each entry's slot-state lock, move out each ring row's slot-state
//      retention (the owner handle and its done event);
//   3. with no instrumented lock but L0 held, hand each retention to
//      `retain_until_event` with its event, then drop the batch.
// The batch's extents and slot handles are not fenced here: anything still queued
// holds its own lease (§2.3.2), so the last reference frees each block by refcount.
// noexcept: a failure to take a lock, a missing fence callback and an exception
// from it each abort with a named message.
// ---------------------------------------------------------------------------
using kv_retain_fn = std::function<void(kv_region_handle &&, uint64_t done_event)>;

inline void kv_region_release(const std::vector<kv_region_registry *> & registries,
                              kv_context_id                             ctx,
                              kv_replan_token &                         l0,
                              const kv_retain_fn &                      retain_until_event) noexcept {
    std::vector<kv_region_entry>   batch;
    std::vector<kv_slot_retention> retentions;
    if (!l0.acquire_outermost()) {
        return;  // the re-entry was reported, and the handler returned
    }
    try {
        for (kv_region_registry * reg : registries) {
            if (reg == nullptr) {
                continue;
            }
            kv_region_entry e;
            if (reg->take(ctx, e)) {
                batch.push_back(std::move(e));
            }
        }
        for (kv_region_entry & e : batch) {
            if (e.tenants) {
                e.tenants->take_retentions(retentions);
            }
        }
    } catch (const std::system_error &) {
        l0.release();
        kv_region_abort("kv region release: kv_region_mutex_ could not be taken in the release proc");
        return;
    } catch (...) {
        l0.release();
        kv_region_abort("kv region release: an exception escaped the release proc");
        return;
    }
    // Step 3: no instrumented lock but L0 is held from here.
    if (!retentions.empty() && !retain_until_event) {
        // Freeing them here would drop a row under the work that may still read it.
        kv_region_abort("kv region release: " + std::to_string(retentions.size()) +
                        " slot-state retention(s) and no retain_handles_until_event callback");
    } else {
        for (kv_slot_retention & r : retentions) {
            try {
                retain_until_event(std::move(r.owner), r.done_event);
            } catch (...) {
                kv_region_abort("kv region release: retain_handles_until_event threw in the release proc");
            }
        }
    }
    retentions.clear();
    batch.clear();
    l0.release();
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_KV_REGION_REGISTRY_HPP
