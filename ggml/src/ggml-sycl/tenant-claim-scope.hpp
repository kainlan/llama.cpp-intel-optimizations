#pragma once

// The claim scope of a context's tenant slots (llama.cpp-moua L4 step 3c; moua design 2.3.2,
// "the host-pinned tier").
//
// A context reserves its host-tier slots at its first publish and its registry entry holds them.
// llama opens a claim scope around the allocation of the compute buffer it measured those slots
// for; inside the scope the SYCL_Host buffer type's alloc_buffer no longer allocates, it claims a
// slot of the entry's table and builds the buffer over the slot's memory.  Outside a scope the
// buffer type keeps its own allocation path, which is what llama's output buffer, the control
// vectors and a host-resident LoRA base take.
//
// The scope is a thread_local, like kv_region_scope: the buffers it covers are allocated
// synchronously on the opening thread, so a concurrent context on another thread never sees it,
// and a nested open is refused rather than stacked (a second scope would hide which context the
// claim belongs to).  The claim itself is the table's (kv_tenant_slots::claim): it checks the size
// against the slot's cap, marks the slot, allocates nothing and logs nothing.  A buffer's free is
// not on the opening thread, so the release goes through the claim record and not through the
// scope.
//
// The index is the lowest slot of the cohort no live claim holds.  A buffer allocated in order
// and freed in order claims index 0, 1, 2 and so on, so the index is the buffer type's live-object
// count; a free out of order frees its own index, which the next claim takes again, and the count
// never names a slot another buffer still holds.
//
// This header names no device and no SYCL type, so a host test builds it.

#include "kv-region-registry.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace ggml_sycl {

// A live claim: the table it was made against (kept alive by this record), the slot, the token
// that alone can release it, and the slot's handle copied out at the claim so the memory outlives
// a table that is dropped while the buffer is still in use.
struct tenant_claim {
    std::shared_ptr<kv_tenant_slots> slots;
    std::string                      cohort;
    uint32_t                         index      = 0;
    uint64_t                         generation = 0;  // 0 until a claim succeeds
    uint64_t                         wait_event = 0;  // the slot's last release event, for the claimant to chain on
    kv_region_handle                 owner;

    bool live() const { return generation != 0; }
};

enum class tenant_claim_status : uint8_t {
    OK,
    NO_SCOPE,   // no scope is open on this thread
    NO_SLOT,    // the table carries no free slot of the cohort: every index is claimed or none was planned
    OVER_PLAN,  // the lowest free slot is smaller than the request
};

struct tenant_claim_outcome {
    tenant_claim_status status = tenant_claim_status::NO_SCOPE;
    uint32_t            index  = 0;  // OK: the claimed index; OVER_PLAN: the index that was too small
    size_t              cap    = 0;  // OVER_PLAN: that slot's cap
};

class tenant_claim_scope {
  public:
    // The state `open` returns an opaque pointer to.  One per thread.
    struct state {
        bool                             open = false;
        std::shared_ptr<kv_tenant_slots> slots;
        size_t                           claims = 0;  // claims that succeeded through this scope
    };

    // Opens the scope on this thread over `slots`.  Null, and nothing changed, when a scope is
    // already open or `slots` is null.
    static state * open(std::shared_ptr<kv_tenant_slots> slots) {
        state & s = tls();
        if (s.open || !slots) {
            return nullptr;
        }
        s.open   = true;
        s.slots  = std::move(slots);
        s.claims = 0;
        return &s;
    }

    // Closes the scope `scope` names.  False, and nothing changed, for anything but this thread's
    // open scope (a stale pointer, another thread's, null).  The table's last drop, if the scope
    // held it, runs here with no registry lock held.
    static bool close(state * scope) {
        state & s = tls();
        if (scope == nullptr || scope != &s || !s.open) {
            return false;
        }
        std::shared_ptr<kv_tenant_slots> dropped = std::move(s.slots);
        s.open                                   = false;
        s.slots.reset();
        return true;
    }

    static bool active() { return tls().open; }

    static size_t claims_made(const state * scope) { return scope == nullptr ? 0 : scope->claims; }

    // Claims the lowest free slot of `cohort` for `bytes`, on the scope this thread has open.
    // `out` is replaced only on OK.
    static tenant_claim_outcome claim(const std::string & cohort, size_t bytes, tenant_claim & out) {
        state & s = tls();
        if (!s.open) {
            return { tenant_claim_status::NO_SCOPE, 0, 0 };
        }
        for (uint32_t index = 0;; ++index) {
            // The table checks a size against the cap before it checks the claim state, so a held slot
            // that is too small for the request would answer OVER_PLAN for a slot that is not free to
            // be checked at all.  Only a free slot is asked.
            if (s.slots->claimed(cohort, index)) {
                continue;
            }
            const kv_claim c = s.slots->claim(cohort, index, bytes);
            switch (c.result) {
                case kv_claim_result::OK:
                    {
                        tenant_claim made;
                        made.slots      = s.slots;
                        made.cohort     = cohort;
                        made.index      = index;
                        made.generation = c.generation;
                        made.wait_event = c.wait_event;
                        made.owner      = s.slots->handle(cohort, index);
                        out             = std::move(made);
                        ++s.claims;
                        return { tenant_claim_status::OK, index, 0 };
                    }
                case kv_claim_result::ALREADY_CLAIMED:
                    continue;
                case kv_claim_result::OVER_PLAN:
                    return { tenant_claim_status::OVER_PLAN, index, s.slots->cap(cohort, index) };
                case kv_claim_result::NO_SLOT:
                    return { tenant_claim_status::NO_SLOT, index, 0 };
            }
        }
    }

    // Releases a live claim, recording `release_event` for the next claimant to chain on, and
    // empties the record.  False when the record holds no live claim or the token is stale (the
    // slot was replaced, or the claim was already released).  Callable from any thread.
    static bool release(tenant_claim & claim, uint64_t release_event = 0) {
        if (!claim.live() || !claim.slots) {
            return false;
        }
        tenant_claim moved = std::move(claim);
        claim              = tenant_claim{};
        return moved.slots->release_claim(moved.cohort, moved.index, moved.generation, release_event);
        // `moved` drops here: the table and the slot handle go with no slot lock held
    }

  private:
    static state & tls() {
        thread_local state s;
        return s;
    }
};

}  // namespace ggml_sycl
