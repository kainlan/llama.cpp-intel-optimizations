#pragma once

// Release rule for the legacy device pool: the free list pool_leg keeps when
// the VRAM arena is off (GGML_SYCL_VRAM_ARENA=0).
//
// Host-only: no SYCL types, so the rule is testable without a device. The pool
// passes its own slot and owner types; each carries a movable owning handle
// with valid().

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ggml_sycl {

enum class pool_legacy_release_result {
    CACHED,          // on the free list; the next alloc() may hand it out
    GRAPH_RETAINED,  // parked with the recording graph's retained handles
    DROPPED,         // free list full; its owner was released
    MISSING_OWNER,   // no allocation-time owner; the caller must fail closed
};

// Moves the owner of ptr out of active. Normally the block goes back on the
// free list, which is safe because later work on the pool's in-order queue
// runs after the work that used it.
//
// A recording command graph breaks that: it bakes the block's address into
// commands that run only when the graph is replayed. On the free list, the
// next alloc() would hand the same memory to other work while the graph still
// points into it; dropped, the memory itself would be freed under the graph.
// So a block freed during recording is parked in graph_retained with its owner
// and released when the graph is dropped, as the arena path does.
template <typename Slot, typename Owned, typename Handle>
pool_legacy_release_result pool_legacy_release(void *                              ptr,
                                               bool                                graph_recording,
                                               Slot *                              slots,
                                               int                                 n_slots,
                                               std::unordered_map<void *, Owned> & active,
                                               std::vector<Handle> &               graph_retained,
                                               size_t &                            pool_size) {
    auto it = active.find(ptr);
    // Ownership must have been retained at allocation time. A metadata lookup
    // is observation-only and cannot mint it here.
    if (it == active.end() || !it->second.handle.valid()) {
        return pool_legacy_release_result::MISSING_OWNER;
    }

    if (graph_recording) {
        pool_size -= it->second.size;
        graph_retained.push_back(std::move(it->second.handle));
        active.erase(it);
        return pool_legacy_release_result::GRAPH_RETAINED;
    }

    for (int i = 0; i < n_slots; ++i) {
        Slot & b = slots[i];
        if (b.ptr == nullptr) {
            b.ptr    = ptr;
            b.size   = it->second.size;
            b.handle = std::move(it->second.handle);
            active.erase(it);
            return pool_legacy_release_result::CACHED;
        }
    }

    pool_size -= it->second.size;
    it->second.handle = {};
    active.erase(it);
    return pool_legacy_release_result::DROPPED;
}

}  // namespace ggml_sycl
