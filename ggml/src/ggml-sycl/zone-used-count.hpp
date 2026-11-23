#pragma once

// How a VRAM arena zone's `used` counter follows the allocator that serves it. Host-only (no SYCL header), so
// tests/test-zone-sizing.cpp checks the rule against a real tlsf_allocator.
//
// KV and WEIGHT can share one physical allocator: in a single-chunk arena WEIGHT has none of its own and allocates
// from KV's, and in an N-chunk arena both allocate from the weight chunks. The allocator's used() is then the two
// zones' bytes together, so storing it into either counter makes that zone report the other zone's bytes as its own,
// and a free stored into the other zone's counter never lowers the zone that held the block (a freed model then
// reads as live WEIGHT bytes). Each counter therefore moves by exactly what the allocator's used() moved by in that one
// operation, block rounding included, and the sum of the counters of the zones sharing an allocator is its used().
//
// A free lowers the counter of the zone the allocation was RECORDED under (the exact-allocation record), never the
// zone the caller names: a caller can name either zone of a shared allocator, and only the record says which counter
// the block was added to. zone_used_free() therefore takes the record's counter and has no parameter for the caller's
// zone.
//
// Both calls run under the lock that serializes the allocator (the arena's allocator-group mutex), which is what keeps
// the before/after reads of used() about this one operation.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// Allocate `size` bytes at `align` from `allocator` for the zone whose counter is `zone_used`, and add to that counter
// exactly what the allocator's used() grew by. Returns the allocator's offset, or SIZE_MAX with nothing charged. The
// zone's count after the charge is written to `zone_used_after` when it is given (a peak profile records it).
template <typename allocator_type>
size_t zone_used_allocate(allocator_type &      allocator,
                          std::atomic<size_t> & zone_used,
                          size_t                size,
                          size_t                align,
                          size_t *              zone_used_after = nullptr) {
    const size_t before = allocator.used();
    const size_t offset = allocator.allocate(size, align);
    if (offset == SIZE_MAX) {
        return SIZE_MAX;
    }
    const size_t delta = allocator.used() - before;
    const size_t total = zone_used.fetch_add(delta, std::memory_order_relaxed) + delta;
    if (zone_used_after) {
        *zone_used_after = total;
    }
    return offset;
}

// Free the block at `offset` back to `allocator` and take what the allocator's used() fell by off `record_zone_used`,
// the counter of the zone the allocation was recorded under. A free the allocator does not recognise changes nothing.
template <typename allocator_type>
void zone_used_free(allocator_type & allocator, size_t offset, std::atomic<size_t> & record_zone_used) {
    const size_t before = allocator.used();
    allocator.free(offset);
    record_zone_used.fetch_sub(before - allocator.used(), std::memory_order_relaxed);
}

}  // namespace ggml_sycl
