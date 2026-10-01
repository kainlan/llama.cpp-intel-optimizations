// MIT license
// SPDX-License-Identifier: MIT
//
// The production reading of one shared-zone TLSF into the geometry kv_region_fit
// consumes (llama.cpp-moua §2.3.1, §2.4.1; L4 1b): the frontier walk, then the
// whole-TLSF census below it cut into runs of free blocks and releasable
// optional tenants.  SYCL-free, so the host tests drive it against the same
// allocator the zone model drives (tests/kv-region-test-model.hpp::snapshot,
// which builds the same two fields from its own bookkeeping; the agreement
// cases in test-kv-geometry-census.cpp pin them equal).
//
// It is a pure function of the allocator and of one predicate, and it runs only
// where frontier_walk() and block_census() may: under the arena group mutex,
// inside the snapshot's copy-out.  The predicate is jehw's
// optional_layout_yieldable_locked, evaluated under the cache locks it needs;
// the census asks it for allocated optional tenants only, never for a free
// block or a block of another tag (H7q: no second reclaimability test).
//
// What it does not fill: takes_kv, takes_weight_named, retained_runs,
// pending_ranges, own_ranges and reservations are the caller's, because they
// come from the zone's plan and the reservation store, not from the allocator.
#pragma once

#include "kv-runtime-demotion.hpp"
#include "shared-zone-tags.hpp"
#include "tlsf-allocator.hpp"

#include <functional>
#include <vector>

namespace ggml_sycl {

// jehw's predicate for the allocated optional tenant at `offset`.
using kv_yieldable_fn = std::function<bool(size_t offset)>;

inline zone_block kv_zone_block_from_extent(const tlsf_allocator::extent & e, const kv_yieldable_fn & yieldable) {
    zone_block b;
    b.offset          = e.offset;
    b.size            = e.size;
    b.free            = e.free;
    b.optional_tenant = !e.free && e.tag == SHARED_ZONE_TAG_OPTIONAL;
    b.yieldable       = b.optional_tenant && yieldable(e.offset);
    return b;
}

// The census below `floor` as runs: one per maximal low-to-high stretch of free
// blocks and yieldable optional tenants, broken by any other block (a weight, a
// context-side block, an optional tenant the predicate refuses).  Pure over the
// block list, so a hand-built census can reach states an allocator cannot.
inline std::vector<std::vector<zone_block>> kv_side_runs_from_census(const std::vector<tlsf_allocator::extent> & census,
                                                                     size_t                                      floor,
                                                                     const kv_yieldable_fn & yieldable) {
    std::vector<std::vector<zone_block>> runs;
    std::vector<zone_block>              run;
    for (const tlsf_allocator::extent & e : census) {
        if (e.offset >= floor) {
            break;
        }
        const zone_block b          = kv_zone_block_from_extent(e, yieldable);
        const bool       releasable = b.free || (b.optional_tenant && b.yieldable);
        if (releasable) {
            run.push_back(b);
        } else if (!run.empty()) {
            runs.push_back(run);
            run.clear();
        }
    }
    // On a census read off an allocator no run is open here: the frontier walk passes
    // every free block and optional tenant, so the block just under the floor is never
    // releasable.  The flush keeps the function total for any block list.
    if (!run.empty()) {
        runs.push_back(run);
    }
    return runs;
}

// The frontier walk below `anchor` (tlsf_allocator::no_anchor for the region
// end), highest block first, and the census below its floor as runs.  `anchor`
// is no_anchor or the offset of an allocated block, as in frontier_walk(); a
// stale offset inside a free block would not clip the block that holds it.
inline tlsf_geometry kv_geometry_from_tlsf(const tlsf_allocator &  tlsf,
                                           size_t                  anchor,
                                           const kv_yieldable_fn & yieldable) {
    tlsf_geometry g;
    // no_anchor is SIZE_MAX, so with no walk the whole census is below the floor.
    size_t        floor = anchor;
    for (const tlsf_allocator::extent & e : tlsf.frontier_walk(anchor, SHARED_ZONE_TAG_OPTIONAL)) {
        g.frontier.push_back(kv_zone_block_from_extent(e, yieldable));
        floor = e.offset;
    }
    g.side_runs = kv_side_runs_from_census(tlsf.block_census(), floor, yieldable);
    return g;
}

}  // namespace ggml_sycl
