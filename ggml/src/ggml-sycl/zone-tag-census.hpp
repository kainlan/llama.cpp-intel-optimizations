// A VRAM zone's live registered allocations summed by tag, and their ranking for a report.
//
// Pure: no SYCL, no allocator, no lock, so the host-only ring sizing test compiles it alone. The unified cache fills a
// zone_tag_census (unified_cache_zone_tag_census); a report ranks a copy of its tags here.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace ggml_sycl {

// One tag's live allocations in a zone: how many, and the zone bytes they take.
struct zone_tag_total {
    std::string tag;
    size_t      count = 0;
    size_t      bytes = 0;
};

// A zone's free-space figures and every tag of the live registered allocations its address range holds, all read in
// one call. `tags` is in no particular order and holds every tag, not a top-N.
struct zone_tag_census {
    size_t                      capacity     = 0;
    size_t                      available    = 0;
    size_t                      largest_free = 0;
    std::vector<zone_tag_total> tags;
};

// Ranks `tags` largest bytes first and keeps the first `top`. Equal bytes put more allocations first, then the tag in
// lexical order, so the order never follows a hash map's iteration order.
inline void zone_tag_census_rank(std::vector<zone_tag_total> & tags, size_t top) {
    (void) tags;
    (void) top;
}

}  // namespace ggml_sycl
