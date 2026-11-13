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
#include <cstdio>
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
    std::sort(tags.begin(), tags.end(), [](const zone_tag_total & a, const zone_tag_total & b) {
#if defined(GGML_SYCL_ZONE_TAG_CENSUS_MUTANT_SMALLEST_FIRST)
        // The positive control's defect: smallest bytes first.
        if (a.bytes != b.bytes) {
            return a.bytes < b.bytes;
        }
#else
        if (a.bytes != b.bytes) {
            return a.bytes > b.bytes;
        }
#endif
        if (a.count != b.count) {
            return a.count > b.count;
        }
        return a.tag < b.tag;
    });
    if (tags.size() > top) {
        tags.erase(tags.begin() + static_cast<std::ptrdiff_t>(top), tags.end());
    }
}

// `head`, then the census's figures in bytes: the zone's size, free bytes and largest free block; used (size - free,
// the allocator's view); attributed, the bytes of every allocation the census counted; unattributed=, used minus
// attributed, signed, so a census that missed a holder shows on the line instead of reading as complete (the
// allocator's alignment and block headers make it a little positive on a complete one); the allocation and tag counts;
// and the `top` largest tags with their count and bytes. The tags are ranked on a copy; the totals count every tag,
// not only those printed.
inline std::string zone_tag_census_line(const char * head, const zone_tag_census & census, size_t top) {
    size_t allocations = 0;
    size_t attributed  = 0;
    for (const zone_tag_total & t : census.tags) {
        allocations += t.count;
        attributed += t.bytes;
    }
    const size_t    used         = census.capacity > census.available ? census.capacity - census.available : 0;
    const long long unattributed = static_cast<long long>(used) - static_cast<long long>(attributed);

    std::vector<zone_tag_total> ranked = census.tags;
    zone_tag_census_rank(ranked, top);

    char figures[320];
    std::snprintf(figures, sizeof(figures),
                  " size=%zu free=%zu largest_free=%zu used=%zu attributed=%zu unattributed=%lld allocations=%zu "
                  "tags=%zu; top %zu by bytes:",
                  census.capacity, census.available, census.largest_free, used, attributed, unattributed, allocations,
                  census.tags.size(), ranked.size());
    std::string line = head;
    line += figures;
    if (ranked.empty()) {
        line += " none";
        return line;
    }
    for (size_t i = 0; i < ranked.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " count=%zu bytes=%zu", ranked[i].count, ranked[i].bytes);
        line += i == 0 ? " " : ", ";
        line += ranked[i].tag;
        line += item;
    }
    return line;
}

}  // namespace ggml_sycl
