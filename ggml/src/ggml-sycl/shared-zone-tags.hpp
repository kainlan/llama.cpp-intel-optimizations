// MIT license
// SPDX-License-Identifier: MIT
//
// Lifetime classes of the shared KV+WEIGHT arena zone (llama.cpp-moua), as
// the tlsf_allocator block tags that carry them.  Weights front-carve, the
// optional (yieldable) tenants sit at the weight frontier, and every
// context-lifetime tenant top-carves from the high end; frontier_walk() reads
// these tags to find what a yield can merge into the gap.  Defined once here so
// the allocator's callers and its tests cannot disagree about the values.

#ifndef GGML_SYCL_SHARED_ZONE_TAGS_HPP
#define GGML_SYCL_SHARED_ZONE_TAGS_HPP

#include <cstdint>

namespace ggml_sycl {

enum shared_zone_tag : uint8_t {
    // 0 is tlsf_allocator's "untagged": a free block, an unknown offset, or an
    // allocate() caller that passed no tag.  It never passes a frontier walk.
    SHARED_ZONE_TAG_UNTAGGED = 0,
    SHARED_ZONE_TAG_WEIGHT   = 1,
    SHARED_ZONE_TAG_OPTIONAL = 2,
    SHARED_ZONE_TAG_CONTEXT  = 3,
};

static_assert(SHARED_ZONE_TAG_OPTIONAL != SHARED_ZONE_TAG_UNTAGGED,
              "frontier_walk treats tag 0 as untagged, so the yieldable class must not be 0");

// The lifetime class a request to the shared zone carries explicitly, because
// the zone it names and its alloc_role cannot separate them (CONTEXT and
// TRANSIENT are both alloc_role::COMPUTE; many non-weight requests name WEIGHT
// only to stay out of the tail zones).  UNSET derives the class from the role:
// WEIGHT gives WEIGHT and every other role gives TRANSIENT.
//
// Enumerators carry the enum prefix (the repo's enum convention), which also
// keeps OPTIONAL clear of the empty OPTIONAL macro windows.h defines.
enum class shared_zone_lifetime : uint8_t {
    SHARED_ZONE_LIFETIME_UNSET,
    SHARED_ZONE_LIFETIME_WEIGHT,
    SHARED_ZONE_LIFETIME_OPTIONAL,
    SHARED_ZONE_LIFETIME_KV_REGION,
    SHARED_ZONE_LIFETIME_CONTEXT,
    SHARED_ZONE_LIFETIME_TRANSIENT,
    // The B50 tail-zone lever: a TRANSIENT request placed on the weight side.
    SHARED_ZONE_LIFETIME_WEIGHT_SIDE_TRANSIENT,
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_SHARED_ZONE_TAGS_HPP
