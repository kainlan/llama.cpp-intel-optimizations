#pragma once

// The WEIGHT-zone allocation of one routed-expert triplet's members. Host-only (no SYCL header, no ggml), so the TLSF
// host test checks the rule against a real tlsf_allocator.
//
// A triplet's three members (gate, up, down) need not be one size: an AOS down projection whose row width is not a
// multiple of MATRIX_ROW_PADDING carries a padded tail its gate and up do not (921816 bytes against 921600 for a
// 640 x 2560 IQ4_NL expert). The load stages each tensor's device experts one after another, so the zone holds runs of
// one member size, and a swap evicts whole triplets and admits others. Were each member minted at its own size, an
// admitted member would split a freed run of the other size and leave a remainder a few grains short of any member;
// the zone then fills with holes no member fits while hundreds of MB read as free. Every member is therefore minted
// at its triplet's slot, so any freed member's block holds any member of a triplet of the same slot.
//
// Three sizes describe one member, and only the slot is an allocation size:
//   - logical bytes: the expert's own bytes, its tensor's bytes per expert. Every copy moves these and no more.
//   - staged bytes: the logical bytes and the row-padding tail the quantized kernels may read (staged_expert_bytes).
//   - slot bytes: what the zone hands out, the owner's size and the cache entry's size. The bytes past the logical
//     bytes read as zero, as direct staging leaves a padded tail.

#include "tlsf-allocator.hpp"

#include <cstddef>

namespace ggml_sycl {

// The slot of a triplet whose members stage `staged` bytes each (gate, up, down): the largest of them, rounded up to
// the TLSF block grain, the size the zone hands out for it. 0 when a member has no staged size (a layout that does
// not stage, so the triplet has no slot and each member keeps its own size) or when the rounding would wrap.
inline size_t expert_triplet_slot_bytes(const size_t (&staged)[3]) {
    size_t slot = 0;
    for (const size_t bytes : staged) {
        const size_t rounded = tlsf_allocator::round_request(bytes, tlsf_allocator::block_grain);
        if (rounded == 0) {
            return 0;
        }
        slot = rounded > slot ? rounded : slot;
    }
    return slot;
}

}  // namespace ggml_sycl
