#pragma once

// The WEIGHT-zone allocation of one routed-expert triplet's members. Host-only (no SYCL header, no ggml), so the TLSF
// host test checks the rule against a real tlsf_allocator.
//
// A triplet's three members (gate, up, down) need not be one size: an AOS down projection whose row width is not a
// multiple of MATRIX_ROW_PADDING carries a padded tail its gate and up do not (921816 bytes against 921600 for a
// 640 x 2560 IQ4_NL expert). The load stages each tensor's device experts one after another, so the zone holds runs of
// one member size, and a swap evicts whole triplets and admits others. Were each member minted at its own size, an
// admitted member would split a freed run of the other size and leave a remainder a few grains short of any member;
// the zone then fills with holes no member fits while hundreds of MB read as free. Such a triplet's members are
// therefore minted at its slot, so any freed member's block holds any member of a triplet of the same slot.
//
// The slot is taken only where it is nearly free: the padding it adds, three slots less the members' own blocks, is at
// most 1/128 of those blocks. A triplet whose members differ by quant type (a Q4_K gate and up beside a Q5_K down)
// would pad by tens of percent, so its members keep their own sizes and the zone keeps its room.
//
// Three sizes describe one member, and only the last is an allocation size:
//   - logical bytes: the expert's own bytes, its tensor's bytes per expert. Every copy moves these and no more.
//   - staged bytes: the logical bytes and the row-padding tail the quantized kernels may read (staged_expert_bytes).
//   - member bytes: what the zone hands out, the owner's size and the cache entry's size: the slot when the triplet
//     has one, else the staged bytes. The bytes past the logical bytes read as zero, as direct staging leaves a padded
//     tail.

#include "tlsf-allocator.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// The share of a triplet's own blocks the slot may add as padding: at most 1/expert_triplet_slot_padding_divisor.
inline constexpr size_t expert_triplet_slot_padding_divisor = 128;

// The slot of a triplet whose members stage `staged` bytes each (gate, up, down): the largest of them, rounded up to
// the TLSF block grain, the size the zone hands each member. 0 when the triplet has no slot and each member keeps its
// own size: a member with no staged size (a layout that does not stage), a size the rounding or the sums would wrap,
// or a slot whose padding (three slots less the three rounded members) is above 1/128 of the rounded members.
inline size_t expert_triplet_slot_bytes(const size_t (&staged)[3]) {
    size_t slot = 0;
    size_t own  = 0;
    for (const size_t bytes : staged) {
        const size_t rounded = tlsf_allocator::round_request(bytes, tlsf_allocator::block_grain);
        if (rounded == 0 || own > SIZE_MAX - rounded) {
            return 0;
        }
        slot = rounded > slot ? rounded : slot;
        own += rounded;
    }
    if (slot > SIZE_MAX / 3) {
        return 0;
    }
    // padding <= own / divisor, exactly: the padding is a whole number of bytes, so it is at most the floor.
    const size_t padding = 3 * slot - own;
    return padding <= own / expert_triplet_slot_padding_divisor ? slot : 0;
}

// The bytes the zone hands member `role` (0 gate, 1 up, 2 down) of a triplet whose members stage `staged` bytes: the
// triplet's slot when it has one, else the member's own staged bytes. The one size the load stages a member at, a
// swap's admit and refill mint it at, and the cache checks its device storage against.
inline size_t expert_triplet_member_bytes(const size_t (&staged)[3], int role) {
    const size_t slot = expert_triplet_slot_bytes(staged);
    return slot != 0 ? slot : staged[role];
}

// What a triplet whose members stage `staged` bytes asks of the zone when all three are minted: its slot (0 when it has
// none) and its members' bytes summed (expert_triplet_member_bytes each), three slots for a triplet with a slot.
struct expert_triplet_zone_ask {
    size_t slot  = 0;
    size_t total = 0;
};

inline expert_triplet_zone_ask expert_triplet_zone_ask_of(const size_t (&staged)[3]) {
    expert_triplet_zone_ask ask;
    ask.slot = expert_triplet_slot_bytes(staged);
    for (int r = 0; r < 3; ++r) {
        const size_t member = ask.slot != 0 ? ask.slot : staged[r];
        ask.total           = member > SIZE_MAX - ask.total ? SIZE_MAX : ask.total + member;
    }
    return ask;
}

}  // namespace ggml_sycl
