// Who holds a lease on a VRAM arena chunk.
//
// A chunk lease keeps the arena from being destroyed, so an arena that a last release, a load start or a rebuild finds
// held has to say who holds it. The cache counts its chunk leases by the site that took them (the tag's site, always
// on), and with GGML_SYCL_LEASE_HOLDERS=1 also by the code that took them: each take records its own return address,
// and the cache, when the knob is on, files the take's call stack from that address outward in the holder table below
// and counts the lease in that stack's slot.
//
// SYCL-free, so a host-only test compiles the table on its own.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace ggml_sycl {

// Where a lease on a VRAM arena chunk was taken. A mem_handle keeps the site it acquired with, so its release lowers
// the count its acquire raised.
enum class chunk_lease_site : uint8_t {
    UNTAGGED,         // a caller of the cache's lease API that names no site (the device tests' probes)
    WEIGHT_SNAPSHOT,  // a WEIGHT handle minted from a lease snapshot
    WEIGHT_RESOLVE,   // a WEIGHT handle's resolve, which pins the chunk its pointer lies in
    CHUNK_VIEW,       // a handle made from a raw pointer for one use (from_chunk_ptr's default)
    TENSOR_EXTRA,     // a handle a tensor extra keeps for its data on a device
    EXPERT_DISPATCH,  // the lease an expert dispatch entry keeps for its raw pointer
    COUNT,
};

// The site a lease was taken at, and whether a copy of a handle took it. A copy acquires a lease of its own and keeps
// its source's site, so copies count apart from the handles they copy and the counts by site and copy sum to the
// chunks' leases.
//
// pc is the address the lease was taken for: the return address of the take (from_chunk_ptr, a weight lease snapshot,
// a weight resolve, a copy), so a copy names the code that copied, not its source. Recording it is one store per take
// whatever the knob says, and a release never reads it. holder is the holder-table slot the cache counted the lease in,
// written back by the acquire when GGML_SYCL_LEASE_HOLDERS=1 and 0 otherwise; the release lowers that slot. A move
// carries the tag with the lease; when the lease is counted in a named slot (the knob is on) the move also records its
// own address and the cache re-counts the lease under the mover's stack, so the report names the code that keeps it
// now as well as the take it came from. With the knob off holder is 0 and a move only compares it.
struct chunk_lease_tag {
    chunk_lease_site site   = chunk_lease_site::UNTAGGED;
    bool             copy   = false;
    const void *     pc     = nullptr;
    uint16_t         holder = 0;
};

// The site's name in the lines that print the counts.
inline const char * chunk_lease_site_name(chunk_lease_site site) {
    switch (site) {
        case chunk_lease_site::UNTAGGED:
            return "untagged";
        case chunk_lease_site::WEIGHT_SNAPSHOT:
            return "weight_snapshot";
        case chunk_lease_site::WEIGHT_RESOLVE:
            return "weight_resolve";
        case chunk_lease_site::CHUNK_VIEW:
            return "chunk_view";
        case chunk_lease_site::TENSOR_EXTRA:
            return "tensor_extra";
        case chunk_lease_site::EXPERT_DISPATCH:
            return "expert_dispatch";
        case chunk_lease_site::COUNT:
            break;
    }
    return "unknown";
}

// Whether GGML_SYCL_LEASE_HOLDERS=1 is set: read once per process, in unified-cache.cpp, and kept by each cache in a
// const member, so a take reads a member and never the environment.
bool chunk_lease_holders_enabled();

// The holder table's capacity, and the two slots that name no holder: slot 0 counts the leases taken with no address
// (a caller of the cache's lease API that keeps no tag), slot 1 those whose holder found the table full. A GPT-OSS run
// filled 63 named slots before its last copy (2026-11-26), so the table holds thousands; a full table is still told
// apart from a missing address.
constexpr size_t   CHUNK_LEASE_HOLDER_SLOTS      = 4096;
constexpr uint16_t CHUNK_LEASE_HOLDER_NO_ADDRESS = 0;
constexpr uint16_t CHUNK_LEASE_HOLDER_FULL       = 1;
constexpr uint16_t CHUNK_LEASE_HOLDER_FIRST      = 2;
constexpr int      CHUNK_LEASE_HOLDER_FRAMES     = 6;

// One holder: the site and copy flag of the leases it counts, and the call stack that took them, innermost first. The
// first frame is the address the tag recorded (the code that took the lease, or that moved it); the rest are its
// callers. A holder a move made names, in `origin`, the take's holder the lease came from (0 for a take), so the last
// keeper and the first taker are both known and repeated moves through one stack stay one holder.
struct chunk_lease_holder {
    chunk_lease_site site                              = chunk_lease_site::UNTAGGED;
    bool             copy                              = false;
    uint16_t         origin                            = 0;
    int              n_frames                          = 0;
    const void *     frames[CHUNK_LEASE_HOLDER_FRAMES] = {};
};

inline bool chunk_lease_holder_same(const chunk_lease_holder & a, const chunk_lease_holder & b) {
    if (a.site != b.site || a.copy != b.copy || a.origin != b.origin || a.n_frames != b.n_frames) {
        return false;
    }
    for (int i = 0; i < a.n_frames; ++i) {
        if (a.frames[i] != b.frames[i]) {
            return false;
        }
    }
    return true;
}

// The holders the process has seen, one slot per distinct holder, claimed in order and never freed, so a slot index
// names the same holder for the life of the process and a release can lower the slot its acquire raised without a
// lookup. Used only with the knob on, where each take already walks the stack, so a mutex costs nothing that matters;
// it is a leaf lock, taken alone. Constant-initialised, so it is usable from the first take to the last report.
class chunk_lease_holder_table {
  public:
    // The slot that counts a lease `tag` describes, taken (origin 0) or moved (origin: the take's holder) with the call
    // stack `stack` (`depth` return addresses, innermost first). The holder's frames start at the frame the tag
    // recorded, so the lease cache's own frames are not part of it; when that frame is not on the stack (the unwinder
    // could not walk it), the holder is that address alone. A tag with no address gets CHUNK_LEASE_HOLDER_NO_ADDRESS
    // and a new holder that finds the table full CHUNK_LEASE_HOLDER_FULL.
    uint16_t claim(const chunk_lease_tag & tag, void * const * stack, int depth, uint16_t origin = 0) {
        if (tag.pc == nullptr) {
            return CHUNK_LEASE_HOLDER_NO_ADDRESS;
        }
        chunk_lease_holder holder;
        holder.site   = tag.site;
        holder.copy   = tag.copy;
        holder.origin = origin;
        int first     = -1;
        for (int i = 0; i < depth; ++i) {
            if (stack[i] == tag.pc) {
                first = i;
                break;
            }
        }
        if (first < 0) {
            holder.frames[holder.n_frames++] = tag.pc;
        } else {
            for (int i = first; i < depth && holder.n_frames < CHUNK_LEASE_HOLDER_FRAMES; ++i) {
                holder.frames[holder.n_frames++] = stack[i];
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        // Open addressing over twice the slots, so a probe ends at an empty cell long before the table is full.
        size_t                      cell = hash(holder) % INDEX_CELLS;
        for (;;) {
            const uint16_t slot = index_[cell];
            if (slot == 0) {
                break;
            }
            if (chunk_lease_holder_same(slots_[slot], holder)) {
                return slot;
            }
            cell = (cell + 1) % INDEX_CELLS;
        }
        if (used_ == CHUNK_LEASE_HOLDER_SLOTS) {
            return CHUNK_LEASE_HOLDER_FULL;
        }
        slots_[used_] = holder;
        index_[cell]  = static_cast<uint16_t>(used_);
        return static_cast<uint16_t>(used_++);
    }

    // The holder in `slot`, false for the two unnamed slots and a slot not claimed yet.
    bool read(uint16_t slot, chunk_lease_holder & out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot < CHUNK_LEASE_HOLDER_FIRST || slot >= used_) {
            return false;
        }
        out = slots_[slot];
        return true;
    }

    // The take a lease counted in `slot` came from: the slot itself for a take, its origin for a move, 0 when the slot
    // names no holder.
    uint16_t origin_of(uint16_t slot) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot < CHUNK_LEASE_HOLDER_FIRST || slot >= used_) {
            return 0;
        }
        return slots_[slot].origin != 0 ? slots_[slot].origin : slot;
    }

  private:
    static constexpr size_t INDEX_CELLS = 2 * CHUNK_LEASE_HOLDER_SLOTS;

    static size_t hash(const chunk_lease_holder & h) {
        uint64_t v   = 1469598103934665603ULL;  // FNV-1a over the key's words
        auto     mix = [&v](uint64_t w) {
            v ^= w;
            v *= 1099511628211ULL;
        };
        mix(static_cast<uint64_t>(h.site));
        mix(h.copy ? 1 : 0);
        mix(h.origin);
        for (int i = 0; i < h.n_frames; ++i) {
            mix(reinterpret_cast<uintptr_t>(h.frames[i]));
        }
        return static_cast<size_t>(v);
    }

    mutable std::mutex mutex_;
    chunk_lease_holder slots_[CHUNK_LEASE_HOLDER_SLOTS] = {};
    uint16_t           index_[INDEX_CELLS]              = {};
    size_t             used_                            = CHUNK_LEASE_HOLDER_FIRST;
};

}  // namespace ggml_sycl
