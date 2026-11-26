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
// written back by the acquire when GGML_SYCL_LEASE_HOLDERS=1 and 0 otherwise; the release lowers that slot.
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

constexpr size_t CHUNK_LEASE_HOLDER_SLOTS  = 64;
constexpr int    CHUNK_LEASE_HOLDER_FRAMES = 6;

// One holder: the site and copy flag of the leases it counts, and the call stack that took them, innermost first. The
// first frame is the address the tag recorded (the code that took the lease); the rest are its callers.
struct chunk_lease_holder {
    chunk_lease_site site                              = chunk_lease_site::UNTAGGED;
    bool             copy                              = false;
    int              n_frames                          = 0;
    const void *     frames[CHUNK_LEASE_HOLDER_FRAMES] = {};
};

inline bool chunk_lease_holder_same(const chunk_lease_holder & a, const chunk_lease_holder & b) {
    if (a.site != b.site || a.copy != b.copy || a.n_frames != b.n_frames) {
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
// lookup. Slot 0 names no holder: it counts the leases taken with no address and those taken after every slot was
// claimed. Used only with the knob on, where each take already walks the stack, so a mutex costs nothing that matters;
// it is a leaf lock, taken alone. Constant-initialised, so it is usable from the first take to the last report.
class chunk_lease_holder_table {
  public:
    // The slot that counts a lease `tag` describes, taken with the call stack `stack` (`depth` return addresses,
    // innermost first). The holder's frames start at the frame the tag recorded, so the lease cache's own frames are
    // not part of it; when that frame is not on the stack (the unwinder could not walk it), the holder is that address
    // alone. A tag with no address gets slot 0.
    uint16_t claim(const chunk_lease_tag & tag, void * const * stack, int depth) {
        if (tag.pc == nullptr) {
            return 0;
        }
        chunk_lease_holder holder;
        holder.site = tag.site;
        holder.copy = tag.copy;
        int first   = -1;
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
        for (size_t s = 1; s < used_; ++s) {
            if (chunk_lease_holder_same(slots_[s], holder)) {
                return static_cast<uint16_t>(s);
            }
        }
        if (used_ == CHUNK_LEASE_HOLDER_SLOTS) {
            return 0;
        }
        slots_[used_] = holder;
        return static_cast<uint16_t>(used_++);
    }

    // The holder in `slot`, false for slot 0 and a slot not claimed yet.
    bool read(uint16_t slot, chunk_lease_holder & out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot == 0 || slot >= used_) {
            return false;
        }
        out = slots_[slot];
        return true;
    }

  private:
    mutable std::mutex mutex_;
    chunk_lease_holder slots_[CHUNK_LEASE_HOLDER_SLOTS] = {};
    size_t             used_                            = 1;
};

}  // namespace ggml_sycl
