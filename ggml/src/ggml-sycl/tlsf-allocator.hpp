// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// TLSF (Two-Level Segregated Fit) sub-allocator for VRAM arena zones.
//
// Provides O(1) guaranteed alloc/free with zero writes to the managed
// memory region. All metadata is stored in host memory (block_meta vector
// + bitmaps). The allocator returns OFFSETS into the managed region;
// callers convert offsets to device pointers via arena_base + offset.
//
// This design avoids CPU writes to device VRAM (which would hang on
// discrete GPUs where sycl::malloc_device memory is not CPU-accessible).
//
// Reference: M. Masmano et al., "TLSF: A New Dynamic Memory Allocator for
// Real-Time Systems" (2004).

#ifndef GGML_SYCL_TLSF_ALLOCATOR_HPP
#define GGML_SYCL_TLSF_ALLOCATOR_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// Always-active assertion (not compiled away by NDEBUG)
#ifndef TLSF_ASSERT
#define TLSF_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "TLSF_ASSERT failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)
#endif

namespace ggml_sycl {

class tlsf_allocator {
  public:
    // Initialize with the SIZE of the managed region [0, size).
    // No pointer to the managed region is needed — all bookkeeping is
    // in host memory.  min_alloc_size: minimum block granularity
    // (256 bytes for GPU alignment).
    tlsf_allocator(size_t size);

    // O(1) allocation. Returns OFFSET into the managed region.
    // Returns SIZE_MAX on failure.
    // Caller computes device_ptr = arena_base + offset.
    // alignment must be a power of 2 and >= min_alloc_size.
    // tag is an opaque caller label kept with the block until it is freed.
    size_t allocate(size_t size, size_t alignment = 256, uint8_t tag = 0);

    // ------------------------------------------------------------------
    // Placement by physical position (llama.cpp-moua).
    //
    // allocate() takes whichever free block the size classes offer, so the
    // address a block lands at says nothing about its lifetime.  The calls
    // below place a block relative to an ANCHOR instead: the allocated block
    // at offset `anchor`, or the physical end of the region when the anchor
    // is no_anchor.  The GAP below an anchor is the free block immediately
    // under it, if there is one.  All of them round like allocate() and
    // return SIZE_MAX, changing nothing, when the gap is missing or too small.
    // ------------------------------------------------------------------
    static constexpr size_t no_anchor = SIZE_MAX;

    // Size of the gap below `anchor`; 0 when the block below is allocated,
    // when there is no block below, or when `anchor` is not an allocated block.
    size_t gap_below(size_t anchor) const;

    // Carve the TOP of the gap below the region's end: the block ends exactly
    // at the end of the managed region.  Same as allocate_below(no_anchor, ...).
    size_t allocate_top(size_t size, size_t alignment = 256, uint8_t tag = 0);

    // Carve the TOP of the gap below `anchor`: the block ends exactly where
    // the anchor begins, so a chain of calls packs downward with no holes.
    size_t allocate_below(size_t anchor, size_t size, size_t alignment = 256, uint8_t tag = 0);

    // Carve the BOTTOM of the gap below `anchor`: the block starts exactly
    // where the allocated block under the gap ends.
    size_t allocate_gap_front(size_t anchor, size_t size, size_t alignment = 256, uint8_t tag = 0);

    // Tag of the allocated block at `offset`; 0 for a free or unknown offset.
    uint8_t tag_at(size_t offset) const;

    struct extent {
        size_t  offset;
        size_t  size;
        bool    free;
        uint8_t tag;
    };

    // The blocks physically below `anchor`, highest first, for as long as each
    // one is free or carries `pass_tag`.  The walk stops at the first block
    // that is neither, so the entries are exactly what releasing every
    // `pass_tag` block in the walk would merge into the gap below `anchor`.
    std::vector<extent> frontier_walk(size_t anchor, uint8_t pass_tag) const;

    // O(1) free with automatic coalescing of physically adjacent blocks.
    // offset must have been returned by allocate().
    void free(size_t offset);

    // Bulk reset — returns ALL memory to the pool as a single free block.
    // Faster than freeing everything individually. O(1).
    void reset();

    // Query methods
    size_t used() const;                // Bytes currently allocated
    size_t available() const;           // Bytes available for allocation
    size_t largest_free_block() const;  // Size of largest contiguous free block

    // No header overhead in the managed region (metadata is external).
    static constexpr size_t header_overhead() { return 0; }

  private:
    // ------------------------------------------------------------------
    // Block metadata: stored in HOST memory (blocks_ vector).
    // No data is written to the managed region.
    // ------------------------------------------------------------------
    struct block_meta {
        size_t offset;      // Start offset in the managed region
        size_t size;        // Usable size (excludes metadata overhead)
        bool   free;        // Is this block free?
        int    prev_block;  // Index of physically previous block (-1 if first)
        int    next_block;  // Index of physically next block (-1 if last)
        int    next_free;   // Index of next block in same (fl,sl) free list (-1 if tail)
        int    prev_free;   // Index of prev block in same (fl,sl) free list (-1 if head)
        uint8_t tag;         // Caller label of an allocated block (0 when free)
    };

    // ------------------------------------------------------------------
    // TLSF configuration constants
    // ------------------------------------------------------------------
    static constexpr int    FL_OFFSET      = 8;  // log2(256) = 8
    static constexpr int    SL_COUNT       = 16;
    static constexpr int    SL_BITS        = 4;  // log2(SL_COUNT)
    // 2^(FL_OFFSET + FL_COUNT - 1) must be >= max arena size.
    // For 16 GB: need FL_OFFSET + FL_COUNT - 1 >= 34, so FL_COUNT >= 27.
    static constexpr int    FL_COUNT       = 28;
    // Minimum block size.  Must be >= 256 for GPU alignment.
    static constexpr size_t MIN_BLOCK_SIZE = 256;

    // ------------------------------------------------------------------
    // Host-side metadata
    // ------------------------------------------------------------------
    std::vector<block_meta>         blocks_;           // Grows as blocks are split
    std::vector<int>                free_pool_;        // Recycled block IDs from coalescing
    std::unordered_map<size_t, int> offset_to_block_;  // offset → block index (O(1) lookup for free())

    // Two-level index structures (all fixed-size, no heap allocation)
    uint32_t fl_bitmap_;                       // Which FL classes have free blocks
    uint32_t sl_bitmap_[FL_COUNT];             // Which SL classes have free blocks per FL
    int      free_lists_[FL_COUNT][SL_COUNT];  // Block ID head of free list per (fl, sl)

    size_t total_size_;                        // Size of managed region
    size_t used_;                              // Currently allocated bytes
    int    last_block_;                        // Block at the physical end of the region (-1 if none)

    // ------------------------------------------------------------------
    // Core TLSF operations
    // ------------------------------------------------------------------

    // Allocate a new block_meta slot. Returns index into blocks_.
    int alloc_block_id();

    // Free block directly below `anchor` (see the public placement calls), or -1.
    int gap_block(size_t anchor) const;

    // Shared body of the placement calls: carve `size` from the top or the
    // front of the gap below `anchor`.
    size_t carve_gap(size_t anchor, size_t size, size_t alignment, uint8_t tag, bool from_top);

    // Map a block size to (fl, sl) indices.
    void mapping(size_t size, int & fl, int & sl) const {
        if (size < MIN_BLOCK_SIZE) {
            size = MIN_BLOCK_SIZE;
        }
        fl = 0;
        if (size >= (1ull << FL_OFFSET)) {
            fl = (int) (sizeof(size_t) * 8 - 1 - __builtin_clzll(size)) - FL_OFFSET;
            sl = (int) ((size >> (fl + FL_OFFSET - SL_BITS)) - SL_COUNT);
            if (fl >= FL_COUNT) {
                fl = FL_COUNT - 1;
                sl = SL_COUNT - 1;
            }
            if (sl >= SL_COUNT) {
                sl = SL_COUNT - 1;
            }
            if (sl < 0) {
                sl = 0;
            }
        } else {
            fl = 0;
            sl = (int) (size / (MIN_BLOCK_SIZE / SL_COUNT));
            if (sl >= SL_COUNT) {
                sl = SL_COUNT - 1;
            }
        }
    }

    // Map size to (fl, sl) then round UP to find the first class that
    // is guaranteed to have blocks >= size.
    void mapping_search(size_t size, int & fl, int & sl) const {
        if (size >= (1ull << FL_OFFSET)) {
            int t = (1 << ((int) (sizeof(size_t) * 8 - 1 - __builtin_clzll(size)) - SL_BITS)) - 1;
            size += t;
        }
        mapping(size, fl, sl);
    }

    // Find a free block from class (fl, sl) or higher. O(1) via bitmaps.
    // Returns block ID, or -1 if no block available.
    int find_suitable(int fl, int sl) {
        // Search current FL class from sl upward
        uint32_t sl_map = sl_bitmap_[fl] & (~0u << sl);
        if (sl_map == 0) {
            // No block in this FL class. Search higher FL classes.
            uint32_t fl_map = fl_bitmap_ & (~0u << (fl + 1));
            if (fl_map == 0) {
                return -1;  // No free block large enough
            }
            fl     = __builtin_ctz(fl_map);
            sl_map = sl_bitmap_[fl];
        }
        sl = __builtin_ctz(sl_map);
        return free_lists_[fl][sl];
    }

    // Remove block from its free list. Update bitmaps if list becomes empty.
    void remove_free(int block_id) {
        auto & block = blocks_[block_id];
        int    fl, sl;
        mapping(block.size, fl, sl);

        int prev = block.prev_free;
        int next = block.next_free;

        if (prev >= 0) {
            blocks_[prev].next_free = next;
        }
        if (next >= 0) {
            blocks_[next].prev_free = prev;
        }

        // If this was the head of the list, update free_lists
        if (free_lists_[fl][sl] == block_id) {
            free_lists_[fl][sl] = next;
            if (next < 0) {
                sl_bitmap_[fl] &= ~(1u << sl);
                if (sl_bitmap_[fl] == 0) {
                    fl_bitmap_ &= ~(1u << fl);
                }
            }
        }

        block.prev_free = -1;
        block.next_free = -1;
    }

    // Insert block into the appropriate free list. Update bitmaps.
    void insert_free(int block_id) {
        auto & block = blocks_[block_id];
        int    fl, sl;
        mapping(block.size, fl, sl);

        int head        = free_lists_[fl][sl];
        block.next_free = head;
        block.prev_free = -1;
        if (head >= 0) {
            blocks_[head].prev_free = block_id;
        }

        free_lists_[fl][sl] = block_id;
        fl_bitmap_ |= (1u << fl);
        sl_bitmap_[fl] |= (1u << sl);
    }

    // Split block: carve off `size` bytes from the front, return remainder
    // to free list if large enough.
    void split_block(int block_id, size_t size) {
        // Read values before alloc_block_id() — it may reallocate blocks_ vector.
        size_t block_offset = blocks_[block_id].offset;
        size_t block_size   = blocks_[block_id].size;
        int    block_next   = blocks_[block_id].next_block;

        if (block_size < size + MIN_BLOCK_SIZE) {
            return;
        }
        size_t remain = block_size - size;
        if (remain >= MIN_BLOCK_SIZE) {
            int    rest_id  = alloc_block_id();  // May reallocate blocks_!
            auto & rest     = blocks_[rest_id];
            rest.offset     = block_offset + size;
            rest.size       = remain;
            rest.free       = true;
            rest.prev_block = block_id;
            rest.next_block = block_next;
            rest.prev_free  = -1;
            rest.next_free  = -1;
            rest.tag        = 0;

            if (block_next >= 0) {
                blocks_[block_next].prev_block = rest_id;
            } else {
                last_block_ = rest_id;
            }

            blocks_[block_id].size       = size;
            blocks_[block_id].next_block = rest_id;

            insert_free(rest_id);
        }
    }

    // Coalesce block with physically adjacent free neighbors.
    // Returns the block ID of the merged block.
    int coalesce(int block_id) {
        // Merge with next physical block if free
        int next_id = blocks_[block_id].next_block;
        if (next_id >= 0 && blocks_[next_id].free) {
            remove_free(next_id);
            blocks_[block_id].size += blocks_[next_id].size;

            int nn_id = blocks_[next_id].next_block;
            if (nn_id >= 0) {
                blocks_[nn_id].prev_block = block_id;
            } else {
                last_block_ = block_id;
            }
            blocks_[block_id].next_block = nn_id;

            // next_id was free — its offset is NOT in offset_to_block_ (only allocated blocks are mapped)
            free_pool_.push_back(next_id);
        }

        // Merge with previous physical block if free
        if (blocks_[block_id].prev_block >= 0) {
            int prev_id = blocks_[block_id].prev_block;
            if (blocks_[prev_id].free) {
                remove_free(prev_id);
                blocks_[prev_id].size += blocks_[block_id].size;

                // Update the block after the merged block
                int nn_id = blocks_[block_id].next_block;
                if (nn_id >= 0) {
                    blocks_[nn_id].prev_block = prev_id;
                } else {
                    last_block_ = prev_id;
                }
                blocks_[prev_id].next_block = nn_id;

                // block_id's offset was already erased in free() before coalesce
                // Recycle the block ID
                free_pool_.push_back(block_id);

                block_id = prev_id;
            }
        }

        return block_id;
    }
};

// ------------------------------------------------------------------
// Constructor: initialize all bitmaps, create one large free block
// ------------------------------------------------------------------
inline tlsf_allocator::tlsf_allocator(size_t size) : fl_bitmap_(0), total_size_(size), used_(0), last_block_(-1) {
    // Initialize all free lists to empty (-1)
    std::memset(sl_bitmap_, 0, sizeof(sl_bitmap_));
    for (int i = 0; i < FL_COUNT; ++i) {
        for (int j = 0; j < SL_COUNT; ++j) {
            free_lists_[i][j] = -1;
        }
    }

    if (size < MIN_BLOCK_SIZE) {
        return;  // Arena too small
    }

    // Create one block_meta covering the entire region
    int    first_id  = alloc_block_id();
    auto & first     = blocks_[first_id];
    first.offset     = 0;
    first.size       = size;
    first.free       = true;
    first.prev_block = -1;
    first.next_block = -1;
    first.prev_free  = -1;
    first.next_free  = -1;
    first.tag        = 0;
    last_block_      = first_id;

    insert_free(first_id);
}

// ------------------------------------------------------------------
// Allocate a new block_meta slot
// ------------------------------------------------------------------
inline int tlsf_allocator::alloc_block_id() {
    if (!free_pool_.empty()) {
        int id = free_pool_.back();
        free_pool_.pop_back();
        return id;
    }
    int id = static_cast<int>(blocks_.size());
    blocks_.emplace_back();
    return id;
}

// ------------------------------------------------------------------
// allocate: O(1) via bitmap search, returns offset (SIZE_MAX on failure)
// ------------------------------------------------------------------
inline size_t tlsf_allocator::allocate(size_t size, size_t alignment, uint8_t tag) {
    if (size == 0) {
        return SIZE_MAX;
    }

    // Every block offset must stay a multiple of MIN_BLOCK_SIZE.  The pool's
    // first block starts at offset 0 and split_block() places the remainder at
    // (offset + size), so that holds only while every allocated size is itself a
    // multiple of MIN_BLOCK_SIZE.  Rounding by the caller's `alignment` instead
    // breaks it: most callers leave alignment at the allocator default of 64, and
    // the resulting remainders are 64- but not 128/256-aligned, so later
    // allocations hand back offsets that violate the alignment this allocator
    // promises (llama.cpp-f8ws).  Rounding by the larger of the two also
    // satisfies any alignment <= MIN_BLOCK_SIZE for free.
    const size_t granularity = alignment > MIN_BLOCK_SIZE ? alignment : MIN_BLOCK_SIZE;
    size                     = (size + granularity - 1) & ~(granularity - 1);

    // Offsets carry MIN_BLOCK_SIZE alignment and no more, so a larger request
    // cannot be honoured here; such callers must align above this allocator.
    TLSF_ASSERT(alignment <= MIN_BLOCK_SIZE && "TLSF only supports alignment <= MIN_BLOCK_SIZE (256)");
    size_t adjusted = size;

    int fl, sl;
    mapping_search(adjusted, fl, sl);

    int block_id = find_suitable(fl, sl);
    if (block_id < 0) {
        // mapping_search rounds the size UP to guarantee all blocks in the
        // returned SL class are >= size. When the block sits exactly at an SL
        // boundary this rounding pushes into the next (empty) SL class. Fall
        // back to the exact (non-rounding) mapping. find_suitable searches from
        // (fl, sl) upward so blocks in higher SL classes are included.
        // Guard: the first block returned may be in a lower SL sub-class whose
        // size is in [lower_bound, size), so verify before committing.
        mapping(adjusted, fl, sl);
        block_id = find_suitable(fl, sl);
        if (block_id < 0 || blocks_[block_id].size < size) {
            return SIZE_MAX;
        }
    }

    remove_free(block_id);
    blocks_[block_id].free = false;
    blocks_[block_id].tag  = tag;

    // Split off excess
    split_block(block_id, size);

    used_ += blocks_[block_id].size;

    // Record offset → block_id mapping for O(1) free()
    offset_to_block_[blocks_[block_id].offset] = block_id;

    // Check the invariant where it is produced rather than only documenting it:
    // every consumer that turns this offset into a device pointer relies on it.
    TLSF_ASSERT((blocks_[block_id].offset % MIN_BLOCK_SIZE) == 0 && "TLSF block offset lost MIN_BLOCK_SIZE alignment");

    return blocks_[block_id].offset;
}

// ------------------------------------------------------------------
// free: O(1) via bitmap insert + coalesce with neighbors
// ------------------------------------------------------------------
inline void tlsf_allocator::free(size_t offset) {
    if (offset == SIZE_MAX) {
        return;
    }

    auto it = offset_to_block_.find(offset);
    if (it == offset_to_block_.end()) {
        return;  // Invalid free — offset not found
    }
    int block_id = it->second;

    used_ -= blocks_[block_id].size;

    blocks_[block_id].free = true;
    blocks_[block_id].tag  = 0;

    // Remove offset mapping (will be re-added if block is re-allocated)
    offset_to_block_.erase(it);

    // Coalesce with adjacent free blocks
    block_id = coalesce(block_id);

    insert_free(block_id);
}

// ------------------------------------------------------------------
// reset: O(1) bulk deallocation
// ------------------------------------------------------------------
inline void tlsf_allocator::reset() {
    fl_bitmap_ = 0;
    std::memset(sl_bitmap_, 0, sizeof(sl_bitmap_));
    for (int i = 0; i < FL_COUNT; i++) {
        for (int j = 0; j < SL_COUNT; j++) {
            free_lists_[i][j] = -1;
        }
    }
    used_ = 0;
    offset_to_block_.clear();
    free_pool_.clear();
    blocks_.clear();
    last_block_ = -1;

    if (total_size_ < MIN_BLOCK_SIZE) {
        return;
    }

    // Create one block covering the entire region
    int    first_id  = alloc_block_id();
    auto & first     = blocks_[first_id];
    first.offset     = 0;
    first.size       = total_size_;
    first.free       = true;
    first.prev_block = -1;
    first.next_block = -1;
    first.prev_free  = -1;
    first.next_free  = -1;
    first.tag        = 0;
    last_block_      = first_id;

    insert_free(first_id);
}

// ------------------------------------------------------------------
// Query methods
// ------------------------------------------------------------------
inline size_t tlsf_allocator::used() const {
    return used_;
}

inline size_t tlsf_allocator::available() const {
    return total_size_ - used_;
}

inline size_t tlsf_allocator::largest_free_block() const {
    // Find highest set bit in fl_bitmap to get largest FL class
    if (fl_bitmap_ == 0) {
        return 0;
    }
    int fl       = (int) (sizeof(uint32_t) * 8 - 1 - __builtin_clz(fl_bitmap_));
    // Find highest set bit in sl_bitmap for that FL class
    int sl       = (int) (sizeof(uint32_t) * 8 - 1 - __builtin_clz(sl_bitmap_[fl]));
    // The head of that list is the largest (approximately)
    int block_id = free_lists_[fl][sl];
    return block_id >= 0 ? blocks_[block_id].size : 0;
}

// ------------------------------------------------------------------
// Placement by physical position.  Defined after allocate() on purpose:
// tests/test-sycl-buffer-base-alignment-source.py locates allocate()'s body
// by the first occurrence of its signature, which allocate_top() and
// allocate_below() share as a prefix.
// ------------------------------------------------------------------
inline int tlsf_allocator::gap_block(size_t anchor) const {
    int id = -1;
    if (anchor == no_anchor) {
        id = last_block_;
    } else {
        auto it = offset_to_block_.find(anchor);
        if (it == offset_to_block_.end()) {
            return -1;
        }
        id = blocks_[it->second].prev_block;
    }
    return id >= 0 && blocks_[id].free ? id : -1;
}

inline size_t tlsf_allocator::gap_below(size_t anchor) const {
    int id = gap_block(anchor);
    return id >= 0 ? blocks_[id].size : 0;
}

inline size_t tlsf_allocator::carve_gap(size_t anchor, size_t size, size_t alignment, uint8_t tag, bool from_top) {
    if (size == 0) {
        return SIZE_MAX;
    }
    // Same rounding and the same alignment limit as allocate(), for the same
    // reason: every block offset must stay a multiple of MIN_BLOCK_SIZE, and a
    // top carve places its block at (end - size), so size must be a multiple too.
    const size_t granularity = alignment > MIN_BLOCK_SIZE ? alignment : MIN_BLOCK_SIZE;
    size                     = (size + granularity - 1) & ~(granularity - 1);
    TLSF_ASSERT(alignment <= MIN_BLOCK_SIZE && "TLSF only supports alignment <= MIN_BLOCK_SIZE (256)");

    int gap_id = gap_block(anchor);
    if (gap_id < 0 || blocks_[gap_id].size < size) {
        return SIZE_MAX;
    }
    remove_free(gap_id);

    int carved_id = gap_id;
    if (from_top && blocks_[gap_id].size >= size + MIN_BLOCK_SIZE) {
        // Read values before alloc_block_id() — it may reallocate blocks_ vector.
        size_t gap_offset = blocks_[gap_id].offset;
        size_t gap_size   = blocks_[gap_id].size;
        int    gap_next   = blocks_[gap_id].next_block;

        carved_id      = alloc_block_id();  // May reallocate blocks_!
        auto & top     = blocks_[carved_id];
        top.offset     = gap_offset + gap_size - size;
        top.size       = size;
        top.free       = false;
        top.prev_block = gap_id;
        top.next_block = gap_next;
        top.prev_free  = -1;
        top.next_free  = -1;
        top.tag        = tag;

        if (gap_next >= 0) {
            blocks_[gap_next].prev_block = carved_id;
        } else {
            last_block_ = carved_id;
        }

        blocks_[gap_id].size       = gap_size - size;
        blocks_[gap_id].next_block = carved_id;
        insert_free(gap_id);
    } else {
        // Front carve, or a gap too small to leave a free remainder: take the
        // front of the gap and return any remainder above it to the free lists.
        blocks_[gap_id].free = false;
        blocks_[gap_id].tag  = tag;
        split_block(gap_id, size);
    }

    const size_t carved_offset = blocks_[carved_id].offset;
    used_ += blocks_[carved_id].size;
    offset_to_block_[carved_offset] = carved_id;
    TLSF_ASSERT((carved_offset % MIN_BLOCK_SIZE) == 0 && "TLSF carved offset lost MIN_BLOCK_SIZE alignment");
    return carved_offset;
}

inline size_t tlsf_allocator::allocate_top(size_t size, size_t alignment, uint8_t tag) {
    return carve_gap(no_anchor, size, alignment, tag, /*from_top=*/true);
}

inline size_t tlsf_allocator::allocate_below(size_t anchor, size_t size, size_t alignment, uint8_t tag) {
    return carve_gap(anchor, size, alignment, tag, /*from_top=*/true);
}

inline size_t tlsf_allocator::allocate_gap_front(size_t anchor, size_t size, size_t alignment, uint8_t tag) {
    return carve_gap(anchor, size, alignment, tag, /*from_top=*/false);
}

inline uint8_t tlsf_allocator::tag_at(size_t offset) const {
    auto it = offset_to_block_.find(offset);
    return it != offset_to_block_.end() ? blocks_[it->second].tag : 0;
}

inline std::vector<tlsf_allocator::extent> tlsf_allocator::frontier_walk(size_t anchor, uint8_t pass_tag) const {
    std::vector<extent> walk;
    int                 id = -1;
    if (anchor == no_anchor) {
        id = last_block_;
    } else {
        auto it = offset_to_block_.find(anchor);
        if (it == offset_to_block_.end()) {
            return walk;
        }
        id = blocks_[it->second].prev_block;
    }
    while (id >= 0 && (blocks_[id].free || blocks_[id].tag == pass_tag)) {
        const block_meta & b = blocks_[id];
        walk.push_back({ b.offset, b.size, b.free, b.free ? (uint8_t) 0 : b.tag });
        id = b.prev_block;
    }
    return walk;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_TLSF_ALLOCATOR_HPP
