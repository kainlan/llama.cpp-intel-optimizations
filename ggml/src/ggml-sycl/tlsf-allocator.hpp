// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// TLSF (Two-Level Segregated Fit) sub-allocator for VRAM arena zones.
//
// Provides O(1) guaranteed alloc/free with zero heap allocations during
// operation. Block headers are stored inline in the arena memory itself.
// Bit-scan intrinsics (__builtin_clzll, __builtin_ctz) provide constant-time
// index search.
//
// Reference: M. Masmano et al., "TLSF: A New Dynamic Memory Allocator for
// Real-Time Systems" (2004).

#ifndef GGML_SYCL_TLSF_ALLOCATOR_HPP
#define GGML_SYCL_TLSF_ALLOCATOR_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ggml_sycl {

class tlsf_allocator {
  public:
    // Initialize with contiguous region [base, base + size).
    // min_alloc_size: minimum block granularity (256 bytes for GPU alignment).
    // Writes initial block_header into the arena memory itself.
    tlsf_allocator(void * base, size_t size, size_t min_alloc_size = 256);

    // O(1) allocation. Returns nullptr if no suitable block found.
    // alignment must be a power of 2 and >= min_alloc_size.
    void * allocate(size_t size, size_t alignment = 256);

    // O(1) free with automatic coalescing of physically adjacent blocks.
    // ptr must have been returned by allocate(). No size parameter needed —
    // the allocator tracks block size internally via block_header.
    void free(void * ptr);

    // Bulk reset — returns ALL memory to the pool as a single free block.
    // Faster than freeing everything individually. O(1).
    void reset();

    // Query methods
    size_t used() const;                // Bytes currently allocated (including headers)
    size_t available() const;           // Bytes available for allocation
    size_t largest_free_block() const;  // Size of largest contiguous free block

    // Padded header overhead per block (256 bytes for GPU alignment).
    // Tests use this instead of sizeof(block_header) which is private.
    static constexpr size_t header_overhead() { return 256; }

  private:
    // ------------------------------------------------------------------
    // Block header: stored at the START of every block (free or allocated).
    // For free blocks, next_free/prev_free are also valid.
    // For allocated blocks, next_free/prev_free are undefined (the space
    // after the header is user data).
    //
    // Physical layout in memory:
    //   [block_header (32 bytes) | padding (224 bytes) | user data ... ]
    //   ^                                                ^
    //   block start                                      returned pointer (block + HEADER_SIZE)
    // ------------------------------------------------------------------
    struct block_header {
        // Size of the usable area (excludes this header).
        // Bit 0: FREE flag (1 = free, 0 = allocated).
        // Bit 1: PREV_FREE flag (1 = physically previous block is free).
        // Actual size = size_and_flags & ~(FLAG_FREE | FLAG_PREV_FREE).
        size_t size_and_flags;

        // Previous block in PHYSICAL (address) order, for backward coalescing.
        // nullptr for the first block in the arena.
        block_header * prev_phys;

        // --- Only valid when block is FREE ---
        block_header * next_free;  // Next in same (fl, sl) doubly-linked list
        block_header * prev_free;  // Prev in same (fl, sl) doubly-linked list

        // Helper methods
        size_t size() const { return size_and_flags & ~(FLAG_FREE | FLAG_PREV_FREE); }

        bool is_free() const { return (size_and_flags & FLAG_FREE) != 0; }

        bool prev_is_free() const { return (size_and_flags & FLAG_PREV_FREE) != 0; }

        void set_size(size_t s) { size_and_flags = s | (size_and_flags & (FLAG_FREE | FLAG_PREV_FREE)); }

        void set_free() { size_and_flags |= FLAG_FREE; }

        void set_used() { size_and_flags &= ~FLAG_FREE; }

        void set_prev_free() { size_and_flags |= FLAG_PREV_FREE; }

        void clear_prev_free() { size_and_flags &= ~FLAG_PREV_FREE; }

        // Pointer to user data area (HEADER_SIZE bytes past the block start).
        // Uses HEADER_SIZE (256) rather than sizeof(*this) (32) so that
        // payloads are always 256-byte aligned when the arena base is aligned.
        void * payload() { return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(this) + HEADER_SIZE); }

        // Get next block in physical order using size arithmetic.
        block_header * next_phys() {
            return reinterpret_cast<block_header *>(reinterpret_cast<uint8_t *>(payload()) + size());
        }

        static constexpr size_t FLAG_FREE      = 1 << 0;
        static constexpr size_t FLAG_PREV_FREE = 1 << 1;
    };

    // ------------------------------------------------------------------
    // TLSF configuration constants
    // ------------------------------------------------------------------
    // FL_OFFSET = log2(min_alloc_size). Blocks smaller than 2^FL_OFFSET
    // are not tracked (below minimum granularity).
    static constexpr int FL_OFFSET = 8;  // log2(256) = 8

    // Number of second-level subdivisions per first-level class.
    // 16 gives good granularity without excessive bookkeeping.
    static constexpr int SL_COUNT = 16;
    static constexpr int SL_BITS  = 4;  // log2(SL_COUNT)

    // FL_COUNT: number of first-level classes. Must cover max arena size.
    // 2^(FL_OFFSET + FL_COUNT - 1) must be >= max arena size.
    // For 16 GB max: need FL_OFFSET + FL_COUNT - 1 >= 34, so FL_COUNT >= 27.
    // Use 28 for safety margin.
    static constexpr int FL_COUNT = 28;

    // Minimum block size: must hold block_header (for free-list pointers).
    // sizeof(block_header) = 32 bytes on 64-bit. But we also need the
    // usable area to hold next_free + prev_free (16 bytes) when free.
    // So min usable = 16 bytes, min total = sizeof(block_header) = 32.
    // With FL_OFFSET=8, min alloc = 256 >> easily holds the header.
    static constexpr size_t MIN_BLOCK_SIZE = 256;  // Matches GPU alignment

    // Padded header size: 256 bytes so that payload() is always 256-byte
    // aligned when the arena base is 256-aligned (which sycl::malloc_device
    // guarantees). The actual block_header struct is only 32 bytes; the
    // remaining 224 bytes are unused padding. For an 11 GB arena with
    // hundreds of blocks, the waste (~224 * N_blocks) is negligible.
    static constexpr size_t HEADER_SIZE    = 256;

    // ------------------------------------------------------------------
    // Two-level index structures (all fixed-size, no heap allocation)
    // ------------------------------------------------------------------
    uint32_t       fl_bitmap_;                       // Which FL classes have free blocks
    uint32_t       sl_bitmap_[FL_COUNT];             // Which SL classes have free blocks per FL
    block_header * free_lists_[FL_COUNT][SL_COUNT];  // Head of free list per (fl, sl)

    void * base_;                                    // Arena base address
    size_t total_size_;                              // Total arena size
    size_t used_;                                    // Currently allocated bytes (including headers)

    // ------------------------------------------------------------------
    // Core TLSF operations
    // ------------------------------------------------------------------

    // Map a block size to (fl, sl) indices.
    void mapping(size_t size, int & fl, int & sl) const {
        if (size < MIN_BLOCK_SIZE) {
            size = MIN_BLOCK_SIZE;
        }
        fl = 0;
        if (size >= (1ull << FL_OFFSET)) {
            // fl = floor(log2(size)) - FL_OFFSET
            // Use __builtin_clzll for bit scan
            fl = (int) (sizeof(size_t) * 8 - 1 - __builtin_clzll(size)) - FL_OFFSET;
            sl = (int) ((size >> (fl + FL_OFFSET - SL_BITS)) - SL_COUNT);
            // Clamp
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
    // is guaranteed to have blocks >= size. Used for allocation search.
    void mapping_search(size_t size, int & fl, int & sl) const {
        // Round up size to next SL boundary to ensure any block in the
        // found class is >= requested size.
        if (size >= (1ull << FL_OFFSET)) {
            int t = (1 << ((int) (sizeof(size_t) * 8 - 1 - __builtin_clzll(size)) - SL_BITS)) - 1;
            size += t;
        }
        mapping(size, fl, sl);
    }

    // Find a free block from class (fl, sl) or higher. O(1) via bitmaps.
    // Returns nullptr if no block available.
    block_header * find_suitable(int fl, int sl) {
        // Search current FL class from sl upward
        uint32_t sl_map = sl_bitmap_[fl] & (~0u << sl);
        if (sl_map == 0) {
            // No block in this FL class. Search higher FL classes.
            uint32_t fl_map = fl_bitmap_ & (~0u << (fl + 1));
            if (fl_map == 0) {
                return nullptr;              // No free block large enough
            }
            fl     = __builtin_ctz(fl_map);  // First set bit = smallest sufficient FL
            sl_map = sl_bitmap_[fl];
            // sl_map must be non-zero since fl_bitmap_ bit was set
        }
        sl = __builtin_ctz(sl_map);  // First set bit = smallest sufficient SL
        return free_lists_[fl][sl];  // Head of the list (guaranteed non-null)
    }

    // Remove block from its free list. Update bitmaps if list becomes empty.
    void remove_free(block_header * block) {
        int fl, sl;
        mapping(block->size(), fl, sl);

        block_header * prev = block->prev_free;
        block_header * next = block->next_free;
        if (prev) {
            prev->next_free = next;
        }
        if (next) {
            next->prev_free = prev;
        }

        // If this was the head of the list, update free_lists
        if (free_lists_[fl][sl] == block) {
            free_lists_[fl][sl] = next;
            // If list is now empty, clear bitmap bits
            if (next == nullptr) {
                sl_bitmap_[fl] &= ~(1u << sl);
                if (sl_bitmap_[fl] == 0) {
                    fl_bitmap_ &= ~(1u << fl);
                }
            }
        }
    }

    // Insert block into the appropriate free list. Update bitmaps.
    void insert_free(block_header * block) {
        int fl, sl;
        mapping(block->size(), fl, sl);

        block_header * head = free_lists_[fl][sl];
        block->next_free    = head;
        block->prev_free    = nullptr;
        if (head) {
            head->prev_free = block;
        }

        free_lists_[fl][sl] = block;
        fl_bitmap_ |= (1u << fl);
        sl_bitmap_[fl] |= (1u << sl);
    }

    // Split block: carve off `size` bytes from the front, return remainder
    // to free list if large enough.
    void split_block(block_header * block, size_t size) {
        // Guard against underflow: block must be large enough to split off
        // a remainder block (header + minimum payload).
        if (block->size() < size + HEADER_SIZE + MIN_BLOCK_SIZE) {
            return;
        }
        size_t remain = block->size() - size - HEADER_SIZE;
        if (remain >= MIN_BLOCK_SIZE) {
            block_header * rest =
                reinterpret_cast<block_header *>(reinterpret_cast<uint8_t *>(block->payload()) + size);
            rest->size_and_flags = 0;
            rest->set_size(remain);
            rest->set_free();
            rest->prev_phys = block;

            // Update the block AFTER rest to point back to rest
            block_header * next = rest->next_phys();
            if (reinterpret_cast<uint8_t *>(next) < reinterpret_cast<uint8_t *>(base_) + total_size_) {
                next->prev_phys = rest;
                next->set_prev_free();
            }

            block->set_size(size);
            insert_free(rest);
        }
        // If remainder is too small, block keeps the extra (internal fragmentation)
    }

    // Coalesce block with physically adjacent free neighbors.
    // Returns the merged block (may be different pointer if left-merged).
    block_header * coalesce(block_header * block) {
        // Merge with next physical block if free
        block_header * next = block->next_phys();
        if (reinterpret_cast<uint8_t *>(next) < reinterpret_cast<uint8_t *>(base_) + total_size_) {
            if (next->is_free()) {
                remove_free(next);
                block->set_size(block->size() + HEADER_SIZE + next->size());
                // Update the block after the merged block
                block_header * nn = block->next_phys();
                if (reinterpret_cast<uint8_t *>(nn) < reinterpret_cast<uint8_t *>(base_) + total_size_) {
                    nn->prev_phys = block;
                }
            }
        }

        // Merge with previous physical block if free
        if (block->prev_is_free() && block->prev_phys) {
            block_header * prev = block->prev_phys;
            if (prev->is_free()) {
                remove_free(prev);
                prev->set_size(prev->size() + HEADER_SIZE + block->size());
                // Update the block after the merged block
                block_header * nn = prev->next_phys();
                if (reinterpret_cast<uint8_t *>(nn) < reinterpret_cast<uint8_t *>(base_) + total_size_) {
                    nn->prev_phys = prev;
                }
                block = prev;
            }
        }

        return block;
    }
};

// ------------------------------------------------------------------
// Constructor: initialize all bitmaps to zero, create one large free block
// ------------------------------------------------------------------
inline tlsf_allocator::tlsf_allocator(void * base, size_t size, size_t min_alloc_size) :
    fl_bitmap_(0),
    base_(base),
    total_size_(size),
    used_(0) {
    (void) min_alloc_size;  // Currently hardcoded to 256 via FL_OFFSET

    // Zero all bitmaps and free lists
    std::memset(sl_bitmap_, 0, sizeof(sl_bitmap_));
    std::memset(free_lists_, 0, sizeof(free_lists_));

    // The entire arena is one free block.
    // Place block_header at the very start of the arena.
    if (size < HEADER_SIZE + MIN_BLOCK_SIZE) {
        return;  // Arena too small
    }

    block_header * first  = reinterpret_cast<block_header *>(base);
    first->size_and_flags = 0;
    first->set_size(size - HEADER_SIZE);
    first->set_free();
    first->prev_phys = nullptr;
    first->next_free = nullptr;
    first->prev_free = nullptr;

    insert_free(first);
}

// ------------------------------------------------------------------
// allocate: O(1) via bitmap search
// ------------------------------------------------------------------
inline void * tlsf_allocator::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    // Round up to minimum block size and alignment
    size = (size + alignment - 1) & ~(alignment - 1);
    if (size < MIN_BLOCK_SIZE) {
        size = MIN_BLOCK_SIZE;
    }

    // Account for alignment: HEADER_SIZE is padded to 256 bytes, so when
    // alignment <= HEADER_SIZE (the common case with 256-byte GPU alignment),
    // no extra space is needed — payloads are naturally aligned.
    // For larger alignments, over-allocate by `alignment` to guarantee we
    // can find an aligned address within the block.
    size_t adjusted = size;
    if (alignment > HEADER_SIZE) {
        adjusted += alignment;
    }

    int fl, sl;
    mapping_search(adjusted, fl, sl);

    block_header * block = find_suitable(fl, sl);
    if (!block) {
        return nullptr;
    }

    remove_free(block);
    block->set_used();

    // Split off excess
    split_block(block, size);

    // Mark next physical block's prev_free flag as false
    block_header * next = block->next_phys();
    if (reinterpret_cast<uint8_t *>(next) < reinterpret_cast<uint8_t *>(base_) + total_size_) {
        next->clear_prev_free();
    }

    used_ += block->size() + HEADER_SIZE;
    return block->payload();
}

// ------------------------------------------------------------------
// free: O(1) via bitmap insert + coalesce with neighbors
// ------------------------------------------------------------------
inline void tlsf_allocator::free(void * ptr) {
    if (!ptr) {
        return;
    }

    // Recover block_header from payload pointer
    block_header * block = reinterpret_cast<block_header *>(reinterpret_cast<uint8_t *>(ptr) - HEADER_SIZE);

    used_ -= block->size() + HEADER_SIZE;

    block->set_free();

    // Mark next physical block's prev_free flag
    block_header * next = block->next_phys();
    if (reinterpret_cast<uint8_t *>(next) < reinterpret_cast<uint8_t *>(base_) + total_size_) {
        next->set_prev_free();
    }

    block = coalesce(block);
    insert_free(block);
}

// ------------------------------------------------------------------
// reset: O(1) bulk deallocation
// ------------------------------------------------------------------
inline void tlsf_allocator::reset() {
    fl_bitmap_ = 0;
    std::memset(sl_bitmap_, 0, sizeof(sl_bitmap_));
    std::memset(free_lists_, 0, sizeof(free_lists_));
    used_ = 0;

    if (total_size_ < HEADER_SIZE + MIN_BLOCK_SIZE) {
        return;
    }

    block_header * first  = reinterpret_cast<block_header *>(base_);
    first->size_and_flags = 0;
    first->set_size(total_size_ - HEADER_SIZE);
    first->set_free();
    first->prev_phys = nullptr;
    first->next_free = nullptr;
    first->prev_free = nullptr;
    insert_free(first);
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
    int            fl    = (int) (sizeof(uint32_t) * 8 - 1 - __builtin_clz(fl_bitmap_));
    // Find highest set bit in sl_bitmap for that FL class
    int            sl    = (int) (sizeof(uint32_t) * 8 - 1 - __builtin_clz(sl_bitmap_[fl]));
    // The head of that list is the largest (approximately)
    block_header * block = free_lists_[fl][sl];
    return block ? block->size() : 0;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_TLSF_ALLOCATOR_HPP
