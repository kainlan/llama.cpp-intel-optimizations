// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The pending-range primitive (llama.cpp-moua L4; moua design 2.3.3, A1-A4).
//
// A pending range is room a transaction has PLANNED on one TLSF but not yet
// carved.  Every other allocator on that TLSF treats it as allocated, and only
// its owner draws inside it.  There is one primitive, shared by moua, zhcn,
// 23mk and 1oxa, and moua's names are the canonical ones: pending_owner,
// pending_term, pending_term_mask, PENDING_TERM_ALL and the operations below.
//
// This header is the host-testable core and has no callers.  It works on one
// TLSF and the pending ranges recorded on it; the device-wide forms take the
// device's TLSFs as a list.  Taking the group mutex around each TLSF, carrying
// the owner in alloc_constraints, and the handle-level replace_within (which
// classifies `old`'s other references and retires its registration, and calls
// replace_within_count_guard first) belong to unified-cache and mem-handle and
// are built on replace_within_block below.  The name differs on purpose: the
// alloc-zone gate's L-GUARD binds every function named replace_within.  Like tlsf_allocator,
// nothing here is synchronized: the caller holds the TLSF's group mutex.
//
// Misuse of a filter is a defect, not an answer.  A clear, retag or fit that
// names PENDING_TERM_ALL outside a load's commit or rollback would wipe terms
// another design owns, and an EMPTY filter selects nothing and hides that the
// caller named no term.  Both abort through pending_range_misuse_handler().

#ifndef GGML_SYCL_PENDING_RANGE_HPP
#define GGML_SYCL_PENDING_RANGE_HPP

#include "tlsf-allocator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ggml_sycl {

enum class pending_owner_kind : uint8_t {
    NONE    = 0,  // matches no range; the default request honours every range
    LOAD    = 1,  // id = LoadTxnId; lives from the early inventory stage to the load's commit or rollback
    MODEL   = 2,  // id = ModelId; from the commit to the unload
    CONTEXT = 3,  // id = the backend context's execution_context_id
    DEVICE  = 4,  // id = the device index
};

struct pending_owner {
    pending_owner_kind kind = pending_owner_kind::NONE;
    uint64_t           id   = 0;

    bool operator==(const pending_owner & o) const { return kind == o.kind && id == o.id; }

    bool operator!=(const pending_owner & o) const { return !(*this == o); }
};

// What a range holds room for.  This is the one definition of the shared term
// list; no design defines a term of its own.  Each term is one bit so a filter
// is a mask.
enum class pending_term : uint32_t {
    WEIGHT               = 1u << 0,   // a load's planned device weights
    ARENA                = 1u << 1,   // a compute-arena range (no arena device records one)
    SCRATCH              = 1u << 2,   // a load's scratch hold
    MODEL_TERM           = 1u << 3,   // a model-lifetime term
    DEVICE_TERM          = 1u << 4,   // reserved, no producer
    REGION               = 1u << 5,   // a context's extents and head slots
    ONEDNN_PP_A          = 1u << 6,   // a transient context term
    SET_ROWS_STAGE       = 1u << 7,   // a transient context term
    ONEDNN_GRAPH_SCRATCH = 1u << 8,   // a context's Graph-scratch range
    FIRST_CONTEXT        = 1u << 9,   // the room a load reserves for its model's first context
    VM_TAIL_SURPLUS      = 1u << 10,  // a VM tail's surplus fence
};

using pending_term_mask = uint32_t;

// Every term, present and future.  Legal only at a load's commit and rollback,
// under a {LOAD, txn} owner.
constexpr pending_term_mask PENDING_TERM_ALL = 0xFFFFFFFFu;

constexpr pending_term_mask pending_term_bit(pending_term t) {
    return static_cast<pending_term_mask>(t);
}

constexpr pending_term_mask operator|(pending_term a, pending_term b) {
    return pending_term_bit(a) | pending_term_bit(b);
}

constexpr pending_term_mask operator|(pending_term_mask a, pending_term b) {
    return a | pending_term_bit(b);
}

struct pending_range {
    pending_owner owner;
    pending_term  term   = pending_term::WEIGHT;
    size_t        offset = 0;
    size_t        size   = 0;

    size_t end() const { return offset + size; }
};

// The abort channel.  The default prints one line and aborts; a test installs
// its own, which must not return.
using pending_range_misuse_fn = void (*)(const char * detail);

inline pending_range_misuse_fn & pending_range_misuse_handler() {
    static pending_range_misuse_fn handler = [](const char * detail) {
        std::fprintf(stderr, "[PENDING-RANGE-MISUSE] %s\n", detail);
        std::abort();
    };
    return handler;
}

#define PENDING_RANGE_MISUSE(detail)                        \
    do {                                                    \
        pending_range_misuse_handler()(detail);             \
        std::abort(); /* a handler that returns is a bug */ \
    } while (0)

// The same, with the operation's name in front: the abort line is the only diagnostic.
#define PENDING_RANGE_MISUSE_IN(op, detail)                                    \
    do {                                                                       \
        char misuse_line[192];                                                 \
        std::snprintf(misuse_line, sizeof(misuse_line), "%s: %s", op, detail); \
        PENDING_RANGE_MISUSE(misuse_line);                                     \
    } while (0)

// A half-open interval, for the geometry below.
struct pending_extent {
    size_t offset = 0;
    size_t size   = 0;
};

// The outcome of a draw inside the owner's own ranges.  A miss is the owner's
// plan bug and never falls through to allocate_excluding outside them; the
// caller reports it under its own tag ([KV-PLAN-BUG], [ZONE-PLAN-BUG]).
enum class pending_within_status : uint8_t {
    OK = 0,
    MISS,               // no free part of the owner's ranges of that term holds the size
    KEY_SIZE_MISMATCH,  // a keyed draw whose size differs from its key's
    KEY_OUTSIDE_RANGE,  // the key is not inside one of the owner's ranges of that term
    KEY_OCCUPIED,       // the keyed range is not free
    ZERO_SIZE,          // a draw of no bytes: a caller bug, not a plan miss
    SIZE_OVERFLOW,      // a size (or key) whose grain rounding wraps the address space
    KEY_OFF_GRAIN,      // the key's offset is not on the block grain
};

struct pending_within_result {
    pending_within_status status = pending_within_status::MISS;
    size_t                offset = SIZE_MAX;  // SIZE_MAX unless status is OK

    bool ok() const { return status == pending_within_status::OK; }
};

// The placement a load's replay recorded for one WEIGHT item.  It names no TLSF:
// the caller draws on the TLSF it already holds, with the set recorded on it.
struct pending_key {
    size_t offset = 0;
    size_t size   = 0;
};

enum class pending_replace_status : uint8_t {
    OK = 0,
    OLD_NOT_FOUND,  // `old` is not an allocated block of this TLSF
    OUTSIDE_RANGE,  // the new range is not inside `old`'s block plus the owner's ranges of the term
    REST_NOT_ONE,   // a term other than WEIGHT left a rest in two pieces: it has one remainder handle
    CARVE_FAILED,   // the new range is not free once `old` is released; `old` is restored whole
};

struct pending_replace_result {
    pending_replace_status status           = pending_replace_status::OLD_NOT_FOUND;
    size_t                 new_offset       = SIZE_MAX;
    size_t                 remainder_offset = SIZE_MAX;  // SIZE_MAX: no remainder block
    size_t                 remainder_size   = 0;

    bool ok() const { return status == pending_replace_status::OK; }
};

// The pending ranges recorded on ONE TLSF.
class pending_range_set {
  public:
    // record_pending(owner, term, offset, size).  Replace or append is decided
    // by the owner's kind, here and nowhere else: a {LOAD, txn} record replaces
    // that owner's earlier range of the same term on this TLSF, so the early
    // stage's recording is idempotent, and so does a {DEVICE, d} record of
    // VM_TAIL_SURPLUS; every other owner's record APPENDS beside its ranges of
    // the same term, never replacing one.  A zero-size record is dropped.
    void record(pending_owner owner, pending_term term, size_t offset, size_t size) {
        require_owner(owner, "record");
        if (offset > SIZE_MAX - size) {
            PENDING_RANGE_MISUSE_IN("record", "the range wraps the address space");
        }
        const bool replaces = owner.kind == pending_owner_kind::LOAD ||
                              (owner.kind == pending_owner_kind::DEVICE && term == pending_term::VM_TAIL_SURPLUS);
        if (replaces) {
            erase_if([&](const pending_range & r) { return r.owner == owner && r.term == term; });
        }
        if (size != 0) {
            ranges_.push_back({ owner, term, offset, size });
        }
    }

    // clear_pending(owner, term_filter): idempotent; returns how many ranges it
    // dropped.  Ranges of the owner's other terms stay.
    size_t clear(pending_owner owner, pending_term_mask filter) {
        require_owner(owner, "clear");
        require_filter(owner, filter, "clear");
        return erase_if(
            [&](const pending_range & r) { return r.owner == owner && (pending_term_bit(r.term) & filter); });
    }

    // retag_pending(owner, term_filter, new_owner): the matching ranges move
    // whole to new_owner, keeping their terms; ranges of other terms stay.  It
    // moves, so it never merges with a range new_owner already holds.
    size_t retag(pending_owner owner, pending_term_mask filter, pending_owner new_owner) {
        require_owner(owner, "retag");
        require_owner(new_owner, "retag target");
        require_filter(owner, filter, "retag");
        size_t moved = 0;
        for (pending_range & r : ranges_) {
            if (r.owner == owner && (pending_term_bit(r.term) & filter)) {
                r.owner = new_owner;
                moved++;
            }
        }
        return moved;
    }

    // pending_ranges(owner, term_filter): the owner's ranges whose term is in
    // the filter, in the order recorded.  A fit's own_ranges come from here and
    // never from filtering the unfiltered list by hand.
    std::vector<pending_range> ranges(pending_owner owner, pending_term_mask filter) const {
        require_owner(owner, "ranges");
        require_filter(owner, filter, "ranges");
        std::vector<pending_range> out;
        for (const pending_range & r : ranges_) {
            if (r.owner == owner && (pending_term_bit(r.term) & filter)) {
                out.push_back(r);
            }
        }
        return out;
    }

    // pending_ranges(): every range with its owner and term.
    const std::vector<pending_range> & all() const { return ranges_; }

    bool empty() const { return ranges_.empty(); }

    // The free bytes of the owner's ranges whose term is in the filter: the
    // bytes of those ranges not occupied by an allocated block.  A scalar for
    // accounting; a fit places by the ranges, not by this number.
    size_t free_bytes(const tlsf_allocator & tlsf, pending_owner owner, pending_term_mask filter) const {
        require_owner(owner, "free_bytes");
        require_filter(owner, filter, "free_bytes");
        return free_bytes_where(
            tlsf, [&](const pending_range & r) { return r.owner == owner && (pending_term_bit(r.term) & filter); });
    }

    // The free bytes of every range except those of `except_owner` AND
    // `except_term` (one term, not a mask).  except_owner of kind NONE matches
    // no range, so every range counts.
    size_t free_bytes_excluding(const tlsf_allocator & tlsf,
                                pending_owner          except_owner,
                                pending_term           except_term) const {
        return free_bytes_where(
            tlsf, [&](const pending_range & r) { return !(r.owner == except_owner && r.term == except_term); });
    }

    // The ranges as the exclusion list for allocate_excluding().
    std::vector<tlsf_allocator::excluded_range> as_excluded() const {
        std::vector<tlsf_allocator::excluded_range> out;
        for (const pending_range & r : ranges_) {
            out.push_back({ r.offset, r.size });
        }
        return out;
    }

    // allocate_within(owner, term, size, align, tag, consume): a first fit over
    // the free parts of the owner's ranges of that term, lowest offset first,
    // carved with allocate_at() semantics so remainders stay whole.
    //   consume == false (temporaries): the range record is untrimmed, so a
    //     freed block returns to the free lists but stays inside the range,
    //     excluded from every other allocator and reusable by its owner for
    //     the range's whole life.
    //   consume == true (weights): the carved part leaves the record.
    // The term keeps a weight draw out of a SCRATCH hold of the same owner and
    // the reverse.  A miss is reported, never retried outside the ranges.
    // A draw lies inside ONE recorded range: two adjacent records of one owner
    // and term are not coalesced here, so free_bytes() can report room that no
    // single draw holds.  The keyed form, which names its bytes, does read the
    // union of the owner's ranges.  A zero size and a size whose rounding wraps
    // are reported as such (ZERO_SIZE, SIZE_OVERFLOW), apart from a MISS.  An
    // alignment over the grain aborts, as it does in the allocator.
    pending_within_result allocate_within(tlsf_allocator & tlsf,
                                          pending_owner    owner,
                                          pending_term     term,
                                          size_t           size,
                                          size_t           alignment,
                                          uint8_t          tag,
                                          bool             consume) {
        require_owner(owner, "allocate_within");
        if (size == 0) {
            return { pending_within_status::ZERO_SIZE, SIZE_MAX };
        }
        size = tlsf_allocator::round_request(size, alignment);
        if (size == 0) {
            return { pending_within_status::SIZE_OVERFLOW, SIZE_MAX };
        }

        std::vector<pending_range> mine = ranges(owner, pending_term_bit(term));
        std::sort(mine.begin(), mine.end(),
                  [](const pending_range & a, const pending_range & b) { return a.offset < b.offset; });
        const std::vector<tlsf_allocator::extent> census = tlsf.block_census();
        for (const pending_range & r : mine) {
            for (const tlsf_allocator::extent & b : census) {
                // An allocated block needs no test here: allocate_at() below refuses any range it holds.
                const size_t lo = std::max(r.offset, b.offset);
                const size_t hi = std::min(r.end(), b.offset + b.size);
                if (lo >= hi) {
                    continue;
                }
                const size_t a = (lo + tlsf_allocator::block_grain - 1) & ~(tlsf_allocator::block_grain - 1);
                if (a < hi && hi - a >= size) {
                    const size_t got = tlsf.allocate_at(a, size, tag);
                    if (got == SIZE_MAX) {
                        continue;
                    }
                    if (consume) {
                        trim(owner, term, got, tlsf.block_size_at(got));
                    }
                    return { pending_within_status::OK, got };
                }
            }
        }
        return {};
    }

    // The keyed form, for every WEIGHT draw: the replay recorded the item's
    // placement as `key`, and the draw carves exactly [key.offset, key.offset +
    // key.size) inside the owner's WEIGHT range, consuming it.  There is no
    // first fit, so the order draws arrive in cannot move an item.  A draw
    // whose size differs from its key's, a key outside the owner's ranges, and
    // an occupied key are each a distinct plan bug, never a fall-through.  The
    // key's size is rounded to the grain before it is tested against the
    // ranges, as the carve rounds it: a range that holds the unrounded bytes
    // but not the rounded ones would let the carve take another owner's room.
    // A zero size, a size that wraps and an off-grain offset are refused apart
    // from the plan-bug results.
    pending_within_result allocate_within_keyed(tlsf_allocator & tlsf,
                                                pending_owner    owner,
                                                pending_key      key,
                                                size_t           size,
                                                uint8_t          tag) {
        require_owner(owner, "allocate_within_keyed");
        if (size != key.size) {
            return { pending_within_status::KEY_SIZE_MISMATCH, SIZE_MAX };
        }
        if (size == 0) {
            return { pending_within_status::ZERO_SIZE, SIZE_MAX };
        }
        const size_t rounded = tlsf_allocator::round_request(size, tlsf_allocator::block_grain);
        if (rounded == 0 || key.offset > SIZE_MAX - rounded) {
            return { pending_within_status::SIZE_OVERFLOW, SIZE_MAX };
        }
        if (key.offset % tlsf_allocator::block_grain != 0) {
            return { pending_within_status::KEY_OFF_GRAIN, SIZE_MAX };
        }
        if (!covered_by(ranges(owner, pending_term_bit(pending_term::WEIGHT)), {}, key.offset, rounded)) {
            return { pending_within_status::KEY_OUTSIDE_RANGE, SIZE_MAX };
        }
        const size_t got = tlsf.allocate_at(key.offset, key.size, tag);
        if (got == SIZE_MAX) {
            return { pending_within_status::KEY_OCCUPIED, SIZE_MAX };
        }
        trim(owner, pending_term::WEIGHT, got, tlsf.block_size_at(got));
        return { pending_within_status::OK, got };
    }

    // replace_within_block(owner, term, old, offset, size): the TLSF-level core
    // of the handle-level replace_within (23mk's zone_replace merge).  `old` is the allocated block at old_offset,
    // released here; the new block [new_offset, new_offset + new_size) is
    // carved with allocate_at() semantics inside `old`'s block PLUS the owner's
    // ranges of `term`, so it never lands in another term's range, and the
    // owner's range parts it consumes leave the record.  The rest of `old`'s
    // block:
    //   WEIGHT (remainder mode): both fragments, before and after the new
    //     block, stay free and are re-recorded {owner, WEIGHT}, untrimmed, so
    //     every other allocator excludes them and only the same owner draws
    //     them again.  No remainder block is returned.
    //   another term: the rest must be ONE piece, and it is carved as an
    //     allocated block of `remainder_tag` that the caller's remainder handle
    //     owns.  An empty rest returns no remainder and carves nothing.
    // A failed carve restores `old`'s block whole (offset, size and tag) and
    // returns CARVE_FAILED; every refusal before the release changes nothing.
    // The handle half (classifying `old`'s other references, retiring its
    // registration, the HOST_TIER refusal) is the caller's.
    // A requirement on that handle half (23mk S4a): this core frees `old`
    // unconditionally and nothing in it counts `old`'s references, so when the
    // handle-level replace_within lands it must carry an L-CALLER-style latch
    // that limits the callers of replace_within_block to replace_within and
    // tests.
    pending_replace_result replace_within_block(tlsf_allocator & tlsf,
                                                pending_owner    owner,
                                                pending_term     term,
                                                size_t           old_offset,
                                                size_t           new_offset,
                                                size_t           new_size,
                                                uint8_t          tag,
                                                uint8_t          remainder_tag) {
        require_owner(owner, "replace_within_block");
        pending_replace_result out;
        const size_t           old_size = tlsf.block_size_at(old_offset);
        if (old_size == 0) {
            out.status = pending_replace_status::OLD_NOT_FOUND;
            return out;
        }
        const size_t rounded_new = tlsf_allocator::round_request(new_size, tlsf_allocator::block_grain);
        if (rounded_new == 0 || new_offset > SIZE_MAX - rounded_new) {
            out.status = pending_replace_status::OUTSIDE_RANGE;
            return out;
        }
        const uint8_t old_tag                    = tlsf.tag_at(old_offset);
        new_size                                 = rounded_new;
        const size_t                     old_end = old_offset + old_size;
        const size_t                     new_end = new_offset + new_size;
        const std::vector<pending_range> mine    = ranges(owner, pending_term_bit(term));
        if (new_offset % tlsf_allocator::block_grain != 0 ||
            !covered_by(mine, pending_extent{ old_offset, old_size }, new_offset, new_size)) {
            out.status = pending_replace_status::OUTSIDE_RANGE;
            return out;
        }
        // The rest of old's block, as the pieces outside the new range.
        const size_t before_lo = old_offset;
        const size_t before_hi = std::min(new_offset, old_end);
        const size_t after_lo  = std::max(new_end, old_offset);
        const size_t after_hi  = old_end;
        const size_t before    = before_hi > before_lo ? before_hi - before_lo : 0;
        const size_t after     = after_hi > after_lo ? after_hi - after_lo : 0;
        if (term != pending_term::WEIGHT && before != 0 && after != 0) {
            out.status = pending_replace_status::REST_NOT_ONE;
            return out;
        }

        tlsf.free(old_offset);
        const size_t got = tlsf.allocate_at(new_offset, new_size, tag);
        if (got == SIZE_MAX) {
            // old's recorded extent goes back as it was: allocate_at() would round a tail block that
            // absorbed a sub-grain remainder up past the region end.
            const size_t back = tlsf.allocate_extent_at(old_offset, old_size, old_tag);
            TLSF_ASSERT(back == old_offset && "replace_within_block could not restore the block it released");
            out.status = pending_replace_status::CARVE_FAILED;
            return out;
        }
        size_t       rest_offset = SIZE_MAX;
        const size_t rest_size   = before != 0 ? before : after;
        if (term != pending_term::WEIGHT && rest_size != 0) {
            rest_offset           = before != 0 ? before_lo : after_lo;
            // The rest lies inside old's freed extent and outside the new block, so it is free by
            // construction; its extent is taken exactly (a tail rest may end off the grain).
            const size_t rest_got = tlsf.allocate_extent_at(rest_offset, rest_size, remainder_tag);
            TLSF_ASSERT(rest_got == rest_offset &&
                        "replace_within_block could not carve the rest of the block it freed");
        }
        trim(owner, term, got, tlsf.block_size_at(got));
        if (term == pending_term::WEIGHT) {
            // Remainder mode: both fragments stay free and are pending again.  `old`'s whole extent leaves the
            // record first, so a draw that was inside a range does not leave the fragments recorded twice.  They
            // are appended, not recorded through record(), whose {LOAD, txn} key-replace would drop the owner's
            // other range.
            trim(owner, term, old_offset, old_size);
            if (before != 0) {
                ranges_.push_back({ owner, pending_term::WEIGHT, before_lo, before });
            }
            if (after != 0) {
                ranges_.push_back({ owner, pending_term::WEIGHT, after_lo, after });
            }
        }
        out.status           = pending_replace_status::OK;
        out.new_offset       = got;
        out.remainder_offset = rest_offset;
        out.remainder_size   = rest_offset == SIZE_MAX ? 0 : rest_size;
        return out;
    }

    // The misuse checks, public so the device-wide queries apply the same rules.
    static void require_owner(pending_owner owner, const char * op) {
        if (owner.kind == pending_owner_kind::NONE) {
            PENDING_RANGE_MISUSE_IN(op, "an operation named owner kind NONE, which matches no range");
        }
    }

    static void require_filter(pending_owner owner, pending_term_mask filter, const char * op) {
        if (filter == 0) {
            PENDING_RANGE_MISUSE_IN(op, "an operation named an empty term filter");
        }
        if (filter == PENDING_TERM_ALL && owner.kind != pending_owner_kind::LOAD) {
            PENDING_RANGE_MISUSE_IN(op, "PENDING_TERM_ALL is legal only under a {LOAD, txn} owner");
        }
    }

  private:
    std::vector<pending_range> ranges_;

    template <typename Pred> size_t erase_if(Pred pred) {
        const size_t before = ranges_.size();
        ranges_.erase(std::remove_if(ranges_.begin(), ranges_.end(), pred), ranges_.end());
        return before - ranges_.size();
    }

    // Is [offset, offset + size) covered, byte for byte, by `extra` plus the
    // given ranges?  The union is taken, so adjacent or overlapping pieces
    // cover a span none of them covers alone.
    static bool covered_by(const std::vector<pending_range> & ranges,
                           pending_extent                     extra,
                           size_t                             offset,
                           size_t                             size) {
        std::vector<pending_extent> pieces;
        for (const pending_range & r : ranges) {
            pieces.push_back({ r.offset, r.size });
        }
        if (extra.size != 0) {
            pieces.push_back(extra);
        }
        std::sort(pieces.begin(), pieces.end(),
                  [](const pending_extent & a, const pending_extent & b) { return a.offset < b.offset; });
        size_t       reach = offset;
        const size_t end   = offset + size;
        for (const pending_extent & p : pieces) {
            if (p.offset > reach) {
                break;
            }
            reach = std::max(reach, p.offset + p.size);
            if (reach >= end) {
                return true;
            }
        }
        return reach >= end;
    }

    // Remove [offset, offset + size) from the owner's ranges of `term`: a range
    // it overlaps keeps the parts outside it.
    void trim(pending_owner owner, pending_term term, size_t offset, size_t size) {
        const size_t               end = offset + size;
        std::vector<pending_range> next;
        for (const pending_range & r : ranges_) {
            if (r.owner != owner || r.term != term || r.end() <= offset || r.offset >= end) {
                next.push_back(r);
                continue;
            }
            if (r.offset < offset) {
                next.push_back({ r.owner, r.term, r.offset, offset - r.offset });
            }
            if (r.end() > end) {
                next.push_back({ r.owner, r.term, end, r.end() - end });
            }
        }
        ranges_.swap(next);
    }

    template <typename Pred> size_t free_bytes_where(const tlsf_allocator & tlsf, Pred pred) const {
        // The union of the selected ranges, so two ranges that overlap count a free byte once.
        std::vector<pending_extent> selected;
        for (const pending_range & r : ranges_) {
            if (pred(r)) {
                selected.push_back({ r.offset, r.size });
            }
        }
        std::sort(selected.begin(), selected.end(),
                  [](const pending_extent & a, const pending_extent & b) { return a.offset < b.offset; });
        std::vector<pending_extent> merged;
        for (const pending_extent & e : selected) {
            if (!merged.empty() && e.offset <= merged.back().offset + merged.back().size) {
                const size_t end   = std::max(merged.back().offset + merged.back().size, e.offset + e.size);
                merged.back().size = end - merged.back().offset;
            } else {
                merged.push_back(e);
            }
        }
        size_t total = 0;
        for (const tlsf_allocator::extent & b : tlsf.block_census()) {
            if (!b.free) {
                continue;
            }
            for (const pending_extent & m : merged) {
                const size_t lo = std::max(m.offset, b.offset);
                const size_t hi = std::min(m.offset + m.size, b.offset + b.size);
                if (lo < hi) {
                    total += hi - lo;
                }
            }
        }
        return total;
    }
};

// One TLSF of a device's zone, for the device-wide queries.
struct pending_range_member {
    int                       zone;
    const tlsf_allocator *    tlsf;
    const pending_range_set * set;
};

// pending_bytes(owner, term_filter): the free bytes of the owner's ranges
// across every listed TLSF.
inline size_t pending_bytes(const std::vector<pending_range_member> & members,
                            pending_owner                             owner,
                            pending_term_mask                         filter) {
    pending_range_set::require_owner(owner, "pending_bytes");
    pending_range_set::require_filter(owner, filter, "pending_bytes");
    size_t total = 0;
    for (const pending_range_member & m : members) {
        total += m.set->free_bytes(*m.tlsf, owner, filter);
    }
    return total;
}

// pending_bytes_excluding(device, zone, except_owner, except_term): the free
// bytes of EVERY pending range on the listed TLSFs of `zone` (or of every zone
// when zone is -1), across all owners, except the ranges whose owner is
// `except_owner` AND whose term is `except_term`.  Accounting only; a fit
// places by the ranges.
inline size_t pending_bytes_excluding(const std::vector<pending_range_member> & members,
                                      int                                       zone,
                                      pending_owner                             except_owner,
                                      pending_term                              except_term) {
    size_t total = 0;
    for (const pending_range_member & m : members) {
        if (zone >= 0 && m.zone != zone) {
            continue;
        }
        total += m.set->free_bytes_excluding(*m.tlsf, except_owner, except_term);
    }
    return total;
}

}  // namespace ggml_sycl

#endif  // GGML_SYCL_PENDING_RANGE_HPP
