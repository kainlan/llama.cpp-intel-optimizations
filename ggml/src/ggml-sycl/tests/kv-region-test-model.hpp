// A host model of one TLSF of the shared KV+WEIGHT zone, for the kv_region_fit
// cases (llama.cpp-moua H2, H3, H6).
//
// It drives the REAL tlsf_allocator the way the zone does -- weights and
// optional tenants front-carve, context-side blocks top-carve below an anchor --
// and keeps the census of what it allocated, so snapshot() can copy out the
// geometry kv_region_fit reads and replay() can carve a fit's plan with the same
// primitives the commit uses.  The fit's offsets are checked against the
// allocator's, not against a second model of it.
//
// SYCL-free: no unified-cache, no device.  Not a production header.
#pragma once

#include "../kv-runtime-demotion.hpp"
#include "../shared-zone-tags.hpp"
#include "../tlsf-allocator.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

namespace kv_region_test {

using ggml_sycl::kv_carve_op;
using ggml_sycl::kv_region_fit_result;
using ggml_sycl::shared_zone_geometry;
using ggml_sycl::tlsf_allocator;
using ggml_sycl::tlsf_geometry;
using ggml_sycl::zone_block;
using ggml_sycl::zone_pending_range;
using ggml_sycl::zone_range;
using ggml_sycl::zone_reservation;

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;

class zone_model {
  public:
    explicit zone_model(size_t size) : size_(size), tlsf_(size), anchor_(tlsf_allocator::no_anchor) {}

    // The TLSF flags kv_region_fit reads.
    bool takes_kv           = true;
    bool takes_weight_named = true;

    // A weight: front-carved at the bottom of the gap.
    size_t weight(size_t size) { return carve_front(size, ggml_sycl::SHARED_ZONE_TAG_WEIGHT, false); }

    // An optional tenant: front-carved after the weights, so the highest one is
    // the first the frontier walk meets.  `leased` tenants are not yieldable.
    size_t optional_tenant(size_t size, bool leased = false) {
        return carve_front(size, ggml_sycl::SHARED_ZONE_TAG_OPTIONAL, leased);
    }

    // A context-side block: top-carved below the anchor, which it becomes.
    size_t context(size_t size) {
        const size_t used = tlsf_.used();
        const size_t off  = tlsf_.allocate_below(anchor_, size, 256, ggml_sycl::SHARED_ZONE_TAG_CONTEXT);
        if (off == SIZE_MAX) {
            return SIZE_MAX;
        }
        census_[off] = { off, tlsf_.used() - used, ggml_sycl::SHARED_ZONE_TAG_CONTEXT, false };
        anchor_      = off;
        return off;
    }

    void free(size_t off) {
        tlsf_.free(off);
        census_.erase(off);
    }

    void pend(size_t offset, size_t size, bool first_context = false) {
        pending_.push_back({ offset, size, first_context });
    }

    void own(size_t offset, size_t size) { own_.push_back({ offset, size }); }

    void retain(size_t offset, size_t size) { retained_.push_back({ offset, size }); }

    void reserve(ggml_sycl::demand_scope scope,
                 uint64_t                owner,
                 const char *            cohort,
                 uint32_t                index,
                 size_t                  offset,
                 size_t                  size) {
        reservations_.push_back({ scope, owner, cohort, index, offset, size });
    }

    size_t size() const { return size_; }

    const std::vector<zone_pending_range> & pending() const { return pending_; }

    // The chunks a placement could occupy, computed from the census and the
    // pending ranges alone (not from the fit's pieces): every free stretch up to
    // the allocated block above it, cut to the part above the highest other-
    // transaction pending range it contains, and the fragments of each retained
    // run that no such range covers.  A stretch whose top is off the 256 grid
    // cannot be top-carved.  An independent reading of "what the commit can
    // carve", for the refusal oracle.
    std::vector<size_t> carvable_chunks() const {
        std::vector<size_t> out;
        auto                cut_floor = [&](size_t a, size_t b) {
            size_t floor = a;
            for (const zone_pending_range & p : pending_) {
                if (p.first_context || p.size == 0 || p.offset >= b || p.offset + p.size <= a) {
                    continue;
                }
                floor = std::max(floor, p.offset + p.size);
            }
            return floor;
        };
        auto stretch = [&](size_t a, size_t b) {
            if (b > a && b % 256 == 0) {
                const size_t f = cut_floor(a, b);
                if (f < b) {
                    out.push_back(b - f);
                }
            }
        };
        size_t at = 0;
        for (const auto & kv : census_) {
            stretch(at, kv.first);
            at = kv.first + kv.second.size;
        }
        stretch(at, size_);
        for (const zone_range & r : retained_) {
            size_t                  lo    = r.offset;
            size_t                  hi    = r.offset + r.size;
            std::vector<zone_range> frags = {
                { lo, hi - lo }
            };
            for (const zone_pending_range & p : pending_) {
                if (p.first_context || p.size == 0) {
                    continue;
                }
                std::vector<zone_range> next;
                for (const zone_range & f : frags) {
                    const size_t f_hi = f.offset + f.size;
                    if (p.offset >= f_hi || p.offset + p.size <= f.offset) {
                        next.push_back(f);
                        continue;
                    }
                    if (p.offset > f.offset) {
                        next.push_back({ f.offset, p.offset - f.offset });
                    }
                    if (p.offset + p.size < f_hi) {
                        next.push_back({ p.offset + p.size, f_hi - (p.offset + p.size) });
                    }
                }
                frags.swap(next);
            }
            for (const zone_range & f : frags) {
                out.push_back(f.size);
            }
        }
        return out;
    }

    size_t anchor() const { return anchor_; }

    // The yieldability the model assigns an optional tenant: the predicate the
    // production census is handed in place of jehw's.
    bool optional_yieldable(size_t offset) const { return !is_leased(offset); }

    tlsf_allocator & allocator() { return tlsf_; }

    const tlsf_allocator & allocator() const { return tlsf_; }

    // The geometry kv_region_fit reads: the frontier walk, then the whole-TLSF
    // census below it, cut into runs of free blocks and releasable optional
    // tenants.
    tlsf_geometry snapshot() const {
        tlsf_geometry g;
        g.takes_kv            = takes_kv;
        g.takes_weight_named  = takes_weight_named;
        size_t frontier_floor = anchor_ == tlsf_allocator::no_anchor ? size_ : anchor_;
        for (const tlsf_allocator::extent & e : tlsf_.frontier_walk(anchor_, ggml_sycl::SHARED_ZONE_TAG_OPTIONAL)) {
            g.frontier.push_back(block_of(e.offset, e.size, e.free, e.tag));
            frontier_floor = e.offset;
        }
        // The census below the frontier, low to high.
        std::vector<zone_block> blocks;
        size_t                  at = 0;
        for (const auto & kv : census_) {
            if (kv.first >= frontier_floor) {
                break;
            }
            if (kv.first > at) {
                blocks.push_back(block_of(at, kv.first - at, true, 0));
            }
            blocks.push_back(block_of(kv.first, kv.second.size, false, kv.second.tag));
            at = kv.first + kv.second.size;
        }
        if (at < frontier_floor) {
            blocks.push_back(block_of(at, frontier_floor - at, true, 0));
        }
        std::vector<zone_block> run;
        for (const zone_block & b : blocks) {
            const bool releasable = b.free || (b.optional_tenant && b.yieldable);
            if (releasable) {
                run.push_back(b);
            } else if (!run.empty()) {
                g.side_runs.push_back(run);
                run.clear();
            }
        }
        if (!run.empty()) {
            g.side_runs.push_back(run);
        }
        g.retained_runs  = retained_;
        g.pending_ranges = pending_;
        g.own_ranges     = own_;
        g.reservations   = reservations_;
        return g;
    }

    // Release what a fit yields: the top `prefix` optional tenants of the
    // frontier walk (highest first), then the named buried tenants.
    void yield(size_t prefix, const std::vector<size_t> & buried = {}) {
        std::vector<size_t> offsets;
        for (const tlsf_allocator::extent & e : tlsf_.frontier_walk(anchor_, ggml_sycl::SHARED_ZONE_TAG_OPTIONAL)) {
            if (!e.free && offsets.size() < prefix) {
                offsets.push_back(e.offset);
            }
        }
        for (size_t o : offsets) {
            free(o);
        }
        for (size_t o : buried) {
            free(o);
        }
    }

    // Carve one op the way the commit does: tlsf_allocator::allocate_at, the
    // offset-fixed carve of the spec, at the op's offset.  The block is carved at
    // exactly [offset, offset + size) or nothing changes: a carve that lands on
    // another offset, or whose block is not the extent the fit planned (a demand
    // that rounds to a different size, an end off the allocator's grain), is freed
    // again, so a failed carve leaves the allocator as it found it.  Returns false,
    // printing why, on failure.
    bool carve(const kv_carve_op & op) {
        if (!op.carve) {
            return true;
        }
        const size_t end = op.offset + op.size;
        for (const auto & kv : census_) {
            if (kv.first < end && op.offset < kv.first + kv.second.size) {
                std::fprintf(stderr, "carve mismatch: fit offset %zu size %zu overlaps an allocated block\n", op.offset,
                             op.size);
                return false;
            }
        }
        const size_t off = tlsf_.allocate_at(op.offset, op.demand, ggml_sycl::SHARED_ZONE_TAG_CONTEXT);
        if (off != op.offset) {
            std::fprintf(stderr, "carve mismatch: fit offset %zu size %zu demand %zu, allocator gave %zu\n", op.offset,
                         op.size, op.demand, off);
            if (off != SIZE_MAX) {
                tlsf_.free(off);
            }
            return false;
        }
        const size_t got = tlsf_.block_size_at(off);
        if (got != op.size) {
            std::fprintf(stderr, "carve mismatch: fit offset %zu size %zu demand %zu, the block is %zu bytes\n",
                         op.offset, op.size, op.demand, got);
            tlsf_.free(off);
            return false;
        }
        census_[off] = { off, got, ggml_sycl::SHARED_ZONE_TAG_CONTEXT, false };
        if (end == anchor_ || (anchor_ == tlsf_allocator::no_anchor && end >= size_)) {
            anchor_ = off;
        }
        return true;
    }

    // Everything the allocator's public surface shows of its state: the byte counts, each
    // allocated block with the gap under it, and the free blocks from the top down.  Two
    // equal fingerprints mean a carve that failed changed nothing.
    std::vector<size_t> fingerprint() const {
        std::vector<size_t> f = { tlsf_.used(), tlsf_.available(), tlsf_.largest_free_block(), census_.size(),
                                  anchor_ };
        for (const auto & kv : census_) {
            f.push_back(kv.first);
            f.push_back(kv.second.size);
            f.push_back(tlsf_.gap_below(kv.first));
        }
        for (const tlsf_allocator::extent & e :
             tlsf_.frontier_walk(tlsf_allocator::no_anchor, ggml_sycl::SHARED_ZONE_TAG_OPTIONAL)) {
            f.push_back(e.offset);
            f.push_back(e.size);
            f.push_back(e.free ? 1 : 0);
        }
        return f;
    }

    bool invariants() const { return tlsf_.check_invariants(); }

  private:
    struct block {
        size_t  offset;
        size_t  size;
        uint8_t tag;
        bool    leased;
    };

    size_t carve_front(size_t size, uint8_t tag, bool leased) {
        const size_t used = tlsf_.used();
        const size_t off  = tlsf_.allocate_gap_front(anchor_, size, 256, tag);
        if (off == SIZE_MAX) {
            return SIZE_MAX;
        }
        census_[off] = { off, tlsf_.used() - used, tag, leased };
        return off;
    }

    zone_block block_of(size_t offset, size_t size, bool free, uint8_t tag) const {
        zone_block b;
        b.offset          = offset;
        b.size            = size;
        b.free            = free;
        b.optional_tenant = !free && tag == ggml_sycl::SHARED_ZONE_TAG_OPTIONAL;
        b.yieldable       = b.optional_tenant && !is_leased(offset);
        return b;
    }

    bool is_leased(size_t offset) const {
        auto it = census_.find(offset);
        return it != census_.end() && it->second.leased;
    }

    size_t                          size_;
    tlsf_allocator                  tlsf_;
    size_t                          anchor_;
    std::map<size_t, block>         census_;
    std::vector<zone_pending_range> pending_;
    std::vector<zone_range>         own_;
    std::vector<zone_range>         retained_;
    std::vector<zone_reservation>   reservations_;
};

// A device: the TLSFs of one zone, snapshotted together.
struct device_model {
    std::vector<zone_model> tlsfs;

    shared_zone_geometry snapshot() const {
        shared_zone_geometry g;
        for (const zone_model & z : tlsfs) {
            g.tlsfs.push_back(z.snapshot());
        }
        return g;
    }

    // Yield, then carve every op in the fit's order.  The carved offsets must be
    // the fit's.
    bool replay(const kv_region_fit_result & fit) {
        for (size_t t = 0; t < tlsfs.size(); ++t) {
            std::vector<size_t> buried;
            for (const ggml_sycl::kv_buried_release & b : fit.buried_released) {
                if (b.tlsf == t) {
                    buried.push_back(b.offset);
                }
            }
            tlsfs[t].yield(t < fit.yield_prefix.size() ? fit.yield_prefix[t] : 0, buried);
        }
        for (const kv_carve_op & op : fit.carve_order) {
            if (!tlsfs[op.tlsf].carve(op)) {
                return false;
            }
        }
        for (const zone_model & z : tlsfs) {
            if (!z.invariants()) {
                return false;
            }
        }
        // plan == reality: nothing the fit placed may lie on another
        // transaction's pending range, whether it was carved or not.
        auto clear_of_pending = [&](size_t t, size_t off, size_t size, const char * what) {
            for (const zone_pending_range & p : tlsfs[t].pending()) {
                if (!p.first_context && p.size != 0 && off < p.offset + p.size && p.offset < off + size) {
                    std::fprintf(stderr, "replay: %s at %zu size %zu on tlsf %zu overlaps pending [%zu, %zu)\n", what,
                                 off, size, t, p.offset, p.offset + p.size);
                    return false;
                }
            }
            return true;
        };
        for (const ggml_sycl::kv_region_extent & e : fit.extents) {
            if (e.kind != ggml_sycl::KV_EXTENT_SELF && !clear_of_pending(e.tlsf, e.offset, e.size, "extent")) {
                return false;
            }
        }
        for (const ggml_sycl::kv_head_placement & h : fit.heads) {
            bool refused = false;
            for (size_t rh : fit.refused_heads) {
                refused = refused || rh == h.head;
            }
            if (h.size != 0 && !h.reused && !refused && !clear_of_pending(h.tlsf, h.offset, h.size, "head slot")) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace kv_region_test
