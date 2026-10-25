//
// MoE prefill expert streaming ring (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// A layer's host-resident expert matrices are copied into equal WEIGHT-zone
// slots and run there with the device kernels resident experts use. A matrix
// staged as SOA spends two slots, the AOS copy destination and the SOA reorder
// destination; a matrix staged AOS-only spends one, read in place (c-hmey).
// This header holds two things:
//
//   moe_stream_ring_size()  the pure geometry: two regions, each holding one
//                           executor window's slots, or no ring. It depends on
//                           nothing, so the host-only test compiles it without
//                           SYCL (GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY).
//   moe_stream_ring         the reservation: one allocation per slot (scattered
//                           room a slot wide gives the ring, llama.cpp-am87),
//                           each slot handed out as a mem_handle, split into two
//                           epoch regions. Layer l+1 streams into one region
//                           while layer l runs from the other, so a region's
//                           reuse is ordered after the event that ended its
//                           previous use. That ordering happens on the device:
//                           the caller chains every copy into a region on
//                           current_region_gate(). No host query, no wait.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// Every slot starts, and every slot is sized, at a multiple of this alignment
// (the zone allocator's default), so wide device loads on a slot are aligned
// whatever the quant's block size.
static constexpr size_t MOE_STREAM_RING_SLOT_ALIGN = 256;

// slot_bytes is one matrix's slot: the largest streamable matrix's bytes on the
// device (max(src_size, dst_size) staged as SOA, src_size staged AOS-only),
// rounded up to MOE_STREAM_RING_SLOT_ALIGN. A region holds one executor
// window: region_slots = moe_stream_plan_index::window_slots, the experts one
// phase of the batched executor reads from one role. n_slots = 2 *
// region_slots and bytes = n_slots * slot_bytes.
// When the ring does not fit, every field is 0 (the default geometry); the
// caller then answers RING_UNAVAILABLE.
struct moe_stream_ring_geometry {
    size_t   slot_bytes   = 0;
    uint32_t n_slots      = 0;
    size_t   bytes        = 0;
    uint32_t region_slots = 0;
};

// Two regions of region_slots slots of matrix_bytes_max (rounded up to
// MOE_STREAM_RING_SLOT_ALIGN) each, or no ring when those bytes, padding
// included, exceed 10 % of what the expert set keeps once the ring's own bytes
// are lent out of it: bytes > (resident_expert_bytes - bytes) / 10, tested as
// bytes > resident_expert_bytes / 11, which cannot overflow. The caller passes
// resident_expert_bytes as the device's resident expert bytes PLUS the bytes
// lent to the ring now (moe_stream_plan_ring_geometry, llama.cpp-7999): a loan
// moves bytes from one term to the other, so the answer does not change with
// it. region_slots is one executor window (moe_stream_plan_index::window_slots,
// min(role_items_max, G_plan)), not a whole layer: the phase stages a layer a
// window at a time.
moe_stream_ring_geometry moe_stream_ring_size(uint32_t region_slots,
                                              size_t   matrix_bytes_max,
                                              size_t   resident_expert_bytes);

}  // namespace ggml_sycl

#if !defined(GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY)

#    include "mem-handle.hpp"

#    include <sycl/sycl.hpp>
#    include <vector>

namespace ggml_sycl {

// Not thread-safe: one ring belongs to one device's prompt executor, which
// acquires, flips and releases from its submitting thread. Nothing here queries
// an event or waits on the host. A region's reuse is ordered on the device: the
// caller passes current_region_gate() as a dependency of every copy into a slot
// of the current region. release() hands the backing to
// retain_handles_until_event().
class moe_stream_ring {
  public:
    moe_stream_ring() = default;

    // release() publishes through retain_handles_until_event(), which can
    // throw; a throw from this destructor terminates. There is deliberately no
    // catch.
    ~moe_stream_ring() { release(); }

    moe_stream_ring(const moe_stream_ring &)             = delete;
    moe_stream_ring & operator=(const moe_stream_ring &) = delete;

    // geo.n_slots WEIGHT-zone allocations on `device` of geo.slot_bytes each,
    // one per slot, split into two regions of geo.region_slots slots;
    // geometry() is geo as given. Each slot goes to the zone, then to the
    // general entry point with the ring's intent (no spill outside the arena),
    // and is kept once it resolves on the device. Returns false, with one WARN
    // and no throw, for every geometry that is not exactly what
    // moe_stream_ring_size() produces (n_slots == 2 * region_slots, at least
    // two slots, bytes == n_slots * slot_bytes, slot_bytes a nonzero multiple
    // of MOE_STREAM_RING_SLOT_ALIGN), and when a slot is refused.
    //
    // Reserved slots outlive a refused attempt until release(): a refused slot
    // stops the attempt, the slots already allocated stay held, and the ring is
    // not valid() until every slot is. The next reserve() of the same geometry
    // on the same device continues from the slots held; one of another
    // geometry or device gives them back first, with one WARN. `bytes_short`,
    // when given, is set to the bytes of the slots still to allocate,
    // (n_slots - held) * slot_bytes, which an expert loan of that much covers
    // whatever the zone's fragmentation, since each slot needs only a hole of
    // its own (llama.cpp-3460, c-0tmn; llama.cpp-am87). It is 0 on success and
    // on every other refusal.
    bool reserve(int device, const moe_stream_ring_geometry & geo, size_t * bytes_short = nullptr);

    // Whether every slot of the reserved geometry is held. Only a valid ring
    // hands out slots; a partial reservation hands out none.
    bool valid() const { return geo_.n_slots != 0 && slots_.size() == geo_.n_slots; }

    // The next slot of the current region: the slot's own allocation, as a
    // handle that resolves with `layout`, the layout the caller will write
    // there (lead ruling c-o9uw-1: the slice's tag is the slot's layout), or an
    // invalid handle when the ring is not valid() or the region is exhausted.
    // It never consults an event: the caller orders its copy into the slot on
    // current_region_gate().
    mem_handle acquire_slot(ggml_layout_mode layout);

    // The event stored by the last epoch_boundary() that left the current
    // region, or a default-constructed (complete) event when it has none. Every
    // copy into a slot of the current region must depend on it.
    sycl::event current_region_gate() const;

    // Ends the current region's use at `region_done` and makes the other region
    // current. The event is stored only when slots were acquired from the region
    // since it became current; otherwise the region keeps its older gate, which
    // still guards its last use.
    void epoch_boundary(sycl::event region_done);

    // Drops the ring's slots. A gated region's slots go back to the unified
    // cache once that region's gate completes and every handle to them is
    // dropped; nothing waits here. Gates exist only for regions closed by
    // epoch_boundary(): the caller must call epoch_boundary(<event of the last
    // use>) before release() and before destruction, or hold its slot handles
    // until that device work completes. Otherwise the current region's bytes
    // return to the cache under live device work. A partial reservation has no
    // gate and handed nothing out: its slots go back at once.
    void release();

    const moe_stream_ring_geometry & geometry() const { return geo_; }

    // Slots the current region can still hand out: 0 unless valid().
    uint32_t slots_free() const;

  private:
    struct region {
        uint32_t    first = 0;
        uint32_t    next  = 0;
        uint32_t    count = 0;
        sycl::event gate;
        bool        has_gate = false;
    };

    moe_stream_ring_geometry geo_;
    int                      device_ = -1;
    // One handle per slot, in slot order, filled left to right: from_owned_alloc()
    // adopts each slot's alloc_owner, so each handle is that allocation's one
    // owner here, and every handle acquire_slot() returns retains it. Fewer than
    // geo_.n_slots is a partial reservation.
    std::vector<mem_handle>  slots_;
    region                   regions_[2];
    int                      current_ = 0;
};

}  // namespace ggml_sycl

#endif  // !GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY
