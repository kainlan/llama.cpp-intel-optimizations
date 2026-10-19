//
// MoE prefill expert streaming ring (llama.cpp-36j0, Task 6 of
// docs/plans/2026-10-19-moe-executor-cost-model-prefill-ring.md).
//
// A layer's host-resident expert triplets are copied into equal slots of one
// WEIGHT-zone allocation and run there with the device kernels resident experts
// use. This header holds two things:
//
//   moe_stream_ring_size()  the pure byte budget and slot count. It depends on
//                           nothing, so the host-only test compiles it without
//                           SYCL (GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY).
//   moe_stream_ring         the reservation: one allocation, sliced into slots,
//                           each slot handed out as a mem_handle, split into two
//                           epoch regions so a region is refilled only after the
//                           event that ends its previous use.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// slot_bytes is the largest host triplet's size (entries are not equal-sized),
// n_slots the whole slots in the budget, bytes = n_slots * slot_bytes. n_slots
// is 0 (and bytes 0) when the ring cannot hold one layer's slots; the caller
// then answers RING_UNAVAILABLE.
struct moe_stream_ring_geometry {
    size_t   slot_bytes = 0;
    uint32_t n_slots    = 0;
    size_t   bytes      = 0;
};

// bytes = min(ring_layers * host_bytes_per_layer_max, resident_expert_bytes / 10),
// rounded down to whole slots of slot_bytes = triplet_bytes_max. One layer's
// slots are the whole slots its host bytes fill (at least one), so a ring sized
// for one layer is never refused for its own rounding.
moe_stream_ring_geometry moe_stream_ring_size(size_t host_bytes_per_layer_max,
                                              size_t resident_expert_bytes,
                                              size_t triplet_bytes_max,
                                              int    ring_layers);

}  // namespace ggml_sycl

#if !defined(GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY)

#    include "mem-handle.hpp"

#    include <sycl/sycl.hpp>

namespace ggml_sycl {

// Not thread-safe: one ring belongs to one device's prompt executor, which
// acquires, flips and releases from its submitting thread. Nothing here waits on
// the host: a region's reuse is gated by one status query of the event that
// ended its previous use, and release() hands the backing to
// retain_handles_until_event().
class moe_stream_ring {
  public:
    moe_stream_ring() = default;

    ~moe_stream_ring() { release(); }

    moe_stream_ring(const moe_stream_ring &)             = delete;
    moe_stream_ring & operator=(const moe_stream_ring &) = delete;

    // One WEIGHT-zone allocation of geo.bytes on `device`, split into two
    // regions of geo.n_slots / 2 slots. Returns false, with one WARN and no
    // throw, when the zone cannot give the bytes or the geometry cannot form two
    // regions.
    bool reserve(int device, const moe_stream_ring_geometry & geo);

    // The next slot of the current region as a slice of the ring's allocation.
    // An invalid handle when the ring is not reserved, the region is exhausted,
    // or the region's previous use has not completed yet (one status query, no
    // wait).
    mem_handle acquire_slot();

    // Ends the current region's use at `region_done` and makes the other region
    // current. The other region's slots are handed out again only after the
    // event stored on it at its own last boundary completes.
    void epoch_boundary(sycl::event region_done);

    // Drops the ring's allocation. The bytes go back to the unified cache once
    // both regions' boundary events complete and every slot handed out is
    // dropped; nothing waits here.
    void release();

    const moe_stream_ring_geometry & geometry() const { return geo_; }

    // Slots the current region can still hand out, not counting a pending
    // boundary event (asking would cost a status query).
    uint32_t slots_free() const;

  private:
    struct region {
        uint32_t    first = 0;
        uint32_t    next  = 0;
        uint32_t    count = 0;
        sycl::event done;
        bool        has_done = false;
    };

    moe_stream_ring_geometry geo_;
    int                      device_ = -1;
    // from_owned_alloc() adopts the reservation's alloc_owner, so this handle is
    // the allocation's one owner here; every slot is a slice that retains it.
    mem_handle               base_;
    region                   regions_[2];
    int                      current_ = 0;
};

}  // namespace ggml_sycl

#endif  // !GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY
