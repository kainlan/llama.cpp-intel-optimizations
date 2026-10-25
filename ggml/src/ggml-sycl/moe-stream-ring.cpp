//
// MoE prefill expert streaming ring. See moe-stream-ring.hpp for the contract.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "moe-stream-ring.hpp"

#include <cstdint>

#if !defined(GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY)
#    include "ggml-impl.h"
#    include "unified-cache.hpp"

#    include <utility>
#    include <vector>
#endif

namespace ggml_sycl {

moe_stream_ring_geometry moe_stream_ring_size(uint32_t region_slots,
                                              size_t   matrix_bytes_max,
                                              size_t   resident_expert_bytes) {
    // Every no-ring path returns the default geometry: every field is 0.
    if (region_slots == 0 || matrix_bytes_max == 0) {
        return {};
    }

    // One executor window's slots per region, two regions.
    const uint64_t n_slots = 2ull * region_slots;

    // Round each slot up to the alignment so every slot starts aligned.
    const size_t align = MOE_STREAM_RING_SLOT_ALIGN;
    if (matrix_bytes_max > SIZE_MAX - (align - 1)) {
        return {};
    }
    const size_t slot_bytes = (matrix_bytes_max + align - 1) / align * align;
    if (n_slots > UINT32_MAX || n_slots > SIZE_MAX / slot_bytes) {
        return {};
    }
    const size_t bytes = static_cast<size_t>(n_slots) * slot_bytes;
    // The ring may take 10 % of what the device's expert set keeps once the ring's own bytes are lent out of it:
    // bytes <= (resident - bytes) / 10, which is 11 * bytes <= resident, tested in the division form, which cannot
    // overflow. The caller passes resident as the device's resident expert bytes plus the bytes lent to the ring now
    // (moe_stream_plan_ring_geometry, llama.cpp-7999). A loan moves bytes from the first term to the second, so the
    // figure, and with it the answer, is the same before the loan and after it.
    if (bytes > resident_expert_bytes / 11) {
        return {};
    }

    moe_stream_ring_geometry geo;
    geo.slot_bytes   = slot_bytes;
    geo.region_slots = region_slots;
    geo.n_slots      = static_cast<uint32_t>(n_slots);
    geo.bytes        = bytes;
    return geo;
}

#if !defined(GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY)

bool moe_stream_ring::reserve(int device, const moe_stream_ring_geometry & geo, size_t * bytes_short) {
    if (bytes_short != nullptr) {
        *bytes_short = 0;
    }
    if (valid()) {
        GGML_LOG_WARN("[MOE-STREAM-RING] reserve: device %d already holds a ring of %.1f MB; release it first\n",
                      device_, geo_.bytes / (1024.0 * 1024.0));
        return false;
    }
    // Only what moe_stream_ring_size() produces: two equal regions, exactly
    // n_slots * slot_bytes bytes, and slots that start aligned. Fewer than two
    // slots cannot give each region one.
    const bool shape_ok = geo.n_slots >= 2 && geo.n_slots % 2 == 0 && geo.region_slots == geo.n_slots / 2 &&
                          geo.slot_bytes != 0 && geo.slot_bytes % MOE_STREAM_RING_SLOT_ALIGN == 0 &&
                          geo.bytes / geo.slot_bytes == geo.n_slots && geo.bytes % geo.slot_bytes == 0;
    if (!shape_ok) {
        GGML_LOG_WARN(
            "[MOE-STREAM-RING] reserve: geometry of %u slots (%u per region) x %zu bytes in %zu bytes is not two "
            "equal aligned regions on device %d\n",
            geo.n_slots, geo.region_slots, geo.slot_bytes, geo.bytes, device);
        return false;
    }
    const uint32_t region_slots = geo.region_slots;

    // A partial reservation continues only as the ring it was started for: the
    // slots of another geometry or device go back first.
    const bool same_ring = device == device_ && geo.slot_bytes == geo_.slot_bytes && geo.n_slots == geo_.n_slots &&
                           geo.bytes == geo_.bytes && geo.region_slots == geo_.region_slots;
    if (!slots_.empty() && !same_ring) {
        GGML_LOG_WARN(
            "[MOE-STREAM-RING] reserve: device %d gives back the %zu of %u slots held for a %.1f MB ring to reserve "
            "%.1f MB on device %d\n",
            device_, slots_.size(), geo_.n_slots, geo_.bytes / (1024.0 * 1024.0), geo.bytes / (1024.0 * 1024.0),
            device);
        release();
    }
    geo_    = geo;
    device_ = device;

    // One allocation per slot (llama.cpp-am87): nothing in the ring needs the
    // slots side by side, so room an expert loan frees in holes a slot wide
    // gives the ring. A refused slot stops the loop, and the slots already held
    // stay held for the next reserve() to continue from.
    slots_.reserve(geo.n_slots);
    while (slots_.size() < geo.n_slots) {
        // The WEIGHT zone first. In single-chunk mode its allocator delegates to
        // the KV zone, which unified_cache_zone_allocate_owner() handles; the
        // zone allocator is never touched directly here.
        allocation_result allocation =
            unified_cache_zone_allocate_owner(device, vram_zone_id::WEIGHT, geo.slot_bytes, MOE_STREAM_RING_SLOT_ALIGN);
        if (!allocation) {
            // The scratch pool's arrangement: the same zone through the general
            // entry point, refusing rather than spilling into raw device memory
            // outside the arena when the arena is active.
            alloc_request req{};
            req.device                                    = device;
            req.size                                      = geo.slot_bytes;
            req.alignment                                 = MOE_STREAM_RING_SLOT_ALIGN;
            req.intent.role                               = alloc_role::EXPERT_STAGING;
            req.intent.category                           = runtime_category::EXPERT_CACHE;
            req.intent.cohort_id                          = "moe_stream_ring";
            req.intent.constraints.must_device            = true;
            req.intent.constraints.prefer_vram_zone       = vram_zone_id::WEIGHT;
            req.intent.constraints.forbid_vram_zone_spill = true;
            allocation                                    = unified_allocate_owner(req);
        }
        if (!allocation) {
            break;
        }
        mem_handle slot     = mem_handle::from_owned_alloc(std::move(allocation.owner), GGML_LAYOUT_AOS);
        const auto resolved = slot.resolve(device);
        if (!resolved.ptr || !resolved.on_device) {
            // No loan makes a slot resolve: no shortfall is reported.
            GGML_LOG_WARN("[MOE-STREAM-RING] reserve: slot %zu of %.2f MB did not resolve on device %d\n",
                          slots_.size(), geo.slot_bytes / (1024.0 * 1024.0), device);
            return false;
        }
        slots_.push_back(std::move(slot));
    }
    const uint32_t held = static_cast<uint32_t>(slots_.size());
    if (held < geo.n_slots) {
        // The slots still to allocate: the expert loan the phase asks for (llama.cpp-3460, c-0tmn). Each needs only a
        // hole of its own, so a loan of that many bytes covers it however the zone is fragmented.
        if (bytes_short != nullptr) {
            *bytes_short = static_cast<size_t>(geo.n_slots - held) * geo.slot_bytes;
        }
        GGML_LOG_WARN(
            "[MOE-STREAM-RING] reserve: the WEIGHT zone cannot give %.1f MB (%u slots x %.2f MB) on device %d "
            "(holding %u of %u slots)\n",
            geo.bytes / (1024.0 * 1024.0), geo.n_slots, geo.slot_bytes / (1024.0 * 1024.0), device, held, geo.n_slots);
        return false;
    }

    regions_[0]       = region{};
    regions_[1]       = region{};
    regions_[0].first = 0;
    regions_[0].count = region_slots;
    regions_[1].first = region_slots;
    regions_[1].count = region_slots;
    current_          = 0;
    return true;
}

mem_handle moe_stream_ring::acquire_slot(ggml_layout_mode layout) {
    if (!valid()) {
        return {};
    }
    region & r = regions_[current_];
    if (r.next >= r.count) {
        return {};
    }
    // No host query, no wait: reuse is ordered on the device through
    // current_region_gate().
    const size_t index = static_cast<size_t>(r.first) + r.next;
    ++r.next;
    return slots_[index].slice(0, geo_.slot_bytes, layout);
}

sycl::event moe_stream_ring::current_region_gate() const {
    const region & r = regions_[current_];
    return r.has_gate ? r.gate : sycl::event();
}

void moe_stream_ring::epoch_boundary(sycl::event region_done) {
    if (!valid()) {
        return;
    }
    region & finished = regions_[current_];
    // A region nothing was acquired from since it became current is still
    // guarded by its older gate; replacing it with an unrelated later event
    // would drop the gate on the earlier use.
    if (finished.next != 0) {
        finished.gate     = std::move(region_done);
        finished.has_gate = true;
    }
    current_                = 1 - current_;
    regions_[current_].next = 0;
}

void moe_stream_ring::release() {
    // Each gated region hands its own slots to the drain until its gate
    // completes; slots still held by a caller retain their allocations on their
    // own. A partial reservation has no gated region: its slots just drop.
    for (region & r : regions_) {
        if (r.has_gate) {
            const auto first = slots_.begin() + r.first;
            retain_handles_until_event(std::vector<mem_handle>(first, first + r.count), r.gate);
        }
        r = region{};
    }
    slots_.clear();
    geo_     = moe_stream_ring_geometry{};
    device_  = -1;
    current_ = 0;
}

void moe_stream_ring::release(sycl::event last_use) {
    // Every slot waits for last_use, and release() adds each gated region's
    // own gate: a slot goes back once both have completed.
    if (!slots_.empty()) {
        retain_handles_until_event(slots_, std::move(last_use));
    }
    release();
}

uint32_t moe_stream_ring::slots_free() const {
    const region & r = regions_[current_];
    return r.count - r.next;
}

#endif  // !GGML_SYCL_MOE_STREAM_RING_SIZING_ONLY

}  // namespace ggml_sycl
