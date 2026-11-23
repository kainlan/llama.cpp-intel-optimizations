#pragma once

// The figure the raw-allocation gates compare, host-only so test-zone-sizing checks it.
//
// A raw device allocation is one made outside the VRAM arena: it needs device memory the arena did not reserve. What
// this process already holds on the device is the arena reservation (committed whole when it is made, so it does not
// move with zone use, on one chunk or N), the raw device rows the cache holds outside the arena, and the bytes the
// device showed in use at first cache creation that this process's accounting does not hold (the driver's own use,
// other processes, in-process allocations that never reach the registry). A raw request fits when the device total
// covers all of that and the request. A request an arena zone serves is not a raw request and never meets the gate.
// Zone counters are not part of the figure: they say how full the arena is, not how much device memory is taken.

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

struct zone_raw_gate_inputs {
    size_t device_total       = 0;  // the device's memory; 0 is unknown, and an unknown total gates nothing
    size_t arena_bytes        = 0;  // the arena reservation, every chunk
    size_t raw_held           = 0;  // raw device rows the cache holds outside the arena
    size_t outside_accounting = 0;  // in use at first cache creation, outside this process's accounting
};

// Bytes a raw request may still take outside the arena: the device total less everything held. Saturating at 0, and
// SIZE_MAX when the total is unknown.
inline size_t zone_raw_gate_free_outside_arena(const zone_raw_gate_inputs & in) {
    if (in.device_total == 0) {
        return SIZE_MAX;
    }
    size_t held = in.arena_bytes;
    held        = in.raw_held > SIZE_MAX - held ? SIZE_MAX : held + in.raw_held;
    held        = in.outside_accounting > SIZE_MAX - held ? SIZE_MAX : held + in.outside_accounting;
    return in.device_total > held ? in.device_total - held : 0;
}

// Does a raw device request of `request` bytes fit outside the arena? Always, when the device total is unknown.
inline bool zone_raw_gate_fits(const zone_raw_gate_inputs & in, size_t request) {
    return request <= zone_raw_gate_free_outside_arena(in);
}

}  // namespace ggml_sycl
