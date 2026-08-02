// Device-free measurement: does drain_retained_handles(true) actually serialise
// the graph-boundary reset against a concurrent scratch owner?
//
// ggml_sycl_graph_boundary_reset_arenas() does:
//     drain_retained_handles(true) -> unified_cache_arena_reset(d) -> check-empty
// and the check aborts if the zone is not fully free. That is only sound if the
// drain is a barrier against every other context's ownership.
//
// It is not. drain_retained_handles(true) waits for state.queue to empty and
// state.active to reach 0. A thread that is holding an allocation but has not
// YET handed its handle to retain_handles_until_event() is in neither. It is
// invisible to the barrier.
//
// This program measures how often the barrier returns "all clear" while another
// thread demonstrably still owns a handle. Each such sample is one graph launch
// that would have found a live allocation in the shared zone and aborted.
//
// No device, no queue, no unified cache: DIRECT handle views over a stack buffer
// and default-constructed (already complete) events.

#include "mem-handle.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    // The real owner holds its scratch across a stream_dma submission -- tens of
    // microseconds at least, not a few hundred cycles. Model that honestly.
    constexpr long kOwnerIterations = 2000;
    constexpr auto kHoldWindow      = std::chrono::microseconds(50);

    alignas(64) unsigned char buf[64] = {};

    std::atomic<bool> owner_started{ false };
    std::atomic<bool> owner_holds{ false };
    std::atomic<bool> owner_done{ false };
    std::atomic<long> drains_total{ 0 };
    std::atomic<long> drains_while_held{ 0 };

    // The "other context": acquires a handle, does a little work with it, then
    // hands it to the event-bound release path -- exactly get_rows' shape.
    std::thread owner([&]() {
        owner_started.store(true, std::memory_order_release);
        for (long i = 0; i < kOwnerIterations; ++i) {
            ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_direct(buf, GGML_LAYOUT_AOS, false);
            owner_holds.store(true, std::memory_order_release);

            // Stand-in for the window between allocating scratch and submitting
            // the DMA that consumes it.
            std::this_thread::sleep_for(kHoldWindow);

            std::vector<ggml_sycl::mem_handle> retained;
            retained.push_back(std::move(h));
            owner_holds.store(false, std::memory_order_release);
            ggml_sycl::retain_handles_until_event(std::move(retained), sycl::event{});
        }
        owner_done.store(true, std::memory_order_release);
    });

    // The "graph boundary": drains, then would reset and demand an empty zone.
    std::thread boundary([&]() {
        // Do not start sampling before the owner is running, or a fast owner
        // makes the run vacuous (measured: one such run reported drains=1).
        while (!owner_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!owner_done.load(std::memory_order_acquire)) {
            ggml_sycl::drain_retained_handles(true);
            // This is the instant between the drain returning and the reset
            // running. If another context owns an allocation here, the reset is
            // refused and graph launch aborts.
            const bool held = owner_holds.load(std::memory_order_acquire);
            drains_total.fetch_add(1, std::memory_order_relaxed);
            if (held) {
                drains_while_held.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    owner.join();
    boundary.join();

    const long total = drains_total.load();
    const long held  = drains_while_held.load();
    std::printf("drains=%ld  drains_that_returned_while_another_thread_held=%ld  (%.2f%%)\n", total, held,
                total ? (100.0 * held) / total : 0.0);
    if (held == 0) {
        std::printf("NOT REPRODUCED in this run\n");
        return 1;
    }
    std::printf("REPRODUCED: drain_retained_handles(true) is not a barrier against a concurrent owner\n");
    return 0;
}
