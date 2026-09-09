// Shared single-work-item SYCL spin kernel for GPU tests that need a
// device-kernel event of a controllable duration (llama.cpp-me60 F7).
//
// Previously duplicated, byte-for-byte identical, in
// tests/test-sycl-event-status-blocking-probe.cpp and
// tests/test-sycl-onednn-graph-scratch-direct.cpp, each carrying its own
// comment claiming it "matches" the other file's copy "exactly" -- nothing
// enforced that claim, so the two copies could silently drift apart.
// Extracted here, next to tests/sycl-selector-fallback.hpp, as the single
// definition all three now include: the two files above, plus
// ggml/src/ggml-sycl/tests/test-unified-runtime-alloc.cpp, whose own
// shutdown-repro cases need the identical kernel shape for their timings to
// stay comparable with the other two files'.
#pragma once

#include <sycl/sycl.hpp>

// Single-work-item spin loop reading and writing a device-global cell every
// iteration (so the loop cannot be folded away at compile time: `acc` and
// `*cell` each iteration's value depends on the PREVIOUS iteration's write
// to device memory, which the compiler cannot know ahead of time) --
// deliberately serial, not parallel: the point is wall-clock duration on
// one device compute unit, matching every caller's own kernel shape exactly
// so timings measured in different files stay comparable.
static inline sycl::event submit_spin_kernel(sycl::queue & q, int * cell, long long iterations) {
    return q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            int acc = 0;
            for (long long i = 0; i < iterations; ++i) {
                acc   = acc + *cell + 1;
                *cell = acc;
            }
        });
    });
}
