//
// What the lazy IO engine's tests share (th32 §5.2): a bounded poll, the gate a read source holds reads on, and the
// read sources that serve the fixture file's bytes. It names no link seam, so the host test that links the fixture side
// of the pool's seam and the device test that links the backend's own pool both include it. Each test includes it once.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "lazy-table-test-util.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

namespace lazy_io_test {

// Polls `done` until it holds or the hook bound passes: an observation, never a sleep that stands in for an event.
template <typename F> bool wait_for(F done) {
    const auto deadline = std::chrono::steady_clock::now() + lazy_table_test::k_hook_bound;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// A gate every holder blocks on until the arm releases it; the arm waits for the holds it needs to observe.
struct gate {
    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  holders  = 0;
    bool                    released = false;

    void hold() {
        std::unique_lock<std::mutex> lock(mutex);
        ++holders;
        cv.notify_all();
        cv.wait(lock, [this] { return released; });
    }

    bool wait_holders(size_t n) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, lazy_table_test::k_hook_bound, [this, n] { return holders >= n; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }

    bool is_released() {
        std::lock_guard<std::mutex> lock(mutex);
        return released;
    }
};

// The fixture file's byte at `offset`: what every read source below writes, so a landing copy is checked by value.
inline uint8_t file_byte(uint64_t offset) {
    return static_cast<uint8_t>(offset * 131u + 7u);
}

// Serves every read with the fixture file's bytes.
inline long fill_source(void * ctx, int fd, uint8_t * dst, size_t length, uint64_t offset) {
    (void) ctx;
    (void) fd;
    for (size_t i = 0; i < length; ++i) {
        dst[i] = file_byte(offset + i);
    }
    return static_cast<long>(length);
}

// Holds every read on the gate in `ctx`, then serves it.
inline long gated_source(void * ctx, int fd, uint8_t * dst, size_t length, uint64_t offset) {
    static_cast<gate *>(ctx)->hold();
    return fill_source(nullptr, fd, dst, length, offset);
}

}  // namespace lazy_io_test
