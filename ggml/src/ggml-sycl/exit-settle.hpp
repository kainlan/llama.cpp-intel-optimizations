#pragma once

// The wait before the process ends on a path that runs no teardown (std::_Exit, abort).
//
// The GPU driver keeps a submission ring running on each engine and parks the engine on memory of this process. A
// driver thread stops an idle ring a few milliseconds after the last submission. A process that ends before that
// leaves the engine parked on memory the kernel has already released: the kernel log then shows a fault and an engine
// reset, and now and then a memory error after which the next process on the card hangs. Waiting here lets the
// driver stop its rings while the process still exists.
//
// It only sleeps and prints. It takes no lock of the backend's (only stdio's own stderr lock) and touches no queue, so
// it cannot hang after a device loss, and the headers that end the process on misuse can include it without the
// backend.

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace ggml_sycl {

// Measured: 200 ms left no kernel line in 20 exits, where 16 of 22 exits without a wait left some. The driver's own
// bound is a few 5 ms timeouts.
constexpr long k_exit_settle_default_ms = 200;
constexpr long k_exit_settle_max_ms     = 5000;

// GGML_SYCL_FATAL_EXIT_SETTLE_MS: the wait in milliseconds. 0 disables it, a value above the maximum counts as the
// maximum, and a value that is not a whole non-negative number counts as the default.
inline long exit_settle_ms() noexcept {
    const char * env = std::getenv("GGML_SYCL_FATAL_EXIT_SETTLE_MS");
    if (env == nullptr || env[0] == '\0') {
        return k_exit_settle_default_ms;
    }
    char * end       = nullptr;
    errno            = 0;
    const long value = std::strtol(env, &end, 10);
    // A number too large for long is still a whole number above the maximum, so it counts as the maximum.
    if (errno == ERANGE && value == LONG_MAX && end != env && *end == '\0') {
        return k_exit_settle_max_ms;
    }
    if (errno != 0 || end == env || *end != '\0' || value < 0) {
        return k_exit_settle_default_ms;
    }
    return value > k_exit_settle_max_ms ? k_exit_settle_max_ms : value;
}

inline void exit_settle() noexcept {
    const long ms = exit_settle_ms();
    if (ms <= 0) {
        return;
    }
    std::fprintf(stderr,
                 "[SYCL-FATAL] settle: waiting %ld ms before the process ends, so the GPU driver can stop its "
                 "idle submission rings (GGML_SYCL_FATAL_EXIT_SETTLE_MS)\n",
                 ms);
    std::fflush(stderr);
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

}  // namespace ggml_sycl
