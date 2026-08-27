#include "../attn-host-pool.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously (llama.cpp-u2mz). Use an explicit check that
// always runs, per the test-kv-runtime-demotion.cpp precedent.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::AttnHostPool;

// This target needs -fsycl solely because attn-host-pool.hpp includes
// common.hpp for the sycl::queue type (init()'s signature) -- there is no
// device kernel here and no device enumeration beyond the explicit
// cpu_selector_v below, so it runs at any parallelism, in a subagent, and on
// a host with no GPU. Same shape as test-kv-slice-sizing.cpp.
int main() {
    sycl::queue q{ sycl::cpu_selector_v };

    AttnHostPool pool;

    // 1. inactive before init()
    CHECK(!pool.is_active(), "case 1: pool starts inactive");

    // 2. init() starts workers; submit() actually runs a task on one of them
    // and the future observes completion.
    pool.init(2, q);
    CHECK(pool.is_active(), "case 2: pool active after init()");
    {
        std::atomic<bool> ran{ false };
        auto              fut = pool.submit([&ran] { ran.store(true, std::memory_order_release); });
        fut.wait();
        CHECK(ran.load(std::memory_order_acquire), "case 2: submitted task actually ran");
    }

    // 3. an exception thrown by the task is observed through the future,
    // not std::terminate -- required so a step-3 caller's flush point can
    // surface a real attention-compute failure instead of crashing the
    // process.
    {
        auto fut   = pool.submit([] { throw std::runtime_error("boom"); });
        bool threw = false;
        try {
            fut.get();
        } catch (const std::runtime_error & e) {
            threw = std::string(e.what()) == "boom";
        }
        CHECK(threw, "case 3: task exception propagates through the future");
    }

    // 4. many tasks, few workers: every submission eventually completes
    // (queue does not drop work under contention).
    {
        constexpr int                  n = 50;
        std::atomic<int>               completed{ 0 };
        std::vector<std::future<void>> futures;
        futures.reserve(n);
        for (int i = 0; i < n; ++i) {
            futures.push_back(pool.submit([&completed] { completed.fetch_add(1, std::memory_order_relaxed); }));
        }
        for (auto & f : futures) {
            f.wait();
        }
        CHECK(completed.load(std::memory_order_relaxed) == n, "case 4: every submitted task completed");
    }

    // 5. double init() is a no-op (warns, does not re-spawn threads or hang).
    pool.init(2, q);
    CHECK(pool.is_active(), "case 5: pool still active after redundant init()");

    // 6. shutdown() joins workers and is idempotent.
    pool.shutdown();
    CHECK(!pool.is_active(), "case 6: pool inactive after shutdown()");
    pool.shutdown();  // must not hang or crash on a second call
    CHECK(!pool.is_active(), "case 6: redundant shutdown() is a safe no-op");

    std::printf("test-attn-host-pool: all ok\n");
    return 0;
}
