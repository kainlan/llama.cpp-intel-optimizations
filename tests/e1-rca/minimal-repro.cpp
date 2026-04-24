// Minimal SYCL repro of the L0 DirectSubmission event.wait() hang (m09zb).
//
// Hypothesis: post-init, async H2D copies on an in-order SYCL queue may not be
// flushed by the L0 DirectSubmission controller in a timely fashion, so a later
// event.wait() blocks indefinitely (sched_yield in queryStatus, controller
// thread parked in pthread_cond_wait).
//
// This repro removes llama.cpp/ggml entirely. It submits N async H2D copies
// via the default in-order queue (BCS/Copy engine for D2H/H2D on Arc) and
// then waits on intermediate events to mimic the staging-pool acquire path
// in common.hpp:1843 ("acquire" reuses a slot with a pending event by calling
// .wait()).
//
// Build (run via build.sh):  ../sycl-canary/build.sh-style icpx -fsycl ...
// Run:  ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 60 ./minimal-repro
//
// Exit codes:
//   0  -> ran to completion (no wedge observed)
//   1  -> wedge detected (per-event wait > 1 s)
//   2  -> initialization or alloc failure
//   3  -> SYCL exception
//   124 (from `timeout`) -> hard hang past `timeout` window

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr size_t CHUNK_BYTES   = 64ULL * 1024ULL * 1024ULL;  // 64 MB
constexpr int    N_CHUNKS      = 64;                          // 4 GB total
constexpr int    LOOKBACK      = 16;                          // wait on event N-16
constexpr long   WEDGE_US      = 1'000'000;                   // 1 s threshold

// Optional periodic mitigations — controlled via env so the same source
// produces the baseline AND each mitigation variant.
//   MITIGATION=barrier      -> ext_oneapi_submit_barrier({}) every BARRIER_PERIOD chunks
//   MITIGATION=wait         -> q.wait_and_throw() every WAIT_PERIOD chunks
//   MITIGATION=none         -> baseline (default)
// Tunables: BARRIER_PERIOD (default 4), WAIT_PERIOD (default 8)
//
// ZE_SERIALIZE / xe driver tunables are set externally in the runner script,
// not in this binary -- they affect L0 driver behavior wholesale.
enum class mitigation { none, barrier, wait_and_throw };

mitigation parse_mitigation() {
    const char * m = std::getenv("MITIGATION");
    if (!m) return mitigation::none;
    if (std::strcmp(m, "barrier") == 0)        return mitigation::barrier;
    if (std::strcmp(m, "wait")    == 0)        return mitigation::wait_and_throw;
    return mitigation::none;
}

int env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (!v) return fallback;
    int parsed = std::atoi(v);
    return parsed > 0 ? parsed : fallback;
}
}  // namespace

int main() {
    using clock = std::chrono::steady_clock;
    using sycl::event;
    using sycl::queue;

    const mitigation mit            = parse_mitigation();
    const int        barrier_period = env_int("BARRIER_PERIOD", 4);
    const int        wait_period    = env_int("WAIT_PERIOD", 8);

    const char * mit_str = "none";
    if (mit == mitigation::barrier)        mit_str = "barrier";
    if (mit == mitigation::wait_and_throw) mit_str = "wait_and_throw";
    std::fprintf(stderr,
                 "[repro] mitigation=%s barrier_period=%d wait_period=%d\n",
                 mit_str, barrier_period, wait_period);

    try {
        queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

        std::fprintf(stderr,
                     "[repro] device=%s vendor=%s\n",
                     q.get_device().get_info<sycl::info::device::name>().c_str(),
                     q.get_device().get_info<sycl::info::device::vendor>().c_str());

        const size_t total = CHUNK_BYTES * N_CHUNKS;
        char * host = sycl::malloc_host<char>(total, q);
        char * dev  = sycl::malloc_device<char>(total, q);
        if (!host || !dev) {
            std::fprintf(stderr, "[repro] alloc failed (host=%p dev=%p)\n",
                         (void *) host, (void *) dev);
            return 2;
        }
        std::memset(host, 0xA5, total);

        std::vector<event> events;
        events.reserve(N_CHUNKS);

        long max_wait_us = 0;

        for (int i = 0; i < N_CHUNKS; ++i) {
            event e = q.memcpy(dev + (size_t) i * CHUNK_BYTES,
                               host + (size_t) i * CHUNK_BYTES,
                               CHUNK_BYTES);
            events.push_back(e);

            if (mit == mitigation::barrier && (i % barrier_period) == (barrier_period - 1)) {
                q.ext_oneapi_submit_barrier();
            } else if (mit == mitigation::wait_and_throw &&
                       (i % wait_period) == (wait_period - 1)) {
                q.wait_and_throw();
            }

            if (i >= LOOKBACK) {
                const auto t0     = clock::now();
                events[i - LOOKBACK].wait();
                const auto t1     = clock::now();
                const long wait_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                if (wait_us > max_wait_us) {
                    max_wait_us = wait_us;
                }
                std::fprintf(stderr, "[repro] chunk=%d wait=%ld us\n", i, wait_us);
                if (wait_us > WEDGE_US) {
                    std::fprintf(stderr,
                                 "[repro] WEDGE detected: chunk=%d wait_us=%ld threshold=%ld\n",
                                 i, wait_us, WEDGE_US);
                    sycl::free(host, q);
                    sycl::free(dev,  q);
                    return 1;
                }
            }
        }

        const auto t_drain0 = clock::now();
        q.wait_and_throw();
        const auto t_drain1 = clock::now();
        const long drain_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_drain1 - t_drain0).count();
        std::fprintf(stderr, "[repro] drain=%ld us max_wait=%ld us\n", drain_us, max_wait_us);

        sycl::free(host, q);
        sycl::free(dev,  q);
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[repro] SYCL exception: %s\n", e.what());
        return 3;
    }
}
