// Variant of minimal-repro that interleaves compute kernels with H2D copies
// across multiple in-order queues, to better mimic the llama.cpp pattern that
// triggers the m09zb wedge.
//
// In llama.cpp, by the time the staging pool's acquire() does
// pending_event.wait() (common.hpp:1863), the SYCL backend has already:
//   1. Initialized device contexts and multiple per-stream queues
//   2. Possibly run probe kernels / built oneDNN engines
//   3. Begun submitting compute kernels for the upload barrier dance
//
// The bare async-H2D repro (minimal-repro.cpp) does NONE of that and does NOT
// wedge -- so this variant adds:
//   * a 'warmup' kernel submitted on the same in-order queue
//   * a second in-order queue used in parallel (mimics multi-stream)
//   * an event.wait() *cross-queue* (q2 waits on event from q1)

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr size_t CHUNK_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr int    N_CHUNKS    = 64;
constexpr int    LOOKBACK    = 16;
constexpr long   WEDGE_US    = 1'000'000;

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

    const int n_warmup_kernels = env_int("WARMUP_KERNELS", 4);
    const int n_streams        = env_int("STREAMS", 2);

    std::fprintf(stderr,
                 "[repro-cmp] warmup_kernels=%d streams=%d\n",
                 n_warmup_kernels, n_streams);

    try {
        sycl::device dev_obj{sycl::gpu_selector_v};
        sycl::context shared{dev_obj};
        std::vector<queue> qs;
        for (int i = 0; i < n_streams; ++i) {
            qs.emplace_back(shared, dev_obj, sycl::property::queue::in_order{});
        }
        queue & q0 = qs[0];
        queue & q1 = qs[n_streams > 1 ? 1 : 0];

        std::fprintf(stderr,
                     "[repro-cmp] device=%s\n",
                     q0.get_device().get_info<sycl::info::device::name>().c_str());

        const size_t total = CHUNK_BYTES * N_CHUNKS;
        char * host = sycl::malloc_host<char>(total, q0);
        char * dev  = sycl::malloc_device<char>(total, q0);
        if (!host || !dev) {
            std::fprintf(stderr, "[repro-cmp] alloc failed\n");
            return 2;
        }
        std::memset(host, 0xA5, total);

        // Warmup kernels: simple parallel_for that touches device memory.
        for (int i = 0; i < n_warmup_kernels; ++i) {
            queue & q = qs[i % n_streams];
            event e = q.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
                    dev[idx[0]] = static_cast<char>(idx[0] & 0xff);
                });
            });
            e.wait();
        }
        std::fprintf(stderr, "[repro-cmp] warmup complete\n");

        std::vector<event> events;
        events.reserve(N_CHUNKS);
        long max_wait_us = 0;

        for (int i = 0; i < N_CHUNKS; ++i) {
            // Alternate copies across queues to mimic llama.cpp's stream
            // rotation in the buffer-set path.
            queue & qi = qs[i % n_streams];
            event e = qi.memcpy(dev + (size_t) i * CHUNK_BYTES,
                                host + (size_t) i * CHUNK_BYTES,
                                CHUNK_BYTES);
            events.push_back(e);

            if (i >= LOOKBACK) {
                const auto t0      = clock::now();
                events[i - LOOKBACK].wait();
                const auto t1      = clock::now();
                const long wait_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                if (wait_us > max_wait_us) max_wait_us = wait_us;
                std::fprintf(stderr, "[repro-cmp] chunk=%d wait=%ld us\n", i, wait_us);
                if (wait_us > WEDGE_US) {
                    std::fprintf(stderr,
                                 "[repro-cmp] WEDGE: chunk=%d wait=%ld us\n",
                                 i, wait_us);
                    // Drain every queue before free to match the success-path
                    // teardown at the bottom of main; otherwise q1's allocations
                    // can leak when n_streams > 1.
                    for (auto & q : qs) q.wait_and_throw();
                    sycl::free(host, q0);
                    sycl::free(dev,  q0);
                    return 1;
                }
            }

            // Cross-queue dependency: q1 waits on q0's last event every 8 iters.
            if (n_streams > 1 && (i % 8) == 7) {
                event dep = qs[0].submit([&](sycl::handler & h) {
                    h.depends_on(events.back());
                    h.parallel_for(sycl::range<1>{64}, [=](sycl::id<1> idx) {
                        dev[idx[0]] ^= 0x01;
                    });
                });
                dep.wait();
            }
        }

        const auto t0 = clock::now();
        for (auto & q : qs) q.wait_and_throw();
        const auto t1 = clock::now();
        const long drain_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::fprintf(stderr, "[repro-cmp] drain=%ld us max_wait=%ld us\n",
                     drain_us, max_wait_us);

        sycl::free(host, q0);
        sycl::free(dev,  q0);
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[repro-cmp] SYCL exception: %s\n", e.what());
        return 3;
    }
}
