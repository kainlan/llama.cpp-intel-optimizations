// v2 of repro-drain-barrier: how MANY owners is the boundary expected to find?
//
// v1 had exactly ONE owner thread, so it could only ever observe 0 or 1 holders.
// Reporting "the TOCTOU predicts exactly one live allocation" from that was
// reading a harness parameter as a prediction about the system.
//
// test-thread-safety runs several contexts. With N of them doing get_rows while
// one runs the graph boundary, the boundary can find anywhere from 0 to N-1
// holders. This histograms that, so the prediction is measured rather than
// assumed.
//
// Device-free: DIRECT handle views over stack buffers, default-constructed
// (already complete) events.
//
//   ./repro-drain-barrier-multi [n_owners]

#include "mem-handle.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    const int  n_owners        = argc > 1 ? std::atoi(argv[1]) : 3;
    const long kIterations     = 1500;
    const auto kHoldWindow     = std::chrono::microseconds(50);

    if (n_owners < 1 || n_owners > 32) {
        std::fprintf(stderr, "n_owners must be 1..32\n");
        return 2;
    }

    std::atomic<int>  live{ 0 };          // owners currently holding
    std::atomic<int>  started{ 0 };
    std::atomic<int>  done{ 0 };
    std::vector<std::atomic<long>> hist(n_owners + 1);
    for (auto & h : hist) {
        h.store(0);
    }

    std::vector<std::thread> owners;
    owners.reserve(n_owners);
    for (int t = 0; t < n_owners; ++t) {
        owners.emplace_back([&]() {
            alignas(64) unsigned char buf[64] = {};
            started.fetch_add(1, std::memory_order_release);
            for (long i = 0; i < kIterations; ++i) {
                ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_direct(buf, GGML_LAYOUT_AOS, false);
                live.fetch_add(1, std::memory_order_acq_rel);

                std::this_thread::sleep_for(kHoldWindow);

                std::vector<ggml_sycl::mem_handle> retained;
                retained.push_back(std::move(h));
                live.fetch_sub(1, std::memory_order_acq_rel);
                ggml_sycl::retain_handles_until_event(std::move(retained), sycl::event{});
            }
            done.fetch_add(1, std::memory_order_release);
        });
    }

    std::thread boundary([&]() {
        while (started.load(std::memory_order_acquire) < n_owners) {
            std::this_thread::yield();
        }
        while (done.load(std::memory_order_acquire) < n_owners) {
            ggml_sycl::drain_retained_handles(true);
            int n = live.load(std::memory_order_acquire);
            if (n < 0) {
                n = 0;
            }
            if (n > n_owners) {
                n = n_owners;
            }
            hist[n].fetch_add(1, std::memory_order_relaxed);
            // Do not spin flat out; a real boundary happens per graph.
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    });

    for (auto & t : owners) {
        t.join();
    }
    boundary.join();

    long total = 0;
    for (auto & h : hist) {
        total += h.load();
    }
    std::printf("owners=%d  boundary_crossings=%ld\n", n_owners, total);
    for (int n = 0; n <= n_owners; ++n) {
        const long c = hist[n].load();
        std::printf("  found %d live holder(s): %8ld  (%5.2f%%)%s\n", n, c, total ? (100.0 * c) / total : 0.0,
                    n == 0 ? "   <- reset would succeed" : "   <- reset REFUSED, graph launch aborts");
    }
    return 0;
}
