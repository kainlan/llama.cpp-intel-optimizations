// Isolated repro: does q.ext_oneapi_submit_barrier() (no waitlist) wedge
// the L0 driver on Arc B580 + xe + oneAPI 2025.3?
//
// Spawned because the 'barrier' mitigation in minimal-repro.cpp triggers
// a guc_exec_queue_timedout_job kernel-WARNING + GT-reset on the B580.
//
// Smaller test: submit one async H2D copy, then a single bare barrier,
// then a wait. If the wait wedges past 1 s -> barrier itself is the bug.

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>

int main() {
    using clock = std::chrono::steady_clock;
    try {
        sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
        std::fprintf(stderr,
                     "[probe] device=%s\n",
                     q.get_device().get_info<sycl::info::device::name>().c_str());

        constexpr size_t N = 64ULL * 1024ULL * 1024ULL;
        char * host = sycl::malloc_host<char>(N, q);
        char * dev  = sycl::malloc_device<char>(N, q);
        std::memset(host, 0xA5, N);

        // Submit one H2D copy.
        sycl::event e_cp = q.memcpy(dev, host, N);
        std::fprintf(stderr, "[probe] copy submitted\n");

        // Submit a bare empty-waitlist barrier.
        const auto t0 = clock::now();
        sycl::event e_br = q.ext_oneapi_submit_barrier();
        const auto t1 = clock::now();
        std::fprintf(stderr,
                     "[probe] barrier submit_us=%ld\n",
                     std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        // Wait on the copy event (the actual question: does this wait return?).
        const auto t2 = clock::now();
        e_cp.wait();
        const auto t3 = clock::now();
        std::fprintf(stderr,
                     "[probe] copy.wait_us=%ld\n",
                     std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

        // Wait on the barrier event.
        const auto t4 = clock::now();
        e_br.wait();
        const auto t5 = clock::now();
        std::fprintf(stderr,
                     "[probe] barrier.wait_us=%ld\n",
                     std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count());

        q.wait_and_throw();
        sycl::free(host, q);
        sycl::free(dev,  q);
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[probe] SYCL exception: %s\n", e.what());
        return 3;
    }
}
