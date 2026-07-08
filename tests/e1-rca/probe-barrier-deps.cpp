// Companion to probe-barrier-bug.cpp.
//
// Question: does ext_oneapi_submit_barrier(WITH a non-empty waitlist) wedge
// the same way as the empty-waitlist variant? If yes, the entire
// submit_barrier call is unsafe on this driver. If no, only the empty form
// (ze "all prior" semantics) trips the GuC timeout.

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    using clock = std::chrono::steady_clock;
    try {
        sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
        std::fprintf(stderr,
                     "[probe-deps] device=%s\n",
                     q.get_device().get_info<sycl::info::device::name>().c_str());

        constexpr size_t N = 64ULL * 1024ULL * 1024ULL;
        char * host = sycl::malloc_host<char>(N, q);
        char * dev  = sycl::malloc_device<char>(N, q);
        if (!host || !dev) {
            std::fprintf(stderr, "[probe-deps] alloc failed (host=%p dev=%p)\n",
                         (void *) host, (void *) dev);
            return 2;
        }
        std::memset(host, 0xA5, N);

        sycl::event e_cp = q.memcpy(dev, host, N);
        std::fprintf(stderr, "[probe-deps] copy submitted\n");

        const auto t0 = clock::now();
        std::vector<sycl::event> deps{e_cp};
        sycl::event e_br = q.ext_oneapi_submit_barrier(deps);
        const auto t1 = clock::now();
        std::fprintf(stderr,
                     "[probe-deps] barrier(deps) submit_us=%ld\n",
                     std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        const auto t2 = clock::now();
        e_br.wait();
        const auto t3 = clock::now();
        std::fprintf(stderr,
                     "[probe-deps] barrier.wait_us=%ld\n",
                     std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

        q.wait_and_throw();
        sycl::free(host, q);
        sycl::free(dev,  q);
        return 0;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "[probe-deps] SYCL exception: %s\n", e.what());
        return 3;
    }
}
