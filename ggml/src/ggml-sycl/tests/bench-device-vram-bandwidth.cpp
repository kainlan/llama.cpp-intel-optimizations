// TKV-13 (B2) step 5.2: B50 VRAM read bandwidth roofline prerequisite
// (docs/plans/2026-08-27-tkv13-b2-addendum.md §5.2). A standalone SYCL
// microbench -- NOT part of the ggml-sycl backend or its runtime, so the
// unified-cache-is-the-sole-allocator rule (CLAUDE.md "SYCL Memory
// Ownership") does not apply to it, exactly as it does not apply to
// bench-dnnl-ops.cpp: this program allocates its own throwaway device
// buffer via the raw SYCL API because it measures the device's memory
// system in isolation, with no model loaded and no unified_cache instance
// ever constructed. Not a ctest target -- lead-run only, single invocation,
// selector pinned (docs/plans/2026-08-27-tkv13-b2-addendum.md §6: "still
// counts as GPU access and stays lead-run").
//
// Usage (matches every other gate in this repo -- ONEAPI_DEVICE_SELECTOR
// controls which physical device is used, never a flag baked into the
// program):
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./bench-device-vram-bandwidth --bytes 268435456 --iters 20
//
// Times ONLY the kernel's device execution window via SYCL profiling
// events (command_start/command_end), never host chrono around the submit
// call -- per [[host-chrono-cannot-see-past-submission-backpressure]], a
// host-side wall-clock around q.submit()/wait() can measure queue
// backpressure rather than device throughput. The queue is constructed
// with sycl::property::queue::enable_profiling{}, without which
// get_profiling_info() would throw.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

struct Args {
    size_t bytes = 256ull * 1024 * 1024;  // 256 MB default
    int    iters = 20;
};

bool parse_args(int argc, char ** argv, Args * args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bytes" && i + 1 < argc) {
            args->bytes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--iters" && i + 1 < argc) {
            args->iters = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            return false;
        }
    }
    return args->bytes > 0 && args->iters > 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        std::fprintf(stderr, "usage: %s [--bytes N] [--iters N]\n", argv[0]);
        return 1;
    }

    sycl::queue q{ sycl::property::queue::enable_profiling{} };
    std::printf("bench-device-vram-bandwidth: device=%s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const size_t n_floats = args.bytes / sizeof(float);
    float *      dev_buf  = sycl::malloc_device<float>(n_floats, q);
    float *      dev_out  = sycl::malloc_device<float>(1, q);
    if (!dev_buf || !dev_out) {
        std::fprintf(stderr, "device allocation failed (%zu bytes)\n", args.bytes);
        return 1;
    }

    // Initialize once (not timed) so every read touches real, resident
    // pages rather than a lazily-committed/zero-fill-on-demand region.
    q.submit([&](sycl::handler & cgh) {
         cgh.parallel_for(sycl::range<1>(n_floats), [=](sycl::id<1> i) { dev_buf[(size_t) i] = 1.0f; });
     }).wait();

    std::vector<double> gbps;
    gbps.reserve(args.iters);

    for (int it = 0; it < args.iters; ++it) {
        // A tree-reduction accumulator so the compiler cannot prove the
        // reads are dead; dev_out is read back to force real completion,
        // but that host-side read happens AFTER the timed event, never
        // inside the timed window.
        sycl::event e = q.submit([&](sycl::handler & cgh) {
            auto reduction = sycl::reduction(dev_out, sycl::plus<float>());
            cgh.parallel_for(sycl::range<1>(n_floats), reduction,
                             [=](sycl::id<1> i, auto & sum) { sum += dev_buf[(size_t) i]; });
        });
        e.wait();

        const auto   t_start_ns = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto   t_end_ns   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        const double s          = static_cast<double>(t_end_ns - t_start_ns) / 1e9;
        const double gb         = static_cast<double>(args.bytes) / (1024.0 * 1024.0 * 1024.0);
        gbps.push_back(gb / s);
    }

    std::sort(gbps.begin(), gbps.end());
    const double gbps_min    = gbps.front();
    const double gbps_max    = gbps.back();
    const double gbps_median = gbps[gbps.size() / 2];

    float readback = 0.0f;
    q.memcpy(&readback, dev_out, sizeof(float)).wait();

    std::printf(
        "bench-device-vram-bandwidth: bytes=%zu (%.2f MB) iters=%d GB/s min=%.2f median=%.2f max=%.2f "
        "(readback=%.3f, ignore)\n",
        args.bytes, args.bytes / (1024.0 * 1024.0), args.iters, gbps_min, gbps_median, gbps_max, readback);

    sycl::free(dev_buf, q);
    sycl::free(dev_out, q);
    return 0;
}
