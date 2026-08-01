// Follow-up probe #3 for llama.cpp-403s.
//
// probe-device-init (serial device/context/queue init) and
// probe-kernel-launch (serial JIT compilation of up to 256 distinct
// kernels x 3 devices) are BOTH flat -- no TTM shmem growth whatsoever.
//
// The one thing neither exercises is CONCURRENCY: test-thread-safety's
// name is literal -- it loads one model per GPU plus a CPU copy and runs
// them from multiple threads *simultaneously*. This probe tests whether
// concurrent (not merely multi-device) SYCL queue creation + kernel
// submission from multiple host threads is what triggers the growth --
// e.g. a driver race in Level-Zero's per-thread command-list/allocator
// bookkeeping under concurrent load.
//
// Usage: ./probe-concurrent <ndevices> <nthreads> <iters_per_thread>
//   Each thread round-robins across the visible devices, creating its own
//   queue per device the first time it touches that device, then launches
//   a small parallel_for kernel iters_per_thread times.

#include <sycl/sycl.hpp>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static std::atomic<bool> g_stop{false};

static void sample_meminfo(const char * label) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    long shmem_kb = -1, avail_kb = -1;
    while (std::getline(f, line)) {
        if (line.rfind("Shmem:", 0) == 0) sscanf(line.c_str(), "Shmem: %ld kB", &shmem_kb);
        else if (line.rfind("MemAvailable:", 0) == 0) sscanf(line.c_str(), "MemAvailable: %ld kB", &avail_kb);
    }
    fprintf(stderr, "[meminfo] %-28s Shmem=%.2f GB  MemAvailable=%.2f GB\n",
            label, shmem_kb / 1048576.0, avail_kb / 1048576.0);
    fflush(stderr);
    if (avail_kb > 0 && avail_kb < 60L * 1048576L) {
        fprintf(stderr, "GUARD TRIP: MemAvailable < 60GB, signaling stop\n");
        g_stop = true;
    }
}

struct kernel_tag;

static void thread_body(sycl::device dev, int tid, int iters) {
    try {
        sycl::queue q(dev, sycl::property::queue::in_order());
        int * buf = sycl::malloc_device<int>(256, q);
        for (int i = 0; i < iters && !g_stop.load(); i++) {
            q.submit([&](sycl::handler & h) {
                h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) {
                    buf[id] = (int) id + tid + i;
                });
            });
            q.wait();
        }
        sycl::free(buf, q);
    } catch (sycl::exception & e) {
        fprintf(stderr, "thread %d: sycl exception: %s\n", tid, e.what());
    }
}

int main(int argc, char ** argv) {
    int want_n  = argc > 1 ? atoi(argv[1]) : 1;
    int nthr    = argc > 2 ? atoi(argv[2]) : 4;
    int iters   = argc > 3 ? atoi(argv[3]) : 20;

    sample_meminfo("start");

    std::vector<sycl::device> devices;
    auto platforms = sycl::platform::get_platforms();
    for (auto & p : platforms) {
        std::string pname = p.get_info<sycl::info::platform::name>();
        if (pname.find("Level-Zero") == std::string::npos && pname.find("Level Zero") == std::string::npos) continue;
        for (auto & d : p.get_devices()) if (d.is_gpu()) devices.push_back(d);
    }
    if ((int) devices.size() < want_n) {
        fprintf(stderr, "requested %d devices but only %zu visible\n", want_n, devices.size());
        return 1;
    }
    devices.resize(want_n);
    for (size_t i = 0; i < devices.size(); i++) {
        fprintf(stderr, "  dev%zu: %s\n", i, devices[i].get_info<sycl::info::device::name>().c_str());
    }
    fprintf(stderr, "spawning %d threads, %d iters each, round-robin over %d device(s)\n", nthr, iters, want_n);

    // Background sampler thread so we see progress even though the worker
    // threads are blocked in q.wait() most of the time.
    std::atomic<bool> sampler_done{false};
    std::thread sampler([&]() {
        while (!sampler_done.load()) {
            sample_meminfo("during concurrent run");
            if (g_stop.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    std::vector<std::thread> workers;
    for (int t = 0; t < nthr; t++) {
        workers.emplace_back(thread_body, devices[t % want_n], t, iters);
    }
    for (auto & w : workers) w.join();

    sampler_done = true;
    sampler.join();

    sample_meminfo("all threads joined");
    return g_stop.load() ? 2 : 0;
}
