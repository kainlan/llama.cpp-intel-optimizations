// Minimal standalone SYCL/Level-Zero probe for llama.cpp-403s.
//
// Purpose: isolate whether per-device SYCL/Level-Zero context+queue
// initialization alone (no ggml, no model load) accounts for the ~160-180 GB
// of TTM shmem that test-thread-safety consumes with a 19 MB model.
//
// Usage: ./probe-device-init <ndevices>
//   Respects ONEAPI_DEVICE_SELECTOR for filtering which devices are visible.
//   Initializes a queue+context on each of the first N visible devices, one
//   at a time, sampling /proc/meminfo Shmem/MemAvailable before and after
//   each device. Does a small USM device alloc+memset+free per device to
//   force real (non-lazy) driver init, not just enumeration.
//
// Safety: this program touches no model, allocates a few KB per device, and
// exits promptly. It is NOT in the never-loop family, but do not loop it
// back-to-back without checking /proc/meminfo between runs regardless.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void sample_meminfo(const char * label) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    long shmem_kb = -1, avail_kb = -1;
    while (std::getline(f, line)) {
        if (line.rfind("Shmem:", 0) == 0) {
            sscanf(line.c_str(), "Shmem: %ld kB", &shmem_kb);
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            sscanf(line.c_str(), "MemAvailable: %ld kB", &avail_kb);
        }
    }
    fprintf(stderr, "[meminfo] %-28s Shmem=%.2f GB  MemAvailable=%.2f GB\n",
            label, shmem_kb / 1048576.0, avail_kb / 1048576.0);
    fflush(stderr);
}

int main(int argc, char ** argv) {
    int want_n = argc > 1 ? atoi(argv[1]) : 1;

    sample_meminfo("start");

    std::vector<sycl::device> devices;
    try {
        auto platforms = sycl::platform::get_platforms();
        for (auto & p : platforms) {
            std::string pname = p.get_info<sycl::info::platform::name>();
            if (pname.find("Level-Zero") == std::string::npos &&
                pname.find("Level Zero") == std::string::npos) {
                continue;
            }
            for (auto & d : p.get_devices()) {
                if (d.is_gpu()) {
                    devices.push_back(d);
                }
            }
        }
    } catch (sycl::exception & e) {
        fprintf(stderr, "enumeration failed: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "found %zu level-zero GPU device(s) visible\n", devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        fprintf(stderr, "  dev%zu: %s\n", i,
                devices[i].get_info<sycl::info::device::name>().c_str());
    }

    if ((int) devices.size() < want_n) {
        fprintf(stderr, "requested %d devices but only %zu visible -- exiting\n",
                want_n, devices.size());
        return 1;
    }
    devices.resize(want_n);

    // Hold contexts/queues alive for the whole run, one per device, added
    // incrementally, sampling after each addition -- mirrors what
    // test-thread-safety does (one llama_context/model per GPU).
    std::vector<sycl::queue> queues;
    for (int i = 0; i < want_n; i++) {
        char label[64];
        snprintf(label, sizeof(label), "before dev%d init", i);
        sample_meminfo(label);

        sycl::queue q(devices[i], sycl::property::queue::in_order());

        // Force real init: alloc device USM, touch it, free it.
        void * p = sycl::malloc_device(4096, q);
        if (p) {
            q.memset(p, 0, 4096).wait();
            sycl::free(p, q);
        }

        queues.push_back(std::move(q));

        snprintf(label, sizeof(label), "after dev%d init", i);
        sample_meminfo(label);
    }

    sample_meminfo("all devices initialized, holding");

    // Hold briefly so an external sampler could also observe steady state if
    // ever needed; queues/contexts are destroyed at scope exit below.
    queues.clear();

    sample_meminfo("after queues destroyed");

    return 0;
}
