// Follow-up probe for llama.cpp-403s.
//
// probe-device-init.cpp showed raw device/context/queue init + a USM
// malloc/memset/free is FLAT across 1, 2, and 3 devices -- no TTM shmem
// growth at all. But `memset` on a SYCL queue typically lowers to a
// copy-engine fill, not a compute-engine kernel dispatch, so it never
// exercises the driver's JIT (IGC) kernel-compilation path that real
// ggml-sycl matmul/dequant/attention kernels use.
//
// This probe launches actual parallel_for compute kernels -- up to
// MAX_KERNELS distinct *template instantiations* (so distinct JIT-compiled
// binaries, matching how ggml-sycl's unified kernel dispatch has many
// XMX/ESIMD/MMVQ template variants) -- and samples /proc/meminfo after each
// batch. It aborts if MemAvailable drops under the 60 GB guard.
//
// Usage: ./probe-kernel-launch <ndevices> <nkernels (<= MAX_KERNELS)>

#include <sycl/sycl.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

static void sample_meminfo(const char * label, bool * abort_flag) {
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
    if (abort_flag && avail_kb > 0 && avail_kb < 60L * 1048576L) {
        fprintf(stderr, "GUARD TRIP: MemAvailable < 60GB, aborting\n");
        *abort_flag = true;
    }
}

// Distinct tag types force distinct kernel-name mangling -> distinct JIT
// compilations, mirroring many template-instantiated dispatch kernels.
template <int N> struct kernel_tag;

template <int N>
static void launch_one(sycl::queue & q, int * buf) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for<kernel_tag<N>>(sycl::range<1>(256), [=](sycl::id<1> i) {
            buf[i] = (int) i + N;
        });
    });
}

using launch_fn_t = void (*)(sycl::queue &, int *);

constexpr int MAX_KERNELS = 256;

template <std::size_t... Is>
constexpr std::array<launch_fn_t, sizeof...(Is)> make_table(std::index_sequence<Is...>) {
    return { &launch_one<(int) Is>... };
}

static const std::array<launch_fn_t, MAX_KERNELS> g_table = make_table(std::make_index_sequence<MAX_KERNELS>{});

int main(int argc, char ** argv) {
    int want_n   = argc > 1 ? atoi(argv[1]) : 1;
    int nkernels = argc > 2 ? atoi(argv[2]) : 64;
    if (nkernels > MAX_KERNELS) nkernels = MAX_KERNELS;

    bool stop = false;
    sample_meminfo("start", &stop);

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

    constexpr int BATCH = 16;

    std::vector<sycl::queue> queues;
    for (int d = 0; d < want_n && !stop; d++) {
        sycl::queue q(devices[d], sycl::property::queue::in_order());
        int * buf = sycl::malloc_device<int>(256, q);
        queues.push_back(std::move(q));

        char label[64];
        snprintf(label, sizeof(label), "dev%d: 0 kernels", d);
        sample_meminfo(label, &stop);

        int launched = 0;
        while (launched < nkernels && !stop) {
            int this_batch = std::min(BATCH, nkernels - launched);
            for (int k = 0; k < this_batch; k++) {
                g_table[launched + k](queues.back(), buf);
            }
            queues.back().wait();
            launched += this_batch;
            snprintf(label, sizeof(label), "dev%d: %d kernels", d, launched);
            sample_meminfo(label, &stop);
        }
    }

    sample_meminfo("all done, holding", &stop);
    queues.clear();
    sample_meminfo("after queues destroyed", nullptr);

    return stop ? 2 : 0;
}
