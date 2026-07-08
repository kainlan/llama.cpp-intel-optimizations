#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr int N_WORK_GROUPS = 1024;
static constexpr int SG_SIZE       = 16;

// FNV-64 hash over a float buffer
static uint64_t fnv64(const float * data, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(data);
    size_t n_bytes = count * sizeof(float);
    for (size_t i = 0; i < n_bytes; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Run one pass of the kernel; fills h_max/h_sum and returns FNV-64 hash
static uint64_t run_pass(sycl::queue & q, float * max_out, float * sum_out,
                         std::vector<float> & h_max, std::vector<float> & h_sum) {
    sycl::range<1> global(N_WORK_GROUPS * SG_SIZE);
    sycl::range<1> local(SG_SIZE);

    q.submit([&](sycl::handler & h) {
        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                auto sg      = item.get_sub_group();
                int wg_id    = static_cast<int>(item.get_group(0));
                int lane_id  = static_cast<int>(sg.get_local_id());

                float v = sycl::sin(wg_id * 13.0f + lane_id * 7.0f) + lane_id * 0.3141f;

                float max_val = sycl::reduce_over_group(sg, v, sycl::maximum<float>{});
                float sum_val = sycl::reduce_over_group(sg, v, sycl::plus<float>{});

                if (lane_id == 0) {
                    max_out[wg_id] = max_val;
                    sum_out[wg_id] = sum_val;
                }
            });
    });
    q.wait();

    q.memcpy(h_max.data(), max_out, N_WORK_GROUPS * sizeof(float)).wait();
    q.memcpy(h_sum.data(), sum_out, N_WORK_GROUPS * sizeof(float)).wait();

    // Build combined buffer for hashing: [max[0..N-1], sum[0..N-1]]
    std::vector<float> combined(N_WORK_GROUPS * 2);
    std::memcpy(combined.data(),                 h_max.data(), N_WORK_GROUPS * sizeof(float));
    std::memcpy(combined.data() + N_WORK_GROUPS, h_sum.data(), N_WORK_GROUPS * sizeof(float));

    return fnv64(combined.data(), combined.size());
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});

        auto dev = q.get_device();
        std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

        float * d_max = sycl::malloc_device<float>(N_WORK_GROUPS, q);
        float * d_sum = sycl::malloc_device<float>(N_WORK_GROUPS, q);

        std::vector<float> h_max(N_WORK_GROUPS);
        std::vector<float> h_sum(N_WORK_GROUPS);

        uint64_t hashes[3];
        for (int pass = 0; pass < 3; pass++) {
            hashes[pass] = run_pass(q, d_max, d_sum, h_max, h_sum);
            std::printf("Pass %d hash: %016llx\n", pass, (unsigned long long)hashes[pass]);
        }

        // h_max/h_sum already contain the last pass's results
        std::printf("\nSample results (wg_id 0..7):\n");
        for (int i = 0; i < 8 && i < N_WORK_GROUPS; i++) {
            std::printf("  wg_id=%d  max=%.8f  sum=%.8f\n", i, h_max[i], h_sum[i]);
        }

        sycl::free(d_max, q);
        sycl::free(d_sum, q);

        bool all_match = (hashes[0] == hashes[1]) && (hashes[1] == hashes[2]);
        std::printf("\nIn-run determinism: %s\n", all_match ? "PASS" : "FAIL");
        return all_match ? 0 : 1;
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
