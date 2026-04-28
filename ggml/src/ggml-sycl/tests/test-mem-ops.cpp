#include "../mem-handle.hpp"
#include "../mem-ops.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

static bool check_bytes(const uint8_t * data, size_t size, uint8_t expected, const char * label) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != expected) {
            std::fprintf(stderr, "FAIL: %s byte %zu expected 0x%02x got 0x%02x\n", label, i, expected, data[i]);
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        std::printf("SKIP: no SYCL GPU devices available\n");
        return 0;
    }

    sycl::queue      q(gpus.front(), sycl::property::queue::in_order{});
    constexpr size_t size = 4096;

    uint8_t * host_a = sycl::malloc_host<uint8_t>(size, q);
    uint8_t * host_b = sycl::malloc_host<uint8_t>(size, q);
    uint8_t * dev_a  = sycl::malloc_device<uint8_t>(size, q);
    uint8_t * dev_b  = sycl::malloc_device<uint8_t>(size, q);
    if (!host_a || !host_b || !dev_a || !dev_b) {
        std::fprintf(stderr, "FAIL: allocation failed\n");
        sycl::free(host_a, q);
        sycl::free(host_b, q);
        sycl::free(dev_a, q);
        sycl::free(dev_b, q);
        return 1;
    }

    int  failed   = 0;
    auto host_a_h = ggml_sycl::mem_handle::from_direct(host_a, GGML_LAYOUT_AOS, false);
    auto host_b_h = ggml_sycl::mem_handle::from_direct(host_b, GGML_LAYOUT_AOS, false);
    auto dev_a_h  = ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0);
    auto dev_b_h  = ggml_sycl::mem_handle::from_direct(dev_b, GGML_LAYOUT_AOS, true, 0);

    std::memset(host_a, 0x11, size);
    std::memset(host_b, 0x00, size);
    ggml_sycl::mem_copy(host_b_h, host_a_h, size, q);
    failed += !check_bytes(host_b, size, 0x11, "H2H copy");

    std::memset(host_a, 0x22, size);
    auto h2d = ggml_sycl::mem_copy_async(dev_a_h, host_a_h, size, q);
    auto d2h = ggml_sycl::mem_copy_async(host_b_h, dev_a_h, size, q, { h2d });
    d2h.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x22, "H2D/D2H copy");

    auto fill = ggml_sycl::mem_fill_async(dev_a_h, 0x33, size, q);
    auto d2d  = ggml_sycl::mem_copy_async(dev_b_h, dev_a_h, size, q, { fill });
    auto back = ggml_sycl::mem_copy_async(host_b_h, dev_b_h, size, q, { d2d });
    back.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x33, "fill+D2D+D2H chain");

    // Temporary handles may die immediately after submission.  mem_ops must
    // keep any handle-owned lease alive until the returned event completes.
    auto temp_fill =
        ggml_sycl::mem_fill_async(ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0), 0x44, size, q);
    auto temp_copy = ggml_sycl::mem_copy_async(ggml_sycl::mem_handle::from_direct(host_b, GGML_LAYOUT_AOS, false),
                                               ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0),
                                               size, q, { temp_fill });
    temp_copy.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x44, "temporary handle event lifetime");

    sycl::free(host_a, q);
    sycl::free(host_b, q);
    sycl::free(dev_a, q);
    sycl::free(dev_b, q);

    if (failed) {
        std::fprintf(stderr, "FAIL: %d mem_ops checks failed\n", failed);
        return 1;
    }
    std::printf("PASS: mem_ops H2H/H2D/D2H/D2D/dependency checks passed\n");
    return 0;
}
