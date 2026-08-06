#include "mem-handle.hpp"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

void cleanup() {
    ggml_sycl::drain_retained_handles(true);
    ggml_sycl::release_graph_retained_handles();
    ggml_sycl::drain_retained_handles(true);
}

int test_single_publisher_blocks_drain_until_publish() {
    cleanup();

    alignas(64) std::uint8_t buf[64] = {};
    auto ticket = ggml_sycl::begin_retained_handle_publish();
    if (ggml_sycl::drain_retained_handles(true, 20)) {
        return fail("drain reported clear while a publisher ticket was active");
    }

    ggml_sycl::retain_handles_until_event({ ggml_sycl::mem_handle::from_direct(buf, GGML_LAYOUT_AOS, false) },
                                          sycl::event{}, std::move(ticket));
    if (!ggml_sycl::drain_retained_handles(true, 1000)) {
        return fail("drain did not clear after the publisher handed off its handle");
    }

    std::puts("PASS: single publisher ticket blocks drain until retain handoff completes");
    return 0;
}

int test_abandoned_publisher_ticket_unblocks_drain() {
    cleanup();

    {
        auto ticket = ggml_sycl::begin_retained_handle_publish();
        if (ggml_sycl::drain_retained_handles(true, 20)) {
            return fail("drain reported clear while an unpublished ticket was active");
        }
        (void) ticket;
    }

    if (!ggml_sycl::drain_retained_handles(true, 1000)) {
        return fail("ticket destruction did not unblock drain");
    }

    std::puts("PASS: scope exit decrements publisher count without an explicit retain");
    return 0;
}

int test_multiple_publishers_keep_drain_blocked_until_last_handoff() {
    cleanup();

    alignas(64) std::uint8_t a[64] = {};
    alignas(64) std::uint8_t b[64] = {};
    alignas(64) std::uint8_t c[64] = {};

    auto ta = ggml_sycl::begin_retained_handle_publish();
    auto tb = ggml_sycl::begin_retained_handle_publish();
    auto tc = ggml_sycl::begin_retained_handle_publish();

    ggml_sycl::retain_handles_until_event({ ggml_sycl::mem_handle::from_direct(a, GGML_LAYOUT_AOS, false) },
                                          sycl::event{}, std::move(ta));
    ggml_sycl::retain_handles_until_event({ ggml_sycl::mem_handle::from_direct(b, GGML_LAYOUT_AOS, false) },
                                          sycl::event{}, std::move(tb));
    if (ggml_sycl::drain_retained_handles(true, 20)) {
        return fail("drain ignored another publisher that was still in flight");
    }

    ggml_sycl::retain_handles_until_event({ ggml_sycl::mem_handle::from_direct(c, GGML_LAYOUT_AOS, false) },
                                          sycl::event{}, std::move(tc));
    if (!ggml_sycl::drain_retained_handles(true, 1000)) {
        return fail("drain did not clear after the last publisher handed off");
    }

    std::puts("PASS: multi-publisher drain waits for the last outstanding handoff");
    return 0;
}

}  // namespace

int main() {
    if (int rc = test_single_publisher_blocks_drain_until_publish()) {
        return rc;
    }
    if (int rc = test_abandoned_publisher_ticket_unblocks_drain()) {
        return rc;
    }
    if (int rc = test_multiple_publishers_keep_drain_blocked_until_last_handoff()) {
        return rc;
    }
    return 0;
}
