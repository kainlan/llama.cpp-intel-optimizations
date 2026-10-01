// Cross-device staging probe for the SET_ROWS stage (design row 113, G0 "the cross-device staging
// dependency").
//
// Question: when the staging legs are chained as leg 1 (source device -> host bounce) and leg 2
// (host bounce -> owner device) with leg 2 depending on leg 1's event, does the submit of leg 2 return
// before leg 1 has completed, or does the runtime block the submitting thread until leg 1 is done? The
// design takes no host wait on the staging dependency, so "returns before" keeps the legs as written and
// "blocks" switches leg 2 to the device-marker form.
//
// Control. A submit that returns quickly proves nothing if leg 1 was already done, so leg 1 is held in
// flight:
//   * leg 1 is submitted behind a gate kernel on the source queue that spins on a host-USM release flag;
//   * the host sets the flag only after leg 2's submit has returned OR 2 s have passed (a watchdog
//     thread owns that decision, so a blocking submit cannot keep the gate shut forever);
//   * a marker kernel, chained behind leg 1 on the source queue, writes a host-USM flag when leg 1 has
//     completed. The marker is a device write, never a status query: event status is not trusted here.
// A submit that returns while the gate holds (marker unset, release not yet sent) is "returns_before";
// a submit that returns only after the 2 s release is "blocks"; a run whose marker is already set when
// leg 2's submit returns is "void" (the gate did not hold) and proves nothing.
//
// The queues are the backend's own: ggml_backend_sycl_init() per device and ggml_backend_sycl_context::
// stream(), the same call the SET_ROWS stage makes, not a queue this test constructs. Every buffer is a
// unified_allocate() allocation held by a mem_handle for the probe's duration. The gate's host-USM flags
// are probe code, not a design path.
//
// Exit codes: 0 for returns_before and for blocks (both are pre-registered outcomes the lead records),
// 1 for void or a setup failure, 77 when fewer than two SYCL devices are visible.
//
// GPU binaries in this fork are run only from the lead session, one at a time (CLAUDE.md). The default
// selector below pins the two discrete cards; the iGPU is never enumerated.
//
//   source /opt/intel/oneapi/setvars.sh --force
//   ./build/bin/test-sycl-staging-probe        # [SYCL-STAGING-PROBE] leg2_submit=returns_before|blocks|void

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "test-skip.h"

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {

constexpr size_t   k_bytes          = 1u << 20;
constexpr uint64_t k_gate_spin_cap  = 1ull << 32;  // a hang guard only: the host always releases first
constexpr int      k_release_after_ms = 2000;

struct probe_buffer {
    ggml_sycl::mem_handle handle;
    void *                ptr = nullptr;
};

bool alloc_probe_buffer(probe_buffer & out, sycl::queue & queue, int device, size_t bytes, bool host_pinned) {
    ggml_sycl::alloc_request req{};
    req.queue                                 = &queue;
    req.device                                = device;
    req.size                                  = bytes;
    req.intent.role                           = ggml_sycl::alloc_role::STAGING;
    req.intent.category                       = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device        = !host_pinned;
    req.intent.constraints.must_host_pinned   = host_pinned;
    req.intent.constraints.require_host_usm_base = host_pinned;
    out.handle = ggml_sycl::unified_allocate(req);
    if (!out.handle.valid()) {
        return false;
    }
    const auto resolved = out.handle.resolve(device);
    out.ptr             = resolved.ptr;
    return out.ptr != nullptr && resolved.on_device == !host_pinned;
}

}  // namespace

int main(int, char ** argv) {
    // ctest supplies ONEAPI_DEVICE_SELECTOR through the registration; a bare run needs it set before
    // libsycl memoizes device enumeration, and setenv() alone is too late (libccl's static initializer
    // constructs a sycl::event at load, llama.cpp-2x3m), so re-exec with it already in the environment.
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        if (setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1", 1) == 0) {
            execv("/proc/self/exe", argv);
        }
        std::fprintf(stderr, "warning: re-exec failed (%s); run with ONEAPI_DEVICE_SELECTOR=level_zero:0,1 set\n",
                     std::strerror(errno));
    }

    ggml_backend_t backend_owner  = ggml_backend_sycl_init(0);
    ggml_backend_t backend_source = ggml_backend_sycl_init(1);
    if (!backend_owner || !backend_source) {
        std::printf("SKIP: fewer than two SYCL GPU devices available\n");
        if (backend_owner) {
            ggml_backend_free(backend_owner);
        }
        if (backend_source) {
            ggml_backend_free(backend_source);
        }
        return LLAMA_TEST_EXIT_SKIP;
    }

    auto *        owner_ctx  = static_cast<ggml_backend_sycl_context *>(backend_owner->context);
    auto *        source_ctx = static_cast<ggml_backend_sycl_context *>(backend_source->context);
    sycl::queue & q_owner    = *owner_ctx->stream(0, 0);
    sycl::queue & q_source   = *source_ctx->stream(1, 0);

    probe_buffer src_dev;
    probe_buffer bounce;
    probe_buffer dst_dev;
    probe_buffer flags;  // release, marker
    bool         ok = alloc_probe_buffer(src_dev, q_source, 1, k_bytes, false) &&
                      alloc_probe_buffer(bounce, q_source, 1, k_bytes, true) &&
                      alloc_probe_buffer(dst_dev, q_owner, 0, k_bytes, false) &&
                      alloc_probe_buffer(flags, q_source, 1, 64, true);
    if (!ok) {
        std::printf("FAIL: could not allocate the probe's buffers through unified_allocate\n");
        ggml_backend_free(backend_source);
        ggml_backend_free(backend_owner);
        return 1;
    }

    // Two ints in host USM: the release flag the gate spins on and the marker leg 1's completion sets.
    int *          release_dev = static_cast<int *>(flags.ptr);
    int *          marker_dev  = release_dev + 1;
    volatile int * release     = release_dev;
    volatile int * marker      = marker_dev;
    *release                   = 0;
    *marker                    = 0;
    void * src_ptr             = src_dev.ptr;
    void * bounce_ptr          = bounce.ptr;
    void * dst_ptr             = dst_dev.ptr;

    // The gate: holds the source queue until the host releases it.
    sycl::event gate = q_source.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            volatile int * flag = release_dev;
            uint64_t       n    = 0;
            while (*flag == 0 && n < k_gate_spin_cap) {
                ++n;
            }
        });
    });
    // Leg 1 behind the gate, then the marker behind leg 1.
    sycl::event leg1   = q_source.submit([&](sycl::handler & h) {
        h.depends_on(gate);
        h.memcpy(bounce_ptr, src_ptr, k_bytes);
    });
    sycl::event marked = q_source.submit([&](sycl::handler & h) {
        h.depends_on(leg1);
        h.single_task([=]() { *reinterpret_cast<volatile int *>(marker_dev) = 1; });
    });

    // The watchdog owns the release: after leg 2's submit returned, or k_release_after_ms, whichever
    // is first. `released_by_timeout` records which, so a blocking submit is told from a fast one.
    std::atomic<bool> leg2_returned{ false };
    std::atomic<bool> released_by_timeout{ false };
    std::thread       watchdog([&]() {
        const auto start = std::chrono::steady_clock::now();
        while (!leg2_returned.load(std::memory_order_acquire)) {
            const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
            if (waited >= k_release_after_ms) {
                released_by_timeout.store(true, std::memory_order_release);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        *release = 1;
    });

    // Leg 2 depends on leg 1; the thing under test is how long this submit takes to return.
    const auto  t0   = std::chrono::steady_clock::now();
    sycl::event leg2 = q_owner.submit([&](sycl::handler & h) {
        h.depends_on(leg1);
        h.memcpy(dst_ptr, bounce_ptr, k_bytes);
    });
    const auto t1             = std::chrono::steady_clock::now();
    const int  marker_at_return = *marker;
    const bool timed_out_first  = released_by_timeout.load(std::memory_order_acquire);
    leg2_returned.store(true, std::memory_order_release);
    watchdog.join();

    const double submit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const char * verdict   = "returns_before";
    if (marker_at_return != 0) {
        verdict = "void";  // leg 1 had already completed: the gate did not hold
    } else if (timed_out_first) {
        verdict = "blocks";
    }

    // Everything drains before the buffers go: leg 2 last, then the marker.
    leg2.wait();
    marked.wait();
    q_source.wait();
    q_owner.wait();

    std::printf("[SYCL-STAGING-PROBE] leg2_submit=%s submit_ms=%.3f marker_at_return=%d released_by_timeout=%d\n",
                verdict, submit_ms, marker_at_return, timed_out_first ? 1 : 0);

    ggml_backend_free(backend_source);
    ggml_backend_free(backend_owner);
    return std::strcmp(verdict, "void") == 0 ? 1 : 0;
}
