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
// a submit that returns only after the 2 s release is "blocks" (a set marker then is expected: leg 1
// finished once the gate opened); a marker already set while the release has NOT been sent is "void"
// (the gate did not hold) and proves nothing. The classification is staging_probe_classify() in
// sycl-staging-probe-verdict.h, which test-sycl-staging-probe-verdict drives over its whole table.
//
// The queues are the backend's own: ggml_backend_sycl_init() per device and ggml_backend_sycl_context::
// stream(), the same call the SET_ROWS stage makes, not a queue this test constructs. Every buffer is a
// unified_allocate() allocation held by a mem_handle that is released before either backend is freed.
// The gate's host-USM flags are probe code, not a design path.
//
// Exit codes: 0 for returns_before and for blocks (both are pre-registered outcomes the lead records),
// 1 for void or a setup failure, 77 when fewer than two SYCL devices are visible (counted with
// ggml_backend_sycl_get_device_count() before any backend init, which asserts on an absent index).
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
#include "sycl-staging-probe-verdict.h"
#include "test-skip.h"

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

namespace {

constexpr size_t   k_bytes            = 1u << 20;
constexpr uint64_t k_gate_spin_cap    = 1ull << 32;  // a hang guard only: the host always releases first
constexpr int      k_release_after_ms = 2000;

struct probe_buffer {
    ggml_sycl::mem_handle handle;
    void *                ptr = nullptr;
};

bool alloc_probe_buffer(probe_buffer & out, sycl::queue & queue, int device, size_t bytes, bool host_pinned) {
    ggml_sycl::alloc_request req{};
    req.queue                                    = &queue;
    req.device                                   = device;
    req.size                                     = bytes;
    req.intent.role                              = ggml_sycl::alloc_role::STAGING;
    req.intent.category                          = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device           = !host_pinned;
    req.intent.constraints.must_host_pinned      = host_pinned;
    req.intent.constraints.require_host_usm_base = host_pinned;
    out.handle                                   = ggml_sycl::unified_allocate(req);
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

    // ggml_backend_sycl_init(i) asserts on an index past the device count and carries on to construct a
    // context for it, so the skip is decided before any init. Zero devices (no setvars) lands here too.
    if (ggml_backend_sycl_get_device_count() < 2) {
        std::printf("SKIP: fewer than two SYCL GPU devices available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    ggml_backend_t backend_owner  = ggml_backend_sycl_init(0);
    ggml_backend_t backend_source = ggml_backend_sycl_init(1);
    if (!backend_owner || !backend_source) {
        std::printf("FAIL: ggml_backend_sycl_init failed on a host that reports two devices\n");
        if (backend_owner) {
            ggml_backend_free(backend_owner);
        }
        if (backend_source) {
            ggml_backend_free(backend_source);
        }
        return 1;
    }

    auto *        owner_ctx  = static_cast<ggml_backend_sycl_context *>(backend_owner->context);
    auto *        source_ctx = static_cast<ggml_backend_sycl_context *>(backend_source->context);
    sycl::queue & q_owner    = *owner_ctx->stream(0, 0);
    sycl::queue & q_source   = *source_ctx->stream(1, 0);

    int  rc          = 1;
    char verdict[32] = "";
    {
        // The handles live in this block, so every buffer is released before either backend is freed.
        probe_buffer src_dev;
        probe_buffer bounce;
        probe_buffer dst_dev;
        probe_buffer flags;  // release, marker
        const bool   ok = alloc_probe_buffer(src_dev, q_source, 1, k_bytes, false) &&
                        alloc_probe_buffer(bounce, q_source, 1, k_bytes, true) &&
                        alloc_probe_buffer(dst_dev, q_owner, 0, k_bytes, false) &&
                        alloc_probe_buffer(flags, q_source, 1, 64, true);
        if (!ok) {
            std::printf("FAIL: could not allocate the probe's buffers through unified_allocate\n");
        } else {
            // Two ints in host USM: the release flag the gate spins on and the marker leg 1's completion
            // sets.
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
            sycl::event gate   = q_source.submit([&](sycl::handler & h) {
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
                h.single_task([=]() {
                    // The host reads this flag without a queue operation, so make leg 1's copy visible
                    // to the system before the flag is (design row 113's marker form).
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    *reinterpret_cast<volatile int *>(marker_dev) = 1;
                });
            });

            // The watchdog owns the release: after leg 2's submit returned, or k_release_after_ms,
            // whichever is first. `release_sent` is set before the flag is written, so a reading of it
            // at leg 2's return can only be true when the timeout fired first.
            std::atomic<bool> leg2_returned{ false };
            std::atomic<bool> release_sent{ false };
            std::thread       watchdog([&]() {
                const auto start = std::chrono::steady_clock::now();
                while (!leg2_returned.load(std::memory_order_acquire)) {
                    const auto waited =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                            .count();
                    if (waited >= k_release_after_ms) {
                        release_sent.store(true, std::memory_order_release);
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                *release = 1;
            });

            // Leg 2 depends on leg 1; the thing under test is how long this submit takes to return. A throw
            // out of the submit must not unwind past the joinable watchdog thread (std::terminate): it is
            // caught, the watchdog is released and joined, and the run is reported as a probe failure.
            const auto  t0 = std::chrono::steady_clock::now();
            sycl::event leg2;
            bool        submit_threw = false;
            std::string submit_error;
            try {
                leg2 = q_owner.submit([&](sycl::handler & h) {
                    h.depends_on(leg1);
                    h.memcpy(dst_ptr, bounce_ptr, k_bytes);
                });
            } catch (const std::exception & ex) {
                submit_threw = true;
                submit_error = ex.what();
            } catch (...) {
                submit_threw = true;
                submit_error = "non-standard exception";
            }
            const auto t1                = std::chrono::steady_clock::now();
            const int  marker_at_return  = *marker;
            const bool release_at_return = release_sent.load(std::memory_order_acquire);
            leg2_returned.store(true, std::memory_order_release);
            watchdog.join();

            // Everything drains before the buffers go: leg 2 last, then the marker. A drain that throws
            // is a probe failure too, not a crash.
            bool drained = true;
            try {
                if (!submit_threw) {
                    leg2.wait();
                }
                marked.wait();
                q_source.wait();
                q_owner.wait();
            } catch (const std::exception & ex) {
                drained      = false;
                submit_error = ex.what();
            } catch (...) {
                drained      = false;
                submit_error = "non-standard exception";
            }

            if (submit_threw || !drained) {
                std::printf("FAIL: the probe's %s threw: %s\n", submit_threw ? "leg 2 submit" : "drain",
                            submit_error.c_str());
            } else {
                const double                submit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                const staging_probe_verdict v = staging_probe_classify(marker_at_return != 0, release_at_return, true);
                std::snprintf(verdict, sizeof(verdict), "%s", staging_probe_verdict_name(v));
                std::printf(
                    "[SYCL-STAGING-PROBE] leg2_submit=%s submit_ms=%.3f marker_at_return=%d "
                    "release_sent_at_return=%d\n",
                    verdict, submit_ms, marker_at_return, release_at_return ? 1 : 0);
                rc = v == staging_probe_verdict::VOID ? 1 : 0;
            }
        }
    }

    ggml_backend_free(backend_source);
    ggml_backend_free(backend_owner);
    return rc;
}
