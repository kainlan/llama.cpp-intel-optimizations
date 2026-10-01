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
// The queues are the backend's own, not queues this test constructs: the owner's is the exposed device 0's
// ggml_backend_sycl_context::stream(), the call the SET_ROWS stage makes, and the source's is
// ggml_sycl::get_shared_context_queue(1), the per-device single-device-context queue the MoE and split paths
// use for a secondary card, created in this process by ggml_sycl::init_shared_context_queues() (the call
// those paths make; nothing else creates them, so a bare get returns null). Neither path builds a multi-device Level Zero context, which is the DEVICE_LOST
// risk on compute-runtime 26.x. The scheduler exposes only device 0 by default ("Multi-GPU: exposing only
// device 0 to scheduler"), so ggml_backend_sycl_get_device_count() is 1 on a two-card host and
// ggml_backend_sycl_init(1) is not available: the probe counts PHYSICAL devices through
// ggml_sycl::test_physical_device_count() (the backend's total_gpu_count), never the scheduler-visible
// count; tests/test-sycl-staging-probe-source.py pins that. Every buffer is a
// unified_allocate() allocation held by a mem_handle that is released before either backend is freed.
// The gate's host-USM flags are probe code, not a design path.
//
// Bounded steps and controls. A cross-context depends_on can hang on this stack (G0: the submit's host task
// waits on a foreign-context event that never signals), so every step runs under a deadline enforced by a
// watchdog thread that prints "FAIL: HANG at <step>" and leaves with _Exit(1); stdout is unbuffered and each
// step prints begin/end lines. The controls run first, each bounded, in an order that puts the one that can
// hang last: g (the gate alone: the host releases it and it must finish), i (leg 1 alone, its own event
// waited), ii (leg 2 alone), iv (the dependency as a host wait between the legs, a CONTROL only: production
// takes no host wait), then iii, the cross-context depends_on under the gate. If i hangs the defect is leg 1
// itself, not the dependency; if g hangs the gate cannot hold-then-release and iii proves nothing.
//
// Exit codes: 0 for returns_before and for blocks (both are pre-registered outcomes the lead records),
// 1 for void or a setup failure, 77 when fewer than two PHYSICAL SYCL devices are held (counted with
// ggml_sycl::test_physical_device_count() before any backend init).
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
#include "ggml-sycl/ggml-sycl-test.hpp"
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
#include <functional>
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

// A step budget. A wait inside the probe can hang forever on this stack (a cross-context dependency lowered
// to a host task that waits on a foreign-context event), so every step runs under a deadline a watchdog
// thread enforces: past it, the thread names the step and leaves the process with _Exit(1). Progress lines
// are written unbuffered before and after each step, so the hang point is in the output.
std::atomic<const char *> g_step_name{ nullptr };
std::atomic<long long>    g_step_deadline_ms{ 0 };

long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void step_watchdog_main() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const char * name = g_step_name.load(std::memory_order_acquire);
        if (name != nullptr && now_ms() > g_step_deadline_ms.load(std::memory_order_acquire)) {
            std::printf("FAIL: HANG at %s\n", name);
            std::fflush(stdout);
            _Exit(1);
        }
    }
}

struct step_scope {
    const char * name;
    long long    t0;

    step_scope(const char * n, int budget_ms) : name(n), t0(now_ms()) {
        std::printf("[SYCL-STAGING-PROBE] begin %s budget_ms=%d\n", name, budget_ms);
        g_step_deadline_ms.store(t0 + budget_ms, std::memory_order_release);
        g_step_name.store(name, std::memory_order_release);
    }

    ~step_scope() {
        g_step_name.store(nullptr, std::memory_order_release);
        std::printf("[SYCL-STAGING-PROBE] end %s ms=%lld\n", name, now_ms() - t0);
    }
};

constexpr int k_step_budget_ms = 20000;

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

    // The scheduler exposes only device 0 by default, so ggml_backend_sycl_get_device_count() is 1 on a
    // two-card host; the physical count is what decides the skip. Zero devices (no setvars) lands here too.
    // ggml_backend_sycl_init(i) asserts on an index past the scheduler-visible count, so the exposed
    // device 0 is the only one initialised as a backend.
    const int physical_devices = ggml_sycl::test_physical_device_count();
    if (physical_devices < 2 || ggml_backend_sycl_get_device_count() < 1) {
        std::printf("SKIP: fewer than two physical SYCL GPU devices available (physical=%d)\n", physical_devices);
        return LLAMA_TEST_EXIT_SKIP;
    }

    ggml_backend_t backend_owner = ggml_backend_sycl_init(0);
    if (!backend_owner) {
        std::printf("FAIL: ggml_backend_sycl_init(0) failed on a host that reports two physical devices\n");
        return 1;
    }
    // The queues are created lazily by the MoE and split paths, so a standalone process creates them the way
    // those paths do: init_shared_context_queues(total_gpu_count), one single-device context per card.
    ggml_sycl::init_shared_context_queues(physical_devices);
    sycl::queue * q_source_ptr = ggml_sycl::get_shared_context_queue(1);
    if (q_source_ptr == nullptr) {
        std::printf("FAIL: no shared-context queue for physical device 1\n");
        ggml_backend_free(backend_owner);
        return 1;
    }

    auto *        owner_ctx = static_cast<ggml_backend_sycl_context *>(backend_owner->context);
    sycl::queue & q_owner   = *owner_ctx->stream(0, 0);
    sycl::queue & q_source  = *q_source_ptr;

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::thread(step_watchdog_main).detach();

    int  rc          = 1;
    char verdict[32] = "";
    {
        // The handles live in this block, so every buffer is released before the backend is freed.
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
            void *         src_ptr     = src_dev.ptr;
            void *         bounce_ptr  = bounce.ptr;
            void *         dst_ptr     = dst_dev.ptr;

            // Which context knows which pointer: a host-USM bounce allocated in one context and copied from
            // on a queue of the other is the first suspect for a leg that never completes.
            const auto kind = [&](void * ptr, sycl::queue & q) {
                return static_cast<int>(sycl::get_pointer_type(ptr, q.get_context()));
            };
            std::printf(
                "[SYCL-STAGING-PROBE] contexts same=%d pointer_type(owner_ctx,source_ctx) "
                "src_dev=(%d,%d) bounce=(%d,%d) dst_dev=(%d,%d) flags=(%d,%d) (0 host, 1 device, 2 shared, 3 "
                "unknown)\n",
                q_owner.get_context() == q_source.get_context() ? 1 : 0, kind(src_ptr, q_owner),
                kind(src_ptr, q_source), kind(bounce_ptr, q_owner), kind(bounce_ptr, q_source), kind(dst_ptr, q_owner),
                kind(dst_ptr, q_source), kind(release_dev, q_owner), kind(release_dev, q_source));

            // Runs `fn` as a bounded step. A throw is a step failure, not a crash; a hang is the watchdog's.
            const auto run_step = [&](const char * name, const std::function<void()> & fn) {
                bool ok_step = true;
                {
                    step_scope scope(name, k_step_budget_ms);
                    try {
                        fn();
                    } catch (const std::exception & ex) {
                        std::printf("FAIL: step %s threw: %s\n", name, ex.what());
                        ok_step = false;
                    } catch (...) {
                        std::printf("FAIL: step %s threw a non-standard exception\n", name);
                        ok_step = false;
                    }
                }
                std::printf("[SYCL-STAGING-PROBE] control=%s result=%s\n", name, ok_step ? "completes" : "THREW");
                return ok_step;
            };
            bool controls_ok = true;

            // Control g: the gate alone. The host sets the release flag after 100 ms and the gate kernel must
            // finish; if it does not, the gate cannot hold-then-release and nothing below it is readable.
            controls_ok &= run_step("control_g_gate_alone", [&]() {
                *release               = 0;
                sycl::event gate_alone = q_source.submit([&](sycl::handler & h) {
                    h.single_task([=]() {
                        volatile int * flag = release_dev;
                        uint64_t       n    = 0;
                        while (*flag == 0 && n < k_gate_spin_cap) {
                            ++n;
                        }
                    });
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                *release = 1;
                gate_alone.wait();
            });

            // Control i: leg 1 alone, no dependency, waited on its own event.
            controls_ok &= run_step("control_i_leg1_alone", [&]() {
                sycl::event e = q_source.submit([&](sycl::handler & h) { h.memcpy(bounce_ptr, src_ptr, k_bytes); });
                e.wait();
            });

            // Control ii: leg 2 alone, no dependency.
            controls_ok &= run_step("control_ii_leg2_alone", [&]() {
                sycl::event e = q_owner.submit([&](sycl::handler & h) { h.memcpy(dst_ptr, bounce_ptr, k_bytes); });
                e.wait();
            });

            // Control iv: the dependency expressed as a host wait, leg 1 done before leg 2 is submitted. A
            // CONTROL only: production takes no host wait. It is run before iii because iii can hang.
            controls_ok &= run_step("control_iv_host_wait_between", [&]() {
                sycl::event l1 = q_source.submit([&](sycl::handler & h) { h.memcpy(bounce_ptr, src_ptr, k_bytes); });
                l1.wait();
                sycl::event l2 = q_owner.submit([&](sycl::handler & h) { h.memcpy(dst_ptr, bounce_ptr, k_bytes); });
                l2.wait();
            });

            // Experiment iii: the cross-context depends_on, with leg 1 held in flight behind the gate.
            *release = 0;
            *marker  = 0;
            sycl::event gate;
            sycl::event leg1;
            sycl::event marked;
            bool        setup_threw = false;
            {
                step_scope scope("iii_submit_gate_leg1_marker", k_step_budget_ms);
                try {
                    // The gate: holds the source queue until the host releases it.
                    gate   = q_source.submit([&](sycl::handler & h) {
                        h.single_task([=]() {
                            volatile int * flag = release_dev;
                            uint64_t       n    = 0;
                            while (*flag == 0 && n < k_gate_spin_cap) {
                                ++n;
                            }
                        });
                    });
                    // Leg 1 behind the gate, then the marker behind leg 1.
                    leg1   = q_source.submit([&](sycl::handler & h) {
                        h.depends_on(gate);
                        h.memcpy(bounce_ptr, src_ptr, k_bytes);
                    });
                    marked = q_source.submit([&](sycl::handler & h) {
                        h.depends_on(leg1);
                        h.single_task([=]() {
                            // The host reads this flag without a queue operation, so make leg 1's copy
                            // visible to the system before the flag is (design row 113's marker form).
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            *reinterpret_cast<volatile int *>(marker_dev) = 1;
                        });
                    });
                } catch (const std::exception & ex) {
                    setup_threw = true;
                    std::printf("FAIL: the gate/leg 1/marker submit threw: %s\n", ex.what());
                }
            }

            if (!setup_threw) {
                // The watchdog owns the release: after leg 2's submit returned, or k_release_after_ms,
                // whichever is first. `release_sent` is set before the flag is written, so a reading of it
                // at leg 2's return can only be true when the timeout fired first.
                std::atomic<bool> leg2_returned{ false };
                std::atomic<bool> release_sent{ false };
                std::thread       releaser([&]() {
                    const auto start = std::chrono::steady_clock::now();
                    while (!leg2_returned.load(std::memory_order_acquire)) {
                        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now() - start)
                                                .count();
                        if (waited >= k_release_after_ms) {
                            release_sent.store(true, std::memory_order_release);
                            break;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    *release = 1;
                });

                // Leg 2 depends on leg 1; the thing under test is how long this submit takes to return. A
                // throw out of the submit must not unwind past the joinable releaser thread
                // (std::terminate): it is caught, the releaser is joined, and the run is a probe failure.
                // A submit that never returns is the step watchdog's: "FAIL: HANG at iii_leg2_submit".
                sycl::event leg2;
                bool        submit_threw = false;
                std::string submit_error;
                const auto  t0 = std::chrono::steady_clock::now();
                {
                    step_scope scope("iii_leg2_submit", k_step_budget_ms);
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
                }
                const auto t1                = std::chrono::steady_clock::now();
                const int  marker_at_return  = *marker;
                const bool release_at_return = release_sent.load(std::memory_order_acquire);
                leg2_returned.store(true, std::memory_order_release);
                releaser.join();

                // Everything drains before the buffers go, each wait a named bounded step.
                bool drained = true;
                try {
                    if (!submit_threw) {
                        step_scope scope("iii_wait_leg2", k_step_budget_ms);
                        leg2.wait();
                    }
                    {
                        step_scope scope("iii_wait_marker", k_step_budget_ms);
                        marked.wait();
                    }
                    {
                        step_scope scope("iii_wait_queues", k_step_budget_ms);
                        q_source.wait();
                        q_owner.wait();
                    }
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
                    const staging_probe_verdict v =
                        staging_probe_classify(marker_at_return != 0, release_at_return, true);
                    std::snprintf(verdict, sizeof(verdict), "%s", staging_probe_verdict_name(v));
                    std::printf(
                        "[SYCL-STAGING-PROBE] leg2_submit=%s submit_ms=%.3f marker_at_return=%d "
                        "release_sent_at_return=%d controls_ok=%d\n",
                        verdict, submit_ms, marker_at_return, release_at_return ? 1 : 0, controls_ok ? 1 : 0);
                    rc = (v == staging_probe_verdict::VOID || !controls_ok) ? 1 : 0;
                }
            }
        }
    }

    ggml_backend_free(backend_owner);
    return rc;
}
