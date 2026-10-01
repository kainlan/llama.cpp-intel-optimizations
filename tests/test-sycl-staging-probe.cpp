// Cross-device staging probe for the SET_ROWS stage (design row 113, G0 "the cross-device staging
// dependency").
//
// Question: when the staging legs are chained as leg 1 (source device -> host bounce) and leg 2
// (host bounce -> owner device) with leg 2 depending on leg 1's event, does the submit of leg 2 return
// before leg 1 has completed, or does the runtime block the submitting thread until leg 1 is done? The
// design takes no host wait on the staging dependency, so "returns before" keeps the legs as written and
// "blocks" switches leg 2 to the device-marker form.
//
// Control. A submit that returns quickly proves nothing if leg 1 was already done, so leg 1 is held behind a
// CALIBRATED BUSY KERNEL on the source queue: pure device arithmetic, sized by a warm-up run inside the probe
// to about k_target_busy_ms of measured run time (the probe prints busy_iters and the measured busy_ms).
// Leg 1 depends on it in order on the same queue, so leg 1 cannot complete before about busy_ms. The probe
// then reads three host clocks from the busy kernel's submit: when leg 2's submit returned (return_ms) and
// when leg 2 had completed (done_ms), against busy_ms. The classification is staging_probe_classify() in
// sycl-staging-probe-verdict.h, driven over its whole table by test-sycl-staging-probe-verdict, with its
// thresholds printed in the result line.
//
// Nothing here polls a word the host writes, and the host never reads a word a device wrote: discrete
// Battlemage has no usm_atomic_host_allocations, so neither direction has a visibility guarantee (G0's first
// probe held leg 1 behind a gate kernel spinning on a host-USM release flag, and the gate never opened). A
// "must-fire" control is a kernel of known duration the host waits on, never a host-released gate.
//
// The queues are the backend's own, not queues this test constructs: the owner's is the exposed device 0's
// ggml_backend_sycl_context::stream(), the call the SET_ROWS stage makes, and the source's is
// ggml_sycl::get_shared_context_queue(1), the per-device single-device-context queue the MoE and split paths
// use for a secondary card, created in this process by ggml_sycl::init_shared_context_queues() (the call
// those paths make; nothing else creates them, so a bare get returns null). Neither path builds a
// multi-device Level Zero context, which is the DEVICE_LOST risk on compute-runtime 26.x. The scheduler
// exposes only device 0 by default ("Multi-GPU: exposing only device 0 to scheduler"), so
// ggml_backend_sycl_get_device_count() is 1 on a two-card host and ggml_backend_sycl_init(1) is not
// available: the probe counts PHYSICAL devices through ggml_sycl::test_physical_device_count() (the
// backend's total_gpu_count), never the scheduler-visible count; tests/test-sycl-staging-probe-source.py
// pins that. Every buffer is a unified_allocate() allocation held by a mem_handle that is released before
// the backend is freed.
//
// Bounded steps and controls. A cross-context depends_on can hang on this stack (the submit's host task
// waits on a foreign-context event that never signals), so every step runs under a deadline enforced by a
// watchdog thread that prints "FAIL: HANG at <step>" and leaves with _Exit(1); stdout is unbuffered and each
// step prints begin/end lines. After calibration the controls run, each bounded, with the one that can hang
// last: g (the busy kernel alone completes within 2x and at least half of its calibrated time), i (leg 1
// alone, its own event waited), ii (leg 2 alone), iv (the dependency as a host wait between the legs, a
// CONTROL only: production takes no host wait), then iii, the cross-context depends_on behind the busy
// kernel. If i hangs the defect is leg 1 itself, not the dependency.
//
// Bytes. Timing alone cannot show a copy that moved stale or no bytes, so src holds a pattern, the bounce and
// the destination start as distinct sentinels, and leg 2's destination is read back and compared after every
// control that moves bytes and after iii (control iv is the positive control for iii's check). A mismatch is
// "FAIL: leg 2 destination bytes wrong" and rc=1. Every step, backend init, queue creation, allocation and
// teardown included, is bounded by the watchdog, which starts first, and a drain guard waits both queues before
// any probe buffer's handle is released, on every path out.
//
// Exit codes: 0 for returns_before and for blocks (both are pre-registered outcomes the lead records),
// 1 for void, undecided, a failed control or a setup failure, 77 when fewer than two PHYSICAL SYCL devices
// are held (counted with ggml_sycl::test_physical_device_count() before any backend init).
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

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t k_bytes          = 1u << 20;
constexpr double k_target_busy_ms = 400.0;

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

// Microseconds on the steady clock, for the busy-kernel timing.
long long now_ms_precise() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Waits both queues before the probe's mem_handles are released, on every path out of the block: a
// handle must outlive the work that uses its allocation, a failure path included. It is itself a bounded
// step, so a drain that hangs is named by the watchdog.
struct queue_drain_guard {
    sycl::queue & a;
    sycl::queue & b;

    queue_drain_guard(sycl::queue & qa, sycl::queue & qb) : a(qa), b(qb) {}

    ~queue_drain_guard() {
        step_scope scope("teardown_drain_queues", k_step_budget_ms);
        try {
            a.wait();
            b.wait();
        } catch (...) {
            std::printf("FAIL: the final queue drain threw\n");
        }
    }
};

// Pure device arithmetic with a loop-carried non-affine dependence (no closed form for the compiler to take),
// its result stored so the loop is live. One work item, `iters` rounds.
sycl::event submit_busy(sycl::queue & q, uint32_t * sink, uint64_t iters) {
    return q.submit([&](sycl::handler & h) {
        h.single_task([=]() {
            uint32_t x = 1;
            for (uint64_t i = 0; i < iters; ++i) {
                x = x * 1664525u + 1013904223u;
                x ^= x >> 13;
            }
            sink[0] = x;
        });
    });
}

double busy_ms_of(sycl::queue & q, uint32_t * sink, uint64_t iters) {
    const long long t0 = now_ms_precise();
    submit_busy(q, sink, iters).wait();
    return static_cast<double>(now_ms_precise() - t0) / 1000.0;
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

    // The watchdog starts before anything that can hang: backend init, queue creation, allocation and
    // teardown are named steps too, because this stack hangs in exactly that area.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::thread(step_watchdog_main).detach();

    // The scheduler exposes only device 0 by default, so ggml_backend_sycl_get_device_count() is 1 on a
    // two-card host; the physical count is what decides the skip. Zero devices (no setvars) lands here too.
    // ggml_backend_sycl_init(i) asserts on an index past the scheduler-visible count, so the exposed
    // device 0 is the only one initialised as a backend.
    int physical_devices = 0;
    int visible_devices  = 0;
    {
        step_scope scope("init_device_counts", k_step_budget_ms);
        physical_devices = ggml_sycl::test_physical_device_count();
        visible_devices  = ggml_backend_sycl_get_device_count();
    }
    if (physical_devices < 2 || visible_devices < 1) {
        std::printf("SKIP: fewer than two physical SYCL GPU devices available (physical=%d)\n", physical_devices);
        return LLAMA_TEST_EXIT_SKIP;
    }

    ggml_backend_t backend_owner = nullptr;
    {
        step_scope scope("init_backend_owner", k_step_budget_ms);
        backend_owner = ggml_backend_sycl_init(0);
    }
    if (!backend_owner) {
        std::printf("FAIL: ggml_backend_sycl_init(0) failed on a host that reports two physical devices\n");
        return 1;
    }
    // The queues are created lazily by the MoE and split paths, so a standalone process creates them the way
    // those paths do: init_shared_context_queues(total_gpu_count), one single-device context per card.
    sycl::queue * q_source_ptr = nullptr;
    {
        step_scope scope("init_shared_context_queues", k_step_budget_ms);
        ggml_sycl::init_shared_context_queues(physical_devices);
        q_source_ptr = ggml_sycl::get_shared_context_queue(1);
    }
    if (q_source_ptr == nullptr) {
        std::printf("FAIL: no shared-context queue for physical device 1\n");
        step_scope scope("teardown_free_backend", k_step_budget_ms);
        ggml_backend_free(backend_owner);
        return 1;
    }

    auto *        owner_ctx = static_cast<ggml_backend_sycl_context *>(backend_owner->context);
    sycl::queue & q_owner   = *owner_ctx->stream(0, 0);
    sycl::queue & q_source  = *q_source_ptr;

    int rc = 1;
    {
        // The handles live in this block, so every buffer is released before the backend is freed.
        probe_buffer src_dev;
        probe_buffer bounce;
        probe_buffer dst_dev;
        probe_buffer sink;  // the busy kernel's result word, on the source device
        bool         ok = false;
        {
            step_scope scope("alloc_probe_buffers", k_step_budget_ms);
            ok = alloc_probe_buffer(src_dev, q_source, 1, k_bytes, false) &&
                 alloc_probe_buffer(bounce, q_source, 1, k_bytes, true) &&
                 alloc_probe_buffer(dst_dev, q_owner, 0, k_bytes, false) &&
                 alloc_probe_buffer(sink, q_source, 1, 64, false);
        }
        // Declared after the handles, so it runs before any of them is released: every command the probe
        // queued on either queue (the busy kernel, leg 1, leg 2, on a failure path too) is finished first.
        queue_drain_guard drain(q_source, q_owner);
        if (!ok) {
            std::printf("FAIL: could not allocate the probe's buffers through unified_allocate\n");
        } else {
            void *     src_ptr    = src_dev.ptr;
            void *     bounce_ptr = bounce.ptr;
            void *     dst_ptr    = dst_dev.ptr;
            uint32_t * sink_ptr   = static_cast<uint32_t *>(sink.ptr);

            // Which context knows which pointer: a host-USM bounce allocated in one context and copied from
            // on a queue of the other is the first suspect for a leg that never completes.
            const auto kind = [&](void * ptr, sycl::queue & q) {
                return static_cast<int>(sycl::get_pointer_type(ptr, q.get_context()));
            };
            {
                step_scope scope("query_pointer_types", k_step_budget_ms);
                std::printf(
                    "[SYCL-STAGING-PROBE] contexts same=%d pointer_type(owner_ctx,source_ctx) "
                    "src_dev=(%d,%d) bounce=(%d,%d) dst_dev=(%d,%d) (0 host, 1 device, 2 shared, 3 unknown)\n",
                    q_owner.get_context() == q_source.get_context() ? 1 : 0, kind(src_ptr, q_owner),
                    kind(src_ptr, q_source), kind(bounce_ptr, q_owner), kind(bounce_ptr, q_source),
                    kind(dst_ptr, q_owner), kind(dst_ptr, q_source));
            }

            // The bytes the legs move. Leg 2's destination is read back and compared, so a copy that moved
            // stale or no bytes cannot pass on timing alone: src holds `pattern`, the bounce and the
            // destination start as distinct sentinels.
            std::vector<unsigned char> pattern(k_bytes);
            for (size_t b = 0; b < k_bytes; ++b) {
                pattern[b] = static_cast<unsigned char>((b * 131 + 7) & 0xFF);
            }
            const std::vector<unsigned char> sentinel_bounce(k_bytes, 0xAA);
            const std::vector<unsigned char> sentinel_dst(k_bytes, 0x55);
            const auto                       prime = [&]() {
                q_source.memcpy(src_ptr, pattern.data(), k_bytes).wait();
                std::memcpy(bounce_ptr, sentinel_bounce.data(), k_bytes);
                q_owner.memcpy(dst_ptr, sentinel_dst.data(), k_bytes).wait();
            };
            const auto bounce_holds_pattern = [&]() {
                return std::memcmp(bounce_ptr, pattern.data(), k_bytes) == 0;
            };
            const auto dst_holds_pattern = [&]() {
                std::vector<unsigned char> readback(k_bytes, 0);
                q_owner.memcpy(readback.data(), dst_ptr, k_bytes).wait();
                return std::memcmp(readback.data(), pattern.data(), k_bytes) == 0;
            };

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

            // Calibration: a first tiny run builds the kernel, then the iteration count is scaled twice
            // from measured run times so the busy kernel runs about k_target_busy_ms. The last calibration
            // round is a first estimate; control g's two further runs join it, and the verdict's reference
            // is the median of the three (one noisy run cannot move the thresholds).
            uint64_t   busy_iters = 1u << 22;
            double     busy_ms    = 0.0;
            const bool calibrated = run_step("calibrate_busy_kernel", [&]() {
                (void) busy_ms_of(q_source, sink_ptr, 1000);  // compile and first launch
                for (int round = 0; round < 3; ++round) {
                    busy_ms = busy_ms_of(q_source, sink_ptr, busy_iters);
                    std::printf("[SYCL-STAGING-PROBE] calibrate round=%d busy_iters=%llu busy_ms=%.3f\n", round,
                                static_cast<unsigned long long>(busy_iters), busy_ms);
                    if (busy_ms < 0.001) {
                        throw std::runtime_error("the busy kernel measured no run time");
                    }
                    if (round < 2) {
                        double scale = k_target_busy_ms / busy_ms;
                        scale        = scale < 1.0 / 4096.0 ? 1.0 / 4096.0 : (scale > 4096.0 ? 4096.0 : scale);
                        busy_iters   = static_cast<uint64_t>(static_cast<double>(busy_iters) * scale);
                        if (busy_iters < 1000) {
                            busy_iters = 1000;
                        }
                    }
                }
                if (busy_ms < 0.5 * k_target_busy_ms || busy_ms > 2.0 * k_target_busy_ms) {
                    throw std::runtime_error("the busy kernel did not calibrate near its target");
                }
            });
            controls_ok &= calibrated;
            if (!calibrated) {
                std::printf("FAIL: calibration failed, so no hold can be sized\n");
            } else {
                // Control g: the busy kernel alone, the host waiting on its event (the must-fire control).
                controls_ok &= run_step("control_g_busy_alone", [&]() {
                    const double calibrated_ms = busy_ms;
                    const double a             = busy_ms_of(q_source, sink_ptr, busy_iters);
                    const double b             = busy_ms_of(q_source, sink_ptr, busy_iters);
                    std::printf("[SYCL-STAGING-PROBE] control_g busy_ms=%.3f,%.3f calibrated_ms=%.3f\n", a, b,
                                calibrated_ms);
                    for (const double ms : { a, b }) {
                        if (ms > 2.0 * calibrated_ms || ms < 0.5 * calibrated_ms) {
                            throw std::runtime_error("the busy kernel's run time left [0.5x, 2x] of its calibration");
                        }
                    }
                    // The median of the calibration's last round and these two.
                    const double lo = std::min(calibrated_ms, std::min(a, b));
                    const double hi = std::max(calibrated_ms, std::max(a, b));
                    busy_ms         = calibrated_ms + a + b - lo - hi;
                    std::printf("[SYCL-STAGING-PROBE] busy_ms reference (median of three)=%.3f\n", busy_ms);
                });

                // Control i: leg 1 alone, no dependency, waited on its own event; the bounce must hold the
                // pattern afterwards.
                controls_ok &= run_step("control_i_leg1_alone", [&]() {
                    prime();
                    sycl::event e = q_source.submit([&](sycl::handler & h) { h.memcpy(bounce_ptr, src_ptr, k_bytes); });
                    e.wait();
                    if (!bounce_holds_pattern()) {
                        throw std::runtime_error("leg 1 bounce bytes wrong");
                    }
                });

                // Control ii: leg 2 alone, no dependency, from a bounce that holds the pattern.
                controls_ok &= run_step("control_ii_leg2_alone", [&]() {
                    prime();
                    std::memcpy(bounce_ptr, pattern.data(), k_bytes);
                    sycl::event e = q_owner.submit([&](sycl::handler & h) { h.memcpy(dst_ptr, bounce_ptr, k_bytes); });
                    e.wait();
                    if (!dst_holds_pattern()) {
                        throw std::runtime_error("leg 2 destination bytes wrong");
                    }
                });

                // Control iv: the dependency as a host wait, leg 1 done before leg 2 is submitted. A CONTROL
                // only: production takes no host wait. Run before iii because iii can hang. It is also the
                // positive control for the byte check iii makes.
                controls_ok &= run_step("control_iv_host_wait_between", [&]() {
                    prime();
                    sycl::event l1 =
                        q_source.submit([&](sycl::handler & h) { h.memcpy(bounce_ptr, src_ptr, k_bytes); });
                    l1.wait();
                    sycl::event l2 = q_owner.submit([&](sycl::handler & h) { h.memcpy(dst_ptr, bounce_ptr, k_bytes); });
                    l2.wait();
                    if (!dst_holds_pattern()) {
                        throw std::runtime_error("leg 2 destination bytes wrong");
                    }
                });

                // Experiment iii: leg 1 held behind the busy kernel, leg 2 depending on it across contexts.
                // Readings are host clocks from the busy kernel's submit.
                {
                    step_scope scope("iii_prime_bytes", k_step_budget_ms);
                    prime();
                }
                sycl::event busy_event;
                sycl::event leg1;
                sycl::event leg2;
                bool        setup_threw = false;
                const auto  t0          = std::chrono::steady_clock::now();
                const auto  since_t0    = [&]() {
                    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                };
                {
                    step_scope scope("iii_submit_busy_and_leg1", k_step_budget_ms);
                    try {
                        busy_event = submit_busy(q_source, sink_ptr, busy_iters);
                        leg1       = q_source.submit([&](sycl::handler & h) {
                            h.depends_on(busy_event);
                            h.memcpy(bounce_ptr, src_ptr, k_bytes);
                        });
                    } catch (const std::exception & ex) {
                        setup_threw = true;
                        std::printf("FAIL: the busy kernel / leg 1 submit threw: %s\n", ex.what());
                    }
                }
                if (!setup_threw) {
                    // A hang in this submit is the step watchdog's: "FAIL: HANG at iii_leg2_submit".
                    bool        submit_threw = false;
                    double      return_ms    = -1.0;
                    std::string error;
                    {
                        step_scope scope("iii_leg2_submit", k_step_budget_ms);
                        try {
                            leg2 = q_owner.submit([&](sycl::handler & h) {
                                h.depends_on(leg1);
                                h.memcpy(dst_ptr, bounce_ptr, k_bytes);
                            });
                        } catch (const std::exception & ex) {
                            submit_threw = true;
                            error        = ex.what();
                        } catch (...) {
                            submit_threw = true;
                            error        = "non-standard exception";
                        }
                        // Read before the scope's destructor prints its end line.
                        return_ms = since_t0();
                    }
                    double done_ms = -1.0;
                    bool   drained = !submit_threw;
                    bool   dst_ok  = false;
                    if (!submit_threw) {
                        try {
                            {
                                step_scope scope("iii_wait_leg2", k_step_budget_ms);
                                leg2.wait();
                                done_ms = since_t0();
                            }
                            {
                                step_scope scope("iii_wait_queues", k_step_budget_ms);
                                q_source.wait();
                                q_owner.wait();
                            }
                            {
                                step_scope scope("iii_verify_destination_bytes", k_step_budget_ms);
                                dst_ok = dst_holds_pattern();
                            }
                        } catch (const std::exception & ex) {
                            drained = false;
                            error   = ex.what();
                        } catch (...) {
                            drained = false;
                            error   = "non-standard exception";
                        }
                    }
                    if (submit_threw || !drained) {
                        std::printf("FAIL: the probe's %s threw: %s\n", submit_threw ? "leg 2 submit" : "drain",
                                    error.c_str());
                    } else {
                        const staging_probe_verdict v = staging_probe_classify(busy_ms, return_ms, done_ms);
                        std::printf(
                            "[SYCL-STAGING-PROBE] leg2_submit=%s busy_ms=%.3f return_ms=%.3f done_ms=%.3f "
                            "thresholds(returns_before<%.2f blocks>=%.2f void_done<%.2f of busy_ms) "
                            "controls_ok=%d dst_ok=%d\n",
                            staging_probe_verdict_name(v), busy_ms, return_ms, done_ms,
                            k_staging_probe_returns_before_frac, k_staging_probe_blocks_frac,
                            k_staging_probe_void_done_frac, controls_ok ? 1 : 0, dst_ok ? 1 : 0);
                        if (!dst_ok) {
                            std::printf("FAIL: leg 2 destination bytes wrong\n");
                        }
                        const bool pass =
                            (v == staging_probe_verdict::RETURNS_BEFORE || v == staging_probe_verdict::BLOCKS) &&
                            controls_ok && dst_ok;
                        rc = pass ? 0 : 1;
                    }
                }
            }
        }
    }

    {
        step_scope scope("teardown_free_backend", k_step_budget_ms);
        ggml_backend_free(backend_owner);
    }
    return rc;
}
