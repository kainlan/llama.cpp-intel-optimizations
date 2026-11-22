// Pins the runtime facts the asynchronous-failure latch rests on, so a runtime upgrade that changes them fails here:
// (a) queue::wait_and_throw delivers to the handler of the queue the command ran on, not its context's, and returns;
// (b) event::wait_and_throw on the failed event also goes to the queue's handler, once;
// (c) a queue on the platform default context with a handler on the queue alone receives both;
// (d) a host task that depends on a failed event still runs, and sees the failure in the latch when its own wait is the
//     one that delivers it (fact (i) shows when it is not);
// (e) a queue built without a handler on a context that has one delivers to the context's handler;
// (i) while one thread's wait_and_throw is delivering a queue's errors (its handler has not returned), another thread's
//     wait_and_throw on an event of the same queue returns at once and delivers nothing: the first wait took the
//     pending errors before it called the handler. A latch read after a wait therefore says nothing about errors that
//     another thread's wait is delivering; only a boundary that drained on its own thread may read it. Each error is
//     still delivered exactly once.
// Host tasks only: no kernel, no allocation. Without --gpu it needs a CPU SYCL device and skips when there is none.
#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP
#include "../async-failure.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <thread>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace {
// Distinct latch slots tell the queue's handler from the context's. The probe runs alone in its process, so device
// slots are free to borrow.
constexpr int k_queue_slot       = 0;
constexpr int k_context_slot     = 1;
constexpr int k_default_ctx_slot = 2;

sycl::event submit_throw(sycl::queue & q, const char * what) {
    return q.submit([what](sycl::handler & h) { h.host_task([what] { throw std::runtime_error(what); }); });
}

bool waits_without_throwing(sycl::queue & q) {
    try {
        q.wait_and_throw();
        return true;
    } catch (...) {
        return false;
    }
}

bool waits_without_throwing(sycl::event & e) {
    try {
        e.wait_and_throw();
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t count(int slot) {
    return ggml_sycl_async_failure_count(slot);
}

// Polls `done` for up to `seconds`; true when it held.
template <typename F> bool wait_for(F done, int seconds) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!done()) {
        if (std::chrono::steady_clock::now() >= until) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Fact (i). A test-local handler that blocks until released, so one thread's delivery stays in flight while a second
// thread waits on an event of the same queue. The handler records nothing in the latch: it counts its calls and the
// errors it was handed.
int probe_delivery_in_flight(const sycl::device & dev, const sycl::property_list & in_order) {
    std::atomic<int>  handler_calls{ 0 };
    std::atomic<int>  delivered{ 0 };
    std::atomic<bool> release{ false };
    sycl::queue       q(
        dev,
        [&](sycl::exception_list errors) {
            handler_calls.fetch_add(1, std::memory_order_acq_rel);
            while (!release.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            delivered.fetch_add(static_cast<int>(errors.size()), std::memory_order_acq_rel);
        },
        in_order);
    sycl::event bad = submit_throw(q, "probe (i)");

    // Thread A delivers, and its handler blocks. It repeats the wait until it has delivered, in case the error reaches
    // the queue's list only after the event completes.
    std::atomic<bool> a_done{ false };
    std::thread       a([&] {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do {
            (void) waits_without_throwing(q);
        } while (handler_calls.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < until);
        a_done.store(true, std::memory_order_release);
    });

    // Releases the handler and joins thread A, on every exit from here on.
    const auto finish = [&] {
        release.store(true, std::memory_order_release);
        a.join();
    };
    if (!wait_for([&] { return handler_calls.load(std::memory_order_acquire) != 0; }, 10)) {
        finish();
        std::fprintf(stderr, "FAIL: (i) the first wait never called the handler (a_done=%d)\n", a_done.load() ? 1 : 0);
        return 1;
    }

    // Thread B waits on the failed event while A's delivery is in flight.
    std::atomic<bool> b_done{ false };
    std::atomic<bool> b_threw{ false };
    std::thread       b([&] {
        b_threw.store(!waits_without_throwing(bad), std::memory_order_release);
        b_done.store(true, std::memory_order_release);
    });
    if (!wait_for([&] { return b_done.load(std::memory_order_acquire); }, 5)) {
        finish();
        b.join();
        std::fprintf(stderr,
                     "FAIL: (i) the second thread's wait blocked behind the delivery in flight: the runtime now holds "
                     "the queue's error list while it calls the handler\n");
        return 1;
    }
    b.join();
    const int  calls_during     = handler_calls.load(std::memory_order_acquire);
    const int  delivered_during = delivered.load(std::memory_order_acquire);
    const bool a_done_during    = a_done.load(std::memory_order_acquire);
    finish();
    CHECK(!b_threw.load(), "(i) the second thread's wait threw");
    CHECK(calls_during == 1 && delivered_during == 0 && !a_done_during,
          "(i) the second thread's wait delivered, or the first delivery was no longer in flight");
    CHECK(handler_calls.load() == 1 && delivered.load() == 1, "(i) the error was not delivered exactly once");
    CHECK(waits_without_throwing(q) && handler_calls.load() == 1, "(i) a later wait delivered again");
    return 0;
}
}  // namespace

int main(int argc, char ** argv) {
    const bool                  gpu = argc > 1 && std::strcmp(argv[1], "--gpu") == 0;
    std::optional<sycl::device> dev;
    try {
        dev = gpu ? sycl::device{ sycl::gpu_selector_v } : sycl::device{ sycl::cpu_selector_v };
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no %s SYCL device: %s\n", gpu ? "GPU" : "CPU", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    const sycl::property_list in_order{ sycl::property::queue::in_order{} };
    sycl::context             ctx = ggml_sycl_make_context(*dev, k_context_slot);
    sycl::queue               q   = ggml_sycl_make_queue(ctx, *dev, k_queue_slot, in_order);

    // (a)
    uint64_t nq = count(k_queue_slot), nc = count(k_context_slot);
    submit_throw(q, "probe (a)");
    CHECK(waits_without_throwing(q), "(a) queue::wait_and_throw threw: the handler must record, not rethrow");
    CHECK(count(k_queue_slot) == nq + 1 && count(k_context_slot) == nc,
          "(a) queue::wait_and_throw did not deliver to the queue's handler alone");

    // (b)
    nq              = count(k_queue_slot);
    nc              = count(k_context_slot);
    sycl::event bad = submit_throw(q, "probe (b)");
    CHECK(waits_without_throwing(bad), "(b) event::wait_and_throw threw");
    CHECK(count(k_queue_slot) == nq + 1 && count(k_context_slot) == nc,
          "(b) event::wait_and_throw did not deliver to the queue's handler alone");
    CHECK(waits_without_throwing(q) && count(k_queue_slot) == nq + 1, "(b) the error was delivered twice");

    // (c)
    sycl::queue qd = ggml_sycl_make_queue(*dev, k_default_ctx_slot, in_order);
    uint64_t    nd = count(k_default_ctx_slot);
    sycl::event bd = submit_throw(qd, "probe (c) event wait");
    CHECK(waits_without_throwing(bd) && count(k_default_ctx_slot) == nd + 1,
          "(c) event::wait_and_throw on a default-context queue missed the queue's handler");
    submit_throw(qd, "probe (c) queue wait");
    CHECK(waits_without_throwing(qd) && count(k_default_ctx_slot) == nd + 2,
          "(c) queue::wait_and_throw on a default-context queue missed the queue's handler");

    // (d)
    nq                      = count(k_queue_slot);
    sycl::event      failed = submit_throw(q, "probe (d)");
    std::atomic<int> saw{ -1 };  // -1 never ran, 0 ran but latch unmoved, 1 saw the failure, 2 the wait threw
    q.submit([&saw, failed, nq](sycl::handler & h) {
        h.depends_on(failed);
        h.host_task([&saw, failed, nq]() mutable {
            bool threw = false;
            try {
                failed.wait_and_throw();
            } catch (...) {
                threw = true;
            }
            saw.store(threw ? 2 : (count(k_queue_slot) != nq ? 1 : 0), std::memory_order_release);
        });
    });
    q.wait();  // a plain wait delivers nothing (fact g); the host task above did the delivery
    CHECK(saw.load(std::memory_order_acquire) == 1,
          "(d) a host task depending on a failed event did not run, threw, or did not see the failure in the latch");
    CHECK(waits_without_throwing(q), "(d) final drain threw");

    // (e)
    sycl::queue qe(ctx, *dev, in_order);  // no handler of its own: the context's is the fallback
    nq = count(k_queue_slot);
    nc = count(k_context_slot);
    submit_throw(qe, "probe (e) queue wait");
    CHECK(waits_without_throwing(qe) && count(k_context_slot) == nc + 1 && count(k_queue_slot) == nq,
          "(e) queue::wait_and_throw on a handlerless queue did not reach its context's handler");
    sycl::event be = submit_throw(qe, "probe (e) event wait");
    CHECK(waits_without_throwing(be) && count(k_context_slot) == nc + 2,
          "(e) event::wait_and_throw on a handlerless queue did not reach its context's handler");

    // (i)
    if (const int rc = probe_delivery_in_flight(*dev, in_order); rc != 0) {
        return rc;
    }

    std::puts("async handler delivery: PASS");
    return 0;
}
