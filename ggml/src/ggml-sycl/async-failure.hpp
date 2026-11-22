#pragma once

#include "ggml-sycl.h"  // GGML_SYCL_MAX_DEVICES

#include <cstdint>
#include <sycl/sycl.hpp>

// Asynchronous SYCL errors (a host task that threw, a device fault reported after submission) reach the backend only
// through an async_handler, and only when a wait_and_throw() or throw_asynchronous() runs on the queue the command ran
// on. On this runtime both queue::wait_and_throw and event::wait_and_throw deliver to that queue's handler, a queue
// built without one falls back to its context's handler, and with neither the runtime's default handler prints and
// calls std::terminate. Each error is delivered once.
//
// A plain wait() -- on a queue or on an event -- neither throws nor delivers, so it is never a failure check. The error
// stays pending on its queue until a wait_and_throw() there, and a queue destroyed with an undelivered error drops it
// silently. So the last drain of a queue before it is destroyed is a wait_and_throw().
//
// A failed command does not cancel the commands that depend on it: they run on inputs that were never produced. That is
// why a backend context whose device latched a failure stays failed for the rest of its life (its outputs and the KV
// rows it wrote may come from that work), and why only a context created afterwards starts clean.
//
// The handler records each error in a per-device latch and does nothing else. The graph boundaries read the latch:
// ggml_backend_sycl_synchronize reports and cleans up, ggml_backend_sycl_graph_compute refuses.

// The slot of queues that serve no single backend device (dpct's per-device queues, the CPU-offload queue, the
// multi-GPU control context). A failure there fails every device.
constexpr int GGML_SYCL_ASYNC_FAILURE_SLOT_SHARED = GGML_SYCL_MAX_DEVICES;

// What a reader saw: a failure happened after the snapshot when either count has moved since.
struct ggml_sycl_async_failure_cursor {
    int      device       = -1;
    uint64_t device_count = 0;
    uint64_t shared_count = 0;
};

// Records one delivered error. Lock-free and allocation-free: it runs on whatever thread delivers the error (a decode
// thread, a SYCL host-task worker, a thread holding backend mutexes) and during static destruction.
void                           ggml_sycl_async_failure_record(int slot, const char * what) noexcept;
uint64_t                       ggml_sycl_async_failure_count(int slot) noexcept;
// The first message recorded on the slot since process start, or nullptr when none is readable yet.
const char *                   ggml_sycl_async_failure_first_message(int slot) noexcept;
ggml_sycl_async_failure_cursor ggml_sycl_async_failure_snapshot(int device) noexcept;
bool                           ggml_sycl_async_failure_since(const ggml_sycl_async_failure_cursor & cursor) noexcept;

// The handler bound to one slot, and the constructors the backend uses for its queues and contexts.
sycl::async_handler ggml_sycl_async_handler(int slot);
sycl::context       ggml_sycl_make_context(const sycl::device & dev, int slot);
sycl::queue         ggml_sycl_make_queue(const sycl::context &       ctx,
                                         const sycl::device &        dev,
                                         int                         slot,
                                         const sycl::property_list & props);
sycl::queue         ggml_sycl_make_queue(const sycl::device & dev, int slot, const sycl::property_list & props);
