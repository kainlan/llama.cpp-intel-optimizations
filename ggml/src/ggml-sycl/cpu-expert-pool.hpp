//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "cpu-dispatch.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ggml_sycl {

// Persistent thread pool for CPU expert computation.
// Replaces per-call std::async with pre-spawned workers.
//
// Thread count default: hardware_concurrency - 2 (reserve for GPU driver).
// Override: GGML_SYCL_CPU_EXPERT_THREADS=N
//
// The pool owns threads and a work queue, no memory. CPU expert staging comes
// from the PinnedBufferPool, with a per-dispatch managed fallback
// (llama.cpp-sfal).
class CpuExpertPool {
  public:
    CpuExpertPool() = default;
    ~CpuExpertPool();

    CpuExpertPool(const CpuExpertPool &)             = delete;
    CpuExpertPool & operator=(const CpuExpertPool &) = delete;
    CpuExpertPool(CpuExpertPool &&)                  = delete;
    CpuExpertPool & operator=(CpuExpertPool &&)      = delete;

    // Initialize the pool. Must be called once (e.g. from moe_hybrid_init_once).
    //   n_threads:    worker thread count (0 = auto: hardware_concurrency - 2)
    void init(int n_threads);

    // Shut down all workers.
    void shutdown();

    // Submit a batch of CPU expert tasks. Takes ownership of the tasks
    // vector; storage is kept alive inside the worker lambda for the
    // entire lifetime of the computation and is freed when the future
    // completes. This closes a UAF where a raw pointer into caller-owned
    // storage was retained by the worker lambda while the caller
    // overwrote / moved / destroyed the backing vector.
    std::future<void> submit_batch(std::vector<cpu_expert_task> tasks);

    bool is_active() const { return active_.load(std::memory_order_acquire); }

  private:
    void worker_thread();
    // Signal and join the workers. Touches no unified-cache state, so it is
    // safe during static destruction; shutdown() does this and then marks the
    // pool inactive.
    void stop_workers();

    std::vector<std::thread>          threads_;
    std::queue<std::function<void()>> work_queue_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    std::atomic<bool>                 active_{ false };
    std::atomic<bool>                 shutting_down_{ false };
};

// Totals over every pool's jobs since the last take (llama.cpp-b2jc), kept
// only while ggml_sycl_cpu_expert_trace_enabled(). The decode wait census
// takes them once per token. Times are sums over jobs:
//   wake     submit to a worker starting the job
//   quant/setup/fanout/compute  the job's batched-kernel phases
//   wall     submit to the job's result being published
// joins counts submitting-thread joins of a pool job; joins_ready the ones
// that found the job already finished.
struct cpu_expert_pool_trace_totals {
    uint64_t jobs        = 0;
    uint64_t tasks       = 0;
    uint64_t rows        = 0;
    uint64_t threads     = 0;
    uint64_t joins       = 0;
    uint64_t joins_ready = 0;
    double   wake_us     = 0.0;
    double   quant_us    = 0.0;
    double   setup_us    = 0.0;
    double   fanout_us   = 0.0;
    double   compute_us  = 0.0;
    double   wall_us     = 0.0;
};

void cpu_expert_pool_trace_take(cpu_expert_pool_trace_totals & out);
void cpu_expert_pool_trace_note_join(bool was_ready);

}  // namespace ggml_sycl
