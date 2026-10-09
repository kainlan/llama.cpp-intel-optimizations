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
#include <string>
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
    //   n_threads: worker thread count (0 = auto: hardware_concurrency - 2)
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

    // The same jobs split by the weight type of their rows (llama.cpp-y9i6),
    // indexed by type. by_type[slot_mixed] holds the jobs whose rows mix types,
    // and by_type[slot_none] the jobs that ran no row loop: the batched kernel
    // returned before it, or every task took the MXFP4 multi-activation path.
    //   threads_max  most threads one job of the type ran on
    //   overlapped   jobs that ran at the same time as another pool job at some
    //                point (both jobs of an overlapping pair count): the pools
    //                share one CPU arena, so those jobs' compute times overlap
    //                and their sum overstates the time they took
    struct type_totals {
        uint64_t jobs        = 0;
        uint64_t rows        = 0;
        uint64_t bytes       = 0;
        uint64_t threads     = 0;
        uint64_t threads_max = 0;
        uint64_t overlapped  = 0;
        double   compute_us  = 0.0;
    };

    static constexpr int slot_mixed = GGML_TYPE_COUNT;
    static constexpr int slot_none  = GGML_TYPE_COUNT + 1;
    static constexpr int n_slots    = GGML_TYPE_COUNT + 2;

    type_totals by_type[n_slots];
};

void cpu_expert_pool_trace_take(cpu_expert_pool_trace_totals & out);
void cpu_expert_pool_trace_note_join(bool was_ready);

// Whether a pool job ran at the same time as another at any point of its run
// (llama.cpp-y9i6): another job was running when it began, or another began
// before it ended. Both jobs of an overlapping pair see it. Used only while
// tracing.
struct cpu_expert_pool_overlap_clock {
    std::atomic<int>      running{ 0 };
    std::atomic<uint64_t> starts{ 0 };
};

struct cpu_expert_pool_overlap_ticket {
    bool     running_at_start = false;
    uint64_t start            = 0;
};

cpu_expert_pool_overlap_ticket cpu_expert_pool_overlap_begin(cpu_expert_pool_overlap_clock & clock);
bool cpu_expert_pool_overlap_end(cpu_expert_pool_overlap_clock & clock, const cpu_expert_pool_overlap_ticket & ticket);

// Adds one finished job to `totals`, in the job's type slot as well (the
// no-row slot when ph.rows is 0, whatever ph.type says).
void cpu_expert_pool_trace_add_job(cpu_expert_pool_trace_totals &         totals,
                                   size_t                                 n_tasks,
                                   const cpu_expert_batched_phase_times & ph,
                                   double                                 wake_us,
                                   double                                 wall_us,
                                   bool                                   overlapped);

// The per-type summary: for each type with a job, in enum order, then "mixed"
// and "none" for the two extra slots,
// " <type>:jobs=J,rows=R,bytes=B,compute=Cus,gbps=G,thr=A/M,ovl=O",
// where G is bytes over the summed compute time (decimal GB/s), A the mean and
// M the most threads per job, and O the jobs that overlapped another pool job.
// Empty when there were no jobs.
std::string cpu_expert_pool_trace_format_types(const cpu_expert_pool_trace_totals & totals);

}  // namespace ggml_sycl
