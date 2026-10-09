//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-expert-pool.hpp"

#include "ggml-impl.h"
#include "unified-cache.hpp"  // ggml_sycl_is_shutting_down()

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <utility>

namespace ggml_sycl {

// ---------------------------------------------------------------------------
// Env var: GGML_SYCL_CPU_EXPERT_THREADS=N overrides thread count.
// ---------------------------------------------------------------------------
static int get_cpu_expert_thread_count() {
    const char * env = std::getenv("GGML_SYCL_CPU_EXPERT_THREADS");
    if (env) {
        int n = std::atoi(env);
        if (n > 0) {
            return n;
        }
    }
    // Default: hardware_concurrency - 2 (reserve for GPU driver threads)
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, hw - 2);
}

// ---------------------------------------------------------------------------
// CpuExpertPool implementation
// ---------------------------------------------------------------------------

void CpuExpertPool::init(int n_threads) {
    if (active_.load(std::memory_order_acquire)) {
        GGML_LOG_WARN("[CPU-EXPERT-POOL] init() called on already-active pool, ignoring\n");
        return;
    }

    if (n_threads <= 0) {
        n_threads = get_cpu_expert_thread_count();
    }

    // Spawn worker threads
    shutting_down_.store(false, std::memory_order_release);
    active_.store(true, std::memory_order_release);
    threads_.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) {
        threads_.emplace_back(&CpuExpertPool::worker_thread, this);
    }

    GGML_LOG_INFO("[CPU-EXPERT-POOL] Initialized: %d threads\n", n_threads);
}

void CpuExpertPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::seconds(2),
                         [this] { return shutting_down_.load(std::memory_order_acquire) || !work_queue_.empty(); });
            if (shutting_down_.load(std::memory_order_acquire) && work_queue_.empty()) {
                return;
            }
            // Spurious wakeup or timeout with empty queue -- loop back
            if (work_queue_.empty()) {
                continue;
            }
            task = std::move(work_queue_.front());
            work_queue_.pop();
        }
        task();
    }
}

// ---------------------------------------------------------------------------
// Job timing (llama.cpp-b2jc): see cpu_expert_pool_trace_totals.
// ---------------------------------------------------------------------------
using pool_trace_clock = std::chrono::steady_clock;

static std::mutex                   g_pool_trace_mutex;
static cpu_expert_pool_trace_totals g_pool_trace;

static double pool_trace_us(pool_trace_clock::time_point a, pool_trace_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

void cpu_expert_pool_trace_take(cpu_expert_pool_trace_totals & out) {
    std::lock_guard<std::mutex> lock(g_pool_trace_mutex);
    out          = g_pool_trace;
    g_pool_trace = {};
}

void cpu_expert_pool_trace_note_join(bool was_ready) {
    std::lock_guard<std::mutex> lock(g_pool_trace_mutex);
    g_pool_trace.joins += 1;
    g_pool_trace.joins_ready += was_ready ? 1 : 0;
}

static void pool_trace_record_job(size_t                       n_tasks,
                                  pool_trace_clock::time_point t_submit,
                                  pool_trace_clock::time_point t_start,
                                  pool_trace_clock::time_point t_done) {
    const cpu_expert_batched_phase_times ph = ggml_sycl_cpu_expert_batched_last_phase_times();
    std::lock_guard<std::mutex>          lock(g_pool_trace_mutex);
    g_pool_trace.jobs += 1;
    g_pool_trace.tasks += n_tasks;
    g_pool_trace.rows += static_cast<uint64_t>(ph.rows);
    g_pool_trace.threads += static_cast<uint64_t>(ph.threads);
    g_pool_trace.wake_us += pool_trace_us(t_submit, t_start);
    g_pool_trace.quant_us += ph.quant_us;
    g_pool_trace.setup_us += ph.setup_us;
    g_pool_trace.fanout_us += ph.fanout_us;
    g_pool_trace.compute_us += ph.compute_us;
    g_pool_trace.wall_us += pool_trace_us(t_submit, t_done);
}

std::future<void> CpuExpertPool::submit_batch(std::vector<cpu_expert_task> tasks) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future  = promise->get_future();

    const bool                         trace    = ggml_sycl_cpu_expert_trace_enabled();
    const pool_trace_clock::time_point t_submit = trace ? pool_trace_clock::now() : pool_trace_clock::time_point{};

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Move the tasks vector into the lambda so that storage is owned
        // by the worker for the entire computation.  Prior to this fix,
        // a raw pointer was captured and the caller remained responsible
        // for keeping the backing vector alive — a contract that was
        // violated by several call sites, producing a UAF in the TBB
        // arena (simd_mxfp4_q8_0_16row reading a stale weight_host).
        work_queue_.push([tasks = std::move(tasks), promise, trace, t_submit]() mutable {
            const pool_trace_clock::time_point t_start = trace ? pool_trace_clock::now() : t_submit;
            ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size()));
            if (trace) {
                pool_trace_record_job(tasks.size(), t_submit, t_start, pool_trace_clock::now());
            }
            promise->set_value();
        });
    }
    cv_.notify_one();
    return future;
}

void CpuExpertPool::stop_workers() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutting_down_.store(true, std::memory_order_release);
    }
    cv_.notify_all();

    for (auto & t : threads_) {
        if (t.joinable()) {
            auto future = std::async(std::launch::async, [&t] { t.join(); });
            if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
                GGML_LOG_WARN("[CPU-EXPERT-POOL] Worker thread did not exit within 5s\n");
                t.detach();
            }
        }
    }
    threads_.clear();
}

void CpuExpertPool::shutdown() {
    if (!active_.load(std::memory_order_acquire)) {
        return;
    }

    stop_workers();
    active_.store(false, std::memory_order_release);

    GGML_LOG_INFO("[CPU-EXPERT-POOL] Shut down\n");
}

CpuExpertPool::~CpuExpertPool() {
    // During static destruction run only stop_workers(): shutdown() also logs
    // through the installed log callback, whose owner may already be destroyed.
    if (!ggml_sycl_is_shutting_down()) {
        shutdown();
        return;
    }
    // The workers still have to stop, and that touches no cache state. Leaving
    // them parked in cv_.wait_for() makes the member destructors below hang the
    // process: glibc's pthread_cond_destroy() waits for every waiter, and a
    // joinable std::thread would std::terminate() after it. Reached whenever a
    // run exits without freeing its backend -- e.g. a failed decode
    // (llama.cpp-ze5y: exit hung in ~CpuExpertPool until `timeout` killed it).
    stop_workers();
}

}  // namespace ggml_sycl
