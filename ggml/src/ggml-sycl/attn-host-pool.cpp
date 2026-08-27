//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "attn-host-pool.hpp"

#include "ggml-impl.h"
#include "unified-cache.hpp"  // ggml_sycl_is_shutting_down()

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <utility>

namespace ggml_sycl {

namespace {

// Env var: GGML_SYCL_ATTN_HOST_THREADS=N overrides thread count. Deliberately
// a separate variable from CpuExpertPool's GGML_SYCL_CPU_EXPERT_THREADS --
// the two pools are independently sized (owner ruling, see the header
// comment on why they are not one pool).
int get_attn_host_thread_count() {
    const char * env = std::getenv("GGML_SYCL_ATTN_HOST_THREADS");
    if (env) {
        int n = std::atoi(env);
        if (n > 0) {
            return n;
        }
    }
    // Default: hardware_concurrency - 2 (reserve for GPU driver threads).
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, hw - 2);
}

}  // namespace

void AttnHostPool::init(int n_threads, sycl::queue & q) {
    GGML_UNUSED(q);  // no owned allocation yet -- step 3 adds pinned staging.
    if (active_.load(std::memory_order_acquire)) {
        GGML_LOG_WARN("[ATTN-HOST-POOL] init() called on already-active pool, ignoring\n");
        return;
    }

    if (n_threads <= 0) {
        n_threads = get_attn_host_thread_count();
    }

    shutting_down_.store(false, std::memory_order_release);
    active_.store(true, std::memory_order_release);
    threads_.reserve(n_threads);
    for (int i = 0; i < n_threads; i++) {
        threads_.emplace_back(&AttnHostPool::worker_thread, this);
    }

    GGML_LOG_INFO("[ATTN-HOST-POOL] Initialized: %d threads\n", n_threads);
}

void AttnHostPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::seconds(2),
                         [this] { return shutting_down_.load(std::memory_order_acquire) || !work_queue_.empty(); });
            if (shutting_down_.load(std::memory_order_acquire) && work_queue_.empty()) {
                return;
            }
            // Spurious wakeup or timeout with empty queue -- loop back.
            if (work_queue_.empty()) {
                continue;
            }
            task = std::move(work_queue_.front());
            work_queue_.pop();
        }
        task();
    }
}

std::future<void> AttnHostPool::submit(std::function<void()> task) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future  = promise->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Propagate an exception through the future rather than letting it
        // escape on the worker thread (std::terminate) -- the caller's
        // consumption point (the DAG-scan-deferred flush, TKV-13 step 3)
        // observes it via future::get()/wait(), the same place it already
        // observes a normal result.
        work_queue_.push([task = std::move(task), promise]() mutable {
            try {
                task();
            } catch (...) {
                promise->set_exception(std::current_exception());
                return;
            }
            promise->set_value();
        });
    }
    cv_.notify_one();
    return future;
}

void AttnHostPool::shutdown() {
    if (!active_.load(std::memory_order_acquire)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutting_down_.store(true, std::memory_order_release);
    }
    cv_.notify_all();

    for (auto & t : threads_) {
        if (t.joinable()) {
            auto join_future = std::async(std::launch::async, [&t] { t.join(); });
            if (join_future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
                GGML_LOG_WARN("[ATTN-HOST-POOL] Worker thread did not exit within 5s\n");
                t.detach();
            }
        }
    }
    threads_.clear();
    active_.store(false, std::memory_order_release);

    GGML_LOG_INFO("[ATTN-HOST-POOL] Shut down\n");
}

AttnHostPool::~AttnHostPool() {
    // Skip cleanup during static destruction -- unified cache statics may
    // already be destroyed (mirrors CpuExpertPool's destructor guard).
    if (!ggml_sycl_is_shutting_down()) {
        shutdown();
    }
}

}  // namespace ggml_sycl
