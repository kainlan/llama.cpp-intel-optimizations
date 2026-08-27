//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.hpp"  // sycl::queue

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ggml_sycl {

// TKV-13 (B2): a dedicated persistent thread pool for demoted-layer
// attention (docs/plans/2026-08-27-tkv13-b2-addendum.md §2, §6 step 2).
// Modeled on CpuExpertPool's shape -- persistent workers, a
// std::future-returning submit -- but its OWN, separate class: per owner
// ruling 2026-08-27 (llama.cpp-sbky comment thread), this campaign does NOT
// generalize CpuExpertPool to a second work kind. Two independent producers
// sharing one pending-result slot is a correctness hazard (a second
// submission before the first is flushed would silently drop the first
// result, per the addendum's blast-radius note, §6) -- a separate pool
// keeps the two overlap mechanisms from ever touching shared pending state.
//
// Unlike CpuExpertPool, whose task struct owns fixed act/output float
// buffers sized at init(), this pool's unit of work is a bare callable: the
// exact Q/K/V/mask/output layout for a demoted-layer attention dispatch
// depends on ggml-cpu's FLASH_ATTN_EXT compute entry point, which TKV-13
// step 3 resolves by reading ggml/src/ggml-cpu/ops.cpp -- not guessed here,
// per the addendum's own caveat (§2, "read ggml-cpu's FA loop... do not
// guess the stride shape here"). This pool owns only the thread lifecycle
// and the work queue; step 3 supplies the callable, and the pinned-host
// staging buffers it reads/writes (Q input, attention-output) are
// unified_allocate()'d with runtime_category::HOST_COMPUTE by that step's
// caller, not by this class.
class AttnHostPool {
  public:
    AttnHostPool() = default;
    ~AttnHostPool();

    AttnHostPool(const AttnHostPool &)             = delete;
    AttnHostPool & operator=(const AttnHostPool &) = delete;
    AttnHostPool(AttnHostPool &&)                  = delete;
    AttnHostPool & operator=(AttnHostPool &&)      = delete;

    // n_threads: worker thread count (0 = auto: hardware_concurrency - 2,
    // reserving cores for the GPU driver -- same default policy as
    // CpuExpertPool, deliberately reimplemented rather than shared code; see
    // the class comment for why the two pools stay separate).
    // `q` is accepted now (not yet used) so step 3 can add pinned-staging
    // allocation without changing this signature.
    void init(int n_threads, sycl::queue & q);
    void shutdown();

    // Submit one unit of host-side attention work. The task itself is
    // responsible for reading pinned KV directly (already host-resident,
    // zero copy) and for writing its result into a pinned staging buffer
    // supplied by its caller -- this pool does not interpret the callable's
    // contents, only runs it on a worker thread and reports completion (or
    // any exception it throws) via the returned future.
    std::future<void> submit(std::function<void()> task);

    bool is_active() const { return active_.load(std::memory_order_acquire); }

  private:
    void worker_thread();

    std::vector<std::thread>          threads_;
    std::queue<std::function<void()>> work_queue_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    std::atomic<bool>                 active_{ false };
    std::atomic<bool>                 shutting_down_{ false };
};

}  // namespace ggml_sycl
