//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-expert-pool.hpp"

#include "ggml-impl.h"
#include "unified-cache.hpp"  // ggml_sycl_is_shutting_down()

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
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
    // Default: one worker.  Each job's row loop runs on the pool's ONE row-worker set
    // (row_workers_, ggml_sycl_cpu_row_loop_threads() threads), which takes one job at a time,
    // so a second worker does not add threads; it queues behind the running job.  When the
    // measurement below was taken the row loop was a parallel_for into the backend's TBB arena,
    // where a second worker split the arena between overlapping jobs.  The jobs gain nothing
    // from running together: the layer chain already orders them (gate/up, then down, then
    // the next layer).  Measured 2026-10-15 on Qwen3.8 decode (B70, -c 4096, -t 4, host
    // load 25-32): one worker was never slower than 22, and was +14% to +29% faster
    // depending on host load (22 vs 1 worker, two windows: 8.49 vs 9.69 t/s over the valid
    // arms of four alternated pairs, 8.1 vs 10.5 t/s over two reps).  The pool finishes
    // sooner (job wall per token 54.1 -> 39.1 ms), but much of that time overlaps device
    // work, so the end-to-end gain is smaller than the pool's saving and varies with load.
    // One worker cannot deadlock because every wait points back in submission order (the
    // z4kd block in ggml-sycl.cpp).  That argument assumes ONE context per device's pool:
    // two contexts on one device would interleave their jobs in this FIFO behind each
    // other's joins, and same-device concurrent inference is unsupported (canonical
    // memory contract, section 5).
    return 1;
}

// ---------------------------------------------------------------------------
// Env var: GGML_SYCL_CPU_ROW_SPIN_US=N sets how long the row workers keep polling after a job
// before they block, in microseconds (0 = block at once; start() clamps to spin_us_max).
// Unset, empty, or not a whole number >= 0: CpuRowWorkers::spin_us_default.
// ---------------------------------------------------------------------------
static int get_cpu_row_spin_us() {
    const char * env = std::getenv("GGML_SYCL_CPU_ROW_SPIN_US");
    if (env == nullptr || *env == '\0') {
        return CpuRowWorkers::spin_us_default;
    }
    char *     end = nullptr;
    const long us  = std::strtol(env, &end, 10);
    if (*end != '\0' || us < 0) {
        GGML_LOG_WARN("[CPU-EXPERT-POOL] GGML_SYCL_CPU_ROW_SPIN_US=%s is not a whole number >= 0, using %d\n", env,
                      CpuRowWorkers::spin_us_default);
        return CpuRowWorkers::spin_us_default;
    }
    return static_cast<int>(std::min<long>(us, CpuRowWorkers::spin_us_max));
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
    row_workers_.start(ggml_sycl_cpu_row_loop_threads(), get_cpu_row_spin_us());

    GGML_LOG_INFO("[CPU-EXPERT-POOL] Initialized: %d threads, row loop on %d threads, spin %d us\n", n_threads,
                  row_workers_.n_threads(), row_workers_.spin_us());
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

void cpu_expert_pool_trace_add_job(cpu_expert_pool_trace_totals &         totals,
                                   size_t                                 n_tasks,
                                   const cpu_expert_batched_phase_times & ph,
                                   double                                 wake_us,
                                   double                                 wall_us,
                                   bool                                   overlapped) {
    totals.jobs += 1;
    totals.tasks += n_tasks;
    totals.rows += static_cast<uint64_t>(ph.rows);
    totals.threads += static_cast<uint64_t>(ph.threads);
    totals.wake_us += wake_us;
    totals.quant_us += ph.quant_us;
    totals.setup_us += ph.setup_us;
    totals.fanout_us += ph.fanout_us;
    totals.compute_us += ph.compute_us;
    totals.wall_us += wall_us;

    int slot = ph.type >= 0 && ph.type < GGML_TYPE_COUNT ? static_cast<int>(ph.type) :
                                                           cpu_expert_pool_trace_totals::slot_mixed;
    if (ph.rows == 0) {
        slot = cpu_expert_pool_trace_totals::slot_none;
    }
    cpu_expert_pool_trace_totals::type_totals & t = totals.by_type[slot];
    t.jobs += 1;
    t.rows += static_cast<uint64_t>(ph.rows);
    t.bytes += ph.bytes;
    t.threads += static_cast<uint64_t>(ph.threads);
    t.threads_max = std::max<uint64_t>(t.threads_max, static_cast<uint64_t>(ph.threads));
    t.overlapped += overlapped ? 1 : 0;
    t.compute_us += ph.compute_us;
}

std::string cpu_expert_pool_trace_format_types(const cpu_expert_pool_trace_totals & totals) {
    std::string out;
    char        buf[256];
    for (int i = 0; i < cpu_expert_pool_trace_totals::n_slots; i++) {
        const cpu_expert_pool_trace_totals::type_totals & t = totals.by_type[i];
        if (t.jobs == 0) {
            continue;
        }
        const char * name = i == cpu_expert_pool_trace_totals::slot_mixed ? "mixed" :
                            i == cpu_expert_pool_trace_totals::slot_none  ? "none" :
                                                                            ggml_type_name(static_cast<ggml_type>(i));
        const double gbps = t.compute_us > 0.0 ? static_cast<double>(t.bytes) / (t.compute_us * 1e3) : 0.0;
        snprintf(buf, sizeof(buf), " %s:jobs=%llu,rows=%llu,bytes=%llu,compute=%.0fus,gbps=%.2f,thr=%.1f/%llu,ovl=%llu",
                 name, (unsigned long long) t.jobs, (unsigned long long) t.rows, (unsigned long long) t.bytes,
                 t.compute_us, gbps, static_cast<double>(t.threads) / static_cast<double>(t.jobs),
                 (unsigned long long) t.threads_max, (unsigned long long) t.overlapped);
        out += buf;
    }
    return out;
}

cpu_expert_pool_overlap_ticket cpu_expert_pool_overlap_begin(cpu_expert_pool_overlap_clock & clock) {
    cpu_expert_pool_overlap_ticket ticket;
    ticket.running_at_start = clock.running.fetch_add(1) > 0;
    ticket.start            = clock.starts.fetch_add(1);
    return ticket;
}

bool cpu_expert_pool_overlap_end(cpu_expert_pool_overlap_clock & clock, const cpu_expert_pool_overlap_ticket & ticket) {
    // Read the starts before leaving, so a job that begins after this one ended
    // is not counted against it.
    const bool started_during = clock.starts.load() > ticket.start + 1;
    clock.running.fetch_sub(1);
    return ticket.running_at_start || started_during;
}

// Pool jobs over every pool (they share one CPU arena).
static cpu_expert_pool_overlap_clock g_pool_trace_overlap;

static void pool_trace_record_job(size_t                       n_tasks,
                                  pool_trace_clock::time_point t_submit,
                                  pool_trace_clock::time_point t_start,
                                  pool_trace_clock::time_point t_done,
                                  bool                         overlapped) {
    const cpu_expert_batched_phase_times ph = ggml_sycl_cpu_expert_batched_last_phase_times();
    std::lock_guard<std::mutex>          lock(g_pool_trace_mutex);
    cpu_expert_pool_trace_add_job(g_pool_trace, n_tasks, ph, pool_trace_us(t_submit, t_start),
                                  pool_trace_us(t_submit, t_done), overlapped);
}

std::future<void> CpuExpertPool::submit_batch(std::vector<cpu_expert_task>                        tasks,
                                              std::function<void()>                               prologue,
                                              std::function<void(std::vector<cpu_expert_task> &)> resolve) {
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
        // The prologue runs first, on this worker: the waits the job needs before it may read
        // its inputs and write its output (llama.cpp-z4kd). The job is timed from the worker
        // picking it up, so `wall` includes those waits and `wake` does not; the overlap clock
        // covers the kernel only. resolve, after the prologue, builds the tasks of a job
        // submitted before they were known (llama.cpp-7pvo).
        work_queue_.push([this, tasks = std::move(tasks), prologue = std::move(prologue), resolve = std::move(resolve),
                          promise, trace, t_submit]() mutable {
            const pool_trace_clock::time_point t_start = trace ? pool_trace_clock::now() : t_submit;
            if (prologue) {
                prologue();
            }
            if (resolve) {
                resolve(tasks);
            }
            cpu_expert_pool_overlap_ticket ticket;
            if (trace) {
                ticket = cpu_expert_pool_overlap_begin(g_pool_trace_overlap);
            }
            ggml_sycl_cpu_expert_mul_mat_batched(tasks.data(), static_cast<int>(tasks.size()), row_workers_);
            if (trace) {
                const bool overlapped = cpu_expert_pool_overlap_end(g_pool_trace_overlap, ticket);
                pool_trace_record_job(tasks.size(), t_submit, t_start, pool_trace_clock::now(), overlapped);
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
    // After the workers: a job still running uses the row workers until it returns.
    //
    // Destruction order.  ggml_backend_sycl_free() reaches shutdown() through
    // ggml_sycl_shutdown_global_runtime_pinned_owners(), so in a normal exit the row workers stop
    // here while the backend's TBB arena (ggml_sycl_cpu_arena(), cpu-dispatch.cpp) is still alive.
    // When a process exits without freeing its backend, ~CpuExpertPool runs during static
    // destruction, and the arena, if anything used it, is already gone: it is a function-local
    // static first used after g_cpu_expert_pools (ggml-sycl.cpp, namespace scope) was constructed,
    // so it is destroyed first.  Stopping the row workers then is still safe because they never enter the arena (a
    // job's Phase 1.5 GEMM, MXFP4 with several activation rows, does, on the pool worker's own
    // thread), and stop() waits for a run in flight before it joins them.
    row_workers_.stop();
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
