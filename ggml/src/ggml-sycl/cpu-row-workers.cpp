// The row-worker set of one CpuExpertPool (llama.cpp-w4e9); see cpu-row-workers.hpp.

#include "cpu-row-workers.hpp"

#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#    include <immintrin.h>
#    define GGML_SYCL_ROW_WORKERS_PAUSE() _mm_pause()
#else
#    define GGML_SYCL_ROW_WORKERS_PAUSE() std::this_thread::yield()
#endif

namespace ggml_sycl {

namespace {

uint32_t claim_gen(uint64_t word) {
    return static_cast<uint32_t>(word >> 32);
}

uint32_t claim_chunks(uint64_t word) {
    return static_cast<uint32_t>((word >> 16) & 0xffff);
}

uint32_t claim_next(uint64_t word) {
    return static_cast<uint32_t>(word & 0xffff);
}

uint64_t claim_word(uint32_t gen, uint32_t n_chunks) {
    return (static_cast<uint64_t>(gen) << 32) | (static_cast<uint64_t>(n_chunks) << 16);
}

}  // namespace

CpuRowWorkers::~CpuRowWorkers() {
    stop();
}

void CpuRowWorkers::start(int n_threads, int spin_us) {
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    if (!threads_.empty() || n_threads < 2) {
        return;
    }
    const int      budget_us = std::min(std::max(spin_us, 0), spin_us_max);
    // A new worker has seen every run so far, so it waits for the next one.
    const uint32_t gen       = claim_gen(claim_.load(std::memory_order_acquire));
    threads_.reserve(static_cast<size_t>(n_threads - 1));
    for (int slot = 1; slot < n_threads; slot++) {
        threads_.emplace_back(&CpuRowWorkers::worker, this, slot, gen, budget_us);
    }
    spin_us_.store(budget_us, std::memory_order_release);
    n_threads_.store(n_threads, std::memory_order_release);
}

void CpuRowWorkers::stop() {
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    if (threads_.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(sleep_mutex_);
        stopping_.store(true, std::memory_order_seq_cst);
    }
    sleep_cv_.notify_all();
    for (std::thread & t : threads_) {
        t.join();
    }
    threads_.clear();
    n_threads_.store(1, std::memory_order_release);
    stopping_.store(false, std::memory_order_release);
}

void CpuRowWorkers::run(int n_rows, const row_body & body) {
    if (n_rows <= 0) {
        return;
    }
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    const int64_t               rows       = n_rows;
    const int64_t               chunk_rows = std::max<int64_t>(chunk_rows_min, (rows + max_chunks - 1) / max_chunks);
    const int                   n_chunks   = static_cast<int>((rows + chunk_rows - 1) / chunk_rows);
    if (threads_.empty()) {
        // The same chunks, on this thread.
        for (int64_t first = 0; first < rows; first += chunk_rows) {
            body(static_cast<int>(first), static_cast<int>(std::min(rows, first + chunk_rows)), 0);
        }
        return;
    }

    n_rows_     = n_rows;
    chunk_rows_ = static_cast<int>(chunk_rows);
    body_       = std::cref(body);
    chunks_done_.store(0, std::memory_order_relaxed);
    const uint32_t gen = claim_gen(claim_.load(std::memory_order_relaxed)) + 1;
    // Publishes the fields above with the claim word, and orders against the sleepers count: either a
    // worker about to block sees the new generation in its wait predicate, or this sees it counted.
    claim_.store(claim_word(gen, static_cast<uint32_t>(n_chunks)), std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
        // Taking the lock first means a worker between counting itself and blocking is already waiting.
        {
            std::lock_guard<std::mutex> lock(sleep_mutex_);
        }
        sleep_cv_.notify_all();
    }
    run_chunks(gen, 0);
    // The chunks other workers claimed are the only ones left; they finish without waiting on anything.
    // Yielding here too lets a worker that was preempted mid-chunk get a core back sooner.
    while (chunks_done_.load(std::memory_order_acquire) < n_chunks) {
        GGML_SYCL_ROW_WORKERS_PAUSE();
        std::this_thread::yield();
    }
    body_ = nullptr;
}

void CpuRowWorkers::run_chunks(uint32_t gen, int slot) {
    uint64_t word = claim_.load(std::memory_order_acquire);
    for (;;) {
        if (claim_gen(word) != gen || claim_next(word) >= claim_chunks(word)) {
            return;
        }
        // The next-chunk field is the low one and stays below the chunk count, so +1 never carries.
        if (!claim_.compare_exchange_weak(word, word + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }
        const int64_t first = static_cast<int64_t>(claim_next(word)) * chunk_rows_;
        const int64_t end   = std::min<int64_t>(n_rows_, first + chunk_rows_);
        body_(static_cast<int>(first), static_cast<int>(end), slot);
        chunks_done_.fetch_add(1, std::memory_order_release);
        word = claim_.load(std::memory_order_acquire);
    }
}

bool CpuRowWorkers::wait_for_run(uint32_t seen_gen, uint32_t & gen, int spin_us) {
    using clock                = std::chrono::steady_clock;
    const auto budget          = std::chrono::microseconds(spin_us);
    auto       idle_since      = clock::now();
    const auto new_run_or_stop = [&] {
        return claim_gen(claim_.load(std::memory_order_seq_cst)) != seen_gen ||
               stopping_.load(std::memory_order_seq_cst);
    };
    for (;;) {
        gen = claim_gen(claim_.load(std::memory_order_acquire));
        if (gen != seen_gen) {
            return true;
        }
        if (stopping_.load(std::memory_order_acquire)) {
            return false;
        }
        if (clock::now() - idle_since < budget) {
            // Every poll gives the core away: a runnable thread of another pool, such as the host
            // attention workers of a token's CPU-resident KV layers, runs before this one polls again.
            GGML_SYCL_ROW_WORKERS_PAUSE();
            std::this_thread::yield();
            continue;
        }
        // The budget is spent (at once when it is 0): block until run() or stop() wakes this worker.
        std::unique_lock<std::mutex> lock(sleep_mutex_);
        sleepers_.fetch_add(1, std::memory_order_seq_cst);
        sleep_cv_.wait(lock, new_run_or_stop);
        sleepers_.fetch_sub(1, std::memory_order_seq_cst);
        idle_since = clock::now();
    }
}

void CpuRowWorkers::worker(int slot, uint32_t seen_gen, int spin_us) {
    uint32_t gen = seen_gen;
    while (wait_for_run(seen_gen, gen, spin_us)) {
        run_chunks(gen, slot);
        seen_gen = gen;
    }
}

}  // namespace ggml_sycl
