#pragma once

// The row-worker set of one CpuExpertPool (llama.cpp-w4e9): the threads that run a host-expert job's
// row loop, Phase 2 of ggml_sycl_cpu_expert_mul_mat_batched.
//
// Pool jobs used the shared TBB arena. A decode token runs ~72 of them, each ~0.45 ms, with 0.3-1 ms
// between them, and in such a gap TBB returns the arena's workers to its sleeping pool: a job found
// 6-9 of 22 threads awake and read its weights at 10-12 GB/s. These workers instead keep polling for
// the next run for a spin budget after their last chunk and then block on a condition variable, so
// a job that follows within the budget starts on awake threads, and a process that has gone idle
// keeps no thread polling for longer than the budget. Every poll pauses and then yields the core, so
// a runnable thread of another pool (host attention workers, ggml-cpu's team) is not kept waiting.
//
// run() splits its rows into chunks that the caller and the workers claim with a compare-exchange on
// one word holding the run's generation, its chunk count and its next chunk. A worker reads a run's
// rows and body only after it has claimed one of that run's chunks, and run() returns only after
// every claimed chunk has finished, so a worker can never apply one run's body to another's rows.
//
// The chunks of a run depend only on its row count, never on the thread count or on which thread
// claims what, so every caller sees the same row ranges.
//
// This header names only the standard library, so a host test builds it without a device.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ggml_sycl {

class CpuRowWorkers {
  public:
    // How long a worker keeps polling after its last chunk before it blocks is start()'s spin_us:
    // spin_us_default unless CpuExpertPool reads GGML_SYCL_CPU_ROW_SPIN_US, and never more than
    // spin_us_max. 0 blocks at once. In a host microbench of the decode shape (llama.cpp-w4e9) a
    // job read 21-29 GB/s at the default and 11-15 GB/s at 0, against 9-15 on the TBB arena. The
    // GGML_SYCL_CPU_ROW_SPIN_US row of docs/backend/sycl-env-vars.md gives the in-situ figures.
    static constexpr int spin_us_default = 2000;
    static constexpr int spin_us_max     = 10000;
    // Rows per chunk. A run of more than max_chunks * chunk_rows_min rows uses larger chunks, so that
    // its chunk count fits the claim word.
    static constexpr int chunk_rows_min  = 64;
    static constexpr int max_chunks      = 0xffff;

    using row_body = std::function<void(int first_row, int end_row, int slot)>;

    CpuRowWorkers() = default;
    ~CpuRowWorkers();

    CpuRowWorkers(const CpuRowWorkers &)             = delete;
    CpuRowWorkers & operator=(const CpuRowWorkers &) = delete;
    CpuRowWorkers(CpuRowWorkers &&)                  = delete;
    CpuRowWorkers & operator=(CpuRowWorkers &&)      = delete;

    // Starts n_threads - 1 workers that poll for spin_us (clamped to [0, spin_us_max]) after their
    // last chunk; the thread that calls run() is the last of the n_threads. Starts nothing when
    // n_threads < 2 or when workers are already running.
    void start(int n_threads, int spin_us = spin_us_default);
    // Signals and joins the workers, after any run in flight. run() then runs every chunk on its caller.
    // The workers block only on this set's condition variable, so the join cannot wait on other work.
    void stop();

    // Threads that run a run's chunks, the caller included: 1 while no worker is running.
    int n_threads() const { return n_threads_.load(std::memory_order_acquire); }

    // The running workers' spin budget in microseconds: start()'s, clamped.
    int spin_us() const { return spin_us_.load(std::memory_order_acquire); }

    // Runs body(first_row, end_row, slot) over the chunks of [0, n_rows), each row in exactly one call,
    // on the calling thread (slot 0) and the workers (slots 1 .. n_threads() - 1), and returns when every
    // call has returned. Calls from several threads run one after another. body must not throw.
    void run(int n_rows, const row_body & body);

  private:
    void worker(int slot, uint32_t seen_gen, int spin_us);
    // Claims and runs chunks of generation gen until none is left or a later run has started.
    void run_chunks(uint32_t gen, int slot);
    // Waits for a generation other than seen_gen, polling for spin_us and then blocking; false when
    // the set is stopping.
    bool wait_for_run(uint32_t seen_gen, uint32_t & gen, int spin_us);

    std::vector<std::thread> threads_;
    std::atomic<int>         n_threads_{ 1 };
    std::atomic<int>         spin_us_{ spin_us_default };
    std::mutex               run_mutex_;  // one run at a time; start() and stop() wait for it
    // (generation << 32) | (chunk count << 16) | next chunk
    std::atomic<uint64_t>    claim_{ 0 };
    std::atomic<int>         chunks_done_{ 0 };
    std::atomic<int>         sleepers_{ 0 };
    std::atomic<bool>        stopping_{ false };
    std::mutex               sleep_mutex_;
    std::condition_variable  sleep_cv_;
    // The current run's rows and body, written by run() before it publishes the run's claim word and
    // read by a worker only after it has claimed a chunk of that run.
    int                      n_rows_     = 0;
    int                      chunk_rows_ = chunk_rows_min;
    row_body                 body_;
};

}  // namespace ggml_sycl
