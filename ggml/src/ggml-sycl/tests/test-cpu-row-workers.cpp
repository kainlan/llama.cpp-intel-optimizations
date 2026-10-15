// The CpuExpertPool's row-worker set (llama.cpp-w4e9): every row of a run is computed exactly once, in
// chunks that depend only on the row count, by the caller and the workers; callers on several threads
// run one after another; the workers stop polling once the spin budget is spent (at once when it is 0),
// wake for the next run, and stop() joins them, also while they are blocked; start() clamps the budget.
// Host only: cpu-row-workers.hpp names only the standard library.

#include "../cpu-row-workers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

// Release builds define NDEBUG, which would compile assert() away and let this
// test pass vacuously, so every check is an explicit one that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::CpuRowWorkers;
using steady = std::chrono::steady_clock;

static void busy_us(int us) {
    const auto until = steady::now() + std::chrono::microseconds(us);
    while (steady::now() < until) {
    }
}

// One run of n_rows rows: how often each row ran, the chunks, and the slots that ran them.
struct run_record {
    std::unique_ptr<std::atomic<int>[]> hits;
    std::vector<std::pair<int, int>>    chunks;
    std::set<int>                       slots;
    std::mutex                          mutex;
    int                                 bad_slot = 0;
};

static void record_run(CpuRowWorkers & w, int n_rows, run_record & rec, int busy_per_chunk_us) {
    rec.hits.reset(new std::atomic<int>[n_rows > 0 ? n_rows : 1]);
    for (int r = 0; r < n_rows; r++) {
        rec.hits[r].store(0);
    }
    rec.chunks.clear();
    rec.slots.clear();
    rec.bad_slot = 0;
    w.run(n_rows, [&](int first, int end, int slot) {
        for (int r = first; r < end; r++) {
            rec.hits[r].fetch_add(1);
        }
        if (busy_per_chunk_us > 0) {
            busy_us(busy_per_chunk_us);
        }
        std::lock_guard<std::mutex> lock(rec.mutex);
        rec.chunks.emplace_back(first, end);
        rec.slots.insert(slot);
        if (slot < 0 || slot >= w.n_threads()) {
            rec.bad_slot++;
        }
    });
}

static bool every_row_once(const run_record & rec, int n_rows) {
    for (int r = 0; r < n_rows; r++) {
        if (rec.hits[r].load() != 1) {
            return false;
        }
    }
    return true;
}

// The chunks run() must produce for n_rows: contiguous, chunk_rows_min rows each (the last may be
// short) unless that would exceed max_chunks, and then the smallest size that does not.
static std::vector<std::pair<int, int>> expected_chunks(int n_rows) {
    const long long rows  = n_rows;
    long long       chunk = CpuRowWorkers::chunk_rows_min;
    if ((rows + chunk - 1) / chunk > CpuRowWorkers::max_chunks) {
        chunk = (rows + CpuRowWorkers::max_chunks - 1) / CpuRowWorkers::max_chunks;
    }
    std::vector<std::pair<int, int>> out;
    for (long long first = 0; first < rows; first += chunk) {
        out.emplace_back(static_cast<int>(first), static_cast<int>(std::min(rows, first + chunk)));
    }
    return out;
}

static double process_cpu_ms() {
    return 1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

int main() {
    const int row_counts[] = { 1, 63, 64, 65, 12480, 100000, CpuRowWorkers::max_chunks * 64 + 100 };

    // 1. Every row exactly once, in the expected chunks, on valid slots, at every thread count; the
    //    chunks are the same with and without workers.
    for (int n_threads : { 1, 2, 4, 8 }) {
        CpuRowWorkers w;
        w.start(n_threads);
        CHECK(w.n_threads() == n_threads, "n_threads() is not the started count");
        for (int n_rows : row_counts) {
            const int reps = n_rows > 200000 ? 2 : 20;
            for (int rep = 0; rep < reps; rep++) {
                run_record rec;
                record_run(w, n_rows, rec, 0);
                CHECK(every_row_once(rec, n_rows), "a row ran zero times or more than once");
                CHECK(rec.bad_slot == 0, "a chunk ran on a slot outside [0, n_threads)");
                std::sort(rec.chunks.begin(), rec.chunks.end());
                CHECK(rec.chunks == expected_chunks(n_rows), "the chunks differ from the row-count partition");
                CHECK(static_cast<int>(rec.chunks.size()) <= CpuRowWorkers::max_chunks, "more than max_chunks");
            }
        }
    }

    // 2. A run of no rows calls nothing.
    {
        CpuRowWorkers w;
        w.start(4);
        int calls = 0;
        w.run(0, [&](int, int, int) { calls++; });
        w.run(-5, [&](int, int, int) { calls++; });
        CHECK(calls == 0, "a run of no rows called its body");
    }

    // 3. The workers take part: over a few runs of slow chunks, some chunk runs on a worker slot.
    CpuRowWorkers w;
    w.start(8);
    {
        bool worker_ran = false;
        for (int rep = 0; rep < 50 && !worker_ran; rep++) {
            run_record rec;
            record_run(w, 64 * 64, rec, 50);
            CHECK(every_row_once(rec, 64 * 64), "a row ran zero times or more than once (slow chunks)");
            worker_ran = rec.slots.size() > 1 || rec.slots.count(0) == 0;
        }
        CHECK(worker_ran, "no chunk ran on a worker in 50 runs of slow chunks");
    }

    // 4. Callers on three threads, each with its own rows, run one after another and each sees its own
    //    rows computed exactly once.
    {
        std::atomic<int>         failures{ 0 };
        std::vector<std::thread> callers;
        for (int c = 0; c < 3; c++) {
            callers.emplace_back([&w, &failures, c] {
                for (int rep = 0; rep < 100; rep++) {
                    run_record rec;
                    const int  n_rows = 1000 + 997 * c + rep;
                    record_run(w, n_rows, rec, 0);
                    if (!every_row_once(rec, n_rows) || rec.bad_slot != 0) {
                        failures.fetch_add(1);
                    }
                }
            });
        }
        for (std::thread & t : callers) {
            t.join();
        }
        CHECK(failures.load() == 0, "a concurrent caller's rows did not each run exactly once");
    }

    // 5. Idle: once the budget is spent the workers block. Polling, 7 workers would burn ~7 CPU-ms per
    //    wall-ms; blocked, the process burns next to nothing.
    {
        CHECK(w.spin_us() == CpuRowWorkers::spin_us_default, "start() without a budget did not take the default");
        std::this_thread::sleep_for(std::chrono::microseconds(CpuRowWorkers::spin_us_default) +
                                    std::chrono::milliseconds(50));
        const double cpu0 = process_cpu_ms();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const double idle_cpu_ms = process_cpu_ms() - cpu0;
        std::printf("idle window: 200 ms wall, %.1f ms CPU over %d threads\n", idle_cpu_ms, w.n_threads());
        CHECK(idle_cpu_ms < 50.0, "the workers keep polling after the spin budget");
    }

    // 6. Blocked workers wake for the next run and take part in it.
    {
        bool worker_ran = false;
        for (int rep = 0; rep < 20 && !worker_ran; rep++) {
            std::this_thread::sleep_for(std::chrono::microseconds(CpuRowWorkers::spin_us_default) +
                                        std::chrono::milliseconds(5));
            run_record rec;
            record_run(w, 64 * 200, rec, 50);
            CHECK(every_row_once(rec, 64 * 200), "a row ran zero times or more than once after blocking");
            worker_ran = rec.slots.size() > 1 || rec.slots.count(0) == 0;
        }
        CHECK(worker_ran, "blocked workers never took part in a run after waking");
    }

    // 7. stop() joins the workers; run() then runs every chunk on its caller, and start() restarts.
    {
        w.stop();
        CHECK(w.n_threads() == 1, "n_threads() after stop() is not 1");
        run_record rec;
        record_run(w, 12480, rec, 0);
        CHECK(every_row_once(rec, 12480), "a row ran zero times or more than once after stop()");
        CHECK(rec.slots.size() == 1 && rec.slots.count(0) == 1, "a chunk ran off the caller after stop()");
        w.stop();  // a second stop is a no-op
        w.start(4);
        CHECK(w.n_threads() == 4, "n_threads() after a restart is not the started count");
        w.start(8);  // a start while running is a no-op
        CHECK(w.n_threads() == 4, "a second start() changed the running set");
        run_record rec2;
        record_run(w, 12480, rec2, 0);
        CHECK(every_row_once(rec2, 12480), "a row ran zero times or more than once after a restart");
    }

    // 8. A budget of 0 blocks at once: the workers still take part in every run (each run wakes them),
    //    and right after a run the process burns next to nothing.
    {
        CpuRowWorkers z;
        z.start(8, 0);
        CHECK(z.spin_us() == 0, "start(n, 0) did not keep a budget of 0");
        bool worker_ran = false;
        for (int rep = 0; rep < 50; rep++) {
            run_record rec;
            record_run(z, 64 * 64, rec, 50);
            CHECK(every_row_once(rec, 64 * 64), "a row ran zero times or more than once at a budget of 0");
            worker_ran = worker_ran || rec.slots.size() > 1 || rec.slots.count(0) == 0;
        }
        CHECK(worker_ran, "no chunk ran on a worker in 50 runs at a budget of 0");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const double cpu0 = process_cpu_ms();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const double idle_cpu_ms = process_cpu_ms() - cpu0;
        std::printf("budget 0: 200 ms wall, %.1f ms CPU over %d threads\n", idle_cpu_ms, z.n_threads());
        CHECK(idle_cpu_ms < 50.0, "the workers poll after a run at a budget of 0");
    }

    // 9. start() clamps the budget to [0, spin_us_max].
    {
        CpuRowWorkers c;
        c.start(4, -5);
        CHECK(c.spin_us() == 0, "a negative budget was not clamped to 0");
        c.stop();
        c.start(4, 999999);
        CHECK(c.spin_us() == CpuRowWorkers::spin_us_max, "a budget above spin_us_max was not clamped to it");
        run_record rec;
        record_run(c, 12480, rec, 0);
        CHECK(every_row_once(rec, 12480), "a row ran zero times or more than once at the largest budget");
    }

    // 10. stop() joins workers that are blocked on the condition variable, and returns promptly.
    {
        CpuRowWorkers p;
        p.start(8);
        run_record rec;
        record_run(p, 12480, rec, 0);
        std::this_thread::sleep_for(std::chrono::microseconds(CpuRowWorkers::spin_us_default) +
                                    std::chrono::milliseconds(50));
        const auto t0 = steady::now();
        p.stop();
        const double stop_ms = std::chrono::duration<double, std::milli>(steady::now() - t0).count();
        std::printf("stop with blocked workers: %.2f ms\n", stop_ms);
        CHECK(p.n_threads() == 1, "n_threads() after stopping blocked workers is not 1");
        CHECK(stop_ms < 1000.0, "stop() took over a second to join blocked workers");
    }

    std::printf("PASS: cpu row workers\n");
    return 0;
}
