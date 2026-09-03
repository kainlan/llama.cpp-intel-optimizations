// Regression test for llama.cpp-41bs.
//
// The defect: ExpertPredictor::predict() and ::record_actual() read the
// plain, non-atomic guard fields initialized_/n_layers_ BEFORE taking
// mutex_, while reset() holds mutex_ while clearing the backing vectors
// (last_experts_, freq_table_, last_prediction_, ...). A concurrent reset()
// landing between the guard read and the lock let predict()/record_actual()
// index into vectors reset() had already cleared -- undefined behavior, and
// on this toolchain a deterministic crash (see the isolated RED reproducer
// referenced in the task report; not committed here since it duplicates the
// class shape rather than testing the real code). init() had the same
// shape: it read/wrote the guard fields with no lock at all.
//
// The fix (shape A, per the task's design addendum): every public accessor
// that reads initialized_/n_layers_ (and the vectors those fields gate) now
// takes mutex_ FIRST, then evaluates the guard under the lock, matching
// reset()'s own ordering. See expert-prefetch.cpp: init(), predict(),
// record_actual(), get_frequency_ranking(), and the top of predict_pregate().
//
// This test exercises the REAL ggml_sycl::ExpertPredictor -- no mock
// reimplementation. None of the methods exercised here (init/reset/predict/
// record_actual/get_frequency_ranking) touch a sycl::queue or any device
// state, so this binary never opens a SYCL device; it needs the SYCL
// compiler only because expert-prefetch.hpp includes sycl/sycl.hpp for
// OTHER members of the class (scores_dev_ etc.) that this test does not
// exercise.
//
// RED/GREEN: see the task report for the isolated pre-fix reproducer
// (crashed 100% of runs, both under ASan and at plain -O2) and the isolated
// post-fix control (0 crashes across the same stress). This file is the
// GREEN acceptance test against the real, fixed class.

#include "../expert-prefetch.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously (llama.cpp-u2mz precedent). Use an explicit
// check that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::ExpertPredictor;

static constexpr int  N_LAYERS         = 8;
static constexpr int  N_EXPERTS        = 16;
static constexpr int  N_EXPERTS_USED   = 4;
static constexpr int  N_HAMMER_THREADS = 6;
static constexpr auto STRESS_DURATION  = std::chrono::milliseconds(3000);

int main() {
    ExpertPredictor predictor;
    predictor.init(N_LAYERS, N_EXPERTS, N_EXPERTS_USED);

    std::atomic<bool> stop{ false };
    std::atomic<bool> violation{ false };
    std::atomic<int>  violation_count{ 0 };

    // Hammer threads: call predict()/record_actual()/get_frequency_ranking()
    // in a tight loop against a fixed set of layer indices that are valid
    // for the ORIGINAL init() call above. Each iteration also re-derives its
    // "actual experts" from the layer/iteration count rather than reading
    // any predictor state, so it never depends on ordering with the reset
    // thread for correctness -- only for memory safety, which is exactly
    // what this test is checking.
    auto hammer = [&](int seed) {
        long iter = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            const int layer = (seed + static_cast<int>(iter)) % N_LAYERS;

            std::vector<int> predicted = predictor.predict(layer);
            if (predicted.size() > static_cast<size_t>(N_EXPERTS_USED)) {
                violation.store(true, std::memory_order_relaxed);
                violation_count.fetch_add(1, std::memory_order_relaxed);
            }
            for (int e : predicted) {
                if (e < 0 || e >= N_EXPERTS) {
                    violation.store(true, std::memory_order_relaxed);
                    violation_count.fetch_add(1, std::memory_order_relaxed);
                }
            }

            std::vector<int> actual;
            for (int k = 0; k < N_EXPERTS_USED; k++) {
                actual.push_back((layer + k + static_cast<int>(iter)) % N_EXPERTS);
            }
            predictor.record_actual(layer, actual);

            auto ranking = predictor.get_frequency_ranking(layer);
            for (const auto & p : ranking) {
                if (p.first < 0 || p.first >= N_EXPERTS) {
                    violation.store(true, std::memory_order_relaxed);
                    violation_count.fetch_add(1, std::memory_order_relaxed);
                }
            }

            iter++;
        }
    };

    // Concurrently tears down and re-initializes the predictor -- this is
    // the reset()/re-init() race the task's design addendum calls out.
    auto reset_loop = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            predictor.reset();
            predictor.init(N_LAYERS, N_EXPERTS, N_EXPERTS_USED);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(N_HAMMER_THREADS + 1);
    for (int i = 0; i < N_HAMMER_THREADS; i++) {
        threads.emplace_back(hammer, i);
    }
    threads.emplace_back(reset_loop);

    std::this_thread::sleep_for(STRESS_DURATION);
    stop.store(true, std::memory_order_relaxed);
    for (auto & t : threads) {
        t.join();
    }

    CHECK(!violation.load(std::memory_order_relaxed),
          "predict()/get_frequency_ranking() returned an "
          "out-of-range index or oversized result under "
          "concurrent reset() -- see violation_count");
    CHECK(violation_count.load(std::memory_order_relaxed) == 0, "violation_count must be exactly 0");

    std::printf(
        "PASS: test-sycl-expert-predictor-guard (0 violations, no crash under %lld ms of concurrent "
        "predict()/record_actual()/get_frequency_ranking() vs reset())\n",
        static_cast<long long>(STRESS_DURATION.count()));
    return 0;
}
