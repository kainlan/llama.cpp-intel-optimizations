// STREAM-style host memory bandwidth bench (copy / scale / triad, plus a
// read-only kernel), for comparing against the CpuExpertPool expert matvec.
//
// Build without CMake:
//   g++ -O3 -mavx2 -std=c++17 -pthread bench-host-stream.cpp -o bench-host-stream
//
// Three double arrays of N elements; default 1 GiB each (3 GiB total, ~80x the
// 36 MiB L3). Threads each own a contiguous static slice. Per (kernel,threads)
// there are --rounds timed trials; threads are visited round-robin inside each
// round so a changing ambient load hits every thread count alike (interleaved).
//
// Byte counting follows STREAM: copy and scale count 2 arrays, triad 3, and
// write-allocate traffic is NOT counted, so copy/scale/triad understate the
// DRAM traffic by up to 1.5x/1.5x/1.33x. The `read` kernel (AVX2 sum over one
// array, no stores) is the figure that matches a weight-streaming matvec.
//
// Scheduling: --sched static (default: each thread owns one contiguous slice;
// a descheduled thread stalls the whole trial) or --sched dynamic (threads pull
// 4 MiB chunks off an atomic counter, as TBB does in the expert path, so a
// preempted thread costs one chunk). Under the permanent ambient load of this
// host dynamic is the fairer model of the production kernel.
//
// Pinning: --pin none (default; the scheduler places threads, as in production)
// or --pin LIST with a comma list of CPU ids; thread i is pinned to LIST[i]
// (--pin-label NAME is what the CSV pin column prints for it).
// Output: kernel,threads,pin,sched,best_gbps,median_gbps,min_gbps (decimal GB/s).
//
// MIT license
// SPDX-License-Identifier: MIT

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static std::vector<int> split_ints(const std::string & s) {
    std::vector<int> out;
    size_t           i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) {
            j = s.size();
        }
        out.push_back(atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

struct spin_barrier {
    std::atomic<int> count{ 0 };
    std::atomic<int> gen{ 0 };
    int              n = 1;

    void wait() {
        const int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) {
                _mm_pause();
            }
        }
    }
};

enum kernel_id { K_COPY, K_SCALE, K_TRIAD, K_READ, K_COUNT };

static const char * k_names[K_COUNT]  = { "copy", "scale", "triad", "read" };
static const int    k_arrays[K_COUNT] = { 2, 2, 3, 1 };

static double *            g_a, *g_b, *g_c;
static size_t              g_n;
static std::atomic<size_t> g_next{ 0 };
static constexpr size_t    k_chunk = (4u << 20) / sizeof(double);

static void run_slice(kernel_id k, size_t lo, size_t hi, double * sink) {
    const double s = 3.0;
    double *     a = g_a, *b = g_b, *c = g_c;
    switch (k) {
        case K_COPY:
            for (size_t i = lo; i < hi; i++) {
                c[i] = a[i];
            }
            break;
        case K_SCALE:
            for (size_t i = lo; i < hi; i++) {
                b[i] = s * c[i];
            }
            break;
        case K_TRIAD:
            for (size_t i = lo; i < hi; i++) {
                a[i] = b[i] + s * c[i];
            }
            break;
        case K_READ:
            {
                __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
                size_t  i = lo;
                for (; i + 16 <= hi; i += 16) {
                    s0 = _mm256_add_pd(s0, _mm256_load_pd(a + i));
                    s1 = _mm256_add_pd(s1, _mm256_load_pd(a + i + 4));
                    s2 = _mm256_add_pd(s2, _mm256_load_pd(a + i + 8));
                    s3 = _mm256_add_pd(s3, _mm256_load_pd(a + i + 12));
                }
                s0 = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
                double t[4];
                _mm256_storeu_pd(t, s0);
                *sink += t[0] + t[1] + t[2] + t[3];
                break;
            }
        default:
            break;
    }
}

int main(int argc, char ** argv) {
    long long        gib     = 1;  // per array; signed so that a negative value reaches the guard below
    int              rounds  = 9;
    std::vector<int> threads = { 1, 2, 4, 8, 12, 16, 20, 24 };
    std::vector<int> pin;
    std::string      pin_name  = "none";
    std::string      pin_label = "none";
    bool             dynamic   = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto        v = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : "";
        };
        if (a == "--gib") {
            gib = atoll(v().c_str());
        } else if (a == "--rounds") {
            rounds = atoi(v().c_str());
        } else if (a == "--pin-label") {
            pin_label = v();
        } else if (a == "--sched") {
            dynamic = v() == "dynamic";
        } else if (a == "--threads") {
            threads = split_ints(v());
        } else if (a == "--pin") {
            pin_name = v();
            if (pin_name != "none") {
                pin = split_ints(pin_name);
            }
        } else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (rounds < 1 || gib < 1) {  // the summary takes best/median/min of the trials; none would be undefined
        fprintf(stderr, "--rounds and --gib must be >= 1\n");
        return 2;
    }
    g_n = ((size_t) gib << 30) / sizeof(double);
    g_a = (double *) aligned_alloc(4096, g_n * sizeof(double));
    g_b = (double *) aligned_alloc(4096, g_n * sizeof(double));
    g_c = (double *) aligned_alloc(4096, g_n * sizeof(double));
    if (!g_a || !g_b || !g_c) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    // First touch with the max thread count; UMA, so placement does not matter,
    // but this commits every page before any timing.
    {
        const int                nt = *std::max_element(threads.begin(), threads.end());
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) {
            th.emplace_back([&, t]() {
                const size_t lo = g_n / nt * t, hi = t == nt - 1 ? g_n : g_n / nt * (t + 1);
                for (size_t i = lo; i < hi; i++) {
                    g_a[i] = 1.0;
                    g_b[i] = 2.0;
                    g_c[i] = 0.5;
                }
            });
        }
        for (auto & t : th) {
            t.join();
        }
    }

    std::map<std::pair<int, int>, std::vector<double>> res;  // (kernel,threads) -> GB/s per round
    double                                             sink = 0;
    for (int r = 0; r < rounds + 1; r++) {                   // round 0 is warm-up, discarded
        for (int nt : threads) {
            for (int k = 0; k < K_COUNT; k++) {
                spin_barrier bar;
                bar.n = nt;
                g_next.store(0);
                std::vector<std::thread>                           th;
                std::vector<double>                                sinks(nt, 0.0);
                std::vector<std::chrono::steady_clock::time_point> t_beg(nt), t_end(nt);
                std::vector<char>                                  worked(nt, 0);
                for (int t = 0; t < nt; t++) {
                    th.emplace_back([&, t]() {
                        if (!pin.empty()) {
                            cpu_set_t set;
                            CPU_ZERO(&set);
                            CPU_SET(pin[t % pin.size()], &set);
                            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
                        }
                        const size_t lo = (g_n / nt / 16 * 16) * t;
                        const size_t hi = t == nt - 1 ? g_n : (g_n / nt / 16 * 16) * (t + 1);
                        bar.wait();
                        t_beg[t] = std::chrono::steady_clock::now();
                        if (dynamic) {
                            for (;;) {
                                const size_t c0 = g_next.fetch_add(k_chunk, std::memory_order_relaxed);
                                if (c0 >= g_n) {
                                    break;
                                }
                                worked[t] = 1;
                                run_slice((kernel_id) k, c0, std::min(c0 + k_chunk, g_n), &sinks[t]);
                            }
                        } else {
                            run_slice((kernel_id) k, lo, hi, &sinks[t]);
                            worked[t] = 1;
                        }
                        t_end[t] = std::chrono::steady_clock::now();
                    });
                }
                for (auto & t : th) {
                    t.join();
                }
                // Elapsed = last working thread out minus first in, taken on the
                // workers themselves so a descheduled main thread cannot skew it.
                // Threads that never got a chunk (the scheduler ran them only after
                // the others had finished everything) are left out of the window.
                auto t_lo = std::chrono::steady_clock::time_point::max();
                auto t_hi = std::chrono::steady_clock::time_point::min();
                for (int t = 0; t < nt; t++) {
                    if (!worked[t]) {
                        continue;
                    }
                    t_lo = std::min(t_lo, t_beg[t]);
                    t_hi = std::max(t_hi, t_end[t]);
                }
                const double sec = std::chrono::duration<double>(t_hi - t_lo).count();
                for (double s : sinks) {
                    sink += s;
                }
                if (r > 0) {
                    res[{ k, nt }].push_back((double) g_n * sizeof(double) * k_arrays[k] / sec / 1e9);
                }
            }
        }
    }
    printf("kernel,threads,pin,sched,best_gbps,median_gbps,min_gbps,rounds\n");
    for (int k = 0; k < K_COUNT; k++) {
        for (int nt : threads) {
            auto v = res[{ k, nt }];
            std::sort(v.begin(), v.end());
            printf("%s,%d,%s,%s,%.1f,%.1f,%.1f,%d\n", k_names[k], nt, (pin_name == "none" ? "none" : pin_label.c_str()),
                   dynamic ? "dynamic" : "static", v.back(), v[v.size() / 2], v.front(), (int) v.size());
        }
    }
    if (sink == 12345.678) {
        printf("# %f\n", sink);  // keep the read kernel live
    }
    return 0;
}
