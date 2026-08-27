// TKV-13 (B2) step 5.1: sustained host DDR5 read bandwidth, matching the
// demoted-layer attention access pattern (docs/plans/2026-08-27-tkv13-b2-
// addendum.md §5.1). Pure host C++, no SYCL, no ggml -- a standalone
// microbench, buildable and runnable on any host with no GPU involvement at
// all. Not a ctest target (mirrors the bench-dnnl-ops.cpp precedent: a plain
// add_executable with no add_test -- this reports numbers for the lead to
// record in the gate evidence, it does not pass/fail).
//
// Access pattern: ggml-cpu's FLASH_ATTN_EXT kernel (ggml_compute_forward_
// flash_attn_ext_f16, ggml/src/ggml-cpu/ops.cpp:9192) reads one contiguous
// K row (DK elements) and one contiguous V row (DV elements) per KV
// position, for every KV position in range, per query. At TG (one query
// token, batch=1) that is a single sequential sweep over the layer's K/V
// rows -- exactly what this benchmark reproduces, scaled across the number
// of demoted full-attention layers a token must read at a given fill.
//
// Usage:
//   bench-host-kv-bandwidth --dk 128 --dv 128 --n-kv 32768 --n-layers 8 --iters 20
//   bench-host-kv-bandwidth --dk 128 --dv 128 --n-kv 32768 --n-layers 8 --threads 4
//
// dk/dv default to a common GQA head_dim (128); n-kv/n-layers should be set
// from the landed plan's actual kv_per_layer / demoted-layer count at the
// gate's fill (docs/plans/2026-08-27-tkv13-b2-addendum.md §5.3) -- this
// program does not read the plan itself, it only reproduces the byte
// geometry the caller supplies.
//
// --threads N runs N concurrent sweeps over independent buffers and reports
// AGGREGATE GB/s (sum of bytes / wall time of the slowest thread) --
// AttnHostPool (attn-host-pool.hpp) dispatches demoted-layer attention
// across several persistent worker threads, so the roofline's host_BW term
// (docs/plans/2026-08-27-tkv13-b2-addendum.md §5.3) should be measured at
// the thread count the pool actually uses, not single-threaded, once memory
// controller contention is the thing the roofline needs to reflect.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
    int64_t dk       = 128;
    int64_t dv       = 128;
    int64_t n_kv     = 32768;
    int64_t n_layers = 8;
    int     iters    = 20;
    int     threads  = 1;
};

bool parse_i64(const char * s, int64_t * out) {
    char *    end = nullptr;
    long long v   = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0) {
        return false;
    }
    *out = static_cast<int64_t>(v);
    return true;
}

bool parse_args(int argc, char ** argv, Args * args) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg        = argv[i];
        auto              need_value = [&](int64_t * dst) -> bool {
            if (i + 1 >= argc) {
                return false;
            }
            return parse_i64(argv[++i], dst);
        };
        if (arg == "--dk") {
            if (!need_value(&args->dk)) {
                return false;
            }
        } else if (arg == "--dv") {
            if (!need_value(&args->dv)) {
                return false;
            }
        } else if (arg == "--n-kv") {
            if (!need_value(&args->n_kv)) {
                return false;
            }
        } else if (arg == "--n-layers") {
            if (!need_value(&args->n_layers)) {
                return false;
            }
        } else if (arg == "--iters") {
            int64_t v;
            if (!need_value(&v) || v > 1000000) {
                return false;
            }
            args->iters = static_cast<int>(v);
        } else if (arg == "--threads") {
            int64_t v;
            if (!need_value(&v) || v > 4096) {
                return false;
            }
            args->threads = static_cast<int>(v);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

// One sequential sweep over the whole (K row, V row) x n_kv x n_layers
// buffer. The buffer is laid out fully contiguously in that iteration order
// (row-major: K row then V row per KV position, positions and layers
// concatenated), so a flat linear scan over the underlying storage IS the
// same access pattern as walking (layer, position, row-element) nested
// loops -- and a flat scan is what the compiler can actually
// autovectorize, which matters here: this measures memory bandwidth, not
// compiler codegen, so the loop shape must not become the bottleneck.
// `dk`/`dv`/`n_kv`/`n_layers` are accepted for the caller's byte-accounting
// (see main()); the sweep itself only needs the flat element count.
// Summing every element into `sink` is what prevents the compiler from
// proving the loads are dead and eliding them.
double sweep_once(const float * buf, size_t n_elems, volatile double * sink) {
    const auto start = std::chrono::steady_clock::now();

    // A single running accumulator makes this a latency-bound (serial
    // add-dependency-chain) benchmark, not a bandwidth-bound one -- the CPU
    // cannot issue the next load until the previous add retires. Several
    // independent accumulators break that chain so the loop is limited by
    // how fast the memory system can stream data in, which is the actual
    // quantity this benchmark exists to measure.
    constexpr int lanes      = 8;
    double        acc[lanes] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const size_t n_main = n_elems - (n_elems % lanes);
    for (size_t i = 0; i < n_main; i += lanes) {
        for (int l = 0; l < lanes; ++l) {
            acc[l] += buf[i + l];
        }
    }
    for (size_t i = n_main; i < n_elems; ++i) {
        acc[0] += buf[i];
    }

    const auto end   = std::chrono::steady_clock::now();
    double     total = 0.0;
    for (int l = 0; l < lanes; ++l) {
        total += acc[l];
    }
    *sink = total;
    return std::chrono::duration<double, std::nano>(end - start).count();
}

}  // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        std::fprintf(stderr, "usage: %s [--dk N] [--dv N] [--n-kv N] [--n-layers N] [--iters N] [--threads N]\n",
                     argv[0]);
        return 1;
    }
    if (args.threads < 1) {
        std::fprintf(stderr, "--threads must be >= 1\n");
        return 1;
    }

    const int64_t row_elems       = args.dk + args.dv;
    const int64_t total_elems     = row_elems * args.n_kv * args.n_layers;
    const size_t  total_bytes     = static_cast<size_t>(total_elems) * sizeof(float);
    const size_t  aggregate_bytes = total_bytes * static_cast<size_t>(args.threads);

    std::printf(
        "bench-host-kv-bandwidth: dk=%lld dv=%lld n_kv=%lld n_layers=%lld iters=%d threads=%d "
        "per_thread_bytes=%zu (%.2f MB) aggregate_bytes=%zu (%.2f MB)\n",
        (long long) args.dk, (long long) args.dv, (long long) args.n_kv, (long long) args.n_layers, args.iters,
        args.threads, total_bytes, total_bytes / (1024.0 * 1024.0), aggregate_bytes,
        aggregate_bytes / (1024.0 * 1024.0));

    // One independent buffer per thread -- each thread must stream through
    // its OWN memory, not contend on the same cache lines, to measure
    // memory-controller bandwidth under concurrency rather than cache
    // sharing effects.
    std::vector<std::vector<float>> bufs(args.threads);
    for (auto & buf : bufs) {
        buf.resize(static_cast<size_t>(total_elems));
        // Touch every page once before timing (first-touch/page-fault cost
        // must not land inside a measured iteration).
        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = static_cast<float>(i & 0xff);
        }
    }

    std::vector<double> sinks(args.threads, 0.0);
    std::vector<double> agg_gbps;
    agg_gbps.reserve(args.iters);

    for (int it = 0; it < args.iters; ++it) {
        std::vector<double>      ns_per_thread(args.threads, 0.0);
        std::vector<std::thread> workers;
        workers.reserve(args.threads);
        for (int t = 0; t < args.threads; ++t) {
            workers.emplace_back([&, t] {
                volatile double sink = 0.0;
                ns_per_thread[t]     = sweep_once(bufs[t].data(), bufs[t].size(), &sink);
                sinks[t]             = sink;
            });
        }
        for (auto & w : workers) {
            w.join();
        }
        // Aggregate bandwidth is bounded by the SLOWEST thread's wall time --
        // every thread's bytes must have been read within that window for
        // the run to have completed at all.
        const double slowest_ns = *std::max_element(ns_per_thread.begin(), ns_per_thread.end());
        const double s          = slowest_ns / 1e9;
        agg_gbps.push_back((aggregate_bytes / (1024.0 * 1024.0 * 1024.0)) / s);
    }

    std::sort(agg_gbps.begin(), agg_gbps.end());
    const double gbps_min    = agg_gbps.front();
    const double gbps_max    = agg_gbps.back();
    const double gbps_median = agg_gbps[agg_gbps.size() / 2];

    double sink_sum = 0.0;
    for (double s : sinks) {
        sink_sum += s;
    }

    std::printf("bench-host-kv-bandwidth: aggregate GB/s min=%.2f median=%.2f max=%.2f (sink=%.3f, ignore)\n", gbps_min,
                gbps_median, gbps_max, sink_sum);
    return 0;
}
