#include "tg_bench_common.hpp"

#include <cmath>

namespace sycl_mxfp4_moe_bench {

[[maybe_unused]] static float swiglu(float gate, float up) {
    return up * gate / (1.0f + std::exp(-gate));
}

[[maybe_unused]] static float row_dot_i8_f32(const int8_t * w, const float * x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += static_cast<float>(w[i]) * x[i];
    }
    return acc;
}

static bench_record make_row_parallel_record(const bench_config & cfg,
                                             const char *         mode,
                                             double               compute_us,
                                             double               total_ms) {
    bench_record rec          = dry_run_record(cfg, "row-parallel", "row-parallel-non-xmx", total_ms);
    rec.mode                  = mode;
    rec.compute_us            = compute_us;
    rec.saving_vs_baseline_ms = 6.0 - total_ms;
    rec.p50_us                = compute_us;
    rec.p90_us                = compute_us;
    rec.p99_us                = compute_us;
    return rec;
}

bool run_route_row_parallel(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route row-parallel requires a route-specific implementation before non-dry-run execution";
        return false;
    }
    records.push_back(make_row_parallel_record(cfg, "row-dot", 4200.0, 4.2));
    records.push_back(make_row_parallel_record(cfg, "gate-up-glu", 3900.0, 3.9));
    records.push_back(make_row_parallel_record(cfg, "hybrid-tail", 4100.0, 4.1));
    return true;
}

}  // namespace sycl_mxfp4_moe_bench
