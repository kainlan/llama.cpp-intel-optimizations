#include "tg_bench_common.hpp"

namespace sycl_mxfp4_moe_bench {

bool run_route_baseline(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route baseline non-dry-run requires lead-owned SYCL execution";
        return false;
    }

    bench_record rec = dry_run_record(cfg, "baseline", "packed-q8-m2", 6.0);
    rec.compute_us = 6000.0;
    rec.p50_us = 6000.0;
    rec.p90_us = 6000.0;
    rec.p99_us = 6000.0;
    rec.saving_vs_baseline_ms = 0.0;
    records.push_back(rec);
    return true;
}

} // namespace sycl_mxfp4_moe_bench
