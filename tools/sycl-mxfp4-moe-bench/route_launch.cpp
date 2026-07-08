#include "tg_bench_common.hpp"

namespace sycl_mxfp4_moe_bench {

static bench_record make_launch_record(const bench_config & cfg,
                                       const char * mode,
                                       double launch_us,
                                       double total_ms,
                                       double saving_ms) {
    bench_record rec = dry_run_record(cfg, "launch", "launch-reduction", total_ms);
    rec.mode = mode;
    rec.launch_us = launch_us;
    rec.saving_vs_baseline_ms = saving_ms;
    rec.p50_us = launch_us;
    rec.p90_us = launch_us;
    rec.p99_us = launch_us;
    return rec;
}

bool run_route_launch(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route launch non-dry-run is lead-owned because command graph and persistent queue probes can hang";
        return false;
    }

    records.push_back(make_launch_record(cfg, "raw-queue", 2400.0, 2.4, 0.0));
    records.push_back(make_launch_record(cfg, "command-graph", 900.0, 0.9, 1.5));
    records.push_back(make_launch_record(cfg, "persistent-descriptor", 700.0, 0.7, 1.7));
    return true;
}

} // namespace sycl_mxfp4_moe_bench
