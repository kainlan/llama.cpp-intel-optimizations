#include "tg_bench_common.hpp"

namespace sycl_mxfp4_moe_bench {

static bench_record make_fused_layer_record(const bench_config & cfg,
                                            const char *         mode,
                                            double               total_ms,
                                            double               saving_ms,
                                            double               max_abs,
                                            double               mean_abs,
                                            double               rel_l2) {
    bench_record rec          = dry_run_record(cfg, "fused-layer", "fused-gateup-glu-down", total_ms);
    rec.mode                  = mode;
    rec.compute_us            = total_ms * 1000.0;
    rec.saving_vs_baseline_ms = saving_ms;
    rec.p50_us                = rec.compute_us;
    rec.p90_us                = rec.compute_us;
    rec.p99_us                = rec.compute_us;
    rec.max_abs               = max_abs;
    rec.mean_abs              = mean_abs;
    rec.rel_l2                = rel_l2;
    return rec;
}

bool run_route_fused_layer(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route fused-layer requires a route-specific implementation before non-dry-run execution";
        return false;
    }

    records.push_back(make_fused_layer_record(cfg, "fp32-accum", 2.2, 3.8, 0.00001, 0.000001, 0.000001));
    records.push_back(make_fused_layer_record(cfg, "fp16-accum", 2.0, 4.0, 0.0002, 0.00002, 0.00002));
    records.push_back(make_fused_layer_record(cfg, "quant-accum", 1.8, 4.2, 0.0005, 0.00005, 0.00005));
    return true;
}

}  // namespace sycl_mxfp4_moe_bench
