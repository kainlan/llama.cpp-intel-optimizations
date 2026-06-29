#include "tg_bench_common.hpp"

namespace sycl_mxfp4_moe_bench {

bool run_route_prepack(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route prepack requires a route-specific implementation before non-dry-run execution";
        return false;
    }
    records.push_back(dry_run_record(cfg, "prepack", "prepack-dry-run", 6.0));
    return true;
}

} // namespace sycl_mxfp4_moe_bench
