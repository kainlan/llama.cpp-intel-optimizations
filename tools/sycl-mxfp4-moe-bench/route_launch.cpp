#include "tg_bench_common.hpp"

namespace sycl_mxfp4_moe_bench {

bool run_route_launch(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route launch requires a route-specific implementation before non-dry-run execution";
        return false;
    }
    records.push_back(dry_run_record(cfg, "launch", "launch-dry-run", 2.0));
    return true;
}

} // namespace sycl_mxfp4_moe_bench
