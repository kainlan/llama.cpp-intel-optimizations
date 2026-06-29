#include "tg_bench_common.hpp"

namespace sycl_mxfp4_moe_bench {

bool run_route_host_bounce(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route host-bounce non-dry-run is lead-owned and host-bounce-only; direct P2P is forbidden";
        return false;
    }

    bench_record copy_only = dry_run_record(cfg, "host-bounce", "host-bounce-no-p2p", 0.5);
    copy_only.mode = "copy-only";
    copy_only.host_bounce_us = 500.0;
    copy_only.saving_vs_baseline_ms = 0.0;
    records.push_back(copy_only);

    bench_record remote_expert = dry_run_record(cfg, "host-bounce", "host-bounce-no-p2p", 3.2);
    remote_expert.mode = "remote-expert";
    remote_expert.host_bounce_us = 700.0;
    remote_expert.compute_us = 2500.0;
    remote_expert.saving_vs_baseline_ms = 2.8;
    records.push_back(remote_expert);

    bench_record overlap_sim = dry_run_record(cfg, "host-bounce", "host-bounce-no-p2p", 2.1);
    overlap_sim.mode = "overlap-sim";
    overlap_sim.host_bounce_us = 700.0;
    overlap_sim.compute_us = 2500.0;
    overlap_sim.saving_vs_baseline_ms = 3.9;
    records.push_back(overlap_sim);

    return true;
}

} // namespace sycl_mxfp4_moe_bench
