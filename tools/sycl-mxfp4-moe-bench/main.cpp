#include "tg_bench_common.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>

using namespace sycl_mxfp4_moe_bench;

static bool run_selected_route(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (cfg.route == "baseline") {
        return run_route_baseline(cfg, records, error);
    }
    if (cfg.route == "prepack") {
        return run_route_prepack(cfg, records, error);
    }
    if (cfg.route == "row-parallel") {
        return run_route_row_parallel(cfg, records, error);
    }
    if (cfg.route == "fused-layer") {
        return run_route_fused_layer(cfg, records, error);
    }
    if (cfg.route == "launch") {
        return run_route_launch(cfg, records, error);
    }
    if (cfg.route == "host-bounce") {
        return run_route_host_bounce(cfg, records, error);
    }
    error = "unknown route: " + cfg.route;
    return false;
}

int main(int argc, char ** argv) {
    bench_config cfg;
    std::string error;
    if (!parse_args(argc, argv, cfg, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }

    std::vector<bench_record> records;
    if (!run_selected_route(cfg, records, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    if (records.empty()) {
        std::fprintf(stderr, "error: route emitted no records\n");
        return 1;
    }

    FILE * out = stdout;
    std::unique_ptr<FILE, int (*)(FILE *)> file(nullptr, std::fclose);
    if (!cfg.output_jsonl.empty()) {
        FILE * opened = std::fopen(cfg.output_jsonl.c_str(), "w");
        if (opened == nullptr) {
            std::fprintf(stderr, "error: failed to open %s: %s\n", cfg.output_jsonl.c_str(), std::strerror(errno));
            return 1;
        }
        file.reset(opened);
        out = opened;
    }

    for (const bench_record & record : records) {
        if (!emit_jsonl(out, record, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
    }
    if (std::fflush(out) != 0) {
        std::fprintf(stderr, "error: failed to flush output: %s\n", std::strerror(errno));
        return 1;
    }
    return 0;
}
