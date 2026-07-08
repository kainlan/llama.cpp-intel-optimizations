#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sycl_mxfp4_moe_bench {

struct bench_config {
    std::string route = "baseline";
    int64_t ncols = 2880;
    int64_t hidden = 2880;
    int topk = 4;
    int layers = 24;
    int tokens = 128;
    int warmup = 10;
    int iters = 100;
    bool validate = false;
    bool dry_run = false;
    std::string output_jsonl;
};

struct bench_record {
    std::string route;
    std::string mode;
    int64_t ncols = 2880;
    int64_t hidden = 2880;
    int topk = 4;
    int layers = 24;
    int tokens = 128;
    double prepack_us = 0.0;
    double compute_us = 0.0;
    double launch_us = 0.0;
    double host_bounce_us = 0.0;
    double total_gateup_equiv_ms = 0.0;
    double saving_vs_baseline_ms = 0.0;
    double p50_us = 0.0;
    double p90_us = 0.0;
    double p99_us = 0.0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rel_l2 = 0.0;
    int fatal_total = 0;
    std::string evidence_path = "none";
    bool dry_run = false;
    std::string device = "none";
};

bool parse_args(int argc, char ** argv, bench_config & cfg, std::string & error);
bool emit_jsonl(FILE * out, const bench_record & record, std::string & error);
bench_record dry_run_record(const bench_config & cfg, const char * route, const char * path, double total_ms);

bool run_route_baseline(const bench_config & cfg, std::vector<bench_record> & records, std::string & error);
bool run_route_prepack(const bench_config & cfg, std::vector<bench_record> & records, std::string & error);
bool run_route_row_parallel(const bench_config & cfg, std::vector<bench_record> & records, std::string & error);
bool run_route_fused_layer(const bench_config & cfg, std::vector<bench_record> & records, std::string & error);
bool run_route_launch(const bench_config & cfg, std::vector<bench_record> & records, std::string & error);
bool run_route_host_bounce(const bench_config & cfg, std::vector<bench_record> & records, std::string & error);

} // namespace sycl_mxfp4_moe_bench
