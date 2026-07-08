#include "tg_bench_common.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace sycl_mxfp4_moe_bench {

static bool parse_i64(const char * value, int64_t & out) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return false;
    }
    out = static_cast<int64_t>(parsed);
    return true;
}

static bool parse_int(const char * value, int & out) {
    int64_t parsed = 0;
    if (!parse_i64(value, parsed) || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

static bool take_value(int & index, int argc, char ** argv, const char * opt, const char * & value, std::string & error) {
    if (index + 1 >= argc) {
        error = std::string("missing value for ") + opt;
        return false;
    }
    value = argv[++index];
    return true;
}

static bool apply_option_value(const std::string & opt, const char * value, bench_config & cfg, std::string & error) {
    if (opt == "--route") {
        cfg.route = value;
        return true;
    }
    if (opt == "--output-jsonl") {
        cfg.output_jsonl = value;
        return true;
    }
    if (opt == "--ncols") {
        return parse_i64(value, cfg.ncols) || (error = "invalid integer for --ncols", false);
    }
    if (opt == "--hidden") {
        return parse_i64(value, cfg.hidden) || (error = "invalid integer for --hidden", false);
    }
    if (opt == "--topk") {
        return parse_int(value, cfg.topk) || (error = "invalid integer for --topk", false);
    }
    if (opt == "--layers") {
        return parse_int(value, cfg.layers) || (error = "invalid integer for --layers", false);
    }
    if (opt == "--tokens") {
        return parse_int(value, cfg.tokens) || (error = "invalid integer for --tokens", false);
    }
    if (opt == "--warmup") {
        return parse_int(value, cfg.warmup) || (error = "invalid integer for --warmup", false);
    }
    if (opt == "--iters") {
        return parse_int(value, cfg.iters) || (error = "invalid integer for --iters", false);
    }
    error = "unknown option: " + opt;
    return false;
}

bool parse_args(int argc, char ** argv, bench_config & cfg, std::string & error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dry-run") {
            cfg.dry_run = true;
            continue;
        }
        if (arg == "--validate") {
            cfg.validate = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            error = "usage: sycl-mxfp4-moe-bench --route=<route> [--dry-run] [--output-jsonl <path>]";
            return false;
        }

        const size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            const std::string opt = arg.substr(0, eq);
            const std::string value = arg.substr(eq + 1);
            if (!apply_option_value(opt, value.c_str(), cfg, error)) {
                return false;
            }
            continue;
        }

        const char * value = nullptr;
        if (!take_value(i, argc, argv, arg.c_str(), value, error)) {
            return false;
        }
        if (!apply_option_value(arg, value, cfg, error)) {
            return false;
        }
    }

    if (cfg.route.empty()) {
        error = "--route must not be empty";
        return false;
    }
    if (cfg.ncols <= 0 || cfg.hidden <= 0 || cfg.topk <= 0 || cfg.layers <= 0 || cfg.tokens <= 0 || cfg.warmup < 0 || cfg.iters <= 0) {
        error = "shape and iteration values must be positive";
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

bool emit_jsonl(FILE * out, const bench_record & record, std::string & error) {
    if (out == nullptr) {
        error = "output stream is null";
        return false;
    }

    const int written = std::fprintf(out,
        "{\"route\":\"%s\",\"mode\":\"%s\","
        "\"shape\":{\"ncols\":%lld,\"hidden\":%lld,\"topk\":%d,\"layers\":%d,\"tokens\":%d},"
        "\"metrics\":{\"prepack_us\":%.6f,\"compute_us\":%.6f,\"launch_us\":%.6f,\"host_bounce_us\":%.6f,"
        "\"total_gateup_equiv_ms\":%.6f,\"saving_vs_baseline_ms\":%.6f,\"p50_us\":%.6f,\"p90_us\":%.6f,\"p99_us\":%.6f},"
        "\"correct\":{\"max_abs\":%.6f,\"mean_abs\":%.6f,\"rel_l2\":%.6f},"
        "\"fatal\":{\"total\":%d},"
        "\"evidence\":{\"path\":\"%s\",\"dry_run\":%s,\"device\":\"%s\"}}\n",
        json_escape(record.route).c_str(), json_escape(record.mode).c_str(),
        static_cast<long long>(record.ncols), static_cast<long long>(record.hidden), record.topk, record.layers, record.tokens,
        record.prepack_us, record.compute_us, record.launch_us, record.host_bounce_us,
        record.total_gateup_equiv_ms, record.saving_vs_baseline_ms, record.p50_us, record.p90_us, record.p99_us,
        record.max_abs, record.mean_abs, record.rel_l2, record.fatal_total,
        json_escape(record.evidence_path).c_str(), record.dry_run ? "true" : "false", json_escape(record.device).c_str());
    if (written < 0) {
        error = std::string("failed to write JSONL: ") + std::strerror(errno);
        return false;
    }
    return true;
}

bench_record dry_run_record(const bench_config & cfg, const char * route, const char * path, double total_ms) {
    bench_record rec;
    rec.route = route;
    rec.mode = "dry-run";
    rec.ncols = cfg.ncols;
    rec.hidden = cfg.hidden;
    rec.topk = cfg.topk;
    rec.layers = cfg.layers;
    rec.tokens = cfg.tokens;
    rec.total_gateup_equiv_ms = total_ms;
    rec.saving_vs_baseline_ms = 6.0 - total_ms;
    rec.evidence_path = path;
    rec.dry_run = true;
    rec.device = "none";
    return rec;
}

} // namespace sycl_mxfp4_moe_bench
