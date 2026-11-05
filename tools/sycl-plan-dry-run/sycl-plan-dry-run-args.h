// The command line of llama-sycl-plan-dry-run. Header-only and host-only: it includes llama.h for the lazy modes, and
// nothing else, so its host test touches no backend, device or model. -c is kept as given; the tool maps it to the
// load's hint with common_n_ctx_hint, as every tool that takes -c does (llama.cpp-h187).
#pragma once

#include "llama.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// The devices --device-budget can name; the tool checks it equals the export's GGML_SYCL_PLAN_DRY_RUN_MAX_DEVICES.
constexpr uint32_t SYCL_PLAN_DRY_RUN_ARGS_MAX_DEVICES = 16;

struct sycl_plan_dry_run_args {
    std::string          model;                        // -m: the GGUF file (the first split of a split model)
    bool                 mem_available_set   = false;  // --mem-available given
    uint64_t             mem_available_bytes = 0;      // the sample's MemAvailable in place of the live reading
    uint32_t             n_device_budgets    = 0;      // device_budget_bytes[0..n) are read; 0 keeps a device's own
    uint64_t             device_budget_bytes[SYCL_PLAN_DRY_RUN_ARGS_MAX_DEVICES] = {};
    int32_t              n_ctx         = 0;            // -c as given: 0 (the default) names the model's
    bool                 lazy_mode_set = false;        // -lzm given
    enum llama_lazy_mode lazy_mode     = LLAMA_LAZY_MODE_AUTO;
    bool                 help          = false;
};

// A count of GiB, fractions allowed, written as digits with at most one '.': no sign, no exponent, no suffix. True
// with `bytes` set for a value above 0 whose bytes fit in 64 bits.
inline bool sycl_plan_dry_run_parse_gib(const char * text, uint64_t & bytes) {
    size_t digits = 0, dots = 0;
    for (const char * c = text; *c != '\0'; ++c) {
        if (*c >= '0' && *c <= '9') {
            ++digits;
        } else if (*c == '.') {
            ++dots;
        } else {
            return false;
        }
    }
    if (digits == 0 || dots > 1) {
        return false;
    }
    const double gib     = std::strtod(text, nullptr);
    const double in_byte = gib * 1024.0 * 1024.0 * 1024.0;
    if (!(gib > 0.0) || !(in_byte < 18446744073709551616.0) || in_byte < 1.0) {
        return false;
    }
    bytes = static_cast<uint64_t>(in_byte);
    return true;
}

// A non-negative decimal integer of at most `max`, digits only.
inline bool sycl_plan_dry_run_parse_uint(const char * text, uint64_t max, uint64_t & value) {
    if (*text == '\0' || std::strlen(text) > 19) {
        return false;
    }
    uint64_t v = 0;
    for (const char * c = text; *c != '\0'; ++c) {
        if (*c < '0' || *c > '9') {
            return false;
        }
        v = v * 10 + static_cast<uint64_t>(*c - '0');
    }
    if (v > max) {
        return false;
    }
    value = v;
    return true;
}

// Parses the command line into `args`. False with `err` naming the problem for an unknown option, a missing or
// malformed value, or no -m without -h. Every value is checked; none falls back to a default.
inline bool sycl_plan_dry_run_parse_args(int                      argc,
                                         const char * const *     argv,
                                         sycl_plan_dry_run_args & args,
                                         std::string &            err) {
    args = sycl_plan_dry_run_args{};
    for (int i = 1; i < argc; ++i) {
        const std::string opt   = argv[i];
        const auto        value = [&](const char *& out) {
            if (i + 1 >= argc) {
                err = opt + " needs a value";
                return false;
            }
            out = argv[++i];
            return true;
        };
        const char * v = nullptr;
        if (opt == "-h" || opt == "--help") {
            args.help = true;
        } else if (opt == "-m" || opt == "--model") {
            if (!value(v)) {
                return false;
            }
            args.model = v;
        } else if (opt == "--mem-available") {
            if (!value(v)) {
                return false;
            }
            if (!sycl_plan_dry_run_parse_gib(v, args.mem_available_bytes)) {
                err = "--mem-available takes a count of GiB above 0, got '" + std::string(v) + "'";
                return false;
            }
            args.mem_available_set = true;
        } else if (opt == "--device-budget") {
            if (!value(v)) {
                return false;
            }
            const char * eq     = std::strchr(v, '=');
            uint64_t     device = 0;
            uint64_t     bytes  = 0;
            if (eq == nullptr ||
                !sycl_plan_dry_run_parse_uint(std::string(v, eq).c_str(), SYCL_PLAN_DRY_RUN_ARGS_MAX_DEVICES - 1,
                                              device) ||
                !sycl_plan_dry_run_parse_gib(eq + 1, bytes)) {
                err = "--device-budget takes D=GiB with D from 0 to " +
                      std::to_string(SYCL_PLAN_DRY_RUN_ARGS_MAX_DEVICES - 1) + " and GiB above 0, got '" +
                      std::string(v) + "'";
                return false;
            }
            args.device_budget_bytes[device] = bytes;
            if (device + 1 > args.n_device_budgets) {
                args.n_device_budgets = static_cast<uint32_t>(device + 1);
            }
        } else if (opt == "-c" || opt == "--ctx-size") {
            if (!value(v)) {
                return false;
            }
            uint64_t n_ctx = 0;
            if (!sycl_plan_dry_run_parse_uint(v, INT32_MAX, n_ctx)) {
                err = "-c takes a context length (0: the model's default), got '" + std::string(v) + "'";
                return false;
            }
            args.n_ctx = static_cast<int32_t>(n_ctx);
        } else if (opt == "-lzm" || opt == "--lazy-mode") {
            if (!value(v)) {
                return false;
            }
            const std::string mode = v;
            if (mode == "on") {
                args.lazy_mode = LLAMA_LAZY_MODE_ON;
            } else if (mode == "auto") {
                args.lazy_mode = LLAMA_LAZY_MODE_AUTO;
            } else if (mode == "off") {
                args.lazy_mode = LLAMA_LAZY_MODE_OFF;
            } else {
                err = opt + " takes on, auto or off, got '" + mode + "'";
                return false;
            }
            args.lazy_mode_set = true;
        } else {
            err = "unknown option '" + opt + "'";
            return false;
        }
    }
    if (!args.help && args.model.empty()) {
        err = "no model: pass -m <file.gguf>";
        return false;
    }
    return true;
}
