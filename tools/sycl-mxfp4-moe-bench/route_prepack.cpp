#include "tg_bench_common.hpp"

#include <array>
#include <cstddef>

namespace sycl_mxfp4_moe_bench {

static int8_t mxfp4_code_value(uint8_t v) {
    static const int8_t table[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    return table[v & 0x0f];
}

static void prepack_row_to_i8_vnni32(const uint8_t * packed, int8_t * out32) {
    for (int i = 0; i < 16; ++i) {
        const uint8_t byte = packed[i];
        out32[i]      = mxfp4_code_value(byte & 0x0f);
        out32[16 + i] = mxfp4_code_value(byte >> 4);
    }
}

static bool prepack_row_to_i8_vnni32_self_check() {
    const std::array<uint8_t, 16> packed = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    };
    std::array<int8_t, 32> actual = {};
    prepack_row_to_i8_vnni32(packed.data(), actual.data());

    const std::array<int8_t, 32> expected = {
        0, 2, 4, 8, 0, -2, -4, -8, 1, 3, 6, 12, -1, -3, -6, -12,
        1, 3, 6, 12, -1, -3, -6, -12, 0, 2, 4, 8, 0, -2, -4, -8,
    };

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

static bench_record make_prepack_record(const bench_config & cfg,
                                        const char * mode,
                                        double prepack_us,
                                        double compute_us,
                                        double total_ms) {
    bench_record rec = dry_run_record(cfg, "prepack", "selected-expert-prepack", total_ms);
    rec.mode = mode;
    rec.prepack_us = prepack_us;
    rec.compute_us = compute_us;
    rec.saving_vs_baseline_ms = 6.0 - total_ms;
    rec.p50_us = total_ms * 1000.0;
    rec.p90_us = total_ms * 1000.0;
    rec.p99_us = total_ms * 1000.0;
    return rec;
}

bool run_route_prepack(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route prepack requires a route-specific implementation before non-dry-run execution";
        return false;
    }
    if (!prepack_row_to_i8_vnni32_self_check()) {
        error = "route prepack CPU layout self-check failed";
        return false;
    }

    records.push_back(make_prepack_record(cfg, "prepack-only", 1200.0, 0.0, 6.0));
    records.push_back(make_prepack_record(cfg, "compute-only", 0.0, 2500.0, 2.5));
    records.push_back(make_prepack_record(cfg, "cache-combined", 200.0, 2500.0, 2.7));
    return true;
}

} // namespace sycl_mxfp4_moe_bench
