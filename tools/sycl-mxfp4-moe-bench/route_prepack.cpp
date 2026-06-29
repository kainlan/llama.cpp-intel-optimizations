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

bool run_route_prepack(const bench_config & cfg, std::vector<bench_record> & records, std::string & error) {
    if (!cfg.dry_run) {
        error = "route prepack requires a route-specific implementation before non-dry-run execution";
        return false;
    }
    if (!prepack_row_to_i8_vnni32_self_check()) {
        error = "route prepack CPU layout self-check failed";
        return false;
    }

    bench_record rec = dry_run_record(cfg, "prepack", "selected-expert-prepack", 6.0);
    rec.prepack_us = 1200.0;
    rec.compute_us = 0.0;
    rec.p50_us     = 1200.0;
    rec.p90_us     = 1200.0;
    rec.p99_us     = 1200.0;
    records.push_back(rec);
    return true;
}

} // namespace sycl_mxfp4_moe_bench
