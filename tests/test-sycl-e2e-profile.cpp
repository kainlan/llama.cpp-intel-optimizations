#include "e2e-profile.hpp"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string read_tmp_file(FILE * file) {
    assert(file != nullptr);
    assert(std::fflush(file) == 0);
    assert(std::fseek(file, 0, SEEK_SET) == 0);

    std::string out;
    char        buffer[512];
    while (true) {
        const size_t n = std::fread(buffer, 1, sizeof(buffer), file);
        out.append(buffer, n);
        if (n < sizeof(buffer)) {
            assert(std::feof(file) != 0);
            break;
        }
    }
    return out;
}

static void assert_contains(const std::string & haystack, const char * needle) {
    assert(haystack.find(needle) != std::string::npos);
}

int main() {
    using namespace ggml_sycl;

    assert(!e2e_tg_profile_enabled_from_env(nullptr));
    assert(!e2e_tg_profile_enabled_from_env(""));
    assert(!e2e_tg_profile_enabled_from_env("0"));
    assert(e2e_tg_profile_enabled_from_env("1"));

    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::DISPATCH), "dispatch") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::CPU_DISPATCH), "cpu_dispatch") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::NON_MOE_MATMUL), "non_moe_matmul") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::MOE), "moe") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::ATTENTION), "attention") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::KV), "kv") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::ELEMENTWISE), "elementwise") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::GRAPH), "graph") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::CACHE), "cache") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::TRANSFER), "transfer") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::OTHER), "other") == 0);

    assert(e2e_tg_stage_from_op(GGML_OP_MUL_MAT_ID, "blk.0.ffn_gate_exps.weight") == e2e_tg_stage::MOE);
    assert(e2e_tg_stage_from_op(GGML_OP_FLASH_ATTN_EXT, "blk.0.attn") == e2e_tg_stage::ATTENTION);
    assert(e2e_tg_stage_from_op(GGML_OP_SET_ROWS, "cache_k_l0") == e2e_tg_stage::KV);
    assert(e2e_tg_stage_from_op(GGML_OP_MUL_MAT, "blk.0.attn_q.weight") == e2e_tg_stage::NON_MOE_MATMUL);
    assert(e2e_tg_stage_from_op(GGML_OP_ADD, "blk.0.ffn_down") == e2e_tg_stage::ELEMENTWISE);

    e2e_tg_profile_reset_for_tests();
    e2e_tg_profile_record(e2e_tg_stage::MOE, "packed-q8-m2", 10.0, 20.0, 64, 2);
    e2e_tg_profile_record(e2e_tg_stage::ATTENTION, "xmx_v2_f16_pp_ncols32", 3.0, 4.0, 128, 1);
    const e2e_tg_profile_snapshot snap = e2e_tg_profile_snapshot_for_tests();
    assert(snap.tokens == 0);
    assert(snap.ops == 3);
    assert(snap.moe_calls == 2);
    assert(snap.stages[static_cast<size_t>(e2e_tg_stage::MOE)].calls == 2);
    assert(snap.stages[static_cast<size_t>(e2e_tg_stage::MOE)].bytes == 64);
    assert(snap.stages[static_cast<size_t>(e2e_tg_stage::ATTENTION)].host_us == 3.0);

    FILE * out = std::tmpfile();
    assert(out != nullptr);
    e2e_tg_profile_flush_for_tests(out);
    const std::string flushed = read_tmp_file(out);
    std::fclose(out);

    assert_contains(flushed, "[SYCL-E2E-TG-PROFILE] tokens=1 ops=3 moe_calls=2 total_host=0.013 ms total_device=0.024 ms\n");
    assert_contains(flushed,
                    "[SYCL-E2E-TG-STAGE] stage=moe calls=2 host=0.010 ms device=0.020 ms bytes=64 "
                    "last_path=packed-q8-m2\n");
    assert_contains(flushed,
                    "[SYCL-E2E-TG-STAGE] stage=attention calls=1 host=0.003 ms device=0.004 ms bytes=128 "
                    "last_path=xmx_v2_f16_pp_ncols32\n");

    e2e_tg_profile_reset_for_tests();
    return 0;
}
