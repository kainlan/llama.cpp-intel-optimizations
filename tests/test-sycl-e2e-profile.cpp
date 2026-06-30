#include "e2e-profile.hpp"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    using namespace ggml_sycl;

    assert(!e2e_tg_profile_enabled_from_env(nullptr));
    assert(!e2e_tg_profile_enabled_from_env(""));
    assert(!e2e_tg_profile_enabled_from_env("0"));
    assert(e2e_tg_profile_enabled_from_env("1"));

    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::MOE), "moe") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::ATTENTION), "attention") == 0);
    assert(std::strcmp(e2e_tg_stage_name(e2e_tg_stage::NON_MOE_MATMUL), "non_moe_matmul") == 0);

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

    e2e_tg_profile_flush_for_tests(stderr);
    e2e_tg_profile_reset_for_tests();
    return 0;
}
