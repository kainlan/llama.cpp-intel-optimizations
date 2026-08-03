// Test: infer_tensor_usage() (ggml-sycl/common.hpp) and
// expert_tensor_role_from_tensor_name() (ggml-sycl/unified-cache.hpp) correctly
// classify expert vs non-expert tensors by name, including grovemoe's chunked
// "_chexps" family.
//
// This replaced a self-contained mock (llama.cpp-u2mz): the original test
// defined its own tensor_type enum and its own is_expert_tensor(), then
// asserted that reimplementation against itself. It never called either real
// classifier, so it stayed green the whole time the real infer_tensor_usage()
// did not recognize grovemoe's ffn_*_chexps tensors as expert weights --
// exactly the property this test's name claims to cover. That bug was found
// by a SIGABRT in an unrelated sweep, months later. A mutation that removes
// the "_chexps" handling from either real classifier must fail this test;
// removing it from the OLD mock proved nothing, because nothing here called
// production code.
//
// CPU-only by construction: both functions under test are pure string
// classifiers (strstr/strcmp over a tensor name) -- no device, queue, or
// cache is touched.

#include "common.hpp"

#include <cstdio>
#include <cstring>

// expert_tensor_role and its helpers live in namespace ggml_sycl
// (unified-cache.hpp:43), while tensor_usage / infer_tensor_usage resolve
// unqualified. Pulling the namespace in keeps the assertions readable rather
// than qualifying thirteen call sites.
using namespace ggml_sycl;

namespace {

int n_pass = 0;
int n_fail = 0;

void check_usage(const char * name, tensor_usage expected) {
    const tensor_usage got = infer_tensor_usage(name);
    if (got == expected) {
        n_pass++;
    } else {
        printf("FAIL usage(%s): expected %d, got %d\n", name, (int) expected, (int) got);
        n_fail++;
    }
}

void check_role(const char * name, expert_tensor_role expected) {
    const expert_tensor_role got = expert_tensor_role_from_tensor_name(name);
    if (got == expected) {
        n_pass++;
    } else {
        printf("FAIL role(%s): expected %s, got %s\n", name, expert_tensor_role_name(expected),
               expert_tensor_role_name(got));
        n_fail++;
    }
}

}  // namespace

int main() {
    // Lifecycle snapshots own one full plan per model, not one summary per
    // device cache. This also verifies explicit no-plan publication and plan
    // deletion without requiring a GPU queue.
    placement_plan plan{};
    plan.weight_host_bytes = 11;
    plan.weight_vram_bytes = 22;
    plan.multi_device      = true;
    plan.devices           = { 0, 1, 2 };
    lifecycle_stage_placement_plan(1001, plan);
    if (!lifecycle_publish_placement_plan(2001, 1001, 3, 7, 0, 0, false)) {
        printf("FAIL lifecycle plan publish\n");
        n_fail++;
    } else {
        auto published = lifecycle_find_placement_plan(2001, 1001);
        if (published && published->planned_host_bytes == 11 && published->actual_host_bytes == 11 &&
            published->verdict == lifecycle_plan_verdict::MIXED) {
            n_pass++;
        } else {
            printf("FAIL lifecycle multi-device single count\n");
            n_fail++;
        }
    }
    lifecycle_stage_no_placement_plan(1002);
    if (!lifecycle_publish_placement_plan(2002, 1002, 4, 8, 99, 99, true)) {
        printf("FAIL lifecycle no-plan publish\n");
        n_fail++;
    } else {
        auto no_plan = lifecycle_find_placement_plan(2002, 1002);
        auto prior   = lifecycle_find_placement_plan(2001, 1001);
        if (no_plan && no_plan->explicit_no_plan && !no_plan->plan &&
            no_plan->verdict == lifecycle_plan_verdict::UNKNOWN && no_plan->planned_host_bytes == 0 && prior) {
            n_pass++;
        } else {
            printf("FAIL lifecycle no-plan isolation\n");
            n_fail++;
        }
    }
    lifecycle_erase_placement_plan(2002, 1002);
    lifecycle_erase_placement_plan(2001, 1001);
    if (lifecycle_published_placement_plan_count_for_test() == 0) {
        n_pass++;
    } else {
        printf("FAIL lifecycle plan deletion\n");
        n_fail++;
    }

    // Non-expert tensors -- must NOT classify as MOE_EXPERT_WEIGHT / a routed role.
    check_usage("blk.0.attn_q.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.attn_k.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.attn_v.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.attn_output.weight", tensor_usage::ATTENTION_WEIGHT);
    check_usage("blk.0.ffn_norm.weight", tensor_usage::NORM);
    check_usage("output_norm.weight", tensor_usage::NORM);
    check_usage("token_embd.weight", tensor_usage::EMBEDDING);
    // Router gate is NOT an expert weight -- it must not collapse onto the
    // "_exps" family it routes to.
    check_usage("blk.0.ffn_gate_inp.weight", tensor_usage::MOE_GATE);

    check_role("blk.0.attn_q.weight", expert_tensor_role::UNKNOWN);
    check_role("blk.0.ffn_gate_inp.weight", expert_tensor_role::UNKNOWN);

    // Plain MoE expert trio -- must classify as expert weights with distinct roles.
    check_usage("blk.0.ffn_gate_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_down_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_up_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_role("blk.0.ffn_gate_exps.weight", expert_tensor_role::GATE);
    check_role("blk.0.ffn_down_exps.weight", expert_tensor_role::DOWN);
    check_role("blk.0.ffn_up_exps.weight", expert_tensor_role::UP);

    // Fused gate+up expert tensor -- has its own literal (does not contain
    // "ffn_gate_exps" or "ffn_up_exps").
    check_usage("blk.0.ffn_gate_up_exps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_role("blk.0.ffn_gate_up_exps.weight", expert_tensor_role::GATE_UP);

    // grovemoe's chunked-expert trio -- the property that broke in production.
    // "ffn_gate_chexps" contains neither "ffn_gate_exps" nor a bare "_exps"
    // substring, so it needs its own literal in BOTH classifiers.
    check_usage("blk.0.ffn_gate_chexps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_down_chexps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_usage("blk.0.ffn_up_chexps.weight", tensor_usage::MOE_EXPERT_WEIGHT);
    check_role("blk.0.ffn_gate_chexps.weight", expert_tensor_role::CHUNK_GATE);
    check_role("blk.0.ffn_down_chexps.weight", expert_tensor_role::CHUNK_DOWN);
    check_role("blk.0.ffn_up_chexps.weight", expert_tensor_role::CHUNK_UP);

    // arctic's ffn_norm_exps is a 1D norm consumed by build_norm, NOT a routed
    // expert weight -- must not be swept up by a bare "_exps" substring match
    // into MOE_EXPERT_WEIGHT (it is not one of the specific "_exps"/"_chexps"
    // literals either classifier matches on, so it falls through to the
    // ordinary "_norm" rule).
    check_usage("blk.0.ffn_norm_exps.weight", tensor_usage::NORM);
    check_role("blk.0.ffn_norm_exps.weight", expert_tensor_role::UNKNOWN);

    // Plain (non-MoE) dense FFN weight -- the ordinary MUL_MAT path.
    check_usage("blk.0.ffn_gate.weight", tensor_usage::FFN_WEIGHT);
    check_usage("blk.0.ffn_up.weight", tensor_usage::FFN_WEIGHT);
    check_usage("blk.0.ffn_down.weight", tensor_usage::FFN_WEIGHT);

    // The dense shared-expert FFN ("_shexp") goes through ordinary MUL_MAT,
    // not MUL_MAT_ID -- must not be swept up by a bare "exps"/"gate" substring
    // match into either the plain-FFN or expert classification.
    check_usage("blk.0.ffn_gate_shexp.weight", tensor_usage::UNKNOWN);
    check_role("blk.0.ffn_gate_shexp.weight", expert_tensor_role::UNKNOWN);

    printf("\n%d/%d tests passed\n", n_pass, n_pass + n_fail);
    if (n_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
