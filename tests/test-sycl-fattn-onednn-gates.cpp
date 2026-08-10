// Host-only gate tests for the oneDNN SDPA layout planner. No GPU, no
// allocation, no oneDNN runtime -- it calls the planner directly.
//
// This file has been lost once, and its coverage failed once. Both are worth
// knowing before adding to it (llama.cpp-4jlv):
//
//  1. It WAS registered, in 3c8f296fd (tests/CMakeLists.txt:990-1002), and it
//     built and passed -- artifacts/kkxtv7-5/ctest-materialization.log records
//     "Test #100: test-sycl-fattn-onednn-gates ... Passed 0.11 sec". The
//     registration is present at 3c8f296fd and absent at HEAD, and nothing
//     announced its removal; the passing log was left behind as a fossil.
//
//     What is NOT established is HOW it was lost, and an earlier version of
//     this comment overstated it. Naming a culprit commit by "first commit in
//     3c8f296fd..HEAD whose tests/CMakeLists.txt lacks the string" does not
//     work: that traversal is topological, so on an integrated upstream line
//     it returns the first upstream commit to TOUCH the file, not one that
//     removed anything. It yields d3dce4e0a -- but d3dce4e0a's parent
//     4974bf53c does not have the registration either, so d3dce4e0a plainly
//     did not drop it, and it is a single-parent commit rather than the merge
//     the earlier text claimed. The loss is a consequence of integrating an
//     upstream line whose copy of this file never carried the fork's
//     registration; rebase vs merge-resolution vs manual overwrite is not
//     distinguished, and does not change the remedy.
//
//     (Recorded because the wrong version was a clean answer about the wrong
//     thing -- the same defect class this file's own gate exists to catch.)
//
//  2. Registration alone would NOT have caught the phi2 abort. Every case here
//     inherited `params.scale = 1.0f / 11.313708f` from mistral_like_params --
//     that is 1/sqrt(128), a CONFORMING scale -- and no case varied it. So all
//     11 cases passed on 2026-05-15 while the phi2 dispatch bug had already
//     been present since be45709a9 (2026-04-21). The planner had a test, the
//     test ran, and the scale axis simply was not in it.
//
// The lesson for anyone extending this file: a planner gate is only covered if
// some case makes the guarded quantity WRONG. A suite of cases that all supply
// well-formed inputs proves the accept path and nothing else.
//
// ---------------------------------------------------------------------------
// THREE-STATE CONTRACT: EVERY STATE MUST EXIT rc=0. There are no by-design
// failures in this file. Any FAIL line, in any state, is a real finding.
//
// Five cases below depend on GGML_SYCL_FA_ONEDNN_MATERIALIZE, because they
// exercise the nc!=D GQA/MQA shapes that variable routes. They read the state
// and assert the plan that state owes:
//
//   unset -> DIRECT expected               -> PASS
//   =0    -> DIRECT expected               -> PASS
//   =1    -> MATERIALIZE_REQUIRED expected -> PASS
//
// ---- Polarity history, because it inverted once ---------------------------
// The variable arrived with llama.cpp-l7rt defaulting ON, as a measurement axis
// only, and this header used to say "=0 makes four cases fail BY DESIGN" and
// name them. That was correct then and is the exact opposite of the contract
// above now: the owner ruled on 2026-08-10 (llama.cpp-olpg) to ship DIRECT as
// the default. Unset now means DIRECT; =1 is the opt-in and the surviving A/B
// axis. Treat any surviving "default ON" text as predating that ruling.
//
// Ending the by-design-red era is the point, not a side effect. A state that is
// already failing cannot detect anything, so the =0 run used to be worth
// nothing as a gate. Now the THREE-STATE RUN IS THE MUTATION DETECTOR, and it
// needs all three to be green-when-correct: =1 catches a planner stuck on
// DIRECT, unset/=0 catch one stuck on MATERIALIZE_REQUIRED. Either state alone
// only proves one direction.
//
// ---- Why no case sets the variable itself ---------------------------------
// The planner latches it into a function-local static on its first call
// (fattn-onednn.cpp), so it is read ONCE PER PROCESS. A setenv() inside a case
// cannot move a plan a previous case already latched, and would silently
// override the state the runner asked for. Each polarity needs its own process
// -- which is why this is a three-state contract and not one self-contained
// test.
//
// test_nc_stride_axis_moves_plan_iff_materialize_enabled is the in-process
// substitute. It cannot switch states, but it checks that within the active
// state the nc-stride axis has exactly the effect that state implies, which is
// what separates "the predicate answered" from "the predicate is dead and this
// shape was going to answer that anyway".
//
// test_source_contract_partition_split_is_refused (llama.cpp-mrld) is the one
// case here that is not a planner call: it reads no environment variable and
// does not invoke the planner, so it answers the same in all three states. If
// it goes red it is a real finding in every state.
// ---------------------------------------------------------------------------

#include "ggml-sycl/fattn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    // 77 (ctest SKIP_RETURN_CODE), not 0: no gate was evaluated in this build,
    // so this must not read as a pass.  A green skip is the exact defect class
    // this file's own comment is about (llama.cpp-ay8c).
    std::fprintf(stderr, "SKIP: GGML SYCL oneDNN not enabled; no gate was evaluated.\n");
    return 77;
}
#else

#    define TEST_ASSERT(cond, msg)                       \
        do {                                             \
            if (!(cond)) {                               \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                            \
            }                                            \
        } while (0)

// The shipped default (llama.cpp-olpg, owner ruling 2026-08-10): with the
// variable unset, the planner plans DIRECT for nc!=D GQA/MQA.
//
// This constant is not a convenience. It IS this file's assertion about the
// product default, written out rather than inferred, so that changing the
// default in fattn-onednn.cpp without touching this line turns the unset run
// red. That redness is the feature -- it is how a silent policy change gets
// caught. If you arrive at a failing unset run, the fix is to establish which
// side is wrong, NOT to edit this constant until the run goes green; doing that
// retires the gate while leaving it looking like it still works.
static constexpr bool k_materialize_default = false;

// Mirrors the planner's own parse (getenv, atoi != 0, else the default above).
// Keep the two spellings identical: a divergence here does not fail loudly, it
// just makes every state-dependent case below assert against a state the
// planner is not in.
static bool materialize_enabled() {
    const char * e = std::getenv("GGML_SYCL_FA_ONEDNN_MATERIALIZE");
    return e ? (std::atoi(e) != 0) : k_materialize_default;
}

static fattn_params mistral_like_params(int k_nc_stride_elems) {
    fattn_params params{};
    params.Q_type    = GGML_TYPE_F16;
    params.K_type    = GGML_TYPE_F16;
    params.V_type    = GGML_TYPE_F16;
    params.mask_type = GGML_TYPE_F16;
    params.scale     = 1.0f / 11.313708f;
    params.ne00      = 128;  // D
    params.ne01      = 512;  // prompt tokens
    params.ne02      = 32;   // Q heads
    params.ne03      = 1;
    params.ne10      = 128;
    params.ne11      = 512;  // KV tokens
    params.ne12      = 8;    // KV heads, so GQA
    params.ne13      = 1;
    params.nb01      = params.ne00 * (int) sizeof(sycl::half);
    params.nb02      = params.nb01 * params.ne01;
    params.nb03      = params.nb02 * params.ne02;
    params.nb11      = k_nc_stride_elems * (int) sizeof(sycl::half);
    params.nb12      = params.nb11 * params.ne11;
    params.nb13      = (int64_t) params.nb12 * params.ne12;
    params.nb21      = params.ne10 * (int) sizeof(sycl::half);
    params.nb22      = params.nb21 * params.ne11;
    params.nb23      = (int64_t) params.nb22 * params.ne12;
    params.prec      = GGML_PREC_F32;
    return params;
}

static fattn_params mha_like_params() {
    fattn_params params = mistral_like_params(/*k_nc_stride_elems=*/128);
    params.ne02         = 32;
    params.ne12         = 32;
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    return params;
}

static fattn_params mqa_like_params(int k_nc_stride_elems) {
    fattn_params params = mistral_like_params(k_nc_stride_elems);
    params.ne02         = 32;
    params.ne12         = 1;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    return params;
}

// ggml_sycl_flash_attn_ext_onednn_eligible is the boolean face of the planner,
// and it must answer "DIRECT-eligible", not merely "not rejected". The two
// differ on exactly one kind -- MATERIALIZE_REQUIRED, which is not a rejection
// and is not direct-eligible either -- and whether the nc_stride != D GQA shape
// lands there is precisely what GGML_SYCL_FA_ONEDNN_MATERIALIZE selects. So
// this case tracks the state rather than asserting one answer.
//
// This case used to wrap the call in setenv("GGML_SYCL_FA_ONEDNN_ALLOW", "1")
// and assert the shape stayed ineligible *even with* the bypass set. That leg
// was vacuous — 3c8f296fd removed that getenv, so it was the unset call made
// twice and could not fail differently. Repointing it at the real variable
// GGML_SYCL_FA_ONEDNN would be vacuous the same way: that name gates
// g_sycl_fa_onednn_enabled in fattn.cpp, a file-scope flag the planner cannot
// see (documented at fattn-onednn.hpp's declaration) and which latches once per
// process. The planner reads exactly two environment variables, each cached in
// a function-local static: GGML_SYCL_FA_ONEDNN_MIN_NCOLS, and
// GGML_SYCL_FA_ONEDNN_MATERIALIZE (added by llama.cpp-l7rt).
//
// That second one is the exact opposite of the vacuous leg described above, and
// the contrast is the point: the old ALLOW bypass could not change this call's
// answer, whereas MATERIALIZE changes it every time. This case still does not
// set it -- but for a third reason again, now that the default is DIRECT. It
// reads the state and asserts whichever answer that state owes. Setting it here
// would pin one polarity and leave the other two states untested, and it would
// not even take effect, because the planner latches the variable once per
// process (see the header).
static bool test_gqa_nc_stride_mismatch_eligibility_follows_materialize_state() {
    fattn_params params   = mistral_like_params(/*k_nc_stride_elems=*/512);
    const bool   eligible = ggml_sycl_flash_attn_ext_onednn_eligible(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                     /*multi_seq=*/false);

    if (materialize_enabled()) {
        TEST_ASSERT(!eligible,
                    "MATERIALIZE=1: nc_stride != D GQA plans MATERIALIZE_REQUIRED and must not report "
                    "direct-eligible");
        return true;
    }
    TEST_ASSERT(eligible, "MATERIALIZE default/=0: nc_stride != D GQA plans DIRECT and must report direct-eligible");
    return true;
}

// In-process state-dependence control, and the reason every state can be green
// without any of them becoming vacuous. A case that asserts only "the active
// state's expected answer" cannot distinguish the predicate answering from the
// predicate being dead on a shape that would have answered that way regardless.
//
// The discriminator is the nc-stride axis at one fixed GQA shape: with
// materialization on, the dense and strided variants plan DIFFERENTLY; with it
// off they plan IDENTICALLY. A predicate stuck in either position therefore
// fails one of the two states. This is the host-side analogue of llama.cpp-l7rt
// R0's perfect complement (64 of 64 Mistral dispatches moved), which needed a
// GPU and two runs to establish the same thing.
static bool test_nc_stride_axis_moves_plan_iff_materialize_enabled() {
    fattn_params dense   = mistral_like_params(/*k_nc_stride_elems=*/128);
    fattn_params strided = mistral_like_params(/*k_nc_stride_elems=*/512);

    const auto plan_dense   = ggml_sycl_flash_attn_ext_onednn_plan(dense, dense.ne02, dense.ne12, dense.kv_is_fp8,
                                                                   /*multi_seq=*/false);
    const auto plan_strided = ggml_sycl_flash_attn_ext_onednn_plan(strided, strided.ne02, strided.ne12,
                                                                   strided.kv_is_fp8, /*multi_seq=*/false);

    // Anchor: if the dense control ever stops planning DIRECT, the comparison
    // below is between two unknowns and proves nothing either way.
    TEST_ASSERT(plan_dense.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "the dense-stride anchor must plan DIRECT in every state, or this differential is meaningless");

    if (materialize_enabled()) {
        TEST_ASSERT(plan_strided.kind != plan_dense.kind,
                    "MATERIALIZE=1: the nc-stride axis must move the plan; identical kinds mean the materialize "
                    "predicate never fired");
        return true;
    }
    TEST_ASSERT(plan_strided.kind == plan_dense.kind,
                "MATERIALIZE default/=0: the nc-stride axis must NOT move the plan; a difference means the "
                "materialize predicate fired with the toggle off");
    return true;
}

static bool test_gqa_nc_stride_equal_d_remains_onednn_eligible() {
    fattn_params params   = mistral_like_params(/*k_nc_stride_elems=*/128);
    const bool   eligible = ggml_sycl_flash_attn_ext_onednn_eligible(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                     /*multi_seq=*/false);

    TEST_ASSERT(eligible, "GQA nc_stride == D shape should remain oneDNN eligible");
    return true;
}

static bool test_planner_direct_mha_contiguous() {
    fattn_params params = mha_like_params();
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT, "contiguous MHA should use direct oneDNN layout");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK, "contiguous MHA should have OK reason");
    return true;
}

// Both `reason` assertions below are load-bearing in BOTH states, which they
// were not before the flip: the old file asserted only the materialize reason,
// and the DIRECT leg's `reason == OK` went unchecked because TEST_ASSERT
// returned on the kind assertion above it first.
static bool test_planner_gqa_mismatch_plan_follows_materialize_state() {
    fattn_params params = mistral_like_params(/*k_nc_stride_elems=*/512);
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    if (materialize_enabled()) {
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED,
                    "MATERIALIZE=1: GQA nc_stride != D should require materialization before oneDNN");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH,
                    "MATERIALIZE=1: GQA materialization reason should identify K/V nc stride mismatch");
        return true;
    }
    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "MATERIALIZE default/=0: GQA nc_stride != D should plan DIRECT");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK,
                "MATERIALIZE default/=0: a DIRECT plan for GQA nc_stride != D should report reason OK");
    return true;
}

static bool test_planner_mqa_mismatch_plan_follows_materialize_state() {
    fattn_params params = mqa_like_params(/*k_nc_stride_elems=*/512);
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    if (materialize_enabled()) {
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED,
                    "MATERIALIZE=1: MQA nc_stride != D should require materialization before oneDNN");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH,
                    "MATERIALIZE=1: MQA materialization reason should identify K/V nc stride mismatch");
        return true;
    }
    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "MATERIALIZE default/=0: MQA nc_stride != D should plan DIRECT");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK,
                "MATERIALIZE default/=0: a DIRECT plan for MQA nc_stride != D should report reason OK");
    return true;
}

// phi2 pre-scales Q by 1/sqrt(n_embd_head) inside the graph and then hands
// build_attn a kq_scale of 1.0 (src/models/phi2.cpp), so params.scale is 1.0
// rather than 1/sqrt(D). D=80 is phi2's head dim (n_embd 2560 / n_head 32).
static fattn_params phi2_like_params() {
    fattn_params params = mha_like_params();
    params.ne00         = 80;
    params.ne10         = 80;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb02         = params.nb01 * params.ne01;
    params.nb03         = params.nb02 * params.ne02;
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    params.scale        = 1.0f;  // NOT 1/sqrt(80) — the whole point of the arch
    return params;
}

// The oneDNN partition bakes sqrt(D) in as the softmax divisor, so a model whose
// scale is not 1/sqrt(D) must never reach it. The execute path asserts this; the
// planner has to reject first or that assert aborts the process.
static bool test_planner_rejects_scale_not_inv_sqrt_d() {
    fattn_params params = phi2_like_params();
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "params.scale != 1/sqrt(D) (phi2) must reject oneDNN, not abort at execute time");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                "scale reject reason should be explicit");
    return true;
}

// Same shape family, but GQA with a non-dense K/V stride — the plan kind that
// would otherwise be MATERIALIZE_REQUIRED. The scale gate must win over it,
// because the execute-time assert sits AFTER materialization.
static bool test_planner_rejects_scale_before_materialization() {
    fattn_params params = phi2_like_params();
    params.ne12         = 8;                                           // GQA
    params.nb11         = 4 * params.ne10 * (int) sizeof(sycl::half);  // nc_stride != D
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "a bad scale must reject even on the materialization path");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                "scale gate must win over KV_NC_STRIDE_MISMATCH");
    return true;
}

// Guard the other direction: the scale gate must not reject a legitimate
// 1/sqrt(D) at a D the existing cases do not cover, or it silently costs
// throughput on every such model.
static bool test_planner_accepts_inv_sqrt_d_at_odd_d() {
    fattn_params params = phi2_like_params();
    params.scale        = 1.0f / std::sqrt(80.0f);
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "a legitimate 1/sqrt(D) scale at D=80 must still reach oneDNN");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK, "1/sqrt(D) at D=80 should have OK reason");
    return true;
}

// The materialization descriptor is a second entry point into the planner; a
// rejected plan must not hand back a descriptor that invites the caller in.
static bool test_materialization_descriptor_rejects_bad_scale() {
    fattn_params                             params = phi2_like_params();
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(!ok, "materializer must refuse a shape the planner rejected on scale");
    return true;
}

static bool test_planner_rejects_unsupported_d() {
    fattn_params params = mha_like_params();
    params.ne00         = 1024;
    params.ne10         = 1024;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT, "D > 512 should reject oneDNN");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::UNSUPPORTED_D,
                "unsupported D reject reason should be explicit");
    return true;
}

static bool test_planner_rejects_unproven_batch() {
    fattn_params params = mha_like_params();
    params.ne03         = 2;
    params.ne13         = 2;
    params.nb03         = params.nb02 * params.ne02;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "batch>1 tensors must reject oneDNN until batch descriptors/materialization are proven");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::BATCH_UNSUPPORTED,
                "batch>1 reject reason should be explicit");
    return true;
}

static bool test_planner_rejects_paged_layout() {
    fattn_params params     = mistral_like_params(/*k_nc_stride_elems=*/128);
    params.use_paged_attn   = true;
    params.use_paged_layout = true;
    const auto plan         = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                   /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "paged K/V layouts must reject oneDNN until descriptor support is proven");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::PAGED_UNSUPPORTED,
                "paged-layout reject reason should be explicit");
    return true;
}

// Only two descriptor fields follow the planner's decision -- `required` and the
// `bytes_per_tensor` derived from it. Every geometry field is filled the same
// way whether the plan came back DIRECT or MATERIALIZE_REQUIRED, so those stay
// unconditional and keep their coverage in all three states.
static bool test_materialization_descriptor_for_gqa_mismatch() {
    fattn_params                             params = mistral_like_params(/*k_nc_stride_elems=*/512);
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(ok, "GQA nc_stride != D should produce a materialization descriptor");
    if (materialize_enabled()) {
        TEST_ASSERT(desc.required, "MATERIALIZE=1: GQA descriptor should mark materialization required");
        TEST_ASSERT(desc.bytes_per_tensor == (size_t) params.ne12 * params.ne11 * params.ne00 * sizeof(sycl::half),
                    "MATERIALIZE=1: descriptor should size one dense f16 K/V tensor");
    } else {
        TEST_ASSERT(!desc.required, "MATERIALIZE default/=0: a DIRECT plan must not mark materialization required");
        TEST_ASSERT(desc.bytes_per_tensor == 0,
                    "MATERIALIZE default/=0: a DIRECT plan must not request materialization bytes");
    }
    TEST_ASSERT(desc.target_device == 0, "descriptor should preserve target device");
    TEST_ASSERT(desc.D == params.ne00, "descriptor should preserve head dimension");
    TEST_ASSERT(desc.n_kv == params.ne11, "descriptor should preserve KV length");
    TEST_ASSERT(desc.H_kv == params.ne12, "descriptor should preserve KV heads");
    TEST_ASSERT(desc.k_target_nb1 == (int64_t) params.ne00 * (int64_t) sizeof(sycl::half),
                "materialized K token stride must be dense D");
    TEST_ASSERT(desc.v_target_nb1 == (int64_t) params.ne00 * (int64_t) sizeof(sycl::half),
                "materialized V token stride must be dense D");
    return true;
}

static bool test_materialization_descriptor_direct_mha_noop() {
    fattn_params                             params = mha_like_params();
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(ok, "direct MHA should still produce a descriptor");
    TEST_ASSERT(!desc.required, "direct MHA descriptor should be a no-op");
    TEST_ASSERT(desc.bytes_per_tensor == 0, "direct MHA no-op should not request materialization bytes");
    return true;
}

static bool test_materialization_descriptor_rejects_unsupported_layout() {
    fattn_params params = mqa_like_params(/*k_nc_stride_elems=*/512);
    params.ne02         = 30;
    params.ne12         = 8;
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(!ok, "materializer should reject non-integral GQA/MQA head ratios");
    return true;
}

// ---------------------------------------------------------------------------
// Source contract: the oneDNN SDPA compile site must refuse a split partition.
//
// This is a SOURCE-CONTRACT assertion, not a behavioural one, and the difference
// matters. build_and_compile_sdpa is static, needs a dnnl::engine and a live
// SYCL queue, and the partition count is decided by oneDNN's pattern matcher --
// none of which this host-only binary has. Nor can a test ASK oneDNN to decline
// fusion: whether the pattern splits depends on the operand layouts the matcher
// happens to reject, which is exactly the thing nobody has managed to trigger
// (llama.cpp-mrld was found by reading, not by a failing run). So a behavioural
// test here would have to fake the split, which would prove only that the fake
// works. It reads the working-tree source instead, which pins the guard against
// deletion during a refactor -- the realistic way this fix gets lost.
//
// What it therefore does NOT establish: that the LOADED libggml-sycl.so carries
// the guard. It asserts a property of the checkout at test-run time, not of the
// artifact. A GPU test that observes the fallback actually happening is the
// missing piece and needs a shape that provably splits the partition.
//
// It cannot pass vacuously: not finding the file, or not finding the compile
// site it anchors on, is a FAIL, not a skip.
//
// The locator is the candidate_roots()/join_path() idiom already used by five
// source-reading tests in this directory (test-sycl-moe-direct-final-scratch-
// plan.cpp and siblings): LLAMA_CPP_REPO_ROOT override first, then the repo
// root recovered from this TU's compile-time __FILE__, then cwd guesses last.
// The __FILE__ anchor is what makes the invocation directory irrelevant, which
// matters because this target is install()ed and so gets run from arbitrary
// cwds. One deliberate deviation from the siblings: they std::exit(1) when the
// file cannot be read, which here would skip the remaining fifteen gates and
// the summary line, so this returns false through TEST_ASSERT instead.
// ---------------------------------------------------------------------------
static std::string join_path(const std::string & root, const char * rel) {
    if (root.empty() || root == ".") {
        return rel;
    }
    return root.back() == '/' ? root + rel : root + "/" + rel;
}

static std::vector<std::string> candidate_roots() {
    std::vector<std::string> roots;
    if (const char * env = std::getenv("LLAMA_CPP_REPO_ROOT")) {
        roots.emplace_back(env);
    }
    const std::string source_file = __FILE__;
    const std::string suffix      = "/tests/test-sycl-fattn-onednn-gates.cpp";
    const size_t      pos         = source_file.rfind(suffix);
    if (pos != std::string::npos) {
        roots.emplace_back(source_file.substr(0, pos));
    }
    roots.emplace_back(".");
    roots.emplace_back("..");
    roots.emplace_back("../..");
    roots.emplace_back("../../..");
    return roots;
}

static std::string read_source_file(const char * rel) {
    for (const std::string & root : candidate_roots()) {
        std::ifstream in(join_path(root, rel), std::ios::binary);
        if (!in.good()) {
            continue;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    return std::string();
}

static bool test_source_contract_partition_split_is_refused() {
    const std::string src = read_source_file("ggml/src/ggml-sycl/fattn-onednn.cpp");
    TEST_ASSERT(!src.empty(),
                "could not read ggml/src/ggml-sycl/fattn-onednn.cpp from any candidate root -- this check must not "
                "report the contract satisfied when it never read the file");
    TEST_ASSERT(src.size() > 1000, "fattn-onednn.cpp read back truncated");

    const size_t compile_at = src.find("parts[0].compile(");
    TEST_ASSERT(compile_at != std::string::npos,
                "no parts[0].compile( site in fattn-onednn.cpp -- this check has lost its anchor and proves nothing");
    TEST_ASSERT(src.find("parts[0].compile(", compile_at + 1) == std::string::npos,
                "more than one parts[0].compile( site: the guard asserted below covers only the first");

    const size_t guard_at = src.find("if (parts.size() != 1)");
    TEST_ASSERT(guard_at != std::string::npos,
                "no `if (parts.size() != 1)` guard before the oneDNN SDPA compile: if oneDNN declines to fuse the "
                "pattern, out_ports[0] binds the softmax-probs intermediate into params.dst and overruns it");
    TEST_ASSERT(guard_at < compile_at, "the partition-count guard must precede parts[0].compile(, not follow it");
    TEST_ASSERT(src.find("throw", guard_at) < compile_at,
                "the partition-count guard must fail closed (throw, which the caller turns into a native-FA "
                "fallback) before the compile -- warning and continuing still binds the wrong tensor");
    return true;
}

int main() {
    // Name the polarity under test on stdout. A green run says nothing about
    // the other state, and a log that does not record which one it verified
    // invites exactly that misreading -- the three-state contract only holds if
    // someone can tell the three runs apart afterwards.
    const char * materialize_env = std::getenv("GGML_SYCL_FA_ONEDNN_MATERIALIZE");
    std::printf("SYCL fattn oneDNN gate tests: GGML_SYCL_FA_ONEDNN_MATERIALIZE=%s -> materialize %s\n",
                materialize_env ? materialize_env : "<unset>", materialize_enabled() ? "ON" : "OFF");

    bool ok = true;
    ok &= test_source_contract_partition_split_is_refused();
    ok &= test_gqa_nc_stride_mismatch_eligibility_follows_materialize_state();
    ok &= test_nc_stride_axis_moves_plan_iff_materialize_enabled();
    ok &= test_gqa_nc_stride_equal_d_remains_onednn_eligible();
    ok &= test_planner_direct_mha_contiguous();
    ok &= test_planner_gqa_mismatch_plan_follows_materialize_state();
    ok &= test_planner_mqa_mismatch_plan_follows_materialize_state();
    ok &= test_materialization_descriptor_for_gqa_mismatch();
    ok &= test_materialization_descriptor_direct_mha_noop();
    ok &= test_materialization_descriptor_rejects_unsupported_layout();
    ok &= test_planner_rejects_unsupported_d();
    ok &= test_planner_rejects_unproven_batch();
    ok &= test_planner_rejects_paged_layout();
    ok &= test_planner_rejects_scale_not_inv_sqrt_d();
    ok &= test_planner_rejects_scale_before_materialization();
    ok &= test_planner_accepts_inv_sqrt_d_at_odd_d();
    ok &= test_materialization_descriptor_rejects_bad_scale();
    std::printf("SYCL fattn oneDNN gate tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
