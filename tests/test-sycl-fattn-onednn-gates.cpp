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
#include "ggml.h"
#include "test-skip.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    // 77 (ctest SKIP_RETURN_CODE), not 0: no gate was evaluated in this build,
    // so this must not read as a pass.  A green skip is the exact defect class
    // this file's own comment is about (llama.cpp-ay8c).
    std::fprintf(stderr, "SKIP: GGML SYCL oneDNN not enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
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

// llama.cpp-p0f5: the oneDNN partition's softmax divisor is no longer baked
// from D alone -- it is 1/scale, taken from the runtime call and keyed into
// the compiled-partition cache (sdpa_shape_key::scale). phi2's pre-scaled-Q
// pattern (kq_scale=1.0 instead of 1/sqrt(D)) is exactly the case this
// generalizes: D=80 <= 256 is within this ticket's relaxed scope (see the
// CONSERVATIVE SCOPE comment on the planner's scale gate), so phi2 must now
// reach oneDNN instead of being rejected. This test used to assert the
// opposite (SCALE_UNSUPPORTED) -- that was correct for the old sqrt(D)-only
// contract and is the exact thing this ticket changes; flipping it here
// rather than deleting it keeps the phi2 shape under coverage.
static bool test_planner_accepts_prescaled_q_at_d80() {
    fattn_params params = phi2_like_params();
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "phi2's pre-scaled-Q kq_scale=1.0 at D=80 must now reach oneDNN (llama.cpp-p0f5), not reject on "
                "the old sqrt(D)-only formula");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK, "D=80 pre-scaled-Q accept should have OK reason");
    return true;
}

// gemma4's SWA layers are D=256 with the same pre-scaled-Q pattern (kq_scale
// hardcoded to 1.0, see phi2_like_params above) -- this is the actual
// hardware-evidence shape from llama.cpp-p0f5 (c-bnd0): a B50 device-event
// census attributed ~56% of a gemma-4-E4B pp512 pass to the native xmx_v2
// kernel these 35 SWA layers were forced onto by the old sqrt(D)-only gate.
// D=256 sits inside the relaxed D<=256 scope (D=512 stays on the strict
// check BY DEFAULT -- see test_planner_d512_gemma3n_scale_follows_relax_hatch
// below and the CONSERVATIVE SCOPE comment in
// ggml_sycl_flash_attn_ext_onednn_plan).
static bool test_planner_accepts_prescaled_q_at_d256() {
    fattn_params params = mha_like_params();
    params.ne00         = 256;
    params.ne10         = 256;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb02         = params.nb01 * params.ne01;
    params.nb03         = params.nb02 * params.ne02;
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    params.scale        = 1.0f;  // gemma4's kq_scale, NOT 1/sqrt(256)
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "gemma4's pre-scaled-Q kq_scale=1.0 at D=256 must reach oneDNN (llama.cpp-p0f5) -- this is the "
                "SWA-layer shape the hardware evidence in c-bnd0 measured at ~56% of a gemma-4-E4B pp512 pass");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK,
                "D=256 pre-scaled-Q accept should have OK reason");
    return true;
}

// Zero and non-finite scales remain unusable as a Divide-op divisor no matter
// how far the formula requirement is relaxed -- this is the invariant that
// actually survives the generalization (see the planner's scale gate).
static bool test_planner_rejects_nonfinite_scale() {
    {
        fattn_params params = phi2_like_params();
        params.scale        = 0.0f;
        const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                   /*multi_seq=*/false);
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT, "scale=0.0 must still reject oneDNN");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                    "scale=0.0 reject reason should be explicit");
    }
    {
        fattn_params params = phi2_like_params();
        params.scale        = std::numeric_limits<float>::quiet_NaN();
        const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                   /*multi_seq=*/false);
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT, "scale=NaN must still reject oneDNN");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                    "scale=NaN reject reason should be explicit");
    }
    {
        fattn_params params = phi2_like_params();
        params.scale        = std::numeric_limits<float>::infinity();
        const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                   /*multi_seq=*/false);
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT, "scale=Inf must still reject oneDNN");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                    "scale=Inf reject reason should be explicit");
    }
    return true;
}

// Same shape family, but GQA with a non-dense K/V stride — the plan kind that
// would otherwise be MATERIALIZE_REQUIRED. The scale gate must win over it,
// because the execute-time assert sits AFTER materialization. Uses scale=0.0
// rather than phi2_like_params' inherited 1.0, which is now admissible
// (llama.cpp-p0f5) and would no longer exercise this precedence at all.
static bool test_planner_rejects_scale_before_materialization() {
    fattn_params params = phi2_like_params();
    params.scale        = 0.0f;                                        // genuinely unusable, not phi2's admissible 1.0
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
// Uses scale=0.0 rather than phi2_like_params' inherited 1.0, which is now
// admissible (llama.cpp-p0f5) and would no longer be a "bad scale" at all.
static bool test_materialization_descriptor_rejects_bad_scale() {
    fattn_params params = phi2_like_params();
    params.scale        = 0.0f;
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(!ok, "materializer must refuse a shape the planner rejected on scale");
    return true;
}

// llama.cpp-jahv: D=512 (e.g. gemma-3n's 7 global-attention layers) is only
// ever routed through this planner -- fattn.cpp has no vec/tile/XMX/ESIMD
// kernel for D=512, so ggml_sycl_flash_attn_ext_supported() admits it only
// when this exact plan call would return DIRECT or MATERIALIZE_REQUIRED. The
// planner's own D range check (`ne00 > 512` rejects) has always covered
// D==512, but nothing exercised that boundary before this pass --
// test_planner_rejects_unsupported_d only probes from the D=1024 side. This
// closes the D=512-exactly gap on the accept side.
static bool test_planner_accepts_d512_with_matching_scale() {
    fattn_params params = mha_like_params();
    params.ne00         = 512;
    params.ne10         = 512;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb02         = params.nb01 * params.ne01;
    params.nb03         = params.nb02 * params.ne02;
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    params.scale        = 1.0f / std::sqrt(512.0f);
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "D=512 with a standard 1/sqrt(D) scale must plan DIRECT -- the planner's own D<=512 range check "
                "has always covered this shape, and this proves the boundary rather than just D=1024's rejection");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK, "D=512 accept should have OK reason");
    return true;
}

// The documenting half of the D=512 story: gemma-3n hardcodes
// hparams.f_attention_scale = 1.0f (src/models/gemma3n.cpp), the same
// phi2-style pre-scaled-Q pattern as test_planner_accepts_prescaled_q_at_d80
// / _at_d256 above -- but D=512 is a different outcome BY DEFAULT.
// llama.cpp-p0f5's relaxation is scoped to D<=256 (see the CONSERVATIVE
// SCOPE comment in ggml_sycl_flash_attn_ext_onednn_plan), so gemma-3n's
// D=512 global-attention layers reject on scale here unless the
// llama.cpp-bn5k item 2 measurement hatch (GGML_SYCL_FA_ONEDNN_D512_SCALE)
// is explicitly turned on. So wiring D=512 into
// ggml_sycl_flash_attn_ext_supported()/ggml_sycl_flash_attn_ext()
// (llama.cpp-jahv) does NOT by itself move gemma-3n's global-attention
// layers off CPU -- in the default state they still reject here, on scale,
// not on D. This case pins that fact so the D<=256 relaxation cannot
// silently widen to D=512 unnoticed, AND that the hatch actually works when
// deliberately enabled -- both directions state-aware, mirroring
// d512_route_expected_admitted()'s pattern below rather than assuming the
// default.
static bool test_planner_d512_gemma3n_scale_follows_relax_hatch() {
    fattn_params params = mha_like_params();
    params.ne00         = 512;
    params.ne10         = 512;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb02         = params.nb01 * params.ne01;
    params.nb03         = params.nb02 * params.ne02;
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    params.scale        = 1.0f;  // gemma3n's hparams.f_attention_scale, NOT 1/sqrt(512)
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    if (ggml_sycl_fa_onednn_d512_scale_relaxed()) {
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                    "GGML_SYCL_FA_ONEDNN_D512_SCALE=1 (llama.cpp-bn5k measurement hatch) must let gemma-3n's "
                    "kq_scale=1.0 at D=512 reach oneDNN via the same finite&&nonzero screen D<=256 already "
                    "uses, not the strict 1/sqrt(D) check");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK,
                    "D=512 relaxed-scale accept should have OK reason");
    } else {
        TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                    "gemma-3n's kq_scale=1.0 at D=512 must still reject oneDNN (SCALE_UNSUPPORTED) when "
                    "GGML_SYCL_FA_ONEDNN_D512_SCALE is off (default) -- not silently reach the compiled "
                    "partition with the wrong softmax divisor");
        TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                    "reject reason should identify the scale mismatch, not e.g. UNSUPPORTED_D -- D=512 itself is "
                    "fine");
    }
    return true;
}

// ---------------------------------------------------------------------------
// spec review llama.cpp-jahv/c-c3y6, finding (d): the two cases above call
// ggml_sycl_flash_attn_ext_onednn_plan() directly, so they pass unchanged on
// the parent commit and do not guard ggml_sycl_fattn_d512_onednn_admissible()
// or the D==512 admission branch in ggml_sycl_flash_attn_ext_supported()
// (fattn.cpp) that this commit actually adds. The two cases below build a
// real GGML_OP_FLASH_ATTN_EXT tensor via the public ggml_flash_attn_ext()
// factory (the same one llama.cpp's graph builder uses) and call
// ggml_sycl_flash_attn_ext_supported() itself instead.
//
// The FIRST of the two (test_supports_op_d512_admission_follows_
// onednn_d512_state) guards the new code against deletion: it expects D=512
// to be ADMITTED in the default state, so reverting to the parent commit's
// unconditional `if (!fattn_vec_supports_head_dim(D)) return false;` turns
// it red. The SECOND case
// (test_supports_op_d512_gemma3n_scale_follows_relax_hatch, llama.cpp-bn5k)
// used to assert an unconditional decline -- against the p0f5 parent commit
// that stayed GREEN whether or not the new admission code existed, since
// that commit declined every D=512 op unconditionally regardless of scale
// (spec review re-review llama.cpp-jahv/c-362a caught the original "both
// would go red" claim this way, by actually re-running the suite under
// D=512 disabled). llama.cpp-bn5k's GGML_SYCL_FA_ONEDNN_D512_SCALE hatch
// makes it a REAL deletion guard too, but only in the hatch-ON state: with
// the hatch enabled, a plan that has not wired the relaxation through to
// this supports_op level stays declined and the test goes red. In the
// hatch-OFF (default) state it still only pins that gemma-3n's specific
// scale is declined all the way through the real supports_op entry point,
// not merely the planner underneath it.
// ---------------------------------------------------------------------------

// D=512, MHA (H_q==H_kv so ggml_can_mul_mat's broadcast check is trivially
// satisfied), no mask -- ggml_sycl_flash_attn_ext_supported() does not
// require one. ne01=8 meets GGML_SYCL_FA_ONEDNN_MIN_NCOLS' default threshold
// (prefill-shaped), which is required for the planner to even consider it.
static ggml_tensor * build_d512_flash_attn_ext_op(ggml_context * ctx, float scale) {
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 8, 4, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 16, 4, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 16, 4, 1);
    return ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
}

// Mirrors materialize_enabled()'s pattern: ggml_sycl_fattn_d512_onednn_admissible()
// latches its env-var reads into function-local statics on first call, once
// per process (same reason the oneDNN gates above are three separate ctest
// runs rather than one). So this reads the actual current state rather than
// assuming unset/default, and the assertion below tracks whichever state the
// process is actually in -- exactly like this file's existing
// MATERIALIZE-state-aware cases.
//
// Spec review re-review llama.cpp-jahv/c-362a, finding 2: the first version
// of this helper checked ONLY GGML_SYCL_FA_ONEDNN_D512 and assumed accept
// otherwise, which FALSE-REDS under three other legitimate states measured
// on the built binary -- GGML_SYCL_FA_ONEDNN=0, GGML_SYCL_PAGED_V2=1, and
// GGML_SYCL_FA_ONEDNN_MIN_NCOLS=16 (our fixed test shape's ne01=8 then falls
// below threshold) -- because the real admissibility helper checks four
// independent gates, not one. This now mirrors all four; GGML_SYCL_FA_NO_XMX=1
// is included too even though the reviewer's measurement didn't name it,
// because init_fa_onednn_config() (fattn.cpp) treats it as an equivalent
// oneDNN-disable to GGML_SYCL_FA_ONEDNN=0 and would false-red here the same
// way if left out.
static bool d512_route_expected_admitted() {
    // Backported from tile_d512_route_expected_enabled() below (spec review
    // rev-dtpk-qual, F6): GGML_SYCL_FLASH_ATTN_EXT is
    // ggml_sycl_flash_attn_ext_supported()'s master switch, checked before
    // D==512 admission is ever reached -- omitting it here false-reds this
    // helper's callers under GGML_SYCL_FLASH_ATTN_EXT=0 for a reason that
    // has nothing to do with the oneDNN route, the same false-red class
    // spec review rev-jahv-spec2 already found and fixed for this
    // function's other four gates (c-362a).
    const char * fa_ext_env = std::getenv("GGML_SYCL_FLASH_ATTN_EXT");
    if (fa_ext_env && (std::strcmp(fa_ext_env, "0") == 0 || std::strcmp(fa_ext_env, "false") == 0)) {
        return false;
    }
    if (!ggml_sycl_fa_onednn_d512_enabled()) {
        return false;
    }
    const char * onednn_env = std::getenv("GGML_SYCL_FA_ONEDNN");
    if (onednn_env && (std::strcmp(onednn_env, "0") == 0 || std::strcmp(onednn_env, "false") == 0)) {
        return false;
    }
    const char * no_xmx_env = std::getenv("GGML_SYCL_FA_NO_XMX");
    if (no_xmx_env && std::atoi(no_xmx_env) != 0) {
        return false;
    }
    const char * paged_v2_env = std::getenv("GGML_SYCL_PAGED_V2");
    if (paged_v2_env && (std::strcmp(paged_v2_env, "1") == 0 || std::strcmp(paged_v2_env, "true") == 0)) {
        return false;
    }
    // Matches ggml_sycl_flash_attn_ext_onednn_plan()'s own parse
    // (fattn-onednn.cpp): getenv, atoi, default 8. build_d512_flash_attn_ext_op()
    // fixes ne01=8, so any threshold above 8 rejects it via BELOW_MIN_NCOLS,
    // independent of D=512 admission entirely.
    const char * min_ncols_env = std::getenv("GGML_SYCL_FA_ONEDNN_MIN_NCOLS");
    const int    min_ncols     = min_ncols_env ? std::atoi(min_ncols_env) : 8;
    if (8 < min_ncols) {
        return false;
    }
    return true;
}

static bool test_supports_op_d512_admission_follows_onednn_d512_state() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the supports_op D=512 accept/kill-switch case");

    ggml_tensor * dst      = build_d512_flash_attn_ext_op(ctx, 1.0f / std::sqrt(512.0f));
    const bool    accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    if (d512_route_expected_admitted()) {
        TEST_ASSERT(accepted,
                    "a real FLASH_ATTN_EXT op at D=512 with a standard 1/sqrt(D) scale, no mask, no paged/seq-id "
                    "sources, must be admitted by ggml_sycl_flash_attn_ext_supported() when the D=512 oneDNN "
                    "route is enabled (default)");
    } else {
        TEST_ASSERT(!accepted,
                    "the current environment disables the D=512 oneDNN route (GGML_SYCL_FA_ONEDNN_D512=0, "
                    "GGML_SYCL_FA_ONEDNN=0/GGML_SYCL_FA_NO_XMX=1, GGML_SYCL_PAGED_V2=1, or "
                    "GGML_SYCL_FA_ONEDNN_MIN_NCOLS above this shape's ne01=8) -- ggml_sycl_flash_attn_ext_supported() "
                    "must decline even an otherwise fully-eligible D=512 op in that state, not only the dispatch "
                    "branch");
    }
    return true;
}

// State-aware across BOTH gates that must be open for admission:
// d512_route_expected_admitted() (GGML_SYCL_FA_ONEDNN_D512 and its three
// sibling switches, plus MIN_NCOLS) and the llama.cpp-bn5k
// GGML_SYCL_FA_ONEDNN_D512_SCALE hatch this ticket adds. Either gate closed
// must decline; both open must admit.
static bool test_supports_op_d512_gemma3n_scale_follows_relax_hatch() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the supports_op D=512 scale-relax-hatch case");

    ggml_tensor * dst      = build_d512_flash_attn_ext_op(ctx, 1.0f);  // gemma-3n's actual kq_scale
    const bool    accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    const bool expect_admitted = d512_route_expected_admitted() && ggml_sycl_fa_onednn_d512_scale_relaxed();
    if (expect_admitted) {
        TEST_ASSERT(accepted,
                    "a real FLASH_ATTN_EXT op at D=512 with gemma-3n's kq_scale=1.0 must be admitted by "
                    "ggml_sycl_flash_attn_ext_supported() when both the D=512 oneDNN route is enabled AND "
                    "GGML_SYCL_FA_ONEDNN_D512_SCALE is on");
    } else {
        TEST_ASSERT(!accepted,
                    "a real FLASH_ATTN_EXT op at D=512 with gemma-3n's kq_scale=1.0 must be declined by "
                    "ggml_sycl_flash_attn_ext_supported() (falls back to CPU) whenever the D=512 oneDNN route "
                    "is disabled OR GGML_SYCL_FA_ONEDNN_D512_SCALE is off (default) -- both gates must be open, "
                    "not just the bare planner level");
    }
    return true;
}

// ---------------------------------------------------------------------------
// D=512 TILE-PATH gates (llama.cpp-dtpk) -- the gemma-viable route. These
// probe ggml_sycl_flash_attn_ext_supported()'s tile-admissibility clause
// specifically, using shapes the oneDNN route above structurally cannot
// serve: a non-standard scale (gemma's kq_scale=1.0) and a decode-shaped
// call (ne01=1, below oneDNN's MIN_NCOLS regardless of scale). The tile
// route's env kill switch (GGML_SYCL_FA_TILE_D512) is read into a
// function-local static latched once per process (same reason
// materialize_enabled()/d512_route_expected_admitted() above are
// state-aware, not by-design-red), so the positive cases here are
// state-aware against the current process's env too.
//
// One case from the ticket's own plan is NOT present here: a non-integer
// GQA ratio (e.g. H_q=5, H_kv=2). ggml_flash_attn_ext()'s own factory
// (ggml.c) asserts ggml_can_mul_mat(k, q), which already forbids
// constructing such an op through the public API at all -- attempting it
// aborts the process, not the assertion this file wants to make. The
// integer-ratio check in ggml_sycl_fattn_d512_tile_admissible() is real
// defensive code (matches this fork's decline-not-assert convention) but is
// consequently untestable via a synthetic op built the way every other case
// here is; ggml core itself is the thing guaranteeing the invariant for any
// real op.
// ---------------------------------------------------------------------------

// D=512, F32 Q (matches what build_attn_mha's per-op cast bypass produces
// for this D -- see llama-graph.cpp), fixed standard GQA (H_q=4, H_kv=2,
// ratio 2), no mask, no paged/seq-id sources. ne01 is a parameter so
// callers can probe the decode shape without a second near-duplicate
// builder. H_q/H_kv were originally parameters too, for a planned
// non-integer-GQA-ratio negative test -- dropped (spec review
// rev-dtpk-qual, F5) once that case turned out to be unconstructible via
// this public factory at all (ggml_flash_attn_ext()'s own
// GGML_ASSERT(ggml_can_mul_mat(k, q)) forbids it -- see the note above
// build_d512_tile_flash_attn_ext_op's callers), since no case here has ever
// called this with anything but the defaults.
static ggml_tensor * build_d512_tile_flash_attn_ext_op(ggml_context * ctx, float scale, int32_t ne01_q = 8) {
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, ne01_q, 4, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    return ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
}

// Mirrors both state-dependent gates a positive tile case actually passes
// through: ggml_sycl_fattn_d512_tile_admissible()'s own env kill switch
// (its only one; every other condition it checks is a pure shape/type test
// a fixed synthetic op either satisfies or doesn't, unlike the oneDNN
// helper's four independent env gates), AND
// ggml_sycl_flash_attn_ext_supported()'s own master switch
// (GGML_SYCL_FLASH_ATTN_EXT), checked before D==512 admission is ever
// reached. Omitting the second one would repeat the exact false-red class
// spec review rev-jahv-spec2 found and fixed for the oneDNN cases (c-362a):
// setting GGML_SYCL_FLASH_ATTN_EXT=0 to debug something unrelated would
// make every positive case here go red for a reason that has nothing to do
// with the tile route.
//
// llama.cpp-bn5k item 2 quality review: the case right below this helper
// used to be named "..._tile_with_gemma_like_scale", but gemma's scale is no
// longer a shape only the tile route can serve -- with GGML_SYCL_FA_ONEDNN_
// D512_SCALE=1 AND GGML_SYCL_FA_TILE_D512=0, oneDNN alone can admit the
// identical op (see the case's own comment for why). Renamed to name the OR
// admission honestly rather than claim a route the assertion doesn't pin.
static bool tile_d512_route_expected_enabled() {
    const char * fa_ext_env = std::getenv("GGML_SYCL_FLASH_ATTN_EXT");
    if (fa_ext_env && (std::strcmp(fa_ext_env, "0") == 0 || std::strcmp(fa_ext_env, "false") == 0)) {
        return false;
    }
    // ggml_sycl_fa_tile_d512_enabled() (fattn.hpp/fattn.cpp), not a second
    // hand-rolled parse of GGML_SYCL_FA_TILE_D512 (spec review
    // rev-dtpk-qual, F2) -- a duplicated inline parse here was a
    // dual-parse drift risk the same way a third hand-rolled copy would
    // have been for the oneDNN switch's own accessor.
    return ggml_sycl_fa_tile_d512_enabled();
}

static bool test_supports_op_admits_d512_gemma_scale_via_tile_or_relaxed_onednn() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the tile-admission gemma-scale case");

    // scale=1.0 -- oneDNN's own admissibility helper rejects this in the
    // default state (SCALE_UNSUPPORTED;
    // test_supports_op_d512_gemma3n_scale_follows_relax_hatch above proves
    // it, using F16 Q at the same scale, declined whenever
    // GGML_SYCL_FA_ONEDNN_D512_SCALE is off). This op differs only in Q's
    // type (F32, matching what build_attn_mha's D=512 cast bypass produces)
    // -- so this being GREEN while that one stays RED (in the default,
    // hatch-off state) demonstrates the tile route performs real, distinct
    // admission, not merely agreeing with oneDNN's answer.
    //
    // llama.cpp-bn5k item 2: unlike the F16-Q and DV-mismatch decline cases
    // above, this op has NOTHING else oneDNN would reject it on (F32 Q is
    // fine, DKQ==DV==512 is fine) -- scale was its only oneDNN blocker. So
    // with GGML_SYCL_FA_ONEDNN_D512_SCALE=1 AND tile disabled
    // (GGML_SYCL_FA_TILE_D512=0), oneDNN itself can now admit this exact
    // shape; the `else` branch below must account for that or it false-reds
    // under that combination (caught live: forgetting this turned up as a
    // real failure the first time the hatch was tested here).
    ggml_tensor * dst      = build_d512_tile_flash_attn_ext_op(ctx, 1.0f);
    const bool    accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    const bool onednn_could_admit_too = d512_route_expected_admitted() && ggml_sycl_fa_onednn_d512_scale_relaxed();
    if (tile_d512_route_expected_enabled() || onednn_could_admit_too) {
        TEST_ASSERT(accepted,
                    "a real FLASH_ATTN_EXT op at D=512 with gemma's non-standard scale=1.0, F32 Q, an integer "
                    "GQA ratio, and no paged/seq-id sources must be admitted via the tile route when "
                    "GGML_SYCL_FA_TILE_D512 is not disabled, OR via oneDNN when GGML_SYCL_FA_ONEDNN_D512_SCALE "
                    "is on -- at least one of the two routes must be able to serve this shape");
    } else {
        TEST_ASSERT(!accepted,
                    "with the tile route disabled AND the oneDNN scale-relax hatch off (or oneDNN's D=512 route "
                    "otherwise disabled), NEITHER route can serve gemma's scale -- this op must be declined");
    }
    return true;
}

static bool test_supports_op_admits_d512_tile_decode_shape() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the tile-admission decode-shape case");

    // ne01=1 (decode), standard 1/sqrt(D) scale -- even a scale oneDNN
    // would otherwise accept can never reach it here: oneDNN's own
    // GGML_SYCL_FA_ONEDNN_MIN_NCOLS default (8) rejects any ne01<8
    // regardless of scale. This is gemma's actual per-token decode shape
    // for its D=512 global layers, and the reason the tile route matters
    // beyond prefill.
    ggml_tensor * dst      = build_d512_tile_flash_attn_ext_op(ctx, 1.0f / std::sqrt(512.0f), /*ne01_q=*/1);
    const bool    accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    if (tile_d512_route_expected_enabled()) {
        TEST_ASSERT(accepted,
                    "a decode-shaped (ne01=1) D=512 FLASH_ATTN_EXT op, F32 Q, standard scale, must be admitted "
                    "via the tile route -- oneDNN's MIN_NCOLS floor can never serve decode at any scale, so "
                    "this shape has no route but tile");
    } else {
        TEST_ASSERT(!accepted, "GGML_SYCL_FA_TILE_D512=0/false must decline the decode shape too");
    }
    return true;
}

static bool test_supports_op_declines_d512_tile_with_f16_q() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the tile F16-Q decline case");

    // Same shape as the gemma-scale acceptance case above, except Q is F16.
    // llama-graph.cpp's per-op cast bypass should never produce this for a
    // real D=512 op, but ggml_sycl_fattn_d512_tile_admissible() must not
    // trust that invariant blindly -- it is a defensive backstop, pinned as
    // a test rather than left as prose. flash_attn_tile<> hardcodes Q as
    // F32 (no Q_type template, unlike every sibling kernel family);
    // admitting F16 Q here would let dispatch reach a kernel that misreads
    // its own input rather than declining to CPU.
    //
    // Uses scale=0.0 rather than gemma's actual 1.0 (llama.cpp-bn5k item 2):
    // oneDNN accepts F16 Q just as readily as F32 (Q_TYPE_UNSUPPORTED allows
    // both), so this op's only would-be oneDNN blocker was ever the scale --
    // and GGML_SYCL_FA_ONEDNN_D512_SCALE=1 relaxes exactly that gate for
    // D=512. With scale=1.0 kept, this test would silently start reaching
    // ggml_sycl_flash_attn_ext_onednn() and pass for the WRONG reason (or
    // start failing) whenever that measurement hatch is on, without ever
    // exercising the tile route's F16-Q rejection this test exists to pin.
    // scale=0.0 hits the planner's unconditional zero/non-finite check
    // (fattn-onednn.cpp), which the hatch does not touch, so oneDNN declines
    // in every hatch state and this stays a pure test of the tile route.
    // Mirrors the identical fix already applied for the p0f5 scale
    // generalization (see test_planner_rejects_scale_before_materialization
    // above).
    ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 8, 4, 1);
    ggml_tensor * k   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    ggml_tensor * v   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    ggml_tensor * dst = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, /*scale=*/0.0f, /*max_bias=*/0.0f,
                                            /*logit_softcap=*/0.0f);
    const bool accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    TEST_ASSERT(!accepted,
                "an F16-Q D=512 op must be declined -- oneDNN rejects an unusable (zero) scale unconditionally, "
                "and the tile route must reject the Q dtype (its kernel has no Q_type template and hardcodes "
                "F32) regardless of GGML_SYCL_FA_TILE_D512's state");
    return true;
}

static bool test_supports_op_declines_d512_tile_with_paged_sources() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 10 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the tile paged-source decline case");

    ggml_tensor * dst = build_d512_tile_flash_attn_ext_op(ctx, 1.0f);
    // Same conservative exclusion as ggml_sycl_fattn_d512_onednn_admissible:
    // any of src[5..8] present declines outright rather than being planned,
    // because submit_fattn_tile_d512 doesn't implement paged-block-table or
    // continuous-batching seq-id skipping. Poke src[7] (block_table)
    // directly -- there is no public setter exercised by this synthetic op,
    // and dst->src[] is a plain public array ggml_flash_attn_ext() leaves
    // nullptr past index 4.
    ggml_tensor * block_table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    dst->src[7]                = block_table;
    const bool accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    TEST_ASSERT(!accepted,
                "a D=512 op carrying a paged block_table (src[7]) must be declined by the tile route -- "
                "submit_fattn_tile_d512 has no paged-layout support");
    return true;
}

// Spec review rev-dtpk-spec, F1(a): flash_attn_tile<>'s own early-out
// (fattn-tile.hpp, `use_logit_softcap && !(DV == 128 || DV == 256)`)
// returns WITHOUT writing dst for any DV outside {128,256} -- exactly
// this kernel's DV=512. Unreachable today (nothing constructs a softcapped
// D=512 op), but a real, silent-garbage-class bug if it were ever admitted
// -- pinned as a test, not left as prose the way the commit message
// describes it.
static bool test_supports_op_declines_d512_tile_with_logit_softcap() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the tile logit_softcap decline case");

    ggml_tensor * q        = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 8, 4, 1);
    ggml_tensor * k        = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    ggml_tensor * v        = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    ggml_tensor * dst      = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, /*scale=*/1.0f, /*max_bias=*/0.0f,
                                                 /*logit_softcap=*/30.0f);
    const bool accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    TEST_ASSERT(!accepted,
                "a D=512 op with logit_softcap != 0 must be declined by the tile route -- flash_attn_tile's own "
                "early-out returns without writing dst for any DV outside {128,256} when use_logit_softcap is "
                "true, which would silently no-op a softcapped D=512 call if admitted");
    return true;
}

// Spec review rev-dtpk-spec, F1(c): the launcher hardcodes
// submit_fattn_tile_d512<512, 512, ...>, but DKQ (Q/K's ne[0]) and DV
// (V's ne[0]) are logically independent -- the config table's 576/512 rows
// exist for exactly this split, and ggml_flash_attn_ext() (ggml.c) asserts
// no relationship between them (dst's ne[0] is set from v->ne[0] alone).
// An op with DKQ==512 but DV!=512 would silently process the wrong number
// of V elements if the tile route admitted it.
static bool test_supports_op_declines_d512_tile_with_dv_mismatch() {
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ggml_tensor_overhead() * 8 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(iparams);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed for the tile DV-mismatch decline case");

    // Uses scale=0.0 rather than gemma's 1.0 (llama.cpp-bn5k item 2): oneDNN's
    // dense GEMM-based SDPA graph does not share the tile kernel's hardcoded
    // DKQ==DV assumption (its QK^T and softmax@V stages are independently
    // shaped), so a DKQ/DV mismatch is not itself an oneDNN blocker -- scale
    // was. GGML_SYCL_FA_ONEDNN_D512_SCALE=1 relaxes exactly that gate, which
    // would let this op start reaching oneDNN instead of exercising the tile
    // route's own DV-mismatch guard this test exists to pin. scale=0.0 hits
    // the planner's unconditional zero/non-finite check, unaffected by the
    // hatch, so oneDNN declines in every hatch state. Same fix as the F16-Q
    // case above and the p0f5-era test_planner_rejects_scale_before_
    // materialization.
    ggml_tensor * q        = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 512, 8, 4, 1);
    ggml_tensor * k        = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 512, 256, 2, 1);
    ggml_tensor * v        = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 256, 2, 1);  // DV=256 != DKQ=512
    ggml_tensor * dst      = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, /*scale=*/0.0f, /*max_bias=*/0.0f,
                                                 /*logit_softcap=*/0.0f);
    const bool accepted = ggml_sycl_flash_attn_ext_supported(dst);
    ggml_free(ctx);

    TEST_ASSERT(!accepted,
                "a D=512 op whose V head dim (256) differs from Q/K's (512) must be declined by the tile route "
                "-- submit_fattn_tile_d512 hardcodes DV=512 and has no DKQ!=DV support");
    return true;
}

// Spec review rev-dtpk-spec, F1(b) is NOT present as a test here, and that
// is a deliberate finding rather than an omission: ggml_sycl_type_is_fp8_
// e4m3() (common.hpp) is a stub that unconditionally `return`s `false`
// regardless of its argument, so kv_is_fp8 is always false in this build --
// there is no ggml_type value that makes it true, and therefore no way to
// construct a synthetic op that reaches ggml_sycl_fattn_d512_tile_
// admissible() with K/V considered FP8 by anything in this tree today. The
// F16-type check the review asked for is real, correct defensive code
// against a future state where that stub is implemented, but is
// consequently untestable via a synthetic op the way every other case here
// is -- the same shape of finding as the non-integer-GQA case noted above
// build_d512_tile_flash_attn_ext_op.

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
// The locator is the candidate_roots()/join_path() idiom already used by six
// source-reading tests in this directory (test-sycl-fattn-xmx-policy.cpp,
// test-sycl-moe-direct-final-scratch-plan.cpp,
// test-sycl-moe-same-expert-grouping.cpp,
// test-sycl-moe-fused-down-sum-policy.cpp, test-sycl-moe-fusion-noactivation.cpp
// and test-sycl-moe-sequence-graphlet-policy.cpp): LLAMA_CPP_REPO_ROOT override
// first, then the absolute LLAMA_CPP_SOURCE_ROOT the build defines, then the repo
// root recovered from this TU's compile-time __FILE__, then cwd guesses last. The
// two build-time anchors are what make the invocation directory irrelevant,
// which matters because this target is install()ed and so gets run from
// arbitrary cwds. All seven copies list the SAME six cwd guesses ("."
// through "../../../../.."); that depth is behavioural, not cosmetic, since the
// guesses are what runs when the __FILE__ anchor fails, so a shallower copy stops
// finding the file from a deeper cwd. Change all seven together. Duplicating
// rather than hoisting into a shared header is the house style here.
// One deliberate deviation from the siblings: they std::exit(1) when the
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
#    ifdef LLAMA_CPP_SOURCE_ROOT
    // Absolute root from the build. Under ccache base_dir (scripts/sycl-build.sh)
    // __FILE__ is relative to the build directory, so it no longer pins the root.
    roots.emplace_back(LLAMA_CPP_SOURCE_ROOT);
#    endif
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
    roots.emplace_back("../../../..");
    roots.emplace_back("../../../../..");
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
    ok &= test_planner_accepts_d512_with_matching_scale();
    ok &= test_planner_d512_gemma3n_scale_follows_relax_hatch();
    ok &= test_supports_op_d512_admission_follows_onednn_d512_state();
    ok &= test_supports_op_d512_gemma3n_scale_follows_relax_hatch();
    ok &= test_supports_op_admits_d512_gemma_scale_via_tile_or_relaxed_onednn();
    ok &= test_supports_op_admits_d512_tile_decode_shape();
    ok &= test_supports_op_declines_d512_tile_with_f16_q();
    ok &= test_supports_op_declines_d512_tile_with_paged_sources();
    ok &= test_supports_op_declines_d512_tile_with_logit_softcap();
    ok &= test_supports_op_declines_d512_tile_with_dv_mismatch();
    ok &= test_planner_rejects_unsupported_d();
    ok &= test_planner_rejects_unproven_batch();
    ok &= test_planner_rejects_paged_layout();
    ok &= test_planner_accepts_prescaled_q_at_d80();
    ok &= test_planner_accepts_prescaled_q_at_d256();
    ok &= test_planner_rejects_nonfinite_scale();
    ok &= test_planner_rejects_scale_before_materialization();
    ok &= test_planner_accepts_inv_sqrt_d_at_odd_d();
    ok &= test_materialization_descriptor_rejects_bad_scale();
    std::printf("SYCL fattn oneDNN gate tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
