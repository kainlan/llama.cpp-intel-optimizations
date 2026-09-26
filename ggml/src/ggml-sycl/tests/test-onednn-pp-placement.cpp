#include "../onednn-pp-placement.hpp"

#include <cstdio>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously. Use an explicit check that always runs, per the
// test-layer-streaming-gate.cpp precedent in this directory.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_PLACEMENT(got, want, msg)                                                                              \
    do {                                                                                                             \
        const ggml_sycl::onednn_pp_placement got_v_  = (got);                                                        \
        const ggml_sycl::onednn_pp_placement want_v_ = (want);                                                       \
        if (got_v_ != want_v_) {                                                                                     \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %s, want %s)\n", __FILE__, __LINE__, msg,                     \
                         ggml_sycl::onednn_pp_placement_name(got_v_), ggml_sycl::onednn_pp_placement_name(want_v_)); \
            return 1;                                                                                                \
        }                                                                                                            \
    } while (0)

using ggml_sycl::onednn_pp_admission_decide;
using ggml_sycl::onednn_pp_admission_inputs;
using ggml_sycl::onednn_pp_executable;
using ggml_sycl::onednn_pp_min_batch_for;
using ggml_sycl::onednn_pp_placement;
using ggml_sycl::onednn_pp_placement_decide;
using ggml_sycl::onednn_pp_placement_inputs;
using ggml_sycl::onednn_pp_refusal;
using ggml_sycl::onednn_pp_route;
using ggml_sycl::onednn_pp_woq_alternates_allowed;

// An op every route admits: oneDNN PP on, not a skipped type, F32 operands, a
// contiguous quantized weight, and a batch at the dense floor.
static onednn_pp_admission_inputs admitted_inputs(onednn_pp_route route, int64_t batch) {
    onednn_pp_admission_inputs in;
    in.enabled                     = true;
    in.skip_type                   = false;
    in.batch                       = batch;
    in.min_batch                   = onednn_pp_min_batch_for(route, /*dense_min_batch=*/16);
    in.f32_operands                = true;
    in.contiguous_quantized_weight = true;
    return in;
}

int main() {
    // 1. The defect case (llama.cpp-1d0n): Mistral 7B Q4_0 split B70 0-29 /
    //    B50 30-31. Each card may run oneDNN PP for the weights it holds, and
    //    the plan still gets no WOQ second copies.
    {
        onednn_pp_placement_inputs in;
        in.multiple_routable_devices = true;
        in.plan_needs_secondaries    = true;
        const onednn_pp_placement p  = onednn_pp_placement_decide(in);
        CHECK_PLACEMENT(p, onednn_pp_placement::SPLIT_RESIDENT_WEIGHTS, "case 1: dense split");
        CHECK(onednn_pp_executable(p, /*weight_resident_on_device=*/true),
              "case 1: a dense split must run oneDNN PP for a weight resident on the executing device");
        CHECK(!onednn_pp_executable(p, /*weight_resident_on_device=*/false),
              "case 1: a dense split must not run oneDNN PP for a weight that is not resident on the device");
        CHECK(!onednn_pp_woq_alternates_allowed(p), "case 1: a dense split must not plan WOQ second copies");
    }

    // 2. Single routable device: everything is allowed, and residency is not
    //    consulted (a single card with planner host layers keeps its previous
    //    answer).
    {
        onednn_pp_placement_inputs in;
        in.multiple_routable_devices = false;
        in.plan_needs_secondaries    = true;  // ignored with one routable device
        in.moe_multi_gpu_active      = true;  // likewise
        const onednn_pp_placement p  = onednn_pp_placement_decide(in);
        CHECK_PLACEMENT(p, onednn_pp_placement::ALLOWED, "case 2: single routable device");
        CHECK(onednn_pp_executable(p, true), "case 2: executable");
        CHECK(onednn_pp_executable(p, false), "case 2: executable without consulting residency");
        CHECK(onednn_pp_woq_alternates_allowed(p), "case 2: WOQ alternates allowed");
    }

    // 3. MoE multi-GPU on several routable devices refuses both, even for a
    //    resident weight. It is checked before the split, so an MoE plan that
    //    also has secondaries stays refused.
    {
        onednn_pp_placement_inputs in;
        in.multiple_routable_devices = true;
        in.moe_multi_gpu_active      = true;
        in.plan_needs_secondaries    = true;
        const onednn_pp_placement p  = onednn_pp_placement_decide(in);
        CHECK_PLACEMENT(p, onednn_pp_placement::REFUSED_MOE_MULTI_GPU, "case 3: moe multi-gpu");
        CHECK(!onednn_pp_executable(p, true), "case 3: moe multi-gpu refuses a resident weight");
        CHECK(!onednn_pp_woq_alternates_allowed(p), "case 3: moe multi-gpu refuses WOQ alternates");
    }

    // 4. Several routable devices but a plan with no secondary work (e.g.
    //    everything fits the first card): unchanged, allowed.
    {
        onednn_pp_placement_inputs in;
        in.multiple_routable_devices = true;
        const onednn_pp_placement p  = onednn_pp_placement_decide(in);
        CHECK_PLACEMENT(p, onednn_pp_placement::ALLOWED, "case 4: multi-device host, single-device plan");
        CHECK(onednn_pp_executable(p, false), "case 4: executable");
        CHECK(onednn_pp_woq_alternates_allowed(p), "case 4: WOQ alternates allowed");
    }

    // 5. MoE multi-GPU on its own, with no plan secondaries: still refused.
    //    Case 3 sets both flags, so it cannot tell a refusal keyed on MoE from
    //    one that only fires inside the split branch.
    {
        onednn_pp_placement_inputs in;
        in.multiple_routable_devices = true;
        in.moe_multi_gpu_active      = true;
        in.plan_needs_secondaries    = false;
        const onednn_pp_placement p  = onednn_pp_placement_decide(in);
        CHECK_PLACEMENT(p, onednn_pp_placement::REFUSED_MOE_MULTI_GPU, "case 5: moe multi-gpu without secondaries");
        CHECK(!onednn_pp_executable(p, true), "case 5: moe multi-gpu refuses a resident weight");
        CHECK(!onednn_pp_woq_alternates_allowed(p), "case 5: moe multi-gpu refuses WOQ alternates");
    }

    // 6. Admission (llama.cpp-je3b). The route moves only the batch floor:
    //    the dense dispatcher keeps GGML_SYCL_ONEDNN_PP_MIN_BATCH, the MXFP4
    //    direct arms keep the M >= 2 floor they always had.
    {
        CHECK(onednn_pp_min_batch_for(onednn_pp_route::DENSE_DISPATCH, 16) == 16, "case 6: dense floor is the env");
        CHECK(onednn_pp_min_batch_for(onednn_pp_route::DENSE_DISPATCH, 64) == 64, "case 6: dense floor follows env");
        CHECK(onednn_pp_min_batch_for(onednn_pp_route::MXFP4_DIRECT, 16) == 2, "case 6: mxfp4 floor is 2");
        CHECK(onednn_pp_min_batch_for(onednn_pp_route::MXFP4_DIRECT, 64) == 2, "case 6: env does not move mxfp4");

        const int64_t small_batches[] = { 2, 8, 15 };
        for (int64_t m : small_batches) {
            CHECK(onednn_pp_admission_decide(admitted_inputs(onednn_pp_route::MXFP4_DIRECT, m)) ==
                      onednn_pp_refusal::NONE,
                  "case 6: mxfp4 direct admits a small PP batch");
            CHECK(onednn_pp_admission_decide(admitted_inputs(onednn_pp_route::DENSE_DISPATCH, m)) ==
                      onednn_pp_refusal::BATCH_UNDER_THRESHOLD,
                  "case 6: dense refuses a batch under its floor");
        }
        CHECK(
            onednn_pp_admission_decide(admitted_inputs(onednn_pp_route::DENSE_DISPATCH, 16)) == onednn_pp_refusal::NONE,
            "case 6: dense admits its floor");
        CHECK(onednn_pp_admission_decide(admitted_inputs(onednn_pp_route::MXFP4_DIRECT, 1)) ==
                  onednn_pp_refusal::BATCH_UNDER_THRESHOLD,
              "case 6: decode (M == 1) is never oneDNN PP");
    }

    // 7. The checks the MXFP4 direct arms used to skip now bind on every
    //    route: GGML_SYCL_ONEDNN_PP=0 and a skipped type refuse even a large
    //    batch, and the refusal is named before the batch is looked at.
    const onednn_pp_route routes[] = { onednn_pp_route::DENSE_DISPATCH, onednn_pp_route::MXFP4_DIRECT };
    for (onednn_pp_route route : routes) {
        onednn_pp_admission_inputs in = admitted_inputs(route, 512);
        CHECK(onednn_pp_admission_decide(in) == onednn_pp_refusal::NONE, "case 7: baseline admitted");

        in.enabled = false;
        CHECK(onednn_pp_admission_decide(in) == onednn_pp_refusal::DISABLED_OR_SKIP_TYPE, "case 7: disabled");
        in.batch = 1;
        CHECK(onednn_pp_admission_decide(in) == onednn_pp_refusal::DISABLED_OR_SKIP_TYPE,
              "case 7: disabled wins over the batch floor");

        in           = admitted_inputs(route, 512);
        in.skip_type = true;
        CHECK(onednn_pp_admission_decide(in) == onednn_pp_refusal::DISABLED_OR_SKIP_TYPE, "case 7: skip type");

        in              = admitted_inputs(route, 512);
        in.f32_operands = false;
        CHECK(onednn_pp_admission_decide(in) == onednn_pp_refusal::TYPE, "case 7: non-F32 operands");

        in                             = admitted_inputs(route, 512);
        in.contiguous_quantized_weight = false;
        CHECK(onednn_pp_admission_decide(in) == onednn_pp_refusal::NOT_CONTIGUOUS_QUANT,
              "case 7: non-contiguous or unquantized weight");
    }

    std::printf("test-onednn-pp-placement: all cases passed\n");
    return 0;
}
