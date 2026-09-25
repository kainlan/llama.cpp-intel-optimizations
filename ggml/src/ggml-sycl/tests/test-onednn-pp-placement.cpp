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

using ggml_sycl::onednn_pp_executable;
using ggml_sycl::onednn_pp_placement;
using ggml_sycl::onednn_pp_placement_decide;
using ggml_sycl::onednn_pp_placement_inputs;
using ggml_sycl::onednn_pp_woq_alternates_allowed;

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

    std::printf("test-onednn-pp-placement: all cases passed\n");
    return 0;
}
