// Test: the load-time measure's plan override (llama.cpp-zhcn), (d), (k4),
// (l)).
//
// While a measure context is built and reserved, every plan accessor on the loading
// thread answers from the plan the measure names, nothing is published, and no
// other thread sees it.  This fixture drives the backend's install and clear procs
// directly and reads the accessors the real readers read:
//
//  * install finds the plan staged for a load, at the probe stage and at the
//    candidate stages, and refuses a load with none;
//  * under the override the global plan accessors answer from it, on this thread
//    only, and clear restores the real answers;
//  * a nest is refused and leaves the outer override in place;
//  * a plan publication under the override is refused and publishes nothing;
//  * the plan-derived multi-GPU latch read follows the override's plan on a mock
//    two-GPU host, where the process latch (false) cannot.
//
// No queue or cache is touched: staging, the accessors and the publish store are
// host code.
//
// Usage:
//   ./build/bin/test-sycl-measure-plan-override

#include "common.hpp"
#include "ggml-sycl-test.hpp"
#include "ggml-sycl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

using namespace ggml_sycl;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

constexpr uint64_t PROBE_TXN     = 41001;
constexpr uint64_t CANDIDATE_TXN = 41002;

ggml_sycl_device_info make_two_gpu_info() {
    ggml_sycl_device_info info{};
    info.device_count    = 2;
    info.total_gpu_count = 2;
    for (int d = 0; d < 2; ++d) {
        info.devices[d].cc                       = 1200;
        info.devices[d].total_vram               = 16ull << 30;
        info.devices[d].free_vram_at_init        = info.devices[d].total_vram;
        info.devices[d].max_alloc_size           = info.devices[d].total_vram;
        info.devices[d].safe_max_alloc_size      = info.devices[d].total_vram;
        info.devices[d].xmx_caps.global_mem_size = info.devices[d].total_vram;
        info.gpu_dpct_ids[d]                     = d;
    }
    return info;
}

// A plan whose only content is one expert on device 1 (needs secondaries) or none.
placement_plan make_plan(bool secondary_expert, size_t marker_bytes) {
    placement_plan plan{};
    plan.multi_device      = true;
    plan.devices           = { 0, 1 };
    plan.weight_host_bytes = marker_bytes;
    if (secondary_expert) {
        placement_entry e;
        e.name          = "blk.0.ffn_gate_exps.weight";
        e.expert_id     = 0;
        e.on_device     = true;
        e.target_device = 1;
        plan.entries.push_back(e);
    }
    return plan;
}

void test_install_finds_the_staged_plan() {
    // Nothing staged: refused, nothing installed.
    CHECK(!ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "no probe plan staged: install refuses");
    CHECK(!ggml_backend_sycl_measure_plan_override_install(CANDIDATE_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, nullptr),
          "no candidate staged: install refuses");
    CHECK(!ggml_backend_sycl_has_active_placement_plan(), "a refused install leaves no plan");

    lifecycle_stage_probe_placement_plan(PROBE_TXN, make_plan(true, 111));
    lifecycle_stage_placement_plan(CANDIDATE_TXN, make_plan(false, 222));

    // A probe plan is not a candidate: the stages read different registries.
    CHECK(!ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_C, nullptr),
          "a probe plan is not found as a candidate");
    CHECK(!ggml_backend_sycl_measure_plan_override_install(CANDIDATE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "a candidate plan is not found as a probe plan");
}

void test_accessors_follow_the_override() {
    const auto real_owner = global_placement_plan_owner();
    CHECK(real_owner && real_owner->entries.empty(), "baseline: no published plan");

    CHECK(ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "probe install succeeds");
    {
        const auto owner = global_placement_plan_owner();
        CHECK(owner && owner->weight_host_bytes == 111, "global_placement_plan_owner answers from the probe plan");
        CHECK(ggml_backend_sycl_has_active_placement_plan(), "has_active_placement_plan reads the override");
        CHECK(coherent_placement_plan_owner(nullptr)->weight_host_bytes == 111,
              "coherent_placement_plan_owner forwards to the override");
        const auto coherence = cache_placement_coherence(nullptr);
        CHECK(coherence.coherence == placement_cache_coherence::MATCH && coherence.owner->weight_host_bytes == 111,
              "cache_placement_coherence answers from the override, MATCH, with no cache snapshot");

        // Another thread sees the real plan.
        std::atomic<int> other_bytes{ -1 };
        std::thread      other(
            [&] { other_bytes.store(static_cast<int>(global_placement_plan_owner()->weight_host_bytes)); });
        other.join();
        CHECK(other_bytes.load() == 0, "another thread reads the real (empty) plan");
    }
    ggml_backend_sycl_measure_plan_override_clear();
    CHECK(global_placement_plan_owner()->weight_host_bytes == 0, "clear restores the real answer");
    CHECK(!ggml_backend_sycl_has_active_placement_plan(), "clear restores has_active_placement_plan");
    CHECK(cache_placement_coherence(nullptr).coherence == placement_cache_coherence::GENUINE_NO_PLAN,
          "clear restores the coherence answer");

    CHECK(ggml_backend_sycl_measure_plan_override_install(CANDIDATE_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, nullptr),
          "candidate install succeeds");
    CHECK(global_placement_plan_owner()->weight_host_bytes == 222, "the candidate stage answers from the candidate");
    ggml_backend_sycl_measure_plan_override_clear();
    ggml_backend_sycl_measure_plan_override_clear();  // clearing a clear override is harmless
}

void test_nest_is_refused() {
    CHECK(ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "outer install succeeds");
    CHECK(!ggml_backend_sycl_measure_plan_override_install(CANDIDATE_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, nullptr),
          "a nested install is refused");
    CHECK(global_placement_plan_owner()->weight_host_bytes == 111, "the outer override is still the one in place");
    ggml_backend_sycl_measure_plan_override_clear();
    CHECK(global_placement_plan_owner()->weight_host_bytes == 0, "one clear ends the outer override");
}

void test_publish_is_refused_under_the_override() {
    const auto before = global_placement_plan_owner();
    CHECK(ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "install for the publish arm");
    // Any store is refused, whatever it carries; this is not the override's own pointer.
    test_set_kv_placement_plan(make_plan(false, 333), 4, 1024);
    ggml_backend_sycl_measure_plan_override_clear();
    CHECK(global_placement_plan_owner().get() == before.get(),
          "a publication under the override published nothing (the real owner is pointer-unchanged)");

    // Positive control: the same call outside the override does publish.
    test_set_kv_placement_plan(make_plan(false, 333), 4, 1024);
    CHECK(global_placement_plan_owner()->weight_host_bytes == 333, "control: a publication outside the override lands");
    test_clear_kv_placement_plan();
    CHECK(global_placement_plan_owner()->weight_host_bytes == 0, "control cleaned up");
}

void test_latch_read_follows_the_plan() {
    test_sycl_info_override_guard info_guard(make_two_gpu_info());
    unsetenv("GGML_SYCL_MOE_MULTI_GPU");

    const bool latch_before = test_moe_multi_gpu_latch();
    CHECK(!latch_before, "baseline: the process latch is off");
    CHECK(!test_moe_multi_gpu_for_executor(), "outside the override the read is the latch");

    CHECK(test_moe_multi_gpu_wanted(make_plan(true, 0)), "a plan with a secondary expert wants multi-GPU");
    CHECK(!test_moe_multi_gpu_wanted(make_plan(false, 0)), "a plan with none does not");

    CHECK(ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "install the secondary-expert probe plan");
    CHECK(test_moe_multi_gpu_for_executor(), "under the override the read answers from the plan, not the latch");
    CHECK(test_moe_multi_gpu_latch() == latch_before, "the process latch itself is untouched");
    ggml_backend_sycl_measure_plan_override_clear();

    CHECK(ggml_backend_sycl_measure_plan_override_install(CANDIDATE_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, nullptr),
          "install the primary-only candidate plan");
    CHECK(!test_moe_multi_gpu_for_executor(), "a primary-only plan reads false under the override");
    ggml_backend_sycl_measure_plan_override_clear();

    // The env switch is part of "wanted", as it is for the writer.
    setenv("GGML_SYCL_MOE_MULTI_GPU", "0", 1);
    CHECK(!test_moe_multi_gpu_wanted(make_plan(true, 0)), "GGML_SYCL_MOE_MULTI_GPU=0 turns the want off");
    CHECK(ggml_backend_sycl_measure_plan_override_install(PROBE_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "install again under the env switch");
    CHECK(!test_moe_multi_gpu_for_executor(), "the executor read honours the switch under the override");
    ggml_backend_sycl_measure_plan_override_clear();
    unsetenv("GGML_SYCL_MOE_MULTI_GPU");
}

// llama.cpp-p6i0: given the measure context's KV shape, install re-fits the staged plan's KV residency for it,
// with the runtime transaction's fit, against the headroom the plan leaves for KV (no cache exists here, so a
// device's capacity is the plan's budget). The staged plan itself is never changed.
constexpr uint64_t REFIT_TXN = 41003;

placement_plan make_kv_plan(size_t vram_budget, size_t weight_vram_bytes) {
    placement_plan plan{};
    plan.multi_device      = false;
    plan.device_id         = 0;
    plan.devices           = { 0 };
    plan.vram_budget       = vram_budget;
    plan.weight_vram_bytes = weight_vram_bytes;
    plan.kv_device         = {
        { 0, 0 },
        { 1, 0 }
    };
    plan.kv_per_layer  = 512u * 512u;  // n_ctx 512 at 256 f16 values a cell
    plan.planner_n_ctx = 512;
    return plan;
}

placement_kv_info make_kv_info() {
    placement_kv_info kv{};
    kv.n_layer      = 2;
    kv.n_embd_k_gqa = 128;
    kv.n_embd_v_gqa = 128;
    kv.n_ctx        = 512;
    kv.n_ubatch     = 512;
    return kv;
}

bool kv_on_device(int layer) {
    return ggml_backend_sycl_kv_layer_on_device_from_dev(nullptr, layer);
}

void test_install_refits_kv_for_the_shape() {
    const size_t                     layer_bytes = (size_t) 512 * (1u << 20);  // one layer at n_ctx 1M
    const ggml_sycl_measure_kv_shape big         = { 1u << 20, 512, 1, false, false };
    char                             msg[160];

    struct arm {
        size_t budget;
        bool   on0;
        bool   on1;
    };

    const arm arms[] = {
        { 0,                                                   false, false }, // no headroom: both on host
        { (size_t) 64 << 30,                                   true,  true  }, // room for both
        { layer_bytes + kv_alloc_slack_per_layer + (1u << 20), true,  false }, // room for one: the latest goes
    };
    for (const arm & a : arms) {
        lifecycle_stage_placement_plan(REFIT_TXN, make_kv_plan(a.budget, 0), make_kv_info(), 2);

        // no shape: the plan as staged, every layer where the load put it
        CHECK(ggml_backend_sycl_measure_plan_override_install(REFIT_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, nullptr),
              "install without a shape succeeds");
        CHECK(kv_on_device(0) && kv_on_device(1), "without a shape the staged residency stands");
        ggml_backend_sycl_measure_plan_override_clear();

        CHECK(ggml_backend_sycl_measure_plan_override_install(REFIT_TXN, GGML_SYCL_MEASURE_STAGE_CANDIDATE_B, &big),
              "install with a shape succeeds");
        std::snprintf(msg, sizeof(msg), "budget %zu: layer 0 on device should be %d, layer 1 %d", a.budget, (int) a.on0,
                      (int) a.on1);
        CHECK(kv_on_device(0) == a.on0 && kv_on_device(1) == a.on1, msg);
        const auto owner = global_placement_plan_owner();
        CHECK(owner && owner->planner_n_ctx == big.n_ctx, "the override's plan carries the shape's n_ctx");
        ggml_backend_sycl_measure_plan_override_clear();

        // the staged candidate is untouched by the re-fit
        const auto staged = lifecycle_find_candidate_placement_plan(REFIT_TXN);
        CHECK(staged && staged->plan && staged->plan->planner_n_ctx == 512 && staged->kv_info.n_ctx == 512 &&
                  staged->plan->get_kv_device(0) == 0 && staged->plan->get_kv_device(1) == 0,
              "the staged plan is not changed by a re-fit");
    }
    lifecycle_abort_placement_plan(REFIT_TXN);
}

// llama.cpp-p6i0: the probe plan the early stage stages beside the load's candidate lives exactly as long as the
// candidate. A load that ends either way, by abort or by publish, leaves no probe plan behind for a later load's
// measure to find.
void test_probe_plan_goes_with_the_candidate() {
    constexpr uint64_t DROP_TXN   = 41004;
    constexpr uint64_t DROP_MODEL = 0x9e6f0004ull;

    // the abort path
    lifecycle_stage_placement_plan(DROP_TXN, make_plan(false, 444));
    lifecycle_stage_probe_placement_plan(DROP_TXN, make_plan(false, 445));
    CHECK(lifecycle_find_probe_placement_plan(DROP_TXN) != nullptr, "the probe plan was not staged");
    lifecycle_abort_placement_plan(DROP_TXN);
    CHECK(lifecycle_find_probe_placement_plan(DROP_TXN) == nullptr, "an aborted load left its probe plan behind");
    CHECK(!ggml_backend_sycl_measure_plan_override_install(DROP_TXN, GGML_SYCL_MEASURE_STAGE_PROBE, nullptr),
          "a measure found the probe plan of an aborted load");

    // the publish path
    lifecycle_stage_placement_plan(DROP_TXN, make_plan(false, 446));
    lifecycle_stage_probe_placement_plan(DROP_TXN, make_plan(false, 447));
    CHECK(lifecycle_publish_placement_plan(DROP_MODEL, DROP_TXN, 0, 1, 0, 0, false, nullptr),
          "the candidate did not publish");
    CHECK(lifecycle_find_probe_placement_plan(DROP_TXN) == nullptr, "a published load left its probe plan behind");
    CHECK(lifecycle_find_candidate_placement_plan(DROP_TXN) == nullptr, "the published candidate is still staged");
    lifecycle_erase_placement_plan(DROP_MODEL, DROP_TXN);
}

}  // namespace

int main() {
    setenv("GGML_SYCL_STRICT_LEASES", "0", 1);
    test_install_finds_the_staged_plan();
    test_accessors_follow_the_override();
    test_nest_is_refused();
    test_publish_is_refused_under_the_override();
    test_latch_read_follows_the_plan();
    test_install_refits_kv_for_the_shape();
    test_probe_plan_goes_with_the_candidate();
    lifecycle_abort_probe_placement_plan(PROBE_TXN);
    lifecycle_abort_placement_plan(CANDIDATE_TXN);
    if (g_failures != 0) {
        std::fprintf(stderr, "test-sycl-measure-plan-override: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-sycl-measure-plan-override: all ok\n");
    return 0;
}
