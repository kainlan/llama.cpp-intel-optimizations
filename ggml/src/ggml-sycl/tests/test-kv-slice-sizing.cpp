// Regression test for KV per-layer slice sizing (llama.cpp-2120).
//
// The defect: kv_tier_manager sized each per-layer KV slice by dividing the
// buffer by the model's TOTAL layer count instead of the number of layers that
// actually hold KV.  For a hybrid model with 1 attention layer of 2 that is
// exactly 2x undersized; the KV cache was then written past its allocation and
// execution continued.  Ten architectures ran with out-of-bounds KV pointers
// and seven of them reported OK.
//
// This test exercises the REAL ggml_sycl::kv_tier_manager and the REAL
// ggml_sycl::kv_slice_size — no mock reimplementation.  The one seam is
// unified_cache_get_layer_vram_bytes(), stubbed below so the weight-aware path
// can be driven without a device: the arithmetic under test is pure and needs
// no GPU.
//
// Written RED: against the pre-fix tree, cases 1, 2, 6 and 7 fail.

#include "../kv-tier-manager.hpp"
#include "../unified-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <vector>

// ---------------------------------------------------------------------------
// Seam: per-layer device weight residency, normally answered by unified-cache.
// The test owns the answer so configure_with_weights() can be driven both ways.
// ---------------------------------------------------------------------------
static std::vector<size_t> g_stub_layer_vram_bytes;

namespace ggml_sycl {
size_t unified_cache_get_layer_vram_bytes(int device, int layer_id) {
    (void) device;
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= g_stub_layer_vram_bytes.size()) {
        return 0;
    }
    return g_stub_layer_vram_bytes[layer_id];
}
}  // namespace ggml_sycl

// ---------------------------------------------------------------------------

static int g_failures = 0;
static int g_checks   = 0;

static void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_failures++;
        printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

static void check_true(const char * what, bool cond) {
    g_checks++;
    if (!cond) {
        g_failures++;
        printf("  FAIL: %s\n", what);
    }
}

// A plan for an n_layer model with every listed layer's KV assigned to `device`.
static ggml_sycl::placement_plan make_plan(uint32_t                      n_layers,
                                           const std::vector<uint32_t> & kv_on_device,
                                           int                           device,
                                           size_t                        plan_kv_per_layer) {
    ggml_sycl::placement_plan plan{};
    plan.kv_per_layer = plan_kv_per_layer;
    for (uint32_t l = 0; l < n_layers; ++l) {
        plan.kv_device[static_cast<int>(l)] = -1;
    }
    for (uint32_t l : kv_on_device) {
        plan.kv_device[static_cast<int>(l)] = device;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// 1. kimi-linear: 1 KV layer of 2, and the plan over-reports 2/2 on device.
//
// Measured pre-fix: buffer 294912 B, slice 147456 B, and
//   [KV-REMAP] ERROR: cache_k_l0 overflows layer alloc!
//              off_in_layer=0 + nbytes=294912 > la.size=147456
//
// The whole buffer belongs to the single KV-bearing layer.  Note the plan's
// own kv_per_layer here is the WRONG 147456: the layer mask describes this
// buffer, the plan is a file-scope global that can be stale, so the mask wins.
// ---------------------------------------------------------------------------
static void test_one_kv_layer_of_two() {
    printf("1. kimi-linear geometry: 1 KV layer of 2, plan claims 2/2 on device\n");

    const size_t   total_bytes = 294912;
    const uint32_t n_layers    = 2;

    ggml_sycl::placement_plan plan  = make_plan(n_layers, { 0, 1 }, /*device=*/0, /*plan_kv_per_layer=*/147456);
    const auto                slice = ggml_sycl::kv_slice_size::from_layer_mask(total_bytes, /*n_kv_layers=*/1);

    check_eq("slice.bytes", slice.bytes(), total_bytes);
    check_eq("slice.kv_layers", slice.kv_layers(), 1u);

    ggml_sycl::kv_tier_manager mgr;
    mgr.configure_from_plan(/*device=*/0, plan, n_layers, slice);

    check_eq("kv_per_layer", mgr.kv_per_layer(), total_bytes);
    check_eq("kv_layer_size(0)", mgr.kv_layer_size(0), total_bytes);

    const auto layout = mgr.compute_region_layout(total_bytes);
    check_true("layout covers layer 0", layout.size() >= 1);
    // The KV-bearing layer must be able to hold the whole buffer's worth of KV.
    check_true("layer 0 region fits the buffer", !layout.empty() && layout[0].size >= total_bytes);
}

// ---------------------------------------------------------------------------
// 2. The plan's device-layer count must NOT be the divisor.
//
// Pre-fix, kimi-linear (plan says 2/2 on device) overflowed while lfm2 (plan
// says 1/2) did not, with identical fixture geometry.  Same mask, different
// plan placement => the slice size must be identical.
// ---------------------------------------------------------------------------
static void test_slice_independent_of_plan_placement() {
    printf("2. slice size is independent of how many layers the plan puts on device\n");

    const size_t   total_bytes = 294912;
    const uint32_t n_layers    = 2;
    const auto     slice       = ggml_sycl::kv_slice_size::from_layer_mask(total_bytes, 1);

    ggml_sycl::placement_plan plan_two = make_plan(n_layers, { 0, 1 }, 0, 147456);
    ggml_sycl::placement_plan plan_one = make_plan(n_layers, { 0 }, 0, 147456);

    ggml_sycl::kv_tier_manager mgr_two;
    ggml_sycl::kv_tier_manager mgr_one;
    mgr_two.configure_from_plan(0, plan_two, n_layers, slice);
    mgr_one.configure_from_plan(0, plan_one, n_layers, slice);

    check_eq("2/2-on-device slice", mgr_two.kv_per_layer(), total_bytes);
    check_eq("1/2-on-device slice", mgr_one.kv_per_layer(), total_bytes);
    check_eq("both agree", mgr_two.kv_per_layer(), mgr_one.kv_per_layer());
}

// ---------------------------------------------------------------------------
// 3. All layers KV-bearing: the ordinary dense case must be unchanged.
// ---------------------------------------------------------------------------
static void test_all_layers_kv_bearing() {
    printf("3. all layers KV-bearing (dense model)\n");

    const uint32_t n_layers    = 4;
    const size_t   per_layer   = 65536;
    const size_t   total_bytes = per_layer * n_layers;

    const auto slice = ggml_sycl::kv_slice_size::from_layer_mask(total_bytes, n_layers);
    check_eq("slice.bytes", slice.bytes(), per_layer);

    ggml_sycl::placement_plan  plan = make_plan(n_layers, { 0, 1, 2, 3 }, 0, per_layer);
    ggml_sycl::kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, n_layers, slice);

    check_eq("kv_per_layer", mgr.kv_per_layer(), per_layer);
    check_eq("hot_layers", mgr.hot_layers(), n_layers);
    check_true("not tiered when everything is on device", !mgr.is_active());

    const auto layout = mgr.compute_region_layout(total_bytes);
    check_eq("layout entries", layout.size(), n_layers);
    size_t sum = 0;
    for (const auto & r : layout) {
        check_eq("region size", r.size, per_layer);
        sum += r.size;
    }
    check_eq("regions sum to the buffer", sum, total_bytes);
}

// ---------------------------------------------------------------------------
// 4. The guarded plan path: with no explicit layer mask, the planner's own
//    per-layer size is authoritative, and an insane one is rejected.
// ---------------------------------------------------------------------------
static void test_guarded_plan_path() {
    printf("4. guarded plan path (no layer mask)\n");

    const uint32_t n_layers    = 4;
    const size_t   total_bytes = 262144;

    {
        // Planner says 65536/layer.  No mask, so the planner wins.
        ggml_sycl::placement_plan plan  = make_plan(n_layers, { 0, 1, 2, 3 }, 0, 65536);
        const auto                slice = ggml_sycl::kv_slice_size::from_model_layers_unverified(total_bytes, n_layers);
        ggml_sycl::kv_tier_manager mgr;
        mgr.configure_from_plan(0, plan, n_layers, slice);
        check_eq("planner value used", mgr.kv_per_layer(), 65536u);
    }
    {
        // Planner's value exceeds the whole buffer — nonsense, fall back to the slice.
        ggml_sycl::placement_plan plan  = make_plan(n_layers, { 0, 1, 2, 3 }, 0, total_bytes * 4);
        const auto                slice = ggml_sycl::kv_slice_size::from_model_layers_unverified(total_bytes, n_layers);
        ggml_sycl::kv_tier_manager mgr;
        mgr.configure_from_plan(0, plan, n_layers, slice);
        check_eq("oversized planner value rejected", mgr.kv_per_layer(), total_bytes / n_layers);
    }
    {
        // Planner has nothing to say (kv_per_layer == 0) — fall back to the slice.
        ggml_sycl::placement_plan  plan  = make_plan(n_layers, { 0, 1 }, 0, 0);
        const auto                 slice = ggml_sycl::kv_slice_size::from_planner_layers(total_bytes, n_layers);
        ggml_sycl::kv_tier_manager mgr;
        mgr.configure_from_plan(0, plan, n_layers, slice);
        check_eq("empty planner value falls back to slice", mgr.kv_per_layer(), total_bytes / n_layers);
        check_eq("hot_layers from plan", mgr.hot_layers(), 2u);
        check_true("tiered when some layers are on host", mgr.is_active());
    }
}

// ---------------------------------------------------------------------------
// 5. Heterogeneous SWA sizing still works: SWA layers keep their own size and
//    full-attention layers get the slice.
// ---------------------------------------------------------------------------
static void test_swa_heterogeneous_sizes() {
    printf("5. heterogeneous SWA per-layer sizes\n");

    const uint32_t n_layers    = 4;
    const size_t   total_bytes = 262144;

    ggml_sycl::placement_plan plan = make_plan(n_layers, { 0, 1, 2, 3 }, 0, 65536);
    plan.kv_per_swa_layer          = 16384;
    plan.swa_layer_mask            = { false, true, false, true };

    ggml_sycl::kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, n_layers, ggml_sycl::kv_slice_size::from_planner_bytes(total_bytes, 65536, 2));

    check_eq("full-attn layer 0", mgr.kv_layer_size(0), 65536u);
    check_eq("swa layer 1", mgr.kv_layer_size(1), 16384u);
    check_eq("full-attn layer 2", mgr.kv_layer_size(2), 65536u);
    check_eq("swa layer 3", mgr.kv_layer_size(3), 16384u);
}

// ---------------------------------------------------------------------------
// 6. set_actual_layer_placement() reconciles PLACEMENT after allocation.  It
//    must not re-derive the slice from the layer-vector length — that was the
//    second unguarded division, and it ran after configure_from_plan() had
//    already got the size right, silently halving it again.
// ---------------------------------------------------------------------------
static void test_actual_placement_preserves_slice() {
    printf("6. set_actual_layer_placement preserves the slice size\n");

    const size_t   total_bytes = 294912;
    const uint32_t n_layers    = 2;

    ggml_sycl::placement_plan  plan = make_plan(n_layers, { 0 }, 0, 147456);
    ggml_sycl::kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, n_layers, ggml_sycl::kv_slice_size::from_layer_mask(total_bytes, 1));
    check_eq("slice before reconciliation", mgr.kv_per_layer(), total_bytes);

    mgr.set_actual_layer_placement(0, { true, false });

    check_eq("slice after reconciliation", mgr.kv_per_layer(), total_bytes);
    check_eq("layer 0 size after reconciliation", mgr.kv_layer_size(0), total_bytes);
    check_eq("hot_layers after reconciliation", mgr.hot_layers(), 1u);
    check_true("layer 0 is hot", mgr.is_hot(0));
    check_true("layer 1 is not hot", !mgr.is_hot(1));
}

// ---------------------------------------------------------------------------
// 7. The weight-aware path (no placement plan) must honour the slice too.
//    Both its branches: with device-resident weights, and the fall-through to
//    the budget-only configure() when the cache knows nothing.
// ---------------------------------------------------------------------------
static void test_weight_aware_path() {
    printf("7. weight-aware configuration honours the slice\n");

    const size_t   total_bytes = 294912;
    const uint32_t n_layers    = 2;
    const auto     slice       = ggml_sycl::kv_slice_size::from_layer_mask(total_bytes, 1);

    {
        // Layer 0 has device weights; budget fits one slice.
        g_stub_layer_vram_bytes = { 1024, 0 };
        ggml_sycl::kv_tier_manager mgr;
        mgr.configure_with_weights(0, n_layers, /*kv_vram_cap=*/total_bytes, slice);
        check_eq("weight-aware slice", mgr.kv_per_layer(), total_bytes);
        check_eq("weight-aware hot_layers", mgr.hot_layers(), 1u);
        check_true("layer 0 co-located with its weights", mgr.is_hot(0));
    }
    {
        // No layer has device weights -> falls through to budget-only configure().
        g_stub_layer_vram_bytes = {};
        ggml_sycl::kv_tier_manager mgr;
        mgr.configure_with_weights(0, n_layers, /*kv_vram_cap=*/total_bytes, slice);
        check_eq("budget-path slice", mgr.kv_per_layer(), total_bytes);
    }
    {
        // Zero-byte buffer is inert, not a division by zero.
        g_stub_layer_vram_bytes = {};
        ggml_sycl::kv_tier_manager mgr;
        mgr.configure_with_weights(0, n_layers, 0, ggml_sycl::kv_slice_size::from_layer_mask(0, 1));
        check_eq("empty buffer slice", mgr.kv_per_layer(), 0u);
        check_true("empty buffer is not tiered", !mgr.is_active());
    }
}

// ---------------------------------------------------------------------------
// 8. kv_slice_size invariants.  Every factory records the denominator it used,
//    and slice * kv_layers reconstitutes the buffer.  A caller that reaches for
//    the model layer count has to say so by name.
// ---------------------------------------------------------------------------
static void test_slice_size_invariants() {
    printf("8. kv_slice_size invariants\n");

    const auto mask = ggml_sycl::kv_slice_size::from_layer_mask(294912, 1);
    check_true("mask source", mask.source() == ggml_sycl::kv_slice_source::LAYER_MASK);
    check_eq("mask reconstitutes the buffer", mask.bytes() * mask.kv_layers(), mask.total_bytes());

    const auto planner_layers = ggml_sycl::kv_slice_size::from_planner_layers(262144, 4);
    check_true("planner-layers source", planner_layers.source() == ggml_sycl::kv_slice_source::PLANNER);
    check_eq("planner-layers reconstitutes the buffer", planner_layers.bytes() * planner_layers.kv_layers(),
             planner_layers.total_bytes());

    const auto planner_bytes = ggml_sycl::kv_slice_size::from_planner_bytes(262144, 65536, 4);
    check_true("planner-bytes source", planner_bytes.source() == ggml_sycl::kv_slice_source::PLANNER);
    check_eq("planner-bytes takes the planner size verbatim", planner_bytes.bytes(), 65536u);

    const auto unverified = ggml_sycl::kv_slice_size::from_model_layers_unverified(262144, 4);
    check_true("unverified source", unverified.source() == ggml_sycl::kv_slice_source::MODEL_LAYERS_UNVERIFIED);
    check_eq("unverified divides by the model layer count", unverified.bytes(), 65536u);

    const ggml_sycl::kv_slice_size empty;
    check_true("default source", empty.source() == ggml_sycl::kv_slice_source::UNKNOWN);
    check_eq("default bytes", empty.bytes(), 0u);

    // A zero KV-layer count means "unknown split", not "zero-byte slices" — a
    // zero slice would make every layer allocation get skipped.
    const auto no_layers = ggml_sycl::kv_slice_size::from_layer_mask(294912, 0);
    check_eq("zero KV layers => whole buffer is one slice", no_layers.bytes(), 294912u);
}

// ---------------------------------------------------------------------------
// 9. get_kv_tier_manager() hands out one instance per device.
// ---------------------------------------------------------------------------
static void test_per_device_singletons() {
    printf("9. per-device singletons are distinct\n");

    auto & a = ggml_sycl::get_kv_tier_manager(0);
    auto & b = ggml_sycl::get_kv_tier_manager(1);
    check_true("distinct instances", &a != &b);

    a.configure_from_plan(0, make_plan(2, { 0 }, 0, 0), 2, ggml_sycl::kv_slice_size::from_layer_mask(294912, 1));
    b.configure_from_plan(1, make_plan(2, { 0, 1 }, 1, 0), 2, ggml_sycl::kv_slice_size::from_layer_mask(131072, 2));

    check_eq("device 0 slice", ggml_sycl::get_kv_tier_manager(0).kv_per_layer(), 294912u);
    check_eq("device 1 slice", ggml_sycl::get_kv_tier_manager(1).kv_per_layer(), 65536u);
}

int main() {
    // Hermetic: GGML_SYCL_KV_HOT_LAYERS short-circuits placement in every
    // configure_*() entry point, so a stray value in the environment would
    // quietly change what this test measures.
    if (const char * env = std::getenv("GGML_SYCL_KV_HOT_LAYERS")) {
        printf("unsetting GGML_SYCL_KV_HOT_LAYERS=%s for a hermetic run\n", env);
        unsetenv("GGML_SYCL_KV_HOT_LAYERS");
    }

    printf("=== KV per-layer slice sizing (llama.cpp-2120) ===\n");

    test_one_kv_layer_of_two();
    test_slice_independent_of_plan_placement();
    test_all_layers_kv_bearing();
    test_guarded_plan_path();
    test_swa_heterogeneous_sizes();
    test_actual_placement_preserves_slice();
    test_weight_aware_path();
    test_slice_size_invariants();
    test_per_device_singletons();

    printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    if (g_failures > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
