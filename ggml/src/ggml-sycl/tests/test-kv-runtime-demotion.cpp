#include "../kv-runtime-demotion.hpp"

#include <cstdio>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously (llama.cpp-u2mz). Use an explicit check that
// always runs, per the test-zone-sizing.cpp precedent in this directory.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::kv_demotion_input;
using ggml_sycl::kv_demotion_result;
using ggml_sycl::plan_runtime_kv_demotion;

// Helper: n_layers alternating geometry like GPT-OSS (even = full-attn, odd = SWA).
static kv_demotion_input make_input(size_t budget, size_t vram, size_t kv_full, size_t kv_swa, int n_layers) {
    kv_demotion_input in;
    in.vram_budget      = budget;
    in.vram_bytes       = vram;
    in.kv_per_layer     = kv_full;
    in.kv_per_swa_layer = kv_swa;
    in.kv_device.assign(n_layers, 0);  // all on device 0
    in.swa_layer_mask.assign(n_layers, 0);
    for (int l = 1; l < n_layers; l += 2) {
        in.swa_layer_mask[l] = 1;
    }
    return in;
}

int main() {
    // 1. identity: already fits -> zero demotions, bytes unchanged
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 900, 100, 10, 4));
        CHECK(r.fits, "case 1: fits");
        CHECK(r.demoted_layers.empty(), "case 1: no demotions");
        CHECK(r.vram_bytes_after == 900, "case 1: vram_bytes_after unchanged");
    }
    // 2. exact boundary: vram_bytes == budget is ADMITTED with zero demotions
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1000, 100, 10, 4));
        CHECK(r.fits, "case 2: exact boundary fits");
        CHECK(r.demoted_layers.empty(), "case 2: no demotions at exact boundary");
    }
    // 3. one-layer overshoot demotes exactly the LAST full-attn layer
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1050, 100, 10, 6));
        CHECK(r.fits, "case 3: fits after one demotion");
        CHECK(r.demoted_layers.size() == 1, "case 3: exactly one demotion");
        CHECK(r.demoted_layers[0] == 4, "case 3: demotes last full-attn layer (4)");
        CHECK(r.vram_bytes_after == 950, "case 3: vram_bytes_after reduced by one layer");
        CHECK(r.host_kv_bytes_added == 100, "case 3: host_kv_bytes_added == one layer");
    }
    // 4. multi-layer overshoot demotes latest-first until fit
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1250, 100, 10, 6));
        CHECK(r.fits, "case 4: fits after multiple demotions");
        CHECK(r.demoted_layers == (std::vector<int>{ 4, 2, 0 }), "case 4: demotes latest-first 4,2,0");
        CHECK(r.vram_bytes_after == 950, "case 4: vram_bytes_after settles at 950");
    }
    // 5. SWA layers are never demoted, even when insufficient
    {
        auto r = plan_runtime_kv_demotion(make_input(100, 500, 100, 10, 4));
        // full-attn layers 0,2 demoted (200 recovered) -> 300 > 100 budget: no fit
        CHECK(!r.fits, "case 5: cannot fit without demoting SWA layers");
        for (int l : r.demoted_layers) {
            CHECK(l % 2 == 0, "case 5: only full-attn (even) layers demoted");
        }
    }
    // 6. layers already on host are not re-demoted or double-counted
    {
        auto in         = make_input(1000, 1050, 100, 10, 6);
        in.kv_device[4] = -1;  // pre-demoted
        auto r          = plan_runtime_kv_demotion(in);
        CHECK(r.fits, "case 6: fits");
        CHECK(r.demoted_layers == (std::vector<int>{ 2 }), "case 6: only layer 2 newly demoted");
    }
    printf("test-kv-runtime-demotion: all ok\n");
    return 0;
}
