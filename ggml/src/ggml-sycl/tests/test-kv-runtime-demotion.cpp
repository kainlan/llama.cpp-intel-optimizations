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

// Numeric variant printing got/want, per the test-kv-slice-sizing.cpp
// check_eq() precedent. Both operands are cast to long long so size_t and int
// fields share one macro without format-specifier mismatches.
#define CHECK_EQ(got, want, msg)                                                                                       \
    do {                                                                                                               \
        const long long got_v_  = (long long) (got);                                                                   \
        const long long want_v_ = (long long) (want);                                                                  \
        if (got_v_ != want_v_) {                                                                                       \
            std::fprintf(stderr, "FAIL: %s:%d: %s (got %lld, want %lld)\n", __FILE__, __LINE__, msg, got_v_, want_v_); \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using ggml_sycl::kv_demotion_input;
using ggml_sycl::kv_demotion_result;
using ggml_sycl::plan_runtime_kv_demotion;

// Helper: n_layers alternating geometry like GPT-OSS (even = full-attn, odd = SWA).
//
// llama.cpp-3aos (round 1 F3): kv_bytes_per_layer is now a REAL per-layer
// vector (kv_per_layer/kv_per_swa_layer scalars removed) -- this helper
// still builds a uniform-width vector (kv_full on every even layer, kv_swa
// on every odd one) so every existing case below keeps its original
// numbers; that uniformity is the TEST's choice, not something the
// production type assumes.
static kv_demotion_input make_input(size_t budget, size_t vram, size_t kv_full, size_t kv_swa, int n_layers) {
    kv_demotion_input in;
    in.vram_budget = budget;
    in.vram_bytes  = vram;
    in.kv_device.assign(n_layers, 0);  // all on device 0
    in.swa_layer_mask.assign(n_layers, 0);
    in.kv_bytes_per_layer.assign(n_layers, kv_full);
    for (int l = 1; l < n_layers; l += 2) {
        in.swa_layer_mask[l]     = 1;
        in.kv_bytes_per_layer[l] = kv_swa;
    }
    return in;
}

int main() {
    // 1. identity: already fits -> zero demotions, bytes unchanged
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 900, 100, 10, 4));
        CHECK(r.fits, "case 1: fits");
        CHECK(r.demoted_layers.empty(), "case 1: no demotions");
        CHECK_EQ(r.vram_bytes_after, 900, "case 1: vram_bytes_after unchanged");
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
        CHECK_EQ(r.demoted_layers.size(), 1, "case 3: exactly one demotion");
        CHECK_EQ(r.demoted_layers[0], 4, "case 3: demotes last full-attn layer (4)");
        CHECK_EQ(r.vram_bytes_after, 950, "case 3: vram_bytes_after reduced by one layer");
        CHECK_EQ(r.host_kv_bytes_added, 100, "case 3: host_kv_bytes_added == one layer");
    }
    // 4. multi-layer overshoot demotes latest-first until fit
    {
        auto r = plan_runtime_kv_demotion(make_input(1000, 1250, 100, 10, 6));
        CHECK(r.fits, "case 4: fits after multiple demotions");
        CHECK(r.demoted_layers == (std::vector<int>{ 4, 2, 0 }), "case 4: demotes latest-first 4,2,0");
        CHECK_EQ(r.vram_bytes_after, 950, "case 4: vram_bytes_after settles at 950");
    }
    // 5. SWA layers are never demoted, even when insufficient
    {
        auto r = plan_runtime_kv_demotion(make_input(100, 500, 100, 10, 4));
        // full-attn layers 0,2 demoted (200 recovered) -> 300 > 100 budget: no fit
        CHECK(!r.fits, "case 5: cannot fit without demoting SWA layers");
        // Non-vacuous: an empty demoted_layers would satisfy the loop below by
        // never iterating, so pin the count before checking parity.
        CHECK_EQ(r.demoted_layers.size(), 2, "case 5: both full-attn layers demoted");
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
    // 7. a single layer's KV exceeds the remaining total VRAM: refuse without
    // wrapping vram_bytes_after (size_t underflow would report a huge total
    // instead of "did not fit").
    {
        kv_demotion_input in;
        in.vram_budget = 0;
        in.vram_bytes  = 50;
        in.kv_bytes_per_layer.assign(3, 100);
        in.kv_device.assign(3, 0);
        in.swa_layer_mask.assign(3, 0);
        auto r = plan_runtime_kv_demotion(in);
        CHECK(!r.fits, "case 7: cannot fit without wrapping");
        CHECK_EQ(r.vram_bytes_after, 50, "case 7: vram_bytes_after unchanged, no underflow");
        CHECK_EQ(r.host_kv_bytes_added, 0, "case 7: no bytes added");
    }
    // 8. llama.cpp-3aos (round 1 F3): a device-resident, non-SWA layer with
    // kv_bytes_per_layer[l] == 0 (e.g. a SHARED layer that holds no
    // independent KV of its own) must never be demoted -- there is nothing
    // for it to give back, and counting it would silently manufacture
    // phantom VRAM savings.
    {
        auto in                  = make_input(1000, 1050, 100, 10, 6);
        in.kv_bytes_per_layer[4] = 0;  // layer 4 (full-attn slot) holds no KV of its own
        auto r                   = plan_runtime_kv_demotion(in);
        // Layer 4 cannot be demoted (0 bytes); the next full-attn layer (2)
        // is demoted instead to reach the same 950 target.
        CHECK(r.fits, "case 8: fits by demoting the next real full-attn layer");
        CHECK(r.demoted_layers == (std::vector<int>{ 2 }), "case 8: layer 4 (0 bytes) skipped, layer 2 demoted");
        CHECK_EQ(r.vram_bytes_after, 950, "case 8: vram_bytes_after reduced by layer 2's real bytes");
        CHECK_EQ(r.host_kv_bytes_added, 100, "case 8: host_kv_bytes_added == layer 2's bytes, not layer 4's 0");
    }
    std::printf("test-kv-runtime-demotion: all ok\n");
    return 0;
}
