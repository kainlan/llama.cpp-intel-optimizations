#include "../kv-runtime-demotion.hpp"
#include "../tlsf-allocator.hpp"

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

using ggml_sycl::kv_admission_mismatch;
using ggml_sycl::kv_alloc_slack_per_layer;
using ggml_sycl::kv_buffer_layer_owner;
using ggml_sycl::kv_demotion_input;
using ggml_sycl::kv_demotion_result;
using ggml_sycl::kv_device_fit_input;
using ggml_sycl::kv_hot_layers_override_active;
using ggml_sycl::kv_reads_device_arena;
using ggml_sycl::kv_residency_input;
using ggml_sycl::kv_residency_needs_refit;
using ggml_sycl::kv_shape;
using ggml_sycl::kv_shape_changed;
using ggml_sycl::kv_vram_available;
using ggml_sycl::kv_weight_capacity;
using ggml_sycl::layer_block_kv_device;
using ggml_sycl::plan_device_kv_fit;
using ggml_sycl::plan_runtime_kv_demotion;
using ggml_sycl::plan_runtime_kv_residency;
using ggml_sycl::tlsf_allocator;

// Helper: n_layers alternating geometry like GPT-OSS (even = full-attn, odd = SWA).
//
// llama.cpp-3aos: layer_kv_bytes is now a REAL per-layer
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
    in.layer_kv_bytes.assign(n_layers, kv_full);
    for (int l = 1; l < n_layers; l += 2) {
        in.swa_layer_mask[l] = 1;
        in.layer_kv_bytes[l] = kv_swa;
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
        in.layer_kv_bytes.assign(3, 100);
        in.kv_device.assign(3, 0);
        in.swa_layer_mask.assign(3, 0);
        auto r = plan_runtime_kv_demotion(in);
        CHECK(!r.fits, "case 7: cannot fit without wrapping");
        CHECK_EQ(r.vram_bytes_after, 50, "case 7: vram_bytes_after unchanged, no underflow");
        CHECK_EQ(r.host_kv_bytes_added, 0, "case 7: no bytes added");
    }
    // 8. llama.cpp-3aos: a device-resident, non-SWA layer with
    // layer_kv_bytes[l] == 0 (e.g. a SHARED layer that holds no
    // independent KV of its own) must never be demoted -- there is nothing
    // for it to give back, and counting it would silently manufacture
    // phantom VRAM savings.
    {
        auto in              = make_input(1000, 1050, 100, 10, 6);
        in.layer_kv_bytes[4] = 0;  // layer 4 (full-attn slot) holds no KV of its own
        auto r               = plan_runtime_kv_demotion(in);
        // Layer 4 cannot be demoted (0 bytes); the next full-attn layer (2)
        // is demoted instead to reach the same 950 target.
        CHECK(r.fits, "case 8: fits by demoting the next real full-attn layer");
        CHECK(r.demoted_layers == (std::vector<int>{ 2 }), "case 8: layer 4 (0 bytes) skipped, layer 2 demoted");
        CHECK_EQ(r.vram_bytes_after, 950, "case 8: vram_bytes_after reduced by layer 2's real bytes");
        CHECK_EQ(r.host_kv_bytes_added, 100, "case 8: host_kv_bytes_added == layer 2's bytes, not layer 4's 0");
    }
    // 9. The KV+weight capacity of a device is the shared arena zone when one
    // exists, never the larger budget the zone was carved from: the B70 at
    // GGML_SYCL_VRAM_BUDGET_PCT=22 has a 5136 MB budget but a 3856 MB zone.
    {
        const size_t mb = 1024 * 1024;
        CHECK_EQ(kv_weight_capacity(5136 * mb, 3856 * mb), 3856 * mb, "case 9: zone below budget wins");
        CHECK_EQ(kv_weight_capacity(3000 * mb, 3856 * mb), 3000 * mb, "case 9: budget below zone wins");
        CHECK_EQ(kv_weight_capacity(5136 * mb, 0), 5136 * mb, "case 9: no arena zone -> budget");
    }
    // 10. The measured split (Mistral 7B, n_ctx=2048): device 0 holds layers
    // 0-29 with 3683.8 MB of weights in a 3856 MB zone, device 1 holds 30-31.
    // As plan_runtime_kv_residency() calls it, the capacity is the live KV
    // headroom the weights left (172.2 MB) less the slack of the 30 resident
    // layers, which fits 21 of device 0's 8 MB layers: exactly 21..29 demote,
    // and device 1's layers are never touched by device 0's pass.
    {
        const size_t        mb = 1024 * 1024;
        kv_device_fit_input in;
        in.device   = 0;
        in.capacity = (3856 * mb - (3683 * mb + 819 * mb / 1000)) - 30 * kv_alloc_slack_per_layer;
        in.layer_kv_bytes.assign(32, 8 * mb);
        in.kv_device.assign(32, 0);
        in.kv_device[30] = 1;
        in.kv_device[31] = 1;
        in.swa_layer_mask.assign(32, 0);
        auto r = plan_device_kv_fit(in);
        CHECK(r.fits, "case 10: device 0 fits after demotion");
        CHECK_EQ(r.demoted_layers.size(), 9, "case 10: nine layers demoted");
        CHECK_EQ(r.demoted_layers.front(), 29, "case 10: latest device-0 layer first");
        CHECK_EQ(r.demoted_layers.back(), 21, "case 10: stops at layer 21");
        CHECK_EQ(r.host_kv_bytes_added, 72 * mb, "case 10: 72 MB of KV moves to host");

        in.device   = 1;
        in.capacity = (676 * mb - 234 * mb) - 2 * kv_alloc_slack_per_layer;
        auto r1     = plan_device_kv_fit(in);
        CHECK(r1.fits, "case 10: device 1 fits");
        CHECK(r1.demoted_layers.empty(), "case 10: device 1 demotes nothing");
    }
    // 11. The same split at n_ctx=1024 (4 MB layers) fits with no demotion.
    {
        const size_t        mb = 1024 * 1024;
        kv_device_fit_input in;
        in.device   = 0;
        in.capacity = (3856 * mb - (3683 * mb + 819 * mb / 1000)) - 30 * kv_alloc_slack_per_layer;
        in.layer_kv_bytes.assign(32, 4 * mb);
        in.kv_device.assign(32, 0);
        in.kv_device[30] = 1;
        in.kv_device[31] = 1;
        in.swa_layer_mask.assign(32, 0);
        auto r = plan_device_kv_fit(in);
        CHECK(r.fits, "case 11: fits");
        CHECK(r.demoted_layers.empty(), "case 11: nothing demoted");
    }
    // 12. No KV headroom left at all (the weights filled the zone): every layer
    // moves to the host tier and the context still fits, with no KV on the
    // device -- never shrink context.
    {
        kv_device_fit_input in;
        in.device   = 0;
        in.capacity = 0;
        in.layer_kv_bytes.assign(2, 10);
        in.kv_device.assign(2, 0);
        in.swa_layer_mask.assign(2, 0);
        auto r = plan_device_kv_fit(in);
        CHECK(r.fits, "case 12: fits with every layer on the host tier");
        CHECK_EQ(r.demoted_layers.size(), 2, "case 12: both layers demoted");
        CHECK_EQ(r.vram_bytes_after, 0, "case 12: no KV left on the device");
    }
    // 13. A block's KV owner: the one device every KV-holding layer uses, -2
    // when they disagree (a demoted layer inside a device block), -1 with no
    // KV at all. Layers with no KV of their own do not vote.
    {
        std::vector<int>    kv_dev   = { 0, 0, -1, 0, 1, 1 };
        std::vector<size_t> kv_bytes = { 8, 8, 8, 0, 8, 8 };
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 0, 1), 0, "case 13: uniform block");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 0, 2), -2, "case 13: demoted layer makes it mixed");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 3, 3), -1, "case 13: no KV -> -1");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 3, 5), 1, "case 13: KV-less layer does not vote");
        CHECK_EQ(layer_block_kv_device(kv_dev, kv_bytes, 4, 9), 1, "case 13: range clamped to the vectors");
    }
    // 14. Only a new KV shape re-decides residency. The load-time plan always
    // does; a republish that changes nothing the KV size depends on (the auto
    // micro-batch trial) keeps the published residency.
    {
        kv_shape published;
        published.runtime   = true;
        published.n_ctx     = 2048;
        published.n_seq_max = 1;
        kv_shape next       = published;
        CHECK(!kv_shape_changed(published, next), "case 14: same shape keeps residency");
        kv_shape load = published;
        load.runtime  = false;
        CHECK(kv_shape_changed(load, next), "case 14: the load-time plan is always re-decided");
        next.n_ctx = 512;
        CHECK(kv_shape_changed(published, next), "case 14: n_ctx");
        next           = published;
        next.n_seq_max = 4;
        CHECK(kv_shape_changed(published, next), "case 14: n_seq_max");
        next            = published;
        next.kv_unified = true;
        CHECK(kv_shape_changed(published, next), "case 14: kv_unified");
        next          = published;
        next.swa_full = true;
        CHECK(kv_shape_changed(published, next), "case 14: swa_full");
    }
    // 15. Demote at a large context, then a small one gets the load-time
    // residency back: the split with 172.2 MB of live headroom on device 0.
    // The slack (30 resident layers x 64 KiB) does not change the count.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.load_kv_device[30] = 1;
        in.load_kv_device[31] = 1;
        in.swa_layer_mask.assign(32, 0);
        in.devices   = { 0, 1 };
        in.available = { 172 * mb + 205 * mb / 1000, 442 * mb };
        in.layer_kv_bytes.assign(32, 8 * mb);
        auto big = plan_runtime_kv_residency(in);
        CHECK(big.fits, "case 15: n_ctx=2048 fits after demotion");
        CHECK_EQ(big.per_device[0].demoted_layers.size(), 9, "case 15: nine layers demoted on device 0");
        CHECK(big.per_device[1].demoted_layers.empty(), "case 15: device 1 keeps its layers");
        for (int l = 0; l < 32; ++l) {
            const int want = l >= 21 && l <= 29 ? -1 : in.load_kv_device[l];
            CHECK_EQ(big.kv_device[l], want, "case 15: layers 21..29 on host, the rest where the planner put them");
        }

        in.layer_kv_bytes.assign(32, 2 * mb);  // n_ctx=512
        auto small = plan_runtime_kv_residency(in);
        CHECK(small.fits, "case 15: n_ctx=512 fits");
        CHECK(small.kv_device == in.load_kv_device, "case 15: the load-time residency comes back");
    }
    // 16. KV that does not fit with every full-attention layer demoted places
    // its SWA layers on the host tier too, full-attention first, instead of
    // refusing the context. When full-attention demotion is enough, SWA stays.
    {
        kv_residency_input in;
        in.load_kv_device = { 0, 0 };
        in.layer_kv_bytes = { 100, 100 };
        in.swa_layer_mask = { 1, 0 };
        in.devices        = { 0 };
        in.available      = { 50 + 2 * kv_alloc_slack_per_layer };
        auto r            = plan_runtime_kv_residency(in);
        CHECK(r.fits, "case 16: SWA KV over headroom is placed on the host tier, not refused");
        CHECK(r.per_device[0].demoted_layers == std::vector<int>({ 1, 0 }),
              "case 16: full-attention layer first, then the SWA layer");
        CHECK(r.kv_device == std::vector<int>({ -1, -1 }), "case 16: both layers on the host tier");

        in.available = { 100 + 2 * kv_alloc_slack_per_layer };
        auto enough  = plan_runtime_kv_residency(in);
        CHECK(enough.fits && enough.kv_device == std::vector<int>({ 0, -1 }),
              "case 16: the SWA layer stays when full-attention demotion is enough");
    }
    // 17. The slack is enough for the arena allocator: layers admitted against
    // exactly sum(bytes) + n * slack of headroom all place, one allocation at a
    // time and in two KV buffers, at the tiered allocator's 512-byte alignment.
    {
        const size_t        mb     = 1024 * 1024;
        std::vector<size_t> layers = { 8 * mb + 1, 8 * mb + 511, 128 * mb + 300, 3 * mb / 2 + 7, 1, 256 };
        size_t              sum    = 0;
        for (size_t b : layers) {
            sum += b;
        }
        tlsf_allocator arena(sum + layers.size() * kv_alloc_slack_per_layer);
        for (size_t pass = 0; pass < 2; ++pass) {  // full-attention buffer, then SWA buffer
            for (size_t i = pass; i < layers.size(); i += 2) {
                const size_t request = (layers[i] + 511) & ~size_t(511);
                const size_t before  = arena.used();
                CHECK(arena.allocate(request) != SIZE_MAX, "case 17: an admitted layer places");
                CHECK(arena.used() - before <= layers[i] + kv_alloc_slack_per_layer,
                      "case 17: one allocation costs at most its bytes plus the slack");
            }
        }
    }
    // 18. Two same-shape contexts against one device. The first is admitted
    // and allocates its KV; the second, not yet admitted, re-fits against the
    // headroom A left and places its overflow on the host tier through the
    // plan -- and the allocator's backstop never fires.
    {
        const size_t       mb = 1024 * 1024;
        kv_residency_input in;
        in.load_kv_device.assign(32, 0);
        in.layer_kv_bytes.assign(32, 8 * mb);
        in.swa_layer_mask.assign(32, 0);
        in.devices   = { 0 };
        in.available = { 300 * mb };
        kv_shape load;
        kv_shape shape;
        shape.runtime   = true;
        shape.n_ctx     = 2048;
        shape.n_seq_max = 1;

        // Admission's demand on device 0: every resident layer plus its slack.
        auto demand = [&](const std::vector<int> & kv_device) {
            size_t bytes = 0;
            for (size_t l = 0; l < kv_device.size(); ++l) {
                bytes += kv_device[l] == 0 ? in.layer_kv_bytes[l] + kv_alloc_slack_per_layer : 0;
            }
            return bytes;
        };

        // Context A: the load-time plan is re-decided and all 32 layers fit.
        CHECK(kv_residency_needs_refit(load, shape, false), "case 18: A is decided");
        auto a = plan_runtime_kv_residency(in);
        CHECK(a.fits && a.per_device[0].demoted_layers.empty(), "case 18: A keeps every layer on the device");
        size_t a_bytes = 0;
        for (int l = 0; l < 32; ++l) {
            a_bytes += a.kv_device[l] == 0 ? in.layer_kv_bytes[l] : 0;
        }
        const size_t shared_available = in.available[0] - a_bytes;  // A's KV is allocated

        // Context B: same shape, published residency is A's.
        const bool       refit = kv_residency_needs_refit(shape, shape, false);
        std::vector<int> b_kv  = a.kv_device;
        if (refit) {
            in.available = { shared_available };
            auto b       = plan_runtime_kv_residency(in);
            CHECK(b.fits, "case 18: B fits after demotion");
            b_kv = b.kv_device;
        }
        size_t b_bytes = 0;
        for (int l = 0; l < 32; ++l) {
            b_bytes += b_kv[l] == 0 ? in.layer_kv_bytes[l] : 0;
        }
        CHECK(!kv_admission_mismatch(b_bytes, shared_available), "case 18: the allocator backstop never fires");
        CHECK(refit, "case 18: the second same-shape context re-fits");
        CHECK(demand(b_kv) <= shared_available, "case 18: B's admitted demand fits");
        CHECK(b_bytes < a_bytes, "case 18: B's overflow is on the host tier");
    }
    // 19. An admitted context's same-shape republish (its micro-batch trial;
    // its KV is allocated, so the live headroom no longer covers it) keeps its
    // residency. A new context always re-fits, from the load-time residency,
    // so it gets full device residency when that fits even if the published
    // residency is more demoted.
    {
        kv_shape shape;
        shape.runtime   = true;
        shape.n_ctx     = 2048;
        shape.n_seq_max = 1;
        CHECK(!kv_residency_needs_refit(shape, shape, true),
              "case 19: an admitted context's republish keeps its residency");
        CHECK(kv_residency_needs_refit(shape, shape, false), "case 19: a new context re-fits");

        kv_residency_input in;
        in.load_kv_device = { 0, 0, 0, 0 };
        in.layer_kv_bytes = { 64, 64, 64, 64 };
        in.swa_layer_mask = { 0, 0, 0, 0 };
        in.devices        = { 0 };
        in.available      = { 256 + 4 * kv_alloc_slack_per_layer };

        const std::vector<int> published = { 0, 0, -1, -1 };  // an earlier, larger context's residency
        auto                   r         = plan_runtime_kv_residency(in);
        CHECK(r.fits && r.kv_device == in.load_kv_device,
              "case 19: a new context that fits gets full device residency");
        CHECK(r.kv_device != published, "case 19: it does not inherit the published residency");
    }
    // 20. A layer of device 0's KV buffer that the plan gives to device 1 is in
    // device 1's VRAM, although device 0's tier layout marks it off-device.
    {
        CHECK(kv_buffer_layer_owner(true, 1, false, 0, false) == 1,
              "case 20: another device's layer is device VRAM, not host memory");
        CHECK(kv_buffer_layer_owner(true, 0, true, 0, false) == 0, "case 20: this device's layer");
        CHECK(kv_buffer_layer_owner(true, -1, true, 0, false) == -1, "case 20: a demoted layer is host memory");
        CHECK(kv_buffer_layer_owner(false, -1, true, 0, false) == 0, "case 20: without a plan the layout decides");
        CHECK(kv_buffer_layer_owner(true, 0, true, 0, true) == -1, "case 20: GGML_SYCL_KV_HOST=1 is host memory");
    }
    // 21. A full arena KV zone leaves no KV headroom; only a device without an
    // arena uses the budget-based headroom.
    {
        CHECK_EQ(kv_vram_available(true, 0, 4096), 0, "case 21: a full KV zone is not a missing arena");
        CHECK_EQ(kv_vram_available(true, 512, 4096), 512, "case 21: the arena's KV zone");
        CHECK_EQ(kv_vram_available(false, 0, 4096), 4096, "case 21: no arena, the budget headroom");
    }
    // 22. On a split, each device's backend publishes the same context and
    // re-fits it. All of them publish before llama_kv_cache allocates that KV,
    // so each re-fit restarts from the load residency against headroom that
    // still includes it (plan_runtime_kv_residency is a pure function, so the
    // same input gives the same answer). Were one to run after the KV is
    // allocated, the headroom would no longer include it and it would demote
    // more; the publish ordering is pinned on the source
    // (test-sycl-kv-layer-sizing-source.py).
    {
        kv_residency_input in;
        in.load_kv_device = { 0, 0, 0, 1, 1, 1 };
        in.layer_kv_bytes = { 64, 64, 64, 64, 64, 64 };
        in.swa_layer_mask = { 0, 0, 0, 0, 0, 0 };
        in.devices        = { 0, 1 };
        in.available      = { 128 + 3 * kv_alloc_slack_per_layer, 192 + 3 * kv_alloc_slack_per_layer };
        auto before       = plan_runtime_kv_residency(in);
        CHECK(before.fits, "case 22: the re-fit before allocation fits");
        CHECK_EQ(before.per_device[0].demoted_layers.size(), 1, "case 22: device 0 demotes one layer");

        in.available[0] -= 2 * 64;  // after allocation: the headroom no longer includes the two resident layers
        auto late = plan_runtime_kv_residency(in);
        CHECK(late.per_device[0].demoted_layers.size() > before.per_device[0].demoted_layers.size(),
              "case 22: a re-fit after allocation would demote more");
    }
    // 23. GGML_SYCL_KV_HOT_LAYERS is read by value, as the tier manager reads
    // it: a count >= 0 overrides, -1 (or unset) does not.
    {
        CHECK(!kv_hot_layers_override_active(nullptr), "case 23: unset");
        CHECK(!kv_hot_layers_override_active("-1"), "case 23: -1 means off");
        CHECK(kv_hot_layers_override_active("0"), "case 23: 0 hot layers is an override");
        CHECK(kv_hot_layers_override_active("16"), "case 23: 16 hot layers");
    }
    // 24. Only a multi-device plan in GLOBAL cache mode reads no per-device
    // arena; every other combination reads the device's own zones.
    {
        CHECK(!kv_reads_device_arena(true, true), "case 24: multi-device GLOBAL has one zone for all devices");
        CHECK(kv_reads_device_arena(true, false), "case 24: multi-device PER_DEVICE");
        CHECK(kv_reads_device_arena(false, true), "case 24: single-device GLOBAL");
        CHECK(kv_reads_device_arena(false, false), "case 24: single-device PER_DEVICE");
    }
    std::printf("test-kv-runtime-demotion: all ok\n");
    return 0;
}
