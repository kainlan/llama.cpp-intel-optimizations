// Regression test for the SYCL KV-cache planner's per-layer sizing
// (llama.cpp-3aos).
//
// Two bugs, one root: placement_kv_info's KV-budget formulas assumed a
// SINGLE (n_embd_k_gqa, n_embd_v_gqa) width pair plus a single SWA/non-SWA
// mask, and kv_bytes_per_swa_layer() hardcoded n_seq_max=1.
//
//   Bug 1 (n_seq_max): llama_kv_cache_iswa allocates
//   GGML_PAD(min(kv_size, n_swa * n_seq_max + n_ubatch), 256) cells per SWA
//   layer, but the planner's formula used n_swa + n_ubatch -- correct only
//   at n_seq_max=1. Any SWA model served with `--parallel > 1` was
//   under-budgeted (Gemma 4 E4B at -c 4096 --parallel 4: planner 1024 cells
//   vs llama's 2560 -> "[KV-REMAP] cache_k_l0 overflows layer alloc!").
//
//   Bug 2 (width): Gemma 4 E4B is heterogeneous -- full-attention layers
//   are WIDER than SWA layers (1024 vs 512 elements on E4B) and a trailing
//   block of layers has NO K/V of their own at all (they reuse an earlier
//   layer's -- llama_hparams::has_kv() == false). A single global width
//   pair cannot represent that; the old formula used layer 0's (SWA, 512)
//   width for every layer, including full-attention ones, and had no way
//   to charge a SHARED layer 0 bytes.
//
// This test exercises the REAL ggml_sycl::placement_kv_info and
// ggml_sycl::placement_plan -- no mock reimplementation. Both structs are
// built by hand (no model load, no device, no queue): the arithmetic under
// test is pure per-layer bookkeeping and needs no GPU.
//
// Written RED against the pre-fix tree (1d577b95c): see
// scratchpad/3aos/red-check.cpp for a standalone (non-SYCL, no ggml-sycl
// dependency) harness that runs the OLD kv_bytes_per_layer()/
// kv_bytes_per_swa_layer() formulas, copied verbatim from that commit's
// unified-cache.hpp, against the same shapes this file checks --
// mismatches: gemma4 l0 (SWA) @ n_seq_max=4 got 2,097,152 want 5,242,880;
// gemma4 l5 (FULL) got 8,388,608 want 16,777,216; gemma4 l24 (SHARED) got
// 2,097,152 want 0. The base struct has no kv_bytes_for_layer()/n_seq_max/
// per-layer arrays at all, so this file's assertions against those symbols
// could not even compile there -- that absence is itself part of the RED.

#include "../unified-cache.hpp"

#include <cstdio>
#include <vector>

using ggml_sycl::placement_kv_info;
using ggml_sycl::placement_plan;

static int g_checks = 0;
static int g_fail   = 0;

static void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_fail++;
        printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

// ---------------------------------------------------------------------------
// Shape builders. Both models' per-layer arrays are built the same way
// llama_model_sycl_build_kv_layer_arrays() (src/llama-model.cpp) builds
// them from real hparams -- SHARED (no K/V) gets width 0, SWA/FULL get
// their real per-layer width -- but hand-populated here from the header
// values this ticket's own investigation read (see the module comment in
// llama.cpp-3aos's task record for the GGUF header reads this cites).
// ---------------------------------------------------------------------------

// Gemma 4 E4B: 42 layers. Sliding-window pattern is "full every 6th"
// (0-indexed: full at 5, 11, 17, 23, 29, 35, 41), but layers 24-41 (18 of
// them, shared_kv_layers=18) have NO K/V tensors of their own regardless of
// where the raw pattern would place them -- they reuse an earlier layer's
// KV (llama_hparams::has_kv() == false for il >= n_layer_kv_from_start=24).
// So among layers 0-23: full-attention are {5, 11, 17, 23} (4 layers,
// width 1024 = 2*512), the remaining 20 are SWA (width 512 = 2*256).
// Layers 24-41 are SHARED (width 0) regardless of the pattern bit.
static placement_kv_info make_gemma4_e4b(uint32_t n_ctx, uint32_t n_ubatch, uint32_t n_seq_max) {
    placement_kv_info kv{};
    kv.n_layer      = 42;
    kv.n_embd_k_gqa = 1024;  // FULL-attention width fallback (n_embd_k_gqa_max())
    kv.n_embd_v_gqa = 1024;
    kv.n_ctx        = n_ctx;
    kv.n_ubatch     = n_ubatch;
    kv.n_seq_max    = n_seq_max;
    kv.n_swa        = 512;

    kv.layer_kind.assign(42, GGML_SYCL_KV_LAYER_SWA);
    kv.layer_k_width.assign(42, 512);
    kv.layer_v_width.assign(42, 512);
    for (uint32_t il : { 5u, 11u, 17u, 23u }) {
        kv.layer_kind[il]    = GGML_SYCL_KV_LAYER_FULL;
        kv.layer_k_width[il] = 1024;
        kv.layer_v_width[il] = 1024;
    }
    for (uint32_t il = 24; il < 42; ++il) {
        kv.layer_kind[il]    = GGML_SYCL_KV_LAYER_SHARED;
        kv.layer_k_width[il] = 0;
        kv.layer_v_width[il] = 0;
    }
    kv.n_swa_layers = 20;  // layers 0-23 minus the 4 full ones, for the legacy aggregate path
    return kv;
}

// GPT-OSS 20B: 24 layers, uniform width (key_length=value_length=64,
// head_count_kv=8 -> n_embd_k_gqa=n_embd_v_gqa=64*8=512 on every layer, SWA
// and full alike -- openai-moe.cpp never overrides n_embd_head_k_swa, so it
// stays equal to n_embd_head_k_full). n_swa=128, swa_period=2,
// dense_first=false (set_swa_pattern's default) -> is_swa(il) == (il % 2 ==
// 0): even layers SWA, odd layers full. No shared-KV layers
// (n_layer_kv_from_start left at its -1 default -- has_kv() is true for
// every layer).
static placement_kv_info make_gptoss_20b(uint32_t n_ctx, uint32_t n_ubatch, uint32_t n_seq_max) {
    placement_kv_info kv{};
    kv.n_layer      = 24;
    kv.n_embd_k_gqa = 512;
    kv.n_embd_v_gqa = 512;
    kv.n_ctx        = n_ctx;
    kv.n_ubatch     = n_ubatch;
    kv.n_seq_max    = n_seq_max;
    kv.n_swa        = 128;

    kv.layer_kind.resize(24);
    kv.layer_k_width.assign(24, 512);
    kv.layer_v_width.assign(24, 512);
    for (uint32_t il = 0; il < 24; ++il) {
        kv.layer_kind[il] = (il % 2 == 0) ? GGML_SYCL_KV_LAYER_SWA : GGML_SYCL_KV_LAYER_FULL;
    }
    kv.n_swa_layers = 12;
    return kv;
}

// ---------------------------------------------------------------------------
// (a) Gemma 4 E4B at n_ctx=4096, n_ubatch=512, n_seq_max=4 (the reported
//     `-c 4096 --parallel 4` shape).
// ---------------------------------------------------------------------------
static void test_gemma4_n_seq_max_4() {
    printf("(a) gemma4 E4B @ n_ctx=4096 n_ubatch=512 n_seq_max=4\n");
    placement_kv_info kv = make_gemma4_e4b(4096, 512, 4);

    // l0 (SWA): cells = pad256(min(4096, 512*4 + 512)) = pad256(min(4096, 2560)) = 2560.
    // bytes = 2560 * (512+512) * 2 = 5,242,880.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 5242880u);

    // l5 (FULL): bytes = 4096 * (1024+1024) * 2 = 16,777,216.
    check_eq("l5 (FULL)", kv.kv_bytes_for_layer(5), 16777216u);

    // l24..l41 (SHARED): 0.
    for (uint32_t il = 24; il < 42; ++il) {
        check_eq("l24..41 (SHARED)", kv.kv_bytes_for_layer(il), 0u);
    }

    // Total: 20 SWA layers + 4 FULL layers + 18 SHARED layers.
    size_t total = 0;
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        total += kv.kv_bytes_for_layer(il);
    }
    check_eq("total", total, 20u * 5242880u + 4u * 16777216u);
}

// ---------------------------------------------------------------------------
// (b) Same shape at n_seq_max=1 -- l0 (SWA) reduces to the OLD number
//     (2,097,152), which is correct only at n_seq_max=1.
// ---------------------------------------------------------------------------
static void test_gemma4_n_seq_max_1() {
    printf("(b) gemma4 E4B @ n_seq_max=1 (the old, n_seq_max=1-only number)\n");
    placement_kv_info kv = make_gemma4_e4b(4096, 512, 1);
    // cells = pad256(min(4096, 512*1 + 512)) = pad256(1024) = 1024.
    // bytes = 1024 * 1024 * 2 = 2,097,152.
    check_eq("l0 (SWA) @ n_seq_max=1", kv.kv_bytes_for_layer(0), 2097152u);
}

// ---------------------------------------------------------------------------
// (c) GPT-OSS 20B @ n_seq_max=4 -- uniform width, alternating SWA/full, no
//     SHARED layers.
// ---------------------------------------------------------------------------
static void test_gptoss_n_seq_max_4() {
    printf("(c) gpt-oss-20b @ n_ctx=4096 n_ubatch=512 n_seq_max=4\n");
    placement_kv_info kv = make_gptoss_20b(4096, 512, 4);

    // SWA (even layers): cells = pad256(min(4096, 128*4 + 512)) = pad256(1024) = 1024.
    // bytes = 1024 * (512+512) * 2 = 2,097,152.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 2097152u);

    // FULL (odd layers): bytes = 4096 * 1024 * 2 = 8,388,608.
    check_eq("l1 (FULL)", kv.kv_bytes_for_layer(1), 8388608u);

    size_t total = 0;
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        total += kv.kv_bytes_for_layer(il);
    }
    check_eq("total", total, 12u * 2097152u + 12u * 8388608u);
}

// ---------------------------------------------------------------------------
// (d) n_ctx SMALLER than n_swa * n_seq_max + n_ubatch, so min() binds on
//     n_ctx instead of the SWA window. Reuses the gemma4 shape at
//     n_ctx=2048 (window would be 2560).
// ---------------------------------------------------------------------------
static void test_min_binds_on_small_n_ctx() {
    printf("(d) gemma4 E4B @ n_ctx=2048 < window (2560) -- min() binds on n_ctx\n");
    placement_kv_info kv = make_gemma4_e4b(2048, 512, 4);

    // cells = pad256(min(2048, 2560)) = pad256(2048) = 2048 (already aligned).
    check_eq("l0 (SWA), n_ctx-bound", kv.kv_bytes_for_layer(0), 2048ull * 1024 * 2);
    // FULL layer scales with n_ctx directly, same as always.
    check_eq("l5 (FULL), n_ctx-bound", kv.kv_bytes_for_layer(5), 2048ull * 2048 * 2);
}

// ---------------------------------------------------------------------------
// (e) The aggregate estimators (kv_bytes_per_layer/kv_bytes_per_swa_layer)
//     now consume n_seq_max too, independent of whether per-layer arrays
//     are populated -- fix (a) from the ticket, checked in isolation.
// ---------------------------------------------------------------------------
static void test_aggregate_uses_n_seq_max() {
    printf("(e) kv_bytes_per_swa_layer() consumes n_seq_max\n");
    placement_kv_info kv{};
    kv.n_layer      = 1;
    kv.n_embd_k_gqa = 512;
    kv.n_embd_v_gqa = 512;
    kv.n_ctx        = 4096;
    kv.n_ubatch     = 512;
    kv.n_swa        = 512;

    kv.n_seq_max = 1;
    check_eq("aggregate @ n_seq_max=1", kv.kv_bytes_per_swa_layer(), 2097152u);
    kv.n_seq_max = 4;
    check_eq("aggregate @ n_seq_max=4", kv.kv_bytes_per_swa_layer(), 5242880u);
}

// ---------------------------------------------------------------------------
// (f) Fallback: when layer_kind/layer_k_width/layer_v_width are empty (an
//     inventory built before this ticket, or a homogeneous model),
//     kv_bytes_for_layer() must match the legacy is_swa_layer()-based split
//     exactly -- no silent behavior change for callers that never adopt
//     per-layer truth.
// ---------------------------------------------------------------------------
static void test_fallback_without_per_layer_arrays() {
    printf("(f) kv_bytes_for_layer() falls back to the legacy split when unpopulated\n");
    placement_kv_info kv{};
    kv.n_layer        = 2;
    kv.n_embd_k_gqa   = 512;
    kv.n_embd_v_gqa   = 512;
    kv.n_ctx          = 4096;
    kv.n_ubatch       = 512;
    kv.n_swa          = 512;
    kv.n_seq_max      = 4;
    kv.swa_layer_mask = { true, false };
    // layer_kind/layer_k_width/layer_v_width intentionally left empty.

    check_eq("SWA layer via fallback", kv.kv_bytes_for_layer(0), kv.kv_bytes_per_swa_layer());
    check_eq("non-SWA layer via fallback", kv.kv_bytes_for_layer(1), kv.kv_bytes_per_layer());
}

// ---------------------------------------------------------------------------
// (g) placement_plan::kv_size_for_layer() -- the function the runtime
//     transaction body's refresh_kv_byte_totals()/rebuild_runtime_per_
//     device_vram() actually query -- must agree with placement_kv_info::
//     kv_bytes_for_layer() layer-for-layer once the plan carries the same
//     per-layer truth. This is the "planner and allocator consult the same
//     per-layer attention kind" property the ticket is about: if these two
//     functions could independently drift, the exact defect this ticket
//     fixes reappears one layer down.
// ---------------------------------------------------------------------------
static void test_plan_kv_size_for_layer_matches_kv_info() {
    printf("(g) placement_plan::kv_size_for_layer() agrees with placement_kv_info::kv_bytes_for_layer()\n");
    placement_kv_info kv = make_gemma4_e4b(4096, 512, 4);

    placement_plan plan{};
    plan.layer_kind        = kv.layer_kind;
    plan.layer_k_width     = kv.layer_k_width;
    plan.layer_v_width     = kv.layer_v_width;
    plan.planner_n_ctx     = kv.n_ctx;
    plan.planner_n_swa     = kv.n_swa;
    plan.planner_n_ubatch  = kv.n_ubatch;
    plan.planner_n_seq_max = kv.n_seq_max;

    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        char label[64];
        snprintf(label, sizeof(label), "layer %u", il);
        check_eq(label, plan.kv_size_for_layer(il), kv.kv_bytes_for_layer(il));
    }
}

int main() {
    printf("=== SYCL KV planner per-layer sizing (llama.cpp-3aos) ===\n");

    test_gemma4_n_seq_max_4();
    test_gemma4_n_seq_max_1();
    test_gptoss_n_seq_max_4();
    test_min_binds_on_small_n_ctx();
    test_aggregate_uses_n_seq_max();
    test_fallback_without_per_layer_arrays();
    test_plan_kv_size_for_layer_matches_kv_info();

    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
