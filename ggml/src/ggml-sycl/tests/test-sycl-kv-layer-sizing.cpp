// Regression test for the SYCL KV-cache planner's per-layer sizing
// (llama.cpp-3aos).
//
// Three bugs, one root: placement_kv_info's KV-budget formulas assumed a
// SINGLE (n_embd_k_gqa, n_embd_v_gqa) width pair, a single SWA/non-SWA
// mask, kv_bytes_per_swa_layer() hardcoded n_seq_max=1, and the SWA formula
// assumed the KV cache is always a single stream shared by every sequence.
//
//   Bug 1 (n_seq_max): llama_kv_cache_iswa allocates
//   GGML_PAD(min(kv_size, n_swa * n_seq_max + n_ubatch), 256) cells per SWA
//   layer's stream, but the planner's formula used n_swa + n_ubatch --
//   correct only at n_seq_max=1. Any SWA model served with `--parallel > 1`
//   was under-budgeted (Gemma 4 E4B at -c 4096 --parallel 4: planner 1024
//   cells vs llama's real per-mode total -> "[KV-REMAP] cache_k_l0
//   overflows layer alloc!").
//
//   Bug 2 (width): Gemma 4 E4B is heterogeneous -- full-attention layers
//   are WIDER than SWA layers (1024 vs 512 elements on E4B) and a trailing
//   block of layers has NO K/V of their own at all (they reuse an earlier
//   layer's -- llama_hparams::has_kv() == false). A single global width
//   pair cannot represent that; the old formula used layer 0's (SWA, 512)
//   width for every layer, including full-attention ones, and had no way
//   to charge a SHARED layer 0 bytes.
//
//   Bug 3 (kv_unified, found on a live GPU acceptance run after Bugs 1-2
//   were fixed): llama_context splits its KV cache into n_seq_max
//   independent STREAMS unless kv_unified==true (llama-completion's and
//   llama-bench's default is kv_unified==false); each stream's SWA window
//   is then capped independently at n_swa + n_ubatch (not
//   n_swa * n_seq_max + n_ubatch), and there are n_seq_max such streams.
//   The Bug-1 fix alone assumed the OPPOSITE (single shared stream, window
//   scaling with n_seq_max) -- correct only when kv_unified==true, which is
//   not this fork's or upstream's default for either tool.
//   kv_layer_bytes_for_kind() (unified-cache.hpp) derives both modes from
//   src/llama-context.cpp:637-650 and src/llama-kv-cache-iswa.cpp:69-81;
//   see that function's own comment for the full derivation and the
//   GPU-run numbers it was checked against.
//
// This test exercises the REAL ggml_sycl::placement_kv_info and
// ggml_sycl::placement_plan -- no mock reimplementation. Both structs are
// built by hand (no model load, no device, no queue): the arithmetic under
// test is pure per-layer bookkeeping and needs no GPU.
//
// Written RED against the pre-fix formula: the pre-fix code charged every
// SWA layer at a single sequence's window and every layer (SWA or
// full-attention) at layer 0's width, with no SHARED concept and no
// kv_unified distinction at all -- the exact numbers each historical fix
// corrected are recorded in the commit bodies of this ticket's own history
// (llama.cpp-3aos), not reproduced here.

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
static placement_kv_info make_gemma4_e4b(uint32_t n_ctx, uint32_t n_ubatch, uint32_t n_seq_max, bool kv_unified) {
    placement_kv_info kv{};
    kv.n_layer      = 42;
    kv.n_embd_k_gqa = 1024;  // FULL-attention width fallback (n_embd_k_gqa_max())
    kv.n_embd_v_gqa = 1024;
    kv.n_ctx        = n_ctx;
    kv.n_ubatch     = n_ubatch;
    kv.n_seq_max    = n_seq_max;
    kv.kv_unified   = kv_unified;
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
static placement_kv_info make_gptoss_20b(uint32_t n_ctx, uint32_t n_ubatch, uint32_t n_seq_max, bool kv_unified) {
    placement_kv_info kv{};
    kv.n_layer      = 24;
    kv.n_embd_k_gqa = 512;
    kv.n_embd_v_gqa = 512;
    kv.n_ctx        = n_ctx;
    kv.n_ubatch     = n_ubatch;
    kv.n_seq_max    = n_seq_max;
    kv.kv_unified   = kv_unified;
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
// (a1) Gemma 4 E4B, kv_unified=false (the DEFAULT for llama-completion/
//      llama-bench, and llama-server unless overridden; the exact shape
//      the lead's GPU acceptance run exercised), n_ctx=4096, n_ubatch=512,
//      n_seq_max=4 (the reported `-c 4096 --parallel 4` shape).
// ---------------------------------------------------------------------------
static void test_gemma4_non_unified_n_seq_max_4() {
    printf("(a1) gemma4 E4B, kv_unified=false, n_ctx=4096 n_ubatch=512 n_seq_max=4\n");
    placement_kv_info kv = make_gemma4_e4b(4096, 512, 4, /*kv_unified=*/false);

    // n_ctx_seq = pad256(4096/4) = 1024, n_stream = 4, window_seqs = 1.
    // l0 (SWA): per-stream cells = pad256(min(1024, 512*1 + 512)) = pad256(1024) = 1024.
    // total cells = 4 * 1024 = 4096. bytes = 4096 * (512+512) * 2 = 4,194,304 *2 = 8,388,608.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 8388608u);

    // l5 (FULL): total cells across streams == n_ctx always (n_ctx_seq * n_stream == n_ctx).
    // bytes = 4096 * (1024+1024) * 2 = 16,777,216.
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
    check_eq("total", total, 20u * 8388608u + 4u * 16777216u);
}

// ---------------------------------------------------------------------------
// (a2) Same shape, kv_unified=true -- the single-shared-stream mode; ITS
//      window scales with n_seq_max, matching the formula this ticket
//      shipped before the GPU run found Bug 3.
// ---------------------------------------------------------------------------
static void test_gemma4_unified_n_seq_max_4() {
    printf("(a2) gemma4 E4B, kv_unified=true, n_ctx=4096 n_ubatch=512 n_seq_max=4\n");
    placement_kv_info kv = make_gemma4_e4b(4096, 512, 4, /*kv_unified=*/true);

    // n_ctx_seq = n_ctx = 4096, n_stream = 1, window_seqs = 4.
    // l0 (SWA): cells = pad256(min(4096, 512*4 + 512)) = pad256(min(4096, 2560)) = 2560.
    // bytes = 2560 * (512+512) * 2 = 5,242,880.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 5242880u);
    // l5 (FULL): unaffected by kv_unified -- same 16,777,216 as (a1).
    check_eq("l5 (FULL)", kv.kv_bytes_for_layer(5), 16777216u);
}

// ---------------------------------------------------------------------------
// (b) Same shape at n_seq_max=1 -- l0 (SWA) reduces to the SAME number in
//     BOTH kv_unified modes (n_stream collapses to 1 either way when
//     n_seq_max=1: kv_unified=false gives n_stream=n_seq_max=1, kv_unified=
//     true always gives n_stream=1), matching the pre-Bug-3 number.
// ---------------------------------------------------------------------------
static void test_gemma4_n_seq_max_1_both_modes() {
    printf("(b) gemma4 E4B @ n_seq_max=1 -- both kv_unified modes agree\n");
    placement_kv_info kv_split   = make_gemma4_e4b(4096, 512, 1, /*kv_unified=*/false);
    placement_kv_info kv_unified = make_gemma4_e4b(4096, 512, 1, /*kv_unified=*/true);
    // n_ctx_seq = pad256(4096/1) = 4096 either way; cells = pad256(min(4096, 512+512)) = 1024.
    // bytes = 1024 * 1024 * 2 = 2,097,152.
    check_eq("l0 (SWA), kv_unified=false @ n_seq_max=1", kv_split.kv_bytes_for_layer(0), 2097152u);
    check_eq("l0 (SWA), kv_unified=true @ n_seq_max=1", kv_unified.kv_bytes_for_layer(0), 2097152u);
}

// ---------------------------------------------------------------------------
// (c1) GPT-OSS 20B, kv_unified=false, n_seq_max=4 (`-np 4`) -- uniform
//      width, alternating SWA/full, no SHARED layers.
// ---------------------------------------------------------------------------
static void test_gptoss_non_unified_np4() {
    printf("(c1) gpt-oss-20b, kv_unified=false, n_ctx=4096 n_ubatch=512 n_seq_max=4\n");
    placement_kv_info kv = make_gptoss_20b(4096, 512, 4, /*kv_unified=*/false);

    // n_ctx_seq = pad256(4096/4) = 1024, n_stream = 4, window_seqs = 1.
    // SWA (even layers): per-stream cells = pad256(min(1024, 128*1 + 512)) = pad256(640) = 768.
    // total cells = 4 * 768 = 3072. bytes = 3072 * (512+512) * 2 = 6,291,456.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 6291456u);

    // FULL (odd layers): bytes = 4096 * (512+512) * 2 = 8,388,608 (n_ctx-total invariant).
    check_eq("l1 (FULL)", kv.kv_bytes_for_layer(1), 8388608u);

    size_t total = 0;
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        total += kv.kv_bytes_for_layer(il);
    }
    check_eq("total", total, 12u * 6291456u + 12u * 8388608u);
}

// ---------------------------------------------------------------------------
// (c2) Same GPT-OSS shape, kv_unified=true -- for symmetry with the Gemma 4
//      pair above.
// ---------------------------------------------------------------------------
static void test_gptoss_unified_np4() {
    printf("(c2) gpt-oss-20b, kv_unified=true, n_ctx=4096 n_ubatch=512 n_seq_max=4\n");
    placement_kv_info kv = make_gptoss_20b(4096, 512, 4, /*kv_unified=*/true);

    // n_ctx_seq = n_ctx = 4096, n_stream = 1, window_seqs = 4.
    // cells = pad256(min(4096, 128*4 + 512)) = pad256(1024) = 1024.
    // bytes = 1024 * (512+512) * 2 = 2,097,152.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 2097152u);
    check_eq("l1 (FULL)", kv.kv_bytes_for_layer(1), 8388608u);
}

// ---------------------------------------------------------------------------
// (d) n_ctx SMALLER than the SWA window, so min() binds on n_ctx instead
//     of the window. kv_unified=true (matches the original derivation);
//     window = 512*4 + 512 = 2560 > n_ctx=2048.
// ---------------------------------------------------------------------------
static void test_min_binds_on_small_n_ctx() {
    printf("(d) gemma4 E4B, kv_unified=true, n_ctx=2048 < window (2560) -- min() binds on n_ctx\n");
    placement_kv_info kv = make_gemma4_e4b(2048, 512, 4, /*kv_unified=*/true);

    // n_ctx_seq = n_ctx = 2048; cells = pad256(min(2048, 2560)) = pad256(2048) = 2048 (already aligned).
    check_eq("l0 (SWA), n_ctx-bound", kv.kv_bytes_for_layer(0), 2048ull * 1024 * 2);
    // FULL layer scales with n_ctx directly, same as always.
    check_eq("l5 (FULL), n_ctx-bound", kv.kv_bytes_for_layer(5), 2048ull * 2048 * 2);
}

// ---------------------------------------------------------------------------
// (e) The aggregate estimator kv_bytes_per_swa_layer() consumes BOTH
//     n_seq_max and kv_unified, independent of whether per-layer arrays are
//     populated.
// ---------------------------------------------------------------------------
static void test_aggregate_uses_n_seq_max_and_kv_unified() {
    printf("(e) kv_bytes_per_swa_layer() consumes n_seq_max and kv_unified\n");
    placement_kv_info kv{};
    kv.n_layer      = 1;
    kv.n_embd_k_gqa = 512;
    kv.n_embd_v_gqa = 512;
    kv.n_ctx        = 4096;
    kv.n_ubatch     = 512;
    kv.n_swa        = 512;

    kv.n_seq_max  = 1;
    kv.kv_unified = false;
    check_eq("aggregate @ n_seq_max=1, kv_unified=false", kv.kv_bytes_per_swa_layer(), 2097152u);
    kv.kv_unified = true;
    check_eq("aggregate @ n_seq_max=1, kv_unified=true", kv.kv_bytes_per_swa_layer(), 2097152u);

    kv.n_seq_max  = 4;
    kv.kv_unified = false;
    check_eq("aggregate @ n_seq_max=4, kv_unified=false", kv.kv_bytes_per_swa_layer(), 8388608u);
    kv.kv_unified = true;
    check_eq("aggregate @ n_seq_max=4, kv_unified=true", kv.kv_bytes_per_swa_layer(), 5242880u);
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
    kv.kv_unified     = false;
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
//     per-layer truth (including kv_unified). This is the "planner and
//     allocator consult the same per-layer attention kind" property the
//     ticket is about: if these two functions could independently drift,
//     the exact defect this ticket fixes reappears one layer down.
// ---------------------------------------------------------------------------
static void test_plan_kv_size_for_layer_matches_kv_info() {
    printf("(g) placement_plan::kv_size_for_layer() agrees with placement_kv_info::kv_bytes_for_layer()\n");
    for (bool kv_unified : { false, true }) {
        placement_kv_info kv = make_gemma4_e4b(4096, 512, 4, kv_unified);

        placement_plan plan{};
        plan.layer_kind         = kv.layer_kind;
        plan.layer_k_width      = kv.layer_k_width;
        plan.layer_v_width      = kv.layer_v_width;
        plan.planner_n_ctx      = kv.n_ctx;
        plan.planner_n_swa      = kv.n_swa;
        plan.planner_n_ubatch   = kv.n_ubatch;
        plan.planner_n_seq_max  = kv.n_seq_max;
        plan.planner_kv_unified = kv.kv_unified;

        for (uint32_t il = 0; il < kv.n_layer; ++il) {
            char label[64];
            snprintf(label, sizeof(label), "layer %u, kv_unified=%d", il, (int) kv_unified);
            check_eq(label, plan.kv_size_for_layer(il), kv.kv_bytes_for_layer(il));
        }
    }
}

int main() {
    printf("=== SYCL KV planner per-layer sizing (llama.cpp-3aos) ===\n");

    test_gemma4_non_unified_n_seq_max_4();
    test_gemma4_unified_n_seq_max_4();
    test_gemma4_n_seq_max_1_both_modes();
    test_gptoss_non_unified_np4();
    test_gptoss_unified_np4();
    test_min_binds_on_small_n_ctx();
    test_aggregate_uses_n_seq_max_and_kv_unified();
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
