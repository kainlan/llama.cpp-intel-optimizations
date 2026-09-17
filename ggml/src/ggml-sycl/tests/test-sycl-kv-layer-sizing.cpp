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
//   llama_context::llama_context()'s n_ctx_seq derivation and
//   llama_kv_cache_iswa::llama_kv_cache_iswa()'s size_swa; see that
//   function's own comment for the full derivation and the
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

// llama.cpp-7yv9 (cases (h)-(k) below): the tier manager that turns the plan
// into per-layer allocations -- kv_tier_manager::configure_from_plan()
// (kv-tier-manager.cpp) -- kept sizing layers from the uniform per-buffer
// scalars (the LAYER_MASK slice average, then plan.kv_per_swa_layer, which
// is computed at the GLOBAL fallback width) after every other consumer had
// moved to placement_plan::kv_size_for_layer(). On Gemma 4 E4B at
// -c 4096 -np 4 every SWA layer was therefore allocated 16 MB instead of
// 8 MB (2x VRAM over-reservation), and a single KV buffer that mixed
// attention kinds would have been mis-sized outright. This file exercises
// the REAL kv_tier_manager (kv-tier-manager.cpp is compiled into this
// target) -- the same arrangement as test-kv-slice-sizing.cpp.

#include "../kv-tier-manager.hpp"
#include "../unified-cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using ggml_sycl::kv_slice_size;
using ggml_sycl::kv_tier_manager;
using ggml_sycl::layer_region;
using ggml_sycl::placement_kv_info;
using ggml_sycl::placement_plan;

// Seam: kv-tier-manager.cpp's configure_with_weights() asks unified-cache for
// per-layer weight residency. configure_from_plan(), the only entry point this
// file drives, never calls it -- but the symbol must resolve for the link, and
// linking libggml-sycl instead would drag a device runtime into a host-only
// test (see the CMake comment on test-kv-slice-sizing).
namespace ggml_sycl {
size_t unified_cache_get_layer_vram_bytes(int device, int layer_id) {
    (void) device;
    (void) layer_id;
    return 0;
}
}  // namespace ggml_sycl

static int g_checks = 0;
static int g_fail   = 0;

// Everything the code under test logs, so a case can assert that a WARN was
// (or was not) emitted instead of trusting that the sizing decision it
// describes happened. Echoed to stdout as well so a failing run still shows
// the [KV-TIER] lines next to the FAIL that cites them.
static std::string g_log;

static void capture_log(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    g_log += text;
    fputs(text, stdout);
}

static void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_fail++;
        printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

static void check_true(const char * what, bool cond) {
    g_checks++;
    if (!cond) {
        g_fail++;
        printf("  FAIL: %s\n", what);
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
    // total cells = 4 * 1024 = 4096 cells * 1024 elements = 4,194,304, * 2 B = 8,388,608.
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
// (a1b) Same shape, n_seq_max=3 -- every other case in this file divides
//       n_ctx by n_seq_max exactly (4096/4, 4096/1), so the floor-then-pad
//       arithmetic (n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)) never
//       actually sees a remainder. 4096/3 does. Precondition worth
//       stating: in real llama usage n_ctx is already rounded DOWN to
//       n_ctx_seq * n_seq_max before the SYCL planner ever sees it
//       (llama_context::llama_context()'s n_ctx_seq derivation), so 4096
//       paired with n_seq_max=3 is not a shape llama would actually send
//       here -- it is used deliberately anyway, to pin today's floor+pad
//       arithmetic against a non-exact quotient, so a change to either
//       that arithmetic or to llama's own rounding shows up as a failure
//       here instead of silently.
// ---------------------------------------------------------------------------
static void test_gemma4_non_unified_n_seq_max_3_uneven_division() {
    printf("(a1b) gemma4 E4B, kv_unified=false, n_ctx=4096 n_ubatch=512 n_seq_max=3 (uneven)\n");
    placement_kv_info kv = make_gemma4_e4b(4096, 512, 3, /*kv_unified=*/false);

    // n_ctx_seq = pad256(4096/3) = pad256(1365) = 1536, n_stream = 3, window_seqs = 1.
    // l0 (SWA): per-stream cells = pad256(min(1536, 512*1 + 512)) = pad256(1024) = 1024.
    // total cells = 3 * 1024 = 3072 cells * 1024 elements = 3,145,728, * 2 B = 6,291,456.
    check_eq("l0 (SWA)", kv.kv_bytes_for_layer(0), 6291456u);

    // l5 (FULL): unaffected by the uneven division -- the FULL branch
    // returns n_ctx * width * 2 directly, never n_ctx_seq * n_stream.
    check_eq("l5 (FULL)", kv.kv_bytes_for_layer(5), 16777216u);
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

// ---------------------------------------------------------------------------
// llama.cpp-7yv9: the tier manager must consume the plan's per-layer truth.
// ---------------------------------------------------------------------------

// The plan compute_placement_plan() (unified-cache.cpp) builds from a
// kv_info: per-layer truth mirrored, the legacy uniform split filled from the
// aggregate estimators -- so kv_per_swa_layer is at the GLOBAL fallback width
// (16 MB for E4B @ -np 4, the exact number the defect reads) -- and every
// layer's KV assigned to `device`. swa_layer_mask follows hparams.is_swa()'s
// raw "full every 6th" pattern for all 42 layers, as llama-model.cpp reports
// it, i.e. it also flags the SHARED layers 24-41 that hold no KV of their own.
static placement_plan make_plan_from_kv_info(const placement_kv_info & kv, int device) {
    placement_plan plan{};
    plan.kv_per_layer     = kv.kv_bytes_per_layer();
    plan.kv_per_swa_layer = kv.kv_bytes_per_swa_layer();
    plan.swa_layer_mask.assign(kv.n_layer, false);
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        plan.swa_layer_mask[il]              = (il % 6) != 5;
        plan.kv_device[static_cast<int>(il)] = device;
    }
    plan.layer_kind         = kv.layer_kind;
    plan.layer_k_width      = kv.layer_k_width;
    plan.layer_v_width      = kv.layer_v_width;
    plan.planner_n_ctx      = kv.n_ctx;
    plan.planner_n_swa      = kv.n_swa;
    plan.planner_n_ubatch   = kv.n_ubatch;
    plan.planner_n_seq_max  = kv.n_seq_max;
    plan.planner_kv_unified = kv.kv_unified;
    plan.planner_swa_full   = kv.swa_full;
    return plan;
}

// llama_kv_cache_iswa gives one KV buffer per attention kind; this is the
// layer mask llama-kv-cache.cpp pushes for the buffer holding `kind`.
static std::vector<uint8_t> kv_buffer_mask(const placement_kv_info & kv, uint8_t kind) {
    std::vector<uint8_t> mask(kv.n_layer, 0);
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        mask[il] = kv.layer_kind[il] == kind ? 1 : 0;
    }
    return mask;
}

static uint32_t count_mask(const std::vector<uint8_t> & mask) {
    uint32_t n = 0;
    for (uint8_t m : mask) {
        n += m ? 1 : 0;
    }
    return n;
}

// What tiered_kv_buft_alloc_buffer (ggml-sycl.cpp) ends up reserving on the
// device for this buffer: layout[l].size summed over the buffer's own layers
// that landed on device (it zeroes every other region before allocating).
static size_t device_bytes_in_mask(const std::vector<layer_region> & layout, const std::vector<uint8_t> & mask) {
    size_t sum = 0;
    for (const auto & r : layout) {
        if (r.layer_id < mask.size() && mask[r.layer_id] && r.on_device) {
            sum += r.size;
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// (h) Gemma 4 E4B @ -c 4096 -np 4, kv_unified=false, both of llama's KV
//     buffers. Each layer's tier size must be placement_plan::
//     kv_size_for_layer(il): 16 MB FULL / 8 MB SWA / 0 SHARED, and the
//     device bytes reserved for a buffer must equal the buffer.
//
//     RED on the pre-fix code, SWA buffer: every SWA layer reports 16777216
//     (plan.kv_per_swa_layer, global width) against the buffer's real 8388608,
//     so the buffer's device bytes come out at 335544320 for a 167772160 B
//     buffer -- the 2x over-reservation the lead's archived run showed
//     ("[KV-ALLOC] kv_per_layer=8.0 MB" followed by size=16777216 allocs).
// ---------------------------------------------------------------------------
static void test_tier_manager_sizes_layers_from_plan_truth() {
    printf("(h) kv_tier_manager::configure_from_plan() sizes each layer from kv_size_for_layer()\n");
    placement_kv_info kv   = make_gemma4_e4b(4096, 512, 4, /*kv_unified=*/false);
    placement_plan    plan = make_plan_from_kv_info(kv, /*device=*/0);
    check_eq("precondition: plan.kv_per_swa_layer is the global-width 16 MB the defect reads", plan.kv_per_swa_layer,
             16777216u);
    check_eq("precondition: plan.kv_per_layer", plan.kv_per_layer, 16777216u);

    // SWA buffer: 20 layers x 8 MB.
    {
        const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_SWA);
        const size_t total = 20u * 8388608u;
        check_eq("SWA buffer: mask covers 20 layers", count_mask(mask), 20u);
        const auto slice = kv_slice_size::from_layer_mask(total, 20);

        kv_tier_manager mgr;
        mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

        check_eq("SWA buffer: l0 (SWA) tier size", mgr.kv_layer_size(0), 8388608u);
        check_eq("SWA buffer: l5 (FULL) tier size", mgr.kv_layer_size(5), 16777216u);
        check_eq("SWA buffer: l24 (SHARED) tier size", mgr.kv_layer_size(24), 0u);
        check_eq("SWA buffer: l29 (SHARED, pattern says full) tier size", mgr.kv_layer_size(29), 0u);
        for (uint32_t il = 0; il < 24; ++il) {
            if (mask[il]) {
                check_eq("SWA buffer: every SWA layer", mgr.kv_layer_size(il), 8388608u);
            }
        }

        const auto layout = mgr.compute_region_layout(total);
        check_eq("SWA buffer: layout entries", layout.size(), 42u);
        check_eq("SWA buffer: device bytes reserved for the buffer's layers", device_bytes_in_mask(layout, mask),
                 total);
        check_true("SWA buffer: no sizing WARN for a buffer that matches its truth",
                   g_log.find("[KV-TIER] per-layer KV truth") == std::string::npos);
    }

    // FULL buffer: 4 layers x 16 MB. Homogeneous, so the LAYER_MASK average
    // already got these layers right pre-fix; what was wrong is every OTHER
    // layer's recorded size (SWA at 16 MB, SHARED at 16 MB).
    {
        g_log.clear();
        const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_FULL);
        const size_t total = 4u * 16777216u;
        check_eq("FULL buffer: mask covers 4 layers", count_mask(mask), 4u);
        const auto slice = kv_slice_size::from_layer_mask(total, 4);

        kv_tier_manager mgr;
        mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

        check_eq("FULL buffer: l5 (FULL) tier size", mgr.kv_layer_size(5), 16777216u);
        check_eq("FULL buffer: l0 (SWA) tier size", mgr.kv_layer_size(0), 8388608u);
        check_eq("FULL buffer: l24 (SHARED) tier size", mgr.kv_layer_size(24), 0u);

        const auto layout = mgr.compute_region_layout(total);
        check_eq("FULL buffer: device bytes reserved for the buffer's layers", device_bytes_in_mask(layout, mask),
                 total);
        check_true("FULL buffer: no sizing WARN for a buffer that matches its truth",
                   g_log.find("[KV-TIER] per-layer KV truth") == std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// (i) No per-layer truth on the plan (an inventory built before llama.cpp-
//     3aos, or a homogeneous model that never populated it): the legacy
//     ranking must be untouched -- LAYER_MASK slice for non-SWA layers,
//     plan.kv_per_swa_layer for SWA layers. Passes before and after the fix;
//     it is here so the fallback cannot be lost while adding the truth path.
// ---------------------------------------------------------------------------
static void test_tier_manager_legacy_split_without_truth() {
    printf("(i) configure_from_plan() keeps the legacy uniform split when the plan has no per-layer truth\n");
    placement_kv_info kv   = make_gemma4_e4b(4096, 512, 4, /*kv_unified=*/false);
    placement_plan    plan = make_plan_from_kv_info(kv, 0);
    plan.layer_kind.clear();
    plan.layer_k_width.clear();
    plan.layer_v_width.clear();

    const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_SWA);
    const size_t total = 20u * 8388608u;
    const auto   slice = kv_slice_size::from_layer_mask(total, 20);

    kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

    check_eq("legacy: SWA layer takes plan.kv_per_swa_layer", mgr.kv_layer_size(0), plan.kv_per_swa_layer);
    check_eq("legacy: non-SWA layer takes the slice", mgr.kv_layer_size(5), slice.bytes());
    check_eq("legacy: kv_per_layer() is the slice", mgr.kv_per_layer(), slice.bytes());
}

// ---------------------------------------------------------------------------
// (j) The buffer is the ground truth about what llama allocated. If the
//     plan's per-layer sum comes out SMALLER than the buffer by more than
//     alignment slack, the plan is stale or wrong, and sizing layers from it
//     would under-allocate them -- the exact "[KV-REMAP] ERROR: overflows
//     layer alloc!" failure. configure_from_plan() must WARN naming both
//     numbers and fall back to the slice (which is derived from the buffer
//     and therefore fits), never proceed with the under-sized truth.
// ---------------------------------------------------------------------------
static void test_tier_manager_truth_under_sizing_buffer_falls_back() {
    printf("(j) per-layer truth that under-sizes the buffer is refused with a WARN and the slice wins\n");
    placement_kv_info kv   = make_gemma4_e4b(4096, 512, 4, /*kv_unified=*/false);
    placement_plan    plan = make_plan_from_kv_info(kv, 0);

    const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_SWA);
    const size_t total = 20u * 16777216u;  // llama allocated 16 MB/layer; truth says 8 MB
    const auto   slice = kv_slice_size::from_layer_mask(total, 20);

    g_log.clear();
    kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

    check_eq("under-size: SWA layer falls back to the slice", mgr.kv_layer_size(0), 16777216u);
    check_true("under-size: WARN names the truth sum and the buffer",
               g_log.find("[KV-TIER] per-layer KV truth sum 167772160 B under-sizes this buffer's 335544320 B") !=
                   std::string::npos);
    const auto layout = mgr.compute_region_layout(total);
    check_eq("under-size: device bytes still cover the buffer", device_bytes_in_mask(layout, mask), total);
}

// ---------------------------------------------------------------------------
// (k) The other direction: a per-layer sum LARGER than the buffer. The
//     planner's formula is fp16 (kv_layer_bytes_for_kind() multiplies by
//     sizeof(ggml_fp16_t)), so this is every run with a quantized KV cache
//     (-ctk/-ctv q8_0 halves the real buffer), not just a stale plan.
//     Keeping the truth would reserve ~2x the buffer on the device; the
//     slice is exact for the buffer llama actually allocated, so the truth is
//     refused -- never silently: a WARN names both numbers. (No clamping of
//     the per-layer sizes to fit: a clamp would hide a planner/allocator
//     disagreement as a working run.)
// ---------------------------------------------------------------------------
static void test_tier_manager_truth_exceeding_buffer_warns_and_falls_back() {
    printf("(k) per-layer truth that exceeds the buffer is refused with a WARN and the slice wins\n");
    placement_kv_info kv   = make_gemma4_e4b(4096, 512, 4, /*kv_unified=*/false);
    placement_plan    plan = make_plan_from_kv_info(kv, 0);

    const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_SWA);
    const size_t total = 20u * 4194304u;  // buffer is 4 MB/layer; truth says 8 MB
    const auto   slice = kv_slice_size::from_layer_mask(total, 20);

    g_log.clear();
    kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

    check_eq("over-size: SWA layer falls back to the slice", mgr.kv_layer_size(0), 4194304u);
    check_true("over-size: WARN names the truth sum and the buffer",
               g_log.find("[KV-TIER] per-layer KV truth sum 167772160 B exceeds this buffer's 83886080 B") !=
                   std::string::npos);
    const auto layout = mgr.compute_region_layout(total);
    check_eq("over-size: device bytes equal the buffer, not the truth", device_bytes_in_mask(layout, mask), total);
}

// ---------------------------------------------------------------------------
// llama.cpp-uajm: llama_context_default_params() sets swa_full=true (full-size
// SWA cache) while common/common.h defaults it false, so every CLI tool runs
// window-sized SWA caches and never exercised the planner against the raw-API
// default. With swa_full, llama_kv_cache_iswa::llama_kv_cache_iswa()
// (src/llama-kv-cache-iswa.cpp) sets size_swa = size_base -- an SWA layer's
// cache is then n_ctx_seq cells per stream, i.e. byte-identical to a FULL
// layer of the same width -- and the planned window-sized slab overflows at
// context init: "[KV-REMAP] ERROR: cache_k_l0 overflows layer alloc!
// off_in_layer=0 + nbytes=4194304 > la.size=1572864" on GPT-OSS 20B at
// -c 4096 -ub 512 (4194304 = 4096 cells * 512 * 2 B, K alone; 1572864 =
// 768 cells * 1024 * 2 B, the windowed K+V slab). Cases (l)-(o) pin that the
// plan describes what llama actually allocates in BOTH modes.
// ---------------------------------------------------------------------------

// (l) placement_kv_info::kv_bytes_for_layer() honours swa_full: the exact
//     shape from the ticket (GPT-OSS 20B, n_ctx=4096, n_ubatch=512,
//     n_seq_max=1, kv_unified=false). swa_full=false keeps the windowed
//     1572864; swa_full=true must be the full-layer 8388608 (K 4194304 + V
//     4194304). FULL layers are unaffected either way.
static void test_gptoss_swa_full_sizes_swa_layers_as_full() {
    printf("(l) gpt-oss-20b, swa_full honoured by kv_bytes_for_layer(), n_ctx=4096 n_ubatch=512 n_seq_max=1\n");
    placement_kv_info kv = make_gptoss_20b(4096, 512, 1, /*kv_unified=*/false);

    kv.swa_full = false;
    check_eq("(l) gpt-oss l0 (SWA), swa_full=false", kv.kv_bytes_for_layer(0), 1572864u);
    check_eq("(l) gpt-oss l1 (FULL), swa_full=false", kv.kv_bytes_for_layer(1), 8388608u);
    check_eq("(l) aggregate kv_bytes_per_swa_layer(), swa_full=false", kv.kv_bytes_per_swa_layer(), 1572864u);

    kv.swa_full = true;
    check_eq("(l) gpt-oss l0 (SWA), swa_full=true", kv.kv_bytes_for_layer(0), 8388608u);
    check_eq("(l) gpt-oss l0 (SWA) equals a FULL layer when swa_full", kv.kv_bytes_for_layer(0),
             kv.kv_bytes_for_layer(1));
    check_eq("(l) gpt-oss l1 (FULL), swa_full=true", kv.kv_bytes_for_layer(1), 8388608u);
    check_eq("(l) aggregate kv_bytes_per_swa_layer(), swa_full=true", kv.kv_bytes_per_swa_layer(), 8388608u);

    // Multi-stream (kv_unified=false, n_seq_max=4): size_swa = size_base =
    // n_ctx_seq = 1024 per stream, 4 streams -> 4096 cells total -- still
    // the FULL byte count, in both kv_unified modes.
    for (bool kv_unified : { false, true }) {
        placement_kv_info kv4 = make_gptoss_20b(4096, 512, 4, kv_unified);
        kv4.swa_full          = true;
        char label[96];
        snprintf(label, sizeof(label), "(l) gpt-oss l0 (SWA), swa_full=true, n_seq_max=4, kv_unified=%d",
                 (int) kv_unified);
        check_eq(label, kv4.kv_bytes_for_layer(0), 8388608u);
    }

    // SHARED layers stay at 0 regardless of swa_full (Gemma 4 E4B l24).
    placement_kv_info g = make_gemma4_e4b(4096, 512, 1, /*kv_unified=*/false);
    g.swa_full          = true;
    check_eq("(l) gemma4 l24 (SHARED), swa_full=true", g.kv_bytes_for_layer(24), 0u);
    check_eq("(l) gemma4 l0 (SWA, width 512), swa_full=true", g.kv_bytes_for_layer(0), 4096ull * 1024 * 2);
}

// (m) placement_plan::kv_size_for_layer() -- the function the runtime
//     transaction body and the tier manager actually query -- must honour
//     planner_swa_full the same way, and agree with kv_bytes_for_layer()
//     layer-for-layer in both modes.
static void test_plan_kv_size_for_layer_honours_swa_full() {
    printf("(m) placement_plan::kv_size_for_layer() honours planner_swa_full\n");
    for (bool swa_full : { false, true }) {
        placement_kv_info kv = make_gptoss_20b(4096, 512, 1, /*kv_unified=*/false);
        kv.swa_full          = swa_full;

        placement_plan plan{};
        plan.layer_kind         = kv.layer_kind;
        plan.layer_k_width      = kv.layer_k_width;
        plan.layer_v_width      = kv.layer_v_width;
        plan.planner_n_ctx      = kv.n_ctx;
        plan.planner_n_swa      = kv.n_swa;
        plan.planner_n_ubatch   = kv.n_ubatch;
        plan.planner_n_seq_max  = kv.n_seq_max;
        plan.planner_kv_unified = kv.kv_unified;
        plan.planner_swa_full   = kv.swa_full;

        char label[96];
        snprintf(label, sizeof(label), "(m) plan l0 (SWA), planner_swa_full=%d", (int) swa_full);
        check_eq(label, plan.kv_size_for_layer(0), swa_full ? 8388608u : 1572864u);
        for (uint32_t il = 0; il < kv.n_layer; ++il) {
            snprintf(label, sizeof(label), "(m) layer %u, swa_full=%d agrees with kv_info", il, (int) swa_full);
            check_eq(label, plan.kv_size_for_layer(il), kv.kv_bytes_for_layer(il));
        }
    }
}

// (n) kv_tier_manager::configure_from_plan() sizes each SWA layer from the
//     plan's swa_full-aware truth: an SWA buffer llama allocated at
//     swa_full=true is 12 x 8388608 B on GPT-OSS 20B; the truth must match
//     it exactly (no "[KV-TIER] per-layer KV truth ... under-sizes" WARN,
//     no fallback to the slice), and every SWA layer's tier size must be the
//     full-layer 8388608, never the windowed 1572864 the defect reserved.
static void test_tier_manager_honours_swa_full() {
    printf("(n) kv_tier_manager::configure_from_plan() sizes SWA layers from the swa_full-aware plan\n");
    placement_kv_info kv = make_gptoss_20b(4096, 512, 1, /*kv_unified=*/false);
    kv.swa_full          = true;
    placement_plan plan  = make_plan_from_kv_info(kv, /*device=*/0);
    plan.swa_layer_mask.assign(kv.n_layer, false);
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        plan.swa_layer_mask[il] = kv.layer_kind[il] == GGML_SYCL_KV_LAYER_SWA;
    }

    const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_SWA);
    const size_t total = 12u * 8388608u;  // what llama_kv_cache_iswa allocates with swa_full
    check_eq("(n) SWA buffer: mask covers 12 layers", count_mask(mask), 12u);
    const auto slice = kv_slice_size::from_layer_mask(total, 12);

    g_log.clear();
    kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

    check_eq("(n) SWA buffer, swa_full=true: l0 tier size", mgr.kv_layer_size(0), 8388608u);
    check_eq("(n) SWA buffer, swa_full=true: l1 (FULL) tier size", mgr.kv_layer_size(1), 8388608u);
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        if (mask[il]) {
            check_eq("(n) SWA buffer, swa_full=true: every SWA layer", mgr.kv_layer_size(il), 8388608u);
        }
    }
    const auto layout = mgr.compute_region_layout(total);
    check_eq("(n) SWA buffer, swa_full=true: device bytes reserved equal the buffer",
             device_bytes_in_mask(layout, mask), total);
    check_true("(n) SWA buffer, swa_full=true: no sizing WARN -- the truth matches the buffer",
               g_log.find("[KV-TIER] per-layer KV truth") == std::string::npos);
}

// (o) The regression guard for the mode every CLI tool runs: the same
//     buffer allocated at swa_full=false is 12 x 1572864 B, and the plan at
//     planner_swa_full=false must match it exactly -- honouring swa_full must
//     not move the windowed answer.
static void test_tier_manager_swa_full_false_unchanged() {
    printf("(o) configure_from_plan() at swa_full=false keeps the windowed sizes\n");
    placement_kv_info kv = make_gptoss_20b(4096, 512, 1, /*kv_unified=*/false);
    kv.swa_full          = false;
    placement_plan plan  = make_plan_from_kv_info(kv, /*device=*/0);
    plan.swa_layer_mask.assign(kv.n_layer, false);
    for (uint32_t il = 0; il < kv.n_layer; ++il) {
        plan.swa_layer_mask[il] = kv.layer_kind[il] == GGML_SYCL_KV_LAYER_SWA;
    }

    const auto   mask  = kv_buffer_mask(kv, GGML_SYCL_KV_LAYER_SWA);
    const size_t total = 12u * 1572864u;
    const auto   slice = kv_slice_size::from_layer_mask(total, 12);

    g_log.clear();
    kv_tier_manager mgr;
    mgr.configure_from_plan(0, plan, kv.n_layer, slice, &mask);

    check_eq("(o) SWA buffer, swa_full=false: l0 tier size", mgr.kv_layer_size(0), 1572864u);
    const auto layout = mgr.compute_region_layout(total);
    check_eq("(o) SWA buffer, swa_full=false: device bytes reserved equal the buffer",
             device_bytes_in_mask(layout, mask), total);
    check_true("(o) SWA buffer, swa_full=false: no sizing WARN",
               g_log.find("[KV-TIER] per-layer KV truth") == std::string::npos);
}

int main() {
    // Hermetic: GGML_SYCL_KV_HOT_LAYERS short-circuits configure_from_plan()
    // before any per-layer sizing, so a stray value in the environment would
    // quietly change what cases (h)-(k) measure.
    if (const char * env = std::getenv("GGML_SYCL_KV_HOT_LAYERS")) {
        printf("unsetting GGML_SYCL_KV_HOT_LAYERS=%s for a hermetic run\n", env);
        unsetenv("GGML_SYCL_KV_HOT_LAYERS");
    }
    ggml_log_set(capture_log, nullptr);

    printf("=== SYCL KV planner per-layer sizing (llama.cpp-3aos) ===\n");

    test_gemma4_non_unified_n_seq_max_4();
    test_gemma4_non_unified_n_seq_max_3_uneven_division();
    test_gemma4_unified_n_seq_max_4();
    test_gemma4_n_seq_max_1_both_modes();
    test_gptoss_non_unified_np4();
    test_gptoss_unified_np4();
    test_min_binds_on_small_n_ctx();
    test_aggregate_uses_n_seq_max_and_kv_unified();
    test_fallback_without_per_layer_arrays();
    test_plan_kv_size_for_layer_matches_kv_info();

    printf("=== kv_tier_manager consumes the plan's per-layer sizes (llama.cpp-7yv9) ===\n");
    test_tier_manager_sizes_layers_from_plan_truth();
    test_tier_manager_legacy_split_without_truth();
    test_tier_manager_truth_under_sizing_buffer_falls_back();
    test_tier_manager_truth_exceeding_buffer_warns_and_falls_back();

    printf("=== swa_full honoured: SWA layers sized as FULL when llama allocates them so (llama.cpp-uajm) ===\n");
    test_gptoss_swa_full_sizes_swa_layers_as_full();
    test_plan_kv_size_for_layer_honours_swa_full();
    test_tier_manager_honours_swa_full();
    test_tier_manager_swa_full_false_unchanged();

    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
