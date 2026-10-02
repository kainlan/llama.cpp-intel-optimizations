//
// Test: structural path-scoped arena zone maxima
//
// Guards the over-provision where every zone was sized from the single
// largest tensor in the model, including tensors that reach none of the
// paths being sized. Host-only: zone_scoped_maxima is a pure function over
// a vector of descriptors, so no GPU and no AOT target are needed.
//
// Classification is STRUCTURAL — (type, ne) group cardinality — never by
// name. Every fixture below therefore names its tensors *wrongly* for the
// GGUF convention: the vocab-sized tensors are called "blk.9N.some_weight"
// and the per-layer families are called "token_embd.weight.NN". A predicate
// that accidentally branched on a name would classify both populations
// backwards and fail loudly here instead of passing by luck.
//
// Byte sizes and group cardinalities are the ones measured in
// docs/plans/2026-07-25-zone-sizing-findings.md, not invented.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "compute-alloc-scope.hpp"
#include "zone-sizing.hpp"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously. Use an explicit check that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::path_scoped_maxima;
using ggml_sycl::zone_onednn_plan;
using ggml_sycl::zone_scoped_maxima;
using ggml_sycl::zone_tensor_desc;

// ggml_type codes, mirrored as plain ints exactly as the production call site
// will cast them (zone-sizing.hpp deliberately does not include ggml.h).
static const int TYPE_Q4_0  = 2;
static const int TYPE_Q8_0  = 8;
static const int TYPE_Q4_K  = 12;
static const int TYPE_Q6_K  = 14;
static const int TYPE_MXFP4 = 39;

// The oneDNN matmul weights reorder holds a DEQUANTIZED f16 copy, so its size
// is the element count times 2 regardless of how the weight is stored. This
// mirrors what unified_cache_adapt_zone_inventory() computes in production and
// what acquire_onednn_pp_scratch()'s callers actually request; the fixtures
// below carry it so the expanded maximum is exercised by every case rather
// than only by the ones written for it.
static const size_t F16_BYTES = 2;

// Measured byte sizes from the Task 1 inventory. The MB figures in the
// comments are binary (bytes / 1024^2), matching the findings document.
static const size_t GPT_OSS_VOCAB_BYTES  = 615329280;  // 586.8 MB, Q8_0  2880 x 201088
static const size_t GPT_OSS_EXPERT_BYTES = 140988600;  // 134.5 MB, MXFP4 2880 x 2880 x 32
static const size_t GPT_OSS_ATTN_BYTES   = 8812800;    //   8.4 MB, Q8_0  2880 x 2880 (dense, per-layer)
static const size_t MISTRAL_OUTPUT_BYTES = 107520000;  // 102.5 MB, Q6_K  4096 x 32000
static const size_t MISTRAL_EMBD_BYTES   = 73728000;   //  70.3 MB, Q4_0  4096 x 32000
static const size_t MISTRAL_FFN_BYTES    = 33030144;   //  31.5 MB, Q4_0  4096 x 14336

static zone_tensor_desc
desc(const std::string & name, size_t size, int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    zone_tensor_desc d;
    d.name         = name;
    d.size         = size;
    d.type         = type;
    d.ne[0]        = ne0;
    d.ne[1]        = ne1;
    d.ne[2]        = ne2;
    d.ne[3]        = ne3;
    d.has_shape    = true;
    d.reorder_size = static_cast<size_t>(ne0) * static_cast<size_t>(ne1) * static_cast<size_t>(ne2) *
                     static_cast<size_t>(ne3) * F16_BYTES;
    return d;
}

// An entry the inventory could not give a shape to. ne stays {0,0,0,0}, which
// is exactly why has_shape must gate grouping: several of these would
// otherwise collapse into one large spurious family.
static zone_tensor_desc shapeless_desc(const std::string & name, size_t size, int type) {
    zone_tensor_desc d;
    d.name      = name;
    d.size      = size;
    d.type      = type;
    d.has_shape = false;
    return d;
}

// Acceptance: a predicate can only ever narrow.
//
// onednn_reorder is deliberately NOT checked here. It is a DEQUANTIZED size,
// not a selection from the inventory, so it is not bounded by any_tensor and
// legitimately exceeds it on Mistral 7B Q4_0 (112.0 MB reorder vs a 102.5 MB
// largest tensor) -- Case 2 asserts exactly that. Adding it to this predicate
// would make the real production layout fail.
static bool is_monotonic(const path_scoped_maxima & m) {
    return m.onednn_eligible <= m.any_tensor && m.cpu_quant_eligible <= m.any_tensor && m.dma_streamed <= m.any_tensor;
}

int main() {
    // ---- Case 1: the real GPT-OSS 20B MXFP4 layout --------------------------
    // 72 expert tensors sharing one MXFP4 key, plus the embedding and the LM
    // head, which share BOTH type (Q8_0) and shape (2880 x 201088) and so
    // collapse into a single group of cardinality 2 — the measured layout.
    {
        std::vector<zone_tensor_desc> inventory;
        for (int i = 0; i < 72; i++) {
            inventory.push_back(
                desc("token_embd.weight." + std::to_string(i), GPT_OSS_EXPERT_BYTES, TYPE_MXFP4, 2880, 2880, 32, 1));
        }
        // A dense per-layer attention family, 24 blocks. The real model has
        // several; one is enough to give the oneDNN path something to fall to
        // once the expert family is excluded from it. Without this the fixture
        // could not tell "correctly narrowed" from "narrowed to nothing".
        for (int i = 0; i < 24; i++) {
            inventory.push_back(
                desc("blk." + std::to_string(i) + ".attn_out", GPT_OSS_ATTN_BYTES, TYPE_Q8_0, 2880, 2880, 1, 1));
        }
        inventory.push_back(desc("blk.99.some_weight", GPT_OSS_VOCAB_BYTES, TYPE_Q8_0, 2880, 201088, 1, 1));
        inventory.push_back(desc("blk.98.some_weight", GPT_OSS_VOCAB_BYTES, TYPE_Q8_0, 2880, 201088, 1, 1));

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.any_tensor == GPT_OSS_VOCAB_BYTES, "gpt-oss any_tensor must equal the global max (586.8 MB)");
        CHECK(maxima.cpu_quant_eligible == GPT_OSS_EXPERT_BYTES,
              "gpt-oss cpu_quant_eligible must fall to the expert family (134.5 MB)");
        CHECK(maxima.dma_streamed == GPT_OSS_EXPERT_BYTES,
              "gpt-oss dma_streamed must fall to the expert family (134.5 MB)");
        CHECK(is_monotonic(maxima), "gpt-oss maxima must all be <= any_tensor");

        // THE DIVERGENCE. The oneDNN path excludes expert tensors -- they are
        // consumed by the PP-MoE ring, and reserve_onednn_scratch is measurably
        // unreachable on this model (observations=0 at -p 512). The other two
        // paths keep them. This is the first time the three predicates disagree
        // and it is the point of their being separate functions.
        CHECK(maxima.onednn_eligible == GPT_OSS_ATTN_BYTES,
              "gpt-oss onednn_eligible must skip the expert family and fall to the dense attention family");
        CHECK(maxima.onednn_eligible < maxima.cpu_quant_eligible,
              "the oneDNN path must narrow strictly further than the paths that do carry expert tensors");
        CHECK(maxima.onednn_reorder == 2880ull * 2880ull * F16_BYTES,
              "gpt-oss onednn_reorder must expand the DENSE family (15.8 MB), not the expert stack");

        // Regression guard with the measured cost attached, so a future change
        // that re-admits expert tensors fails here with the reason in hand.
        // Including them sized the zone from 2880 x 2880 x 32 x 2 B = 506.2 MB,
        // which produced a 640.7 MB ONEDNN zone on a model that issues zero
        // reserve calls -- 384.7 MB taken out of the arena weight zone, roughly
        // 1.5 granted down-i8 layers at ~261 MB each.
        CHECK(maxima.onednn_reorder != 2880ull * 2880ull * 32ull * F16_BYTES,
              "expert stack must NOT size the oneDNN reorder: that cost 384.7 MB of weight zone on GPT-OSS");
    }

    // ---- Case 1b: the expert predicate itself ------------------------------
    // ne[2] is the expert count. Pinned directly because the whole exclusion
    // rests on it and a shape convention change would otherwise fail silently
    // by simply not excluding anything.
    {
        const zone_tensor_desc expert = desc("e", GPT_OSS_EXPERT_BYTES, TYPE_MXFP4, 2880, 2880, 32, 1);
        const zone_tensor_desc dense  = desc("d", GPT_OSS_ATTN_BYTES, TYPE_Q8_0, 2880, 2880, 1, 1);

        CHECK(ggml_sycl::zone_is_moe_expert_tensor(expert), "ne[2]=32 must read as an expert stack");
        CHECK(!ggml_sycl::zone_is_moe_expert_tensor(dense), "ne[2]=1 must read as a dense operand");
        CHECK(!ggml_sycl::zone_is_moe_expert_tensor(shapeless_desc("s", 1024, TYPE_Q8_0)),
              "a shapeless entry must not be classified an expert stack on zeroed dimensions");

        // Cardinality is high enough to clear the per-layer threshold in both
        // cases, so the only thing separating them here is the expert test.
        CHECK(!ggml_sycl::zone_is_onednn_reorder_eligible(expert, 72),
              "an expert tensor must be oneDNN-ineligible however many siblings it has");
        CHECK(ggml_sycl::zone_is_cpu_quant_eligible(expert, 72), "the CPU quant path must still accept expert tensors");
        CHECK(ggml_sycl::zone_is_dma_streamed(expert, 72), "the DMA stream path must still accept expert tensors");
    }

    // ---- Case 2: the real Mistral 7B Q4_0 layout ----------------------------
    // 64 FFN tensors sharing a Q4_0 key, plus two DISTINCT singletons: the LM
    // head is Q6_K and the embedding Q4_0, so unlike GPT-OSS they do not
    // collapse into a pair. Both configurations must land below the threshold.
    {
        std::vector<zone_tensor_desc> inventory;
        for (int i = 0; i < 64; i++) {
            inventory.push_back(
                desc("token_embd.weight." + std::to_string(i), MISTRAL_FFN_BYTES, TYPE_Q4_0, 4096, 14336, 1, 1));
        }
        inventory.push_back(desc("blk.99.some_weight", MISTRAL_OUTPUT_BYTES, TYPE_Q6_K, 4096, 32000, 1, 1));
        inventory.push_back(desc("blk.98.some_weight", MISTRAL_EMBD_BYTES, TYPE_Q4_0, 4096, 32000, 1, 1));

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.any_tensor == MISTRAL_OUTPUT_BYTES, "mistral any_tensor must equal the global max (102.5 MB)");
        CHECK(maxima.onednn_eligible == MISTRAL_FFN_BYTES,
              "mistral onednn_eligible must fall to the FFN family (31.5 MB)");
        CHECK(maxima.cpu_quant_eligible == MISTRAL_FFN_BYTES,
              "mistral cpu_quant_eligible must fall to the FFN family (31.5 MB)");
        CHECK(maxima.dma_streamed == MISTRAL_FFN_BYTES, "mistral dma_streamed must fall to the FFN family (31.5 MB)");
        CHECK(is_monotonic(maxima), "mistral maxima must all be <= any_tensor");

        // The measured defect in llama.cpp-2wgg, pinned. The FFN weight is
        // 31.5 MB stored and 112.0 MB once dequantized to f16 (4096 x 14336 x
        // 2 B) -- exactly Q4_0's 4.5 bits/weight going to 16, a 3.5556x
        // expansion. Sizing the ONEDNN zone's weights half from the stored
        // 31.5 MB planned 63.0 MB against a measured 126.0 MB peak.
        CHECK(maxima.onednn_reorder == 4096ull * 14336ull * F16_BYTES,
              "mistral onednn_reorder must be the f16 expansion of the FFN family (112.0 MB)");
        CHECK(maxima.onednn_reorder * 9 == maxima.onednn_eligible * 32,
              "mistral Q4_0 expansion must be exactly 32/9 = 3.5556x (16 bits / 4.5 bits)");

        // The property that breaks naive assertions: a dequantized reorder
        // buffer can be LARGER than the biggest tensor in the model. Here the
        // Q6_K LM head is the largest stored tensor at 102.5 MB and the reorder
        // needs 112.0 MB. Any `onednn_reorder <= any_tensor` check -- in an
        // assert, a zone-budget calculation, or a future collapse detector --
        // would fire on a healthy Mistral.
        CHECK(maxima.onednn_reorder > maxima.any_tensor,
              "mistral onednn_reorder must be allowed to exceed the global max (112.0 > 102.5 MB)");
    }

    // ---- Case 2b: mixed quantization, where the two winners diverge ---------
    // The reason onednn_reorder is its own accumulator rather than
    // onednn_eligible scaled by a factor at the call site.
    //
    // Expansion is per type, so on a mixed-quantization model the largest
    // STORED eligible tensor and the largest EXPANDED one can be different
    // tensors. Two per-layer families, both above the cardinality threshold:
    //
    //   A: Q6_K 4096 x 11008 -> stored 36988800 B (35.3 MB), reorder  86.0 MB
    //   B: Q4_K 4096 x 14336 -> stored 33030144 B (31.5 MB), reorder 112.0 MB
    //
    // A is bigger stored; B is bigger expanded. Scaling the stored winner by
    // its own Q6_K factor (256 x 2 / 210 = 2.438x) gives 86.0 MB and under-sizes
    // the real 112.0 MB requirement by 23%. Taking the maximum over the
    // expanded sizes is the only formulation that survives this.
    {
        const size_t A_STORED = 36988800;  // 4096 x 11008 / 256 x 210
        const size_t B_STORED = 33030144;  // 4096 x 14336 / 256 x 144

        std::vector<zone_tensor_desc> inventory;
        for (int i = 0; i < 32; i++) {
            inventory.push_back(desc("blk." + std::to_string(i) + ".a", A_STORED, TYPE_Q6_K, 4096, 11008, 1, 1));
            inventory.push_back(desc("blk." + std::to_string(i) + ".b", B_STORED, TYPE_Q4_K, 4096, 14336, 1, 1));
        }

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.onednn_eligible == A_STORED, "mixed-quant onednn_eligible must pick the larger STORED family (A)");
        CHECK(maxima.onednn_reorder == 4096ull * 14336ull * F16_BYTES,
              "mixed-quant onednn_reorder must pick the larger EXPANDED family (B, 112.0 MB)");
        CHECK(maxima.onednn_reorder > A_STORED * 256ull * F16_BYTES / 210ull,
              "expanding the stored winner by its own factor must under-size the real requirement");
    }

    // ---- Case 3: the threshold boundary -------------------------------------
    // A group of exactly 3 is excluded, a group of exactly 4 is included. This
    // pins the constant: it must be >= 3 (so a collapsed embd/output pair never
    // qualifies) and <= 4 (so a family present in only a few blocks still does).
    {
        std::vector<zone_tensor_desc> inventory;
        for (int i = 0; i < 3; i++) {
            inventory.push_back(
                desc("group_of_three." + std::to_string(i), 900u * 1024u * 1024u, TYPE_Q8_0, 64, 64, 1, 1));
        }
        for (int i = 0; i < 4; i++) {
            inventory.push_back(
                desc("group_of_four." + std::to_string(i), 100u * 1024u * 1024u, TYPE_Q8_0, 128, 128, 1, 1));
        }

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.any_tensor == 900u * 1024u * 1024u, "boundary any_tensor must equal the global max");
        CHECK(maxima.onednn_eligible == 100u * 1024u * 1024u,
              "a group of 3 must be excluded and a group of 4 included");
        CHECK(is_monotonic(maxima), "boundary maxima must all be <= any_tensor");
    }

    // ---- Case 4: empty inventory --------------------------------------------
    {
        const path_scoped_maxima empty = zone_scoped_maxima(std::vector<zone_tensor_desc>());

        CHECK(empty.any_tensor == 0, "empty inventory any_tensor must be 0");
        CHECK(empty.onednn_eligible == 0, "empty inventory onednn_eligible must be 0");
        CHECK(empty.cpu_quant_eligible == 0, "empty inventory cpu_quant_eligible must be 0");
        CHECK(empty.dma_streamed == 0, "empty inventory dma_streamed must be 0");
        CHECK(is_monotonic(empty), "empty maxima must all be <= any_tensor");
    }

    // ---- Case 5: singletons only --------------------------------------------
    // No per-layer family exists, so the path-scoped maxima must be 0 rather
    // than silently falling back to the global max. A silent fallback is the
    // exact failure mode this whole unit exists to make impossible.
    {
        std::vector<zone_tensor_desc> inventory;
        inventory.push_back(desc("blk.99.some_weight", GPT_OSS_VOCAB_BYTES, TYPE_Q8_0, 2880, 201088, 1, 1));
        inventory.push_back(desc("blk.98.some_weight", MISTRAL_EMBD_BYTES, TYPE_Q4_0, 4096, 32000, 1, 1));

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.any_tensor == GPT_OSS_VOCAB_BYTES, "singletons any_tensor must equal the global max");
        CHECK(maxima.onednn_eligible == 0, "an inventory with no per-layer family must yield 0, not the global max");
        CHECK(maxima.cpu_quant_eligible == 0, "singleton-only cpu_quant_eligible must be 0");
        CHECK(maxima.dma_streamed == 0, "singleton-only dma_streamed must be 0");
        CHECK(is_monotonic(maxima), "singleton maxima must all be <= any_tensor");
    }

    // ---- Case 6: shapeless entries ------------------------------------------
    // Eight shapeless entries share a type. If has_shape did not gate grouping
    // they would all key on ne = {0,0,0,0}, form a family of 8, clear the
    // threshold, and drag every path-scoped maximum up to their size.
    {
        std::vector<zone_tensor_desc> inventory;
        for (int i = 0; i < 8; i++) {
            inventory.push_back(shapeless_desc("shapeless." + std::to_string(i), 800u * 1024u * 1024u, TYPE_Q8_0));
        }
        for (int i = 0; i < 4; i++) {
            inventory.push_back(
                desc("token_embd.weight." + std::to_string(i), 50u * 1024u * 1024u, TYPE_Q8_0, 256, 256, 1, 1));
        }

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.any_tensor == 800u * 1024u * 1024u, "shapeless entries must still count toward any_tensor");
        CHECK(maxima.onednn_eligible == 50u * 1024u * 1024u, "shapeless entries must never form a per-layer family");
        CHECK(is_monotonic(maxima), "shapeless maxima must all be <= any_tensor");
        // A shapeless entry has no reorder size either (the adapter returns 0
        // when has_shape() is false), so it cannot inflate the expanded maximum
        // any more than it can the stored one.
        CHECK(maxima.onednn_reorder == 256ull * 256ull * F16_BYTES,
              "shapeless entries must never inflate onednn_reorder");

        // Shapeless-only: nothing can be classified, so every scoped max is 0.
        std::vector<zone_tensor_desc> only_shapeless;
        for (int i = 0; i < 8; i++) {
            only_shapeless.push_back(shapeless_desc("shapeless." + std::to_string(i), 800u * 1024u * 1024u, TYPE_Q8_0));
        }

        const path_scoped_maxima shapeless_maxima = zone_scoped_maxima(only_shapeless);

        CHECK(shapeless_maxima.any_tensor == 800u * 1024u * 1024u,
              "shapeless-only any_tensor must equal the global max");
        CHECK(shapeless_maxima.onednn_eligible == 0, "shapeless-only onednn_eligible must be 0");
        CHECK(shapeless_maxima.cpu_quant_eligible == 0, "shapeless-only cpu_quant_eligible must be 0");
        CHECK(shapeless_maxima.dma_streamed == 0, "shapeless-only dma_streamed must be 0");
        CHECK(shapeless_maxima.onednn_reorder == 0, "shapeless-only onednn_reorder must be 0");
        CHECK(is_monotonic(shapeless_maxima), "shapeless-only maxima must all be <= any_tensor");
    }

    // ---- Case 7: mispredict accounting --------------------------------------
    // The counters are the plan's own regression detector: a predicate that is
    // wrong for some future model degrades into "grow every time", which is
    // slower than the over-provision this sizing removed and looks exactly like
    // an unrelated regression unless it is counted.
    {
        ggml_sycl::zone_sizing_reset_underestimates();
        CHECK(ggml_sycl::zone_sizing_underestimate_count("onednn") == 0, "counter must start at zero");

        ggml_sycl::zone_sizing_record_underestimate("onednn", 300u * 1024u * 1024u, 160u * 1024u * 1024u);
        ggml_sycl::zone_sizing_record_underestimate("onednn", 200u * 1024u * 1024u, 160u * 1024u * 1024u);
        CHECK(ggml_sycl::zone_sizing_underestimate_count("onednn") == 2, "two records must count as two");
        CHECK(ggml_sycl::zone_sizing_underestimate_count("dma") == 0, "unrelated path must stay at zero");

        // The worst overshoot is what sizes the fix, so it must be retained.
        CHECK(ggml_sycl::zone_sizing_max_underestimate_bytes("onednn") == 300u * 1024u * 1024u,
              "max underestimate must track the largest request, not the last");

        ggml_sycl::zone_sizing_reset_underestimates();
        CHECK(ggml_sycl::zone_sizing_underestimate_count("onednn") == 0, "reset must clear counters");
        CHECK(ggml_sycl::zone_sizing_max_underestimate_bytes("onednn") == 0, "reset must clear the maximum too");
    }

    // ---- Case 8: the summary is silent on a clean run -----------------------
    // Acceptance criterion, not a nicety: a warning that also fires when
    // nothing is wrong carries no information. The return value is what makes
    // that testable host-only — it is true only when something was reported.
    {
        ggml_sycl::zone_sizing_reset_underestimates();
        CHECK(!ggml_sycl::zone_sizing_log_underestimate_summary(), "an all-zero table must report nothing");

        // Observations alone are not a defect, so they must not break silence.
        ggml_sycl::zone_sizing_record_observation("onednn");
        ggml_sycl::zone_sizing_record_observation("onednn");
        CHECK(!ggml_sycl::zone_sizing_log_underestimate_summary(),
              "observations without an under-estimate must stay silent");

        ggml_sycl::zone_sizing_record_underestimate("onednn", 4096, 2048);
        CHECK(ggml_sycl::zone_sizing_log_underestimate_summary(), "a non-zero counter must be reported");

        ggml_sycl::zone_sizing_reset_underestimates();
        CHECK(!ggml_sycl::zone_sizing_log_underestimate_summary(), "reset must restore silence");
    }

    // ---- Case 9: observations separate "never entered" from "never wrong" ----
    // Both leave the under-estimate counter at zero and they mean opposite
    // things. GPT-OSS 20B never enters reserve_onednn_scratch at all (its MoE
    // work uses a separate PP-MoE oneDNN ring), so a zero there says nothing
    // about whether the oneDNN predicate is correct for that model.
    {
        ggml_sycl::zone_sizing_reset_underestimates();
        CHECK(ggml_sycl::zone_sizing_observation_count("onednn") == 0, "observations must start at zero");

        ggml_sycl::zone_sizing_record_observation("onednn");
        ggml_sycl::zone_sizing_record_observation("onednn");
        ggml_sycl::zone_sizing_record_observation("onednn");
        CHECK(ggml_sycl::zone_sizing_observation_count("onednn") == 3, "three observations must count as three");
        CHECK(ggml_sycl::zone_sizing_underestimate_count("onednn") == 0,
              "an observed path with no miss must still report zero under-estimates");
        CHECK(ggml_sycl::zone_sizing_observation_count("dma") == 0, "an unentered path must report zero observations");

        ggml_sycl::zone_sizing_reset_underestimates();
        CHECK(ggml_sycl::zone_sizing_observation_count("onednn") == 0, "reset must clear observations too");

        // A caller that already counts its own uses reports them in one call (one mutex take, not one per use).
        ggml_sycl::zone_sizing_record_observations("mmq-src1-q8", 1387);
        ggml_sycl::zone_sizing_record_observations("mmq-src1-q8", 0);
        CHECK(ggml_sycl::zone_sizing_observation_count("mmq-src1-q8") == 1387,
              "a batched observation must add its whole count, and a zero batch must add nothing");
        CHECK(ggml_sycl::zone_sizing_underestimate_count("mmq-src1-q8") == 0,
              "a batched observation is not an under-estimate");
        ggml_sycl::zone_sizing_reset_underestimates();
    }

    // ---- Case 10: classifier collapse ---------------------------------------
    // The failure mode no assert can catch. If the inventory adapter ever
    // stopped carrying type / ne / has_shape into zone_tensor_desc, every
    // tensor would key uniquely, no group would clear the threshold, every
    // path-scoped maximum would stop narrowing, and the zones would revert to
    // the global-max sizing this unit exists to remove — with every existing
    // check, both correctness gates and this whole test file still passing.
    // This tests zone_scoped_maxima's OUTPUT, which is why it is testable here
    // while the adapter itself (placement_tensor_info lives in the SYCL-side
    // unified-cache.hpp) is not.
    {
        // Healthy: both reference layouts narrow, so neither may signal.
        std::vector<zone_tensor_desc> healthy;
        for (int i = 0; i < 64; i++) {
            healthy.push_back(desc("family." + std::to_string(i), MISTRAL_FFN_BYTES, TYPE_Q4_0, 4096, 14336, 1, 1));
        }
        healthy.push_back(desc("singleton.0", MISTRAL_OUTPUT_BYTES, TYPE_Q6_K, 4096, 32000, 1, 1));

        CHECK(ggml_sycl::zone_detect_collapse(healthy, zone_scoped_maxima(healthy)) ==
                  ggml_sycl::zone_collapse_signal::NONE,
              "a healthy per-layer layout must not signal collapse");

        // Adapter dropped has_shape: nothing can be grouped at all.
        std::vector<zone_tensor_desc> shapeless;
        for (int i = 0; i < 64; i++) {
            shapeless.push_back(shapeless_desc("family." + std::to_string(i), MISTRAL_FFN_BYTES, TYPE_Q4_0));
        }
        CHECK(ggml_sycl::zone_detect_collapse(shapeless, zone_scoped_maxima(shapeless)) ==
                  ggml_sycl::zone_collapse_signal::NO_FAMILY,
              "an inventory with no usable shape must signal NO_FAMILY");

        // Adapter dropped ne: every entry keys uniquely, so every group is a
        // singleton and no maximum survives the threshold.
        std::vector<zone_tensor_desc> all_distinct;
        for (int i = 0; i < 64; i++) {
            all_distinct.push_back(
                desc("distinct." + std::to_string(i), MISTRAL_FFN_BYTES, TYPE_Q4_0, 4096, 1024 + i, 1, 1));
        }
        CHECK(ggml_sycl::zone_detect_collapse(all_distinct, zone_scoped_maxima(all_distinct)) ==
                  ggml_sycl::zone_collapse_signal::NO_FAMILY,
              "an inventory of all-distinct shapes must signal NO_FAMILY");

        // Adapter zeroed ne but left has_shape true: every entry of a type keys
        // identically, one spurious family swallows the inventory, and every
        // maximum equals the global one — narrowing nothing.
        std::vector<zone_tensor_desc> degenerate;
        for (int i = 0; i < 64; i++) {
            degenerate.push_back(desc("degenerate." + std::to_string(i), MISTRAL_FFN_BYTES, TYPE_Q4_0, 1, 1, 1, 1));
        }
        const path_scoped_maxima degenerate_maxima = zone_scoped_maxima(degenerate);
        CHECK(degenerate_maxima.onednn_eligible == degenerate_maxima.any_tensor,
              "the degenerate fixture must in fact narrow nothing");
        CHECK(ggml_sycl::zone_detect_collapse(degenerate, degenerate_maxima) ==
                  ggml_sycl::zone_collapse_signal::NO_NARROWING,
              "an inventory that narrows nothing must signal NO_NARROWING");

        // A genuinely small model of singletons is legitimate and must stay
        // quiet: below the minimum inventory size there is no evidence either
        // way, and this diagnostic must never fire on a model it cannot judge.
        std::vector<zone_tensor_desc> too_small;
        for (size_t i = 0; i < ggml_sycl::k_zone_collapse_min_inventory - 1; i++) {
            too_small.push_back(desc("small." + std::to_string(i), MISTRAL_FFN_BYTES, TYPE_Q4_0, 4096, 1024 + i, 1, 1));
        }
        CHECK(ggml_sycl::zone_detect_collapse(too_small, zone_scoped_maxima(too_small)) ==
                  ggml_sycl::zone_collapse_signal::NONE,
              "an inventory too small to contain a family must not signal collapse");

        CHECK(ggml_sycl::zone_detect_collapse(std::vector<zone_tensor_desc>(), path_scoped_maxima()) ==
                  ggml_sycl::zone_collapse_signal::NONE,
              "an empty inventory must not signal collapse");
    }

    // ---- Case 11: the collapse detector's boolean shape ---------------------
    // This case deliberately does NOT call zone_scoped_maxima, and must not be
    // "simplified" to do so.
    //
    // All three path predicates currently delegate to zone_is_per_layer_weight
    // with the same arguments, so every maxima struct zone_scoped_maxima can
    // produce has onednn_eligible == cpu_quant_eligible == dma_streamed. Over
    // three pairwise-identical comparisons `a && b && c` is logically the same
    // as `a || b || c`, which makes the detector's "every path" conditions
    // indistinguishable from "any path" through any fixture built that way --
    // mutating && to || passes every other case in this file. Constructing
    // path_scoped_maxima by hand is the only way to reach the diverged inputs
    // that zone-sizing.hpp promises the predicates will eventually produce, and
    // it pins the shape before divergence makes the bug reachable. Routing this
    // back through zone_scoped_maxima would silently delete the coverage while
    // leaving the case looking like it still tests something.
    //
    // What the wrong shape would cost: `||` fires NO_FAMILY on a healthy model
    // as soon as any single path legitimately classifies nothing -- a false
    // positive on the one diagnostic whose entire value is staying silent.
    {
        // Only the entry count is read from the inventory here; the maxima are
        // supplied directly. It just has to clear the minimum-size guard.
        const std::vector<zone_tensor_desc> big(ggml_sycl::k_zone_collapse_min_inventory,
                                                desc("entry", MISTRAL_FFN_BYTES, TYPE_Q4_0, 4096, 14336, 1, 1));

        // One path classified nothing; the other two narrowed. Not a collapse.
        path_scoped_maxima one_path_unclassified;
        one_path_unclassified.any_tensor         = MISTRAL_OUTPUT_BYTES;
        one_path_unclassified.onednn_eligible    = 0;
        one_path_unclassified.cpu_quant_eligible = MISTRAL_FFN_BYTES;
        one_path_unclassified.dma_streamed       = MISTRAL_FFN_BYTES;
        CHECK(ggml_sycl::zone_detect_collapse(big, one_path_unclassified) == ggml_sycl::zone_collapse_signal::NONE,
              "one path classifying nothing while the others narrow is not a collapse");

        // One path reached the global max; the other two narrowed. Also not a
        // collapse -- that path's largest eligible tensor is simply the largest
        // tensor in the model.
        path_scoped_maxima one_path_unnarrowed;
        one_path_unnarrowed.any_tensor         = MISTRAL_OUTPUT_BYTES;
        one_path_unnarrowed.onednn_eligible    = MISTRAL_OUTPUT_BYTES;
        one_path_unnarrowed.cpu_quant_eligible = MISTRAL_FFN_BYTES;
        one_path_unnarrowed.dma_streamed       = MISTRAL_FFN_BYTES;
        CHECK(ggml_sycl::zone_detect_collapse(big, one_path_unnarrowed) == ggml_sycl::zone_collapse_signal::NONE,
              "one path reaching the global max while the others narrow is not a collapse");
    }

    // ---- Case 12: the dense MMQ/MMVQ Q8_1 src1 scratch (llama.cpp-479i) -----
    // A dense quantized MUL_MAT quantizes its activations to Q8_1 into a scratch
    // that nobody planned: Qwen3.6-27B on the B50 with oneDNN PP off minted 122
    // raw device allocations (508.5 MB) of it and ran the card out. Every golden
    // below is a size that incident logged: 512 rows * K * 36/32 plus the 32 B
    // Q6_K scale overflow pad, K = 17408 (ffn_down) and K = 5120 (hidden).
    {
        size_t row = 0;
        CHECK(ggml_sycl::zone_mmq_src1_row_bytes(17408, &row) && row == 19584,
              "K=17408 is 544 Q8_1 blocks of 36 bytes = 19584 bytes per row");
        CHECK(ggml_sycl::zone_mmq_src1_row_bytes(5120, &row) && row == 5760, "K=5120 is 160 blocks = 5760 bytes");
        // K is padded to 512 before blocking: a short K costs a whole padded row.
        CHECK(ggml_sycl::zone_mmq_src1_row_bytes(128, &row) && row == 576, "K=128 pads to 512 = 16 blocks = 576 bytes");
        CHECK(ggml_sycl::zone_mmq_src1_row_bytes(2880, &row) && row == 3456, "K=2880 pads to 3072 = 96 blocks");
        CHECK(!ggml_sycl::zone_mmq_src1_row_bytes(0, &row), "K=0 is not a matmul operand");
        CHECK(!ggml_sycl::zone_mmq_src1_row_bytes(-1, &row), "a negative K is refused");
        CHECK(!ggml_sycl::zone_mmq_src1_row_bytes(17408, nullptr), "a null out is refused");

        // The exact per-op figure the dispatch computes (required_size).
        size_t need = 0;
        CHECK(ggml_sycl::zone_mmq_src1_required_bytes(512, 17408, false, &need) && need == 10027008,
              "AOS ffn_down at 512 rows is the logged 10027008");
        CHECK(ggml_sycl::zone_mmq_src1_required_bytes(512, 17408, true, &need) && need == 10027040,
              "SOA ffn_down at 512 rows is the logged 10027040");
        CHECK(ggml_sycl::zone_mmq_src1_required_bytes(512, 5120, false, &need) && need == 2949120,
              "AOS hidden at 512 rows is the logged 2949120");
        CHECK(ggml_sycl::zone_mmq_src1_required_bytes(512, 5120, true, &need) && need == 2949152,
              "SOA hidden at 512 rows is the logged 2949152");
        // A src1 with no rows has no demand and is refused here, so the graph-entry walk must skip it by its own
        // predicate (a ubatch with no outputs trims the last layer to zero rows) rather than treat the refusal as an
        // overflow.
        CHECK(!ggml_sycl::zone_mmq_src1_required_bytes(0, 4096, true, &need), "zero rows must be refused, not sized");
        CHECK(!ggml_sycl::zone_mmq_src1_required_bytes(-1, 4096, true, &need), "negative rows must be refused");
        CHECK(!ggml_sycl::zone_mmq_src1_required_bytes(INT64_MAX / 2, 17408, true, &need),
              "an overflowing row count is refused, not wrapped into a small size");

        // bytes per token, which is what the inventory adapter hands the maxima.
        size_t bpt = 0;
        CHECK(ggml_sycl::zone_mmq_src1_bytes_per_token(17408, 1, 1, &bpt) && bpt == 19584,
              "a 2-D weight costs one Q8_1 row per token");
        // MLA-shaped dense 3-D weight, e.g. wk_b {qk_nope=128, kv_lora_rank=512, n_head=128}: a
        // dense batched MUL_MAT operand whose src1 has n_tokens * n_head rows, NOT an expert stack.
        CHECK(ggml_sycl::zone_mmq_src1_bytes_per_token(128, 128, 1, &bpt) && bpt == 576 * 128,
              "a dense 3-D weight costs ne[2] Q8_1 rows per token");
        CHECK(ggml_sycl::zone_mmq_src1_bytes_per_token(128, 8, 4, &bpt) && bpt == 576 * 8 * 4,
              "ne[3] multiplies the rows as well");

        // The plan figure: n_ubatch * bytes per token plus the pad, aligned to 256.
        size_t plan = 0;
        CHECK(ggml_sycl::zone_mmq_src1_scratch_bytes(19584, 512, &plan) && plan == 10027264,
              "ffn_down at n_ubatch 512 plans 10027040 aligned up to 256");
        CHECK(ggml_sycl::zone_mmq_src1_scratch_bytes(576 * 128, 512, &plan) && plan == 37748992,
              "the MLA 3-D case at n_ubatch 512 plans 37748768 aligned up to 256");
        CHECK(plan >= 512u * 576u * 128u, "the plan covers the exact MLA demand, it must not under-reserve it");
        CHECK(ggml_sycl::zone_mmq_src1_scratch_bytes(0, 512, &plan) && plan == 0,
              "a model with no dense quantized operand plans nothing");
        CHECK(!ggml_sycl::zone_mmq_src1_scratch_bytes(SIZE_MAX / 2, 512, &plan),
              "an overflowing plan is refused, not wrapped into a small size");

        // The maximum over the inventory. The classifier is told by the adapter which
        // tensors are dense MUL_MAT operands (field > 0); it never decides that from
        // ne[2] > 1, which is the expert predicate and misclassifies MLA wk_b / wv_b.
        // Names are deliberately wrong here too.
        std::vector<zone_tensor_desc> inv;
        zone_tensor_desc              ffn_down = desc("blk.0.attn_q.weight", 70000000, TYPE_Q6_K, 17408, 5120, 1, 1);
        ffn_down.mmq_src1_bytes_per_token      = 19584;
        zone_tensor_desc mla_wk_b              = desc("token_embd.weight", 9000000, TYPE_Q8_0, 128, 512, 128, 1);
        mla_wk_b.mmq_src1_bytes_per_token      = 576 * 128;
        // An expert stack the adapter did not mark: ne[2] is large and K is large, and it
        // must not contribute (its own moe_q8 workspace sizes it).
        zone_tensor_desc experts = desc("blk.0.ffn_down.weight", GPT_OSS_EXPERT_BYTES, TYPE_MXFP4, 2880, 2880, 32, 1);
        zone_tensor_desc lm_head = desc("output.weight", MISTRAL_OUTPUT_BYTES, TYPE_Q6_K, 4096, 32000, 1, 1);
        lm_head.mmq_src1_bytes_per_token = 4608;  // a singleton still counts: it IS a MUL_MAT src0
        inv.push_back(ffn_down);
        inv.push_back(mla_wk_b);
        inv.push_back(experts);
        inv.push_back(lm_head);
        inv.push_back(shapeless_desc("mystery", 1, TYPE_Q4_0));
        const path_scoped_maxima m = zone_scoped_maxima(inv);
        CHECK(m.mmq_src1_bytes_per_token == 576 * 128,
              "the maximum is the MLA 3-D dense operand, not the 2-D one and not the expert stack");
        CHECK(zone_scoped_maxima(std::vector<zone_tensor_desc>()).mmq_src1_bytes_per_token == 0,
              "an empty inventory plans no Q8 scratch");
    }

    // ---- Case 13: the dense f16 dequant scratch (llama.cpp-479i) --
    // With oneDNN PP off, a dense Q8_0 MUL_MAT still routes through the f16 dequant arm
    // (ONEDNN_SOA / ONEDNN_COALESCED are selected independent of that knob) and minted its f16
    // copy of the WHOLE weight from the SCRATCH pool per op: Qwen3.6-27B on the B50 logged 11
    // raw 60 MiB allocations (ssm_out, Q8_0 6144x5120) behind a queue nothing drains. The
    // goldens are that weight and the largest Q8_0 weight in the same model (10240x5120).
    {
        size_t w = 0;
        CHECK(ggml_sycl::zone_dequant_f16_weight_bytes(6144, 5120, &w) && w == 62914560,
              "Q8_0 6144x5120 dequantizes to 60 MiB of f16");
        CHECK(ggml_sycl::zone_dequant_f16_weight_bytes(10240, 5120, &w) && w == 104857600,
              "Q8_0 10240x5120 dequantizes to 100 MiB of f16");
        CHECK(!ggml_sycl::zone_dequant_f16_weight_bytes(0, 5120, &w), "a zero extent is not a weight");
        CHECK(!ggml_sycl::zone_dequant_f16_weight_bytes(INT64_MAX / 2, INT64_MAX / 2, &w),
              "an overflowing weight is refused, not wrapped into a small size");
        CHECK(!ggml_sycl::zone_dequant_f16_weight_bytes(6144, 5120, nullptr), "a null out is refused");

        size_t bpt = 0;
        CHECK(ggml_sycl::zone_dequant_f16_src1_bytes_per_token(6144, 1, 1, &bpt) && bpt == 12288,
              "one f16 activation row per token: 6144 * 2 bytes");
        CHECK(ggml_sycl::zone_dequant_f16_src1_bytes_per_token(128, 8, 4, &bpt) && bpt == 128 * 2 * 8 * 4,
              "ne[2] and ne[3] multiply the activation rows");

        // The exact per-buffer figures the dispatch computes: one buffer per f16 copy, 256-aligned.
        size_t need = 0;
        CHECK(ggml_sycl::zone_dequant_f16_region_bytes(6144LL * 5120, &need) && need == 62914560,
              "ssm_out's src0 copy is its 60 MiB of f16, already aligned");
        CHECK(ggml_sycl::zone_dequant_f16_region_bytes(512LL * 6144, &need) && need == 6291456,
              "512 tokens of K=6144 f16 rows are 6 MiB");
        CHECK(ggml_sycl::zone_dequant_f16_region_bytes(0, &need) && need == 0,
              "an operand needing no copy costs nothing");
        CHECK(ggml_sycl::zone_dequant_f16_region_bytes(3, &need) && need == 256, "a buffer is aligned up to 256");
        CHECK(ggml_sycl::zone_dequant_f16_region_bytes(128, &need) && need == 256, "128 halves are exactly 256 bytes");
        CHECK(!ggml_sycl::zone_dequant_f16_region_bytes(-1, &need), "a negative count is refused");
        CHECK(!ggml_sycl::zone_dequant_f16_region_bytes(INT64_MAX, &need),
              "an overflowing demand is refused, not wrapped into a small size");
        CHECK(!ggml_sycl::zone_dequant_f16_region_bytes(1, nullptr), "a null out is refused");

        // The plan figure, per buffer: the largest weight copy, and n_ubatch activation rows at the widest K,
        // each 256-aligned. They are separate numbers because they are separate buffers: the graph walk ensures
        // each at max(plan, demand) so a later graph never regrows (and retires) a buffer a recorded graph baked.
        size_t plan0 = 0, plan1 = 0;
        CHECK(ggml_sycl::zone_dequant_f16_plan_bytes(104857600, 20480, 512, &plan0, &plan1) && plan0 == 104857600 &&
                  plan1 == 10485760,
              "100 MiB weights + 512 tokens of K=10240 f16 rows");
        CHECK(plan0 >= 6144u * 5120u * 2u && plan1 >= 512u * 6144u * 2u,
              "each buffer covers the incident op's exact demand");
        CHECK(ggml_sycl::zone_dequant_f16_plan_bytes(100, 100, 1, &plan0, &plan1) && plan0 == 256 && plan1 == 256,
              "each buffer is aligned up to 256");
        CHECK(ggml_sycl::zone_dequant_f16_plan_bytes(0, 0, 512, &plan0, &plan1) && plan0 == 0 && plan1 == 0,
              "a model with no dense dequant candidate plans nothing");
        CHECK(!ggml_sycl::zone_dequant_f16_plan_bytes(SIZE_MAX / 2, SIZE_MAX / 2, 512, &plan0, &plan1),
              "an overflowing plan is refused, not wrapped into a small size");
        CHECK(!ggml_sycl::zone_dequant_f16_plan_bytes(1, 1, 1, nullptr, &plan1) &&
                  !ggml_sycl::zone_dequant_f16_plan_bytes(1, 1, 1, &plan0, nullptr),
              "a null out is refused");

        // The maxima take the adapter's marks; the classifier decides nothing from ne[2] or names.
        std::vector<zone_tensor_desc> inv;
        zone_tensor_desc              ssm_out = desc("blk.0.attn_q.weight", 33000000, TYPE_Q8_0, 6144, 5120, 1, 1);
        ssm_out.dequant_f16_weight_bytes      = 62914560;
        ssm_out.dequant_f16_src1_bytes_per_token = 12288;
        zone_tensor_desc wide                 = desc("token_embd.weight", 55000000, TYPE_Q8_0, 10240, 5120, 1, 1);
        wide.dequant_f16_weight_bytes         = 104857600;
        wide.dequant_f16_src1_bytes_per_token = 20480;
        zone_tensor_desc unmarked             = desc("blk.1.ffn_down.weight", 90000000, TYPE_Q4_K, 17408, 5120, 1, 1);
        inv.push_back(ssm_out);
        inv.push_back(wide);
        inv.push_back(unmarked);
        const path_scoped_maxima m = zone_scoped_maxima(inv);
        CHECK(m.dequant_f16_weight_bytes == 104857600, "the largest marked weight, not the unmarked Q4_K one");
        CHECK(m.dequant_f16_src1_bytes_per_token == 20480, "the widest marked K");
        CHECK(zone_scoped_maxima(std::vector<zone_tensor_desc>()).dequant_f16_weight_bytes == 0,
              "an empty inventory plans no dequant scratch");
    }

    // ---- Case 14a: oneDNN PP scratch admission (llama.cpp-8ony) ---------------
    // GPT-OSS 20B on the B50, perplexity -c 512 -ub 512: the ONEDNN zone is 256 MiB and the LM head
    // (output.weight, Q8_0 2880 x 201088) wants a 1104.6 MiB f16 weight copy plus 256 x 2880 f16 activations.
    // The head is outside the ONEDNN zone's sizing and inside the RUNTIME dequant plan, so it must not be
    // sent to the oneDNN scratch: the arena refuses to grow the zone once weights are resident.
    {
        const size_t zone_256mib = 256u * 1024u * 1024u;
        const size_t head_w      = 1158266880;  // 2880 x 201088 x 2
        const size_t head_a      = 1474560;     // 256 x 2880 x 2
        const size_t attn_w      = 23592960;    // 2880 x 4096 x 2
        const size_t attn_a      = 2949120;     // 512 x 2880 x 2

        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, zone_256mib, attn_w, attn_a),
              "a per-layer attention weight fits the 256 MiB ONEDNN zone");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, zone_256mib, head_w, head_a),
              "the LM head does not fit the ONEDNN zone, so it is not planned there");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(false, 0, head_w, head_a),
              "with no arena there is no zone to disagree with: the unified-cache path serves it");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, 100, 60, 40), "a pair that exactly fills the zone fits");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, 100, 60, 41), "one byte over the zone does not fit");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, 0, 1, 1), "an empty zone holds nothing");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, SIZE_MAX, SIZE_MAX, 2),
              "an overflowing pair is refused, not wrapped into a small sum");
    }

    // ---- Case 14b: a smaller request never shrinks a held oneDNN scratch (llama.cpp-8ony) ----
    // Perplexity chunk 2, layer 0: a 512-row attention op (weights 23.6 MB, activations 2.9 MB) arrived while the
    // cache held the LM head's pair (weights 1104.6 MiB, activations 1.4 MB for 256 rows). Replacing the pair by
    // the request freed the big weights block; the head then had to regrow it and could not.
    {
        const size_t mib = 1024u * 1024u;
        size_t       w   = 0;
        size_t       a   = 0;

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 0, 0, 24 * mib, 3 * mib, &w, &a);
        CHECK(w == 24 * mib && a == 3 * mib, "nothing held: the request is the target");

        ggml_sycl::zone_onednn_scratch_reserve_target(false, 0, 1105 * mib, 1 * mib, 24 * mib, 3 * mib, &w, &a);
        CHECK(w == 1105 * mib, "a smaller weights request never shrinks the held weights block");
        CHECK(a == 3 * mib, "a larger activations request still grows the activations half on its own");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 100 * mib, 2 * mib, 50 * mib, 3 * mib, &w, &a);
        CHECK(w == 100 * mib && a == 3 * mib, "inside the zone the merged pair is the target");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 200 * mib, 2 * mib, 50 * mib, 100 * mib, &w, &a);
        CHECK(w == 50 * mib && a == 100 * mib,
              "a merge that would overflow the zone falls back to the request instead of wedging every op");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, SIZE_MAX, 1, 50 * mib, 3 * mib, &w, &a);
        CHECK(w == 50 * mib && a == 3 * mib, "an unrepresentable merged sum is refused, not wrapped");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 1, 1, 2, 2, nullptr, nullptr);
    }

    // ---- Case 14c: the oneDNN scratch supplies an op only when its type is enabled too (llama.cpp-8ony) -----
    // acquire_onednn_pp_scratch also turns away every type but Q4_0 / Q8_0 / MXFP4 (unless the env var forces it),
    // and every type under GGML_SYCL_ONEDNN_PP_UNIFIED_SCRATCH=0. The graph-entry walk skipped an op on admission
    // plus plan alone, so a Q6_K op (Qwen3.5-9B-UD-Q6_K_XL, Mistral Q4_K_M) was skipped by the walk and refused by
    // acquire, then drew a planned dequant buffer nothing had sized.
    {
        const size_t zone_256mib = 256u * 1024u * 1024u;
        const size_t attn_w      = 23592960;
        const size_t attn_a      = 2949120;

        CHECK(ggml_sycl::zone_onednn_pp_scratch_type_enabled(-1, true), "unset: Q4_0 / Q8_0 / MXFP4 are enabled");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_type_enabled(-1, false), "unset: a K-quant type is not enabled");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_type_enabled(0, true), "=0 turns the scratch off for every type");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_type_enabled(0, false), "=0: a K-quant stays off");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_type_enabled(1, false), "=1 enables a K-quant type too");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_type_enabled(1, true), "=1 keeps the default types on");

        CHECK(ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, zone_256mib, attn_w, attn_a),
              "an admitted, enabled, planned op is supplied by the scratch");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(true, false, true, zone_256mib, attn_w, attn_a),
              "an op whose type is not enabled is NOT supplied, so the walk must size its planned dequant buffers");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(true, false, false, 0, attn_w, attn_a),
              "type refusal holds with no arena too");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(false, true, true, zone_256mib, attn_w, attn_a),
              "an op that fails the PP admission is not supplied");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, zone_256mib, 1158266880, 1474560),
              "an enabled op whose pair is over the zone is not supplied");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, 100, 60, 40),
              "a pair that exactly fills the zone is supplied");
    }

    // ---- Case 14d: the unified kernel's oneDNN f16 route draws the planned dequant buffers (llama.cpp-8ony) ----
    // A Q4_0 / MXFP4 dense op the unified kernel serves, outside a layer group (an LM head or tied embedding) with
    // a pair over the ONEDNN zone: acquire refuses it, the route fell back to a per-op pool copy of the whole
    // weight, and the walk never counted it (a unified-served node was "not counted").
    {
        CHECK(ggml_sycl::zone_unified_pp_draws_dequant(true, true, true, true, false),
              "an over-zone unified-served op draws the planned dequant buffers");
        CHECK(!ggml_sycl::zone_unified_pp_draws_dequant(true, true, true, true, true),
              "a unified-served op the scratch supplies draws nothing from them");
        CHECK(!ggml_sycl::zone_unified_pp_draws_dequant(false, true, true, true, false),
              "a node the router did not send to the unified kernel is the legacy route's business");
        CHECK(!ggml_sycl::zone_unified_pp_draws_dequant(true, false, true, true, false),
              "a type the unified kernel does not serve never reaches its oneDNN f16 route");
        CHECK(!ggml_sycl::zone_unified_pp_draws_dequant(true, true, false, true, false),
              "a non-plain src1 skips the unified route's f16 arm");
        CHECK(!ggml_sycl::zone_unified_pp_draws_dequant(true, true, true, false, false),
              "an op that fails the PP admission never takes the f16 arm");
    }

    // ---- Case 14e: the walk asks the unified route before it filters on precision (llama.cpp-8ony) -----------------
    // The unified kernel's oneDNN f16 route has no precision check, the legacy f16 arm requires GGML_PREC_DEFAULT.
    // A Q4_0 / MXFP4 node with GGML_PREC_F32 (a GLM4 attention output) that the scratch does not supply takes the
    // unified route and draws the planned buffers, so the walk must count it.
    {
        CHECK(ggml_sycl::zone_walk_f16_node_draws(true, true, false), "a default-precision legacy node draws");
        CHECK(ggml_sycl::zone_walk_f16_node_draws(true, false, true), "a default-precision unified node draws");
        CHECK(!ggml_sycl::zone_walk_f16_node_draws(true, false, false), "a node no route draws for draws nothing");
        CHECK(!ggml_sycl::zone_walk_f16_node_draws(false, true, false),
              "a legacy-route node with another precision is not on the legacy f16 arm");
        CHECK(ggml_sycl::zone_walk_f16_node_draws(false, false, true),
              "a unified-route node draws whatever its precision: Route A has no precision check");
        CHECK(ggml_sycl::zone_walk_f16_node_draws(false, true, true), "either arm that draws counts the node");
    }

    // ---- Case 14f: the pair bound leaves the Graph SDPA floor free, and never drops below the plan (llama.cpp-8ony) --
    // The ONEDNN zone is max(256 MiB, plan + floor), so capacity can sit far above plan + floor. A pair is
    // planned up to capacity - floor (slack nobody else reserved), never below the plan, never above the capacity.
    {
        const size_t mib = 1024u * 1024u;
        // (a) a small-model head (stories15M class): plan ~2 MB, head pair ~19 MB, floor ~10 MB, the 256 MiB zone.
        {
            const size_t bound = ggml_sycl::zone_onednn_pp_pair_bound(256 * mib, 2 * mib, 10 * mib);
            CHECK(bound == 246 * mib, "small model: the bound is capacity minus the floor, well above the plan");
            CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, 18 * mib, 1 * mib),
                  "small model: a 19 MB head pair fits the slack, so it stays supplied by the scratch");
        }
        // (b) the Mistral head window: plan 143.5 MB, floor ~33 MB, capacity 256 MiB, head pair 266 MB.
        {
            const size_t plan  = 150470656;  // 143.5 MiB
            const size_t floor = 34603008;   // 33 MiB
            const size_t bound = ggml_sycl::zone_onednn_pp_pair_bound(256 * mib, plan, floor);
            CHECK(bound == 256 * mib - floor, "Mistral: the bound is capacity minus the floor");
            CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, 262144000, 4194304),
                  "Mistral: a 266 MB head pair would eat the Graph SDPA floor, so it is not supplied");
            CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, 117440512, 4194304),
                  "Mistral: a per-layer pair (the plan's own op) is still supplied");
        }
        // (c) a clamped zone: capacity - floor falls under the plan, so the bound is exactly the plan.
        {
            CHECK(ggml_sycl::zone_onednn_pp_pair_bound(100 * mib, 80 * mib, 50 * mib) == 80 * mib,
                  "clamped zone: the bound is the plan itself, not capacity minus the floor");
            CHECK(ggml_sycl::zone_onednn_pp_pair_bound(100 * mib, 80 * mib, 20 * mib) == 80 * mib,
                  "equal reading: capacity minus the floor equals the plan");
        }
        // edges
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(100 * mib, 20 * mib, 0) == 100 * mib,
              "no floor: the whole capacity");
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(10 * mib, 80 * mib, 50 * mib) == 10 * mib,
              "never above the capacity, even when the plan is");
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(100 * mib, 20 * mib, 500 * mib) == 20 * mib,
              "a floor larger than the capacity leaves the plan, with no wrapped subtraction");
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(SIZE_MAX, 0, SIZE_MAX) == 0, "a saturated floor leaves nothing");
    }

    // ---- Case 14g: the dequant plan covers the ops the oneDNN PP scratch will not supply (llama.cpp-8ony) ----------
    // Mistral Q4_0 under the default scratch: the per-layer weights are the ONEDNN zone's own plan and are supplied by
    // the scratch, but the LM head is outside that plan and (over the pair bound) draws the dequant buffers; with
    // GGML_SYCL_ONEDNN_PP_UNIFIED_SCRATCH=0 every dense Q4_0 op draws them. Both used to allocate with planned=0.
    {
        CHECK(!ggml_sycl::zone_dequant_f16_planned_when_unsupplied(true, true),
              "an enabled type's per-layer weight is the zone's own plan: supplied, nothing for the dequant plan");
        CHECK(ggml_sycl::zone_dequant_f16_planned_when_unsupplied(true, false),
              "a weight the zone was not sized for (the LM head) is planned into the dequant buffers");
        CHECK(ggml_sycl::zone_dequant_f16_planned_when_unsupplied(false, true),
              "a type the scratch is off for draws the dequant buffers even when it is a per-layer weight");
        CHECK(ggml_sycl::zone_dequant_f16_planned_when_unsupplied(false, false), "neither: planned");

        const size_t layer_w = 117440512;  // 4096 x 14336 f16
        const size_t head_w  = 262144000;  // 4096 x 32000 f16
        auto         marked  = [&](const char * name, int64_t ne0, int64_t ne1, size_t f16_w, bool enabled) {
            zone_tensor_desc d                               = desc(name, 1000, TYPE_Q4_0, ne0, ne1, 1, 1);
            d.dequant_f16_if_unsupplied_weight_bytes         = f16_w;
            d.dequant_f16_if_unsupplied_src1_bytes_per_token = static_cast<size_t>(ne0) * F16_BYTES;
            d.pp_scratch_type_enabled                        = enabled;
            return d;
        };
        std::vector<zone_tensor_desc> layers;
        for (int i = 0; i < 4; i++) {
            layers.push_back(marked("blk.0.ffn_gate.weight", 4096, 14336, layer_w, true));
        }
        CHECK(zone_scoped_maxima(layers).dequant_f16_weight_bytes == 0 &&
                  zone_scoped_maxima(layers).dequant_f16_src1_bytes_per_token == 0,
              "default scratch, layer weights only: the zone's plan supplies them all, the dequant plan is empty");

        std::vector<zone_tensor_desc> with_head = layers;
        with_head.push_back(marked("output.weight", 4096, 32000, head_w, true));
        const path_scoped_maxima mh = zone_scoped_maxima(with_head);
        CHECK(mh.dequant_f16_weight_bytes == head_w, "the LM head is the dequant plan's weight copy");
        CHECK(mh.dequant_f16_src1_bytes_per_token == 4096 * F16_BYTES, "and its activation row");

        std::vector<zone_tensor_desc> off;
        for (int i = 0; i < 4; i++) {
            off.push_back(marked("blk.0.ffn_gate.weight", 4096, 14336, layer_w, false));
        }
        CHECK(zone_scoped_maxima(off).dequant_f16_weight_bytes == layer_w,
              "scratch off: the layer weights draw the dequant buffers, so they size the plan");

        // The unconditional marks (dense Q8_0) still count, and the two sources take the larger.
        std::vector<zone_tensor_desc> mixed = layers;
        zone_tensor_desc              q8    = desc("blk.1.attn_q.weight", 1000, TYPE_Q8_0, 6144, 5120, 1, 1);
        q8.dequant_f16_weight_bytes         = 62914560;
        q8.dequant_f16_src1_bytes_per_token = 12288;
        mixed.push_back(q8);
        CHECK(zone_scoped_maxima(mixed).dequant_f16_weight_bytes == 62914560,
              "an unconditionally planned Q8_0 weight counts although the Q4_0 layers do not");
        mixed.push_back(marked("output.weight", 4096, 32000, head_w, true));
        CHECK(zone_scoped_maxima(mixed).dequant_f16_weight_bytes == head_w, "the larger of the two sources");

        // An expert stack never gets the mark (the adapter excludes it), so an unmarked tensor changes nothing.
        std::vector<zone_tensor_desc> unmarked = layers;
        unmarked.push_back(desc("output.weight", 1000, TYPE_Q4_0, 4096, 32000, 1, 1));
        CHECK(zone_scoped_maxima(unmarked).dequant_f16_weight_bytes == 0, "no mark, no plan");
    }

    // ---- Case 14h: the figures a zone is described by outlive a later plan (llama.cpp-8ony) -----------------------
    // A draft model loaded beside the target overwrites the planner's live (bare plan, Graph floor) with its own
    // smaller figures. The arena's zones are found sufficient and kept, so the zone is still the target's: the
    // snapshot it is described by must keep the larger of each figure, or the bound it yields would describe the
    // draft's zone (a head pair then eats the target's Graph SDPA floor, or a clamped zone's bound drops below the
    // target's own planned pair).
    {
        const size_t           mib    = 1024u * 1024u;
        const zone_onednn_plan target = { 144 * mib, 64 * mib };
        const zone_onednn_plan draft  = { 30 * mib, 10 * mib };
        const zone_onednn_plan kept   = ggml_sycl::zone_onednn_plan_keep(target, draft);
        CHECK(kept.bare_bytes == 144 * mib && kept.graph_floor_bytes == 64 * mib,
              "a smaller later plan does not shrink what the kept zone is described by");
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(256 * mib, kept.bare_bytes, kept.graph_floor_bytes) == 192 * mib,
              "the bound stays the target's: capacity minus the target's floor");
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(256 * mib, draft.bare_bytes, draft.graph_floor_bytes) == 246 * mib,
              "the draft's live figures alone would have admitted a pair that eats the target's Graph floor");
        const zone_onednn_plan grown = ggml_sycl::zone_onednn_plan_keep(draft, target);
        CHECK(grown.bare_bytes == 144 * mib && grown.graph_floor_bytes == 64 * mib,
              "a larger later plan the zone was found sufficient for is described by its own figures");
        const zone_onednn_plan wide_plan  = { 144 * mib, 10 * mib };
        const zone_onednn_plan wide_floor = { 30 * mib, 64 * mib };
        const zone_onednn_plan mixed      = ggml_sycl::zone_onednn_plan_keep(wide_plan, wide_floor);
        CHECK(mixed.bare_bytes == 144 * mib && mixed.graph_floor_bytes == 64 * mib,
              "each figure keeps its own larger value");
        const zone_onednn_plan none = ggml_sycl::zone_onednn_plan_keep(zone_onednn_plan(), draft);
        CHECK(none.bare_bytes == draft.bare_bytes && none.graph_floor_bytes == draft.graph_floor_bytes,
              "with nothing held, the live plan is what the zone is described by");

        // The reserve merges per-component maxima, so two ops that each fit the bound can hold a pair above it. The
        // merge is bounded by the PAIR BOUND, not the capacity: the held pair would otherwise eat the Graph floor.
        const size_t bound = 235 * mib;
        size_t       w = 0, a = 0;
        ggml_sycl::zone_onednn_scratch_reserve_target(true, bound, 200 * mib, 4 * mib, 40 * mib, 40 * mib, &w, &a);
        CHECK(w == 40 * mib && a == 40 * mib, "a merge above the bound is not held: the request is used as asked");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 200 * mib, 4 * mib, 40 * mib, 40 * mib, &w, &a);
        CHECK(w == 200 * mib && a == 40 * mib,
              "the same two ops against the raw capacity merge to 240 MiB, past the 235 MiB bound");
    }

    // ---- Case 14i: the first reservation is the planned pair, so it never regrows (llama.cpp-8ony) ----------------
    // The zone is sized for one pair plus the Graph floor. A reservation that starts at the first op's size and grows
    // stepwise needs the superseded block and the new one in the zone at once (the release barrier holds the old
    // one), which does not fit: the regrow fell through to an unplanned direct allocation. So the first reservation is
    // max(request, planned) per component, and every later planned op is reused from it.
    {
        const size_t mib = 1024u * 1024u;
        size_t       w = 0, a = 0;
        const size_t plan_w = 112 * mib;  // the largest dequantized per-layer weight
        const size_t plan_a = 32 * mib;   // the activations half
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, 0, 0, plan_w, plan_a, 80 * mib, 14 * mib, &w, &a);
        CHECK(w == plan_w && a == plan_a, "nothing held: the first reservation is the planned pair, not the first op's");
        const size_t first_w = w, first_a = a;
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, first_w, first_a, plan_w, plan_a, 112 * mib,
                                                      28 * mib, &w, &a);
        CHECK(w == first_w && a == first_a, "a later planned op, even the widest, is covered by what is held");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, 0, 0, plan_w, plan_a, 120 * mib, 14 * mib, &w, &a);
        CHECK(w == 120 * mib && a == plan_a, "a request above the plan in one half still grows that half");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 100 * mib, 0, 0, plan_w, plan_a, 40 * mib, 10 * mib, &w, &a);
        CHECK(w == 40 * mib && a == 10 * mib, "a zone clamped below its plan reserves what is asked, not the plan");
        ggml_sycl::zone_onednn_scratch_reserve_target(false, 0, 0, 0, plan_w, plan_a, 40 * mib, 10 * mib, &w, &a);
        CHECK(w == 40 * mib && a == 10 * mib, "without an arena there is no planned pair to reserve");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, 0, 0, plan_w, plan_a, 40 * mib, 10 * mib, nullptr,
                                                      nullptr);
        // the pair halves are part of the snapshot a kept zone is described by
        const zone_onednn_plan big   = { 144 * mib, 64 * mib, plan_w, plan_a };
        const zone_onednn_plan small = { 30 * mib, 10 * mib, 20 * mib, 4 * mib };
        const zone_onednn_plan kept  = ggml_sycl::zone_onednn_plan_keep(big, small);
        CHECK(kept.weights_bytes == plan_w && kept.activations_bytes == plan_a,
              "a smaller later plan does not shrink the pair halves the kept zone was built for");
    }

    // ---- Case 14: the planned dense scratch is ONE reservation that follows the runtime n_ubatch (llama.cpp-kpjw) --
    // B70, full card, Qwen3.6-27B perplexity (-c 512 gives n_ctx 2048 with 4 sequences, n_batch 2048): the plan was
    // sized at the load-time n_ubatch (512, 10027264 B), auto-ubatch then chose 2048 (40108288 B), and the compute
    // buffers of the 2048 rung had already filled the RUNTIME zone ("zone has 0.3 MB free"), so the planned buffer
    // was never allocated and the first graph was refused.
    {
        // The plan is a function of n_ubatch: the same figure the three buffers' own helpers give, summed.
        size_t q8 = 0, src0 = 0, src1 = 0, total = 0;
        CHECK(ggml_sycl::zone_dense_scratch_total_bytes(19584, 0, 0, 512, &total) && total == 10027264,
              "the incident's load-time plan: 512 rows of K=17408");
        CHECK(ggml_sycl::zone_dense_scratch_total_bytes(19584, 0, 0, 2048, &total) && total == 40108288,
              "the runtime rung the incident chose plans 4x the load-time figure");
        CHECK(ggml_sycl::zone_mmq_src1_scratch_bytes(19584, 2048, &q8) &&
                  ggml_sycl::zone_dequant_f16_plan_bytes(104857600, 20480, 2048, &src0, &src1) &&
                  ggml_sycl::zone_dense_scratch_total_bytes(19584, 104857600, 20480, 2048, &total) &&
                  total == q8 + src0 + src1,
              "the total is exactly the three buffers' own plan figures");
        CHECK(ggml_sycl::zone_dense_scratch_total_bytes(0, 0, 0, 2048, &total) && total == 0,
              "a model with no dense candidate plans nothing at any n_ubatch");
        CHECK(!ggml_sycl::zone_dense_scratch_total_bytes(SIZE_MAX / 2, 0, 0, 2048, &total),
              "an overflowing total is refused, not wrapped into a small size");
        CHECK(!ggml_sycl::zone_dense_scratch_total_bytes(19584, 0, 0, 512, nullptr), "a null out is refused");

        // The runtime-context transaction refuses a rung the zone cannot hold, and names the largest that fits.
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 0, 512 * MiB, 4096) == 4096,
              "a 512 MiB RUNTIME zone holds every rung the ladder can try");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 0, 36 * MiB, 4096) == 1920,
              "a 36 MiB zone holds 1920 rows of K=17408, a multiple of 32");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 0, 36 * MiB, 1000) == 992,
              "the search bound is honoured and rounded down to a multiple of 32");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 20 * MiB, 36 * MiB, 4096) == 832,
              "other planned RUNTIME consumers come off the capacity first");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 104857600, 20480, 0, 100 * MiB, 4096) == 0,
              "a weight copy larger than the zone fits no n_ubatch at all");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 40 * MiB, 36 * MiB, 4096) == 0,
              "other consumers larger than the zone leave nothing");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 0, 512 * MiB, 31) == 0,
              "a search bound under one row group finds nothing");
        CHECK(ggml_sycl::zone_dense_scratch_largest_ubatch(0, 0, 0, 0, 512 * MiB, 4096) == 4096,
              "with nothing planned every searched rung fits");
        {
            // The answer is a fixed point of the total: it fits, and the next row group does not.
            const uint32_t ub = ggml_sycl::zone_dense_scratch_largest_ubatch(19584, 0, 0, 0, 36 * MiB, 4096);
            size_t         fits = 0, over = 0;
            CHECK(ggml_sycl::zone_dense_scratch_total_bytes(19584, 0, 0, ub, &fits) && fits <= 36 * MiB &&
                      ggml_sycl::zone_dense_scratch_total_bytes(19584, 0, 0, ub + 32, &over) && over > 36 * MiB,
                  "the largest rung fits and the next row group does not");
        }

        // The hold: the whole plan of every buffer still short of it.
        using ggml_sycl::zone_planned_buffer;
        size_t hold = 1;
        {
            const zone_planned_buffer fresh[] = { { 10027264, 0 }, { 0, 0 }, { 0, 0 } };
            CHECK(ggml_sycl::zone_planned_scratch_hold_bytes(fresh, 3, &hold) && hold == 10027264,
                  "an unmaterialized plan is held in full: the incident's 9.6 MB");
        }
        {
            const zone_planned_buffer met[] = { { 10027264, 10027264 }, { 4096, 8192 } };
            CHECK(ggml_sycl::zone_planned_scratch_hold_bytes(met, 2, &hold) && hold == 0,
                  "a buffer at or above its plan holds nothing back");
        }
        {
            // Growth allocates the replacement while the old backing is live, so the shortfall is not enough.
            const zone_planned_buffer growing[] = { { 40108288, 10027264 }, { 1048576, 1048576 } };
            CHECK(ggml_sycl::zone_planned_scratch_hold_bytes(growing, 2, &hold) && hold == 40108288,
                  "a buffer short of a larger plan holds the whole new plan, not the difference");
        }
        {
            const zone_planned_buffer mixed[] = { { 100, 0 }, { 200, 200 }, { 300, 1 } };
            CHECK(ggml_sycl::zone_planned_scratch_hold_bytes(mixed, 3, &hold) && hold == 400,
                  "only the buffers still short of their plan are summed");
        }
        {
            const zone_planned_buffer huge[] = { { SIZE_MAX, 0 }, { 1, 0 } };
            CHECK(!ggml_sycl::zone_planned_scratch_hold_bytes(huge, 2, &hold), "an overflowing hold is refused");
        }
        CHECK(ggml_sycl::zone_planned_scratch_hold_bytes(nullptr, 0, &hold) && hold == 0, "no buffers hold nothing");
        CHECK(!ggml_sycl::zone_planned_scratch_hold_bytes(nullptr, 0, nullptr), "a null out is refused");

        // A spill-capable request cannot take the held bytes.
        CHECK(!ggml_sycl::zone_runtime_alloc_respects_hold(64 * MiB, 10027264, 64 * MiB),
              "a request that would leave less than the hold free spills");
        CHECK(!ggml_sycl::zone_runtime_alloc_respects_hold(300 * 1024, 10027264, 300 * 1024),
              "the incident: 0.3 MB free, a 9.6 MB hold, nothing may take the 0.3 MB");
        CHECK(ggml_sycl::zone_runtime_alloc_respects_hold(100 * MiB, 10027264, 64 * MiB),
              "a request that leaves the hold free is served from the zone");
        CHECK(ggml_sycl::zone_runtime_alloc_respects_hold(64 * MiB + 10027264, 10027264, 64 * MiB),
              "the boundary is inclusive: exactly the hold is left");
        CHECK(!ggml_sycl::zone_runtime_alloc_respects_hold(64 * MiB + 10027263, 10027264, 64 * MiB),
              "one byte under the boundary spills");
        CHECK(ggml_sycl::zone_runtime_alloc_respects_hold(64 * MiB, 0, 64 * MiB),
              "with no hold the whole zone is available, as before");
        CHECK(!ggml_sycl::zone_runtime_alloc_respects_hold(64 * MiB, 0, 64 * MiB + 1),
              "a request larger than the zone never fits");
        CHECK(!ggml_sycl::zone_runtime_alloc_respects_hold(SIZE_MAX, SIZE_MAX, 1),
              "a hold that wraps must not read as no hold");
    }

    // ---- Case 15: review r1 of llama.cpp-kpjw: the held-back branch, the route a decline serves, merged inputs --
    {
        const size_t MiB  = 1024 * 1024;
        const size_t hold = 10027264;

        // The branch unified_alloc takes. Only a spill-capable request for the RUNTIME zone can be held back.
        CHECK(ggml_sycl::zone_runtime_alloc_held_back(true, false, 300 * 1024, hold, 300 * 1024),
              "the incident: a compute buffer asking a 0.3 MB-free zone is held back and spills");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 100 * MiB, hold, 64 * MiB),
              "a request that leaves the hold free is served from the zone");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, true, 300 * 1024, hold, 300 * 1024),
              "a forbid-spill request is a claimant of the plan and is never held back");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(false, false, 300 * 1024, hold, 300 * 1024),
              "the hold is a RUNTIME-zone fact: no other zone's request is held back");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 300 * 1024, 0, 300 * 1024),
              "with no hold nothing is held back");
        CHECK(ggml_sycl::zone_runtime_alloc_held_back(true, false, SIZE_MAX, SIZE_MAX, 1),
              "a hold that wraps must not read as no hold");
        // Review r2 F1: held back means the ZONE ALONE would have served the request and the hold is what keeps it
        // out. A request the zone cannot hold anyway spills exactly as it always did, with no hold involved, and
        // the overcommit guard keeps evicting for it as it did before the hold existed.
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 50, 0, 100),
              "an ordinary zone-full spill with no hold is not held back");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 50, 10, 100),
              "a request larger than the zone's free bytes spills with or without a hold: not held back");
        CHECK(ggml_sycl::zone_runtime_alloc_held_back(true, false, 100, 10, 95),
              "a request the zone could serve that would eat into the hold is held back");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 100, 0, 95),
              "the same request with no hold is served from the zone");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 100, 10, 90),
              "a request that leaves exactly the hold is served from the zone");
        CHECK(ggml_sycl::zone_runtime_alloc_held_back(true, false, 100, 10, 91),
              "one byte into the hold is held back");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 300 * 1024, hold, 300 * 1024 + 1),
              "the incident's zone-full request one byte over the free bytes is an ordinary spill");
        // The B70 / Qwen3.6-27B run at auto-ub2048 (hardware, review r2): three compute-buffer requests of about
        // 1 GiB each asked a RUNTIME zone with 0.3 MB free while the hold was 9.6 MB. They are ordinary zone-full
        // spills, 2.9 GB in all; none of them is held back by the hold.
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 300 * 1024, 10027264, 1003413 * 1024),
              "a ~1 GiB compute buffer asking a 0.3 MB-free zone is an ordinary spill, not held back");
        CHECK(!ggml_sycl::zone_runtime_alloc_held_back(true, false, 300 * 1024, 10027264, (size_t) 3 << 30),
              "a multi-GB request larger than the free bytes is never held back");
        // Argument order is part of the contract: (runtime, forbid, available, hold, size). The zone's free bytes
        // are the first of the three sizes; swapped with the request, the same numbers answer another question.
        CHECK(ggml_sycl::zone_runtime_alloc_held_back(true, false, 100, 10, 95) &&
                  !ggml_sycl::zone_runtime_alloc_held_back(true, false, 95, 10, 100),
              "available is the zone's free bytes and size is the request, not the other way round");

        // The route a node takes: the walks and the dispatch must agree, including after a runtime decline.
        // (valid, unified, primary draws, fallback valid, fallback draws)
        CHECK(ggml_sycl::zone_route_draws_scratch(true, false, true, false, false),
              "a legacy kernel that draws the scratch is counted");
        CHECK(!ggml_sycl::zone_route_draws_scratch(true, false, false, true, true),
              "a legacy kernel that does not draw it is not counted, whatever the fallback would be");
        CHECK(!ggml_sycl::zone_route_draws_scratch(true, true, false, false, false),
              "a unified kernel the dispatch does not decline draws nothing");
        CHECK(ggml_sycl::zone_route_draws_scratch(true, true, false, true, true),
              "a node the unified kernel declines and a drawing legacy kernel then serves is counted: the f16 gap");
        CHECK(!ggml_sycl::zone_route_draws_scratch(true, true, true, false, true),
              "a decline with no valid legacy fallback draws nothing");
        CHECK(!ggml_sycl::zone_route_draws_scratch(false, false, true, true, true), "an invalid decision draws nothing");

        // Plan inputs are device-global. A second model on the device must not shrink the first one's plan.
        CHECK(ggml_sycl::zone_dense_scratch_merge_input(19584, 4096, true) == 19584,
              "another live model's larger plan input survives the load of a smaller one (draft + target)");
        CHECK(ggml_sycl::zone_dense_scratch_merge_input(4096, 19584, true) == 19584,
              "the larger of two live models wins either way");
        CHECK(ggml_sycl::zone_dense_scratch_merge_input(19584, 4096, false) == 4096,
              "with no other live model the new inputs replace the old: a model swap shrinks the plan");
        CHECK(ggml_sycl::zone_dense_scratch_merge_input(0, 0, true) == 0, "nothing planned stays nothing");
    }

    // ---- Case 16: review r2/r3 (hardware): a rung is refused for its hold-induced spill only when that spill is
    // what pushed the card under the driver headroom. B50, Qwen PPL at auto-ub1024: a 461 MB compute buffer was
    // held back, spilled outside the arena, the card was left with 107.8 MB free against the 256 MB the arena
    // expects outside it, and flash attention then ran out of resources. A card that was ALREADY under the headroom
    // for another reason (a full B70, KB-scale spills) is not blamed on the hold. -----------------------------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(!ggml_sycl::zone_hold_spill_bound_fits(569 * MiB, 256 * MiB, 461 * MiB),
              "the B50 ub1024 rung: 569 MB free before a 461 MB spill, 108 MB after: the spill pushed it under");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(0, 256 * MiB, 0),
              "no hold-induced spill: the check asks nothing, whatever the free memory is");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(256 * MiB + 1, 256 * MiB, 1),
              "a spill that leaves exactly the headroom fits");
        CHECK(!ggml_sycl::zone_hold_spill_bound_fits(256 * MiB, 256 * MiB, 1),
              "one byte below the headroom, and the spill's one byte is what crossed it: blamed");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(4096 * MiB + 461 * MiB, 256 * MiB, 461 * MiB),
              "a spill the card can take with its headroom intact fits: the hold costs a rung only when it must");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(100 * MiB + 300 * 1024, 256 * MiB, 300 * 1024),
              "a full card with a KB-scale spill was under the headroom before the spill: not blamed on the hold");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(1, 256 * MiB, 1),
              "a spill too small to have crossed the headroom is not what made the card short");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(200 * MiB, 256 * MiB, 100 * MiB),
              "the card was short without the hold: its spill is not what made it so");
        CHECK(ggml_sycl::zone_hold_spill_bound_fits(200 * MiB, 256 * MiB, 966 * MiB),
              "a demand larger than the card, on a card already under the headroom, is not the hold's doing either");
        CHECK(!ggml_sycl::zone_hold_spill_bound_fits(300 * MiB, 256 * MiB, SIZE_MAX),
              "an overflowing spill must not read as a small one");
    }

    // ---- Case 17: the spill bound follows the candidate rung (review r3 I1). The largest compute-buffer request
    // is observed at one n_ubatch; a rung above it asks for proportionally more, so the bound scales with the rung. --
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, 230 * MiB, 512, 1024) == 536 * MiB,
              "B50: a 230 MB request seen at ub512 scales to 460 MB at ub1024, plus the 76 MB plan");
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, 230 * MiB, 512, 512) == 306 * MiB,
              "the rung the request was seen at needs no scaling");
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, 460 * MiB, 1024, 512) == 306 * MiB,
              "a smaller rung than the one observed scales down (the settle re-publish of last_good)");
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, 230 * MiB, 0, 1024) == 306 * MiB,
              "a request seen at an unknown n_ubatch is not scaled");
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, 0, 512, 1024) == 76 * MiB,
              "nothing observed yet: the bound is the plan alone, and the comments say it is a lower bound");
        CHECK(ggml_sycl::zone_hold_spill_bound(0, 230 * MiB, 512, 1024) == 0,
              "no plan, no hold, nothing a hold can push out");
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, SIZE_MAX, 1, 2) == SIZE_MAX,
              "a scaled request that overflows saturates");
        CHECK(ggml_sycl::zone_hold_spill_bound(SIZE_MAX - 1, 230 * MiB, 512, 512) == SIZE_MAX,
              "a plan plus a request that overflows saturates");
        CHECK(ggml_sycl::zone_hold_spill_bound(76 * MiB, 230 * MiB, 512, 0) == 306 * MiB,
              "an unknown candidate n_ubatch is not scaled");
    }

    // ---- Case 18: only the part of the worst-case spill that the arena's KV zone cannot take is OUTSIDE-arena
    // demand (kpjw r3 design change). A compute buffer the RUNTIME zone will not serve is placed in the KV zone
    // first; raw device memory is the last resort. -------------------------------------------------------------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(536 * MiB, 600 * MiB) == 0,
              "a bound the KV zone can hold entirely is no outside-arena demand");
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(536 * MiB, 300 * MiB) == 236 * MiB,
              "the KV zone takes what it can, the rest is raw");
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(536 * MiB, 0) == 536 * MiB,
              "a full KV zone leaves the whole bound outside the arena");
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(536 * MiB, 536 * MiB) == 0,
              "an exact fit is not a spill");
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(0, 100 * MiB) == 0, "no bound, no demand");
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(SIZE_MAX, 1) == SIZE_MAX - 1,
              "a saturated bound minus the KV room stays huge, never wraps small");
    }

    // ---- Case 19: which RUNTIME-zone requests go to the KV zone before raw memory. Only a compute-buffer
    // request (the caller's flag), spill-capable, that the RUNTIME zone will not serve, and only when the KV zone
    // can hold it whole. Every other request class keeps today's path. -------------------------------------------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_runtime_spill_prefers_kv_zone(true, true, false, true, 600 * MiB, 460 * MiB),
              "a held-back or zone-full compute buffer that the KV zone can hold goes there");
        CHECK(!ggml_sycl::zone_runtime_spill_prefers_kv_zone(true, true, false, true, 400 * MiB, 460 * MiB),
              "a KV zone too small for it is no placement: the raw path decides");
        CHECK(!ggml_sycl::zone_runtime_spill_prefers_kv_zone(true, true, false, false, 600 * MiB, 460 * MiB),
              "a request the RUNTIME zone serves stays in the RUNTIME zone");
        CHECK(!ggml_sycl::zone_runtime_spill_prefers_kv_zone(false, true, false, true, 600 * MiB, 460 * MiB),
              "a request that is not a compute buffer keeps the pre-existing spill path");
        CHECK(!ggml_sycl::zone_runtime_spill_prefers_kv_zone(true, false, false, true, 600 * MiB, 460 * MiB),
              "only the RUNTIME zone's misses are redirected");
        CHECK(!ggml_sycl::zone_runtime_spill_prefers_kv_zone(true, true, true, true, 600 * MiB, 460 * MiB),
              "a forbid-spill claimant is refused, never placed elsewhere");
        CHECK(ggml_sycl::zone_runtime_spill_prefers_kv_zone(true, true, false, true, 460 * MiB, 460 * MiB),
              "an exact fit in the KV zone is a fit");
    }

    // ---- Case 20: the KV room a compute buffer can count on (review r4 I1). The runtime transaction publishes
    // BEFORE the context's KV exists, so the zone still shows free the bytes this context's own KV is about to
    // take; and a buffer is indivisible, so only the largest free block counts. -----------------------------------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_kv_room_for_compute(600 * MiB, 0) == 600 * MiB,
              "KV already live (the recheck, a settle): the whole largest block is room");
        CHECK(ggml_sycl::zone_kv_room_for_compute(600 * MiB, 400 * MiB) == 200 * MiB,
              "the KV this transaction will place is not room for a compute buffer");
        CHECK(ggml_sycl::zone_kv_room_for_compute(300 * MiB, 400 * MiB) == 0,
              "a KV that takes more than the largest block leaves nothing, never a wrapped huge figure");
        CHECK(ggml_sycl::zone_kv_room_for_compute(0, 0) == 0, "an empty zone is no room");
        CHECK(ggml_sycl::zone_kv_room_for_compute(SIZE_MAX, SIZE_MAX) == 0, "an exact consumption is no room");
        // The F3 estimate with the netting: B50 shape, a 536 MB worst-case spill, KV zone showing 900 MB of which
        // this context's KV will take 600 MB.
        CHECK(ggml_sycl::zone_hold_spill_raw_demand(536 * MiB, ggml_sycl::zone_kv_room_for_compute(900 * MiB, 600 * MiB)) ==
                  236 * MiB,
              "un-netted the same bound reads as 0 raw demand and F3 checks nothing");
    }

    // ---- Case 24 (kpjw-g7 unification, P4: one fact, one source): ONE fit function answers the transaction-time
    // bound (F3), the ladder's realized check and the pinned -ub check, over the same inputs: the plan at the rung,
    // the rung's recorded worst-case request, the KV room net of what is pending, and the card's free memory with
    // the rung's own buffers released. The B50 / Qwen3.6-27B case of g7 (-c 512): plan 75.6 MB, request 495.0 MB at
    // 512 (990.0 MB at 1024), 100.6 MB of KV-zone room, 1097 MB free once the rung's buffers are gone. -------------
    {
        const size_t MiB  = 1024 * 1024;
        const size_t head = 256 * MiB;
        auto         plan = [](void *, uint32_t) -> size_t { return 76 * 1024 * 1024; };
        ggml_sycl::zone_hold_rung_request rungs[] = { { 512, 495 * MiB }, { 1024, 990 * MiB } };
        ggml_sycl::zone_hold_fit_inputs   in      = {};
        in.headroom_target                        = head;
        in.free_before                            = 1097 * MiB;
        in.kv_room                                = 100 * MiB;
        in.rungs                                  = rungs;
        in.n_rungs                                = 2;
        in.plan_of                                = plan;
        CHECK(ggml_sycl::zone_hold_fit(in, 512), "512 fits the B50 once its own buffers are released (pinned 512 runs)");
        CHECK(!ggml_sycl::zone_hold_fit(in, 1024), "1024 does not: 966 MB of demand leaves 131 MB of a 1097 MB card");
        const uint32_t named = ggml_sycl::zone_hold_fit_largest_ub(in, 1024);
        CHECK(named == 512, "the -ub a refusal at 1024 prints is 512, the same function's largest accepted rung");
        CHECK(ggml_sycl::zone_hold_fit(in, named), "the printed N passes F3 under the same inputs");
        CHECK(!ggml_sycl::zone_hold_fit(in, named * 2), "and N*2 fails");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 512) == 512, "a rung that fits is returned unchanged");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 0) == 0, "an unknown n_ubatch names nothing");
        // The demand the function judges is the plan plus the rung's request net of the KV room, whatever consumer asks.
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 1024) == 76 * MiB + 990 * MiB - 100 * MiB,
              "demand = plan + the rung's own recorded request - KV room");
        // The refusal is the hold's doing only when the card was above the headroom without it.
        in.free_before = 200 * MiB;
        CHECK(ggml_sycl::zone_hold_fit(in, 1024), "short without the demand too: not the hold's doing");
        // No rung fits: 0, never a made-up -ub.
        in.free_before = 1097 * MiB;
        in.kv_room     = 0;
        ggml_sycl::zone_hold_rung_request big[] = { { 1024, 1200 * MiB } };
        in.rungs                                = big;
        in.n_rungs                              = 1;
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 1024) != 1024, "a rung whose own request cannot fit is not named");
    }

    // ---- Case 25 (kpjw-g7 B, order independence): the verdict for a rung is a function of the configuration and
    // THAT rung's measured request, not of which other rungs ran before it or in what order. A rung with its own
    // record uses it and only it; a rung without one is scaled from the SET of records (order never matters). -----
    {
        const size_t MiB  = 1024 * 1024;
        auto         plan = [](void *, uint32_t) -> size_t { return 76 * 1024 * 1024; };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 1097 * MiB;
        in.kv_room                         = 100 * MiB;
        in.plan_of                         = plan;
        ggml_sycl::zone_hold_rung_request own[]   = { { 512, 495 * MiB } };
        ggml_sycl::zone_hold_rung_request other[] = { { 512, 495 * MiB }, { 1024, 990 * MiB }, { 2048, 4000 * MiB } };
        ggml_sycl::zone_hold_rung_request rev[]   = { { 2048, 4000 * MiB }, { 1024, 990 * MiB }, { 512, 495 * MiB } };
        in.rungs = own;
        in.n_rungs = 1;
        const bool first = ggml_sycl::zone_hold_fit(in, 512);
        const size_t d1  = ggml_sycl::zone_hold_fit_demand(in, 512);
        in.rungs = other;
        in.n_rungs = 3;
        CHECK(ggml_sycl::zone_hold_fit(in, 512) == first && ggml_sycl::zone_hold_fit_demand(in, 512) == d1,
              "the same rung twice, with other rungs' history in between, gives the same demand and verdict");
        in.rungs = rev;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 512) == d1, "the order the records were made in does not matter");
        // A rung nobody measured is scaled from the set, the same way whatever order the set is in.
        ggml_sycl::zone_hold_rung_request sparse_a[] = { { 512, 495 * MiB }, { 2048, 1980 * MiB } };
        ggml_sycl::zone_hold_rung_request sparse_b[] = { { 2048, 1980 * MiB }, { 512, 495 * MiB } };
        in.rungs = sparse_a;
        in.n_rungs = 2;
        const size_t scaled_a = ggml_sycl::zone_hold_fit_demand(in, 1024);
        in.rungs = sparse_b;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 1024) == scaled_a, "an unmeasured rung is scaled order-independently");
        in.n_rungs = 0;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 1024) == 0, "with no record the demand is the plan alone, less the room");
        in.kv_room = 0;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 1024) == 76 * MiB, "and the plan alone when there is no room (a lower bound)");
    }

    // ---- Case 26 (kpjw-g7 C, one model of free memory): the card's free memory is the cache's own ledger, never a
    // driver read taken after a release (credit lags: 602.7 MB read where 1097 MB was true). cold = the driver's
    // reading plus the outside-arena bytes the cache holds live; free_before = cold less what stays live without the
    // rung. Releasing the rung's buffers and evaluating gives the answer evaluating before releasing gives, with the
    // released bytes credited. -----------------------------------------------------------------------------------
    {
        const size_t MiB        = 1024 * 1024;
        const size_t persistent = 461 * MiB;  // the recurrent-state buffer: live before the ladder, and after it
        const size_t own        = 495 * MiB;  // the rung's raw compute landing
        const size_t driver_true = 1097 * MiB;
        // Evaluated live: the driver sees the rung's buffer, the ledger holds it.
        const size_t cold_live   = ggml_sycl::zone_hold_free_cold(driver_true - own, persistent + own);
        const size_t before_live = ggml_sycl::zone_hold_free_before(cold_live, persistent);
        CHECK(before_live == driver_true, "evaluated live, the rung's own bytes are credited back: the true free memory");
        // The registered-then-released raw row (the real ledger, not this arithmetic) is test-sycl-hold-ledger.
        CHECK(ggml_sycl::zone_hold_free_before(100 * MiB, 300 * MiB) == 0, "never below zero");
        CHECK(ggml_sycl::zone_hold_free_cold(SIZE_MAX, 1) == SIZE_MAX, "saturating");
    }

    // ---- Case 27 (kpjw-r7 I3, identity by origin): what stays live without the rung is every raw byte the cache
    // holds except the rung's OWN scheduler compute buffers, and only when the rung's buffers are live. A raw byte
    // allocated after the epoch began that is not a scheduler compute buffer (the recurrent state, made after a
    // pinned -ub's one publish) is persistent, however late it came. -----------------------------------------------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_hold_persistent_raw(956 * MiB, 495 * MiB, true) == 461 * MiB,
              "a live rung: the held raw bytes less its own compute buffers (the 461 MB state stays)");
        CHECK(ggml_sycl::zone_hold_persistent_raw(956 * MiB, 495 * MiB, false) == 956 * MiB,
              "a transaction (no live rung): every held raw byte is persistent");
        CHECK(ggml_sycl::zone_hold_persistent_raw(461 * MiB, 0, true) == 461 * MiB,
              "no scheduler compute rows live: nothing is credited back");
        CHECK(ggml_sycl::zone_hold_persistent_raw(100 * MiB, 300 * MiB, true) == 0,
              "compute rows can never credit more than is held");
    }

    // ---- Case 28 (kpjw-r7 I2, the baseline): within a window the cold reading is the MAXIMUM of the readings taken
    // (the driver's lag only ever lowers a reading, never raises it, so a later, healthier read repairs a stale-low
    // first one and a later, lagged read never lowers it); the first reading of a window is taken as it comes. ------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_hold_cold_update(false, 0, 600 * MiB) == 600 * MiB,
              "the first reading of a window stands");
        CHECK(ggml_sycl::zone_hold_cold_update(true, 1097 * MiB, 602 * MiB) == 1097 * MiB,
              "a later lagged (stale-low) reading does not lower the baseline");
        CHECK(ggml_sycl::zone_hold_cold_update(true, 602 * MiB, 1097 * MiB) == 1097 * MiB,
              "a later healthier reading repairs a stale-low first one");
        CHECK(ggml_sycl::zone_hold_cold_update(false, 5000 * MiB, 100 * MiB) == 100 * MiB,
              "a new window forgets the previous one (another tenant may have arrived since)");
    }

    // ---- Case 29 (kpjw-r7 I3, the KV room): a rung that is live is judged with the room its epoch began with; any
    // other asker (a transaction) reads the zone as it is. An epoch whose room was never stored does not stand in. --
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_hold_pick_kv_room(true, true, 100 * MiB, 40 * MiB) == 100 * MiB,
              "live rung: the epoch's room");
        CHECK(ggml_sycl::zone_hold_pick_kv_room(true, false, 100 * MiB, 40 * MiB) == 40 * MiB,
              "no epoch yet: the zone now");
        CHECK(ggml_sycl::zone_hold_pick_kv_room(false, true, 100 * MiB, 40 * MiB) == 40 * MiB,
              "a transaction: the zone now");
        CHECK(ggml_sycl::zone_hold_pick_kv_room(true, true, 0, 40 * MiB) == 0,
              "an epoch that stored no room is honoured as stored");
    }

    // ---- Case 30 (kpjw-r7 I1): the non-FA demand is the non-FA scratch PLUS the hold's worst-case spill, saturating
    // -- the hold term is never dropped. --------------------------------------------------------------------------
    {
        const size_t MiB = 1024 * 1024;
        CHECK(ggml_sycl::zone_hold_nonfa_demand(300 * MiB, 200 * MiB) == 500 * MiB, "scratch + hold spill");
        CHECK(ggml_sycl::zone_hold_nonfa_demand(300 * MiB, 0) == 300 * MiB, "no hold: the scratch alone");
        CHECK(ggml_sycl::zone_hold_nonfa_demand(SIZE_MAX, 1) == SIZE_MAX,
              "a wrapped sum must not read as a small demand");
        CHECK(ggml_sycl::zone_hold_nonfa_demand(1, SIZE_MAX) == SIZE_MAX, "saturating on either term");
    }

    // ---- Case 31 (kpjw-r7 M6/M8): the -ub a refusal names. N passes and N*2 fails only while N*2 is a rung that
    // was ever asked (not above n_ubatch); a non-power-of-two n_ubatch names the power of two under it; an N under the
    // descent floor (64) is still returned (it is the largest the function accepts, not a rung the descent walks). ---
    {
        const size_t MiB  = 1024 * 1024;
        auto         plan = [](void *, uint32_t) -> size_t {
            return 10 * 1024 * 1024;
        };
        ggml_sycl::zone_hold_rung_request rungs[] = {
            { 600, 600 * MiB }
        };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 700 * MiB;
        in.kv_room                         = 0;
        in.rungs                           = rungs;
        in.n_rungs                         = 1;
        in.plan_of                         = plan;
        // Plan 10 MB. 600 (600 MB) leaves 90 MB < 256 MB: refused; 512 scales to 512 MB, leaving 178 MB: refused; 256
        // scales to 256 MB, leaving 434 MB: fits.
        CHECK(!ggml_sycl::zone_hold_fit(in, 600), "600 does not fit");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 600) == 256,
              "a non-power-of-two n_ubatch names the power of two under it");
        in.free_before = 300 * MiB;
        // Only tiny rungs fit: 64 leaves 226 MB, refused; 32 scales to 32 MB, leaving 258 MB >= 256 MB.
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 600) == 32, "an N under the descent floor is still named");
        ggml_sycl::zone_hold_rung_request huge[] = {
            { 600, 6000 * MiB }
        };
        in.rungs = huge;
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 600) == 0, "when no rung fits, nothing is named");
        // Below the 32 floor nothing is searched: a refused n_ubatch under 32 names 0, never a rung ABOVE it. The plan
        // here is deliberately larger for the small n_ubatch, so a search that started at 32 would find a fit and name
        // 32 for a request of 31.
        auto plan_small_costs_more = [](void *, uint32_t n_ubatch) -> size_t {
            return n_ubatch < 32 ? 400 * 1024 * 1024 : 1024 * 1024;
        };
        ggml_sycl::zone_hold_rung_request tiny[] = {
            { 32, 1 * MiB }
        };
        in.rungs   = tiny;
        in.plan_of = plan_small_costs_more;
        CHECK(!ggml_sycl::zone_hold_fit(in, 31), "31 does not fit under the 400 MB plan");
        CHECK(ggml_sycl::zone_hold_fit(in, 32), "32 fits under the 1 MB plan");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 31) == 0, "a refused n_ubatch under 32 names nothing above it");
    }

    // ---- Case 32 (kpjw-r7 I5, N4/N5): the compute-allocation scope is a per-thread depth that cannot go negative. A
    // leave without an enter (a guard destroyed after an exception unwound past its enter) leaves the scope closed,
    // and one thread's open scope is not another's. ---------------------------------------------------------------
    {
        CHECK(!ggml_sycl::compute_alloc_scope_active(), "closed to start with");
        ggml_sycl::compute_alloc_scope_leave();
        ggml_sycl::compute_alloc_scope_enter();
        CHECK(ggml_sycl::compute_alloc_scope_active(),
              "a stray leave does not push the depth negative: the next enter opens it");
        bool        other_thread_active = true;
        std::thread t([&] { other_thread_active = ggml_sycl::compute_alloc_scope_active(); });
        t.join();
        CHECK(!other_thread_active, "another thread does not see this thread's open scope");
        ggml_sycl::compute_alloc_scope_enter();
        ggml_sycl::compute_alloc_scope_leave();
        CHECK(ggml_sycl::compute_alloc_scope_active(), "nested: still open after the inner leave");
        ggml_sycl::compute_alloc_scope_leave();
        CHECK(!ggml_sycl::compute_alloc_scope_active(), "closed after the matching leaves");
    }

    std::printf("PASS: zone-sizing structural path-scoped maxima\n");
    return 0;
}
