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

#include "zone-sizing.hpp"

#include <cstdio>
#include <string>
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

    // ---- Case 14: oneDNN PP scratch admission (llama.cpp-8ony) ---------------
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

    std::printf("PASS: zone-sizing structural path-scoped maxima\n");
    return 0;
}
