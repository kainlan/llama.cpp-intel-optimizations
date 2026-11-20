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
#include "tlsf-allocator.hpp"
#include "unified-types.hpp"
#include "zone-raw-gate.hpp"
#include "zone-sizing.hpp"
#include "zone-used-count.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
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
        // must not contribute (zone_moe_mmvq_op_bytes sizes its per-op buffers).
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

        // No planned pair (0, 0): the request alone is held against the bound (Case 14n has the planned pair).
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, zone_256mib, 0, 0, attn_w, attn_a),
              "a per-layer attention weight fits the 256 MiB ONEDNN zone");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, zone_256mib, 0, 0, head_w, head_a),
              "the LM head does not fit the ONEDNN zone, so it is not planned there");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(false, 0, 0, 0, head_w, head_a),
              "with no arena there is no zone to disagree with: the unified-cache path serves it");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, 100, 0, 0, 60, 40),
              "a pair that exactly fills the zone fits");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, 100, 0, 0, 60, 41),
              "one byte over the zone does not fit");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, 0, 0, 0, 1, 1), "an empty zone holds nothing");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, SIZE_MAX, 0, 0, SIZE_MAX, 2),
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

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 0, 0, 0, 0, 24 * mib, 3 * mib, &w, &a);
        CHECK(w == 24 * mib && a == 3 * mib, "nothing held: the request is the target");

        ggml_sycl::zone_onednn_scratch_reserve_target(false, 0, 1105 * mib, 1 * mib, 0, 0, 24 * mib, 3 * mib, &w, &a);
        CHECK(w == 1105 * mib, "a smaller weights request never shrinks the held weights block");
        CHECK(a == 3 * mib, "a larger activations request still grows the activations half on its own");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 100 * mib, 2 * mib, 0, 0, 50 * mib, 3 * mib, &w,
                                                      &a);
        CHECK(w == 100 * mib && a == 3 * mib, "inside the zone the merged pair is the target");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 200 * mib, 2 * mib, 0, 0, 50 * mib, 100 * mib,
                                                      &w, &a);
        CHECK(w == 50 * mib && a == 100 * mib,
              "a merge that would overflow the zone falls back to the request instead of wedging every op");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, SIZE_MAX, 1, 0, 0, 50 * mib, 3 * mib, &w, &a);
        CHECK(w == 50 * mib && a == 3 * mib, "an unrepresentable merged sum is refused, not wrapped");

        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 1, 1, 0, 0, 2, 2, nullptr, nullptr);
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

        // The planned pair is the attention op's own (Case 14n covers an op above the planned pair in one half).
        CHECK(ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, zone_256mib, attn_w, attn_a, attn_w, attn_a),
              "an admitted, enabled, planned op is supplied by the scratch");
        CHECK(
            !ggml_sycl::zone_onednn_pp_scratch_supplies(true, false, true, zone_256mib, attn_w, attn_a, attn_w, attn_a),
            "an op whose type is not enabled is NOT supplied, so the walk must size its planned dequant buffers");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(true, false, false, 0, 0, 0, attn_w, attn_a),
              "type refusal holds with no arena too");
        CHECK(
            !ggml_sycl::zone_onednn_pp_scratch_supplies(false, true, true, zone_256mib, attn_w, attn_a, attn_w, attn_a),
            "an op that fails the PP admission is not supplied");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, zone_256mib, attn_w, attn_a, 1158266880,
                                                          1474560),
              "an enabled op whose pair is over the zone is not supplied");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, 100, 60, 40, 60, 40),
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
            CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, 1 * mib, 1 * mib, 18 * mib, 1 * mib),
                  "small model: a 19 MB head pair fits the slack, so it stays supplied by the scratch");
        }
        // (b) the Mistral head window: plan 143.5 MB, floor ~33 MB, capacity 256 MiB, head pair 266 MB.
        {
            const size_t plan  = 150470656;  // 143.5 MiB
            const size_t floor = 34603008;   // 33 MiB
            const size_t bound = ggml_sycl::zone_onednn_pp_pair_bound(256 * mib, plan, floor);
            CHECK(bound == 256 * mib - floor, "Mistral: the bound is capacity minus the floor");
            const size_t plan_w = 117440512;      // the planned weights half (14336 x 4096 x f16)
            const size_t plan_a = plan - plan_w;  // the planned activations half
            CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, plan_w, plan_a, 262144000, 4194304),
                  "Mistral: a 266 MB head pair would eat the Graph SDPA floor, so it is not supplied");
            CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, plan_w, plan_a, 117440512, 4194304),
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

    // ---- Case 14j: a row-gather tensor is not a MUL_MAT operand (llama.cpp-8ony) ---------------------------------
    // token_embd.weight is consumed by GET_ROWS, which draws neither the dequant buffers nor the Q8_1 src1 buffer.
    // Planned as if it were the head it cost a Q4_0 Mistral a 254 MB RUNTIME zone for nothing. The loader's role for
    // the tensor decides: a tied embedding that also serves as the output head IS a MUL_MAT operand.
    {
        const size_t head_w  = 262144000;  // 4096 x 32000 f16
        auto         marked  = [&](const char * name, int type, size_t f16_w, bool gather_only) {
            zone_tensor_desc d                               = desc(name, 1000, type, 4096, 32000, 1, 1);
            d.dequant_f16_if_unsupplied_weight_bytes         = f16_w;
            d.dequant_f16_if_unsupplied_src1_bytes_per_token = 4096 * F16_BYTES;
            d.pp_scratch_type_enabled                        = true;
            d.mmq_src1_bytes_per_token                       = 4608;
            d.get_rows_only                                  = gather_only;
            return d;
        };
        std::vector<zone_tensor_desc> layers;
        for (int i = 0; i < 4; i++) {
            layers.push_back(desc("blk.0.ffn_gate.weight", 1000, TYPE_Q4_0, 4096, 14336, 1, 1));
        }

        // (a) Q4_0 token_embd beside a separate Q6_K head: the embedding is gather-only, the head is not a candidate.
        std::vector<zone_tensor_desc> separate_head = layers;
        separate_head.push_back(marked("token_embd.weight", TYPE_Q4_0, head_w, true));
        separate_head.push_back(desc("output.weight", 1000, TYPE_Q6_K, 4096, 32000, 1, 1));
        const path_scoped_maxima ma = zone_scoped_maxima(separate_head);
        CHECK(ma.dequant_f16_weight_bytes == 0 && ma.dequant_f16_src1_bytes_per_token == 0,
              "(a) a gather-only Q4_0 embedding is not planned into the dequant buffers");
        CHECK(ma.mmq_src1_bytes_per_token == 0, "(a) nor into the Q8_1 src1 buffer");

        // (b) tied: the one token_embd tensor is also the head, so the loader does not call it gather-only.
        std::vector<zone_tensor_desc> tied = layers;
        tied.push_back(marked("token_embd.weight", TYPE_Q4_0, head_w, false));
        const path_scoped_maxima mb = zone_scoped_maxima(tied);
        CHECK(mb.dequant_f16_weight_bytes == head_w && mb.dequant_f16_src1_bytes_per_token == 4096 * F16_BYTES,
              "(b) a tied embedding used as the head is planned");
        CHECK(mb.mmq_src1_bytes_per_token == 4608, "(b) and draws the Q8_1 src1 buffer");

        // (c) a Q4_0 head beside a gather-only Q4_0 embedding: the head alone is the plan.
        std::vector<zone_tensor_desc> q4_head = layers;
        q4_head.push_back(marked("token_embd.weight", TYPE_Q4_0, head_w, true));
        q4_head.push_back(marked("output.weight", TYPE_Q4_0, head_w, false));
        const path_scoped_maxima mc = zone_scoped_maxima(q4_head);
        CHECK(mc.dequant_f16_weight_bytes == head_w, "(c) a Q4_0 head is planned");

        // The head's own mark is what carries it: with the head removed, nothing is left to plan.
        std::vector<zone_tensor_desc> embd_only = layers;
        embd_only.push_back(marked("token_embd.weight", TYPE_Q4_0, head_w, true));
        CHECK(zone_scoped_maxima(embd_only).dequant_f16_weight_bytes == 0,
              "the gather-only embedding alone leaves the plan empty");

        // The unconditional Q8_0 marks are held to the same rule.
        zone_tensor_desc q8_embd                 = desc("token_embd.weight", 1000, TYPE_Q8_0, 2880, 201088, 1, 1);
        q8_embd.dequant_f16_weight_bytes         = 1159372800;
        q8_embd.dequant_f16_src1_bytes_per_token = 2880 * F16_BYTES;
        q8_embd.get_rows_only                    = true;
        zone_tensor_desc q8_head                 = q8_embd;
        q8_head.get_rows_only                    = false;
        CHECK(zone_scoped_maxima({ q8_embd }).dequant_f16_weight_bytes == 0,
              "a gather-only Q8_0 embedding is not planned");
        CHECK(zone_scoped_maxima({ q8_embd, q8_head }).dequant_f16_weight_bytes == 1159372800,
              "a Q8_0 head beside it still is");
    }

    // ---- Case 14k: a quantized type no kernel but the f16 dequant arm serves takes it as its PP route (llama.cpp-gldu) ----
    // The router walks its priority list: a type an MMQ kernel serves is taken by MMQ at PP batch unless the MMQ claim
    // moves it to the arm (Case 14l), a type with a coalesced layout by its coalesced / unified kernel, and what is
    // left (the IQ family) has only the oneDNN dequant arm. The planner asks the SAME two type lists the router's
    // eligibility terms are built from
    // (unified-types.hpp), so it claims exactly that remainder; claiming an MMQ type sized the whole plan from the
    // 1.2 GB f16 copy of a Q5_K LM head on a device run.
    {
        CHECK(ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_IQ4_XS), "IQ4_XS draws the dequant arm");
        CHECK(ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_IQ3_S), "IQ3_S draws the dequant arm");
        CHECK(ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_IQ4_NL), "IQ4_NL draws the dequant arm");
        CHECK(ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_IQ3_XXS), "IQ3_XXS draws the dequant arm");
        CHECK(ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_IQ2_XS), "IQ2_XS draws the dequant arm");
        const ggml_type mmq_types[] = { GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q2_K,
                                        GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K };
        for (ggml_type ty : mmq_types) {
            CHECK(ggml_sycl::mmq_capable_type(ty), "the MMQ list carries the type the router's use_mmq asks about");
            CHECK(!ggml_sycl::dense_pp_route_is_f16_dequant_arm(ty),
                  "an MMQ type is not the AOS claim's (the MMQ claim of Case 14l covers the measured ones)");
        }
        CHECK(!ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_Q8_0), "Q8_0 keeps its own unconditional mark");
        CHECK(!ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_Q4_0), "Q4_0 keeps the unified-kernel mark");
        CHECK(!ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_MXFP4),
              "MXFP4 is coalesced-capable (and keeps the unified-kernel mark)");
        CHECK(!ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_Q6_K),
              "Q6_K is materialized COALESCED and routed to MMQ_COALESCED");
        // The composition the adapter calls: type lists, dense-MUL_MAT support, dequant support, head and expert.
        auto claims = [](ggml_type ty, bool head, bool expert, bool dequant) {
            return ggml_sycl::aos_dequant_f16_plan_claims(ty, true, head, expert, dequant);
        };
        CHECK(claims(GGML_TYPE_IQ3_S, false, false, true), "a dense IQ3_S weight is claimed");
        CHECK(claims(GGML_TYPE_IQ4_NL, false, false, true), "a dense IQ4_NL weight is claimed");
        CHECK(!claims(GGML_TYPE_IQ4_XS, true, false, true), "the LM head is not claimed");
        CHECK(!claims(GGML_TYPE_IQ3_S, false, true, true), "an expert stack is not claimed");
        CHECK(!claims(GGML_TYPE_IQ3_S, false, false, false), "a type the router cannot dequantize is not claimed");
        CHECK(!ggml_sycl::aos_dequant_f16_plan_claims(GGML_TYPE_IQ3_S, false, false, false, true),
              "a non-quantized operand is not claimed");
        CHECK(!claims(GGML_TYPE_Q5_K, false, false, true) && !claims(GGML_TYPE_Q4_K, false, false, true) &&
                  !claims(GGML_TYPE_Q2_K, false, false, true) && !claims(GGML_TYPE_Q4_1, false, false, true) &&
                  !claims(GGML_TYPE_Q5_0, false, false, true),
              "an MMQ type is not claimed");
        CHECK(!claims(GGML_TYPE_Q8_0, false, false, true) && !claims(GGML_TYPE_Q4_0, false, false, true) &&
                  !claims(GGML_TYPE_MXFP4, false, false, true) && !claims(GGML_TYPE_Q6_K, false, false, true),
              "a coalesced-capable type is not claimed");
        // SYCL refuses a dense MUL_MAT for these although the router can dequantize them: no copy is reserved.
        CHECK(!claims(GGML_TYPE_NVFP4, false, false, true), "NVFP4 is refused by SYCL's dense MUL_MAT, so not claimed");
        CHECK(!claims(GGML_TYPE_Q1_0, false, false, true), "Q1_0 is refused by SYCL's dense MUL_MAT, so not claimed");
        // Q2_0 left that group when it gained its SYCL kernels (llama.cpp-s36q phase 4): dense Q2_0 is executed, is
        // neither MMQ-served nor coalesced, and has a to_fp16 converter, so its PP route is the f16 dequant arm.
        CHECK(ggml_sycl::dense_pp_route_is_f16_dequant_arm(GGML_TYPE_Q2_0), "Q2_0 draws the dequant arm");
        CHECK(claims(GGML_TYPE_Q2_0, false, false, true), "a dense Q2_0 weight is claimed");
        CHECK(!claims(GGML_TYPE_Q2_0, false, true, true), "a Q2_0 expert stack is not claimed");
        CHECK(ggml_sycl::dense_mul_mat_type_supported(GGML_TYPE_IQ4_XS) &&
                  !ggml_sycl::dense_mul_mat_type_supported(GGML_TYPE_NVFP4),
              "the dense MUL_MAT list carries the IQ family and not NVFP4");
        // The set is disjoint from the unified kernel's, so the two marks never double-count one weight.
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const ggml_type ty = static_cast<ggml_type>(t);
            CHECK(!(ggml_sycl::dense_pp_route_is_f16_dequant_arm(ty) && ggml_sycl::unified_kernel_serves_type(ty)),
                  "no type is both a unified-kernel type and an AOS dequant-arm type");
        }

        // The adapter marks such a weight through the if-unsupplied fields (scratch off for its type by default),
        // so it sizes the dequant plan; a gather-only IQ4_NL (per_layer_token_embd, 26.8 GiB) never does.
        const size_t layer_w = 50331648;  // 4096 x 6144 f16
        auto         marked  = [&](const char * name, int type, int64_t ne0, int64_t ne1, size_t f16_w, bool gather) {
            zone_tensor_desc d                               = desc(name, 1000, type, ne0, ne1, 1, 1);
            d.dequant_f16_if_unsupplied_weight_bytes         = f16_w;
            d.dequant_f16_if_unsupplied_src1_bytes_per_token = static_cast<size_t>(ne0) * F16_BYTES;
            d.pp_scratch_type_enabled                        = false;
            d.get_rows_only                                  = gather;
            return d;
        };
        std::vector<zone_tensor_desc> iq;
        for (int i = 0; i < 4; i++) {
            iq.push_back(marked("blk.0.ffn_gate_shexp.weight", GGML_TYPE_IQ4_XS, 4096, 6144, layer_w, false));
        }
        CHECK(zone_scoped_maxima(iq).dequant_f16_weight_bytes == layer_w,
              "a dense IQ4_XS weight sizes the dequant plan");
        const size_t                  per_layer_embd_w = 160ull * 320001536ull * F16_BYTES;
        std::vector<zone_tensor_desc> with_embd        = iq;
        with_embd.push_back(
            marked("per_layer_token_embd.weight", GGML_TYPE_IQ4_NL, 160, 320001536, per_layer_embd_w, true));
        CHECK(zone_scoped_maxima(with_embd).dequant_f16_weight_bytes == layer_w,
              "a gather-only IQ4_NL table never sizes the dequant plan");
    }

    // ---- Case 14l: the MMQ types the oneDNN f16 dequant arm serves at PP batch (llama.cpp-h9c8) ------------------
    // At PP batch the MMQ kernel ran these types at ~3.8 TOPS where the oneDNN f16 GEMM runs the same shapes at
    // ~133 TFLOPS, so they take the arm. The planner and the router ask ONE list and ONE claim (unified-types.hpp),
    // so the plan reserves what the arm draws. The list is explicit rather than "MMQ minus Q4_0 / Q8_0": a type added
    // to the MMQ list later does not join the arm before it is measured.
    {
        auto prefers = [](ggml_type ty) {
            return ggml_sycl::dense_pp_route_prefers_f16_dequant_arm(ty);
        };
        const ggml_type listed[] = { GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q2_K,
                                     GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
        for (ggml_type ty : listed) {
            CHECK(prefers(ty), "a measured MMQ type is listed");
        }
        int n_listed = 0;
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const ggml_type ty = static_cast<ggml_type>(t);
            if (!prefers(ty)) {
                continue;
            }
            n_listed++;
            CHECK(ggml_sycl::mmq_capable_type(ty), "a listed type is one an MMQ kernel serves");
            CHECK(ggml_sycl::dense_mul_mat_type_supported(ty), "a listed type is a dense MUL_MAT SYCL executes");
            CHECK(!ggml_sycl::dense_pp_route_is_f16_dequant_arm(ty), "the list is disjoint from the AOS claim's types");
            CHECK(!ggml_sycl::unified_kernel_serves_type(ty), "the list is disjoint from the unified kernel's types");
        }
        CHECK(n_listed == 8, "exactly the eight measured types are listed");
        CHECK(!prefers(GGML_TYPE_Q4_0) && !prefers(GGML_TYPE_Q8_0) && !prefers(GGML_TYPE_MXFP4),
              "Q4_0, Q8_0 and MXFP4 keep their own oneDNN routes and plan marks");
        CHECK(!prefers(GGML_TYPE_IQ4_XS) && !prefers(GGML_TYPE_IQ2_XXS) && !prefers(GGML_TYPE_F16) &&
                  !prefers(GGML_TYPE_F32),
              "the IQ family (the AOS claim's) and the float types are not listed");

        // The composition the adapter and the router helper call.
        auto claims = [](ggml_type ty, bool head, bool expert, bool matrix, bool dequant) {
            return ggml_sycl::mmq_dequant_f16_plan_claims(ty, true, head, expert, matrix, dequant);
        };
        for (ggml_type ty : listed) {
            CHECK(claims(ty, false, false, true, true), "a dense 2-D weight of a listed type is claimed");
        }
        CHECK(!claims(GGML_TYPE_Q6_K, true, false, true, true), "the LM head stays on MMQ and is not claimed");
        CHECK(!claims(GGML_TYPE_Q4_K, false, true, true, true), "an expert stack is not claimed");
        CHECK(!claims(GGML_TYPE_Q4_K, false, false, false, true), "a 3-D weight stays on MMQ and is not claimed");
        CHECK(!claims(GGML_TYPE_Q5_K, false, false, true, false), "a type the router cannot dequantize is not claimed");
        CHECK(!ggml_sycl::mmq_dequant_f16_plan_claims(GGML_TYPE_Q4_K, false, false, false, true, true),
              "a non-quantized operand is not claimed");
        CHECK(!claims(GGML_TYPE_Q4_0, false, false, true, true) && !claims(GGML_TYPE_Q8_0, false, false, true, true) &&
                  !claims(GGML_TYPE_MXFP4, false, false, true, true) &&
                  !claims(GGML_TYPE_IQ4_XS, false, false, true, true),
              "a type the list leaves out is not claimed");
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const ggml_type ty = static_cast<ggml_type>(t);
            CHECK(!(claims(ty, false, false, true, true) &&
                    ggml_sycl::aos_dequant_f16_plan_claims(ty, true, false, false, true)),
                  "no weight is claimed by both the AOS claim and the MMQ claim");
        }

        // Mistral 7B Q4_K_M as the adapter marks it: the claim decides, the adapter's own arithmetic sizes the marks,
        // and by default the scratch is not enabled for these types, so every mark is planned. Two layers suffice: the
        // plan is a maximum over the marked weights, not a sum.
        auto marked = [&](const char * name, ggml_type type, int64_t ne0, int64_t ne1, bool head) {
            zone_tensor_desc d = desc(name, 1000, type, ne0, ne1, 1, 1);
            size_t           w = 0;
            size_t           s = 0;
            if (claims(type, head, false, true, true) && ggml_sycl::zone_dequant_f16_weight_bytes(ne0, ne1, &w) &&
                ggml_sycl::zone_dequant_f16_src1_bytes_per_token(ne0, 1, 1, &s)) {
                d.dequant_f16_if_unsupplied_weight_bytes         = w;
                d.dequant_f16_if_unsupplied_src1_bytes_per_token = s;
            }
            d.pp_scratch_type_enabled = false;
            return d;
        };
        std::vector<zone_tensor_desc> mistral;
        for (int il = 0; il < 2; il++) {
            mistral.push_back(marked("blk.N.attn_q.weight", GGML_TYPE_Q4_K, 4096, 4096, false));
            mistral.push_back(marked("blk.N.attn_k.weight", GGML_TYPE_Q4_K, 4096, 1024, false));
            mistral.push_back(marked("blk.N.attn_v.weight", GGML_TYPE_Q6_K, 4096, 1024, false));
            mistral.push_back(marked("blk.N.attn_output.weight", GGML_TYPE_Q4_K, 4096, 4096, false));
            mistral.push_back(marked("blk.N.ffn_gate.weight", GGML_TYPE_Q4_K, 4096, 14336, false));
            mistral.push_back(marked("blk.N.ffn_up.weight", GGML_TYPE_Q4_K, 4096, 14336, false));
            mistral.push_back(marked("blk.N.ffn_down.weight", GGML_TYPE_Q6_K, 14336, 4096, false));
        }
        std::vector<zone_tensor_desc> with_head = mistral;
        mistral.push_back(marked("output.weight", GGML_TYPE_Q6_K, 4096, 32000, true));
        const path_scoped_maxima plan = zone_scoped_maxima(mistral);
        CHECK(plan.dequant_f16_weight_bytes == 117440512, "the plan is the largest FFN weight's f16 copy (112 MiB)");
        CHECK(plan.dequant_f16_src1_bytes_per_token == 28672,
              "the activation row per token is ffn_down's 14336-wide input in f16");
        // Had the head been claimed, its f16 copy (250 MiB) would have sized the whole plan.
        with_head.push_back(marked("output.weight", GGML_TYPE_Q6_K, 4096, 32000, false));
        CHECK(zone_scoped_maxima(with_head).dequant_f16_weight_bytes == 262144000,
              "a claimed head would size the plan, which is why the claim leaves it out");

        // Negative controls, so the two figures above are read from the inventory and not passed by any plan. A
        // claimed weight one row wider than ffn_up is still claimed (the claim asks for a 2-D weight, not a shape) and
        // moves the plan off 112 MiB; the head marked as the adapter marks it carries no mark and sizes nothing.
        std::vector<zone_tensor_desc> wider = mistral;
        wider.push_back(marked("blk.N.ffn_up.weight", GGML_TYPE_Q4_K, 4096, 14337, false));
        const size_t wider_w = zone_scoped_maxima(wider).dequant_f16_weight_bytes;
        CHECK(wider_w != 117440512 && wider_w == 4096ull * 14337ull * F16_BYTES,
              "a one-row-wider FFN weight moves the plan off the 112 MiB figure");
        const zone_tensor_desc head = marked("output.weight", GGML_TYPE_Q6_K, 4096, 32000, true);
        CHECK(head.dequant_f16_if_unsupplied_weight_bytes == 0 &&
                  head.dequant_f16_if_unsupplied_src1_bytes_per_token == 0,
              "the head as the adapter marks it carries no dequant mark");
        CHECK(zone_scoped_maxima({ head }).dequant_f16_weight_bytes == 0 &&
                  zone_scoped_maxima({ head }).dequant_f16_src1_bytes_per_token == 0,
              "an inventory of only the head plans nothing");
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

        // The halves are the two components of ONE pair plan and travel with the bare plan they sum into. Maxing
        // each half on its own builds a pair no plan had: held (weights 100, activations 10) against live (10, 100)
        // is 110 either way, but the per-half maxima are (100, 100), above the zone's 110, so a bound equal to the bare
        // plan could not hold the planned pair and the first reservation would regrow after all.
        {
            zone_onednn_plan held_plan;
            held_plan.bare_bytes        = 110 * mib;
            held_plan.weights_bytes     = 100 * mib;
            held_plan.activations_bytes = 10 * mib;
            zone_onednn_plan live_plan;
            live_plan.bare_bytes           = 110 * mib;
            live_plan.weights_bytes        = 10 * mib;
            live_plan.activations_bytes    = 100 * mib;
            const zone_onednn_plan crossed = ggml_sycl::zone_onednn_plan_keep(held_plan, live_plan);
            CHECK(crossed.weights_bytes + crossed.activations_bytes <= crossed.bare_bytes,
                  "the kept halves never sum above the kept bare plan");
            size_t cw = 0, ca = 0;
            ggml_sycl::zone_onednn_scratch_reserve_target(true, crossed.bare_bytes, 0, 0, crossed.weights_bytes,
                                                          crossed.activations_bytes, 50 * mib, 5 * mib, &cw, &ca);
            CHECK(cw == 100 * mib && ca == 10 * mib,
                  "the kept plan's first reservation is its planned pair, which a bound equal to the bare plan holds");
            // An op of the OTHER plan, whose pair the kept one is not: (10, 100) fits the bound alone, but its per-half
            // merge with the kept planned pair is (100, 100), above it. The admission refuses it, so it draws the
            // planned dequant buffers and never reaches the reserve: the reserve would replace the kept pair with the
            // request as asked, and the superseded reservation's release is event-deferred, so the old and new blocks
            // would both be in the zone.
            CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, crossed.bare_bytes, crossed.weights_bytes,
                                                             crossed.activations_bytes, 10 * mib, 100 * mib),
                  "a crossed request the kept pair cannot merge with is refused at admission");
            size_t xw = 0, xa = 0;
            ggml_sycl::zone_onednn_scratch_reserve_target(true, crossed.bare_bytes, cw, ca, crossed.weights_bytes,
                                                          crossed.activations_bytes, 10 * mib, 100 * mib, &xw, &xa);
            CHECK(xw == 10 * mib && xa == 100 * mib,
                  "were it reserved, a crossed request would be used as asked, not merged above the bound");
            CHECK(xw + xa <= crossed.bare_bytes, "the crossed request stays within the bound");

            zone_onednn_plan big_plan;
            big_plan.bare_bytes        = 200 * mib;
            big_plan.weights_bytes     = 120 * mib;
            big_plan.activations_bytes = 80 * mib;
            zone_onednn_plan small_plan;
            small_plan.bare_bytes         = 150 * mib;
            small_plan.weights_bytes      = 140 * mib;
            small_plan.activations_bytes  = 10 * mib;
            const zone_onednn_plan larger = ggml_sycl::zone_onednn_plan_keep(small_plan, big_plan);
            CHECK(larger.bare_bytes == 200 * mib && larger.weights_bytes == 120 * mib &&
                      larger.activations_bytes == 80 * mib,
                  "the halves come from the plan with the larger bare plan, whichever side it is on");
            const zone_onednn_plan smaller = ggml_sycl::zone_onednn_plan_keep(big_plan, small_plan);
            CHECK(smaller.weights_bytes == 120 * mib && smaller.activations_bytes == 80 * mib,
                  "a later, smaller plan does not replace the halves of the larger one");
        }

        // The reserve merges per-component maxima, so two ops that each fit the bound can hold a pair above it. The
        // merge is bounded by the PAIR BOUND, not the capacity: the held pair would otherwise eat the Graph floor.
        const size_t bound = 235 * mib;
        size_t       w = 0, a = 0;
        ggml_sycl::zone_onednn_scratch_reserve_target(true, bound, 200 * mib, 4 * mib, 0, 0, 40 * mib, 40 * mib, &w,
                                                      &a);
        CHECK(w == 40 * mib && a == 40 * mib, "a merge above the bound is not held: the request is used as asked");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 256 * mib, 200 * mib, 4 * mib, 0, 0, 40 * mib, 40 * mib, &w,
                                                      &a);
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
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, 0, 0, plan_w, plan_a, 80 * mib, 14 * mib, &w,
                                                      &a);
        CHECK(w == plan_w && a == plan_a,
              "nothing held: the first reservation is the planned pair, not the first op's");
        const size_t first_w = w, first_a = a;
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, first_w, first_a, plan_w, plan_a, 112 * mib,
                                                      28 * mib, &w, &a);
        CHECK(w == first_w && a == first_a, "a later planned op, even the widest, is covered by what is held");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, 0, 0, plan_w, plan_a, 120 * mib, 14 * mib, &w,
                                                      &a);
        CHECK(w == 120 * mib && a == plan_a, "a request above the plan in one half still grows that half");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 100 * mib, 0, 0, plan_w, plan_a, 40 * mib, 10 * mib, &w,
                                                      &a);
        CHECK(w == 40 * mib && a == 10 * mib, "a zone clamped below its plan reserves what is asked, not the plan");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 100 * mib, 60 * mib, 5 * mib, plan_w, plan_a, 50 * mib,
                                                      12 * mib, &w, &a);
        CHECK(w == 60 * mib && a == 12 * mib,
              "a clamped zone still never shrinks what is held: the held-and-requested merge, without the plan");
        ggml_sycl::zone_onednn_scratch_reserve_target(false, 0, 0, 0, plan_w, plan_a, 40 * mib, 10 * mib, &w, &a);
        CHECK(w == 40 * mib && a == 10 * mib, "without an arena there is no planned pair to reserve");
        ggml_sycl::zone_onednn_scratch_reserve_target(true, 192 * mib, 0, 0, plan_w, plan_a, 40 * mib, 10 * mib,
                                                      nullptr, nullptr);
        // the pair halves are part of the snapshot a kept zone is described by
        const zone_onednn_plan big   = { 144 * mib, 64 * mib, plan_w, plan_a };
        const zone_onednn_plan small = { 30 * mib, 10 * mib, 20 * mib, 4 * mib };
        const zone_onednn_plan kept  = ggml_sycl::zone_onednn_plan_keep(big, small);
        CHECK(kept.weights_bytes == plan_w && kept.activations_bytes == plan_a,
              "a smaller later plan does not shrink the pair halves the kept zone was built for");
    }

    // ---- Case 14m: the planned pair's activations half is the micro-batch times the widest row the scratch takes ----
    // Qwen3.8 Flash Next IQ4_NL on the B70 at -p 2048 -ub 2048: the pair was planned as (largest dequantized eligible
    // weight, largest eligible weight AS STORED) = (60.0, 31.9) MB. The second figure is no activation size: the ops
    // ask for 2048 rows x 10240 x f16 = 40.0 MB. With the zone exactly pair + Graph floor the pair bound was 91.9 MB,
    // the per-component maximum (60, 40) did not fit it, and every op of a layer replaced the held pair.
    {
        const size_t mib = 1024u * 1024u;

        // The classifier: the row is the widest marked row inside the eligible set the weights half covers. Names are
        // wrong for the GGUF convention on purpose, as in every fixture of this file.
        std::vector<zone_tensor_desc> inv;
        const auto                    marked = [](zone_tensor_desc d) {
            d.onednn_pp_src1_bytes_per_token = static_cast<size_t>(d.ne[0]) * F16_BYTES;
            return d;
        };
        for (int i = 0; i < 6; ++i) {
            const std::string n = std::to_string(i);
            // the widest row the scratch takes: K = 10240, 20480 B/token (reorder 40 MiB)
            inv.push_back(marked(desc("token_embd.weight.a" + n, 11796480, TYPE_Q4_0, 10240, 2048, 1, 1)));
            // the largest dequantized weight: 60 MiB, and the largest eligible weight as stored (31.9 MB)
            inv.push_back(marked(desc("token_embd.weight.b" + n, 33423360, TYPE_Q8_0, 2048, 15360, 1, 1)));
            // an eligible weight of a type the scratch is not enabled for: the adapter leaves it unmarked
            inv.push_back(desc("token_embd.weight.c" + n, 9437184, TYPE_Q4_K, 16384, 1024, 1, 1));
            // an expert stack with a wider row: the PP MoE ring takes it, not this scratch
            inv.push_back(marked(desc("token_embd.weight.d" + n, 570425344, TYPE_MXFP4, 16384, 2048, 32, 1)));
        }
        // a singleton (the head) with a wider row: outside the eligible set, so outside the pair plan
        inv.push_back(marked(desc("blk.0.ffn_up.weight", 2644901888, TYPE_Q8_0, 16384, 151936, 1, 1)));
        for (int i = 0; i < 4; ++i) {
            // a gather-only family with a wider row: no MUL_MAT operand
            zone_tensor_desc g = marked(desc("blk.9" + std::to_string(i), 14155776, TYPE_Q4_0, 24576, 1024, 1, 1));
            g.get_rows_only    = true;
            inv.push_back(g);
        }
        const path_scoped_maxima m = zone_scoped_maxima(inv);
        CHECK(m.onednn_pp_src1_bytes_per_token == 20480,
              "the activations row is the widest marked row of the eligible set, no expert, head or gather row");
        CHECK(m.onednn_reorder == 60 * mib, "the weights half is the largest dequantized eligible weight");
        CHECK(m.onednn_eligible == 33423360, "the largest eligible weight as stored is the 31.9 MB placeholder");

        // The pure plan: n_ubatch rows of that row, beside the weights half.
        const ggml_sycl::zone_onednn_pp_pair at_2048 =
            ggml_sycl::zone_onednn_pp_pair_plan(m.onednn_reorder, m.onednn_pp_src1_bytes_per_token, 2048);
        CHECK(at_2048.sized && at_2048.weights_bytes == 60 * mib && at_2048.activations_bytes == 40 * mib,
              "the planned pair at n_ubatch=2048 is (60 MiB, 40 MiB)");
        CHECK(at_2048.src1_bytes_per_token == 20480 && at_2048.n_ubatch == 2048,
              "the plan records the row and the micro-batch it was sized from");
        CHECK(at_2048.activations_bytes != m.onednn_eligible,
              "the activations half is not the stored-weight placeholder");
        const ggml_sycl::zone_onednn_pp_pair at_512 =
            ggml_sycl::zone_onednn_pp_pair_plan(m.onednn_reorder, m.onednn_pp_src1_bytes_per_token, 512);
        CHECK(at_512.sized && at_512.weights_bytes == 60 * mib && at_512.activations_bytes == 10 * mib,
              "the activations half follows the micro-batch: 10 MiB at n_ubatch=512");
        const ggml_sycl::zone_onednn_pp_pair no_rows = ggml_sycl::zone_onednn_pp_pair_plan(60 * mib, 20480, 0);
        CHECK(no_rows.sized && no_rows.weights_bytes == 60 * mib && no_rows.activations_bytes == 0,
              "no micro-batch plans no activations half");
        const ggml_sycl::zone_onednn_pp_pair no_row = ggml_sycl::zone_onednn_pp_pair_plan(60 * mib, 0, 2048);
        CHECK(no_row.sized && no_row.activations_bytes == 0, "no marked row plans no activations half");
        const ggml_sycl::zone_onednn_pp_pair wrapped =
            ggml_sycl::zone_onednn_pp_pair_plan(60 * mib, SIZE_MAX / 2 + 1, 2);
        CHECK(!wrapped.sized && wrapped.weights_bytes == 60 * mib && wrapped.activations_bytes == 0,
              "an overflowing activations half is not wrapped into a small one: it is 0 and the pair is unsized");
        const ggml_sycl::zone_onednn_pp_pair wrapped_sum = ggml_sycl::zone_onednn_pp_pair_plan(SIZE_MAX, 1, 1);
        CHECK(!wrapped_sum.sized && wrapped_sum.weights_bytes == SIZE_MAX && wrapped_sum.activations_bytes == 0,
              "a pair whose sum overflows keeps only the weights half");

        // The reserve over the layer's five requests: with the zone exactly pair + Graph floor (576 MiB at
        // n_head_ctx_max=24, n_ubatch=2048, n_ctx=2048), the first reservation is the planned pair and no later op of
        // the layer replaces it.
        const size_t floor_b        = 576 * mib;
        const size_t requests[5][2] = {
            { 62 * mib / 10, 40 * mib       },
            { 50 * mib,      40 * mib       },
            { 60 * mib,      10 * mib       },
            { 60 * mib,      319 * mib / 10 },
            { mib / 10,      40 * mib       },
        };
        const size_t bare  = at_2048.weights_bytes + at_2048.activations_bytes;
        const size_t bound = ggml_sycl::zone_onednn_pp_pair_bound(bare + floor_b, bare, floor_b);
        CHECK(bound == 100 * mib, "a zone of exactly pair + floor bounds the pair at its plan");
        size_t held_w = 0, held_a = 0;
        for (const auto & r : requests) {
            CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, at_2048.weights_bytes,
                                                            at_2048.activations_bytes, r[0], r[1]),
                  "every request is admitted");
            size_t w = 0, a = 0;
            ggml_sycl::zone_onednn_scratch_reserve_target(true, bound, held_w, held_a, at_2048.weights_bytes,
                                                          at_2048.activations_bytes, r[0], r[1], &w, &a);
            CHECK(w == 60 * mib && a == 40 * mib, "the target is (60, 40) from the first request on and never changes");
            held_w = w;
            held_a = a;
        }

        // Control: the placeholder plan (60, 31.9) at the same floor replaces the held pair on every op after the
        // first, which is what filled the zone with superseded blocks inside one graph.
        const size_t old_a     = 319 * mib / 10;
        const size_t old_bare  = 60 * mib + old_a;
        const size_t old_bound = ggml_sycl::zone_onednn_pp_pair_bound(old_bare + floor_b, old_bare, floor_b);
        held_w                 = 0;
        held_a                 = 0;
        int replaced           = 0;
        for (size_t i = 0; i < 5; ++i) {
            size_t w = 0, a = 0;
            ggml_sycl::zone_onednn_scratch_reserve_target(true, old_bound, held_w, held_a, 60 * mib, old_a,
                                                          requests[i][0], requests[i][1], &w, &a);
            if (i > 0 && (w != held_w || a != held_a)) {
                ++replaced;
            }
            held_w = w;
            held_a = a;
        }
        CHECK(replaced == 4, "the placeholder plan replaces the pair four times in the five-op layer");
    }

    // ---- Case 14n: admission refuses an op whose per-half merge with the planned pair leaves the bound -------------
    // The pair plan is (60, 40) MiB and the zone is exactly pair + Graph floor, so the bound is 100 MiB. A context that
    // runs a wider micro-batch than the load planned asks (0.1, 80) MiB: that fits the bound alone, but its per-half
    // maximum with the planned pair is (60, 80) = 140 MiB. Admitted on the request alone, the reserve could merge it
    // with neither the planned nor the held pair and replaced the held pair with the request as asked; the superseded
    // pair's release is event-deferred, so each replacement inside one graph left two blocks in the ONEDNN zone.
    // Refused by the predicate the graph-entry walk asks too, the op draws the planned dequant buffers the walk sizes
    // for it, and the held pair stays.
    {
        const size_t mib    = 1024u * 1024u;
        const size_t plan_w = 60 * mib;
        const size_t plan_a = 40 * mib;
        const size_t bound  = 100 * mib;

        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, plan_w, plan_a, mib / 10, 80 * mib),
              "(0.1, 80) fits the bound alone but cannot merge with the planned (60, 40): refused");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, plan_w, plan_a, 62 * mib / 10, 40 * mib),
              "(6.2, 40) merges into the planned (60, 40): admitted");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, bound, plan_w, plan_a, mib / 10, 80 * mib),
              "the verdict the op arm and the walk share refuses it too, so the walk sizes its dequant buffers");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_supplies(true, true, true, bound, plan_w, plan_a, 62 * mib / 10,
                                                         40 * mib),
              "and supplies the op that merges");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, plan_w, plan_a, 10 * mib, plan_a + 1),
              "one byte above the planned activations half is refused although the request alone is 50 MiB");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, plan_w, plan_a, plan_w + 1, 1 * mib),
              "one byte above the planned weights half is refused too");

        // A zone with slack above the plan: a half may grow past its plan while the merge still fits the bound.
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, 120 * mib, plan_w, plan_a, 10 * mib, 60 * mib),
              "slack: (10, 60) merges to (60, 60), exactly the 120 MiB bound: admitted");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, 120 * mib, plan_w, plan_a, 10 * mib, 60 * mib + 1),
              "slack: one byte past the bound in the merge is refused although the request alone is 70 MiB");

        // A stored plan whose halves do not fit the bound keeps the request-only rule: no request can merge with a
        // pair the zone cannot hold. The arena floors the zone at the bare plan, so only an inconsistent stored plan
        // reaches this.
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, 80 * mib, plan_w, plan_a, 10 * mib, 70 * mib),
              "clamped zone: a request that fits the bound alone is admitted");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, 80 * mib, plan_w, plan_a, 10 * mib, 70 * mib + 1),
              "clamped zone: a request above the bound alone is refused");
        // No planned pair, and no arena.
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, bound, 0, 0, mib / 10, 80 * mib),
              "no planned pair: the request alone is held against the bound");
        CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(false, 0, plan_w, plan_a, mib / 10, 80 * mib),
              "no arena: there is no zone to plan against");
        CHECK(!ggml_sycl::zone_onednn_pp_scratch_planned(true, SIZE_MAX, SIZE_MAX - 10, 5, 1, 20),
              "a merged sum that overflows is refused, not wrapped into a small one");

        // For every admitted request the reserve target is the per-half maximum of the planned pair, the held pair and
        // the request, never the request as asked. The held pair is what the first admitted request reserved. In a
        // zone of exactly pair + floor nothing admitted exceeds the plan, so the target is the planned pair itself.
        // With slack, two admitted requests that exceed the plan in DIFFERENT halves can still fail to merge: that is
        // the one reach of the "as asked" target in a zone that holds its plan, counted here and shown below.
        const size_t step = 5 * mib;
        const size_t top  = 120 * mib;
        for (const size_t zone_bound : { bound, 120 * mib }) {
            size_t admitted   = 0;
            size_t refused    = 0;  // fits the bound alone, refused because it cannot merge with the plan
            size_t crossed    = 0;
            bool   first_ok   = true;
            bool   later_ok   = true;
            bool   planned_ok = true;
            for (size_t w1 = 0; w1 <= top; w1 += step) {
                for (size_t a1 = 0; a1 <= top; a1 += step) {
                    if (!ggml_sycl::zone_onednn_pp_scratch_planned(true, zone_bound, plan_w, plan_a, w1, a1)) {
                        refused += w1 + a1 <= zone_bound ? 1 : 0;
                        continue;
                    }
                    ++admitted;
                    size_t held_w = 0, held_a = 0;
                    ggml_sycl::zone_onednn_scratch_reserve_target(true, zone_bound, 0, 0, plan_w, plan_a, w1, a1,
                                                                  &held_w, &held_a);
                    first_ok   = first_ok && held_w == std::max(plan_w, w1) && held_a == std::max(plan_a, a1);
                    planned_ok = planned_ok && (zone_bound != bound || (held_w == plan_w && held_a == plan_a));
                    for (size_t w2 = 0; w2 <= top; w2 += step) {
                        for (size_t a2 = 0; a2 <= top; a2 += step) {
                            if (!ggml_sycl::zone_onednn_pp_scratch_planned(true, zone_bound, plan_w, plan_a, w2, a2)) {
                                continue;
                            }
                            if ((w1 > plan_w && a2 > plan_a) || (a1 > plan_a && w2 > plan_w)) {
                                ++crossed;
                                continue;
                            }
                            size_t w = 0, a = 0;
                            ggml_sycl::zone_onednn_scratch_reserve_target(true, zone_bound, held_w, held_a, plan_w,
                                                                          plan_a, w2, a2, &w, &a);
                            later_ok = later_ok && w == std::max({ plan_w, held_w, w2 }) &&
                                       a == std::max({ plan_a, held_a, a2 });
                        }
                    }
                }
            }
            CHECK(first_ok, "the first reservation of an admitted request is max(planned, request) per half");
            CHECK(later_ok,
                  "a later admitted request is reserved at max(planned, held, request) per half, never as asked");
            CHECK(planned_ok, "with the zone exactly pair + floor every admitted request is held as the planned pair");
            CHECK(admitted > 0, "the sweep admits requests");
            CHECK(refused > 0, "the sweep reaches requests that fit the bound alone and cannot merge with the plan");
            if (zone_bound == bound) {
                CHECK(crossed == 0,
                      "with no slack nothing admitted exceeds the plan, so no two admitted requests cross");
            } else {
                CHECK(crossed > 0, "with slack two admitted requests can exceed the plan in different halves");
            }
        }

        // The cross, explicitly: planned (60, 40) at a 120 MiB bound. A head-shaped op (75, 10) is admitted (merge
        // (75, 40) = 115) and held as (75, 40); a wider-micro-batch op (10, 55) is admitted (merge (60, 55) = 115), but
        // (75, 55) = 130 is above the bound: the target is the request as asked, which replaces the held pair.
        {
            const size_t slack_bound = 120 * mib;
            CHECK(ggml_sycl::zone_onednn_pp_scratch_planned(true, slack_bound, plan_w, plan_a, 75 * mib, 10 * mib) &&
                      ggml_sycl::zone_onednn_pp_scratch_planned(true, slack_bound, plan_w, plan_a, 10 * mib, 55 * mib),
                  "both crossing ops are admitted on their own");
            size_t hw = 0, ha = 0, w = 0, a = 0;
            ggml_sycl::zone_onednn_scratch_reserve_target(true, slack_bound, 0, 0, plan_w, plan_a, 75 * mib, 10 * mib,
                                                          &hw, &ha);
            CHECK(hw == 75 * mib && ha == plan_a, "the head-shaped op holds (75, 40)");
            ggml_sycl::zone_onednn_scratch_reserve_target(true, slack_bound, hw, ha, plan_w, plan_a, 10 * mib, 55 * mib,
                                                          &w, &a);
            CHECK(w == 10 * mib && a == 55 * mib,
                  "the crossing op is reserved as asked: the one reach of that target in a zone holding its plan");
        }
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
        // meets the allocator's raw gate like any raw request.
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

        // The plan inputs' merge across live models is the owner table's (planned-load-terms.hpp, llama.cpp-45qo),
        // unit-tested in tests/test-planned-load-terms.cpp.
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

    // ---- Case 24b (llama.cpp-kjef, ring fit term): the prefill ring is lent bytes, so it costs the card nothing
    // EXCEPT the part a loan cannot supply; that remainder is a demand on free memory and is part of the ONE demand
    // every fit consumer reads. The Case 24 inputs (512: plan 76 MiB + request 495 MiB - 100 MiB of room = 471 MiB,
    // 626 MiB left of 1097 MiB) with 400 MiB of the ring uncovered leave 226 MiB against a 256 MiB headroom. -------
    {
        const size_t MiB  = 1024 * 1024;
        auto         plan = [](void *, uint32_t) -> size_t {
            return 76 * 1024 * 1024;
        };
        ggml_sycl::zone_hold_rung_request rungs[] = {
            { 512,  495 * MiB },
            { 1024, 990 * MiB }
        };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 1097 * MiB;
        in.kv_room                         = 100 * MiB;
        in.rungs                           = rungs;
        in.n_rungs                         = 2;
        in.plan_of                         = plan;
        CHECK(in.ring_bytes == 0, "no uncovered ring unless a caller says so");
        CHECK(ggml_sycl::zone_hold_fit(in, 512), "baseline: 512 fits with no ring bytes to charge");
        const size_t base = ggml_sycl::zone_hold_fit_demand(in, 512);
        in.ring_bytes     = 400 * MiB;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 512) == base + 400 * MiB,
              "demand = today's value + the uncovered ring");
        CHECK(!ggml_sycl::zone_hold_fit(in, 512), "512 does not fit once 400 MiB of the ring is uncovered");
        CHECK(ggml_sycl::zone_hold_fit(in, 256), "256 passes under the same inputs");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 512) == 256, "the -ub the refusal names is 256");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 256) == 256, "a rung that fits is returned unchanged");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 0) == 0, "an unknown n_ubatch still names no demand");
        // A ring a loan covers is no demand at all, so the verdict is today's.
        in.ring_bytes = 0;
        CHECK(ggml_sycl::zone_hold_fit(in, 512), "a fully covered ring changes nothing");
        // Saturating: a huge figure never wraps to a small demand that reads as a fit.
        in.ring_bytes = SIZE_MAX;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 512) == SIZE_MAX, "a huge ring saturates the demand");
        CHECK(!ggml_sycl::zone_hold_fit(in, 512), "and the rung is refused, never fitted by a wrapped sum");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 512) == 0, "no rung fits an unbounded ring: nothing is named");
        // No plan function, no hold: the ring is still a demand, and the only one.
        in.plan_of    = nullptr;
        in.ring_bytes = 400 * MiB;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 512) == 400 * MiB, "with no plan the demand is the ring alone");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 0) == 0, "an unknown n_ubatch names no demand, plan or not");
        in.ring_bytes = 0;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 512) == 0, "no plan and no ring is no demand");
    }

    // ---- Case 24c (compute room): the RUNTIME zone of an explicit micro-batch the load placed holds the planned
    // compute term beside that micro-batch's ring and dense scratch, so at that rung a compute buffer no larger than
    // the term is served by the zone and the hold cannot keep it out. The B70 / Qwen3.8 IQ3_XXS context at -ub 2048:
    // plan 98 MiB, one 1960 MiB compute buffer, 64 MiB of KV-zone room, 1946 MiB free outside the arena, and the
    // 2055 MiB compute term the load measured at its planning shape (n_ctx_train, ubatch 512). Without the room the
    // bound charges the whole buffer as a spill and refuses the context, although no request was held out. ------------
    {
        const size_t MiB  = 1024 * 1024;
        auto         plan = [](void *, uint32_t) -> size_t {
            return 98 * 1024 * 1024;
        };
        ggml_sycl::zone_hold_rung_request rungs[] = {
            { 2048, 1960 * MiB }
        };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 1946 * MiB;
        in.kv_room                         = 64 * MiB;
        in.rungs                           = rungs;
        in.n_rungs                         = 1;
        in.plan_of                         = plan;
        CHECK(in.compute_room_n_ubatch == 0 && in.compute_room_bytes == 0, "no compute room unless a caller says so");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 2048) == 98 * MiB + 1960 * MiB - 64 * MiB,
              "without a room: plan + the whole buffer - KV room, 1994 MiB");
        CHECK(!ggml_sycl::zone_hold_fit(in, 2048), "and 2048 is refused against 1946 MiB free");
        const uint32_t largest_without = ggml_sycl::zone_hold_fit_largest_ub(in, 2048);
        const size_t   room            = 2055 * MiB;
        in.compute_room_n_ubatch       = 2048;
        in.compute_room_bytes          = room;
        CHECK(ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 2048, 1960 * MiB, true), "the buffer fits the room");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 2048) == 98 * MiB - 64 * MiB,
              "with the room the buffer spills nothing; the plan stays in the bound");
        CHECK(ggml_sycl::zone_hold_fit(in, 2048), "so 2048 fits: 1912 MiB stay free");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 2048) == 2048, "and a refusal would name 2048 itself");
        // The ring no loan covers is still a demand beside the credited buffer.
        in.ring_bytes = 100 * MiB;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 2048) == 98 * MiB - 64 * MiB + 100 * MiB,
              "the uncovered ring is charged with the room as without it");
        in.ring_bytes         = 0;
        // Whole or nothing: a room one MiB short of the buffer gives no part of itself.
        in.compute_room_bytes = 1959 * MiB;
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 2048, 1960 * MiB, true), "a buffer above the room");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 2048) == 98 * MiB + 1960 * MiB - 64 * MiB,
              "is charged whole, not less the room");
        CHECK(!ggml_sycl::zone_hold_fit(in, 2048), "and refused as without a room");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 2048) == largest_without, "with the advice given without one");
        // A buffer above the term the zone holds is held back whole, so it is charged whole: the case the kpjw refusal
        // and its largest -ub advice still answer.
        in.compute_room_bytes                     = room;
        ggml_sycl::zone_hold_rung_request above[] = {
            { 2048, 2100 * MiB }
        };
        in.rungs = above;
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 2048, 2100 * MiB, true),
              "a 2100 MiB buffer above the term");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 2048) == 98 * MiB + 2100 * MiB - 64 * MiB, "is charged whole");
        in.rungs   = rungs;
        // Before the rung's buffers exist (no record at all) the credit changes nothing: the demand is the plan alone.
        in.n_rungs = 0;
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 2048) == 98 * MiB - 64 * MiB, "no record: the plan alone");
        in.n_rungs            = 1;
        in.compute_room_bytes = room;

        // The load measured the term at its planning shape, ubatch 512, below the placed 2048. A rung without a record
        // of its own is served only at or below that rung, since compute buffers grow with the micro-batch: 1024,
        // scaled from the 2048 record, is judged as before, and 512 is served.
        in.compute_room_measured_n_ubatch = 512;
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 1024, 980 * MiB, false),
              "1024 is above the rung the term was measured at");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 1024) == 98 * MiB + 980 * MiB - 64 * MiB,
              "so 1024 is charged its scaled buffer whole");
        CHECK(ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 512, 490 * MiB, false), "512 is at the measured rung");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 512) == 98 * MiB - 64 * MiB,
              "so 512's scaled buffer is credited: the plan alone");
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 4096, 1, false), "a rung above the room is not");
        // An unknown micro-batch never matches a room, not even a room for micro-batch 0.
        in.compute_room_n_ubatch = 0;
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 0, 0, false), "n_ubatch 0 has no room");
        CHECK(ggml_sycl::zone_hold_fit_demand(in, 0) == 0, "and no demand");
        // The scaler, through the spill bound that adds a 1-byte plan to it: rounded up, unchanged for an unknown or
        // equal micro-batch, saturating on overflow.
        CHECK(ggml_sycl::zone_hold_spill_bound(1, 3, 2, 3) == 1 + 5, "4.5 rounds up to 5");
        CHECK(ggml_sycl::zone_hold_spill_bound(1, 7 * MiB, 0, 2048) == 1 + 7 * MiB, "an unknown source micro-batch");
        CHECK(ggml_sycl::zone_hold_spill_bound(1, 7 * MiB, 2048, 0) == 1 + 7 * MiB, "an unknown target micro-batch");
        CHECK(ggml_sycl::zone_hold_spill_bound(1, 7 * MiB, 512, 512) == 1 + 7 * MiB, "an equal one");
        CHECK(ggml_sycl::zone_hold_spill_bound(1, SIZE_MAX / 2, 512, 2048) == SIZE_MAX, "an overflowing product");
    }

    // The terms the hold-spill refusal prints are the ones the demand is made of.
    {
        const size_t MiB  = 1024 * 1024;
        auto         plan = [](void *, uint32_t) -> size_t {
            return 1100 * 1024 * 1024;
        };
        ggml_sycl::zone_hold_rung_request rungs[] = {
            { 512, 490 * MiB }
        };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 1764 * MiB;
        in.kv_room                         = 64 * MiB;
        in.rungs                           = rungs;
        in.n_rungs                         = 1;
        in.plan_of                         = plan;
        in.ring_bytes                      = 2 * MiB;
        ggml_sycl::zone_hold_fit_terms t   = ggml_sycl::zone_hold_fit_terms_at(in, 512);
        CHECK(t.plan == 1100 * MiB && t.request == 490 * MiB && !t.request_credited, "plan and the uncredited request");
        CHECK(t.kv_room == 64 * MiB && t.ring_uncovered == 2 * MiB, "the KV room and the uncovered ring");
        CHECK(t.demand == ggml_sycl::zone_hold_fit_demand(in, 512), "the terms' demand is the fit's demand");
        CHECK(t.demand == 1100 * MiB + 490 * MiB - 64 * MiB + 2 * MiB, "1528 MiB");
        in.compute_room_n_ubatch = 512;
        in.compute_room_bytes    = 2057 * MiB;
        t                        = ggml_sycl::zone_hold_fit_terms_at(in, 512);
        CHECK(t.request == 490 * MiB && t.request_credited, "a credited request is still named, and marked");
        CHECK(t.demand == 1100 * MiB - 64 * MiB + 2 * MiB && t.demand == ggml_sycl::zone_hold_fit_demand(in, 512),
              "and adds nothing");
        t = ggml_sycl::zone_hold_fit_terms_at(in, 0);
        CHECK(t.demand == 0 && t.plan == 0 && t.request == 0, "an unknown n_ubatch names nothing");
    }

    // The micro-batch the compute term was planned at, as each load records it: the first load's value stands while
    // later loads agree; a load that disagrees while another load still holds a term turns the record off until a load
    // is taken in while none holds one, so no rung is credited at a shape the live term was not measured at.
    {
        using ggml_sycl::zone_compute_term_n_ubatch_read;
        using ggml_sycl::zone_compute_term_n_ubatch_store;
        uint32_t s = 0;
        CHECK(zone_compute_term_n_ubatch_read(s) == 0, "nothing recorded reads 0");
        s = zone_compute_term_n_ubatch_store(s, 512, true);
        CHECK(zone_compute_term_n_ubatch_read(s) == 512, "0 then 512 records 512");
        s = zone_compute_term_n_ubatch_store(s, 512, true);
        CHECK(zone_compute_term_n_ubatch_read(s) == 512, "512 then 512 keeps 512");
        CHECK(zone_compute_term_n_ubatch_store(s, 0, true) == s, "a load naming no micro-batch changes nothing");
        s = zone_compute_term_n_ubatch_store(s, 1024, true);
        CHECK(zone_compute_term_n_ubatch_read(s) == 0, "512 then 1024 disagree: 0");
        s = zone_compute_term_n_ubatch_store(s, 1024, true);
        CHECK(zone_compute_term_n_ubatch_read(s) == 0, "and a later load cannot restore it while a term is live");
        s = zone_compute_term_n_ubatch_store(s, 512, true);
        CHECK(zone_compute_term_n_ubatch_read(s) == 0, "not even with the first value, while a term is live");
        CHECK(zone_compute_term_n_ubatch_store(0, UINT32_MAX, true) == 0,
              "a micro-batch equal to the disagreement mark records nothing on a first load");
        CHECK(zone_compute_term_n_ubatch_read(zone_compute_term_n_ubatch_store(512, UINT32_MAX, true)) == 512,
              "and turns no record off: only a real disagreement does");
        // Loads one after another (llama-bench -ub 512,1024): the first load's term is gone when the second load's
        // inventory is taken in, so its micro-batch replaces the record instead of turning it off.
        CHECK(zone_compute_term_n_ubatch_store(512, 1024, /*term_live=*/false) == 1024,
              "no live term: the new load's micro-batch replaces the record");
        CHECK(zone_compute_term_n_ubatch_store(512, 1024, /*term_live=*/true) ==
                  ggml_sycl::k_zone_compute_term_n_ubatch_disagreed,
              "two live loads that disagree still turn it off");
        CHECK(zone_compute_term_n_ubatch_store(ggml_sycl::k_zone_compute_term_n_ubatch_disagreed, 1024,
                                               /*term_live=*/false) == 1024,
              "a record turned off restarts once no load holds a term");
        CHECK(zone_compute_term_n_ubatch_store(ggml_sycl::k_zone_compute_term_n_ubatch_disagreed, 1024,
                                               /*term_live=*/true) == ggml_sycl::k_zone_compute_term_n_ubatch_disagreed,
              "but not while a load still holds one");
        CHECK(zone_compute_term_n_ubatch_store(512, 0, /*term_live=*/false) == 512,
              "a load naming no micro-batch changes nothing with no live term");
        CHECK(zone_compute_term_n_ubatch_store(512, ggml_sycl::k_zone_compute_term_n_ubatch_disagreed,
                                               /*term_live=*/false) == 512,
              "a micro-batch equal to the mark changes nothing with no live term either");
    }

    // The setter WARNs on exactly the store that turns the record off: from a shape to the mark, never from the mark
    // again, never on a first record, a repeat or a restart.
    {
        using ggml_sycl::zone_compute_term_n_ubatch_turns_off;
        CHECK(zone_compute_term_n_ubatch_turns_off(512, UINT32_MAX), "512 to the mark turns it off");
        CHECK(!zone_compute_term_n_ubatch_turns_off(UINT32_MAX, UINT32_MAX), "the mark to the mark does not again");
        CHECK(!zone_compute_term_n_ubatch_turns_off(0, 512), "a first record does not");
        CHECK(!zone_compute_term_n_ubatch_turns_off(512, 512), "a repeat does not");
        CHECK(!zone_compute_term_n_ubatch_turns_off(0, 0), "no record and no store does not");
        CHECK(!zone_compute_term_n_ubatch_turns_off(UINT32_MAX, 1024), "a restart from the mark does not");
    }

    // ---- Case 24d (compute room, scoped): the room serves only rungs at or below the rung the plan sized it for. Over
    // every record set the cases above use, every rung from 32 to 8192 gets the demand the fit gave before the room
    // existed (spill bound over the plan and the rung's request, net of the KV room, plus the uncovered ring) when no
    // rung is named, even with a room of SIZE_MAX. When one is named, a rung gets the demand with its request credited
    // (the plan alone, net of the KV room, plus the uncovered ring) exactly when the rule written out below credits
    // it, and the demand before the room otherwise: at or below the room's rung, not the held-out rung, and either
    // with a record of its own (a room of SIZE_MAX holds any) or at or below the rung the term was measured at. The
    // arms name the term measured at the room's rung, below it (the planning shape under a placed micro-batch), and
    // not at all, and hold one rung out. So a fit given no room, and every rung above the room's rung of one given a
    // room, are judged as before. -------------------------------------------------------------------------------------
    {
        const size_t MiB = 1024 * 1024;

        // The demand without the room: zone_hold_fit_demand as it was before the room fields existed.
        auto demand_before_room = [](const ggml_sycl::zone_hold_fit_inputs & in, uint32_t n_ubatch) -> size_t {
            if (n_ubatch == 0) {
                return 0;
            }
            const size_t plan    = in.plan_of ? in.plan_of(in.plan_ctx, n_ubatch) : 0;
            const size_t request = ggml_sycl::zone_hold_fit_request(in, n_ubatch);
            const size_t spill   = ggml_sycl::zone_hold_spill_raw_demand(
                ggml_sycl::zone_hold_spill_bound(plan, request, n_ubatch, n_ubatch), in.kv_room);
            return in.ring_bytes > SIZE_MAX - spill ? SIZE_MAX : spill + in.ring_bytes;
        };
        // The demand of a rung the room serves: its request adds nothing to the bound, the plan stays in it.
        auto demand_credited = [](const ggml_sycl::zone_hold_fit_inputs & in, uint32_t n_ubatch) -> size_t {
            const size_t plan  = in.plan_of ? in.plan_of(in.plan_ctx, n_ubatch) : 0;
            const size_t spill = ggml_sycl::zone_hold_spill_raw_demand(
                ggml_sycl::zone_hold_spill_bound(plan, 0, n_ubatch, n_ubatch), in.kv_room);
            return in.ring_bytes > SIZE_MAX - spill ? SIZE_MAX : spill + in.ring_bytes;
        };
        auto plan_76 = [](void *, uint32_t) -> size_t {
            return 76 * 1024 * 1024;
        };
        auto plan_98 = [](void *, uint32_t) -> size_t {
            return 98 * 1024 * 1024;
        };
        auto plan_10 = [](void *, uint32_t) -> size_t {
            return 10 * 1024 * 1024;
        };
        auto plan_var = [](void *, uint32_t n_ubatch) -> size_t {
            return n_ubatch < 32 ? 400 * 1024 * 1024 : 1024 * 1024;
        };
        const ggml_sycl::zone_hold_plan_fn      plans[]  = { plan_76, plan_98, plan_10, plan_var, nullptr };
        const ggml_sycl::zone_hold_rung_request set_g7[] = {
            { 512,  495 * MiB },
            { 1024, 990 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_big[] = {
            { 1024, 1200 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_qwen[] = {
            { 2048, 1960 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_three[] = {
            { 512,  495 * MiB  },
            { 1024, 990 * MiB  },
            { 2048, 4000 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_sparse[] = {
            { 512,  495 * MiB  },
            { 2048, 1980 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_600[] = {
            { 600, 600 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_huge[] = {
            { 600, 6000 * MiB }
        };
        const ggml_sycl::zone_hold_rung_request set_tiny[] = {
            { 32, 1 * MiB }
        };

        struct record_set {
            const ggml_sycl::zone_hold_rung_request * rungs;
            size_t                                    n;
        };

        // The credit rule, written out from its statement rather than from the function under test.
        auto credited_by_rule = [](const record_set & set, uint32_t n, uint32_t room_n, uint32_t measured_n,
                                   uint32_t held_n) -> bool {
            if (n > room_n || n == held_n) {
                return false;
            }
            for (size_t i = 0; i < set.n; ++i) {
                if (set.rungs[i].n_ubatch == n) {
                    return true;
                }
            }
            return measured_n != 0 && n <= measured_n;
        };

        const record_set sets[] = {
            { nullptr,    0 },
            { set_g7,     2 },
            { set_big,    1 },
            { set_qwen,   1 },
            { set_three,  3 },
            { set_sparse, 2 },
            { set_600,    1 },
            { set_huge,   1 },
            { set_tiny,   1 },
        };
        const size_t kv_rooms[] = { 0, 64 * MiB, 100 * MiB };
        const size_t rings[]    = { 0, 400 * MiB };
        size_t       mismatches = 0;
        size_t       asked      = 0;
        for (const record_set & set : sets) {
            for (ggml_sycl::zone_hold_plan_fn plan : plans) {
                for (size_t kv_room : kv_rooms) {
                    for (size_t ring : rings) {
                        ggml_sycl::zone_hold_fit_inputs before  = {};
                        before.headroom_target                  = 256 * MiB;
                        before.free_before                      = 1946 * MiB;
                        before.kv_room                          = kv_room;
                        before.rungs                            = set.rungs;
                        before.n_rungs                          = set.n;
                        before.plan_of                          = plan;
                        before.ring_bytes                       = ring;
                        // No rung named: a room of any size is inert.
                        ggml_sycl::zone_hold_fit_inputs unnamed = before;
                        unnamed.compute_room_bytes              = SIZE_MAX;
                        // The rung the Qwen3.8 load placed named, with the term measured at that rung.
                        ggml_sycl::zone_hold_fit_inputs named   = unnamed;
                        named.compute_room_n_ubatch             = 2048;
                        named.compute_room_measured_n_ubatch    = 2048;
                        // The term measured at the load's planning shape, ubatch 512, below the placed rung.
                        ggml_sycl::zone_hold_fit_inputs low     = named;
                        low.compute_room_measured_n_ubatch      = 512;
                        // The rung the term was measured at unknown.
                        ggml_sycl::zone_hold_fit_inputs blind   = named;
                        blind.compute_room_measured_n_ubatch    = 0;
                        // A realized check at 1024 whose own buffer was held out.
                        ggml_sycl::zone_hold_fit_inputs held    = named;
                        held.held_out_n_ubatch                  = 1024;
                        for (uint32_t n = 32; n <= 8192; n += 32) {
                            const size_t want   = demand_before_room(before, n);
                            const size_t credit = demand_credited(before, n);
                            auto         expect = [&](uint32_t measured_n, uint32_t held_n) -> size_t {
                                return credited_by_rule(set, n, 2048, measured_n, held_n) ? credit : want;
                            };
                            ++asked;
                            if (ggml_sycl::zone_hold_fit_demand(before, n) != want ||
                                ggml_sycl::zone_hold_fit_demand(unnamed, n) != want ||
                                ggml_sycl::zone_hold_fit(unnamed, n) != ggml_sycl::zone_hold_fit(before, n) ||
                                ggml_sycl::zone_hold_fit_demand(named, n) != expect(2048, 0) ||
                                ggml_sycl::zone_hold_fit_demand(low, n) != expect(512, 0) ||
                                ggml_sycl::zone_hold_fit_demand(blind, n) != expect(0, 0) ||
                                ggml_sycl::zone_hold_fit_demand(held, n) != expect(2048, 1024)) {
                                ++mismatches;
                            }
                        }
                        if (ggml_sycl::zone_hold_fit_largest_ub(unnamed, 8192) !=
                            ggml_sycl::zone_hold_fit_largest_ub(before, 8192)) {
                            ++mismatches;
                        }
                    }
                }
            }
        }
        CHECK(asked == 9 * 5 * 3 * 2 * 256, "the sweep asked every rung of every configuration");
        CHECK(mismatches == 0,
              "a named room credits exactly the rungs its rule credits; every other rung is judged as before the "
              "room");
    }

    // ---- Case 24e (a RUNTIME zone an earlier load grew): the arena reserves the larger of the floor and the
    // requirement, rounded up to 256 bytes, so a zone a load sized for itself is never reported as kept above its
    // plan, and only a zone above that reservation is. The second instance of a -ub 2048,512 sweep plans 512
    // against the zone the 2048 load grew. -------------------------------------------------------------------------
    {
        const size_t MiB   = 1024 * 1024;
        const size_t floor = 64 * MiB;

        // The reservation itself, against figures written out by hand.
        size_t zone = 0;
        CHECK(ggml_sycl::zone_runtime_reserved_bytes(floor, 10 * MiB, &zone) && zone == floor, "the floor wins");
        CHECK(ggml_sycl::zone_runtime_reserved_bytes(floor, 18050 * MiB, &zone) && zone == 18050 * MiB,
              "an aligned requirement is reserved as it is");
        CHECK(ggml_sycl::zone_runtime_reserved_bytes(floor, 18050 * MiB + 1, &zone) && zone == 18050 * MiB + 256,
              "one byte over rounds up to the next 256-byte grain");
        CHECK(ggml_sycl::zone_runtime_reserved_bytes(0, 0, &zone) && zone == 0, "nothing reserves nothing");
        zone = 7;
        CHECK(!ggml_sycl::zone_runtime_reserved_bytes(floor, SIZE_MAX - 254, &zone) && zone == 7,
              "a rounding that overflows fails and leaves the output alone");
        CHECK(ggml_sycl::zone_runtime_reserved_bytes(floor, SIZE_MAX - 255, &zone) && zone == SIZE_MAX - 255,
              "the largest aligned requirement is reserved");

        // The live zones below are the reservations above, written out: 18050 MiB, 18050 MiB + 256, the floor.
        CHECK(!ggml_sycl::zone_runtime_kept_above_plan(18050 * MiB, floor, 18050 * MiB),
              "a zone the load sized for its own requirement is not kept above its plan");
        CHECK(!ggml_sycl::zone_runtime_kept_above_plan(18050 * MiB + 256, floor, 18050 * MiB + 1),
              "nor is one rounded up to the arena's 256-byte grain (255 bytes above)");
        CHECK(!ggml_sycl::zone_runtime_kept_above_plan(floor, floor, 10 * MiB),
              "nor one the floor sized, far above a small requirement");
        CHECK(ggml_sycl::zone_runtime_kept_above_plan(18050 * MiB, floor, 2264 * MiB),
              "a zone a 2048 load grew is kept above the plan of a later 512 load");
        CHECK(ggml_sycl::zone_runtime_kept_above_plan(2264 * MiB + 256, floor, 2264 * MiB),
              "one grain above the plan is kept above it");
        CHECK(ggml_sycl::zone_runtime_kept_above_plan(18050 * MiB + 512, floor, 18050 * MiB + 1),
              "and one grain above a rounded plan is too");
        CHECK(!ggml_sycl::zone_runtime_kept_above_plan(2264 * MiB, floor, 18050 * MiB),
              "a zone below the plan is not kept above it");
        CHECK(!ggml_sycl::zone_runtime_kept_above_plan(SIZE_MAX, floor, SIZE_MAX),
              "a saturated plan never wraps into a kept zone");
    }

    // ---- Case 24f: the context shape a plan sizes the per-context scratch for. At load it is the context the model
    // opens with (the caller's, or the model's default) and the micro-batch the caller named, each else the load's
    // planning shape; at a context transaction it is the context's own.
    {
        using ggml_sycl::zone_plan_context_shape;
        ggml_sycl::zone_context_shape s = zone_plan_context_shape(false, 512, 4096, 512, 2048);
        CHECK(s.n_ctx == 4096 && s.n_ubatch == 2048, "a load the caller named a shape for: that shape");
        s = zone_plan_context_shape(false, 512, 0, 512, 0);
        CHECK(s.n_ctx == 512 && s.n_ubatch == 512, "a load with no shape named: the planning shape");
        s = zone_plan_context_shape(false, 512, 4096, 512, 0);
        CHECK(s.n_ctx == 4096 && s.n_ubatch == 512, "a context with no micro-batch named: the planning micro-batch");
        s = zone_plan_context_shape(true, 8192, 4096, 1024, 2048);
        CHECK(s.n_ctx == 8192 && s.n_ubatch == 1024, "a context transaction: the context's own shape");
        s = zone_plan_context_shape(false, 512, 256, 512, 2048);
        CHECK(s.n_ubatch == 256, "a micro-batch is never above its context");
        s = zone_plan_context_shape(false, 0, 0, 512, 0);
        CHECK(s.n_ctx == 0 && s.n_ubatch == 512, "no context at all: the micro-batch is not clamped to 0");
        s = zone_plan_context_shape(false, 512, 0, 512, 2048);
        CHECK(s.n_ctx == 512 && s.n_ubatch == 512, "a named micro-batch with no context named: the planning context");

        // Before a plan exists the planning micro-batch is the planner's own choice: the envelope's when it names
        // one, else the model's. The shape published before the zones are sized must be the one the plan records.
        CHECK(ggml_sycl::zone_planning_n_ubatch(8, 512) == 8, "the planning micro-batch: the envelope's when named");
        CHECK(ggml_sycl::zone_planning_n_ubatch(0, 512) == 512, "the planning micro-batch: else the model's");
        using ggml_sycl::zone_plan_context_shape_for_envelope;
        s = zone_plan_context_shape_for_envelope(false, 512, 0, 8, 512, 0);
        CHECK(s.n_ctx == 512 && s.n_ubatch == 8, "an envelope micro-batch with none named: the envelope's wins");
        s = zone_plan_context_shape_for_envelope(false, 512, 0, 0, 512, 0);
        CHECK(s.n_ubatch == 512, "no envelope micro-batch: the model's");
        s = zone_plan_context_shape_for_envelope(false, 512, 4096, 8, 512, 2048);
        CHECK(s.n_ctx == 4096 && s.n_ubatch == 2048, "a named micro-batch wins over the envelope's");
    }

    // ---- Case 24g: the oneDNN Graph floor, bounded by the dispatcher's chunk budget. Each execute is split to half
    // the DIRECT cap, so a full chunk and the last one of a split execute fit under the cap together (unless one query
    // row alone exceeds half of it, when the chunks are one row each): the floor does not hold more.
    {
        const size_t MiB = 1024 * 1024;
        using ggml_sycl::zone_onednn_graph_scratch_floor_bytes;
        // 512 x 512, 16 heads: 16 * 512 * 512 * 4 * 3 / 2 = 24 MiB, under the 64 MiB minimum
        CHECK(zone_onednn_graph_scratch_floor_bytes(16, 0, 0, 512, 512, 1024 * MiB) == 64 * MiB, "the minimum");
        // 2048 x 4096, 16 heads: 768 MiB, under a 1 GiB cap
        CHECK(zone_onednn_graph_scratch_floor_bytes(16, 0, 0, 2048, 4096, 1024 * MiB) == 768 * MiB, "the formula");
        // 2048 x 32768, 32 heads: 12 GiB, bounded by the cap
        CHECK(zone_onednn_graph_scratch_floor_bytes(32, 0, 0, 2048, 32768, 1024 * MiB) == 1024 * MiB, "the cap");
        CHECK(zone_onednn_graph_scratch_floor_bytes(32, 0, 0, 2048, 32768, 0) == size_t(12) * 1024 * MiB,
              "no cap known: the formula");
        // The cap bounds the formula, not the minimum.
        CHECK(zone_onednn_graph_scratch_floor_bytes(32, 0, 0, 2048, 32768, 32 * MiB) == 64 * MiB,
              "a cap below the minimum: the minimum");
        // SWA window: min(n_ctx, n_swa + n_ubatch)
        CHECK(zone_onednn_graph_scratch_floor_bytes(0, 8, 512, 2048, 32768, 1024 * MiB) == 8 * 2560 * 2048 * 6,
              "the window, not the context, for SWA heads");
        CHECK(zone_onednn_graph_scratch_floor_bytes(0, 0, 0, 2048, 4096, 1024 * MiB) == 64 * MiB, "no eligible head");
    }

    // ---- Case 24h: a context's oneDNN Graph scratch above the zone the load planned. The part above the zone's
    // Graph room is served outside the arena and must leave the card its headroom; otherwise the refusal names the
    // largest power-of-two micro-batch whose part fits.
    {
        const size_t MiB = 1024 * 1024;
        using ggml_sycl::zone_onednn_graph_fit_largest_ub;
        using ggml_sycl::zone_onednn_graph_uncovered_bytes;
        CHECK(zone_onednn_graph_uncovered_bytes(768 * MiB, 1024 * MiB) == 0, "inside the zone: nothing uncovered");
        CHECK(zone_onednn_graph_uncovered_bytes(768 * MiB, 512 * MiB) == 256 * MiB, "the part above the room");
        // 16 heads, n_ctx 4096, a 1 GiB cap, 192 MiB of Graph room (planned at ubatch 512), 600 MiB free outside the
        // arena, 256 MiB headroom, no hold spill. 2048: 768 MiB, 576 uncovered, refused; 1024: 384 MiB, 192 uncovered,
        // 408 left: fits.
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 192 * MiB, 600 * MiB, 256 * MiB, 0) ==
                  1024,
              "the refusal at 2048 names 1024");
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 1024, 4096, 1024 * MiB, 192 * MiB, 600 * MiB, 256 * MiB, 0) ==
                  1024,
              "a micro-batch that fits is returned unchanged");
        // 1500: 562.5 MiB, 370.5 uncovered, 229.5 left: refused; the rung below it is 1024, which fits.
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 1500, 4096, 1024 * MiB, 192 * MiB, 600 * MiB, 256 * MiB, 0) ==
                  1024,
              "a micro-batch that is not a power of two names the power of two below it");
        // With no Graph room every rung draws at least the 64 MiB minimum outside the arena.
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 0, 200 * MiB, 256 * MiB, 0) == 0,
              "a card already under the headroom, no room: nothing named");
        // A scratch the room holds draws nothing outside the arena, so the card's free memory and the hold's spill
        // (which the hold check judges on its own) do not refuse it: 768 MiB at 2048 inside 1024 MiB of room.
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 1024 * MiB, 200 * MiB, 256 * MiB, 0) ==
                  2048,
              "a scratch inside the room fits on a card under the headroom");
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 1024 * MiB, 200 * MiB, 256 * MiB,
                                               300 * MiB) == 2048,
              "a scratch inside the room fits whatever the hold's spill");
        // The cap binds: 32 heads at n_ctx 32768 and 512 rows model 3 GiB, bounded by the 1 GiB cap, and the load
        // planned the room at that same bound, so the demand equals the room and nothing is drawn outside the arena.
        CHECK(zone_onednn_graph_fit_largest_ub(32, 0, 0, 512, 32768, 1024 * MiB, 1024 * MiB, 200 * MiB, 256 * MiB, 0) ==
                  512,
              "a demand the cap bounds to the room fits on a card under the headroom");
        // 192 MiB of room on a card under the headroom: 2048 and 1024 draw outside the arena and are refused; 512
        // (192 MiB) lies inside the room and is named.
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 192 * MiB, 200 * MiB, 256 * MiB, 0) ==
                  512,
              "a card under the headroom still gets the largest rung inside the room named");
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 192 * MiB, 600 * MiB, 256 * MiB,
                                               200 * MiB) == 512,
              "the hold's spill is charged beside the uncovered part");
        // 32 at 64 MiB with 300 MiB of spill leaves 236 of 600: no rung fits.
        CHECK(
            zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, 0, 600 * MiB, 256 * MiB, 300 * MiB) == 0,
            "no rung fits: nothing named");
        // The smallest rung and below it, with no room: 32 and 16 draw the 64 MiB minimum outside the arena, which
        // leaves 536 of 600 MiB (fits) and 236 of 300 MiB (refused, and no smaller rung is tried).
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 32, 4096, 1024 * MiB, 0, 600 * MiB, 256 * MiB, 0) == 32,
              "a micro-batch of 32 that fits is returned unchanged");
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 32, 4096, 1024 * MiB, 0, 300 * MiB, 256 * MiB, 0) == 0,
              "a micro-batch of 32 that does not fit: nothing named");
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 16, 4096, 1024 * MiB, 0, 600 * MiB, 256 * MiB, 0) == 16,
              "a micro-batch below 32 that fits is returned unchanged");
        CHECK(zone_onednn_graph_fit_largest_ub(16, 0, 0, 16, 4096, 1024 * MiB, 0, 300 * MiB, 256 * MiB, 0) == 0,
              "a micro-batch below 32 that does not fit: nothing named, no rung above it");
        // A 2^k+1 boundary. 32 heads, n_ctx 32768: 6 MiB a row. 65: 390 MiB, which leaves 250 of 640 (refused); the
        // rung below it is 64 (384 MiB, leaves exactly the 256 MiB headroom).
        CHECK(zone_onednn_graph_fit_largest_ub(32, 0, 0, 65, 32768, 1024 * MiB, 0, 640 * MiB, 256 * MiB, 0) == 64,
              "65 when only 64 fits names 64");
        // A sliding-window shape: 16 window heads, n_swa 1024, n_ctx 32768, no full-context heads. A window head sees
        // n_swa + n_ubatch keys: 2048 gives 16 * 3072 * 2048 f32 * 3/2 = 576 MiB, 384 above the 192 MiB room, which
        // leaves 216 of 600 (refused); 1024 gives 16 * 2048 * 1024 f32 * 3/2 = 192 MiB, inside the room. A window of
        // n_swa alone would put 2048 at 192 MiB (fits), and of n_ctx at the 1 GiB cap for 1024 (refused).
        CHECK(zone_onednn_graph_fit_largest_ub(0, 16, 1024, 2048, 32768, 1024 * MiB, 192 * MiB, 600 * MiB, 256 * MiB,
                                               0) == 1024,
              "a sliding-window shape is sized from its window, n_swa + n_ubatch keys");
    }

    // ---- Case 24i: the -ub a hold-spill refusal names is never below a rung the zone was planned to serve. The B50 /
    // GPT-OSS 20B MXFP4 context at -ub 1024: a planned dense scratch of about 1108 MiB of f16 dequant weight plus
    // 9 KiB per token (1117.1 MiB at 1024, 1112.6 MiB at 512), one 808 MiB compute buffer at 1024 held outside the
    // arena, 141.3 MiB of KV-zone room, 1622.8 MiB free before the rung, and the 404.0 MiB compute term the load
    // measured at 512. 512 runs; the refusal at 1024 named 256. -----------------------------------------------------
    {
        const size_t KiB  = 1024;
        const size_t MiB  = 1024 * KiB;
        auto         plan = [](void *, uint32_t n_ubatch) -> size_t {
            return size_t(11081) * 1024 * 1024 / 10 + size_t(n_ubatch) * 9 * 1024;
        };
        // 808.04 MiB prints as 808.0: a buffer a little over twice the 512 term, as a real one can be
        ggml_sycl::zone_hold_rung_request rungs[] = {
            { 1024, 808 * MiB + 40 * KiB }
        };
        ggml_sycl::zone_hold_fit_inputs in = {};
        in.headroom_target                 = 256 * MiB;
        in.free_before                     = 1622 * MiB + 819 * KiB;
        in.kv_room                         = 141 * MiB + 307 * KiB;
        in.rungs                           = rungs;
        in.n_rungs                         = 1;
        in.plan_of                         = plan;

        // What production fed the fit at the refused rung: the held-out buffer had switched the room off for every
        // rung, so the descent charged 512 its scaled buffer and named 256.
        CHECK(!ggml_sycl::zone_hold_fit(in, 1024), "1024 does not fit: 1783.8 MiB of demand on 1622.8 MiB free");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 1024) == 256, "with no room the descent names 256");

        // The room a load that named no micro-batch planned, at the rung it measured, and the rung whose buffer was
        // held out.
        in.compute_room_n_ubatch          = 512;
        in.compute_room_bytes             = 404 * MiB;
        in.compute_room_measured_n_ubatch = 512;
        in.held_out_n_ubatch              = 1024;
        CHECK(!ggml_sycl::zone_hold_fit(in, 1024), "1024 stays refused: it is above the room's rung");
        CHECK(!ggml_sycl::zone_hold_fit_terms_at(in, 1024).request_credited, "and gets no credit");
        CHECK(ggml_sycl::zone_hold_fit_terms_at(in, 512).request_credited,
              "512 has no record of its own and is at the room's and the measured rung: the zone holds its buffers");
        CHECK(ggml_sycl::zone_hold_fit(in, 512), "so 512 fits: 971.3 MiB of demand leaves 651.5 MiB");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 1024) == 512, "and the refusal names 512, the width that runs");

        // The shape of the B50 gate log: the load placed -ub 1024, so the room's rung is 1024, but its bytes are the
        // 404.0 MiB term measured at 512. 1024 was held out; 512 is at the measured rung.
        in.compute_room_n_ubatch          = 1024;
        in.compute_room_bytes             = 404 * MiB;
        in.compute_room_measured_n_ubatch = 512;
        in.held_out_n_ubatch              = 1024;
        CHECK(!ggml_sycl::zone_hold_fit_terms_at(in, 1024).request_credited, "the gate's 1024 gets no credit");
        CHECK(ggml_sycl::zone_hold_fit_terms_at(in, 512).request_credited, "the gate's 512 is credited");
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 1024) == 512, "and the gate's refusal names 512");

        // A placed 2048 over a term measured at 512, before any rung has a record: a rung without one is credited
        // only at or below the rung the term was measured at, never between it and the room's rung.
        in.n_rungs                        = 0;
        in.compute_room_n_ubatch          = 2048;
        in.compute_room_measured_n_ubatch = 512;
        in.held_out_n_ubatch              = 0;
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 1024, 0, false),
              "1024 is above the measured rung: no credit");
        CHECK(ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 512, 0, false), "512 is at the measured rung");
        CHECK(ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 256, 0, false), "256 is below it");

        // An unknown measured rung gives no rung without a record any credit; a rung's own record is judged as before.
        in.compute_room_n_ubatch          = 512;
        in.compute_room_measured_n_ubatch = 0;
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 512, 0, false), "no measured rung: 512 no credit");
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 256, 0, false), "nor 256");
        CHECK(ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 512, 400 * MiB, true),
              "a record within the room is still credited");
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 512, 405 * MiB, true),
              "a record above the room still is not");
        in.n_rungs = 1;

        // The room moves to the caller's micro-batch once the load measures there: a rung below it is still credited.
        in.compute_room_n_ubatch          = 1024;
        in.compute_room_bytes             = 808 * MiB;
        in.compute_room_measured_n_ubatch = 1024;
        in.held_out_n_ubatch              = 1024;
        CHECK(ggml_sycl::zone_hold_fit_largest_ub(in, 1024) == 512, "a rung below the room's rung keeps the credit");

        // Only the held-out flag decides: a room that holds 1024's own record, at the rung the term was measured at.
        in.compute_room_bytes = 809 * MiB;
        CHECK(!ggml_sycl::zone_hold_fit_terms_at(in, 1024).request_credited,
              "1024's record fits the room, but its own buffer was held out: no credit");
        in.held_out_n_ubatch = 0;
        CHECK(ggml_sycl::zone_hold_fit_terms_at(in, 1024).request_credited, "with no rung held out it is credited");

        // A rung's own record is compared with the room, whole or nothing, as before.
        ggml_sycl::zone_hold_rung_request own[] = {
            { 512, 405 * MiB }
        };
        in.rungs                          = own;
        in.compute_room_n_ubatch          = 512;
        in.compute_room_bytes             = 404 * MiB;
        in.compute_room_measured_n_ubatch = 512;
        CHECK(!ggml_sycl::zone_hold_fit_terms_at(in, 512).request_credited, "a record above the room is not credited");
        // A rung above the room's rung never borrows it.
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 1024, 1, false), "above the room's rung: no credit");
        CHECK(!ggml_sycl::zone_hold_fit_rung_has_compute_room(in, 0, 0, false), "an unknown rung: no credit");
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

    // ---- Case 33: the compute term is the RUNTIME allocator's occupancy of each chunk, summed (llama.cpp-p6i0) ----
    // The scheduler's compute buffer arrives as one allocation per gallocr chunk, so the zone must hold each chunk as
    // the TLSF rounds it, and nothing more: no headroom.
    {
        const size_t grain = ggml_sycl::tlsf_allocator::block_grain;
        // the rule is the allocator's: a 128-aligned request rounds to the 256 B grain
        CHECK(ggml_sycl::tlsf_allocator::round_request(1, 128) == grain, "the grain is not the TLSF's");
        // the Qwen3.8 B70 no -c ub512 buffer: a 1543.1 MB chunk and a 512 MB chunk
        const uint64_t qwen[] = { 1618052864ull, 536870912ull };
        size_t         term   = 0;
        CHECK(ggml_sycl::zone_compute_term_bytes(qwen, 2, 128, &term), "a two-chunk term was refused");
        CHECK(term == ggml_sycl::tlsf_allocator::round_request(1618052864ull, 128) +
                          ggml_sycl::tlsf_allocator::round_request(536870912ull, 128),
              "the term is not the sum of the allocator's per-chunk occupancy");
        // each chunk rounds on its own: two odd chunks take two grains each, not the rounding of their sum
        const uint64_t odd[] = { grain + 1, grain + 1 };
        CHECK(ggml_sycl::zone_compute_term_bytes(odd, 2, 128, &term) && term == 4 * grain,
              "chunks were summed before rounding");
        // an exact multiple of the grain takes exactly itself: no headroom
        const uint64_t exact[] = { 4 * grain };
        CHECK(ggml_sycl::zone_compute_term_bytes(exact, 1, 128, &term) && term == 4 * grain, "headroom was added");
        // a zero chunk allocates nothing; no chunks is a zero term
        const uint64_t zero[] = { 0, grain };
        CHECK(ggml_sycl::zone_compute_term_bytes(zero, 2, 128, &term) && term == grain, "a zero chunk was charged");
        CHECK(ggml_sycl::zone_compute_term_bytes(nullptr, 0, 128, &term) && term == 0, "no chunks is not zero");
        // a wrap refuses, and leaves the answer untouched: never a smaller term
        term                 = 12345;
        const uint64_t big[] = { SIZE_MAX - grain, SIZE_MAX - grain };
        CHECK(!ggml_sycl::zone_compute_term_bytes(big, 2, 128, &term) && term == 12345, "a wrapped sum was accepted");
        const uint64_t top[] = { SIZE_MAX };
        CHECK(!ggml_sycl::zone_compute_term_bytes(top, 1, 128, &term), "a chunk the rounding wraps was accepted");
        CHECK(!ggml_sycl::zone_compute_term_bytes(nullptr, 1, 128, &term), "a null chunk list was accepted");
    }

    // ---- Case 34: each arena zone counts its own bytes over a shared allocator ----------------------------------
    // KV and WEIGHT share one TLSF in a single-chunk arena. A counter that stored the allocator's total gave WEIGHT the
    // KV bytes, and a WEIGHT free stored into KV's counter never lowered WEIGHT: a freed model read as live WEIGHT
    // bytes and refused the next load's zone rebuild. Each counter moves by the allocator's own delta, a free lowers
    // the zone the allocation was recorded under, and the counters of the zones sharing the allocator sum to its
    // used(). The record and the free below have the shape unified_cache::zone_free() has: the zone comes from the
    // exact allocation's record, looked up by offset, and the zone the caller names only finds the record.
    {
        constexpr int             ZONE_KV     = 0;
        constexpr int             ZONE_WEIGHT = 1;
        const size_t              grain       = ggml_sycl::tlsf_allocator::block_grain;
        ggml_sycl::tlsf_allocator shared(1024 * grain);
        std::atomic<size_t>       used[2];
        used[ZONE_KV].store(0);
        used[ZONE_WEIGHT].store(0);
        std::vector<std::pair<size_t, int>> records;  // offset -> the zone the allocation was charged to

        auto alloc_in = [&](int zone, size_t size) {
            const size_t offset = ggml_sycl::zone_used_allocate(shared, used[zone], size, 64);
            if (offset != SIZE_MAX) {
                records.emplace_back(offset, zone);
            }
            return offset;
        };
        // The caller's zone finds nothing here: the record says whose counter the block was added to.
        auto free_named = [&](int caller_zone, size_t offset) {
            (void) caller_zone;
            for (size_t i = 0; i < records.size(); ++i) {
                if (records[i].first == offset) {
                    const int record_zone = records[i].second;
                    records.erase(records.begin() + static_cast<std::ptrdiff_t>(i));
                    ggml_sycl::zone_used_free(shared, offset, used[record_zone]);
                    return true;
                }
            }
            return false;
        };
        auto sum_is_used = [&] {
            return used[ZONE_KV].load() + used[ZONE_WEIGHT].load() == shared.used();
        };

        // a WEIGHT free lowers WEIGHT, and leaves KV holding only KV's bytes
        const size_t w = alloc_in(ZONE_WEIGHT, 64 * grain);
        CHECK(w != SIZE_MAX && used[ZONE_WEIGHT].load() == 64 * grain && used[ZONE_KV].load() == 0,
              "a WEIGHT allocation was not charged to WEIGHT alone");
        const size_t k = alloc_in(ZONE_KV, 16 * grain);
        CHECK(k != SIZE_MAX && used[ZONE_KV].load() == 16 * grain && used[ZONE_WEIGHT].load() == 64 * grain,
              "a KV allocation on the shared allocator took WEIGHT's bytes into KV's counter");
        CHECK(sum_is_used(), "the two counters do not sum to the shared allocator's used()");
        CHECK(free_named(ZONE_WEIGHT, w), "the WEIGHT record was lost");
        CHECK(used[ZONE_WEIGHT].load() == 0, "a WEIGHT free did not lower WEIGHT to zero");
        CHECK(used[ZONE_KV].load() == 16 * grain, "a WEIGHT free changed KV's counter");
        CHECK(sum_is_used(), "after the WEIGHT free the counters do not sum to used()");
        CHECK(free_named(ZONE_KV, k) && used[ZONE_KV].load() == 0 && shared.used() == 0, "the KV free left bytes");

        // the counter takes what the allocator took, block rounding included, not the requested size
        const size_t odd = alloc_in(ZONE_WEIGHT, 100);
        CHECK(odd != SIZE_MAX && used[ZONE_WEIGHT].load() == ggml_sycl::tlsf_allocator::round_request(100, 64) &&
                  used[ZONE_WEIGHT].load() == grain && used[ZONE_WEIGHT].load() == shared.used(),
              "the counter was charged the requested size instead of the allocator's delta");
        const size_t odd_kv = alloc_in(ZONE_KV, grain + 1);
        CHECK(odd_kv != SIZE_MAX && used[ZONE_KV].load() == 2 * grain && sum_is_used(),
              "a rounded KV request was not charged as the allocator rounded it");
        CHECK(free_named(ZONE_WEIGHT, odd) && free_named(ZONE_KV, odd_kv), "the rounded records were lost");
        CHECK(used[ZONE_WEIGHT].load() == 0 && used[ZONE_KV].load() == 0 && shared.used() == 0,
              "a rounded block did not come back off its counter whole");

        // a free named with the other zone of the shared allocator lowers the zone the record names
        const size_t w2 = alloc_in(ZONE_WEIGHT, 8 * grain);
        const size_t k2 = alloc_in(ZONE_KV, 4 * grain);
        CHECK(w2 != SIZE_MAX && k2 != SIZE_MAX, "the wrong-zone fixture did not allocate");
        CHECK(free_named(ZONE_KV, w2), "the WEIGHT record named as KV was not found");
        CHECK(used[ZONE_WEIGHT].load() == 0 && used[ZONE_KV].load() == 4 * grain,
              "a WEIGHT block freed under the KV name lowered KV instead of WEIGHT");
        CHECK(free_named(ZONE_WEIGHT, k2), "the KV record named as WEIGHT was not found");
        CHECK(used[ZONE_KV].load() == 0 && used[ZONE_WEIGHT].load() == 0 && shared.used() == 0,
              "a KV block freed under the WEIGHT name lowered WEIGHT instead of KV");

        // a free the allocator does not recognise, and an allocation it refuses, move no counter
        const size_t w3 = alloc_in(ZONE_WEIGHT, 2 * grain);
        CHECK(w3 != SIZE_MAX && free_named(ZONE_WEIGHT, w3), "the double-free fixture did not allocate and free");
        ggml_sycl::zone_used_free(shared, w3, used[ZONE_WEIGHT]);
        CHECK(used[ZONE_WEIGHT].load() == 0 && shared.used() == 0, "a second free of a block moved the counter");
        CHECK(ggml_sycl::zone_used_allocate(shared, used[ZONE_KV], 2048 * grain, 64) == SIZE_MAX,
              "an allocation larger than the allocator succeeded");
        CHECK(used[ZONE_KV].load() == 0, "a refused allocation was charged");

        // interleaved allocations and frees of both zones: the sum holds after every step, and both counters end at 0
        std::vector<size_t> live;
        const size_t        sizes[] = { 3 * grain, 100, 7 * grain + 5, grain, 33 * grain, 2 * grain - 1 };
        for (int round = 0; round < 3; ++round) {
            for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
                const int    zone   = (i + static_cast<size_t>(round)) % 2 == 0 ? ZONE_WEIGHT : ZONE_KV;
                const size_t offset = alloc_in(zone, sizes[i]);
                CHECK(offset != SIZE_MAX, "an interleaved allocation failed");
                live.push_back(offset);
                CHECK(sum_is_used(), "an interleaved allocation broke the sum");
                if (live.size() % 3 == 0) {
                    // free the middle one, named by the other zone than the one it was charged to
                    const size_t victim = live[live.size() / 2];
                    int          named  = ZONE_KV;
                    for (const auto & r : records) {
                        if (r.first == victim) {
                            named = r.second == ZONE_KV ? ZONE_WEIGHT : ZONE_KV;
                        }
                    }
                    CHECK(free_named(named, victim), "an interleaved record was lost");
                    live.erase(live.begin() + static_cast<std::ptrdiff_t>(live.size() / 2));
                    CHECK(sum_is_used(), "an interleaved free broke the sum");
                }
            }
        }
        while (!live.empty()) {
            CHECK(free_named(ZONE_WEIGHT, live.back()), "a remaining record was lost");
            live.pop_back();
            CHECK(sum_is_used(), "a closing free broke the sum");
        }
        CHECK(used[ZONE_KV].load() == 0 && used[ZONE_WEIGHT].load() == 0 && shared.used() == 0 && records.empty(),
              "interleaved allocations and frees did not return both counters to zero");
    }

    // ---- Case 35: a raw device allocation is gated on the device's physical bytes, not on zone counters ----------
    // A raw allocation takes device memory the arena did not reserve. The arena reservation is committed whole, so what
    // the device has left for a raw request is its total less the arena, less the raw rows already held outside it,
    // less what was in use outside this process's accounting when the first cache was created. A gate that read zone
    // counters instead let a raw request past an arena whose zones were empty but whose bytes were all reserved.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_raw_gate_fits;
        using ggml_sycl::zone_raw_gate_free_outside_arena;
        using ggml_sycl::zone_raw_gate_inputs;

        // the B50 GPT-OSS load: 16304 MiB device, 14292 MiB arena, 381 MiB in use before the first cache
        zone_raw_gate_inputs b50;
        b50.device_total       = 16304 * MiB;
        b50.arena_bytes        = 14292 * MiB;
        b50.raw_held           = 0;
        b50.outside_accounting = 381 * MiB;
        const size_t b50_free  = 16304 * MiB - 14292 * MiB - 381 * MiB;
        CHECK(zone_raw_gate_free_outside_arena(b50) == b50_free, "the free figure is not total - arena - outside");
        CHECK(zone_raw_gate_fits(b50, 5 * MiB), "a small raw request next to the B50 arena was refused");
        CHECK(zone_raw_gate_fits(b50, b50_free), "a request of exactly the free bytes was refused");
        CHECK(!zone_raw_gate_fits(b50, b50_free + 1), "a request one byte past the free bytes was admitted");

        // the bytes outside this process's accounting are part of the figure: the same request flips with the term
        zone_raw_gate_inputs no_outside = b50;
        no_outside.outside_accounting   = 0;
        CHECK(zone_raw_gate_fits(no_outside, b50_free + 200 * MiB), "without the term the larger request must fit");
        CHECK(!zone_raw_gate_fits(b50, b50_free + 200 * MiB), "the term outside the accounting was not counted");

        // raw rows already held outside the arena lower what is left
        zone_raw_gate_inputs held = b50;
        held.raw_held             = 1000 * MiB;
        CHECK(zone_raw_gate_free_outside_arena(held) == b50_free - 1000 * MiB, "the raw rows held were not counted");
        CHECK(zone_raw_gate_fits(b50, 700 * MiB) && !zone_raw_gate_fits(held, 700 * MiB),
              "a request that fit before 1000 MiB of raw rows still fits after them");

        // an arena of two chunks reserves both: the figure takes their sum, and a request that clears one is refused
        const size_t         chunks[] = { 16384 * MiB, 13468 * MiB };
        zone_raw_gate_inputs two_chunk;
        two_chunk.device_total = 31104 * MiB;
        two_chunk.arena_bytes  = chunks[0] + chunks[1];
        CHECK(zone_raw_gate_free_outside_arena(two_chunk) == 1252 * MiB, "the two-chunk arena was not counted whole");
        CHECK(!zone_raw_gate_fits(two_chunk, 2000 * MiB), "a request that clears only the first chunk was admitted");

        // saturating: an arena larger than the device, or held bytes that would wrap, leave nothing, never a wrap
        zone_raw_gate_inputs over;
        over.device_total = 1000 * MiB;
        over.arena_bytes  = 2000 * MiB;
        CHECK(zone_raw_gate_free_outside_arena(over) == 0 && !zone_raw_gate_fits(over, 1),
              "an arena larger than the device wrapped the figure");
        zone_raw_gate_inputs wrap = b50;
        wrap.raw_held             = SIZE_MAX - 10 * MiB;
        CHECK(zone_raw_gate_free_outside_arena(wrap) == 0 && !zone_raw_gate_fits(wrap, 1),
              "the raw rows held wrapped the sum");
        wrap.raw_held           = 0;
        wrap.outside_accounting = SIZE_MAX;
        CHECK(zone_raw_gate_free_outside_arena(wrap) == 0, "the term outside the accounting wrapped the sum");

        // an unknown device total gates nothing
        zone_raw_gate_inputs unknown;
        unknown.arena_bytes        = 14292 * MiB;
        unknown.outside_accounting = 381 * MiB;
        CHECK(zone_raw_gate_free_outside_arena(unknown) == SIZE_MAX && zone_raw_gate_fits(unknown, SIZE_MAX),
              "an unknown device total gated a request");
    }

    // ---- Case 36: the arena is rebuilt only when nothing points into it ------------------------------------------
    // A rebuild destroys the arena and reserves it again, so every byte in it moves. A zone byte still allocated, a
    // lease on one of its chunks, or a cache scratch buffer whose pointer lies inside it each refuse the rebuild. A
    // scratch buffer allocated outside the arena keeps its own memory through a rebuild, so it must not refuse one: a
    // gate that read any non-null scratch pointer refused rebuilds the arena could take.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_arena_holders;
        using ggml_sycl::zone_arena_rebuild_allowed;

        // the second load of one process: 269.7 MB live in the zones, 49 chunk leases, the PP MoE ring's three slots
        // and the oneDNN pair inside the arena, the reorder temp outside it
        zone_arena_holders second_load;
        second_load.live_zone_bytes  = static_cast<size_t>(269.7 * MiB);
        second_load.chunk_leases     = 49;
        second_load.scratch_in_arena = 5;
        second_load.scratch_outside  = 1;
        CHECK(!zone_arena_rebuild_allowed(second_load), "the second load's holders allowed a rebuild");

        // each holder refuses on its own, down to one byte or one lease
        zone_arena_holders leases_only;
        leases_only.chunk_leases = 49;
        CHECK(!zone_arena_rebuild_allowed(leases_only), "49 chunk leases alone allowed a rebuild");
        leases_only.chunk_leases = 1;
        CHECK(!zone_arena_rebuild_allowed(leases_only), "one chunk lease allowed a rebuild");
        zone_arena_holders bytes_only;
        bytes_only.live_zone_bytes = 1;
        CHECK(!zone_arena_rebuild_allowed(bytes_only), "one live zone byte allowed a rebuild");
        zone_arena_holders inside_only;
        inside_only.scratch_in_arena = 1;
        CHECK(!zone_arena_rebuild_allowed(inside_only), "a scratch buffer inside the arena allowed a rebuild");

        // nothing points into the arena: the rebuild goes ahead, whatever is allocated outside it
        zone_arena_holders nothing;
        CHECK(zone_arena_rebuild_allowed(nothing), "an arena nothing points into was refused");
        zone_arena_holders outside_only;
        outside_only.scratch_outside = 7;
        CHECK(zone_arena_rebuild_allowed(outside_only), "scratch allocated outside the arena refused a rebuild");

        // scratch outside the arena does not hide a holder inside it
        zone_arena_holders outside_and_lease = outside_only;
        outside_and_lease.chunk_leases       = 1;
        CHECK(!zone_arena_rebuild_allowed(outside_and_lease), "a lease beside outside scratch allowed a rebuild");
    }

    // ---- Case 37: the RUNTIME requirement is the checked sum of its terms -----------------------------------------
    // zone_runtime_requirement, the sum the cache's getter returns, on the figures of the run where a fallback asked
    // for 92160 bytes more than the zone held: it adds every term, a sum that would wrap is unanswerable, and so is a
    // call with nowhere to write. That the fallback puts the discarded plan's terms back is not checked here:
    // test-sycl-runtime-alloc's planner-terms case checks the restore helper against the cache's own state.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_runtime_requirement;
        using ggml_sycl::zone_runtime_terms;

        // The planning shape (n_ubatch 512): the ring's RUNTIME part (weight 134.5 MB, activation 33.6 MB, output
        // 67.1 MB at depth 1; no PP pipeline scratch and no CONTROL pool), mmq_src1 2.3 MB, dequant_f16 1108.6 MB,
        // compute 404 MB, state 0 and no host scatter: 1750.1 MB.
        zone_runtime_terms planning;
        planning.base        = 141008896 + 35232256 + 70359040;
        planning.mmq_src1    = 2411520;
        planning.dequant_f16 = 1162444800;
        planning.compute     = 404 * MiB;
        size_t before        = 0;
        CHECK(zone_runtime_requirement(planning, &before), "the planning shape's requirement is unanswerable");
        CHECK(before == 1835081216, "the planning shape's requirement is not the sum of its terms");

        // The micro-batch (n_ubatch 2048) grows the ring and the dense scratch, with a 90 KB host scatter term:
        // 1971.4 MB.
        zone_runtime_terms grown_terms = planning;
        grown_terms.moe_host_scatter   = 92160;
        grown_terms.base               = 458960896;
        grown_terms.mmq_src1           = 9 * MiB;
        grown_terms.dequant_f16        = 1175027712;
        size_t grown                   = 0;
        CHECK(zone_runtime_requirement(grown_terms, &grown) && grown == 2067142656,
              "the micro-batch's requirement is not the sum of its terms");

        // The planning shape with the host scatter term published: exactly that term more, 1750.2 MB, the figure the
        // run's refused second attempt asked for.
        zone_runtime_terms with_scatter = planning;
        with_scatter.moe_host_scatter   = 92160;
        size_t scattered                = 0;
        CHECK(zone_runtime_requirement(with_scatter, &scattered) && scattered == before + 92160,
              "the host scatter term does not add exactly its 92160 bytes to the planning shape's requirement");

        // Every term counts, and a sum that would wrap is unanswerable: a wrapped sum would size a smaller zone.
        zone_runtime_terms probe   = planning;
        size_t * const     terms[] = { &probe.base,        &probe.moe_host_scatter, &probe.mmq_src1,
                                       &probe.dequant_f16, &probe.compute,          &probe.state };
        for (size_t * term : terms) {
            const size_t kept = *term;
            size_t       sum  = 0;
            *term += 1;
            CHECK(zone_runtime_requirement(probe, &sum) && sum == before + 1, "a RUNTIME term is not counted");
            *term = SIZE_MAX;
            CHECK(!zone_runtime_requirement(probe, &sum), "a RUNTIME requirement that wraps was answered");
            *term = kept;
        }
        CHECK(!zone_runtime_requirement(probe, nullptr), "a requirement with nowhere to write it was answered");
    }

    // ---- Case 38: a load refused for zones it cannot size says where, how much and why ----------------------------
    // The line names the device, each zone's need and have, the shortfall of every short zone and only theirs, the
    // reason, and what to free. The zones are planned at the load's fixed shape, so it offers no smaller -c or -ub.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_arena_refusal;
        using ggml_sycl::zone_arena_refusal_reason;
        using ggml_sycl::zone_arena_refusal_text;

        const auto has = [](const std::string & text, const char * part) {
            return text.find(part) != std::string::npos;
        };

        zone_arena_refusal held;
        held.reason                  = zone_arena_refusal_reason::REBUILD_HELD;
        held.device                  = 1;
        held.need                    = { 512 * MiB, 300 * MiB, 1985 * MiB };
        held.have                    = { 512 * MiB, 256 * MiB, 1792 * MiB };
        held.holders.live_zone_bytes = 4096;
        const std::string hl         = zone_arena_refusal_text(held);
        CHECK(hl.rfind("[SYCL-PLAN] model load refused on device 1: ", 0) == 0, "the refusal does not name its device");
        CHECK(has(hl, "SCRATCH need 512.0 MB have 512.0 MB, ONEDNN"), "a zone that fits is not named, or is short");
        CHECK(has(hl, "ONEDNN need 300.0 MB have 256.0 MB (44.0 MB short)"), "the ONEDNN need, have or shortfall");
        CHECK(has(hl, "RUNTIME need 1985.0 MB have 1792.0 MB (193.0 MB short)"), "the RUNTIME need, have or shortfall");
        CHECK(!has(hl, "-ub") && !has(hl, "-c ") && !has(hl, "ubatch"), "the refusal offers a smaller -c or -ub");
        CHECK(!hl.empty() && hl.back() != '\n', "the refusal text carries its own newline");

        // The remedy follows what holds the arena. Live allocations in its zones belong to a model or context, which
        // freeing releases; a byte count under 0.1 MB prints in KB, or in bytes under 1 KB, so it never reads 0.0 MB.
        CHECK(has(hl,
                  "; 4.0 KB of live allocations in its zones hold the arena (the [VRAM-ARENA] lines above name "
                  "them), so its zones cannot grow. Free the model or context they belong to, then load this one "
                  "again"),
              "a held arena's live allocations or their remedy are not named");
        zone_arena_refusal with_lease      = held;
        with_lease.holders.live_zone_bytes = 512;
        with_lease.holders.chunk_leases    = 1;
        with_lease.holders.scratch_outside = 3;
        const std::string wl               = zone_arena_refusal_text(with_lease);
        CHECK(has(wl, "; 512 bytes of live allocations in its zones and 1 lease on its chunks hold the arena") &&
                  has(wl, "Free the model or context they belong to"),
              "live allocations with a lease beside them are not named with the free-the-model remedy");
        CHECK(!has(wl, "outside") && !has(wl, "scratch"), "scratch allocated outside the arena is named as a holder");
        // Leases with nothing live in the zones outlived what they point at: no free releases them.
        zone_arena_refusal leases_only   = held;
        leases_only.holders              = {};
        leases_only.holders.chunk_leases = 48;
        const std::string ll             = zone_arena_refusal_text(leases_only);
        CHECK(has(ll,
                  "; 48 leases on its chunks hold the arena (the [VRAM-ARENA] lines above name them), so its "
                  "zones cannot grow. No allocation is live in its zones, so the leases outlived what they point "
                  "at and are held until the process exits. Load this model in a new process before any other on "
                  "device 1"),
              "leases with nothing live in the zones are not named with the new-process remedy");
        CHECK(!has(ll, "Free the model"), "leases with nothing live in the zones are blamed on a model to free");
        // The cache's own scratch buffers inside the arena stay until the process exits, whatever else holds it.
        zone_arena_refusal scratch_in       = held;
        scratch_in.holders.live_zone_bytes  = static_cast<size_t>(269.7 * MiB);
        scratch_in.holders.chunk_leases     = 49;
        scratch_in.holders.scratch_in_arena = 5;
        scratch_in.holders.scratch_outside  = 1;
        const std::string sl                = zone_arena_refusal_text(scratch_in);
        CHECK(has(sl,
                  "; 269.7 MB of live allocations in its zones, 49 leases on its chunks and 5 of the cache's own "
                  "scratch buffers hold the arena (the [VRAM-ARENA] lines above name them), so its zones cannot "
                  "grow. The cache keeps its own scratch buffers in the arena until the process exits, so freeing "
                  "a model or context does not let its zones grow. Load this model in a new process before any "
                  "other on device 1"),
              "the cache's scratch buffers inside the arena are not named with the new-process remedy");
        CHECK(!has(sl, "Free the model"), "the cache's own scratch buffers are blamed on a model to free");
        // Live bytes are a model's or a context's only when the cache has no release of its own still to run: a free
        // queued in the arena, a release waiting for a retry, a retired entry not yet finalized and an eviction in
        // flight all keep zone bytes with no owner left. With any of them the remedy says the bytes may be those, and
        // asks for the load again before it blames a model or context.
        zone_arena_refusal pending      = held;
        pending.holders.queued_frees    = 2;
        pending.holders.release_retries = 1;
        const std::string pl            = zone_arena_refusal_text(pending);
        CHECK(has(pl,
                  "; 4.0 KB of live allocations in its zones hold the arena (the [VRAM-ARENA] lines above name "
                  "them), so its zones cannot grow. Some of those bytes may be releases the cache has not finished "
                  "(2 queued frees and 1 release waiting for a retry), which it finishes on its own. Load this one "
                  "again; if it is refused with the same bytes, free the model or context they belong to"),
              "live bytes beside the cache's unfinished releases are not named with the load-again remedy");
        CHECK(!has(pl, ". Free the model or context they belong to, then load this one again"),
              "live bytes beside the cache's unfinished releases are blamed on a model to free");
        zone_arena_refusal retiring      = held;
        retiring.holders.retired_pending = 1;
        retiring.holders.evictions       = 3;
        CHECK(has(zone_arena_refusal_text(retiring), "(1 retired entry and 3 evictions in flight)"),
              "retired entries and evictions in flight are not named as unfinished releases");
        // Beside the ring the same holds for the bytes live beside it.
        zone_arena_refusal ring_and_free       = held;
        ring_and_free.holders.live_zone_bytes  = 3 * MiB;
        ring_and_free.holders.scratch_in_arena = 3;
        ring_and_free.holders.ring_in_arena    = 3;
        ring_and_free.holders.live_beside_ring = 1;
        ring_and_free.holders.queued_frees     = 1;
        const std::string rf                   = zone_arena_refusal_text(ring_and_free);
        CHECK(has(rf,
                  "Some of those bytes may be releases the cache has not finished (1 queued free), which it "
                  "finishes on its own"),
              "a queued free beside the ring is not named as an unfinished release");

        // A shortfall under 0.1 MB would print as 0.0 MB short beside a need and a have that print the same: the short
        // zone's need, have and shortfall are printed in KB then, and in bytes under 1 KB, so the line still shows
        // which zone is short and by how much. The figures of the run whose control load was refused 8 KiB short of
        // RUNTIME.
        zone_arena_refusal kib_short = held;
        kib_short.need               = { 512 * MiB, 256 * MiB, 1792 * MiB + 8192 };
        kib_short.have               = { 512 * MiB, 256 * MiB, 1792 * MiB };
        const std::string kl         = zone_arena_refusal_text(kib_short);
        CHECK(has(kl, "RUNTIME need 1835016.0 KB have 1835008.0 KB (8.0 KB short)"),
              "an 8 KiB RUNTIME shortfall is not printed in KB with the zone's need and have");
        CHECK(has(kl, "SCRATCH need 512.0 MB have 512.0 MB, ONEDNN need 256.0 MB have 256.0 MB, RUNTIME"),
              "a zone that fits is not printed in MB beside a short one in KB");
        CHECK(!has(kl, "MB short"), "an 8 KiB shortfall is printed in MB");
        zone_arena_refusal bytes_short = kib_short;
        bytes_short.need.runtime       = 1792 * MiB + 512;
        CHECK(has(zone_arena_refusal_text(bytes_short),
                  "RUNTIME need 1879048704 bytes have 1879048192 bytes (512 bytes short)"),
              "a 512-byte RUNTIME shortfall is not printed in bytes with the zone's need and have");
        // 0.1 MB is 104857.6 bytes: a shortfall one byte under it is printed in KB, one byte over it in MB.
        zone_arena_refusal edge = kib_short;
        edge.need.runtime       = 1792 * MiB + 104857;
        CHECK(has(zone_arena_refusal_text(edge), "(102.4 KB short)"), "a shortfall just under 0.1 MB is not in KB");
        edge.need.runtime = 1792 * MiB + 104858;
        CHECK(has(zone_arena_refusal_text(edge), "RUNTIME need 1792.1 MB have 1792.0 MB (0.1 MB short)"),
              "a shortfall of 0.1 MB is not printed in MB");

        // A reserve that failed after a rebuild destroyed the arena: the device has no zones now, which the have says,
        // and the arena it had is named apart as released for the rebuild.
        zone_arena_refusal reserve = held;
        reserve.reason             = zone_arena_refusal_reason::RESERVE_FAILED;
        reserve.have               = {};
        reserve.released           = held.have;
        const std::string rl       = zone_arena_refusal_text(reserve);
        CHECK(has(rl, "RUNTIME need 1985.0 MB have 0.0 MB (1985.0 MB short)"), "a reserve failure's need or have");
        CHECK(has(rl,
                  "; the device could not reserve an arena with these zones, and the arena it had (SCRATCH "
                  "512.0 MB, ONEDNN 256.0 MB, RUNTIME 1792.0 MB) was released for the rebuild. Free VRAM on "
                  "device 1, then load this one again"),
              "a reserve failure after a rebuild does not say the arena it had was released, or its remedy");
        CHECK(!has(rl, "hold the arena"), "a reserve failure blames holders");
        // The first reserve on a device released nothing.
        zone_arena_refusal first_reserve = reserve;
        first_reserve.released           = {};
        const std::string fl             = zone_arena_refusal_text(first_reserve);
        CHECK(has(fl,
                  "; the device could not reserve an arena with these zones. Free VRAM on device 1, then load "
                  "this one again") &&
                  !has(fl, "released"),
              "a first reserve's failure names an arena released for a rebuild");

        zone_arena_refusal no_cache;
        no_cache.reason      = zone_arena_refusal_reason::NO_CACHE;
        no_cache.device      = 0;
        const std::string nl = zone_arena_refusal_text(no_cache);
        CHECK(has(nl, "refused on device 0: ") && has(nl, "no unified cache"), "a missing cache is not named");

        CHECK(zone_arena_refusal_text(zone_arena_refusal{}).empty(), "a refusal with no reason printed a line");
    }

    // ---- Case 39: a load refused because the arena could not be torn down says why, and whether it is still usable
    // A rebuild tears the arena down before it reserves the larger one. A teardown that refuses before it has freed any
    // chunk puts the arena back as it was before the attempt, so the load may be tried again, and the line names what
    // to do first. A teardown that refuses after it freed a chunk cannot put it back: the arena is partly freed and
    // stays closed to new allocations, so no reserve may follow it and no later load in the process can use the
    // device's arena. The line says so and sends the load to a new process.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_arena_refusal;
        using ggml_sycl::zone_arena_refusal_reason;
        using ggml_sycl::zone_arena_refusal_text;
        using ggml_sycl::zone_arena_teardown_left_closed;
        using ggml_sycl::zone_arena_teardown_refusal;
        using ggml_sycl::zone_arena_unfreed_chunk_refusal;

        const auto has = [](const std::string & text, const char * part) {
            return text.find(part) != std::string::npos;
        };

        CHECK(!zone_arena_teardown_left_closed(zone_arena_teardown_refusal::NONE), "no refusal left the arena closed");
        for (const zone_arena_teardown_refusal restored :
             { zone_arena_teardown_refusal::HOST_STATE, zone_arena_teardown_refusal::LIVE_ALLOCATIONS,
               zone_arena_teardown_refusal::DRAIN_TIMEOUT, zone_arena_teardown_refusal::LEASES_OUTSTANDING,
               zone_arena_teardown_refusal::FIRST_FREE_FAILED }) {
            CHECK(!zone_arena_teardown_left_closed(restored),
                  "a teardown refused before it freed any chunk left the arena closed");
        }
        for (const zone_arena_teardown_refusal partly :
             { zone_arena_teardown_refusal::FREE_FAILED, zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE }) {
            CHECK(zone_arena_teardown_left_closed(partly),
                  "a teardown refused after it freed a chunk left the arena open");
        }

        // A chunk the teardown could not free names a restorable cause only while no chunk was freed before it, and
        // says whether a lease kept it or the device refused it.
        CHECK(zone_arena_unfreed_chunk_refusal(true, false) == zone_arena_teardown_refusal::LEASES_OUTSTANDING &&
                  zone_arena_unfreed_chunk_refusal(false, false) == zone_arena_teardown_refusal::FIRST_FREE_FAILED,
              "a chunk the teardown could not free before any was freed is given the wrong cause");
        CHECK(zone_arena_unfreed_chunk_refusal(true, true) ==
                      zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE &&
                  zone_arena_unfreed_chunk_refusal(false, true) == zone_arena_teardown_refusal::FREE_FAILED,
              "a chunk the teardown could not free after one was freed is given the wrong cause");
        for (const bool leased : { false, true }) {
            CHECK(!zone_arena_teardown_left_closed(zone_arena_unfreed_chunk_refusal(leased, false)) &&
                      zone_arena_teardown_left_closed(zone_arena_unfreed_chunk_refusal(leased, true)),
                  "whether a chunk was freed before the failed one does not decide whether the arena stays closed");
        }

        zone_arena_refusal torn;
        torn.reason          = zone_arena_refusal_reason::TEARDOWN_REFUSED;
        torn.device          = 1;
        torn.need            = { 512 * MiB, 256 * MiB, 1985 * MiB };
        torn.have            = { 512 * MiB, 256 * MiB, 1792 * MiB };
        torn.teardown        = zone_arena_teardown_refusal::FREE_FAILED;
        const std::string fl = zone_arena_refusal_text(torn);
        CHECK(fl.rfind("[SYCL-PLAN] model load refused on device 1: ", 0) == 0, "the refusal does not name its device");
        CHECK(has(fl, "RUNTIME need 1985.0 MB have 1792.0 MB (193.0 MB short)"), "the RUNTIME need, have or shortfall");
        CHECK(has(fl,
                  "; the arena could not be torn down: the device refused to free one of its chunks after the chunks "
                  "before it were freed, so the arena is partly freed. The arena stays closed to new allocations, so "
                  "no model can load on device 1 in this process. Load this model in a new process"),
              "a partly freed arena is not named with the failed free, the closed arena and the new-process remedy");
        CHECK(!has(fl, "hold the arena") && !has(fl, "Free VRAM") && !has(fl, "load this one again") &&
                  !has(fl, "as it was"),
              "a partly freed arena is given another reason's remedy, or described as intact");

        const auto text_for = [&](zone_arena_teardown_refusal cause) {
            zone_arena_refusal refused = torn;
            refused.teardown           = cause;
            return zone_arena_refusal_text(refused);
        };
        const std::string hl = text_for(zone_arena_teardown_refusal::HOST_STATE);
        const std::string ll = text_for(zone_arena_teardown_refusal::LIVE_ALLOCATIONS);
        const std::string tl = text_for(zone_arena_teardown_refusal::DRAIN_TIMEOUT);
        CHECK(has(hl,
                  "; the arena could not be torn down: the host could not reserve the teardown's own bookkeeping, "
                  "and the arena is as it was. Free host memory, then load this one again"),
              "a teardown the host could not keep the books of is not named with the arena intact and its remedy");
        CHECK(has(ll,
                  "; the arena could not be torn down: allocations were still live in it, and the arena is as it "
                  "was. Free the models and contexts on device 1 that hold them, then load this one again"),
              "live allocations at the teardown are not named with the arena intact and their remedy");
        CHECK(has(tl,
                  "; the arena could not be torn down: a lease on one of its allocations did not end while the "
                  "teardown waited, and the arena is as it was. Load this one again once the work on device 1 has "
                  "finished"),
              "a drain that timed out is not named with the arena intact and its remedy");
        const std::string ol = text_for(zone_arena_teardown_refusal::LEASES_OUTSTANDING);
        const std::string fx = text_for(zone_arena_teardown_refusal::FIRST_FREE_FAILED);
        CHECK(has(ol,
                  "; the arena could not be torn down: a lease on the first chunk the teardown tried outlived its "
                  "wait, so none was freed and the arena is as it was. Free what holds a pointer into it on device 1, "
                  "then load this one again"),
              "a chunk lease that outlived the wait before any chunk was freed is not named with the arena intact");
        CHECK(has(fx,
                  "; the arena could not be torn down: the device refused to free the first chunk the teardown tried, "
                  "so none was freed and the arena is as it was. Load this one again, and load it in a new process if "
                  "the device refuses again"),
              "a refused first free is not named with the arena intact and its remedy");
        for (const std::string * intact : { &hl, &ll, &tl, &ol, &fx }) {
            CHECK(!has(*intact, "stays closed") && !has(*intact, "partly freed"),
                  "an arena a refused teardown put back is described as closed");
        }
        for (const std::string * intact : { &hl, &ll, &tl, &ol }) {
            CHECK(!has(*intact, "new process"), "an arena a refused teardown put back is sent to a new process");
        }

        // A lease that outlived the wait after a chunk was freed leaves the arena partly freed, like a refused free.
        const std::string pl = text_for(zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE);
        CHECK(has(pl,
                  "; the arena could not be torn down: a lease on one of its chunks outlived the teardown's wait after "
                  "the chunks before it were freed, so the arena is partly freed. The arena stays closed to new "
                  "allocations, so no model can load on device 1 in this process. Load this model in a new process"),
              "a chunk lease that outlived the wait after a chunk was freed is not named as a partly freed arena");
        CHECK(!has(pl, "as it was") && !has(pl, "load this one again"),
              "a partly freed arena is described as intact or given an intact arena's remedy");

        // A refusal that names no cause cannot say the arena is intact, so it is read as closed.
        zone_arena_refusal no_cause = torn;
        no_cause.teardown           = zone_arena_teardown_refusal::NONE;
        const std::string nl        = zone_arena_refusal_text(no_cause);
        CHECK(has(nl, "; the arena could not be torn down. The arena stays closed") &&
                  has(nl, "Load this model in a new process"),
              "a refused teardown with no cause is not read as closed");

        // The clause and the predicate agree for every cause: the clause calls the arena closed for exactly the causes
        // the predicate reads as closed, and for NONE, a refusal that lost its cause, and calls it as it was for every
        // other. FREE_FAILED is the last enumerator, so the loop covers them all.
        for (int v = static_cast<int>(zone_arena_teardown_refusal::NONE);
             v <= static_cast<int>(zone_arena_teardown_refusal::FREE_FAILED); ++v) {
            const auto        cause = static_cast<zone_arena_teardown_refusal>(v);
            const std::string text  = text_for(cause);
            const bool closed = cause == zone_arena_teardown_refusal::NONE || zone_arena_teardown_left_closed(cause);
            CHECK(has(text, "The arena stays closed to new allocations") == closed && has(text, "as it was") == !closed,
                  "a teardown clause and zone_arena_teardown_left_closed disagree about whether the arena is closed");
        }
    }

    // ---- Case 40: a load releases the PP MoE ring only when the ring alone keeps the arena from a rebuild it needs
    // The ring is the cache's, not a model's: it outlives the context that sized it. A load may release it to rebuild
    // the arena, but only as the last resort, when a rebuild is needed and the ring's slot buffers are the only
    // holders. Any other holder refuses the rebuild whatever happens to the ring, so the load must not release a ring a
    // live context may be using. A scratch buffer outside the arena holds nothing, so it does not stop the release.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_arena_holders;
        using ggml_sycl::zone_arena_only_ring_holds;
        using ggml_sycl::zone_arena_refusal;
        using ggml_sycl::zone_arena_refusal_reason;
        using ggml_sycl::zone_arena_refusal_text;

        const auto has = [](const std::string & text, const char * part) {
            return text.find(part) != std::string::npos;
        };

        // a depth-2 ring left by a freed context: six slot buffers, 96 MB in the zones, nothing else in the arena
        zone_arena_holders ring_only;
        ring_only.live_zone_bytes  = 96 * MiB;
        ring_only.scratch_in_arena = 6;
        ring_only.ring_in_arena    = 6;
        CHECK(zone_arena_only_ring_holds(true, ring_only), "a ring alone in an arena that must grow was not released");
        CHECK(!zone_arena_only_ring_holds(false, ring_only), "a ring was released for an arena that need not grow");
        zone_arena_holders ring_outside = ring_only;
        ring_outside.scratch_outside    = 2;
        CHECK(zone_arena_only_ring_holds(true, ring_outside), "scratch outside the arena kept a lone ring");

        // each other holder keeps the ring, down to one allocation, one lease or one other scratch buffer
        zone_arena_holders ring_and_live = ring_only;
        ring_and_live.live_beside_ring   = 1;
        CHECK(!zone_arena_only_ring_holds(true, ring_and_live), "a ring beside a live allocation was released");
        zone_arena_holders ring_and_lease = ring_only;
        ring_and_lease.chunk_leases       = 1;
        CHECK(!zone_arena_only_ring_holds(true, ring_and_lease), "a ring beside a chunk lease was released");
        zone_arena_holders ring_and_scratch = ring_only;
        ring_and_scratch.scratch_in_arena   = 7;
        CHECK(!zone_arena_only_ring_holds(true, ring_and_scratch), "a ring beside other cache scratch was released");
        // no ring inside the arena: nothing to release
        CHECK(!zone_arena_only_ring_holds(true, zone_arena_holders{}), "an arena with no ring had a ring to release");
        zone_arena_holders ring_outside_only;
        ring_outside_only.scratch_outside = 6;
        CHECK(!zone_arena_only_ring_holds(true, ring_outside_only), "a ring outside the arena was released for it");

        // The refusal a load logs when the release kept the ring (a prefill's claim on one of its slots is held): the
        // ring is the cache's, so no model or context frees it, and it is released on a later load, not only at
        // process exit. The claim outlives its prefill, so waiting for the prefill to finish lets nothing go; freeing
        // the context that ran it drains the device's slots.
        zone_arena_refusal held;
        held.reason          = zone_arena_refusal_reason::REBUILD_HELD;
        held.device          = 1;
        held.need            = { 512 * MiB, 256 * MiB, 1985 * MiB };
        held.have            = { 512 * MiB, 256 * MiB, 1792 * MiB };
        held.holders         = ring_only;
        const std::string rl = zone_arena_refusal_text(held);
        CHECK(has(rl,
                  "; 96.0 MB of live allocations in its zones and 6 of the cache's own scratch buffers (6 of them "
                  "its PP MoE ring's slots) hold the arena (the [VRAM-ARENA] lines above name them), so its zones "
                  "cannot grow. Only the cache's PP MoE ring holds it, and a prefill's claim on one of its slots "
                  "keeps it: the claim outlives its prefill until a context on device 1 is freed. Free the context "
                  "that ran the prefill, then load this one again"),
              "a ring alone in the arena is not named with its remedy");
        CHECK(!has(rl, "Free the model") && !has(rl, "until the process exits"),
              "a ring alone in the arena is blamed on a model to free or on the process");
        CHECK(!has(rl, "has finished"), "a ring alone in the arena is said to be let go when its prefill finishes");

        // the ring beside a live allocation: freeing the model or context releases that, and a later load the ring
        zone_arena_refusal with_live       = held;
        with_live.holders.live_beside_ring = 4;
        const std::string wl               = zone_arena_refusal_text(with_live);
        CHECK(has(wl, ". Free the model or context they belong to, then load this one again"),
              "a ring beside a live allocation is not given the free-the-model remedy");
        CHECK(!has(wl, "until the process exits"), "a ring beside a live allocation is held until the process exits");

        // other cache scratch inside the arena stays until the process exits, ring or no ring
        zone_arena_refusal with_scratch       = held;
        with_scratch.holders.scratch_in_arena = 8;
        const std::string sl                  = zone_arena_refusal_text(with_scratch);
        CHECK(has(sl, "8 of the cache's own scratch buffers (6 of them its PP MoE ring's slots) hold the arena") &&
                  has(sl, "The cache keeps its own scratch buffers in the arena until the process exits"),
              "other cache scratch beside the ring is not given the new-process remedy");

        // leases beside the ring with nothing else live outlived what they point at
        zone_arena_refusal with_lease   = held;
        with_lease.holders.chunk_leases = 2;
        const std::string ll            = zone_arena_refusal_text(with_lease);
        CHECK(has(ll,
                  ". Nothing but the cache's PP MoE ring is live in its zones, so the leases outlived what they "
                  "point at and are held until the process exits. Load this model in a new process before any "
                  "other on device 1"),
              "leases beside the ring alone are not named with the new-process remedy");
    }

    // ---- Case 41: a rebuild something live blocks keeps a short ONEDNN zone when the Graph floor is what it lacks
    // The ONEDNN zone holds the primitive-API pair, which has no DIRECT fallback, and the oneDNN Graph scratch floor,
    // which is a reservation: what the zone cannot hold of it goes to the DIRECT path under its cap. So a second load
    // whose Graph floor needs a larger zone than the live arena has keeps that zone when SCRATCH and RUNTIME hold their
    // needs and the zone still holds the pair. The kept zone is described by a floor of what it holds beyond the pair,
    // so the pair bound over it is the capacity less that stored floor, never below the kept pair, and the Graph room a
    // context check reads is the stored floor.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_arena_shortfall_is_reservation_only;
        using ggml_sycl::zone_arena_sizes;
        using ggml_sycl::zone_onednn_kept_pair_bytes;
        using ggml_sycl::zone_onednn_plan_keep_short;

        const zone_arena_sizes have = { 512 * MiB, 400 * MiB, 2048 * MiB };
        zone_arena_sizes       need = have;
        need.onednn                 = 1100 * MiB;
        CHECK(zone_arena_shortfall_is_reservation_only(have, need, 300 * MiB),
              "a shortfall of the ONEDNN zone alone, with the pair inside the zone, was not kept");
        CHECK(zone_arena_shortfall_is_reservation_only(have, need, 400 * MiB),
              "a zone that holds exactly the pair was not kept");
        CHECK(!zone_arena_shortfall_is_reservation_only(have, need, 400 * MiB + 1),
              "a zone below the pair it must hold was kept: the pair has no DIRECT fallback");
        zone_arena_sizes scratch_short = need;
        scratch_short.scratch          = have.scratch + 1;
        CHECK(!zone_arena_shortfall_is_reservation_only(have, scratch_short, 300 * MiB),
              "a SCRATCH shortfall was kept");
        zone_arena_sizes runtime_short = need;
        runtime_short.runtime          = have.runtime + 1;
        CHECK(!zone_arena_shortfall_is_reservation_only(have, runtime_short, 300 * MiB),
              "a RUNTIME shortfall was kept");
        CHECK(!zone_arena_shortfall_is_reservation_only(have, have, 300 * MiB),
              "zones that hold every need were taken for a keep, though nothing is short");
        zone_arena_sizes onednn_equal = need;
        onednn_equal.onednn           = have.onednn;
        CHECK(!zone_arena_shortfall_is_reservation_only(have, onednn_equal, 300 * MiB),
              "an ONEDNN zone holding exactly its need was taken for a keep");

        // The first model's zone: a 200 MiB pair and a 128 MiB floor. The second plans a 300 MiB pair (its own halves)
        // and a 1 GiB floor in the 400 MiB zone that stays.
        const zone_onednn_plan held = { 200 * MiB, 128 * MiB, 150 * MiB, 50 * MiB };
        const zone_onednn_plan live = { 300 * MiB, 1024 * MiB, 200 * MiB, 100 * MiB };
        const zone_onednn_plan kept = zone_onednn_plan_keep_short(held, live, 400 * MiB, 0);
        CHECK(kept.bare_bytes == 300 * MiB && kept.weights_bytes == 200 * MiB && kept.activations_bytes == 100 * MiB,
              "the kept zone is not described by the larger plan's whole pair");
        CHECK(kept.graph_floor_bytes == 100 * MiB, "the kept floor is not the zone less the pair plan");
        CHECK(zone_onednn_kept_pair_bytes(kept, 0) == 300 * MiB, "with no pair held, the pair plan is not kept");

        // A pair already held above the plan (two plans' halves merged): the floor gives way to it, so the bound over
        // the stored snapshot is the held pair, and the pair stays admitted.
        const zone_onednn_plan merged = zone_onednn_plan_keep_short(held, live, 400 * MiB, 350 * MiB);
        CHECK(zone_onednn_kept_pair_bytes(merged, 350 * MiB) == 350 * MiB, "a held pair above the plan was not kept");
        CHECK(merged.graph_floor_bytes == 50 * MiB, "the kept floor overlaps the pair already held");
        const size_t bound =
            ggml_sycl::zone_onednn_pp_pair_bound(400 * MiB, merged.bare_bytes, merged.graph_floor_bytes);
        CHECK(bound == 350 * MiB, "the pair bound over the kept snapshot fell below the pair already held");
        CHECK(400 * MiB - bound == merged.graph_floor_bytes,
              "the Graph room a context check reads over the kept snapshot is not the kept floor");
        CHECK(ggml_sycl::zone_onednn_graph_uncovered_bytes(80 * MiB, 400 * MiB - bound) == 30 * MiB,
              "a context's Graph scratch beyond the kept room is not the part the room cannot hold");

        // The runtime context's Graph scratch check over a kept zone (Case 24h's fit, with its room read as the
        // accessor reads it: the capacity less the pair bound over the stored snapshot). The 400 MiB zone kept for
        // the 300 MiB pair leaves 100 MiB. 16 heads at n_ctx 4096 under a 1 GiB cap: 384 KiB a row. With 600 MiB free
        // outside the arena and 256 MiB of headroom, 2048 (768 MiB, 668 above the room) is refused and 1024 (384 MiB,
        // 284 above, 316 left) named; on a card under the headroom only 256 (96 MiB, inside the room) is.
        const size_t kept_room =
            400 * MiB - ggml_sycl::zone_onednn_pp_pair_bound(400 * MiB, kept.bare_bytes, kept.graph_floor_bytes);
        CHECK(kept_room == 100 * MiB, "the Graph room the context check reads over a kept zone is not the kept floor");
        CHECK(ggml_sycl::zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, kept_room, 600 * MiB,
                                                          256 * MiB, 0) == 1024,
              "a context's Graph scratch beyond the kept room is not refused by the context check with 1024 named");
        CHECK(ggml_sycl::zone_onednn_graph_fit_largest_ub(16, 0, 0, 2048, 4096, 1024 * MiB, kept_room, 200 * MiB,
                                                          256 * MiB, 0) == 256,
              "on a card under the headroom the kept room does not serve the rung it holds");

        // zone_onednn_plan_keep's maximum floor is what the keep must not store: its bound would fall to the bare plan
        // and the 350 MiB pair already held would exceed it.
        const zone_onednn_plan maxed = ggml_sycl::zone_onednn_plan_keep(held, live);
        CHECK(ggml_sycl::zone_onednn_pp_pair_bound(400 * MiB, maxed.bare_bytes, maxed.graph_floor_bytes) < 350 * MiB,
              "sanity: the maximum floor's bound admits the held pair, so this case proves nothing");

        // The pair fills the zone: no Graph room, all Graph scratch on the DIRECT path.
        const zone_onednn_plan full = zone_onednn_plan_keep_short(held, live, 300 * MiB, 0);
        CHECK(full.graph_floor_bytes == 0, "a zone the pair fills was given a Graph floor");
        const zone_onednn_plan over = zone_onednn_plan_keep_short(held, live, 250 * MiB, 0);
        CHECK(over.graph_floor_bytes == 0, "a pair above the zone did not leave the floor at 0");

        // A zone with room beyond both plans' floors stores no floor above what a plan asked for.
        const zone_onednn_plan small_held = { 50 * MiB, 64 * MiB, 40 * MiB, 10 * MiB };
        const zone_onednn_plan small_live = { 60 * MiB, 64 * MiB, 50 * MiB, 10 * MiB };
        const zone_onednn_plan slack      = zone_onednn_plan_keep_short(small_held, small_live, 300 * MiB, 0);
        CHECK(slack.graph_floor_bytes == 64 * MiB, "the kept floor rose above the larger of the plans' floors");

        // A tie keeps the held pair, as zone_onednn_plan_keep does.
        const zone_onednn_plan tie_live = { 200 * MiB, 1024 * MiB, 100 * MiB, 100 * MiB };
        const zone_onednn_plan tie      = zone_onednn_plan_keep_short(held, tie_live, 400 * MiB, 0);
        CHECK(tie.weights_bytes == 150 * MiB && tie.activations_bytes == 50 * MiB && tie.graph_floor_bytes == 200 * MiB,
              "a tie did not keep the held pair, or its floor is not the zone less that pair");
    }

    // ---- Case 42: the DIRECT Graph-scratch cap's budget snapshot does not rise while the arena is kept -------------
    // The cap bounds the Graph floor, so a snapshot that rose with the budget (another tenant leaving) would raise the
    // floor of a zone nothing rebuilt, and a keep would be taken where a sufficient zone stood.
    {
        constexpr size_t GiB = 1024ull * 1024ull * 1024ull;
        using ggml_sycl::zone_graph_direct_cap_snapshot_kept;
        CHECK(zone_graph_direct_cap_snapshot_kept(0, 6 * GiB) == 6 * GiB, "an unset snapshot did not take the figure");
        CHECK(zone_graph_direct_cap_snapshot_kept(8 * GiB, 6 * GiB) == 6 * GiB, "a smaller budget was not taken");
        CHECK(zone_graph_direct_cap_snapshot_kept(6 * GiB, 8 * GiB) == 6 * GiB, "the snapshot rose while kept");
        CHECK(zone_graph_direct_cap_snapshot_kept(6 * GiB, 6 * GiB) == 6 * GiB, "an equal budget changed the snapshot");
    }

    // ---- Case 43: a kept zone warns once per load and device, and at every keep outside a load ------------------
    {
        using ggml_sycl::zone_kept_onednn_warn_due;
        CHECK(zone_kept_onednn_warn_due(0, 7), "a load's first keep did not warn");
        CHECK(!zone_kept_onednn_warn_due(7, 7), "a load's second keep warned again");
        CHECK(zone_kept_onednn_warn_due(7, 8), "the next load's keep did not warn");
        CHECK(zone_kept_onednn_warn_due(0, 0) && zone_kept_onednn_warn_due(7, 0),
              "a keep outside a staged load did not warn");
    }

    // ---- Case 44: a failed load puts the stored zone plan back as the zone holds it now ----------------------------
    // The restore stores zone_onednn_plan_keep_short(prior, prior, capacity now, held pair now): the prior plan exactly
    // while the zone still holds it, and a smaller Graph floor when a pair reserved in between under the failed load's
    // wider bound, or a smaller zone, leaves less room. In each outcome the pair bound over the stored plan is at least
    // the held pair, and the Graph room a context check reads (capacity less that bound) is the stored floor.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_onednn_plan;
        using ggml_sycl::zone_onednn_plan_keep_short;
        using ggml_sycl::zone_onednn_plan_same;
        using ggml_sycl::zone_onednn_pp_pair_bound;
        const zone_onednn_plan prior{ 200 * MiB, 300 * MiB, 150 * MiB, 50 * MiB };
        const auto             room_over = [](const zone_onednn_plan & p, size_t capacity) {
            return capacity - zone_onednn_pp_pair_bound(capacity, p.bare_bytes, p.graph_floor_bytes);
        };

        CHECK(zone_onednn_plan_same(prior, prior), "a plan does not compare the same as itself");
        for (int field = 0; field < 4; ++field) {
            zone_onednn_plan other = prior;
            size_t * const   f[]   = { &other.bare_bytes, &other.graph_floor_bytes, &other.weights_bytes,
                                       &other.activations_bytes };
            *f[field] += MiB;
            CHECK(!zone_onednn_plan_same(prior, other), "plans that differ in one field compare the same");
        }

        // The zone still holds the prior plan: 600 MiB less the 200 MiB pair leaves room for its 300 MiB floor.
        const zone_onednn_plan holds = zone_onednn_plan_keep_short(prior, prior, 600 * MiB, 100 * MiB);
        CHECK(zone_onednn_plan_same(holds, prior),
              "a zone that still holds the prior plan did not get it back exactly");
        CHECK(room_over(holds, 600 * MiB) == 300 * MiB, "the Graph room over the prior plan is not its floor");

        // A 360 MiB pair was reserved under the failed load's wider bound, above the prior plan's 300 MiB bound.
        const zone_onednn_plan crowded = zone_onednn_plan_keep_short(prior, prior, 600 * MiB, 360 * MiB);
        CHECK(crowded.bare_bytes == prior.bare_bytes && crowded.weights_bytes == prior.weights_bytes &&
                  crowded.activations_bytes == prior.activations_bytes,
              "a held pair above the prior bound changed the prior pair plan");
        CHECK(crowded.graph_floor_bytes == 240 * MiB, "the floor beside a 360 MiB held pair is not 240 MiB");
        CHECK(zone_onednn_pp_pair_bound(600 * MiB, crowded.bare_bytes, crowded.graph_floor_bytes) >= 360 * MiB,
              "the held pair is above the bound over the restored plan");
        CHECK(room_over(crowded, 600 * MiB) == 240 * MiB, "the Graph room beside the held pair is not the floor");

        // The zone was rebuilt to 400 MiB, below the prior plan's 500 MiB.
        const zone_onednn_plan smaller = zone_onednn_plan_keep_short(prior, prior, 400 * MiB, 0);
        CHECK(smaller.graph_floor_bytes == 200 * MiB, "the floor in a 400 MiB zone is not 200 MiB");
        CHECK(room_over(smaller, 400 * MiB) == 200 * MiB, "the Graph room in the smaller zone is not the floor");
    }

    // ---- Case 45: the SCRATCH zone's use over a cache's life, and the line its teardown logs ----------------------
    // The high-water keeps the live allocations of the moment it was set; a later placement below it changes neither.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_scratch_use;
        using ggml_sycl::zone_scratch_use_note_alloc;
        using ggml_sycl::zone_scratch_use_note_miss;
        using ggml_sycl::zone_scratch_use_text;

        zone_scratch_use use;
        CHECK(zone_scratch_use_text(use, 1, 512 * MiB).empty(), "an unused zone printed a line");

        zone_scratch_use_note_alloc(use, 2 * MiB, 2 * MiB, 1);
        zone_scratch_use_note_alloc(use, 9 * MiB, 11 * MiB, 2);
        zone_scratch_use_note_alloc(use, 1 * MiB, 3 * MiB, 2);  // after frees: below the high-water
        CHECK(use.requests == 3 && use.largest_request_bytes == 9 * MiB, "placements were not counted");
        CHECK(use.high_water_bytes == 11 * MiB && use.live_at_high_water == 2,
              "the high-water did not keep the allocations live when it was set");
        zone_scratch_use_note_alloc(use, 4 * MiB, 11 * MiB, 5);
        CHECK(use.live_at_high_water == 2, "an equal used figure replaced the high-water's moment");
        CHECK(use.misses == 0 && use.miss_bytes == 0, "a placement counted as a miss");

        zone_scratch_use_note_miss(use, 600 * MiB);
        zone_scratch_use_note_miss(use, 40 * MiB);
        CHECK(use.misses == 2 && use.miss_bytes == 640 * MiB && use.largest_miss_bytes == 600 * MiB,
              "misses were not counted");
        zone_scratch_use_note_miss(use, SIZE_MAX);
        CHECK(use.miss_bytes == SIZE_MAX, "the missed bytes did not saturate");

        zone_scratch_use only_misses;
        zone_scratch_use_note_miss(only_misses, MiB);
        CHECK(!zone_scratch_use_text(only_misses, 0, 0).empty(), "a zone that only missed printed nothing");

        zone_scratch_use shown;
        zone_scratch_use_note_alloc(shown, 9 * MiB, 13 * MiB + MiB / 2, 3);
        zone_scratch_use_note_miss(shown, 2 * MiB);
        const std::string line = zone_scratch_use_text(shown, 1, 512 * MiB);
        CHECK(line ==
                  "[UNIFIED-CACHE] SCRATCH zone use: device=1 high-water=13.50 MB of 512.00 MB with 3 allocations "
                  "live, largest request=9.00 MB, requests placed=1; zone misses (placements the zone refused, "
                  "each spilled to raw device memory or refused by its caller): requests=1 bytes=2.00 MB "
                  "largest=2.00 MB",
              "the teardown line's text changed");
    }

    // ---- Case 46: the MoE batched MMVQ scratch one op holds, and the sum over a model's expert ops ---------------
    // The figures are mmvq_moe_batched_dispatch's own: K padded to QK8_1, 36 bytes a block, ne11 * ne12 rows, and
    // n_tokens * n_expert_used int32 ids, each buffer as the SCRATCH zone's allocator places it: rounded up to its
    // 256-byte block grain (k_zone_arena_alignment). The vehicle is Qwen3.6-35B-A3B NVFP4 (top-8): gate and up have
    // K = 2048 and one activation row a token, down has K = 512 and one row a slot. The file holds 41 blocks; the 41st
    // is the MTP block, whose experts load only with MTP, so the main graph has 40 layers of gate, up and down.
    {
        using ggml_sycl::zone_moe_mmvq_op_bytes;
        using ggml_sycl::zone_moe_mmvq_operand;
        using ggml_sycl::zone_moe_mmvq_scratch;
        using ggml_sycl::zone_moe_mmvq_scratch_bytes;

        const zone_moe_mmvq_operand gate{ 2048, false };
        const zone_moe_mmvq_operand down{ 512, true };
        size_t                      bytes = 0;

        // 512 rows: gate 512 x 2304 + 16384 ids; down 4096 x 576 + 16384 ids. Every buffer is whole grains already.
        CHECK(zone_moe_mmvq_op_bytes(gate, 512, 8, &bytes) && bytes == 1196032, "the gate op at 512 rows is wrong");
        CHECK(zone_moe_mmvq_op_bytes(down, 512, 8, &bytes) && bytes == 2375680, "the down op at 512 rows is wrong");
        // 2048 rows: gate 2048 x 2304 + 65536 ids; down 16384 x 576 + 65536 ids.
        CHECK(zone_moe_mmvq_op_bytes(gate, 2048, 8, &bytes) && bytes == 4784128, "the gate op at 2048 rows is wrong");
        CHECK(zone_moe_mmvq_op_bytes(down, 2048, 8, &bytes) && bytes == 9502720, "the down op at 2048 rows is wrong");

        // K pads to QK8_1, not to MATRIX_ROW_PADDING: 33 columns are two blocks (72 bytes), and one id (4 bytes). The
        // allocator places each in one grain.
        CHECK(zone_moe_mmvq_op_bytes({ 33, false }, 1, 1, &bytes) && bytes == 256 + 256,
              "K did not pad to one Q8_1 block, or a buffer was not placed in a whole grain");

        // Each buffer is placed on its own, so each rounds up alone: one token of 8 slots takes a 2304-byte gate row
        // (nine grains exactly) and 32 bytes of ids (one grain); a 288-byte row (K = 256) takes two grains; 8 tokens
        // of 8 slots take exactly one grain of ids, and a ninth token a second.
        CHECK(zone_moe_mmvq_op_bytes(gate, 1, 8, &bytes) && bytes == 2304 + 256,
              "a whole-grain buffer was rounded, or the ids were not");
        CHECK(zone_moe_mmvq_op_bytes(down, 1, 8, &bytes) && bytes == 4608 + 256, "the down op's ids were not rounded");
        CHECK(zone_moe_mmvq_op_bytes({ 256, false }, 1, 1, &bytes) && bytes == 512 + 256,
              "a buffer one block over a grain did not take two");
        CHECK(zone_moe_mmvq_op_bytes(gate, 8, 8, &bytes) && bytes == 8 * 2304 + 256,
              "ids of exactly one grain were rounded");
        CHECK(zone_moe_mmvq_op_bytes(gate, 9, 8, &bytes) && bytes == 9 * 2304 + 512,
              "ids one token over a grain did not take two");
        // A buffer within one grain of SIZE_MAX is one the allocator refuses (tlsf_allocator::round_request): K = 32
        // and one slot is 36 bytes a token, so 2^64 / 36 tokens are 2^64 - 16 bytes of Q8_1, which only wraps.
        CHECK(!zone_moe_mmvq_op_bytes({ 32, false }, int64_t(SIZE_MAX / 36), 1, &bytes) && bytes == SIZE_MAX,
              "a buffer the rounding wraps did not saturate");

        // An MXFP4 stack's grouped dispatch also takes its tables from the zone, six buffers of int32 apiece. GPT-OSS
        // 20B's shape: K = 2880 (90 blocks, 3240 bytes a row), 32 experts, top-4. At 512 tokens: 2048 routed rows,
        // 128 chunks of 16 an expert, so the tables are 32 experts (128 bytes, one grain), 33 offsets (132, one grain),
        // 2048 rows (8192), two of 4096 chunks (16384 each) and the counter (4, one grain): 41728 bytes beside the
        // Q8_1 rows and 8192 ids bytes, which are whole grains.
        const zone_moe_mmvq_operand mxfp4_gate{ 2880, false, 32 };
        const zone_moe_mmvq_operand mxfp4_down{ 2880, true, 32 };
        CHECK(zone_moe_mmvq_op_bytes({ 2880, false, 0 }, 512, 4, &bytes) && bytes == 1658880 + 8192,
              "an ungrouped op took grouped tables");
        CHECK(zone_moe_mmvq_op_bytes(mxfp4_gate, 512, 4, &bytes) && bytes == 1658880 + 8192 + 41728,
              "the grouped gate op at 512 rows is wrong");
        CHECK(zone_moe_mmvq_op_bytes(mxfp4_down, 512, 4, &bytes) && bytes == 6635520 + 8192 + 41728,
              "the grouped down op at 512 rows is wrong");
        // 3 tokens of 4 slots: 12 rows make one chunk an expert (rounded up), so every table is under one grain. The
        // 9720 Q8_1 bytes take 38 grains, the 48 ids bytes one, and the six tables one each.
        CHECK(zone_moe_mmvq_op_bytes(mxfp4_gate, 3, 4, &bytes) && bytes == 9728 + 256 + 6 * 256,
              "a partial chunk did not round up to one chunk an expert, or a table was not placed in a grain");
        // 17 tokens of one slot: two chunks an expert, 64 chunks (256 bytes a table). 55080 Q8_1 bytes take 216
        // grains.
        CHECK(zone_moe_mmvq_op_bytes(mxfp4_gate, 17, 1, &bytes) && bytes == 55296 + 256 + 6 * 256,
              "17 rows did not make two chunks an expert");
        // 64 experts fill the experts table's grain exactly, so the 65 offsets (260 bytes) take two. 11776 bytes: 9728
        // Q8_1 and 256 ids, then experts 256, offsets 512, rows 256 (12), two chunk tables of 256 (64), counter 256.
        CHECK(zone_moe_mmvq_op_bytes({ 2880, false, 64 }, 3, 4, &bytes) && bytes == 11776,
              "the offsets table is not one entry longer than the experts table");
        CHECK(!zone_moe_mmvq_op_bytes({ 2880, false, -1 }, 512, 4, &bytes) && bytes == 0,
              "a negative expert count sized an op");
        CHECK(!zone_moe_mmvq_op_bytes({ 32, false, INT64_MAX }, 1, 1, &bytes) && bytes == SIZE_MAX,
              "an expert count whose tables overflow did not saturate");
        // Overflows whose wrapped products are small, which only the tables' guard sees: 2^62 experts are 2^64 bytes
        // of expert ids (0 once wrapped), and 2^40 experts of 2^22 chunks each (2^26 rows) are 2^64 bytes a chunk
        // table.
        CHECK(!zone_moe_mmvq_op_bytes({ 32, false, int64_t(1) << 62 }, 1, 1, &bytes) && bytes == SIZE_MAX,
              "2^62 experts wrapped to an expert table of no bytes");
        const int64_t chunk_rows = int64_t(1) << 26;
        CHECK(!zone_moe_mmvq_op_bytes({ 32, false, int64_t(1) << 40 }, chunk_rows, 1, &bytes) && bytes == SIZE_MAX,
              "2^62 chunks wrapped to a chunk table of no bytes");

        std::vector<zone_moe_mmvq_operand> vehicle;
        for (int layer = 0; layer < 40; ++layer) {
            vehicle.push_back(gate);  // ffn_gate_exps
            vehicle.push_back(gate);  // ffn_up_exps
            vehicle.push_back(down);  // ffn_down_exps
        }
        zone_moe_mmvq_scratch scratch;
        CHECK(zone_moe_mmvq_scratch_bytes(vehicle, 512, 8, &scratch), "the vehicle at 512 rows did not size");
        CHECK(scratch.per_op == 2375680, "one op's most at 512 rows is not the down op");
        CHECK(scratch.graph == 190709760, "the 40-layer sum at 512 rows is not 40 x 4767744");
        CHECK(zone_moe_mmvq_scratch_bytes(vehicle, 2048, 8, &scratch), "the vehicle at 2048 rows did not size");
        CHECK(scratch.per_op == 9502720, "one op's most at 2048 rows is not the down op");
        CHECK(scratch.graph == 762839040, "the 40-layer sum at 2048 rows is not 40 x 19070976");

        CHECK(zone_moe_mmvq_scratch_bytes({}, 512, 8, &scratch) && scratch.per_op == 0 && scratch.graph == 0,
              "a model with no admitted expert op owes something");

        // A non-positive input sizes nothing, and says so.
        CHECK(!zone_moe_mmvq_op_bytes({ 0, false }, 512, 8, &bytes) && bytes == 0, "K = 0 sized an op");
        CHECK(!zone_moe_mmvq_op_bytes(gate, 0, 8, &bytes) && bytes == 0, "no tokens sized an op");
        CHECK(!zone_moe_mmvq_op_bytes(gate, 512, 0, &bytes) && bytes == 0, "no routed slot sized an op");
        CHECK(!zone_moe_mmvq_op_bytes(gate, -1, 8, &bytes) && bytes == 0, "a negative token count sized an op");
        const zone_moe_mmvq_operand negative_k{ -4, true };
        CHECK(!zone_moe_mmvq_scratch_bytes({ gate, negative_k }, 512, 8, &scratch) && scratch.per_op == 0 &&
                  scratch.graph == 0,
              "an op with a negative K left a bound");
        const zone_moe_mmvq_operand negative_experts{ 2880, false, -32 };
        CHECK(!zone_moe_mmvq_scratch_bytes({ gate, negative_experts }, 512, 8, &scratch) && scratch.per_op == 0 &&
                  scratch.graph == 0,
              "an op with a negative expert count left a bound");
        CHECK(!zone_moe_mmvq_scratch_bytes(vehicle, 512, 0, &scratch) && scratch.per_op == 0 && scratch.graph == 0,
              "no routed slot left a bound");

        // The widest K is 2^58 blocks, 9 x 2^60 bytes a row: one row fits a 64-bit size_t, two do not.
        const zone_moe_mmvq_operand widest{ INT64_MAX, false };
        CHECK(zone_moe_mmvq_op_bytes(widest, 1, 1, &bytes) && bytes == (size_t(9) << 60) + 256,
              "one row of the widest K did not size exactly");
        // Overflow saturates: in the Q8_1 bytes, the slot count, the ids, one op's two buffers, and the sum.
        CHECK(!zone_moe_mmvq_op_bytes(widest, 2, 1, &bytes) && bytes == SIZE_MAX,
              "the Q8_1 bytes overflow did not saturate");
        CHECK(!zone_moe_mmvq_op_bytes(down, INT64_MAX, 8, &bytes) && bytes == SIZE_MAX,
              "the slot count overflow did not saturate");
        // 2^32 tokens of 2^32 slots wrap to exactly zero slots, which the later checks would pass as a zero-byte op.
        CHECK(!zone_moe_mmvq_op_bytes(down, int64_t(1) << 32, int64_t(1) << 32, &bytes) && bytes == SIZE_MAX,
              "a slot count that wraps to zero did not saturate");
        CHECK(!zone_moe_mmvq_op_bytes({ 32, false }, int64_t(1) << 31, int64_t(1) << 32, &bytes) && bytes == SIZE_MAX,
              "the ids bytes overflow did not saturate");
        // 9 x 2^60 bytes of Q8_1 and 2^63 bytes of ids each fit; together they do not.
        CHECK(!zone_moe_mmvq_op_bytes(widest, 1, int64_t(1) << 61, &bytes) && bytes == SIZE_MAX,
              "the op's two buffers overflowing together did not saturate");
        CHECK(!zone_moe_mmvq_scratch_bytes({ gate, widest }, 512, 8, &scratch) && scratch.per_op == SIZE_MAX &&
                  scratch.graph == SIZE_MAX,
              "an op that overflows did not saturate both bounds");
        // Alone, the overflowing op's SIZE_MAX adds to a zero sum without overflowing it, and is still not exact.
        CHECK(!zone_moe_mmvq_scratch_bytes({ widest }, 512, 8, &scratch) && scratch.per_op == SIZE_MAX &&
                  scratch.graph == SIZE_MAX,
              "a lone op that overflows answered exact");
        // K = 32 and one slot: 40 bytes a token. Each op holds 2^63 + 256 bytes as the allocator places its two
        // buffers (2^63 + 32 requested), which fits; two of them do not.
        const zone_moe_mmvq_operand one_block{ 32, false };
        const int64_t               half = int64_t(SIZE_MAX / 40 / 2 + 1);
        CHECK(zone_moe_mmvq_op_bytes(one_block, half, 1, &bytes) && bytes == (size_t(1) << 63) + 256,
              "the op just over half of SIZE_MAX did not size exactly");
        CHECK(!zone_moe_mmvq_scratch_bytes({ one_block, one_block }, half, 1, &scratch) &&
                  scratch.per_op == (size_t(1) << 63) + 256 && scratch.graph == SIZE_MAX,
              "the sum overflow did not saturate the graph bound alone");
    }

    // ---- Case 47: the SCRATCH zone's size, raised by the MoE batched MMVQ bound at the context's micro-batch -------
    // The bound is every expert op's buffers at once (zone_scratch_plan says why no smaller depth holds). The vehicle
    // is Case 46's. At a row count that is a multiple of 8 every buffer is whole grains, and its down op holds 4640
    // bytes a row, the figure the zone's use line measured (2.27 MB at 512 rows and 9.06 MB at 2048, one op live), and
    // gate and up 2336 each. The expected zones below are the rule applied to those figures: the larger of the default
    // and layers x (2 x 2336 + 4640) x rows there, and at any other row count the same with each op's 32 bytes of ids a
    // row rounded up to the grain. Neither limit binds here (Case 49 has them): `plan_of` passes SIZE_MAX for both,
    // which is no limit.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_moe_mmvq_op_bytes;
        using ggml_sycl::zone_moe_mmvq_operand;
        using ggml_sycl::zone_moe_mmvq_scratch;
        using ggml_sycl::zone_moe_mmvq_scratch_bytes;
        using ggml_sycl::zone_scratch_clamp;
        using ggml_sycl::zone_scratch_kept_bytes;
        using ggml_sycl::zone_scratch_plan;
        using ggml_sycl::zone_scratch_warn_due;
        using ggml_sycl::zone_scratch_zone_plan;

        const size_t default_bytes = 512 * MiB;
        // SIZE_MAX is no limit: this case holds neither the budget quarter nor the reserve's room (Case 49 does).
        const auto   plan_of       = [&](size_t nonfa_bytes, size_t bound_bytes) {
            return zone_scratch_zone_plan(default_bytes, nonfa_bytes, bound_bytes, SIZE_MAX, SIZE_MAX);
        };
        const zone_moe_mmvq_operand gate{ 2048, false };
        const zone_moe_mmvq_operand down{ 512, true };
        size_t                      down_rows = 0;
        size_t                      gate_rows = 0;
        CHECK(zone_moe_mmvq_op_bytes(down, 8, 8, &down_rows) && down_rows == 8 * 4640,
              "the down op does not hold the measured 4640 bytes a row");
        CHECK(zone_moe_mmvq_op_bytes(gate, 8, 8, &gate_rows) && gate_rows == 8 * 2336,
              "the gate op does not hold 2336 bytes a row");
        const size_t layer_row = (2 * gate_rows + down_rows) / 8;
        // One layer at `rows`: the Q8_1 rows (2 x 2304 + 4608 a row, whole grains) and three ops' ids, each rounded up.
        const auto   layer_at  = [](size_t rows) {
            const size_t ids = (32 * rows + 255) / 256 * 256;
            return (2 * 2304 + 4608) * rows + 3 * ids;
        };

        // The late plan reads the tensors the load created: 40 layers. The early plan reads the file's weights, which
        // hold the MTP block's experts too: 41 layers.
        std::vector<zone_moe_mmvq_operand> main_graph;
        for (int layer = 0; layer < 40; ++layer) {
            main_graph.insert(main_graph.end(), { gate, gate, down });
        }
        std::vector<zone_moe_mmvq_operand> with_mtp = main_graph;
        with_mtp.insert(with_mtp.end(), { gate, gate, down });

        zone_moe_mmvq_scratch scratch;
        CHECK(zone_moe_mmvq_scratch_bytes(main_graph, 512, 8, &scratch), "the vehicle at 512 rows did not size");
        zone_scratch_plan plan = plan_of(0, scratch.graph);
        CHECK(plan.required == default_bytes && plan.wanted == default_bytes,
              "at 512 rows the bound (181.9 MiB) moved the zone off its default");
        CHECK(plan.wanted == std::max(default_bytes, 40 * layer_row * 512), "the zone at 512 rows is not the rule's");

        CHECK(zone_moe_mmvq_scratch_bytes(main_graph, 2048, 8, &scratch), "the vehicle at 2048 rows did not size");
        plan = plan_of(0, scratch.graph);
        CHECK(plan.required == default_bytes && plan.wanted == 40 * layer_row * 2048 && plan.wanted > default_bytes,
              "at 2048 rows the zone is not the bound (727.5 MiB)");
        CHECK(zone_moe_mmvq_scratch_bytes(with_mtp, 2048, 8, &scratch), "the early plan at 2048 rows did not size");
        CHECK(plan_of(0, scratch.graph).wanted == 41 * layer_row * 2048,
              "the early plan's 123 ops at 2048 rows do not ask 745.7 MiB");

        // Monotone in the micro-batch, never below the default, and the rule's at every row count to 4096. The bound
        // passes the default between 1441 and 1442 rows: 536770560 bytes at 1441, 537139200 at 1442.
        size_t previous = 0;
        for (int64_t rows = 1; rows <= 4096; ++rows) {
            CHECK(zone_moe_mmvq_scratch_bytes(main_graph, rows, 8, &scratch), "the vehicle did not size");
            const size_t wanted = plan_of(0, scratch.graph).wanted;
            CHECK(wanted >= previous, "the zone shrank as the micro-batch grew");
            CHECK(wanted == std::max(default_bytes, 40 * layer_at(size_t(rows))), "the zone is not the rule's");
            CHECK(rows % 8 != 0 || layer_at(size_t(rows)) == layer_row * size_t(rows),
                  "a row count of whole-grain ids is not the per-row figure");
            previous = wanted;
        }
        CHECK(zone_moe_mmvq_scratch_bytes(main_graph, 1441, 8, &scratch) && scratch.graph == 536770560 &&
                  plan_of(0, scratch.graph).wanted == default_bytes,
              "the zone left its default at 1441 rows");
        CHECK(zone_moe_mmvq_scratch_bytes(main_graph, 1442, 8, &scratch) && scratch.graph == 537139200 &&
                  plan_of(0, scratch.graph).wanted > default_bytes,
              "the zone stayed at its default at 1442 rows");

        // The non-FA estimate raises what the zone must hold; the bound raises only what it is sized to.
        plan = plan_of(640 * MiB, 100 * MiB);
        CHECK(plan.required == 640 * MiB && plan.wanted == 640 * MiB, "a bound under the non-FA raise moved the zone");
        plan = plan_of(640 * MiB, 700 * MiB);
        CHECK(plan.required == 640 * MiB && plan.wanted == 700 * MiB, "the bound is not on top of the non-FA raise");
        plan = plan_of(100 * MiB, 0);
        CHECK(plan.required == default_bytes && plan.wanted == default_bytes,
              "a small non-FA estimate shrank the zone");
        plan = zone_scratch_zone_plan(16 * MiB, 0, 0, SIZE_MAX, SIZE_MAX);
        CHECK(plan.required == 16 * MiB && plan.wanted == 16 * MiB, "a zone with no raise is not its default");
        CHECK(plan_of(0, 40 * layer_row * 2048).clamp == zone_scratch_clamp::NONE, "no limit clamped the bound");

        // A rebuild something live blocks keeps a zone short of the bound alone, and asks the whole plan of one short
        // of what it must hold, or of one that holds it all.
        plan = plan_of(0, 40 * layer_row * 2048);
        CHECK(zone_scratch_kept_bytes(plan, 1024 * MiB) == plan.wanted, "a zone that holds the bound was kept short");
        CHECK(zone_scratch_kept_bytes(plan, plan.wanted) == plan.wanted, "a zone of exactly the bound was kept short");
        CHECK(zone_scratch_kept_bytes(plan, 600 * MiB) == 600 * MiB, "a zone short of the bound alone was not kept");
        CHECK(zone_scratch_kept_bytes(plan, default_bytes) == default_bytes,
              "a zone of exactly what it must hold was not kept");
        CHECK(zone_scratch_kept_bytes(plan, 256 * MiB) == plan.wanted,
              "a zone short of what it must hold was kept as if only the bound were short");
        plan = plan_of(640 * MiB, 700 * MiB);
        CHECK(zone_scratch_kept_bytes(plan, 600 * MiB) == 700 * MiB, "a zone short of the non-FA raise was kept");
        CHECK(zone_scratch_kept_bytes(plan, 660 * MiB) == 660 * MiB, "a zone that holds the non-FA raise was not kept");

        // A SCRATCH WARN that latches on a figure: once per figure, again only for a larger one.
        CHECK(zone_scratch_warn_due(0, 700 * MiB), "the first keep did not warn");
        CHECK(!zone_scratch_warn_due(700 * MiB, 700 * MiB), "a keep of the same figure warned again");
        CHECK(!zone_scratch_warn_due(745 * MiB, 700 * MiB), "a keep of a smaller figure warned again");
        CHECK(zone_scratch_warn_due(700 * MiB, 745 * MiB), "a keep of a larger figure did not warn");
    }

    // ---- Case 48: the arena reserve's checks on its tail zones, and the SCRATCH zone's room ----------------------
    // arena_reserve calls these checks itself. The room is the most SCRATCH the reserve accepts on top of what the
    // zone must hold with every check it passed still passing. Each of the three ways a larger SCRATCH zone makes the
    // reserve refuse where the zone without it passed is a scenario below, and the room keeps each one a pass.
    {
        constexpr size_t MiB   = 1024ull * 1024ull;
        constexpr size_t GiB   = 1024ull * MiB;
        constexpr size_t M     = ggml_sycl::k_zone_arena_min_shared_bytes;
        constexpr size_t grain = ggml_sycl::k_zone_arena_alignment;
        using ggml_sycl::zone_arena_chunk_holds_tail;
        using ggml_sycl::zone_arena_n_chunk_accepts;
        using ggml_sycl::zone_arena_per_chunk_cap;
        using ggml_sycl::zone_arena_reserve_shape;
        using ggml_sycl::zone_arena_reserve_shape_of;
        using ggml_sycl::zone_arena_round_alloc_bytes;
        using ggml_sycl::zone_arena_scratch_accepts;
        using ggml_sycl::zone_arena_scratch_room;
        using ggml_sycl::zone_arena_single_chunk_accepts;
        using ggml_sycl::zone_arena_single_chunk_tried;
        using ggml_sycl::zone_arena_tail_bytes;
        using ggml_sycl::zone_arena_tail_chunk_bytes;
        using ggml_sycl::zone_arena_tail_kv_reserve;
        using ggml_sycl::zone_arena_weight_chunks_fit;
        using ggml_sycl::zone_arena_zone_bytes;
        using ggml_sycl::zone_scratch_clamp;
        using ggml_sycl::zone_scratch_plan;
        using ggml_sycl::zone_scratch_zone_plan;

        CHECK(M == 16 * MiB && grain == 256, "the reserve's minimum or grain moved");

        // The sizes, as arena_reserve derives them.
        CHECK(zone_arena_zone_bytes(1) == 256 && zone_arena_zone_bytes(256) == 256 && zone_arena_zone_bytes(257) == 512,
              "a zone did not round up to the grain");
        CHECK(zone_arena_round_alloc_bytes(5 * MiB + 1) == 4 * MiB && zone_arena_round_alloc_bytes(4 * MiB) == 4 * MiB,
              "the arena did not round down to 2 MiB");
        CHECK(zone_arena_per_chunk_cap(8 * GiB, 0) == 8 * GiB, "no safe cap did not leave the device's cap");
        CHECK(zone_arena_per_chunk_cap(8 * GiB, 7 * GiB) == 7 * GiB, "the safe cap below the device's did not hold");
        CHECK(zone_arena_per_chunk_cap(4 * GiB, 6 * GiB) == 4 * GiB, "a safe cap above the device's replaced it");
        CHECK(zone_arena_per_chunk_cap(0, 5 * GiB) == 5 * GiB, "the safe cap alone was not the cap");
        CHECK(zone_arena_per_chunk_cap(0, 0) == 0, "no cap at all named one");
        CHECK(zone_arena_single_chunk_tried(9 * GiB, 0, 0), "a device with no cap did not try one chunk");
        CHECK(zone_arena_single_chunk_tried(4 * GiB, 8 * GiB, 4 * GiB),
              "an arena of exactly the cap was not one chunk");
        CHECK(!zone_arena_single_chunk_tried(4 * GiB + 1, 8 * GiB, 4 * GiB), "an arena above the cap tried one chunk");
        CHECK(zone_arena_tail_kv_reserve(1 * GiB) == 128 * MiB, "a small arena's KV share is not 128 MiB");
        CHECK(zone_arena_tail_kv_reserve(32 * GiB) == 256 * MiB, "a 32 GiB arena's KV share is not a 128th");
        CHECK(zone_arena_tail_kv_reserve(128 * GiB) == 512 * MiB, "a large arena's KV share is not 512 MiB");

        // The tail's sum refuses a wrap at either add and holds a sum of exactly SIZE_MAX, leaving *out alone when it
        // refuses.
        size_t tail_sum = 7;
        CHECK(zone_arena_tail_bytes({ 256, 512, 1024 }, &tail_sum) && tail_sum == 1792,
              "the tail is not its zones' sum");
        tail_sum = 7;
        CHECK(!zone_arena_tail_bytes({ SIZE_MAX, 1, 0 }, &tail_sum) && tail_sum == 7,
              "a wrap on the first add was a tail");
        CHECK(!zone_arena_tail_bytes({ SIZE_MAX - 1, 1, 1 }, &tail_sum) && tail_sum == 7,
              "a wrap on the second add was a tail");
        CHECK(zone_arena_tail_bytes({ SIZE_MAX - 2, 1, 1 }, &tail_sum) && tail_sum == SIZE_MAX,
              "a tail of exactly SIZE_MAX was refused");
        CHECK(zone_arena_tail_bytes({ SIZE_MAX - 256, 256 }, &tail_sum) && tail_sum == SIZE_MAX &&
                  !zone_arena_tail_bytes({ SIZE_MAX - 255, 256 }, &tail_sum),
              "a two-zone tail did not refuse exactly at the wrap");
        CHECK(zone_arena_tail_chunk_bytes(12 * GiB, 1459 * MiB, 1280 * MiB, 128 * MiB) == 1408 * MiB,
              "the tail chunk is not the tail and its KV share");
        CHECK(zone_arena_tail_chunk_bytes(12 * GiB, 1459 * MiB, 1400 * MiB, 128 * MiB) == 1459 * MiB,
              "the tail chunk is above the per-chunk cap");
        CHECK(zone_arena_tail_chunk_bytes(1 * GiB, 4 * GiB, 1000 * MiB, 128 * MiB) == 1 * GiB,
              "the tail chunk is above the arena");
        CHECK(zone_arena_tail_chunk_bytes(SIZE_MAX, SIZE_MAX, SIZE_MAX, 128 * MiB) == SIZE_MAX,
              "the tail chunk wrapped");
        CHECK(zone_arena_chunk_holds_tail(100 * MiB, 84 * MiB), "a chunk of exactly the tail and the minimum refused");
        CHECK(!zone_arena_chunk_holds_tail(100 * MiB, 84 * MiB + 1), "a chunk one byte short held the tail");
        CHECK(!zone_arena_chunk_holds_tail(SIZE_MAX, SIZE_MAX - M + 1), "a tail at the top of size_t wrapped");
        CHECK(zone_arena_weight_chunks_fit(100 * MiB, 84 * MiB), "weight chunks of exactly the minimum refused");
        CHECK(!zone_arena_weight_chunks_fit(100 * MiB, 84 * MiB + 1), "weight chunks one byte short fit");
        CHECK(!zone_arena_weight_chunks_fit(100 * MiB, 101 * MiB), "a tail chunk above the arena fit");

        zone_arena_reserve_shape shape = zone_arena_reserve_shape_of(20 * GiB + 3 * MiB, 16 * GiB, 0, 0);
        CHECK(shape.alloc_bytes == 16 * GiB && shape.per_chunk_cap == 0 && shape.single_chunk &&
                  shape.tail_kv_reserve == 128 * MiB,
              "the shape did not hold the arena to the device's VRAM");
        shape = zone_arena_reserve_shape_of(14 * GiB + 3 * MiB, 0, 16 * GiB, 15 * GiB);
        CHECK(shape.alloc_bytes == 14 * GiB + 2 * MiB && shape.per_chunk_cap == 15 * GiB && shape.single_chunk,
              "an unknown device size clamped the arena, or it did not round");
        shape = zone_arena_reserve_shape_of(30 * GiB, 32 * GiB, 30 * GiB, 29 * GiB);
        CHECK(!shape.single_chunk && shape.per_chunk_cap == 29 * GiB && shape.tail_kv_reserve == 240 * MiB,
              "an arena above the safe cap tried one chunk");

        // Every check a tail of `base` passes still passes at `base` + `room`, and one grain more fails one of them.
        const auto room_is_tight = [&](const zone_arena_reserve_shape & s, size_t base, size_t room) {
            const bool single  = zone_arena_single_chunk_accepts(s, base);
            const bool n_chunk = zone_arena_n_chunk_accepts(s, base);
            const auto passes  = [&](size_t tail) {
                return (!single || zone_arena_single_chunk_accepts(s, tail)) &&
                       (!n_chunk || zone_arena_n_chunk_accepts(s, tail));
            };
            return room % grain == 0 && passes(base + room) && !passes(base + room + grain);
        };
        const size_t required     = 512 * MiB;
        const size_t bound        = 1 * GiB;
        const auto   reserve_tail = [&](size_t other_tail, size_t scratch) {
            return other_tail + zone_arena_zone_bytes(scratch);
        };

        // Case (i), one chunk: a 4 GiB arena on a device with no per-allocation cap (P = 0, max_alloc = 0) whose
        // ONEDNN and RUNTIME zones take 3 GiB. The zone it must hold passes; the 1 GiB bound under the 1 GiB quarter
        // would leave no 16 MiB beside the tail. The room is 4096 - 16 - 3584 = 496 MiB.
        shape        = zone_arena_reserve_shape_of(4 * GiB, 0, 0, 0);
        size_t other = 3 * GiB;
        CHECK(shape.single_chunk && shape.per_chunk_cap == 0, "case (i) is not a single chunk with no cap");
        CHECK(!zone_arena_n_chunk_accepts(shape, 0), "a device with no cap accepted an N-chunk arena");
        CHECK(zone_arena_single_chunk_accepts(shape, reserve_tail(other, required)), "case (i): required refused");
        CHECK(!zone_arena_single_chunk_accepts(shape, reserve_tail(other, bound)), "case (i): the bound passed alone");
        size_t room = zone_arena_scratch_room(shape, other, required);
        CHECK(room == 496 * MiB && room_is_tight(shape, reserve_tail(other, required), room),
              "case (i): the room is not 496 MiB");
        zone_scratch_plan plan = zone_scratch_zone_plan(required, 0, bound, 1 * GiB, room);
        CHECK(plan.wanted == 1008 * MiB && plan.clamp == zone_scratch_clamp::ROOM, "case (i): the room did not hold");
        CHECK(zone_arena_single_chunk_accepts(shape, reserve_tail(other, plan.wanted)), "case (i): the zone refused");

        // Case (ii), the N-chunk tail: a 12 GiB arena whose per-chunk cap is 1459 MiB (a 1.5 GiB device cap at 95%),
        // ONEDNN 256 MiB and RUNTIME 512 MiB. The quarter (3 GiB) is no limit next to the tail chunk: the most tail
        // the cap holds is 1443 MiB, so the room is 1443 - 1280 = 163 MiB.
        shape = zone_arena_reserve_shape_of(12 * GiB, 0, 1536 * MiB, 1459 * MiB);
        other = 768 * MiB;
        CHECK(!shape.single_chunk && shape.tail_kv_reserve == 128 * MiB, "case (ii) is not an N-chunk arena");
        CHECK(zone_arena_n_chunk_accepts(shape, reserve_tail(other, required)), "case (ii): required refused");
        CHECK(!zone_arena_n_chunk_accepts(shape, reserve_tail(other, bound)), "case (ii): the bound passed alone");
        room = zone_arena_scratch_room(shape, other, required);
        CHECK(room == 163 * MiB && room_is_tight(shape, reserve_tail(other, required), room),
              "case (ii): the room is not 163 MiB");
        plan = zone_scratch_zone_plan(required, 0, bound, 3 * GiB, room);
        CHECK(plan.wanted == 675 * MiB && plan.clamp == zone_scratch_clamp::ROOM, "case (ii): the room did not hold");
        CHECK(zone_arena_n_chunk_accepts(shape, reserve_tail(other, plan.wanted)), "case (ii): the zone refused");
        // An arena above the cap never tries one chunk, so a 700 MiB zone, which the tail chunk refuses (1468 + 16 >
        // 1459), has no room although the whole arena would hold it.
        CHECK(!zone_arena_single_chunk_accepts(shape, reserve_tail(other, required)),
              "case (ii): an arena above the cap accepted one chunk");
        CHECK(zone_arena_scratch_room(shape, other, 700 * MiB) == 0, "case (ii): a zone the reserve refuses had room");

        // Case (iii), the weight remainder: a 4096 MiB arena 6 MiB above a 4090 MiB per-chunk cap, with 3000 MiB of
        // ONEDNN and RUNTIME. Past a tail of 3952 MiB the tail chunk and its 128 MiB KV share leave the weight chunks
        // under 16 MiB, so the room is 3952 - 3512 = 440 MiB.
        shape = zone_arena_reserve_shape_of(4096 * MiB, 0, 4090 * MiB, 0);
        other = 3000 * MiB;
        CHECK(!shape.single_chunk && shape.per_chunk_cap == 4090 * MiB, "case (iii) is not an N-chunk arena");
        CHECK(zone_arena_n_chunk_accepts(shape, reserve_tail(other, required)), "case (iii): required refused");
        CHECK(!zone_arena_n_chunk_accepts(shape, reserve_tail(other, bound)), "case (iii): the bound passed alone");
        CHECK(zone_arena_chunk_holds_tail(shape.per_chunk_cap, reserve_tail(other, bound)),
              "case (iii): the bound failed on the per-chunk cap, not on the weight chunks");
        room = zone_arena_scratch_room(shape, other, required);
        CHECK(room == 440 * MiB && room_is_tight(shape, reserve_tail(other, required), room),
              "case (iii): the room is not 440 MiB");
        plan = zone_scratch_zone_plan(required, 0, bound, 1 * GiB, room);
        CHECK(plan.wanted == 952 * MiB && plan.clamp == zone_scratch_clamp::ROOM, "case (iii): the room did not hold");
        CHECK(zone_arena_n_chunk_accepts(shape, reserve_tail(other, plan.wanted)), "case (iii): the zone refused");

        // A single-chunk arena's N-chunk fallback, reached when its one allocation fails, keeps its own checks: a
        // 16 GiB arena with 1 GiB of other zones passes both, and the room holds both (the fallback's KV share is the
        // tighter: 16384 - 16 - 128 - 1536 = 14704 MiB).
        shape = zone_arena_reserve_shape_of(16 * GiB, 0, 16 * GiB, 0);
        other = 1 * GiB;
        CHECK(shape.single_chunk && zone_arena_n_chunk_accepts(shape, reserve_tail(other, required)),
              "the single chunk's fallback refused the zone it must hold");
        room = zone_arena_scratch_room(shape, other, required);
        CHECK(room == 14704 * MiB && room_is_tight(shape, reserve_tail(other, required), room),
              "the room did not keep the single chunk's fallback");

        // The boundaries: a zone that leaves exactly 16 MiB beside the tail has no room; one grain less of the other
        // zones gives one grain of room; one grain more refuses the zone it must hold, and nothing is added to it.
        shape = zone_arena_reserve_shape_of(4096 * MiB, 0, 0, 0);
        CHECK(zone_arena_scratch_room(shape, 3568 * MiB, required) == 0, "a full single chunk had room");
        CHECK(zone_arena_scratch_room(shape, 3568 * MiB - grain, required) == grain, "one grain of room was not one");
        CHECK(zone_arena_scratch_room(shape, 3568 * MiB + grain, required) == 0, "a refused zone had room");
        // A room of 0 does not say which of those two it is; whether the reserve accepts the zone at all does.
        CHECK(zone_arena_scratch_accepts(shape, 3568 * MiB, required), "a full single chunk refused the zone");
        CHECK(!zone_arena_scratch_accepts(shape, 3568 * MiB + grain, required), "a refused zone was accepted");
        CHECK(!zone_arena_scratch_accepts(shape, SIZE_MAX - 100, required), "a wrapped tail was accepted");
        CHECK(!zone_arena_scratch_accepts(shape, 0, SIZE_MAX - 100), "a zone whose rounding wraps was accepted");
        plan = zone_scratch_zone_plan(required, 0, bound, SIZE_MAX, 0);
        CHECK(plan.wanted == required && plan.clamp == zone_scratch_clamp::ROOM, "no room still raised the zone");
        // The room is on the zone as the reserve rounds it: 156 bytes beside a zone the reserve rounds up by 200 are
        // no room, where the unrounded zone would leave 356 bytes (one grain).
        CHECK(zone_arena_scratch_room(shape, 3568 * MiB - 156, required - 200) == 0,
              "the room did not count the zone as the reserve rounds it");
        CHECK(zone_arena_scratch_room(shape, 3568 * MiB - grain - 156, required - 200) == grain,
              "the room did not count the zone as the reserve rounds it");
        CHECK(zone_arena_scratch_room(shape, SIZE_MAX - 100, required) == 0, "a wrapped tail had room");

        // The grid: on every shape and zone mix, the zone the rule sizes passes every check the zone it must hold
        // passed, whatever the bound and the quarter, and with no other limit binding it is the room's whole.
        const size_t budgets[]    = { 1 * GiB, 4 * GiB, 4096 * MiB + 3 * MiB, 12 * GiB, 16 * GiB, 30 * GiB };
        const size_t max_allocs[] = { 0, 1536 * MiB, 4090 * MiB, 8 * GiB, 32 * GiB };
        const size_t others[]     = { 0, 256 * MiB, 768 * MiB, 3 * GiB, 3568 * MiB, 11 * GiB };
        const size_t requireds[]  = { 16 * MiB, 512 * MiB + 1, 2 * GiB };
        const size_t bounds[]     = { 0, 100 * MiB, 1 * GiB, 6 * GiB, SIZE_MAX };
        for (size_t budget : budgets) {
            for (size_t max_alloc : max_allocs) {
                const size_t safe = max_alloc / 20 * 19;
                shape             = zone_arena_reserve_shape_of(budget, 0, max_alloc, safe);
                for (size_t other_tail : others) {
                    for (size_t req : requireds) {
                        const size_t base    = reserve_tail(other_tail, req);
                        const bool   single  = zone_arena_single_chunk_accepts(shape, base);
                        const bool   n_chunk = zone_arena_n_chunk_accepts(shape, base);
                        room                 = zone_arena_scratch_room(shape, other_tail, req);
                        CHECK((single || n_chunk) ? room_is_tight(shape, base, room) : room == 0,
                              "the room is not the largest every passed check keeps");
                        CHECK(zone_arena_scratch_accepts(shape, other_tail, req) == (single || n_chunk),
                              "the reserve's acceptance of the zone is not its single-chunk or N-chunk check's");
                        for (size_t b : bounds) {
                            for (size_t quarter : { budget / 4, SIZE_MAX }) {
                                plan              = zone_scratch_zone_plan(req, 0, b, quarter, room);
                                const size_t tail = reserve_tail(other_tail, plan.wanted);
                                CHECK(plan.wanted >= req && plan.wanted <= std::max(req, b),
                                      "the zone left the span between what it must hold and the bound");
                                CHECK(!single || zone_arena_single_chunk_accepts(shape, tail),
                                      "the zone broke a single chunk the zone it must hold fit");
                                CHECK(!n_chunk || zone_arena_n_chunk_accepts(shape, tail),
                                      "the zone broke an N-chunk arena the zone it must hold fit");
                                if (quarter == SIZE_MAX && b == SIZE_MAX) {
                                    CHECK(zone_arena_zone_bytes(plan.wanted) == zone_arena_zone_bytes(req) + room,
                                          "an unlimited bound did not take the whole room");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Case 49: the two limits on the MoE batched MMVQ bound -----------------------------------------------------
    // wanted = max(required, min(bound, budget quarter, required + room)). Which limit held the zone below the bound is
    // named, the room on a tie. A limit below what the zone must hold leaves it at that, clamped.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::zone_moe_mmvq_operand;
        using ggml_sycl::zone_moe_mmvq_scratch;
        using ggml_sycl::zone_moe_mmvq_scratch_bytes;
        using ggml_sycl::zone_scratch_clamp;
        using ggml_sycl::zone_scratch_kept_bytes;
        using ggml_sycl::zone_scratch_plan;
        using ggml_sycl::zone_scratch_required_bytes;
        using ggml_sycl::zone_scratch_warn_due;
        using ggml_sycl::zone_scratch_zone_plan;

        const size_t base = 512 * MiB;
        CHECK(zone_scratch_required_bytes(base, 640 * MiB) == 640 * MiB &&
                  zone_scratch_required_bytes(base, 100 * MiB) == base,
              "what the zone must hold is not the larger of the default and the non-FA estimate");

        // The bound below, at and above the quarter, with room to spare.
        zone_scratch_plan plan = zone_scratch_zone_plan(base, 0, 700 * MiB, 800 * MiB, 4096 * MiB);
        CHECK(plan.wanted == 700 * MiB && plan.clamp == zone_scratch_clamp::NONE, "a bound below the quarter moved");
        plan = zone_scratch_zone_plan(base, 0, 800 * MiB, 800 * MiB, 4096 * MiB);
        CHECK(plan.wanted == 800 * MiB && plan.clamp == zone_scratch_clamp::NONE, "a bound at the quarter clamped");
        plan = zone_scratch_zone_plan(base, 0, 900 * MiB, 800 * MiB, 4096 * MiB);
        CHECK(plan.wanted == 800 * MiB && plan.clamp == zone_scratch_clamp::BUDGET, "the quarter did not hold");
        // The room is on top of what the zone must hold, not a size of its own.
        plan = zone_scratch_zone_plan(base, 0, 900 * MiB, SIZE_MAX, 100 * MiB);
        CHECK(plan.wanted == 612 * MiB && plan.clamp == zone_scratch_clamp::ROOM, "the room is not on top of required");
        plan = zone_scratch_zone_plan(base, 0, 900 * MiB, 612 * MiB, 100 * MiB);
        CHECK(plan.wanted == 612 * MiB && plan.clamp == zone_scratch_clamp::ROOM, "a tie did not name the room");
        plan = zone_scratch_zone_plan(base, 0, 900 * MiB, 611 * MiB, 100 * MiB);
        CHECK(plan.wanted == 611 * MiB && plan.clamp == zone_scratch_clamp::BUDGET, "the lower quarter did not hold");
        // A quarter below what the zone must hold: the zone holds that, and the bound is clamped.
        plan = zone_scratch_zone_plan(base, 640 * MiB, 900 * MiB, 300 * MiB, 4096 * MiB);
        CHECK(plan.required == 640 * MiB && plan.wanted == 640 * MiB && plan.clamp == zone_scratch_clamp::BUDGET,
              "a quarter below what the zone must hold shrank it");
        // A bound not above what the zone must hold is no clamp, whatever the limits.
        plan = zone_scratch_zone_plan(base, 0, 400 * MiB, 100 * MiB, 0);
        CHECK(plan.wanted == base && plan.clamp == zone_scratch_clamp::NONE, "a bound below the default clamped");
        // SIZE_MAX is no limit, and the room's sum with what the zone must hold saturates.
        plan = zone_scratch_zone_plan(base, 0, SIZE_MAX - 1, SIZE_MAX, SIZE_MAX);
        CHECK(plan.wanted == SIZE_MAX - 1 && plan.clamp == zone_scratch_clamp::NONE, "no limit was a limit");

        // The vehicle at every micro-batch to 4096 rows with the quarter at 600 MiB: the zone follows the bound,
        // never shrinks, and is flat at the quarter once the bound passes it (the bound is exactly 600 MiB at 1689
        // rows and above it at 1690).
        const zone_moe_mmvq_operand        gate{ 2048, false };
        const zone_moe_mmvq_operand        down{ 512, true };
        std::vector<zone_moe_mmvq_operand> vehicle;
        for (int layer = 0; layer < 40; ++layer) {
            vehicle.insert(vehicle.end(), { gate, gate, down });
        }
        zone_moe_mmvq_scratch scratch;
        size_t                previous = 0;
        for (int64_t rows = 1; rows <= 4096; ++rows) {
            CHECK(zone_moe_mmvq_scratch_bytes(vehicle, rows, 8, &scratch), "the vehicle did not size");
            plan = zone_scratch_zone_plan(base, 0, scratch.graph, 600 * MiB, 4096 * MiB);
            CHECK(plan.wanted >= previous, "the clamped zone shrank as the micro-batch grew");
            CHECK(plan.wanted == std::max(base, std::min(scratch.graph, 600 * MiB)),
                  "the clamped zone is not the rule's");
            CHECK((plan.clamp == zone_scratch_clamp::BUDGET) == (scratch.graph > 600 * MiB),
                  "the quarter was named where it did not hold, or not where it did");
            previous = plan.wanted;
        }
        CHECK(previous == 600 * MiB, "the zone is not flat at the quarter at 4096 rows");
        CHECK(
            zone_moe_mmvq_scratch_bytes(vehicle, 1689, 8, &scratch) && scratch.graph == 600 * MiB &&
                zone_scratch_zone_plan(base, 0, scratch.graph, 600 * MiB, 4096 * MiB).clamp == zone_scratch_clamp::NONE,
            "a bound of exactly the quarter at 1689 rows was clamped, or the rows' ids were not placed in grains");

        // The keep rule and the WARN latch on a clamped zone: a zone between what it must hold and the clamped figure
        // is kept, and the clamp WARN latches on the bound it held back.
        plan = zone_scratch_zone_plan(base, 0, 900 * MiB, 675 * MiB, 4096 * MiB);
        CHECK(zone_scratch_kept_bytes(plan, 600 * MiB) == 600 * MiB, "a zone short of the clamped figure was not kept");
        CHECK(zone_scratch_kept_bytes(plan, 700 * MiB) == 675 * MiB, "a zone above the clamped figure was kept short");
        CHECK(zone_scratch_warn_due(0, 900 * MiB), "the first clamp did not warn");
        CHECK(!zone_scratch_warn_due(900 * MiB, 900 * MiB), "a clamp of the same bound warned again");
        CHECK(zone_scratch_warn_due(900 * MiB, 1200 * MiB), "a clamp of a larger bound did not warn");
    }

    // ---- Case 50: the WARN a clamp of the MoE batched MMVQ bound logs -----------------------------------------------
    // The line states the zone the plan asks for ("plans": an arena kept as it was built may hold more), the bound, the
    // limit that held it with its figures, what the zone must hold, the miss path as the dispatch takes it, and the one
    // remedy a user has. A room of 0 because the reserve refuses even the zone the plan must hold names no zone the
    // reserve accepts, so that line says the reserve refuses the zone instead.
    {
        constexpr size_t MiB = 1024ull * 1024ull;
        using ggml_sycl::k_zone_scratch_miss_path_text;
        using ggml_sycl::zone_scratch_clamp;
        using ggml_sycl::zone_scratch_clamp_text;
        using ggml_sycl::zone_scratch_plan;
        using ggml_sycl::zone_scratch_zone_plan;

        // The miss path names the consequence for an op whose type has a fallback and for one whose type has none.
        const std::string miss = k_zone_scratch_miss_path_text;
        CHECK(miss ==
                  "Expert ops whose scratch the zone cannot place take the miss path: device memory outside the arena "
                  "while the card has room for it, else the dispatch refuses the buffer by name (q8-scratch-alloc) and "
                  "the op takes its fallback where its type has one, while a Q1_0 or NVFP4 op, which has no other "
                  "executor on the card, ends with that named refusal and a failed compute",
              "the miss path does not name what an op with no fallback does");
        const std::string head   = "[VRAM-ARENA] device 1 plans its SCRATCH zone at ";
        const std::string remedy = ". " + miss + ". The bound follows the micro-batch, so a smaller -ub lowers it";

        // The quarter held it, beside a room that would have taken 812 MiB.
        zone_scratch_plan plan = zone_scratch_zone_plan(512 * MiB, 0, 900 * MiB, 600 * MiB, 300 * MiB);
        CHECK(plan.clamp == zone_scratch_clamp::BUDGET, "the quarter case is not the quarter's clamp");
        CHECK(zone_scratch_clamp_text(1, plan, 900 * MiB, 600 * MiB, 300 * MiB, true) ==
                  head +
                      "600.0 MB, short of the 900.0 MB MoE batched MMVQ scratch bound: a quarter of the available "
                      "budget holds it (600.0 MB; a quarter of the available budget is 600.0 MB, the arena's reserve "
                      "accepts a 812.0 MB zone, and the zone must hold 512.0 MB without the bound)" +
                      remedy,
              "the quarter's line is not the plan's");

        // The room held it.
        plan = zone_scratch_zone_plan(512 * MiB, 0, 900 * MiB, 1000 * MiB, 100 * MiB);
        CHECK(plan.clamp == zone_scratch_clamp::ROOM, "the room case is not the room's clamp");
        CHECK(zone_scratch_clamp_text(1, plan, 900 * MiB, 1000 * MiB, 100 * MiB, true) ==
                  head +
                      "612.0 MB, short of the 900.0 MB MoE batched MMVQ scratch bound: the room the arena's reserve "
                      "accepts holds it (612.0 MB; a quarter of the available budget is 1000.0 MB, the arena's reserve "
                      "accepts a 612.0 MB zone, and the zone must hold 512.0 MB without the bound)" +
                      remedy,
              "the room's line is not the plan's");

        // No room above what the zone must hold, which the reserve accepts: the line names that zone.
        plan = zone_scratch_zone_plan(512 * MiB, 0, 900 * MiB, 1000 * MiB, 0);
        CHECK(zone_scratch_clamp_text(1, plan, 900 * MiB, 1000 * MiB, 0, true) ==
                  head +
                      "512.0 MB, short of the 900.0 MB MoE batched MMVQ scratch bound: the room the arena's reserve "
                      "accepts holds it (512.0 MB; a quarter of the available budget is 1000.0 MB, the arena's reserve "
                      "accepts a 512.0 MB zone, and the zone must hold 512.0 MB without the bound)" +
                      remedy,
              "a zone the reserve accepts with no room was not named as accepted");

        // The reserve refuses even the zone the plan must hold: whichever limit held the bound, the line says the
        // reserve refuses that zone and names no zone it accepts.
        const std::string refused_head =
            head +
            "512.0 MB, short of the 900.0 MB MoE batched MMVQ scratch bound: the arena's reserve refuses even the "
            "512.0 MB zone the plan must hold, so the bound adds nothing to it (a quarter of the available budget is ";
        CHECK(zone_scratch_clamp_text(1, plan, 900 * MiB, 1000 * MiB, 0, false) == refused_head + "1000.0 MB)" + remedy,
              "a room of 0 from a refused zone named a zone the reserve accepts");
        plan = zone_scratch_zone_plan(512 * MiB, 0, 900 * MiB, 300 * MiB, 0);
        CHECK(plan.clamp == zone_scratch_clamp::BUDGET && plan.wanted == 512 * MiB,
              "a quarter below what the zone must hold is not the quarter's clamp");
        CHECK(zone_scratch_clamp_text(1, plan, 900 * MiB, 300 * MiB, 0, false) == refused_head + "300.0 MB)" + remedy,
              "the quarter's clamp of a refused zone named a zone the reserve accepts");

        // No clamp, no line.
        plan = zone_scratch_zone_plan(512 * MiB, 0, 400 * MiB, 1000 * MiB, 100 * MiB);
        CHECK(plan.clamp == zone_scratch_clamp::NONE &&
                  zone_scratch_clamp_text(1, plan, 400 * MiB, 1000 * MiB, 100 * MiB, true).empty(),
              "a plan with no clamp had a clamp line");
    }

    std::printf("PASS: zone-sizing structural path-scoped maxima\n");
    return 0;
}
