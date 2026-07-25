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
static const int TYPE_Q6_K  = 14;
static const int TYPE_MXFP4 = 39;

// Measured byte sizes from the Task 1 inventory. The MB figures in the
// comments are binary (bytes / 1024^2), matching the findings document.
static const size_t GPT_OSS_VOCAB_BYTES  = 615329280;  // 586.8 MB, Q8_0  2880 x 201088
static const size_t GPT_OSS_EXPERT_BYTES = 140988600;  // 134.5 MB, MXFP4 2880 x 2880 x 32
static const size_t MISTRAL_OUTPUT_BYTES = 107520000;  // 102.5 MB, Q6_K  4096 x 32000
static const size_t MISTRAL_EMBD_BYTES   = 73728000;   //  70.3 MB, Q4_0  4096 x 32000
static const size_t MISTRAL_FFN_BYTES    = 33030144;   //  31.5 MB, Q4_0  4096 x 14336

static zone_tensor_desc
desc(const std::string & name, size_t size, int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    zone_tensor_desc d;
    d.name      = name;
    d.size      = size;
    d.type      = type;
    d.ne[0]     = ne0;
    d.ne[1]     = ne1;
    d.ne[2]     = ne2;
    d.ne[3]     = ne3;
    d.has_shape = true;
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
        inventory.push_back(desc("blk.99.some_weight", GPT_OSS_VOCAB_BYTES, TYPE_Q8_0, 2880, 201088, 1, 1));
        inventory.push_back(desc("blk.98.some_weight", GPT_OSS_VOCAB_BYTES, TYPE_Q8_0, 2880, 201088, 1, 1));

        const path_scoped_maxima maxima = zone_scoped_maxima(inventory);

        CHECK(maxima.any_tensor == GPT_OSS_VOCAB_BYTES, "gpt-oss any_tensor must equal the global max (586.8 MB)");
        CHECK(maxima.onednn_eligible == GPT_OSS_EXPERT_BYTES,
              "gpt-oss onednn_eligible must fall to the expert family (134.5 MB)");
        CHECK(maxima.cpu_quant_eligible == GPT_OSS_EXPERT_BYTES,
              "gpt-oss cpu_quant_eligible must fall to the expert family (134.5 MB)");
        CHECK(maxima.dma_streamed == GPT_OSS_EXPERT_BYTES,
              "gpt-oss dma_streamed must fall to the expert family (134.5 MB)");
        CHECK(is_monotonic(maxima), "gpt-oss maxima must all be <= any_tensor");
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
        CHECK(is_monotonic(shapeless_maxima), "shapeless-only maxima must all be <= any_tensor");
    }

    std::printf("PASS: zone-sizing structural path-scoped maxima\n");
    return 0;
}
