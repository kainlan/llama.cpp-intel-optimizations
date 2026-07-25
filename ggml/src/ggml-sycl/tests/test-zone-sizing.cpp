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

    std::printf("PASS: zone-sizing structural path-scoped maxima\n");
    return 0;
}
