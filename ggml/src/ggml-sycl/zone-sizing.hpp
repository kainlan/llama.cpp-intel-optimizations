//
// Structural path-scoped arena zone sizing.
//
// populate_host_zone_sizing used to size every zone from one global
// max_tensor_bytes. Consumers each need "the largest tensor MY path can
// reach"; handing all of them the largest tensor in the model over-provisions
// each zone by the difference. These predicates are pure and live in their own
// TU so they can be unit-tested without a GPU (see tests/test-zone-sizing.cpp).
//
// Classification is STRUCTURAL, never by name. A per-layer weight family
// repeats once per block, so it shares its (type, ne) key with many siblings;
// the vocabulary embedding and the LM head are singletons, or a pair when they
// happen to share type and shape. Names are a GGUF convention, not a
// guarantee, and a name predicate that matches nothing fails *silently* —
// every maximum degrades straight back to the global one, which is
// indistinguishable from the reclaim genuinely being zero.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_sycl {

// Minimal descriptor: deliberately NOT placement_tensor_info, so this header
// stays free of unified-cache.hpp (which pulls in SYCL and the whole backend).
// populate_host_zone_sizing adapts its inventory into these at the call site.
//
// `type` mirrors ggml_type as a plain int rather than including ggml.h, which
// would drag the whole tensor API into a unit whose entire point is that it
// depends on nothing. The value is only ever compared for equality when
// grouping, so the enum's identity is not needed here; the call site casts.
struct zone_tensor_desc {
    size_t      size      = 0;      // THE byte size. Authoritative for every magnitude comparison.
    int         type      = -1;     // ggml_type mirror; grouping input only.
    int64_t     ne[4]     = {};     // shape; GROUPING ONLY — never derive a size from it.
    bool        has_shape = false;  // when false, ne is meaningless and must not group.
    std::string name;               // DIAGNOSTIC ONLY — never a decision input.
};

struct path_scoped_maxima {
    size_t any_tensor         = 0;  // the legacy global max; consumers not yet repointed use this
    size_t onednn_eligible    = 0;  // largest tensor that can be a oneDNN matmul reorder subject
    size_t cpu_quant_eligible = 0;  // largest tensor the CPU quantization slots can hold
    size_t dma_streamed       = 0;  // largest tensor the host->device weight stream can carry
};

// A (type, ne) group must have at least this many members to be a per-layer
// weight family. Measured cardinality histograms (Task 1) show two cleanly
// separated populations on both reference models:
//
//   GPT-OSS 20B (459 tensors, 11 groups):  2x1  24x5  48x2  72x1  73x1  96x1
//   Mistral 7B  (291 tensors,  7 groups):  1x2  32x1  64x3  65x1
//
// 4 sits inside the 2 -> 24 gap and the 1 -> 32 gap. It must be >= 3 because
// GPT-OSS's embedding and LM head share type and shape and so collapse into a
// single group of cardinality 2; a `>= 2` rule would admit them and reclaim
// nothing. It is deliberately not n_layer/2 (which would be 12 and 16): a
// family present in only a subset of blocks would fall below that and be
// wrongly excluded, under-sizing the zone. Uncertainty resolves toward
// inclusion — over-inclusion costs today's over-provision, under-inclusion
// costs a runtime grow.
const size_t ZONE_PER_LAYER_MIN_GROUP = 4;

// True when the tensor is one member of a repeated per-layer weight family.
// `group_cardinality` is how many inventory entries share its (type, ne) key.
bool zone_is_per_layer_weight(const zone_tensor_desc & tensor, size_t group_cardinality);

// The three path predicates are intentionally identical today. They stay
// separate functions so each consumer's maximum can be narrowed independently
// once its real constraint is known — do not collapse them into one.
bool zone_is_onednn_reorder_eligible(const zone_tensor_desc & tensor, size_t group_cardinality);
bool zone_is_cpu_quant_eligible(const zone_tensor_desc & tensor, size_t group_cardinality);
bool zone_is_dma_streamed(const zone_tensor_desc & tensor, size_t group_cardinality);

path_scoped_maxima zone_scoped_maxima(const std::vector<zone_tensor_desc> & inventory);

}  // namespace ggml_sycl
