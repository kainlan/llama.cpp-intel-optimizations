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

// NAMING. The two function prefixes below are a contract, not drift:
//
//   zone_*        pure. No global state, no output, same answer for the same
//                 arguments. Free to call from anywhere, including a hot path.
//   zone_sizing_* touches the process-global accounting table (under a mutex)
//                 or emits log output.
//
// zone_detect_collapse and zone_sizing_warn_if_collapsed are the pattern in
// miniature: a pure detector and its logging wrapper, deliberately named on
// opposite sides of the line. Put a new function on the side its behaviour
// puts it, and do not rename across the line for prefix uniformity — the
// prefix is how a call site knows, without reading the body, whether it is
// calling something free or something that takes a mutex and writes to a log.

// Minimal descriptor: deliberately NOT placement_tensor_info, so this header
// stays free of unified-cache.hpp (which pulls in SYCL and the whole backend).
// populate_host_zone_sizing adapts its inventory into these at the call site.
//
// `type` mirrors ggml_type as a plain int rather than including ggml.h, which
// would drag the whole tensor API into a unit whose entire point is that it
// depends on nothing. The value is only ever compared for equality when
// grouping, so the enum's identity is not needed here; the call site casts.
struct zone_tensor_desc {
    size_t      size      = 0;      // THE byte size AS STORED. Authoritative for every magnitude comparison.
    int         type      = -1;     // ggml_type mirror; grouping input only.
    int64_t     ne[4]     = {};     // shape; GROUPING ONLY — never derive a size from it.
    bool        has_shape = false;  // when false, ne is meaningless and must not group.
    std::string name;               // DIAGNOSTIC ONLY — never a decision input.

    // Bytes this tensor occupies once oneDNN has DEQUANTIZED it into the type
    // its matmul reorder consumes. A second authoritative magnitude, not a
    // multiple of `size`: a quantized weight expands on the way into the
    // reorder buffer, so `size` describes what the model file holds and this
    // describes what the ONEDNN zone must hold. Both are supplied by the
    // adapter, which is the only party that knows ggml's type traits — this TU
    // must never compute one from the other, nor from `ne`.
    //
    // Zero when the adapter could not establish it (no shape). A zero here
    // narrows nothing and can only under-size, so an adapter that stops
    // populating it degrades the same way a dropped `type` does — see the
    // classifier-collapse section below.
    size_t reorder_size = 0;
};

struct path_scoped_maxima {
    size_t any_tensor         = 0;  // the legacy global max; consumers not yet repointed use this
    size_t onednn_eligible    = 0;  // largest tensor that can be a oneDNN matmul reorder subject, AS STORED
    size_t cpu_quant_eligible = 0;  // largest tensor the CPU quantization slots can hold
    size_t dma_streamed       = 0;  // largest tensor the host->device weight stream can carry

    // Largest DEQUANTIZED oneDNN reorder buffer, i.e. the max over the eligible
    // set of `reorder_size`. This is what the ONEDNN zone actually has to hold.
    //
    // Two properties surprise people, and both are correct:
    //
    // 1. It may EXCEED `any_tensor`. Measured on Mistral 7B Q4_0: the largest
    //    tensor in the model is 102.5 MB, and the largest reorder buffer is
    //    112.0 MB. Dequantization can make a per-layer weight outgrow the
    //    model's biggest stored tensor. Do not assert `<= any_tensor` on it.
    //
    // 2. It is NOT `onednn_eligible` times a constant. The expansion factor is
    //    per type (Q4_0 3.56x, Q8_0 1.88x, MXFP4 3.76x), so across a
    //    mixed-quantization model the largest STORED eligible tensor and the
    //    largest EXPANDED one can be different tensors. Scaling
    //    `onednn_eligible` by the winner's factor therefore under-sizes
    //    whenever a lower-bit-rate tensor with more elements exists. The
    //    maximum has to be taken over the expanded sizes, which is why this is
    //    its own accumulator rather than a multiplier at the call site.
    size_t onednn_reorder = 0;
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
constexpr size_t k_zone_per_layer_min_group = 4;

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

// ---------------------------------------------------------------------------
// Mispredict accounting
// ---------------------------------------------------------------------------
//
// A zone sized from a path predicate grows on demand rather than failing when
// the predicate under-estimates. That makes a wrong predicate survivable; it
// also makes it invisible. Without a counter, a predicate that mispredicts on
// some future model degrades silently into "grow every time" — slower than the
// over-provision this sizing removed, and indistinguishable from an unrelated
// regression. These counters are this unit's own regression detector.
//
// ONLY a genuine sizing miss belongs here. The growth path in
// reserve_onednn_scratch is reached by two causes that must not be conflated:
// a request larger than the planned zone (a predicate defect — record it) and
// a sub-allocation that failed while the zone was large enough (allocator
// fragmentation — do NOT record it, the predicate was right).
//
// The state is process-global and is reached from the multi-threaded SYCL
// backend, so every entry point below takes a mutex.
void   zone_sizing_record_underestimate(const char * path, size_t requested_bytes, size_t planned_bytes);
size_t zone_sizing_underestimate_count(const char * path);
size_t zone_sizing_max_underestimate_bytes(const char * path);

// Records that a path consulted its planned zone, whether or not the zone was
// big enough. This is what separates "the path was never entered" from "the
// path was entered and never under-estimated": both leave the under-estimate
// counter at zero and they mean opposite things. Concretely, a GPT-OSS run
// never enters reserve_onednn_scratch at all (its MoE work goes through a
// separate PP-MoE oneDNN ring), so a zero under-estimate count on that model
// is not evidence that the oneDNN predicate is correct — it is evidence that
// nothing tested it. Observations alone are not a defect and never break the
// summary's silence.
void   zone_sizing_record_observation(const char * path);
size_t zone_sizing_observation_count(const char * path);

void zone_sizing_reset_underestimates();

// Reports every path with a non-zero under-estimate count. Returns true when
// anything was reported. NOTHING is emitted, and false is returned, when all
// counters are zero: a clean run must stay silent or the warning stops meaning
// anything. The return value is what makes that property testable host-only.
bool zone_sizing_log_underestimate_summary();

// ---------------------------------------------------------------------------
// Classifier collapse
// ---------------------------------------------------------------------------
//
// The descriptors above are copied out of the backend's own tensor inventory
// by an adapter this TU cannot see and cannot unit-test. If that adapter ever
// stopped carrying `type` / `ne` / `has_shape` — someone "simplifies" it, or a
// new call site writes its own loop copying only name and size — then every
// tensor keys uniquely, no group clears the threshold, and every path-scoped
// maximum stops narrowing anything. The zones revert to the global-max sizing
// this unit exists to remove, and NOTHING notices: the maxima stay internally
// consistent, so every existing assert, both correctness gates and the unit
// tests all keep passing while the plan reclaims nothing.
//
// The under-estimate counters above cannot catch it either — a collapse makes
// zones LARGER, so no request ever exceeds one. Hence a separate signal.
enum class zone_collapse_signal {
    NONE,          // at least one path-scoped maximum strictly narrowed
    NO_FAMILY,     // no group cleared the threshold: every scoped maximum is 0
    NO_NARROWING,  // every scoped maximum equals any_tensor: grouping is degenerate
};

// An inventory smaller than this cannot be expected to contain a per-layer
// family, so its lack of one is not evidence of anything. Four times the
// per-layer threshold: both reference models are an order of magnitude above
// it (GPT-OSS 459 tensors, Mistral 291), so a healthy run never approaches it.
constexpr size_t k_zone_collapse_min_inventory = 4 * k_zone_per_layer_min_group;

zone_collapse_signal zone_detect_collapse(const std::vector<zone_tensor_desc> & inventory,
                                          const path_scoped_maxima &            maxima);

// Emits one warning naming the likely cause when zone_detect_collapse fires,
// and nothing otherwise. Deliberately a warning and NOT an assert: a model
// that genuinely consists of singletons is legitimate, and so is one whose
// largest tensor is itself a per-layer weight.
void zone_sizing_warn_if_collapsed(const std::vector<zone_tensor_desc> & inventory, const path_scoped_maxima & maxima);

}  // namespace ggml_sycl
