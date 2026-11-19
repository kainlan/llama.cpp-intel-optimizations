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

    // Q8_1 bytes a dense quantized MUL_MAT quantizes its activations into, PER
    // TOKEN (llama.cpp-479i): zone_mmq_src1_bytes_per_token(ne[0], ne[2], ne[3]).
    // Supplied by the adapter, never derived here: whether a tensor is a dense
    // MUL_MAT operand is the caller's knowledge. Zero means "not one" -- an expert
    // stack (MUL_MAT_ID, its own moe_q8 workspace), a non-quantized weight, or a
    // tensor with no shape. The classifier deliberately does NOT decide this from
    // ne[2] > 1: that is the expert predicate and it misclassifies dense 3-D
    // operands such as the MLA wk_b / wv_b (llama.cpp-8xbt).
    size_t mmq_src1_bytes_per_token = 0;

    // dense f16 dequant scratch (llama.cpp-479i): f16 bytes of the WHOLE dequantized weight a
    // dense MUL_MAT's oneDNN arm materializes (zone_dequant_f16_weight_bytes(ne[0], ne[1])), and the
    // f16 activation bytes per token it converts alongside (zone_dequant_f16_src1_bytes_per_token).
    // Supplied by the adapter, which knows the type and the expert role; zero means "not a candidate".
    size_t dequant_f16_weight_bytes         = 0;
    size_t dequant_f16_src1_bytes_per_token = 0;

    // The same two figures for a dense weight the oneDNN PP scratch may or may not supply (llama.cpp-8ony): a type
    // the unified kernel's oneDNN f16 route serves (Q4_0, MXFP4). Its f16 copies come from the scratch when the
    // scratch is enabled for its type and the tensor is a per-layer weight the ONEDNN zone was sized for;
    // otherwise they come from the planned dequant buffers, so only then do they count
    // (zone_dequant_f16_planned_when_unsupplied). Zero means "not such a tensor". Unlike the two fields above,
    // these are not planned unconditionally: that would reserve a copy for every layer weight the scratch supplies.
    size_t dequant_f16_if_unsupplied_weight_bytes         = 0;
    size_t dequant_f16_if_unsupplied_src1_bytes_per_token = 0;
    // The adapter's answer to "is the oneDNN PP scratch enabled for this tensor's type" (environment and the
    // default type set). Only read together with the two fields above.
    bool   pp_scratch_type_enabled                        = false;

    // True for a tensor whose only consumer is a row gather (GET_ROWS: the token / position embedding lookup), so
    // none of the MUL_MAT-side marks above describe it (llama.cpp-8ony). Supplied by the adapter from the model
    // loader's own role for the tensor, which makes a tied token embedding that doubles as the output head a
    // MUL_MAT operand and so NOT gather-only. The marks are the adapter's to set; a gather-only tensor is
    // excluded from every one of them here, in one place, rather than by each mark's own condition.
    bool   get_rows_only                                  = false;
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

    // Largest Q8_1 src1 bytes-per-token over every dense MUL_MAT operand. Not a
    // per-layer-family maximum: a singleton such as the LM head is a MUL_MAT src0
    // and counts. See zone_tensor_desc::mmq_src1_bytes_per_token.
    size_t mmq_src1_bytes_per_token = 0;

    // Largest f16 dequant weight and widest f16 activation row over the adapter's marked
    // candidates. Both are plain maxima over marked tensors, like mmq_src1_bytes_per_token.
    size_t dequant_f16_weight_bytes         = 0;
    size_t dequant_f16_src1_bytes_per_token = 0;
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

// True when the tensor is a stack of per-expert matrices rather than a single
// matmul operand. ne[2] carries the expert count on a MoE weight (32 on
// GPT-OSS) and is 1 — or unset — on a dense one, so this is structural, like
// everything else here, and needs no name.
bool zone_is_moe_expert_tensor(const zone_tensor_desc & tensor);

// The three path predicates are NO LONGER identical: the oneDNN one excludes
// MoE expert tensors, the other two do not. That divergence is the whole reason
// they were kept as separate functions — do not re-collapse them.
//
// Expert weights are consumed by a separate PP-MoE oneDNN ring with its own
// sizing (plan.pp_moe_onednn_*), never by reserve_onednn_scratch. Measured, on
// GPT-OSS 20B MXFP4 at -p 512: `zone sizing coverage: onednn observations=0`
// and `[ARENA-PP-ONEDNN] reserve_calls=0 get_calls=0`. The path is not merely
// cold on that model, it is unreachable.
//
// Including them cost real VRAM rather than only tidiness. The expert family
// dominates the eligible set, so it set the zone size for a zone the model
// never touches: 640.7 MB of ONEDNN zone on GPT-OSS, taken straight out of the
// arena weight zone, which the MoE down-i8 layout pass then could not spend on
// granted layers at ~261 MB each.
//
// KNOWN, INSTRUMENTED RISK, and the same shape as the LM-head note below. If
// some future primitive does route an expert tensor through
// reserve_onednn_scratch, this under-estimates. That is survivable by
// construction — the zone grows on demand and zone_sizing_record_underestimate
// counts it — whereas over-inclusion is silent and permanent. Uncertainty
// resolves toward exclusion HERE, opposite to the per-layer threshold, because
// here the error is instrumented and there it is not.
bool zone_is_onednn_reorder_eligible(const zone_tensor_desc & tensor, size_t group_cardinality);

// These two keep expert tensors: the CPU quantization slots and the host->device
// weight stream both genuinely carry them.
bool zone_is_cpu_quant_eligible(const zone_tensor_desc & tensor, size_t group_cardinality);
bool zone_is_dma_streamed(const zone_tensor_desc & tensor, size_t group_cardinality);

path_scoped_maxima zone_scoped_maxima(const std::vector<zone_tensor_desc> & inventory);

// ---------------------------------------------------------------------------
// Dense MMQ/MMVQ Q8_1 src1 scratch (llama.cpp-479i)
// ---------------------------------------------------------------------------
//
// A dense quantized MUL_MAT quantizes its F32 activations to Q8_1 before the
// MMQ/MMVQ kernel runs. These mirror ggml_sycl_op_mul_mat's own arithmetic
// (`required_size`) so the planner, the graph-entry check and the dispatch agree on
// one number; the backend static_asserts each constant against its source of truth
// (MATRIX_ROW_PADDING, QK8_1, sizeof(block_q8_1)). Pure: no state, no log.
constexpr int64_t k_zone_mmq_src1_row_padding  = 512;  // MATRIX_ROW_PADDING: K is padded to this
constexpr int64_t k_zone_mmq_src1_block_elems  = 32;   // QK8_1
constexpr size_t  k_zone_mmq_src1_block_bytes  = 36;   // sizeof(block_q8_1): half2 ds + 32 int8
constexpr size_t  k_zone_mmq_src1_overflow_pad = 32;   // Q6K_DS_OVERFLOW_PAD: SOA Q6_K reads 8 half2 past the ds region
constexpr size_t  k_zone_mmq_src1_align        = 256;  // plan figures are 256-aligned like the other scratch pools

// Bytes of one quantized row of `ne10` columns. False on a non-positive K or overflow.
bool zone_mmq_src1_row_bytes(int64_t ne10, size_t * out);

// Exact bytes of the buffer for `nrows` rows of `ne10` columns, exactly the dispatch's
// `required_size` (the overflow pad only when the weight layout is SOA/COALESCED).
bool zone_mmq_src1_required_bytes(int64_t nrows, int64_t ne10, bool with_overflow_pad, size_t * out);

// Bytes per TOKEN for a weight whose K is `ne0`: one quantized row per token, times
// ne2 * ne3 for a dense batched weight (src1 then has n_tokens * ne2 * ne3 rows).
bool zone_mmq_src1_bytes_per_token(int64_t ne0, int64_t ne2, int64_t ne3, size_t * out);

// The plan figure: n_ubatch tokens of `bytes_per_token`, plus the overflow pad,
// aligned up to 256. Zero when bytes_per_token is zero (no dense quantized operand).
bool zone_mmq_src1_scratch_bytes(size_t bytes_per_token, uint32_t n_ubatch, size_t * out);

// ---------------------------------------------------------------------------
// dense f16 dequant scratch (llama.cpp-479i)
// ---------------------------------------------------------------------------
//
// ggml_sycl_op_mul_mat_sycl's f16 arm converts the WHOLE src0 weight and the src1 activations to f16
// before the oneDNN GEMM, each into its own planned buffer. These mirror the dispatch's own arithmetic; the
// backend static_asserts the element size against sizeof(sycl::half). Pure: no state, no log.
constexpr size_t k_zone_dequant_f16_elem_bytes = 2;    // sizeof(sycl::half)
constexpr size_t k_zone_dequant_f16_align      = 256;  // buffer and plan figures are 256-aligned

// f16 bytes of a dequantized 2-D weight slice of ne0 x ne1. False on a non-positive extent or overflow.
bool zone_dequant_f16_weight_bytes(int64_t ne0, int64_t ne1, size_t * out);

// f16 activation bytes per token for a weight whose K is `ne0`, times ne2 * ne3 for a batched operand.
bool zone_dequant_f16_src1_bytes_per_token(int64_t ne0, int64_t ne2, int64_t ne3, size_t * out);

// Bytes of one buffer holding `elems` f16 elements (the src0 copy or the src1 copy of one op), rounded up
// to the region alignment. `elems` is 0 for an operand that needs no copy. False on a negative count or overflow.
bool zone_dequant_f16_region_bytes(int64_t elems, size_t * out);

// The plan figure, one number PER BUFFER (they are separate buffers): the largest weight copy, and n_ubatch
// activation rows at the widest K, each aligned to 256. Zero when there is no candidate. Separate numbers
// because the graph walk ensures each buffer at max(plan, demand), so a later graph never regrows (and
// retires) a buffer that a recorded graph baked.
bool zone_dequant_f16_plan_bytes(size_t   max_weight_bytes,
                                 size_t   src1_bytes_per_token,
                                 uint32_t n_ubatch,
                                 size_t * src0_bytes,
                                 size_t * src1_bytes);

// Whether a dense weight's f16 copies are planned into the dequant buffers because the oneDNN PP scratch will not
// supply them (llama.cpp-8ony). The scratch supplies an op when it is enabled for the weight's type and the pair
// fits the ONEDNN zone; the zone's own plan covers exactly the per-layer weights (zone_is_onednn_reorder_eligible).
// An eligible weight of an enabled type is therefore supplied, and needs no dequant plan, on the condition that its
// activations half is within the placeholder the zone is sized with (the largest eligible STORED weight) and the pair
// is within the pair bound. That condition fails at a large -ub (Mistral ffn_down at -ub 4096 needs ~234 MB against a
// ~223 MB bound): the op is then refused by the scratch, draws the dequant buffers, and nothing planned them. Closing
// that gap needs the real n_ubatch at planning time, which is llama.cpp-fkpg; until then it is walk-grown.
// A weight the zone was not sized for (the LM head, a tied embedding) or a type the scratch is off for
// (GGML_SYCL_ONEDNN_PP_UNIFIED_SCRATCH=0) draws the dequant buffers instead. A head that the zone's slack happens to
// supply is still planned: whether the head runs on many rows or on the last row only is unknown until llama.cpp-fkpg
// delivers n_outputs, and an unused plan is bounded by that one weight's f16 copy. Pure.
bool zone_dequant_f16_planned_when_unsupplied(bool pp_scratch_type_enabled, bool pair_eligible);

// ---------------------------------------------------------------------------
// oneDNN PP scratch admission (llama.cpp-8ony)
// ---------------------------------------------------------------------------
//
// Whether an op's f16 weight + activation copies are PLANNED to live in the ONEDNN zone (the oneDNN PP reorder
// scratch), as opposed to the planned RUNTIME-zone dense f16 dequant buffers above. One fact with one source:
// the zone the arena was actually built with. The LM head is deliberately outside the ONEDNN zone's sizing
// (zone_is_onednn_reorder_eligible) and inside the dequant plan, so asking "is this op a oneDNN PP candidate?"
// alone sends it to a scratch the plan never provisioned, and the arena then refuses to grow once weights are
// resident. Both the op arm and the graph-entry walk must ask THIS question, with the same numbers.
//
// `arena_active` is false when there is no ONEDNN zone at all (no arena): nothing was planned, nothing can
// disagree, and the scratch comes from the unified-cache allocation path as it always did. With an arena the
// pair must fit the bound the zone was planned to hold (sum <= `pair_bound_bytes`, overflow-checked: a wrapped sum
// compares as small). `pair_bound_bytes` is zone_onednn_pp_pair_bound over the zone's capacity, NOT the capacity
// itself: a caller passing the raw capacity would admit a pair that takes the bytes reserved for the Graph SDPA
// scratch. Pure.
bool zone_onednn_pp_scratch_planned(bool   arena_active,
                                    size_t pair_bound_bytes,
                                    size_t weights_bytes,
                                    size_t activations_bytes);

// The pair reserve_onednn_scratch should size a reservation to, given what the cache already holds and what the
// op now asks for. Never smaller than what is held, per component: the weights and activations halves are separate
// blocks and different ops are largest in different halves (a 512-row layer op needs a wider activations half than
// the 256-row LM-head op, which needs the wider weights half), so replacing the held pair by the latest request
// shrinks one half every time and forces the regrowth that the plan never provisioned (llama.cpp-8ony).
//
// With an arena the merged pair is still bounded by `pair_bound_bytes` (zone_onednn_pp_pair_bound, not the raw
// capacity): two ops that each fit the bound can merge, per component, into a pair above it. A held pair that
// cannot be merged inside the bound (left over from a smaller or rebuilt arena) must not wedge every later
// request, so the request is used as asked.
//
// The zone's PLANNED pair is a floor (llama.cpp-8ony): the first reservation is sized to max(held, requested, planned)
// per component, so the pair the plan provisioned is reserved once and does not regrow for the ops that plan sized
// (a regrow needs the superseded reservation and the new one in the zone at once, which a zone sized for one pair plus
// the Graph floor cannot hold). That holds only for the ops of the plan whose pair is kept: an op that needs the other
// plan's halves (kept (100, 10) at a bound of 110, request (10, 100)) fits neither the planned pair nor the merge, so
// the target is the request as asked and the held pair is replaced. The held pair's release is event-deferred, so for
// a moment the superseded block and the new one can both occupy the zone; if the zone allocation then fails, the
// request is served through the unified-cache direct path (the transient old-plus-new case in
// reserve_onednn_scratch). The request fits the bound alone, which acquire has already admitted it against.
// Used only with an arena, where the planned pair exists; without one the planned halves are ignored. When that pair
// does not fit `pair_bound_bytes` (a zone clamped below its plan) the target is the held-and-requested merge. Pass 0, 0
// for no planned pair. Pure; a null out is ignored.
void zone_onednn_scratch_reserve_target(bool     arena_active,
                                        size_t   pair_bound_bytes,
                                        size_t   held_weights_bytes,
                                        size_t   held_activations_bytes,
                                        size_t   planned_weights_bytes,
                                        size_t   planned_activations_bytes,
                                        size_t   requested_weights_bytes,
                                        size_t   requested_activations_bytes,
                                        size_t * weights_bytes,
                                        size_t * activations_bytes);

// The most an op's f16 pair may be for the ONEDNN zone to count it as planned there (the `pair_bound_bytes` that
// zone_onednn_pp_scratch_planned and zone_onednn_scratch_reserve_target take). The zone is sized as the
// primitive-API pair's own plan plus a floor for the oneDNN Graph SDPA scratch that shares it, and the zone is never
// smaller than a fixed minimum, so its capacity can sit well above both. A pair is admitted up to capacity - floor:
// that is slack nobody planned for, so admitting it cannot push the Graph SDPA scratch onto its DIRECT path (an
// unplanned device allocation).
// It is never admitted below the pair plan itself (a zone clamped so that capacity - floor falls under the plan
// still holds the plan, which is what the planner's own ops are sized from), and never above the capacity.
// `bare_plan_bytes` and `graph_floor_bytes` are the stored figures the zone was sized from; neither is recomputed
// by the caller. A floor larger than the capacity leaves only the plan. Pure.
size_t zone_onednn_pp_pair_bound(size_t capacity_bytes, size_t bare_plan_bytes, size_t graph_floor_bytes);

// The two figures the ONEDNN zone was sized from, kept together as ONE snapshot (llama.cpp-8ony): the pair's own
// plan and the Graph SDPA floor that shares the zone. Each is a plain number the planner can overwrite for the next
// model; the zone is built once, so a bound derived from the live figures of a later plan (a draft model loaded beside
// the target) would describe a zone that does not exist.
struct zone_onednn_plan {
    size_t bare_bytes        = 0;
    size_t graph_floor_bytes = 0;
    // The two halves of the pair plan: the weights half (the largest dequantized per-layer weight) and the
    // activations half. The first reservation is sized to them, so the pair never regrows (llama.cpp-8ony).
    size_t weights_bytes     = 0;
    size_t activations_bytes = 0;
};

// The snapshot to keep when the arena's zones were found sufficient for `live` and the zone is NOT rebuilt: the zone
// stays the size an earlier plan built it to, so it must still be described by the larger plan, never by a later,
// smaller one's. The pair is kept whole (its halves sum into its bare plan): the pair of whichever plan has the larger
// bare plan, the held one on a tie. Maxing each half on its own would build a pair no plan had, above what the zone
// holds. Keeping one pair means an op sized by the other plan's halves is reserved as asked (see
// zone_onednn_scratch_reserve_target), not from the kept pair. The Graph floor is a separate requirement on the same
// zone and keeps its own maximum. A rebuilt zone is described by the live plan outright (the caller stores it
// directly). Pure.
zone_onednn_plan zone_onednn_plan_keep(const zone_onednn_plan & held, const zone_onednn_plan & live);

// Whether two snapshots are the same in all four figures: the test a load's published record uses to tell that the
// stored snapshot still holds what the load left in it. Pure.
bool zone_onednn_plan_same(const zone_onednn_plan & a, const zone_onednn_plan & b);

// Whether the oneDNN PP scratch is allowed to supply a dense op's f16 copies at all, as a function of the
// GGML_SYCL_ONEDNN_PP_UNIFIED_SCRATCH setting and the weight's type (llama.cpp-8ony). `env_mode` is the parsed
// variable: negative when unset, 0 when it turns the scratch off, positive when it turns it on for every type.
// `default_type` is the type's default (Q4_0, Q8_0 and MXFP4). Unset, only the default types are supplied; set to
// 0, none; set non-zero, all. Pure.
bool zone_onednn_pp_scratch_type_enabled(int env_mode, bool default_type);

// "The oneDNN PP scratch supplies this op's f16 copies": the op passes the PP admission, the scratch is enabled for
// its type, and the pair is planned into the ONEDNN zone (zone_onednn_pp_scratch_planned). This is the ONE question
// the op arm, acquire_onednn_pp_scratch and the graph-entry walk must answer the same way: an op the scratch does
// not supply draws the planned dequant buffers, and the walk sizes those only for the ops it also says are not
// supplied (a K-quant weight, or any op under UNIFIED_SCRATCH=0, was skipped by the walk and refused by acquire).
// Pure.
bool zone_onednn_pp_scratch_supplies(bool   pp_candidate,
                                     bool   type_enabled,
                                     bool   arena_active,
                                     size_t pair_bound_bytes,
                                     size_t weights_bytes,
                                     size_t activations_bytes);

// Whether the unified kernel's oneDNN f16 route (the "Route A" of the unified dispatch) draws the planned dequant
// buffers for a node: the router picked the unified kernel, the type is one the unified kernel serves, src1 is
// plain (contiguous, not transposed or permuted), the node passes the PP admission, and the oneDNN scratch does
// NOT supply its pair (zone_onednn_pp_scratch_supplies). That is the over-zone LM head or tied embedding of a
// Q4_0 / MXFP4 model, which took a per-op pool copy of the whole weight that no plan sized (llama.cpp-8ony).
// Pure.
bool zone_unified_pp_draws_dequant(bool primary_unified,
                                   bool unified_type,
                                   bool src1_plain,
                                   bool pp_candidate,
                                   bool scratch_supplies);

// Whether the graph-entry walk counts a node toward the planned f16 dequant buffers. Two arms can draw them and they
// do not share a precision condition: the legacy f16 arm only runs for GGML_PREC_DEFAULT, the unified kernel's
// oneDNN f16 route has no precision check at all. A walk that filtered every node on precision first would leave
// an F32-precision node the unified route draws for unsized (growth, or a plan-breach abort where the old pool copy
// degraded silently). Pure.
bool zone_walk_f16_node_draws(bool prec_default, bool legacy_route_draws, bool unified_route_draws);

// ---------------------------------------------------------------------------
// The planned dense scratch as ONE reservation (llama.cpp-kpjw)
// ---------------------------------------------------------------------------
//
// The three buffers above (Q8_1 src1, f16 src0, f16 src1) live in the RUNTIME zone. Two things were
// missing when they were only COUNTED in the zone requirement:
//
//   * They were sized at the load-time n_ubatch. The runtime n_ubatch (auto-ubatch picks it at context
//     creation) can be 4x larger, so the plan is a function of n_ubatch and the runtime-context
//     transaction must re-derive it. These helpers are the pure arithmetic of that re-derivation.
//   * Nothing RESERVED them. A compute buffer asks the same zone, may spill, and fills it before the first
//     graph materializes the planned buffer, which is then refused with 0.3 MB free. The hold is the part of
//     the plan a spill-capable allocation must leave alone.
//
// Pure: no state, no log.

// Total bytes of the three planned buffers at `n_ubatch`: exactly the sum of zone_mmq_src1_scratch_bytes and
// both zone_dequant_f16_plan_bytes figures. False on overflow.
bool zone_dense_scratch_total_bytes(size_t   mmq_bytes_per_token,
                                    size_t   f16_weight_bytes,
                                    size_t   f16_src1_bytes_per_token,
                                    uint32_t n_ubatch,
                                    size_t * out);

// Largest n_ubatch, a multiple of 32 no larger than `search_max`, whose planned dense scratch plus
// `other_runtime_bytes` fits `capacity_bytes`. Zero when not even 32 rows fit.
uint32_t zone_dense_scratch_largest_ubatch(size_t   mmq_bytes_per_token,
                                           size_t   f16_weight_bytes,
                                           size_t   f16_src1_bytes_per_token,
                                           size_t   other_runtime_bytes,
                                           size_t   capacity_bytes,
                                           uint32_t search_max);

// One planned buffer: its plan figure and the bytes its backing holds now.
struct zone_planned_buffer {
    size_t plan     = 0;
    size_t capacity = 0;
};

// Bytes of the RUNTIME zone a spill-capable allocation must leave free: the whole plan of every buffer whose
// backing is still short of it. The whole plan, not the shortfall, because growth allocates the replacement
// while the old backing is still live (it retires behind a queue marker). Zero once every buffer holds its plan.
// False on overflow.
bool zone_planned_scratch_hold_bytes(const zone_planned_buffer * buffers, size_t count, size_t * out);

// Whether a spill-capable RUNTIME request of `size` bytes may take the zone's `available` bytes while `hold`
// bytes are held. A request that does not may spill exactly as one does when the zone is full.
bool zone_runtime_alloc_respects_hold(size_t available, size_t hold, size_t size);

// The decision unified_alloc takes for a request that prefers a zone: true when the request must NOT be served
// from the zone although the zone could serve it, because the hold keeps those bytes for the planned scratch.
// Only a spill-capable request for the RUNTIME zone is ever held back; a forbid-spill request is one of the
// planned consumers the hold exists for, and no other zone has a hold. A request larger than the zone's free
// bytes is NOT held back: it spills as it always did and meets the allocator's raw gate like any raw request. The
// zone's free bytes come first, then the hold, then the request size: swapped, the same numbers answer a
// different question.
bool zone_runtime_alloc_held_back(bool   runtime_zone,
                                  bool   forbid_spill,
                                  size_t zone_available,
                                  size_t hold,
                                  size_t alloc_size);

// The arena's grain: it reserves every zone as a whole number of these bytes. It is the zone allocator's block grain
// (tlsf_allocator::block_grain), which this dependency-free header cannot include, so unified-cache.cpp asserts the
// two are equal.
constexpr size_t k_zone_arena_alignment = 256;

// The RUNTIME zone the arena reserves for a plan into *out: the larger of `floor_bytes` and `requirement`, rounded up
// to k_zone_arena_alignment. False, leaving *out untouched, when the rounding would overflow. The arena's reserve, the
// zone the cache plans, the room an explicit micro-batch plans against, and the check for a zone an earlier load grew
// all ask this one function, so the zone a plan counts on and the zone the arena holds cannot round apart. Inline so
// that the host-only admission unit, which links no zone-sizing code, asks it too.
inline bool zone_runtime_reserved_bytes(size_t floor_bytes, size_t requirement, size_t * out) {
    const size_t planned = requirement > floor_bytes ? requirement : floor_bytes;
    if (planned > SIZE_MAX - (k_zone_arena_alignment - 1)) {
        return false;
    }
    *out = (planned + k_zone_arena_alignment - 1) & ~(k_zone_arena_alignment - 1);
    return true;
}

// Whether a live RUNTIME zone of `live_zone_bytes` is larger than the zone the arena reserves for a plan of
// `floor_bytes` and `requirement` (zone_runtime_reserved_bytes). The arena keeps a zone larger than a plan needs and
// rebuilds a zone only while it holds nothing, so a zone kept above a load's plan was grown by an earlier load in the
// process, and the load's routed experts are planned against it. A zone the load sized for itself, as every first
// load of a process does, is the zone its plan reserves and never qualifies; nor does any zone when the plan's
// rounding overflows.
bool zone_runtime_kept_above_plan(size_t live_zone_bytes, size_t floor_bytes, size_t requirement);

// The RUNTIME zone's planned terms, as unified_cache_get_planned_runtime_zone_requirement() reads them for the zone
// ensure_planned_arena_zones() sizes. Each is the term as published now, so the requirement follows whatever was
// published last: a term left behind by a plan that was then discarded is still asked for.
struct zone_runtime_terms {
    size_t base             = 0;  // PP pipeline scratch + the ring's RUNTIME part + MoE CONTROL, checked as one sum
    size_t moe_host_scatter = 0;  // the host-expert MoE scatter scratch, published by the placement plan
    size_t mmq_src1         = 0;  // the dense MMQ/MMVQ Q8_1 src1 buffer
    size_t dequant_f16      = 0;  // the dense f16 dequant buffer
    size_t compute          = 0;  // the compute buffer the load measured
    size_t state            = 0;  // the context memory the load measured on the plain buffer type
};

// The RUNTIME zone's requirement: the sum of `terms`. False, with *out unwritten, when the sum would wrap or `out` is
// null: a wrapped sum would size a smaller zone instead of refusing one.
bool zone_runtime_requirement(const zone_runtime_terms & terms, size_t * out);

// What points into a VRAM arena that a plan wants larger. The arena grows only by being destroyed and reserved again,
// which moves every byte in it, so each of these keeps it as it is. The last four are not holders: they count the
// cache's own releases still to run, whose bytes the zones still count as allocated with no owner left, so the remedy
// can tell those bytes from a model's or a context's. No gate reads them.
struct zone_arena_holders {
    size_t   live_zone_bytes  = 0;  // bytes the zones count as allocated
    uint32_t chunk_leases     = 0;  // leases handles hold on the arena's chunks
    uint32_t scratch_in_arena = 0;  // the cache's scratch buffers whose pointer lies inside the arena
    uint32_t scratch_outside  = 0;  // the cache's scratch buffers allocated outside it, printed and never a holder
    uint32_t ring_in_arena    = 0;  // of scratch_in_arena, the PP MoE ring's slot buffers (current and retired slots)
    uint32_t live_beside_ring = 0;  // allocations live in the zones other than the ring's slot buffers
    size_t   queued_frees     = 0;  // frees queued in the arena and not yet run
    size_t   release_retries  = 0;  // the device's releases refused once and waiting for a retry
    size_t   retired_pending  = 0;  // the cache's retired entries not yet finalized
    size_t   evictions        = 0;  // the cache's evictions still in flight
};

// Whether the arena may be rebuilt for larger zones: only when nothing points into it. A scratch buffer allocated
// outside the arena keeps its own memory through a rebuild, so it does not refuse one; a buffer inside it does.
bool zone_arena_rebuild_allowed(const zone_arena_holders & holders);

// Whether the PP MoE ring's slot buffers are all that keeps the arena from a rebuild it needs: a rebuild is needed, a
// slot buffer of the ring lies inside the arena, and nothing else holds it (no other allocation live in its zones, no
// lease on its chunks, no other cache scratch buffer inside it). Only then may a load release the ring to rebuild.
bool zone_arena_only_ring_holds(bool rebuild_needed, const zone_arena_holders & holders);

// Why the arena's planned zones could not be sized.
enum class zone_arena_refusal_reason {
    NONE,              // they were sized
    REBUILD_HELD,      // a zone must grow and something points into the arena, so it cannot be rebuilt
    RESERVE_FAILED,    // the device could not reserve an arena with the planned zones
    NO_CACHE,          // the device has no unified cache to size them in
    TEARDOWN_REFUSED,  // the arena could not be torn down, by this call's rebuild or by an earlier one
};

// Why a teardown of the VRAM arena refused, as the teardown records it. Every cause but HOST_STATE is found after the
// teardown closed the arena to new allocations, and it stays closed until a later teardown completes.
enum class zone_arena_teardown_refusal {
    NONE,              // no teardown was refused
    HOST_STATE,        // the host could not reserve the teardown's own bookkeeping; the arena is as it was
    LIVE_ALLOCATIONS,  // allocations were still live in the arena when the teardown closed it
    DRAIN_SNAPSHOT,    // the host could not record the allocations the teardown waits on
    DRAIN_TIMEOUT,     // a lease on one of its allocations did not end while the teardown waited
    FREE_FAILED,       // the device refused to free one of its chunks, after the chunks before it were freed
};

// Whether a teardown refused for `cause` left the arena closed to new allocations.
bool zone_arena_teardown_left_closed(zone_arena_teardown_refusal cause);

// The three zones a plan sizes, in bytes.
struct zone_arena_sizes {
    size_t scratch = 0;
    size_t onednn  = 0;
    size_t runtime = 0;
};

// A refusal to size the arena's planned zones: the device, the reason, and per zone the bytes the plan needs and the
// bytes the device's arena has when the decision refuses (have is 0 where no arena exists then). At REBUILD_HELD,
// holders is what the decision found pointing into the arena, which the line names and chooses its remedy from. At
// RESERVE_FAILED, released is the zones of the arena a rebuild destroyed before the reserve failed. At
// TEARDOWN_REFUSED, teardown is why the teardown refused. Each is 0 or NONE otherwise. keep_deferred is true at
// REBUILD_HELD when the PP MoE ring alone holds the arena, the caller asked to release it first
// (zone_arena_ensure_options::ring_release_first), and the zones would otherwise have been kept
// (zone_arena_shortfall_is_reservation_only): asked again without that request, they are kept.
struct zone_arena_refusal {
    zone_arena_refusal_reason   reason = zone_arena_refusal_reason::NONE;
    int                         device = -1;
    zone_arena_sizes            need;
    zone_arena_sizes            have;
    zone_arena_holders          holders;
    zone_arena_sizes            released;
    zone_arena_teardown_refusal teardown      = zone_arena_teardown_refusal::NONE;
    bool                        keep_deferred = false;
};

// How a caller asks for the arena's planned zones (unified_cache::ensure_planned_arena_zones).
struct zone_arena_ensure_options {
    // The caller releases the PP MoE ring and asks again when the ring alone holds an arena that must grow (the load
    // stage's entry ensure). A rebuild gives the load its whole Graph floor, and keeping a short ONEDNN zone is the
    // fallback for a rebuild something live blocks, so while that last resort is still to come no zone is kept: the
    // call refuses as it would without the keep and sets the refusal's keep_deferred.
    bool     ring_release_first = false;
    // The load transaction the call is part of, 0 outside a staged load. A keep of a short ONEDNN zone warns once per
    // load transaction and device (zone_kept_onednn_warn_due), and at every keep for 0.
    uint64_t load_txn           = 0;
};

// Whether a rebuild of the arena that something live blocks can be answered by keeping the zones as they are: SCRATCH
// and RUNTIME hold what the plan needs, the ONEDNN zone does not, and it still holds `onednn_pair_bytes`, the pair it
// must keep room for (zone_onednn_kept_pair_bytes). The primitive-API pair has no DIRECT fallback, but the oneDNN
// Graph scratch floor is a reservation, not a requirement: what the zone cannot hold of it is served on the DIRECT path
// under its cap, or a context needing it is refused by name. A zone that is not short is not a keep (the caller has
// nothing to decide). Pure.
bool zone_arena_shortfall_is_reservation_only(const zone_arena_sizes & have,
                                              const zone_arena_sizes & need,
                                              size_t                   onednn_pair_bytes);

// The pair bytes an ONEDNN zone kept short must keep room for: the pair held in the zone now (`held_pair_bytes`, 0 when
// none is, or when it lies outside the arena) or the pair plan `kept` describes, whichever is larger. Pure.
size_t zone_onednn_kept_pair_bytes(const zone_onednn_plan & kept, size_t held_pair_bytes);

// The snapshot to store for an ONEDNN zone of `capacity_bytes` kept short for `live` while `held` describes it: the
// pair of zone_onednn_plan_keep, and a Graph floor of what the zone holds beyond that pair (capacity less
// zone_onednn_kept_pair_bytes, 0 when the pair fills it), never above the larger of the two plans' floors. Not
// zone_onednn_plan_keep's maximum floor: that would raise the stored floor above what the kept zone holds, the pair
// bound (zone_onednn_pp_pair_bound) would fall with it, and a pair already held could then exceed the bound, so the
// Graph room a context check reads (capacity less the bound) would be larger than the zone has. Over the stored
// snapshot the bound is the capacity less the stored floor, never below the kept pair bytes (equal to them when the
// capacity limits the floor), so the Graph room is the stored floor. Pure.
zone_onednn_plan zone_onednn_plan_keep_short(const zone_onednn_plan & held,
                                             const zone_onednn_plan & live,
                                             size_t                   capacity_bytes,
                                             size_t                   held_pair_bytes);

// The budget snapshot the DIRECT Graph-scratch cap is derived from (onednn_graph_scratch_direct_cap), stored when the
// zones are found sufficient or kept short: it never rises while the arena is kept, so a cap that rose with the budget
// cannot raise the Graph floor of a zone nothing rebuilt. `held_bytes` 0 is unset, and `current_bytes` is stored. A
// rebuilt arena stores the current figure outright. Pure.
size_t zone_graph_direct_cap_snapshot_kept(size_t held_bytes, size_t current_bytes);

// Whether a call that kept an ONEDNN zone short logs its WARN: once per load transaction and device. `latched_txn` is
// the load the device's last WARN was for. A load's stage plans in two passes, and a micro-batch's grow leg asks for
// the zones again, so one kept zone would otherwise warn up to three times. Every call of a staged load carries its
// transaction, a test's staged load included. A call outside a staged load (`load_txn` 0: a direct planner entry, a
// test that asks the cache itself, a context's best-effort re-plan, the oneDNN pair's reserve) has no load to latch
// on, and warns at every keep. Pure.
bool zone_kept_onednn_warn_due(uint64_t latched_txn, uint64_t load_txn);

// The line a model load refused for `refusal` logs, without a newline: the device, each zone's need and have in MB, the
// shortfall of each zone that is short (a zone short by under 0.1 MB prints its three figures in KB, or in bytes under
// 1 KB, so no shortfall prints as nothing), the reason, and what the caller can do. Empty for NONE. The zones are
// planned at the load's fixed planning shape, not at the context's -c or -ub, so the line never offers a smaller
// context or micro-batch as the remedy.
std::string zone_arena_refusal_text(const zone_arena_refusal & refusal);

// The worst-case bytes the hold can push outside the arena at candidate rung `n_ubatch`: the plan (the most the
// hold can be) plus the largest spill-capable RUNTIME request, because a held-back request spills whole. The
// request was observed at `hwm_n_ubatch`; compute buffers scale about linearly with n_ubatch, so it is scaled to
// the candidate (up or down) when both are known. No plan means no hold and the bound is 0. Saturating. The
// request term is a heuristic (it is what a previous rung asked), a lower bound before any rung has reserved.
size_t zone_hold_spill_bound(size_t plan, size_t request_hwm, uint32_t hwm_n_ubatch, uint32_t n_ubatch);

// An ESTIMATE of the part of a worst-case spill (zone_hold_spill_bound) that lands OUTSIDE the arena. A compute
// buffer the RUNTIME zone will not serve is placed in the arena's KV zone first
// (zone_runtime_spill_prefers_kv_zone), so only what the KV zone cannot take is raw device memory, the thing that
// eats the driver headroom. `kv_zone_free` is what a compute buffer can count on in that zone NOW
// (zone_kv_room_for_compute: its largest free block, net of the KV this context has yet to place). It is an
// estimate in three ways: other buffers may take that room before the spill does, a spill is several buffers and the
// room is one block, and the spill figure it is subtracted from is itself a heuristic. The realized check, run once
// the rung's buffers exist, asks the same bound with the rung's measured request; the RAW spills the rung made reach
// only its refusal text, so this estimate is never checked against them.
size_t zone_hold_spill_raw_demand(size_t spill_bound, size_t kv_zone_free);

// The KV-zone room a compute buffer can count on. The runtime-context transaction publishes BEFORE this context's KV
// cache exists (a pinned -ub, the first rung), so the zone still shows free the bytes its own KV is about to take:
// `kv_pending_bytes`, the KV this transaction's plan places that is not live yet, is not room. A buffer is
// indivisible, so the room is a block, `kv_largest_free`, never the sum of the zone's free bytes. Clamped at 0;
// KV already live (the recheck, a settle) passes 0 pending.
size_t zone_kv_room_for_compute(size_t kv_largest_free, size_t kv_pending_bytes);

// llama.cpp-kpjw (kpjw-g7, P4: one fact, one source): ONE predicate for "does this demand leave the card its headroom".
// `free_before` is the card WITHOUT the demand in place, `bound` the demand. The hold is blamed only when ITS demand
// is what pushes the card under the driver headroom the arena expects outside itself (free_before >= headroom and
// free_before - bound < headroom): a card already short without the demand (a full B70 with KB-scale spills) is not
// the hold's doing, and a rung with no demand is never refused. A rung that is refused runs the card out of
// resources at its first graph (B50, Qwen PPL at auto-ub1024: 107.8 MB left against 256 MB, flash attention out of
// resources). It is the verdict zone_hold_fit gives.
bool zone_hold_spill_bound_fits(size_t free_before, size_t headroom_target, size_t bound);

// What the hold-spill fit is asked of: the compute buffers of ONE rung (an n_ubatch) are, or would be, placed while
// the planned dense scratch is held out of the RUNTIME zone. The transaction-time bound (F3), the auto-ubatch
// trial's per-rung realized check and the context-init check on a pinned -ub all call zone_hold_fit over these
// inputs, so they cannot disagree about one fact. A pure function of the inputs: it reads nothing and remembers
// nothing, so the same rung asked twice, with other rungs' history in between, answers the same.
//
// One rung's measured compute-buffer request: the largest request the scheduler's compute buffers made while
// the rung's plan was the published one. Kept per rung (n_ubatch), not as a running maximum, so a rung's verdict
// does not depend on which other rungs ran before it.
struct zone_hold_rung_request {
    uint32_t n_ubatch = 0;
    size_t   bytes    = 0;
};

// The planned dense scratch (the most the hold can be) at a rung.
typedef size_t (*zone_hold_plan_fn)(void * ctx, uint32_t n_ubatch);

// ring_bytes is the prefill ring's bytes neither the planned room nor a loan covers (llama.cpp-kjef): the ring's slots
// are the room's bytes and lent bytes, so it is 0 when the two supply all of it, and a demand on free memory otherwise.
//
// compute_room_n_ubatch and compute_room_bytes name the rung whose compute buffers the plan sized the RUNTIME zone to
// hold beside the hold, and that room (placement_plan_compute_room): the explicit micro-batch the load placed, or, for
// a load that named none, the micro-batch the load measured the term at (the planning micro-batch only for an
// inventory that sent none). The room is the planned compute term (the compute buffers the load measured at that
// shape), held beside that micro-batch's ring and dense scratch. 0 for none: a multi-device plan, another device, an
// unknown measured micro-batch, or a named micro-batch the plan did not place.
// A rung can also be named with a room of 0 bytes, when no live load owns a compute term on the device (the record
// keeps the last load's micro-batch until the next load replaces it): such a room credits no request above 0 bytes,
// and the measured rung is withheld (0) with it.
//
// compute_room_measured_n_ubatch is the micro-batch the term was measured at, 0 when the record names none (no load
// recorded one, or two loads that both held a term disagreed: that mark outlives both loads and stays until the next
// inventory intake that finds no live term), when the record changed between the fit's two reads of it, or when the
// room is none or 0 bytes.
// It is a separate fact from compute_room_n_ubatch because the plan can place a micro-batch above the one the load
// measured its term at: the room's rung then says where the zone was sized, not which rung's buffers the term covers.
struct zone_hold_fit_inputs {
    size_t                         headroom_target = 0;  // the driver headroom the arena expects outside itself
    size_t                         free_before = 0;  // the cache's ledger: the card with the rung's own buffers gone
    size_t                         kv_room     = 0;  // zone_kv_room_for_compute, already net of the KV pending
    const zone_hold_rung_request * rungs       = nullptr;
    size_t                         n_rungs     = 0;
    zone_hold_plan_fn              plan_of     = nullptr;
    void *                         plan_ctx    = nullptr;
    size_t                         ring_bytes  = 0;
    uint32_t                       compute_room_n_ubatch          = 0;
    size_t                         compute_room_bytes             = 0;
    uint32_t                       compute_room_measured_n_ubatch = 0;
    // The rung whose own compute buffer the hold kept outside the arena, 0 for none: the hold epoch's rung (the asked
    // rung when the owner has no epoch), set when the epoch's spill counters show a request held outside the arena.
    // The zone did not serve that rung's buffer, so the room is not credited there. Every other rung keeps the credit
    // the plan gives it.
    uint32_t                       held_out_n_ubatch              = 0;
};

// The worst-case compute-buffer request at `n_ubatch`: the rung's own measured record when it has one (and only
// that, whatever else is recorded); otherwise the largest record scaled linearly to the rung (compute buffers grow
// about linearly with n_ubatch), so a set of records gives one answer whatever order it was made in. 0 when nothing
// is recorded: the demand is then the plan alone, a lower bound.
size_t zone_hold_fit_request(const zone_hold_fit_inputs & in, uint32_t n_ubatch);

// Whether the RUNTIME zone was planned to serve the rung's compute buffers beside the hold. Only a rung at or below
// compute_room_n_ubatch is, the rung the zone was sized for. With its own measured request (`own_record`) the rung is
// served when the request is no larger than the room (whole or nothing). Without one it is served only when it is also
// at or below compute_room_measured_n_ubatch: the room is the term measured at that rung and compute buffers grow with
// the micro-batch, so a rung no larger needs no more than the term, while a rung above it (below a placed micro-batch)
// can need more. A measured rung of 0 gives no such credit. A request scaled from another rung is an estimate and does
// not refuse a rung the term covers. Never the rung whose own buffer was held out, never a rung above the room's,
// never n_ubatch 0.
//
// It can over-credit a rung whose compute buffers span several chunks, because the request is the rung's largest
// single buffer (zone_hold_fit_request) while the room is a sum of per-chunk peaks that the other chunks draw on too.
// The bound has the same limit, since it charges one request rather than the rung's chunks. The credit at the
// micro-batch the load measured the term at (placement_plan_compute_room) shares the limit, since its room is the same
// summed term.
bool zone_hold_fit_rung_has_compute_room(const zone_hold_fit_inputs & in,
                                         uint32_t                     n_ubatch,
                                         size_t                       request,
                                         bool                         own_record);

// Whether the rung has a measured request of its own among the records.
bool zone_hold_fit_rung_has_record(const zone_hold_fit_inputs & in, uint32_t n_ubatch);

// The raw (outside-arena) demand the hold can cause at the rung: zone_hold_spill_raw_demand over
// zone_hold_spill_bound(plan, the rung's request) and the KV room, plus the ring bytes neither the planned room nor a
// loan covers (saturating). A request the zone was planned to serve (zone_hold_fit_rung_has_compute_room) is never
// held back, since the hold keeps a request out only when the zone's free bytes less the hold are short of it, so
// it adds nothing to the bound. 0 for an unknown n_ubatch.
size_t zone_hold_fit_demand(const zone_hold_fit_inputs & in, uint32_t n_ubatch);

// The terms zone_hold_fit_demand is made of at `n_ubatch`, for a refusal to print. `request` is the rung's request
// whether or not the compute room credits it; `request_credited` says the demand left it out. `demand` is the value
// zone_hold_fit_demand returns for the same inputs.
struct zone_hold_fit_terms {
    size_t plan             = 0;
    size_t request          = 0;
    bool   request_credited = false;
    size_t kv_room          = 0;
    size_t ring_uncovered   = 0;
    size_t demand           = 0;
};

zone_hold_fit_terms zone_hold_fit_terms_at(const zone_hold_fit_inputs & in, uint32_t n_ubatch);

// Whether the rung fits: zone_hold_spill_bound_fits(free_before, headroom, demand).
bool zone_hold_fit(const zone_hold_fit_inputs & in, uint32_t n_ubatch);

// The -ub a refusal names: `n_ubatch` itself when it fits, otherwise the largest power of two (at least 32) not above
// it that zone_hold_fit accepts, over the same inputs. 0 when none does or n_ubatch is unknown. By construction the
// number printed passes the fit, and twice it does not whenever twice it is a rung that was asked (not above
// `n_ubatch`; a non-power-of-two n_ubatch such as 600 names 512 or lower and never asks 1024). It can be under the
// auto-ubatch descent's floor of 64: it is the largest the function accepts, not a rung the descent walks.
uint32_t zone_hold_fit_largest_ub(const zone_hold_fit_inputs & in, uint32_t n_ubatch);

// The cache's ledger of free memory, in two steps so that no driver read after a release is ever needed (the
// driver's credit for a freed buffer lags: 602.7 MB was read where 1097 MB was true). `cold` is the free memory the
// card would show with no outside-arena cache allocation live: the driver's reading plus the bytes the cache holds
// live outside the arena at that moment (taken once per context). `before` is the free memory with `persistent_raw`
// of those still live (everything but the rung's own buffers). Saturating; never below zero.
size_t zone_hold_free_cold(size_t driver_free, size_t raw_live);
size_t zone_hold_free_before(size_t cold, size_t persistent_raw);

// The context shape a plan sizes its per-context scratch for (the oneDNN Graph attention floor).
// At load (`n_ctx_is_runtime` false) the plan's own shape is the load's planning shape, so the shape is the context the
// model opens with (`n_ctx_context`: the one the caller named, or the model's default) and the micro-batch the caller
// named (`n_ubatch_context`), each falling back to the plan's own when 0 (no request reached the load, or no
// micro-batch was named). At a context transaction the plan's own shape is the context's. The micro-batch never exceeds
// the context.
struct zone_context_shape {
    uint32_t n_ctx    = 0;
    uint32_t n_ubatch = 0;
};

zone_context_shape zone_plan_context_shape(bool     n_ctx_is_runtime,
                                           uint32_t plan_n_ctx,
                                           uint32_t n_ctx_context,
                                           uint32_t plan_n_ubatch,
                                           uint32_t n_ubatch_context);

// The planning micro-batch a plan is built at: the envelope's when the caller's envelope names one
// (`envelope_n_ubatch` != 0), the model's (`n_ubatch`) otherwise. The plan builders take their planning shape from it,
// and zone_plan_context_shape_for_envelope() below its default micro-batch.
uint32_t zone_planning_n_ubatch(uint32_t envelope_n_ubatch, uint32_t n_ubatch);

// zone_plan_context_shape() over the planner's own inputs, before a plan exists: the planning context `n_ctx`, and the
// planning micro-batch zone_planning_n_ubatch() gives. A shape published before the plan is built is then the shape the
// plan itself sizes for.
zone_context_shape zone_plan_context_shape_for_envelope(bool     n_ctx_is_runtime,
                                                        uint32_t n_ctx,
                                                        uint32_t n_ctx_context,
                                                        uint32_t envelope_n_ubatch,
                                                        uint32_t n_ubatch,
                                                        uint32_t n_ubatch_context);

// The oneDNN Graph attention scratch the ONEDNN zone holds: the modeled peak of one execute,
// max(n_head_ctx_max * n_ctx, n_head_swa_max * min(n_ctx, n_swa + n_ubatch)) * n_ubatch f32 elements, times 3/2,
// bounded by `direct_cap` (the dispatcher splits an execute to half the cap, so a full chunk and the last chunk of a
// split execute fit under it together; when one query row alone exceeds half the cap the chunks are one row each and
// can hold more), and at least 64 MiB. `direct_cap` 0 is no cap known: the formula.
// A sliding-window head sees n_swa + n_ubatch keys, not n_swa: the first query of the micro-batch looks n_swa keys
// back and the last one n_ubatch further along. The two head classes are never outstanding together, so the peak is
// the larger class, not their sum. The measurements the formula was fit to are with its caller in unified-cache.cpp.
size_t zone_onednn_graph_scratch_floor_bytes(uint32_t n_head_ctx_max,
                                             uint32_t n_head_swa_max,
                                             uint32_t n_swa,
                                             uint32_t n_ubatch,
                                             uint32_t n_ctx,
                                             size_t   direct_cap);

// The part of a context's oneDNN Graph scratch the ONEDNN zone's Graph room cannot hold: served outside the arena.
size_t zone_onednn_graph_uncovered_bytes(size_t demand, size_t graph_room);

// The largest power of two (at least 32, not above n_ubatch) whose Graph scratch fits: either the zone's Graph room
// holds all of it (nothing drawn outside the arena, so `free_outside`, `headroom` and `hold_spill` do not count; the
// hold check judges its own spill), or the part above the room plus the hold's worst-case spill leaves
// `free_outside` at least `headroom`. n_ubatch itself when it fits; 0 when no rung does. A card already under the
// headroom still gets the largest rung inside the room.
uint32_t zone_onednn_graph_fit_largest_ub(uint32_t n_head_ctx_max,
                                          uint32_t n_head_swa_max,
                                          uint32_t n_swa,
                                          uint32_t n_ubatch,
                                          uint32_t n_ctx,
                                          size_t   direct_cap,
                                          size_t   graph_room,
                                          size_t   free_outside,
                                          size_t   headroom,
                                          size_t   hold_spill);

// llama.cpp-kpjw (r7 I3: identity by ORIGIN, not by timing): what stays live without the rung is every raw byte the
// cache holds except the rung's OWN scheduler compute buffers (rows registered by a request flagged
// scheduler_compute), and only when the rung's buffers are live (the realized check). A raw byte allocated after an
// epoch began that is not one of those (the recurrent state, made after a pinned -ub's one publish) is persistent
// however late it came; a transaction (no live rung) counts every held raw byte. `compute_live` is clamped to
// `raw_held`.
size_t zone_hold_persistent_raw(size_t raw_held, size_t compute_live, bool rung_live);

// llama.cpp-kpjw (r7 I2): the baseline `cold` reading of a window (the span between two publishes). The first reading
// of a window stands; each later one can only RAISE it: the driver's credit for a freed buffer lags, and a lag only
// ever lowers a reading, so the maximum is the reading with the least lag. A new window (a publish, a quiescent
// point) forgets the previous one: another tenant may have arrived since.
size_t zone_hold_cold_update(bool have_baseline, size_t baseline, size_t candidate);

// llama.cpp-kpjw (r7 I3): the KV room the fit judges with. A rung that is live is judged with the room its own epoch
// began with (the rung's own KV-zone placements have used the zone since); any other asker, and a live rung with no
// epoch yet, reads the zone as it is now.
size_t zone_hold_pick_kv_room(bool rung_live, bool have_epoch, size_t epoch_kv_room, size_t live_kv_room);

// llama.cpp-kpjw (r7 I1): the demand the non-FA headroom check compares with the card: the non-FA attention scratch
// plus the hold's worst-case spill (zone_hold_fit_demand). Saturating, so a wrapped sum never reads as a small demand.
size_t zone_hold_nonfa_demand(size_t nonfa_scratch, size_t hold_spill);

// Whether a RUNTIME-zone request goes to the KV zone instead of the zone / raw device memory: only a request the
// caller marked as a compute buffer (`compute_spill_flag`), spill-capable (not `forbid_spill`), that the RUNTIME zone
// will not serve (`zone_misses`: held back by the hold, or larger than the zone's free bytes), and only when the KV
// zone can hold it whole. Any other request class keeps the pre-existing path unchanged.
bool zone_runtime_spill_prefers_kv_zone(bool   compute_spill_flag,
                                        bool   runtime_zone,
                                        bool   forbid_spill,
                                        bool   zone_misses,
                                        size_t kv_zone_free,
                                        size_t alloc_size);

// Whether a multi-row MUL_MAT draws a given planned scratch (the Q8_1 src1 buffer, the f16 dequant buffers),
// from the two answers the dispatch can give. `primary_*` is the router's first decision; when it picks the
// unified kernel the dispatch can still decline at run time and re-select a legacy kernel, which is
// `fallback_*`. A node the unified kernel serves draws neither buffer; a node it declines draws what the
// legacy kernel draws. One function for both buffers: two predicates for one fact eventually disagree.
bool zone_route_draws_scratch(bool decision_valid,
                              bool primary_is_unified,
                              bool primary_draws,
                              bool fallback_valid,
                              bool fallback_draws);

// llama.cpp-p6i0: the RUNTIME-zone bytes a scheduler compute buffer measured as `chunk_bytes[0..n_chunks)` takes.
// Each gallocr chunk is one backend buffer and so one allocation in the zone, which occupies
// tlsf_allocator::round_request(chunk, alignment) -- the zone allocator's own rule, read from it, not restated. A zero
// chunk allocates nothing (ggml_backend_buft_alloc_buffer answers a size-0 request without the backend). No headroom
// is added. False, with *out untouched, when the sum or a rounding would wrap: never a smaller term.
bool zone_compute_term_bytes(const uint64_t * chunk_bytes, size_t n_chunks, size_t alignment, size_t * out);

// The micro-batch the planned compute term was measured at, as the loads on a device record it, one store per
// inventory intake (a load's early plan and late intake carry the same inventory, and at its late intake the load's
// own term is live and the record already holds its value, so the repeat store keeps an equal value and changes
// nothing): the hold-spill fit reads it to place its compute room, and zone_hold_fit_rung_has_compute_room decides
// which rungs that room credits. The first load's value stands while later loads agree. A load that names a different
// one while another load still holds a compute term on the device turns the record off
// (k_zone_compute_term_n_ubatch_disagreed), because a credit at a shape the live term was not measured at is the
// defect the record prevents. When no load holds a term (`term_live` false: the merged term is 0 bytes), the earlier
// loads are gone and the new load's value replaces the record, even one turned off, so loads made one after another
// at different micro-batches each get their own credit. A load's early plan is taken in before its own term is
// reserved, so `term_live` there names only earlier loads. A load naming 0, or naming the mark itself, changes
// nothing, so the mark only ever means two live loads really disagreed and the WARN that reports it names two values
// loads did plan. A micro-batch left behind after every load is gone is harmless: the term merges to 0 bytes, and a
// 0-byte room credits nothing. So a disagreement only removes the credit, never grants one, for every term on the
// device: a load's measure can reserve a term on a device only after that device took in the load's inventory, since
// the load's stage runs the intake on every SYCL device.
constexpr uint32_t k_zone_compute_term_n_ubatch_disagreed = UINT32_MAX;

inline uint32_t zone_compute_term_n_ubatch_store(uint32_t stored, uint32_t n_ubatch, bool term_live) {
    if (n_ubatch == 0 || n_ubatch == k_zone_compute_term_n_ubatch_disagreed) {
        return stored;
    }
    if (!term_live) {
        return n_ubatch;
    }
    if (stored == k_zone_compute_term_n_ubatch_disagreed) {
        return stored;
    }
    return stored == 0 || stored == n_ubatch ? n_ubatch : k_zone_compute_term_n_ubatch_disagreed;
}

// Whether a store moved the record from `before` to `after` by turning it off: the one store the setter WARNs on. A
// record already off that stays off does not turn off again, so a later disagreeing load does not repeat the WARN.
inline bool zone_compute_term_n_ubatch_turns_off(uint32_t before, uint32_t after) {
    return before != k_zone_compute_term_n_ubatch_disagreed && after == k_zone_compute_term_n_ubatch_disagreed;
}

// The micro-batch a record names: 0 for none, and for one that disagreement turned off.
inline uint32_t zone_compute_term_n_ubatch_read(uint32_t stored) {
    return stored == k_zone_compute_term_n_ubatch_disagreed ? 0 : stored;
}

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
// Since llama.cpp-ndn9 the second cause has a second source, and it is NOT
// fragmentation: the superseded reservation's bytes are released behind a
// barrier rather than immediately, so a growth step transiently occupies
// old+new. Still not a predicate defect, so it stays out of the under-estimate
// counter — but it does mean the ONEDNN zone wants headroom for one superseded
// reservation above the largest single one, and that a run can report
// zone-large-enough sub-allocation failures without the zone being fragmented
// at all.
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
// The batched form, for a caller that already counts its own uses: one mutex take for `count` observations.
void   zone_sizing_record_observations(const char * path, size_t count);
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
