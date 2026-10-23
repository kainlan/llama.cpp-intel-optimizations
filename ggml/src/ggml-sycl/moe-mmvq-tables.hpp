//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MOE_MMVQ_TABLES_HPP
#define GGML_SYCL_MOE_MMVQ_TABLES_HPP

#include "ggml.h"

// Single source of truth for batched-MoE MUL_MAT_ID (type, layout) coverage.
//
// Three tables, one invariant and one publication predicate. Two tables describe
// what the executors can actually run; the third describes what the capability
// query is allowed to advertise (the predicate, moe_aos_expert_publication_wanted at
// the end, says when a tensor's per-expert handles are published):
//
//   moe_mmvq_batched_dispatch_supports_layout   -- mmvq_moe_batched_dispatch()
//   moe_mmvq_pair_glu_dispatch_supports_layout  -- the MXFP4 gate/up pair path
//   moe_mmvq_capability_supports_layout         -- ggml_sycl_moe_query_route_capability()
//
// INVARIANT: capability must be a SUBSET of the union of the executor tables.
//
// Advertising a pair no executor accepts is not a harmless optimism. The route
// is admitted, the executor then declines at submit, and the decline escapes as
// ggml_sycl_fallback_error -> GGML_STATUS_FAILED. ggml_backend_compare_graph_backend
// (ggml/src/ggml-backend.cpp) discards that status, so the harness compares a dst
// the backend never wrote and reports a numeric error instead of a refusal. A
// clean pre-submit refusal is strictly better than a false capability.
//
// The tables live here, outside any SYCL translation unit, so the invariant can
// be gated by a host-only test with no oneAPI toolchain and no GPU:
// tests/test-sycl-moe-mmvq-tables.cpp. Adding a type to one table without the
// other turns that test red.
//
// This header must stay free of SYCL and of ggml-sycl internals -- ggml.h only.

inline bool moe_mmvq_batched_dispatch_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        // gx30. AoS only: these dispatch through mmvq_submit_quant_aos_id(), which
        // instantiates the same generic kernel as the four above. No SoA or packed
        // variant exists for them, so do not widen the layout here without adding
        // the corresponding instantiation.
        //
        // This is the complete set reachable by transcribing an existing generic
        // mul_mat_vec_q<> tuple, plus the iq* types s36q has covered so far: IQ4_NL,
        // whose dense kernel is that same generic body with vec_dot_iq4_nl_q8_1
        // (Q4_0-shaped, qi=4, vdr=2), and the IQ3 and IQ2 types, whose dense tuples
        // (qi = QI3_x / 2 or QI2_x / 2, vdr=1) take their grid-table vec_dots behind
        // generic-signature adaptors (IQ2_S's is already generic). Q2_0 (phase 4) is the
        // Q1_0 tuple shape: qi = QK2_0 / 32 q8_1 chunks, vdr = 1, generic vec_dot. IQ4_XS,
        // IQ1_S and IQ1_M stay refused until their _id variants exist.
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return layout == GGML_LAYOUT_AOS;
        case GGML_TYPE_MXFP4:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED ||
                   layout == GGML_LAYOUT_MXFP4_I8 || layout == GGML_LAYOUT_MXFP4_DPAS ||
                   layout == GGML_LAYOUT_XMX_TILED;
        default:
            return false;
    }
}

// True when the type has an _id kernel family at all, whatever the layout. Lets
// a caller tell "this type is not covered" from "this type is covered but not in
// this layout" without restating the table.
inline bool moe_mmvq_batched_dispatch_supports_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return true;
        default:
            return false;
    }
}

// The MXFP4 gate/up pair path accepts the bundled tiled layout that the generic
// batched dispatch does not; without this table the invariant below would read
// GGML_LAYOUT_XMX_TILED_BUNDLE4 as an over-advertisement when it is in fact
// covered by a different executor.
inline bool moe_mmvq_pair_glu_dispatch_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    if (type != GGML_TYPE_MXFP4) {
        return false;
    }
    return layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_XMX_TILED || layout == GGML_LAYOUT_XMX_TILED_BUNDLE4;
}

// Whether the grouped MXFP4 XMX_TILED executor in mmvq_moe_batched_dispatch() can take a
// route whose device entries cover n_gpu_entries slots.
//
// It always could for FULL cover. It must also for PARTIAL cover: a hybrid decode
// (some experts device-resident, the rest on the host) routes the device slots here
// and the host slots to the CPU arm, whose scatter writes them afterwards. The kernel is
// driven by the per-slot route arrays (expert id, token, slot), so it needs no
// all-slots cover -- only that those arrays exist when cover is partial. Refusing
// partial cover sent the op to a per-expert fallback that did not read this layout
// at all (llama.cpp-4hg7): a support gap closed here rather than routed around.
inline bool moe_mmvq_xmx_tiled_grouped_accepts_cover(bool full_cover, int n_gpu_entries, bool route_arrays_present) {
    return n_gpu_entries > 0 && (full_cover || route_arrays_present);
}

// Layouts the per-expert MXFP4 "direct" dispatch in ggml_sycl_mul_mat can read from the
// bytes it is handed. Its kernels decode exactly these three; every other layout
// (XMX_TILED, XMX_TILED_BUNDLE4, MXFP4_I8, MXFP4_DPAS, ...) is a different byte format and
// reading it as one of these is garbage from the first element out. A caller handed any
// other layout must refuse, never map it onto AOS (llama.cpp-4hg7).
inline bool moe_mmvq_mxfp4_direct_reads_layout(enum ggml_layout_mode layout) {
    return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED;
}

// Whether a prompt-phase MoE layout is executable over a tensor whose probe at that layout found
// `local` device entries, `secondary` entries on another device, `host` host-planned entries and
// `missing` unresolved ones, out of n_experts.
//
// Placement decides the executor: a device entry runs on the device at the layout it is loaded in,
// a host entry runs on the CPU. So every expert is covered when nothing is missing, nothing sits on
// a secondary device (the secondary prompt executor is unvalidated), and at least one entry is on
// the device. Requiring host == 0 -- "all experts on the device" -- turned a mixed tensor (a few
// experts in VRAM, the rest on the host) into an abort that asked for a SOA copy the single-layout
// planner never builds (llama.cpp-f6zo).
//
// local > 0 only separates the two outcomes for an all-host tensor: its route layout stays SOA with
// host operands, because a host entry needs no device layout. local + host == n_experts is a defence
// of the probe's invariant (each expert is counted exactly once), not a condition production can
// reach with missing == 0 and secondary == 0; it fails closed if the probe ever double counts.
inline bool moe_mmvq_prompt_layout_cover_executable(size_t local,
                                                    size_t secondary,
                                                    size_t host,
                                                    size_t missing,
                                                    size_t n_experts) {
    return missing == 0 && secondary == 0 && local > 0 && local + host == n_experts;
}

inline bool moe_mmvq_any_dispatch_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    return moe_mmvq_batched_dispatch_supports_layout(type, layout) ||
           moe_mmvq_pair_glu_dispatch_supports_layout(type, layout);
}

inline bool moe_mmvq_capability_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return layout == GGML_LAYOUT_AOS;
        case GGML_TYPE_MXFP4:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED ||
                   layout == GGML_LAYOUT_MXFP4_I8 || layout == GGML_LAYOUT_MXFP4_DPAS ||
                   layout == GGML_LAYOUT_XMX_TILED || layout == GGML_LAYOUT_XMX_TILED_BUNDLE4;
        default:
            return false;
    }
}

// Whether MUL_MAT_ID *admission* (ggml_backend_sycl_device_supports_op) may
// claim this src0 type at all -- the question "is there an MMID executor for
// this type", asked on both axes the tables describe.
//
// Why admission needs its own predicate instead of the dense MUL_MAT allowlist
// (llama.cpp-yitq): ggml_sycl_mul_mat_type_supported() answers "can this
// backend multiply this type", which is true for F32/F16/IQ1_S..IQ4_XS because
// they have real dense kernels. None of them has an _id kernel family. Sharing
// that allowlist therefore admitted 11 types whose MMID route the oracle then
// refuses one layer down, and per the header comment above that refusal escapes
// as a discarded GGML_STATUS_FAILED -- so the op reported wrong numbers instead
// of falling back to the CPU backend. Admission must key on MMID coverage.
//
// Both axes, because each alone fails open in a different direction: a type
// could have an _id kernel family while the capability query advertises no
// layout for it (admitted, then refused at route time -- the bug above), or a
// layout could be advertised for a type with no executor (the subset invariant
// this header already gates). Requiring both keeps admission, capability, and
// the executor tables in one shape.
//
// The layout scan walks the contiguous enum range rather than restating the
// list. A layout APPENDED after the current last one is not scanned, which can
// only withhold admission, never grant it -- fail-closed, same reasoning as
// all_layouts() in tests/test-sycl-moe-mmvq-tables.cpp. A layout inserted
// mid-enum is caught by that test's shape assertion.
inline bool moe_mmvq_admission_supports_type(enum ggml_type type) {
    if (!moe_mmvq_batched_dispatch_supports_type(type)) {
        return false;
    }
    for (int layout = 0; layout <= (int) GGML_LAYOUT_MXFP4_DPAS; ++layout) {
        if (moe_mmvq_capability_supports_layout(type, (enum ggml_layout_mode) layout)) {
            return true;
        }
    }
    return false;
}

// Whether a device AoS weight tensor gets per-expert retained handles published for
// it (ggml_sycl_publish_backend_aos_expert_handles). The buffer cannot see its
// consumer when the weights are uploaded, so two proxies stand in for "this is an
// expert tensor": the name-based usage classification (classified_expert) and the
// structural test ne[2] > 1. A MUL_MAT_ID dispatch publishes again just before its
// non-materializing retained resolver runs, and there the consumer IS known.
inline bool moe_aos_expert_publication_wanted(bool classified_expert, int64_t ne2, bool consumer_is_mul_mat_id) {
    return classified_expert || ne2 > 1 || consumer_is_mul_mat_id;
}

// The batched mul_mat_id prompt executor (try_pp_onednn_f16_batched in ggml-sycl.cpp): which
// (type, layout) pairs it runs, as two arms (llama.cpp-348c).
//
//   moe_pp_batched_f16_supports_layout  -- the f16 arm: dequantize every active expert into one planned f16
//                                          slot, then one batched oneDNN GEMM over them.
//   moe_pp_batched_woq_supports_layout  -- the MXFP4 weight-only-quantized arm: repack each expert into the
//                                          WOQ nibble/scale slots and let oneDNN dequantize inside the GEMM.
//
// The f16 arm admits a (type, layout) pair only where a kernel decodes one expert of that layout to f16,
// and the executor picks that kernel from the layout the expert is loaded in. Every listed type is admitted
// on AOS. Q4_0, Q8_0, Q4_K and Q6_K are admitted on SOA too, because each SOA plane is flat over one
// expert's blocks: Q4_0, Q4_K and Q6_K decode it with ggml_get_to_fp16_sycl_for_layout's reorder kernels,
// Q8_0 with dequantize_row_q8_0_soa_to_fp16_rowmajor, since that table answers Q8_0 with the AOS kernel for
// every layout. The planner materializes Q8_0 experts SOA on an XMX card and Q6_K experts SOA (the B50
// loads both that way). It plans Q4_0 and Q4_K experts AOS (layout_policy::get_optimal, common.hpp) and no
// knob changes that, so the Q4_0 and Q4_K SOA rows have no device evidence: nothing reaches them until a plan
// chooses SOA for those types. MXFP4's f16 arm has its own SOA row-major dequant and no AOS one in this
// executor. Q1_0 and NVFP4 have converters but are left out, because no Q1_0 or NVFP4 MoE weight has been
// measured here.
// IQ1_S, IQ1_M and IQ4_XS are in although they have no MMVQ id kernel: the f16 arm needs only the dequant,
// so for IQ4_XS this executor is the only device route for prompts (decode stays refused).
//
// The executor's candidate gate and its claim predicate (ggml_sycl_moe_pp_onednn_batched_claims_tensor)
// both ask moe_pp_batched_supports_layout about the type. The claim then also asks
// moe_pp_batched_has_expert_entries (below), which declines a non-AOS layout of a weight in a device buffer:
// under GGML_SYCL_WEIGHTS_EVICTABLE=0 the SOA rows here are therefore never claimed, since that upload holds
// no per-expert entry at SOA, and such a weight keeps the routes it had before. The plan sizes the f16 weight
// slot by asking moe_pp_batched_supports_layout at AOS. That is enough because every non-MXFP4 type taken in
// any layout is taken on AOS and the slot does not depend on the layout, so a type the arm takes always has
// its slot planned; with GGML_SYCL_MOE_PP_F16_BATCHED_AOS=0 or GGML_SYCL_MOE_PP_ONEDNN_F16=0 no such slot is
// planned at all.
inline bool moe_pp_batched_f16_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_MXFP4:
            return layout == GGML_LAYOUT_SOA;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA;
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return layout == GGML_LAYOUT_AOS;
        default:
            return false;
    }
}

inline bool moe_pp_batched_woq_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    return type == GGML_TYPE_MXFP4 && (layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_XMX_TILED);
}

// The one question both gates ask. woq_enabled is GGML_SYCL_MOE_PP_WOQ; aos_enabled is
// GGML_SYCL_MOE_PP_F16_BATCHED_AOS, which set to 0 drops every non-MXFP4 entry (the SOA ones too, despite
// the name) and so restricts the executor to MXFP4 as it was before llama.cpp-348c.
inline bool moe_pp_batched_supports_layout(enum ggml_type        type,
                                           enum ggml_layout_mode layout,
                                           bool                  woq_enabled,
                                           bool                  aos_enabled) {
    if (woq_enabled && moe_pp_batched_woq_supports_layout(type, layout)) {
        return true;
    }
    if (type != GGML_TYPE_MXFP4 && !aos_enabled) {
        return false;
    }
    return moe_pp_batched_f16_supports_layout(type, layout);
}

// Whether the cache holds one entry per expert of a `type` weight at `layout`, which the batched executor needs:
// the retained prompt admission resolves each active expert's own entry before the executor runs. The claim
// predicate asks this after the table, so the claim never takes a layout the cache does not hold per expert.
//   - host buffer (device_buffer false; every default load, since GGML_SYCL_WEIGHTS_EVICTABLE defaults on): the
//     load-end preload stages each device-planned expert in its planned layout, one entry per expert;
//   - device buffer, AOS: ggml_backend_sycl_buffer_set_tensor publishes one slice per expert
//     (ggml_sycl_publish_backend_aos_expert_handles) only when three conditions hold: the upload is complete, it is
//     not reordered, and the buffer is not in the host-pinned tier. This predicate encodes only the second, through
//     the layout: an AOS upload is never reordered. The first holds for every model load, because the device reports
//     async uploads off while a load is open (ggml_backend_sycl_device_get_props), so the loader writes each tensor
//     with one full set_tensor. The third holds only because ggml_backend_sycl_buffer_publish refuses a host-pinned
//     fallback of a device buffer (llama.cpp-mmi1), and that refusal is an accident of the pinned pool's alignment,
//     not a policy. If it is lifted, this predicate must also decline a host-pinned device buffer, or the claim
//     takes an AOS weight that has no per-expert entries;
//   - device buffer, any other layout: the upload reorders the whole tensor into one block over every expert, so
//     no expert has an entry of its own and the admission refuses the op.
// MXFP4 keeps the route it had before llama.cpp-348c, so it is not declined here.
inline bool moe_pp_batched_has_expert_entries(enum ggml_type type, enum ggml_layout_mode layout, bool device_buffer) {
    return type == GGML_TYPE_MXFP4 || !device_buffer || layout == GGML_LAYOUT_AOS;
}

// Why the batched prompt executor declined a dispatch: one value per refusal site, so a refusal is traced
// once per (device, cause) by its name (GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED_TRACE=1). The latch is a 32-bit
// mask per device, so this enum must stay under 32 values.
enum class moe_pp_batched_refusal : uint8_t {
    NONE = 0,
    CPU_REFERENCE,      // GGML_SYCL_MOE_CPU_REFERENCE asked for the CPU reference instead
    ROUTE_OFF,          // GGML_SYCL_MOE_PP_ONEDNN_F16_BATCHED=0 (route not selected) or GGML_SYCL_MOE_PP_ONEDNN_F16=0
    CANDIDATE,          // the (type, layout) pair is not in the tables above, or src1/dst are not f32
    NO_EXPERT_CACHE,    // the dispatch has no expert cache to resolve experts through
    NO_PLAN,            // the layer has no execution plan
    DECODE,             // one token per expert row: decode stays on MMVQ
    NO_ROWS,            // zero tokens
    DIMS,               // the weight row is not a whole number of quant blocks, or a stride is not packed
    DIMS_OVERFLOW,      // a dimension does not fit the GEMM's integer types
    EXPERT_NOT_LOCAL,   // an active expert is not device-resident on this device
    EXPERT_LAYOUT,      // an active expert is materialized in another layout than the route's
    NO_ACTIVE_EXPERTS,  // no expert received a token
    XMX_TILE_CAPS,      // the XMX-tiled WOQ repack is not supported on this device
    F16_TILED,          // XMX-tiled experts reached the f16 arm, which has no tiled dequant
    NO_DEQUANT,         // no f16 dequant kernel for the expert's materialized layout
    SCRATCH_SIZE,       // the scratch arithmetic overflowed
    SCRATCH_CACHE,      // no unified cache for this device
    SCRATCH_UNPLANNED,  // the plan's ring is smaller than this dispatch needs
    SCRATCH_RESERVE,    // the planned ring could not be reserved
    SCRATCH_SLOT,       // a ring slot could not be acquired
    SCRATCH_SLOT_SIZE,  // an acquired slot is smaller than planned
    GEMM,               // the batched GEMM failed after the experts were staged
};

inline const char * moe_pp_batched_refusal_name(moe_pp_batched_refusal refusal) {
    switch (refusal) {
        case moe_pp_batched_refusal::NONE:
            return "none";
        case moe_pp_batched_refusal::CPU_REFERENCE:
            return "cpu-reference";
        case moe_pp_batched_refusal::ROUTE_OFF:
            return "route-off";
        case moe_pp_batched_refusal::CANDIDATE:
            return "candidate";
        case moe_pp_batched_refusal::NO_EXPERT_CACHE:
            return "no-expert-cache";
        case moe_pp_batched_refusal::NO_PLAN:
            return "no-plan";
        case moe_pp_batched_refusal::DECODE:
            return "decode";
        case moe_pp_batched_refusal::NO_ROWS:
            return "no-rows";
        case moe_pp_batched_refusal::DIMS:
            return "dims";
        case moe_pp_batched_refusal::DIMS_OVERFLOW:
            return "dims-overflow";
        case moe_pp_batched_refusal::EXPERT_NOT_LOCAL:
            return "expert-not-local";
        case moe_pp_batched_refusal::EXPERT_LAYOUT:
            return "expert-layout";
        case moe_pp_batched_refusal::NO_ACTIVE_EXPERTS:
            return "no-active-experts";
        case moe_pp_batched_refusal::XMX_TILE_CAPS:
            return "xmx-tile-caps";
        case moe_pp_batched_refusal::F16_TILED:
            return "f16-tiled";
        case moe_pp_batched_refusal::NO_DEQUANT:
            return "no-dequant";
        case moe_pp_batched_refusal::SCRATCH_SIZE:
            return "scratch-size";
        case moe_pp_batched_refusal::SCRATCH_CACHE:
            return "scratch-cache";
        case moe_pp_batched_refusal::SCRATCH_UNPLANNED:
            return "scratch-unplanned";
        case moe_pp_batched_refusal::SCRATCH_RESERVE:
            return "scratch-reserve";
        case moe_pp_batched_refusal::SCRATCH_SLOT:
            return "scratch-slot";
        case moe_pp_batched_refusal::SCRATCH_SLOT_SIZE:
            return "scratch-slot-size";
        case moe_pp_batched_refusal::GEMM:
            return "gemm";
    }
    return "unknown";
}

#endif  // GGML_SYCL_MOE_MMVQ_TABLES_HPP
