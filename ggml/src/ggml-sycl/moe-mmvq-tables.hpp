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
// Three tables and one invariant. Two describe what the executors can actually
// run; the third describes what the capability query is allowed to advertise:
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
        // generic-signature adaptors (IQ2_S's is already generic). IQ4_XS, IQ1_S and
        // IQ1_M stay refused until their _id variants exist.
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

#endif  // GGML_SYCL_MOE_MMVQ_TABLES_HPP

// Whether a device AoS weight tensor gets per-expert retained handles published for
// it (ggml_sycl_publish_backend_aos_expert_handles). The buffer cannot see its
// consumer when the weights are uploaded, so two proxies stand in for "this is an
// expert tensor": the name-based usage classification (classified_expert) and the
// structural test ne[2] > 1. A MUL_MAT_ID dispatch publishes again just before its
// non-materializing retained resolver runs, and there the consumer IS known.
inline bool moe_aos_expert_publication_wanted(bool classified_expert, int64_t ne2, bool consumer_is_mul_mat_id) {
    (void) consumer_is_mul_mat_id;
    return classified_expert || ne2 > 1;
}
