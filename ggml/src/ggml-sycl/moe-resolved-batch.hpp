// Retained MoE route batch shared by decode and prompt dispatch.
//
// This layer deliberately owns no allocation.  It converts canonical resolver
// results into occurrence-preserving operands whose mem_handles retain the
// backing storage.  Runtime callers are migrated separately.
#pragma once

#include "mem-handle.hpp"
#include "moe-layer-plan.hpp"
#include "moe-mmid-workspace.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_tensor;

namespace ggml_sycl {

enum class moe_batch_residency : uint8_t {
    PRIMARY_DEVICE,
    SECONDARY_DEVICE,
    HOST,
    UNAVAILABLE,
};

enum class moe_batch_reject_reason : uint8_t {
    NONE,
    INVALID_REQUEST,
    MISSING_ROLE,
    ROLE_ALIGNMENT_MISMATCH,
    ROUTE_UNAVAILABLE,
    MISSING_HANDLE,
    RAW_COMPAT_HANDLE,
    UNSTABLE_IDENTITY,
    STALE_HANDLE,
    POINTER_MISMATCH,
    LAYOUT_MISMATCH,
    WRONG_DEVICE,
    PLAN_MISMATCH,
    WRONG_QUEUE,
    CAPABILITY_UNSUPPORTED,
    RECIPE_MISSING,
    RECIPE_MISMATCH,
    WORKSPACE_UNDERSIZED,
    WORKSPACE_LEASE_MISSING,
};

const char * moe_batch_reject_reason_name(moe_batch_reject_reason reason);

struct moe_resolved_batch_result;

enum class moe_batch_executor : uint8_t {
    HOST_CPU,
    PRIMARY_DEVICE,
    SECONDARY_DEVICE,
};

struct moe_route_request {
    moe_route_phase phase  = moe_route_phase::DECODE;
    int64_t         K      = 0;
    int64_t         N      = 0;
    size_t          rows   = 0;
    ggml_type       type   = GGML_TYPE_COUNT;
    int             submit = -1;
};

enum class moe_recipe_queue : uint8_t { NONE, SUBMIT, OWNER };
enum class moe_recipe_transfer : uint8_t { NONE, HOST_ACTIVATION, PEER_DIRECT, HOST_BOUNCE };

struct moe_workspace_recipe {
    size_t alignment            = 64;
    size_t activation_f32_bytes = 0;
    size_t activation_q8_bytes  = 0;
    size_t output_f32_bytes     = 0;
    size_t descriptor_bytes     = 0;
    size_t total_bytes          = 0;
    size_t chunk_rows           = 0;
};

struct moe_execution_recipe {
    moe_batch_executor   kind     = moe_batch_executor::HOST_CPU;
    moe_route_kernel     kernel   = moe_route_kernel::NONE;
    moe_recipe_queue     queue    = moe_recipe_queue::NONE;
    moe_recipe_transfer  transfer = moe_recipe_transfer::NONE;
    moe_route_request    request{};
    moe_workspace_recipe workspace{};
    ggml_layout_mode     layout       = GGML_LAYOUT_AOS;
    int                  owner_device = mem_handle::HOST_DEVICE;
    bool                 valid        = false;
};

inline bool plan_moe_host_workspace(const moe_route_request & request,
                                    size_t                    q8_row_bytes,
                                    size_t                    descriptor_stride,
                                    moe_workspace_recipe *    out) {
    if (!out || request.K <= 0 || request.N <= 0 || request.rows == 0 || q8_row_bytes == 0) {
        return false;
    }
    moe_workspace_recipe plan;
    auto                 checked_mul = [](size_t a, size_t b, size_t * value) {
        if (a != 0 && b > SIZE_MAX / a) {
            return false;
        }
        *value = a * b;
        return true;
    };
    auto checked_align_add = [&](size_t bytes, size_t * total) {
        const size_t mask = plan.alignment - 1;
        if (*total > SIZE_MAX - mask) {
            return false;
        }
        const size_t aligned = (*total + mask) & ~mask;
        if (bytes > SIZE_MAX - aligned) {
            return false;
        }
        *total = aligned + bytes;
        return true;
    };
    if (!checked_mul(request.rows, static_cast<size_t>(request.K), &plan.activation_f32_bytes) ||
        !checked_mul(plan.activation_f32_bytes, sizeof(float), &plan.activation_f32_bytes) ||
        !checked_mul(request.rows, q8_row_bytes, &plan.activation_q8_bytes) ||
        !checked_mul(request.rows, static_cast<size_t>(request.N), &plan.output_f32_bytes) ||
        !checked_mul(plan.output_f32_bytes, sizeof(float), &plan.output_f32_bytes) ||
        !checked_mul(request.rows, descriptor_stride, &plan.descriptor_bytes)) {
        return false;
    }
    size_t total = 0;
    if (!checked_align_add(plan.activation_f32_bytes, &total) || !checked_align_add(plan.activation_q8_bytes, &total) ||
        !checked_align_add(plan.output_f32_bytes, &total) || !checked_align_add(plan.descriptor_bytes, &total)) {
        return false;
    }
    plan.total_bytes = total;
    plan.chunk_rows = request.rows;
    *out             = plan;
    return true;
}

// Device recipes are deliberately bounded: prompt execution reuses one chunk
// of AoS Q8_1/output/descriptor scratch instead of reserving the whole prompt.
inline bool plan_moe_q1_nvfp4_device_workspace(const moe_route_request & request,
                                                size_t                    q8_row_bytes,
                                                size_t                    max_chunk_rows,
                                                moe_workspace_recipe *    out) {
    if (!out || (request.type != GGML_TYPE_Q1_0 && request.type != GGML_TYPE_NVFP4) ||
        request.K <= 0 || request.N <= 0 || request.rows == 0 || q8_row_bytes == 0 || max_chunk_rows == 0) {
        return false;
    }
    const int64_t block_k = request.type == GGML_TYPE_Q1_0 ? 128 : 64;
    if (request.K % block_k != 0 || request.K > INT32_MAX || request.N > INT32_MAX) {
        return false;
    }
    moe_workspace_recipe plan;
    plan.chunk_rows = std::min(request.rows, max_chunk_rows);
    auto checked_mul = [](size_t a, size_t b, size_t * value) {
        if (a != 0 && b > SIZE_MAX / a) return false;
        *value = a * b;
        return true;
    };
    if (!checked_mul(plan.chunk_rows, q8_row_bytes, &plan.activation_q8_bytes) ||
        !checked_mul(plan.chunk_rows, static_cast<size_t>(request.N), &plan.output_f32_bytes) ||
        !checked_mul(plan.output_f32_bytes, sizeof(float), &plan.output_f32_bytes) ||
        !checked_mul(plan.chunk_rows, sizeof(void *) + sizeof(int32_t), &plan.descriptor_bytes)) {
        return false;
    }
    auto aligned_add = [&](size_t bytes, size_t * total) {
        const size_t mask = plan.alignment - 1;
        if (*total > SIZE_MAX - mask) return false;
        const size_t aligned = (*total + mask) & ~mask;
        if (bytes > SIZE_MAX - aligned) return false;
        *total = aligned + bytes;
        return true;
    };
    size_t total = 0;
    if (!aligned_add(plan.activation_q8_bytes, &total) || !aligned_add(plan.output_f32_bytes, &total) ||
        !aligned_add(plan.descriptor_bytes, &total)) {
        return false;
    }
    plan.total_bytes = total;
    *out = plan;
    return true;
}

inline size_t moe_execution_recipe_signature(const moe_execution_recipe & recipe) {
    size_t h   = 1469598103934665603ULL;
    auto   mix = [&](size_t value) {
        h = (h ^ value) * 1099511628211ULL;
    };
    mix(static_cast<size_t>(recipe.valid));
    mix(static_cast<size_t>(recipe.kind));
    mix(static_cast<size_t>(recipe.kernel));
    mix(static_cast<size_t>(recipe.queue));
    mix(static_cast<size_t>(recipe.transfer));
    mix(static_cast<size_t>(recipe.request.phase));
    mix(static_cast<size_t>(recipe.request.K));
    mix(static_cast<size_t>(recipe.request.N));
    mix(recipe.request.rows);
    mix(static_cast<size_t>(recipe.request.type));
    mix(static_cast<size_t>(recipe.request.submit));
    mix(static_cast<size_t>(recipe.layout));
    mix(static_cast<size_t>(recipe.owner_device));
    mix(recipe.workspace.alignment);
    mix(recipe.workspace.activation_f32_bytes);
    mix(recipe.workspace.activation_q8_bytes);
    mix(recipe.workspace.output_f32_bytes);
    mix(recipe.workspace.descriptor_bytes);
    mix(recipe.workspace.total_bytes);
    mix(recipe.workspace.chunk_rows);
    return h;
}

inline size_t moe_admitted_recipe_signature(const moe_execution_recipe & recipe, const mem_handle & lease) {
    size_t h = moe_execution_recipe_signature(recipe);
    // Bind recipe authority to the allocator/cache-minted identity. A recipe
    // copied onto a different same-layout lease is not an admitted recipe.
    h = (h ^ lease.stable_identity_hash()) * 1099511628211ULL;
    return h;
}

inline bool validate_moe_execution_recipe(const moe_execution_recipe & recipe,
                                          size_t                       admitted_signature,
                                          const mem_handle &           lease,
                                          moe_batch_residency          residency,
                                          int                          submit_device,
                                          bool                         owning_queue_available,
                                          size_t                       reserved_workspace_bytes,
                                          moe_batch_reject_reason *    reject) {
    auto refuse = [&](moe_batch_reject_reason why) {
        if (reject) {
            *reject = why;
        }
        return false;
    };
    if (!recipe.valid) {
        return refuse(moe_batch_reject_reason::RECIPE_MISSING);
    }
    const size_t alignment = recipe.workspace.alignment;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || recipe.request.rows == 0) {
        return refuse(moe_batch_reject_reason::RECIPE_MISMATCH);
    }
    if (recipe.kind == moe_batch_executor::HOST_CPU &&
        (recipe.request.type == GGML_TYPE_Q1_0 || recipe.request.type == GGML_TYPE_NVFP4)) {
        moe_workspace_recipe expected;
        if (recipe.workspace.activation_q8_bytes % recipe.request.rows != 0 ||
            recipe.workspace.descriptor_bytes % recipe.request.rows != 0 ||
            !plan_moe_host_workspace(recipe.request,
                                     recipe.workspace.activation_q8_bytes / recipe.request.rows,
                                     recipe.workspace.descriptor_bytes / recipe.request.rows, &expected) ||
            expected.alignment != recipe.workspace.alignment || expected.activation_f32_bytes != recipe.workspace.activation_f32_bytes ||
            expected.activation_q8_bytes != recipe.workspace.activation_q8_bytes ||
            expected.output_f32_bytes != recipe.workspace.output_f32_bytes ||
            expected.descriptor_bytes != recipe.workspace.descriptor_bytes ||
            expected.total_bytes != recipe.workspace.total_bytes || expected.chunk_rows != recipe.workspace.chunk_rows) {
            return refuse(moe_batch_reject_reason::RECIPE_MISMATCH);
        }
    }
    if (moe_admitted_recipe_signature(recipe, lease) != admitted_signature || recipe.request.submit != submit_device) {
        return refuse(moe_batch_reject_reason::RECIPE_MISMATCH);
    }
    const resolved_ptr resolved = lease.resolve();
    if (!resolved.ptr) {
        return refuse(moe_batch_reject_reason::STALE_HANDLE);
    }
    if (resolved.layout != recipe.layout) {
        return refuse(moe_batch_reject_reason::LAYOUT_MISMATCH);
    }
    if (reserved_workspace_bytes < recipe.workspace.total_bytes) {
        return refuse(moe_batch_reject_reason::WORKSPACE_UNDERSIZED);
    }
    if (recipe.kind == moe_batch_executor::HOST_CPU) {
        if (residency != moe_batch_residency::HOST || recipe.kernel != moe_route_kernel::HOST_CPU ||
            recipe.layout != GGML_LAYOUT_AOS || recipe.owner_device != mem_handle::HOST_DEVICE || resolved.on_device) {
            return refuse(moe_batch_reject_reason::RECIPE_MISMATCH);
        }
    } else {
        const bool direct_q1_nvfp4 = recipe.request.type == GGML_TYPE_Q1_0 || recipe.request.type == GGML_TYPE_NVFP4;
        if (direct_q1_nvfp4) {
            moe_workspace_recipe expected;
            if (recipe.kernel != moe_route_kernel::DEVICE_MMVQ_Q1_NVFP4_AOS || recipe.layout != GGML_LAYOUT_AOS ||
                recipe.workspace.chunk_rows == 0 ||
                !plan_moe_q1_nvfp4_device_workspace(recipe.request,
                                                     recipe.workspace.activation_q8_bytes /
                                                         recipe.workspace.chunk_rows,
                                                     recipe.workspace.chunk_rows, &expected) ||
                expected.activation_q8_bytes != recipe.workspace.activation_q8_bytes ||
                expected.output_f32_bytes != recipe.workspace.output_f32_bytes ||
                expected.descriptor_bytes != recipe.workspace.descriptor_bytes ||
                expected.total_bytes != recipe.workspace.total_bytes ||
                expected.chunk_rows != recipe.workspace.chunk_rows) {
                return refuse(moe_batch_reject_reason::RECIPE_MISMATCH);
            }
            if ((recipe.kind == moe_batch_executor::PRIMARY_DEVICE &&
                 (recipe.queue != moe_recipe_queue::SUBMIT || recipe.transfer != moe_recipe_transfer::NONE)) ||
                (recipe.kind == moe_batch_executor::SECONDARY_DEVICE &&
                 (recipe.queue != moe_recipe_queue::OWNER || recipe.transfer != moe_recipe_transfer::HOST_BOUNCE))) {
                return refuse(moe_batch_reject_reason::RECIPE_MISMATCH);
            }
        }
        if (recipe.kind == moe_batch_executor::PRIMARY_DEVICE &&
            (residency != moe_batch_residency::PRIMARY_DEVICE || recipe.owner_device != submit_device)) {
            return refuse(moe_batch_reject_reason::WRONG_DEVICE);
        }
        if (recipe.kind == moe_batch_executor::SECONDARY_DEVICE &&
            (residency != moe_batch_residency::SECONDARY_DEVICE || recipe.owner_device == submit_device ||
             recipe.queue != moe_recipe_queue::OWNER)) {
            return refuse(moe_batch_reject_reason::WRONG_DEVICE);
        }
        if (recipe.queue == moe_recipe_queue::OWNER && !owning_queue_available) {
            return refuse(moe_batch_reject_reason::WRONG_QUEUE);
        }
    }
    if (reject) {
        *reject = moe_batch_reject_reason::NONE;
    }
    return true;
}

// Normalized mirror of the existing canonical moe_expert_route.  transient_ptr
// is checked while building and is never retained in the resulting batch.
struct moe_batch_route {
    moe_batch_residency  residency         = moe_batch_residency::UNAVAILABLE;
    void *               transient_ptr     = nullptr;
    int                  owning_device     = -1;
    int                  planned_device    = -2;
    bool                 plan_found        = false;
    bool                 planned_on_device = false;
    ggml_layout_mode     requested_layout  = GGML_LAYOUT_AOS;
    ggml_layout_mode     actual_layout     = GGML_LAYOUT_AOS;
    size_t               byte_offset       = 0;
    bool                 has_ready_event   = false;
    sycl::event          ready_event;
    mem_handle           lease;
    moe_execution_recipe           recipe;
    moe_mmid_queue_capability      recipe_queue_capability;
    const char *                   recipe_reason = "recipe-unavailable";
    int                            source_reason = 0;

    bool has_authoritative_planned_alternate() const { return authoritative_planned_alternate_; }

  private:
    // Non-forgeable outside the production canonical wrapper. Generic
    // normalized-route callers can only construct the default (false) proof.
    bool authoritative_planned_alternate_ = false;

    friend moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch(const ggml_tensor *,
                                                                        int,
                                                                        const int32_t *,
                                                                        size_t,
                                                                        size_t,
                                                                        ggml_layout_mode,
                                                                        bool,
                                                                        const void *,
                                                                        const void *,
                                                                        const moe_mmid_queue_capability *);
    friend bool                      test_moe_resolved_batch_accepts_actual_planned_alternate(mem_handle);
};

// llama.cpp-iikr (memo_hit fix, design note c-2cc8, approved): everything
// here except the 4 genuinely per-OCCURRENCE fields on moe_resolved_operand
// itself (expert_id/occurrence/token_index/slot_index) is invariant across
// every occurrence of the SAME expert within one build_moe_resolved_batch()
// call -- residency/device/layout/lease/recipe are all resolved once per
// unique expert_id and were previously being deep-copied onto every REPEAT
// occurrence's own moe_resolved_operand (measured: 279,432 such copies/eval,
// 69ms, HOST5 resolve_memo_hit -- each copy included a mem_handle, whose own
// copy constructor bumps its target entry's refcount under a lock,
// mem-handle.hpp:278-284, plus a sycl::event, another refcounted resource).
struct moe_resolved_operand_canonical {
    moe_batch_residency       residency        = moe_batch_residency::UNAVAILABLE;
    int                       owning_device    = -1;
    int                       planned_device   = -2;
    bool                      plan_found       = false;
    ggml_layout_mode          requested_layout = GGML_LAYOUT_AOS;
    ggml_layout_mode          actual_layout    = GGML_LAYOUT_AOS;
    size_t                    byte_offset      = 0;
    bool                      has_ready_event  = false;
    sycl::event               ready_event;
    mem_handle                lease;
    moe_execution_recipe      recipe;
    moe_mmid_queue_capability recipe_queue_capability;
    const char *              recipe_reason             = "recipe-unavailable";
    size_t                    admitted_recipe_signature = 0;
};

// Every occurrence of one expert within a call shares ONE
// moe_resolved_operand_canonical via `canonical` -- built once per unique
// expert_id (build_moe_resolved_batch's first-occurrence path), copied only
// as a shared_ptr (one atomic refcount increment) onto every repeat
// occurrence instead of the whole payload. `canonical` is set unconditionally
// by construction (build_moe_resolved_batch never pushes an operand without
// it); the accessor methods below dereference it without a null check,
// mirroring how a default-constructed mem_handle/sycl::event/etc. was always
// safe to read as "empty" in the pre-restructuring struct -- the invariant
// moves from "every field has a safe default" to "canonical is always set",
// not from "safe" to "unsafe".
struct moe_resolved_operand {
    int32_t                                               expert_id   = -1;
    size_t                                                occurrence  = 0;
    size_t                                                token_index = 0;
    size_t                                                slot_index  = 0;
    std::shared_ptr<const moe_resolved_operand_canonical> canonical;

    const moe_resolved_operand_canonical & c() const { return *canonical; }

    moe_batch_residency residency() const { return c().residency; }

    int owning_device() const { return c().owning_device; }

    int planned_device() const { return c().planned_device; }

    bool plan_found() const { return c().plan_found; }

    ggml_layout_mode requested_layout() const { return c().requested_layout; }

    ggml_layout_mode actual_layout() const { return c().actual_layout; }

    size_t byte_offset() const { return c().byte_offset; }

    bool has_ready_event() const { return c().has_ready_event; }

    const sycl::event & ready_event() const { return c().ready_event; }

    const mem_handle & lease() const { return c().lease; }

    const moe_execution_recipe & recipe() const { return c().recipe; }

    const moe_mmid_queue_capability & recipe_queue_capability() const { return c().recipe_queue_capability; }

    const char * recipe_reason() const { return c().recipe_reason; }

    size_t admitted_recipe_signature() const { return c().admitted_recipe_signature; }
};

// Opaque execution authority created only from an admitted operand. Runtime
// partitions may copy this ticket, but cannot construct or rewrite its recipe.
class moe_admitted_recipe_ticket {
  public:
    bool valid() const { return valid_; }
    const moe_execution_recipe & recipe() const { return recipe_; }
    size_t signature() const { return signature_; }

  private:
    moe_execution_recipe recipe_{};
    size_t               signature_ = 0;
    bool                 valid_     = false;

    friend moe_admitted_recipe_ticket make_moe_admitted_recipe_ticket(const moe_resolved_operand & operand);
};

inline moe_admitted_recipe_ticket make_moe_admitted_recipe_ticket(const moe_resolved_operand & operand) {
    moe_admitted_recipe_ticket ticket;
    if (operand.recipe().valid &&
        operand.admitted_recipe_signature() == moe_admitted_recipe_signature(operand.recipe(), operand.lease())) {
        ticket.recipe_    = operand.recipe();
        ticket.signature_ = operand.admitted_recipe_signature();
        ticket.valid_     = true;
    }
    return ticket;
}

struct moe_resolved_batch {
    int                               submit_device   = -1;
    size_t                            slots_per_token = 0;
    std::vector<int32_t>              expert_ids;
    std::vector<moe_resolved_operand> operands;

    // Returns the retained operand for an exact token/slot occurrence.  Prompt
    // executors must not look the expert up again: this occurrence is the
    // routing and ownership authority for the submit.
    const moe_resolved_operand * occurrence(size_t token_index, size_t slot_index) const {
        if (slots_per_token == 0 || slot_index >= slots_per_token) {
            return nullptr;
        }
        const size_t index = token_index * slots_per_token + slot_index;
        return index < operands.size() ? &operands[index] : nullptr;
    }
};

struct moe_batch_local_view {
    std::vector<int32_t>     expert_ids;
    std::vector<void *>      expert_ptrs;
    std::vector<mem_handle>  leases;
    std::vector<sycl::event> ready_events;
    moe_batch_reject_reason  reject     = moe_batch_reject_reason::NONE;
    size_t                   occurrence = 0;

    explicit operator bool() const { return reject == moe_batch_reject_reason::NONE; }
};

// Metadata-only local-executor preflight. Pointer payloads are derived from
// the retained handles at submit time; repeated occurrences are preserved in
// the batch and table entries are deduplicated only when expert ID and stable
// handle identity agree. Conflicting identities fail closed.
//
// build_moe_resolved_batch() memoizes by expert_id (see its comment) and
// copies the whole operand -- residency, owning_device, actual_layout, lease
// included -- onto every repeat occurrence of an expert already seen in that
// call. So the admission checks below (residency/device/layout, then
// lease.resolve(), which each take a lock/atomic per call) produce an
// identical result for every occurrence of one expert; only occurrence/
// token_index/slot_index vary. Run the checks and the resolve once per
// unique expert_id, on first sight, and reuse that result for repeats --
// this preserves the exact reject reason/occurrence a full per-occurrence
// re-check would report, because a repeat can never diverge from its first
// occurrence's outcome.
inline moe_batch_local_view make_moe_batch_local_view(const moe_resolved_batch & batch, ggml_layout_mode layout) {
    moe_batch_local_view                out;
    std::unordered_map<int32_t, size_t> expert_slots;
    expert_slots.reserve(batch.operands.size());
    for (const moe_resolved_operand & operand : batch.operands) {
        const auto [existing, inserted] = expert_slots.emplace(operand.expert_id, out.expert_ids.size());
        if (!inserted) {
            if (!out.leases[existing->second].stable_identity_equal(operand.lease())) {
                out.reject     = moe_batch_reject_reason::POINTER_MISMATCH;
                out.occurrence = operand.occurrence;
                return out;
            }
            if (operand.has_ready_event()) {
                out.ready_events.push_back(operand.ready_event());
            }
            continue;
        }
        if (operand.residency() != moe_batch_residency::PRIMARY_DEVICE ||
            operand.owning_device() != batch.submit_device || operand.actual_layout() != layout) {
            out.reject     = operand.actual_layout() != layout ? moe_batch_reject_reason::LAYOUT_MISMATCH :
                                                                 moe_batch_reject_reason::WRONG_DEVICE;
            out.occurrence = operand.occurrence;
            return out;
        }
        resolved_ptr resolved = operand.lease().resolve(batch.submit_device);
        if (!resolved.ptr) {
            out.reject     = moe_batch_reject_reason::STALE_HANDLE;
            out.occurrence = operand.occurrence;
            return out;
        }
        if (!resolved.on_device || resolved.layout != layout) {
            out.reject     = resolved.layout != layout ? moe_batch_reject_reason::LAYOUT_MISMATCH :
                                                         moe_batch_reject_reason::WRONG_DEVICE;
            out.occurrence = operand.occurrence;
            return out;
        }
        out.expert_ids.push_back(operand.expert_id);
        out.expert_ptrs.push_back(resolved.ptr);
        out.leases.push_back(operand.lease());
        if (operand.has_ready_event()) {
            out.ready_events.push_back(operand.ready_event());
        }
    }
    return out;
}

// llama.cpp-iikr (B50 residual-pool cycle, HOST6 priority 2): boolean-only
// sibling of make_moe_batch_local_view() for a caller that only needs "is
// this batch admissible" and never consumes a resolved pointer table --
// skips the per-unique-expert lease().resolve() call (and the 4 output
// vectors) entirely, reading only the operand's already-verified
// accessors. Safe ONLY when the caller runs this immediately after the
// SAME batch's own build, with no queue submission or yield point in
// between: build_moe_resolved_batch's resolver (detail::
// validate_moe_batch_route) already ran a real lease().resolve() per
// unique expert and confirmed residency/owning_device/actual_layout
// consistency with it, and the unified cache's eviction guard for the
// whole graph-compute pass means nothing can invalidate a lease in that
// gap -- so re-reading the same cached fields here answers the identical
// question a second resolve would, without paying for one. NOT a
// substitute for make_moe_batch_local_view() at a call site temporally
// distant from the build (e.g. actual dispatch-time pointer-table
// construction), which still needs a live resolve.
inline bool moe_batch_role_admissible(const moe_resolved_batch & batch, ggml_layout_mode layout) {
    std::unordered_map<int32_t, size_t> expert_slots;
    expert_slots.reserve(batch.operands.size());
    for (size_t i = 0; i < batch.operands.size(); ++i) {
        const moe_resolved_operand & operand = batch.operands[i];
        const auto [existing, inserted]      = expert_slots.emplace(operand.expert_id, i);
        if (!inserted) {
            if (!batch.operands[existing->second].lease().stable_identity_equal(operand.lease())) {
                return false;
            }
            continue;
        }
        if (operand.residency() != moe_batch_residency::PRIMARY_DEVICE ||
            operand.owning_device() != batch.submit_device || operand.actual_layout() != layout) {
            return false;
        }
    }
    return true;
}

enum class moe_batch_role : uint8_t {
    GATE,
    UP,
    DOWN,
};

inline const char * moe_batch_role_name(moe_batch_role role) {
    switch (role) {
        case moe_batch_role::GATE:
            return "gate";
        case moe_batch_role::UP:
            return "up";
        case moe_batch_role::DOWN:
            return "down";
    }
    return "unknown";
}

// Each role is admitted independently. Weight identity and layout deliberately
// are not alignment keys: gate/up/down are different tensors and may use
// different kernel layouts. The immutable ID snapshot and exact occurrence
// coordinates are the cross-role authority.
struct moe_retained_role_batch {
    moe_batch_role      role            = moe_batch_role::GATE;
    const ggml_tensor * weight_identity = nullptr;
    moe_resolved_batch  batch;
    // llama.cpp-iikr (B50 residual-pool cycle, mechanism 1 fix): the LAYOUT
    // role_layout() actually REQUESTED when building `batch` -- deliberately
    // separate from any individual operand's `actual_layout()`. The two can
    // legitimately diverge: ggml_sycl_resolve_moe_expert_route_for_dispatch
    // has a genuine secondary-layout-fallback path, so one expert's resolved
    // layout is not a reliable proxy for what was asked of the tensor as a
    // whole. A consumer that needs "the layout this role was built at" (as
    // opposed to "what a specific operand actually resolved to") must read
    // this field, never `batch.operands.front().actual_layout()` or similar.
    ggml_layout_mode    requested_layout = GGML_LAYOUT_AOS;
};

struct moe_retained_role_bundle {
    moe_retained_role_batch gate{ moe_batch_role::GATE, nullptr, {} };
    moe_retained_role_batch up{ moe_batch_role::UP, nullptr, {} };
    moe_retained_role_batch down{ moe_batch_role::DOWN, nullptr, {} };

    const moe_retained_role_batch & for_role(moe_batch_role role) const {
        switch (role) {
            case moe_batch_role::GATE:
                return gate;
            case moe_batch_role::UP:
                return up;
            case moe_batch_role::DOWN:
                return down;
        }
        return gate;
    }

    size_t retained_lease_count() const {
        return gate.batch.operands.size() + up.batch.operands.size() + down.batch.operands.size();
    }
};

struct moe_retained_role_bundle_result {
    moe_retained_role_bundle bundle;
    moe_batch_reject_reason  reject     = moe_batch_reject_reason::MISSING_ROLE;
    moe_batch_role           role       = moe_batch_role::GATE;
    size_t                   occurrence = 0;

    explicit operator bool() const { return reject == moe_batch_reject_reason::NONE; }
};

inline moe_retained_role_bundle_result align_moe_retained_role_batches(moe_retained_role_batch gate,
                                                                       moe_retained_role_batch up,
                                                                       moe_retained_role_batch down) {
    moe_retained_role_bundle_result out;
    out.bundle.gate                      = std::move(gate);
    out.bundle.up                        = std::move(up);
    out.bundle.down                      = std::move(down);
    const moe_resolved_batch & authority = out.bundle.gate.batch;
    if (authority.slots_per_token == 0 || authority.operands.empty()) {
        out.reject = moe_batch_reject_reason::MISSING_ROLE;
        out.role   = moe_batch_role::GATE;
        return out;
    }
    for (moe_batch_role role : { moe_batch_role::UP, moe_batch_role::DOWN }) {
        const moe_resolved_batch & candidate = out.bundle.for_role(role).batch;
        if (candidate.submit_device != authority.submit_device ||
            candidate.slots_per_token != authority.slots_per_token ||
            candidate.expert_ids.size() != authority.expert_ids.size() ||
            candidate.operands.size() != authority.operands.size()) {
            out.reject = candidate.operands.empty() ? moe_batch_reject_reason::MISSING_ROLE :
                                                      moe_batch_reject_reason::ROLE_ALIGNMENT_MISMATCH;
            out.role   = role;
            return out;
        }
        for (size_t i = 0; i < authority.operands.size(); ++i) {
            const moe_resolved_operand & expected = authority.operands[i];
            const moe_resolved_operand & actual   = candidate.operands[i];
            if (candidate.expert_ids[i] != authority.expert_ids[i] || actual.expert_id != expected.expert_id ||
                actual.occurrence != expected.occurrence || actual.token_index != expected.token_index ||
                actual.slot_index != expected.slot_index) {
                out.reject     = moe_batch_reject_reason::ROLE_ALIGNMENT_MISMATCH;
                out.role       = role;
                out.occurrence = i;
                return out;
            }
        }
    }
    out.reject = moe_batch_reject_reason::NONE;
    return out;
}

// A transient pointer table is an owned submission result, never just a raw
// ABI pointer. Exact leases and the upload event travel with the table handle.
struct moe_retained_pointer_table {
    mem_handle              table_handle;
    std::vector<mem_handle> role_leases;
    sycl::event             ready_event;
    bool                    has_ready_event = false;

    bool valid() const { return table_handle.has_stable_owner_identity() && !role_leases.empty(); }

    const void * const * resolve_abi(int device) const {
        const resolved_ptr resolved = table_handle.resolve(device);
        return resolved.ptr && resolved.on_device ? static_cast<const void * const *>(resolved.ptr) : nullptr;
    }
};

// The terminal owner is copied into the completion callback/retention sink.
// Keeping every role, table and intermediate handle here makes early scope exit
// harmless while gate/up/GLU/down/secondary/scatter work is in flight.
struct moe_retained_terminal_bundle {
    moe_retained_role_bundle                roles;
    std::vector<moe_retained_pointer_table> tables;
    std::vector<mem_handle>                 intermediates;
    sycl::event                             terminal_event;
    bool                                    terminal_submitted = false;

    size_t retained_handle_count() const {
        size_t count = roles.retained_lease_count() + intermediates.size();
        for (const auto & table : tables) {
            count += 1 + table.role_leases.size();
        }
        return count;
    }
};

struct moe_batch_executor_choice {
    moe_batch_executor      executor = moe_batch_executor::HOST_CPU;
    moe_batch_reject_reason reject   = moe_batch_reject_reason::NONE;

    explicit operator bool() const { return reject == moe_batch_reject_reason::NONE; }
};

// Executor selection is deliberately metadata-only. Callers supply capability
// and owning-queue facts; pointer address spaces are never consulted.
inline moe_batch_executor_choice choose_moe_batch_executor(
    const moe_resolved_operand & operand,
    int                          submit_device,
    bool                         owning_queue_available,
    size_t                       reserved_workspace_bytes,
    const moe_admitted_workspace_bundle * workspace_bundle = nullptr) {
    moe_batch_executor_choice out;
    const bool                direct_q1_nvfp4 =
        operand.recipe().kind != moe_batch_executor::HOST_CPU &&
        (operand.recipe().request.type == GGML_TYPE_Q1_0 || operand.recipe().request.type == GGML_TYPE_NVFP4);
    if (direct_q1_nvfp4) {
        // Production capability remains closed until all four B70 oracle cases
        // pass. A matching bundle is necessary but not sufficient to advertise.
        //
        // This gate, not the capability query, is why Q1_0/NVFP4 do not execute.
        // ggml_sycl_moe_query_route_capability() (ggml-sycl.cpp, the Q1_0/NVFP4
        // fall-through) deliberately ADMITS them as MMVQ_COMPAT, so a reader
        // tracing "capability says local-mmvq but nothing runs" lands here. The
        // two layers disagree on purpose; do not "fix" the admission to match.
        (void) workspace_bundle;
        out.reject = moe_batch_reject_reason::CAPABILITY_UNSUPPORTED;
        return out;
    }
    if (!validate_moe_execution_recipe(operand.recipe(), operand.admitted_recipe_signature(), operand.lease(),
                                       operand.residency(), submit_device, owning_queue_available,
                                       reserved_workspace_bytes, &out.reject)) {
        return out;
    }
    out.executor = operand.recipe().kind;
    return out;
}

struct moe_resolved_batch_result {
    moe_resolved_batch      batch;
    moe_batch_reject_reason reject        = moe_batch_reject_reason::NONE;
    size_t                  occurrence    = 0;
    int32_t                 expert_id     = -1;
    // ⚠ source_reason's default TIES WITH A REAL VALUE: expert_resolve_reason::FOUND
    // is 0, so an unset field reads as "the expert resolved".  The RECIPE_MISSING
    // path below used to leave it unset, and a census duly reported
    // source_reason=0 on 264 refusals -- which decodes to FOUND and invites the
    // conclusion that residency succeeded.  It happened to BE true there (the
    // route must pass validate_moe_batch_route to reach that branch at all), but
    // the number was evidence for nothing.  Every reject path now assigns it.
    int                     source_reason = 0;
    // Which capability clause declined, when reject == RECIPE_MISSING.  Without
    // it a recipe refusal is anonymous and every investigation starts from zero.
    const char *            recipe_reason = nullptr;

    explicit operator bool() const { return reject == moe_batch_reject_reason::NONE; }
};

namespace detail {

inline moe_batch_reject_reason validate_moe_batch_route(const moe_batch_route & route, int submit_device) {
    if (route.residency == moe_batch_residency::UNAVAILABLE || route.transient_ptr == nullptr) {
        return moe_batch_reject_reason::ROUTE_UNAVAILABLE;
    }
    if (!route.lease.valid()) {
        return moe_batch_reject_reason::MISSING_HANDLE;
    }
    if (route.lease.kind() == mem_handle_kind::DIRECT && !route.lease.has_stable_owner_identity()) {
        return moe_batch_reject_reason::RAW_COMPAT_HANDLE;
    }
    if (!route.lease.has_stable_owner_identity()) {
        return moe_batch_reject_reason::UNSTABLE_IDENTITY;
    }

    const resolved_ptr resolved = route.lease.resolve();
    if (!resolved.ptr) {
        return moe_batch_reject_reason::STALE_HANDLE;
    }
    if (resolved.ptr != route.transient_ptr) {
        return moe_batch_reject_reason::POINTER_MISMATCH;
    }
    if (resolved.layout != route.actual_layout) {
        return moe_batch_reject_reason::LAYOUT_MISMATCH;
    }

    const bool primary   = route.residency == moe_batch_residency::PRIMARY_DEVICE;
    const bool secondary = route.residency == moe_batch_residency::SECONDARY_DEVICE;
    const bool host      = route.residency == moe_batch_residency::HOST;
    if ((primary &&
         (!resolved.on_device || route.owning_device != submit_device || route.lease.device() != submit_device)) ||
        (secondary && (!resolved.on_device || route.owning_device < 0 || route.owning_device == submit_device ||
                       route.lease.device() != route.owning_device)) ||
        (host && (resolved.on_device || route.owning_device != mem_handle::HOST_DEVICE))) {
        return moe_batch_reject_reason::WRONG_DEVICE;
    }

    if (route.plan_found) {
        const bool primary_plan_matches =
            route.planned_on_device ?
                (route.planned_device >= 0 && !host && route.planned_device == route.owning_device) :
                (route.planned_device == mem_handle::HOST_DEVICE && host);
        const bool explicit_alternate_matches = route.has_authoritative_planned_alternate() &&
                                                route.planned_on_device && primary &&
                                                route.owning_device == submit_device && route.planned_device >= 0 &&
                                                route.planned_device != route.owning_device;
        if (!primary_plan_matches && !explicit_alternate_matches) {
            return moe_batch_reject_reason::PLAN_MISMATCH;
        }
    }
    return moe_batch_reject_reason::NONE;
}

}  // namespace detail

// llama.cpp-iikr (promptadmit remainder cycle): team-lead's discriminating
// question for the admit_resolve remainder (44 ms/23 misses = 1.9 ms/miss,
// after the by-value-copy fix landed) -- does that cost live in the INNER
// per-unique-expert resolver call (resolver(expert_id) below, ~32 calls/role
// at most, memoized so each unique expert pays it once) or in the OUTER
// per-operand work that runs once per TOKEN OCCURRENCE (up to ~2048
// operands/role for a 512-token prompt: the memo-hit fast copy path for
// repeats, plus the occurrence/token_index/slot_index stamping this file's
// own build_moe_resolved_batch comment already identifies as the ids-
// dependent part, see lines 771/792-794/825-827 below)? Declared here, not
// in ggml-sycl.cpp's mxfp4_pp_batched_profile_accum, for the same reason as
// ggml_sycl_gemm_profile in gemm.hpp: this header is included before that
// struct is defined in the same translation unit. Unconditional measurement
// (two cheap chrono reads per UNIQUE EXPERT, not per operand -- see the
// running-clock placement in the loop below, which brackets the resolver()
// call itself as INNER and attributes every other iteration's time,
// including every memo-hit repeat, to OUTER via the same shared clock) so
// this header carries no dependency on ggml-sycl.cpp's profiling-enabled
// check; ggml-sycl.cpp's print_and_reset drains these via exchange(0) and
// folds them into its own report only when ITS OWN gate is on.
namespace ggml_sycl_resolve_batch_profile {
inline std::atomic<int64_t> & inner_resolve_ns_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}

inline std::atomic<int64_t> & inner_resolve_calls_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}

inline std::atomic<int64_t> & outer_stamp_ns_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}

inline std::atomic<int64_t> & outer_stamp_calls_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}

// llama.cpp-iikr (outer-stamping remainder cycle): outer_stamp_ns turned out
// to dominate inner 12:1 (HOST5, c35ddbb71). It conflates two structurally
// different things: loop overhead + full field stamping for a genuinely NEW
// expert (~32/role, matches inner's own volume) vs. the memo-HIT branch for
// every REPEAT token occurrence (~2000+/role for a typical prompt), which
// copies the WHOLE moe_resolved_operand -- including its mem_handle lease
// (a copy bumps the target entry's refcount under a lock, mem-handle.hpp:
// 278-284) and its sycl::event -- just to overwrite 3 scalar fields
// afterward. Split out here so the two are no longer lumped together: the
// memo-hit branch now marks itself separately (right before its own
// `continue`), so outer_stamp_ns naturally narrows to loop-overhead +
// genuinely-new-expert stamping only, and memo_hit_ns isolates the repeat-
// occurrence copy specifically -- the design note's leading hypothesis.
inline std::atomic<int64_t> & memo_hit_ns_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}

inline std::atomic<int64_t> & memo_hit_calls_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}
}  // namespace ggml_sycl_resolve_batch_profile

// Every occurrence is resolved and produces one operand with its original
// token and slot.  Retained handles may be canonicalized only by stable owner
// identity; expert IDs and raw pointers are never identity keys.
template <typename Resolver>
moe_resolved_batch_result build_moe_resolved_batch(const int32_t * ids,
                                                   size_t          count,
                                                   size_t          slots_per_token,
                                                   int             submit_device,
                                                   Resolver &&     resolver) {
    moe_resolved_batch_result out;
    out.batch.submit_device   = submit_device;
    out.batch.slots_per_token = slots_per_token;
    if ((count != 0 && ids == nullptr) || slots_per_token == 0 || submit_device < 0) {
        out.reject = moe_batch_reject_reason::INVALID_REQUEST;
        return out;
    }

    // The only ID copy.  Everything below indexes this retained snapshot.
    if (count != 0) {
        out.batch.expert_ids.assign(ids, ids + count);
    }
    out.batch.operands.reserve(count);
    std::vector<mem_handle> canonical_leases;

    // resolver(expert_id) depends only on expert_id (plus the fixed src0/device/
    // layout/count captured by the caller's closure) -- it is not a function of
    // the occurrence index.  So every repeat occurrence of an expert already seen
    // this call resolves to the identical route.  Memoize by expert_id and skip
    // the resolver (string ops, mem_handle::resolve mutex, recipe lookup) on
    // repeats; only the per-occurrence fields below (occurrence/token_index/
    // slot_index) vary and are set on every iteration regardless of path taken.
    std::unordered_map<int32_t, size_t> expert_first_index;
    expert_first_index.reserve(count);

    // llama.cpp-iikr (promptadmit remainder cycle): running clock, same idiom
    // as ggml-sycl.cpp's own host_phase_mark -- a shared timestamp reset at
    // every mark, so consecutive marks partition elapsed time rather than
    // re-measuring from a fixed origin. Marked at RESOLVER-CALL boundaries
    // (up to ~32/role, memoized), not per operand (up to ~2048/role) -- the
    // latter would make the measurement's own chrono overhead comparable to
    // what it is trying to measure.
    auto resolve_batch_loop_clock = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < count; ++i) {
        const int32_t expert_id = out.batch.expert_ids[i];

        const auto memo = expert_first_index.find(expert_id);
        if (memo != expert_first_index.end()) {
            moe_resolved_operand operand = out.batch.operands[memo->second];
            operand.occurrence           = i;
            operand.token_index          = i / slots_per_token;
            operand.slot_index           = i % slots_per_token;
            out.batch.operands.push_back(std::move(operand));
            {
                const auto now = std::chrono::high_resolution_clock::now();
                ggml_sycl_resolve_batch_profile::memo_hit_ns_accum().fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - resolve_batch_loop_clock).count(),
                    std::memory_order_relaxed);
                ggml_sycl_resolve_batch_profile::memo_hit_calls_accum().fetch_add(1, std::memory_order_relaxed);
                resolve_batch_loop_clock = now;
            }
            continue;
        }

        {
            const auto now = std::chrono::high_resolution_clock::now();
            ggml_sycl_resolve_batch_profile::outer_stamp_ns_accum().fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - resolve_batch_loop_clock).count(),
                std::memory_order_relaxed);
            ggml_sycl_resolve_batch_profile::outer_stamp_calls_accum().fetch_add(1, std::memory_order_relaxed);
            resolve_batch_loop_clock = now;
        }
        moe_batch_route route = resolver(expert_id);
        {
            const auto now = std::chrono::high_resolution_clock::now();
            ggml_sycl_resolve_batch_profile::inner_resolve_ns_accum().fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - resolve_batch_loop_clock).count(),
                std::memory_order_relaxed);
            ggml_sycl_resolve_batch_profile::inner_resolve_calls_accum().fetch_add(1, std::memory_order_relaxed);
            resolve_batch_loop_clock = now;
        }
        const moe_batch_reject_reason reject = detail::validate_moe_batch_route(route, submit_device);
        if (reject != moe_batch_reject_reason::NONE) {
            out.reject        = reject;
            out.occurrence    = i;
            out.expert_id     = expert_id;
            out.source_reason = route.source_reason;
            out.recipe_reason = route.recipe_reason;
            out.batch.operands.clear();
            return out;
        }

        bool known_identity = false;
        for (const mem_handle & retained : canonical_leases) {
            if (retained.stable_identity_equal(route.lease)) {
                route.lease    = retained;
                known_identity = true;
                break;
            }
        }
        if (!known_identity) {
            canonical_leases.push_back(route.lease);
        }

        // llama.cpp-iikr (memo_hit fix): built ONCE per unique expert_id here
        // and shared (via `operand.canonical`) by every repeat occurrence's
        // own moe_resolved_operand -- see moe_resolved_operand_canonical's
        // own comment for why. This is exactly the same per-field population
        // the pre-restructuring code did directly onto `operand`; only the
        // destination changed.
        auto canonical_entry              = std::make_shared<moe_resolved_operand_canonical>();
        canonical_entry->residency        = route.residency;
        canonical_entry->owning_device    = route.owning_device;
        canonical_entry->planned_device   = route.planned_device;
        canonical_entry->plan_found       = route.plan_found;
        canonical_entry->requested_layout = route.requested_layout;
        canonical_entry->actual_layout    = route.actual_layout;
        canonical_entry->byte_offset      = route.byte_offset;
        canonical_entry->has_ready_event  = route.has_ready_event;
        if (route.has_ready_event) {
            canonical_entry->ready_event = route.ready_event;
        }
        canonical_entry->lease                     = route.lease;
        canonical_entry->recipe                    = route.recipe;
        canonical_entry->recipe_queue_capability   = route.recipe_queue_capability;
        canonical_entry->recipe_reason             = route.recipe_reason;
        canonical_entry->admitted_recipe_signature = moe_admitted_recipe_signature(route.recipe, route.lease);
        if (!route.recipe.valid) {
            out.reject        = moe_batch_reject_reason::RECIPE_MISSING;
            out.occurrence    = i;
            out.expert_id     = expert_id;
            out.source_reason = route.source_reason;
            out.recipe_reason = route.recipe_reason;
            out.batch.operands.clear();
            return out;
        }

        moe_resolved_operand operand;
        operand.expert_id   = expert_id;
        operand.occurrence  = i;
        operand.token_index = i / slots_per_token;
        operand.slot_index  = i % slots_per_token;
        operand.canonical   = std::move(canonical_entry);
        expert_first_index.emplace(expert_id, out.batch.operands.size());
        out.batch.operands.push_back(std::move(operand));
    }
    // Final mark: attributes the tail -- any memo-hit iterations after the
    // last resolver call, through loop end -- to OUTER. An early return
    // above (reject/RECIPE_MISSING) skips this, same posture as every other
    // phase mark in this codebase: an early exit before a mark leaves that
    // call's contribution to it unaccounted.
    {
        const auto now = std::chrono::high_resolution_clock::now();
        ggml_sycl_resolve_batch_profile::outer_stamp_ns_accum().fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - resolve_batch_loop_clock).count(),
            std::memory_order_relaxed);
        ggml_sycl_resolve_batch_profile::outer_stamp_calls_accum().fetch_add(1, std::memory_order_relaxed);
    }
    return out;
}

moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch(const ggml_tensor * src0,
                                                             int                 submit_device,
                                                             const int32_t *     ids,
                                                             size_t              count,
                                                             size_t              slots_per_token,
                                                             ggml_layout_mode    requested_layout,
                                                             bool                allow_materialize = false,
                                                             const void *        invocation_backend = nullptr,
                                                             const void *        invocation_queue = nullptr,
                                                             const moe_mmid_queue_capability * queue_capability = nullptr);

}  // namespace ggml_sycl
