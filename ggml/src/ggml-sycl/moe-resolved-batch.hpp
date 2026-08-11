// Retained MoE route batch shared by decode and prompt dispatch.
//
// This layer deliberately owns no allocation.  It converts canonical resolver
// results into occurrence-preserving operands whose mem_handles retain the
// backing storage.  Runtime callers are migrated separately.
#pragma once

#include "mem-handle.hpp"

#include <cstddef>
#include <cstdint>
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
    ROUTE_UNAVAILABLE,
    MISSING_HANDLE,
    RAW_COMPAT_HANDLE,
    UNSTABLE_IDENTITY,
    STALE_HANDLE,
    POINTER_MISMATCH,
    LAYOUT_MISMATCH,
    WRONG_DEVICE,
    PLAN_MISMATCH,
};

const char * moe_batch_reject_reason_name(moe_batch_reject_reason reason);

struct moe_resolved_batch_result;

// Normalized mirror of the existing canonical moe_expert_route.  transient_ptr
// is checked while building and is never retained in the resulting batch.
struct moe_batch_route {
    moe_batch_residency residency        = moe_batch_residency::UNAVAILABLE;
    void *              transient_ptr    = nullptr;
    int                 owning_device    = -1;
    int                 planned_device   = -2;
    bool                plan_found        = false;
    bool                planned_on_device = false;
    ggml_layout_mode    requested_layout = GGML_LAYOUT_AOS;
    ggml_layout_mode    actual_layout    = GGML_LAYOUT_AOS;
    size_t              byte_offset      = 0;
    bool                has_ready_event  = false;
    sycl::event         ready_event;
    mem_handle          lease;
    int                 source_reason    = 0;

    bool has_authoritative_planned_alternate() const { return authoritative_planned_alternate_; }

  private:
    // Non-forgeable outside the production canonical wrapper. Generic
    // normalized-route callers can only construct the default (false) proof.
    bool authoritative_planned_alternate_ = false;

    friend moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch(const ggml_tensor *, int, const int32_t *,
                                                                        size_t, size_t, ggml_layout_mode, bool);
    friend bool test_moe_resolved_batch_accepts_actual_planned_alternate();
};

struct moe_resolved_operand {
    int32_t             expert_id       = -1;
    size_t              occurrence      = 0;
    size_t              token_index     = 0;
    size_t              slot_index      = 0;
    moe_batch_residency residency       = moe_batch_residency::UNAVAILABLE;
    int                 owning_device   = -1;
    int                 planned_device  = -2;
    bool                plan_found      = false;
    ggml_layout_mode    requested_layout = GGML_LAYOUT_AOS;
    ggml_layout_mode    actual_layout   = GGML_LAYOUT_AOS;
    size_t              byte_offset     = 0;
    bool                has_ready_event = false;
    sycl::event         ready_event;
    mem_handle          lease;
};

struct moe_resolved_batch {
    int                               submit_device = -1;
    size_t                            slots_per_token = 0;
    std::vector<int32_t>              expert_ids;
    std::vector<moe_resolved_operand> operands;
};

struct moe_resolved_batch_result {
    moe_resolved_batch       batch;
    moe_batch_reject_reason  reject       = moe_batch_reject_reason::NONE;
    size_t                   occurrence   = 0;
    int32_t                  expert_id     = -1;
    int                      source_reason = 0;

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
    if (route.lease.kind() == mem_handle_kind::DIRECT) {
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

    const bool primary = route.residency == moe_batch_residency::PRIMARY_DEVICE;
    const bool secondary = route.residency == moe_batch_residency::SECONDARY_DEVICE;
    const bool host = route.residency == moe_batch_residency::HOST;
    if ((primary && (!resolved.on_device || route.owning_device != submit_device ||
                     route.lease.device() != submit_device)) ||
        (secondary && (!resolved.on_device || route.owning_device < 0 || route.owning_device == submit_device ||
                       route.lease.device() != route.owning_device)) ||
        (host && (resolved.on_device || route.owning_device != mem_handle::HOST_DEVICE))) {
        return moe_batch_reject_reason::WRONG_DEVICE;
    }

    if (route.plan_found) {
        const bool primary_plan_matches = route.planned_on_device ?
                                              (route.planned_device >= 0 && !host &&
                                               route.planned_device == route.owning_device) :
                                              (route.planned_device == mem_handle::HOST_DEVICE && host);
        const bool explicit_alternate_matches =
            route.has_authoritative_planned_alternate() && route.planned_on_device && primary &&
            route.owning_device == submit_device && route.planned_device >= 0 &&
            route.planned_device != route.owning_device;
        if (!primary_plan_matches && !explicit_alternate_matches) {
            return moe_batch_reject_reason::PLAN_MISMATCH;
        }
    }
    return moe_batch_reject_reason::NONE;
}

} // namespace detail

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
    out.batch.submit_device  = submit_device;
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

    for (size_t i = 0; i < count; ++i) {
        const int32_t   expert_id = out.batch.expert_ids[i];
        moe_batch_route route     = resolver(expert_id);
        const moe_batch_reject_reason reject = detail::validate_moe_batch_route(route, submit_device);
        if (reject != moe_batch_reject_reason::NONE) {
            out.reject        = reject;
            out.occurrence    = i;
            out.expert_id     = expert_id;
            out.source_reason = route.source_reason;
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

        moe_resolved_operand operand;
        operand.expert_id        = expert_id;
        operand.occurrence       = i;
        operand.token_index      = i / slots_per_token;
        operand.slot_index       = i % slots_per_token;
        operand.residency        = route.residency;
        operand.owning_device    = route.owning_device;
        operand.planned_device   = route.planned_device;
        operand.plan_found       = route.plan_found;
        operand.requested_layout = route.requested_layout;
        operand.actual_layout    = route.actual_layout;
        operand.byte_offset      = route.byte_offset;
        operand.has_ready_event  = route.has_ready_event;
        if (route.has_ready_event) {
            operand.ready_event = route.ready_event;
        }
        operand.lease = route.lease;
        out.batch.operands.push_back(std::move(operand));
    }
    return out;
}

moe_resolved_batch_result ggml_sycl_build_moe_resolved_batch(const ggml_tensor * src0,
                                                             int                 submit_device,
                                                             const int32_t *     ids,
                                                             size_t              count,
                                                             size_t              slots_per_token,
                                                             ggml_layout_mode    requested_layout,
                                                             bool                allow_materialize = false);

} // namespace ggml_sycl
