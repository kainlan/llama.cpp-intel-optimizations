// Retained MoE route batch shared by decode and prompt dispatch.
//
// This layer deliberately owns no allocation.  It converts canonical resolver
// results into occurrence-preserving operands whose mem_handles retain the
// backing storage.  Runtime callers are migrated separately.
#pragma once

#include "mem-handle.hpp"
#include "moe-layer-plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    *out             = plan;
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
    if (moe_execution_recipe_signature(recipe) != admitted_signature || recipe.request.submit != submit_device) {
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
        if (recipe.request.type == GGML_TYPE_Q1_0 || recipe.request.type == GGML_TYPE_NVFP4) {
            return refuse(moe_batch_reject_reason::CAPABILITY_UNSUPPORTED);
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
    moe_execution_recipe recipe;
    int                  source_reason = 0;

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
                                                                        bool);
    friend bool                      test_moe_resolved_batch_accepts_actual_planned_alternate(mem_handle);
};

struct moe_resolved_operand {
    int32_t              expert_id        = -1;
    size_t               occurrence       = 0;
    size_t               token_index      = 0;
    size_t               slot_index       = 0;
    moe_batch_residency  residency        = moe_batch_residency::UNAVAILABLE;
    int                  owning_device    = -1;
    int                  planned_device   = -2;
    bool                 plan_found       = false;
    ggml_layout_mode     requested_layout = GGML_LAYOUT_AOS;
    ggml_layout_mode     actual_layout    = GGML_LAYOUT_AOS;
    size_t               byte_offset      = 0;
    bool                 has_ready_event  = false;
    sycl::event          ready_event;
    mem_handle           lease;
    moe_execution_recipe recipe;
    size_t               admitted_recipe_signature = 0;
};

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
inline moe_batch_local_view make_moe_batch_local_view(const moe_resolved_batch & batch, ggml_layout_mode layout) {
    moe_batch_local_view                out;
    std::unordered_map<int32_t, size_t> expert_slots;
    expert_slots.reserve(batch.operands.size());
    for (const moe_resolved_operand & operand : batch.operands) {
        if (operand.residency != moe_batch_residency::PRIMARY_DEVICE || operand.owning_device != batch.submit_device ||
            operand.actual_layout != layout) {
            out.reject     = operand.actual_layout != layout ? moe_batch_reject_reason::LAYOUT_MISMATCH :
                                                               moe_batch_reject_reason::WRONG_DEVICE;
            out.occurrence = operand.occurrence;
            return out;
        }
        resolved_ptr resolved = operand.lease.resolve(batch.submit_device);
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
        const auto [existing, inserted] = expert_slots.emplace(operand.expert_id, out.expert_ids.size());
        if (!inserted) {
            if (!out.leases[existing->second].stable_identity_equal(operand.lease)) {
                out.reject     = moe_batch_reject_reason::POINTER_MISMATCH;
                out.occurrence = operand.occurrence;
                return out;
            }
        } else {
            out.expert_ids.push_back(operand.expert_id);
            out.expert_ptrs.push_back(resolved.ptr);
            out.leases.push_back(operand.lease);
        }
        if (operand.has_ready_event) {
            out.ready_events.push_back(operand.ready_event);
        }
    }
    return out;
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
    moe_batch_reject_reason  reject     = moe_batch_reject_reason::NONE;
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

// Publication is a one-way transaction. Skip/destination-ready state may be
// committed only after the terminal submit succeeds. Once writes begin, a
// failure cannot select an unfused fallback because that could race/duplicate
// destination writes.
class moe_terminal_publication {
  public:
    void terminal_submitted(bool writes_started = true) {
        terminal_submitted_ = true;
        writes_started_     = writes_started;
    }

    bool publish() {
        if (!terminal_submitted_) {
            return false;
        }
        published_ = true;
        return true;
    }

    bool fallback_allowed() const { return !writes_started_ && !published_; }

    bool published() const { return published_; }

  private:
    bool terminal_submitted_ = false;
    bool writes_started_     = false;
    bool published_          = false;
};

struct moe_batch_executor_choice {
    moe_batch_executor      executor = moe_batch_executor::HOST_CPU;
    moe_batch_reject_reason reject   = moe_batch_reject_reason::NONE;

    explicit operator bool() const { return reject == moe_batch_reject_reason::NONE; }
};

// Executor selection is deliberately metadata-only. Callers supply capability
// and owning-queue facts; pointer address spaces are never consulted.
inline moe_batch_executor_choice choose_moe_batch_executor(const moe_resolved_operand & operand,
                                                           int                          submit_device,
                                                           bool                         owning_queue_available,
                                                           size_t reserved_workspace_bytes = SIZE_MAX) {
    moe_batch_executor_choice out;
    if (!validate_moe_execution_recipe(operand.recipe, operand.admitted_recipe_signature, operand.lease,
                                       operand.residency, submit_device, owning_queue_available,
                                       reserved_workspace_bytes, &out.reject)) {
        return out;
    }
    out.executor = operand.recipe.kind;
    return out;
}

struct moe_resolved_batch_result {
    moe_resolved_batch      batch;
    moe_batch_reject_reason reject        = moe_batch_reject_reason::NONE;
    size_t                  occurrence    = 0;
    int32_t                 expert_id     = -1;
    int                     source_reason = 0;

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

    for (size_t i = 0; i < count; ++i) {
        const int32_t                 expert_id = out.batch.expert_ids[i];
        moe_batch_route               route     = resolver(expert_id);
        const moe_batch_reject_reason reject    = detail::validate_moe_batch_route(route, submit_device);
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
        operand.lease                     = route.lease;
        operand.recipe                    = route.recipe;
        operand.admitted_recipe_signature = moe_execution_recipe_signature(route.recipe);
        if (!route.recipe.valid) {
            out.reject     = moe_batch_reject_reason::RECIPE_MISSING;
            out.occurrence = i;
            out.expert_id  = expert_id;
            out.batch.operands.clear();
            return out;
        }
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

}  // namespace ggml_sycl
