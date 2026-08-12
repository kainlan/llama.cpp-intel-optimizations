#include "moe-mmid-workspace.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>

namespace ggml_sycl {
namespace {

bool checked_add(size_t a, size_t b, size_t * out) noexcept {
    if (out == nullptr || b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    *out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t * out) noexcept {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<size_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

bool checked_aligned_place(size_t bytes, size_t * total, size_t * offset = nullptr) noexcept {
    const size_t mask = MOE_MMID_DEVICE_ALIGNMENT - 1;
    size_t       aligned;
    if (!checked_add(*total, mask, &aligned)) {
        return false;
    }
    aligned &= ~mask;
    if (offset != nullptr) {
        *offset = aligned;
    }
    return checked_add(aligned, bytes, total);
}

std::atomic<uint64_t> next_pool_identity{ 1 };

}  // namespace

bool moe_mmid_capacity(size_t n_ctx, size_t n_ubatch, size_t n_seq_max, size_t * out) noexcept {
    (void) n_seq_max;
    if (out == nullptr || n_ubatch == 0) {
        return false;
    }
    *out = n_ctx != 0 ? std::min(n_ctx, n_ubatch) : n_ubatch;
    return true;
}

bool moe_mmid_q8_1_row_bytes(size_t elements, size_t * out) noexcept {
    // block_q8_1 is 32 quants plus two fp16 scalars: exactly 36 bytes per
    // complete 32-element block. MMID kernels do not accept a partial block.
    if (out == nullptr || elements == 0 || elements % 32 != 0) {
        return false;
    }
    return checked_mul(elements / 32, size_t{ 36 }, out);
}

bool moe_mmid_plan_workspace(const moe_mmid_shape &        shape,
                             bool                          secondary_owner,
                             moe_mmid_workspace_geometry * out) noexcept {
    if (out == nullptr || shape.ne10 == 0 || shape.ne01 == 0 || shape.top_k == 0 || shape.c == 0 ||
        (shape.ne11 != 1 && shape.ne11 != shape.top_k)) {
        return false;
    }

    moe_mmid_workspace_geometry g;
    if (!checked_mul(shape.ne11, shape.c, &g.activation_rows) || !checked_mul(shape.top_k, shape.c, &g.occurrences) ||
        !moe_mmid_q8_1_row_bytes(shape.ne10, &g.q8_ne10_row_bytes) ||
        !moe_mmid_q8_1_row_bytes(shape.ne01, &g.q8_ne01_row_bytes) ||
        !checked_mul(g.activation_rows, shape.ne10, &g.activation_f32_bytes) ||
        !checked_mul(g.activation_f32_bytes, sizeof(float), &g.activation_f32_bytes) ||
        !checked_mul(g.activation_rows, g.q8_ne10_row_bytes, &g.activation_q8_bytes) ||
        !checked_mul(g.occurrences, shape.ne01, &g.output_f32_bytes) ||
        !checked_mul(g.output_f32_bytes, sizeof(float), &g.output_f32_bytes) ||
        !checked_mul(g.occurrences, g.q8_ne01_row_bytes, &g.output_q8_bytes) ||
        !checked_mul(g.occurrences, sizeof(moe_mmid_descriptor), &g.descriptor_host_bytes)) {
        return false;
    }

    size_t device = 0;
    if (!checked_aligned_place(g.activation_f32_bytes, &device, &g.activation_f32_offset) ||
        !checked_aligned_place(g.activation_q8_bytes, &device, &g.activation_q8_offset) ||
        !checked_aligned_place(g.output_f32_bytes, &device, &g.output_f32_offset) ||
        !checked_aligned_place(g.output_q8_bytes, &device, &g.output_q8_offset) || !checked_aligned_place(0, &device)) {
        return false;
    }
    g.device_slot_bytes = device;

    if (secondary_owner) {
        g.secondary_activation_d2h_bytes = g.activation_f32_bytes;
        g.secondary_activation_h2d_bytes = g.activation_f32_bytes;
        g.secondary_output_d2h_bytes     = g.output_f32_bytes;
        g.secondary_output_h2d_bytes     = g.output_f32_bytes;
        if (!checked_add(g.secondary_activation_d2h_bytes, g.secondary_activation_h2d_bytes,
                         &g.secondary_bounce_bytes) ||
            !checked_add(g.secondary_bounce_bytes, g.secondary_output_d2h_bytes, &g.secondary_bounce_bytes) ||
            !checked_add(g.secondary_bounce_bytes, g.secondary_output_h2d_bytes, &g.secondary_bounce_bytes)) {
            return false;
        }
    }
    if (!checked_add(g.descriptor_host_bytes, g.secondary_bounce_bytes, &g.host_slot_bytes)) {
        return false;
    }
    g.valid = true;
    *out    = g;
    return true;
}

bool moe_mmid_component_max(moe_mmid_workspace_geometry * a, const moe_mmid_workspace_geometry & b) noexcept {
    if (a == nullptr || !b.valid) {
        return false;
    }
    moe_mmid_workspace_geometry merged = *a;
#define MMID_MAX(field) merged.field = std::max(merged.field, b.field)
    MMID_MAX(activation_rows);
    MMID_MAX(occurrences);
    MMID_MAX(q8_ne10_row_bytes);
    MMID_MAX(q8_ne01_row_bytes);
    MMID_MAX(activation_f32_bytes);
    MMID_MAX(activation_q8_bytes);
    MMID_MAX(output_f32_bytes);
    MMID_MAX(output_q8_bytes);
    MMID_MAX(descriptor_host_bytes);
    MMID_MAX(secondary_activation_d2h_bytes);
    MMID_MAX(secondary_activation_h2d_bytes);
    MMID_MAX(secondary_output_d2h_bytes);
    MMID_MAX(secondary_output_h2d_bytes);
#undef MMID_MAX
    size_t device = 0;
    if (!checked_aligned_place(merged.activation_f32_bytes, &device, &merged.activation_f32_offset) ||
        !checked_aligned_place(merged.activation_q8_bytes, &device, &merged.activation_q8_offset) ||
        !checked_aligned_place(merged.output_f32_bytes, &device, &merged.output_f32_offset) ||
        !checked_aligned_place(merged.output_q8_bytes, &device, &merged.output_q8_offset) ||
        !checked_aligned_place(0, &device)) {
        return false;
    }
    size_t bounce = 0;
    if (!checked_add(bounce, merged.secondary_activation_d2h_bytes, &bounce) ||
        !checked_add(bounce, merged.secondary_activation_h2d_bytes, &bounce) ||
        !checked_add(bounce, merged.secondary_output_d2h_bytes, &bounce) ||
        !checked_add(bounce, merged.secondary_output_h2d_bytes, &bounce) ||
        !checked_add(merged.descriptor_host_bytes, bounce, &merged.host_slot_bytes)) {
        return false;
    }
    merged.secondary_bounce_bytes = bounce;
    merged.device_slot_bytes      = device;
    merged.valid                  = true;
    *a                            = merged;
    return true;
}

bool moe_mmid_checked_pool_bytes(const moe_mmid_workspace_geometry & g,
                                 uint32_t                            depth,
                                 size_t *                            device_bytes,
                                 size_t *                            host_bytes) noexcept {
    if (!g.valid || depth != MOE_MMID_WORKSPACE_DEPTH || device_bytes == nullptr || host_bytes == nullptr) {
        return false;
    }
    size_t d, h;
    if (!checked_mul(g.device_slot_bytes, depth, &d) || !checked_mul(g.host_slot_bytes, depth, &h)) {
        return false;
    }
    *device_bytes = d;
    *host_bytes   = h;
    return true;
}

bool moe_mmid_checked_zone_total(size_t base_bytes, size_t workspace_bytes, size_t * out) noexcept {
    size_t total;
    if (out == nullptr || !checked_add(base_bytes, workspace_bytes, &total)) {
        return false;
    }
    *out = total;
    return true;
}

bool moe_mmid_checked_product(size_t count, size_t bytes, size_t * out) noexcept {
    size_t total;
    if (out == nullptr || !checked_mul(count, bytes, &total)) {
        return false;
    }
    *out = total;
    return true;
}

bool moe_mmid_debit_device_budget(size_t workspace_bytes, size_t * remaining_bytes) noexcept {
    if (remaining_bytes == nullptr || workspace_bytes > *remaining_bytes) {
        return false;
    }
    *remaining_bytes -= workspace_bytes;
    return true;
}

bool moe_mmid_account_actual_owners(const std::vector<std::pair<int, size_t>> & charges,
                                    std::vector<moe_mmid_owner_accounting> *    owners,
                                    size_t *                                    total_vram_bytes) noexcept {
    if (owners == nullptr || total_vram_bytes == nullptr) {
        return false;
    }
    std::vector<moe_mmid_owner_accounting> merged = *owners;
    size_t                                 total  = *total_vram_bytes;
    for (const auto & charge : charges) {
        auto   owner     = std::find_if(merged.begin(), merged.end(), [&](const moe_mmid_owner_accounting & candidate) {
            return candidate.owner_device == charge.first;
        });
        size_t next_used = 0;
        size_t next_workspace = 0;
        size_t next_total     = 0;
        if (owner == merged.end() || !checked_add(owner->used_bytes, charge.second, &next_used) ||
            next_used > owner->budget_bytes || !checked_add(owner->workspace_bytes, charge.second, &next_workspace) ||
            !checked_add(total, charge.second, &next_total)) {
            return false;
        }
        owner->used_bytes      = next_used;
        owner->workspace_bytes = next_workspace;
        total                  = next_total;
    }
    *owners           = std::move(merged);
    *total_vram_bytes = total;
    return true;
}

struct moe_mmid_workspace_pool::state {
    struct slot_state {
        uint64_t generation = 0;
        uint64_t queue      = 0;
        uint64_t epoch      = 0;
        bool     busy       = false;
    };

    explicit state(uint32_t n) : identity(next_pool_identity.fetch_add(1)), slots(n) {}

    uint64_t                identity;
    std::vector<slot_state> slots;
    std::mutex              mutex;
};

moe_mmid_workspace_pool::moe_mmid_workspace_pool(uint32_t depth) :
    state_(new state(depth == MOE_MMID_WORKSPACE_DEPTH ? depth : 0)) {}

moe_mmid_workspace_pool::~moe_mmid_workspace_pool() = default;

uint32_t moe_mmid_workspace_pool::depth() const noexcept {
    return static_cast<uint32_t>(state_->slots.size());
}

moe_mmid_lease_result moe_mmid_workspace_pool::acquire(uint64_t queue, uint64_t epoch) noexcept {
    moe_mmid_lease_result result;
    if (queue == 0 || epoch == 0 || state_->slots.empty()) {
        return result;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (uint32_t i = 0; i < state_->slots.size(); ++i) {
        auto & slot = state_->slots[i];
        if (slot.busy) {
            continue;
        }
        if (slot.generation == UINT64_MAX) {
            continue;
        }
        slot.busy  = true;
        slot.queue = queue;
        slot.epoch = epoch;
        ++slot.generation;
        result.status                = moe_mmid_lease_status::ACQUIRED;
        result.lease.pool_identity_  = state_->identity;
        result.lease.slot_           = i;
        result.lease.generation_     = slot.generation;
        result.lease.queue_identity_ = queue;
        result.lease.busy_epoch_     = epoch;
        return result;
    }
    result.status = moe_mmid_lease_status::BUSY;
    return result;
}

bool moe_mmid_workspace_pool::set_generation_for_test(uint32_t slot, uint64_t generation) noexcept {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (slot >= state_->slots.size() || state_->slots[slot].busy) {
        return false;
    }
    state_->slots[slot].generation = generation;
    return true;
}

moe_mmid_release_status moe_mmid_workspace_pool::terminal_release(const moe_mmid_workspace_lease & lease,
                                                                  uint64_t                         queue,
                                                                  uint64_t                         epoch) noexcept {
    if (!lease.valid() || lease.pool_identity_ != state_->identity || lease.slot_ >= state_->slots.size()) {
        return moe_mmid_release_status::INVALID;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto &                      slot = state_->slots[lease.slot_];
    if (!slot.busy || slot.generation != lease.generation_) {
        return moe_mmid_release_status::STALE;
    }
    if (slot.queue != queue || lease.queue_identity_ != queue) {
        return moe_mmid_release_status::WRONG_QUEUE;
    }
    if (slot.epoch != epoch || lease.busy_epoch_ != epoch) {
        return moe_mmid_release_status::WRONG_EPOCH;
    }
    slot.busy  = false;
    slot.queue = 0;
    slot.epoch = 0;
    return moe_mmid_release_status::RELEASED;
}

moe_mmid_blob moe_mmid_blob::slice(size_t offset, size_t length) const noexcept {
    moe_mmid_blob out;
    if (!valid() || offset > bytes || length > bytes - offset) {
        return out;
    }
    out.owner = slice_owner ? slice_owner(offset, length) : owner;
    if (!out.owner) {
        return {};
    }
    out.ptr         = static_cast<unsigned char *>(ptr) + offset;
    out.bytes       = length;
    out.device      = device;
    out.host_pinned = host_pinned;
    return out;
}

namespace {
struct registry_slot_state {
    uint64_t generation = 0;
    bool     busy       = false;
};

struct registry_pool_state {
    int                         owner_device  = -1;
    int                         submit_device = -1;
    uint64_t                    plan_identity = 0;
    uint64_t                    queue_cookie  = 0;
    moe_mmid_workspace_geometry geometry;
    moe_mmid_blob               device_pool;
    moe_mmid_blob               host_pool;
    registry_slot_state         slots[MOE_MMID_WORKSPACE_DEPTH];
    std::mutex                  mutex;
};

struct registry_context {
    moe_mmid_model_token                              token;
    uint64_t                                          plan_identity = 0;
    int                                               submit_device = -1;
    std::vector<std::shared_ptr<registry_pool_state>> pools;
    std::vector<std::shared_ptr<void>>                authorities;
};

bool same_token(const moe_mmid_model_token & a, const moe_mmid_model_token & b) {
    return a.model_id == b.model_id && a.load_txn_id == b.load_txn_id && a.generation == b.generation;
}
}  // namespace

struct moe_mmid_registry_lease::authority {
    std::shared_ptr<registry_pool_state> pool;
    uint32_t                             slot = UINT32_MAX;
    moe_mmid_materialized_slices         slices;
};

bool moe_mmid_registry_lease::valid() const noexcept {
    return authority_ && authority_->pool;
}

uint32_t moe_mmid_registry_lease::slot() const noexcept {
    return valid() ? authority_->slot : UINT32_MAX;
}

uint64_t moe_mmid_registry_lease::generation() const noexcept {
    return valid() ? generation_ : 0;
}

uint64_t moe_mmid_registry_lease::plan_identity() const noexcept {
    return valid() ? authority_->pool->plan_identity : 0;
}

uint64_t moe_mmid_registry_lease::queue_cookie() const noexcept {
    return valid() ? queue_cookie_ : 0;
}

int moe_mmid_registry_lease::submit_device() const noexcept {
    return valid() ? authority_->pool->submit_device : -1;
}

int moe_mmid_registry_lease::owner_device() const noexcept {
    return valid() ? authority_->pool->owner_device : -1;
}

const moe_mmid_workspace_geometry & moe_mmid_registry_lease::geometry() const noexcept {
    static const moe_mmid_workspace_geometry invalid;
    return valid() ? authority_->pool->geometry : invalid;
}

const moe_mmid_materialized_slices & moe_mmid_registry_lease::slices() const noexcept {
    static const moe_mmid_materialized_slices invalid;
    return valid() ? authority_->slices : invalid;
}

moe_mmid_release_status moe_mmid_registry_lease::terminal_release(uint64_t queue, uint64_t generation) noexcept {
    if (!valid() || authority_->slot >= MOE_MMID_WORKSPACE_DEPTH) {
        return moe_mmid_release_status::INVALID;
    }
    std::lock_guard<std::mutex> lock(authority_->pool->mutex);
    registry_slot_state &       slot_state = authority_->pool->slots[authority_->slot];
    if (!slot_state.busy || slot_state.generation != generation_ || generation != generation_) {
        return moe_mmid_release_status::STALE;
    }
    if (queue != queue_cookie_ || queue != authority_->pool->queue_cookie) {
        return moe_mmid_release_status::WRONG_QUEUE;
    }
    slot_state.busy = false;
    return moe_mmid_release_status::RELEASED;
}

struct moe_mmid_workspace_registry::state {
    mutable std::mutex                             mutex;
    std::vector<std::shared_ptr<registry_context>> contexts;
};

moe_mmid_workspace_registry::moe_mmid_workspace_registry() : state_(new state) {}

moe_mmid_workspace_registry::~moe_mmid_workspace_registry() = default;

moe_mmid_materialize_status moe_mmid_workspace_registry::materialize(
    const moe_mmid_model_token &                          token,
    uint64_t                                              plan_identity,
    int                                                   submit_device,
    const std::vector<moe_mmid_materialized_owner_plan> & owners,
    const moe_mmid_blob_allocator &                       allocator) noexcept {
    if (!token.valid() || plan_identity == 0 || submit_device < 0 || owners.empty() || !allocator) {
        return moe_mmid_materialize_status::INVALID;
    }
    try {
        auto candidate           = std::make_shared<registry_context>();
        candidate->token         = token;
        candidate->plan_identity = plan_identity;
        candidate->submit_device = submit_device;
        candidate->pools.reserve(owners.size());
        for (const auto & owner : owners) {
            size_t expected_device = 0, expected_host = 0;
            if (owner.owner_device < 0 || owner.queue_cookie == 0 ||
                !moe_mmid_checked_pool_bytes(owner.geometry, MOE_MMID_WORKSPACE_DEPTH, &expected_device,
                                             &expected_host) ||
                owner.device_pool_bytes != expected_device || owner.host_pool_bytes != expected_host) {
                return moe_mmid_materialize_status::INVALID;
            }
            auto pool           = std::make_shared<registry_pool_state>();
            pool->owner_device  = owner.owner_device;
            pool->submit_device = submit_device;
            pool->plan_identity = plan_identity;
            pool->queue_cookie  = owner.queue_cookie;
            pool->geometry      = owner.geometry;
            pool->device_pool   = allocator(false, owner.owner_device, expected_device, MOE_MMID_DEVICE_ALIGNMENT);
            if (!pool->device_pool.valid() || pool->device_pool.host_pinned ||
                pool->device_pool.bytes != expected_device || pool->device_pool.device != owner.owner_device ||
                reinterpret_cast<uintptr_t>(pool->device_pool.ptr) % MOE_MMID_DEVICE_ALIGNMENT != 0) {
                return moe_mmid_materialize_status::ALLOCATION_FAILED;
            }
            if (expected_host != 0) {
                pool->host_pool = allocator(true, owner.owner_device, expected_host, alignof(moe_mmid_descriptor));
                if (!pool->host_pool.valid() || !pool->host_pool.host_pinned ||
                    pool->host_pool.bytes != expected_host) {
                    return moe_mmid_materialize_status::ALLOCATION_FAILED;
                }
            }
            // Build every authority and retained offset view before publication.
            // Acquire only copies one of these shared tokens; it never slices or allocates.
            for (uint32_t index = 0; index < MOE_MMID_WORKSPACE_DEPTH; ++index) {
                auto authority                   = std::make_shared<moe_mmid_registry_lease::authority>();
                authority->pool                  = pool;
                authority->slot                  = index;
                const size_t device_base         = static_cast<size_t>(index) * pool->geometry.device_slot_bytes;
                const size_t host_base           = static_cast<size_t>(index) * pool->geometry.host_slot_bytes;
                authority->slices.activation_f32 = pool->device_pool.slice(
                    device_base + pool->geometry.activation_f32_offset, pool->geometry.activation_f32_bytes);
                authority->slices.activation_q8 = pool->device_pool.slice(
                    device_base + pool->geometry.activation_q8_offset, pool->geometry.activation_q8_bytes);
                authority->slices.output_f32 = pool->device_pool.slice(device_base + pool->geometry.output_f32_offset,
                                                                       pool->geometry.output_f32_bytes);
                authority->slices.output_q8  = pool->device_pool.slice(device_base + pool->geometry.output_q8_offset,
                                                                       pool->geometry.output_q8_bytes);
                if (pool->geometry.host_slot_bytes != 0) {
                    authority->slices.host = pool->host_pool.slice(host_base, pool->geometry.host_slot_bytes);
                }
                if (!authority->slices.activation_f32.valid() || !authority->slices.activation_q8.valid() ||
                    !authority->slices.output_f32.valid() || !authority->slices.output_q8.valid() ||
                    (pool->geometry.host_slot_bytes != 0 && !authority->slices.host.valid())) {
                    return moe_mmid_materialize_status::ALLOCATION_FAILED;
                }
                candidate->authorities.push_back(std::move(authority));
            }
            candidate->pools.push_back(std::move(pool));
        }
        std::lock_guard<std::mutex> lock(state_->mutex);
        for (const auto & existing : state_->contexts) {
            if (same_token(existing->token, token) && existing->plan_identity == plan_identity &&
                existing->submit_device == submit_device) {
                return moe_mmid_materialize_status::ALREADY_PUBLISHED;
            }
        }
        state_->contexts.push_back(std::move(candidate));
        return moe_mmid_materialize_status::PUBLISHED;
    } catch (...) {
        return moe_mmid_materialize_status::ALLOCATION_FAILED;
    }
}

moe_mmid_registry_lease_result moe_mmid_workspace_registry::acquire(const moe_mmid_model_token & token,
                                                                    uint64_t                     plan_identity,
                                                                    int                          submit_device,
                                                                    int                          owner_device,
                                                                    uint64_t queue_cookie) noexcept {
    moe_mmid_registry_lease_result out;
    if (!token.valid() || plan_identity == 0 || submit_device < 0 || owner_device < 0 || queue_cookie == 0) {
        return out;
    }
    try {
        std::shared_ptr<registry_context> context_owner;
        size_t                            pool_index = SIZE_MAX;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            for (const auto & context : state_->contexts) {
                if (!same_token(context->token, token) || context->plan_identity != plan_identity ||
                    context->submit_device != submit_device) {
                    continue;
                }
                for (size_t i = 0; i < context->pools.size(); ++i) {
                    if (context->pools[i]->owner_device == owner_device) {
                        context_owner = context;
                        pool_index    = i;
                        break;
                    }
                }
                break;
            }
        }
        if (!context_owner || pool_index == SIZE_MAX) {
            return out;
        }
        const auto & pool = context_owner->pools[pool_index];
        if (pool->queue_cookie != queue_cookie) {
            return out;
        }
        std::lock_guard<std::mutex> lock(pool->mutex);
        for (uint32_t index = 0; index < MOE_MMID_WORKSPACE_DEPTH; ++index) {
            registry_slot_state & slot = pool->slots[index];
            if (slot.busy || slot.generation == UINT64_MAX) {
                continue;
            }
            const size_t authority_index = pool_index * MOE_MMID_WORKSPACE_DEPTH + index;
            if (authority_index >= context_owner->authorities.size()) {
                return out;  // handoff unavailable: slot remains free
            }
            auto authority = std::static_pointer_cast<moe_mmid_registry_lease::authority>(
                context_owner->authorities[authority_index]);
            if (!authority) {
                return out;  // handoff unavailable: slot remains free
            }
            const uint64_t next_generation = slot.generation + 1;
            // shared_ptr handoff and scalar writes are nonthrowing. Publish BUSY last.
            out.lease.authority_           = std::move(authority);
            out.lease.generation_          = next_generation;
            out.lease.queue_cookie_        = queue_cookie;
            out.status                     = moe_mmid_lease_status::ACQUIRED;
            slot.generation                = next_generation;
            slot.busy                      = true;
            return out;
        }
        out.status = moe_mmid_lease_status::BUSY;
        return out;
    } catch (...) {
        return out;
    }
}

bool moe_mmid_workspace_registry::retire(const moe_mmid_model_token & token, uint64_t plan_identity) noexcept {
    if (!token.valid()) {
        return false;
    }
    try {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const size_t                before = state_->contexts.size();
        state_->contexts.erase(
            std::remove_if(state_->contexts.begin(), state_->contexts.end(),
                           [&](const std::shared_ptr<registry_context> & context) {
                               return same_token(context->token, token) &&
                                      (plan_identity == 0 || context->plan_identity == plan_identity);
                           }),
            state_->contexts.end());
        return before != state_->contexts.size();
    } catch (...) {
        return false;
    }
}

std::vector<moe_mmid_registry_context_info> moe_mmid_workspace_registry::list() const {
    std::lock_guard<std::mutex>                 lock(state_->mutex);
    std::vector<moe_mmid_registry_context_info> result;
    result.reserve(state_->contexts.size());
    for (const auto & context : state_->contexts) {
        result.push_back({ context->token, context->plan_identity, context->submit_device, context->pools.size() });
    }
    return result;
}

size_t moe_mmid_workspace_registry::published_contexts() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->contexts.size();
    } catch (...) {
        return 0;
    }
}

}  // namespace ggml_sycl
