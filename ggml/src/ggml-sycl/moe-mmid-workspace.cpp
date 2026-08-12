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
std::atomic<uint64_t> next_bundle_capability{ 1 };

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

uint64_t moe_mmid_mint_monotonic_cookie(std::atomic<uint64_t> & last_issued) noexcept {
    uint64_t observed = last_issued.load(std::memory_order_relaxed);
    for (;;) {
        if (observed >= UINT64_MAX - 1) {
            return 0;
        }
        const uint64_t next = observed + 1;
        if (last_issued.compare_exchange_weak(observed, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return next;
        }
    }
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
    try {
        std::vector<moe_mmid_owner_accounting> merged = *owners;
        size_t                                 total  = *total_vram_bytes;
        for (const auto & charge : charges) {
            auto   owner = std::find_if(merged.begin(), merged.end(), [&](const moe_mmid_owner_accounting & candidate) {
                return candidate.owner_device == charge.first;
            });
            size_t next_used      = 0;
            size_t next_workspace = 0;
            size_t next_total     = 0;
            if (owner == merged.end() || !checked_add(owner->used_bytes, charge.second, &next_used) ||
                next_used > owner->budget_bytes ||
                !checked_add(owner->workspace_bytes, charge.second, &next_workspace) ||
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
    } catch (...) {
        return false;
    }
}

bool moe_mmid_rebuild_per_device_usage(const std::vector<size_t> &                 base_usage,
                                       const std::vector<std::pair<int, size_t>> & kv_charges,
                                       const std::vector<std::pair<int, size_t>> & mmid_charges,
                                       const std::vector<int> &                    devices,
                                       const std::vector<size_t> &                 budgets,
                                       std::vector<size_t> *                       used) noexcept {
    if (used == nullptr || base_usage.size() != devices.size() || budgets.size() != devices.size()) {
        return false;
    }
    try {
        auto candidate = base_usage;
        auto apply     = [&](const std::vector<std::pair<int, size_t>> & charges) {
            for (const auto & charge : charges) {
                const auto it = std::find(devices.begin(), devices.end(), charge.first);
                if (it == devices.end()) {
                    return false;
                }
                const size_t index = static_cast<size_t>(it - devices.begin());
                if (candidate[index] > SIZE_MAX - charge.second) {
                    return false;
                }
                candidate[index] += charge.second;
                if (candidate[index] > budgets[index]) {
                    return false;
                }
            }
            return true;
        };
        if (!apply(kv_charges) || !apply(mmid_charges)) {
            return false;
        }
        *used = std::move(candidate);
        return true;
    } catch (...) {
        return false;
    }
}

bool moe_mmid_admit_single_device_total(size_t   base_bytes,
                                        size_t   workspace_bytes,
                                        size_t   budget_bytes,
                                        size_t * admitted_total) noexcept {
    if (admitted_total == nullptr || base_bytes > SIZE_MAX - workspace_bytes) {
        return false;
    }
    const size_t candidate = base_bytes + workspace_bytes;
    if (candidate > budget_bytes) {
        return false;
    }
    *admitted_total = candidate;
    return true;
}

bool moe_mmid_reaccount_replacement(const std::vector<std::pair<int, size_t>> & old_charges,
                                    const std::vector<std::pair<int, size_t>> & new_charges,
                                    const std::vector<int> &                    devices,
                                    const std::vector<size_t> &                 budgets,
                                    std::vector<size_t> *                       used,
                                    size_t *                                    total) noexcept {
    if (used == nullptr || total == nullptr || devices.size() != budgets.size() || used->size() != devices.size()) {
        return false;
    }
    try {
        auto   candidate  = *used;
        size_t old_global = 0;
        auto   index_for  = [&](int device) {
            const auto it = std::find(devices.begin(), devices.end(), device);
            return it == devices.end() ? SIZE_MAX : static_cast<size_t>(it - devices.begin());
        };
        for (const auto & charge : old_charges) {
            const size_t index = index_for(charge.first);
            if (index == SIZE_MAX || candidate[index] < charge.second || old_global > SIZE_MAX - charge.second) {
                return false;
            }
            candidate[index] -= charge.second;
            old_global += charge.second;
        }
        if (*total < old_global) {
            return false;
        }
        size_t replacement_total = *total - old_global;
        for (const auto & charge : new_charges) {
            const size_t index = index_for(charge.first);
            if (index == SIZE_MAX || candidate[index] > SIZE_MAX - charge.second ||
                replacement_total > SIZE_MAX - charge.second) {
                return false;
            }
            candidate[index] += charge.second;
            replacement_total += charge.second;
            if (candidate[index] > budgets[index]) {
                return false;
            }
        }
        *used  = std::move(candidate);
        *total = replacement_total;
        return true;
    } catch (...) {
        return false;
    }
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
    try {
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
    } catch (...) {
        return {};
    }
}

namespace {
struct registry_slot_state {
    uint64_t generation = 0;
    uint64_t epoch      = 0;
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

uint64_t moe_mmid_registry_lease::epoch() const noexcept {
    return valid() ? epoch_ : 0;
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
    if (epoch_ != 0 && slot_state.epoch != epoch_) {
        return moe_mmid_release_status::WRONG_EPOCH;
    }
    slot_state.busy  = false;
    slot_state.epoch = 0;
    return moe_mmid_release_status::RELEASED;
}

moe_admitted_workspace_bundle::~moe_admitted_workspace_bundle() {
    if (admitted_ && !possible_submit_) {
        release_all();
    }
}

moe_admitted_workspace_bundle::moe_admitted_workspace_bundle(moe_admitted_workspace_bundle && other) noexcept {
    *this = std::move(other);
}

moe_admitted_workspace_bundle & moe_admitted_workspace_bundle::operator=(moe_admitted_workspace_bundle && other) noexcept {
    if (this != &other) {
        // Swap rather than discard: if *this is quarantined, ownership moves
        // back to `other` and its destructor intentionally keeps it quarantined.
        std::swap(admitted_, other.admitted_);
        std::swap(possible_submit_, other.possible_submit_);
        std::swap(capability_, other.capability_);
        std::swap(identity_digest_, other.identity_digest_);
        std::swap(epoch_, other.epoch_);
        std::swap(plan_identity_, other.plan_identity_);
        std::swap(submit_device_, other.submit_device_);
        std::swap(K_, other.K_);
        std::swap(N_, other.N_);
        std::swap(type_, other.type_);
        leases_.swap(other.leases_);
        owners_.swap(other.owners_);
        identities_.swap(other.identities_);
    }
    return *this;
}

bool moe_admitted_workspace_bundle::valid() const noexcept {
    return admitted_ && capability_ != 0 && epoch_ != 0 && plan_identity_ != 0 && !leases_.empty() &&
           leases_.size() == owners_.size();
}

bool moe_admitted_workspace_bundle::matches(int submit, int owner, int64_t K, int64_t N, int32_t type) const noexcept {
    if (!valid() || submit != submit_device_ || K != K_ || N != N_ || type != type_) {
        return false;
    }
    return std::any_of(leases_.begin(), leases_.end(), [&](const moe_mmid_registry_lease & lease) {
        return lease.owner_device() == owner && lease.submit_device() == submit;
    });
}

bool moe_admitted_workspace_bundle::mark_possible_submit() noexcept {
    if (!valid()) {
        return false;
    }
    possible_submit_ = true;
    return true;
}

bool moe_admitted_workspace_bundle::release_all() noexcept {
    bool ok = valid();
    for (auto & lease : leases_) {
        if (lease.valid() &&
            lease.terminal_release(lease.queue_cookie(), lease.generation()) != moe_mmid_release_status::RELEASED) {
            ok = false;
        }
    }
    admitted_ = false;
    capability_ = 0;
    return ok;
}

bool moe_admitted_workspace_bundle::terminal_release() noexcept {
    return release_all();
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
            out.lease.epoch_               = 0;
            out.status                     = moe_mmid_lease_status::ACQUIRED;
            slot.generation                = next_generation;
            slot.epoch                     = 0;
            slot.busy                      = true;
            return out;
        }
        out.status = moe_mmid_lease_status::BUSY;
        return out;
    } catch (...) {
        return out;
    }
}

moe_mmid_admitted_result moe_mmid_workspace_registry::admit(const moe_mmid_admission_request & request) noexcept {
    moe_mmid_admitted_result out;
    if (!request.token.valid() || request.plan_identity == 0 || request.submit_device < 0 || request.epoch == 0 ||
        request.top_k == 0 || request.ne11 == 0 || request.K <= 0 || request.N <= 0 || request.type < 0 ||
        request.owners.empty() || request.retained_occurrences.empty() ||
        request.retained_occurrences.size() % request.top_k != 0) {
        return out;
    }
    for (const auto & identity : request.retained_occurrences) {
        if (identity.weight == 0 || identity.table == 0) {
            return out;
        }
    }
    try {
        // All potentially throwing bundle assembly precedes slot mutation.
        out.bundle.leases_.resize(request.owners.size());
        out.bundle.owners_     = request.owners;
        out.bundle.identities_ = request.retained_occurrences;
        out.bundle.epoch_      = request.epoch;
        out.bundle.plan_identity_ = request.plan_identity;
        out.bundle.submit_device_ = request.submit_device;
        out.bundle.K_             = request.K;
        out.bundle.N_             = request.N;
        out.bundle.type_          = request.type;

        uint64_t digest = 1469598103934665603ULL;
        auto mix = [&](uint64_t value) { digest = (digest ^ value) * 1099511628211ULL; };
        mix(request.token.model_id); mix(request.token.load_txn_id); mix(request.token.generation);
        mix(request.plan_identity); mix(static_cast<uint64_t>(request.submit_device)); mix(request.epoch);
        mix(request.top_k); mix(request.ne11); mix(static_cast<uint64_t>(request.K));
        mix(static_cast<uint64_t>(request.N)); mix(static_cast<uint64_t>(request.type));
        for (const auto & owner : request.owners) {
            mix(static_cast<uint64_t>(owner.owner_device)); mix(owner.queue_cookie);
        }
        for (const auto & identity : request.retained_occurrences) {
            mix(identity.weight); mix(identity.table);
        }
        out.bundle.identity_digest_ = digest != 0 ? digest : 1;

        std::shared_ptr<registry_context> context;
        std::vector<size_t> pool_for_request(request.owners.size(), SIZE_MAX);
        std::vector<std::pair<int, size_t>> lock_order;
        lock_order.reserve(request.owners.size());
        std::vector<uint32_t> selected_slots(request.owners.size(), UINT32_MAX);
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(request.owners.size());

        std::lock_guard<std::mutex> registry_lock(state_->mutex);
        for (const auto & candidate : state_->contexts) {
            if (same_token(candidate->token, request.token) && candidate->plan_identity == request.plan_identity &&
                candidate->submit_device == request.submit_device) {
                context = candidate;
                break;
            }
        }
        if (!context || context->pools.size() != request.owners.size()) {
            return out;
        }

        const size_t capacity = request.retained_occurrences.size() / request.top_k;
        for (size_t r = 0; r < request.owners.size(); ++r) {
            const auto & wanted = request.owners[r];
            if (wanted.owner_device < 0 || wanted.queue_cookie == 0) {
                return out;
            }
            for (size_t p = 0; p < context->pools.size(); ++p) {
                const auto & pool = context->pools[p];
                if (pool->owner_device != wanted.owner_device) {
                    continue;
                }
                if (pool_for_request[r] != SIZE_MAX || pool->queue_cookie != wanted.queue_cookie) {
                    return out;
                }
                for (size_t prior = 0; prior < r; ++prior) {
                    if (pool_for_request[prior] == p) {
                        return out;
                    }
                }
                moe_mmid_workspace_geometry expected;
                if (!moe_mmid_plan_workspace({ static_cast<size_t>(request.K), static_cast<size_t>(request.N),
                                                request.ne11, request.top_k, capacity },
                                              wanted.owner_device != request.submit_device, &expected) ||
                    expected.device_slot_bytes != pool->geometry.device_slot_bytes ||
                    expected.host_slot_bytes != pool->geometry.host_slot_bytes ||
                    expected.activation_f32_bytes != pool->geometry.activation_f32_bytes ||
                    expected.activation_q8_bytes != pool->geometry.activation_q8_bytes ||
                    expected.output_f32_bytes != pool->geometry.output_f32_bytes ||
                    expected.output_q8_bytes != pool->geometry.output_q8_bytes ||
                    expected.descriptor_host_bytes != pool->geometry.descriptor_host_bytes) {
                    return out;
                }
                pool_for_request[r] = p;
            }
            if (pool_for_request[r] == SIZE_MAX) {
                return out;
            }
            lock_order.emplace_back(wanted.owner_device, r);
        }
        std::sort(lock_order.begin(), lock_order.end());
        for (const auto & item : lock_order) {
            locks.emplace_back(context->pools[pool_for_request[item.second]]->mutex);
        }

        // Inspect every owner while all locks are held. No BUSY/generation write
        // occurs unless every exact owner has a usable slot and authority.
        for (size_t r = 0; r < request.owners.size(); ++r) {
            const size_t p = pool_for_request[r];
            const auto & pool = context->pools[p];
            for (uint32_t slot = 0; slot < MOE_MMID_WORKSPACE_DEPTH; ++slot) {
                if (!pool->slots[slot].busy && pool->slots[slot].generation != UINT64_MAX) {
                    selected_slots[r] = slot;
                    break;
                }
            }
            if (selected_slots[r] == UINT32_MAX) {
                out.status = moe_mmid_lease_status::BUSY;
                return out;
            }
            const size_t authority_index = p * MOE_MMID_WORKSPACE_DEPTH + selected_slots[r];
            if (authority_index >= context->authorities.size()) {
                return out;
            }
            auto authority = std::static_pointer_cast<moe_mmid_registry_lease::authority>(
                context->authorities[authority_index]);
            if (!authority || authority->pool.get() != pool.get()) {
                return out;
            }
            auto & lease         = out.bundle.leases_[r];
            lease.authority_     = std::move(authority);
            lease.generation_    = pool->slots[selected_slots[r]].generation + 1;
            lease.queue_cookie_  = request.owners[r].queue_cookie;
            lease.epoch_         = request.epoch;
        }

        for (size_t r = 0; r < request.owners.size(); ++r) {
            auto & slot = context->pools[pool_for_request[r]]->slots[selected_slots[r]];
            slot.generation = out.bundle.leases_[r].generation_;
            slot.epoch      = request.epoch;
            slot.busy       = true;
        }
        const uint64_t capability = next_bundle_capability.fetch_add(1, std::memory_order_relaxed);
        if (capability == 0 || capability == UINT64_MAX) {
            // Capability exhaustion is fail-closed and rolls back exact slots.
            for (size_t r = 0; r < request.owners.size(); ++r) {
                auto & slot = context->pools[pool_for_request[r]]->slots[selected_slots[r]];
                slot.busy = false;
                slot.epoch = 0;
            }
            return out;
        }
        out.bundle.capability_ = capability;
        out.bundle.admitted_   = true;
        out.status             = moe_mmid_lease_status::ACQUIRED;
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
