#include "moe-mmid-workspace.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
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
std::atomic<uint64_t> next_bundle_capability{ 0 };

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

bool moe_mmid_validate_workspace_geometry(const moe_mmid_workspace_geometry & g,
                                          bool secondary_owner) noexcept {
    if (!g.valid || g.activation_rows == 0 || g.occurrences == 0 || g.q8_ne10_row_bytes == 0 ||
        g.q8_ne01_row_bytes == 0 || g.activation_f32_bytes == 0 || g.activation_q8_bytes == 0 ||
        g.output_f32_bytes == 0 || g.output_q8_bytes == 0 || g.descriptor_host_bytes == 0 ||
        g.activation_f32_bytes % g.activation_rows != 0 ||
        g.activation_q8_bytes % g.activation_rows != 0 ||
        g.activation_q8_bytes / g.activation_rows != g.q8_ne10_row_bytes ||
        g.output_f32_bytes % g.occurrences != 0 ||
        g.output_q8_bytes % g.occurrences != 0 ||
        g.output_q8_bytes / g.occurrences != g.q8_ne01_row_bytes ||
        g.descriptor_host_bytes / sizeof(moe_mmid_descriptor) != g.occurrences ||
        g.descriptor_host_bytes % sizeof(moe_mmid_descriptor) != 0) {
        return false;
    }
    size_t device = 0;
    size_t activation_f32_offset = 0, activation_q8_offset = 0, output_f32_offset = 0, output_q8_offset = 0;
    if (!checked_aligned_place(g.activation_f32_bytes, &device, &activation_f32_offset) ||
        !checked_aligned_place(g.activation_q8_bytes, &device, &activation_q8_offset) ||
        !checked_aligned_place(g.output_f32_bytes, &device, &output_f32_offset) ||
        !checked_aligned_place(g.output_q8_bytes, &device, &output_q8_offset) ||
        !checked_aligned_place(0, &device) || activation_f32_offset != g.activation_f32_offset ||
        activation_q8_offset != g.activation_q8_offset || output_f32_offset != g.output_f32_offset ||
        output_q8_offset != g.output_q8_offset || device != g.device_slot_bytes) {
        return false;
    }
    const size_t expected_activation_d2h = secondary_owner ? g.activation_f32_bytes : 0;
    const size_t expected_activation_h2d = secondary_owner ? g.activation_f32_bytes : 0;
    const size_t expected_output_d2h = secondary_owner ? g.output_f32_bytes : 0;
    const size_t expected_output_h2d = secondary_owner ? g.output_f32_bytes : 0;
    size_t bounce = 0, host = 0;
    if (g.secondary_activation_d2h_bytes != expected_activation_d2h ||
        g.secondary_activation_h2d_bytes != expected_activation_h2d ||
        g.secondary_output_d2h_bytes != expected_output_d2h ||
        g.secondary_output_h2d_bytes != expected_output_h2d ||
        !checked_add(expected_activation_d2h, expected_activation_h2d, &bounce) ||
        !checked_add(bounce, expected_output_d2h, &bounce) ||
        !checked_add(bounce, expected_output_h2d, &bounce) ||
        !checked_add(g.descriptor_host_bytes, bounce, &host)) {
        return false;
    }
    return bounce == g.secondary_bounce_bytes && host == g.host_slot_bytes;
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
    uint64_t generation  = 0;
    uint64_t epoch       = 0;
    bool     busy        = false;
    bool     quarantined = false;
    std::shared_ptr<const moe::graph_retention_record> quarantine_graph;
    std::shared_ptr<void> quarantine_authority;
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
    std::shared_ptr<const lifecycle_plan_snapshot>    exact_plan;
    std::vector<moe_mmid_queue_capability>            queues;
    bool                                               accepting = true;
};

bool same_token(const moe_mmid_model_token & a, const moe_mmid_model_token & b) {
    return a.model_id == b.model_id && a.load_txn_id == b.load_txn_id && a.generation == b.generation;
}

moe_mmid_blob retained_subrange(const moe_mmid_blob & blob, size_t offset, size_t bytes) noexcept {
    moe_mmid_blob out;
    if (!blob.valid() || bytes == 0 || offset > blob.bytes || bytes > blob.bytes - offset) return out;
    out.owner = blob.owner;
    out.ptr = static_cast<unsigned char *>(blob.ptr) + offset;
    out.bytes = bytes;
    out.device = blob.device;
    out.host_pinned = blob.host_pinned;
    return out;
}

bool same_geometry(const moe_mmid_workspace_geometry & a, const moe_mmid_workspace_geometry & b) noexcept {
#define MMID_GEOMETRY_EQ(f) a.f == b.f
    return MMID_GEOMETRY_EQ(valid) && MMID_GEOMETRY_EQ(activation_rows) && MMID_GEOMETRY_EQ(occurrences) &&
           MMID_GEOMETRY_EQ(q8_ne10_row_bytes) && MMID_GEOMETRY_EQ(q8_ne01_row_bytes) &&
           MMID_GEOMETRY_EQ(activation_f32_offset) && MMID_GEOMETRY_EQ(activation_f32_bytes) &&
           MMID_GEOMETRY_EQ(activation_q8_offset) && MMID_GEOMETRY_EQ(activation_q8_bytes) &&
           MMID_GEOMETRY_EQ(output_f32_offset) && MMID_GEOMETRY_EQ(output_f32_bytes) &&
           MMID_GEOMETRY_EQ(output_q8_offset) && MMID_GEOMETRY_EQ(output_q8_bytes) &&
           MMID_GEOMETRY_EQ(device_slot_bytes) && MMID_GEOMETRY_EQ(descriptor_host_bytes) &&
           MMID_GEOMETRY_EQ(secondary_activation_d2h_bytes) && MMID_GEOMETRY_EQ(secondary_activation_h2d_bytes) &&
           MMID_GEOMETRY_EQ(secondary_output_d2h_bytes) && MMID_GEOMETRY_EQ(secondary_output_h2d_bytes) &&
           MMID_GEOMETRY_EQ(secondary_bounce_bytes) && MMID_GEOMETRY_EQ(host_slot_bytes);
#undef MMID_GEOMETRY_EQ
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
    return valid() ? admitted_geometry_ : invalid;
}

const moe_mmid_materialized_slices & moe_mmid_registry_lease::slices() const noexcept {
    static const moe_mmid_materialized_slices invalid;
    return valid() ? admitted_slices_ : invalid;
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
    if (slot_state.quarantined) {
        return moe_mmid_release_status::WRONG_EPOCH;
    }
    if (queue != queue_cookie_ || queue != authority_->pool->queue_cookie) {
        return moe_mmid_release_status::WRONG_QUEUE;
    }
    if (epoch_ != 0 && slot_state.epoch != epoch_) {
        return moe_mmid_release_status::WRONG_EPOCH;
    }
    slot_state.busy        = false;
    slot_state.epoch       = 0;
    slot_state.quarantined = false;
    slot_state.quarantine_graph.reset();
    slot_state.quarantine_authority.reset();
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
        std::swap(owner_count_, other.owner_count_);
        leases_.swap(other.leases_);
        owners_.swap(other.owners_);
        identities_.swap(other.identities_);
        graph_snapshot_.swap(other.graph_snapshot_);
        common_authority_.swap(other.common_authority_);
    }
    return *this;
}

bool moe_admitted_workspace_bundle::valid() const noexcept {
    return admitted_ && capability_ != 0 && epoch_ != 0 && plan_identity_ != 0 && owner_count_ != 0 &&
           owner_count_ <= leases_.size() && identities_ && (graph_snapshot_ || common_authority_);
}

const std::vector<moe::mmid_batch_binding> & moe_admitted_workspace_bundle::retained_occurrences() const noexcept {
    static const std::vector<moe::mmid_batch_binding> empty;
    return identities_ ? *identities_ : empty;
}

bool moe_admitted_workspace_bundle::matches(int submit, int owner, int64_t K, int64_t N, int32_t type,
                                             const moe::mmid_operand_identity * expected,
                                             size_t expected_count) const noexcept {
    if (!valid() || submit != submit_device_ || K != K_ || N != N_ || type != type_) {
        return false;
    }
    bool found_owner = false;
    for (size_t i = 0; i < owner_count_; ++i) {
        found_owner |= leases_[i].owner_device() == owner && leases_[i].submit_device() == submit;
    }
    if (!found_owner || expected == nullptr || expected_count == 0 || expected_count != identities_->size()) {
        return false;
    }
    {
        for (size_t i = 0; i < expected_count; ++i) {
            const auto & actual = identities_->at(i).identity;
            if (actual.allocation_id != expected[i].allocation_id || actual.generation != expected[i].generation ||
                actual.layout_id != expected[i].layout_id || actual.device != expected[i].device ||
                actual.byte_offset != expected[i].byte_offset || actual.byte_size != expected[i].byte_size ||
                actual.occurrence != expected[i].occurrence) return false;
        }
    }
    return true;
}

bool moe_admitted_workspace_bundle::mark_possible_submit() noexcept {
    if (!valid()) return false;
    for (size_t i = 0; i < owner_count_; ++i) {
        auto & lease = leases_[i];
        std::lock_guard<std::mutex> lock(lease.authority_->pool->mutex);
        auto & slot = lease.authority_->pool->slots[lease.slot()];
        if (!slot.busy || slot.generation != lease.generation() || slot.epoch != epoch_) return false;
        slot.quarantine_graph     = graph_snapshot_;
        slot.quarantine_authority = common_authority_;
        slot.quarantined          = true;
    }
    possible_submit_ = true;
    return true;
}

bool moe_admitted_workspace_bundle::release_all() noexcept {
    bool ok = valid();
    for (size_t i = 0; i < owner_count_; ++i) {
        auto & lease = leases_[i];
        if (lease.valid() &&
            lease.terminal_release(lease.queue_cookie(), lease.generation()) != moe_mmid_release_status::RELEASED) {
            ok = false;
        }
    }
    admitted_ = false;
    capability_ = 0;
    common_authority_.reset();
    return ok;
}

bool moe_admitted_workspace_bundle::terminal_release() noexcept {
    if (!valid()) return false;
    if (possible_submit_) {
        const auto & terminals = graph_snapshot_ ? graph_snapshot_->terminals : common_authority_->terminals;
        const auto & proofs = graph_snapshot_ ? graph_snapshot_->quiescence_proofs : common_authority_->quiescence_proofs;
        for (size_t i = 0; i < owner_count_; ++i) {
            const int device = owners_[i].owner_device;
            const auto terminal = terminals.find(device);
            const auto proof = proofs.find(device);
            const bool ready = (terminal != terminals.end() && terminal->second && terminal->second->ready()) ||
                               (proof != proofs.end() && proof->second && proof->second->ready());
            if (!ready) return false;
        }
        for (size_t i = 0; i < owner_count_; ++i) {
            auto & lease = leases_[i];
            std::lock_guard<std::mutex> lock(lease.authority_->pool->mutex);
            auto & slot = lease.authority_->pool->slots[lease.slot()];
            if (!slot.busy || !slot.quarantined || slot.generation != lease.generation()) return false;
            slot.quarantined = false;
        }
    }
    return release_all();
}

struct moe_mmid_workspace_registry::state {
    mutable std::mutex                             mutex;
    std::vector<std::shared_ptr<registry_context>> contexts;
};

moe_mmid_workspace_registry::moe_mmid_workspace_registry() : state_(new state) {}

moe_mmid_workspace_registry::~moe_mmid_workspace_registry() = default;

moe_mmid_materialize_status moe_mmid_workspace_registry::materialize(
    const moe_mmid_model_token & token, uint64_t plan_identity, int submit_device,
    const std::vector<moe_mmid_materialized_owner_plan> & owners,
    const moe_mmid_blob_allocator & allocator) noexcept {
    return materialize(token, plan_identity, {}, submit_device, owners, allocator);
}

moe_mmid_materialize_status moe_mmid_workspace_registry::materialize(
    const moe_mmid_model_token & token, uint64_t plan_identity,
    std::shared_ptr<const lifecycle_plan_snapshot> exact_plan, int submit_device,
    const std::vector<moe_mmid_materialized_owner_plan> & owners,
    const moe_mmid_blob_allocator & allocator) noexcept {
    if (!token.valid() || plan_identity == 0 || submit_device < 0 || owners.empty() || !allocator) {
        return moe_mmid_materialize_status::INVALID;
    }
    try {
        auto candidate           = std::make_shared<registry_context>();
        candidate->token         = token;
        candidate->plan_identity = plan_identity;
        candidate->submit_device = submit_device;
        candidate->exact_plan    = std::move(exact_plan);
        candidate->pools.reserve(owners.size());
        candidate->queues.reserve(owners.size());
        for (const auto & owner : owners) {
            size_t expected_device = 0, expected_host = 0;
            const uint64_t queue_cookie = owner.queue_capability.valid() ? owner.queue_capability.cookie() : owner.queue_cookie;
            if (owner.owner_device < 0 || queue_cookie == 0 ||
                (candidate->exact_plan && (!owner.queue_capability.valid() ||
                 owner.queue_capability.owner_device() != owner.owner_device)) ||
                !moe_mmid_validate_workspace_geometry(owner.geometry, owner.secondary_owner) ||
                !moe_mmid_checked_pool_bytes(owner.geometry, MOE_MMID_WORKSPACE_DEPTH, &expected_device,
                                             &expected_host) ||
                owner.device_pool_bytes != expected_device || owner.host_pool_bytes != expected_host) {
                return moe_mmid_materialize_status::INVALID;
            }
            auto pool           = std::make_shared<registry_pool_state>();
            pool->owner_device  = owner.owner_device;
            pool->submit_device = submit_device;
            pool->plan_identity = plan_identity;
            pool->queue_cookie  = queue_cookie;
            pool->geometry      = owner.geometry;
            candidate->queues.push_back(owner.queue_capability);
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
                    context->submit_device != submit_device || !context->accepting) {
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
            out.lease.admitted_geometry_   = pool->geometry;
            out.lease.admitted_slices_     = authority->slices;
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
    const auto & bindings = request.retained_occurrences;
    const auto & snapshot = request.graph_snapshot;
    if (!request.token.valid() || request.plan_identity == 0 || request.submit_device < 0 ||
        request.graph_registry == nullptr || !request.graph_token.valid() || !snapshot ||
        snapshot->phase != moe::retention_phase::INSTALLED ||
        snapshot->root.model.value != request.token.model_id || snapshot->root.load.value != request.token.load_txn_id ||
        snapshot->root.owner.generation != request.token.generation ||
        !bindings || bindings->empty() || !request.table_owner || request.top_k == 0 || request.ne11 == 0 ||
        request.K <= 0 || request.N <= 0 || request.type < 0 || request.owners.empty() ||
        request.owners.size() > execution::max_devices || bindings->size() % request.top_k != 0) {
        return out;
    }

    const auto canonical_snapshot = request.graph_registry->snapshot(snapshot->key);
    moe::published_graph_token canonical_token;
    if (canonical_snapshot.get() != snapshot.get() ||
        request.graph_registry->acquire_published_token(snapshot->key, &canonical_token) != moe::retention_error::OK ||
        std::memcmp(&canonical_token, &request.graph_token, sizeof(canonical_token)) != 0) return out;

    const auto table_key = request.table_owner->owner();
    if (snapshot->publication_serial == 0 ||
        table_key.context.value != snapshot->key.context.value || table_key.epoch.value != snapshot->key.epoch.value) {
        return out;
    }

    bool exact_table = false;
    for (const auto & table : snapshot->tables) {
        exact_table |= table.owner == request.table_owner && table.table_id == request.table_owner->table_id() &&
                       table.layout_id == request.table_owner->layout_id() && table.device == request.table_owner->device();
    }
    if (!exact_table) return out;

    uint64_t digest = 1469598103934665603ULL;
    auto mix = [&](uint64_t value) { digest = (digest ^ value) * 1099511628211ULL; };
    for (size_t i = 0; i < bindings->size(); ++i) {
        const auto & binding = bindings->at(i);
        const auto & id = binding.identity;
        if (id.allocation_id == 0 || id.generation == 0 || id.layout_id == 0 || id.device < 0 || id.byte_size == 0 ||
            id.occurrence != i || id.byte_offset > binding.owner.extent() ||
            id.byte_size > binding.owner.extent() - id.byte_offset ||
            id.allocation_id != binding.owner.allocation_id() || id.generation != binding.owner.generation() ||
            id.device != binding.owner.device()) return out;
        bool graph_retains = false;
        for (const auto & published : snapshot->batches) {
            const auto & p = published.identity;
            graph_retains |= p.allocation_id == id.allocation_id && p.generation == id.generation &&
                             p.layout_id == id.layout_id && p.device == id.device && p.byte_offset == id.byte_offset &&
                             p.byte_size == id.byte_size && p.occurrence == id.occurrence;
        }
        if (!graph_retains) return out;
        bool table_retains = false;
        for (const auto & entry : request.table_owner->entries()) {
            table_retains |= entry.allocation_id() == id.allocation_id && entry.generation() == id.generation &&
                             entry.device() == id.device && entry.extent() == binding.owner.extent();
        }
        if (!table_retains) return out;
        mix(id.allocation_id); mix(id.generation); mix(id.layout_id); mix(static_cast<uint64_t>(id.device));
        mix(id.byte_offset); mix(id.byte_size); mix(id.occurrence);
    }

    std::shared_ptr<registry_context> context;
    std::array<size_t, execution::max_devices> pool_map{};
    std::array<size_t, execution::max_devices> order{};
    std::array<uint32_t, execution::max_devices> selected{};
    std::array<std::unique_lock<std::mutex>, execution::max_devices> locks{};
    pool_map.fill(SIZE_MAX); selected.fill(UINT32_MAX);

    std::lock_guard<std::mutex> registry_lock(state_->mutex);
    for (const auto & candidate : state_->contexts) {
        if (same_token(candidate->token, request.token) && candidate->plan_identity == request.plan_identity &&
            candidate->submit_device == request.submit_device && candidate->accepting) { context = candidate; break; }
    }
    if (!context || context->pools.size() != request.owners.size()) return out;

    const size_t capacity = bindings->size() / request.top_k;
    std::array<moe_mmid_workspace_geometry, execution::max_devices> requested_geometry{};
    for (size_t r = 0; r < request.owners.size(); ++r) {
        const auto & wanted = request.owners[r];
        if (wanted.owner_device < 0 || wanted.queue_cookie == 0) return out;
        for (size_t p = 0; p < context->pools.size(); ++p) {
            const auto & pool = context->pools[p];
            if (pool->owner_device != wanted.owner_device) continue;
            if (pool_map[r] != SIZE_MAX || pool->queue_cookie != wanted.queue_cookie) return out;
            for (size_t prior = 0; prior < r; ++prior) if (pool_map[prior] == p) return out;
            auto & needed = requested_geometry[r];
            const auto & available = pool->geometry;
            if (!moe_mmid_plan_workspace({ static_cast<size_t>(request.K), static_cast<size_t>(request.N),
                                            request.ne11, request.top_k, capacity },
                                          wanted.owner_device != request.submit_device, &needed) ||
                !moe_mmid_validate_workspace_geometry(available, wanted.owner_device != request.submit_device) ||
                available.activation_rows < needed.activation_rows || available.occurrences < needed.occurrences ||
                available.q8_ne10_row_bytes < needed.q8_ne10_row_bytes ||
                available.q8_ne01_row_bytes < needed.q8_ne01_row_bytes ||
                available.activation_f32_bytes < needed.activation_f32_bytes ||
                available.activation_q8_bytes < needed.activation_q8_bytes ||
                available.output_f32_bytes < needed.output_f32_bytes ||
                available.output_q8_bytes < needed.output_q8_bytes ||
                available.descriptor_host_bytes < needed.descriptor_host_bytes ||
                available.secondary_activation_d2h_bytes < needed.secondary_activation_d2h_bytes ||
                available.secondary_activation_h2d_bytes < needed.secondary_activation_h2d_bytes ||
                available.secondary_output_d2h_bytes < needed.secondary_output_d2h_bytes ||
                available.secondary_output_h2d_bytes < needed.secondary_output_h2d_bytes ||
                available.secondary_bounce_bytes < needed.secondary_bounce_bytes ||
                available.host_slot_bytes < needed.host_slot_bytes) return out;
            pool_map[r] = p;
        }
        if (pool_map[r] == SIZE_MAX) return out;
        order[r] = r;
    }
    std::sort(order.begin(), order.begin() + request.owners.size(), [&](size_t a, size_t b) {
        return request.owners[a].owner_device < request.owners[b].owner_device;
    });
    for (size_t i = 0; i < request.owners.size(); ++i)
        locks[i] = std::unique_lock<std::mutex>(context->pools[pool_map[order[i]]]->mutex);

    for (size_t r = 0; r < request.owners.size(); ++r) {
        const size_t p = pool_map[r];
        const auto & pool = context->pools[p];
        for (uint32_t slot = 0; slot < MOE_MMID_WORKSPACE_DEPTH; ++slot)
            if (!pool->slots[slot].busy && pool->slots[slot].generation != UINT64_MAX) { selected[r] = slot; break; }
        if (selected[r] == UINT32_MAX) { out.status = moe_mmid_lease_status::BUSY; return out; }
        const size_t ai = p * MOE_MMID_WORKSPACE_DEPTH + selected[r];
        if (ai >= context->authorities.size()) return out;
        auto authority = std::static_pointer_cast<moe_mmid_registry_lease::authority>(context->authorities[ai]);
        if (!authority || authority->pool.get() != pool.get()) return out;
        auto & lease = out.bundle.leases_[r];
        lease.authority_ = std::move(authority);
        lease.generation_ = pool->slots[selected[r]].generation + 1;
        lease.queue_cookie_ = request.owners[r].queue_cookie;
        lease.epoch_ = snapshot->key.epoch.value;
        lease.admitted_geometry_ = requested_geometry[r];
        const size_t device_base = static_cast<size_t>(selected[r]) * pool->geometry.device_slot_bytes;
        const size_t host_base = static_cast<size_t>(selected[r]) * pool->geometry.host_slot_bytes;
        lease.admitted_slices_.activation_f32 = retained_subrange(
            pool->device_pool, device_base + pool->geometry.activation_f32_offset,
            requested_geometry[r].activation_f32_bytes);
        lease.admitted_slices_.activation_q8 = retained_subrange(
            pool->device_pool, device_base + pool->geometry.activation_q8_offset,
            requested_geometry[r].activation_q8_bytes);
        lease.admitted_slices_.output_f32 = retained_subrange(
            pool->device_pool, device_base + pool->geometry.output_f32_offset,
            requested_geometry[r].output_f32_bytes);
        lease.admitted_slices_.output_q8 = retained_subrange(
            pool->device_pool, device_base + pool->geometry.output_q8_offset,
            requested_geometry[r].output_q8_bytes);
        if (requested_geometry[r].host_slot_bytes != 0)
            lease.admitted_slices_.host = retained_subrange(pool->host_pool, host_base,
                                                            requested_geometry[r].host_slot_bytes);
        if (!lease.admitted_slices_.activation_f32.valid() || !lease.admitted_slices_.activation_q8.valid() ||
            !lease.admitted_slices_.output_f32.valid() || !lease.admitted_slices_.output_q8.valid() ||
            (requested_geometry[r].host_slot_bytes != 0 && !lease.admitted_slices_.host.valid())) return out;
        out.bundle.owners_[r] = request.owners[r];
    }

    const uint64_t capability = moe_mmid_mint_monotonic_cookie(next_bundle_capability);
    if (capability == 0) return out;
    for (size_t r = 0; r < request.owners.size(); ++r) {
        auto & slot = context->pools[pool_map[r]]->slots[selected[r]];
        slot.generation = out.bundle.leases_[r].generation_;
        slot.epoch = snapshot->key.epoch.value;
        slot.quarantined = false;
        slot.busy = true;
    }
    out.bundle.owner_count_ = request.owners.size();
    out.bundle.identities_ = bindings;
    out.bundle.graph_snapshot_ = snapshot;
    out.bundle.epoch_ = snapshot->key.epoch.value;
    out.bundle.plan_identity_ = request.plan_identity;
    out.bundle.submit_device_ = request.submit_device;
    out.bundle.K_ = request.K; out.bundle.N_ = request.N; out.bundle.type_ = request.type;
    out.bundle.identity_digest_ = digest != 0 ? digest : 1;
    out.bundle.capability_ = capability;
    out.bundle.admitted_ = true;
    out.status = moe_mmid_lease_status::ACQUIRED;
    return out;
}

moe_mmid_admitted_result moe_mmid_workspace_registry::admit(
    moe_mmid_authoritative_admission_request && request) noexcept {
    moe_mmid_admitted_result out;
    auto authority = std::move(request.authority.state_);
    if (!authority || !authority->plan || authority->plan_identity == 0 || authority->epoch == 0) return out;

    // GRAPH is an adapter over the already-proven publication path. DIRECT is
    // admitted below from the exact outer-invocation pin without inventing a
    // graph token.
    if (authority->graph_snapshot) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            bool exact = false;
            for (const auto & context : state_->contexts)
                exact |= same_token(context->token, authority->token) &&
                         context->plan_identity == authority->plan_identity &&
                         context->submit_device == authority->submit_device && context->accepting &&
                         context->exact_plan.get() == authority->plan.get();
            if (!exact) return out;
        }
        moe_mmid_admission_request legacy;
        legacy.token = authority->token;
        legacy.plan_identity = authority->plan_identity;
        legacy.submit_device = authority->submit_device;
        legacy.graph_registry = authority->graph_registry;
        legacy.graph_token = authority->graph_token;
        legacy.graph_snapshot = authority->graph_snapshot;
        legacy.retained_occurrences = authority->occurrences;
        legacy.table_owner = authority->table;
        legacy.top_k = request.shape.top_k; legacy.ne11 = request.shape.ne11;
        legacy.K = request.shape.K; legacy.N = request.shape.N; legacy.type = request.shape.type;
        legacy.owners = authority->owners;
        out = admit(legacy);
        if (out.status == moe_mmid_lease_status::ACQUIRED) out.bundle.common_authority_ = std::move(authority);
        return out;
    }

    const auto & bindings = authority->occurrences;
    if (!authority->token.valid() || authority->submit_device < 0 || !bindings || bindings->empty() ||
        !authority->table || request.shape.top_k == 0 || request.shape.ne11 == 0 || request.shape.K <= 0 ||
        request.shape.N <= 0 || request.shape.type < 0 || authority->owners.empty() ||
        authority->owners.size() > execution::max_devices || bindings->size() % request.shape.top_k != 0 ||
        authority->queues.size() != authority->owners.size()) return out;

    uint64_t digest = 1469598103934665603ULL;
    auto mix = [&](uint64_t value) { digest = (digest ^ value) * 1099511628211ULL; };
    for (size_t i = 0; i < bindings->size(); ++i) {
        const auto & binding = bindings->at(i); const auto & id = binding.identity;
        if (id.allocation_id == 0 || id.generation == 0 || id.layout_id == 0 || id.device < 0 || id.byte_size == 0 ||
            id.occurrence != i || id.byte_offset > binding.owner.extent() ||
            id.byte_size > binding.owner.extent() - id.byte_offset || id.allocation_id != binding.owner.allocation_id() ||
            id.generation != binding.owner.generation() || id.device != binding.owner.device()) return out;
        bool retained = false;
        for (const auto & entry : authority->table->entries())
            retained |= entry.allocation_id() == id.allocation_id && entry.generation() == id.generation &&
                        entry.device() == id.device && entry.extent() == binding.owner.extent();
        if (!retained) return out;
        mix(id.allocation_id); mix(id.generation); mix(id.layout_id); mix(static_cast<uint64_t>(id.device));
        mix(id.byte_offset); mix(id.byte_size); mix(id.occurrence);
    }

    std::shared_ptr<registry_context> context;
    std::array<size_t, execution::max_devices> pool_map{}, order{};
    std::array<uint32_t, execution::max_devices> selected{};
    std::array<std::unique_lock<std::mutex>, execution::max_devices> locks{};
    pool_map.fill(SIZE_MAX); selected.fill(UINT32_MAX);
    std::lock_guard<std::mutex> registry_lock(state_->mutex);
    for (const auto & candidate : state_->contexts) {
        if (same_token(candidate->token, authority->token) && candidate->plan_identity == authority->plan_identity &&
            candidate->submit_device == authority->submit_device && candidate->accepting &&
            candidate->exact_plan.get() == authority->plan.get()) { context = candidate; break; }
    }
    if (!context || context->pools.size() != authority->owners.size() || context->queues.size() != authority->queues.size())
        return out;

    const size_t capacity = bindings->size() / request.shape.top_k;
    std::array<moe_mmid_workspace_geometry, execution::max_devices> geometry{};
    for (size_t r = 0; r < authority->owners.size(); ++r) {
        const auto & wanted = authority->owners[r]; const auto & queue = authority->queues[r];
        if (!queue.valid() || queue.owner_device_ != wanted.owner_device || queue.cookie_ != wanted.queue_cookie) return out;
        for (size_t p = 0; p < context->pools.size(); ++p) {
            const auto & pool = context->pools[p];
            if (pool->owner_device != wanted.owner_device) continue;
            if (pool_map[r] != SIZE_MAX || p >= context->queues.size() ||
                context->queues[p].identity_.get() != queue.identity_.get() ||
                context->queues[p].queue_object_ != queue.queue_object_ || pool->queue_cookie != queue.cookie_) return out;
            for (size_t prior = 0; prior < r; ++prior) if (pool_map[prior] == p) return out;
            const auto & available = pool->geometry; auto & needed = geometry[r];
            if (!moe_mmid_plan_workspace({ static_cast<size_t>(request.shape.K), static_cast<size_t>(request.shape.N),
                                            request.shape.ne11, request.shape.top_k, capacity },
                                          wanted.owner_device != authority->submit_device, &needed) ||
                available.activation_rows < needed.activation_rows || available.occurrences < needed.occurrences ||
                available.q8_ne10_row_bytes < needed.q8_ne10_row_bytes ||
                available.q8_ne01_row_bytes < needed.q8_ne01_row_bytes ||
                available.activation_f32_bytes < needed.activation_f32_bytes ||
                available.activation_q8_bytes < needed.activation_q8_bytes ||
                available.output_f32_bytes < needed.output_f32_bytes || available.output_q8_bytes < needed.output_q8_bytes ||
                available.host_slot_bytes < needed.host_slot_bytes) return out;
            pool_map[r] = p;
        }
        if (pool_map[r] == SIZE_MAX) return out; order[r] = r;
    }
    std::sort(order.begin(), order.begin() + authority->owners.size(), [&](size_t a, size_t b) {
        return authority->owners[a].owner_device < authority->owners[b].owner_device;
    });
    for (size_t i = 0; i < authority->owners.size(); ++i)
        locks[i] = std::unique_lock<std::mutex>(context->pools[pool_map[order[i]]]->mutex);
    for (size_t r = 0; r < authority->owners.size(); ++r) {
        const size_t p = pool_map[r]; const auto & pool = context->pools[p];
        for (uint32_t slot = 0; slot < MOE_MMID_WORKSPACE_DEPTH; ++slot)
            if (!pool->slots[slot].busy && pool->slots[slot].generation != UINT64_MAX) { selected[r] = slot; break; }
        if (selected[r] == UINT32_MAX) { out.status = moe_mmid_lease_status::BUSY; return out; }
        const size_t ai = p * MOE_MMID_WORKSPACE_DEPTH + selected[r];
        if (ai >= context->authorities.size()) return out;
        auto lease_authority = std::static_pointer_cast<moe_mmid_registry_lease::authority>(context->authorities[ai]);
        if (!lease_authority || lease_authority->pool.get() != pool.get()) return out;
        auto & lease = out.bundle.leases_[r]; lease.authority_ = std::move(lease_authority);
        lease.generation_ = pool->slots[selected[r]].generation + 1; lease.queue_cookie_ = authority->queues[r].cookie_;
        lease.epoch_ = authority->epoch; lease.admitted_geometry_ = geometry[r];
        const size_t db = static_cast<size_t>(selected[r]) * pool->geometry.device_slot_bytes;
        const size_t hb = static_cast<size_t>(selected[r]) * pool->geometry.host_slot_bytes;
        lease.admitted_slices_.activation_f32 = retained_subrange(pool->device_pool, db + pool->geometry.activation_f32_offset, geometry[r].activation_f32_bytes);
        lease.admitted_slices_.activation_q8 = retained_subrange(pool->device_pool, db + pool->geometry.activation_q8_offset, geometry[r].activation_q8_bytes);
        lease.admitted_slices_.output_f32 = retained_subrange(pool->device_pool, db + pool->geometry.output_f32_offset, geometry[r].output_f32_bytes);
        lease.admitted_slices_.output_q8 = retained_subrange(pool->device_pool, db + pool->geometry.output_q8_offset, geometry[r].output_q8_bytes);
        if (geometry[r].host_slot_bytes) lease.admitted_slices_.host = retained_subrange(pool->host_pool, hb, geometry[r].host_slot_bytes);
        if (!lease.admitted_slices_.activation_f32.valid() || !lease.admitted_slices_.activation_q8.valid() ||
            !lease.admitted_slices_.output_f32.valid() || !lease.admitted_slices_.output_q8.valid() ||
            (geometry[r].host_slot_bytes && !lease.admitted_slices_.host.valid())) return out;
        out.bundle.owners_[r] = authority->owners[r];
    }
    const uint64_t capability = moe_mmid_mint_monotonic_cookie(next_bundle_capability); if (!capability) return out;
    for (size_t r = 0; r < authority->owners.size(); ++r) { auto & slot = context->pools[pool_map[r]]->slots[selected[r]];
        slot.generation = out.bundle.leases_[r].generation_; slot.epoch = authority->epoch; slot.quarantined = false; slot.busy = true; }
    out.bundle.owner_count_ = authority->owners.size(); out.bundle.identities_ = bindings;
    out.bundle.common_authority_ = std::move(authority); out.bundle.epoch_ = out.bundle.common_authority_->epoch;
    out.bundle.plan_identity_ = out.bundle.common_authority_->plan_identity; out.bundle.submit_device_ = out.bundle.common_authority_->submit_device;
    out.bundle.K_ = request.shape.K; out.bundle.N_ = request.shape.N; out.bundle.type_ = request.shape.type;
    out.bundle.identity_digest_ = digest ? digest : 1; out.bundle.capability_ = capability;
    out.bundle.admitted_ = true; out.status = moe_mmid_lease_status::ACQUIRED; return out;
}

size_t moe_mmid_workspace_registry::invalidate_plan(const lifecycle_plan_snapshot * exact_plan) noexcept {
    if (!exact_plan) return 0;
    size_t invalidated = 0;
    try {
        std::lock_guard<std::mutex> lock(state_->mutex);
        for (const auto & context : state_->contexts)
            if (context->accepting && context->exact_plan.get() == exact_plan) { context->accepting = false; ++invalidated; }
    } catch (...) { return 0; }
    return invalidated;
}

size_t moe_mmid_workspace_registry::recover_quarantined(const moe_mmid_model_token & token, uint64_t plan_identity,
                                                         bool wait) noexcept {
    if (!token.valid() || plan_identity == 0) return 0;
    size_t released = 0;
    std::lock_guard<std::mutex> registry_lock(state_->mutex);
    for (const auto & context : state_->contexts) {
        if (!same_token(context->token, token) || context->plan_identity != plan_identity) continue;
        for (const auto & pool : context->pools) {
            std::lock_guard<std::mutex> pool_lock(pool->mutex);
            for (auto & slot : pool->slots) {
                if (!slot.busy || !slot.quarantined || (!slot.quarantine_graph && !slot.quarantine_authority)) continue;
                auto common = std::static_pointer_cast<workspace_admission_authority::state>(slot.quarantine_authority);
                const auto & terminals = slot.quarantine_graph ? slot.quarantine_graph->terminals : common->terminals;
                const auto & proofs = slot.quarantine_graph ? slot.quarantine_graph->quiescence_proofs : common->quiescence_proofs;
                const auto terminal = terminals.find(pool->owner_device);
                const auto proof = proofs.find(pool->owner_device);
                bool ready = terminal != terminals.end() && terminal->second && terminal->second->ready();
                if (!ready && proof != proofs.end() && proof->second)
                    ready = wait ? proof->second->wait_and_confirm() : proof->second->ready();
                if (!ready) continue;
                slot.busy = false;
                slot.quarantined = false;
                slot.epoch = 0;
                slot.quarantine_graph.reset();
                slot.quarantine_authority.reset();
                ++released;
            }
        }
    }
    return released;
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
                               if (!same_token(context->token, token) ||
                                   (plan_identity != 0 && context->plan_identity != plan_identity)) return false;
                               for (const auto & pool : context->pools) {
                                   std::lock_guard<std::mutex> pool_lock(pool->mutex);
                                   for (const auto & slot : pool->slots) if (slot.quarantined) return false;
                               }
                               return true;
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
