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

}  // namespace ggml_sycl
