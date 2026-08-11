// Pure MMID workspace geometry and fixed-slot lease contracts.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ggml_sycl {

constexpr size_t   MOE_MMID_DEVICE_ALIGNMENT = 256;
constexpr uint32_t MOE_MMID_WORKSPACE_DEPTH  = 2;

// Host/device ABI copied once per routed occurrence.
struct alignas(16) moe_mmid_descriptor {
    uint32_t token_index = 0;
    uint32_t slot_index  = 0;
    int32_t  expert_id   = -1;
    uint32_t flags       = 0;
};

static_assert(sizeof(moe_mmid_descriptor) == 16, "MMID descriptor ABI must remain 16 bytes");

struct moe_mmid_shape {
    size_t ne10  = 0;  // activation columns (K)
    size_t ne01  = 0;  // result columns (N)
    size_t ne11  = 0;  // activation broadcast rows: exactly 1 or top_k
    size_t top_k = 0;
    size_t c     = 0;  // token/sequence capacity
};

// Every named device slice starts on a 256-byte boundary. The host descriptor
// and four bounce slices are separately accounted and are never hidden in a
// device total.
struct moe_mmid_workspace_geometry {
    bool   valid             = false;
    size_t activation_rows   = 0;  // A = ne11*c
    size_t occurrences       = 0;  // O = top_k*c
    size_t q8_ne10_row_bytes = 0;
    size_t q8_ne01_row_bytes = 0;

    size_t activation_f32_bytes = 0;
    size_t activation_q8_bytes  = 0;
    size_t output_f32_bytes     = 0;
    size_t output_q8_bytes      = 0;
    size_t device_slot_bytes    = 0;

    size_t descriptor_host_bytes          = 0;
    size_t secondary_activation_d2h_bytes = 0;
    size_t secondary_activation_h2d_bytes = 0;
    size_t secondary_output_d2h_bytes     = 0;
    size_t secondary_output_h2d_bytes     = 0;
    size_t secondary_bounce_bytes         = 0;
    size_t host_slot_bytes                = 0;
};

bool moe_mmid_q8_1_row_bytes(size_t elements, size_t * out) noexcept;
bool moe_mmid_plan_workspace(const moe_mmid_shape &        shape,
                             bool                          secondary_owner,
                             moe_mmid_workspace_geometry * out) noexcept;
bool moe_mmid_component_max(moe_mmid_workspace_geometry *       aggregate,
                            const moe_mmid_workspace_geometry & candidate) noexcept;
bool moe_mmid_checked_pool_bytes(const moe_mmid_workspace_geometry & geometry,
                                 uint32_t                            depth,
                                 size_t *                            device_bytes,
                                 size_t *                            host_bytes) noexcept;

enum class moe_mmid_lease_status : uint8_t { ACQUIRED, BUSY, INVALID };
enum class moe_mmid_release_status : uint8_t { RELEASED, STALE, WRONG_QUEUE, WRONG_EPOCH, INVALID };

class moe_mmid_workspace_pool;

// Opaque authority: callers can inspect identity metadata but cannot forge a
// releasable lease because the pool validates a private pool identity.
class moe_mmid_workspace_lease {
  public:
    bool valid() const noexcept { return pool_identity_ != 0 && slot_ != UINT32_MAX; }

    uint32_t slot() const noexcept { return slot_; }

    uint64_t generation() const noexcept { return generation_; }

    uint64_t queue_identity() const noexcept { return queue_identity_; }

    uint64_t busy_epoch() const noexcept { return busy_epoch_; }

  private:
    uint64_t pool_identity_  = 0;
    uint32_t slot_           = UINT32_MAX;
    uint64_t generation_     = 0;
    uint64_t queue_identity_ = 0;
    uint64_t busy_epoch_     = 0;
    friend class moe_mmid_workspace_pool;
};

struct moe_mmid_lease_result {
    moe_mmid_lease_status    status = moe_mmid_lease_status::INVALID;
    moe_mmid_workspace_lease lease;
};

class moe_mmid_workspace_pool {
  public:
    explicit moe_mmid_workspace_pool(uint32_t depth = MOE_MMID_WORKSPACE_DEPTH);
    ~moe_mmid_workspace_pool();
    moe_mmid_workspace_pool(const moe_mmid_workspace_pool &)             = delete;
    moe_mmid_workspace_pool & operator=(const moe_mmid_workspace_pool &) = delete;

    uint32_t                depth() const noexcept;
    moe_mmid_lease_result   acquire(uint64_t queue_identity, uint64_t busy_epoch) noexcept;
    moe_mmid_release_status terminal_release(const moe_mmid_workspace_lease & lease,
                                             uint64_t                         queue_identity,
                                             uint64_t                         terminal_epoch) noexcept;

  private:
    struct state;
    std::unique_ptr<state> state_;
};

}  // namespace ggml_sycl
