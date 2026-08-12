// Pure MMID workspace geometry and fixed-slot lease contracts.
#pragma once

#include "moe-graph-retention.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

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

    size_t activation_f32_offset = 0;
    size_t activation_f32_bytes  = 0;
    size_t activation_q8_offset  = 0;
    size_t activation_q8_bytes   = 0;
    size_t output_f32_offset     = 0;
    size_t output_f32_bytes      = 0;
    size_t output_q8_offset      = 0;
    size_t output_q8_bytes       = 0;
    size_t device_slot_bytes     = 0;

    size_t descriptor_host_bytes          = 0;
    size_t secondary_activation_d2h_bytes = 0;
    size_t secondary_activation_h2d_bytes = 0;
    size_t secondary_output_d2h_bytes     = 0;
    size_t secondary_output_h2d_bytes     = 0;
    size_t secondary_bounce_bytes         = 0;
    size_t host_slot_bytes                = 0;
};

bool moe_mmid_capacity(size_t n_ctx, size_t n_ubatch, size_t n_seq_max, size_t * out) noexcept;
bool moe_mmid_q8_1_row_bytes(size_t elements, size_t * out) noexcept;
bool moe_mmid_plan_workspace(const moe_mmid_shape &        shape,
                             bool                          secondary_owner,
                             moe_mmid_workspace_geometry * out) noexcept;
bool moe_mmid_component_max(moe_mmid_workspace_geometry *       aggregate,
                            const moe_mmid_workspace_geometry & candidate) noexcept;
// Validates registry-consumable canonical/component-max geometry without
// requiring a fictitious single source shape.
bool moe_mmid_validate_workspace_geometry(const moe_mmid_workspace_geometry & geometry,
                                          bool secondary_owner) noexcept;
bool moe_mmid_checked_pool_bytes(const moe_mmid_workspace_geometry & geometry,
                                 uint32_t                            depth,
                                 size_t *                            device_bytes,
                                 size_t *                            host_bytes) noexcept;
bool moe_mmid_checked_zone_total(size_t base_bytes, size_t workspace_bytes, size_t * out) noexcept;
bool moe_mmid_checked_product(size_t count, size_t bytes, size_t * out) noexcept;
bool moe_mmid_debit_device_budget(size_t workspace_bytes, size_t * remaining_bytes) noexcept;

// Mints [1, UINT64_MAX-1] exactly once. Saturates fail-closed at exhaustion.
// The caller-owned atomic permits deterministic near-overflow testing.
uint64_t moe_mmid_mint_monotonic_cookie(std::atomic<uint64_t> & last_issued) noexcept;

struct moe_mmid_owner_accounting {
    int    owner_device    = -1;
    size_t budget_bytes    = 0;
    size_t used_bytes      = 0;
    size_t workspace_bytes = 0;
};

// Atomically adds finalized actual-owner workspace charges. Owners absent from
// `charges` are untouched; failure writes neither per-owner usage nor total.
bool moe_mmid_account_actual_owners(const std::vector<std::pair<int, size_t>> & charges,
                                    std::vector<moe_mmid_owner_accounting> *    owners,
                                    size_t *                                    total_vram_bytes) noexcept;

// Atomically replaces one already-accounted owner charge set with another.
bool moe_mmid_rebuild_per_device_usage(const std::vector<size_t> &                 base_usage,
                                       const std::vector<std::pair<int, size_t>> & kv_charges,
                                       const std::vector<std::pair<int, size_t>> & mmid_charges,
                                       const std::vector<int> &                    devices,
                                       const std::vector<size_t> &                 budgets,
                                       std::vector<size_t> *                       used) noexcept;

// Checked single-device admission shared by stable-fit and replacement paths.
bool moe_mmid_admit_single_device_total(size_t   base_bytes,
                                        size_t   workspace_bytes,
                                        size_t   budget_bytes,
                                        size_t * admitted_total) noexcept;

bool moe_mmid_reaccount_replacement(const std::vector<std::pair<int, size_t>> & old_charges,
                                    const std::vector<std::pair<int, size_t>> & new_charges,
                                    const std::vector<int> &                    devices,
                                    const std::vector<size_t> &                 budgets,
                                    std::vector<size_t> *                       used,
                                    size_t *                                    total) noexcept;

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
    bool                    set_generation_for_test(uint32_t slot, uint64_t generation) noexcept;

  private:
    struct state;
    std::unique_ptr<state> state_;
};

// Dependency-free materialization contracts. Production wraps each allocation's
// mem_handle in `owner`; host tests use the same API with fake byte blobs.
struct moe_mmid_blob {
    std::shared_ptr<void>                                owner;
    std::function<std::shared_ptr<void>(size_t, size_t)> slice_owner;
    void *                                               ptr         = nullptr;
    size_t                                               bytes       = 0;
    int                                                  device      = -1;
    bool                                                 host_pinned = false;

    bool valid() const noexcept { return owner && ptr != nullptr && bytes != 0; }

    moe_mmid_blob slice(size_t offset, size_t length) const noexcept;
};

struct moe_mmid_model_token {
    uint64_t model_id    = 0;
    uint64_t load_txn_id = 0;
    uint64_t generation  = 0;

    bool valid() const noexcept { return model_id != 0 && load_txn_id != 0 && generation != 0; }
};

struct moe_mmid_materialized_owner_plan {
    int                         owner_device = -1;
    uint64_t                    queue_cookie = 0;
    bool                        secondary_owner = false;
    moe_mmid_workspace_geometry geometry;
    size_t                      device_pool_bytes = 0;
    size_t                      host_pool_bytes   = 0;
};

using moe_mmid_blob_allocator =
    std::function<moe_mmid_blob(bool host_pinned, int device, size_t bytes, size_t alignment)>;

enum class moe_mmid_materialize_status : uint8_t { PUBLISHED, ALREADY_PUBLISHED, INVALID, ALLOCATION_FAILED };

struct moe_mmid_materialized_slices {
    moe_mmid_blob activation_f32;
    moe_mmid_blob activation_q8;
    moe_mmid_blob output_f32;
    moe_mmid_blob output_q8;
    moe_mmid_blob host;
};

class moe_mmid_registry_lease {
  public:
    bool                                 valid() const noexcept;
    uint32_t                             slot() const noexcept;
    uint64_t                             generation() const noexcept;
    uint64_t                             plan_identity() const noexcept;
    uint64_t                             queue_cookie() const noexcept;
    uint64_t                             epoch() const noexcept;
    int                                  submit_device() const noexcept;
    int                                  owner_device() const noexcept;
    const moe_mmid_workspace_geometry &  geometry() const noexcept;
    const moe_mmid_materialized_slices & slices() const noexcept;
    moe_mmid_release_status              terminal_release(uint64_t queue_cookie, uint64_t generation) noexcept;

  private:
    struct authority;
    std::shared_ptr<authority> authority_;
    uint64_t                   generation_   = 0;
    uint64_t                   queue_cookie_ = 0;
    uint64_t                   epoch_        = 0;
    friend class moe_mmid_workspace_registry;
    friend class moe_admitted_workspace_bundle;
};

struct moe_mmid_admission_owner {
    int      owner_device = -1;
    uint64_t queue_cookie = 0;
};

struct moe_mmid_admission_request {
    moe_mmid_model_token                    token;
    uint64_t                                plan_identity = 0;
    int                                     submit_device = -1;
    moe::graph_retention_registry *         graph_registry = nullptr;
    moe::published_graph_token              graph_token;
    std::shared_ptr<const moe::graph_retention_record> graph_snapshot;
    std::shared_ptr<const std::vector<moe::mmid_batch_binding>> retained_occurrences;
    std::shared_ptr<const moe::graph_private_table_owner> table_owner;
    size_t                                  top_k = 0;
    size_t                                  ne11 = 0;
    int64_t                                 K = 0;
    int64_t                                 N = 0;
    int32_t                                 type = -1;
    std::vector<moe_mmid_admission_owner>   owners;
};

class moe_admitted_workspace_bundle {
  public:
    moe_admitted_workspace_bundle() = default;
    ~moe_admitted_workspace_bundle();
    moe_admitted_workspace_bundle(const moe_admitted_workspace_bundle &) = delete;
    moe_admitted_workspace_bundle & operator=(const moe_admitted_workspace_bundle &) = delete;
    moe_admitted_workspace_bundle(moe_admitted_workspace_bundle && other) noexcept;
    moe_admitted_workspace_bundle & operator=(moe_admitted_workspace_bundle && other) noexcept;

    bool valid() const noexcept;
    bool matches(int submit_device, int owner_device, int64_t K, int64_t N, int32_t type,
                 const moe::mmid_operand_identity * identities = nullptr, size_t identity_count = 0) const noexcept;
    uint64_t identity_digest() const noexcept { return identity_digest_; }
    uint64_t epoch() const noexcept { return epoch_; }
    uint64_t plan_identity() const noexcept { return plan_identity_; }
    size_t owner_count() const noexcept { return owner_count_; }
    const moe_mmid_registry_lease * owner_leases() const noexcept { return leases_.data(); }
    const moe_mmid_admission_owner * graph_owners() const noexcept { return owners_.data(); }
    const std::vector<moe::mmid_batch_binding> & retained_occurrences() const noexcept;

    // Must be called before any queue submission. Once possible submission is
    // recorded, destruction quarantines the slots; only terminal_release after
    // the caller has drained the queues may recycle them.
    bool mark_possible_submit() noexcept;
    bool terminal_release() noexcept;
    bool quarantined() const noexcept { return possible_submit_ && admitted_; }

  private:
    bool release_all() noexcept;
    bool                                    admitted_       = false;
    bool                                    possible_submit_ = false;
    size_t                                  owner_count_     = 0;
    uint64_t                                capability_     = 0;
    uint64_t                                identity_digest_ = 0;
    uint64_t                                epoch_          = 0;
    uint64_t                                plan_identity_  = 0;
    int                                     submit_device_  = -1;
    int64_t                                 K_              = 0;
    int64_t                                 N_              = 0;
    int32_t                                 type_           = -1;
    std::array<moe_mmid_registry_lease, execution::max_devices> leases_{};
    std::array<moe_mmid_admission_owner, execution::max_devices> owners_{};
    std::shared_ptr<const std::vector<moe::mmid_batch_binding>> identities_;
    std::shared_ptr<const moe::graph_retention_record> graph_snapshot_;
    friend class moe_mmid_workspace_registry;
};

struct moe_mmid_admitted_result {
    moe_mmid_lease_status          status = moe_mmid_lease_status::INVALID;
    moe_admitted_workspace_bundle bundle;
};

struct moe_mmid_registry_context_info {
    moe_mmid_model_token token;
    uint64_t             plan_identity = 0;
    int                  submit_device = -1;
    size_t               owner_count   = 0;
};

struct moe_mmid_registry_lease_result {
    moe_mmid_lease_status   status = moe_mmid_lease_status::INVALID;
    moe_mmid_registry_lease lease;
};

class moe_mmid_workspace_registry {
  public:
    moe_mmid_workspace_registry();
    ~moe_mmid_workspace_registry();
    moe_mmid_workspace_registry(const moe_mmid_workspace_registry &)             = delete;
    moe_mmid_workspace_registry & operator=(const moe_mmid_workspace_registry &) = delete;

    moe_mmid_materialize_status    materialize(const moe_mmid_model_token &                          token,
                                               uint64_t                                              plan_identity,
                                               int                                                   submit_device,
                                               const std::vector<moe_mmid_materialized_owner_plan> & owners,
                                               const moe_mmid_blob_allocator &                       allocator) noexcept;
    moe_mmid_registry_lease_result acquire(const moe_mmid_model_token & token,
                                           uint64_t                     plan_identity,
                                           int                          submit_device,
                                           int                          owner_device,
                                           uint64_t                     queue_cookie) noexcept;
    moe_mmid_admitted_result       admit(const moe_mmid_admission_request & request) noexcept;
    size_t                         recover_quarantined(const moe_mmid_model_token & token, uint64_t plan_identity,
                                                       bool wait) noexcept;
    bool                           retire(const moe_mmid_model_token & token, uint64_t plan_identity = 0) noexcept;
    size_t                         published_contexts() const noexcept;
    std::vector<moe_mmid_registry_context_info> list() const;

  private:
    struct state;
    std::unique_ptr<state> state_;
};

}  // namespace ggml_sycl
