//
// Host zone routing, host-allocation lifetimes, the host SCRATCH demand ledger, and the host holder plan
// (docs/design/sycl-mmap-host-tier.md §4.4: "Host-side non-weight requests: routed by role", "Lifetimes", the notify
// path, "DEVICE holders", and the attention host staging plan).
//
// Every decision here is a pure function or a small state object over ggml-base types and caller-supplied callbacks,
// so the four host-only tests (test-sycl-host-zone-select, test-sycl-host-scratch-demand, test-sycl-plan-walk-sched,
// test-sycl-attn-host-staging-plan) drive exactly the code the backend runs. The backend supplies the real
// allocator, unified cache and placement plan through the callbacks; the tests supply fakes. No SYCL header is
// included, and this TU is also globbed into libggml-sycl, where it is the production copy.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "host-zone-kinds.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_sycl {

// ---------------------------------------------------------------------------------------------------------------
// Routing (§4.4 "Route by role, in select_zone, the one place").
// ---------------------------------------------------------------------------------------------------------------

// The host zone a host request lands in, in rule order:
//   role KV, or a device-KV spill to host                     -> KV
//   role WEIGHT                                               -> WEIGHT
//   role EXPERT_STAGING                                       -> SCRATCH
//   lifetime UNOWNED, or a declared persistent owner          -> WEIGHT
//   any other role with category HOST_COMPUTE or EXPERT_CACHE -> SCRATCH
//   anything else                                             -> STAGING
// No category clause selects WEIGHT. A persistent owner (alloc_constraints::persistent_owner) is a lifetime claim
// that outlives every graph, so it may carry neither role KV nor EXPERT_STAGING nor be a KV spill: those choose KV and
// SCRATCH before WEIGHT is considered, and the combination is asserted against rather than routed.
host_zone_id select_zone(alloc_role       role,
                         runtime_category category,
                         alloc_lifetime   lifetime,
                         bool             persistent_owner,
                         bool             kv_spill_to_host);

// ---------------------------------------------------------------------------------------------------------------
// Lifetimes and owner boundaries (§4.4 "Lifetimes", §M57 (b), §M62 (D)).
// ---------------------------------------------------------------------------------------------------------------

const char * alloc_lifetime_name(alloc_lifetime lifetime);
const char * alloc_owner_kind_name(alloc_owner_kind kind);

// A fresh owner serial for a buffer or a backend context. Never 0, so 0 always means "no owner".
uint64_t host_zone_owner_next_id();

// One registry row as the lifetime checks see it.
struct host_zone_record_view {
    host_zone_id    zone     = host_zone_id::COUNT;
    alloc_lifetime  lifetime = alloc_lifetime::GRAPH;
    alloc_owner_ref owner{};
    const char *    cohort = nullptr;
    size_t          bytes  = 0;
};

// Whether a row counts at the SCRATCH/STAGING graph-boundary check: only GRAPH rows do. A CONTEXT-lifetime compute
// buffer, a MODEL row, a DEVICE holder and an UNOWNED holder are live there by design.
bool host_zone_lifetime_counts_at_graph_boundary(alloc_lifetime lifetime);

// The rows of `zone` the graph-boundary check counts.
size_t host_zone_graph_boundary_live(const host_zone_record_view * records, size_t n, host_zone_id zone);

// Whether `record` belongs to the owner whose boundary is being checked: the same lifetime AND the same owner. A row of
// another context or model with the same lifetime is not this owner's, which is the multi-model case.
bool host_zone_record_owned_by(const host_zone_record_view & record, alloc_lifetime lifetime, alloc_owner_ref owner);

// The owner-boundary check: every row of `records` that host_zone_record_owned_by(lifetime, owner) and is still live
// prints `[HOST-ZONE] leaked <CONTEXT|MODEL|DEVICE> record: <cohort> <bytes> owner=<kind>/<id>` at WARN. With
// `strict` (GGML_SYCL_STRICT_LEASES=1, read by the caller through the cache's existing accessor) a leak aborts after
// every line has printed. Returns the number of leaked rows.
size_t host_zone_owner_boundary_check(const host_zone_record_view * records,
                                      size_t                        n,
                                      alloc_lifetime                lifetime,
                                      alloc_owner_ref               owner,
                                      bool                          strict);

// The DEVICE check runs after `delete sycl_ctx` only when the backend just freed was the last live one on its device
// (§M62 (D)): a second live backend there still shares every DEVICE holder.
bool host_zone_device_check_due(int live_backends_on_device_after_free);

// ---------------------------------------------------------------------------------------------------------------
// The host SCRATCH demand ledger (§4.4, "What the SYCL side does with it").
// ---------------------------------------------------------------------------------------------------------------

// The zone the ledger ensures, as the device's unified cache presents it.
struct host_scratch_zone_ops {
    void * user                               = nullptr;
    // SCRATCH's committed capacity now.
    size_t (*capacity)(void * user)           = nullptr;
    // One grow_zone(SCRATCH, request) call. False when the zone could not grow.
    bool (*grow)(void * user, size_t request) = nullptr;
    // The plan's host_zone_scratch_bytes: the SCRATCH reserve the demands sit on top of.
    size_t plan_scratch_bytes                 = 0;
    // The host buffer chunk cap (GGML_SYCL_HOST_BUFFER_CHUNK_MB): a demand splits into buffers of at most this size.
    size_t chunk_cap                          = 0;
    // pinned_chunk_pool::DEFAULT_ALIGNMENT: the slack one buffer can need, as on the miss path.
    size_t alignment                          = 0;
};

struct host_scratch_ensure_result {
    bool   ok              = true;  // false only when a needed grow failed
    size_t total           = 0;     // the sum of the live demands
    size_t target          = 0;     // plan_scratch_bytes + total
    size_t capacity_before = 0;
    size_t request         = 0;     // the grow request made (0 when nothing grew)
    size_t grown           = 0;     // capacity after - capacity before
};

// The number of buffers `bytes` splits into at `chunk_cap`.
size_t host_scratch_buffers_at_cap(size_t bytes, size_t chunk_cap);

// One device's demands: one per llama_context, keyed by the notifying backend context, and one per DEVICE holder.
// Thread-safe. The ledger's mutex is a leaf held only across its own map updates and arithmetic, never across the
// grow (canonical contract §12.5).
class host_scratch_ledger {
  public:
    // Replaces the context's demand with (compute, output), not adds to it, then ensures capacity. Prints
    // `[HOST-ARENA] SCRATCH demand ctx=... compute=... output=... total=... grown=...` at INFO on every call.
    host_scratch_ensure_result notify_context(uint64_t                      context_key,
                                              size_t                        compute_bytes,
                                              size_t                        output_bytes,
                                              const host_scratch_zone_ops & ops);
    // Replaces the DEVICE holder's demand, then ensures capacity. Prints `[HOST-ARENA] SCRATCH demand holder=<name>
    // dev=... bytes=... total=... grown=...` at INFO.
    host_scratch_ensure_result holder_demand(const char *                  holder,
                                             int                           device,
                                             size_t                        bytes,
                                             const host_scratch_zone_ops & ops);
    // Drops the context's demand at its owner boundary. Capacity is never returned.
    void                       drop_context(uint64_t context_key);
    size_t                     total() const;
    size_t                     context_demand(uint64_t context_key) const;

  private:
    struct demand {
        size_t bytes     = 0;
        size_t n_buffers = 0;
    };

    host_scratch_ensure_result ensure_locked_then_grow(std::unique_lock<std::mutex> & lock,
                                                       const host_scratch_zone_ops &  ops);

    mutable std::mutex            mutex_;
    std::map<uint64_t, demand>    contexts_;
    std::map<std::string, demand> holders_;
};

// The notify's core (ggml_backend_sycl_notify_host_buffer_sizes): refuses a non-SYCL backend with a WARN before it
// reads anything else, resolves the device the host buft allocates from (never the notifying backend's own device and
// never device 0), and records the demand on that device's ledger.
struct host_scratch_context_info {
    uint64_t key    = 0;   // the notifying ggml_backend_sycl_context's owner serial
    int      device = -1;  // that context's own device (the demand's owner, not the zone's device)
};

struct host_scratch_notify_env {
    bool (*is_sycl)(ggml_backend_t backend)                                                   = nullptr;
    // The device of ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend)): ggml_sycl_host_buft_device.
    int (*host_buft_device)(ggml_backend_t backend)                                           = nullptr;
    host_scratch_context_info (*context_of)(ggml_backend_t backend)                           = nullptr;
    // The ledger of `device`'s unified cache and that cache's zone operations; false when the device has no cache.
    bool (*ledger_of)(int device, host_scratch_ledger ** ledger, host_scratch_zone_ops * ops) = nullptr;
};

bool host_scratch_notify(const host_scratch_notify_env & env,
                         ggml_backend_t                  backend,
                         size_t                          host_compute_bytes,
                         size_t                          host_output_bytes);

// ---------------------------------------------------------------------------------------------------------------
// The plan walk (§4.4 "The plan walk is SYCL-side", "The walk runs where the split is live").
// ---------------------------------------------------------------------------------------------------------------

// The buffer a tensor's bytes live in: its view_src's when it is a view of an allocated tensor (a view's own buffer is
// often still NULL before allocation), else its own.
ggml_backend_buffer_t host_tensor_buffer(const ggml_tensor * tensor);

// Whether the tensor's resolved buffer is of the buffer type whose iface.get_name is `get_name`. The keying is the
// function pointer, never a buft pointer or a name string: SYCL_KV_Host copies SYCL_Host's iface and differs only in
// get_name. ggml_sycl_tensor_is_in_kv_host_buft is this with the KV-host type's get_name.
bool host_tensor_in_buft_named(const ggml_tensor * tensor, const char * (*get_name)(ggml_backend_buffer_type_t buft));

typedef bool (*host_plan_backend_pred)(ggml_backend_t backend);
// The scheduler's node-to-backend lookup: production uses ggml_backend_sched_get_tensor_backend.
typedef ggml_backend_t (*host_plan_tensor_backend)(ggml_backend_sched_t sched, ggml_tensor * node);
typedef bool (*host_plan_node_pred)(void * user, const ggml_tensor * node);
typedef void (*host_plan_visit)(void * user, ggml_tensor * node, ggml_backend_t backend);
// Placement's executor fact: whether the plan put at least one expert of src0 on the host for `device`. *known false
// means the fact is absent, and *cause names why ("usage not MOE_EXPERT_WEIGHT" or "no placement plan").
typedef bool (*host_plan_host_experts)(const ggml_tensor * src0, int device, bool * known, const char ** cause);

struct host_plan_attribution {
    int attributed = 0;  // nodes passing node_pred whose scheduler backend passes is_target
    int matched    = 0;  // nodes passing node_pred, whatever their backend
};

// Reads the scheduler's node-to-backend map, which exists only between a split and the next reset:
// ggml_backend_sched_reserve_size leaves it live, ggml_backend_sched_reserve resets at its end. For every node
// passing `node_pred` it asks ggml_backend_sched_get_tensor_backend and calls `visit` for those whose backend passes
// `is_target` (production passes ggml_backend_is_sycl). `user` goes to both callbacks. With `warn_zero`, an attribution of zero prints
// `[ATTN-HOST] plan attributed 0 host-KV nodes to a SYCL backend (%d matched the predicate)` at WARN.
host_plan_attribution host_plan_attribute(ggml_backend_sched_t   sched,
                                          ggml_cgraph *          graph,
                                          host_plan_backend_pred is_target,
                                          host_plan_node_pred    node_pred,
                                          host_plan_visit        visit,
                                          void *                 user,
                                          bool                   warn_zero);

// ---------------------------------------------------------------------------------------------------------------
// The host holder plan: attention host staging slots and the CPU-expert pools, planned per backend context and
// committed per device (§4.4 "Pending, then committed", "The walks of one reserve merge as a maximum").
// ---------------------------------------------------------------------------------------------------------------

enum host_staging_kind {
    HOST_STAGING_Q = 0,
    HOST_STAGING_DST,
    HOST_STAGING_MASK,
    HOST_STAGING_VAL,
    HOST_STAGING_IDS,
    HOST_STAGING_KIND_COUNT,
};

const char * host_staging_kind_name(int kind);

// The slot rule: room for `bytes` plus the 64-byte canary tail, rounded up to 1 MiB, of which all but the tail is
// usable. host_staging_slot_bytes is the allocation; host_staging_slot_capacity the usable capacity. 0 stays 0.
size_t host_staging_slot_bytes(size_t bytes);
size_t host_staging_slot_capacity(size_t bytes);

// The pools' figures: D = the largest top-K x tokens over the host-expert MUL_MAT_ID nodes, K and N their src0's
// largest ne[0] and ne[1].
struct host_pool_dims {
    int64_t D                 = 0;
    int64_t K                 = 0;
    int64_t N                 = 0;
    int     host_expert_nodes = 0;
};

// PinnedBufferPool's bytes at (D, K, N): act D x K and out D x N floats, plus D int32 expert ids.
size_t host_pool_pinned_bytes(const host_pool_dims & dims);

struct host_holder_plan {
    int            device = -1;
    size_t         staging[HOST_STAGING_KIND_COUNT]{};  // the largest ggml_nbytes per kind, unrounded
    host_pool_dims pools{};
};

// What the walk reads from the backend.
struct host_holder_walk_env {
    host_plan_backend_pred   is_target                  = nullptr;
    // Null: ggml_backend_sched_get_tensor_backend. The staging test injects a fixed assignment.
    host_plan_tensor_backend tensor_backend             = nullptr;
    int (*backend_device)(ggml_backend_t backend)       = nullptr;
    uint64_t (*backend_key)(ggml_backend_t backend)     = nullptr;
    // The dispatch's own predicate: the tensor (or the tensor its view_src names) lives in a SYCL_KV_Host buffer.
    bool (*in_kv_host_buft)(const ggml_tensor * tensor) = nullptr;
    // Placement's executor fact, the pools' one eligibility rule (host_pools_init_from_plan's check reads it too).
    host_plan_host_experts host_experts                 = nullptr;
    bool                   host_dispatch_on             = false;
};

// One device's committed figures: the per-field high-water over the live contexts' committed plans.
struct host_holder_device_state {
    bool           planned = false;  // a committed plan exists for the device
    size_t         slot_capacity[HOST_STAGING_KIND_COUNT]{};
    size_t         slot_bytes[HOST_STAGING_KIND_COUNT]{};
    host_pool_dims pools{};
};

class host_holder_plans {
  public:
    // Clears the pending plan of every listed backend context, so no backend carries a walk from an earlier reserve and
    // a failed candidate's walk is never promoted.
    void                     begin(const uint64_t * keys, size_t n);
    // Merges one reserve graph's figures into the pending plans of the backend contexts it attributes nodes to, as a
    // per-field maximum. Registers nothing.
    host_plan_attribution    walk(ggml_backend_sched_t sched, ggml_cgraph * graph, const host_holder_walk_env & env);
    // Consumes the context's pending plan and replaces its committed plan with it, so a second commit from the same
    // context replaces its first. A pending plan the walks attributed nothing to commits as no plan. Returns false when
    // the context had no pending plan (no begin and no walk since its last commit). *device_out is the new plan's
    // device and *previous_out the device of the plan it replaced, each -1 for none: both devices' figures change.
    bool                     commit(uint64_t key, int * device_out, int * previous_out);
    // Drops the context's pending and committed plans at its owner boundary. *device_out is the device its committed
    // plan was on, or -1.
    void                     drop(uint64_t key, int * device_out);
    host_holder_device_state device_state(int device) const;
    // Whether the context holds a pending plan (diagnostics and tests).
    bool                     has_pending(uint64_t key) const;
    bool                     has_committed(uint64_t key) const;

  private:
    struct entry {
        bool             pending_valid   = false;
        bool             committed_valid = false;
        host_holder_plan pending{};
        host_holder_plan committed{};
    };

    mutable std::mutex        mutex_;  // leaf; never held across an allocation, a grow or a callback
    std::map<uint64_t, entry> entries_;
    std::atomic<bool>         staging_off_warned_{ false };
};

// ---------------------------------------------------------------------------------------------------------------
// Attention host staging slots (§4.4 "At dispatch no slot grows").
// ---------------------------------------------------------------------------------------------------------------

struct host_staging_block {
    void * ptr   = nullptr;
    size_t bytes = 0;        // the allocation, slot rule applied
    size_t cap   = 0;        // usable: bytes - 64
    void * token = nullptr;  // the allocator's owner of the block
};

struct host_staging_allocator {
    void * user                                                                              = nullptr;
    // Allocates `bytes` (already slot-rounded) of host-pinned memory on `device`'s cache; fills ptr and token.
    bool (*alloc)(void * user, int device, int kind, size_t bytes, host_staging_block * out) = nullptr;
    // Releases a block alloc produced.
    void (*release)(void * user, host_staging_block * block)                                 = nullptr;
};

class attn_host_dispatch_counters;

// One process's slots, [device][kind]. Touched by commit and drop in sched_reserve and by the graph-walker thread at
// dispatch; same-device concurrent inference is unsupported (canonical contract §5), so no dispatch on a device runs
// while that device's slots are re-sized. Whether a device is planned is not kept here: it is asked of `plans`, the
// one record of the committed plans, so no free that releases slots can leave a committed plan out of force.
class host_staging_slots {
  public:
    static constexpr int max_devices = 48;  // GGML_SYCL_MAX_DEVICES

    explicit host_staging_slots(const host_holder_plans & plans) : plans_(plans) {}

    // Re-sizes `device`'s slots to the committed state: a slot below its planned capacity grows once, one above it is
    // re-allocated down, a slot planned at 0 is released. False if an allocation failed (the slot is then left at
    // capacity 0, so its dispatch refuses by name).
    bool   apply(int device, const host_holder_device_state & state, const host_staging_allocator & alloc);
    // Dispatch admission. At or below the slot's capacity: the slot. Above it, or a slot never planned on a planned
    // device: nullptr, with `[ATTN-HOST] staging request %zu B for kind %s exceeds the planned %zu B on device %d;
    // declining host dispatch` at WARN, counted as a refusal. On a device no plan has been committed for, the slot
    // grows at dispatch as before the notify path existed, with a once-per-device WARN naming that fallback.
    void * admit(int                            device,
                 int                            kind,
                 size_t                         bytes,
                 const host_staging_allocator & alloc,
                 attn_host_dispatch_counters *  counters);
    // Releases the slots of every device with no committed plan: the pre-notify fallback's slots, which grow again at
    // their next dispatch. A planned device's slots stay at its plan; they go when its last plan is dropped and the
    // device is re-applied (a backend's free of a load's temporary backend, or of a context with no plan, frees here).
    void   release_unplanned(const host_staging_allocator & alloc);
    // Releases every slot (module teardown).
    void   release_all(const host_staging_allocator & alloc);
    size_t capacity(int device, int kind) const;
    // Whether `device` has a committed plan, asked of the plans.
    bool   planned(int device) const;
    const host_staging_block & block(int device, int kind) const;

  private:
    const host_holder_plans & plans_;
    host_staging_block        slots_[max_devices][HOST_STAGING_KIND_COUNT]{};
    std::atomic<bool>         fallback_warned_[max_devices]{};
};

// ---------------------------------------------------------------------------------------------------------------
// The CPU-expert pools (§4.4 "And the pools are sized from the plan").
// ---------------------------------------------------------------------------------------------------------------

// The pools as the plan drives them.
struct host_pool_sink {
    void * user                                                         = nullptr;
    // Whether the device's pools are initialized, and at which figures.
    bool (*initialized)(void * user, int device, host_pool_dims * held) = nullptr;
    void (*init)(void * user, int device, const host_pool_dims & dims)  = nullptr;
    void (*shutdown)(void * user, int device)                           = nullptr;
};

// At a commit: when the pools are initialized and hold figures other than the device's committed high-water, shuts
// them down and re-initializes them at it, printing `[HOST-ARENA] DEVICE pools dev=%d resized D=%d->%d` at INFO.
// Before their first initialization nothing happens; moe_hybrid_init_once initializes at the high-water.
void host_pools_apply_commit(int device, const host_holder_device_state & state, const host_pool_sink & sink);

// The once-per-holder flags of host_pools_init_from_plan.
struct host_pool_init_warnings {
    std::atomic<bool> over_plan{ false };
    std::atomic<bool> unplanned{ false };
};

struct host_pool_init_decision {
    bool           init = false;  // initialize the pools at `dims`
    host_pool_dims dims{};
};

// moe_hybrid_init_once's sizing over its graph. With a committed plan for the device the pools take its figures. The
// graph only checks them, over the nodes the plan folds: a MUL_MAT_ID whose src0 `host_experts` says has an expert on
// the host for `device` (an unknown placement is not counted, as the walk does not fold it). A graph needing more
// prints `[HOST-ARENA] DEVICE holder pinned_buffer_pool dev=%d graph needs D=%d K=%lld N=%lld, exceeds its planned
// D=%d K=%lld N=%lld` once at WARN. With no committed plan, the pre-notify fallback: the pools take the graph's storage
// scan (host_pool_has_storage), with a once-per-device WARN `[HOST-ARENA] DEVICE pools dev=%d unplanned; ...` naming
// that fallback.
host_pool_init_decision host_pools_init_from_plan(int                              device,
                                                  const host_holder_device_state & state,
                                                  const ggml_cgraph *              graph,
                                                  host_plan_host_experts           host_experts,
                                                  host_pool_init_warnings &        warnings);

// The dimension fold moe_hybrid_init_once and the plan walk share. host_pool_dims_add folds one MUL_MAT_ID node in:
// D = max top-K x tokens (its ids' ne[0] x ne[1]), K = max src0 ne[0], N = max src0 ne[1], and one host-expert node.
// host_pool_scan folds every MUL_MAT_ID node whose src0 passes `eligible` (null: every node with a src0).
void           host_pool_dims_add(host_pool_dims & dims, const ggml_tensor * node);
host_pool_dims host_pool_scan(const ggml_cgraph * graph,
                              bool (*eligible)(void * user, const ggml_tensor * src0),
                              void * user);

// The storage filter of moe_hybrid_init_once's own scan: a MUL_MAT_ID whose src0 has storage, device-placed experts
// included. That scan registers the experts and sizes n_embd and the secondary staging buffer. It is not the node set
// the plan folds, so the over-plan check never reads it.
bool host_pool_has_storage(void * user, const ggml_tensor * src0);

// One device's commit, applied: registers the DEVICE holders' demand on the device's ledger (the staging slots' sum
// as attn_host_staging, only with the host dispatch on, and PinnedBufferPool's bytes as pinned_buffer_pool), ensures
// SCRATCH capacity for it, then re-sizes the staging slots and the pools to the committed figures. Runs in
// sched_reserve after its synchronize, and at a backend context's drop. False when the ledger could not ensure
// capacity or a slot could not be allocated.
bool host_holders_apply_device(int                            device,
                               const host_holder_plans &      plans,
                               host_staging_slots &           slots,
                               const host_staging_allocator & alloc,
                               host_scratch_ledger *          ledger,
                               const host_scratch_zone_ops &  ops,
                               const host_pool_sink &         pools,
                               bool                           host_dispatch_on);

// ---------------------------------------------------------------------------------------------------------------
// The dispatch counter and the fail-closed helper (§4.4 "The dispatch counter", "The six sites fail the graph by
// throwing").
// ---------------------------------------------------------------------------------------------------------------

class attn_host_dispatch_counters {
  public:
    static constexpr int max_devices = 48;  // GGML_SYCL_MAX_DEVICES

    // One interception: the entry into the host dispatch of `op`, counted before any staging is asked for. The first
    // per device prints `[ATTN-HOST] first host dispatch dev=%d op=%s` at WARN.
    void   intercept(int device, ggml_op op);
    void   refusal(int device);
    // `[ATTN-HOST] host dispatches dev=%d fa=%zu set_rows=%zu refusals=%zu` at WARN.
    void   print(int device) const;
    size_t fa(int device) const;
    size_t set_rows(int device) const;
    size_t refusals(int device) const;

  private:
    std::atomic<size_t> fa_[max_devices]{};
    std::atomic<size_t> set_rows_[max_devices]{};
    std::atomic<size_t> refusals_[max_devices]{};
    std::atomic<bool>   first_[max_devices]{};
};

// The fail-closed helper of the segmented record and replay paths: returns when `ok`; otherwise prints the device's
// counter line, then the ordinary loop's `[SYCL] graph compute unsupported node ...` line naming `node` and `role`,
// and throws ggml_sycl_fallback_error. It never asserts, aborts or returns on a refusal.
void host_dispatch_fail_closed(bool                                ok,
                               const attn_host_dispatch_counters * counters,
                               int                                 device,
                               const ggml_tensor *                 node,
                               const char *                        role);

}  // namespace ggml_sycl
