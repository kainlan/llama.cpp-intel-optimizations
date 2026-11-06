//
// Host zone routing, host-allocation lifetimes, the host SCRATCH demand ledger, and the host holder plan. See
// host-zone-plan.hpp and docs/design/sycl-mmap-host-tier.md §4.4.
//
// The HOST_ZONE_PLAN_MUTANT_* macros below are the positive controls of the four host-only tests: each plants one
// defect the design names, a test target compiles this TU with exactly one of them, and ctest scores the design's
// failure text (ggml/src/ggml-sycl/CMakeLists.txt). No production build defines any of them.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "host-zone-plan.hpp"

#include "fallback-error.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace ggml_sycl {

namespace {

constexpr double k_mib = 1024.0 * 1024.0;

double to_mib(size_t bytes) {
    return (double) bytes / k_mib;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Routing.
// ---------------------------------------------------------------------------------------------------------------

host_zone_id select_zone(alloc_role       role,
                         runtime_category category,
                         alloc_lifetime   lifetime,
                         bool             persistent_owner,
                         bool             kv_spill_to_host) {
    // persistent_owner is a lifetime claim, not a routing preference: WEIGHT is the one host zone no graph boundary
    // checks, so a persistent request that would land in KV or SCRATCH is a defect.
    GGML_ASSERT(!persistent_owner ||
                (!kv_spill_to_host && role != alloc_role::KV && role != alloc_role::EXPERT_STAGING));
    if (kv_spill_to_host || role == alloc_role::KV) {
        return host_zone_id::KV;
    }
    if (role == alloc_role::WEIGHT) {
        return host_zone_id::WEIGHT;
    }
    if (role == alloc_role::EXPERT_STAGING) {
        return host_zone_id::SCRATCH;
    }
    // persistent_owner stays here: the cache's CPU-job input marker is a CONTROL request that lives as long as the
    // cache, and only WEIGHT keeps it out of the STAGING graph-boundary check that catches a leaked scoped CONTROL.
    if (lifetime == alloc_lifetime::UNOWNED || persistent_owner) {
        return host_zone_id::WEIGHT;
    }
    if (category == runtime_category::HOST_COMPUTE || category == runtime_category::EXPERT_CACHE) {
        return host_zone_id::SCRATCH;
    }
    return host_zone_id::STAGING;
}

// ---------------------------------------------------------------------------------------------------------------
// Lifetimes.
// ---------------------------------------------------------------------------------------------------------------

const char * alloc_lifetime_name(alloc_lifetime lifetime) {
    switch (lifetime) {
        case alloc_lifetime::GRAPH:
            return "GRAPH";
        case alloc_lifetime::CONTEXT:
            return "CONTEXT";
        case alloc_lifetime::MODEL:
            return "MODEL";
        case alloc_lifetime::DEVICE:
            return "DEVICE";
        case alloc_lifetime::UNOWNED:
            return "UNOWNED";
    }
    return "?";
}

const char * alloc_owner_kind_name(alloc_owner_kind kind) {
    switch (kind) {
        case alloc_owner_kind::NONE:
            return "none";
        case alloc_owner_kind::BUFFER:
            return "buffer";
        case alloc_owner_kind::BACKEND:
            return "backend";
        case alloc_owner_kind::MODEL:
            return "model";
        case alloc_owner_kind::DEVICE:
            return "device";
        case alloc_owner_kind::EXECUTION_CONTEXT:
            return "execution_context";
    }
    return "?";
}

uint64_t host_zone_owner_next_id() {
    // Process-wide and monotonic: an owner serial is never reused, so a check after an owner is gone cannot match a
    // later owner that happens to sit at the same address.
    static std::atomic<uint64_t> next{ 1 };
    return next.fetch_add(1, std::memory_order_relaxed);
}

bool host_zone_lifetime_counts_at_graph_boundary(alloc_lifetime lifetime) {
    return lifetime == alloc_lifetime::GRAPH;
}

size_t host_zone_graph_boundary_live(const host_zone_record_view * records, size_t n, host_zone_id zone) {
    size_t live = 0;
    for (size_t i = 0; i < n; ++i) {
        if (records[i].zone == zone && host_zone_lifetime_counts_at_graph_boundary(records[i].lifetime)) {
            ++live;
        }
    }
    return live;
}

bool host_zone_record_owned_by(const host_zone_record_view & record, alloc_lifetime lifetime, alloc_owner_ref owner) {
#if defined(HOST_ZONE_PLAN_MUTANT_OWNER_BY_LIFETIME)
    // Positive control: the owner check keyed by lifetime alone fires on another context's live record.
    (void) owner;
    return record.lifetime == lifetime;
#else
    return record.lifetime == lifetime && record.owner.kind == owner.kind && record.owner.id == owner.id;
#endif
}

size_t host_zone_owner_boundary_check(const host_zone_record_view * records,
                                      size_t                        n,
                                      alloc_lifetime                lifetime,
                                      alloc_owner_ref               owner,
                                      bool                          strict) {
    size_t leaked = 0;
    for (size_t i = 0; i < n; ++i) {
        const host_zone_record_view & r = records[i];
        if (!host_zone_record_owned_by(r, lifetime, owner)) {
            continue;
        }
        ++leaked;
        GGML_LOG_WARN("[HOST-ZONE] leaked %s record: %s %zu owner=%s/%llu\n", alloc_lifetime_name(r.lifetime),
                      r.cohort && r.cohort[0] ? r.cohort : "(no cohort)", r.bytes, alloc_owner_kind_name(r.owner.kind),
                      (unsigned long long) r.owner.id);
    }
    if (leaked > 0 && strict) {
        GGML_ABORT("[HOST-ZONE] %zu %s record(s) live at their owner's boundary under GGML_SYCL_STRICT_LEASES=1",
                   leaked, alloc_lifetime_name(lifetime));
    }
    return leaked;
}

bool host_zone_device_check_due(int live_backends_on_device_after_free) {
#if defined(HOST_ZONE_PLAN_MUTANT_DEVICE_CHECK_EVERY_FREE)
    // Positive control: the DEVICE check taken at every backend free fires while a second backend is live.
    (void) live_backends_on_device_after_free;
    return true;
#else
    return live_backends_on_device_after_free == 0;
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// The host SCRATCH demand ledger.
// ---------------------------------------------------------------------------------------------------------------

size_t host_scratch_buffers_at_cap(size_t bytes, size_t chunk_cap) {
    if (bytes == 0) {
        return 0;
    }
    if (chunk_cap == 0) {
        return 1;
    }
    return (bytes + chunk_cap - 1) / chunk_cap;
}

namespace {

// Read with no ledger lock held: the capacity is the zone's, behind a callback.
size_t scratch_zone_capacity(const host_scratch_zone_ops & ops) {
    return ops.capacity ? ops.capacity(ops.user) : 0;
}

}  // namespace

host_scratch_ensure_result host_scratch_ledger::ensure_locked_then_grow(std::unique_lock<std::mutex> & lock,
                                                                        const host_scratch_zone_ops &  ops,
                                                                        size_t capacity_before) {
    host_scratch_ensure_result r;
    r.capacity_before         = capacity_before;
    size_t n_buffers          = 0;
    size_t reserved_increment = 0;
    for (const auto & kv : contexts_) {
        r.total += kv.second.bytes;
        n_buffers += kv.second.n_buffers;
    }
    for (const auto & kv : holders_) {
        r.total += kv.second.bytes;
        n_buffers += kv.second.n_buffers;
    }
    r.target = ops.plan_scratch_bytes + r.total;
#if defined(HOST_ZONE_PLAN_MUTANT_LEDGER_ADDITIVE_GROW)
    // Positive control: the additive grow_scratch_zone(additional_bytes) this ledger replaced, which grew by the whole
    // demand on every call, whatever the zone already held (the ratchet).
    r.request = r.total;
#else
    // Reserve, then grow. What the zone holds, or what a grow already in flight will bring it to, may cover the target;
    // otherwise the deficit is reserved here, before the mutex is released, so a notify that arrives while this grow
    // runs finds it covered and does not grow for it again.
    const size_t covered = std::max(r.capacity_before, reserved_target_);
    if (covered < r.target) {
        reserved_increment = r.target - covered;
#    if defined(HOST_ZONE_PLAN_MUTANT_LEDGER_NO_RESERVE)
        // Positive control: the deficit is not reserved before the unlock, so a notify that arrives while this grow
        // runs sees the same deficit and grows for it again.
#    else
        reserved_target_ = r.target;
#    endif
        // One grow_zone call for the deficit plus one alignment of slack per buffer the demands split into, the same
        // slack rule as the miss path. The pool commits one chunk of max(2048 MB, align_up(request)).
        r.request = reserved_increment + n_buffers * ops.alignment;
    }
#endif
    // The grow runs with the ledger's mutex released: it takes the pool's own locks and may commit a chunk, and no
    // lock of this ledger is held across a grow (canonical contract §12.5).
    lock.unlock();
    if (r.request > 0) {
        r.ok = ops.grow ? ops.grow(ops.user, r.request) : false;
#if !defined(HOST_ZONE_PLAN_MUTANT_LEDGER_NO_ROLLBACK)
        if (!r.ok && reserved_increment > 0) {
            // The zone did not grow, so the reservation goes back and the next notify for this deficit grows again.
            lock.lock();
            reserved_target_ = reserved_target_ > reserved_increment ? reserved_target_ - reserved_increment : 0;
            lock.unlock();
        }
#endif
    }
    const size_t capacity_after = scratch_zone_capacity(ops);
    r.grown                     = capacity_after > r.capacity_before ? capacity_after - r.capacity_before : 0;
    return r;
}

host_scratch_ensure_result host_scratch_ledger::notify_context(uint64_t                      context_key,
                                                               size_t                        compute_bytes,
                                                               size_t                        output_bytes,
                                                               const host_scratch_zone_ops & ops) {
    const size_t                 capacity_before = scratch_zone_capacity(ops);
    std::unique_lock<std::mutex> lock(mutex_);
    demand &                     d        = contexts_[context_key];
    // The commit's re-notify replaces the measured compute figure with the allocated one and prints both.
    const bool                   renotify = d.committed_mark;
    const size_t                 measured = d.compute;
    d.committed_mark                      = false;
    d.compute                             = compute_bytes;
#if defined(HOST_ZONE_PLAN_MUTANT_LEDGER_ACCUMULATE)
    // Positive control: accumulate instead of replace, so a context's second notify grows again for the same bytes.
    d.bytes += compute_bytes + output_bytes;
    d.n_buffers += host_scratch_buffers_at_cap(compute_bytes, ops.chunk_cap) +
                   host_scratch_buffers_at_cap(output_bytes, ops.chunk_cap);
#else
    d.bytes     = compute_bytes + output_bytes;
    d.n_buffers = host_scratch_buffers_at_cap(compute_bytes, ops.chunk_cap) +
                  host_scratch_buffers_at_cap(output_bytes, ops.chunk_cap);
#endif
    const host_scratch_ensure_result r        = ensure_locked_then_grow(lock, ops, capacity_before);
    char                             both[64] = "";
    if (renotify) {
        snprintf(both, sizeof(both), " measured=%.1f allocated=%.1f", to_mib(measured), to_mib(compute_bytes));
    }
    GGML_LOG_INFO("[HOST-ARENA] SCRATCH demand ctx=%llu compute=%.1f MB output=%.1f MB total=%.1f MB grown=%.1f MB%s\n",
                  (unsigned long long) context_key, to_mib(compute_bytes), to_mib(output_bytes), to_mib(r.total),
                  to_mib(r.grown), both);
    return r;
}

host_scratch_ensure_result host_scratch_ledger::holder_demand(const char *                  holder,
                                                              int                           device,
                                                              size_t                        bytes,
                                                              const host_scratch_zone_ops & ops) {
    const size_t                 capacity_before = scratch_zone_capacity(ops);
    std::unique_lock<std::mutex> lock(mutex_);
    demand &                     d     = holders_[std::string(holder ? holder : "?") + "@" + std::to_string(device)];
    d.bytes                            = bytes;
    d.n_buffers                        = host_scratch_buffers_at_cap(bytes, ops.chunk_cap);
    const host_scratch_ensure_result r = ensure_locked_then_grow(lock, ops, capacity_before);
    GGML_LOG_INFO("[HOST-ARENA] SCRATCH demand holder=%s dev=%d bytes=%.1f MB total=%.1f MB grown=%.1f MB\n",
                  holder ? holder : "?", device, to_mib(bytes), to_mib(r.total), to_mib(r.grown));
    return r;
}

void host_scratch_ledger::drop_context(uint64_t context_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_.erase(context_key);
}

void host_scratch_ledger::mark_committed(uint64_t context_key) {
#if defined(HOST_ZONE_PLAN_MUTANT_LEDGER_MARK_IGNORED)
    // Positive control: the mark is dropped, so the commit's re-notify prints the plain demand line.
    (void) context_key;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = contexts_.find(context_key);
    if (it != contexts_.end()) {
        it->second.committed_mark = true;
    }
#endif
}

void host_scratch_ledger::release_reserved(size_t bytes) {
#if defined(HOST_ZONE_PLAN_MUTANT_LEDGER_WITHDRAW_KEPT)
    // Positive control: the reservation keeps the withdrawn chunk, so the same target finds it covered and never grows.
    bytes = 0;
#endif
    std::lock_guard<std::mutex> lock(mutex_);
    reserved_target_ = reserved_target_ > bytes ? reserved_target_ - bytes : 0;
}

size_t host_scratch_ledger::total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      total = 0;
    for (const auto & kv : contexts_) {
        total += kv.second.bytes;
    }
    for (const auto & kv : holders_) {
        total += kv.second.bytes;
    }
    return total;
}

size_t host_scratch_ledger::context_demand(uint64_t context_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = contexts_.find(context_key);
    return it == contexts_.end() ? 0 : it->second.bytes;
}

// The ledger the notify records a context's demand on and the commit marks it on: that of the device the host buft
// allocates from, the zone the host buffers land in. False when that device has no unified cache.
static bool host_scratch_resolve_ledger(const host_scratch_notify_env &   env,
                                        ggml_backend_t                    backend,
                                        const host_scratch_context_info & info,
                                        int *                             device,
                                        host_scratch_ledger **            ledger,
                                        host_scratch_zone_ops *           ops) {
#if defined(HOST_ZONE_PLAN_MUTANT_NOTIFY_DEVICE0)
    // Positive control: the device-0 resolve.
    (void) backend;
    (void) info;
    *device = 0;
#elif defined(HOST_ZONE_PLAN_MUTANT_NOTIFY_CTX_DEVICE)
    // Positive control: the notifying backend's own device in place of the host buft's.
    (void) backend;
    *device = info.device;
#else
    (void) info;
    *device = env.host_buft_device(backend);
#endif
    return *device >= 0 && env.ledger_of && env.ledger_of(*device, ledger, ops) && *ledger;
}

bool host_scratch_notify(const host_scratch_notify_env & env,
                         ggml_backend_t                  backend,
                         size_t                          host_compute_bytes,
                         size_t                          host_output_bytes) {
    // The backend is checked before anything reads its context: a non-SYCL backend's context is not a
    // ggml_backend_sycl_context.
    if (!backend || !env.is_sycl || !env.is_sycl(backend)) {
        GGML_LOG_WARN("[HOST-ARENA] SCRATCH notify refused: the backend is not a SYCL backend; nothing recorded\n");
        return false;
    }
    const host_scratch_context_info info   = env.context_of(backend);
    int                             device = -1;
    host_scratch_ledger *           ledger = nullptr;
    host_scratch_zone_ops           ops{};
    if (!host_scratch_resolve_ledger(env, backend, info, &device, &ledger, &ops)) {
        GGML_LOG_WARN("[HOST-ARENA] SCRATCH notify for ctx=%llu found no unified cache on device %d; nothing grown\n",
                      (unsigned long long) info.key, device);
        return false;
    }
    return ledger->notify_context(info.key, host_compute_bytes, host_output_bytes, ops).ok;
}

bool host_scratch_mark_committed(const host_scratch_notify_env & env, ggml_backend_t backend) {
    if (!backend || !env.is_sycl || !env.is_sycl(backend)) {
        return false;
    }
    const host_scratch_context_info info   = env.context_of(backend);
    int                             device = -1;
    host_scratch_ledger *           ledger = nullptr;
    host_scratch_zone_ops           ops{};
    if (!host_scratch_resolve_ledger(env, backend, info, &device, &ledger, &ops)) {
        return false;
    }
    ledger->mark_committed(info.key);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// The plan walk.
// ---------------------------------------------------------------------------------------------------------------

ggml_backend_buffer_t host_tensor_buffer(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    return (tensor->view_src && tensor->view_src->buffer) ? tensor->view_src->buffer : tensor->buffer;
}

bool host_tensor_in_buft_named(const ggml_tensor * tensor, const char * (*get_name)(ggml_backend_buffer_type_t buft)) {
    const ggml_backend_buffer_t buf = host_tensor_buffer(tensor);
    return get_name && buf && buf->buft && buf->buft->iface.get_name == get_name;
}

namespace {

host_plan_attribution attribute_with(host_plan_tensor_backend lookup,
                                     ggml_backend_sched_t     sched,
                                     ggml_cgraph *            graph,
                                     host_plan_backend_pred   is_target,
                                     host_plan_node_pred      node_pred,
                                     host_plan_visit          visit,
                                     void *                   user,
                                     bool                     warn_zero) {
    host_plan_attribution a;
    if (!sched || !graph || !is_target || !node_pred || !lookup) {
        return a;
    }
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (!node || !node_pred(user, node)) {
            continue;
        }
        ++a.matched;
        ggml_backend_t backend = lookup(sched, node);
        if (!backend || !is_target(backend)) {
            continue;
        }
        ++a.attributed;
        if (visit) {
            visit(user, node, backend);
        }
    }
    if (warn_zero && a.attributed == 0) {
        GGML_LOG_WARN("[ATTN-HOST] plan attributed 0 host-KV nodes to a SYCL backend (%d matched the predicate)\n",
                      a.matched);
    }
    return a;
}

}  // namespace

host_plan_attribution host_plan_attribute(ggml_backend_sched_t   sched,
                                          ggml_cgraph *          graph,
                                          host_plan_backend_pred is_target,
                                          host_plan_node_pred    node_pred,
                                          host_plan_visit        visit,
                                          void *                 user,
                                          bool                   warn_zero) {
    return attribute_with(ggml_backend_sched_get_tensor_backend, sched, graph, is_target, node_pred, visit, user,
                          warn_zero);
}

// ---------------------------------------------------------------------------------------------------------------
// The host holder plan.
// ---------------------------------------------------------------------------------------------------------------

const char * host_staging_kind_name(int kind) {
    switch (kind) {
        case HOST_STAGING_Q:
            return "attn_host_q";
        case HOST_STAGING_DST:
            return "attn_host_dst";
        case HOST_STAGING_MASK:
            return "attn_host_mask";
        case HOST_STAGING_VAL:
            return "attn_host_setrows_val";
        case HOST_STAGING_IDS:
            return "attn_host_setrows_ids";
        default:
            return "?";
    }
}

size_t host_staging_slot_bytes(size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    const size_t mib = (size_t) 1 << 20;
    return (bytes + 64 + mib - 1) & ~(mib - 1);
}

size_t host_staging_slot_capacity(size_t bytes) {
    const size_t slot = host_staging_slot_bytes(bytes);
    return slot == 0 ? 0 : slot - 64;
}

size_t host_pool_pinned_bytes(const host_pool_dims & dims) {
    if (dims.D <= 0 || dims.K <= 0 || dims.N <= 0) {
        return 0;
    }
    const size_t d = (size_t) dims.D;
    return d * ((size_t) dims.K + (size_t) dims.N) * sizeof(float) + d * sizeof(int32_t);
}

void host_pool_dims_add(host_pool_dims & dims, const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) {
        return;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * ids  = node->src[2];
    if (ids && ids->ne[0] > 0) {
        const int64_t dispatch = ids->ne[0] * std::max<int64_t>(ids->ne[1], 1);
        dims.D                 = std::max(dims.D, dispatch);
    }
    dims.K = std::max(dims.K, src0->ne[0]);
    dims.N = std::max(dims.N, src0->ne[1]);
    ++dims.host_expert_nodes;
}

host_pool_dims host_pool_scan(const ggml_cgraph * graph,
                              bool (*eligible)(void * user, const ggml_tensor * src0),
                              void * user) {
    host_pool_dims dims;
    if (!graph) {
        return dims;
    }
    ggml_cgraph * g       = const_cast<ggml_cgraph *>(graph);
    const int     n_nodes = ggml_graph_n_nodes(g);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(g, i);
        if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) {
            continue;
        }
        if (eligible && !eligible(user, node->src[0])) {
            continue;
        }
        host_pool_dims_add(dims, node);
    }
    return dims;
}

bool host_pool_has_storage(void * user, const ggml_tensor * src0) {
    (void) user;
    return src0 && src0->data != nullptr;  // raw-ok: storage presence only (ggml_sycl_host_data), never read
}

namespace {

void plan_merge_max(host_holder_plan & into, const host_holder_plan & from) {
    if (into.device < 0) {
        into.device = from.device;
    }
    for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
        into.staging[k] = std::max(into.staging[k], from.staging[k]);
    }
    into.pools.D                 = std::max(into.pools.D, from.pools.D);
    into.pools.K                 = std::max(into.pools.K, from.pools.K);
    into.pools.N                 = std::max(into.pools.N, from.pools.N);
    into.pools.host_expert_nodes = std::max(into.pools.host_expert_nodes, from.pools.host_expert_nodes);
}

// One walk's figures, gathered with no lock held, per backend context.
struct walk_scratch {
    const host_holder_walk_env *         env = nullptr;
    std::map<uint64_t, host_holder_plan> plans;
};

host_holder_plan & walk_plan_for(walk_scratch & w, ggml_backend_t backend, int device) {
    host_holder_plan & p = w.plans[w.env->backend_key(backend)];
    if (p.device < 0) {
        p.device = device;
    }
    return p;
}

bool walk_in_kv_host(const host_holder_walk_env & env, const ggml_tensor * t) {
#if defined(HOST_ZONE_PLAN_MUTANT_PREDICATE_IS_HOST)
    // Positive control: ggml_backend_buffer_is_host as the predicate counts CPU and SYCL_Host K and misses the view,
    // whose own buffer is NULL.
    (void) env;
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
#else
    return t && env.in_kv_host_buft(t);
#endif
}

// The dispatch's own predicate (the FLASH_ATTN_EXT and SET_ROWS interceptions): attention whose K or V is in a
// SYCL_KV_Host buffer, and a SET_ROWS into one.
bool walk_is_host_kv_node(void * user, const ggml_tensor * node) {
    const walk_scratch & w = *static_cast<const walk_scratch *>(user);
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        return walk_in_kv_host(*w.env, node->src[1]) || walk_in_kv_host(*w.env, node->src[2]);
    }
    if (node->op == GGML_OP_SET_ROWS) {
        return walk_in_kv_host(*w.env, node);
    }
    return false;
}

// The pools' one eligibility rule, the walk's and the over-plan check's: placement's executor fact says `device` runs
// at least one expert of src0 on the host. *known false means the fact is absent (*cause says why), and such a node is
// not eligible.
bool pool_host_placed(host_plan_host_experts host_experts,
                      const ggml_tensor *    src0,
                      int                    device,
                      bool *                 known,
                      const char **          cause) {
#if defined(HOST_ZONE_PLAN_MUTANT_POOLS_GATE_IS_HOST)
    // Positive control: the buffer gate. SYCL_Mapped is is_host, so every expert of a mapped load would plan the
    // pools, device-placed ones included.
    (void) host_experts;
    (void) device;
    (void) cause;
    *known = true;
    return src0->buffer && ggml_backend_buffer_is_host(src0->buffer);
#else
    *known = false;
    return host_experts && host_experts(src0, device, known, cause);
#endif
}

bool walk_is_mul_mat_id(void * user, const ggml_tensor * node) {
    (void) user;
    return node->op == GGML_OP_MUL_MAT_ID && node->src[0] != nullptr;
}

int walk_staging_device(const walk_scratch & w, ggml_backend_t backend) {
#if defined(HOST_ZONE_PLAN_MUTANT_HELPER_DEVICE)
    // Positive control: the notify helper's device (the SYCL_Host buft's device 0) for every slot, which leaves the
    // dispatching device 1's slots unplanned.
    (void) w;
    (void) backend;
    return 0;
#else
    // The device that dispatches owns the slot: the dispatch indexes its slots by ctx.device.
    return w.env->backend_device(backend);
#endif
}

void walk_visit_host_kv(void * user, ggml_tensor * node, ggml_backend_t backend) {
    walk_scratch &     w = *static_cast<walk_scratch *>(user);
    host_holder_plan & p = walk_plan_for(w, backend, walk_staging_device(w, backend));
#if !defined(HOST_ZONE_PLAN_MUTANT_PLAN_IGNORES_SWITCH)
    // With the host dispatch off nothing stages: the scheduler's CPU split runs demoted-layer attention, in the CPU
    // backend's compute buffer the notify covers.
    if (!w.env->host_dispatch_on) {
        return;
    }
#endif
    // The operands the dispatch stages, sized by the calls it makes.
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        p.staging[HOST_STAGING_Q]   = std::max(p.staging[HOST_STAGING_Q], ggml_nbytes(node->src[0]));
        p.staging[HOST_STAGING_DST] = std::max(p.staging[HOST_STAGING_DST], ggml_nbytes(node));
        if (node->src[3]) {
            p.staging[HOST_STAGING_MASK] = std::max(p.staging[HOST_STAGING_MASK], ggml_nbytes(node->src[3]));
        }
    } else if (node->op == GGML_OP_SET_ROWS) {
        p.staging[HOST_STAGING_VAL] = std::max(p.staging[HOST_STAGING_VAL], ggml_nbytes(node->src[0]));
        if (node->src[1]) {
            p.staging[HOST_STAGING_IDS] = std::max(p.staging[HOST_STAGING_IDS], ggml_nbytes(node->src[1]));
        }
    }
}

void walk_visit_mul_mat_id(void * user, ggml_tensor * node, ggml_backend_t backend) {
    walk_scratch &      w      = *static_cast<walk_scratch *>(user);
    const int           device = w.env->backend_device(backend);
    host_holder_plan &  p      = walk_plan_for(w, backend, device);
    const ggml_tensor * src0   = node->src[0];
    // Placement's executor fact, the one the dispatch's CPU-island gate routes on, never the buffer.
    bool                known  = false;
    const char *        cause  = nullptr;
    const bool          host   = pool_host_placed(w.env->host_experts, src0, device, &known, &cause);
    if (!known) {
        GGML_LOG_WARN("[HOST-ARENA] DEVICE pools dev=%d %s: placement unknown (%s)\n", device, src0->name,
                      cause ? cause : "no placement plan");
        return;
    }
    if (!host) {
        return;
    }
    host_pool_dims_add(p.pools, node);
}

}  // namespace

void host_holder_plans::begin(const uint64_t * keys, size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
        entry & e       = entries_[keys[i]];
        e.pending       = host_holder_plan{};
        e.pending_valid = true;
    }
}

host_plan_attribution host_holder_plans::walk(ggml_backend_sched_t         sched,
                                              ggml_cgraph *                graph,
                                              const host_holder_walk_env & env) {
    walk_scratch w;
    w.env = &env;
#if defined(HOST_ZONE_PLAN_MUTANT_PLAN_IGNORES_SWITCH)
    const bool staging_planned = true;
#else
    const bool staging_planned = env.host_dispatch_on;
#endif
    // The host-KV attention nodes and the MUL_MAT_ID nodes, attributed by the scheduler's live split. The callbacks run
    // here with no lock of this object held: the placement fact takes the inventory's own lock.
    const host_plan_tensor_backend lookup =
        env.tensor_backend ? env.tensor_backend : ggml_backend_sched_get_tensor_backend;
    const host_plan_attribution a = attribute_with(lookup, sched, graph, env.is_target, walk_is_host_kv_node,
                                                   walk_visit_host_kv, &w, /*warn_zero=*/staging_planned);
    (void) attribute_with(lookup, sched, graph, env.is_target, walk_is_mul_mat_id, walk_visit_mul_mat_id, &w,
                          /*warn_zero=*/false);
    if (!staging_planned && !staging_off_warned_.exchange(true)) {
        GGML_LOG_WARN("[ATTN-HOST] staging not planned (host dispatch off)\n");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & kv : w.plans) {
        entry & e = entries_[kv.first];
#if defined(HOST_ZONE_PLAN_MUTANT_WALK_TG_REPLACES_PP)
        // Positive control: a later walk of the same reserve replaces what an earlier one recorded, so the tg walk's
        // 1-token figures overwrite the pp walk's.
        e.pending = kv.second;
#else
        if (!e.pending_valid) {
            e.pending = host_holder_plan{};
        }
        plan_merge_max(e.pending, kv.second);
#endif
        e.pending_valid = true;
    }
    return a;
}

bool host_holder_plans::commit(uint64_t key, int * device_out, int * previous_out) {
    host_holder_device_state state;
    int                      device   = -1;
    int                      previous = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = entries_.find(key);
        if (it == entries_.end() || !it->second.pending_valid) {
            if (device_out) {
                *device_out = -1;
            }
            if (previous_out) {
                *previous_out = -1;
            }
            return false;
        }
        entry & e = it->second;
        if (e.committed_valid) {
            previous = e.committed.device;
        }
#if defined(HOST_ZONE_PLAN_MUTANT_CAPACITY_HIGH_WATER)
        // Positive control: a high-water over every plan ever committed, so a smaller re-commit keeps the larger.
        if (e.committed_valid && e.pending.device >= 0) {
            plan_merge_max(e.committed, e.pending);
        } else {
            e.committed = e.pending;
        }
#else
        e.committed = e.pending;
#endif
        e.committed_valid = e.committed.device >= 0;
        e.pending         = host_holder_plan{};
        e.pending_valid   = false;
        device            = e.committed_valid ? e.committed.device : -1;
        // The merge happens here, once per commit, for both devices whose figures changed.
        merge_device_locked(device);
        if (previous != device) {
            merge_device_locked(previous);
        }
        if (device >= 0 && device < host_zone_max_devices) {
            state = merged_[device];
        }
    }
    if (device_out) {
        *device_out = device;
    }
    if (previous_out) {
        *previous_out = previous;
    }
    if (device >= 0) {
        GGML_LOG_INFO("[HOST-ARENA] DEVICE pools dev=%d committed D=%lld K=%lld N=%lld host_expert_nodes=%d\n", device,
                      (long long) state.pools.D, (long long) state.pools.K, (long long) state.pools.N,
                      state.pools.host_expert_nodes);
        size_t total = 0;
        for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
            total += state.slot_bytes[k];
        }
        if (total > 0) {
            GGML_LOG_INFO("[ATTN-HOST] planned staging dev=%d q=%zu dst=%zu mask=%zu val=%zu ids=%zu total=%.1f MB\n",
                          device, state.slot_capacity[HOST_STAGING_Q], state.slot_capacity[HOST_STAGING_DST],
                          state.slot_capacity[HOST_STAGING_MASK], state.slot_capacity[HOST_STAGING_VAL],
                          state.slot_capacity[HOST_STAGING_IDS], to_mib(total));
        }
    }
    return true;
}

void host_holder_plans::drop(uint64_t key, int * device_out) {
    std::lock_guard<std::mutex> lock(mutex_);
    int                         device = -1;
    auto                        it     = entries_.find(key);
    if (it != entries_.end()) {
        if (it->second.committed_valid) {
            device = it->second.committed.device;
        }
#if defined(HOST_ZONE_PLAN_MUTANT_CAPACITY_HIGH_WATER)
        // Positive control: a high-water over every plan ever committed, so a dropped context's figures still size
        // its device.
        it->second.pending_valid = false;
#else
        entries_.erase(it);
#endif
        merge_device_locked(device);
    }
    if (device_out) {
        *device_out = device;
    }
}

void host_holder_plans::merge_device_locked(int device) {
    if (device < 0 || device >= host_zone_max_devices) {
        return;
    }
    host_holder_device_state s;
    host_holder_plan         high{};
    for (const auto & kv : entries_) {
        const entry & e = kv.second;
        if (!e.committed_valid || e.committed.device != device) {
            continue;
        }
        s.planned = true;
        plan_merge_max(high, e.committed);
    }
    for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
        s.slot_capacity[k] = host_staging_slot_capacity(high.staging[k]);
        s.slot_bytes[k]    = host_staging_slot_bytes(high.staging[k]);
    }
    s.pools         = high.pools;
    merged_[device] = s;
}

host_holder_device_state host_holder_plans::device_state(int device) const {
    if (device < 0 || device >= host_zone_max_devices) {
        return host_holder_device_state{};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return merged_[device];
}

bool host_holder_plans::has_pending(uint64_t key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = entries_.find(key);
    return it != entries_.end() && it->second.pending_valid;
}

bool host_holder_plans::has_committed(uint64_t key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = entries_.find(key);
    return it != entries_.end() && it->second.committed_valid;
}

// ---------------------------------------------------------------------------------------------------------------
// Attention host staging slots.
// ---------------------------------------------------------------------------------------------------------------

namespace {

bool slot_replace(host_staging_block & slot, int device, int kind, size_t bytes, const host_staging_allocator & alloc) {
    host_staging_block fresh;
    if (bytes > 0) {
        if (!alloc.alloc || !alloc.alloc(alloc.user, device, kind, bytes, &fresh) || !fresh.ptr) {
            return false;
        }
        fresh.bytes = bytes;
        fresh.cap   = bytes - 64;
    }
    if (slot.ptr && alloc.release) {
        alloc.release(alloc.user, &slot);
    }
    slot = fresh;
    return true;
}

bool slot_device_ok(int device) {
    return device >= 0 && device < host_zone_max_devices;
}

}  // namespace

bool host_staging_slots::apply(int                              device,
                               const host_holder_device_state & state,
                               const host_staging_allocator &   alloc) {
    if (!slot_device_ok(device)) {
        return false;
    }
    bool ok = true;
    for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
        host_staging_block & slot = slots_[device][k];
        const size_t         want = state.slot_bytes[k];
        if (slot.bytes == want) {
            continue;
        }
        if (!slot_replace(slot, device, k, want, alloc)) {
            // Left at capacity 0: its dispatch refuses by name rather than reading a slot of the wrong size.
            if (slot.ptr && alloc.release) {
                alloc.release(alloc.user, &slot);
            }
            slot = host_staging_block{};
            ok   = false;
        }
    }
    return ok;
}

void * host_staging_slots::admit(int                            device,
                                 int                            kind,
                                 size_t                         bytes,
                                 const host_staging_allocator & alloc,
                                 attn_host_dispatch_counters *  counters) {
    if (!slot_device_ok(device) || kind < 0 || kind >= HOST_STAGING_KIND_COUNT || bytes == 0) {
        return nullptr;
    }
    host_staging_block & slot = slots_[device][kind];
    (void) alloc;  // a slot grows only in apply, under the committed plan; only the positive controls allocate here
    if (planned(device)) {
        if (bytes <= slot.cap) {
            return slot.ptr;
        }
#if defined(HOST_ZONE_PLAN_MUTANT_DISPATCH_GROW)
        // Positive control: the dispatch-time grow restored, which allocates for an over-plan request instead of
        // refusing.
        if (slot_replace(slot, device, kind, host_staging_slot_bytes(bytes), alloc)) {
            return slot.ptr;
        }
#endif
    } else {
#if defined(HOST_ZONE_PLAN_MUTANT_UNPLANNED_GROWS)
        // Positive control: the transitional fallback restored, which grows an unplanned device's slot at dispatch.
        if (bytes > slot.cap && !slot_replace(slot, device, kind, host_staging_slot_bytes(bytes), alloc)) {
            return nullptr;
        }
        return slot.ptr;
#endif
        // No host holder plan is committed for this device, so nothing sized its slots: each has capacity 0 and
        // takes the refusal below, as a planned slot's over-plan request does (§4.4 "At dispatch no slot grows"),
        // whatever a dropped plan left in it until the device is re-applied.
    }
    // The one refusal: by name and counted, quoting capacity 0 for an unplanned device's slot.
    GGML_LOG_WARN(
        "[ATTN-HOST] staging request %zu B for kind %s exceeds the planned %zu B on device %d; declining host "
        "dispatch\n",
        bytes, host_staging_kind_name(kind), planned(device) ? slot.cap : 0, device);
    if (counters) {
        counters->refusal(device);
    }
    return nullptr;
}

namespace {

void slots_release_device(host_staging_block (&slots)[HOST_STAGING_KIND_COUNT], const host_staging_allocator & alloc) {
    for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
        if (slots[k].ptr && alloc.release) {
            alloc.release(alloc.user, &slots[k]);
        }
        slots[k] = host_staging_block{};
    }
}

}  // namespace

void host_staging_slots::release_unplanned(const host_staging_allocator & alloc) {
    for (int d = 0; d < host_zone_max_devices; ++d) {
#if defined(HOST_ZONE_PLAN_MUTANT_FREE_RELEASES_PLANNED)
        // Positive control: every device's slots released, so an unrelated backend's free takes a committed plan's
        // slots with it.
#else
        if (planned(d)) {
            continue;  // released only when the device's last plan is dropped and the device re-applied
        }
#endif
        slots_release_device(slots_[d], alloc);
    }
}

void host_staging_slots::release_all(const host_staging_allocator & alloc) {
    for (int d = 0; d < host_zone_max_devices; ++d) {
        slots_release_device(slots_[d], alloc);
    }
}

size_t host_staging_slots::capacity(int device, int kind) const {
    if (!slot_device_ok(device) || kind < 0 || kind >= HOST_STAGING_KIND_COUNT) {
        return 0;
    }
    return slots_[device][kind].cap;
}

bool host_staging_slots::planned(int device) const {
    return slot_device_ok(device) && plans_.device_state(device).planned;
}

const host_staging_block & host_staging_slots::block(int device, int kind) const {
    static const host_staging_block none{};
    if (!slot_device_ok(device) || kind < 0 || kind >= HOST_STAGING_KIND_COUNT) {
        return none;
    }
    return slots_[device][kind];
}

// ---------------------------------------------------------------------------------------------------------------
// The pools.
// ---------------------------------------------------------------------------------------------------------------

void host_pools_apply_commit(int device, const host_holder_device_state & state, const host_pool_sink & sink) {
#if defined(HOST_ZONE_PLAN_MUTANT_POOLS_INIT_ONCE)
    // Positive control: init-once kept, so the pools stay at their first figures after a larger commit.
    (void) device;
    (void) state;
    (void) sink;
#else
    host_pool_dims held{};
    if (!sink.initialized || !sink.initialized(sink.user, device, &held)) {
        return;  // before the first initialization the commit only records
    }
    const host_pool_dims & want = state.pools;
    if (held.D == want.D && held.K == want.K && held.N == want.N) {
        return;
    }
    if (sink.shutdown) {
        sink.shutdown(sink.user, device);
    }
    if (want.D > 0 && want.K > 0 && want.N > 0 && sink.init) {
        sink.init(sink.user, device, want);
    }
    GGML_LOG_INFO("[HOST-ARENA] DEVICE pools dev=%d resized D=%lld->%lld\n", device, (long long) held.D,
                  (long long) want.D);
#endif
}

namespace {

struct host_placed_scan {
    host_plan_host_experts host_experts = nullptr;
    int                    device       = -1;
};

// The over-plan check's node set: the walk's eligibility, so the check reads exactly the nodes the plan folded.
bool scan_host_placed(void * user, const ggml_tensor * src0) {
    const host_placed_scan & s     = *static_cast<const host_placed_scan *>(user);
    bool                     known = false;
    const char *             cause = nullptr;
    const bool               host  = pool_host_placed(s.host_experts, src0, s.device, &known, &cause);
    return known && host;
}

}  // namespace

host_pool_init_decision host_pools_init_from_plan(int                              device,
                                                  const host_holder_device_state & state,
                                                  const ggml_cgraph *              graph,
                                                  host_plan_host_experts           host_experts,
                                                  host_pool_init_warnings &        warnings) {
    host_pool_init_decision d;
    if (!state.planned) {
#if defined(HOST_ZONE_PLAN_MUTANT_UNPLANNED_POOLS_SCAN)
        // Positive control: the transitional fallback restored, which sizes an unplanned device's pools from this
        // graph's own storage scan.
        d.dims = host_pool_scan(graph, host_pool_has_storage, nullptr);
        d.init = d.dims.D > 0 && d.dims.K > 0 && d.dims.N > 0;
        return d;
#endif
        // No host holder plan is committed for this device, so nothing sized the pools: they are not initialized
        // (§4.4 "And the pools are sized from the plan"). The line prints once per device.
        if (!warnings.unplanned.exchange(true)) {
            GGML_LOG_WARN("[HOST-ARENA] DEVICE pools dev=%d unplanned; not initialized\n", device);
        }
        return d;
    }
#if defined(HOST_ZONE_PLAN_MUTANT_POOLS_OWN_SCAN)
    // Positive control: the init sized from its own graph (rev 3g), which prints nothing.
    (void) host_experts;
    d.dims = host_pool_scan(graph, host_pool_has_storage, nullptr);
    d.init = d.dims.D > 0 && d.dims.K > 0 && d.dims.N > 0;
    return d;
#else
    d.dims = state.pools;
    d.init = state.pools.D > 0 && state.pools.K > 0 && state.pools.N > 0;
#    if defined(HOST_ZONE_PLAN_MUTANT_POOLS_CHECK_REMOVED)
    (void) graph;
    (void) host_experts;
#    else
#        if defined(HOST_ZONE_PLAN_MUTANT_POOLS_CHECK_STORAGE_SCAN)
    // Positive control: the check reads moe_hybrid_init_once's storage scan again, every MUL_MAT_ID whose src0 has
    // storage, not the nodes the plan folded, so an all-device plan warns.
    (void) host_experts;
    const host_pool_dims graph_needs = host_pool_scan(graph, host_pool_has_storage, nullptr);
#        else
    host_placed_scan s;
    s.host_experts                   = host_experts;
    s.device                         = device;
    const host_pool_dims graph_needs = host_pool_scan(graph, scan_host_placed, &s);
#        endif
    const bool exceeds =
        graph_needs.D > state.pools.D || graph_needs.K > state.pools.K || graph_needs.N > state.pools.N;
    if (exceeds && !warnings.over_plan.exchange(true)) {
        GGML_LOG_WARN(
            "[HOST-ARENA] DEVICE holder pinned_buffer_pool dev=%d graph needs D=%lld K=%lld N=%lld, exceeds its "
            "planned "
            "D=%lld K=%lld N=%lld\n",
            device, (long long) graph_needs.D, (long long) graph_needs.K, (long long) graph_needs.N,
            (long long) state.pools.D, (long long) state.pools.K, (long long) state.pools.N);
    }
#    endif
    return d;
#endif
}

bool host_holders_apply_device(int                            device,
                               const host_holder_plans &      plans,
                               host_staging_slots &           slots,
                               const host_staging_allocator & alloc,
                               host_scratch_ledger *          ledger,
                               const host_scratch_zone_ops &  ops,
                               const host_pool_sink &         pools,
                               bool                           host_dispatch_on) {
    const host_holder_device_state state = plans.device_state(device);
    bool                           ok    = true;
    if (ledger) {
        size_t staging_bytes = 0;
        if (host_dispatch_on) {
            for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
                staging_bytes += state.slot_bytes[k];
            }
        }
        // The capacity is ensured before any slot or pool allocates against it. No cpu_expert_pool demand: in this
        // tree CpuExpertPool holds no memory, only workers to join (as the backend free's pinned-owner shutdown notes).
        ok = ledger->holder_demand("attn_host_staging", device, staging_bytes, ops).ok && ok;
        ok = ledger->holder_demand("pinned_buffer_pool", device, host_pool_pinned_bytes(state.pools), ops).ok && ok;
    }
    ok = slots.apply(device, state, alloc) && ok;
    host_pools_apply_commit(device, state, pools);
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// The dispatch counter and the fail-closed helper.
// ---------------------------------------------------------------------------------------------------------------

namespace {

bool counter_device_ok(int device) {
    return device >= 0 && device < host_zone_max_devices;
}

}  // namespace

void attn_host_dispatch_counters::intercept(int device, ggml_op op) {
    if (!counter_device_ok(device)) {
        return;
    }
    if (op == GGML_OP_FLASH_ATTN_EXT) {
        fa_[device].fetch_add(1, std::memory_order_relaxed);
    } else if (op == GGML_OP_SET_ROWS) {
        set_rows_[device].fetch_add(1, std::memory_order_relaxed);
    }
    if (!first_[device].exchange(true)) {
        GGML_LOG_WARN("[ATTN-HOST] first host dispatch dev=%d op=%s\n", device, ggml_op_name(op));
    }
}

void attn_host_dispatch_counters::refusal(int device) {
    if (counter_device_ok(device)) {
        refusals_[device].fetch_add(1, std::memory_order_relaxed);
    }
}

void attn_host_dispatch_counters::print(int device) const {
    if (!counter_device_ok(device)) {
        return;
    }
    GGML_LOG_WARN("[ATTN-HOST] host dispatches dev=%d fa=%zu set_rows=%zu refusals=%zu\n", device, fa(device),
                  set_rows(device), refusals(device));
}

size_t attn_host_dispatch_counters::fa(int device) const {
    return counter_device_ok(device) ? fa_[device].load(std::memory_order_relaxed) : 0;
}

size_t attn_host_dispatch_counters::set_rows(int device) const {
    return counter_device_ok(device) ? set_rows_[device].load(std::memory_order_relaxed) : 0;
}

size_t attn_host_dispatch_counters::refusals(int device) const {
    return counter_device_ok(device) ? refusals_[device].load(std::memory_order_relaxed) : 0;
}

void host_dispatch_fail_closed(bool                                ok,
                               const attn_host_dispatch_counters * counters,
                               int                                 device,
                               const ggml_tensor *                 node,
                               const char *                        role) {
    if (ok) {
        return;
    }
    // The counter line first, so it survives the failed graph.
    if (counters) {
        counters->print(device);
    }
    fprintf(stderr, "[SYCL] graph compute unsupported node name=%s op=%s type=%s (segmented %s)\n",
            node && node->name[0] ? node->name : "", node ? ggml_op_name(node->op) : "(null)",
            node ? ggml_type_name(node->type) : "(null)", role ? role : "?");
#if defined(HOST_ZONE_PLAN_MUTANT_FAIL_CLOSED_ASSERTS)
    // Positive control: the helper asserting, which kills the process instead of failing the graph.
    GGML_ASSERT(ok && "segmented graph node refused");
#endif
    // A string literal: the exception stores the pointer and the catch prints what() after the unwind.
    throw ggml_sycl_fallback_error("segmented graph node refused");
}

}  // namespace ggml_sycl
