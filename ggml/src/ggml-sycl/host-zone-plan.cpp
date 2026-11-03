//
// Host zone routing, host-allocation lifetimes, the host SCRATCH demand ledger, and the host holder plan. See
// host-zone-plan.hpp and docs/design/sycl-mmap-host-tier.md §4.4.
//
// This is the tests' RED form: the routing, the grow and the slot admission are today's, and the parts of the plan
// that do not exist yet are inert.
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
// Routing: today's rule, in which the HOST_COMPUTE and EXPERT_CACHE categories select WEIGHT.
// ---------------------------------------------------------------------------------------------------------------

host_zone_id select_zone(alloc_role       role,
                         runtime_category category,
                         alloc_lifetime   lifetime,
                         bool             persistent_owner,
                         bool             kv_spill_to_host) {
    (void) lifetime;
    GGML_ASSERT(!persistent_owner ||
                (!kv_spill_to_host && role != alloc_role::KV && role != alloc_role::EXPERT_STAGING));
    if (kv_spill_to_host || role == alloc_role::KV) {
        return host_zone_id::KV;
    }
    if (role == alloc_role::EXPERT_STAGING) {
        return host_zone_id::SCRATCH;
    }
    if (role == alloc_role::WEIGHT || category == runtime_category::HOST_COMPUTE ||
        category == runtime_category::EXPERT_CACHE || persistent_owner) {
        return host_zone_id::WEIGHT;
    }
    return host_zone_id::STAGING;
}

// ---------------------------------------------------------------------------------------------------------------
// Lifetimes: today no record carries one, the graph boundary counts every row, and no owner boundary is checked.
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
    (void) lifetime;
    return true;
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
    (void) record;
    (void) lifetime;
    (void) owner;
    return false;
}

size_t host_zone_owner_boundary_check(const host_zone_record_view * records,
                                      size_t                        n,
                                      alloc_lifetime                lifetime,
                                      alloc_owner_ref               owner,
                                      bool                          strict) {
    (void) records;
    (void) n;
    (void) lifetime;
    (void) owner;
    (void) strict;
    return 0;
}

bool host_zone_device_check_due(int live_backends_on_device_after_free) {
    (void) live_backends_on_device_after_free;
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// The SCRATCH demand: today's additive grow_scratch_zone(additional_bytes), which grows by every call's bytes.
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

host_scratch_ensure_result host_scratch_ledger::ensure_locked_then_grow(std::unique_lock<std::mutex> & lock,
                                                                        const host_scratch_zone_ops &  ops) {
    (void) lock;
    (void) ops;
    return host_scratch_ensure_result{};
}

namespace {

host_scratch_ensure_result additive_grow(size_t bytes, size_t total, const host_scratch_zone_ops & ops) {
    host_scratch_ensure_result r;
    r.total           = total;
    r.target          = ops.plan_scratch_bytes + total;
    r.capacity_before = ops.capacity ? ops.capacity(ops.user) : 0;
    r.request         = bytes;
    if (bytes > 0 && ops.grow) {
        r.ok = ops.grow(ops.user, bytes);
    }
    const size_t after = ops.capacity ? ops.capacity(ops.user) : 0;
    r.grown            = after > r.capacity_before ? after - r.capacity_before : 0;
    return r;
}

}  // namespace

host_scratch_ensure_result host_scratch_ledger::notify_context(uint64_t                      context_key,
                                                               size_t                        compute_bytes,
                                                               size_t                        output_bytes,
                                                               const host_scratch_zone_ops & ops) {
    size_t total = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        contexts_[context_key].bytes += compute_bytes + output_bytes;
        for (const auto & kv : contexts_) {
            total += kv.second.bytes;
        }
    }
    return additive_grow(compute_bytes + output_bytes, total, ops);
}

host_scratch_ensure_result host_scratch_ledger::holder_demand(const char *                  holder,
                                                              int                           device,
                                                              size_t                        bytes,
                                                              const host_scratch_zone_ops & ops) {
    (void) holder;
    (void) device;
    return additive_grow(bytes, bytes, ops);
}

void host_scratch_ledger::drop_context(uint64_t context_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_.erase(context_key);
}

size_t host_scratch_ledger::total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      total = 0;
    for (const auto & kv : contexts_) {
        total += kv.second.bytes;
    }
    return total;
}

size_t host_scratch_ledger::context_demand(uint64_t context_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = contexts_.find(context_key);
    return it == contexts_.end() ? 0 : it->second.bytes;
}

bool host_scratch_notify(const host_scratch_notify_env & env,
                         ggml_backend_t                  backend,
                         size_t                          host_compute_bytes,
                         size_t                          host_output_bytes) {
    // Today nothing notifies the SYCL side.
    (void) env;
    (void) backend;
    (void) host_compute_bytes;
    (void) host_output_bytes;
    return false;
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
// The host holder plan: today there is none. Slots grow at dispatch and the pools size from their own graph.
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

void host_holder_plans::begin(const uint64_t * keys, size_t n) {
    (void) keys;
    (void) n;
}

host_plan_attribution host_holder_plans::walk(ggml_backend_sched_t         sched,
                                              ggml_cgraph *                graph,
                                              const host_holder_walk_env & env) {
    (void) sched;
    (void) graph;
    (void) env;
    return host_plan_attribution{};
}

bool host_holder_plans::commit(uint64_t key, int * device_out, int * previous_out) {
    (void) key;
    if (device_out) {
        *device_out = -1;
    }
    if (previous_out) {
        *previous_out = -1;
    }
    return false;
}

void host_holder_plans::drop(uint64_t key, int * device_out) {
    (void) key;
    if (device_out) {
        *device_out = -1;
    }
}

host_holder_device_state host_holder_plans::device_state(int device) const {
    (void) device;
    return host_holder_device_state{};
}

bool host_holder_plans::has_pending(uint64_t key) const {
    (void) key;
    return false;
}

bool host_holder_plans::has_committed(uint64_t key) const {
    (void) key;
    return false;
}

bool host_staging_slots::apply(int                              device,
                               const host_holder_device_state & state,
                               const host_staging_allocator &   alloc) {
    (void) device;
    (void) state;
    (void) alloc;
    return true;
}

void * host_staging_slots::admit(int                            device,
                                 int                            kind,
                                 size_t                         bytes,
                                 const host_staging_allocator & alloc,
                                 attn_host_dispatch_counters *  counters) {
    (void) counters;
    if (device < 0 || device >= max_devices || kind < 0 || kind >= HOST_STAGING_KIND_COUNT || bytes == 0) {
        return nullptr;
    }
    host_staging_block & slot = slots_[device][kind];
    if (bytes > slot.cap) {
        host_staging_block fresh;
        const size_t       want = host_staging_slot_bytes(bytes);
        if (!alloc.alloc || !alloc.alloc(alloc.user, device, kind, want, &fresh) || !fresh.ptr) {
            return nullptr;
        }
        fresh.bytes = want;
        fresh.cap   = want - 64;
        if (slot.ptr && alloc.release) {
            alloc.release(alloc.user, &slot);
        }
        slot = fresh;
    }
    return slot.ptr;
}

void host_staging_slots::release_all(const host_staging_allocator & alloc) {
    for (int d = 0; d < max_devices; ++d) {
        for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
            if (slots_[d][k].ptr && alloc.release) {
                alloc.release(alloc.user, &slots_[d][k]);
            }
            slots_[d][k] = host_staging_block{};
        }
    }
}

size_t host_staging_slots::capacity(int device, int kind) const {
    if (device < 0 || device >= max_devices || kind < 0 || kind >= HOST_STAGING_KIND_COUNT) {
        return 0;
    }
    return slots_[device][kind].cap;
}

bool host_staging_slots::planned(int device) const {
    (void) device;
    return false;
}

const host_staging_block & host_staging_slots::block(int device, int kind) const {
    static const host_staging_block none{};
    if (device < 0 || device >= max_devices || kind < 0 || kind >= HOST_STAGING_KIND_COUNT) {
        return none;
    }
    return slots_[device][kind];
}

void host_pools_apply_commit(int device, const host_holder_device_state & state, const host_pool_sink & sink) {
    (void) device;
    (void) state;
    (void) sink;
}

host_pool_init_decision host_pools_init_from_plan(int                              device,
                                                  const host_holder_device_state & state,
                                                  const host_pool_dims &           graph_needs,
                                                  host_pool_init_warnings &        warnings) {
    (void) device;
    (void) state;
    (void) warnings;
    host_pool_init_decision d;
    d.dims = graph_needs;
    d.init = graph_needs.D > 0 && graph_needs.K > 0 && graph_needs.N > 0;
    return d;
}

bool host_holders_apply_device(int                            device,
                               const host_holder_plans &      plans,
                               host_staging_slots &           slots,
                               const host_staging_allocator & alloc,
                               host_scratch_ledger *          ledger,
                               const host_scratch_zone_ops &  ops,
                               const host_pool_sink &         pools,
                               bool                           host_dispatch_on) {
    (void) device;
    (void) plans;
    (void) slots;
    (void) alloc;
    (void) ledger;
    (void) ops;
    (void) pools;
    (void) host_dispatch_on;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// The dispatch counter does not exist yet; the segmented sites throw without it.
// ---------------------------------------------------------------------------------------------------------------

void attn_host_dispatch_counters::intercept(int device, ggml_op op) {
    (void) device;
    (void) op;
}

void attn_host_dispatch_counters::refusal(int device) {
    (void) device;
}

void attn_host_dispatch_counters::print(int device) const {
    (void) device;
}

size_t attn_host_dispatch_counters::fa(int device) const {
    (void) device;
    return 0;
}

size_t attn_host_dispatch_counters::set_rows(int device) const {
    (void) device;
    return 0;
}

size_t attn_host_dispatch_counters::refusals(int device) const {
    (void) device;
    return 0;
}

void host_dispatch_fail_closed(bool                                ok,
                               const attn_host_dispatch_counters * counters,
                               int                                 device,
                               const ggml_tensor *                 node,
                               const char *                        role) {
    (void) counters;
    (void) device;
    if (ok) {
        return;
    }
    fprintf(stderr, "[SYCL] graph compute unsupported node name=%s op=%s type=%s (segmented %s)\n",
            node && node->name[0] ? node->name : "", node ? ggml_op_name(node->op) : "(null)",
            node ? ggml_type_name(node->type) : "(null)", role ? role : "?");
    throw ggml_sycl_fallback_error("segmented graph node refused");
}

}  // namespace ggml_sycl
