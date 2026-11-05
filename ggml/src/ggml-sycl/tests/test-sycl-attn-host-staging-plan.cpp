//
// test-sycl-attn-host-staging-plan: the attention host staging slots' admission, the host holder plan's walk, commit
// and drop, the CPU-expert pools' sizing from the plan, the dispatch counter and the fail-closed helper
// (docs/design/sycl-mmap-host-tier.md §4.4 "At dispatch no slot grows", "Pending, then committed", "The walks of one
// reserve merge as a maximum", "And the pools are sized from the plan", "The dispatch counter"; §M62 (E), §M69).
//
// Host-only. The allocator, the SCRATCH zone and the pools are fakes that record what they are asked for, and the
// scheduler is a fixed node-to-backend assignment injected through host_holder_walk_env::tensor_backend. No
// unified_cache is constructed and SYCL is not initialized: the pools take their host allocations through the fake
// sink, not a cache.
//
// Positive controls, one compile-time macro and one ctest target each (ggml/src/ggml-sycl/CMakeLists.txt):
//   HOST_ZONE_PLAN_MUTANT_WALK_TG_REPLACES_PP     a later walk of one reserve replaces an earlier one
//   HOST_STAGING_PLAN_MUTANT_NO_BEGIN  (this file) a stale pending plan from a failed candidate is promoted
//   HOST_ZONE_PLAN_MUTANT_CAPACITY_HIGH_WATER     the high-water over every plan ever committed
//   HOST_STAGING_PLAN_MUTANT_FA_AFTER_STAGING (this file) fa counted only after the staging succeeds
//   HOST_ZONE_PLAN_MUTANT_FAIL_CLOSED_ASSERTS     the fail-closed helper asserting
//   HOST_ZONE_PLAN_MUTANT_POOLS_OWN_SCAN          the pools sized from their own graph
//   HOST_ZONE_PLAN_MUTANT_POOLS_CHECK_REMOVED     the over-plan check removed
//   HOST_ZONE_PLAN_MUTANT_POOLS_CHECK_STORAGE_SCAN the over-plan check reading the storage scan again
//   HOST_ZONE_PLAN_MUTANT_POOLS_GATE_IS_HOST      ggml_backend_buffer_is_host as the pools' gate
//   HOST_ZONE_PLAN_MUTANT_POOLS_INIT_ONCE         init-once kept
//   HOST_ZONE_PLAN_MUTANT_DISPATCH_GROW           the dispatch-time grow restored
//   HOST_ZONE_PLAN_MUTANT_PLAN_IGNORES_SWITCH     the plan ignoring GGML_SYCL_ATTN_HOST_DISPATCH
//   HOST_ZONE_PLAN_MUTANT_PREDICATE_IS_HOST       ggml_backend_buffer_is_host as the walk's predicate
//   HOST_ZONE_PLAN_MUTANT_HELPER_DEVICE           the notify helper's device for every slot
//   HOST_ZONE_PLAN_MUTANT_UNPLANNED_REFUSES       an unplanned device refusing instead of growing at dispatch
//   HOST_ZONE_PLAN_MUTANT_FREE_RELEASES_PLANNED   a backend's free releasing a planned device's slots too
// HOST_STAGING_PLAN_MUTANT_FA_AFTER_STAGING moves the count in simulate_host_fa, this file's re-enactment of the
// dispatch's entry, so it scores the fixture's model of the order, not production. Production's order is pinned by
// tests/test-sycl-host-zone-boundary-owner-source.py (D1-D3).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fallback-error.hpp"
#include "ggml-backend-impl.h"
#include "host-zone-plan-test-util.hpp"
#include "host-zone-plan.hpp"

#include <cstdlib>
#include <map>
#include <vector>

using namespace ggml_sycl;
using namespace host_zone_plan_test;

namespace {

constexpr size_t MiB = (size_t) 1 << 20;

// ---------------------------------------------------------------------------------------------------------------
// Fake buffer types and buffers. Only get_name and is_host are read.
// ---------------------------------------------------------------------------------------------------------------

const char * kv_host_name(ggml_backend_buffer_type_t) {
    return "SYCL_KV_Host";
}

const char * sycl_host_name(ggml_backend_buffer_type_t) {
    return "SYCL_Host";
}

const char * cpu_name(ggml_backend_buffer_type_t) {
    return "CPU";
}

const char * device_name(ggml_backend_buffer_type_t) {
    return "SYCL0";
}

const char * mapped_name(ggml_backend_buffer_type_t) {
    return "SYCL_Mapped";
}

bool buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

bool buft_is_not_host(ggml_backend_buffer_type_t) {
    return false;
}

ggml_backend_buffer_type make_buft(const char * (*name)(ggml_backend_buffer_type_t),
                                   bool (*is_host)(ggml_backend_buffer_type_t)) {
    ggml_backend_buffer_type t{};
    t.iface.get_name = name;
    t.iface.is_host  = is_host;
    return t;
}

ggml_backend_buffer_type g_kv_host_buft   = make_buft(kv_host_name, buft_is_host);
ggml_backend_buffer_type g_sycl_host_buft = make_buft(sycl_host_name, buft_is_host);
ggml_backend_buffer_type g_cpu_buft       = make_buft(cpu_name, buft_is_host);
ggml_backend_buffer_type g_device_buft    = make_buft(device_name, buft_is_not_host);
ggml_backend_buffer_type g_mapped_buft    = make_buft(mapped_name, buft_is_host);

ggml_backend_buffer make_buf(ggml_backend_buffer_type * buft) {
    ggml_backend_buffer b{};
    b.buft = buft;
    return b;
}

ggml_backend_buffer g_kv_host_buf   = make_buf(&g_kv_host_buft);
ggml_backend_buffer g_sycl_host_buf = make_buf(&g_sycl_host_buft);
ggml_backend_buffer g_cpu_buf       = make_buf(&g_cpu_buft);
ggml_backend_buffer g_device_buf    = make_buf(&g_device_buft);
ggml_backend_buffer g_mapped_buf    = make_buf(&g_mapped_buft);

// ---------------------------------------------------------------------------------------------------------------
// Fake backends and the fixed node-to-backend assignment.
// ---------------------------------------------------------------------------------------------------------------

char                 g_backend_storage[8];
const ggml_backend_t A = reinterpret_cast<ggml_backend_t>(&g_backend_storage[1]);  // device 0, key 1
const ggml_backend_t B = reinterpret_cast<ggml_backend_t>(&g_backend_storage[2]);  // device 0, key 2
const ggml_backend_t C = reinterpret_cast<ggml_backend_t>(&g_backend_storage[3]);  // device 1, key 3
const ggml_backend_t E = reinterpret_cast<ggml_backend_t>(&g_backend_storage[4]);  // device 0, key 4

std::map<const ggml_tensor *, ggml_backend_t> g_assign;
std::map<const ggml_tensor *, int>            g_placement;  // 1: host experts, 0: all on the device; absent: unknown
char                                          g_sched_storage;
const ggml_backend_sched_t                    g_sched = reinterpret_cast<ggml_backend_sched_t>(&g_sched_storage);

bool fake_is_target(ggml_backend_t b) {
    return b == A || b == B || b == C || b == E;
}

int fake_backend_device(ggml_backend_t b) {
    return b == C ? 1 : 0;
}

uint64_t fake_backend_key(ggml_backend_t b) {
    return (uint64_t) (reinterpret_cast<char *>(b) - g_backend_storage);
}

ggml_backend_t fake_tensor_backend(ggml_backend_sched_t, ggml_tensor * node) {
    const auto it = g_assign.find(node);
    return it == g_assign.end() ? nullptr : it->second;
}

bool fake_in_kv_host(const ggml_tensor * t) {
    // The production predicate's own keying: the view_src hop, then get_name by function pointer.
    return host_tensor_in_buft_named(t, kv_host_name);
}

bool fake_host_experts(const ggml_tensor * src0, int device, bool * known, const char ** cause) {
    (void) device;
    const auto it = g_placement.find(src0);
    if (it == g_placement.end()) {
        *known = false;
        *cause = "no placement plan";
        return false;
    }
    *known = true;
    return it->second == 1;
}

host_holder_walk_env walk_env(bool dispatch_on) {
    host_holder_walk_env env;
    env.is_target        = fake_is_target;
    env.tensor_backend   = fake_tensor_backend;
    env.backend_device   = fake_backend_device;
    env.backend_key      = fake_backend_key;
    env.in_kv_host_buft  = fake_in_kv_host;
    env.host_experts     = fake_host_experts;
    env.host_dispatch_on = dispatch_on;
    return env;
}

// ---------------------------------------------------------------------------------------------------------------
// The reserve graphs.
// ---------------------------------------------------------------------------------------------------------------

enum k_where { K_VIEW_KV_HOST, K_CPU, K_SYCL_HOST, K_DEVICE };

// The storage behind every expert tensor: a loaded weight has a data pointer whichever buffer holds it, and
// moe_hybrid_init_once's storage scan keys on that pointer. Never dereferenced.
char g_expert_storage[1];

struct graph_builder {
    ggml_context * ctx   = nullptr;
    ggml_cgraph *  graph = nullptr;

    graph_builder() {
        ggml_init_params p = { 512 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ctx                = ggml_init(p);
        graph              = ggml_new_graph(ctx);
    }

    ~graph_builder() { ggml_free(ctx); }

    // FLASH_ATTN_EXT, D=128, 32 query heads, 8 KV heads. Q and DST: 16384 B per token; MASK: 2 B per (n_kv, token).
    ggml_tensor * fa(int64_t n_tokens, int64_t n_kv, k_where where, ggml_backend_t backend) {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, n_tokens, 32, 1);
        ggml_tensor * k = nullptr;
        if (where == K_VIEW_KV_HOST) {
            ggml_tensor * kv = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, 8192, 8);
            kv->buffer       = &g_kv_host_buf;
            k                = ggml_view_3d(ctx, kv, 128, n_kv, 8, kv->nb[1], kv->nb[2], 0);
        } else {
            k         = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, n_kv, 8);
            k->buffer = where == K_CPU ? &g_cpu_buf : where == K_SYCL_HOST ? &g_sycl_host_buf : &g_device_buf;
        }
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens);
        ggml_tensor * node = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 32, n_tokens, 1);
        node->op           = GGML_OP_FLASH_ATTN_EXT;
        node->src[0]       = q;
        node->src[1]       = k;
        node->src[2]       = k;
        node->src[3]       = mask;
        ggml_set_name(node, "fa_node");
        ggml_build_forward_expand(graph, node);
        g_assign[node] = backend;
        return node;
    }

    // SET_ROWS into a SYCL_KV_Host cache: VAL 4096 B per token, IDS 8 B per token.
    ggml_tensor * set_rows(int64_t n_tokens, ggml_backend_t backend) {
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 1024, 8192);
        cache->buffer       = &g_kv_host_buf;
        ggml_tensor * val   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, n_tokens);
        ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        ggml_tensor * node  = ggml_set_rows(ctx, cache, val, ids);
        ggml_build_forward_expand(graph, node);
        g_assign[node] = backend;
        return node;
    }

    // MUL_MAT_ID over a src0 of K=2048 x N=2880 x 32 experts, top_k experts per token: D = top_k x n_tokens.
    ggml_tensor * moe(int64_t               n_tokens,
                      int64_t               top_k,
                      ggml_backend_buffer * buf,
                      int                   placement,
                      ggml_backend_t        backend) {
        ggml_tensor * src0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 2880, 32);
        src0->buffer       = buf;
        src0->data         = g_expert_storage;
        ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, top_k, n_tokens);
        ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_tokens);
        ggml_tensor * node = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2880, top_k, n_tokens);
        node->op           = GGML_OP_MUL_MAT_ID;
        node->src[0]       = src0;
        node->src[1]       = b;
        node->src[2]       = ids;
        ggml_build_forward_expand(graph, node);
        g_assign[node] = backend;
        if (placement >= 0) {
            g_placement[src0] = placement;
        }
        return node;
    }
};

size_t q_bytes(int64_t n_tokens) {
    return (size_t) (16384 * n_tokens);
}

size_t mask_bytes(int64_t n_kv, int64_t n_tokens) {
    return (size_t) (2 * n_kv * n_tokens);
}

// ---------------------------------------------------------------------------------------------------------------
// Fake allocator, SCRATCH zone and pools.
// ---------------------------------------------------------------------------------------------------------------

struct fake_alloc_state {
    int                 allocs   = 0;
    int                 releases = 0;
    std::vector<size_t> sizes;
};

fake_alloc_state g_alloc_state;

bool fake_alloc(void * user, int device, int kind, size_t bytes, host_staging_block * out) {
    (void) user;
    (void) device;
    (void) kind;
    ++g_alloc_state.allocs;
    g_alloc_state.sizes.push_back(bytes);
    out->ptr   = malloc(bytes);
    out->token = out->ptr;
    return out->ptr != nullptr;
}

void fake_release(void * user, host_staging_block * block) {
    (void) user;
    ++g_alloc_state.releases;
    free(block->ptr);
    *block = host_staging_block{};
}

host_staging_allocator fake_allocator() {
    host_staging_allocator a;
    a.alloc   = fake_alloc;
    a.release = fake_release;
    return a;
}

size_t huge_capacity(void *) {
    return (size_t) 1 << 40;
}

host_scratch_zone_ops zone_ops() {
    host_scratch_zone_ops ops;
    ops.capacity  = huge_capacity;
    ops.chunk_cap = 256 * MiB;
    ops.alignment = 64;
    return ops;
}

struct fake_pools {
    bool           initialized[2]{};
    host_pool_dims held[2]{};
    int            inits[2]{};
};

fake_pools g_pools;

bool pools_initialized(void *, int device, host_pool_dims * held) {
    *held = g_pools.held[device];
    return g_pools.initialized[device];
}

void pools_init(void *, int device, const host_pool_dims & dims) {
    g_pools.initialized[device] = true;
    g_pools.held[device]        = dims;
    ++g_pools.inits[device];
}

void pools_shutdown(void *, int device) {
    g_pools.initialized[device] = false;
    g_pools.held[device]        = host_pool_dims{};
}

host_pool_sink pool_sink() {
    host_pool_sink s;
    s.initialized = pools_initialized;
    s.init        = pools_init;
    s.shutdown    = pools_shutdown;
    return s;
}

// One process's worth of the backend's holder state.
struct holders {
    host_holder_plans   plans;
    host_staging_slots  slots{ plans };
    host_scratch_ledger ledger;

    ~holders() { slots.release_all(fake_allocator()); }

    bool apply(int device, bool dispatch_on) {
        return host_holders_apply_device(device, plans, slots, fake_allocator(), &ledger, zone_ops(), pool_sink(),
                                         dispatch_on);
    }

    void begin(std::initializer_list<uint64_t> keys) {
        std::vector<uint64_t> v(keys);
        plans.begin(v.data(), v.size());
    }

    void commit_apply(uint64_t key, bool dispatch_on) {
        int device   = -1;
        int previous = -1;
        plans.commit(key, &device, &previous);
        if (previous >= 0 && previous != device) {
            apply(previous, dispatch_on);
        }
        if (device >= 0) {
            apply(device, dispatch_on);
        }
    }

    void drop_apply(uint64_t key, bool dispatch_on) {
        int device = -1;
        plans.drop(key, &device);
        if (device >= 0) {
            apply(device, dispatch_on);
        }
    }
};

void reset_fakes() {
    g_alloc_state = fake_alloc_state{};
    g_pools       = fake_pools{};
    g_assign.clear();
    g_placement.clear();
}

// ---------------------------------------------------------------------------------------------------------------
// The cases.
// ---------------------------------------------------------------------------------------------------------------

void test_dispatch_off() {
    reset_fakes();
    holders       h;
    graph_builder pp;
    pp.fa(512, 512, K_VIEW_KV_HOST, A);
    pp.moe(512, 2, &g_mapped_buf, 1, A);
    graph_builder tg;
    tg.fa(1, 512, K_VIEW_KV_HOST, A);
    capture_begin();
    h.begin({ 1 });
    h.plans.walk(g_sched, pp.graph, walk_env(false));
    h.plans.walk(g_sched, tg.graph, walk_env(false));
    h.commit_apply(1, false);
    HZP_CHECK(count_of(captured_log(), "[ATTN-HOST] staging not planned (host dispatch off)") == 1,
              "with the host dispatch off the `staging not planned` line printed %zu time(s), expected 1",
              count_of(captured_log(), "[ATTN-HOST] staging not planned (host dispatch off)"));
    HZP_CHECK(g_alloc_state.allocs == 0, "with the host dispatch off the plan allocated %d slot(s)",
              g_alloc_state.allocs);
    const size_t pools = host_pool_pinned_bytes(h.plans.device_state(0).pools);
    HZP_CHECK(pools > 0 && h.ledger.total() == pools,
              "with the host dispatch off the ledger holds %zu B, expected the pools' %zu B and no staging demand",
              h.ledger.total(), pools);
}

void test_walk_alone_changes_nothing() {
    reset_fakes();
    holders       h;
    graph_builder g;
    g.fa(32, 1024, K_VIEW_KV_HOST, A);
    h.begin({ 1 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    HZP_CHECK(g_alloc_state.allocs == 0 && h.ledger.total() == 0 && !h.plans.device_state(0).planned,
              "a walk alone allocated %d slot(s) or registered %zu B of demand", g_alloc_state.allocs,
              h.ledger.total());
    h.commit_apply(1, true);
    HZP_CHECK(g_alloc_state.allocs > 0 && h.ledger.total() > 0, "the commit after the walk allocated no slot");
}

void test_walks_merge_as_max() {
    reset_fakes();
    holders       h;
    // A top-10 MoE graph with n_seq_max 1: a pp walk at n_tokens 512, then a tg walk at 1 token.
    graph_builder pp;
    pp.fa(512, 512, K_VIEW_KV_HOST, A);
    pp.moe(512, 10, &g_mapped_buf, 1, A);
    graph_builder tg;
    tg.fa(1, 512, K_VIEW_KV_HOST, A);
    tg.moe(1, 10, &g_mapped_buf, 1, A);
    h.begin({ 1 });
    h.plans.walk(g_sched, pp.graph, walk_env(true));
    h.plans.walk(g_sched, tg.graph, walk_env(true));
    h.commit_apply(1, true);
    const host_holder_device_state s = h.plans.device_state(0);
    const size_t                   q = host_staging_slot_capacity(q_bytes(512));
    HZP_CHECK(s.slot_capacity[HOST_STAGING_Q] == q, "committed q=%zu expected %zu", s.slot_capacity[HOST_STAGING_Q], q);
    HZP_CHECK(s.slot_capacity[HOST_STAGING_DST] == q, "committed dst=%zu expected %zu",
              s.slot_capacity[HOST_STAGING_DST], q);
    HZP_CHECK(s.slot_capacity[HOST_STAGING_MASK] == host_staging_slot_capacity(mask_bytes(512, 512)),
              "committed mask=%zu expected %zu", s.slot_capacity[HOST_STAGING_MASK],
              host_staging_slot_capacity(mask_bytes(512, 512)));
    HZP_CHECK(s.pools.D == 5120, "committed D=%lld expected D=5120", (long long) s.pools.D);
}

void test_begin_clears_every_pending() {
    reset_fakes();
    holders h;
    // The first reserve walks nodes on backends A and B, then fails before its commit.
    {
        graph_builder g;
        g.fa(64, 1024, K_VIEW_KV_HOST, A);
        g.fa(64, 1024, K_VIEW_KV_HOST, B);
        h.begin({ 1, 2 });
        h.plans.walk(g_sched, g.graph, walk_env(true));
    }
    // The second reserve's walk attributes only to A, and it commits both.
    graph_builder g2;
    g2.fa(32, 1024, K_VIEW_KV_HOST, A);
#if !defined(HOST_STAGING_PLAN_MUTANT_NO_BEGIN)
    h.begin({ 1, 2 });
#endif
    h.plans.walk(g_sched, g2.graph, walk_env(true));
    h.commit_apply(1, true);
    h.commit_apply(2, true);
    HZP_CHECK(h.plans.has_committed(1), "backend A committed no plan");
    HZP_CHECK(!h.plans.has_committed(2), "backend B committed a plan its reserve never walked");
}

void test_slot_rule_and_admission() {
    reset_fakes();
    holders       h;
    // Planned at n_ctx 4096, 32 tokens: Q, DST, MASK, VAL, IDS.
    graph_builder g;
    g.fa(32, 4096, K_VIEW_KV_HOST, A);
    g.set_rows(32, A);
    h.begin({ 1 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);

    const size_t figures[HOST_STAGING_KIND_COUNT] = { q_bytes(32), q_bytes(32), mask_bytes(4096, 32), 4096 * 32,
                                                      8 * 32 };
    size_t       sum                              = 0;
    for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
        HZP_CHECK(h.slots.capacity(0, k) == host_staging_slot_capacity(figures[k]),
                  "slot %s capacity %zu, expected %zu (the slot rule over %zu B)", host_staging_kind_name(k),
                  h.slots.capacity(0, k), host_staging_slot_capacity(figures[k]), figures[k]);
        sum += host_staging_slot_bytes(figures[k]);
    }
    HZP_CHECK(h.ledger.total() == sum, "the DEVICE staging demand is %zu B, expected the slots' sum %zu B",
              h.ledger.total(), sum);

    // Every request at or below its slot's capacity returns the slot and allocates nothing, including a MASK request
    // that grows with n_kv up to the planned n_ctx.
    const int                   allocs = g_alloc_state.allocs;
    attn_host_dispatch_counters counters;
    for (int64_t n_kv : { 256, 1024, 4096 }) {
        void * p = h.slots.admit(0, HOST_STAGING_MASK, mask_bytes(n_kv, 32), fake_allocator(), &counters);
        HZP_CHECK(p == h.slots.block(0, HOST_STAGING_MASK).ptr && p, "MASK at n_kv %lld did not return its slot",
                  (long long) n_kv);
    }
    for (int k = 0; k < HOST_STAGING_KIND_COUNT; ++k) {
        HZP_CHECK(h.slots.admit(0, k, h.slots.capacity(0, k), fake_allocator(), &counters) != nullptr,
                  "a %s request at its capacity was refused", host_staging_kind_name(k));
    }
    HZP_CHECK(g_alloc_state.allocs == allocs, "requests within capacity allocated %d time(s)",
              g_alloc_state.allocs - allocs);

    // One byte above its capacity: null and the WARN naming the kind; no allocation.
    capture_begin();
    const size_t over = h.slots.capacity(0, HOST_STAGING_Q) + 1;
    void *       p    = h.slots.admit(0, HOST_STAGING_Q, over, fake_allocator(), &counters);
    char         want[256];
    snprintf(want, sizeof(want),
             "[ATTN-HOST] staging request %zu B for kind attn_host_q exceeds the planned %zu B on "
             "device 0; declining host dispatch",
             over, over - 1);
    HZP_CHECK(p == nullptr && log_has(want), "an over-plan Q request was not refused with the WARN");
    HZP_CHECK(g_alloc_state.allocs == allocs, "the over-plan request recorded an allocation");
    HZP_CHECK(counters.refusals(0) == 1, "the refusal counted %zu, expected 1", counters.refusals(0));
}

void test_unplanned_slot_refused() {
    reset_fakes();
    holders       h;
    graph_builder g;
    g.fa(32, 1024, K_VIEW_KV_HOST, A);  // no SET_ROWS: VAL and IDS are planned at 0 on a planned device
    h.begin({ 1 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);
    capture_begin();
    attn_host_dispatch_counters counters;
    void *                      p = h.slots.admit(0, HOST_STAGING_VAL, 100, fake_allocator(), &counters);
    HZP_CHECK(p == nullptr && log_has("for kind attn_host_setrows_val exceeds the planned 0 B on device 0"),
              "a request to a slot never planned was not refused with the WARN");
}

// The slots' pre-notify fallback (§4.4, ruling 2's transitional fallback): on a device no plan is committed for, a slot
// grows at dispatch, and one WARN names the fallback across every admit. A commit on that device ends it: from then
// on a request over the plan is refused.
void test_unplanned_device_grows_until_a_commit() {
    reset_fakes();
    holders                     h;
    attn_host_dispatch_counters counters;
    const char *                warn = "[ATTN-HOST] staging dev=1 unplanned; slots grow at dispatch";
    capture_begin();
    void *    small    = h.slots.admit(1, HOST_STAGING_Q, q_bytes(16), fake_allocator(), &counters);  // a first slot
    void *    large    = h.slots.admit(1, HOST_STAGING_Q, q_bytes(64), fake_allocator(), &counters);  // grows past it
    void *    fits     = h.slots.admit(1, HOST_STAGING_Q, q_bytes(32), fake_allocator(), &counters);  // the grown slot
    const int returned = (small ? 1 : 0) + (large ? 1 : 0) + (fits ? 1 : 0);
    HZP_CHECK(returned == 3 && g_alloc_state.allocs == 2 && counters.refusals(1) == 0,
              "an unplanned device refused a dispatch: %d of 3 admits returned a slot, %d allocation(s), %zu "
              "refusal(s), expected 3, 2 and 0",
              returned, g_alloc_state.allocs, counters.refusals(1));
    HZP_CHECK(h.slots.capacity(1, HOST_STAGING_Q) >= q_bytes(64) && fits == large,
              "the fallback slot holds %zu B after a %zu B request, or a smaller request did not reuse it",
              h.slots.capacity(1, HOST_STAGING_Q), q_bytes(64));
    HZP_CHECK(count_of(captured_log(), warn) == 1,
              "the unplanned-device WARN printed %zu time(s) across three admits, expected 1",
              count_of(captured_log(), warn));

    // The device's first commit: the design behaviour, refusal over the plan, and no fallback line.
    graph_builder g;
    g.fa(16, 1024, K_VIEW_KV_HOST, C);
    h.begin({ 3 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(3, true);
    capture_begin();
    const int    allocs = g_alloc_state.allocs;
    const size_t over   = h.slots.capacity(1, HOST_STAGING_Q) + 1;
    void *       p      = h.slots.admit(1, HOST_STAGING_Q, over, fake_allocator(), &counters);
    HZP_CHECK(p == nullptr && g_alloc_state.allocs == allocs && counters.refusals(1) == 1 &&
                  log_has("exceeds the planned") && !log_has("unplanned"),
              "after a commit on device 1 an over-plan request was not refused over the plan: admitted=%d, %d "
              "allocation(s), %zu refusal(s)",
              p ? 1 : 0, g_alloc_state.allocs - allocs, counters.refusals(1));

    // An unrelated backend's free, a load's temporary backend or a context with no plan, leaves device 1's committed
    // plan in force. It drops no plan, and of the slots it releases only the unplanned device 0's fallback slot goes.
    void *       fallback    = h.slots.admit(0, HOST_STAGING_Q, q_bytes(16), fake_allocator(), &counters);
    const size_t planned_cap = h.slots.capacity(1, HOST_STAGING_Q);
    capture_begin();
    const int releases = g_alloc_state.releases;
    h.drop_apply(99, true);                       // the freed backend's own plan: it has none
    h.slots.release_unplanned(fake_allocator());  // what every backend free does to the slots
    const int    allocs_after = g_alloc_state.allocs;
    const size_t refusals     = counters.refusals(1);
    void *       kept         = h.slots.admit(1, HOST_STAGING_Q, planned_cap, fake_allocator(), &counters);
    HZP_CHECK(fallback && kept && h.slots.planned(1) && g_alloc_state.allocs == allocs_after &&
                  counters.refusals(1) == refusals && !log_has("unplanned") && !log_has("exceeds the planned"),
              "after an unrelated backend's free, device 1 did not admit by its plan: admitted=%d planned=%d, %d "
              "allocation(s), %zu refusal(s)",
              kept ? 1 : 0, h.slots.planned(1) ? 1 : 0, g_alloc_state.allocs - allocs_after,
              counters.refusals(1) - refusals);
    HZP_CHECK(g_alloc_state.releases == releases + 1 && h.slots.capacity(0, HOST_STAGING_Q) == 0 &&
                  h.slots.capacity(1, HOST_STAGING_Q) == planned_cap,
              "the unrelated free released %d slot(s), leaving device 0 at %zu B and device 1 at %zu B, expected 1 "
              "(device 0's fallback slot), 0 and %zu",
              g_alloc_state.releases - releases, h.slots.capacity(0, HOST_STAGING_Q),
              h.slots.capacity(1, HOST_STAGING_Q), planned_cap);
}

void test_recommit_replaces() {
    reset_fakes();
    holders       h;
    graph_builder big;
    big.fa(64, 1024, K_VIEW_KV_HOST, A);
    graph_builder small;
    small.fa(16, 1024, K_VIEW_KV_HOST, A);
    graph_builder big_e;
    big_e.fa(64, 1024, K_VIEW_KV_HOST, E);
    graph_builder small_a;
    small_a.fa(16, 1024, K_VIEW_KV_HOST, A);

    // A larger plan grows each slot once.
    h.begin({ 1 });
    h.plans.walk(g_sched, big.graph, walk_env(true));
    h.commit_apply(1, true);
    HZP_CHECK(g_alloc_state.allocs == 3, "the first plan allocated %d slot(s), expected 3 (Q, DST, MASK)",
              g_alloc_state.allocs);
    // A smaller one from the same context re-allocates each down to it.
    h.begin({ 1 });
    h.plans.walk(g_sched, small.graph, walk_env(true));
    h.commit_apply(1, true);
    HZP_CHECK(h.slots.capacity(0, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(16)),
              "a smaller re-commit left Q at %zu, expected %zu", h.slots.capacity(0, HOST_STAGING_Q),
              host_staging_slot_capacity(q_bytes(16)));

    // A second live context holds the larger plan: the smaller commit leaves the slots at the larger.
    h.begin({ 4 });
    h.plans.walk(g_sched, big_e.graph, walk_env(true));
    h.commit_apply(4, true);
    h.begin({ 1 });
    h.plans.walk(g_sched, small_a.graph, walk_env(true));
    h.commit_apply(1, true);
    HZP_CHECK(h.slots.capacity(0, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(64)),
              "with a second context holding the larger plan, Q is %zu, expected %zu",
              h.slots.capacity(0, HOST_STAGING_Q), host_staging_slot_capacity(q_bytes(64)));
    // Freeing that context re-allocates them down.
    h.drop_apply(4, true);
    HZP_CHECK(h.slots.capacity(0, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(16)),
              "freeing the larger context left Q at %zu, expected %zu", h.slots.capacity(0, HOST_STAGING_Q),
              host_staging_slot_capacity(q_bytes(16)));
}

void test_auto_ubatch_trial() {
    reset_fakes();
    holders h;
    // Candidates 512, 1024, 2048, settling on 512: each is a reserve of the same context.
    for (int64_t n : { 512, 1024, 2048, 512 }) {
        graph_builder g;
        g.fa(n, 1024, K_VIEW_KV_HOST, C);
        h.begin({ 3 });
        h.plans.walk(g_sched, g.graph, walk_env(true));
        h.commit_apply(3, true);
    }
    HZP_CHECK(h.slots.capacity(1, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(512)),
              "an auto-n_ubatch trial settled on 512 ends at Q capacity %zu, expected %zu (the 512 plan)",
              h.slots.capacity(1, HOST_STAGING_Q), host_staging_slot_capacity(q_bytes(512)));
}

// This fixture's model of the host FLASH_ATTN_EXT dispatch's entry: the interception counts before any staging is
// asked for. Production's entry is in ggml_sycl_compute_forward_impl, pinned by D1 of the boundary-owner pin.
bool simulate_host_fa(attn_host_dispatch_counters & counters, host_staging_slots & slots, int device, size_t q) {
#if !defined(HOST_STAGING_PLAN_MUTANT_FA_AFTER_STAGING)
    counters.intercept(device, GGML_OP_FLASH_ATTN_EXT);
#endif
    void * p = slots.admit(device, HOST_STAGING_Q, q, fake_allocator(), &counters);
    if (!p) {
        return false;
    }
#if defined(HOST_STAGING_PLAN_MUTANT_FA_AFTER_STAGING)
    // Positive control: fa counted only after the staging succeeded.
    counters.intercept(device, GGML_OP_FLASH_ATTN_EXT);
#endif
    return true;
}

void test_dispatch_counter_and_fail_closed() {
    reset_fakes();
    holders       h;
    graph_builder g;
    ggml_tensor * node = g.fa(16, 1024, K_VIEW_KV_HOST, A);
    h.begin({ 1 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);

    attn_host_dispatch_counters counters;
    capture_begin();
    const bool refused_ok = simulate_host_fa(counters, h.slots, 0, h.slots.capacity(0, HOST_STAGING_Q) + 1);
    HZP_CHECK(!refused_ok, "an over-plan host FLASH_ATTN_EXT was admitted");
    HZP_CHECK(counters.fa(0) == 1 && counters.refusals(0) == 1,
              "a refused host FLASH_ATTN_EXT reads fa=%zu refusals=%zu, expected fa=1 refusals=1", counters.fa(0),
              counters.refusals(0));
    simulate_host_fa(counters, h.slots, 0, q_bytes(16));
    HZP_CHECK(count_of(captured_log(), "[ATTN-HOST] first host dispatch dev=0 op=FLASH_ATTN_EXT") == 1,
              "the first-host-dispatch line printed %zu time(s), expected 1",
              count_of(captured_log(), "[ATTN-HOST] first host dispatch dev=0 op=FLASH_ATTN_EXT"));

    // The fail-closed helper: the counter line, then the unsupported-node line, then a throw the caller catches.
    const child_result r = run_in_child([&]() {
        try {
            host_dispatch_fail_closed(false, &counters, 0, node, "record");
            fprintf(stderr, "returned without throwing\n");
        } catch (const ggml_sycl_fallback_error & e) {
            fprintf(stderr, "caught: %s\n", e.what());
        }
    });
    char               counter_line[160];
    snprintf(counter_line, sizeof(counter_line), "[ATTN-HOST] host dispatches dev=0 fa=%zu set_rows=0 refusals=%zu",
             counters.fa(0), counters.refusals(0));
    const size_t at_counter = r.err.find(counter_line);
    const size_t at_node =
        r.err.find("[SYCL] graph compute unsupported node name=fa_node op=FLASH_ATTN_EXT type=f32 (segmented record)");
    const size_t at_caught = r.err.find("caught: segmented graph node refused");
    HZP_CHECK(r.rc == 0, "the fail-closed helper killed the process: rc %d; child stderr:\n%s", r.rc, r.err.c_str());
    HZP_CHECK(at_counter != std::string::npos && at_node != std::string::npos && at_caught != std::string::npos &&
                  at_counter < at_node && at_node < at_caught,
              "the fail-closed helper did not print the counter line, then the unsupported-node line, then throw; "
              "child stderr:\n%s",
              r.err.c_str());
}

void test_walk_uses_the_dispatch_predicate() {
    reset_fakes();
    holders       h;
    graph_builder g;
    g.fa(8, 1024, K_VIEW_KV_HOST, A);  // K a view of a SYCL_KV_Host tensor: counts
    g.fa(16, 1024, K_CPU, A);          // K in a CPU buffer: does not
    g.fa(32, 1024, K_SYCL_HOST, A);    // K in a SYCL_Host buffer (-nkvo): does not
    g.fa(64, 1024, K_DEVICE, A);       // K on the device: does not
    h.begin({ 1 });
    const host_plan_attribution a = h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);
    HZP_CHECK(a.matched == 1 && a.attributed == 1, "the walk counted %d host-KV node(s), expected 1 (the view)",
              a.matched);
    HZP_CHECK(h.slots.capacity(0, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(8)),
              "the planned Q is %zu, expected the view node's %zu", h.slots.capacity(0, HOST_STAGING_Q),
              host_staging_slot_capacity(q_bytes(8)));
}

void test_per_dispatching_device() {
    reset_fakes();
    holders       h;
    graph_builder g;
    g.fa(64, 1024, K_VIEW_KV_HOST, A);  // layers 0-1 on the device-0 backend
    g.fa(64, 1024, K_VIEW_KV_HOST, A);
    g.fa(16, 1024, K_VIEW_KV_HOST, C);  // layers 2-3 on the device-1 backend
    g.fa(16, 1024, K_VIEW_KV_HOST, C);
    h.begin({ 1, 3 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);
    h.commit_apply(3, true);
    HZP_CHECK(h.slots.planned(1), "device 1's slots are unplanned");
    HZP_CHECK(h.slots.capacity(0, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(64)) &&
                  h.slots.capacity(1, HOST_STAGING_Q) == host_staging_slot_capacity(q_bytes(16)),
              "per device, Q is %zu on device 0 and %zu on device 1, expected %zu and %zu",
              h.slots.capacity(0, HOST_STAGING_Q), h.slots.capacity(1, HOST_STAGING_Q),
              host_staging_slot_capacity(q_bytes(64)), host_staging_slot_capacity(q_bytes(16)));
    const int                   allocs = g_alloc_state.allocs;
    attn_host_dispatch_counters counters;
    void *                      p = h.slots.admit(1, HOST_STAGING_Q, q_bytes(16), fake_allocator(), &counters);
    HZP_CHECK(p != nullptr && g_alloc_state.allocs == allocs,
              "a device-1 dispatch within its plan was refused or allocated");
}

void test_pools_sized_from_the_plan() {
    reset_fakes();
    holders       h;
    graph_builder g8;
    g8.moe(8, 2, &g_mapped_buf, 1, A);  // D = 16
    h.begin({ 1 });
    h.plans.walk(g_sched, g8.graph, walk_env(true));
    h.commit_apply(1, true);
    const host_holder_device_state state = h.plans.device_state(0);

    // moe_hybrid_init_once given the same graph at n_tokens 16 (D = 32): the plan's D = 16, and the over-plan WARN.
    graph_builder g16;
    g16.moe(16, 2, &g_mapped_buf, 1, A);
    host_pool_init_warnings warnings;
    capture_begin();
    const host_pool_init_decision d = host_pools_init_from_plan(0, state, g16.graph, fake_host_experts, warnings);
    HZP_CHECK(d.init && d.dims.D == 16, "the pools initialized at D=%lld, expected the plan's D=16",
              (long long) d.dims.D);
    HZP_CHECK(log_has("exceeds its planned") && log_has("graph needs D=32") && log_has("planned D=16"),
              "no `exceeds its planned` line for graph needs D=32");

    // At n_tokens 4 nothing prints.
    graph_builder g4;
    g4.moe(4, 2, &g_mapped_buf, 1, A);
    host_pool_init_warnings quiet;
    capture_begin();
    host_pools_init_from_plan(0, state, g4.graph, fake_host_experts, quiet);
    HZP_CHECK(!log_has("exceeds its planned"), "a graph within the plan printed the over-plan line");

    // With no commit, the pre-notify fallback: the graph's own scan, and the once-only `unplanned` WARN.
    host_holder_plans       none;
    host_pool_init_warnings fresh;
    capture_begin();
    const host_pool_init_decision u =
        host_pools_init_from_plan(0, none.device_state(0), g16.graph, fake_host_experts, fresh);
    host_pools_init_from_plan(0, none.device_state(0), g16.graph, fake_host_experts, fresh);
    HZP_CHECK(u.dims.D == 32 && count_of(captured_log(), "[HOST-ARENA] DEVICE pools dev=0 unplanned;") == 1,
              "with no commit the pools took D=%lld and the `unplanned` line printed %zu time(s), expected D=32 and 1",
              (long long) u.dims.D, count_of(captured_log(), "[HOST-ARENA] DEVICE pools dev=0 unplanned;"));
}

void test_pools_all_device_plan_is_quiet() {
    // Every expert in VRAM: the walk visits the MUL_MAT_ID node but folds none, so the committed plan holds D=K=N=0.
    // The graph moe_hybrid_init_once sees still has storage behind every src0, so its storage scan finds the node. The
    // over-plan check must read the nodes the plan folded, not that scan.
    reset_fakes();
    holders       h;
    graph_builder g;
    g.moe(8, 2, &g_device_buf, 0, A);
    h.begin({ 1 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);
    const host_holder_device_state state   = h.plans.device_state(0);
    const host_pool_dims           storage = host_pool_scan(g.graph, host_pool_has_storage, nullptr);
    HZP_CHECK(state.planned && state.pools.D == 0, "the all-device plan reads planned=%d D=%lld, expected planned D=0",
              (int) state.planned, (long long) state.pools.D);
    HZP_CHECK(storage.D == 16, "the storage scan of the all-device graph found D=%lld, expected 16",
              (long long) storage.D);

    host_pool_init_warnings warnings;
    capture_begin();
    const host_pool_init_decision d = host_pools_init_from_plan(0, state, g.graph, fake_host_experts, warnings);
    HZP_CHECK(!d.init, "an all-device plan initialized the pools at D=%lld", (long long) d.dims.D);
    HZP_CHECK(!log_has("exceeds its planned"), "an all-device plan printed the over-plan WARN: %s",
              captured_log().c_str());
}

void test_pools_gate_is_placement() {
    reset_fakes();
    holders       h;
    graph_builder g;
    g.moe(8, 2, &g_mapped_buf, 0, A);  // SYCL_Mapped is is_host; every expert placed on the device
    capture_begin();
    h.begin({ 1 });
    h.plans.walk(g_sched, g.graph, walk_env(true));
    h.commit_apply(1, true);
    const size_t demand = host_pool_pinned_bytes(h.plans.device_state(0).pools);
    HZP_CHECK(demand == 0, "pool demand %.1f MiB for an all-device src0", (double) demand / MiB);
    HZP_CHECK(log_has("[HOST-ARENA] DEVICE pools dev=0 committed D=0 K=0 N=0 host_expert_nodes=0"),
              "the commit did not print `committed D=0 ... host_expert_nodes=0`");

    // A src0 placement knows nothing about: the WARN names it and nothing is planned.
    reset_fakes();
    holders       h2;
    graph_builder g2;
    g2.moe(8, 2, &g_mapped_buf, -1, A);
    capture_begin();
    h2.begin({ 1 });
    h2.plans.walk(g_sched, g2.graph, walk_env(true));
    HZP_CHECK(log_has("placement unknown (no placement plan)"), "an unknown placement printed no WARN");
}

void test_two_contexts_resize_pools() {
    reset_fakes();
    holders       h;
    graph_builder a8;
    a8.moe(8, 2, &g_mapped_buf, 1, A);    // D = 16
    graph_builder e16;
    e16.moe(16, 2, &g_mapped_buf, 1, E);  // D = 32
    graph_builder a8_again;
    a8_again.moe(8, 2, &g_mapped_buf, 1, A);

    h.begin({ 1 });
    h.plans.walk(g_sched, a8.graph, walk_env(true));
    h.commit_apply(1, true);
    // moe_hybrid_init_once initializes at the committed figures.
    host_pool_init_warnings       w;
    const host_pool_init_decision d =
        host_pools_init_from_plan(0, h.plans.device_state(0), a8.graph, fake_host_experts, w);
    pools_init(nullptr, 0, d.dims);

    capture_begin();
    h.begin({ 4 });
    h.plans.walk(g_sched, e16.graph, walk_env(true));
    h.commit_apply(4, true);
    HZP_CHECK(g_pools.held[0].D == 32, "pools hold D=%lld below committed high-water D=32",
              (long long) g_pools.held[0].D);
    HZP_CHECK(count_of(captured_log(), "[HOST-ARENA] DEVICE pools dev=0 resized D=16->32") == 1,
              "the pools did not re-size once with `resized D=16->32`");

    // B's backend context is destroyed and A re-commits: back to D = 16.
    h.drop_apply(4, true);
    h.begin({ 1 });
    h.plans.walk(g_sched, a8_again.graph, walk_env(true));
    h.commit_apply(1, true);
    HZP_CHECK(g_pools.held[0].D == 16, "after the larger context's drop the pools hold D=%lld, expected 16",
              (long long) g_pools.held[0].D);
}

}  // namespace

int main() {
    test_dispatch_off();
    test_walk_alone_changes_nothing();
    test_walks_merge_as_max();
    test_begin_clears_every_pending();
    test_slot_rule_and_admission();
    test_unplanned_slot_refused();
    test_unplanned_device_grows_until_a_commit();
    test_recommit_replaces();
    test_auto_ubatch_trial();
    test_dispatch_counter_and_fail_closed();
    test_walk_uses_the_dispatch_predicate();
    test_per_dispatching_device();
    test_pools_sized_from_the_plan();
    test_pools_all_device_plan_is_quiet();
    test_pools_gate_is_placement();
    test_two_contexts_resize_pools();
    return finish("test-sycl-attn-host-staging-plan");
}
