//
// What the block ring's device tests share (th32 §3.3, §11 T5): the Q8_0 row geometries their spans and needs derive
// from, the ring's LAZY_STAGING reading, the q2k loader inventory with one lazy table, the coverage pins, the plan
// lines' ring= readings, one model's load through the stage, a read held on an IO worker, and every arm's closing reap.
// Both tests link ggml-sycl and initialise the SYCL backends before any of it runs. Each test includes it once.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../host-cpu-mapping.hpp"
#include "../lazy-io-pool.hpp"
#include "../unified-cache.hpp"
#include "ggml-sycl.h"
#include "host-plan-shaped-inventory.hpp"
#include "lazy-io-test-util.hpp"
#include "lazy-table-test-util.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lazy_ring_fixture {

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using namespace lazy_table_test;
using host_plan_shaped_inventory::loader_inventory;
using lazy_io_test::gate;
using lazy_io_test::gated_source;
using lazy_io_test::wait_for;

// --- the geometry ---

// Three Q8_0 row geometries (32 elements in 34 bytes), named by the block ring read span of one row
// (lazy_block_ring_span): a 34 B row reads 8192 B, an 8194 B row 16384 B and a 24582 B row 32768 B.
constexpr int64_t k_ne0_8k   = 32;
constexpr int64_t k_ne0_16k  = 7712;
constexpr int64_t k_ne0_32k  = 23136;
constexpr size_t  k_row_8k   = 34;
constexpr size_t  k_row_16k  = 8194;
constexpr size_t  k_row_32k  = 24582;
constexpr size_t  k_span_8k  = 8192;
constexpr size_t  k_span_16k = 16384;
constexpr size_t  k_span_32k = 32768;
static_assert(lazy_block_ring_span(k_row_8k) == k_span_8k, "a 34 B row reads 8192 B");
static_assert(lazy_block_ring_span(k_row_16k) == k_span_16k, "an 8194 B row reads 16384 B");
static_assert(lazy_block_ring_span(k_row_32k) == k_span_32k, "a 24582 B row reads 32768 B");

// A plan's need for its widest STORAGE span: LAZY_BLOCK_RING_QD read spans (§3.3).
constexpr size_t k_need_8k  = LAZY_BLOCK_RING_QD * k_span_8k;
constexpr size_t k_need_16k = LAZY_BLOCK_RING_QD * k_span_16k;
static_assert(k_need_8k == 512 * 1024 && k_need_16k == 1024 * 1024, "the needs are 512 KiB and 1 MiB");

// A table no reading holds HOST.
constexpr uint64_t k_huge_table = 1024 * GiB;

// The held read's file identity: one 8192 B span at offset 0, the slot of a 512 KiB ring.
constexpr uint64_t k_read_file = 9;

// --- small readings ---

inline size_t mapped(size_t bytes) {
    return host_cpu_mapped_size(bytes);
}

inline std::string name(lazy_ring_ensure_status s) {
    return lazy_ring_ensure_status_name(s);
}

// The line without its line end, for a FAIL text.
inline std::string chomp(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

// The live bytes of role LAZY_STAGING on device 0, the ring's device in every ensure here. The reader is VOID for a
// device with no cache (§3.1), which main creates before any arm. An unreadable reading VOIDs the arm and never reads
// as 0.
inline uint64_t role_bytes(arm_result & r) {
    uint64_t v = 0;
    if (!unified_cache_role_bytes_for_testing(0, alloc_role::LAZY_STAGING, &v)) {
        void_arm(r, "device 0 has no cache, so its LAZY_STAGING reading is VOID");
    }
    return v;
}

// --- the inventories ---

// A loader-form lazy table: Q8_0 rows of `ne0` elements, `bytes` of them rounded down to a row, read one row per token
// by layer 1, at offset 0 of file 0.
inline ggml_sycl_lazy_table_info loader_table(const char * table_name, int64_t ne0, uint64_t bytes) {
    ggml_sycl_lazy_table_info t{};
    t.name             = table_name;
    t.type             = GGML_TYPE_Q8_0;
    t.rows_per_token   = 1;
    t.row_size         = ggml_row_size(GGML_TYPE_Q8_0, ne0);
    const int64_t rows = static_cast<int64_t>(bytes / t.row_size);
    t.ne[0]            = ne0;
    t.ne[1]            = rows;
    t.ne[2]            = 1;
    t.ne[3]            = 1;
    t.consumer_layer   = 1;
    t.size             = t.row_size * static_cast<size_t>(rows);
    return t;
}

// The q2k loader inventory with one lazy table, which it points at, so the object is neither copied nor moved.
struct lazy_inventory {
    loader_inventory          loader;
    ggml_sycl_lazy_table_info table;

    lazy_inventory(const char * table_name, int64_t ne0, uint64_t bytes) : table(loader_table(table_name, ne0, bytes)) {
        loader.inv.lazy_tables      = &table;
        loader.inv.lazy_table_count = 1;
    }

    lazy_inventory(const lazy_inventory &)             = delete;
    lazy_inventory & operator=(const lazy_inventory &) = delete;

    const ggml_sycl_tensor_inventory * inv() const { return &loader.inv; }
};

// --- the coverage pins ---

using pin_ptr = std::unique_ptr<lazy_test_coverage_pin>;

inline pin_ptr make_pin(uint64_t ring_bytes) {
    return pin_ptr(new lazy_test_coverage_pin(ggml_sycl_test_make_coverage_pin(ring_bytes)));
}

// The pin holds a lease on the one live pool, whose ring reads `ring_bytes`, read from the registry: the helper returns
// a guard even when its take came back empty. VOID naming the reading otherwise.
inline bool pin_engaged(arm_result & r, const pin_ptr & pin, uint64_t ring_bytes) {
    const lazy_registry_state s = registry();
    if (pin && pin->pin() != nullptr && s.live_pools == 1 && s.live_leases >= 1 && s.ring_bytes == ring_bytes) {
        return true;
    }
    void_arm(r, "the fixture's pin holds no lease on a ring of " + num(ring_bytes) + " B: " + describe(s) +
                    ", the ring reads " + num(s.ring_bytes) + " B");
    return false;
}

// --- the log lines ---

// ring= of every [HOST-PLAN] lazy line since the last clear, and the last of them (SIZE_MAX with none). A stage plans
// once per device on the multi-device path, each device's call running the planner, so a pass prints one line per
// card; the last is the plan its candidate keeps.
inline std::vector<size_t> plan_rings() {
    std::vector<size_t> rings;
    for (const std::string & line : log_lines_containing("[HOST-PLAN] lazy ")) {
        rings.push_back(printed(line, "ring"));
    }
    return rings;
}

inline size_t last_ring() {
    const std::vector<size_t> rings = plan_rings();
    return rings.empty() ? SIZE_MAX : rings.back();
}

inline std::string ring_list(const std::vector<size_t> & rings) {
    std::string list;
    for (size_t v : rings) {
        list += (list.empty() ? "" : ", ") + num(v);
    }
    return list.empty() ? std::string("none") : list;
}

// The first of `rings` that is not `want`, or `want` when every one is and there is at least one; SIZE_MAX with none.
inline size_t first_ring_not(const std::vector<size_t> & rings, size_t want) {
    if (rings.empty()) {
        return SIZE_MAX;
    }
    for (size_t v : rings) {
        if (v != want) {
            return v;
        }
    }
    return want;
}

// The one ring= every device's plan line of a pass prints, SIZE_MAX with none; plan lines that disagree within the pass
// FAIL naming them, since the devices plan the same inventory against the same pool.
inline size_t pass_ring(arm_result & r, const std::string & what) {
    const std::vector<size_t> rings = plan_rings();
    if (rings.empty()) {
        fail(r, what + " printed no [HOST-PLAN] lazy line");
        return SIZE_MAX;
    }
    if (first_ring_not(rings, rings.front()) != rings.front()) {
        fail(r, what + " printed ring= " + ring_list(rings) + " across its devices, want one charge");
        return SIZE_MAX;
    }
    return rings.front();
}

inline std::string refused_line() {
    const std::vector<std::string> lines = log_lines_containing("[HOST-PLAN] load refused:");
    return lines.empty() ? std::string("no load-refused line") : chomp(lines.back());
}

// --- the stage and the load ---

inline ggml_sycl_lifecycle_result stage(const ggml_sycl_tensor_inventory * inv, bool early) {
    const ggml_sycl_placement_envelope envelope{ inv->n_ctx, inv->n_ubatch, 1, -1 };
    return ggml_backend_sycl_stage_load_inventory_plan(inv, &envelope, early);
}

// A stage that did not return OK: VOID when the plan refused for want of host memory for the floors and the ring,
// which the host decided, and FAIL naming the stage's refusal line otherwise.
inline void stage_failed(arm_result & r, const std::string & what, ggml_sycl_lifecycle_result rc) {
    const std::string text = what + " returned " + num(rc) + ", want OK: " + refused_line();
    if (log_count_with("[HOST-PLAN] load refused: lazy-table-cache-below-floor") > 0) {
        void_arm(r, text + " (the host's reading cannot hold the floors and the ring)");
    } else {
        fail(r, text);
    }
}

struct model_load {
    bool                  ok = false;
    ggml_sycl_load_txn    txn{};
    ggml_sycl_model_token token{};
    size_t                early_ring = SIZE_MAX;
    size_t                late_ring  = SIZE_MAX;
};

// One model's load: begin, the early and the late stage, and the load's successful end, with each pass's one ring=.
inline model_load load_model(arm_result & r, const lazy_inventory & inv, const std::string & who) {
    model_load m;
    if (ggml_backend_sycl_model_load_begin(&m.txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, who + "'s load transaction did not begin");
        return m;
    }
    for (int pass = 0; pass < 2; ++pass) {
        const bool early = pass == 0;
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv.inv(), early);
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            stage_failed(r, who + "'s " + (early ? "early" : "late") + " stage", rc);
            (void) ggml_backend_sycl_model_load_end(m.txn, false, nullptr);
            return m;
        }
        const size_t ring = pass_ring(r, who + "'s " + (early ? "early" : "late") + " stage");
        if (ring == SIZE_MAX) {
            (void) ggml_backend_sycl_model_load_end(m.txn, false, nullptr);
            return m;
        }
        (early ? m.early_ring : m.late_ring) = ring;
    }
    if (ggml_backend_sycl_model_load_end(m.txn, true, &m.token) != GGML_SYCL_LIFECYCLE_OK || m.token.model_id == 0) {
        void_arm(r, who + "'s load did not end with a model token");
        return m;
    }
    m.ok = true;
    return m;
}

inline void unload(const model_load & m) {
    if (m.ok) {
        (void) ggml_backend_sycl_model_unloaded_token(m.token);
    }
}

// A model's unload that leaves its pool parked: the unload reaps after its erase (§3.3 point (ii)), so a join lease
// held across it keeps the pool live through that reap, and its drop here, the pool's last, parks the pool with no reap
// after it.
inline void unload_parked(const model_load & m) {
    ring_lease held = lazy_io_pool_join();
    unload(m);
    held.reset();
}

// A model's STORAGE materialization's ensure: its 8192 B table's need, with the late plan's charge and the model's
// transaction (§3.3, §10's ring check).
inline lazy_ring_ensure_status ensure_for(const model_load & m) {
    return block_ring_ensure(k_need_8k, k_span_8k, 0, m.late_ring, m.txn.id);
}

// --- the engine's reads ---

inline lazy_io_key read_key() {
    return { lazy_io_class::PREFETCH, 1, 1, lazy_io_kind::TABLE_ROWS };
}

// One read held on a worker by the gate: its waiter is the ticket alone (no landing, no cache), so its delivery only
// takes the ticket to 0, and the read holds its lease on the pool from its submission until that delivery (§3.3).
struct held_read {
    gate           g;
    lazy_io_ticket ticket;

    // Queues the read and waits until a worker holds it; VOID naming why otherwise.
    bool start(arm_result & r) {
        ggml_sycl_test_lazy_io_read_source(gated_source, &g);
        lazy_io_waiter w;
        w.ticket           = &ticket;
        w.epoch            = 1;
        const size_t added = lazy_io_submit(1, -1, { k_read_file, 0, k_span_8k }, read_key(), w);
        if (added != 1 || !g.wait_holders(1)) {
            void_arm(r, "the read was not held on a worker within " + num(k_hook_bound.count()) + " s (" + num(added) +
                            " waiter(s) queued)");
            return false;
        }
        return true;
    }

    // Lets the read go and waits for its delivery. False when it is still outstanding at the bound: the caller then
    // leaks the object, since a worker still holds its ticket.
    bool finish() {
        g.release();
        const bool delivered = wait_for([this] { return ticket.outstanding.load() == 0; });
        ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
        return delivered;
    }
};

// Ends a held read: delivered, or leaked with a FAIL when it is not.
inline void end_read(arm_result & r, std::unique_ptr<held_read> & read) {
    if (read && !read->finish()) {
        fail(r, "the held read was not delivered within " + num(k_hook_bound.count()) + " s of its release");
        (void) read.release();
    }
    read.reset();
}

// --- every arm's end ---

// No read source, no armed hook, a FULL reap, and no pool left, live or parked; with `role0` given, the ring's
// LAZY_STAGING bytes back to it.
inline arm_result settle(arm_result r, uint64_t role0 = UINT64_MAX) {
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    ggml_sycl_test_lazy_plan_hold(nullptr, nullptr);
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    const lazy_registry_state s = registry();
    expect(r, registry_is(s, 0, 0, 0), "after the arm's closing reap: " + describe(s) + ", want 0, 0, 0");
    if (role0 != UINT64_MAX) {
        expect(r, role_bytes(r) == role0,
               "after the arm's closing reap the ring's LAZY_STAGING bytes are " + num(role_bytes(r)) + ", want " +
                   num(role0));
    }
    return r;
}

}  // namespace lazy_ring_fixture
