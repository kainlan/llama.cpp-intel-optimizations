//
// The block ring's plan, its leases and the stage's adoption of the coverage wrapper (th32 §3.3, §4.2, §10, §11 T5):
// the ring a plan charges and the decline of a table wider than the live ring, the lease a staged candidate carries
// across another model's unload and a read's drop, the reaps around every planner caller's take, the owner check's
// silence for a live pool's ring rows, and the strict abort of a STORAGE materialization whose ensure refuses.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The stage initialises the SYCL backends, constructs their caches and plans through them, so the binary is
// registered gpu-serial, serial, with the selector pinned to both discrete cards (level_zero:0,1). It runs iff the
// planner caller takes its multi-device path, by the caller's own predicate,
// ggml_backend_sycl_moe_multi_gpu_requested(), and exits 77 otherwise. No libllama: the fixture builds each inventory
// and drives the load transaction itself, the engine's reads come from the PRIVATE_TESTING read source, and the strict
// arm's child materializes a table from a file the fixture writes in the working directory.
//
// --mutant <name> selects one runtime mutant of the PRIVATE_TESTING selector (§11) and runs only the arm that kills it;
// each --mutant registration passes only when that arm prints its FAIL line. Every arm names its engagement check and
// its mutant. An arm whose engagement check fails is VOID, which fails the binary like a FAIL. Every arm runs, so one
// failure does not hide the rest, and each ends with a FULL reap that must leave no pool, live or parked.

#include "../host-cpu-mapping.hpp"
#include "../host-zone-kinds.hpp"
#include "../lazy-io-pool.hpp"
#include "../lazy-table.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-plan-shaped-inventory.hpp"
#include "lazy-io-test-util.hpp"
#include "lazy-table-test-util.hpp"
#include "sycl-test-skip.hpp"

#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using namespace lazy_table_test;
using host_plan_shaped_inventory::loader_inventory;
using host_plan_shaped_inventory::planner_inventory;
using host_plan_shaped_inventory::q2k_shape;
using lazy_io_test::fill_source;
using lazy_io_test::gate;
using lazy_io_test::gated_source;
using lazy_io_test::wait_for;

namespace {

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

// The direct planner's host budget, far above every plan here, so its fit never decides; a table no reading holds
// HOST; and a table this host's reading holds HOST.
constexpr uint64_t k_budget     = 4096 * GiB;
constexpr uint64_t k_huge_table = 1024 * GiB;
constexpr uint64_t k_host_table = 64 * MiB;

// The held and the probing read: file identity 9, one 8192 B span at offset 0, the slot of a 512 KiB ring.
constexpr uint64_t k_read_file = 9;

// The wider-model arms' table and the transaction each arm's plan names: one per arm, since a load prints a table's
// decline line once (§3.3) and each arm is its own load. The decline arm's second table and its two transactions.
const char *       k_wider_table       = "blk.1.wider.weight";
const char *       k_wider_table_2     = "blk.2.wider.weight";
constexpr uint64_t k_wider_host_txn    = 77;
constexpr uint64_t k_wider_refused_txn = 78;
constexpr uint64_t k_wider_p3_txn      = 79;
constexpr uint64_t k_decline_txn       = 80;
constexpr uint64_t k_decline_next_txn  = 81;

// The strict arm's table: Q8_0 rows of 34 B, 4096 of them, the whole 139264 B fixture file, a STORAGE cache of all
// 4096 rows over a floor of 1024. Its plan's need is one block per slot short of LAZY_BLOCK_RING_QD x 8192 B, so the
// creating ensure refuses it as too narrow (§3.3) with one [HOST-PLAN-BUG] line naming 262144 and 524288.
constexpr int64_t  k_strict_rows       = 4096;
constexpr size_t   k_strict_file_bytes = k_row_8k * k_strict_rows;
constexpr size_t   k_strict_need       = k_need_8k - LAZY_BLOCK_RING_QD * LAZY_IO_BLOCK_BYTES;
constexpr size_t   k_strict_floor      = 1024;
constexpr uint64_t k_strict_budget     = 64 * GiB;
constexpr uint64_t k_strict_model_id   = 0x7a4b2f;
constexpr uint64_t k_strict_txn        = 0x4a12;

// The arms' names, which the mutant registrations' PASS_REGULAR_EXPRESSIONs match.
const char * k_arm_ring_plan =
    "the ring plan: one ensure allocates the need, a covered plan charges 0, a wider one declines";
const char * k_arm_wider_host    = "a wider table on a live ring declines STORAGE and plans HOST where it fits";
const char * k_arm_wider_refused = "a wider table that fits nowhere refuses as lazy-table-ring-too-narrow";
const char * k_arm_wider_p3      = "a wider table whose HOST fit holds but P3 does not refuses and plans no HOST";
const char * k_arm_decline_once  = "a load prints each declined table once across its plans, and again after its abort";
const char * k_arm_staging       = "the early and the late pass charge the ring once, on the same STORAGE set";
const char * k_arm_lease         = "a model planned as covered across an unload finds the ring through a read's lease";
const char * k_arm_window        = "the stage's lease holds the pool across a read's drop inside the planner pass";
const char * k_arm_no_storage    = "an inventory with no lazy table creates no pool, alone and after a lazy model";
const char * k_arm_all_host   = "an all-HOST inventory takes the provisional lease and leaves nothing after the call";
const char * k_arm_two_device = "two devices, early then late then unload, give back the pool and the ring's bytes";
const char * k_arm_partial    = "a stage failed after the first device leaves no lease and no parked pool";
const char * k_arm_dry_run    = "the dry run reaps before its take and after its drop and charges the ring as a stage";
const char * k_arm_empty_slot =
    "the public entries read no coverage, and a null pin refuses as lazy-table-ring-unpinned";
const char * k_arm_owner_check = "the block ring's POOL rows are silent while their generation is live or parked";
const char * k_arm_strict = "under strict leases the STORAGE materialization aborts on a ring refusal after its WARN";
const char * k_arm_strict_mismatch =
    "under strict leases the STORAGE materialization aborts on a ring mismatch after its WARN";
const char * k_arm_strict_concurrent =
    "under strict leases the STORAGE materialization aborts on a concurrent ensure after its WARN";

// The strict cases (§10), one for each refusal ring_refusal_is_plan_bug names, each in its own child, since its abort
// ends the process. TOO_NARROW is the strict table's short need on a ring-less pool. MISMATCH and CONCURRENT
// plan the full need, LAZY_BLOCK_RING_QD x 8192 B, charged at its mapped size: MISMATCH on a live ring that a fixture's
// ensure created under transaction 0, which is no plan's creator, so the charge is refused; CONCURRENT on a ring-less
// pool while the ensure hook holds another caller between its reserve and its allocation. k_strict_cases lists them in
// enum order, which indexes g_strict_runs.
enum class strict_case : uint8_t { TOO_NARROW, MISMATCH, CONCURRENT };

struct strict_case_entry {
    strict_case       kind;
    const char *      name;    // the child's argument
    const char *      arm;     // the arm that scores it
    lazy_table_status status;  // the materialization's refusal, which its abort names
};

const strict_case_entry k_strict_cases[] = {
    { strict_case::TOO_NARROW, "too-narrow", k_arm_strict,            lazy_table_status::RING_TOO_NARROW },
    { strict_case::MISMATCH,   "mismatch",   k_arm_strict_mismatch,   lazy_table_status::RING_MISMATCH   },
    { strict_case::CONCURRENT, "concurrent", k_arm_strict_concurrent, lazy_table_status::RING_CONCURRENT },
};
constexpr size_t k_strict_case_count = 3;
static_assert(sizeof(k_strict_cases) / sizeof(k_strict_cases[0]) == k_strict_case_count, "one entry per strict case");

// The two accessors T5 publishes (§10), resolved through the proc table: null in a build that does not export them.
using u64_reader                   = bool (*)(uint64_t *);
const char * k_ring_bytes_proc     = "ggml_sycl_lazy_block_ring_bytes_for_testing";
const char * k_coverage_reads_proc = "ggml_sycl_lazy_ring_coverage_reads_for_testing";
using fail_next_fn                 = void (*)();
const char * k_fail_early_proc     = "ggml_backend_sycl_test_fail_next_stage_inventory_plan_early_after_first_device";
const char * k_fail_late_proc      = "ggml_backend_sycl_test_fail_next_stage_inventory_plan_late_after_first_device";

// The strict child's switch, its whole setting: unified-cache.cpp is the one place the variable's name is read, so this
// file names it only with its "=1" and cuts the name from it to clear it (strict_name).
const char * k_strict_setting = "GGML_SYCL_STRICT_LEASES=1";

ggml_backend_t g_backend0 = nullptr;          // main's backend 0, alive for every arm
std::string    g_fixture_path;
bool           g_direct_unsupported = false;  // the fixture's filesystem refuses O_DIRECT: the strict arms are VOID
// Each strict case's child, by strict_case.
child_run      g_strict_runs[k_strict_case_count];

// --- small readings ---

size_t mapped(size_t bytes) {
    return host_cpu_mapped_size(bytes);
}

std::string name(lazy_ring_ensure_status s) {
    return lazy_ring_ensure_status_name(s);
}

// The line without its line end, for a FAIL text.
std::string chomp(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

template <typename F> F proc(const char * proc_name) {
    return reinterpret_cast<F>(ggml_backend_reg_get_proc_address(ggml_backend_sycl_reg(), proc_name));
}

// The live bytes of role LAZY_STAGING on device 0, the ring's device in every ensure here. The reader is VOID for a
// device with no cache (§3.1), which main creates before any arm. An unreadable reading VOIDs the arm and never reads
// as 0.
uint64_t role_bytes(arm_result & r) {
    uint64_t v = 0;
    if (!unified_cache_role_bytes_for_testing(0, alloc_role::LAZY_STAGING, &v)) {
        void_arm(r, "device 0 has no cache, so its LAZY_STAGING reading is VOID");
    }
    return v;
}

// --- the inventories ---

// A loader-form lazy table: Q8_0 rows of `ne0` elements, `bytes` of them rounded down to a row, read one row per token
// by layer 1, at offset 0 of file 0.
ggml_sycl_lazy_table_info loader_table(const char * table_name, int64_t ne0, uint64_t bytes) {
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

// The planner-form lazy table, as loader_table.
placement_lazy_table_info planner_table(const char * table_name, int64_t ne0, uint64_t bytes) {
    placement_lazy_table_info t;
    t.name             = table_name;
    t.type             = GGML_TYPE_Q8_0;
    t.row_size         = ggml_row_size(GGML_TYPE_Q8_0, ne0);
    const int64_t rows = static_cast<int64_t>(bytes / t.row_size);
    t.ne[0]            = ne0;
    t.ne[1]            = rows;
    t.ne[2]            = 1;
    t.ne[3]            = 1;
    t.rows_per_token   = 1;
    t.consumer_layer   = 1;
    t.table_bytes      = t.row_size * static_cast<size_t>(rows);
    return t;
}

placement_kv_info kv_with(const std::vector<placement_lazy_table_info> & tables, size_t reading, uint64_t txn) {
    placement_kv_info kv{};
    kv.lazy_tables            = tables;
    kv.host_mem_available_pre = reading;
    kv.lazy_load_txn          = txn;
    return kv;
}

// The direct planner on the q2k inventory over both cards, each with a 1 MiB VRAM budget, so the weights are host-
// resident and step 2 is the host zones they size, through compute_multi_device_plan as the stage's multi-device path
// calls it.
placement_plan plan_for(const placement_kv_info & kv, const ring_lease * pin) {
    std::vector<device_budget> budgets;
    for (int d = 0; d < 2; ++d) {
        budgets.push_back({ d, 1 * MiB, 16ull * 1024ull * MiB, 1.0, true });
    }
    return compute_multi_device_plan(budgets, planner_inventory(q2k_shape), q2k_shape.layers, multi_gpu_mode::HYBRID,
                                     kv, nullptr, q2k_shape.experts, k_budget, pin);
}

// The zone terms with no table: step 2, the figure every reading here is derived from.
bool step2_of(arm_result & r, const ring_lease * pin, size_t & step2) {
    const placement_plan bare = plan_for(kv_with({}, 0, 0), pin);
    step2                     = bare.host_zone_terms.total();
    if (bare.entries.empty() || bare.weight_host_bytes == 0) {
        fail(r, "the plan with no table has " + num(bare.entries.size()) + " entries and " +
                    num(bare.weight_host_bytes) + " host weight bytes, so step 2 is not the host zones it sizes");
        return false;
    }
    return true;
}

std::string refusal_name(const placement_plan & plan) {
    return plan.lazy_refusal.empty() ? std::string("none") : plan.lazy_refusal;
}

bool one_table(const placement_plan & plan, lazy_residency residency) {
    return plan.lazy_refusal.empty() && plan.lazy_tables.size() == 1 && plan.lazy_tables[0].residency == residency;
}

// --- the coverage pins ---

using pin_ptr = std::unique_ptr<lazy_test_coverage_pin>;

pin_ptr make_pin(uint64_t ring_bytes) {
    return pin_ptr(new lazy_test_coverage_pin(ggml_sycl_test_make_coverage_pin(ring_bytes)));
}

// The pin holds a lease on the one live pool, whose ring reads `ring_bytes`, read from the registry: the helper returns
// a guard even when its take came back empty. VOID naming the reading otherwise.
bool pin_engaged(arm_result & r, const pin_ptr & pin, uint64_t ring_bytes) {
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
std::vector<size_t> plan_rings() {
    std::vector<size_t> rings;
    for (const std::string & line : log_lines_containing("[HOST-PLAN] lazy ")) {
        rings.push_back(printed(line, "ring"));
    }
    return rings;
}

size_t last_ring() {
    const std::vector<size_t> rings = plan_rings();
    return rings.empty() ? SIZE_MAX : rings.back();
}

std::string ring_list(const std::vector<size_t> & rings) {
    std::string list;
    for (size_t v : rings) {
        list += (list.empty() ? "" : ", ") + num(v);
    }
    return list.empty() ? std::string("none") : list;
}

// The first of `rings` that is not `want`, or `want` when every one is and there is at least one; SIZE_MAX with none.
size_t first_ring_not(const std::vector<size_t> & rings, size_t want) {
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
size_t pass_ring(arm_result & r, const std::string & what) {
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

// The decline line of §3.3, verbatim.
std::string decline_line(uint64_t txn, const char * table, size_t span, size_t live_span) {
    return "[LAZY-PLAN] STORAGE declined for model load=" + num(txn) + " " + table + ": span " + num(span) +
           " B needed, live block ring spans " + num(live_span) + " B";
}

size_t declines() {
    return log_count_with("[LAZY-PLAN] STORAGE declined");
}

size_t placed(const char * table, const char * residency) {
    const std::string head = std::string("[PLACEMENT] lazy ") + table + " ";
    const std::string res  = std::string(" residency=") + residency + " ";
    return log_count_with(head.c_str(), res.c_str());
}

std::string refused_line() {
    const std::vector<std::string> lines = log_lines_containing("[HOST-PLAN] load refused:");
    return lines.empty() ? std::string("no load-refused line") : chomp(lines.back());
}

// The last [HOST-PLAN] lazy line's reading cannot clear P3 with `bytes` resident: the arm is VOID, since the host,
// not the plan, decided.
bool reading_too_low(arm_result & r, size_t bytes) {
    const std::vector<std::string> lines = log_lines_containing("[HOST-PLAN] lazy ");
    if (lines.empty()) {
        return false;
    }
    const size_t reading = printed(lines.back(), "mem_available_pre");
    const size_t zones   = printed(lines.back(), "zones");
    if (reading == SIZE_MAX || zones == SIZE_MAX || reading >= zones + bytes + k_p3_margin) {
        return false;
    }
    void_arm(r, "the load's MemAvailable reading " + num(reading) + " cannot clear P3 with " + num(bytes) +
                    " bytes resident over zones of " + num(zones));
    return true;
}

// --- the stage and the load ---

ggml_sycl_lifecycle_result stage(const ggml_sycl_tensor_inventory * inv, bool early) {
    const ggml_sycl_placement_envelope envelope{ inv->n_ctx, inv->n_ubatch, 1, -1 };
    return ggml_backend_sycl_stage_load_inventory_plan(inv, &envelope, early);
}

// The tables the candidate staged for `txn` carries, UINT64_MAX with no candidate.
uint64_t candidate_tables(ggml_sycl_load_txn txn) {
    ggml_sycl_model_token token{};
    token.load_txn_id = txn.id;
    uint64_t out[2]   = { 0, 0 };
    return ggml_sycl_plan_lazy_tables_for_testing(token, out) ? out[0] : UINT64_MAX;
}

// A stage that did not return OK: VOID when the plan refused for want of host memory for the floors and the ring,
// which the host decided, and FAIL naming the stage's refusal line otherwise.
void stage_failed(arm_result & r, const std::string & what, ggml_sycl_lifecycle_result rc) {
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
model_load load_model(arm_result & r, const lazy_inventory & inv, const std::string & who) {
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

void unload(const model_load & m) {
    if (m.ok) {
        (void) ggml_backend_sycl_model_unloaded_token(m.token);
    }
}

// A model's STORAGE materialization's ensure: its 8192 B table's need, with the late plan's charge and the model's
// transaction (§3.3, §10's ring check).
lazy_ring_ensure_status ensure_for(const model_load & m) {
    return block_ring_ensure(k_need_8k, k_span_8k, 0, m.late_ring, m.txn.id);
}

// --- the engine's reads ---

lazy_io_key read_key() {
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
void end_read(arm_result & r, std::unique_ptr<held_read> & read) {
    if (read && !read->finish()) {
        fail(r, "the held read was not delivered within " + num(k_hook_bound.count()) + " s of its release");
        (void) read.release();
    }
    read.reset();
}

// --- every arm's end ---

// No read source, no armed hook, a FULL reap, and no pool left, live or parked; with `role0` given, the ring's
// LAZY_STAGING bytes back to it.
arm_result settle(arm_result r, uint64_t role0 = UINT64_MAX) {
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

// --- the direct planner arms ---

// The ring plan (§3.3, §4.2, §10's ring check). On a fixture pin's ring-less pool, plan A's widest STORAGE table reads
// 16384 B, so the plan needs LAZY_BLOCK_RING_QD x 16384 B = 1 MiB and charges its mapped size. An ensure one block per
// slot short of that need refuses as too narrow with one [HOST-PLAN-BUG] line naming both numbers and allocates
// nothing; the full need allocates one ring, which the published accessor reads as the need, under role LAZY_STAGING; a
// second call allocates nothing. Plan B, another model's 8192 B table, is covered by that ring: STORAGE with ring= 0
// and no decline, and its ensure with a charge of 0 passes. Plan C's 32768 B table is wider than the ring's 16384 B
// slot: it prints the decline line and, since it cannot be HOST at its reading, refuses as lazy-table-ring-too-narrow.
// Engagement: the pin holds a lease on a ring-less pool, and plan A plans its table STORAGE.
// Mutant: no-span-check, under which the ensure one block per slot short allocates a ring.
arm_result arm_ring_plan() {
    arm_result r;
    pin_ptr    pin   = make_pin(0);
    size_t     step2 = 0;
    if (!pin_engaged(r, pin, 0) || !step2_of(r, pin->pin(), step2)) {
        pin.reset();
        return settle(r);
    }
    const size_t   reading = step2 + 64 * GiB;
    const uint64_t role0   = role_bytes(r);

    const placement_lazy_table_info a = planner_table("blk.1.ring_a.weight", k_ne0_16k, k_huge_table);
    log_clear();
    const placement_plan plan_a = plan_for(kv_with({ a }, reading, 301), pin->pin());
    if (!one_table(plan_a, lazy_residency::STORAGE)) {
        void_arm(r, "plan A did not plan its table STORAGE: refusal " + placement_lazy_refusal_text(plan_a));
        pin.reset();
        return settle(r);
    }
    expect(r, plan_a.block_ring_need == k_need_16k,
           "plan A needs " + num(plan_a.block_ring_need) + " B of ring, want " + num(k_need_16k));
    expect(r, plan_a.block_ring_charge == mapped(k_need_16k),
           "plan A charges " + num(plan_a.block_ring_charge) + " B, want the need's mapped " + num(mapped(k_need_16k)));
    expect(r, last_ring() == plan_a.block_ring_charge,
           "plan A's line prints ring=" + num(last_ring()) + ", want its charge " + num(plan_a.block_ring_charge));

    const size_t   short_need = k_need_16k - LAZY_BLOCK_RING_QD * LAZY_IO_BLOCK_BYTES;
    const uint64_t a0         = registry().ring_allocations;
    log_clear();
    lazy_ring_ensure_status st = block_ring_ensure(short_need, k_span_16k, 0, mapped(short_need), 301);
    const uint64_t          a1 = registry().ring_allocations;
    expect(r, st == lazy_ring_ensure_status::TOO_NARROW && a1 == a0,
           "the ensure one block per slot short returned " + name(st) + " with " + num(a1 - a0) +
               " allocation(s), want lazy-table-ring-too-narrow and none");
    const std::string short_numbers = "need " + num(short_need) + " B, want " + num(k_need_16k) + " B";
    expect(r, log_count_with("[HOST-PLAN-BUG]", short_numbers.c_str()) == 1,
           "the ensure one block per slot short printed " +
               num(log_count_with("[HOST-PLAN-BUG]", short_numbers.c_str())) + " [HOST-PLAN-BUG] lines naming '" +
               short_numbers + "', want 1");

    st = block_ring_ensure(k_need_16k, k_span_16k, 0, plan_a.block_ring_charge, 301);
    expect(r, st == lazy_ring_ensure_status::OK, "the full-need ensure returned " + name(st) + ", want ok");
    const u64_reader ring_bytes = proc<u64_reader>(k_ring_bytes_proc);
    uint64_t         published  = 0;
    if (ring_bytes == nullptr) {
        fail(r, std::string("the block ring accessor ") + k_ring_bytes_proc + " is not exported");
    } else {
        const bool read = ring_bytes(&published);
        expect(r, read && published == k_need_16k,
               "the block ring accessor reads " + (read ? num(published) + " B" : std::string("no ring")) +
                   ", want the need " + num(k_need_16k) + " B");
    }
    expect(r, registry().ring_allocations == a0 + 1,
           "the full-need ensure made " + num(registry().ring_allocations - a0) + " allocation(s), want 1");
    const uint64_t rose = role_bytes(r) - role0;
    expect(r, rose == k_need_16k,
           "the ring's LAZY_STAGING bytes rose by " + num(rose) + ", want the need " + num(k_need_16k));

    st = block_ring_ensure(k_need_16k, k_span_16k, 0, plan_a.block_ring_charge, 301);
    expect(r, st == lazy_ring_ensure_status::OK && registry().ring_allocations == a0 + 1,
           "the second ensure with the same need returned " + name(st) + " with " +
               num(registry().ring_allocations - a0) + " allocation(s) in all, want ok and 1");

    const placement_lazy_table_info b = planner_table("blk.1.ring_b.weight", k_ne0_8k, k_huge_table);
    log_clear();
    const placement_plan plan_b = plan_for(kv_with({ b }, reading, 302), pin->pin());
    expect(r, one_table(plan_b, lazy_residency::STORAGE),
           "plan B did not plan its table STORAGE: refusal " + placement_lazy_refusal_text(plan_b));
    expect(r, plan_b.block_ring_need == k_need_8k && plan_b.block_ring_charge == 0 && last_ring() == 0,
           "plan B needs " + num(plan_b.block_ring_need) + " B and charges " + num(plan_b.block_ring_charge) +
               " B with ring=" + num(last_ring()) + ", want " + num(k_need_8k) + ", 0 and 0: the live ring covers it");
    expect(r, declines() == 0, "plan B printed " + num(declines()) + " decline lines, want none");
    st = block_ring_ensure(k_need_8k, k_span_8k, 0, 0, 302);
    expect(r, st == lazy_ring_ensure_status::OK, "plan B's covered ensure returned " + name(st) + ", want ok");

    const placement_lazy_table_info c = planner_table("blk.1.ring_c.weight", k_ne0_32k, k_huge_table);
    log_clear();
    const placement_plan plan_c = plan_for(kv_with({ c }, reading, 303), pin->pin());
    const std::string    line   = decline_line(303, "blk.1.ring_c.weight", k_span_32k, k_span_16k);
    expect(r, log_count_with(line.c_str()) == 1,
           "plan C printed " + num(log_count_with(line.c_str())) + " lines '" + line + "', want 1");
    expect(r, plan_c.lazy_refusal == "lazy-table-ring-too-narrow",
           "plan C refused '" + refusal_name(plan_c) + "', want lazy-table-ring-too-narrow");

    pin.reset();
    return settle(r, role0);
}

// The wider-model fixture: a fixture's pin on a live 512 KiB ring, whose slot spans 8192 B, step 2 under it, and the
// registry's reading once the ring is live.
struct wider_fixture {
    pin_ptr             pin;
    size_t              step2 = 0;
    lazy_registry_state before;
};

bool wider_begin(arm_result & r, wider_fixture & f) {
    f.pin = make_pin(k_need_8k);
    if (!pin_engaged(r, f.pin, k_need_8k) || !step2_of(r, f.pin->pin(), f.step2)) {
        return false;
    }
    f.before = registry();
    return true;
}

// Nothing overran or replaced the live ring across the plan: its bytes, its allocations and its pool's generation are
// what the pin made, and one read through it ends within the ring by the ring-write recorder.
void wider_ring_intact(arm_result & r, const wider_fixture & f) {
    const lazy_registry_state now = registry();
    expect(r,
           now.ring_bytes == f.before.ring_bytes && now.ring_allocations == f.before.ring_allocations &&
               now.live_generation == f.before.live_generation,
           "the live ring changed across the plan: " + num(now.ring_bytes) + " B, " + num(now.ring_allocations) +
               " allocation(s), generation " + num(now.live_generation) + ", want " + num(f.before.ring_bytes) +
               " B, " + num(f.before.ring_allocations) + " and " + num(f.before.live_generation));
    ggml_sycl_test_lazy_ring_write_end_reset();
    ggml_sycl_test_lazy_io_read_source(fill_source, nullptr);
    std::unique_ptr<lazy_io_ticket> ticket(new lazy_io_ticket());
    lazy_io_waiter                  w;
    w.ticket               = ticket.get();
    w.epoch                = 1;
    const size_t     added = lazy_io_submit(1, -1, { k_read_file, 0, k_span_8k }, read_key(), w);
    lazy_io_ticket * t     = ticket.get();
    const bool       done  = added == 1 && wait_for([t] { return t->outstanding.load() == 0; });
    ggml_sycl_test_lazy_io_read_source(nullptr, nullptr);
    const uint64_t end = ggml_sycl_test_lazy_ring_write_end();
    if (!done) {
        (void) ticket.release();  // a worker may still deliver to it
    }
    expect(r, done,
           "the read through the live ring queued " + num(added) + " waiter(s) and was not delivered within " +
               num(k_hook_bound.count()) + " s");
    expect(r, end > 0 && end <= now.ring_bytes,
           "the largest ring write ends at " + num(end) + ", want within the ring's " + num(now.ring_bytes) + " B");
}

// The wider-model RED, its HOST arm (§4.3 step 4: a table the live ring declines takes the HOST test at its position):
// the second model's 64 MiB table of 16384 B reads fits HOST at a reading of step 2 + its body + 29 GiB + 1 GiB. The
// planner prints the decline line for it before the HOST test, the table then takes the HOST test at its position and
// plans HOST, and the plan charges no ring.
// Engagement: the pin holds a lease on a live 512 KiB ring.
// Mutant: none of its own; a planner that prints no decline fails it.
arm_result arm_wider_host() {
    arm_result    r;
    wider_fixture f;
    if (!wider_begin(r, f)) {
        f.pin.reset();
        return settle(r);
    }
    const placement_lazy_table_info t    = planner_table(k_wider_table, k_ne0_16k, k_host_table);
    const size_t                    body = mapped(lazy_table_host_body_request_bytes(0, t.table_bytes));
    log_clear();
    const placement_plan plan =
        plan_for(kv_with({ t }, f.step2 + body + k_p3_margin + 1 * GiB, k_wider_host_txn), f.pin->pin());
    const std::string line = decline_line(k_wider_host_txn, k_wider_table, k_span_16k, k_span_8k);
    expect(r, log_count_with(line.c_str()) == 1,
           "the plan printed " + num(log_count_with(line.c_str())) + " lines '" + line + "', want 1");
    expect(r, one_table(plan, lazy_residency::HOST),
           "the wider table did not plan HOST: refusal " + placement_lazy_refusal_text(plan) + ", " +
               num(plan.lazy_tables.size()) + " table(s) planned");
    expect(
        r, plan.block_ring_charge == 0 && last_ring() == 0,
        "the HOST plan charges " + num(plan.block_ring_charge) + " B with ring=" + num(last_ring()) + ", want 0 and 0");
    wider_ring_intact(r, f);
    f.pin.reset();
    return settle(r);
}

// The wider-model RED, its refusal arm: the second model's ~1 TiB table of 16384 B reads fits HOST nowhere, so after
// the decline line the plan refuses as lazy-table-ring-too-narrow, its figures naming the live ring's 524288 B.
// Engagement: the pin holds a lease on a live 512 KiB ring.
// Mutant: ignore-live-ring, under which the coverage read returns 0 for the live ring, so the table is STORAGE-eligible
// and the plan refuses nothing.
arm_result arm_wider_refused() {
    arm_result    r;
    wider_fixture f;
    if (!wider_begin(r, f)) {
        f.pin.reset();
        return settle(r);
    }
    const placement_lazy_table_info t = planner_table(k_wider_table, k_ne0_16k, k_huge_table);
    log_clear();
    const placement_plan plan = plan_for(kv_with({ t }, f.step2 + 64 * GiB, k_wider_refused_txn), f.pin->pin());
    expect(r, plan.lazy_refusal == "lazy-table-ring-too-narrow",
           "the wider table planned with refusal '" + refusal_name(plan) + "', want lazy-table-ring-too-narrow");
    expect(r, plan.lazy_refusal_detail.find("ring_bytes=" + num(k_need_8k)) != std::string::npos,
           "the refusal's figures '" + plan.lazy_refusal_detail +
               "' do not name the live ring's ring_bytes=" + num(k_need_8k));
    const std::string line = decline_line(k_wider_refused_txn, k_wider_table, k_span_16k, k_span_8k);
    expect(r, log_count_with(line.c_str()) == 1,
           "the plan printed " + num(log_count_with(line.c_str())) + " lines '" + line + "', want 1");
    wider_ring_intact(r, f);
    f.pin.reset();
    return settle(r);
}

// The wider-model RED, its P3 arm (§4.3 step 4: ring eligibility is an input, and the HOST test a declined table takes
// includes P3): the 64 MiB table's HOST fit holds but P3 does not, at a reading of step 2 + its body + 29 GiB - 1 GiB,
// so after the decline line the plan refuses as lazy-table-ring-too-narrow and plans no HOST. A planner that ignores
// the live ring declines nothing and admits the table STORAGE, and with no P3 room for its floor and the ring's charge
// the plan refuses as lazy-table-cache-below-floor instead.
// Engagement: the pin holds a lease on a live 512 KiB ring.
// Mutant: ignore-live-ring-p3, the ignore-live-ring selector run against this arm.
arm_result arm_wider_p3() {
    arm_result    r;
    wider_fixture f;
    if (!wider_begin(r, f)) {
        f.pin.reset();
        return settle(r);
    }
    const placement_lazy_table_info t    = planner_table(k_wider_table, k_ne0_16k, k_host_table);
    const size_t                    body = mapped(lazy_table_host_body_request_bytes(0, t.table_bytes));
    log_clear();
    const placement_plan plan =
        plan_for(kv_with({ t }, f.step2 + body + k_p3_margin - 1 * GiB, k_wider_p3_txn), f.pin->pin());
    expect(r, plan.lazy_refusal == "lazy-table-ring-too-narrow" && plan.lazy_tables.empty(),
           "the wider table planned with refusal '" + refusal_name(plan) + "' and " + num(plan.lazy_tables.size()) +
               " table(s), want lazy-table-ring-too-narrow and none: its HOST fit holds and P3 does not");
    const std::string line = decline_line(k_wider_p3_txn, k_wider_table, k_span_16k, k_span_8k);
    expect(r, log_count_with(line.c_str()) == 1,
           "the plan printed " + num(log_count_with(line.c_str())) + " lines '" + line + "', want 1");
    wider_ring_intact(r, f);
    f.pin.reset();
    return settle(r);
}

// The decline line's once-per-load rule (§3.3: each declined table prints one loud WARN): a load plans once per device
// and once per pass, so a second plan of the same load and table prints nothing more, while another table of that load
// and the same table under another load each print one. The load's abort erases its record, so the record holds only
// the open loads: the same load and table planned after it prints again.
// Engagement: the pin holds a lease on a live 512 KiB ring, and the first plan prints its line.
// Mutant: decline-every-plan, under which the second plan prints the line again.
arm_result arm_decline_once() {
    arm_result    r;
    wider_fixture f;
    if (!wider_begin(r, f)) {
        f.pin.reset();
        return settle(r);
    }
    const placement_lazy_table_info t       = planner_table(k_wider_table, k_ne0_16k, k_huge_table);
    const placement_lazy_table_info t2      = planner_table(k_wider_table_2, k_ne0_16k, k_huge_table);
    const size_t                    reading = f.step2 + 64 * GiB;
    const std::string               line    = decline_line(k_decline_txn, k_wider_table, k_span_16k, k_span_8k);
    log_clear();
    (void) plan_for(kv_with({ t }, reading, k_decline_txn), f.pin->pin());
    if (log_count_with(line.c_str()) != 1) {
        void_arm(r, "the first plan of load " + num(k_decline_txn) + " printed " + num(log_count_with(line.c_str())) +
                        " lines '" + line + "', want 1");
        (void) unified_cache_note_model_load_abort(k_decline_txn);
        f.pin.reset();
        return settle(r);
    }

    (void) plan_for(kv_with({ t }, reading, k_decline_txn), f.pin->pin());
    expect(r, log_count_with(line.c_str()) == 1,
           "the second plan of load " + num(k_decline_txn) + " printed " + num(log_count_with(line.c_str())) +
               " lines '" + line + "' in all, want 1");
    const std::string line_2 = decline_line(k_decline_txn, k_wider_table_2, k_span_16k, k_span_8k);
    (void) plan_for(kv_with({ t2 }, reading, k_decline_txn), f.pin->pin());
    expect(
        r, log_count_with(line_2.c_str()) == 1,
        "the load's second table printed " + num(log_count_with(line_2.c_str())) + " lines '" + line_2 + "', want 1");
    const std::string line_next = decline_line(k_decline_next_txn, k_wider_table, k_span_16k, k_span_8k);
    (void) plan_for(kv_with({ t }, reading, k_decline_next_txn), f.pin->pin());
    expect(r, log_count_with(line_next.c_str()) == 1,
           "the table under load " + num(k_decline_next_txn) + " printed " + num(log_count_with(line_next.c_str())) +
               " lines '" + line_next + "', want 1");

    const bool aborted = unified_cache_note_model_load_abort(k_decline_txn);
    (void) plan_for(kv_with({ t }, reading, k_decline_txn), f.pin->pin());
    expect(r, aborted && log_count_with(line.c_str()) == 2,
           "after load " + num(k_decline_txn) + "'s abort (" + (aborted ? "clean" : "not clean") +
               ") its plan printed " + num(log_count_with(line.c_str())) + " lines '" + line +
               "' in all, want 2 after a clean abort");
    (void) unified_cache_note_model_load_abort(k_decline_txn);
    (void) unified_cache_note_model_load_abort(k_decline_next_txn);
    f.pin.reset();
    return settle(r);
}

// --- the stage arms ---

// The staging-pass RED (§3.3: coverage is a live ring, not a pool a lease holds): the early and the late stage of one
// load on the ~1 TiB table plan it STORAGE on both passes, and every device's line of each pass charges the ring row's
// charged bytes, the need's mapped size, since no ring is live: the early pass's take built a ring-less pool, which the
// early candidate's lease keeps for the late pass. The ensure with the late charge allocates one ring of the need, and
// the load's abort and a reap give it back.
// Engagement: both stages return OK.
// Mutant: lease-presence-as-coverage, which reads a pool holding a lease as covered and charges 0 on the late pass.
arm_result arm_staging_pass() {
    arm_result         r;
    lazy_inventory     inv("blk.1.stage.weight", k_ne0_8k, k_huge_table);
    const uint64_t     role0 = role_bytes(r);
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return settle(r);
    }
    std::vector<size_t> rings[2];
    size_t              storage[2] = { 0, 0 };
    for (int pass = 0; pass < 2; ++pass) {
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv.inv(), pass == 0);
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            stage_failed(r, pass == 0 ? "the early stage" : "the late stage", rc);
            (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
            return settle(r);
        }
        rings[pass]   = plan_rings();
        storage[pass] = placed("blk.1.stage.weight", "STORAGE");
    }
    // The late pass first: it is where a pool kept by the early candidate's lease could read as covered.
    const size_t charge = mapped(k_need_8k);
    const size_t late   = first_ring_not(rings[1], charge);
    const size_t early  = first_ring_not(rings[0], charge);
    expect(r, late == charge, "the late pass's ring= is " + num(late) + ", want " + num(charge));
    expect(r, early == charge, "the early pass's ring= is " + num(early) + ", want " + num(charge));
    expect(r, storage[0] == rings[0].size() && storage[1] == rings[1].size() && candidate_tables(txn) == 1,
           "the passes placed the table STORAGE " + num(storage[0]) + " and " + num(storage[1]) + " times over " +
               num(rings[0].size()) + " and " + num(rings[1].size()) + " device plans and the candidate carries " +
               num(candidate_tables(txn)) + " table(s), want once per device plan and 1");
    const lazy_ring_ensure_status st = block_ring_ensure(k_need_8k, k_span_8k, 0, charge, txn.id);
    expect(r, st == lazy_ring_ensure_status::OK,
           "the ensure with the late pass's charge returned " + name(st) + ", want ok");
    expect(r, role_bytes(r) - role0 == k_need_8k && registry().ring_allocations == 1,
           "the ledger shows " + num(role_bytes(r) - role0) + " ring bytes from " + num(registry().ring_allocations) +
               " allocation(s), want one ring of " + num(k_need_8k) + " B");
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    return settle(r, role0);
}

// The lease RED (§3.3: a read's lease keeps the pool and its ring past its model's unload): model A is loaded and its
// ensure makes the ring; a read is held in flight on it and A is unloaded, so the read's lease alone holds the pool;
// model B then plans as covered (ring= 0 on both passes) and loads. Once the read is released B's ensure with a charge
// of 0 finds the live ring, and the ledger shows exactly one ring: the need's bytes under LAZY_STAGING from one
// allocation. B's unload and a reap give them back.
// Engagement: A loads, the read is held on a worker, and B loads.
// Mutant: no-candidate-lease, under which a staged plan carries no lease, so no pool survives A's stages and A's ensure
// finds none.
arm_result arm_lease() {
    arm_result     r;
    lazy_inventory inv_a("blk.1.lease_a.weight", k_ne0_8k, k_huge_table);
    lazy_inventory inv_b("blk.1.lease_b.weight", k_ne0_8k, k_huge_table);
    const uint64_t role0 = role_bytes(r);
    model_load     a     = load_model(r, inv_a, "model A");
    if (!a.ok) {
        return settle(r);
    }
    lazy_ring_ensure_status st = ensure_for(a);
    if (st != lazy_ring_ensure_status::OK) {
        fail(r, "model A's ensure returned " + name(st) + ", want ok");
        unload(a);
        return settle(r);
    }
    std::unique_ptr<held_read> read(new held_read());
    if (!read->start(r)) {
        end_read(r, read);
        unload(a);
        return settle(r);
    }
    unload(a);
    model_load b = load_model(r, inv_b, "model B");
    if (!b.ok) {
        end_read(r, read);
        return settle(r);
    }
    expect(r, b.early_ring == 0 && b.late_ring == 0,
           "model B planned ring=" + num(b.early_ring) + " early and ring=" + num(b.late_ring) +
               " late, want 0 and 0: the read's lease keeps A's ring live");
    end_read(r, read);
    const bool drained = wait_for([] { return registry().live_leases == 1; });
    expect(r, drained,
           "after the read's delivery the pool holds " + num(registry().live_leases) + " leases, want 1, B's");
    st = ensure_for(b);
    expect(r, st == lazy_ring_ensure_status::OK, "model B's ensure returned " + name(st) + ", want ok");
    expect(r, role_bytes(r) - role0 == k_need_8k && registry().ring_allocations == 1,
           "the ledger shows " + num(role_bytes(r) - role0) + " ring bytes from " + num(registry().ring_allocations) +
               " allocation(s), want one ring of " + num(k_need_8k) + " B");
    unload(b);
    return settle(r, role0);
}

// What the window arm's hook saw: the leases at its entry, whether the read's drop landed within the bound, and the
// registry after it.
struct window_hook {
    gate *   read_gate    = nullptr;
    bool     entered      = false;
    uint64_t entry_leases = 0;
    bool     observed     = false;
    uint64_t pools_after  = 0;
    uint64_t leases_after = 0;
};

// The plan point's hook (§3.3): B's plan is made and not yet staged. It releases the held read and waits for its drop,
// one lease fewer than at its entry or the pool gone, with no lock held.
void window_hold(void * ctx) {
    window_hook * h = static_cast<window_hook *>(ctx);
    h->entered      = true;
    h->entry_leases = registry().live_leases;
    h->read_gate->release();
    if (h->entry_leases > 0) {
        const uint64_t want = h->entry_leases - 1;
        h->observed         = wait_for([want] {
            const lazy_registry_state s = registry();
            return s.live_leases == want || s.live_pools == 0;
        });
    }
    const lazy_registry_state s = registry();
    h->pools_after              = s.live_pools;
    h->leases_after             = s.live_leases;
}

// The window arm (§3.3: a ring live when the stage plans stays live to its ensure): model A is loaded, ensured and
// unloaded with a read held in flight, and the plan-point hook inside B's early pass releases the read and waits for
// its drop. The stage's lease, taken at its entry, holds the pool through that drop, so B's plan, made while A's ring
// was live, charges ring= 0, and B's ensure with that charge finds the same ring: one ring of the need from one
// allocation, in the pool A made. The hook is one-shot, so it holds the first device's plan, whose line is the pass's
// first; the plan's own figures are scored before the stage's return code, which the second device's plan, made after
// the drop, decides.
// Engagement, VOID counting as failed: the hook ran and observed the read's drop while it held B's planner.
// Mutant: lease-at-stage, under which the stage drops its lease once planned and takes the plan's past the plan point,
// so the drop empties the pool, the stage builds a new empty one, and the ensure of B's 0 charge allocates a ring the
// ledger check refuses as lazy-table-ring-mismatch, naming both numbers.
arm_result arm_window() {
    arm_result     r;
    lazy_inventory inv_a("blk.1.window_a.weight", k_ne0_8k, k_huge_table);
    lazy_inventory inv_b("blk.1.window_b.weight", k_ne0_8k, k_huge_table);
    const uint64_t role0 = role_bytes(r);
    model_load     a     = load_model(r, inv_a, "model A");
    if (!a.ok) {
        return settle(r);
    }
    lazy_ring_ensure_status st = ensure_for(a);
    if (st != lazy_ring_ensure_status::OK) {
        fail(r, "model A's ensure returned " + name(st) + ", want ok");
        unload(a);
        return settle(r);
    }
    std::unique_ptr<held_read> read(new held_read());
    if (!read->start(r)) {
        end_read(r, read);
        unload(a);
        return settle(r);
    }
    const uint64_t generation_a = registry().live_generation;
    unload(a);

    window_hook hook;
    hook.read_gate = &read->g;
    ggml_sycl_test_lazy_plan_hold(window_hold, &hook);
    ggml_sycl_load_txn txn_b{};
    if (ggml_backend_sycl_model_load_begin(&txn_b) != GGML_SYCL_LIFECYCLE_OK) {
        ggml_sycl_test_lazy_plan_hold(nullptr, nullptr);
        void_arm(r, "model B's load transaction did not begin");
        end_read(r, read);
        return settle(r);
    }
    log_clear();
    const ggml_sycl_lifecycle_result rc = stage(inv_b.inv(), true);
    ggml_sycl_test_lazy_plan_hold(nullptr, nullptr);
    const std::vector<size_t> rings_b = plan_rings();
    const size_t              ring_b  = rings_b.empty() ? SIZE_MAX : rings_b.front();
    end_read(r, read);

    if (!hook.entered || !hook.observed) {
        void_arm(r, hook.entered ?
                        "the read's drop was not observed within " + num(k_hook_bound.count()) +
                            " s while the hook held B's planner (" + num(hook.entry_leases) + " leases at its entry)" :
                        std::string("B's planner never reached the plan point"));
    }
    expect(r, ring_b == 0, "model B's early plan charged ring=" + num(ring_b) + ", want 0: A's ring was live");
    st = block_ring_ensure(k_need_8k, k_span_8k, 0, ring_b == SIZE_MAX ? 0 : ring_b, txn_b.id);
    expect(r, st == lazy_ring_ensure_status::OK, "model B's ensure returned " + name(st) + ", want ok");
    expect(r, hook.pools_after == 1 && hook.leases_after + 1 == hook.entry_leases,
           "the read's drop inside the window left " + num(hook.pools_after) + " live pools and " +
               num(hook.leases_after) + " leases, want 1 and " + num(hook.entry_leases - 1) +
               ": the stage's lease did not hold the pool");
    expect(r, registry().live_generation == generation_a,
           "the pool's generation is " + num(registry().live_generation) + ", want A's " + num(generation_a) +
               ": the stage built a new pool");
    if (rc != GGML_SYCL_LIFECYCLE_OK) {
        stage_failed(r, "model B's early stage", rc);
    }
    expect(r, role_bytes(r) - role0 == k_need_8k && registry().ring_allocations == 1,
           "the ledger shows " + num(role_bytes(r) - role0) + " ring bytes from " + num(registry().ring_allocations) +
               " allocation(s), want one ring of " + num(k_need_8k) + " B");
    (void) ggml_backend_sycl_model_load_end(txn_b, false, nullptr);
    return settle(r, role0);
}

// The no-STORAGE arm (§3.3: an inventory with no lazy-table candidate takes no provisional lease): the q2k inventory
// with no lazy table, staged early and then late and loaded, takes no lease and leaves the registry's generation where
// it was, with no pool; and a swap from a lazy model, whose unload left its pool parked, to that inventory leaves none:
// the stage's entry reap frees it.
// Engagement: the dense stages return OK, and the lazy model's unload leaves its pool parked.
// Mutant: take-without-candidate, under which the coverage scope takes a lease for an inventory with no lazy table.
arm_result arm_no_storage() {
    arm_result                         r;
    loader_inventory                   dense;
    const ggml_sycl_tensor_inventory * inv = &dense.inv;
    ggml_sycl_load_txn                 txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the dense load transaction did not begin");
        return settle(r);
    }
    for (int pass = 0; pass < 2; ++pass) {
        const char *   which = pass == 0 ? "early" : "late";
        const uint64_t g0    = registry().generation;
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv, pass == 0);
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            void_arm(r, std::string("the dense ") + which + " stage returned " + num(rc) + ", want OK");
            (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
            return settle(r);
        }
        const lazy_registry_state s = registry();
        expect(r, s.generation == g0,
               "the dense stage moved the registry generation from " + num(g0) + " to " + num(s.generation));
        expect(r, registry_is(s, 0, 0, 0),
               std::string("after the dense ") + which + " stage: " + describe(s) + ", want 0, 0, 0");
    }
    ggml_sycl_model_token token{};
    if (ggml_backend_sycl_model_load_end(txn, true, &token) == GGML_SYCL_LIFECYCLE_OK && token.model_id != 0) {
        (void) ggml_backend_sycl_model_unloaded_token(token);
    }

    lazy_inventory lazy("blk.1.swap.weight", k_ne0_8k, k_huge_table);
    model_load     a = load_model(r, lazy, "the lazy model");
    if (!a.ok) {
        return settle(r);
    }
    const lazy_ring_ensure_status st = ensure_for(a);
    if (st != lazy_ring_ensure_status::OK) {
        fail(r, "the lazy model's ensure returned " + name(st) + ", want ok");
    }
    unload(a);
    const lazy_registry_state parked = registry();
    if (!registry_is(parked, 0, 0, 1)) {
        void_arm(r,
                 "after the lazy model's unload: " + describe(parked) + ", want 0, 0, 1: no parked pool to swap from");
        return settle(r);
    }
    ggml_sycl_load_txn swap{};
    if (ggml_backend_sycl_model_load_begin(&swap) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the swap's load transaction did not begin");
        return settle(r);
    }
    const ggml_sycl_lifecycle_result rc = stage(inv, true);
    const lazy_registry_state        s  = registry();
    expect(r, rc == GGML_SYCL_LIFECYCLE_OK, "the swap's dense early stage returned " + num(rc) + ", want OK");
    expect(r, registry_is(s, 0, 0, 0), "after the swap's dense early stage: " + describe(s) + ", want 0, 0, 0");
    (void) ggml_backend_sycl_model_load_end(swap, false, nullptr);
    return settle(r);
}

// The all-HOST arm (§3.3: when no candidate ends STORAGE, the provisional lease's drop is the only one): a 64 MiB table
// this host's reading holds HOST. Each stage takes the provisional lease (the registry's generation moves by exactly
// 1), records none, and returns with no pool, no parked node and the ring's LAZY_STAGING bytes unchanged: the scope's
// closing reap runs after its lease's drop.
// Engagement: both stages return OK and place the table HOST; a reading too low for P3 is VOID.
// Mutant: lease-before-guard, under which the scope's lease is declared before its reap guard, so the pool parks after
// the closing reap and survives the call.
arm_result arm_all_host() {
    arm_result         r;
    lazy_inventory     inv("blk.1.host.weight", k_ne0_8k, k_host_table);
    const size_t       body  = mapped(lazy_table_host_body_request_bytes(0, inv.table.size));
    const uint64_t     role0 = role_bytes(r);
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return settle(r);
    }
    for (int pass = 0; pass < 2; ++pass) {
        const std::string which = pass == 0 ? "early" : "late";
        const uint64_t    g0    = registry().generation;
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv.inv(), pass == 0);
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            stage_failed(r, "the " + which + " stage", rc);
            (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
            return settle(r);
        }
        if (reading_too_low(r, body)) {
            (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
            return settle(r);
        }
        if (placed("blk.1.host.weight", "HOST") == 0 || placed("blk.1.host.weight", "STORAGE") != 0) {
            void_arm(r, "the " + which + " stage did not place the table HOST at this host's reading: " +
                            num(placed("blk.1.host.weight", "HOST")) + " HOST and " +
                            num(placed("blk.1.host.weight", "STORAGE")) + " STORAGE line(s)");
            (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
            return settle(r);
        }
        const lazy_registry_state s = registry();
        expect(r, s.generation - g0 == 1,
               "the " + which + " stage moved the registry generation by " + num(s.generation - g0) +
                   ", want 1: it took no provisional lease");
        expect(r, registry_is(s, 0, 0, 0), "after the " + which + " stage: " + describe(s) + ", want 0, 0, 0");
    }
    ggml_sycl_model_token token{};
    if (ggml_backend_sycl_model_load_end(txn, true, &token) == GGML_SYCL_LIFECYCLE_OK && token.model_id != 0) {
        (void) ggml_backend_sycl_model_unloaded_token(token);
    }
    return settle(r, role0);
}

// The two-device arm (§3.3: each record releases exactly its own reference, and a replacement hands the displaced lease
// back): on both cards, the early and the late stage, the load's end, the ensure with the late charge and the unload;
// after a FULL reap the pool's count is zero and the ring's LAZY_STAGING bytes are back at the pre-ensure figure.
// Engagement: the model loads and its ensure makes the ring.
// Mutant: replace-keeps-displaced, under which the late candidate's replacement keeps the early candidate's lease, so
// the pool and its ring outlive the unload.
arm_result arm_two_device() {
    arm_result     r;
    lazy_inventory inv("blk.1.pair.weight", k_ne0_8k, k_huge_table);
    const uint64_t role0 = role_bytes(r);
    model_load     m     = load_model(r, inv, "the model");
    if (!m.ok) {
        return settle(r);
    }
    const lazy_ring_ensure_status st = ensure_for(m);
    if (st != lazy_ring_ensure_status::OK) {
        void_arm(r, "the model's ensure returned " + name(st) + ", want ok");
        unload(m);
        return settle(r);
    }
    unload(m);
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    const lazy_registry_state s = registry();
    expect(r, registry_is(s, 0, 0, 0), "after the unload and the reap: " + describe(s) + ", want 0, 0, 0");
    expect(r, role_bytes(r) == role0,
           "after the unload and the reap the ring's LAZY_STAGING bytes are " + num(role_bytes(r)) +
               ", want the pre-ensure " + num(role0));
    return settle(r, role0);
}

// The partial-failure arm (§3.3: a throw after the first device leaves the staged candidate's reference to the
// rollback): the PRIVATE_TESTING hooks fail the stage after the first device. A failed early stage takes its lease (the
// generation moves by 1) and returns with no lease and no parked node; so does a failed late stage after an early one
// that succeeded, whose candidate the rollback drops too.
// Engagement: both hooks are exported, and each armed hook makes its stage return EFFECT_FAILED.
// Mutant: rollback-keeps-clone, under which the rollback leaves the candidate's lease alive.
arm_result arm_partial_failure() {
    arm_result         r;
    lazy_inventory     inv("blk.1.partial.weight", k_ne0_8k, k_huge_table);
    const fail_next_fn fail_early = proc<fail_next_fn>(k_fail_early_proc);
    const fail_next_fn fail_late  = proc<fail_next_fn>(k_fail_late_proc);
    if (fail_early == nullptr || fail_late == nullptr) {
        void_arm(r, "the stage's failure hooks are not exported");
        return settle(r);
    }
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return settle(r);
    }
    const uint64_t g0 = registry().generation;
    fail_early();
    ggml_sycl_lifecycle_result rc = stage(inv.inv(), true);
    lazy_registry_state        s  = registry();
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (rc != GGML_SYCL_LIFECYCLE_EFFECT_FAILED) {
        void_arm(r, "the armed early stage returned " + num(rc) + ", want EFFECT_FAILED");
        return settle(r);
    }
    expect(r, registry_is(s, 0, 0, 0), "after the failed early stage: " + describe(s) + ", want 0, 0, 0");
    expect(r, s.generation - g0 == 1,
           "the failed early stage moved the registry generation by " + num(s.generation - g0) + ", want 1");

    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the second load transaction did not begin");
        return settle(r);
    }
    log_clear();
    rc = stage(inv.inv(), true);
    if (rc != GGML_SYCL_LIFECYCLE_OK) {
        stage_failed(r, "the second load's early stage", rc);
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        return settle(r);
    }
    fail_late();
    rc = stage(inv.inv(), false);
    s  = registry();
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (rc != GGML_SYCL_LIFECYCLE_EFFECT_FAILED) {
        void_arm(r, "the armed late stage returned " + num(rc) + ", want EFFECT_FAILED");
        return settle(r);
    }
    expect(r, registry_is(s, 0, 0, 0), "after the failed late stage: " + describe(s) + ", want 0, 0, 0");
    return settle(r);
}

ggml_sycl_lifecycle_result dry_run(const ggml_sycl_tensor_inventory * inv) {
    ggml_sycl_plan_dry_run_result result{};
    result.struct_size = sizeof(result);
    result.version     = GGML_SYCL_PLAN_DRY_RUN_VERSION;
    return ggml_backend_sycl_plan_dry_run(inv, nullptr, &result);
}

// The dry-run arm (§3.3: the dry run and the load plan by one rule, and no parked node outlives the dry run): the dry
// run adopts the coverage wrapper, scored as §11 T3's wrapper arm by the take-time snapshot. A helper guard's pool is
// parked by the guard's own drop, whose FULL reap the hook holds after its count lands and before it swaps the list
// out; the fixture reads c0, the FULL reaps run. The dry run's entry reap is the one that frees the node, so its take
// snapshots c0 + 1 reaps run and no parked node, and its closing reap frees its own pool before it returns, c0 + 2 in
// all. Then, with model A loaded, ensured and unloaded, so its pool and ring are parked, the dry run's plan charges
// ring= the need's mapped size from a fresh pool, as the early stage of the same inventory does.
// Engagement: the guard holds a lease and its drop parks the pool held in the hook within the bound; the dry run
// returns OK and installs one pool, so a dry run that takes no lease fails before the snapshot is read; A's unload
// leaves its pool parked.
// Mutants: no-entry-reap (the take snapshots c0 reaps run) and no-closing-reap (the dry run's pool survives it parked).
arm_result arm_dry_run() {
    arm_result     r;
    lazy_inventory inv("blk.1.dry.weight", k_ne0_8k, k_huge_table);
    reap_hold      hold;
    pin_ptr        guard = make_pin(0);
    if (!pin_engaged(r, guard, 0)) {
        guard.reset();
        return settle(r);
    }
    ggml_sycl_test_lazy_reap_hold(hold_reap, &hold);
    std::thread dropper([&guard] { guard.reset(); });
    bool        entered = false;
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        entered = hold.cv.wait_for(lock, k_hook_bound, [&hold] { return hold.entered; });
    }
    const lazy_registry_state held = registry();
    const uint64_t            c0   = held.full_reaps_run;
    const uint64_t            g0   = held.generation;
    if (!entered || !registry_is(held, 0, 0, 1)) {
        release(hold);
        dropper.join();
        ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
        void_arm(r, entered ?
                        "while the guard's reap is held: " + describe(held) + ", want 0, 0, 1" :
                        "the guard's drop never reached the hold hook within " + num(k_hook_bound.count()) + " s");
        return settle(r);
    }
    const ggml_sycl_lifecycle_result rc    = dry_run(inv.inv());
    const lazy_registry_state        after = registry();
    expect(r, rc == GGML_SYCL_LIFECYCLE_OK, "the dry run returned " + num(rc) + ", want OK");
    const bool took = after.generation - g0 == 1;
    expect(r, took,
           "the dry run moved the registry generation by " + num(after.generation - g0) + ", want 1: it took no lease");
    // The take-time snapshot is the dry run's own only when its take installed a pool; otherwise it is an earlier
    // take's, so a dry run that took no lease is never scored by it.
    if (took) {
        expect(r, after.reaps_run_at_last_take == c0 + 1,
               "the dry run's take snapshot " + num(after.reaps_run_at_last_take) + " reaps run, want " + num(c0 + 1) +
                   ": the entry reap did not run before the take");
        expect(r, after.parked_at_last_take == 0,
               "the dry run's take found " + num(after.parked_at_last_take) + " parked nodes, want 0");
    }
    expect(r, registry_is(after, 0, 0, 0), "after the dry run: " + describe(after) + ", want 0, 0, 0");
    release(hold);
    dropper.join();
    ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
    const lazy_registry_state done = registry();
    expect(r, done.full_reaps_run == c0 + 2,
           "after the dry run " + num(done.full_reaps_run) + " FULL reaps have run, want " + num(c0 + 2) +
               " (the held reap, the entry reap and the closing reap)");

    model_load a = load_model(r, inv, "model A");
    if (!a.ok) {
        return settle(r);
    }
    const lazy_ring_ensure_status st = ensure_for(a);
    unload(a);
    const lazy_registry_state parked = registry();
    if (st != lazy_ring_ensure_status::OK || !registry_is(parked, 0, 0, 1)) {
        void_arm(r, "model A's ensure returned " + name(st) + " and its unload left " + describe(parked) +
                        ", want ok and 0, 0, 1");
        return settle(r);
    }
    log_clear();
    const ggml_sycl_lifecycle_result rc2      = dry_run(inv.inv());
    const size_t                     dry_ring = last_ring();
    const lazy_registry_state        taken    = registry();
    expect(r, rc2 == GGML_SYCL_LIFECYCLE_OK, "the second dry run returned " + num(rc2) + ", want OK");
    expect(r, dry_ring == mapped(k_need_8k) && taken.parked_at_last_take == 0,
           "the dry run over A's parked pool charged ring=" + num(dry_ring) + " with " +
               num(taken.parked_at_last_take) + " parked node(s) at its take, want " + num(mapped(k_need_8k)) +
               " and 0");
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the stage's load transaction did not begin");
        return settle(r);
    }
    log_clear();
    const ggml_sycl_lifecycle_result rc3 = stage(inv.inv(), true);
    if (rc3 != GGML_SYCL_LIFECYCLE_OK) {
        stage_failed(r, "the early stage", rc3);
    }
    expect(r, last_ring() == dry_ring,
           "the early stage charged ring=" + num(last_ring()) + ", the dry run ring=" + num(dry_ring));
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    return settle(r);
}

// The empty-slot arm (§3.3's channel: a plan made outside a stage carries a null pin): the public
// compute_placement_plan_early and the legacy setter, with no load open, plan on an empty slot: the ~1 TiB table cannot
// be HOST and has no pin, so each plan refuses as lazy-table-ring-unpinned, and the unconditional coverage-read counter
// does not move. ggml_sycl_publish_test_plan is a static of ggml-sycl.cpp with no export, so its staging is not reached
// from here.
// Engagement (positive control): the stage run of the same inventory reads at least one coverage read.
// Mutant: count-without-pin, under which a coverage read with no pin is counted.
arm_result arm_empty_slot() {
    arm_result       r;
    lazy_inventory   inv("blk.1.slot.weight", k_ne0_8k, k_huge_table);
    const u64_reader reads = proc<u64_reader>(k_coverage_reads_proc);
    if (reads == nullptr) {
        fail(r, std::string("the coverage-read accessor ") + k_coverage_reads_proc + " is not exported");
        return settle(r);
    }
    uint64_t c0 = 0, c1 = 0, c2 = 0;
    (void) reads(&c0);
    log_clear();
    ggml_backend_sycl_compute_placement_plan_early(g_backend0, inv.inv());
    const size_t early_unpinned = log_count_with("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-unpinned");
    log_clear();
    ggml_backend_sycl_set_tensor_inventory(g_backend0, inv.inv());
    const size_t setter_unpinned = log_count_with("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-unpinned");
    (void) reads(&c1);

    ggml_sycl_load_txn txn{};
    bool               counted = false;
    if (ggml_backend_sycl_model_load_begin(&txn) == GGML_SYCL_LIFECYCLE_OK) {
        (void) stage(inv.inv(), true);
        (void) reads(&c2);
        counted = c2 > c1;
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    }
    expect(r, c1 == c0, "the public entries counted " + num(c1 - c0) + " coverage reads, want 0");
    expect(r, early_unpinned >= 1 && setter_unpinned >= 1,
           "the early entry and the setter printed " + num(early_unpinned) + " and " + num(setter_unpinned) +
               " plan lines with refusal=lazy-table-ring-unpinned, want at least 1 each");
    if (!counted) {
        void_arm(r, "the stage counted no coverage read (" + num(c1) + " before, " + num(c2) +
                        " after), so the counter cannot tell");
    }
    return settle(r);
}

// The owner-check gate (§3.3, §4.4): the block ring's POOL rows, owned by {IO_POOL, the pool's generation}, are silent
// at the owner check while that generation is live in the registry's slot, and while its pool is parked with its reap
// held, and there are none once the FULL reap frees it.
// Engagement (positive control): a POOL row owned by a generation the registry never held is reported, once.
// Mutant: generation-live-ignores-parked, under which the predicate reads only the slot, so the parked pool's ring row
// is reported.
arm_result arm_owner_check() {
    arm_result r;
    {
        const lazy_registry_state            s     = registry();
        const uint64_t                       ghost = s.generation + 1000;
        std::unique_ptr<lazy_io_host_memory> row   = lazy_io_allocate_host(64 * 1024, 0, ghost, 0);
        if (row == nullptr) {
            void_arm(r, "the positive control's POOL row was not allocated");
            return settle(r);
        }
        const size_t reported =
            unified_cache_host_zone_owner_check(alloc_lifetime::POOL, { alloc_owner_kind::IO_POOL, ghost });
        row.reset();
        if (reported != 1) {
            void_arm(r, "the owner check reports " + num(reported) + " row(s) for generation " + num(ghost) +
                            ", which the registry never held, want 1");
            return settle(r);
        }
    }
    pin_ptr pin = make_pin(k_need_8k);
    if (!pin_engaged(r, pin, k_need_8k)) {
        pin.reset();
        return settle(r);
    }
    const uint64_t gen  = registry().live_generation;
    const size_t   live = unified_cache_host_zone_owner_check(alloc_lifetime::POOL, { alloc_owner_kind::IO_POOL, gen });
    expect(r, live == 0, "the live generation " + num(gen) + " reports " + num(live) + " ring row(s), want 0");

    reap_hold hold;
    ggml_sycl_test_lazy_reap_hold(hold_reap, &hold);
    std::thread dropper([&pin] { pin.reset(); });
    bool        entered = false;
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        entered = hold.cv.wait_for(lock, k_hook_bound, [&hold] { return hold.entered; });
    }
    const lazy_registry_state held = registry();
    if (!entered || !registry_is(held, 0, 0, 1)) {
        release(hold);
        dropper.join();
        ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
        void_arm(r, entered ? "while the pin's reap is held: " + describe(held) + ", want 0, 0, 1" :
                              "the pin's drop never reached the hold hook within " + num(k_hook_bound.count()) + " s");
        return settle(r);
    }
    const size_t parked = unified_cache_host_zone_owner_check(alloc_lifetime::POOL, { alloc_owner_kind::IO_POOL, gen });
    expect(r, parked == 0, "the parked generation " + num(gen) + " reports " + num(parked) + " ring row(s), want 0");
    release(hold);
    dropper.join();
    ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
    const size_t gone = unified_cache_host_zone_owner_check(alloc_lifetime::POOL, { alloc_owner_kind::IO_POOL, gen });
    expect(r, gone == 0, "after the reap generation " + num(gen) + " reports " + num(gone) + " ring row(s), want 0");
    return settle(r);
}

// --- the strict arm ---

// The strict table's descriptor (see k_strict_rows): its plan's need is one block per slot short of its span's.
lazy_table_desc strict_desc() {
    lazy_table_desc d;
    d.name                 = "strict.ring";
    d.identity.valid       = true;
    d.identity.model_id    = k_strict_model_id;
    d.identity.load_txn_id = k_strict_txn;
    d.identity.type        = GGML_TYPE_Q8_0;
    d.model.model_id       = k_strict_model_id;
    d.model.load_txn_id    = k_strict_txn;
    d.type                 = GGML_TYPE_Q8_0;
    d.ne0                  = k_ne0_8k;
    d.n_rows               = k_strict_rows;
    d.row_size             = k_row_8k;
    d.file_index           = 0;
    d.file_offset          = 0;
    d.table_bytes          = k_strict_file_bytes;
    d.file_path            = g_fixture_path;
    d.residency            = lazy_residency::STORAGE;
    d.consumer_layer       = 1;
    d.consumer_device      = 0;
    // The production chunk holds the whole slab, so the cache is the slab and its index, each at its mapping (§3.3).
    d.host_resident_bytes  = mapped(k_strict_file_bytes) + mapped(lazy_row_cache_index_raw_bytes(k_strict_rows));
    d.cache_slots          = k_strict_rows;
    d.cache_floor_slots    = k_strict_floor;
    d.block_ring_need      = k_strict_need;
    d.block_ring_charge    = mapped(k_strict_need);
    return d;
}

bool write_fixture(const std::string & path, size_t bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        return false;
    }
    std::vector<uint8_t> data(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<uint8_t>(i * 131u + 7u);
    }
    const bool ok = ::write(fd, data.data(), bytes) == static_cast<ssize_t>(bytes) && ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

// Whether the fixture's filesystem takes O_DIRECT, which a STORAGE table's descriptor requires and tmpfs refuses.
bool probe_direct(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    ::close(fd);
    return true;
}

// A strict child's first STRICT-CHILD line, which it prints when its materialization returns or when it cannot
// engage, or what stands for it.
std::string strict_child_line(const child_run & run) {
    const size_t at = run.err.find("STRICT-CHILD ");
    if (at == std::string::npos) {
        return "it printed no STRICT-CHILD line";
    }
    const size_t end = run.err.find('\n', at);
    return run.err.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

// The figures a strict case's one [HOST-PLAN-BUG] line names.
std::string strict_bug(strict_case kind) {
    switch (kind) {
        case strict_case::TOO_NARROW:
            return "need " + num(k_strict_need) + " B, want " + num(k_need_8k) + " B";
        case strict_case::MISMATCH:
            return "ring charge " + num(mapped(k_need_8k)) + " B, want 0 B";
        case strict_case::CONCURRENT:
            return "concurrent ring ensure: need " + num(k_need_8k) + " B, span " + num(k_span_8k) + " B";
    }
    return "an unknown strict case";
}

// Scores a strict case's child (§10): it aborted (SIGABRT) at the materialization's strict abort, whose line names the
// table and the case's refusal, after the ensure's one [HOST-PLAN-BUG] line naming the case's figures, and it never
// returned. Ensure itself never aborts, so the abort is the materialization's. Returns false when the arm is VOID or
// the child did not finish, so the arm stops there.
bool strict_child_scored(arm_result & r, const strict_case_entry & c) {
    if (g_direct_unsupported) {
        void_arm(r, "the fixture's filesystem refuses O_DIRECT, which the STORAGE descriptor requires");
        return false;
    }
    const child_run & run = g_strict_runs[static_cast<size_t>(c.kind)];
    if (!run.ran) {
        fail(r, "the strict child did not finish within its bound");
        return false;
    }
    if (WIFEXITED(run.status) && WEXITSTATUS(run.status) == SYCL_TEST_SKIP) {
        void_arm(r, "the strict child did not engage: " + strict_child_line(run));
        return false;
    }
    const bool aborted = WIFSIGNALED(run.status) && WTERMSIG(run.status) == SIGABRT;
    if (!aborted) {
        fail(r, "the strict child did not abort: status " + std::to_string(run.status) + ", " + strict_child_line(run));
    }
    const std::string numbers = strict_bug(c.kind);
    const size_t      bugs    = text_count_with(run.err, "[HOST-PLAN-BUG]", numbers.c_str());
    expect(r, bugs == 1,
           "the strict child printed " + num(bugs) + " [HOST-PLAN-BUG] lines naming '" + numbers + "', want 1");
    const std::string abort_line =
        std::string("lazy table strict.ring: ") + lazy_table_status_name(c.status) + " under " + k_strict_setting;
    const size_t aborts = text_count_with(run.err, abort_line.c_str());
    expect(r, aborts == 1, "the strict child printed " + num(aborts) + " lines '" + abort_line + "', want 1");
    expect(r, text_count_with(run.err, "STRICT-CHILD returned") == 0, "the strict child returned");
    return true;
}

// The strict arm, its too-narrow case (§10: a [HOST-PLAN-BUG] aborts under strict leases): run as a child under
// GGML_SYCL_STRICT_LEASES=1 before this process initialises SYCL, the materialization of a STORAGE table whose plan's
// need is one block per slot short of its span's aborts at the ensure's refusal, after the ensure's one [HOST-PLAN-BUG]
// line naming 262144 and 524288 (exit 134, SIGABRT); ensure itself never aborts. The same materialization in this
// process, not strict, returns lazy-table-ring-too-narrow.
// Engagement: the child ran within its bound on a filesystem that takes O_DIRECT.
// Mutant: strict-abort-dropped, under which the materialization returns the refusal under strict leases too.
arm_result arm_strict() {
    arm_result r;
    if (!strict_child_scored(r, k_strict_cases[static_cast<size_t>(strict_case::TOO_NARROW)])) {
        return settle(r);
    }

    pin_ptr pin = make_pin(0);
    if (!pin_engaged(r, pin, 0)) {
        pin.reset();
        return settle(r);
    }
    {
        const lazy_table_result t = lazy_table_create_storage(strict_desc());
        expect(r, t.status == lazy_table_status::RING_TOO_NARROW,
               std::string("the non-strict materialization returned ") + lazy_table_status_name(t.status) +
                   ", want lazy-table-ring-too-narrow");
    }
    pin.reset();
    return settle(r);
}

// The strict arm's mismatch case (§10): run as a child under GGML_SYCL_STRICT_LEASES=1, the materialization of a
// STORAGE table whose plan charges a live ring it did not create aborts at the ensure's refusal, after the ensure's
// one [HOST-PLAN-BUG] line naming the plan's charge and 0.
// Engagement: the child ran within its bound on a filesystem that takes O_DIRECT, and its backend 0 initialised.
// Mutant: strict-abort-dropped-mismatch, the strict-abort-dropped selector run against this arm.
arm_result arm_strict_mismatch() {
    arm_result r;
    (void) strict_child_scored(r, k_strict_cases[static_cast<size_t>(strict_case::MISMATCH)]);
    return settle(r);
}

// The strict arm's concurrent case (§10): run as a child under GGML_SYCL_STRICT_LEASES=1, the materialization of a
// STORAGE table whose ensure finds another caller reserved aborts at the ensure's refusal, after the ensure's one
// "[HOST-PLAN-BUG] concurrent ring ensure" line naming the need and the span.
// Engagement: the child ran within its bound on a filesystem that takes O_DIRECT, its backend 0 initialised, and the
// ensure hook held the other caller.
// Mutant: strict-abort-dropped-concurrent, the strict-abort-dropped selector run against this arm.
arm_result arm_strict_concurrent() {
    arm_result r;
    (void) strict_child_scored(r, k_strict_cases[static_cast<size_t>(strict_case::CONCURRENT)]);
    return settle(r);
}

// --- the mutants ---

struct mutant_entry {
    const char *     text;
    lazy_ring_mutant mutant;
    const char *     arm;
};

const mutant_entry k_mutants[] = {
    { "no-span-check",                   lazy_ring_mutant::NO_SPAN_CHECK,                  k_arm_ring_plan         },
    { "ignore-live-ring",                lazy_ring_mutant::IGNORE_LIVE_RING,               k_arm_wider_refused     },
    { "ignore-live-ring-p3",             lazy_ring_mutant::IGNORE_LIVE_RING,               k_arm_wider_p3          },
    { "no-candidate-lease",              lazy_ring_mutant::NO_CANDIDATE_LEASE,             k_arm_lease             },
    { "lease-at-stage",                  lazy_ring_mutant::LEASE_AT_STAGE,                 k_arm_window            },
    { "lease-presence-as-coverage",      lazy_ring_mutant::LEASE_PRESENCE_AS_COVERAGE,     k_arm_staging           },
    { "take-without-candidate",          lazy_ring_mutant::TAKE_WITHOUT_CANDIDATE,         k_arm_no_storage        },
    { "lease-before-guard",              lazy_ring_mutant::LEASE_BEFORE_GUARD,             k_arm_all_host          },
    { "replace-keeps-displaced",         lazy_ring_mutant::REPLACE_KEEPS_DISPLACED,        k_arm_two_device        },
    { "rollback-keeps-clone",            lazy_ring_mutant::ROLLBACK_KEEPS_CLONE,           k_arm_partial           },
    { "count-without-pin",               lazy_ring_mutant::COUNT_WITHOUT_PIN,              k_arm_empty_slot        },
    { "no-entry-reap",                   lazy_ring_mutant::NO_ENTRY_REAP,                  k_arm_dry_run           },
    { "no-closing-reap",                 lazy_ring_mutant::NO_CLOSING_REAP,                k_arm_dry_run           },
    { "generation-live-ignores-parked",  lazy_ring_mutant::GENERATION_LIVE_IGNORES_PARKED, k_arm_owner_check       },
    { "strict-abort-dropped",            lazy_ring_mutant::STRICT_ABORT_DROPPED,           k_arm_strict            },
    { "strict-abort-dropped-mismatch",   lazy_ring_mutant::STRICT_ABORT_DROPPED,           k_arm_strict_mismatch   },
    { "strict-abort-dropped-concurrent", lazy_ring_mutant::STRICT_ABORT_DROPPED,           k_arm_strict_concurrent },
    { "decline-every-plan",              lazy_ring_mutant::DECLINE_EVERY_PLAN,             k_arm_decline_once      },
};

const mutant_entry * mutant_named(const char * text) {
    for (const mutant_entry & e : k_mutants) {
        if (std::strcmp(e.text, text) == 0) {
            return &e;
        }
    }
    return nullptr;
}

const strict_case_entry * strict_case_named(const char * name) {
    for (const strict_case_entry & c : k_strict_cases) {
        if (std::strcmp(c.name, name) == 0) {
            return &c;
        }
    }
    return nullptr;
}

// The strict child's way out once its materialization returns or it cannot engage: its STRICT-CHILD line, then
// _Exit, the process end the abort takes, without its signal. Nothing is torn down, so no teardown under strict leases
// can stand in for the abort the arm scores.
[[noreturn]] void strict_child_exit(const std::string & line, int code) {
    std::fprintf(stderr, "STRICT-CHILD %s\n", line.c_str());
    std::fflush(stderr);
    std::_Exit(code);
}

// The strict arms' child: under GGML_SYCL_STRICT_LEASES=1, one case's STORAGE materialization refuses at its ensure
// and aborts there. The full-need cases first initialise backend 0, as main does before any arm allocates a ring.
int strict_child(const char * which, const char * mutant) {
    const strict_case_entry * c = strict_case_named(which);
    if (c == nullptr) {
        std::fprintf(stderr, "STRICT-CHILD unknown case %s\n", which);
        return 2;
    }
    if (mutant != nullptr) {
        const mutant_entry * e = mutant_named(mutant);
        if (e == nullptr) {
            std::fprintf(stderr, "STRICT-CHILD unknown mutant %s\n", mutant);
            return 2;
        }
        ggml_sycl_test_lazy_ring_mutant_select(e->mutant);
    }
    ggml_sycl_test_host_ledger_store_budget(k_strict_budget);
    lazy_table_desc desc = strict_desc();
    if (c->kind != strict_case::TOO_NARROW) {
        if (ggml_backend_sycl_init(0) == nullptr) {
            strict_child_exit("void: backend 0 did not initialise", SYCL_TEST_SKIP);
        }
        desc.block_ring_need   = k_need_8k;
        desc.block_ring_charge = mapped(k_need_8k);
    }
    pin_ptr pin = make_pin(c->kind == strict_case::MISMATCH ? k_need_8k : 0);
    gate    reserve_gate;
    if (c->kind == strict_case::CONCURRENT) {
        // The other caller: its ensure reserves the ring-less pool and the hook holds it there until the process ends.
        ggml_sycl_test_lazy_ensure_hold([](void * ctx) { static_cast<gate *>(ctx)->hold(); }, &reserve_gate);
        std::thread([] {
            (void) block_ring_ensure(k_need_8k, k_span_8k, 0, mapped(k_need_8k), k_strict_txn);
        }).detach();
        if (!reserve_gate.wait_holders(1)) {
            strict_child_exit(
                "void: the ensure hook did not hold the other caller within " + num(k_hook_bound.count()) + " s",
                SYCL_TEST_SKIP);
        }
    }
    const lazy_table_status status = lazy_table_create_storage(desc).status;
    strict_child_exit(std::string("returned status=") + lazy_table_status_name(status),
                      status == lazy_table_status::OPEN_FAILED ? SYCL_TEST_SKIP : 0);
}

child_run spawn_strict(const char * which, const char * mutant) {
    std::vector<std::string> args = { "test-sycl-lazy-ring", "--strict-child", g_fixture_path, which };
    if (mutant != nullptr) {
        args.emplace_back(mutant);
    }
    return run_child_strings(args, child_environment({ k_strict_setting, "GGML_NO_BACKTRACE=1" }),
                             std::chrono::milliseconds(120000));
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--strict-child") == 0) {
        g_fixture_path = argv[2];
        return strict_child(argv[3], argc >= 5 ? argv[4] : nullptr);
    }
    const mutant_entry * mutant = nullptr;
    if (argc == 3 && std::strcmp(argv[1], "--mutant") == 0) {
        mutant = mutant_named(argv[2]);
        if (mutant == nullptr) {
            std::printf("FAIL: unknown mutant %s\n", argv[2]);
            return 1;
        }
    } else if (argc != 1) {
        std::printf("FAIL: unknown arguments; the option is --mutant <name>\n");
        return 1;
    }
    if (ggml_row_size(GGML_TYPE_Q8_0, k_ne0_8k) != k_row_8k || ggml_row_size(GGML_TYPE_Q8_0, k_ne0_16k) != k_row_16k ||
        ggml_row_size(GGML_TYPE_Q8_0, k_ne0_32k) != k_row_32k) {
        std::printf("FAIL: the Q8_0 row sizes are not the geometry's 34, 8194 and 24582 B\n");
        return 1;
    }

    // The owner check and the refusal arms read typed results, which strict mode turns into aborts; the strict
    // children set it alone.
    const std::string strict_name(k_strict_setting, std::strchr(k_strict_setting, '=') - k_strict_setting);
    unsetenv(strict_name.c_str());
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        std::printf("FAIL: getcwd: %s\n", std::strerror(errno));
        return 1;
    }
    g_fixture_path = std::string(cwd) + "/test-sycl-lazy-ring-" + std::to_string(getpid()) + ".bin";
    if (!write_fixture(g_fixture_path, k_strict_file_bytes)) {
        std::printf("FAIL: the fixture file %s was not written: %s\n", g_fixture_path.c_str(), std::strerror(errno));
        ::unlink(g_fixture_path.c_str());
        return 1;
    }
    g_direct_unsupported = !probe_direct(g_fixture_path);

    // The strict children start before this process initialises SYCL, one at a time: every case's without a mutant,
    // else the one the mutant's arm scores.
    for (const strict_case_entry & c : k_strict_cases) {
        if (g_direct_unsupported || (mutant != nullptr && std::strcmp(mutant->arm, c.arm) != 0)) {
            continue;
        }
        child_run & run = g_strict_runs[static_cast<size_t>(c.kind)];
        run             = spawn_strict(c.name, mutant != nullptr ? mutant->text : nullptr);
        std::fputs(run.err.c_str(), stderr);
    }

    int rc = SYCL_TEST_SKIP;
    if (!sycl_test_require_gpu("the lazy ring arms")) {
        ::unlink(g_fixture_path.c_str());
        return rc;
    }
    // The stage plans through the multi-device planner call iff this predicate holds, which the two-device and the
    // direct planner arms score; a host on the other path would score the wrong call.
    if (!ggml_backend_sycl_moe_multi_gpu_requested()) {
        std::printf(
            "SKIP: the planner caller takes its single-device path: ggml_backend_sycl_moe_multi_gpu_requested() "
            "is false\n");
        ::unlink(g_fixture_path.c_str());
        return rc;
    }
    ggml_log_set(capture_log, nullptr);

    // The backends the scheduler is shown live for the whole run, as at a load; the stage reaches the second card's
    // cache itself on the multi-device path.
    std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> backends;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        backends.emplace_back(ggml_backend_sycl_init(d), ggml_backend_free);
        if (!backends.back()) {
            std::printf("FAIL: backend %d did not initialise\n", d);
            ::unlink(g_fixture_path.c_str());
            return 1;
        }
    }
    g_backend0 = backends.front().get();
    // The ring's LAZY_STAGING reading needs device 0's cache, and neither the direct planner arms nor the ring's
    // HOST_CPU allocation builds one; a load's stage builds it before any ensure, so it is built here before any arm.
    if (get_unified_cache_for_device(0) == nullptr) {
        std::printf("FAIL: device 0's cache was not created, so every LAZY_STAGING reading would be VOID\n");
        ::unlink(g_fixture_path.c_str());
        return 1;
    }

    const std::vector<named_arm> arms = {
        { k_arm_ring_plan,         arm_ring_plan         },
        { k_arm_wider_host,        arm_wider_host        },
        { k_arm_wider_refused,     arm_wider_refused     },
        { k_arm_wider_p3,          arm_wider_p3          },
        { k_arm_decline_once,      arm_decline_once      },
        { k_arm_staging,           arm_staging_pass      },
        { k_arm_lease,             arm_lease             },
        { k_arm_window,            arm_window            },
        { k_arm_no_storage,        arm_no_storage        },
        { k_arm_all_host,          arm_all_host          },
        { k_arm_two_device,        arm_two_device        },
        { k_arm_partial,           arm_partial_failure   },
        { k_arm_dry_run,           arm_dry_run           },
        // After the dry run: the legacy setter leaves a published legacy plan behind.
        { k_arm_empty_slot,        arm_empty_slot        },
        { k_arm_owner_check,       arm_owner_check       },
        { k_arm_strict,            arm_strict            },
        { k_arm_strict_mismatch,   arm_strict_mismatch   },
        { k_arm_strict_concurrent, arm_strict_concurrent },
    };
    if (mutant == nullptr) {
        rc = run_arms(arms, "lazy ring arms");
    } else {
        ggml_sycl_test_lazy_ring_mutant_select(mutant->mutant);
        std::vector<named_arm> target;
        for (const named_arm & arm : arms) {
            if (std::strcmp(arm.name, mutant->arm) == 0) {
                target.push_back(arm);
            }
        }
        rc = run_arms(target, std::string("lazy ring arms (--mutant ") + mutant->text + ")");
    }
    ::unlink(g_fixture_path.c_str());
    return rc;
}
