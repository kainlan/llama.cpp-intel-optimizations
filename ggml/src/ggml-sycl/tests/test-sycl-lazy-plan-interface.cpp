//
// The lazy tables' backend-level interface (th32 §11 T3): a lazy table enters through the inventory libllama hands the
// backend, the plan the load stages and publishes carries its residency, and the whole of it is read back through the
// model token. No libllama is involved: the fixture builds the inventory and drives the load transaction itself.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The stage initialises the SYCL backends, constructs their caches and plans through them, so the binary is
// registered gpu-serial, serial, with the selector pinned: test-sycl-lazy-plan-interface-1dev runs on the B70 alone,
// where the planner caller takes its single-device path, and test-sycl-lazy-plan-interface-2dev on both cards, where it
// takes the multi-device path. Each runs iff the host takes its path, by the planner caller's own predicate,
// ggml_backend_sycl_moe_multi_gpu_requested(), and exits 77 otherwise. The scheduler-exposed device count is never the
// test: this backend exposes one device whenever more than one GPU is visible.
//
// Every arm names its engagement check and the mutant it kills. An arm whose engagement check fails is VOID, and a VOID
// arm fails the binary like a FAIL. Every arm runs, so one failure does not hide the rest.

#include "../host-cpu-mapping.hpp"
#include "../lazy-io-pool.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-plan-shaped-inventory.hpp"
#include "lazy-table-test-util.hpp"
#include "sycl-test-skip.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace host_ledger_test_harness;
using namespace lazy_table_test;
using ggml_sycl::host_cpu_mapped_size;
using ggml_sycl::lazy_table_host_body_request_bytes;
using host_plan_shaped_inventory::loader_inventory;

namespace {

int            g_devices  = 1;
ggml_backend_t g_backend0 = nullptr;  // main's backend 0, alive for every arm

// A Q8_0 lazy table of 32-element rows (34 bytes each), read one row per token by layer 1.
ggml_sycl_lazy_table_info lazy_table(const char * name, int64_t rows) {
    ggml_sycl_lazy_table_info t{};
    t.name           = name;
    t.type           = GGML_TYPE_Q8_0;
    t.rows_per_token = 1;
    t.ne[0]          = 32;
    t.ne[1]          = rows;
    t.ne[2]          = 1;
    t.ne[3]          = 1;
    t.row_size       = ggml_row_size(GGML_TYPE_Q8_0, 32);
    t.consumer_layer = 1;
    t.size           = t.row_size * static_cast<size_t>(rows);
    return t;
}

struct lazy_reading {
    bool     ok       = false;
    uint64_t count    = 0;
    uint64_t resident = 0;
};

// The plan published for `model`, read through the export.
lazy_reading read_published(ggml_sycl_model_token model) {
    uint64_t     out[2] = { 0, 0 };
    lazy_reading r;
    r.ok       = ggml_sycl_plan_lazy_tables_for_testing(model, out);
    r.count    = out[0];
    r.resident = out[1];
    return r;
}

// The candidate staged for `txn`: a token whose model_id is 0 names the transaction's candidate.
lazy_reading read_candidate(ggml_sycl_load_txn txn) {
    ggml_sycl_model_token token{};
    token.load_txn_id = txn.id;
    return read_published(token);
}

void expect_reading(arm_result & r, const lazy_reading & got, uint64_t resident, const std::string & at) {
    if (!got.ok) {
        fail(r, at + ": no plan to read");
        return;
    }
    expect(r, got.count == 1, at + ": the plan carries " + num(got.count) + " lazy tables, want 1");
    expect(r, got.resident == resident,
           at + ": the plan's resident bytes are " + num(got.resident) + ", want " + num(resident));
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

// The last [HOST-PLAN] lazy line's budget cannot hold `bytes` beside its zones, reserve and envelope, the fit-room term
// of the planner's HOST test (budget - (zones + reserve + envelope) >= the table's body): the arm is VOID, since the
// budget, not the plan, decided.
bool fit_room_too_low(arm_result & r, size_t bytes) {
    const std::vector<std::string> lines = log_lines_containing("[HOST-PLAN] lazy ");
    if (lines.empty()) {
        return false;
    }
    const size_t budget   = printed(lines.back(), "budget");
    const size_t zones    = printed(lines.back(), "zones");
    const size_t reserve  = printed(lines.back(), "reserve");
    const size_t envelope = printed(lines.back(), "envelope");
    if (budget == SIZE_MAX || zones == SIZE_MAX || reserve == SIZE_MAX || envelope == SIZE_MAX ||
        budget >= zones + reserve + envelope + bytes) {
        return false;
    }
    void_arm(r, "the load's host budget " + num(budget) + " cannot hold " + num(bytes) + " bytes beside zones of " +
                    num(zones) + ", a reserve of " + num(reserve) + " and an envelope of " + num(envelope));
    return true;
}

// (a) Through the stage: a 64 MiB table in the inventory is HOST in the early candidate, the late candidate and the
// published plan, each holding the table's body, and the published plan leaves with the model. The table sits at
// 4032, the unaligned offset of DeepSeek's blk.14.engram_embd, where the body's O_DIRECT window crosses one more huge
// page than the table alone (66 MiB against 64 MiB), so the plan's term is the materialization's rule (§3.3) and
// not the table's own size.
// Engagement: no candidate exists before the first stage (the reader answers VOID, not 0), both stages and the load's
// end succeed, and the published plan is gone after the unload.
// Mutants: the inventory's tables not copied into the planner's input (count 0); the late stage or the publish
// dropping the tables; the reader answering a stale or zero plan.
arm_result arm_a_through_the_stage() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.1.engram_embd.weight", (64 * MiB) / 34);
    table.file_offset               = 4032;
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;
    const size_t resident = host_cpu_mapped_size(lazy_table_host_body_request_bytes(table.file_offset, table.size));

    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    const lazy_reading before = read_candidate(txn);
    expect(arr, !before.ok, "the reader answered a plan before any stage");

    log_clear();
    const ggml_sycl_lifecycle_result early_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    if (early_rc != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the early stage returned " + num(early_rc));
        return arr;
    }
    if (reading_too_low(arr, resident)) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        return arr;
    }
    expect_reading(arr, read_candidate(txn), resident, "early candidate");

    const ggml_sycl_lifecycle_result late_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, false);
    if (late_rc != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the late stage returned " + num(late_rc));
        return arr;
    }
    expect_reading(arr, read_candidate(txn), resident, "late candidate");

    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_end(txn, true, &model) != GGML_SYCL_LIFECYCLE_OK || model.model_id == 0) {
        void_arm(arr, "the load did not end with a model token");
        return arr;
    }
    expect_reading(arr, read_published(model), resident, "published plan");
    (void) ggml_backend_sycl_model_unloaded_token(model);
    expect(arr, !read_published(model).ok, "the published plan is still readable after the unload");
    return arr;
}

// (b) The planner caller reads the thread's plan slot: a direct early entry with a fixture's pin on the slot plans a
// ~1 TiB table STORAGE, which only a pinned plan can do, and the candidate carries it at its floor and yieldable
// share, the figure the plan's [PLACEMENT] line prints.
// Engagement: the pin holds a lease, read from the registry (its live leases rise across coverage_pin_make, since the
// helper returns a guard even when its take came back empty), and the staged-load seam made the direct entry stage a
// candidate for the transaction. Without the lease check a pool that built nothing would read as a dropped forward.
// Mutants: the slot not read at either planner call of compute_and_store_plan_for_inventory (one per path: the
// single-device call on the -1dev registration, the multi-device one on -2dev), or not forwarded by the one-device
// delegation or a pair overload: the planner gets no pin, the table cannot be HOST (it exceeds the budget) and is not
// STORAGE-eligible, the plan refuses as lazy-table-ring-unpinned and the candidate carries no table.
arm_result arm_b_the_slot_reaches_the_planner() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.14.engram_embd.weight", static_cast<int64_t>((1024 * GiB) / 34));
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;
    const size_t floor_raw          = 2ull * inv.inv.n_ubatch * table.rows_per_token * table.row_size;

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    const ggml_sycl::lazy_registry_state before = registry();
    void *                               pin    = ggml_sycl_test_coverage_pin_make(0);
    const ggml_sycl::lazy_registry_state after  = registry();
    if (pin == nullptr || after.live_leases <= before.live_leases) {
        ggml_sycl_test_coverage_pin_release(pin);
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the pin holds no lease: the registry's live leases went from " + num(before.live_leases) +
                          " to " + num(after.live_leases) + " across coverage_pin_make, so the arm cannot tell a " +
                          "dropped slot forward from a pool that was never built");
        return arr;
    }
    log_clear();
    void *         slot     = ggml_sycl_test_lazy_slot_begin(pin);
    const uint64_t previous = ggml_sycl_test_plan_effect_begin(txn.id);
    ggml_backend_sycl_compute_placement_plan_early(g_backend0, &inv.inv);
    ggml_sycl_test_plan_effect_end(previous);
    ggml_sycl_test_lazy_slot_end(slot);
    const lazy_reading got = read_candidate(txn);
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    ggml_sycl_test_coverage_pin_release(pin);

    if (!got.ok) {
        void_arm(arr, "the direct entry staged no candidate for the transaction");
        return arr;
    }
    if (got.count != 1) {
        const size_t refused = log_lines_containing("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-unpinned").size();
        fail(arr, "the candidate carries " + num(got.count) + " lazy tables, want 1 STORAGE table; the plan logged " +
                      "refusal=lazy-table-ring-unpinned " + num(refused) + " times");
        return arr;
    }
    const std::vector<std::string> placed = log_lines_containing("[PLACEMENT] lazy blk.14.engram_embd.weight ");
    expect(arr, !placed.empty() && placed.back().find(" residency=STORAGE ") != std::string::npos,
           "no [PLACEMENT] line places the table STORAGE");
    expect(arr, got.resident >= floor_raw,
           "the cache holds " + num(got.resident) + " bytes, below its floor's " + num(floor_raw) + " slab bytes");
    expect(arr, got.resident < host_cpu_mapped_size(lazy_table_host_body_request_bytes(table.file_offset, table.size)),
           "the cache holds " + num(got.resident) + " bytes, not less than the table's HOST body");
    if (!placed.empty()) {
        expect(arr, printed(placed.back(), "resident") == got.resident,
               "the candidate's resident bytes " + num(got.resident) + " are not the [PLACEMENT] line's " +
                   num(printed(placed.back(), "resident")));
    }
    return arr;
}

// The live ring the refusal arms hold: 256 KiB, so its slot spans 256 KiB / LAZY_BLOCK_RING_QD = 4096 B, narrower than
// the 8192 B read span of a table's 34 B rows (lazy_block_ring_span), which declines the table for STORAGE (§3.3).
constexpr uint64_t k_narrow_ring = 256 * 1024;

// The fixture's pin and the ring it reads, or null after a VOID naming why: a lease on a pool whose ring holds
// `ring_bytes`, read from the registry, since the helper returns a guard even when its take came back empty.
void * pin_with_ring(arm_result & arr, uint64_t ring_bytes) {
    const ggml_sycl::lazy_registry_state before = registry();
    void *                               pin    = ggml_sycl_test_coverage_pin_make(ring_bytes);
    const ggml_sycl::lazy_registry_state after  = registry();
    if (pin == nullptr || after.live_leases <= before.live_leases || after.ring_bytes != ring_bytes) {
        ggml_sycl_test_coverage_pin_release(pin);
        void_arm(arr, "the pin holds no lease on a ring of " + num(ring_bytes) + " B: the live leases went from " +
                          num(before.live_leases) + " to " + num(after.live_leases) + ", the ring reads " +
                          num(after.ring_bytes) + " B");
        return nullptr;
    }
    return pin;
}

// (c) A refused plan refuses the load at the stage (th32 §4.3 step 4): a fixture's pin holds a live ring of 256 KiB,
// the stage's own lease joins its pool, and the ring's 4096 B slot is narrower than the ~1 TiB table's 8192 B read
// span, so the planner declines the table for STORAGE; it cannot be HOST either (it exceeds the budget at any
// reading), so the plan refuses as lazy-table-ring-too-narrow, and the early stage returns PLAN_REJECTED with exactly
// one "[HOST-PLAN] load refused:" WARN naming the reason, and its rollback leaves no candidate for the transaction.
// Engagement: the pin holds a lease on a pool whose ring reads 256 KiB, and the planner's [HOST-PLAN] lazy line carries
// refusal=lazy-table-ring-too-narrow, so the stage was shown a refused plan.
// Mutants: the stage's refusal check dropped (the early stage returns OK and the load goes on with a plan that holds no
// table); the WARN printed at both checks (the count is not 1). A WARN printed once per device is not among them: both
// registrations expose one device to the backend (see the header), so it would still print once.
arm_result arm_c_the_stage_refuses_a_refused_plan() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.14.engram_embd.weight", static_cast<int64_t>((1024 * GiB) / 34));
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;

    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    void * pin = pin_with_ring(arr, k_narrow_ring);
    if (pin == nullptr) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        return arr;
    }
    log_clear();
    const ggml_sycl_lifecycle_result rc   = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    const lazy_reading               left = read_candidate(txn);
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    ggml_sycl_test_coverage_pin_release(pin);

    if (log_lines_containing("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-too-narrow").empty()) {
        void_arm(arr, "no plan line carries refusal=lazy-table-ring-too-narrow, so the stage saw no refused plan");
        return arr;
    }
    expect(arr, rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
           "the early stage returned " + num(rc) + ", want PLAN_REJECTED (" + num(GGML_SYCL_LIFECYCLE_PLAN_REJECTED) +
               ")");
    const size_t warned = log_lines_containing("[HOST-PLAN] load refused: lazy-table-ring-too-narrow ").size();
    expect(arr, warned == 1,
           "the stage logged " + num(warned) + " load-refused lines naming lazy-table-ring-too-narrow, want 1");
    expect(arr, !left.ok, "a candidate is still staged for the refused load");
    return arr;
}

// (d) The late pass refuses on its own check (th32 §4.3 step 4): a fixture's pin holds a ring-less pool, which the
// early stage's own lease joins, so the early plan holds the ~1 TiB table STORAGE and the stage returns OK; the
// fixture then ensures a 256 KiB ring on that pool, as an earlier model's materialization would have, and the late
// stage's lease reads a live ring whose 4096 B slot is narrower than the table's 8192 B span, so its plan refuses as
// lazy-table-ring-too-narrow, and the late stage returns PLAN_REJECTED with exactly one load-refused WARN, the late
// check's (early=0), and none from the early check.
// Engagement: the pin holds a lease, the early stage returns OK with its candidate carrying the table, the ensure
// installs the 256 KiB ring, and the late plan's [HOST-PLAN] lazy line carries refusal=lazy-table-ring-too-narrow.
// Mutant: the late pass's refusal check dropped (the late stage configures the refused plan's host zones and returns
// OK).
arm_result arm_d_the_late_pass_refuses() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.14.engram_embd.weight", static_cast<int64_t>((1024 * GiB) / 34));
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;

    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    void * pin = pin_with_ring(arr, 0);
    if (pin == nullptr) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        return arr;
    }
    log_clear();
    const ggml_sycl_lifecycle_result early_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    const lazy_reading               early    = read_candidate(txn);
    if (early_rc != GGML_SYCL_LIFECYCLE_OK || !early.ok || early.count != 1) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        ggml_sycl_test_coverage_pin_release(pin);
        void_arm(arr, "the early stage returned " + num(early_rc) + " with " +
                          (early.ok ? num(early.count) + " lazy tables" : std::string("no candidate")) +
                          ", want OK with 1");
        return arr;
    }
    const ggml_sycl::lazy_ring_ensure_status ensured = ggml_sycl::block_ring_ensure(
        k_narrow_ring, k_narrow_ring / ggml_sycl::LAZY_BLOCK_RING_QD, 0, host_cpu_mapped_size(k_narrow_ring), txn.id);
    if (ensured != ggml_sycl::lazy_ring_ensure_status::OK || registry().ring_bytes != k_narrow_ring) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        ggml_sycl_test_coverage_pin_release(pin);
        void_arm(arr, std::string("the fixture's ensure of a ") + num(k_narrow_ring) + " B ring returned " +
                          ggml_sycl::lazy_ring_ensure_status_name(ensured) + " and the ring reads " +
                          num(registry().ring_bytes) + " B");
        return arr;
    }
    const ggml_sycl_lifecycle_result late_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, false);
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    ggml_sycl_test_coverage_pin_release(pin);

    if (log_lines_containing("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-too-narrow").empty()) {
        void_arm(arr, "no plan line carries refusal=lazy-table-ring-too-narrow, so the late stage saw no refused plan");
        return arr;
    }
    expect(arr, late_rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
           "the late stage returned " + num(late_rc) + ", want PLAN_REJECTED (" +
               num(GGML_SYCL_LIFECYCLE_PLAN_REJECTED) + ")");
    const size_t late_warned =
        log_lines_containing("[HOST-PLAN] load refused: lazy-table-ring-too-narrow ", "(early=0)").size();
    const size_t early_warned = log_lines_containing("[HOST-PLAN] load refused: ", "(early=1)").size();
    expect(arr, late_warned == 1 && early_warned == 0,
           "the two stages logged " + num(late_warned) + " late and " + num(early_warned) +
               " early load-refused lines, want 1 late naming lazy-table-ring-too-narrow and no early one");
    return arr;
}

// (e) An inventory naming lazy tables with no array behind the count is refused at the stage as a null tensor array is,
// with NULL_OUTPUT, before anything reads it (th32 §4.1).
// Engagement: the same inventory with the array present, staged with no load open, is refused for the transaction
// (WRONG_TRANSACTION), so the stage's inventory check is what returns NULL_OUTPUT.
// Mutant: the stage's check not reading the lazy array (the inventory reaches the transaction check and returns
// WRONG_TRANSACTION).
arm_result arm_e_stage_refuses_a_count_with_no_array() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.1.engram_embd.weight", (64 * MiB) / 34);
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;

    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };

    const ggml_sycl_lifecycle_result whole = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    if (whole != GGML_SYCL_LIFECYCLE_WRONG_TRANSACTION) {
        void_arm(arr, "with the array present and no load open the stage returned " + num(whole) +
                          ", want WRONG_TRANSACTION (" + num(GGML_SYCL_LIFECYCLE_WRONG_TRANSACTION) + ")");
        return arr;
    }
    inv.inv.lazy_tables                 = nullptr;
    const ggml_sycl_lifecycle_result rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    expect(arr, rc == GGML_SYCL_LIFECYCLE_NULL_OUTPUT,
           "the stage returned " + num(rc) + " for a lazy count of 1 with no array, want NULL_OUTPUT (" +
               num(GGML_SYCL_LIFECYCLE_NULL_OUTPUT) + ")");
    return arr;
}

// (g) storage_only crosses the boundary: arm (a)'s 64 MiB table, flagged, is STORAGE in the published plan, holding at
// most its 66 MiB body (the ceiling's body term), and its own [PLACEMENT] line says so.
// Engagement: arm (a) publishes the same table HOST, holding exactly the body; the log is cleared before the stage,
// and the line matched names this table. The arm VOIDs when the host could not have held the body: the load's
// MemAvailable reading cannot clear P3 with the body resident, or the budget's fit room (budget less zones, reserve and
// envelope) is short of it. An unflagged table would then fall through to STORAGE too, and a dropped flag would pass
// every check below.
// Mutants: the copy loop not copying storage_only (the table is HOST, its line says residency=HOST); the planner not
// holding a storage-only cache to its ceiling (the cache takes every row, a 64 MiB slab plus a ~25 MiB index, past the
// 66 MiB body).
arm_result arm_g_storage_only_crosses_the_boundary() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.1.engram_embd.weight", (64 * MiB) / 34);
    table.file_offset               = 4032;
    table.storage_only              = true;
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;
    const size_t body = host_cpu_mapped_size(lazy_table_host_body_request_bytes(table.file_offset, table.size));

    const ggml_sycl_placement_envelope envelope{ inv.inv.n_ctx, inv.inv.n_ubatch, 1, -1 };
    ggml_sycl_load_txn                 txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    log_clear();
    const ggml_sycl_lifecycle_result early_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    if (early_rc != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the early stage returned " + num(early_rc));
        return arr;
    }
    if (reading_too_low(arr, body) || fit_room_too_low(arr, body)) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        return arr;
    }
    const ggml_sycl_lifecycle_result late_rc = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, false);
    if (late_rc != GGML_SYCL_LIFECYCLE_OK) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the late stage returned " + num(late_rc));
        return arr;
    }
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_end(txn, true, &model) != GGML_SYCL_LIFECYCLE_OK || model.model_id == 0) {
        void_arm(arr, "the load did not end with a model token");
        return arr;
    }
    const lazy_reading got = read_published(model);
    expect(arr, got.ok && got.count == 1, "the published plan carries " + num(got.count) + " tables, want 1");
    expect(arr, got.resident != 0 && got.resident <= body,
           "the table holds " + num(got.resident) + ", want a cache of at most its body " + num(body));
    expect(arr, !log_lines_containing("[PLACEMENT] lazy blk.1.engram_embd.weight residency=STORAGE").empty(),
           "this table's [PLACEMENT] lazy line does not say STORAGE");
    (void) ggml_backend_sycl_model_unloaded_token(model);
    return arr;
}

// (f) The dry run refuses the same inventory with NULL_OUTPUT. It runs last: with its check dropped the intake reads
// the null array, so the binary faults after every other arm has printed its verdict.
// Engagement: the same call with the array present returns OK, so the result header and the host's state let the dry
// run plan, and the inventory check is what returns NULL_OUTPUT.
// Mutant: the dry run's check not reading the lazy array (the intake dereferences the null array and the binary faults
// with no summary line, which fails the registration).
arm_result arm_f_dry_run_refuses_a_count_with_no_array() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.1.engram_embd.weight", (64 * MiB) / 34);
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;

    ggml_sycl_plan_dry_run_result result{};
    result.struct_size                     = sizeof(result);
    result.version                         = GGML_SYCL_PLAN_DRY_RUN_VERSION;
    const ggml_sycl_lifecycle_result whole = ggml_backend_sycl_plan_dry_run(&inv.inv, nullptr, &result);
    if (whole != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "with the array present the dry run returned " + num(whole) + ", want OK");
        return arr;
    }
    inv.inv.lazy_tables                 = nullptr;
    const ggml_sycl_lifecycle_result rc = ggml_backend_sycl_plan_dry_run(&inv.inv, nullptr, &result);
    expect(arr, rc == GGML_SYCL_LIFECYCLE_NULL_OUTPUT,
           "the dry run returned " + num(rc) + " for a lazy count of 1 with no array, want NULL_OUTPUT (" +
               num(GGML_SYCL_LIFECYCLE_NULL_OUTPUT) + ")");
    return arr;
}

}  // namespace

int main(int argc, char ** argv) {
    if (!parse_devices(argc, argv, g_devices)) {
        return 1;
    }
    if (!sycl_test_require_gpu("the lazy plan interface arms")) {
        return SYCL_TEST_SKIP;
    }
    // The planner caller takes the multi-device path iff this predicate holds, so each registration scores the path
    // it names or skips; a run on the other path would score the wrong planner call.
    const bool multi_path = ggml_backend_sycl_moe_multi_gpu_requested();
    if (multi_path != (g_devices == 2)) {
        std::printf(
            "SKIP: --devices %d scores the %s planner call, and this host takes the other path: "
            "ggml_backend_sycl_moe_multi_gpu_requested() is %s\n",
            g_devices, g_devices == 2 ? "multi-device" : "single-device", multi_path ? "true" : "false");
        return SYCL_TEST_SKIP;
    }
    ggml_log_set(capture_log, nullptr);

    // The backends the scheduler is shown live for the whole run, as at a load; the stage reaches the second card's
    // cache itself on the multi-device path, as the internal multi-GPU path does.
    std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> backends;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        backends.emplace_back(ggml_backend_sycl_init(d), ggml_backend_free);
        if (!backends.back()) {
            std::printf("FAIL: backend %d did not initialise\n", d);
            return 1;
        }
    }
    g_backend0 = backends.front().get();

    const std::vector<named_arm> arms = {
        { "(a) a table through the stage, the publish and the unload", arm_a_through_the_stage                     },
        { "(b) the thread's plan slot reaches the planner",            arm_b_the_slot_reaches_the_planner          },
        { "(c) a refused plan refuses the load at the stage",          arm_c_the_stage_refuses_a_refused_plan      },
        { "(d) the late pass refuses a plan the early pass held",      arm_d_the_late_pass_refuses                 },
        { "(e) the stage refuses a lazy count with no array",          arm_e_stage_refuses_a_count_with_no_array   },
        { "(g) storage_only crosses the boundary",                     arm_g_storage_only_crosses_the_boundary     },
        // Last: with its check dropped the dry run faults the binary.
        { "(f) the dry run refuses a lazy count with no array",        arm_f_dry_run_refuses_a_count_with_no_array },
    };
    return run_arms(arms, "lazy plan interface arms (--devices " + std::to_string(g_devices) + ")");
}
