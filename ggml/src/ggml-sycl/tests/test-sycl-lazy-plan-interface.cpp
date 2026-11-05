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

// (a) Through the stage: a 64 MiB table in the inventory is HOST in the early candidate, the late candidate and the
// published plan, each holding the table's mapping, and the published plan leaves with the model.
// Engagement: no candidate exists before the first stage (the reader answers VOID, not 0), both stages and the load's
// end succeed, and the published plan is gone after the unload.
// Mutants: the inventory's tables not copied into the planner's input (count 0); the late stage or the publish
// dropping the tables; the reader answering a stale or zero plan.
arm_result arm_a_through_the_stage() {
    arm_result                arr;
    loader_inventory          inv;
    ggml_sycl_lazy_table_info table = lazy_table("blk.1.engram_embd.weight", (64 * MiB) / 34);
    inv.inv.lazy_tables             = &table;
    inv.inv.lazy_table_count        = 1;
    const size_t resident           = host_cpu_mapped_size(table.size);

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
    expect(arr, got.resident < host_cpu_mapped_size(table.size),
           "the cache holds " + num(got.resident) + " bytes, not less than the table's mapping");
    if (!placed.empty()) {
        expect(arr, printed(placed.back(), "resident") == got.resident,
               "the candidate's resident bytes " + num(got.resident) + " are not the [PLACEMENT] line's " +
                   num(printed(placed.back(), "resident")));
    }
    return arr;
}

// (c) A refused plan refuses the load at the stage (th32 §4.3 step 4): with no pin on the slot, a ~1 TiB table can be
// neither HOST (it exceeds the budget at any reading) nor STORAGE, so the plan refuses as lazy-table-ring-unpinned, and
// the early stage returns PLAN_REJECTED with exactly one "[HOST-PLAN] load refused:" WARN naming the reason, and its
// rollback leaves no candidate for the transaction.
// Engagement: the planner's [HOST-PLAN] lazy line carries refusal=lazy-table-ring-unpinned, so the stage was shown a
// refused plan.
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
    log_clear();
    const ggml_sycl_lifecycle_result rc   = ggml_backend_sycl_stage_load_inventory_plan(&inv.inv, &envelope, true);
    const lazy_reading               left = read_candidate(txn);
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);

    if (log_lines_containing("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-unpinned").empty()) {
        void_arm(arr, "no plan line carries refusal=lazy-table-ring-unpinned, so the stage saw no refused plan");
        return arr;
    }
    expect(arr, rc == GGML_SYCL_LIFECYCLE_PLAN_REJECTED,
           "the early stage returned " + num(rc) + ", want PLAN_REJECTED (" + num(GGML_SYCL_LIFECYCLE_PLAN_REJECTED) +
               ")");
    const size_t warned = log_lines_containing("[HOST-PLAN] load refused: lazy-table-ring-unpinned ").size();
    expect(arr, warned == 1,
           "the stage logged " + num(warned) + " load-refused lines naming lazy-table-ring-unpinned, want 1");
    expect(arr, !left.ok, "a candidate is still staged for the refused load");
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
        { "(a) a table through the stage, the publish and the unload", arm_a_through_the_stage                },
        { "(b) the thread's plan slot reaches the planner",            arm_b_the_slot_reaches_the_planner     },
        { "(c) a refused plan refuses the load at the stage",          arm_c_the_stage_refuses_a_refused_plan },
    };
    return run_arms(arms, "lazy plan interface arms (--devices " + std::to_string(g_devices) + ")");
}
