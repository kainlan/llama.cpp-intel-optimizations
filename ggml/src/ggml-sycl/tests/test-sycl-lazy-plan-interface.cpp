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
// takes the multi-device path (the default with two GPUs) and a host with fewer exits 77.
//
// Every arm names its engagement check and the mutant it kills. An arm whose engagement check fails is VOID, and a VOID
// arm fails the binary like a FAIL. Every arm runs, so one failure does not hide the rest.

#include "../host-cpu-mapping.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-ledger-test-harness.hpp"
#include "host-plan-shaped-inventory.hpp"
#include "sycl-test-skip.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace host_ledger_test_harness;
using ggml_sycl::host_cpu_mapped_size;
using host_plan_shaped_inventory::loader_inventory;

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;
constexpr uint64_t GiB = 1024ull * MiB;

// P3's margin over expected_host (§4.3 step 4).
constexpr uint64_t k_p3_margin = 29 * GiB;

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

std::vector<std::string> lines_with(const char * a, const char * b = nullptr) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::vector<std::string>    out;
    for (const std::string & line : g_log_lines) {
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            out.push_back(line);
        }
    }
    return out;
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
    const std::vector<std::string> lines = lines_with("[HOST-PLAN] lazy ");
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
// Engagement: the pin holds a lease (coverage_pin_make returned one), and the staged-load seam made the direct entry
// stage a candidate for the transaction.
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
    const size_t floor_raw =
        2ull * (inv.inv.n_ubatch != 0 ? inv.inv.n_ubatch : 512) * table.rows_per_token * table.row_size;

    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(arr, "the load transaction did not begin");
        return arr;
    }
    void * pin = ggml_sycl_test_coverage_pin_make(0);
    if (pin == nullptr) {
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        void_arm(arr, "the coverage pin could not be made");
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
        const size_t refused = lines_with("[HOST-PLAN] lazy ", "refusal=lazy-table-ring-unpinned").size();
        fail(arr, "the candidate carries " + num(got.count) + " lazy tables, want 1 STORAGE table; the plan logged " +
                      "refusal=lazy-table-ring-unpinned " + num(refused) + " times");
        return arr;
    }
    const std::vector<std::string> placed = lines_with("[PLACEMENT] lazy blk.14.engram_embd.weight ");
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

const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

}  // namespace

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--devices") == 0 && i + 1 < argc) {
            g_devices = std::atoi(argv[++i]);
        }
    }
    if (g_devices != 1 && g_devices != 2) {
        std::printf("FAIL: --devices must be 1 or 2, got %d\n", g_devices);
        return 1;
    }
    if (!sycl_test_require_gpu("the lazy plan interface arms")) {
        return SYCL_TEST_SKIP;
    }
    if (ggml_backend_sycl_get_device_count() != g_devices) {
        std::printf("SKIP: --devices %d needs exactly %d SYCL GPUs visible, %d are\n", g_devices, g_devices,
                    ggml_backend_sycl_get_device_count());
        return SYCL_TEST_SKIP;
    }
    ggml_log_set(capture_log, nullptr);

    // Every device's backend lives for the whole run, as at a load: the stage plans through each device's cache.
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b0(ggml_backend_sycl_init(0), ggml_backend_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b1(g_devices > 1 ? ggml_backend_sycl_init(1) : nullptr,
                                                                   ggml_backend_free);
    if (!b0 || (g_devices > 1 && !b1)) {
        std::printf("FAIL: a backend did not initialise\n");
        return 1;
    }
    g_backend0 = b0.get();

    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    const named_arm arms[] = {
        { "(a) a table through the stage, the publish and the unload", arm_a_through_the_stage            },
        { "(b) the thread's plan slot reaches the planner",            arm_b_the_slot_reaches_the_planner },
    };
    int passed = 0, failed = 0, voided = 0;
    for (const named_arm & arm : arms) {
        const arm_result result = arm.run();
        std::printf("%s %s%s%s\n", outcome_name(result.result), arm.name, result.why.empty() ? "" : ": ",
                    result.why.c_str());
        std::fflush(stdout);
        passed += result.result == outcome::PASS;
        failed += result.result == outcome::FAIL;
        voided += result.result == outcome::VOID;
    }
    std::printf("lazy plan interface arms (--devices %d): %d passed, %d failed, %d void\n", g_devices, passed, failed,
                voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}
