//
// The lazy tables' residency (th32 §4.3 step 4, §11 T3): the planner decides each table HOST or STORAGE from the
// load's one MemAvailable reading, P3 (MemAvailable_pre - expected_host - 5 >= 24 GiB) and the coverage pin, and a
// table never enters a zone maximum.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The planner's shaped path initialises SYCL and reads each device's cache, so the binary is registered
// gpu-serial, serial, with the selector pinned to the discrete cards. --devices 2 runs iff the host takes the
// multi-device path, which is the planner caller's own predicate, ggml_backend_sycl_moe_multi_gpu_requested();
// otherwise it exits 77 before any arm runs. The scheduler-exposed device count is never the test: this backend exposes
// one device whenever more than one GPU is visible.
//
//   test-sycl-lazy-residency-plan --devices 1   compute_placement_plan, and compute_multi_device_plan with one budget
//                                               (its one-device delegation)
//   test-sycl-lazy-residency-plan --devices 2   compute_multi_device_plan with two budgets
//
// The readings are synthetic: each arm sets placement_kv_info::host_mem_available_pre itself, at a threshold derived
// from the plan's own zone terms (step 2 of §4.3), so the arms decide the same way on any host. The budget is 4 TiB so
// the planner's fit never binds and P3 alone decides.
//
// Every arm names its engagement check and the mutant it kills. An arm whose engagement check fails FAILS: each check
// is derived from the plan's own terms, so a miss is a defect in the planner or the fixture, never an unmet
// precondition. Every arm runs, so one failure does not hide the rest.

#include "../host-cpu-mapping.hpp"
#include "../lazy-io-pool.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-ledger-test-harness.hpp"
#include "host-plan-shaped-inventory.hpp"
#include "sycl-test-skip.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using host_plan_shaped_inventory::moe_shape;
using host_plan_shaped_inventory::planner_inventory;
using host_plan_shaped_inventory::qwen38_shape;

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;
constexpr uint64_t GiB = 1024ull * MiB;

// P3's margin over expected_host: 24 GiB left after load, plus P3's 5 GiB runtime allowance (§4.3 step 4).
constexpr uint64_t k_p3_margin = 29 * GiB;
// The budget the planner is handed: far above every plan here, so the fit never decides.
constexpr uint64_t k_budget    = 4096 * GiB;
// One yieldable chunk with its index share and huge-page rounding stays under this, for every row size here.
constexpr uint64_t k_chunk_cap = 320 * MiB;

// DeepSeek-V4.1-Flash's quant type at a size a test can hold, with 16 layers so both engram consumers (blocks 1 and
// 14) are layers of the model.
constexpr moe_shape ds_shape = { GGML_TYPE_Q2_K, 16, 8, 2, 1024, 512, 4096, 16, 1, 0, 0, 0 };

int g_devices = 1;

// A lazy table at its real extent (design §2.1): DeepSeek's two Q2_K engram tables, 24 rows per token each, and
// Qwen3.8-Flash-Next's Q8_0 PLE table, 16 rows per token.
placement_lazy_table_info lazy_table(const char * name,
                                     ggml_type    type,
                                     int64_t      ne0,
                                     int64_t      rows,
                                     uint32_t     per_token,
                                     int32_t      layer) {
    placement_lazy_table_info t;
    t.name           = name;
    t.type           = type;
    t.ne[0]          = ne0;
    t.ne[1]          = rows;
    t.ne[2]          = 1;
    t.ne[3]          = 1;
    t.row_size       = ggml_row_size(type, ne0);
    t.rows_per_token = per_token;
    t.consumer_layer = layer;
    t.table_bytes    = t.row_size * static_cast<size_t>(rows);
    return t;
}

std::vector<placement_lazy_table_info> deepseek_tables() {
    return {
        lazy_table("blk.1.engram_embd.weight", GGML_TYPE_Q2_K, 256, 384006168, 24, 1),
        lazy_table("blk.14.engram_embd.weight", GGML_TYPE_Q2_K, 256, 384016682, 24, 14),
    };
}

std::vector<placement_lazy_table_info> qwen_tables() {
    return { lazy_table("per_layer_token_embd.weight", GGML_TYPE_Q8_0, 160, 320001536, 16, 1) };
}

size_t mapped(size_t bytes) {
    return host_cpu_mapped_size(bytes);
}

// The cache floor's slots (§4.3 step 4: one double-buffered ubatch working set) and a lower bound on its bytes: the
// slab's raw bytes, before any chunk rounding or index.
size_t floor_slots(const placement_lazy_table_info & t, const placement_kv_info & kv) {
    return 2ull * kv.n_ubatch * t.rows_per_token;
}

size_t floor_raw_bytes(const placement_lazy_table_info & t, const placement_kv_info & kv) {
    return floor_slots(t, kv) * t.row_size;
}

// An upper bound on a floor's mapped bytes, for the arms' preconditions only: the slab and a generous index
// (4 x ceil(4s/3) + 8s bytes), each rounded to its mapping.
size_t floor_upper_bytes(const placement_lazy_table_info & t, const placement_kv_info & kv) {
    const size_t s = floor_slots(t, kv);
    return mapped(floor_raw_bytes(t, kv)) + mapped(4 * ((4 * s + 2) / 3) + 8 * s);
}

// The plan for `shape` and `kv` against the --devices count: every device gets a 1 MiB VRAM budget, so the weights are
// host-resident and step 2 is the host zones they size. `delegate` routes a one-device plan through
// compute_multi_device_plan's one-budget delegation instead of compute_placement_plan.
placement_plan plan_for(const moe_shape &         shape,
                        const placement_kv_info & kv,
                        const ring_lease *        pin,
                        bool                      delegate = false) {
    const std::vector<placement_tensor_info> inventory = planner_inventory(shape);
    if (g_devices == 1 && !delegate) {
        return compute_placement_plan(inventory, 1 * MiB, 0, kv, nullptr, shape.experts, k_budget, pin);
    }
    std::vector<device_budget> budgets;
    for (int d = 0; d < g_devices; ++d) {
        budgets.push_back({ d, 1 * MiB, 16ull * 1024ull * MiB, 1.0, true });
    }
    return compute_multi_device_plan(budgets, inventory, shape.layers, multi_gpu_mode::HYBRID, kv, nullptr,
                                     shape.experts, k_budget, pin);
}

placement_kv_info kv_with(const std::vector<placement_lazy_table_info> & tables, size_t reading) {
    placement_kv_info kv{};
    kv.lazy_tables            = tables;
    kv.host_mem_available_pre = reading;
    return kv;
}

const placement_lazy_table * find_table(const placement_plan & plan, const std::string & name) {
    for (const placement_lazy_table & t : plan.lazy_tables) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

size_t resident_sum(const placement_plan & plan) {
    size_t sum = 0;
    for (const placement_lazy_table & t : plan.lazy_tables) {
        sum += t.host_resident_bytes;
    }
    return sum;
}

const char * residency_name(lazy_residency r) {
    return r == lazy_residency::HOST ? "HOST" : "STORAGE";
}

// A STORAGE table's checks: its cache holds at least the floor, and less than the whole table's mapping.
void expect_storage(arm_result &                      r,
                    const placement_plan &            plan,
                    const placement_lazy_table_info & info,
                    const placement_kv_info &         kv,
                    const std::string &               at) {
    const placement_lazy_table * t = find_table(plan, info.name);
    if (t == nullptr) {
        fail(r, at + info.name + " is not in the plan");
        return;
    }
    expect(r, t->residency == lazy_residency::STORAGE,
           at + info.name + " is " + residency_name(t->residency) + ", want STORAGE");
    expect(r, t->host_resident_bytes >= floor_raw_bytes(info, kv),
           at + info.name + " holds " + num(t->host_resident_bytes) + " resident bytes, below its floor's " +
               num(floor_raw_bytes(info, kv)) + " slab bytes");
    expect(r, t->host_resident_bytes < mapped(info.table_bytes),
           at + info.name + " holds " + num(t->host_resident_bytes) + " bytes, not less than the table's mapping " +
               num(mapped(info.table_bytes)));
    expect(r, t->cache_floor_slots == floor_slots(info, kv),
           at + info.name + " floor is " + num(t->cache_floor_slots) + " slots, want " + num(floor_slots(info, kv)));
    expect(r, t->cache_slots >= t->cache_floor_slots,
           at + info.name + " cache of " + num(t->cache_slots) + " slots is below its floor");
}

void expect_host(arm_result &                      r,
                 const placement_plan &            plan,
                 const placement_lazy_table_info & info,
                 const std::string &               at) {
    const placement_lazy_table * t = find_table(plan, info.name);
    if (t == nullptr) {
        fail(r, at + info.name + " is not in the plan");
        return;
    }
    expect(r, t->residency == lazy_residency::HOST,
           at + info.name + " is " + residency_name(t->residency) + ", want HOST");
    expect(r, t->host_resident_bytes == mapped(info.table_bytes),
           at + info.name + " holds " + num(t->host_resident_bytes) + " resident bytes, want its mapping " +
               num(mapped(info.table_bytes)));
    expect(r, t->cache_slots == 0, at + info.name + " is HOST with a cache of " + num(t->cache_slots) + " slots");
}

// P3 with the tables resident: step 2, the tables' resident bytes and the ring charge plus the margin fit the reading.
void expect_p3(arm_result & r, const placement_plan & plan, size_t step2, size_t reading, const std::string & at) {
    const size_t need = step2 + resident_sum(plan) + plan.block_ring_charge + k_p3_margin;
    expect(r, need <= reading,
           at + "P3 does not hold: step 2 " + num(step2) + " + resident " + num(resident_sum(plan)) + " + ring " +
               num(plan.block_ring_charge) + " + 29 GiB = " + num(need) + " > the reading " + num(reading));
}

// The zone terms with no table: step 2, the figure every threshold is derived from.
bool step2_of(arm_result & r, const moe_shape & shape, const ring_lease * pin, size_t & step2, bool delegate = false) {
    const placement_plan bare = plan_for(shape, kv_with({}, 0), pin, delegate);
    step2                     = bare.host_zone_terms.total();
    if (bare.entries.empty() || bare.weight_host_bytes == 0) {
        fail(r, "the plan with no table put no weight on the host, so step 2 is not the host zones it sizes");
        return false;
    }
    if (!bare.lazy_tables.empty() || !bare.lazy_refusal.empty()) {
        fail(r, "a plan with no lazy table carries " + num(bare.lazy_tables.size()) + " tables and refusal '" +
                    bare.lazy_refusal + "'");
        return false;
    }
    return true;
}

void expect_same_step2(arm_result & r, const placement_plan & plan, size_t step2, const std::string & at) {
    if (plan.host_zone_terms.total() != step2) {
        fail(r, at + "the zone terms moved between plans: " + num(plan.host_zone_terms.total()) + " vs step 2 " +
                    num(step2));
    }
}

void expect_pin_held(arm_result & r, const lazy_test_coverage_pin & pin) {
    expect(r, pin.pin() != nullptr, "the coverage pin holds no lease: the registry's take returned an empty one");
}

// (a) The tables stay out of every zone maximum, and the plan's host figure is the zones plus the tables' terms.
// Engagement: both tables are carried and the plan without them carries none, at one pin and one budget.
// Mutants: a table counted as max_tensor (STAGING, SCRATCH and max_tensor_bytes move); host_planned_bytes left at the
// zone total (the tables' terms missing) or charged twice; the never-HOST stub (the floors are absent).
arm_result arm_a_tables_stay_out_of_the_zone_maxima() {
    arm_result                                   r;
    const lazy_test_coverage_pin                 pin    = ggml_sycl_test_make_coverage_pin(0);
    const std::vector<placement_lazy_table_info> tables = deepseek_tables();
    size_t                                       step2  = 0;
    if (!step2_of(r, ds_shape, pin.pin(), step2)) {
        return r;
    }
    const size_t            reading = step2 + mapped(tables[0].table_bytes) + k_p3_margin - 1 * GiB;
    const placement_kv_info kv      = kv_with(tables, reading);
    const placement_plan    without = plan_for(ds_shape, kv_with({}, reading), pin.pin());
    const placement_plan    with    = plan_for(ds_shape, kv, pin.pin());
    expect_same_step2(r, with, step2, "");
    if (with.lazy_tables.size() != tables.size() || !without.lazy_tables.empty()) {
        fail(r, "the plan carries " + num(with.lazy_tables.size()) + " of " + num(tables.size()) +
                    " tables (refusal '" + with.lazy_refusal + "'), the bare plan " + num(without.lazy_tables.size()));
        return r;
    }
    expect(r, with.host_zone_staging_bytes == without.host_zone_staging_bytes,
           "STAGING moved: " + num(with.host_zone_staging_bytes) + " vs " + num(without.host_zone_staging_bytes));
    expect(r, with.host_zone_scratch_bytes == without.host_zone_scratch_bytes,
           "SCRATCH moved: " + num(with.host_zone_scratch_bytes) + " vs " + num(without.host_zone_scratch_bytes));
    expect(r, with.host_zone_weight_bytes == without.host_zone_weight_bytes,
           "WEIGHT moved: " + num(with.host_zone_weight_bytes) + " vs " + num(without.host_zone_weight_bytes));
    expect(r, with.max_tensor_bytes == without.max_tensor_bytes,
           "max_tensor_bytes moved: " + num(with.max_tensor_bytes) + " vs " + num(without.max_tensor_bytes));
    expect(r, with.max_staging_pair_bytes == without.max_staging_pair_bytes,
           "max_staging_pair_bytes moved: " + num(with.max_staging_pair_bytes) + " vs " +
               num(without.max_staging_pair_bytes));
    for (const placement_lazy_table_info & t : tables) {
        expect_storage(r, with, t, kv, "");
    }
    expect(r, without.host_planned_bytes == without.host_zone_terms.total(),
           "the bare plan's host figure " + num(without.host_planned_bytes) + " is not its zone total " +
               num(without.host_zone_terms.total()));
    expect(r, with.host_planned_bytes == without.host_planned_bytes + resident_sum(with) + with.block_ring_charge,
           "host_planned_bytes " + num(with.host_planned_bytes) + ", want the bare plan's " +
               num(without.host_planned_bytes) + " + resident " + num(resident_sum(with)) + " + ring " +
               num(with.block_ring_charge));
    expect_pin_held(r, pin);
    return r;
}

// (b) DeepSeek, both tables STORAGE: at threshold_one - 1 GiB neither table passes P3 HOST, both caches hold at least
// their floors, P3 holds, and the caches take what is left above it as yieldable shares.
// Engagement: threshold_one < threshold_both, and the reading clears step 2 plus both floors and the margin (a failed
// precondition FAILS: the reading is derived, so missing it is the fixture's defect, not the host's).
// Mutants: the never-HOST stub (no floor); HOST without the P3 test (both go HOST); the yieldable share given to one
// table or left unspent; the ascending consumer_layer order reversed (checked by the plan's order).
arm_result deepseek_both_storage(bool delegate) {
    arm_result                                   r;
    const lazy_test_coverage_pin                 pin    = ggml_sycl_test_make_coverage_pin(0);
    const std::vector<placement_lazy_table_info> tables = deepseek_tables();
    size_t                                       step2  = 0;
    if (!step2_of(r, ds_shape, pin.pin(), step2, delegate)) {
        return r;
    }
    const size_t threshold_one  = step2 + mapped(tables[0].table_bytes) + k_p3_margin;
    const size_t threshold_both = step2 + mapped(tables[0].table_bytes) + mapped(tables[1].table_bytes) + k_p3_margin;
    const size_t reading        = threshold_one - 1 * GiB;
    std::printf("  deepseek both-STORAGE: step2=%zu threshold_one=%zu threshold_both=%zu reading=%zu\n", step2,
                threshold_one, threshold_both, reading);
    const placement_kv_info kv = kv_with(tables, reading);
    expect(r, threshold_one < threshold_both, "threshold_one is not below threshold_both");
    expect(r, reading >= step2 + floor_upper_bytes(tables[0], kv) + floor_upper_bytes(tables[1], kv) + k_p3_margin,
           "the reading does not clear step 2, both floors and the margin");
    const placement_plan plan = plan_for(ds_shape, kv, pin.pin(), delegate);
    expect_same_step2(r, plan, step2, "");
    expect(r, plan.lazy_refusal.empty(), "the plan refused: " + plan.lazy_refusal + " " + plan.lazy_refusal_detail);
    expect(r, plan.lazy_tables.size() == 2 && plan.lazy_tables[0].name == tables[0].name,
           "the plan does not carry blk.1 first of two tables (ascending consumer_layer)");
    for (const placement_lazy_table_info & t : tables) {
        expect_storage(r, plan, t, kv, "");
    }
    expect_p3(r, plan, step2, reading, "");
    const size_t need = step2 + resident_sum(plan) + plan.block_ring_charge + k_p3_margin;
    const size_t left = reading > need ? reading - need : 0;
    expect(r, left < 2 * k_chunk_cap,
           num(left) +
               " bytes are left above P3, more than one yieldable chunk per table: the caches did not take "
               "what was left");
    const placement_lazy_table * a = find_table(plan, tables[0].name);
    const placement_lazy_table * b = find_table(plan, tables[1].name);
    if (a != nullptr && b != nullptr) {
        const size_t slots_per_chunk = (256 * MiB) / tables[0].row_size;
        const size_t gap =
            a->cache_slots > b->cache_slots ? a->cache_slots - b->cache_slots : b->cache_slots - a->cache_slots;
        expect(r, gap <= slots_per_chunk,
               "the two caches differ by " + num(gap) + " slots, more than one chunk's " + num(slots_per_chunk) +
                   ": the yieldable share is not split by table_bytes x rows_per_token");
    }
    expect(r, plan.block_ring_qd == LAZY_BLOCK_RING_QD, "block_ring_qd " + num(plan.block_ring_qd));
    expect(r, plan.block_ring_need == LAZY_BLOCK_RING_QD * lazy_block_ring_span(tables[0].row_size),
           "block_ring_need " + num(plan.block_ring_need) + ", want QD x the widest STORAGE span");
    expect(r, plan.block_ring_charge == mapped(plan.block_ring_need),
           "a ring-less pool's charge is " + num(plan.block_ring_charge) + ", want the need's mapping");
    expect_pin_held(r, pin);
    return r;
}

arm_result arm_b_deepseek_both_storage() {
    return deepseek_both_storage(false);
}

arm_result arm_b_deepseek_both_storage_delegated() {
    return deepseek_both_storage(true);
}

// (c) DeepSeek, one table HOST: at threshold_one + 1 GiB blk.1 (the lower consumer_layer) passes P3 HOST and blk.14
// cannot pass it beside blk.1, so it is STORAGE.
// Engagement: the reading is above threshold_one and below threshold_both.
// Mutants: the never-HOST stub (blk.1 STORAGE); HOST without the P3 test (blk.14 HOST too); the order reversed (blk.14
// takes the HOST slot); a HOST table's bytes not counted in the next table's P3 test.
arm_result arm_c_deepseek_one_host() {
    arm_result                                   r;
    const lazy_test_coverage_pin                 pin    = ggml_sycl_test_make_coverage_pin(0);
    const std::vector<placement_lazy_table_info> tables = deepseek_tables();
    size_t                                       step2  = 0;
    if (!step2_of(r, ds_shape, pin.pin(), step2)) {
        return r;
    }
    const size_t threshold_one  = step2 + mapped(tables[0].table_bytes) + k_p3_margin;
    const size_t threshold_both = step2 + mapped(tables[0].table_bytes) + mapped(tables[1].table_bytes) + k_p3_margin;
    const size_t reading        = threshold_one + 1 * GiB;
    std::printf("  deepseek one-HOST: step2=%zu threshold_one=%zu threshold_both=%zu reading=%zu\n", step2,
                threshold_one, threshold_both, reading);
    if (reading >= threshold_both) {
        fail(r, "the reading is not below threshold_both");
        return r;
    }
    const placement_kv_info kv   = kv_with(tables, reading);
    const placement_plan    plan = plan_for(ds_shape, kv, pin.pin());
    expect_same_step2(r, plan, step2, "");
    expect(r, plan.lazy_refusal.empty(), "the plan refused: " + plan.lazy_refusal + " " + plan.lazy_refusal_detail);
    expect_host(r, plan, tables[0], "");
    expect_storage(r, plan, tables[1], kv, "");
    expect_p3(r, plan, step2, reading, "");
    expect_pin_held(r, pin);
    return r;
}

// (d) Qwen3.8-Flash-Next's PLE table flips at its threshold: STORAGE 1 GiB below step 2 + mapping + 29 GiB, HOST 1 GiB
// above it, on the Qwen-shaped inventory whose experts put ~130 GiB of weight zones on the host.
// Engagement: step 2 carries the host experts (non-zero weight bytes), and the same plan is made at both readings.
// Mutants: the never-HOST stub (the HOST reading fails); HOST without P3, or P3 read against the 4 TiB budget rather
// than the reading (the STORAGE reading fails).
arm_result arm_d_qwen_flip() {
    arm_result                                   r;
    const lazy_test_coverage_pin                 pin    = ggml_sycl_test_make_coverage_pin(0);
    const std::vector<placement_lazy_table_info> tables = qwen_tables();
    size_t                                       step2  = 0;
    if (!step2_of(r, qwen38_shape, pin.pin(), step2)) {
        return r;
    }
    const size_t threshold = step2 + mapped(tables[0].table_bytes) + k_p3_margin;
    std::printf("  qwen flip: step2=%zu threshold=%zu\n", step2, threshold);

    const placement_kv_info below_kv = kv_with(tables, threshold - 1 * GiB);
    const placement_plan    below    = plan_for(qwen38_shape, below_kv, pin.pin());
    expect_same_step2(r, below, step2, "below: ");
    expect(r, below.lazy_refusal.empty(), "below: the plan refused: " + below.lazy_refusal);
    expect_storage(r, below, tables[0], below_kv, "below: ");
    expect_p3(r, below, step2, threshold - 1 * GiB, "below: ");

    const placement_kv_info above_kv = kv_with(tables, threshold + 1 * GiB);
    const placement_plan    above    = plan_for(qwen38_shape, above_kv, pin.pin());
    expect_same_step2(r, above, step2, "above: ");
    expect(r, above.lazy_refusal.empty(), "above: the plan refused: " + above.lazy_refusal);
    expect_host(r, above, tables[0], "above: ");
    expect_p3(r, above, step2, threshold + 1 * GiB, "above: ");
    expect_pin_held(r, pin);
    return r;
}

// (e) With no coverage pin no table is STORAGE-eligible: at a reading where neither DeepSeek table passes the HOST test
// the plan refuses as lazy-table-ring-unpinned, carrying no table and no ring.
// Engagement: arm (b) plans the same tables at the same reading with a pin and carries both STORAGE.
// Mutants: the never-HOST stub (STORAGE without a pin); a null pin treated as a ring-less pool; the refusal leaving
// the tables or the ring on the plan.
arm_result arm_e_null_pin_refuses() {
    arm_result                                   r;
    const std::vector<placement_lazy_table_info> tables = deepseek_tables();
    size_t                                       step2  = 0;
    if (!step2_of(r, ds_shape, nullptr, step2)) {
        return r;
    }
    const size_t         reading = step2 + mapped(tables[0].table_bytes) + k_p3_margin - 1 * GiB;
    const placement_plan plan    = plan_for(ds_shape, kv_with(tables, reading), nullptr);
    expect(r, plan.lazy_refusal == "lazy-table-ring-unpinned",
           "the plan's refusal is '" + plan.lazy_refusal + "', want lazy-table-ring-unpinned");
    expect(r, !plan.lazy_refusal_detail.empty(), "the refusal carries no figures");
    expect(r, plan.lazy_tables.empty(), "a refused plan carries " + num(plan.lazy_tables.size()) + " tables");
    expect(r, plan.block_ring_need == 0 && plan.block_ring_charge == 0, "a refused plan carries a ring");
    expect(r, plan.host_planned_bytes == plan.host_zone_terms.total(),
           "a refused plan's host figure " + num(plan.host_planned_bytes) + " is not its zone total");
    return r;
}

// (f) A cold expert class is not implemented: a plan that carries lazy tables refuses it by name; one that carries
// none refuses nothing.
// Engagement: the same tables without the class plan without a refusal (arm (b)).
// Mutants: the class ignored (no refusal); the refusal raised for a model with no lazy table.
arm_result arm_f_cold_class_refuses() {
    arm_result                   r;
    const lazy_test_coverage_pin pin   = ggml_sycl_test_make_coverage_pin(0);
    size_t                       step2 = 0;
    if (!step2_of(r, ds_shape, pin.pin(), step2)) {
        return r;
    }
    placement_kv_info kv      = kv_with(deepseek_tables(), step2 + 512 * GiB);
    kv.cold_expert_class      = true;
    const placement_plan cold = plan_for(ds_shape, kv, pin.pin());
    expect(r, cold.lazy_refusal == "cold-set-not-implemented",
           "the plan's refusal is '" + cold.lazy_refusal + "', want cold-set-not-implemented");
    expect(r, cold.lazy_tables.empty(), "a refused plan carries " + num(cold.lazy_tables.size()) + " tables");

    placement_kv_info none    = kv_with({}, step2 + 512 * GiB);
    none.cold_expert_class    = true;
    const placement_plan bare = plan_for(ds_shape, none, pin.pin());
    expect(r, bare.lazy_refusal.empty(), "a plan with no lazy table refused '" + bare.lazy_refusal + "'");
    expect_pin_held(r, pin);
    return r;
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
    if (!sycl_test_require_gpu("the lazy residency plan arms")) {
        return SYCL_TEST_SKIP;
    }
    if (g_devices == 2 && !ggml_backend_sycl_moe_multi_gpu_requested()) {
        std::printf(
            "SKIP: --devices 2 plans on the multi-device path, and this host does not take it: "
            "ggml_backend_sycl_moe_multi_gpu_requested() is false\n");
        return SYCL_TEST_SKIP;
    }
    ggml_log_set(capture_log, nullptr);

    // The backends the scheduler is shown stay alive across the arms, as at a load, so every plan reads the same
    // caches; the multi-device planner reaches the second card's cache itself, as the internal multi-GPU path does.
    std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> backends;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        backends.emplace_back(ggml_backend_sycl_init(d), ggml_backend_free);
        if (!backends.back()) {
            std::printf("FAIL: backend %d did not initialise\n", d);
            return 1;
        }
    }

    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    std::vector<named_arm> arms = {
        { "(a) the tables stay out of every zone maximum",   arm_a_tables_stay_out_of_the_zone_maxima },
        { "(b) DeepSeek: both tables STORAGE below P3",      arm_b_deepseek_both_storage              },
        { "(c) DeepSeek: blk.1 HOST, blk.14 STORAGE",        arm_c_deepseek_one_host                  },
        { "(d) Qwen PLE flips at its P3 threshold",          arm_d_qwen_flip                          },
        { "(e) a null pin refuses lazy-table-ring-unpinned", arm_e_null_pin_refuses                   },
        { "(f) a cold expert class refuses by name",         arm_f_cold_class_refuses                 },
    };
    if (g_devices == 1) {
        arms.push_back(
            { "(b') DeepSeek both STORAGE through the one-budget delegation", arm_b_deepseek_both_storage_delegated });
    }
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
    std::printf("lazy residency plan arms (--devices %d): %d passed, %d failed, %d void\n", g_devices, passed, failed,
                voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}
