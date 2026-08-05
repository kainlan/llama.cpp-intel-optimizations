//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Integration contract for the current-model planner-placement verdict.

#include "ggml-sycl.h"
#include "ggml-sycl/tiered-plan-clear.hpp"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

struct inventory_fixture {
    std::vector<std::string>           names;
    std::vector<ggml_sycl_tensor_info> tensors;
    ggml_sycl_tensor_inventory         inventory{};
    bool                               valid = false;

    inventory_fixture(const char * prefix, size_t count, size_t bytes_per_tensor) {
        if (count > std::numeric_limits<uint32_t>::max() ||
            (count != 0 && bytes_per_tensor > std::numeric_limits<size_t>::max() / count)) {
            return;
        }

        names.reserve(count);
        tensors.reserve(count);
        const size_t total = count * bytes_per_tensor;
        for (size_t i = 0; i < count; ++i) {
            names.emplace_back(std::string(prefix) + ".blk." + std::to_string(i) + ".attn_q.weight");
            ggml_sycl_tensor_info tensor{};
            tensor.name = names.back().c_str();
            tensor.size = bytes_per_tensor;
            tensors.push_back(tensor);
        }
        inventory.tensors    = tensors.data();
        inventory.count      = tensors.size();
        inventory.total_size = total;
        inventory.n_layer    = static_cast<uint32_t>(count);
        valid                = true;
    }
};

bool planner_target_oracle(const inventory_fixture & fixture, const char * label) {
    bool any_host = false;
    for (const std::string & name : fixture.names) {
        const int target = ggml_backend_sycl_planned_target_device(name.c_str());
        if (target == GGML_SYCL_PLANNED_NO_PLAN) {
            std::fprintf(stderr, "FAIL: %s has no planner target for %s\n", label, name.c_str());
            ++failures;
            continue;
        }
        any_host |= target == -1;
    }
    return any_host;
}

void run_placement_case(ggml_backend_t backend,
                        const char *   label,
                        size_t         count,
                        size_t         bytes_per_tensor,
                        bool           expect_host) {
    const int failures_before = failures;

    // This outer boundary must invalidate the preceding model's verdict before
    // the incoming inventory is known.
    ggml_sycl_load_txn load{};
    ggml_sycl_model_token model{};
    check(ggml_backend_sycl_model_load_begin(&load) == GGML_SYCL_LIFECYCLE_OK, "explicit model lifecycle begin failed");
    check(!ggml_backend_sycl_is_tiered_enabled(backend), "outer model-load boundary did not reset verdict");

    inventory_fixture fixture(label, count, bytes_per_tensor);
    check(fixture.valid, "inventory fixture size overflowed");
    if (!fixture.valid) {
        return;
    }
    ggml_backend_sycl_set_tensor_inventory(backend, &fixture.inventory);

    const bool oracle  = planner_target_oracle(fixture, label);
    const bool verdict = ggml_backend_sycl_is_tiered_enabled(backend);
    check(oracle == expect_host, expect_host ? "over-budget planner oracle found no host target" :
                                               "all-device planner oracle found a host target");
    check(verdict == oracle, "tiered query disagrees with independent planned-target oracle");
    check(ggml_backend_sycl_has_tensor_cache(backend), "unified cache gate unavailable for planned model");

    check(ggml_backend_sycl_model_load_end(load, true, &model) == GGML_SYCL_LIFECYCLE_OK, "explicit model lifecycle commit failed");
    check(ggml_backend_sycl_is_tiered_enabled(backend) == oracle,
          "planner verdict did not survive model-load completion");

    (void) ggml_backend_sycl_model_unloaded_token(model);
    if (failures == failures_before) {
        std::printf("PASS: %s (%zu tensors, %.1f MiB each, planner_host=%s)\n", label, count,
                    bytes_per_tensor / (1024.0 * 1024.0), oracle ? "true" : "false");
    }
}

void check_clear_helper_contract() {
    bool has_plan = true;
    check(!ggml_sycl::clear_completed_multi_device_global_plan(has_plan, false),
          "single-device plan was reported as cleared");
    check(has_plan, "single-device plan was cleared");
    check(ggml_sycl::clear_completed_multi_device_global_plan(has_plan, true),
          "completed multi-device plan was not cleared");
    check(!has_plan, "completed multi-device plan remained active");
    check(!ggml_sycl::clear_completed_multi_device_global_plan(has_plan, true),
          "already-cleared plan was reported as cleared again");
}

void check_initial_cache_stats(ggml_backend_t backend) {
    ggml_backend_sycl_get_cache_stats(backend, nullptr, nullptr);

    uint64_t hits = 99;
    ggml_backend_sycl_get_cache_stats(backend, &hits, nullptr);
    check(hits == 0, "initial cache hits were not zero");

    uint64_t misses = 99;
    ggml_backend_sycl_get_cache_stats(backend, nullptr, &misses);
    check(misses == 0, "initial cache misses were not zero");

    hits   = 99;
    misses = 99;
    ggml_backend_sycl_get_cache_stats(backend, &hits, &misses);
    check(hits == 0 && misses == 0, "initial cache stats were not both zero");
}

bool arena_contract_available() {
    const char * arena = std::getenv("GGML_SYCL_VRAM_ARENA");
    if (arena && std::atoi(arena) == 0) {
        std::fprintf(stderr, "SKIP: tiered-dispatch requires the VRAM arena (GGML_SYCL_VRAM_ARENA=1)\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!arena_contract_available()) {
        return 77;
    }

    const int device_count = ggml_backend_sycl_get_device_count();
    if (device_count <= 0) {
        std::fprintf(stderr, "SKIP: no SYCL device\n");
        return 77;
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::fprintf(stderr, "FAIL: SYCL device was enumerated but backend initialization failed\n");
        return 1;
    }

    check_clear_helper_contract();
    check_initial_cache_stats(backend);

    size_t aggregate_device_bytes = 0;
    for (int device = 0; device < device_count; ++device) {
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        ggml_backend_sycl_get_device_memory(device, &free_bytes, &total_bytes);
        const size_t capacity = total_bytes > 0 ? total_bytes : free_bytes;
        if (aggregate_device_bytes <= std::numeric_limits<size_t>::max() - capacity) {
            aggregate_device_bytes += capacity;
        } else {
            aggregate_device_bytes = std::numeric_limits<size_t>::max();
        }
    }
    check(aggregate_device_bytes > 0, "SYCL devices reported no memory capacity");

    // Three model loads exercise both orderings. The 4 KiB inventory must fit
    // wholly on devices. Sixty-four equal slices collectively exceed aggregate
    // visible capacity, forcing planner host targets without overflowing size_t.
    run_placement_case(backend, "all-device-first", 4, 4096, false);

    constexpr size_t over_budget_count  = 64;
    const size_t     over_budget_tensor = aggregate_device_bytes / over_budget_count + 1;
    if (over_budget_tensor > std::numeric_limits<size_t>::max() / over_budget_count) {
        ggml_backend_free(backend);
        std::fprintf(stderr, "SKIP: aggregate device capacity cannot be exceeded by a representable fixture\n");
        return 77;
    }
    run_placement_case(backend, "over-budget-second", over_budget_count, over_budget_tensor, true);
    run_placement_case(backend, "all-device-third", 4, 4096, false);

    ggml_backend_free(backend);
    if (failures != 0) {
        std::fprintf(stderr, "%d tiered-dispatch check(s) failed\n", failures);
        return 1;
    }
    std::printf("All tiered-dispatch checks passed\n");
    return 0;
}
