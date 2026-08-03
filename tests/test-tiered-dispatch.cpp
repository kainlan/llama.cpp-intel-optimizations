//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Integration contract for the current-model planner-placement verdict.

#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
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

    inventory_fixture(const char * prefix, size_t count, size_t bytes_per_tensor) {
        names.reserve(count);
        tensors.reserve(count);
        size_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            names.emplace_back(std::string(prefix) + ".blk." + std::to_string(i) + ".attn_q.weight");
            ggml_sycl_tensor_info tensor{};
            tensor.name = names.back().c_str();
            tensor.size = bytes_per_tensor;
            tensors.push_back(tensor);
            total += bytes_per_tensor;
        }
        inventory.tensors    = tensors.data();
        inventory.count      = tensors.size();
        inventory.total_size = total;
        inventory.n_layer    = static_cast<uint32_t>(count);
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
                        bool           expect_host,
                        bool           exercise_multi_device_clear = false) {
    // This outer boundary must invalidate the preceding model's verdict before
    // the incoming inventory is known.
    ggml_backend_sycl_set_model_loading(true);
    check(!ggml_backend_sycl_is_tiered_enabled(backend), "outer model-load boundary did not reset verdict");

    inventory_fixture fixture(label, count, bytes_per_tensor);
    ggml_backend_sycl_set_tensor_inventory(backend, &fixture.inventory);

    const bool oracle  = planner_target_oracle(fixture, label);
    const bool verdict = ggml_backend_sycl_is_tiered_enabled(backend);
    check(oracle == expect_host, expect_host ? "over-budget planner oracle found no host target" :
                                               "all-device planner oracle found a host target");
    check(verdict == oracle, "tiered query disagrees with independent planned-target oracle");
    check(ggml_backend_sycl_has_tensor_cache(backend), "unified cache gate unavailable for planned model");

    if (exercise_multi_device_clear) {
        check(ggml_backend_sycl_has_active_placement_plan(), "planner plan was not active before completion clear");
        check(ggml_backend_sycl_test_complete_multi_device_plan_clear(),
              "production multi-device completion helper did not clear plan");
        check(!ggml_backend_sycl_has_active_placement_plan(),
              "multi-device completion left global placement plan active");
        check(ggml_backend_sycl_is_tiered_enabled(backend) == oracle,
              "planner-host verdict did not survive demonstrated global-plan clear");
    }

    ggml_backend_sycl_set_model_loading(false);
    check(ggml_backend_sycl_is_tiered_enabled(backend) == oracle,
          "planner verdict did not survive model-load completion");

    std::printf("PASS: %s (%zu tensors, %.1f MiB each, planner_host=%s)\n", label, count,
                bytes_per_tensor / (1024.0 * 1024.0), oracle ? "true" : "false");
}

}  // namespace

int main() {
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
    // wholly on devices. Sixty-four aggregate-visible-capacity tensors exceed
    // even GGML_SYCL_MAX_DEVICES device budgets, forcing planner host targets.
    run_placement_case(backend, "all-device-first", 4, 4096, false);
    const size_t over_budget_tensor = std::max<size_t>(aggregate_device_bytes, 1024 * 1024);
    run_placement_case(backend, "over-budget-second", 64, over_budget_tensor, true, true);
    run_placement_case(backend, "all-device-third", 4, 4096, false);

    ggml_backend_free(backend);
    if (failures != 0) {
        std::fprintf(stderr, "%d tiered-dispatch check(s) failed\n", failures);
        return 1;
    }
    std::printf("All tiered-dispatch checks passed\n");
    return 0;
}
