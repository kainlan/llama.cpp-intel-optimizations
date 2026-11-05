//
// test-sycl-plan-walk-sched: the plan walk's attribution over the REAL ggml-base scheduler (docs/design/
// sycl-mmap-host-tier.md §4.4 "The walk runs where the split is live"). The vehicle is a ggml_backend_sched over two
// CPU backend instances: the first, at index 0 (the higher priority), stands in for the SYCL backend the walk
// attributes to; the second is the scheduler's required last CPU backend. No GPU.
//
// The attribution function takes the target predicate as a parameter. Production passes ggml_backend_is_sycl; this
// test passes one that matches the stand-in by pointer (ggml_backend_is_sycl would reject a CPU stand-in).
//
// The graph's four nodes are SET_ROWS writing views of a tensor pre-allocated in a CPU buffer, so the split's first
// pass assigns each by its view_src buffer to the first backend supporting that buffer, the stand-in. No node is
// pinned with ggml_backend_sched_set_tensor_backend: reserve_size resets at its start, which would erase a pin.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "host-zone-plan-test-util.hpp"
#include "host-zone-plan.hpp"

using namespace ggml_sycl;
using namespace host_zone_plan_test;

namespace {

ggml_backend_t g_stand_in = nullptr;

bool is_stand_in(ggml_backend_t backend) {
    return backend == g_stand_in;
}

bool is_set_rows(void * user, const ggml_tensor * node) {
    (void) user;
    return node->op == GGML_OP_SET_ROWS;
}

}  // namespace

int main() {
    g_stand_in         = ggml_backend_cpu_init();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!g_stand_in || !cpu) {
        printf("FAIL: no CPU backend\n");
        return 1;
    }

    // The KV-like tensor, allocated in a CPU buffer before any split.
    ggml_init_params static_params = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context *   static_ctx    = ggml_init(static_params);
    ggml_tensor *    cache         = ggml_new_tensor_2d(static_ctx, GGML_TYPE_F32, 64, 16);
    ggml_set_name(cache, "cache");
    ggml_backend_buffer_t cache_buf = ggml_backend_alloc_ctx_tensors(static_ctx, g_stand_in);
    HZP_CHECK(cache_buf != nullptr, "the cache tensor could not be allocated in a CPU buffer");

    // Four SET_ROWS, each writing one row of a view of the cache.
    ggml_init_params graph_params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   graph_ctx    = ggml_init(graph_params);
    ggml_cgraph *    graph        = ggml_new_graph(graph_ctx);
    for (int i = 0; i < 4; ++i) {
        ggml_tensor * dst  = ggml_view_2d(graph_ctx, cache, 64, 4, cache->nb[1], (size_t) i * 4 * cache->nb[1]);
        ggml_tensor * rows = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 64, 1);
        ggml_tensor * idx  = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I64, 1);
        ggml_set_input(rows);
        ggml_set_input(idx);
        ggml_build_forward_expand(graph, ggml_set_rows(graph_ctx, dst, rows, idx));
    }

    ggml_backend_t       backends[2] = { g_stand_in, cpu };
    ggml_backend_sched_t sched       = ggml_backend_sched_new(backends, nullptr, 2, 256, false, false);
    size_t               sizes[2]    = { 0, 0 };

    // After ggml_backend_sched_reserve_size the split is live: all four nodes attribute to the stand-in.
    ggml_backend_sched_reserve_size(sched, graph, sizes);
    capture_begin();
    const host_plan_attribution live =
        host_plan_attribute(sched, graph, is_stand_in, is_set_rows, nullptr, nullptr, /*warn_zero=*/true);
    HZP_CHECK(live.attributed == 4, "after reserve_size: attributed %d of %d", live.attributed, live.matched);
    HZP_CHECK(!log_has("plan attributed 0"), "the zero-attribution WARN printed while the split was live");

    // After ggml_backend_sched_reserve the scheduler has reset: zero, and the zero-attribution WARN prints.
    ggml_backend_sched_reserve(sched, graph);
    capture_begin();
    const host_plan_attribution reset =
        host_plan_attribute(sched, graph, is_stand_in, is_set_rows, nullptr, nullptr, /*warn_zero=*/true);
    HZP_CHECK(reset.attributed == 0 && reset.matched == 4, "after reserve: attributed %d of %d, expected 0 of 4",
              reset.attributed, reset.matched);
    HZP_CHECK(log_has("[ATTN-HOST] plan attributed 0 host-KV nodes to a SYCL backend (4 matched the predicate)"),
              "after reserve: no zero-attribution WARN");

    // A second reserve_size on the same scheduler: four again.
    ggml_backend_sched_reserve_size(sched, graph, sizes);
    const host_plan_attribution again =
        host_plan_attribute(sched, graph, is_stand_in, is_set_rows, nullptr, nullptr, /*warn_zero=*/true);
    HZP_CHECK(again.attributed == 4, "after a second reserve_size: attributed %d of %d", again.attributed,
              again.matched);

    ggml_backend_sched_free(sched);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(cache_buf);
    ggml_free(static_ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(g_stand_in);
    return finish("test-sycl-plan-walk-sched");
}
