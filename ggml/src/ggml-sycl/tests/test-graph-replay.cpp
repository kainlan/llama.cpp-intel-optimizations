// Standalone test: Does L0 graph replay re-read from updated device memory?
// Tests 4 scenarios:
//   1. malloc_device input, bare replay (no update API)
//   2. malloc_host input, bare replay
//   3. malloc_device input, updatable graph + re-record + update()
//   4. dynamic_parameter<int*> with updatable graph
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -o test-graph-replay test-graph-replay.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-graph-replay

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>

namespace sycl_ex = sycl::ext::oneapi::experimental;

// Simple kernel: dst[0] = src[0] * 2
static void run_test_1_device_bare_replay(sycl::queue & q) {
    printf("\n=== Test 1: malloc_device input, bare replay ===\n");

    int * src = sycl::malloc_device<int>(1, q);
    int * dst = sycl::malloc_device<int>(1, q);

    // Initial value
    int val = 10;
    q.memcpy(src, &val, sizeof(int)).wait();

    // Record graph
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src[0] * 2;
        });
    });
    graph.end_recording();

    auto exec = graph.finalize();

    // Execute once
    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  First execute:  src=10, dst=%d (expect 20) %s\n", result, result == 20 ? "OK" : "FAIL");

    // Update src to 42
    val = 42;
    q.memcpy(src, &val, sizeof(int)).wait();

    // Replay
    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  Bare replay:    src=42, dst=%d (expect 84) %s\n", result, result == 84 ? "OK" : "FAIL");
    printf("  --> L0 graph %s from malloc_device on bare replay\n",
           result == 84 ? "DOES re-read" : "CACHES data (stale!)");

    sycl::free(src, q);
    sycl::free(dst, q);
}

static void run_test_2_host_bare_replay(sycl::queue & q) {
    printf("\n=== Test 2: malloc_host input, bare replay ===\n");

    int * src = sycl::malloc_host<int>(1, q);
    int * dst = sycl::malloc_device<int>(1, q);

    // Initial value
    src[0] = 10;

    // Record graph
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src[0] * 2;
        });
    });
    graph.end_recording();

    auto exec = graph.finalize();

    // Execute once
    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  First execute:  src=10, dst=%d (expect 20) %s\n", result, result == 20 ? "OK" : "FAIL");

    // Update src
    src[0] = 42;

    // Replay
    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  Bare replay:    src=42, dst=%d (expect 84) %s\n", result, result == 84 ? "OK" : "FAIL");
    printf("  --> L0 graph %s from malloc_host on bare replay\n",
           result == 84 ? "DOES re-read" : "CACHES data (stale!)");

    sycl::free(src, q);
    sycl::free(dst, q);
}

static void run_test_3_device_rerecord_update(sycl::queue & q) {
    printf("\n=== Test 3: malloc_device, updatable graph + re-record + update() ===\n");

    int * src = sycl::malloc_device<int>(1, q);
    int * dst = sycl::malloc_device<int>(1, q);

    int val = 10;
    q.memcpy(src, &val, sizeof(int)).wait();

    // Record initial graph (updatable)
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src[0] * 2;
        });
    });
    graph.end_recording();

    auto exec = graph.finalize(sycl_ex::property::graph::updatable{});

    // Execute once
    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  First execute:  src=10, dst=%d (expect 20) %s\n", result, result == 20 ? "OK" : "FAIL");

    // Update src
    val = 42;
    q.memcpy(src, &val, sizeof(int)).wait();

    // Re-record with same pointers (data changed)
    sycl_ex::command_graph graph2(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph2.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src[0] * 2;
        });
    });
    graph2.end_recording();

    // Update executable
    try {
        exec.update(graph2);
        q.ext_oneapi_graph(exec);
        q.wait();
        q.memcpy(&result, dst, sizeof(int)).wait();
        printf("  Re-record+update: src=42, dst=%d (expect 84) %s\n", result, result == 84 ? "OK" : "FAIL");
    } catch (const sycl::exception & e) {
        printf("  Re-record+update: EXCEPTION: %s\n", e.what());
    }

    sycl::free(src, q);
    sycl::free(dst, q);
}

static void run_test_4_dynamic_parameter(sycl::queue & q) {
    printf("\n=== Test 4: dynamic_parameter<int*> with updatable graph ===\n");

    int * src1 = sycl::malloc_device<int>(1, q);
    int * src2 = sycl::malloc_device<int>(1, q);
    int * dst  = sycl::malloc_device<int>(1, q);

    int val1 = 10, val2 = 42;
    q.memcpy(src1, &val1, sizeof(int)).wait();
    q.memcpy(src2, &val2, sizeof(int)).wait();

    // Create graph with dynamic parameter
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

    sycl_ex::dynamic_parameter dynSrc(graph, src1);

    graph.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.set_arg(0, dynSrc);
        h.set_arg(1, dst);
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src1[0] * 2;  // Note: uses captured src1, not dynamic
        });
    });
    graph.end_recording();

    auto exec = graph.finalize(sycl_ex::property::graph::updatable{});

    // Execute with src1
    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  First execute (src1=10): dst=%d (expect 20) %s\n", result, result == 20 ? "OK" : "FAIL");

    // Update dynamic param to point to src2
    try {
        dynSrc.update(src2);
        // Must re-record same structure then exec.update(graph) for whole-graph update
        sycl_ex::command_graph graph2(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
        graph2.begin_recording(q);
        q.submit([&](sycl::handler & h) {
            h.set_arg(0, dynSrc);
            h.set_arg(1, dst);
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                dst[0] = src2[0] * 2;  // captures src2 now
            });
        });
        graph2.end_recording();
        exec.update(graph2);
        q.ext_oneapi_graph(exec);
        q.wait();
        q.memcpy(&result, dst, sizeof(int)).wait();
        printf("  After dynamic update (src2=42): dst=%d (expect 84) %s\n", result, result == 84 ? "OK" : "FAIL");
    } catch (const sycl::exception & e) {
        printf("  dynamic_parameter update: EXCEPTION: %s\n", e.what());
    }

    sycl::free(src1, q);
    sycl::free(src2, q);
    sycl::free(dst, q);
}

static void run_test_5_device_data_update_same_ptr(sycl::queue & q) {
    printf("\n=== Test 5: malloc_device, same pointer, data updated, updatable + update() ===\n");

    int * src = sycl::malloc_device<int>(1, q);
    int * dst = sycl::malloc_device<int>(1, q);

    int val = 10;
    q.memcpy(src, &val, sizeof(int)).wait();

    // Record (updatable)
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src[0] * 2;
        });
    });
    graph.end_recording();
    auto exec = graph.finalize(sycl_ex::property::graph::updatable{});

    // Execute once
    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  First execute: src=10, dst=%d (expect 20) %s\n", result, result == 20 ? "OK" : "FAIL");

    // Update data at same src pointer
    val = 42;
    q.memcpy(src, &val, sizeof(int)).wait();

    // Re-record same structure graph, then exec.update(graph) — whole-graph update
    sycl_ex::command_graph graph2(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph2.begin_recording(q);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            dst[0] = src[0] * 2;
        });
    });
    graph2.end_recording();

    try {
        exec.update(graph2);
    } catch (const sycl::exception & e) {
        printf("  exec.update(graph): %s\n", e.what());
    }
    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, dst, sizeof(int)).wait();
    printf("  After re-record + update(): src=42, dst=%d (expect 84) %s\n",
           result, result == 84 ? "OK" : "FAIL");
    printf("  --> updatable graph + re-record %s from updated malloc_device\n",
           result == 84 ? "DOES re-read" : "still CACHES (stale!)");

    sycl::free(src, q);
    sycl::free(dst, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        printf("Testing L0 graph replay behavior with different memory types...\n");

        run_test_1_device_bare_replay(q);
        run_test_2_host_bare_replay(q);
        run_test_3_device_rerecord_update(q);
        run_test_5_device_data_update_same_ptr(q);

        // Test 4 may fail if dynamic_parameter isn't fully supported
        try {
            run_test_4_dynamic_parameter(q);
        } catch (const std::exception & e) {
            printf("\n=== Test 4: SKIPPED (exception: %s) ===\n", e.what());
        }

        printf("\nDone.\n");
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
