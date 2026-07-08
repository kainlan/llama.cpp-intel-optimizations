// Extended graph replay test: mimics llama.cpp TG patterns
// Tests whether chained kernels, pointer indirection, and various
// data flow patterns work correctly with L0 graph replay.
//
// Build:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -o test-graph-replay-chain test-graph-replay-chain.cpp
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./test-graph-replay-chain

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <functional>

namespace sycl_ex = sycl::ext::oneapi::experimental;

static bool check(const char * label, int got, int expected) {
    bool ok = (got == expected);
    printf("  %-40s got=%d expect=%d %s\n", label, got, expected, ok ? "OK" : "FAIL");
    return ok;
}

// Test 1: Chain of 3 kernels, input changes between replays
// Mimics: GET_ROWS -> MUL_MAT -> SOFT_MAX chain
static void test_chained_kernels(sycl::queue & q) {
    printf("\n=== Test 1: Chained kernels, input updated between replays ===\n");

    const int N = 16;
    int * input  = sycl::malloc_host<int>(N, q);     // like inp_tokens (host USM)
    int * scratch = sycl::malloc_device<int>(N, q);   // like compute scratch
    int * output = sycl::malloc_device<int>(N, q);    // like output buffer

    // Initial input: [1, 2, 3, ..., 16]
    for (int i = 0; i < N; i++) input[i] = i + 1;

    // Record graph: 3 chained kernels
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);

    // Kernel 1: read from host input, write to scratch (like GET_ROWS)
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        scratch[i] = input[i] * 2;
    });
    // Kernel 2: read scratch, write back (like MUL_MAT)
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        scratch[i] = scratch[i] + 100;
    });
    // Kernel 3: read scratch, write output (like SOFT_MAX)
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        output[i] = scratch[i] * 3;
    });

    graph.end_recording();
    auto exec = graph.finalize();

    // Execute
    q.ext_oneapi_graph(exec);
    q.wait();
    int result[N];
    q.memcpy(result, output, N * sizeof(int)).wait();
    // Expected: ((1*2)+100)*3 = 306, ((2*2)+100)*3 = 312, ...
    check("First execute [0]", result[0], 306);
    check("First execute [1]", result[1], 312);

    // Update input: [10, 20, 30, ..., 160]
    for (int i = 0; i < N; i++) input[i] = (i + 1) * 10;

    // Bare replay
    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(result, output, N * sizeof(int)).wait();
    // Expected: ((10*2)+100)*3 = 360, ((20*2)+100)*3 = 420, ...
    check("Replay [0] (expect fresh)", result[0], 360);
    check("Replay [1] (expect fresh)", result[1], 420);

    sycl::free(input, q);
    sycl::free(scratch, q);
    sycl::free(output, q);
}

// Test 2: Pointer resolved through indirection table (mimics ggml_sycl_get_data_ptr)
// The kernel receives a pointer looked up from a table, not directly captured
static void test_pointer_indirection(sycl::queue & q) {
    printf("\n=== Test 2: Pointer resolved through indirection (like get_data_ptr) ===\n");

    int * src_host = sycl::malloc_host<int>(1, q);
    int * dst_dev  = sycl::malloc_device<int>(1, q);

    // Simulate pointer resolution: a "registry" that maps tensor->resolved_ptr
    // In llama.cpp, ggml_sycl_get_data_ptr returns this
    void * resolved_ptr = src_host;  // resolved to same host address

    src_host[0] = 7;

    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);

    // Kernel uses resolved_ptr (captured by value in lambda)
    int * src_captured = static_cast<int *>(resolved_ptr);
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        dst_dev[0] = src_captured[0] * 3;
    });

    graph.end_recording();
    auto exec = graph.finalize();

    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, dst_dev, sizeof(int)).wait();
    check("First execute (src=7)", result, 21);

    // Update data at the SAME host address
    src_host[0] = 99;

    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, dst_dev, sizeof(int)).wait();
    check("Replay (src=99, expect 297)", result, 297);

    sycl::free(src_host, q);
    sycl::free(dst_dev, q);
}

// Test 3: Host data copied to device staging before recording,
// then staging refreshed before replay (EXACT llama.cpp pattern)
static void test_staged_input(sycl::queue & q) {
    printf("\n=== Test 3: Host→device staging before recording, refresh before replay ===\n");

    int * host_buf    = sycl::malloc_host<int>(1, q);    // ggml scheduler buffer
    int * staging_dev = sycl::malloc_device<int>(1, q);   // prestage device copy
    int * output_dev  = sycl::malloc_device<int>(1, q);

    // Initial data
    host_buf[0] = 5;
    q.memcpy(staging_dev, host_buf, sizeof(int)).wait();  // prestage

    // Record graph reading from staging_dev (not host_buf)
    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        output_dev[0] = staging_dev[0] * 4;
    });
    graph.end_recording();
    auto exec = graph.finalize();

    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, output_dev, sizeof(int)).wait();
    check("First execute (staged=5)", result, 20);

    // Update host, refresh staging, replay
    host_buf[0] = 50;
    q.memcpy(staging_dev, host_buf, sizeof(int)).wait();  // refresh staging

    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, output_dev, sizeof(int)).wait();
    check("After refresh replay (staged=50)", result, 200);

    sycl::free(host_buf, q);
    sycl::free(staging_dev, q);
    sycl::free(output_dev, q);
}

// Test 4: Large graph (100 kernels) with input dependency
static void test_large_graph(sycl::queue & q) {
    printf("\n=== Test 4: Large graph (100 chained kernels) ===\n");

    int * input  = sycl::malloc_host<int>(1, q);
    int * buf    = sycl::malloc_device<int>(1, q);
    int * output = sycl::malloc_device<int>(1, q);

    input[0] = 1;

    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);

    // First kernel: read from host input
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        buf[0] = input[0];
    });

    // 98 intermediate kernels: buf += 1
    for (int k = 0; k < 98; k++) {
        q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            buf[0] = buf[0] + 1;
        });
    }

    // Final kernel: write to output
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        output[0] = buf[0];
    });

    graph.end_recording();
    auto exec = graph.finalize();

    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, output, sizeof(int)).wait();
    check("First execute (input=1, +98)", result, 99);

    // Update input
    input[0] = 1000;

    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, output, sizeof(int)).wait();
    check("Replay (input=1000, +98)", result, 1098);

    sycl::free(input, q);
    sycl::free(buf, q);
    sycl::free(output, q);
}

// Test 5: memcpy inside the recorded graph (mimics set_tensor_async during recording)
static void test_memcpy_in_graph(sycl::queue & q) {
    printf("\n=== Test 5: queue.memcpy() inside recorded graph ===\n");

    int * host_src = sycl::malloc_host<int>(1, q);
    int * dev_buf  = sycl::malloc_device<int>(1, q);
    int * output   = sycl::malloc_device<int>(1, q);

    host_src[0] = 7;

    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);

    // memcpy inside recording (like set_tensor_async during compute_impl)
    q.memcpy(dev_buf, host_src, sizeof(int));

    // Kernel reads from dev_buf
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        output[0] = dev_buf[0] * 5;
    });

    graph.end_recording();
    auto exec = graph.finalize();

    q.ext_oneapi_graph(exec);
    q.wait();
    int result = 0;
    q.memcpy(&result, output, sizeof(int)).wait();
    check("First execute (host=7)", result, 35);

    // Update host data
    host_src[0] = 42;

    // Replay — does the memcpy re-read from host?
    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, output, sizeof(int)).wait();
    check("Replay (host=42, expect 210)", result, 210);
    printf("  --> memcpy inside graph %s from updated host\n",
           result == 210 ? "DOES re-read" : "uses STALE data");

    sycl::free(host_src, q);
    sycl::free(dev_buf, q);
    sycl::free(output, q);
}

// Test 6: Simulate the exact llama.cpp pattern
// - Warmup pass (compute_impl without recording)
// - Recording pass (compute_impl with recording)
// - Multiple replay passes with input update + refresh
static void test_llama_pattern(sycl::queue & q) {
    printf("\n=== Test 6: Full llama.cpp pattern (warmup → record → replay×N) ===\n");

    const int N = 4;
    int * inp_tokens = sycl::malloc_host<int>(1, q);    // token ID
    int * inp_pos    = sycl::malloc_host<int>(1, q);    // position
    int * weights    = sycl::malloc_device<int>(N, q);   // model weights (stable)
    int * kv_cache   = sycl::malloc_device<int>(64, q);  // KV cache (grows)
    int * scratch    = sycl::malloc_device<int>(N, q);   // compute buffer
    int * output     = sycl::malloc_device<int>(1, q);

    // Initialize weights
    int w[N] = {1, 2, 3, 4};
    q.memcpy(weights, w, N * sizeof(int)).wait();
    q.memset(kv_cache, 0, 64 * sizeof(int)).wait();

    // Lambda simulating compute_impl
    auto compute = [&]() {
        int * tok = inp_tokens;
        int * pos = inp_pos;

        // Kernel 1: embedding lookup (token → scratch)
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            scratch[i] = weights[i] * tok[0];
        });
        // Kernel 2: write to KV cache at position
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            kv_cache[pos[0] * N + i] = scratch[i];
        });
        // Kernel 3: read KV cache, produce output
        q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            int sum = 0;
            for (int p = 0; p <= pos[0]; p++) {
                for (int j = 0; j < N; j++) {
                    sum += kv_cache[p * N + j];
                }
            }
            output[0] = sum;
        });
    };

    // --- Warmup (iteration 0, token=6, pos=0) ---
    inp_tokens[0] = 6;
    inp_pos[0] = 0;
    compute();
    q.wait();
    int result = 0;
    q.memcpy(&result, output, sizeof(int)).wait();
    // KV[0] = weights*6 = [6,12,18,24], sum = 60
    check("Warmup (tok=6, pos=0)", result, 60);

    // --- Record (iteration 1, token=7, pos=1) ---
    inp_tokens[0] = 7;
    inp_pos[0] = 1;

    sycl_ex::command_graph graph(q, {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
    graph.begin_recording(q);
    compute();
    graph.end_recording();
    auto exec = graph.finalize();

    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, output, sizeof(int)).wait();
    // KV[0] = [6,12,18,24], KV[1] = [7,14,21,28], sum = 60+70 = 130
    check("Record+execute (tok=7, pos=1)", result, 130);

    // --- Replay 1 (iteration 2, token=8, pos=2) ---
    inp_tokens[0] = 8;
    inp_pos[0] = 2;

    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, output, sizeof(int)).wait();
    // KV[2] = [8,16,24,32], sum = 60+70+80 = 210
    check("Replay 1 (tok=8, pos=2)", result, 210);

    // --- Replay 2 (iteration 3, token=9, pos=3) ---
    inp_tokens[0] = 9;
    inp_pos[0] = 3;

    q.ext_oneapi_graph(exec);
    q.wait();
    q.memcpy(&result, output, sizeof(int)).wait();
    // KV[3] = [9,18,27,36], sum = 60+70+80+90 = 300
    check("Replay 2 (tok=9, pos=3)", result, 300);

    sycl::free(inp_tokens, q);
    sycl::free(inp_pos, q);
    sycl::free(weights, q);
    sycl::free(kv_cache, q);
    sycl::free(scratch, q);
    sycl::free(output, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

        test_chained_kernels(q);
        test_pointer_indirection(q);
        test_staged_input(q);
        test_large_graph(q);
        test_memcpy_in_graph(q);
        test_llama_pattern(q);

        printf("\nAll tests complete.\n");
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
