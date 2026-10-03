// llama.cpp-5efe C4 (part b): a stream_dma refusal of a file-mapped source is terminal in get_rows' slice loop.
//
// get_rows streams a host-resident table through the DMA pool, and its slice loop halves the slice on any non-ok
// result (`get_rows_dma_retry`, which getrows.cpp's streaming path runs). A deterministic refusal (the source is a
// file mapping, so a device read of it is forbidden whatever the slice size) would repeat log2(slice / min) times, each
// attempt counted by stream_dma and each logging "[GET_ROWS] DMA retry", and the CPU fallback keys on
// mmap_direct_failed, which a refusal does not set. The loop breaks on refused_file_mapped before its retry test and
// the caller routes it to the CPU executor.
//
// Pure: the loop is the real one, driven over a fake stream_dma; the registry predicate stream_dma reads is the real
// alloc_registry. No device, no queue.

#include "../alloc-registry.hpp"
#include "../getrows-dma-retry.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using ggml_sycl::alloc_info;
using ggml_sycl::alloc_registry;
using ggml_sycl::alloc_type;
using ggml_sycl::get_rows_dma_attempt;
using ggml_sycl::get_rows_dma_outcome;
using ggml_sycl::get_rows_dma_retry;

static int g_checks = 0;
static int g_fail   = 0;

static void check_eq(const char * what, size_t got, size_t want) {
    g_checks++;
    if (got != want) {
        g_fail++;
        printf("  FAIL: %s: got %zu, want %zu\n", what, got, want);
    }
}

static void check_true(const char * what, bool cond) {
    g_checks++;
    if (!cond) {
        g_fail++;
        printf("  FAIL: %s\n", what);
    }
}

constexpr size_t MiB = 1024u * 1024u;

// A stand-in for unified_cache::stream_dma: it counts every call and every refusal the way the real one does, and
// answers from a script (one entry per call; the last entry repeats).
struct fake_stream_dma {
    std::vector<get_rows_dma_attempt> script;
    size_t                            calls    = 0;
    size_t                            refusals = 0;
    std::vector<size_t>               slices;
    std::vector<size_t>               buffers;

    get_rows_dma_attempt operator()(size_t slice_bytes, size_t buffer_count) {
        const get_rows_dma_attempt answer = script[calls < script.size() ? calls : script.size() - 1];
        calls++;
        slices.push_back(slice_bytes);
        buffers.push_back(buffer_count);
        if (answer.refused_file_mapped) {
            refusals++;
        }
        return answer;
    }
};

struct retry_log {
    size_t lines = 0;
};

static get_rows_dma_outcome run(fake_stream_dma & dma, retry_log & log, size_t slice, size_t min_slice, size_t limit) {
    return get_rows_dma_retry(
        slice, 2, min_slice, limit, [&](size_t s, size_t b) { return dma(s, b); },
        [&](size_t bytes) { return bytes < min_slice ? min_slice : bytes; },
        [&](size_t, size_t, size_t) { log.lines++; });
}

static get_rows_dma_attempt refused() {
    get_rows_dma_attempt a;
    a.refused_file_mapped = true;
    return a;
}

static get_rows_dma_attempt failed() {
    return get_rows_dma_attempt{};
}

static get_rows_dma_attempt ok() {
    get_rows_dma_attempt a;
    a.ok = true;
    return a;
}

static get_rows_dma_attempt oom() {
    get_rows_dma_attempt a;
    a.device_oom = true;
    return a;
}

static void test_a_refusal_is_one_call_one_count_no_retry_line() {
    printf("(a) a refusal: one stream_dma call, one counted refusal, no retry line\n");
    fake_stream_dma dma;
    dma.script = { refused() };
    retry_log log;
    // 64 MiB slices down to a 1 MiB floor, two retries allowed: the old loop would call stream_dma three times.
    const get_rows_dma_outcome out = run(dma, log, 64 * MiB, 1 * MiB, 2);
    check_eq("stream_dma calls", dma.calls, 1);
    check_eq("counted refusals", dma.refusals, 1);
    check_eq("retry lines", log.lines, 0);
    check_true("the outcome names the refusal", out.refused_file_mapped);
    check_true("not streamed", !out.streamed_ok);
    check_eq("attempts", out.attempts, 1);
}

static void test_the_refusal_is_terminal_even_on_a_later_attempt() {
    printf("(b) an ordinary failure retries, and a refusal on the retry still ends the loop\n");
    fake_stream_dma dma;
    dma.script = { failed(), refused() };
    retry_log log;
    const get_rows_dma_outcome out = run(dma, log, 64 * MiB, 1 * MiB, 5);
    check_eq("stream_dma calls", dma.calls, 2);
    check_eq("counted refusals", dma.refusals, 1);
    check_eq("retry lines", log.lines, 1);
    check_true("refused", out.refused_file_mapped && !out.streamed_ok);
}

static void test_ordinary_failures_still_halve_and_retry() {
    printf("(c) control: a non-refusal failure halves the slice and retries to the limit\n");
    fake_stream_dma dma;
    dma.script = { failed() };
    retry_log log;
    const get_rows_dma_outcome out = run(dma, log, 64 * MiB, 1 * MiB, 2);
    check_eq("stream_dma calls", dma.calls, 3);
    check_eq("retry lines", log.lines, 2);
    check_true("not refused, not streamed", !out.refused_file_mapped && !out.streamed_ok);
    check_eq("first slice", dma.slices[0], 64 * MiB);
    check_eq("second slice", dma.slices[1], 32 * MiB);
    check_eq("third slice", dma.slices[2], 16 * MiB);
    check_eq("buffers on the first call", dma.buffers[0], 2);
    check_eq("buffers on a retry", dma.buffers[1], 1);
}

static void test_a_classified_device_oom_retries_and_success_ends_the_loop() {
    printf("(d) a device out-of-memory retries at a smaller slice, and success ends the loop\n");
    fake_stream_dma dma;
    dma.script = { oom(), ok() };
    retry_log log;
    const get_rows_dma_outcome out = run(dma, log, 64 * MiB, 1 * MiB, 2);
    check_eq("stream_dma calls", dma.calls, 2);
    check_eq("retry lines", log.lines, 1);
    check_true("streamed", out.streamed_ok && !out.refused_file_mapped);
    check_eq("final slice", out.final_slice_bytes, 32 * MiB);
}

static void test_a_failure_at_the_floor_ends_the_loop() {
    printf("(e) a failure at the minimum slice does not retry\n");
    fake_stream_dma dma;
    dma.script = { failed() };
    retry_log log;
    const get_rows_dma_outcome out = run(dma, log, 1 * MiB, 1 * MiB, 2);
    check_eq("stream_dma calls", dma.calls, 1);
    check_eq("retry lines", log.lines, 0);
    check_true("neither streamed nor refused", !out.streamed_ok && !out.refused_file_mapped);
}

// stream_dma refuses a source the registry knows as a file mapping, before the copy_fn test: this is the predicate it
// reads. An interior pointer of the mapping counts, every other tier and an unregistered pointer do not.
static void test_the_registry_predicate_stream_dma_reads() {
    printf("(f) alloc_registry::is_file_mapped: the mapping and its interior, and nothing else\n");
    alloc_registry & reg = alloc_registry::instance();
    reg.clear();
    alignas(64) static char bytes[4096];
    char *                  base = bytes;
    reg.register_alloc(base, 1024, -1, alloc_type::FILE_MAPPED);
    reg.register_alloc(base + 1024, 1024, 0, alloc_type::DEVICE);
    reg.register_alloc(base + 2048, 512, -1, alloc_type::HOST_PINNED);
    reg.register_alloc(base + 2560, 512, -1, alloc_type::HOST_CPU);

    check_true("the mapping's base", reg.is_file_mapped(base));
    check_true("an interior pointer of the mapping", reg.is_file_mapped(base + 777));
    check_true("a device allocation", !reg.is_file_mapped(base + 1024));
    check_true("a pinned host allocation", !reg.is_file_mapped(base + 2048));
    check_true("a HOST_CPU allocation", !reg.is_file_mapped(base + 2560));
    check_true("an unregistered pointer", !reg.is_file_mapped(base + 3584));
    check_true("a null pointer", !reg.is_file_mapped(nullptr));
    reg.unregister_alloc(base);
    check_true("after the mapping is unregistered", !reg.is_file_mapped(base + 777));
    reg.clear();
}

int main() {
    printf("=== a stream_dma refusal is terminal in get_rows' slice loop (llama.cpp-5efe C4 part b) ===\n");
    test_a_refusal_is_one_call_one_count_no_retry_line();
    test_the_refusal_is_terminal_even_on_a_later_attempt();
    test_ordinary_failures_still_halve_and_retry();
    test_a_classified_device_oom_retries_and_success_ends_the_loop();
    test_a_failure_at_the_floor_ends_the_loop();
    test_the_registry_predicate_stream_dma_reads();
    printf("=== %d checks, %d failures ===\n", g_checks, g_fail);
    if (g_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
