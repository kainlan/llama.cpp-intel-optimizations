// Stress test for unpin_on_event via binbcast ops.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=safe
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=barrier
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=reuse
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-unified-cache-unpin-event --mode=compare
//
// `compare` covers all three GGML_SYCL_BINBCAST_EVENT_MODE values and is both
// what ctest passes and what a bare invocation defaults to, so no mode is left
// ungated by either entry point.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

// Test hooks exported by ggml/src/ggml-sycl/binbcast.cpp.  Both read the
// GGML_SYCL_BINBCAST_EVENT_MODE environment variable through the same parser the
// backend uses, so they exercise the real policy rather than a copy of it.
const char * ggml_sycl_test_binbcast_event_mode_name();
const char * ggml_sycl_test_binbcast_event_source_name(bool kernel_event_valid);

namespace {

enum class event_mode {
    SAFE,
    BARRIER,
    REUSE,
    COMPARE,
};

const char * mode_name(event_mode mode) {
    switch (mode) {
        case event_mode::SAFE:
            return "safe";
        case event_mode::BARRIER:
            return "barrier";
        case event_mode::REUSE:
            return "reuse";
        case event_mode::COMPARE:
            return "compare";
        default:
            return "unknown";
    }
}

event_mode parse_mode(const char * mode_str) {
    if (!mode_str) {
        return event_mode::COMPARE;
    }
    if (std::strcmp(mode_str, "safe") == 0) {
        return event_mode::SAFE;
    }
    if (std::strcmp(mode_str, "barrier") == 0) {
        return event_mode::BARRIER;
    }
    if (std::strcmp(mode_str, "reuse") == 0) {
        return event_mode::REUSE;
    }
    if (std::strcmp(mode_str, "compare") == 0 || std::strcmp(mode_str, "both") == 0) {
        return event_mode::COMPARE;
    }
    return event_mode::COMPARE;
}

int parse_iters(const char * iters_str, int default_iters) {
    if (!iters_str || !*iters_str) {
        return default_iters;
    }
    const int value = std::atoi(iters_str);
    return value > 0 ? value : default_iters;
}

const char * get_arg(int argc, char ** argv, const char * prefix) {
    const size_t prefix_len = std::strlen(prefix);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], prefix, prefix_len) == 0) {
            return argv[i] + prefix_len;
        }
    }
    return nullptr;
}

// Host-only gate on the completion-event policy: no device work, so it also runs
// on a busy or absent GPU.
//
// The invariant: every (mode, kernel_event_valid) combination must resolve to a
// REAL completion event.  Both consumers of the binbcast completion event need
// one -- `ggml_sycl_set_tensor_ready_event` for GGML_OP_MUL, and
// `unified_cache::unpin_on_event`, which parks the weight-cache lease release on
// it.  A default-constructed sycl::event reads as ALREADY COMPLETE, so "no
// event" would unpin a weight the GPU is still reading (DEVICE_LOST or silent
// corruption) -- and it would look like a speedup.  "reuse" without a captured
// kernel event must therefore still fall back to a real submission.
static bool check_event_source_policy() {
    struct expectation {
        const char * env;                 // GGML_SYCL_BINBCAST_EVENT_MODE value, nullptr = unset
        const char * expect_mode;         // mode the backend parses out of it
        const char * expect_with_kernel;  // event source when a kernel event was captured
        const char * expect_no_kernel;    // event source when none was captured
    };

    const expectation cases[] = {
        { nullptr,   "barrier", "submission", "submission" },
        { "barrier", "barrier", "submission", "submission" },
        { "safe",    "safe",    "submission", "submission" },
        { "reuse",   "reuse",   "kernel",     "submission" },
        { "bogus",   "barrier", "submission", "submission" },
    };

    const char * saved     = std::getenv("GGML_SYCL_BINBCAST_EVENT_MODE");
    std::string  saved_str = saved ? saved : "";
    const bool   had_saved = saved != nullptr;

    bool ok = true;
    for (const expectation & c : cases) {
        if (c.env) {
            setenv("GGML_SYCL_BINBCAST_EVENT_MODE", c.env, 1);
        } else {
            unsetenv("GGML_SYCL_BINBCAST_EVENT_MODE");
        }

        const char * got_mode        = ggml_sycl_test_binbcast_event_mode_name();
        const char * got_with_kernel = ggml_sycl_test_binbcast_event_source_name(true);
        const char * got_no_kernel   = ggml_sycl_test_binbcast_event_source_name(false);

        if (std::strcmp(got_mode, c.expect_mode) != 0) {
            fprintf(stderr, "[policy] env=%s: mode is '%s', expected '%s'\n", c.env ? c.env : "(unset)", got_mode,
                    c.expect_mode);
            ok = false;
        }
        if (std::strcmp(got_with_kernel, c.expect_with_kernel) != 0) {
            fprintf(stderr, "[policy] env=%s kernel_event_valid=1: source is '%s', expected '%s'\n",
                    c.env ? c.env : "(unset)", got_with_kernel, c.expect_with_kernel);
            ok = false;
        }
        if (std::strcmp(got_no_kernel, c.expect_no_kernel) != 0) {
            fprintf(stderr, "[policy] env=%s kernel_event_valid=0: source is '%s', expected '%s'\n",
                    c.env ? c.env : "(unset)", got_no_kernel, c.expect_no_kernel);
            ok = false;
        }
    }

    if (had_saved) {
        setenv("GGML_SYCL_BINBCAST_EVENT_MODE", saved_str.c_str(), 1);
    } else {
        unsetenv("GGML_SYCL_BINBCAST_EVENT_MODE");
    }

    printf("Event source policy: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_binbcast_stress(event_mode mode, int iters) {
    const char * env_mode = mode_name(mode);
    setenv("GGML_SYCL_BINBCAST_EVENT_MODE", env_mode, 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to init SYCL backend\n");
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    if (!host_buft) {
        fprintf(stderr, "Failed to get SYCL host buffer type\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to init ggml context\n");
        ggml_backend_free(backend);
        return false;
    }

    const int64_t ne0 = 1024;
    const int64_t ne1 = 4;

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(weight, (std::string("weight.") + env_mode).c_str());

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(input, (std::string("input.") + env_mode).c_str());

    ggml_tensor * out = ggml_mul(ctx, input, weight);
    ggml_set_name(out, (std::string("out.") + env_mode).c_str());

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, out);

    const size_t weight_buf_size = ggml_backend_buft_get_alloc_size(host_buft, weight);
    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(host_buft, weight_buf_size);
    if (!weight_buf) {
        fprintf(stderr, "Failed to allocate weight buffer\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf));

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    std::vector<float> weight_data(ggml_nelements(weight));
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 1.0f + static_cast<float>(i % 7) * 0.01f;
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    std::vector<float> input_data(ggml_nelements(input));
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i % 13) * 0.02f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    bool ok = true;
    for (int i = 0; i < iters; ++i) {
        // Re-upload `input` every iteration. `ggml_gallocr` allocates `out`
        // IN-PLACE over `input` (same shape and type, and the mul is input's only
        // consumer), so without this each compute multiplies the PREVIOUS
        // result again and the tensor ends up holding input*weight^iters --
        // which no fixed expected value can check. Resetting it makes every
        // iteration's result
        // exactly input*weight while still driving `iters` pin/unpin cycles.
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[%s] graph compute failed at iter %d (%d)\n", env_mode, i, (int) status);
            ok = false;
            break;
        }
    }

    ggml_backend_sycl_submit_barrier(backend);
    ggml_backend_synchronize(backend);

    // Verify the result. An early pin release lets the cache evict or overwrite a
    // weight the kernel is still reading, which shows up here as wrong output --
    // the only symptom this test can observe from outside the backend. Throughput
    // alone would read an early unpin as a win.
    if (ok) {
        std::vector<float> out_data(ggml_nelements(out));
        ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

        int bad = 0;
        for (int64_t i1 = 0; i1 < ne1 && bad < 8; ++i1) {
            for (int64_t i0 = 0; i0 < ne0 && bad < 8; ++i0) {
                const size_t idx      = static_cast<size_t>(i1) * static_cast<size_t>(ne0) + static_cast<size_t>(i0);
                const float  expected = input_data[idx] * weight_data[i0];
                const float  actual   = out_data[idx];
                const float  diff     = std::fabs(actual - expected);
                if (diff > 1e-5f * (1.0f + std::fabs(expected))) {
                    fprintf(stderr, "[%s] mismatch at [%lld,%lld]: got %.9g expected %.9g\n", env_mode, (long long) i0,
                            (long long) i1, actual, expected);
                    ++bad;
                }
            }
        }
        if (bad > 0) {
            ok = false;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    const char * mode_arg  = get_arg(argc, argv, "--mode=");
    const char * iters_arg = get_arg(argc, argv, "--iters=");

    event_mode mode = parse_mode(mode_arg ? mode_arg : std::getenv("GGML_SYCL_UNPIN_EVENT_MODE"));
    int        iters = parse_iters(iters_arg ? iters_arg : std::getenv("GGML_SYCL_UNPIN_ITERS"), 200);

    // Host-only, so it runs first and reports even if device work later fails.
    bool ok = check_event_source_policy();

    if (mode == event_mode::COMPARE) {
        printf("Mode: compare (safe, barrier, reuse), iters=%d\n", iters);
        const event_mode modes[] = { event_mode::SAFE, event_mode::BARRIER, event_mode::REUSE };
        for (event_mode m : modes) {
            if (!ok) {
                break;
            }
            ok = run_binbcast_stress(m, iters);
        }
    } else if (ok) {
        printf("Mode: %s, iters=%d\n", mode_name(mode), iters);
        ok = run_binbcast_stress(mode, iters);
    }

    printf("\nUnified cache unpin event test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
