// GPU regression test for llama.cpp-yke2: test translation units that include
// ggml-sycl/common.hpp (and transitively ggml-sycl/dpct/helper.hpp) compile
// their OWN copy of dpct::device_ext's header-only inline queue-construction
// functions (init_queues()/create_queue_impl()) WITHOUT DPCT_PROFILING_ENABLED
// -- that macro is a PRIVATE compile definition of the ggml-sycl CMake target
// (ggml/src/ggml-sycl/CMakeLists.txt) -- and ordinary ELF symbol resolution
// lets the executable's copy interpose over the library's copy for the whole
// process. Under exactly one visible GPU (every canonical
// ONEAPI_DEVICE_SELECTOR=level_zero:N run) that interposition silently
// dropped sycl::property::queue::enable_profiling from the unified cache's
// owner queue, because create_cache_for_device() (unified-cache.cpp) used to
// fall back to dpct's raw, header-cached ggml_sycl_get_device(dev).
// default_queue() whenever total_gpus <= 1.
//
// Discovered at llama.cpp-6f73 (comment c-1rsj) while diagnosing why the G4
// numerics test needed a PRIVATE profiling-enabled queue of its own
// (tests/test-sycl-mxfp4-stored-gemm-soa-small-m.cpp) even though it shares
// ctx->stream() with production dispatch. This test is the general
// regression gate for that root cause: it deliberately compiles its own copy
// of the same dpct inlines (by including ggml-sycl/common.hpp, exactly like
// every other affected GPU test TU -- see the census on llama.cpp-yke2) and
// checks the queue the PUBLIC dispatch path actually hands out --
// ggml_backend_sycl_context::stream() -- rather than constructing a private
// workaround queue the way the G4 test had to.
//
// Fixed in create_cache_for_device() (ggml/src/ggml-sycl/unified-cache.cpp,
// llama.cpp-yke2): the cache's owner queue is now ALWAYS built via
// ensure_single_device_context_queue() (`new sycl::queue(ctx, dev,
// default_queue_properties())`), which never touches dpct's cached
// default_queue() machinery and so cannot be affected by which TU's copy of
// the header-only inlines the linker resolves. Before that fix this test
// FAILS on a single-visible-GPU run (has_property(enable_profiling)=0);
// after it, it PASSES. See tests/test-sycl-profiling-queue-property-source.py
// for the accompanying source-level invariant.
//
// NOT wired into any production dispatch path beyond what
// ggml_backend_sycl_init()/ggml_backend_sycl_context::stream() already do --
// this test calls no kernel and modifies no tensor; it only inspects the
// queue object production dispatch would submit work on.
//
// GPU and model-loading binaries in this fork are run only from the lead
// session, one at a time (CLAUDE.md, Hard-Won Rules) -- this binary is no
// exception:
//
//   source /opt/intel/oneapi/setvars.sh --force
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-profiling-queue-property   # B50
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-profiling-queue-property   # B70

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "test-skip.h"

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int, char ** argv) {
    // tests/sycl-selector-fallback.hpp (task/S6, llama.cpp-5q1r) is not yet on
    // this branch, so this is the canonical inline re-exec block used by the
    // sibling GPU tests in this directory (e.g.
    // test-sycl-mxfp4-stored-gemm-soa-small-m.cpp on task/G4): ctest supplies
    // ONEAPI_DEVICE_SELECTOR via the registration's ENVIRONMENT, but a bare
    // invocation needs it set before libsycl memoizes device enumeration.
    // setenv() alone is too late -- libccl's static initializer constructs a
    // sycl::event at load, which makes libsycl memoize the selector before
    // main() runs (llama.cpp-2x3m). Re-exec so the child process starts with
    // it already set (llama.cpp-403s: left unpinned, the iGPU's 231 GB
    // "VRAM" is claimed and this test's queue-property check is no longer
    // exercising the discrete-card single-GPU path it is meant to cover).
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        if (setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1) == 0) {
            execv("/proc/self/exe", argv);
        }
        std::fprintf(stderr, "warning: re-exec failed (%s); run with ONEAPI_DEVICE_SELECTOR=level_zero:1 set\n",
                     std::strerror(errno));
    }

    // Requested by the plan (llama.cpp-yke2): exercise the same env-gated
    // configuration a real profiling run uses. The queue PROPERTY under test
    // (enable_profiling) is unconditional in default_queue_properties() and
    // does not itself depend on this variable, but setting it documents that
    // the fix holds under the configuration it is meant to serve, and
    // matches the G4 test's own profiling_requested gate.
    setenv("GGML_SYCL_KERNEL_PROFILE", "1", 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    // The PUBLIC dispatch path, not a private workaround queue: this is
    // exactly the queue_ptr production ggml_backend_graph_compute() submits
    // work on for this backend context (ggml_backend_sycl_context::stream(),
    // ggml-sycl/common.hpp), which resolves to the unified cache's owner
    // queue (unified_cache::get_queue()) once the cache for this device
    // exists.
    auto *        sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);
    sycl::queue & q        = *sycl_ctx->stream();

    const bool has_profiling = q.has_property<sycl::property::queue::enable_profiling>();
    const bool is_in_order   = q.has_property<sycl::property::queue::in_order>();
    std::printf("queue diag: has_property(enable_profiling)=%d has_property(in_order)=%d\n", (int) has_profiling,
                (int) is_in_order);

    ggml_backend_free(backend);

    if (!has_profiling) {
        std::printf(
            "FAIL: ggml_backend_sycl_context::stream() lacks sycl::property::queue::enable_profiling under a "
            "single visible GPU -- see llama.cpp-yke2\n");
        return 1;
    }
    std::printf("PASS: single-GPU unified-cache queue carries enable_profiling\n");
    return 0;
}
