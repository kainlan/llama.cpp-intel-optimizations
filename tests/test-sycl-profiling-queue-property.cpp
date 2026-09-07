// GPU regression test for llama.cpp-yke2: test translation units that include
// ggml-sycl/common.hpp (and transitively ggml-sycl/dpct/helper.hpp) compile
// their OWN copy of dpct::device_ext's header-only inline queue-construction
// functions (init_queues()/create_queue_impl()) WITHOUT DPCT_PROFILING_ENABLED
// -- that macro is a PRIVATE compile definition of the ggml-sycl CMake target
// (ggml/src/ggml-sycl/CMakeLists.txt) -- and ordinary ELF symbol resolution
// lets the executable's copy interpose over the library's copy for the whole
// process. Under exactly one visible GPU (every canonical
// ONEAPI_DEVICE_SELECTOR=level_zero:N run) that interposition silently
// dropped sycl::property::queue::enable_profiling from EVERY queue built
// through dpct::device_ext::default_queue() -- not just the unified cache's
// owner queue, but also ggml_backend_sycl_context::stream()'s own pre-cache
// fallback (ggml-sycl/common.hpp:5586-5589), which this test's PHASE 1
// exercises directly (no tensor is allocated yet, so no cache exists for the
// device, and stream() falls through to
// `&(ggml_sycl_get_device(device).default_queue())`).
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
// FIXED in dpct::device_ext::create_queue_impl() (ggml/src/ggml-sycl/dpct/
// helper.hpp:888, :903, llama.cpp-yke2): sycl::property::queue::
// enable_profiling() is now added UNCONDITIONALLY, not `#ifdef
// DPCT_PROFILING_ENABLED` -- every TU (this one included) now compiles an
// identical copy of these inlines, so it no longer matters which TU's copy
// the linker resolves. Before that fix, PHASE 1 below FAILS on a
// single-visible-GPU run (has_property(enable_profiling)=0); after it, it
// PASSES. This is the fix PHASE 1 actually exercises -- an earlier,
// narrower attempt at this ticket instead removed a `total_gpus > 1` gate on
// ensure_single_device_context_queue() in
// ggml/src/ggml-sycl/unified-cache.cpp's create_cache_for_device(), which is
// a real, independently-motivated fix (it stops the unified cache's OWN
// queue construction from ever touching dpct's default_queue() machinery at
// all, regardless of the helper.hpp fix) but does NOT by itself turn PHASE 1
// green, since PHASE 1 never reaches the cache -- confirmed on hardware
// (has_property(enable_profiling)=0 on both cards) before the helper.hpp fix
// was added. PHASE 2 below is what actually exercises that cache-side change:
// it forces cache creation and checks the resulting queue is the cache's
// own, distinct queue object, and that IT ALSO carries the property (which,
// after the helper.hpp fix, would be true even without the unified-cache.cpp
// change -- the two fixes are independent, and this test's two phases are
// what keep either one from silently regressing without detection: reverting
// PHASE 1's fix alone re-fails PHASE 1, and reverting PHASE 2's fix alone
// would leave the cache's queue coming from
// ensure_single_device_context_queue() only when >1 GPU is visible, still
// passing PHASE 2's property check post-helper.hpp-fix but on the WRONG
// queue construction path for a single-GPU run -- see PHASE 2's own comment
// for the identity check that catches that).
//
// See tests/test-sycl-profiling-queue-property-source.py for the
// accompanying source-level invariants on both fixes.
//
// NOT wired into any production dispatch path beyond what
// ggml_backend_sycl_init()/ggml_backend_sycl_context::stream() and a single
// small tensor allocation already do -- this test launches no kernel.
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
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"
#include "test-skip.h"

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool check_queue_properties(const char * label, sycl::queue & q) {
    const bool has_profiling = q.has_property<sycl::property::queue::enable_profiling>();
    const bool is_in_order   = q.has_property<sycl::property::queue::in_order>();
    std::printf("queue diag [%s]: has_property(enable_profiling)=%d has_property(in_order)=%d addr=%p\n", label,
                (int) has_profiling, (int) is_in_order, (void *) &q);
    if (!has_profiling) {
        std::printf("FAIL [%s]: queue lacks sycl::property::queue::enable_profiling -- see llama.cpp-yke2\n", label);
    }
    return has_profiling;
}

}  // namespace

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
    // (enable_profiling) is unconditional in both fixes and does not itself
    // depend on this variable, but setting it documents that the fix holds
    // under the configuration it is meant to serve, and matches the G4
    // test's own profiling_requested gate.
    setenv("GGML_SYCL_KERNEL_PROFILE", "1", 1);

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

    // PHASE 1: the PUBLIC dispatch path BEFORE any cache exists for this
    // device. No tensor has been allocated yet, so
    // ggml_backend_sycl_context::stream() (ggml-sycl/common.hpp) falls
    // through its pre-cache branch to
    // `&(ggml_sycl_get_device(device).default_queue())` -- dpct's raw,
    // interposition-vulnerable queue. This is the branch the helper.hpp fix
    // actually turns green; see the file header comment above.
    sycl::queue & q_pre_cache      = *sycl_ctx->stream();
    const bool    pre_cache_ok     = check_queue_properties("phase1-pre-cache", q_pre_cache);
    void *        q_pre_cache_addr = (void *) &q_pre_cache;

    // Force cache creation for this device the same way real weight/compute
    // allocation does: ggml_backend_sycl_buffer_type_alloc_buffer()
    // (ggml-sycl.cpp) unconditionally calls the CREATING accessor
    // ggml_sycl::get_unified_cache_for_device() before it does anything
    // else, for every allocation regardless of KV/compute/weights policy --
    // so a single small buffer allocation through the backend's default
    // buffer type is sufficient, without needing to load a real model (the
    // heavier approach test-sycl-two-context-ownership.cpp uses).
    ggml_init_params params = { 4 * 1024, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL: ggml_init failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t, "profiling_queue_property_probe");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t               size = ggml_backend_buft_get_alloc_size(buft, t);
    ggml_backend_buffer_t      buf  = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buf) {
        std::printf("FAIL: failed to allocate a device buffer to force cache creation\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    if (ggml_backend_tensor_alloc(buf, t, ggml_backend_buffer_get_base(buf)) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL: ggml_backend_tensor_alloc failed\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // PHASE 2: the PUBLIC dispatch path AFTER a cache now exists for this
    // device. stream()'s cache branch runs first on every call
    // (ggml-sycl/common.hpp), so this must now return the cache's OWN queue
    // -- a DIFFERENT sycl::queue object from PHASE 1's (identity-checked
    // below via unified_cache::get_queue(), not just another property read),
    // which is exactly the queue create_cache_for_device()
    // (unified-cache.cpp) constructs. Checking identity, not just the
    // property, is what would catch a regression of the unified-cache.cpp
    // fix specifically: reverting it re-introduces the `total_gpus > 1`
    // gate, so on a single-GPU run the cache's queue would come from
    // ggml_sycl_get_device(device_id).default_queue() again -- the SAME
    // object PHASE 1 already read -- even though, post-helper.hpp-fix, that
    // object would still (correctly) carry enable_profiling and so would
    // not be caught by a property check alone.
    ggml_sycl::unified_cache * cache = ggml_sycl::get_existing_unified_cache_for_device(sycl_ctx->device);
    if (!cache) {
        std::printf("FAIL: expected a unified cache to exist for device %d after a buffer allocation\n",
                    sycl_ctx->device);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    sycl::queue & q_post_cache  = *sycl_ctx->stream();
    const bool    post_cache_ok = check_queue_properties("phase2-post-cache", q_post_cache);

    const bool is_cache_queue = &q_post_cache == &cache->get_queue();
    std::printf("queue diag [phase2-post-cache]: is_cache_queue=%d cache_queue_addr=%p\n", (int) is_cache_queue,
                (void *) &cache->get_queue());
    if (!is_cache_queue) {
        std::printf(
            "FAIL: ggml_backend_sycl_context::stream() did not return the unified cache's own queue after cache "
            "creation -- see llama.cpp-yke2\n");
    }
    const bool is_distinct_queue = (void *) &q_post_cache != q_pre_cache_addr;
    std::printf("queue diag [phase2-post-cache]: distinct_from_phase1=%d\n", (int) is_distinct_queue);
    if (!is_distinct_queue) {
        std::printf(
            "FAIL: the post-cache queue is the SAME object as the pre-cache queue -- expected "
            "ensure_single_device_context_queue() to construct a distinct queue for the cache -- see "
            "llama.cpp-yke2\n");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!pre_cache_ok || !post_cache_ok || !is_cache_queue || !is_distinct_queue) {
        std::printf("FAILED: one or more checks above\n");
        return 1;
    }
    std::printf(
        "PASS: single-GPU stream() carries enable_profiling both before and after unified-cache creation, and the "
        "cache queue is distinct from the pre-cache fallback queue\n");
    return 0;
}
