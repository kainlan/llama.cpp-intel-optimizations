// Regression gate for llama.cpp-1lrh.
//
// ggml_backend_reg_by_name("SYCL") is a discovery/registration query -- exactly
// what a caller does when it does not yet know whether the SYCL backend is
// available. Until this fix, dpct::dev_mgr::dev_mgr() (helper.hpp) called
// GGML_ABORT("No SYCL devices available") the moment platform enumeration
// found zero devices, so a registration query on a device-less environment
// SIGABRTed the whole process (rc=134) instead of answering "zero devices".
//
// This test forces that zero-device condition without touching any real GPU:
// ONEAPI_DEVICE_SELECTOR=level_zero:99 selects a device INDEX that cannot
// exist (this host never has 100 Level-Zero devices), which the loader
// accepts as syntactically valid and resolves to "zero devices" rather than a
// selector parse error -- verified empirically for llama.cpp-1lrh by
// comparing against the pre-fix build, where this exact selector reproduced
// the abort at helper.hpp:1186 through the real
// ggml_backend_reg_by_name -> ggml_backend_sycl_reg -> ggml_sycl_init ->
// dpct::dev_mgr::dev_mgr() call chain.
//
// The test asserts the registration half of the fix: ggml_backend_reg_by_name
// ("SYCL") returns a non-null registry reporting exactly 0 devices. Simply
// reaching this line already proves the constructor did not GGML_ABORT.
//
// It deliberately does NOT also drive ggml_backend_sycl_init(0) (a genuine
// "use" attempt) inside this ctest binary. Measured for llama.cpp-1lrh: with
// 0 registered devices, ggml_backend_sycl_init(0) reaches
// ggml_backend_reg_dev_get -> ggml_backend_sycl_reg_get_device, whose
// GGML_ASSERT(index < ctx->devices.size()) is unconditional (unlike a bare
// assert(), it is NOT compiled out under -DNDEBUG) and calls GGML_ABORT --
// i.e. the use path fails LOUDLY, exactly as intended, but by aborting the
// process rather than by returning an error. Encoding that as an in-process
// ctest assertion would make this test SIGABRT on every green run, which is
// not a usable regression gate. That behaviour is documented and verified
// manually instead (see the llama.cpp-1lrh task comment for the captured
// backtrace); a dedicated fork()+waitpid() harness to assert "child died of
// SIGABRT" was considered and rejected as unsafe to add after a SYCL runtime
// has begun platform enumeration.
//
// This test does not use test-skip.h's LLAMA_TEST_EXIT_SKIP path as its
// normal outcome: with ONEAPI_DEVICE_SELECTOR pinned by the ENVIRONMENT
// property below, the zero-device condition is forced on every host, so the
// property under test is exercised unconditionally. SKIP_RETURN_CODE is
// still registered (matching the family convention -- see
// test-sycl-env-report.cpp) as a defensive exit for the one case this test
// cannot itself repair: a host whose SYCL loader does not treat an
// out-of-range device index as "zero devices" the way this one does.

#include "ggml-backend.h"
#include "test-skip.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main() {
    // Reaching this point at all (rather than SIGABRT) is the proof: the
    // pre-fix dev_mgr() constructor aborted from inside this call.
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("SYCL");
    CHECK(reg != nullptr, "ggml_backend_reg_by_name(\"SYCL\") returned null");

    size_t dev_count = ggml_backend_reg_dev_count(reg);
    if (dev_count != 0) {
        // The chosen mechanism did not yield zero devices on this host --
        // this test cannot exercise the property it exists to check. Skip
        // rather than report a false pass or a false fail.
        std::fprintf(stderr,
                     "SKIP: ONEAPI_DEVICE_SELECTOR=level_zero:99 reported %zu device(s) on this "
                     "host; expected 0 -- cannot exercise the zero-device registration path\n",
                     dev_count);
        return LLAMA_TEST_EXIT_SKIP;
    }

    std::fprintf(stderr, "PASS: registration reported 0 devices without aborting\n");
    return 0;
}
