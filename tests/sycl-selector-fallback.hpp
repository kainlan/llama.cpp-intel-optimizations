// Shared ONEAPI_DEVICE_SELECTOR re-exec fallback for SYCL GPU tests.
//
// WHY A PLAIN setenv() IN main() DOES NOT WORK (llama.cpp-2x3m, llama.cpp-5q1r):
// every SYCL test here links ggml-sycl, and ggml-sycl links oneCCL for its
// tensor-parallelism ALL_REDUCE path (ggml/src/ggml-sycl/CMakeLists.txt,
// `GGML_SYCL_ONECCL`, default ON). /opt/intel/oneapi/ccl/2022.1/lib/libccl.so.1
// carries a static initializer (`_GLOBAL__sub_I_comm.cpp`) that constructs a
// sycl::event at library LOAD time -- before main() runs. That construction
// makes libsycl read and memoize ONEAPI_DEVICE_SELECTOR immediately, so a plain
// `setenv()` inside main() is always too late: the value the runtime already
// cached wins regardless of what main() sets afterward. A bare (non-ctest)
// invocation of an affected test therefore silently enumerates every device,
// including the integrated GPU, whose `global_mem_size` reports 231.7 GB of
// system RAM as phantom "VRAM" (llama.cpp-403s) -- exactly the outcome the
// fallback exists to prevent, with nothing in the test's output to say so.
//
// THE FIX: re-exec the whole process. setenv() the desired selector, then
// execv("/proc/self/exe", argv) -- the child starts fresh, and libccl's static
// initializer (which now runs before the CHILD's main(), same as any process
// start) observes the variable already set. `/proc/self/exe` is used instead
// of argv[0] because execv() does no PATH search and argv[0] is
// caller-controlled -- it may be relative, or missing a directory component
// the shell resolved for it.
//
// Re-exec is gated on setenv() itself succeeding: an ENOMEM setenv() failure
// followed by an unconditional execv() would re-launch the same unfixed
// process forever. Whether setenv() fails or the execv() call itself fails,
// this warns and continues UNPINNED rather than treating either as fatal --
// under ctest this function is a no-op every run (ctest supplies the selector
// via each registration's ENVIRONMENT, so getenv() already sees it set); the
// fallback exists only to make a bare invocation behave the same way, and a
// working bare run should stay a working bare run even if re-exec cannot
// happen. This is also why the failure path is a warning, never
// `exit(77)` (ctest's SKIP_RETURN_CODE): 77 cannot help a bare run, and would
// turn a correct-but-unpinned enumeration into a reported skip.
//
// Ported from the form S2 landed first (759b5647d,
// tests/test-sycl-mmvq-q8-0-soa-numerics.cpp) to every sibling GPU test that
// pins a selector this way (llama.cpp-5q1r), so there is exactly one
// definition instead of ~20 independent copies.
#pragma once

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Pins `selector` via ONEAPI_DEVICE_SELECTOR before the SYCL runtime can
// enumerate devices. Call this as the FIRST statement of main(int, char **
// argv) -- before any other statement, including any SYCL/backend call:
// re-exec restarts the whole process, so anything run before this call runs
// twice (once before the re-exec, once after).
static inline void sycl_test_selector_fallback(char ** argv, const char * selector) {
    if (std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        // ctest's ENVIRONMENT already set it, or the caller pinned one by hand;
        // either way, honour it and do not re-exec.
        return;
    }
    if (setenv("ONEAPI_DEVICE_SELECTOR", selector, 1) != 0) {
        std::fprintf(stderr,
                     "warning: setenv(ONEAPI_DEVICE_SELECTOR) failed (%s); continuing unpinned -- run with "
                     "ONEAPI_DEVICE_SELECTOR=%s set to pin the validation card\n",
                     std::strerror(errno), selector);
        return;
    }
    execv("/proc/self/exe", argv);
    // execv() only returns on failure.
    std::fprintf(stderr,
                 "warning: re-exec failed (%s); continuing unpinned -- run with ONEAPI_DEVICE_SELECTOR=%s set to "
                 "pin the validation card\n",
                 std::strerror(errno), selector);
}
