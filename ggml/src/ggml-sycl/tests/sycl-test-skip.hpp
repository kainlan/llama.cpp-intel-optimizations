#pragma once

// Device selection for the SYCL test binaries, with a skip that is VISIBLE as a skip.
//
// ⚠️ The ordering these helpers enforce is the whole point. A bare
//
//     sycl::device dev;                    // or: sycl::queue q;
//     try {
//         dev = sycl::device(sycl::gpu_selector_v);
//     } catch (const sycl::exception & e) {
//         fprintf(stderr, "No GPU device found: %s\n", e.what());
//         return 1;                        // <-- NEVER REACHED
//     }
//
// does not do what it looks like. `sycl::device`'s default constructor is not
// exception-safe: it selects through the DEFAULT selector and THROWS when the
// runtime exposes no device at all -- exactly like `sycl::queue`'s. The
// declaration sits outside the try, so on a device-less host the throw is
// uncaught, std::terminate runs, and the process dies:
//
//     terminate called after throwing an instance of 'sycl::_V1::exception'
//     what(): No device of requested type available
//     Aborted (core dumped)                <- exit 134
//
// without ever reaching the no-device check the author wrote for this case.
// Measured 2026-08-01 with ONEAPI_DEVICE_SELECTOR=level_zero:99 (matches no
// device): 13 of the tests in this directory aborted this way (llama.cpp-ivin;
// llama.cpp-k208 fixed the first one found). The `catch`-and-fall-back-to-
// default_selector variant is no safer -- that fallback throws too, from inside a
// catch handler, with the same result.
//
// ⚠️ This is also why these helpers RETURN std::optional<sycl::device> rather than
// filling in a `sycl::device &` out-parameter. An out-parameter forces the CALLER
// to write `sycl::device dev;` -- reintroducing the exact default construction,
// and the exact abort, that this header exists to remove. That is not theoretical:
// the first draft of this header used an out-parameter, and every one of its call
// sites still aborted with 134. Verified directly that a lone `sycl::device dev;`
// aborts with the message above. std::optional holds no device until one is known
// to be constructible.
//
// ⚠️ Do NOT "align" a fixed test back to the surrounding convention. In this
// directory the surrounding convention is the bug.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <optional>
#include <vector>

// ctest's SKIP_RETURN_CODE. Every registration in ggml/src/ggml-sycl/CMakeLists.txt
// whose test can return this MUST carry `SKIP_RETURN_CODE 77`; without it ctest
// renders 77 as FAILED rather than the intended Skipped.
//
// 77 rather than 0 or 1 because "no device" is not a test result in either
// direction: exit 0 is a vacuous pass that reads as verification (a CPU-only
// runner reporting SKIP and exit 0 proves nothing, yet goes green), and exit 1
// renders a legitimately CPU-only runner as a hard failure. The goal is to make a
// skip visible AS a skip, never to forbid skipping.
static const int SYCL_TEST_SKIP = 77;

// Enumerate GPU devices, treating an enumeration failure as "none". Never throws.
static inline std::vector<sycl::device> sycl_test_gpu_devices() {
    std::vector<sycl::device> devices;
    try {
        devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    } catch (const sycl::exception & e) {
        // Enumeration failing means no device work is possible either -- same
        // outcome, but say which of the two happened rather than flattening them.
        fprintf(stderr, "SYCL exception while enumerating GPU devices: %s\n", e.what());
        devices.clear();
    }
    return devices;
}

// Print the skip diagnostic. `what` names what this run did NOT verify, so the
// line cannot be mistaken for a pass by someone skimming CI output.
static inline void sycl_test_print_skip(const char * what) {
    fprintf(stderr,
            "SKIP: no SYCL GPU devices available; this run proves NOTHING about %s.\n"
            "      Source oneAPI and re-run to actually test it:\n"
            "      (source /opt/intel/oneapi/setvars.sh --force)\n",
            what);
}

// A GPU is required. Returns the selected GPU, or an empty optional when the
// caller must `return SYCL_TEST_SKIP`.
//
// Selection still goes through gpu_selector_v, exactly as these tests always did
// -- enumeration is only the guard, so behaviour on a host that HAS a GPU is
// unchanged (gpu_selector_v scores devices; devices[0] would not).
static inline std::optional<sycl::device> sycl_test_require_gpu(const char * what) {
    if (sycl_test_gpu_devices().empty()) {
        sycl_test_print_skip(what);
        return std::nullopt;
    }

    try {
        return sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception & e) {
        // A GPU was enumerated a moment ago, so this is not the device-less case.
        // Report it and skip rather than letting it terminate the process.
        fprintf(stderr, "SKIP: GPU enumerated but could not be selected: %s\n", e.what());
        sycl_test_print_skip(what);
        return std::nullopt;
    }
}

// A GPU is preferred but any device will do -- for the tests written with a
// deliberate "No GPU found, using default device" fallback. Returns an empty
// optional only when there is no device at all, i.e. when the caller must
// `return SYCL_TEST_SKIP`.
static inline std::optional<sycl::device> sycl_test_prefer_gpu(const char * what) {
    if (!sycl_test_gpu_devices().empty()) {
        try {
            return sycl::device(sycl::gpu_selector_v);
        } catch (const sycl::exception & e) {
            fprintf(stderr, "No GPU could be selected (%s), trying the default device\n", e.what());
        }
    }

    // The historical fallback, which used to run from inside a catch handler and
    // throw again on a device-less host. It is now the last resort, and its throw
    // is caught.
    try {
        sycl::device dev(sycl::default_selector_v);
        fprintf(stderr, "No GPU found, using default device\n");
        return dev;
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No SYCL device of any type available: %s\n", e.what());
    }

    sycl_test_print_skip(what);
    return std::nullopt;
}
