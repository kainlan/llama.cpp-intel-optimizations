//
// Test: a Battlemage device can never be gated onto the legacy mul_mat path
//
// dispatch.hpp turns the unified kernel off when XMXConfig::supports_esimd_dpas
// is false, and that flag is exactly
// family_supports_esimd_dpas(family_from_device(dev)). The Arc Pro B70 fell
// through that gate for an unknown period: its name matched the broad
// ("Arc" && "Graphics") Alchemist heuristic, so a Xe2 part was classified
// XeHPG and silently ran the legacy kernels.
//
// This test pins the whole gate, not just the classifier: every input we
// classify as Battlemage — by architecture enum or by name — must come out the
// far side with ESIMD dpas enabled. The last section then calls
// is_unified_kernel_enabled_for_device() itself and asserts on the diagnostic
// it prints, so the disable path is covered by running it rather than by
// reading it.
//
// Host-only, no GPU: the classifiers are pure functions, and the gate is
// reached with a device id that returns before any device is queried (see the
// link-seam note below).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "dispatch.hpp"
#include "gpu-arch.hpp"

#include <cstdio>
#include <cstring>

// The build is -DNDEBUG (Release), so assert() would compile away and this
// test would pass vacuously. Use an explicit check that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Same check, reporting on stdout. The last section captures stderr to assert
// on the diagnostic's text, and a failure written to a redirected stderr would
// land in the capture file instead of the ctest log.
#define CHECK_STDOUT(cond, msg)                                        \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                  \
        }                                                              \
    } while (0)

namespace syclex = sycl::ext::oneapi::experimental;

using ggml_sycl::sycl_gpu_family;

// ---- Link seams -------------------------------------------------------------
// dispatch.hpp's gate needs exactly two symbols from the backend library:
// ggml_sycl_map_device_id (common.cpp) and XMXConfig::from_device
// (unified-kernel.cpp). Defining them here instead of linking libggml-sycl is
// what keeps this test host-only — the library would bring in dpct::dev_mgr,
// whose first use enumerates every SYCL device on the machine, which is far
// more than a unit test should do and has hung this host in the past.
//
// The cost is stated plainly: the section below covers the gate and its
// diagnostic, NOT the real XMXConfig::from_device capability query.
int ggml_sycl_map_device_id(int device) {
    return device;
}

namespace ggml_sycl_unified {
// Default-constructed XMXConfig has supports_esimd_dpas = false, which is what
// the real from_device() also returns for an id it cannot resolve.
XMXConfig XMXConfig::from_device(int /*device_id*/) {
    return XMXConfig();
}
}  // namespace ggml_sycl_unified

int main() {
    // ---- Names that must reach the unified kernel ------------------------
    // The reported names of the Battlemage parts this fork runs on, plus the
    // remaining B-series forms the heuristic claims to know. A miss here is
    // the B70 bug again: the device drops to the legacy mul_mat path.
    const char * const battlemage_names[] = {
        "Intel(R) Arc(TM) Pro B70 Graphics",  //
        "Intel(R) Arc(TM) Pro B50 Graphics",  //
        "Intel(R) Arc(TM) B580 Graphics",     //
        "Intel(R) Arc(TM) B570 Graphics",     //
        "Intel(R) Arc(TM) Pro B60 Graphics",  //
        "Intel(R) Arc(TM) Battlemage Graphics",
    };

    for (const char * name : battlemage_names) {
        const sycl_gpu_family family = ggml_sycl::family_from_name(name);
        CHECK(family == sycl_gpu_family::ARC_BATTLEMAGE, "Battlemage device name must classify as ARC_BATTLEMAGE");
        CHECK(ggml_sycl::family_supports_esimd_dpas(family),
              "Battlemage device must not be gated onto the legacy mul_mat path");
        std::printf("ok: %s -> %s (ESIMD dpas enabled)\n", name, ggml_sycl::family_name(family));
    }

    // ---- Architectures that must reach the unified kernel ----------------
    // The architecture query is the primary path; the names above only run
    // when it yields UNKNOWN or throws. Both legs have to clear the gate.
    const syclex::architecture battlemage_archs[] = {
        syclex::architecture::intel_gpu_bmg_g21,  // B580 / B50
        syclex::architecture::intel_gpu_bmg_g31,  // B70
    };

    for (const syclex::architecture arch : battlemage_archs) {
        const sycl_gpu_family family = ggml_sycl::family_from_architecture(arch);
        CHECK(family == sycl_gpu_family::ARC_BATTLEMAGE, "Battlemage architecture must classify as ARC_BATTLEMAGE");
        CHECK(ggml_sycl::family_supports_esimd_dpas(family),
              "Battlemage architecture must not be gated onto the legacy mul_mat path");
    }
    std::printf("ok: bmg_g21 and bmg_g31 -> %s (ESIMD dpas enabled)\n",
                ggml_sycl::family_name(sycl_gpu_family::ARC_BATTLEMAGE));

    // ---- Negative controls ------------------------------------------------
    // Without these, a family_supports_esimd_dpas() that returned true for
    // everything would satisfy every assertion above. Alchemist only has
    // ESIMD ExecutionSize=8, so it must stay on the legacy path.
    CHECK(ggml_sycl::family_from_name("Intel(R) Arc(TM) A770 Graphics") == sycl_gpu_family::ARC_ALCHEMIST,
          "A770 must classify as ARC_ALCHEMIST");
    CHECK(!ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::ARC_ALCHEMIST),
          "Alchemist must remain gated onto the legacy mul_mat path");
    CHECK(!ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::UNKNOWN),
          "an unidentified device must remain gated onto the legacy mul_mat path");
    std::printf("ok: Arc Alchemist and Unknown stay on the legacy path\n");

    // ---- The gate as dispatch.hpp actually calls it -----------------------
    // Everything above exercises gpu-arch.hpp's pure classifiers. This section
    // calls the production entry point, so "does the diagnostic ever fire?" is
    // a question a test run answers rather than one a reader has to trace.
    // A negative device id is used because it reaches the disable path without
    // depending on which devices this machine happens to have.
    {
        const char * const capture_path = "test-esimd-dpas-gate-stderr.txt";

        // stderr stays redirected for the remainder of main.
        CHECK_STDOUT(std::freopen(capture_path, "w", stderr) != nullptr, "could not redirect stderr for capture");

        const bool enabled = ggml_sycl::is_unified_kernel_enabled_for_device(-1);
        std::fflush(stderr);
        CHECK_STDOUT(!enabled, "a device id that cannot be resolved must not enable the unified kernel");

        char        captured[1024] = { 0 };
        std::FILE * f              = std::fopen(capture_path, "r");
        CHECK_STDOUT(f != nullptr, "could not read back the captured diagnostic");
        const size_t n = std::fread(captured, 1, sizeof(captured) - 1, f);
        std::fclose(f);
        captured[n] = '\0';

        CHECK_STDOUT(std::strstr(captured, "Unified kernel disabled") != nullptr,
                     "the disable diagnostic must be printed, not silently skipped");
        CHECK_STDOUT(std::strstr(captured, "-1") != nullptr, "the diagnostic must name the device id it was given");
        CHECK_STDOUT(std::strstr(captured, "legacy mul_mat kernels") != nullptr,
                     "the diagnostic must say which path is taken instead");
        std::printf("ok: is_unified_kernel_enabled_for_device(-1) returned false and printed:\n%s", captured);

        // The `static bool logged` guard: a second call must add nothing, or
        // the message floods stderr once per dispatch.
        CHECK_STDOUT(std::freopen(capture_path, "w", stderr) != nullptr, "could not redirect stderr for the recheck");
        const bool enabled_again = ggml_sycl::is_unified_kernel_enabled_for_device(-1);
        std::fflush(stderr);
        CHECK_STDOUT(!enabled_again, "the second call must also refuse the unified kernel");

        f = std::fopen(capture_path, "r");
        CHECK_STDOUT(f != nullptr, "could not read back the second capture");
        const size_t n_again = std::fread(captured, 1, sizeof(captured) - 1, f);
        std::fclose(f);
        std::remove(capture_path);
        CHECK_STDOUT(n_again == 0, "the diagnostic must print only once per process");
        std::printf("ok: the diagnostic printed once; the second call was silent\n");
    }

    std::printf("=== All esimd-dpas-gate tests passed ===\n");
    return 0;
}
