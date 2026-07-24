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
// far side with ESIMD dpas enabled.
//
// Host-only: family_from_architecture / family_from_name /
// family_supports_esimd_dpas are pure functions, so no GPU is required.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "gpu-arch.hpp"

#include <cstdio>

// The build is -DNDEBUG (Release), so assert() would compile away and this
// test would pass vacuously. Use an explicit check that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace syclex = sycl::ext::oneapi::experimental;

using ggml_sycl::sycl_gpu_family;

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

    std::printf("=== All esimd-dpas-gate tests passed ===\n");
    return 0;
}
