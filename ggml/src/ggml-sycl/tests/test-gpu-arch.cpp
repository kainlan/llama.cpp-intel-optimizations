//
// Test: architecture-derived GPU family detection
//
// Guards the B70/G31 regression: "Intel(R) Arc(TM) Pro B70 Graphics"
// matched the broad ("Arc" && "Graphics") Alchemist heuristic and was
// classified XeHPG, disabling ESIMD dpas on a Xe2 part.
//
// Host-only: family_from_architecture / family_from_name / family_name /
// family_supports_esimd_dpas are pure functions, so no GPU is required.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "gpu-arch.hpp"

// cold-start.hpp is here for the derive_initial_config coverage below. Note it
// pulls in <string>; the strcmp use in the family_name checks stays deliberate
// (gpu-arch.hpp itself must not need <string>), it is just no longer enforced
// by this TU alone.
#include "cold-start.hpp"

#include <cstdio>
#include <cstring>

// The build is -DNDEBUG (Release), so assert() would compile away and the
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
    // ---- Architecture-derived mapping ------------------------------------
    // Battlemage (Xe2). G31 is the Arc Pro B70, G21 the B580/B50.
    CHECK(
        ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_bmg_g21) == sycl_gpu_family::ARC_BATTLEMAGE,
        "bmg_g21 must map to ARC_BATTLEMAGE");
    CHECK(
        ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_bmg_g31) == sycl_gpu_family::ARC_BATTLEMAGE,
        "bmg_g31 must map to ARC_BATTLEMAGE");

    // Alchemist (XeHPG).
    CHECK(
        ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_acm_g10) == sycl_gpu_family::ARC_ALCHEMIST,
        "acm_g10 must map to ARC_ALCHEMIST");
    CHECK(
        ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_acm_g11) == sycl_gpu_family::ARC_ALCHEMIST,
        "acm_g11 must map to ARC_ALCHEMIST");
    CHECK(
        ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_acm_g12) == sycl_gpu_family::ARC_ALCHEMIST,
        "acm_g12 must map to ARC_ALCHEMIST");

    // Data Center GPU Max (PVC).
    CHECK(ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_pvc) == sycl_gpu_family::DATA_CENTER_MAX,
          "pvc must map to DATA_CENTER_MAX");
    CHECK(
        ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_pvc_vg) == sycl_gpu_family::DATA_CENTER_MAX,
        "pvc_vg must map to DATA_CENTER_MAX");

    // Architectures we deliberately do not classify fall through to UNKNOWN
    // so family_from_device can try the name heuristic.
    CHECK(ggml_sycl::family_from_architecture(syclex::architecture::intel_gpu_mtl_u) == sycl_gpu_family::UNKNOWN,
          "unmapped architecture must yield UNKNOWN, not a guess");

    // The architecture enum's underlying type is 64-bit and the low 32 bits
    // alias between architectures (bmg_g31 is 0x0000000500800000, which
    // truncates to 0x00800000). Anything that narrows to int would collide;
    // assert the two Battlemage parts are still distinct enum values.
    CHECK(syclex::architecture::intel_gpu_bmg_g21 != syclex::architecture::intel_gpu_bmg_g31,
          "bmg_g21 and bmg_g31 must be distinct architecture values");

    // ---- Name fallback ---------------------------------------------------
    // The B70 regression that motivated this task: "Arc" + "Graphics"
    // previously matched the Alchemist branch.
    CHECK(ggml_sycl::family_from_name("Intel(R) Arc(TM) Pro B70 Graphics") == sycl_gpu_family::ARC_BATTLEMAGE,
          "B70 name must fall back to ARC_BATTLEMAGE, not ARC_ALCHEMIST");
    CHECK(ggml_sycl::family_from_name("Intel(R) Arc(TM) Pro B50 Graphics") == sycl_gpu_family::ARC_BATTLEMAGE,
          "B50 name must fall back to ARC_BATTLEMAGE, not ARC_ALCHEMIST");
    CHECK(ggml_sycl::family_from_name("Intel(R) Arc(TM) B580 Graphics") == sycl_gpu_family::ARC_BATTLEMAGE,
          "B580 name must fall back to ARC_BATTLEMAGE");
    CHECK(ggml_sycl::family_from_name("Intel(R) Arc(TM) A770 Graphics") == sycl_gpu_family::ARC_ALCHEMIST,
          "A770 name must remain ARC_ALCHEMIST");
    CHECK(ggml_sycl::family_from_name("Intel(R) Data Center GPU Max 1100") == sycl_gpu_family::DATA_CENTER_MAX,
          "Data Center GPU Max name must map to DATA_CENTER_MAX");
    CHECK(ggml_sycl::family_from_name("Intel(R) Data Center GPU Flex 170") == sycl_gpu_family::DATA_CENTER_FLEX,
          "Data Center GPU Flex name must map to DATA_CENTER_FLEX");
    CHECK(ggml_sycl::family_from_name("NVIDIA GeForce RTX 4090") == sycl_gpu_family::UNKNOWN,
          "non-Intel name must map to UNKNOWN");
    CHECK(ggml_sycl::family_from_name(nullptr) == sycl_gpu_family::UNKNOWN, "null name must map to UNKNOWN");
    // The empty string takes a different path than the nullptr guard: it
    // reaches every name_contains call and must match none of them.
    CHECK(ggml_sycl::family_from_name("") == sycl_gpu_family::UNKNOWN, "empty name must map to UNKNOWN");

    // Matching is case-insensitive; the fixtures above are all correctly cased,
    // so exercise name_contains's lowering through the public API explicitly.
    CHECK(ggml_sycl::family_from_name("intel(r) arc(tm) pro b70 graphics") == sycl_gpu_family::ARC_BATTLEMAGE,
          "lowercase B70 name must still map to ARC_BATTLEMAGE");
    CHECK(ggml_sycl::family_from_name("INTEL(R) ARC(TM) PRO B70 GRAPHICS") == sycl_gpu_family::ARC_BATTLEMAGE,
          "uppercase B70 name must still map to ARC_BATTLEMAGE");
    CHECK(ggml_sycl::family_from_name("intel(r) data center gpu max 1100") == sycl_gpu_family::DATA_CENTER_MAX,
          "lowercase Data Center GPU Max name must still map to DATA_CENTER_MAX");

    // RESIDUAL RISK, pinned deliberately: a generic/truncated name with no
    // model number hits the broad ("Arc" && "Graphics") catch-all and is
    // classified Alchemist — the same failure shape as the B70 bug, for a
    // different device shape. This is NOT fixable from the name alone (there
    // is no correct answer without a model number, and guessing Battlemage
    // would enable ESIMD dpas on parts that cannot run it); the architecture
    // query is the real answer. This assertion exists so that changing the
    // ordering or the catch-all is a deliberate act, not a silent one.
    CHECK(ggml_sycl::family_from_name("Intel(R) Arc(TM) Graphics") == sycl_gpu_family::ARC_ALCHEMIST,
          "generic Arc Graphics name is classified Alchemist by the catch-all (known residual gap)");

    // ---- ESIMD dpas capability ------------------------------------------
    // Battlemage and PVC run ESIMD dpas at ExecutionSize=16; Alchemist does not.
    CHECK(ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::ARC_BATTLEMAGE), "Battlemage must support ESIMD dpas");
    CHECK(ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::DATA_CENTER_MAX),
          "Data Center GPU Max must support ESIMD dpas");
    CHECK(!ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::ARC_ALCHEMIST),
          "Alchemist must not support ESIMD dpas");
    CHECK(!ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::DATA_CENTER_FLEX),
          "Data Center GPU Flex must not support ESIMD dpas");
    CHECK(!ggml_sycl::family_supports_esimd_dpas(sycl_gpu_family::UNKNOWN),
          "unknown family must not support ESIMD dpas");

    // ---- Human-readable names -------------------------------------------
    // strcmp rather than std::string: gpu-arch.hpp deliberately does not pull
    // in <string>, and this test is the guard that it stays that way.
    CHECK(std::strcmp("Arc Battlemage", ggml_sycl::family_name(sycl_gpu_family::ARC_BATTLEMAGE)) == 0,
          "family_name(ARC_BATTLEMAGE)");
    CHECK(std::strcmp("Arc Alchemist", ggml_sycl::family_name(sycl_gpu_family::ARC_ALCHEMIST)) == 0,
          "family_name(ARC_ALCHEMIST)");
    CHECK(std::strcmp("Data Center GPU Max", ggml_sycl::family_name(sycl_gpu_family::DATA_CENTER_MAX)) == 0,
          "family_name(DATA_CENTER_MAX)");
    CHECK(std::strcmp("Data Center GPU Flex", ggml_sycl::family_name(sycl_gpu_family::DATA_CENTER_FLEX)) == 0,
          "family_name(DATA_CENTER_FLEX)");
    CHECK(std::strcmp("Unknown", ggml_sycl::family_name(sycl_gpu_family::UNKNOWN)) == 0, "family_name(UNKNOWN)");

    // ---- cold-start heuristics ------------------------------------------
    // cold-start.cpp used to carry a second, independent copy of the name
    // heuristic, so the B70 regression hit twice: misdetected as Alchemist it
    // also fell to the conservative 16x16 tiles instead of Battlemage's 64x64.
    // These assertions pin cold-start to the shared helper.
    {
        ggml_sycl::GPUCapabilities caps;
        caps.device_name = "Intel(R) Arc(TM) Pro B70 Graphics";
        caps.eu_count    = 256;
        caps.has_dpas    = true;

        CHECK(ggml_sycl::detect_gpu_family(caps) == ggml_sycl::GPUFamily::ARC_BATTLEMAGE,
              "cold-start must classify the B70 as ARC_BATTLEMAGE");

        const ggml_sycl::KernelConfig cfg = ggml_sycl::derive_initial_config(caps);
        CHECK(cfg.tile_m == 64, "B70 must get Battlemage tile_m=64, not the conservative 16");
        CHECK(cfg.tile_n == 64, "B70 must get Battlemage tile_n=64, not the conservative 16");
        CHECK(cfg.use_dpas, "B70 must enable dpas");
    }

    // The shared enum has five values, cold-start's local one has three. The
    // extras must collapse to UNKNOWN (conservative tiles, no dpas) rather
    // than being smuggled into the public 3-value header.
    {
        ggml_sycl::GPUCapabilities caps;
        caps.device_name = "Intel(R) Data Center GPU Max 1100";
        caps.eu_count    = 512;
        caps.has_dpas    = true;

        CHECK(ggml_sycl::detect_gpu_family(caps) == ggml_sycl::GPUFamily::UNKNOWN,
              "families absent from cold-start's 3-value enum must collapse to UNKNOWN");

        const ggml_sycl::KernelConfig cfg = ggml_sycl::derive_initial_config(caps);
        CHECK(cfg.tile_m == 16 && cfg.tile_n == 16, "UNKNOWN family keeps the conservative tiles");
        CHECK(!cfg.use_dpas, "UNKNOWN family must not enable dpas");
    }

    // Alchemist must still take its EU-count-driven branch, unchanged.
    {
        ggml_sycl::GPUCapabilities caps;
        caps.device_name = "Intel(R) Arc(TM) A770 Graphics";
        caps.eu_count    = 512;
        caps.has_dpas    = true;

        CHECK(ggml_sycl::detect_gpu_family(caps) == ggml_sycl::GPUFamily::ARC_ALCHEMIST,
              "A770 must remain ARC_ALCHEMIST in cold-start");

        const ggml_sycl::KernelConfig cfg = ggml_sycl::derive_initial_config(caps);
        CHECK(cfg.tile_m == 64 && cfg.tile_n == 64, "A770 (512 EUs) keeps its 64x64 tiles");
    }

    std::printf("=== All gpu-arch tests passed ===\n");
    return 0;
}
