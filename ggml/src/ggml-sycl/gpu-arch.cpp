//
// Canonical Intel GPU family detection for the SYCL backend.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "gpu-arch.hpp"

#include <algorithm>
#include <cctype>
#include <string>

// GGML_SYCL_DEBUG lives in common.hpp and expands to a reference to the
// g_ggml_sycl_debug globals, which are defined in a libggml-sycl TU. The
// host-only test compiles this file WITHOUT linking libggml-sycl, so it stubs
// the macro out instead — the same pattern unified-kernel.cpp uses under
// XMX_TEST_STANDALONE. common.hpp is included here and never from
// gpu-arch.hpp: Task 5 pulls that header into dispatch.hpp, which is included
// widely, so it must stay light.
#if defined(GGML_SYCL_GPU_ARCH_STANDALONE)
#    include <cstdio>
// The stub must still COMPILE its arguments, not discard them: a macro that
// swallows __VA_ARGS__ would let a bad format specifier or an undeclared
// variable in the log call survive the test build and only break the library
// build. Guarding a real fprintf with `if (false)` keeps -Wformat checking
// while the optimizer drops the call.
#    define GGML_SYCL_DEBUG(...)                   \
        do {                                       \
            if (false) {                           \
                std::fprintf(stderr, __VA_ARGS__); \
            }                                      \
        } while (0)
#else
#    include "common.hpp"
#endif

namespace syclex = sycl::ext::oneapi::experimental;

namespace ggml_sycl {

static bool name_contains(const char * name, const char * substr) {
    if (!name || !substr) {
        return false;
    }
    std::string lower_name(name);
    std::string lower_sub(substr);
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(lower_sub.begin(), lower_sub.end(), lower_sub.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower_name.find(lower_sub) != std::string::npos;
}

// Switch on the enumerators directly. `architecture`'s underlying type is
// 64-bit and its low 32 bits alias across families, so any narrowing compare
// would silently misclassify (see gpu-arch.hpp).
sycl_gpu_family family_from_architecture(syclex::architecture arch) {
    switch (arch) {
        case syclex::architecture::intel_gpu_bmg_g21:
        case syclex::architecture::intel_gpu_bmg_g31:
            return sycl_gpu_family::ARC_BATTLEMAGE;
        case syclex::architecture::intel_gpu_acm_g10:
        case syclex::architecture::intel_gpu_acm_g11:
        case syclex::architecture::intel_gpu_acm_g12:
            return sycl_gpu_family::ARC_ALCHEMIST;
        case syclex::architecture::intel_gpu_pvc:
        case syclex::architecture::intel_gpu_pvc_vg:
            return sycl_gpu_family::DATA_CENTER_MAX;
        default:
            return sycl_gpu_family::UNKNOWN;
    }
}

sycl_gpu_family family_from_name(const char * name) {
    if (!name) {
        return sycl_gpu_family::UNKNOWN;
    }
    // B-series FIRST. The historical bug: "Intel(R) Arc(TM) Pro B70 Graphics"
    // fell through to the ("Arc" && "Graphics") Alchemist heuristic below and
    // was classified XeHPG, disabling ESIMD dpas on a Xe2 part.
    //
    // These SKU numbers are unanchored substrings. No collision exists among
    // the current set, but Intel's A-series went 3-digit (A310/A380/A580/
    // A750/A770), so a future 3-digit B-series SKU could share a 2-digit
    // prefix here (e.g. a "B505" would match "B50"). Revisit if that ships.
    if (name_contains(name, "Battlemage") || name_contains(name, "B580") || name_contains(name, "B570") ||
        name_contains(name, "B50") || name_contains(name, "B60") || name_contains(name, "B70")) {
        return sycl_gpu_family::ARC_BATTLEMAGE;
    }
    // RESIDUAL RISK, deliberately left in place: the ("Arc" && "Graphics")
    // clause is a broad catch-all, so a device reporting a generic or
    // truncated name with no model number — "Intel(R) Arc(TM) Graphics" — is
    // classified Alchemist. That is the same failure shape as the B70 bug.
    // It is NOT fixable from the name: without a model number there is no
    // correct answer, and guessing Battlemage would enable ESIMD dpas on
    // parts that cannot run it. The architecture query is the real answer and
    // is exactly what this fallback exists to back up; the name path only
    // runs when that query yielded UNKNOWN or threw. test-gpu-arch pins this
    // behaviour so any future reordering here is a deliberate change.
    if (name_contains(name, "A770") || name_contains(name, "A750") || name_contains(name, "A580") ||
        name_contains(name, "A380") || name_contains(name, "A310") ||
        (name_contains(name, "Arc") && name_contains(name, "Graphics"))) {
        return sycl_gpu_family::ARC_ALCHEMIST;
    }
    if (name_contains(name, "Data Center GPU Max") || name_contains(name, "Ponte Vecchio")) {
        return sycl_gpu_family::DATA_CENTER_MAX;
    }
    if (name_contains(name, "Data Center GPU Flex")) {
        return sycl_gpu_family::DATA_CENTER_FLEX;
    }
    return sycl_gpu_family::UNKNOWN;
}

sycl_gpu_family family_from_device(const sycl::device & dev) {
    // Falling back to the name heuristic must stay non-fatal, but it must not
    // be silent: family_from_name has its own edge cases (see the residual
    // risk noted above), so an unlogged fallback would recreate exactly the
    // silent misclassification this file exists to eliminate.
    std::string fallback_reason;
    try {
        const sycl_gpu_family from_arch = family_from_architecture(dev.get_info<syclex::info::device::architecture>());
        if (from_arch != sycl_gpu_family::UNKNOWN) {
            return from_arch;
        }
        fallback_reason = "architecture is not one this backend maps";
    } catch (const sycl::exception & e) {
        // Copy the message: e.what()'s storage dies with the exception object.
        fallback_reason = e.what();
    }

    const std::string     name   = dev.get_info<sycl::info::device::name>();
    const sycl_gpu_family family = family_from_name(name.c_str());
    GGML_SYCL_DEBUG("[gpu-arch] architecture query did not resolve for device '%s' (%s); name heuristic -> %s\n",
                    name.c_str(), fallback_reason.c_str(), family_name(family));
    return family;
}

bool family_supports_esimd_dpas(sycl_gpu_family family) {
    switch (family) {
        case sycl_gpu_family::ARC_BATTLEMAGE:
        case sycl_gpu_family::DATA_CENTER_MAX:
            return true;
        default:
            return false;
    }
}

const char * family_name(sycl_gpu_family family) {
    switch (family) {
        case sycl_gpu_family::ARC_ALCHEMIST:
            return "Arc Alchemist";
        case sycl_gpu_family::ARC_BATTLEMAGE:
            return "Arc Battlemage";
        case sycl_gpu_family::DATA_CENTER_MAX:
            return "Data Center GPU Max";
        case sycl_gpu_family::DATA_CENTER_FLEX:
            return "Data Center GPU Flex";
        default:
            return "Unknown";
    }
}

}  // namespace ggml_sycl
