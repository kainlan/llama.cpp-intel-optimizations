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
    if (name_contains(name, "Battlemage") || name_contains(name, "B580") || name_contains(name, "B570") ||
        name_contains(name, "B50") || name_contains(name, "B60") || name_contains(name, "B70")) {
        return sycl_gpu_family::ARC_BATTLEMAGE;
    }
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
    try {
        const sycl_gpu_family from_arch = family_from_architecture(dev.get_info<syclex::info::device::architecture>());
        if (from_arch != sycl_gpu_family::UNKNOWN) {
            return from_arch;
        }
    } catch (const sycl::exception &) {
        // Runtime lacks the architecture query; fall through to the name heuristic.
    }
    return family_from_name(dev.get_info<sycl::info::device::name>().c_str());
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
