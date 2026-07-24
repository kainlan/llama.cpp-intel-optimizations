//
// Canonical Intel GPU family detection for the SYCL backend.
//
// Family is derived from the SYCL architecture enum, which the runtime
// reports directly. The legacy device-name substring heuristic is retained
// ONLY as a fallback for devices whose architecture is unknown, so no
// currently-working device regresses.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

// Keep this header light: Task 5 includes it from dispatch.hpp, which is
// included widely. Nothing here needs <string> — family_from_name takes a
// const char * and family_name returns one.
#include <sycl/sycl.hpp>

namespace ggml_sycl {

enum class sycl_gpu_family {
    UNKNOWN,
    ARC_ALCHEMIST,     // XeHPG  — ESIMD ExecutionSize=8 only
    ARC_BATTLEMAGE,    // Xe2    — ESIMD ExecutionSize=16
    DATA_CENTER_MAX,   // PVC    — ESIMD ExecutionSize=16
    DATA_CENTER_FLEX,  // XeHPG-based
};

// Primary path: derive family from the SYCL architecture enum.
//
// NOTE: the underlying type of `architecture` is 64-bit and the low 32 bits
// alias between architectures (intel_gpu_bmg_g31 is 0x0000000500800000, which
// truncates to 0x00800000). Always switch on the enumerators themselves;
// never cast to int or compare against a numeric literal.
sycl_gpu_family family_from_architecture(sycl::ext::oneapi::experimental::architecture arch);

// Fallback ONLY when architecture is unknown. Ordered so B-series is tested
// before the broad ("Arc" && "Graphics") Alchemist heuristic.
sycl_gpu_family family_from_name(const char * name);

// Resolve for a live device: architecture first, name only as fallback.
sycl_gpu_family family_from_device(const sycl::device & dev);

bool         family_supports_esimd_dpas(sycl_gpu_family family);
const char * family_name(sycl_gpu_family family);

}  // namespace ggml_sycl
