#pragma once

#include "ggml-cpu.h"

// SYCL-private, module-independent CPU traits. The returned table contains only
// portable ggml-base reference/scalar functions and has no backend state.
const ggml_type_traits_cpu * ggml_sycl_get_type_traits_cpu(enum ggml_type type) noexcept;

// Alias retained for explicit baseline/parity tests; both access the same table.
const ggml_type_traits_cpu * ggml_sycl_get_baseline_type_traits_cpu(enum ggml_type type) noexcept;
