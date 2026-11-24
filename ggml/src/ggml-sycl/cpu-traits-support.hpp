#pragma once

#include "ggml-cpu.h"

// SYCL-private, module-independent CPU traits. Under GGML_BACKEND_DL the returned table is the portable baseline below:
// ggml-base reference/scalar functions and no backend state. Otherwise it is ggml-cpu's own table, whose functions read
// ggml-cpu's lookup tables; the first call fills those (ggml_cpu_init), so the entries are usable in a process that
// never initialized the CPU backend.
const ggml_type_traits_cpu * ggml_sycl_get_type_traits_cpu(enum ggml_type type) noexcept;

// The portable baseline table, for explicit baseline/parity tests; the getter above returns it under GGML_BACKEND_DL.
const ggml_type_traits_cpu * ggml_sycl_get_baseline_type_traits_cpu(enum ggml_type type) noexcept;
