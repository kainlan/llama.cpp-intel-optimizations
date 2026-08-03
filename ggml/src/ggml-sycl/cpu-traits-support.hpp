#pragma once

#include "ggml-cpu.h"

// SYCL-private CPU traits access. The baseline is always available and contains
// no CPU-backend state. If a CPU registry is loaded, the public-facing helper
// asks it for a fresh traits pointer on every call (the proc is never cached).
const ggml_type_traits_cpu * ggml_sycl_get_type_traits_cpu(enum ggml_type type) noexcept;

// Kept private to ggml-sycl; exposed here only for host parity tests.
const ggml_type_traits_cpu * ggml_sycl_get_baseline_type_traits_cpu(enum ggml_type type) noexcept;
