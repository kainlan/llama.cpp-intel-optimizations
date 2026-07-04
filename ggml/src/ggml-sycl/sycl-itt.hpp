#pragma once

#include <cstdint>

namespace ggml_sycl {

bool     sycl_itt_enabled_from_env(const char * value);
bool     sycl_itt_enabled();
void     sycl_itt_task_begin(const char * category, const char * name);
void     sycl_itt_task_end();
void     sycl_itt_reset_for_tests();
uint64_t sycl_itt_begin_count_for_tests();
uint64_t sycl_itt_end_count_for_tests();

}  // namespace ggml_sycl
