//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_CONCAT_HPP
#define GGML_SYCL_CONCAT_HPP

#include "common.hpp"

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst);

// True for exactly the element types ggml_sycl_op_concat instantiates a kernel
// for; everything else hits its trailing GGML_ASSERT(false).  Defined in
// concat.cpp so it sees the same GGML_SYCL_HAS_BF16 state as that switch.
bool ggml_sycl_concat_type_supported(ggml_type type);

#endif // GGML_SYCL_CONCAT_HPP
