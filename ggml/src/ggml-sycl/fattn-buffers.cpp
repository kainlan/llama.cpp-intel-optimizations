//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "fattn-buffers.hpp"

#include "common.hpp"

#include <utility>

sycl::half * ggml_sycl_fattn_kv_buffers::kv_buffer::ensure_half(size_t n_elems) {
    const size_t need_bytes = n_elems * sizeof(sycl::half);

    if (capacity >= need_bytes) {
        return ptr;
    }

    if (ptr) {
        SYCL_CHECK(CHECK_TRY_ERROR(qptr->wait()));
        owner    = {};
        ptr      = nullptr;
        capacity = 0;
    }

    size_t cap = 0;
    while (cap < need_bytes) {
        cap += CHUNK_SIZE;
    }

    ggml_sycl::alloc_request req{};
    req.queue                               = qptr;
    req.device                              = device;
    req.size                                = cap;
    req.intent.role                         = ggml_sycl::alloc_role::KV;
    req.intent.category                     = ggml_sycl::runtime_category::KV_CACHE;
    req.intent.cohort_id                    = "fattn_kv_buffer";
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::KV;

    owner = ggml_sycl::unified_allocate(req);
    if (!owner.valid()) {
        GGML_LOG_ERROR("%s: can't allocate %lu Bytes of memory on device\n", __func__, cap);
        GGML_ABORT("fattn buffer alloc failed");
    }

    auto resolved = owner.resolve(device);
    if (!resolved || !resolved.ptr || !resolved.on_device) {
        owner = {};
        GGML_LOG_ERROR("%s: can't resolve %lu Bytes of device memory\n", __func__, cap);
        GGML_ABORT("fattn buffer resolve failed");
    }

    ptr      = static_cast<sycl::half *>(resolved.ptr);
    capacity = cap;
    return ptr;
}

ggml_sycl_fattn_kv_buffers::kv_buffer::~kv_buffer() {
#ifdef DEBUG_SYCL_POOL
    GGML_LOG_INFO("ggml_sycl_fattn_kv_buffer[%d]: %.2f MiB\n", device, capacity / 1024.0 / 1024.0);
#endif
    if (ptr) {
        SYCL_CHECK(CHECK_TRY_ERROR(qptr->wait()));
        owner = {};
    }
}
