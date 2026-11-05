//
// The lazy table (th32 §3.2, §3.3, §10): its two constructors, the HOST body's eager O_DIRECT read into the locked
// HOST_CPU tier, the STORAGE descriptor, the activation check and the sole release point. See lazy-table.hpp.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "lazy-table.hpp"

#include "ggml-impl.h"

#include <unistd.h>

#include <string>

namespace ggml_sycl {

namespace {

lazy_table_result refuse(const lazy_table_desc & desc, lazy_table_status status, const std::string & reason) {
    GGML_LOG_WARN("[LAZY-TABLE] %s refused: %s: %s\n", desc.name.c_str(), lazy_table_status_name(status),
                  reason.c_str());
    lazy_table_result result;
    result.status = status;
    result.reason = reason;
    return result;
}

void close_fd(int fd) {
    if (fd >= 0) {
        // Linux releases the descriptor even when close reports EINTR, so it is never retried.
        (void) ::close(fd);
    }
}

// The ledger's figure for one record (th32 §10): the charge its registry row took, host_cpu_mapped_size of the
// request, never the handle's requested size.
size_t record_charged_bytes(const mem_handle & record) {
    alloc_metadata meta{};
    if (!record.owner_metadata(&meta)) {
        return 0;
    }
    return unified_alloc_host_cpu_charged_bytes(meta);
}

}  // namespace

const char * lazy_table_status_name(lazy_table_status status) {
    switch (status) {
        case lazy_table_status::OK:
            return "ok";
        case lazy_table_status::INVALID_DESC:
            return "lazy-table-invalid-desc";
        case lazy_table_status::OPEN_FAILED:
            return "lazy-table-open-failed";
        case lazy_table_status::SOURCE_SHORT:
            return "lazy-table-source-short";
        case lazy_table_status::BODY_REFUSED:
            return "lazy-table-body-refused";
        case lazy_table_status::READ_FAILED:
            return "lazy-table-read-failed";
        case lazy_table_status::ACTIVATION_MISMATCH:
            return "lazy-table-activation-mismatch";
        case lazy_table_status::ROW_CACHE_UNBUILT:
            return "lazy-table-row-cache-unbuilt";
    }
    return "lazy-table-unknown";
}

lazy_table::~lazy_table() {
    records_.clear();
    close_fd(fd_);
}

void lazy_table::release() {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete this;
    }
}

mem_handle lazy_table::rows() const {
    if (desc_.residency != lazy_residency::HOST || records_.empty()) {
        return {};
    }
    return records_.front().slice(body_offset_, desc_.table_bytes);
}

size_t lazy_table::charged_bytes() const {
    size_t sum = 0;
    for (const mem_handle & record : records_) {
        sum += record_charged_bytes(record);
    }
    return sum;
}

lazy_table_result lazy_table::materialize(const lazy_table_desc & desc,
                                          lazy_residency          residency,
                                          size_t                  extra_record_bytes) {
    (void) residency;
    (void) extra_record_bytes;
    return refuse(desc, lazy_table_status::READ_FAILED, "no materialization is built");
}

lazy_table_result lazy_table_create_host(const lazy_table_desc & desc) {
    return lazy_table::materialize(desc, lazy_residency::HOST, 0);
}

lazy_table_result lazy_table_create_storage(const lazy_table_desc & desc) {
    return lazy_table::materialize(desc, lazy_residency::STORAGE, 0);
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
lazy_table_result lazy_table_test_create_host_with_extra_record(const lazy_table_desc & desc, size_t extra_bytes) {
    return lazy_table::materialize(desc, lazy_residency::HOST, extra_bytes);
}
#endif

}  // namespace ggml_sycl
