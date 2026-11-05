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
#include "host-cpu-mapping.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

namespace ggml_sycl {

namespace {

// The largest single pread of the eager read. Linux returns at most 0x7ffff000 bytes from one call, so a block-aligned
// request of at most 1 GiB is returned whole, and every later call starts on a block boundary, as O_DIRECT requires.
constexpr size_t LAZY_TABLE_READ_CHUNK_BYTES = size_t{ 1 } << 30;

std::string errno_text(int err) {
    return std::string(std::strerror(err)) + " (errno " + std::to_string(err) + ")";
}

const char * residency_name(lazy_residency residency) {
    return residency == lazy_residency::HOST ? "HOST" : "STORAGE";
}

std::string device_text(int device) {
    return device < 0 ? std::string("CPU") : std::to_string(device);
}

lazy_table_result refuse(const lazy_table_desc & desc, lazy_table_status status, const std::string & reason) {
    GGML_LOG_WARN("[LAZY-TABLE] %s refused: %s: %s\n", desc.name.c_str(), lazy_table_status_name(status),
                  reason.c_str());
    lazy_table_result result;
    result.status = status;
    result.reason = reason;
    return result;
}

// Why `desc` is not a table of `residency`, before anything is opened or allocated; empty when it is one.
std::string desc_invalid(const lazy_table_desc & desc, lazy_residency residency) {
    if (desc.name.empty()) {
        return "the table has no name";
    }
    if (desc.residency != residency) {
        return std::string("the plan's residency is ") + residency_name(desc.residency) + ", this constructor builds " +
               residency_name(residency);
    }
    if ((int) desc.type < 0 || desc.type >= GGML_TYPE_COUNT || ggml_blck_size(desc.type) <= 0 || desc.ne0 <= 0 ||
        desc.n_rows <= 0 || desc.ne0 % ggml_blck_size(desc.type) != 0) {
        return "type=" + std::to_string((int) desc.type) + " ne0=" + std::to_string(desc.ne0) +
               " n_rows=" + std::to_string(desc.n_rows) + " is not a row geometry";
    }
    const size_t row_size = ggml_row_size(desc.type, desc.ne0);
    if (desc.row_size != row_size) {
        return "row_size=" + std::to_string(desc.row_size) +
               " B, ggml_row_size(type, ne0)=" + std::to_string(row_size) + " B";
    }
    if (static_cast<uint64_t>(desc.n_rows) > SIZE_MAX / row_size ||
        row_size * static_cast<size_t>(desc.n_rows) != desc.table_bytes) {
        return "table_bytes=" + std::to_string(desc.table_bytes) + " B, row_size * n_rows=" + std::to_string(row_size) +
               " * " + std::to_string(desc.n_rows);
    }
    if (desc.file_path.empty()) {
        return "the table has no source file";
    }
    if (desc.file_offset > UINT64_MAX - desc.table_bytes) {
        return "offset=" + std::to_string(desc.file_offset) + " + size=" + std::to_string(desc.table_bytes) +
               " overflows";
    }
    if (desc.consumer_device >= GGML_SYCL_MAX_DEVICES) {
        return "consumer_device=" + std::to_string(desc.consumer_device) + " is no SYCL device";
    }
    return {};
}

// Opens the source for direct IO (D1): the body is the copy, and a page-cache copy beside it would be a second,
// unplanned one.
int open_direct(const std::string & path, int * err) {
    int fd = -1;
    do {
        fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    } while (fd < 0 && errno == EINTR);
    *err = fd < 0 ? errno : 0;
    return fd;
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

// The device a HOST_CPU record names: the consumer's, or device 0 for a CPU consumer. A HOST_CPU request names a device
// only for its owner's release coordinator, never for a queue (§3.3).
int record_device(const lazy_table_desc & desc) {
    return desc.consumer_device >= 0 ? desc.consumer_device : 0;
}

// One HOST_CPU record of `bytes` for the table: owner first, WEIGHT, MODEL lifetime under the table's model (§3.3).
// The body's charge is one of the two sites that may yield (§4.2): the load's thread holds no lock here, and the yield
// skips this load's own chunks.
allocation_result allocate_host_record(const lazy_table_desc & desc, size_t bytes) {
    alloc_request req{};
    req.device                           = record_device(desc);
    req.size                             = bytes;
    req.alignment                        = LAZY_IO_BLOCK_BYTES;
    req.intent.role                      = alloc_role::WEIGHT;
    req.intent.constraints.must_host_cpu = true;
    req.intent.lifetime                  = alloc_lifetime::MODEL;
    req.intent.owner                     = { alloc_owner_kind::MODEL, desc.model.model_id };
    req.may_yield                        = true;
    req.load_txn_id                      = desc.model.load_txn_id;
    return unified_allocate_owner(req);
}

// Reads [file_offset - head, file_offset - head + window) into `dst` with O_DIRECT, stopping once the table's last
// byte is in. Returns the bytes read; *err is the errno of a failed call, 0 otherwise.
size_t read_window(int fd, uint64_t aligned_offset, uint8_t * dst, size_t window, size_t need, int * err) {
    size_t done = 0;
    *err        = 0;
    while (done < need) {
        const size_t  ask = std::min(window - done, LAZY_TABLE_READ_CHUNK_BYTES);
        const ssize_t got = ::pread(fd, dst + done, ask, static_cast<off_t>(aligned_offset + done));
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            *err = errno;
            break;
        }
        if (got == 0) {
            break;  // end of file before the table's end: the caller names the short read
        }
        done += static_cast<size_t>(got);
    }
    return done;
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

lazy_table_result lazy_table::materialize(const lazy_table_desc &                 desc,
                                          lazy_residency                          residency,
                                          [[maybe_unused]] const test_variation & variation) {
    const std::string invalid = desc_invalid(desc, residency);
    if (!invalid.empty()) {
        return refuse(desc, lazy_table_status::INVALID_DESC, invalid);
    }

    int       open_err = 0;
    const int fd       = open_direct(desc.file_path, &open_err);
    if (fd < 0) {
        return refuse(desc, lazy_table_status::OPEN_FAILED,
                      "open(" + desc.file_path + ", O_DIRECT): " + errno_text(open_err));
    }
    // The table owns the descriptor from here: a STORAGE table keeps it, a HOST table closes it after its read, and
    // every refusal below closes it with the table.
    lazy_table_ref table(new lazy_table(desc));
    table->fd_ = fd;

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        return refuse(desc, lazy_table_status::OPEN_FAILED, "fstat(" + desc.file_path + "): " + errno_text(errno));
    }
    const uint64_t end = desc.file_offset + desc.table_bytes;
    if (static_cast<uint64_t>(st.st_size) < end) {
        return refuse(desc, lazy_table_status::SOURCE_SHORT,
                      "file_size=" + std::to_string(static_cast<uint64_t>(st.st_size)) +
                          " B < offset+size=" + std::to_string(end) + " B (offset=" + std::to_string(desc.file_offset) +
                          " size=" + std::to_string(desc.table_bytes) + ")");
    }

    if (residency == lazy_residency::HOST) {
        const size_t      head       = static_cast<size_t>(desc.file_offset % LAZY_IO_BLOCK_BYTES);
        const size_t      request    = lazy_table_host_body_request_bytes(desc.file_offset, desc.table_bytes);
        allocation_result body_alloc = request == SIZE_MAX ? allocation_result{} : allocate_host_record(desc, request);
        if (!body_alloc) {
            const size_t mapped = request == SIZE_MAX ? 0 : host_cpu_mapped_size(request);
            return refuse(desc, lazy_table_status::BODY_REFUSED,
                          "request=" + std::to_string(request) + " B (mapped " + std::to_string(mapped) +
                              " B) HOST_CPU on device " + std::to_string(record_device(desc)) +
                              ": allocation_error=" + std::to_string((int) body_alloc.error));
        }
        table->records_.push_back(mem_handle::from_owned_alloc(std::move(body_alloc.owner), GGML_LAYOUT_AOS));
        table->body_offset_ = head;

        // The body's bytes are a transient CPU view, resolved for this read alone (P2). A HOST_CPU record is the
        // CPU's: resolve() gives its bytes, and resolve(device) is empty, since no device may be handed it. The read
        // writes the whole window, so the view must authorize it: an extent of 0 is unknown, never unbounded.
        const resolved_ptr view = table->records_.front().resolve();
        if (!view) {
            return refuse(desc, lazy_table_status::BODY_REFUSED, "the body did not resolve for the CPU");
        }
        if (view.extent < request) {
            return refuse(
                desc, lazy_table_status::BODY_REFUSED,
                "the body resolved to " + std::to_string(view.extent) + " CPU bytes, want " + std::to_string(request));
        }
        // The read ends once it holds the table's last byte: a table that ends its file has a window past the end
        // (Qwen's PLE table by 832 B), and the short read there is the end of the file, not a failure.
        int read_err = 0;
#if defined(GGML_SYCL_PRIVATE_TESTING)
        // Fixture seam: the end-of-file fixture's positive control asks for the whole window; no constructor does.
        const size_t need = variation.read_whole_window ? request : head + desc.table_bytes;
#else
        const size_t need = head + desc.table_bytes;
#endif
        const size_t got =
            read_window(fd, desc.file_offset - head, static_cast<uint8_t *>(view.ptr), request, need, &read_err);
        if (got < need) {
            return refuse(desc, lazy_table_status::READ_FAILED,
                          "read " + std::to_string(got) + " of " + std::to_string(need) + " B from offset " +
                              std::to_string(desc.file_offset - head) +
                              (read_err != 0 ? ": " + errno_text(read_err) : std::string(": end of file")));
        }
        // A HOST table reads its rows from the body and never from the file again.
        close_fd(table->fd_);
        table->fd_ = -1;
    }

#if defined(GGML_SYCL_PRIVATE_TESTING)
    // Fixture seam: the activation fixture's one extra record, which no constructor asks for.
    if (variation.extra_record_bytes != 0) {
        allocation_result extra = allocate_host_record(desc, variation.extra_record_bytes);
        if (!extra) {
            return refuse(
                desc, lazy_table_status::BODY_REFUSED,
                "the fixture's extra record of " + std::to_string(variation.extra_record_bytes) + " B was refused");
        }
        table->records_.push_back(mem_handle::from_owned_alloc(std::move(extra.owner), GGML_LAYOUT_AOS));
    }
#endif

    // The activation check (th32 §10): the ledger's charge for the table's own records, the block ring excluded (it is
    // the pool's), against the plan's host_resident_bytes. Plan != reality is a defect, never tolerated. A STORAGE
    // table's plan figure is its row cache, whose chunks would be its records; until the cache is built that figure is
    // refused by name, which is a missing part and not a plan bug.
    if (residency == lazy_residency::STORAGE && desc.host_resident_bytes != 0) {
        return refuse(desc, lazy_table_status::ROW_CACHE_UNBUILT,
                      "planned host_resident_bytes=" + std::to_string(desc.host_resident_bytes) +
                          " B is the row cache (§6.3), which this build does not construct; records charge " +
                          std::to_string(table->charged_bytes()) + " B");
    }
    const size_t charged = table->charged_bytes();
    if (charged != desc.host_resident_bytes) {
        const std::string reason = "charged_bytes=" + std::to_string(charged) + " B over " +
                                   std::to_string(table->records_.size()) +
                                   " record(s), the block ring excluded, planned host_resident_bytes=" +
                                   std::to_string(desc.host_resident_bytes) + " B";
        GGML_LOG_WARN("[HOST-PLAN-BUG] lazy table %s activation check: %s\n", desc.name.c_str(), reason.c_str());
        if (ggml_sycl_strict_enabled()) {
            GGML_ABORT("lazy table %s failed its activation check under GGML_SYCL_STRICT_LEASES=1", desc.name.c_str());
        }
        lazy_table_result result;
        result.status = lazy_table_status::ACTIVATION_MISMATCH;
        result.reason = reason;
        return result;
    }

    GGML_LOG_INFO(
        "[LAZY-TABLE] %s residency=%s table=%zu B body=%zu B charged=%zu B planned=%zu B locked=%s consumer_layer=%d "
        "device=%s file=%u off=%llu\n",
        desc.name.c_str(), residency_name(residency), desc.table_bytes,
        residency == lazy_residency::HOST ? lazy_table_host_body_request_bytes(desc.file_offset, desc.table_bytes) : 0,
        charged, desc.host_resident_bytes, residency == lazy_residency::HOST ? "yes" : "none", desc.consumer_layer,
        device_text(desc.consumer_device).c_str(), desc.file_index, (unsigned long long) desc.file_offset);

    lazy_table_result result;
    result.table  = std::move(table);
    result.status = lazy_table_status::OK;
    return result;
}

lazy_table_result lazy_table_create_host(const lazy_table_desc & desc) {
    return lazy_table::materialize(desc, lazy_residency::HOST, lazy_table::test_variation{});
}

lazy_table_result lazy_table_create_storage(const lazy_table_desc & desc) {
    return lazy_table::materialize(desc, lazy_residency::STORAGE, lazy_table::test_variation{});
}

#if defined(GGML_SYCL_PRIVATE_TESTING)
lazy_table_result lazy_table_test_create_host_with_extra_record(const lazy_table_desc & desc, size_t extra_bytes) {
    lazy_table::test_variation variation;
    variation.extra_record_bytes = extra_bytes;
    return lazy_table::materialize(desc, lazy_residency::HOST, variation);
}

lazy_table_result lazy_table_test_create_host_reading_window(const lazy_table_desc & desc) {
    lazy_table::test_variation variation;
    variation.read_whole_window = true;
    return lazy_table::materialize(desc, lazy_residency::HOST, variation);
}
#endif

}  // namespace ggml_sycl
