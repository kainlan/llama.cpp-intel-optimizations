//
// The lazy table (th32 §3.2, §3.3, §6.3, §6.4, §10): its two constructors, the HOST body's eager O_DIRECT read into the
// locked HOST_CPU tier, the STORAGE descriptor and the row cache it owns, the activation check and the sole release
// point. See lazy-table.hpp.
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

// The typed result of a refusal, after its WARN: refuse() and refuse_plan_bug() build theirs here.
lazy_table_result refusal(lazy_table_status status, const std::string & reason) {
    lazy_table_result result;
    result.status = status;
    result.reason = reason;
    return result;
}

lazy_table_result refuse(const lazy_table_desc & desc, lazy_table_status status, const std::string & reason) {
    GGML_LOG_WARN("[LAZY-TABLE] %s refused: %s: %s\n", desc.name.c_str(), lazy_table_status_name(status),
                  reason.c_str());
    return refusal(status, reason);
}

// A plan bug, plan != reality (§10), refuses with one [HOST-PLAN-BUG] WARN naming the table, the `subject` that fell
// short of its plan and the figures, aborts under GGML_SYCL_STRICT_LEASES=1, and otherwise returns the typed refusal.
lazy_table_result refuse_plan_bug(const lazy_table_desc & desc,
                                  const char *            subject,
                                  lazy_table_status       status,
                                  const std::string &     reason) {
    GGML_LOG_WARN("[HOST-PLAN-BUG] lazy table %s %s: %s: %s\n", desc.name.c_str(), subject,
                  lazy_table_status_name(status), reason.c_str());
    if (ggml_sycl_strict_enabled()) {
        GGML_ABORT("lazy table %s %s: %s under GGML_SYCL_STRICT_LEASES=1", desc.name.c_str(), subject,
                   lazy_table_status_name(status));
    }
    return refusal(status, reason);
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
    // A HOST body is the O_DIRECT window over the table (§3.3), so a window no size_t holds is a geometry fact, refused
    // here and never handed to the allocator.
    if (residency == lazy_residency::HOST &&
        lazy_table_host_body_request_bytes(desc.file_offset, desc.table_bytes) == SIZE_MAX) {
        return "the O_DIRECT window over offset=" + std::to_string(desc.file_offset) +
               " size=" + std::to_string(desc.table_bytes) + " does not fit a size_t";
    }
    if (desc.consumer_device < -1 || desc.consumer_device >= GGML_SYCL_MAX_DEVICES) {
        return "consumer_device=" + std::to_string(desc.consumer_device) + " is neither -1, the CPU, nor a SYCL device";
    }
    // A STORAGE table's row cache holds at least its floor, which the plan sizes and never yields (§6.4); a HOST table
    // has no cache.
    if (residency == lazy_residency::STORAGE &&
        (desc.cache_floor_slots == 0 || desc.cache_slots < desc.cache_floor_slots)) {
        return "cache_slots=" + std::to_string(desc.cache_slots) +
               " cache_floor_slots=" + std::to_string(desc.cache_floor_slots) +
               ": a STORAGE plan's cache holds a floor of at least one slot";
    }
    if (residency == lazy_residency::HOST && (desc.cache_slots != 0 || desc.cache_floor_slots != 0)) {
        return "cache_slots=" + std::to_string(desc.cache_slots) +
               " cache_floor_slots=" + std::to_string(desc.cache_floor_slots) + ": a HOST table has no row cache";
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

// One HOST_CPU request of `bytes` for the table (§3.3): owner first, MODEL lifetime under the table's model, named by
// its load, on record_device's device. Its callers differ only in the role, whether the charge may yield, and the
// alignment (0 is the allocator's default).
alloc_request host_record_request(const lazy_table_desc & desc,
                                  size_t                  bytes,
                                  alloc_role              role,
                                  bool                    may_yield,
                                  size_t                  alignment) {
    alloc_request req{};
    req.device                           = record_device(desc);
    req.size                             = bytes;
    req.alignment                        = alignment;
    req.intent.role                      = role;
    req.intent.constraints.must_host_cpu = true;
    req.intent.lifetime                  = alloc_lifetime::MODEL;
    req.intent.owner                     = { alloc_owner_kind::MODEL, desc.model.model_id };
    req.may_yield                        = may_yield;
    req.load_txn_id                      = desc.model.load_txn_id;
    return req;
}

// One HOST_CPU record of `bytes` for the table, a WEIGHT at the O_DIRECT block alignment. The body's charge is one of
// the two sites that may yield (§4.2): the load's thread holds no lock here, and the yield skips this load's own
// chunks.
allocation_result allocate_host_record(const lazy_table_desc & desc, size_t bytes) {
    return unified_allocate_owner(host_record_request(desc, bytes, alloc_role::WEIGHT, true, LAZY_IO_BLOCK_BYTES));
}

// One row-cache allocation through the owner entry (§6.4): its mem_handle, and the CPU view resolved from it once,
// which the cache reads only under a shard lock for a slot of a chunk not withdrawn. Destroying it drops the handle,
// the sole release point (P2), whose free credits the ledger; a chunk that was published yieldable leaves the
// ledger's yieldable bytes first.
class owner_cache_memory final : public lazy_row_cache_memory {
  public:
    owner_cache_memory(mem_handle handle, uint8_t * data, size_t bytes, size_t charged) :
        lazy_row_cache_memory(data, bytes, charged),
        handle_(std::move(handle)) {}

    ~owner_cache_memory() override {
        if (yieldable_) {
            host_ledger_yieldable_sub(charged_bytes());
        }
    }

    void set_yieldable() override {
        if (!yieldable_) {
            host_ledger_yieldable_add(charged_bytes());
            yieldable_ = true;
        }
    }

    void clear_yieldable() override {
        if (yieldable_) {
            host_ledger_yieldable_sub(charged_bytes());
            yieldable_ = false;
        }
    }

  private:
    mem_handle handle_;
    bool       yieldable_ = false;
};

// The production side of the cache's memory seam (§6.4): every allocation HOST_CPU, role LAZY_ROW_CACHE, owned by the
// table's model for its lifetime and named by its load. The cache's own charge passes may_yield false: a step-4
// shortfall is a plan bug, never masked by yielding chunks (§4.2). The fixture source in the host test is the only
// other implementation.
class owner_cache_source final : public lazy_row_cache_memory_source {
  public:
    explicit owner_cache_source(const lazy_table_desc & desc) : desc_(desc) {}

    std::unique_ptr<lazy_row_cache_memory> allocate(size_t bytes) override {
        allocation_result got =
            unified_allocate_owner(host_record_request(desc_, bytes, alloc_role::LAZY_ROW_CACHE, false, 0));
        if (!got) {
            last_error_ = (int) got.error;
            return nullptr;
        }
        mem_handle         handle = mem_handle::from_owned_alloc(std::move(got.owner), GGML_LAYOUT_AOS);
        const resolved_ptr view   = handle.resolve();
        if (!view || view.extent < bytes) {
            last_error_ = -1;  // the allocation did not resolve for the CPU; dropping the handle releases it
            return nullptr;
        }
        const size_t charged = record_charged_bytes(handle);
        return std::unique_ptr<lazy_row_cache_memory>(
            new owner_cache_memory(std::move(handle), static_cast<uint8_t *>(view.ptr), bytes, charged));
    }

    // The last refusal's allocation_error, or -1 when the allocation did not resolve for the CPU.
    int last_error() const { return last_error_; }

  private:
    const lazy_table_desc & desc_;
    int                     last_error_ = 0;
};

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
        case lazy_table_status::ROW_CACHE_SHORTFALL:
            return "lazy-table-row-cache-shortfall";
    }
    return "lazy-table-unknown";
}

lazy_table::~lazy_table() {
    // The yielder goes first, with no lock held: the unregister waits for a yield already running, so none runs
    // against the cache once it is gone (§4.2). Then the cache's allocations, the records and the descriptor.
    if (cache_yielder_slot_ >= 0) {
        host_ledger_unregister_yielder(cache_yielder_slot_);
        cache_yielder_slot_ = -1;
    }
    cache_.reset();
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
    // The cache's allocations, each counted at its registry row's charge as the records are.
    return sum + (cache_ != nullptr ? cache_->charged_bytes() : 0);
}

lazy_table_result lazy_table::materialize(const lazy_table_desc &                 desc,
                                          lazy_residency                          residency,
                                          [[maybe_unused]] const test_variation & variation) {
    const std::string invalid = desc_invalid(desc, residency);
    if (!invalid.empty()) {
        return refuse(desc, lazy_table_status::INVALID_DESC, invalid);
    }
    // A HOST body's request, a window desc_invalid has found fits a size_t; a STORAGE table allocates no body.
    const size_t request =
        residency == lazy_residency::HOST ? lazy_table_host_body_request_bytes(desc.file_offset, desc.table_bytes) : 0;

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
        const int fstat_err = errno;  // read before the reason's allocations can change it
        return refuse(desc, lazy_table_status::OPEN_FAILED, "fstat(" + desc.file_path + "): " + errno_text(fstat_err));
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
        allocation_result body_alloc = allocate_host_record(desc, request);
        if (!body_alloc) {
            return refuse(desc, lazy_table_status::BODY_REFUSED,
                          "request=" + std::to_string(request) + " B (mapped " +
                              std::to_string(host_cpu_mapped_size(request)) + " B) HOST_CPU on device " +
                              std::to_string(record_device(desc)) +
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

    if (residency == lazy_residency::STORAGE) {
        // The row cache at the plan's slots (§6.3, §6.4): the index and the slab's chunks, each its own owner-entry
        // allocation. A refused one is a step-4 shortfall, a plan bug: the cache released everything it made before it
        // returned, and the table refuses with one [HOST-PLAN-BUG] line naming the plan's figure and what was charged.
        lazy_row_cache_config config;
        config.name        = desc.name;
        config.row_size    = desc.row_size;
        config.n_rows      = desc.n_rows;
        config.cache_slots = desc.cache_slots;
        config.floor_slots = desc.cache_floor_slots;
        config.load_txn_id = desc.model.load_txn_id;
#if defined(GGML_SYCL_PRIVATE_TESTING)
        // Fixture seam: the owner-path fixtures cut the slab into small chunks; no constructor does.
        if (variation.cache_chunk_bytes != 0) {
            config.chunk_bytes = variation.cache_chunk_bytes;
        }
#endif
        owner_cache_source   source(desc);
        lazy_row_cache_build built = lazy_row_cache::build(config, source);
        if (built.status == lazy_row_cache_status::SHORTFALL) {
            const std::string reason =
                "planned=" + std::to_string(desc.host_resident_bytes) +
                " B charged=" + std::to_string(built.charged_bytes) + " B, " + std::to_string(built.allocations_made) +
                " of " + std::to_string(built.allocations_planned) +
                " allocations made, allocation_error=" + std::to_string(source.last_error()) + ": " + built.reason;
#if defined(GGML_SYCL_PRIVATE_TESTING)
            // Fixture seam: the shortfall leg's positive control refuses without the line; no constructor does.
            if (variation.shortfall_unreported) {
                return refuse(desc, lazy_table_status::ROW_CACHE_SHORTFALL, reason);
            }
#endif
            return refuse_plan_bug(desc, "row cache", lazy_table_status::ROW_CACHE_SHORTFALL, reason);
        }
        if (!built) {
            return refuse(desc, lazy_table_status::INVALID_DESC, "the row cache refused its plan: " + built.reason);
        }
        table->cache_ = std::move(built.cache);
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

    // The activation check (th32 §10): the ledger's charge for the table's own records and its row cache's
    // allocations, the block ring excluded (it is the pool's), against the plan's host_resident_bytes. Plan != reality
    // is a defect, never tolerated.
    const size_t charged = table->charged_bytes();
    if (charged != desc.host_resident_bytes) {
        const std::string reason =
            "charged_bytes=" + std::to_string(charged) + " B over " + std::to_string(table->records_.size()) +
            " record(s) and the row cache, the block ring excluded, planned host_resident_bytes=" +
            std::to_string(desc.host_resident_bytes) + " B";
        return refuse_plan_bug(desc, "activation check", lazy_table_status::ACTIVATION_MISMATCH, reason);
    }

    // The cache's yielder registers once the table is whole, through T2's per-cache hook (§4.2), so a refusal above
    // leaves none behind. Its chunks above the floor are published yieldable first and the yielder registered after,
    // so no yield can reach a chunk while it is published: no lock orders the two, the order does. §4.2 gives the same
    // from the other side: the lifecycle admits one load transaction per process, and only its loader thread, this
    // one, makes yielding charges. The chunks count as yieldable only while the yielder is registered: a cache the
    // ledger's slots cannot take is unpublished again and runs with none (the ledger's WARN names it).
    size_t yieldable = 0;
    if (table->cache_ != nullptr) {
        table->cache_->publish_yieldable();
        table->cache_yielder_slot_ = host_ledger_register_yielder(&lazy_row_cache::yield_entry, table->cache_.get());
        if (table->cache_yielder_slot_ >= 0) {
            yieldable = table->cache_->yieldable_bytes();
        } else {
            table->cache_->unpublish_yieldable();
        }
    }

    // locked= is the tier's property: every resident record and cache allocation is HOST_CPU, which is mlocked (§3.1).
    GGML_LOG_INFO(
        "[LAZY-TABLE] %s residency=%s table=%zu B body=%zu B charged=%zu B planned=%zu B locked=%s cache_slots=%zu "
        "floor_slots=%zu yieldable=%zu B consumer_layer=%d device=%s file=%u off=%llu\n",
        desc.name.c_str(), residency_name(residency), desc.table_bytes, request, charged, desc.host_resident_bytes,
        charged != 0 ? "yes" : "none", desc.cache_slots, desc.cache_floor_slots, yieldable, desc.consumer_layer,
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

lazy_table_result lazy_table_test_create_storage(const lazy_table_desc & desc,
                                                 size_t                  chunk_bytes,
                                                 bool                    shortfall_unreported) {
    lazy_table::test_variation variation;
    variation.cache_chunk_bytes    = chunk_bytes;
    variation.shortfall_unreported = shortfall_unreported;
    return lazy_table::materialize(desc, lazy_residency::STORAGE, variation);
}
#endif

}  // namespace ggml_sycl
