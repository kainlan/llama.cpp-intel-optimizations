//
// The lazy table (th32 §3.2): one refcounted object per lazy weight table, whose rows are gathered on demand instead of
// loaded as a tensor. It holds the table's identity, geometry and source, the planner's immutable residency, and the
// memory that residency needs: a HOST table's body, the whole table read eagerly with O_DIRECT into the locked
// HOST_CPU tier (§3.3, D1), or a STORAGE table's owned O_DIRECT descriptor and its hot-row cache (§6.3). The model's
// descriptor buffer holds one reference and every outstanding ticket one; the last reference destroys the object,
// whose destructor unregisters the cache's yielder, drops its handles and closes its descriptor, once. That is the
// sole release point (P2).
//
// The descriptor buffer types live in ggml-sycl.cpp beside the other SYCL buffer types and are declared at the end of
// this header. A descriptor owns no bytes and no mem_handle, only a table reference: its base is a sentinel and every
// data call aborts naming the table.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "lazy-io-pool.hpp"
#include "lazy-row-cache.hpp"
#include "mem-handle.hpp"
#include "unified-cache.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml_sycl {

static_assert(sizeof(uintptr_t) == 8, "the descriptor sentinel is a 64-bit address");

// The descriptor buffer's base (th32 §3.2). It is non-null, so ggml_gallocr reads a bound lazy tensor as allocated and
// never places it in a compute buffer, and it is non-canonical under both 4-level and 5-level x86-64 paging (bit 62
// set, bit 63 clear), so a dereference faults instead of reading memory. A nominal size below 2^62 bytes keeps every
// address of the buffer in that range.
constexpr uintptr_t LAZY_TABLE_SENTINEL_BASE = uintptr_t{ 1 } << 62;

// What a table is made from (th32 §3.2). The source is the lazy-range record's, handed over by value at plan publish,
// never derived again; the residency, the consumer's executor and the resident bytes are the plan's.
struct lazy_table_desc {
    std::string           name;             // the tensor's name: the identity's and every line's
    ggml_sycl_cache_id    identity{};       // the usual weight key (owner = the model token, name = the tensor's)
    ggml_sycl_model_token model{};          // owns the HOST body (MODEL lifetime) and names its load transaction
    ggml_type             type        = GGML_TYPE_COUNT;
    int64_t               ne0         = 0;  // row elements
    int64_t               n_rows      = 0;
    size_t                row_size    = 0;  // ggml_row_size(type, ne0)
    uint32_t              file_index  = 0;  // the loader's file table index
    uint64_t              file_offset = 0;  // the record's offset of the table's first byte in that file
    size_t                table_bytes = 0;  // the record's size: row_size * n_rows
    std::string           file_path;        // the loader's file table entry for file_index
    lazy_residency        residency       = lazy_residency::HOST;
    int32_t               consumer_layer  = -1;
    int                   consumer_device = -1;  // the plan's executor for consumer_layer: a SYCL device, -1 the CPU
    size_t host_resident_bytes            = 0;   // the plan's figure for this table, which the activation check reads

    // STORAGE: the plan's row-cache slots, the floor and the yieldable share (§4.3 steps 4-5); HOST: 0.
    size_t cache_slots       = 0;
    // STORAGE: the plan's floor slots, never yielded (§6.4); at least 1, and at most cache_slots. HOST: 0.
    size_t cache_floor_slots = 0;

    // STORAGE: the plan's block ring (§3.3), the pool's and not the table's: block_ring_need, LAZY_BLOCK_RING_QD read
    // spans of the widest STORAGE row, which the materialization passes to block_ring_ensure, and block_ring_charge,
    // the ring the plan charged (§4.2): the need's mapped size for the model whose ensure creates the ring, 0 for a
    // model a live ring covers. HOST: 0.
    size_t block_ring_need   = 0;
    size_t block_ring_charge = 0;
};

// Why a materialization refused (th32 §3.2, §10). Every refusal logs one WARN naming the table, its status name and
// the figures that decided it: "[LAZY-TABLE] <table> refused: ...", or "[HOST-PLAN-BUG] lazy table <table> activation
// check: ..." for an activation mismatch, and "[HOST-PLAN-BUG] lazy table <table> row cache: ..." for a row-cache
// shortfall, which are plan bugs. The RING_ statuses are block_ring_ensure's refusals, one each, with its names; the
// ensure's own line carries the figures.
enum class lazy_table_status : uint8_t {
    OK = 0,
    INVALID_DESC,         // the descriptor's geometry, source or residency is not a table's
    OPEN_FAILED,          // the source file did not open with O_DIRECT
    SOURCE_SHORT,         // the file ends before offset + size
    BODY_REFUSED,         // the HOST body's owner-first allocation was refused
    READ_FAILED,          // the eager read failed or came up short
    ACTIVATION_MISMATCH,  // the records' charged bytes are not the plan's host_resident_bytes
    ROW_CACHE_SHORTFALL,  // the row cache's planned allocations were not all made: a step-4 shortfall (§6.4)
    RING_UNPINNED,        // no live IO pool, so no lease pins a block ring
    RING_TOO_NARROW,      // the plan's need or the table's span does not fit the ring (§3.3, §10)
    RING_CONCURRENT,      // another caller holds the ring's ensuring state
    RING_SHUT_DOWN,       // the IO pool is shut down
    RING_MISMATCH,        // the ring's bytes or charge differ from the plan's (§10's ring check)
    RING_REFUSED,         // the ring's planned allocation was refused
};

const char * lazy_table_status_name(lazy_table_status status);

class lazy_table;
struct lazy_table_result;

lazy_table_result lazy_table_create_host(const lazy_table_desc & desc);
lazy_table_result lazy_table_create_storage(const lazy_table_desc & desc);

#if defined(GGML_SYCL_PRIVATE_TESTING)
// lazy_table_create_host, with one extra HOST_CPU record of `extra_bytes` allocated beside the body and owned by the
// table like its own (th32 §10, T4's activation-check fixture: one extra chunk must fail the check).
lazy_table_result lazy_table_test_create_host_with_extra_record(const lazy_table_desc & desc, size_t extra_bytes);

// lazy_table_create_host, with the eager read asked for the whole O_DIRECT window instead of stopping once it holds
// the table's last byte (th32 §3.3). The positive control of T4's end-of-file fixture: a table that ends its file at an
// unaligned end has a window past the end of the file, and this read must refuse it.
lazy_table_result lazy_table_test_create_host_reading_window(const lazy_table_desc & desc);

// lazy_table_create_storage, with the row cache's slab cut into chunks of `chunk_bytes` instead of 256 MiB, so a
// fixture holds several chunks in a few MiB (§6.4). With `shortfall_unreported`, a row-cache shortfall refuses with
// the plain [LAZY-TABLE] WARN and no [HOST-PLAN-BUG] line: the shortfall leg's positive control.
lazy_table_result lazy_table_test_create_storage(const lazy_table_desc & desc,
                                                 size_t                  chunk_bytes,
                                                 bool                    shortfall_unreported);
#endif

class lazy_table {
  public:
    lazy_table(const lazy_table &)             = delete;
    lazy_table & operator=(const lazy_table &) = delete;

    const lazy_table_desc & desc() const { return desc_; }

    const char * name() const { return desc_.name.c_str(); }

    lazy_residency residency() const { return desc_.residency; }

    // The references held: the descriptor buffer's and every ticket's. A reading, never a release decision.
    uint32_t ref_count() const { return refs_.load(std::memory_order_acquire); }

    // HOST: the table's bytes, a retained view of the body at file_offset % LAZY_IO_BLOCK_BYTES. The body is HOST_CPU,
    // so the view is the CPU's: resolve() gives the bytes and resolve(device) is empty. STORAGE: an empty handle, since
    // a STORAGE table holds no rows.
    mem_handle rows() const;

    // STORAGE: the owned O_DIRECT descriptor the row cache reads through. HOST: -1, since the eager read closed it.
    int storage_fd() const { return fd_; }

    // STORAGE: the table's hot-row cache, built at the plan's cache_slots (§6.3), which the gather reads and fills.
    // HOST: null. Borrowed: valid while the table lives.
    lazy_row_cache * row_cache() const { return cache_.get(); }

    // The ledger charge of the table's own records and its row cache's allocations (th32 §10): the sum the activation
    // check compares with the plan.
    size_t charged_bytes() const;

  private:
    // References are taken and dropped only through lazy_table_ref, so release() is the sole release point (P2).
    friend class lazy_table_ref;
    friend lazy_table_result lazy_table_create_host(const lazy_table_desc & desc);
    friend lazy_table_result lazy_table_create_storage(const lazy_table_desc & desc);
#if defined(GGML_SYCL_PRIVATE_TESTING)
    friend lazy_table_result lazy_table_test_create_host_with_extra_record(const lazy_table_desc & desc,
                                                                           size_t                  extra_bytes);
    friend lazy_table_result lazy_table_test_create_host_reading_window(const lazy_table_desc & desc);
    friend lazy_table_result lazy_table_test_create_storage(const lazy_table_desc & desc,
                                                            size_t                  chunk_bytes,
                                                            bool                    shortfall_unreported);
#endif

    explicit lazy_table(const lazy_table_desc & desc) : desc_(desc) {}

    ~lazy_table();

    void retain() { refs_.fetch_add(1, std::memory_order_relaxed); }

    // Drops one reference; the last destroys the table.
    void release();

    // The fixtures' variations of a materialization: one extra HOST_CPU record (the activation fixture), an eager
    // read asked for the whole window (the end-of-file fixture's positive control), a row cache cut into small chunks,
    // and a row-cache shortfall refused without its [HOST-PLAN-BUG] line (the shortfall leg's positive control). A
    // fixture seam: the fields, and the branches of materialize that read them, are compiled only into test targets,
    // so a production variation has no fields, and the two constructors pass a default one in every build.
    struct test_variation {
#if defined(GGML_SYCL_PRIVATE_TESTING)
        size_t extra_record_bytes   = 0;
        bool   read_whole_window    = false;
        size_t cache_chunk_bytes    = 0;  // 0: LAZY_ROW_CACHE_CHUNK_BYTES
        bool   shortfall_unreported = false;
#endif
    };

    // Both constructors: validates `desc` for `residency`, builds the residency's memory, runs the activation check.
    static lazy_table_result materialize(const lazy_table_desc & desc,
                                         lazy_residency          residency,
                                         const test_variation &  variation);

    lazy_table_desc         desc_;
    // Every handle the table's resident bytes are (§3.3), each minted by unified_allocate_owner: a HOST table's body
    // is the first.
    std::vector<mem_handle> records_;
    size_t                  body_offset_ = 0;  // the table's first byte inside the body
    int                     fd_          = -1;
    std::atomic<uint32_t>   refs_{ 0 };

    // A STORAGE table's row cache (§6.3), and its slot in the ledger's yielder table, or -1 when it is not registered.
    std::unique_ptr<lazy_row_cache> cache_;
    int                             cache_yielder_slot_ = -1;
};

// An intrusive reference: a copy retains, a destruction or reassignment releases.
class lazy_table_ref {
  public:
    lazy_table_ref() = default;

    explicit lazy_table_ref(lazy_table * table) : table_(table) {
        if (table_ != nullptr) {
            table_->retain();
        }
    }

    lazy_table_ref(const lazy_table_ref & other) : lazy_table_ref(other.table_) {}

    lazy_table_ref(lazy_table_ref && other) noexcept : table_(other.table_) { other.table_ = nullptr; }

    lazy_table_ref & operator=(lazy_table_ref other) noexcept {
        lazy_table * const old = table_;
        table_                 = other.table_;
        other.table_           = old;
        return *this;
    }

    ~lazy_table_ref() { reset(); }

    void reset() {
        lazy_table * const table = table_;
        table_                   = nullptr;
        if (table != nullptr) {
            table->release();
        }
    }

    lazy_table * get() const { return table_; }

    lazy_table * operator->() const { return table_; }

    explicit operator bool() const { return table_ != nullptr; }

  private:
    lazy_table * table_ = nullptr;
};

struct lazy_table_result {
    lazy_table_ref    table;
    lazy_table_status status = lazy_table_status::INVALID_DESC;
    std::string       reason;  // the refusal's figures; empty on success

    explicit operator bool() const { return status == lazy_table_status::OK && table; }
};

}  // namespace ggml_sycl

// --- the descriptor buffer types (th32 §3.2), defined in ggml-sycl.cpp ---

// A SYCL-executed consumer's descriptor buffer type on `device`, the device the plan's query returns for the table's
// consumer layer. NULL for a device that is not a current SYCL device.
ggml_backend_buffer_type_t ggml_backend_sycl_lazy_buffer_type(int device);

// A CPU-executed consumer's descriptor buffer type: the host tier's, with the same sentinel contract.
ggml_backend_buffer_type_t ggml_backend_sycl_lazy_host_buffer_type();

// Whether `buffer` is a descriptor of either form, the no_alloc pass's zero-byte dummy included. The one predicate:
// nothing else derives "this is a lazy table" from a buffer's context.
bool ggml_backend_sycl_buffer_is_lazy(ggml_backend_buffer_t buffer);

// Binds `table` to a new descriptor buffer of `buft`, one of the two types above: a zero-byte descriptor whose nominal
// size is the table's bytes, usage WEIGHTS, holding one reference. NULL, after a WARN, when `buft` is not a descriptor
// type or `table` is empty.
ggml_backend_buffer_t ggml_backend_sycl_lazy_buffer_init(ggml_backend_buffer_type_t buft,
                                                         ggml_sycl::lazy_table_ref  table);

// The table a descriptor buffer holds; NULL for the no_alloc dummy and for any other buffer. Borrowed: valid while
// `buffer` lives; wrap it in a lazy_table_ref to hold it longer.
ggml_sycl::lazy_table * ggml_backend_sycl_lazy_buffer_table(ggml_backend_buffer_t buffer);

#if defined(GGML_SYCL_PRIVATE_TESTING)
// ggml_sycl_is_device_vram_buffer's answer for `tensor`.
bool ggml_sycl_test_is_device_vram_buffer(const ggml_tensor * tensor);

// The read today's callers make: `tensor`'s buffer context taken as a SYCL buffer context and its tier compared with
// DEVICE_VRAM, with no buffer-type test. Undefined for a buffer with another context; a fixture hands it one whose
// bytes it controls.
bool ggml_sycl_test_device_vram_context_read(const ggml_tensor * tensor);

// The positive control of T4's buffer-type table: while on, ggml_backend_sycl_buffer_is_lazy also answers true for the
// SYCL device buffer type, the defect the table must catch.
void ggml_sycl_test_lazy_predicate_too_broad(bool on);
#endif
