//
// The lazy table object (th32 §3.2, §3.3, §10; row T4): the refcounted lazy_table and its two constructors, the HOST
// body's eager O_DIRECT read into the locked HOST_CPU tier, the STORAGE descriptor and its hot-row cache's owner path
// (§6.3, §6.4: the chunks charged through the owner entry, the yielder registered, a yield on the loader thread, the
// step-4 shortfall's [HOST-PLAN-BUG] line), the activation check, the descriptor buffer types with their own
// interface, the guard at ggml_sycl_get_data_ptr_slow's entry, and ggml_sycl_is_device_vram_buffer's descriptor test
// before its cast. No libllama is involved: every table is built
// from a fixture-written file through ggml_sycl::lazy_table_create_host / lazy_table_create_storage.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. A table's memory comes from the unified cache's owner-first entry, so the binary links the private
// fixtures, initialises SYCL and is registered gpu-serial, serial, with the selector pinned to the B50.
//
// Two arms run in their own processes, started by fork and execve before this process initialises SYCL: the
// descriptor guard (row T4: exactly one "[LAZY-PLAN-BUG] data pointer requested for lazy table" line and exit 134,
// where the guard absent prints none) and the activation check's strict abort. Every other arm runs here.
//
// Every arm names its engagement check. An arm whose engagement check fails is VOID, and a VOID arm fails the binary
// like a FAIL. Every arm runs, so one failure does not hide the rest.
//
// --mutant predicate-too-broad runs the buffer-type table alone with ggml_backend_sycl_buffer_is_lazy also claiming
// the SYCL device type, and the table must FAIL on that row: its registered positive control. Without the descriptor
// test no other type's answer can change, so the table needs a mutant to show it can fail.
//
// --mutant read-whole-window runs arm (2) alone with its end-of-file table built by a read that asks for the whole
// O_DIRECT window, and the arm must FAIL naming that geometry: the end-of-file sub-check's registered positive
// control. A read that stops at the table's last byte and one that reads the whole window differ only when the window
// runs past the end of the file, so the sub-check needs a mutant to show it can fail.
//
// --mutant shortfall-unreported runs arm (15) alone with the row-cache shortfall refused through the plain
// [LAZY-TABLE] WARN and no [HOST-PLAN-BUG] line, and the arm must FAIL counting no such line: the shortfall leg's
// registered positive control.

#include "../common.hpp"
#include "../host-cpu-mapping.hpp"
#include "../lazy-table.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "lazy-table-test-util.hpp"
#include "sycl-test-skip.hpp"

#include <fcntl.h>
#include <limits.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace host_ledger_test_harness;
using namespace lazy_table_test;
using ggml_sycl::host_cpu_mapped_size;
using ggml_sycl::LAZY_IO_BLOCK_BYTES;
using ggml_sycl::lazy_residency;
using ggml_sycl::LAZY_ROW_CACHE_CHUNK_BYTES;
using ggml_sycl::lazy_table_create_host;
using ggml_sycl::lazy_table_create_storage;
using ggml_sycl::lazy_table_desc;
using ggml_sycl::lazy_table_host_body_request_bytes;
using ggml_sycl::lazy_table_result;
using ggml_sycl::lazy_table_status;

namespace {

// --- the fixture file ---

// Table A sits at the unaligned offset of a real table (DeepSeek's blk.14.engram_embd, off=4032 in its dry run), with
// the row of another (Qwen's PLE table): Q8_0 rows of 160 elements, 170 bytes each, 24680 of them. Table B starts at
// the same offset with 12335 rows, 2096950 bytes: below one 2 MiB huge page on its own, past it once the O_DIRECT
// window's head is counted, so its body maps 4 MiB.
constexpr uint64_t k_table_offset = 4032;
constexpr int64_t  k_ne0          = 160;
constexpr int64_t  k_rows_a       = 24680;
constexpr int64_t  k_rows_b       = 12335;
constexpr uint64_t k_file_bytes   = 8 * MiB;
constexpr uint64_t k_model_id     = 0x7a4b1e;
constexpr uint64_t k_load_txn_id  = 0x4a11;

// The arm-(2) figures worked by hand, not through the helper: table A is 4195600 bytes, its window is
// ceil((4032 + 4195600) / 4096) * 4096 = 4202496 bytes, mapped to 6 MiB; table B is 2096950 bytes, its window
// ceil((4032 + 2096950) / 4096) * 4096 = 2101248 bytes, mapped to 4 MiB.
constexpr size_t k_table_a_bytes  = 4195600;
constexpr size_t k_window_a_bytes = 4202496;
constexpr size_t k_mapped_a_bytes = 6 * MiB;
constexpr size_t k_table_b_bytes  = 2096950;
constexpr size_t k_window_b_bytes = 2101248;
constexpr size_t k_mapped_b_bytes = 4 * MiB;

const char * k_guard_phrase = "[LAZY-PLAN-BUG] data pointer requested for lazy table";

// Arm (2)'s end-of-file table, in Qwen's geometry (its PLE table sits at offset 192 and ends its file): table A's rows
// at offset 192 in a second file that ends at the table's last byte, 4195792 bytes, 1488 bytes into a block. Its
// window is ceil((192 + 4195600) / 4096) * 4096 = 4198400 bytes, 2608 past the end of the file, mapped to 6 MiB.
constexpr uint64_t k_eof_offset       = 192;
constexpr uint64_t k_eof_file_bytes   = 4195792;
constexpr size_t   k_eof_window_bytes = 4198400;
constexpr size_t   k_eof_past_end     = 2608;

// A STORAGE table's row cache, in table A's geometry (§6.3, §6.4): its 24680 rows cached whole, with a floor of 1024
// slots. The index is 4 x ((4 x 24680 + 2) / 3) + 8 x 24680 = 329068 B, mapped to 2 MiB; the slab is table A's
// 4195600 B. At the production chunk of 256 MiB the slab is one chunk, mapped to 6 MiB, which holds the floor, so
// the cache yields nothing; at the fixtures' 2 MiB chunk it is two whole chunks and a 1296 B remainder, 2 MiB mapped
// each, and the two above the floor's are yieldable. Either way the plan's figure is 8 MiB, and the bytes requested
// under role LAZY_ROW_CACHE are the index's and the slab's, 4524668.
constexpr size_t   k_cache_floor_slots     = 1024;
constexpr size_t   k_cache_index_bytes     = 329068;
constexpr size_t   k_cache_chunk_bytes     = 2 * MiB;
constexpr size_t   k_storage_planned_bytes = 8 * MiB;
constexpr size_t   k_cache_yieldable_bytes = 4 * MiB;
constexpr size_t   k_cache_role_bytes      = k_cache_index_bytes + k_table_a_bytes;
constexpr uint64_t k_other_load_txn_id     = 0x5e11;

// A ledger budget far above the fixture's bytes, seeded in every process, so no charge logs a would-refuse WARN and
// strict mode meets only the check under test.
constexpr uint64_t k_ledger_budget = 64 * GiB;

std::string g_fixture_path;
std::string g_eof_fixture_path;
bool        g_direct_unsupported       = false;  // the fixture's filesystem refuses O_DIRECT: the reading arms are VOID
bool        g_mutant_read_whole_window = false;  // --mutant read-whole-window
bool        g_mutant_shortfall_unreported = false;  // --mutant shortfall-unreported

uint8_t pattern_byte(uint64_t offset) {
    return static_cast<uint8_t>((offset * 2654435761ull) >> 13);
}

// Writes `bytes` of the pattern, each byte the pattern of its own offset, so any table in any fixture file reads the
// same bytes at the same offset.
bool write_fixture(const std::string & path, uint64_t bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        return false;
    }
    std::vector<uint8_t> chunk(1 * MiB);
    bool                 ok = true;
    for (uint64_t base = 0; base < bytes && ok; base += chunk.size()) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(chunk.size(), bytes - base));
        for (size_t i = 0; i < n; ++i) {
            chunk[i] = pattern_byte(base + i);
        }
        ok = ::write(fd, chunk.data(), n) == static_cast<ssize_t>(n);
    }
    ok = ok && ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

// Whether the fixture's filesystem takes O_DIRECT, which D1 requires and which tmpfs refuses with EINVAL.
bool probe_direct(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    ::close(fd);
    return true;
}

size_t row_size() {
    return ggml_row_size(GGML_TYPE_Q8_0, k_ne0);
}

// The block ring a STORAGE table of these rows needs (§3.3): LAZY_BLOCK_RING_QD read spans of one row, 512 KiB. The
// process's coverage pin ensures it before any arm runs, as a model's load would, so a STORAGE materialization finds
// the ring live and its plan charged none.
size_t ring_need() {
    return ggml_sycl::LAZY_BLOCK_RING_QD * ggml_sycl::lazy_block_ring_span(row_size());
}

// The coverage pin the STORAGE arms' materializations find live: a lease on a pool whose ring holds ring_need().
std::unique_ptr<ggml_sycl::lazy_test_coverage_pin> make_ring_pin() {
    return std::unique_ptr<ggml_sycl::lazy_test_coverage_pin>(
        new ggml_sycl::lazy_test_coverage_pin(ggml_sycl::ggml_sycl_test_make_coverage_pin(ring_need())));
}

// A STORAGE table's resident bytes at `slots` slots with its slab cut at `chunk` bytes: the whole chunks, the
// remainder and the index, each counted at host_cpu_mapped_size, as the ledger charges them and the planner sums them
// (§3.3).
size_t storage_resident_bytes(size_t slots, size_t chunk) {
    const size_t slab  = slots * row_size();
    size_t       bytes = (slab / chunk) * host_cpu_mapped_size(chunk);
    if (slab % chunk != 0) {
        bytes += host_cpu_mapped_size(slab % chunk);
    }
    return bytes + host_cpu_mapped_size(ggml_sycl::lazy_row_cache_index_raw_bytes(slots));
}

lazy_table_desc fixture_desc(const char *   name,
                             uint64_t       offset,
                             int64_t        n_rows,
                             lazy_residency residency,
                             int            consumer_device) {
    lazy_table_desc d;
    d.name                 = name;
    d.identity.valid       = true;
    d.identity.model_id    = k_model_id;
    d.identity.load_txn_id = k_load_txn_id;
    d.identity.type        = GGML_TYPE_Q8_0;
    d.model.model_id       = k_model_id;
    d.model.load_txn_id    = k_load_txn_id;
    d.type                 = GGML_TYPE_Q8_0;
    d.ne0                  = k_ne0;
    d.n_rows               = n_rows;
    d.row_size             = row_size();
    d.file_index           = 0;
    d.file_offset          = offset;
    d.table_bytes          = row_size() * static_cast<size_t>(n_rows);
    d.file_path            = g_fixture_path;
    d.residency            = residency;
    d.consumer_layer       = 1;
    d.consumer_device      = consumer_device;
    d.host_resident_bytes  = residency == lazy_residency::HOST ?
                                 host_cpu_mapped_size(lazy_table_host_body_request_bytes(offset, d.table_bytes)) :
                                 storage_resident_bytes(static_cast<size_t>(n_rows), LAZY_ROW_CACHE_CHUNK_BYTES);
    if (residency == lazy_residency::STORAGE) {
        d.cache_slots       = static_cast<size_t>(n_rows);
        d.cache_floor_slots = k_cache_floor_slots;
        // The ring the process's coverage pin made, which this table's plan found live, so it charged none (§10).
        d.block_ring_need   = ring_need();
        d.block_ring_charge = 0;
    }
    return d;
}

uint64_t host_cpu_live() {
    uint64_t live = 0;
    return ggml_sycl_test_host_cpu_live_bytes(&live) ? live : UINT64_MAX;
}

// The status a materialization returned, by its printed name.
std::string status_name(const lazy_table_result & t) {
    return ggml_sycl::lazy_table_status_name(t.status);
}

// A refusal the environment caused, not the table: no O_DIRECT, or a lock limit below the body or the row cache.
bool environment_refusal(const lazy_table_result & r, size_t locked_bytes, std::string & why) {
    if (r.status == lazy_table_status::OPEN_FAILED && g_direct_unsupported) {
        why = "the fixture's filesystem refuses O_DIRECT: " + r.reason;
        return true;
    }
    if (r.status == lazy_table_status::BODY_REFUSED || r.status == lazy_table_status::ROW_CACHE_SHORTFALL) {
        rlimit lim{};
        if (getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY && lim.rlim_cur < locked_bytes) {
            why = "RLIMIT_MEMLOCK is " + num(lim.rlim_cur) + " B, below the body or the row cache: " + r.reason;
            return true;
        }
    }
    return false;
}

// What an arm records when its table refuses for a reason the environment did not cause: an arm whose subject is the
// materialization FAILs, and an arm that only needs a built table to reach its subject is VOID.
enum class other_refusal { FAIL, VOID };

// Whether `t` built. When it did not, records the arm's verdict naming `what` and the refusal: VOID for a refusal the
// environment caused (no O_DIRECT, or a lock limit below `locked_bytes`), `other` for any other.
bool table_built(arm_result &              r,
                 const lazy_table_result & t,
                 size_t                    locked_bytes,
                 other_refusal             other,
                 const std::string &       what = "the table") {
    if (t) {
        return true;
    }
    std::string why;
    if (environment_refusal(t, locked_bytes, why)) {
        void_arm(r, what + " refused, by the environment: " + why);
        return false;
    }
    const std::string text = what + " refused: " + status_name(t) + ": " + t.reason;
    if (other == other_refusal::FAIL) {
        fail(r, text);
    } else {
        void_arm(r, text);
    }
    return false;
}

// The table's bytes, compared with the file's at the record's offset. Returns the first mismatching index, or
// SIZE_MAX when every byte matches.
size_t first_mismatch(const uint8_t * rows, uint64_t file_offset, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        if (rows[i] != pattern_byte(file_offset + i)) {
            return i;
        }
    }
    return SIZE_MAX;
}

// --- the children ---

// Child of the descriptor-guard arm: binds a WEIGHTS tensor to a host-form descriptor over table A and asks
// ggml_sycl_get_data_ptr_slow for its pointer. With the guard it aborts there; without it the call runs on.
int guard_child() {
    ggml_sycl_test_host_ledger_store_budget(k_ledger_budget);
    lazy_table_result r = lazy_table_create_host(
        fixture_desc("guard.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, /*consumer_device=*/-1));
    if (!r) {
        std::fprintf(stderr, "GUARD-CHILD-SKIP the table did not build: %s\n", r.reason.c_str());
        return SYCL_TEST_SKIP;
    }
    ggml_backend_buffer_t buffer =
        ggml_backend_sycl_lazy_buffer_init(ggml_backend_sycl_lazy_host_buffer_type(), r.table);
    ggml_init_params params = { ggml_tensor_overhead() * 4, nullptr, /*no_alloc=*/true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    t      = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k_ne0, k_rows_a);
    ggml_set_name(t, "guard.lazy");
    if (buffer == nullptr ||
        ggml_backend_tensor_alloc(buffer, t, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "GUARD-CHILD-SKIP the descriptor did not bind\n");
        return SYCL_TEST_SKIP;
    }
    std::fprintf(stderr, "GUARD-CHILD bound extra=%s\n", t->extra == nullptr ? "null" : "set");
    std::fflush(stderr);
    void * ptr = ggml_sycl_get_data_ptr_slow(t, 0);
    std::fprintf(stderr, "GUARD-CHILD returned %p\n", ptr);
    return 0;
}

// Child of the strict arm: under GGML_SYCL_STRICT_LEASES=1 the activation check's mismatch aborts.
int strict_child() {
    ggml_sycl_test_host_ledger_store_budget(k_ledger_budget);
    lazy_table_result r = ggml_sycl::lazy_table_test_create_host_with_extra_record(
        fixture_desc("strict.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, /*consumer_device=*/0), 1);
    std::fprintf(stderr, "STRICT-CHILD returned status=%s\n", ggml_sycl::lazy_table_status_name(r.status));
    return r.status == lazy_table_status::OPEN_FAILED || r.status == lazy_table_status::BODY_REFUSED ? SYCL_TEST_SKIP :
                                                                                                       0;
}

// Runs this binary as `mode` on the fixture. The child's environment is this one's with GGML_NO_BACKTRACE=1, so an
// abort prints its message and attaches no gdb, and with `extra_env` when given.
child_run spawn(const char * mode, const char * extra_env) {
    std::vector<std::string> sets = { "GGML_NO_BACKTRACE=1" };
    if (extra_env != nullptr) {
        sets.emplace_back(extra_env);
    }
    return run_child_strings({ "test-sycl-lazy-table", mode, g_fixture_path }, child_environment(sets),
                             std::chrono::milliseconds(120000));
}

child_run g_guard_run;
child_run g_strict_run;

// --- the arms ---

// (1) The body request is the O_DIRECT window over the table, worked by hand for the fixture's two tables and the
// aligned edges. Engagement: none needed, the helper is pure.
arm_result arm_body_request_window() {
    arm_result r;
    expect(r, row_size() * k_rows_a == k_table_a_bytes && row_size() * k_rows_b == k_table_b_bytes,
           "the fixture's tables are not the sizes the hand figures assume");
    expect(r, lazy_table_host_body_request_bytes(k_table_offset, k_table_a_bytes) == k_window_a_bytes,
           "table A's window is " + num(lazy_table_host_body_request_bytes(k_table_offset, k_table_a_bytes)) +
               " B, want " + num(k_window_a_bytes));
    expect(r, lazy_table_host_body_request_bytes(k_table_offset, k_table_b_bytes) == k_window_b_bytes,
           "table B's window is " + num(lazy_table_host_body_request_bytes(k_table_offset, k_table_b_bytes)) +
               " B, want " + num(k_window_b_bytes));
    expect(r, host_cpu_mapped_size(k_table_b_bytes) == 2 * MiB && host_cpu_mapped_size(k_window_b_bytes) == 4 * MiB,
           "table B no longer crosses a huge page once its head is counted");
    expect(r, lazy_table_host_body_request_bytes(8192, 4096) == 4096,
           "an aligned one-block table's window is not 4096");
    expect(r, lazy_table_host_body_request_bytes(0, 1) == 4096, "a one-byte table's window is not one block");
    expect(r, lazy_table_host_body_request_bytes(4095, 2) == 8192, "a table straddling a block is not two blocks");
    // Both sides of the size_t edge: 1 + (SIZE_MAX - 4096) is 2^64 - 4096, a whole number of blocks and the largest
    // window size_t holds, so it is exact; one byte more rounds up to 2^64, which saturates.
    expect(r, lazy_table_host_body_request_bytes(1, SIZE_MAX - 4096) == SIZE_MAX - 4095,
           "the largest window size_t holds is not exact");
    expect(r, lazy_table_host_body_request_bytes(1, SIZE_MAX - 4095) == SIZE_MAX,
           "an overflowing window does not saturate");
    return r;
}

// Arm (2)'s end-of-file sub-check: a table that ends its file at an unaligned end, as Qwen's PLE table does, has a
// window past the end of the file, and the read must end once it holds the table's last byte, so the short read at the
// end of the file is not a failure. Engagement: as (2). Mutant: --mutant read-whole-window.
void expect_eof_table_reads(arm_result & r) {
    const std::string geometry = "the table that ends its file at an unaligned end (offset " + num(k_eof_offset) +
                                 ", " + num(k_table_a_bytes) + " B, the window " + num(k_eof_past_end) +
                                 " B past the end of the file)";
    expect(r,
           k_eof_offset + k_table_a_bytes == k_eof_file_bytes && k_eof_file_bytes % LAZY_IO_BLOCK_BYTES != 0 &&
               lazy_table_host_body_request_bytes(k_eof_offset, k_table_a_bytes) == k_eof_window_bytes &&
               k_eof_window_bytes - k_eof_file_bytes == k_eof_past_end,
           "the end-of-file fixture is not the geometry the hand figures assume");
    lazy_table_desc d   = fixture_desc("host.eof.lazy", k_eof_offset, k_rows_a, lazy_residency::HOST, 0);
    d.file_path         = g_eof_fixture_path;
    lazy_table_result t = g_mutant_read_whole_window ? ggml_sycl::lazy_table_test_create_host_reading_window(d) :
                                                       lazy_table_create_host(d);
    if (!table_built(r, t, k_mapped_a_bytes, other_refusal::FAIL, geometry)) {
        return;
    }
    const ggml_sycl::resolved_ptr view = t.table->rows().resolve();
    if (!view) {
        fail(r, geometry + ": the rows view does not resolve");
        return;
    }
    const size_t bad = first_mismatch(static_cast<const uint8_t *>(view.ptr), k_eof_offset, k_table_a_bytes);
    expect(r, bad == SIZE_MAX, geometry + ": row byte " + num(bad) + " differs from the file's");
}

// (2) A HOST table reads its rows eagerly with O_DIRECT into one locked HOST_CPU record, owned by its model, and keeps
// no descriptor; its rows are the CPU's, never a device's; a table that ends its file reads whole too (the sub-check
// above). Engagement: the environment takes O_DIRECT and locks the body; a refusal it caused is VOID, any other
// refusal FAILs.
arm_result arm_host_eager_read() {
    arm_result      r;
    const uint64_t  live_before = host_cpu_live();
    lazy_table_desc d           = fixture_desc("host.a.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    log_clear();
    lazy_table_result t = lazy_table_create_host(d);
    if (!table_built(r, t, k_mapped_a_bytes, other_refusal::FAIL)) {
        return r;
    }
    expect(r, t.table->ref_count() == 1, "the result holds " + num(t.table->ref_count()) + " references, want 1");
    expect(r, t.table->storage_fd() == -1, "a HOST table kept descriptor " + std::to_string(t.table->storage_fd()));
    expect(r, t.table->charged_bytes() == k_mapped_a_bytes,
           "the records charge " + num(t.table->charged_bytes()) + " B, want " + num(k_mapped_a_bytes));
    expect(
        r, host_cpu_live() - live_before == k_mapped_a_bytes,
        "the live HOST_CPU rows grew by " + num(host_cpu_live() - live_before) + " B, want " + num(k_mapped_a_bytes));

    ggml_sycl::mem_handle         rows = t.table->rows();
    ggml_sycl::alloc_metadata     meta{};
    const ggml_sycl::resolved_ptr view = rows.resolve();
    expect(r, rows.owner_metadata(&meta), "the rows view retains no owner");
    expect(r, meta.tier == ggml_sycl::alloc_tier::HOST_CPU, "the body is not HOST_CPU");
    expect(r, meta.role == ggml_sycl::alloc_role::WEIGHT, "the body's role is not WEIGHT");
    expect(r,
           meta.lifetime == ggml_sycl::alloc_lifetime::MODEL && meta.owner.kind == ggml_sycl::alloc_owner_kind::MODEL &&
               meta.owner.id == k_model_id,
           "the body is not owned by the table's model");
    expect(r, meta.size == k_window_a_bytes,
           "the body requested " + num(meta.size) + " B, want the window " + num(k_window_a_bytes));
    if (!view) {
        fail(r, "the rows view does not resolve for the CPU");
        return r;
    }
    expect(r, !rows.resolve(0), "a device was handed the HOST_CPU rows: resolve(0) is not empty");
    expect(r, reinterpret_cast<uintptr_t>(view.ptr) % LAZY_IO_BLOCK_BYTES == k_table_offset % LAZY_IO_BLOCK_BYTES,
           "the rows do not sit at the window's head inside the body");
    const size_t bad = first_mismatch(static_cast<const uint8_t *>(view.ptr), k_table_offset, k_table_a_bytes);
    expect(r, bad == SIZE_MAX, "row byte " + num(bad) + " differs from the file's");
    const auto lines = log_lines_containing("[LAZY-TABLE] host.a.lazy residency=HOST");
    expect(r, lines.size() == 1, "the materialization printed " + num(lines.size()) + " [LAZY-TABLE] lines, want 1");
    if (lines.size() == 1) {
        expect(r, printed(lines[0], "charged") == k_mapped_a_bytes && printed(lines[0], "planned") == k_mapped_a_bytes,
               "the [LAZY-TABLE] line does not carry charged= and planned=: " + lines[0]);
    }
    expect_eof_table_reads(r);
    return r;
}

// (3) A table whose head pushes its window past a huge page maps the larger figure, and its last row is the file's.
// Engagement: as (2).
arm_result arm_host_window_crosses_huge_page() {
    arm_result        r;
    lazy_table_result t =
        lazy_table_create_host(fixture_desc("host.b.lazy", k_table_offset, k_rows_b, lazy_residency::HOST, 0));
    if (!table_built(r, t, k_mapped_b_bytes, other_refusal::FAIL)) {
        return r;
    }
    expect(r, t.table->charged_bytes() == k_mapped_b_bytes,
           "the records charge " + num(t.table->charged_bytes()) + " B, want " + num(k_mapped_b_bytes));
    const ggml_sycl::resolved_ptr view = t.table->rows().resolve();
    if (!view) {
        fail(r, "the rows view does not resolve");
        return r;
    }
    const size_t    last_row = row_size() * static_cast<size_t>(k_rows_b - 1);
    const uint8_t * rows     = static_cast<const uint8_t *>(view.ptr);
    const size_t    bad      = first_mismatch(rows + last_row, k_table_offset + last_row, row_size());
    expect(r, bad == SIZE_MAX, "the last row's byte " + num(bad) + " differs from the file's");
    return r;
}

// (4) The last reference destroys the table and releases its body, once; a copy of the reference is a reference.
// Engagement: the built table's body is live.
arm_result arm_last_reference_releases() {
    arm_result     r;
    const uint64_t before = host_cpu_live();
    {
        lazy_table_result t =
            lazy_table_create_host(fixture_desc("release.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0));
        if (!table_built(r, t, k_mapped_a_bytes, other_refusal::FAIL)) {
            return r;
        }
        if (host_cpu_live() - before != k_mapped_a_bytes) {
            void_arm(r, "the built table's body is not among the live HOST_CPU rows, so its release cannot be seen");
            return r;
        }
        {
            ggml_sycl::lazy_table_ref copy = t.table;
            expect(r, t.table->ref_count() == 2, "a copied reference counts " + num(t.table->ref_count()));
        }
        expect(r, t.table->ref_count() == 1, "the dropped copy left " + num(t.table->ref_count()) + " references");
        const uint64_t live_after_copy = host_cpu_live() - before;
        expect(r, live_after_copy == k_mapped_a_bytes,
               "a dropped copy released the body: the live HOST_CPU rows are " + num(live_after_copy) +
                   " B above the start, want " + num(k_mapped_a_bytes));
    }
    expect(r, host_cpu_live() == before,
           "the last reference left " + num(host_cpu_live() - before) + " HOST_CPU bytes live, want 0");
    return r;
}

// The open flags /proc reports for `fd`, or 0 when the line is absent.
unsigned long fd_flags(int fd) {
    std::ifstream in("/proc/self/fdinfo/" + std::to_string(fd));
    std::string   line;
    while (std::getline(in, line)) {
        if (line.rfind("flags:", 0) == 0) {
            return std::strtoul(line.c_str() + 6, nullptr, 8);
        }
    }
    return 0;
}

std::string fd_target(int fd) {
    char          buf[PATH_MAX];
    const ssize_t n = ::readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

// (5) A STORAGE table holds an O_DIRECT descriptor on its source and its row cache, built through the owner entry at
// the plan's slots (§6.3), and its last reference closes the descriptor and releases the cache. At the production
// chunk of 256 MiB table A's slab is one chunk, the floor's, so the cache charges the plan's 8 MiB and publishes no
// yieldable bytes. Engagement: the environment takes O_DIRECT and locks the cache; a refusal it caused is VOID, any
// other refusal FAILs.
arm_result arm_storage_holds_its_descriptor() {
    arm_result        r;
    const uint64_t    before           = host_cpu_live();
    const uint64_t    yieldable_before = ggml_sycl::host_ledger_yieldable_bytes();
    lazy_table_result t =
        lazy_table_create_storage(fixture_desc("storage.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0));
    if (!table_built(r, t, k_storage_planned_bytes, other_refusal::FAIL)) {
        return r;
    }
    expect(r, storage_resident_bytes(k_rows_a, LAZY_ROW_CACHE_CHUNK_BYTES) == k_storage_planned_bytes,
           "the hand figure for the production cache is not the planner's sum");
    const int fd = t.table->storage_fd();
    expect(r, fd >= 0, "a STORAGE table holds no descriptor");
    char * real = realpath(g_fixture_path.c_str(), nullptr);
    expect(r, real != nullptr && fd_target(fd) == real, "the descriptor is not the source: " + fd_target(fd));
    std::free(real);
    expect(r, (fd_flags(fd) & O_DIRECT) != 0, "the descriptor is not O_DIRECT");
    expect(r, !t.table->rows().valid(), "a STORAGE table holds rows");
    const ggml_sycl::lazy_row_cache * cache = t.table->row_cache();
    if (cache == nullptr) {
        fail(r, "a STORAGE table built no row cache");
        return r;
    }
    expect(r, cache->cache_slots() == static_cast<size_t>(k_rows_a) && cache->floor_slots() == k_cache_floor_slots,
           "the cache holds " + num(cache->cache_slots()) + " slots with a floor of " + num(cache->floor_slots()) +
               ", want the plan's " + num(k_rows_a) + " and " + num(k_cache_floor_slots));
    expect(r, cache->n_chunks() == 1 && cache->n_floor_chunks() == 1,
           "the production cache has " + num(cache->n_chunks()) + " chunks, " + num(cache->n_floor_chunks()) +
               " of them the floor's, want 1 and 1");
    expect(
        r, t.table->charged_bytes() == k_storage_planned_bytes,
        "the table charges " + num(t.table->charged_bytes()) + " B, want the plan's " + num(k_storage_planned_bytes));
    expect(
        r, host_cpu_live() - before == k_storage_planned_bytes,
        "the live HOST_CPU rows grew by " + num(host_cpu_live() - before) + " B, want " + num(k_storage_planned_bytes));
    expect(r, ggml_sycl::host_ledger_yieldable_bytes() == yieldable_before,
           "a cache whose one chunk is the floor's published yieldable bytes");
    t.table.reset();
    errno = 0;
    expect(r, ::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "the last reference left the descriptor open");
    expect(r, host_cpu_live() == before,
           "the last reference left " + num(host_cpu_live() - before) + " HOST_CPU bytes live, want 0");
    return r;
}

// (6) A source that ends before offset + size refuses, naming both numbers, and leaves nothing live. Engagement: the
// file opened, since the refusal names the file's size.
arm_result arm_source_short_refuses() {
    arm_result     r;
    const uint64_t before = host_cpu_live();
    const uint64_t offset = k_file_bytes - LAZY_IO_BLOCK_BYTES;  // the table runs past the end of the file
    log_clear();
    lazy_table_result t = lazy_table_create_host(fixture_desc("short.lazy", offset, k_rows_a, lazy_residency::HOST, 0));
    if (t.status == lazy_table_status::OPEN_FAILED) {
        void_arm(r, "the source did not open: " + t.reason);
        return r;
    }
    const uint64_t end = offset + k_table_a_bytes;
    expect(
        r, t.status == lazy_table_status::SOURCE_SHORT,
        std::string("the status is ") + ggml_sycl::lazy_table_status_name(t.status) + ", want lazy-table-source-short");
    expect(r,
           t.reason.find("file_size=" + num(k_file_bytes)) != std::string::npos &&
               t.reason.find("offset+size=" + num(end)) != std::string::npos,
           "the refusal does not name both numbers: " + t.reason);
    expect(r, log_count_with("[LAZY-TABLE] short.lazy refused: lazy-table-source-short") == 1,
           "the refusal printed no WARN naming it");
    expect(r, !t.table && host_cpu_live() == before, "the refusal left a table or HOST_CPU bytes live");
    return r;
}

// (7) A descriptor that is not a table refuses before anything is opened or allocated: a wrong row size, table bytes
// that are not row_size * n_rows, the other residency, a HOST window no size_t holds, a consumer that is neither -1
// nor a SYCL device, a STORAGE cache with no floor or a floor above its slots, and a HOST plan with a cache.
// Engagement: none needed; every case but the first and the third names a file that does not exist, so a refusal
// after the open would read lazy-table-open-failed.
arm_result arm_invalid_desc_refuses() {
    arm_result      r;
    lazy_table_desc d = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    d.row_size += 1;
    const lazy_table_result row = lazy_table_create_host(d);
    expect(r, row.status == lazy_table_status::INVALID_DESC,
           "a wrong row_size gave " + status_name(row) + ", want lazy-table-invalid-desc");
    d           = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    d.file_path = "/nonexistent/path/that/is/never/opened";
    d.table_bytes += 1;
    const lazy_table_result bytes = lazy_table_create_host(d);
    expect(r, bytes.status == lazy_table_status::INVALID_DESC,
           "a table_bytes that is not row_size * n_rows gave " + status_name(bytes) + ", want lazy-table-invalid-desc");
    d = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0);
    const lazy_table_result storage = lazy_table_create_host(d);
    expect(r, storage.status == lazy_table_status::INVALID_DESC,
           "the HOST constructor gave " + status_name(storage) +
               " for a table the plan made STORAGE, want lazy-table-invalid-desc");

    // Arm (1)'s saturated window: 2^62 - 1009 F32 rows of 4 B at offset 4032 end 4 B short of 2^64, so offset + size
    // fits, but the window rounds past 2^64.
    d             = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    d.file_path   = "/nonexistent/path/that/is/never/opened";
    d.type        = GGML_TYPE_F32;
    d.ne0         = 1;
    d.n_rows      = (int64_t{ 1 } << 62) - 1009;
    d.row_size    = ggml_row_size(GGML_TYPE_F32, 1);
    d.table_bytes = d.row_size * static_cast<size_t>(d.n_rows);
    expect(r,
           d.file_offset <= UINT64_MAX - d.table_bytes &&
               lazy_table_host_body_request_bytes(d.file_offset, d.table_bytes) == SIZE_MAX,
           "the saturated case is not a window past size_t over an offset + size that fits");
    const lazy_table_result window = lazy_table_create_host(d);
    expect(r, window.status == lazy_table_status::INVALID_DESC,
           "a window no size_t holds gave " + status_name(window) + ", want lazy-table-invalid-desc");

    d                             = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, -7);
    d.file_path                   = "/nonexistent/path/that/is/never/opened";
    const lazy_table_result other = lazy_table_create_host(d);
    expect(r, other.status == lazy_table_status::INVALID_DESC,
           "consumer_device=-7 gave " + status_name(other) + ", want lazy-table-invalid-desc");

    // A STORAGE plan whose cache has no floor or a floor above its slots, and a HOST plan with a cache (§6.4).
    d                   = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0);
    d.file_path         = "/nonexistent/path/that/is/never/opened";
    d.cache_floor_slots = 0;
    const lazy_table_result no_floor = lazy_table_create_storage(d);
    expect(r, no_floor.status == lazy_table_status::INVALID_DESC,
           "a STORAGE plan with no floor gave " + status_name(no_floor) + ", want lazy-table-invalid-desc");
    d.cache_floor_slots                 = d.cache_slots + 1;
    const lazy_table_result floor_above = lazy_table_create_storage(d);
    expect(r, floor_above.status == lazy_table_status::INVALID_DESC,
           "a STORAGE floor above its slots gave " + status_name(floor_above) + ", want lazy-table-invalid-desc");
    d                   = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    d.file_path         = "/nonexistent/path/that/is/never/opened";
    d.cache_slots       = k_cache_floor_slots;
    d.cache_floor_slots = k_cache_floor_slots;
    const lazy_table_result host_cache = lazy_table_create_host(d);
    expect(r, host_cache.status == lazy_table_status::INVALID_DESC,
           "a HOST plan with a row cache gave " + status_name(host_cache) + ", want lazy-table-invalid-desc");
    return r;
}

// (8) Row T4's activation fixture: a materialization that allocates one extra chunk fails the check, naming both
// numbers, as a typed refusal outside strict mode, and frees what it allocated. Engagement: the same table without
// the extra chunk passes, and the extra chunk is real (the charged figure is the body's plus its mapped size).
arm_result arm_activation_extra_chunk_refuses() {
    arm_result     r;
    const uint64_t before = host_cpu_live();
    {
        lazy_table_result control =
            lazy_table_create_host(fixture_desc("activation.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0));
        if (!table_built(r, control, k_mapped_a_bytes, other_refusal::VOID,
                         "the control table without the extra chunk")) {
            return r;
        }
    }
    log_clear();
    lazy_table_result t = ggml_sycl::lazy_table_test_create_host_with_extra_record(
        fixture_desc("activation.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0), 1);
    const size_t charged = k_mapped_a_bytes + host_cpu_mapped_size(1);
    if (t) {
        fail(r, "the table with one extra chunk passed its activation check (charged " + num(t.table->charged_bytes()) +
                    " B, planned " + num(k_mapped_a_bytes) + " B)");
        return r;
    }
    expect(r, t.status == lazy_table_status::ACTIVATION_MISMATCH,
           std::string("the status is ") + ggml_sycl::lazy_table_status_name(t.status) +
               ", want lazy-table-activation-mismatch: " + t.reason);
    const auto lines = log_lines_containing("[HOST-PLAN-BUG] lazy table activation.lazy activation check");
    expect(r, lines.size() == 1, "the mismatch printed " + num(lines.size()) + " [HOST-PLAN-BUG] lines, want 1");
    if (lines.size() == 1) {
        expect(r, lines[0].find("lazy-table-activation-mismatch") != std::string::npos,
               "the [HOST-PLAN-BUG] line does not name its status: " + lines[0]);
        expect(r, printed(lines[0], "charged_bytes") == charged,
               "the line's charged_bytes is not the body plus one chunk (" + num(charged) + "): " + lines[0]);
        expect(r, printed(lines[0], "host_resident_bytes") == k_mapped_a_bytes,
               "the line's host_resident_bytes is not the plan's (" + num(k_mapped_a_bytes) + "): " + lines[0]);
    }
    expect(r, host_cpu_live() == before, "the refusal left " + num(host_cpu_live() - before) + " HOST_CPU bytes live");
    return r;
}

// (9) Under GGML_SYCL_STRICT_LEASES=1 the same mismatch aborts after its WARN. Engagement: the child built the
// table's body, so it reached the check.
arm_result arm_activation_strict_aborts() {
    arm_result r;
    if (!g_strict_run.ran) {
        fail(r, "the strict child did not finish within its bound");
        return r;
    }
    if (WIFEXITED(g_strict_run.status) && WEXITSTATUS(g_strict_run.status) == SYCL_TEST_SKIP) {
        void_arm(r, "the strict child could not build its body");
        return r;
    }
    const size_t charged = k_mapped_a_bytes + host_cpu_mapped_size(1);
    expect(r, WIFSIGNALED(g_strict_run.status) && WTERMSIG(g_strict_run.status) == SIGABRT,
           "the strict child did not abort (status " + std::to_string(g_strict_run.status) + ")");
    expect(r,
           text_count_with(g_strict_run.err, "[HOST-PLAN-BUG] lazy table strict.lazy activation check",
                           ("charged_bytes=" + num(charged)).c_str()) == 1,
           "the strict child printed no [HOST-PLAN-BUG] line naming charged_bytes=" + num(charged));
    expect(r, text_count_with(g_strict_run.err, "STRICT-CHILD returned") == 0, "the strict child returned");
    return r;
}

ggml_tensor * new_table_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k_ne0, k_rows_a);
    ggml_set_name(t, name);
    return t;
}

// (10) A WEIGHTS tensor bound to a descriptor of either form keeps tensor->extra null: the null init_tensor registers
// nothing. The descriptor holds one reference, has the sentinel base and the table's nominal size, is not host, and
// refuses to be allocated. Engagement: the table built.
arm_result arm_bound_tensor_keeps_extra_null() {
    arm_result        r;
    lazy_table_result t =
        lazy_table_create_host(fixture_desc("bind.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0));
    if (!table_built(r, t, k_mapped_a_bytes, other_refusal::VOID)) {
        return r;
    }
    ggml_init_params params = { ggml_tensor_overhead() * 4, nullptr, /*no_alloc=*/true };
    ggml_context *   ctx    = ggml_init(params);

    const struct {
        const char *               form;
        ggml_backend_buffer_type_t buft;
    } forms[] = {
        { "SYCL form", ggml_backend_sycl_lazy_buffer_type(0)     },
        { "host form", ggml_backend_sycl_lazy_host_buffer_type() },
    };

    for (const auto & f : forms) {
        if (f.buft == nullptr) {
            void_arm(r, std::string("the ") + f.form + " buffer type is null");
            continue;
        }
        expect(r, ggml_backend_buft_alloc_buffer(f.buft, 4096) == nullptr,
               std::string("the ") + f.form + " descriptor type allocated a buffer");
        ggml_backend_buffer_t dummy = ggml_backend_buft_alloc_buffer(f.buft, 0);
        expect(r,
               dummy != nullptr && ggml_backend_sycl_buffer_is_lazy(dummy) &&
                   ggml_backend_sycl_lazy_buffer_table(dummy) == nullptr,
               std::string("the ") + f.form + " no_alloc dummy is not a lazy buffer without a table");
        ggml_backend_buffer_free(dummy);

        ggml_backend_buffer_t buffer = ggml_backend_sycl_lazy_buffer_init(f.buft, t.table);
        if (buffer == nullptr) {
            fail(r, std::string("the ") + f.form + " descriptor did not bind a built table");
            continue;
        }
        ggml_tensor * tensor = new_table_tensor(ctx, f.form);
        expect(r, t.table->ref_count() == 2,
               std::string("the ") + f.form + " descriptor holds no reference: the table counts " +
                   num(t.table->ref_count()) + ", want 2");
        const ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(buffer);
        expect(r, usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
               std::string("the ") + f.form + " descriptor's usage is " + std::to_string((int) usage) +
                   ", want WEIGHTS (" + std::to_string((int) GGML_BACKEND_BUFFER_USAGE_WEIGHTS) + ")");
        expect(r, ggml_backend_buffer_get_size(buffer) == k_table_a_bytes,
               std::string("the ") + f.form + " descriptor's size is " + num(ggml_backend_buffer_get_size(buffer)) +
                   " B, want the table's " + num(k_table_a_bytes));
        expect(r, !ggml_backend_buffer_is_host(buffer), std::string("the ") + f.form + " descriptor is host");
        expect(r,
               reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer)) == ggml_sycl::LAZY_TABLE_SENTINEL_BASE,
               std::string("the ") + f.form + " descriptor's base is not the sentinel");
        expect(r,
               ggml_backend_sycl_buffer_is_lazy(buffer) && ggml_backend_sycl_lazy_buffer_table(buffer) == t.table.get(),
               std::string("the ") + f.form + " descriptor does not hold its table");
        const ggml_status st = ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
        expect(r, st == GGML_STATUS_SUCCESS, std::string("the ") + f.form + " binding failed");
        expect(r, tensor->extra == nullptr, std::string("the ") + f.form + " binding set tensor->extra");
        ggml_backend_buffer_free(buffer);
        expect(r, t.table->ref_count() == 1,
               std::string("the ") + f.form + " descriptor's free kept its reference: the table counts " +
                   num(t.table->ref_count()) + ", want 1");
    }
    ggml_free(ctx);
    return r;
}

// (11) Row T4's guard fixture, run in its own process: exactly one guard line naming the table, then exit 134.
// Engagement: the child bound its tensor and reported extra null before the call.
arm_result arm_descriptor_guard() {
    arm_result r;
    if (!g_guard_run.ran) {
        fail(r, "the guard child did not finish within its bound");
        return r;
    }
    if (WIFEXITED(g_guard_run.status) && WEXITSTATUS(g_guard_run.status) == SYCL_TEST_SKIP) {
        void_arm(r, "the guard child could not build and bind its table");
        return r;
    }
    if (text_count_with(g_guard_run.err, "GUARD-CHILD bound extra=null") != 1) {
        void_arm(r, "the guard child did not bind its tensor with extra null before the call");
        return r;
    }
    const size_t lines = text_count_with(g_guard_run.err, k_guard_phrase);
    expect(r, lines == 1, "the child printed " + num(lines) + " guard lines, want exactly 1");
    expect(r, text_count_with(g_guard_run.err, k_guard_phrase, "guard.lazy") == lines,
           "the guard line does not name the table");
    expect(r, WIFSIGNALED(g_guard_run.status) && WTERMSIG(g_guard_run.status) == SIGABRT,
           "the child did not exit 134 (status " + std::to_string(g_guard_run.status) + ")");
    expect(r, text_count_with(g_guard_run.err, "GUARD-CHILD returned") == 0,
           "ggml_sycl_get_data_ptr_slow returned a pointer for a lazy table");
    return r;
}

// A context whose bytes the fixture controls: zero, so today's read of it is DEVICE_VRAM (alloc_tier 0) for every
// buffer type, and a descriptor test that fired on a type would flip that type's answer to false. It is smaller than a
// SYCL buffer context, which holds GGML_SYCL_MAX_DEVICES mem_handles and their metadata. The read stays inside it
// because it touches only managed_meta.tier, which lies within the first 128 bytes of ggml_backend_sycl_buffer_context.
alignas(64) unsigned char g_zero_context[4096] = {};

// A buffer of `buft` over g_zero_context and a tensor on it, so the fixture controls every byte a cast of the context
// reads. Not copyable: the tensor points at the buffer beside it.
struct zeroed_tensor {
    explicit zeroed_tensor(ggml_backend_buffer_type_t buft) {
        buffer.buft    = buft;
        buffer.context = g_zero_context;
        buffer.size    = sizeof(g_zero_context);
        tensor.buffer  = &buffer;
    }

    zeroed_tensor(const zeroed_tensor &)             = delete;
    zeroed_tensor & operator=(const zeroed_tensor &) = delete;

    ggml_backend_buffer buffer{};
    ggml_tensor         tensor{};
};

struct buft_row {
    std::string                name;
    ggml_backend_buffer_type_t buft;
    bool                       descriptor;
};

// (12) is_device_vram_buffer answers false for a descriptor before its cast, for both forms: over a real bound
// descriptor, whose context is a table reference, and over a zeroed context, where the cast would read
// DEVICE_VRAM. Engagement: the table built, and today's read of the zeroed context is true.
arm_result arm_vram_answer_for_descriptors() {
    arm_result        r;
    lazy_table_result t =
        lazy_table_create_host(fixture_desc("vram.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0));
    if (!table_built(r, t, k_mapped_a_bytes, other_refusal::VOID)) {
        return r;
    }
    const buft_row rows[] = {
        { "SYCL_Lazy",      ggml_backend_sycl_lazy_buffer_type(0),     true },
        { "SYCL_Lazy_Host", ggml_backend_sycl_lazy_host_buffer_type(), true },
    };
    for (const buft_row & row : rows) {
        if (row.buft == nullptr) {
            void_arm(r, row.name + " is null");
            continue;
        }
        ggml_backend_buffer_t bound = ggml_backend_sycl_lazy_buffer_init(row.buft, t.table);
        if (bound == nullptr) {
            fail(r, row.name + " did not bind a built table");
            continue;
        }
        ggml_tensor real{};
        real.buffer = bound;
        expect(r, !ggml_sycl_test_is_device_vram_buffer(&real), row.name + " bound to a table answered true");
        ggml_backend_buffer_free(bound);

        zeroed_tensor zeroed(row.buft);
        if (!ggml_sycl_test_device_vram_context_read(&zeroed.tensor)) {
            void_arm(r, "today's read of a zeroed context is not DEVICE_VRAM, so the zeroed row cannot fail");
            continue;
        }
        expect(r, !ggml_sycl_test_is_device_vram_buffer(&zeroed.tensor),
               row.name + " over a zeroed context answered true: the cast ran before the descriptor test");
    }
    return r;
}

// Every buffer type the backend exposes, through its factories, the descriptor types last. Duplicates (a factory that
// returns another's type) are kept: each is a separate way to reach the predicate.
std::vector<buft_row> exposed_bufts() {
    ggml_backend_dev_t    dev0 = ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0);
    std::vector<buft_row> rows = {
        { "SYCL0 (ggml_backend_sycl_buffer_type)", ggml_backend_sycl_buffer_type(0), false },
        { "device default (ggml_backend_dev_buffer_type)", ggml_backend_dev_buffer_type(dev0), false },
        { "tiered KV (ggml_backend_sycl_kv_buffer_type)", ggml_backend_sycl_kv_buffer_type(0), false },
        { "tiered KV (ggml_backend_sycl_kv_buffer_type_from_dev)", ggml_backend_sycl_kv_buffer_type_from_dev(dev0),
         false },
        { "split (ggml_backend_sycl_split_buffer_type)", ggml_backend_sycl_split_buffer_type(0, nullptr), false },
        { "SYCL_Host (ggml_backend_sycl_host_buffer_type)", ggml_backend_sycl_host_buffer_type(), false },
        { "SYCL_Host (ggml_backend_sycl_host_buffer_type_for_device)",
         ggml_backend_sycl_host_buffer_type_for_device(dev0), false },
        { "device host (ggml_backend_dev_host_buffer_type)", ggml_backend_dev_host_buffer_type(dev0), false },
        { "SYCL_Mapped (ggml_backend_sycl_mapped_buffer_type_for_device)",
         ggml_backend_sycl_mapped_buffer_type_for_device(dev0), false },
        { "SYCL_KV_Host (ggml_backend_sycl_kv_host_buffer_type)", ggml_backend_sycl_kv_host_buffer_type(), false },
        { "CPU activation (ggml_backend_sycl_cpu_activation_buffer_type)",
         ggml_backend_sycl_cpu_activation_buffer_type(), false },
        { "SYCL_HostCompute (ggml_backend_sycl_host_compute_buffer_type)",
         ggml_backend_sycl_host_compute_buffer_type(0), false },
        { "CPU offload compute (ggml_backend_sycl_cpu_offload_compute_buffer_type)",
         ggml_backend_sycl_cpu_offload_compute_buffer_type(0), false },
        { "SYCL0 test double (ggml_backend_sycl_buffer_type_make_for_testing)",
         ggml_backend_sycl_buffer_type_make_for_testing(0), false },
        { "SYCL_Host test double (ggml_backend_sycl_host_buffer_type_make_for_testing)",
         ggml_backend_sycl_host_buffer_type_make_for_testing(), false },
        // Last among the others: the factory latches its tensor-parallel setup once per process. With no device list
        // and the one device the selector leaves, it enables nothing and returns its type for device 0.
        { "TP (ggml_backend_sycl_tp_buffer_type)", ggml_backend_sycl_tp_buffer_type(0, nullptr), false },
        { "SYCL_Lazy (ggml_backend_sycl_lazy_buffer_type)", ggml_backend_sycl_lazy_buffer_type(0), true },
        { "SYCL_Lazy_Host (ggml_backend_sycl_lazy_host_buffer_type)", ggml_backend_sycl_lazy_host_buffer_type(), true },
    };
    return rows;
}

// (13) No other buffer type's is_device_vram_buffer answer changes, over every type the backend exposes:
// each row's buffer carries a zeroed context, so today's read is true for every type and the descriptor test is the
// only thing that can make the answer false. Without the descriptor test no answer can change, so the arm alone
// cannot show it can fail; --mutant predicate-too-broad is its positive control. Engagement: every factory returned a
// type and today's read of every row is true.
arm_result arm_vram_answer_unchanged_for_other_types() {
    arm_result r;
    for (const buft_row & row : exposed_bufts()) {
        if (row.buft == nullptr) {
            void_arm(r, row.name + " returned no buffer type");
            continue;
        }
        zeroed_tensor zeroed(row.buft);
        const bool    today     = ggml_sycl_test_device_vram_context_read(&zeroed.tensor);
        const bool    answer    = ggml_sycl_test_is_device_vram_buffer(&zeroed.tensor);
        const bool    predicate = ggml_backend_sycl_buffer_is_lazy(&zeroed.buffer);
        if (!today) {
            void_arm(r, row.name + ": today's read of a zeroed context is not DEVICE_VRAM");
            continue;
        }
        const bool want = row.descriptor ? false : today;
        if (answer != want || predicate != row.descriptor) {
            fail(r, row.name + ": is_device_vram_buffer answered " + std::to_string(answer) + ", today's read " +
                        std::to_string(today) + ", want " + std::to_string(want) +
                        "; ggml_backend_sycl_buffer_is_lazy answered " + std::to_string(predicate));
        }
    }
    return r;
}

const char * k_table_arm_name = "(13) no other buffer type's is_device_vram_buffer answer changes";
const char * k_eager_arm_name = "(2) a HOST table reads its rows eagerly into one locked HOST_CPU record";

}  // namespace

// (14) Row T4b's owner path (§6.4): a STORAGE table at the fixtures' 2 MiB chunk charges its index and its three chunks
// through the owner entry as HOST_CPU under role LAZY_ROW_CACHE, at the plan's figure; the two chunks above the
// floor's are yieldable, and its yielder is registered. A charge of another load that does not fit, asked on this
// thread, frees the last chunk with no wait and then fits: the chunk leaves the live HOST_CPU rows, the ledger is
// credited its charge, the yield logs its chunk count and the asking load, and the cache counts it. A charge of the
// table's own load that does not fit frees nothing. The last reference releases every allocation, takes the
// yieldable bytes back and unregisters the yielder: the claimed slots are one more while the table lives and back to
// their count before it after. Engagement: the cache holds three chunks, one the floor's, and the budget is stored at
// the committed figure before each charge, so neither fits without a yield.
// The role reading needs device 0's cache: the reader is VOID in a process that has none (§3.1), and the HOST_CPU owner
// path builds none, so the arm creates it before any baseline and scores an unreadable reading VOID, never as a figure.
arm_result arm_storage_cache_owner_path() {
    arm_result r;
    if (ggml_sycl::get_unified_cache_for_device(0) == nullptr) {
        void_arm(r, "device 0 has no cache, so the LAZY_ROW_CACHE role reading would be VOID");
        return r;
    }
    const uint64_t live0      = host_cpu_live();
    const uint64_t committed0 = ggml_sycl::host_ledger_committed();
    const uint64_t yieldable0 = ggml_sycl::host_ledger_yieldable_bytes();
    const int      slots0     = ggml_sycl::host_ledger_yielder_slots_claimed_for_testing();
    uint64_t       role0      = 0;
    const bool     role_read0 =
        ggml_sycl::unified_cache_role_bytes_for_testing(0, ggml_sycl::alloc_role::LAZY_ROW_CACHE, &role0);
    lazy_table_desc d     = fixture_desc("storage.owner.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0);
    d.host_resident_bytes = storage_resident_bytes(k_rows_a, k_cache_chunk_bytes);
    expect(r, d.host_resident_bytes == k_storage_planned_bytes,
           "the hand figure for the 2 MiB-chunk cache is not the planner's sum: " + num(d.host_resident_bytes));
    expect(r, ggml_sycl::lazy_row_cache_index_raw_bytes(k_rows_a) == k_cache_index_bytes,
           "the index of " + num(k_rows_a) + " slots is not the hand figure " + num(k_cache_index_bytes));
    {
        lazy_table_result t = ggml_sycl::lazy_table_test_create_storage(d, k_cache_chunk_bytes, false);
        if (!table_built(r, t, k_storage_planned_bytes, other_refusal::FAIL)) {
            return r;
        }
        ggml_sycl::lazy_row_cache * cache = t.table->row_cache();
        if (cache == nullptr || cache->n_chunks() != 3 || cache->n_floor_chunks() != 1) {
            void_arm(r, "the table's cache is not three chunks with one the floor's, so no yield can be scored");
            return r;
        }
        expect(r, t.table->charged_bytes() == k_storage_planned_bytes,
               "the table charges " + num(t.table->charged_bytes()) + " B, want the plan's " +
                   num(k_storage_planned_bytes));
        expect(r, host_cpu_live() - live0 == k_storage_planned_bytes,
               "the live HOST_CPU rows grew by " + num(host_cpu_live() - live0) + " B, want " +
                   num(k_storage_planned_bytes));
        expect(r, ggml_sycl::host_ledger_committed() - committed0 == k_storage_planned_bytes,
               "the ledger's committed bytes grew by " + num(ggml_sycl::host_ledger_committed() - committed0) +
                   " B, want " + num(k_storage_planned_bytes));
        expect(r, ggml_sycl::host_ledger_yieldable_bytes() - yieldable0 == k_cache_yieldable_bytes,
               "the ledger's yieldable bytes grew by " + num(ggml_sycl::host_ledger_yieldable_bytes() - yieldable0) +
                   " B, want the two chunks above the floor's, " + num(k_cache_yieldable_bytes));
        const int slots1 = ggml_sycl::host_ledger_yielder_slots_claimed_for_testing();
        expect(r, slots1 == slots0 + 1,
               "the live table holds " + num(slots1) + " yielder slots, want one more than the " + num(slots0) +
                   " before it: its cache is not registered");
        uint64_t   role1 = 0;
        const bool role_read1 =
            ggml_sycl::unified_cache_role_bytes_for_testing(0, ggml_sycl::alloc_role::LAZY_ROW_CACHE, &role1);
        if (!role_read0 || !role_read1) {
            void_arm(r, "the LAZY_ROW_CACHE role reading was VOID with device 0's cache created (readings " +
                            num(role_read0) + "/" + num(role_read1) + ")");
        }
        expect(r, role1 - role0 == k_cache_role_bytes,
               "the bytes requested under role LAZY_ROW_CACHE grew by " + num(role1 - role0) + ", want " +
                   num(k_cache_role_bytes));

        // Another load's charge that does not fit: the cache's yielder frees its last chunk on this thread.
        ggml_sycl_test_host_ledger_store_budget(ggml_sycl::host_ledger_committed());
        const uint64_t                        committed1 = ggml_sycl::host_ledger_committed();
        const uint64_t                        live1      = host_cpu_live();
        ggml_sycl::host_ledger_charge_request ask{};
        ask.bytes       = 1 * MiB;
        ask.may_yield   = true;
        ask.load_txn_id = k_other_load_txn_id;
        log_clear();
        const ggml_sycl::host_ledger_charge_result other = ggml_sycl::host_ledger_try_charge(ask);
        const ggml_sycl::lazy_row_cache_counters   after = cache->counters();
        expect(r, after.chunks_yielded == 1 && after.yields == 1 && after.chunks_resident == 2,
               "the other load's charge left chunks_yielded=" + num(after.chunks_yielded) + " yields=" +
                   num(after.yields) + " chunks_resident=" + num(after.chunks_resident) + ", want 1, 1 and 2");
        expect(r, other.charged, "the other load's charge did not fit after the yield");
        expect(r, live1 - host_cpu_live() == k_cache_chunk_bytes,
               "the yield took " + num(live1 - host_cpu_live()) + " B off the live HOST_CPU rows, want one chunk, " +
                   num(k_cache_chunk_bytes));
        expect(r, ggml_sycl::host_ledger_committed() == committed1 - k_cache_chunk_bytes + ask.bytes,
               "committed is " + num(ggml_sycl::host_ledger_committed()) + ", want " +
                   num(committed1 - k_cache_chunk_bytes + ask.bytes) + ": the chunk's charge was not credited once");
        expect(r,
               ggml_sycl::host_ledger_yieldable_bytes() - yieldable0 == k_cache_yieldable_bytes - k_cache_chunk_bytes,
               "the yieldable bytes did not drop by the freed chunk");
        const std::string yielded = "[LAZY-ROW-CACHE] storage.owner.lazy yielded chunks=1";
        const std::string tenant  = " tenant_load_txn=" + num(k_other_load_txn_id);
        expect(r, log_count_with(yielded.c_str(), tenant.c_str()) == 1,
               "the yield did not log its chunk count and the asking load");
        if (other.charged) {
            ggml_sycl::host_ledger_credit(ask.bytes);
        }

        // The table's own load's charge that does not fit: the yield skips the load's own chunks.
        ggml_sycl_test_host_ledger_store_budget(ggml_sycl::host_ledger_committed());
        ask.load_txn_id                                = k_load_txn_id;
        const ggml_sycl::host_ledger_charge_result own = ggml_sycl::host_ledger_try_charge(ask);
        expect(r, !own.charged, "the own load's charge fit with no room");
        expect(r, cache->counters().chunks_yielded == 1, "the own load's charge freed one of its own chunks");
        if (own.charged) {
            ggml_sycl::host_ledger_credit(ask.bytes);
        }
        ggml_sycl_test_host_ledger_store_budget(k_ledger_budget);
    }
    uint64_t   role2 = 0;
    const bool role_read2 =
        ggml_sycl::unified_cache_role_bytes_for_testing(0, ggml_sycl::alloc_role::LAZY_ROW_CACHE, &role2);
    expect(r, host_cpu_live() == live0,
           "the last reference left " + num(host_cpu_live() - live0) + " HOST_CPU bytes live, want 0");
    expect(
        r, ggml_sycl::host_ledger_committed() == committed0,
        "the last reference left committed " + num(ggml_sycl::host_ledger_committed()) + ", want " + num(committed0));
    expect(r, ggml_sycl::host_ledger_yieldable_bytes() == yieldable0,
           "the last reference left " + num(ggml_sycl::host_ledger_yieldable_bytes() - yieldable0) +
               " yieldable bytes behind");
    expect(r, role_read2 && role2 == role0, "the last reference left LAZY_ROW_CACHE bytes live");
    const int slots2 = ggml_sycl::host_ledger_yielder_slots_claimed_for_testing();
    expect(r, slots2 == slots0,
           "the last reference left " + num(slots2) + " yielder slots claimed, want the " + num(slots0) +
               " before the table: its cache was not unregistered");
    return r;
}

// The shortfall leg's mapping hook: counts the HOST_CPU mappings and fails the g_mmap_fail_at-th.
std::atomic<int> g_mmap_calls{ 0 };
std::atomic<int> g_mmap_fail_at{ 0 };

int32_t mmap_fail_hook(int32_t point, int32_t) {
    if (point != GGML_SYCL_TEST_HOST_LEDGER_POINT_HOST_CPU_MMAP_FAIL) {
        return 0;
    }
    return g_mmap_calls.fetch_add(1) + 1 == g_mmap_fail_at.load() ? 1 : 0;
}

const char * k_shortfall_arm_name =
    "(15) a row-cache shortfall refuses with one [HOST-PLAN-BUG] line naming planned and charged";

// (15) Row T2's arm (n), its [HOST-PLAN-BUG] leg, on the row cache (§6.4): a step-4 shortfall, the owner entry
// refusing one of the cache's planned allocations, refuses the table as lazy-table-row-cache-shortfall with exactly
// one [HOST-PLAN-BUG] line naming the plan's figure and the bytes the cache had charged, and leaves nothing live and
// nothing yieldable. The ledger is WARN-only in phase (a), so the owner entry's charge never refuses; the refusal is
// injected at the HOST_CPU mapping of chunk 1, the third mapping after the index's and chunk 0's, which had charged
// 4 MiB. Engagement: the hook saw three mappings and failed the third, and the owner entry logged that refusal.
// Mutant: --mutant shortfall-unreported (the shortfall refuses with the plain [LAZY-TABLE] WARN and no [HOST-PLAN-BUG]
// line).
arm_result arm_storage_cache_shortfall_refuses() {
    arm_result      r;
    const uint64_t  live0      = host_cpu_live();
    const uint64_t  committed0 = ggml_sycl::host_ledger_committed();
    const uint64_t  yieldable0 = ggml_sycl::host_ledger_yieldable_bytes();
    lazy_table_desc d = fixture_desc("storage.shortfall.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0);
    d.host_resident_bytes = storage_resident_bytes(k_rows_a, k_cache_chunk_bytes);
    const size_t charged  = host_cpu_mapped_size(k_cache_index_bytes) + k_cache_chunk_bytes;
    g_mmap_calls.store(0);
    g_mmap_fail_at.store(3);
    if (!ggml_sycl_test_host_ledger_hook(GGML_SYCL_TEST_HOST_LEDGER_POINT_HOST_CPU_MMAP_FAIL, mmap_fail_hook)) {
        void_arm(r, "the mapping hook did not install");
        return r;
    }
    log_clear();
    lazy_table_result t =
        ggml_sycl::lazy_table_test_create_storage(d, k_cache_chunk_bytes, g_mutant_shortfall_unreported);
    (void) ggml_sycl_test_host_ledger_hook(GGML_SYCL_TEST_HOST_LEDGER_POINT_HOST_CPU_MMAP_FAIL, nullptr);
    if (g_mmap_calls.load() != 3 || log_count_with("[UNIFIED-ALLOC]") != 1) {
        void_arm(r, "the hook saw " + num(g_mmap_calls.load()) + " mappings and the owner entry logged " +
                        num(log_count_with("[UNIFIED-ALLOC]")) + " refusals, want 3 and 1: " + status_name(t) + ": " +
                        t.reason);
        return r;
    }
    expect(r, t.status == lazy_table_status::ROW_CACHE_SHORTFALL && !t.table,
           "the shortfall built as " + status_name(t) + ", want lazy-table-row-cache-shortfall");
    const auto lines = log_lines_containing("[HOST-PLAN-BUG] lazy table storage.shortfall.lazy row cache");
    expect(r, lines.size() == 1, "the shortfall printed " + num(lines.size()) + " [HOST-PLAN-BUG] lines, want 1");
    if (lines.size() == 1) {
        expect(r, lines[0].find("lazy-table-row-cache-shortfall") != std::string::npos,
               "the [HOST-PLAN-BUG] line does not name its status: " + lines[0]);
        expect(r, printed(lines[0], "planned") == k_storage_planned_bytes,
               "the line's planned= is not the plan's " + num(k_storage_planned_bytes) + ": " + lines[0]);
        expect(r, printed(lines[0], "charged") == charged,
               "the line's charged= is not the index's and chunk 0's " + num(charged) + ": " + lines[0]);
    }
    expect(r, log_count_with("[HOST-PLAN-BUG]") == lines.size(), "another [HOST-PLAN-BUG] line was printed");
    expect(r, host_cpu_live() == live0, "the shortfall left " + num(host_cpu_live() - live0) + " HOST_CPU bytes live");
    expect(r, ggml_sycl::host_ledger_committed() == committed0,
           "the shortfall left committed " + num(ggml_sycl::host_ledger_committed()) + ", want " + num(committed0));
    expect(r, ggml_sycl::host_ledger_yieldable_bytes() == yieldable0, "the shortfall left yieldable bytes behind");
    return r;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--guard-child") == 0) {
        g_fixture_path = argv[2];
        return guard_child();
    }
    if (argc >= 3 && std::strcmp(argv[1], "--strict-child") == 0) {
        g_fixture_path = argv[2];
        return strict_child();
    }
    const bool mutant_args        = argc == 3 && std::strcmp(argv[1], "--mutant") == 0;
    const bool mutant_table       = mutant_args && std::strcmp(argv[2], "predicate-too-broad") == 0;
    g_mutant_read_whole_window    = mutant_args && std::strcmp(argv[2], "read-whole-window") == 0;
    g_mutant_shortfall_unreported = mutant_args && std::strcmp(argv[2], "shortfall-unreported") == 0;
    if (argc >= 2 && !mutant_table && !g_mutant_read_whole_window && !g_mutant_shortfall_unreported) {
        std::printf(
            "FAIL: unknown arguments; the options are --mutant predicate-too-broad, --mutant read-whole-window and "
            "--mutant shortfall-unreported\n");
        return 1;
    }
    if (mutant_table) {
        if (!sycl_test_require_gpu("the lazy table buffer-type table")) {
            return SYCL_TEST_SKIP;
        }
        ggml_sycl_test_lazy_predicate_too_broad(true);
        std::vector<named_arm> table_only;
        table_only.push_back({ k_table_arm_name, arm_vram_answer_unchanged_for_other_types });
        return run_arms(table_only, "lazy table arms (--mutant predicate-too-broad)");
    }

    // The refusal arms read the typed refusal, which strict mode turns into an abort; the strict arm sets it alone.
    unsetenv("GGML_SYCL_STRICT_LEASES");

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        std::printf("FAIL: getcwd: %s\n", std::strerror(errno));
        return 1;
    }
    g_fixture_path     = std::string(cwd) + "/test-sycl-lazy-table-" + std::to_string(getpid()) + ".bin";
    g_eof_fixture_path = std::string(cwd) + "/test-sycl-lazy-table-eof-" + std::to_string(getpid()) + ".bin";
    if (!write_fixture(g_fixture_path, k_file_bytes) || !write_fixture(g_eof_fixture_path, k_eof_file_bytes)) {
        std::printf("FAIL: the fixture files %s and %s were not written: %s\n", g_fixture_path.c_str(),
                    g_eof_fixture_path.c_str(), std::strerror(errno));
        ::unlink(g_fixture_path.c_str());
        ::unlink(g_eof_fixture_path.c_str());
        return 1;
    }
    g_direct_unsupported = !probe_direct(g_fixture_path);

    if (g_mutant_read_whole_window) {
        int rc = SYCL_TEST_SKIP;
        if (sycl_test_require_gpu("the lazy table end-of-file sub-check")) {
            ggml_log_set(capture_log, nullptr);
            ggml_sycl_test_host_ledger_store_budget(k_ledger_budget);
            std::vector<named_arm> eof_only;
            eof_only.push_back({ k_eager_arm_name, arm_host_eager_read });
            rc = run_arms(eof_only, "lazy table arms (--mutant read-whole-window)");
        }
        ::unlink(g_fixture_path.c_str());
        ::unlink(g_eof_fixture_path.c_str());
        return rc;
    }
    if (g_mutant_shortfall_unreported) {
        int rc = SYCL_TEST_SKIP;
        if (sycl_test_require_gpu("the lazy table row-cache shortfall leg")) {
            ggml_log_set(capture_log, nullptr);
            ggml_sycl_test_host_ledger_store_budget(k_ledger_budget);
            std::unique_ptr<ggml_sycl::lazy_test_coverage_pin> ring_pin = make_ring_pin();
            std::vector<named_arm>                             shortfall_only;
            shortfall_only.push_back({ k_shortfall_arm_name, arm_storage_cache_shortfall_refuses });
            rc = run_arms(shortfall_only, "lazy table arms (--mutant shortfall-unreported)");
        }
        ::unlink(g_fixture_path.c_str());
        ::unlink(g_eof_fixture_path.c_str());
        return rc;
    }

    // Both children start before this process initialises SYCL.
    g_guard_run  = spawn("--guard-child", nullptr);
    g_strict_run = spawn("--strict-child", "GGML_SYCL_STRICT_LEASES=1");
    std::fputs(g_guard_run.err.c_str(), stderr);
    std::fputs(g_strict_run.err.c_str(), stderr);

    if (!sycl_test_require_gpu("the lazy table arms")) {
        ::unlink(g_fixture_path.c_str());
        ::unlink(g_eof_fixture_path.c_str());
        return SYCL_TEST_SKIP;
    }
    ggml_log_set(capture_log, nullptr);
    ggml_sycl_test_host_ledger_store_budget(k_ledger_budget);
    std::unique_ptr<ggml_sycl::lazy_test_coverage_pin> ring_pin = make_ring_pin();

    std::vector<named_arm> arms;
    arms.push_back({ "(1) the HOST body requests the O_DIRECT window over the table", arm_body_request_window });
    arms.push_back({ k_eager_arm_name, arm_host_eager_read });
    arms.push_back({ "(3) a window past a huge page maps the larger figure", arm_host_window_crosses_huge_page });
    arms.push_back({ "(4) the last reference releases the body, once", arm_last_reference_releases });
    arms.push_back({ "(5) a STORAGE table holds its descriptor and its row cache until its last reference",
                     arm_storage_holds_its_descriptor });
    arms.push_back({ "(6) a short source refuses naming both numbers", arm_source_short_refuses });
    arms.push_back({ "(7) a descriptor that is not a table refuses before any open", arm_invalid_desc_refuses });
    arms.push_back(
        { "(8) one extra chunk fails the activation check naming both numbers", arm_activation_extra_chunk_refuses });
    arms.push_back({ "(9) under strict leases the activation mismatch aborts", arm_activation_strict_aborts });
    arms.push_back(
        { "(10) a WEIGHTS tensor bound to either descriptor keeps extra null", arm_bound_tensor_keeps_extra_null });
    arms.push_back({ "(11) the descriptor guard prints one line and exits 134", arm_descriptor_guard });
    arms.push_back({ "(12) is_device_vram_buffer answers false for a descriptor before its cast",
                     arm_vram_answer_for_descriptors });
    arms.push_back({ "(14) a STORAGE table's row cache charges, yields and releases through the owner entry",
                     arm_storage_cache_owner_path });
    arms.push_back({ k_shortfall_arm_name, arm_storage_cache_shortfall_refuses });
    // Last: its tensor-parallel factory latches process-wide state.
    arms.push_back({ k_table_arm_name, arm_vram_answer_unchanged_for_other_types });
    const int rc = run_arms(arms, "lazy table arms");
    ring_pin.reset();
    ::unlink(g_fixture_path.c_str());
    ::unlink(g_eof_fixture_path.c_str());
    return rc;
}
