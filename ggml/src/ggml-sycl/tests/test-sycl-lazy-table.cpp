//
// The lazy table object (th32 §3.2, §3.3, §10; row T4): the refcounted lazy_table and its two constructors, the HOST
// body's eager O_DIRECT read into the locked HOST_CPU tier, the STORAGE descriptor, the activation check, the
// descriptor buffer types with their own interface, the guard at ggml_sycl_get_data_ptr_slow's entry, and
// ggml_sycl_is_device_vram_buffer's descriptor test before its cast. No libllama is involved: every table is built
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
// the SYCL device type, and the table must FAIL on that row: the registered positive control of an arm whose RED
// cannot fail (with no descriptor test, no other type's answer can change).
//
// --mutant read-whole-window runs arm (2) alone with its end-of-file table built by a read that asks for the whole
// O_DIRECT window, and the arm must FAIL naming that geometry: the registered positive control of a sub-check the
// stub cannot fail for the right reason (a read that stops at the table's last byte and one that does not differ only
// when the window runs past the end of the file).

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
#include <string>
#include <vector>

extern char ** environ;

using namespace host_ledger_test_harness;
using namespace lazy_table_test;
using ggml_sycl::host_cpu_mapped_size;
using ggml_sycl::lazy_residency;
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

// A ledger budget far above the fixture's bytes, seeded in every process, so no charge logs a would-refuse WARN and
// strict mode meets only the check under test.
constexpr uint64_t k_ledger_budget = 64 * GiB;

std::string g_fixture_path;
std::string g_eof_fixture_path;
bool        g_direct_unsupported       = false;  // the fixture's filesystem refuses O_DIRECT: the reading arms are VOID
bool        g_mutant_read_whole_window = false;  // --mutant read-whole-window

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
                                 0;
    return d;
}

uint64_t host_cpu_live() {
    uint64_t live = 0;
    return ggml_sycl_test_host_cpu_live_bytes(&live) ? live : UINT64_MAX;
}

// A refusal the environment caused, not the table: no O_DIRECT, or a lock limit below the body.
bool environment_refusal(const lazy_table_result & r, size_t locked_bytes, std::string & why) {
    if (r.status == lazy_table_status::OPEN_FAILED && g_direct_unsupported) {
        why = "the fixture's filesystem refuses O_DIRECT: " + r.reason;
        return true;
    }
    if (r.status == lazy_table_status::BODY_REFUSED) {
        rlimit lim{};
        if (getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY && lim.rlim_cur < locked_bytes) {
            why = "RLIMIT_MEMLOCK is " + num(lim.rlim_cur) + " B, below the body: " + r.reason;
            return true;
        }
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

// The children's environment: this one's, with GGML_NO_BACKTRACE so an abort prints its message and no gdb attach,
// and `extra` when given.
std::vector<std::string> child_environment(const char * extra) {
    std::vector<std::string> env;
    for (char ** e = environ; e != nullptr && *e != nullptr; ++e) {
        env.emplace_back(*e);
    }
    env.emplace_back("GGML_NO_BACKTRACE=1");
    if (extra != nullptr) {
        env.emplace_back(extra);
    }
    return env;
}

child_run spawn(const char * mode, const char * extra_env) {
    std::vector<std::string> args = { "test-sycl-lazy-table", mode, g_fixture_path };
    std::vector<std::string> env  = child_environment(extra_env);
    std::vector<char *>      argv;
    std::vector<char *>      envp;
    for (std::string & a : args) {
        argv.push_back(a.data());
    }
    argv.push_back(nullptr);
    for (std::string & e : env) {
        envp.push_back(e.data());
    }
    envp.push_back(nullptr);
    return run_child(argv.data(), envp.data(), std::chrono::milliseconds(120000));
}

size_t count_lines_with(const std::string & text, const char * a, const char * b = nullptr) {
    size_t count = 0;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string line = text.substr(start, end - start);
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            count++;
        }
        start = end + 1;
    }
    return count;
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
           k_eof_offset + k_table_a_bytes == k_eof_file_bytes && k_eof_file_bytes % 4096 != 0 &&
               lazy_table_host_body_request_bytes(k_eof_offset, k_table_a_bytes) == k_eof_window_bytes &&
               k_eof_window_bytes - k_eof_file_bytes == k_eof_past_end,
           "the end-of-file fixture is not the geometry the hand figures assume");
    lazy_table_desc d   = fixture_desc("host.eof.lazy", k_eof_offset, k_rows_a, lazy_residency::HOST, 0);
    d.file_path         = g_eof_fixture_path;
    lazy_table_result t = g_mutant_read_whole_window ? ggml_sycl::lazy_table_test_create_host_reading_window(d) :
                                                       lazy_table_create_host(d);
    std::string       why;
    if (environment_refusal(t, k_mapped_a_bytes, why)) {
        void_arm(r, geometry + ": " + why);
        return;
    }
    if (!t) {
        fail(r, geometry + " refused: " + ggml_sycl::lazy_table_status_name(t.status) + ": " + t.reason);
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
    std::string       why;
    if (environment_refusal(t, k_mapped_a_bytes, why)) {
        void_arm(r, why);
        return r;
    }
    if (!t) {
        fail(r, std::string("the table refused: ") + ggml_sycl::lazy_table_status_name(t.status) + ": " + t.reason);
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
    expect(r, reinterpret_cast<uintptr_t>(view.ptr) % ggml_sycl::LAZY_IO_BLOCK_BYTES == k_table_offset % 4096,
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
    std::string why;
    if (environment_refusal(t, k_mapped_b_bytes, why)) {
        void_arm(r, why);
        return r;
    }
    if (!t) {
        fail(r, std::string("the table refused: ") + ggml_sycl::lazy_table_status_name(t.status) + ": " + t.reason);
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
        std::string why;
        if (environment_refusal(t, k_mapped_a_bytes, why)) {
            void_arm(r, why);
            return r;
        }
        if (!t) {
            fail(r, std::string("the table refused: ") + ggml_sycl::lazy_table_status_name(t.status) + ": " + t.reason);
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
        expect(r, host_cpu_live() - before == k_mapped_a_bytes, "a dropped copy released the body");
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

// (5) A STORAGE table holds an O_DIRECT descriptor on its source and nothing else in T4, and its last reference closes
// it; a STORAGE plan figure, which is its row cache's, is refused by name until the cache exists. Engagement: the
// environment takes O_DIRECT; a refusal it caused is VOID, any other refusal FAILs.
arm_result arm_storage_holds_its_descriptor() {
    arm_result        r;
    const uint64_t    before = host_cpu_live();
    lazy_table_result t =
        lazy_table_create_storage(fixture_desc("storage.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0));
    std::string why;
    if (environment_refusal(t, 0, why)) {
        void_arm(r, why);
        return r;
    }
    if (!t) {
        fail(r, std::string("the table refused: ") + ggml_sycl::lazy_table_status_name(t.status) + ": " + t.reason);
        return r;
    }
    const int fd = t.table->storage_fd();
    expect(r, fd >= 0, "a STORAGE table holds no descriptor");
    char * real = realpath(g_fixture_path.c_str(), nullptr);
    expect(r, real != nullptr && fd_target(fd) == real, "the descriptor is not the source: " + fd_target(fd));
    std::free(real);
    expect(r, (fd_flags(fd) & O_DIRECT) != 0, "the descriptor is not O_DIRECT");
    expect(r, !t.table->rows().valid(), "a STORAGE table holds rows");
    expect(r, t.table->charged_bytes() == 0 && host_cpu_live() == before, "a STORAGE table charged HOST_CPU bytes");
    t.table.reset();
    errno = 0;
    expect(r, ::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "the last reference left the descriptor open");

    lazy_table_desc cached = fixture_desc("storage.cached.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0);
    cached.host_resident_bytes = 4 * MiB;
    const lazy_table_result c  = lazy_table_create_storage(cached);
    expect(r, c.status == lazy_table_status::ROW_CACHE_UNBUILT && !c.table,
           std::string("a STORAGE plan holding 4 MiB of row cache built as ") +
               ggml_sycl::lazy_table_status_name(c.status) + ", want lazy-table-row-cache-unbuilt");
    expect(r, c.reason.find("host_resident_bytes=" + num(4 * MiB)) != std::string::npos,
           "the row-cache refusal does not name the plan's figure: " + c.reason);
    return r;
}

// (6) A source that ends before offset + size refuses, naming both numbers, and leaves nothing live. Engagement: the
// file opened, since the refusal names the file's size.
arm_result arm_source_short_refuses() {
    arm_result     r;
    const uint64_t before = host_cpu_live();
    const uint64_t offset = k_file_bytes - 4096;  // the table runs past the end of the file
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
// that are not row_size * n_rows, the other residency, a HOST window no size_t holds, and a consumer that is neither -1
// nor a SYCL device. Engagement: none needed; the last two name a file that does not exist, so a refusal after the
// open would read lazy-table-open-failed.
arm_result arm_invalid_desc_refuses() {
    arm_result      r;
    lazy_table_desc d = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    d.row_size += 1;
    expect(r, lazy_table_create_host(d).status == lazy_table_status::INVALID_DESC, "a wrong row_size was accepted");
    d           = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, 0);
    d.file_path = "/nonexistent/path/that/is/never/opened";
    d.table_bytes += 1;
    expect(r, lazy_table_create_host(d).status == lazy_table_status::INVALID_DESC,
           "a table_bytes that is not row_size * n_rows reached the open");
    d = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::STORAGE, 0);
    expect(r, lazy_table_create_host(d).status == lazy_table_status::INVALID_DESC,
           "the HOST constructor built a table the plan made STORAGE");

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
           std::string("a window no size_t holds gave ") + ggml_sycl::lazy_table_status_name(window.status) +
               ", want lazy-table-invalid-desc");

    d                             = fixture_desc("invalid.lazy", k_table_offset, k_rows_a, lazy_residency::HOST, -7);
    d.file_path                   = "/nonexistent/path/that/is/never/opened";
    const lazy_table_result other = lazy_table_create_host(d);
    expect(r, other.status == lazy_table_status::INVALID_DESC,
           std::string("consumer_device=-7 gave ") + ggml_sycl::lazy_table_status_name(other.status) +
               ", want lazy-table-invalid-desc");
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
        std::string why;
        if (environment_refusal(control, k_mapped_a_bytes, why)) {
            void_arm(r, why);
            return r;
        }
        if (!control) {
            void_arm(r, "the control table without the extra chunk refused: " + control.reason);
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
           count_lines_with(g_strict_run.err, "[HOST-PLAN-BUG] lazy table strict.lazy activation check",
                            ("charged_bytes=" + num(charged)).c_str()) == 1,
           "the strict child printed no [HOST-PLAN-BUG] line naming charged_bytes=" + num(charged));
    expect(r, count_lines_with(g_strict_run.err, "STRICT-CHILD returned") == 0, "the strict child returned");
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
    std::string why;
    if (environment_refusal(t, k_mapped_a_bytes, why)) {
        void_arm(r, why);
        return r;
    }
    if (!t) {
        void_arm(r, "the table did not build: " + t.reason);
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
        expect(r, t.table->ref_count() == 2, std::string("the ") + f.form + " descriptor holds no reference");
        expect(r, ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
               std::string("the ") + f.form + " descriptor's usage is not WEIGHTS");
        expect(r, ggml_backend_buffer_get_size(buffer) == k_table_a_bytes,
               std::string("the ") + f.form + " descriptor's size is not the table's");
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
        expect(r, t.table->ref_count() == 1, std::string("the ") + f.form + " descriptor's free kept its reference");
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
    if (count_lines_with(g_guard_run.err, "GUARD-CHILD bound extra=null") != 1) {
        void_arm(r, "the guard child did not bind its tensor with extra null before the call");
        return r;
    }
    const size_t lines = count_lines_with(g_guard_run.err, k_guard_phrase);
    expect(r, lines == 1, "the child printed " + num(lines) + " guard lines, want exactly 1");
    expect(r, count_lines_with(g_guard_run.err, k_guard_phrase, "guard.lazy") == lines,
           "the guard line does not name the table");
    expect(r, WIFSIGNALED(g_guard_run.status) && WTERMSIG(g_guard_run.status) == SIGABRT,
           "the child did not exit 134 (status " + std::to_string(g_guard_run.status) + ")");
    expect(r, count_lines_with(g_guard_run.err, "GUARD-CHILD returned") == 0,
           "ggml_sycl_get_data_ptr_slow returned a pointer for a lazy table");
    return r;
}

// A context whose bytes the fixture controls: zero, so today's read of it is DEVICE_VRAM (alloc_tier 0) for every
// buffer type, and a descriptor test that fired on a type would flip that type's answer to false. Larger than any
// SYCL buffer context, so the read stays inside it.
alignas(64) unsigned char g_zero_context[4096] = {};

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
    std::string why;
    if (environment_refusal(t, k_mapped_a_bytes, why)) {
        void_arm(r, why);
        return r;
    }
    if (!t) {
        void_arm(r, "the table did not build: " + t.reason);
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

        ggml_backend_buffer zeroed{};
        zeroed.buft    = row.buft;
        zeroed.context = g_zero_context;
        zeroed.size    = sizeof(g_zero_context);
        ggml_tensor fake{};
        fake.buffer = &zeroed;
        if (!ggml_sycl_test_device_vram_context_read(&fake)) {
            void_arm(r, "today's read of a zeroed context is not DEVICE_VRAM, so the zeroed row cannot fail");
            continue;
        }
        expect(r, !ggml_sycl_test_is_device_vram_buffer(&fake),
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
// only thing that can make the answer false. Its RED cannot fail, since without the descriptor test no answer can
// change; --mutant predicate-too-broad is its positive control. Engagement: every factory returned a type and
// today's read of every row is true.
arm_result arm_vram_answer_unchanged_for_other_types() {
    arm_result r;
    for (const buft_row & row : exposed_bufts()) {
        if (row.buft == nullptr) {
            void_arm(r, row.name + " returned no buffer type");
            continue;
        }
        ggml_backend_buffer zeroed{};
        zeroed.buft    = row.buft;
        zeroed.context = g_zero_context;
        zeroed.size    = sizeof(g_zero_context);
        ggml_tensor fake{};
        fake.buffer          = &zeroed;
        const bool today     = ggml_sycl_test_device_vram_context_read(&fake);
        const bool answer    = ggml_sycl_test_is_device_vram_buffer(&fake);
        const bool predicate = ggml_backend_sycl_buffer_is_lazy(&zeroed);
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

int main(int argc, char ** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--guard-child") == 0) {
        g_fixture_path = argv[2];
        return guard_child();
    }
    if (argc >= 3 && std::strcmp(argv[1], "--strict-child") == 0) {
        g_fixture_path = argv[2];
        return strict_child();
    }
    const bool mutant_args     = argc == 3 && std::strcmp(argv[1], "--mutant") == 0;
    const bool mutant_table    = mutant_args && std::strcmp(argv[2], "predicate-too-broad") == 0;
    g_mutant_read_whole_window = mutant_args && std::strcmp(argv[2], "read-whole-window") == 0;
    if (argc >= 2 && !mutant_table && !g_mutant_read_whole_window) {
        std::printf(
            "FAIL: unknown arguments; the options are --mutant predicate-too-broad and --mutant "
            "read-whole-window\n");
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

    std::vector<named_arm> arms;
    arms.push_back({ "(1) the HOST body requests the O_DIRECT window over the table", arm_body_request_window });
    arms.push_back({ k_eager_arm_name, arm_host_eager_read });
    arms.push_back({ "(3) a window past a huge page maps the larger figure", arm_host_window_crosses_huge_page });
    arms.push_back({ "(4) the last reference releases the body, once", arm_last_reference_releases });
    arms.push_back({ "(5) a STORAGE table holds its O_DIRECT descriptor until its last reference",
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
    // Last: its tensor-parallel factory latches process-wide state.
    arms.push_back({ k_table_arm_name, arm_vram_answer_unchanged_for_other_types });
    const int rc = run_arms(arms, "lazy table arms");
    ::unlink(g_fixture_path.c_str());
    ::unlink(g_eof_fixture_path.c_str());
    return rc;
}
