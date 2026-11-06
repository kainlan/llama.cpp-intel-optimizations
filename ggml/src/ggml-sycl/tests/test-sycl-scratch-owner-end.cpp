//
// The owner's end of a withdrawn SCRATCH chunk (docs/design/sycl-mmap-host-tier.md §7 C4 test-sycl-scratch-owner-end,
// §4.4 "Every withdrawn chunk has an owner and a release"; r9 m-5, r10 m-5, r11 m-5, r15 M-1, r17 M-4).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN: it creates real SYCL backends and grows pinned SCRATCH chunks on a real queue, so it is registered
// `sycl;cache;mem-handle` plus gpu-serial, serial, with the selector pinned to one discrete card. It loads no model.
//
// Each case starts as sched_reserve's own transaction would when its reserve throws: a context backend MARKs, its
// notify grows one chunk, a stand-in for its host compute buffer lands in that chunk through the host buffer type the
// scheduler allocates from, and the transaction RETURNs with no begin after it. The context is then freed with
// ggml_backend_free, which reaches ggml_backend_sycl_free and its owner-end step; no case calls the owner's end itself.
// The transaction is driven through the ggml_backend_sycl_host_scratch_trial proc-table entry, as llama_context does.
//
// Cases (each in its own child process; see scratch-return-test-common.hpp):
//   orphaned       the context is freed with the stand-in live: nothing is released and the chunk is orphaned; the
//                  stand-in's free releases it.
//   empty-at-free  the stand-in is freed first; the context's free releases the chunk.
//   two-empty      the transaction grew two chunks, both emptied before the context's free, which releases both.
//   lease-a        a lease on the stand-in's chunk, the stand-in freed before the context: the owner's end orphans the
//                  empty, leased chunk and the lease's drop releases it.
//   lease-b        a lease and a live stand-in at the context's free: orphaned (live is checked first); the stand-in's
//                  free finds the lease and releases nothing; the lease's drop releases it.
// Every release is followed by the checks that the marked set is empty and that a following begin prints nothing.
//
// Mutants (scratch-test-mutant.hpp), each a ctest of its own with its expected line: owner-end-removed,
// orphan-free-ignored, owner-end-no-empty, release-no-erase, owner-end-first-only, lease-drop-no-index.

#include "scratch-return-test-common.hpp"

#include <chrono>
#include <string>
#include <vector>

using namespace scratch_return_test;

namespace {

// The stand-in's size: it fits a freshly grown 64 MiB chunk and no base block left by scratch_fill.
constexpr size_t k_stand_in = 48 * MiB;

using notify_fn = decltype(&ggml_backend_sycl_notify_host_buffer_sizes);

struct entry_points {
    trial_fn  trial  = nullptr;
    notify_fn notify = nullptr;
};

// By proc name, as llama_context resolves them: the private fixtures embed the backend, so the registry is theirs.
bool entry_points_resolve(checks & c, entry_points & e) {
    e.trial = trial_entry_resolve(c);
    if (e.trial == nullptr) {
        return false;
    }
    e.notify = reinterpret_cast<notify_fn>(
        ggml_backend_reg_get_proc_address(ggml_backend_sycl_reg(), "ggml_backend_sycl_notify_host_buffer_sizes"));
    if (e.notify == nullptr) {
        c.void_case("the proc table has no ggml_backend_sycl_notify_host_buffer_sizes entry");
        return false;
    }
    return true;
}

// The state every case starts from and returns to.
struct baseline {
    size_t chunks    = 0;
    size_t allocated = 0;
};

baseline baseline_take(const device_fixture & f) {
    return { f.pool->chunk_count(), f.pool->allocated() };
}

// One chunk grown by a context's MARKed notify, and a stand-in for its host compute buffer in that chunk. Returns the
// chunk's index, or SIZE_MAX after a VOID.
size_t grow_with_stand_in(checks &                c,
                          const device_fixture &  f,
                          const entry_points &    e,
                          ggml_backend_t          ctx,
                          size_t                  notify_bytes,
                          size_t                  chunks_before,
                          ggml_backend_buffer_t * stand_in) {
    if (!e.trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 512)) {
        c.void_case("the MARK entry refused the context backend");
        return SIZE_MAX;
    }
    if (!e.notify(ctx, notify_bytes, 0) || f.pool->chunk_count() != chunks_before + 1) {
        c.void_case("the notify of " + mib(notify_bytes) + " MiB grew " +
                    std::to_string(f.pool->chunk_count() - chunks_before) + " chunk(s), not 1");
        return SIZE_MAX;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(ctx));
    *stand_in                       = buft ? ggml_backend_buft_alloc_buffer(buft, k_stand_in) : nullptr;
    if (*stand_in == nullptr) {
        c.void_case("the host buffer type refused the " + mib(k_stand_in) + " MiB stand-in");
        return SIZE_MAX;
    }
    const size_t idx = chunk_of(f.pool, ggml_backend_buffer_get_base(*stand_in));
    if (idx == SIZE_MAX || f.pool->chunk_count() != chunks_before + 1 ||
        !f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx).found) {
        c.void_case("the stand-in did not land in the chunk the notify grew");
        return SIZE_MAX;
    }
    return idx;
}

// After a release: back to the baseline, one more `released` line, an empty marked set, and a begin that moves and
// prints nothing.
void expect_released(checks &               c,
                     const device_fixture & f,
                     const baseline &       b,
                     size_t                 idx,
                     size_t                 released_before,
                     const std::string &    when,
                     const std::string &    cause) {
    const size_t chunks = f.pool->chunk_count();
    c.expect(chunks == b.chunks, "chunk_count " + std::to_string(chunks) + " " + when + ", expected " +
                                     std::to_string(b.chunks) + (cause.empty() ? "" : ": " + cause));
    c.expect(f.pool->allocated() == b.allocated,
             "allocated() " + mib(f.pool->allocated()) + " MiB " + when + ", expected " + mib(b.allocated) + " MiB");
    c.expect(lines(k_released_line) > released_before, "no SCRATCH released line for chunk " + std::to_string(idx));
    expect_marked_set_quiet(c, f, idx, 0);
}

int case_orphaned(checks & c, const device_fixture & f, const entry_points & e) {
    ggml_backend_t ctx = ggml_backend_sycl_init(0);
    if (ctx == nullptr) {
        c.void_case("the context backend did not initialise");
        return c.status();
    }
    const baseline        b        = baseline_take(f);
    ggml_backend_buffer_t stand_in = nullptr;
    const size_t          idx      = grow_with_stand_in(c, f, e, ctx, k_stand_in, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    (void) e.trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    expect_returned(c, f, idx);

    ggml_backend_free(ctx);
    c.expect(f.pool->chunk_count() == b.chunks + 1, "chunk_count " + std::to_string(f.pool->chunk_count()) +
                                                        " after the context is freed with the stand-in live, "
                                                        "expected " +
                                                        std::to_string(b.chunks + 1) + ": nothing is released");
    c.expect(f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx).orphaned,
             "chunk " + std::to_string(idx) + " not orphaned by the owner's end");

    const size_t released_before = lines(k_released_line);
    ggml_backend_buffer_free(stand_in);
    expect_released(c, f, b, idx, released_before, "after the stand-in's free", "");
    return c.status();
}

int case_empty_at_free(checks & c, const device_fixture & f, const entry_points & e) {
    ggml_backend_t ctx = ggml_backend_sycl_init(0);
    if (ctx == nullptr) {
        c.void_case("the context backend did not initialise");
        return c.status();
    }
    const baseline        b        = baseline_take(f);
    ggml_backend_buffer_t stand_in = nullptr;
    const size_t          idx      = grow_with_stand_in(c, f, e, ctx, k_stand_in, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    (void) e.trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    expect_returned(c, f, idx);

    ggml_backend_buffer_free(stand_in);
    c.expect(f.pool->chunk_count() == b.chunks + 1,
             "chunk_count " + std::to_string(f.pool->chunk_count()) +
                 " after the stand-in's free with its owner live, expected " + std::to_string(b.chunks + 1) +
                 ": only the owner's begin or end releases a chunk that is not orphaned");

    const size_t released_before = lines(k_released_line);
    ggml_backend_free(ctx);
    expect_released(c, f, b, idx, released_before, "after the context is freed",
                    "withdrawn chunk " + std::to_string(idx) + " idle until pool destruction");
    return c.status();
}

int case_two_empty(checks & c, const device_fixture & f, const entry_points & e) {
    ggml_backend_t ctx = ggml_backend_sycl_init(0);
    if (ctx == nullptr) {
        c.void_case("the context backend did not initialise");
        return c.status();
    }
    const baseline        b       = baseline_take(f);
    ggml_backend_buffer_t stand_a = nullptr;
    const size_t          idx_a   = grow_with_stand_in(c, f, e, ctx, k_stand_in, b.chunks, &stand_a);
    if (idx_a == SIZE_MAX) {
        return c.status();
    }
    // The context's second notify replaces its demand: 100 MiB on the 64 MiB plan term needs 164 MiB against the
    // 128 MiB the zone now holds, so one more 64 MiB chunk grows. Its stand-in cannot fit chunk A's 16 MiB remainder.
    if (!e.notify(ctx, 100 * MiB, 0) || f.pool->chunk_count() != b.chunks + 2) {
        c.void_case("the second notify did not grow a second chunk");
        return c.status();
    }
    ggml_backend_buffer_type_t buft    = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(ctx));
    ggml_backend_buffer_t      stand_b = ggml_backend_buft_alloc_buffer(buft, k_stand_in);
    const size_t               idx_b   = stand_b ? chunk_of(f.pool, ggml_backend_buffer_get_base(stand_b)) : SIZE_MAX;
    if (idx_b == SIZE_MAX || idx_b == idx_a || f.pool->chunk_count() != b.chunks + 2) {
        c.void_case("the second stand-in did not land in the second grown chunk");
        return c.status();
    }
    (void) e.trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    expect_returned(c, f, idx_a);
    expect_returned(c, f, idx_b);
    ggml_backend_buffer_free(stand_a);
    ggml_backend_buffer_free(stand_b);

    const size_t released_before = lines(k_released_line);
    ggml_backend_free(ctx);
    const size_t chunks = f.pool->chunk_count();
    const size_t freed  = b.chunks + 2 >= chunks ? b.chunks + 2 - chunks : 0;
    c.expect(chunks == b.chunks, "chunk_count " + std::to_string(chunks) + " after the context is freed, expected " +
                                     std::to_string(b.chunks) + ": owner-end freed " +
                                     (freed == 1 ? "one" :
                                      freed == 0 ? "none" :
                                                   "both") +
                                     " of two chunks");
    c.expect(f.pool->allocated() == b.allocated, "allocated() " + mib(f.pool->allocated()) +
                                                     " MiB after the context is freed, expected " + mib(b.allocated) +
                                                     " MiB");
    c.expect(lines(k_released_line, "(2 chunk(s))") > 0 || lines(k_released_line) >= released_before + 2,
             "no SCRATCH released line for chunks " + std::to_string(idx_a) + " and " + std::to_string(idx_b));
    c.expect(f.cache->marked_entries() == 0, "marked set holds " + std::to_string(f.cache->marked_entries()) +
                                                 " entry(ies) after the release of chunks " + std::to_string(idx_a) +
                                                 " and " + std::to_string(idx_b) + ", expected 0");
    return c.status();
}

// Lease order (a): the stand-in is freed before the context, so the chunk is empty with a lease held at the owner's
// end. Order (b): the stand-in is live at the owner's end, beside the lease.
int case_lease(checks & c, const device_fixture & f, const entry_points & e, bool stand_in_freed_first) {
    ggml_backend_t ctx = ggml_backend_sycl_init(0);
    if (ctx == nullptr) {
        c.void_case("the context backend did not initialise");
        return c.status();
    }
    const baseline        b        = baseline_take(f);
    ggml_backend_buffer_t stand_in = nullptr;
    const size_t          idx      = grow_with_stand_in(c, f, e, ctx, k_stand_in, b.chunks, &stand_in);
    if (idx == SIZE_MAX) {
        return c.status();
    }
    const uint64_t lease = f.cache->host_acquire_chunk_lease(ggml_backend_buffer_get_base(stand_in));
    if (lease == pinned_chunk_pool::INVALID_CHUNK_HANDLE) {
        c.void_case("no lease could be taken on chunk " + std::to_string(idx));
        return c.status();
    }
    (void) e.trial(ctx, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    expect_returned(c, f, idx);

    if (stand_in_freed_first) {
        ggml_backend_buffer_free(stand_in);
        stand_in = nullptr;
    }
    ggml_backend_free(ctx);
    c.expect(f.pool->chunk_count() == b.chunks + 1, "chunk_count " + std::to_string(f.pool->chunk_count()) +
                                                        " after the context is freed under a lease, expected " +
                                                        std::to_string(b.chunks + 1) + ": nothing is released");
    c.expect(f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx).orphaned,
             "chunk " + std::to_string(idx) + " not orphaned by the owner's end under a lease");
    if (stand_in != nullptr) {
        ggml_backend_buffer_free(stand_in);
        c.expect(f.pool->chunk_count() == b.chunks + 1,
                 "chunk_count " + std::to_string(f.pool->chunk_count()) + " after the stand-in's free, expected " +
                     std::to_string(b.chunks + 1) + ": the free released chunk " + std::to_string(idx) +
                     " under a held lease");
    }

    const size_t released_before = lines(k_released_line);
    f.cache->host_release_chunk_lease(lease);
    expect_released(c, f, b, idx, released_before, "after the lease dropped",
                    "withdrawn chunk " + std::to_string(idx) + " waits for a begin");
    return c.status();
}

int run_case(const child_args & a) {
    checks c{ a.case_name.c_str() };
    if (!child_begin(a)) {
        return 2;
    }
    device_fixture f;
    entry_points   e;
    if (!entry_environment_ok(c) || !device_fixture_open(c, f) || !entry_points_resolve(c, e)) {
        return c.status();
    }
    // Every existing SCRATCH block is filled, so each stand-in lands in the chunk its notify grew.
    std::vector<void *> fillers = scratch_fill(f.cache);
    int                 rc      = 2;
    if (a.case_name == "orphaned") {
        rc = case_orphaned(c, f, e);
    } else if (a.case_name == "empty-at-free") {
        rc = case_empty_at_free(c, f, e);
    } else if (a.case_name == "two-empty") {
        rc = case_two_empty(c, f, e);
    } else if (a.case_name == "lease-a") {
        rc = case_lease(c, f, e, true);
    } else if (a.case_name == "lease-b") {
        rc = case_lease(c, f, e, false);
    } else {
        fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    }
    for (void * p : fillers) {
        f.cache->host_zone_free(host_zone_id::SCRATCH, p);
    }
    ggml_backend_free(f.backend);
    return rc;
}

}  // namespace

int main(int argc, char ** argv) {
    return cases_main(argc, argv, { "orphaned", "empty-at-free", "two-empty", "lease-a", "lease-b" }, run_case,
                      std::chrono::milliseconds(120000));
}
